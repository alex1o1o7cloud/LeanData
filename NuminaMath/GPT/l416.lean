import Mathlib
import Mathlib.Algebra.ArithSeq
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Quotient
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Bitwise
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.String.Defs
import Mathlib.Data.Time
import Mathlib.Data.Zmod.Basic
import Mathlib.NumberTheory.ArithmeticFunction.Euler
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace divisor_of_product_of_four_consecutive_integers_l416_416721

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416721


namespace most_numerous_fruit_l416_416113

-- Define the number of boxes
def num_boxes_tangerines := 5
def num_boxes_apples := 3
def num_boxes_pears := 4

-- Define the number of fruits per box
def tangerines_per_box := 30
def apples_per_box := 20
def pears_per_box := 15

-- Calculate the total number of each fruit
def total_tangerines := num_boxes_tangerines * tangerines_per_box
def total_apples := num_boxes_apples * apples_per_box
def total_pears := num_boxes_pears * pears_per_box

-- State the theorem and prove it
theorem most_numerous_fruit :
  total_tangerines = 150 ∧ total_tangerines > total_apples ∧ total_tangerines > total_pears :=
by
  -- Add here the necessary calculations to verify the conditions
  sorry

end most_numerous_fruit_l416_416113


namespace last_integer_in_sequence_div_by_4_l416_416106

theorem last_integer_in_sequence_div_by_4 (S : ℕ → ℚ) :
  (S 0 = 1024000) →
  (∀ n, S (n + 1) = S n / 4) →
  ∃ N, ∃ K, (S N = K) ∧ (64 < K ≤ 250) ∧ (∀ n > N, (S n) ∉ ℤ) :=
begin
  sorry
end

end last_integer_in_sequence_div_by_4_l416_416106


namespace number_of_functions_l416_416538

-- Definitions and conditions
def A : Set ℕ := {1, 2, ..., 2012}
def B : Set ℕ := {1, 2, ..., 19}
def S : Set (Set ℕ) := set.powerset A

-- Definition of the mapping f and the condition it must fulfill
variable (f : Set ℕ → ℕ)
axiom f_condition : ∀ A1 A2 ∈ S, f (A1 ∩ A2) = min (f A1) (f A2)

-- Theorem stating the number of such functions f
theorem number_of_functions : 
  (∑ i in finset.range (19 + 1), i ^ 2012) = 1 ^ 2012 + 2 ^ 2012 + ... + 19 ^ 2012 :=
sorry

end number_of_functions_l416_416538


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416875

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416875


namespace inverse_of_half_l416_416288

theorem inverse_of_half :
  (1 / 2)⁻¹ = 2 := 
by 
  sorry

end inverse_of_half_l416_416288


namespace find_n_l416_416314

theorem find_n (n : ℕ) : 2^8 * 3^4 * 5^1 * n = nat.factorial 10 → n = 35 := by
  sorry

end find_n_l416_416314


namespace mean_of_set_median_is_128_l416_416097

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l416_416097


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416879

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416879


namespace Victor_last_week_sale_l416_416186

/-- Victor's chocolate sales for each week -/
variables (w1 w2 w3 w4: ℕ) (mean: ℕ)

/-- Conditions -/
noncomputable def w1 := 75
noncomputable def w2 := 67
noncomputable def w3 := 75
noncomputable def w4 := 70
noncomputable def mean := 71

/-- Compute the chocolates sold in the last week -/
theorem Victor_last_week_sale (w5: ℕ) (H: (w1 + w2 + w3 + w4 + w5) / 5 = mean) : w5 = 68 :=
by 
  sorry

end Victor_last_week_sale_l416_416186


namespace solution_l416_416613

noncomputable def problem := 
let radius : ℝ := 10
let AB : ℝ := 9
let BC : ℝ := 10
let CA : ℝ := 11
let x : ℕ := 3
let y : ℕ := 6911
let z : ℕ := 20
in 
radius = 10 ∧ AB = 9 ∧ BC = 10 ∧ CA = 11 → (x + y + z = 6934)

open_locale real
open_locale big_operators

theorem solution : problem :=
by
  -- Proof here is omitted: sorry is used to indicate this step
  sorry

end solution_l416_416613


namespace four_consecutive_product_divisible_by_12_l416_416987

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416987


namespace sum_all_3digit_numbers_with_remainder_2_when_divided_by_6_l416_416219

theorem sum_all_3digit_numbers_with_remainder_2_when_divided_by_6 :
  let seq := (List.range' 17 150).map (λ k, 6 * k + 2)
  seq.sum = 82500 :=
by
  sorry

end sum_all_3digit_numbers_with_remainder_2_when_divided_by_6_l416_416219


namespace prod_inequality_l416_416563

open_locale big_operators

noncomputable def odd_even_prod (n : ℕ) : ℝ :=
  (∏ k in range n, (2*k+1 : ℝ)) / (∏ k in range n, (2*(k+1) : ℝ))

theorem prod_inequality (n : ℕ) (h : n > 1) : 
  (sqrt 2 / 2) * (1 / sqrt (2 * n)) < odd_even_prod n < (sqrt 3 / 2) * (1 / sqrt (2 * n)) :=
sorry

end prod_inequality_l416_416563


namespace greatest_divisor_of_four_consecutive_integers_l416_416981

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416981


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416666

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416666


namespace sum_of_digits_property_mod_9_l416_416540

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_property_mod_9 (n : ℕ) : sum_of_digits n % 9 = n % 9 :=
  sorry

example : sum_of_digits(sum_of_digits(sum_of_digits(2005 ^ 2005))) = 7 :=
  sorry

end sum_of_digits_property_mod_9_l416_416540


namespace four_consecutive_product_divisible_by_12_l416_416982

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416982


namespace expected_number_of_heads_l416_416019

def probability_heads_after_up_to_four_flips : ℝ :=
  1 / 2 + 1 / 4 + 1 / 8 + 1 / 16

theorem expected_number_of_heads (n : ℕ) (h : n = 80)
  (p_heads : ℝ) (h_p_heads : p_heads = probability_heads_after_up_to_four_flips):
  (n : ℝ) * p_heads = 75 :=
by
  intros
  rw [h, h_p_heads]
  sorry

end expected_number_of_heads_l416_416019


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416917

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416917


namespace probability_sum_less_than_product_l416_416140

def set_of_numbers := {1, 2, 3, 4, 5, 6, 7}

def count_valid_pairs : ℕ :=
  set_of_numbers.to_list.product set_of_numbers.to_list
    |>.count (λ (ab : ℕ × ℕ), (ab.1 - 1) * (ab.2 - 1) > 1)

def total_combinations := (set_of_numbers.to_list).length ^ 2

theorem probability_sum_less_than_product :
  (count_valid_pairs : ℚ) / total_combinations = 36 / 49 :=
by
  -- Placeholder for proof, since proof is not requested
  sorry

end probability_sum_less_than_product_l416_416140


namespace vasya_numbers_l416_416184

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416184


namespace divisor_of_four_consecutive_integers_l416_416661

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416661


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416729

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416729


namespace c10_is_107_l416_416043

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- this will not be used, but required for ℕ to ℕ functions.
  | 1 => 3
  | 2 => 5
  | (n + 1) => 2 * (sequence n) - (sequence (n - 1)) + 2

theorem c10_is_107 : sequence 10 = 107 := 
sorry

end c10_is_107_l416_416043


namespace horner_poly_at_point_l416_416286

noncomputable def horner (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldr (λ coeff acc, acc * x + coeff) 0

def poly_coeffs : List ℝ := [1, 8, 7, 6, 5, 4, 3]

theorem horner_poly_at_point :
  horner poly_coeffs 0.4 = 5.2 :=
by
  simp [horner, poly_coeffs]
  sorry

end horner_poly_at_point_l416_416286


namespace seventieth_even_integer_l416_416188

theorem seventieth_even_integer : 2 * 70 = 140 :=
by
  sorry

end seventieth_even_integer_l416_416188


namespace positive_difference_jo_kate_sum_l416_416018

theorem positive_difference_jo_kate_sum :
  let jo_sum := (100 * 101) / 2
  let rounded_sum := 5050 + 500 -- Based on the calculated sum of 5550
  in rounded_sum - jo_sum = 500 :=
by
  sorry

end positive_difference_jo_kate_sum_l416_416018


namespace value_is_66_l416_416238

noncomputable def certain_number : ℝ := 22.142857142857142
def add_five (x : ℝ) : ℝ := x + 5
def multiply_seven (a : ℝ) : ℝ := a * 7
def divide_five (b : ℝ) : ℝ := b / 5
def subtract_five (c : ℝ) : ℝ := c - 5
def double (d : ℝ) : ℝ := d * 2

theorem value_is_66 : 
  let x := certain_number,
      a := add_five x,
      b := multiply_seven a,
      c := divide_five b,
      d := subtract_five c,
      y := double d 
  in y = 66 :=
by 
  let x := certain_number,
  let a := add_five x,
  let b := multiply_seven a,
  let c := divide_five b,
  let d := subtract_five c,
  let y := double d,
  sorry

end value_is_66_l416_416238


namespace total_visitors_l416_416626

noncomputable def visitors_questionnaire (V E U : ℕ) : Prop :=
  (130 ≠ E ∧ E ≠ U) ∧ 
  (E = U) ∧ 
  (3 * V = 4 * E) ∧ 
  (V = 130 + 3 / 4 * V)

theorem total_visitors (V : ℕ) : visitors_questionnaire V V V → V = 520 :=
by sorry

end total_visitors_l416_416626


namespace dhoni_spent_300_dollars_l416_416308

theorem dhoni_spent_300_dollars :
  ∀ (L S X : ℝ),
  L = 6 →
  S = L - 2 →
  (X / S) - (X / L) = 25 →
  X = 300 :=
by
intros L S X hL hS hEquation
sorry

end dhoni_spent_300_dollars_l416_416308


namespace greatest_divisor_of_consecutive_product_l416_416860

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416860


namespace taxi_speed_l416_416269

variable (v_b v_t : ℕ) -- speeds of the bus and taxi respectively

def overtakes_in_two_hours (v_b v_t : ℕ) :=
  4 * v_b + 2 * v_t = 6 * v_b

theorem taxi_speed :
  (v_t = v_b + 30) → 
  (overtakes_in_two_hours v_b v_t) →
  v_t = 45 :=
by
  intros h1 h2
  rw [<-h1] at h2
  sorry

end taxi_speed_l416_416269


namespace area_closed_figure_l416_416081

theorem area_closed_figure (A : ℝ) :
  (∫ x in 0..1, (x^2 - x^3)) = A :=
sorry

end area_closed_figure_l416_416081


namespace probability_of_same_length_segments_l416_416511

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l416_416511


namespace simplify_and_evaluate_expression_l416_416582

theorem simplify_and_evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 2 → -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l416_416582


namespace negation_of_p_l416_416455

def p (x : ℝ) : Prop := x^3 - x^2 + 1 < 0

theorem negation_of_p : (¬ ∀ x : ℝ, p x) ↔ ∃ x : ℝ, ¬ p x := by
  sorry

end negation_of_p_l416_416455


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416732

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416732


namespace distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side_l416_416568

-- Let the geometry context be defined
variable (A B C H O : Point)
variable [triangle A B C]
variable [orthocenter H A B C]
variable [circumcenter O A B C]

-- Statement to prove
theorem distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side :
  dist A H = 2 * dist O (line B C) :=
begin
  sorry
end

end distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side_l416_416568


namespace ChickenFriedSteakMenuPrice_l416_416594

variables (SalisburySteakPrice RobMealPriceHalfOff TimeOfMeal TotalBill : ℝ)

-- Conditions
def SalisburySteakCosts : ℝ := 16
def HalfOffMeal (price : ℝ) : ℝ := price / 2
def TotalBillAtDiscount (curtisPrice robPrice : ℝ) : ℝ := curtisPrice + robPrice
def DiscountTime := TimeOfMeal >= 14 ∧ TimeOfMeal <= 16
def CurtisCost := HalfOffMeal SalisburySteakCosts
def MealTimeDiscounted := 15 -- time they ate

-- Mathematical problem to prove
theorem ChickenFriedSteakMenuPrice 
  ( MealTime : TimeOfMeal = 15 ) 
  ( CurtisMeal : SalisburySteakPrice = 16 ) 
  ( BetweenDiscountTime : DiscountTime ) 
  ( TotalCost : TotalBill = 17 ) 
  : RobMealPriceHalfOff * 2 = 18 :=
by
  sorry

end ChickenFriedSteakMenuPrice_l416_416594


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416722

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416722


namespace profit_maximization_l416_416262

-- Define the conditions 
variable (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5)

-- Expression for yield ω
noncomputable def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

-- Expression for profit function L(x)
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

-- Theorem stating the profit function expression and its maximum
theorem profit_maximization (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) :
  profit x = 64 - 48 / (x + 1) - 3 * x ∧ 
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ 5 → profit x₀ ≤ profit 3) :=
sorry

end profit_maximization_l416_416262


namespace sequence_arithmetic_expression_an_l416_416373

noncomputable def Sn (n : ℕ) : ℚ := -- To be defined: S_n in terms of n
sorry

noncomputable def an (n : ℕ) : ℚ := -- To be defined: a_n in terms of n
sorry

theorem sequence_arithmetic (n : ℕ) (h1 : ∀ n ≥ 2, an n + 2 * Sn n * Sn (n-1) = 0) (h2 : an 1 = 1 / 2) :
  ∃ (a : ℚ) (d : ℚ), a = 2 ∧ d = 2 ∧ ∀ n, (1 / Sn n) = a + n * d := 
sorry

theorem expression_an (n : ℕ) (h1 : ∀ n ≥ 2, an n + 2 * Sn n * Sn (n-1) = 0) (h2 : an 1 = 1 / 2) (h3 : ∀ n, 1 / Sn n = 2 + 2 * (n - 1)) :
  an n = if n = 1 then (1 / 2) else (-1 / (2 * n * (n - 1))) := 
sorry

end sequence_arithmetic_expression_an_l416_416373


namespace divisor_of_product_of_four_consecutive_integers_l416_416711

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416711


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416877

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416877


namespace greatest_divisor_four_consecutive_l416_416697

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416697


namespace ramu_total_profit_percent_l416_416572

def cost_price (purchase repairs taxes insurance : ℕ) : ℕ :=
  purchase + repairs + taxes + insurance

def total_profit_percent (purchaseA purchaseB purchaseC repairsA repairsB repairsC
                         taxesA taxesB taxesC insuranceA insuranceB insuranceC
                         sellingA sellingB sellingC : ℕ) : ℚ :=
  let CP_A := cost_price purchaseA repairsA taxesA insuranceA
  let CP_B := cost_price purchaseB repairsB taxesB insuranceB
  let CP_C := cost_price purchaseC repairsC taxesC insuranceC
  let total_CP := CP_A + CP_B + CP_C
  let total_SP := sellingA + sellingB + sellingC
  let total_profit := total_SP - total_CP
  (total_profit : ℚ) / total_CP * 100

theorem ramu_total_profit_percent :
  total_profit_percent 36400 45200 52000
                       8000 6000 4000
                       4500 5000 6000
                       2500 3000 3500
                       68400 82000 96000 ≈ 39.93 := by
  sorry

end ramu_total_profit_percent_l416_416572


namespace num_integer_values_of_n_l416_416357

theorem num_integer_values_of_n : 
  {n : ℤ | ∃ k : ℕ, 8000 * (2^n) * (5^(-n)) = k}.card = 10 := sorry

end num_integer_values_of_n_l416_416357


namespace parallelepiped_volume_l416_416001

-- Define the entities used in the problem
variables {A A1 B D C1 : Type}
variable [metric_space A]
variable [metric_space A1]
variable [metric_space B]
variable [metric_space D]
variable [metric_space C1]

-- Define the diagonal length and area of the triangle
def diagonal_length (d : ℝ) : Prop := true
def triangle_area (S : ℝ) : Prop := true

-- Define the existence of the required triangle and volume condition
theorem parallelepiped_volume
  (ABCDA1B1C1D1 : Type)
  (AC1 : Type)
  (A1 : A1)
  (B : B)
  (D : D)
  (d : ℝ)
  (S : ℝ)
  (h_diagonal : diagonal_length d)
  (h_area : triangle_area S) :
  ∃ T : Type, 
    (T = ({ distance : ℝ // distance_to A1 AC1 distance ∧ distance_to B AC1 distance ∧ distance_to D AC1 distance }) ∧
    volume ABCDA1B1C1D1 = 2 * d * S) :=
begin
  sorry
end

end parallelepiped_volume_l416_416001


namespace find_a_tangent_range_of_a_l416_416407

noncomputable def f (x : ℝ) (a : ℝ) := log x - a * x
noncomputable def g (x : ℝ) := (log x / (x + 1)) + (1 / (Real.exp 1 * (x + 1)))

theorem find_a_tangent (a : ℝ) (x : ℝ) 
  (h1 : f x a = log x - a * x) 
  (h2 : x - (log x - a * x) - 1 - log 2 = 0) 
  (h3 : 1/x - a = 1) : 
  a = 1 := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, (x + 1) * f x a ≤ log x - x / (Real.exp 1)) → 
  a ∈ Set.Ici (1 / Real.exp 1) := sorry

end find_a_tangent_range_of_a_l416_416407


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416889

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416889


namespace complex_determinant_solution_l416_416300

theorem complex_determinant_solution (z : ℂ) (H : (1 * z * complex.I - (-1) * z) = 2) :
  z = 1 - complex.I :=
by
  sorry

end complex_determinant_solution_l416_416300


namespace divisor_of_product_of_four_consecutive_integers_l416_416715

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416715


namespace product_of_consecutive_integers_l416_416744

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416744


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416864

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416864


namespace two_lines_no_common_points_is_necessary_but_not_sufficient_for_parallel_lines_l416_416474

def two_lines_no_common_points (L1 L2 : Type) [LinearMap L1 L2] : Prop :=
  ∀ x : L1, ∀ y : L2, x ≠ y

def two_lines_parallel (L1 L2 : Type) [LinearMap L1 L2] : Prop :=
  ∀ x y : L1, ∃ k : ℝ, x = k • y

theorem two_lines_no_common_points_is_necessary_but_not_sufficient_for_parallel_lines
    (L1 L2 : Type) [LinearMap L1 L2] :
  (two_lines_no_common_points L1 L2 → ¬ two_lines_parallel L1 L2) ∧
  (two_lines_parallel L1 L2 → two_lines_no_common_points L1 L2) :=
by
  sorry

end two_lines_no_common_points_is_necessary_but_not_sufficient_for_parallel_lines_l416_416474


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416901

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416901


namespace circuit_does_not_fail_l416_416617

def probability_of_failure (p : ℝ) (T : ℝ) : Prop := p > 0 ∧ p < 1

def event_A1 (p1 : ℝ) (T : ℝ) : Prop :=
  probability_of_failure p1 T

def event_A2 (p2 : ℝ) (T : ℝ) : Prop :=
  probability_of_failure p2 T

def event_A3 (p3 : ℝ) (T : ℝ) : Prop :=
  probability_of_failure p3 T

def event_A4 (p4 : ℝ) (T : ℝ) : Prop :=
  probability_of_failure p4 T

theorem circuit_does_not_fail (p1 p2 p3 p4 : ℝ) (T : ℝ)
  (h1 : event_A1 p1 T)
  (h2 : event_A2 p2 T)
  (h3 : event_A3 p3 T)
  (h4 : event_A4 p4 T) :
  let P_A := (1 - p1) * (1 - p2 * p3) * (1 - p4)
  in P_A = (1 - p1) * (1 - p2 * p3) * (1 - p4) :=
by
  sorry

end circuit_does_not_fail_l416_416617


namespace min_number_of_zeros_l416_416084

noncomputable def f : ℝ → ℝ := sorry

lemma symmetry_1 (x : ℝ) : f (2 - x) = f (2 + x) := sorry
lemma symmetry_2 (x : ℝ) : f (5 + x) = f (5 - x) := sorry
lemma initial_zero : f 0 = 0 := sorry

theorem min_number_of_zeros : ∃ n ≥ 14, ∀ x ∈ set.Icc (-21 : ℝ) 21, f x = 0 := sorry

end min_number_of_zeros_l416_416084


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416674

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416674


namespace four_consecutive_product_divisible_by_12_l416_416993

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416993


namespace product_inequality_l416_416038

theorem product_inequality (n : ℕ) (a : Fin n → ℝ) (k : ℕ) (h1 : 0 < a ∧ ∀ i, 0 < a i) 
  (h2 : 1 ≤ k ∧ k ≤ n - 1) :
  let S (m : ℕ) : ℝ := ∑ s in Finset.powersetLen m (Finset.univ : Finset (Fin n)), ∏ i in s, a i
  in (S k) * (S (n - k)) ≥ (Nat.choose n k)^2 * ∏ i in (Finset.univ : Finset (Fin n)), a i :=
by
  sorry

end product_inequality_l416_416038


namespace problem_l416_416453

def operation (a b : ℕ) : ℕ :=
  (a + b) * (a ^ 2 - a * b + b ^ 2)

theorem problem (a b : ℕ) (h : a = 1 ∧ b = 0) : operation a b = 1 :=
  by 
    cases h with ha hb
    rw [ha, hb]
    sorry

end problem_l416_416453


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416865

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416865


namespace vasya_numbers_l416_416175

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416175


namespace greatest_integer_less_than_neg_21_over_5_l416_416196

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ n : ℤ, n < -21 / 5 ∧ ∀ m : ℤ, m < -21 / 5 → m ≤ n :=
begin
  use -5,
  split,
  { linarith },
  { intros m h,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416196


namespace doodads_for_thingamabobs_l416_416488

-- Definitions for the conditions
def doodads_per_widgets : ℕ := 18
def widgets_per_thingamabobs : ℕ := 11
def widgets_count : ℕ := 5
def thingamabobs_count : ℕ := 4
def target_thingamabobs : ℕ := 80

-- Definition for the final proof statement
theorem doodads_for_thingamabobs : 
    doodads_per_widgets * (target_thingamabobs * widgets_per_thingamabobs / thingamabobs_count / widgets_count) = 792 := 
by
  sorry

end doodads_for_thingamabobs_l416_416488


namespace divisor_of_product_of_four_consecutive_integers_l416_416708

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416708


namespace correct_calculation_l416_416209

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l416_416209


namespace max_distance_is_eight_l416_416027

-- Definition of a circle and a line
def circle (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0

-- Function to calculate the distance from a point to a line
def point_to_line_distance (x y : ℝ) : ℝ := 
  abs (3 * x + 4 * y - 2) / sqrt (3^2 + 4^2)

-- Function to calculate the maximum distance from a point on the circle to the line
def max_distance_from_circle_to_line : ℝ := 
  let center_x := 5
  let center_y := 3
  let radius := 3
  let center_to_line_distance := point_to_line_distance center_x center_y
  radius + center_to_line_distance

-- The theorem stating the desired maximum distance
theorem max_distance_is_eight : max_distance_from_circle_to_line = 8 :=
by
  sorry

end max_distance_is_eight_l416_416027


namespace sequence_common_formula_and_sum_l416_416380

theorem sequence_common_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = 2 * n) ∧
  (∀ n, b n = 1 / ((n + 1) * a n)) ∧
  (∀ n, S n = ∑ i in finset.range n, b i) →
  (∀ n, S n = (1 / 2 * (1 - 1 / (n + 1)))) →
  S 100 = 50 / 101 := 
sorry

end sequence_common_formula_and_sum_l416_416380


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416871

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416871


namespace cat_toy_cost_correct_l416_416013

-- Define the initial amount of money Jessica had.
def initial_amount : ℝ := 11.73

-- Define the amount left after spending.
def amount_left : ℝ := 1.51

-- Define the cost of the cat toy.
def toy_cost : ℝ := initial_amount - amount_left

-- Theorem and statement to prove the cost of the cat toy.
theorem cat_toy_cost_correct : toy_cost = 10.22 := sorry

end cat_toy_cost_correct_l416_416013


namespace vasya_numbers_l416_416171

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416171


namespace faucets_fill_correct_liters_l416_416588

variable (f g h i j : ℕ) -- Define the variables as natural numbers

-- Defining the condition that f faucets can fill g liters in h minutes
def rate_per_faucet_per_minute (f g h : ℕ) : ℚ := g / (f * h : ℝ)

-- Defining the target liters of water filled by i faucets in j minutes
def liters_filled (f g h i j : ℕ) : ℚ := (g * i * j) / (f * h)

-- The theorem we want to prove
theorem faucets_fill_correct_liters (f g h i j : ℕ) :
  liters_filled f g h i j = (g * i * j) / (f * h) :=
sorry

end faucets_fill_correct_liters_l416_416588


namespace length_MN_l416_416487

open Real

-- Define the trapezoid structure
structure Trapezoid (A B C D : Type) :=
  (AB : ℝ)
  (BC AD : ℝ)
  (angleA angleD : ℝ)
  (midM : ℝ)
  (midN : ℝ)
  (parallel : BC = AD)
  (bc_length : BC = 600)
  (ad_length : AD = 1800)
  (angleA_eq : angleA = 45)
  (angleD_eq : angleD = 45)
  (midM_eq : midM = BC / 2)
  (midN_eq : midN = AD / 2)

-- Define the trapezoid ABCD with given properties
def ABCD := Trapezoid ℝ ℝ ℝ ℝ

-- Define a theorem stating MN equals 948.683
theorem length_MN (t : ABCD) : 
  ∀ (MN : ℝ), MN = 948.683 :=
by
  intro MN
  -- midpoints calculation
  have h1 : t.midN = 900 := by sorry
  have h2 : t.midM = 300 := by sorry
  -- Calculate MN using Pythagorean theorem
  -- MN = sqrt(900^2 + 300^2) = sqrt(810000 + 90000) = sqrt(900000) = 948.683
  have h3 : MN = sqrt (t.midN * t.midN + t.midM * t.midM) := by sorry
  exact calc
    MN = sqrt (900^2 + 300^2) : by sorry
    ... = sqrt (900000) : by sorry
    ... = 948.683 : by sorry

end length_MN_l416_416487


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416892

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416892


namespace polynomial_descending_order_l416_416102

def poly_original : ℤ[X] := 2 * (X^2) * C Y - 3 * (X^3) - (X * Y^3) + 1
def poly_expected : ℤ[X] := -3 * (X^3) + 2 * (X^2) * C Y - (X * Y^3) + 1

theorem polynomial_descending_order :
  poly_original = poly_expected := by
  sorry

end polynomial_descending_order_l416_416102


namespace domain_of_function_l416_416603

theorem domain_of_function (x : ℝ) : (log 0.5 (4 * x - 3) ≥ 0 ∧ 4 * x - 3 > 0) ↔ (3 / 4 < x ∧ x ≤ 1) :=
sorry

end domain_of_function_l416_416603


namespace area_of_region_enclosed_by_graph_l416_416303

theorem area_of_region_enclosed_by_graph :
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * (abs (x - y)) + 4 * (abs (x + y))) → 
  (64 : ℝ) := 
sorry

end area_of_region_enclosed_by_graph_l416_416303


namespace divisor_of_product_of_four_consecutive_integers_l416_416718

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416718


namespace angle_between_hyperbola_asymptotes_l416_416319

theorem angle_between_hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 3) = 1) →
  angle_between_asymptotes x y = π / 3 :=
by
  sorry

end angle_between_hyperbola_asymptotes_l416_416319


namespace internal_angle_bisectors_concur_l416_416498

-- Define the given conditions
variables {A B C M N P : Point}
variables (AB AC BC CA BAM CNM: Line) /- All lines -/
variables (isos_triangle_ABC : AB = AC)
variables (M_on_BC : M ∈ BC) (N_on_CA : N ∈ CA)
variables (angle_condition : ∠BAM = ∠CNM)
variables (P_meeting : P ∈ AB ∧ P ∈ MN)

-- Define the proof problem in Lean 4
theorem internal_angle_bisectors_concur :
  ∃ Q : Point, isos_triangle_ABC ∧ M_on_BC ∧ N_on_CA ∧ angle_condition ∧ P_meeting → 
  (Q ∈ BC ∧ Q ∈ angle_bisector ∠BAM ∧ Q ∈ angle_bisector ∠BPM) := 
sorry

end internal_angle_bisectors_concur_l416_416498


namespace sum_third_sequence_l416_416605

theorem sum_third_sequence (n : ℕ) (a₁ b₁ sₐ s_b : ℝ) (h₁ : n ≥ 2) :
  ∑ i in finset.range n, (a₁ + i * (sₐ - 2 * a₁) / (n - 1)) * (b₁ + i * (s_b - 2 * b₁) / (n - 1)) =
  n / (6 * (n - 1)) * ((2 * n - 1) * sₐ * s_b - (n + 1) * (a₁ * s_b + b₁ * sₐ - 2 * a₁ * b₁)) :=
sorry

end sum_third_sequence_l416_416605


namespace Vasya_numbers_l416_416162

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416162


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416958

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416958


namespace find_m_l416_416527

/-- Definition of the set of positive integer divisors of \( 15^{10} \) -/
def S : Set ℕ := {d | d ∣ 15^10}

/-- Value of m that makes the probability that both \( a_1 \) divides \( a_2 \) and \( a_2 \) divides \( a_3 \) given the conditions is \( \frac{m}{n} \) where \( m \) and \( n \) are relatively prime positive integers -/
theorem find_m :
  let chosen_numbers := (fin 3) → S,
  let condition_divisibility (a1 a2 a3 : S) : Prop :=
    a1 ∣ a2 ∧ a2 ∣ a3,
  ∃ m n : ℕ, gcd m n = 1 ∧
             m = 48400 ∧
             (∃ (a1 a2 a3 : S), chosen_numbers a1 a2 a3 ∧ condition_divisibility a1 a2 a3) :=
sorry

end find_m_l416_416527


namespace divisor_of_product_of_four_consecutive_integers_l416_416720

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416720


namespace mean_is_12_point_8_l416_416094

variable (m : ℝ)
variable median_condition : m + 7 = 12

theorem mean_is_12_point_8 (m : ℝ) (median_condition : m + 7 = 12) : 
(mean := (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5) = 64 / 5 :=
by {
  sorry
}

end mean_is_12_point_8_l416_416094


namespace div_product_four_consecutive_integers_l416_416797

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416797


namespace complex_div_conjugate_l416_416437

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416437


namespace ken_pencils_kept_l416_416021

theorem ken_pencils_kept :
  let total_pencils := 50
  let pencils_to_manny := 10
  let pencils_to_nilo := pencils_to_manny + 10
  let pencils_given_away := pencils_to_manny + pencils_to_nilo
  let pencils_kept := total_pencils - pencils_given_away
  in
  pencils_kept = 20 :=
by
  sorry

end ken_pencils_kept_l416_416021


namespace greatest_divisor_of_four_consecutive_integers_l416_416803

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416803


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416907

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416907


namespace bridge_length_l416_416241

theorem bridge_length (train_length : ℕ) (signal_time : ℕ) (cross_time_minutes : ℕ) :
  train_length = 600 →
  signal_time = 40 →
  cross_time_minutes = 6 →
  let speed := train_length / signal_time in
  let cross_time_seconds := cross_time_minutes * 60 in
  let time_just_bridge := cross_time_seconds - signal_time in
  let bridge_length := speed * time_just_bridge in
  bridge_length = 4800 :=
by
  intros hTrainLength hSignalTime hCrossTimeMinutes
  simp [hTrainLength, hSignalTime, hCrossTimeMinutes]
  let speed := 600 / 40
  let cross_time_seconds := 6 * 60
  let time_just_bridge := cross_time_seconds - 40
  let bridge_length := speed * time_just_bridge
  have speed_15 : speed = 15 := by norm_num
  rw speed_15
  have bridge_length_4800 : 15 * 320 = 4800 := by norm_num
  rw bridge_length_4800
  rfl

end bridge_length_l416_416241


namespace calculation_division_l416_416282

theorem calculation_division :
  ((27 * 0.92 * 0.85) / (23 * 1.7 * 1.8)) = 0.3 :=
by
  sorry

end calculation_division_l416_416282


namespace factorial_fraction_l416_416197

theorem factorial_fraction :
  (16.factorial / (6.factorial * 10.factorial) : ℚ) = 728 :=
by
  sorry

end factorial_fraction_l416_416197


namespace sequence_not_periodic_l416_416532

def leading_digit (n : ℕ) : ℕ :=
  Nat.floor (Real.sqrt n) / 10 ^ (Nat.log10 (Nat.floor (Real.sqrt n)))

def is_periodic (a : ℕ → ℕ) (P : ℕ) : Prop :=
  ∀ n : ℕ, a (n + P) = a n

theorem sequence_not_periodic : ¬ ∃ P : ℕ, is_periodic (fun n => leading_digit n) P :=
sorry

end sequence_not_periodic_l416_416532


namespace isosceles_right_triangle_mid_segment_l416_416389

open EuclideanGeometry

theorem isosceles_right_triangle_mid_segment (A B C D E : Point) 
(h_triangle : isIsoscelesRightTriangle A B C)
(hD : isOnLineSegment D B C (1 / 3))
(hE : isPerpendicular BE AD ∧ pointOnLineSegment E A C) :
  isEqualSegment AE EC :=
sorry

end isosceles_right_triangle_mid_segment_l416_416389


namespace greatest_divisor_four_consecutive_l416_416690

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416690


namespace sum_of_reciprocals_l416_416110

variable (x y : ℝ)

theorem sum_of_reciprocals (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1 / x) + (1 / y) = 3 := 
sorry

end sum_of_reciprocals_l416_416110


namespace player_A_wins_iff_n_is_odd_l416_416251

-- Definitions of the problem conditions
structure ChessboardGame (n : ℕ) :=
  (stones : ℕ := 99)
  (playerA_first : Prop := true)
  (turns : ℕ := n * 99)

-- Statement of the problem
theorem player_A_wins_iff_n_is_odd (n : ℕ) (g : ChessboardGame n) : 
  PlayerA_has_winning_strategy ↔ n % 2 = 1 := 
sorry

end player_A_wins_iff_n_is_odd_l416_416251


namespace rationalize_and_sum_l416_416573

def gcd (a b : Int) : Int := sorry
def is_not_divisible_by_square_of_any_prime (x : Int) : Prop := sorry

theorem rationalize_and_sum :
  (∃ (A B C D : ℤ), 
      (A * Int.sqrt B + C) / D = (6 - 3)^((1:ℝ)/2) - (3 - 3)^((1:ℝ)/2) ∧ 
      D > 0 ∧ 
      is_not_divisible_by_square_of_any_prime B ∧ 
      gcd A (gcd C D) = 1 ∧ 
      A + B + C + D = 7) :=
sorry

end rationalize_and_sum_l416_416573


namespace maria_original_number_l416_416050

-- Definition of the conditions
def maria_final_result (x : ℝ) : ℝ := (2 * (x + 3) - 2) / 3

-- The proof statement
theorem maria_original_number : (maria_final_result x = 8) → x = 10 :=
by
  intros h
  calc
    maria_final_result x = 8 : h
    (2 * (x + 3) - 2) / 3 = 8 : by rw [h]
    2 * (x + 3) - 2 = 24 : by linarith
    2x + 6 - 2 = 24 : by rw [mul_add, mul_one, add_sub_assoc]
    2x + 4 = 24 : by simp
    2x = 20 : by linarith
    x = 10 : by linarith

end maria_original_number_l416_416050


namespace find_length_of_AC_l416_416372

-- Define the right triangle and its properties
structure RightTriangle :=
  (A B C : Type)
  (angle_BAC : ℝ)
  (angle_ABC : ℝ)
  (inscribed_circle_radius : ℝ)

-- Define the specific triangle in question
def triangle_ABC : RightTriangle :=
{ A := ℝ,
  B := ℝ,
  C := ℝ,
  angle_BAC := 90,
  angle_ABC := 60,
  inscribed_circle_radius := 8 }

-- Define the length of AC
def length_of_AC (triangle : RightTriangle) : ℝ :=
  if triangle = triangle_ABC then (24 * Real.sqrt 3) + 24 else 0

-- The proof problem statement
theorem find_length_of_AC :
  length_of_AC triangle_ABC = 24 * Real.sqrt 3 + 24 := 
sorry

end find_length_of_AC_l416_416372


namespace find_m_n_l416_416612
-- Import the required mathematics library

-- Definitions
noncomputable def m_n_sum (ABC_perimeter : ℝ) (angle_BAC : ℝ) (circle_radius : ℝ) (center_O_on_AB: Prop) (tangent_circle_AC : Prop) (tangent_circle_BC : Prop) : ℝ :=
  let OB := 15 * (9 + 3 * real.sqrt 3) in 
  15

-- Theorem
theorem find_m_n (ABC_perimeter : ℝ) (angle_BAC : ℝ) (circle_radius : ℝ) (center_O_on_AB: Prop) (tangent_circle_AC : Prop) (tangent_circle_BC : Prop) : m_n_sum ABC_perimeter angle_BAC circle_radius center_O_on_AB tangent_circle_AC tangent_circle_BC = 15 :=
by
  have h1 : OB = 15 * (9 + 3 * real.sqrt 3) := sorry  -- Conditions are incorporated
  sorry

-- Given conditions
#eval find_m_n 180 90 15 
(λ x, true)  -- Angle BAC is a right angle
(λ x, true)  -- Circle on AB
(λ x, true)  -- Circle tangent to AC and BC

end find_m_n_l416_416612


namespace four_consecutive_product_divisible_by_12_l416_416995

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416995


namespace C_should_pay_correct_amount_l416_416272

def A_oxen_months : ℕ := 10 * 7
def B_oxen_months : ℕ := 12 * 5
def C_oxen_months : ℕ := 15 * 3
def D_oxen_months : ℕ := 20 * 6

def total_rent : ℚ := 225

def C_share_of_rent : ℚ :=
  total_rent * (C_oxen_months : ℚ) / (A_oxen_months + B_oxen_months + C_oxen_months + D_oxen_months)

theorem C_should_pay_correct_amount : C_share_of_rent = 225 * (45 : ℚ) / 295 := by
  sorry

end C_should_pay_correct_amount_l416_416272


namespace greatest_divisor_four_consecutive_l416_416685

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416685


namespace problem1_problem2_l416_416235

-- Problem 1 Statement
theorem problem1 (A B C D: Point) 
  (triangle_ABC: Point_triangle A B C) 
  (D_internal: D ∈ interior_triangle A B C):
  (BC: Float) / (min {AD, BD, CD}) >= 
  (if (angle_A < 90) then 
    2 * sin A 
   else 
    2) := sorry

-- Problem 2 Statement
theorem problem2 (A B C D E: Point) 
  (convex_quadrilateral: Quadrilateral A B C D) 
  (E_internal: E ∈ interior_quadrilateral A B C D):
  (k >= (2 * sin 70)) := sorry

end problem1_problem2_l416_416235


namespace probability_of_summer_and_autumn_l416_416593

theorem probability_of_summer_and_autumn :
  let stamps := { "Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold" }
  let draws := (stamps.powerset.filter (fun s => s.card = 2)).toList
  let favorable := draws.count (fun s => "Summer Begins" ∈ s ∧ "Autumn Equinox" ∈ s)
  (favorable : ℚ) / draws.length = 1 / 6 := by
  sorry

end probability_of_summer_and_autumn_l416_416593


namespace rebecca_eggs_l416_416067

theorem rebecca_eggs (groups eggs_per_group : ℕ) (h1 : groups = 3) (h2 : eggs_per_group = 6) : 
  (groups * eggs_per_group = 18) :=
by
  sorry

end rebecca_eggs_l416_416067


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416885

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416885


namespace apples_in_each_crate_l416_416117

theorem apples_in_each_crate
  (num_crates : ℕ) 
  (num_rotten : ℕ) 
  (num_boxes : ℕ) 
  (apples_per_box : ℕ) 
  (total_good_apples : ℕ) 
  (total_apples : ℕ)
  (h1 : num_crates = 12) 
  (h2 : num_rotten = 160) 
  (h3 : num_boxes = 100) 
  (h4 : apples_per_box = 20) 
  (h5 : total_good_apples = num_boxes * apples_per_box) 
  (h6 : total_apples = total_good_apples + num_rotten) : 
  total_apples / num_crates = 180 := 
by 
  sorry

end apples_in_each_crate_l416_416117


namespace greatest_divisor_of_four_consecutive_integers_l416_416812

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416812


namespace greatest_divisor_of_four_consecutive_integers_l416_416967

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416967


namespace friendship_graph_n_values_l416_416633

theorem friendship_graph_n_values (n : ℕ) (students : Finset α)
  (friends : α → Finset α)
  (Hfriends_count : ∀ s ∈ students, (friends s).card = 2023)
  (Hcommon_friends : ∀ s t ∈ students, (s ≠ t) → (s ∉ friends t) → ((friends s) ∩ (friends t)).card = 2022) :
  n = 2024 ∨ n = 2026 ∨ n = 2028 ∨ n = 2696 ∨ n = 4044 := 
sorry

end friendship_graph_n_values_l416_416633


namespace angle_alpha_range_l416_416382

/-- Given point P (tan α, sin α - cos α) is in the first quadrant, 
and 0 ≤ α ≤ 2π, then the range of values for angle α is (π/4, π/2) ∪ (π, 5π/4). -/
theorem angle_alpha_range (α : ℝ) 
  (h0 : 0 ≤ α) (h1 : α ≤ 2 * Real.pi) 
  (h2 : Real.tan α > 0) (h3 : Real.sin α - Real.cos α > 0) : 
  (Real.pi / 4 < α ∧ α < Real.pi / 2) ∨ 
  (Real.pi < α ∧ α < 5 * Real.pi / 4) :=
sorry

end angle_alpha_range_l416_416382


namespace magnitude_of_a_add_2b_l416_416422

-- Conditions
def vector_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
def vector_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

-- Statement of the theorem
theorem magnitude_of_a_add_2b : Real.sqrt ((vector_a.1 + 2 * vector_b.1) ^ 2 + (vector_a.2 + 2 * vector_b.2) ^ 2) = Real.sqrt 7 := 
sorry

end magnitude_of_a_add_2b_l416_416422


namespace Vasya_numbers_l416_416160

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416160


namespace quadratic_inequality_l416_416564

theorem quadratic_inequality (a b c : ℝ) (h : a^2 + a * b + a * c < 0) : b^2 > 4 * a * c := 
sorry

end quadratic_inequality_l416_416564


namespace four_consecutive_integers_divisible_by_12_l416_416777

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416777


namespace divisor_of_product_of_four_consecutive_integers_l416_416703

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416703


namespace greatest_divisor_four_consecutive_integers_l416_416838

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416838


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416867

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416867


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416928

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416928


namespace smallest_number_l416_416074

-- Define that Sunwoo has cards with the numbers 2 and 4
def has_card (n : Nat) : Prop := n = 2 ∨ n = 4

-- Prove that the smallest number Sunwoo can make with the cards is 24
theorem smallest_number (h2 : has_card 2) (h4 : has_card 4) : Nat :=
  let numbers := [2, 4]
  numbers.foldl (λ acc x => acc * 10 + x) 0
  = 24 := by sorry

end smallest_number_l416_416074


namespace lcm_ge_na1_l416_416065

theorem lcm_ge_na1 {n : ℕ} {a : ℕ} (a1 a2 : Fin n → ℕ) (h : ∀ i j : Fin n, i < j → a1 i < a2 j) 
  (ha : a = Nat.lcm (Finset.univ.image a1)) : a ≥ n * a1 0 :=
sorry

end lcm_ge_na1_l416_416065


namespace divisor_of_product_of_four_consecutive_integers_l416_416707

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416707


namespace greatest_divisor_of_four_consecutive_integers_l416_416972

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416972


namespace greatest_divisor_four_consecutive_l416_416700

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416700


namespace vasya_numbers_l416_416181

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416181


namespace greatest_divisor_four_consecutive_integers_l416_416835

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416835


namespace divisor_of_product_of_four_consecutive_integers_l416_416716

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416716


namespace worth_of_each_gold_bar_l416_416580

theorem worth_of_each_gold_bar
  (rows : ℕ) (gold_bars_per_row : ℕ) (total_worth : ℕ)
  (h1 : rows = 4) (h2 : gold_bars_per_row = 20) (h3 : total_worth = 1600000)
  (total_gold_bars : ℕ) (h4 : total_gold_bars = rows * gold_bars_per_row) :
  total_worth / total_gold_bars = 20000 :=
by sorry

end worth_of_each_gold_bar_l416_416580


namespace problem1_problem2_l416_416534

def f (a x : ℝ) : ℝ := log (4 ^ x + 1) / log 4 + a * x

-- Question (1)
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) → a = -1/2 :=
by
  sorry

-- Question (2)
theorem problem2 (m : ℝ) : (∀ x t : ℝ, x ∈ set.Ioo (-2 : ℝ) 1 → f m x + f m (-x) ≥ m * x + m) → -1 ≤ m ∧ m ≤ 1/2 :=
by
  sorry

end problem1_problem2_l416_416534


namespace point_D_on_circumcircle_l416_416240

theorem point_D_on_circumcircle
  (Q R : Type)
  (QC RB : ℝ)
  (B C M D : Type)
  (circle_Q: circle Q QC)
  (circle_R: circle R RB)
  (tangent_Q_B: tangent circle_Q B)
  (tangent_R_C: tangent circle_R C)
  (M_intersect: M = tangent_Q_B ∩ tangent_R_C)
  (triangle_MBC: triangle M B C)
  (angle_D_property: ∀ A : Type, angle D B C = 180 - angle M A C ∧ angle M B C ∧ angle M C B = 90)
  : lies_on_circumcircle D (circumcircle triangle_MBC) :=
sorry

end point_D_on_circumcircle_l416_416240


namespace vasya_numbers_l416_416156

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416156


namespace Vasya_numbers_l416_416158

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416158


namespace jo_kate_sum_difference_l416_416015

theorem jo_kate_sum_difference :
  let S_J := (100 * 101) / 2,
      S_K := 60 + 160 + 260 + 360 + 460 + 560 + 660 + 760 + 860 + 960
  in abs (S_J - S_K) = 150 := 
by 
  let S_J := (100 * 101) / 2
  let S_K := 60 + 160 + 260 + 360 + 460 + 560 + 660 + 760 + 860 + 960
  sorry

end jo_kate_sum_difference_l416_416015


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416908

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416908


namespace factorial_sum_div_l416_416356

theorem factorial_sum_div (n : ℕ) (hn : 0 < n) : 
  1 * ((2 * n - 1) !)
  - (2 ! * (2 * n - 2) !)
  + ⋯
  - ((2 * n - 2) ! * 2 !)
  + ((2 * n - 1) ! * 1 !) = 
  (2 * n)! / (n + 1) := 
  sorry

end factorial_sum_div_l416_416356


namespace solutions_to_f_eq_zero_in_0_6_l416_416033

noncomputable def f : ℝ → ℝ := sorry

def prop_f_periodic : Prop := ∀ x : ℝ, f(x + 3) = f(x)

def prop_f_odd : Prop := ∀ x : ℝ, f(-x) = -f(x)

def prop_f_2_zero : Prop := f(2) = 0

theorem solutions_to_f_eq_zero_in_0_6 :
  prop_f_periodic ∧ prop_f_odd ∧ prop_f_2_zero →
  (finset.card (finset.filter (λ x, f x = 0) (finset.Icc 0 6)) = 9) :=
begin
  sorry
end

end solutions_to_f_eq_zero_in_0_6_l416_416033


namespace dismissed_cases_l416_416254

theorem dismissed_cases (total_cases : Int) (X : Int)
  (total_cases_eq : total_cases = 17)
  (remaining_cases_eq : X = (2 * X / 3) + 1 + 4) :
  total_cases - X = 2 :=
by
  -- Placeholder for the proof
  sorry

end dismissed_cases_l416_416254


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416869

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416869


namespace length_of_BE_l416_416231

noncomputable def side_length_of_square : ℝ := real.sqrt 256

def CE_CF_eq (CE CF : ℝ) : Prop := CE = CF

def area_triangle_CEF (CE : ℝ) : Prop := 1 / 2 * CE^2 = 200

def BE_length (BE : ℝ) (CE CB : ℝ) : Prop :=
  BE^2 = CE^2 - CB^2

theorem length_of_BE : 
  ∃ BE : ℝ, 
    side_length_of_square = 16 ∧
    (∀ CE CF, CE_CF_eq CE CF → area_triangle_CEF CE) ∧
    BE_length BE 20 16 →
  BE = 12 :=
begin
  sorry
end

end length_of_BE_l416_416231


namespace mean_of_set_median_is_128_l416_416096

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l416_416096


namespace greatest_divisor_four_consecutive_l416_416692

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416692


namespace divisor_of_four_consecutive_integers_l416_416643

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416643


namespace four_consecutive_integers_divisible_by_12_l416_416770

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416770


namespace statement_A_statement_D_l416_416030

-- Definitions of parallel lines and planes
variables (a b : Line) (α β γ : Plane)

-- Statement A
theorem statement_A (h1 : Parallel a b) (h2 : InPlane b α) (h3 : ¬ InPlane a α) : ParallelLinePlane a α :=
sorry

-- Statement D
theorem statement_D (h4 : ParallelPlane α β) (h5 : Intersection α γ = a) (h6 : Intersection β γ = b) : Parallel a b :=
sorry

end statement_A_statement_D_l416_416030


namespace cost_of_16_pencils_and_10_notebooks_l416_416620

noncomputable def pencil_cost := 0.09
noncomputable def notebook_cost := 0.44

theorem cost_of_16_pencils_and_10_notebooks :
  ∃ (p n : ℝ), (7 * p + 8 * n = 4.15) ∧ (5 * p + 3 * n = 1.77) ∧ (16 * p + 10 * n = 5.84) :=
by
  use pencil_cost, notebook_cost
  have h1 : 7*pencil_cost + 8*notebook_cost = 4.15 := by sorry
  have h2 : 5*pencil_cost + 3*notebook_cost = 1.77 := by sorry
  have h3 : 16*pencil_cost + 10*notebook_cost = 5.84 := by sorry
  exact ⟨h1, h2, h3⟩

end cost_of_16_pencils_and_10_notebooks_l416_416620


namespace four_consecutive_product_divisible_by_12_l416_416990

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416990


namespace greatest_divisor_four_consecutive_l416_416699

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416699


namespace probability_of_drawing_stamps_l416_416591

theorem probability_of_drawing_stamps : 
  let stamps := ["Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold"]
  in (2 / (list.permutations stamps).length) = (1 / 6) := by
  sorry

end probability_of_drawing_stamps_l416_416591


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416921

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416921


namespace smallest_positive_non_palindrome_power_of_12_l416_416340

def is_palindrome (n : ℕ) : Bool :=
  let s := toDigits 10 n
  s = s.reverse

theorem smallest_positive_non_palindrome_power_of_12 : ∃ k : ℕ, k > 0 ∧ (12^k = 12 ∧ ¬ is_palindrome (12^k)) :=
by {
  sorry
}

end smallest_positive_non_palindrome_power_of_12_l416_416340


namespace greatest_divisor_of_four_consecutive_integers_l416_416807

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416807


namespace greatest_divisor_of_four_consecutive_integers_l416_416813

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416813


namespace sum_of_first_n_terms_is_n_squared_l416_416358

def sequence (n : ℕ) : ℕ := 2 * n - 1

def sum_of_sequence (n : ℕ) : ℕ :=
  (n * (sequence 1) + (n * (n - 1) / 2) * 2)

theorem sum_of_first_n_terms_is_n_squared (n : ℕ) : 
  sum_of_sequence n = n * n :=
by
  sorry

end sum_of_first_n_terms_is_n_squared_l416_416358


namespace four_consecutive_integers_divisible_by_12_l416_416780

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416780


namespace parabola_equation_l416_416075

def vertex := (5, -3)
def point := (3, 7)

def a := 5 / 2
def b := -25
def c := 119 / 2

theorem parabola_equation :
  ∃ (a b c : ℚ), 
  let y := λ x : ℚ, a * x^2 + b * x + c in 
  y = λ x, a * x^2 + b * x + c ∧
  y 5 = -3 ∧ y 3 = 7 :=
by
  use a, b, c
  sorry

end parabola_equation_l416_416075


namespace probability_of_sum_less_than_product_l416_416138

-- Define the problem conditions
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the event condition
def is_valid_pair (a b : ℕ) : Prop := (a ∈ set_of_numbers) ∧ (b ∈ set_of_numbers) ∧ (a * b > a + b)

-- Count the number of valid pairs
noncomputable def count_valid_pairs : ℕ :=
  set_of_numbers.sum (λ a, set_of_numbers.filter (is_valid_pair a).card)

-- Count the total possible pairs
noncomputable def total_pairs : ℕ :=
  set_of_numbers.card * set_of_numbers.card

-- Calculate the probability
noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / total_pairs

-- State the theorem
theorem probability_of_sum_less_than_product :
  probability = 25 / 49 :=
by sorry

end probability_of_sum_less_than_product_l416_416138


namespace triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint_l416_416570

-- Definitions
variable {A B C : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (G : A → B)
variable (H : A → A)
variable (O : A → A)

-- Theorem statement
theorem triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint 
  (homothety : ∀ (x : A), G (O x) = H x) 
  (midpoint_rotation : ∀ (A' B' C' : A), G (O (A')) = H (A')) 
  (distance_ratio: ∀ (x : A), dist O x = 2 * dist H (G x)) 
  (GA_AHAO: ∀ (x : A), dist G (A x) = (1/2) * dist G (H x) ) 
  : ∀ (x A' : A), dist x (H x) = 2 * dist (O x) (midpoint_rotation A' B' C') := 
by
  sorry

end triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint_l416_416570


namespace cos_alpha_minus_7pi_over_6_l416_416427

theorem cos_alpha_minus_7pi_over_6 (α : ℝ) (h : sin (π / 3 + α) = 1 / 3) :
  cos (α - 7 * π / 6) = -1 / 3 :=
by
  sorry

end cos_alpha_minus_7pi_over_6_l416_416427


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416934

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416934


namespace product_of_consecutive_integers_l416_416759

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416759


namespace cosine_product_l416_416226

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l416_416226


namespace four_consecutive_integers_divisible_by_12_l416_416763

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416763


namespace angles_B1OB2_eq_C1OC2_l416_416562

-- Defining the main theorem
theorem angles_B1OB2_eq_C1OC2 (A M K : Point) (B1 B2 C1 C2 O : Point)
  (hB1_on_AM : B1 ∈ line_segment A M)
  (hB2_on_AM : B2 ∈ line_segment A M)
  (hC1_on_AK : C1 ∈ line_segment A K)
  (hC2_on_AK : C2 ∈ line_segment A K)
  (hO_incribed_ABC1 : is_incenter O (triangle A B1 C1))
  (hO_incribed_ABC2 : is_incenter O (triangle A B2 C2)) :
  ∠ B1 O B2 = ∠ C1 O C2 := 
by
  -- Proof omitted; this theorem statement matches the given proof problem
  sorry

end angles_B1OB2_eq_C1OC2_l416_416562


namespace domain_of_h_l416_416641

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_of_h :
  {x : ℝ | 2 * x - 10 ≠ 0} = {x : ℝ | x ≠ 5} :=
by
  sorry

end domain_of_h_l416_416641


namespace hexagon_probability_l416_416506

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l416_416506


namespace arithmetic_sequence_geometric_relation_l416_416378

theorem arithmetic_sequence_geometric_relation (a_n : ℕ → ℤ)
  (h1 : ∀ n, a_n (n + 1) = a_n n + 2)
  (h2 : ∃ k, a_n 1 * a_n 4 = (a_n 3) * (a_n 3)) :
  a_n 2 = -6 :=
by
  sorry

end arithmetic_sequence_geometric_relation_l416_416378


namespace smallest_non_palindrome_power_of_12_l416_416347

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ¬is_palindrome (12^n) ∧
  ∀ m : ℕ, m > 0 → ¬is_palindrome (12^m) → 12^m ≥ 12^n :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num,
    -- Show that 144 is not a palindrome
    sorry },
  { intros m hm hnp,
    -- Show that for any other power of 12 that is not a palindrome, the result is ≥ 144
    -- essentially proving that 12^2 is the smallest such number.
    sorry }
end

end smallest_non_palindrome_power_of_12_l416_416347


namespace plane_speed_in_still_air_l416_416259

-- Given conditions
variables (p : ℝ) (wind_speed : ℝ) (dist_with_wind : ℝ) (dist_against_wind : ℝ)
variables (same_time : (dist_with_wind / (p + wind_speed)) = (dist_against_wind / (p - wind_speed)))

-- Specific values for the problem
def wind_speed_val : ℝ := 23
def dist_with_wind_val : ℝ := 420
def dist_against_wind_val : ℝ := 350

-- The proof goal
theorem plane_speed_in_still_air : p = 253 :=
begin
  sorry
end

end plane_speed_in_still_air_l416_416259


namespace triangle_angle_l416_416460

theorem triangle_angle (a b c : ℝ) (h : a^2 = b^2 + c^2 + b * c) :
  ∃ (A : ℝ), A = 2 * real.pi / 3 ∧ real.cos A = (b^2 + c^2 - a^2) / (2 * b * c) :=
begin
  use 2 * real.pi / 3,
  split,
  { refl },
  { rw h,
    norm_num }
end

end triangle_angle_l416_416460


namespace train_crossing_man_time_l416_416270

theorem train_crossing_man_time (v_kmph : ℕ) (t1 : ℕ) (d_p : ℕ) (t2 : ℕ) :
  v_kmph = 72 → t1 = 30 → d_p = 280 → t2 = 16 :=
by
  assume h1 h2 h3
  -- Convert km/h to m/s
  let v_mps := v_kmph * 1000 / 3600
  have v_equiv : v_mps = 20 := by
    rw [h1]
    norm_num
  -- Calculate the length of the train (L)
  let total_distance := v_mps * t1
  have distance_equiv : total_distance = 600 := by
    rw [v_equiv, h2]
    norm_num
  let L := total_distance - d_p
  have length_train : L = 320 := by
    rw [distance_equiv, h3]
    norm_num
  -- Calculate the time to cross a man
  let time_to_cross_man := L / v_mps
  have time_equiv : time_to_cross_man = t2 := by
    rw [length_train, v_equiv]
    norm_num
  exact time_equiv

end train_crossing_man_time_l416_416270


namespace total_yellow_balloons_l416_416131

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end total_yellow_balloons_l416_416131


namespace swimming_pool_cost_l416_416121

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l416_416121


namespace greatest_divisor_four_consecutive_integers_l416_416840

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416840


namespace divisor_of_four_consecutive_integers_l416_416642

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416642


namespace greatest_divisor_of_consecutive_product_l416_416855

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416855


namespace mean_of_set_median_is_128_l416_416098

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l416_416098


namespace divisor_of_four_consecutive_integers_l416_416652

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416652


namespace divisor_of_product_of_four_consecutive_integers_l416_416714

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416714


namespace divisor_of_four_consecutive_integers_l416_416651

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416651


namespace vasya_numbers_l416_416153

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416153


namespace James_watch_time_l416_416489

def Jeopardy_length : ℕ := 20
def Wheel_of_Fortune_length : ℕ := Jeopardy_length * 2
def Jeopardy_episodes : ℕ := 2
def Wheel_of_Fortune_episodes : ℕ := 2

theorem James_watch_time :
  (Jeopardy_episodes * Jeopardy_length + Wheel_of_Fortune_episodes * Wheel_of_Fortune_length) / 60 = 2 :=
by
  sorry

end James_watch_time_l416_416489


namespace total_team_cost_correct_l416_416624

variable (jerseyCost shortsCost socksCost cleatsCost waterBottleCost : ℝ)
variable (numPlayers : ℕ)
variable (discountThreshold discountRate salesTaxRate : ℝ)

noncomputable def totalTeamCost : ℝ :=
  let totalCostPerPlayer := jerseyCost + shortsCost + socksCost + cleatsCost + waterBottleCost
  let totalCost := totalCostPerPlayer * numPlayers
  let discount := if totalCost > discountThreshold then totalCost * discountRate else 0
  let discountedTotal := totalCost - discount
  let tax := discountedTotal * salesTaxRate
  let finalCost := discountedTotal + tax
  finalCost

theorem total_team_cost_correct :
  totalTeamCost 25 15.20 6.80 40 12 25 500 0.10 0.07 = 2383.43 := by
  sorry

end total_team_cost_correct_l416_416624


namespace greatest_divisor_four_consecutive_l416_416696

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416696


namespace greatest_divisor_four_consecutive_l416_416687

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416687


namespace four_consecutive_integers_divisible_by_12_l416_416773

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416773


namespace JessieScore_l416_416556

-- Define the conditions as hypotheses
variables (correct_answers : ℕ) (incorrect_answers : ℕ) (unanswered_questions : ℕ)
variables (points_per_correct : ℕ) (points_deducted_per_incorrect : ℤ)

-- Define the values for the specific problem instance
def JessieCondition := correct_answers = 16 ∧ incorrect_answers = 4 ∧ unanswered_questions = 10 ∧
                       points_per_correct = 2 ∧ points_deducted_per_incorrect = -1 / 2

-- Define the statement that Jessie's score is 30 given the conditions
theorem JessieScore (h : JessieCondition correct_answers incorrect_answers unanswered_questions points_per_correct points_deducted_per_incorrect) :
  (correct_answers * points_per_correct : ℤ) + (incorrect_answers * points_deducted_per_incorrect) = 30 :=
by
  sorry

end JessieScore_l416_416556


namespace four_consecutive_integers_divisible_by_12_l416_416774

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416774


namespace chime_time_12_oclock_l416_416249

theorem chime_time_12_oclock (h : ∀ (n : Nat), n = 2 → time_to_chime n = 2 → 1 second per interval ) :
  time_to_chime 12 = 12 :=
by
  sorry

end chime_time_12_oclock_l416_416249


namespace AE_length_l416_416297

theorem AE_length (AB CD AC AE : ℝ) (hAB : AB = 15) (hCD : CD = 20) (hAC : AC = 18) 
  (hEqualAreas : ∃ E : ℝ,  (triangle_area AED = triangle_area BEC)) : 
  AE = 54 / 7 :=
by
  -- The proof goes here.
  sorry

end AE_length_l416_416297


namespace four_consecutive_product_divisible_by_12_l416_416998

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416998


namespace problem_equivalent_condition_l416_416376

theorem problem_equivalent_condition (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) :
  a ≠ 0 → a ^ 2004 + b ^ 2005 = 1 := by
  sorry

end problem_equivalent_condition_l416_416376


namespace exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l416_416009

-- Part (a): Proving the existence of such an arithmetic sequence with 2003 terms.
theorem exists_arithmetic_seq_2003_terms_perfect_powers :
  ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, n ≤ 2002 → ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

-- Part (b): Proving the non-existence of such an infinite arithmetic sequence.
theorem no_infinite_arithmetic_seq_perfect_powers :
  ¬ ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

end exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l416_416009


namespace incorrect_propositions_l416_416575

theorem incorrect_propositions : 
  (∀ x, x ≤ 0 → ∃ y, y = 2^x ∧ y ≤ 1) = false ∧
  (∀ x, x > 2 → ∃ y, y = 1/x ∧ y ≤ 1/2) = false ∧
  (∀ y, 0 ≤ y ∧ y ≤ 4 → ∃ x, x^2 = y ∧ -2 ≤ x ∧ x ≤ 2) = false ∧
  (∀ y, y ≤ 3 → ∃ x, log 2 x = y ∧ 0 < x ∧ x ≤ 8) = true :=
by sorry

end incorrect_propositions_l416_416575


namespace equal_segments_of_isosceles_triangle_l416_416635

theorem equal_segments_of_isosceles_triangle 
  {A B C M P Q : Point} (is_isosceles : is_isosceles_triangle A B C)
  (M_on_AB : on_points_segment A B M)
  (secant_intersects : secant_intersects_points C M P Q)
  (MP_eq_MQ : dist M P = dist M Q) :
  dist A P = dist B Q :=
sorry

end equal_segments_of_isosceles_triangle_l416_416635


namespace hexagon_same_length_probability_l416_416521

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l416_416521


namespace weight_of_closest_crate_total_weight_of_crates_l416_416115

noncomputable def numberOfCrates : ℕ := 8
noncomputable def standardWeight : ℝ := 25
noncomputable def weightDeviations : List ℝ := [1.5, -3, 2, -0.5, 1, -2, -2, -2.5]

theorem weight_of_closest_crate :
  let closestDeviation := (weightDeviations.map (λ x => standardWeight + x)).min (by decide)
  closestDeviation = 24.5 :=
by
  sorry

theorem total_weight_of_crates :
  let totalWeight := standardWeight * numberOfCrates + weightDeviations.sum
  totalWeight = 194.5 :=
by
  sorry

end weight_of_closest_crate_total_weight_of_crates_l416_416115


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416862

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416862


namespace inequality_integer_solution_l416_416385

theorem inequality_integer_solution (a b : ℝ) (h1 : 0 < b) (h2 : b < 1 + a) :
  (∃ S : set ℤ, {x : ℝ | (x - b)^2 > (a * x)^2}.indicator (λ x, 1) '' (S : set ℝ) = {1} ∧ S.card = 3) →
  1 < a ∧ a < 3 := by
  sorry

end inequality_integer_solution_l416_416385


namespace non_union_women_percentage_l416_416216

theorem non_union_women_percentage 
    (total_employees : ℕ)
    (men_percent : ℝ)
    (union_percent : ℝ)
    (union_men_percent : ℝ)
    (h1 : total_employees = 100)
    (h2 : men_percent = 0.56)
    (h3 : union_percent = 0.60)
    (h4 : union_men_percent = 0.70) :
  let men := men_percent * total_employees,
      women := total_employees - men,
      union := union_percent * total_employees,
      non_union := total_employees - union,
      union_men := union_men_percent * union,
      union_women := union - union_men,
      non_union_men := men - union_men,
      non_union_women := women - union_women in
  (non_union_women / non_union * 100) = 65 := 
by 
  -- This is where the proof would go
  sorry

end non_union_women_percentage_l416_416216


namespace greatest_integer_less_than_neg_21_over_5_l416_416195

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ n : ℤ, n < -21 / 5 ∧ ∀ m : ℤ, m < -21 / 5 → m ≤ n :=
begin
  use -5,
  split,
  { linarith },
  { intros m h,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416195


namespace probability_of_same_length_segments_l416_416502

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l416_416502


namespace pentagon_sum_l416_416237

noncomputable section

def regular_pentagon (P : ℝ → ℝ → Prop) : Prop := 
∀ (x y : ℝ) (A B C D E : ℝ), P A B ∧ P B C ∧ P C D ∧ P D E ∧ P E A

def perpendicular_segment (p q r : ℝ) : Prop := 
∀ (A P Q R : ℝ), p = P ∧ q = Q ∧ r = R

def center_of_pentagon (O : ℝ) : Prop := 
∃ (P O : ℝ), O = P / 2

def given_conditions (O : ℝ) (P : ℝ → ℝ → Prop) (AP AQ AR : ℝ) : Prop := 
regular_pentagon P ∧ perpendicular_segment AP AQ AR ∧ center_of_pentagon O ∧ OP = 1

theorem pentagon_sum (O AP AQ AR : ℝ) (P : ℝ → ℝ → Prop) : 
given_conditions O P AP AQ AR → AO + AQ + AR = 4 := 
by
  sorry

end pentagon_sum_l416_416237


namespace balls_in_boxes_count_l416_416627

theorem balls_in_boxes_count :
  let balls := {1, 2, 3, 4, 5},
      boxes := {1, 2, 3, 4, 5},
      arrangements := finset.permutations (finset.univ 5) in
  (∃ arrangement ∈ arrangements, at_most_two_fixed_points arrangement) → 
  arrangements.card - fixed_points_count arrangements - three_fixed_points_count arrangements = 109 := 
sorry

end balls_in_boxes_count_l416_416627


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416870

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416870


namespace centroid_tetrahedra_volume_fraction_l416_416639

noncomputable def volume_of_tetrahedron (v1 v2 v3 v4 : ℝ³) : ℝ := 
  (1 / 6) * abs ((v2 - v1) × (v3 - v1)).dot (v4 - v1)

theorem centroid_tetrahedra_volume_fraction (V : ℝ) (centroids : List (ℝ³)) (parallelepiped_volume : ℝ) :
  (List.length centroids = 4) →
  (parallelepiped_volume = V) →
  (∀ (c1 c2 c3 c4 : ℝ³), (c1 ∈ centroids) ∧ (c2 ∈ centroids) ∧ (c3 ∈ centroids) ∧ (c4 ∈ centroids) →
  (volume_of_tetrahedron c1 c2 c3 c4 = (V / 24))) := 
sorry

end centroid_tetrahedra_volume_fraction_l416_416639


namespace div_product_four_consecutive_integers_l416_416790

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416790


namespace smallest_non_palindrome_power_of_12_l416_416325

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 12 ^ k ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, m > 0 ∧ (∃ j : ℕ, m = 12 ^ j ∧ ¬ is_palindrome m) → m ≥ n) :=
by
  sorry

end smallest_non_palindrome_power_of_12_l416_416325


namespace box_dimensions_l416_416486

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) : 
  a = 5 ∧ b = 8 ∧ c = 12 := 
by
  sorry

end box_dimensions_l416_416486


namespace smallest_positive_non_palindromic_power_of_12_is_12_l416_416337

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def smallest_non_palindromic_power_of_12 : ℕ :=
  Nat.find (λ n, ∃ k : ℕ, n = 12^k ∧ ¬is_palindrome n)

theorem smallest_positive_non_palindromic_power_of_12_is_12 :
  smallest_non_palindromic_power_of_12 = 12 :=
by
  sorry

end smallest_positive_non_palindromic_power_of_12_is_12_l416_416337


namespace binary_to_decimal_10111_l416_416296

theorem binary_to_decimal_10111 : Nat.bitwise 1 (Nat.bitwise 0 (Nat.bitwise 1 (Nat.bitwise 1 (Nat.bitwise 1 0)))) = 23 := by
  sorry -- We are skipping the proof as per the instructions.

end binary_to_decimal_10111_l416_416296


namespace translation_possible_l416_416558

open Set Metric

variables {n : ℕ} 
variables (A : Fin n → ℝ × ℝ) 
variables (Φ : Set (ℝ × ℝ))

def pairwise_distances_greater_than_2 :=
  ∀ i j : Fin n, i ≠ j → dist (A i) (A j) > 2

def area_less_than_pi (s : Set (ℝ × ℝ)) :=
  measure_theory.Measure.volume s < real.pi

axiom distance_le_1 (v : ℝ × ℝ) : dist (0, 0) v ≤ 1

theorem translation_possible (h_dist : pairwise_distances_greater_than_2 A)
    (h_area : area_less_than_pi Φ) :
  ∃ v : ℝ × ℝ, dist (0, 0) v ≤ 1 ∧ ∀ i : Fin n, (A i) ∉ (image (λ p, (p.1 + v.1, p.2 + v.2)) Φ) :=
sorry

end translation_possible_l416_416558


namespace greatest_divisor_four_consecutive_integers_l416_416823

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416823


namespace socks_ratio_l416_416547

theorem socks_ratio 
  (g : ℕ) -- number of pairs of green socks
  (y : ℝ) -- price per pair of green socks
  (h1 : y > 0) -- price per pair of green socks is positive
  (h2 : 3 * g * y + 3 * y = 1.2 * (9 * y + g * y)) -- swapping resulted in a 20% increase in the bill
  : 3 / g = 3 / 4 :=
by sorry

end socks_ratio_l416_416547


namespace project_completion_l416_416105

/-- Definitions for input conditions --/
def days_to_complete (x y : ℕ) (both_cost single_cost : ℤ) (A_days B_days : ℕ) (A_cost B_cost : ℤ) : Prop :=
  let eq1 := (24 : ℝ) / (x : ℝ) + (24 : ℝ) / (y : ℝ) = 1
  let eq2 := (18 : ℝ) / (x : ℝ) + (48 : ℝ) / (y : ℝ) = 1
  let cost1 := (24 : ℝ) * (m : ℝ) + (24 : ℝ) * (n : ℝ) = (1.2 : ℝ)
  let cost2 := (18 : ℝ) * (m : ℝ) + (48 : ℝ) * (n : ℝ) = (1.1 : ℝ)
  (eq1 ∧ eq2 ∧ cost1 ∧ cost2)

/-- Conditions of the problem --/
def conditions (x y : ℕ) (m n : ℚ) :
  (24 / (x : ℚ) + 24 / (y : ℚ) = 1) ∧ (18 / (x : ℚ) + 48 / (y : ℚ) = 1) :=
by sorry

/-- Questions and correct answers --/
theorem project_completion (x y : ℕ) (m n : ℚ) :
  conditions x y m n → (x = 30 ∧ y = 120) ∧ (m = 13 / 3 ∧ n = 2 / 3) :=
by sorry

end project_completion_l416_416105


namespace square_free_odd_integers_l416_416302

open Nat

-- A helper function to determine if an integer n is square-free
def is_square_free (n : ℕ) : Prop :=
  ∀ d : ℕ, d * d ∣ n → d = 1

noncomputable def count_square_free_odds (a b : ℕ) : ℕ :=
  ∑ i in Finset.filter (λ x, is_odd x ∧ x > 1 ∧ x < 200 ∧ is_square_free x) (Finset.range (b+1)), 1

theorem square_free_odd_integers : count_square_free_odds 1 199 = 80 := 
by sorry

end square_free_odd_integers_l416_416302


namespace vasya_numbers_l416_416168

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416168


namespace greatest_divisor_of_four_consecutive_integers_l416_416808

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416808


namespace divisor_of_four_consecutive_integers_l416_416653

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416653


namespace greatest_divisor_four_consecutive_integers_l416_416834

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416834


namespace sale_in_fifth_month_l416_416252

-- Define the sales in the first, second, third, fourth, and sixth months
def a1 : ℕ := 7435
def a2 : ℕ := 7927
def a3 : ℕ := 7855
def a4 : ℕ := 8230
def a6 : ℕ := 5991

-- Define the average sale
def avg_sale : ℕ := 7500

-- Define the number of months
def months : ℕ := 6

-- The total sales required for the average sale to be 7500 over 6 months.
def total_sales : ℕ := avg_sale * months

-- Calculate the sales in the first four months
def sales_first_four_months : ℕ := a1 + a2 + a3 + a4

-- Calculate the total sales for the first four months plus the sixth month.
def sales_first_four_and_sixth : ℕ := sales_first_four_months + a6

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : ∃ a5 : ℕ, total_sales = sales_first_four_and_sixth + a5 ∧ a5 = 7562 :=
by
  sorry


end sale_in_fifth_month_l416_416252


namespace divisor_of_four_consecutive_integers_l416_416649

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416649


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416738

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416738


namespace vasya_numbers_l416_416165

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416165


namespace smallest_power_of_12_not_palindrome_l416_416332

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def power_of_12_not_palindrome (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 12^k ∧ ¬ is_palindrome (n.repr)

theorem smallest_power_of_12_not_palindrome : ∃ n : ℕ, power_of_12_not_palindrome n ∧ ∀ m : ℕ, power_of_12_not_palindrome m → n ≤ m :=
sorry

end smallest_power_of_12_not_palindrome_l416_416332


namespace number_of_planting_methods_l416_416361

theorem number_of_planting_methods :
  let vegetables := ["cucumbers", "cabbages", "rape", "flat beans"]
  let plots := ["plot1", "plot2", "plot3"]
  (∀ v ∈ vegetables, v = "cucumbers") →
  (∃! n : ℕ, n = 18)
:= by
  sorry

end number_of_planting_methods_l416_416361


namespace divisor_of_product_of_four_consecutive_integers_l416_416702

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416702


namespace point_not_in_S_for_t_l416_416528

def greatest_integer (t : ℝ) : ℝ := floor t

noncomputable def point_in_S (t : ℝ) (x y : ℝ) : Prop :=
  let T := t - greatest_integer t
  (x - T)^2 + (y - T)^2 ≤ T^2

theorem point_not_in_S_for_t (t : ℝ) (x y : ℝ) (h : t = 3.5) : ¬ point_in_S t x y :=
  by
  sorry
  
example : ¬ point_in_S 3.5 1 1 :=
  point_not_in_S_for_t 3.5 1 1 rfl

end point_not_in_S_for_t_l416_416528


namespace probability_of_same_length_segments_l416_416504

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l416_416504


namespace f_eq_g_iff_l416_416535

noncomputable def f (m n x : ℝ) := m * x^2 + n * x
noncomputable def g (p q x : ℝ) := p * x + q

theorem f_eq_g_iff (m n p q : ℝ) :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n := by
  sorry

end f_eq_g_iff_l416_416535


namespace greatest_divisor_four_consecutive_l416_416684

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416684


namespace greatest_divisor_of_four_consecutive_integers_l416_416976

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416976


namespace greatest_possible_value_x_y_l416_416264

noncomputable def max_x_y : ℕ :=
  let s1 := 150
  let s2 := 210
  let s3 := 270
  let s4 := 330
  (3 * (s3 + s4) - (s1 + s2 + s3 + s4))

theorem greatest_possible_value_x_y :
  max_x_y = 840 := by
  sorry

end greatest_possible_value_x_y_l416_416264


namespace vasya_numbers_l416_416166

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416166


namespace divisor_of_four_consecutive_integers_l416_416645

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416645


namespace mean_is_12_point_8_l416_416095

variable (m : ℝ)
variable median_condition : m + 7 = 12

theorem mean_is_12_point_8 (m : ℝ) (median_condition : m + 7 = 12) : 
(mean := (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5) = 64 / 5 :=
by {
  sorry
}

end mean_is_12_point_8_l416_416095


namespace AE_eq_EC_l416_416386

variable (A B C D E F : Type)
variable [Metric (A B C D E F)]
variable [TriangleABC : IsIsoscelesRightTriangle A B C (λ (BC : Segment A C))]
variable [PointDOnBC : ∃ D, OnSegment D ('segment AB) ∧ (length (segment DB) = (1/3) * (length (segment AB)))]
variable [LineBEPerpAD : ∃ E, Perpendicular (line BE) (line AD) ∧ IntersectAt (line BE) (segment AC) E]

theorem AE_eq_EC :
  ∀ (A B C D E : Point), 
  IsIsoscelesRightTriangle A B C ∧ 
  PointOnSegment D B C ∧ length (Segment D C) = (1 / 3) * length (Segment B C) ∧
  Perpendicular (line B E) (line A D) ∧ PointOnSegment E A C := 
  AE = EC := 
sorry

end AE_eq_EC_l416_416386


namespace greatest_divisor_of_four_consecutive_integers_l416_416809

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416809


namespace four_consecutive_product_divisible_by_12_l416_416988

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416988


namespace sqrt_sum_inequality_l416_416413

open Real

theorem sqrt_sum_inequality (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ sqrt 21 :=
sorry

end sqrt_sum_inequality_l416_416413


namespace smallest_non_palindrome_power_of_12_l416_416327

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 12 ^ k ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, m > 0 ∧ (∃ j : ℕ, m = 12 ^ j ∧ ¬ is_palindrome m) → m ≥ n) :=
by
  sorry

end smallest_non_palindrome_power_of_12_l416_416327


namespace digit_sum_2007_is_square_l416_416602

def digit_sum (n : Nat) : Nat :=
  n.digits.sum

theorem digit_sum_2007_is_square :
  digit_sum 2007 = 9 ∧ ∃ k : Nat, k^2 = digit_sum 2007 :=
by
  sorry

end digit_sum_2007_is_square_l416_416602


namespace angle_between_a_b_l416_416419

-- Given unit vectors e1 and e2
variables (e1 e2 : ℝ^3)
-- e1 and e2 are unit vectors
axiom unit_e1 : ∥e1∥ = 1
axiom unit_e2 : ∥e2∥ = 1

-- e1 and e2 are at 60 degrees
axiom angle_60 : real.angle e1 e2 = real.angle.of_deg 60

-- Define vectors a and b
def a := 2 • e1 + e2
def b := -3 • e1 + 2 • e2

-- State the theorem to prove the angle between a and b is 120 degrees
theorem angle_between_a_b : real.angle a b = real.angle.of_deg 120 :=
sorry

end angle_between_a_b_l416_416419


namespace find_area_BPM_l416_416475

open_locale big_operators
open_locale classical

noncomputable theory

structure Point :=
  (x y : ℝ)

def square (A B C D : Point) (side : ℝ) : Prop :=
  A.x = 0 ∧ A.y = 0 ∧
  B.x = 0 ∧ B.y = side ∧
  C.x = side ∧ C.y = side ∧
  D.x = side ∧ D.y = 0

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def proportion_point (A D : Point) (k : ℝ) : Point :=
  ⟨k * D.x / (k + 1), k * D.y / (k + 1)⟩

def line_intersection (P Q R S : Point) : Point :=
  let a1 := Q.y - P.y,
      b1 := P.x - Q.x,
      c1 := a1 * P.x + b1 * P.y,
      a2 := S.y - R.y,
      b2 := R.x - S.x,
      c2 := a2 * R.x + b2 * R.y,
      det := a1 * b2 - a2 * b1,
      x := (b2 * c1 - b1 * c2) / det,
      y := (a1 * c2 - a2 * c1) / det
  in ⟨x, y⟩

def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * |A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)|

theorem find_area_BPM :
  ∀ (A B C D M N P : Point),
    square A B C D 2 →
    M = midpoint A B →
    N = proportion_point A D 2 →
    P = line_intersection M N A C →
    triangle_area B P M = 2 / 7 :=
by sorry

end find_area_BPM_l416_416475


namespace four_consecutive_product_divisible_by_12_l416_416985

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416985


namespace envelope_total_extra_postage_l416_416596

def length_to_height_ratio (length height : ℝ) : ℝ :=
  length / height

def extra_charge (ratio : ℝ) : ℝ :=
  if ratio < 1.5 ∨ ratio > 3.0 then 0.15 else 0

def envelope_A_length : ℝ := 7
def envelope_A_height : ℝ := 6

def envelope_B_length : ℝ := 8
def envelope_B_height : ℝ := 2

def envelope_C_length : ℝ := 7
def envelope_C_height : ℝ := 7

def envelope_D_length : ℝ := 13
def envelope_D_height : ℝ := 4

def total_extra_postage : ℝ :=
  extra_charge (length_to_height_ratio envelope_A_length envelope_A_height) +
  extra_charge (length_to_height_ratio envelope_B_length envelope_B_height) +
  extra_charge (length_to_height_ratio envelope_C_length envelope_C_height) +
  extra_charge (length_to_height_ratio envelope_D_length envelope_D_height)

theorem envelope_total_extra_postage : total_extra_postage = 0.6 :=
  sorry

end envelope_total_extra_postage_l416_416596


namespace find_a_l416_416426

theorem find_a (a : ℝ) (h : 3 ∈ {a + 3, 2 * a + 1, a^2 + a + 1}) : a = -2 :=
sorry

end find_a_l416_416426


namespace number_of_elements_in_S_l416_416039

noncomputable def f (x : ℝ) : ℝ := (x + 8) / x

def seq_f (n : ℕ) : ℝ → ℝ
| 1 => f
| n + 1 => f ∘ seq_f n

def S : set ℝ := {x : ℝ | ∃ n : ℕ, f (seq_f n x) = x}

theorem number_of_elements_in_S : (S.size = 2) :=
sorry

end number_of_elements_in_S_l416_416039


namespace divisor_of_product_of_four_consecutive_integers_l416_416710

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416710


namespace whole_numbers_divisible_by_3_or_5_l416_416425

theorem whole_numbers_divisible_by_3_or_5 (n : ℕ) (h : n = 46) : 
  (nat.divisible_count 3 n + nat.divisible_count 5 n - nat.divisible_count 15 n) = 21 :=
by
  about n
  replace h : n = 46 := sorry
  have h3_divisibles : count_divisibles 3 46 = 15 := sorry
  have h5_divisibles : count_divisibles 5 46 = 9 := sorry
  have h15_divisibles : count_divisibles 15 46 = 3 := sorry
  calc
    (count_divisibles 3 n + count_divisibles 5 n - count_divisibles 15 n)
        = 15 + 9 - 3           : by rw [h3_divisibles, h5_divisibles, h15_divisibles]
        ... = 21              : by norm_num
  done

end whole_numbers_divisible_by_3_or_5_l416_416425


namespace range_of_p_l416_416456

noncomputable def f (x p : ℝ) : ℝ := x - p/x + p/2

theorem range_of_p (p : ℝ) :
  (∀ x : ℝ, 1 < x → (1 + p / x^2) > 0) → p ≥ -1 :=
by
  intro h
  sorry

end range_of_p_l416_416456


namespace max_x_add_2y_l416_416034

theorem max_x_add_2y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + 2 * y ≤ 4 :=
sorry

end max_x_add_2y_l416_416034


namespace sum_of_reciprocal_products_l416_416276

theorem sum_of_reciprocal_products (n : ℕ) (h : n > 0) :
  ∑ s in (finset.powerset (finset.range n.succ)).filter (λ s, s.nonempty),
    (∏ x in s, (x : ℕ) + 1)⁻¹ = n := sorry

end sum_of_reciprocal_products_l416_416276


namespace chocolate_distribution_l416_416236

theorem chocolate_distribution (n : ℕ) 
  (h1 : 12 * 2 ≤ n * 2 ∨ n * 2 ≤ 12 * 2) 
  (h2 : ∃ d : ℚ, (12 / n) = d ∧ d * n = 12) : 
  n = 15 :=
by 
  sorry

end chocolate_distribution_l416_416236


namespace smallest_positive_non_palindrome_power_of_12_l416_416343

def is_palindrome (n : ℕ) : Bool :=
  let s := toDigits 10 n
  s = s.reverse

theorem smallest_positive_non_palindrome_power_of_12 : ∃ k : ℕ, k > 0 ∧ (12^k = 12 ∧ ¬ is_palindrome (12^k)) :=
by {
  sorry
}

end smallest_positive_non_palindrome_power_of_12_l416_416343


namespace complex_div_conjugate_l416_416439

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416439


namespace mean_of_set_is_12_point_8_l416_416091

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l416_416091


namespace regular_polygon_on_lines_l416_416359

theorem regular_polygon_on_lines (n : ℕ) (h_regular : is_regular_polygon n) (h_lines : are_parallel_and_evenly_spaced_lines) :
  n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end regular_polygon_on_lines_l416_416359


namespace range_alpha_beta_l416_416394

variables {A D E G P B C : Type} [simp_ge A D E G P B C]

-- Conditions
def is_centroid (G : Type) (ADE : Triangle A D E) : Prop := centroid ADE = G
def is_trisection_point (B C : Type) (AD AE : Segment A D A E) : Prop := 
  B = (2/3) • A + (1/3) • D ∧ C = (2/3) • A + (1/3) • E
def point_in_triangle (P : Type) (DEG : Triangle D E G) : Prop := P ∈ DEG

-- Translate vector condition
def vector_condition (α β : ℝ) (AB AC : Vector A B A C) (AP : Vector A P) : Prop :=
  AP = α • AB + β • AC

-- Lean statement
theorem range_alpha_beta (G : Type) (ADE : Triangle A D E) (DEG : Triangle D E G) (AB AC : Vector A B A C) :
  is_centroid G ADE →
  is_trisection_point B C AD AE →
  point_in_triangle P DEG →
  ∃ (α β : ℝ), vector_condition α β AB AC →
  ∀ (α β : ℝ), (α + 1/2 * β) ∈ (3/2..3) :=
begin
  sorry
end

end range_alpha_beta_l416_416394


namespace evaluate_expression_l416_416445

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416445


namespace smallest_power_of_12_not_palindrome_l416_416330

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def power_of_12_not_palindrome (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 12^k ∧ ¬ is_palindrome (n.repr)

theorem smallest_power_of_12_not_palindrome : ∃ n : ℕ, power_of_12_not_palindrome n ∧ ∀ m : ℕ, power_of_12_not_palindrome m → n ≤ m :=
sorry

end smallest_power_of_12_not_palindrome_l416_416330


namespace div_product_four_consecutive_integers_l416_416786

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416786


namespace max_quad_int_solutions_l416_416352

theorem max_quad_int_solutions :
  ∃ (a b c : ℤ), (∀ n : ℤ, n ∈ {0, 1, 2, -3, 4}) ∧
  ∀ (p : ℤ → ℤ), 
    p(x) = ax^2 + bx + c →
      ∃ n, p(n) = p(n^2) :=
begin
  sorry
end

end max_quad_int_solutions_l416_416352


namespace mold_growth_problem_l416_416581

/-- Given the conditions:
    - Initial mold spores: 50 at 9:00 a.m.
    - Colony doubles in size every 10 minutes.
    - Time elapsed: 70 minutes from 9:00 a.m. to 10:10 a.m.,

    Prove that the number of mold spores at 10:10 a.m. is 6400 -/
theorem mold_growth_problem : 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  final_population = 6400 :=
by 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  sorry

end mold_growth_problem_l416_416581


namespace AK_eq_DC_l416_416608

variables {A B C D M K : Point}

def incircle_touches_AC_at_D (triangle : Triangle ABC) (D : Point) : Prop :=
  touches_incircle triangle AC D

def diameter_DM (D M : Point) : Prop :=
  diameter_incircle D M

def intersects_BM_at_K (B M K : Point) (AC : Segment) : Prop :=
  intersects_line B M K AC

theorem AK_eq_DC
  (h1 : incircle_touches_AC_at_D ABC D)
  (h2 : diameter_DM D M)
  (h3 : intersects_BM_at_K B M K AC) : 
  length_segment (A, K) = length_segment (D, C) :=
sorry  -- proof to be provided

end AK_eq_DC_l416_416608


namespace div_product_four_consecutive_integers_l416_416793

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416793


namespace negation_and_nonP_l416_416064

-- Define the proposition P
def P : Prop := ∀ (Q : Type) [IsQuadrilateral Q], InscribedInCircle Q → SupplementaryOppositeAngles Q

-- State the negation of P
def notP : Prop := ¬P

-- State non-P directly as it describes an alternate definition
def nonP : Prop := ∃ (Q : Type) [IsQuadrilateral Q], InscribedInCircle Q ∧ ¬SupplementaryOppositeAngles Q

-- The main theorem we want to prove
theorem negation_and_nonP : P → notP ∧ nonP :=
by
  intro hP
  have hNotP : ¬P := by {sorry}
  have hNonP : nonP := by {sorry}
  exact ⟨hNotP, hNonP⟩

end negation_and_nonP_l416_416064


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416736

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416736


namespace floor_sum_value_l416_416031

theorem floor_sum_value (a b c d : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
(h1 : a^2 + b^2 = 2016) (h2 : c^2 + d^2 = 2016) (h3 : a * c = 1024) (h4 : b * d = 1024) :
  ⌊a + b + c + d⌋ = 127 := sorry

end floor_sum_value_l416_416031


namespace angle_BCM_is_10_degrees_l416_416080

theorem angle_BCM_is_10_degrees
  (A B C M : Type) 
  [triangle A B C]
  (isosceles : AB = AC)
  (angle_A : angle B A C = 100) 
  (AM_eq_BC : dist A M = dist B C) :
  angle B C M = 10 :=
by
  sorry

end angle_BCM_is_10_degrees_l416_416080


namespace jason_borrowed_amount_l416_416491

def earning_per_six_hours : ℤ :=
  2 + 4 + 6 + 2 + 4 + 6

def total_hours_worked : ℤ :=
  48

def cycle_length : ℤ :=
  6

def total_cycles : ℤ :=
  total_hours_worked / cycle_length

def total_amount_borrowed : ℤ :=
  total_cycles * earning_per_six_hours

theorem jason_borrowed_amount : total_amount_borrowed = 192 :=
  by
    -- Here we use the definition and conditions to prove the equivalence
    -- of the calculation to the problem statement.
    sorry

end jason_borrowed_amount_l416_416491


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416920

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416920


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416922

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416922


namespace prob_four_vertices_form_tetrahedron_l416_416284

theorem prob_four_vertices_form_tetrahedron :
  let total_ways := Nat.choose 8 4
  let non_tetrahedron_ways := 12
  total_ways > 0 → 
  (1 - (non_tetrahedron_ways / total_ways) = 29/35) :=
by
   intros
   unfold total_ways non_tetrahedron_ways
   sorry

end prob_four_vertices_form_tetrahedron_l416_416284


namespace greatest_divisor_four_consecutive_l416_416682

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416682


namespace diameter_of_larger_sphere_l416_416601

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end diameter_of_larger_sphere_l416_416601


namespace four_consecutive_integers_divisible_by_12_l416_416771

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416771


namespace divisor_of_four_consecutive_integers_l416_416658

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416658


namespace hexagon_same_length_probability_l416_416525

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l416_416525


namespace total_balloons_l416_416133

-- Defining constants for the number of balloons each person has
def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

-- Theorem stating the total number of balloons
theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  simp [tom_balloons, sara_balloons]
  sorry

end total_balloons_l416_416133


namespace smallest_three_digit_number_l416_416023

theorem smallest_three_digit_number (n : ℕ) : 
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 10 = 3) ∧ (n % 7 = 3) → n = 143 :=
by
  intros h,
  sorry

end smallest_three_digit_number_l416_416023


namespace cost_to_fill_pool_l416_416127

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l416_416127


namespace marcias_hair_length_l416_416548

theorem marcias_hair_length (initial_length: ℕ) (first_cut_fraction: ℕ) (growth: ℕ) (second_cut: ℕ) :
  initial_length = 24 → first_cut_fraction = 2 → growth = 4 → second_cut = 2 →
  (initial_length / first_cut_fraction - second_cut + growth - second_cut) = 14 :=
begin
  sorry
end

end marcias_hair_length_l416_416548


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416956

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416956


namespace product_of_consecutive_integers_l416_416746

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416746


namespace greatest_divisor_four_consecutive_integers_l416_416841

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416841


namespace sin_angle_FAE_correct_l416_416026

-- Definitions based on the problem conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨4, 4 * Real.sqrt 3⟩
def B : Point := ⟨0, 0⟩
def C : Point := ⟨8, 0⟩
def F : Point := ⟨2, 0⟩ -- Since F is one-fourth the segment BC from B
def E : Point := ⟨6, 0⟩ -- Since G is one-fourth the segment BC from C

noncomputable def sin_angle_FAE : ℝ := sorry

-- Theorem that we need to prove
theorem sin_angle_FAE_correct : sin_angle_FAE = (3 * Real.sqrt 3) / 56 := sorry

end sin_angle_FAE_correct_l416_416026


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416874

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416874


namespace smallest_power_of_12_not_palindrome_l416_416331

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def power_of_12_not_palindrome (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 12^k ∧ ¬ is_palindrome (n.repr)

theorem smallest_power_of_12_not_palindrome : ∃ n : ℕ, power_of_12_not_palindrome n ∧ ∀ m : ℕ, power_of_12_not_palindrome m → n ≤ m :=
sorry

end smallest_power_of_12_not_palindrome_l416_416331


namespace count_valid_integer_values_l416_416301

theorem count_valid_integer_values :
  {n : ℤ | 3200 * (2 / 5 : ℚ)^n ∈ ℤ}.to_finset.card = 9 := 
sorry

end count_valid_integer_values_l416_416301


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416945

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416945


namespace visible_factor_count_200_to_250_l416_416257

def visible_factor_number (n : ℕ) : Prop :=
  (∃ d1 d2 d3 : ℕ, n = 100 * d1 + 10 * d2 + d3 ∧ d1 ≠ 0 ∧ ∀ d i, i ∈ [d1, d2, d3]
    → d ≠ 0 ∧ d ∣ n)

def count_visible_factor_numbers : ℕ :=
  (Finset.range' 200 51).filter (λ n, visible_factor_number n).card

theorem visible_factor_count_200_to_250 : count_visible_factor_numbers = 20 := sorry

end visible_factor_count_200_to_250_l416_416257


namespace total_arrangements_l416_416273

-- Define the conditions as logical propositions
def num_leaders : Nat := 21
def front_row : Nat := 11
def back_row : Nat := 10
def middle_chinese_leader : Prop := true -- Placeholder for the condition
def adjacent_us_russia : Prop := true -- Placeholder for the condition

-- Define the main theorem statement
theorem total_arrangements (h1 : num_leaders = 21)
                           (h2 : front_row = 11)
                           (h3 : back_row = 10)
                           (h4 : middle_chinese_leader)
                           (h5 : adjacent_us_russia) :
  num_leaders = front_row + back_row ∧
  middle_chinese_leader ∧ 
  adjacent_us_russia → 
  ∃ arrangements : Nat, arrangements = A^2_1 * A^18_18 :=
by
  sorry

end total_arrangements_l416_416273


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416895

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416895


namespace sum_of_roots_l416_416614

open Complex

theorem sum_of_roots (P : Polynomial ℝ)
  (hP : P.monic)
  (h_roots : ∃ θ : ℝ, 0 < θ ∧ θ < π / 4 ∧ 
    (P.root_set ℂ).erase 0 = {[cos θ + sin θ * I, sin θ + cos θ * I, cos θ - sin θ * I, sin θ - cos θ * I]}) :
  let roots := {(cos θ + sin θ * I), (cos θ - sin θ * I), (sin θ + cos θ * I), (sin θ - cos θ * I)} in
  P.eval 0 / 2 = 
  (cos θ ^ 2 - sin θ ^ 2) -> (0 < θ ∧ θ < π / 4) -> θ = π / 6 → 
  (cos θ + sin θ + cos θ - sin θ) = 1 + sqrt 3 :=
by {
  sorry
}

end sum_of_roots_l416_416614


namespace domain_of_tan_2x_plus_pi_over_3_l416_416604

noncomputable def domain_tan_transformed : Set ℝ :=
  {x : ℝ | ∀ (k : ℤ), x ≠ k * (Real.pi / 2) + (Real.pi / 12)}

theorem domain_of_tan_2x_plus_pi_over_3 :
  (∀ x : ℝ, x ∉ domain_tan_transformed ↔ ∃ (k : ℤ), x = k * (Real.pi / 2) + (Real.pi / 12)) :=
sorry

end domain_of_tan_2x_plus_pi_over_3_l416_416604


namespace four_consecutive_product_divisible_by_12_l416_416989

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416989


namespace swimming_pool_cost_l416_416123

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l416_416123


namespace four_consecutive_integers_divisible_by_12_l416_416762

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416762


namespace first_player_wins_for_6x8_chocolate_first_player_wins_for_odd_rows_even_columns_first_player_loses_for_equal_dimensions_l416_416606

theorem first_player_wins_for_6x8_chocolate (marked : Fin 6 × Fin 8) :
  Player1Wins (mk_square 6 8 marked) :=
sorry

theorem first_player_wins_for_odd_rows_even_columns (a b : ℕ) (h1 : Odd a) (h2 : Even b) (marked : Fin a × Fin b) :
  Player1Wins (mk_square a b marked) := 
sorry 

theorem first_player_loses_for_equal_dimensions (n : ℕ) (marked : Fin n × Fin n) :
  ¬Player1Wins (mk_square n n marked) := 
sorry

end first_player_wins_for_6x8_chocolate_first_player_wins_for_odd_rows_even_columns_first_player_loses_for_equal_dimensions_l416_416606


namespace no_consecutive_numbers_adjacent_implies_probability_l416_416101

noncomputable def cube_faces := Fin 6

def consecutive_numbers (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 6) ∨ (a = 6 ∧ b = 1) ∨ (b = a + 1) ∨ (a = b + 1)

def valid_cube_configuration (f : cube_faces → ℕ) : Prop :=
  ∀ i j, (f i = 1 ∧ f j = 2) ∨
         consecutive_numbers (f i) (f j) →
         ¬ adjacent i j

theorem no_consecutive_numbers_adjacent_implies_probability
  (f : cube_faces → ℕ) :
  (∀ i j : cube_faces, consecutive_numbers (f i) (f j) → ¬ adjacent i j) →
  24.to_rat / 120.to_rat = 1.to_rat / 5.to_rat :=
sorry

end no_consecutive_numbers_adjacent_implies_probability_l416_416101


namespace area_of_triangle_l416_416002

-- condition definitions
def line_l1_polar_eqn (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2
def curve_C_polar_eqn (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- point M in polar coordinates
def point_M : ℝ × ℝ := (3, π / 6)

-- condition that must be satisfied
def intersection_condition (α : ℝ) : Prop := 
  0 < α ∧ α < π / 2 ∧ 
  let ρ_A := 2 / Real.cos α in 
  let ρ_B := 4 * Real.sin α in
  ρ_A * ρ_B = 8 * Real.sqrt 3

-- area calculation 
theorem area_of_triangle (α : ℝ) :
  intersection_condition α →
  let OB := 2 * Real.sqrt 3 in
  let θ_MOB := π / 6 in
  let OM := 3 in
  let area := (1 / 2) * OM * OB * Real.sin θ_MOB in
  area = (3 * Real.sqrt 3) / 2 :=
begin
  sorry
end

end area_of_triangle_l416_416002


namespace work_problem_l416_416245

theorem work_problem (x : ℝ) (h1 : x > 0) 
                      (h2 : (2 * (1 / 4 + 1 / x) + 2 * (1 / x) = 1)) : 
                      x = 8 := sorry

end work_problem_l416_416245


namespace greatest_divisor_four_consecutive_integers_l416_416837

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416837


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416878

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416878


namespace Vasya_numbers_l416_416164

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416164


namespace trihedral_angle_planes_l416_416275

theorem trihedral_angle_planes (O A B C: Point) (h₁ : angle A O B = 60) (h₂ : angle B O C = 60) (h₃ : angle C O A = 60): 
  ∃ θ, θ = Real.arccos (√3 / 3) ∧ 
  (angle A (foot (O ←> B C)) O = θ) ∧ 
  (angle B (foot (O ←> C A)) O = θ) ∧ 
  (angle C (foot (O ←> A B)) O = θ) := sorry

end trihedral_angle_planes_l416_416275


namespace faster_train_speed_l416_416638

theorem faster_train_speed
  (length_per_train : ℝ)
  (speed_slower_train : ℝ)
  (passing_time_secs : ℝ)
  (speed_faster_train : ℝ) :
  length_per_train = 80 / 1000 →
  speed_slower_train = 36 →
  passing_time_secs = 36 →
  speed_faster_train = 52 :=
by
  intro h_length_per_train h_speed_slower_train h_passing_time_secs
  -- Skipped steps would go here
  sorry

end faster_train_speed_l416_416638


namespace proof_problem_l416_416436

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416436


namespace patrol_boat_safety_l416_416258

-- Definitions based on the problem's conditions
def speed_still_water := 40 -- speed in still water in km/h
def downstream_speed (flow_speed : ℝ) := speed_still_water + flow_speed
def upstream_speed (flow_speed : ℝ) := speed_still_water - flow_speed

-- Condition: downstream speed is twice the upstream speed
def condition1 (flow_speed : ℝ) : Prop :=
  downstream_speed flow_speed = 2 * upstream_speed flow_speed

-- Derived quantities
def flow_speed_solution := 40 / 3

-- Condition for catching up with the raft
def catch_up_time (flow_speed : ℝ) (time : ℝ) : Prop :=
  (downstream_speed flow_speed * time) =
  ((time + 1/2) * flow_speed)

-- Derived catchup time
def catch_up_time_solution := 1 / 6

-- Complete Lean 4 statement verifying the conditions and the answers
theorem patrol_boat_safety (flow_speed : ℝ) (catch_up_time : ℝ) :
  (condition1 flow_speed ∧
  flow_speed = flow_speed_solution ∧
  catch_up_time flow_speed_solution catch_up_time ∧
  catch_up_time = catch_up_time_solution) :=
by
  sorry

end patrol_boat_safety_l416_416258


namespace find_n_and_sum_of_digits_l416_416116

theorem find_n_and_sum_of_digits :
  ∃ (n : ℕ), (n + 1)! + (n + 3)! = n! * 1190 ∧ nat.digits 10 n = [8] :=
by 
  sorry

end find_n_and_sum_of_digits_l416_416116


namespace whale_consumption_l416_416271

-- Define the conditions
def first_hour_consumption (x : ℕ) := x
def second_hour_consumption (x : ℕ) := x + 3
def third_hour_consumption (x : ℕ) := x + 6
def fourth_hour_consumption (x : ℕ) := x + 9
def fifth_hour_consumption (x : ℕ) := x + 12
def sixth_hour_consumption (x : ℕ) := x + 15
def seventh_hour_consumption (x : ℕ) := x + 18
def eighth_hour_consumption (x : ℕ) := x + 21
def ninth_hour_consumption (x : ℕ) := x + 24

def total_consumed (x : ℕ) := 
  first_hour_consumption x + 
  second_hour_consumption x + 
  third_hour_consumption x + 
  fourth_hour_consumption x + 
  fifth_hour_consumption x + 
  sixth_hour_consumption x + 
  seventh_hour_consumption x + 
  eighth_hour_consumption x + 
  ninth_hour_consumption x

-- Prove that the total sum consumed equals 540
theorem whale_consumption : ∃ x : ℕ, total_consumed x = 540 ∧ sixth_hour_consumption x = 63 :=
by
  sorry

end whale_consumption_l416_416271


namespace greatest_divisor_of_consecutive_product_l416_416845

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416845


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416932

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416932


namespace smallest_non_palindrome_power_of_12_l416_416328

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 12 ^ k ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, m > 0 ∧ (∃ j : ℕ, m = 12 ^ j ∧ ¬ is_palindrome m) → m ≥ n) :=
by
  sorry

end smallest_non_palindrome_power_of_12_l416_416328


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416726

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416726


namespace div_product_four_consecutive_integers_l416_416783

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416783


namespace sum_inequality_l416_416384

theorem sum_inequality (n : ℕ) (a : Fin (n+2) → ℝ) (h : ∀ i, a i > 0) : 
  (∑ i in Finset.range n, a i.succ / a ((i + 1) % (n + 1))) 
  ≥ (∑ i in Finset.range n, (a i.succ + a ((i + 1) % (n + 1)) + 1) / (a ((i + 1) % (n + 1)) + a ((i + 2) % (n + 1)) + 1)) := sorry

end sum_inequality_l416_416384


namespace total_balloons_l416_416132

-- Defining constants for the number of balloons each person has
def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

-- Theorem stating the total number of balloons
theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  simp [tom_balloons, sara_balloons]
  sorry

end total_balloons_l416_416132


namespace proof_problem_l416_416363

-- Define the function
def f (x : ℝ) : ℝ := Real.sin (1 / 2 * x + Real.pi / 3)

-- Define the smallest positive period T
def T : ℝ := 4 * Real.pi

-- Define the maximum value condition
def max_value_condition (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 4 * k * Real.pi + Real.pi / 3

-- Define the interval of monotonic increase on [-2π, 2π]
def monotonic_increase_interval : Set ℝ := Icc (- 5 * Real.pi / 3) (Real.pi / 3)

-- The problem statement
theorem proof_problem (x : ℝ) :
  (∀ y : ℝ, f (y + T) = f y) ∧
  (f x = 1 → max_value_condition x) ∧
  (∀ y : ℝ, -2 * Real.pi ≤ y ∧ y ≤ 2 * Real.pi → 
    (f' (y) > 0) → y ∈ monotonic_increase_interval) :=
by
  sorry

end proof_problem_l416_416363


namespace cost_to_fill_pool_l416_416128

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l416_416128


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416894

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416894


namespace greatest_divisor_four_consecutive_l416_416698

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416698


namespace point_positions_relative_to_circle_l416_416211

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

def center : ℝ × ℝ := (1, -2)
def radius : ℝ := 3
def M : ℝ × ℝ := (4, -2)
def N : ℝ × ℝ := (1, 0)
def P : ℝ × ℝ := (5, 1)

theorem point_positions_relative_to_circle :
  dist M center = radius ∧
  dist N center < radius ∧
  dist P center > radius :=
by
    split
    · sorry -- Prove that dist M center = radius
    split
    · sorry -- Prove that dist N center < radius
    · sorry -- Prove that dist P center > radius

end point_positions_relative_to_circle_l416_416211


namespace num_common_tangents_l416_416305

-- Define the first circle
def circle1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 4
-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Prove that the number of common tangent lines between the given circles is 2
theorem num_common_tangents : ∃ (n : ℕ), n = 2 ∧
  -- The circles do not intersect nor are they internally tangent
  (∀ (x y : ℝ), ¬(circle1 x y ∧ circle2 x y) ∧ 
  -- There exist exactly n common tangent lines
  ∃ (C : ℕ), C = n) :=
sorry

end num_common_tangents_l416_416305


namespace greatest_divisor_of_consecutive_product_l416_416856

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416856


namespace abs_neg_six_l416_416597

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l416_416597


namespace smallest_positive_non_palindrome_power_of_12_l416_416344

def is_palindrome (n : ℕ) : Bool :=
  let s := toDigits 10 n
  s = s.reverse

theorem smallest_positive_non_palindrome_power_of_12 : ∃ k : ℕ, k > 0 ∧ (12^k = 12 ∧ ¬ is_palindrome (12^k)) :=
by {
  sorry
}

end smallest_positive_non_palindrome_power_of_12_l416_416344


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416863

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416863


namespace div_product_four_consecutive_integers_l416_416795

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416795


namespace vasya_numbers_l416_416170

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416170


namespace cosine_product_l416_416225

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l416_416225


namespace number_of_players_l416_416586

-- Define the conditions
def total_games (n : ℕ) : ℕ := 2 * n * (n - 1)

-- Theorem: If a given number of games is 342,
-- then the number of players in the tournament is 19.
theorem number_of_players : (∃ n : ℕ, total_games n = 342) → ∃ n : ℕ, n = 19 :=
by
  intros h
  cases h with n hn
  use n
  sorry

end number_of_players_l416_416586


namespace number_of_middle_school_classes_l416_416260

theorem number_of_middle_school_classes
  (elementary_classes : ℕ := 4)
  (per_class_donation : ℕ := 5)
  (total_donation : ℕ := 90)
  (middle_classes_unknown : ℕ) :
  2 * per_class_donation * (elementary_classes + middle_classes_unknown) = total_donation →
  middle_classes_unknown = 5 :=
begin
  sorry,
end

end number_of_middle_school_classes_l416_416260


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416927

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416927


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416669

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416669


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416913

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416913


namespace not_necessarily_divisible_by_28_l416_416589

theorem not_necessarily_divisible_by_28 (k : ℤ) (h : 7 ∣ (k * (k + 1) * (k + 2))) : ¬ (28 ∣ (k * (k + 1) * (k + 2))) :=
sorry

end not_necessarily_divisible_by_28_l416_416589


namespace greatest_divisor_four_consecutive_integers_l416_416827

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416827


namespace probability_between_C_and_E_l416_416062

variable (A B C D E : Point)
variable (l : Line)
variable (h1 : segment l A B = 4 * segment l A D)
variable (h2 : segment l A B = 8 * segment l B C)
variable (h3 : bisects_segment E (segment l C D))

theorem probability_between_C_and_E :
  (probability (λ x, x ∈ segment l C E) (λ x, x ∈ segment l A B) = 5 / 16) := by
sorry

end probability_between_C_and_E_l416_416062


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416884

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416884


namespace greatest_divisor_four_consecutive_integers_l416_416830

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416830


namespace sum_of_arith_geo_seq_l416_416480

noncomputable def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range n).sum a

theorem sum_of_arith_geo_seq (a : ℕ → ℚ) (n : ℕ) (q : ℚ) 
  (h1 : a 1 = 1)
  (hrec : ∀ k, k ≥ 3 → a k = (a (k-1) + a (k-2)) / 2)
  (hq : q = 1 ∨ q = -1/2) : 
  sum_of_first_n_terms a n = 
    if q = 1 then n else (2/3 - (2/3) * (-1/2)^n) :=
sorry

end sum_of_arith_geo_seq_l416_416480


namespace find_lambda_l416_416365

open Real

variables (λ : ℝ)
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (-2, 3)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2
  
def vec_lambda_a_plus_b (λ : ℝ) : ℝ × ℝ :=
  (λ * a.1 + b.1, λ * a.2 + b.2)
  
theorem find_lambda (h : dot_product (vec_lambda_a_plus_b (λ)) c = 0) : λ = -1 / 2 := by
  sorry

end find_lambda_l416_416365


namespace divisor_of_four_consecutive_integers_l416_416659

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416659


namespace divisor_of_four_consecutive_integers_l416_416644

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416644


namespace part1_part2_l416_416408

-- Definitions for Part 1
def f (x : ℝ) : ℝ := Real.log x - 1

theorem part1 (tangent_line_eqn : String) :
  tangent_line_eqn = "y = (1 / exp 2) * x" :=
sorry

-- Definitions for Part 2
def g (a x : ℝ) : ℝ := Real.exp (-a * x) + (Real.log x - a * x - 1) / x

theorem part2 {a x1 x2 m : ℝ} (ha : a > 0) (hx1 : 0 < x1) (hx2 : x1 < x2)
  (H1 : g a x1 = 0)
  (H2 : g a x2 = 0)
  (H3 : x1 * x2^3 > Real.exp m) :
  m ≤ 4 :=
sorry

end part1_part2_l416_416408


namespace tower_height_l416_416636

-- Define given conditions
variables (h : ℝ) (CD : ℝ) (theta_ACB theta_BDA : ℝ)
variables (AC AD : ℝ)

-- Given constants:
def theta_ACB : ℝ := 45
def theta_BDA : ℝ := 30
def CD : ℝ := 30

-- Define triangle properties:
def AC := h
def AD := h * Real.sqrt 3
def sin_theta_120 := Real.sin (120 * Real.pi / 180) -- sin 120 degrees

-- Law of Sines relates sides and angles:
theorem tower_height (h : ℝ) (CD : ℝ) (theta_ACB theta_BDA : ℝ) 
: AC = CD → theta_ACB = 45 → theta_BDA = 30 → h = 30 :=
by
  sorry

end tower_height_l416_416636


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416672

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416672


namespace initial_volume_of_mixture_l416_416632

theorem initial_volume_of_mixture :
  ∃ V : ℝ, (∀ (initial_volume: ℝ), 
  0.84 * initial_volume = 0.64 * (initial_volume + 18.75) → initial_volume = 225) :=
begin
  use 225,
  assume initial_volume h,
  linarith,
end

end initial_volume_of_mixture_l416_416632


namespace matching_polygons_pairs_l416_416142

noncomputable def are_matching_pairs (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons_pairs (n m : ℕ) :
  are_matching_pairs n m → (n, m) = (3, 9) ∨ (n, m) = (4, 6) ∨ (n, m) = (5, 5) ∨ (n, m) = (8, 4) :=
sorry

end matching_polygons_pairs_l416_416142


namespace greatest_divisor_of_four_consecutive_integers_l416_416971

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416971


namespace min_value_of_expression_l416_416042

noncomputable def minValueExpr (a b c : ℝ) : ℝ :=
  a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minValueExpr a b c >= 60 :=
by
  sorry

end min_value_of_expression_l416_416042


namespace polynomial_solution_l416_416317

noncomputable def satisfies_identity (P : ℝ × ℝ → ℝ) : Prop :=
  ∀ x y z t : ℝ, P(x, y) * P(z, t) = P(x * z - y * t, x * t + y * z)

theorem polynomial_solution (P : ℝ × ℝ → ℝ) (hP : satisfies_identity P) :
  ∃ (a : ℕ), P = (λ p : ℝ × ℝ, (p.1^2 + p.2^2 : ℝ) ^ a) ∨ P = (λ p : ℝ × ℝ, 0) :=
sorry

end polynomial_solution_l416_416317


namespace greatest_divisor_of_four_consecutive_integers_l416_416804

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416804


namespace exists_infinite_triples_a_no_triples_b_l416_416066

-- Question (a)
theorem exists_infinite_triples_a : ∀ k : ℕ, ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2 - 1) :=
by {
  sorry
}

-- Question (b)
theorem no_triples_b : ¬ ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2) :=
by {
  sorry
}

end exists_infinite_triples_a_no_triples_b_l416_416066


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416944

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416944


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416882

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416882


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416881

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416881


namespace function_increasing_interval_l416_416404

theorem function_increasing_interval :
  ∀ (ϕ : ℝ) (x : ℝ), -π < ϕ ∧ ϕ < 0 ∧ (∃ y : ℝ, y = sin (2 * x + ϕ + (2 * π) / 3) ∧ y = 1) →
  ϕ = -π / 6 →
  sin (2 * x - π / 6) < sin (2 * x' - π / 6) ∀ (x x' : ℝ), -π / 6 < x ∧ x < x' ∧ x' < π / 3 :=
sorry

end function_increasing_interval_l416_416404


namespace Vasya_numbers_l416_416163

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416163


namespace triangle_right_angle_l416_416006

theorem triangle_right_angle
  (A B C : Type)
  (angle : A → ℝ)
  (triangle : A → A → A → Prop)
  (ABC : triangle A B C)
  (sum_angles : ∀ {x y z : A}, triangle x y z → angle x + angle y + angle z = 180)
  (h : angle A = angle C - angle B) :
  angle C = 90 :=
by
  -- The proof will be provided here
  sorry

end triangle_right_angle_l416_416006


namespace four_consecutive_product_divisible_by_12_l416_416986

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416986


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416872

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416872


namespace poly_division_l416_416036

open Complex Polynomial

noncomputable def Q (a : List ℂ) : ℂ → ℂ
| z := 
  if h : a.length = 1 then z - List.head a else 
  let n := a.length - 1;
  let prevQ : ℂ → ℂ := Q (List.dropLast a);
  prevQ (prevQ z - List.getLast a sorry)

theorem poly_division {P : Polynomial ℂ} (hdeg : P.natDegree = 1992)
  (hroot : ∀ z1 z2 : ℂ, z1 ≠ z2 → P.eval z1 = 0 → P.eval z2 ≠ 0) :
  ∃ a : List ℂ, a.length = 1992 ∧ ∀ z : ℂ, P.eval z = 0 → Q a z = 0 :=
sorry

end poly_division_l416_416036


namespace swimming_pool_cost_l416_416122

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l416_416122


namespace admission_ticket_parsing_l416_416465

-- Define admission ticket parsing
structure AdmissionTicket :=
  (year: Nat)
  (exam_room: Nat)
  (seat_number: Nat)
  (gender: String)

-- Admission ticket based on the problem description
def parseAdmissionTicket (ticket: String) : AdmissionTicket :=
  { year := ticket.toSubstring.is_extract 0 2).toNat,
    exam_room := (ticket.toSubstring 2 4).toNat,
    seat_number := (ticket.toSubstring 4 6).toNat,
    gender := if (ticket.toSubstring 6 7) = "1" then "male" else "female" }

-- Given ticket number
def ticket := "0202022"

-- Parse the provided admission ticket number
noncomputable def parsedTicket := parseAdmissionTicket ticket

theorem admission_ticket_parsing :
  parsedTicket = { year := 2002, exam_room := 2, seat_number := 2, gender := "female" } :=
by
  sorry

end admission_ticket_parsing_l416_416465


namespace square_begins_with_nines_l416_416289

theorem square_begins_with_nines :
  ∃ N : ℕ, let digits := (10^(1984) - 5) in 
  N = digits → (N^2 / 10^1985) = 999...(1983 nines) := 
by
  sorry

end square_begins_with_nines_l416_416289


namespace perfect_square_trinomial_l416_416430

theorem perfect_square_trinomial (m : ℤ) : 
  (x^2 - (m - 3) * x + 16 = (x - 4)^2) ∨ (x^2 - (m - 3) * x + 16 = (x + 4)^2) ↔ (m = -5 ∨ m = 11) := by
  sorry

end perfect_square_trinomial_l416_416430


namespace fraction_division_l416_416187

-- Definitions derived from conditions
def dec18 := 0.181818... -- repeating decimal 0.\overline{18}
def dec36 := 0.363636... -- repeating decimal 0.\overline{36}

lemma dec18_as_fraction : dec18 = 18 / 99 := sorry
lemma dec36_as_fraction : dec36 = 36 / 99 := sorry

-- The statement of the problem
theorem fraction_division : (dec18 / dec36) = (1 / 2) :=
by
  rw [dec18_as_fraction, dec36_as_fraction]
  -- Further steps involve the simplification done in the original solution
  sorry

end fraction_division_l416_416187


namespace product_of_consecutive_integers_l416_416761

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416761


namespace fourth_student_id_correct_l416_416466

structure SamplingProblem :=
  (total_students : ℕ)
  (sample_size : ℕ)
  (included_ids : Set ℕ)
  (student_id : ℕ)

def sampling_problem_instance : SamplingProblem :=
  { total_students := 50,
    sample_size := 4,
    included_ids := {6, 30, 42},
    student_id := 18 }

theorem fourth_student_id_correct (prob : SamplingProblem) :
  prob.total_students = 50 →
  prob.sample_size = 4 →
  prob.included_ids = {6, 30, 42} →
  (exists k, k = (prob.total_students / prob.sample_size).floor ∧
             6 + k * 1 = 18 ∧
             6 + k * 2 = 30 ∧
             6 + k * 3 = 42) →
  prob.student_id = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end fourth_student_id_correct_l416_416466


namespace product_of_consecutive_integers_l416_416757

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416757


namespace find_subtracted_value_l416_416268

theorem find_subtracted_value :
  ∃ x : ℤ, let n := 121 in let result := 2 * n in result - x = 104 ∧ x = 138 :=
by
  sorry

end find_subtracted_value_l416_416268


namespace complex_arithmetic_l416_416526

variable (A M S P : ℂ)
variable hA : A = 6 - 2 * Complex.I
variable hM : M = -5 + 3 * Complex.I
variable hS : S = 2 * Complex.I
variable hP : P = 3

theorem complex_arithmetic : A - M + S - P = 8 - 3 * Complex.I := by
  rw [hA, hM, hS, hP]
  have real_part := (6 : ℂ) - (-5 : ℂ) + 0 - 3
  have imag_part := (-2 : ℂ) - 3 + (2 : ℂ) + 0
  rw [
    Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im,
    Complex.neg_re, Complex.neg_im, Complex.of_real_im, Complex.of_real_re,
    Complex.I_re, Complex.I_im
  ]
  simp [real_part, imag_part]
  sorry

end complex_arithmetic_l416_416526


namespace num_perfect_cube_factors_of_120_l416_416424

def is_factor (n d : ℕ) : Prop :=
  d ∣ n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k ^ 3

def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (3, 1), (5, 1)]  -- Prime factorization of 120: 2^3 * 3 * 5

def cube_factors_of_120 : List ℕ :=
  [1, 8]  -- The only perfect cube factors of 120: 1 = 1^3, 8 = 2^3

theorem num_perfect_cube_factors_of_120 :
  ∃ l : List ℕ, l.length = 2 ∧ ∀ x ∈ l, is_factor 120 x ∧ is_perfect_cube x :=
by
  use cube_factors_of_120
  split
  · -- proof that the length of cube_factors_of_120 is 2
    sorry
  · -- proof that each element in cube_factors_of_120 is a factor of 120 and a perfect cube
    sorry

end num_perfect_cube_factors_of_120_l416_416424


namespace pentagon_cosine_identity_l416_416230

    variable (A B C D E : Point)
    variable (circle : Circle)

    -- Given conditions
    variable (inscribed : Inscribed circle [A, B, C, D, E])
    variable (AB_eq : AB = 3)
    variable (BC_eq : BC = 3)
    variable (CD_eq : CD = 3)
    variable (DE_eq : DE = 3)
    variable (AE_eq : AE = 2)

    -- Goal: prove the given equation
    theorem pentagon_cosine_identity : 
      (1 - cos (∠ B)) * (1 - cos (∠ ACE)) = 1 / 9 := 
    by
      sorry
    
end pentagon_cosine_identity_l416_416230


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416883

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416883


namespace correct_calculation_l416_416208

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l416_416208


namespace all_positive_integers_are_nice_l416_416215

def isNice (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, ∃ m : ℕ, a i = 2 ^ m) ∧ n = (Finset.univ.sum a) / k

theorem all_positive_integers_are_nice : ∀ n : ℕ, 0 < n → isNice n := sorry

end all_positive_integers_are_nice_l416_416215


namespace no_infinite_positive_integer_sequence_l416_416233

theorem no_infinite_positive_integer_sequence (a : ℕ → ℕ) :
  ¬(∀ n, a (n - 1) ^ 2 ≥ 2 * a n * a (n + 2)) :=
sorry

end no_infinite_positive_integer_sequence_l416_416233


namespace bonus_points_amount_l416_416552

def points_per_10_dollars : ℕ := 50

def beef_price : ℕ := 11
def beef_quantity : ℕ := 3

def fruits_vegetables_price : ℕ := 4
def fruits_vegetables_quantity : ℕ := 8

def spices_price : ℕ := 6
def spices_quantity : ℕ := 3

def other_groceries_total : ℕ := 37

def total_points : ℕ := 850

def total_spent : ℕ :=
  (beef_price * beef_quantity) +
  (fruits_vegetables_price * fruits_vegetables_quantity) +
  (spices_price * spices_quantity) +
  other_groceries_total

def points_from_spending : ℕ :=
  (total_spent / 10) * points_per_10_dollars

theorem bonus_points_amount :
  total_spent > 100 → total_points - points_from_spending = 250 :=
by
  sorry

end bonus_points_amount_l416_416552


namespace probability_of_same_length_segments_l416_416513

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l416_416513


namespace Mr_Bhaskar_expenses_l416_416052

variable (M D : ℝ)

theorem Mr_Bhaskar_expenses (h1 : 20 * D = 24 * (D - 3)) : M = 360 :=
by
  -- Initial setup and assumptions
  have daily_expenses := 18
  -- Calculate total expenses for 20 days
  have total_expenses : M = 20 * daily_expenses := by 
    sorry
  -- Conclude the proof
  exact total_expenses
  sorry

end Mr_Bhaskar_expenses_l416_416052


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416737

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416737


namespace cost_to_fill_pool_l416_416129

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l416_416129


namespace length_AE_is_correct_l416_416467

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨8, 0⟩
def D : Point := ⟨3, 0⟩
def C : Point := ⟨6, 3⟩

def distance (p1 p2 : Point) : ℚ :=
  (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

noncomputable def E : Point :=
  ⟨14 / 3, 2 / 3⟩ -- This point E is directly calculated from given conditions

theorem length_AE_is_correct :
  sqrt (distance A E) = sqrt (296 / 9) :=
by
  sorry

end length_AE_is_correct_l416_416467


namespace div_product_four_consecutive_integers_l416_416800

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416800


namespace greatest_divisor_of_four_consecutive_integers_l416_416975

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416975


namespace four_consecutive_product_divisible_by_12_l416_416996

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416996


namespace greatest_divisor_of_four_consecutive_integers_l416_416973

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416973


namespace slope_angle_of_line_is_120_degrees_l416_416107

-- Define the equation of the line
def line_equation : ℝ → ℝ := λ x => -sqrt 3 * x + 1

-- Define the slope of the line
def slope : ℝ := -sqrt 3

-- Define the slope angle function (tan θ = slope)
def slope_angle (m : ℝ) : ℝ :=
  real.arctan m * 180 / real.pi

-- Prove that the slope angle of the line y = -sqrt 3 * x + 1 is 120 degrees
theorem slope_angle_of_line_is_120_degrees : slope_angle slope = 120 := 
by
  sorry

end slope_angle_of_line_is_120_degrees_l416_416107


namespace find_range_of_x_l416_416371

theorem find_range_of_x (f : ℝ → ℝ) 
  (h_dom : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) > 2) 
  (h_f1 : f 1 = 2020) :
  {x : ℝ | f (x - 2021) > 2 * (x - 1012)} = set.Ioi 2022 := 
by 
  sorry

end find_range_of_x_l416_416371


namespace marcia_hair_length_l416_416551

theorem marcia_hair_length :
  ∀ (initial_length : ℝ), initial_length = 24 → 
  ∀ (first_cut_fraction growth second_cut : ℝ), first_cut_fraction = 1/2 → growth = 4 → second_cut = 2 → 
  let length_after_first_cut := initial_length * (1 - first_cut_fraction) in
  let length_after_growth := length_after_first_cut + growth in
  let final_length := length_after_growth - second_cut in
  final_length = 14 :=
by
  intros initial_length h1 first_cut_fraction h2 growth h3 second_cut h4;
  simp [h1, h2, h3, h4];
  sorry

end marcia_hair_length_l416_416551


namespace vasya_numbers_l416_416185

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416185


namespace probability_sum_less_than_product_l416_416141

def set_of_numbers := {1, 2, 3, 4, 5, 6, 7}

def count_valid_pairs : ℕ :=
  set_of_numbers.to_list.product set_of_numbers.to_list
    |>.count (λ (ab : ℕ × ℕ), (ab.1 - 1) * (ab.2 - 1) > 1)

def total_combinations := (set_of_numbers.to_list).length ^ 2

theorem probability_sum_less_than_product :
  (count_valid_pairs : ℚ) / total_combinations = 36 / 49 :=
by
  -- Placeholder for proof, since proof is not requested
  sorry

end probability_sum_less_than_product_l416_416141


namespace greatest_divisor_of_four_consecutive_integers_l416_416815

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416815


namespace vasya_numbers_l416_416157

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416157


namespace divisor_of_product_of_four_consecutive_integers_l416_416704

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416704


namespace number_of_obtuse_triangles_l416_416377

theorem number_of_obtuse_triangles:
  (∃ triangles : set (ℕ × ℕ × ℕ), 
     (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a < b ∧ b < c ∧ a^2 + b^2 < c^2 ∧ a + b > c ∧ 
       a > 0 ∧ a < 7 ∧ b > 0 ∧ b < 7 ∧ c > 0 ∧ c < 7) ∧ 
     set.finite triangles ∧ 
     triangles.to_finset.card = 8) :=
sorry

end number_of_obtuse_triangles_l416_416377


namespace trajectory_midpoint_l416_416321

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

-- Define the condition that a line passes through the point (0, 1)
def line_through_fixed_point (k x y : ℝ) := y = k * x + 1

-- Define the theorem to prove the trajectory of the midpoint of the chord
theorem trajectory_midpoint (x y k : ℝ) (h : ∃ x y, hyperbola x y ∧ line_through_fixed_point k x y) : 
    4 * x^2 - y^2 + y = 0 := 
sorry

end trajectory_midpoint_l416_416321


namespace mean_of_set_is_12_point_8_l416_416090

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l416_416090


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416662

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416662


namespace product_of_consecutive_integers_l416_416745

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416745


namespace distance_A_to_B_l416_416623

noncomputable def dist_point_A_to_point_B (perimeter_smaller_square : ℝ) (area_larger_square : ℝ) : ℝ :=
  let side_smaller_square := perimeter_smaller_square / 4 in
  let side_larger_square := Real.sqrt area_larger_square in
  let horizontal_distance := side_smaller_square + side_larger_square in
  let vertical_distance := side_larger_square - side_smaller_square in
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2)

theorem distance_A_to_B :
  dist_point_A_to_point_B 8 36 ≈ 8.9 :=
by
  let perimeter_smaller_square := 8
  let area_larger_square := 36
  let side_smaller_square := perimeter_smaller_square / 4
  let side_larger_square := Real.sqrt area_larger_square
  let horizontal_distance := side_smaller_square + side_larger_square
  let vertical_distance := side_larger_square - side_smaller_square
  let distance := Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2)
  have h : distance = 4 * Real.sqrt 5 := by sorry
  have approx : 4 * Real.sqrt 5 ≈ 8.9 := by sorry
  exact approx

end distance_A_to_B_l416_416623


namespace greatest_divisor_of_consecutive_product_l416_416853

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416853


namespace probability_of_triangle_angles_l416_416362

open_locale big_operators

def triangle_probability: ℚ :=
  let n := 179 in
  let total_combinations := n.choose 3 in
  let valid_triples := (∑ a in finset.range (n / 2),
    if a % 2 = 0 then
      89 - 3 * a / 2
    else
      91 - 3 * (a / 2 + 1))  in
  (valid_triples : ℚ) / total_combinations

theorem probability_of_triangle_angles:
  triangle_probability = 2611 / 939929 :=
sorry

end probability_of_triangle_angles_l416_416362


namespace mean_is_12_point_8_l416_416093

variable (m : ℝ)
variable median_condition : m + 7 = 12

theorem mean_is_12_point_8 (m : ℝ) (median_condition : m + 7 = 12) : 
(mean := (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5) = 64 / 5 :=
by {
  sorry
}

end mean_is_12_point_8_l416_416093


namespace max_f_value_sin_4theta_l416_416421

def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.sin x - Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (1, Real.sin x + Real.cos x)
def f (x : ℝ) : ℝ := vec_dot (a x) (b x)

theorem max_f_value : ∃ (x : ℝ) (k : ℤ), f x = 1 + Real.sqrt 2 ∧ x = k * Real.pi + (3 / 8) * Real.pi := sorry

theorem sin_4theta (θ : ℝ) : f θ = 8 / 5 → Real.sin (4 * θ) = 16 / 25 := sorry

end max_f_value_sin_4theta_l416_416421


namespace max_students_l416_416310

theorem max_students (books students : ℕ) (h1 : books = 61) (h2 : ∀ s, ∃ k : ℕ, k ≥ 3) : students ≤ 30 :=
sorry

end max_students_l416_416310


namespace arc_MTN_constant_measure_l416_416482

-- Define the given conditions
variables {P Q R T M N : Point}
variables {circle : Circle}
variables {altitude h : ℝ}
variables (PQ PR : Line)

-- Conditions of the problem
axiom PQ_eq_PR : PQ = PR
axiom angle_QPR : ∠QPR = 72
axiom radius_half_altitude : circle.radius = 0.5 * h
axiom circle_tangent_at_T : circle.tangency_point = T
axiom intersects_at_M : Line_intersect circle PR = M
axiom intersects_at_N : Line_intersect circle QR = N

-- The statement we need to prove
theorem arc_MTN_constant_measure : arc_measure circle MTN = 72 := 
by sorry

-- Additional definitions to complete the theorem statement.
def Point := Type*
def Line := Point → Point → Prop
def Circle := {center : Point // radius : ℝ}
def arc_measure : Circle → Point → Point → ℝ := sorry
def Line_intersect : Circle → Line → Point := sorry

#check arc_MTN_constant_measure

end arc_MTN_constant_measure_l416_416482


namespace coloring_impossible_l416_416221

-- Define vertices for the outer pentagon and inner star
inductive Vertex
| A | B | C | D | E | A' | B' | C' | D' | E'

open Vertex

-- Define segments in the figure
def Segments : List (Vertex × Vertex) :=
  [(A, B), (B, C), (C, D), (D, E), (E, A),
   (A, A'), (B, B'), (C, C'), (D, D'), (E, E'),
   (A', C), (C, E'), (E, B'), (B, D'), (D, A')]

-- Color type
inductive Color
| Red | Green | Blue

open Color

-- Condition for coloring: no two segments of the same color share a common endpoint
def distinct_color (c : Vertex → Color) : Prop :=
  ∀ (v1 v2 v3 : Vertex) (h1 : (v1, v2) ∈ Segments) (h2 : (v2, v3) ∈ Segments),
  c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v1 ≠ c v3

-- Statement of the proof problem
theorem coloring_impossible : ¬ ∃ (c : Vertex → Color), distinct_color c := 
by 
  sorry

end coloring_impossible_l416_416221


namespace total_yellow_balloons_l416_416130

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end total_yellow_balloons_l416_416130


namespace max_min_distance_l416_416479

theorem max_min_distance :
  ∀ (ρ θ t : ℝ),
  (ρ = 2 * cos θ + 2 * sin θ) →
  (-1 + t, -1 - t) ∈ set.univ →
  ∃ (max_dist min_dist : ℝ),
  max_dist = 3 * real.sqrt 2 ∧ min_dist = real.sqrt 2 :=
by sorry

end max_min_distance_l416_416479


namespace greatest_divisor_of_four_consecutive_integers_l416_416970

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416970


namespace scale_model_height_l416_416279

theorem scale_model_height (scale_ratio actual_height : ℝ) 
  (h_scale : scale_ratio = 25)
  (h_actual : actual_height = 305) :
  (actual_height / scale_ratio).round = 12 :=
by {
  rw [h_scale, h_actual],
  norm_num,
  sorry -- Proof steps would go here
}

end scale_model_height_l416_416279


namespace greatest_divisor_four_consecutive_integers_l416_416825

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416825


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416912

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416912


namespace area_trapezoid_proof_l416_416598

variables (A B C D F G E : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace F] [MetricSpace G] [MetricSpace E]
variables [HasArea A] [HasArea B] [HasArea C] [HasArea D] [HasArea F] [HasArea G] [HasArea E]

-- Rectangle ABCD with area 416 square cm
constant area_ABC : ℝ
constant area_ABCD : area_ABC = 416

-- Midpoint D of segment EG
constant midpoint_D : D

-- F is on BC
constant F_on_BC : F

-- Area of trapezoid AFGE
def area_trapezoid_AFGE := 416

-- Prove that the area of the trapezoid AFGE is 416 square cm
theorem area_trapezoid_proof (h1 : area_ABCD = 416) (h2: D = midpoint_D) (h3: F = F_on_BC) : area_trapezoid_AFGE = 416 :=
by {
  -- Proof omitted
  sorry
}

end area_trapezoid_proof_l416_416598


namespace percent_boys_l416_416463

def ratio_boys_girls := 3 / 7
def total_students := 42

theorem percent_boys (h : total_students > 0) : 
  (ratio_boys_girls * 100) = 42.857 := sorry

end percent_boys_l416_416463


namespace greatest_divisor_four_consecutive_integers_l416_416833

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416833


namespace equilateral_triangle_sum_of_sides_l416_416087

/-- The sum of the lengths of the three sides of an equilateral triangle,
    each of side 14/8 cm, is 21/4 cm. -/
theorem equilateral_triangle_sum_of_sides :
  ∀ (a : ℝ), a = 14 / 8 → 3 * a = 21 / 4 :=
by
  intro a h
  rw h
  norm_num
  sorry

end equilateral_triangle_sum_of_sides_l416_416087


namespace greatest_divisor_of_consecutive_product_l416_416844

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416844


namespace value_of_D_l416_416324

theorem value_of_D (D : ℤ) (h : 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89) : D = -5 :=
by sorry

end value_of_D_l416_416324


namespace inverse_proposition_correct_l416_416086

-- Definition of the given proposition
def congruent_triangles_implies_equal_areas (T1 T2 : Type) [triangle T1] [triangle T2] : Prop :=
  (congruent T1 T2) → (area T1 = area T2)

-- Definition of the inverse proposition
def equal_areas_implies_congruent_triangles (T1 T2 : Type) [triangle T1] [triangle T2] : Prop :=
  (area T1 = area T2) → (congruent T1 T2)

-- Proving that the inverse proposition is correct
theorem inverse_proposition_correct (T1 T2 : Type) [triangle T1] [triangle T2] :
    congruent_triangles_implies_equal_areas T1 T2 → equal_areas_implies_congruent_triangles T1 T2 :=
  sorry

end inverse_proposition_correct_l416_416086


namespace rectangular_prism_volume_l416_416202

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end rectangular_prism_volume_l416_416202


namespace correct_option_l416_416205

-- Definitions based on conditions in the problem
def optionA (a b : ℕ) : Prop := 3 * a * b - 2 * a * b = a * b
def optionB (y : ℕ) : Prop := 6 * y^2 - 2 * y^2 = 4
def optionC (a : ℕ) : Prop := 5 * a + a = 5 * a^2
def optionD (m n : ℕ) : Prop := m^2 * n - 3 * m * n^2 = -2 * m * n^2

-- The goal is to prove the correctness of Option A and the incorrectness of others
theorem correct_option (a b y m n : ℕ) : optionA a b ∧ ¬optionB y ∧ ¬optionC a ∧ ¬optionD m n :=
by {
  sorry -- proof goes here
}

end correct_option_l416_416205


namespace vasya_numbers_l416_416144

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416144


namespace sqrt3_infinite_non_repeating_decimal_l416_416100

theorem sqrt3_infinite_non_repeating_decimal :
  ∀ x : ℝ, x = Real.sqrt 3 → ¬(∃ p q : ℤ, q ≠ 0 ∧ x = p / q) ∧
  ¬(∃ m : ℕ, x = m) ∧
  x ≠ 1.73205 ∧
  ¬(∃ y : ℝ, y > 0 ∧ x = y * Real.floor x + (1 - y) * Real.floor (x + 1)) →
  ∃ seq : ℕ → ℚ, (∀ n : ℕ, seq n.succ ≠ seq n) ∧ x = ↑(Real.ofRat (seq n)) :=
sorry

end sqrt3_infinite_non_repeating_decimal_l416_416100


namespace correct_option_l416_416206

-- Definitions based on conditions in the problem
def optionA (a b : ℕ) : Prop := 3 * a * b - 2 * a * b = a * b
def optionB (y : ℕ) : Prop := 6 * y^2 - 2 * y^2 = 4
def optionC (a : ℕ) : Prop := 5 * a + a = 5 * a^2
def optionD (m n : ℕ) : Prop := m^2 * n - 3 * m * n^2 = -2 * m * n^2

-- The goal is to prove the correctness of Option A and the incorrectness of others
theorem correct_option (a b y m n : ℕ) : optionA a b ∧ ¬optionB y ∧ ¬optionC a ∧ ¬optionD m n :=
by {
  sorry -- proof goes here
}

end correct_option_l416_416206


namespace smallest_positive_non_palindromic_power_of_12_is_12_l416_416335

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def smallest_non_palindromic_power_of_12 : ℕ :=
  Nat.find (λ n, ∃ k : ℕ, n = 12^k ∧ ¬is_palindrome n)

theorem smallest_positive_non_palindromic_power_of_12_is_12 :
  smallest_non_palindromic_power_of_12 = 12 :=
by
  sorry

end smallest_positive_non_palindromic_power_of_12_is_12_l416_416335


namespace range_of_m_l416_416458

theorem range_of_m (x y m : ℝ) (h1 : x - 2 * y = 1) (h2 : 2 * x + y = 4 * m) (h3 : x + 3 * y < 6) : m < 7 / 4 :=
sorry

end range_of_m_l416_416458


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416677

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416677


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416664

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416664


namespace four_consecutive_integers_divisible_by_12_l416_416776

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416776


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416731

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416731


namespace investment_ratio_l416_416470

theorem investment_ratio (total_profit b_profit : ℝ) (a c b : ℝ) :
  total_profit = 150000 ∧ b_profit = 75000 ∧ a / c = 2 ∧ a + b + c = total_profit →
  a / b = 2 / 3 :=
by
  sorry

end investment_ratio_l416_416470


namespace unique_number_not_in_range_l416_416542

variable {ℝ : Type*} [Real ℝ]

def g (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)
def condition_20 (a b c d : ℝ) : Prop := g a b c d 20 = 20
def condition_99 (a b c d : ℝ) : Prop := g a b c d 99 = 99
def involution (a b c d : ℝ) : Prop := ∀ (x : ℝ), x ≠ -d / c → g a b c d (g a b c d x) = x

theorem unique_number_not_in_range (a b c d : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_20 : condition_20 a b c d)
  (h_99 : condition_99 a b c d)
  (h_involution : involution a b c d) :
  ∀ y : ℝ, ¬∃ x : ℝ, g a b c d x = y → y = 59.5 := 
by sorry

end unique_number_not_in_range_l416_416542


namespace collinear_I_O_M_l416_416395

-- Given definitions and conditions
variables {ABC : Type} [triangle ABC]
variable {I : incenter ABC}
variable {O : circumcenter ABC}
variables {A B C : points ABC}
variables {K L M : points ABC}
variable {P : point ABC}

-- Given the excircle touching points are K, L, M
axiom Excircle_touches (A B C K L M : points ABC) : touches_excircle_opposite_A ABC K L M

-- Given the midpoint of KL lies on the circumcircle
axiom Midpoint_KL_Circumcircle (K L : points ABC) : midpoint KL on circumcircle ABC

-- The theorem to prove
theorem collinear_I_O_M (ABC : Type) [triangle ABC]
  (I : incenter ABC) (O : circumcenter ABC)
  (K L M : points ABC) (P : midpoint KL)
  (h_excircle : touches_excircle_opposite_A ABC K L M)
  (h_circum : midpoint KL on circumcircle ABC) :
  collinear I O M :=
sorry

end collinear_I_O_M_l416_416395


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416939

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416939


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416910

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416910


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416904

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416904


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416727

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416727


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416893

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416893


namespace correct_option_l416_416204

-- Definitions based on conditions in the problem
def optionA (a b : ℕ) : Prop := 3 * a * b - 2 * a * b = a * b
def optionB (y : ℕ) : Prop := 6 * y^2 - 2 * y^2 = 4
def optionC (a : ℕ) : Prop := 5 * a + a = 5 * a^2
def optionD (m n : ℕ) : Prop := m^2 * n - 3 * m * n^2 = -2 * m * n^2

-- The goal is to prove the correctness of Option A and the incorrectness of others
theorem correct_option (a b y m n : ℕ) : optionA a b ∧ ¬optionB y ∧ ¬optionC a ∧ ¬optionD m n :=
by {
  sorry -- proof goes here
}

end correct_option_l416_416204


namespace divisor_of_four_consecutive_integers_l416_416650

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416650


namespace div_product_four_consecutive_integers_l416_416784

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416784


namespace divisor_of_product_of_four_consecutive_integers_l416_416713

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416713


namespace sequence_sum_formula_l416_416396

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_S : ∀ n, S n = (1 / 6) * (a n ^ 2 + 3 * a n - 4)) : 
  ∀ n, S n = (3 / 2) * n ^ 2 + (5 / 2) * n :=
by
  sorry

end sequence_sum_formula_l416_416396


namespace range_of_g_minus_x_on_interval_l416_416293

noncomputable def g (x : ℝ) : ℝ := x^2 - 3 * x + 4

theorem range_of_g_minus_x_on_interval : 
  (set.range (λ x, g x - x) ∩ set.Icc (-2 : ℝ) (2 : ℝ)) = set.Icc 0 16 :=
by
  sorry

end range_of_g_minus_x_on_interval_l416_416293


namespace proof_problem_l416_416431

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416431


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416950

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416950


namespace largest_power_of_2_divides_P2020_l416_416025

variable {P : ℤ → ℤ}
variable {n : ℕ}

-- Define the conditions
def P_conditions := ∀ i : ℕ, 0 ≤ i ∧ i ≤ 2018 → P i = Nat.choose 2018 i
def degree_condition := ∃ q : ℕ → ℤ, ∃ d : ℕ, (∀ x, P x = (polynomial.eval x).toFunction q) ∧ d ≤ 2018 

-- The main theorem to state
theorem largest_power_of_2_divides_P2020 (hP : P_conditions) (hdeg : degree_condition) :
  ∃ n : ℕ, 2^n ∣ P 2020 ∧ ∀ m : ℕ, 2^m ∣ P 2020 → m ≤ 6 :=
sorry

end largest_power_of_2_divides_P2020_l416_416025


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416734

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416734


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416675

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416675


namespace area_of_smaller_circle_l416_416135

noncomputable def radius_large_circle (x : ℝ) : ℝ := 2 * x
noncomputable def radius_small_circle (y : ℝ) : ℝ := y

theorem area_of_smaller_circle 
(pa ab : ℝ)
(r : ℝ)
(area : ℝ) 
(h1 : pa = 5) 
(h2 : ab = 5) 
(h3 : radius_large_circle r = 2 * radius_small_circle r)
(h4 : 2 * radius_small_circle r + radius_large_circle r = 10)
(h5 : area = Real.pi * (radius_small_circle r)^2) 
: area = 6.25 * Real.pi :=
by
  sorry

end area_of_smaller_circle_l416_416135


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416671

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416671


namespace product_of_consecutive_integers_l416_416756

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416756


namespace smallest_non_palindrome_power_of_12_l416_416345

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ¬is_palindrome (12^n) ∧
  ∀ m : ℕ, m > 0 → ¬is_palindrome (12^m) → 12^m ≥ 12^n :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num,
    -- Show that 144 is not a palindrome
    sorry },
  { intros m hm hnp,
    -- Show that for any other power of 12 that is not a palindrome, the result is ≥ 144
    -- essentially proving that 12^2 is the smallest such number.
    sorry }
end

end smallest_non_palindrome_power_of_12_l416_416345


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416876

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416876


namespace greatest_divisor_of_four_consecutive_integers_l416_416974

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416974


namespace vasya_numbers_l416_416146

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416146


namespace four_consecutive_integers_divisible_by_12_l416_416778

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416778


namespace remainder_of_c_plus_d_l416_416047

theorem remainder_of_c_plus_d (c d : ℕ) (k l : ℕ) 
  (hc : c = 120 * k + 114) 
  (hd : d = 180 * l + 174) : 
  (c + d) % 60 = 48 := 
by sorry

end remainder_of_c_plus_d_l416_416047


namespace divisor_of_four_consecutive_integers_l416_416654

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416654


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416667

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416667


namespace compound_interest_rate_solution_l416_416599

noncomputable def compound_interest_rate_problem : Prop :=
  let A : ℝ := 720
  let P : ℝ := 600
  let n : ℕ := 1
  let t : ℕ := 4
  let r := (real.exp (log (A / P) / (n * t)) - 1) in
  r = 0.04622

theorem compound_interest_rate_solution : compound_interest_rate_problem :=
by
  sorry

end compound_interest_rate_solution_l416_416599


namespace smallest_non_palindrome_power_of_12_l416_416346

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ¬is_palindrome (12^n) ∧
  ∀ m : ℕ, m > 0 → ¬is_palindrome (12^m) → 12^m ≥ 12^n :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num,
    -- Show that 144 is not a palindrome
    sorry },
  { intros m hm hnp,
    -- Show that for any other power of 12 that is not a palindrome, the result is ≥ 144
    -- essentially proving that 12^2 is the smallest such number.
    sorry }
end

end smallest_non_palindrome_power_of_12_l416_416346


namespace system_of_equations_has_two_solutions_l416_416111

theorem system_of_equations_has_two_solutions :
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  xy + yz = 63 ∧ 
  xz + yz = 23 :=
sorry

end system_of_equations_has_two_solutions_l416_416111


namespace greatest_divisor_four_consecutive_integers_l416_416829

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416829


namespace hexagon_probability_l416_416507

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l416_416507


namespace fraction_eq_l416_416578

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l416_416578


namespace dhoni_savings_l416_416309

theorem dhoni_savings
  (rent_percentage : ℕ)
  (dishwasher_percentage : ℕ)
  (bills_percentage : ℕ)
  (car_payments_percentage : ℕ)
  (grocery_percentage : ℕ)
  (h_rent : rent_percentage = 20)
  (h_dishwasher : dishwasher_percentage = 15)
  (h_bills : bills_percentage = 10)
  (h_car : car_payments_percentage = 8)
  (h_grocery : grocery_percentage = 12) :
  100 - (rent_percentage + dishwasher_percentage + bills_percentage + car_payments_percentage + grocery_percentage) = 35 :=
by {
  rw [h_rent, h_dishwasher, h_bills, h_car, h_grocery],
  norm_num,
  sorry -- Proof is omitted
}

end dhoni_savings_l416_416309


namespace abc_eq_bc_l416_416615

theorem abc_eq_bc (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
(h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c :=
by 
  sorry

end abc_eq_bc_l416_416615


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416949

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416949


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416960

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416960


namespace cheaper_to_buy_more_books_l416_416244

def C (n : ℕ) : ℕ :=
  if n < 1 then 0
  else if n ≤ 20 then 15 * n
  else if n ≤ 40 then 14 * n - 5
  else 13 * n

noncomputable def apply_discount (n : ℕ) (cost : ℕ) : ℕ :=
  cost - 10 * (n / 10)

theorem cheaper_to_buy_more_books : 
  ∃ (n_vals : Finset ℕ), n_vals.card = 5 ∧ ∀ n ∈ n_vals, apply_discount (n + 1) (C (n + 1)) < apply_discount n (C n) :=
sorry

end cheaper_to_buy_more_books_l416_416244


namespace quadrilateral_area_is_354_l416_416571

-- Define the points and conditions
variables (A B C D E : Type _) 
           [metric_space A] [inner_product_space ℝ A] [affine_space A A]
           [metric_space B] [inner_product_space ℝ B] [affine_space B B]
           [metric_space C] [inner_product_space ℝ C] [affine_space C C]
           [metric_space D] [inner_product_space ℝ D] [affine_space D D]
           [metric_space E] [inner_product_space ℝ E] [affine_space E E]

noncomputable theory

-- Conditions on the angles and side lengths
variables 
  (angle_ABC : ∠ A B C = 90)
  (angle_ACD : ∠ A C D = 90)
  (AC_length : dist A C = 20)
  (CD_length : dist C D = 30)
  (AE_length : dist A E = 8)
  (intersection_AC_BD : ∃ E, E ∈ line A C ∧ E ∈ line B D)

-- We are to prove that the area of quadrilateral ABCD is 354
def area_of_quadrilateral (A B C D : E) : ℝ :=
  let AC := dist A C,
      CD := dist C D in
  area_of_triangle A C D + area_of_triangle A B C

theorem quadrilateral_area_is_354 
  (h1 : angle_ABC)
  (h2 : angle_ACD)
  (h3 : AC_length)
  (h4 : CD_length)
  (h5 : AE_length)
  (h6 : intersection_AC_BD) :
  area_of_quadrilateral A B C D = 354 := 
sorry

end quadrilateral_area_is_354_l416_416571


namespace area_of_triangle_BQC_l416_416003

theorem area_of_triangle_BQC (A B C E Q I_B I_C O_3 O_4 : Point)
  (hAB : dist A B = 12) (hBC : dist B C = 15) (hCA : dist C A = 17)
  (hE_internal : E ∈ line_segment B C)
  (hI_B : incenter I_B A B E) (hI_C : incenter I_C A C E)
  (hCircum1 : circumcenter O_3 B I_B E) (hCircum2 : circumcenter O_4 C I_C E)
  (hIntersect : ∃ Q ≠ E, Q ∈ circumcircle B I_B E ∧ Q ∈ circumcircle C I_C E) :
  ∃ Q ≠ E, area (triangle B Q C) = 56.25 * Real.sqrt 3 := 
by sorry

end area_of_triangle_BQC_l416_416003


namespace line_circle_separate_l416_416561

theorem line_circle_separate (a : ℝ) (x₀ y₀ : ℝ) (h1 : a > 0) (h2 : x₀^2 + y₀^2 < a^2) :
(exists(c d : ℝ), c * x₀ + d * y₀ = a^2) → (a^2 / sqrt(x₀^2 + y₀^2)) > a := 
sorry

end line_circle_separate_l416_416561


namespace positive_difference_jo_kate_sum_l416_416017

theorem positive_difference_jo_kate_sum :
  let jo_sum := (100 * 101) / 2
  let rounded_sum := 5050 + 500 -- Based on the calculated sum of 5550
  in rounded_sum - jo_sum = 500 :=
by
  sorry

end positive_difference_jo_kate_sum_l416_416017


namespace mean_of_set_is_12_point_8_l416_416092

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l416_416092


namespace greatest_integer_less_than_neg_21_over_5_l416_416192

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ (z : ℤ), z < -21 / 5 ∧ ∀ (w : ℤ), w < -21 / 5 → w ≤ z :=
begin
  use -5,
  split,
  { norm_num },
  { intros w hw,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416192


namespace vasya_numbers_l416_416145

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416145


namespace problem_1_problem_2_problem_3_l416_416247

noncomputable def problem_1_cond (a b c d : ℕ) (n : ℕ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem problem_1 :
  let a := 60
  let b := 68
  let c := 40
  let d := 48
  let n := 216
  problem_1_cond a b c d n > 6.635 :=
by {
  let k2 := problem_1_cond a b c d n
  have : k2 > 6.635 := by norm_num; sorry
  exact this
}

theorem problem_2 :
  ∀ X : ℕ → ℕ → ℕ,
  X 0 0 = 27 ∧ X 1 0 = 54 ∧ X 2 0 = 36 ∧ X 3 0 = 8 :=
by {
  intros X,
  have H0 : X 0 0 = 27 := by norm_num; sorry,
  have H1 : X 1 0 = 54 := by norm_num; sorry,
  have H2 : X 2 0 = 36 := by norm_num; sorry,
  have H3 : X 3 0 = 8 := by norm_num; sorry,
  exact ⟨H0, H1, H2, H3⟩
}

theorem problem_3 (m : ℕ) :
  m ≤ 2 :=
by {
  sorry,
  -- Apply solver for inequality and natural number conditions.
}

end problem_1_problem_2_problem_3_l416_416247


namespace sin_eq_one_half_l416_416625

noncomputable def sin_value := 
  sin (-31 * Real.pi / 6)

theorem sin_eq_one_half : sin_value = 1 / 2 :=
by
  sorry

end sin_eq_one_half_l416_416625


namespace packs_with_extra_red_pencils_eq_3_l416_416496

def total_packs : Nat := 15
def regular_red_per_pack : Nat := 1
def total_red_pencils : Nat := 21
def extra_red_per_pack : Nat := 2

theorem packs_with_extra_red_pencils_eq_3 :
  ∃ (packs_with_extra : Nat), packs_with_extra * extra_red_per_pack + (total_packs - packs_with_extra) * regular_red_per_pack = total_red_pencils ∧ packs_with_extra = 3 :=
by
  sorry

end packs_with_extra_red_pencils_eq_3_l416_416496


namespace AE_eq_EC_l416_416387

variable (A B C D E F : Type)
variable [Metric (A B C D E F)]
variable [TriangleABC : IsIsoscelesRightTriangle A B C (λ (BC : Segment A C))]
variable [PointDOnBC : ∃ D, OnSegment D ('segment AB) ∧ (length (segment DB) = (1/3) * (length (segment AB)))]
variable [LineBEPerpAD : ∃ E, Perpendicular (line BE) (line AD) ∧ IntersectAt (line BE) (segment AC) E]

theorem AE_eq_EC :
  ∀ (A B C D E : Point), 
  IsIsoscelesRightTriangle A B C ∧ 
  PointOnSegment D B C ∧ length (Segment D C) = (1 / 3) * length (Segment B C) ∧
  Perpendicular (line B E) (line A D) ∧ PointOnSegment E A C := 
  AE = EC := 
sorry

end AE_eq_EC_l416_416387


namespace hexagon_probability_l416_416508

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l416_416508


namespace min_value_l416_416454

-- Given conditions
def on_line (a b : ℝ) : Prop :=
  b = sqrt 3 * a - sqrt 3

-- Distance from point M (-1, 0) to the line
def distance_to_line_squared (a b : ℝ) : ℝ :=
  (a + 1)^2 + b^2

theorem min_value (a b : ℝ) (h : on_line a b) : distance_to_line_squared a b = 3 := 
sorry

end min_value_l416_416454


namespace sheep_with_only_fleas_l416_416462

variable (S F L B : ℕ)

def problem_conditions :=
  (L + B = S / 2) ∧ (B = 84) ∧ (L + B = 94)

theorem sheep_with_only_fleas (h : problem_conditions S F L B) : F = 94 :=
sorry

end sheep_with_only_fleas_l416_416462


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416914

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416914


namespace four_consecutive_product_divisible_by_12_l416_416984

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416984


namespace greatest_divisor_of_four_consecutive_integers_l416_416820

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416820


namespace product_of_consecutive_integers_l416_416753

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416753


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416918

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416918


namespace angle_condition_l416_416315

theorem angle_condition
  {θ : ℝ}
  (h₀ : 0 ≤ θ)
  (h₁ : θ < π)
  (h₂ : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :
  0 < θ ∧ θ < π / 2 :=
by
  sorry

end angle_condition_l416_416315


namespace greatest_divisor_of_four_consecutive_integers_l416_416805

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416805


namespace simplify_fraction_to_9_l416_416070

-- Define the necessary terms and expressions
def problem_expr := (3^12)^2 - (3^10)^2
def problem_denom := (3^11)^2 - (3^9)^2
def simplified_expr := problem_expr / problem_denom

-- State the theorem we want to prove
theorem simplify_fraction_to_9 : simplified_expr = 9 := 
by sorry

end simplify_fraction_to_9_l416_416070


namespace greatest_divisor_of_four_consecutive_integers_l416_416979

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416979


namespace reflection_matrix_squared_identity_l416_416530

noncomputable def reflectionMatrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let a := (v.1^2 - v.2^2) / (v.1^2 + v.2^2)
  let b := (2 * v.1 * v.2) / (v.1^2 + v.2^2)
  ![![a, b], [b, -a]]

theorem reflection_matrix_squared_identity :
  let S := reflectionMatrix (2, -1)
  S * S = 1 := sorry

end reflection_matrix_squared_identity_l416_416530


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416929

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416929


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416681

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416681


namespace four_consecutive_integers_divisible_by_12_l416_416768

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416768


namespace hypotenuse_length_l416_416609

def longer_leg (shorter_leg : ℝ) : ℝ := 3 * shorter_leg - 2

def area (shorter_leg longer_leg : ℝ) : ℝ := 0.5 * shorter_leg * longer_leg

def hypotenuse (shorter_leg longer_leg : ℝ) : ℝ :=
  Real.sqrt (shorter_leg ^ 2 + longer_leg ^ 2)

def right_triangle (shorter_leg longer_leg : ℝ) := ∃ shorter_leg longer_leg, 
  longer_leg = 3 * shorter_leg - 2 ∧ 
  0.5 * shorter_leg * longer_leg = 90

theorem hypotenuse_length : 
  ∀ shorter_leg longer_leg, right_triangle shorter_leg longer_leg → 
  hypotenuse shorter_leg longer_leg ≈ 23.65 :=
by
  sorry

end hypotenuse_length_l416_416609


namespace greatest_divisor_four_consecutive_integers_l416_416839

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416839


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416953

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416953


namespace hexagon_same_length_probability_l416_416523

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l416_416523


namespace average_of_first_201_terms_l416_416294

def sequence (n : ℕ) : ℤ := (-1 : ℤ) ^ (n + 1) * n

theorem average_of_first_201_terms :
  (∑ i in Finset.range 201, sequence (i + 1)) / 201 = 101 / 201 := 
by
  sorry

end average_of_first_201_terms_l416_416294


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416670

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416670


namespace greatest_divisor_of_four_consecutive_integers_l416_416806

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416806


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416930

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416930


namespace evaluate_expression_l416_416448

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416448


namespace distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side_l416_416567

-- Let the geometry context be defined
variable (A B C H O : Point)
variable [triangle A B C]
variable [orthocenter H A B C]
variable [circumcenter O A B C]

-- Statement to prove
theorem distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side :
  dist A H = 2 * dist O (line B C) :=
begin
  sorry
end

end distance_vertex_orthocenter_twice_distance_circumcenter_opposite_side_l416_416567


namespace greatest_divisor_of_four_consecutive_integers_l416_416817

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416817


namespace inverse_condition_l416_416109

theorem inverse_condition (a : ℝ) :
  (∀ x1 x2, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → f a x1 = f a x2 → x1 = x2) ↔ a ∈ set.Icc (set.Iic 1) (set.Ici 2)
where
  f (a x : ℝ) : ℝ := x^2 - 2 * a * x - 3 :=
sorry

end inverse_condition_l416_416109


namespace triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint_l416_416569

-- Definitions
variable {A B C : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (G : A → B)
variable (H : A → A)
variable (O : A → A)

-- Theorem statement
theorem triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint 
  (homothety : ∀ (x : A), G (O x) = H x) 
  (midpoint_rotation : ∀ (A' B' C' : A), G (O (A')) = H (A')) 
  (distance_ratio: ∀ (x : A), dist O x = 2 * dist H (G x)) 
  (GA_AHAO: ∀ (x : A), dist G (A x) = (1/2) * dist G (H x) ) 
  : ∀ (x A' : A), dist x (H x) = 2 * dist (O x) (midpoint_rotation A' B' C') := 
by
  sorry

end triangle_vertex_to_orthocenter_twice_circumcenter_to_opposite_midpoint_l416_416569


namespace slope_of_line_l416_416622

theorem slope_of_line : ∀ x y : ℝ, (x + sqrt 3 * y + 1 = 0) → (∃ m : ℝ, m = - (sqrt 3) / 3) :=
by
  sorry

end slope_of_line_l416_416622


namespace range_of_b_over_a_l416_416367

noncomputable def g (a b x : ℝ) : ℝ := (a * x - b / x - 2 * a) * Real.exp x

noncomputable def g'' (a b x : ℝ) : ℝ := (a * (4 - x) * Real.exp x + b * (3 / x^3) * Real.exp x - Real.exp x * (6 a - 3a/x^2 + 3a/x)) -- Second derivative manually computed

theorem range_of_b_over_a (a b : ℝ) (h : a > 0) (x0 : ℝ) (h1 : 1 < x0) (h2 : g a b x0 + g'' a b x0 = 0) :
  -1 < b / a ∧ ∀ x > 0, b / a < x :=
sorry

end range_of_b_over_a_l416_416367


namespace greatest_divisor_of_four_consecutive_integers_l416_416977

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416977


namespace integer_values_of_x_for_equation_l416_416292

theorem integer_values_of_x_for_equation 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : a = b + c ∨ b = c + a ∨ c = b + a) : 
  ∃ x : ℤ, a * x + b = c :=
sorry

end integer_values_of_x_for_equation_l416_416292


namespace probability_of_same_length_segments_l416_416501

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l416_416501


namespace find_second_number_l416_416631

theorem find_second_number 
  (k m : ℤ) 
  (greatest_number : ℤ) 
  (first_number : ℤ) 
  (remainder1 remainder2 : ℤ) 
  (h1 : greatest_number = 127)
  (h2 : first_number = 1657)
  (h3 : remainder1 = 6)
  (h4 : remainder2 = 5)
  (div1 : first_number = k * greatest_number + remainder1)
  (div2 : ∃ x : ℤ, x = m * greatest_number + remainder2)
  (divK : k = 13)
  (m_calc : m = k - 1) :
  ∃ x : ℤ, x = 1529 :=
begin
  sorry,
end

end find_second_number_l416_416631


namespace marcias_hair_length_l416_416549

theorem marcias_hair_length (initial_length: ℕ) (first_cut_fraction: ℕ) (growth: ℕ) (second_cut: ℕ) :
  initial_length = 24 → first_cut_fraction = 2 → growth = 4 → second_cut = 2 →
  (initial_length / first_cut_fraction - second_cut + growth - second_cut) = 14 :=
begin
  sorry
end

end marcias_hair_length_l416_416549


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416936

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416936


namespace probability_different_colors_l416_416464

def total_chips := 7 + 5 + 4

def probability_blue_draw : ℚ := 7 / total_chips
def probability_red_draw : ℚ := 5 / total_chips
def probability_yellow_draw : ℚ := 4 / total_chips
def probability_different_color (color1_prob color2_prob : ℚ) : ℚ := color1_prob * (1 - color2_prob)

theorem probability_different_colors :
  (probability_blue_draw * probability_different_color 7 (7 / total_chips)) +
  (probability_red_draw * probability_different_color 5 (5 / total_chips)) +
  (probability_yellow_draw * probability_different_color 4 (4 / total_chips)) 
  = 83 / 128 := 
by 
  sorry

end probability_different_colors_l416_416464


namespace greatest_divisor_four_consecutive_integers_l416_416831

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416831


namespace greatest_divisor_four_consecutive_l416_416688

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416688


namespace range_of_a_l416_416416

open Set

variable {α : Type} [LinearOrder α] [FloorRing α] [TopOrder α]

theorem range_of_a (a : ℝ) :
  ( {x : ℝ | x > a} ∩ ({-1, 0, 1} : Set ℝ) = { 0, 1 } ) ↔ (-1 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l416_416416


namespace max_integer_solutions_l416_416355

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end max_integer_solutions_l416_416355


namespace angle_at_5_50_l416_416189

def hour_hand_position (hour : ℕ) (minute : ℕ) : ℝ :=
  let hour_deg := 30 * hour in
  let minute_contribution := (30.0 / 60.0) * minute in
  hour_deg + minute_contribution

def minute_hand_position (minute : ℕ) : ℝ :=
  6 * minute

noncomputable def angle_between_hands (hour : ℕ) (minute : ℕ) : ℝ :=
  let hour_pos := hour_hand_position hour minute in
  let minute_pos := minute_hand_position minute in
  abs (minute_pos - hour_pos)

theorem angle_at_5_50 : angle_between_hands 5 50 = 125 :=
by {
  sorry
}

end angle_at_5_50_l416_416189


namespace game_of_24_l416_416630

theorem game_of_24 : 
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  ((b + c / a) * d = 24) :=
by
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  show (b + c / a) * d = 24
  sorry

end game_of_24_l416_416630


namespace vasya_numbers_l416_416152

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416152


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416891

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416891


namespace Vasya_numbers_l416_416161

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416161


namespace projection_a_onto_b_is_neg_4_l416_416393

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
def magnitude_a : ℝ := 6
def magnitude_b : ℝ := 3
def dot_product_ab : ℝ := -12

-- Projection definition
def projection (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (dot_product_ab) / (magnitude_b)

theorem projection_a_onto_b_is_neg_4 : projection a b = -4 :=
by
  sorry

end projection_a_onto_b_is_neg_4_l416_416393


namespace distinct_four_digit_numbers_product_12_equals_36_l416_416618

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digits_product_equals (n : ℕ) (p : ℕ) : Prop :=
  let digits := (n.to_string.to_list.map (λ c, (c.to_nat - '0'.to_nat))) in
  digits.prod = p

theorem distinct_four_digit_numbers_product_12_equals_36 :
  {n : ℕ // is_four_digit n ∧ digits_product_equals n 12}.to_finset.card = 36 := 
sorry

end distinct_four_digit_numbers_product_12_equals_36_l416_416618


namespace pentagon_cosine_identity_l416_416228

    variable (A B C D E : Point)
    variable (circle : Circle)

    -- Given conditions
    variable (inscribed : Inscribed circle [A, B, C, D, E])
    variable (AB_eq : AB = 3)
    variable (BC_eq : BC = 3)
    variable (CD_eq : CD = 3)
    variable (DE_eq : DE = 3)
    variable (AE_eq : AE = 2)

    -- Goal: prove the given equation
    theorem pentagon_cosine_identity : 
      (1 - cos (∠ B)) * (1 - cos (∠ ACE)) = 1 / 9 := 
    by
      sorry
    
end pentagon_cosine_identity_l416_416228


namespace students_just_passed_l416_416217

theorem students_just_passed (total_students : ℕ) (first_division : ℕ) (second_division : ℕ) (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division = 26 * total_students / 100)
  (h3 : second_division = 54 * total_students / 100)
  (h4 : just_passed = total_students - (first_division + second_division)) :
  just_passed = 60 :=
sorry

end students_just_passed_l416_416217


namespace sasha_can_get_123_l416_416069

theorem sasha_can_get_123 : 
  ∃ (a b c d e : ℕ), 
    ((a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5) ∧ 
     (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5) ∧ 
     (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5) ∧ 
     (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) ∧ 
     (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 5) ∧ 
     (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
      c ≠ d ∧ c ≠ e ∧ 
      d ≠ e)) ∧ 
    (∃ (op1 op2 op3 op4 : char), 
      (op1 = '+' ∨ op1 = '-' ∨ op1 = '*') ∧ 
      (op2 = '+' ∨ op2 = '-' ∨ op2 = '*') ∧ 
      (op3 = '+' ∨ op3 = '-' ∨ op3 = '*') ∧ 
      (op4 = '+' ∨ op4 = '-' ∨ op4 = '*') ∧ 
      (eval_expression_with_parens a b c d e op1 op2 op3 op4) = 123) :=
sorry

end sasha_can_get_123_l416_416069


namespace c_6030_value_exists_l416_416500

theorem c_6030_value_exists (c : ℕ → ℝ) (a b : ℕ → ℝ) :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6030 →
   a n = ∑ k in finset.range (n + 1), b (nat.gcd k n)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6030 →
   b n = ∑ d in (nat.divisors n), c d * a (n / d)) →
  c 6030 = 528 :=
by
  sorry

end c_6030_value_exists_l416_416500


namespace divisor_of_product_of_four_consecutive_integers_l416_416709

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416709


namespace divisor_of_four_consecutive_integers_l416_416656

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416656


namespace vasya_numbers_l416_416148

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416148


namespace equal_angles_l416_416040

-- Definitions of basic geometric entities and conditions
variables {A B C M D K : Type} [linear_ordered_field K]

def is_midpoint (M A C : K) : Prop := M = (A + C) / 2
def is_parallel (l1 l2 : K) : Prop := ∃ m b, l1 = λ x, m * x + b ∧ l2 = λ x, m * x + b

-- The problem statement in Lean 4 (without the proof)
theorem equal_angles
  (ABC : Type)
  (h1 : A B C : ABC)
  (h2 : AB > AC)
  (M : is_midpoint M A C)
  (D : B on AB ∧ DB = DC)
  (K : is_parallel (line_through D BC) (BM intersects at K))
  : angle K C D = angle D A C :=
sorry

end equal_angles_l416_416040


namespace Vasims_share_l416_416278

-- Let the shares of Faruk, Vasim, and Ranjith be represented as 3x, 5x, and 9x respectively
-- Given that 9x - 3x = 1800, solve for Vasim's share 5x

theorem Vasims_share (x : ℕ) (hf : 3 * x) (hv : 5 * x) (hr : 9 * x) 
  (h_diff : hr - hf = 1800) : hv = 1500 :=
by
  -- The steps of the proof are omitted as we are only required to write the statement.
  sorry

end Vasims_share_l416_416278


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416730

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416730


namespace sum_of_shared_triangles_angles_l416_416118

theorem sum_of_shared_triangles_angles 
  (T1 T2 T3 : Triangle)
  (h1 : ∃ A1 A2 A3 : Angle, T1 = ⟨A1, A2, A3⟩)
  (h2 : ∃ B1 B2 B3 : Angle, T2 = ⟨B1, B2, B3⟩)
  (h3 : ∃ C1 C2 C3 : Angle, T3 = ⟨C1, C2, C3⟩)
  (shared_angles : ∃ x y z w v u : Angle, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ u ∧ x ≠ v ∧ x ≠ w ∧ y ≠ u ∧ y ≠ v ∧ y ≠ w ∧ z ≠ u ∧ z ≠ v ∧ z ≠ w ∧ u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ T1.has_angle x ∧ T1.has_angle y ∧ T1.has_angle z ∧ T2.has_angle u ∧ T2.has_angle v ∧ T2.has_angle w ∧ T3.has_angle x ∧ T3.has_angle y ∧ T3.has_angle z ∧ T1.has_angle u ∧ T1.has_angle y ∧ T1.has_angle z ∧ T2.has_angle u ∧ T2.has_angle v ∧ T2.has_angle w ∧ T3.has_angle x ∧ T3.has_angle y ∧ T3.has_angle z ∧ (x + y + z = 180) ∧ (u + v + w = 180) ∧ (x + y + z + u + v + w = 360)) : 
  (x + y + z + u + v + w = 360) := 
by
  sorry

end sum_of_shared_triangles_angles_l416_416118


namespace find_root_ln_x_minus_one_l416_416322

theorem find_root_ln_x_minus_one :
  ∃ x ∈ Ioo (2:ℝ) 3, (ln x - 1) = 0 :=
by
  have f_strict_mono : strict_mono (λ x:ℝ, ln x - 1) :=
    λ a b h, (Real.log_strict_mono h).sub_monotone 1
  have f_continuous : continuous (λ x:ℝ, ln x - 1) :=
    by continuity
  have f2_lt_0 : ln 2 - 1 < 0 :=
    by norm_num; linarith only [Real.log_pos 2 zero_lt_two, Real.log_lt_self zero_lt_two]
  have f3_gt_0 : ln 3 - 1 > 0 :=
    by norm_num; linarith only [Real.log_pos 3 (by norm_num), Real.log_lt_one_iff one_lt_two]
  have ivt := @intermediate_value_Ioo (λ x:ℝ, ln x - 1) 2 3 (λ _, f_continuous) f2_lt_0 f3_gt_0
  exact Set.exists_of_mem_subset (λ x hx, ⟨x, hx, ivt hx⟩) sorry

end find_root_ln_x_minus_one_l416_416322


namespace smallest_positive_non_palindromic_power_of_12_is_12_l416_416338

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def smallest_non_palindromic_power_of_12 : ℕ :=
  Nat.find (λ n, ∃ k : ℕ, n = 12^k ∧ ¬is_palindrome n)

theorem smallest_positive_non_palindromic_power_of_12_is_12 :
  smallest_non_palindromic_power_of_12 = 12 :=
by
  sorry

end smallest_positive_non_palindromic_power_of_12_is_12_l416_416338


namespace mean_days_jogged_l416_416054

open Real

theorem mean_days_jogged 
  (p1 : ℕ := 5) (d1 : ℕ := 1)
  (p2 : ℕ := 4) (d2 : ℕ := 3)
  (p3 : ℕ := 10) (d3 : ℕ := 5)
  (p4 : ℕ := 7) (d4 : ℕ := 10)
  (p5 : ℕ := 3) (d5 : ℕ := 15)
  (p6 : ℕ := 1) (d6 : ℕ := 20) : 
  ( (p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6) / (p1 + p2 + p3 + p4 + p5 + p6) : ℝ) = 6.73 :=
by
  sorry

end mean_days_jogged_l416_416054


namespace quadratic_function_range_l416_416619

theorem quadratic_function_range (x : ℝ) (h : x ≥ 0) : 
  3 ≤ x^2 + 2 * x + 3 :=
by {
  sorry
}

end quadratic_function_range_l416_416619


namespace polynomial_decomposition_l416_416298

theorem polynomial_decomposition :
  (x^3 - 2*x^2 + 3*x + 5) = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 :=
by sorry

end polynomial_decomposition_l416_416298


namespace expected_value_unfair_die_correct_l416_416049

noncomputable def expected_value_unfair_die : ℚ :=
  (2 / 15) * (1 + 2 + 3 + 4 + 5 + 6 + 7) + (1 / 3) * 8

theorem expected_value_unfair_die_correct :
  expected_value_unfair_die = 6.4 :=
by
  rw [expected_value_unfair_die]
  have h1 : (2 / 15 : ℚ) * 28 = 56 / 15 := by norm_num
  have h2 : (1 / 3 : ℚ) * 8 = 8 / 3 := by norm_num
  have h3 : 56 / 15 + 8 / 3 = 96 / 15 := by norm_num
  have h4 : 96 / 15 = 32 / 5 := by norm_num
  have h5 : 32 / 5 = 6.4 := by norm_num
  exact Eq.trans (Eq.trans (Eq.trans (Eq.trans h1 h2) h3) h4) h5

end expected_value_unfair_die_correct_l416_416049


namespace vasya_numbers_l416_416149

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416149


namespace pentagon_cosine_identity_l416_416229

    variable (A B C D E : Point)
    variable (circle : Circle)

    -- Given conditions
    variable (inscribed : Inscribed circle [A, B, C, D, E])
    variable (AB_eq : AB = 3)
    variable (BC_eq : BC = 3)
    variable (CD_eq : CD = 3)
    variable (DE_eq : DE = 3)
    variable (AE_eq : AE = 2)

    -- Goal: prove the given equation
    theorem pentagon_cosine_identity : 
      (1 - cos (∠ B)) * (1 - cos (∠ ACE)) = 1 / 9 := 
    by
      sorry
    
end pentagon_cosine_identity_l416_416229


namespace sum_of_interior_angles_l416_416082

theorem sum_of_interior_angles (n : ℕ) (h₁ : 180 * (n - 2) = 2340) : 
  180 * ((n - 3) - 2) = 1800 := by
  -- Here, we'll solve the theorem using Lean's capabilities.
  sorry

end sum_of_interior_angles_l416_416082


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416900

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416900


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416924

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416924


namespace sequence_sum_positive_l416_416374

theorem sequence_sum_positive
  (n : ℕ)
  (a : ℕ → ℝ)
  (h_sum : (∑ i in finRange n, a (i + 1)) = 0)
  (h_not_all_zero : ∃ i, 1 ≤ i ∧ i ≤ n ∧ a i ≠ 0)
  (h_exists_k : ∃ k, 1 ≤ k ∧ k ≤ n ∧ (∀ j, j ≤ k → a j ≤ 0) ∧ (∀ j, k < j → j ≤ n → a j ≥ 0)) :
  (∑ i in finRange n, (i + 1) * a (i + 1)) > 0 :=
by
  sorry

end sequence_sum_positive_l416_416374


namespace factorial_ratio_l416_416200

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end factorial_ratio_l416_416200


namespace proof_problem_l416_416434

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416434


namespace car_rental_cost_per_mile_l416_416246

theorem car_rental_cost_per_mile:
  (total_paid daily_rate miles_driven cost_per_mile : ℝ) 
  (h1 : daily_rate = 29) 
  (h2 : total_paid = 46.12)
  (h3 : miles_driven = 214) 
  (h4 : cost_per_mile = (total_paid - daily_rate) / miles_driven) :
  cost_per_mile = 0.08 :=
by
  sorry

end car_rental_cost_per_mile_l416_416246


namespace volume_of_sphere_circumscribing_rectangular_solid_l416_416398

-- Definitions of the dimensions of the rectangular solid.
def length := 2
def width := 1
def height := 2

-- Definition of the space diagonal of the rectangular solid
def space_diagonal : Real := Real.sqrt (length^2 + width^2 + height^2)

-- Definition of the radius of the circumscribing sphere
def radius_sphere : Real := space_diagonal / 2

-- Volume of the sphere calculated using the formula for the volume of a sphere
def volume_sphere : Real := (4 / 3) * Real.pi * radius_sphere^3

-- Theorem: volume of the sphere that circumscribes the rectangular solid is \frac{9}{2} \pi
theorem volume_of_sphere_circumscribing_rectangular_solid : volume_sphere = (9 / 2) * Real.pi :=
by
  sorry

end volume_of_sphere_circumscribing_rectangular_solid_l416_416398


namespace greatest_divisor_of_consecutive_product_l416_416849

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416849


namespace inscribed_polygon_cosine_l416_416223

noncomputable def angle_B (ABC : ℝ) : ℝ := 
  let B := 18 (1 - Mathlib.cos ABC) in
  B

noncomputable def angle_ACE (AC : ℝ) : ℝ :=
  let ACE := 2*AC^2 * (1 - Mathlib.cos AC) = 4 in
  ACE

theorem inscribed_polygon_cosine :
  ∀ (A B C D E : ℝ), A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧ D ∈ Circle ∧ E ∈ Circle ∧
    (AB = 3) ∧ (BC = 3) ∧ (CD = 3) ∧ (DE = 3) ∧ (AE = 2) →
    (1 - Mathlib.cos (angle_B 3)) * (1 - Mathlib.cos (angle_ACE 3)) = (1 / 9) :=
  by
  sorry

end inscribed_polygon_cosine_l416_416223


namespace cosine_product_l416_416227

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l416_416227


namespace problem_1_problem_2_l416_416044

-- Define the propositions p and q
def proposition_p (x a : ℝ) := x^2 - (a + 1/a) * x + 1 < 0
def proposition_q (x : ℝ) := x^2 - 4 * x + 3 ≤ 0

-- Problem 1: Given a = 2 and both p and q are true, find the range of x
theorem problem_1 (a : ℝ) (x : ℝ) (ha : a = 2) (hp : proposition_p x a) (hq : proposition_q x) :
  1 ≤ x ∧ x < 2 :=
sorry

-- Problem 2: Prove that if p is a necessary but not sufficient condition for q, then 3 < a
theorem problem_2 (a : ℝ)
  (h_ns : ∀ x, proposition_q x → proposition_p x a)
  (h_not_s : ∃ x, ¬ (proposition_q x → proposition_p x a)) :
  3 < a :=
sorry

end problem_1_problem_2_l416_416044


namespace complex_div_conjugate_l416_416442

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416442


namespace translation_cos_to_sin_l416_416120

noncomputable def f : ℝ → ℝ := λ x, Real.cos (2 * x + Real.pi / 3)
noncomputable def g : ℝ → ℝ := λ x, Real.sin (2 * x)
noncomputable def h : ℝ → ℝ := λ x, g (x + 5 * Real.pi / 12)

theorem translation_cos_to_sin :
  ∀ x, f x = h x := by
  sorry

end translation_cos_to_sin_l416_416120


namespace avg_first_18_even_ap_l416_416320

theorem avg_first_18_even_ap : 
  let a := 4;
  let d := 6;
  let T : ℕ → ℕ := λ n, a + (n - 1) * d;
  let T1 := T 1;
  let T18 := T 18;
  (T1 + T18) / 2 = 55 :=
by
  sorry

end avg_first_18_even_ap_l416_416320


namespace moment_of_inertia_proof_l416_416283

-- Parameters for the cylinder
variables (R H k : ℝ)

-- Definition of the moment of inertia I_x given the conditions
noncomputable def moment_of_inertia_cylinder_midsection (R H k : ℝ) : ℝ :=
  k * real.pi * H * R^2 * (2/3 * H^2 + 1/2 * R^2)

-- The theorem to prove
theorem moment_of_inertia_proof : 
  moment_of_inertia_cylinder_midsection R H k = k * real.pi * H * R^2 * (2/3 * H^2 + 1/2 * R^2) :=
by
  sorry

end moment_of_inertia_proof_l416_416283


namespace product_of_consecutive_integers_l416_416760

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416760


namespace max_quad_int_solutions_l416_416353

theorem max_quad_int_solutions :
  ∃ (a b c : ℤ), (∀ n : ℤ, n ∈ {0, 1, 2, -3, 4}) ∧
  ∀ (p : ℤ → ℤ), 
    p(x) = ax^2 + bx + c →
      ∃ n, p(n) = p(n^2) :=
begin
  sorry
end

end max_quad_int_solutions_l416_416353


namespace isosceles_right_triangle_mid_segment_l416_416388

open EuclideanGeometry

theorem isosceles_right_triangle_mid_segment (A B C D E : Point) 
(h_triangle : isIsoscelesRightTriangle A B C)
(hD : isOnLineSegment D B C (1 / 3))
(hE : isPerpendicular BE AD ∧ pointOnLineSegment E A C) :
  isEqualSegment AE EC :=
sorry

end isosceles_right_triangle_mid_segment_l416_416388


namespace vasya_numbers_l416_416177

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416177


namespace evaluate_expression_l416_416443

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416443


namespace sum_area_of_R_eq_20_l416_416255

noncomputable def sum_m_n : ℝ := 
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  let m := 20
  let n := 12 * Real.sqrt 2
  m + n

theorem sum_area_of_R_eq_20 :
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  area_R = 20 + 12 * Real.sqrt 2 :=
by
  sorry

end sum_area_of_R_eq_20_l416_416255


namespace coin_flip_sequences_l416_416250

theorem coin_flip_sequences (n : ℕ) (h : n = 10) :
  let total_sequences := 2 ^ (n - 2)
  in total_sequences = 256 :=
by
  -- The proof is skipped here
  sorry

end coin_flip_sequences_l416_416250


namespace polynomial_product_roots_l416_416323

theorem polynomial_product_roots (a b c : ℝ) : 
  (∀ x, (x - (Real.sin (Real.pi / 6))) * (x - (Real.sin (Real.pi / 3))) * (x - (Real.sin (5 * Real.pi / 6))) = x^3 + a * x^2 + b * x + c) → 
  a * b * c = Real.sqrt 3 / 2 :=
by
  sorry

end polynomial_product_roots_l416_416323


namespace first_book_word_count_l416_416493

def reading_conditions (total_days : ℕ) (reading_speed : ℕ) (avg_minutes_per_day : ℕ)
  (second_book_words : ℕ) (third_book_words : ℕ) (first_book_words : ℕ) : Prop :=
  let total_minutes := total_days * avg_minutes_per_day in
  let total_hours := total_minutes / 60 in
  let total_readable_words := total_hours * reading_speed in
  total_readable_words = first_book_words + second_book_words + third_book_words

theorem first_book_word_count :
  reading_conditions 10 100 54 400 300 200 :=
by
  dsimp [reading_conditions]
  sorry

end first_book_word_count_l416_416493


namespace trigonometric_identity_simplification_l416_416313

theorem trigonometric_identity_simplification :
  (sin (10 * degree) / (1 - (sqrt 3 * tan (10 * degree)))) = 1 / 2 :=
by
  sorry

end trigonometric_identity_simplification_l416_416313


namespace negation_of_existence_implies_universal_l416_416099

theorem negation_of_existence_implies_universal :
  ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_existence_implies_universal_l416_416099


namespace div_product_four_consecutive_integers_l416_416787

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416787


namespace ticket_representation_l416_416452

-- Define a structure for representing a movie ticket
structure Ticket where
  rows : Nat
  seats : Nat

-- Define the specific instance of representing 7 rows and 5 seats
def ticket_7_5 : Ticket := ⟨7, 5⟩

-- The theorem stating our problem: the representation of 7 rows and 5 seats is (7,5)
theorem ticket_representation : ticket_7_5 = ⟨7, 5⟩ :=
  by
    -- Proof goes here (omitted as per instructions)
    sorry

end ticket_representation_l416_416452


namespace greatest_divisor_four_consecutive_integers_l416_416832

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416832


namespace greatest_divisor_of_four_consecutive_integers_l416_416810

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416810


namespace four_consecutive_product_divisible_by_12_l416_416997

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416997


namespace greatest_divisor_four_consecutive_integers_l416_416822

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416822


namespace time_after_seconds_l416_416010

def initialTime := Time.mk 10 15 30
def elapsedSeconds := 9999
def expectedTime := Time.mk 13 2 9

theorem time_after_seconds :
  initialTime.addSeconds elapsedSeconds = expectedTime :=
by
  sorry

end time_after_seconds_l416_416010


namespace compute_binomial_coefficient_l416_416390

theorem compute_binomial_coefficient :
  (nat.choose 19 9 = 92378) ∧
  (nat.choose 18 9 = 43758) ∧
  (nat.choose 18 10 = 43758) →
  nat.choose 20 10 = 179894 := by
  sorry

end compute_binomial_coefficient_l416_416390


namespace fraction_eq_l416_416579

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l416_416579


namespace prove_VQ_over_QD_l416_416499

variables (a b c d : ℝ) (V A B C D M N P Q : Point) (λ : Plane)

-- Definitions for the given conditions
def is_pyramid (V A B C D : Point) : Prop :=
  ∃ λ, ∀ P,
  (P = V ∨ P = A ∨ P = B ∨ P = C ∨ P = D) ∧ 
  coplanar {V, A, B, C, D}

def intersects_ratios (V M A N B P C Q D : Point) : Prop :=
  divides_ratio V M A 2 3 ∧
  divides_ratio V N B 1 2 ∧
  divides_ratio V P C 1 3

def divides_ratio (X Y Z : Point) (p q : ℝ) : Prop :=
  dist X Y = p / (p + q) * dist X Z

-- Problem statement
theorem prove_VQ_over_QD (a b c d V A B C D M N P Q : Point) (λ : Plane) 
  (hp : is_pyramid V A B C D)
  (hi : intersects_ratios V M A N B P C Q D) :
  divides_ratio V Q D 2 3 :=
sorry

end prove_VQ_over_QD_l416_416499


namespace solution_x_x_sub_1_eq_x_l416_416108

theorem solution_x_x_sub_1_eq_x (x : ℝ) : x * (x - 1) = x ↔ (x = 0 ∨ x = 2) :=
by {
  sorry
}

end solution_x_x_sub_1_eq_x_l416_416108


namespace sequence_general_term_l416_416607

theorem sequence_general_term (a : ℕ → ℤ) (h : ∀ n, a n = (-1)^{n+1} * (4 * n - 1)) :
  (a 1 = 3) ∧ (a 2 = -7) ∧ (a 3 = 11) ∧ (a 4 = -15) :=
by
  sorry

end sequence_general_term_l416_416607


namespace greatest_divisor_four_consecutive_l416_416691

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416691


namespace percentage_of_x_eq_y_l416_416449

theorem percentage_of_x_eq_y
  (x y : ℝ) 
  (h : 0.60 * (x - y) = 0.20 * (x + y)) :
  y = 0.50 * x := 
sorry

end percentage_of_x_eq_y_l416_416449


namespace four_consecutive_integers_divisible_by_12_l416_416764

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416764


namespace Berlov_inequality_l416_416061

-- Definitions of points and relations
variable (A B C D M N : Point)
variable [Geometry]

-- Given conditions
def midpoint (N A D : Point) : Prop := 
  dist A N = dist N D

def perpendicular (CM BD : Line) : Prop := 
  angle CM BD = 90

def bm_gt_ma (BM MA : ℝ) : Prop :=
  BM > MA

-- The problem statement
theorem Berlov_inequality (A B C D M N : Point) [Geometry]
  (h1 : midpoint N A D)
  (h2 : perpendicular (line_through_command C M) (line_through_command B D))
  (h3 : dist B M > dist M A) :
  2 * dist B C + dist A D > 2 * dist C N :=
by
  sorry

end Berlov_inequality_l416_416061


namespace geometric_seq_arithmetic_condition_l416_416468

open Real

noncomputable def common_ratio (q : ℝ) := (q > 0) ∧ (q^2 - q - 1 = 0)

def arithmetic_seq_condition (a1 a2 a3 : ℝ) := (a2 = (a1 + a3) / 2)

theorem geometric_seq_arithmetic_condition (a1 a2 a3 a4 a5 : ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : q^2 - q - 1 = 0)
  (h3 : a2 = q * a1)
  (h4 : a3 = q * a2)
  (h5 : a4 = q * a3)
  (h6 : a5 = q * a4)
  (h7 : arithmetic_seq_condition a1 a2 a3) :
  (a4 + a5) / (a3 + a4) = (1 + sqrt 5) / 2 := 
sorry

end geometric_seq_arithmetic_condition_l416_416468


namespace smallest_positive_period_of_transformed_function_l416_416134

open Real

def original_function (x : ℝ) : ℝ := sin x + cos x

def transformed_function (x : ℝ) : ℝ := sin (2*x) + cos (2*x)

theorem smallest_positive_period_of_transformed_function :
  ∃ T > 0, ∀ x, transformed_function (x + T) = transformed_function x ∧
               (∀ T' > 0, (∀ x, transformed_function (x + T') = transformed_function x) → T ≤ T') :=
sorry

end smallest_positive_period_of_transformed_function_l416_416134


namespace greatest_divisor_of_four_consecutive_integers_l416_416969

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416969


namespace four_consecutive_product_divisible_by_12_l416_416999

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416999


namespace parabola_equation_l416_416544

theorem parabola_equation (p : ℝ) (h1 : p > 0) (F : ℝ) (h2 : F = p / 4) (M : ℝ × ℝ) 
  (h3 : M.1^2 = p * M.2) (h4 : real.dist M F = 3) (h5 : ∀ (x : ℝ), (x, 0) ∈ set_of (λ x, real.dist (x, 0) (M / 2, 3 / 2) = 3 / 2)) :
  (p = 4 ∨ p = 8) :=
sorry

end parabola_equation_l416_416544


namespace div_product_four_consecutive_integers_l416_416792

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416792


namespace new_value_f1_l416_416076

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_f : ∀ x : ℝ, f(x + 2) = f(x)
axiom functional_f : ∀ x : ℝ, 0 < x ∧ x < 1 → f(x) = x^2

theorem new_value_f1 (a : ℝ) : f(-3/2) + a = -1/4 + a :=
by {
  -- Proof steps skipped
  sorry
}

end new_value_f1_l416_416076


namespace rolls_per_pack_is_12_l416_416559

/-- Stella stocks 1 roll of toilet paper per bathroom per day. -/
def rolls_per_bathroom_per_day : Nat := 1

/-- Stella stocks toilet papers in 6 bathrooms. -/
def number_of_bathrooms : Nat := 6

/-- Stella restocks every day of the week. -/
def days_per_week : Nat := 7

/-- Stella restocks for 4 weeks before making a large purchase. -/
def weeks : Nat := 4

/-- Stella buys 14 packs of toilet paper after 4 weeks. -/
def packs_bought : Nat := 14

/-- The total number of rolls Stella uses in a day. -/
def daily_rolls : Nat := rolls_per_bathroom_per_day * number_of_bathrooms

/-- The total number of rolls Stella uses in a week. -/
def weekly_rolls : Nat := daily_rolls * days_per_week

/-- The total number of rolls Stella uses in 4 weeks. -/
def total_rolls : Nat := weekly_rolls * weeks

/-- Calculate the number of rolls per pack. -/
def rolls_per_pack : Nat := total_rolls / packs_bought

theorem rolls_per_pack_is_12 : rolls_per_pack = 12 := by
  unfold rolls_per_pack daily_rolls weekly_rolls total_rolls
  calc
    rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks / packs_bought
      = 1 * 6 * 7 * 4 / 14 := by rfl
      _ = 168 / 14 := by norm_num
      _ = 12 := by norm_num

end rolls_per_pack_is_12_l416_416559


namespace find_a_value_l416_416364

theorem find_a_value (a : ℝ) (A B : Set ℝ) (hA : A = {3, 5}) (hB : B = {x | a * x - 1 = 0}) :
  B ⊆ A → a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by sorry

end find_a_value_l416_416364


namespace greatest_divisor_four_consecutive_l416_416701

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416701


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416898

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416898


namespace cost_to_fill_pool_l416_416126

-- Define the given conditions as constants
def filling_time : ℝ := 50
def flow_rate : ℝ := 100
def cost_per_10_gallons : ℝ := 0.01

-- Calculate total volume in gallons
def total_volume : ℝ := filling_time * flow_rate

-- Calculate the cost per gallon in dollars
def cost_per_gallon : ℝ := cost_per_10_gallons / 10

-- Define the total cost to fill the pool in dollars
def total_cost : ℝ := total_volume * cost_per_gallon

-- Prove that the total cost equals $5
theorem cost_to_fill_pool : total_cost = 5 := by
  unfold total_cost
  unfold total_volume
  unfold cost_per_gallon
  unfold filling_time
  unfold flow_rate
  unfold cost_per_10_gallons
  sorry

end cost_to_fill_pool_l416_416126


namespace smallest_positive_non_palindrome_power_of_12_l416_416341

def is_palindrome (n : ℕ) : Bool :=
  let s := toDigits 10 n
  s = s.reverse

theorem smallest_positive_non_palindrome_power_of_12 : ∃ k : ℕ, k > 0 ∧ (12^k = 12 ∧ ¬ is_palindrome (12^k)) :=
by {
  sorry
}

end smallest_positive_non_palindrome_power_of_12_l416_416341


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416943

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416943


namespace p_implies_q_p_iff_q_l416_416381

variable {m : ℝ}

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (m + 3)^x < (m + 3)^y

theorem p_implies_q : ∀ m : ℝ, p m → q m :=
by
  sorry

theorem p_iff_q : ∀ m : ℝ, p m ↔ q m :=
by
  sorry

example : ∀ m : ℝ, (p m ⊆ q m) ∧ (p m ≠ q m) :=
by
  intros
  split
  · exact p_implies_q m (by assumption)
  · exact λ h, by
    sorry

end p_implies_q_p_iff_q_l416_416381


namespace problem_solution_l416_416428

noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 - p * x + q

theorem problem_solution
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : p > 0)
  (h3 : q > 0)
  (h4 : f a p q = 0)
  (h5 : f b p q = 0)
  (h6 : ∃ k : ℝ, (a = -2 + k ∧ b = -2 - k) ∨ (a = -2 - k ∧ b = -2 + k))
  (h7 : ∃ l : ℝ, (a = -2 * l ∧ b = 4 * l) ∨ (a = 4 * l ∧ b = -2 * l))
  : p + q = 9 :=
sorry

end problem_solution_l416_416428


namespace probability_of_same_length_segments_l416_416515

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l416_416515


namespace kendra_more_buttons_l416_416497

theorem kendra_more_buttons {K M S : ℕ} (hM : M = 8) (hS : S = 22) (hHalfK : S = K / 2) :
  K - 5 * M = 4 :=
by
  sorry

end kendra_more_buttons_l416_416497


namespace greatest_divisor_of_four_consecutive_integers_l416_416968

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416968


namespace terminal_side_of_angle_l416_416383

theorem terminal_side_of_angle (θ : Real) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ > 0) :
  θ ∈ {φ : Real | π < φ ∧ φ < 3 * π / 2} :=
sorry

end terminal_side_of_angle_l416_416383


namespace div_product_four_consecutive_integers_l416_416782

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416782


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416668

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416668


namespace greatest_divisor_four_consecutive_l416_416689

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416689


namespace four_consecutive_integers_divisible_by_12_l416_416779

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416779


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416663

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416663


namespace greatest_divisor_of_consecutive_product_l416_416851

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416851


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416740

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416740


namespace greatest_integer_less_than_neg_21_over_5_l416_416193

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ (z : ℤ), z < -21 / 5 ∧ ∀ (w : ℤ), w < -21 / 5 → w ≤ z :=
begin
  use -5,
  split,
  { norm_num },
  { intros w hw,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416193


namespace complex_div_conjugate_l416_416441

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416441


namespace divisor_of_product_of_four_consecutive_integers_l416_416717

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416717


namespace hexagon_probability_l416_416510

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l416_416510


namespace hexagon_same_length_probability_l416_416520

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l416_416520


namespace divisor_of_four_consecutive_integers_l416_416648

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416648


namespace M_computation_l416_416041

open Matrix

-- Let 𝔽 be the field of real numbers, and let matrix and vector be over 𝔽
variables {𝔽 : Type*} [Field 𝔽]
variables {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
variables {M : Matrix m n 𝔽} {v w : Matrix n (Fin 1) 𝔽}

-- Define the given conditions
def Mv_condition : Prop := M.mul_vec v = ![5, -1]
def Mw_condition : Prop := M.mul_vec w = ![-1, 4]

-- The main goal to prove
theorem M_computation (h1 : Mv_condition) (h2 : Mw_condition) :
  M.mul_vec (2 • v - w) = (![11, -6] : Matrix m (Fin 1) 𝔽) :=
sorry

end M_computation_l416_416041


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416955

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416955


namespace inverse_proportion_function_pass_through_l416_416457

theorem inverse_proportion_function_pass_through (k : ℝ) (hk : k ≠ 0)
  (h₁ : (1, -2) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  (-2, 1) ∈ {p : ℝ × ℝ | p.snd = k / p.fst} :=
by
  rw Set.mem_setOf_eq at h₁ ⊢
  rcases h₁ with rfl
  sorry    

end inverse_proportion_function_pass_through_l416_416457


namespace triangle_incenter_length_l416_416007

theorem triangle_incenter_length 
  (J L M N P Q R : Type)
  [Inhabited P] [Inhabited Q] [Inhabited R]
  [Inhabited J] [Inhabited L] [Inhabited M] [Inhabited N]
  (PQ PR QR : ℝ)
  (hPQ : PQ = 17)
  (hPR : PR = 19)
  (hQR : QR = 20)
  (incenter : Incenter J P Q R)
  (touches : IncircleTouchPoints J P Q R L M N) :
  let QL := 9,
      r  := 6
  in (dist Q J = 3 * sqrt 13) :=
by
  sorry

end triangle_incenter_length_l416_416007


namespace greatest_divisor_of_consecutive_product_l416_416857

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416857


namespace max_extreme_point_interval_l416_416403

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2)*exp(x) - (1/3)*x^3 - (1/2)*x^2

theorem max_extreme_point_interval (x₀ : ℝ) (n : ℤ) (hn : n = 0) :
  (∀ x, f x = (x^2 - 2*x + 2)*exp(x) - (1/3)*x^3 - (1/2)*x^2) → 
  (x₀ ∈ set_of (λ x : ℝ, f' x = 0) ∧ (∀ y ∈ set_of (λ x : ℝ, f' x = 0), f y ≤ f x₀)) →
  (x₀ ∈ set.Ioo ↑n (↑n + 1)) :=
begin
  sorry
end

end max_extreme_point_interval_l416_416403


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416899

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416899


namespace jo_kate_sum_difference_l416_416016

theorem jo_kate_sum_difference :
  let S_J := (100 * 101) / 2,
      S_K := 60 + 160 + 260 + 360 + 460 + 560 + 660 + 760 + 860 + 960
  in abs (S_J - S_K) = 150 := 
by 
  let S_J := (100 * 101) / 2
  let S_K := 60 + 160 + 260 + 360 + 460 + 560 + 660 + 760 + 860 + 960
  sorry

end jo_kate_sum_difference_l416_416016


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416919

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416919


namespace div_product_four_consecutive_integers_l416_416789

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416789


namespace polynomial_divisor_l416_416360

-- Define the requirements and the main problem statement
theorem polynomial_divisor (a : ℤ) : 
  a = 2 → ∃ (p : Polynomial ℤ), (Polynomial.X^17 + Polynomial.X^2 + 45) = (Polynomial.X^2 - 2 * Polynomial.X + a) * p :=
by {
  assume h : a = 2,
  use Polynomial.of_coefficients [terms that form the polynomial p],
  sorry -- proof details go here
}

end polynomial_divisor_l416_416360


namespace find_value_of_a_l416_416265

theorem find_value_of_a (a : ℝ) (h: (1 + 3 + 2 + 5 + a) / 5 = 3) : a = 4 :=
by
  sorry

end find_value_of_a_l416_416265


namespace complex_div_conjugate_l416_416440

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416440


namespace product_of_consecutive_integers_l416_416752

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416752


namespace four_consecutive_integers_divisible_by_12_l416_416775

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416775


namespace smallest_digit_divisible_by_7_l416_416306

theorem smallest_digit_divisible_by_7 :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 9) ∧ (52 * 10^2 + x * 10 + 4) % 7 = 0 ∧
  ∀ y : ℕ, (0 ≤ y ∧ y ≤ 9) → (52 * 10^2 + y * 10 + 4) % 7 = 0 → x ≤ y :=
begin
  use 2,
  split,
  { exact ⟨le_refl 2, nat.le_succ 8⟩, },
  split,
  { norm_num, },
  { intros y hy hp,
    interval_cases y; linarith [hp] },
end

end smallest_digit_divisible_by_7_l416_416306


namespace smallest_positive_non_palindromic_power_of_12_is_12_l416_416339

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def smallest_non_palindromic_power_of_12 : ℕ :=
  Nat.find (λ n, ∃ k : ℕ, n = 12^k ∧ ¬is_palindrome n)

theorem smallest_positive_non_palindromic_power_of_12_is_12 :
  smallest_non_palindromic_power_of_12 = 12 :=
by
  sorry

end smallest_positive_non_palindromic_power_of_12_is_12_l416_416339


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416896

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416896


namespace coupons_price_diff_l416_416267

theorem coupons_price_diff (P : ℝ) (hP : P > 150) (p : ℝ) (hp_pos : p > 0) (hP_eq : P = 150 + p) (hp_range : 100 ≤ p ∧ p ≤ 300) : 
  let x := 150 + 100 in
  let y := 150 + 300 in
  y - x = 200 := 
by 
  let x := 150 + 100;
  let y := 150 + 300;
  have : y - x = 200 := sorry;
  exact this

end coupons_price_diff_l416_416267


namespace greatest_divisor_of_consecutive_product_l416_416848

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416848


namespace proof_problem_l416_416432

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416432


namespace four_consecutive_integers_divisible_by_12_l416_416767

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416767


namespace div_product_four_consecutive_integers_l416_416785

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416785


namespace find_cylinder_radius_l416_416263

-- Define the problem conditions
def cone_diameter := 10
def cone_altitude := 12
def cylinder_height_eq_diameter (r: ℚ) := 2 * r

-- Define the cone and cylinder inscribed properties
noncomputable def inscribed_cylinder_radius (r : ℚ) : Prop :=
  (cylinder_height_eq_diameter r) ≤ cone_altitude ∧
  2 * r ≤ cone_diameter ∧
  cone_altitude - cylinder_height_eq_diameter r = (cone_altitude * r) / (cone_diameter / 2)

-- The proof goal
theorem find_cylinder_radius : ∃ r : ℚ, inscribed_cylinder_radius r ∧ r = 30/11 :=
by
  sorry

end find_cylinder_radius_l416_416263


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416723

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416723


namespace vasya_numbers_l416_416176

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416176


namespace div_product_four_consecutive_integers_l416_416794

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416794


namespace license_plate_count_l416_416256

-- Define the conditions
def num_digits : ℕ := 5
def num_letters : ℕ := 2
def digit_choices : ℕ := 10
def letter_choices : ℕ := 26

-- Define the statement to prove the total number of distinct licenses plates
theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * 2 = 2704000 :=
by
  sorry

end license_plate_count_l416_416256


namespace four_consecutive_integers_divisible_by_12_l416_416765

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416765


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416741

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416741


namespace smallest_non_divisible_l416_416055

theorem smallest_non_divisible :
  let A := k * Nat.factorial 65
  let B := m * Nat.factorial 65
  ∃ n : ℕ, (n > 65 ∧ ¬ (k + m) * Nat.factorial 65 % n = 0) ∧ 
           ∀ x : ℕ, (x > 65 ∧ ¬ ((k + m) * Nat.factorial 65 % x = 0)) → x ≥ n :=
  n = 67 :=
by
  sorry

end smallest_non_divisible_l416_416055


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416925

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416925


namespace counts_of_arson_l416_416012

-- Define variables A (arson), B (burglary), L (petty larceny)
variables (A B L : ℕ)

-- Conditions given in the problem
def burglary_charges : Prop := B = 2
def petty_larceny_charges_relation : Prop := L = 6 * B
def total_sentence_calculation : Prop := 36 * A + 18 * B + 6 * L = 216

-- Prove that given these conditions, the counts of arson (A) is 3
theorem counts_of_arson (h1 : burglary_charges B)
                        (h2 : petty_larceny_charges_relation B L)
                        (h3 : total_sentence_calculation A B L) :
                        A = 3 :=
sorry

end counts_of_arson_l416_416012


namespace greatest_divisor_of_consecutive_product_l416_416842

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416842


namespace f_is_periodic_with_period_two_a_l416_416370

variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (h_a_pos : a > 0)
variable (h_f_eq : ∀ x : ℝ, f x + f (x + a) + f x * f (x + a) = 1)

theorem f_is_periodic_with_period_two_a : ∀ x : ℝ, f (x + 2 * a) = f x :=
begin
  sorry
end

end f_is_periodic_with_period_two_a_l416_416370


namespace green_chips_correct_l416_416022

-- Definitions
def total_chips : ℕ := 120
def blue_chips : ℕ := total_chips / 4
def red_chips : ℕ := total_chips * 20 / 100
def yellow_chips : ℕ := total_chips / 10
def non_green_chips : ℕ := blue_chips + red_chips + yellow_chips
def green_chips : ℕ := total_chips - non_green_chips

-- Statement to prove
theorem green_chips_correct : green_chips = 54 := by
  -- Proof would go here
  sorry

end green_chips_correct_l416_416022


namespace solve_for_q_l416_416450

variable (R t m q : ℝ)

def given_condition : Prop :=
  R = t / ((2 + m) ^ q)

theorem solve_for_q (h : given_condition R t m q) : 
  q = (Real.log (t / R)) / (Real.log (2 + m)) := 
sorry

end solve_for_q_l416_416450


namespace find_a_l416_416414

theorem find_a' (a : ℝ) (h : 3 ∈ ({a + 2, 2 * a^2 + a} : set ℝ)) : a = -3 / 2 :=
sorry

end find_a_l416_416414


namespace max_value_a_l416_416409

theorem max_value_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x ∈ Ioc 0 2, f x = x^2 + a * x + 4 ∧ f x ≤ 6) → (a ≤ -1) :=
by
  sorry

end max_value_a_l416_416409


namespace distinct_rational_numbers_count_l416_416304

theorem distinct_rational_numbers_count :
  ∃ N : ℕ, 
    (N = 49) ∧
    ∀ (k : ℚ), |k| < 50 →
      (∃ x : ℤ, x^2 - k * x + 18 = 0) →
        ∃ m: ℤ, k = 2 * m ∧ |m| < 25 :=
sorry

end distinct_rational_numbers_count_l416_416304


namespace greatest_divisor_of_consecutive_product_l416_416850

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416850


namespace thm1_thm2_thm3_thm4_l416_416045

variable {D : Type*} (f : D → ℝ)
def property_M : Prop :=
∀ x₁ ∈ D, ∃ x₂ ∈ D, f x₁ * f x₂ = 1

namespace Examples

theorem thm1 : ¬ property_M (λ x : ℝ, x^3 - x) :=
sorry

theorem thm2 : property_M (λ x : ℝ, (exp x + exp(-x)) / 2) :=
sorry

theorem thm3 (t : ℝ) : property_M (λ x : ℝ, real.log x / real.log 8 + t) → t = 510 :=
sorry

theorem thm4 (a : ℝ) : property_M (λ x : ℝ, (3 * real.sin x + a) / 4) → a = 5 :=
sorry

end Examples

end thm1_thm2_thm3_thm4_l416_416045


namespace cost_to_fill_pool_l416_416124

-- Define the given conditions as constants
def filling_time : ℝ := 50
def flow_rate : ℝ := 100
def cost_per_10_gallons : ℝ := 0.01

-- Calculate total volume in gallons
def total_volume : ℝ := filling_time * flow_rate

-- Calculate the cost per gallon in dollars
def cost_per_gallon : ℝ := cost_per_10_gallons / 10

-- Define the total cost to fill the pool in dollars
def total_cost : ℝ := total_volume * cost_per_gallon

-- Prove that the total cost equals $5
theorem cost_to_fill_pool : total_cost = 5 := by
  unfold total_cost
  unfold total_volume
  unfold cost_per_gallon
  unfold filling_time
  unfold flow_rate
  unfold cost_per_10_gallons
  sorry

end cost_to_fill_pool_l416_416124


namespace vasya_numbers_l416_416155

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416155


namespace roots_condition_l416_416405

theorem roots_condition (m : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = x^2 + 2*(m - 1)*x - 5*m - 2) 
  (h_roots : ∃ x1 x2, x1 < 1 ∧ 1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) : 
  m > 1 := 
by
  sorry

end roots_condition_l416_416405


namespace four_consecutive_product_divisible_by_12_l416_416992

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416992


namespace evaluate_expression_l416_416446

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416446


namespace vasya_numbers_l416_416167

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416167


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416954

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416954


namespace four_consecutive_product_divisible_by_12_l416_416991

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416991


namespace ellipse_and_line_fixed_point_l416_416402

-- Definitions for the conditions
def ellipse_eq (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1

def eccentricity (a c : ℝ) : Prop :=
  c / a = sqrt 6 / 3

def fixed_point_on_x_axis (A B M N D : ℝ × ℝ) : Prop :=
  let l := (x : ℝ) in (1, 0) -- line l passing through M(1, 0)
  let D := (3, sqrt 6 / 3)  -- perpendicular line from A to x = 3, foot on D
  ∀ A B, (x = 3) ∧ (y = sqrt 6 / 3) --> (x = 2, y = 0)

-- The main theorem
theorem ellipse_and_line_fixed_point
  (a c : ℝ)
  (h_ellipse_eq : ellipse_eq a (3:ℝ) (sqrt 6 / 3))
  (h_eccentricity : eccentricity a c)
  (h_a_gt_1 : a > 1) :
  (a = sqrt 3) ∧ (fixed_point_on_x_axis (1, 0) (1, sqrt 6 / 3) (2, 0) (3, sqrt 6 / 3) (3, sqrt 6 / 3)) :=
sorry

end ellipse_and_line_fixed_point_l416_416402


namespace ellipse_focus_ratio_of_distances_l416_416401

noncomputable def ellipse_equation (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  a^2 = 4 ∧ b^2 = 3

theorem ellipse_focus (a b e : ℝ) (h : a > b ∧ b > 0 ∧ b = Real.sqrt (3 * e * a) ∧ (a^2 = b^2 + 1)) :
  (C: (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) ↔
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

theorem ratio_of_distances {α β a b e : ℝ} (h : α + β = Real.pi ∧ b = Real.sqrt(3 * e * a) ∧ a^2 = 4 ∧ b^2 = 3) :
  ∀ A B D E : ℝ × ℝ, 
  (⊢ (∀ x y, x^2 / 4 + y^2 / 3 = 1 ↔ |A - B|^2 / |D - E| = 4)) :=
by sorry

end ellipse_focus_ratio_of_distances_l416_416401


namespace distance_traveled_l416_416469

theorem distance_traveled (A B C Q : Type) (radius : Q)
  (right_triangle : A B C)
  (side_9 : ∃ x y : Q, x*2 + y*2 = 81)
  (side_12 : ∃ x y : Q, x*2 + y*2 = 144)
  (side_15 : ∃ x y : Q, x*2 + y*2 = 225)
  (circle_radius : radius = 2)
  (tangent_to_sides : ∀ x y z : Q, x^2 + y^2 = z^2 -> x + y = 24) :
  Q = 24 := sorry

end distance_traveled_l416_416469


namespace mean_of_three_l416_416399

theorem mean_of_three (x y z a : ℝ)
  (h₁ : (x + y) / 2 = 5)
  (h₂ : (y + z) / 2 = 9)
  (h₃ : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 :=
by
  sorry

end mean_of_three_l416_416399


namespace greatest_divisor_of_consecutive_product_l416_416854

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416854


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416931

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416931


namespace projectors_can_cover_plane_l416_416471

theorem projectors_can_cover_plane (points : Fin 4 → ℝ × ℝ) : 
  ∃ (directions : Fin 4 → ℝ × ℝ), 
    (∀ i : Fin 4, directions i ∈ {north, south, east, west}) ∧ 
    (∀ p : ℝ × ℝ, ∃ i : Fin 4, in_projector_range (points i) (directions i) p) :=
sorry

end projectors_can_cover_plane_l416_416471


namespace John_alone_typing_time_l416_416494

variables {P : ℝ} {R : ℝ}
  
-- Conditions
def John_rate_is_constant (R : ℝ) (t : ℝ) : Prop := P = R * t
def John_types_for_3_hours := 3 * R
def Jack_rate_is_five_over_two_of_John := (2 / 5) * R
def Jack_works_for_almost_5_hours := 4.999999999999999
def total_pages_written_by_both := 
  (3 * R) + (4.999999999999999 * (2 / 5) * R)

theorem John_alone_typing_time :
  (3 * R) + (4.999999999999999 * (2 / 5) * R) = P →
  P = 5 * R →
  P / R = 5 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end John_alone_typing_time_l416_416494


namespace log_exp_eval_l416_416285

theorem log_exp_eval :
  log 3 (sqrt 27) + log 10 25 + log 10 4 + 7^log 7 2 + (-9.8)^0 = 13/2 :=
by
  sorry

end log_exp_eval_l416_416285


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416873

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416873


namespace evaluate_sqrt_expression_l416_416312

theorem evaluate_sqrt_expression (x : ℝ) (h : x < -1) :
  (sqrt (x / (1 - (x - 2) / (x + 1)))) = - x * (x + 1) / sqrt 3 :=
by {
  sorry
}

end evaluate_sqrt_expression_l416_416312


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416676

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416676


namespace candles_not_fit_l416_416051

theorem candles_not_fit (R : ℝ) (n m : ℕ) (d : ℝ) 
  (h_r : R = 18) (h_n : n = 13) (h_d : d = 10) : 
  ¬(∃ (points : fin n → ℝ × ℝ), 
    (∀ i j : fin n, i ≠ j → dist (points i) (points j) ≥ d) 
    ∧ ∀ i : fin n, dist (0, 0) (points i) = R) :=
by sorry

end candles_not_fit_l416_416051


namespace triangle_ABC_l416_416485

noncomputable def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2,
    y := (A.y + B.y) / 2 }

structure Triangle :=
  (A B C : Point)
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

theorem triangle_ABC :
  ∀ (A B C M : Point), 
  (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) -- Non-degenerate triangle
  → A.dist B = 4  -- AB = 4
  → A.dist C = 8  -- AC = 8
  → midpoint B C = M
  → A.dist M = 3  -- AM = 3
  → B.dist C = 2 * Real.sqrt 31 -- BC = 2√31
:= sorry

end triangle_ABC_l416_416485


namespace probability_of_drawing_stamps_l416_416590

theorem probability_of_drawing_stamps : 
  let stamps := ["Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold"]
  in (2 / (list.permutations stamps).length) = (1 / 6) := by
  sorry

end probability_of_drawing_stamps_l416_416590


namespace g_zero_or_identity_l416_416536

def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x y z, g (x^2 + y^2 + y * g z) = x * g x + z^2 * g y

theorem g_zero_or_identity : g = (λ x, 0) ∨ g = (λ x, x) :=
sorry

end g_zero_or_identity_l416_416536


namespace route_comparison_l416_416555

def time_needed (distance speed : ℝ) : ℝ := distance / speed

theorem route_comparison :
  let tX1 := time_needed 3 25 * 60
  let tX2 := time_needed 7 50 * 60
  let tX := tX1 + tX2

  let tY1 := time_needed 2 20 * 60
  let tY2 := time_needed 6 40 * 60
  let tY := tY1 + tY2

  tX - tY = 0.6 :=
by
  sorry

end route_comparison_l416_416555


namespace greatest_divisor_four_consecutive_l416_416683

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416683


namespace ideal_function_f0_zero_two_x_minus_one_is_ideal_ideal_function_fixed_point_l416_416351

-- Definition of an ideal function
def is_ideal_function (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧
  (f 1 = 1) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

-- Proof Problem 1: If f is an ideal function, prove that f(0) = 0
theorem ideal_function_f0_zero {f : ℝ → ℝ} (h : is_ideal_function f) : f 0 = 0 := by
  sorry

-- Proof Problem 2: Prove that the function f(x) = 2x - 1 is an ideal function
theorem two_x_minus_one_is_ideal : is_ideal_function (λ x, 2 * x - 1) := by
  sorry

-- Proof Problem 3: If f is an ideal function and there exists x0 such that 
-- f(x0) in [0,1] and f(f(x0)) = x0, prove that f(x0) = x0
theorem ideal_function_fixed_point {f : ℝ → ℝ} (h : is_ideal_function f) (x0 : ℝ) 
  (hx0 : 0 ≤ x0 ∧ x0 ≤ 1 ∧ 0 ≤ f x0 ∧ f x0 ≤ 1 ∧ f (f x0) = x0) : f x0 = x0 := by
  sorry

end ideal_function_f0_zero_two_x_minus_one_is_ideal_ideal_function_fixed_point_l416_416351


namespace probability_sum_less_than_product_l416_416139

def set_of_numbers := {1, 2, 3, 4, 5, 6, 7}

def count_valid_pairs : ℕ :=
  set_of_numbers.to_list.product set_of_numbers.to_list
    |>.count (λ (ab : ℕ × ℕ), (ab.1 - 1) * (ab.2 - 1) > 1)

def total_combinations := (set_of_numbers.to_list).length ^ 2

theorem probability_sum_less_than_product :
  (count_valid_pairs : ℚ) / total_combinations = 36 / 49 :=
by
  -- Placeholder for proof, since proof is not requested
  sorry

end probability_sum_less_than_product_l416_416139


namespace greatest_divisor_of_four_consecutive_integers_l416_416802

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416802


namespace simplify_and_evaluate_l416_416583

theorem simplify_and_evaluate (a b : ℤ) (h₁ : a = -1) (h₂ : b = 3) :
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 :=
by
  sorry

end simplify_and_evaluate_l416_416583


namespace kate_average_speed_l416_416020

-- Conditions
def time_biking : ℝ := 45 / 60 -- 45 minutes to hours
def speed_biking : ℝ := 20 -- mph
def time_walking : ℝ := 60 / 60 -- 60 minutes to hours
def speed_walking : ℝ := 3 -- mph

-- Total distance traveled
def total_distance : ℝ := (speed_biking * time_biking) + (speed_walking * time_walking)

-- Total time spent traveling
def total_time : ℝ := time_biking + time_walking

-- Overall average speed
def average_speed : ℝ := total_distance / total_time

-- Theorem statement
theorem kate_average_speed : average_speed = 10 := 
by
  -- Sorry used to skip proof
  sorry

end kate_average_speed_l416_416020


namespace ways_to_select_books_l416_416472

theorem ways_to_select_books (total_books : ℕ) (books_to_select : ℕ) (fixed_books : ℕ) :
  total_books = 7 ∧ books_to_select = 5 ∧ fixed_books = 2 →
  (nat.choose (total_books - fixed_books) (books_to_select - fixed_books)) = 10 :=
by
  intros h
  cases h with h_total h_rest
  cases h_rest with h_select h_fixed
  rw [h_total, h_select, h_fixed]
  rw [nat.choose]
  -- Apply the binomial coefficient calculation steps:
  sorry -- Completing the proof is not required.

end ways_to_select_books_l416_416472


namespace BLCK_has_incircle_l416_416600

-- Definitions of inner structures and assumptions
variable (A B C D O K L : Type)

-- Cyclic quadrilateral condition
axiom cyclic_quadrilateral : cyclic A B C D

-- Intersection of diagonals
axiom intersection_O : intersects (diagonal A C) (diagonal B D) O

-- Intersection of circumcircles condition
axiom circumcircles_intersection_K : intersects (circumcircle A O B) (circumcircle C O D) K

-- Similar and equally oriented triangles condition
axiom similar_triangles : similar_oriented (triangle B L C) (triangle A K D)

-- Concave condition of quadrilateral BLCK
axiom convex_BLCK : convex B L C K

-- Proof of the statement
theorem BLCK_has_incircle : has_incircle B L C K :=
sorry

end BLCK_has_incircle_l416_416600


namespace product_of_consecutive_integers_l416_416747

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416747


namespace greatest_integer_less_than_neg_21_over_5_l416_416191

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ (z : ℤ), z < -21 / 5 ∧ ∀ (w : ℤ), w < -21 / 5 → w ≤ z :=
begin
  use -5,
  split,
  { norm_num },
  { intros w hw,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416191


namespace current_number_of_women_is_24_l416_416008

-- Define initial person counts based on the given ratio and an arbitrary factor x.
variables (x : ℕ)
def M_initial := 4 * x
def W_initial := 5 * x
def C_initial := 3 * x
def E_initial := 2 * x

-- Define the changes that happened to the room.
def men_after_entry := M_initial x + 2
def women_after_leaving := W_initial x - 3
def women_after_doubling := 2 * women_after_leaving x
def children_after_leaving := C_initial x - 5
def elderly_after_leaving := E_initial x - 3

-- Define the current counts after all changes.
def men_current := 14
def children_current := 7
def elderly_current := 6

-- Prove that the current number of women is 24.
theorem current_number_of_women_is_24 :
  men_after_entry x = men_current ∧
  children_after_leaving x = children_current ∧
  elderly_after_leaving x = elderly_current →
  women_after_doubling x = 24 :=
by
  sorry

end current_number_of_women_is_24_l416_416008


namespace probability_one_of_each_color_l416_416242

/-- Probability of selecting one marble of each color (red, blue, and green) when 3 out of 7 specified marbles are randomly selected without replacement -/
theorem probability_one_of_each_color :
  let marbles := {red, blue, green, yellow}
  let red_count := 2
  let blue_count := 2
  let green_count := 2
  let yellow_count := 1
  let total_marbles := red_count + blue_count + green_count + yellow_count
  let selection := 3
  let total_combinations := (nat.choose total_marbles selection)
  let favorable_combinations := red_count * blue_count * green_count
  (favorable_combinations / total_combinations : ℚ) = 8 / 35 :=
by
  let marbles := {red, blue, green, yellow}
  let red_count := 2
  let blue_count := 2
  let green_count := 2
  let yellow_count := 1
  let total_marbles := red_count + blue_count + green_count + yellow_count
  let selection := 3
  let total_combinations := (nat.choose total_marbles selection)
  let favorable_combinations := red_count * blue_count * green_count
  have total_non_zero : total_combinations ≠ 0 := sorry
  have favorable_non_zero : favorable_combinations ≠ 0 := sorry
  have favorable := favorable_combinations : ℚ / total_combinations
  have expected_ratio := (8 : ℚ) / 35
  exact (by norm_num : favorable = expected_ratio)

end probability_one_of_each_color_l416_416242


namespace hexagon_same_length_probability_l416_416524

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l416_416524


namespace product_of_consecutive_integers_l416_416749

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416749


namespace greatest_divisor_of_four_consecutive_integers_l416_416962

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416962


namespace proof_problem_l416_416435

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416435


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416880

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416880


namespace range_of_a_l416_416418

def P (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def Q (a : ℝ) : Prop :=
  5 - 2 * a > 1

theorem range_of_a (a : ℝ) : (xor (P a) (Q a)) → a ∈ set.Iic (-2) := sorry

end range_of_a_l416_416418


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416938

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416938


namespace greatest_divisor_of_four_consecutive_integers_l416_416980

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416980


namespace product_of_consecutive_integers_l416_416748

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416748


namespace system_of_equations_has_solution_l416_416295

theorem system_of_equations_has_solution :
  ∃ k : ℕ, ∃ (x : Fin k → ℝ),
  (∑ i, x i = 0) ∧
  (∑ i, (x i)^3 = 0) ∧
  (∑ i, (x i)^5 = 0) ∧
  (∑ i, (x i)^7 = 0) ∧
  (∑ i, (x i)^9 = 0) ∧
  (∑ i, (x i)^11 = 0) ∧
  (∑ i, (x i)^13 = 0) ∧
  (∑ i, (x i)^15 = 0) ∧
  (∑ i, (x i)^17 = 0) ∧
  (∑ i, (x i)^19 = 0) ∧
  (∑ i, (x i)^21 = 1) :=
sorry

end system_of_equations_has_solution_l416_416295


namespace greatest_divisor_four_consecutive_l416_416694

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416694


namespace multiple_of_persons_l416_416073

variable (Persons Work : ℕ) (Rate : ℚ)

def work_rate (P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D
def multiple_work_rate (m P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D

theorem multiple_of_persons
  (P : ℕ) (W : ℕ)
  (h1 : work_rate P W 12 = W / 12)
  (h2 : multiple_work_rate 1 P (W / 2) 3 = (W / 6)) :
  m = 2 :=
by sorry

end multiple_of_persons_l416_416073


namespace vasya_numbers_l416_416179

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416179


namespace product_of_consecutive_integers_l416_416750

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416750


namespace interest_rates_correct_l416_416495

variables (r : ℚ) (interest1 interest2 : ℚ)

-- Account investments
def investment1 := 4000
def investment2 := 8200

-- Interest rates
def rate1 := r
def rate2 := r + 1.5

-- Total interest received
def total_interest := 1282

-- Expressions for the interests from each account
def interest1 := (investment1 * rate1) / 100
def interest2 := (investment2 * rate2) / 100

-- The proof problem
theorem interest_rates_correct :
  interest1 + interest2 = total_interest →
  r = 9.5 ∧ (r + 1.5) = 11 :=
begin
  sorry
end

end interest_rates_correct_l416_416495


namespace solved_only_B_l416_416634

variables (N_a N_b N_c N_ab N_ac N_bc N_abc : ℕ)

-- Conditions from the problem
def condition1 : Prop := N_a + N_b + N_c + N_ab + N_ac + N_bc + N_abc = 25
def condition2 : Prop := N_b + N_bc = 2 * (N_c + N_bc)
def condition3 : Prop := N_a = 1 + (N_ab + N_ac + N_abc)
def condition4 : Prop := N_a = (N_a + N_b + N_c) / 2

theorem solved_only_B :
  condition1 N_a N_b N_c N_ab N_ac N_bc N_abc →
  condition2 N_a N_b N_c N_ab N_ac N_bc N_abc →
  condition3 N_a N_b N_c N_ab N_ac N_bc N_abc →
  condition4 N_a N_b N_c N_ab N_ac N_bc N_abc →
  N_b = 6 :=
by
  intro h1 h2 h3 h4
  sorry

end solved_only_B_l416_416634


namespace greatest_divisor_of_consecutive_product_l416_416858

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416858


namespace smallest_power_of_12_not_palindrome_l416_416334

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def power_of_12_not_palindrome (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 12^k ∧ ¬ is_palindrome (n.repr)

theorem smallest_power_of_12_not_palindrome : ∃ n : ℕ, power_of_12_not_palindrome n ∧ ∀ m : ℕ, power_of_12_not_palindrome m → n ≤ m :=
sorry

end smallest_power_of_12_not_palindrome_l416_416334


namespace possible_to_make_pairwise_coprime_circle_l416_416277

theorem possible_to_make_pairwise_coprime_circle (a : Fin 100 → ℕ) :
  (∀ i j : Fin 100, i ≠ j → Nat.coprime (a i) (a j)) →
  ∃ b : Fin 100 → ℕ,
    (∀ i j : Fin 100, i ≠ j → Nat.coprime (b i) (b j)) ∧
    (∀ i : Fin 100, ∃ c : ℕ, b = Function.update a i (a i + c * Nat.gcd (a ((i - 1 + 100) % 100)) (a ((i + 1) % 100)))) :=
by
  sorry

end possible_to_make_pairwise_coprime_circle_l416_416277


namespace hexagon_same_length_probability_l416_416516

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l416_416516


namespace kyler_wins_one_game_l416_416059

theorem kyler_wins_one_game :
  ∃ (Kyler_wins : ℕ),
    (Kyler_wins + 3 + 2 + 2 = 6 ∧
    Kyler_wins + 3 = 6 ∧
    Kyler_wins = 1) := by
  sorry

end kyler_wins_one_game_l416_416059


namespace total_animals_l416_416114

theorem total_animals (giraffes pigs dogs : ℕ) (h1 : giraffes = 6) (h2 : pigs = 8) (h3 : dogs = 4) : giraffes + pigs + dogs = 18 :=
by {
  rw [h1, h2, h3],
  exact rfl,
}

end total_animals_l416_416114


namespace greatest_divisor_four_consecutive_integers_l416_416824

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416824


namespace compare_abc_l416_416366

theorem compare_abc (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.log 2) (h3 : c = Real.logb 3 (Real.sin (Real.pi / 6))) :
  a > b ∧ b > c :=
by
  sorry

end compare_abc_l416_416366


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416678

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416678


namespace product_of_consecutive_integers_l416_416755

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416755


namespace product_of_consecutive_integers_l416_416758

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416758


namespace jessica_initial_money_l416_416014

def amount_spent : ℝ := 10.22
def amount_left : ℝ := 1.51
def initial_amount : ℝ := 11.73

theorem jessica_initial_money :
  amount_spent + amount_left = initial_amount := 
  by
    sorry

end jessica_initial_money_l416_416014


namespace sum_of_segments_l416_416483

theorem sum_of_segments (perimeter_larger_triangle : ℝ) (side_length_smaller : ℝ) :
  let side_length_larger := perimeter_larger_triangle / 3 in
  perimeter_larger_triangle = 24 → 
  ∀ T : Type, T = "equilateral triangles" → 
  side_length_smaller * 3 = side_length_larger → 
  sum_of_all_segments = 81 :=
begin
  sorry
end

end sum_of_segments_l416_416483


namespace Mrs_Kovacs_meet_count_l416_416079

theorem Mrs_Kovacs_meet_count :
  ∃ P : Finset ℕ, (P.card = 10) ∧ 
  (∀ x ∈ P.erase 0, ∃ n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ 
    (∀ m ∈ P.erase 0, (x ≠ m → (∃! y ∈ P.erase 0, y = n))) ∧
    ∃! z : ℕ, z ∈ P.erase 0 ∧ z ≠ x) → 
  (∃ M : ℕ, M ∈ P.erase 0 ∧ M = 4) :=
by
  sorry

end Mrs_Kovacs_meet_count_l416_416079


namespace intersecting_circles_tangent_l416_416417

open Real

theorem intersecting_circles_tangent
  (O₁ O₂ : Point)
  (r₁ r₂ : ℝ)
  (h₁ : 0 < r₁)
  (h₂ : 0 < r₂)
  (A B : Point)
  (M N : Point)
  (h_intersect : dist O₁ A = r₁ ∧ dist O₂ A = r₂ ∧ dist O₁ B = r₁ ∧ dist O₂ B = r₂)
  (h_tangent : is_tangent_to_circles M N O₁ O₂ A B):
  (circumradius (triangle.mk A M N) = sqrt(r₁ * r₂)) ∧
  (circumradius (triangle.mk B M N) = sqrt(r₁ * r₂)) ∧
  ((dist M A / dist N A) = sqrt(r₁ / r₂)) ∧
  ((dist M B / dist N B) = sqrt(r₁ / r₂)) := sorry

end intersecting_circles_tangent_l416_416417


namespace total_trees_cut_down_l416_416011

-- Definitions based on conditions in the problem
def trees_per_day_james : ℕ := 20
def days_with_just_james : ℕ := 2
def total_trees_by_james := trees_per_day_james * days_with_just_james

def brothers : ℕ := 2
def days_with_brothers : ℕ := 3
def trees_per_day_brothers := (20 * (100 - 20)) / 100 -- 20% fewer than James
def trees_per_day_total := brothers * trees_per_day_brothers + trees_per_day_james

def total_trees_with_brothers := trees_per_day_total * days_with_brothers

-- The statement to be proved
theorem total_trees_cut_down : total_trees_by_james + total_trees_with_brothers = 136 := by
  sorry

end total_trees_cut_down_l416_416011


namespace probability_of_same_length_segments_l416_416505

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l416_416505


namespace trig_identity_l416_416291

theorem trig_identity : real.sin (3 * real.pi / 180) * real.sin (39 * real.pi / 180) * real.sin (63 * real.pi / 180) * real.sin (75 * real.pi / 180) = 1 / 2 :=
by sorry

end trig_identity_l416_416291


namespace vasya_numbers_l416_416150

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416150


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416665

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416665


namespace part1_part2_l416_416420

variables (θ φ λ : ℝ)

def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)

-- Part 1
theorem part1 : 
  ∀ θ φ, abs (θ - φ) = Real.pi / 3 →
  Real.norm ((a θ).1 - (b φ).1, (a θ).2 - (b φ).2) = 1 :=
by sorry

-- Part 2
def f (θ λ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.cos (Real.pi / 3 - θ), Real.sin (Real.pi / 3 - θ))
  a.1 * b.1 + a.2 * b.2 - λ * Real.norm (a.1 + b.1, a.2 + b.2)

theorem part2 :
  ∀ θ λ, θ ∈ Set.Icc 0 (Real.pi / 2) → 1 ≤ λ ∧ λ ≤ 2 →
  f θ λ = - λ^2 / 4 - 1 :=
by sorry

end part1_part2_l416_416420


namespace four_consecutive_integers_divisible_by_12_l416_416769

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416769


namespace a_n_formula_S_n_formula_l416_416000

noncomputable def a_n (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n-1)

def b_n (n : ℕ) : ℝ := a_n (n+1) + Real.log (a_n n) / log 2

def S_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i => b_n (i + 1))

theorem a_n_formula (n : ℕ) (hn : n ≠ 0) (h1 : a_n 1 * a_n 3 = 4) (h2 : a_n 3 + 1 = (a_n 2 + a_n 4) / 2) :
  a_n n = if n = 0 then 0 else 2^(n-1) :=
sorry

theorem S_n_formula (n : ℕ) :
  S_n n = 2^(n+1) - 2 + n * (n-1) / 2 :=
sorry

end a_n_formula_S_n_formula_l416_416000


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416935

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416935


namespace min_expression_at_8_l416_416112

theorem min_expression_at_8 (n : ℕ) (h : 0 < n) : 
  (∃ n : ℕ, 0 < n ∧ (∀ m : ℕ, 0 < m → (n ≤ m → (n / 3 + 24 / n ≤ m / 3 + 24 / m))) ) := 
begin
  use 8,
  split,
  { norm_num, },
  sorry,
end

end min_expression_at_8_l416_416112


namespace blocks_differing_in_three_ways_l416_416248

noncomputable def count_blocks_differing_in_three_ways : ℕ :=
  let materials := 3
  let sizes := 3
  let colors := 4
  let patterns := 2
  let shapes := 4
  let generating_function := (1 + 2 * X) ^ 2 * (1 + 3 * X) ^ 2 * (1 + X)
  let polynomial := generating_function.expand
  polynomial.coeff 3

theorem blocks_differing_in_three_ways : count_blocks_differing_in_three_ways = 112 := by
  sorry

end blocks_differing_in_three_ways_l416_416248


namespace complement_A_is_interval_l416_416415

def A : set ℝ := {x | ∃ y, y = Real.log (x + 1)}

theorem complement_A_is_interval :
  (Aᶜ = {x : ℝ | x ≤ -1}) := 
sorry

end complement_A_is_interval_l416_416415


namespace four_consecutive_product_divisible_by_12_l416_416994

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416994


namespace travel_time_equation_l416_416103

theorem travel_time_equation (x : ℝ) (h1 : ∀ d : ℝ, d > 0) :
  (x / 160) - (x / 200) = 2.5 :=
sorry

end travel_time_equation_l416_416103


namespace petya_higher_chance_l416_416560

noncomputable section

def total_candies := 25
def P_A := 0.54
def P_B := 1 - P_A
def P_Vasya_wins := P_B
def P_Petya_wins := 1 - P_Vasya_wins

theorem petya_higher_chance : P_Petya_wins > P_Vasya_wins := by
  have h1 : P_Vasya_wins = P_B := rfl
  have h2 : P_B = 0.46 := by 
    have h3 : P_A = 0.54 := rfl
    show P_B = 1 - P_A
    show P_B = 1 - 0.54
    show P_B = 0.46
  have h4 : P_Petya_wins = 1 - P_Vasya_wins := rfl
  have h5 : P_Petya_wins = 0.54 := by 
    show P_Petya_wins = 1 - 0.46
    show P_Petya_wins = 0.54
  show 0.54 > 0.46
  exact h5

end petya_higher_chance_l416_416560


namespace sqrt_64_sqrt_25_eq_8_sqrt_5_l416_416201

theorem sqrt_64_sqrt_25_eq_8_sqrt_5 : sqrt (64 * sqrt 25) = 8 * sqrt 5 :=
by
  sorry

end sqrt_64_sqrt_25_eq_8_sqrt_5_l416_416201


namespace eccentricity_range_of_hyperbola_l416_416411

theorem eccentricity_range_of_hyperbola 
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (P : ℝ × ℝ) (xo yo : ℝ) 
  (P_on_hyperbola : P = (xo, yo))
  (P_cond : xo > a)
  (hyperbola_cond : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (sine_ratio_cond : ∀ (PF1 PF2 : ℝ), PF1 = PF2 * (a / c)) :
  1 < eccentricity a b < sqrt 2 + 1 := sorry

end eccentricity_range_of_hyperbola_l416_416411


namespace probability_of_same_length_segments_l416_416512

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l416_416512


namespace greatest_divisor_four_consecutive_l416_416693

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416693


namespace marcia_hair_length_l416_416550

theorem marcia_hair_length :
  ∀ (initial_length : ℝ), initial_length = 24 → 
  ∀ (first_cut_fraction growth second_cut : ℝ), first_cut_fraction = 1/2 → growth = 4 → second_cut = 2 → 
  let length_after_first_cut := initial_length * (1 - first_cut_fraction) in
  let length_after_growth := length_after_first_cut + growth in
  let final_length := length_after_growth - second_cut in
  final_length = 14 :=
by
  intros initial_length h1 first_cut_fraction h2 growth h3 second_cut h4;
  simp [h1, h2, h3, h4];
  sorry

end marcia_hair_length_l416_416550


namespace correct_calculation_l416_416207

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l416_416207


namespace smallest_non_palindrome_power_of_12_l416_416349

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ¬is_palindrome (12^n) ∧
  ∀ m : ℕ, m > 0 → ¬is_palindrome (12^m) → 12^m ≥ 12^n :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num,
    -- Show that 144 is not a palindrome
    sorry },
  { intros m hm hnp,
    -- Show that for any other power of 12 that is not a palindrome, the result is ≥ 144
    -- essentially proving that 12^2 is the smallest such number.
    sorry }
end

end smallest_non_palindrome_power_of_12_l416_416349


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416725

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416725


namespace function_range_l416_416104

open Real

theorem function_range : 
  let f (x : ℝ) := 1 - 1 / (x^2 + 1) 
  in (∀ y, ∃ x, f x = y ↔ 0 ≤ y ∧ y < 1) :=
by 
  let f (x : ℝ) := 1 - 1 / (x^2 + 1)
  sorry

end function_range_l416_416104


namespace divisor_of_four_consecutive_integers_l416_416646

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416646


namespace partitional_points_difference_l416_416028

def unit_square (S : set (ℝ × ℝ)) : Prop :=
  ∃ a b c d : ℝ, S = { (x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 }

noncomputable def n_ray_partitional_points (S : set (ℝ × ℝ)) (n : ℕ) : set (ℝ × ℝ) :=
  {Y | is_in_interior Y S ∧ has_n_rays_emitting_from Y n S }

def is_in_interior (Y : ℝ × ℝ) (S : set (ℝ × ℝ)) : Prop :=
  ∃ x y, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1

def has_n_rays_emitting_from (Y : ℝ × ℝ) (n : ℕ) (S : set (ℝ × ℝ)) : Prop :=
  ∃ rs : list (ℝ × ℝ), 
    (rs.length = n) ∧ 
    (∀ r ∈ rs, ray_emits_from Y r S) ∧ 
    (disjoint_partition_of_unit_square rs n S)

def ray_emits_from (Y : ℝ × ℝ) (r : ℝ × ℝ) (S : set (ℝ × ℝ)) : Prop :=
  -- Define ray emitting property; skipping detailed definition
  sorry

def disjoint_partition_of_unit_square (rs : list (ℝ × ℝ)) (n : ℕ) (S : set (ℝ × ℝ)) : Prop :=
  -- Define disjoint partition property; skipping detailed definition
  sorry

def number_150_ray_partitional (S : set (ℝ × ℝ)) : ℕ :=
  set.card (n_ray_partitional_points S 150)

def number_90_ray_partitional (S : set (ℝ × ℝ)) : ℕ :=
  set.card (n_ray_partitional_points S 90)

def number_overlapping_points (S : set (ℝ × ℝ)) : ℕ :=
  set.card (n_ray_partitional_points S 150 ∩ n_ray_partitional_points S 90)

theorem partitional_points_difference (S : set (ℝ × ℝ)) (hS : unit_square S) :
  number_150_ray_partitional S - number_overlapping_points S = 5160 :=
by
  sorry

end partitional_points_difference_l416_416028


namespace find_alpha_l416_416587

theorem find_alpha (n : ℕ) (h : ∀ x : ℤ, x * x * x + α * x + 4 - 2 * 2016 ^ n = 0 → ∀ r : ℤ, x = r)
  : α = -3 :=
sorry

end find_alpha_l416_416587


namespace part_I_part_II_part_III_l416_416375

theorem part_I (a b : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  {n : ℕ | a_n n = 1 ∨ a_n n = 2 ∨ a_n n = -1 ∨ a_n n = 0} = 
  {1, 2, -1, 0} :=
sorry

theorem part_II (a b : ℝ) (h₁ : a < 0) (h₂ : b < 0) : 
  ∃ p : ℕ, p = 9 ∧ (∀ n : ℕ, a_n (n + p) = a_n n) :=
sorry

theorem part_III (a b : ℝ) (h₁ : a ≥ 0) (h₂ : b ≥ 0) (h₃ : a + b ≠ 0) : 
  ∃ n : ℕ, n = 4 ∧ ∀ m : ℕ, m < 4 → m ∈ M :=
sorry

-- Auxiliary function definitions to represent the sequence a_n
noncomputable def a_n : ℕ → ℝ
| 0 := 0   -- Usually a_0 is not defined, but needed for Lean's definition
| 1 := a
| 2 := b
| n + 3 := abs (a_n (n + 2)) - a_n (n + 1)

-- Auxiliary definition of the set M
noncomputable def M : set ℝ := {a_n n | n ∈ ℤ}

end part_I_part_II_part_III_l416_416375


namespace product_of_consecutive_integers_l416_416742

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416742


namespace greatest_divisor_of_four_consecutive_integers_l416_416966

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416966


namespace value_of_f_2012_l416_416410

noncomputable def f : ℝ → ℝ
| x => if x ≤ 2011 then sin (π / 3 * x) else f (x - 4)

theorem value_of_f_2012 : f 2012 = - (√3 / 2) := by
  sorry

end value_of_f_2012_l416_416410


namespace vasya_numbers_l416_416174

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416174


namespace hexagon_probability_l416_416509

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l416_416509


namespace range_of_m_l416_416234

-- Define a function for the quadratic expression
def quadratic (x m : ℝ) : ℝ := x^2 + m * x - 1

-- Define the condition in Lean
axiom quadratic_lt_zero (m : ℝ) : ∀ x, m ≤ x ∧ x ≤ m + 1 → quadratic x m < 0

-- Prove that the range of m is (-sqrt(2)/2, 0)
theorem range_of_m : ∃ (m : ℝ), -real.sqrt 2 / 2 < m ∧ m < 0 ∧ (∀ x, m ≤ x ∧ x ≤ m + 1 → quadratic x m < 0) :=
sorry

end range_of_m_l416_416234


namespace vasya_numbers_l416_416182

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416182


namespace greatest_divisor_four_consecutive_l416_416686

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416686


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416937

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416937


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416733

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416733


namespace simplify_expression_l416_416287

theorem simplify_expression (a : Int) : 2 * a - a = a :=
by
  sorry

end simplify_expression_l416_416287


namespace rise_in_water_level_is_correct_l416_416213

-- Definitions based on conditions
def edge : ℝ := 10 -- cm
def length : ℝ := 20 -- cm
def width : ℝ := 15 -- cm

-- Volume of the cube
def V_cube : ℝ := edge ^ 3

-- Base area of the rectangular vessel
def A_base : ℝ := length * width

-- Rise in water level
def h_rise : ℝ := V_cube / A_base

-- Lean theorem to prove the rise in water level
theorem rise_in_water_level_is_correct : h_rise = 3.33 := by
  calc
    h_rise = V_cube / A_base : by rfl
    ... = (edge ^ 3) / (length * width) : by rfl
    ... = (10 ^ 3) / (20 * 15) : by rfl
    ... = 1000 / 300 : by norm_num
    ... = 3.33 : by norm_num

end rise_in_water_level_is_correct_l416_416213


namespace polar_equation_and_distance_product_l416_416476

noncomputable def parametric_to_polar_equation (x y : ℝ) : ℝ := 4 * (y - 2) / x

noncomputable def distance_product (t1 t2 : ℝ) : ℝ := t1 * t2

theorem polar_equation_and_distance_product (theta : ℝ) :
  (∀ x y : ℝ, (x = 2 * cos theta ∧ y = 2 + 2 * sin theta) → (parametric_to_polar_equation x y = 4 * sin theta)) ∧
  (∀ t1 t2 : ℝ, distance_product t1 t2 = -2 → (sqrt 2, π / 4) → (distance_product t1 t2 = 2)) :=
by sorry

end polar_equation_and_distance_product_l416_416476


namespace lambda_bounds_l416_416400

theorem lambda_bounds (λ : ℝ) (a : ℕ → ℝ)
  (h_def : ∀ n, a n = 
    (if n ≤ 4 then (λ - 1) * n + 5 else (3 - λ)^(n - 4) + 5))
  (h_increasing : ∀ n, a n ≤ a (n + 1)) :
  1 < λ ∧ λ < 7 / 5 := sorry

end lambda_bounds_l416_416400


namespace greatest_divisor_of_four_consecutive_integers_l416_416965

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416965


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416942

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416942


namespace greatest_divisor_of_four_consecutive_integers_l416_416819

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416819


namespace product_of_consecutive_integers_l416_416754

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416754


namespace problem_statement_l416_416533

noncomputable def f (a : ℝ) (x : ℝ) := log a (1 + x) + log a (3 - x)

theorem problem_statement
  (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h_f1 : f a 1 = 2) :
  a = 2 ∧ (∀ x, -1 < x ∧ x < 3 → true) ∧ (let I := set.Icc (0 : ℝ) 3/2 in ∀ x ∈ I, f a x ≤ 2) := 
sorry

end problem_statement_l416_416533


namespace number_of_ways_to_place_letters_l416_416060

open Finset

-- Define a set of positions
def positions : Finset (ℕ × ℕ) := Finset.univ.product Finset.univ

-- Define the condition that no letter appears more than once per row and column
def valid_placement (p1 p2 p3 p4 : (ℕ × ℕ)) : Prop :=
  p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2 ∧ 
  p1.1 ≠ p3.1 ∧ p1.2 ≠ p3.2 ∧ 
  p1.1 ≠ p4.1 ∧ p1.2 ≠ p4.2 ∧ 
  p2.1 ≠ p3.1 ∧ p2.2 ≠ p3.2 ∧ 
  p2.1 ≠ p4.1 ∧ p2.2 ≠ p4.2 ∧ 
  p3.1 ≠ p4.1 ∧ p3.2 ≠ p4.2

theorem number_of_ways_to_place_letters : 
  ∃ (ls : list (ℕ × ℕ)),
  (ls.length = 4) ∧ 
  (∀ p1 p2 p3 p4 ∈ ls, valid_placement p1 p2 p3 p4) ∧
  ls.nodup ∧
  3960 = 16.choose 2 * 4 * 14.choose 2 * 4 := 
sorry

end number_of_ways_to_place_letters_l416_416060


namespace vasya_numbers_l416_416151

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416151


namespace max_integer_solutions_l416_416354

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end max_integer_solutions_l416_416354


namespace divisor_of_four_consecutive_integers_l416_416647

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416647


namespace probability_of_sum_less_than_product_l416_416136

-- Define the problem conditions
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the event condition
def is_valid_pair (a b : ℕ) : Prop := (a ∈ set_of_numbers) ∧ (b ∈ set_of_numbers) ∧ (a * b > a + b)

-- Count the number of valid pairs
noncomputable def count_valid_pairs : ℕ :=
  set_of_numbers.sum (λ a, set_of_numbers.filter (is_valid_pair a).card)

-- Count the total possible pairs
noncomputable def total_pairs : ℕ :=
  set_of_numbers.card * set_of_numbers.card

-- Calculate the probability
noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / total_pairs

-- State the theorem
theorem probability_of_sum_less_than_product :
  probability = 25 / 49 :=
by sorry

end probability_of_sum_less_than_product_l416_416136


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416926

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416926


namespace greatest_divisor_of_four_consecutive_integers_l416_416964

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416964


namespace quadratic_equation_completing_square_l416_416490

theorem quadratic_equation_completing_square :
  ∃ a b c : ℤ, a > 0 ∧ (25 * x^2 + 30 * x - 75 = 0 → (a * x + b)^2 = c) ∧ a + b + c = -58 :=
  sorry

end quadratic_equation_completing_square_l416_416490


namespace yellow_balls_in_bag_l416_416369

open Classical

theorem yellow_balls_in_bag (Y : ℕ) (hY1 : (Y/(Y+2): ℝ) * ((Y-1)/(Y+1): ℝ) = 0.5) : Y = 5 := by
  sorry

end yellow_balls_in_bag_l416_416369


namespace greatest_integer_less_than_neg_21_over_5_l416_416194

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ n : ℤ, n < -21 / 5 ∧ ∀ m : ℤ, m < -21 / 5 → m ≤ n :=
begin
  use -5,
  split,
  { linarith },
  { intros m h,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l416_416194


namespace total_value_of_item_l416_416203

-- Define the necessary conditions
def import_tax_paid (V : ℝ) : ℝ :=
  if V > 1000 then 0.07 * (V - 1000) else 0

theorem total_value_of_item (V : ℝ) (h1 : import_tax_paid V = 110.60) : V = 2580 :=
by
  sorry

end total_value_of_item_l416_416203


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416739

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416739


namespace greatest_divisor_of_consecutive_product_l416_416861

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416861


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416724

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416724


namespace smallest_non_palindrome_power_of_12_l416_416348

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ¬is_palindrome (12^n) ∧
  ∀ m : ℕ, m > 0 → ¬is_palindrome (12^m) → 12^m ≥ 12^n :=
begin
  use 2,
  split,
  { norm_num },
  split,
  { norm_num,
    -- Show that 144 is not a palindrome
    sorry },
  { intros m hm hnp,
    -- Show that for any other power of 12 that is not a palindrome, the result is ≥ 144
    -- essentially proving that 12^2 is the smallest such number.
    sorry }
end

end smallest_non_palindrome_power_of_12_l416_416348


namespace evaluate_expression_l416_416447

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416447


namespace man_present_age_l416_416214

variable {P : ℝ}

theorem man_present_age (h1 : P = 1.25 * (P - 10)) (h2 : P = (5 / 6) * (P + 10)) : P = 50 :=
  sorry

end man_present_age_l416_416214


namespace maximize_l_l416_416543

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 8 * x + 3

theorem maximize_l (a : ℝ) (la : ℝ) (h1 : a < 0) (h2 : ∀ x, 0 ≤ x → x ≤ la → abs (f a x) ≤ 5) :
  a = -8 → la = (Real.sqrt 5 + 1) / 2 :=
begin
  sorry
end

end maximize_l_l416_416543


namespace product_of_consecutive_integers_l416_416743

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416743


namespace div_product_four_consecutive_integers_l416_416799

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416799


namespace greatest_divisor_of_four_consecutive_integers_l416_416814

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416814


namespace vasya_numbers_l416_416180

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416180


namespace sequence_solution_l416_416574

variable (C : ℝ) (a : ℕ → ℝ)

noncomputable def condition_1 := C > 1
noncomputable def condition_2 := a 1 = 1
noncomputable def condition_3 := a 2 = 2
noncomputable def condition_4 := ∀ (m n : ℕ), a (m * n) = a m * a n
noncomputable def condition_5 := ∀ (m n : ℕ), a (m + n) ≤ C * (a m + a n)

theorem sequence_solution (hyp1 : condition_1 C) (hyp2 : condition_2 a) (hyp3 : condition_3 a)
  (hyp4 : condition_4 a) (hyp5 : condition_5 a) : ∀ (n : ℕ), a n = n := 
sorry

end sequence_solution_l416_416574


namespace div_product_four_consecutive_integers_l416_416801

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416801


namespace tan_add_pi_over_3_l416_416451

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_add_pi_over_3_l416_416451


namespace div_product_four_consecutive_integers_l416_416798

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416798


namespace vasya_numbers_l416_416178

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416178


namespace greatest_divisor_of_consecutive_product_l416_416852

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416852


namespace greatest_divisor_four_consecutive_integers_l416_416836

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416836


namespace geometric_sequence_sum_l416_416484

-- The sequence a_n is geometric with ratio q = 2
def a (n : ℕ) : ℕ := sorry

-- Condition: the sum of the first 10 logarithms in base 2 equals 35
axiom log_sum_condition :
  (Finset.range 10).sum (λ n, Real.log (a n) / Real.log 2) = 35

theorem geometric_sequence_sum :
  let q := 2 in
  let S_10 := (Finset.range 10).sum (λ n, a n) in
  S_10 = 1023 / 2 := by
  sorry

end geometric_sequence_sum_l416_416484


namespace divides_totient_prime_power_minus_one_l416_416553

noncomputable def euler_totient (x : ℕ) : ℕ := Nat.totient x

theorem divides_totient_prime_power_minus_one 
  (p : ℕ) (n : ℕ)
  (hp_prime : Nat.Prime p) :
  n ∣ euler_totient (p^n - 1) :=
by
  sorry

end divides_totient_prime_power_minus_one_l416_416553


namespace solve_for_x_l416_416584

   theorem solve_for_x (x : ℝ) (h : (27^x * 27^x * 27^x * 27^x = 243^4)) : x = 5 / 3 :=
   by
     sorry
   
end solve_for_x_l416_416584


namespace fraction_of_kiwis_l416_416628

theorem fraction_of_kiwis (total_fruits : ℕ) (num_strawberries : ℕ) (h₁ : total_fruits = 78) (h₂ : num_strawberries = 52) :
  (total_fruits - num_strawberries) / total_fruits = 1 / 3 :=
by
  -- proof to be provided, this is just the statement
  sorry

end fraction_of_kiwis_l416_416628


namespace total_square_footage_l416_416616

-- Definitions from the problem conditions
def price_per_square_foot : ℝ := 98
def total_property_value : ℝ := 333200

-- The mathematical statement to prove
theorem total_square_footage : (total_property_value / price_per_square_foot) = 3400 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end total_square_footage_l416_416616


namespace cost_to_fill_pool_l416_416125

-- Define the given conditions as constants
def filling_time : ℝ := 50
def flow_rate : ℝ := 100
def cost_per_10_gallons : ℝ := 0.01

-- Calculate total volume in gallons
def total_volume : ℝ := filling_time * flow_rate

-- Calculate the cost per gallon in dollars
def cost_per_gallon : ℝ := cost_per_10_gallons / 10

-- Define the total cost to fill the pool in dollars
def total_cost : ℝ := total_volume * cost_per_gallon

-- Prove that the total cost equals $5
theorem cost_to_fill_pool : total_cost = 5 := by
  unfold total_cost
  unfold total_volume
  unfold cost_per_gallon
  unfold filling_time
  unfold flow_rate
  unfold cost_per_10_gallons
  sorry

end cost_to_fill_pool_l416_416125


namespace divisor_of_product_of_four_consecutive_integers_l416_416705

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416705


namespace greatest_divisor_four_consecutive_integers_l416_416826

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416826


namespace proof_problem_l416_416433

theorem proof_problem (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) : 
  z / (z * complex.conj z - 1) = -1/3 + complex.I * real.sqrt 3 / 3 :=
by
  sorry

end proof_problem_l416_416433


namespace probability_of_summer_and_autumn_l416_416592

theorem probability_of_summer_and_autumn :
  let stamps := { "Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold" }
  let draws := (stamps.powerset.filter (fun s => s.card = 2)).toList
  let favorable := draws.count (fun s => "Summer Begins" ∈ s ∧ "Autumn Equinox" ∈ s)
  (favorable : ℚ) / draws.length = 1 / 6 := by
  sorry

end probability_of_summer_and_autumn_l416_416592


namespace circumscribed_circle_radius_l416_416392

noncomputable def radius_of_circumscribed_circle 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) : ℝ :=
2

theorem circumscribed_circle_radius 
  {a b c A B C : ℝ} 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) :
  radius_of_circumscribed_circle a b c A B C h1 h2 h3 = 2 :=
sorry

end circumscribed_circle_radius_l416_416392


namespace div_product_four_consecutive_integers_l416_416788

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416788


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416959

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416959


namespace percentage_markup_l416_416611

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 3000 for a computer table, and the cost price of the computer table was Rs. 2500.
Prove that the percentage markup on the cost price is 20%.
-/
theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 3000) (h₂ : cost_price = 2500) :
  ((selling_price - cost_price) / cost_price) * 100 = 20 :=
by
  -- proof omitted
  sorry

end percentage_markup_l416_416611


namespace integer_roots_polynomial_l416_416318

theorem integer_roots_polynomial 
(m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∃ a b c : ℤ, a + b + c = 17 ∧ a * b * c = n^2 ∧ a * b + b * c + c * a = m) ↔ 
  (m, n) = (80, 10) ∨ (m, n) = (88, 12) ∨ (m, n) = (80, 8) ∨ (m, n) = (90, 12) := 
sorry

end integer_roots_polynomial_l416_416318


namespace hexagon_same_length_probability_l416_416519

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l416_416519


namespace div_product_four_consecutive_integers_l416_416796

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416796


namespace tim_will_cover_60_miles_l416_416218

noncomputable def tim_and_élan_meeting_distance
  (d : ℕ) (t_speed : ℕ) (e_speed : ℕ)
  (speed_doubling : ℕ → ℕ → ℕ) : ℕ :=
let first_hour_distance := t_speed + e_speed in
let remaining_distance := d - first_hour_distance in
let second_hour_speed := speed_doubling t_speed e_speed in
let second_hour_distance := second_hour_speed + second_hour_speed / 2 in
let remaining_distance_after_second_hour := remaining_distance - second_hour_distance in
let third_hour_speed := speed_doubling (2 * t_speed) (2 * e_speed) in
if remaining_distance_after_second_hour = 0 then
  (t_speed + 2 * t_speed)
else
  2/3 * remaining_distance_after_second_hour + (10 + 20 + 30)

theorem tim_will_cover_60_miles :
  tim_and_élan_meeting_distance 90 10 5 (fun x y => 2 * (x + y)) = 60 := by
sorry

end tim_will_cover_60_miles_l416_416218


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416933

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416933


namespace four_consecutive_integers_divisible_by_12_l416_416772

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416772


namespace four_consecutive_integers_divisible_by_12_l416_416766

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416766


namespace distance_between_points_l416_416190

-- Define the points as pair of real numbers
def point1 := (3 : ℝ, 6 : ℝ)
def point2 := (-5 : ℝ, 2 : ℝ)

-- Define the distance function in Euclidean space for two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem to be proved
theorem distance_between_points :
  distance point1 point2 = 4 * real.sqrt 5 :=
by
  -- Proof placeholder
  sorry

end distance_between_points_l416_416190


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416897

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416897


namespace standard_eq_curve_C_cartesian_eq_line_L_min_distance_P_to_L_l416_416477

-- Defining the parametric equations of curve C
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sin θ)

-- Defining the polar equation of line L
def polar_line_L (θ : ℝ) : ℝ :=
  3 * Real.sqrt 2 / (Real.cos θ + 2 * Real.sin θ)

-- Standard equation of curve C
theorem standard_eq_curve_C (x y θ : ℝ) 
  (h1 : x = 2 * Real.cos θ)
  (h2 : y = Real.sin θ) : 
  x^2 / 4 + y^2 = 1 := 
  sorry

-- Cartesian equation of line L
theorem cartesian_eq_line_L (x y θ ρ : ℝ)
  (h1 : ρ = 3 * Real.sqrt 2 / (Real.cos θ + 2 * Real.sin θ))
  (h2 : x = ρ * Real.cos θ)
  (h3 : y = ρ * Real.sin θ) : 
  x + 2 * y = 3 * Real.sqrt 2 :=
  sorry

-- Minimum distance from point P to line L
theorem min_distance_P_to_L (θ : ℝ) (d : ℝ) :
  d = Real.sqrt 10 / 5 :=
  sorry

end standard_eq_curve_C_cartesian_eq_line_L_min_distance_P_to_L_l416_416477


namespace probability_of_sum_less_than_product_l416_416137

-- Define the problem conditions
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the event condition
def is_valid_pair (a b : ℕ) : Prop := (a ∈ set_of_numbers) ∧ (b ∈ set_of_numbers) ∧ (a * b > a + b)

-- Count the number of valid pairs
noncomputable def count_valid_pairs : ℕ :=
  set_of_numbers.sum (λ a, set_of_numbers.filter (is_valid_pair a).card)

-- Count the total possible pairs
noncomputable def total_pairs : ℕ :=
  set_of_numbers.card * set_of_numbers.card

-- Calculate the probability
noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / total_pairs

-- State the theorem
theorem probability_of_sum_less_than_product :
  probability = 25 / 49 :=
by sorry

end probability_of_sum_less_than_product_l416_416137


namespace max_diff_y_coords_of_intersections_l416_416085

theorem max_diff_y_coords_of_intersections : 
  let f1 := λ x : ℝ, 4 - x^2 + x^3,
      f2 := λ x : ℝ, 2 + 2x^2 + x^3,
      inters := {x | f1 x = f2 x},
      y_coords := λ x : ℝ, 2 + 2 * x^2 + x^3 in
  2 * (2/3)^(3/2) = 
  max (abs (y_coords (sqrt (2/3)) - y_coords (-sqrt (2/3)))) (abs (y_coords (-sqrt (2/3)) - y_coords (sqrt (2/3))))
:= 
by {
  sorry
}

end max_diff_y_coords_of_intersections_l416_416085


namespace angle_AOC_right_l416_416063

theorem angle_AOC_right (A B C D E O : Type) [equilateral_triangle A B C]
  (hD : divides_ratio D A C (1/3)) (hE : divides_ratio E B A (1/3))
  (hBD : intersects_line O B D) (hCE : intersects_line O C E) :
  angle A O C = 90 :=
sorry

end angle_AOC_right_l416_416063


namespace angle_ECB_plus_180_eq_2_angle_EBC_l416_416035

-- Definitions derived from conditions
variables (A B C D E : Type)
variables (angle_ACB angle_ABC : ℝ)

-- Assumptions based on the problem statement
axiom triangle_ABC : A ≠ B ∧ B ≠ C ∧ C ≠ A
axiom angle_condition : angle_ACB = 2 * angle_ABC
axiom point_D_on_BC : ∃ D, 2 * dist B D = dist D C
axiom segment_AD_extended_to_E : ∃ E, dist A D = dist D E

-- Theorem to prove
theorem angle_ECB_plus_180_eq_2_angle_EBC
  (A B C D E : Type)
  (angle_ACB angle_ABC : ℝ)
  (h_triangle: triangle_ABC)
  (h_angle: angle_condition)
  (h_point_D: point_D_on_BC)
  (h_AD_equals_DE: segment_AD_extended_to_E)
  : ∠ ECB + 180 = 2 * ∠ EBC :=
sorry -- Proof goes here

end angle_ECB_plus_180_eq_2_angle_EBC_l416_416035


namespace problem1_problem2_problem3_l416_416350

theorem problem1 (x : ℤ) (h : 263 - x = 108) : x = 155 :=
by sorry

theorem problem2 (x : ℤ) (h : 25 * x = 1950) : x = 78 :=
by sorry

theorem problem3 (x : ℤ) (h : x / 15 = 64) : x = 960 :=
by sorry

end problem1_problem2_problem3_l416_416350


namespace greatest_divisor_of_consecutive_product_l416_416859

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416859


namespace greatest_divisor_of_consecutive_product_l416_416847

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416847


namespace smallest_non_palindrome_power_of_12_l416_416326

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 12 ^ k ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, m > 0 ∧ (∃ j : ℕ, m = 12 ^ j ∧ ¬ is_palindrome m) → m ≥ n) :=
by
  sorry

end smallest_non_palindrome_power_of_12_l416_416326


namespace greatest_divisor_of_four_consecutive_integers_l416_416821

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416821


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416940

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416940


namespace abc_inequality_l416_416037

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) ≥ (a * b + b * c + c * a)^2 :=
sorry

end abc_inequality_l416_416037


namespace total_time_correct_l416_416243

-- Define the given problem parameters
def boat_speed_still_water := 16
def stream_speed_first_section := 4
def stream_speed_second_section := 6
def wind_speed := 2
def distance_first_section := 40
def distance_second_section := 60

-- Compute the effective speeds in each section
def effective_speed_first_section := boat_speed_still_water + stream_speed_first_section + wind_speed
def effective_speed_second_section := boat_speed_still_water + stream_speed_second_section + wind_speed

-- Compute the time taken to travel each section
def time_first_section := distance_first_section / effective_speed_first_section
def time_second_section := distance_second_section / effective_speed_second_section

-- Compute the total time to travel both sections
def total_time_taken := time_first_section + time_second_section

-- Statement of the problem to be proved
theorem total_time_correct : total_time_taken = 4.31818 := by
  sorry

end total_time_correct_l416_416243


namespace complete_square_ratio_l416_416576

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l416_416576


namespace vasya_numbers_l416_416183

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l416_416183


namespace sin_square_non_periodic_l416_416220

noncomputable def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f(x + p) = f(x)

theorem sin_square_non_periodic : ¬ ∃ p > 0, is_periodic_function (λ x, Real.sin (x^2)) p := 
sorry

end sin_square_non_periodic_l416_416220


namespace scientific_notation_of_11090000_l416_416071

theorem scientific_notation_of_11090000 :
  ∃ (x : ℝ) (n : ℤ), 11090000 = x * 10^n ∧ x = 1.109 ∧ n = 7 :=
by
  -- skip the proof
  sorry

end scientific_notation_of_11090000_l416_416071


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416915

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416915


namespace factorial_fraction_l416_416198

theorem factorial_fraction :
  (16.factorial / (6.factorial * 10.factorial) : ℚ) = 728 :=
by
  sorry

end factorial_fraction_l416_416198


namespace greatest_divisor_of_four_consecutive_integers_l416_416963

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416963


namespace hexagon_same_length_probability_l416_416522

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l416_416522


namespace sums_of_powers_l416_416565

theorem sums_of_powers (a b c : ℝ) (s : ℕ → ℝ) (h_eq : ∀ (x : ℝ), a * x^2 + b * x + c = 0) :
  (∀ n : ℕ, n >= 2 → a * s n + b * s (n - 1) + if n = 2 then 2 * c else c * s (n - 2) = 0) :=
sorry

end sums_of_powers_l416_416565


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416903

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416903


namespace ineq_proof_l416_416461

variable (A B C P Q R I: Type) [Inhabited I]

-- Definitions of the points and their properties
def incenter(I A B C : Type) : Prop := 
  ∀ (X : Type), ( (X = A ∨ X = B ∨ X = C) → distance I X = distance from I to sides of triangel X )

def circumcircle_intersect (P Q R A B C: Type) : Prop := 
  (intersect at P on angle bisector A) ∧ 
  (intersect at Q on angle bisector B) ∧ 
  (intersect at R on angle bisector C)

-- The condition is that points P, Q, R are where the angle bisectors intersect the circumcircle
variable (h1 : circumcircle_intersect P Q R A B C)
-- The second condition is that I is the incenter of the triangle and is equidistant from all sides
variable (h2 : incenter I A B C)

theorem ineq_proof (h1: circumcircle_intersect P Q R A B C) 
                    (h2: incenter I A B C)  :
  distance(A, P) + distance(B, Q) + distance(C, R) > distance(A, B) + distance(B, C) + distance(C, A) := 
sorry

end ineq_proof_l416_416461


namespace ways_to_draw_without_first_grade_ways_to_draw_with_each_grade_l416_416629

-- Lean 4 statement for part (1)
def problem1 : Nat := 5

-- Lean 4 statement proving 5 ways to draw 4 products without any first-grade products
theorem ways_to_draw_without_first_grade (products : Fin 8 → Nat) :
  (∃ (f1 : Fin 3 → Fin 8) (f2 : Fin 3 → Fin 8) (f3 : Fin 2 → Fin 8),
    bijective f1 ∧ bijective f2 ∧ bijective f3 ∧ 
    (∀ i, products (f1 i) = 1) ∧
    (∀ i, products (f2 i) = 2) ∧
    (∀ i, products (f3 i) = 3) ∧
    ∃ (d : Fin 4 → Fin 5),
    bijective d ∧
    (∀ i, products (d i) ≠ 1))
  → problem1 = 5 :=
by
  sorry

-- Lean 4 statement for part (2)
def problem2 : Nat := 45

-- Lean 4 statement proving 45 ways to draw 4 products with at least one from each grade
theorem ways_to_draw_with_each_grade (products : Fin 8 → Nat) :
  (∃ (f1 : Fin 3 → Fin 8) (f2 : Fin 3 → Fin 8) (f3 : Fin 2 → Fin 8),
    bijective f1 ∧ bijective f2 ∧ bijective f3 ∧ 
    (∀ i, products (f1 i) = 1) ∧
    (∀ i, products (f2 i) = 2) ∧
    (∀ i, products (f3 i) = 3) ∧
    ∃ (d1 d2 d3 : Fin 4 → Fin 3),
    bijective d1 ∧ bijective d2 ∧ bijective d3 ∧ 
    (∀ i, products (d1 i) ≠ 0) ∧ (∀ i, products (d2 i) ≠ 0))
  → problem2 = 45 :=
by
  sorry

end ways_to_draw_without_first_grade_ways_to_draw_with_each_grade_l416_416629


namespace divisor_of_product_of_four_consecutive_integers_l416_416719

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416719


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416866

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416866


namespace bounded_below_polynomial_count_l416_416077

-- Define the problem in Lean.
theorem bounded_below_polynomial_count :
  ∃ (a : Fin 4 → Fin 4 → Bool), 
    (∀ (c : Fin 4 → Fin 4 → ℝ), 
      (∀ i j, c i j > 0) → 
        ∃ (S : Finset (Fin 4 × Fin 4)), 
        (∀ (i j : Fin 4), ((i, j) ∈ S ↔ a i j = true)) → 
          let f := ∑ (i j : Fin 4), if a i j = true then c i j * (x^i) * (y^j) else 0
          in (∃ (x y : ℝ), f x y > 0) ∧ 
             (∃ (x y : ℝ), f x y < 0)) ∧
    (Finset.card (Finset.filter (λ (a : (Fin 4 × Fin 4)), 
      (a.1 % 2 = 0 ∧ a.2 % 2 = 0)) (Finset.univ)) = 126) := sorry

end bounded_below_polynomial_count_l416_416077


namespace minimize_tangent_triangle_area_l416_416057

open Real

theorem minimize_tangent_triangle_area {a b x y : ℝ} 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  (∃ x y : ℝ, (x = a / sqrt 2 ∨ x = -a / sqrt 2) ∧ (y = b / sqrt 2 ∨ y = -b / sqrt 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_tangent_triangle_area_l416_416057


namespace balls_sold_eq_13_l416_416056

-- Let SP be the selling price, CP be the cost price per ball, and loss be the loss incurred.
def SP : ℕ := 720
def CP : ℕ := 90
def loss : ℕ := 5 * CP
def total_CP (n : ℕ) : ℕ := n * CP

-- Given the conditions:
axiom loss_eq : loss = 5 * CP
axiom ball_CP_value : CP = 90
axiom selling_price_value : SP = 720

-- Loss is defined as total cost price minus selling price
def calculated_loss (n : ℕ) : ℕ := total_CP n - SP

-- The proof statement:
theorem balls_sold_eq_13 (n : ℕ) (h1 : calculated_loss n = loss) : n = 13 :=
by sorry

end balls_sold_eq_13_l416_416056


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416961

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416961


namespace greatest_divisor_of_four_consecutive_integers_l416_416818

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416818


namespace hexagon_same_length_probability_l416_416517

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l416_416517


namespace div_product_four_consecutive_integers_l416_416791

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l416_416791


namespace inscribed_polygon_cosine_l416_416224

noncomputable def angle_B (ABC : ℝ) : ℝ := 
  let B := 18 (1 - Mathlib.cos ABC) in
  B

noncomputable def angle_ACE (AC : ℝ) : ℝ :=
  let ACE := 2*AC^2 * (1 - Mathlib.cos AC) = 4 in
  ACE

theorem inscribed_polygon_cosine :
  ∀ (A B C D E : ℝ), A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧ D ∈ Circle ∧ E ∈ Circle ∧
    (AB = 3) ∧ (BC = 3) ∧ (CD = 3) ∧ (DE = 3) ∧ (AE = 2) →
    (1 - Mathlib.cos (angle_B 3)) * (1 - Mathlib.cos (angle_ACE 3)) = (1 / 9) :=
  by
  sorry

end inscribed_polygon_cosine_l416_416224


namespace four_consecutive_integers_divisible_by_12_l416_416781

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l416_416781


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416911

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416911


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416916

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416916


namespace theater_line_permutations_l416_416088

theorem theater_line_permutations : ∃ n : ℕ, n = 8! ∧ n = 40320 := by
  use 8!
  constructor
  . rfl
  . norm_num

end theater_line_permutations_l416_416088


namespace angle_YZX_125_l416_416005

theorem angle_YZX_125 
  (P Q R X Y Z : Point)
  (hPQ_PR : PQ = PR)
  (h_angle_P : angle P = 70)
  (hX_on_QR : X ∈ line QR)
  (hY_on_PR : Y ∈ line PR)
  (hZ_on_PQ : Z ∈ line PQ)
  (hQX_QY : QX = QY)
  (hPX_PZ : PX = PZ) :
  angle YZX = 125 :=
sorry

end angle_YZX_125_l416_416005


namespace alice_basketball_probability_l416_416274

/-- Alice and Bob play a game with a basketball. On each turn, if Alice has the basketball,
 there is a 5/8 chance that she will toss it to Bob and a 3/8 chance that she will keep the basketball.
 If Bob has the basketball, there is a 1/4 chance that he will toss it to Alice, and if he doesn't toss it to Alice,
 he keeps it. Alice starts with the basketball. What is the probability that Alice has the basketball again after two turns? -/
theorem alice_basketball_probability :
  (5 / 8) * (1 / 4) + (3 / 8) * (3 / 8) = 19 / 64 := 
by
  sorry

end alice_basketball_probability_l416_416274


namespace sum_distances_equal_l416_416566

open Complex

theorem sum_distances_equal 
  {n m : ℕ} (h : n = 2 * m + 1)
  {A : Fin n → ℂ} 
  (h_regular : ∀ i, A (⟨i, Nat.lt_succ_of_lt (odd_iff_bodd_eq_one.1 (h ▸ odd_of_not_even (by simp [h])))⟩) = exp (2 * pi * i / n)) 
  {P : ℂ} 
  (hP : P ∈ arc_of_circle_segment 0 1 A 0 (succ m)) :
  (∑ k in Finset.range (m + 1), abs (P - A ⟨2 * k + 1, by linarith [k.zero_le, lt_succ_self m]⟩)) 
  = (∑ k in Finset.range m, abs (P - A ⟨2 * (k+1), by linarith [(k+1).zero_le, add_one_lt_odd h]⟩)) :=
sorry

end sum_distances_equal_l416_416566


namespace solve_equation_l416_416585

theorem solve_equation (x : ℝ) (hx : x + 8 ≥ 0) :
  sqrt (x + 8) - 4 / sqrt (x + 8) = 3 → x = 8 :=
sorry

end solve_equation_l416_416585


namespace smallest_power_of_12_not_palindrome_l416_416333

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def power_of_12_not_palindrome (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 12^k ∧ ¬ is_palindrome (n.repr)

theorem smallest_power_of_12_not_palindrome : ∃ n : ℕ, power_of_12_not_palindrome n ∧ ∀ m : ℕ, power_of_12_not_palindrome m → n ≤ m :=
sorry

end smallest_power_of_12_not_palindrome_l416_416333


namespace number_of_distinct_lines_l416_416083

theorem number_of_distinct_lines (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  (S.card.choose 2) - 2 = 18 :=
by
  -- Conditions
  have hS : S = {1, 2, 3, 4, 5} := h
  -- Conclusion
  sorry

end number_of_distinct_lines_l416_416083


namespace smallest_positive_integer_x_l416_416307

theorem smallest_positive_integer_x (p q r s u v w t: ℕ) (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (hu: 0 < u) (hv: 0 < v) (hw: 0 < w) (ht: 0 < t) :
  (145 = (p.factorial * q.factorial) / (r.factorial * s.factorial) + (u.factorial * v.factorial) / (w.factorial * t.factorial))
  → (∀ p' q' r' s' u' v' w' t', 
      (145 = (p'.factorial * q'.factorial) / (r'.factorial * s'.factorial) + (u'.factorial * v'.factorial) / (w'.factorial * t'.factorial)) 
      → (p + r + u + w) ≤ (p' + r' + u' + w'))
  → (u + w = 7) :=
sorry

end smallest_positive_integer_x_l416_416307


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416905

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416905


namespace nested_op_value_is_4_l416_416412

-- Define the operation
def op (a b c : ℚ) (h : c ≠ 0) : ℚ := (2 * a + b) / c

-- Define the nested operation function
def nested_op (x y z : ℚ) (h1 : z ≠ 0) : ℚ :=
  op (op x y z h1) (op x y z h1) (op x y z h1) (by linarith [h1]) -- h1 ensures the inner operations are well-defined

-- The proof that [[30,60,90],[3,6,9],[6,12,18]] using the operation == 4
theorem nested_op_value_is_4 : 
  nested_op (op 30 60 90 (by norm_num)) (op 3 6 9 (by norm_num)) (op 6 12 18 (by norm_num)) (by norm_num) = 4 :=
  sorry

end nested_op_value_is_4_l416_416412


namespace f_n_expression_l416_416143

theorem f_n_expression (n : ℕ) : 
  let f : ℕ → ℤ := λ n, (int.sqrt (4 * n + 1) - 1) % 4
  in (f n) ∷ ℤ

end f_n_expression_l416_416143


namespace Paco_cookies_left_l416_416058

/-
Problem: Paco had 36 cookies. He gave 14 cookies to his friend and ate 10 cookies. How many cookies did Paco have left?
Solution: Paco has 12 cookies left.

To formally state this in Lean:
-/

def initial_cookies := 36
def cookies_given_away := 14
def cookies_eaten := 10

theorem Paco_cookies_left : initial_cookies - (cookies_given_away + cookies_eaten) = 12 :=
by
  sorry

/-
This theorem states that Paco has 12 cookies left given initial conditions.
-/

end Paco_cookies_left_l416_416058


namespace complete_square_ratio_l416_416577

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l416_416577


namespace vasya_numbers_l416_416154

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l416_416154


namespace smallest_non_palindrome_power_of_12_l416_416329

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

theorem smallest_non_palindrome_power_of_12 : ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 12 ^ k ∧ ¬ is_palindrome n ∧ (∀ m : ℕ, m > 0 ∧ (∃ j : ℕ, m = 12 ^ j ∧ ¬ is_palindrome m) → m ≥ n) :=
by
  sorry

end smallest_non_palindrome_power_of_12_l416_416329


namespace divisor_of_product_of_four_consecutive_integers_l416_416712

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416712


namespace greatest_divisor_of_four_consecutive_integers_l416_416978

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l416_416978


namespace students_playing_all_three_sports_l416_416557

variables (F C B F_and_C F_and_B C_and_B F_and_C_and_B N Total : ℕ)

theorem students_playing_all_three_sports : 
  F = 300 ∧ 
  C = 250 ∧ 
  B = 180 ∧ 
  N = 60 ∧ 
  F_and_C = 120 ∧ 
  F_and_B = 80 ∧ 
  C_and_B = 70 ∧ 
  Total = 580 →
  F + C + B - (F_and_C + F_and_B + C_and_B) + F_and_C_and_B + N = Total →
  F_and_C_and_B = 140 
:= by
  intros h1 h2
  let ⟨hF, hC, hB, hN, hF_and_C, hF_and_B, hC_and_B, hTotal⟩ := h1
  have h := calc 
    F + C + B - (F_and_C + F_and_B + C_and_B) + F_and_C_and_B + N
        = 300 + 250 + 180 - (120 + 80 + 70) + F_and_C_and_B + 60 : by rw [hF, hC, hB, hF_and_C, hF_and_B, hC_and_B, hN]
    ... = 710 - 270 + F_and_C_and_B + 60 : by ring
    ... = 440 + F_and_C_and_B + 60 : by ring
    ... = 500 + F_and_C_and_B : by ring
  have h_final : 500 + F_and_C_and_B = 580 := by rw [h]
  have h_F_and_C_and_B : F_and_C_and_B = 80 := by linarith
  exact h_F_and_C_and_B

end students_playing_all_three_sports_l416_416557


namespace tina_total_leftover_l416_416119

def monthly_income : ℝ := 1000

def june_savings : ℝ := 0.25 * monthly_income
def june_expenses : ℝ := 200 + 0.05 * monthly_income
def june_leftover : ℝ := monthly_income - june_savings - june_expenses

def july_savings : ℝ := 0.20 * monthly_income
def july_expenses : ℝ := 250 + 0.15 * monthly_income
def july_leftover : ℝ := monthly_income - july_savings - july_expenses

def august_savings : ℝ := 0.30 * monthly_income
def august_expenses : ℝ := 250 + 50 + 0.10 * monthly_income
def august_gift : ℝ := 50
def august_leftover : ℝ := (monthly_income - august_savings - august_expenses) + august_gift

def total_leftover : ℝ :=
  june_leftover + july_leftover + august_leftover

theorem tina_total_leftover (I : ℝ) (hI : I = 1000) :
  total_leftover = 1250 := by
  rw [←hI] at *
  show total_leftover = 1250
  sorry

end tina_total_leftover_l416_416119


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416951

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416951


namespace greatest_divisor_of_four_consecutive_integers_l416_416816

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416816


namespace vasya_numbers_l416_416169

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l416_416169


namespace find_a_l416_416046

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = -f (-x) a) → a = 1 :=
by
  sorry

end find_a_l416_416046


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416680

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416680


namespace slower_train_speed_is_18_l416_416637

-- Define the necessary conditions
def speed_of_faster_train : ℝ := 162          -- in kmph
def crossing_time : ℝ := 33                  -- in seconds
def length_of_faster_train : ℝ := 1320       -- in meters

-- Define the relative speed conversion factor (kmph to m/s)
def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Define the theorem to prove the speed of the slower train
theorem slower_train_speed_is_18 :
  ∃ (v : ℝ), v = 18 ∧ 1320 = (speed_of_faster_train - v) * kmph_to_mps(1) * crossing_time :=
by
  sorry

end slower_train_speed_is_18_l416_416637


namespace find_integer_pairs_l416_416316

theorem find_integer_pairs (n m : ℤ) : n = 2 ∧ m = 2 ↔ 7 ^ m = 5 ^ n + 24 :=
by
  sorry

end find_integer_pairs_l416_416316


namespace hexagon_can_tile_plane_l416_416210

def internal_angle (n : ℕ) : ℝ :=
  if h : n > 2 then (180 * (n - 2) / n) else 0

def can_tile (n : ℕ) : Prop :=
  let angle := internal_angle n in
  ∃ k : ℕ, k > 0 ∧ k * angle = 360

theorem hexagon_can_tile_plane :
  can_tile 6 ∧ ¬ can_tile 5 ∧ ¬ can_tile 7 ∧ ¬ can_tile 8 := by
  sorry

end hexagon_can_tile_plane_l416_416210


namespace concert_distance_l416_416053

theorem concert_distance (d1 d2 : ℕ) (h1 : d1 = 32) (h2 : d2 = 46) :
  d1 + d2 = 78 :=
by
  rw [h1, h2]
  norm_num

end concert_distance_l416_416053


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416941

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416941


namespace four_consecutive_product_divisible_by_12_l416_416983

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l416_416983


namespace Vasya_numbers_l416_416159

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l416_416159


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416906

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416906


namespace Archimedes_cylinder_surface_area_l416_416024

noncomputable def cylinder_surface_area (V_sphere : ℝ) (h_cylinder : ℝ) : ℝ :=
  let R := (3 * (sqrt 3)) ^ (1 / 3)
  let h := 2 * R
  2 * Real.pi * R ^ 2 + 2 * Real.pi * R * h

theorem Archimedes_cylinder_surface_area:
  let V_sphere := 4 * (sqrt 3) * Real.pi in
  let h_cylinder := 2 * (3 * (sqrt 3)) ^ (1 / 3) in
  cylinder_surface_area V_sphere h_cylinder = 18 * Real.pi
:= by
  -- Proof steps are not provided
  sorry

end Archimedes_cylinder_surface_area_l416_416024


namespace even_cubes_book_painted_l416_416261

-- Define the dimensions of the block
def length : ℕ := 3
def width : ℕ := 4
def height : ℕ := 2

-- Condition: The block is painted and then cut into 1-inch cubes
def is_painted (x y z : ℕ) : Prop := 
  x < length ∧ y < width ∧ z < height ∧ 
  (x = 0 ∨ x = length - 1 ∨ y = 0 ∨ y = width - 1 ∨ z = 0 ∨ z = height - 1)

-- Calculate the number of cubes with even number of blue faces
def even_blue_faces_cubes : ℕ := 12

-- Prove that the number of cubes with even number of blue faces is 12
theorem even_cubes_book_painted : ∃ n : ℕ, n = 12 :=
by 
  existsi even_blue_faces_cubes
  sorry

end even_cubes_book_painted_l416_416261


namespace least_n_sum_zeros_gt_500000_l416_416299

noncomputable def f_seq : ℕ → (ℝ → ℝ)
| 1 := λ x, abs (x - 1)
| (n + 2) := λ x, f_seq (n + 1) (abs (x - (n + 2)))

noncomputable def sum_zeros : ℕ → ℝ 
| 1 := 1
| 2 := 4
| 3 := 12
| 4 := 28
| 5 := 55
| n := sorry -- The full calculation or pattern for general n is abstracted here.

theorem least_n_sum_zeros_gt_500000 :
  ∃ n : ℕ, n ≥ 101 ∧ sum_zeros n > 500000 :=
begin
  use 101,
  split,
  { exact nat.le_refl 101 },
  { -- This part would involve demonstrating that sum_zeros 101 > 500000
    -- Which typically involves proving that sum_zeros 101 does exceed 500,000
    sorry
  }
end

end least_n_sum_zeros_gt_500000_l416_416299


namespace smallest_positive_non_palindromic_power_of_12_is_12_l416_416336

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

noncomputable def smallest_non_palindromic_power_of_12 : ℕ :=
  Nat.find (λ n, ∃ k : ℕ, n = 12^k ∧ ¬is_palindrome n)

theorem smallest_positive_non_palindromic_power_of_12_is_12 :
  smallest_non_palindromic_power_of_12 = 12 :=
by
  sorry

end smallest_positive_non_palindromic_power_of_12_is_12_l416_416336


namespace probability_of_same_length_segments_l416_416514

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l416_416514


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416888

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416888


namespace product_of_four_consecutive_integers_divisible_by_12_l416_416868

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l416_416868


namespace hexagon_same_length_probability_l416_416518

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l416_416518


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416923

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416923


namespace solve_average_and_variance_l416_416266

noncomputable def original_average_and_variance (n : ℕ) (x : ℕ → ℝ) 
  (transformed_y : ℕ → ℝ) 
  (average_y : ℝ) 
  (variance_y : ℝ) : Prop :=
  let y := (λ i, 2 * x i - 80) in
  (average_y = 1.2) →
  (variance_y = 4.4) →
  (∑ i in finset.range n, y i) / n = 1.2 →
  (∑ i in finset.range n, x i) / n = 40.6 ∧
  (1 / 2^2 * 4.4) = 1.1

theorem solve_average_and_variance (n : ℕ) (x : ℕ → ℝ) 
  (transformed_y : ℕ → ℝ) 
  (average_y : ℝ := 1.2) 
  (variance_y : ℝ := 4.4) : 
  original_average_and_variance n x transformed_y average_y variance_y :=
begin
  sorry,
end

end solve_average_and_variance_l416_416266


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416947

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416947


namespace intersection_eq_l416_416545

def setM : Set ℝ := { x | x^2 - 2*x < 0 }
def setN : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : setM ∩ setN = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_eq_l416_416545


namespace alcohol_percentage_l416_416072

theorem alcohol_percentage (P : ℝ) : 
  (0.10 * 300) + (P / 100 * 450) = 0.22 * 750 → P = 30 :=
by
  intros h
  sorry

end alcohol_percentage_l416_416072


namespace vasya_numbers_l416_416173

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416173


namespace smallest_positive_non_palindrome_power_of_12_l416_416342

def is_palindrome (n : ℕ) : Bool :=
  let s := toDigits 10 n
  s = s.reverse

theorem smallest_positive_non_palindrome_power_of_12 : ∃ k : ℕ, k > 0 ∧ (12^k = 12 ∧ ¬ is_palindrome (12^k)) :=
by {
  sorry
}

end smallest_positive_non_palindrome_power_of_12_l416_416342


namespace probability_of_same_length_segments_l416_416503

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l416_416503


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416679

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416679


namespace evaluate_expression_l416_416444

-- Define the complex number z and its conjugate
def z := -1 + complex.I * (sqrt 3)
def z_conj := -1 - complex.I * (sqrt 3)

-- Prove the required equivalence
theorem evaluate_expression : 
  (z / (z * z_conj - 1)) = -1/3 + (sqrt 3) / 3 * complex.I := 
by
  have z_def: complex.re z = -1 ∧ complex.im z = sqrt 3 := by simp [z, complex.re, complex.im]
  have z_conj_def: complex.re z_conj = -1 ∧ complex.im z_conj = -sqrt 3 := by simp [z_conj, complex.re, complex.im]
  have z_conj_correct: z_conj = conj z := by simp [z, z_conj, conj]
  have z_mult_z_conj: z * z_conj = 4 := by 
    calc
      z * z_conj = (-1 + complex.I * (sqrt 3)) * (-1 - complex.I * (sqrt 3)) : by simp [z, z_conj]
            ... = (1 - 3) : by
              simp only [complex.mul_def, complex.I_mul_I, complex.I_re, sqr, mul_eq_mul_right_iff, one_add_neg_one_eq_zero]
              ring
            ... = 4 : by ring
  sorry

end evaluate_expression_l416_416444


namespace question_condition_l416_416429

def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
  (1 - 2 * x) * (x + 1) < 0 → x > 1 / 2 ∨ x < -1

theorem question_condition
(x : ℝ) : sufficient_but_not_necessary_condition x := sorry

end question_condition_l416_416429


namespace convert_coordinates_to_polar_l416_416478

def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
let ρ := Real.sqrt (x^2 + y^2) in
let θ := if x = 0 then if y > 0 then Real.pi / 2 else 3 * Real.pi / 2 else Real.atan2 y x in
(ρ, θ)

theorem convert_coordinates_to_polar :
  cartesian_to_polar (-3) 3 = (3 * Real.sqrt 2, 3 * Real.pi / 4) :=
by
  -- sorry: proof goes here
  sorry

end convert_coordinates_to_polar_l416_416478


namespace log2_15_eq_formula_l416_416391

theorem log2_15_eq_formula (a b : ℝ) (h1 : a = Real.log 6 / Real.log 3) (h2 : b = Real.log 20 / Real.log 5) :
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) :=
by
  sorry

end log2_15_eq_formula_l416_416391


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416890

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416890


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416946

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416946


namespace vasya_numbers_l416_416172

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l416_416172


namespace inscribed_polygon_cosine_l416_416222

noncomputable def angle_B (ABC : ℝ) : ℝ := 
  let B := 18 (1 - Mathlib.cos ABC) in
  B

noncomputable def angle_ACE (AC : ℝ) : ℝ :=
  let ACE := 2*AC^2 * (1 - Mathlib.cos AC) = 4 in
  ACE

theorem inscribed_polygon_cosine :
  ∀ (A B C D E : ℝ), A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧ D ∈ Circle ∧ E ∈ Circle ∧
    (AB = 3) ∧ (BC = 3) ∧ (CD = 3) ∧ (DE = 3) ∧ (AE = 2) →
    (1 - Mathlib.cos (angle_B 3)) * (1 - Mathlib.cos (angle_ACE 3)) = (1 / 9) :=
  by
  sorry

end inscribed_polygon_cosine_l416_416222


namespace treaty_of_versailles_signed_on_wednesday_l416_416595

/-- The Treaty of Versailles was signed on June 28, 1919, marking the official end of World War I.
   The war had begun 1,566 days earlier, on a Wednesday, July 28, 1914.
   Determine the day of the week on which the treaty was signed. -/
theorem treaty_of_versailles_signed_on_wednesday :
  let days_in_week := 7
  let num_days := 1566
  let starting_day := 3 -- Representing Wednesday as an integer (e.g., Sun = 0, Mon = 1, ..., Wed = 3, ...)
  (starting_day + num_days) % days_in_week = 3 :=
by
  let days_in_week := 7
  let num_days := 1566
  let starting_day := 3
  have mod_eq_4 : num_days % days_in_week = 4 := by norm_num
  rw [mod_eq_4, nat.add_mod]; norm_num
  sorry

end treaty_of_versailles_signed_on_wednesday_l416_416595


namespace product_of_consecutive_integers_l416_416751

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l416_416751


namespace total_turnips_l416_416554

theorem total_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := 
by sorry

end total_turnips_l416_416554


namespace cost_of_new_pencil_l416_416492

-- Define the conditions
def number_of_sharpenings := 5
def hours_per_sharpening := 1.5
def initial_pencils := 10
def total_writing_hours := 105
def total_spending := 8

-- Derived values
def total_hours_per_pencil := number_of_sharpenings * hours_per_sharpening
def total_pencils_needed := total_writing_hours / total_hours_per_pencil
def additional_pencils_needed := total_pencils_needed - initial_pencils
def cost_per_pencil := total_spending / additional_pencils_needed

-- The theorem to prove
theorem cost_of_new_pencil : cost_per_pencil = 2 := by
  sorry

end cost_of_new_pencil_l416_416492


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416902

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416902


namespace greatest_divisor_of_consecutive_product_l416_416843

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416843


namespace find_second_number_l416_416078

theorem find_second_number:
  ∃ X : ℕ, 1000 + 20 + 1000 + X + 1000 + 40 + 1000 + 10 = 4100 :=
by
  use 30
  sorry

end find_second_number_l416_416078


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416673

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416673


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416952

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416952


namespace value_of_coins_l416_416459

theorem value_of_coins (m : ℕ) : 25 * 25 + 15 * 10 = m * 25 + 40 * 10 ↔ m = 15 :=
by
sorry

end value_of_coins_l416_416459


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416948

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416948


namespace distance_between_points_l416_416640

theorem distance_between_points:
  let A := (2 : ℝ, -3 : ℝ, 1 : ℝ)
  let B := (8 : ℝ, 4 : ℝ, -3 : ℝ)
  dist A B = real.sqrt (101) :=
by
  let x1 := 2
  let y1 := -3
  let z1 := 1
  let x2 := 8
  let y2 := 4
  let z2 := -3
  sorry

end distance_between_points_l416_416640


namespace arithmetic_sequence_max_sum_l416_416379

theorem arithmetic_sequence_max_sum (a d t : ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a > 0) 
  (h2 : (9 * t) = a + 5 * d) 
  (h3 : (11 * t) = a + 4 * d) 
  (h4 : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2) :
  n = 10 :=
sorry

end arithmetic_sequence_max_sum_l416_416379


namespace area_of_pentagon_ADCZB_l416_416068

open Real

variables {AB BC ZH : ℝ}
variables (RectangleABCD : (AB * BC = 2016))
variables (ZH_ratio : (ZH = 4 / 7 * BC))

def pentagon_area (AB BC ZH : ℝ) : ℝ :=
  2016 - (1 / 2) * AB * ZH

theorem area_of_pentagon_ADCZB :
  (ZH_ratio : (ZH = 4 / 7 * BC)) →
  pentagon_area AB BC (4 / 7 * BC) = 1440 :=
by
  introZH_ratio,
  sorry

end area_of_pentagon_ADCZB_l416_416068


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416887

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416887


namespace part1_part2_l416_416406

noncomputable def f (x : ℝ) := 2^x

theorem part1 (f_inv : ℝ → ℝ)
  (h1 : ∀ x, f_inv (f x) = x)
  (h2 : ∀ x, f (f_inv x) = x) 
  (cond : f_inv(x) - f_inv(1 - x) = 1) :
  x = 2 / 3 :=
sorry

theorem part2 
  (cond : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ f(x) + f(1-x) = m) :
  2 * Real.sqrt 2 ≤ m ∧ m ≤ 9 / 2 :=
sorry

end part1_part2_l416_416406


namespace binomial_mod_prime_eq_floor_l416_416541

-- Define the problem's conditions and goal in Lean.
theorem binomial_mod_prime_eq_floor (n p : ℕ) (hp : Nat.Prime p) : (Nat.choose n p) % p = n / p := by
  sorry

end binomial_mod_prime_eq_floor_l416_416541


namespace trace_zero_implies_exists_beta_l416_416529

variable {q : ℕ} [Fact (Nat.Prime q)] {n : ℕ}

noncomputable def trace (α : GaloisField q n) : GaloisField q 1 := 
  Finset.univ.sum (λ i : Fin n, α^(q^i))

theorem trace_zero_implies_exists_beta (α : GaloisField q n) (h : trace α = 0) 
  : ∃ β : GaloisField q n, α = β^(q) - β := 
  sorry

end trace_zero_implies_exists_beta_l416_416529


namespace divisor_of_four_consecutive_integers_l416_416655

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416655


namespace inequality_proof_l416_416368

noncomputable theory

open Real

theorem inequality_proof (a b : ℝ) (n : ℕ) (h₁ : a ≠ b) (h₂ : 0 < n) :
  (a + b) * (a ^ n + b ^ n) < 2 * (a ^ (n + 1) + b ^ (n + 1)) :=
sorry

end inequality_proof_l416_416368


namespace problem_1_problem_2_l416_416032

noncomputable def f (x : ℝ) := (x - Real.exp 1) / Real.exp x

theorem problem_1 :
  (∀ x, f' x > 0 → x ∈ set.Iic (Real.exp 1 + 1)) ∧
  (∀ x, f' x < 0 → x ∈ set.Ioi (Real.exp 1 + 1)) ∧
  (f (Real.exp 1 + 1) = Real.exp (-Real.exp 1 - 1)) :=
sorry

theorem problem_2 (c : ℝ) :
  (∀ x, x ∈ set.Ioi 0 →  2 * |Real.log x - Real.log 2| ≥ f x + c - Real.exp (-2)) → 
  c ≤ (Real.exp 1 - 1) / Real.exp 2 :=
sorry

end problem_1_problem_2_l416_416032


namespace passing_three_levels_probability_l416_416239

def pass_level(n : ℕ) (throws : list ℕ) : Prop :=
  (throws.length = n) ∧ (throws.sum > 2^n)

def probability_of_passing_level_1 : ℚ := 2 / 3
def probability_of_passing_level_2 : ℚ := 5 / 6
def probability_of_passing_level_3 : ℚ := 20 / 27

def probability_of_passing_three_levels_consecutively : ℚ :=
  probability_of_passing_level_1 * probability_of_passing_level_2 * probability_of_passing_level_3

theorem passing_three_levels_probability :
  probability_of_passing_three_levels_consecutively = 100 / 243 :=
by
  sorry

end passing_three_levels_probability_l416_416239


namespace smallest_nonnegative_a_l416_416531

open Real

theorem smallest_nonnegative_a (a b : ℝ) (h_b : b = π / 4)
(sin_eq : ∀ (x : ℤ), sin (a * x + b) = sin (17 * x)) : 
a = 17 - π / 4 := by 
  sorry

end smallest_nonnegative_a_l416_416531


namespace fraction_of_nails_size_2d_l416_416281

theorem fraction_of_nails_size_2d (x : ℝ) :
  (∃ x : ℝ, x + 0.5 = 0.75) → x = 0.25 :=
by
  intro h
  cases h with a ha
  exact Eq.subst ha sorry

end fraction_of_nails_size_2d_l416_416281


namespace solve_problem_l416_416397

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function_property : Prop := ∀ x, f(x - 1) = f(-(x - 1))
def symmetry_about_point : Prop := ∀ x, f(x) = -f(2 - x)

-- Given piecewise function definition in the interval 
def function_definition : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f(x) = a * x - 1

-- Target statement to be proven
theorem solve_problem (h1 : even_function_property f)
                      (h2 : symmetry_about_point f)
                      (h3 : function_definition f a) : f 2022 = -1 :=
sorry

end solve_problem_l416_416397


namespace total_number_of_trapezoids_l416_416481

-- Define the midpoints and regular pentagon.
def is_midpoint {X Y Z M : Type} [AffineSpace X Y Z] (A B : X) :=
  ∃ M : X, dist A M = dist B M ∧ dist A B = 2 * dist A M

def regular_pentagon (A B C D E : Type) : Prop :=
  is_midpoint A B ∧ is_midpoint B C ∧ is_midpoint C D ∧ is_midpoint D E ∧ is_midpoint E A

noncomputable def number_of_trapezoids (A B C D E : Type) [regular_pentagon A B C D E] : ℕ :=
15

-- The theorem to prove
theorem total_number_of_trapezoids (A B C D E : Type) [regular_pentagon A B C D E] : number_of_trapezoids A B C D E = 15 := 
by
  sorry

end total_number_of_trapezoids_l416_416481


namespace increasing_interval_f_l416_416610

noncomputable def f (x : ℝ) : ℝ := log x - 2 * x^2

-- The domain is a condition on the input of the function.
def domain (x : ℝ) : Prop := (0 < x)

-- The interval we claim the function is monotonically increasing.
def increasing_interval (x : ℝ) := (0 < x) ∧ (x < 1/2)

-- The derivative of f.
noncomputable def f_prime (x : ℝ) : ℝ := (1 - 4 * x^2) / x

-- The final theorem statement.
theorem increasing_interval_f :
  ∀ x : ℝ, domain x → increasing_interval x → f_prime x > 0 :=
by 
  sorry

end increasing_interval_f_l416_416610


namespace angle_BAD_eq_angle_EAC_l416_416539

theorem angle_BAD_eq_angle_EAC
  (ABC : Type) [Triangle ABC]
  (I I_a I_b I_c : Point)
  (h1 : Incenter I ABC)
  (h2 : Excenter I_a A ABC)
  (h3 : Excenter I_b B ABC)
  (h4 : Excenter I_c C ABC)
  (D : Point)
  (h5 : OnCircumcircle D ABC)
  (h6 : ¬OnLine D II_a ∧ ¬OnLine D I_bI_c ∧ ¬OnLine D BC)
  (F : Point)
  (h7 : IntersectionPoints (Circumcircle DII_a) (Circumcircle DI_bI_c) = {D, F})
  (E : Point)
  (h8 : Intersection E DF BC) :
  ∠BAD = ∠EAC :=
by
  sorry

end angle_BAD_eq_angle_EAC_l416_416539


namespace project_c_onto_a_sub_b_correct_l416_416423

variables (λ : ℝ)
def a : ℝ × ℝ := (λ + 1, 4)
def b : ℝ × ℝ := (3, λ)
def c : ℝ × ℝ := (1, 2)
def sub_vec (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

def is_opposite (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ (a = (k * b.1, k * b.2))

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2 in
  let mag_sq := v.1 * v.1 + v.2 * v.2 in
  ((dot / mag_sq) * v.1, (dot / mag_sq) * v.2)

theorem project_c_onto_a_sub_b_correct
  (h : is_opposite (a λ) (b λ)) :
  projection c (sub_vec (a λ) (b λ)) = (-3/5, 4/5) :=
sorry

end project_c_onto_a_sub_b_correct_l416_416423


namespace CarlaDailyItems_l416_416290

theorem CarlaDailyItems (leaves bugs days : ℕ) 
  (h_leaves : leaves = 30) 
  (h_bugs : bugs = 20) 
  (h_days : days = 10) : 
  (leaves + bugs) / days = 5 := 
by 
  sorry

end CarlaDailyItems_l416_416290


namespace sum_of_digits_product_l416_416280

theorem sum_of_digits_product :
  let anya_product := 2^20
  let vanya_product := 5^17
  let combined_product := anya_product * vanya_product
  let final_number := 8 * 10^17
  (sum_digits final_number) = 8 :=
by
  sorry

end sum_of_digits_product_l416_416280


namespace markup_percentage_correct_l416_416253

def purchase_price : ℝ := 240
def gross_profit : ℝ := 16
def reduction_percent : ℝ := 0.20
def one_hundred_percent : ℝ := 1.0

noncomputable def original_markup_percentage (M : ℝ) :=
  let original_selling_price := purchase_price * (one_hundred_percent + M)
  let new_selling_price := original_selling_price * (one_hundred_percent - reduction_percent)
  new_selling_price - purchase_price = gross_profit

theorem markup_percentage_correct :
  ∃ M : ℝ, original_markup_percentage M ∧ M ≈ 0.25083 := 
sorry

end markup_percentage_correct_l416_416253


namespace max_real_part_sum_l416_416537

noncomputable def max_real_sum (z w : Fin 10 → ℂ) :=
  let z := λ j, 8 * ((Complex.cos (2 * Real.pi * j / 10)) + Complex.i * (Complex.sin (2 * Real.pi * j / 10)))
  let w := λ j, if (Complex.cos (2 * Real.pi * j / 10) > 0) then z j else -Complex.i * (z j)
  (Fin.sum w).re

theorem max_real_part_sum : max_real_sum = 8 + 16 * ((Real.sqrt 5 + 1) / 4 + (-Real.sqrt 5 + 1) / 4 + (3 / 2) * (Real.sqrt (10 - 2 * Real.sqrt 5) / 4)) :=
by
  sorry

end max_real_part_sum_l416_416537


namespace area_union_of_rotated_triangle_l416_416004

/-- In triangle PQR, with specific side lengths and centroid H, points P', Q', and R' are 
    obtained by 180° rotation about H. Prove the area of the union of triangles PQR and P'Q'R' is 78. -/
theorem area_union_of_rotated_triangle
  (P Q R H P' Q' R' : Point)
  (PQ_length : dist P Q = 12)
  (QR_length : dist Q R = 15)
  (PR_length : dist P R = 13)
  (H_centroid : H = centroid P Q R)
  (P'_rotated : P' = rotate_180 H P)
  (Q'_rotated : Q' = rotate_180 H Q)
  (R'_rotated : R' = rotate_180 H R)
  : area (triangle_union P Q R P' Q' R') = 78 :=
sorry

end area_union_of_rotated_triangle_l416_416004


namespace group_size_l416_416621

theorem group_size (boys girls groups : ℕ) (total_students : boys + girls) (h_boys : boys = 26) (h_girls : girls = 46) (h_groups : groups = 8) :
  total_students / groups = 9 :=
by
  have h_total_students : total_students = 72 := by sorry
  have h : 72 / 8 = 9 := by sorry
  exact h

end group_size_l416_416621


namespace num_pens_multiple_of_16_l416_416089

theorem num_pens_multiple_of_16 (Pencils Students : ℕ) (h1 : Pencils = 928) (h2 : Students = 16)
  (h3 : ∃ (Pn : ℕ), Pencils = Pn * Students) :
  ∃ (k : ℕ), ∃ (Pens : ℕ), Pens = 16 * k :=
by
  sorry

end num_pens_multiple_of_16_l416_416089


namespace factorial_ratio_l416_416199

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end factorial_ratio_l416_416199


namespace divisor_of_four_consecutive_integers_l416_416657

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416657


namespace greatest_divisor_of_four_consecutive_integers_l416_416811

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l416_416811


namespace inscribed_circle_radius_inequality_l416_416232

open Real

variables (ABC ABD BDC : Type) -- Representing the triangles

noncomputable def r (ABC : Type) : ℝ := sorry -- radius of the inscribed circle in ABC
noncomputable def r1 (ABD : Type) : ℝ := sorry -- radius of the inscribed circle in ABD
noncomputable def r2 (BDC : Type) : ℝ := sorry -- radius of the inscribed circle in BDC

noncomputable def p (ABC : Type) : ℝ := sorry -- semiperimeter of ABC
noncomputable def p1 (ABD : Type) : ℝ := sorry -- semiperimeter of ABD
noncomputable def p2 (BDC : Type) : ℝ := sorry -- semiperimeter of BDC

noncomputable def S (ABC : Type) : ℝ := sorry -- area of ABC
noncomputable def S1 (ABD : Type) : ℝ := sorry -- area of ABD
noncomputable def S2 (BDC : Type) : ℝ := sorry -- area of BDC

lemma triangle_area_sum (ABC ABD BDC : Type) :
  S ABC = S1 ABD + S2 BDC := sorry

lemma semiperimeter_area_relation (ABC ABD BDC : Type) :
  S ABC = p ABC * r ABC ∧
  S1 ABD = p1 ABD * r1 ABD ∧
  S2 BDC = p2 BDC * r2 BDC := sorry

theorem inscribed_circle_radius_inequality (ABC ABD BDC : Type) :
  r1 ABD + r2 BDC > r ABC := sorry

end inscribed_circle_radius_inequality_l416_416232


namespace probability_correct_l416_416546

structure SockDrawSetup where
  total_socks : ℕ
  color_pairs : ℕ
  socks_per_color : ℕ
  draw_size : ℕ

noncomputable def probability_one_pair (S : SockDrawSetup) : ℚ :=
  let total_combinations := Nat.choose S.total_socks S.draw_size
  let favorable_combinations := (Nat.choose S.color_pairs 3) * (Nat.choose 3 1) * 2 * 2
  favorable_combinations / total_combinations

theorem probability_correct (S : SockDrawSetup) (h1 : S.total_socks = 12) (h2 : S.color_pairs = 6) (h3 : S.socks_per_color = 2) (h4 : S.draw_size = 6) :
  probability_one_pair S = 20 / 77 :=
by
  apply sorry

end probability_correct_l416_416546


namespace complex_div_conjugate_l416_416438

theorem complex_div_conjugate (z : ℂ) (hz : z = -1 + complex.I * real.sqrt 3) :
  z / (z * conj(z) - 1) = -1 / 3 + (complex.I * real.sqrt 3) / 3 :=
by
  sorry

end complex_div_conjugate_l416_416438


namespace num_partitions_at_least_one_element_num_partitions_at_least_two_elements_l416_416473

-- Part 1: At least one element
theorem num_partitions_at_least_one_element (n : ℕ) :
  (let N := {1, 2, ..., n}
   let A_i := {a_1, a_2, ..., a_k} -- each A_i forms arithmetic progression with common difference d
   ∑ i in range(1, n), (2^(n-i)) = 2^n - 2) :=
sorry

-- Part 2: At least two elements
theorem num_partitions_at_least_two_elements (n : ℕ) :
  (let N := {1, 2, ..., n}
   let A_i := {a_1, a_2, ..., a_k} -- each A_i forms arithmetic progression with common difference d, length at least 2
   ∑ i in range(1, n), (2^(n-i) - i) = 2^n - (n * (n - 1)) / 2 - 2) :=
sorry

end num_partitions_at_least_one_element_num_partitions_at_least_two_elements_l416_416473


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416909

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416909


namespace quadratic_roots_sum_square_l416_416048

theorem quadratic_roots_sum_square (u v : ℝ) 
  (h1 : u^2 - 5*u + 3 = 0) (h2 : v^2 - 5*v + 3 = 0) 
  (h3 : u ≠ v) : u^2 + v^2 + u*v = 22 := 
by
  sorry

end quadratic_roots_sum_square_l416_416048


namespace greatest_divisor_four_consecutive_l416_416695

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l416_416695


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416735

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416735


namespace greatest_divisor_four_consecutive_integers_l416_416828

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l416_416828


namespace product_of_four_consecutive_integers_divisible_by_twelve_l416_416957

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l416_416957


namespace greatest_divisor_of_consecutive_product_l416_416846

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l416_416846


namespace vasya_numbers_l416_416147

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l416_416147


namespace max_value_complex_frac_l416_416029

open Complex

theorem max_value_complex_frac (α β : ℂ) (hβ : abs β = real.sqrt 2) (hαβ : α.conj * β ≠ 0) :
  (∀ z, abs z ≤ abs (β - α) / abs (1 - α.conj * β)) → abs (β - α) / abs (1 - α.conj * β) = real.sqrt 2⁻¹ :=
sorry

end max_value_complex_frac_l416_416029


namespace divisor_of_four_consecutive_integers_l416_416660

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l416_416660


namespace divisor_of_product_of_four_consecutive_integers_l416_416706

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l416_416706


namespace proportion_of_time_travelled_first_quarter_is_correct_l416_416212

variables (D V : ℝ)

-- Condition: Speed for the remaining distance is V.
-- Condition: Speed for the first quarter of the distance is 4V.

def proportion_of_time_travelled_first_quarter (D V : ℝ) : ℝ :=
  let t1 := (D / 4) / (4 * V) in
  let t2 := (3 * D / 4) / V in
  t1 / (t1 + t2)

theorem proportion_of_time_travelled_first_quarter_is_correct (D V : ℝ) :
  proportion_of_time_travelled_first_quarter D V = 1 / 13 :=
sorry

end proportion_of_time_travelled_first_quarter_is_correct_l416_416212


namespace proof_star_ast_l416_416311

noncomputable def star (a b : ℕ) : ℕ := sorry  -- representing binary operation for star
noncomputable def ast (a b : ℕ) : ℕ := sorry  -- representing binary operation for ast

theorem proof_star_ast :
  star 12 2 * ast 9 3 = 2 →
  (star 7 3 * ast 12 6) = 7 / 6 :=
by
  sorry

end proof_star_ast_l416_416311


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416886

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l416_416886


namespace greatest_divisor_of_product_of_four_consecutive_integers_l416_416728

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l416_416728
