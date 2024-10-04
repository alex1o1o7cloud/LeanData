import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Combinatorics.Combination
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Irrational
import Mathlib.Data.Real.Monotone
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.EuclideanDomain
import Mathlib.Probability.Basic
import Mathlib.Set
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real

namespace tan_frac_sum_eq_l346_346283

theorem tan_frac_sum_eq :
  ∀ (x y : ℝ),
  (sin x / cos y + sin y / cos x = 2) →
  (cos x / sin y + cos y / sin x = 8) →
  (tan x / tan y + tan y / tan x = 56 / 5) :=
by
  intro x y,
  intro h1 h2,
  sorry

end tan_frac_sum_eq_l346_346283


namespace sqrt_mul_sqrt_eq_l346_346166

theorem sqrt_mul_sqrt_eq (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) :=
by {
  sorry
}

example : Real.sqrt 2 * Real.sqrt 8 = 4 :=
by {
  have h: Real.sqrt 2 * Real.sqrt 8 = Real.sqrt (2 * 8) := sqrt_mul_sqrt_eq 2 8 (by norm_num) (by norm_num),
  rw h,
  norm_num
}

end sqrt_mul_sqrt_eq_l346_346166


namespace smallest_four_digit_in_pascals_triangle_l346_346472

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346472


namespace square_of_radius_l346_346118

theorem square_of_radius (ER RF GS SH : ℝ) (h1: ER = 22) (h2: RF = 21) (h3: GS = 40) (h4: SH = 35) : ∃ r : ℝ, r^2 ≈ 747.97 :=
by
  let r := 3225 / 118
  have h_square_of_radius : r^2 ≈ 747.97 := 
    sorry 
  exact ⟨r, h_square_of_radius⟩

end square_of_radius_l346_346118


namespace find_z_l346_346243

-- Define the complex number z and the given condition
variable (z : ℂ)
axiom h : z * (3 - complex.i) = 1 - complex.i

-- State the theorem to be proved
theorem find_z : z = (2 / 5) - (1 / 5) * complex.i :=
by
  sorry

end find_z_l346_346243


namespace smallest_four_digit_in_pascals_triangle_l346_346486

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346486


namespace problem_l346_346828

noncomputable def f (x : ℝ) : ℝ := |2 - real.log x|

theorem problem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c)
  (h6 : f a = 2) (h7 : f b = 2) (h8 : f c = 2) :
  a * c / b = 9 :=
sorry

end problem_l346_346828


namespace divisibility_by_7_l346_346006

theorem divisibility_by_7 (n : ℕ) (h : 0 < n) : 7 ∣ (3 ^ (2 * n + 2) - 2 ^ (n + 1)) :=
sorry

end divisibility_by_7_l346_346006


namespace circle_center_radius_l346_346384

open Real

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 6*x = 0 ↔ (x - 3)^2 + y^2 = 9 :=
by sorry

end circle_center_radius_l346_346384


namespace area_enclosed_curves_l346_346676

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^3
def curve2 (x : ℝ) : ℝ := 8 * (6 - x)^3

-- Define the integrand for the area calculation
def integrand (y : ℝ) : ℝ := (6 - (y^((2/3):ℝ))/2) - y^((2/3):ℝ)

-- Define the area calculation using the integral
def area : ℝ := 2 * ∫ y in 0..8, integrand y

theorem area_enclosed_curves : area = 38 + 2/5 :=
by
  sorry

end area_enclosed_curves_l346_346676


namespace snow_probability_l346_346033

theorem snow_probability :
  let P1 := 3 / 4 in
  let P2 := 4 / 5 in
  let P3 := 3 / 4 in
  -- complement probabilities
  let P_no_snow_1_3 := 1 - P1 in
  let P_no_snow_4_if_1_3 := 1 - P2 in
  let P_no_snow_4_if_not_1_3 := 1 - P3 in
  -- probabilities calculations
  let P_no_snow_first_3 := (P_no_snow_1_3)^3 in
  let P_no_snow_all_4_if_not_1_3 := P_no_snow_first_3 * P_no_snow_4_if_not_1_3 in
  let P_snow_at_least_1_3 := 1 - P_no_snow_first_3 in
  let P_no_snow_4_if_at_least_1_3 := P_snow_at_least_1_3 * (P_no_snow_4_if_1_3) in
  let P_no_snow_all_4 := P_no_snow_all_4_if_not_1_3 + P_no_snow_4_if_at_least_1_3 in
  1 - P_no_snow_all_4 = 1023 / 1280 :=
by sorry

end snow_probability_l346_346033


namespace stratified_sampling_l346_346775

-- We are defining the data given in the problem
def numStudents : ℕ := 50
def numFemales : ℕ := 20
def sampledFemales : ℕ := 4
def genderRatio := (numFemales : ℚ) / (numStudents : ℚ)

-- The theorem stating the given problem and its conclusion
theorem stratified_sampling : ∀ (n : ℕ), (sampledFemales : ℚ) / (n : ℚ) = genderRatio → n = 10 :=
by
  intro n
  intro h
  sorry

end stratified_sampling_l346_346775


namespace factor_expression_l346_346973

theorem factor_expression (y : ℝ) : 
  (16 * y ^ 6 + 36 * y ^ 4 - 9) - (4 * y ^ 6 - 6 * y ^ 4 - 9) = 6 * y ^ 4 * (2 * y ^ 2 + 7) := 
by sorry

end factor_expression_l346_346973


namespace find_j_l346_346332

theorem find_j (y : ℝ) (j : ℝ) (h1 : log 9 5 = y) (h2 : log 3 125 = j * y) : j = 6 :=
by
  sorry

end find_j_l346_346332


namespace pie_left_in_fridge_l346_346369

theorem pie_left_in_fridge (weight_ate : ℚ) (fraction_ate : ℚ) (total_pie_weight : ℚ) : 
  weight_ate = 240 ∧ fraction_ate = 1 / 6 ∧ total_pie_weight = weight_ate * 6 →
  (5 / 6) * total_pie_weight = 1200 :=
by
  intro h
  obtain ⟨hw, hf, ht⟩ := h
  rw [hw, hf] at ht
  rw ht
  linarith

end pie_left_in_fridge_l346_346369


namespace smallest_four_digit_number_in_pascals_triangle_l346_346462

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346462


namespace goods_train_length_l346_346931

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) :
  speed_kmph = 72 → platform_length_m = 350 → time_sec = 26 → 
  let speed_mps := speed_kmph * (5 / 18 : ℝ)
      distance_covered := speed_mps * time_sec
  in
  distance_covered - platform_length_m = 170 :=
by
  intro h_speed h_platform h_time
  let speed_mps := 72 * (5 / 18 : ℝ) -- Convert speed to m/s
  let distance_covered := speed_mps * 26 -- Calculate distance
  have h_distance_covered : distance_covered = 520 := by sorry
  have h_length_of_train : 520 - 350 = 170 := by sorry
  exact h_length_of_train

end goods_train_length_l346_346931


namespace largest_sum_value_l346_346229

theorem largest_sum_value (N k : ℕ) (n : ℕ → ℕ) (hN : N ≥ 4) (hk : k ≥ 3)
    (hn1 : ∀ i j, 1 ≤ i → i ≤ j → j ≤ k → n i ≥ n j)
    (hn2 : ∀ i, 1 ≤ i → i ≤ k → n i ≥ 1)
    (hn_sum : (Finset.range k).sum n = N) :
  let m : ℕ := k / 2 in
  (N % 3 = 1 → ∑ i in (Finset.range (m + 1)), (n i / 2) + 1 = 2 * m + 2) ∧
  (N % 3 = 2 → ∑ i in (Finset.range (m + 1)), (n i / 2) + 1 = 2 * m + 2) ∧
  (N % 3 = 0 → ∑ i in (Finset.range (m + 1)), (n i / 2) + 1 = 2 * m + 3) :=
by
  sorry

end largest_sum_value_l346_346229


namespace smallest_four_digit_in_pascal_l346_346504

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346504


namespace squares_and_cubes_less_than_100_l346_346752

theorem squares_and_cubes_less_than_100 : {
  val count := λ s, s.count
  let numbers := { n : ℕ | (∃ k : ℕ, n = k^6) ∧ n < 100 }
  sorry -- This is where the proof would go

-- The expected number of such positive integers
2 = count numbers
} :=
by sorry

end squares_and_cubes_less_than_100_l346_346752


namespace first_month_sale_l346_346932

def sale_second_month : ℕ := 5744
def sale_third_month : ℕ := 5864
def sale_fourth_month : ℕ := 6122
def sale_fifth_month : ℕ := 6588
def sale_sixth_month : ℕ := 4916
def average_sale_six_months : ℕ := 5750

def expected_total_sales : ℕ := 6 * average_sale_six_months
def known_sales : ℕ := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month

theorem first_month_sale :
  (expected_total_sales - (known_sales + sale_sixth_month)) = 5266 :=
by
  sorry

end first_month_sale_l346_346932


namespace find_four_numbers_l346_346197

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def sum_divisible_by (n : ℕ) (a b c d : ℕ) : Prop :=
  n % a = 0 ∨ n % b = 0 ∨ n % c = 0 ∨ n % d = 0

def starts_with_digit (n : ℕ) (d : ℕ) : Prop :=
  (n / 100) = d

theorem find_four_numbers 
  (a b c d : ℕ) 
  (H1 : is_three_digit a) 
  (H2 : is_three_digit b) 
  (H3 : is_three_digit c) 
  (H4 : is_three_digit d) 
  (H5 : starts_with_digit a 1) 
  (H6 : starts_with_digit b 1) 
  (H7 : starts_with_digit c 1) 
  (H8 : starts_with_digit d 1) 
  (H9 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (H10 : sum_divisible_by (a + b + c + d) a b c d) : 
  ∃ (a b c d : ℕ), 
    is_three_digit a ∧ 
    is_three_digit b ∧ 
    is_three_digit c ∧ 
    is_three_digit d ∧ 
    starts_with_digit a 1 ∧ 
    starts_with_digit b 1 ∧ 
    starts_with_digit c 1 ∧ 
    starts_with_digit d 1 ∧ 
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    sum_divisible_by (a + b + c + d) a b c d :=
by {
  let a := 108,
  let b := 117,
  let c := 135,
  let d := 180,

  have ha : is_three_digit a := by 
    unfold is_three_digit; 
    exact ⟨by norm_num, by norm_num⟩,
  have hb : is_three_digit b := by 
    unfold is_three_digit; 
    exact ⟨by norm_num, by norm_num⟩,
  have hc : is_three_digit c := by 
    unfold is_three_digit; 
    exact ⟨by norm_num, by norm_num⟩,
  have hd : is_three_digit d := by 
    unfold is_three_digit; 
    exact ⟨by norm_num, by norm_num⟩,

  have hda : starts_with_digit a 1 := by 
    unfold starts_with_digit; 
    exact by norm_num,
  have hdb : starts_with_digit b 1 := by 
    unfold starts_with_digit; 
    exact by norm_num,
  have hdc : starts_with_digit c 1 := by 
    unfold starts_with_digit; 
    exact by norm_num,
  have hdd : starts_with_digit d 1 := by 
    unfold starts_with_digit; 
    exact by norm_num,

  have hneq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d := by
    repeat {split}; norm_num,

  have hsum : sum_divisible_by (a + b + c + d) a b c d := by 
    unfold sum_divisible_by; 
    norm_num,

  exact ⟨a, b, c, d, ha, hb, hc, hd, hda, hdb, hdc, hdd, hneq, hsum⟩,
}

end find_four_numbers_l346_346197


namespace digit_in_decimal_expansion_l346_346382

theorem digit_in_decimal_expansion (n : ℕ) : 
  (let decimal_expansion := "153846"
   in decimal_expansion[(2008 % decimal_expansion.length) - 1] = '8') :=
by 
  sorry

end digit_in_decimal_expansion_l346_346382


namespace combined_class_average_score_l346_346142

theorem combined_class_average_score
  (avg_A : ℕ := 65) (avg_B : ℕ := 90) (avg_C : ℕ := 77)
  (ratio_A : ℕ := 4) (ratio_B : ℕ := 6) (ratio_C : ℕ := 5) :
  ((avg_A * ratio_A + avg_B * ratio_B + avg_C * ratio_C) / (ratio_A + ratio_B + ratio_C) = 79) :=
by 
  sorry

end combined_class_average_score_l346_346142


namespace part_a_part_b_l346_346844

-- Part (a)
theorem part_a :
  ∃ (S : Finset (Fin 5) → (ℝ × ℝ)), (S.card = 5) ∧ (∀ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k → is_right_triangle (S i) (S j) (S k)) = 8 :=
sorry

-- Part (b)
theorem part_b :
  ∃ (S : Finset (Fin 64) → (ℝ × ℝ)), (S.card = 64) ∧ (∀ (i j k : Fin 64), i ≠ j ∧ j ≠ k ∧ i ≠ k → is_right_triangle (S i) (S j) (S k).card ≥ 2005) :=
sorry

-- Definition of a right triangle (for completeness, not a part of the theorem statement)
def is_right_triangle (A B C : (ℝ × ℝ)) : Prop :=
  ((A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0) ∨
  ((B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0) ∨
  ((C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0)

end part_a_part_b_l346_346844


namespace simplify_sqrt_expr_is_2205_l346_346176

noncomputable def simplify_expr : ℤ :=
  let term1 := real.sqrt 80
  let term2 := 3 * real.sqrt 10
  let term3 := 2 * real.sqrt 500 / real.sqrt 5
  let simplified := term1 - term2 + term3
  real.to_ntr (simplified * simplified)

theorem simplify_sqrt_expr_is_2205 : simplify_expr = 2205 :=
  by
    sorry

end simplify_sqrt_expr_is_2205_l346_346176


namespace count_integers_not_divisible_by_4_or_6_lt_2000_l346_346397

theorem count_integers_not_divisible_by_4_or_6_lt_2000 : 
  let total := 1999 in
  let divisible_by_4 := 499 in
  let divisible_by_6 := 333 in
  let divisible_by_12 := 166 in
  total - (divisible_by_4 + divisible_by_6 - divisible_by_12) = 1333 := by
  let total := 1999
  let divisible_by_4 := 499
  let divisible_by_6 := 333
  let divisible_by_12 := 166
  have count_divisible_by_4_or_6 := divisible_by_4 + divisible_by_6 - divisible_by_12
  have count_not_divisible_by_4_or_6 := total - count_divisible_by_4_or_6
  show count_not_divisible_by_4_or_6 = 1333 from by
    sorry

end count_integers_not_divisible_by_4_or_6_lt_2000_l346_346397


namespace ellipse_foci_on_y_axis_l346_346238

theorem ellipse_foci_on_y_axis (theta : ℝ) (h1 : 0 < theta ∧ theta < π)
  (h2 : Real.sin theta + Real.cos theta = 1 / 2) :
  (0 < theta ∧ theta < π / 2) → 
  (0 < theta ∧ theta < 3 * π / 4) → 
  -- The equation x^2 * sin theta - y^2 * cos theta = 1 represents an ellipse with foci on the y-axis
  ∃ foci_on_y_axis : Prop, foci_on_y_axis := 
sorry

end ellipse_foci_on_y_axis_l346_346238


namespace largest_divisor_n4_minus_5n2_plus_6_l346_346685

theorem largest_divisor_n4_minus_5n2_plus_6 :
  ∀ (n : ℤ), (n^4 - 5 * n^2 + 6) % 1 = 0 :=
by
  sorry

end largest_divisor_n4_minus_5n2_plus_6_l346_346685


namespace trey_more_turtles_than_kristen_l346_346900

theorem trey_more_turtles_than_kristen (kristen_turtles : ℕ) 
  (H1 : kristen_turtles = 12) 
  (H2 : ∀ kris_turtles, kris_turtles = (1 / 4) * kristen_turtles)
  (H3 : ∀ kris_turtles trey_turtles, trey_turtles = 7 * kris_turtles) :
  ∃ trey_turtles, trey_turtles - kristen_turtles = 9 :=
by {
  sorry
}

end trey_more_turtles_than_kristen_l346_346900


namespace factors_of_expr_l346_346387

def expr :=  29 * 26 * (2^48 - 1)

theorem factors_of_expr :
  ∃ a b, 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧ a ≠ b ∧ a * b ∣ expr ∧ a ∣ expr ∧ b ∣ expr :=
  by
    use 63, 65
    split ; linarith
    split ; linarith
    split ; try {simp}
    sorry

end factors_of_expr_l346_346387


namespace weight_of_a_l346_346916

variables (a b c d e : ℝ)

-- Conditions
def condition1 := a + b + c = 252
def condition2 := a + b + c + d = 320
def condition3 := e = d + 3
def condition4 := b + c + d + e = 316

theorem weight_of_a :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 → a = 75 :=
by
  intro h

  -- Destructuring the conjunction
  rcases h with ⟨h1, h2, h3, h4⟩

  -- We don't need to prove; just placeholders
  sorry

end weight_of_a_l346_346916


namespace unique_solution_of_quadratic_l346_346857

theorem unique_solution_of_quadratic (b : ℚ) (h : b ≠ 0) (hs : ∀ x1 x2 : ℚ, (x1 = x2) ↔ (∃ x : ℚ, bx^2 + 15x + 6 = 0) ) : 
∃ x : ℚ, x = -4 / 5 ∧ b = 75 / 8 :=
by
  sorry

end unique_solution_of_quadratic_l346_346857


namespace rational_terms_count_l346_346068

open BigOperators

noncomputable def count_rational_terms : ℕ :=
  (List.range 501).filter (λ k, k % 4 = 0).length

theorem rational_terms_count :
  count_rational_terms = 126 :=
begin
  sorry
end

end rational_terms_count_l346_346068


namespace min_lambda_l346_346216

open Real

-- Define the vectors AB and AC
def AB (x : ℝ) : ℝ × ℝ × ℝ := (x, 1 / x, x)
def AC : ℝ × ℝ × ℝ := (1, 2, 2)

-- Given the condition x > 0, find the minimum value of λ
theorem min_lambda (x : ℝ) (h : 0 < x) :
  let λ := (3 * x + 2 / x) / 9
  in λ >= (2 * sqrt 6) / 9 := sorry

end min_lambda_l346_346216


namespace sum_of_angles_l346_346974

noncomputable def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ)

theorem sum_of_angles (z : ℂ) (θs : list ℝ) (n : ℕ)
  (h1 : z ^ 26 - z ^ 10 - 1 = 0)
  (h2 : complex.abs z = 1)
  (h3 : θs = list.map (λ k, 360 * k / (16 * n)) (list.range (2 * n)))
  (h_sorted : list.sorted (<=) θs)
  (h_unique : list.nodup θs)
  : ∑ i in list.range n, (θs.nth_le (2 * i + 1) sorry) = sorry :=
sorry

end sum_of_angles_l346_346974


namespace solution_set_inequality_l346_346041

theorem solution_set_inequality (x : ℝ) : 
  (3 * x) / (2 * x + 1) ≤ 1 ↔ (x ∈ Ioo (-1 / 2) 1) ∨ (x = 1) :=
by
  sorry

end solution_set_inequality_l346_346041


namespace curve_is_parabola_l346_346680

def curve_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - sin θ)

theorem curve_is_parabola :
  ∀ r θ x y : ℝ,
  curve_eq r θ →
  x = r * cos θ →
  y = r * sin θ →
  y = (x^2 - 1) / 2 :=
by
  sorry

end curve_is_parabola_l346_346680


namespace sophomores_in_sample_l346_346933

-- Define the number of freshmen, sophomores, and juniors
def freshmen : ℕ := 400
def sophomores : ℕ := 600
def juniors : ℕ := 500

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the total number of students in the sample
def sample_size : ℕ := 100

-- Define the expected number of sophomores in the sample
def expected_sophomores : ℕ := (sample_size * sophomores) / total_students

-- Statement of the problem we want to prove
theorem sophomores_in_sample : expected_sophomores = 40 := by
  sorry

end sophomores_in_sample_l346_346933


namespace b_minus_c_l346_346689

noncomputable def a (n : ℕ) (h : n > 1) := 1 / (Real.log 1001 / Real.log n)

def b (h3 : 3 > 1) (h4 : 4 > 1) (h5 : 5 > 1) (h6 : 6 > 1) : ℝ :=
  a 3 h3 + a 4 h4 + a 5 h5 + a 6 h6

def c (h8 : 8 > 1) (h9 : 9 > 1) (h10 : 10 > 1) (h11 : 11 > 1) (h12 : 12 > 1) : ℝ :=
  a 8 h8 + a 9 h9 + a 10 h10 + a 11 h11 + a 12 h12

theorem b_minus_c :
  ∀ (h3 : 3 > 1) (h4 : 4 > 1) (h5 : 5 > 1) (h6 : 6 > 1) (h8 : 8 > 1) (h9 : 9 > 1) (h10 : 10 > 1) (h11 : 11 > 1) (h12 : 12 > 1),
  b h3 h4 h5 h6 - c h8 h9 h10 h11 h12 = Real.log (3 / 792) / Real.log 1001 :=
by sorry

end b_minus_c_l346_346689


namespace ratio_of_trees_l346_346132

theorem ratio_of_trees (plums pears apricots : ℕ) (h_plums : plums = 3) (h_pears : pears = 3) (h_apricots : apricots = 3) :
  plums = pears ∧ pears = apricots :=
by
  sorry

end ratio_of_trees_l346_346132


namespace largest_angle_of_consecutive_integer_angles_of_hexagon_l346_346377

theorem largest_angle_of_consecutive_integer_angles_of_hexagon 
  (angles : Fin 6 → ℝ)
  (h_consecutive : ∃ (x : ℝ), angles = ![
    x - 3, x - 2, x - 1, x, x + 1, x + 2 ])
  (h_sum : (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5) = 720) :
  (angles 5 = 122.5) :=
by
  sorry

end largest_angle_of_consecutive_integer_angles_of_hexagon_l346_346377


namespace smallest_four_digit_in_pascal_l346_346517

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346517


namespace keith_stored_bales_l346_346048

theorem keith_stored_bales (initial_bales added_bales final_bales : ℕ) :
  initial_bales = 22 → final_bales = 89 → final_bales = initial_bales + added_bales → added_bales = 67 :=
by
  intros h_initial h_final h_eq
  sorry

end keith_stored_bales_l346_346048


namespace smallest_four_digit_in_pascals_triangle_l346_346538

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346538


namespace part1_part2_l346_346748

noncomputable def a (n : ℕ) : ℕ := 3 ^ n

noncomputable def b (n : ℕ) : ℕ := 2 * n - 1

theorem part1 (n : ℕ) (h1 : a 1 = 3) (h4 : a 4 = 81) (b2_eq_a1 : b 2 = a 1) (b5_eq_a2 : b 5 = a 2) :
  b n = 2 * n - 1 := 
by
  sorry

theorem part2 (n : ℕ) (h_a : ∀ n, a n = 3 ^ n) : 
  ∑ i in Finset.range n, 1 / (i + 1 * (i + 2)) = n / (n + 1) := 
by
  sorry

end part1_part2_l346_346748


namespace inversely_proportional_find_p_l346_346373

theorem inversely_proportional_find_p (p q : ℕ) (h1 : p * 8 = 160) (h2 : q = 10) : p * q = 160 → p = 16 :=
by
  intro h
  sorry

end inversely_proportional_find_p_l346_346373


namespace appropriate_length_of_presentation_l346_346344

theorem appropriate_length_of_presentation (wpm : ℕ) (min_time min_words max_time max_words total_words : ℕ) 
  (h1 : total_words = 160) 
  (h2 : min_time = 45) 
  (h3 : min_words = min_time * wpm) 
  (h4 : max_time = 60) 
  (h5 : max_words = max_time * wpm) : 
  7200 ≤ 9400 ∧ 9400 ≤ 9600 :=
by 
  sorry

end appropriate_length_of_presentation_l346_346344


namespace ferris_wheel_seat_capacity_l346_346861

theorem ferris_wheel_seat_capacity
  (total_seats : ℕ)
  (broken_seats : ℕ)
  (total_people : ℕ)
  (seats_available : ℕ)
  (people_per_seat : ℕ)
  (h1 : total_seats = 18)
  (h2 : broken_seats = 10)
  (h3 : total_people = 120)
  (h4 : seats_available = total_seats - broken_seats)
  (h5 : people_per_seat = total_people / seats_available) :
  people_per_seat = 15 := 
by sorry

end ferris_wheel_seat_capacity_l346_346861


namespace candy_probability_l346_346896

/-
Question: 
Given three bags with the following compositions:
  - Bag 1: Three green candies, one red candy.
  - Bag 2: Two green candies, two red candies.
  - Bag 3: One green candy, three red candies.
A child randomly selects one of the bags and then randomly picks a first candy:
  - If the first candy is green, the second candy is chosen from one of the other two bags.
  - If the first candy is red, the second candy is chosen from the same bag.
Prove that the probability that the second candy is green can be expressed as the fraction \( \frac{73}{144} \), and thus \( m + n = 217 \).
-/

theorem candy_probability :
  let m := 73
  let n := 144
  m + n = 217 :=
by {
  sorry -- proof is omitted
}

end candy_probability_l346_346896


namespace workers_together_time_l346_346572

-- Definitions: Workers' individual times and job rates
def Worker_A_time := 7 -- time in hours for Worker A to complete one job
def Worker_B_time := 10 -- time in hours for Worker B to complete one job

-- The combined work rate of A and B
def combined_rate := (1 / Worker_A_time) + (1 / Worker_B_time)
-- The time to complete one job together
def combined_time := 1 / combined_rate

-- The main theorem 
theorem workers_together_time : combined_time = 70 / 17 :=
by
  -- Definitions and calculations derived from problem conditions
  have h1 : 1 / Worker_A_time = 1 / 7 := by rfl
  have h2 : 1 / Worker_B_time = 1 / 10 := by rfl
  have h3 : combined_rate = (1 / 7) + (1 / 10) := by rfl
  have h4 : (1 / 7) + (1 / 10) = 17 / 70 :=
    by
      norm_num
  have h5 : combined_time = 1 / (17 / 70) := by rfl
  have h6 : 1 / (17 / 70) = 70 / 17 :=
    by
      field_simp
  exact h6

end workers_together_time_l346_346572


namespace limit_calculation_l346_346966

-- Define the given limit problem and its equivalence to the expected result
theorem limit_calculation :
  (lim (x -> 0) (fun x => (3^(5*x) - 2^(-7*x)) / (2*x - tan x)) = ln (3^5 * 2^7)) :=
by 
  sorry

end limit_calculation_l346_346966


namespace smallest_four_digit_in_pascals_triangle_l346_346467

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346467


namespace tina_total_time_l346_346899

theorem tina_total_time (assignment_time keys_count key_clean_time : ℕ) (keys_left_to_clean : ℕ) (assignment_minutes : ℕ) :
  assignment_time = 15 → 
  keys_count = 25 →
  key_clean_time = 5 →
  keys_left_to_clean = 24 →
  assignment_minutes = assignment_time + (keys_left_to_clean * key_clean_time) →
  assignment_minutes = 135 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h4, h3]
  sorry

end tina_total_time_l346_346899


namespace analytical_expression_satisfies_conditions_l346_346704

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := 1 + Real.exp x

theorem analytical_expression_satisfies_conditions :
  is_increasing f ∧ (∀ x : ℝ, f x > 1) :=
by
  sorry

end analytical_expression_satisfies_conditions_l346_346704


namespace no_exactly_1000_good_in_2000_l346_346633

/-- Dividing natural numbers into good and bad. -/
inductive NumberClass
| good : NumberClass
| bad : NumberClass

open NumberClass

/-- Predicate to determine if a number is good or bad. -/
variable (goodNumber : ℕ → Prop)

axiom good_add_six (m : ℕ) : goodNumber m → goodNumber (m + 6)
axiom bad_add_fifteen (n : ℕ) : ¬ goodNumber n → ¬ goodNumber (n + 15)

/-- The main theorem stating it is impossible to have exactly 1000 good numbers among the first 2000 numbers. -/
theorem no_exactly_1000_good_in_2000 :
  ¬ (∃ S : Finset ℕ, S.card = 1000 ∧ ∀ n ∈ S, goodNumber n ∧ ∀ n ∈ (Finset.range 2000 \ S), ¬ goodNumber n) :=
sorry

end no_exactly_1000_good_in_2000_l346_346633


namespace number_of_combinations_with_constraints_l346_346141

theorem number_of_combinations_with_constraints :
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose n k
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 13 :=
by
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose 6 2
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 13
  sorry

end number_of_combinations_with_constraints_l346_346141


namespace number_of_correct_props_l346_346635

noncomputable theory

open Set

def prop1 : Prop := ∅ ≠ ({0} : Set ℕ)
def prop2 : Prop := ¬ (∀ s, s ≠ ∅ → (∅ ⊆ s) → False)
def prop3 : Prop := ¬ (∀ s, s ≠ ∅ → (∃ t1 t2 : Set ℕ, t1 ≠ t2 ∧ t1 ⊆ s ∧ t2 ⊆ s))
def prop4 : Prop := ∀ (s : Set ℕ), ∅ ⊆ s

theorem number_of_correct_props : (1 : ℕ) = (if prop1 then 1 else 0) +
                                      (if prop2 then 1 else 0) +
                                      (if prop3 then 1 else 0) +
                                      (if prop4 then 1 else 0) := 
by
  sorry

end number_of_correct_props_l346_346635


namespace abs_gt_x_iff_x_lt_0_l346_346919

theorem abs_gt_x_iff_x_lt_0 (x : ℝ) : |x| > x ↔ x < 0 := 
by
  sorry

end abs_gt_x_iff_x_lt_0_l346_346919


namespace solution_set_inequality_l346_346816

noncomputable def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x : ℝ, x > 0 → (x * (deriv f x) - f x) / x^2 < 0

theorem solution_set_inequality :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
begin
  sorry
end

end solution_set_inequality_l346_346816


namespace probability_outside_circle_l346_346765

   theorem probability_outside_circle :
     let outcomes := [(m, n) | m ← ([1, 2, 3, 4, 5, 6] : List ℕ), n ← ([1, 2, 3, 4, 5, 6] : List ℕ)],
         inside_circle := (m: ℕ) (n: ℕ) : (m^2 + n^2 < 16),
         outside_circle := (m: ℕ) (n: ℕ) : ¬inside_circle m n in
     (List.length (outcomes.filter (λ (P: ℕ × ℕ), outside_circle P.fst P.snd)).toList / List.length outcomes.toList) = 7 / 9 :=
   sorry
   
end probability_outside_circle_l346_346765


namespace complex_problem_l346_346294

noncomputable def z (a : ℝ) : ℂ := (a - 2 * (complex.I)) / 2

theorem complex_problem
  (a : ℝ)
  (h : (z a).im = - (z a).re) :
  z a * complex.conj(z a) = 2 :=
by
  sorry

end complex_problem_l346_346294


namespace find_f_g_3_l346_346814

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_f_g_3 :
  f (g 3) = -2 := by
  sorry

end find_f_g_3_l346_346814


namespace max_squares_crossed_by_straight_line_l346_346627

-- Define the checkerboard as an 8 by 8 grid.
def checkerboard := matrix (Fin 8) (Fin 8) (Fin 1)

-- Define the line crossing through the checkerboard.
def straight_line {α : Type*} (board : matrix (Fin 8) (Fin 8) α) := sorry

-- The theorem we want to prove.
theorem max_squares_crossed_by_straight_line :
  ∀ (board : checkerboard),
  ∃ line : straight_line board, 
    (count_squares_crossed line board) = 15 := 
sorry

end max_squares_crossed_by_straight_line_l346_346627


namespace length_of_train_is_approximately_90_l346_346939

noncomputable def speed_kmph := 36 -- Speed of the train in kmph
noncomputable def time_sec := 8.999280057595392 -- Time in seconds

noncomputable def speed_mps := (speed_kmph * 1000) / 3600 -- Convert speed from kmph to m/s
noncomputable def length_of_train := speed_mps * time_sec -- Calculate the length of the train

theorem length_of_train_is_approximately_90 :
  length_of_train ≈ 89.99280057595392 := sorry

end length_of_train_is_approximately_90_l346_346939


namespace percentage_decrease_equivalent_l346_346886

theorem percentage_decrease_equivalent :
  ∀ (P D : ℝ), 
    (D = 10) →
    ((1.25 * P) - (D / 100) * (1.25 * P) = 1.125 * P) :=
by
  intros P D h
  rw [h]
  sorry

end percentage_decrease_equivalent_l346_346886


namespace a_investment_correct_l346_346913

noncomputable def a_investment 
  (b_profit_share : ℝ) (diff_a_c_profit : ℝ) (b_invest : ℝ) (c_invest : ℝ) : ℝ :=
  let P := b_invest / b_profit_share in
  let x := (diff_a_c_profit + (c_invest / P)) * P in
  x

theorem a_investment_correct : 
  a_investment 3500 1399.9999999999998 10000 12000 = 12628.57142857143 :=
by
  sorry

end a_investment_correct_l346_346913


namespace range_of_lambda_max_triangle_area_l346_346228

noncomputable def ellipse_equation : string :=
  "x^2 / 2 + y^2 = 1"

theorem range_of_lambda :
  ∃ (λ : ℝ), -2 < λ ∧ λ < 2 ∧ λ ≠ 0 := sorry

theorem max_triangle_area :
  let S := sqrt(2) / 2
  ∃ (λ : ℝ), (λ = sqrt 2 ∨ λ = -sqrt 2) ∧ 
             max_area = S :=
  sorry

end range_of_lambda_max_triangle_area_l346_346228


namespace range_of_g_l346_346652

def g (x : ℝ) : ℝ := 1 / (x - 1) ^ 2

theorem range_of_g : Set.range g = {y : ℝ | 0 < y} :=
by
  sorry

end range_of_g_l346_346652


namespace sum_of_money_l346_346084

theorem sum_of_money (P R : ℝ) (h : (P * 2 * (R + 3) / 100) = (P * 2 * R / 100) + 300) : P = 5000 :=
by
    -- We are given that the sum of money put at 2 years SI rate is Rs. 300 more when rate is increased by 3%.
    sorry

end sum_of_money_l346_346084


namespace smallest_four_digit_number_in_pascals_triangle_l346_346443

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346443


namespace graph_of_conic_is_pair_of_lines_l346_346390

theorem graph_of_conic_is_pair_of_lines :
  ∀ {x y : ℝ}, x^2 - 9 * y^2 = 0 ↔ (x = 3 * y ∨ x = -3 * y) :=
by
  intro x y
  split
  sorry

end graph_of_conic_is_pair_of_lines_l346_346390


namespace normal_probability_l346_346729

variables (X : ℝ → ℝ) (σ : ℝ)
noncomputable def normal_dist (x : ℝ) := 
  1 / (σ * real.sqrt (2 * real.pi)) * real.exp (-(x - 2) ^ 2 / (2 * σ ^ 2))

def prob_le (μ : ℝ) (σ : ℝ) (z : ℝ) : ℝ := 
  1 / 2 * (1 + real.erf ((z - μ) / (σ * real.sqrt 2)))

theorem normal_probability (h1 : ∀x, X x = normal_dist x) (h2 : prob_le 2 σ 4 = 0.84) :
  prob_le 2 σ 0 = 0.16 :=
sorry

end normal_probability_l346_346729


namespace smallest_valid_N_l346_346616

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (fun d => d > 1 ∧ d < n ∧ n % d = 0)

def two_largest (l : List ℕ) : ℕ × ℕ :=
  match l.reverse with
  | a :: b :: _ => (a, b)
  | _           => (0, 0)

def two_smallest (l : List ℕ) : ℕ × ℕ :=
  match l with
  | a :: b :: _ => (a, b)
  | _           => (0, 0)

def valid_number (N : ℕ) : Prop :=
  N % 10 = 5 ∧
  let divisors := proper_divisors N in
  let (d_max1, d_max2) := two_largest divisors in
  let (d_min1, d_min2) := two_smallest divisors in
  (d_max1 + d_max2) % (d_min1 + d_min2) ≠ 0

theorem smallest_valid_N : ∃ N : ℕ, valid_number N ∧ N = 725 :=
by
  existsi 725
  unfold valid_number
  unfold proper_divisors
  unfold two_largest
  unfold two_smallest
  sorry

end smallest_valid_N_l346_346616


namespace intersection_probability_l346_346936

theorem intersection_probability (a b : ℝ)
  (h : a^2 + b^2 ≤ 1/4) :
  let p := (calculate_intersection_probability a b)
  in 100 * p = 61 :=
sorry  -- Proof skipped

noncomputable def calculate_intersection_probability (a b : ℝ) : ℝ :=
  -- Calculate the intersection probability based on the given a and b.
  -- This is a placeholder; the actual implementation is omitted.
  0.61  -- Assuming the known result for illustration purposes

end intersection_probability_l346_346936


namespace least_positive_integer_with_seven_distinct_factors_l346_346064

theorem least_positive_integer_with_seven_distinct_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ 1 → (∀ k : ℕ, k ∣ n → k ∣ m → k ∣ m)) → n = 64) ∧
    (∃ p : ℕ, (nat.prime p ∧ n = p^6)) :=
begin
  sorry
end

end least_positive_integer_with_seven_distinct_factors_l346_346064


namespace convert_mps_to_kmph_l346_346668

theorem convert_mps_to_kmph (speed_mps : ℝ) (conversion_factor : ℝ) (speed_kmph : ℝ) :
  speed_mps = 20 → conversion_factor = 3.6 → speed_kmph = speed_mps * conversion_factor → speed_kmph = 72 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end convert_mps_to_kmph_l346_346668


namespace sin_inequalities_correct_l346_346078

theorem sin_inequalities_correct :
  sin 11 * pi / 180 < sin 168 * pi / 180 ∧ sin 168 * pi / 180 < cos 10 * pi / 180 := 
sorry

end sin_inequalities_correct_l346_346078


namespace part1_a2_part1_a3_part2_general_formula_part3_find_m_l346_346225

open Real

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0 else
    nat.rec_on n (λ _, 1) (λ n' a_n, 2 * a_n - cos (n' * π / 2) + 2 * sin (n' * π / 2))

-- Part (1)
theorem part1_a2 : a 2 = 4 := by sorry
theorem part1_a3 : a 3 = 9 := by sorry

-- Part (2)
theorem part2_general_formula (n : ℕ) : a n = 2 ^ n - sin (n * π / 2) := by sorry

-- Part (3)
def b (n : ℕ) : ℝ := n * (a n - 2 ^ n)

noncomputable def T (n : ℕ) : ℝ := ∑ i in range n, b (i + 1)

theorem part3_find_m (m : ℕ) (h : T m = 2024) : m = 4048 := by sorry

end part1_a2_part1_a3_part2_general_formula_part3_find_m_l346_346225


namespace waiting_probability_l346_346960

theorem waiting_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t ≤ 50 →
    (∃ t' : ℝ, 0 ≤ t' ∧ t' ≤ 30 ∧ t = t')) →
  (30 / 50 = 3 / 5) :=
by
  intro h
  norm_num
  sorry

end waiting_probability_l346_346960


namespace cyclic_shift_unique_positive_partial_sums_l346_346912

theorem cyclic_shift_unique_positive_partial_sums
  (n : ℕ) (a : ℕ → ℤ) (Hsum : (Finset.range n).sum a = 1) :
  ∃! b: ℕ → ℤ, ∃ (k : ℕ), (∀ m < n, (Finset.range (m + 1)).sum (λi, (λ j, a ((j + k) % n)) i) > 0) :=
sorry

end cyclic_shift_unique_positive_partial_sums_l346_346912


namespace smallest_four_digit_in_pascals_triangle_l346_346431

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346431


namespace cone_volume_constants_and_variables_l346_346318

theorem cone_volume_constants_and_variables :
  ∀ (V r h : ℝ), 
  (∃ (const1 const2 : ℝ), const1 = 1/3 ∧ const2 = real.pi) ∧ 
  (∃ (var1 var2 var3 : ℝ), 
    (var1 = V  ∧ var2 = r ∧ var3 = h)) := by
  sorry

end cone_volume_constants_and_variables_l346_346318


namespace incorrect_inequality_l346_346757

theorem incorrect_inequality (a b : ℝ) (h : a > b ∧ b > 0) :
  ¬ (1 / a > 1 / b) :=
by
  sorry

end incorrect_inequality_l346_346757


namespace icosahedron_vertices_l346_346723

noncomputable def number_of_vertices_of_icosahedron : ℕ :=
  12

theorem icosahedron_vertices (V F : ℕ) (h1 : ∀ (f: F), is_equilateral_triangle f) (h2 : ∀ v, v ∈ V → degree v = 3) (h3 : 2 * F - V = 4) :
  V = number_of_vertices_of_icosahedron :=
sorry

end icosahedron_vertices_l346_346723


namespace smallest_four_digit_in_pascals_triangle_l346_346477

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346477


namespace complex_division_proof_l346_346389

theorem complex_division_proof :
  (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I :=
by
  -- Conditions
  have h : Complex.I * Complex.I = -1 := by simp [Complex.I_sq, Complex.I]
  -- the proof goes as expected
  sorry

end complex_division_proof_l346_346389


namespace max_metro_lines_l346_346935

theorem max_metro_lines (h1 : ∀ line : ℕ, line.stations ≥ 4) 
                        (h2 : ∀ line : ℕ, line.transfer_stations ≤ 3) 
                        (h3 : ∀ station : ℕ, station.lines_crossing ≤ 2) 
                        (h4 : ∀ (start_line end_line : ℕ), reachable_within_2_transfers start_line end_line) :
                        ∃ max_lines : ℕ, max_lines = 10 :=
by
  sorry

end max_metro_lines_l346_346935


namespace coeff_of_x_in_expansion_l346_346789

theorem coeff_of_x_in_expansion :
  let f := (1 : ℚ) + (Polynomial.X : Polynomial ℚ)
  let g := (Polynomial.X : Polynomial ℚ) - Polynomial.C (2 / Polynomial.X)
  Polynomial.coeff ((f * g ^ 3).expand ℚ) 1 = -6 := sorry

end coeff_of_x_in_expansion_l346_346789


namespace arith_geom_seq_term_formula_sum_of_seq_b_l346_346709

def arith_seq (a : ℕ → ℤ) (d : ℤ) :=
∀ n : ℕ, a (n + 1) = a n + d

def geom_seq (b : ℕ → ℤ) :=
∀ n : ℕ, b (n + 1) ^ 2 = b n * b (n + 2)

theorem arith_geom_seq_term_formula :
  ∃ d : ℤ, d ≠ 0 ∧
  (arith_seq a d) ∧ (geom_seq (λ i, a i)) ∧
  a 1 = 1 ∧ ((a 1, a 2, a 6) ∈ (set.form_a_geom_seq)) →
  (∀ n : ℕ, a n = 3 * n - 2) :=
begin
  sorry
end

def seq_b (a : ℕ → ℤ) (b : ℕ → ℚ) := 
∀ n : ℕ, b n = 1 / (a n * a (n + 1))

theorem sum_of_seq_b :
  ∃ d : ℤ, d ≠ 0 ∧
  (arith_seq a d) ∧ (geom_seq (λ i, a i)) ∧
  a 1 = 1 ∧ ((a 1, a 2, a 6) ∈ (set.form_a_geom_seq)) →
  (∀ n : ℕ, let b := (λ n, 1 / (a n * a (n + 1))) in sum_of_first_n_terms b n = n / (3n + 1)) :=
begin
  sorry
end

end arith_geom_seq_term_formula_sum_of_seq_b_l346_346709


namespace find_k_l346_346153

def distances (S x y k : ℝ) := (S - x * 0.75) * x / (x + y) + 0.75 * x = S * x / (x + y) - 18 ∧
                              S * x / (x + y) - (S - y / 3) * x / (x + y) = k

theorem find_k (S x y k : ℝ) (h₁ : x * y / (x + y) = 24) (h₂ : k = 24 / 3)
  : k = 8 :=
by 
  -- We need to fill in the proof steps here
  sorry

end find_k_l346_346153


namespace angle_C_is_80_l346_346053

-- Define the angles A, B, and C
def isoscelesTriangle (A B C : ℕ) : Prop :=
  -- Triangle ABC is isosceles with A = B, and C is 30 degrees more than A
  A = B ∧ C = A + 30 ∧ A + B + C = 180

-- Problem: Prove that angle C is 80 degrees given the conditions
theorem angle_C_is_80 (A B C : ℕ) (h : isoscelesTriangle A B C) : C = 80 :=
by sorry

end angle_C_is_80_l346_346053


namespace prism_faces_count_l346_346133

theorem prism_faces_count (h : hexagon_base_prism → with_edges (21)) : faces (h) = 9 :=
sorry

end prism_faces_count_l346_346133


namespace intersection_height_l346_346768

theorem intersection_height (h1 h2 d : ℝ) (h1_eq : h1 = 30) (h2_eq : h2 = 50) (d_eq : d = 150) :
  let m1 := -h1 / d,
      b1 := h1,
      m2 := h2 / d,
      x_intersect := (b1 / (m2 - m1)) in
  1/3 * x_intersect = 18.75 :=
by
  have m1_eq : m1 = -30 / 150 := by rw [h1_eq, d_eq]; norm_num,
  have b1_eq : b1 = 30 := by rw h1_eq,
  have m2_eq : m2 = 50 / 150 := by rw [h2_eq, d_eq]; norm_num,
  have x_intersect_eq : x_intersect = 30 / ((50 / 150) + (30 / 150)) := by
    rw [m1_eq, m2_eq, b1_eq, div_div, div_add_div_same],
    norm_num,
  rw [x_intersect_eq, h1_eq, h2_eq, d_eq],
  norm_num,
  sorry

end intersection_height_l346_346768


namespace plane_KPH_intersects_cube_edges_and_ratios_l346_346180

def edge_length := 60
def AH_HB_ratio := 1 / 2
def DK_KD1_ratio := 1 / 3

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2, z := (A.z + B.z) / 2 }

noncomputable def divide_segment (A B : Point) (r : ℝ) : Point :=
  { x := A.x * (1 - r) + B.x * r, y := A.y * (1 - r) + B.y * r, z := A.z * (1 - r) + B.z * r }

def cube_vertices := {A := Point.mk 0 0 0, B := Point.mk edge_length 0 0, C := Point.mk edge_length edge_length 0,
                      D := Point.mk 0 edge_length 0, A1 := Point.mk 0 0 edge_length, B1 := Point.mk edge_length 0 edge_length,
                      C1 := Point.mk edge_length edge_length edge_length, D1 := Point.mk 0 edge_length edge_length}

def H := divide_segment cube_vertices.A cube_vertices.B (1 / 3)
def P := midpoint cube_vertices.B1 cube_vertices.C1
def K := divide_segment cube_vertices.D cube_vertices.D1 (1 / 4)

theorem plane_KPH_intersects_cube_edges_and_ratios :
  {e | e ∈ finset_univ ({
    intersecting_edges := [{(cube_vertices.A, cube_vertices.B)}, {(cube_vertices.BB1)}, {(cube_vertices.AD)}, {(cube_vertices.DD1)},
                          {(cube_vertices.B1C1)}, {(cube_vertices.C1D1)}],
    segment_ratios := { (cube_vertices.A, cube_vertices.E, cube_vertices.B), (cube_vertices.D, cube_vertices.KD1, cube_vertices.D1), etc. }}):
  sorry

end plane_KPH_intersects_cube_edges_and_ratios_l346_346180


namespace handrail_length_approximation_l346_346140

noncomputable def length_of_handrail (rise height radius : ℝ) (turns_degrees : ℝ) : ℝ :=
  let height := 10
  let radius := 4
  let turns_degrees := 180
  let hypotenuse := Real.sqrt (height^2 + (radius * Real.pi)^2)
  Real.ceil (hypotenuse * 10) / 10

theorem handrail_length_approximation :
  length_of_handrail 10 4 180 = 15.7 :=
begin
  -- Proof would go here
  sorry
end

end handrail_length_approximation_l346_346140


namespace sum_of_possible_values_of_s_r_eq_zero_l346_346817

def r : Set ℝ := { -2, -1, 0, 1 }
def r_range : Set ℝ := { -1, 0, 3, 5 }

def s (x : ℝ) : ℝ := 2 * x + 1
def s_domain : Set ℝ := { -1, 0, 1, 2 }

theorem sum_of_possible_values_of_s_r_eq_zero :
  ∑ v in ({x | x ∈ r_range ∧ x ∈ s_domain}), s v = 0 :=
by
  sorry

end sum_of_possible_values_of_s_r_eq_zero_l346_346817


namespace smallest_four_digit_number_in_pascals_triangle_l346_346449

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346449


namespace smallest_four_digit_in_pascals_triangle_l346_346430

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346430


namespace find_measurement_of_angle_Q_l346_346830
open_locale classical

noncomputable def m : Type := sorry
noncomputable def n : Type := sorry

def parallel (a b : Type) : Prop := sorry

noncomputable def angle_P : ℝ := 100
noncomputable def angle_R : ℝ := 70

theorem find_measurement_of_angle_Q (h1 : parallel m n)
  (h2 : angle_P = 100)
  (h3 : angle_R = 70) :
  ∃ angle_Q : ℝ, angle_Q = 80 :=
begin
  sorry,
end

end find_measurement_of_angle_Q_l346_346830


namespace total_boxes_sold_over_two_days_l346_346015

def boxes_sold_on_Saturday : ℕ := 60

def percentage_more_on_Sunday : ℝ := 0.50

def boxes_sold_on_Sunday (boxes_sold_on_Saturday : ℕ) (percentage_more_on_Sunday : ℝ) : ℕ :=
  (boxes_sold_on_Saturday : ℝ) * (1 + percentage_more_on_Sunday) |> Nat.floor

def total_boxes_sold (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  saturday_sales + sunday_sales

theorem total_boxes_sold_over_two_days :
  total_boxes_sold boxes_sold_on_Saturday (boxes_sold_on_Sunday boxes_sold_on_Saturday percentage_more_on_Sunday) = 150 := by
  sorry

end total_boxes_sold_over_two_days_l346_346015


namespace find_pure_gala_trees_l346_346929

variables (T F G : ℕ)
variables (H1 : F + 0.10 * T = 170) (H2 : F = (3/4) * T)

theorem find_pure_gala_trees (H1 : F + 0.10 * T = 170) (H2 : F = (3/4) * T) : G = T - F := 
by {
  have H3 : 0.75 * T + 0.10 * T = 170 := by rw [H2],
  have H4 : 0.85 * T = 170 := by linarith,
  have H5 : T = 200 := by linarith,
  have H6 : F = (3/4) * 200 := by rw [H2, H5],
  have H7 : F = 150 := by linarith,
  exact nat.sub_eq_of_eq_add H7
}

end find_pure_gala_trees_l346_346929


namespace battery_difference_l346_346052

noncomputable def flashlights_batteries := 3
noncomputable def toys_batteries := 15
noncomputable def remote_controllers_batteries := 7
noncomputable def wall_clock_batteries := 5
noncomputable def wireless_mouse_batteries := 4

def total_other_devices_batteries := flashlights_batteries + remote_controllers_batteries + wall_clock_batteries + wireless_mouse_batteries

theorem battery_difference : abs (toys_batteries - total_other_devices_batteries) = 4 :=
by sorry

end battery_difference_l346_346052


namespace geometric_sequence_product_correct_l346_346320

noncomputable def geometric_sequence_product (a_1 a_5 : ℝ) (a_2 a_3 a_4 : ℝ) :=
  a_1 = 1 / 2 ∧ a_5 = 8 ∧ a_2 * a_4 = a_1 * a_5 ∧ a_3^2 = a_1 * a_5

theorem geometric_sequence_product_correct:
  ∃ a_2 a_3 a_4 : ℝ, geometric_sequence_product (1 / 2) 8 a_2 a_3 a_4 ∧ (a_2 * a_3 * a_4 = 8) :=
by
  sorry

end geometric_sequence_product_correct_l346_346320


namespace calvin_weight_after_one_year_l346_346175

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l346_346175


namespace coupon_savings_l346_346139

noncomputable def price_difference : ℝ :=
  let x := 200 in  -- smallest price satisfying the condition
  let y := 450 in  -- largest price satisfying the condition
  y - x

theorem coupon_savings (P : ℝ) (hP : P > 150) :
  let savingA := 0.20 * P in
  let savingB := 40 in
  let savingC := 0.30 * (P - 150) in
  (savingA >= savingB) ∧ (savingA >= savingC) → 
  price_difference = 250 :=
sorry

end coupon_savings_l346_346139


namespace possible_n_l346_346674

theorem possible_n (n : ℕ) (h1 : n ≥ 3) :
  (∃ (a : Fin n → ℝ), (∀ i j : Fin n, i < j → a i ≠ a j) ∧
   (∃ (q : ℝ), q > 1 ∧
    ∀ i j k l : Fin n, i < j → j < k → k < l →
    a i * a j * q ^ (k - j) = a k * a l)) →
  n = 3 ∨ n = 4 :=
by
  sorry

end possible_n_l346_346674


namespace probability_neither_red_nor_purple_correct_l346_346115

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

def neither_red_nor_purple_balls : ℕ := total_balls - (red_balls + purple_balls)
def probability_neither_red_nor_purple : ℚ := (neither_red_nor_purple_balls : ℚ) / (total_balls : ℚ)

theorem probability_neither_red_nor_purple_correct : 
  probability_neither_red_nor_purple = 13 / 20 := 
by sorry

end probability_neither_red_nor_purple_correct_l346_346115


namespace unique_rs_exists_l346_346821

theorem unique_rs_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (gcd_ab : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), (0 < r ∧ r < b) ∧ (0 < s ∧ s < a) ∧ (a * r - b * s = 1) :=
  sorry

end unique_rs_exists_l346_346821


namespace smallest_four_digit_in_pascal_triangle_l346_346557

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346557


namespace number_of_triangles_in_decagon_l346_346975

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346975


namespace circle_tangent_circumcircle_l346_346784

-- Definitions
variables {A B C Q P T E F : Point}
variables (Γ : Circle)
variable (triangle_ABC : AcuteTriangle A B C)
variable (diameter_BC : Diameter Γ B C)
variable (intersection_AB_AC : Intersects Γ A B Q ∧ Intersects Γ A C P)
variable (lower_semicircle_t : OnLowerSemicircle Γ T)
variable (tangent_EF : TangentThroughPoint Γ T E ∧ TangentThroughPoint Γ T F)

-- Theorem statement
theorem circle_tangent_circumcircle :
  Tangent (CircleDiameter E F) (Circumcircle (Triangle A P Q)) :=
sorry

end circle_tangent_circumcircle_l346_346784


namespace correct_statements_l346_346570

def coin_toss (n : ℕ) (k : ℕ) : Prop :=
  ¬(k = n / 2)

def variance (A B : ℝ) : Prop :=
  A < B

def median (data : List ℕ) : ℕ :=
  let sorted_data := data.qsort (· < ·)
  sorted_data[sorted_data.length / 2]

def mode (data : List ℕ) : ℕ :=
  data.foldr (λ x count_map =>
    count_map.insert x (count_map.findD x 0 + 1)) HashMap.empty
  |> (λ count_map => count_map.fold (λ acc key count =>
      if count > acc.2 then (key, count) else acc) (0, 0)).1

def comprehensiveness_survey (large_population : Bool) : Prop :=
  large_population = false

theorem correct_statements 
  (coin_toss_1000 : coin_toss 1000 500)
  (var_A : ℝ) (var_B : ℝ) (var_ineq : variance var_A var_B)
  (data_set : List ℕ)
  (med : median data_set = 3)
  (mod : mode data_set = 5)
  (survey_required : comprehensiveness_survey true)
  : Prop :=
  var_ineq ∧ med = 3 ∧ mod = 5

end correct_statements_l346_346570


namespace exists_n_colored_triangle_free_graph_l346_346603

noncomputable def ramsey_number (r s : ℕ) : ℕ := sorry

theorem exists_n_colored_triangle_free_graph (n : ℕ) : ∃ G : SimpleGraph ℕ, G.chromaticNumber ≥ n ∧ ¬G.contains (completeGraph 3) :=
by 
  assume n : ℕ,
  let t := (some appropriate value based on Ramsey number bounds), -- assuming we have some function to determine an appropriate t
  have R3t := ramsey_number 3 t,
  have h1 : R3t > n * t := by sorry, -- the condition where Ramsey number exceeds nt
  let G := some_construction_of_G, -- some construction method defined elsewhere
  have h2 : ¬ G.contains (completeGraph 3) := by sorry, -- proof that G does not have a triangle
  have h3 : G.chromaticNumber ≥ n := by sorry, -- proof that G requires at least n colors
  exact ⟨G, h3, h2⟩

end exists_n_colored_triangle_free_graph_l346_346603


namespace triangles_from_decagon_l346_346992

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346992


namespace paint_comparison_l346_346090

theorem paint_comparison (D : ℝ) (π : ℝ) : 
  let A_small := π * (D / 2)^2
  let A_large := π * (7 * D / 2)^2
  A_large / A_small = 49 :=
by 
  have A_small := π * (D / 2)^2
  have A_large := π * (7 * D / 2)^2
  calc 
    A_large / A_small = (π * (49 * D^2 / 4)) / (π * (D^2 / 4)) : by sorry
                      ... = 49 : by sorry

end paint_comparison_l346_346090


namespace rate_of_stream_l346_346590

theorem rate_of_stream (v : ℝ) : 
  (∀ (s_b W : ℝ), s_b = 16 ∧ W = 168 ∧ (∀ t, t = 8 → W = (s_b + v) * t) → v = 5) :=
begin
  sorry
end

end rate_of_stream_l346_346590


namespace triangle_area_correct_l346_346677

/-- Define the vertices as vectors -/
def u := (1: ℝ, 2: ℝ, 3: ℝ)
def v := (4: ℝ, 6: ℝ, 8: ℝ)
def w := (2: ℝ, 1: ℝ, 7: ℝ)

/-- Define vector subtraction -/
def vector_sub (a b : ℝ × ℝ × ℝ) := (a.1 - b.1, a.2 - b.2, a.3 - b.3)
def uv := vector_sub v u
def uw := vector_sub w u

/-- Define the cross product -/ 
def cross_product (a b : ℝ × ℝ × ℝ) :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

def cross_uw_uv := cross_product uv uw

/-- Compute the magnitude of the cross product -/ 
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

/-- Define the area of the triangle -/ 
def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ :=
  0.5 * magnitude (cross_product (vector_sub b a) (vector_sub c a))

/-- Theorem: The area of the triangle with the given vertices is 1/2 * sqrt 539 -/
theorem triangle_area_correct : triangle_area u v w = 0.5 * real.sqrt 539 := 
by {
  sorry
}

end triangle_area_correct_l346_346677


namespace smallest_four_digit_in_pascals_triangle_l346_346437

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346437


namespace number_of_triangles_in_decagon_l346_346986

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346986


namespace good_numbers_l346_346617

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → (d + 1) ∣ (n + 1)

theorem good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n) :=
by
  sorry

end good_numbers_l346_346617


namespace triangle_perimeter_is_49_l346_346887

theorem triangle_perimeter_is_49 (a b c : ℕ) (a_gt_b_gt_c : a > b ∧ b > c) 
  (T1 : ℕ := 4 * b * c)
  (T2 : ℕ := 4 * b * c * (6 * a^2 + 2 * b^2 + 2 * c^2))
  (eqn : (T2 / (2 * T1)) = 2023) : a + b + c = 49 :=
by
  have h1 : T1 = 4 * b * c := rfl
  have h2 : T2 = 4 * b * c * (6 * a^2 + 2 * b^2 + 2 * c^2) := rfl
  have h_eq : 3 * a^2 + b^2 + c^2 = 2023 := sorry
  have sol : (a = 23 ∧ b = 20 ∧ c = 6) := sorry
  exact sol.1 + sol.2.1 + sol.2.2

end triangle_perimeter_is_49_l346_346887


namespace interval_monotonicity_a_le_0_interval_monotonicity_a_g_0_decreasing_before_ln_a_interval_monotonicity_a_g_0_decreasing_range_of_a_l346_346252

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

theorem interval_monotonicity_a_le_0 {a : ℝ} (h : a <= 0) :
  ∀ x : ℝ, 0 < Real.exp x - a :=
by
  sorry

theorem interval_monotonicity_a_g_0 {a : ℝ} (h : a > 0) :
  ∀ x : ℝ, x ∈ Ioi (Real.log a) → 0 < Real.exp x - a :=
by
  sorry

theorem decreasing_before_ln_a {a : ℝ} (h : a > 0) :
  ∀ x : ℝ, x ∈ Iio (Real.log a) → Real.exp x - a < 0 :=
by
  sorry

theorem interval_monotonicity_a_g_0_decreasing {a : ℝ} (h : a > 0) :
  ∀ x : ℝ, x ∈ Iio (Real.log a) → Real.exp x - a < 0 :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x: ℝ, 2 ≤ x -> f x a > x + x^2) ↔ a < (Real.exp 2 / 2 - 3) :=
by
  sorry

end interval_monotonicity_a_le_0_interval_monotonicity_a_g_0_decreasing_before_ln_a_interval_monotonicity_a_g_0_decreasing_range_of_a_l346_346252


namespace general_formula_of_a_general_formula_of_T_l346_346710

variables (a_n S_n : ℕ → ℝ)
variables (T_n : ℕ → ℝ)

def arithmetic_sequence_condition (a : ℕ → ℝ) (n : ℕ) : Prop := 
  ∃ a_1 d : ℝ, a 2 = 9 ∧ (5 * a_1 + (5 * 4 * d) / 2) = 65

def sum_first_n_terms (S : ℕ → ℝ) (n : ℕ) : Prop := 
  S n = (n * (2 * n + 3)) / 2

def sum_modified_terms (T : ℕ → ℝ) (n : ℕ) : Prop := 
  T n = (1 / 2) * (1 - (1 / (n + 1)))

open scoped Classical

theorem general_formula_of_a (a : ℕ → ℝ) 
  (ha : arithmetic_sequence_condition a 2) : 
  ∀ (n : ℕ), a n = 4 * n + 1 := 
sorry

theorem general_formula_of_T (S : ℕ → ℝ) (T : ℕ → ℝ)
  (hS : sum_first_n_terms S) 
  (hT : ∀ (n : ℕ), ∀ m, T n = ∑ i in finset.range n, (1 / (S (i + 1) - (i + 1)))) :
  ∀ (n : ℕ), T n = (n / (2 * (n + 1))) := 
sorry

end general_formula_of_a_general_formula_of_T_l346_346710


namespace fencing_cost_l346_346135

theorem fencing_cost (area : ℝ) (short_side : ℝ) (cost_per_meter : ℝ) 
  (h_area : area = 1200) (h_short_side : short_side = 30) (h_cost_per_meter : cost_per_meter = 14) : 
  let long_side := area / short_side,
      diagonal := Real.sqrt (short_side ^ 2 + long_side ^ 2),
      total_length := long_side + short_side + diagonal,
      total_cost := total_length * cost_per_meter
  in total_cost = 1680 := 
by
  sorry

end fencing_cost_l346_346135


namespace richter_scale_frequency_ratio_l346_346350

theorem richter_scale_frequency_ratio:
  (∀ x : ℝ, energy (x - 1) = (1 / 10) * energy x) →
  (energy 5 / energy 3 = 100) :=
by
  intro h
  sorry

end richter_scale_frequency_ratio_l346_346350


namespace part_1_part_2_part_3_l346_346262

-- Definitions of the given functions
def f (x : ℝ) (m : ℝ) : ℝ := m * x + 3
def g (x : ℝ) (m : ℝ) : ℝ := x^2 + 2 * x + m
def G (x : ℝ) (m : ℝ) : ℝ := (f x m) - (g x m) - 1

-- (1) Prove that for all real values m, the function f(x) - g(x) has a zero point
theorem part_1 (m : ℝ) : ∃ x : ℝ, (f x m - g x m = 0) :=
sorry 

-- (2) Prove that if |G(x)| is a decreasing function on [-1, 0], then m ∈ (-∞, 0] ∪ [2, +∞)
theorem part_2 (h : ∀ x, -1 ≤ x ∧ x ≤ 0 → |G x m| ≤ |G (x + 1) m|) : 
  m ∈ set.Icc (2 : ℝ) (6 : ℝ) ∨ m ∈ set.Iic (0 : ℝ) :=
sorry

-- (3) Prove that there exist integers a and b such that the solution set of a ≤ G(x) ≤ b
-- is exactly [a, b], and these values of a and b are (a = -1, b = 1) or (a = 2, b = 4)
theorem part_3 : ∃ a b : ℤ, 
  (a ≤ b ∧ (∀ x : ℤ, a ≤ G x m ∧ G x m ≤ b ↔ a ≤ x ∧ x ≤ b)) ∧ 
  ((a = -1 ∧ b = 1) ∨ (a = 2 ∧ b = 4)) :=
sorry

end part_1_part_2_part_3_l346_346262


namespace smallest_four_digit_in_pascals_triangle_l346_346420

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346420


namespace fifteenth_prime_is_59_l346_346206

/-- Define the prime numbers -/
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

/-- The fifteenth prime number given that 5 is the third prime number is 59 -/
theorem fifteenth_prime_is_59 : primes.nth 14 = some 59 := 
by
  sorry

end fifteenth_prime_is_59_l346_346206


namespace area_of_triangle_l346_346319

theorem area_of_triangle
  (A B C : EuclideanSpace ℝ (Fin 2))
  (hAB : ∥B - A∥ = 1)
  (hBC : ∥C - B∥ = 3)
  (hDot : (B - A) • (C - B) = -1) :
  ∃ (area : ℝ), area = Real.sqrt 2 := 
sorry

end area_of_triangle_l346_346319


namespace problem_statement_l346_346733

noncomputable def func (x : ℝ) : ℝ := abs (Real.log x)

theorem problem_statement (a b : ℝ) (h1: 0 < a) (h2: 0 < b) (h3 : a < b) (h4 : func a = func b) 
  (h5 : ∀ x ∈ set.Icc (a^2) b, func x ≤ 2) (hmax : ∃ x ∈ set.Icc (a^2) b, func x = 2) :
  2 * a + b = 2 / Real.exp 1 + Real.exp 1 :=
by
  sorry

end problem_statement_l346_346733


namespace product_mod_equiv_one_l346_346808

open Nat

theorem product_mod_equiv_one :
  let A := {n : ℕ | 1 ≤ n ∧ n ≤ 2009^2009}
  let S := {n ∈ A | gcd n (2009^2009) = 1}
  let P := ∏ x in S, x
  in P ≡ 1 [MOD 2009^2009] := 
by
  sorry

end product_mod_equiv_one_l346_346808


namespace cube_volume_in_pyramid_l346_346134

-- Define the dimensions and properties given in the problem
def side_length_base : ℝ := 2
def base_angle : ℝ := 60 -- degrees
def volume_of_cube : ℝ := 3 * real.sqrt 6 / 32

-- Define the properties of the pyramid
def pyramid_volume_cube (s : ℝ) : Prop :=
  (∃ (length : ℝ), length = s ∧ s = real.sqrt 6 / 4) ∧
  s * s * s = volume_of_cube

-- The Lean 4 statement to prove the volume of the cube
theorem cube_volume_in_pyramid : ∃ s, pyramid_volume_cube s := by
  sorry

end cube_volume_in_pyramid_l346_346134


namespace trader_gain_percentage_l346_346576

variable (x : ℝ) (cost_of_one_pen : ℝ := x) (selling_cost_90_pens : ℝ := 90 * x) (gain : ℝ := 30 * x)

theorem trader_gain_percentage :
  30 * cost_of_one_pen / (90 * cost_of_one_pen) * 100 = 33.33 := by
  sorry

end trader_gain_percentage_l346_346576


namespace sequence_ineq_l346_346741

theorem sequence_ineq (a : ℕ → ℝ) (h1 : a 1 = 15) 
  (h2 : ∀ n, a (n + 1) = a n - 2 / 3) 
  (hk : a k * a (k + 1) < 0) : k = 23 :=
sorry

end sequence_ineq_l346_346741


namespace f_three_l346_346726

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_succ : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom f_one : f 1 = 1 

-- Goal
theorem f_three : f 3 = -1 :=
by
  -- The proof will be provided here
  sorry

end f_three_l346_346726


namespace y_percentage_less_than_A_l346_346854

variable {A B y : ℝ}

theorem y_percentage_less_than_A (h : A > B > 0) (h₁ : B = A - (y / 100) * A) : y = 100 * (A - B) / A :=
sorry

end y_percentage_less_than_A_l346_346854


namespace part1_case1_part1_case2_part2_l346_346316

variable (m : ℝ)

-- Part 1: Proving the coordinates of M given the distance to y-axis
theorem part1_case1 (h1 : abs (2 - m) = 3) (hm : m = -1) : M = (3, -1) :=
sorry

theorem part1_case2 (h1 : abs (2 - m) = 3) (hm : m = 5) : M = (-3, 11) :=
sorry

-- Part 2: Proving the coordinates of M when M lies on the angle bisector
theorem part2 (h2 : 2 - m = 1 + 2m) : m = (1 / 3) → M = ((5 / 3), (5 / 3)) :=
sorry

end part1_case1_part1_case2_part2_l346_346316


namespace range_of_a_l346_346291

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2 * x - 1| + |x + 1| > a) ↔ a < 3 / 2 := by
  sorry

end range_of_a_l346_346291


namespace minimum_cosine_of_triangle_l346_346245

noncomputable def min_cosine_value : ℝ :=
  min ( (3^2 + 4^2 - 2^2) / (2 * 3 * 4) ) ( (4^2 + 6^2 - 3^2) / (2 * 4 * 6) )

theorem minimum_cosine_of_triangle (a b c d : ℝ) (h : {2, 3, 4, 6} ⊆ {a, b, c, d}) :
  min_cosine_value = 43/48 :=
  sorry

end minimum_cosine_of_triangle_l346_346245


namespace door_challenge_sequences_l346_346364

theorem door_challenge_sequences : ∃ n : ℕ, n = 64 ∧ 
  ∃ (order_of_challenges : list ℕ), order_of_challenges.length = 7 ∧
  ∀ i, i ∈ (list.range 7) →
  ∃ (choices : list (ℕ × ℕ)), 
    (choices.length = 6) ∧
    (∀ j, j < 6 → (choices.nth_le j sorry).snd ∈ [(choices.nth_le j sorry).fst - 1, (choices.nth_le j sorry).fst + 1]) :=
  by 
  let choices_length := 6
  have h1 : (1:ℕ) ≤ choices_length, from nat.succ_le_succ nat.zero_le
  have h2 : (7:ℕ) = 6 + 1, by norm_num
  have h3 : (2^6 = 64), by norm_num
  exact ⟨64, h3, sorry⟩

end door_challenge_sequences_l346_346364


namespace largest_term_binomial_expansion_l346_346105

theorem largest_term_binomial_expansion :
  let a := 1
  let b := Real.sqrt 3
  let n := 100
  ∃ k : ℕ, k = 64 ∧ (λ k, (choose n k) * b ^ k) k = max (λ k, (choose n k) * b ^ k) (finset.range (n + 1)) :=
begin
  sorry
end

end largest_term_binomial_expansion_l346_346105


namespace smallest_four_digit_in_pascal_l346_346495

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346495


namespace calvin_final_weight_l346_346170

def initial_weight : ℕ := 250
def weight_loss_per_month : ℕ := 8
def duration_months : ℕ := 12
def total_weight_loss : ℕ := weight_loss_per_month * duration_months
def final_weight : ℕ := initial_weight - total_weight_loss

theorem calvin_final_weight : final_weight = 154 :=
by {
  have h1 : total_weight_loss = 96 := by norm_num,
  rw [h1],
  norm_num,
  sorry
}

-- We have used 'sorry' to mark the place where the proof would be completed.

end calvin_final_weight_l346_346170


namespace only_integer_triplet_solution_l346_346673

theorem only_integer_triplet_solution 
  (a b c : ℤ) : 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by 
  intro h
  sorry

end only_integer_triplet_solution_l346_346673


namespace root_of_quadratic_with_geometric_sequence_l346_346725

theorem root_of_quadratic_with_geometric_sequence (p q r k : ℝ) (hq : q = k * p) (hr : r = k^2 * p) 
  (hseq : p ≤ q ∧ q ≤ r ∧ r ≤ 0) (hdisc : q^2 - 4 * p * r = 0) : 
  (∃ x : ℝ, (p ≠ 0) ∧ x = -1 ∧ p * x^2 + q * x + r = 0) :=
begin
  sorry
end

end root_of_quadratic_with_geometric_sequence_l346_346725


namespace die_top_face_after_path_l346_346918

def opposite_face (n : ℕ) : ℕ :=
  7 - n

def roll_die (start : ℕ) (sequence : List String) : ℕ :=
  sequence.foldl
    (λ top movement =>
      match movement with
      | "left" => opposite_face (7 - top) -- simplified assumption for movements
      | "forward" => opposite_face (top - 1)
      | "right" => opposite_face (7 - top + 1)
      | "back" => opposite_face (top + 1)
      | _ => top) start

theorem die_top_face_after_path : roll_die 3 ["left", "forward", "right", "back", "forward", "back"] = 4 :=
  by
  sorry

end die_top_face_after_path_l346_346918


namespace smallest_four_digit_in_pascals_triangle_l346_346532

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346532


namespace xy_identity_l346_346002

theorem xy_identity (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) : x^2 + y^2 = 5 :=
  sorry

end xy_identity_l346_346002


namespace smallest_four_digit_in_pascals_triangle_l346_346478

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346478


namespace negation_of_proposition_l346_346030

theorem negation_of_proposition (P : ∃ x : ℝ, x^2 - 2 * x + 1 < 0) :
  (¬ P ↔ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by 
  sorry

end negation_of_proposition_l346_346030


namespace milk_left_is_correct_l346_346963

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end milk_left_is_correct_l346_346963


namespace smallest_four_digit_in_pascals_triangle_l346_346482

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346482


namespace percentage_material_B_new_mixture_l346_346349

theorem percentage_material_B_new_mixture :
  let mixtureA := 8 -- kg of Mixture A
  let addOil := 2 -- kg of additional oil
  let addMixA := 6 -- kg of additional Mixture A
  let oil_percent := 0.20 -- 20% oil in Mixture A
  let materialB_percent := 0.80 -- 80% material B in Mixture A

  -- Initial amounts in 8 kg of Mixture A
  let initial_oil := oil_percent * mixtureA
  let initial_materialB := materialB_percent * mixtureA

  -- New mixture after adding 2 kg oil
  let new_oil := initial_oil + addOil
  let new_materialB := initial_materialB

  -- Adding 6 kg of Mixture A
  let added_oil := oil_percent * addMixA
  let added_materialB := materialB_percent * addMixA

  -- Total amounts in the new mixture
  let total_oil := new_oil + added_oil
  let total_materialB := new_materialB + added_materialB
  let total_weight := mixtureA + addOil + addMixA

  -- Percent calculation
  let percent_materialB := (total_materialB / total_weight) * 100

  percent_materialB = 70 := sorry

end percentage_material_B_new_mixture_l346_346349


namespace total_profit_proof_l346_346949

-- Define the amounts invested and the relationships among them.
variables {C B A P : ℝ}

def B_investment (C : ℝ) : ℝ := (2/3) * C
def A_investment (B : ℝ) : ℝ := 3 * B
def total_investment (A B C : ℝ) : ℝ := A + B + C

-- Define the profit sharing ratio and total profit computation.
def profit_ratio (A B C : ℝ) : (ℝ × ℝ × ℝ) := (A/total_investment A B C, B/total_investment A B C, C/total_investment A B C)
def total_profit (B_share : ℝ) : ℝ := (11 / 2) * B_share

-- Given conditions
axiom A_eq_3B (B : ℝ) : A_investment B = 3 * B
axiom B_eq_2div3C (C : ℝ) : B_investment C = (2/3) * C
axiom B_share_value : B_share = 1000

-- Main theorem to prove
theorem total_profit_proof (B : ℝ) (C : ℝ) (B_share : ℝ) (hA : A_investment B = A)
  (hB : B_investment C = B) (hB_share : B_share = 1000) : 
  total_profit B_share = 5500 :=
by
  sorry

end total_profit_proof_l346_346949


namespace course_selection_schemes_l346_346126

theorem course_selection_schemes {C₀ C₁ C₂ C₃ : Type} [Fintype C₀] [Fintype C₁] [Fintype C₂] [Fintype C₃]
    (students : Fin 4 → C₀ ⊕ C₁ ⊕ C₂ ⊕ C₃) 
    (no_students : ∀ c : C₀ ⊕ C₁ ⊕ C₂ ⊕ C₃, (∀ i : Fin 4, students i ≠ c) → (c = Sum.inl C₀ ∨ c = Sum.inl C₁)) :
  (∃ (course_schemes : Fin 4 → Fin 2), 
    ∀ i : Fin 2, ∃ j : Fin 4, students j = Sum.inr (Sum.inr (Sum.inr (Sum.inl i)))) →
  card (Finset.univ : Finset (Fin 84)) :=
begin
  sorry
end

end course_selection_schemes_l346_346126


namespace least_positive_integer_with_exactly_seven_factors_l346_346061

theorem least_positive_integer_with_exactly_seven_factors : 
  ∃ (n : ℕ), (∀ n > 0 → n ∈ ℕ) ∧ (∀ m : ℕ, m > 0 ∧ m < n → number_of_factors m ≠ 7) ∧ number_of_factors n = 7 ∧ n = 64 :=
by
  sorry

open Nat

/--
Defines the number of factors of a given number.
-/
def number_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0).card

attribute [simp] number_of_factors -- Simplifies and registers the number_of_factors as a known attribute

end least_positive_integer_with_exactly_seven_factors_l346_346061


namespace sum_of_coefficients_l346_346968

def polynomial_coeff_sum : ℕ := 4

theorem sum_of_coefficients (x : ℝ) :
  let P := 4 * (2*x^6 + 9*x^3 - 6) + 8 * (x^4 - 6*x^2 + 3)
  P.eval 1 = (4 : ℝ) := by
  sorry

end sum_of_coefficients_l346_346968


namespace degree_to_radians_l346_346655

theorem degree_to_radians (deg : ℝ) (h : deg = 240) : 
  deg * (real.pi / 180) = 4 * real.pi / 3 :=
by 
  -- Convert degrees to radians
  rw h,
  -- Simplify the multiplication
  norm_num

end degree_to_radians_l346_346655


namespace smallest_four_digit_in_pascal_l346_346493

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346493


namespace calvin_weight_after_one_year_l346_346173

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l346_346173


namespace ordered_pairs_count_l346_346201

theorem ordered_pairs_count :
  {n : ℕ // ∃ pairs : Finset (ℕ × ℕ),
     (∀ (x y : ℕ), (x, y) ∈ pairs → x > 0 ∧ y > 0 ∧ x ≤ 2 * y ∧ 2 * y ≤ 60 ∧ y ≤ 2 * x ∧ 2 * x ≤ 60) ∧
     n = pairs.card} = ⟨480, _⟩ :=
by
  sorry

end ordered_pairs_count_l346_346201


namespace smallest_four_digit_in_pascals_triangle_l346_346440

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346440


namespace angle_of_inclination_is_120_degrees_l346_346264

/-- Given the parametric equations of a line, 
    prove that the angle of inclination of the line is 120 degrees. -/
theorem angle_of_inclination_is_120_degrees
    (x y : ℝ)
    (t : ℝ)
    (h1 : x = 1 + real.sqrt 3 * t)
    (h2 : y = 3 - 3 * t) :
  ∃ θ : ℝ, θ = 120 ∧ x = 1 + real.sqrt 3 * t ∧ y = 3 - 3 * t := 
by
  -- The proof will go here
  sorry

end angle_of_inclination_is_120_degrees_l346_346264


namespace player_a_wins_l346_346103

/-- 
There are 1993 points and some pairs of points are connected by non-intersecting line segments, 
and no set of these line segments forms a closed polygon. Two players, A and B, take turns placing 
chess pieces on the marked points, with the condition that except for the first piece placed by 
player A, each subsequent piece must be placed on a point that is adjacent to the previous one. 
The player who cannot place a piece loses. Prove that player A has a winning strategy.
-/
theorem player_a_wins :
  ∃ strategy : nat → nat, -- strategy is a function defining a sequence of moves
    (∀ turn : nat, -- for every turn 
      strategy (turn + 1) = strategy turn + 1 ∨ strategy (turn + 1) = strategy turn - 1) ∧ -- next move must be adjacent
    (A_has_winning_strategy strategy) :=
sorry

end player_a_wins_l346_346103


namespace maximize_area_slope_l346_346612

-- Define the curve as a function
def curve_y (x : ℝ) : ℝ := sqrt (1 - x^2)

-- Line passing through the given point
def line_l (k x : ℝ) : ℝ := k * (x - sqrt 2)

-- Conditions: The curve is the upper part of the unit circle
def on_curve (A : ℝ × ℝ) : Prop := A.2 = curve_y A.1

-- A point on the line
def on_line (A : ℝ × ℝ) (k : ℝ) : Prop := A.2 = line_l k A.1

-- The area of the triangle given points A, B, and O
def triangle_area (A B : ℝ × ℝ) : ℝ :=
  let OA := sqrt (A.1^2 + A.2^2)
  let OB := sqrt (B.1^2 + B.2^2)
  let AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  0.5 * OA * OB * sin (angle_between OA OB AB)

-- Variables
variable (k : ℝ)

-- The origin point O
def O : ℝ × ℝ := (0, 0)

-- Prove that the slope of the line l that maximizes the area of triangle AOB is -sqrt(3)/3
theorem maximize_area_slope :
  ∃ (A B : ℝ × ℝ), on_curve A ∧ on_curve B ∧ on_line A k ∧ on_line B k ∧
  (∀ k' : ℝ, on_line A k' → on_line B k' → triangle_area A B O ≤ triangle_area A B O)
  → k = -√3 / 3 :=
sorry

end maximize_area_slope_l346_346612


namespace sin_alpha_plus_beta_l346_346236

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : cos (π / 4 - α) = 3 / 5)
  (h2 : sin (5 * π / 4 + β) = -12 / 13)
  (h3 : α ∈ Ioo (π / 4) (3 * π / 4))
  (h4 : β ∈ Ioo 0 (π / 4)) :
  sin (α + β) = 56 / 65 := 
by
  sorry

end sin_alpha_plus_beta_l346_346236


namespace correlation_coefficient_property_l346_346571

theorem correlation_coefficient_property (r : ℝ) (h : |r| ≤ 1) :
  (∀ ε > 0, (1 - ε < |r| → "high correlation") ∧ (|r| < ε → "low correlation")) :=
sorry

end correlation_coefficient_property_l346_346571


namespace smallest_four_digit_in_pascal_l346_346496

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346496


namespace adjacent_girl_pairs_l346_346044

variable (boyCount girlCount : ℕ) 
variable (adjacentBoyPairs adjacentGirlPairs: ℕ)

theorem adjacent_girl_pairs
  (h1 : boyCount = 10)
  (h2 : girlCount = 15)
  (h3 : adjacentBoyPairs = 5) :
  adjacentGirlPairs = 10 :=
sorry

end adjacent_girl_pairs_l346_346044


namespace can_assemble_natural_sum_l346_346092

-- Definitions based on the problem conditions
def alpha : ℝ := (-1 + Real.sqrt 29) / 2
def is_natural (n : ℕ) : Prop := n ≥ 0

-- Conditions (no denomination used more than 6 times, and each is irrational except 1)
def valid_coins (k : ℕ) : Prop := (k = 0 ∨ irrational (alpha ^ k)) ∧ alpha > 2

-- Main statement to prove
theorem can_assemble_natural_sum (n : ℕ) (h_nat : is_natural n) : 
  (∀ k : ℕ, valid_coins k) → ∃ (coefficients : ℕ → ℕ), (∀ k : ℕ, coefficients k ≤ 6) ∧ 
  (n = ∑ (k : ℕ) in (Finset.range (n + 1)), (coefficients k) * (alpha ^ k)) :=
by
  sorry

end can_assemble_natural_sum_l346_346092


namespace cos_sin_gt_sin_cos_l346_346569

theorem cos_sin_gt_sin_cos (x : ℝ) : cos (sin x) > sin (cos x) :=
by
  sorry -- The proof is omitted as per the instructions

end cos_sin_gt_sin_cos_l346_346569


namespace monotonic_increasing_interval_f_l346_346396

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 8)

theorem monotonic_increasing_interval_f :
  ∃ I : Set ℝ, (I = Set.Icc (-2) 1) ∧ (∀x1 ∈ I, ∀x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

end monotonic_increasing_interval_f_l346_346396


namespace portion_spent_on_utility_bills_l346_346342

def maria_salary : ℕ := 2000
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def remaining_after_bills : ℕ := 1125

theorem portion_spent_on_utility_bills :
  let tax_deduction := tax_rate * maria_salary
  let insurance_deduction := insurance_rate * maria_salary
  let total_deductions := tax_deduction + insurance_deduction
  let money_left_after_deductions := maria_salary - total_deductions
  let amount_spent_on_utility_bills := money_left_after_deductions - remaining_after_bills in
  (amount_spent_on_utility_bills / money_left_after_deductions : ℝ) = 0.25 :=
by
  sorry

end portion_spent_on_utility_bills_l346_346342


namespace city_population_l346_346774

theorem city_population (P : ℕ) (h1 : 0.20 * 0.20 * P = 20000) : P = 500000 :=
by 
  sorry

end city_population_l346_346774


namespace oddly_powerful_integers_less_than_3000_l346_346648

def oddly_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a % 2 = 1 ∧ b % 2 = 1 ∧ b > 1 ∧ a^b = n

def count_oddly_powerful (m : ℕ) : ℕ :=
  Finset.card (Finset.filter oddly_powerful (Finset.range m))

theorem oddly_powerful_integers_less_than_3000 :
  count_oddly_powerful 3000 = 9 :=
  by
    sorry

end oddly_powerful_integers_less_than_3000_l346_346648


namespace tangent_slope_angle_eq_pi_over_4_l346_346889

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)

theorem tangent_slope_angle_eq_pi_over_4 :
  let θ := Real.arctan ((f' 1) : ℝ)
  in θ = Real.pi / 4 :=
by
  sorry

end tangent_slope_angle_eq_pi_over_4_l346_346889


namespace trigonometric_identity_l346_346724

noncomputable def point_on_terminal_side (x y : ℝ) : Prop :=
    ∃ α : ℝ, x = Real.cos α ∧ y = Real.sin α

theorem trigonometric_identity (x y : ℝ) (h : point_on_terminal_side 1 3) :
    (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by
  sorry

end trigonometric_identity_l346_346724


namespace find_f_prime_at_one_l346_346073

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l346_346073


namespace smallest_four_digit_in_pascal_l346_346514

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346514


namespace find_removed_number_satisfies_avg_l346_346907

def avg := (l : List ℕ) → (l.sum.toReal / l.length.toReal)

theorem find_removed_number_satisfies_avg : 
    ∀ (x : ℕ), x ∈ [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] →
    avg ([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].erase x) = 8.2 ↔ x = 6 := by
    sorry

end find_removed_number_satisfies_avg_l346_346907


namespace smallest_four_digit_in_pascal_triangle_l346_346551

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346551


namespace last_three_digits_7_pow_123_l346_346199

theorem last_three_digits_7_pow_123 : (7^123 % 1000) = 717 := sorry

end last_three_digits_7_pow_123_l346_346199


namespace complement_in_U_l346_346269

open Set

def complement_U_A : Set ℤ := {x : ℤ | x ∈ (λ x : ℤ, x^2 < 9) \ {(-2), 2}}

theorem complement_in_U :
  complement_U_A = {-1, 0, 1} := by
  sorry

end complement_in_U_l346_346269


namespace line_passes_through_point_l346_346028

theorem line_passes_through_point (m : ℝ) : ∃ p : ℝ × ℝ, p = (1, 1) ∧ m * p.1 + p.2 - m - 1 = 0 :=
by
  exists (1, 1)
  split
  · rfl
  · sorry

end line_passes_through_point_l346_346028


namespace gcf_75_100_l346_346184

theorem gcf_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcf_75_100_l346_346184


namespace sphere_volume_in_cone_l346_346942

-- Definitions according to the problem conditions
def cone_diameter : ℝ := 12
def cone_radius : ℝ := cone_diameter / 2
def hypotenuse_length : ℝ := cone_radius * Real.sqrt 2
def altitude_length : ℝ := cone_radius * Real.sqrt 2 / Real.sqrt 2
def sphere_radius : ℝ := altitude_length / 2

-- The volume of a sphere
def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- The theorem to prove
theorem sphere_volume_in_cone : sphere_volume sphere_radius = 36 * Real.pi :=
by
  -- Proof goes here
  sorry

end sphere_volume_in_cone_l346_346942


namespace euclidean_division_l346_346822

theorem euclidean_division (a b : ℕ) (hb : b ≠ 0) : ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ a = b * q + r :=
by sorry

end euclidean_division_l346_346822


namespace not_divisible_l346_346845

-- Defining the necessary conditions
variable (m : ℕ)

theorem not_divisible (m : ℕ) : ¬ (1000^m - 1 ∣ 1978^m - 1) :=
sorry

end not_divisible_l346_346845


namespace smallest_four_digit_number_in_pascals_triangle_l346_346452

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346452


namespace sum_of_interior_edges_l346_346624

-- Definitions derived from given conditions
def width_of_frame : ℝ := 2
def area_of_frame : ℝ := 36
def outer_edge_length : ℝ := 7

-- Theorem: Sum of the lengths of the four interior edges
theorem sum_of_interior_edges :
  let y := (36 - 7 * 4) / 4 in
  let interior_length1 := 7 - 2 * width_of_frame in
  let interior_length2 := y in
  2 * (interior_length1 + interior_length2) = 10 :=
by
  sorry

end sum_of_interior_edges_l346_346624


namespace hexagon_side_equality_l346_346777

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
          [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
          [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

def parallel (x y : A) : Prop := ∀ r : ℝ, x = r • y
noncomputable def length_eq (x y : A) : Prop := ∃ r : ℝ, r • x = y

variables (AB DE BC EF CD FA : A)
variables (h1 : parallel AB DE)
variables (h2 : parallel BC EF)
variables (h3 : parallel CD FA)
variables (h4 : length_eq AB DE)

theorem hexagon_side_equality :
  length_eq BC EF ∧ length_eq CD FA :=
by
  sorry

end hexagon_side_equality_l346_346777


namespace combined_time_to_finish_cereal_l346_346838

theorem combined_time_to_finish_cereal : 
  let rate_fat := 1 / 15
  let rate_thin := 1 / 45
  let combined_rate := rate_fat + rate_thin
  let time_needed := 4 / combined_rate
  time_needed = 45 := 
by 
  sorry

end combined_time_to_finish_cereal_l346_346838


namespace hyperbola_imaginary_axis_l346_346879

theorem hyperbola_imaginary_axis (m : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ m = -a^2 ∧ b = 2a ∧ b^2 + a^2 = 1) → m = -1 / 4 :=
by
  sorry

end hyperbola_imaginary_axis_l346_346879


namespace number_of_triangles_in_decagon_l346_346978

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346978


namespace count_ways_to_select_l346_346769

def choose_a1_a2_a3_count (S : Finset ℕ) : ℕ :=
  (S.filter (λ (t : ℕ × ℕ × ℕ), t.1 < t.2 ∧ t.2 < t.3 ∧ t.2 - t.1 ≥ 3 ∧ t.3 - t.2 ≥ 3)).card

theorem count_ways_to_select (f : (Finset ℕ)) 
  (h1 : f = Finset.range 15 \ {0}) :
  choose_a1_a2_a3_count f = 120 :=
by {
  sorry
}

end count_ways_to_select_l346_346769


namespace smallest_four_digit_in_pascals_triangle_l346_346489

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346489


namespace evaluate_expression_l346_346194

theorem evaluate_expression : 
  - (16 / 2 * 8 - 72 + 4^2) = -8 :=
by 
  -- here, the proof would typically go
  sorry

end evaluate_expression_l346_346194


namespace sheila_earnings_per_hour_l346_346005

noncomputable def total_hours_worked (hours_per_day_MWF : ℕ) (days_MWF : ℕ) 
                                     (hours_per_day_TT : ℕ) (days_TT : ℕ) : ℕ :=
hours_per_day_MWF * days_MWF + hours_per_day_TT * days_TT

noncomputable def earnings_per_hour (weekly_earnings : ℕ) (total_hours : ℕ) : ℕ :=
weekly_earnings / total_hours

theorem sheila_earnings_per_hour : 
  let hours_MWF := 8,
      days_MWF := 3,
      hours_TT := 6,
      days_TT := 2,
      weekly_earnings := 288 in
  earnings_per_hour weekly_earnings (total_hours_worked hours_MWF days_MWF hours_TT days_TT) = 8 := 
by 
  sorry

end sheila_earnings_per_hour_l346_346005


namespace count_seating_arrangements_l346_346778

/-
  Definition of the seating problem at the round table:
  - The committee has six members from each of three species: Martians (M), Venusians (V), and Earthlings (E).
  - The table has 18 seats numbered from 1 to 18.
  - Seat 1 is occupied by a Martian, and seat 18 is occupied by an Earthling.
  - Martians cannot sit immediately to the left of Venusians.
  - Venusians cannot sit immediately to the left of Earthlings.
  - Earthlings cannot sit immediately to the left of Martians.
-/
def num_arrangements_valid_seating : ℕ := -- the number of valid seating arrangements
  sorry

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def N : ℕ := 347

theorem count_seating_arrangements :
  num_arrangements_valid_seating = N * (factorial 6)^3 :=
sorry

end count_seating_arrangements_l346_346778


namespace bathroom_new_area_l346_346038

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l346_346038


namespace midpoint_TS_l346_346703

-- Definitions of geometric objects and their properties
variables {P : Type} [metric_space P] [normed_group P] [normed_space ℝ P]
variables {A B C D O M T S : P}

-- Defining conditions of the problem
def convex_quadrilateral (A B C D : P) : Prop := sorry
def is_diagonal_intersect (A B C D O : P) : Prop := sorry
def is_circumcircle (O A D M : P) : Prop := sorry
def lines_intersect_at (P Q X : P) : Prop := sorry
def point_on_line (P Q X : P) : Prop := sorry

-- Main theorem
theorem midpoint_TS (A B C D O M T S : P)
  (h1 : convex_quadrilateral A B C D)
  (h2 : is_diagonal_intersect A B C D O)
  (h3 : is_circumcircle O A D M)
  (h4 : is_circumcircle O B C M)
  (h5 : lines_intersect_at O M T)
  (h6 : point_on_line O M S) :
  dist T M = dist M S :=
begin
  sorry
end

end midpoint_TS_l346_346703


namespace expression_value_l346_346629

-- Define the difference of squares identity
lemma diff_of_squares (x y : ℤ) : x^2 - y^2 = (x + y) * (x - y) :=
by sorry

-- Define the specific values for x and y
def x := 7
def y := 3

-- State the theorem to be proven
theorem expression_value : ((x^2 - y^2)^2) = 1600 :=
by sorry

end expression_value_l346_346629


namespace triangle_right_triangle_of_consecutive_integers_sum_l346_346664

theorem triangle_right_triangle_of_consecutive_integers_sum (
  m n : ℕ
) (h1 : 0 < m) (h2 : n^2 = 2*m + 1) : 
  n * n + m * m = (m + 1) * (m + 1) := 
sorry

end triangle_right_triangle_of_consecutive_integers_sum_l346_346664


namespace sin_alpha_correct_l346_346699

variables {α β : ℝ}
variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)

-- declaring the conditions
axiom distance_condition : |vector_a α - vector_b β| = 2 * Real.sqrt 5 / 5
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : -Real.pi / 2 < β ∧ β < 0
axiom sin_beta_definition : Real.sin β = -5 / 13

-- declaring the target to be proven
theorem sin_alpha_correct : ∀ α β : ℝ, distance_condition → alpha_range → beta_range → sin_beta_definition → Real.sin α = 33 / 65 :=
sorry

end sin_alpha_correct_l346_346699


namespace length_of_AC_l346_346771

theorem length_of_AC (A B C : Type) (angle_A_eq_90 : ∠ A = 90) 
  (tan_B_eq_3_div_5 : tan B = 3/5) (AB_eq_45 : dist A B = 45) :
  ∃ AC : ℝ, AC ≈ 23.154 :=
by
  sorry

end length_of_AC_l346_346771


namespace sum_of_sequence_first_n_terms_eq_4n_l346_346728

noncomputable def f (x : ℝ) : ℝ :=
if x = 1 then 2 else real.pow 2 x  -- Since f(n) = 2^n for n natural numbers

def a_n (n : ℕ) : ℝ :=
(f n ^ 2 + f (2 * n)) / f (2 * n - 1)

theorem sum_of_sequence_first_n_terms_eq_4n (n : ℕ) : 
  (∑ i in finset.range n, a_n (i + 1)) = 4 * n :=
sorry

end sum_of_sequence_first_n_terms_eq_4n_l346_346728


namespace iterative_average_difference_l346_346640

theorem iterative_average_difference :
  let numbers : List ℕ := [2, 4, 6, 8, 10] 
  let avg2 (a b : ℝ) := (a + b) / 2
  let avg (init : ℝ) (lst : List ℕ) := lst.foldl (λ acc x => avg2 acc x) init
  let max_avg := avg 2 [4, 6, 8, 10]
  let min_avg := avg 10 [8, 6, 4, 2] 
  max_avg - min_avg = 4.25 := 
by
  sorry

end iterative_average_difference_l346_346640


namespace smallest_four_digit_number_in_pascals_triangle_l346_346545

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346545


namespace sum_of_non_domain_points_of_g_l346_346651

def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem sum_of_non_domain_points_of_g :
  let non_domain_points := [-1/2, -2/5, 0]
  (∑ x in non_domain_points, x) = -9 / 10 := by
  sorry

end sum_of_non_domain_points_of_g_l346_346651


namespace smallest_four_digit_in_pascals_triangle_l346_346487

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346487


namespace smallest_four_digit_in_pascals_triangle_l346_346527

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346527


namespace g_at_zero_l346_346025

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_at_zero : g 0 = -Real.sqrt 2 :=
by
  -- proof to be completed
  sorry

end g_at_zero_l346_346025


namespace smallest_four_digit_in_pascal_l346_346491

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346491


namespace parabola_focus_coordinates_l346_346021

theorem parabola_focus_coordinates : 
  (∀ x y : ℝ, y = (1 / 8) * x ^ 2 → (0, 2) is the focus of the parabola) :=
by
  sorry

end parabola_focus_coordinates_l346_346021


namespace total_number_of_students_l346_346306

theorem total_number_of_students (B G T : ℕ)
  (h1 : B = 60)
  (h2 : G = 0.60 * (B + G))
  (h3 : T = B + G) :
  T = 150 := by
  sorry

end total_number_of_students_l346_346306


namespace product_evaluation_l346_346159

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l346_346159


namespace salary_increase_l346_346404

theorem salary_increase (original_salary reduced_salary : ℝ) (hx : reduced_salary = original_salary * 0.5) : 
  (reduced_salary + reduced_salary * 1) = original_salary :=
by
  -- Prove the required increase percent to return to original salary
  sorry

end salary_increase_l346_346404


namespace overall_discount_rate_l346_346928

theorem overall_discount_rate 
  (marked_price_bag : ℝ := 200)
  (marked_price_shirt : ℝ := 80)
  (marked_price_shoes : ℝ := 120)
  (selling_price_bag : ℝ := 120)
  (selling_price_shirt : ℝ := 60)
  (selling_price_shoes : ℝ := 90)
  (total_marked_price : ℝ := marked_price_bag + marked_price_shirt + marked_price_shoes := 400)
  (total_selling_price : ℝ := selling_price_bag + selling_price_shirt + selling_price_shoes := 270)
  (total_discount : ℝ := total_marked_price - total_selling_price := 130) :
  (total_discount / total_marked_price) * 100 = 32.5 := 
  by 
    simp [total_marked_price, total_selling_price, total_discount]
    sorry

end overall_discount_rate_l346_346928


namespace variation_relationship_l346_346287

theorem variation_relationship (k j : ℝ) (y z x : ℝ) (h1 : x = k * y^3) (h2 : y = j * z^(1/5)) :
  ∃ m : ℝ, x = m * z^(3/5) :=
by
  sorry

end variation_relationship_l346_346287


namespace population_of_canada_1998_l346_346772

noncomputable def million : ℕ := 1000000

theorem population_of_canada_1998 : 30.3 * million = 30300000 := by
  sorry

end population_of_canada_1998_l346_346772


namespace apple_tree_percentage_decrease_l346_346637

noncomputable def percentage_decrease (first_season: ℕ) (second_season: ℕ): ℕ :=
  (first_season - second_season) * 100 / first_season

theorem apple_tree_percentage_decrease:
  ∀ (F: ℕ),
    200 + F + (2 * F) = 680 →
    200 > F →
    percentage_decrease 200 F = 20 := 
by
  intros F h1 h2
  calc
    percentage_decrease 200 F = (200 - F) * 100 / 200 := rfl
    ... = 20 := 
      by
        have h_F: F = 160 := by
          linarith
        rw h_F
        norm_num


end apple_tree_percentage_decrease_l346_346637


namespace correct_observation_value_l346_346029

theorem correct_observation_value (mean : ℕ) (n : ℕ) (incorrect_obs : ℕ) (corrected_mean : ℚ) (original_sum : ℚ) (remaining_sum : ℚ) (corrected_sum : ℚ) :
  mean = 30 →
  n = 50 →
  incorrect_obs = 23 →
  corrected_mean = 30.5 →
  original_sum = (n * mean) →
  remaining_sum = (original_sum - incorrect_obs) →
  corrected_sum = (n * corrected_mean) →
  ∃ x : ℕ, remaining_sum + x = corrected_sum → x = 48 :=
by
  intros h_mean h_n h_incorrect_obs h_corrected_mean h_original_sum h_remaining_sum h_corrected_sum
  have original_mean := h_mean
  have observations := h_n
  have incorrect_observation := h_incorrect_obs
  have new_mean := h_corrected_mean
  have original_sum_calc := h_original_sum
  have remaining_sum_calc := h_remaining_sum
  have corrected_sum_calc := h_corrected_sum
  use 48
  sorry

end correct_observation_value_l346_346029


namespace min_value_m_n_l346_346232

variable (a b : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (h_geom_mean : Real.sqrt (a * b) = 2)
def m := b + 1 / a
def n := a + 1 / b

theorem min_value_m_n : m + n = 5 := sorry

end min_value_m_n_l346_346232


namespace take_home_pay_correct_l346_346670

noncomputable def faith_take_home_pay : Float :=
  let regular_hourly_rate := 13.50
  let regular_hours_per_day := 8
  let days_per_week := 5
  let regular_hours_per_week := regular_hours_per_day * days_per_week
  let regular_earnings_per_week := regular_hours_per_week * regular_hourly_rate

  let overtime_rate_multiplier := 1.5
  let overtime_hourly_rate := regular_hourly_rate * overtime_rate_multiplier
  let overtime_hours_per_day := 2
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week
  let overtime_earnings_per_week := overtime_hours_per_week * overtime_hourly_rate

  let total_sales := 3200.0
  let commission_rate := 0.10
  let commission := total_sales * commission_rate

  let total_earnings_before_deductions := regular_earnings_per_week + overtime_earnings_per_week + commission

  let deduction_rate := 0.25
  let amount_withheld := total_earnings_before_deductions * deduction_rate
  let amount_withheld_rounded := (amount_withheld * 100).round / 100

  let take_home_pay := total_earnings_before_deductions - amount_withheld_rounded
  take_home_pay

theorem take_home_pay_correct : faith_take_home_pay = 796.87 :=
by
  /- Proof omitted -/
  sorry

end take_home_pay_correct_l346_346670


namespace monotonicity_and_extremum_intervals_l346_346198

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ -1 ∨ x ≥ 1 then |x^2 - 1| - 2 else 1 - x^2 - 2

theorem monotonicity_and_extremum_intervals :
  (∀ x, x ∈ Ioo (-∞) (-√3) → piecewise_function x' < piecewise_function x) ∧
  (∀ x, x ∈ Ioo (-√3) (-1) → piecewise_function x' > piecewise_function x) ∧
  (∀ x, x ∈ Ioo (-1) 0 → piecewise_function x' < piecewise_function x) ∧
  (∀ x, x ∈ Ioo 0 1 → piecewise_function x' > piecewise_function x) ∧
  (∀ x, x ∈ Ioo 1 (√3) → piecewise_function x' < piecewise_function x) ∧
  (∀ x, x ∈ Ioo (√3) ∞ → piecewise_function x' > piecewise_function x) ∧
  (piecewise_function (-1) = max_piecewise_function) ∧
  (piecewise_function 1 = max_piecewise_function) ∧
  (piecewise_function (-√3) = min_piecewise_function) ∧
  (piecewise_function (√3) = min_piecewise_function) ∧
  (piecewise_function 0 = min_piecewise_function) :=
sorry

end monotonicity_and_extremum_intervals_l346_346198


namespace increasing_interval_of_f_l346_346391

noncomputable def f (x : ℝ) : ℝ := log 2 (2 * x - x^2)

theorem increasing_interval_of_f :
  (∀ x y : ℝ, (0 < x ∧ x ≤ 1 ∧ 0 < y ∧ y ≤ 1 ∧ x < y) → f(x) < f(y)) :=
sorry

end increasing_interval_of_f_l346_346391


namespace collin_savings_l346_346972

def cans_at_home := 12
def cans_at_grandparents := 3 * cans_at_home
def cans_from_neighbor := 46
def cans_from_dad := 250
def total_cans := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_dad

def price_per_can := 0.25
def total_money := total_cans * price_per_can

def amount_to_save := total_money / 2

theorem collin_savings : amount_to_save = 43 := by
  sorry

end collin_savings_l346_346972


namespace smallest_four_digit_in_pascals_triangle_l346_346427

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346427


namespace smallest_four_digit_in_pascals_triangle_l346_346528

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346528


namespace number_of_ways_to_feed_animals_l346_346121

-- Definitions for the conditions
def pairs_of_animals := 5
def alternating_feeding (start_with_female : Bool) (remaining_pairs : ℕ) : ℕ :=
if start_with_female then
  (pairs_of_animals.factorial / 2 ^ pairs_of_animals)
else
  0 -- we can ignore this case as it is not needed

-- Theorem statement
theorem number_of_ways_to_feed_animals :
  alternating_feeding true pairs_of_animals = 2880 :=
sorry

end number_of_ways_to_feed_animals_l346_346121


namespace comic_cost_is_4_l346_346001

-- Define initial amount of money Raul had.
def initial_money : ℕ := 87

-- Define number of comics bought by Raul.
def num_comics : ℕ := 8

-- Define the amount of money left after buying comics.
def money_left : ℕ := 55

-- Define the hypothesis condition about the money spent.
def total_spent : ℕ := initial_money - money_left

-- Define the main assertion that each comic cost $4.
def cost_per_comic (total_spent : ℕ) (num_comics : ℕ) : Prop :=
  total_spent / num_comics = 4

-- Main theorem statement
theorem comic_cost_is_4 : cost_per_comic total_spent num_comics :=
by
  -- Here we're skipping the proof for this exercise.
  sorry

end comic_cost_is_4_l346_346001


namespace contrapositive_example_l346_346870

theorem contrapositive_example (x : ℝ) : (x > 1 → x^2 > 1) → (x^2 ≤ 1 → x ≤ 1) :=
sorry

end contrapositive_example_l346_346870


namespace _l346_346302

-- Definitions of the conditions
structure TriangleABC where
  A B C : Type
  angle_A : ℝ -- ∠A in degrees
  side_AB : ℝ -- Length of AB
  side_AC : ℝ -- Length of AC
  side_BC : ℝ -- Length of BC
  (h_angle_A : angle_A = 20) -- angle ∠A = 20°
  (h_side_AB_AC : side_AB = side_AC) -- AB = AC = a
  (h_side_BC : side_BC = b) -- BC = b

noncomputable def proof_triangle_theorem (ABC : TriangleABC) : Prop :=
  let a := ABC.side_AB
  let b := ABC.side_BC
  a^3 + b^3 = 3 * a^2 * b

lemma triangle_proof (ABC : TriangleABC) : proof_triangle_theorem ABC := 
  sorry

end _l346_346302


namespace mutually_exclusive_not_contradictory_events_l346_346217

theorem mutually_exclusive_not_contradictory_events :
  ∃ (products : Finset (Fin 5)),
    let genuine := {0, 1, 2} : Finset (Fin 5)
    let defective := {3, 4} : Finset (Fin 5)
    (∀ (selected : Finset (Fin 5)), selected.card = 2 →
      ((∀ x ∈ selected, x ∈ defective) ∨ 
      (∃ d ∈ defective, d ∈ selected ∧ ∃ g1 g2 ∈ genuine, g1 ≠ g2 ∧ g1 ∈ selected ∧ g2 ∈ selected)) ∧
      ¬((∀ x ∈ selected, x ∈ defective) ∧ 
      (∃ d ∈ defective, d ∈ selected ∧ ∃ g1 g2 ∈ genuine, g1 ≠ g2 ∧ g1 ∈ selected ∧ g2 ∈ selected))) :=
begin
  sorry
end

end mutually_exclusive_not_contradictory_events_l346_346217


namespace smallest_four_digit_in_pascal_l346_346509

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346509


namespace smallest_four_digit_in_pascal_l346_346526

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346526


namespace point_on_line_probability_l346_346298

noncomputable def dice_probability : ℚ :=
  let valid_points := { (4, 2), (5, 4), (6, 6) }
  let total_outcomes := 36
  valid_points.card / total_outcomes

theorem point_on_line_probability : dice_probability = 1 / 12 := by
  sorry

end point_on_line_probability_l346_346298


namespace log_base_13_of_x_plus_1_l346_346721

theorem log_base_13_of_x_plus_1 (x : ℝ) (h : Real.logBase 7 (x + 5) = 2) : 
  Real.logBase 13 (x + 1) = Real.logBase 13 45 := 
by
  sorry

end log_base_13_of_x_plus_1_l346_346721


namespace cube_faces_one_third_blue_l346_346934

theorem cube_faces_one_third_blue (n : ℕ) (h1 : ∃ n, n > 0 ∧ (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 := by
  sorry

end cube_faces_one_third_blue_l346_346934


namespace correct_choice_option_D_l346_346634

theorem correct_choice_option_D : (500 - 9 * 7 = 437) := by sorry

end correct_choice_option_D_l346_346634


namespace range_of_a_l346_346259

-- Define the function g(x)
def g (x : ℝ) (a : ℝ) : ℝ := (Real.log x) / x + a / (x^2) - 1 / x

-- State the condition for a such that g(x) has extreme values in the interval (1, e^2)
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ ∃ y : ℝ, g.g''' x a = 0) ↔ (0 < a ∧ a < Real.exp 1 / 2) := sorry

end range_of_a_l346_346259


namespace compute_alpha_l346_346331

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end compute_alpha_l346_346331


namespace winning_strategy_for_player_A_l346_346412

def initial_piles := (100, 200, 300)

theorem winning_strategy_for_player_A : 
  ∃ winning_strategy : (nat × nat × nat) → bool, 
    winning_strategy initial_piles = true :=
sorry

end winning_strategy_for_player_A_l346_346412


namespace hydrangea_cost_l346_346367

def cost_of_each_plant : ℕ :=
  let total_years := 2021 - 1989
  let total_amount_spent := 640
  total_amount_spent / total_years

theorem hydrangea_cost :
  cost_of_each_plant = 20 :=
by
  -- skipping the proof for Lean statement
  sorry

end hydrangea_cost_l346_346367


namespace water_percentage_in_fresh_grapes_l346_346692

theorem water_percentage_in_fresh_grapes 
  (P : ℝ) -- the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 40) -- weight of fresh grapes in kg
  (dry_grapes_weight : ℝ := 5) -- weight of dry grapes in kg
  (dried_grapes_water_percentage : ℝ := 20) -- percentage of water in dried grapes
  (solid_content : ℝ := 4) -- solid content in both fresh and dried grapes in kg
  : P = 90 :=
by
  sorry

end water_percentage_in_fresh_grapes_l346_346692


namespace equation_of_line_l346_346722

theorem equation_of_line {M : ℝ × ℝ} {a b : ℝ} (hM : M = (4,2)) 
  (hAB : ∃ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ 
    A ≠ B ∧ ∀ x y : ℝ, 
    (x^2 + 4 * y^2 = 36 → (∃ k : ℝ, y - 2 = k * (x - 4) ) )):
  (x + 2 * y - 8 = 0) :=
sorry

end equation_of_line_l346_346722


namespace smallest_four_digit_number_in_pascals_triangle_l346_346546

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346546


namespace intersection_M_N_l346_346273

def M : Set ℝ := { x | x^2 - 1 ≤ 0 }
def N : Set ℤ := { x | (1 / 2 : ℝ) < 2^(x + 1) ∧ 2^(x + 1) < 4 }

theorem intersection_M_N : M ∩ (N : Set ℝ) = { -1, 0 } := by
  sorry

end intersection_M_N_l346_346273


namespace volume_of_remaining_body_surface_area_of_remaining_body_central_symmetry_of_remaining_body_regularity_condition_of_remaining_body_l346_346947

-- Define the original tetrahedron with volume V and surface area F
variables (V : ℝ) (F : ℝ)

-- Define the conditions of the problem
def tetrahedron_cuts : Prop := 
  ∀ (T : Type) (tetrahedron : T)
    (midpoint_cut_condition : ∀ (v : T), Prop), 
    (cuts : ∀ v : T, midpoint_cut_condition v → T)

-- Volume of the remaining body property
theorem volume_of_remaining_body (V : ℝ) (h : tetrahedron_cuts V F) : 
  (remaining_volume V : ℝ) = V / 2 :=
sorry

-- Surface area of the remaining body property
theorem surface_area_of_remaining_body (F : ℝ) (h : tetrahedron_cuts V F) : 
  (remaining_surface_area F : ℝ) = 2 * (F / 4) :=
sorry

-- Central symmetry of the remaining body property
theorem central_symmetry_of_remaining_body (h : tetrahedron_cuts V F) : 
  ∀ (centroid : ℝ), (centrally_symmetric (centroid)) :=
sorry

-- Regularity condition for the remaining body property
theorem regularity_condition_of_remaining_body (h : tetrahedron_cuts V F) : 
  (original_tetrahedron_regular → remaining_body_regular) :=
sorry

end volume_of_remaining_body_surface_area_of_remaining_body_central_symmetry_of_remaining_body_regularity_condition_of_remaining_body_l346_346947


namespace plant_arrangement_count_l346_346000

-- Define the count of identical plants
def basil_count := 3
def aloe_count := 2

-- Define the count of identical lamps in each color
def white_lamp_count := 3
def red_lamp_count := 3

-- Define the total ways to arrange the plants under the lamps.
def arrangement_ways := 128

-- Formalize the problem statement proving the arrangements count
theorem plant_arrangement_count :
  (∃ f : Fin (basil_count + aloe_count) → Fin (white_lamp_count + red_lamp_count), True) ↔
  arrangement_ways = 128 :=
sorry

end plant_arrangement_count_l346_346000


namespace rectangle_side_length_l346_346622

noncomputable def length_parallel_to_x_axis {a : ℝ} (h1 : 2 * a^3 = 81) : ℝ :=
2 * real.cbrt 40.5

theorem rectangle_side_length {a : ℝ} (h1 : 2 * a^3 = 81) :
  length_parallel_to_x_axis h1 = 2 * real.cbrt 40.5 :=
sorry

end rectangle_side_length_l346_346622


namespace simson_lines_l346_346660

-- Definitions related to the problem setup
noncomputable def Triangle (α : Type*) [Field α] := 
  {A B C : α × α}

-- Definition of a point on the circumcircle of a triangle
noncomputable def on_circumcircle (A B C P : ℝ × ℝ) : Prop := sorry

-- Definition of the Simson line of a point P on the circumcircle
noncomputable def Simson_line (A B C P : ℝ × ℝ) : set (ℝ × ℝ) := sorry

-- Definition of the altitudes of a triangle (as sets of points)
noncomputable def altitude (A B C : ℝ × ℝ) : set (ℝ × ℝ) := sorry

-- Definition of the sides of a triangle (as sets of points)
noncomputable def side (A B C : ℝ × ℝ) : set (ℝ × ℝ) := sorry

-- The theorem we want to prove
theorem simson_lines (A B C : ℝ × ℝ) (P : ℝ × ℝ):
  on_circumcircle A B C P →
  (∃ (Alt : set (ℝ × ℝ)), Simson_line A B C P = Alt ∧ (Alt = altitude A B C)) ∨
  (∃ (Side : set (ℝ × ℝ)), Simson_line A B C P = Side ∧ (Side = side A B C)) →
  (P = A ∨ P = B ∨ P = C ∨ 
  (P = diametrically_opposite A) ∨ (P = diametrically_opposite B) ∨ (P = diametrically_opposite C)) :=
by 
  sorry

end simson_lines_l346_346660


namespace construct_isosceles_triangle_from_points_l346_346358

-- Definitions of the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)
  (is_isosceles : A ≠ B → A ≠ C → B ≠ C → (dist A B = dist A C))

def incenter (A B C : Point) : Point := sorry
def centroid (A B C : Point) : Point := sorry
def orthocenter (A B C : Point) : Point := sorry

-- Assuming the points are given and satisfy their conditions
constant I M H : Point
axiom I_incenter : ∃ (A B C : Point), incenter A B C = I
axiom M_centroid : ∃ (A B C : Point), centroid A B C = M
axiom H_orthocenter : ∃ (A B C : Point), orthocenter A B C = H

-- Theorem to prove the construction of the triangle
theorem construct_isosceles_triangle_from_points :
  ∃ (A B C : Point), (dist A B = dist A C) ∧ incenter A B C = I ∧ centroid A B C = M ∧ orthocenter A B C = H :=
sorry

end construct_isosceles_triangle_from_points_l346_346358


namespace evaluate_expression_l346_346042

theorem evaluate_expression :
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 :=
by 
  sorry

end evaluate_expression_l346_346042


namespace triangles_from_decagon_l346_346988

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346988


namespace trapezoid_area_l346_346308

-- Define the given conditions.
variables (BC AD CD AC BD h : ℝ) -- BC and AD are the bases, CD is one side, AC and BD are the diagonals, h is the height.
axiom BC_AD_eq_5 : BC = 5
axiom CD_eq_3 : CD = 3
axiom AC_perp_CD : AC ⊥ CD
axiom BD_bisects_angle_ADC : ∃ α : ℝ, α ∈ (angle CD (ADCD)) ∧ α ∈ (angle BD (ADBD))

-- Formalize the problem to prove the area's value
theorem trapezoid_area : ∃ h : ℝ, area_of_trapezoid BC AD h = 9.6 :=
sorry

end trapezoid_area_l346_346308


namespace sum_of_valid_z_for_divisibility_6_is_0_l346_346662

theorem sum_of_valid_z_for_divisibility_6_is_0 :
  (∑ z in finset.range 10, if (35 * 10000 + z * 1000 + 4 * 100 + 5 * 10 + 5) % 6 = 0 then z else 0) = 0 :=
sorry

end sum_of_valid_z_for_divisibility_6_is_0_l346_346662


namespace ellipse_eccentricity_range_of_ratio_l346_346151

-- The setup conditions
variables {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (h1 : a^2 - b^2 = c^2)
variables (M : ℝ) (m : ℝ)
variables (hM : M = a + c) (hm : m = a - c) (hMm : M * m = 3 / 4 * a^2)

-- Proof statement for the eccentricity of the ellipse
theorem ellipse_eccentricity : c / a = 1 / 2 := by
  sorry

-- The setup for the second part
variables {S1 S2 : ℝ}
variables (ellipse_eq : ∀ x y : ℝ, (x^2 / (4 * c^2) + y^2 / (3 * c^2) = 1) → x + y = 0)
variables (range_S : S1 / S2 > 9)

-- Proof statement for the range of the given ratio
theorem range_of_ratio : 0 < (2 * S1 * S2) / (S1^2 + S2^2) ∧ (2 * S1 * S2) / (S1^2 + S2^2) < 9 / 41 := by
  sorry

end ellipse_eccentricity_range_of_ratio_l346_346151


namespace overall_loss_percentage_is_correct_l346_346124

-- Define the cost prices of the products
def CP_r := 1500
def CP_m := 2200
def CP_h := 1000

-- Define the selling prices of the products
def SP_r := 1305
def SP_m := 2000
def SP_h := 850

-- Define the losses for each product 
def Loss_r := CP_r - SP_r 
def Loss_m := CP_m - SP_m 
def Loss_h := CP_h - SP_h 

-- Define the total cost price and total selling price
def TCP := CP_r + CP_m + CP_h 
def TSP := SP_r + SP_m + SP_h 

-- Define the total loss
def Total_Loss := TCP - TSP 

-- Define the loss percentage calculation
def Loss_Percentage := (Total_Loss * 100.0) / TCP 

-- The theorem to prove the overall loss percentage is approximately 11.60%
theorem overall_loss_percentage_is_correct : Loss_Percentage ≈ 11.60 := 
sorry

end overall_loss_percentage_is_correct_l346_346124


namespace coeff_of_x_in_expansion_l346_346791

theorem coeff_of_x_in_expansion :
  let f := (1 : ℚ) + (Polynomial.X : Polynomial ℚ)
  let g := (Polynomial.X : Polynomial ℚ) - Polynomial.C (2 / Polynomial.X)
  Polynomial.coeff ((f * g ^ 3).expand ℚ) 1 = -6 := sorry

end coeff_of_x_in_expansion_l346_346791


namespace BC_angle_bisector_ABD_l346_346800

variables (K L M P A D B C O1 O2 : Type) [inner_right_angle K L M]
variables [point P_inside_KLM : P ∈ KLM]
variables [circle_S1 : Type] (center_O1 : O1 ∈ circle_S1) [circle_S1_tangent_LK_LP : is_tangent circle_S1 LK ∧ is_tangent circle_S1 LP]
variables [point_A_on_LK : A ∈ LK] [point_D_on_LP : D ∈ LP]
variables [circle_S2 : Type] (center_O2 : O2 ∈ circle_S2) [circle_S2_tangent_ML_LP : is_tangent circle_S2 ML ∧ is_tangent circle_S2 LP]
variables [point_B_on_LP : B ∈ LP]
variables [O1_lies_on_segment_AB : O1 ∈ segment AB]
variables [C_intersection_O2D_KL : C ∈ intersection (line_through O2 D) (line_through K L)]

theorem BC_angle_bisector_ABD :
  is_angle_bisector (line_through B C) (angle_through A B D) :=
sorry

end BC_angle_bisector_ABD_l346_346800


namespace smallest_four_digit_in_pascal_triangle_l346_346560

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346560


namespace laura_total_owed_l346_346806

-- Define the principal amounts charged each month
def january_charge : ℝ := 35
def february_charge : ℝ := 45
def march_charge : ℝ := 55
def april_charge : ℝ := 25

-- Define the respective interest rates for each month, as decimals
def january_interest_rate : ℝ := 0.05
def february_interest_rate : ℝ := 0.07
def march_interest_rate : ℝ := 0.04
def april_interest_rate : ℝ := 0.06

-- Define the interests accrued for each month's charges
def january_interest : ℝ := january_charge * january_interest_rate
def february_interest : ℝ := february_charge * february_interest_rate
def march_interest : ℝ := march_charge * march_interest_rate
def april_interest : ℝ := april_charge * april_interest_rate

-- Define the totals including original charges and their respective interests
def january_total : ℝ := january_charge + january_interest
def february_total : ℝ := february_charge + february_interest
def march_total : ℝ := march_charge + march_interest
def april_total : ℝ := april_charge + april_interest

-- Define the total amount owed a year later
def total_owed : ℝ := january_total + february_total + march_total + april_total

-- Prove that the total amount owed a year later is $168.60
theorem laura_total_owed :
  total_owed = 168.60 := by
  sorry

end laura_total_owed_l346_346806


namespace smallest_four_digit_number_in_pascals_triangle_l346_346460

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346460


namespace least_positive_integer_with_exactly_seven_factors_l346_346060

theorem least_positive_integer_with_exactly_seven_factors : 
  ∃ (n : ℕ), (∀ n > 0 → n ∈ ℕ) ∧ (∀ m : ℕ, m > 0 ∧ m < n → number_of_factors m ≠ 7) ∧ number_of_factors n = 7 ∧ n = 64 :=
by
  sorry

open Nat

/--
Defines the number of factors of a given number.
-/
def number_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0).card

attribute [simp] number_of_factors -- Simplifies and registers the number_of_factors as a known attribute

end least_positive_integer_with_exactly_seven_factors_l346_346060


namespace smallest_four_digit_in_pascal_l346_346521

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346521


namespace parallel_lines_l346_346880

theorem parallel_lines (a : ℝ) : (2 * a = a * (a + 4)) → a = -2 :=
by
  intro h
  sorry

end parallel_lines_l346_346880


namespace fraction_transformation_l346_346290

variables (a b : ℝ)

theorem fraction_transformation (ha : a ≠ 0) (hb : b ≠ 0) : 
  (4 * a * b) / (2 * (2 * a) + 2 * b) = 2 * (a * b) / (2 * a + b) :=
by
  sorry

end fraction_transformation_l346_346290


namespace smallest_four_digit_in_pascal_l346_346516

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346516


namespace polynomial_root_constant_term_l346_346374

theorem polynomial_root_constant_term 
  (P Q : Polynomial ℝ)
  (p q r : ℝ)
  (hp : P = Polynomial.C 1 * polynomial.X^3 + Polynomial.C 5 * polynomial.X^2 + Polynomial.C 7 * polynomial.X - Polynomial.C 18)
  (hP : P = (Polynomial.x - p) * (Polynomial.x - q) * (Polynomial.x - r))
  (hQ : Q = (Polynomial.x - (p + q)) * (Polynomial.x - (q + r)) * (Polynomial.x - (r + p))) : 
  Q.coeff 0 = 179 := 
sorry

end polynomial_root_constant_term_l346_346374


namespace semi_axes_of_ellipse_l346_346860

theorem semi_axes_of_ellipse (x y : ℝ) :
  (x^2 + y^2 - 12 = 2x + 4y) →
  (∃ (a b : ℝ), a = sqrt 17 ∧ b = sqrt 8.5) :=
by
  intro h
  -- Here you would transform and solve the equation as described in the solution steps.
  sorry

end semi_axes_of_ellipse_l346_346860


namespace coefficient_of_x_eq_neg_6_l346_346794

noncomputable def coefficient_of_x_in_expansion : ℤ :=
  let expanded_expr := (1 + x) * (x - 2 / x) ^ 3
  in collect (x : ℝ) expanded_expr

theorem coefficient_of_x_eq_neg_6 : coefficient_of_x_in_expansion = -6 :=
  sorry

end coefficient_of_x_eq_neg_6_l346_346794


namespace sufficient_but_not_necessary_condition_l346_346693

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / 4) ^ a > (1 / 4) ^ b :=
by
  sorry

end sufficient_but_not_necessary_condition_l346_346693


namespace smallest_four_digit_in_pascal_l346_346505

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346505


namespace flagpole_height_l346_346601

/-
A flagpole is of certain height. It breaks, folding over in half, such that what was the tip of the flagpole is now dangling two feet above the ground. 
The flagpole broke 7 feet from the base. Prove that the height of the flagpole is 16 feet.
-/

theorem flagpole_height (H : ℝ) (h1 : H > 0) (h2 : H - 7 > 0) (h3 : H - 9 = 7) : H = 16 :=
by
  /- the proof is omitted -/
  sorry

end flagpole_height_l346_346601


namespace smallest_four_digit_in_pascals_triangle_l346_346476

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346476


namespace quadratic_root_condition_l346_346767

theorem quadratic_root_condition (m n : ℝ) (h : m * (-1)^2 - n * (-1) - 2023 = 0) :
  m + n = 2023 :=
sorry

end quadratic_root_condition_l346_346767


namespace one_by_one_square_positions_l346_346631

theorem one_by_one_square_positions (decomposed: set (ℕ × ℕ)) 
  (H1 : ∀ x ∈ decomposed, (x.1 < 24) ∧ (x.2 < 24))
  (H2 : ∃! x, x ∈ decomposed ∧ x = (1, 1)) :
  ∃ (positions : set (ℕ × ℕ)), positions = {(6, 6), (6, 12), (6, 18), (12, 6), (12, 12), (12, 18), (18, 6), (18, 12), (18, 18)} :=
sorry

end one_by_one_square_positions_l346_346631


namespace probability_all_boxes_non_empty_equals_4_over_9_l346_346945

structure PaintingPlacement :=
  (paintings : Finset ℕ)
  (boxes : Finset ℕ)
  (num_paintings : paintings.card = 4)
  (num_boxes : boxes.card = 3)

noncomputable def probability_non_empty_boxes (pp : PaintingPlacement) : ℚ :=
  let total_outcomes := 3^4
  let favorable_outcomes := Nat.choose 4 2 * Nat.factorial 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_all_boxes_non_empty_equals_4_over_9
  (pp : PaintingPlacement) : pp.paintings.card = 4 → pp.boxes.card = 3 →
  probability_non_empty_boxes pp = 4 / 9 :=
by
  intros h1 h2
  sorry

end probability_all_boxes_non_empty_equals_4_over_9_l346_346945


namespace coeff_of_x_in_expansion_l346_346790

theorem coeff_of_x_in_expansion :
  let f := (1 : ℚ) + (Polynomial.X : Polynomial ℚ)
  let g := (Polynomial.X : Polynomial ℚ) - Polynomial.C (2 / Polynomial.X)
  Polynomial.coeff ((f * g ^ 3).expand ℚ) 1 = -6 := sorry

end coeff_of_x_in_expansion_l346_346790


namespace sequence_properties_l346_346265

def sequence_sum (S_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 2 * n ^ 2 - 10 * n

def general_term (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) : Prop :=
  ∀ n ≥ 1, a_n n = S_n n - S_n (n - 1)

def minimum_value (S_n : ℕ → ℤ) : Prop :=
  ∃ n, S_n n = -12

theorem sequence_properties :
  ∃ a_n S_n, sequence_sum S_n ∧ general_term a_n S_n ∧ minimum_value S_n :=
by
  let a_n := λ n, 4 * n - 5
  let S_n := λ n, 2 * n ^ 2 - 10 * n
  existsi [a_n, S_n]
  intro n
  unfold sequence_sum general_term minimum_value
  sorry

end sequence_properties_l346_346265


namespace cos_angle_eq_sqrt5_div_5_l346_346242

open Real

-- Conditions
def m : EuclideanSpace ℝ (Fin 2) := ![2, 2]
def n : EuclideanSpace ℝ (Fin 2) := ![-1, 3]
def θ : ℝ := angle m n

-- Goal
theorem cos_angle_eq_sqrt5_div_5 : 
  cos θ = sqrt 5 / 5 := 
by 
  sorry

end cos_angle_eq_sqrt5_div_5_l346_346242


namespace decreasing_interval_l346_346395

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x > -2 ∧ x < 0 → deriv f x < 0 := 
by
  intro x h
  sorry

end decreasing_interval_l346_346395


namespace minimize_distance_l346_346231

theorem minimize_distance (m : ℚ) (A B C : ℝ×ℝ) 
  (hA : A = (-2, -3)) (hB : B = (3, 1)) (hC : C = (0, m)) :
  (∀ C', ∀ x : ℚ, x = 0 → C' = (x, m) → 
    dist A C' + dist C' B ≤ dist A C + dist C B) ↔ m = -7/5 := 
by
  sorry

end minimize_distance_l346_346231


namespace min_value_2x2_minus_1_max_value_neg2_x_plus_1_squared_plus_1_min_value_2x2_minus_4x_plus_1_l346_346847

-- Problem 1: Prove \(2x^2 - 1 = -1\) when \(x = 0\)
theorem min_value_2x2_minus_1 : ∀ x : ℝ, x = 0 → 2 * x^2 - 1 = -1 :=
by
  intro x h
  rw h
  sorry

-- Problem 2: Prove \(-2 (x+1)^2 + 1 = 1\) when \(x = -1\)
theorem max_value_neg2_x_plus_1_squared_plus_1 : ∀ x : ℝ, x = -1 → -2 * (x + 1)^2 + 1 = 1 :=
by
  intro x h
  rw h
  sorry

-- Problem 3: Prove \(2x^2 - 4x + 1 = -1\) when \(x = 1\)
theorem min_value_2x2_minus_4x_plus_1 : ∀ x : ℝ, x = 1 → 2 * x^2 - 4 * x + 1 = -1 :=
by
  intro x h
  rw h
  sorry

end min_value_2x2_minus_1_max_value_neg2_x_plus_1_squared_plus_1_min_value_2x2_minus_4x_plus_1_l346_346847


namespace percentage_not_even_integers_l346_346195

variable (T : ℝ) (E : ℝ)
variables (h1 : 0.36 * T = E * 0.60) -- Condition 1 translated: 36% of T are even multiples of 3.
variables (h2 : E * 0.40)            -- Condition 2 translated: 40% of E are not multiples of 3.

theorem percentage_not_even_integers : 0.40 * T = T - E :=
by
  sorry

end percentage_not_even_integers_l346_346195


namespace curve_is_parabola_l346_346679

def curve_eq (r θ : ℝ) : Prop :=
  r = 1 / (1 - sin θ)

theorem curve_is_parabola :
  ∀ r θ x y : ℝ,
  curve_eq r θ →
  x = r * cos θ →
  y = r * sin θ →
  y = (x^2 - 1) / 2 :=
by
  sorry

end curve_is_parabola_l346_346679


namespace minimum_value_function_l346_346687

theorem minimum_value_function (x : ℝ) (h : x > -1) : 
  (∃ y, y = (x^2 + 7 * x + 10) / (x + 1) ∧ y ≥ 9) :=
sorry

end minimum_value_function_l346_346687


namespace total_cost_for_photos_l346_346596

def total_cost (n : ℕ) (f : ℝ) (c : ℝ) : ℝ :=
  f + (n - 4) * c

theorem total_cost_for_photos :
  total_cost 54 24.5 2.3 = 139.5 :=
by
  sorry

end total_cost_for_photos_l346_346596


namespace sqrt_product_simplification_l346_346645

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end sqrt_product_simplification_l346_346645


namespace probability_three_of_a_kind_l346_346851

noncomputable def probability_of_three_of_a_kind_after_rerolling 
  (dice : Fin 6 → Fin 6)
  (no_four_of_a_kind : ∀ n : Fin 6, ∃! t : Finset (Fin 6), t.card = 3 ∧ (∀ i ∈ t, dice i = n) → false)
  (two_pairs : ∃ a b : Fin 6, a ≠ b ∧ (∃ t1 t2 : Finset (Fin 6), t1.card = 2 ∧ t2.card = 2 ∧ 
  (∀ i ∈ t1, dice i = a) ∧ (∀ i ∈ t2, dice i = b) ∧ disjoint t1 t2)) : 
  ℝ := 2 / 3

theorem probability_three_of_a_kind 
  (dice : Fin 6 → Fin 6)
  (no_four_of_a_kind : ∀ n : Fin 6, ∃! t : Finset (Fin 6), t.card = 3 ∧ (∀ i ∈ t, dice i = n) → false)
  (two_pairs : ∃ a b : Fin 6, a ≠ b ∧ (∃ t1 t2 : Finset (Fin 6), t1.card = 2 ∧ t2.card = 2 ∧ 
  (∀ i ∈ t1, dice i = a) ∧ (∀ i ∈ t2, dice i = b) ∧ disjoint t1 t2)) : 
  probability_of_three_of_a_kind_after_rerolling dice no_four_of_a_kind two_pairs = 2 / 3 := 
  sorry

end probability_three_of_a_kind_l346_346851


namespace area_of_triangle_is_correct_l346_346678

def point := ℚ × ℚ

def A : point := (4, -4)
def B : point := (-1, 1)
def C : point := (2, -7)

def vector_sub (p1 p2 : point) : point :=
(p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℚ :=
v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℚ :=
(abs (determinant (vector_sub C A) (vector_sub C B))) / 2

theorem area_of_triangle_is_correct :
  area_of_triangle A B C = 12.5 :=
by sorry

end area_of_triangle_is_correct_l346_346678


namespace min_sum_seq_l346_346815

noncomputable def f : ℝ → ℝ := sorry -- assume f satisfies all given conditions

-- Define sequence a_n based on the function f
def a (n : ℕ) : ℝ :=
  match n with
  | 0   => 0 -- a_0 is not defined in the problem, so we use 0 here
  | n+1 => f (n + 1)

-- Define the sum S_n of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- The theorem to prove that states the minimum value of sum S_n is 1
theorem min_sum_seq : ∀ n : ℕ, (∃ m : ℕ, S m = 1) :=
  sorry

end min_sum_seq_l346_346815


namespace truncated_pyramid_lateral_surface_area_l346_346143

noncomputable def lateralSurfaceAreaTruncatedPyramid (s1 s2 h : ℝ) :=
  let l := Real.sqrt (h^2 + ((s1 - s2) / 2)^2)
  let P1 := 4 * s1
  let P2 := 4 * s2
  (1 / 2) * (P1 + P2) * l

theorem truncated_pyramid_lateral_surface_area :
  lateralSurfaceAreaTruncatedPyramid 10 5 7 = 222.9 :=
by
  sorry

end truncated_pyramid_lateral_surface_area_l346_346143


namespace product_simplification_l346_346969

theorem product_simplification :
  ∏ (n : ℕ) in (finset.range 99).filter (λ n, n ≥ 2), (n * (n + 2)) / ((n + 1) ^ 2) = 101 / 150 :=
by
  sorry

end product_simplification_l346_346969


namespace smallest_four_digit_number_in_pascals_triangle_l346_346544

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346544


namespace smallest_four_digit_number_in_pascals_triangle_l346_346457

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346457


namespace range_of_common_ratio_l346_346731

theorem range_of_common_ratio (q : ℝ) (a : ℝ) (h_q : q > 0) :
    ((a > 0) ∧ (a + q * a > q^2 * a) ∧ (a + q^2 * a > q * a) ∧ (q * a + q^2 * a > a)) →
    q ∈ set.Ioo ((sqrt 5 - 1) / 2) ((1 + sqrt 5) / 2) := 
  by
  sorry

end range_of_common_ratio_l346_346731


namespace number_of_integers_leq_zero_l346_346177

theorem number_of_integers_leq_zero :
  let P (x : ℤ) := (list.range 50).foldl (λ acc i, acc * (x - (i + 1))) 1
  in finset.filter (λ n, P n ≤ 0) (finset.range 100).card = 75 :=
by sorry

end number_of_integers_leq_zero_l346_346177


namespace church_full_capacity_l346_346047

theorem church_full_capacity
  (chairs_per_row : ℕ)
  (rows : ℕ)
  (people_per_chair : ℕ)
  (h1 : chairs_per_row = 6)
  (h2 : rows = 20)
  (h3 : people_per_chair = 5) :
  (chairs_per_row * rows * people_per_chair) = 600 := by
  sorry

end church_full_capacity_l346_346047


namespace A_wears_white_hat_l346_346414

-- Define the hats and their colors
inductive HatColor
| white : HatColor
| black : HatColor

open HatColor

-- Define the individuals A, B, and C
structure Individual :=
(name : String)

def A : Individual := ⟨"A"⟩
def B : Individual := ⟨"B"⟩
def C : Individual := ⟨"C"⟩

-- Define the state of visibility
structure Visibility :=
(can_see : Individual → Individual → Bool)

-- Conditions for visibility
def visibility : Visibility :=
{ can_see := λ i j,
    (i = C ∧ (j = A ∨ j = B)) ∨
    (i = B ∧ j = A) }

-- Define the hat assigning conditions
variable (hats : Individual → HatColor)
def hat_assignment_condition (hats : Individual → HatColor) : Prop := 
  ∃ h1 h2 h3 h4 h5, 
    Multiset.card (Multiset.filter (λ h, h = white) [h1, h2, h3, h4, h5]) = 3 ∧
    Multiset.card (Multiset.filter (λ h, h = black) [h1, h2, h3, h4, h5]) = 2 ∧
    ∃ htA htB htC,
      hats A = htA ∧ htA ∈ [h1, h2, h3, h4, h5] ∧
      hats B = htB ∧ htB ∈ [h1, h2, h3, h4, h5] ∧
      hats C = htC ∧ htC ∈ [h1, h2, h3, h4, h5]

-- Define the knowledge reasoning properties as given in the problem statement
def C_knows_hat (hats : Individual → HatColor) : Prop :=
  (hats A = black ∧ hats B = black) → false

def B_knows_hat (hats : Individual → HatColor) : Prop :=
  (hats A = black) → false

def A_knows_hat (hats : Individual → HatColor) : Prop :=
  hats A = white

-- The theorem proving that A is wearing a white hat
theorem A_wears_white_hat :
  ∀ (hats : Individual → HatColor),
  hat_assignment_condition hats →
  (C_knows_hat hats) →
  (B_knows_hat hats) →
  A_knows_hat hats :=
sorry

end A_wears_white_hat_l346_346414


namespace hans_deposit_l346_346280

noncomputable def calculate_deposit : ℝ :=
  let flat_fee := 30
  let kid_deposit := 2 * 3
  let adult_deposit := 8 * 6
  let senior_deposit := 5 * 4
  let student_deposit := 3 * 4.5
  let employee_deposit := 2 * 2.5
  let total_deposit_before_service := flat_fee + kid_deposit + adult_deposit + senior_deposit + student_deposit + employee_deposit
  let service_charge := total_deposit_before_service * 0.05
  total_deposit_before_service + service_charge

theorem hans_deposit : calculate_deposit = 128.63 :=
by
  sorry

end hans_deposit_l346_346280


namespace chess_tournament_games_count_l346_346773

/-- In a chess tournament with 16 participants, each playing exactly one game with each other, the total number of games played is 120.-/
theorem chess_tournament_games_count : ∑ k in range 15, 16 - k = 120 :=
by
  sorry

end chess_tournament_games_count_l346_346773


namespace cos_alpha_minus_half_beta_l346_346284

theorem cos_alpha_minus_half_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α - β / 2) = Real.sqrt 6 / 3 :=
by
  sorry

end cos_alpha_minus_half_beta_l346_346284


namespace opposite_and_reciprocal_numbers_l346_346718

theorem opposite_and_reciprocal_numbers (a b c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1) :
  2019 * a + (7 / (c * d)) + 2019 * b = 7 :=
sorry

end opposite_and_reciprocal_numbers_l346_346718


namespace proportional_segments_parallel_l346_346104

theorem proportional_segments_parallel (A B C B' C' : Point) :
  (B' ∈ line A B) → (C' ∈ line A C) → (segment B' C' ∥ segment B C ↔
  (dist A B') / (dist B' B) = (dist A C') / (dist C' C)) := by
  sorry

end proportional_segments_parallel_l346_346104


namespace maximum_area_l346_346957

variable {l w : ℝ}

theorem maximum_area (h1 : l + w = 200) (h2 : l ≥ 90) (h3 : w ≥ 50) (h4 : l ≤ 2 * w) : l * w ≤ 10000 :=
sorry

end maximum_area_l346_346957


namespace tablet_battery_life_l346_346832

theorem tablet_battery_life :
  ∀ (active_usage_hours idle_usage_hours : ℕ),
  active_usage_hours + idle_usage_hours = 12 →
  active_usage_hours = 3 →
  ((active_usage_hours / 2) + (idle_usage_hours / 10)) > 1 →
  idle_usage_hours = 9 →
  0 = 0 := 
by
  intros active_usage_hours idle_usage_hours h1 h2 h3 h4
  sorry

end tablet_battery_life_l346_346832


namespace plotted_points_parabola_l346_346208

variable {t : ℝ}

def x := 3^t - 4
def y := 9^t - 7 * 3^t + 2

theorem plotted_points_parabola : y = x^2 + x - 10 := by
  unfold x y
  calc
  y = 9^t - 7 * 3^t + 2 : rfl
  ... = (3^t)^2 - 7 * 3^t + 2 : by rw [Real.mul_self_eq_square]
  ... = (x + 4)^2 - 7 * (x + 4) + 2 : by rw [(show 3^t = x + 4, by sorry)]
  ... = x^2 + 8 * x + 16 - 7 * x - 28 + 2 : by sorry
  ... = x^2 + x - 10 : by sorry

end plotted_points_parabola_l346_346208


namespace calvin_weight_after_one_year_l346_346168

theorem calvin_weight_after_one_year :
  ∀ (initial_weight weight_loss_per_month months : ℕ),
  initial_weight = 250 →
  weight_loss_per_month = 8 →
  months = 12 →
  (initial_weight - (weight_loss_per_month * months) = 154) := by
  intros initial_weight weight_loss_per_month months 
  intro h1 h2 h3
  rw [h1, h2, h3]
  show (250 - (8 * 12) = 154)
  norm_num
  sorry

end calvin_weight_after_one_year_l346_346168


namespace percent_increase_decrease_condition_l346_346286

theorem percent_increase_decrease_condition (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq50 : q < 50) :
  (M * (1 + p / 100) * (1 - q / 100) < M) ↔ (p < 100 * q / (100 - q)) := 
sorry

end percent_increase_decrease_condition_l346_346286


namespace part1_subsets_m_0_part2_range_m_l346_346233

namespace MathProof

variables {α : Type*} {m : ℝ}

def A := {x : ℝ | x^2 + 5 * x - 6 = 0}
def B (m : ℝ) := {x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 3 = 0}
def subsets (A : Set ℝ) := {s : Set ℝ | s ⊆ A}

theorem part1_subsets_m_0 :
  subsets (A ∪ B 0) = {∅, {-6}, {1}, {-3}, {-6,1}, {-6,-3}, {1,-3}, {-6,1,-3}} :=
sorry

theorem part2_range_m (h : ∀ x, x ∈ B m → x ∈ A) : m ≤ -2 :=
sorry

end MathProof

end part1_subsets_m_0_part2_range_m_l346_346233


namespace smallest_four_digit_in_pascals_triangle_l346_346529

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346529


namespace same_halves_exist_l346_346587

theorem same_halves_exist (total_dominoes : ℕ) (row_dominoes : ℕ)
  (remaining_halves : set ℕ) (distinct_halves : remaining_halves = {a_1, a_2, b_1, b_2})
  (total_eq : total_dominoes = 28) (row_eq : row_dominoes = 26) :
  ∃ x y ∈ {a_1, a_2, b_1, b_2}, a_1 = a_2 ∨ a_1 = b_1 ∨ a_1 = b_2 ∨ a_2 = b_1 ∨ a_2 = b_2 ∨ b_1 = b_2 :=
by sorry

end same_halves_exist_l346_346587


namespace smallest_four_digit_in_pascals_triangle_l346_346530

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346530


namespace sum_numbers_greater_than_1_1_l346_346147

def numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
def threshold := 1.1
def greater_than_threshold := numbers.filter (λ x, x > threshold)
def sum_of_greater_than_threshold := greater_than_threshold.sum

theorem sum_numbers_greater_than_1_1 : sum_of_greater_than_threshold = 3.9 :=
by {
  sorry
}

end sum_numbers_greater_than_1_1_l346_346147


namespace marvin_substitute_correct_l346_346833

theorem marvin_substitute_correct {a b c d f : ℤ} (ha : a = 3) (hb : b = 4) (hc : c = 7) (hd : d = 5) :
  (a + (b - (c + (d - f))) = 5 - f) → f = 5 :=
sorry

end marvin_substitute_correct_l346_346833


namespace expansion_sum_l346_346300

theorem expansion_sum (A B C : ℤ) (h1 : A = (2 - 1)^10) (h2 : B = (2 + 0)^10) (h3 : C = -5120) : 
A + B + C = -4095 :=
by 
  sorry

end expansion_sum_l346_346300


namespace max_profit_l346_346602

variable (x y : ℕ)

-- Definition of linear relationship between sales quantity y and selling price x with given points
def linear_sales_function : (x = 45 ∧ y = 110) ∨ (x = 60 ∧ y = 80) ∨ (x = 70 ∧ y = 60) ∨ (x = 75 ∧ y = 50) → y = -2 * x + 200 :=
sorry

-- Definition for the cost price of the fruit
def cost_price : ℕ := 40

-- Function to calculate the profit given the selling price and sales quantity
def profit (x : ℕ) : ℕ := (x - cost_price) * (-2 * x + 200)

-- Prove that the maximum profit per week is obtained at 70 yuan per kg
theorem max_profit : profit 70 = 1800 :=
sorry

end max_profit_l346_346602


namespace min_value_of_expression_l346_346226

theorem min_value_of_expression (A B C : ℝ) (area : ℝ) (a : ℝ) (sin_A : ℝ) :
  (area = 1) →
  (a = 2 / sin_A) →
  (sin_A = sin A) →
  (a^2 + 1 / sin_A) ≥ 3 := 
sorry

end min_value_of_expression_l346_346226


namespace greatest_int_le_sqrt_expr_l346_346684

theorem greatest_int_le_sqrt_expr : ∃ N : ℤ, N ≤ int.sqrt (2007^2 - 20070 + 31) ∧ ∀ x : ℤ, x ≤ int.sqrt (2007^2 - 20070 + 31) → x ≤ 2002 :=
by
  sorry

end greatest_int_le_sqrt_expr_l346_346684


namespace calvin_weight_after_one_year_l346_346167

theorem calvin_weight_after_one_year :
  ∀ (initial_weight weight_loss_per_month months : ℕ),
  initial_weight = 250 →
  weight_loss_per_month = 8 →
  months = 12 →
  (initial_weight - (weight_loss_per_month * months) = 154) := by
  intros initial_weight weight_loss_per_month months 
  intro h1 h2 h3
  rw [h1, h2, h3]
  show (250 - (8 * 12) = 154)
  norm_num
  sorry

end calvin_weight_after_one_year_l346_346167


namespace midpoint_AB_l346_346010

variable (A B C D E F G M : Point)
variable (circle : Point → Point → Circle) -- a function to create a circle with a given center and a point on its circumference

-- Definitions of circles as per conditions
def circle_A_through_B := circle A B
def circle_B_through_A := circle B A
def circle_C_through_A := circle C A
def circle_D_through_B := circle D B
def circle_E_through_A := circle E A
def circle_F_through_A := circle F A
def circle_G_through_A := circle G A

-- Theorem stating M is the midpoint of AB
theorem midpoint_AB : is_midpoint M A B := sorry

end midpoint_AB_l346_346010


namespace interest_rate_correct_l346_346359

-- Define the principal amount P and simple interest SI
def P : ℝ := 1200
def SI : ℝ := 192

-- Define the rate of interest R (which is equal to T)
def R : ℝ := 4

-- Prove that the simple interest calculation satisfies the given condition
theorem interest_rate_correct : 12 * R^2 = SI :=
by
  have h : R^2 = 16 := by sorry
  rw [h]
  norm_num
-- skip the proof temporarily

end interest_rate_correct_l346_346359


namespace new_bathroom_area_l346_346036

variable (area : ℕ) (width : ℕ) (extension : ℕ)

theorem new_bathroom_area (h1 : area = 96) (h2 : width = 8) (h3 : extension = 2) :
  (let orig_length := area / width;
       new_length := orig_length + extension;
       new_width := width + extension;
       new_area := new_length * new_width
   in new_area) = 140 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end new_bathroom_area_l346_346036


namespace necessary_condition_ac2_bc2_true_proposition_l346_346953

theorem necessary_condition_ac2_bc2
  (a b c : ℝ)
  (h : a > b) :
  (ac2: a * (c ^ 2)) > (bc2: b * (c ^ 2)) :=
by sorry

theorem true_proposition :
  ∀ (A B C D : Prop),
  (A = (∃ x : ℝ, (x > 3) → (x > 5))) →
  (B = (∃ x : ℝ, (x^2 = 1) → (x = 1))) →
  (C = (∀ a b c : ℝ, (a > b) → (a * c ^ 2 > b * c ^ 2))) →
  (D = (∀ α : ℝ, (α = π / 2) → (sin α = 1))) →
  (C = true) ∧ (A = false) ∧ (B = false) ∧ (D = false) :=
by
  sorry

end necessary_condition_ac2_bc2_true_proposition_l346_346953


namespace smallest_four_digit_in_pascal_l346_346492

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346492


namespace number_of_even_digits_in_1250_base7_sum_of_even_digits_in_1250_base7_not_divisible_by_3_l346_346200

def is_even (n : ℕ) : Prop := n % 2 = 0

def base7_representation (n : ℕ) : List ℕ :=
  let rec digits (n r : ℕ) : List ℕ :=
    if r = 0 then []
    else let x := r / n
         in x :: digits n (r % n)
  digits 7 n

theorem number_of_even_digits_in_1250_base7 :
  let digits := base7_representation 1250
  let even_digits := List.filter is_even digits
  List.length even_digits = 2 :=
by
  let digits := base7_representation 1250
  let even_digits := List.filter is_even digits
  sorry

theorem sum_of_even_digits_in_1250_base7_not_divisible_by_3 :
  let digits := base7_representation 1250
  let even_digits := List.filter is_even digits
  let sum_even_digits := List.sum even_digits
  ¬ sum_even_digits % 3 = 0 :=
by
  let digits := base7_representation 1250
  let even_digits := List.filter is_even digits
  let sum_even_digits := List.sum even_digits
  sorry

end number_of_even_digits_in_1250_base7_sum_of_even_digits_in_1250_base7_not_divisible_by_3_l346_346200


namespace number_of_valid_x_values_l346_346690

def is_valid_integer_x (x : ℤ) : Prop :=
  let a := 6 + x
  let b := 3 - x
  a ≥ 0 ∧ b ≥ 0

def count_valid_integers : ℕ :=
  (List.range (6 + 1)).length + (List.range 1 4).length

theorem number_of_valid_x_values : count_valid_integers = 10 :=
  by
  sorry

end number_of_valid_x_values_l346_346690


namespace true_propositions_l346_346636

theorem true_propositions :
  (∀ x : ℚ, ∃ y : ℚ, y = (1/3 : ℚ) * x^2 + (1/2 : ℚ) * x + 1) ∧
  (∃ x y : ℤ, 3 * x - 2 * y = 10) :=
by {
  sorry
}

end true_propositions_l346_346636


namespace num_good_words_length_8_l346_346656

-- Define the concept of a good word of length 8
def is_good_word (s : String) : Prop :=
  (∀ i < s.length - 1, s.get i ≠ 'A' ∨ s.get (i + 1) ≠ 'B') ∧
  (∀ i < s.length - 1, s.get i ≠ 'B' ∨ s.get (i + 1) ≠ 'C') ∧
  (∀ i < s.length - 1, s.get i ≠ 'C' ∨ s.get (i + 1) ≠ 'D') ∧
  (∀ i < s.length - 1, s.get i ≠ 'D' ∨ s.get (i + 1) ≠ 'A')

def good_words_of_length_8 : Finset String :=
  {s | s.length = 8 ∧ is_good_word s}

theorem num_good_words_length_8 : (good_words_of_length_8.card = 8748) :=
  sorry

end num_good_words_length_8_l346_346656


namespace max_path_length_cp_diam_l346_346186

theorem max_path_length_cp_diam (A B O C D P : Type) [metric_space A] [metric_space B] [metric_space O] [metric_space C] [metric_space D] [metric_space P]
  (h_diameter_AB : dist A B = 12)
  (h_C_on_AB : dist C A = 3 ∧ dist C B = 9)
  (h_D_on_AB : dist D B = 3 ∧ dist D A = 9)
  (h_P_on_circle : dist P O = 6)
  :
  ∃ P, (dist C P + dist P D = 6 * real.sqrt 3) ↔ is_right_triangle (triangle C P D) :=
sorry

end max_path_length_cp_diam_l346_346186


namespace smallest_four_digit_in_pascals_triangle_l346_346419

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346419


namespace smallest_four_digit_in_pascal_l346_346523

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346523


namespace smallest_four_digit_number_in_pascals_triangle_l346_346446

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346446


namespace remainder_of_N_l346_346649

-- Definition of the sequence constraints
def valid_sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ (∀ i, a i < 512) ∧ (∀ k, 1 ≤ k → k ≤ 9 → ∃ m, 0 ≤ m ∧ m ≤ k - 1 ∧ ((a k - 2 * a m) * (a k - 2 * a m - 1) = 0))

-- Defining N as the number of sequences that are valid.
noncomputable def N : ℕ :=
  Nat.factorial 10 - 2^9

-- The goal is to prove that N mod 1000 is 288
theorem remainder_of_N : N % 1000 = 288 :=
  sorry

end remainder_of_N_l346_346649


namespace geometric_product_l346_346795

theorem geometric_product (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 10) 
  (h2 : 1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 = 5) : 
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end geometric_product_l346_346795


namespace find_m_l346_346762

theorem find_m (m : ℝ) (h : (1 : ℝ) ^ 2 - m * (1 : ℝ) + 2 = 0) : m = 3 :=
by
  sorry

end find_m_l346_346762


namespace find_values_of_r_l346_346583

theorem find_values_of_r (n : ℕ) (a r x : Fin n → ℝ) 
  (h_nonzero : ∃ i, a i ≠ 0) 
  (h_ineq : ∀ x : Fin n → ℝ, 
    ∑ i, r i * (x i - a i) ≤ Real.sqrt (∑ i, (x i) ^ 2) - Real.sqrt (∑ i, (a i) ^ 2)) :
  ∀ i, r i = a i / Real.sqrt (∑ i, (a i) ^ 2) :=
sorry

end find_values_of_r_l346_346583


namespace inequality_proof_equality_condition_l346_346582

variables {a b c x y z : ℕ}

theorem inequality_proof (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 ≤ (c + z) ^ 2 :=
sorry

theorem equality_condition (h1 : a ^ 2 + b ^ 2 = c ^ 2) (h2 : x ^ 2 + y ^ 2 = z ^ 2) : 
  (a + x) ^ 2 + (b + y) ^ 2 = (c + z) ^ 2 ↔ a * z = c * x ∧ a * y = b * x :=
sorry

end inequality_proof_equality_condition_l346_346582


namespace sqrt_product_is_four_l346_346164

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end sqrt_product_is_four_l346_346164


namespace probability_area_equilateral_triangle_between_sqrt3_and_4sqrt3_l346_346221

-- Define the problem conditions
variables (AB_length : ℝ) (AP_length : ℝ)

-- Assumptions
axiom AB_length_eq_5 : AB_length = 5
axiom 0_le_AP_length : 0 ≤ AP_length
axiom AP_length_le_AB_length : AP_length ≤ AB_length

-- Define the area of the equilateral triangle
def area_equilateral (a : ℝ) : ℝ :=
  (sqrt 3 / 4) * (a ^ 2)

-- Define the probability problem
noncomputable def probability_of_area_in_range (AB_length AP_length : ℝ) : ℝ :=
  if sqrt 3 < area_equilateral AP_length ∧ area_equilateral AP_length < 4 * sqrt 3 then
    (4 - 2) / AB_length
  else
    0

-- Theorem statement
theorem probability_area_equilateral_triangle_between_sqrt3_and_4sqrt3 :
  probability_of_area_in_range 5 AP_length = 2 / 5 :=
by
  -- Assumptions
  rw [AB_length_eq_5]
  -- Skipping the proof
  sorry

end probability_area_equilateral_triangle_between_sqrt3_and_4sqrt3_l346_346221


namespace find_number_l346_346070

theorem find_number (n : ℝ) (h : 3 / 5 * ((2 / 3 + 3 / 8) / n) - 1 / 16 = 0.24999999999999994) : n = 48 :=
  sorry

end find_number_l346_346070


namespace steve_total_time_on_roads_l346_346873

noncomputable def distance := 28 -- Distance in km
noncomputable def speed_back := 14 -- Speed back in km/h
noncomputable def speed_to := speed_back / 2 -- Speed to in km/h
noncomputable def time_to := distance / speed_to -- Time to work in hours
noncomputable def time_back := distance / speed_back -- Time back in hours
noncomputable def total_time := time_to + time_back -- Total time on roads each day

theorem steve_total_time_on_roads : total_time = 6 :=
by
  calc
    total_time = time_to + time_back := rfl
          ... = (distance / speed_to) + (distance / speed_back) := rfl
          ... = (28 / (14 / 2)) + (28 / 14) := by norm_num
          ... = 4 + 2 := by norm_num
          ... = 6 := by norm_num

end steve_total_time_on_roads_l346_346873


namespace problem_lean_statement_l346_346336

def g (n : ℕ) : ℕ :=
  let k := (Float.ofNat n) ** (1 / 4)
  (k + 0.5).floor

theorem problem_lean_statement : 
  ∑ k in Finset.range (4095 + 1), (1 / (g k)^2 : ℝ) = 817.4236 := 
  by sorry

end problem_lean_statement_l346_346336


namespace coefficient_of_x_in_expansion_l346_346786

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l346_346786


namespace path_area_cost_l346_346623

theorem path_area_cost (length_field width_field path_width cost_per_sq_m : ℝ) :
  length_field = 75 ∧ width_field = 55 ∧ path_width = 2.5 ∧ cost_per_sq_m = 2 →
  let 
      length_total := length_field + 2 * path_width,
      width_total := width_field + 2 * path_width,
      area_total := length_total * width_total,
      area_field := length_field * width_field,
      area_path := area_total - area_field,
      cost_path := area_path * cost_per_sq_m
  in
  area_path = 675 ∧ cost_path = 1350 :=
by
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end path_area_cost_l346_346623


namespace smallest_four_digit_in_pascal_l346_346512

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346512


namespace find_speed_of_first_train_l346_346055

variable (L1 L2 : ℝ) (V1 V2 : ℝ) (t : ℝ)

theorem find_speed_of_first_train (hL1 : L1 = 100) (hL2 : L2 = 200) (hV2 : V2 = 30) (ht: t = 14.998800095992321) :
  V1 = 42.005334224 := by
  -- Proof to be completed
  sorry

end find_speed_of_first_train_l346_346055


namespace smallest_four_digit_number_in_pascals_triangle_l346_346547

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346547


namespace Adeline_hourly_wage_l346_346144

theorem Adeline_hourly_wage
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 9) 
  (h2 : days_per_week = 5) 
  (h3 : weeks = 7) 
  (h4 : total_earnings = 3780) :
  total_earnings = 12 * (hours_per_day * days_per_week * weeks) :=
by
  sorry

end Adeline_hourly_wage_l346_346144


namespace parallelogram_with_right_angles_is_rectangle_l346_346600

-- Define a parallelogram structure
structure Parallelogram where
  -- This definition assumes the properties of a parallelogram
  a b c d : Prop

-- Define a Parallelogram with angles of 90 degrees
def ParallelogramWithRightAngles (p : Parallelogram) : Prop :=
  -- All angles are 90 degrees
  (∀ angle : ℝ, angle = 90)

-- Define what it means to be a Rectangle 
def IsRectangle (p : Parallelogram) : Prop :=
  -- A parallelogram with all right angles is a rectangle
  ParallelogramWithRightAngles p

-- The proof statement
theorem parallelogram_with_right_angles_is_rectangle (p : Parallelogram) (h : ParallelogramWithRightAngles p) : 
  IsRectangle p :=
sorry

end parallelogram_with_right_angles_is_rectangle_l346_346600


namespace probability_neither_event_l346_346088

theorem probability_neither_event (P_A P_B P_A_and_B : ℝ)
  (h1 : P_A = 0.25)
  (h2 : P_B = 0.40)
  (h3 : P_A_and_B = 0.20) :
  1 - (P_A + P_B - P_A_and_B) = 0.55 :=
by
  sorry

end probability_neither_event_l346_346088


namespace crazy_silly_school_movies_books_l346_346409

theorem crazy_silly_school_movies_books :
  ∀ (n_movies n_books : ℕ) (n_read_books n_watched_movies : ℕ),
    n_movies = 47 →
    n_books = 23 →
    n_read_books = 19 →
    n_watched_movies = 81 →
    n_movies - n_books = 24 :=
by
  intros n_movies n_books n_read_books n_watched_movies
  intros h_movies h_books h_read_books h_watched_movies
  rw [h_movies, h_books]
  exact rfl

end crazy_silly_school_movies_books_l346_346409


namespace K_time_for_distance_l346_346109

theorem K_time_for_distance (x : ℝ) : 
  ∀ (K M : ℝ), (K = x) ∧ (M = x - 1 / 3) ∧ (30 / M - 30 / K = 1 / 2) → (30 / K = 30 / x) :=
by
  intros K M h
  cases h with hK hRest
  cases hRest with hM hDiff
  rw [hK, hM] at hDiff
  unfold function.has_inv.inv (/) at hDiff
  sorry

end K_time_for_distance_l346_346109


namespace smallest_four_digit_in_pascals_triangle_l346_346426

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346426


namespace ella_incorrect_answers_l346_346343

theorem ella_incorrect_answers
  (marion_score : ℕ)
  (ella_score : ℕ)
  (total_items : ℕ)
  (h1 : marion_score = 24)
  (h2 : marion_score = (ella_score / 2) + 6)
  (h3 : total_items = 40) : 
  total_items - ella_score = 4 :=
by
  sorry

end ella_incorrect_answers_l346_346343


namespace coefficient_of_x_in_expansion_l346_346788

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l346_346788


namespace number_of_nonempty_proper_subsets_l346_346742

open Set

theorem number_of_nonempty_proper_subsets {α : Type*} (A : Set α) (hA : A = {1, 2, 3, 4}) : 
  ((2^4 - 2) = 14) :=
by
  sorry

end number_of_nonempty_proper_subsets_l346_346742


namespace number_of_triangles_in_decagon_l346_346982

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346982


namespace Tanika_boxes_l346_346016

/--
Tanika is selling boxes of crackers for her scout troop's fund-raiser.
On Saturday, she sold 60 boxes.
On Sunday, she sold 50% more than on Saturday.
Prove that the total number of boxes she sold, in total, over the two days is 150.
-/
theorem Tanika_boxes (sunday_more_than_saturday : real) (saturday_boxes : real) (sunday_percentage_increase : real)
    (h1 : saturday_boxes = 60) (h2 : sunday_percentage_increase = 0.5) 
    (h3 : sunday_more_than_saturday = saturday_boxes * sunday_percentage_increase) : 
    saturday_boxes + (saturday_boxes + sunday_more_than_saturday) = 150 :=
by
  sorry

end Tanika_boxes_l346_346016


namespace max_good_diagonals_l346_346776

def is_good_diagonal (n : ℕ) (d : ℕ) : Prop := ∀ (P : Fin n → Prop), ∃! (i j : Fin n), P i ∧ P j ∧ (d = i + j)

theorem max_good_diagonals (n : ℕ) (h : 2 ≤ n) :
  (∃ (m : ℕ), is_good_diagonal n m ∧ (m = n - 2 ↔ Even n) ∧ (m = n - 3 ↔ Odd n)) :=
by
  sorry

end max_good_diagonals_l346_346776


namespace smallest_four_digit_in_pascals_triangle_l346_346485

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346485


namespace smallest_four_digit_number_in_pascals_triangle_l346_346466

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346466


namespace part_I_part_II_l346_346734

theorem part_I (x : ℝ) : 
  (|x-1| + |x+1| - x - 2 > 0) ↔ (x < 0 ∨ x > 2) := sorry

theorem part_II (a x_0 : ℝ) (h1 : a > -1) (h2 : x_0 ∈ set.Ico (-a) 1) (h3 : (|x_0-1| + |x_0+a| - x_0 - 2) ≤ 0) :
  a < 2 := sorry

end part_I_part_II_l346_346734


namespace smallest_four_digit_in_pascals_triangle_l346_346534

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346534


namespace probability_first_greater_or_equal_second_l346_346125

-- Define the finite sample space for two rolls of a fair 8-sided die
def sample_space := (fin 8 × fin 8)

-- Define the event that the first number is greater than or equal to the second
def event_A (x : fin 8 × fin 8) : Prop := x.1 ≥ x.2

-- Define the probability measure for this event by counting the favorable outcomes
def probability_event_A : ℚ :=
  let favorable_outcomes := finset.univ.filter event_A
  let total_outcomes := finset.univ : finset sample_space
  (favorable_outcomes.card : ℚ) / (total_outcomes.card : ℚ)

-- Prove that the probability_event_A equals to 9/16
theorem probability_first_greater_or_equal_second : probability_event_A = 9 / 16 := by {
  -- Proof goes here
  sorry
}

end probability_first_greater_or_equal_second_l346_346125


namespace triangles_from_decagon_l346_346993

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346993


namespace sum_outside_layers_l346_346122

theorem sum_outside_layers (cube : ℕ → ℕ → ℕ → ℝ) (h1 : ∀ x, ∑ y in finset.range 20, cube x y 0 = 1)
  (h2 : ∀ y, ∑ x in finset.range 20, cube x y 0 = 1) 
  (h3 : ∀ z, ∑ x in finset.range 20, cube x 0 z = 1)
  (h4 : ∃ a b c, cube a b c = 10 ∧ a < 20 ∧ b < 20 ∧ c < 20): 
  (∑ x in finset.range 20, ∑ y in finset.range 20, ∑ z in finset.range 20, cube x y z) - 
  (∑ x in finset.range 20, ∑ y in finset.range 20, cube x y 0 + ∑ y in finset.range 20, ∑ z in finset.range 20, cube 0 y z + 
  ∑ x in finset.range 20, ∑ z in finset.range 20, cube x 0 z - 2 * ∥10∥) = 6820 := 
sorry

end sum_outside_layers_l346_346122


namespace gauss_family_mean_age_l346_346376

theorem gauss_family_mean_age :
  let ages := [8, 8, 8, 8, 16, 17]
  let num_children := 6
  let sum_ages := 65
  (sum_ages : ℚ) / (num_children : ℚ) = 65 / 6 :=
by
  sorry

end gauss_family_mean_age_l346_346376


namespace range_of_c_l346_346296

noncomputable def f (c x : ℝ) : ℝ := 2 * x^2 + c * x + Real.log x

theorem range_of_c (c : ℝ) : 
  (∀ x > 0, ∃ x' > 0, (f c x').deriv = 0 ∧ 
  ((f c x').deriv)' < 0) ↔ (c < -4) := 
by sorry

end range_of_c_l346_346296


namespace smallest_four_digit_in_pascal_l346_346515

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346515


namespace total_money_l346_346578

theorem total_money (p q r : ℕ)
  (h1 : r = 2000)
  (h2 : r = (2 / 3) * (p + q)) : 
  p + q + r = 5000 :=
by
  sorry

end total_money_l346_346578


namespace triangles_congruent_angle_bpc_l346_346650

-- Define the setup
variable (A B C D E F G P : Type)

-- Given conditions
axiom h1 : BA = EA (sides of square ABDE)
axiom h2 : AG = AC (sides of square ACFG)
axiom h3 : ∠BAG = 90 + ∠BAC
axiom h4 : ∠EAC = 90 + ∠BAC
axiom h_intersection : ∃ P, Line BG ∩ Line EC = P

-- Statements to prove
theorem triangles_congruent : ∀ (A B C D E F G : Type),
  BA = EA → AG = AC →  (∠BAG = 90 + ∠BAC) → ∠BAG = ∠EAC → 
  Congruent (Triangle BAG) (Triangle EAC) :=
by 
  intro A B C D E F G,
  intro h1 h2 h3 h4,
  sorry

theorem angle_bpc : ∀ (A B C D E F G P : Type),
  Congruent (Triangle BAG) (Triangle EAC) →
  h_intersection →
  ∠BPC = 90 :=
by
  intro A B C D E F G P,
  intro triangle_congruent h_intersection,
  sorry


end triangles_congruent_angle_bpc_l346_346650


namespace remainder_when_divided_by_x_add_1_l346_346072

def q (x : ℝ) : ℝ := 2 * x^4 - 3 * x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_when_divided_by_x_add_1 :
  q 2 = 6 → q (-1) = 20 :=
by
  intro hq2
  sorry

end remainder_when_divided_by_x_add_1_l346_346072


namespace smallest_four_digit_number_in_pascals_triangle_l346_346455

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346455


namespace calculate_product_value_l346_346157

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l346_346157


namespace pentagon_dimension_l346_346375

theorem pentagon_dimension (ABCD : rectangle) (h1 : ABCD.length = 20) (h2 : ABCD.width = 10)
  (h3 : ∃ (P Q : pentagon), congruent P Q ∧ (ABCD.cut P Q) ∧ (P ∪ Q = square s))
  (h4 : ∀y, 3 * y = ABCD.width) : y = 20 / 3 :=
by
  sorry

end pentagon_dimension_l346_346375


namespace triangles_from_decagon_l346_346997

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346997


namespace trapezoid_area_l346_346948

variable (x : ℝ)

def base1 := 5 * x
def base2 := 8 * x
def height := x

theorem trapezoid_area :
  (1 / 2) * (base1 + base2) * height = (13 * x^2) / 2 := by
  sorry

end trapezoid_area_l346_346948


namespace smallest_four_digit_in_pascal_triangle_l346_346561

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346561


namespace smallest_four_digit_in_pascal_l346_346499

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346499


namespace boat_speed_in_still_water_l346_346591

/-- Prove that the speed of the boat in still water is 15 km/h given the conditions. -/
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, ∀ D : ℝ, (1 * (V_b + 3) = 1.5 * (V_b - 3)) → V_b = 15 :=
begin
  sorry
end

end boat_speed_in_still_water_l346_346591


namespace smallest_four_digit_in_pascals_triangle_l346_346484

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346484


namespace simplify_expr1_simplify_expr2_l346_346008

variable (a b x y : ℝ)

theorem simplify_expr1 : 6 * a + 7 * b^2 - 9 + 4 * a - b^2 + 6 = 10 * a + 6 * b^2 - 3 :=
by
  sorry

theorem simplify_expr2 : 5 * x - 2 * (4 * x + 5 * y) + 3 * (3 * x - 4 * y) = 6 * x - 22 * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l346_346008


namespace question_1_question_2_l346_346261

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.log x
noncomputable def h (x : ℝ) : ℝ := g x - f x (-1)
noncomputable def phi (x : ℝ) : ℝ := Real.log x + 1 / x - 1

theorem question_1 (a : ℝ) (h_neg_a : a < 0) : 
  (∀ x : ℝ, f x a > 0) ↔ a ∈ Ioo (-real.exp 1) 0 := sorry

theorem question_2 : ∃! x : ℝ, x ∈ set.Ioi 0 ∧ h' x = 1 := sorry

end question_1_question_2_l346_346261


namespace gayle_rainy_time_l346_346212

-- Define the given speeds in miles per minute
def sunny_speed : ℝ := 40 / 60
def rainy_speed : ℝ := 25 / 60

-- Define the total time in minutes and total distance in miles
def total_time : ℝ := 50
def total_distance : ℝ := 20

-- State the theorem to prove the rainy time is 32 minutes
theorem gayle_rainy_time : ∃ t_r : ℝ, sunny_speed * (total_time - t_r) + rainy_speed * t_r = total_distance ∧ t_r = 32 :=
by
  -- Provide the computed answer
  existsi 32
  split
  have h : sunny_speed = 2 / 3 := by norm_num
  rw [h, ←rat.cast_coe_nat, ←rat.cast_coe_nat],
  change (2/3 : ℝ) * (50 - 32) + (25 / 60) * 32 = 20,
  norm_num,
  linarith,
  -- Conclude the proof. The proof steps (solution steps) were provided as per conditions.
  sorry

end gayle_rainy_time_l346_346212


namespace right_triangle_medians_length_l346_346625

section
variables {A B C : ℝ}
variables (M N : ℝ)
variables (AC BC AB : ℝ)
variables (AM : ℝ := 5)
variables (BN : ℝ := 3 * Real.sqrt 3)
variables (ratio_AC_BC : AC / BC = 3 / 2)

-- The problem statement: Given \(AM = 5\), \(BN = 3\sqrt{3} \), and \( AC : BC = 3 : 2 \),
-- prove that \( AB = 2\sqrt{13} \).
theorem right_triangle_medians_length
  (hAC : AC = 3 * sqrt 13 / 2)
  (hBC : BC = 2 * sqrt 13 / 2)
  (hAmTA : (AC^2 + (1/2 * BC)^2) = 25)
  (hBnMN: (BC^2 + (1/2 * AC)^2) = 27):
  AB = 2 * sqrt 13 :=
sorry

end

end right_triangle_medians_length_l346_346625


namespace probability_of_two_boys_three_girls_l346_346904

noncomputable def probability_family_five_children_two_boys_three_girls 
  (n k : ℕ) (p : ℝ) (h : n = 5) (h1 : k = 2) (h2 : p = 0.5) : ℝ :=
  if h ∧ h1 ∧ h2 then (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) else 0

theorem probability_of_two_boys_three_girls : probability_family_five_children_two_boys_three_girls 5 2 0.5 5 rfl 2 rfl 0.5 rfl = 0.3125 := 
by {  sorry }

end probability_of_two_boys_three_girls_l346_346904


namespace product_evaluation_l346_346160

theorem product_evaluation :
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by sorry

end product_evaluation_l346_346160


namespace find_a_l346_346730

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  a * x - Real.log (x + 1)

noncomputable def curve_derivative_at (a : ℝ) (x : ℝ) :=
  D (λ x : ℝ, curve a x) x

theorem find_a (a : ℝ) :
  (curve_derivative_at a 0 = 2) → a = 3 :=
by
  intro h
  -- Proof omitted
  sorry

end find_a_l346_346730


namespace second_sum_correct_l346_346946

theorem second_sum_correct:
  let total_sum := 2795
  let interest_rate_first_part := 3
  let interest_rate_second_part := 5
  let time_first_part := 8
  let time_second_part := 3
  let x := 1075 -- first part of the sum
  let second_part := total_sum - x
  in (x * interest_rate_first_part * time_first_part) / 100 =
     (second_part * interest_rate_second_part * time_second_part) / 100 ->
  second_part = 1720 :=
by
  sorry

end second_sum_correct_l346_346946


namespace minimum_resistors_required_l346_346964

-- Define the grid configuration and the connectivity condition
def isReliableGrid (m : ℕ) (n : ℕ) (failures : Finset (ℕ × ℕ)) : Prop :=
m * n > 9 ∧ (∀ (a b : ℕ), a ≠ b → (a, b) ∉ failures)

-- Minimum number of resistors ensuring connectivity with up to 9 failures
theorem minimum_resistors_required :
  ∃ (m n : ℕ), 5 * 5 = 25 ∧ isReliableGrid 5 5 ∅ :=
by
  let m : ℕ := 5
  let n : ℕ := 5
  have h₁ : m * n = 25 := by rfl
  have h₂ : isReliableGrid 5 5 ∅ := by
    unfold isReliableGrid
    exact ⟨by norm_num, sorry⟩ -- formal proof omitted for brevity
  exact ⟨m, n, h₁, h₂⟩

end minimum_resistors_required_l346_346964


namespace least_positive_integer_with_seven_distinct_factors_l346_346062

theorem least_positive_integer_with_seven_distinct_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ 1 → (∀ k : ℕ, k ∣ n → k ∣ m → k ∣ m)) → n = 64) ∧
    (∃ p : ℕ, (nat.prime p ∧ n = p^6)) :=
begin
  sorry
end

end least_positive_integer_with_seven_distinct_factors_l346_346062


namespace smallest_pos_int_for_congruence_l346_346564

theorem smallest_pos_int_for_congruence :
  ∃ (n : ℕ), 5 * n % 33 = 980 % 33 ∧ n > 0 ∧ n = 19 := 
by {
  sorry
}

end smallest_pos_int_for_congruence_l346_346564


namespace min_marked_numbers_l346_346840

theorem min_marked_numbers (S : set ℕ) (N : ℕ) :
  (\{1, 2, ... , 2000\} ⊆ S ∧
   ∀ k : ℕ, 1 ≤ k ∧ k ≤ 1000 → (2 * k - 1 ∈ S ∨ 2 * k ∈ S)) →
   N = 666 :=
sorry

end min_marked_numbers_l346_346840


namespace smallest_four_digit_number_in_pascals_triangle_l346_346448

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346448


namespace balls_removed_by_each_of_other_two_students_l346_346120

-- Definitions based on conditions
def tennis_balls_per_basket := 15
def soccer_balls_per_basket := 5
def number_of_baskets := 5
def balls_removed_by_three_students := 3 * 8
def balls_remaining_in_baskets := 56

-- Proof problem statement
theorem balls_removed_by_each_of_other_two_students :
  let total_balls := (tennis_balls_per_basket + soccer_balls_per_basket) * number_of_baskets in
  let balls_after_three_students_removed := total_balls - balls_removed_by_three_students in
  let balls_removed_by_other_two_students := balls_after_three_students_removed - balls_remaining_in_baskets in
  balls_removed_by_other_two_students / 2 = 10 :=
by
  let total_balls := (tennis_balls_per_basket + soccer_balls_per_basket) * number_of_baskets
  let balls_after_three_students_removed := total_balls - balls_removed_by_three_students
  let balls_removed_by_other_two_students := balls_after_three_students_removed - balls_remaining_in_baskets
  show balls_removed_by_other_two_students / 2 = 10 from
    sorry -- proof steps are not required

end balls_removed_by_each_of_other_two_students_l346_346120


namespace parameterization_theorem_l346_346392

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem_l346_346392


namespace smallest_four_digit_number_in_pascals_triangle_l346_346461

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346461


namespace ratio_AP_PQ_l346_346712

-- Given definitions and conditions
variables {A B C D E F M P Q : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space M] [metric_space P] [metric_space Q]
variable isosceles_triangle : triangle A B C
variable midpoint_M : midpoint M B C
variable semicircle_center_M : semicircle M A B
variable tangent_EF : tangent_line E F D M
variable intersection_P : intersection P E F
variable perp_PQ : perpendicular PQ B C Q

-- The required statement to prove the ratio.
theorem ratio_AP_PQ (AP PQ AB BC : ℝ) : AP / PQ = 2 * AB / BC :=
sorry

end ratio_AP_PQ_l346_346712


namespace part1_solution_count_part2_solution_count_l346_346110

theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card = 7 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = 2 * (m + n + r) := sorry

theorem part2_solution_count (k : ℕ) (h : 1 < k) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card ≥ 3 * k + 1 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = k * (m + n + r) := sorry

end part1_solution_count_part2_solution_count_l346_346110


namespace smallest_four_digit_in_pascal_l346_346522

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346522


namespace area_percent_difference_l346_346921

theorem area_percent_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  percent_difference = 4 := 
by
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  sorry

end area_percent_difference_l346_346921


namespace rightmost_three_digits_of_7_pow_1994_l346_346058

theorem rightmost_three_digits_of_7_pow_1994 :
  (7 ^ 1994) % 800 = 49 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1994_l346_346058


namespace lawn_area_lawn_cost_l346_346020

-- Define the conditions as constants
constant r_base : ℝ := 15
constant w_lawn : ℝ := 5
constant cost_per_sqm : ℝ := 500

-- Define the radii of the smaller and larger circles
def r_large := r_base + w_lawn

-- Define the areas of the circles
def area_circle (r : ℝ) := 3.14 * r * r

-- Define the areas of the small and large circles
def area_small_circle := area_circle r_base
def area_large_circle := area_circle r_large

-- The first proposition: Area of the lawn is 549.5 square meters
theorem lawn_area : area_large_circle - area_small_circle = 549.5 :=
by
  sorry

-- The second proposition: Cost to plant the lawn is 274750 yuan
theorem lawn_cost : cost_per_sqm * (area_large_circle - area_small_circle) = 274750 :=
by
  sorry

end lawn_area_lawn_cost_l346_346020


namespace pages_of_shorter_book_is_10_l346_346884

theorem pages_of_shorter_book_is_10
  (x : ℕ) 
  (h_diff : ∀ (y : ℕ), x = y - 10)
  (h_divide : (x + 10) / 2 = x) 
  : x = 10 :=
by
  sorry

end pages_of_shorter_book_is_10_l346_346884


namespace find_x_value_l346_346818

theorem find_x_value (x y z : ℕ) (h1 : x ≥ y) (h2 : y ≥ z) 
(h3 : x^2 - y^2 - z^2 + x * y = 1005)
(h4 : x^2 + 2 * y^2 + 2 * z^2 - 2 * x * y - x * z - y * z = -995)
(h5 : x % 2 = 0) : x = 505 := by
  sorry

end find_x_value_l346_346818


namespace system_correct_l346_346863

def correct_system (x y : ℝ) : Prop :=
  5 * x + 2 * y = 19 ∧ 2 * x + 3 * y = 12

theorem system_correct (x y : ℝ) (h1 : 5 * x + 2 * y = 19) (h2 : 2 * x + 3 * y = 12) :
  correct_system x y :=
by {
  split,
  exact h1,
  exact h2,
}

end system_correct_l346_346863


namespace triangles_from_decagon_l346_346994

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346994


namespace math_problem_l346_346827

noncomputable def f (n : ℕ) : ℝ → ℝ
| 0       => sin
| (n + 1) => λ x => (derivative (f n)) x

def problem (x : ℝ) : ℝ :=
  (List.sum $ List.map (λ n => f n 15) (List.range 2017))

theorem math_problem :
  problem 15 = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
sorry

end math_problem_l346_346827


namespace find_an_value_l346_346641

-- Define the variables and conditions using Lean
variables (a : ℕ)
def Brian := 2 * a
def Caden := a + Brian
def Daryl := 2 * Caden
axiom total_marbles : a + Brian + Caden + Daryl = 144

-- The statement we need to prove
theorem find_an_value : a = 12 :=
by
  -- Proof steps will be filled in here
  sorry

end find_an_value_l346_346641


namespace tv_show_duration_l346_346348

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end tv_show_duration_l346_346348


namespace positionXiaoWen_determineA_pointDifference_l346_346573

-- Definitions for the movements done by Xiao Wen and Xiao Li over the five games
def movementsXiaoWen : List ℤ := [+3, -2, -1, +4, -5]
def movementsXiaoLi (a : ℤ) : List ℤ := [-4, +3, +5, -2, a]

-- Final position of Xiao Wen after five games
def finalPositionXiaoWen : ℤ := movementsXiaoWen.sum

-- Value of 'a' that makes Xiao Li end 5 centimeters east of point A
def requiredValueOfA : ℤ := 3

-- Final position of Xiao Li after five games, assuming given 'a'
def finalPositionXiaoLi (a : ℤ) : ℤ := (movementsXiaoLi a).sum

-- Scoring mechanism
def scoreXiaoWen : ℤ := (7 * 3) - (8 * 2)
def scoreXiaoLi : ℤ := (11 * 3) - (6 * 2)
def scoreDifference : ℤ := scoreXiaoLi - scoreXiaoWen

-- Theorems to be proved
theorem positionXiaoWen : finalPositionXiaoWen = -1 := by
  sorry

theorem determineA (a : ℤ) : finalPositionXiaoLi a = 5 → a = 3 := by
  sorry

theorem pointDifference : scoreDifference = 16 := by
  sorry

end positionXiaoWen_determineA_pointDifference_l346_346573


namespace only_sphere_and_cylinder_have_circular_cross_section_l346_346938

/-
  Prove that a plane cutting through a given geometric shape 
  results in a circular cross-section exactly if the shape is a sphere or a cylinder.
-/

inductive GeometricShape
| cube : GeometricShape
| sphere : GeometricShape
| cylinder : GeometricShape
| pentagonal_prism : GeometricShape

def has_circular_cross_section (shape : GeometricShape) : Prop :=
match shape with
| GeometricShape.sphere => true
| GeometricShape.cylinder => true
| _ => false
end

theorem only_sphere_and_cylinder_have_circular_cross_section :
  ∀ (shape : GeometricShape), has_circular_cross_section(shape) ↔ shape = GeometricShape.sphere ∨ shape = GeometricShape.cylinder := by
  intros shape
  cases shape
  case cube => simp [has_circular_cross_section]
  case sphere => simp [has_circular_cross_section]
  case cylinder => simp [has_circular_cross_section]
  case pentagonal_prism => simp [has_circular_cross_section]
  sorry

end only_sphere_and_cylinder_have_circular_cross_section_l346_346938


namespace inequality_sigma_l346_346329

noncomputable def sigma (x : ℕ → ℝ) (k n : ℕ) : ℝ :=
  if k = 0 then 1 else
    ∑ s in (Finset.powersetLen k (Finset.range n)).filter (λ t, t.card = k), 
    (s : Finset ℕ).prod (λ i, x i)

theorem inequality_sigma (x : ℕ → ℝ) (n : ℕ) :
  (∏ k in Finset.range n, (x k) ^ 2 + 1) 
  ≥ 
  2 * abs (∑ k in Finset.range ((n + 1) / 2), (-1) ^ k * sigma x (2 * k) n) 
   * abs (∑ k in Finset.range (n / 2), (-1) ^ k * sigma x (2 * k + 1) n) :=
by 
  sorry

end inequality_sigma_l346_346329


namespace average_age_is_26_l346_346190

noncomputable def devin_age : ℕ := 12
noncomputable def eden_age : ℕ := 2 * devin_age
noncomputable def eden_mom_age : ℕ := 2 * eden_age
noncomputable def eden_grandfather_age : ℕ := (devin_age + eden_age + eden_mom_age) / 2
noncomputable def eden_aunt_age : ℕ := eden_mom_age / devin_age

theorem average_age_is_26 : 
  (devin_age + eden_age + eden_mom_age + eden_grandfather_age + eden_aunt_age) / 5 = 26 :=
by {
  sorry
}

end average_age_is_26_l346_346190


namespace ratio_of_A_to_B_l346_346003

theorem ratio_of_A_to_B (A B C : ℝ) (hB : B = 270) (hBC : B = (1 / 4) * C) (hSum : A + B + C = 1440) : A / B = 1 / 3 :=
by
  -- The proof is omitted for this example
  sorry

end ratio_of_A_to_B_l346_346003


namespace avg_marks_correct_l346_346378

-- Define the average marks of the first class
def avg_marks_class1 : ℝ := 40

-- Define the number of students in the first class
def num_students_class1 : ℕ := 20

-- Define the average marks of the second class
def avg_marks_class2 : ℝ := 60

-- Define the number of students in the second class
def num_students_class2 : ℕ := 50

-- Define the total marks calculation for the first class
def total_marks_class1 : ℝ := avg_marks_class1 * num_students_class1

-- Define the total marks calculation for the second class
def total_marks_class2 : ℝ := avg_marks_class2 * num_students_class2

-- Define the total marks of both classes
def total_marks : ℝ := total_marks_class1 + total_marks_class2

-- Define the total number of students in both classes
def total_students : ℕ := num_students_class1 + num_students_class2

-- Define the average marks for all students
def avg_marks_all_students : ℝ := total_marks / total_students

-- The main theorem stating that the average marks of all students is 54.29
theorem avg_marks_correct : avg_marks_all_students = 54.29 := by
  -- Proof is omitted
  sorry

end avg_marks_correct_l346_346378


namespace triangle_obtuse_angles_contradiction_l346_346910

theorem triangle_obtuse_angles_contradiction (T : Type) [triangle T] : 
  (∃ A B C : T, obtuse A ∧ obtuse B ∧ obtuse C) → ¬ (∃ A B C : T, obtuse A ∧ obtuse B) :=
by
  sorry

end triangle_obtuse_angles_contradiction_l346_346910


namespace math_problem_l346_346405

variable {p q : ℝ}

theorem math_problem
  (hM : ∀ x, x^2 - p * x + 6 = 0 ↔ x ∈ ({2} : set ℝ))
  (hN : ∀ x, x^2 + 6 * x - q = 0 ↔ x ∈ ({2} : set ℝ))
  (hMN : ({2} : set ℝ) = {2}) :
  p + q = 21 :=
sorry

end math_problem_l346_346405


namespace triangle_angles_l346_346925

open Real

theorem triangle_angles
  (A B C D E : Point)
  (incircle : Circle)
  (incircle_touches_AB_at_D : incircle.Touches AB at D)
  (incircle_touches_BC_at_E : incircle.Touches BC at E)
  (AD DB BE EC : ℝ)
  (h1 : AD / DB = 2)
  (h2 : BE / EC = 1 / 3) :
  ∃ (α β γ : ℝ), α = 90 ∧ β = arccos (4 / 5) ∧ γ = arcsin (4 / 5) :=
by
  sorry

end triangle_angles_l346_346925


namespace dive_point_value_l346_346914

noncomputable def dive_scores := [7.5, 7.8, 9.0, 6.0, 8.5]
noncomputable def degree_of_difficulty := 3.2

theorem dive_point_value : 
  (let sorted_scores := dive_scores.qsort (≤)
   in let truncated_scores := (sorted_scores.drop 1).take (sorted_scores.length - 2)
   in truncated_scores.sum * degree_of_difficulty) = 76.16 := by
  sorry

end dive_point_value_l346_346914


namespace volume_of_stone_l346_346123

theorem volume_of_stone 
  (width length initial_height final_height : ℕ)
  (h_width : width = 15)
  (h_length : length = 20)
  (h_initial_height : initial_height = 10)
  (h_final_height : final_height = 15)
  : (width * length * final_height - width * length * initial_height = 1500) :=
by
  sorry

end volume_of_stone_l346_346123


namespace smallest_four_digit_number_in_pascals_triangle_l346_346444

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346444


namespace find_phi_l346_346738

open Real

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := cos (2 * x - π/2 + φ)

theorem find_phi 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (symmetry_condition : ∀ x, g (π/2 - x) φ = g (π/2 + x) φ) 
  : φ = π / 2 
:= by 
  sorry

end find_phi_l346_346738


namespace exists_a_l346_346801

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + (5 / 8) * a - (3 / 2)

theorem exists_a : ∃ a : ℝ, (max_value_on_interval (func a) 0 (π / 2) = 1) ∧ (a = 3 / 2) :=
sorry

end exists_a_l346_346801


namespace number_of_negatives_l346_346751

def count_negatives : Nat :=
  (Finset.univ.filter (λ (a : Fin 3 × Fin 3 × Fin 3 × Fin 3 × Fin 3 × Fin 3), 
    (a.prod.fst : Int) * 5 + (a.prod.snd.fst : Int) * 5^2 + (a.prod.snd.snd.fst : Int) * 5^3 + 
    (a.prod.snd.snd.snd.fst : Int) * 5^4 + (a.prod.snd.snd.snd.snd.fst : Int) * 5^5 + 
    (a.prod.snd.snd.snd.snd.snd : Int) * 5^6 < 0)).card

theorem number_of_negatives : count_negatives = 364 :=
by
  sorry

end number_of_negatives_l346_346751


namespace trig_equation_integer_solutions_l346_346354

theorem trig_equation_integer_solutions (x : ℝ):
  (2 * real.sqrt 2 * real.cos (real.pi / 180 * 25) - 1) * real.tan (real.pi / 180 * x) =
  (2 * real.sqrt 2 * real.sin (real.pi / 180 * 25) - 1) * real.tan (real.pi / 180 * 3 * x) →
  ∃ k ∈ ℤ, x = 180 * k ∨ x = 180 * k + 25 ∨ x = 180 * k - 25 :=
begin
  sorry
end

end trig_equation_integer_solutions_l346_346354


namespace cone_volume_divided_by_pi_l346_346926

-- Given Conditions
def r := 40 / 3
def r_squared_plus_h_squared := r ^ 2 + h ^ 2 = 20 ^ 2

-- The volume of the cone V
def cone_volume (r h : ℝ) : ℝ := (1/3) * real.pi * r^2 * h

-- The theorem to prove
theorem cone_volume_divided_by_pi (h : ℝ) (r h_sq : ℝ) (r_h_relation : r^2 + h^2 = 20^2) : 
(cone_volume r h) / real.pi = 32000 * real.sqrt 10 / 27 :=
by 
  -- Definitions already used in the proof steps
  have r := 40 / 3,
  have h := 20 * real.sqrt 10 / 3,
  sorry

end cone_volume_divided_by_pi_l346_346926


namespace length_AB_l346_346249

noncomputable def ellipse := {x y : ℝ // (x^2 / 2) + y^2 = 1}
noncomputable def circle := {x y : ℝ // x^2 + y^2 = 4}
def F1 : ℝ × ℝ := (-1, 0)
def line_l (x : ℝ) : ℝ := x + 1
def distance (p1 p2 : ℝ × ℝ) : ℝ := (real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2))

theorem length_AB :
  (let A B : ℝ × ℝ := sorry in  -- Points of intersection (to be found)
   distance A B = real.sqrt 14) :=
sorry


end length_AB_l346_346249


namespace basketball_team_score_l346_346589

theorem basketball_team_score (n : ℕ) (h_n : n = 12) (min_points max_points : ℕ) (h_min : min_points = 7) (h_max : max_points = 23) 
  (scores : Fin n → ℕ) (h_scores_min : ∀ i, scores i ≥ min_points) (h_scores_max : ∀ i, scores i ≤ max_points) : 
  (∑ i, scores i) ≥ n * min_points := 
by
  sorry

end basketball_team_score_l346_346589


namespace find_added_value_l346_346323

theorem find_added_value
  (n : ℕ) (x : ℝ)
  (h1 : n = 15)
  (h2 : ∀ numbers : Fin n → ℝ, (∑ i, numbers i) / n = 40)
  (h3 : ∀ numbers : Fin n → ℝ, (∑ i, numbers i + x) / n = 50) :
  x = 10 := 
sorry

end find_added_value_l346_346323


namespace hyperbola_asymptote_value_l346_346739

theorem hyperbola_asymptote_value {b : ℝ} (h : b > 0) 
  (asymptote_eq : ∀ x : ℝ, y = x * (1 / 2) ∨ y = -x * (1 / 2)) :
  b = 1 :=
sorry

end hyperbola_asymptote_value_l346_346739


namespace transformed_sine_function_l346_346415

theorem transformed_sine_function :
  ∀ (x : ℝ), 
  let f := λ x, sin (x + π / 6) in
  let g := λ x, sin (2 * x + π / 6) in
  let h := λ x, sin (2 * (x + π / 4) + π / 6) in
  h x = sin (2 * x + 2 * π / 3) :=
by
  -- Proof steps would be here, but we'll use sorry to skip the proof
  sorry

end transformed_sine_function_l346_346415


namespace common_integer_solutions_l346_346750

theorem common_integer_solutions
    (y : ℤ)
    (h1 : -4 * y ≥ 2 * y + 10)
    (h2 : -3 * y ≤ 15)
    (h3 : -5 * y ≥ 3 * y + 24)
    (h4 : y ≤ -1) :
  y = -3 ∨ y = -4 ∨ y = -5 :=
by 
  sorry

end common_integer_solutions_l346_346750


namespace smallest_four_digit_in_pascals_triangle_l346_346428

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346428


namespace smallest_four_digit_in_pascals_triangle_l346_346424

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346424


namespace parabola_standard_equation_l346_346295

theorem parabola_standard_equation
  (focus : ℝ × ℝ)
  (h : focus = (-2, 0)) :
  ∃ p : ℝ, y^2 = -2 * p * x ∧ p = 4 :=
by
  -- Defining the conditions from the problem statement
  let focus := (-2, 0)
  assume (h: (-2 : ℝ, 0) = (-2, 0))
  
  -- Using the conditions to show the result
  sorry

end parabola_standard_equation_l346_346295


namespace knight_tour_impossible_l346_346101

-- Define types for the problem
inductive Color
| black | white

-- Define the function to get the color of a square
def square_color : ℕ × ℕ → Color
| (x, y) := if (x + y) % 2 = 0 then Color.black else Color.white

-- Define the knight's tour conditions and question
theorem knight_tour_impossible :
  square_color (0, 0) = Color.black ∧ square_color (7, 7) = Color.black →
  ¬∃ (tour : Fin 64 → ℕ × ℕ), (tour 0 = (0, 0)) ∧ (tour 63 = (7, 7)) ∧
  (∀ i : Fin 63, let (x1, y1) := tour i, (x2, y2) := tour (i + 1) in
    ((abs (x2 - x1) = 2 ∧ abs (y2 - y1) = 1) ∨ (abs (x2 - x1) = 1 ∧ abs (y2 - y1) = 2))
  ) :=
begin
  intros H,
  sorry
end

end knight_tour_impossible_l346_346101


namespace smallest_four_digit_number_in_pascals_triangle_l346_346458

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346458


namespace value_of_f_l346_346076

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l346_346076


namespace remainder_when_divided_by_39_l346_346116

theorem remainder_when_divided_by_39 (N k : ℤ) (h : N = 13 * k + 4) : (N % 39) = 4 :=
sorry

end remainder_when_divided_by_39_l346_346116


namespace evaluate_expression_l346_346161

theorem evaluate_expression :
  (2^3 + 2^2 + 2^1 + 2^0) / (2^{-1} + 2^{-2} + 2^{-3} + 2^{-4}) = 16 :=
by
  sorry

end evaluate_expression_l346_346161


namespace proof_solve_for_xy_l346_346829

noncomputable def solve_for_xy : ℝ × ℝ :=
  let x := 10
  let y := 12.5 / 0.30
  (x, y)

theorem proof_solve_for_xy :
  let (x, y) := solve_for_xy 
  in (0.50 * x = 0.05 * 500 - 20) ∧ (0.30 * y = 0.25 * x + 10) :=
by
  let (x, y) := solve_for_xy
  have h1 : 0.50 * x = 0.05 * 500 - 20
  { sorry }
  have h2 : 0.30 * y = 0.25 * x + 10
  { sorry }
  exact ⟨h1, h2⟩

end proof_solve_for_xy_l346_346829


namespace geometric_inequality_l346_346819

theorem geometric_inequality 
  (A B C P A_0 B_0 C_0 : ℝ) -- A, B, C, P, A_0, B_0, C_0 are points in 2D (represented by coordinates)
  (hA0 : perpendicular A_0 P B C) -- A_0 is the perpendicular projection of P onto BC
  (hB0 : perpendicular B_0 P C A) -- B_0 is the perpendicular projection of P onto CA
  (hC0 : perpendicular C_0 P A B) -- C_0 is the perpendicular projection of P onto AB
  (PA PB PC : ℝ) (PA_0 PB_0 PC_0 : ℝ)  -- lengths of relevant segments
  (hPA : PA = dist P A)
  (hPB : PB = dist P B)
  (hPC : PC = dist P C)
  (hPA0 : PA_0 = dist P A_0)
  (hPB0 : PB_0 = dist P B_0)
  (hPC0 : PC_0 = dist P C_0) :
  PA * PB * PC ≥ (PA_0 + PB_0) * (PB_0 + PC_0) * (PC_0 + PA_0) :=
sorry

end geometric_inequality_l346_346819


namespace min_value_y_l346_346240

theorem min_value_y (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ y, y = (1 / (sin θ)^2) + (9 / (cos θ)^2) ∧ y = 16 :=
begin
  sorry
end

end min_value_y_l346_346240


namespace least_positive_integer_with_seven_factors_l346_346066

theorem least_positive_integer_with_seven_factors : ∃ n : ℕ, n = 64 ∧ 
  ∃ a p : ℕ, a = 6 ∧ (nat.prime p) ∧ n = p^a :=
by
  sorry

end least_positive_integer_with_seven_factors_l346_346066


namespace number_of_pupils_l346_346083

theorem number_of_pupils 
  (n : ℕ) 
  (mark_wrongly_entered : 73) 
  (actual_mark : 63) 
  (average_increased_by : ℚ) :
  (mark_wrongly_entered - actual_mark) / n = average_increased_by → 
  average_increased_by = 0.5 →
  n = 20 := 
by 
  intros h1 h2 
  sorry

end number_of_pupils_l346_346083


namespace watermelon_price_in_units_of_1000_l346_346289

theorem watermelon_price_in_units_of_1000
  (initial_price discounted_price: ℝ)
  (h_price: initial_price = 5000)
  (h_discount: discounted_price = initial_price - 200) :
  discounted_price / 1000 = 4.8 :=
by
  sorry

end watermelon_price_in_units_of_1000_l346_346289


namespace find_a_plus_h_l346_346386

noncomputable def hyperbola_equation (x y h k a b : ℝ) : Prop :=
(y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

noncomputable def is_hyperbola_passing_through_point (x₀ y₀ h k a b : ℝ) : Prop :=
hyperbola_equation x₀ y₀ h k a b

theorem find_a_plus_h (x₀ y₀ : ℝ) 
  (asymptote1 asymptote2 : ℝ → ℝ) 
  (hx₀ : x₀ = 1) 
  (hy₀ : y₀ = 8)
  (hasym1 : asymptote1 = (λ x, 3 * x + 6)) 
  (hasym2 : asymptote2 = (λ x, -3 * x + 2)) : 
  ∃ (a b h k : ℝ) (a_pos : a > 0) (b_pos : b > 0), 
    (hyperbola_equation x₀ y₀ h k a b ∧
    h = -2 / 3 ∧
    k = 4 ∧
    a = √(119) / 3 ∧
    b = √(119) / 9) → 
    a + h = (√(119) - 2) / 3 :=
sorry

end find_a_plus_h_l346_346386


namespace f_neg_x_l346_346727

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then sqrt x + 1 else if x < 0 then -sqrt (-x) - 1 else 0

theorem f_neg_x (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f(x)) (h_positive : ∀ x, x > 0 → f(x) = sqrt x + 1) :
  ∀ x, x < 0 → f(x) = -sqrt (-x) - 1 :=
by
  assume x hx
  have : f (-x) = sqrt (-x) + 1 := h_positive (-x) (by linarith)
  have : f (x) = -f (-x) := h_odd (-x)
  rw this
  rw this at h_odd
  linarith

end f_neg_x_l346_346727


namespace collinear_B_H_M_l346_346745

theorem collinear_B_H_M 
  (A B C L M H K : Type)
  [IncidenceGeometry A B C L M H K]
  (triangle_ABC : Triangle A B C)
  (angle_BAC : Angle A B C)
  (angle_BCA : Angle A C B)
  (angle_condition : measure angle_BAC = 2 * measure angle_BCA)
  (point_L_on_BC : OnLine L (line B C))
  (angle_BAL : Angle B A L)
  (angle_CAL : Angle C A L)
  (equal_angles : measure angle_BAL = measure angle_CAL)
  (midpoint_M : Midpoint M A C)
  (point_H_on_AL : OnSegment H A L)
  (perpendicular_MH_AL : Perpendicular (line M H) (line A L))
  (point_K_on_BC : OnLine K (line B C))
  (triangle_KMH : EquilateralTriangle K M H) :
  Collinear B H M :=
sorry

end collinear_B_H_M_l346_346745


namespace smallest_four_digit_number_in_pascals_triangle_l346_346550

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346550


namespace smallest_four_digit_in_pascal_l346_346497

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346497


namespace equal_roots_B_value_l346_346638

theorem equal_roots_B_value (B : ℝ) :
  (∀ k : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (k = 1 → (B^2 - 4 * (2 * 1) * 2 = 0))) → B = 4 ∨ B = -4 :=
by
  sorry

end equal_roots_B_value_l346_346638


namespace only_positive_integer_cube_less_than_triple_l346_346069

theorem only_positive_integer_cube_less_than_triple (n : ℕ) (h : 0 < n ∧ n^3 < 3 * n) : n = 1 :=
sorry

end only_positive_integer_cube_less_than_triple_l346_346069


namespace rectangle_properties_l346_346621

theorem rectangle_properties :
  ∀ (width : ℝ), width = 5 →
  let length := 2 * width in
  let diagonal := Real.sqrt (width^2 + length^2) in
  let area := length * width in
  diagonal = Real.sqrt 125 ∧ area = 50 := 
by
  intros width h_width
  let length := 2 * width
  let diagonal := Real.sqrt (width^2 + length^2)
  let area := length * width
  have h1 : diagonal = Real.sqrt 125 := sorry
  have h2 : area = 50 := sorry
  exact ⟨h1, h2⟩

end rectangle_properties_l346_346621


namespace coefficient_of_x_eq_neg_6_l346_346792

noncomputable def coefficient_of_x_in_expansion : ℤ :=
  let expanded_expr := (1 + x) * (x - 2 / x) ^ 3
  in collect (x : ℝ) expanded_expr

theorem coefficient_of_x_eq_neg_6 : coefficient_of_x_in_expansion = -6 :=
  sorry

end coefficient_of_x_eq_neg_6_l346_346792


namespace smallest_four_digit_in_pascal_l346_346508

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346508


namespace units_digit_sum_factorials_l346_346156

theorem units_digit_sum_factorials : 
  let T := (∑ i in Finset.range 1 11, i.factorial) + 3
  units_digit T = 6 := 
sorry

end units_digit_sum_factorials_l346_346156


namespace sum_of_squared_residuals_eq_0_03_l346_346740

theorem sum_of_squared_residuals_eq_0_03 :
  ∀ (data : List (ℕ × ℝ)), 
  (data = [(2, 5.1), (3, 6.9), (4, 9.1)]) →
  (∀ x y, (x, y) ∈ data → y = 2 * x + 1 + residual x) →
  (let residuals := data.map (λ (p : ℕ × ℝ), p.snd - (2 * p.fst + 1)) in
   residuals.sum_sq = 0.03) := 
sorry

end sum_of_squared_residuals_eq_0_03_l346_346740


namespace total_seats_in_concert_hall_l346_346046

theorem total_seats_in_concert_hall 
  (c : ℕ) (k : ℕ) (seats_in_middle_row : ℕ) (first_row_seats : ℕ) (rows : ℕ)
  (h1 : rows = 31) 
  (h2 : k = rows / 2)
  (h3 : seats_in_middle_row = 64)
  (h4 : ∀ n, 1 ≤ n ∧ n ≤ rows → (seats_in_middle_row + (n - k) * 2)) : 
  let total_seats := (rows / 2) * (2 * seats_in_middle_row + (rows - 1) * 2) / 2 in
  total_seats = 1984 := 
sorry

end total_seats_in_concert_hall_l346_346046


namespace coefficient_of_x_in_expansion_l346_346787

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end coefficient_of_x_in_expansion_l346_346787


namespace circleTangent_and_intersection_l346_346782

-- Define the main conditions and properties
variables (O : Point)
def circleO := {x, y | x^2 + y^2 = 4}

def isTangent (l : Line) (c : Circle) := 
  let d := abs(l.normalize.equation.pointDist(0, 0)) / sqrt(l.normalize.sqNorm)
  d = c.radius

noncomputable def lineL (k : ℝ) : Line := { x, y | y = k * x + 3}

-- State the main theorem combining both questions from the problem
theorem circleTangent_and_intersection (l: Line) :
  (isTangent l circleO) ∧ (∃ M, M ∈ circleO ∧ (M = A + B))
     ↔ (l.slope = 2 * sqrt(2)) ∨ (l.slope = -2 * sqrt(2)) := by
  sorry

end circleTangent_and_intersection_l346_346782


namespace smallest_four_digit_in_pascals_triangle_l346_346474

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346474


namespace find_value_of_a_l346_346958

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a_l346_346958


namespace equilateral_A1B1C1_iff_isosceles_120_degree_angles_l346_346222

-- Definitions and conditions
variables {A B C A1 B1 C1 : Type}
variables [CommRing A] [CommRing B] [CommRing C]
variables [IsTriangle A B C] [IsNonEquilateralTriangle A B C]
variables [IsChosenPoint A1] [IsChosenPoint B1] [IsChosenPoint C1]
variables (similar1 : Similar (Triangle B A1 C)) (similar2 : Similar (Triangle C B1 A))
variables (similar3 : Similar (Triangle A C1 B))
variables (isosceles1 : Isosceles (Triangle B A1 C) (Angle A1 120))
variables (isosceles2 : Isosceles (Triangle C B1 A) (Angle B1 120))
variables (isosceles3 : Isosceles (Triangle A C1 B) (Angle C1 120))

-- Theorem statement
theorem equilateral_A1B1C1_iff_isosceles_120_degree_angles :
  (Equilateral (Triangle A1 B1 C1)) ↔
  (Isosceles (Triangle B A1 C) (Angle A1 120) ∧
   Isosceles (Triangle C B1 A) (Angle B1 120) ∧
   Isosceles (Triangle A C1 B) (Angle C1 120)) :=
by sorry

end equilateral_A1B1C1_iff_isosceles_120_degree_angles_l346_346222


namespace smallest_four_digit_in_pascals_triangle_l346_346436

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346436


namespace originally_invited_people_l346_346597

theorem originally_invited_people (f t s : ℕ) : t = 8 → f = 5 → s = 7 → 47 = t * f + s := 
by
  intros ht hf hs
  simp [ht, hf, hs]
  sorry

end originally_invited_people_l346_346597


namespace smallest_four_digit_in_pascal_triangle_l346_346554

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346554


namespace limit_calculation_l346_346967

-- Define the given limit problem and its equivalence to the expected result
theorem limit_calculation :
  (lim (x -> 0) (fun x => (3^(5*x) - 2^(-7*x)) / (2*x - tan x)) = ln (3^5 * 2^7)) :=
by 
  sorry

end limit_calculation_l346_346967


namespace cubed_expression_l346_346755

theorem cubed_expression (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 :=
sorry

end cubed_expression_l346_346755


namespace total_real_guppies_correct_l346_346281

noncomputable def haylee_guppies : ℕ := 3 * 12

noncomputable def jose_guppies : ℕ := (nat.sqrt haylee_guppies) / 2

noncomputable def charliz_guppies : ℝ := jose_guppies / 3 - 1.5

def charliz_real_guppies : ℕ := if charliz_guppies < 0 then 0 else floor charliz_guppies

noncomputable def nicolai_guppies : ℝ := 4 * charliz_real_guppies.toReal - 2

def nicolai_real_guppies : ℕ := if nicolai_guppies < 0 then 0 else floor nicolai_guppies

def alice_guppies : ℕ := nicolai_real_guppies + 5

noncomputable def bob_guppies : ℝ := ((jose_guppies ^ 2 / 3) + charliz_real_guppies.toReal) / 2

def bob_real_guppies : ℕ := sorry -- rounding decisions

def cameron_guppies : ℕ := 2 ^ 3

noncomputable def total_real_guppies : ℕ :=
  haylee_guppies + jose_guppies + charliz_real_guppies + nicolai_real_guppies + alice_guppies + (floor bob_guppies) + cameron_guppies

theorem total_real_guppies_correct : total_real_guppies = 53 := sorry

end total_real_guppies_correct_l346_346281


namespace arrangement_count_l346_346211

-- Proposition: The number of different arrangements of 5 people selected and 
-- arranged in a row from 7 people, where person A and person B must be selected, 
-- and person A must be placed to the left of person B, is 600.
theorem arrangement_count (people : Finset ℕ) (A B : ℕ) (hpeople : people.card = 7) 
    (hAB : A ∈ people ∧ B ∈ people) : 
    ∃ (selected : Finset ℕ), selected.card = 5 ∧ (A ∈ selected ∧ B ∈ selected) 
    ∧ arrangement_count selected A B = 600 := 
sorry

end arrangement_count_l346_346211


namespace gcd_lcm_product_l346_346203

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 75) (h2 : b = 90) : Nat.gcd a b * Nat.lcm a b = 6750 :=
by
  sorry

end gcd_lcm_product_l346_346203


namespace log_sum_is_one_l346_346715

theorem log_sum_is_one (a b : ℝ) (h1 : 10^a = 2) (h2 : b = Real.log 5) : a + b = 1 :=
by {
  have h3 : a = Real.log 2,
  { exact Real.log_eq_log_of_exp_eq h1 },
  rw [h3, h2],
  linarith [Real.log_mul 0 10 two_ne_zero (by norm_num), Real.log_10],
  sorry
}

end log_sum_is_one_l346_346715


namespace smallest_four_digit_in_pascals_triangle_l346_346469

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346469


namespace total_time_l346_346128

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l346_346128


namespace total_nails_used_l346_346588

-- Given definitions from the conditions
def square_side_length : ℕ := 36
def nails_per_side : ℕ := 40
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

-- Statement of the problem proof
theorem total_nails_used : nails_per_side * sides_of_square - corners_of_square = 156 := by
  sorry

end total_nails_used_l346_346588


namespace problem_l346_346085

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem_l346_346085


namespace physics_class_size_l346_346961

variable (students : ℕ)
variable (physics math both : ℕ)

-- Conditions
def conditions := students = 75 ∧ physics = 2 * (math - both) + both ∧ both = 9

-- The proof goal
theorem physics_class_size : conditions students physics math both → physics = 56 := 
by 
  sorry

end physics_class_size_l346_346961


namespace simplify_sqrt_expression_l346_346850

theorem simplify_sqrt_expression : (sqrt 18 * sqrt 72 - sqrt 32) = (36 - 4 * sqrt 2) :=
by
  -- Proof goes here
  sorry

end simplify_sqrt_expression_l346_346850


namespace find_value_l346_346214

theorem find_value (x y : ℝ) 
    (h : (sqrt (x - 3 * y) + abs (x^2 - 9)) / (x + 3)^2 = 0) :
    sqrt (x + 2) / sqrt (y + 1) = sqrt 5 / sqrt 2 :=
begin
  sorry
end

end find_value_l346_346214


namespace coin_toss_probability_l346_346307

-- Define the sample space of the coin toss
inductive Coin
| heads : Coin
| tails : Coin

-- Define the probability function
def probability (outcome : Coin) : ℝ :=
  match outcome with
  | Coin.heads => 0.5
  | Coin.tails => 0.5

-- The theorem to be proved: In a fair coin toss, the probability of getting "heads" or "tails" is 0.5
theorem coin_toss_probability (outcome : Coin) : probability outcome = 0.5 :=
sorry

end coin_toss_probability_l346_346307


namespace swans_count_l346_346783

def numberOfSwans : Nat := 12

theorem swans_count (y : Nat) (x : Nat) (h1 : y = 5) (h2 : ∃ n m : Nat, x = 2 * n + 2 ∧ x = 3 * m - 3) : x = numberOfSwans := 
  by 
    sorry

end swans_count_l346_346783


namespace left_handed_classical_music_lovers_l346_346893

-- Define the conditions
variables (total_people left_handed classical_music right_handed_dislike : ℕ)
variables (x : ℕ) -- x will represent the number of left-handed classical music lovers

-- State the assumptions based on conditions
axiom h1 : total_people = 30
axiom h2 : left_handed = 12
axiom h3 : classical_music = 20
axiom h4 : right_handed_dislike = 3
axiom h5 : 30 = x + (12 - x) + (20 - x) + 3

-- State the theorem to prove
theorem left_handed_classical_music_lovers : x = 5 :=
by {
  -- Skip the proof using sorry
  sorry
}

end left_handed_classical_music_lovers_l346_346893


namespace point_in_fourth_quadrant_l346_346908

theorem point_in_fourth_quadrant (m : ℝ) (h : (2 / 3) < m ∧ m < 1) : 
  let z := m * (3 + (1:ℂ) * Complex.i) - (2 + (1:ℂ) * Complex.i) in
  Complex.re z > 0 ∧ Complex.im z < 0 :=
by 
  let z := m * (3 + (1:ℂ) * Complex.i) - (2 + (1:ℂ) * Complex.i)
  have hz : z = ((3 * m - 2) + ((m - 1) * Complex.i)), by
  { simp [z, Complex.add_re, Complex.add_im, Complex.sub_re, Complex.sub_im, Complex.mul_re, Complex.mul_im] }
  rw hz
  split
  { linarith [h.1, h.2] } -- to handle the real part
  { linarith [h.1, h.2] } -- to handle the imaginary part

end point_in_fourth_quadrant_l346_346908


namespace simplify_cosines_l346_346849

noncomputable def omega := Complex.exp (2 * Real.pi * Complex.I / 15)

theorem simplify_cosines : 
  let x := (Real.cos (4 * Real.pi / 15) + Real.cos (10 * Real.pi / 15) + Real.cos (14 * Real.pi / 15))
  in x = 1 :=
by
  sorry

end simplify_cosines_l346_346849


namespace hyperbola_focus_exists_l346_346653

-- Define the basic premises of the problem
def is_hyperbola (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

-- Define a condition for the focusing property of the hyperbola.
def is_focus (x y : ℝ) : Prop :=
  (x = -2) ∧ (y = 4 + (10 * Real.sqrt 3 / 3))

-- The theorem to be proved
theorem hyperbola_focus_exists : ∃ x y : ℝ, is_hyperbola x y ∧ is_focus x y :=
by
  -- Proof to be filled in
  sorry

end hyperbola_focus_exists_l346_346653


namespace number_of_triangles_in_decagon_l346_346985

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346985


namespace complex_distance_l346_346702

open Complex

theorem complex_distance (z : ℂ) (h : z * (1 + Complex.i) = 4) : 
  (Complex.abs (z - (1 : ℂ))) = Real.sqrt 5 := by
  sorry

end complex_distance_l346_346702


namespace complex_ratio_simplification_and_percentage_increase_l346_346366

theorem complex_ratio_simplification_and_percentage_increase
  (a b c d : ℕ) 
  (h_ratio: (4 : ℕ → a) ∧ (16 : ℕ → b) ∧ (20 : ℕ → c) ∧ (12 : ℕ → d)) 
  : (a, b, c, d = (1, 4, 5, 3)) ∧ (∃ p, p = 200) :=
by
  sorry

end complex_ratio_simplification_and_percentage_increase_l346_346366


namespace december_24_day_of_week_l346_346049

theorem december_24_day_of_week (nov_24_is_friday : ∃ n, n % 7 = 5) : ∃ k, k % 7 = 0 :=
begin
  sorry
end

end december_24_day_of_week_l346_346049


namespace g_value_at_800_l346_346335

noncomputable def g : ℝ+ → ℝ := sorry

theorem g_value_at_800 (h1 : ∀ (x y : ℝ+), g (x * y) = g x / y)
    (h2 : g 1000 = 4) :
  g 800 = 5 := by
    sorry

end g_value_at_800_l346_346335


namespace calvin_weight_after_one_year_l346_346174

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l346_346174


namespace total_days_stayed_l346_346868

-- Definitions of given conditions as variables
def cost_first_week := 18
def days_first_week := 7
def cost_additional_week := 13
def total_cost := 334

-- Formulation of the target statement in Lean
theorem total_days_stayed :
  (days_first_week + 
  ((total_cost - (days_first_week * cost_first_week)) / cost_additional_week)) = 23 :=
by
  sorry

end total_days_stayed_l346_346868


namespace expected_value_of_dodecahedral_die_l346_346416

-- Given a dodecahedral die with 12 faces numbered from 1 to 12
def faces : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- List of probabilities for each face (since the die is fair, the probability is equal for all faces)
def probabilities : List ℚ := List.repeat (1/12 : ℚ) 12

-- Sum of face values multiplied by their corresponding probabilities
def expectedValue (faces : List ℕ) (probabilities : List ℚ) : ℚ :=
  (List.zipWith (· *) (faces.map (· : ℚ)) probabilities).sum

-- The expected value of rolling a dodecahedral die is 6.5
theorem expected_value_of_dodecahedral_die :
  expectedValue faces probabilities = 6.5 := by
  -- Proof will be provided here.
  sorry

end expected_value_of_dodecahedral_die_l346_346416


namespace locus_of_centers_l346_346686

-- Definitions
variables (R : ℝ) (O : Euclidean_Space 2)

-- Expected Answer
theorem locus_of_centers :
  ∀ C : Euclidean_Space 2, (dist O C = R) ↔ (∃ P : Euclidean_Space 2, dist C P = R ∧ P = O) :=
by sorry

end locus_of_centers_l346_346686


namespace least_positive_integer_with_seven_distinct_factors_l346_346063

theorem least_positive_integer_with_seven_distinct_factors :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 → m ≠ 1 → (∀ k : ℕ, k ∣ n → k ∣ m → k ∣ m)) → n = 64) ∧
    (∃ p : ℕ, (nat.prime p ∧ n = p^6)) :=
begin
  sorry
end

end least_positive_integer_with_seven_distinct_factors_l346_346063


namespace min_cubes_l346_346930

-- Definitions of the conditions
def unit_cube : Type := ℝ × ℝ × ℝ
def shares_face (c1 c2 : unit_cube) : Prop :=
  let (x1, y1, z1) := c1 in
  let (x2, y2, z2) := c2 in
  abs (x1 - x2) + abs (y1 - y2) + abs (z1 - z2) = 1

def front_view (figure : list unit_cube) : Prop :=
  -- figure matches the front view: 2-high on the left, 1-high on the right
  ∃ left right : list unit_cube,
    left.length = 2 ∧ right.length = 1 ∧
    all (λ c, c.1 = 0) left ∧
    all (λ c, c.1 = 1) right

def side_view (figure : list unit_cube) : Prop :=
  -- figure matches the side view: 3-high column on the right
  ∃ right : list unit_cube,
    right.length = 3 ∧ all (λ c, c.2 = 1) right

-- Main theorem statement
theorem min_cubes (figure : list unit_cube) :
  (∀ c1 c2 ∈ figure, shares_face c1 c2) ∧ front_view figure ∧ side_view figure → figure.length = 5 := 
sorry

end min_cubes_l346_346930


namespace percentage_of_blue_flowers_l346_346831

theorem percentage_of_blue_flowers 
  (total_flowers : Nat)
  (red_flowers : Nat)
  (white_flowers : Nat)
  (total_flowers_eq : total_flowers = 10)
  (red_flowers_eq : red_flowers = 4)
  (white_flowers_eq : white_flowers = 2)
  :
  ( (total_flowers - (red_flowers + white_flowers)) * 100 ) / total_flowers = 40 :=
by
  sorry

end percentage_of_blue_flowers_l346_346831


namespace housewife_money_left_l346_346606

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end housewife_money_left_l346_346606


namespace range_of_a_if_f_of_a_gt_half_l346_346735

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then log (1/3) x else 2 ^ x 

theorem range_of_a_if_f_of_a_gt_half (a : ℝ) : 
  f a > 1 / 2 → a ∈ set.Ioo (-1 : ℝ) (real.sqrt 3 / 3) := 
begin
  sorry
end

end range_of_a_if_f_of_a_gt_half_l346_346735


namespace fraction_of_single_men_l346_346643

theorem fraction_of_single_men (total_faculty : ℕ) (percent_women percent_married percent_men_married : ℚ)
  (H1 : percent_women = 60 / 100) (H2 : percent_married = 60 / 100) (H3 : percent_men_married = 1 / 4) :
  let total_men := total_faculty * (1 - percent_women)
  let married_men := total_men * percent_men_married
  let single_men := total_men - married_men
  single_men / total_men = 3 / 4 :=
by {
  let total_men := total_faculty * (1 - percent_women),
  let married_men := total_men * percent_men_married,
  let single_men := total_men - married_men,
  have H_total_men : total_men = total_faculty * (1 - percent_women), from rfl,
  rw H_total_men,
  have H_married_men : married_men = total_men * percent_men_married, from rfl,
  rw H_married_men,
  have H_single_men : single_men = total_men - married_men, from rfl,
  rw H_single_men,
  have H_fraction : (total_men - married_men) / total_men = 3 / 4,
  { sorry }, -- detailed proof can be filled in as needed
  exact H_fraction,
}

end fraction_of_single_men_l346_346643


namespace equilateral_triangle_condition_l346_346102

theorem equilateral_triangle_condition 
  {Z1 Z2 Z3 : ℂ} 
  (omega : ℂ) 
  (h_omega_pow3 : omega^3 = 1)
  (h_omega_sum : 1 + omega + omega^2 = 0) :
  (∃ (Z : set (ℂ × ℂ × ℂ)), (Z1, Z2, Z3) ∈ Z ∧ ∀ (A B C : ℂ), (A, B, C) ∈ Z → (A = B ∧ B = C ∧ C = A)) ↔ (Z1 + omega * Z2 + omega^2 * Z3 = 0) :=
sorry

end equilateral_triangle_condition_l346_346102


namespace base8_subtraction_correct_l346_346155

theorem base8_subtraction_correct :
  ∀ (a b : ℕ) (h1 : a = 7534) (h2 : b = 3267),
      (a - b) % 8 = 4243 % 8 := by
  sorry

end base8_subtraction_correct_l346_346155


namespace number_of_true_propositions_l346_346411

noncomputable def proposition1 : Prop := ∀ (x : ℝ), x^2 - 3 * x + 2 > 0
noncomputable def proposition2 : Prop := ∃ (x : ℚ), x^2 = 2
noncomputable def proposition3 : Prop := ∃ (x : ℝ), x^2 - 1 = 0
noncomputable def proposition4 : Prop := ∀ (x : ℝ), 4 * x^2 > 2 * x - 1 + 3 * x^2

theorem number_of_true_propositions : (¬ proposition1 ∧ ¬ proposition2 ∧ proposition3 ∧ ¬ proposition4) → 1 = 1 :=
by
  intros
  sorry

end number_of_true_propositions_l346_346411


namespace count_positive_integer_terms_in_sequence_l346_346266

theorem count_positive_integer_terms_in_sequence :
  let c (n : ℕ) := (4 * ↑n + 31) / (2 * ↑n - 1) in
  ∃ (N : ℕ), N = 4 ∧ (∀ n : ℕ, c n ∈ ℕ → n ≤ 20 ∧ n = 1 ∨ n = 2 ∨ n = 6 ∨ n = 17) :=
by
  sorry

end count_positive_integer_terms_in_sequence_l346_346266


namespace trig_identity_l346_346754

theorem trig_identity 
  (α β : Real) 
  (h : (cos α)^6 / (cos β)^3 + (sin α)^6 / (sin β)^3 = 1) : 
  (sin β)^6 / (sin α)^3 + (cos β)^6 / (cos α)^3 = 1 := 
sorry

end trig_identity_l346_346754


namespace remainder_of_5n_minus_9_l346_346566

theorem remainder_of_5n_minus_9 (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 :=
by
  sorry -- Proof is omitted, as per instruction.

end remainder_of_5n_minus_9_l346_346566


namespace muffin_to_banana_ratio_l346_346804

theorem muffin_to_banana_ratio (m b : ℝ) 
  (jenny_cost : 5 * m + 4 * b) 
  (michael_cost : 3 * (5 * m + 4 * b) = 3 * m + 20 * b) :
  (m / b) = 3 / 2 :=
by
  sorry

end muffin_to_banana_ratio_l346_346804


namespace line_intersects_circle_slope_angle_range_l346_346882

noncomputable def line_eq (k : ℝ) : ℝ × ℝ → Prop := 
  λ p, p.2 = k * p.1 + 3

def circle_eq (p : ℝ × ℝ) : Prop :=
  (p.1 - 2)^2 + (p.2 - 3)^2 = 4

def chord_length_ge (k : ℝ) : Prop :=
  2 * Real.sqrt (4 - (2*k)^2 / (1+k^2)) ≥ 2 * Real.sqrt 3

def slope_angle_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ ≤ Real.pi / 6 ∨ 5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi

theorem line_intersects_circle_slope_angle_range (θ : ℝ) (k : ℝ) :
  k = Real.tan θ →
  (∃ (p1 p2 : ℝ × ℝ), line_eq k p1 ∧ circle_eq p1 ∧ line_eq k p2 ∧ circle_eq p2 ∧ chord_length_ge k) →
  slope_angle_range θ :=
sorry

end line_intersects_circle_slope_angle_range_l346_346882


namespace elite_academy_geometry_test_l346_346959

theorem elite_academy_geometry_test (total_problems : ℕ) (passing_score_percent : ℝ) :
  total_problems = 40 → passing_score_percent = 75 → 
  ∃ max_missed_problems : ℕ, max_missed_problems = 10 :=
by
  intro h_total_problems h_passing_score_percent
  use 10
  sorry

end elite_academy_geometry_test_l346_346959


namespace least_positive_integer_with_exactly_seven_factors_l346_346059

theorem least_positive_integer_with_exactly_seven_factors : 
  ∃ (n : ℕ), (∀ n > 0 → n ∈ ℕ) ∧ (∀ m : ℕ, m > 0 ∧ m < n → number_of_factors m ≠ 7) ∧ number_of_factors n = 7 ∧ n = 64 :=
by
  sorry

open Nat

/--
Defines the number of factors of a given number.
-/
def number_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, x > 0 ∧ n % x = 0).card

attribute [simp] number_of_factors -- Simplifies and registers the number_of_factors as a known attribute

end least_positive_integer_with_exactly_seven_factors_l346_346059


namespace bisection_method_applies_l346_346902

noncomputable def f (x : ℝ) : ℝ := x^3 + 1.1*x^2 + 0.9*x - 1.4

theorem bisection_method_applies : 
  ∃ (c : ℝ), c ∈ set.Ioo 0 1 ∧ |c - 0.6875| < 0.1 ∧ f c = 0 :=
by
  have h_interval : 0 ∈ set.Icc (0: ℝ) 1 := by norm_num
  have h_f0 : f 0 = -1.4 := by norm_num [f]
  have h_f1 : f 1 = 1.6 := by norm_num [f]
  have h_sign_change : f 0 < 0 ∧ f 1 > 0 := by norm_num [h_f0, h_f1]
  have h_continuous : continuous f := by continuity
  have h_zero_exists : ∃ (c : ℝ), c ∈ set.Ioo 0 1 ∧ f c = 0 := 
    intermediate_value_Icc (set.mem_Icc_of_Ioo h_interval) h_f0 h_f1 h_continuous
  obtain ⟨c, hc₁, hc₂⟩ := h_zero_exists
  use c
  split
  · exact hc₁
  · norm_num at hc₁
    linarith [hc₁]
  · exact hc₂

end bisection_method_applies_l346_346902


namespace smallest_four_digit_in_pascal_triangle_l346_346558

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346558


namespace solve_for_x_l346_346663

theorem solve_for_x : ∃ x : ℝ, (24 / 36 = real.sqrt (x / 36)) ∧ (x = 16) :=
by {
    sorry
}

end solve_for_x_l346_346663


namespace smallest_room_size_l346_346112

theorem smallest_room_size (S : ℕ) (h1 : ∃ t1 t2, t1 = 12 ∧ t2 = 9 ∧ √(t1^2 + t2^2) = 15) : S ≥ 15 := by
  sorry

end smallest_room_size_l346_346112


namespace arithmetic_seq_problem_l346_346785

theorem arithmetic_seq_problem (a b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) :
  (a 2 = 5) →
  (a 5 = 11) →
  (∀ n, S n = n^2 + a n) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, b n = if n = 1 then 4 else 2 * n + 1) ∧
  (∀ n, T n = (6 * n - 1) / (20 * (2 * n + 3))) :=
begin
  intros,
  sorry,
end

end arithmetic_seq_problem_l346_346785


namespace bear_problem_l346_346971

variables (w b br : ℕ)

theorem bear_problem 
    (h1 : b = 2 * w)
    (h2 : br = b + 40)
    (h3 : w + b + br = 190) :
    b = 60 :=
by
  sorry

end bear_problem_l346_346971


namespace f_n_one_l346_346736

def f (x : ℝ) (h : x > 0) : ℝ := x / (2 * x + 2)

noncomputable def f_n (n : ℕ) (x : ℝ) (h : x > 0) : ℝ :=
nat.rec_on n (f x h) (λ n fn, f (fn) sorry)

theorem f_n_one (n : ℕ) (hn : n > 0) : (f_n n 1 (by norm_num)) = 1 / (3 * 2^n - 2) :=
sorry

end f_n_one_l346_346736


namespace part1_part2_l346_346077

-- Define properties for the first part of the problem
def condition1 (weightA weightB : ℕ) : Prop :=
  weightA + weightB = 7500 ∧ weightA = 3 * weightB / 2

def question1_answer : Prop :=
  ∃ weightA weightB : ℕ, condition1 weightA weightB ∧ weightA = 4500 ∧ weightB = 3000

-- Combined condition for the second part of the problem scenarios
def condition2a (y : ℕ) : Prop := y ≤ 1800 ∧ 18 * y - 10 * y = 17400
def condition2b (y : ℕ) : Prop := 1800 < y ∧ y ≤ 3000 ∧ 18 * y - (15 * y - 9000) = 17400
def condition2c (y : ℕ) : Prop := y > 3000 ∧ 18 * y - (20 * y - 24000) = 17400

def question2_answer : Prop :=
  (∃ y : ℕ, condition2b y ∧ y = 2800) ∨ (∃ y : ℕ, condition2c y ∧ y = 3300)

-- The Lean statements for both parts of the problem
theorem part1 : question1_answer := sorry

theorem part2 : question2_answer := sorry

end part1_part2_l346_346077


namespace addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l346_346672

theorem addition_comm (a b : ℕ) : a + b = b + a :=
by sorry

theorem subtraction_compare {a b c : ℕ} (h1 : a < b) (h2 : c = 28) : 56 - c < 65 - c :=
by sorry

theorem multiplication_comm (a b : ℕ) : a * b = b * a :=
by sorry

theorem subtraction_greater {a b c : ℕ} (h1 : a - b = 18) (h2 : a - c = 27) (h3 : 32 = b) (h4 : 23 = c) : a - b > a - c :=
by sorry

end addition_comm_subtraction_compare_multiplication_comm_subtraction_greater_l346_346672


namespace cos_2x_range_l346_346399

def cos_range (a b : ℝ) (f : ℝ → ℝ) : set ℝ :=
  {y : ℝ | ∃ x ∈ set.Icc a b, f x = y}

theorem cos_2x_range : 
  cos_range (↑(Real.pi/6)) (↑(5*Real.pi/6)) (λ x, Real.cos (2*x)) = set.Icc (-1) (1/2) := 
by
  sorry

end cos_2x_range_l346_346399


namespace second_to_last_number_in_sequence_l346_346193

theorem second_to_last_number_in_sequence : 
  ∀ (seq : ℕ → ℕ) (sum_seq : ℕ), 
    (∀ n, seq n = if even n then (100 - n / 2) ^ 2 else -(100 - (n - 1) / 2) ^ 2) ∧ 
    sum_seq = (List.range 100).sum (λ n, seq n) ∧ 
    sum_seq = 5050 
  → (sum_seq - 1 = 5049) := 
begin
  intros seq sum_seq h,
  cases h with seq_def h_sum,
  cases h_sum with h_sum_eq h_total_sum,
  sorry
end

end second_to_last_number_in_sequence_l346_346193


namespace tan_double_angle_solution_l346_346700

theorem tan_double_angle_solution (α : ℝ) (hα : α ∈ (π, 3 * π / 2)) 
  (h : (1 + Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 9/2) : 
  Real.tan (2 * α) = -4/3 :=
sorry

end tan_double_angle_solution_l346_346700


namespace milk_left_is_correct_l346_346962

def total_morning_milk : ℕ := 365
def total_evening_milk : ℕ := 380
def milk_sold : ℕ := 612
def leftover_milk_from_yesterday : ℕ := 15

def total_milk_left : ℕ :=
  (total_morning_milk + total_evening_milk - milk_sold) + leftover_milk_from_yesterday

theorem milk_left_is_correct : total_milk_left = 148 := by
  sorry

end milk_left_is_correct_l346_346962


namespace melanie_average_speed_l346_346836

theorem melanie_average_speed
  (bike_distance run_distance total_time : ℝ)
  (h_bike : bike_distance = 15)
  (h_run : run_distance = 5)
  (h_time : total_time = 4) :
  (bike_distance + run_distance) / total_time = 5 :=
by
  sorry

end melanie_average_speed_l346_346836


namespace opposite_face_of_lime_is_black_l346_346368

-- Define the colors
inductive Color
| P | C | M | S | K | L

-- Define the problem conditions
def face_opposite (c : Color) : Color := sorry

-- Theorem statement
theorem opposite_face_of_lime_is_black : face_opposite Color.L = Color.K := sorry

end opposite_face_of_lime_is_black_l346_346368


namespace material_decrease_cost_comparison_l346_346022

-- Proof Problem for Question 1
def changes : List Int := [-3, 4, -1, 2, -5]
def counts : List Int := [2, 1, 3, 3, 2]

theorem material_decrease :
  List.sum (List.zipWith (*) changes counts) = -9 :=
sorry

-- Proof Problem for Question 2
def incoming_cost : Int := 5
def outgoing_cost : Int := 8
def uniform_cost : Int := 6

def calculate_cost_option1 (changes counts : List Int) (in_cost out_cost : Int) : Int :=
let (inc, out) := List.foldl (λ (acc : Int × Int) (p : Int × Int) => 
  if p.fst > 0 then (acc.fst + p.fst * p.snd, acc.snd) else (acc.fst, acc.snd + p.fst.abs * p.snd)) (0, 0) (List.zip changes counts) in
inc * in_cost + out * out_cost

def calculate_cost_option2 (counts : List Int) (uniform_cost : Int) : Int :=
uniform_cost * List.sum counts

theorem cost_comparison :
  calculate_cost_option2 counts uniform_cost < calculate_cost_option1 changes counts incoming_cost outgoing_cost :=
sorry

end material_decrease_cost_comparison_l346_346022


namespace sugar_cheaper_than_apples_l346_346669

/-- Given conditions about the prices and quantities of items that Fabian wants to buy,
    prove the price difference between one pack of sugar and one kilogram of apples. --/
theorem sugar_cheaper_than_apples
  (price_kg_apples : ℝ)
  (price_kg_walnuts : ℝ)
  (total_cost : ℝ)
  (cost_diff : ℝ)
  (num_kg_apples : ℕ := 5)
  (num_packs_sugar : ℕ := 3)
  (num_kg_walnuts : ℝ := 0.5)
  (price_kg_apples_val : price_kg_apples = 2)
  (price_kg_walnuts_val : price_kg_walnuts = 6)
  (total_cost_val : total_cost = 16) :
  cost_diff = price_kg_apples - (total_cost - (num_kg_apples * price_kg_apples + num_kg_walnuts * price_kg_walnuts))/num_packs_sugar → 
  cost_diff = 1 :=
by
  sorry

end sugar_cheaper_than_apples_l346_346669


namespace sum_of_natural_numbers_l346_346099

noncomputable def alpha : ℝ :=
  (-1 + Real.sqrt 29) / 2

theorem sum_of_natural_numbers (n : ℕ) :
  (α > 2) →
  (∀ k : ℕ, k ≥ 1 → Irrational (α^k)) →
  ∃ (x : ℕ → ℕ), (∀ k : ℕ, x k ≤ 6) ∧ (n = ∑ k in Finset.range (n+1), x k * α^k) := by
  sorry

end sum_of_natural_numbers_l346_346099


namespace tangent_equality_l346_346152

theorem tangent_equality 
  (P A B C D E F : Point) (circle : Circle) 
  (h1 : Tangent PA circle) 
  (h2 : Tangent PB circle)
  (h3 : SecantLine P C D circle)
  (h4 : Parallel (LineThrough B P) (LineThrough A P))
  (h5 : Collinear A C E)
  (h6 : Collinear A D F) 
  : distance B E = distance B F := 
by 
  sorry

end tangent_equality_l346_346152


namespace sampling_methods_correctness_l346_346585

open Classical

noncomputable def community_size : ℕ := 4000
noncomputable def sample_size_community : ℕ := 200
noncomputable def ratio_young_middle_aged_elderly : (ℕ × ℕ × ℕ) := (1, 2, 4)

noncomputable def class_size : ℕ := 45
noncomputable def sample_size_class : ℕ := 5

theorem sampling_methods_correctness :
  (∃ method1 method2 : string, method1 = "stratified sampling" ∧ method2 = "simple random sampling") := 
by
  let method1 := "stratified sampling"
  let method2 := "simple random sampling"
  exists method1 method2
  split
  all_goals { reflexivity }
  sorry

end sampling_methods_correctness_l346_346585


namespace coefficient_of_x_eq_neg_6_l346_346793

noncomputable def coefficient_of_x_in_expansion : ℤ :=
  let expanded_expr := (1 + x) * (x - 2 / x) ^ 3
  in collect (x : ℝ) expanded_expr

theorem coefficient_of_x_eq_neg_6 : coefficient_of_x_in_expansion = -6 :=
  sorry

end coefficient_of_x_eq_neg_6_l346_346793


namespace sum_of_natural_numbers_l346_346098

noncomputable def alpha : ℝ :=
  (-1 + Real.sqrt 29) / 2

theorem sum_of_natural_numbers (n : ℕ) :
  (α > 2) →
  (∀ k : ℕ, k ≥ 1 → Irrational (α^k)) →
  ∃ (x : ℕ → ℕ), (∀ k : ℕ, x k ≤ 6) ∧ (n = ∑ k in Finset.range (n+1), x k * α^k) := by
  sorry

end sum_of_natural_numbers_l346_346098


namespace at_least_one_negative_l346_346293

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) : a < 0 ∨ b < 0 := by
  sorry

end at_least_one_negative_l346_346293


namespace dot_product_b_c_eq_neg_one_l346_346277

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

theorem dot_product_b_c_eq_neg_one
  (h1 : a + b + c = 0)
  (h2 : ⟪a - b, c⟫ = 0)
  (h3 : ⟪a, b⟫ = 0)
  (h4 : ⟪a, a⟫ = 1) : 
  ⟪b, c⟫ = -1 :=
by {
  sorry
}

end dot_product_b_c_eq_neg_one_l346_346277


namespace juniors_score_l346_346303

theorem juniors_score (juniors seniors total_students avg_score avg_seniors_score total_score : ℝ)
  (hj: juniors = 0.2 * total_students)
  (hs: seniors = 0.8 * total_students)
  (ht: total_students = 20)
  (ha: avg_score = 78)
  (hp: (seniors * avg_seniors_score + juniors * c) / total_students = avg_score)
  (havg_seniors: avg_seniors_score = 76)
  (hts: total_score = total_students * avg_score)
  (total_seniors_score : ℝ)
  (hts_seniors: total_seniors_score = seniors * avg_seniors_score)
  (total_juniors_score : ℝ)
  (hts_juniors: total_juniors_score = total_score - total_seniors_score)
  (hjs: c = total_juniors_score / juniors) :
  c = 86 :=
sorry

end juniors_score_l346_346303


namespace simplify_expression_1_simplify_expression_2_l346_346965

-- Problem 1
theorem simplify_expression_1 (m : ℝ) (h1 : m ≠ -4) (h2 : m ≠ 4) :
  (m^2 - 16) / (m^2 + 8m + 16) / (m - 4) / (2 * m + 8) * (m - 2) / (m + 2) = (2 * (m - 2)) / (m + 2) :=
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (3 / (x + 2)) + (1 / (2 - x)) - (2 * x / (4 - x^2)) = 4 / (x + 2) :=
  sorry

end simplify_expression_1_simplify_expression_2_l346_346965


namespace bathroom_new_area_l346_346037

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l346_346037


namespace compute_alpha_l346_346330

-- Define the main hypothesis with complex numbers
variable (α γ : ℂ)
variable (h1 : γ = 4 + 3 * Complex.I)
variable (h2 : ∃r1 r2: ℝ, r1 > 0 ∧ r2 > 0 ∧ (α + γ = r1) ∧ (Complex.I * (α - 3 * γ) = r2))

-- The main theorem
theorem compute_alpha : α = 12 + 3 * Complex.I :=
by
  sorry

end compute_alpha_l346_346330


namespace sum_of_3digit_numbers_remainder_2_l346_346906

theorem sum_of_3digit_numbers_remainder_2 :
  let s := {x ∈ finset.range 1000 | 100 ≤ x ∧ x ≤ 999 ∧ x % 3 = 2},
  (s.sum (id) = 164850) := sorry

end sum_of_3digit_numbers_remainder_2_l346_346906


namespace sum_modulo_seven_l346_346646

theorem sum_modulo_seven :
  let s := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999
  s % 7 = 2 :=
by
  sorry

end sum_modulo_seven_l346_346646


namespace find_a8_l346_346248

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem find_a8 : a 8 = 14 :=
by
  have h1 : S 8 = 64 - 8 :=
    by rw S; simp
  have h2 : S 7 = 49 - 7 :=
    by rw S; simp
  have h3 : a 8 = 56 - 42 :=
    by rw [a, h1, h2]; simp
  rw h3
  norm_num

end find_a8_l346_346248


namespace arithmetic_geometric_fraction_l346_346719

theorem arithmetic_geometric_fraction (a x₁ x₂ b y₁ y₂ : ℝ) 
  (h₁ : x₁ + x₂ = a + b) 
  (h₂ : y₁ * y₂ = ab) : 
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (ab) := 
by
  sorry

end arithmetic_geometric_fraction_l346_346719


namespace least_positive_integer_with_seven_factors_l346_346067

theorem least_positive_integer_with_seven_factors : ∃ n : ℕ, n = 64 ∧ 
  ∃ a p : ℕ, a = 6 ∧ (nat.prime p) ∧ n = p^a :=
by
  sorry

end least_positive_integer_with_seven_factors_l346_346067


namespace calvin_weight_after_one_year_l346_346169

theorem calvin_weight_after_one_year :
  ∀ (initial_weight weight_loss_per_month months : ℕ),
  initial_weight = 250 →
  weight_loss_per_month = 8 →
  months = 12 →
  (initial_weight - (weight_loss_per_month * months) = 154) := by
  intros initial_weight weight_loss_per_month months 
  intro h1 h2 h3
  rw [h1, h2, h3]
  show (250 - (8 * 12) = 154)
  norm_num
  sorry

end calvin_weight_after_one_year_l346_346169


namespace rectangle_perimeter_l346_346781

-- Define the conditions and the problem
variable (EF FG EH : ℝ)

def is_rectangle (EF FG EH : ℝ) : Prop :=
  (EH^2 = FG^2 + EF^2) ∧ (EF = 2 * FG) ∧ (FG = 10) ∧ (EH = 26)

-- Define the perimeter calculation
def perimeter (EF FG : ℝ) : ℝ :=
  2 * (EF + FG)

-- Formulate the theorem for the given conditions
theorem rectangle_perimeter (EF FG EH : ℝ) (h : is_rectangle EF FG EH) :
  perimeter EF FG = 60 :=
begin
  -- Start by extracting the conditions
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h4,

  -- Substitute the known lengths
  have EF_def : EF = 2 * FG := h2,
  have FG_def : FG = 10 := h3,
  have EH_def : EH = 26 := h4,

  -- Calculate and ensure the correct perimeter
  calc
    perimeter EF FG = 2 * (EF + FG) : rfl
                  ... = 2 * (2 * FG + FG) : by rw EF_def
                  ... = 2 * (2 * 10 + 10) : by rw FG_def
                  ... = 2 * (20 + 10)    : rfl
                  ... = 2 * 30           : rfl
                  ... = 60               : rfl,
end

end rectangle_perimeter_l346_346781


namespace number_of_triangles_in_decagon_l346_346984

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346984


namespace smallest_four_digit_in_pascal_l346_346506

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346506


namespace prob_55_equals_one_third_probability_distribution_correct_expected_value_correct_l346_346614

-- Definitions based on the given conditions
def num_questions : ℕ := 12
def correct_questions : ℕ := 9
def typeA_questions : ℕ := 2
def typeB_question : ℕ := 1
def points_per_correct : ℕ := 5
def total_points := num_questions * points_per_correct

-- Probability calculations for Type A and Type B questions
def prob_correct_A : ℚ := 1 / 2
def prob_correct_B : ℚ := 1 / 3
def prob_incorrect_B : ℚ := 2 / 3

-- Define the score x based on conditions
def score (num_correct_A : ℕ) (correct_B : bool) : ℕ :=
  (correct_questions + num_correct_A + (if correct_B then 1 else 0)) * points_per_correct

-- The probability distribution and expected value calculations
def probability_x_55 : ℚ :=
  1 / 6 + 2 / 12

def probability_distribution (x : ℕ) : ℚ :=
  if x = 45 then 1 / 6
  else if x = 50 then 5 / 12
  else if x = 55 then 1 / 3
  else if x = 60 then 1 / 12
  else 0

def expected_value : ℚ :=
  45 * (1 / 6) + 50 * (5 / 12) + 55 * (1 / 3) + 60 * (1 / 12)

-- Proof statements
theorem prob_55_equals_one_third :
  probability_x_55 = 1 / 3 := sorry

theorem probability_distribution_correct :
  ∀ x, probability_distribution x = match x with
                                      | 45 => 1 / 6
                                      | 50 => 5 / 12
                                      | 55 => 1 / 3
                                      | 60 => 1 / 12
                                      | _ => 0 := sorry

theorem expected_value_correct :
  expected_value = 155 / 3 := sorry

end prob_55_equals_one_third_probability_distribution_correct_expected_value_correct_l346_346614


namespace isosceles_triangle_congruent_side_length_l346_346867

theorem isosceles_triangle_congruent_side_length 
  (base : ℝ) (area : ℝ) (a b c : ℝ) 
  (h1 : a = c)
  (h2 : a = base / 2)
  (h3 : (base * a) / 2 = area)
  : b = 5 * Real.sqrt 10 := 
by sorry

end isosceles_triangle_congruent_side_length_l346_346867


namespace constant_term_expansion_eq_neg20_l346_346183

theorem constant_term_expansion_eq_neg20 :
  let f : ℤ → ℤ := λ x: ℤ,  2*x - 1 / (2*x)
  ∃ r : ℤ, T (6 : ℤ) r = -20 := by
  sorry

end constant_term_expansion_eq_neg20_l346_346183


namespace domain_of_fraction_function_l346_346260

noncomputable def domain_of_composite_function (f : ℝ → ℝ) : set ℝ :=
  { x : ℝ | x ∈ set.Icc 0 4 }

theorem domain_of_fraction_function (f : ℝ → ℝ) :
  domain_of_composite_function f = set.Icc 0 4 →
  { x : ℝ | x ≠ 1 ∧ -3 ≤ x ∧ x < 1 } = set.Ico (-3 : ℝ) 1 :=
by
  sorry

end domain_of_fraction_function_l346_346260


namespace smallest_four_digit_number_in_pascals_triangle_l346_346541

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346541


namespace count_people_taller_than_right_neighbor_l346_346894

noncomputable def number_of_people_taller_than_right_neighbor (n : ℕ) (k : ℕ) : ℕ :=
  n - 1 - k

theorem count_people_taller_than_right_neighbor :
  ∀ (n k : ℕ), n = 50 → k = 15 → number_of_people_taller_than_right_neighbor n k = 34 :=
by
  intros n k hn hk
  rw [hn, hk]
  unfold number_of_people_taller_than_right_neighbor
  norm_num
  sorry

end count_people_taller_than_right_neighbor_l346_346894


namespace probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l346_346618

namespace ProbabilityKeys

-- Define the problem conditions and the probability computations
def keys : ℕ := 4
def successful_keys : ℕ := 2
def unsuccessful_keys : ℕ := 2

def probability_first_fail (k : ℕ) (s : ℕ) : ℚ := (s : ℚ) / (k : ℚ)
def probability_second_success_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (s + 1 - 1: ℚ) 
def probability_second_success_not_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (k : ℚ)

-- The statements to be proved
theorem probability_door_opened_second_attempt_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_discarded unsuccessful_keys keys) = (1 : ℚ) / (3 : ℚ) :=
by sorry

theorem probability_door_opened_second_attempt_not_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_not_discarded successful_keys keys) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end ProbabilityKeys

end probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l346_346618


namespace radii_of_externally_tangent_circles_l346_346027

theorem radii_of_externally_tangent_circles (L x R r : ℝ) (h1 : L = 20) 
  (h2 : x = (100 / ((Real.sqrt R) + (Real.sqrt r)) ^ 2)) (h3 : R * r = 100) :
  R = 10 ∧ r = 10 :=
by
  sorry

# Print Lean version of the theorem statement

end radii_of_externally_tangent_circles_l346_346027


namespace binomial_theorem_l346_346846

theorem binomial_theorem (a b : ℝ) (n : ℕ) (h : n > 0) :
  (a + b)^n = ∑ r in finset.range (n + 1), nat.choose n r * a^(n - r) * b^r :=
sorry

end binomial_theorem_l346_346846


namespace fill_board_count_l346_346780

theorem fill_board_count (m n : ℕ) : 
  ∃ count : ℕ, (∀ (board : fin m → fin n → ℤ), 
  (∀ i, (∏ j, board i j) = -1) ∧ 
  (∀ j, (∏ i, board i j) = -1) →
  count) ∧ 
  (
  (m % 2 = n % 2 → count = 2^((m-1)*(n-1))) ∧ 
  (m % 2 ≠ n % 2 → count = 0)
  ) := 
sorry

end fill_board_count_l346_346780


namespace bases_divisibility_rules_l346_346799

theorem bases_divisibility_rules (A : ℕ) : 
  (∃ n : ℤ, A = 12 * n + 5) ↔ 
  (let k := (A - 1) / 4 in (A = 4 * k + 1 ∧ (8 * k + 1) % 3 = 0)) := 
by {
  sorry
}

end bases_divisibility_rules_l346_346799


namespace ratio_S7_S3_l346_346697

variable {a_n : ℕ → ℕ} -- Arithmetic sequence {a_n}
variable (S_n : ℕ → ℕ) -- Sum of the first n terms of the arithmetic sequence

-- Conditions
def ratio_a2_a4 (a_2 a_4 : ℕ) : Prop := a_2 = 7 * (a_4 / 6)
def sum_formula (n a_1 d : ℕ) : ℕ := n * (2 * a_1 + (n - 1) * d) / 2

-- Proof goal
theorem ratio_S7_S3 (a_1 d : ℕ) (h : ratio_a2_a4 (a_1 + d) (a_1 + 3 * d)): 
  (S_n 7 = sum_formula 7 a_1 d) ∧ (S_n 3 = sum_formula 3 a_1 d) →
  (S_n 7 / S_n 3 = 2) :=
by
  sorry

end ratio_S7_S3_l346_346697


namespace smallest_four_digit_in_pascals_triangle_l346_346423

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346423


namespace smallest_four_digit_in_pascals_triangle_l346_346470

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346470


namespace min_c_value_l346_346761

theorem min_c_value 
  (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + 1 = b)
  (h6 : b + 1 = c)
  (h7 : c + 1 = d)
  (h8 : d + 1 = e)
  (h9 : ∃ k : ℕ, k ^ 2 = b + c + d)
  (h10 : ∃ m : ℕ, m ^ 3 = a + b + c + d + e) : 
  c = 675 := 
sorry

end min_c_value_l346_346761


namespace domain_of_func_l346_346383

def func_domain (x : ℝ) : Set ℝ := {y : ℝ | y = √(log (1 / 3) x)}

theorem domain_of_func :
  ∃ x, (√(log (1 / 3) x) : ℝ) ∧ (0 < x ∧ x ≤ 1) := sorry

end domain_of_func_l346_346383


namespace intersection_when_a_is_1_range_of_a_l346_346276

def setA : Set ℝ := { x | 1 ≤ 2^x ∧ 2^x ≤ 4 }
def setB (a : ℝ) : Set ℝ := { x | x - a > 0 }

theorem intersection_when_a_is_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

theorem range_of_a (a : ℝ) : (setA ∪ setB a = setB a) ↔ a < 0 :=
by
  sorry

end intersection_when_a_is_1_range_of_a_l346_346276


namespace smallest_four_digit_number_in_pascals_triangle_l346_346539

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346539


namespace solution_l346_346841

variables (P A G J K R p : ℕ) (z : ℕ)
-- All variables are positive integers
axiom (hP : P > 0)
axiom (hA : A > 0)
axiom (hG : G > 0)
axiom (hJ : J > 0)
axiom (hK : K > 0)
axiom (hR : R > 0)

-- Define the relationships given in the problem
axiom (h_p : P * p + A * (p / 2) + G * p = z)
axiom (h_jk : J * (p / 2) + K * p = z)
axiom (h_r : R * p = z)
-- P, A, G, J, K, R are distinct
axiom (distinct : P ≠ A ∧ P ≠ G ∧ P ≠ J ∧ P ≠ K ∧ P ≠ R ∧ A ≠ G ∧ A ≠ J ∧ A ≠ K ∧ A ≠ R ∧ G ≠ J ∧ G ≠ K ∧ G ≠ R ∧ J ≠ K ∧ J ≠ R ∧ K ≠ R)

theorem solution :
  R = 2 * G + A / 2 + P ∨ R = J / 2 + K := sorry

end solution_l346_346841


namespace perpendicular_sin_value_cos_alpha_value_l346_346746

-- Definitions based on given conditions
def a (x : ℝ) : ℝ × ℝ := (Real.sin (x + π / 6), 1)
def b (x : ℝ) : ℝ × ℝ := (4, 4 * Real.cos x - Real.sqrt 3)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Problem (I)
theorem perpendicular_sin_value (x : ℝ) (h : (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0) : 
  Real.sin (x + 4 * π / 3) = -1 / 4 :=
sorry

-- Problem (II)
theorem cos_alpha_value (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π / 2) (h : f (α - π / 6) = 2 * Real.sqrt 3) : 
  Real.cos α = (3 + Real.sqrt 21) / 8 :=
sorry

end perpendicular_sin_value_cos_alpha_value_l346_346746


namespace goose_eggs_laid_l346_346086

def hatch_ratio : ℝ := 2 / 3
def survive_first_month_ratio : ℝ := 3 / 4
def survive_first_year_ratio : ℝ := 2 / 5
def geese_survived_first_year : ℕ := 110

theorem goose_eggs_laid (E : ℝ) : 
  (2 / 5) * (3 / 4) * (2 / 3) * E = 110 → E = 2200 :=
by
  assume h : (2 / 5) * (3 / 4) * (2 / 3) * E = 110
  sorry

end goose_eggs_laid_l346_346086


namespace sum_of_coordinates_l346_346858

theorem sum_of_coordinates : 
  ∀ (f : ℝ → ℝ), f 2 = 1 → (1, 6) ∈ {p : ℝ × ℝ | p.2 = 3 * inverse f p.1} ∧ (1 + 6 = 7) :=
by
  intro f h
  use 1
  split
  { unfold inverse,
    have : inverse f 1 = 2 := sorry,
    rw [this, mul_comm],
    simp },
  { norm_num }
  sorry

end sum_of_coordinates_l346_346858


namespace find_P_l346_346871

noncomputable theory

-- Define the conditions as predicate logic
def is_digit (d : ℕ) : Prop := d ∈ {1, 2, 3, 4, 5}

def divisible_by (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def sum_digits (P Q R S T : ℕ) : ℕ := P + Q + R + S + T

theorem find_P (P Q R S T : ℕ) :
  is_digit P ∧ is_digit Q ∧ is_digit R ∧ is_digit S ∧ is_digit T ∧
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T ∧
  divisible_by (100 * P + 10 * Q + R) 3 ∧
  divisible_by (100 * Q + 10 * R + S) 5 ∧
  divisible_by (100 * R + 10 * S + T) 2 ∧
  divisible_by (sum_digits P Q R S T) 3
  → P = 1 :=
by sorry

end find_P_l346_346871


namespace new_bathroom_area_l346_346035

variable (area : ℕ) (width : ℕ) (extension : ℕ)

theorem new_bathroom_area (h1 : area = 96) (h2 : width = 8) (h3 : extension = 2) :
  (let orig_length := area / width;
       new_length := orig_length + extension;
       new_width := width + extension;
       new_area := new_length * new_width
   in new_area) = 140 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end new_bathroom_area_l346_346035


namespace fiona_considers_pairs_l346_346205

theorem fiona_considers_pairs : 
  ∀ (students : ℕ) (ignored_pairs : ℕ), students = 12 → ignored_pairs = 3 → (Nat.choose students 2 - ignored_pairs) = 63 :=
by
  intros students ignored_pairs h1 h2
  rw [h1, h2]
  exact (by decide : (Nat.choose 12 2 - 3) = 63)
  sorry

end fiona_considers_pairs_l346_346205


namespace irreducible_fractions_divisible_by_n_l346_346823

theorem irreducible_fractions_divisible_by_n (a n : ℤ) (h1 : 1 < a) (h2 : 1 < n) :
  ∃ k : ℕ, (({ m : ℕ | 1 ≤ m ∧ m < a ^ n ∧ gcd m (a ^ n - 1) = 1}).card = k * n) :=
begin
  sorry
end

end irreducible_fractions_divisible_by_n_l346_346823


namespace log_between_integers_and_sum_l346_346407

theorem log_between_integers_and_sum :
  ∃ (a b : ℤ), a < real.log 50 / real.log 10 ∧ real.log 50 / real.log 10 < b ∧ a + b = 3 :=
by {
  sorry
}

end log_between_integers_and_sum_l346_346407


namespace curve_is_degenerate_circle_l346_346682

theorem curve_is_degenerate_circle (r θ : ℝ) (h : r = 1 / (1 - sin θ)) :
  (∃ p : ℝ × ℝ, ∀ (x y : ℝ), (x, y) = p → x = 0 ∧ y = 1) :=
sorry

end curve_is_degenerate_circle_l346_346682


namespace circle_eq_line_eq_l346_346220

-- Define the circle passing through point A(0, 3) with center on negative half of y-axis and radius 5
def circleC (x y : ℝ) : Prop := ∃ b : ℝ, b < 0 ∧ (x^2 + (y - b)^2 = 25)

-- Define the line passing through point M(-3, -3)
def line_l (k : ℝ) (x y : ℝ) : Prop := y + 3 = k * (x + 3)

-- Prove the standard equation of circle C
theorem circle_eq : circleC 0 3 → ∃ b : ℝ, b = -2 ∧ (∀ x y : ℝ, circleC x y ↔ x^2 + (y + 2)^2 = 25) :=
by
  intro h
  sorry

-- Define the property of line l that the chord it intercepts has length 4√5
def chord_length (x y : ℝ) : Prop := |x ≈ (k * y)| -- Placeholder for the actual equation derivable from conditions

-- Prove the equations of line l
theorem line_eq : ∃ k : ℝ, (line_l k (-3) (-3) ∧ chord_length (x, y)) → 
                       (∀ k, k = -1/2 ∨ k = 2 ∧ ((x + 2 * y + 9 = 0) ∨ (2 * x - y + 3 = 0))) :=
by
  intro h
  sorry

end circle_eq_line_eq_l346_346220


namespace union_of_M_and_N_l346_346744

open Set

theorem union_of_M_and_N :
  let M := {x : ℝ | x ≥ -1}
  let N := {x : ℝ | -real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2}
  M ∪ N = {x : ℝ | x ≥ -real.sqrt 2} :=
by
  sorry

end union_of_M_and_N_l346_346744


namespace intersection_point_in_AB_l346_346695

def A (p : ℝ × ℝ) : Prop := p.snd = 2 * p.fst - 1
def B (p : ℝ × ℝ) : Prop := p.snd = p.fst + 3

theorem intersection_point_in_AB : (4, 7) ∈ {p : ℝ × ℝ | A p} ∩ {p : ℝ × ℝ | B p} :=
by
  sorry

end intersection_point_in_AB_l346_346695


namespace tetrahedron_min_g_l346_346019

def g (A B C D X : Point) : ℝ := 
  dist A X + dist B X + dist C X + dist D X

theorem tetrahedron_min_g (A B C D : Point) (dist_AD : dist A D = 30) (dist_BC : dist B C = 30)
  (dist_AC : dist A C = 46) (dist_BD : dist B D = 46) (dist_AB : dist A B = 50) (dist_CD : dist C D = 50) :
  ∃ X, g A B C D X = 32 * real.sqrt 10 := 
sorry


end tetrahedron_min_g_l346_346019


namespace housewife_money_left_l346_346607

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end housewife_money_left_l346_346607


namespace odd_function_a_eq_1_l346_346763

theorem odd_function_a_eq_1 (a : ℝ) (h_odd : ∀ x : ℝ, (exp x + a * exp (-x)) * sin x = - (exp (-x) + a * exp x) * sin (-x)) 
: a = 1 :=
sorry

end odd_function_a_eq_1_l346_346763


namespace compute_y_l346_346756

theorem compute_y (y : ℝ) (h : log 3 (y^3) + log (1/3) y = 6) : y = 27 :=
sorry

end compute_y_l346_346756


namespace find_missing_number_l346_346665

theorem find_missing_number : 
  (let number := 11 in number + Real.sqrt (-4 + (6 * 4) / 3) = 13) := 
by
  let number := 11
  have h : number + Real.sqrt (-4 + (6 * 4) / 3) = 13 := sorry
  exact h

end find_missing_number_l346_346665


namespace inclination_angle_l346_346235

theorem inclination_angle (α : ℝ) (t : ℝ) (h : 0 < α ∧ α < π / 2) :
  let x := 1 + t * Real.cos (α + 3 * π / 2)
  let y := 2 + t * Real.sin (α + 3 * π / 2)
  ∃ θ, θ = α + π / 2 := by
  sorry

end inclination_angle_l346_346235


namespace magnitude_of_complex_multiplication_l346_346666

-- Define the initial complex number z and the real multiplier c
def z : ℂ := 3 - 2 * complex.I
def c : ℝ := -1 / 3
-- Define the product of z and c
def w : ℂ := c * z

-- State that the magnitude of w is equal to sqrt(13) divided by 3
theorem magnitude_of_complex_multiplication : complex.abs w = real.sqrt 13 / 3 :=
by
  sorry

end magnitude_of_complex_multiplication_l346_346666


namespace divide_weights_n_99_not_divide_weights_n_98_l346_346413

theorem divide_weights_n_99 : 
  ∃ (A B : finset ℕ), A ∪ B = finset.range 100 ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

theorem not_divide_weights_n_98 : 
  ¬ ∃ (A B : finset ℕ), A ∪ B = finset.range 99 ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end divide_weights_n_99_not_divide_weights_n_98_l346_346413


namespace functional_equation_solution_l346_346239

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (h : ∀ x : ℝ, f(x + 1) = 2 * f x) : f = (λ x, 2^x) :=
sorry

end functional_equation_solution_l346_346239


namespace inequality_solution_l346_346182

noncomputable def solveInequality (x : ℝ) : Prop :=
  (Real.cbrt x) + (3 / ((Real.cbrt x) + 4)) ≤ 0

theorem inequality_solution :
    { x : ℝ | solveInequality x } = set.Ioo (-27 : ℝ) (-1 : ℝ) :=
  sorry

end inequality_solution_l346_346182


namespace smallest_integer_value_of_x_l346_346905

theorem smallest_integer_value_of_x (x : ℤ) (h : 7 + 3 * x < 26) : x = 6 :=
sorry

end smallest_integer_value_of_x_l346_346905


namespace set_difference_correct_l346_346263

-- Define the sets A and B
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}

-- Define the set difference A - B
def A_minus_B : Set ℤ := {x | x ∈ A ∧ x ∉ B} -- This is the operation A - B

-- The theorem stating the required proof
theorem set_difference_correct : A_minus_B = {1, 3, 9} :=
by {
  -- Proof goes here; however, we have requested no proof, so we put sorry.
  sorry
}

end set_difference_correct_l346_346263


namespace vehicle_speeds_l346_346592

theorem vehicle_speeds (x : ℕ) : 
  (∀ (t : ℕ), (384 = 4 * (2 * x + 8))) → (x = 44) ∧ (x + 8 = 52) :=
by
  intro h
  have hx : 8 * x + 32 = 384 := by 
    rw [←nat.mul_add_8, mul_comm]
    apply h
  have h_sum := congr_arg (λ a, a - 32) hx
  rw [384 - 32] at h_sum
  have h_div := congr_arg (λ a, a / 8) h_sum
  rw [352 / 8] at h_div
  rwa h_div at h
  exact sorry

end vehicle_speeds_l346_346592


namespace smallest_four_digit_in_pascal_l346_346510

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346510


namespace smallest_four_digit_in_pascals_triangle_l346_346483

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346483


namespace smallest_four_digit_in_pascal_l346_346503

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346503


namespace sum_of_natural_numbers_l346_346100

noncomputable def alpha : ℝ :=
  (-1 + Real.sqrt 29) / 2

theorem sum_of_natural_numbers (n : ℕ) :
  (α > 2) →
  (∀ k : ℕ, k ≥ 1 → Irrational (α^k)) →
  ∃ (x : ℕ → ℕ), (∀ k : ℕ, x k ≤ 6) ∧ (n = ∑ k in Finset.range (n+1), x k * α^k) := by
  sorry

end sum_of_natural_numbers_l346_346100


namespace orthocenter_circumcircle_constant_property_l346_346327

variables {α : Type*} [normed_field α] [inner_product_space α] {a a' b b' c c' R' k : α}
variable (A B C H Q : α)

/-- Given that H is the orthocenter of triangle ABC, Q is any point on the circumcircle, 
and the transformed side lengths are a' = ka, b' = kb, c' = kc, we need to prove that 
QA^2 + QB^2 + QC^2 - QH^2 is a constant value in terms of the side lengths and circumradius R'. 
The constant value is k^2a^2 + k^2b^2 + k^2c^2 - 4R'^2. --/
theorem orthocenter_circumcircle_constant_property
  (h_H : H = A + B + C)
  (h_Q : ∥Q∥ = R')
  (h_a' : a' = k * a)
  (h_b' : b' = k * b)
  (h_c' : c' = k * c) :
  ∥Q - A∥^2 + ∥Q - B∥^2 + ∥Q - C∥^2 - ∥Q - H∥^2 = k^2 * a^2 + k^2 * b^2 + k^2 * c^2 - 4 * R'^2 :=
sorry

end orthocenter_circumcircle_constant_property_l346_346327


namespace exists_natural_sum_l346_346096

noncomputable def exists_alpha (α : ℝ) : Prop :=
  α > 2 ∧ ∀ k : ℕ, k ≥ 1 → irrational (α ^ k)

theorem exists_natural_sum (α : ℝ) (hα : exists_alpha α) :
  ∀ n : ℕ, ∃ (a b : ℕ) (k : ℕ), (b ≤ 6) ∧ (n = a + b * (α ^ k)) :=
by sorry

end exists_natural_sum_l346_346096


namespace log_inequality_exp_inequality_l346_346234

variable (a : ℝ) (h : 0 < a ∧ a < 1)

theorem log_inequality (h : 0 < a ∧ a < 1) : Real.log a (1 + a) > Real.log a (1 + (1 / a)) :=
sorry

theorem exp_inequality (h : 0 < a ∧ a < 1) : a^(1 + a) > a^(1 + (1 / a)) :=
sorry

end log_inequality_exp_inequality_l346_346234


namespace yard_length_eq_250_l346_346311

noncomputable def number_of_trees : ℕ := 26
noncomputable def distance_between_trees : ℕ := 10
noncomputable def number_of_gaps := number_of_trees - 1
noncomputable def length_of_yard := number_of_gaps * distance_between_trees

theorem yard_length_eq_250 : 
  length_of_yard = 250 := 
sorry

end yard_length_eq_250_l346_346311


namespace sequence_sum_zero_l346_346667

-- Define the sequence as a function
def seq (n : ℕ) : ℤ :=
  if (n-1) % 8 < 4
  then (n+1) / 2
  else - (n / 2)

-- Define the sum of the sequence up to a given number
def seq_sum (m : ℕ) : ℤ :=
  (Finset.range (m+1)).sum (λ n => seq n)

-- The actual problem statement
theorem sequence_sum_zero : seq_sum 2012 = 0 :=
  sorry

end sequence_sum_zero_l346_346667


namespace r_value_l346_346810

-- Definitions and given conditions
variables {A B C D E F : Type}

-- Assume ABCD is a rectangle
axiom ABCD_is_rectangle 
    (A B C D : Type) 
    (AD : A → D)
    (BC : B → C)
    (AC : A → C) 
    (BD : B → D) 
    (F : Type) 
    (intersection_F_AC_BD : F = intersection_of_diagonals AC BD) 
    : is_rectangle A B C D

-- Definitions of angles
def angle_CDE (A B C D E : Type): ℝ := 180
def angle_DCE (A B C D E : Type): ℝ := 180
def angle_BFA (A B C D E F : Type): ℝ := 90
def angle_AFE (A B C D E F : Type): ℝ := 90

-- Definitions of degree-sum S and S'
def S (A B C D E : Type) := angle_CDE A B C D E + angle_DCE A B C D E
def S' (A B C D E F : Type) := angle_BFA A B C D E F + angle_AFE A B C D E F

-- The final proof problem
theorem r_value (A B C D E F : Type)
    (AD : A → D)
    (BC : B → C)
    (AC : A → C)
    (BD : B → D)
    (intersection_F_AC_BD : F = intersection_of_diagonals AC BD)
    (ABCD_cond : ABCD_is_rectangle A B C D AD BC AC BD intersection_F_AC_BD)
    : S A B C D E / S' A B C D E F = 2 :=
by
  sorry

end r_value_l346_346810


namespace first_player_strategic_win_l346_346892

-- Definition of the game setup
def initial_piles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- The end condition of the game
def game_ends (piles : List ℕ) : Prop := piles.sum = 3

-- Winning condition for the second player
def second_player_wins (piles : List ℕ) : Prop := game_ends piles ∧ (count (λ n, n = 1) piles = 3)

-- Winning condition for the first player
def first_player_wins (initial_piles : List ℕ) : Prop :=
  (∀ piles : List ℕ, game_ends piles → ¬ second_player_wins piles)

theorem first_player_strategic_win : first_player_wins initial_piles := by
  sorry

end first_player_strategic_win_l346_346892


namespace problem1_problem2_l346_346107

-- Problem 1 equivalent proof problem
theorem problem1 : 
  (Real.sqrt 3 * Real.sqrt 6 - (Real.sqrt (1 / 2) - Real.sqrt 8)) = (9 * Real.sqrt 2 / 2) :=
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2 (x : Real) (hx : x = Real.sqrt 5) : 
  ((1 + 1 / x) / ((x^2 + x) / x)) = (Real.sqrt 5 / 5) :=
by
  sorry

end problem1_problem2_l346_346107


namespace find_sequences_l346_346255

-- Definition of f(x) and its conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 2 + b * x
def slope_of_tangent (x : ℝ) (a b : ℝ) : ℝ := 2 * a * x + b

-- Conditions
variable (a b : ℝ)
axiom passes_through_point : f (-1) a b = 0
axiom tangent_slope_at_neg1 : slope_of_tangent (-1) a b = -1

-- Sequence and Sum of Sequence conditions
noncomputable def S (n : ℕ) (a b : ℝ) : ℝ := f n a b
noncomputable def a_n (n : ℕ) := S n a b - S (n-1) a b
noncomputable def T_n (n : ℕ) : ℝ := (1 / (a_n n * a_n (n + 1)))

-- Proof goal
theorem find_sequences (n : ℕ) (h : 0 < n) : 
  (∃ a b : ℝ, f (-1) a b = 0 ∧ slope_of_tangent (-1) a b = -1 ∧ 
  (∀ n : ℕ, S n a b = n ^ 2 + n ∧ a_n n = 2 * n) ∧ T_n n = n / (4 * n + 4)) :=
sorry

end find_sequences_l346_346255


namespace units_digit_35_pow_7_plus_93_pow_45_l346_346089

-- Definitions of units digit calculations for the specific values
def units_digit (n : ℕ) : ℕ := n % 10

def units_digit_35_pow_7 : ℕ := units_digit (35 ^ 7)
def units_digit_93_pow_45 : ℕ := units_digit (93 ^ 45)

-- Statement to prove that the sum of the units digits is 8
theorem units_digit_35_pow_7_plus_93_pow_45 : 
  units_digit (35 ^ 7) + units_digit (93 ^ 45) = 8 :=
by 
  sorry -- proof omitted

end units_digit_35_pow_7_plus_93_pow_45_l346_346089


namespace shortest_distance_parabola_to_point_l346_346204

theorem shortest_distance_parabola_to_point :
  let P := (λ a : ℝ, (a^2 / 4, a))
  let distance (P1 P2 : ℝ × ℝ) := real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)
  let parabola := (λ y : ℝ, (y^2 / 4, y))
  ∀ p : ℝ × ℝ,
    p ∈ set.range parabola →
    distance p (8, 16) = 2 * real.sqrt 34 :=
begin
  sorry
end

end shortest_distance_parabola_to_point_l346_346204


namespace polynomial_h_solution_l346_346855

theorem polynomial_h_solution (h : Polynomial ℝ → Polynomial ℝ) 
  (f : Polynomial ℝ → Polynomial ℝ) 
  (Hf : ∀ x, f(x) = x^2)
  (Hcond : ∀ x, f(h(x)) = 9 * x^2 - 6 * x + 1) :
  (h = λ x, 3 * x - 1) ∨ (h = λ x, -3 * x + 1) :=
by
  sorry

end polynomial_h_solution_l346_346855


namespace trajectory_of_Q_l346_346706

/-- Let P(m, n) be a point moving on the circle x^2 + y^2 = 2.
     The trajectory of the point Q(m+n, 2mn) is y = x^2 - 2. -/
theorem trajectory_of_Q (m n : ℝ) (hyp : m^2 + n^2 = 2) : 
  ∃ x y : ℝ, x = m + n ∧ y = 2 * m * n ∧ y = x^2 - 2 :=
by
  sorry

end trajectory_of_Q_l346_346706


namespace car_travel_distance_approx_l346_346593

-- Define the given conditions
def tire_diameter : ℝ := 10
def tire_revolutions : ℝ := 1008.2442067736184
def pi_approx : ℝ := 3.14159 -- using the given approximation for π

-- Define the circumference formula
def circumference (d : ℝ) : ℝ := pi_approx * d

-- Calculate the quarter-mile distance
def distance_in_inches (d : ℝ) (n : ℝ) : ℝ := n * circumference d

-- Convert distance from inches to feet
def distance_in_feet (distance_inches : ℝ) : ℝ := distance_inches / 12

-- Convert distance from feet to miles
def distance_in_miles (distance_feet : ℝ) : ℝ := distance_feet / 5280

def distance_traveled : ℝ := distance_in_miles (distance_in_feet (distance_in_inches tire_diameter tire_revolutions))

theorem car_travel_distance_approx : distance_traveled ≈ 0.5 :=
by
  sorry

end car_travel_distance_approx_l346_346593


namespace intersection_of_sets_l346_346713

theorem intersection_of_sets :
  let A := {x : ℝ | -2 ≤ x ∧ x < 3}
  let B := {x : ℝ | x < -1}
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} :=
begin
  -- proof goes here; omitted according to instructions
  sorry
end

end intersection_of_sets_l346_346713


namespace smallest_four_digit_in_pascals_triangle_l346_346475

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346475


namespace mean_equality_l346_346393

theorem mean_equality (x : ℝ) 
  (h : (7 + 9 + 23) / 3 = (16 + x) / 2) : 
  x = 10 := 
sorry

end mean_equality_l346_346393


namespace a_formula_b_formula_no_m_exists_l346_346796

def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * a (n - 1) + 3

def S (n : ℕ) : ℤ :=
  n * (b 1 + b n) / 2

def b (n : ℕ) : ℕ :=
  2

def c (n : ℕ) : ℤ :=
  (a n + 3 * b n) * if n % 2 = 0 then 1 else -1

def T (n : ℕ) : ℤ :=
  ∑ i in range n, c (i + 1)

theorem a_formula (n : ℕ) : a n = 2^n - 3 := 
sorry

theorem b_formula (n : ℕ) : b n = n + 1 :=
sorry

theorem no_m_exists : ¬ ∃ m : ℕ, T m = 2023 :=
sorry

end a_formula_b_formula_no_m_exists_l346_346796


namespace quadratic_has_positive_min_value_l346_346178

theorem quadratic_has_positive_min_value :
  ∃ x : ℝ, 3 * x^2 - 9 * x + 2 = f(1.5) ∧ 1.5 > 0 :=
by
  sorry

end quadratic_has_positive_min_value_l346_346178


namespace concurrency_of_lines_l346_346326

-- Definitions of the relevant geometric entities
variables {A B C L_a L_b L_c K_a K_b K_c : Type*}

-- Assumptions based on the problem conditions
variable [triangle ABC : Triangle A B C]
variable [angle_bisector AL_a : AngleBisector A L_a]
variable [angle_bisector BL_b : AngleBisector B L_b]
variable [angle_bisector CL_c : AngleBisector C L_c]
variable [tangent_circumcircle ABC B C K_a : TangentCircumcircle B C K_a]
variable [tangent_circumcircle ABC A C K_b : TangentCircumcircle A C K_b]
variable [tangent_circumcircle ABC A B K_c : TangentCircumcircle A B K_c]

-- Proof goal
theorem concurrency_of_lines :
  Concurrent (Line K_a L_a) (Line K_b L_b) (Line K_c L_c) :=
sorry

end concurrency_of_lines_l346_346326


namespace sqrt_of_square_is_identity_l346_346758

variable {a : ℝ} (h : a > 0)

theorem sqrt_of_square_is_identity (h : a > 0) : Real.sqrt (a^2) = a := 
  sorry

end sqrt_of_square_is_identity_l346_346758


namespace n_power_of_3_l346_346339

theorem n_power_of_3 (n : ℕ) (h_prime : Prime (4^n + 2^n + 1)) : ∃ k : ℕ, n = 3^k := 
sorry

end n_power_of_3_l346_346339


namespace find_a_l346_346253

def f (x : ℝ) : ℝ :=
if x < 1 then log (2, 1 - x) + 1 else x⁻²

theorem find_a (a : ℝ) (h : f a = 3) : a = -3 :=
by sorry

end find_a_l346_346253


namespace min_shirts_to_save_money_l346_346950

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 60 + 11 * x < 20 + 15 * x ∧ (∀ y : ℕ, 60 + 11 * y < 20 + 15 * y → y ≥ x) ∧ x = 11 :=
by
  sorry

end min_shirts_to_save_money_l346_346950


namespace integer_solutions_of_linear_diophantine_eq_l346_346584

theorem integer_solutions_of_linear_diophantine_eq 
  (a b c : ℤ)
  (h_gcd : Int.gcd a b = 1)
  (x₀ y₀ : ℤ)
  (h_particular_solution : a * x₀ + b * y₀ = c) :
  ∀ x y : ℤ, (a * x + b * y = c) → ∃ (k : ℤ), (x = x₀ + k * b) ∧ (y = y₀ - k * a) := 
by
  sorry

end integer_solutions_of_linear_diophantine_eq_l346_346584


namespace intersection_A_B_subset_A_B_l346_346275

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def set_B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

-- Problem 1: Prove A ∩ B when a = -1
theorem intersection_A_B (a : ℝ) (h : a = -1) : set_A a ∩ set_B = {x | 1 / 2 < x ∧ x < 2} :=
sorry

-- Problem 2: Find the range of a such that A ⊆ B
theorem subset_A_B (a : ℝ) : (-1 < a ∧ a ≤ 1) ↔ (set_A a ⊆ set_B) :=
sorry

end intersection_A_B_subset_A_B_l346_346275


namespace smallest_four_digit_number_in_pascals_triangle_l346_346464

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346464


namespace Arun_weight_lower_limit_l346_346150

theorem Arun_weight_lower_limit :
  ∀ (W : ℝ), (L : ℝ),
  (L < W ∧ W < 72) ∧
  (60 < W ∧ W < 70) ∧
  (W ≤ 69) ∧
  (W = 68) →
  L = 67 :=
by
  intros W L h
  sorry

end Arun_weight_lower_limit_l346_346150


namespace smallest_four_digit_number_in_pascals_triangle_l346_346549

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346549


namespace tv_show_duration_l346_346346

theorem tv_show_duration (total_airing_time_in_hours : ℝ) (num_commercials : ℕ) (commercial_duration_in_minutes : ℝ) :
  num_commercials = 3 → commercial_duration_in_minutes = 10 → total_airing_time_in_hours = 1.5 →
  (total_airing_time_in_hours * 60 - num_commercials * commercial_duration_in_minutes) / 60 = 1 := 
by {
  intros,
  simp,
  sorry
}

end tv_show_duration_l346_346346


namespace nails_painted_purple_l346_346322

variable (P S : ℕ)

theorem nails_painted_purple :
  (P + 8 + S = 20) ∧ ((8 / 20 : ℚ) * 100 - (S / 20 : ℚ) * 100 = 10) → P = 6 :=
by
  sorry

end nails_painted_purple_l346_346322


namespace train_crosses_platform_in_39_seconds_l346_346922

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

end train_crosses_platform_in_39_seconds_l346_346922


namespace find_angle_C_find_area_l346_346698

open Real

-- Definition of the problem conditions and questions

-- Condition: Given a triangle and the trigonometric relationship
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Trigonometric identity provided in the problem
axiom trig_identity : (sqrt 3) * c / (cos C) = a / (cos (3 * π / 2 + A))

-- First part of the problem
theorem find_angle_C (h1 : sqrt 3 * c / cos C = a / cos (3 * π / 2 + A)) : C = π / 6 :=
sorry

-- Second part of the problem
noncomputable def area_of_triangle (a b C : ℝ) : ℝ := 1 / 2 * a * b * sin C

variables {c' b' : ℝ}
-- Given conditions for the second question 
axiom condition_c_a : c' / a = 2
axiom condition_b : b' = 4 * sqrt 3

-- Definitions to align with the given problem
def c_from_a (a : ℝ) : ℝ := 2 * a

-- The final theorem for the second part
theorem find_area (hC : C = π / 6) (hc : c_from_a a = c') (hb : b' = 4 * sqrt 3) :
  area_of_triangle a b' C = 2 * sqrt 15 - 2 * sqrt 3 :=
sorry

end find_angle_C_find_area_l346_346698


namespace coefficient_of_z_first_eq_l346_346732

theorem coefficient_of_z_first_eq (x y z : ℝ)
  (h1 : 6 * x - 5 * y + z = 22 / 3)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  ∃ c, (6 * x - 5 * y + z = 22 / 3) ∧ c = 1 :=
by
  use 1
  sorry

end coefficient_of_z_first_eq_l346_346732


namespace no_valid_x_l346_346875

theorem no_valid_x (x y : ℝ) (h : y = 2 * x) : ¬(3 * y ^ 2 - 2 * y + 5 = 2 * (6 * x ^ 2 - 3 * y + 3)) :=
by
  sorry

end no_valid_x_l346_346875


namespace nested_H_value_l346_346620

def H : ℝ → ℝ := sorry

theorem nested_H_value :
  H 2 = -4 ∧
  H (-4) = 7 ∧
  H 7 = 7 →
  H (H (H (H (H 2)))) = 7 :=
by
  intro h
  cases h with h1 h23
  cases h23 with h2 h3
  sorry

end nested_H_value_l346_346620


namespace smallest_x_l346_346565

theorem smallest_x (x : ℕ) (h : 67 * 89 * x % 35 = 0) : x = 35 := 
by sorry

end smallest_x_l346_346565


namespace smallest_four_digit_in_pascal_l346_346502

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346502


namespace range_of_x_l346_346400

theorem range_of_x (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
  sorry

end range_of_x_l346_346400


namespace tv_show_duration_l346_346345

theorem tv_show_duration (total_airing_time_in_hours : ℝ) (num_commercials : ℕ) (commercial_duration_in_minutes : ℝ) :
  num_commercials = 3 → commercial_duration_in_minutes = 10 → total_airing_time_in_hours = 1.5 →
  (total_airing_time_in_hours * 60 - num_commercials * commercial_duration_in_minutes) / 60 = 1 := 
by {
  intros,
  simp,
  sorry
}

end tv_show_duration_l346_346345


namespace total_fishes_caught_l346_346835

def melanieCatches : ℕ := 8
def tomCatches : ℕ := 3 * melanieCatches
def totalFishes : ℕ := melanieCatches + tomCatches

theorem total_fishes_caught : totalFishes = 32 := by
  sorry

end total_fishes_caught_l346_346835


namespace find_a_l346_346244

noncomputable def f (a x : ℝ) : ℝ := log (x - a + 1) / log 2

theorem find_a (a : ℝ) 
  (h1: ∀ x y : ℝ, f a x = f a (-x)) 
  (h2: ∀ x : ℝ, log (-x - a) = log (x - a + 1))
  (h3: f 2 3 = 1) 
  : a = 2 := 
sorry

end find_a_l346_346244


namespace parallelogram_area_l346_346890

theorem parallelogram_area :
  let z1 := Complex.mk (Real.sqrt 82 / 2) (Real.sqrt 14 / 2)
  ∧ let z2 := Complex.mk (-Real.sqrt 82 / 2) (-Real.sqrt 14 / 2)
  ∧ let w1 := Complex.mk (Real.sqrt 10 / 2) (Real.sqrt 10 / 2)
  ∧ let w2 := Complex.mk (-Real.sqrt 10 / 2) (-Real.sqrt 10 / 2)
  ∧ let a := abs ((z1 - w1) * (conj z1 - conj w1))
  in a = 2 * Real.sqrt 96 - 2 * Real.sqrt 2 := sorry

end parallelogram_area_l346_346890


namespace problem_I_problem_II_l346_346352

noncomputable def setA (a : ℝ) : set ℝ := {x : ℝ | (3 - a) / 4 < x ∧ x < (3 + a) / 4 ∧ a > 0}
noncomputable def setB : set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem problem_I (a : ℝ) :
  setA a = {x : ℝ | (3 - a) / 4 < x ∧ x < (3 + a) / 4 ∧ a > 0} ∧
  setB = {x : ℝ | -4 < x ∧ x < 2} :=
by
  sorry

theorem problem_II (a : ℝ) :
  (∀ x, x ∈ setA a → x ∈ setB) ↔ (0 < a ∧ a ≤ 5) :=
by
  sorry

end problem_I_problem_II_l346_346352


namespace smallest_four_digit_in_pascal_l346_346511

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346511


namespace find_n_l346_346301

theorem find_n (q d : ℕ) (hq : q = 25) (hd : d = 10) (h : 30 * q + 20 * d = 5 * q + n * d) : n = 83 := by
  sorry

end find_n_l346_346301


namespace smallest_four_digit_in_pascals_triangle_l346_346537

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346537


namespace find_f_prime_at_one_l346_346074

theorem find_f_prime_at_one (a b : ℝ)
  (h1 : ∀ x, f x = a * Real.exp x + b * x) 
  (h2 : f 0 = 1)
  (h3 : ∀ x, deriv f x = a * Real.exp x + b)
  (h4 : deriv f 0 = 0) :
  deriv f 1 = Real.exp 1 - 1 :=
by {
  sorry
}

end find_f_prime_at_one_l346_346074


namespace smallest_four_digit_in_pascal_l346_346524

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346524


namespace equivalent_modulo_l346_346853

theorem equivalent_modulo:
  123^2 * 947 % 60 = 3 :=
by
  sorry

end equivalent_modulo_l346_346853


namespace cube_without_lid_configurations_l346_346599

-- Introduce assumption for cube without a lid
structure CubeWithoutLid

-- Define the proof statement
theorem cube_without_lid_configurations : 
  ∃ (configs : Nat), (configs = 8) :=
by
  sorry

end cube_without_lid_configurations_l346_346599


namespace sum_50th_set_correct_l346_346363

noncomputable def sum_of_fiftieth_set : ℕ := 195 + 197

theorem sum_50th_set_correct : sum_of_fiftieth_set = 392 :=
by 
  -- The proof would go here
  sorry

end sum_50th_set_correct_l346_346363


namespace perpendicular_condition_l346_346337

-- Define lines and plane
variables {l m n : Type} [linear_ordered_field ℝ] (α : set ℝ) 
  (line_l : l)
  (line_m : m)
  (line_n : n)
  -- Assuming m and n are within plane α
  (m_in_plane_alpha : ∀ (p : ℝ), p ∈ line_m → p ∈ α)
  (n_in_plane_alpha : ∀ (p : ℝ), p ∈ line_n → p ∈ α)
  -- Line l is perpendicular to plane α
  (l_perp_alpha : ∀ (p1 p2 : ℝ), p1 ∈ α ∧ p2 ∈ α → (l ⊥ α))
  -- Lines l perpendicular to m and n
  (l_perp_m_n : ∀ (p1 p2 : ℝ), (p1 ∈ line_m → p2 ∈ line_n) → (l ⊥ m ∧ l ⊥ n))

-- The proof statement
theorem perpendicular_condition (l m n : Type) [linear_ordered_field ℝ] (α : set ℝ)
  (line_l : l)
  (line_m : m)
  (line_n : n)
  (m_in_plane_alpha : ∀ (p : ℝ), p ∈ line_m → p ∈ α)
  (n_in_plane_alpha : ∀ (p : ℝ), p ∈ line_n → p ∈ α)
  (l_perp_alpha : ∀ (p1 p2 : ℝ), p1 ∈ α ∧ p2 ∈ α → (l ⊥ α))
  (l_perp_m_n : ∀ (p1 p2 : ℝ), (p1 ∈ line_m → p2 ∈ line_n) → (l ⊥ m ∧ l ⊥ n)) :
  (∀ (p1 p2 : ℝ), p1 ∈ α ∧ p2 ∈ α → (l ⊥ α)) →
  (∃ (m n : set ℝ), (m ∈ α ∧ n ∈ α ∧ l ⊥ m ∧ l ⊥ n) ) :=
sorry

end perpendicular_condition_l346_346337


namespace smallest_four_digit_number_in_pascals_triangle_l346_346451

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346451


namespace cotangent_tangent_difference_l346_346365

theorem cotangent_tangent_difference :
  ∃ (c30 s30 c15 s15 : ℝ),
    c30 = (Real.cos (π / 6)) ∧
    s30 = (Real.sin (π / 6)) ∧
    c15 = 0.9659 ∧
    s15 = 0.2588 ∧
    (Real.cot (π / 6) - Real.tan (π / 12)) = (Real.cos (π / 4)) / (Real.sin (π / 6) * 0.9659) :=
by
  -- Sorry is used here to skip the proof
  sorry

end cotangent_tangent_difference_l346_346365


namespace smallest_four_digit_in_pascals_triangle_l346_346531

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346531


namespace convex_ngon_diagonals_l346_346843

theorem convex_ngon_diagonals (n : ℕ) (n_pos : n > 2) (cn : ∃ (V : Type) (N : set V) (E : set (V × V)), 
   (|N| = n) ∧ (|E| = n * (n - 3) ÷ 2)) : 
    ¬∃ (D : set (V × V)), (|D| > n) ∧ (∀ (d1 d2 : (V × V)), d1 ≠ d2 → (d1.1 = d2.1 ∨ d1.2 = d2.2)) := sorry

end convex_ngon_diagonals_l346_346843


namespace grey_black_difference_l346_346018

theorem grey_black_difference :
  ∃ (brown black white grey : ℕ),
    brown = 4 ∧
    black = brown + 1 ∧
    white = 2 * brown ∧
    avg_weight (brown, black, white, grey) = 5 →
    black - grey = 2 :=
begin
  sorry
end

def avg_weight (weights : (ℕ, ℕ, ℕ, ℕ)) : ℕ :=
  (weights.1 + weights.2 + weights.3 + weights.4) / 4

end grey_black_difference_l346_346018


namespace option_A_is_translation_l346_346954

-- Define what constitutes a translation transformation
def is_translation (description : String) : Prop :=
  description = "Pulling open a drawer"

-- Define each option
def option_A : String := "Pulling open a drawer"
def option_B : String := "Viewing text through a magnifying glass"
def option_C : String := "The movement of the minute hand on a clock"
def option_D : String := "You and the image in a plane mirror"

-- The main theorem stating that option A is the translation transformation
theorem option_A_is_translation : is_translation option_A :=
by
  -- skip the proof, adding sorry
  sorry

end option_A_is_translation_l346_346954


namespace smallest_four_digit_in_pascals_triangle_l346_346488

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346488


namespace can_assemble_natural_sum_l346_346094

-- Definitions based on the problem conditions
def alpha : ℝ := (-1 + Real.sqrt 29) / 2
def is_natural (n : ℕ) : Prop := n ≥ 0

-- Conditions (no denomination used more than 6 times, and each is irrational except 1)
def valid_coins (k : ℕ) : Prop := (k = 0 ∨ irrational (alpha ^ k)) ∧ alpha > 2

-- Main statement to prove
theorem can_assemble_natural_sum (n : ℕ) (h_nat : is_natural n) : 
  (∀ k : ℕ, valid_coins k) → ∃ (coefficients : ℕ → ℕ), (∀ k : ℕ, coefficients k ≤ 6) ∧ 
  (n = ∑ (k : ℕ) in (Finset.range (n + 1)), (coefficients k) * (alpha ^ k)) :=
by
  sorry

end can_assemble_natural_sum_l346_346094


namespace dilation_image_l346_346872

open Complex

theorem dilation_image :
  ∀ (z_0 c : ℂ) (k : ℤ), z_0 = -1 - I → c = 2 + 3 * I → k = 3 → ∃ z : ℂ, z - c = k • (z_0 - c) ∧ z = -7 - 9 * I :=
by
  intros z_0 c k h_z0 h_c h_k
  use -7 - 9 * I
  have h : z_0 - c = -3 - 4 * I, by rw [h_z0, h_c]; ring
  rw [h, h_k]
  split; ring

end dilation_image_l346_346872


namespace max_min_values_a_minus_1_minimum_value_a_l346_346256

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 2

section
variable (a : ℝ)

-- Question (1)
def f_for_a_minus_1 := f x (-1)

theorem max_min_values_a_minus_1 :
  (∀ x ∈ Icc (-5 : ℝ) 5, f x (-1) ≤ 37) ∧ (∃ x ∈ Icc (-5 : ℝ) 5, f x (-1) = 37) ∧
  (∀ x ∈ Icc (-5 : ℝ) 5, 1 ≤ f x (-1)) ∧ (∃ x ∈ Icc (-5 : ℝ) 5, f x (-1) = 1) :=
by
  sorry -- proof not required as per instructions

-- Question (2)
theorem minimum_value_a (a : ℝ) :
  ∀ x ∈ Icc (-5 : ℝ) 5, 
    (f x a ≥ 
      if a < -5 then 27 + 10 * a
      else if -5 ≤ a ∧ a ≤ 5 then 2 - a^2
      else 27 - 10 * a) := 
by
  sorry -- proof not required as per instructions

end max_min_values_a_minus_1_minimum_value_a_l346_346256


namespace victor_percentage_of_marks_l346_346057

theorem victor_percentage_of_marks 
    (marks_obtained : ℝ) (max_marks : ℝ) 
    (h1 : marks_obtained = 405) 
    (h2 : max_marks = 450) : 
    (marks_obtained / max_marks) * 100 = 90 :=
by
  rw [h1, h2]
  simp
  norm_num
  sorry

end victor_percentage_of_marks_l346_346057


namespace smallest_n_condition_l346_346563

theorem smallest_n_condition (n : ℕ) (h : (sqrt n - sqrt (n - 1) < 0.01)) : n = 2501 :=
sorry

end smallest_n_condition_l346_346563


namespace smallest_positive_period_solve_for_a_l346_346251

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * sin (π / 2 - x) 
                                      - 2 * cos (π + x) * cos x + 2

-- Problem 1: Prove the smallest positive period of f(x) is π
theorem smallest_positive_period : ∀ x, f(x + π) = f(x) :=
  sorry

-- Problem 2: Given conditions f(A) = 4, b = 1, and area of triangle ABC is √3 / 2, prove a = √3
def triangle_area (a b c A : ℝ) : ℝ := 0.5 * b * c * sin A

def law_of_cosines (a b c A : ℝ) : ℝ :=
  b ^ 2 + c ^ 2 - 2 * b * c * cos A

theorem solve_for_a (A b : ℝ) (area : ℝ) (ha : f(A) = 4) (hb : b = 1) (harea : area = sqrt 3 / 2) : 
    ∃ a, a * a = law_of_cosines a b 2 A :=
  sorry

end smallest_positive_period_solve_for_a_l346_346251


namespace exists_natural_sum_l346_346097

noncomputable def exists_alpha (α : ℝ) : Prop :=
  α > 2 ∧ ∀ k : ℕ, k ≥ 1 → irrational (α ^ k)

theorem exists_natural_sum (α : ℝ) (hα : exists_alpha α) :
  ∀ n : ℕ, ∃ (a b : ℕ) (k : ℕ), (b ≤ 6) ∧ (n = a + b * (α ^ k)) :=
by sorry

end exists_natural_sum_l346_346097


namespace smallest_four_digit_in_pascal_l346_346519

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346519


namespace squirrel_stockpiling_days_l346_346834

/-- 
Mason opens the hood of his car and discovers that 2 busy squirrels have been using his engine 
compartment to store nuts. These squirrels have been stockpiling 30 nuts/day and one sleepy squirrel 
has been stockpiling 20 nuts/day, all for some days. There are 3200 nuts in Mason's car. For how 
many days have the squirrels been stockpiling nuts?
-/
theorem squirrel_stockpiling_days :
  let busy_squirrels := 2
  let busy_squirrels_rate := 30
  let sleepy_squirrel_rate := 20
  let total_nuts := 3200
  let busy_squirrels_nuts_per_day := busy_squirrels * busy_squirrels_rate
  let total_nuts_per_day := busy_squirrels_nuts_per_day + sleepy_squirrel_rate
  total_nuts / total_nuts_per_day = 40 :=
by
  let busy_squirrels := 2
  let busy_squirrels_rate := 30
  let sleepy_squirrel_rate := 20
  let total_nuts := 3200
  let busy_squirrels_nuts_per_day := busy_squirrels * busy_squirrels_rate
  let total_nuts_per_day := busy_squirrels_nuts_per_day + sleepy_squirrel_rate
  have nuts_per_day : total_nuts_per_day = 80 := by rfl
  show total_nuts / total_nuts_per_day = 40 from sorry

end squirrel_stockpiling_days_l346_346834


namespace triangles_from_decagon_l346_346991

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346991


namespace value_of_y_l346_346394

theorem value_of_y (y : ℤ) (h_neg : y < 0) 
  (h_median : median ({15, 42, 50, y, 18} : set ℤ) = mean ({15, 42, 50, y, 18} : set ℤ) - 6) : 
  y = -5 := 
by 
  sorry

end value_of_y_l346_346394


namespace cone_height_ratio_l346_346137

noncomputable def cone_base_circumference := 24 * Real.pi
noncomputable def original_height := 40
noncomputable def new_volume := 432 * Real.pi

def radius_from_circumference (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

def new_height (V r : ℝ) : ℝ :=
  (3 * V) / (Real.pi * r^2)

theorem cone_height_ratio :
  let r := radius_from_circumference cone_base_circumference,
      h := new_height new_volume r
  in h / original_height = (9 : ℝ) / 40 :=
by
  sorry

end cone_height_ratio_l346_346137


namespace pig_duck_ratio_l346_346410

theorem pig_duck_ratio (G C D P : ℕ)
(h₁ : G = 66)
(h₂ : C = 2 * G)
(h₃ : D = (G + C) / 2)
(h₄ : P = G - 33) :
  P / D = 1 / 3 :=
by {
  sorry
}

end pig_duck_ratio_l346_346410


namespace team_b_fraction_calls_l346_346080

noncomputable def fraction_calls_team_B (A B CA CB : ℝ) : ℝ :=
(A = (5/8) * B) → 
(CA = (7/5) * CB) →
(B * CB) / ((A * CA) + (B * CB))

theorem team_b_fraction_calls : 
  ∀ (A B CA CB : ℝ), 
  A = (5/8) * B → 
  CA = (7/5) * CB → 
  fraction_calls_team_B A B CA CB = 8 / 15 :=
by
  intros A B CA CB hA hCA
  rw [fraction_calls_team_B, hA, hCA]
  sorry

end team_b_fraction_calls_l346_346080


namespace intersection_of_sets_l346_346272

theorem intersection_of_sets :
  let M := {1, 2, 3, 4, 5}
  let N := {2, 4, 6, 8, 10}
  M ∩ N = {2, 4} :=
by 
  let M := {1, 2, 3, 4, 5}
  let N := {2, 4, 6, 8, 10}
  have h : M ∩ N = {2, 4} := sorry
  exact h

end intersection_of_sets_l346_346272


namespace scott_monthly_miles_l346_346362

theorem scott_monthly_miles :
  let miles_per_mon_wed := 3
  let mon_wed_days := 3
  let thur_fri_factor := 2
  let thur_fri_days := 2
  let weeks_per_month := 4
  let miles_mon_wed := miles_per_mon_wed * mon_wed_days
  let miles_thur_fri_per_day := thur_fri_factor * miles_per_mon_wed
  let miles_thur_fri := miles_thur_fri_per_day * thur_fri_days
  let miles_per_week := miles_mon_wed + miles_thur_fri
  let total_miles_in_month := miles_per_week * weeks_per_month
  total_miles_in_month = 84 := 
  by
    sorry

end scott_monthly_miles_l346_346362


namespace number_of_triangles_in_decagon_l346_346976

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346976


namespace smallest_four_digit_in_pascals_triangle_l346_346435

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346435


namespace value_a4_is_15_l346_346707

-- Define the sequence
def seq : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (seq n) + 1

-- State the theorem
theorem value_a4_is_15 : seq 4 = 15 := sorry

end value_a4_is_15_l346_346707


namespace trig_identity_solution_l346_346574

theorem trig_identity_solution (x : ℝ) (k : ℤ):
  (8.487 * (sin x ^ 2 - tan x ^ 2) / (cos x ^ 2 - cot x ^ 2) - tan x ^ 6 + tan x ^ 4 - tan x ^ 2 = 0) ↔
  (∃ k : ℤ, x = (π / 4) * (2 * k + 1)) :=
sorry

end trig_identity_solution_l346_346574


namespace least_positive_integer_with_seven_factors_l346_346065

theorem least_positive_integer_with_seven_factors : ∃ n : ℕ, n = 64 ∧ 
  ∃ a p : ℕ, a = 6 ∧ (nat.prime p) ∧ n = p^a :=
by
  sorry

end least_positive_integer_with_seven_factors_l346_346065


namespace train_length_l346_346610

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def distance_ahead : ℝ := 270
noncomputable def time_to_pass : ℝ := 39

noncomputable def jogger_speed_mps := jogger_speed_kmph * (1000 / 1) * (1 / 3600)
noncomputable def train_speed_mps := train_speed_kmph * (1000 / 1) * (1 / 3600)

noncomputable def relative_speed_mps := train_speed_mps - jogger_speed_mps

theorem train_length :
  let jogger_speed := 9 * (1000 / 3600)
  let train_speed := 45 * (1000 / 3600)
  let relative_speed := train_speed - jogger_speed
  let distance := 270
  let time := 39
  distance + relative_speed * time = 390 → relative_speed * time = 120 := by
  sorry

end train_length_l346_346610


namespace paisa_per_rupee_z_gets_l346_346628

theorem paisa_per_rupee_z_gets
  (y_share : ℝ)
  (y_per_x_paisa : ℝ)
  (total_amount : ℝ)
  (x_share : ℝ)
  (z_share : ℝ)
  (paisa_per_rupee : ℝ)
  (h1 : y_share = 36)
  (h2 : y_per_x_paisa = 0.45)
  (h3 : total_amount = 140)
  (h4 : x_share = y_share / y_per_x_paisa)
  (h5 : z_share = total_amount - (x_share + y_share))
  (h6 : paisa_per_rupee = (z_share / x_share) * 100) :
  paisa_per_rupee = 30 :=
by
  sorry

end paisa_per_rupee_z_gets_l346_346628


namespace GR_eq_GS_l346_346820

-- Definitions and conditions of the problem
variables (A B C Z Y G R S : Type) 
          
-- Assume all points and definitions related to the problem
variables [triangle ∆ABC : Triangle A B C] 
          [incircle : Incircle ∆ABC]
          [incircleTouchZ : incircle.Touches A B Z]
          [incircleTouchY : incircle.Touches A C Y]
          [G_Point : Intersection (Line B Y) (Line C Z) G]
          [parallelogram_1 : Parallelogram B C Y R]
          [parallelogram_2 : Parallelogram B C S Z]

-- The theorem to prove that GR = GS
theorem GR_eq_GS : Distance G R = Distance G S :=
by sorry -- Proof goes here

end GR_eq_GS_l346_346820


namespace problem_statement_l346_346675

open Nat

def prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem problem_statement : ∀ (x y z : ℕ), prime y ∧ (¬ y ∣ z) ∧ (¬ 3 ∣ z) ∧ (x^3 - y^3 = z^2) → (x, y, z) = (8, 7, 13) :=
by {
  intros x y z h,
  sorry -- the proof goes here
}

end problem_statement_l346_346675


namespace sum_of_first_nine_terms_l346_346299

theorem sum_of_first_nine_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 = 3 * a 3 - 6) : 
  (9 * (a 0 + a 8)) / 2 = 27 := 
sorry

end sum_of_first_nine_terms_l346_346299


namespace smallest_x_mod_61_smallest_positive_x_l346_346661

theorem smallest_x_mod_61 :
  (∃ x : ℕ, (3 * x)^2 + 3 * 58 * 3 * x + 58^2 % 61 = 0) ↔ ∃ x : ℕ, (3 * x + 58) ^ 2 % 61 = 0 :=
sorry

theorem smallest_positive_x :
  ∀ x : ℕ, ((3 * x + 58) % 61 = 0) → x = 1 :=
begin
  assume x,
  assume h : (3 * x + 58) % 61 = 0,
  have h_eq : 3 * x + 58 = 61 * k for some k : ℕ,
  sorry
end

end smallest_x_mod_61_smallest_positive_x_l346_346661


namespace count_valid_functions_l346_346179

def valid_functions_count : ℕ :=
  {f : ℤ → string // f.period 22 ∧ ∀ y, ¬ (f y = "green" ∧ f (y + 2) = "green")}

theorem count_valid_functions :
  (valid_functions_count : Type).card = 39601 :=
sorry

end count_valid_functions_l346_346179


namespace smallest_four_digit_in_pascals_triangle_l346_346481

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346481


namespace initial_markup_percentage_proof_l346_346941

-- Definition of the conditions
variables (W initial_price final_price initial_markup_percentage : ℝ)

-- Defining conditions from problem statement
def initial_price_eq : Prop := W + (initial_price - W) = 36
def final_price_eq : Prop := 2 * W = 36 + 4
def wholesale_price_eq : Prop := W = 20

-- The initial markup percentage calculation definition
def initial_markup_percentage_eq : Prop := initial_markup_percentage = ((36 - W) / W) * 100

-- The theorem to prove
theorem initial_markup_percentage_proof :
  initial_price_eq W initial_price
  ∧ final_price_eq W
  ∧ wholesale_price_eq W
  → initial_markup_percentage_eq W initial_price initial_markup_percentage :=
by
  sorry

end initial_markup_percentage_proof_l346_346941


namespace determine_s_l346_346658

theorem determine_s 
  (s : ℝ) 
  (h : (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
       6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30) : 
  s = 4 :=
by
  sorry

end determine_s_l346_346658


namespace gcd_90_270_l346_346683

theorem gcd_90_270 : Int.gcd 90 270 = 90 :=
by
  sorry

end gcd_90_270_l346_346683


namespace find_f_ln_1_over_3_l346_346254

noncomputable def f : ℝ → ℝ → ℝ := λ x a, (2^x)/(2^x + 1) + ax

theorem find_f_ln_1_over_3 (a : ℝ) (h : f (Real.log 3) a = 3) :
  f (Real.log (1/3)) a = -2 := by
  sorry

end find_f_ln_1_over_3_l346_346254


namespace length_DC_of_ABCD_l346_346671

open Real

structure Trapezoid (ABCD : Type) :=
  (AB DC : ℝ)
  (BC : ℝ := 0)
  (angleBCD angleCDA : ℝ)

noncomputable def given_trapezoid : Trapezoid ℝ :=
{ AB := 5,
  DC := 8 + sqrt 3, -- this is from the answer
  BC := 3 * sqrt 2,
  angleBCD := π / 4,   -- 45 degrees in radians
  angleCDA := π / 3 }  -- 60 degrees in radians

variable (ABCD : Trapezoid ℝ)

theorem length_DC_of_ABCD :
  ABCD.AB = 5 ∧
  ABCD.BC = 3 * sqrt 2 ∧
  ABCD.angleBCD = π / 4 ∧
  ABCD.angleCDA = π / 3 →
  ABCD.DC = 8 + sqrt 3 :=
sorry

end length_DC_of_ABCD_l346_346671


namespace sqrt_product_is_four_l346_346163

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end sqrt_product_is_four_l346_346163


namespace total_sticks_needed_l346_346007

theorem total_sticks_needed (simon_sticks gerry_sticks micky_sticks darryl_sticks : ℕ):
  simon_sticks = 36 →
  gerry_sticks = (2 * simon_sticks) / 3 →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  darryl_sticks = simon_sticks + gerry_sticks + micky_sticks + 1 →
  simon_sticks + gerry_sticks + micky_sticks + darryl_sticks = 259 :=
by
  intros h_simon h_gerry h_micky h_darryl
  rw [h_simon, h_gerry, h_micky, h_darryl]
  norm_num
  sorry

end total_sticks_needed_l346_346007


namespace hannah_books_per_stocking_l346_346279

theorem hannah_books_per_stocking
  (candy_canes_per_stocking : ℕ)
  (beanie_babies_per_stocking : ℕ)
  (num_kids : ℕ)
  (total_stuffers : ℕ)
  (books_per_stocking : ℕ) :
  candy_canes_per_stocking = 4 →
  beanie_babies_per_stocking = 2 →
  num_kids = 3 →
  total_stuffers = 21 →
  books_per_stocking = (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids →
  books_per_stocking = 1 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  simp at h5
  sorry

end hannah_books_per_stocking_l346_346279


namespace exist_1010_cities_l346_346598

-- Define the basic properties of cities and flight routes
variable (City : Type) (endpoint : City → City → Prop)

-- There are 2020 cities
constant total_cities : Finset City
axiom cities_cardinality : total_cities.card = 2020

-- For any 1009 cities, they have a unique common endpoint
axiom unique_common_endpoint_for_1009 : ∀ (S : Finset City), S.card = 1009 → ∃! (b : City), ∀ a ∈ S, endpoint a b

-- The theorem to prove: there exist 1010 cities such that if flight routes from any one city are removed,
-- the remaining 1009 cities still have a unique common endpoint.
theorem exist_1010_cities (City : Type) (endpoint : City → City → Prop) :
  ( ∃ (S : Finset City), S.card = 1010 ∧ ∀ (a ∈ S), ∃! (b : City), ∀ (c ∈ (S \ {a})), endpoint c b ) := sorry

end exist_1010_cities_l346_346598


namespace P_lt_Q_l346_346219

theorem P_lt_Q (x : ℝ) (hx : x > 0) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.sqrt (1 + x)) 
  (hQ : Q = 1 + x / 2) : P < Q := 
by
  sorry

end P_lt_Q_l346_346219


namespace greatest_n_less_than_1000_l346_346207

def h (y : ℕ) : ℕ :=
  if y % 3 ≠ 0 then 1
  else 3 ^ (Nat.find_greatest (λ j, 3^j ∣ y) (Nat.log 3 y))

def R_n (n : ℕ) : ℕ :=
  ∑ k in Finset.range (3^(n - 1)), h (3 * (k + 1))

theorem greatest_n_less_than_1000 (n : ℕ) : 
  n < 1000 → ∃ n = 333, ∃ k, k^3 = R_n n  :=
by
  sorry

end greatest_n_less_than_1000_l346_346207


namespace total_time_l346_346129

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l346_346129


namespace percent_difference_l346_346909

theorem percent_difference:
  let a := (1/10 : ℝ) * 7000 in
  let b := (1/1000 : ℝ) * 7000 in
  a - b = 693 :=
by
  sorry

end percent_difference_l346_346909


namespace fraction_transferred_out_l346_346567

def initial_students : ℕ := 160
def new_students : ℕ := 20
def end_year_students : ℕ := 120

theorem fraction_transferred_out :
    let total_students := initial_students + new_students in
    let students_transferred := total_students - end_year_students in
    ((students_transferred : ℚ) / total_students) = 1 / 3 := 
by
  sorry

end fraction_transferred_out_l346_346567


namespace g_inv_f_10_eq_l346_346759

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq (x : ℝ) : f_inv (g x) = x^4 - 1
axiom g_has_inv : has_inv g

theorem g_inv_f_10_eq : g_inv (f 10) = real.sqrt (real.sqrt 11) :=
by
  sorry

end g_inv_f_10_eq_l346_346759


namespace regression_and_estimation_l346_346013

def linear_regression {n : ℕ} (xs : Fin n → ℝ) (ys : Fin n → ℝ) : ℝ × ℝ :=
  let x_bar := ∑ i, xs i / n
  let y_bar := ∑ i, ys i / n
  let sum_xiyi := ∑ i, (xs i) * (ys i)
  let sum_xi_squared := ∑ i, (xs i) ^ 2
  let b := (sum_xiyi - n * x_bar * y_bar) / (sum_xi_squared - n * x_bar ^ 2)
  let a := y_bar - b * x_bar
  (b, a)

theorem regression_and_estimation :
  let xs := ![2, 3, 4, 5, 6] : Fin 5 → ℝ
  let ys := ![2.2, 3.8, 5.5, 6.5, 7] : Fin 5 → ℝ
  let (b, a) := linear_regression xs ys
  b = 1.23 ∧ a = 0.08 ∧ a + 10 * b = 12.38 := by
  let xs := ![2, 3, 4, 5, 6] : Fin 5 → ℝ
  let ys := ![2.2, 3.8, 5.5, 6.5, 7] : Fin 5 → ℝ
  let (b, a) := linear_regression xs ys
  have h_b : b = 1.23 := sorry
  have h_a : a = 0.08 := sorry
  have h_estimate : a + 10 * b = 12.38 := sorry
  exact ⟨h_b, h_a, h_estimate⟩

end regression_and_estimation_l346_346013


namespace power_function_evaluation_l346_346026

variables {α : Type*} [HasPow α ℕ] [DecidableEq α] [HasMul α] [HasAdd α] [HasOne α]

theorem power_function_evaluation (f : α → α) (x : α) (a : ℕ)
  (h : ∀ (x : α), f x = x ^ a)
  (h_point : f 3 = 9) :
  f 2 = 4 ∧ f (2 * x + 1) = 4 * x ^ 2 + 4 * x + 1 :=
by
  sorry

end power_function_evaluation_l346_346026


namespace find_z_l346_346581

-- Definitions from the problem statement
variables {x y z : ℤ}
axiom consecutive (h1: x = z + 2) (h2: y = z + 1) : true
axiom ordered (h3: x > y) (h4: y > z) : true
axiom equation (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : true

-- The proof goal
theorem find_z (h1: x = z + 2) (h2: y = z + 1) (h3: x > y) (h4: y > z) (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : z = 2 :=
by 
  sorry

end find_z_l346_346581


namespace row_min_equals_col_min_l346_346708

variable {α : Type} [LinearOrder α] {n : ℕ} (T : Fin n → Fin n → α)

def row_min (T : Fin n → Fin n → α) (i : Fin n) : Fin n :=
  Fin.find (λ j, ∀ k, T i j ≤ T i k)

def col_min (T : Fin n → Fin n → α) (j : Fin n) : Fin n :=
  Fin.find (λ i, ∀ k, T i j ≤ T k j)

theorem row_min_equals_col_min (T : Fin n → Fin n → α) :
  let R := Finset.image (λ i : Fin n, T i (row_min T i)) Finset.univ
      C := Finset.image (λ j : Fin n, T (col_min T j) j) Finset.univ
  in R = C := by
  sorry

end row_min_equals_col_min_l346_346708


namespace find_x_l346_346897

def avg_volume (v1 v2 v3 : ℝ) : ℝ := (v1 + v2 + v3) / 3

def total_volume (v1 v2 v3 : ℝ) : ℝ := v1 + v2 + v3

theorem find_x
  (x : ℝ)
  (h1 : total_volume (3^3) (12^3) (x^3) = 2100)
  (h2 : avg_volume (3^3) (12^3) (x^3) = 700) :
  x = 7 :=
by
  sorry

end find_x_l346_346897


namespace bathroom_new_area_l346_346039

theorem bathroom_new_area
  (current_area : ℕ)
  (current_width : ℕ)
  (extension : ℕ)
  (current_area_eq : current_area = 96)
  (current_width_eq : current_width = 8)
  (extension_eq : extension = 2) :
  ∃ new_area : ℕ, new_area = 144 :=
by
  sorry

end bathroom_new_area_l346_346039


namespace problem_solution_l346_346825

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x. Prove
    that the number of real solutions to the equation x² - 2⌊x⌋ - 3 = 0 is 3. -/
theorem problem_solution : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^2 - 2 * ⌊x⌋ - 3 = 0 := 
sorry

end problem_solution_l346_346825


namespace int_solutions_exist_for_x2_plus_15y2_eq_4n_l346_346355

theorem int_solutions_exist_for_x2_plus_15y2_eq_4n (n : ℕ) (hn : n > 0) : 
  ∃ S : Finset (ℤ × ℤ), S.card ≥ n ∧ ∀ (xy : ℤ × ℤ), xy ∈ S → xy.1^2 + 15 * xy.2^2 = 4^n :=
by
  sorry

end int_solutions_exist_for_x2_plus_15y2_eq_4n_l346_346355


namespace determine_champion_l346_346312

-- Define the candidates A, B, and C
inductive Candidate
| A : Candidate
| B : Candidate
| C : Candidate

open Candidate

-- Define the spectators’ statements
def statement_A (champion : Candidate) : Prop :=
  champion ≠ A ∧ champion ≠ B

def statement_B (champion : Candidate) : Prop :=
  champion ≠ A ∧ champion = C

def statement_C (champion : Candidate) : Prop :=
  champion ≠ C ∧ champion = A

-- Define the conditions about the correctness of statements
def correctness_conditions
  (correct_A correct_B correct_C : Prop)
  (incorrect_A incorrect_B incorrect_C : Prop) : Prop :=
  -- One made two correct judgments
  (correct_A ∧ correct_B ∧ ¬correct_C) ∨
  (correct_A ∧ ¬correct_B ∧ correct_C) ∨
  (¬correct_A ∧ correct_B ∧ correct_C) ∨
  (correct_A ∧ correct_B ∧ ¬correct_C) ∨
  (-- One made two incorrect judgments
  (incorrect_A ∧ incorrect_B ∧ ¬incorrect_C) ∨
  (incorrect_A ∧ ¬incorrect_B ∧ incorrect_C) ∨
  (¬incorrect_A ∧ incorrect_B ∧ incorrect_C) ∨
  (incorrect_A ∧ incorrect_B ∧ ¬incorrect_C) ∨
  -- One made one correct and one incorrect judgment
  (correct_A ∧ incorrect_A ∧ ¬correct_B ∧ incorrect_B ¬correct_C ∧ incorrect_C))

-- Statement to be verified
theorem determine_champion : ∃ champion : Candidate, champion = A ∧ correctness_conditions (
  -- Correctness of statements given the champion is A
  statement_A A = false ∧ statement_B A = false ∧ statement_C A = true ∧
  statement_A A = true ∧ statement_B A = false  statement_C A = true ∧
  statement_A A = true ∧ statement_B A = false
) :=
sorry

end determine_champion_l346_346312


namespace fixed_point_l346_346288

theorem fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ∃ x y, x = 1 ∧ y = -1 ∧ y = a^(x-1) - 2 :=
by
  use 1, -1
  split
  sorry
  sorry

end fixed_point_l346_346288


namespace find_x_plus_y_l346_346230

-- Define the initial assumptions and conditions
variables {x y : ℝ}
axiom geom_sequence : 1 > 0 ∧ x > 0 ∧ y > 0 ∧ 3 > 0 ∧ 1 * x = y
axiom arith_sequence : 2 * y = x + 3

-- Prove that x + y = 15 / 4
theorem find_x_plus_y : x + y = 15 / 4 := sorry

end find_x_plus_y_l346_346230


namespace calculate_product_value_l346_346158

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l346_346158


namespace mandy_chocolate_pieces_l346_346113

def chocolate_pieces_total : ℕ := 60
def half (n : ℕ) : ℕ := n / 2

def michael_taken : ℕ := half chocolate_pieces_total
def paige_taken : ℕ := half (chocolate_pieces_total - michael_taken)
def ben_taken : ℕ := half (chocolate_pieces_total - michael_taken - paige_taken)
def mandy_left : ℕ := chocolate_pieces_total - michael_taken - paige_taken - ben_taken

theorem mandy_chocolate_pieces : mandy_left = 8 :=
  by
  -- proof to be provided here
  sorry

end mandy_chocolate_pieces_l346_346113


namespace midpoint_reflect_sum_l346_346842

theorem midpoint_reflect_sum :
  let P := (3 : ℝ, 2 : ℝ),
      R := (13 : ℝ, 21 : ℝ),
      M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2),
      M_reflect := (M.1, -M.2)
  in M_reflect.1 + M_reflect.2 = -3.5 :=
by
  let P := (3 : ℝ, 2 : ℝ)
  let R := (13 : ℝ, 21 : ℝ)
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let M_reflect := (M.1, -M.2)
  show M_reflect.1 + M_reflect.2 = -3.5
  sorry

end midpoint_reflect_sum_l346_346842


namespace housewife_left_money_l346_346609

def initial_amount : ℝ := 150
def spent_fraction : ℝ := 2 / 3
def remaining_fraction : ℝ := 1 - spent_fraction

theorem housewife_left_money :
  initial_amount * remaining_fraction = 50 := by
  sorry

end housewife_left_money_l346_346609


namespace find_g_inv_l346_346043

noncomputable def g (x : ℕ) : ℕ :=
if x = 1 then 4 else
if x = 2 then 12 else
if x = 3 then 7 else
if x = 5 then 2 else
if x = 8 then 1 else
if x = 13 then 6 else 0

theorem find_g_inv :
  g⁻¹ ( (g⁻¹ 6 + g⁻¹ 12) / g⁻¹ 2 ) = 3 := by
  sorry

end find_g_inv_l346_346043


namespace smallest_four_digit_in_pascals_triangle_l346_346490

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346490


namespace trig_identity_l346_346714

theorem trig_identity (α : ℝ) (h1 : (-Real.pi / 2) < α ∧ α < 0)
  (h2 : Real.sin α + Real.cos α = 1 / 5) :
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25 / 7 := 
by 
  sorry

end trig_identity_l346_346714


namespace peter_present_age_l346_346586

def age_problem (P J : ℕ) : Prop :=
  J = P + 12 ∧ P - 10 = (1 / 3 : ℚ) * (J - 10)

theorem peter_present_age : ∃ (P : ℕ), ∃ (J : ℕ), age_problem P J ∧ P = 16 :=
by {
  -- Add the proof here, which is not required
  sorry
}

end peter_present_age_l346_346586


namespace team_count_l346_346004

theorem team_count (chs math eng : ℕ) (hchs : chs = 2) (hmath : math = 2) (heng : eng = 4) : 
  ∃ n : ℕ, (n = 44) ∧ 
  (∃ (chs_teams : ℕ) (math_teams : ℕ) (eng_teams : ℕ),
    chs_teams = (∑ i in finset.range (chs+1), 
                  ∑ j in finset.range (math+1), 
                  ∑ k in finset.range (eng+1), 
                  if i + j + k = 5 ∧ 0 < i ∧ 0 < j ∧ 0 < k then 
                    nat.choose chs i * nat.choose math j * nat.choose eng k 
                  else 0) ∧ 
    math_teams = (∑ i in finset.range (chs+1), 
                  ∑ j in finset.range (math+1), 
                  ∑ k in finset.range (eng+1), 
                  if i + j + k = 5 ∧ 0 < i ∧ 0 < j ∧ 0 < k then 
                    nat.choose chs i * nat.choose math j * nat.choose eng k 
                  else 0) ∧ 
    eng_teams = (∑ i in finset.range (chs+1), 
                 ∑ j in finset.range (math+1), 
                 ∑ k in finset.range (eng+1), 
                 if i + j + k = 5 ∧ 0 < i ∧ 0 < j ∧ 0 < k then 
                   nat.choose chs i * nat.choose math j * nat.choose eng k 
                 else 0)) sorry

end team_count_l346_346004


namespace direct_sum_intersection_l346_346012

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (W W₁ W₂ : Submodule ℝ V)

theorem direct_sum_intersection (h : W₁ ⊔ W₂ = ⊤) :
  W = (W ⊓ W₁) ⊕ (W ⊓ W₂) ↔ (W : Set V).dim = ((W ⊓ W₁) : Set V).dim + ((W ⊓ W₂) : Set V).dim :=
sorry

end direct_sum_intersection_l346_346012


namespace findEccentricity_l346_346711

-- Definitions for ellipse and its properties
def isEllipse (a b : ℝ) := a > b ∧ b > 0

def ellipseEquation (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- Foci (F1 and F2) setup
def f1 (c : ℝ) := (-c, 0)
def f2 (c : ℝ) := (c, 0)

-- Point P and angle
def pointPIntersect (a b c : ℝ) := 
  ∃ (x y : ℝ), ellipseEquation x y a b ∧ x = c

def angleF1PF2 (P : ℝ × ℝ) (c : ℝ) := ∠(f1 c, P, f2 c) = 45

-- Definiton of eccentricity
def eccentricity (a c : ℝ) := c / a

-- Main proof statement
theorem findEccentricity 
  (a b c : ℝ)
  (h_ellipse : isEllipse a b)
  (h_pointP : pointPIntersect a b c)
  (h_angle : angleF1PF2 (classical.some h_pointP) c) :
  eccentricity a c = √2 - 1 :=
sorry

end findEccentricity_l346_346711


namespace arc_length_ln_sin_eval_l346_346091

noncomputable def arc_length_ln_sin : ℝ :=
  ∫ x in (Real.pi / 3)..(Real.pi / 2), Real.sqrt (1 + (Real.cot x)^2)

theorem arc_length_ln_sin_eval :
  arc_length_ln_sin = (1 / 2) * Real.log 3 :=
by
  sorry

end arc_length_ln_sin_eval_l346_346091


namespace find_a_extreme_point_find_max_min_on_interval_l346_346720

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a*x - 2*a - 3) * Real.exp x

theorem find_a_extreme_point :
  (∀ a : ℝ, ∃ x : ℝ, (x = 2) → (Deriv f a x) = 0) → a = -5 :=
by
  sorry

theorem find_max_min_on_interval (a : ℝ) (h : a = -5) :
  let f' := fun (x : ℝ) => (Deriv (f x) a)
  ∀ x : ℝ, x ∈ Set.Icc (3/2:ℝ) 3 → 
  (let critical_points := { x | f' x = 0}
   ∧ ∀ x ∈ critical_points, 
     Set.max (λ x, f x a) Set.Icc (3/2:ℝ) 3 = Real.exp 3
     ∧ Set.min (λ x, f x a) Set.Icc (3/2:ℝ) 3 = Real.exp 2) :=
by
  sorry

end find_a_extreme_point_find_max_min_on_interval_l346_346720


namespace speed_ratio_equidistant_l346_346108

theorem speed_ratio_equidistant:
  ∀ (v_A v_B : ℝ),
  v_A > 0 ∧ v_B > 0 ∧ 
  (3 * v_A = | -800 + 3 * v_B |) ∧ 
  (9 * v_A = | -800 + 9 * v_B |) →
  v_A / v_B = 3 / 4 :=
begin
  sorry
end

end speed_ratio_equidistant_l346_346108


namespace average_root_cross_sectional_area_is_correct_average_volume_is_correct_sample_correlation_coefficient_is_correct_estimated_total_volume_is_correct_l346_346951

-- Definitions of the given data
def n := 10
def x := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06] : List ℝ
def y := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40] : List ℝ

def sum_x := 0.6
def sum_y := 3.9
def sum_x_squared := 0.038
def sum_y_squared := 1.6158
def sum_xy := 0.2474
def total_area := 186

-- Calculations for the average root cross-sectional area and volume
def avg_x := sum_x / n
def avg_y := sum_y / n

-- Correlation coefficient calculation
def r := (sum_xy - n * avg_x * avg_y) / (Real.sqrt ((sum_x_squared - n * avg_x^2) * (sum_y_squared - n * avg_y^2)))

-- Total volume estimation
def estimated_volume := (avg_y / avg_x) * total_area

-- Lean theorem statements
theorem average_root_cross_sectional_area_is_correct : avg_x = 0.06 := by
  sorry

theorem average_volume_is_correct : avg_y = 0.39 := by
  sorry

theorem sample_correlation_coefficient_is_correct : Real.round (r * 100) / 100 = 0.97 := by
  sorry

theorem estimated_total_volume_is_correct : estimated_volume = 1209 := by
  sorry

end average_root_cross_sectional_area_is_correct_average_volume_is_correct_sample_correlation_coefficient_is_correct_estimated_total_volume_is_correct_l346_346951


namespace correct_system_l346_346865

-- Define the variables and equations
variables (x y : Real)

-- First condition: 5 cows and 2 sheep together have 19 taels of silver
def condition1 : Prop := 5 * x + 2 * y = 19

-- Second condition: 2 cows and 3 sheep together have 12 taels of silver
def condition2 : Prop := 2 * x + 3 * y = 12

-- The goal is to show both conditions together form the system of equations
theorem correct_system : condition1 ∧ condition2 := 
by 
  -- Proof is not required, use sorry as a placeholder
  sorry

end correct_system_l346_346865


namespace product_of_f_in_T_l346_346812

-- Define the set S excluding (2,2)
def is_in_S (x y : ℕ) : Prop := 
  (x ∈ {0, 1, 2, 3}) ∧ 
  (y ∈ {0, 1, 2, 3}) ∧ 
  ¬(x = 2 ∧ y = 2)

-- Define triangle t is valid in T as having vertices in S with right angle at B
structure triangle := (A B C : ℕ × ℕ)

def is_in_T (t : triangle) : Prop :=
  (is_in_S (t.A).fst (t.A).snd) ∧ 
  (is_in_S (t.B).fst (t.B).snd) ∧ 
  (is_in_S (t.C).fst (t.C).snd) ∧
  -- Additional condition to ensure the right angle at B, define a right angle constraint
  ((t.B).fst - (t.A).fst) * ((t.B).fst - (t.C).fst) + ((t.B).snd - (t.A).snd) * ((t.B).snd - (t.C).snd) = 0

-- Define f(t) using the tangent of the angle ACB
def f (t : triangle) : ℚ :=
  -- This will need to be properly defined, but we are focusing on the structure
  -- For the purpose of this statement definition, assume we have a tangent calculation function
  sorry

-- The statement to be proved:
theorem product_of_f_in_T : ∏ t in (set_of is_in_T), f t = 1 := 
by sorry

end product_of_f_in_T_l346_346812


namespace unique_solution_for_a_eq_1_l346_346210

theorem unique_solution_for_a_eq_1 :
  (∀ a x : ℝ, 
    let lhs := 3^(x^2 - 2*a*x + a^2),
        rhs := a*x^2 - 2*a^2*x + a^3 + a^2 - 4*a + 4
    in lhs = rhs → x = a) ↔ a = 1 := 
by
  sorry

end unique_solution_for_a_eq_1_l346_346210


namespace percent_increase_correct_l346_346191

noncomputable def last_year_ticket_price : ℝ := 85
noncomputable def last_year_tax_rate : ℝ := 0.10
noncomputable def this_year_ticket_price : ℝ := 102
noncomputable def this_year_tax_rate : ℝ := 0.12
noncomputable def student_discount_rate : ℝ := 0.15

noncomputable def last_year_total_cost : ℝ := last_year_ticket_price * (1 + last_year_tax_rate)
noncomputable def discounted_ticket_price_this_year : ℝ := this_year_ticket_price * (1 - student_discount_rate)
noncomputable def total_cost_this_year : ℝ := discounted_ticket_price_this_year * (1 + this_year_tax_rate)

noncomputable def percent_increase : ℝ := ((total_cost_this_year - last_year_total_cost) / last_year_total_cost) * 100

theorem percent_increase_correct :
  abs (percent_increase - 3.854) < 0.001 := sorry

end percent_increase_correct_l346_346191


namespace a_n_formula_b_n_sum_l346_346223

-- Given the sequence {a_n} where a_1 = 1 and a_{n+1} = a_n / (a_n + 3)
variable (a : ℕ → ℝ)
axiom a_seq : a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n / (a n + 3)

-- Definition of the sequence {b_n}
def b (n : ℕ) : ℝ := (3^n - 1) * n / 2^n * a n

-- The general formula for a_n
theorem a_n_formula : ∀ n : ℕ, n > 0 → a n = 2 / (3^n - 1)
:= sorry

-- The sum of the first n terms of {b_n}
theorem b_n_sum (n : ℕ) : 
(∑ i in Finset.range n, b (i + 1)) = 4 - (n + 2) / 2^(n - 1)
:= sorry

end a_n_formula_b_n_sum_l346_346223


namespace can_assemble_natural_sum_l346_346093

-- Definitions based on the problem conditions
def alpha : ℝ := (-1 + Real.sqrt 29) / 2
def is_natural (n : ℕ) : Prop := n ≥ 0

-- Conditions (no denomination used more than 6 times, and each is irrational except 1)
def valid_coins (k : ℕ) : Prop := (k = 0 ∨ irrational (alpha ^ k)) ∧ alpha > 2

-- Main statement to prove
theorem can_assemble_natural_sum (n : ℕ) (h_nat : is_natural n) : 
  (∀ k : ℕ, valid_coins k) → ∃ (coefficients : ℕ → ℕ), (∀ k : ℕ, coefficients k ≤ 6) ∧ 
  (n = ∑ (k : ℕ) in (Finset.range (n + 1)), (coefficients k) * (alpha ^ k)) :=
by
  sorry

end can_assemble_natural_sum_l346_346093


namespace ezekiel_third_day_hike_l346_346196

-- Ezekiel's total hike distance
def total_distance : ℕ := 50

-- Distance covered on the first day
def first_day_distance : ℕ := 10

-- Distance covered on the second day
def second_day_distance : ℕ := total_distance / 2

-- Distance remaining for the third day
def third_day_distance : ℕ := total_distance - first_day_distance - second_day_distance

-- The distance Ezekiel had to hike on the third day
theorem ezekiel_third_day_hike : third_day_distance = 15 := by
  sorry

end ezekiel_third_day_hike_l346_346196


namespace line_plane_relationship_l346_346127

-- Definitions for the problem
variables {Point : Type} [MetricSpace Point] -- Assuming a general space with points
variables {Line Plane : Type} -- Assuming Line and Plane types
variable m : Line -- Line m
variable α β : Plane -- Planes α and β
variable perpendicular : Line → Plane → Prop -- Definition of perpendicular for line to plane
variable perpendicular_planes : Plane → Plane → Prop -- Definition of perpendicular for plane to plane
variable parallel : Line → Plane → Prop -- Definition of parallel for line to plane
variable contained : Line → Plane → Prop -- Definition of containing a line in a plane

-- Conditions
axiom m_perp_beta : perpendicular m β
axiom alpha_perp_beta : perpendicular_planes α β

-- Theorem: Proving the positional relationship based on the given conditions
theorem line_plane_relationship :
  parallel m α ∨ contained m α :=
sorry -- Proof omitted

end line_plane_relationship_l346_346127


namespace tv_show_duration_l346_346347

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end tv_show_duration_l346_346347


namespace smallest_four_digit_in_pascals_triangle_l346_346429

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346429


namespace polynomial_has_three_rational_roots_l346_346577

-- Define the polynomial with rational coefficients
def is_degree_three_polynomial {R : Type*} [Field R] (f : R[X]) : Prop :=
  nat_degree f = 3 ∧ ∀ i, coeff f i ∈ ℚ

-- Define the property that the graph touches the x-axis (has a repeated root)
def touches_x_axis {R : Type*} [Field R] (f : R[X]) : Prop :=
  ∃ x : R, deriv f.eval x = 0

theorem polynomial_has_three_rational_roots
  {R : Type*} [Field R] (f : R[X])
  (h_deg : is_degree_three_polynomial f)
  (h_touch : touches_x_axis f) :
  ∃ x y z : ℚ, f = C (f.leading_coeff) * (X - C x) * (X - C y) * (X - C z) :=
sorry

end polynomial_has_three_rational_roots_l346_346577


namespace tan_expression_l346_346694

theorem tan_expression (θ : ℝ) 
  (h : 4 * cos (θ + π / 3) * cos (θ - π / 6) = sin (2 * θ)) : 
  tan (2 * θ - π / 6) = sqrt 3 / 9 := 
sorry

end tan_expression_l346_346694


namespace valid_m_values_l346_346149

theorem valid_m_values :
  ∃ (m : ℕ), 5 ≤ m ∧ m ≤ 25 ∧ m % 2 = 1 :=
begin
  sorry
end

end valid_m_values_l346_346149


namespace area_of_right_triangle_l346_346613

theorem area_of_right_triangle 
  (intersect1 : Line) 
  (intersect2 : Line)
  (h_origin : ∃ t : ℝ, intersect1.eval t = (0, 0))
  (h_y2 : ∃ t : ℝ, intersect1.eval t = (x, 2))
  (h_intersect : ∃ t : ℝ, intersect1.eval t = (1 + sqrt 3 * y, y)) :
  let (x2, y2) := (1 + sqrt 3 * 2, 2) in
  let base := 1 in
  let height := 2 in
  (1/2 : ℝ) * base * height = 1 :=
by
  sorry

end area_of_right_triangle_l346_346613


namespace smallest_four_digit_in_pascal_triangle_l346_346556

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346556


namespace perimeter_triangle_PXY_l346_346920

-- Definitions and conditions setting up the problem
noncomputable theory

variables {P Q R X Y I : Type}

-- Given triangle PQR
variable [triangle P Q R]
variable (PQ : ℝ) (QR : ℝ) (PR : ℝ)
-- Given side lengths
variable (hPQ : PQ = 14)
variable (hQR : QR = 28)
variable (hPR : PR = 21)

-- Given incenter I of triangle PQR
variable [incenter P Q R I]
-- Line through I parallel to QR intersects PQ at X and PR at Y
variable [parallel (line_through I QR) (line_through X PQ)]
variable [parallel (line_through I QR) (line_through Y PR)]

-- Proof statement
theorem perimeter_triangle_PXY : perimeter P X Y = 35 := sorry

end perimeter_triangle_PXY_l346_346920


namespace find_number_l346_346568

theorem find_number (x : ℕ) (h : 23 + x = 34) : x = 11 :=
by
  sorry

end find_number_l346_346568


namespace sufficient_not_necessary_condition_l346_346106

theorem sufficient_not_necessary_condition {x : ℝ} (h : 1 < x ∧ x < 2) : x < 2 ∧ ¬(∀ x, x < 2 → (1 < x ∧ x < 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l346_346106


namespace total_flowers_eaten_l346_346839

-- Definitions based on conditions
def num_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Statement asserting the total number of flowers eaten
theorem total_flowers_eaten : num_bugs * flowers_per_bug = 6 := by
  sorry

end total_flowers_eaten_l346_346839


namespace last_digit_101_pow_100_l346_346418

theorem last_digit_101_pow_100 :
  (101^100) % 10 = 1 :=
by
  sorry

end last_digit_101_pow_100_l346_346418


namespace smallest_four_digit_in_pascal_l346_346525

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346525


namespace triangle_inequality_l346_346779

theorem triangle_inequality (A B C A1 B1 C1 : Type) [Metric_Space A] [Metric_Space B] [Metric_Space C]
  (AB BC CA AA1 BB1 CC1 : Real) (h1 : AA1 = sqrt(A * B^2 + λ * A ^ 2 + λ * (B^2 - A ^ 2 - B^2)))
  (h2 : BB1 = sqrt(B * C ^ 2 + λ * B^2 + λ * (C^2 - B^2 - C^2)))
  (h3 : CC1 = sqrt(C * A ^ 2 + λ * C^2 + λ * (A^2 - C^2 - A^2)))
  (hinequality : (B * A1 == B C = C B1 == B A)) :
  (3 / 4) * (A^2 + B^2 + C^2) ≤ AA1^2 + BB1^2 + CC1^2 ∧ AA1^2 + BB1^2 + CC1^2 ≤ A^2 + B^2 + C^2 :=
sorry

end triangle_inequality_l346_346779


namespace infinite_harmonic_progressions_l346_346848

open Finset

/-- The arithmetic progression considered as a set of natural numbers (with offset) -/
def arithmetic_prog : Set ℕ := {n | ∃ k : ℕ, n = 1 + 3 * k}

/-- Predicate for a harmonic progression -/
def harmonic_progression (a b c : ℕ) : Prop :=
  (2:ℚ) / b = (1:ℚ) / a + (1:ℚ) / c

theorem infinite_harmonic_progressions :
  ∃ (f : ℕ → ℕ × ℕ × ℕ),
  (∀ n, (f n).1 ∈ arithmetic_prog ∧ (f n).2.1 ∈ arithmetic_prog ∧ (f n).2.2 ∈ arithmetic_prog ∧
  harmonic_progression (f n).1 (f n).2.1 (f n).2.2) ∧
  ∀ m, (f m).1 ∈ arithmetic_prog ∧ (f m).2.1 ∈ arithmetic_prog ∧ (f m).2.2 ∈ arithmetic_prog ∧
  harmonic_progression (f m).1 (f m).2.1 (f m).2.2) ∧ 
  ((f n).1 * (f m).2.1 ≠ (f m).1 * (f n).2.1)) :=
sorry

end infinite_harmonic_progressions_l346_346848


namespace system_correct_l346_346864

def correct_system (x y : ℝ) : Prop :=
  5 * x + 2 * y = 19 ∧ 2 * x + 3 * y = 12

theorem system_correct (x y : ℝ) (h1 : 5 * x + 2 * y = 19) (h2 : 2 * x + 3 * y = 12) :
  correct_system x y :=
by {
  split,
  exact h1,
  exact h2,
}

end system_correct_l346_346864


namespace smallest_four_digit_number_in_pascals_triangle_l346_346454

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346454


namespace a_4_eq_9_a_n_div_fact_eq_sum_l346_346388

noncomputable def a : ℕ → ℕ
  | 0       => 1
  | 1       => 0
  | (n + 2) => (n + 1) * (a (n + 1) + a n)

theorem a_4_eq_9 : a 4 = 9 :=
  sorry

theorem a_n_div_fact_eq_sum (n : ℕ) :
    (a n : ℚ) / n.factorial = ∑ i in finset.range (n + 1).erase 0, ((-1 : ℚ)^(i + 1)) / i.factorial := 
  sorry

end a_4_eq_9_a_n_div_fact_eq_sum_l346_346388


namespace theta_quadrants_l346_346213

theorem theta_quadrants (θ : ℝ) : (1/2)^(sin(2 * θ)) < 1 → 
  (∃ k : ℤ, k * π < θ ∧ θ < k * π + π / 2) :=
sorry

end theta_quadrants_l346_346213


namespace evaluate_f_l346_346250

def f : ℝ → ℝ
| x := if x < 4 then f(x + 1) else 2^x

theorem evaluate_f : f (2 + log 2 3) = 24 := sorry

end evaluate_f_l346_346250


namespace prob_1_leq_X_leq_2_l346_346246

noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) : Type :=
sorry -- formal definition of a normal distribution (omitted for simplicity)

constant X : normal_distribution 1 σ

axiom prob_X_leq_0 : P(X ≤ 0) = 0.1

theorem prob_1_leq_X_leq_2 : P(1 ≤ X ≤ 2) = 0.4 :=
by {
  -- Proof (not required per instructions)
  sorry
}

end prob_1_leq_X_leq_2_l346_346246


namespace correct_system_l346_346866

-- Define the variables and equations
variables (x y : Real)

-- First condition: 5 cows and 2 sheep together have 19 taels of silver
def condition1 : Prop := 5 * x + 2 * y = 19

-- Second condition: 2 cows and 3 sheep together have 12 taels of silver
def condition2 : Prop := 2 * x + 3 * y = 12

-- The goal is to show both conditions together form the system of equations
theorem correct_system : condition1 ∧ condition2 := 
by 
  -- Proof is not required, use sorry as a placeholder
  sorry

end correct_system_l346_346866


namespace solve_quadratic_equation_l346_346406

theorem solve_quadratic_equation (x : ℝ) : x^2 = 100 → x = -10 ∨ x = 10 :=
by
  intro h
  sorry

end solve_quadratic_equation_l346_346406


namespace sqrt_mul_sqrt_eq_l346_346165

theorem sqrt_mul_sqrt_eq (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) :=
by {
  sorry
}

example : Real.sqrt 2 * Real.sqrt 8 = 4 :=
by {
  have h: Real.sqrt 2 * Real.sqrt 8 = Real.sqrt (2 * 8) := sqrt_mul_sqrt_eq 2 8 (by norm_num) (by norm_num),
  rw h,
  norm_num
}

end sqrt_mul_sqrt_eq_l346_346165


namespace average_distinct_t_for_positive_integer_roots_l346_346657

theorem average_distinct_t_for_positive_integer_roots :
  let t_values := { t | ∃ r1 r2 : ℕ, r1 + r2 = 6 ∧ r1 * r2 = t }
  let distinct_t_values := {5, 8, 9}
  let average := (5 + 8 + 9) / 3
  average = 7 + 1/3 := sorry

end average_distinct_t_for_positive_integer_roots_l346_346657


namespace new_bathroom_area_l346_346034

variable (area : ℕ) (width : ℕ) (extension : ℕ)

theorem new_bathroom_area (h1 : area = 96) (h2 : width = 8) (h3 : extension = 2) :
  (let orig_length := area / width;
       new_length := orig_length + extension;
       new_width := width + extension;
       new_area := new_length * new_width
   in new_area) = 140 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end new_bathroom_area_l346_346034


namespace fraction_addition_l346_346079

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := 
by 
  sorry

end fraction_addition_l346_346079


namespace general_term_formula_sum_terms_less_than_l346_346247

-- Conditions
def S (a : ℕ → ℤ) (n : ℕ) := ∑ i in Finset.range n, a i
def Sn (a : ℕ → ℤ) (n : ℕ) := S a n
def cond1 (a : ℕ → ℤ) (Sn : ℕ → ℤ) (n : ℕ) := Sn n + (1 / 2) * a n = 1
def bn (a : ℕ → ℤ) (n : ℕ) := real.log (a n ^ 2 / 4) / real.log 3

-- 1. Prove the general term formula for the sequence {a_n}.
theorem general_term_formula (a : ℕ → ℤ) (Sn : ℕ → ℤ) (n : ℕ) (h1 : ∀ n, cond1 a Sn n) :
  a n = 2 / 3^n := 
sorry

-- 2. Prove that the sum of the first n terms of the sequence {1 / (b_n * b_{n+2})} is less than 3/16.
theorem sum_terms_less_than (a : ℕ → ℤ) (Sn : ℕ → ℤ) (bn : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n, cond1 a Sn n)
  (h2 : ∀ n, bn n = real.log (a n ^ 2 / 4) / real.log 3) :
  (∑ i in Finset.range n, 1 / (bn i * bn (i + 2))) < 3 / 16 := 
sorry

end general_term_formula_sum_terms_less_than_l346_346247


namespace distinct_subscription_choices_l346_346324

theorem distinct_subscription_choices (num_providers : ℕ) (num_siblings : ℕ) : 
  num_providers = 25 → num_siblings = 4 → 
  (∏ i in finset.range num_siblings, num_providers - i) = 303600 :=
begin
  intros h_providers h_siblings,
  rw [h_providers, h_siblings],
  norm_num,
end

end distinct_subscription_choices_l346_346324


namespace smallest_four_digit_number_in_pascals_triangle_l346_346450

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346450


namespace bill_face_value_l346_346579

theorem bill_face_value
  (TD : ℝ) (T : ℝ) (r : ℝ) (FV : ℝ)
  (h1 : TD = 210)
  (h2 : T = 0.75)
  (h3 : r = 0.16) :
  FV = 1960 :=
by 
  sorry

end bill_face_value_l346_346579


namespace CaitlinAge_l346_346154

theorem CaitlinAge (age_AuntAnna : ℕ) (age_Brianna : ℕ) (age_Caitlin : ℕ)
  (h1 : age_AuntAnna = 42)
  (h2 : age_Brianna = age_AuntAnna / 2)
  (h3 : age_Caitlin = age_Brianna - 5) :
  age_Caitlin = 16 :=
by 
  sorry

end CaitlinAge_l346_346154


namespace number_of_positive_integers_l346_346903

def greatest_integer (x : ℝ) : ℤ :=
  Int.floor x

def condition (x : ℤ) : Prop :=
  greatest_integer ((x + 1 : ℤ) / 3) = 3

theorem number_of_positive_integers (count : ℕ) :
  count = (Finset.range (11) \ Finset.range (8)).filter (λ x, x > 0).card :=
by {
  sorry
}

end number_of_positive_integers_l346_346903


namespace smallest_four_digit_number_in_pascals_triangle_l346_346463

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346463


namespace solve_equation_125_eq_5_25_exp_x_min_2_l346_346009

theorem solve_equation_125_eq_5_25_exp_x_min_2 :
    ∃ x : ℝ, 125 = 5 * (25 : ℝ)^(x - 2) ∧ x = 3 := 
by
  sorry

end solve_equation_125_eq_5_25_exp_x_min_2_l346_346009


namespace smallest_four_digit_in_pascal_l346_346513

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346513


namespace similar_triangle_perimeter_l346_346901

theorem similar_triangle_perimeter
  (a b c : ℝ)
  (ha : a = 15)
  (hb : b = 30)
  (hc : c = 30)
  (a' : ℝ)
  (ha' : a' = 45)
  (similar : a' / a = 3) :
  a' + b * 3 + c * 3 = 225 :=
by
  rw [ha, hb, hc, ha']
  rw [← mul_assoc, ← mul_assoc]
  have scale : 3 = a' / a := sorry
  rw [←scale]
  exact sorry

end similar_triangle_perimeter_l346_346901


namespace possible_values_l346_346691

theorem possible_values (a b c : ℝ)
    (h : det (Matrix.of 3 3 (λ i j, ([a, b, c][i]) * ([a, b, c][j]))) = 0) :
    let result := (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2))
    in result = 3 / 2 ∨ result = -3 :=
by
  sorry

end possible_values_l346_346691


namespace exists_unique_integer_pair_l346_346227

theorem exists_unique_integer_pair (a : ℕ) (ha : 0 < a) :
  ∃! (x y : ℕ), 0 < x ∧ 0 < y ∧ x + (x + y - 1) * (x + y - 2) / 2 = a :=
by
  sorry

end exists_unique_integer_pair_l346_346227


namespace monotonicity_and_k_range_l346_346826

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x + (1 / 2 : ℝ) * x^2 - x

theorem monotonicity_and_k_range :
  (∀ x : ℝ, x ≥ 0 → f x ≥ k * x - 2) ↔ k ∈ Set.Iic (-2) := sorry

end monotonicity_and_k_range_l346_346826


namespace bela_always_wins_l346_346304

-- Defining the conditions of the game.
def Game (interval : Set ℝ) (distance : ℝ) :=
  ∀ x y ∈ interval, x ≠ y → |x - y| ≥ distance

-- The definition of the interval [0, 10] and the distance of 2 units.
def interval : Set ℝ := Set.Icc 0 10
def distance := 2

-- Theorem that states "Bela always wins".
theorem bela_always_wins : (∃ interval, (interval = Set.Icc 0 10)) →
  (∃ distance, (distance = 2)) →
  ∀ (G : Set ℝ → ℝ → Prop), G interval distance → 
  (∀ optimal_play : (ℝ → ℝ), ∃ (result : ℝ), result = 5 → result ∈ interval) → 
  (∀ optimal_play : (ℝ → ℝ), ∀ (result : ℝ), result = 1 ∨ result = 9 → result ∈ interval) → 
  True :=
by
  intros h_interval h_distance h_game h_bela_turn h_jenn_turn
  sorry

end bela_always_wins_l346_346304


namespace intersection_points_l346_346807

def f (x : ℝ) : ℝ := (x^2 - 8*x + 10) / (3*x - 6)

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ :=
  (2*a*x^2 + b*x + c) / (x - d)

theorem intersection_points (a b c d : ℝ) (h_vert_asymptote : d = 2)
  (h_oblique_perpendicular : ∀ x, (3 * (1 / 3) * x - 10) = (-3 * x + k))
  (h_intersection_at_3 : f 3 = g a b c d 3) :
  (∃ y1, f 3 = y1 ∧ g a b c d 3 = y1 ∧ y1 = 1 / 3)
  ∧ (∃ y2, f 4 = y2 ∧ g a b c d 4 = y2 ∧ y2 = -2)
  ∧ (∃ y3, f (-5) = y3 ∧ g a b c d (-5) = y3 ∧ y3 = -1 / 3) :=
sorry

end intersection_points_l346_346807


namespace find_a_for_tangent_line_l346_346764

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * Real.log x

theorem find_a_for_tangent_line :
  (∀ (a : ℝ), ∃ (f : ℝ → ℝ), 
    -- Function definition
    f = (λ x, x - a * Real.log x) ∧ 
    -- Given point and tangent line condition
    f 1 = 1 ∧ 
    -- Slope condition at point (1,1)
    derivative f 1 = 3) → 
  a = -2 :=
sorry

end find_a_for_tangent_line_l346_346764


namespace no_ring_with_exactly_five_regular_elements_l346_346357

-- Define a ring and regular element condition
class Ring (R : Type*) extends AddCommGroup R, Mul R, Zero R :=
(mul_assoc : ∀ a b c : R, a * b * c = a * (b * c))
(add_mul : ∀ a b c : R, (a + b) * c = a * c + b * c)
(mul_add : ∀ a b c : R, a * (b + c) = a * b + a * c)

def is_regular {R : Type*} [Ring R] (a : R) : Prop :=
∀ x : R, a * x = 0 ∨ x * a = 0 → x = 0

-- Assume G is the set of all regular elements in R
def regular_elements (R : Type*) [Ring R] : set R := {a : R | is_regular a}

theorem no_ring_with_exactly_five_regular_elements :
  ∀ (R : Type*) [Ring R], ¬ (set.finite (regular_elements R) ∧ finset.card (regular_elements R) = 5) :=
by
  intros R _ assumption
  sorry

end no_ring_with_exactly_five_regular_elements_l346_346357


namespace evaluate_powers_of_i_l346_346192

-- The imaginary unit i and its powers repeating every four steps.
def i_pow (n : ℕ) : ℂ :=
  if n % 4 = 0 then 1
  else if n % 4 = 1 then complex.I
  else if n % 4 = 2 then -1
  else -complex.I

theorem evaluate_powers_of_i :
  i_pow 17 + i_pow 2023 = 0 := by
  sorry

end evaluate_powers_of_i_l346_346192


namespace smallest_four_digit_in_pascal_l346_346500

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346500


namespace no_real_x_condition_l346_346292

theorem no_real_x_condition (a : ℝ) :
  (¬ ∃ x : ℝ, |x - 3| + |x - 1| ≤ a) ↔ a < 2 := 
by
  sorry

end no_real_x_condition_l346_346292


namespace range_of_a_if_monotonic_decreasing_l346_346218

theorem range_of_a_if_monotonic_decreasing (a : ℝ) (h : a ≥ 0)
  (H : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), (deriv (λ x, (x^2 - 2*a*x)*Real.exp x) x) ≤ 0) :
  a ≥ 3 / 4 := 
sorry

end range_of_a_if_monotonic_decreasing_l346_346218


namespace worker_B_time_comparison_l346_346056

-- Definition of the conditions and the question in Lean 4
def worker_A_time := 14
def combined_time := 6
def combined_rate := (1 : ℝ) / worker_A_time + (1 : ℝ) / worker_B_time = (1 : ℝ) / combined_time

theorem worker_B_time_comparison (worker_B_time : ℝ) :
  combined_rate → worker_B_time - combined_time = 4.5 :=
by
  sorry

end worker_B_time_comparison_l346_346056


namespace min_value_of_expr_l346_346881

open Real

theorem min_value_of_expr 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_line : ∀ x y : ℝ, a * x - b * y + 2 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y = 0)
  (h_ab_max : ∀ A B : ℝ × ℝ, |A - B| is maximized under h_line and h_circle) :
  (1 / a + 4 / b) = 9 / 2 :=
by
  sorry

end min_value_of_expr_l346_346881


namespace triangles_from_decagon_l346_346990

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346990


namespace smallest_four_digit_in_pascal_l346_346501

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346501


namespace smallest_four_digit_in_pascal_l346_346494

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346494


namespace smallest_four_digit_in_pascals_triangle_l346_346432

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346432


namespace sum_of_legs_of_isosceles_right_triangle_l346_346031

theorem sum_of_legs_of_isosceles_right_triangle
  (hypotenuse : ℝ)
  (h_hypotenuse : hypotenuse = 5.656854249492381) :
  ∃ (a : ℝ), a * 2 = 8 ∧ hypotenuse = a * Real.sqrt 2 :=
by
  have : Real.sqrt 2 ≠ 0 := sorry
  use hypotenuse / Real.sqrt 2
  have : hypotenuse = (hypotenuse / Real.sqrt 2) * Real.sqrt 2 := sorry
  have : (hypotenuse / Real.sqrt 2) * 2 = 8 := sorry
  split
  { exact this }
  { exact h_hypotenuse }

end sum_of_legs_of_isosceles_right_triangle_l346_346031


namespace impossible_gather_all_coins_in_one_sector_l346_346117

-- Definition of the initial condition with sectors and coins
def initial_coins_in_sectors := [1, 1, 1, 1, 1, 1] -- Each sector has one coin, represented by a list

-- Function to check if all coins are in one sector
def all_coins_in_one_sector (coins : List ℕ) := coins.count 6 == 1

-- Function to make a move (this is a helper; its implementation isn't necessary here but illustrates the idea)
def make_move (coins : List ℕ) (src dst : ℕ) : List ℕ := sorry

-- Proving that after 20 moves, coins cannot be gathered in one sector due to parity constraints
theorem impossible_gather_all_coins_in_one_sector : 
  ¬ ∃ (moves : List (ℕ × ℕ)), moves.length = 20 ∧ all_coins_in_one_sector (List.foldl (λ coins move => make_move coins move.1 move.2) initial_coins_in_sectors moves) :=
sorry

end impossible_gather_all_coins_in_one_sector_l346_346117


namespace solution_set_inequality_l346_346688

theorem solution_set_inequality (x : ℝ) : 4 * x^2 - 3 * x > 5 ↔ x < -5/4 ∨ x > 1 :=
by
  sorry

end solution_set_inequality_l346_346688


namespace determine_b2050_l346_346813

theorem determine_b2050 (b : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h₁ : b 1 = 3 + Real.sqrt 2)
  (h₂ : b 2021 = 7 + 2 * Real.sqrt 2) :
  b 2050 = (7 - 2 * Real.sqrt 2) / 41 := 
sorry

end determine_b2050_l346_346813


namespace pet_store_cages_l346_346937

theorem pet_store_cages (total_puppies sold_puppies puppies_per_cage : ℕ) (h1 : total_puppies = 45) (h2 : sold_puppies = 39) (h3 : puppies_per_cage = 2) :
  (total_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l346_346937


namespace fifth_sixth_digits_of_N_l346_346883

def is_prime_pair (a b : ℕ) : Prop :=
  let n := 10 * a + b in
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨
  n = 23 ∨ n = 29 ∨ n = 31 ∨ n = 37 ∨ n = 41 ∨ n = 43 ∨ n = 47 ∨
  n = 53 ∨ n = 59 ∨ n = 61 ∨ n = 67 ∨ n = 71 ∨ n = 73 ∨ n = 79 ∨
  n = 83 ∨ n = 89 ∨ n = 97

noncomputable def largest_number_no_prime_digits : ℕ :=
  987635421  -- By the solution steps, this is determined manually

theorem fifth_sixth_digits_of_N :
  ∃ (N : ℕ), N = largest_number_no_prime_digits ∧
       (N / 10000 % 10 = 3) ∧
       (N / 1000 % 10 = 5) :=
begin
  use largest_number_no_prime_digits,
  split,
  { refl },
  split;
  { norm_num },
end

end fifth_sixth_digits_of_N_l346_346883


namespace postage_cost_l346_346927

theorem postage_cost (weight : ℝ) (base_rate : ℝ) (additional_rate : ℝ) (cap : ℝ) :
  weight = 2.8 ∧ base_rate = 0.50 ∧ additional_rate = 0.15 ∧ cap = 1.30 →
  let additional_weight := weight - 1
  let num_additional_charges := Int.ceil (additional_weight / 0.5)
  let additional_cost := num_additional_charges * additional_rate
  let total_cost := base_rate + additional_cost
  total_cost ≤ cap ∧ total_cost = 1.10 :=
by 
  intros h
  cases h with hw hb
  cases hb with hbr har
  cases har with har hc
  let additional_weight := 2.8 - 1
  let num_additional_charges := Int.ceil (additional_weight / 0.5)
  let additional_cost := num_additional_charges * 0.15
  let total_cost := 0.50 + additional_cost
  have h1 : additional_weight = 1.8 := rfl
  have h2 : num_additional_charges = 4 := rfl
  have h3 : additional_cost = 0.60 := by linarith
  have h4 : total_cost = 1.10 := by linarith
  have h5 : total_cost ≤ 1.30 := by linarith
  exact ⟨h5, h4⟩

end postage_cost_l346_346927


namespace chef_initial_cherries_l346_346594

theorem chef_initial_cherries : ∃ (C : ℕ), C = 60 + 17 := by
  use 77
  sorry

end chef_initial_cherries_l346_346594


namespace solve_payment_difference_l346_346341

noncomputable def plan1_payment_difference (principal : ℝ) (annual_rate : ℝ) (years_plan1 : ℝ) (years_plan2 : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let amount_after_4_years := principal * (1 + monthly_rate / 100) ^ (years_plan1 * 12 / 3)
  let payment_at_4_years := amount_after_4_years / 3
  let remaining_balance := amount_after_4_years - payment_at_4_years
  let final_amount_plan1 := remaining_balance * (1 + monthly_rate / 100) ^ (years_plan1 * 12 * 2 / 3)
  let total_payment_plan1 := payment_at_4_years + final_amount_plan1
  
  let final_amount_plan2 := principal * real.exp (annual_rate / 100 * years_plan2)
  
  final_amount_plan2 - total_payment_plan1

theorem solve_payment_difference :
  plan1_payment_difference 12000 8 12 12 ≈ 12124.38 :=
by
  sorry

end solve_payment_difference_l346_346341


namespace smallest_four_digit_in_pascals_triangle_l346_346536

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346536


namespace slope_of_line_in_terms_of_angle_l346_346040

variable {x y : ℝ}

theorem slope_of_line_in_terms_of_angle (h : 2 * Real.sqrt 3 * x - 2 * y - 1 = 0) :
    ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧ Real.tan α = Real.sqrt 3 ∧ α = Real.pi / 3 :=
by
  sorry

end slope_of_line_in_terms_of_angle_l346_346040


namespace speed_of_man_in_still_water_l346_346082

theorem speed_of_man_in_still_water
  (v_m v_s : ℝ)
  (h1 : v_m + v_s = 4)
  (h2 : v_m - v_s = 2) :
  v_m = 3 := 
by sorry

end speed_of_man_in_still_water_l346_346082


namespace isosceles_obtuse_triangle_smallest_angle_measure_l346_346956

theorem isosceles_obtuse_triangle_smallest_angle_measure :
  ∀ (A B C : ℝ),
  let α := 1.6 * 90,
  α < 180 →
  α > 90 →
  2 * B + α = 180 →
  B = 18 :=
by
  intros A B C α h1 h2 h3
  sorry

end isosceles_obtuse_triangle_smallest_angle_measure_l346_346956


namespace donna_dryers_count_l346_346189

/-- Define the parameters and conditions given in the problem --/
variables (bridge_limit : ℕ) (empty_truck_weight : ℕ) (soda_crates : ℕ) (soda_weight_per_crate : ℕ)
          (dryer_weight : ℕ) (fresh_produce_multiple : ℕ) (fully_loaded_truck_weight : ℕ)

/-- Define the specific values for this problem --/
def bridge_limit := 20000
def empty_truck_weight := 12000
def soda_crates := 20
def soda_weight_per_crate := 50
def dryer_weight := 3000
def fresh_produce_multiple := 2
def fully_loaded_truck_weight := 24000

/-- Define the calculation of total soda weight and fresh produce weight --/
def total_soda_weight := soda_crates * soda_weight_per_crate
def fresh_produce_weight := fresh_produce_multiple * total_soda_weight
def base_truck_weight := empty_truck_weight + total_soda_weight + fresh_produce_weight
def total_dryer_weight := fully_loaded_truck_weight - base_truck_weight

/-- Finally, prove that Donna is carrying 3 dryers --/
theorem donna_dryers_count : total_dryer_weight / dryer_weight = 3 := by
  sorry

end donna_dryers_count_l346_346189


namespace glucose_amount_in_45cc_l346_346081

noncomputable def glucose_in_container (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) : ℝ :=
  (concentration * poured_volume) / total_volume

theorem glucose_amount_in_45cc (concentration : ℝ) (total_volume : ℝ) (poured_volume : ℝ) :
  concentration = 10 → total_volume = 100 → poured_volume = 45 →
  glucose_in_container concentration total_volume poured_volume = 4.5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end glucose_amount_in_45cc_l346_346081


namespace locus_of_P_l346_346619

variables {P O Q : Type} [metric_space P]

-- Conditions
def distance (P O : P) := dist P O = 5

def angle (Q O P : P) := ∠ Q O P = π / 3  -- 60 degrees in radians

-- Correct Answer: The locus of P consists of up to two intersection points
theorem locus_of_P (P O Q : P) [hqop : ∠ Q O P = π / 3] : 
  {P | dist P O = 5 ∧ ∠ Q O P = π / 3}.finite :=
sorry -- Proof of finiteness not required, just the statement.

end locus_of_P_l346_346619


namespace Tanika_boxes_l346_346017

/--
Tanika is selling boxes of crackers for her scout troop's fund-raiser.
On Saturday, she sold 60 boxes.
On Sunday, she sold 50% more than on Saturday.
Prove that the total number of boxes she sold, in total, over the two days is 150.
-/
theorem Tanika_boxes (sunday_more_than_saturday : real) (saturday_boxes : real) (sunday_percentage_increase : real)
    (h1 : saturday_boxes = 60) (h2 : sunday_percentage_increase = 0.5) 
    (h3 : sunday_more_than_saturday = saturday_boxes * sunday_percentage_increase) : 
    saturday_boxes + (saturday_boxes + sunday_more_than_saturday) = 150 :=
by
  sorry

end Tanika_boxes_l346_346017


namespace smallest_four_digit_in_pascal_triangle_l346_346562

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346562


namespace log_base_change_l346_346237

theorem log_base_change (a b : ℝ) (h1 : real.logb 3 5 = a) (h2 : real.logb 3 7 = b) : 
  real.logb 15 35 = (a + b) / (1 + a) :=
by 
  sorry

end log_base_change_l346_346237


namespace train_cars_count_l346_346837

theorem train_cars_count (cars_passed_in_12sec : ℕ) (total_pass_time_sec : ℕ) 
  (car_rate : (cars_passed_in_12sec : ℤ) / (12 : ℤ) = (2 / 3 : ℤ)) 
  (total_cars : (total_pass_time_sec : ℤ) = (12 + 3 * 60) : ℤ ) : 
  ∃ (n : ℕ), n = 140 :=
by 
  use 140 
  sorry

end train_cars_count_l346_346837


namespace sum_of_coordinates_l346_346297

-- Define the function f and specify that f(5) = 10
def f : ℝ → ℝ := sorry
axiom f_at_5 : f 5 = 10

-- Define the function k as the square of f
def k (x : ℝ) : ℝ := (f x)^2

-- State the point and the problem to be proved
theorem sum_of_coordinates : (5, k 5) = (5, 100) → 5 + k 5 = 105 := 
by
  intro h
  rw [←h]
  simp [f_at_5]
  sorry

end sum_of_coordinates_l346_346297


namespace smallest_four_digit_in_pascals_triangle_l346_346439

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346439


namespace triangle_square_ratio_l346_346138

theorem triangle_square_ratio :
  ∀ (x y : ℝ), (x = 60 / 17) → (y = 780 / 169) → (x / y = 78 / 102) :=
by
  intros x y hx hy
  rw [hx, hy]
  -- the proof is skipped, as instructed
  sorry

end triangle_square_ratio_l346_346138


namespace largest_prime_factor_sum_divisors_144_l346_346811

theorem largest_prime_factor_sum_divisors_144 :
  ∃ p : ℕ, Nat.Prime p ∧ p = 31 ∧ 
  (∀ q : ℕ, Nat.Prime q ∧ q ∣ (Finset.sum (Finset.filter (λ x, (Nat.gcd x 144) = 1) (Finset.range 145))) → q ≤ p) :=
begin
  sorry
end

end largest_prime_factor_sum_divisors_144_l346_346811


namespace limit_inscribed_circle_series_l346_346148

noncomputable def inscribed_circle_area (s : ℝ) : ℝ := (Real.pi * s^2) / 12

noncomputable def circle_series_sum (s : ℝ) (n : ℕ) : ℝ := 
  (inscribed_circle_area s) * ((1 - (1/2)^n) / (1 - 1/2))

theorem limit_inscribed_circle_series :
  ∀ (s : ℝ), (s > 0) → (∀ n : ℕ, ∃ k : ℕ, n ≥ k) →
  ∃ L : ℝ, filter.tendsto (λ n, circle_series_sum s n) filter.at_top (nhds (Real.pi * s^2 / 6)) :=
begin
  intros s hs h,
  use Real.pi * s^2 / 6,
  sorry
end

end limit_inscribed_circle_series_l346_346148


namespace kenny_jumping_jacks_l346_346805

theorem kenny_jumping_jacks : 
  let Sunday := 324 - ((20) + 123 + 64 + 23 + 61) in
  Sunday = 33 :=
by
  let Sunday := 324 - (20 + 123 + 64 + 23 + 61)
  show Sunday = 33
  sorry

end kenny_jumping_jacks_l346_346805


namespace smallest_four_digit_in_pascal_l346_346507

theorem smallest_four_digit_in_pascal :
  ∃ n k : ℕ, binomial n k = 1000 ∧ 
             (∀ a b : ℕ, binomial a b ∈ set.Icc 1000 9999 → a ≥ n) :=
sorry

end smallest_four_digit_in_pascal_l346_346507


namespace total_dogs_l346_346408

-- Definitions of conditions
def brown_dogs : Nat := 20
def white_dogs : Nat := 10
def black_dogs : Nat := 15

-- Theorem to prove the total number of dogs
theorem total_dogs : brown_dogs + white_dogs + black_dogs = 45 := by
  -- Placeholder for proof
  sorry

end total_dogs_l346_346408


namespace number_of_triangles_in_decagon_l346_346983

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346983


namespace total_difference_in_cups_l346_346370

theorem total_difference_in_cups (h1: Nat) (h2: Nat) (h3: Nat) (hrs: Nat) : 
  h1 = 4 → h2 = 7 → h3 = 5 → hrs = 3 → 
  ((h2 * hrs - h1 * hrs) + (h3 * hrs - h1 * hrs) + (h2 * hrs - h3 * hrs)) = 18 :=
by
  intros h1_eq h2_eq h3_eq hrs_eq
  sorry

end total_difference_in_cups_l346_346370


namespace smallest_four_digit_in_pascals_triangle_l346_346480

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346480


namespace ten_times_L_l346_346340

theorem ten_times_L {a b c d : ℕ} (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)
 (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
 (h2 : odd (a + b) ∧ odd (a + c) ∧ odd (a + d))
 (h3 : ∃ x y z : ℕ, (a + b) = x^2 ∧ (a + c) = y^2 ∧ (a + d) = z^2)
 (h_min : ∀ a' b' c' d' : ℕ, a' + b' + c' + d' < a + b + c + d → False) :
 10 * (a + b + c + d) = 670 :=
sorry

end ten_times_L_l346_346340


namespace smallest_four_digit_in_pascals_triangle_l346_346441

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346441


namespace find_k_of_sequence_l346_346267

theorem find_k_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 9 * n)
  (hS_recurr : ∀ n ≥ 2, a n = S n - S (n-1)) (h_a_k : ∃ k, 5 < a k ∧ a k < 8) : ∃ k, k = 8 :=
by
  sorry

end find_k_of_sequence_l346_346267


namespace number_of_triangles_in_decagon_l346_346977

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346977


namespace triangles_from_decagon_l346_346996

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346996


namespace probability_of_drawing_2_l346_346895

theorem probability_of_drawing_2 : 
  let cards := [1, 2, 2, 3, 5] in
  let total_number_of_cards := 5 in
  let number_of_2_cards := 2 in
  (number_of_2_cards / total_number_of_cards : ℚ) = 2 / 5 :=
by
  -- Proof goes here
  sorry

end probability_of_drawing_2_l346_346895


namespace smallest_four_digit_in_pascals_triangle_l346_346422

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346422


namespace part1_part2_l346_346257

def f (a x : ℝ) : ℝ := |x + 1| - |x - a|

theorem part1 (x : ℝ) :
  f 1 x ≥ 1 ↔ x ∈ (set.Ici (1 / 2)) := sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ 0, f a x < 2) ↔ (a < 1) := sorry

end part1_part2_l346_346257


namespace regular_polygon_pair_count_l346_346401

theorem regular_polygon_pair_count : 
  ∃ r k : ℕ, (180 - 360 / r) / (180 - 360 / k) = 5 / 3 ∧ r * 1 + k * 1 = 30 ∧ 2 = 1 :=
begin
  sorry
end

end regular_polygon_pair_count_l346_346401


namespace num_ordered_triples_unique_l346_346202

theorem num_ordered_triples_unique : 
  (∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1) := 
by 
  sorry 

end num_ordered_triples_unique_l346_346202


namespace initial_guests_count_l346_346325

theorem initial_guests_count (x : ℕ) (h1 : 1 / 3 * x).nat_floor = x / 3 
                            (h2 : 3 / 5 * (2 / 3 * x)).nat_floor = x * 2 / 15
                            (h3 : 4 = 4 * (x / 15)).nat_floor = x := 
by
  sorry

end initial_guests_count_l346_346325


namespace tan_values_l346_346285

theorem tan_values (a b : ℝ) (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 2) = 0) : 
  let x := Real.tan (a / 2)
  let y := Real.tan (b / 2)
  x * y = 4 ∨ x * y = -4 :=
by {
  -- Due to the problem's requirements, the proof steps are not included.
  let x := Real.tan (a / 2),
  let y := Real.tan (b / 2),
  sorry
}

end tan_values_l346_346285


namespace calvin_final_weight_l346_346171

def initial_weight : ℕ := 250
def weight_loss_per_month : ℕ := 8
def duration_months : ℕ := 12
def total_weight_loss : ℕ := weight_loss_per_month * duration_months
def final_weight : ℕ := initial_weight - total_weight_loss

theorem calvin_final_weight : final_weight = 154 :=
by {
  have h1 : total_weight_loss = 96 := by norm_num,
  rw [h1],
  norm_num,
  sorry
}

-- We have used 'sorry' to mark the place where the proof would be completed.

end calvin_final_weight_l346_346171


namespace price_change_on_eggs_and_apples_l346_346309

theorem price_change_on_eggs_and_apples :
  let initial_egg_price := 1.00
  let initial_apple_price := 1.00
  let egg_drop_percent := 0.10
  let apple_increase_percent := 0.02
  let new_egg_price := initial_egg_price * (1 - egg_drop_percent)
  let new_apple_price := initial_apple_price * (1 + apple_increase_percent)
  let initial_total := initial_egg_price + initial_apple_price
  let new_total := new_egg_price + new_apple_price
  let percent_change := ((new_total - initial_total) / initial_total) * 100
  percent_change = -4 :=
by
  sorry

end price_change_on_eggs_and_apples_l346_346309


namespace manager_monthly_salary_l346_346379

theorem manager_monthly_salary :
  let avg_salary := 1200
  let num_employees := 20
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + 100
  let num_people_with_manager := num_employees + 1
  let new_total_salary := num_people_with_manager * new_avg_salary
  let manager_salary := new_total_salary - total_salary
  manager_salary = 3300 := by
  sorry

end manager_monthly_salary_l346_346379


namespace car_dealership_percentage_l346_346114

theorem car_dealership_percentage (N₁ N₂ : ℕ) (P₁ P₂ : ℝ) (hN₁ : N₁ = 40) (hP₁ : P₁ = 0.20) (hN₂ : N₂ = 80) (hP₂ : P₂ = 0.35) :
  let silver_initial := P₁ * N₁,
      non_silver_new := P₂ * N₂,
      silver_new := (1 - P₂) * N₂,
      total_silver := silver_initial + silver_new,
      total_cars := N₁ + N₂
  in (total_silver / total_cars) * 100 = 50 :=
by
  sorry

end car_dealership_percentage_l346_346114


namespace jaden_toy_cars_l346_346321

theorem jaden_toy_cars :
  ∀ (initial: ℕ) (birthday: ℕ) (sister: ℕ) (friend: ℕ) (left: ℕ) (bought: ℕ),
  initial = 14 →
  birthday = 12 →
  sister = 8 →
  friend = 3 →
  left = 43 →
  bought = left + sister + friend - (initial + birthday) →
  bought = 28 :=
by
  intros initial birthday sister friend left bought
  assume h1 h2 h3 h4 h5 h6
  sorry

end jaden_toy_cars_l346_346321


namespace exists_natural_sum_l346_346095

noncomputable def exists_alpha (α : ℝ) : Prop :=
  α > 2 ∧ ∀ k : ℕ, k ≥ 1 → irrational (α ^ k)

theorem exists_natural_sum (α : ℝ) (hα : exists_alpha α) :
  ∀ n : ℕ, ∃ (a b : ℕ) (k : ℕ), (b ≤ 6) ∧ (n = a + b * (α ^ k)) :=
by sorry

end exists_natural_sum_l346_346095


namespace triangles_from_decagon_l346_346999

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346999


namespace smallest_four_digit_number_in_pascals_triangle_l346_346445

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346445


namespace total_time_is_12_years_l346_346131

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l346_346131


namespace parallelogram_area_example_l346_346753

def point := (ℚ × ℚ)
def parallelogram_area (A B C D : point) : ℚ :=
  let base := B.1 - A.1
  let height := C.2 - A.2
  base * height

theorem parallelogram_area_example : 
  parallelogram_area (1, 1) (7, 1) (4, 9) (10, 9) = 48 := by
  sorry

end parallelogram_area_example_l346_346753


namespace other_juice_costs_5_l346_346604

variable (total_spent : ℕ) (pineapple_spent : ℕ) (pineapple_price : ℕ) (total_people : ℕ)

def cost_per_glass_other_juice (total_spent pineapple_spent pineapple_price total_people : ℕ) : ℕ :=
  let pineapple_glasses := pineapple_spent / pineapple_price in
  let other_people := total_people - pineapple_glasses in
  let other_spent := total_spent - pineapple_spent in
  other_spent / other_people

theorem other_juice_costs_5 :
  total_spent = 94 →
  pineapple_spent = 54 →
  pineapple_price = 6 →
  total_people = 17 →
  cost_per_glass_other_juice 94 54 6 17 = 5 := sorry

end other_juice_costs_5_l346_346604


namespace calculation_l346_346181

def M (x : ℝ) := 0.4 * x + 2

theorem calculation : M (M (M 40)) = 5.68 :=
by
  simp [M]
  simp only [M]
  sorry

end calculation_l346_346181


namespace sum_of_array_elements_lt_n_l346_346940

theorem sum_of_array_elements_lt_n {n : ℕ} (h_odd : n % 2 = 1) 
    (a : Fin n → Fin n → ℝ) 
    (h_abs : ∀ i j, |a i j| < 1) 
    (h_sum_2x2 : ∀ i j, i < n - 1 → j < n - 1 → 
                       a i j + a (i+1) j + a i (j+1) + a (i+1) (j+1) = 0) : 
    (∑ i, ∑ j, a i j) < n := 
by
  sorry

end sum_of_array_elements_lt_n_l346_346940


namespace lcm_equality_iff_not_power_of_prime_l346_346338

def is_lcm (a : ℕ) (l : ℕ) : Prop := ∀ b, b ∣ l ↔ ∀ c ∈ finset.Icc 1 a, c ∣ b

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ p k : ℕ, p.prime ∧ k > 0 ∧ n = p^k

theorem lcm_equality_iff_not_power_of_prime (n : ℕ) :
  (∃ a, is_lcm (n - 1) a ∧ ∃ b, is_lcm n b ∧ a = b) ↔ ¬ is_power_of_prime n :=
  sorry

end lcm_equality_iff_not_power_of_prime_l346_346338


namespace simplify_expression_l346_346162

theorem simplify_expression (m : ℝ) (h : m ≠ 0) : (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end simplify_expression_l346_346162


namespace minimum_value_x_squared_plus_12x_plus_5_l346_346071

theorem minimum_value_x_squared_plus_12x_plus_5 : ∃ x : ℝ, x^2 + 12 * x + 5 = -31 :=
by sorry

end minimum_value_x_squared_plus_12x_plus_5_l346_346071


namespace gcd_40_120_45_l346_346417

theorem gcd_40_120_45 : Nat.gcd (Nat.gcd 40 120) 45 = 5 :=
by
  sorry

end gcd_40_120_45_l346_346417


namespace smallest_four_digit_in_pascal_triangle_l346_346555

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346555


namespace solve_for_a_l346_346696

open Set

variable (a : ℝ)

-- Define the sets A and B using the conditions given
def A : Set ℝ := {a^2, a + 1, -3}
def B : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}

-- State the condition that A ∩ B = {-3}
def condition : A ∩ B = {-3} := sorry

-- Formulate the theorem
theorem solve_for_a (h : A ∩ B = {-3}) : a = -1 := 
by
  sorry

end solve_for_a_l346_346696


namespace smallest_four_digit_number_in_pascals_triangle_l346_346540

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346540


namespace problem_solution_l346_346654

-- Definitions for the statements
def statement_I (a b : ℝ) : Prop := sqrt (a^2 + b^2) = 0
def statement_II (a b : ℝ) : Prop := sqrt (a^2 + b^2) = a * b
def statement_III (a b : ℝ) : Prop := sqrt (a^2 + b^2) = a + b
def statement_IV (a b : ℝ) : Prop := sqrt (a^2 + b^2) = a * b

-- The theorem that to be proved
theorem problem_solution : (∀ a b : ℝ, statement_I a b → a = 0 ∧ b = 0) ∧
                           (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_II a b) ∧
                           (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_III a b) ∧
                           (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_IV a b) :=
by
  -- There are no solutions for these conditions
  have h1 : ∀ a b : ℝ, statement_I a b → a = 0 ∧ b = 0 := sorry
  -- There exist non-zero solutions for these conditions
  have h2 : ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_II a b := sorry
  have h3 : ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_III a b := sorry
  have h4 : ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ statement_IV a b := sorry
  exact ⟨h1, h2, h3, h4⟩

end problem_solution_l346_346654


namespace time_addition_correct_l346_346802

theorem time_addition_correct :
  let current_time := (3, 0, 0)  -- Representing 3:00:00 PM as a tuple (hours, minutes, seconds)
  let duration := (313, 45, 56)  -- Duration to be added: 313 hours, 45 minutes, and 56 seconds
  let new_time := ((3 + (313 % 12) + 45 / 60 + (56 / 3600)), (0 + 45 % 60), (0 + 56 % 60))
  let A := (4 : ℕ)  -- Extracted hour part of new_time
  let B := (45 : ℕ)  -- Extracted minute part of new_time
  let C := (56 : ℕ)  -- Extracted second part of new_time
  A + B + C = 105 := 
by
  -- Placeholder for the actual proof.
  sorry

end time_addition_correct_l346_346802


namespace smallest_four_digit_in_pascals_triangle_l346_346433

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346433


namespace triangles_from_decagon_l346_346995

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346995


namespace problem_solution_l346_346271

noncomputable def set_M (x : ℝ) : Prop := x^2 - 4*x < 0
noncomputable def set_N (m x : ℝ) : Prop := m < x ∧ x < 5
noncomputable def set_intersection (x : ℝ) : Prop := 3 < x ∧ x < 4

theorem problem_solution (m n : ℝ) :
  (∀ x, set_M x ↔ (0 < x ∧ x < 4)) →
  (∀ x, set_N m x ↔ (m < x ∧ x < 5)) →
  (∀ x, (set_M x ∧ set_N m x) ↔ set_intersection x) →
  m + n = 7 :=
by
  intros H1 H2 H3
  sorry

end problem_solution_l346_346271


namespace complete_square_quadratic_t_l346_346360

theorem complete_square_quadratic_t : 
  ∀ x : ℝ, (16 * x^2 - 32 * x - 512 = 0) → (∃ q t : ℝ, (x + q)^2 = t ∧ t = 33) :=
by sorry

end complete_square_quadratic_t_l346_346360


namespace smallest_four_digit_number_in_pascals_triangle_l346_346542

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346542


namespace alpha_range_midpoint_trajectory_l346_346315

noncomputable def circle_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * Real.pi) :
  (Real.tan α) > 1 ∨ (Real.tan α) < -1 ↔ (Real.pi / 4 < α ∧ α < 3 * Real.pi / 4) ∨ 
                                          (5 * Real.pi / 4 < α ∧ α < 7 * Real.pi / 4) := 
  sorry

theorem midpoint_trajectory (m : ℝ) (h2 : -1 < m ∧ m < 1) :
  ∃ x y : ℝ, x = (Real.sqrt 2 * m) / (m^2 + 1) ∧ 
             y = -(Real.sqrt 2 * m^2) / (m^2 + 1) :=
  sorry

end alpha_range_midpoint_trajectory_l346_346315


namespace segments_connecting_midpoints_l346_346023

theorem segments_connecting_midpoints (c d : ℝ) (h_diag_angle : ∠ A O B = 45) :
  ∃ MN KP : ℝ, MN = 1 / 2 * √(c^2 + d^2 - c * d * √2) ∧ KP = 1 / 2 * √(c^2 + d^2 + c * d * √2) :=
begin
  sorry
end

end segments_connecting_midpoints_l346_346023


namespace number_less_than_reciprocal_greater_than_neg3_l346_346911

theorem number_less_than_reciprocal_greater_than_neg3 (x : ℚ) :
  (x = -3/2 ∨ x = 1/2) → (x > -3 ∧ x < 1/x) :=
by {
  intro h,
  cases h,
  { subst h, norm_num, split,
    { norm_num },
    { norm_num } },
  { subst h, norm_num, split,
    { norm_num },
    { norm_num } }
  }

end number_less_than_reciprocal_greater_than_neg3_l346_346911


namespace smallest_four_digit_in_pascals_triangle_l346_346442

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346442


namespace smallest_four_digit_number_in_pascals_triangle_l346_346447

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346447


namespace exists_finite_set_of_lattice_points_l346_346611

def lattice_point (p : Point) : Prop :=
  ∃ x y : ℤ, p = (x, y)

def line_with_given_slope (l : Line) : Prop :=
  l.slope = 0 ∨ l.slope = 1 ∨ l.slope = -1 ∨ l.slope = Option.none

theorem exists_finite_set_of_lattice_points :
  ∃ (S : set Point), (finite S) ∧
  (∀ (l : Line), (line_with_given_slope l) → 
    (∃ (n : ℕ), n = 2022 ∧ l ∩ S = n → (l ∩ S = ∅ ∨ l ∩ S = 2022))) :=
sorry

end exists_finite_set_of_lattice_points_l346_346611


namespace triangle_ABC_interior_angle_A_eq_120_l346_346241

open real

variables {A B C O : Point}
variables (OA OB OC : Vector)

def center_of_circumcircle_of_triangle (O A B C : Point) : Prop :=
  ∃ (R : ℝ), dist O A = R ∧ dist O B = R ∧ dist O C = R

def vector_sum_zero (OA OB OC : Vector) : Prop :=
  OA + OB + OC = 0

def angle_A_eq_120 (O A B C : Point) (OA OB OC : Vector) 
  (center_cond : center_of_circumcircle_of_triangle O A B C)
  (vector_cond : vector_sum_zero OA OB OC) : Prop :=
  ∠A = 120

-- Math problem proving that angle A of triangle ABC is 120 degrees given conditions
theorem triangle_ABC_interior_angle_A_eq_120 {O A B C : Point} (OA OB OC : Vector)
  (center_cond : center_of_circumcircle_of_triangle O A B C)
  (vector_cond : vector_sum_zero OA OB OC) : angle_A_eq_120 O A B C OA OB OC center_cond vector_cond :=
sorry

end triangle_ABC_interior_angle_A_eq_120_l346_346241


namespace side_length_of_inscribed_decagon_l346_346356

theorem side_length_of_inscribed_decagon
  (A B C D O : Point)
  (r : Real)
  (h_circle : Circle O r)
  (h_A_on_circle : A ∈ h_circle)
  (h_B_on_circle : B ∈ h_circle)
  (h_C_on_circle : C ∈ h_circle)
  (h_AB_hex : dist A B = r)
  (h_BD_square : dist B D = r * Real.sqrt 2)
  (h_C_square : dist A C = r * Real.sqrt 2)
  (h_AD_extension : lies_on_extension O A D) :
  dist A D = (r / 2) * (Real.sqrt 5 - 1) :=
sorry

end side_length_of_inscribed_decagon_l346_346356


namespace smallest_four_digit_number_in_pascals_triangle_l346_346453

theorem smallest_four_digit_number_in_pascals_triangle : ∃ n k, binom n k = 1000 ∧ (∀ n' k', binom n' k' > 1000 → nat.zero < n' k' ∧ n' k' < (binom n k)) := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346453


namespace minimum_value_of_phi_l346_346876

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def minimum_positive_period (ω : ℝ) := 2 * Real.pi / ω

theorem minimum_value_of_phi {A ω φ : ℝ} (hA : A > 0) (hω : ω > 0) 
  (h_period : minimum_positive_period ω = Real.pi) 
  (h_symmetry : ∀ x, f A ω φ x = f A ω φ (2 * Real.pi / ω - x)) : 
  ∃ k : ℤ, |φ| = |k * Real.pi - Real.pi / 6| → |φ| = Real.pi / 6 :=
by
  sorry

end minimum_value_of_phi_l346_346876


namespace _l346_346898

-- Definition of side lengths of squares
def side1 : ℕ := 8
def side2 : ℕ := 11
def hypotenuse : ℕ := 13

-- Pythagorean theorem check
lemma pythagorean_theorem : side1^2 + side2^2 = hypotenuse^2 := by
  sorry

-- Area of the triangle formed by the squares
lemma area_of_triangle : (1 / 2 : ℝ) * side1.toReal * side2.toReal = 44 := by
  sorry

end _l346_346898


namespace smallest_four_digit_in_pascals_triangle_l346_346533

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346533


namespace seating_permutations_count_l346_346146

-- Define people
inductive Person
| Alice
| Bob
| Carla
| Derek
| Eric
deriving DecidableEq, Repr

-- Define the seating restriction conditions
def seating_conditions (seating: List Person) : Prop :=
  -- Alice refuses to sit next to either Bob or Carla
  (∃ i, seating[i] = Person.Alice ∧ (i > 0 → seating[i - 1] ≠ Person.Bob ∧ seating[i - 1] ≠ Person.Carla) ∧ (i < 4 → seating[i + 1] ≠ Person.Bob ∧ seating[i + 1] ≠ Person.Carla)) ∧
  -- Derek refuses to sit next to Eric
  (∀ i, seating[i] = Person.Derek → (i > 0 → seating[i - 1] ≠ Person.Eric) ∧ (i < 4 → seating[i + 1] ≠ Person.Eric)) ∧
  -- Carla refuses to sit next to Bob
  (∀ i, seating[i] = Person.Carla → (i > 0 → seating[i - 1] ≠ Person.Bob) ∧ (i < 4 → seating[i + 1] ≠ Person.Bob))

-- Define the theorem to prove the number of valid configurations is 6
theorem seating_permutations_count : 
  {s : List Person // s.length = 5 ∧ seating_conditions s}.card = 6 := sorry

end seating_permutations_count_l346_346146


namespace sum_of_all_3_digit_numbers_l346_346580

def three_digit_numbers_sum (digits : List ℕ) : ℕ :=
  let perms := digits.permutations
  perms.map (λ l, l.foldl (λ acc x => 10 * acc + x) 0).sum

theorem sum_of_all_3_digit_numbers : 
  three_digit_numbers_sum [2, 3, 4] = 1998 :=
by
  sorry

end sum_of_all_3_digit_numbers_l346_346580


namespace calvin_final_weight_l346_346172

def initial_weight : ℕ := 250
def weight_loss_per_month : ℕ := 8
def duration_months : ℕ := 12
def total_weight_loss : ℕ := weight_loss_per_month * duration_months
def final_weight : ℕ := initial_weight - total_weight_loss

theorem calvin_final_weight : final_weight = 154 :=
by {
  have h1 : total_weight_loss = 96 := by norm_num,
  rw [h1],
  norm_num,
  sorry
}

-- We have used 'sorry' to mark the place where the proof would be completed.

end calvin_final_weight_l346_346172


namespace newtons_number_l346_346145

theorem newtons_number (a n : ℂ) (h : a * n = 40 - 24 * complex.I) (ha : a = 8 - 4 * complex.I) : n = 2.8 - 0.4 * complex.I := 
sorry

end newtons_number_l346_346145


namespace housewife_left_money_l346_346608

def initial_amount : ℝ := 150
def spent_fraction : ℝ := 2 / 3
def remaining_fraction : ℝ := 1 - spent_fraction

theorem housewife_left_money :
  initial_amount * remaining_fraction = 50 := by
  sorry

end housewife_left_money_l346_346608


namespace ammonium_chloride_production_l346_346760

def hydrochloric_acid_moles : ℕ := 3
def ammonia_moles : ℕ := 3
def ammonium_chloride_moles (h : hydrochloric_acid_moles = ammonia_moles) : ℕ := hydrochloric_acid_moles

theorem ammonium_chloride_production
  (h : hydrochloric_acid_moles = ammonia_moles) :
  ammonium_chloride_moles h = 3 :=
by
  unfold ammonium_chloride_moles
  rw h
  exact rfl

end ammonium_chloride_production_l346_346760


namespace identify_proposition_l346_346955

theorem identify_proposition : 
  (∃ a : ℕ, prime a ∧ (a ≠ 2 → odd a)) ↔ 
  ("Is the exponential function increasing?" ∨ 
   "Prove that the square root of 2 is irrational." ∨ 
   ∃ x : ℝ, x > 15) :=
by sorry

end identify_proposition_l346_346955


namespace hyperbola_standard_equation_l346_346385

theorem hyperbola_standard_equation (a b : ℝ) (h_asymptote : a = b) (h_point : (2 : ℝ) = a ∧ (4 : ℝ) = b) : 
  ∃ k : ℝ, y = b ∧ x = a := y^2 - x^2 = k := -12) :=  sorry

end hyperbola_standard_equation_l346_346385


namespace baseball_wins_l346_346615

-- Define the constants and conditions
def total_games : ℕ := 130
def won_more_than_lost (L W : ℕ) : Prop := W = 3 * L + 14
def total_games_played (L W : ℕ) : Prop := W + L = total_games

-- Define the theorem statement
theorem baseball_wins (L W : ℕ) 
  (h1 : won_more_than_lost L W)
  (h2 : total_games_played L W) : 
  W = 101 :=
  sorry

end baseball_wins_l346_346615


namespace number_of_triangles_in_decagon_l346_346987

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346987


namespace ratio_YQ_QG_l346_346798

variables {X Y Z F G Q : Point}
variables {XY YZ XZ : ℝ}
variables {xf yg : Line}
variables (xy_len : XY = 6) (xz_len : XZ = 4) (yz_len : YZ = 5)
variables (angle_bisec_xf : Line) (angle_bisec_yg : Line)
variables (q_intersection : Q = intersection angle_bisec_xf angle_bisec_yg)

theorem ratio_YQ_QG (triangle_XYZ : triangle X Y Z)
  (angle_bisec_condition : angle_bisector triangle_XYZ angle_bisec_xf ∧ angle_bisector triangle_XYZ angle_bisec_yg)
  (xy_len: XY = 6) (xz_len: XZ = 4) (yz_len: YZ = 5)
  (ratio_YQ_QG_eq : YQ/QG = 2) : 
  (YQ/QG = 2) := sorry

end ratio_YQ_QG_l346_346798


namespace last_two_nonzero_digits_of_80_factorial_l346_346185

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Problem: Prove that the last two nonzero digits of 80! are 21
theorem last_two_nonzero_digits_of_80_factorial :
  last_two_nonzero_digits (fact 80) = 21 :=
sorry

-- Auxiliary function to find the last two nonzero digits
def last_two_nonzero_digits (n : ℕ) : ℕ :=
let digits := nat.digits 10 n in
let nonzero_digits := digits.filter (λ d, d ≠ 0) in
(nonzero_digits.nth (nonzero_digits.length - 2)).iget * 10 + (nonzero_digits.nth (nonzero_digits.length - 1)).iget

end last_two_nonzero_digits_of_80_factorial_l346_346185


namespace area_of_sector_of_central_angle_is_one_l346_346381

-- Problem statement
theorem area_of_sector_of_central_angle_is_one:
  (∀ r, 2 * sin 1 = 2 * r * (sin 1)) → 
  (1/2 : ℝ) * (1 : ℝ) ^ 2 * (2 : ℝ) = 1 :=
by
  sorry

end area_of_sector_of_central_angle_is_one_l346_346381


namespace volume_of_sphere_is_correct_l346_346605

-- Define the conditions as hypotheses
variables {hexagon_base : Type} [is_regular_hexagon hexagon_base]
variables {prism : Type} [is_hexagonal_prism_with_base hexagon_base]
variables {edge_length : ℝ} (h_edge_length : edge_length = 1)
variables {vertices_on_sphere : Prop} (h_vertices_on_sphere : ∀ v ∈ vertices prism, v ∈ sphere)

-- Define the radius of the sphere
def radius_of_sphere : ℝ := (real.sqrt 5) / 2

-- Define the volume of a sphere
noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 * real.pi / 3) * r^3

-- Statement to prove
theorem volume_of_sphere_is_correct :
  volume_of_sphere radius_of_sphere = (5 * real.sqrt 5 * real.pi) / 6 :=
sorry

end volume_of_sphere_is_correct_l346_346605


namespace smallest_four_digit_number_in_pascals_triangle_l346_346459

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346459


namespace convex_quadrilateral_midpoints_squared_l346_346313

noncomputable def XY_squared (AB BC CD DA : ℝ) (angle_D : ℝ) (X Y : ℝ × ℝ) : ℝ := (X.1 - Y.1)^2 + (X.2 - Y.2)^2

theorem convex_quadrilateral_midpoints_squared :
  ∀ (A B C D X Y : ℝ × ℝ)
  (h_AB_BC : AB = 15) (h_BC_CD : BC = 15) 
  (h_CD_DA : CD = 20) (h_DA_AB : DA = 20)
  (h_angle_D : angle_D = 90) 
  (h_X_midpoint_BC : X = (B.1 + C.1) / 2, (B.2 + C.2) / 2)
  (h_Y_midpoint_DA : Y = (D.1 + A.1) / 2, (D.2 + A.2) / 2),
  XY_squared AB BC CD DA angle_D X Y = 156.25 :=
by
  sorry -- proof is to be filled in

end convex_quadrilateral_midpoints_squared_l346_346313


namespace triangles_from_decagon_l346_346989

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346989


namespace triangles_from_decagon_l346_346998

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l346_346998


namespace average_score_in_all_matches_l346_346087

theorem average_score_in_all_matches (runs_match1_match2 : ℤ) (runs_other_matches : ℤ) (total_matches : ℤ) 
  (average1 : ℤ) (average2 : ℤ)
  (h1 : average1 = 40) (h2 : average2 = 10) (h3 : runs_match1_match2 = 2 * average1)
  (h4 : runs_other_matches = 3 * average2) (h5 : total_matches = 5) :
  ((runs_match1_match2 + runs_other_matches) / total_matches) = 22 := 
by
  sorry

end average_score_in_all_matches_l346_346087


namespace jose_investment_l346_346051

theorem jose_investment (T_investment : ℕ) (T_months : ℕ) (profit : ℕ) (J_share : ℕ) (J_months : ℕ) (expected_J_investment : ℕ) :
  T_investment * T_months * J_share = (profit - J_share) * (J_months * expected_J_investment) → 
  expected_J_investment = 45000 :=
by
  intros h
  -- The rest of the proof would follow here
  sorry

-- Given data
def Tom_investment := 30000
def Tom_months := 12
def Total_profit := 36000
def Jose_share := 20000
def Jose_months := 10

example : jose_investment Tom_investment Tom_months Total_profit Jose_share Jose_months 45000 :=
by
  apply jose_investment
  -- The specific equation to be solved based on conditions
  sorry


end jose_investment_l346_346051


namespace pizzas_returned_l346_346136

theorem pizzas_returned (total_pizzas served_pizzas : ℕ) (h_total : total_pizzas = 9) (h_served : served_pizzas = 3) : (total_pizzas - served_pizzas) = 6 :=
by
  sorry

end pizzas_returned_l346_346136


namespace smallest_four_digit_number_in_pascals_triangle_l346_346548

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346548


namespace find_tax_rate_l346_346314

variable (total_value tax_free_limit tax_paid taxable_amount tax_rate : ℚ)

-- Given conditions
def condition1 : total_value = 730 := by sorry
def condition2 : tax_free_limit = 500 := by sorry
def condition3 : tax_paid = 18.40 := by sorry

-- Definitions based on conditions
def taxable_amount_def : taxable_amount = total_value - tax_free_limit := by sorry
def tax_rate_def : tax_rate = (tax_paid / taxable_amount) * 100 := by sorry

-- Proof statement
theorem find_tax_rate : tax_rate = 8 :=
by
  rw [taxable_amount_def, tax_rate_def, condition1, condition2, condition3]
  sorry

end find_tax_rate_l346_346314


namespace part1_part2_l346_346258

def g (x m : ℝ) := 2 * sqrt 3 * sin x * cos x + 2 * cos (x:ℝ)^2 + m

theorem part1 (h : ∀ x ∈ Icc 0 (π / 2), g x m ≤ 6) : m = 3 := sorry

def y (x : ℝ) := g (-x) 3

theorem part2 (k : ℤ) : ∃ I : Set ℝ, ∀ x ∈ I, y x ≤ y (x + ω) :=
  I ∈ set_of (λ x, k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6) := sorry

end part1_part2_l346_346258


namespace largest_possible_s_l346_346328

-- Define the condition for interior angles of regular polygons
def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

-- Define the specific condition for P₁ and P₂
def condition (r s : ℕ) := 
  r ≥ s ∧ s ≥ 3 ∧ (interior_angle r / interior_angle s = 57 / 56)

-- The theorem asserting the largest possible value of s
theorem largest_possible_s
  (s : ℕ)
  (r : ℕ)
  (h : condition r s) : s = 111 :=
sorry

end largest_possible_s_l346_346328


namespace not_divisible_l346_346809

-- Definitions and conditions
variables (a : ℕ) (a1 a2 a3 an : ℕ) (n : ℕ) 
  (h1 : 1 < a) (h2 : ∀ i, i ∈ (list.range n).map (λ i, i+1) → 0 < i)
  (h3 : a1 * a2 * a3 * an ∣ a)

-- A theorem that asserts the conclusion
theorem not_divisible (a : ℕ) (a1 a2 a3 an : ℕ) (n : ℕ) (h1 : 1 < a) 
  (h2 : (∀ i, i ∈ (list.range n).map (λ i, i+1) → 0 < i)) (h3 : a1 * a2 * a3 * an ∣ a) :
  ¬ (((a + a1 - 1) * (a + a2 - 1) * (a + a3 - 1) * (a + an - 1)) ∣ (a^(n+1) + a - 1)) :=
begin
  sorry
end

end not_divisible_l346_346809


namespace Sequential_structure_not_conditional_l346_346305

-- Definitions based on provided conditions
def is_conditional (s : String) : Prop :=
  s = "Loop structure" ∨ s = "If structure" ∨ s = "Until structure"

-- Theorem stating that Sequential structure is the one that doesn't contain a conditional judgment box
theorem Sequential_structure_not_conditional :
  ¬ is_conditional "Sequential structure" :=
by
  intro h
  cases h <;> contradiction

end Sequential_structure_not_conditional_l346_346305


namespace circumcenter_B_l346_346642

open EuclideanGeometry Real

noncomputable def ABCD_inscribed := sorry -- proof that ABCD is cyclic (concyclic)
noncomputable def angle_ABD := 60 -- angle ABD = 60 degrees
noncomputable def AE_eq_AD := sorry -- proof that AE = AD
noncomputable def E := sorry -- intersection point of AC and BD

theorem circumcenter_B (A B C D E F : Point) : 
  (cyclic [A, B, C, D]) → 
  (between A C E) → 
  (between B D E) →
  (∠ A B D = 60) →
  (segment A E = segment A D) →
  (∃ F : Point, intersects (line A B) (line D C) = F) →
  (circumcenter B (triangle C E F))
:=
sorry

end circumcenter_B_l346_346642


namespace parallelogram_area_l346_346705

-- Define a plane rectangular coordinate system
structure PlaneRectangularCoordinateSystem :=
(axis : ℝ)

-- Define the properties of a square
structure Square :=
(side_length : ℝ)

-- Define the properties of a parallelogram in a perspective drawing
structure Parallelogram :=
(side_length: ℝ)

-- Define the conditions of the problem
def problem_conditions (s : Square) (p : Parallelogram) :=
  s.side_length = 4 ∨ s.side_length = 8 ∧ 
  p.side_length = 4

-- Statement of the problem
theorem parallelogram_area (s : Square) (p : Parallelogram)
  (h : problem_conditions s p) :
  p.side_length * p.side_length = 16 ∨ p.side_length * p.side_length = 64 :=
by {
  sorry
}

end parallelogram_area_l346_346705


namespace P_10_value_l346_346011

-- Define the polynomial and conditions
def P (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8

-- Equations given in the problem
axiom a₀ : ℝ
axiom a₁ : ℝ
axiom a₂ : ℝ
axiom a₃ : ℝ
axiom a₄ : ℝ
axiom a₅ : ℝ
axiom a₆ : ℝ
axiom a₇ : ℝ
axiom a₈ : ℝ
axiom h₁ : P 1 = 1
axiom h₂ : P 2 = 1 / 2
axiom h₃ : P 3 = 1 / 3
axiom h₄ : P 4 = 1 / 4
axiom h₅ : P 5 = 1 / 5
axiom h₆ : P 6 = 1 / 6
axiom h₇ : P 7 = 1 / 7
axiom h₈ : P 8 = 1 / 8
axiom h₉ : P 9 = 1 / 9
axiom h₁₀ : ∀ x: ℝ, P x = P x
axiom h₁₁ : ∃ (b : ℝ), b = P 10

-- Theorem stating the required value of P(10)
theorem P_10_value : ∃! b, b = P 10 ∧ b = 1 / 5 := 
by 
  sorry

end P_10_value_l346_346011


namespace total_boxes_sold_over_two_days_l346_346014

def boxes_sold_on_Saturday : ℕ := 60

def percentage_more_on_Sunday : ℝ := 0.50

def boxes_sold_on_Sunday (boxes_sold_on_Saturday : ℕ) (percentage_more_on_Sunday : ℝ) : ℕ :=
  (boxes_sold_on_Saturday : ℝ) * (1 + percentage_more_on_Sunday) |> Nat.floor

def total_boxes_sold (saturday_sales : ℕ) (sunday_sales : ℕ) : ℕ :=
  saturday_sales + sunday_sales

theorem total_boxes_sold_over_two_days :
  total_boxes_sold boxes_sold_on_Saturday (boxes_sold_on_Sunday boxes_sold_on_Saturday percentage_more_on_Sunday) = 150 := by
  sorry

end total_boxes_sold_over_two_days_l346_346014


namespace abc_sum_l346_346372

theorem abc_sum :
  ∃ a b c : ℤ,
  ∀ x : ℤ[x], 
  (x^2 + a * x + b).gcd(x^2 + b * x + c) = (x + 1) ∧
  (x^2 + a * x + b).lcm(x^2 + b * x + c) = x^3 - x^2 - 4 * x + 4 ∧
  a + b + c = 7 :=
sorry

end abc_sum_l346_346372


namespace vector_linear_property_l346_346274

structure Vector (α : Type) :=
  (x : α) (y : α)

def f {α : Type} [Add α] [Sub α] : Vector α → Vector α
| ⟨x, y⟩ => ⟨y, (2 : α) * y - x⟩

variables {α : Type} [Add α] [Sub α] [Mul α] [HasSmul α α]

theorem vector_linear_property (a b : Vector α) (m n : α) :
  f (m • a + n • b) = m • f a + n • f b :=
by 
  sorry

end vector_linear_property_l346_346274


namespace line_perpendicular_iff_projection_perpendicular_l346_346353

-- Definitions corresponding to the conditions in a)
variables {α : Type*} [inner_product_space ℝ α]
variables {a l b : α}
variable {plane : set α}

-- Definitions for lines and planes
def line_in_plane (a : α) (plane : set α) : Prop := a ∈ plane
def perpendicular (a b : α) : Prop := ⟪a, b⟫ = 0
def orthogonal_projection (l : α) (plane : set α) : α := sorry  -- Assuming orthogonal_projection is defined

-- Given conditions
variable (h1 : line_in_plane a plane)
variable (h2 : perpendicular a l)
variable (h3 : b = orthogonal_projection l plane)

-- Theorem to prove the equivalence
theorem line_perpendicular_iff_projection_perpendicular : 
  (perpendicular a l ↔ perpendicular a (orthogonal_projection l plane)) :=
by sorry

end line_perpendicular_iff_projection_perpendicular_l346_346353


namespace find_x_l346_346747

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (sin x, 3/4)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (1/3, 1/2 * cos x)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (1/6, cos x)
noncomputable def vector_d (x : ℝ) : ℝ × ℝ := (1/2, 3/2 * cos x)

theorem find_x : 
  ∀ (x : ℝ), 
  vector_a x = (sin x, 3/4) →
  vector_b x = (1/3, 1/2 * cos x) →
  vector_c x = (1/6, cos x) →
  vector_d x = (1/2, 3/2 * cos x) →
  (0 < x ∧ x < 5 * π / 12) →
  (vector_a x ∥ vector_d x) →
  x = π / 12 := 
by
  intros
  sorry

end find_x_l346_346747


namespace smallest_four_digit_number_in_pascals_triangle_l346_346543

theorem smallest_four_digit_number_in_pascals_triangle :
  (pascal : ℕ → ℕ → ℕ) 
  (h_start : ∀ n, pascal n 0 = 1) 
  (h_increase : ∀ n k, pascal n k ≤ pascal n (k + 1)) : 
  ∃ n k, pascal n k = 1000 := 
sorry

end smallest_four_digit_number_in_pascals_triangle_l346_346543


namespace quadratic_equal_roots_l346_346888

theorem quadratic_equal_roots :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 → (0 ≤ 0) ∧ 
  (∀ a b : ℝ, 0 = b^2 - 4 * a * 1 → (x = -b / (2 * a))) :=
by
  sorry

end quadratic_equal_roots_l346_346888


namespace alcohol_mixture_percentage_l346_346923

theorem alcohol_mixture_percentage :
  let x := 30 in
  (∀ y : ℝ, y = x/100 * 2.5 + 0.5 * 7.5 → y = 0.45 * 10) →
  x = 30 :=
by
  sorry

end alcohol_mixture_percentage_l346_346923


namespace triangle_side_length_l346_346770

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 - c^2 = 2 * b)
  (h2 : sin A * cos C = 3 * (cos A * sin A)) :
  b = 4 :=
by
  sorry

end triangle_side_length_l346_346770


namespace smallest_four_digit_in_pascals_triangle_l346_346421

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346421


namespace semi_circle_perimeter_l346_346398

theorem semi_circle_perimeter (r : ℝ) (h : r = 3.1) : 
    let π := 3.14 in
    let half_circumference := π * r in
    let diameter := 2 * r in
    let P := half_circumference + diameter in
    P = 15.934 :=
by
    sorry

end semi_circle_perimeter_l346_346398


namespace amy_hours_per_week_l346_346856

theorem amy_hours_per_week {h w summer_salary school_weeks school_salary} 
  (hours_per_week_summer : h = 45)
  (weeks_summer : w = 8)
  (summer_salary_h : summer_salary = 3600)
  (school_weeks_h : school_weeks = 24)
  (school_salary_h : school_salary = 3600) :
  ∃ hours_per_week_school, hours_per_week_school = 15 :=
by
  sorry

end amy_hours_per_week_l346_346856


namespace diana_wins_l346_346187

noncomputable def probability_diana_wins : ℚ :=
  45 / 100

theorem diana_wins (d : ℕ) (a : ℕ) (hd : 1 ≤ d ∧ d ≤ 10) (ha : 1 ≤ a ∧ a ≤ 10) :
  probability_diana_wins = 9 / 20 :=
by
  sorry

end diana_wins_l346_346187


namespace ticket_price_divisors_l346_346630

theorem ticket_price_divisors :
  let x_vals := {x : ℕ | x ∣ 72 ∧ x ∣ 120} in
  x_vals.card = 8 :=
by
  sorry

end ticket_price_divisors_l346_346630


namespace correct_option_l346_346268

open Set

def M : Set ℕ := {x | x^2 - 1 = 0}

theorem correct_option : {-1, 0, 1} ∩ M = {1} :=
by 
  -- skipping the proof
  sorry

end correct_option_l346_346268


namespace members_play_both_l346_346915

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end members_play_both_l346_346915


namespace smallest_four_digit_number_in_pascals_triangle_l346_346465

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346465


namespace set_inter_complement_U_B_l346_346743

-- Define sets U, A, B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- Statement to prove
theorem set_inter_complement_U_B :
  A ∩ (Uᶜ \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end set_inter_complement_U_B_l346_346743


namespace expansion_num_terms_l346_346891

theorem expansion_num_terms (x y z : ℕ) : (x + y + z)^4 = 15 := 
begin 
  sorry 
end

end expansion_num_terms_l346_346891


namespace smallest_four_digit_in_pascals_triangle_l346_346471

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346471


namespace total_stamps_l346_346644

-- Definitions based on the conditions
def AJ := 370
def KJ := AJ / 2
def CJ := 2 * KJ + 5

-- Proof Statement
theorem total_stamps : AJ + KJ + CJ = 930 := by
  sorry

end total_stamps_l346_346644


namespace num_factors_of_46464_l346_346282

theorem num_factors_of_46464 : 
  let n := 46464
  let prime_factors := (3, 1) :: (2, 5) :: (11, 2) :: []
  (prime_factors.foldl (λ acc (p, k), acc * (k + 1)) 1) = 36 :=
by
  sorry

end num_factors_of_46464_l346_346282


namespace number_of_triangles_in_decagon_l346_346979

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346979


namespace average_score_remaining_students_l346_346119

theorem average_score_remaining_students (n : ℕ) (h : n > 15) (avg_all : ℚ) (avg_15 : ℚ) :
  avg_all = 12 → avg_15 = 20 →
  (∃ avg_remaining : ℚ, avg_remaining = (12 * n - 300) / (n - 15)) :=
by
  sorry

end average_score_remaining_students_l346_346119


namespace perimeter_of_triangle_l346_346317

-- Define the points P, Q, R
variables (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R]

-- Define the angles and sides of the triangle
variables (angle_PQR angle_PRQ : ℝ) (QR PR PQ : ℝ)

-- Conditions: given angles are equal and given sides
axiom angle_equality : angle_PQR = angle_PRQ
axiom QR_length : QR = 8
axiom PR_length : PR = 10

-- The geometry relationship from the isosceles triangle
axiom isosceles : angle_PQR = angle_PRQ → PR = PQ

-- The proof goal statement: perimeter of the triangle is 28
theorem perimeter_of_triangle : QR + PR + PQ = 28 :=
  by
  have PQ_eq_PR : PQ = PR := isosceles angle_equality
  rw [PQ_eq_PR, PR_length]
  rw QR_length
  sorry

end perimeter_of_triangle_l346_346317


namespace customer_paid_l346_346885

theorem customer_paid (cost_price : ℝ) (increase_percent : ℝ) (final_price : ℝ) :
  cost_price = 3840 ∧ increase_percent = 0.25 ∧ final_price = cost_price * (1 + increase_percent) →
  final_price = 4800 :=
by
  intros h
  cases h with hc1 h'
  cases h' with hi1 hf
  rw [hc1, hi1] at hf
  rw mul_add at hf
  rw mul_one at hf
  exact hf
-- sorry

end customer_paid_l346_346885


namespace sequence_first_four_terms_sequence_general_formula_l346_346278

noncomputable def sequence_a : ℕ → ℚ
| 0 := 7
| (n + 1) := 7 * sequence_a n / (sequence_a n + 7)

theorem sequence_first_four_terms :
  sequence_a 0 = 7 ∧
  sequence_a 1 = 7 / 2 ∧
  sequence_a 2 = 7 / 3 ∧
  sequence_a 3 = 7 / 4 :=
by
  split
  { rw sequence_a, norm_num },
  split
  { rw [sequence_a, sequence_a], norm_num },
  split
  { rw [sequence_a, sequence_a, sequence_a], norm_num },
  { rw [sequence_a, sequence_a, sequence_a, sequence_a], norm_num }

theorem sequence_general_formula (n : ℕ) (h : n > 0) : 
  sequence_a n = 7 / n :=
sorry

end sequence_first_four_terms_sequence_general_formula_l346_346278


namespace find_g_seven_l346_346878

noncomputable def g : ℝ → ℝ :=
  sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_six : g 6 = 7

theorem find_g_seven : g 7 = 49 / 6 :=
by
  -- Proof omitted here
  sorry

end find_g_seven_l346_346878


namespace smallest_four_digit_in_pascals_triangle_l346_346438

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346438


namespace largest_n_dividing_30_factorial_l346_346659

theorem largest_n_dividing_30_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, 30^m ∣ nat.factorial 30 ↔ m ≤ n) ∧ n = 7 := 
by sorry

end largest_n_dividing_30_factorial_l346_346659


namespace zero_point_interval_l346_346737

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.exp x
def f' (a x : ℝ) : ℝ := 2 * a * x - Real.exp x

theorem zero_point_interval (a : ℝ) (h_deriv : f' a (-1) = -4) :
  ∃ x ∈ Ioo (-1 : ℝ) (0 : ℝ), f a x = 0 :=
begin
  -- Definition of a
  have ha : a = 2 - 1 / (2 * Real.exp 1),
  { -- Derivation of a from the given condition
    sorry },
  -- Substitute a back into the function
  let g := λ x, (2 - 1 / (2 * Real.exp 1)) * x^2 - Real.exp x,
  -- Evaluate function values at specific points
  have hg_neg1 : g (-1) > 0 := by sorry,
  have hg_0 : g 0 < 0 := by sorry,
  -- Use Intermediate Value Theorem on the continuous function g on (-1, 0)
  exact exists_solution (g) hg_neg1 hg_0,
end

end zero_point_interval_l346_346737


namespace train_speed_is_50_l346_346575

-- Definitions of given conditions
def train_length : ℝ := 2500 -- meters
def crossing_time : ℝ := 50 -- seconds

-- Definition to calculate speed
def speed_of_train (length : ℝ) (time : ℝ) : ℝ := length / time

-- Theorem to be proved
theorem train_speed_is_50 : speed_of_train train_length crossing_time = 50 :=
by
  sorry

end train_speed_is_50_l346_346575


namespace isosceles_triangle_base_length_l346_346380

theorem isosceles_triangle_base_length {b : ℝ} (hb : b > 1/2) :
  let α := (2 * real.arccos (1 / (2 * b))) / 2 in 
  ∃ x : ℝ, x = real.sqrt (2 - 1 / b) :=
by
  sorry

end isosceles_triangle_base_length_l346_346380


namespace number_of_triangles_in_decagon_l346_346981

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346981


namespace term_containing_x_squared_coefficient_l346_346869

theorem term_containing_x_squared_coefficient (n : ℕ) (h : binomial_term_coefficient (n choose 2) (- √6)^2 = 36) :
  specific_term_containing_x_squared (x - √6)^n 2 = 36 * x^2 :=
by {
  sorry
}

end term_containing_x_squared_coefficient_l346_346869


namespace calculate_expression_eq_l346_346647

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)

theorem calculate_expression_eq :
  (x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y) →
  (x⁻³ - y⁻³) / (x⁻¹ - y⁻¹) = x⁻² + x⁻¹ * y⁻¹ + y⁻² := by
  intro h
  sorry

end calculate_expression_eq_l346_346647


namespace extreme_values_range_of_m_l346_346334

section ExtremeValues

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem extreme_values :
  (∀ x ∈ set.Icc (0 : ℝ) 2, f x ≥ -2) ∧
  (∀ x ∈ set.Icc (0 : ℝ) 2, f x ≤ 2) ∧
  (∃ x ∈ set.Icc (0 : ℝ) 2, f x = -2) ∧
  (∃ x ∈ set.Icc (0 : ℝ) 2, f x = 2) :=
by sorry

end ExtremeValues

section TangentLines

theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
  ∀ x : ℝ, ∃ b : ℝ, f x + b = m + (f (2 : ℝ) - m) * (x - 2)) ↔
  -6 < m ∧ m < 2 :=
by sorry

end TangentLines

end extreme_values_range_of_m_l346_346334


namespace smallest_four_digit_in_pascal_triangle_l346_346553

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346553


namespace smallest_four_digit_in_pascals_triangle_l346_346434

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, 1000 ≤ nat.choose n k ∧ nat.choose n k < 10000 ∧ 
  (∀ m l : ℕ, (nat.choose m l ≥ 1000 ∧ nat.choose m l < 10000) → nat.choose n k ≤ nat.choose m l) :=
begin
  -- This is a statement only, no proof is provided
  sorry
end

end smallest_four_digit_in_pascals_triangle_l346_346434


namespace percent_workday_in_meetings_l346_346361

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 3 * first_meeting_duration
def third_meeting_duration : ℕ := 2 * second_meeting_duration
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration
def workday_duration : ℕ := 10 * 60

theorem percent_workday_in_meetings : (total_meeting_time : ℚ) / workday_duration * 100 = 50 := by
  sorry

end percent_workday_in_meetings_l346_346361


namespace grill_run_time_l346_346924

-- Definitions of conditions
def coals_burned_per_minute : ℕ := 15
def minutes_per_coal_burned : ℕ := 20
def coals_per_bag : ℕ := 60
def bags_burned : ℕ := 3

-- Theorems to prove the question
theorem grill_run_time (coals_burned_per_minute: ℕ) (minutes_per_coal_burned: ℕ) (coals_per_bag: ℕ) (bags_burned: ℕ): (coals_burned_per_minute * (minutes_per_coal_burned * bags_burned * coals_per_bag / (coals_burned_per_minute * coals_per_bag))) / 60 = 4 := 
by 
  -- Lean statement skips detailed proof steps for conciseness
  sorry

end grill_run_time_l346_346924


namespace Tahir_contribution_l346_346859

theorem Tahir_contribution
  (headphone_cost : ℕ := 200)
  (kenji_yen : ℕ := 15000)
  (exchange_rate : ℕ := 100)
  (kenji_contribution : ℕ := kenji_yen / exchange_rate)
  (tahir_contribution : ℕ := headphone_cost - kenji_contribution) :
  tahir_contribution = 50 := 
  by sorry

end Tahir_contribution_l346_346859


namespace abs_neg_2023_l346_346862

def abs (x : ℤ) : ℤ := if x < 0 then -x else x

theorem abs_neg_2023 : abs (-2023) = 2023 :=
by
  sorry

end abs_neg_2023_l346_346862


namespace general_term_a_n_max_m_value_l346_346716

noncomputable def SeqSum (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i

theorem general_term_a_n (a : ℕ → ℝ) (Sn : ℕ → ℝ) (h1 : ∀ n, a (n + 1) ≥ a n) 
  (h2 : a 1 > 1) 
  (h3 : ∀ n, 10 * Sn n = (2 * a n + 1) * (a n + 2)) 
  (hSn : ∀ n, Sn n = SeqSum a n) :
  ∀ n, a n = (1 / 2) * (5 * n - 1) := 
by 
  sorry

theorem max_m_value (a b : ℕ → ℝ) (n m : ℕ) (Sn : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) ≥ a n) 
  (h2 : a 1 > 1) 
  (h3 : ∀ n, 10 * Sn n = (2 * a n + 1) * (a n + 2)) 
  (hSn : ∀ n, Sn n = SeqSum a n) 
  (hb : ∀ n, b n = a n - (n - 3) / 2)
  (ineq : ∀ n, (sqrt 5 * m / 31) ≤ (∏ k in finset.range n, 1 + 1 / b k) - 1 / sqrt (2 * n + 3)) :
  m = 8 := 
by 
  sorry

end general_term_a_n_max_m_value_l346_346716


namespace child_ticket_cost_l346_346632

-- Define the conditions
def adult_ticket_cost : ℕ := 11
def total_people : ℕ := 23
def total_revenue : ℕ := 246
def children_count : ℕ := 7
def adults_count := total_people - children_count

-- Define the target to prove that the child ticket cost is 10
theorem child_ticket_cost (child_ticket_cost : ℕ) :
  16 * adult_ticket_cost + 7 * child_ticket_cost = total_revenue → 
  child_ticket_cost = 10 := by
  -- The proof is omitted
  sorry

end child_ticket_cost_l346_346632


namespace divide_set_same_sum_l346_346054

theorem divide_set_same_sum :
  ∃ (A : Fin 117 → Set ℕ) (h₁ : ∀ i, A i ⊆ {1, 2, ..., 1989}) (h₂ : ∀ i j, i ≠ j → Disjoint (A i) (A j))
    (h₃ : ∀ i, (A i).card = 17) (h₄ : ∀ i, (A i).sum id = 528), True :=
by {
  let A : Fin 117 → Set ℕ := sorry,
  sorry
}

end divide_set_same_sum_l346_346054


namespace inequality_l346_346824

theorem inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 3) :
  1 / (4 - a^2) + 1 / (4 - b^2) + 1 / (4 - c^2) ≤ 9 / (a + b + c)^2 :=
by
  sorry

end inequality_l346_346824


namespace parts_of_second_liquid_l346_346351

theorem parts_of_second_liquid (x : ℝ) :
    (0.10 * 5 + 0.15 * x) / (5 + x) = 11.42857142857143 / 100 ↔ x = 2 :=
by
  sorry

end parts_of_second_liquid_l346_346351


namespace smallest_four_digit_in_pascal_triangle_l346_346552

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346552


namespace profit_percentage_with_discount_is_20_l346_346944

open_locale big_operators

theorem profit_percentage_with_discount_is_20 
(CP MP SP : ℚ)
(h_discount : SP = 0.96 * MP)
(h_no_discount_profit : ((MP - CP) / CP) * 100 = 25) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end profit_percentage_with_discount_is_20_l346_346944


namespace geometric_seq_and_sum_l346_346224

variable {a : ℕ → ℤ}
def seq_a : Prop := a 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n - 1

theorem geometric_seq_and_sum (seq_a) :
  (∀ n : ℕ, ∃ r : ℤ, ∃ b : ℤ, ∀ k : ℕ, k > 0 → a k - 1 = b * r^(k-1)) ∧
  (∀ n : ℕ, a n = 2^(n-1) + 1) ∧
  (∀ n : ℕ, ∑ i in finset.range n, i * (a (i + 1) - 1) = (n-1) * 2^n + 1) :=
by  apply lite:intro sorry. 

end geometric_seq_and_sum_l346_346224


namespace vector_solution_l346_346333

def a : ℝ × ℝ × ℝ := (2, 2, 1)
def b : ℝ × ℝ × ℝ := (3, 1, -2)
def v : ℝ × ℝ × ℝ := (5, 3, -1)

-- Cross product function for 3D vectors
def cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2 - u.2 * v.2.1, u.2 * v.1 - u.1 * v.2, u.1 * v.2.1 - u.2.1 * v.1)

theorem vector_solution :
  (cross v a = cross b a) ∧ (cross v b = cross a b) :=
sorry

end vector_solution_l346_346333


namespace smallest_four_digit_in_pascals_triangle_l346_346425

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346425


namespace smallest_four_digit_in_pascals_triangle_l346_346479

-- Define Pascal's triangle
def pascals_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| _, 0 => 1
| n+1, k+1 => pascals_triangle n k + pascals_triangle n (k+1)

-- State the theorem
theorem smallest_four_digit_in_pascals_triangle : ∃ (n k : ℕ), pascals_triangle n k = 1000 ∧ ∀ m, m < 1000 → ∀ r s, pascals_triangle r s ≠ m :=
sorry

end smallest_four_digit_in_pascals_triangle_l346_346479


namespace sum_a_b_equals_five_l346_346852

open Nat

theorem sum_a_b_equals_five (a b : ℕ) (h : a^2 + 2 = factorial b) : a + b = 5 :=
by
  have h₁ : b < 5 :=
    sorry
  have h₂ : b = 3 :=
    sorry
  have h₃ : a = 2 :=
    sorry
  rw [h₂, h₃]
  norm_num

end sum_a_b_equals_five_l346_346852


namespace n_gon_decomposition_l346_346209

theorem n_gon_decomposition (n : ℕ) (h : n ≥ 3) :
  ∃ (divide : ∀ (P : List (ℝ × ℝ)), P.length = n → 
     (∃ (black white : List (List (ℝ × ℝ))),
        (∀ t ∈ black, ∀ s ∈ white, disjoint t s) ∧
        (∀ t ∈ black, ¬any_edge_of_t_is_on_boundary t P) ∧
        (∀ t ∈ white, ¬any_edge_of_t_is_on_boundary t P))) :=
sorry

end n_gon_decomposition_l346_346209


namespace probability_of_hyperbola_l346_346701

def is_hyperbola (m n : ℤ) : Prop :=
(m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0)

noncomputable def probability_hyperbola : ℚ :=
let m_set := [-2, -1, 0, 1, 2, 3] in
let n_set := [-3, -2, -1, 0, 1, 2] in
let valid_pairs := [(m, n) | m ∈ m_set, n ∈ n_set, is_hyperbola m n] in
(valid_pairs.length : ℚ) / (m_set.length * n_set.length)

theorem probability_of_hyperbola :
  probability_hyperbola = 13 / 25 :=
sorry

end probability_of_hyperbola_l346_346701


namespace find_Xk_minus_Yk_l346_346371

section
variables (k X Y : ℕ)
variables (H1 : k > 8)
variables (H2 : k * X + Y + k * X = 2 * k + 1)

theorem find_Xk_minus_Yk (H1 : k > 8) (H2 : 2 * k * X + Y = 2 * k + 1) :
  X = k - 1 ∧ Y = 3 → X - Y = k - 4 :=
begin
  intro h,
  cases h with HX HY,
  rw [HX, HY],
  exact nat.sub_eq_of_eq_add (eq.symm (nat.add_sub_cancel k 4))
end

end find_Xk_minus_Yk_l346_346371


namespace list_price_proof_l346_346952

-- Define the list price of the item
noncomputable def list_price : ℝ := 33

-- Define the selling price and commission for Alice
def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_selling_price x

-- Define the selling price and commission for Charles
def charles_selling_price (x : ℝ) : ℝ := x - 18
def charles_commission (x : ℝ) : ℝ := 0.18 * charles_selling_price x

-- The main theorem: proving the list price given Alice and Charles receive the same commission
theorem list_price_proof (x : ℝ) (h : alice_commission x = charles_commission x) : x = list_price :=
by 
  sorry

end list_price_proof_l346_346952


namespace smallest_four_digit_number_in_pascals_triangle_l346_346456

theorem smallest_four_digit_number_in_pascals_triangle :
  ∃ (n : ℕ), 1000 = (n choose min n) ∧ (n > 0 ∧ (∀ m < n, (m choose min m) < 1000)) :=
begin
  sorry
end

end smallest_four_digit_number_in_pascals_triangle_l346_346456


namespace smallest_four_digit_in_pascal_l346_346518

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346518


namespace range_of_m_l346_346270

def A (x : ℝ) : Prop := x^2 - x - 6 > 0
def B (x m : ℝ) : Prop := (x - m) * (x - 2 * m) ≤ 0
def is_disjoint (A B : ℝ → Prop) : Prop := ∀ x, ¬ (A x ∧ B x)

theorem range_of_m (m : ℝ) : 
  is_disjoint (A) (B m) ↔ -1 ≤ m ∧ m ≤ 3 / 2 := by
  sorry

end range_of_m_l346_346270


namespace no_bijections_exist_l346_346188

theorem no_bijections_exist:
  ¬ ∃ (f g : ℕ+ → ℕ+) (h_f : function.bijective f) (h_g : function.bijective g), 
    ∀ (n : ℕ+), g n = (finset.range n).sum (λ i, f ⟨i + 1, nat.succ_pos i⟩) / n :=
sorry

end no_bijections_exist_l346_346188


namespace smallest_four_digit_in_pascals_triangle_l346_346535

theorem smallest_four_digit_in_pascals_triangle :
  ∃ n k : ℕ, (1000 ≤ (nat.choose n k)) ∧ (∀ m l : ℕ, 1000 ≤ nat.choose m l → nat.choose m l ≥ 1000) :=
begin
  use [1000, 1],
  split,
  { exact nat.choose_eq_one_right 1000 1 },
  { intros m l hml,
    linarith }
end

end smallest_four_digit_in_pascals_triangle_l346_346535


namespace curve_is_degenerate_circle_l346_346681

theorem curve_is_degenerate_circle (r θ : ℝ) (h : r = 1 / (1 - sin θ)) :
  (∃ p : ℝ × ℝ, ∀ (x y : ℝ), (x, y) = p → x = 0 ∧ y = 1) :=
sorry

end curve_is_degenerate_circle_l346_346681


namespace number_of_satisfying_integers_is_12_l346_346749

noncomputable def num_satisfying_integers : ℕ :=
  finset.card (finset.filter (λ n : ℤ, (n - 3) * (n + 2) * (n + 6) < 0) (finset.Icc (-11 : ℤ) 11))

theorem number_of_satisfying_integers_is_12 :
  num_satisfying_integers = 12 :=
sorry

end number_of_satisfying_integers_is_12_l346_346749


namespace rooted_set_is_integers_l346_346943

def isRooted (S : Set ℤ) : Prop :=
  ∀ (n : ℕ) (a : Fin (n + 1) → ℤ), -- a represents coefficients of the polynomial
    let p : ℤ → ℤ := λ x => ∑ i in Finset.range (n + 1), a ⟨i, Nat.lt_succ_self i⟩ * x ^ i
    ∀ root : ℤ, p root = 0 → root ∈ S

theorem rooted_set_is_integers :
  ∀ S : Set ℤ, (isRooted S) → (∀ a b : ℕ, (2^a - 2^b) ∈ S) → S = Set.univ :=
by
  intro S hRooted h2Diff
  sorry

end rooted_set_is_integers_l346_346943


namespace price_increase_x_annual_l346_346032

/-- 
Given:
1. The price of commodity x increases by some amount every year.
2. The price of commodity y increases by 20 cents every year.
3. In 2001, the price of commodity x was $4.20.
4. In 2001, the price of commodity y was $4.40.
5. In 2012, commodity x cost 90 cents more than commodity y.

Prove: 
The annual increase in the price of commodity x is 30 cents per year.
-/
theorem price_increase_x_annual :
  let annual_increase_x : ℕ := 30 in
  let start_price_x : ℕ := 420 in
  let start_price_y : ℕ := 440 in
  let annual_increase_y : ℕ := 20 in
  let years : ℕ := 11 in
  let price_diff_2012 : ℕ := 90 in
  let final_price_y := start_price_y + years * annual_increase_y in
  let final_price_x := start_price_x + years * annual_increase_x in
  final_price_x = final_price_y + price_diff_2012 :=
by 
  -- Proof combination of given conditions and question
  sorry

end price_increase_x_annual_l346_346032


namespace value_of_f_l346_346075

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x 
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b

theorem value_of_f'_at_1 (a b : ℝ)
  (h₁ : f a b 0 = 1)
  (h₂ : f' (a := a) (b := b) 0 = 0) :
  f' (a := a) (b := b) 1 = Real.exp 1 - 1 :=
by
  sorry

end value_of_f_l346_346075


namespace total_distance_traveled_l346_346111

theorem total_distance_traveled 
  (h₀ : ℝ := 20)
  (r : ℝ := 2 / 3) :
  let up₁ := h₀ * r
      down₁ := up₁
      up₂ := down₁ * r
      down₂ := up₂
      up₃ := down₂ * r
      down₃ := up₃ in
  h₀ + up₁ + down₁ + up₂ + down₂ + up₃ + down₃ = 76 := 
by
  let h := h₀
  let u1 := h * r
  let d1 := u1
  let u2 := d1 * r
  let d2 := u2
  let u3 := d2 * r
  let d3 := u3
  have : h + u1 + d1 + u2 + d2 + u3 + d3 = 76,
    by sorry
  exact this

end total_distance_traveled_l346_346111


namespace jill_has_1_more_peach_than_jake_l346_346803

theorem jill_has_1_more_peach_than_jake
    (jill_peaches : ℕ)
    (steven_peaches : ℕ)
    (jake_peaches : ℕ)
    (h1 : jake_peaches = steven_peaches - 16)
    (h2 : steven_peaches = jill_peaches + 15)
    (h3 : jill_peaches = 12) :
    12 - (steven_peaches - 16) = 1 := 
sorry

end jill_has_1_more_peach_than_jake_l346_346803


namespace area_decrease_of_equilateral_triangle_l346_346639

def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s ^ 2 * Real.sqrt 3) / 4

theorem area_decrease_of_equilateral_triangle :
  let A := 100 * Real.sqrt 3 in
  let s := Real.sqrt 400 in
  let s_new1 := s - 6 in
  let A_new1 := equilateral_triangle_area s_new1 in
  let s_new2 := s_new1 - 2 in
  let A_new2 := equilateral_triangle_area s_new2 in
  A_new2 = 36 * Real.sqrt 3 ∧ (A - A_new2) = 64 * Real.sqrt 3 :=
by {
  sorry
}

end area_decrease_of_equilateral_triangle_l346_346639


namespace original_price_of_computer_l346_346766

theorem original_price_of_computer
  (P : ℝ)
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 :=
by
  sorry

end original_price_of_computer_l346_346766


namespace total_time_is_12_years_l346_346130

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l346_346130


namespace find_value_of_f_l346_346215

theorem find_value_of_f : 
  ∀ (k : ℝ), (f : ℝ → ℝ), 
  (f = λ x, k * x + 2 / x^3 - 3) → f (Real.log 6) = 1 → f (Real.log (1 / 6)) = -7 :=
by 
  intro k f h₁ h₂
  sorry

end find_value_of_f_l346_346215


namespace original_difference_is_eight_l346_346917

-- Define the variables and conditions
variables (x y : ℝ) -- weights of the sugar in the first and second sacks respectively
variables (h1 : x + y = 40)  -- total weight condition
variables (h2 : x - 1 = 0.6 * (y + 1))  -- weight transformation condition

-- State the theorem to prove the original difference in weights is 8 kilograms
theorem original_difference_is_eight (h1 : x + y = 40) (h2 : x - 1 = 0.6 * (y + 1)) : |x - y| = 8 :=
begin
  sorry  -- proof to be completed
end

end original_difference_is_eight_l346_346917


namespace smallest_four_digit_in_pascals_triangle_l346_346468

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346468


namespace students_with_high_scores_l346_346595

theorem students_with_high_scores (μ σ : ℝ) (n : ℕ) 
  (h1 : μ = 110) 
  (h2 : σ = 10) 
  (h3 : n = 50) 
  (h4 : ∀ x : ℝ, x ∈ set.Icc 100 110 → 0.34 = ((measure_theory.measure_space.volume) (set.Ioc (x - 1 * 10) x)) ) :
  let ξ := measure_theory.measure_space.volume in
  let P := ξ {x : ℝ | x ≥ 120} in 
  let num_students := n * P.to_real in 
  approx_num_students = 8 :=
by
  sorry

end students_with_high_scores_l346_346595


namespace smallest_four_digit_in_pascal_triangle_l346_346559

noncomputable section

def is_in_pascal_triangle (n : ℕ) : Prop :=
  ∃ r c : ℕ, c ≤ r ∧ binomial r c = n

theorem smallest_four_digit_in_pascal_triangle :
  ¬ ∃ n : ℕ, is_in_pascal_triangle n ∧ n < 1000 ∧ (n ≥ 1000 ∨ n < 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_triangle_l346_346559


namespace find_a_l346_346877

noncomputable def f (x a : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem find_a : (∃ a : ℝ, ((∀ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a ≤ -3) ∧ (∃ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a = -3)) ↔ a = Real.sqrt 6 + 2) :=
by
  sorry

end find_a_l346_346877


namespace smallest_four_digit_in_pascal_l346_346498

-- define what it means for a number to be in Pascal's triangle
def in_pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- define the concept of a smallest four-digit number
def smallest_four_digit : ℕ := 1000

-- prove that the smallest four-digit number in Pascal's Triangle is 1000
theorem smallest_four_digit_in_pascal : 
  ∃ (n k : ℕ), in_pascal n k = smallest_four_digit := 
sorry

end smallest_four_digit_in_pascal_l346_346498


namespace smallest_four_digit_in_pascals_triangle_l346_346473

/-- Pascal's triangle is defined using binomial coefficients.
    We want to prove that the smallest four-digit number in Pascal's triangle is 1001. -/
theorem smallest_four_digit_in_pascals_triangle : ∃ n k : ℕ, n ≥ k ∧ (binomial n k = 1001 ∧ ∀ m l, m ≥ l → binomial m l ≥ 1000 → binomial m l = 1001 → n = m ∧ k = l) := by
  sorry

end smallest_four_digit_in_pascals_triangle_l346_346473


namespace smallest_four_digit_in_pascal_l346_346520

-- Pascal's triangle definition
def pascal (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

-- Condition definition
def exists_in_pascal (m : ℕ) : Prop :=
  ∃ n k, pascal n k = m

theorem smallest_four_digit_in_pascal :
  ∃ n k, pascal n k = 1000 ∧ ∀ m, (exists_in_pascal m) → (m >= 1000) → (m = 1000) :=
by
  sorry

end smallest_four_digit_in_pascal_l346_346520


namespace probability_area_less_than_circumference_of_dice_roll_l346_346050

theorem probability_area_less_than_circumference_of_dice_roll :
  let sum_dice := {n : ℕ | n = (d1 + d2 + d3 ∧ d1 ∈ {1, 2, 3, 4, 5, 6} ∧ d2 ∈ {1, 2, 3, 4, 5, 6} ∧ d3 ∈ {1, 2, 3, 4, 5, 6})} 
  ∃ d : ℕ, (3 ≤ d ∧ d ≤ 18) → (d^2 < 4 * d) → d ≠ 4 ∧ (d(d - 4) < 0) ∧ (d ∈ {3, 4, 5, 6, ..., 18}) →
  let event_sum_3 := {d : ℕ | d = 3} in
  ∑ (d = 3), P(event_sum_3) = 1 / 216 := sorry

end probability_area_less_than_circumference_of_dice_roll_l346_346050


namespace rayman_works_10_hours_l346_346310

noncomputable def rayman_hours (J : ℕ) : ℕ := 
  (1 / 2 : ℚ) * J

noncomputable def wolverine_hours (R J : ℕ) : ℕ := 
  2 * (R + J)

theorem rayman_works_10_hours (J R : ℕ) (h1 : R = (1 / 2 : ℚ) * J) (h2 : wolverine_hours R J = 60) : R = 10 := 
by
  sorry

end rayman_works_10_hours_l346_346310


namespace domain_f_l346_346874

open Set Real

def f (x : ℝ) : ℝ := sqrt x + sqrt (x * (x - 3))

theorem domain_f :
  {x : ℝ | 0 ≤ x} ∩ {x : ℝ | 0 ≤ x * (x - 3)} = {0} ∪ {x | 3 ≤ x} := sorry

end domain_f_l346_346874


namespace surface_generates_solid_l346_346626

/-
Given a right-angled triangle rotating around one of its right-angle sides, forming a cone,
prove that a surface generates a solid.
-/

def right_angled_triangle_rotates (triangle : Type) :=
  -- Definition for a right-angled triangle rotating to form a cone
  ∃ (rotate : triangle → Prop), 
    (∃ (side : triangle), rotate side)

theorem surface_generates_solid (triangle : Type) 
  (H : right_angled_triangle_rotates triangle) : 
  ∃ (solid : Type), (surface : Type) ∧ (surface -> solid) :=
sorry

end surface_generates_solid_l346_346626


namespace sign_up_combinations_l346_346045

theorem sign_up_combinations (students : ℕ) (groups : ℕ) (Hstudents : students = 5) (Hgroups : groups = 3) :
  (groups ^ students) = 243 :=
by
  rw [Hgroups, Hstudents]
  exact 3^5
  sorry
 
end sign_up_combinations_l346_346045


namespace sum_of_reciprocals_negative_l346_346402

theorem sum_of_reciprocals_negative {a b c : ℝ} (h₁ : a + b + c = 0) (h₂ : a * b * c > 0) :
  1/a + 1/b + 1/c < 0 :=
sorry

end sum_of_reciprocals_negative_l346_346402


namespace length_angle_bisector_l346_346797

/-- In triangle ABC, given AB = 6, BC = 8, AC = 10, and CD is the angle bisector of angle ACB,
compute the length of CD. -/
theorem length_angle_bisector 
  (A B C D : Type)
  [PlaneGeometry A B C]
  (AB BC AC : ℝ) (h : AB = 6 ∧ BC = 8 ∧ AC = 10)
  (angle_bisector : AngleBisector A C B D) : 
  length (segment C D) = (8 * Real.sqrt 10) / 3 :=
sorry

end length_angle_bisector_l346_346797


namespace circle_equation_l346_346024

theorem circle_equation :
  ∃ (h k r : ℝ), 
    (∀ (x y : ℝ), (x, y) = (-6, 2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ (∀ (x y : ℝ), (x, y) = (2, -2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ r = 5
    ∧ h - k = -1
    ∧ (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end circle_equation_l346_346024


namespace exists_sequence_polynomials_real_distinct_roots_l346_346970

theorem exists_sequence_polynomials_real_distinct_roots :
  ∃ (a : ℕ → ℝ), (∀ i, a i ≠ 0) ∧ 
  (∀ n, ∀ p : Polynomial ℝ, p = Polynomial.sum (Finset.range (n+1)) (λ k, Polynomial.C (a k) * (Polynomial.X ^ k)) → 
  (∀ x : ℝ, p.roots.count x ≤ 1) ∧ p.roots.length = n) :=
by
  sorry

end exists_sequence_polynomials_real_distinct_roots_l346_346970


namespace minimum_triangle_area_of_polygon_l346_346403

noncomputable def complex_roots (k : ℕ) : ℂ :=
  4 + 2 * complex.exp (complex.I * 2 * real.pi * k / 12)

theorem minimum_triangle_area_of_polygon :
  let D := complex_roots 0,
      E := complex_roots 4,
      F := complex_roots 8,
      area := (λ (a b c : ℂ), (complex.abs ((b - a) * (c - a) - (c - a) * (b - a)) / 2)) D E F
  in area = real.sqrt 3 := by
  sorry

end minimum_triangle_area_of_polygon_l346_346403


namespace inverse_function_passes_through_point_a_l346_346717

theorem inverse_function_passes_through_point_a
  (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ (∀ x, (a^(x-3) + 1) = 2 ↔ x = 3) → (2 - 1)/(3-3) = 0 :=
by
  sorry

end inverse_function_passes_through_point_a_l346_346717


namespace number_of_triangles_in_decagon_l346_346980

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l346_346980
