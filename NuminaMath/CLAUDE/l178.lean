import Mathlib

namespace NUMINAMATH_CALUDE_capacitance_calculation_l178_17883

theorem capacitance_calculation (U ε Q : ℝ) (hε : ε > 0) (hU : U ≠ 0) :
  ∃ C : ℝ, C > 0 ∧ C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_capacitance_calculation_l178_17883


namespace NUMINAMATH_CALUDE_class_size_l178_17856

theorem class_size : ∃ n : ℕ, 
  (20 < n ∧ n < 30) ∧ 
  (∃ x : ℕ, n = 3 * x) ∧ 
  (∃ y : ℕ, n = 4 * y - 1) ∧ 
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l178_17856


namespace NUMINAMATH_CALUDE_kim_average_increase_l178_17890

/-- Given Kim's exam scores, prove that her average increases by 1 after the fourth exam. -/
theorem kim_average_increase (score1 score2 score3 score4 : ℕ) 
  (h1 : score1 = 87)
  (h2 : score2 = 83)
  (h3 : score3 = 88)
  (h4 : score4 = 90) :
  (score1 + score2 + score3 + score4) / 4 - (score1 + score2 + score3) / 3 = 1 := by
  sorry

#eval (87 + 83 + 88 + 90) / 4 - (87 + 83 + 88) / 3

end NUMINAMATH_CALUDE_kim_average_increase_l178_17890


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l178_17884

theorem price_decrease_percentage (P : ℝ) (x : ℝ) (h₁ : P > 0) :
  (1.20 * P) * (1 - x / 100) = 0.75 * P → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l178_17884


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_l178_17840

theorem sqrt_two_times_sqrt_three : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_l178_17840


namespace NUMINAMATH_CALUDE_identity_proof_l178_17843

-- Define the necessary functions and series
def infiniteProduct (f : ℕ → ℝ) : ℝ := sorry

def infiniteSum (f : ℤ → ℝ) : ℝ := sorry

-- State the theorem
theorem identity_proof (x : ℝ) (h : |x| < 1) :
  -- First identity
  (infiniteProduct (λ m => (1 - x^(2*m - 1))^2)) =
  (1 / infiniteProduct (λ m => (1 - x^(2*m)))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2)))
  ∧
  -- Second identity
  (infiniteProduct (λ m => (1 - x^m))) =
  (infiniteProduct (λ m => (1 + x^m))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2))) :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_l178_17843


namespace NUMINAMATH_CALUDE_max_value_and_sum_l178_17816

theorem max_value_and_sum (x y z v w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_sq : x^2 + y^2 + z^2 + v^2 + w^2 = 2016) : 
  ∃ (N x_N y_N z_N v_N w_N : ℝ),
    (∀ (a b c d e : ℝ), 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 
      a^2 + b^2 + c^2 + d^2 + e^2 = 2016 → 
      4*a*c + 3*b*c + 2*c*d + 4*c*e ≤ N) ∧
    (4*x_N*z_N + 3*y_N*z_N + 2*z_N*v_N + 4*z_N*w_N = N) ∧
    (x_N^2 + y_N^2 + z_N^2 + v_N^2 + w_N^2 = 2016) ∧
    (N + x_N + y_N + z_N + v_N + w_N = 78 + 2028 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_max_value_and_sum_l178_17816


namespace NUMINAMATH_CALUDE_danes_daughters_flowers_l178_17815

def flowers_per_basket (people : ℕ) (flowers_per_person : ℕ) (additional_growth : ℕ) (died : ℕ) (baskets : ℕ) : ℕ :=
  ((people * flowers_per_person + additional_growth - died) / baskets)

theorem danes_daughters_flowers :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_danes_daughters_flowers_l178_17815


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l178_17881

/-- A four-digit palindrome between 1000 and 10000 -/
def FourDigitPalindrome : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, n = 1000 * a + 100 * b + 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 }

/-- The theorem stating that all four-digit palindromes are divisible by 11 -/
theorem all_four_digit_palindromes_divisible_by_11 (n : FourDigitPalindrome) : 11 ∣ n.val := by
  sorry

/-- The probability that a randomly chosen four-digit palindrome is divisible by 11 -/
theorem probability_palindrome_divisible_by_11 : ℚ :=
  1

/-- The main theorem proving that the probability is 1 -/
theorem main_theorem : probability_palindrome_divisible_by_11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l178_17881


namespace NUMINAMATH_CALUDE_kim_laura_difference_l178_17888

/-- Proves that Kim paints 3 fewer tiles per minute than Laura -/
theorem kim_laura_difference (don ken laura kim : ℕ) : 
  don = 3 →  -- Don paints 3 tiles per minute
  ken = don + 2 →  -- Ken paints 2 more tiles than Don per minute
  laura = 2 * ken →  -- Laura paints twice as many tiles as Ken per minute
  don + ken + laura + kim = 25 →  -- They paint 375 tiles in 15 minutes (375 / 15 = 25)
  laura - kim = 3 := by  -- Kim paints 3 fewer tiles than Laura per minute
sorry

end NUMINAMATH_CALUDE_kim_laura_difference_l178_17888


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_15_l178_17867

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the function g(n) as the sum of digits of 1/3^n to the right of the decimal point
def g (n : ℕ) : ℕ := sumOfDigits (10^n / 3^n)

-- Theorem statement
theorem smallest_n_exceeding_15 :
  (∀ k < 6, g k ≤ 15) ∧ g 6 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_15_l178_17867


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l178_17864

/-- The number of distinct terms in the expansion of (x+y+z+w)(p+q+r+s+t) -/
def distinctTerms (x y z w p q r s t : ℝ) : ℕ :=
  4 * 5

theorem expansion_distinct_terms (x y z w p q r s t : ℝ) 
  (h : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
       p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) :
  distinctTerms x y z w p q r s t = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l178_17864


namespace NUMINAMATH_CALUDE_gcd_1755_1242_l178_17875

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1755_1242_l178_17875


namespace NUMINAMATH_CALUDE_f_min_value_l178_17882

/-- The function f(x) = (x^2 + 33) / x for x ∈ ℕ* -/
def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

/-- The minimum value of f(x) is 23/2 -/
theorem f_min_value : ∀ x : ℕ+, f x ≥ 23/2 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l178_17882


namespace NUMINAMATH_CALUDE_find_number_l178_17833

theorem find_number : ∃! x : ℝ, (((48 - x) * 4 - 26) / 2) = 37 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l178_17833


namespace NUMINAMATH_CALUDE_union_M_N_complement_N_P_subset_M_iff_l178_17827

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 4) * (x - 6) < 0}
def N : Set ℝ := {x | x - 5 < 0}
def P (t : ℝ) : Set ℝ := {x | |x| = t}

-- Theorem 1: M ∪ N = {x | x < 6}
theorem union_M_N : M ∪ N = {x | x < 6} := by sorry

-- Theorem 2: N̄ₘ = {x | x ≥ 5}
theorem complement_N : (Nᶜ : Set ℝ) = {x | x ≥ 5} := by sorry

-- Theorem 3: P ⊆ M if and only if t ∈ (-∞, 4)
theorem P_subset_M_iff (t : ℝ) : P t ⊆ M ↔ t < 4 := by sorry

end NUMINAMATH_CALUDE_union_M_N_complement_N_P_subset_M_iff_l178_17827


namespace NUMINAMATH_CALUDE_mlb_game_ratio_l178_17863

theorem mlb_game_ratio (misses : ℕ) (total : ℕ) : 
  misses = 50 → total = 200 → (misses : ℚ) / (total - misses : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mlb_game_ratio_l178_17863


namespace NUMINAMATH_CALUDE_derivative_from_limit_l178_17886

theorem derivative_from_limit (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_from_limit_l178_17886


namespace NUMINAMATH_CALUDE_quotient_60_55_is_recurring_l178_17850

/-- Represents a recurring decimal with an integer part and a repeating fractional part -/
structure RecurringDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- The quotient of 60 divided by 55 as a recurring decimal -/
def quotient_60_55 : RecurringDecimal :=
  { integerPart := 1,
    repeatingPart := 9 }

/-- Theorem stating that 60 divided by 55 is equal to the recurring decimal 1.090909... -/
theorem quotient_60_55_is_recurring : (60 : ℚ) / 55 = 1 + (9 : ℚ) / 99 := by sorry

end NUMINAMATH_CALUDE_quotient_60_55_is_recurring_l178_17850


namespace NUMINAMATH_CALUDE_parallelogram_to_rhombus_l178_17836

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_convex (Q : Quadrilateral) : Prop := sorry

def is_parallelogram (Q : Quadrilateral) : Prop := sorry

def is_rhombus (Q : Quadrilateral) : Prop := sorry

def is_similar_not_congruent (Q1 Q2 : Quadrilateral) : Prop := sorry

def perpendicular_move (Q : Quadrilateral) : Quadrilateral := sorry

theorem parallelogram_to_rhombus (P : Quadrilateral) 
  (h_convex : is_convex P) 
  (h_initial : is_parallelogram P) 
  (h_final : ∃ (P_final : Quadrilateral), 
    (∃ (n : ℕ), n > 0 ∧ P_final = (perpendicular_move^[n] P)) ∧ 
    is_similar_not_congruent P P_final) :
  is_rhombus P := by sorry

end NUMINAMATH_CALUDE_parallelogram_to_rhombus_l178_17836


namespace NUMINAMATH_CALUDE_smallest_n_for_3003_combinations_l178_17803

theorem smallest_n_for_3003_combinations : ∃ (N : ℕ), N > 0 ∧ (
  (∀ k < N, Nat.choose k 5 < 3003) ∧
  Nat.choose N 5 = 3003
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_3003_combinations_l178_17803


namespace NUMINAMATH_CALUDE_polynomial_factorization_l178_17859

theorem polynomial_factorization (x : ℝ) :
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 1)*(x^4 - 2*x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l178_17859


namespace NUMINAMATH_CALUDE_steak_knife_set_cost_is_80_l178_17814

/-- Represents the cost of a steak knife set -/
def steak_knife_set_cost (knives_per_set : ℕ) (single_knife_cost : ℕ) : ℕ :=
  knives_per_set * single_knife_cost

/-- Proves that the cost of a steak knife set with 4 knives at $20 each is $80 -/
theorem steak_knife_set_cost_is_80 :
  steak_knife_set_cost 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_set_cost_is_80_l178_17814


namespace NUMINAMATH_CALUDE_average_tip_fraction_l178_17860

-- Define the weekly tip fractions
def week1_tip_fraction : ℚ := 2/4
def week2_tip_fraction : ℚ := 3/8
def week3_tip_fraction : ℚ := 5/16
def week4_tip_fraction : ℚ := 1/4

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Theorem statement
theorem average_tip_fraction :
  (week1_tip_fraction + week2_tip_fraction + week3_tip_fraction + week4_tip_fraction) / num_weeks = 23/64 := by
  sorry

end NUMINAMATH_CALUDE_average_tip_fraction_l178_17860


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l178_17878

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l178_17878


namespace NUMINAMATH_CALUDE_silver_coin_value_proof_l178_17870

/-- The value of a silver coin -/
def silver_coin_value : ℝ := 25

theorem silver_coin_value_proof :
  let gold_coin_value : ℝ := 50
  let num_gold_coins : ℕ := 3
  let num_silver_coins : ℕ := 5
  let cash : ℝ := 30
  let total_value : ℝ := 305
  silver_coin_value = (total_value - gold_coin_value * num_gold_coins - cash) / num_silver_coins :=
by
  sorry

end NUMINAMATH_CALUDE_silver_coin_value_proof_l178_17870


namespace NUMINAMATH_CALUDE_pythagorean_theorem_3d_l178_17804

/-- The Pythagorean theorem extended to a rectangular solid -/
theorem pythagorean_theorem_3d (p q r d : ℝ) 
  (h : d > 0) 
  (h_diagonal : d = Real.sqrt (p^2 + q^2 + r^2)) : 
  p^2 + q^2 + r^2 = d^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_3d_l178_17804


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l178_17807

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l178_17807


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l178_17895

/-- Given a triangle ABC with side AC of length 2 and satisfying the equation
    √3 tan A tan C = tan A + tan C + √3, its perimeter is in (4, 2 + 2√3) ∪ (2 + 2√3, 6] -/
theorem triangle_perimeter_range (A C : Real) (hAC : Real) :
  hAC = 2 →
  Real.sqrt 3 * Real.tan A * Real.tan C = Real.tan A + Real.tan C + Real.sqrt 3 →
  ∃ (p : Real), p ∈ Set.union (Set.Ioo 4 (2 + 2 * Real.sqrt 3)) (Set.Ioc (2 + 2 * Real.sqrt 3) 6) ∧
                p = hAC + 2 * Real.sin (A + π / 6) + 2 * Real.sin (C + π / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l178_17895


namespace NUMINAMATH_CALUDE_hex_tile_difference_l178_17841

/-- Represents the number of tiles in a hexagonal arrangement --/
structure HexTileArrangement where
  blue : ℕ
  green : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal arrangement --/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Adds a border of green tiles to an existing arrangement --/
def add_border (arrangement : HexTileArrangement) (border_size : ℕ) : HexTileArrangement :=
  { blue := arrangement.blue,
    green := arrangement.green + border_tiles border_size }

/-- The main theorem to prove --/
theorem hex_tile_difference :
  let initial := HexTileArrangement.mk 12 8
  let first_border := add_border initial 3
  let second_border := add_border first_border 4
  second_border.green - second_border.blue = 38 := by sorry

end NUMINAMATH_CALUDE_hex_tile_difference_l178_17841


namespace NUMINAMATH_CALUDE_calculation_proof_l178_17873

theorem calculation_proof : (3^2 * (-2 + 3) / (1/3) - |-28|) = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l178_17873


namespace NUMINAMATH_CALUDE_distance_between_first_two_points_l178_17802

theorem distance_between_first_two_points
  (n : ℕ)
  (sum_first : ℝ)
  (sum_second : ℝ)
  (h_n : n = 11)
  (h_sum_first : sum_first = 2018)
  (h_sum_second : sum_second = 2000) :
  ∃ (x : ℝ),
    x = 2 ∧
    x * (n - 2) = sum_first - sum_second :=
by sorry

end NUMINAMATH_CALUDE_distance_between_first_two_points_l178_17802


namespace NUMINAMATH_CALUDE_corn_kernel_weight_theorem_l178_17872

/-- Calculates the total weight of corn kernels after shucking and accounting for losses -/
def corn_kernel_weight (
  ears_per_stalk : ℕ)
  (total_stalks : ℕ)
  (bad_ear_percentage : ℚ)
  (kernel_distribution : List (ℚ × ℕ))
  (kernel_weight : ℚ)
  (lost_kernel_percentage : ℚ) : ℚ :=
  let total_ears := ears_per_stalk * total_stalks
  let good_ears := total_ears - (bad_ear_percentage * total_ears).floor
  let total_kernels := (kernel_distribution.map (fun (p, k) => 
    ((p * good_ears).floor * k))).sum
  let kernels_after_loss := total_kernels - 
    (lost_kernel_percentage * total_kernels).floor
  kernels_after_loss * kernel_weight

/-- The total weight of corn kernels is approximately 18527.9 grams -/
theorem corn_kernel_weight_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.1 ∧ 
  |corn_kernel_weight 4 108 (1/5) 
    [(3/5, 500), (3/10, 600), (1/10, 700)] (1/10) (3/200) - 18527.9| < ε :=
sorry

end NUMINAMATH_CALUDE_corn_kernel_weight_theorem_l178_17872


namespace NUMINAMATH_CALUDE_ohara_triple_x_value_l178_17885

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℝ) : Prop :=
  Real.sqrt (abs a) + Real.sqrt (abs b) = x

/-- Theorem: If (-49, 64, x) is an O'Hara triple, then x = 15 -/
theorem ohara_triple_x_value :
  ∀ x : ℝ, is_ohara_triple (-49) 64 x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_x_value_l178_17885


namespace NUMINAMATH_CALUDE_bus_problem_l178_17824

/-- Calculates the number of students remaining on a bus after a given number of stops,
    where one-third of the students get off at each stop. -/
def studentsRemaining (initialStudents : ℚ) (stops : ℕ) : ℚ :=
  initialStudents * (2/3)^stops

/-- Proves that if a bus starts with 60 students and loses one-third of its passengers
    at each of four stops, the number of students remaining after the fourth stop is 320/27. -/
theorem bus_problem : studentsRemaining 60 4 = 320/27 := by
  sorry

#eval studentsRemaining 60 4

end NUMINAMATH_CALUDE_bus_problem_l178_17824


namespace NUMINAMATH_CALUDE_max_coprime_partition_l178_17828

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def valid_partition (A B : Finset ℕ) : Prop :=
  (∀ a ∈ A, 2 ≤ a ∧ a ≤ 20) ∧
  (∀ b ∈ B, 2 ≤ b ∧ b ≤ 20) ∧
  (∀ a ∈ A, ∀ b ∈ B, is_coprime a b) ∧
  A ∩ B = ∅ ∧
  A ∪ B ⊆ Finset.range 19 ∪ {20}

theorem max_coprime_partition :
  ∃ A B : Finset ℕ,
    valid_partition A B ∧
    A.card * B.card = 49 ∧
    ∀ C D : Finset ℕ, valid_partition C D → C.card * D.card ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_max_coprime_partition_l178_17828


namespace NUMINAMATH_CALUDE_subtraction_as_addition_of_negative_l178_17820

theorem subtraction_as_addition_of_negative (a b : ℚ) : a - b = a + (-b) := by sorry

end NUMINAMATH_CALUDE_subtraction_as_addition_of_negative_l178_17820


namespace NUMINAMATH_CALUDE_tommy_pencil_case_items_l178_17812

/-- The number of items in Tommy's pencil case -/
theorem tommy_pencil_case_items (pencils : ℕ) (pens : ℕ) (eraser : ℕ) 
    (h1 : pens = 2 * pencils) 
    (h2 : eraser = 1)
    (h3 : pencils = 4) : 
  pencils + pens + eraser = 13 := by
  sorry

end NUMINAMATH_CALUDE_tommy_pencil_case_items_l178_17812


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l178_17853

def sandwich_price : ℚ := 4
def soda_price : ℚ := 3
def tax_rate : ℚ := 0.1
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 6

theorem total_cost_with_tax :
  let subtotal := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  let tax := subtotal * tax_rate
  let total := subtotal + tax
  total = 50.6 := by sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l178_17853


namespace NUMINAMATH_CALUDE_eight_possible_values_for_d_l178_17844

def is_digit (n : ℕ) : Prop := n < 10

def distinct_digits (a b c d e : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def valid_subtraction (a b c d e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * e + a) =
  10000 * d + 1000 * b + 100 * d + 10 * d + d

theorem eight_possible_values_for_d :
  ∃ (s : Finset ℕ), s.card = 8 ∧
  (∀ d, d ∈ s ↔ ∃ (a b c e : ℕ), distinct_digits a b c d e ∧ valid_subtraction a b c d e) :=
sorry

end NUMINAMATH_CALUDE_eight_possible_values_for_d_l178_17844


namespace NUMINAMATH_CALUDE_commodity_price_increase_l178_17848

/-- The annual price increase of commodity X in cents -/
def annual_increase_X : ℝ := 30

/-- The annual price increase of commodity Y in cents -/
def annual_increase_Y : ℝ := 20

/-- The price of commodity X in 2001 in dollars -/
def price_X_2001 : ℝ := 4.20

/-- The price of commodity Y in 2001 in dollars -/
def price_Y_2001 : ℝ := 4.40

/-- The number of years between 2001 and 2010 -/
def years : ℕ := 9

/-- The difference in price between X and Y in 2010 in cents -/
def price_difference_2010 : ℝ := 70

theorem commodity_price_increase :
  annual_increase_X = 30 ∧
  price_X_2001 + (annual_increase_X / 100 * years) =
  price_Y_2001 + (annual_increase_Y / 100 * years) + price_difference_2010 / 100 :=
by sorry

end NUMINAMATH_CALUDE_commodity_price_increase_l178_17848


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l178_17805

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (p q : Plane) : Prop := sorry

/-- A line is distinct from another line -/
def distinct_lines (a b : Line) : Prop := sorry

/-- A plane is distinct from another plane -/
def distinct_planes (p q : Plane) : Prop := sorry

theorem parallel_planes_from_perpendicular_lines 
  (a b : Line) (α β : Plane) 
  (h1 : distinct_lines a b) 
  (h2 : distinct_planes α β) 
  (h3 : perpendicular_line_plane a α) 
  (h4 : perpendicular_line_plane b β) 
  (h5 : parallel_lines a b) : 
  parallel_planes α β := 
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l178_17805


namespace NUMINAMATH_CALUDE_monitor_pixels_l178_17810

/-- Calculates the total number of pixels on a monitor given its dimensions and resolution. -/
def totalPixels (width : ℕ) (height : ℕ) (dotsPerInch : ℕ) : ℕ :=
  (width * dotsPerInch) * (height * dotsPerInch)

/-- Theorem stating that a 21x12 inch monitor with 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixels :
  totalPixels 21 12 100 = 2520000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixels_l178_17810


namespace NUMINAMATH_CALUDE_son_is_eighteen_l178_17854

theorem son_is_eighteen (father_age son_age : ℕ) : 
  father_age + son_age = 55 →
  ∃ (y : ℕ), father_age + y + (son_age + y) = 93 ∧ son_age + y = father_age →
  (father_age = 18 ∨ son_age = 18) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_is_eighteen_l178_17854


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l178_17897

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 7 + a 11 = 44 →
  a 3 + a 5 + a 10 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l178_17897


namespace NUMINAMATH_CALUDE_no_real_solutions_l178_17817

theorem no_real_solutions : ∀ x : ℝ, (2*x - 10*x + 24)^2 + 4 ≠ -2*|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l178_17817


namespace NUMINAMATH_CALUDE_quadratic_sum_l178_17887

theorem quadratic_sum (x : ℝ) : ∃ (a h k : ℝ),
  (3 * x^2 - 6 * x - 2 = a * (x - h)^2 + k) ∧ (a + h + k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l178_17887


namespace NUMINAMATH_CALUDE_slope_angle_range_l178_17855

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + (3 - Real.sqrt 3)*x + 3/4

def is_on_curve (p : ℝ × ℝ) : Prop := p.2 = f p.1

theorem slope_angle_range (p q : ℝ × ℝ) (hp : is_on_curve p) (hq : is_on_curve q) :
  let α := Real.arctan ((q.2 - p.2) / (q.1 - p.1))
  α ∈ Set.union (Set.Ico 0 (Real.pi / 2)) (Set.Icc (2 * Real.pi / 3) Real.pi) :=
sorry

end NUMINAMATH_CALUDE_slope_angle_range_l178_17855


namespace NUMINAMATH_CALUDE_restaurant_check_amount_l178_17892

theorem restaurant_check_amount
  (tax_rate : Real)
  (total_payment : Real)
  (tip_amount : Real)
  (h1 : tax_rate = 0.20)
  (h2 : total_payment = 20)
  (h3 : tip_amount = 2) :
  ∃ (original_amount : Real),
    original_amount * (1 + tax_rate) = total_payment - tip_amount ∧
    original_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_check_amount_l178_17892


namespace NUMINAMATH_CALUDE_probability_yellow_marble_l178_17826

theorem probability_yellow_marble (blue red yellow : ℕ) 
  (h_blue : blue = 7)
  (h_red : red = 11)
  (h_yellow : yellow = 6) :
  (yellow : ℚ) / (blue + red + yellow) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_marble_l178_17826


namespace NUMINAMATH_CALUDE_parabola_c_value_l178_17845

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (2, 3)

theorem parabola_c_value (p : Parabola) :
  p.vertex = (2, 3) →
  p.x_coord 2 = 0 →
  p.c = -16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l178_17845


namespace NUMINAMATH_CALUDE_smallest_c_value_l178_17825

theorem smallest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 7 * c + 6) :
  c ≥ (9 - Real.sqrt 249) / 6 ∧ ∃ (c₀ : ℝ), (3 * c₀ + 4) * (c₀ - 2) = 7 * c₀ + 6 ∧ c₀ = (9 - Real.sqrt 249) / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l178_17825


namespace NUMINAMATH_CALUDE_diagonal_intersection_minimizes_sum_distances_l178_17819

-- Define a quadrilateral in 2D space
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p q : Point) : ℝ := sorry

-- Function to find the intersection of two line segments
def intersectionPoint (p1 p2 q1 q2 : Point) : Point := sorry

-- Function to calculate the sum of distances from a point to all vertices of a quadrilateral
def sumDistances (quad : Quadrilateral) (p : Point) : ℝ := sorry

-- Theorem stating that the intersection of diagonals minimizes the sum of distances
theorem diagonal_intersection_minimizes_sum_distances (quad : Quadrilateral) :
  let M := intersectionPoint quad.A quad.C quad.B quad.D
  ∀ p : Point, sumDistances quad M ≤ sumDistances quad p :=
sorry

end NUMINAMATH_CALUDE_diagonal_intersection_minimizes_sum_distances_l178_17819


namespace NUMINAMATH_CALUDE_arctan_sum_equals_arctan_29_22_l178_17801

theorem arctan_sum_equals_arctan_29_22 (a b : ℝ) : 
  a = 3/4 → (a + 1) * (b + 1) = 9/4 → Real.arctan a + Real.arctan b = Real.arctan (29/22) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_arctan_29_22_l178_17801


namespace NUMINAMATH_CALUDE_largest_class_has_61_students_l178_17831

/-- Represents a school with a given number of classes and students. -/
structure School where
  num_classes : ℕ
  total_students : ℕ
  class_diff : ℕ

/-- Calculates the number of students in the largest class of a school. -/
def largest_class_size (s : School) : ℕ :=
  (s.total_students + s.class_diff * (s.num_classes - 1) * s.num_classes / 2) / s.num_classes

/-- Theorem stating that for a school with 8 classes, 380 total students,
    and 4 students difference between classes, the largest class has 61 students. -/
theorem largest_class_has_61_students :
  let s : School := { num_classes := 8, total_students := 380, class_diff := 4 }
  largest_class_size s = 61 := by
  sorry

#eval largest_class_size { num_classes := 8, total_students := 380, class_diff := 4 }

end NUMINAMATH_CALUDE_largest_class_has_61_students_l178_17831


namespace NUMINAMATH_CALUDE_johns_patients_l178_17830

/-- The number of patients John sees each day at the first hospital -/
def patients_first_hospital : ℕ := sorry

/-- The number of patients John sees each day at the second hospital -/
def patients_second_hospital : ℕ := sorry

/-- The number of days John works per year -/
def work_days_per_year : ℕ := 5 * 50

/-- The total number of patients John treats in a year -/
def total_patients_per_year : ℕ := 11000

theorem johns_patients :
  patients_first_hospital = 20 ∧
  patients_second_hospital = (6 * patients_first_hospital) / 5 ∧
  work_days_per_year * (patients_first_hospital + patients_second_hospital) = total_patients_per_year :=
sorry

end NUMINAMATH_CALUDE_johns_patients_l178_17830


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_l178_17818

def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![1, 2]

theorem perpendicular_vectors_k (k : ℝ) :
  (∀ i : Fin 2, (k * a i - b i) * (b i + a i) = 0) →
  k = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_l178_17818


namespace NUMINAMATH_CALUDE_factorial_ratio_l178_17896

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio (n : ℕ) (h : n > 0) : 
  factorial n / factorial (n - 1) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l178_17896


namespace NUMINAMATH_CALUDE_product_inequality_l178_17852

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a*x₁ + b) * (a*x₂ + b) * (a*x₃ + b) * (a*x₄ + b) * (a*x₅ + b) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l178_17852


namespace NUMINAMATH_CALUDE_det_B_equals_five_l178_17858

theorem det_B_equals_five (b c : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![b, 3; -1, c]
  B + 3 * B⁻¹ = 0 → Matrix.det B = 5 := by
sorry

end NUMINAMATH_CALUDE_det_B_equals_five_l178_17858


namespace NUMINAMATH_CALUDE_different_types_of_players_l178_17811

/-- Represents the types of players in the game. -/
inductive PlayerType
  | Cricket
  | Hockey
  | Football
  | Softball

/-- The number of players for each type. -/
def num_players (t : PlayerType) : ℕ :=
  match t with
  | .Cricket => 12
  | .Hockey => 17
  | .Football => 11
  | .Softball => 10

/-- The total number of players on the ground. -/
def total_players : ℕ := 50

/-- The list of all player types. -/
def all_player_types : List PlayerType :=
  [PlayerType.Cricket, PlayerType.Hockey, PlayerType.Football, PlayerType.Softball]

theorem different_types_of_players :
  (List.length all_player_types = 4) ∧
  (List.sum (List.map num_players all_player_types) = total_players) := by
  sorry

end NUMINAMATH_CALUDE_different_types_of_players_l178_17811


namespace NUMINAMATH_CALUDE_man_downstream_speed_l178_17880

/-- Calculates the downstream speed of a person given their upstream speed and the stream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem: Given a man's upstream speed of 8 km/h and a stream speed of 1 km/h, his downstream speed is 10 km/h. -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 1
  downstream_speed upstream_speed stream_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l178_17880


namespace NUMINAMATH_CALUDE_items_deleted_l178_17893

theorem items_deleted (initial : ℕ) (remaining : ℕ) (deleted : ℕ) : 
  initial = 100 → remaining = 20 → deleted = initial - remaining → deleted = 80 :=
by sorry

end NUMINAMATH_CALUDE_items_deleted_l178_17893


namespace NUMINAMATH_CALUDE_exactly_one_black_ball_remains_l178_17837

/-- Represents the color of a ball -/
inductive Color
| Black
| Gray
| White

/-- Represents the state of the box -/
structure BoxState :=
  (black : Nat)
  (gray : Nat)
  (white : Nat)

/-- Simulates drawing two balls from the box -/
def drawTwoBalls (state : BoxState) : BoxState :=
  sorry

/-- Checks if the given state has exactly two balls remaining -/
def hasTwoballsRemaining (state : BoxState) : Bool :=
  state.black + state.gray + state.white = 2

/-- Represents the final state of the box after the procedure -/
def finalState (initialState : BoxState) : BoxState :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_black_ball_remains :
  let initialState : BoxState := ⟨105, 89, 5⟩
  let finalState := finalState initialState
  hasTwoballsRemaining finalState ∧ finalState.black = 1 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_black_ball_remains_l178_17837


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l178_17857

theorem infinite_solutions_condition (c : ℝ) : 
  (∀ y : ℝ, y ≠ 0 → 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l178_17857


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l178_17849

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l178_17849


namespace NUMINAMATH_CALUDE_physics_class_size_l178_17822

theorem physics_class_size :
  ∀ (boys_biology girls_biology students_physics : ℕ),
    girls_biology = 3 * boys_biology →
    boys_biology = 25 →
    students_physics = 2 * (boys_biology + girls_biology) →
    students_physics = 200 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l178_17822


namespace NUMINAMATH_CALUDE_work_completion_time_l178_17808

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

theorem work_completion_time : 
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l178_17808


namespace NUMINAMATH_CALUDE_drape_cost_calculation_l178_17839

/-- The cost of window treatments for a house with the given conditions. -/
def window_treatment_cost (num_windows : ℕ) (sheer_cost drape_cost total_cost : ℚ) : Prop :=
  num_windows * (sheer_cost + drape_cost) = total_cost

/-- The theorem stating the cost of a pair of drapes given the conditions. -/
theorem drape_cost_calculation :
  ∃ (drape_cost : ℚ),
    window_treatment_cost 3 40 drape_cost 300 ∧ drape_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_drape_cost_calculation_l178_17839


namespace NUMINAMATH_CALUDE_school_picnic_attendees_l178_17861

/-- The number of attendees at the school picnic. -/
def num_attendees : ℕ := 1006

/-- The total number of plates prepared by the school. -/
def total_plates : ℕ := 2015 - num_attendees

theorem school_picnic_attendees :
  (∀ n : ℕ, n ≤ num_attendees → total_plates - (n - 1) > 0) ∧
  (total_plates - (num_attendees - 1) = 4) ∧
  (num_attendees + total_plates = 2015) :=
sorry

end NUMINAMATH_CALUDE_school_picnic_attendees_l178_17861


namespace NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l178_17894

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Theorem stating that q is sufficient but not necessary for p
theorem q_sufficient_not_necessary_for_p :
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l178_17894


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l178_17842

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n^3 - 200| ≥ |6^3 - 200| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_200_l178_17842


namespace NUMINAMATH_CALUDE_hyperbola_equation_l178_17834

/-- A hyperbola and a parabola sharing a common focus -/
structure HyperbolaParabola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  F : ℝ × ℝ
  P : ℝ × ℝ
  h_parabola : (P.2)^2 = 8 * P.1
  h_hyperbola : (P.1)^2 / a^2 - (P.2)^2 / b^2 = 1
  h_common_focus : F = (2, 0)
  h_distance : Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5

/-- The equation of the hyperbola is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation (hp : HyperbolaParabola) : 
  hp.a = 1 ∧ hp.b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l178_17834


namespace NUMINAMATH_CALUDE_tan_45_degrees_l178_17866

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l178_17866


namespace NUMINAMATH_CALUDE_kittens_at_shelter_l178_17874

theorem kittens_at_shelter (puppies : ℕ) (kittens : ℕ) : 
  puppies = 32 → 
  kittens = 2 * puppies + 14 → 
  kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_at_shelter_l178_17874


namespace NUMINAMATH_CALUDE_equation_solution_l178_17832

theorem equation_solution : ∃! y : ℚ, 
  (y ≠ 3 ∧ y ≠ 5/4) ∧ 
  (y^2 - 7*y + 12)/(y - 3) + (4*y^2 + 20*y - 25)/(4*y - 5) = 2 ∧
  y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l178_17832


namespace NUMINAMATH_CALUDE_favorite_sports_survey_l178_17871

/-- Given a survey of students' favorite sports, prove the number of students who like chess or basketball. -/
theorem favorite_sports_survey (total_students : ℕ) 
  (basketball_percent chess_percent soccer_percent badminton_percent : ℚ) :
  basketball_percent = 40/100 →
  chess_percent = 10/100 →
  soccer_percent = 28/100 →
  badminton_percent = 22/100 →
  basketball_percent + chess_percent + soccer_percent + badminton_percent = 1 →
  total_students = 250 →
  ⌊(basketball_percent + chess_percent) * total_students⌋ = 125 := by
  sorry


end NUMINAMATH_CALUDE_favorite_sports_survey_l178_17871


namespace NUMINAMATH_CALUDE_art_club_artworks_art_club_two_years_collection_l178_17877

theorem art_club_artworks (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) 
  (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * num_years

theorem art_club_two_years_collection : 
  art_club_artworks 15 2 4 2 = 240 := by sorry

end NUMINAMATH_CALUDE_art_club_artworks_art_club_two_years_collection_l178_17877


namespace NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l178_17838

/-- A point on the contour of a square -/
structure ContourPoint where
  x : ℝ
  y : ℝ

/-- Color of a point -/
inductive Color
  | Blue
  | Red

/-- A coloring of the contour of a square -/
def Coloring := ContourPoint → Color

/-- Predicate to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : ContourPoint) : Prop :=
  sorry

/-- Theorem: For any coloring of the contour of a square, there exists a right triangle
    with vertices of the same color -/
theorem monochromatic_right_triangle_exists (coloring : Coloring) :
  ∃ (p1 p2 p3 : ContourPoint),
    is_right_triangle p1 p2 p3 ∧
    coloring p1 = coloring p2 ∧
    coloring p2 = coloring p3 :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l178_17838


namespace NUMINAMATH_CALUDE_decimal_expression_simplification_l178_17865

theorem decimal_expression_simplification :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) := by
  sorry

end NUMINAMATH_CALUDE_decimal_expression_simplification_l178_17865


namespace NUMINAMATH_CALUDE_parallel_lines_l178_17846

/-- Two lines in the form ax + by + c = 0 are parallel if and only if they have the same a and b coefficients. -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2

/-- The line x + 2y + 2 = 0 is parallel to the line x + 2y + 1 = 0. -/
theorem parallel_lines : are_parallel 1 2 2 1 2 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l178_17846


namespace NUMINAMATH_CALUDE_monotone_f_range_l178_17889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

theorem monotone_f_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_monotone_f_range_l178_17889


namespace NUMINAMATH_CALUDE_sqrt_factorial_over_88_l178_17879

theorem sqrt_factorial_over_88 : 
  let factorial_10 : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let n : ℚ := factorial_10 / 88
  Real.sqrt n = (180 * Real.sqrt 7) / Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_over_88_l178_17879


namespace NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_one_l178_17851

theorem sufficient_condition_for_product_greater_than_one :
  ∀ (a b : ℝ), a > 1 ∧ b > 1 → a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_product_greater_than_one_l178_17851


namespace NUMINAMATH_CALUDE_fraction_above_line_is_five_sixths_l178_17823

/-- A square in the coordinate plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the coordinate plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The fraction of the square's area above a given line -/
def fractionAboveLine (s : Square) (l : Line) : ℝ := sorry

/-- The specific square from the problem -/
def problemSquare : Square :=
  { bottomLeft := (2, 1),
    topRight := (5, 4) }

/-- The specific line from the problem -/
def problemLine : Line :=
  { point1 := (2, 3),
    point2 := (5, 1) }

theorem fraction_above_line_is_five_sixths :
  fractionAboveLine problemSquare problemLine = 5/6 := by sorry

end NUMINAMATH_CALUDE_fraction_above_line_is_five_sixths_l178_17823


namespace NUMINAMATH_CALUDE_largest_circle_area_l178_17806

theorem largest_circle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Right triangle condition
  (h5 : π * (a/2)^2 + π * (b/2)^2 + π * (c/2)^2 = 338 * π) : -- Sum of circle areas
  π * (c/2)^2 = 169 * π := by
sorry

end NUMINAMATH_CALUDE_largest_circle_area_l178_17806


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l178_17809

theorem similar_triangles_shortest_side 
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (h1 : a^2 + b^2 = c^2)  -- first triangle is right-angled
  (h2 : d^2 + e^2 = f^2)  -- second triangle is right-angled
  (h3 : b = 15)  -- given side of first triangle
  (h4 : c = 17)  -- hypotenuse of first triangle
  (h5 : f = 51)  -- hypotenuse of second triangle
  (h6 : (a / d) = (b / e) ∧ (b / e) = (c / f))  -- triangles are similar
  : min d e = 24 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l178_17809


namespace NUMINAMATH_CALUDE_divisor_properties_l178_17862

def N (a b c : ℕ) (α β γ : ℕ) : ℕ := a^α * b^β * c^γ

variable (a b c α β γ : ℕ)
variable (ha : Nat.Prime a)
variable (hb : Nat.Prime b)
variable (hc : Nat.Prime c)

theorem divisor_properties :
  let n := N a b c α β γ
  -- Total number of divisors
  ∃ d : ℕ → ℕ, d n = (α + 1) * (β + 1) * (γ + 1) ∧
  -- Product of equidistant divisors
  ∀ x y : ℕ, x ∣ n → y ∣ n → x * y = n →
    ∃ z : ℕ, z ∣ n ∧ z * z = n ∧
  -- Product of all divisors
  ∃ P : ℕ, P = n ^ ((α + 1) * (β + 1) * (γ + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_divisor_properties_l178_17862


namespace NUMINAMATH_CALUDE_decimal_product_sum_l178_17813

-- Define the structure for our decimal representation
structure DecimalPair :=
  (whole : Nat)
  (decimal : Nat)

-- Define the multiplication operation for DecimalPair
def multiply_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) * (y.whole + y.decimal / 10 : Rat)

-- Define the addition operation for DecimalPair
def add_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) + (y.whole + y.decimal / 10 : Rat)

-- The main theorem
theorem decimal_product_sum (a b c d : Nat) :
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) → (d ≠ 0) →
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  multiply_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (56 : Rat) / 10 →
  add_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (51 : Rat) / 10 := by
sorry

end NUMINAMATH_CALUDE_decimal_product_sum_l178_17813


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l178_17891

-- Define the cycle length of the last two digits of powers of 3
def cycleLengthPowersOf3 : ℕ := 20

-- Define the function that gives the last two digits of 3^n
def lastTwoDigits (n : ℕ) : ℕ := 3^n % 100

-- Define the function that gives the tens digit of a number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_3_to_2023 :
  tensDigit (lastTwoDigits 2023) = 2 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2023_l178_17891


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l178_17835

-- Define a regular decagon inscribed in a circle
def RegularDecagon : Type := Unit

-- Define a segment in the decagon
def Segment (d : RegularDecagon) : Type := Unit

-- Define a function to check if three segments form a triangle with positive area
def formsTriangle (d : RegularDecagon) (s1 s2 s3 : Segment d) : Prop := sorry

-- Define a function to calculate the probability
def probabilityOfTriangle (d : RegularDecagon) : ℚ := sorry

-- Theorem statement
theorem decagon_triangle_probability (d : RegularDecagon) : 
  probabilityOfTriangle d = 153 / 190 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l178_17835


namespace NUMINAMATH_CALUDE_circle_center_sum_l178_17829

/-- Given a circle defined by the equation x^2 + y^2 = 6x - 8y + 9,
    prove that the sum of its center coordinates is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x - 8*y + 9 → 
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x + 8*y - 9) ∧ h + k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l178_17829


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l178_17869

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_union_A_B : 
  (A ∪ B)ᶜ = {0, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l178_17869


namespace NUMINAMATH_CALUDE_f_8_5_equals_1_5_l178_17876

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_8_5_equals_1_5 (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 3)
  (h3 : ∀ x ∈ Set.Icc 0 1, f x = 3 * x) :
  f 8.5 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_f_8_5_equals_1_5_l178_17876


namespace NUMINAMATH_CALUDE_green_bows_count_l178_17899

theorem green_bows_count (total : ℕ) (white : ℕ) : 
  (1 / 5 : ℚ) * total + (1 / 2 : ℚ) * total + (1 / 10 : ℚ) * total + white = total →
  white = 30 →
  (1 / 10 : ℚ) * total = 15 := by
sorry

end NUMINAMATH_CALUDE_green_bows_count_l178_17899


namespace NUMINAMATH_CALUDE_sector_central_angle_l178_17800

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l178_17800


namespace NUMINAMATH_CALUDE_alpha_not_rational_l178_17821

theorem alpha_not_rational (α : ℝ) (h : Real.cos (α * π / 180) = 1/3) : ¬ (∃ (m n : ℤ), α = m / n) := by
  sorry

end NUMINAMATH_CALUDE_alpha_not_rational_l178_17821


namespace NUMINAMATH_CALUDE_complete_square_and_calculate_l178_17898

theorem complete_square_and_calculate :
  ∀ m n p : ℝ,
  (∀ x : ℝ, 2 * x^2 - 8 * x + 19 = m * (x - n)^2 + p) →
  2017 + m * p - 5 * n = 2029 := by
sorry

end NUMINAMATH_CALUDE_complete_square_and_calculate_l178_17898


namespace NUMINAMATH_CALUDE_refrigerator_profit_percentage_l178_17868

/-- Calculates the percentage of profit for a refrigerator sale --/
theorem refrigerator_profit_percentage
  (discounted_price : ℝ)
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : discounted_price = 14500)
  (h2 : discount_percentage = 20)
  (h3 : transport_cost = 125)
  (h4 : installation_cost = 250)
  (h5 : selling_price = 20350) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 36.81) < 0.01 ∧
    profit_percentage = (selling_price - (discounted_price + transport_cost + installation_cost)) /
                        (discounted_price + transport_cost + installation_cost) * 100 :=
by sorry


end NUMINAMATH_CALUDE_refrigerator_profit_percentage_l178_17868


namespace NUMINAMATH_CALUDE_chromatic_number_lower_bound_l178_17847

/-- A simple graph -/
structure Graph (V : Type*) where
  adj : V → V → Prop

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The maximum size of cliques in a graph -/
def omega (G : Graph V) : ℕ :=
  sorry

/-- The maximum size of independent sets in a graph -/
def omegaBar (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ :=
  sorry

/-- The main theorem -/
theorem chromatic_number_lower_bound (G : Graph V) :
  chromaticNumber G ≥ max (omega G) (Fintype.card V / omegaBar G) :=
sorry

end NUMINAMATH_CALUDE_chromatic_number_lower_bound_l178_17847
