import Mathlib

namespace strongest_correlation_l579_57968

-- Define the correlation coefficients
def r₁ : ℝ := 0
def r₂ : ℝ := -0.95
def r₃ : ℝ := 0.89  -- We use the absolute value directly as it's given
def r₄ : ℝ := 0.75

-- Theorem stating that r₂ has the largest absolute value
theorem strongest_correlation :
  abs r₂ > abs r₁ ∧ abs r₂ > abs r₃ ∧ abs r₂ > abs r₄ := by
  sorry


end strongest_correlation_l579_57968


namespace sum_two_smallest_prime_factors_of_180_l579_57973

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def prime_factors (n : ℕ) : Set ℕ := {p : ℕ | is_prime p ∧ p ∣ n}

theorem sum_two_smallest_prime_factors_of_180 :
  ∃ (p q : ℕ), p ∈ prime_factors 180 ∧ q ∈ prime_factors 180 ∧
  p < q ∧
  (∀ r ∈ prime_factors 180, r ≠ p → r ≥ q) ∧
  p + q = 5 :=
sorry

end sum_two_smallest_prime_factors_of_180_l579_57973


namespace min_value_product_l579_57951

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 4 ∧
    (3 * a' + b') * (2 * b' + 3 * c') * (a' * c' + 4) = 384 :=
by sorry

end min_value_product_l579_57951


namespace bicycle_wheels_l579_57961

/-- Proves that each bicycle has 2 wheels given the conditions of the problem -/
theorem bicycle_wheels :
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 4
  let num_unicycles : ℕ := 7
  let tricycle_wheels : ℕ := 3
  let unicycle_wheels : ℕ := 1
  let total_wheels : ℕ := 25
  ∃ (bicycle_wheels : ℕ),
    bicycle_wheels * num_bicycles +
    tricycle_wheels * num_tricycles +
    unicycle_wheels * num_unicycles = total_wheels ∧
    bicycle_wheels = 2 :=
by
  sorry

end bicycle_wheels_l579_57961


namespace no_cyclic_quadratic_trinomial_l579_57922

/-- A quadratic trinomial is a polynomial of degree 2 -/
def QuadraticTrinomial (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem stating that no quadratic trinomial satisfies the cyclic property -/
theorem no_cyclic_quadratic_trinomial :
  ¬ ∃ (f : ℝ → ℝ) (a b c : ℝ),
    QuadraticTrinomial f ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    f a = b ∧ f b = c ∧ f c = a :=
sorry

end no_cyclic_quadratic_trinomial_l579_57922


namespace coin_division_problem_l579_57950

theorem coin_division_problem : ∃ n : ℕ,
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) ∧
  n % 8 = 6 ∧
  n % 7 = 5 ∧
  n % 9 = 0 :=
sorry

end coin_division_problem_l579_57950


namespace intersection_m_complement_n_l579_57904

/-- The intersection of set M and the complement of set N in the real numbers -/
theorem intersection_m_complement_n :
  let U : Set ℝ := Set.univ
  let M : Set ℝ := {x | x^2 - 2*x < 0}
  let N : Set ℝ := {x | x ≥ 1}
  M ∩ (U \ N) = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_m_complement_n_l579_57904


namespace tan_negative_thirteen_fourths_pi_l579_57939

theorem tan_negative_thirteen_fourths_pi : Real.tan (-13/4 * π) = -1 := by
  sorry

end tan_negative_thirteen_fourths_pi_l579_57939


namespace sine_cosine_acute_less_than_one_l579_57916

-- Define an acute angle
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define sine and cosine for an acute angle in a right-angled triangle
def sine_acute (α : Real) (h : is_acute_angle α) : Real := sorry
def cosine_acute (α : Real) (h : is_acute_angle α) : Real := sorry

-- Theorem statement
theorem sine_cosine_acute_less_than_one (α : Real) (h : is_acute_angle α) :
  sine_acute α h < 1 ∧ cosine_acute α h < 1 := by sorry

end sine_cosine_acute_less_than_one_l579_57916


namespace expand_expression_l579_57986

theorem expand_expression (x : ℝ) : 4 * (5 * x^3 - 3 * x^2 + 7 * x - 2) = 20 * x^3 - 12 * x^2 + 28 * x - 8 := by
  sorry

end expand_expression_l579_57986


namespace book_price_increase_l579_57926

theorem book_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 390) :
  (new_price - original_price) / original_price * 100 = 30 := by
sorry

end book_price_increase_l579_57926


namespace beth_crayons_count_l579_57960

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayons_count : total_crayons = 46 := by
  sorry

end beth_crayons_count_l579_57960


namespace perfect_square_digits_l579_57905

theorem perfect_square_digits (a b x y : ℕ) : 
  (∃ n : ℕ, a = n^2) →  -- a is a perfect square
  (∃ m : ℕ, b = m^2) →  -- b is a perfect square
  a % 10 = 1 →         -- unit digit of a is 1
  (a / 10) % 10 = x →  -- tens digit of a is x
  b % 10 = 6 →         -- unit digit of b is 6
  (b / 10) % 10 = y →  -- tens digit of b is y
  Even x ∧ Odd y :=
by sorry

end perfect_square_digits_l579_57905


namespace smallest_k_with_remainder_one_l579_57934

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_k_with_remainder_one_l579_57934


namespace sum_of_ten_and_hundredth_l579_57930

theorem sum_of_ten_and_hundredth : 10 + 0.01 = 10.01 := by
  sorry

end sum_of_ten_and_hundredth_l579_57930


namespace largest_difference_is_209_l579_57958

/-- A type representing a 20 × 20 square table filled with distinct natural numbers from 1 to 400. -/
def Table := Fin 20 → Fin 20 → Fin 400

/-- The property that all numbers in the table are distinct. -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l → (i = k ∧ j = l)

/-- The property that there exist two numbers in the same row or column with a difference of at least N. -/
def has_difference_at_least (t : Table) (N : ℕ) : Prop :=
  ∃ i j k, (j = k ∧ |t i j - t i k| ≥ N) ∨ (i = k ∧ |t i j - t k j| ≥ N)

/-- The main theorem stating that 209 is the largest value satisfying the condition. -/
theorem largest_difference_is_209 :
  (∀ t : Table, all_distinct t → has_difference_at_least t 209) ∧
  ¬(∀ t : Table, all_distinct t → has_difference_at_least t 210) :=
sorry

end largest_difference_is_209_l579_57958


namespace trigonometric_sum_equals_sqrt_three_l579_57941

theorem trigonometric_sum_equals_sqrt_three (x : ℝ) 
  (h : Real.tan (4 * x) = Real.sqrt 3 / 3) : 
  (Real.sin (4 * x)) / (Real.cos (8 * x) * Real.cos (4 * x)) + 
  (Real.sin (2 * x)) / (Real.cos (4 * x) * Real.cos (2 * x)) + 
  (Real.sin x) / (Real.cos (2 * x) * Real.cos x) + 
  (Real.sin x) / (Real.cos x) = Real.sqrt 3 := by
  sorry

end trigonometric_sum_equals_sqrt_three_l579_57941


namespace crate_stacking_probability_l579_57987

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of ways to arrange n crates with given counts of each orientation -/
def arrangementCount (n a b c : ℕ) : ℕ := sorry

/-- The probability of stacking crates to achieve a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates targetHeight : ℕ) : ℚ :=
  let totalArrangements := 3^numCrates
  let validArrangements := 
    arrangementCount numCrates 8 0 4 + 
    arrangementCount numCrates 6 3 3 + 
    arrangementCount numCrates 4 6 2 + 
    arrangementCount numCrates 2 9 1 + 
    arrangementCount numCrates 0 12 0
  validArrangements / totalArrangements

theorem crate_stacking_probability : 
  let dimensions := CrateDimensions.mk 3 4 6
  stackProbability dimensions 12 48 = 37522 / 531441 := by sorry

end crate_stacking_probability_l579_57987


namespace ashley_champagne_toast_l579_57947

/-- The number of bottles of champagne needed for a wedding toast --/
def bottlesNeeded (guests : ℕ) (glassesPerGuest : ℕ) (servingsPerBottle : ℕ) : ℕ :=
  (guests * glassesPerGuest + servingsPerBottle - 1) / servingsPerBottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_toast :
  bottlesNeeded 120 2 6 = 40 := by
  sorry

end ashley_champagne_toast_l579_57947


namespace new_girl_weight_l579_57979

/-- Given a group of 8 girls, if replacing one girl weighing 70 kg with a new girl
    increases the average weight by 3 kg, then the weight of the new girl is 94 kg. -/
theorem new_girl_weight (W : ℝ) (new_weight : ℝ) : 
  (W / 8 + 3) * 8 = W - 70 + new_weight →
  new_weight = 94 := by
  sorry

end new_girl_weight_l579_57979


namespace walter_zoo_time_l579_57918

theorem walter_zoo_time (total_time seals penguins elephants : ℕ) : 
  total_time = 130 ∧ 
  penguins = 8 * seals ∧ 
  elephants = 13 ∧ 
  seals + penguins + elephants = total_time → 
  seals = 13 := by
sorry

end walter_zoo_time_l579_57918


namespace second_most_frequent_is_23_l579_57921

-- Define the function m(i) which represents the number of drawings where i appears in the second position
def m (i : ℕ) : ℕ := 
  if 2 ≤ i ∧ i ≤ 87 then
    (i - 1) * (90 - i).choose 3
  else
    0

-- Define the lottery parameters
def lotterySize : ℕ := 6
def lotteryRange : ℕ := 90

-- Theorem statement
theorem second_most_frequent_is_23 : 
  ∀ i, 2 ≤ i ∧ i ≤ 87 → m i ≤ m 23 :=
sorry

end second_most_frequent_is_23_l579_57921


namespace closest_integer_to_double_sum_l579_57959

/-- The number of distinct prime divisors of n that are at least k -/
def mho (n k : ℕ+) : ℕ := sorry

/-- The double sum in the problem -/
noncomputable def doubleSum : ℝ := sorry

theorem closest_integer_to_double_sum : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1/2 ∧ doubleSum = 167 + ε := by sorry

end closest_integer_to_double_sum_l579_57959


namespace sum_of_ages_l579_57991

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 20 →
  jill_age = 13 →
  henry_age - 6 = 2 * (jill_age - 6) →
  henry_age + jill_age = 33 :=
by
  sorry

end sum_of_ages_l579_57991


namespace inequality_holds_l579_57910

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : a / c^2 > b / c^2 := by
  sorry

end inequality_holds_l579_57910


namespace problem_solution_l579_57946

theorem problem_solution (x y z : ℝ) : 
  (3 * x = 0.75 * y) →
  (x = 24) →
  (z = 0.5 * y) →
  (z = 48) := by
  sorry

end problem_solution_l579_57946


namespace market_equilibrium_and_subsidy_effect_l579_57920

-- Define the supply function
def supply (p : ℝ) : ℝ := 2 + 8 * p

-- Define the demand function (to be derived)
def demand (p : ℝ) : ℝ := -2 * p + 12

-- Define equilibrium
def is_equilibrium (p : ℝ) : Prop := supply p = demand p

-- Define the subsidy amount
def subsidy : ℝ := 1

-- Define the new supply function with subsidy
def supply_with_subsidy (p : ℝ) : ℝ := supply (p + subsidy)

-- Define the new equilibrium with subsidy
def is_equilibrium_with_subsidy (p : ℝ) : Prop := supply_with_subsidy p = demand p

theorem market_equilibrium_and_subsidy_effect :
  -- Original equilibrium
  (∃ p q : ℝ, p = 1 ∧ q = 10 ∧ is_equilibrium p ∧ supply p = q) ∧
  -- Effect of subsidy
  (∃ p' q' : ℝ, is_equilibrium_with_subsidy p' ∧ supply_with_subsidy p' = q' ∧ q' - 10 = 1.6) :=
by sorry

end market_equilibrium_and_subsidy_effect_l579_57920


namespace fourth_circle_radius_l579_57940

def circle_configuration (radii : Fin 7 → ℝ) : Prop :=
  ∀ i : Fin 6, radii i < radii (i + 1) ∧ 
  ∃ (r : ℝ), ∀ i : Fin 6, radii (i + 1) = radii i * r

theorem fourth_circle_radius 
  (radii : Fin 7 → ℝ) 
  (h_config : circle_configuration radii) 
  (h_smallest : radii 0 = 6) 
  (h_largest : radii 6 = 24) : 
  radii 3 = 12 :=
sorry

end fourth_circle_radius_l579_57940


namespace monic_cubic_polynomial_uniqueness_l579_57927

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) (x : ℂ) : ℂ := x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let q := MonicCubicPolynomial a b c
  (q (2 - I) = 0) → (q 0 = -40) →
  (∀ x, q x = x^3 - (61/4)*x^2 + (305/4)*x - 225/4) :=
sorry

end monic_cubic_polynomial_uniqueness_l579_57927


namespace quadratic_function_passes_through_points_l579_57924

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem quadratic_function_passes_through_points :
  f (-1) = 0 ∧ f 3 = 0 ∧ f 2 = 3 := by
  sorry

end quadratic_function_passes_through_points_l579_57924


namespace boy_age_theorem_l579_57949

/-- The age of the boy not included in either group -/
def X (A : ℝ) : ℝ := 606 - 11 * A

/-- Theorem stating the relationship between X and A -/
theorem boy_age_theorem (A : ℝ) :
  let first_six_total : ℝ := 6 * 49
  let last_six_total : ℝ := 6 * 52
  let total_boys : ℕ := 11
  X A = first_six_total + last_six_total - total_boys * A := by
  sorry

end boy_age_theorem_l579_57949


namespace musicians_count_l579_57983

theorem musicians_count : ∃! n : ℕ, 
  80 < n ∧ n < 130 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 3 ∧ 
  n = 123 := by
sorry

end musicians_count_l579_57983


namespace preceding_binary_number_l579_57909

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : BinaryNumber :=
  sorry

theorem preceding_binary_number (M : BinaryNumber) :
  M = [0, 1, 0, 1, 0, 0, 1] →
  decimal_to_binary (binary_to_decimal M - 1) = [1, 0, 0, 1, 0, 0, 1] :=
sorry

end preceding_binary_number_l579_57909


namespace average_book_width_l579_57907

theorem average_book_width :
  let book_widths : List ℝ := [3, 0.5, 1.5, 4, 2, 5, 8]
  let sum_widths : ℝ := book_widths.sum
  let num_books : ℕ := book_widths.length
  let average_width : ℝ := sum_widths / num_books
  average_width = 3.43 := by sorry

end average_book_width_l579_57907


namespace reina_kevin_marble_ratio_l579_57929

/-- Proves that the ratio of Reina's marbles to Kevin's marbles is 4:1 -/
theorem reina_kevin_marble_ratio :
  let kevin_counters : ℕ := 40
  let kevin_marbles : ℕ := 50
  let reina_counters : ℕ := 3 * kevin_counters
  let reina_total : ℕ := 320
  let reina_marbles : ℕ := reina_total - reina_counters
  (reina_marbles : ℚ) / kevin_marbles = 4 / 1 := by
  sorry

end reina_kevin_marble_ratio_l579_57929


namespace arithmetic_sequence_problem_l579_57975

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where 
    2(a_1 + a_3 + a_5) + 3(a_8 + a_10) = 36, prove that a_6 = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) :
  a 6 = 3 := by
  sorry

end arithmetic_sequence_problem_l579_57975


namespace rectangle_area_ratio_l579_57923

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 4/5, then the ratio of their areas is 16:25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
  sorry

end rectangle_area_ratio_l579_57923


namespace interest_rate_calculation_l579_57996

theorem interest_rate_calculation (total_sum second_part : ℝ)
  (h1 : total_sum = 2691)
  (h2 : second_part = 1656)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_rate_second := (first_part * interest_rate_first * time_first) / (second_part * time_second)
  ∃ ε > 0, |interest_rate_second - 0.05| < ε :=
sorry

end interest_rate_calculation_l579_57996


namespace negation_of_proposition_l579_57902

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end negation_of_proposition_l579_57902


namespace batsman_average_l579_57928

theorem batsman_average (total_innings : ℕ) (last_score : ℕ) (average_increase : ℝ) : 
  total_innings = 20 →
  last_score = 90 →
  average_increase = 2 →
  (↑total_innings * (average_after_last_innings - average_increase) + ↑last_score) / ↑total_innings = average_after_last_innings →
  average_after_last_innings = 52 :=
by
  sorry

#check batsman_average

end batsman_average_l579_57928


namespace champion_is_C_l579_57974

-- Define the contestants
inductive Contestant : Type
  | A | B | C | D | E

-- Define the predictions
def father_prediction (c : Contestant) : Prop :=
  c = Contestant.A ∨ c = Contestant.C

def mother_prediction (c : Contestant) : Prop :=
  c ≠ Contestant.B ∧ c ≠ Contestant.C

def child_prediction (c : Contestant) : Prop :=
  c = Contestant.D ∨ c = Contestant.E

-- Define the condition that only one prediction is correct
def only_one_correct (c : Contestant) : Prop :=
  (father_prediction c ∧ ¬mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ ¬mother_prediction c ∧ child_prediction c)

-- Theorem statement
theorem champion_is_C :
  ∃ (c : Contestant), only_one_correct c → c = Contestant.C :=
sorry

end champion_is_C_l579_57974


namespace complex_solution_l579_57955

theorem complex_solution (z : ℂ) (h : (2 + Complex.I) * z = 3 + 4 * Complex.I) :
  z = 2 + Complex.I := by
  sorry

end complex_solution_l579_57955


namespace interior_edge_sum_is_eight_l579_57943

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  borderWidth : ℝ

/-- Calculate the area of the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.borderWidth) * (frame.outerHeight - 2 * frame.borderWidth)

/-- Calculate the sum of the interior edge lengths -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * ((frame.outerWidth - 2 * frame.borderWidth) + (frame.outerHeight - 2 * frame.borderWidth))

/-- Theorem: The sum of interior edges is 8 inches for a frame with given properties -/
theorem interior_edge_sum_is_eight (frame : PictureFrame) 
  (h1 : frame.borderWidth = 2)
  (h2 : frameArea frame = 32)
  (h3 : frame.outerWidth = 7) : 
  interiorEdgeSum frame = 8 := by
  sorry


end interior_edge_sum_is_eight_l579_57943


namespace triangle_perimeter_l579_57985

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 15) (hc : c = 19) :
  a + b + c = 44 := by
  sorry

end triangle_perimeter_l579_57985


namespace cubic_function_uniqueness_l579_57966

/-- Given a cubic function f(x) = ax³ + bx² passing through (-1, 2) with slope -3 at x = -1,
    prove that f(x) = x³ + 3x² -/
theorem cubic_function_uniqueness (a b : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^3 + b * x^2
  let f' := fun (x : ℝ) ↦ 3 * a * x^2 + 2 * b * x
  (f (-1) = 2) → (f' (-1) = -3) → (a = 1 ∧ b = 3) := by sorry

end cubic_function_uniqueness_l579_57966


namespace board_cutting_theorem_l579_57982

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end board_cutting_theorem_l579_57982


namespace building_height_calculation_l579_57948

-- Define the given constants
def box_height : ℝ := 3
def box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- Define the theorem
theorem building_height_calculation :
  ∃ (building_height : ℝ),
    (box_height / box_shadow = building_height / building_shadow) ∧
    building_height = 9 := by
  sorry

end building_height_calculation_l579_57948


namespace true_discount_calculation_l579_57912

/-- Given a bill with face value 540 and banker's discount 108, prove the true discount is 90 -/
theorem true_discount_calculation (face_value banker_discount : ℚ) 
  (h1 : face_value = 540)
  (h2 : banker_discount = 108)
  (h3 : ∀ (td : ℚ), banker_discount = td + (td * banker_discount / face_value)) :
  ∃ (true_discount : ℚ), true_discount = 90 ∧ 
    banker_discount = true_discount + (true_discount * banker_discount / face_value) := by
  sorry

#check true_discount_calculation

end true_discount_calculation_l579_57912


namespace country_z_diploma_percentage_l579_57938

theorem country_z_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_job_choice : ℝ := 18
  let diploma_no_job_choice_ratio : ℝ := 0.25
  let job_choice : ℝ := 40

  let diploma_job_choice : ℝ := job_choice - no_diploma_job_choice
  let no_job_choice : ℝ := total_population - job_choice
  let diploma_no_job_choice : ℝ := diploma_no_job_choice_ratio * no_job_choice

  diploma_job_choice + diploma_no_job_choice = 37 :=
by sorry

end country_z_diploma_percentage_l579_57938


namespace larger_number_proof_l579_57965

theorem larger_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 3) (h3 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end larger_number_proof_l579_57965


namespace absolute_value_and_exponent_zero_sum_l579_57964

theorem absolute_value_and_exponent_zero_sum : |-5| + (2 - Real.sqrt 3)^0 = 6 := by
  sorry

end absolute_value_and_exponent_zero_sum_l579_57964


namespace system_solution_l579_57962

def solution_set : Set (ℂ × ℂ × ℂ) :=
  {(0, 0, 0), (2/3, -1/3, -1/3), (1/3, (-1+Complex.I*Real.sqrt 3)/6, (-1-Complex.I*Real.sqrt 3)/6),
   (1/3, (-1-Complex.I*Real.sqrt 3)/6, (-1+Complex.I*Real.sqrt 3)/6), (1, 0, 0), (1/3, 1/3, 1/3),
   (2/3, (1+Complex.I*Real.sqrt 3)/6, (1-Complex.I*Real.sqrt 3)/6),
   (2/3, (1-Complex.I*Real.sqrt 3)/6, (1+Complex.I*Real.sqrt 3)/6)}

theorem system_solution (x y z : ℂ) :
  (x^2 + 2*y*z = x ∧ y^2 + 2*z*x = z ∧ z^2 + 2*x*y = y) ↔ (x, y, z) ∈ solution_set :=
sorry

end system_solution_l579_57962


namespace average_income_P_and_R_l579_57917

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of P and R is 5200. -/
theorem average_income_P_and_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 →
  (Q + R) / 2 = 6250 →
  P = 4000 →
  (P + R) / 2 = 5200 := by
  sorry

end average_income_P_and_R_l579_57917


namespace power_tower_mod_1000_l579_57963

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 625 [ZMOD 1000] := by sorry

end power_tower_mod_1000_l579_57963


namespace problem_1999_squared_minus_1998_times_2002_l579_57906

theorem problem_1999_squared_minus_1998_times_2002 : 1999^2 - 1998 * 2002 = -3991 := by
  sorry

end problem_1999_squared_minus_1998_times_2002_l579_57906


namespace polynomial_properties_l579_57901

-- Define the polynomial coefficients
variable (a : Fin 12 → ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (x - 2)^11 = a 0 + a 1 * (x - 1) + a 2 * (x - 1)^2 + 
               a 3 * (x - 1)^3 + a 4 * (x - 1)^4 + a 5 * (x - 1)^5 + 
               a 6 * (x - 1)^6 + a 7 * (x - 1)^7 + a 8 * (x - 1)^8 + 
               a 9 * (x - 1)^9 + a 10 * (x - 1)^10 + a 11 * (x - 1)^11

-- Theorem to prove
theorem polynomial_properties (a : Fin 12 → ℚ) 
  (h : ∀ x, main_equation a x) : 
  a 10 = -11 ∧ a 2 + a 4 + a 6 + a 8 + a 10 = -1023 := by
  sorry

end polynomial_properties_l579_57901


namespace sum_s_r_equals_negative_62_l579_57997

def r (x : ℝ) : ℝ := abs x + 1

def s (x : ℝ) : ℝ := -2 * abs x

def xValues : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_equals_negative_62 :
  (xValues.map (fun x => s (r x))).sum = -62 := by
  sorry

end sum_s_r_equals_negative_62_l579_57997


namespace sum_of_roots_eq_fourteen_l579_57969

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, 
  (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by sorry

end sum_of_roots_eq_fourteen_l579_57969


namespace packaging_combinations_l579_57937

/-- The number of wrapping paper designs --/
def num_wrapping_paper : ℕ := 10

/-- The number of ribbon colors --/
def num_ribbons : ℕ := 4

/-- The number of gift card varieties --/
def num_gift_cards : ℕ := 5

/-- The number of decorative sticker styles --/
def num_stickers : ℕ := 6

/-- The total number of unique packaging combinations --/
def total_combinations : ℕ := num_wrapping_paper * num_ribbons * num_gift_cards * num_stickers

/-- Theorem stating that the total number of unique packaging combinations is 1200 --/
theorem packaging_combinations : total_combinations = 1200 := by
  sorry

end packaging_combinations_l579_57937


namespace range_of_m_l579_57953

def p (x : ℝ) : Prop := (x - 1) / x ≤ 0

def q (x m : ℝ) : Prop := (x - m) * (x - m + 2) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 1 ≤ m ∧ m ≤ 2 := by
  sorry

end range_of_m_l579_57953


namespace twenty_fifth_term_is_173_l579_57977

/-- The nth term of an arithmetic progression -/
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 25th term of the arithmetic progression with first term 5 and common difference 7 is 173 -/
theorem twenty_fifth_term_is_173 :
  arithmetic_progression 5 7 25 = 173 := by
  sorry

end twenty_fifth_term_is_173_l579_57977


namespace five_arithmetic_operations_l579_57952

theorem five_arithmetic_operations :
  -- 1. 5555 = 7
  (5 + 5 + 5) / 5 = 3 ∧
  (5 + 5) / 5 + 5 = 7 ∧
  -- 2. 5555 = 55
  (5 + 5) * 5 + 5 = 55 ∧
  -- 3. 5,5,5,5 = 4
  (5 * 5 - 5) / 5 = 4 ∧
  -- 4. 5,5,5,5 = 26
  5 * 5 + (5 / 5) = 26 ∧
  -- 5. 5,5,5,5 = 120
  5 * 5 * 5 - 5 = 120 ∧
  -- 6. 5,5,5,5 = 5
  (5 - 5) * 5 + 5 = 5 ∧
  -- 7. 5555 = 30
  (5 / 5 + 5) * 5 = 30 ∧
  -- 8. 5,5,5,5 = 130
  5 * 5 * 5 + 5 = 130 ∧
  -- 9. 5555 = 6
  (5 * 5 + 5) / 5 = 6 ∧
  -- 10. 5555 = 50
  5 * 5 + 5 * 5 = 50 ∧
  -- 11. 5555 = 625
  5 * 5 * 5 * 5 = 625 := by
  sorry

#check five_arithmetic_operations

end five_arithmetic_operations_l579_57952


namespace combined_average_marks_specific_average_marks_l579_57903

/-- Given two classes of students with their respective sizes and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) := by
  sorry

/-- The average mark of all students given the specific class sizes and averages. -/
theorem specific_average_marks :
  let n1 := 24
  let n2 := 50
  let avg1 := 40
  let avg2 := 60
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (24 * 40 + 50 * 60) / (24 + 50) := by
  sorry

end combined_average_marks_specific_average_marks_l579_57903


namespace negation_of_proposition_l579_57989

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n : ℕ, 3^n ≥ n + 1) ↔ (∃ n : ℕ, 3^n < n + 1) :=
by sorry

end negation_of_proposition_l579_57989


namespace perfect_square_property_l579_57998

theorem perfect_square_property (x y z : ℤ) (h : x * y + y * z + z * x = 1) :
  (1 + x^2) * (1 + y^2) * (1 + z^2) = ((x + y) * (y + z) * (x + z))^2 := by
  sorry

end perfect_square_property_l579_57998


namespace cricket_average_increase_l579_57992

/-- Represents the problem of calculating the increase in average runs -/
def calculateAverageIncrease (initialMatches : ℕ) (initialAverage : ℚ) (nextMatchRuns : ℕ) : ℚ :=
  let totalInitialRuns := initialMatches * initialAverage
  let totalMatches := initialMatches + 1
  let totalRuns := totalInitialRuns + nextMatchRuns
  (totalRuns / totalMatches) - initialAverage

/-- The theorem stating the solution to the cricket player's average problem -/
theorem cricket_average_increase :
  calculateAverageIncrease 10 32 76 = 4 := by
  sorry


end cricket_average_increase_l579_57992


namespace hall_area_l579_57972

/-- The area of a rectangular hall with given length and breadth relationship -/
theorem hall_area (length breadth : ℝ) : 
  length = 30 ∧ length = breadth + 5 → length * breadth = 750 := by
  sorry

end hall_area_l579_57972


namespace javiers_cats_l579_57980

/-- Calculates the number of cats in Javier's household -/
def number_of_cats (adults children dogs total_legs : ℕ) : ℕ :=
  let human_legs := 2 * (adults + children)
  let dog_legs := 4 * dogs
  let remaining_legs := total_legs - human_legs - dog_legs
  remaining_legs / 4

/-- Theorem stating that the number of cats in Javier's household is 1 -/
theorem javiers_cats :
  number_of_cats 2 3 2 22 = 1 :=
by sorry

end javiers_cats_l579_57980


namespace average_unchanged_with_double_inclusion_l579_57925

theorem average_unchanged_with_double_inclusion (n : ℕ) (scores : Fin n → ℝ) :
  let original_avg := (Finset.sum Finset.univ (λ i => scores i)) / n
  let new_sum := (Finset.sum Finset.univ (λ i => scores i)) + 2 * original_avg
  let new_avg := new_sum / (n + 2)
  new_avg = original_avg :=
by sorry

end average_unchanged_with_double_inclusion_l579_57925


namespace maryann_rescue_time_l579_57911

/-- The time (in minutes) it takes Maryann to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℕ := 6

/-- The time (in minutes) it takes Maryann to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℕ := 8

/-- The number of friends Maryann needs to rescue -/
def number_of_friends : ℕ := 3

/-- The time it takes to free one friend -/
def time_per_friend : ℕ := cheap_handcuff_time + expensive_handcuff_time

/-- The total time it takes to free all friends -/
def total_rescue_time : ℕ := time_per_friend * number_of_friends

theorem maryann_rescue_time : total_rescue_time = 42 := by
  sorry

end maryann_rescue_time_l579_57911


namespace find_m_l579_57981

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def C_UA (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, C_UA m = {1, 4} := by
  sorry

end find_m_l579_57981


namespace max_achievable_grade_l579_57993

theorem max_achievable_grade (test1_score test2_score test3_score : ℝ)
  (test1_weight test2_weight test3_weight test4_weight : ℝ)
  (max_extra_credit : ℝ) (target_grade : ℝ) :
  test1_score = 95 ∧ test2_score = 80 ∧ test3_score = 90 ∧
  test1_weight = 0.25 ∧ test2_weight = 0.3 ∧ test3_weight = 0.25 ∧ test4_weight = 0.2 ∧
  max_extra_credit = 5 ∧ target_grade = 93 →
  let current_weighted_grade := test1_score * test1_weight + test2_score * test2_weight + test3_score * test3_weight
  let max_fourth_test_score := 100 + max_extra_credit
  let max_achievable_grade := current_weighted_grade + max_fourth_test_score * test4_weight
  max_achievable_grade < target_grade ∧ max_achievable_grade = 91.25 :=
by sorry

end max_achievable_grade_l579_57993


namespace lcm_gcd_product_l579_57931

theorem lcm_gcd_product (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end lcm_gcd_product_l579_57931


namespace billy_weight_l579_57919

theorem billy_weight (carl_weight brad_weight dave_weight billy_weight edgar_weight : ℝ) :
  carl_weight = 145 ∧
  brad_weight = carl_weight + 5 ∧
  dave_weight = carl_weight + 8 ∧
  dave_weight = 2 * brad_weight ∧
  edgar_weight = 3 * dave_weight - 20 ∧
  billy_weight = brad_weight + 9 →
  billy_weight = 85.5 := by
sorry

end billy_weight_l579_57919


namespace non_integer_mean_arrangement_l579_57999

theorem non_integer_mean_arrangement (N : ℕ) (h : Even N) :
  ∃ (arr : List ℕ),
    (arr.length = N) ∧
    (∀ x, x ∈ arr ↔ 1 ≤ x ∧ x ≤ N) ∧
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ N →
      ¬(∃ (k : ℕ), (arr.take j).sum - (arr.take (i-1)).sum = k * (j - i + 1))) :=
by sorry

end non_integer_mean_arrangement_l579_57999


namespace classroom_books_count_l579_57936

theorem classroom_books_count (num_children : ℕ) (books_per_child : ℕ) (teacher_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → teacher_books = 8 →
  num_children * books_per_child + teacher_books = 78 := by
sorry

end classroom_books_count_l579_57936


namespace bus_average_speed_l579_57956

/-- The average speed of a bus traveling three equal-length sections of a road -/
theorem bus_average_speed (a : ℝ) (h : a > 0) : 
  let v1 : ℝ := 50  -- speed of first section in km/h
  let v2 : ℝ := 30  -- speed of second section in km/h
  let v3 : ℝ := 70  -- speed of third section in km/h
  let total_distance : ℝ := 3 * a  -- total distance traveled
  let total_time : ℝ := a / v1 + a / v2 + a / v3  -- total time taken
  let average_speed : ℝ := total_distance / total_time
  ∃ (ε : ℝ), ε > 0 ∧ |average_speed - 44| < ε :=
by
  sorry


end bus_average_speed_l579_57956


namespace red_card_value_is_three_l579_57957

/-- The value of a red card in credits -/
def red_card_value : ℕ := sorry

/-- The value of a blue card in credits -/
def blue_card_value : ℕ := 5

/-- The total number of cards needed to play a game -/
def total_cards : ℕ := 20

/-- The total number of credits available to buy cards -/
def total_credits : ℕ := 84

/-- The number of red cards used when playing -/
def red_cards_used : ℕ := 8

theorem red_card_value_is_three :
  red_card_value = 3 :=
by sorry

end red_card_value_is_three_l579_57957


namespace box_volume_l579_57995

theorem box_volume (l w h : ℝ) (shortest_path : ℝ) : 
  l = 6 → w = 6 → shortest_path = 20 → 
  shortest_path^2 = (l + w + h)^2 + w^2 →
  l * w * h = 576 := by
sorry

end box_volume_l579_57995


namespace min_value_of_function_l579_57984

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 2 / (x - 1) ≥ 2 * Real.sqrt 2 + 1 := by
  sorry

end min_value_of_function_l579_57984


namespace sum_of_distances_l579_57913

/-- A circle touches the sides of an angle at points A and B. 
    C is a point on the circle. -/
structure CircleConfig where
  A : Point
  B : Point
  C : Point

/-- The distance from C to line AB is 6 -/
def distance_to_AB (config : CircleConfig) : ℝ := 6

/-- The distances from C to the sides of the angle -/
def distance_to_sides (config : CircleConfig) : ℝ × ℝ := sorry

/-- One distance is 9 times less than the other -/
axiom distance_ratio (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  d₁ = (1/9) * d₂ ∨ d₂ = (1/9) * d₁

theorem sum_of_distances (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  distance_to_AB config + d₁ + d₂ = 12 :=
sorry

end sum_of_distances_l579_57913


namespace wrong_number_calculation_l579_57935

theorem wrong_number_calculation (n : Nat) (initial_avg correct_avg correct_num : ℝ) 
  (h1 : n = 10)
  (h2 : initial_avg = 21)
  (h3 : correct_avg = 22)
  (h4 : correct_num = 36) :
  ∃ wrong_num : ℝ,
    n * correct_avg - n * initial_avg = correct_num - wrong_num ∧
    wrong_num = 26 := by
  sorry

end wrong_number_calculation_l579_57935


namespace coord_sum_of_point_B_l579_57908

/-- Given two points A(0, 0) and B(x, 3) where the slope of AB is 3/4,
    prove that the sum of B's coordinates is 7. -/
theorem coord_sum_of_point_B (x : ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (x, 3)
  (3 - 0) / (x - 0) = 3 / 4 →
  x + 3 = 7 := by
sorry

end coord_sum_of_point_B_l579_57908


namespace intersection_equals_two_l579_57933

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem intersection_equals_two (a : ℝ) :
  A ∩ B a = {2} → a = 2 := by
  sorry

end intersection_equals_two_l579_57933


namespace emma_bank_account_l579_57978

/-- Calculates the final amount in a bank account after a withdrawal and deposit -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  let remaining := initial_savings - withdrawal
  let deposit := 2 * withdrawal
  remaining + deposit

/-- Proves that given the specific conditions, the final amount is $290 -/
theorem emma_bank_account : final_amount 230 60 = 290 := by
  sorry

end emma_bank_account_l579_57978


namespace correct_operation_l579_57990

theorem correct_operation (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end correct_operation_l579_57990


namespace fruit_basket_total_l579_57944

/-- Represents the number of fruit pieces in a basket -/
structure FruitBasket where
  redApples : Nat
  greenApples : Nat
  purpleGrapes : Nat
  yellowBananas : Nat
  orangeOranges : Nat

/-- Calculates the total number of fruit pieces in the basket -/
def totalFruits (basket : FruitBasket) : Nat :=
  basket.redApples + basket.greenApples + basket.purpleGrapes + basket.yellowBananas + basket.orangeOranges

/-- Theorem stating that the total number of fruit pieces in the given basket is 24 -/
theorem fruit_basket_total :
  let basket : FruitBasket := {
    redApples := 9,
    greenApples := 4,
    purpleGrapes := 3,
    yellowBananas := 6,
    orangeOranges := 2
  }
  totalFruits basket = 24 := by
  sorry

end fruit_basket_total_l579_57944


namespace paperclip_growth_l579_57914

theorem paperclip_growth (n : ℕ) : (8 * 3^n > 1000) ↔ n ≥ 5 := by sorry

end paperclip_growth_l579_57914


namespace f_shifted_f_identity_l579_57971

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 + 1

-- State the theorem
theorem f_shifted (x : ℝ) : f (x - 1) = x^2 - 2*x + 2 := by
  sorry

-- Prove that f(x) = x^2 + 1
theorem f_identity (x : ℝ) : f x = x^2 + 1 := by
  sorry

end f_shifted_f_identity_l579_57971


namespace phil_bought_cards_for_52_weeks_l579_57942

/-- Represents the number of weeks Phil bought baseball card packs --/
def weeks_buying_cards (cards_per_pack : ℕ) (cards_after_fire : ℕ) : ℕ :=
  (2 * cards_after_fire) / cards_per_pack

/-- Theorem stating that Phil bought cards for 52 weeks --/
theorem phil_bought_cards_for_52_weeks :
  weeks_buying_cards 20 520 = 52 := by
  sorry

end phil_bought_cards_for_52_weeks_l579_57942


namespace creature_probability_l579_57988

/-- Represents the type of creature on the island -/
inductive Creature
| Hare
| Rabbit

/-- The probability of a creature being mistaken -/
def mistakeProbability (c : Creature) : ℚ :=
  match c with
  | Creature.Hare => 1/4
  | Creature.Rabbit => 1/3

/-- The probability of a creature being correct -/
def correctProbability (c : Creature) : ℚ :=
  1 - mistakeProbability c

/-- The probability of a creature being of a certain type -/
def populationProbability (c : Creature) : ℚ := 1/2

theorem creature_probability (A B C : Prop) :
  let pA := populationProbability Creature.Hare
  let pNotA := populationProbability Creature.Rabbit
  let pBA := mistakeProbability Creature.Hare
  let pCA := correctProbability Creature.Hare
  let pBNotA := correctProbability Creature.Rabbit
  let pCNotA := mistakeProbability Creature.Rabbit
  let pABC := pA * pBA * pCA
  let pNotABC := pNotA * pBNotA * pCNotA
  let pBC := pABC + pNotABC
  pABC / pBC = 27/59 := by sorry

end creature_probability_l579_57988


namespace apple_baskets_l579_57976

/-- 
Given two baskets A and B with apples, prove that:
1. If the total amount of apples in both baskets is 75 kg
2. And after transferring 5 kg from A to B, A has 7 kg more than B
Then the original amounts in A and B were 46 kg and 29 kg, respectively
-/
theorem apple_baskets (a b : ℕ) : 
  a + b = 75 → 
  (a - 5) = (b + 5) + 7 → 
  (a = 46 ∧ b = 29) := by
sorry

end apple_baskets_l579_57976


namespace ingrid_income_calculation_l579_57932

def john_income : ℝ := 57000
def john_tax_rate : ℝ := 0.30
def ingrid_tax_rate : ℝ := 0.40
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_income_calculation (ingrid_income : ℝ) : 
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = combined_tax_rate →
  ingrid_income = 72000 := by
sorry

end ingrid_income_calculation_l579_57932


namespace lizzy_money_theorem_l579_57900

def lizzy_money_problem (mother_gave uncle_gave father_gave spent_on_candy : ℕ) : Prop :=
  let initial_amount := mother_gave + father_gave
  let amount_after_spending := initial_amount - spent_on_candy
  let final_amount := amount_after_spending + uncle_gave
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end lizzy_money_theorem_l579_57900


namespace max_value_complex_expression_l579_57915

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ Real.sqrt 637 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = Real.sqrt 637 :=
by sorry

end max_value_complex_expression_l579_57915


namespace base_13_conversion_l579_57970

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its value in base 13 -/
def toBase13Value (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Represents a two-digit number in base 13 -/
structure Base13Number :=
  (msb : Base13Digit)
  (lsb : Base13Digit)

/-- Converts a Base13Number to its decimal (base 10) value -/
def toDecimal (n : Base13Number) : ℕ :=
  13 * (toBase13Value n.msb) + (toBase13Value n.lsb)

theorem base_13_conversion :
  toDecimal (Base13Number.mk Base13Digit.C Base13Digit.D0) = 156 := by
  sorry

end base_13_conversion_l579_57970


namespace dans_helmet_craters_l579_57967

theorem dans_helmet_craters :
  ∀ (d D R r : ℕ),
  D = d + 10 →                   -- Dan's helmet has 10 more craters than Daniel's
  R = D + d + 15 →               -- Rin's helmet has 15 more craters than Dan's and Daniel's combined
  r = 2 * R - 10 →               -- Rina's helmet has double the number of craters in Rin's minus 10
  R = 75 →                       -- Rin's helmet has 75 craters
  d + D + R + r = 540 →          -- Total craters on all helmets is 540
  Even d ∧ Even D ∧ Even R ∧ Even r →  -- Number of craters in each helmet is even
  D = 168 :=
by sorry

end dans_helmet_craters_l579_57967


namespace checkerboard_corner_sum_l579_57994

theorem checkerboard_corner_sum : 
  let n : ℕ := 8  -- size of the checkerboard
  let total_squares : ℕ := n * n
  let top_left : ℕ := 1
  let top_right : ℕ := n
  let bottom_left : ℕ := total_squares - n + 1
  let bottom_right : ℕ := total_squares
  top_left + top_right + bottom_left + bottom_right = 130 :=
by sorry

end checkerboard_corner_sum_l579_57994


namespace age_pencil_ratio_l579_57945

/-- Given the ages and pencil counts of Asaf and Alexander, prove the ratio of their age difference to Asaf's pencil count -/
theorem age_pencil_ratio (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  (alexander_age - asaf_age : ℚ) / asaf_pencils = 1 / 2 := by
sorry

end age_pencil_ratio_l579_57945


namespace parallel_lines_m_value_l579_57954

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line: x + my + 6 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

/-- The second line: 3x + (m - 2)y + 2m = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + (m - 2) * y + 2 * m = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (are_parallel 1 m 3 (m - 2)) → m = -1 := by
  sorry

end parallel_lines_m_value_l579_57954
