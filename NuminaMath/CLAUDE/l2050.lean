import Mathlib

namespace NUMINAMATH_CALUDE_total_students_l2050_205019

/-- The number of students taking history -/
def H : ℕ := 36

/-- The number of students taking statistics -/
def S : ℕ := 32

/-- The number of students taking history or statistics or both -/
def H_or_S : ℕ := 57

/-- The number of students taking history but not statistics -/
def H_not_S : ℕ := 25

/-- The theorem stating that the total number of students in the group is 57 -/
theorem total_students : H_or_S = 57 := by sorry

end NUMINAMATH_CALUDE_total_students_l2050_205019


namespace NUMINAMATH_CALUDE_ellipse_properties_l2050_205016

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Theorem stating the properties of the ellipse and its intersections -/
theorem ellipse_properties :
  ∀ (k m : ℝ),
  m > 0 →
  (∃ (A B : ℝ × ℝ),
    ellipse_C A.1 A.2 ∧
    ellipse_C B.1 B.2 ∧
    line_l k m A.1 A.2 ∧
    line_l k m B.1 B.2 ∧
    (k = 1/2 ∨ k = -1/2) →
    (∃ (c : ℝ), A.1^2 + A.2^2 + B.1^2 + B.2^2 = c) ∧
    (∃ (area : ℝ), area ≤ 1 ∧
      (k = 1/2 ∨ k = -1/2) →
      area = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2050_205016


namespace NUMINAMATH_CALUDE_platform_length_l2050_205036

/-- The length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- km/h
  (h2 : platform_crossing_time = 30)  -- seconds
  (h3 : man_crossing_time = 15)  -- seconds
  : ∃ (platform_length : ℝ), platform_length = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2050_205036


namespace NUMINAMATH_CALUDE_apple_distribution_result_l2050_205044

/-- Represents the apple distribution problem --/
def apple_distribution (jim jane jerry jack jill jasmine jacob : ℕ) : ℚ :=
  let jack_to_jill := jack / 4
  let jasmine_jacob_shared := jasmine + jacob
  let jim_final := jim + (jasmine_jacob_shared / 10)
  let total_apples := jim_final + jane + jerry + (jack - jack_to_jill) + 
                      (jill + jack_to_jill) + (jasmine_jacob_shared / 2) + 
                      (jasmine_jacob_shared / 2)
  let average_apples := total_apples / 7
  average_apples / jim_final

/-- Theorem stating the result of the apple distribution problem --/
theorem apple_distribution_result : 
  ∃ ε > 0, |apple_distribution 20 60 40 80 50 30 90 - 1.705| < ε :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_result_l2050_205044


namespace NUMINAMATH_CALUDE_inequality_proof_l2050_205071

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c + a*b + b*c + c*a + a*b*c = 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2050_205071


namespace NUMINAMATH_CALUDE_domino_placement_theorem_l2050_205082

/-- Represents a 6x6 chessboard -/
def Chessboard : Type := Fin 6 × Fin 6

/-- Represents a domino placement on the chessboard -/
def Domino : Type := Chessboard × Chessboard

/-- Check if two squares are adjacent -/
def adjacent (s1 s2 : Chessboard) : Prop :=
  (s1.1 = s2.1 ∧ s1.2.succ = s2.2) ∨
  (s1.1 = s2.1 ∧ s1.2 = s2.2.succ) ∨
  (s1.1.succ = s2.1 ∧ s1.2 = s2.2) ∨
  (s1.1 = s2.1.succ ∧ s1.2 = s2.2)

/-- Check if a domino placement is valid -/
def valid_domino (d : Domino) : Prop :=
  adjacent d.1 d.2

/-- The main theorem -/
theorem domino_placement_theorem
  (dominos : Finset Domino)
  (h1 : dominos.card = 11)
  (h2 : ∀ d ∈ dominos, valid_domino d)
  (h3 : ∀ s1 s2 : Chessboard, s1 ≠ s2 →
        (∃ d ∈ dominos, d.1 = s1 ∨ d.2 = s1) →
        (∃ d ∈ dominos, d.1 = s2 ∨ d.2 = s2) →
        s1 ≠ s2) :
  ∃ s1 s2 : Chessboard, adjacent s1 s2 ∧
    (∀ d ∈ dominos, d.1 ≠ s1 ∧ d.2 ≠ s1 ∧ d.1 ≠ s2 ∧ d.2 ≠ s2) :=
by sorry

end NUMINAMATH_CALUDE_domino_placement_theorem_l2050_205082


namespace NUMINAMATH_CALUDE_parabola_point_distance_l2050_205009

/-- Given a parabola y^2 = x with focus F(1/4, 0), prove that for any point A(x₀, y₀) on the parabola,
    if AF = |5/4 * x₀|, then x₀ = 1. -/
theorem parabola_point_distance (x₀ y₀ : ℝ) : 
  y₀^2 = x₀ →  -- Point A is on the parabola
  ((x₀ - 1/4)^2 + y₀^2)^(1/2) = |5/4 * x₀| →  -- AF = |5/4 * x₀|
  x₀ = 1 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l2050_205009


namespace NUMINAMATH_CALUDE_max_binomial_coefficient_for_given_sum_l2050_205014

-- Define the function for the sum of coefficients
def sumOfCoefficients (m : ℕ) : ℝ := (5 - 1)^m

-- Define the function for the maximum binomial coefficient
def maxBinomialCoefficient (m : ℕ) : ℕ := Nat.choose m (m / 2)

-- Theorem statement
theorem max_binomial_coefficient_for_given_sum :
  ∃ m : ℕ, sumOfCoefficients m = 256 ∧ maxBinomialCoefficient m = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_binomial_coefficient_for_given_sum_l2050_205014


namespace NUMINAMATH_CALUDE_incircle_excircle_center_distance_l2050_205048

/-- Given a triangle DEF with side lengths, prove the distance between incircle and excircle centers --/
theorem incircle_excircle_center_distance (DE DF EF : ℝ) (h_DE : DE = 20) (h_DF : DF = 21) (h_EF : EF = 29) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let I := Real.sqrt (DE^2 + r^2)
  let E := (DF * I) / DE
  E - I = Real.sqrt 232 / 14 := by sorry

end NUMINAMATH_CALUDE_incircle_excircle_center_distance_l2050_205048


namespace NUMINAMATH_CALUDE_john_change_proof_l2050_205066

/-- Calculates the change received when buying oranges -/
def calculate_change (num_oranges : ℕ) (cost_per_orange_cents : ℕ) (paid_dollars : ℕ) : ℚ :=
  paid_dollars - (num_oranges * cost_per_orange_cents) / 100

theorem john_change_proof :
  calculate_change 4 75 10 = 7 := by
  sorry

#eval calculate_change 4 75 10

end NUMINAMATH_CALUDE_john_change_proof_l2050_205066


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l2050_205088

/-- Represents the digit reversal of a natural number -/
def digitReversal (n : ℕ) : ℕ := sorry

/-- Theorem stating that the difference between a natural number and its digit reversal is divisible by 9 -/
theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, (n : ℤ) - (digitReversal n : ℤ) = 9 * k := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l2050_205088


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2050_205075

theorem missing_fraction_sum (x : ℚ) : 
  1/3 + 1/2 + (-5/6) + 1/5 + 1/4 + (-9/20) + x = 45/100 → x = 27/60 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2050_205075


namespace NUMINAMATH_CALUDE_optimal_price_l2050_205090

/-- Represents the selling price and corresponding daily sales volume -/
structure PriceSales where
  price : ℝ
  sales : ℝ

/-- The cost price of the fruit in yuan per kilogram -/
def costPrice : ℝ := 22

/-- The initial selling price and sales volume -/
def initialSale : PriceSales :=
  { price := 38, sales := 160 }

/-- The change in sales volume per yuan price reduction -/
def salesIncrease : ℝ := 40

/-- The required daily profit in yuan -/
def requiredProfit : ℝ := 3640

/-- Calculates the daily profit given a selling price -/
def calculateProfit (sellingPrice : ℝ) : ℝ :=
  let priceReduction := initialSale.price - sellingPrice
  let salesVolume := initialSale.sales + salesIncrease * priceReduction
  (sellingPrice - costPrice) * salesVolume

/-- The theorem to be proved -/
theorem optimal_price :
  ∃ (optimalPrice : ℝ),
    calculateProfit optimalPrice = requiredProfit ∧
    optimalPrice = 29 ∧
    ∀ (price : ℝ),
      calculateProfit price = requiredProfit →
      price ≥ optimalPrice :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l2050_205090


namespace NUMINAMATH_CALUDE_percentage_study_both_math_and_sociology_l2050_205002

theorem percentage_study_both_math_and_sociology :
  ∀ (S : ℕ) (So Ma Bi MaSo : ℕ),
    S = 200 →
    So = (56 * S) / 100 →
    Ma = (44 * S) / 100 →
    Bi = (40 * S) / 100 →
    Bi - (S - So - Ma + MaSo) ≤ 60 →
    MaSo ≤ Bi - 60 →
    (MaSo * 100) / S = 10 :=
by sorry

end NUMINAMATH_CALUDE_percentage_study_both_math_and_sociology_l2050_205002


namespace NUMINAMATH_CALUDE_C_is_integer_l2050_205086

/-- Represents a number consisting of k ones -/
def ones (k : ℕ) : ℕ :=
  if k = 0 then 0 else 10^(k-1) + ones (k-1)

/-- The factorial-like function [n]! -/
def special_factorial : ℕ → ℕ
  | 0 => 1
  | n+1 => (ones (n+1)) * special_factorial n

/-- The combinatorial-like function C[m, n] -/
def C (m n : ℕ) : ℚ :=
  (special_factorial (m + n)) / ((special_factorial m) * (special_factorial n))

/-- Theorem stating that C[m, n] is always an integer -/
theorem C_is_integer (m n : ℕ) : ∃ k : ℤ, C m n = k :=
  sorry

end NUMINAMATH_CALUDE_C_is_integer_l2050_205086


namespace NUMINAMATH_CALUDE_B_max_at_181_l2050_205060

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : 
  ∀ k : ℕ, k ≤ 2000 → B k ≤ B 181 :=
sorry

end NUMINAMATH_CALUDE_B_max_at_181_l2050_205060


namespace NUMINAMATH_CALUDE_ages_solution_l2050_205007

/-- Represents the ages of Ann, Kristine, and Brad -/
structure Ages where
  ann : ℕ
  kristine : ℕ
  brad : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.ann = ages.kristine + 5 ∧
  ages.brad = ages.ann - 3 ∧
  ages.brad = 2 * ages.kristine

/-- The theorem to be proved -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.kristine = 2 ∧ ages.ann = 7 ∧ ages.brad = 4 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l2050_205007


namespace NUMINAMATH_CALUDE_function_inequality_l2050_205077

open Real

theorem function_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  a * exp x + 2 * x - 1 ≥ (x + a * exp 1) * x := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2050_205077


namespace NUMINAMATH_CALUDE_negation_equivalence_l2050_205094

theorem negation_equivalence :
  (¬ ∀ n : ℕ, 3^n > 500^n) ↔ (∃ n₀ : ℕ, 3^n₀ ≤ 500) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2050_205094


namespace NUMINAMATH_CALUDE_arccos_one_half_l2050_205015

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l2050_205015


namespace NUMINAMATH_CALUDE_no_solution_exists_l2050_205045

theorem no_solution_exists : ¬ ∃ (a b : ℤ), (2006 * 2006) ∣ (a^2006 + b^2006 + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2050_205045


namespace NUMINAMATH_CALUDE_team_score_l2050_205021

/-- Given a basketball team where each person scores 2 points and there are 9 people playing,
    the total points scored by the team is 18. -/
theorem team_score (points_per_person : ℕ) (num_players : ℕ) (total_points : ℕ) :
  points_per_person = 2 →
  num_players = 9 →
  total_points = points_per_person * num_players →
  total_points = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_score_l2050_205021


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l2050_205073

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug : ℕ) (ed_lost : ℕ) (ed_doug_diff : ℕ) :
  ed_initial > doug →
  ed_initial = 91 →
  ed_lost = 21 →
  ed_initial - ed_lost - doug = ed_doug_diff →
  ed_doug_diff = 9 →
  ed_initial - doug = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l2050_205073


namespace NUMINAMATH_CALUDE_stock_price_increase_l2050_205028

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * 0.75 * 1.25 = 1.125 → x = 20 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2050_205028


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l2050_205047

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let d := Real.sqrt (max a b ^ 2 - min a b ^ 2)
  c * d = 20 * Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l2050_205047


namespace NUMINAMATH_CALUDE_system_solutions_l2050_205027

def system (x y z : ℝ) : Prop :=
  x + y + z = 8 ∧ x * y * z = 8 ∧ 1/x - 1/y - 1/z = 1/8

def solution_set : Set (ℝ × ℝ × ℝ) :=
  { (1, (7 + Real.sqrt 17)/2, (7 - Real.sqrt 17)/2),
    (1, (7 - Real.sqrt 17)/2, (7 + Real.sqrt 17)/2),
    (-1, (9 + Real.sqrt 113)/2, (9 - Real.sqrt 113)/2),
    (-1, (9 - Real.sqrt 113)/2, (9 + Real.sqrt 113)/2) }

theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l2050_205027


namespace NUMINAMATH_CALUDE_exists_compound_interest_l2050_205070

/-- Represents the compound interest scenario -/
def compound_interest (P : ℝ) : Prop :=
  let r : ℝ := 0.06  -- annual interest rate
  let n : ℝ := 12    -- number of compounding periods per year
  let t : ℝ := 0.25  -- time in years (3 months)
  let A : ℝ := 1014.08  -- final amount after 3 months
  let two_month_amount : ℝ := P * (1 + r / n) ^ (2 * n * (t / 3))
  A = P * (1 + r / n) ^ (n * t) ∧ 
  (A - two_month_amount) * 100 = 13

/-- Theorem stating the existence of an initial investment satisfying the compound interest scenario -/
theorem exists_compound_interest : ∃ P : ℝ, compound_interest P :=
  sorry

end NUMINAMATH_CALUDE_exists_compound_interest_l2050_205070


namespace NUMINAMATH_CALUDE_necklaces_sold_l2050_205008

theorem necklaces_sold (total : ℕ) (given_away : ℕ) (left : ℕ) (sold : ℕ) : 
  total = 60 → given_away = 18 → left = 26 → sold = total - given_away - left → sold = 16 := by
  sorry

end NUMINAMATH_CALUDE_necklaces_sold_l2050_205008


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2050_205041

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 3 / 4 = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2050_205041


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l2050_205067

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  coins_and_beads : ℝ
  beads : ℝ
  gold_coins : ℝ

/-- The percentage of gold coins in the urn is 36% -/
theorem gold_coins_percentage (urn : UrnComposition) : 
  urn.coins_and_beads / urn.total = 0.75 →
  urn.beads / urn.total = 0.15 →
  urn.gold_coins / (urn.coins_and_beads - urn.beads) = 0.6 →
  urn.gold_coins / urn.total = 0.36 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l2050_205067


namespace NUMINAMATH_CALUDE_bella_position_at_102_l2050_205046

/-- Represents a point on a 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Bella's state at any given point -/
structure BellaState where
  position : Point
  facing : Direction
  lastMove : ℕ

/-- Defines the movement rules for Bella -/
def moveRules (n : ℕ) (state : BellaState) : BellaState :=
  sorry

/-- The main theorem to prove -/
theorem bella_position_at_102 :
  let initialState : BellaState := {
    position := { x := 0, y := 0 },
    facing := Direction.North,
    lastMove := 0
  }
  let finalState := (moveRules 102 initialState)
  finalState.position = { x := -23, y := 29 } :=
sorry

end NUMINAMATH_CALUDE_bella_position_at_102_l2050_205046


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2050_205038

theorem rationalize_denominator : 3 / Real.sqrt 48 = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2050_205038


namespace NUMINAMATH_CALUDE_min_sum_abs_values_l2050_205076

def matrix_condition (a b c d : ℤ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℤ := !![a, b; c, d]
  M ^ 2 = !![5, 0; 0, 5]

theorem min_sum_abs_values (a b c d : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h_matrix : matrix_condition a b c d) :
  (∀ a' b' c' d' : ℤ, a' ≠ 0 → b' ≠ 0 → c' ≠ 0 → d' ≠ 0 → 
    matrix_condition a' b' c' d' → 
    |a| + |b| + |c| + |d| ≤ |a'| + |b'| + |c'| + |d'|) ∧
  |a| + |b| + |c| + |d| = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_abs_values_l2050_205076


namespace NUMINAMATH_CALUDE_lollipop_collection_time_l2050_205064

theorem lollipop_collection_time (total_sticks : ℕ) (visits_per_week : ℕ) (completion_percentage : ℚ) : 
  total_sticks = 400 →
  visits_per_week = 3 →
  completion_percentage = 3/5 →
  (total_sticks * completion_percentage / visits_per_week : ℚ) = 80 := by
sorry

end NUMINAMATH_CALUDE_lollipop_collection_time_l2050_205064


namespace NUMINAMATH_CALUDE_sweater_price_proof_l2050_205063

/-- Price of a T-shirt in dollars -/
def t_shirt_price : ℝ := 8

/-- Price of a jacket before discount in dollars -/
def jacket_price : ℝ := 80

/-- Discount rate for jackets -/
def jacket_discount : ℝ := 0.1

/-- Sales tax rate -/
def sales_tax : ℝ := 0.05

/-- Number of T-shirts purchased -/
def num_tshirts : ℕ := 6

/-- Number of sweaters purchased -/
def num_sweaters : ℕ := 4

/-- Number of jackets purchased -/
def num_jackets : ℕ := 5

/-- Total cost including tax in dollars -/
def total_cost : ℝ := 504

/-- Price of a sweater in dollars -/
def sweater_price : ℝ := 18

theorem sweater_price_proof :
  (num_tshirts * t_shirt_price +
   num_sweaters * sweater_price +
   num_jackets * jacket_price * (1 - jacket_discount)) *
  (1 + sales_tax) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_proof_l2050_205063


namespace NUMINAMATH_CALUDE_bucket_water_calculation_l2050_205049

/-- Given an initial amount of water and an amount poured out, 
    calculate the remaining amount of water in the bucket. -/
def water_remaining (initial : ℝ) (poured_out : ℝ) : ℝ :=
  initial - poured_out

/-- Theorem stating that given 0.8 gallon initially and 0.2 gallon poured out,
    the remaining amount is 0.6 gallon. -/
theorem bucket_water_calculation :
  water_remaining 0.8 0.2 = 0.6 := by
  sorry

#eval water_remaining 0.8 0.2

end NUMINAMATH_CALUDE_bucket_water_calculation_l2050_205049


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_l2050_205084

theorem probability_at_least_one_head (p : ℝ) (h1 : p = 1 / 2) :
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_l2050_205084


namespace NUMINAMATH_CALUDE_possible_a_values_l2050_205081

theorem possible_a_values (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 5, x^2 - 6*x + 2 - a > 0) →
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_possible_a_values_l2050_205081


namespace NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l2050_205083

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ) (d : ℝ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence b)
  (h_equal_1 : a 1 = b 1)
  (h_equal_2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l2050_205083


namespace NUMINAMATH_CALUDE_nanometer_scientific_notation_l2050_205053

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem nanometer_scientific_notation :
  scientific_notation 0.000000022 = (2.2, -8) :=
sorry

end NUMINAMATH_CALUDE_nanometer_scientific_notation_l2050_205053


namespace NUMINAMATH_CALUDE_max_x_value_l2050_205078

theorem max_x_value (x : ℝ) : 
  ((5*x - 20)/(4*x - 5))^2 + (5*x - 20)/(4*x - 5) = 20 → x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l2050_205078


namespace NUMINAMATH_CALUDE_cost_per_sandwich_is_correct_l2050_205017

-- Define the problem parameters
def sandwiches_per_loaf : ℕ := 10
def total_sandwiches : ℕ := 50
def bread_cost : ℚ := 4
def meat_cost : ℚ := 5
def cheese_cost : ℚ := 4
def meat_packs_per_loaf : ℕ := 2
def cheese_packs_per_loaf : ℕ := 2
def cheese_coupon : ℚ := 1
def meat_coupon : ℚ := 1
def discount_threshold : ℚ := 60
def discount_rate : ℚ := 0.1

-- Define the function to calculate the cost per sandwich
def cost_per_sandwich : ℚ :=
  let loaves := total_sandwiches / sandwiches_per_loaf
  let meat_packs := loaves * meat_packs_per_loaf
  let cheese_packs := loaves * cheese_packs_per_loaf
  let total_cost := loaves * bread_cost + meat_packs * meat_cost + cheese_packs * cheese_cost
  let discounted_cost := total_cost - cheese_coupon - meat_coupon
  let final_cost := if discounted_cost > discount_threshold
                    then discounted_cost * (1 - discount_rate)
                    else discounted_cost
  final_cost / total_sandwiches

-- Theorem to prove
theorem cost_per_sandwich_is_correct :
  cost_per_sandwich = 1.944 := by sorry

end NUMINAMATH_CALUDE_cost_per_sandwich_is_correct_l2050_205017


namespace NUMINAMATH_CALUDE_edward_book_purchase_l2050_205035

theorem edward_book_purchase (total_spent : ℝ) (num_books : ℕ) (cost_per_book : ℝ) : 
  total_spent = 6 ∧ num_books = 2 ∧ total_spent = num_books * cost_per_book → cost_per_book = 3 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_purchase_l2050_205035


namespace NUMINAMATH_CALUDE_number_difference_proof_l2050_205004

theorem number_difference_proof (x : ℝ) : x - (3 / 5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l2050_205004


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l2050_205080

/-- Calculates the remaining bottle caps after sharing. -/
def remaining_bottle_caps (start : ℕ) (shared : ℕ) : ℕ :=
  start - shared

/-- Proves that Marilyn ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps : remaining_bottle_caps 51 36 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l2050_205080


namespace NUMINAMATH_CALUDE_certain_number_proof_l2050_205033

theorem certain_number_proof (y x : ℝ) (h1 : y > 0) (h2 : (1/2) * Real.sqrt x = y^(1/3)) (h3 : y = 64) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2050_205033


namespace NUMINAMATH_CALUDE_base_b_problem_l2050_205003

/-- Given that 1325 in base b is equal to the square of 35 in base b, prove that b = 10 in base 10 -/
theorem base_b_problem (b : ℕ) : 
  (3 * b + 5)^2 = b^3 + 3 * b^2 + 2 * b + 5 → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_base_b_problem_l2050_205003


namespace NUMINAMATH_CALUDE_charlie_coins_l2050_205051

/-- The number of coins Alice and Charlie have satisfy the given conditions -/
def satisfy_conditions (a c : ℕ) : Prop :=
  (c + 2 = 5 * (a - 2)) ∧ (c - 2 = 4 * (a + 2))

/-- Charlie has 98 coins given the conditions -/
theorem charlie_coins : ∃ a : ℕ, satisfy_conditions a 98 := by
  sorry

end NUMINAMATH_CALUDE_charlie_coins_l2050_205051


namespace NUMINAMATH_CALUDE_three_times_m_minus_n_squared_correct_l2050_205029

/-- The algebraic expression for "3 times m minus n squared" -/
def three_times_m_minus_n_squared (m n : ℝ) : ℝ := (3*m - n)^2

/-- Theorem stating that the expression correctly represents "3 times m minus n squared" -/
theorem three_times_m_minus_n_squared_correct (m n : ℝ) :
  three_times_m_minus_n_squared m n = (3*m - n)^2 := by sorry

end NUMINAMATH_CALUDE_three_times_m_minus_n_squared_correct_l2050_205029


namespace NUMINAMATH_CALUDE_initial_ace_cards_l2050_205058

/-- Represents the number of cards Nell has --/
structure NellCards where
  initialBaseballCards : ℕ
  finalBaseballCards : ℕ
  finalAceCards : ℕ
  aceBaseballDifference : ℕ

/-- Theorem stating the initial number of Ace cards Nell had --/
theorem initial_ace_cards (n : NellCards) 
  (h1 : n.initialBaseballCards = 239)
  (h2 : n.finalBaseballCards = 111)
  (h3 : n.finalAceCards = 376)
  (h4 : n.aceBaseballDifference = 265)
  (h5 : n.finalAceCards - n.finalBaseballCards = n.aceBaseballDifference) :
  n.finalAceCards + (n.initialBaseballCards - n.finalBaseballCards) = 504 := by
  sorry

end NUMINAMATH_CALUDE_initial_ace_cards_l2050_205058


namespace NUMINAMATH_CALUDE_power_expression_equality_l2050_205052

theorem power_expression_equality : (3^5 / 3^2) * 2^7 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_equality_l2050_205052


namespace NUMINAMATH_CALUDE_jim_travels_two_miles_l2050_205091

/-- The distance John travels in miles -/
def john_distance : ℝ := 15

/-- The difference between John's and Jill's travel distances in miles -/
def distance_difference : ℝ := 5

/-- The percentage of Jill's distance that Jim travels -/
def jim_percentage : ℝ := 0.20

/-- Jill's travel distance in miles -/
def jill_distance : ℝ := john_distance - distance_difference

/-- Jim's travel distance in miles -/
def jim_distance : ℝ := jill_distance * jim_percentage

theorem jim_travels_two_miles :
  jim_distance = 2 := by sorry

end NUMINAMATH_CALUDE_jim_travels_two_miles_l2050_205091


namespace NUMINAMATH_CALUDE_sin_cos_relation_l2050_205026

theorem sin_cos_relation (α β : ℝ) (h : 2 * Real.sin α - Real.cos β = 2) :
  Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l2050_205026


namespace NUMINAMATH_CALUDE_polar_point_equivalence_l2050_205057

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to standard form where r > 0 and 0 ≤ θ < 2π -/
def toStandardForm (p : PolarPoint) : PolarPoint :=
  sorry

theorem polar_point_equivalence :
  let p := PolarPoint.mk (-4) (5 * Real.pi / 6)
  let standardP := toStandardForm p
  standardP.r = 4 ∧ standardP.θ = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_polar_point_equivalence_l2050_205057


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l2050_205031

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 50)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 106 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l2050_205031


namespace NUMINAMATH_CALUDE_apple_slice_packing_l2050_205068

/-- The number of apple slices per group that satisfies the packing conditions -/
def apple_slices_per_group : ℕ := sorry

/-- The number of grapes per group -/
def grapes_per_group : ℕ := 9

/-- The smallest total number of grapes -/
def smallest_total_grapes : ℕ := 18

theorem apple_slice_packing :
  (apple_slices_per_group > 0) ∧
  (apple_slices_per_group * (smallest_total_grapes / grapes_per_group) = smallest_total_grapes) ∧
  (apple_slices_per_group ∣ smallest_total_grapes) ∧
  (grapes_per_group ∣ apple_slices_per_group * grapes_per_group) →
  apple_slices_per_group = 9 := by sorry

end NUMINAMATH_CALUDE_apple_slice_packing_l2050_205068


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_right_isosceles_triangle_l2050_205013

/-- A right triangle with a 45-45-90 degree angle configuration. -/
structure RightIsoscelesTriangle where
  /-- The length of a leg of the triangle -/
  leg : ℝ
  /-- The length of the hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The leg is √2 times the inradius -/
  leg_eq : leg = Real.sqrt 2 * inradius
  /-- The hypotenuse is √2 times the leg -/
  hypotenuse_eq : hypotenuse = Real.sqrt 2 * leg

/-- 
If a right isosceles triangle has an inscribed circle with radius 4,
then its hypotenuse has length 8.
-/
theorem hypotenuse_length_of_right_isosceles_triangle 
  (t : RightIsoscelesTriangle) (h : t.inradius = 4) : 
  t.hypotenuse = 8 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_right_isosceles_triangle_l2050_205013


namespace NUMINAMATH_CALUDE_roots_eq1_roots_eq2_l2050_205056

-- Define the quadratic equations
def eq1 (x : ℝ) := x^2 - 2*x - 8
def eq2 (x : ℝ) := 2*x^2 - 4*x + 1

-- Theorem for the roots of the first equation
theorem roots_eq1 : 
  (eq1 4 = 0 ∧ eq1 (-2) = 0) ∧ 
  ∀ x : ℝ, eq1 x = 0 → x = 4 ∨ x = -2 := by sorry

-- Theorem for the roots of the second equation
theorem roots_eq2 : 
  (eq2 ((2 + Real.sqrt 2) / 2) = 0 ∧ eq2 ((2 - Real.sqrt 2) / 2) = 0) ∧ 
  ∀ x : ℝ, eq2 x = 0 → x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_roots_eq1_roots_eq2_l2050_205056


namespace NUMINAMATH_CALUDE_henry_jill_age_ratio_l2050_205098

/-- Proves that the ratio of Henry's age to Jill's age 11 years ago is 2:1 -/
theorem henry_jill_age_ratio : 
  ∀ (henry_age jill_age : ℕ),
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (k : ℕ), (henry_age - 11) = k * (jill_age - 11) →
  (henry_age - 11) / (jill_age - 11) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_henry_jill_age_ratio_l2050_205098


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_29_div_9_l2050_205089

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determinant of a 3x3 matrix -/
def det3 (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1)

/-- Two lines are coplanar if the determinant of their direction vectors and the vector between their points is zero -/
def areCoplanar (l1 l2 : Line3D) : Prop :=
  let (x1, y1, z1) := l1.point
  let (x2, y2, z2) := l2.point
  let v := (x2 - x1, y2 - y1, z2 - z1)
  det3 l1.direction l2.direction v = 0

/-- The main theorem -/
theorem lines_coplanar_iff_k_eq_neg_29_div_9 :
  let l1 : Line3D := ⟨(3, 2, 4), (2, -1, 3)⟩
  let l2 : Line3D := ⟨(0, 4, 1), (3*k, 1, 2)⟩
  areCoplanar l1 l2 ↔ k = -29/9 := by
  sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_29_div_9_l2050_205089


namespace NUMINAMATH_CALUDE_problem_statement_l2050_205085

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 12) : 5 * (x + 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2050_205085


namespace NUMINAMATH_CALUDE_estimate_two_sqrt_five_l2050_205037

theorem estimate_two_sqrt_five : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_estimate_two_sqrt_five_l2050_205037


namespace NUMINAMATH_CALUDE_divide_and_power_l2050_205001

theorem divide_and_power : (5 / (1 / 5)) ^ 3 = 15625 := by sorry

end NUMINAMATH_CALUDE_divide_and_power_l2050_205001


namespace NUMINAMATH_CALUDE_expression_equality_l2050_205062

theorem expression_equality (x : ℝ) (Q : ℝ) (h : 2 * (5 * x + 3 * Real.sqrt 2) = Q) :
  4 * (10 * x + 6 * Real.sqrt 2) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2050_205062


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l2050_205095

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def pointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Represent a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Calculate the area of a part of the triangle cut by a line -/
def areaPartition (t : Triangle) (l : Line) : ℝ := sorry

theorem triangle_division_theorem (t : Triangle) (P : ℝ × ℝ) (m n : ℝ) 
  (h_point : pointInTriangle P t) (h_positive : m > 0 ∧ n > 0) :
  ∃ (l : Line), 
    pointOnLine P l ∧ 
    areaPartition t l / (triangleArea t - areaPartition t l) = m / n :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l2050_205095


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2050_205050

theorem average_of_six_numbers 
  (total_average : ℝ)
  (second_pair_average : ℝ)
  (third_pair_average : ℝ)
  (h1 : total_average = 6.40)
  (h2 : second_pair_average = 6.1)
  (h3 : third_pair_average = 6.9) :
  ∃ (first_pair_average : ℝ),
    first_pair_average = 6.2 ∧
    (first_pair_average + second_pair_average + third_pair_average) / 3 = total_average :=
by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2050_205050


namespace NUMINAMATH_CALUDE_first_half_time_l2050_205032

/-- Represents the time taken for the elevator to travel down different sections of floors -/
structure ElevatorTime where
  firstHalf : ℕ
  secondQuarter : ℕ
  thirdQuarter : ℕ

/-- The total number of floors the elevator needs to travel -/
def totalFloors : ℕ := 20

/-- The time taken per floor for the second quarter of the journey -/
def timePerFloorSecondQuarter : ℕ := 5

/-- The time taken per floor for the third quarter of the journey -/
def timePerFloorThirdQuarter : ℕ := 16

/-- The total time taken for the elevator to reach the bottom floor -/
def totalTime : ℕ := 120

/-- Calculates the time taken for the second quarter of the journey -/
def secondQuarterTime : ℕ := (totalFloors / 4) * timePerFloorSecondQuarter

/-- Calculates the time taken for the third quarter of the journey -/
def thirdQuarterTime : ℕ := (totalFloors / 4) * timePerFloorThirdQuarter

/-- Theorem stating that the time taken for the first half of the journey is 15 minutes -/
theorem first_half_time (t : ElevatorTime) : 
  t.firstHalf = 15 ∧ 
  t.secondQuarter = secondQuarterTime ∧ 
  t.thirdQuarter = thirdQuarterTime ∧ 
  t.firstHalf + t.secondQuarter + t.thirdQuarter = totalTime := by
  sorry

end NUMINAMATH_CALUDE_first_half_time_l2050_205032


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2050_205042

theorem three_digit_number_problem :
  ∃ (A : ℕ),
    (A ≥ 100 ∧ A < 1000) ∧  -- A is a three-digit number
    (A / 100 ≠ 0 ∧ (A / 10) % 10 ≠ 0 ∧ A % 10 ≠ 0) ∧  -- A does not contain zeroes
    (∃ (B : ℕ),
      (B ≥ 10 ∧ B < 100) ∧  -- B is a two-digit number
      (B = (A / 100 + (A / 10) % 10) * 10 + A % 10) ∧  -- B is formed by summing first two digits of A
      (A = 3 * B)) ∧  -- A = 3B
    A = 135  -- The specific value of A
  := by sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2050_205042


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2050_205099

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2050_205099


namespace NUMINAMATH_CALUDE_perimeter_ratio_of_squares_with_diagonal_ratio_l2050_205011

theorem perimeter_ratio_of_squares_with_diagonal_ratio (d : ℝ) :
  let d1 := d
  let d2 := 4 * d
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let p1 := 4 * s1
  let p2 := 4 * s2
  p2 / p1 = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_ratio_of_squares_with_diagonal_ratio_l2050_205011


namespace NUMINAMATH_CALUDE_dorchester_puppies_washed_l2050_205010

/-- Calculates the number of puppies washed given the daily base pay, per-puppy pay, and total earnings. -/
def puppies_washed (base_pay per_puppy_pay total_earnings : ℚ) : ℚ :=
  (total_earnings - base_pay) / per_puppy_pay

/-- Proves that Dorchester washed 16 puppies given the specified conditions. -/
theorem dorchester_puppies_washed :
  puppies_washed 40 2.25 76 = 16 := by
  sorry

#eval puppies_washed 40 2.25 76

end NUMINAMATH_CALUDE_dorchester_puppies_washed_l2050_205010


namespace NUMINAMATH_CALUDE_lada_elevator_speed_ratio_l2050_205087

/-- The ratio of Lada's original speed to the elevator's speed -/
def speed_ratio : ℚ := 11/4

/-- The number of floors in the first scenario -/
def floors_first : ℕ := 3

/-- The number of floors in the second scenario -/
def floors_second : ℕ := 7

/-- The factor by which Lada increases her speed in the second scenario -/
def speed_increase : ℚ := 2

/-- The factor by which Lada's waiting time increases in the second scenario -/
def wait_time_increase : ℚ := 3

theorem lada_elevator_speed_ratio :
  ∀ (V U : ℚ) (S : ℝ),
  V > 0 → U > 0 → S > 0 →
  (floors_second : ℚ) / (speed_increase * U) - floors_second / V = 
    wait_time_increase * (floors_first / U - floors_first / V) →
  U / V = speed_ratio := by sorry

end NUMINAMATH_CALUDE_lada_elevator_speed_ratio_l2050_205087


namespace NUMINAMATH_CALUDE_new_person_weight_specific_new_person_weight_l2050_205061

/-- Given a group of people, calculate the weight of a new person who causes the average weight to increase when replacing another person. -/
theorem new_person_weight (initial_size : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let total_increase := initial_size * avg_increase
  replaced_weight + total_increase

/-- Prove that for the given conditions, the weight of the new person is 61.3 kg. -/
theorem specific_new_person_weight :
  new_person_weight 12 1.3 45.7 = 61.3 := by sorry

end NUMINAMATH_CALUDE_new_person_weight_specific_new_person_weight_l2050_205061


namespace NUMINAMATH_CALUDE_coin_count_proof_l2050_205020

/-- Represents the total number of coins given the following conditions:
  - There are coins of 20 paise and 25 paise denominations
  - The total value of all coins is 7100 paise (71 Rs)
  - There are 200 coins of 20 paise denomination
-/
def totalCoins (totalValue : ℕ) (value20p : ℕ) (value25p : ℕ) (count20p : ℕ) : ℕ :=
  count20p + (totalValue - count20p * value20p) / value25p

theorem coin_count_proof :
  totalCoins 7100 20 25 200 = 324 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_proof_l2050_205020


namespace NUMINAMATH_CALUDE_log_inequality_for_negative_reals_l2050_205092

theorem log_inequality_for_negative_reals (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  Real.log (-a) > Real.log (-b) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_for_negative_reals_l2050_205092


namespace NUMINAMATH_CALUDE_unique_division_problem_l2050_205054

theorem unique_division_problem :
  ∃! (dividend divisor : ℕ),
    dividend ≥ 1000000 ∧ dividend < 2000000 ∧
    divisor ≥ 300 ∧ divisor < 400 ∧
    (dividend / divisor : ℚ) = 5243 / 1000 ∧
    dividend % divisor = 0 ∧
    ∃ (r1 r2 r3 : ℕ),
      r1 % 10 = 9 ∧
      r2 % 10 = 6 ∧
      r3 % 10 = 3 ∧
      r1 < divisor ∧
      r2 < divisor ∧
      r3 < divisor ∧
      dividend = 1000000 + (dividend / 100000 % 10) * 100000 + 50000 + (dividend % 10000) :=
by sorry

end NUMINAMATH_CALUDE_unique_division_problem_l2050_205054


namespace NUMINAMATH_CALUDE_point_coordinates_l2050_205022

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of a 2D coordinate system -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: Given the conditions, prove that the point P has coordinates (-2, 5) -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 5) 
  (h3 : DistanceToYAxis P = 2) : 
  P.x = -2 ∧ P.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2050_205022


namespace NUMINAMATH_CALUDE_select_shoes_four_pairs_l2050_205006

/-- The number of ways to select 4 shoes from 4 pairs such that no two form a pair -/
def selectShoes (n : ℕ) : ℕ :=
  if n = 4 then 2^4 else 0

theorem select_shoes_four_pairs :
  selectShoes 4 = 16 :=
by sorry

end NUMINAMATH_CALUDE_select_shoes_four_pairs_l2050_205006


namespace NUMINAMATH_CALUDE_three_y_squared_l2050_205074

theorem three_y_squared (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_three_y_squared_l2050_205074


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2050_205040

def Digits := Finset.range 8

theorem min_fraction_sum (A B C D : ℕ) 
  (hA : A ∈ Digits) (hB : B ∈ Digits) (hC : C ∈ Digits) (hD : D ∈ Digits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 28 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2050_205040


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2050_205039

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The surface area of a rectangular solid with dimensions a, b, and c. -/
def surface_area (a b c : ℕ) : ℕ :=
  2 * (a * b + b * c + c * a)

/-- The volume of a rectangular solid with dimensions a, b, and c. -/
def volume (a b c : ℕ) : ℕ :=
  a * b * c

theorem rectangular_solid_surface_area (a b c : ℕ) :
  is_prime a ∧ is_prime b ∧ is_prime c ∧ volume a b c = 308 →
  surface_area a b c = 226 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2050_205039


namespace NUMINAMATH_CALUDE_inverse_proportion_function_l2050_205043

/-- 
If the inverse proportion function y = m/x passes through the point (m, m/8),
then the function can be expressed as y = 8/x.
-/
theorem inverse_proportion_function (m : ℝ) (h : m ≠ 0) : 
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = m / x) ∧ f m = m / 8) → 
  (∃ (g : ℝ → ℝ), ∀ x, x ≠ 0 → g x = 8 / x) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_function_l2050_205043


namespace NUMINAMATH_CALUDE_sector_angle_l2050_205096

/-- Given a sector with area 1 and perimeter 4, its central angle in radians is 2 -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l2050_205096


namespace NUMINAMATH_CALUDE_solve_system_for_p_l2050_205034

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10)
  (eq2 : 5 * p + 2 * q = 20) : 
  p = 80 / 21 := by sorry

end NUMINAMATH_CALUDE_solve_system_for_p_l2050_205034


namespace NUMINAMATH_CALUDE_iron_ball_surface_area_l2050_205069

/-- The surface area of a spherical iron ball that displaces a specific volume of water -/
theorem iron_ball_surface_area (r : ℝ) (h : ℝ) (R : ℝ) : 
  r = 10 → h = 5/3 → (4/3) * Real.pi * R^3 = Real.pi * r^2 * h → 4 * Real.pi * R^2 = 100 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_iron_ball_surface_area_l2050_205069


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2050_205065

/-- An arithmetic sequence with a_2 = 1 and a_5 = 7 has common difference 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 2 = 1)  -- Given: a_2 = 1
  (h2 : a 5 = 7)  -- Given: a_5 = 7
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : a 3 - a 2 = 2 :=  -- Conclusion: The common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2050_205065


namespace NUMINAMATH_CALUDE_root_existence_l2050_205018

theorem root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h₁ : a * x₁^2 + b * x₁ + c = 0)
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) :=
by sorry

end NUMINAMATH_CALUDE_root_existence_l2050_205018


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l2050_205093

/-- The coefficient of x^2 in the expansion of (1+x)+(1+x)^2+(1+x)^3+...+(1+x)^9 -/
def coefficient_x_squared : ℕ :=
  (Finset.range 9).sum (λ n => Nat.choose (n + 1) 2)

/-- The theorem stating that the coefficient of x^2 in the expansion is 120 -/
theorem coefficient_x_squared_is_120 : coefficient_x_squared = 120 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_120_l2050_205093


namespace NUMINAMATH_CALUDE_jerome_toy_cars_l2050_205024

theorem jerome_toy_cars (original : ℕ) : original = 25 :=
  let last_month := 5
  let this_month := 2 * last_month
  let total := 40
  have h : original + last_month + this_month = total := by sorry
  sorry

end NUMINAMATH_CALUDE_jerome_toy_cars_l2050_205024


namespace NUMINAMATH_CALUDE_problem_solution_l2050_205055

theorem problem_solution :
  let A : ℕ := 3009 / 3
  let B : ℕ := A / 3
  let Y : ℕ := A - 2 * B
  Y = 335 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2050_205055


namespace NUMINAMATH_CALUDE_custom_multiplication_l2050_205025

theorem custom_multiplication (a b : ℤ) : a * b = a^2 + a*b - b^2 → 5 * (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_custom_multiplication_l2050_205025


namespace NUMINAMATH_CALUDE_min_sum_given_product_l2050_205072

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → 4 * a + b = a * b → a + b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l2050_205072


namespace NUMINAMATH_CALUDE_pet_store_cages_l2050_205059

def total_cages (initial_puppies initial_adult_dogs initial_kittens : ℕ)
                (sold_puppies sold_adult_dogs sold_kittens : ℕ)
                (puppies_per_cage adult_dogs_per_cage kittens_per_cage : ℕ) : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_adult_dogs := initial_adult_dogs - sold_adult_dogs
  let remaining_kittens := initial_kittens - sold_kittens
  let puppy_cages := (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage
  let adult_dog_cages := (remaining_adult_dogs + adult_dogs_per_cage - 1) / adult_dogs_per_cage
  let kitten_cages := (remaining_kittens + kittens_per_cage - 1) / kittens_per_cage
  puppy_cages + adult_dog_cages + kitten_cages

theorem pet_store_cages : 
  total_cages 45 30 25 39 15 10 3 2 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2050_205059


namespace NUMINAMATH_CALUDE_seventh_term_is_384_l2050_205097

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

/-- The seventh term of the specific geometric sequence -/
def seventhTerm : ℝ :=
  geometricSequenceTerm 6 (-2) 7

theorem seventh_term_is_384 : seventhTerm = 384 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_384_l2050_205097


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2050_205023

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (-2, 3) → b.1 = 3 → b.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2050_205023


namespace NUMINAMATH_CALUDE_minimum_jellybeans_l2050_205000

theorem minimum_jellybeans : ∃ n : ℕ,
  n ≥ 150 ∧
  n % 17 = 15 ∧
  (∀ m : ℕ, m ≥ 150 → m % 17 = 15 → n ≤ m) ∧
  n = 151 :=
by sorry

end NUMINAMATH_CALUDE_minimum_jellybeans_l2050_205000


namespace NUMINAMATH_CALUDE_teacher_volunteers_count_l2050_205030

/-- Calculates the number of teacher volunteers for a school Christmas play. -/
def teacher_volunteers (total_needed : ℕ) (math_classes : ℕ) (students_per_class : ℕ) (more_needed : ℕ) : ℕ :=
  total_needed - (math_classes * students_per_class) - more_needed

/-- Theorem stating that the number of teacher volunteers is 13. -/
theorem teacher_volunteers_count : teacher_volunteers 50 6 5 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_teacher_volunteers_count_l2050_205030


namespace NUMINAMATH_CALUDE_two_students_not_invited_l2050_205079

/-- Represents the social network of students in Mia's class -/
structure ClassNetwork where
  total_students : ℕ
  mia_friends : ℕ
  friends_of_friends : ℕ

/-- Calculates the number of students not invited to Mia's study session -/
def students_not_invited (network : ClassNetwork) : ℕ :=
  network.total_students - (1 + network.mia_friends + network.friends_of_friends)

/-- Theorem stating that 2 students will not be invited to Mia's study session -/
theorem two_students_not_invited (network : ClassNetwork) 
  (h1 : network.total_students = 15)
  (h2 : network.mia_friends = 4)
  (h3 : network.friends_of_friends = 8) : 
  students_not_invited network = 2 := by
  sorry

#eval students_not_invited ⟨15, 4, 8⟩

end NUMINAMATH_CALUDE_two_students_not_invited_l2050_205079


namespace NUMINAMATH_CALUDE_angle_BCA_measure_l2050_205005

-- Define the points
variable (A B C D M O : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define M as the midpoint of AD
def is_midpoint (M A D : EuclideanPlane) : Prop := sorry

-- Define the intersection of BM and AC at O
def intersect_at (B M A C O : EuclideanPlane) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : EuclideanPlane) : ℝ := sorry

-- State the theorem
theorem angle_BCA_measure 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_midpoint : is_midpoint M A D)
  (h_intersect : intersect_at B M A C O)
  (h_ABM : angle_measure A B M = 55)
  (h_AMB : angle_measure A M B = 70)
  (h_BOC : angle_measure B O C = 80)
  (h_ADC : angle_measure A D C = 60) :
  angle_measure B C A = 35 := by sorry

end NUMINAMATH_CALUDE_angle_BCA_measure_l2050_205005


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l2050_205012

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x ∧ x < y → f y < f x

theorem even_decreasing_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_dec : decreasing_nonneg f) :
  ∀ m : ℝ, f (1 - m) < f m ↔ m < (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l2050_205012
