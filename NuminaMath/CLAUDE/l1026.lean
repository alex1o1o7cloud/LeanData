import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1026_102653

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 9 + a 15 + a 21 = 8 →
  a 1 + a 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1026_102653


namespace NUMINAMATH_CALUDE_jonathan_phone_time_l1026_102650

/-- 
Given that Jonathan spends some hours on his phone daily, half of which is spent on social media,
and he spends 28 hours on social media in a week, prove that he spends 8 hours on his phone daily.
-/
theorem jonathan_phone_time (x : ℝ) 
  (daily_phone_time : x > 0) 
  (social_media_half : x / 2 * 7 = 28) : 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_jonathan_phone_time_l1026_102650


namespace NUMINAMATH_CALUDE_horner_method_v3_l1026_102626

def f (x : ℝ) : ℝ := 2*x^5 - x + 3*x^2 + x + 1

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * x + 0
  let v2 := v1 * x - 1
  v2 * x + 3

theorem horner_method_v3 : horner_v3 3 = 54 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1026_102626


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1026_102616

/-- Given a hyperbola with equation x^2 - y^2 = 3, 
    if k₁ and k₂ are the slopes of its two asymptotes, 
    then k₁k₂ = -1 -/
theorem hyperbola_asymptote_slopes (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, x^2 - y^2 = 3 → 
    (∃ a b : ℝ, (y = k₁ * x + a ∨ y = k₂ * x + b) ∧ 
      (∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, 
        |y - k₁ * x| < ε ∨ |y - k₂ * x| < ε))) →
  k₁ * k₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l1026_102616


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1026_102670

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if (1,x) and (-2,3) are perpendicular, then x = 2/3 -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (1, x) (-2, 3) → x = 2/3 := by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1026_102670


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_bound_negation_l1026_102687

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_bound_negation : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_bound_negation_l1026_102687


namespace NUMINAMATH_CALUDE_age_birth_year_problem_l1026_102638

theorem age_birth_year_problem :
  ∃ (age1 age2 : ℕ) (birth_year1 birth_year2 : ℕ),
    age1 > 11 ∧ age2 > 11 ∧
    birth_year1 ≥ 1900 ∧ birth_year1 < 2010 ∧
    birth_year2 ≥ 1900 ∧ birth_year2 < 2010 ∧
    age1 = (birth_year1 / 1000) + ((birth_year1 % 1000) / 100) + ((birth_year1 % 100) / 10) + (birth_year1 % 10) ∧
    age2 = (birth_year2 / 1000) + ((birth_year2 % 1000) / 100) + ((birth_year2 % 100) / 10) + (birth_year2 % 10) ∧
    2010 - birth_year1 = age1 ∧
    2009 - birth_year2 = age2 ∧
    age1 ≠ age2 ∧
    birth_year1 ≠ birth_year2 :=
by sorry

end NUMINAMATH_CALUDE_age_birth_year_problem_l1026_102638


namespace NUMINAMATH_CALUDE_certain_number_proof_l1026_102614

theorem certain_number_proof (x : ℝ) : x + 6 = 8 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1026_102614


namespace NUMINAMATH_CALUDE_proposition_logic_l1026_102673

theorem proposition_logic : 
  let p := (3 : ℝ) ≥ 3
  let q := (3 : ℝ) > 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l1026_102673


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1026_102608

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1026_102608


namespace NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l1026_102662

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → ∀ a : ℝ, x^2 + 2*a*x + 1 ≥ 0) →
  (∃ a_min : ℝ, a_min = -5/4 ∧
    (∀ a : ℝ, (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ a_min) ∧
    (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a_min*x + 1 ≥ 0)) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_quadratic_inequality_l1026_102662


namespace NUMINAMATH_CALUDE_arithmetic_progression_bound_l1026_102676

theorem arithmetic_progression_bound :
  ∃ (C : ℝ), C > 1 ∧
  ∀ (n : ℕ) (a : ℕ → ℕ),
    n > 1 →
    (∀ i j, i < j ∧ j ≤ n → a i < a j) →
    (∃ (d : ℚ), ∀ i j, i ≤ n ∧ j ≤ n → (1 : ℚ) / a i - (1 : ℚ) / a j = d * (i - j)) →
    (a 0 : ℝ) > C^n :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_bound_l1026_102676


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1026_102635

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 + I^2023)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1026_102635


namespace NUMINAMATH_CALUDE_total_absent_students_l1026_102661

def total_students : ℕ := 280

def absent_third_day (total : ℕ) : ℕ := total / 7

def absent_second_day (absent_third : ℕ) : ℕ := 2 * absent_third

def present_first_day (total : ℕ) (absent_second : ℕ) : ℕ := total - absent_second

def absent_first_day (total : ℕ) (present_first : ℕ) : ℕ := total - present_first

theorem total_absent_students :
  let absent_third := absent_third_day total_students
  let absent_second := absent_second_day absent_third
  let present_first := present_first_day total_students absent_second
  let absent_first := absent_first_day total_students present_first
  absent_first + absent_second + absent_third = 200 := by sorry

end NUMINAMATH_CALUDE_total_absent_students_l1026_102661


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_greater_than_two_l1026_102600

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo (-3) a

-- Theorem statement
theorem intersection_equality_implies_a_greater_than_two (a : ℝ) :
  A ∩ B a = A → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_greater_than_two_l1026_102600


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l1026_102636

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) :=
sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l1026_102636


namespace NUMINAMATH_CALUDE_vanya_can_always_win_l1026_102678

/-- Represents a sequence of signs (+1 for "+", -1 for "-") -/
def SignSequence := List Int

/-- Represents a move that swaps two adjacent signs -/
def Move := Nat

/-- Applies a move to a sign sequence -/
def applyMove (seq : SignSequence) (m : Move) : SignSequence :=
  sorry

/-- Evaluates the expression given a sign sequence -/
def evaluateExpression (seq : SignSequence) : Int :=
  sorry

/-- Checks if a number is divisible by 7 -/
def isDivisibleBy7 (n : Int) : Prop :=
  n % 7 = 0

/-- The main theorem: Vanya can always achieve a sum divisible by 7 -/
theorem vanya_can_always_win (initialSeq : SignSequence) :
  ∃ (moves : List Move), isDivisibleBy7 (evaluateExpression (moves.foldl applyMove initialSeq)) :=
sorry

end NUMINAMATH_CALUDE_vanya_can_always_win_l1026_102678


namespace NUMINAMATH_CALUDE_school_boys_count_l1026_102664

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l1026_102664


namespace NUMINAMATH_CALUDE_speeding_motorists_percentage_l1026_102682

theorem speeding_motorists_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_speeders : ℝ)
  (h2 : ticketed_speeders = 0.2 * total_motorists)
  (h3 : ticketed_speeders = 0.8 * (ticketed_speeders + (0.2 * (ticketed_speeders / 0.8))))
  : (ticketed_speeders / 0.8) / total_motorists = 0.25 := by
sorry

end NUMINAMATH_CALUDE_speeding_motorists_percentage_l1026_102682


namespace NUMINAMATH_CALUDE_fraction_comparison_l1026_102663

theorem fraction_comparison : 
  (100 : ℚ) / 101 > 199 / 201 ∧ 199 / 201 > 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1026_102663


namespace NUMINAMATH_CALUDE_window_treatment_cost_l1026_102605

/-- The number of windows Laura needs to buy window treatments for -/
def num_windows : ℕ := 3

/-- The cost of sheers for one window in cents -/
def sheer_cost : ℕ := 4000

/-- The cost of drapes for one window in cents -/
def drape_cost : ℕ := 6000

/-- The total cost for all windows in cents -/
def total_cost : ℕ := 30000

/-- Theorem stating that the number of windows is correct given the costs -/
theorem window_treatment_cost : 
  (sheer_cost + drape_cost) * num_windows = total_cost := by
  sorry


end NUMINAMATH_CALUDE_window_treatment_cost_l1026_102605


namespace NUMINAMATH_CALUDE_smallest_area_increase_l1026_102632

theorem smallest_area_increase (l w : ℕ) (hl : l > 0) (hw : w > 0) :
  ∃ (x : ℕ), x > 0 ∧ (w + 1) * (l - 1) - w * l = x ∧
  ∀ (y : ℕ), y > 0 → (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b + 1) * (a - 1) - b * a = y) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_increase_l1026_102632


namespace NUMINAMATH_CALUDE_tangent_problem_l1026_102681

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (π + α) = -1/3)
  (h2 : Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)) :
  (Real.tan (α + β) = 5/16) ∧ (Real.tan β = 31/43) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l1026_102681


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l1026_102686

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) :
  Prime p ∧ p^(x : ℕ) = y^3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l1026_102686


namespace NUMINAMATH_CALUDE_final_stamp_collection_l1026_102668

def initial_stamps : ℕ := 3000
def mikes_gift : ℕ := 17
def damaged_stamps : ℕ := 37

def harrys_gift (mikes_gift : ℕ) : ℕ := 2 * mikes_gift + 10
def sarahs_gift (mikes_gift : ℕ) : ℕ := 3 * mikes_gift - 5

def total_gift_stamps (mikes_gift : ℕ) : ℕ :=
  mikes_gift + harrys_gift mikes_gift + sarahs_gift mikes_gift

def final_stamp_count (initial_stamps mikes_gift damaged_stamps : ℕ) : ℕ :=
  initial_stamps + total_gift_stamps mikes_gift - damaged_stamps

theorem final_stamp_collection :
  final_stamp_count initial_stamps mikes_gift damaged_stamps = 3070 :=
by sorry

end NUMINAMATH_CALUDE_final_stamp_collection_l1026_102668


namespace NUMINAMATH_CALUDE_function_bound_l1026_102683

-- Define the properties of functions f and g
def satisfies_functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y

def not_identically_zero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

def bounded_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

-- Theorem statement
theorem function_bound (f g : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f g)
  (h2 : not_identically_zero f)
  (h3 : bounded_by_one f) :
  bounded_by_one g :=
sorry

end NUMINAMATH_CALUDE_function_bound_l1026_102683


namespace NUMINAMATH_CALUDE_apples_used_for_lunch_l1026_102651

theorem apples_used_for_lunch (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 38 → bought = 28 → final = 46 → initial - (final - bought) = 20 := by
sorry

end NUMINAMATH_CALUDE_apples_used_for_lunch_l1026_102651


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1026_102667

def P (x : ℝ) : Prop := |x - 2| ≤ 3

def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬(P x)) := by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l1026_102667


namespace NUMINAMATH_CALUDE_slower_train_speed_l1026_102622

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem -/
theorem slower_train_speed (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  faster_speed = 46 →
  passing_time = 72 →
  ∃ (slower_speed : ℝ),
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length ∧
    slower_speed = 36 :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l1026_102622


namespace NUMINAMATH_CALUDE_negation_of_existence_l1026_102631

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x : ℤ), x^2 ≥ x) ↔ (∀ (x : ℤ), x^2 < x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1026_102631


namespace NUMINAMATH_CALUDE_area_increase_is_204_l1026_102674

/-- Represents the increase in vegetables from last year to this year -/
structure VegetableIncrease where
  broccoli : ℕ
  cauliflower : ℕ
  cabbage : ℕ

/-- Calculates the total increase in area given the increase in vegetables -/
def totalAreaIncrease (v : VegetableIncrease) : ℝ :=
  v.broccoli * 1 + v.cauliflower * 2 + v.cabbage * 1.5

/-- The theorem stating that the total increase in area is 204 square feet -/
theorem area_increase_is_204 (v : VegetableIncrease) 
  (h1 : v.broccoli = 79)
  (h2 : v.cauliflower = 25)
  (h3 : v.cabbage = 50) : 
  totalAreaIncrease v = 204 := by
  sorry

#eval totalAreaIncrease { broccoli := 79, cauliflower := 25, cabbage := 50 }

end NUMINAMATH_CALUDE_area_increase_is_204_l1026_102674


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l1026_102649

theorem four_digit_numbers_count : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l1026_102649


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1026_102658

theorem algebraic_expression_value :
  let x : ℤ := -2
  let y : ℤ := -4
  2 * x^2 - y + 3 = 15 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1026_102658


namespace NUMINAMATH_CALUDE_equal_diagonal_distances_l1026_102693

/-- Represents a cuboid with edge lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a pair of diagonals on adjacent faces of a cuboid. -/
inductive DiagonalPair
  | AB_AC
  | AB_BC
  | AC_BC

/-- Calculates the distance between a pair of diagonals on adjacent faces of a cuboid. -/
def diagonalDistance (cuboid : Cuboid) (pair : DiagonalPair) : ℝ :=
  sorry

/-- Theorem stating that the distances between diagonals of each pair of adjacent faces are equal
    for a cuboid with edge lengths 7, 14, and 21. -/
theorem equal_diagonal_distances (cuboid : Cuboid)
    (h1 : cuboid.a = 7)
    (h2 : cuboid.b = 14)
    (h3 : cuboid.c = 21) :
    ∀ p q : DiagonalPair, diagonalDistance cuboid p = diagonalDistance cuboid q :=
  sorry

end NUMINAMATH_CALUDE_equal_diagonal_distances_l1026_102693


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1026_102680

/-- Calculate the minimum number of linear feet of framing needed for an enlarged picture with border -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ⌈(perimeter_inches : ℚ) / 12⌉₊

/-- Theorem stating the minimum number of linear feet of framing needed for the specific picture -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 7 4 3 = 10 := by sorry

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l1026_102680


namespace NUMINAMATH_CALUDE_polynomial_factoring_l1026_102699

theorem polynomial_factoring (a x y : ℝ) : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factoring_l1026_102699


namespace NUMINAMATH_CALUDE_power_of_five_equality_l1026_102637

theorem power_of_five_equality (n : ℕ) : 5^n = 5 * 25^3 * 625^2 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_equality_l1026_102637


namespace NUMINAMATH_CALUDE_B_profit_share_l1026_102606

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_difference_AC : ℕ := 560

theorem B_profit_share :
  let total_investment := investment_A + investment_B + investment_C
  let profit_ratio_A := investment_A / total_investment
  let profit_ratio_B := investment_B / total_investment
  let profit_ratio_C := investment_C / total_investment
  let total_profit := profit_difference_AC * total_investment / (profit_ratio_C - profit_ratio_A)
  profit_ratio_B * total_profit = 1400 := by sorry

end NUMINAMATH_CALUDE_B_profit_share_l1026_102606


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_range_l1026_102659

theorem prime_quadratic_roots_range (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 520*p = 0 ∧ y^2 + p*y - 520*p = 0 ∧ x ≠ y) →
  11 < p ∧ p ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_range_l1026_102659


namespace NUMINAMATH_CALUDE_ellipse_fraction_bounds_l1026_102689

theorem ellipse_fraction_bounds (x y : ℝ) (h : (x - 3)^2 + 4*(y - 1)^2 = 4) :
  ∃ (t : ℝ), (x + y - 3) / (x - y + 1) = t ∧ -1 ≤ t ∧ t ≤ 1 ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 3)^2 + 4*(y₁ - 1)^2 = 4 ∧ (x₁ + y₁ - 3) / (x₁ - y₁ + 1) = -1) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 3)^2 + 4*(y₂ - 1)^2 = 4 ∧ (x₂ + y₂ - 3) / (x₂ - y₂ + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_fraction_bounds_l1026_102689


namespace NUMINAMATH_CALUDE_fractional_parts_inequality_l1026_102610

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), m^3 = q) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (n : ℕ+),
    (nq^(1/3:ℝ) - ⌊nq^(1/3:ℝ)⌋) + (nq^(2/3:ℝ) - ⌊nq^(2/3:ℝ)⌋) ≥ c * n^(-1/2:ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fractional_parts_inequality_l1026_102610


namespace NUMINAMATH_CALUDE_midpoint_sum_invariant_problem_solution_l1026_102688

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  { vertices := sorry }  -- Implementation details omitted

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  List.sum (List.map Prod.fst p.vertices)

theorem midpoint_sum_invariant (p : Polygon) (h : p.vertices.length = 50) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = sumXCoordinates p := by
  sorry

/-- The main theorem that proves the result for the specific case in the problem -/
theorem problem_solution (p : Polygon) (h1 : p.vertices.length = 50) (h2 : sumXCoordinates p = 1005) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = 1005 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_invariant_problem_solution_l1026_102688


namespace NUMINAMATH_CALUDE_marble_distribution_l1026_102685

theorem marble_distribution (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) :
  angela = a ∧ 
  brian = 2 * a ∧ 
  caden = 6 * a ∧ 
  daryl = 42 * a ∧
  angela + brian + caden + daryl = 126 →
  a = 42 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l1026_102685


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l1026_102607

def n : ℕ := 2014

theorem rearrangement_theorem (x y : Fin n → ℤ)
  (hx : ∀ i j, i ≠ j → x i % n ≠ x j % n)
  (hy : ∀ i j, i ≠ j → y i % n ≠ y j % n) :
  ∃ σ : Equiv.Perm (Fin n), ∀ i j, i ≠ j → (x i + y (σ i)) % (2 * n) ≠ (x j + y (σ j)) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l1026_102607


namespace NUMINAMATH_CALUDE_cars_for_sale_l1026_102645

theorem cars_for_sale 
  (num_salespeople : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (num_months : ℕ)
  (h1 : num_salespeople = 10)
  (h2 : cars_per_salesperson_per_month = 10)
  (h3 : num_months = 5) :
  num_salespeople * cars_per_salesperson_per_month * num_months = 500 := by
  sorry

end NUMINAMATH_CALUDE_cars_for_sale_l1026_102645


namespace NUMINAMATH_CALUDE_honey_servings_l1026_102617

/-- Proves that a container with 47 1/3 cups of honey contains 40 12/21 servings when each serving is 1 1/6 cups -/
theorem honey_servings (container : ℚ) (serving : ℚ) :
  container = 47 + 1 / 3 →
  serving = 1 + 1 / 6 →
  container / serving = 40 + 12 / 21 := by
sorry

end NUMINAMATH_CALUDE_honey_servings_l1026_102617


namespace NUMINAMATH_CALUDE_complex_expression_equals_19_l1026_102627

-- Define lg as base 2 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem complex_expression_equals_19 :
  27 ^ (2/3) - 2 ^ (lg 3) * lg (1/8) + 2 * lg (Real.sqrt (3 + Real.sqrt 5) + Real.sqrt (3 - Real.sqrt 5)) = 19 :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_19_l1026_102627


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1026_102629

/-- The curve y = x^2 - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem tangent_parallel_to_x_axis :
  let P : ℝ × ℝ := (3/2, -9/4)
  (f P.1 = P.2) ∧ (f' P.1 = 0) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1026_102629


namespace NUMINAMATH_CALUDE_pump_count_proof_l1026_102603

/-- The number of pumps in the first scenario -/
def num_pumps : ℕ := 3

/-- The number of hours worked per day in the first scenario -/
def hours_per_day_1 : ℕ := 8

/-- The number of days to empty the tank in the first scenario -/
def days_to_empty_1 : ℕ := 2

/-- The number of pumps in the second scenario -/
def num_pumps_2 : ℕ := 8

/-- The number of hours worked per day in the second scenario -/
def hours_per_day_2 : ℕ := 6

/-- The number of days to empty the tank in the second scenario -/
def days_to_empty_2 : ℕ := 1

/-- The capacity of the tank in pump-hours -/
def tank_capacity : ℕ := num_pumps_2 * hours_per_day_2 * days_to_empty_2

theorem pump_count_proof :
  num_pumps * hours_per_day_1 * days_to_empty_1 = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_pump_count_proof_l1026_102603


namespace NUMINAMATH_CALUDE_custom_op_solution_l1026_102615

/-- Custom operation defined for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that given the custom operation, if 11b = 110, then b = 12 -/
theorem custom_op_solution :
  ∀ b : ℤ, customOp 11 b = 110 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l1026_102615


namespace NUMINAMATH_CALUDE_some_number_value_l1026_102654

theorem some_number_value (x : ℝ) : (45 + 23 / x) * x = 4028 → x = 89 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l1026_102654


namespace NUMINAMATH_CALUDE_jordan_terry_income_difference_l1026_102644

/-- Calculates the difference in weekly income between two people given their daily incomes and the number of days worked per week. -/
def weekly_income_difference (terry_daily_income jordan_daily_income days_per_week : ℕ) : ℕ :=
  (jordan_daily_income * days_per_week) - (terry_daily_income * days_per_week)

/-- Proves that the difference in weekly income between Jordan and Terry is $42. -/
theorem jordan_terry_income_difference :
  weekly_income_difference 24 30 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_jordan_terry_income_difference_l1026_102644


namespace NUMINAMATH_CALUDE_four_valid_m_l1026_102628

/-- The number of positive integers m for which 2310 / (m^2 - 4) is a positive integer -/
def count_valid_m : ℕ := 4

/-- Predicate to check if 2310 / (m^2 - 4) is a positive integer -/
def is_valid (m : ℕ) : Prop :=
  m > 0 ∧ ∃ k : ℕ+, k * (m^2 - 4) = 2310

/-- Theorem stating that there are exactly 4 positive integers m satisfying the condition -/
theorem four_valid_m :
  (∃! (s : Finset ℕ), s.card = count_valid_m ∧ ∀ m, m ∈ s ↔ is_valid m) :=
sorry

end NUMINAMATH_CALUDE_four_valid_m_l1026_102628


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1026_102647

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a - 4) + (b + 5)^2 = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1026_102647


namespace NUMINAMATH_CALUDE_max_value_of_f_l1026_102620

def f (x y : ℝ) : ℝ := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 951625 / 256 ∧
  (∀ (x y : ℝ), x + y = 5 → f x y ≤ M) ∧
  (∃ (x y : ℝ), x + y = 5 ∧ f x y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1026_102620


namespace NUMINAMATH_CALUDE_f_as_difference_of_increasing_functions_l1026_102633

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3

theorem f_as_difference_of_increasing_functions :
  ∃ (g h : ℝ → ℝ), 
    (∀ x y, x < y → g x < g y) ∧ 
    (∀ x y, x < y → h x < h y) ∧ 
    (∀ x, f x = g x - h x) :=
sorry

end NUMINAMATH_CALUDE_f_as_difference_of_increasing_functions_l1026_102633


namespace NUMINAMATH_CALUDE_fundraising_solution_correct_l1026_102655

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_price : ℕ
  basketball_qty : ℕ
  soccer_qty : ℕ

/-- Represents the fundraising conditions -/
structure FundraisingConditions where
  original_budget : ℕ
  original_total_items : ℕ
  actual_raised : ℕ
  new_total_items : ℕ

/-- Checks if a purchase satisfies the original plan -/
def satisfies_original_plan (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.original_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty = conditions.original_budget

/-- Checks if a purchase is valid under the new conditions -/
def is_valid_new_purchase (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.new_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty ≤ conditions.actual_raised

/-- Theorem stating the correctness of the solution -/
theorem fundraising_solution_correct 
  (purchase : BallPurchase) 
  (conditions : FundraisingConditions) 
  (h_basketball_price : purchase.basketball_price = 100)
  (h_soccer_price : purchase.soccer_price = 80)
  (h_original_budget : conditions.original_budget = 5600)
  (h_original_total_items : conditions.original_total_items = 60)
  (h_actual_raised : conditions.actual_raised = 6890)
  (h_new_total_items : conditions.new_total_items = 80) :
  (satisfies_original_plan purchase conditions ∧ purchase.basketball_qty = 40 ∧ purchase.soccer_qty = 20) ∧
  (∀ new_purchase : BallPurchase, is_valid_new_purchase new_purchase conditions → new_purchase.basketball_qty ≤ 24) :=
by sorry

end NUMINAMATH_CALUDE_fundraising_solution_correct_l1026_102655


namespace NUMINAMATH_CALUDE_packets_in_box_l1026_102697

/-- The number of packets in a box of sugar substitute -/
def packets_per_box : ℕ := sorry

/-- The daily usage of sugar substitute packets -/
def daily_usage : ℕ := 2

/-- The number of days for which sugar substitute is needed -/
def duration : ℕ := 90

/-- The total cost of sugar substitute for the given duration -/
def total_cost : ℚ := 24

/-- The cost of one box of sugar substitute -/
def cost_per_box : ℚ := 4

/-- Theorem stating that the number of packets in a box is 30 -/
theorem packets_in_box :
  packets_per_box = 30 :=
by sorry

end NUMINAMATH_CALUDE_packets_in_box_l1026_102697


namespace NUMINAMATH_CALUDE_uncle_age_l1026_102611

/-- Given Bud's age and the relationship to his uncle's age, calculate the uncle's age -/
theorem uncle_age (bud_age : ℕ) (h : bud_age = 8) : 
  3 * bud_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_uncle_age_l1026_102611


namespace NUMINAMATH_CALUDE_even_sum_converse_true_l1026_102640

theorem even_sum_converse_true (a b : ℤ) : 
  (∀ (a b : ℤ), Even (a + b) → Even a ∧ Even b) → 
  (Even a ∧ Even b → Even (a + b)) := by sorry

end NUMINAMATH_CALUDE_even_sum_converse_true_l1026_102640


namespace NUMINAMATH_CALUDE_rectangle_perimeter_16_l1026_102695

def rectangle_perimeter (length width : ℚ) : ℚ := 2 * (length + width)

theorem rectangle_perimeter_16 :
  let length : ℚ := 5
  let width : ℚ := 30 / 10
  rectangle_perimeter length width = 16 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_16_l1026_102695


namespace NUMINAMATH_CALUDE_expression_simplification_l1026_102641

theorem expression_simplification (x : ℝ) (h : x = (Real.sqrt 3 - 1) / 3) :
  (2 / (x - 1) + 1 / (x + 1)) * (x^2 - 1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1026_102641


namespace NUMINAMATH_CALUDE_john_total_spend_l1026_102612

-- Define the given quantities
def silver_amount : Real := 1.5
def silver_price_per_ounce : Real := 20
def gold_amount : Real := 2 * silver_amount
def gold_price_per_ounce : Real := 50 * silver_price_per_ounce

-- Define the total cost function
def total_cost : Real :=
  silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

-- Theorem statement
theorem john_total_spend :
  total_cost = 3030 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l1026_102612


namespace NUMINAMATH_CALUDE_tan_negative_23pi_over_6_sin_75_degrees_l1026_102669

-- Part 1
theorem tan_negative_23pi_over_6 : 
  Real.tan (-23 * π / 6) = Real.sqrt 3 / 3 := by sorry

-- Part 2
theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 2 + Real.sqrt 6) / 4 := by sorry

end NUMINAMATH_CALUDE_tan_negative_23pi_over_6_sin_75_degrees_l1026_102669


namespace NUMINAMATH_CALUDE_average_weight_increase_l1026_102621

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)  -- number of people in the group
  (w_old : ℝ)  -- weight of the person being replaced
  (w_new : ℝ)  -- weight of the new person
  (h_n : n = 8)  -- given that there are 8 people
  (h_w_old : w_old = 67)  -- given that the old person weighs 67 kg
  (h_w_new : w_new = 87)  -- given that the new person weighs 87 kg
  : (w_new - w_old) / n = 2.5 := by
sorry


end NUMINAMATH_CALUDE_average_weight_increase_l1026_102621


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1026_102634

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 5) : 
  2 * (a 1) - (a 5) + (a 11) = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1026_102634


namespace NUMINAMATH_CALUDE_freds_marbles_l1026_102672

/-- Given Fred's marble collection, prove the number of dark blue marbles. -/
theorem freds_marbles (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 63 →
  red = 38 →
  green = red / 2 →
  total = red + green + blue →
  blue = 6 := by sorry

end NUMINAMATH_CALUDE_freds_marbles_l1026_102672


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l1026_102679

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → a ≥ b → b ≥ 0 → 
  (Nat.choose (p * a) (p * b)) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l1026_102679


namespace NUMINAMATH_CALUDE_min_lines_same_quadrant_l1026_102665

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- A family of lines in a Cartesian coordinate system --/
def LineFamily := Set Line

/-- The minimum number of lines needed to guarantee at least two lines in the same quadrant --/
def minLinesForSameQuadrant (family : LineFamily) : ℕ := 7

/-- Theorem stating that 7 is the minimum number of lines needed --/
theorem min_lines_same_quadrant (family : LineFamily) :
  minLinesForSameQuadrant family = 7 :=
sorry

end NUMINAMATH_CALUDE_min_lines_same_quadrant_l1026_102665


namespace NUMINAMATH_CALUDE_percent_difference_l1026_102652

theorem percent_difference (w e y z : ℝ) 
  (hw : w = 0.6 * e) 
  (he : e = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  z = 1.5 * w := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l1026_102652


namespace NUMINAMATH_CALUDE_work_left_for_given_days_l1026_102696

/-- The fraction of work left after two workers collaborate for a given time --/
def work_left (a_days b_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / a_days + 1 / b_days)

/-- Theorem: If A can complete the work in 15 days and B in 20 days,
    then after working together for 3 days, the fraction of work left is 13/20 --/
theorem work_left_for_given_days :
  work_left 15 20 3 = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_for_given_days_l1026_102696


namespace NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l1026_102602

theorem square_area_with_five_equal_rectangles (s : ℝ) (x : ℝ) (y : ℝ) : 
  s > 0 →  -- side length of square is positive
  x > 0 →  -- width of central rectangle is positive
  y > 0 →  -- height of bottom rectangle is positive
  s = 5 + 2 * y →  -- relationship between side length and rectangles
  x * (s / 2) = 5 * y →  -- equal area condition
  s^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l1026_102602


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1026_102601

/-- The number of different math books -/
def num_math_books : ℕ := 4

/-- The number of different history books -/
def num_history_books : ℕ := 6

/-- The number of ways to arrange the books under the given conditions -/
def arrangement_count : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 3)

theorem book_arrangement_count :
  arrangement_count = 60480 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1026_102601


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_is_809_l1026_102630

/-- The number of odd numbers in the nth row of the pattern -/
def oddNumbersInRow (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sumOddNumbersInRows (n : ℕ) : ℕ :=
  (oddNumbersInRow n + 1) * n / 2

/-- The nth positive odd number -/
def nthPositiveOdd (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row_is_809 :
  let totalPreviousRows := sumOddNumbersInRows 20
  let positionInSequence := totalPreviousRows + 5
  nthPositiveOdd positionInSequence = 809 := by sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_is_809_l1026_102630


namespace NUMINAMATH_CALUDE_alexander_new_galleries_l1026_102624

/-- Represents the number of pictures Alexander draws for the first gallery -/
def first_gallery_pictures : ℕ := 9

/-- Represents the number of pictures Alexander draws for each new gallery -/
def new_gallery_pictures : ℕ := 2

/-- Represents the number of pencils Alexander needs for each picture -/
def pencils_per_picture : ℕ := 4

/-- Represents the number of pencils Alexander needs for signing at each exhibition -/
def pencils_for_signing : ℕ := 2

/-- Represents the total number of pencils Alexander uses for all exhibitions -/
def total_pencils : ℕ := 88

/-- Calculates the number of new galleries Alexander drew for -/
def new_galleries : ℕ :=
  let first_gallery_pencils := first_gallery_pictures * pencils_per_picture + pencils_for_signing
  let remaining_pencils := total_pencils - first_gallery_pencils
  let pencils_per_new_gallery := new_gallery_pictures * pencils_per_picture + pencils_for_signing
  remaining_pencils / pencils_per_new_gallery

theorem alexander_new_galleries :
  new_galleries = 5 := by sorry

end NUMINAMATH_CALUDE_alexander_new_galleries_l1026_102624


namespace NUMINAMATH_CALUDE_vkontakte_users_l1026_102656

-- Define the people as propositions (being on VKontakte)
variable (M : Prop) -- Marya Ivanovna
variable (I : Prop) -- Ivan Ilyich
variable (A : Prop) -- Alexandra Varfolomeevna
variable (P : Prop) -- Petr Petrovich

-- Define the conditions
def condition1 : Prop := M → (I ∧ A)
def condition2 : Prop := (A ∧ ¬P) ∨ (¬A ∧ P)
def condition3 : Prop := I ∨ M
def condition4 : Prop := I ↔ P

-- Theorem statement
theorem vkontakte_users 
  (h1 : condition1 M I A)
  (h2 : condition2 A P)
  (h3 : condition3 I M)
  (h4 : condition4 I P) :
  I ∧ P ∧ ¬M ∧ ¬A :=
sorry

end NUMINAMATH_CALUDE_vkontakte_users_l1026_102656


namespace NUMINAMATH_CALUDE_book_price_l1026_102666

def original_price : ℝ → Prop :=
  fun price =>
    let first_discount := price * (1 - 1/5)
    let second_discount := first_discount * (1 - 1/5)
    second_discount = 32

theorem book_price : original_price 50 := by
  sorry

end NUMINAMATH_CALUDE_book_price_l1026_102666


namespace NUMINAMATH_CALUDE_not_necessary_condition_l1026_102618

theorem not_necessary_condition : ¬(∀ x y : ℝ, x * y = 0 → x^2 + y^2 = 0) := by sorry

end NUMINAMATH_CALUDE_not_necessary_condition_l1026_102618


namespace NUMINAMATH_CALUDE_points_per_vegetable_l1026_102619

/-- Proves that the number of points given for each vegetable eaten is 2 --/
theorem points_per_vegetable (total_points : ℕ) (num_students : ℕ) (num_weeks : ℕ) (veggies_per_week : ℕ)
  (h1 : total_points = 200)
  (h2 : num_students = 25)
  (h3 : num_weeks = 2)
  (h4 : veggies_per_week = 2) :
  total_points / (num_students * num_weeks * veggies_per_week) = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_per_vegetable_l1026_102619


namespace NUMINAMATH_CALUDE_cuboid_circumscribed_sphere_area_l1026_102657

theorem cuboid_circumscribed_sphere_area (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x * y = Real.sqrt 6 ∧ 
  y * z = Real.sqrt 2 ∧ 
  z * x = Real.sqrt 3 → 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cuboid_circumscribed_sphere_area_l1026_102657


namespace NUMINAMATH_CALUDE_mean_problem_l1026_102684

theorem mean_problem (x y : ℝ) : 
  (28 + x + 42 + y + 78 + 104) / 6 = 62 → 
  x + y = 120 ∧ (x + y) / 2 = 60 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l1026_102684


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l1026_102698

def nth_odd_multiple_of_five (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_five : 
  nth_odd_multiple_of_five 15 = 145 := by sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_five_l1026_102698


namespace NUMINAMATH_CALUDE_circle_equation_from_chord_l1026_102660

/-- Given a circle with center at the origin and a chord of length 8 cut by the line 3x + 4y + 15 = 0,
    prove that the equation of the circle is x^2 + y^2 = 25 -/
theorem circle_equation_from_chord (x y : ℝ) :
  let center := (0 : ℝ × ℝ)
  let chord_line := {(x, y) | 3 * x + 4 * y + 15 = 0}
  let chord_length := 8
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), p ∈ chord_line → dist center p ≤ r) ∧
    (∃ (p q : ℝ × ℝ), p ∈ chord_line ∧ q ∈ chord_line ∧ p ≠ q ∧ dist p q = chord_length) →
  x^2 + y^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_chord_l1026_102660


namespace NUMINAMATH_CALUDE_sum_of_powers_l1026_102639

theorem sum_of_powers (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a^1 + a^2 + a^3 + a^4 + a^5 + a^6 + a^7 + a^8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1026_102639


namespace NUMINAMATH_CALUDE_f_difference_at_5_and_neg_5_l1026_102671

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_5_and_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_5_and_neg_5_l1026_102671


namespace NUMINAMATH_CALUDE_variance_estimation_l1026_102675

/-- Represents the data for a group of students -/
structure GroupData where
  count : ℕ
  average_score : ℝ
  variance : ℝ

/-- Calculates the estimated variance of test scores given two groups of students -/
def estimated_variance (male : GroupData) (female : GroupData) : ℝ :=
  let total_count := male.count + female.count
  let male_weight := male.count / total_count
  let female_weight := female.count / total_count
  let overall_average := male_weight * male.average_score + female_weight * female.average_score
  male_weight * (male.variance + (overall_average - male.average_score)^2) +
  female_weight * (female.variance + (female.average_score - overall_average)^2)

theorem variance_estimation (male : GroupData) (female : GroupData) :
  male.count = 400 →
  female.count = 600 →
  male.average_score = 80 →
  male.variance = 10 →
  female.average_score = 60 →
  female.variance = 20 →
  estimated_variance male female = 112 := by
  sorry

end NUMINAMATH_CALUDE_variance_estimation_l1026_102675


namespace NUMINAMATH_CALUDE_solution_characterization_l1026_102625

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0),
   (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
   (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
   (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
   (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2)}

def satisfies_equations (p : ℝ × ℝ) : Prop :=
  let x := p.1
  let y := p.2
  x = 3 * x^2 * y - y^3 ∧ y = x^3 - 3 * x * y^2

theorem solution_characterization :
  ∀ p : ℝ × ℝ, satisfies_equations p ↔ p ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_solution_characterization_l1026_102625


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1026_102677

theorem fraction_to_decimal : (7 : ℚ) / 50 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1026_102677


namespace NUMINAMATH_CALUDE_percentage_of_girls_l1026_102694

theorem percentage_of_girls (boys girls : ℕ) (h1 : boys = 300) (h2 : girls = 450) :
  (girls : ℚ) / ((boys : ℚ) + (girls : ℚ)) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l1026_102694


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l1026_102604

/-- Given a train of length 1400 m that crosses a tree in 100 sec,
    prove that it takes 150 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1400)
  (h2 : tree_crossing_time = 100)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l1026_102604


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1026_102691

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  ((2 * x - 2) / x - 1) / ((x^2 - 4*x + 4) / (x^2 - x)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1026_102691


namespace NUMINAMATH_CALUDE_total_pizzas_ordered_l1026_102623

/-- Represents the number of pizzas ordered for a group of students. -/
def pizzas_ordered (num_boys : ℕ) (num_girls : ℕ) : ℚ :=
  22 + (22 / num_boys) * (num_girls / 2)

/-- Theorem stating the total number of pizzas ordered is 33. -/
theorem total_pizzas_ordered :
  ∃ (num_boys : ℕ),
    num_boys > 13 ∧
    pizzas_ordered num_boys 13 = 33 ∧
    (∃ (n : ℕ), pizzas_ordered num_boys 13 = n) :=
sorry

end NUMINAMATH_CALUDE_total_pizzas_ordered_l1026_102623


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l1026_102692

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l1026_102692


namespace NUMINAMATH_CALUDE_difference_of_squares_l1026_102642

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1026_102642


namespace NUMINAMATH_CALUDE_polynomial_sum_l1026_102648

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) :
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1026_102648


namespace NUMINAMATH_CALUDE_cat_relocation_proportion_l1026_102613

/-- Calculates the proportion of cats relocated in the second mission -/
def proportion_relocated (initial_cats : ℕ) (first_mission_relocated : ℕ) (final_remaining : ℕ) : ℚ :=
  let remaining_after_first := initial_cats - first_mission_relocated
  let relocated_second := remaining_after_first - final_remaining
  relocated_second / remaining_after_first

theorem cat_relocation_proportion :
  proportion_relocated 1800 600 600 = 1/2 := by
  sorry

#eval proportion_relocated 1800 600 600

end NUMINAMATH_CALUDE_cat_relocation_proportion_l1026_102613


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l1026_102646

theorem solution_set_x_squared_minus_one (x : ℝ) : 
  {x : ℝ | x^2 - 1 = 0} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_l1026_102646


namespace NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1026_102690

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_are_congruent_l1026_102690


namespace NUMINAMATH_CALUDE_min_value_expression_l1026_102643

theorem min_value_expression (x y : ℝ) (h : x ≥ 4) :
  x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1026_102643


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1026_102609

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1026_102609
