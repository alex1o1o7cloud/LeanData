import Mathlib

namespace NUMINAMATH_CALUDE_smallest_proportional_part_l3294_329496

theorem smallest_proportional_part :
  let total : ℕ := 120
  let ratios : List ℕ := [3, 5, 7]
  let parts : List ℕ := ratios.map (λ r => r * (total / ratios.sum))
  parts.minimum? = some 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l3294_329496


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3294_329412

theorem arithmetic_series_sum (t : ℝ) : 
  let first_term := t^2 + 3
  let num_terms := 3*t + 2
  let common_difference := 1
  let last_term := first_term + (num_terms - 1) * common_difference
  (num_terms / 2) * (first_term + last_term) = (3*t + 2) * (t^2 + 1.5*t + 3.5) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3294_329412


namespace NUMINAMATH_CALUDE_rectangle_area_l3294_329456

/-- The area of a rectangle with length 0.4 meters and width 0.22 meters is 0.088 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 0.4
  let width : ℝ := 0.22
  length * width = 0.088 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3294_329456


namespace NUMINAMATH_CALUDE_profit_decrease_l3294_329441

theorem profit_decrease (march_profit : ℝ) (april_may_decrease : ℝ) : 
  (1 + 0.35) * (1 - april_may_decrease / 100) * (1 + 0.5) = 1.62000000000000014 →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_decrease_l3294_329441


namespace NUMINAMATH_CALUDE_volume_for_weight_less_than_112_l3294_329461

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- The volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ := s.k * weight

theorem volume_for_weight_less_than_112 (s : Substance) (weight : ℝ) 
  (h1 : volume s 112 = 48) (h2 : 0 < weight) (h3 : weight < 112) :
  volume s weight = (48 / 112) * weight := by
sorry

end NUMINAMATH_CALUDE_volume_for_weight_less_than_112_l3294_329461


namespace NUMINAMATH_CALUDE_oil_measurement_l3294_329445

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) :
  initial_oil = 17/100 → added_oil = 67/100 → initial_oil + added_oil = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_oil_measurement_l3294_329445


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3294_329498

/-- The complex number z = (2-i)/i corresponds to a point in the third quadrant -/
theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / i
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3294_329498


namespace NUMINAMATH_CALUDE_equation_solutions_l3294_329463

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0)} = {1, -9, 3, -3} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3294_329463


namespace NUMINAMATH_CALUDE_function_value_at_two_l3294_329449

theorem function_value_at_two (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3294_329449


namespace NUMINAMATH_CALUDE_equation_solution_l3294_329436

theorem equation_solution (x : ℝ) : 
  (x^2 + x - 2)^3 + (2*x^2 - x - 1)^3 = 27*(x^2 - 1)^3 ↔ 
  x = 1 ∨ x = -1 ∨ x = -2 ∨ x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3294_329436


namespace NUMINAMATH_CALUDE_house_sale_price_l3294_329429

theorem house_sale_price (initial_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  initial_price = 100000 ∧ profit_percent = 10 ∧ loss_percent = 10 →
  initial_price * (1 + profit_percent / 100) * (1 - loss_percent / 100) = 99000 := by
sorry

end NUMINAMATH_CALUDE_house_sale_price_l3294_329429


namespace NUMINAMATH_CALUDE_tank_capacity_l3294_329413

theorem tank_capacity : ∀ (initial_fraction final_fraction added_volume capacity : ℚ),
  initial_fraction = 1 / 4 →
  final_fraction = 3 / 4 →
  added_volume = 160 →
  (final_fraction - initial_fraction) * capacity = added_volume →
  capacity = 320 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3294_329413


namespace NUMINAMATH_CALUDE_fifth_term_is_sixteen_l3294_329427

/-- A geometric sequence with first term 1 and a_2 * a_4 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 2 * a 4 = 16

theorem fifth_term_is_sixteen 
  (a : ℕ → ℝ) 
  (h : geometric_sequence a) : 
  a 5 = 16 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_is_sixteen_l3294_329427


namespace NUMINAMATH_CALUDE_find_k_l3294_329403

theorem find_k (k : ℝ) (h : 64 / k = 4) : k = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3294_329403


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l3294_329443

theorem quadratic_roots_theorem (p q : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (p + 3 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (p + 3 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  (q + 6 * Complex.I) ^ 2 - (16 + 9 * Complex.I) * (q + 6 * Complex.I) + (40 + 57 * Complex.I) = 0 →
  p = 9.5 ∧ q = 6.5 := by
sorry


end NUMINAMATH_CALUDE_quadratic_roots_theorem_l3294_329443


namespace NUMINAMATH_CALUDE_pascal_triangle_distinct_elements_l3294_329455

theorem pascal_triangle_distinct_elements :
  ∃ (n : ℕ) (k l m p : ℕ),
    0 < k ∧ k < l ∧ l < m ∧ m < p ∧ p < n ∧
    2 * (n.choose k) = n.choose l ∧
    2 * (n.choose m) = n.choose p ∧
    (n.choose k) ≠ (n.choose l) ∧
    (n.choose k) ≠ (n.choose m) ∧
    (n.choose k) ≠ (n.choose p) ∧
    (n.choose l) ≠ (n.choose m) ∧
    (n.choose l) ≠ (n.choose p) ∧
    (n.choose m) ≠ (n.choose p) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_distinct_elements_l3294_329455


namespace NUMINAMATH_CALUDE_solve_for_P_l3294_329432

theorem solve_for_P : ∃ P : ℝ, (P^4)^(1/3) = 9 * 81^(1/9) → P = 3^(11/6) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_P_l3294_329432


namespace NUMINAMATH_CALUDE_cement_mixture_percentage_l3294_329417

/-- Proves that in a concrete mixture, given specific conditions, the remaining mixture must be 20% cement. -/
theorem cement_mixture_percentage
  (total_concrete : ℝ)
  (final_cement_percentage : ℝ)
  (high_cement_mixture_amount : ℝ)
  (high_cement_percentage : ℝ)
  (h1 : total_concrete = 10)
  (h2 : final_cement_percentage = 0.62)
  (h3 : high_cement_mixture_amount = 7)
  (h4 : high_cement_percentage = 0.8)
  : (total_concrete * final_cement_percentage - high_cement_mixture_amount * high_cement_percentage) / (total_concrete - high_cement_mixture_amount) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_percentage_l3294_329417


namespace NUMINAMATH_CALUDE_joan_lost_balloons_l3294_329425

theorem joan_lost_balloons (initial_balloons current_balloons : ℕ) 
  (h1 : initial_balloons = 9)
  (h2 : current_balloons = 7) : 
  initial_balloons - current_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_joan_lost_balloons_l3294_329425


namespace NUMINAMATH_CALUDE_min_words_to_learn_l3294_329490

/-- Represents the French vocabulary exam setup -/
structure FrenchExam where
  totalWords : ℕ
  guessSuccessRate : ℚ
  targetScore : ℚ

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : FrenchExam) (wordsLearned : ℕ) : ℚ :=
  let correctGuesses := exam.guessSuccessRate * (exam.totalWords - wordsLearned)
  (wordsLearned + correctGuesses) / exam.totalWords

/-- Theorem stating the minimum number of words to learn for the given exam conditions -/
theorem min_words_to_learn (exam : FrenchExam) 
    (h1 : exam.totalWords = 800)
    (h2 : exam.guessSuccessRate = 1/20)
    (h3 : exam.targetScore = 9/10) : 
    ∀ n : ℕ, (∀ m : ℕ, m < n → examScore exam m < exam.targetScore) ∧ 
              examScore exam n ≥ exam.targetScore ↔ n = 716 := by
  sorry

end NUMINAMATH_CALUDE_min_words_to_learn_l3294_329490


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l3294_329486

theorem matrix_inverse_proof :
  let N : Matrix (Fin 3) (Fin 3) ℝ := !![4, 2.5, 0; 3, 2, 0; 0, 0, 1]
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![-4, 5, 0; 6, -8, 0; 0, 0, 1]
  N * A = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l3294_329486


namespace NUMINAMATH_CALUDE_total_students_in_high_school_l3294_329474

/-- Proves that the total number of students in a high school is 500 given the number of students in different course combinations. -/
theorem total_students_in_high_school : 
  ∀ (music art both neither : ℕ),
    music = 30 →
    art = 10 →
    both = 10 →
    neither = 470 →
    music + art - both + neither = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_in_high_school_l3294_329474


namespace NUMINAMATH_CALUDE_number_divided_by_2000_l3294_329492

theorem number_divided_by_2000 : ∃ x : ℝ, x / 2000 = 0.012625 ∧ x = 25.25 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_2000_l3294_329492


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3294_329489

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  (x + (x + 2) + (x + 4) ≥ 51) →  -- sum is at least 51
  (x ≥ 15) ∧  -- x is at least 15
  (∀ y : ℤ, (y % 2 = 1) ∧ (y + (y + 2) + (y + 4) ≥ 51) → y ≥ x) -- x is the smallest such integer
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l3294_329489


namespace NUMINAMATH_CALUDE_total_feathers_is_11638_l3294_329447

/-- The total number of feathers needed for all animals in the circus performance --/
def total_feathers : ℕ :=
  let group1_animals : ℕ := 934
  let group1_feathers_per_crown : ℕ := 7
  let group2_animals : ℕ := 425
  let group2_feathers_per_crown : ℕ := 12
  (group1_animals * group1_feathers_per_crown) + (group2_animals * group2_feathers_per_crown)

/-- Theorem stating that the total number of feathers needed is 11638 --/
theorem total_feathers_is_11638 : total_feathers = 11638 := by
  sorry

end NUMINAMATH_CALUDE_total_feathers_is_11638_l3294_329447


namespace NUMINAMATH_CALUDE_absolute_value_equation_extrema_l3294_329487

theorem absolute_value_equation_extrema :
  ∀ x : ℝ, |x - 3| = 10 → (∃ y : ℝ, |y - 3| = 10 ∧ y ≥ x) ∧ (∃ z : ℝ, |z - 3| = 10 ∧ z ≤ x) ∧
  (∀ w : ℝ, |w - 3| = 10 → w ≤ 13 ∧ w ≥ -7) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_extrema_l3294_329487


namespace NUMINAMATH_CALUDE_ab_negative_necessary_not_sufficient_l3294_329470

-- Define what it means for an equation to represent a hyperbola
def represents_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, a * x^2 + b * y^2 = c ∧ 
  ((a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) ∧ 
  c ≠ 0

-- State the theorem
theorem ab_negative_necessary_not_sufficient :
  (∀ a b c : ℝ, represents_hyperbola a b c → a * b < 0) ∧
  ¬(∀ a b c : ℝ, a * b < 0 → represents_hyperbola a b c) :=
by sorry

end NUMINAMATH_CALUDE_ab_negative_necessary_not_sufficient_l3294_329470


namespace NUMINAMATH_CALUDE_equation_solution_l3294_329457

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 15))) = 54 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3294_329457


namespace NUMINAMATH_CALUDE_fraction_increase_l3294_329479

theorem fraction_increase (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * ((2 * x * y) / (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_increase_l3294_329479


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3294_329472

theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3294_329472


namespace NUMINAMATH_CALUDE_incorrect_roots_correct_roots_l3294_329411

-- Define the original quadratic equation
def original_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Define the roots of the original equation
def is_root (x : ℝ) : Prop := original_eq x

-- Define the pairs of equations
def pair_A (x y : ℝ) : Prop := y = x^2 ∧ y = 3*x - 2
def pair_B (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 0
def pair_C (x y : ℝ) : Prop := y = x ∧ y = Real.sqrt (x + 2)
def pair_D (x y : ℝ) : Prop := y = x^2 - 3*x + 2 ∧ y = 2
def pair_E (x y : ℝ) : Prop := y = Real.sin x ∧ y = 3*x - 4

-- Theorem stating that (C), (D), and (E) do not yield the correct roots
theorem incorrect_roots :
  (∃ x y : ℝ, pair_C x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_D x y ∧ ¬(is_root x)) ∧
  (∃ x y : ℝ, pair_E x y ∧ ¬(is_root x)) :=
sorry

-- Theorem stating that (A) and (B) yield the correct roots
theorem correct_roots :
  (∀ x y : ℝ, pair_A x y → is_root x) ∧
  (∀ x y : ℝ, pair_B x y → is_root x) :=
sorry

end NUMINAMATH_CALUDE_incorrect_roots_correct_roots_l3294_329411


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_seven_l3294_329402

theorem sum_of_a_and_b_is_seven (A B : Set ℕ) (a b : ℕ) : 
  A = {1, 2} →
  B = {2, a, b} →
  A ∪ B = {1, 2, 3, 4} →
  a + b = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_seven_l3294_329402


namespace NUMINAMATH_CALUDE_simplify_cube_root_l3294_329499

theorem simplify_cube_root : 
  (20^3 + 30^3 + 40^3 + 60^3 : ℝ)^(1/3) = 10 * 315^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l3294_329499


namespace NUMINAMATH_CALUDE_workshop_equation_system_l3294_329426

/-- Represents the production capabilities of workers for desks and chairs -/
structure ProductionRate where
  desk : ℕ
  chair : ℕ

/-- Represents the composition of a set of furniture -/
structure FurnitureSet where
  desk : ℕ
  chair : ℕ

/-- The problem setup for the furniture workshop -/
structure WorkshopSetup where
  totalWorkers : ℕ
  productionRate : ProductionRate
  furnitureSet : FurnitureSet

/-- Theorem stating the correct system of equations for the workshop problem -/
theorem workshop_equation_system 
  (setup : WorkshopSetup)
  (h_setup : setup.totalWorkers = 32 ∧ 
             setup.productionRate = { desk := 5, chair := 6 } ∧
             setup.furnitureSet = { desk := 1, chair := 2 }) :
  ∃ (x y : ℕ), 
    x + y = setup.totalWorkers ∧ 
    2 * (setup.productionRate.desk * x) = setup.productionRate.chair * y :=
sorry

end NUMINAMATH_CALUDE_workshop_equation_system_l3294_329426


namespace NUMINAMATH_CALUDE_system_solution_unique_l3294_329421

theorem system_solution_unique (x y : ℝ) : 
  (5 * x + 2 * y = 25 ∧ 3 * x + 4 * y = 15) ↔ (x = 5 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3294_329421


namespace NUMINAMATH_CALUDE_total_limes_is_195_l3294_329460

/-- The number of limes picked by each person -/
def fred_limes : ℕ := 36
def alyssa_limes : ℕ := 32
def nancy_limes : ℕ := 35
def david_limes : ℕ := 42
def eileen_limes : ℕ := 50

/-- The total number of limes picked -/
def total_limes : ℕ := fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes

/-- Theorem stating that the total number of limes picked is 195 -/
theorem total_limes_is_195 : total_limes = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_195_l3294_329460


namespace NUMINAMATH_CALUDE_unique_representation_of_two_over_prime_l3294_329477

theorem unique_representation_of_two_over_prime (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y ∧ x = p * (p + 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_representation_of_two_over_prime_l3294_329477


namespace NUMINAMATH_CALUDE_crude_oil_temperature_l3294_329437

-- Define the function f(x) = x^2 - 7x + 15 on the interval [0, 8]
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) : 
  -- The derivative of f at x = 4 is 1
  (deriv f) 4 = 1 ∧ 
  -- The function is increasing at x = 4
  (deriv f) 4 > 0 := by
  sorry

end NUMINAMATH_CALUDE_crude_oil_temperature_l3294_329437


namespace NUMINAMATH_CALUDE_nancy_carrots_count_l3294_329423

/-- The number of carrots Nancy's mother picked -/
def mother_carrots : ℕ := 47

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy picked -/
def nancy_carrots : ℕ := 38

theorem nancy_carrots_count : 
  nancy_carrots = (good_carrots + bad_carrots) - mother_carrots := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_count_l3294_329423


namespace NUMINAMATH_CALUDE_a_minus_b_equals_four_l3294_329434

theorem a_minus_b_equals_four :
  ∀ (A B : ℕ),
    (A ≥ 10 ∧ A ≤ 99) →  -- A is a two-digit number
    (B ≥ 10 ∧ B ≤ 99) →  -- B is a two-digit number
    A = 23 - 8 →         -- A is 8 less than 23
    B + 7 = 18 →         -- The number that is 7 greater than B is 18
    A - B = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_four_l3294_329434


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_negative_curve_intersection_not_one_l3294_329465

/-- Represents a quadratic equation of the form x^2 + (a-3)x + a = 0 -/
def QuadraticEquation (a : ℝ) := λ x : ℝ => x^2 + (a-3)*x + a

/-- Represents the curve y = |3-x^2| -/
def Curve := λ x : ℝ => |3 - x^2|

theorem quadratic_roots_imply_a_negative (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ QuadraticEquation a x = 0 ∧ QuadraticEquation a y = 0) →
  a < 0 :=
sorry

theorem curve_intersection_not_one (a : ℝ) :
  ¬(∃! x : ℝ, Curve x = a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_negative_curve_intersection_not_one_l3294_329465


namespace NUMINAMATH_CALUDE_total_molecular_weight_eq_1284_07_l3294_329497

/-- Atomic weights in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "O"  => 16.00
  | "H"  => 1.01
  | "Al" => 26.98
  | "S"  => 32.07
  | "K"  => 39.10
  | "N"  => 14.01
  | _    => 0

/-- Molecular weight of Ca(OH)2 in g/mol -/
def mw_calcium_hydroxide : ℝ :=
  atomic_weight "Ca" + 2 * (atomic_weight "O" + atomic_weight "H")

/-- Molecular weight of Al2(SO4)3 in g/mol -/
def mw_aluminum_sulfate : ℝ :=
  2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")

/-- Molecular weight of KNO3 in g/mol -/
def mw_potassium_nitrate : ℝ :=
  atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"

/-- Total molecular weight of the mixture in grams -/
def total_molecular_weight : ℝ :=
  4 * mw_calcium_hydroxide + 2 * mw_aluminum_sulfate + 3 * mw_potassium_nitrate

theorem total_molecular_weight_eq_1284_07 :
  total_molecular_weight = 1284.07 := by
  sorry


end NUMINAMATH_CALUDE_total_molecular_weight_eq_1284_07_l3294_329497


namespace NUMINAMATH_CALUDE_ian_money_left_l3294_329439

def hourly_rate : ℝ := 18
def hours_worked : ℝ := 8
def spending_ratio : ℝ := 0.5

def total_earnings : ℝ := hourly_rate * hours_worked
def amount_spent : ℝ := total_earnings * spending_ratio
def amount_left : ℝ := total_earnings - amount_spent

theorem ian_money_left : amount_left = 72 := by
  sorry

end NUMINAMATH_CALUDE_ian_money_left_l3294_329439


namespace NUMINAMATH_CALUDE_min_balls_to_guarantee_fifteen_l3294_329475

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minBallsToGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem min_balls_to_guarantee_fifteen (counts : BallCounts)
  (h_red : counts.red = 28)
  (h_green : counts.green = 20)
  (h_yellow : counts.yellow = 19)
  (h_blue : counts.blue = 13)
  (h_white : counts.white = 11)
  (h_black : counts.black = 9) :
  minBallsToGuarantee counts 15 = 76 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_to_guarantee_fifteen_l3294_329475


namespace NUMINAMATH_CALUDE_least_integer_greater_than_two_plus_sqrt_three_squared_l3294_329459

theorem least_integer_greater_than_two_plus_sqrt_three_squared :
  ∃ n : ℤ, (n = 14 ∧ (2 + Real.sqrt 3)^2 < n ∧ ∀ m : ℤ, (2 + Real.sqrt 3)^2 < m → n ≤ m) :=
sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_two_plus_sqrt_three_squared_l3294_329459


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_positive_l3294_329469

/-- The function f(x) = a(x^3 - x) is monotonically decreasing on the interval (-√3/3, √3/3) -/
def is_monotone_decreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, -Real.sqrt 3 / 3 < x ∧ x < y ∧ y < Real.sqrt 3 / 3 → f x > f y

/-- The definition of the function f(x) = a(x^3 - x) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x^3 - x)

theorem monotone_decreasing_implies_a_positive (a : ℝ) :
  is_monotone_decreasing (f a) a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_positive_l3294_329469


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3294_329448

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 162 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 9 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3294_329448


namespace NUMINAMATH_CALUDE_female_officers_count_l3294_329401

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 200 →
  female_on_duty_ratio = 1/2 →
  female_ratio = 1/10 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3294_329401


namespace NUMINAMATH_CALUDE_pencils_per_pack_l3294_329410

/-- Given information about Faye's pencils, prove the number of pencils in each pack -/
theorem pencils_per_pack 
  (total_packs : ℕ) 
  (pencils_per_row : ℕ) 
  (total_rows : ℕ) 
  (h1 : total_packs = 28) 
  (h2 : pencils_per_row = 16) 
  (h3 : total_rows = 42) : 
  (total_rows * pencils_per_row) / total_packs = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_pack_l3294_329410


namespace NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l3294_329446

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part (I)
theorem ln_f_greater_than_one :
  ∀ x : ℝ, Real.log (f (-1) x) > 1 := by sorry

-- Theorem for part (II)
theorem max_a_value :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≥ a) ∧
  (∀ b : ℝ, (∀ x : ℝ, f b x ≥ b) → b ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ln_f_greater_than_one_max_a_value_l3294_329446


namespace NUMINAMATH_CALUDE_min_product_sum_l3294_329468

theorem min_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  48 ≤ a * b + b * c + c * d + d * a :=
by sorry

end NUMINAMATH_CALUDE_min_product_sum_l3294_329468


namespace NUMINAMATH_CALUDE_count_odd_numbers_l3294_329458

def digits : Finset Nat := {0, 1, 2, 3, 4}

def is_odd (n : Nat) : Bool := n % 2 = 1

def is_valid_number (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n / 100) ∈ digits ∧ ((n / 10) % 10) ∈ digits ∧ (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧ (n / 100) ≠ (n % 10) ∧ ((n / 10) % 10) ≠ (n % 10)

theorem count_odd_numbers :
  (Finset.filter (λ n => is_valid_number n ∧ is_odd n) (Finset.range 1000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_count_odd_numbers_l3294_329458


namespace NUMINAMATH_CALUDE_ducks_in_marsh_l3294_329407

/-- Given a marsh with geese and ducks, calculate the number of ducks -/
theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) :
  total_birds - geese = 37 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_marsh_l3294_329407


namespace NUMINAMATH_CALUDE_min_value_theorem_l3294_329491

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3294_329491


namespace NUMINAMATH_CALUDE_fish_in_large_aquarium_l3294_329420

def fish_redistribution (initial_fish : ℕ) (additional_fish : ℕ) (small_aquarium_capacity : ℕ) : ℕ :=
  let total_fish := initial_fish + additional_fish
  total_fish - small_aquarium_capacity

theorem fish_in_large_aquarium :
  fish_redistribution 125 250 150 = 225 :=
by sorry

end NUMINAMATH_CALUDE_fish_in_large_aquarium_l3294_329420


namespace NUMINAMATH_CALUDE_negation_of_cosine_inequality_l3294_329473

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos (2 * x) ≤ Real.cos x ^ 2) ↔
  (∃ x : ℝ, Real.cos (2 * x) > Real.cos x ^ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_cosine_inequality_l3294_329473


namespace NUMINAMATH_CALUDE_rectangle_area_l3294_329471

theorem rectangle_area (L W : ℝ) (h1 : L / W = 5 / 3) (h2 : L - 5 = W + 3) : L * W = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3294_329471


namespace NUMINAMATH_CALUDE_z_range_l3294_329406

-- Define the region (as we don't have specific inequalities, we'll use a general set)
variable (R : Set (ℝ × ℝ))

-- Define the function z = x - y
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- State the theorem
theorem z_range (h : Set.Nonempty R) :
  Set.Icc (-1 : ℝ) 2 = {t | ∃ p ∈ R, z p = t} := by sorry

end NUMINAMATH_CALUDE_z_range_l3294_329406


namespace NUMINAMATH_CALUDE_unique_non_six_order_l3294_329484

theorem unique_non_six_order (a : ℤ) : 
  (a > 1 ∧ ∀ p : ℕ, Nat.Prime p → ∀ n : ℕ, n > 0 ∧ a^n ≡ 1 [ZMOD p] → n ≠ 6) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_non_six_order_l3294_329484


namespace NUMINAMATH_CALUDE_complex_pattern_cannot_be_formed_l3294_329454

-- Define the types of shapes
inductive Shape
| Triangle
| Square

-- Define the set of available pieces
def available_pieces : Multiset Shape :=
  Multiset.replicate 8 Shape.Triangle + Multiset.replicate 7 Shape.Square

-- Define the possible figures
inductive Figure
| LargeRectangle
| Triangle
| Square
| ComplexPattern
| LongNarrowRectangle

-- Define a function to check if a figure can be formed
def can_form_figure (pieces : Multiset Shape) (figure : Figure) : Prop :=
  match figure with
  | Figure.LargeRectangle => true
  | Figure.Triangle => true
  | Figure.Square => true
  | Figure.ComplexPattern => false
  | Figure.LongNarrowRectangle => true

-- Theorem statement
theorem complex_pattern_cannot_be_formed :
  ∀ (figure : Figure),
    figure ≠ Figure.ComplexPattern ↔ can_form_figure available_pieces figure :=
by sorry

end NUMINAMATH_CALUDE_complex_pattern_cannot_be_formed_l3294_329454


namespace NUMINAMATH_CALUDE_wage_difference_l3294_329483

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 7.5 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.2

/-- The theorem stating the difference between manager's and chef's wages -/
theorem wage_difference (w : SteakhouseWages) (h : validSteakhouseWages w) :
  w.manager - w.chef = 3 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l3294_329483


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l3294_329435

/-- Converts a natural number from base 2 to base 10 --/
def base2ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a natural number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- The base 2 representation of the number --/
def base2Number : ℕ := 101101100

/-- The expected base 4 representation of the number --/
def expectedBase4Number : ℕ := 23110

theorem base2_to_base4_conversion :
  base10ToBase4 (base2ToBase10 base2Number) = expectedBase4Number := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l3294_329435


namespace NUMINAMATH_CALUDE_final_result_calculation_l3294_329493

theorem final_result_calculation (chosen_number : ℕ) : 
  chosen_number = 60 → (chosen_number * 4 - 138 = 102) := by
  sorry

end NUMINAMATH_CALUDE_final_result_calculation_l3294_329493


namespace NUMINAMATH_CALUDE_event_classification_l3294_329467

-- Define the type for events
inductive Event
| Certain : Event
| Impossible : Event

-- Define a function to classify events
def classify_event (e : Event) : String :=
  match e with
  | Event.Certain => "certain event"
  | Event.Impossible => "impossible event"

-- State the theorem
theorem event_classification :
  (∃ e : Event, e = Event.Certain) ∧ 
  (∃ e : Event, e = Event.Impossible) →
  (classify_event Event.Certain = "certain event") ∧
  (classify_event Event.Impossible = "impossible event") := by
  sorry

end NUMINAMATH_CALUDE_event_classification_l3294_329467


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l3294_329488

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

theorem max_sum_squared_distances (a b c : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hc : is_unit_vector c) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ 9 ∧ 
  ∃ (a' b' c' : E), is_unit_vector a' ∧ is_unit_vector b' ∧ is_unit_vector c' ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖b' - c'‖^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l3294_329488


namespace NUMINAMATH_CALUDE_expression_evaluation_l3294_329476

theorem expression_evaluation : 3^(0^(1^2)) + ((3^0)^2)^1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3294_329476


namespace NUMINAMATH_CALUDE_game_result_depends_only_on_blue_parity_l3294_329418

/-- Represents the color of a sprite -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the game -/
structure GameState :=
  (red : Nat)   -- Number of red sprites
  (blue : Nat)  -- Number of blue sprites

/-- Represents the result of the game -/
def gameResult (initialState : GameState) : Color :=
  if initialState.blue % 2 = 1 then Color.Blue else Color.Red

/-- The main theorem stating that the game result only depends on the initial number of blue sprites -/
theorem game_result_depends_only_on_blue_parity (m n : Nat) :
  gameResult { red := m, blue := n } = 
  if n % 2 = 1 then Color.Blue else Color.Red :=
by sorry

end NUMINAMATH_CALUDE_game_result_depends_only_on_blue_parity_l3294_329418


namespace NUMINAMATH_CALUDE_square_pentagon_alignment_l3294_329428

/-- The number of sides in a square -/
def squareSides : ℕ := 4

/-- The number of sides in a regular pentagon -/
def pentagonSides : ℕ := 5

/-- The least common multiple of the number of sides of a square and a regular pentagon -/
def lcmSides : ℕ := Nat.lcm squareSides pentagonSides

/-- The minimum number of full rotations required for a square to align with a regular pentagon -/
def minRotations : ℕ := lcmSides / squareSides

theorem square_pentagon_alignment :
  minRotations = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_pentagon_alignment_l3294_329428


namespace NUMINAMATH_CALUDE_joshua_shares_with_five_friends_l3294_329494

/-- The number of Skittles Joshua has -/
def total_skittles : ℕ := 40

/-- The number of Skittles each friend receives -/
def skittles_per_friend : ℕ := 8

/-- The number of friends Joshua shares his Skittles with -/
def number_of_friends : ℕ := total_skittles / skittles_per_friend

theorem joshua_shares_with_five_friends :
  number_of_friends = 5 :=
by sorry

end NUMINAMATH_CALUDE_joshua_shares_with_five_friends_l3294_329494


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l3294_329462

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_hcf : Nat.gcd a b = 30) : 
  a * b = 18000 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l3294_329462


namespace NUMINAMATH_CALUDE_snacks_at_dawn_l3294_329451

theorem snacks_at_dawn (S : ℕ) : 
  (3 * S / 5 : ℚ) = 180 → S = 300 := by
  sorry

end NUMINAMATH_CALUDE_snacks_at_dawn_l3294_329451


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_2_100_without_zero_l3294_329431

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ k : ℕ, (n / (10^k)) % 10 = 0

/-- Theorem stating that there exists an integer divisible by 2^100 that does not contain the digit 0 -/
theorem exists_number_divisible_by_2_100_without_zero :
  ∃ n : ℕ, (n % (2^100) = 0) ∧ ¬(containsZero n) := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_2_100_without_zero_l3294_329431


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3294_329416

theorem largest_integer_satisfying_inequality :
  ∃ (x : ℤ), (3 * |2 * x + 1| - 5 < 22) ∧
  (∀ (y : ℤ), y > x → ¬(3 * |2 * y + 1| - 5 < 22)) ∧
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3294_329416


namespace NUMINAMATH_CALUDE_scientific_notation_of_goat_wool_fineness_l3294_329481

theorem scientific_notation_of_goat_wool_fineness :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.000015 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_goat_wool_fineness_l3294_329481


namespace NUMINAMATH_CALUDE_race_course_length_60m_l3294_329464

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  speedB : ℝ     -- Speed of runner B (base speed)
  speedA : ℝ     -- Speed of runner A
  speedC : ℝ     -- Speed of runner C
  headStartA : ℝ -- Head start given by A to B
  headStartC : ℝ -- Head start given by C to B

/-- Calculates the race course length for simultaneous finish -/
def calculateRaceCourseLength (race : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating the race course length for the given scenario -/
theorem race_course_length_60m :
  ∀ (v : ℝ), v > 0 →
  let race : RaceScenario :=
    { speedB := v
      speedA := 4 * v
      speedC := 2 * v
      headStartA := 60
      headStartC := 30 }
  calculateRaceCourseLength race = 60 :=
sorry

end NUMINAMATH_CALUDE_race_course_length_60m_l3294_329464


namespace NUMINAMATH_CALUDE_exists_nonprime_between_primes_l3294_329433

/-- A number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d ∣ n → d = 1 ∨ d = n

/-- There exists a natural number n such that n is not prime, but both n-2 and n+2 are prime. -/
theorem exists_nonprime_between_primes : ∃ n : ℕ, 
  ¬ isPrime n ∧ isPrime (n - 2) ∧ isPrime (n + 2) :=
sorry

end NUMINAMATH_CALUDE_exists_nonprime_between_primes_l3294_329433


namespace NUMINAMATH_CALUDE_president_and_committee_choices_l3294_329442

/-- The number of ways to choose a president and committee from a group --/
def choose_president_and_committee (total_group : ℕ) (senior_members : ℕ) (committee_size : ℕ) : ℕ :=
  let non_senior_members := total_group - senior_members
  let president_choices := non_senior_members
  let remaining_for_committee := total_group - 1
  president_choices * (Nat.choose remaining_for_committee committee_size)

/-- Theorem stating the number of ways to choose a president and committee --/
theorem president_and_committee_choices :
  choose_president_and_committee 10 4 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_choices_l3294_329442


namespace NUMINAMATH_CALUDE_unique_a_value_l3294_329452

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3294_329452


namespace NUMINAMATH_CALUDE_calculate_loss_percentage_l3294_329440

/-- Calculates the percentage of loss given the selling prices and profit percentage --/
theorem calculate_loss_percentage
  (sp_profit : ℝ)       -- Selling price with profit
  (profit_percent : ℝ)  -- Profit percentage
  (sp_loss : ℝ)         -- Selling price with loss
  (h1 : sp_profit = 800)
  (h2 : profit_percent = 25)
  (h3 : sp_loss = 512)
  : (sp_profit * (100 / (100 + profit_percent)) - sp_loss) / 
    (sp_profit * (100 / (100 + profit_percent))) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_loss_percentage_l3294_329440


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3294_329419

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 6 / a 3)^(1/3)) →  -- Geometric sequence property
  a 3 = 2 →                                     -- Given condition
  a 6 = 16 →                                    -- Given condition
  ∃ q : ℝ, (∀ n, a (n + 1) = a n * q) ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3294_329419


namespace NUMINAMATH_CALUDE_additional_barking_dogs_l3294_329405

theorem additional_barking_dogs (initial_dogs final_dogs : ℕ) 
  (h1 : initial_dogs = 30)
  (h2 : final_dogs = 40)
  (h3 : initial_dogs < final_dogs) : 
  final_dogs - initial_dogs = 10 := by
sorry

end NUMINAMATH_CALUDE_additional_barking_dogs_l3294_329405


namespace NUMINAMATH_CALUDE_fence_pole_count_l3294_329480

/-- Calculates the number of fence poles required for a path with bridges -/
def fence_poles (total_length : ℕ) (pole_spacing : ℕ) (bridge_lengths : List ℕ) : ℕ :=
  let fenced_length := total_length - bridge_lengths.sum
  let poles_per_side := fenced_length / pole_spacing
  let total_poles := 2 * poles_per_side + 2
  total_poles

theorem fence_pole_count : 
  fence_poles 2300 8 [48, 58, 62] = 534 := by
  sorry

end NUMINAMATH_CALUDE_fence_pole_count_l3294_329480


namespace NUMINAMATH_CALUDE_number_problem_l3294_329485

theorem number_problem (x : ℝ) : (1/3) * x - 5 = 10 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3294_329485


namespace NUMINAMATH_CALUDE_bella_steps_to_meet_ella_l3294_329478

/-- The distance between Bella's and Ella's houses in feet -/
def total_distance : ℕ := 15840

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- Ella's speed relative to Bella's -/
def speed_ratio : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps_to_meet_ella :
  total_distance * speed_ratio = steps_taken * feet_per_step * (speed_ratio + 1) :=
sorry

end NUMINAMATH_CALUDE_bella_steps_to_meet_ella_l3294_329478


namespace NUMINAMATH_CALUDE_no_common_elements_except_one_l3294_329444

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) + 2 * sequence_a n

def sequence_b : ℕ → ℕ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * sequence_b (n + 1) + 3 * sequence_b n

theorem no_common_elements_except_one :
  ∀ n : ℕ, n > 0 → sequence_a n ≠ sequence_b n :=
by sorry

end NUMINAMATH_CALUDE_no_common_elements_except_one_l3294_329444


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l3294_329409

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 25 →
  capacity_ratio = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets = 63 ∧ 
    (↑new_buckets : ℚ) * capacity_ratio ≥ ↑original_buckets ∧
    (↑new_buckets - 1 : ℚ) * capacity_ratio < ↑original_buckets :=
by sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l3294_329409


namespace NUMINAMATH_CALUDE_trig_problem_l3294_329424

theorem trig_problem (x : ℝ) (h : Real.sin (x + π/6) = 1/3) :
  Real.sin (5*π/6 - x) - (Real.sin (π/3 - x))^2 = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l3294_329424


namespace NUMINAMATH_CALUDE_molecular_weight_AlBr3_10_moles_value_l3294_329422

/-- The molecular weight of 10 moles of AlBr3 -/
def molecular_weight_AlBr3_10_moles : ℝ :=
  let atomic_weight_Al : ℝ := 26.98
  let atomic_weight_Br : ℝ := 79.90
  let molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br
  10 * molecular_weight_AlBr3

/-- Theorem stating that the molecular weight of 10 moles of AlBr3 is 2666.8 grams -/
theorem molecular_weight_AlBr3_10_moles_value :
  molecular_weight_AlBr3_10_moles = 2666.8 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_AlBr3_10_moles_value_l3294_329422


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3294_329495

/-- Hyperbola with given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let l (x : ℝ) := Real.sqrt 3 / 3 * (x + c)
  c^2 = a^2 + b^2 →
  f c ((2 * Real.sqrt 3 * c) / 3) = 1 →
  l (-c) = 0 →
  l 0 = l c / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3294_329495


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3294_329415

theorem geometric_arithmetic_sequence_sum : 
  ∃ (x y : ℝ), 3 < x ∧ x < y ∧ y < 9 ∧ 
  (x^2 = 3*y) ∧ (2*y = x + 9) ∧ 
  (x + y = 11.25) := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l3294_329415


namespace NUMINAMATH_CALUDE_triangle_side_length_l3294_329453

theorem triangle_side_length (AB : ℝ) (cosA sinC : ℝ) (angleADB : ℝ) :
  AB = 30 →
  angleADB = 90 →
  cosA = 4/5 →
  sinC = 2/5 →
  ∃ (AD : ℝ), AD = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3294_329453


namespace NUMINAMATH_CALUDE_solution_a_solution_b_l3294_329430

-- Part (a)
theorem solution_a (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : a ≠ b) :
  let x := (2 * a * b) / (a + b)
  (x + a) / (x - a) + (x + b) / (x - b) = 2 :=
sorry

-- Part (b)
theorem solution_b (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) (h5 : a * b + c ≠ 0) :
  let x := (a * b * c) / d
  c * (d / (a * b) - a * b / x) + d = c^2 / x :=
sorry

end NUMINAMATH_CALUDE_solution_a_solution_b_l3294_329430


namespace NUMINAMATH_CALUDE_zero_score_students_l3294_329404

theorem zero_score_students (total_students : ℕ) (high_scorers : ℕ) (high_score : ℕ) 
  (rest_average : ℚ) (class_average : ℚ) :
  total_students = 28 →
  high_scorers = 4 →
  high_score = 95 →
  rest_average = 45 →
  class_average = 47.32142857142857 →
  ∃ (zero_scorers : ℕ),
    zero_scorers = 3 ∧
    (high_scorers * high_score + zero_scorers * 0 + 
     (total_students - high_scorers - zero_scorers) * rest_average) / total_students = class_average :=
by sorry

end NUMINAMATH_CALUDE_zero_score_students_l3294_329404


namespace NUMINAMATH_CALUDE_pythagorean_triple_value_l3294_329450

theorem pythagorean_triple_value (a : ℝ) : 
  (3 : ℝ)^2 + a^2 = 5^2 → a = 4 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_value_l3294_329450


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l3294_329400

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l3294_329400


namespace NUMINAMATH_CALUDE_sum_product_difference_l3294_329408

theorem sum_product_difference (x y : ℝ) : 
  x + y = 24 → x * y = 23 → |x - y| = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_product_difference_l3294_329408


namespace NUMINAMATH_CALUDE_ice_cream_cones_sold_l3294_329482

theorem ice_cream_cones_sold (milkshakes : ℕ) (difference : ℕ) : 
  milkshakes = 82 → 
  milkshakes = ice_cream_cones + difference → 
  difference = 15 →
  ice_cream_cones = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cones_sold_l3294_329482


namespace NUMINAMATH_CALUDE_pi_approximation_l3294_329466

theorem pi_approximation (S : ℝ) (h : S > 0) :
  4 * S = (1 + 1/4) * (π * S) → π = 3 := by
sorry

end NUMINAMATH_CALUDE_pi_approximation_l3294_329466


namespace NUMINAMATH_CALUDE_safari_leopards_l3294_329414

theorem safari_leopards (total_animals : ℕ) 
  (saturday_lions sunday_buffaloes monday_rhinos : ℕ)
  (saturday_elephants monday_warthogs : ℕ) :
  total_animals = 20 →
  saturday_lions = 3 →
  saturday_elephants = 2 →
  sunday_buffaloes = 2 →
  monday_rhinos = 5 →
  monday_warthogs = 3 →
  ∃ (sunday_leopards : ℕ),
    total_animals = 
      saturday_lions + saturday_elephants + 
      sunday_buffaloes + sunday_leopards +
      monday_rhinos + monday_warthogs ∧
    sunday_leopards = 5 := by
  sorry

end NUMINAMATH_CALUDE_safari_leopards_l3294_329414


namespace NUMINAMATH_CALUDE_inequality_interval_length_l3294_329438

theorem inequality_interval_length (c d : ℝ) : 
  (∃ (x : ℝ), c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) ∧ 
  ((d - 5) / 3 - (c - 5) / 3 = 12) → 
  d - c = 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_interval_length_l3294_329438
