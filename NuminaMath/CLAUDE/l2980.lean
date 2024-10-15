import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2980_298087

/-- A hyperbola with focus and asymptotes -/
structure Hyperbola where
  /-- The right focus of the hyperbola -/
  focus : ℝ × ℝ
  /-- The asymptotes of the hyperbola, represented as slopes -/
  asymptotes : ℝ × ℝ

/-- The symmetric point of a point with respect to a line -/
def symmetricPoint (p : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ := sorry

/-- Check if a point lies on a line given by its slope -/
def liesOn (p : ℝ × ℝ) (slope : ℝ) : Prop := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: If the symmetric point of the focus with respect to one asymptote
    lies on the other asymptote, then the eccentricity is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) :
  let (slope1, slope2) := h.asymptotes
  let symPoint := symmetricPoint h.focus slope1
  liesOn symPoint slope2 → eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2980_298087


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2980_298043

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 10 < 0 ↔ -2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2980_298043


namespace NUMINAMATH_CALUDE_fifteenth_valid_number_l2980_298004

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 14

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem fifteenth_valid_number :
  nth_valid_number 15 = 266 := by sorry

end NUMINAMATH_CALUDE_fifteenth_valid_number_l2980_298004


namespace NUMINAMATH_CALUDE_smallest_fraction_l2980_298018

theorem smallest_fraction :
  let f1 := 5 / 12
  let f2 := 7 / 17
  let f3 := 20 / 41
  let f4 := 125 / 252
  let f5 := 155 / 312
  f2 ≤ f1 ∧ f2 ≤ f3 ∧ f2 ≤ f4 ∧ f2 ≤ f5 :=
by
  sorry

#check smallest_fraction

end NUMINAMATH_CALUDE_smallest_fraction_l2980_298018


namespace NUMINAMATH_CALUDE_total_apples_collected_l2980_298054

-- Define the number of green apples
def green_apples : ℕ := 124

-- Define the number of red apples in terms of green apples
def red_apples : ℕ := 3 * green_apples

-- Define the total number of apples
def total_apples : ℕ := red_apples + green_apples

-- Theorem to prove
theorem total_apples_collected : total_apples = 496 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_collected_l2980_298054


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2980_298072

/-- The common ratio of a geometric sequence starting with 10, -20, 40, -80 is -2 -/
theorem geometric_sequence_ratio : ∀ (a : ℕ → ℤ), 
  a 0 = 10 ∧ a 1 = -20 ∧ a 2 = 40 ∧ a 3 = -80 → 
  (∀ n : ℕ, a (n + 1) = a n * (-2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2980_298072


namespace NUMINAMATH_CALUDE_jeffrey_steps_l2980_298038

/-- Represents Jeffrey's walking pattern --/
structure WalkingPattern where
  forward : Nat
  backward : Nat

/-- Calculates the effective steps for a given pattern and number of repetitions --/
def effectiveSteps (pattern : WalkingPattern) (repetitions : Nat) : Nat :=
  (pattern.forward - pattern.backward) * repetitions

/-- Calculates the total steps taken for a given pattern and number of repetitions --/
def totalSteps (pattern : WalkingPattern) (repetitions : Nat) : Nat :=
  (pattern.forward + pattern.backward) * repetitions

/-- Theorem stating the total number of steps Jeffrey takes --/
theorem jeffrey_steps :
  let initialPattern : WalkingPattern := ⟨3, 2⟩
  let changedPattern : WalkingPattern := ⟨4, 1⟩
  let totalDistance : Nat := 66
  let initialEffectiveSteps : Nat := 30
  let initialRepetitions : Nat := initialEffectiveSteps / (initialPattern.forward - initialPattern.backward)
  let remainingDistance : Nat := totalDistance - initialEffectiveSteps
  let changedRepetitions : Nat := remainingDistance / (changedPattern.forward - changedPattern.backward)
  totalSteps initialPattern initialRepetitions + totalSteps changedPattern changedRepetitions = 210 := by
  sorry

end NUMINAMATH_CALUDE_jeffrey_steps_l2980_298038


namespace NUMINAMATH_CALUDE_bo_flashcard_knowledge_percentage_l2980_298055

theorem bo_flashcard_knowledge_percentage :
  let total_flashcards : ℕ := 800
  let days_to_learn : ℕ := 40
  let words_per_day : ℕ := 16
  let total_words_to_learn : ℕ := days_to_learn * words_per_day
  let words_already_known : ℕ := total_flashcards - total_words_to_learn
  let percentage_known : ℚ := (words_already_known : ℚ) / (total_flashcards : ℚ) * 100
  percentage_known = 20 := by
sorry

end NUMINAMATH_CALUDE_bo_flashcard_knowledge_percentage_l2980_298055


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l2980_298013

theorem greatest_three_digit_divisible_by_3_6_5 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 ∣ n ∧ 6 ∣ n ∧ 5 ∣ n → n ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l2980_298013


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2980_298041

/-- The dividend polynomial -/
def P (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - x - 30

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x^2 - 1

/-- The remainder polynomial -/
def R (x : ℝ) : ℝ := -13*x - 22

theorem polynomial_division_remainder :
  ∃ (Q : ℝ → ℝ), ∀ x, P x = D x * Q x + R x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2980_298041


namespace NUMINAMATH_CALUDE_r_value_when_n_is_2_l2980_298075

theorem r_value_when_n_is_2 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 2^n + 1) 
  (h2 : r = 3^s - s) 
  (h3 : n = 2) : 
  r = 238 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_2_l2980_298075


namespace NUMINAMATH_CALUDE_min_workers_for_profit_l2980_298029

/-- Represents the problem of finding the minimum number of workers needed for profit --/
theorem min_workers_for_profit :
  let maintenance_fee : ℝ := 470
  let hourly_wage : ℝ := 10
  let widgets_per_hour : ℝ := 6
  let widget_price : ℝ := 3.5
  let work_hours : ℝ := 8
  let min_workers : ℕ := 6

  ∀ n : ℕ, n ≥ min_workers →
    work_hours * widgets_per_hour * widget_price * n > maintenance_fee + work_hours * hourly_wage * n ∧
    ∀ m : ℕ, m < min_workers →
      work_hours * widgets_per_hour * widget_price * m ≤ maintenance_fee + work_hours * hourly_wage * m :=
by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_profit_l2980_298029


namespace NUMINAMATH_CALUDE_cat_relocation_l2980_298056

theorem cat_relocation (initial_cats : ℕ) (first_removal : ℕ) : 
  initial_cats = 1800 →
  first_removal = 600 →
  initial_cats - first_removal - (initial_cats - first_removal) / 2 = 600 := by
sorry

end NUMINAMATH_CALUDE_cat_relocation_l2980_298056


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l2980_298023

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := total_trees - birch_trees

theorem birch_tree_arrangement_probability :
  let favorable_arrangements := (non_birch_trees + 1).choose birch_trees
  let total_arrangements := total_trees.choose birch_trees
  (favorable_arrangements : ℚ) / total_arrangements = 6 / 143 := by
sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l2980_298023


namespace NUMINAMATH_CALUDE_hypotenuse_length_is_double_short_leg_l2980_298033

/-- A right triangle with a 30-60-90 degree angle configuration -/
structure RightTriangle30_60_90 where
  -- The length of the side opposite to the 30° angle
  short_leg : ℝ
  -- Assertion that the short leg is positive
  short_leg_pos : short_leg > 0

/-- The length of the hypotenuse in a 30-60-90 right triangle -/
def hypotenuse_length (t : RightTriangle30_60_90) : ℝ :=
  2 * t.short_leg

/-- Theorem: In a 30-60-90 right triangle with short leg of length 5,
    the hypotenuse has length 10 -/
theorem hypotenuse_length_is_double_short_leg :
  let t : RightTriangle30_60_90 := ⟨5, by norm_num⟩
  hypotenuse_length t = 10 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_is_double_short_leg_l2980_298033


namespace NUMINAMATH_CALUDE_bicycle_count_l2980_298040

/-- Represents the number of acrobats, elephants, and bicycles in a parade. -/
structure ParadeCount where
  acrobats : ℕ
  elephants : ℕ
  bicycles : ℕ

/-- Checks if the given parade count satisfies the conditions of the problem. -/
def isValidParadeCount (count : ParadeCount) : Prop :=
  count.acrobats + count.elephants = 25 ∧
  2 * count.acrobats + 4 * count.elephants + 2 * count.bicycles = 68

/-- Theorem stating that there are 9 bicycles in the parade. -/
theorem bicycle_count : ∃ (count : ParadeCount), isValidParadeCount count ∧ count.bicycles = 9 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_count_l2980_298040


namespace NUMINAMATH_CALUDE_max_carps_eaten_l2980_298000

/-- Represents the eating behavior of pikes in a pond -/
structure PikePond where
  initialPikes : ℕ
  pikesForFull : ℕ
  carpPerFull : ℕ

/-- Calculates the maximum number of full pikes -/
def maxFullPikes (pond : PikePond) : ℕ :=
  (pond.initialPikes - 1) / pond.pikesForFull

/-- Theorem: The maximum number of crucian carps eaten is 9 given the initial conditions -/
theorem max_carps_eaten (pond : PikePond) 
  (h1 : pond.initialPikes = 30)
  (h2 : pond.pikesForFull = 3)
  (h3 : pond.carpPerFull = 1) : 
  maxFullPikes pond * pond.carpPerFull = 9 := by
  sorry

#eval maxFullPikes { initialPikes := 30, pikesForFull := 3, carpPerFull := 1 }

end NUMINAMATH_CALUDE_max_carps_eaten_l2980_298000


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_greater_than_two_l2980_298039

theorem sufficient_condition_implies_a_greater_than_two (a : ℝ) :
  (∀ x, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) ∧
  (∃ x, ((a + x) * (1 + x) < 0) ∧ (x ≤ -2 ∨ x ≥ -1))
  → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_greater_than_two_l2980_298039


namespace NUMINAMATH_CALUDE_point_on_line_l2980_298049

theorem point_on_line (m n : ℝ) :
  let line := fun (x y : ℝ) => x = y / 2 - 2 / 5
  let some_value := 2
  (line m n ∧ line (m + some_value) (n + 4)) →
  some_value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2980_298049


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l2980_298003

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that missing both times is the complement of hitting at least once
theorem complement_of_hit_at_least_once :
  ∀ ω : Ω, ¬(hit_at_least_once ω) ↔ miss_both_times ω :=
by sorry

end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l2980_298003


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2980_298070

theorem complex_division_simplification :
  let z₁ : ℂ := 5 + 3 * I
  let z₂ : ℂ := 2 + I
  z₁ / z₂ = 13/5 + (1/5) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2980_298070


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l2980_298032

theorem solution_set_of_equation (x : ℝ) : 
  {x | 3 * x - 4 = 2} = {2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l2980_298032


namespace NUMINAMATH_CALUDE_sum_of_composite_function_at_specific_points_l2980_298025

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := |x + 1| - 3

def x_values : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_composite_function_at_specific_points :
  (x_values.map (λ x => q (p x))).sum = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_at_specific_points_l2980_298025


namespace NUMINAMATH_CALUDE_vector_magnitude_cos_sin_l2980_298045

theorem vector_magnitude_cos_sin (x : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.cos x, Real.sin x]
  ‖a‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_cos_sin_l2980_298045


namespace NUMINAMATH_CALUDE_sqrt_expressions_l2980_298015

theorem sqrt_expressions (x y : ℝ) 
  (hx : x = Real.sqrt 3 + Real.sqrt 2) 
  (hy : y = Real.sqrt 3 - Real.sqrt 2) : 
  x^2 + 2*x*y + y^2 = 12 ∧ 1/y - 1/x = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expressions_l2980_298015


namespace NUMINAMATH_CALUDE_expression_simplification_l2980_298092

theorem expression_simplification :
  (0.2 * 0.4 - 0.3 / 0.5) + (0.6 * 0.8 + 0.1 / 0.2) - 0.9 * (0.3 - 0.2 * 0.4) = 0.262 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2980_298092


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2980_298034

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 9 * x ≤ 15) ↔ ((3 - Real.sqrt 29) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 29) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2980_298034


namespace NUMINAMATH_CALUDE_cars_rented_at_3600_optimal_rent_max_revenue_l2980_298083

/-- Represents the rental company's car fleet and pricing model -/
structure RentalCompany where
  totalCars : ℕ
  baseRent : ℕ
  rentIncrement : ℕ
  maintenanceCost : ℕ
  decreaseRate : ℕ

/-- Calculates the number of cars rented at a given rent -/
def carsRented (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.totalCars - (rent - company.baseRent) / company.rentIncrement

/-- Calculates the revenue at a given rent -/
def revenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  (carsRented company rent) * (rent - company.maintenanceCost)

/-- Theorem stating the correct number of cars rented at 3600 yuan -/
theorem cars_rented_at_3600 (company : RentalCompany) 
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  carsRented company 3600 = 88 := by sorry

/-- Theorem stating the rent that maximizes revenue -/
theorem optimal_rent (company : RentalCompany)
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  ∃ (r : ℕ), ∀ (rent : ℕ), revenue company rent ≤ revenue company r ∧ r = 4100 := by sorry

/-- Theorem stating the maximum revenue -/
theorem max_revenue (company : RentalCompany)
  (h1 : company.totalCars = 100)
  (h2 : company.baseRent = 3000)
  (h3 : company.rentIncrement = 50)
  (h4 : company.maintenanceCost = 200)
  (h5 : company.decreaseRate = 1) :
  ∃ (r : ℕ), ∀ (rent : ℕ), revenue company rent ≤ revenue company r ∧ revenue company r = 304200 := by sorry

end NUMINAMATH_CALUDE_cars_rented_at_3600_optimal_rent_max_revenue_l2980_298083


namespace NUMINAMATH_CALUDE_pseudoprime_propagation_l2980_298017

/-- A number n is a pseudoprime to base 2 if 2^n ≡ 2 (mod n) --/
def is_pseudoprime_base_2 (n : ℕ) : Prop :=
  2^n % n = 2 % n

theorem pseudoprime_propagation (n : ℕ) (h : is_pseudoprime_base_2 n) :
  is_pseudoprime_base_2 (2^n - 1) :=
sorry

end NUMINAMATH_CALUDE_pseudoprime_propagation_l2980_298017


namespace NUMINAMATH_CALUDE_debate_team_boys_l2980_298042

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) (boys : ℕ) : 
  girls = 32 →
  groups = 7 →
  group_size = 9 →
  total = groups * group_size →
  boys = total - girls →
  boys = 31 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l2980_298042


namespace NUMINAMATH_CALUDE_xiao_zhao_grade_l2980_298016

/-- Calculates the final grade based on component scores and weights -/
def calculate_grade (class_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) 
  (class_weight : ℝ) (midterm_weight : ℝ) (final_weight : ℝ) : ℝ :=
  class_score * class_weight + midterm_score * midterm_weight + final_score * final_weight

/-- Theorem stating that Xiao Zhao's physical education grade is 44.5 -/
theorem xiao_zhao_grade : 
  let max_score : ℝ := 50
  let class_weight : ℝ := 0.3
  let midterm_weight : ℝ := 0.2
  let final_weight : ℝ := 0.5
  let class_score : ℝ := 40
  let midterm_score : ℝ := 50
  let final_score : ℝ := 45
  calculate_grade class_score midterm_score final_score class_weight midterm_weight final_weight = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_zhao_grade_l2980_298016


namespace NUMINAMATH_CALUDE_chairs_per_rectangular_table_l2980_298074

theorem chairs_per_rectangular_table :
  let round_tables : ℕ := 2
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let total_chairs : ℕ := 26
  (total_chairs - round_tables * chairs_per_round_table) / rectangular_tables = 7 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_rectangular_table_l2980_298074


namespace NUMINAMATH_CALUDE_total_bills_calculation_l2980_298071

def withdrawal1 : ℕ := 450
def withdrawal2 : ℕ := 750
def bill_value : ℕ := 20

theorem total_bills_calculation : 
  (withdrawal1 + withdrawal2) / bill_value = 60 := by sorry

end NUMINAMATH_CALUDE_total_bills_calculation_l2980_298071


namespace NUMINAMATH_CALUDE_solution_difference_l2980_298095

-- Define the equation
def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ (2 * x^2 - 5 * x - 31) / (x - 3) = 3 * x + 8

-- Define the set of solutions
def solutions : Set ℝ :=
  {x | equation x}

-- State the theorem
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 2 * Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l2980_298095


namespace NUMINAMATH_CALUDE_real_nut_findable_l2980_298047

/-- Represents the type of a nut -/
inductive NutType
| Real
| Artificial

/-- Represents the result of a weighing -/
inductive WeighResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a collection of nuts -/
structure NutCollection :=
  (nuts : Fin 6 → NutType)
  (realCount : Nat)
  (artificialCount : Nat)
  (real_count_correct : realCount = 4)
  (artificial_count_correct : artificialCount = 2)

/-- Represents a weighing operation -/
def weighNuts (collection : NutCollection) (left right sacrificed : Fin 6) : WeighResult :=
  sorry

/-- Represents the process of finding a real nut -/
def findRealNut (collection : NutCollection) : Fin 6 :=
  sorry

/-- The main theorem: it's possible to find a real nut without sacrificing it -/
theorem real_nut_findable (collection : NutCollection) :
  ∃ (n : Fin 6), collection.nuts n = NutType.Real ∧ n ≠ findRealNut collection :=
sorry

end NUMINAMATH_CALUDE_real_nut_findable_l2980_298047


namespace NUMINAMATH_CALUDE_count_four_digit_distinct_prime_last_l2980_298099

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of single-digit prime numbers -/
def singleDigitPrimes : Finset ℕ := sorry

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ := sorry

/-- The count of four-digit numbers with distinct digits and a prime last digit -/
def countValidNumbers : ℕ := sorry

theorem count_four_digit_distinct_prime_last :
  countValidNumbers = 1344 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_distinct_prime_last_l2980_298099


namespace NUMINAMATH_CALUDE_second_grade_sample_count_l2980_298088

/-- Represents a high school with three grades forming an arithmetic sequence -/
structure HighSchool where
  total_students : ℕ
  sampled_students : ℕ
  grade_sequence : Fin 3 → ℕ
  is_arithmetic_sequence : ∃ (d : ℤ), 
    (grade_sequence 1 : ℤ) = (grade_sequence 0 : ℤ) + d ∧
    (grade_sequence 2 : ℤ) = (grade_sequence 1 : ℤ) + d
  sum_equals_total : (grade_sequence 0) + (grade_sequence 1) + (grade_sequence 2) = total_students

/-- The number of students sampled from the second grade in a stratified sampling -/
def sampled_from_second_grade (school : HighSchool) : ℕ :=
  (school.grade_sequence 1 * school.sampled_students) / school.total_students

/-- Theorem stating the number of students sampled from the second grade -/
theorem second_grade_sample_count 
  (school : HighSchool)
  (h1 : school.total_students = 1200)
  (h2 : school.sampled_students = 48) :
  sampled_from_second_grade school = 16 := by
  sorry


end NUMINAMATH_CALUDE_second_grade_sample_count_l2980_298088


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2980_298085

theorem quadratic_roots_reciprocal_sum (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → (n^2 + 4*n - 1 = 0) → (1/m + 1/n = 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2980_298085


namespace NUMINAMATH_CALUDE_least_subtraction_l2980_298065

theorem least_subtraction (n : Nat) (a b c : Nat) (r : Nat) : 
  (∀ m : Nat, m < n → 
    ((2590 - m) % a ≠ r ∨ (2590 - m) % b ≠ r ∨ (2590 - m) % c ≠ r)) →
  (2590 - n) % a = r ∧ (2590 - n) % b = r ∧ (2590 - n) % c = r →
  n = 16 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l2980_298065


namespace NUMINAMATH_CALUDE_biased_coin_prob_l2980_298077

/-- Represents the probability of getting heads on a single flip of a biased coin -/
def h : ℚ := 3 / 8

/-- The number of flips -/
def n : ℕ := 7

/-- Probability of getting exactly k heads in n flips -/
def prob_k_heads (k : ℕ) : ℚ :=
  (n.choose k) * h^k * (1 - h)^(n - k)

theorem biased_coin_prob :
  prob_k_heads 2 = prob_k_heads 3 ∧
  prob_k_heads 4 = 675 / 3999 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_prob_l2980_298077


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2980_298001

theorem inscribed_quadrilateral_fourth_side 
  (r : ℝ) 
  (a b c : ℝ) 
  (h1 : r = 150 * Real.sqrt 3)
  (h2 : a = 150)
  (h3 : b = 300)
  (h4 : c = 150)
  (h5 : ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ Real.cos θ = 0) :
  ∃ d : ℝ, d = 300 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l2980_298001


namespace NUMINAMATH_CALUDE_inequality_properties_l2980_298076

theorem inequality_properties (x y : ℝ) (h : x > y) : 
  (x - 3 > y - 3) ∧ 
  (x / 3 > y / 3) ∧ 
  (x + 3 > y + 3) ∧ 
  (-3 * x < -3 * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2980_298076


namespace NUMINAMATH_CALUDE_F_inequalities_l2980_298067

/-- A function is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is monotonically increasing on [0,+∞) if for any x ≥ y ≥ 0, f(x) ≥ f(y) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ y → y ≤ x → f y ≤ f x

/-- Definition of the function F -/
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + g (1 - x) - |f x - g (1 - x)|

theorem F_inequalities (f g : ℝ → ℝ) (a : ℝ)
    (hf_even : EvenFunction f) (hg_even : EvenFunction g)
    (hf_mono : MonoIncreasing f) (hg_mono : MonoIncreasing g)
    (ha : a > 0) :
    F f g (-a) ≥ F f g a ∧ F f g (1 + a) ≥ F f g (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_F_inequalities_l2980_298067


namespace NUMINAMATH_CALUDE_birthday_crayons_l2980_298080

theorem birthday_crayons (given_away lost remaining : ℕ) 
  (h1 : given_away = 111)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  given_away + lost + remaining = 440 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l2980_298080


namespace NUMINAMATH_CALUDE_number_of_subjects_proof_l2980_298037

/-- Given the average scores and individual subject scores, prove the number of subjects. -/
theorem number_of_subjects_proof (physics chemistry mathematics : ℝ) : 
  (physics + chemistry + mathematics) / 3 = 75 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  physics = 95 →
  ∃ (n : ℕ), n = 3 ∧ n > 0 := by
  sorry

#check number_of_subjects_proof

end NUMINAMATH_CALUDE_number_of_subjects_proof_l2980_298037


namespace NUMINAMATH_CALUDE_inscribed_polygon_area_l2980_298053

/-- A polygon inscribed around a circle -/
structure InscribedPolygon where
  /-- The radius of the circle -/
  r : ℝ
  /-- The semiperimeter of the polygon -/
  p : ℝ
  /-- The area of the polygon -/
  area : ℝ

/-- Theorem: The area of a polygon inscribed around a circle is equal to the product of its semiperimeter and the radius of the circle -/
theorem inscribed_polygon_area (poly : InscribedPolygon) : poly.area = poly.p * poly.r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_area_l2980_298053


namespace NUMINAMATH_CALUDE_triangle_side_length_l2980_298079

theorem triangle_side_length (A B C : ℝ) (h1 : Real.cos (3*A - B) + Real.sin (A + B) = 2) 
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π) (h5 : A + B + C = π) 
  (h6 : (4 : ℝ) = 4 * Real.sin A / Real.sin C) : 
  4 * Real.sin B / Real.sin C = 2 * Real.sqrt (2 - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2980_298079


namespace NUMINAMATH_CALUDE_abc_fraction_value_l2980_298010

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 2)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 140 / 59 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l2980_298010


namespace NUMINAMATH_CALUDE_rectangle_area_from_quadratic_roots_l2980_298035

theorem rectangle_area_from_quadratic_roots : 
  ∀ (length width : ℝ),
  (2 * length^2 - 11 * length + 5 = 0) →
  (2 * width^2 - 11 * width + 5 = 0) →
  (length * width = 5 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_from_quadratic_roots_l2980_298035


namespace NUMINAMATH_CALUDE_smallest_perimeter_square_sides_l2980_298094

theorem smallest_perimeter_square_sides : ∃ (a b c : ℕ), 
  (0 < a ∧ 0 < b ∧ 0 < c) ∧  -- positive integers
  (a < b ∧ b < c) ∧  -- distinct
  (a^2 + b^2 > c^2) ∧  -- triangle inequality
  (a^2 + c^2 > b^2) ∧
  (b^2 + c^2 > a^2) ∧
  (a^2 + b^2 + c^2 = 77) ∧  -- perimeter is 77
  (∀ (x y z : ℕ), (0 < x ∧ 0 < y ∧ 0 < z) →
    (x < y ∧ y < z) →
    (x^2 + y^2 > z^2) →
    (x^2 + z^2 > y^2) →
    (y^2 + z^2 > x^2) →
    (x^2 + y^2 + z^2 ≥ 77)) :=
by
  sorry

#check smallest_perimeter_square_sides

end NUMINAMATH_CALUDE_smallest_perimeter_square_sides_l2980_298094


namespace NUMINAMATH_CALUDE_diana_charge_account_debt_l2980_298044

/-- Calculate the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the specified conditions, the total amount owed is $63.60 --/
theorem diana_charge_account_debt : 
  let principal : ℝ := 60
  let rate : ℝ := 0.06
  let time : ℝ := 1
  total_amount_owed principal rate time = 63.60 := by
  sorry

end NUMINAMATH_CALUDE_diana_charge_account_debt_l2980_298044


namespace NUMINAMATH_CALUDE_fraction_inequalities_l2980_298005

theorem fraction_inequalities (a b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (a < b → (a + c) / (b + c) > a / b) ∧
  (a > b → (a + c) / (b + c) < a / b) ∧
  ((a < b → a / b < (a + c) / (b + c) ∧ (a + c) / (b + c) < 1) ∧
   (a > b → 1 < (a + c) / (b + c) ∧ (a + c) / (b + c) < a / b)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequalities_l2980_298005


namespace NUMINAMATH_CALUDE_equation_solutions_l2980_298007

def equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6 ∧
  (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3) /
  ((x - 4) * (x - 6) * (x - 4)) = 1

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2980_298007


namespace NUMINAMATH_CALUDE_exists_Q_on_x_axis_l2980_298024

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the locus E
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4 + p.2^2 / 3 = 1)}

-- Define point A as the intersection of E with negative x-axis
def A : ℝ × ℝ := (-2, 0)

-- Define a function to represent a line through F not coinciding with x-axis
def lineThruF (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = m * p.2 + 1}

-- Define B and C as intersections of E with lineThruF
def B (m : ℝ) : ℝ × ℝ := sorry
def C (m : ℝ) : ℝ × ℝ := sorry

-- Define M and N as intersections of AB and AC with l
def M (m : ℝ) : ℝ × ℝ := sorry
def N (m : ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem exists_Q_on_x_axis :
  ∃ x₀ : ℝ, let Q := (x₀, 0)
  ∀ m : ℝ, m ≠ 0 →
    (Q.1 - (M m).1) * (Q.1 - (N m).1) +
    (Q.2 - (M m).2) * (Q.2 - (N m).2) = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_Q_on_x_axis_l2980_298024


namespace NUMINAMATH_CALUDE_identical_circles_inside_no_common_tangents_l2980_298009

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if one circle is fully inside another without touching -/
def IsFullyInside (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 < (c1.radius - c2.radius) ^ 2

/-- The number of common tangents between two circles -/
def CommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two identical circles, one fully inside the other without touching, have zero common tangents -/
theorem identical_circles_inside_no_common_tangents (c1 c2 : Circle) :
  c1.radius = c2.radius → IsFullyInside c1 c2 → CommonTangents c1 c2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_identical_circles_inside_no_common_tangents_l2980_298009


namespace NUMINAMATH_CALUDE_class_arrangement_probability_l2980_298008

/-- The number of classes in a school day -/
def num_classes : ℕ := 6

/-- The total number of possible arrangements of classes -/
def total_arrangements : ℕ := num_classes.factorial

/-- The number of arrangements where Mathematics is not the last class
    and Physical Education is not the first class -/
def valid_arrangements : ℕ :=
  (num_classes - 1).factorial + (num_classes - 2) * (num_classes - 2) * (num_classes - 2).factorial

/-- The probability that Mathematics is not the last class
    and Physical Education is not the first class -/
theorem class_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_class_arrangement_probability_l2980_298008


namespace NUMINAMATH_CALUDE_special_sequence_properties_l2980_298059

/-- A sequence of 2000 positive integers satisfying the given conditions -/
def special_sequence : Fin 2000 → ℕ
  | ⟨i, _⟩ => 2^(2000 + i) * 3^(3999 - i)

theorem special_sequence_properties :
  ∃ (seq : Fin 2000 → ℕ),
    (∀ i j, i ≠ j → ¬(seq i ∣ seq j)) ∧
    (∀ i j, i ≠ j → (seq i)^2 ∣ seq j) :=
by
  use special_sequence
  sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l2980_298059


namespace NUMINAMATH_CALUDE_retractable_door_unique_non_triangle_l2980_298058

/-- A design that may or may not utilize the stability of a triangle. -/
inductive Design
  | RetractableDoor
  | BicycleFrame
  | WindowFrame
  | CameraTripod

/-- Predicate indicating whether a design utilizes the stability of a triangle. -/
def utilizesTriangleStability (d : Design) : Prop :=
  match d with
  | Design.RetractableDoor => False
  | Design.BicycleFrame => True
  | Design.WindowFrame => True
  | Design.CameraTripod => True

/-- Theorem stating that only the retractable door does not utilize triangle stability. -/
theorem retractable_door_unique_non_triangle :
    ∀ (d : Design), ¬(utilizesTriangleStability d) ↔ d = Design.RetractableDoor := by
  sorry

end NUMINAMATH_CALUDE_retractable_door_unique_non_triangle_l2980_298058


namespace NUMINAMATH_CALUDE_scheduling_methods_count_l2980_298089

/-- Represents the number of days in the schedule -/
def num_days : ℕ := 7

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 4

/-- Calculates the number of scheduling methods -/
def scheduling_methods : ℕ :=
  -- This function should implement the logic to calculate the number of scheduling methods
  -- based on the given conditions
  sorry

/-- Theorem stating that the number of scheduling methods is 420 -/
theorem scheduling_methods_count : scheduling_methods = 420 := by
  sorry

end NUMINAMATH_CALUDE_scheduling_methods_count_l2980_298089


namespace NUMINAMATH_CALUDE_bicycle_cost_calculation_l2980_298078

theorem bicycle_cost_calculation (selling_price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : 
  selling_price = 990 ∧ 
  profit_rate = 0.1 ∧ 
  loss_rate = 0.1 → 
  (selling_price / (1 + profit_rate)) + (selling_price / (1 - loss_rate)) = 2000 := by
sorry

end NUMINAMATH_CALUDE_bicycle_cost_calculation_l2980_298078


namespace NUMINAMATH_CALUDE_common_tangent_count_possibilities_l2980_298002

/-- The number of possible values for the count of common tangents between two circles -/
def possible_tangent_counts : ℕ := 5

/-- The radii of the two circles -/
def circle_radii : Fin 2 → ℝ
  | 0 => 2
  | 1 => 3

/-- The set of possible numbers of common tangents -/
def tangent_counts : Finset ℕ := {0, 1, 2, 3, 4}

/-- Theorem stating that the number of possible values for the count of common tangents
    between two circles with radii 2 and 3 is equal to the cardinality of the set of
    possible numbers of common tangents -/
theorem common_tangent_count_possibilities :
  possible_tangent_counts = Finset.card tangent_counts :=
sorry

end NUMINAMATH_CALUDE_common_tangent_count_possibilities_l2980_298002


namespace NUMINAMATH_CALUDE_estimate_comparison_l2980_298031

theorem estimate_comparison (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 1) (hw : w > 0) (hxy : x > y) :
  (x + w) - (y - w) * z > x - y * z := by
  sorry

end NUMINAMATH_CALUDE_estimate_comparison_l2980_298031


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2980_298069

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = 5 ∧ x = 3) → 
  (∃ y : ℝ, 3 * y^2 + k * y = 5 ∧ y = -5/9) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2980_298069


namespace NUMINAMATH_CALUDE_number_puzzle_l2980_298050

theorem number_puzzle (N : ℝ) : (N / 4) * 12 - 18 = 3 * 12 + 27 → N = 27 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2980_298050


namespace NUMINAMATH_CALUDE_modulus_of_3_minus_i_l2980_298036

theorem modulus_of_3_minus_i :
  let z : ℂ := 3 - I
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_3_minus_i_l2980_298036


namespace NUMINAMATH_CALUDE_max_value_problem_l2980_298030

theorem max_value_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 1 / Real.sqrt 3) : 
  27 * a * b * c + a * Real.sqrt (a^2 + 2*b*c) + b * Real.sqrt (b^2 + 2*c*a) + c * Real.sqrt (c^2 + 2*a*b) 
  ≤ 2 / (3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l2980_298030


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2980_298027

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2980_298027


namespace NUMINAMATH_CALUDE_pell_equation_infinite_solutions_l2980_298086

theorem pell_equation_infinite_solutions (a : ℤ) (h_a : a > 1) 
  (u v : ℤ) (h_sol : u^2 - a * v^2 = -1) : 
  ∃ (S : Set (ℤ × ℤ)), Infinite S ∧ ∀ (p : ℤ × ℤ), p ∈ S → (p.1)^2 - a * (p.2)^2 = -1 :=
sorry

end NUMINAMATH_CALUDE_pell_equation_infinite_solutions_l2980_298086


namespace NUMINAMATH_CALUDE_borrow_methods_eq_seven_l2980_298057

/-- The number of ways to borrow at least one book from a set of three books -/
def borrow_methods : ℕ :=
  2^3 - 1

/-- Theorem stating that the number of ways to borrow at least one book from three books is 7 -/
theorem borrow_methods_eq_seven : borrow_methods = 7 := by
  sorry

end NUMINAMATH_CALUDE_borrow_methods_eq_seven_l2980_298057


namespace NUMINAMATH_CALUDE_elevator_height_after_20_seconds_l2980_298073

/-- Calculates the height of a descending elevator after a given time. -/
def elevatorHeight (initialHeight : ℝ) (descentSpeed : ℝ) (time : ℝ) : ℝ :=
  initialHeight - descentSpeed * time

/-- Theorem: An elevator starting at 120 meters above ground and descending
    at 4 meters per second will be at 40 meters after 20 seconds. -/
theorem elevator_height_after_20_seconds :
  elevatorHeight 120 4 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_elevator_height_after_20_seconds_l2980_298073


namespace NUMINAMATH_CALUDE_system_solution_l2980_298046

theorem system_solution (a b : ℝ) 
  (h1 : 2 * a * 3 + 3 * 4 = 18) 
  (h2 : -(3) + 5 * b * 4 = 17) : 
  ∃ (x y : ℝ), 2 * a * (x + y) + 3 * (x - y) = 18 ∧ 
               (x + y) - 5 * b * (x - y) = -17 ∧ 
               x = 3.5 ∧ y = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2980_298046


namespace NUMINAMATH_CALUDE_units_digit_of_F_F_10_l2980_298064

def modifiedFibonacci : ℕ → ℕ
  | 0 => 4
  | 1 => 3
  | (n + 2) => modifiedFibonacci (n + 1) + modifiedFibonacci n

theorem units_digit_of_F_F_10 : 
  (modifiedFibonacci (modifiedFibonacci 10)) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_F_F_10_l2980_298064


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2980_298020

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 2 + a 6 = 8) ∧
  (a 3 + a 4 = 3)

/-- The common difference of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2980_298020


namespace NUMINAMATH_CALUDE_complex_fractions_sum_l2980_298026

theorem complex_fractions_sum (x y z : ℂ) 
  (h1 : x / (y + z) + y / (z + x) + z / (x + y) = 9)
  (h2 : x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y) = 64)
  (h3 : x^3 / (y + z) + y^3 / (z + x) + z^3 / (x + y) = 488) :
  x / (y * z) + y / (z * x) + z / (x * y) = 3 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fractions_sum_l2980_298026


namespace NUMINAMATH_CALUDE_complex_equation_proof_l2980_298082

theorem complex_equation_proof (z : ℂ) (h : Complex.abs z = 1 + 3 * I - z) :
  ((1 + I)^2 * (3 + 4*I)) / (2 * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l2980_298082


namespace NUMINAMATH_CALUDE_thirty_percent_greater_than_88_l2980_298081

theorem thirty_percent_greater_than_88 (x : ℝ) : 
  x = 88 * (1 + 30 / 100) → x = 114.4 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_greater_than_88_l2980_298081


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2980_298022

/-- A quadratic function with vertex at (2,1) and opening upward -/
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem quadratic_function_properties :
  (∀ x y z : ℝ, f ((x + y) / 2) ≤ (f x + f y) / 2) ∧  -- Convexity (implies upward opening)
  (∀ x : ℝ, f x ≥ f 2) ∧                              -- Minimum at x = 2
  f 2 = 1                                             -- Vertex y-coordinate is 1
:= by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2980_298022


namespace NUMINAMATH_CALUDE_hippopotamus_crayons_l2980_298084

theorem hippopotamus_crayons (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 62)
  (h2 : remaining_crayons = 10) :
  initial_crayons - remaining_crayons = 52 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_crayons_l2980_298084


namespace NUMINAMATH_CALUDE_no_geometric_subsequence_in_arithmetic_sequence_l2980_298066

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n + r

def contains_one_and_sqrt_two (a : ℕ → ℝ) : Prop :=
  ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem no_geometric_subsequence_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_contains : contains_one_and_sqrt_two a) :
  ¬ ∃ m n p : ℕ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ is_geometric_sequence (a m) (a n) (a p) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_subsequence_in_arithmetic_sequence_l2980_298066


namespace NUMINAMATH_CALUDE_salary_increase_after_three_years_l2980_298062

/-- The annual percentage increase in salary -/
def annual_increase : ℝ := 0.12

/-- The number of years of salary increase -/
def years : ℕ := 3

/-- The total percentage increase after a given number of years -/
def total_increase (y : ℕ) : ℝ := (1 + annual_increase) ^ y - 1

theorem salary_increase_after_three_years :
  ∃ ε > 0, abs (total_increase years - 0.4057) < ε :=
sorry

end NUMINAMATH_CALUDE_salary_increase_after_three_years_l2980_298062


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l2980_298068

/-- Given a rectangular room carpeted with multiple strips of carpet, calculate the breadth of the room. -/
theorem room_breadth_calculation 
  (room_length : ℝ)
  (carpet_width : ℝ)
  (carpet_cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : room_length = 15)
  (h2 : carpet_width = 0.75)
  (h3 : carpet_cost_per_meter = 0.30)
  (h4 : total_cost = 36) :
  (total_cost / carpet_cost_per_meter) / room_length * carpet_width = 6 :=
by sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l2980_298068


namespace NUMINAMATH_CALUDE_two_bedroom_units_l2980_298019

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ)
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950)
  (h5 : ∃ (x y : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost) :
  ∃ (y : ℕ), y = 7 ∧ ∃ (x : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l2980_298019


namespace NUMINAMATH_CALUDE_x_cubed_minus_y_equals_plus_minus_17_l2980_298093

theorem x_cubed_minus_y_equals_plus_minus_17 
  (x y : ℝ) 
  (h1 : x^2 = 4) 
  (h2 : |y| = 9) 
  (h3 : x * y < 0) : 
  x^3 - y = 17 ∨ x^3 - y = -17 := by
sorry

end NUMINAMATH_CALUDE_x_cubed_minus_y_equals_plus_minus_17_l2980_298093


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l2980_298098

theorem cos_angle_between_vectors (a b : ℝ × ℝ) :
  a = (-2, 1) →
  a + 2 • b = (2, 3) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l2980_298098


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l2980_298097

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 15)
  (h2 : c + a = 18)
  (h3 : a + b = 21) :
  Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l2980_298097


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2980_298011

theorem multiplication_addition_equality : 3.6 * 0.5 + 1.2 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2980_298011


namespace NUMINAMATH_CALUDE_sasha_work_hours_l2980_298091

/-- Calculates the number of hours Sasha worked given her question completion rate,
    total number of questions, and remaining questions. -/
def hours_worked (completion_rate : ℕ) (total_questions : ℕ) (remaining_questions : ℕ) : ℚ :=
  (total_questions - remaining_questions) / completion_rate

/-- Proves that Sasha worked for 2 hours given the problem conditions. -/
theorem sasha_work_hours :
  let completion_rate : ℕ := 15
  let total_questions : ℕ := 60
  let remaining_questions : ℕ := 30
  hours_worked completion_rate total_questions remaining_questions = 2 := by
  sorry

end NUMINAMATH_CALUDE_sasha_work_hours_l2980_298091


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2980_298096

theorem quadratic_root_difference (p q : ℝ) : 
  let r := (p + Real.sqrt (p^2 + q))
  let s := (p - Real.sqrt (p^2 + q))
  abs (r - s) = Real.sqrt (2 * p^2 + 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2980_298096


namespace NUMINAMATH_CALUDE_simplify_fraction_l2980_298090

theorem simplify_fraction : (98 : ℚ) / 210 = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2980_298090


namespace NUMINAMATH_CALUDE_katherines_apples_katherines_apples_proof_l2980_298048

theorem katherines_apples : ℕ → Prop :=
  fun a : ℕ =>
    let p := 3 * a  -- number of pears
    let b := 5  -- number of bananas
    (a + p + b = 21) →  -- total number of fruits
    (a = 4)

-- Proof
theorem katherines_apples_proof : ∃ a : ℕ, katherines_apples a :=
  sorry

end NUMINAMATH_CALUDE_katherines_apples_katherines_apples_proof_l2980_298048


namespace NUMINAMATH_CALUDE_final_alcohol_percentage_l2980_298052

def initial_volume : ℝ := 40
def initial_alcohol_percentage : ℝ := 5
def added_alcohol : ℝ := 5.5
def added_water : ℝ := 4.5

theorem final_alcohol_percentage :
  let initial_alcohol := initial_volume * (initial_alcohol_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  (final_alcohol / final_volume) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_final_alcohol_percentage_l2980_298052


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l2980_298060

def carrot_sticks_before_dinner (total : ℕ) (after_dinner : ℕ) : ℕ :=
  total - after_dinner

theorem james_carrot_sticks :
  carrot_sticks_before_dinner 37 15 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l2980_298060


namespace NUMINAMATH_CALUDE_dividing_sum_theorem_l2980_298061

def is_valid_solution (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧
  a ∣ (b + c + d) ∧
  b ∣ (a + c + d) ∧
  c ∣ (a + b + d) ∧
  d ∣ (a + b + c)

def basis_solutions : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 3, 6), (1, 2, 6, 9), (1, 3, 8, 12), (1, 4, 5, 10), (1, 6, 14, 21), (2, 3, 10, 15)]

theorem dividing_sum_theorem :
  ∀ a b c d : ℕ, is_valid_solution a b c d →
    ∃ k : ℕ, ∃ (x y z w : ℕ), (x, y, z, w) ∈ basis_solutions ∧
      a = k * x ∧ b = k * y ∧ c = k * z ∧ d = k * w :=
sorry

end NUMINAMATH_CALUDE_dividing_sum_theorem_l2980_298061


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2980_298012

theorem price_increase_percentage (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_increase := original_price * 1.2
  let final_price := price_after_first_increase * 1.15
  let total_increase := final_price - original_price
  (total_increase / original_price) * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2980_298012


namespace NUMINAMATH_CALUDE_necessarily_negative_l2980_298063

theorem necessarily_negative (y z : ℝ) (h1 : -1 < y) (h2 : y < 0) (h3 : 0 < z) (h4 : z < 1) :
  y - z < 0 := by
  sorry

end NUMINAMATH_CALUDE_necessarily_negative_l2980_298063


namespace NUMINAMATH_CALUDE_smallest_percentage_correct_l2980_298051

theorem smallest_percentage_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8) 
  (h3 : p3 = 0.7) : 
  (1 - ((1 - p1) + (1 - p2) + (1 - p3))) ≥ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percentage_correct_l2980_298051


namespace NUMINAMATH_CALUDE_angle_sum_undetermined_l2980_298006

/-- Two angles are consecutive interior angles -/
def consecutive_interior (α β : ℝ) : Prop := sorry

/-- The statement that the sum of two angles equals 180 degrees cannot be proven or disproven -/
def undetermined_sum (α β : ℝ) : Prop :=
  ¬(∀ (h : consecutive_interior α β), α + β = 180) ∧
  ¬(∀ (h : consecutive_interior α β), α + β ≠ 180)

theorem angle_sum_undetermined (α β : ℝ) :
  consecutive_interior α β → α = 78 → undetermined_sum α β :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_undetermined_l2980_298006


namespace NUMINAMATH_CALUDE_duck_selling_price_l2980_298021

/-- Calculates the selling price per pound of ducks -/
def selling_price_per_pound (num_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (total_profit : ℚ) : ℚ :=
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_cost + total_profit
  total_revenue / total_weight

/-- Proves that the selling price per pound is $5 given the problem conditions -/
theorem duck_selling_price :
  selling_price_per_pound 30 10 4 300 = 5 := by
  sorry

#eval selling_price_per_pound 30 10 4 300

end NUMINAMATH_CALUDE_duck_selling_price_l2980_298021


namespace NUMINAMATH_CALUDE_sum_of_digits_in_divisible_number_l2980_298014

/-- 
Given a number in the form ̄1ab76 that is divisible by 72, 
prove that the sum a+b can only be 4 or 13.
-/
theorem sum_of_digits_in_divisible_number (a b : Nat) : 
  (∃ (n : Nat), n = 10000 + 1000 * a + 100 * b + 76) →
  (10000 + 1000 * a + 100 * b + 76) % 72 = 0 →
  a + b = 4 ∨ a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_divisible_number_l2980_298014


namespace NUMINAMATH_CALUDE_square_sum_digits_l2980_298028

theorem square_sum_digits (n : ℕ) : 
  let A := 4 * (10^(2*n) - 1) / 9
  let B := 8 * (10^n - 1) / 9
  A + 2*B + 4 = (2*(10^n + 2)/3)^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_digits_l2980_298028
