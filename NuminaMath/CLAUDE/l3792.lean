import Mathlib

namespace NUMINAMATH_CALUDE_vector_equation_l3792_379237

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equation : c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l3792_379237


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l3792_379292

def num_people : ℕ := 5
def die_sides : ℕ := 6

def probability_no_adjacent_same : ℚ :=
  375 / 2592

theorem circular_table_dice_probability :
  let total_outcomes := die_sides ^ num_people
  let favorable_outcomes := 
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 2)) +
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 1) / die_sides)
  favorable_outcomes / total_outcomes = probability_no_adjacent_same := by
  sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l3792_379292


namespace NUMINAMATH_CALUDE_train_length_calculation_l3792_379291

/-- Proves that given a train and platform of equal length, if the train crosses the platform
    in one minute at a speed of 216 km/hr, then the length of the train is 1800 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 216 →
  time = 1 / 60 →
  train_length = 1800 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3792_379291


namespace NUMINAMATH_CALUDE_jason_shelves_needed_l3792_379262

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (regular_books : ℕ) (large_books : ℕ) : ℕ :=
  let regular_shelves := (regular_books + 44) / 45
  let large_shelves := (large_books + 29) / 30
  regular_shelves + large_shelves

/-- Theorem stating that Jason needs 9 shelves to store all his books -/
theorem jason_shelves_needed : shelves_needed 240 75 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_shelves_needed_l3792_379262


namespace NUMINAMATH_CALUDE_probability_inequalities_l3792_379209

open ProbabilityTheory MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] [Fintype Ω]
variable (P : Measure Ω) [IsProbabilityMeasure P]
variable (A B : Set Ω)

theorem probability_inequalities
  (h1 : P A = P (Aᶜ))
  (h2 : P (Bᶜ ∩ A) / P A > P (B ∩ Aᶜ) / P Aᶜ) :
  (P (A ∩ Bᶜ) > P (Aᶜ ∩ B)) ∧ (P (A ∩ B) < P (Aᶜ ∩ Bᶜ)) := by
  sorry

end NUMINAMATH_CALUDE_probability_inequalities_l3792_379209


namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l3792_379274

/-- The value of k for which the line x = k intersects the parabola x = 3y² - 7y + 2 at exactly one point -/
def k : ℚ := -25/12

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3*y^2 - 7*y + 2

/-- Theorem stating that k is the unique value for which the line x = k intersects the parabola at exactly one point -/
theorem line_intersects_parabola_once :
  ∀ y : ℝ, (∃! y, parabola y = k) ∧ 
  (∀ k' : ℚ, k' ≠ k → ¬(∃! y, parabola y = k')) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l3792_379274


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3792_379212

/-- Represents the total number of employees -/
def total_employees : ℕ := 750

/-- Represents the number of young employees -/
def young_employees : ℕ := 350

/-- Represents the number of middle-aged employees -/
def middle_aged_employees : ℕ := 250

/-- Represents the number of elderly employees -/
def elderly_employees : ℕ := 150

/-- Represents the number of young employees in the sample -/
def young_in_sample : ℕ := 7

/-- Theorem stating that the sample size is 15 given the conditions -/
theorem stratified_sample_size :
  (total_employees = young_employees + middle_aged_employees + elderly_employees) →
  (young_in_sample * total_employees = 15 * young_employees) :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l3792_379212


namespace NUMINAMATH_CALUDE_max_dimes_and_nickels_l3792_379258

def total_amount : ℚ := 485 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100

theorem max_dimes_and_nickels :
  ∃ (d : ℕ), d * dime_value + d * nickel_value ≤ total_amount ∧
  ∀ (n : ℕ), n * dime_value + n * nickel_value ≤ total_amount → n ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_and_nickels_l3792_379258


namespace NUMINAMATH_CALUDE_derivative_at_one_l3792_379283

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_at_one (x : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → |((f (1 + h) - f 1) / h) - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3792_379283


namespace NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3792_379221

/-- Given a spherical ball partially submerged in a frozen surface,
    where the hole left by the ball has a diameter of 30 cm and a depth of 8 cm,
    prove that the radius of the ball is 18.0625 cm. -/
theorem ball_radius_from_hole_dimensions (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 →
  depth = 8 →
  radius = (((diameter / 2) ^ 2 + depth ^ 2) / (2 * depth)).sqrt →
  radius = 18.0625 := by
sorry

end NUMINAMATH_CALUDE_ball_radius_from_hole_dimensions_l3792_379221


namespace NUMINAMATH_CALUDE_solution_values_l3792_379282

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_valid_p (a b p : E) : Prop :=
  ‖p - b‖ = 3 * ‖p - a‖

def fixed_distance (a b p : E) (t u : ℝ) : Prop :=
  ∃ (k : ℝ), ‖p - (t • a + u • b)‖ = k

theorem solution_values (a b : E) :
  ∃ (p : E), is_valid_p a b p ∧
  fixed_distance a b p (9/8) (-1/8) :=
sorry

end NUMINAMATH_CALUDE_solution_values_l3792_379282


namespace NUMINAMATH_CALUDE_total_games_in_league_l3792_379261

theorem total_games_in_league (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_league_l3792_379261


namespace NUMINAMATH_CALUDE_tan_sum_special_l3792_379250

theorem tan_sum_special : Real.tan (17 * π / 180) + Real.tan (28 * π / 180) + Real.tan (17 * π / 180) * Real.tan (28 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l3792_379250


namespace NUMINAMATH_CALUDE_sqrt_two_squared_inverse_l3792_379290

theorem sqrt_two_squared_inverse : ((-Real.sqrt 2)^2)⁻¹ = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_inverse_l3792_379290


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3792_379210

-- 1. 78 × 4 + 488
theorem problem_1 : 78 * 4 + 488 = 800 := by sorry

-- 2. 350 × (12 + 342 ÷ 9)
theorem problem_2 : 350 * (12 + 342 / 9) = 17500 := by sorry

-- 3. (3600 - 18 × 200) ÷ 253
theorem problem_3 : (3600 - 18 * 200) / 253 = 0 := by sorry

-- 4. 1903 - 475 × 4
theorem problem_4 : 1903 - 475 * 4 = 3 := by sorry

-- 5. 480 ÷ (125 - 117)
theorem problem_5 : 480 / (125 - 117) = 60 := by sorry

-- 6. (243 - 162) ÷ 27 × 380
theorem problem_6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3792_379210


namespace NUMINAMATH_CALUDE_log_relation_l3792_379202

theorem log_relation (a b : ℝ) (ha : a = Real.log 128 / Real.log 4) (hb : b = Real.log 16 / Real.log 2) :
  a = (7 * b) / 8 := by sorry

end NUMINAMATH_CALUDE_log_relation_l3792_379202


namespace NUMINAMATH_CALUDE_problem_solution_l3792_379231

theorem problem_solution : ∃ x : ℚ, x + (1/4 * x) = 90 - (30/100 * 90) ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3792_379231


namespace NUMINAMATH_CALUDE_exam_score_per_correct_answer_l3792_379226

/-- Proves the number of marks scored for each correct answer in an exam -/
theorem exam_score_per_correct_answer 
  (total_questions : ℕ) 
  (total_marks : ℤ) 
  (correct_answers : ℕ) 
  (wrong_penalty : ℤ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 110)
  (h3 : correct_answers = 34)
  (h4 : wrong_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (score_per_correct : ℤ), 
    score_per_correct * correct_answers + wrong_penalty * (total_questions - correct_answers) = total_marks ∧ 
    score_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_per_correct_answer_l3792_379226


namespace NUMINAMATH_CALUDE_triangular_prism_float_l3792_379263

theorem triangular_prism_float (x : ℝ) : ¬ (0 < x ∧ x < Real.sqrt 3 / 2 ∧ (Real.sqrt 3 / 4) * x = x * (1 - x / Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_float_l3792_379263


namespace NUMINAMATH_CALUDE_smallest_k_for_two_roots_l3792_379296

/-- A quadratic trinomial with natural number coefficients -/
structure QuadraticTrinomial where
  k : ℕ
  p : ℕ
  q : ℕ

/-- Predicate to check if a quadratic trinomial has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (qt : QuadraticTrinomial) : Prop :=
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    qt.k * x₁^2 - qt.p * x₁ + qt.q = 0 ∧
    qt.k * x₂^2 - qt.p * x₂ + qt.q = 0

/-- The main theorem stating that 5 is the smallest natural number k satisfying the condition -/
theorem smallest_k_for_two_roots : 
  (∀ k < 5, ¬∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨k, p, q⟩) ∧
  (∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨5, p, q⟩) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_two_roots_l3792_379296


namespace NUMINAMATH_CALUDE_f_property_l3792_379257

def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem f_property (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3792_379257


namespace NUMINAMATH_CALUDE_mod_inverse_two_mod_221_l3792_379217

theorem mod_inverse_two_mod_221 : ∃ x : ℕ, x < 221 ∧ (2 * x) % 221 = 1 :=
by
  use 111
  sorry

end NUMINAMATH_CALUDE_mod_inverse_two_mod_221_l3792_379217


namespace NUMINAMATH_CALUDE_largest_n_with_1992_divisors_and_phi_divides_l3792_379216

/-- The number of positive divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

theorem largest_n_with_1992_divisors_and_phi_divides (n : ℕ) :
  (phi n ∣ n) →
  (divisor_count n = 1992) →
  n ≤ 2^1991 :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_1992_divisors_and_phi_divides_l3792_379216


namespace NUMINAMATH_CALUDE_star_three_two_l3792_379240

/-- The star operation defined as a * b = a * b^3 - b^2 + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - b^2 + 2

/-- Theorem stating that 3 star 2 equals 22 -/
theorem star_three_two : star 3 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_star_three_two_l3792_379240


namespace NUMINAMATH_CALUDE_optimal_allocation_l3792_379260

/-- Represents the allocation of workers in a furniture factory -/
structure WorkerAllocation where
  total_workers : ℕ
  tabletop_workers : ℕ
  tableleg_workers : ℕ
  tabletops_per_worker : ℕ
  tablelegs_per_worker : ℕ
  legs_per_table : ℕ

/-- Checks if the allocation produces matching numbers of tabletops and table legs -/
def is_matching_production (w : WorkerAllocation) : Prop :=
  w.tabletop_workers * w.tabletops_per_worker * w.legs_per_table = 
  w.tableleg_workers * w.tablelegs_per_worker

/-- The theorem stating the optimal worker allocation -/
theorem optimal_allocation :
  ∀ w : WorkerAllocation,
    w.total_workers = 60 ∧
    w.tabletops_per_worker = 3 ∧
    w.tablelegs_per_worker = 6 ∧
    w.legs_per_table = 4 ∧
    w.tabletop_workers + w.tableleg_workers = w.total_workers →
    (w.tabletop_workers = 20 ∧ w.tableleg_workers = 40) ↔ 
    is_matching_production w :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l3792_379260


namespace NUMINAMATH_CALUDE_remaining_money_is_16000_l3792_379200

def salary : ℚ := 160000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℚ := salary - (food_fraction * salary + rent_fraction * salary + clothes_fraction * salary)

theorem remaining_money_is_16000 : remaining_money = 16000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_16000_l3792_379200


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3792_379246

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 55) :
  1 / x + 1 / y = 16 / 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3792_379246


namespace NUMINAMATH_CALUDE_scientific_notation_35_million_l3792_379248

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 ^ 7) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_35_million_l3792_379248


namespace NUMINAMATH_CALUDE_distance_traveled_l3792_379284

theorem distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) 
  (h1 : original_speed = 8)
  (h2 : increased_speed = 12)
  (h3 : additional_distance = 20)
  : ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = original_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3792_379284


namespace NUMINAMATH_CALUDE_two_incorrect_statements_l3792_379247

/-- Represents the coefficients of a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Predicate to check if both roots are positive -/
def both_roots_positive (roots : QuadraticRoots) : Prop :=
  roots.x₁ > 0 ∧ roots.x₂ > 0

/-- Predicate to check if the given coefficients satisfy Vieta's formulas for the given roots -/
def satisfies_vieta (coeff : QuadraticCoefficients) (roots : QuadraticRoots) : Prop :=
  roots.x₁ + roots.x₂ = -coeff.b / coeff.a ∧ roots.x₁ * roots.x₂ = coeff.c / coeff.a

/-- The four statements about the signs of coefficients -/
def statement_1 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0
def statement_2 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_3 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_4 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b > 0 ∧ coeff.c > 0

/-- Main theorem: Exactly 2 out of 4 statements are incorrect for a quadratic equation with two positive roots -/
theorem two_incorrect_statements
  (coeff : QuadraticCoefficients)
  (roots : QuadraticRoots)
  (h_positive : both_roots_positive roots)
  (h_vieta : satisfies_vieta coeff roots) :
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ ¬statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ ¬statement_3 coeff ∧ statement_4 coeff) :=
by sorry

end NUMINAMATH_CALUDE_two_incorrect_statements_l3792_379247


namespace NUMINAMATH_CALUDE_eugene_sunday_swim_time_l3792_379201

/-- Represents the swim times for Eugene over three days -/
structure SwimTimes where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ

/-- Calculates the average swim time over three days -/
def averageSwimTime (times : SwimTimes) : ℚ :=
  (times.sunday + times.monday + times.tuesday : ℚ) / 3

theorem eugene_sunday_swim_time :
  ∃ (times : SwimTimes),
    times.monday = 30 ∧
    times.tuesday = 45 ∧
    averageSwimTime times = 34 ∧
    times.sunday = 27 :=
  sorry

end NUMINAMATH_CALUDE_eugene_sunday_swim_time_l3792_379201


namespace NUMINAMATH_CALUDE_find_divisor_l3792_379273

theorem find_divisor (divisor : ℕ) : 
  (∃ k : ℕ, (228712 + 5) = divisor * k) ∧ 
  (∀ n < 5, ¬∃ m : ℕ, (228712 + n) = divisor * m) →
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3792_379273


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3792_379266

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f 1 = 7 ∧
    f 2 = 12 ∧
    f 3 = 19

/-- Theorem stating that if f is a QuadraticFunction, then f(4) = 28 -/
theorem quadratic_function_value (f : ℝ → ℝ) (hf : QuadraticFunction f) : f 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3792_379266


namespace NUMINAMATH_CALUDE_prob_greater_than_four_l3792_379279

-- Define a fair 6-sided die
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the probability of an event on a fair die
def prob (event : Finset ℕ) : ℚ :=
  (event ∩ fair_die).card / fair_die.card

-- Define the event of rolling a number greater than 4
def greater_than_four : Finset ℕ := {5, 6}

-- Theorem statement
theorem prob_greater_than_four :
  prob greater_than_four = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_greater_than_four_l3792_379279


namespace NUMINAMATH_CALUDE_transformation_result_l3792_379215

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation φ
def φ (x y : ℝ) : ℝ × ℝ := (3*x, 4*y)

-- Define the new curve
def new_curve (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 16) = 1

-- Theorem statement
theorem transformation_result :
  ∀ (x y : ℝ), original_curve x y → new_curve (φ x y).1 (φ x y).2 :=
by
  sorry

end NUMINAMATH_CALUDE_transformation_result_l3792_379215


namespace NUMINAMATH_CALUDE_hoseok_additional_jumps_theorem_l3792_379259

/-- The number of additional jumps Hoseok needs to match Minyoung's total -/
def additional_jumps (hoseok_jumps minyoung_jumps : ℕ) : ℕ :=
  minyoung_jumps - hoseok_jumps

theorem hoseok_additional_jumps_theorem (hoseok_jumps minyoung_jumps : ℕ) 
    (h : minyoung_jumps > hoseok_jumps) :
  additional_jumps hoseok_jumps minyoung_jumps = 17 :=
by
  sorry

#eval additional_jumps 34 51

end NUMINAMATH_CALUDE_hoseok_additional_jumps_theorem_l3792_379259


namespace NUMINAMATH_CALUDE_chris_age_l3792_379276

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 12 →
  c - 5 = 2 * a →
  b + 2 = (a + 2) / 2 →
  c = 163 / 7 := by
sorry

end NUMINAMATH_CALUDE_chris_age_l3792_379276


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3792_379227

def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 = 7}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3792_379227


namespace NUMINAMATH_CALUDE_elsa_token_count_l3792_379288

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := 55

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in dollar value between Elsa's and Angus's tokens -/
def value_difference : ℕ := 20

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

theorem elsa_token_count : elsa_tokens = 60 := by
  sorry

end NUMINAMATH_CALUDE_elsa_token_count_l3792_379288


namespace NUMINAMATH_CALUDE_hair_length_calculation_l3792_379280

theorem hair_length_calculation (initial_length cut_length growth_length : ℕ) 
  (h1 : initial_length = 16)
  (h2 : cut_length = 11)
  (h3 : growth_length = 12) :
  initial_length - cut_length + growth_length = 17 := by
  sorry

end NUMINAMATH_CALUDE_hair_length_calculation_l3792_379280


namespace NUMINAMATH_CALUDE_base_13_conversion_l3792_379232

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its decimal value -/
def toDecimal (d : Base13Digit) : Nat :=
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

/-- Converts a three-digit base 13 number to its decimal equivalent -/
def base13ToDecimal (d1 d2 d3 : Base13Digit) : Nat :=
  (toDecimal d1) * 169 + (toDecimal d2) * 13 + (toDecimal d3)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.D1 Base13Digit.D2 Base13Digit.D1 = 196 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l3792_379232


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3792_379254

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 1) ≤ 0}
def B : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3792_379254


namespace NUMINAMATH_CALUDE_f_has_one_root_l3792_379233

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x - 2

theorem f_has_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_one_root_l3792_379233


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l3792_379278

theorem crayons_lost_or_given_away (start_crayons end_crayons : ℕ) 
  (h1 : start_crayons = 253)
  (h2 : end_crayons = 183) :
  start_crayons - end_crayons = 70 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l3792_379278


namespace NUMINAMATH_CALUDE_honda_cars_sold_l3792_379238

/-- Represents the total number of cars sold -/
def total_cars : ℕ := 300

/-- Represents the percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- Represents the percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- Represents the percentage of Acura cars sold -/
def acura_percent : ℚ := 30 / 100

/-- Represents the percentage of BMW cars sold -/
def bmw_percent : ℚ := 15 / 100

/-- Theorem stating that the number of Honda cars sold is 75 -/
theorem honda_cars_sold : 
  (total_cars : ℚ) * (1 - (audi_percent + toyota_percent + acura_percent + bmw_percent)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_honda_cars_sold_l3792_379238


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l3792_379294

theorem merchant_pricing_strategy (L : ℝ) (h : L > 0) :
  let purchase_price := L * 0.7
  let marked_price := L * 1.25
  let selling_price := marked_price * 0.8
  selling_price = purchase_price * 1.3 := by
sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l3792_379294


namespace NUMINAMATH_CALUDE_scout_saturday_customers_l3792_379252

/-- Scout's delivery earnings over a weekend --/
def scout_earnings (base_pay hourly_rate tip_rate : ℚ) 
                   (saturday_hours sunday_hours : ℚ) 
                   (sunday_customers : ℕ) 
                   (total_earnings : ℚ) : Prop :=
  let saturday_base := base_pay * saturday_hours
  let sunday_base := base_pay * sunday_hours
  let sunday_tips := tip_rate * sunday_customers
  ∃ saturday_customers : ℕ,
    saturday_base + sunday_base + sunday_tips + (tip_rate * saturday_customers) = total_earnings

theorem scout_saturday_customers :
  scout_earnings 10 10 5 4 5 8 155 →
  ∃ saturday_customers : ℕ, saturday_customers = 5 :=
by sorry

end NUMINAMATH_CALUDE_scout_saturday_customers_l3792_379252


namespace NUMINAMATH_CALUDE_fruit_drink_composition_l3792_379224

/-- Represents a fruit drink mixture -/
structure FruitDrink where
  total_volume : ℝ
  grape_volume : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ

/-- The theorem statement -/
theorem fruit_drink_composition (drink : FruitDrink)
  (h1 : drink.total_volume = 150)
  (h2 : drink.grape_volume = 45)
  (h3 : drink.orange_percent = drink.watermelon_percent)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 100)
  (h5 : drink.grape_volume / drink.total_volume * 100 = drink.grape_percent) :
  drink.orange_percent = 35 ∧ drink.watermelon_percent = 35 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_composition_l3792_379224


namespace NUMINAMATH_CALUDE_prism_pyramid_sum_l3792_379242

/-- Represents a shape formed by adding a pyramid to one rectangular face of a right rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior faces, edges, and vertices of the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces + shape.pyramid_new_faces - 1) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating that the sum of exterior faces, edges, and vertices is 34 -/
theorem prism_pyramid_sum :
  ∀ (shape : PrismPyramid),
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces = 4 ∧
    shape.pyramid_new_edges = 4 ∧
    shape.pyramid_new_vertex = 1 →
    total_elements shape = 34 := by
  sorry

end NUMINAMATH_CALUDE_prism_pyramid_sum_l3792_379242


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3792_379253

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (3 * n) ≤ Real.sqrt (5 * n - 8) ∧
                                Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
                     S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3792_379253


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3792_379208

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := Real.sqrt 3 * x = 2 * y ∨ Real.sqrt 3 * x = -2 * y

-- Theorem statement
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3792_379208


namespace NUMINAMATH_CALUDE_tan_2a_values_l3792_379211

theorem tan_2a_values (a : ℝ) (h : 2 * Real.sin (2 * a) = 1 + Real.cos (2 * a)) :
  Real.tan (2 * a) = 4 / 3 ∨ Real.tan (2 * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2a_values_l3792_379211


namespace NUMINAMATH_CALUDE_product_equation_l3792_379225

theorem product_equation : 935420 * 625 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_equation_l3792_379225


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3792_379239

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3792_379239


namespace NUMINAMATH_CALUDE_clara_hughes_book_purchase_l3792_379245

theorem clara_hughes_book_purchase :
  let total_volumes : ℕ := 12
  let paperback_cost : ℕ := 18
  let hardcover_cost : ℕ := 28
  let total_spent : ℕ := 276
  ∃ (hardcover_count : ℕ) (paperback_count : ℕ),
    hardcover_count + paperback_count = total_volumes ∧
    hardcover_cost * hardcover_count + paperback_cost * paperback_count = total_spent ∧
    hardcover_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_clara_hughes_book_purchase_l3792_379245


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3792_379229

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b (F₂.1) (F₂.2) ∧ 
  F₁.1 < F₂.1 ∧
  F₁.2 = F₂.2

-- Define the circle with diameter F₁F₂
def circle_diameter (F₁ F₂ P : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 / 4 + (F₂.2 - F₁.2)^2 / 4

-- Define the intersection of PF₁ and the hyperbola
def intersection (P Q F₁ : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ t : ℝ, 
    Q.1 = F₁.1 + t * (P.1 - F₁.1) ∧
    Q.2 = F₁.2 + t * (P.2 - F₁.2) ∧
    hyperbola a b Q.1 Q.2

-- Define the distance condition
def distance_condition (P Q F₁ : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2)

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) :
  hyperbola a b P.1 P.2 →
  foci F₁ F₂ a b →
  circle_diameter F₁ F₂ P →
  intersection P Q F₁ a b →
  distance_condition P Q F₁ →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3792_379229


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l3792_379218

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2000 ∧ n = 9 * sum_of_digits n

theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_condition_l3792_379218


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3792_379214

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n + 4 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 8 * r - 1

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l3792_379214


namespace NUMINAMATH_CALUDE_unique_solution_l3792_379220

def base7_to_decimal (a b : Nat) : Nat := 7 * a + b

theorem unique_solution :
  ∀ P Q R : Nat,
    P ≠ 0 ∧ Q ≠ 0 ∧ R ≠ 0 →
    P < 7 ∧ Q < 7 ∧ R < 7 →
    P ≠ Q ∧ P ≠ R ∧ Q ≠ R →
    base7_to_decimal P Q + R = base7_to_decimal R 0 →
    base7_to_decimal P Q + base7_to_decimal Q P = base7_to_decimal R R →
    P = 4 ∧ Q = 3 ∧ R = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3792_379220


namespace NUMINAMATH_CALUDE_combination_sum_theorem_l3792_379289

theorem combination_sum_theorem : 
  ∃ (n : ℕ+), 
    (0 ≤ 38 - n.val ∧ 38 - n.val ≤ 3 * n.val) ∧ 
    (n.val + 21 ≥ 3 * n.val) ∧ 
    (Nat.choose (3 * n.val) (38 - n.val) + Nat.choose (n.val + 21) (3 * n.val) = 466) := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_theorem_l3792_379289


namespace NUMINAMATH_CALUDE_linear_function_negative_slope_l3792_379299

/-- Given a linear function y = kx + b passing through points A(1, m) and B(-1, n), 
    where m < n and k ≠ 0, prove that k < 0. -/
theorem linear_function_negative_slope (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : m < n)
  (h3 : m = k + b)  -- Point A(1, m) satisfies the equation
  (h4 : n = -k + b) -- Point B(-1, n) satisfies the equation
  : k < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_negative_slope_l3792_379299


namespace NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l3792_379281

/-- A fifteen-digit integer formed by repeating a five-digit integer three times -/
def repeatedNumber (n : ℕ) : ℕ := n * 10000100001

/-- The set of all such fifteen-digit numbers -/
def S : Set ℕ := {m : ℕ | ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ m = repeatedNumber n}

theorem gcd_of_repeated_numbers : 
  ∃ d : ℕ, d > 0 ∧ ∀ m ∈ S, d ∣ m ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ∣ d :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_numbers_l3792_379281


namespace NUMINAMATH_CALUDE_triangle_properties_l3792_379241

/-- Given a triangle ABC with the following properties:
    1. (1 - tan A)(1 - tan B) = 2
    2. b = 2√2
    3. c = 4
    Prove that:
    1. Angle C = π/4
    2. Area of triangle ABC = 2√3 + 2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  (1 - Real.tan A) * (1 - Real.tan B) = 2 →
  b = 2 * Real.sqrt 2 →
  c = 4 →
  C = π / 4 ∧
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3792_379241


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l3792_379298

theorem min_value_complex_expression (a b c : ℤ) (ξ : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    Complex.abs (↑x + ↑y * ξ + ↑z * ξ^3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l3792_379298


namespace NUMINAMATH_CALUDE_set_equality_l3792_379219

def I : Set Char := {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
def M : Set Char := {'c', 'd', 'e'}
def N : Set Char := {'a', 'c', 'f'}

theorem set_equality : (I \ M) ∩ (I \ N) = {'b', 'g'} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3792_379219


namespace NUMINAMATH_CALUDE_solution_is_ray_iff_a_is_pm1_l3792_379256

/-- The polynomial function in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions for the inequality -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- Definition of a ray (half-line) in ℝ -/
def is_ray (S : Set ℝ) : Prop :=
  ∃ (x₀ : ℝ), S = {x : ℝ | x ≥ x₀} ∨ S = {x : ℝ | x ≤ x₀}

/-- The main theorem -/
theorem solution_is_ray_iff_a_is_pm1 :
  ∀ a : ℝ, is_ray (solution_set a) ↔ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_solution_is_ray_iff_a_is_pm1_l3792_379256


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l3792_379228

-- Define the number and side length of Carl's cubes
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3

-- Define the number and side length of Kate's cubes
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Function to calculate the volume of a cube
def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

-- Theorem statement
theorem total_volume_of_cubes :
  carl_cubes * cube_volume carl_side_length + kate_cubes * cube_volume kate_side_length = 114 := by
  sorry


end NUMINAMATH_CALUDE_total_volume_of_cubes_l3792_379228


namespace NUMINAMATH_CALUDE_cube_root_cube_equality_l3792_379267

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_cube_equality_l3792_379267


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3792_379265

theorem quadratic_inequality_solution (x : ℝ) :
  -4 * x^2 + 7 * x + 2 < 0 ↔ x < -1/4 ∨ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3792_379265


namespace NUMINAMATH_CALUDE_mart_income_percentage_of_juan_l3792_379213

/-- Represents the income relationships between Tim, Mart, Juan, and Alex -/
structure IncomeRelationships where
  tim : ℝ
  mart : ℝ
  juan : ℝ
  alex : ℝ
  mart_tim_ratio : mart = 1.6 * tim
  tim_juan_ratio : tim = 0.6 * juan
  alex_mart_ratio : alex = 1.25 * mart
  juan_alex_ratio : juan = 1.2 * alex

/-- Theorem stating that Mart's income is 96% of Juan's income -/
theorem mart_income_percentage_of_juan (ir : IncomeRelationships) :
  ir.mart = 0.96 * ir.juan := by
  sorry

end NUMINAMATH_CALUDE_mart_income_percentage_of_juan_l3792_379213


namespace NUMINAMATH_CALUDE_square_side_length_l3792_379244

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 625 →
  side * side = area →
  side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3792_379244


namespace NUMINAMATH_CALUDE_jasmine_milk_purchase_jasmine_milk_purchase_holds_l3792_379205

/-- Proves that Jasmine bought 2 gallons of milk given the problem conditions -/
theorem jasmine_milk_purchase : ℝ → Prop :=
  fun gallons_of_milk =>
    let coffee_pounds : ℝ := 4
    let coffee_price_per_pound : ℝ := 2.5
    let milk_price_per_gallon : ℝ := 3.5
    let total_cost : ℝ := 17
    coffee_pounds * coffee_price_per_pound + gallons_of_milk * milk_price_per_gallon = total_cost →
    gallons_of_milk = 2

/-- The theorem holds -/
theorem jasmine_milk_purchase_holds : jasmine_milk_purchase 2 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_milk_purchase_jasmine_milk_purchase_holds_l3792_379205


namespace NUMINAMATH_CALUDE_sports_meeting_medals_l3792_379293

/-- The number of medals awarded on day x -/
def f (x m : ℕ) : ℚ :=
  if x = 1 then
    1 + (m - 1) / 7
  else
    (6 / 7) ^ (x - 1) * ((m - 36) / 7) + 6

/-- The total number of medals awarded over n days -/
def total_medals (n : ℕ) : ℕ := 36

/-- The number of days the sports meeting lasted -/
def meeting_duration : ℕ := 6

theorem sports_meeting_medals :
  ∀ n m : ℕ,
  (∀ i : ℕ, i < n → f i m = i + (f (i+1) m - i) / 7) →
  f n m = n →
  n = meeting_duration ∧ m = total_medals n :=
sorry

end NUMINAMATH_CALUDE_sports_meeting_medals_l3792_379293


namespace NUMINAMATH_CALUDE_house_transaction_problem_l3792_379236

/-- Represents a person's assets -/
structure Assets where
  cash : Int
  has_house : Bool

/-- Represents a transaction between two people -/
def transaction (seller buyer : Assets) (price : Int) : Assets × Assets :=
  ({ cash := seller.cash + price, has_house := false },
   { cash := buyer.cash - price, has_house := true })

/-- The problem statement -/
theorem house_transaction_problem :
  let initial_a : Assets := { cash := 15000, has_house := true }
  let initial_b : Assets := { cash := 20000, has_house := false }
  let house_value := 15000

  let (a1, b1) := transaction initial_a initial_b 18000
  let (b2, a2) := transaction b1 a1 12000
  let (a3, b3) := transaction a2 b2 16000

  (a3.cash - initial_a.cash = 22000) ∧
  (b3.cash + house_value - initial_b.cash = -7000) := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_problem_l3792_379236


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3792_379243

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 63) (hb : b = 84) (hc : c = 105) :
  Nat.gcd a (Nat.gcd b c) = 21 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3792_379243


namespace NUMINAMATH_CALUDE_nth_equation_and_specific_case_l3792_379277

theorem nth_equation_and_specific_case :
  (∀ n : ℕ, n > 0 → Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n) ∧
  Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_and_specific_case_l3792_379277


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3792_379285

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 - 3*x^2 - 8*x + 40 - 8*(4*x + 4)^(1/4) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3792_379285


namespace NUMINAMATH_CALUDE_dining_bill_share_l3792_379255

theorem dining_bill_share (total_bill : ℝ) (num_people : ℕ) (tip_percentage : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ tip_percentage = 0.1 →
  (total_bill * (1 + tip_percentage)) / num_people = 30.58 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l3792_379255


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3792_379269

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 230) : x + y = 660 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3792_379269


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l3792_379264

/-- The coefficient of x^2 in the expansion of (x^2+x+1)(1-x)^6 is 10 -/
theorem x_squared_coefficient : Int := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l3792_379264


namespace NUMINAMATH_CALUDE_selection_problem_l3792_379223

theorem selection_problem (n_boys : ℕ) (n_girls : ℕ) : 
  n_boys = 4 → n_girls = 3 → 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 := by
sorry

end NUMINAMATH_CALUDE_selection_problem_l3792_379223


namespace NUMINAMATH_CALUDE_second_platform_length_l3792_379251

/-- Calculates the length of a second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 310)
  (h2 : first_platform_length = 110)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * (train_length + first_platform_length) / first_crossing_time) - train_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l3792_379251


namespace NUMINAMATH_CALUDE_another_divisor_of_increased_number_l3792_379230

theorem another_divisor_of_increased_number : ∃ (n : ℕ), n ≠ 12 ∧ n ≠ 30 ∧ n ≠ 74 ∧ n ≠ 100 ∧ (44402 + 2) % n = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_another_divisor_of_increased_number_l3792_379230


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3792_379287

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of forming an increasing arithmetic sequence
def is_increasing_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

-- State the theorem
theorem fibonacci_arithmetic_sequence :
  ∃ (a b c : ℕ),
    is_increasing_arithmetic_seq a b c ∧
    a + b + c = 2000 ∧
    a = 665 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3792_379287


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3792_379270

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 4*x^3 - 5*x + 2
def q (x : ℝ) : ℝ := 3*x^4 - x^3 + x^2 + 4*x - 1

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-18)*x^2 + f :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3792_379270


namespace NUMINAMATH_CALUDE_volume_submerged_object_iron_block_volume_l3792_379222

/-- The volume of a submerged object in a cylindrical container --/
theorem volume_submerged_object
  (r h₁ h₂ : ℝ)
  (hr : r > 0)
  (hh : h₂ > h₁) :
  let V := π * r^2 * (h₂ - h₁)
  V = π * r^2 * h₂ - π * r^2 * h₁ :=
by sorry

/-- The volume of the irregular iron block --/
theorem iron_block_volume
  (r h₁ h₂ : ℝ)
  (hr : r = 5)
  (hh₁ : h₁ = 6)
  (hh₂ : h₂ = 8) :
  π * r^2 * (h₂ - h₁) = 50 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_submerged_object_iron_block_volume_l3792_379222


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3792_379234

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_8_factorial_10_factorial : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3792_379234


namespace NUMINAMATH_CALUDE_fraction_sum_and_product_l3792_379271

theorem fraction_sum_and_product (x y : ℚ) :
  x + y = 13/14 ∧ x * y = 3/28 →
  (x = 3/7 ∧ y = 1/4) ∨ (x = 1/4 ∧ y = 3/7) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_and_product_l3792_379271


namespace NUMINAMATH_CALUDE_scientific_notation_657000_l3792_379207

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_657000 :
  toScientificNotation 657000 = ScientificNotation.mk 6.57 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_657000_l3792_379207


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3792_379275

def grape_quantity : ℝ := 7
def grape_rate : ℝ := 70
def mango_quantity : ℝ := 9
def total_paid : ℝ := 985

theorem mango_rate_calculation :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 :=
by sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3792_379275


namespace NUMINAMATH_CALUDE_lcm_15_18_20_l3792_379203

theorem lcm_15_18_20 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_15_18_20_l3792_379203


namespace NUMINAMATH_CALUDE_coordinates_of_point_A_l3792_379286

-- Define a Point type for 2D coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem coordinates_of_point_A (A B : Point) : 
  (B.x = 2 ∧ B.y = 4) →  -- Coordinates of point B
  (A.y = B.y) →  -- AB is parallel to x-axis
  ((A.x - B.x)^2 + (A.y - B.y)^2 = 3^2) →  -- Length of AB is 3
  ((A.x = 5 ∧ A.y = 4) ∨ (A.x = -1 ∧ A.y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_A_l3792_379286


namespace NUMINAMATH_CALUDE_only_D_greater_than_one_l3792_379272

theorem only_D_greater_than_one : 
  (0 / 0.16 ≤ 1) ∧ (1 * 0.16 ≤ 1) ∧ (1 / 1.6 ≤ 1) ∧ (1 * 1.6 > 1) := by
  sorry

end NUMINAMATH_CALUDE_only_D_greater_than_one_l3792_379272


namespace NUMINAMATH_CALUDE_area_outside_small_inside_large_l3792_379297

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region outside the smaller circle and inside the two larger circles -/
def areaOutsideSmallInsideLarge (smallCircle : Circle) (largeCircle1 largeCircle2 : Circle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific configuration -/
theorem area_outside_small_inside_large :
  let smallCircle : Circle := { center := (0, 0), radius := 2 }
  let largeCircle1 : Circle := { center := (0, -2), radius := 3 }
  let largeCircle2 : Circle := { center := (0, 2), radius := 3 }
  areaOutsideSmallInsideLarge smallCircle largeCircle1 largeCircle2 = (5 * Real.pi / 2) - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_small_inside_large_l3792_379297


namespace NUMINAMATH_CALUDE_no_consecutive_triples_sum_squares_equal_repeating_digit_l3792_379204

theorem no_consecutive_triples_sum_squares_equal_repeating_digit : 
  ¬ ∃ (n a : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (n-1)^2 + n^2 + (n+1)^2 = 1111 * a := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_triples_sum_squares_equal_repeating_digit_l3792_379204


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l3792_379249

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  2 * (-2)^2 + 2 * |-2| + 7 < 25 :=
by
  sorry

theorem least_integer_is_negative_two :
  ∃ x : ℤ, (2 * x^2 + 2 * |x| + 7 < 25) ∧ 
    (∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ x) ∧
    x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l3792_379249


namespace NUMINAMATH_CALUDE_philip_monthly_mileage_l3792_379206

/-- Represents Philip's driving routine and calculates the total monthly mileage -/
def philipDrivingMileage : ℕ :=
  let schoolRoundTrip : ℕ := 5 /- 2.5 * 2, rounded to nearest integer -/
  let workOneWay : ℕ := 8
  let marketRoundTrip : ℕ := 2
  let gymRoundTrip : ℕ := 4
  let friendRoundTrip : ℕ := 6
  let weekdayMileage : ℕ := (schoolRoundTrip * 2 + workOneWay * 2) * 5
  let saturdayMileage : ℕ := marketRoundTrip + gymRoundTrip + friendRoundTrip
  let weeklyMileage : ℕ := weekdayMileage + saturdayMileage
  let weeksInMonth : ℕ := 4
  weeklyMileage * weeksInMonth

/-- Theorem stating that Philip's total monthly mileage is 468 miles -/
theorem philip_monthly_mileage : philipDrivingMileage = 468 := by
  sorry

end NUMINAMATH_CALUDE_philip_monthly_mileage_l3792_379206


namespace NUMINAMATH_CALUDE_money_made_washing_cars_l3792_379235

def initial_amount : ℕ := 74
def current_amount : ℕ := 86

theorem money_made_washing_cars :
  current_amount - initial_amount = 12 :=
by sorry

end NUMINAMATH_CALUDE_money_made_washing_cars_l3792_379235


namespace NUMINAMATH_CALUDE_expected_coincidences_value_l3792_379295

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- Vasya's probability of guessing correctly -/
def p_vasya : ℚ := 6 / 20

/-- Misha's probability of guessing correctly -/
def p_misha : ℚ := 8 / 20

/-- The probability of a coincidence (both correct or both incorrect) for a single question -/
def p_coincidence : ℚ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
def expected_coincidences : ℚ := num_questions * p_coincidence

theorem expected_coincidences_value :
  expected_coincidences = 54 / 5 := by sorry

end NUMINAMATH_CALUDE_expected_coincidences_value_l3792_379295


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3792_379268

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 3 * p^2 + 6 * p - 9 = 0) →
  (3 * q^3 - 3 * q^2 + 6 * q - 9 = 0) →
  (3 * r^3 - 3 * r^2 + 6 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3792_379268
