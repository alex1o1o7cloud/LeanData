import Mathlib

namespace NUMINAMATH_CALUDE_fifth_day_income_l1964_196484

def cab_driver_income (income_4_days : List ℝ) (average_income : ℝ) : ℝ :=
  5 * average_income - income_4_days.sum

theorem fifth_day_income 
  (income_4_days : List ℝ) 
  (average_income : ℝ) 
  (h1 : income_4_days.length = 4) 
  (h2 : average_income = (income_4_days.sum + cab_driver_income income_4_days average_income) / 5) :
  cab_driver_income income_4_days average_income = 
    5 * average_income - income_4_days.sum :=
by
  sorry

#eval cab_driver_income [300, 150, 750, 200] 400

end NUMINAMATH_CALUDE_fifth_day_income_l1964_196484


namespace NUMINAMATH_CALUDE_lars_bakery_production_l1964_196489

-- Define the baking rates and working hours
def bread_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def working_hours : ℕ := 6

-- Define the function to calculate total breads per day
def total_breads_per_day : ℕ :=
  (bread_per_hour * working_hours) + (baguettes_per_two_hours * (working_hours / 2))

-- Theorem statement
theorem lars_bakery_production :
  total_breads_per_day = 150 := by sorry

end NUMINAMATH_CALUDE_lars_bakery_production_l1964_196489


namespace NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_eight_l1964_196487

theorem three_digit_cubes_divisible_by_eight :
  (∃! (s : Finset Nat), 
    (∀ n ∈ s, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k, n = k^3 ∧ 8 ∣ n) ∧ 
    s.card = 2) :=
sorry

end NUMINAMATH_CALUDE_three_digit_cubes_divisible_by_eight_l1964_196487


namespace NUMINAMATH_CALUDE_composition_value_l1964_196480

theorem composition_value (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c)
  (hg : ∀ x, g x = c*x + 3)
  (h_comp : ∀ x, f (g x) = 15*x + d) : 
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_composition_value_l1964_196480


namespace NUMINAMATH_CALUDE_curve_C_range_l1964_196493

/-- The curve C is defined by the equation x^2 + y^2 + 2ax - 4ay + 5a^2 - 4 = 0 -/
def C (a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: If all points on curve C are in the second quadrant, then a > 2 -/
theorem curve_C_range (a : ℝ) :
  (∀ x y : ℝ, C a x y → second_quadrant x y) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_range_l1964_196493


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l1964_196474

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 6| + |y + 5| = 0 → x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l1964_196474


namespace NUMINAMATH_CALUDE_range_of_m_l1964_196426

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m)) → 
  (-3 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1964_196426


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_and_convergence_l1964_196453

theorem no_real_solution_for_equation_and_convergence : 
  ¬∃ y : ℝ, y = 2 / (1 + y) ∧ abs y < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_and_convergence_l1964_196453


namespace NUMINAMATH_CALUDE_range_of_m_specific_m_value_l1964_196430

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m - 1

-- Define the condition for two real roots
def has_two_real_roots (m : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) (h : has_two_real_roots m) : m ≤ 2 := by sorry

-- Theorem for the specific value of m
theorem specific_m_value (m : ℝ) (h : has_two_real_roots m) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m = 0 ∧ quadratic_equation x₂ m = 0 ∧ x₁^2 + x₂^2 = 6*x₁*x₂) →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_range_of_m_specific_m_value_l1964_196430


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_l1964_196412

theorem sum_consecutive_odd_integers (n : ℕ) : n = 1 →
  (List.range (2 * n + 2)).sum = (n + 1) * (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_l1964_196412


namespace NUMINAMATH_CALUDE_division_problem_l1964_196445

theorem division_problem (x : ℝ) : 25.25 / x = 0.012625 → x = 2000 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1964_196445


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a3_l1964_196423

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ b : ℕ → ℝ, is_geometric_sequence b → (∀ n : ℕ, b n > 0) → b 2 - b 1 = 1 → a 3 ≤ b 3) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a3_l1964_196423


namespace NUMINAMATH_CALUDE_cos_six_arccos_two_fifths_l1964_196402

theorem cos_six_arccos_two_fifths :
  Real.cos (6 * Real.arccos (2/5)) = 12223/15625 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_two_fifths_l1964_196402


namespace NUMINAMATH_CALUDE_root_product_sum_l1964_196460

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (Real.sqrt 2015) * x₁^3 - 4030 * x₁^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₂^3 - 4030 * x₂^2 + 2 = 0 ∧
  (Real.sqrt 2015) * x₃^3 - 4030 * x₃^2 + 2 = 0 →
  x₂ * (x₁ + x₃) = 2 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1964_196460


namespace NUMINAMATH_CALUDE_min_value_theorem_l1964_196406

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1964_196406


namespace NUMINAMATH_CALUDE_assignment_plans_l1964_196454

theorem assignment_plans (n_females : ℕ) (n_males : ℕ) (n_positions : ℕ) 
  (h_females : n_females = 10)
  (h_males : n_males = 40)
  (h_positions : n_positions = 5) :
  (n_females.choose 2) * 3 * 24 * (n_males.choose 3) = 
    Nat.choose n_females 2 * (Nat.factorial 3 / Nat.factorial 2) * 
    (Nat.factorial 4 / Nat.factorial 0) * Nat.choose n_males 3 :=
by sorry

end NUMINAMATH_CALUDE_assignment_plans_l1964_196454


namespace NUMINAMATH_CALUDE_family_movie_night_l1964_196475

/-- Proves the number of adults in a family given ticket prices and payment information -/
theorem family_movie_night (regular_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_children : ℕ) : 
  regular_price = 9 →
  child_discount = 2 →
  total_payment = 40 →
  change = 1 →
  num_children = 3 →
  (total_payment - change - num_children * (regular_price - child_discount)) / regular_price = 2 := by
sorry

end NUMINAMATH_CALUDE_family_movie_night_l1964_196475


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1964_196486

theorem trigonometric_equation_solution (x : ℝ) : 
  Real.cos (7 * x) + Real.sin (8 * x) = Real.cos (3 * x) - Real.sin (2 * x) → 
  (∃ n : ℤ, x = n * Real.pi / 5) ∨ 
  (∃ k : ℤ, x = Real.pi / 2 * (4 * k - 1)) ∨ 
  (∃ l : ℤ, x = Real.pi / 10 * (4 * l + 1)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1964_196486


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l1964_196447

theorem equivalence_of_statements (p q : Prop) :
  (¬p ∧ ¬q → p ∨ q) ↔ (p ∧ ¬q ∨ ¬p ∧ q) := by sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l1964_196447


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l1964_196470

-- Define a quadrilateral in a plane
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define vector equality
def vector_equal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 = v2.1 ∧ v1.2 = v2.2

-- Define vector scaling
def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

-- Define condition p
def condition_p (q : Quadrilateral) : Prop :=
  vector_equal (q.B.1 - q.A.1, q.B.2 - q.A.2) (vector_scale 2 (q.C.1 - q.D.1, q.C.2 - q.D.2))

-- Define a trapezoid
def is_trapezoid (q : Quadrilateral) : Prop :=
  (q.A.2 - q.B.2) / (q.A.1 - q.B.1) = (q.D.2 - q.C.2) / (q.D.1 - q.C.1) ∨
  (q.A.2 - q.D.2) / (q.A.1 - q.D.1) = (q.B.2 - q.C.2) / (q.B.1 - q.C.1)

-- Theorem statement
theorem condition_p_sufficient_not_necessary (q : Quadrilateral) :
  (condition_p q → is_trapezoid q) ∧ ¬(is_trapezoid q → condition_p q) :=
sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l1964_196470


namespace NUMINAMATH_CALUDE_population_growth_l1964_196419

theorem population_growth (x : ℝ) : 
  (((1 + x / 100) * 4) - 1) * 100 = 1100 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l1964_196419


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l1964_196488

theorem quadratic_solution_set (b c : ℝ) : 
  (∀ x, x^2 + 2*b*x + c ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) → b + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l1964_196488


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l1964_196498

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_strictly_increasing :
  StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l1964_196498


namespace NUMINAMATH_CALUDE_remaining_typing_orders_l1964_196485

/-- The number of letters in total -/
def totalLetters : ℕ := 10

/-- The label of the letter that has been typed by midday -/
def typedLetter : ℕ := 9

/-- The number of different orders for typing the remaining letters -/
def typingOrders : ℕ := 1280

/-- 
Theorem: Given 10 letters labeled from 1 to 10, where letter 9 has been typed by midday,
the number of different orders for typing the remaining letters is 1280.
-/
theorem remaining_typing_orders :
  (totalLetters = 10) →
  (typedLetter = 9) →
  (typingOrders = 1280) :=
by sorry

end NUMINAMATH_CALUDE_remaining_typing_orders_l1964_196485


namespace NUMINAMATH_CALUDE_interval_length_theorem_l1964_196499

theorem interval_length_theorem (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 2*x + 3 ∧ 2*x + 3 ≤ b) ∧ 
  ((b - 3) / 2 - (a - 3) / 2 = 10) → 
  b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_interval_length_theorem_l1964_196499


namespace NUMINAMATH_CALUDE_inequality_proof_l1964_196448

theorem inequality_proof (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  2 * (a + b + c) * (a^2 + b^2 + c^2) / 3 > a^3 + b^3 + c^3 + a*b*c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1964_196448


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l1964_196408

theorem arctan_sum_equals_pi_fourth (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l1964_196408


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1964_196403

theorem complex_fraction_equality : ∃ z : ℂ, z = (2 - I) / (1 - I) ∧ z = (3/2 : ℂ) + (1/2 : ℂ) * I :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1964_196403


namespace NUMINAMATH_CALUDE_range_of_a_l1964_196425

open Set

def p (a : ℝ) : Prop := a ≤ -2 ∨ a ≥ 2
def q (a : ℝ) : Prop := a ≥ -10

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ Iio (-10) ∪ Ioo (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1964_196425


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1964_196482

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1964_196482


namespace NUMINAMATH_CALUDE_perfect_squares_between_200_and_600_l1964_196407

theorem perfect_squares_between_200_and_600 :
  (Finset.filter (fun n => 200 < n^2 ∧ n^2 < 600) (Finset.range 25)).card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_200_and_600_l1964_196407


namespace NUMINAMATH_CALUDE_brendan_weekly_taxes_l1964_196471

/-- Calculates Brendan's weekly taxes paid after deduction -/
def weekly_taxes_paid (wage1 wage2 wage3 : ℚ) 
                      (hours1 hours2 hours3 : ℚ)
                      (tips1 tips2 tips3 : ℚ)
                      (reported_tips1 reported_tips2 reported_tips3 : ℚ)
                      (tax_rate1 tax_rate2 tax_rate3 : ℚ)
                      (deduction : ℚ) : ℚ :=
  let income1 := wage1 * hours1 + reported_tips1 * tips1 * hours1
  let income2 := wage2 * hours2 + reported_tips2 * tips2 * hours2
  let income3 := wage3 * hours3 + reported_tips3 * tips3 * hours3
  let taxes1 := income1 * tax_rate1
  let taxes2 := income2 * tax_rate2
  let taxes3 := income3 * tax_rate3
  taxes1 + taxes2 + taxes3 - deduction

theorem brendan_weekly_taxes :
  weekly_taxes_paid 12 15 10    -- wages
                    12 8 10     -- hours
                    20 15 5     -- tips
                    (1/2) (1/4) (3/5)  -- reported tips percentages
                    (22/100) (18/100) (16/100)  -- tax rates
                    50  -- deduction
  = 5588 / 100 := by
  sorry

end NUMINAMATH_CALUDE_brendan_weekly_taxes_l1964_196471


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1964_196401

theorem quadratic_distinct_roots (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄) :
  let discriminant := (a₁ + a₂ + a₃ + a₄)^2 - 4*(a₁*a₃ + a₂*a₄)
  discriminant > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1964_196401


namespace NUMINAMATH_CALUDE_sine_symmetry_l1964_196468

/-- Given a sinusoidal function y = 2sin(3x + φ) with |φ| < π/2,
    if the line of symmetry is x = π/12, then φ = π/4 -/
theorem sine_symmetry (φ : Real) : 
  (|φ| < π/2) →
  (∀ x : Real, 2 * Real.sin (3*x + φ) = 2 * Real.sin (3*(π/6 - x) + φ)) →
  φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_sine_symmetry_l1964_196468


namespace NUMINAMATH_CALUDE_original_movie_length_l1964_196459

/-- The original length of a movie, given the length of a cut scene and the final length -/
theorem original_movie_length (cut_scene_length final_length : ℕ) :
  cut_scene_length = 8 ∧ final_length = 52 →
  cut_scene_length + final_length = 60 := by
  sorry

#check original_movie_length

end NUMINAMATH_CALUDE_original_movie_length_l1964_196459


namespace NUMINAMATH_CALUDE_f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l1964_196483

/-- The quadratic function f(x) = x^2 + 15x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 15*x + 3

/-- Theorem stating that f(x) is minimized when x = -15/2 -/
theorem f_min_at_neg_15_div_2 :
  ∀ x : ℝ, f (-15/2) ≤ f x :=
by
  sorry

/-- Theorem stating that -15/2 is the unique minimizer of f(x) -/
theorem f_unique_min_at_neg_15_div_2 :
  ∀ x : ℝ, x ≠ -15/2 → f (-15/2) < f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l1964_196483


namespace NUMINAMATH_CALUDE_square_perimeter_proof_l1964_196439

theorem square_perimeter_proof (p1 p2 p3 : ℝ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24)
  (h4 : (p3 / 4) ^ 2 = ((p1 / 4) ^ 2) - ((p2 / 4) ^ 2)) : p1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_proof_l1964_196439


namespace NUMINAMATH_CALUDE_rectangle_area_l1964_196463

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its area is 156 cm² -/
theorem rectangle_area (perimeter width length : ℝ) : 
  perimeter = 50 → 
  length = 13 → 
  width = (perimeter - 2 * length) / 2 → 
  length * width = 156 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1964_196463


namespace NUMINAMATH_CALUDE_expression_value_l1964_196449

theorem expression_value : (2023 : ℚ) / 2022 - 2022 / 2023 + 1 = 4098551 / (2022 * 2023) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1964_196449


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1964_196472

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 23 ∧ x - y = 7 → x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1964_196472


namespace NUMINAMATH_CALUDE_one_alligator_per_week_l1964_196435

/-- The number of Burmese pythons -/
def num_pythons : ℕ := 5

/-- The number of alligators eaten in the given time period -/
def num_alligators : ℕ := 15

/-- The number of weeks in the given time period -/
def num_weeks : ℕ := 3

/-- The number of alligators one Burmese python can eat per week -/
def alligators_per_python_per_week : ℚ := num_alligators / (num_pythons * num_weeks)

theorem one_alligator_per_week : 
  alligators_per_python_per_week = 1 :=
sorry

end NUMINAMATH_CALUDE_one_alligator_per_week_l1964_196435


namespace NUMINAMATH_CALUDE_questionnaire_C_count_l1964_196461

def population : ℕ := 960
def sample_size : ℕ := 32
def first_number : ℕ := 9
def questionnaire_A_upper : ℕ := 450
def questionnaire_B_upper : ℕ := 750

theorem questionnaire_C_count :
  let group_size := population / sample_size
  let groups_AB := questionnaire_B_upper / group_size
  sample_size - groups_AB = 7 := by sorry

end NUMINAMATH_CALUDE_questionnaire_C_count_l1964_196461


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1964_196473

theorem inequality_system_solution :
  ∃ (x : ℤ),
    (3 * (2 * x - 1) < 2 * x + 8) ∧
    (2 + (3 * (x + 1)) / 8 > 3 - (x - 1) / 4) ∧
    (x = 2) ∧
    (∀ a : ℝ, (a * x + 6 ≤ x - 2 * a) → (|a + 1| - |a - 1| = -2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1964_196473


namespace NUMINAMATH_CALUDE_f_properties_l1964_196428

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 2 * Real.sin x

theorem f_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≤ max ∧ max = Real.sqrt 3) ∧
  (∃ (θ : ℝ), ∀ (x : ℝ), f x ≤ f θ ∧ Real.cos (θ - π/6) = (3 + Real.sqrt 6) / 6) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1964_196428


namespace NUMINAMATH_CALUDE_fraction_of_number_l1964_196400

theorem fraction_of_number (original : ℕ) (target : ℚ) : 
  original = 5040 → target = 756.0000000000001 → 
  (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * original = target := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l1964_196400


namespace NUMINAMATH_CALUDE_odd_prime_sqrt_integer_l1964_196490

theorem odd_prime_sqrt_integer (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k) : 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_sqrt_integer_l1964_196490


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l1964_196405

theorem sqrt_difference_equals_five_sixths :
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l1964_196405


namespace NUMINAMATH_CALUDE_total_crayons_l1964_196433

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 12) (h2 : num_children = 18) :
  crayons_per_child * num_children = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1964_196433


namespace NUMINAMATH_CALUDE_evaluate_expression_l1964_196479

theorem evaluate_expression : 
  (30 ^ 20 : ℝ) / (90 ^ 10) = 10 ^ 10 := by
  sorry

#check evaluate_expression

end NUMINAMATH_CALUDE_evaluate_expression_l1964_196479


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l1964_196410

/-- Given two points A and B as endpoints of a diameter of a circle,
    this theorem proves the equation of the circle. -/
theorem circle_equation_from_diameter_endpoints 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 4)) 
  (h_B : B = (3, -2)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 2)^2 + (y - 1)^2 = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l1964_196410


namespace NUMINAMATH_CALUDE_envelope_printing_equation_l1964_196415

/-- The equation for two envelope-printing machines to print 500 envelopes in 2 minutes -/
theorem envelope_printing_equation (x : ℝ) : x > 0 → 500 / 8 + 500 / x = 500 / 2 := by
  sorry

end NUMINAMATH_CALUDE_envelope_printing_equation_l1964_196415


namespace NUMINAMATH_CALUDE_infinitely_many_triples_l1964_196418

theorem infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b p : ℕ),
    Prime p ∧
    0 < a ∧ a ≤ b ∧ b < p ∧
    (p^5 ∣ (a + b)^p - a^p - b^p) ∧
    p > n :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_triples_l1964_196418


namespace NUMINAMATH_CALUDE_zoe_family_cost_l1964_196429

/-- The total cost of soda and pizza for a group --/
def total_cost (num_people : ℕ) (soda_cost pizza_cost : ℚ) : ℚ :=
  num_people * (soda_cost + pizza_cost)

/-- Theorem: The total cost for Zoe and her family is $9 --/
theorem zoe_family_cost : 
  total_cost 6 (1/2) 1 = 9 := by sorry

end NUMINAMATH_CALUDE_zoe_family_cost_l1964_196429


namespace NUMINAMATH_CALUDE_mens_wages_75_l1964_196420

/-- Represents the wage distribution problem -/
structure WageDistribution where
  men_count : ℕ
  boys_count : ℕ
  total_earnings : ℕ

/-- Calculates the men's wages given the wage distribution -/
def mens_wages (wd : WageDistribution) : ℕ :=
  (wd.total_earnings * wd.men_count) / (wd.men_count + wd.boys_count)

/-- Theorem stating that for the given conditions, the men's wages are 75 -/
theorem mens_wages_75 (wd : WageDistribution) 
  (h1 : wd.men_count = 5)
  (h2 : wd.boys_count = 8)
  (h3 : wd.total_earnings = 150) :
  mens_wages wd = 75 := by
  sorry

#eval mens_wages { men_count := 5, boys_count := 8, total_earnings := 150 }

end NUMINAMATH_CALUDE_mens_wages_75_l1964_196420


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1964_196495

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → r^2 + s^2 = p^2 - 2*q :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1964_196495


namespace NUMINAMATH_CALUDE_certain_number_problem_l1964_196424

theorem certain_number_problem (x : ℝ) : 
  (0.90 * x = 0.50 * 1080) → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1964_196424


namespace NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l1964_196436

theorem factorization_ax_squared_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ax_squared_minus_a_l1964_196436


namespace NUMINAMATH_CALUDE_unique_divisor_problem_l1964_196451

theorem unique_divisor_problem (dividend : Nat) (divisor : Nat) : 
  dividend = 12128316 →
  divisor * 7 < 1000 →
  divisor * 7 ≥ 100 →
  dividend % divisor = 0 →
  (∀ d : Nat, d ≠ divisor → 
    (d * 7 < 1000 ∧ d * 7 ≥ 100 ∧ dividend % d = 0) → False) →
  divisor = 124 := by
sorry

end NUMINAMATH_CALUDE_unique_divisor_problem_l1964_196451


namespace NUMINAMATH_CALUDE_power_of_seven_expansion_l1964_196497

theorem power_of_seven_expansion : 7^3 - 3*(7^2) + 3*7 - 1 = 216 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_expansion_l1964_196497


namespace NUMINAMATH_CALUDE_inequality_proof_l1964_196462

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  a * b > a * c ∧ c * b^2 < a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1964_196462


namespace NUMINAMATH_CALUDE_fill_fraction_in_three_minutes_l1964_196440

/-- Represents the fraction of a cistern filled in a given time -/
def fractionFilled (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem fill_fraction_in_three_minutes :
  let totalTime : ℚ := 33
  let minutes : ℚ := 3
  fractionFilled totalTime minutes = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fill_fraction_in_three_minutes_l1964_196440


namespace NUMINAMATH_CALUDE_total_amount_calculation_l1964_196442

theorem total_amount_calculation (part1 : ℝ) (part2 : ℝ) (total_interest : ℝ) :
  part1 = 1500.0000000000007 →
  part1 * 0.05 + part2 * 0.06 = 135 →
  part1 + part2 = 2500.000000000000 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l1964_196442


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1964_196476

theorem complex_modulus_product : Complex.abs ((10 - 7*I) * (9 + 11*I)) = Real.sqrt 30098 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1964_196476


namespace NUMINAMATH_CALUDE_train_length_l1964_196421

/-- Calculates the length of a train given its speed and the time it takes to pass through a tunnel of known length. -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 54 * 1000 / 3600 →
  tunnel_length = 1200 →
  time_to_pass = 100 →
  (train_speed * time_to_pass) - tunnel_length = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1964_196421


namespace NUMINAMATH_CALUDE_mystery_books_ratio_l1964_196469

def total_books : ℕ := 46
def top_section_books : ℕ := 12 + 8 + 4
def bottom_section_books : ℕ := total_books - top_section_books
def known_bottom_books : ℕ := 5 + 6
def mystery_books : ℕ := bottom_section_books - known_bottom_books

theorem mystery_books_ratio :
  (mystery_books : ℚ) / bottom_section_books = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mystery_books_ratio_l1964_196469


namespace NUMINAMATH_CALUDE_max_power_under_500_l1964_196491

theorem max_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 2) (h_less_500 : a^b < 500) :
  ∃ (a_max b_max : ℕ),
    a_max > 0 ∧ b_max > 2 ∧ a_max^b_max < 500 ∧
    ∀ (x y : ℕ), x > 0 → y > 2 → x^y < 500 → x^y ≤ a_max^b_max ∧
    a_max + b_max = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_power_under_500_l1964_196491


namespace NUMINAMATH_CALUDE_remaining_balance_l1964_196492

def house_price : ℝ := 100000

def down_payment_percentage : ℝ := 0.20

def parents_contribution_percentage : ℝ := 0.30

theorem remaining_balance (hp : ℝ) (dp : ℝ) (pc : ℝ) : 
  hp * (1 - dp) * (1 - pc) = 56000 :=
by
  sorry

#check remaining_balance house_price down_payment_percentage parents_contribution_percentage

end NUMINAMATH_CALUDE_remaining_balance_l1964_196492


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l1964_196446

/-- The number of letters in the word BANANA -/
def word_length : ℕ := 6

/-- The number of occurrences of the letter B in BANANA -/
def b_count : ℕ := 1

/-- The number of occurrences of the letter N in BANANA -/
def n_count : ℕ := 2

/-- The number of occurrences of the letter A in BANANA -/
def a_count : ℕ := 3

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := word_length.factorial / (b_count.factorial * n_count.factorial * a_count.factorial)

theorem banana_arrangement_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l1964_196446


namespace NUMINAMATH_CALUDE_dinner_time_calculation_l1964_196417

/-- Calculates the time spent eating dinner during a train ride given the total duration and time spent on other activities. -/
theorem dinner_time_calculation (total_duration reading_time movie_time nap_time : ℕ) 
  (h1 : total_duration = 9)
  (h2 : reading_time = 2)
  (h3 : movie_time = 3)
  (h4 : nap_time = 3) :
  total_duration - (reading_time + movie_time + nap_time) = 1 := by
  sorry

#check dinner_time_calculation

end NUMINAMATH_CALUDE_dinner_time_calculation_l1964_196417


namespace NUMINAMATH_CALUDE_total_card_units_traded_l1964_196481

/-- Represents the types of trading cards -/
inductive CardType
| A
| B
| C

/-- Represents a trading round -/
structure TradingRound where
  padmaInitial : CardType → ℕ
  robertInitial : CardType → ℕ
  padmaTrades : CardType → ℕ
  robertTrades : CardType → ℕ
  ratios : CardType → CardType → ℚ

/-- Calculates the total card units traded in a round -/
def cardUnitsTradedInRound (round : TradingRound) : ℚ :=
  sorry

/-- The three trading rounds -/
def round1 : TradingRound := {
  padmaInitial := λ | CardType.A => 50 | CardType.B => 45 | CardType.C => 30,
  robertInitial := λ _ => 0,  -- Not specified in the problem
  padmaTrades := λ | CardType.A => 5 | CardType.B => 12 | CardType.C => 0,
  robertTrades := λ | CardType.C => 20 | _ => 0,
  ratios := λ | CardType.A, CardType.C => 2 | CardType.B, CardType.C => 3/2 | _, _ => 1
}

def round2 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ | CardType.A => 60 | CardType.B => 50 | CardType.C => 40,
  robertTrades := λ | CardType.A => 10 | CardType.B => 3 | CardType.C => 15,
  padmaTrades := λ | CardType.A => 8 | CardType.B => 18 | CardType.C => 0,
  ratios := λ | CardType.A, CardType.B => 3/2 | CardType.B, CardType.C => 2 | CardType.C, CardType.A => 1 | _, _ => 1
}

def round3 : TradingRound := {
  padmaInitial := λ _ => 0,  -- Not relevant for this round
  robertInitial := λ _ => 0,  -- Not relevant for this round
  padmaTrades := λ | CardType.B => 15 | CardType.C => 10 | CardType.A => 0,
  robertTrades := λ | CardType.A => 12 | _ => 0,
  ratios := λ | CardType.A, CardType.B => 5/4 | CardType.C, CardType.A => 6/5 | _, _ => 1
}

/-- The main theorem stating the total card units traded -/
theorem total_card_units_traded :
  cardUnitsTradedInRound round1 + cardUnitsTradedInRound round2 + cardUnitsTradedInRound round3 = 94.75 := by
  sorry

end NUMINAMATH_CALUDE_total_card_units_traded_l1964_196481


namespace NUMINAMATH_CALUDE_power_difference_evaluation_l1964_196496

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_evaluation_l1964_196496


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1964_196457

/-- The number of ways to distribute n indistinguishable items among k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops of 4 basic flavors -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1964_196457


namespace NUMINAMATH_CALUDE_cylinder_volume_tripled_radius_cylinder_volume_increase_l1964_196438

/-- Proves that tripling the radius of a cylinder while keeping the height constant
    results in a volume that is 9 times the original volume. -/
theorem cylinder_volume_tripled_radius 
  (r h : ℝ) 
  (original_volume : ℝ) 
  (h_original_volume : original_volume = π * r^2 * h) 
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 9 * original_volume :=
by sorry

/-- Proves that if a cylinder with volume 10 cubic feet has its radius tripled
    while its height remains constant, its new volume is 90 cubic feet. -/
theorem cylinder_volume_increase
  (r h : ℝ)
  (h_original_volume : π * r^2 * h = 10)
  (h_positive : r > 0 ∧ h > 0) :
  let new_volume := π * (3*r)^2 * h
  new_volume = 90 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_tripled_radius_cylinder_volume_increase_l1964_196438


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1964_196413

theorem cubic_polynomials_common_roots (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    r^3 + a*r^2 + 20*r + 10 = 0 ∧
    r^3 + b*r^2 + 17*r + 12 = 0 ∧
    s^3 + a*s^2 + 20*s + 10 = 0 ∧
    s^3 + b*s^2 + 17*s + 12 = 0) →
  a = 1 ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l1964_196413


namespace NUMINAMATH_CALUDE_coupon1_best_discount_l1964_196464

def coupon1_discount (x : ℝ) : ℝ := 0.1 * x

def coupon2_discount : ℝ := 20

def coupon3_discount (x : ℝ) : ℝ := 0.18 * (x - 100)

theorem coupon1_best_discount (x : ℝ) : 
  (coupon1_discount x > coupon2_discount ∧ 
   coupon1_discount x > coupon3_discount x) ↔ 
  (200 < x ∧ x < 225) :=
sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_l1964_196464


namespace NUMINAMATH_CALUDE_vector_dot_product_l1964_196404

/-- Given two vectors a and b in ℝ², prove that their dot product is -29 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
  (h1 : a.1 + b.1 = 2 ∧ a.2 + b.2 = -4)
  (h2 : 3 * a.1 - b.1 = -10 ∧ 3 * a.2 - b.2 = 16) :
  a.1 * b.1 + a.2 * b.2 = -29 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l1964_196404


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_is_100_l1964_196441

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time it takes for them to pass each other completely. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (pass_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * 1000 / 3600
  let total_distance := relative_speed_mps * pass_time
  total_distance - length1

/-- Proves that the length of the second train is 100 meters under the given conditions. -/
theorem second_train_length_is_100 : 
  second_train_length 80 70 150 5.999520038396928 = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_is_100_l1964_196441


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1964_196465

theorem simplify_sqrt_expression : 
  Real.sqrt 5 - Real.sqrt 40 + Real.sqrt 45 = 4 * Real.sqrt 5 - 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1964_196465


namespace NUMINAMATH_CALUDE_solution_equation_l1964_196416

theorem solution_equation (x : ℝ) (k : ℤ) : 
  (8.492 * (Real.log (Real.sin x) / Real.log (Real.sin x * Real.cos x)) * 
           (Real.log (Real.cos x) / Real.log (Real.sin x * Real.cos x)) = 1/4) →
  (Real.sin x > 0) →
  (x = π/4 * (8 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l1964_196416


namespace NUMINAMATH_CALUDE_balls_sold_l1964_196458

/-- Proves that the number of balls sold is 13 given the selling price, loss, and cost price per ball. -/
theorem balls_sold (selling_price : ℕ) (cost_price_per_ball : ℕ) :
  selling_price = 720 →
  cost_price_per_ball = 90 →
  selling_price + cost_price_per_ball * 5 = cost_price_per_ball * (5 + 13) :=
by
  sorry

#check balls_sold

end NUMINAMATH_CALUDE_balls_sold_l1964_196458


namespace NUMINAMATH_CALUDE_siblings_age_multiple_l1964_196434

theorem siblings_age_multiple (kay_age : ℕ) (oldest_age : ℕ) (num_siblings : ℕ) : 
  kay_age = 32 →
  oldest_age = 44 →
  num_siblings = 14 →
  ∃ (youngest_age : ℕ), 
    youngest_age = kay_age / 2 - 5 ∧
    oldest_age / youngest_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_multiple_l1964_196434


namespace NUMINAMATH_CALUDE_complete_square_sum_l1964_196422

theorem complete_square_sum (a b c : ℤ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, 49 * x^2 + 56 * x - 64 = 0 ↔ (a * x + b)^2 = c) : 
  a + b + c = 91 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1964_196422


namespace NUMINAMATH_CALUDE_constant_function_theorem_l1964_196452

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The property that a function satisfies the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * floor y) = floor (f x) * f y

/-- The theorem stating that functions satisfying the equation are constant functions with values in [1, 2) -/
theorem constant_function_theorem (f : ℝ → ℝ) (h : satisfies_equation f) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c) ∧ 1 ≤ c ∧ c < 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l1964_196452


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l1964_196427

/-- Proves that given a principal amount P put at simple interest for 2 years,
    if an increase of 4% in the interest rate results in Rs. 60 more interest,
    then P = 750. -/
theorem principal_amount_calculation (P R : ℝ) (h1 : P > 0) (h2 : R > 0) :
  (P * (R + 4) * 2) / 100 = (P * R * 2) / 100 + 60 → P = 750 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l1964_196427


namespace NUMINAMATH_CALUDE_problem_solution_l1964_196466

theorem problem_solution (x y : ℝ) (h1 : 3 * x + 2 = 11) (h2 : y = x - 1) : 6 * y - 3 * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1964_196466


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l1964_196444

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m/2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

def B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2*m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2*m + 1}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) :
  (A m ∩ B m).Nonempty → 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l1964_196444


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1964_196467

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Define the y-coordinate of the vertex
def vertex_y : ℝ := -3

-- Theorem statement
theorem parabola_vertex_y_coordinate :
  ∃ x : ℝ, ∀ t : ℝ, f t ≥ f x ∧ f x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l1964_196467


namespace NUMINAMATH_CALUDE_expression_evaluation_l1964_196411

/-- Proves that the given expression evaluates to -5 when x = -2 and y = -1 -/
theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  2 * (x + y) * (-x - y) - (2 * x + y) * (-2 * x + y) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1964_196411


namespace NUMINAMATH_CALUDE_car_numbers_proof_l1964_196431

theorem car_numbers_proof :
  ∃! (x y : ℕ), 
    100 ≤ x ∧ x ≤ 999 ∧
    100 ≤ y ∧ y ≤ 999 ∧
    (∃ (a b c d : ℕ), x = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    (∃ (a b c d : ℕ), y = 100 * a + 10 * b + 3 ∧ (c = 3 ∨ d = 3)) ∧
    119 * x + 179 * y = 105080 ∧
    x = 337 ∧ y = 363 := by
  sorry

end NUMINAMATH_CALUDE_car_numbers_proof_l1964_196431


namespace NUMINAMATH_CALUDE_min_swaps_for_initial_number_l1964_196478

def initial_number : ℕ := 9072543681

def is_divisible_by_99 (n : ℕ) : Prop :=
  n % 99 = 0

def adjacent_swap (n : ℕ) (i : ℕ) : ℕ :=
  sorry

def min_swaps_to_divisible_by_99 (n : ℕ) : ℕ :=
  sorry

theorem min_swaps_for_initial_number :
  min_swaps_to_divisible_by_99 initial_number = 2 :=
sorry

end NUMINAMATH_CALUDE_min_swaps_for_initial_number_l1964_196478


namespace NUMINAMATH_CALUDE_janets_crayons_l1964_196437

theorem janets_crayons (michelle_initial : ℕ) (michelle_final : ℕ) (janet_initial : ℕ) : 
  michelle_initial = 2 → 
  michelle_final = 4 → 
  michelle_final = michelle_initial + janet_initial → 
  janet_initial = 2 := by
sorry

end NUMINAMATH_CALUDE_janets_crayons_l1964_196437


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_baked_l1964_196456

/-- The number of dozens of cookies Ann baked for each type -/
structure CookieBatch where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

/-- The number of dozens of cookies Ann gave away for each type -/
structure CookiesGivenAway where
  oatmeal_raisin : ℚ
  sugar : ℚ
  chocolate_chip : ℚ

def cookies_kept (baked : CookieBatch) (given_away : CookiesGivenAway) : ℚ :=
  (baked.oatmeal_raisin - given_away.oatmeal_raisin +
   baked.sugar - given_away.sugar +
   baked.chocolate_chip - given_away.chocolate_chip) * 12

theorem chocolate_chip_cookies_baked 
  (baked : CookieBatch)
  (given_away : CookiesGivenAway)
  (h1 : baked.oatmeal_raisin = 3)
  (h2 : baked.sugar = 2)
  (h3 : given_away.oatmeal_raisin = 2)
  (h4 : given_away.sugar = 3/2)
  (h5 : given_away.chocolate_chip = 5/2)
  (h6 : cookies_kept baked given_away = 36) :
  baked.chocolate_chip = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_baked_l1964_196456


namespace NUMINAMATH_CALUDE_suzanna_textbooks_pages_l1964_196477

/-- Calculates the total number of pages in Suzanna's textbooks -/
def total_pages (history : ℕ) : ℕ :=
  let geography := history + 70
  let math := (history + geography) / 2
  let science := 2 * history
  let literature := history + geography - 30
  let economics := math + literature + 25
  history + geography + math + science + literature + economics

/-- Theorem stating that the total number of pages in Suzanna's textbooks is 1845 -/
theorem suzanna_textbooks_pages : total_pages 160 = 1845 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_textbooks_pages_l1964_196477


namespace NUMINAMATH_CALUDE_inverse_proportion_doubling_l1964_196455

/-- Given two positive real numbers x and y that are inversely proportional,
    if x doubles, then y decreases by 50%. -/
theorem inverse_proportion_doubling (x y x' y' : ℝ) (k : ℝ) (hxy_pos : x > 0 ∧ y > 0) 
    (hk_pos : k > 0) (hxy : x * y = k) (hx'y' : x' * y' = k) (hx_double : x' = 2 * x) : 
    y' = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_doubling_l1964_196455


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1964_196432

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 8/y = 1) :
  x + y ≥ 18 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1964_196432


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1964_196450

theorem quadratic_coefficient (c : ℝ) : (5 : ℝ)^2 + c * 5 + 45 = 0 → c = -14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1964_196450


namespace NUMINAMATH_CALUDE_surjective_sum_iff_constant_l1964_196414

-- Define a surjective function over ℤ
def Surjective (g : ℤ → ℤ) : Prop :=
  ∀ y : ℤ, ∃ x : ℤ, g x = y

-- Define the property that f + g is surjective for all surjective g
def SurjectiveSum (f : ℤ → ℤ) : Prop :=
  ∀ g : ℤ → ℤ, Surjective g → Surjective (fun x ↦ f x + g x)

-- Define a constant function
def ConstantFunction (f : ℤ → ℤ) : Prop :=
  ∃ c : ℤ, ∀ x : ℤ, f x = c

-- Theorem statement
theorem surjective_sum_iff_constant (f : ℤ → ℤ) :
  SurjectiveSum f ↔ ConstantFunction f :=
sorry

end NUMINAMATH_CALUDE_surjective_sum_iff_constant_l1964_196414


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1964_196409

/-- Arithmetic sequence type -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  first : α
  diff : α

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- n-th term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_ratio 
  (a b : ArithmeticSequence ℚ) 
  (h : ∀ n : ℕ, sum_n a n / sum_n b n = (3 * n - 1) / (n + 3)) :
  nth_term a 8 / (nth_term b 5 + nth_term b 11) = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1964_196409


namespace NUMINAMATH_CALUDE_biggest_collection_l1964_196494

def yoongi_collection : ℕ := 4
def jungkook_collection : ℕ := 6 * 3
def yuna_collection : ℕ := 5

theorem biggest_collection :
  max yoongi_collection (max jungkook_collection yuna_collection) = jungkook_collection :=
by sorry

end NUMINAMATH_CALUDE_biggest_collection_l1964_196494


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1964_196443

/-- The ratio of Sandy's current age to Molly's current age is 4:3, given that Sandy will be 38 years old in 6 years and Molly is currently 24 years old. -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 38
  let years_until_future : ℕ := 6
  let molly_current_age : ℕ := 24
  let sandy_current_age : ℕ := sandy_future_age - years_until_future
  (sandy_current_age : ℚ) / molly_current_age = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1964_196443
