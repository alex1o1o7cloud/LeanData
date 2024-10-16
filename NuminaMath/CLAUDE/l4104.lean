import Mathlib

namespace NUMINAMATH_CALUDE_cement_mixture_weight_l4104_410455

theorem cement_mixture_weight :
  ∀ (W : ℚ),
  (2 / 7 : ℚ) * W +  -- Sand
  (3 / 7 : ℚ) * W +  -- Water
  (1 / 14 : ℚ) * W + -- Gravel
  (1 / 14 : ℚ) * W + -- Cement
  12 = W             -- Crushed stones
  →
  W = 84 :=
by sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l4104_410455


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4104_410481

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 4) > 1

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < -1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4104_410481


namespace NUMINAMATH_CALUDE_r_squared_ssr_inverse_relation_l4104_410419

/-- Represents a regression model -/
structure RegressionModel where
  R_squared : ℝ  -- Coefficient of determination
  SSR : ℝ        -- Sum of squares of residuals

/-- States that as R² increases, SSR decreases in a regression model -/
theorem r_squared_ssr_inverse_relation (model1 model2 : RegressionModel) :
  model1.R_squared > model2.R_squared → model1.SSR < model2.SSR := by
  sorry

end NUMINAMATH_CALUDE_r_squared_ssr_inverse_relation_l4104_410419


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4104_410443

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4104_410443


namespace NUMINAMATH_CALUDE_total_students_surveyed_l4104_410475

/-- Represents the number of students speaking different combinations of languages --/
structure LanguageCounts where
  french : ℕ
  english : ℕ
  spanish : ℕ
  frenchEnglish : ℕ
  frenchSpanish : ℕ
  englishSpanish : ℕ
  allThree : ℕ
  none : ℕ

/-- The conditions given in the problem --/
def languageConditions (counts : LanguageCounts) : Prop :=
  -- 230 students speak only one language
  counts.french + counts.english + counts.spanish = 230 ∧
  -- 190 students speak exactly two languages
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish = 190 ∧
  -- 40 students speak all three languages
  counts.allThree = 40 ∧
  -- 60 students do not speak any of the three languages
  counts.none = 60 ∧
  -- Among French speakers, 25% speak English, 15% speak Spanish, and 10% speak both English and Spanish
  4 * (counts.frenchEnglish + counts.allThree) = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  20 * (counts.frenchSpanish + counts.allThree) = 3 * (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  10 * counts.allThree = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  -- Among English speakers, 20% also speak Spanish
  5 * (counts.englishSpanish + counts.allThree) = (counts.english + counts.frenchEnglish + counts.englishSpanish + counts.allThree)

/-- The theorem to be proved --/
theorem total_students_surveyed (counts : LanguageCounts) :
  languageConditions counts →
  counts.french + counts.english + counts.spanish +
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish +
  counts.allThree + counts.none = 520 :=
by sorry

end NUMINAMATH_CALUDE_total_students_surveyed_l4104_410475


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l4104_410448

theorem least_positive_integer_multiple_of_53 : 
  ∃ (x : ℕ+), (x = 4) ∧ 
  (∀ (y : ℕ+), y < x → ¬(53 ∣ (3*y)^2 + 2*41*3*y + 41^2)) ∧ 
  (53 ∣ (3*x)^2 + 2*41*3*x + 41^2) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l4104_410448


namespace NUMINAMATH_CALUDE_proportion_solution_l4104_410424

theorem proportion_solution (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l4104_410424


namespace NUMINAMATH_CALUDE_divisibility_property_l4104_410425

theorem divisibility_property (A B n : ℕ) (hn : n = 7 ∨ n = 11 ∨ n = 13) 
  (h : n ∣ (B - A)) : n ∣ (1000 * A + B) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l4104_410425


namespace NUMINAMATH_CALUDE_flowchart_output_l4104_410462

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem flowchart_output (N : ℕ) (h : N = 6) : factorial N = 720 := by
  sorry

end NUMINAMATH_CALUDE_flowchart_output_l4104_410462


namespace NUMINAMATH_CALUDE_decreasing_function_positive_l4104_410450

/-- A function f is decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

/-- The condition that f'(x) satisfies f(x) / f''(x) < 1 - x -/
def DerivativeCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ (deriv f) x ∧
    f x / (deriv (deriv f) x) < 1 - x

theorem decreasing_function_positive
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingOn f)
  (h_condition : DerivativeCondition f) :
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_l4104_410450


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_l4104_410416

/-- Given a line ax + by + c = 0, returns the line symmetric to it with respect to the y-axis -/
def symmetricLineY (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def lineThroughPoints (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ × ℝ :=
  let a := y₂ - y₁
  let b := x₁ - x₂
  let c := x₂ * y₁ - x₁ * y₂
  (a, b, c)

theorem symmetric_line_y_axis :
  let original_line := (3, -4, 5)
  let symmetric_line := symmetricLineY 3 (-4) 5
  let y_intercept := (0, 5/4)
  let x_intercept_symmetric := (5/3, 0)
  let line_through_points := lineThroughPoints 0 (5/4) (5/3) 0
  symmetric_line = line_through_points := by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_l4104_410416


namespace NUMINAMATH_CALUDE_smallest_a_correct_l4104_410465

/-- The smallest natural number a such that there are exactly 50 perfect squares in the interval (a, 3a) -/
def smallest_a : ℕ := 4486

/-- The number of perfect squares in the interval (a, 3a) -/
def count_squares (a : ℕ) : ℕ :=
  (Nat.sqrt (3 * a) - Nat.sqrt a).pred

theorem smallest_a_correct :
  (∀ b < smallest_a, count_squares b ≠ 50) ∧
  count_squares smallest_a = 50 :=
sorry

#eval smallest_a
#eval count_squares smallest_a

end NUMINAMATH_CALUDE_smallest_a_correct_l4104_410465


namespace NUMINAMATH_CALUDE_correct_selection_count_l4104_410445

/-- The number of ways to select representatives satisfying given conditions -/
def select_representatives (num_boys num_girls : ℕ) : ℕ :=
  let total_students := num_boys + num_girls
  let num_subjects := 5
  3360

/-- The theorem stating the correct number of ways to select representatives -/
theorem correct_selection_count :
  select_representatives 5 3 = 3360 := by sorry

end NUMINAMATH_CALUDE_correct_selection_count_l4104_410445


namespace NUMINAMATH_CALUDE_subset_divisibility_property_l4104_410496

theorem subset_divisibility_property (A : Finset ℕ) (hA : A.card = 3) :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card = 2 ∧
  ∀ (x y : ℕ) (hx : x ∈ B) (hy : y ∈ B) (m n : ℕ) (hm : Odd m) (hn : Odd n),
  (10 : ℕ) ∣ (x^m * y^n - x^n * y^m) :=
sorry

end NUMINAMATH_CALUDE_subset_divisibility_property_l4104_410496


namespace NUMINAMATH_CALUDE_equivalent_form_proof_l4104_410407

theorem equivalent_form_proof (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 7) 
  (h : (5 / x) + (4 / y) = (1 / 3)) : 
  x = (15 * y) / (y - 12) := by
sorry

end NUMINAMATH_CALUDE_equivalent_form_proof_l4104_410407


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l4104_410489

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 24*x + y^2 + 10*y + 160 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 10

theorem shortest_distance_to_circle :
  ∀ p : ℝ × ℝ, circle_equation p.1 p.2 →
  ∃ q : ℝ × ℝ, circle_equation q.1 q.2 ∧
  ∀ r : ℝ × ℝ, circle_equation r.1 r.2 →
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) ≤ Real.sqrt ((r.1 - 0)^2 + (r.2 - 0)^2) ∧
  Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = shortest_distance := by
  sorry


end NUMINAMATH_CALUDE_shortest_distance_to_circle_l4104_410489


namespace NUMINAMATH_CALUDE_diane_gingerbreads_l4104_410414

/-- Proves that given Diane's baking conditions, each of the four trays contains 25 gingerbreads -/
theorem diane_gingerbreads :
  ∀ (x : ℕ),
  (4 * x + 3 * 20 = 160) →
  x = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_diane_gingerbreads_l4104_410414


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l4104_410499

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^2 < 2000 → x ≤ 44 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^2 < 2000 ∧ y = 44 := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l4104_410499


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l4104_410485

-- Define the curve
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k + 2) - y^2 / (6 - 2*k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop := -2 < k ∧ k < 3

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l4104_410485


namespace NUMINAMATH_CALUDE_exactly_two_rotational_homotheties_l4104_410464

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A rotational homothety with 90° rotation --/
structure RotationalHomothety where
  center : ℝ × ℝ
  scale : ℝ

/-- The number of rotational homotheties with 90° rotation that map one circle to another --/
def num_rotational_homotheties (S₁ S₂ : Circle) : ℕ :=
  sorry

/-- Two circles are non-concentric if their centers are different --/
def non_concentric (S₁ S₂ : Circle) : Prop :=
  S₁.center ≠ S₂.center

theorem exactly_two_rotational_homotheties (S₁ S₂ : Circle) 
  (h : non_concentric S₁ S₂) : num_rotational_homotheties S₁ S₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_rotational_homotheties_l4104_410464


namespace NUMINAMATH_CALUDE_binomial_expansion_degree_l4104_410491

theorem binomial_expansion_degree (n : ℕ) :
  (∀ x, (1 + x)^n = 1 + 6*x + 15*x^2 + 20*x^3 + 15*x^4 + 6*x^5 + x^6) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_degree_l4104_410491


namespace NUMINAMATH_CALUDE_rhombus_area_l4104_410459

/-- The area of a rhombus with diagonals measuring 9 cm and 14 cm is 63 square centimeters. -/
theorem rhombus_area (d1 d2 area : ℝ) : 
  d1 = 9 → d2 = 14 → area = (d1 * d2) / 2 → area = 63 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l4104_410459


namespace NUMINAMATH_CALUDE_sphere_surface_area_l4104_410460

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  (π * d^2) = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l4104_410460


namespace NUMINAMATH_CALUDE_math_test_problem_l4104_410418

theorem math_test_problem (total : ℕ) (word_problems : ℕ) (answered : ℕ) (blank : ℕ) :
  total = 45 →
  word_problems = 17 →
  answered = 38 →
  blank = 7 →
  total = answered + blank →
  total - word_problems - blank = 21 := by
  sorry

end NUMINAMATH_CALUDE_math_test_problem_l4104_410418


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l4104_410437

theorem stewart_farm_ratio : 
  ∀ (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
    sheep = 8 →
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    horses * horse_food_per_day = total_horse_food →
    sheep.gcd horses = 1 →
    sheep / horses = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l4104_410437


namespace NUMINAMATH_CALUDE_total_books_proof_l4104_410463

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 19

/-- The number of books already read -/
def books_read : ℕ := 4

/-- The number of books yet to be read -/
def books_to_read : ℕ := 15

/-- Theorem stating that the total number of books is the sum of books read and books to be read -/
theorem total_books_proof : total_books = books_read + books_to_read := by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l4104_410463


namespace NUMINAMATH_CALUDE_fourth_power_sum_equals_108_to_fourth_l4104_410480

theorem fourth_power_sum_equals_108_to_fourth : ∃ m : ℕ+, 
  97^4 + 84^4 + 27^4 + 3^4 = m^4 ∧ m = 108 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equals_108_to_fourth_l4104_410480


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l4104_410427

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetric_x (A B : Point) : Prop :=
  B.x = A.x ∧ B.y = -A.y

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, -1⟩
  ∀ B : Point, symmetric_x A B → B = ⟨2, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l4104_410427


namespace NUMINAMATH_CALUDE_sally_monday_shirts_l4104_410444

def shirts_sewn_tuesday : ℕ := 3
def shirts_sewn_wednesday : ℕ := 2
def buttons_per_shirt : ℕ := 5
def total_buttons_needed : ℕ := 45

theorem sally_monday_shirts :
  ∃ (monday_shirts : ℕ),
    monday_shirts + shirts_sewn_tuesday + shirts_sewn_wednesday = 
    total_buttons_needed / buttons_per_shirt ∧
    monday_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_sally_monday_shirts_l4104_410444


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l4104_410430

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l4104_410430


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l4104_410482

/-- The imaginary unit -/
def i : ℂ := Complex.I

theorem complex_simplification_and_multiplication :
  ((6 - 3 * i) - (2 - 5 * i)) * (1 + 2 * i) = 10 * i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l4104_410482


namespace NUMINAMATH_CALUDE_funfair_tickets_l4104_410426

theorem funfair_tickets (total_rolls : ℕ) (fourth_grade_percent : ℚ) 
  (fifth_grade_percent : ℚ) (sixth_grade_bought : ℕ) (tickets_left : ℕ) :
  total_rolls = 30 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 50 / 100 →
  sixth_grade_bought = 100 →
  tickets_left = 950 →
  ∃ (tickets_per_roll : ℕ),
    tickets_per_roll * total_rolls * (1 - fourth_grade_percent) * (1 - fifth_grade_percent) - sixth_grade_bought = tickets_left ∧
    tickets_per_roll = 100 := by
  sorry

end NUMINAMATH_CALUDE_funfair_tickets_l4104_410426


namespace NUMINAMATH_CALUDE_total_students_both_schools_l4104_410436

-- Define the number of students in Middle School
def middle_school_students : ℕ := 50

-- Define the number of students in Elementary School
def elementary_school_students : ℕ := 4 * middle_school_students - 3

-- Theorem to prove
theorem total_students_both_schools :
  elementary_school_students + middle_school_students = 247 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_both_schools_l4104_410436


namespace NUMINAMATH_CALUDE_constant_term_of_x_minus_inverse_x_power_8_l4104_410434

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the constant term of (x - 1/x)^n
def constantTerm (n : ℕ) : ℤ :=
  if n % 2 = 0
  then (-1)^(n/2) * binomial n (n/2)
  else 0

-- Theorem statement
theorem constant_term_of_x_minus_inverse_x_power_8 :
  constantTerm 8 = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_x_minus_inverse_x_power_8_l4104_410434


namespace NUMINAMATH_CALUDE_condo_units_calculation_l4104_410417

/-- Calculates the total number of units in a condo development -/
theorem condo_units_calculation (total_floors : ℕ) (regular_units_per_floor : ℕ) 
  (penthouse_units_per_floor : ℕ) (penthouse_floors : ℕ) : 
  total_floors = 23 → 
  regular_units_per_floor = 12 → 
  penthouse_units_per_floor = 2 → 
  penthouse_floors = 2 → 
  (total_floors - penthouse_floors) * regular_units_per_floor + 
    penthouse_floors * penthouse_units_per_floor = 256 := by
  sorry

#check condo_units_calculation

end NUMINAMATH_CALUDE_condo_units_calculation_l4104_410417


namespace NUMINAMATH_CALUDE_problem_statement_l4104_410456

open Real

theorem problem_statement :
  (∀ x > 0, exp x - 2 > x - 1 ∧ x - 1 ≥ log x) ∧
  (∀ m : ℤ, m < 1 → ¬∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - m - 2 = 0 ∧ exp y - log y - m - 2 = 0) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ exp x - log x - 1 - 2 = 0 ∧ exp y - log y - 1 - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4104_410456


namespace NUMINAMATH_CALUDE_f_inequality_implies_a_range_l4104_410442

open Set Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (log x + 2) / x + a * (x - 1) - 2

def domain : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

theorem f_inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ domain, (f a x) / (1 - x) < a / x) → a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_implies_a_range_l4104_410442


namespace NUMINAMATH_CALUDE_quadratic_ratio_l4104_410492

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 800*x + 2400

-- Define the completed square form
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

-- Theorem statement
theorem quadratic_ratio :
  ∃ (b c : ℝ), (∀ x, f x = g x b c) ∧ (c / b = -394) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l4104_410492


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4104_410495

theorem modulus_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I * Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4104_410495


namespace NUMINAMATH_CALUDE_fraction_evaluation_l4104_410483

theorem fraction_evaluation : (4 * 3) / (2 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l4104_410483


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4104_410409

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where:
    - The distance from the focus to the asymptote is 2√3
    - The minimum distance from a point on the right branch to the right focus is 2
    Then the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 
    b * c / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 3 ∧ 
    c - a = 2) → 
  c / a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4104_410409


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4104_410454

theorem min_value_expression (x : ℝ) (h : x > 0) :
  x^2 / x + 2 + 5 / x ≥ 2 * Real.sqrt 5 + 2 :=
sorry

theorem min_value_achievable :
  ∃ (x : ℝ), x > 0 ∧ x^2 / x + 2 + 5 / x = 2 * Real.sqrt 5 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l4104_410454


namespace NUMINAMATH_CALUDE_raft_sticks_total_l4104_410411

theorem raft_sticks_total (simon_sticks : ℕ) (gerry_sticks : ℕ) (micky_sticks : ℕ) : 
  simon_sticks = 36 →
  gerry_sticks = 2 * (simon_sticks / 3) →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  simon_sticks + gerry_sticks + micky_sticks = 129 :=
by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l4104_410411


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_l4104_410484

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff (k : ℝ) :
  is_not_monotonic f (k - 1) (k + 1) ↔ 1 ≤ k ∧ k < 3/2 := by sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_l4104_410484


namespace NUMINAMATH_CALUDE_coupon_difference_l4104_410432

/-- Represents the savings from a coupon given a price -/
def coupon_savings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def coupon_a (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: flat $40 off -/
def coupon_b (_ : ℝ) : ℝ := 40

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 plus an additional $20 -/
def coupon_c (price : ℝ) : ℝ := 0.3 * (price - 120) + 20

/-- The proposition that Coupon A is at least as good as Coupon B or C for a given price -/
def coupon_a_best (price : ℝ) : Prop :=
  coupon_savings price coupon_a ≥ max (coupon_savings price coupon_b) (coupon_savings price coupon_c)

theorem coupon_difference :
  ∃ (x y : ℝ),
    x > 120 ∧
    y > 120 ∧
    x ≤ y ∧
    coupon_a_best x ∧
    coupon_a_best y ∧
    (∀ p, x < p ∧ p < y → coupon_a_best p) ∧
    (∀ p, p < x → ¬coupon_a_best p) ∧
    (∀ p, p > y → ¬coupon_a_best p) ∧
    y - x = 100 :=
  sorry

end NUMINAMATH_CALUDE_coupon_difference_l4104_410432


namespace NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_problem_specific_savings_l4104_410469

/-- Represents the store's window offer -/
structure WindowOffer where
  normalPrice : ℕ
  freeWindowsPer : ℕ

/-- Calculates the cost of purchasing windows under the offer -/
def costUnderOffer (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let paidWindows := windowsNeeded - (windowsNeeded / (offer.freeWindowsPer + 1))
  paidWindows * offer.normalPrice

/-- Calculates the savings for a given number of windows -/
def savings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.normalPrice - costUnderOffer offer windowsNeeded

/-- Theorem: The combined savings equal the sum of individual savings -/
theorem combined_savings_equal_individual_savings 
  (offer : WindowOffer) 
  (daveWindows : ℕ) 
  (dougWindows : ℕ) : 
  savings offer (daveWindows + dougWindows) = 
    savings offer daveWindows + savings offer dougWindows :=
by sorry

/-- The specific offer in the problem -/
def storeOffer : WindowOffer := { normalPrice := 100, freeWindowsPer := 3 }

/-- Theorem: For Dave (9 windows) and Doug (6 windows), 
    the combined savings equal the sum of their individual savings -/
theorem problem_specific_savings : 
  savings storeOffer (9 + 6) = savings storeOffer 9 + savings storeOffer 6 :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_individual_savings_problem_specific_savings_l4104_410469


namespace NUMINAMATH_CALUDE_deductive_reasoning_example_l4104_410468

-- Define the property of conducting electricity
def conducts_electricity (x : Type) : Prop := sorry

-- Define the concept of metal
def is_metal (x : Type) : Prop := sorry

-- Define deductive reasoning
def is_deductive_reasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Define the specific metals
def gold : Type := sorry
def silver : Type := sorry
def copper : Type := sorry

-- Theorem statement
theorem deductive_reasoning_example :
  let premise1 : Prop := ∀ x, is_metal x → conducts_electricity x
  let premise2 : Prop := is_metal gold ∧ is_metal silver ∧ is_metal copper
  let conclusion : Prop := conducts_electricity gold ∧ conducts_electricity silver ∧ conducts_electricity copper
  is_deductive_reasoning premise1 premise2 conclusion := by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_example_l4104_410468


namespace NUMINAMATH_CALUDE_james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l4104_410422

/-- The sum of James and Louise's current ages given the conditions -/
theorem james_and_louise_age_sum : ℝ :=
  let james_age : ℝ := sorry
  let louise_age : ℝ := sorry

  -- James is eight years older than Louise
  have h1 : james_age = louise_age + 8 := by sorry

  -- Ten years from now, James will be five times as old as Louise was five years ago
  have h2 : james_age + 10 = 5 * (louise_age - 5) := by sorry

  -- The sum of their current ages
  have h3 : james_age + louise_age = 29.5 := by sorry

  29.5

theorem james_and_louise_age_sum_is_correct : james_and_louise_age_sum = 29.5 := by sorry

end NUMINAMATH_CALUDE_james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l4104_410422


namespace NUMINAMATH_CALUDE_expression_evaluation_l4104_410447

/-- Proves that the given expression evaluates to 11 when x = -2 and y = -1 -/
theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := -1
  3 * (2 * x^2 + x*y + 1/3) - (3 * x^2 + 4*x*y - y^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4104_410447


namespace NUMINAMATH_CALUDE_triangle_area_l4104_410405

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that if 4S = √3(a² + b² - c²) and 
    f(x) = 4sin(x)cos(x + π/6) + 1 attains its maximum value b when x = A,
    then the area S of the triangle is √3/2. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  4 * S = Real.sqrt 3 * (a^2 + b^2 - c^2) →
  (∀ x, 4 * Real.sin x * Real.cos (x + π/6) + 1 ≤ b) →
  (4 * Real.sin A * Real.cos (A + π/6) + 1 = b) →
  S = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4104_410405


namespace NUMINAMATH_CALUDE_opposite_not_positive_implies_non_negative_l4104_410404

theorem opposite_not_positive_implies_non_negative (a : ℝ) :
  (-a ≤ 0) → (a ≥ 0) := by sorry

end NUMINAMATH_CALUDE_opposite_not_positive_implies_non_negative_l4104_410404


namespace NUMINAMATH_CALUDE_volunteer_distribution_l4104_410440

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 7 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l4104_410440


namespace NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l4104_410490

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l4104_410490


namespace NUMINAMATH_CALUDE_smallest_k_for_minimal_period_l4104_410461

-- Define the concept of a minimal period for a rational number's decimal representation
def has_minimal_period (q : ℚ) (n : ℕ) : Prop := sorry

-- Define the theorem
theorem smallest_k_for_minimal_period 
  (a b : ℚ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab_period : has_minimal_period a 30 ∧ has_minimal_period b 30) 
  (hdiff_period : has_minimal_period (a - b) 15) :
  ∃ (k : ℕ), k = 6 ∧ 
    (∀ (j : ℕ), has_minimal_period (a + j * b) 15 → k ≤ j) ∧ 
    has_minimal_period (a + k * b) 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_minimal_period_l4104_410461


namespace NUMINAMATH_CALUDE_right_triangle_in_circle_l4104_410493

theorem right_triangle_in_circle (diameter : ℝ) (leg1 : ℝ) (leg2 : ℝ) : 
  diameter = 10 → leg1 = 6 → leg2 * leg2 = diameter * diameter - leg1 * leg1 → leg2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_in_circle_l4104_410493


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l4104_410470

theorem geometric_sequence_middle_term (χ : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ -1 * r = χ ∧ χ * r = -4) → χ = 2 ∨ χ = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l4104_410470


namespace NUMINAMATH_CALUDE_A_intersect_B_l4104_410471

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l4104_410471


namespace NUMINAMATH_CALUDE_water_formed_equals_three_l4104_410408

-- Define the chemical species
inductive ChemicalSpecies
| NH4Cl
| NaOH
| NaCl
| NH3
| H2O

-- Define the reaction equation
def reactionEquation : List (ChemicalSpecies × Int) :=
  [(ChemicalSpecies.NH4Cl, -1), (ChemicalSpecies.NaOH, -1),
   (ChemicalSpecies.NaCl, 1), (ChemicalSpecies.NH3, 1), (ChemicalSpecies.H2O, 1)]

-- Define the initial amounts of reactants
def initialNH4Cl : ℕ := 3
def initialNaOH : ℕ := 3

-- Function to calculate the moles of water formed
def molesOfWaterFormed (nh4cl : ℕ) (naoh : ℕ) : ℕ :=
  min nh4cl naoh

-- Theorem statement
theorem water_formed_equals_three :
  molesOfWaterFormed initialNH4Cl initialNaOH = 3 := by
  sorry


end NUMINAMATH_CALUDE_water_formed_equals_three_l4104_410408


namespace NUMINAMATH_CALUDE_division_exponent_rule_l4104_410415

theorem division_exponent_rule (x : ℝ) : -6 * x^5 / (2 * x^3) = -3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_division_exponent_rule_l4104_410415


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l4104_410477

/-- Given the following conditions about wildflowers:
  * The total number of wildflowers is 44
  * There are 13 yellow and white flowers
  * There are 17 red and yellow flowers
  * There are 14 red and white flowers

  Prove that the number of flowers containing red minus
  the number of flowers containing white equals 4.
-/
theorem wildflower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (h_total : total = 44)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l4104_410477


namespace NUMINAMATH_CALUDE_new_total_observations_l4104_410402

theorem new_total_observations 
  (initial_sum : ℝ) 
  (initial_count : ℕ) 
  (new_observation : ℝ) 
  (new_average : ℝ) 
  (h1 : initial_sum / initial_count = 13)
  (h2 : initial_count = 6)
  (h3 : new_observation = 6)
  (h4 : (initial_sum + new_observation) / (initial_count + 1) = new_average)
  (h5 : new_average = 13 - 1) :
  initial_count + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_new_total_observations_l4104_410402


namespace NUMINAMATH_CALUDE_dave_earnings_l4104_410457

/-- Calculates the total money earned from selling video games -/
def total_money_earned (action_games adventure_games roleplaying_games : ℕ) 
  (action_price adventure_price roleplaying_price : ℕ) : ℕ :=
  action_games * action_price + 
  adventure_games * adventure_price + 
  roleplaying_games * roleplaying_price

/-- Proves that Dave earns $49 by selling all working games -/
theorem dave_earnings : 
  total_money_earned 3 2 3 6 5 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_dave_earnings_l4104_410457


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l4104_410453

theorem fraction_sum_equals_one (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) + 1 / (1 - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l4104_410453


namespace NUMINAMATH_CALUDE_smallest_number_proof_l4104_410428

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_number_proof (x : ℕ) (hx : x > 0) : 
  (∀ n : ℕ, n < 5064 → ¬(is_divisible_by (n - 24) x ∧ 
                         is_divisible_by (n - 24) 10 ∧ 
                         is_divisible_by (n - 24) 15 ∧ 
                         is_divisible_by (n - 24) 20 ∧ 
                         (n - 24) / x = 84 ∧ 
                         (n - 24) / 10 = 84 ∧ 
                         (n - 24) / 15 = 84 ∧ 
                         (n - 24) / 20 = 84)) ∧
  (is_divisible_by (5064 - 24) x ∧ 
   is_divisible_by (5064 - 24) 10 ∧ 
   is_divisible_by (5064 - 24) 15 ∧ 
   is_divisible_by (5064 - 24) 20 ∧ 
   (5064 - 24) / x = 84 ∧ 
   (5064 - 24) / 10 = 84 ∧ 
   (5064 - 24) / 15 = 84 ∧ 
   (5064 - 24) / 20 = 84) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l4104_410428


namespace NUMINAMATH_CALUDE_history_not_statistics_l4104_410487

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 90 →
  history = 36 →
  statistics = 30 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 29 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l4104_410487


namespace NUMINAMATH_CALUDE_positive_integers_divisibility_l4104_410446

theorem positive_integers_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_divisibility_l4104_410446


namespace NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l4104_410431

theorem smallest_n_with_partial_divisibility : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > 0 → m < n →
    (∃ (k : ℕ), k > 0 ∧ k ≤ m ∧ (m * (m + 1)) % k = 0) ∧
    (∀ (k : ℕ), k > 0 ∧ k ≤ m → (m * (m + 1)) % k = 0)) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k = 0) ∧
  (∃ (k : ℕ), k > 0 ∧ k ≤ n ∧ (n * (n + 1)) % k ≠ 0) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_partial_divisibility_l4104_410431


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l4104_410412

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 500 ∧ selling_price = 800 →
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l4104_410412


namespace NUMINAMATH_CALUDE_jana_height_l4104_410439

theorem jana_height (jess_height kelly_height jana_height : ℕ) : 
  jess_height = 72 →
  kelly_height = jess_height - 3 →
  jana_height = kelly_height + 5 →
  jana_height = 74 := by
  sorry

end NUMINAMATH_CALUDE_jana_height_l4104_410439


namespace NUMINAMATH_CALUDE_xxyy_perfect_square_l4104_410438

theorem xxyy_perfect_square : 
  ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ 
  (1100 * x + 11 * y = 88 * 88) := by
sorry

end NUMINAMATH_CALUDE_xxyy_perfect_square_l4104_410438


namespace NUMINAMATH_CALUDE_second_frog_hops_second_frog_hops_proof_l4104_410478

/-- Given three frogs hopping across a road, prove the number of hops taken by the second frog -/
theorem second_frog_hops : ℕ → ℕ → ℕ → Prop :=
  fun frog1 frog2 frog3 =>
    frog1 = 4 * frog2 ∧            -- First frog takes 4 times as many hops as the second
    frog2 = 2 * frog3 ∧            -- Second frog takes twice as many hops as the third
    frog1 + frog2 + frog3 = 99 →   -- Total hops is 99
    frog2 = 18                     -- Second frog takes 18 hops

theorem second_frog_hops_proof : ∃ (frog1 frog2 frog3 : ℕ), second_frog_hops frog1 frog2 frog3 := by
  sorry

end NUMINAMATH_CALUDE_second_frog_hops_second_frog_hops_proof_l4104_410478


namespace NUMINAMATH_CALUDE_cubic_polynomial_special_roots_l4104_410441

/-- A polynomial of degree 3 with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
structure CubicRoots where
  r : ℝ
  s : ℝ
  t : ℝ

/-- Proposition: For a cubic polynomial x^3 - 5x^2 + 2bx - c with real and positive roots,
    where one root is twice another and four times the third, c = 1000/343 -/
theorem cubic_polynomial_special_roots (p : CubicPolynomial) (roots : CubicRoots) :
  p.a = 1 ∧ p.b = -5 ∧  -- Coefficients of the polynomial
  (∀ x, x^3 - 5*x^2 + 2*p.b*x - p.c = 0 ↔ x = roots.r ∨ x = roots.s ∨ x = roots.t) ∧  -- Roots definition
  roots.r > 0 ∧ roots.s > 0 ∧ roots.t > 0 ∧  -- Roots are positive
  roots.s = 2 * roots.t ∧ roots.r = 4 * roots.t  -- Root relationships
  →
  p.c = 1000 / 343 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_special_roots_l4104_410441


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l4104_410451

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 7 * a 11 = 6)
  (h_sum : a 4 + a 14 = 5) :
  a 20 / a 10 = 3/2 ∨ a 20 / a 10 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l4104_410451


namespace NUMINAMATH_CALUDE_batsman_average_increase_l4104_410467

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : Nat) : Rat :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : newAverage b 85 = b.average + 4) 
  (h3 : b.average > 0) : 
  newAverage b 85 = 9 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l4104_410467


namespace NUMINAMATH_CALUDE_sum_of_product_of_roots_l4104_410474

theorem sum_of_product_of_roots (p q r : ℂ) : 
  (4 * p^3 - 8 * p^2 + 16 * p - 12 = 0) ∧ 
  (4 * q^3 - 8 * q^2 + 16 * q - 12 = 0) ∧ 
  (4 * r^3 - 8 * r^2 + 16 * r - 12 = 0) →
  p * q + q * r + r * p = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_product_of_roots_l4104_410474


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l4104_410421

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 := by sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l4104_410421


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l4104_410466

theorem reciprocal_sum_pairs : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)),
    pairs.card = count ∧
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 3) ∧
    count = 3 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l4104_410466


namespace NUMINAMATH_CALUDE_kenny_time_ratio_l4104_410429

/-- Proves that the ratio of Kenny's running time to basketball playing time is 2:1 -/
theorem kenny_time_ratio : 
  ∀ (basketball_time trumpet_time running_time : ℕ),
    basketball_time = 10 →
    trumpet_time = 40 →
    trumpet_time = 2 * running_time →
    running_time / basketball_time = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_kenny_time_ratio_l4104_410429


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l4104_410400

theorem smallest_integer_bound (a b c d e f : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f →  -- 6 different integers
  (a + b + c + d + e + f) / 6 = 85 →  -- average is 85
  f = 180 →  -- largest is 180
  e = 100 →  -- second largest is 100
  a ≥ -64 :=  -- smallest is not less than -64
by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l4104_410400


namespace NUMINAMATH_CALUDE_k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l4104_410449

-- Define the polynomials A and B
def A (k : ℝ) (x : ℝ) : ℝ := -2 * x^2 - (k - 1) * x + 1
def B (x : ℝ) : ℝ := -2 * (x^2 - x + 2)

-- Define what it means for a polynomial to be a quadratic binomial
def is_quadratic_binomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x, p x = a * x^2 + b * x + c

-- Theorem 1: When A is a quadratic binomial, k = 1
theorem k_value_when_A_is_quadratic_binomial :
  ∀ k : ℝ, is_quadratic_binomial (A k) → k = 1 :=
sorry

-- Theorem 2: When k = -1 and C + 2A = B, then C = 2x^2 - 2x - 6
theorem C_value_when_k_is_negative_one :
  ∀ C : ℝ → ℝ, (∀ x, C x + 2 * A (-1) x = B x) →
  ∀ x, C x = 2 * x^2 - 2 * x - 6 :=
sorry

end NUMINAMATH_CALUDE_k_value_when_A_is_quadratic_binomial_C_value_when_k_is_negative_one_l4104_410449


namespace NUMINAMATH_CALUDE_rods_in_mile_l4104_410479

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 10

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 50

/-- Theorem stating that one mile is equal to 500 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 500 := by
  sorry

end NUMINAMATH_CALUDE_rods_in_mile_l4104_410479


namespace NUMINAMATH_CALUDE_complex_fraction_equation_l4104_410435

theorem complex_fraction_equation : 
  (13/6 : ℚ) + ((432/100 - 168/100 - 33/25) * (5/11) - 2/7) / (44/35) = 521/210 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equation_l4104_410435


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l4104_410452

theorem no_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(x - 2*a > 0 ∧ 3 - 2*x > x - 6)) → a ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l4104_410452


namespace NUMINAMATH_CALUDE_equal_fractions_imply_x_equals_four_l4104_410420

theorem equal_fractions_imply_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_x_equals_four_l4104_410420


namespace NUMINAMATH_CALUDE_angle_at_larger_base_l4104_410433

/-- An isosceles trapezoid with a regular triangle on its smaller base -/
structure IsoscelesTrapezoidWithTriangle where
  /-- The smaller base of the trapezoid -/
  smallerBase : ℝ
  /-- The larger base of the trapezoid -/
  largerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The angle at the larger base of the trapezoid -/
  angle : ℝ
  /-- The area of the trapezoid -/
  areaT : ℝ
  /-- The area of the triangle -/
  areaTriangle : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The triangle is regular -/
  isRegular : True
  /-- The height of the triangle equals the height of the trapezoid -/
  heightEqual : True
  /-- The area of the triangle is 5 times less than the area of the trapezoid -/
  areaRelation : areaT = 5 * areaTriangle

/-- The theorem to be proved -/
theorem angle_at_larger_base (t : IsoscelesTrapezoidWithTriangle) :
  t.angle = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_at_larger_base_l4104_410433


namespace NUMINAMATH_CALUDE_function_properties_l4104_410413

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.cos x * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * (Real.sin x)^2

theorem function_properties (a : ℝ) :
  f a (Real.pi / 12) = 0 →
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  T = Real.pi ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = -2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4104_410413


namespace NUMINAMATH_CALUDE_average_marks_l4104_410403

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 93 := by sorry

end NUMINAMATH_CALUDE_average_marks_l4104_410403


namespace NUMINAMATH_CALUDE_smallest_coverage_l4104_410472

/-- Represents a checkerboard configuration -/
structure CheckerBoard :=
  (rows : Nat)
  (cols : Nat)
  (checkers : Nat)
  (at_most_one_per_square : checkers ≤ rows * cols)

/-- Defines the coverage property for a given k -/
def covers (board : CheckerBoard) (k : Nat) : Prop :=
  ∀ (arrangement : Fin board.checkers → Fin board.rows × Fin board.cols),
    ∃ (rows : Fin k → Fin board.rows) (cols : Fin k → Fin board.cols),
      ∀ (c : Fin board.checkers),
        (arrangement c).1 ∈ Set.range rows ∨ (arrangement c).2 ∈ Set.range cols

/-- The main theorem statement -/
theorem smallest_coverage (board : CheckerBoard) 
  (h_rows : board.rows = 2011)
  (h_cols : board.cols = 2011)
  (h_checkers : board.checkers = 3000) :
  (covers board 1006 ∧ ∀ k < 1006, ¬covers board k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_coverage_l4104_410472


namespace NUMINAMATH_CALUDE_beef_cost_theorem_l4104_410476

/-- Calculates the total cost of beef given the number of packs, pounds per pack, and price per pound. -/
def total_cost (num_packs : ℕ) (pounds_per_pack : ℕ) (price_per_pound : ℚ) : ℚ :=
  (num_packs * pounds_per_pack : ℚ) * price_per_pound

/-- Proves that given 5 packs of beef, 4 pounds per pack, and a price of $5.50 per pound, the total cost is $110. -/
theorem beef_cost_theorem :
  total_cost 5 4 (11/2) = 110 := by
  sorry

end NUMINAMATH_CALUDE_beef_cost_theorem_l4104_410476


namespace NUMINAMATH_CALUDE_sum_30_45_base3_l4104_410410

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else go (m / 3) ((m % 3) :: acc)
  go n []

/-- Theorem: The sum of 30 and 45 in base 10 is equal to 22010 in base 3 -/
theorem sum_30_45_base3 : toBase3 (30 + 45) = [2, 2, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_30_45_base3_l4104_410410


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4104_410401

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (3 + Real.sqrt (4 * y - 5)) = Real.sqrt 8 → y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4104_410401


namespace NUMINAMATH_CALUDE_nth_term_is_3012_l4104_410494

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem nth_term_is_3012 (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 7 * x - 14
  let a₃ := 4 * x + 6
  (∃ n : ℕ, arithmetic_sequence a₁ a₂ a₃ n = 3012) →
  (∃ n : ℕ, n = 392 ∧ arithmetic_sequence a₁ a₂ a₃ n = 3012) :=
by
  sorry

#check nth_term_is_3012

end NUMINAMATH_CALUDE_nth_term_is_3012_l4104_410494


namespace NUMINAMATH_CALUDE_smallest_mersenne_prime_above_30_l4104_410488

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem smallest_mersenne_prime_above_30 :
  ∃ p : ℕ, mersenne_prime p ∧ p > 30 ∧ 
  ∀ q : ℕ, mersenne_prime q → q > 30 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_mersenne_prime_above_30_l4104_410488


namespace NUMINAMATH_CALUDE_platform_length_l4104_410473

/-- Given a train crossing a platform, calculate the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 225) 
  (h2 : train_speed_kmph = 90) 
  (h3 : crossing_time = 25) : 
  ℝ := by
  
  -- Convert train speed from km/h to m/s
  let train_speed_ms := train_speed_kmph * 1000 / 3600

  -- Calculate total distance covered (train + platform)
  let total_distance := train_speed_ms * crossing_time

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that the platform length is 400 meters
  have : platform_length = 400 := by sorry

  -- Return the platform length
  exact platform_length


end NUMINAMATH_CALUDE_platform_length_l4104_410473


namespace NUMINAMATH_CALUDE_gcd_90_252_l4104_410486

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_252_l4104_410486


namespace NUMINAMATH_CALUDE_divisibility_by_six_l4104_410458

theorem divisibility_by_six (y : ℕ) : y < 10 → (62000 + y * 100 + 16) % 6 = 0 ↔ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l4104_410458


namespace NUMINAMATH_CALUDE_equation_solutions_count_l4104_410498

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), (∀ x ∈ s, 9 * x^2 - 63 * ⌊x⌋ + 72 = 0) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l4104_410498


namespace NUMINAMATH_CALUDE_magnitude_a_plus_2b_eq_sqrt_2_l4104_410497

def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b : Fin 2 → ℝ := ![1, -2]

theorem magnitude_a_plus_2b_eq_sqrt_2 :
  Real.sqrt ((a 0 + 2 * b 0)^2 + (a 1 + 2 * b 1)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_2b_eq_sqrt_2_l4104_410497


namespace NUMINAMATH_CALUDE_max_quarters_l4104_410406

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The total amount Sasha has in dollars -/
def total_amount : ℚ := 4.50

/-- Proves that the maximum number of quarters (and dimes) Sasha can have is 12 -/
theorem max_quarters : 
  ∀ q : ℕ, 
    (q : ℚ) * (quarter_value + dime_value) ≤ total_amount → 
    q ≤ 12 := by
  sorry

#check max_quarters

end NUMINAMATH_CALUDE_max_quarters_l4104_410406


namespace NUMINAMATH_CALUDE_max_axes_of_symmetry_is_six_l4104_410423

/-- A line segment in a plane -/
structure LineSegment where
  -- Define properties of a line segment here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- A configuration of three line segments in a plane -/
structure ThreeSegmentConfiguration where
  segments : Fin 3 → LineSegment

/-- An axis of symmetry for a configuration of line segments -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry here
  -- For simplicity, we'll just use a placeholder
  id : Nat

/-- The set of axes of symmetry for a given configuration -/
def axesOfSymmetry (config : ThreeSegmentConfiguration) : Set AxisOfSymmetry :=
  sorry

/-- The maximum number of axes of symmetry for any configuration of three line segments -/
def maxAxesOfSymmetry : Nat :=
  sorry

theorem max_axes_of_symmetry_is_six :
  maxAxesOfSymmetry = 6 :=
sorry

end NUMINAMATH_CALUDE_max_axes_of_symmetry_is_six_l4104_410423
