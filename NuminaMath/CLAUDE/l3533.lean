import Mathlib

namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l3533_353345

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / total_selections

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l3533_353345


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3533_353367

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 70 ∧ x - y = 10 → x * y = 1200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3533_353367


namespace NUMINAMATH_CALUDE_target_cube_volume_l3533_353314

-- Define the volume of the reference cube
def reference_volume : ℝ := 8

-- Define the surface area of the target cube in terms of the reference cube
def target_surface_area (reference_side : ℝ) : ℝ := 2 * (6 * reference_side^2)

-- Define the volume of a cube given its side length
def cube_volume (side : ℝ) : ℝ := side^3

-- Define the surface area of a cube given its side length
def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem target_cube_volume :
  ∃ (target_side : ℝ),
    cube_surface_area target_side = target_surface_area (reference_volume^(1/3)) ∧
    cube_volume target_side = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_target_cube_volume_l3533_353314


namespace NUMINAMATH_CALUDE_tomato_production_relationship_l3533_353326

/-- Represents the production of three tomato plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def tomato_problem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.second = (p.first / 2) + 5 ∧
  p.first + p.second + p.third = 60

theorem tomato_production_relationship (p : TomatoProduction) 
  (h : tomato_problem p) : p.third = p.second + 2 := by
  sorry

#check tomato_production_relationship

end NUMINAMATH_CALUDE_tomato_production_relationship_l3533_353326


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l3533_353325

theorem perpendicular_vectors_tan_theta (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h2 : a = (Real.cos θ, 2))
  (h3 : b = (-1, Real.sin θ))
  (h4 : a.1 * b.1 + a.2 * b.2 = 0) : 
  Real.tan θ = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l3533_353325


namespace NUMINAMATH_CALUDE_complete_square_h_l3533_353378

theorem complete_square_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_complete_square_h_l3533_353378


namespace NUMINAMATH_CALUDE_intersection_equals_set_iff_complement_subset_l3533_353391

universe u

theorem intersection_equals_set_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = A ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end NUMINAMATH_CALUDE_intersection_equals_set_iff_complement_subset_l3533_353391


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_right_triangle_perimeter_proof_l3533_353365

/-- The perimeter of a right triangle with legs 8 and 6 is 24. -/
theorem right_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun PQ QR PR =>
    QR = 8 ∧ PR = 6 ∧ PQ ^ 2 = QR ^ 2 + PR ^ 2 →
    PQ + QR + PR = 24

/-- Proof of the theorem -/
theorem right_triangle_perimeter_proof : right_triangle_perimeter 10 8 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_right_triangle_perimeter_proof_l3533_353365


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3533_353354

/-- Reflects a point about the line y = -x -/
def reflect_about_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (9, -4)
  let reflected_center := reflect_about_y_neg_x original_center
  reflected_center = (4, -9) := by sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3533_353354


namespace NUMINAMATH_CALUDE_third_group_size_l3533_353383

/-- The number of students in each group and the total number of students -/
structure StudentGroups where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  group4 : ℕ
  total : ℕ

/-- The theorem stating that the third group has 7 students -/
theorem third_group_size (sg : StudentGroups) 
  (h1 : sg.group1 = 5)
  (h2 : sg.group2 = 8)
  (h3 : sg.group4 = 4)
  (h4 : sg.total = 24)
  (h5 : sg.total = sg.group1 + sg.group2 + sg.group3 + sg.group4) :
  sg.group3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_group_size_l3533_353383


namespace NUMINAMATH_CALUDE_system_solution_l3533_353370

theorem system_solution (x y k : ℝ) : 
  (4 * x + 2 * y = 5 * k - 4) → 
  (2 * x + 4 * y = -1) → 
  (x - y = 1) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3533_353370


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3533_353305

theorem quadratic_form_h_value (a b c : ℝ) :
  (∃ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 7) →
  (∃ n k, ∀ x, 4 * (a * x^2 + b * x + c) = n * (x - 5)^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3533_353305


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3533_353306

theorem arithmetic_calculation : 2^3 + 4 * 5 - 6 + 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3533_353306


namespace NUMINAMATH_CALUDE_solution_set_max_t_value_l3533_353343

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution of f(x) < 5
theorem solution_set (x : ℝ) : f x < 5 ↔ -7/4 < x ∧ x < 3/4 := by sorry

-- Theorem for the maximum value of t
theorem max_t_value : ∃ (a : ℝ), a = 4 ∧ ∀ (t : ℝ), (∀ (x : ℝ), f x - t ≥ 0) ↔ t ≤ a := by sorry

end NUMINAMATH_CALUDE_solution_set_max_t_value_l3533_353343


namespace NUMINAMATH_CALUDE_negative_463_terminal_side_l3533_353304

-- Define the concept of terminal side equality for angles
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_463_terminal_side :
  ∀ K : ℤ, same_terminal_side (-463) (K * 360 + 257) :=
sorry

end NUMINAMATH_CALUDE_negative_463_terminal_side_l3533_353304


namespace NUMINAMATH_CALUDE_range_of_m_l3533_353398

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the condition that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m, sufficient_not_necessary m ↔ (2 < m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3533_353398


namespace NUMINAMATH_CALUDE_appropriate_sampling_methods_l3533_353359

-- Define the structure for a survey
structure Survey where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool

-- Define the sampling methods
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling

-- Define the function to determine the appropriate sampling method
def appropriate_sampling_method (survey : Survey) : SamplingMethod :=
  if survey.has_distinct_groups && survey.total_population > survey.sample_size * 10
  then SamplingMethod.StratifiedSampling
  else SamplingMethod.SimpleRandomSampling

-- Define the surveys
def survey1 : Survey := {
  total_population := 125 + 280 + 95,
  sample_size := 100,
  has_distinct_groups := true
}

def survey2 : Survey := {
  total_population := 15,
  sample_size := 3,
  has_distinct_groups := false
}

-- Theorem to prove
theorem appropriate_sampling_methods :
  appropriate_sampling_method survey1 = SamplingMethod.StratifiedSampling ∧
  appropriate_sampling_method survey2 = SamplingMethod.SimpleRandomSampling :=
by sorry


end NUMINAMATH_CALUDE_appropriate_sampling_methods_l3533_353359


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l3533_353368

/-- A linear function passing through three given points. -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- The function passes through the given points. -/
def PassesThroughPoints (k b : ℝ) : Prop :=
  LinearFunction k b (-2) = 7 ∧
  LinearFunction k b 1 = 4 ∧
  LinearFunction k b 3 = 2

/-- The function passes through the first, second, and fourth quadrants. -/
def PassesThroughQuadrants (k b : ℝ) : Prop :=
  (∃ x y, x > 0 ∧ y > 0 ∧ LinearFunction k b x = y) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ LinearFunction k b x = y) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ LinearFunction k b x = y)

/-- Main theorem: If the linear function passes through the given points,
    then it passes through the first, second, and fourth quadrants. -/
theorem linear_function_quadrants (k b : ℝ) (h : k ≠ 0) :
  PassesThroughPoints k b → PassesThroughQuadrants k b :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_quadrants_l3533_353368


namespace NUMINAMATH_CALUDE_equation_root_implies_m_value_l3533_353350

theorem equation_root_implies_m_value (m x : ℝ) : 
  (m / (x - 3) - 1 / (3 - x) = 2) → 
  (∃ x > 0, m / (x - 3) - 1 / (3 - x) = 2) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_implies_m_value_l3533_353350


namespace NUMINAMATH_CALUDE_percentage_calculation_l3533_353356

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 500) (h2 : part = 125) :
  (part / total) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3533_353356


namespace NUMINAMATH_CALUDE_max_square_sum_max_square_sum_achievable_l3533_353389

/-- The maximum value of x^2 + y^2 given 0 ≤ x ≤ 1 and 0 ≤ y ≤ 1 -/
theorem max_square_sum : 
  ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → x^2 + y^2 ≤ 2 := by
  sorry

/-- The maximum value 2 is achievable -/
theorem max_square_sum_achievable : 
  ∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x^2 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_square_sum_max_square_sum_achievable_l3533_353389


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_jason_initial_cards_l3533_353382

theorem jason_pokemon_cards : ℕ → Prop :=
  fun initial_cards =>
    let given_away := 9
    let remaining := 4
    initial_cards = given_away + remaining

theorem jason_initial_cards : ∃ x : ℕ, jason_pokemon_cards x ∧ x = 13 :=
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_jason_initial_cards_l3533_353382


namespace NUMINAMATH_CALUDE_remaining_money_for_gas_and_maintenance_l3533_353317

def monthly_income : ℕ := 3200

def rent : ℕ := 1250
def utilities : ℕ := 150
def retirement_savings : ℕ := 400
def groceries : ℕ := 300
def insurance : ℕ := 200
def miscellaneous : ℕ := 200
def car_payment : ℕ := 350

def total_expenses : ℕ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + car_payment

theorem remaining_money_for_gas_and_maintenance :
  monthly_income - total_expenses = 350 := by sorry

end NUMINAMATH_CALUDE_remaining_money_for_gas_and_maintenance_l3533_353317


namespace NUMINAMATH_CALUDE_cherry_tomato_yield_theorem_l3533_353321

-- Define the relationship between number of plants and yield
def yield_function (x : ℕ) : ℝ := -0.5 * (x : ℝ) + 5

-- Define the total yield function
def total_yield (x : ℕ) : ℝ := (x : ℝ) * yield_function x

-- Theorem statement
theorem cherry_tomato_yield_theorem (x : ℕ) 
  (h1 : 2 ≤ x ∧ x ≤ 8) 
  (h2 : yield_function 2 = 4) 
  (h3 : ∀ n : ℕ, 2 ≤ n ∧ n < 8 → yield_function (n + 1) = yield_function n - 0.5) :
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → yield_function n = -0.5 * (n : ℝ) + 5) ∧
  (∃ max_yield : ℝ, max_yield = 12.5 ∧ 
    ∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → total_yield n ≤ max_yield) ∧
  (total_yield 5 = 12.5) :=
by sorry

end NUMINAMATH_CALUDE_cherry_tomato_yield_theorem_l3533_353321


namespace NUMINAMATH_CALUDE_school_time_problem_l3533_353351

/-- Given a boy who reaches school 3 minutes early when walking at 7/6 of his usual rate,
    his usual time to reach school is 21 minutes. -/
theorem school_time_problem (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0)
  (h2 : usual_time > 0)
  (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 3)) :
  usual_time = 21 := by
  sorry

end NUMINAMATH_CALUDE_school_time_problem_l3533_353351


namespace NUMINAMATH_CALUDE_original_number_is_four_fifths_l3533_353361

theorem original_number_is_four_fifths (x : ℚ) :
  1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_four_fifths_l3533_353361


namespace NUMINAMATH_CALUDE_like_terms_sum_l3533_353395

theorem like_terms_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^2 * y^4 = 3 * y - x^n * y^(2*m)) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l3533_353395


namespace NUMINAMATH_CALUDE_cricket_matches_l3533_353316

theorem cricket_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 60)
  (h2 : score2 = 50)
  (h3 : overall_avg = 54)
  (h4 : matches1 = 2)
  (h5 : matches2 = 3) :
  matches1 + matches2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_matches_l3533_353316


namespace NUMINAMATH_CALUDE_solution_pairs_l3533_353315

-- Define the predicate for the conditions
def satisfies_conditions (x y : ℕ+) : Prop :=
  (y ∣ x^2 + 1) ∧ (x^2 ∣ y^3 + 1)

-- State the theorem
theorem solution_pairs :
  ∀ x y : ℕ+, satisfies_conditions x y →
    ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3533_353315


namespace NUMINAMATH_CALUDE_cube_of_negative_two_ab_squared_l3533_353349

theorem cube_of_negative_two_ab_squared (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_ab_squared_l3533_353349


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l3533_353333

def a : ℝ × ℝ := (1, 2)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem perpendicular_vectors_magnitude (y : ℝ) 
  (h : a.1 * (b y).1 + a.2 * (b y).2 = 0) : 
  ‖(2 : ℝ) • a + b y‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l3533_353333


namespace NUMINAMATH_CALUDE_ryan_english_hours_l3533_353358

/-- Ryan's learning schedule --/
structure LearningSchedule where
  days : ℕ
  english_total : ℕ
  chinese_daily : ℕ

/-- Calculates the daily hours spent on learning English --/
def daily_english_hours (schedule : LearningSchedule) : ℚ :=
  schedule.english_total / schedule.days

/-- Theorem stating Ryan's daily English learning hours --/
theorem ryan_english_hours (schedule : LearningSchedule) 
  (h1 : schedule.days = 2)
  (h2 : schedule.english_total = 12)
  (h3 : schedule.chinese_daily = 5) :
  daily_english_hours schedule = 6 := by
  sorry


end NUMINAMATH_CALUDE_ryan_english_hours_l3533_353358


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_max_l3533_353387

theorem inscribed_rectangle_area_max (R : ℝ) (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 4*R^2) : 
  x * y ≤ 2 * R^2 ∧ (x * y = 2 * R^2 ↔ x = y ∧ x = R * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_max_l3533_353387


namespace NUMINAMATH_CALUDE_sum_first_50_digits_1_101_l3533_353313

/-- The decimal representation of 1/101 -/
def decimal_rep_1_101 : ℕ → ℕ
| 0 => 0  -- First digit after decimal point
| 1 => 0  -- Second digit
| 2 => 9  -- Third digit
| 3 => 9  -- Fourth digit
| n + 4 => decimal_rep_1_101 n  -- Repeating pattern

/-- Sum of the first n digits after the decimal point in 1/101 -/
def sum_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_rep_1_101 |>.sum

/-- The sum of the first 50 digits after the decimal point in the decimal representation of 1/101 is 216 -/
theorem sum_first_50_digits_1_101 : sum_digits 50 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_50_digits_1_101_l3533_353313


namespace NUMINAMATH_CALUDE_logarithm_inequality_and_root_comparison_l3533_353340

theorem logarithm_inequality_and_root_comparison : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (((a + b) / 2) : ℝ) ≥ (Real.log a + Real.log b) / 2) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_inequality_and_root_comparison_l3533_353340


namespace NUMINAMATH_CALUDE_daniels_painting_area_l3533_353347

/-- Calculate the total paintable wall area for multiple bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_rooms * paintable_area

/-- The problem statement -/
theorem daniels_painting_area :
  total_paintable_area 4 15 12 9 80 = 1624 := by
  sorry

end NUMINAMATH_CALUDE_daniels_painting_area_l3533_353347


namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_2_l3533_353375

def sumOfIntegersEndingIn2 (lower upper : ℕ) : ℕ :=
  let firstTerm := (lower + 2 - lower % 10)
  let lastTerm := (upper - upper % 10 + 2)
  let numTerms := (lastTerm - firstTerm) / 10 + 1
  numTerms * (firstTerm + lastTerm) / 2

theorem sum_of_integers_ending_in_2 :
  sumOfIntegersEndingIn2 60 460 = 10280 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_2_l3533_353375


namespace NUMINAMATH_CALUDE_remainder_of_3_99_plus_5_mod_9_l3533_353364

theorem remainder_of_3_99_plus_5_mod_9 : (3^99 + 5) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_99_plus_5_mod_9_l3533_353364


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3533_353381

/-- Given a quadratic function f(x) = ax^2 - (1/2)x + c where a and c are real numbers,
    f(1) = 0, and f(x) ≥ 0 for all real x, prove that a = 1/4, c = 1/4, and
    there exists m = 3 such that g(x) = 4f(x) - mx has a minimum value of -5 
    in the interval [m, m+2] -/
theorem quadratic_function_properties (a c : ℝ) 
    (f : ℝ → ℝ)
    (h1 : ∀ x, f x = a * x^2 - (1/2) * x + c)
    (h2 : f 1 = 0)
    (h3 : ∀ x, f x ≥ 0) :
    a = (1/4) ∧ c = (1/4) ∧
    ∃ m : ℝ, m = 3 ∧
    (∀ x ∈ Set.Icc m (m + 2), 4 * (f x) - m * x ≥ -5) ∧
    (∃ x₀ ∈ Set.Icc m (m + 2), 4 * (f x₀) - m * x₀ = -5) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3533_353381


namespace NUMINAMATH_CALUDE_system_solutions_l3533_353307

/-- The system of equations has exactly eight solutions -/
theorem system_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    solutions.card = 8 ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      ((x - 2)^2 + (y + 1)^2 = 5 ∧
       (x - 2)^2 + (z - 3)^2 = 13 ∧
       (y + 1)^2 + (z - 3)^2 = 10)) ∧
    solutions = {(0, 0, 0), (0, -2, 0), (0, 0, 6), (0, -2, 6),
                 (4, 0, 0), (4, -2, 0), (4, 0, 6), (4, -2, 6)} := by
  sorry


end NUMINAMATH_CALUDE_system_solutions_l3533_353307


namespace NUMINAMATH_CALUDE_colored_grid_rectangle_exists_l3533_353380

-- Define the color type
inductive Color
  | Red
  | White
  | Blue

-- Define the grid type
def Grid := Fin 12 → Fin 12 → Color

-- Define a rectangle in the grid
structure Rectangle where
  x1 : Fin 12
  y1 : Fin 12
  x2 : Fin 12
  y2 : Fin 12
  h_x : x1 < x2
  h_y : y1 < y2

-- Define a function to check if a rectangle has all vertices of the same color
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.x1 r.y1 = g r.x1 r.y2 ∧
  g r.x1 r.y1 = g r.x2 r.y1 ∧
  g r.x1 r.y1 = g r.x2 r.y2

-- State the theorem
theorem colored_grid_rectangle_exists (g : Grid) :
  ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end NUMINAMATH_CALUDE_colored_grid_rectangle_exists_l3533_353380


namespace NUMINAMATH_CALUDE_unique_solution_3x_minus_5y_equals_z2_l3533_353336

theorem unique_solution_3x_minus_5y_equals_z2 :
  ∀ x y z : ℕ+, 3^(x : ℕ) - 5^(y : ℕ) = (z : ℕ)^2 → x = 2 ∧ y = 2 ∧ z = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3x_minus_5y_equals_z2_l3533_353336


namespace NUMINAMATH_CALUDE_minimum_value_and_inequality_l3533_353394

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- State the theorem
theorem minimum_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 6) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 → a^2 + b^2 + c^2 ≥ 12) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_inequality_l3533_353394


namespace NUMINAMATH_CALUDE_age_difference_l3533_353362

/-- Given three people A, B, and C, with B being 8 years old, B twice as old as C,
    and the total of their ages being 22, prove that A is 2 years older than B. -/
theorem age_difference (B C : ℕ) (A : ℕ) : 
  B = 8 → B = 2 * C → A + B + C = 22 → A = B + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3533_353362


namespace NUMINAMATH_CALUDE_customers_added_l3533_353397

theorem customers_added (initial : ℕ) (no_tip : ℕ) (tip : ℕ) : 
  initial = 39 → no_tip = 49 → tip = 2 → 
  (no_tip + tip) - initial = 12 := by
  sorry

end NUMINAMATH_CALUDE_customers_added_l3533_353397


namespace NUMINAMATH_CALUDE_sharons_journey_l3533_353399

/-- The distance between Sharon's house and her mother's house -/
def total_distance : ℝ := 200

/-- The time Sharon usually takes to complete the journey -/
def usual_time : ℝ := 200

/-- The time taken on the day with traffic -/
def traffic_time : ℝ := 300

/-- The speed reduction due to traffic in miles per hour -/
def speed_reduction : ℝ := 30

theorem sharons_journey :
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    initial_speed * usual_time = total_distance ∧
    (total_distance / 2) / initial_speed +
    (total_distance / 2) / (initial_speed - speed_reduction / 60) = traffic_time :=
  sorry

end NUMINAMATH_CALUDE_sharons_journey_l3533_353399


namespace NUMINAMATH_CALUDE_total_students_amc8_l3533_353346

/-- Represents a math teacher at Pythagoras Middle School -/
inductive Teacher : Type
| Euler : Teacher
| Noether : Teacher
| Gauss : Teacher
| Riemann : Teacher

/-- Returns the number of students in a teacher's class -/
def studentsInClass (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 13
  | Teacher.Noether => 10
  | Teacher.Gauss => 12
  | Teacher.Riemann => 7

/-- The list of all teachers at Pythagoras Middle School -/
def allTeachers : List Teacher :=
  [Teacher.Euler, Teacher.Noether, Teacher.Gauss, Teacher.Riemann]

/-- Theorem stating that the total number of students taking the AMC 8 contest is 42 -/
theorem total_students_amc8 :
  (allTeachers.map studentsInClass).sum = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_students_amc8_l3533_353346


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l3533_353335

theorem two_digit_number_sum (tens : ℕ) (units : ℕ) : 
  tens < 10 → 
  units = 6 * tens → 
  10 * tens + units = 16 → 
  tens + units = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l3533_353335


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3533_353303

/-- Converts a natural number to its binary representation as a list of digits (0 or 1) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinaryAux (m / 2) ((m % 2) :: acc)
    toBinaryAux n []

/-- Sums a list of natural numbers --/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits in the binary representation of 300 is 3 --/
theorem sum_of_binary_digits_300 :
  sumList (toBinary 300) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_300_l3533_353303


namespace NUMINAMATH_CALUDE_cafeteria_stacking_l3533_353388

theorem cafeteria_stacking (initial_cartons : ℕ) (cartons_per_stack : ℕ) (teacher_cartons : ℕ) :
  initial_cartons = 799 →
  cartons_per_stack = 6 →
  teacher_cartons = 23 →
  let remaining_cartons := initial_cartons - teacher_cartons
  let full_stacks := remaining_cartons / cartons_per_stack
  let double_stacks := full_stacks / 2
  let leftover_cartons := remaining_cartons % cartons_per_stack + (full_stacks % 2) * cartons_per_stack
  double_stacks = 64 ∧ leftover_cartons = 8 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_stacking_l3533_353388


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3533_353302

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3533_353302


namespace NUMINAMATH_CALUDE_product_in_S_and_counterexample_l3533_353308

def S : Set ℤ := {x | ∃ n : ℤ, x = n^2 + n + 1}

theorem product_in_S_and_counterexample :
  (∀ n : ℤ, (n^2 + n + 1) * ((n+1)^2 + (n+1) + 1) ∈ S) ∧
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ a * b ∉ S) := by
  sorry

end NUMINAMATH_CALUDE_product_in_S_and_counterexample_l3533_353308


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l3533_353327

/-- Calculates the tip percentage given the total bill, food price, and sales tax rate. -/
def calculate_tip_percentage (total_bill : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let total_before_tip := food_price + sales_tax
  let tip_amount := total_bill - total_before_tip
  (tip_amount / total_before_tip) * 100

/-- Proves that the tip percentage is 20% given the specified conditions. -/
theorem tip_percentage_is_twenty_percent :
  calculate_tip_percentage 158.40 120 0.10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l3533_353327


namespace NUMINAMATH_CALUDE_nested_squares_perimeter_difference_l3533_353372

/-- The difference between the perimeters of two nested squares -/
theorem nested_squares_perimeter_difference :
  ∀ (x : ℝ),
  x > 0 →
  let small_square_side : ℝ := x
  let large_square_side : ℝ := x + 8
  let small_perimeter : ℝ := 4 * small_square_side
  let large_perimeter : ℝ := 4 * large_square_side
  large_perimeter - small_perimeter = 32 :=
by
  sorry

#check nested_squares_perimeter_difference

end NUMINAMATH_CALUDE_nested_squares_perimeter_difference_l3533_353372


namespace NUMINAMATH_CALUDE_divisibility_condition_l3533_353376

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3533_353376


namespace NUMINAMATH_CALUDE_keegan_class_time_l3533_353363

theorem keegan_class_time (total_hours : Real) (num_classes : Nat) (other_class_time : Real) :
  total_hours = 7.5 →
  num_classes = 7 →
  other_class_time = 72 / 60 →
  let other_classes_time := other_class_time * (num_classes - 2 : Real)
  let history_chem_time := total_hours - other_classes_time
  history_chem_time = 1.5 := by
sorry

end NUMINAMATH_CALUDE_keegan_class_time_l3533_353363


namespace NUMINAMATH_CALUDE_inequality_solution_l3533_353396

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≤ 5/4) ↔ (x < -8 ∨ (-2 < x ∧ x ≤ -8/5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3533_353396


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3533_353353

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N = !![2, 1; 7, -2] ∧
  N.mulVec ![2, 0] = ![4, 14] ∧
  N.mulVec ![-2, 10] = ![6, -34] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3533_353353


namespace NUMINAMATH_CALUDE_cos_2phi_nonpositive_l3533_353328

theorem cos_2phi_nonpositive (α β φ : Real) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_2phi_nonpositive_l3533_353328


namespace NUMINAMATH_CALUDE_job_completion_time_B_l3533_353355

theorem job_completion_time_B (r_A r_B r_C : ℝ) : 
  (r_A + r_B = 1 / 3) →
  (r_B + r_C = 2 / 7) →
  (r_A + r_C = 1 / 4) →
  (1 / r_B = 168 / 31) :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_B_l3533_353355


namespace NUMINAMATH_CALUDE_pharmacy_tubs_l3533_353390

def tubs_needed : ℕ := 100
def tubs_in_storage : ℕ := 20

def tubs_to_buy : ℕ := tubs_needed - tubs_in_storage

def tubs_from_new_vendor : ℕ := tubs_to_buy / 4

def tubs_from_usual_vendor : ℕ := tubs_needed - (tubs_in_storage + tubs_from_new_vendor)

theorem pharmacy_tubs :
  tubs_from_usual_vendor = 60 := by sorry

end NUMINAMATH_CALUDE_pharmacy_tubs_l3533_353390


namespace NUMINAMATH_CALUDE_largest_number_problem_l3533_353344

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 75)
  (h_diff_large : c - b = 5)
  (h_diff_small : b - a = 4) :
  c = 89 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l3533_353344


namespace NUMINAMATH_CALUDE_analysis_time_per_bone_l3533_353379

/-- The number of bones in the human body -/
def num_bones : ℕ := 206

/-- The total time Aria needs to analyze all bones (in hours) -/
def total_analysis_time : ℕ := 1030

/-- The time spent analyzing each bone (in hours) -/
def time_per_bone : ℚ := total_analysis_time / num_bones

theorem analysis_time_per_bone : time_per_bone = 5 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_per_bone_l3533_353379


namespace NUMINAMATH_CALUDE_symmetry_center_l3533_353369

open Real

/-- Given a function f and its symmetric function g, prove that (π/4, 0) is a center of symmetry of g -/
theorem symmetry_center (f g : ℝ → ℝ) : 
  (∀ x, f x = sin (2*x + π/6)) →
  (∀ x, f (π/6 - x) = g x) →
  (π/4, 0) ∈ {p : ℝ × ℝ | ∀ x, g (p.1 + x) = g (p.1 - x)} :=
by sorry

end NUMINAMATH_CALUDE_symmetry_center_l3533_353369


namespace NUMINAMATH_CALUDE_complex_power_195_deg_36_l3533_353338

theorem complex_power_195_deg_36 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 36 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_195_deg_36_l3533_353338


namespace NUMINAMATH_CALUDE_equation_has_real_root_l3533_353360

theorem equation_has_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a > b) (hbc : b > c) : 
  ∃ x : ℝ, (1 / (x + a) + 1 / (x + b) + 1 / (x + c) - 3 / x = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l3533_353360


namespace NUMINAMATH_CALUDE_range_of_a_for_union_complement_l3533_353322

-- Define the sets M, N, and A
def M : Set ℝ := {x | 1 < x ∧ x < 4}
def N : Set ℝ := {x | 3 < x ∧ x < 5}
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- Define the theorem
theorem range_of_a_for_union_complement (a : ℝ) : 
  (A a ∪ (Set.univ \ N) = Set.univ) ↔ (2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_union_complement_l3533_353322


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_relation_l3533_353386

theorem angle_with_special_supplement_complement_relation :
  ∀ x : ℝ,
  (0 < x) ∧ (x < 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_relation_l3533_353386


namespace NUMINAMATH_CALUDE_circle_equation_l3533_353309

/-- The equation of a circle with center (±2, 1) and radius 2, given specific conditions -/
theorem circle_equation (x y : ℝ) : 
  (∃ t : ℝ, t = 2 ∨ t = -2) →
  (1 : ℝ) = (1/4) * t^2 →
  (abs t = abs ((1/4) * t^2 + 1)) →
  (x^2 + y^2 + 4*x - 2*y - 1 = 0) ∨ (x^2 + y^2 - 4*x - 2*y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3533_353309


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l3533_353339

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l3533_353339


namespace NUMINAMATH_CALUDE_polynomial_fixed_point_l3533_353374

theorem polynomial_fixed_point (P : ℤ → ℤ) (h_poly : ∃ (coeffs : List ℤ), ∀ x, P x = (coeffs.map (λ (c : ℤ) (i : ℕ) => c * x ^ i)).sum) :
  P 1 = 2013 → P 2013 = 1 → ∃ k : ℤ, P k = k → k = 1007 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_fixed_point_l3533_353374


namespace NUMINAMATH_CALUDE_tory_cookie_sales_l3533_353334

/-- Proves that Tory sold 7 packs of cookies to his uncle given the problem conditions -/
theorem tory_cookie_sales : 
  ∀ (total_goal : ℕ) (sold_to_grandmother : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ),
    total_goal = 50 →
    sold_to_grandmother = 12 →
    sold_to_neighbor = 5 →
    remaining_to_sell = 26 →
    ∃ (sold_to_uncle : ℕ),
      sold_to_uncle = total_goal - remaining_to_sell - sold_to_grandmother - sold_to_neighbor ∧
      sold_to_uncle = 7 :=
by sorry

end NUMINAMATH_CALUDE_tory_cookie_sales_l3533_353334


namespace NUMINAMATH_CALUDE_number_problem_l3533_353301

theorem number_problem (x : ℝ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3533_353301


namespace NUMINAMATH_CALUDE_expected_reflections_l3533_353348

open Real

/-- The expected number of reflections for a billiard ball on a rectangular table -/
theorem expected_reflections (table_length table_width ball_travel : ℝ) :
  table_length = 3 →
  table_width = 1 →
  ball_travel = 2 →
  ∃ (expected_reflections : ℝ),
    expected_reflections = (2 / π) * (3 * arccos (1/4) - arcsin (3/4) + arccos (3/4)) := by
  sorry

end NUMINAMATH_CALUDE_expected_reflections_l3533_353348


namespace NUMINAMATH_CALUDE_cube_root_of_three_cubes_of_three_to_fifth_l3533_353371

theorem cube_root_of_three_cubes_of_three_to_fifth (x : ℝ) : 
  x = (3^5 + 3^5 + 3^5)^(1/3) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_three_cubes_of_three_to_fifth_l3533_353371


namespace NUMINAMATH_CALUDE_simplest_radical_l3533_353300

-- Define a function to check if a number is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ y ≠ 1

-- Define a function to check if a quadratic radical is in its simplest form
def is_simplest_radical (x : ℝ) : Prop :=
  is_quadratic_radical x ∧
  ∀ (y : ℝ), y ≠ x → (is_quadratic_radical y → ¬(x^2 = y^2))

-- Theorem statement
theorem simplest_radical :
  is_simplest_radical (Real.sqrt 7) ∧
  ¬(is_simplest_radical 1) ∧
  ¬(is_simplest_radical (Real.sqrt 12)) ∧
  ¬(is_simplest_radical (1 / Real.sqrt 13)) :=
sorry

end NUMINAMATH_CALUDE_simplest_radical_l3533_353300


namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l3533_353366

-- Define the expression
def expression (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 5| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  ∃ (some_expression : ℝ), 
    (∀ x : ℝ, expression x some_expression ≥ 10) ∧ 
    (∃ x : ℝ, expression x some_expression = 10) ∧
    |some_expression| = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_some_expression_l3533_353366


namespace NUMINAMATH_CALUDE_factorization_proof_l3533_353330

theorem factorization_proof (a b : ℝ) : a * b^2 - 5 * a * b = a * b * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3533_353330


namespace NUMINAMATH_CALUDE_stating_ancient_chinese_problem_correct_l3533_353323

/-- Represents the system of equations for the ancient Chinese mathematical problem. -/
def ancient_chinese_problem (x y : ℝ) : Prop :=
  (y = 8 * x - 3) ∧ (y = 7 * x + 4)

/-- 
Theorem stating that the system of equations correctly represents the given problem,
where x is the number of people and y is the price of the items in coins.
-/
theorem ancient_chinese_problem_correct (x y : ℝ) :
  ancient_chinese_problem x y ↔
  (∃ (total_price : ℝ),
    (8 * x = total_price + 3) ∧
    (7 * x = total_price - 4) ∧
    (y = total_price)) :=
by sorry

end NUMINAMATH_CALUDE_stating_ancient_chinese_problem_correct_l3533_353323


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l3533_353332

/-- Given an initial number of pencils and a number of pencils added, 
    calculate the total number of pencils -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that 27 initial pencils plus 45 added pencils equal 72 total pencils -/
theorem pencils_in_drawer : total_pencils 27 45 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l3533_353332


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l3533_353342

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l3533_353342


namespace NUMINAMATH_CALUDE_solve_for_y_l3533_353357

theorem solve_for_y : ∃ y : ℚ, ((2^5 : ℚ) * y) / ((8^2 : ℚ) * (3^5 : ℚ)) = 1/6 ∧ y = 81 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3533_353357


namespace NUMINAMATH_CALUDE_log_8_512_l3533_353312

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_8_512 : log 8 512 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_8_512_l3533_353312


namespace NUMINAMATH_CALUDE_february_to_january_ratio_l3533_353377

def january_bill : ℚ := 180

def february_bill : ℚ := 270

theorem february_to_january_ratio :
  (february_bill / january_bill) = 3 / 2 ∧
  ((february_bill + 30) / january_bill) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_february_to_january_ratio_l3533_353377


namespace NUMINAMATH_CALUDE_max_distance_C_D_l3533_353337

def C : Set ℂ := {z : ℂ | z^4 - 16 = 0}
def D : Set ℂ := {z : ℂ | z^3 - 27 = 0}

theorem max_distance_C_D : 
  ∃ (c : C) (d : D), ∀ (c' : C) (d' : D), Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧ 
  Complex.abs (c - d) = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_max_distance_C_D_l3533_353337


namespace NUMINAMATH_CALUDE_difference_of_squares_262_258_l3533_353324

theorem difference_of_squares_262_258 : 262^2 - 258^2 = 2080 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_262_258_l3533_353324


namespace NUMINAMATH_CALUDE_min_values_theorem_l3533_353384

theorem min_values_theorem (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h : (r + s - r * s) * (r + s + r * s) = r * s) : 
  (∃ (r' s' : ℝ), r' > 0 ∧ s' > 0 ∧ 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' ∧
    r + s - r * s ≥ -3 + 2 * Real.sqrt 3 ∧
    r + s + r * s ≥ 3 + 2 * Real.sqrt 3) ∧
  (∀ (r' s' : ℝ), r' > 0 → s' > 0 → 
    (r' + s' - r' * s') * (r' + s' + r' * s') = r' * s' →
    r' + s' - r' * s' ≥ -3 + 2 * Real.sqrt 3 ∧
    r' + s' + r' * s' ≥ 3 + 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_min_values_theorem_l3533_353384


namespace NUMINAMATH_CALUDE_unique_solution_l3533_353310

-- Define the equation
def equation (p x : ℝ) : Prop := x + 1 = Real.sqrt (p * x)

-- Define the conditions
def conditions (p x : ℝ) : Prop := p * x ≥ 0 ∧ x + 1 ≥ 0

-- Theorem statement
theorem unique_solution (p : ℝ) :
  (∃! x, equation p x ∧ conditions p x) ↔ (p = 4 ∨ p ≤ 0) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3533_353310


namespace NUMINAMATH_CALUDE_expression_value_l3533_353373

theorem expression_value : 
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3533_353373


namespace NUMINAMATH_CALUDE_special_polynomial_value_l3533_353331

/-- A polynomial of the form ± x^6 ± x^5 ± x^4 ± x^3 ± x^2 ± x ± 1 -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f g : ℤ), ∀ x,
    P x = a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g ∧
    (a = 1 ∨ a = -1) ∧ (b = 1 ∨ b = -1) ∧ (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧ (e = 1 ∨ e = -1) ∧ (f = 1 ∨ f = -1) ∧ (g = 1 ∨ g = -1)

theorem special_polynomial_value (P : ℝ → ℝ) :
  SpecialPolynomial P → P 2 = 27 → P 3 = 439 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l3533_353331


namespace NUMINAMATH_CALUDE_power_equality_l3533_353311

theorem power_equality (p : ℕ) (h : (81 : ℕ)^10 = 3^p) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3533_353311


namespace NUMINAMATH_CALUDE_derivative_f_l3533_353319

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_f :
  (∀ x : ℝ, x ≠ 0 → HasDerivAt f ((- 2 / x ^ 2) + 1) x) ∧
  HasDerivAt f (-1) 1 ∧
  HasDerivAt f (1/2) (-2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_l3533_353319


namespace NUMINAMATH_CALUDE_prob_sum_div_3_correct_l3533_353318

/-- The probability of the sum of three fair six-sided dice being divisible by 3 -/
def prob_sum_div_3 : ℚ :=
  13 / 27

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Finset ℕ :=
  Finset.range 6

/-- The probability of rolling any specific number on a fair six-sided die -/
def single_roll_prob : ℚ :=
  1 / 6

/-- The set of all possible outcomes when rolling three dice -/
def all_outcomes : Finset (ℕ × ℕ × ℕ) :=
  die_outcomes.product (die_outcomes.product die_outcomes)

/-- The sum of the numbers shown on three dice -/
def sum_of_dice (roll : ℕ × ℕ × ℕ) : ℕ :=
  roll.1 + roll.2.1 + roll.2.2 + 3

/-- The set of favorable outcomes (sum divisible by 3) -/
def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  all_outcomes.filter (λ roll ↦ (sum_of_dice roll) % 3 = 0)

theorem prob_sum_div_3_correct :
  (favorable_outcomes.card : ℚ) / all_outcomes.card = prob_sum_div_3 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_div_3_correct_l3533_353318


namespace NUMINAMATH_CALUDE_red_balloons_count_l3533_353329

def total_balloons : ℕ := 17
def green_balloons : ℕ := 9

theorem red_balloons_count :
  total_balloons - green_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_count_l3533_353329


namespace NUMINAMATH_CALUDE_opposite_to_A_is_F_l3533_353392

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six connected squares --/
structure Cube where
  faces : Fin 6 → Label
  is_valid : ∀ (l : Label), ∃ (i : Fin 6), faces i = l

/-- Defines the opposite face relation on a cube --/
def opposite (c : Cube) (l1 l2 : Label) : Prop :=
  ∃ (i j : Fin 6), c.faces i = l1 ∧ c.faces j = l2 ∧ i ≠ j ∧
    ∀ (k : Fin 6), k ≠ i → k ≠ j → 
      ∃ (m : Fin 6), m ≠ i ∧ m ≠ j ∧ (c.faces k = c.faces m)

/-- Theorem stating that F is opposite to A in the cube --/
theorem opposite_to_A_is_F (c : Cube) : opposite c Label.A Label.F := by
  sorry

end NUMINAMATH_CALUDE_opposite_to_A_is_F_l3533_353392


namespace NUMINAMATH_CALUDE_max_profit_and_volume_l3533_353320

/-- Represents the annual profit function for a company producing a certain product. -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    -(1/360) * x^3 + 30*x - 250
  else
    1200 - (x + 10000/x)

/-- Theorem stating the maximum annual profit and the production volume that achieves it. -/
theorem max_profit_and_volume :
  (∃ (max_profit : ℝ) (optimal_volume : ℝ),
    max_profit = 1000 ∧
    optimal_volume = 100 ∧
    ∀ x, x > 0 → annual_profit x ≤ max_profit ∧
    annual_profit optimal_volume = max_profit) :=
sorry


end NUMINAMATH_CALUDE_max_profit_and_volume_l3533_353320


namespace NUMINAMATH_CALUDE_aaron_can_lids_l3533_353393

/-- The number of can lids Aaron is taking to the recycling center -/
def total_can_lids (num_boxes : ℕ) (existing_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  num_boxes * lids_per_box + existing_lids

/-- Proof that Aaron is taking 53 can lids to the recycling center -/
theorem aaron_can_lids : total_can_lids 3 14 13 = 53 := by
  sorry

end NUMINAMATH_CALUDE_aaron_can_lids_l3533_353393


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3533_353341

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3533_353341


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3533_353352

def cube_numbers (start : ℕ) : List ℕ := List.range 6 |>.map (· + start)

def opposite_faces_sum_equal (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧ 
  ∃ (sum : ℕ), 
    numbers[0]! + numbers[5]! = sum ∧
    numbers[1]! + numbers[4]! = sum ∧
    numbers[2]! + numbers[3]! = sum

theorem cube_sum_theorem :
  let numbers := cube_numbers 15
  opposite_faces_sum_equal numbers →
  numbers.sum = 105 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3533_353352


namespace NUMINAMATH_CALUDE_petyas_class_l3533_353385

theorem petyas_class (x y : ℕ) : 
  (2 * x : ℚ) / 3 + y / 7 = (x + y : ℚ) / 3 →  -- Condition 1, 2, 3
  x + y ≤ 40 →                                -- Condition 4
  x = 12                                      -- Conclusion
  := by sorry

end NUMINAMATH_CALUDE_petyas_class_l3533_353385
