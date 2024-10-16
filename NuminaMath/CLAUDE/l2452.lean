import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l2452_245204

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 3 ∧ m % 11 = 0 → n ≤ m :=
by
  use 33
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l2452_245204


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2452_245234

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 40)
  (h2 : second_catch = 40)
  (h3 : total_fish = 800) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2452_245234


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2452_245200

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_2 + a_7 + a_15 = 12,
    prove that a_8 = 4. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 + a 15 = 12) : a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2452_245200


namespace NUMINAMATH_CALUDE_first_class_students_l2452_245209

/-- Proves that given the conditions of the problem, the number of students in the first class is 12 --/
theorem first_class_students (x : ℕ) : 
  (40 * x + 60 * 28) / (x + 28) = 54 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_first_class_students_l2452_245209


namespace NUMINAMATH_CALUDE_richard_solves_1099_problems_l2452_245277

/-- The number of problems Richard solves in 2013 --/
def problems_solved_2013 : ℕ :=
  let days_in_2013 : ℕ := 365
  let problems_per_week : ℕ := 2 + 1 + 2 + 1 + 2 + 5 + 7
  let full_weeks : ℕ := days_in_2013 / 7
  let extra_day : ℕ := days_in_2013 % 7
  let normal_tuesday_problems : ℕ := 1
  let special_tuesday_problems : ℕ := 60
  full_weeks * problems_per_week + extra_day * normal_tuesday_problems + 
    (special_tuesday_problems - normal_tuesday_problems)

theorem richard_solves_1099_problems : problems_solved_2013 = 1099 := by
  sorry

end NUMINAMATH_CALUDE_richard_solves_1099_problems_l2452_245277


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l2452_245267

/-- The percentage of problems Lucky Lacy got correct on an algebra test -/
theorem lucky_lacy_correct_percentage :
  ∀ x : ℕ,
  x > 0 →
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  let correct_fraction : ℚ := correct_problems / total_problems
  let correct_percentage := correct_fraction * 100
  ∃ ε > 0, abs (correct_percentage - 71.43) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l2452_245267


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2452_245205

theorem concentric_circles_ratio (s S : ℝ) (h : s > 0) (H : S > s) :
  (π * S^2 = 3/2 * (π * S^2 - π * s^2)) → S/s = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2452_245205


namespace NUMINAMATH_CALUDE_box_dimensions_l2452_245281

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17)
  (h2 : a + b = 13)
  (h3 : b + c = 20)
  (h4 : a < b)
  (h5 : b < c) :
  a = 5 ∧ b = 8 ∧ c = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_l2452_245281


namespace NUMINAMATH_CALUDE_ice_cream_sales_l2452_245217

def tuesday_sales : ℕ := 12000

def wednesday_sales : ℕ := 2 * tuesday_sales

def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales : total_sales = 36000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l2452_245217


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2452_245246

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 3*x = 4) ↔ (∀ x : ℝ, x^2 + 3*x ≠ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2452_245246


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2452_245263

theorem sqrt_equation_solution (y : ℚ) :
  (Real.sqrt (4 * y + 3) / Real.sqrt (8 * y + 10) = Real.sqrt 3 / 2) →
  y = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2452_245263


namespace NUMINAMATH_CALUDE_angle_cosine_equality_l2452_245253

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side on the ray 3x + 4y = 0 (x ≤ 0), prove that cos(2α + π/6) = (7√3 + 24) / 50 -/
theorem angle_cosine_equality (α : Real) 
    (h1 : ∃ (x y : Real), x ≤ 0 ∧ 3 * x + 4 * y = 0 ∧ 
          x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
          y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
    Real.cos (2 * α + π / 6) = (7 * Real.sqrt 3 + 24) / 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_equality_l2452_245253


namespace NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l2452_245269

-- Define the geometric bodies
inductive GeometricBody
  | Pyramid
  | Prism
  | Frustum
  | Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaceCount (body : GeometricBody) : Nat :=
  match body with
  | GeometricBody.Pyramid => 0
  | GeometricBody.Prism => 6
  | GeometricBody.Frustum => 2
  | GeometricBody.Cuboid => 6

-- Theorem statement
theorem only_frustum_has_two_parallel_surfaces :
  ∀ (body : GeometricBody),
    parallelSurfaceCount body = 2 ↔ body = GeometricBody.Frustum :=
by
  sorry


end NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l2452_245269


namespace NUMINAMATH_CALUDE_most_suitable_sampling_methods_l2452_245257

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SystematicSampling
  | StratifiedSampling
  | SimpleRandomSampling

/-- Represents a survey scenario --/
structure SurveyScenario where
  description : String
  sampleSize : Nat

/-- Determines the most suitable sampling method for a given scenario --/
def mostSuitableSamplingMethod (scenario : SurveyScenario) : SamplingMethod :=
  sorry

/-- The three survey scenarios described in the problem --/
def scenario1 : SurveyScenario :=
  { description := "First-year high school students' mathematics learning, 2 students from each class",
    sampleSize := 2 }

def scenario2 : SurveyScenario :=
  { description := "Math competition results, 12 students selected from different score ranges",
    sampleSize := 12 }

def scenario3 : SurveyScenario :=
  { description := "Sports meeting, arranging tracks for 6 students in 400m race",
    sampleSize := 6 }

/-- Theorem stating the most suitable sampling methods for the given scenarios --/
theorem most_suitable_sampling_methods :
  (mostSuitableSamplingMethod scenario1 = SamplingMethod.SystematicSampling) ∧
  (mostSuitableSamplingMethod scenario2 = SamplingMethod.StratifiedSampling) ∧
  (mostSuitableSamplingMethod scenario3 = SamplingMethod.SimpleRandomSampling) :=
by sorry

end NUMINAMATH_CALUDE_most_suitable_sampling_methods_l2452_245257


namespace NUMINAMATH_CALUDE_c_share_is_75_l2452_245268

-- Define the total payment
def total_payment : ℚ := 600

-- Define the time taken by each worker individually
def a_time : ℚ := 6
def b_time : ℚ := 8

-- Define the time taken by all three workers together
def abc_time : ℚ := 3

-- Define the shares of A and B
def a_share : ℚ := 300
def b_share : ℚ := 225

-- Define C's share as a function of the given parameters
def c_share (total : ℚ) (a_t b_t abc_t : ℚ) (a_s b_s : ℚ) : ℚ :=
  total - (a_s + b_s)

-- Theorem statement
theorem c_share_is_75 :
  c_share total_payment a_time b_time abc_time a_share b_share = 75 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_75_l2452_245268


namespace NUMINAMATH_CALUDE_math_contest_correct_answers_l2452_245294

theorem math_contest_correct_answers 
  (total_problems : ℕ)
  (correct_points : ℤ)
  (incorrect_points : ℤ)
  (total_score : ℤ)
  (min_guesses : ℕ)
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45)
  (h5 : min_guesses ≥ 4)
  : ∃ (correct_answers : ℕ), 
    correct_answers * correct_points + (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_contest_correct_answers_l2452_245294


namespace NUMINAMATH_CALUDE_estimate_viewers_l2452_245260

theorem estimate_viewers (total_population : ℕ) (sample_size : ℕ) (sample_viewers : ℕ) 
  (h1 : total_population = 3600)
  (h2 : sample_size = 200)
  (h3 : sample_viewers = 160) :
  (total_population : ℚ) * (sample_viewers : ℚ) / (sample_size : ℚ) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_estimate_viewers_l2452_245260


namespace NUMINAMATH_CALUDE_volleyball_tournament_wins_l2452_245238

theorem volleyball_tournament_wins (n m : ℕ) : 
  ∀ (x : ℕ), 0 < x ∧ x < 73 →
  x * n + (73 - x) * m = 36 * 73 →
  n = m := by
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_wins_l2452_245238


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2452_245232

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d) ∧
  a 4 = -8 ∧
  a 8 = 2

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) (h : arithmetic_sequence a) :
  a 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2452_245232


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l2452_245274

/-- Conversion from rectangular to cylindrical coordinates --/
theorem rect_to_cylindrical :
  let x : ℝ := -4
  let y : ℝ := -4 * Real.sqrt 3
  let z : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 4 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) →
  (x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ z = z) :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l2452_245274


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2452_245270

theorem inequality_solution_set : 
  ∀ x : ℝ, (3/20 : ℝ) + |x - 7/40| < (11/40 : ℝ) ↔ x ∈ Set.Ioo (1/20 : ℝ) (3/10 : ℝ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2452_245270


namespace NUMINAMATH_CALUDE_product_sum_fractions_l2452_245256

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 - 1/5) = 23 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l2452_245256


namespace NUMINAMATH_CALUDE_tan_periodic_angle_l2452_245271

theorem tan_periodic_angle (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1720 * π / 180) → n = -80 :=
by sorry

end NUMINAMATH_CALUDE_tan_periodic_angle_l2452_245271


namespace NUMINAMATH_CALUDE_harolds_utilities_car_ratio_l2452_245261

/-- Harold's financial situation --/
structure HaroldFinances where
  income : ℕ
  rent : ℕ
  car_payment : ℕ
  groceries : ℕ
  retirement_savings : ℕ
  remaining : ℕ

/-- Calculate the ratio of utilities cost to car payment --/
def utilities_to_car_ratio (h : HaroldFinances) : ℚ :=
  let total_expenses := h.rent + h.car_payment + h.groceries
  let money_before_retirement := h.income - total_expenses
  let utilities := money_before_retirement - h.retirement_savings - h.remaining
  utilities / h.car_payment

/-- Theorem stating the ratio of Harold's utilities cost to his car payment --/
theorem harolds_utilities_car_ratio :
  ∃ h : HaroldFinances,
    h.income = 2500 ∧
    h.rent = 700 ∧
    h.car_payment = 300 ∧
    h.groceries = 50 ∧
    h.retirement_savings = (h.income - h.rent - h.car_payment - h.groceries) / 2 ∧
    h.remaining = 650 ∧
    utilities_to_car_ratio h = 1 / 4 :=
  sorry

end NUMINAMATH_CALUDE_harolds_utilities_car_ratio_l2452_245261


namespace NUMINAMATH_CALUDE_curve_C_parametric_equations_l2452_245276

/-- Given a curve C with polar equation ρ = 2cosθ, prove that its parametric equations are x = 1 + cosθ and y = sinθ -/
theorem curve_C_parametric_equations (θ : ℝ) :
  let ρ := 2 * Real.cos θ
  let x := 1 + Real.cos θ
  let y := Real.sin θ
  (x, y) ∈ {(x, y) : ℝ × ℝ | x^2 + y^2 = ρ^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ} :=
by sorry

end NUMINAMATH_CALUDE_curve_C_parametric_equations_l2452_245276


namespace NUMINAMATH_CALUDE_first_game_score_l2452_245288

def basketball_scores : List ℕ := [68, 70, 61, 74, 62, 65, 74]

theorem first_game_score (mean : ℚ) (h1 : mean = 67.9) :
  ∃ x : ℕ, (x :: basketball_scores).length = 8 ∧ 
  (((x :: basketball_scores).sum : ℚ) / 8 = mean) ∧
  x = 69 := by
  sorry

end NUMINAMATH_CALUDE_first_game_score_l2452_245288


namespace NUMINAMATH_CALUDE_ramanujan_number_l2452_245291

theorem ramanujan_number (r h : ℂ) : 
  r * h = 50 - 14 * I →
  h = 7 + 2 * I →
  r = 6 - (198 / 53) * I := by
sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2452_245291


namespace NUMINAMATH_CALUDE_average_playing_time_is_ten_l2452_245236

/-- Represents the number of players --/
def num_players : ℕ := 8

/-- Represents the start time in hours since midnight --/
def start_time : ℕ := 8

/-- Represents the end time in hours since midnight --/
def end_time : ℕ := 18

/-- Represents the number of chess games being played simultaneously --/
def num_games : ℕ := 2

/-- Calculates the average playing time per person --/
def average_playing_time : ℚ :=
  (end_time - start_time : ℚ) * num_games / num_players

theorem average_playing_time_is_ten :
  average_playing_time = 10 := by sorry

end NUMINAMATH_CALUDE_average_playing_time_is_ten_l2452_245236


namespace NUMINAMATH_CALUDE_crackers_per_person_l2452_245254

theorem crackers_per_person 
  (num_friends : ℕ)
  (cracker_ratio cake_ratio : ℕ)
  (initial_crackers initial_cakes : ℕ)
  (h1 : num_friends = 6)
  (h2 : cracker_ratio = 3)
  (h3 : cake_ratio = 5)
  (h4 : initial_crackers = 72)
  (h5 : initial_cakes = 180) :
  initial_crackers / (cracker_ratio * num_friends) = 12 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_person_l2452_245254


namespace NUMINAMATH_CALUDE_light_bulb_probability_l2452_245293

theorem light_bulb_probability (qualification_rate : ℝ) 
  (h1 : qualification_rate = 0.99) : 
  ℝ :=
by
  -- The probability of selecting a qualified light bulb
  -- is equal to the qualification rate
  sorry

#check light_bulb_probability

end NUMINAMATH_CALUDE_light_bulb_probability_l2452_245293


namespace NUMINAMATH_CALUDE_profit_decrease_calculation_l2452_245229

theorem profit_decrease_calculation (march_profit : ℝ) (april_may_decrease : ℝ) :
  march_profit > 0 →
  (march_profit * 1.3 * (1 - april_may_decrease / 100) * 1.5 = march_profit * 1.5600000000000001) →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_decrease_calculation_l2452_245229


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2452_245216

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2452_245216


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2452_245249

theorem intersection_of_three_lines (k : ℚ) : 
  (∃ (x y : ℚ), y = 6*x + 5 ∧ y = -3*x - 30 ∧ y = 4*x + k) → 
  k = -25/9 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2452_245249


namespace NUMINAMATH_CALUDE_area_of_special_quadrilateral_l2452_245266

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a b c : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral 
  (mainTriangle : Triangle) 
  (smallTriangles : Fin 4 → Triangle)
  (quadrilaterals : Fin 3 → Quadrilateral)
  (h1 : ∀ i, triangleArea (smallTriangles i) = 1)
  (h2 : quadrilateralArea (quadrilaterals 0) = quadrilateralArea (quadrilaterals 1))
  (h3 : quadrilateralArea (quadrilaterals 1) = quadrilateralArea (quadrilaterals 2)) :
  quadrilateralArea (quadrilaterals 0) = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_quadrilateral_l2452_245266


namespace NUMINAMATH_CALUDE_cloth_cost_price_l2452_245221

/-- Given a trader selling cloth, calculates the cost price per meter. -/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Proves that the cost price of one metre of cloth is 128 rupees. -/
theorem cloth_cost_price :
  cost_price_per_meter 60 8400 12 = 128 := by
  sorry

#eval cost_price_per_meter 60 8400 12

end NUMINAMATH_CALUDE_cloth_cost_price_l2452_245221


namespace NUMINAMATH_CALUDE_retirement_percentage_l2452_245231

def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_pay : ℝ := 740

theorem retirement_percentage :
  (gross_pay - net_pay - tax_deduction) / gross_pay * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_retirement_percentage_l2452_245231


namespace NUMINAMATH_CALUDE_third_grade_swim_caps_l2452_245243

theorem third_grade_swim_caps (b r : ℕ) : 
  b = 4 * r + 2 →
  b = r + 24 →
  b + r = 37 :=
by sorry

end NUMINAMATH_CALUDE_third_grade_swim_caps_l2452_245243


namespace NUMINAMATH_CALUDE_total_count_is_900_l2452_245283

/-- Represents the count of type A components -/
def a : ℕ := 400

/-- Represents the count of type B components -/
def b : ℕ := 300

/-- Represents the count of type C components -/
def c : ℕ := 200

/-- Represents the total sample size -/
def sample_size : ℕ := 45

/-- Represents the number of type C components sampled -/
def c_sampled : ℕ := 10

/-- Represents the total count of all components -/
def total_count : ℕ := a + b + c

/-- Theorem stating that the total count of all components is 900 -/
theorem total_count_is_900 : total_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_count_is_900_l2452_245283


namespace NUMINAMATH_CALUDE_completing_square_sum_l2452_245292

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x, 36 * x^2 - 60 * x + 25 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 26 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2452_245292


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_find_m_for_given_intersection_l2452_245287

-- Define set A
def A : Set ℝ := {x | |x - 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Theorem 2
theorem find_m_for_given_intersection :
  A ∩ B 8 = {x | -1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_find_m_for_given_intersection_l2452_245287


namespace NUMINAMATH_CALUDE_prob_third_batch_value_l2452_245225

/-- Represents a batch of parts -/
structure Batch :=
  (total : ℕ)
  (standard : ℕ)
  (h : standard ≤ total)

/-- Represents the experiment of selecting two standard parts from a batch -/
def select_two_standard (b : Batch) : ℚ :=
  (b.standard : ℚ) / b.total * ((b.standard - 1) : ℚ) / (b.total - 1)

/-- The probability of selecting the third batch given that two standard parts were selected -/
def prob_third_batch (b1 b2 b3 : Batch) : ℚ :=
  let p1 := select_two_standard b1
  let p2 := select_two_standard b2
  let p3 := select_two_standard b3
  p3 / (p1 + p2 + p3)

theorem prob_third_batch_value :
  let b1 : Batch := ⟨30, 20, by norm_num⟩
  let b2 : Batch := ⟨30, 15, by norm_num⟩
  let b3 : Batch := ⟨30, 10, by norm_num⟩
  prob_third_batch b1 b2 b3 = 3 / 68 := by
  sorry

end NUMINAMATH_CALUDE_prob_third_batch_value_l2452_245225


namespace NUMINAMATH_CALUDE_square_difference_l2452_245272

theorem square_difference (a b : ℝ) 
  (h1 : a^2 - a*b = 10) 
  (h2 : a*b - b^2 = -15) : 
  a^2 - b^2 = -5 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2452_245272


namespace NUMINAMATH_CALUDE_knit_socks_together_l2452_245214

/-- The number of days it takes for two people to knit a certain number of socks together -/
def days_to_knit (a_rate b_rate : ℚ) (pairs : ℚ) : ℚ :=
  pairs / (a_rate + b_rate)

/-- Theorem: Given the rates at which A and B can knit socks individually, 
    prove that they can knit two pairs of socks in 4 days when working together -/
theorem knit_socks_together 
  (a_rate : ℚ) (b_rate : ℚ) 
  (ha : a_rate = 1 / 3) 
  (hb : b_rate = 1 / 6) : 
  days_to_knit a_rate b_rate 2 = 4 := by
  sorry

#eval days_to_knit (1/3) (1/6) 2

end NUMINAMATH_CALUDE_knit_socks_together_l2452_245214


namespace NUMINAMATH_CALUDE_toms_living_room_length_l2452_245278

def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250
def boxes_needed : ℕ := 7

theorem toms_living_room_length : 
  (flooring_laid + boxes_needed * flooring_per_box) / room_width = 16 := by
  sorry

end NUMINAMATH_CALUDE_toms_living_room_length_l2452_245278


namespace NUMINAMATH_CALUDE_cube_root_three_identity_l2452_245280

theorem cube_root_three_identity (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_three_identity_l2452_245280


namespace NUMINAMATH_CALUDE_max_true_statements_l2452_245210

theorem max_true_statements (a b : ℝ) : 
  let statements := [
    (1/a > 1/b),
    (a^2 < b^2),
    (a > b),
    (a > 0),
    (b > 0)
  ]
  ∃ (trueStatements : List Bool), 
    trueStatements.length ≤ 3 ∧ 
    ∀ (i : Nat), i < statements.length → 
      (trueStatements.get? i = some true → statements.get! i) ∧
      (statements.get! i → trueStatements.get? i = some true) :=
sorry

end NUMINAMATH_CALUDE_max_true_statements_l2452_245210


namespace NUMINAMATH_CALUDE_equation_roots_l2452_245282

theorem equation_roots (m : ℝ) : 
  (∃! x : ℝ, (m - 2) * x^2 - 2 * (m - 1) * x + m = 0) → 
  (∃ x : ℝ, ∀ y : ℝ, m * y^2 - (m + 2) * y + (4 - m) = 0 ↔ y = x) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l2452_245282


namespace NUMINAMATH_CALUDE_exactly_one_false_proposition_l2452_245285

theorem exactly_one_false_proposition :
  let prop1 := (∀ x : ℝ, (x ^ 2 - 3 * x + 2 ≠ 0) → (x ≠ 1)) ↔ (∀ x : ℝ, (x ≠ 1) → (x ^ 2 - 3 * x + 2 ≠ 0))
  let prop2 := (∀ x : ℝ, x > 2 → x ^ 2 - 3 * x + 2 > 0) ∧ (∃ x : ℝ, x ≤ 2 ∧ x ^ 2 - 3 * x + 2 > 0)
  let prop3 := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)
  let prop4 := (∃ x : ℝ, x ^ 2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x ^ 2 + x + 1 ≥ 0)
  ∃! i : Fin 4, ¬(match i with
    | 0 => prop1
    | 1 => prop2
    | 2 => prop3
    | 3 => prop4) :=
by
  sorry

end NUMINAMATH_CALUDE_exactly_one_false_proposition_l2452_245285


namespace NUMINAMATH_CALUDE_razorback_shop_profit_l2452_245265

theorem razorback_shop_profit : 
  let tshirt_profit : ℕ := 67
  let jersey_profit : ℕ := 165
  let hat_profit : ℕ := 32
  let jacket_profit : ℕ := 245
  let tshirts_sold : ℕ := 74
  let jerseys_sold : ℕ := 156
  let hats_sold : ℕ := 215
  let jackets_sold : ℕ := 45
  (tshirt_profit * tshirts_sold + 
   jersey_profit * jerseys_sold + 
   hat_profit * hats_sold + 
   jacket_profit * jackets_sold) = 48603 :=
by sorry

end NUMINAMATH_CALUDE_razorback_shop_profit_l2452_245265


namespace NUMINAMATH_CALUDE_largest_divisor_with_equal_quotient_remainder_l2452_245241

theorem largest_divisor_with_equal_quotient_remainder :
  ∀ (A B C : ℕ),
    (10 = A * B + C) →
    (B = C) →
    A ≤ 9 ∧
    (∃ (A' : ℕ), A' = 9 ∧ ∃ (B' C' : ℕ), 10 = A' * B' + C' ∧ B' = C') :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_with_equal_quotient_remainder_l2452_245241


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2452_245222

theorem min_reciprocal_sum : ∀ a b : ℕ+, 
  4 * a + b = 6 → 
  (1 : ℝ) / 1 + (1 : ℝ) / 2 ≤ (1 : ℝ) / a + (1 : ℝ) / b :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2452_245222


namespace NUMINAMATH_CALUDE_parabola_sum_l2452_245245

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, -2), and passing through (0, 5) -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ
  vertex_x : ℚ := 3
  vertex_y : ℚ := -2
  point_x : ℚ := 0
  point_y : ℚ := 5
  eq_at_vertex : -2 = a * 3^2 + b * 3 + c
  eq_at_point : 5 = c
  vertex_formula : b = -2 * a * vertex_x

theorem parabola_sum (p : Parabola) : p.a + p.b + p.c = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l2452_245245


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2452_245224

theorem inequality_equivalence (p : ℝ) (hp : p > 0) : 
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → (1 / Real.sin x ^ 2) + (p / Real.cos x ^ 2) ≥ 9) ↔ 
  p ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2452_245224


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l2452_245219

theorem complex_number_subtraction (z : ℂ) (a b : ℝ) : 
  z = 2 + I → Complex.re z = a → Complex.im z = b → a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l2452_245219


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2452_245264

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ) = (a^2 + 2*a - 3 : ℝ) + Complex.I * (a - 1) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2452_245264


namespace NUMINAMATH_CALUDE_solve_equation_l2452_245202

theorem solve_equation : ∃ x : ℝ, 4*x + 9*x = 360 - 9*(x - 4) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2452_245202


namespace NUMINAMATH_CALUDE_income_calculation_l2452_245286

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 7 * expenditure / 6 →
  savings = income - expenditure →
  savings = 2000 →
  income = 14000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2452_245286


namespace NUMINAMATH_CALUDE_intersection_probability_l2452_245297

def U : Finset Nat := {1, 2, 3, 4, 5}
def I : Finset (Finset Nat) := Finset.powerset U

def favorable_pairs : Nat :=
  (Finset.powerset U).card * (Finset.filter (fun s => s.card = 3) (Finset.powerset U)).card

def total_pairs : Nat := (I.card.choose 2)

theorem intersection_probability :
  (favorable_pairs : ℚ) / total_pairs = 5 / 62 := by sorry

end NUMINAMATH_CALUDE_intersection_probability_l2452_245297


namespace NUMINAMATH_CALUDE_sam_long_sleeve_shirts_l2452_245206

/-- Given information about Sam's shirts to wash -/
structure ShirtWashing where
  short_sleeve : ℕ
  washed : ℕ
  unwashed : ℕ

/-- The number of long sleeve shirts Sam had to wash -/
def long_sleeve_shirts (s : ShirtWashing) : ℕ :=
  s.washed + s.unwashed - s.short_sleeve

/-- Theorem stating the number of long sleeve shirts Sam had to wash -/
theorem sam_long_sleeve_shirts :
  ∀ s : ShirtWashing,
  s.short_sleeve = 40 →
  s.washed = 29 →
  s.unwashed = 34 →
  long_sleeve_shirts s = 23 := by
  sorry

end NUMINAMATH_CALUDE_sam_long_sleeve_shirts_l2452_245206


namespace NUMINAMATH_CALUDE_johns_family_members_l2452_245252

/-- The number of family members on John's father's side -/
def fathers_side : ℕ := sorry

/-- The total number of family members -/
def total_members : ℕ := 23

/-- The ratio of mother's side to father's side -/
def mother_ratio : ℚ := 13/10

theorem johns_family_members :
  fathers_side = 10 ∧
  (fathers_side : ℚ) * (1 + mother_ratio - 1) + fathers_side = total_members :=
sorry

end NUMINAMATH_CALUDE_johns_family_members_l2452_245252


namespace NUMINAMATH_CALUDE_trigonometric_equation_implications_l2452_245258

theorem trigonometric_equation_implications (x : Real) 
  (h : (Real.sin (Real.pi + x) + 2 * Real.cos (3 * Real.pi / 2 + x)) / 
       (Real.cos (Real.pi - x) - Real.sin (Real.pi / 2 - x)) = 1) : 
  Real.tan x = 2/3 ∧ Real.sin (2*x) - Real.cos x ^ 2 = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_implications_l2452_245258


namespace NUMINAMATH_CALUDE_largest_class_size_l2452_245223

theorem largest_class_size (n : ℕ) (total : ℕ) : 
  n = 5 → 
  total = 115 → 
  ∃ x : ℕ, x > 0 ∧ 
    (x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total) ∧ 
    x = 27 := by
  sorry

end NUMINAMATH_CALUDE_largest_class_size_l2452_245223


namespace NUMINAMATH_CALUDE_candy_problem_smallest_n_l2452_245242

theorem candy_problem (x y z n : ℕ+) : 
  (18 * x = 21 * y) ∧ (21 * y = 10 * z) ∧ (10 * z = 30 * n) → n ≥ 21 := by
  sorry

theorem smallest_n : ∃ (x y z : ℕ+), 18 * x = 21 * y ∧ 21 * y = 10 * z ∧ 10 * z = 30 * 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_smallest_n_l2452_245242


namespace NUMINAMATH_CALUDE_new_person_weight_new_person_weight_is_87_l2452_245255

/-- The weight of a new person joining a group, given specific conditions -/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) : ℝ :=
  let total_increase := initial_count * avg_increase
  leaving_weight + total_increase

/-- Proof that the new person's weight is 87 kg under given conditions -/
theorem new_person_weight_is_87 :
  new_person_weight 8 67 2.5 = 87 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_new_person_weight_is_87_l2452_245255


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2452_245239

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2452_245239


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2452_245275

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2452_245275


namespace NUMINAMATH_CALUDE_sales_increase_percentage_l2452_245298

theorem sales_increase_percentage (price_reduction : ℝ) (receipts_increase : ℝ) : 
  price_reduction = 30 → receipts_increase = 5 → 
  ∃ (sales_increase : ℝ), sales_increase = 50 ∧ 
  (100 - price_reduction) / 100 * (1 + sales_increase / 100) = 1 + receipts_increase / 100 :=
by sorry

end NUMINAMATH_CALUDE_sales_increase_percentage_l2452_245298


namespace NUMINAMATH_CALUDE_floor_with_57_diagonal_tiles_has_841_total_tiles_l2452_245299

/-- Represents a rectangular floor covered with square tiles -/
structure TiledFloor where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles on a rectangular floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.length * floor.width

/-- Theorem stating that a rectangular floor with 57 tiles on its diagonals has 841 tiles in total -/
theorem floor_with_57_diagonal_tiles_has_841_total_tiles :
  ∃ (floor : TiledFloor), floor.diagonal_tiles = 57 ∧ total_tiles floor = 841 := by
  sorry


end NUMINAMATH_CALUDE_floor_with_57_diagonal_tiles_has_841_total_tiles_l2452_245299


namespace NUMINAMATH_CALUDE_weight_of_three_liters_l2452_245218

/-- Given that 1 liter weighs 2.2 pounds, prove that 3 liters weigh 6.6 pounds. -/
theorem weight_of_three_liters (weight_per_liter : ℝ) (h : weight_per_liter = 2.2) :
  3 * weight_per_liter = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_three_liters_l2452_245218


namespace NUMINAMATH_CALUDE_square_of_negative_two_x_squared_l2452_245295

theorem square_of_negative_two_x_squared (x : ℝ) : (-2 * x^2)^2 = 4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_x_squared_l2452_245295


namespace NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l2452_245201

def purchase1 : ℚ := 1.98
def purchase2 : ℚ := 5.04
def purchase3 : ℚ := 9.89

def roundToNearestInteger (x : ℚ) : ℤ :=
  if x - ↑(Int.floor x) < 1/2 then Int.floor x else Int.ceil x

theorem total_rounded_to_nearest_dollar :
  roundToNearestInteger (purchase1 + purchase2 + purchase3) = 17 := by sorry

end NUMINAMATH_CALUDE_total_rounded_to_nearest_dollar_l2452_245201


namespace NUMINAMATH_CALUDE_kiley_crayons_l2452_245273

theorem kiley_crayons (initial_crayons : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 48)
  (h2 : remaining_crayons = 18)
  (f : ℚ) -- fraction of crayons Kiley took
  (h3 : 0 ≤ f ∧ f < 1)
  (h4 : remaining_crayons = (initial_crayons : ℚ) * (1 - f) / 2) :
  f = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_kiley_crayons_l2452_245273


namespace NUMINAMATH_CALUDE_team_selection_count_l2452_245284

def total_athletes : ℕ := 10
def veteran_players : ℕ := 2
def new_players : ℕ := 8
def team_size : ℕ := 3
def excluded_new_player : ℕ := 1

theorem team_selection_count :
  (Nat.choose veteran_players 1 * Nat.choose (new_players - excluded_new_player) 2) +
  (Nat.choose (new_players - excluded_new_player) team_size) = 77 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2452_245284


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2452_245227

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2452_245227


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_digits_l2452_245296

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 3 ∨ d = 6

def contains_both (n : ℕ) : Prop :=
  3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_valid_number_last_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 3 = 0 ∧
    m % 6 = 0 ∧
    is_valid_number m ∧
    contains_both m ∧
    (∀ k : ℕ, k > 0 ∧ k % 3 = 0 ∧ k % 6 = 0 ∧ is_valid_number k ∧ contains_both k → m ≤ k) ∧
    last_four_digits m = 3630 :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_digits_l2452_245296


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2452_245220

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| ≤ 4
def q (x : ℝ) : Prop := x^2 < 5*x - 6

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2452_245220


namespace NUMINAMATH_CALUDE_disk_color_difference_l2452_245207

theorem disk_color_difference (total : ℕ) (blue_ratio yellow_ratio green_ratio : ℕ) : 
  total = 126 →
  blue_ratio = 3 →
  yellow_ratio = 7 →
  green_ratio = 8 →
  let ratio_sum := blue_ratio + yellow_ratio + green_ratio
  let blue_count := (blue_ratio * total) / ratio_sum
  let green_count := (green_ratio * total) / ratio_sum
  green_count - blue_count = 35 := by
sorry

end NUMINAMATH_CALUDE_disk_color_difference_l2452_245207


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2452_245244

/-- Proves that the cost price of a bicycle for seller A is 150 given the selling conditions --/
theorem bicycle_cost_price
  (profit_A_to_B : ℝ) -- Profit percentage when A sells to B
  (profit_B_to_C : ℝ) -- Profit percentage when B sells to C
  (price_C : ℝ)       -- Price C pays for the bicycle
  (h1 : profit_A_to_B = 20)
  (h2 : profit_B_to_C = 25)
  (h3 : price_C = 225) :
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B / 100) * (1 + profit_B_to_C / 100) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2452_245244


namespace NUMINAMATH_CALUDE_angle_sum_is_420_l2452_245248

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfiguration where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- The theorem stating that if E = 30°, then the sum of all angles is 420° -/
theorem angle_sum_is_420 (config : GeometricConfiguration) 
  (h_E : config.E = 30) : 
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end NUMINAMATH_CALUDE_angle_sum_is_420_l2452_245248


namespace NUMINAMATH_CALUDE_sequence_length_l2452_245237

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem sequence_length : 
  ∃ n : ℕ, n > 0 ∧ 
  arithmetic_sequence 2.5 5 n = 62.5 ∧ 
  ∀ k : ℕ, k > n → arithmetic_sequence 2.5 5 k > 62.5 ∧
  n = 13 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_l2452_245237


namespace NUMINAMATH_CALUDE_journey_distance_l2452_245212

/-- Proves that the total distance of a journey is 70 km given specific travel conditions. -/
theorem journey_distance (v1 v2 : ℝ) (t_late : ℝ) : 
  v1 = 40 →  -- Average speed for on-time arrival (km/h)
  v2 = 35 →  -- Average speed for late arrival (km/h)
  t_late = 0.25 →  -- Time of late arrival (hours)
  ∃ (d t : ℝ), 
    d = v1 * t ∧  -- Distance equation for on-time arrival
    d = v2 * (t + t_late) ∧  -- Distance equation for late arrival
    d = 70  -- Total distance of the journey (km)
  := by sorry

end NUMINAMATH_CALUDE_journey_distance_l2452_245212


namespace NUMINAMATH_CALUDE_bookstore_inventory_theorem_l2452_245250

theorem bookstore_inventory_theorem (historical_fiction : ℝ) (mystery : ℝ) (science_fiction : ℝ) (romance : ℝ)
  (historical_fiction_new : ℝ) (mystery_new : ℝ) (science_fiction_new : ℝ) (romance_new : ℝ)
  (h1 : historical_fiction = 0.4)
  (h2 : mystery = 0.3)
  (h3 : science_fiction = 0.2)
  (h4 : romance = 0.1)
  (h5 : historical_fiction_new = 0.35 * historical_fiction)
  (h6 : mystery_new = 0.6 * mystery)
  (h7 : science_fiction_new = 0.45 * science_fiction)
  (h8 : romance_new = 0.8 * romance) :
  historical_fiction_new / (historical_fiction_new + mystery_new + science_fiction_new + romance_new) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_inventory_theorem_l2452_245250


namespace NUMINAMATH_CALUDE_statistics_properties_l2452_245251

def data : List ℝ := [2, 3, 6, 9, 3, 7]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem statistics_properties :
  mode data = 3 ∧
  median data = 4.5 ∧
  mean data = 5 ∧
  range data = 7 := by sorry

end NUMINAMATH_CALUDE_statistics_properties_l2452_245251


namespace NUMINAMATH_CALUDE_sum_of_four_variables_l2452_245203

theorem sum_of_four_variables (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 250)
  (h2 : a*b + b*c + c*a + a*d + b*d + c*d = 3) : 
  a + b + c + d = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_variables_l2452_245203


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2452_245211

theorem quadratic_equation_coefficients (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -7) →
  (∀ x : ℝ, |x + 3| = 4 ↔ x = 1 ∨ x = -7) →
  b = 6 ∧ c = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2452_245211


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_increasing_sequence_set_l2452_245262

def sequence_sum (a : ℝ) (n : ℕ) : ℝ := sorry

def sequence_term (a : ℝ) (n : ℕ) : ℝ := sorry

axiom sequence_sum_property (a : ℝ) (n : ℕ) :
  n ≥ 2 → (sequence_sum a n)^2 = 3 * n^2 * (sequence_term a n) + (sequence_sum a (n-1))^2

axiom nonzero_terms (a : ℝ) (n : ℕ) : sequence_term a n ≠ 0

def is_arithmetic_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → sequence_term a (n+1) - sequence_term a n = sequence_term a n - sequence_term a (n-1)

def is_increasing_sequence (a : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → sequence_term a n < sequence_term a (n+1)

theorem arithmetic_sequence_value (a : ℝ) :
  is_arithmetic_sequence a → a = 3 := sorry

theorem increasing_sequence_set :
  {a : ℝ | is_increasing_sequence a} = Set.Ioo (9/4) (15/4) := sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_increasing_sequence_set_l2452_245262


namespace NUMINAMATH_CALUDE_total_cookies_l2452_245213

/-- Given 272 bags of cookies with 45 cookies in each bag, 
    prove that the total number of cookies is 12240 -/
theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 272) (h2 : cookies_per_bag = 45) : 
  bags * cookies_per_bag = 12240 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l2452_245213


namespace NUMINAMATH_CALUDE_square_circle_union_area_l2452_245226

/-- The area of the union of a square and a circle with specific dimensions -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l2452_245226


namespace NUMINAMATH_CALUDE_unique_solution_for_quadratic_equation_l2452_245235

/-- Given an equation (x+m)^2 - (x^2+n^2) = (m-n)^2 where m and n are unequal non-zero constants,
    prove that the unique solution for x in the form x = am + bn has a = 0 and b = -m + n. -/
theorem unique_solution_for_quadratic_equation 
  (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ (x : ℝ), 
    (x + m)^2 - (x^2 + n^2) = (m - n)^2 ↔ x = a*m + b*n ∧ a = 0 ∧ b = -m + n := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_quadratic_equation_l2452_245235


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l2452_245215

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l2452_245215


namespace NUMINAMATH_CALUDE_martha_cakes_per_child_l2452_245233

/-- Given that Martha has 3.0 children and needs to buy 54 cakes in total,
    prove that each child will get 18 cakes. -/
theorem martha_cakes_per_child :
  let num_children : ℝ := 3.0
  let total_cakes : ℕ := 54
  (total_cakes : ℝ) / num_children = 18 :=
by sorry

end NUMINAMATH_CALUDE_martha_cakes_per_child_l2452_245233


namespace NUMINAMATH_CALUDE_no_solutions_for_inequality_system_l2452_245240

theorem no_solutions_for_inequality_system :
  ¬ ∃ (x y : ℝ), (11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3) ∧ (5 * x + y ≤ -10) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_inequality_system_l2452_245240


namespace NUMINAMATH_CALUDE_exists_composite_power_sum_l2452_245230

theorem exists_composite_power_sum (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, k > 1 ∧ k ∣ (x^(2^n) + y^(2^n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_composite_power_sum_l2452_245230


namespace NUMINAMATH_CALUDE_circle_equation_l2452_245289

-- Define the line L
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ ∃ t : ℝ, p.2 = 1 + t}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 3 = 0}

-- Define the center of the circle
def center : ℝ × ℝ := ((-1 : ℝ), (0 : ℝ))

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 2}

theorem circle_equation (h1 : center ∈ L ∩ x_axis) 
  (h2 : ∀ p ∈ C, p ∈ tangent_line → (∃ q ∈ C, q ≠ p ∧ Set.Subset C {r | r = p ∨ r = q})) :
  C = {p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 2} := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l2452_245289


namespace NUMINAMATH_CALUDE_melody_reading_pages_l2452_245208

theorem melody_reading_pages (science civics chinese : ℕ) (total_tomorrow : ℕ) (english : ℕ) : 
  science = 16 → 
  civics = 8 → 
  chinese = 12 → 
  total_tomorrow = 14 → 
  (english / 4 + science / 4 + civics / 4 + chinese / 4 : ℚ) = total_tomorrow → 
  english = 20 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_pages_l2452_245208


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l2452_245259

theorem fruit_basket_problem (total_fruits : ℕ) 
  (mangoes pears pawpaws kiwis lemons : ℕ) : 
  total_fruits = 58 →
  mangoes = 18 →
  pears = 10 →
  pawpaws = 12 →
  kiwis = lemons →
  total_fruits = mangoes + pears + pawpaws + kiwis + lemons →
  lemons = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l2452_245259


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l2452_245279

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then |x| - 2 else 2*x - 6 + Real.log x

theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
by sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l2452_245279


namespace NUMINAMATH_CALUDE_negative_one_point_five_less_than_negative_one_and_one_fifth_l2452_245290

theorem negative_one_point_five_less_than_negative_one_and_one_fifth : -1.5 < -(1 + 1/5) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_point_five_less_than_negative_one_and_one_fifth_l2452_245290


namespace NUMINAMATH_CALUDE_range_of_n_over_m_l2452_245228

def A (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n}
def B (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m}

theorem range_of_n_over_m (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (hA : Set.Nonempty (A m n)) (hB : Set.Nonempty (B m n)) : 
  n / m ≤ -2 ∧ ∀ k : ℝ, ∃ m n : ℝ, m ≠ 0 ∧ n ≠ 0 ∧ Set.Nonempty (A m n) ∧ Set.Nonempty (B m n) ∧ n / m < k :=
sorry

end NUMINAMATH_CALUDE_range_of_n_over_m_l2452_245228


namespace NUMINAMATH_CALUDE_no_blonde_girls_added_l2452_245247

/-- The number of blonde girls added to a choir -/
def blonde_girls_added (initial_total : ℕ) (initial_blonde : ℕ) (black_haired : ℕ) : ℕ :=
  initial_total - initial_blonde - black_haired

/-- Theorem: Given the initial conditions, no blonde girls were added to the choir -/
theorem no_blonde_girls_added :
  blonde_girls_added 80 30 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_blonde_girls_added_l2452_245247
