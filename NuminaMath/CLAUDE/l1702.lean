import Mathlib

namespace NUMINAMATH_CALUDE_no_m_exists_for_equal_sets_l1702_170284

theorem no_m_exists_for_equal_sets : ¬∃ m : ℝ, 
  {x : ℝ | x^2 - 8*x - 20 ≤ 0} = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equal_sets_l1702_170284


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_l1702_170210

/-- The tangent line to a cubic curve at a specific point -/
theorem tangent_line_to_cubic (a k b : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ 
    y = x^3 + a*x + 1 ∧ 
    y = k*x + b ∧ 
    (3 * x^2 + a) = k) →
  b = -15 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_l1702_170210


namespace NUMINAMATH_CALUDE_difference_of_squares_65_55_l1702_170248

theorem difference_of_squares_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_55_l1702_170248


namespace NUMINAMATH_CALUDE_chris_win_probability_l1702_170276

theorem chris_win_probability :
  let chris_head_prob : ℝ := 1/4
  let drew_head_prob : ℝ := 1/3
  let both_tail_prob : ℝ := (1 - chris_head_prob) * (1 - drew_head_prob)
  chris_head_prob / (1 - both_tail_prob) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chris_win_probability_l1702_170276


namespace NUMINAMATH_CALUDE_f_2009_equals_3_l1702_170280

/-- Given a function f and constants a, b, α, β, prove that f(2009) = 3 -/
theorem f_2009_equals_3 
  (f : ℝ → ℝ) 
  (a b α β : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4)
  (h2 : f 2000 = 5) :
  f 2009 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_equals_3_l1702_170280


namespace NUMINAMATH_CALUDE_lilys_calculation_l1702_170253

theorem lilys_calculation (a b c : ℝ) 
  (h1 : a - 2 * (b - 3 * c) = 14) 
  (h2 : a - 2 * b - 3 * c = 2) : 
  a - 2 * b = 6 := by sorry

end NUMINAMATH_CALUDE_lilys_calculation_l1702_170253


namespace NUMINAMATH_CALUDE_probability_two_nondefective_pens_l1702_170212

/-- Given a box of 8 pens with 3 defective pens, the probability of selecting 2 non-defective pens without replacement is 5/14. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat) 
  (h1 : total_pens = 8) (h2 : defective_pens = 3) :
  let nondefective_pens := total_pens - defective_pens
  let prob_first := nondefective_pens / total_pens
  let prob_second := (nondefective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_nondefective_pens_l1702_170212


namespace NUMINAMATH_CALUDE_min_value_theorem_l1702_170252

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 3 → 1 / (a + 1) + 2 / b ≥ 1 / (x + 1) + 2 / y) →
  1 / (x + 1) + 2 / y = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1702_170252


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l1702_170298

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (x_leq_2y : x ≤ 2*y)
  (y_leq_2z : y ≤ 2*z) :
  x * y * z ≥ 6 / 343 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l1702_170298


namespace NUMINAMATH_CALUDE_ellipse_equation_l1702_170208

/-- An ellipse with the given properties has the standard equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = c * Real.sqrt 3) 
  (h2 : c = Real.sqrt 3) : 
  ∃ (x y : ℝ), x^2 / 4 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1702_170208


namespace NUMINAMATH_CALUDE_divisibility_by_seventeen_l1702_170203

theorem divisibility_by_seventeen (x : ℤ) (y z w : ℕ) 
  (hy : Odd y) (hz : Odd z) (hw : Odd w) : 
  ∃ k : ℤ, x^(y^(z^w)) - x^(y^z) = 17 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_seventeen_l1702_170203


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l1702_170216

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (correct_value : ℝ) 
  (new_mean : ℝ) 
  (h1 : n = 40)
  (h2 : initial_mean = 100)
  (h3 : correct_value = 50)
  (h4 : new_mean = 99.075) :
  (n : ℝ) * initial_mean - (n : ℝ) * new_mean + correct_value = 87 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l1702_170216


namespace NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l1702_170221

theorem square_triangle_equal_perimeter (x : ℝ) : 
  (4 * (x + 2) = 3 * (2 * x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_perimeter_l1702_170221


namespace NUMINAMATH_CALUDE_decimal_power_division_l1702_170243

theorem decimal_power_division : (0.4 : ℝ)^4 / (0.04 : ℝ)^3 = 400 := by
  sorry

end NUMINAMATH_CALUDE_decimal_power_division_l1702_170243


namespace NUMINAMATH_CALUDE_diaries_calculation_l1702_170219

theorem diaries_calculation (initial_diaries : ℕ) : initial_diaries = 8 →
  (initial_diaries + 2 * initial_diaries) * 3 / 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_diaries_calculation_l1702_170219


namespace NUMINAMATH_CALUDE_track_length_is_480_l1702_170226

/-- Represents the circular track and the runners' movements --/
structure TrackSystem where
  trackLength : ℝ
  janetSpeed : ℝ
  leahSpeed : ℝ

/-- Conditions of the problem --/
def ProblemConditions (s : TrackSystem) : Prop :=
  s.janetSpeed > 0 ∧ 
  s.leahSpeed > 0 ∧ 
  120 / s.janetSpeed = (s.trackLength / 2 - 120) / s.leahSpeed ∧
  (s.trackLength / 2 - 40) / s.janetSpeed = 200 / s.leahSpeed

/-- The main theorem to prove --/
theorem track_length_is_480 (s : TrackSystem) : 
  ProblemConditions s → s.trackLength = 480 :=
sorry

end NUMINAMATH_CALUDE_track_length_is_480_l1702_170226


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1702_170286

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is a necessary but not sufficient condition for the curve to be an ellipse -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (is_ellipse m n → m * n > 0) ∧
  ∃ m n, m * n > 0 ∧ ¬(is_ellipse m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1702_170286


namespace NUMINAMATH_CALUDE_chemistry_physics_score_difference_l1702_170241

theorem chemistry_physics_score_difference
  (math_score physics_score chemistry_score : ℕ)
  (total_math_physics : math_score + physics_score = 60)
  (avg_math_chemistry : (math_score + chemistry_score) / 2 = 40)
  (chemistry_higher : chemistry_score > physics_score) :
  chemistry_score - physics_score = 20 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_physics_score_difference_l1702_170241


namespace NUMINAMATH_CALUDE_min_abs_beta_plus_delta_l1702_170271

open Complex

theorem min_abs_beta_plus_delta :
  ∀ β δ : ℂ,
  let g : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β*z + δ
  (g 1).im = 0 →
  (g (-I)).im = 0 →
  ∃ (β' δ' : ℂ),
    let g' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β'*z + δ'
    (g' 1).im = 0 ∧
    (g' (-I)).im = 0 ∧
    Complex.abs β' + Complex.abs δ' = 4 ∧
    ∀ (β'' δ'' : ℂ),
      let g'' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β''*z + δ''
      (g'' 1).im = 0 →
      (g'' (-I)).im = 0 →
      Complex.abs β'' + Complex.abs δ'' ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_beta_plus_delta_l1702_170271


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l1702_170295

/-- Represents the age ratio of a man to his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  (son_age + age_difference + 2) / (son_age + 2)

theorem man_son_age_ratio :
  let son_age : ℕ := 20
  let age_difference : ℕ := 22
  age_ratio son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l1702_170295


namespace NUMINAMATH_CALUDE_event_organization_ways_l1702_170255

def number_of_friends : ℕ := 5
def number_of_organizers : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem event_organization_ways :
  choose number_of_friends number_of_organizers = 10 := by
  sorry

end NUMINAMATH_CALUDE_event_organization_ways_l1702_170255


namespace NUMINAMATH_CALUDE_point_transformation_l1702_170270

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let reflected := reflectYeqX rotated.1 rotated.2
  reflected = (-4, 2) → b - a = -6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1702_170270


namespace NUMINAMATH_CALUDE_h_over_g_equals_64_l1702_170234

theorem h_over_g_equals_64 (G H : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
    G / (x + 3) + H / (x^2 - 5*x) = (x^2 - 3*x + 8) / (x^3 + x^2 - 15*x)) →
  (H : ℚ) / (G : ℚ) = 64 := by
sorry

end NUMINAMATH_CALUDE_h_over_g_equals_64_l1702_170234


namespace NUMINAMATH_CALUDE_home_run_multiple_l1702_170237

/-- The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to -/
theorem home_run_multiple (aaron_hr winfield_hr : ℕ) (difference : ℕ) : 
  aaron_hr = 755 →
  winfield_hr = 465 →
  aaron_hr + difference = 2 * winfield_hr →
  2 = (aaron_hr + difference) / winfield_hr :=
by
  sorry

end NUMINAMATH_CALUDE_home_run_multiple_l1702_170237


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l1702_170233

/-- Represents a cube sculpture with three layers --/
structure CubeSculpture where
  topLayer : ℕ
  middleLayer : ℕ
  bottomLayer : ℕ
  totalCubes : ℕ
  (total_eq : totalCubes = topLayer + middleLayer + bottomLayer)

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : CubeSculpture) : ℕ :=
  5 * s.topLayer +  -- Top cube: 5 sides exposed
  (5 + 4 * 4) +     -- Middle layer: 1 cube with 5 sides, 4 cubes with 4 sides
  s.bottomLayer     -- Bottom layer: only top faces exposed

/-- The main theorem to prove --/
theorem sculpture_surface_area :
  ∃ (s : CubeSculpture),
    s.topLayer = 1 ∧
    s.middleLayer = 5 ∧
    s.bottomLayer = 11 ∧
    s.totalCubes = 17 ∧
    exposedSurfaceArea s = 37 :=
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l1702_170233


namespace NUMINAMATH_CALUDE_shopping_expenses_total_l1702_170294

/-- Represents the shopping expenses of Lisa and Carly -/
def ShoppingExpenses (lisa_tshirt : ℝ) : Prop :=
  let lisa_jeans := lisa_tshirt / 2
  let lisa_coat := lisa_tshirt * 2
  let shoe_cost := lisa_jeans * 3
  let carly_tshirt := lisa_tshirt / 4
  let carly_jeans := lisa_jeans * 3
  let carly_coat := lisa_coat / 2
  let carly_dress := shoe_cost * 2
  let lisa_total := lisa_tshirt + lisa_jeans + lisa_coat + shoe_cost
  let carly_total := carly_tshirt + carly_jeans + carly_coat + shoe_cost + carly_dress
  lisa_tshirt = 40 ∧ lisa_total + carly_total = 490

/-- Theorem stating that the total amount spent by Lisa and Carly is $490 -/
theorem shopping_expenses_total : ShoppingExpenses 40 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expenses_total_l1702_170294


namespace NUMINAMATH_CALUDE_pi_half_irrational_l1702_170261

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l1702_170261


namespace NUMINAMATH_CALUDE_total_weight_proof_l1702_170232

theorem total_weight_proof (jim_weight steve_weight stan_weight : ℕ) 
  (h1 : stan_weight = steve_weight + 5)
  (h2 : steve_weight = jim_weight - 8)
  (h3 : jim_weight = 110) : 
  jim_weight + steve_weight + stan_weight = 319 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_proof_l1702_170232


namespace NUMINAMATH_CALUDE_joes_test_count_l1702_170247

theorem joes_test_count (n : ℕ) (initial_average final_average lowest_score : ℚ) 
  (h1 : initial_average = 50)
  (h2 : final_average = 55)
  (h3 : lowest_score = 35)
  (h4 : n * initial_average = (n - 1) * final_average + lowest_score)
  (h5 : n > 1) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_joes_test_count_l1702_170247


namespace NUMINAMATH_CALUDE_probability_matching_shoes_l1702_170296

theorem probability_matching_shoes (n : ℕ) (h : n = 9) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_matching_shoes_l1702_170296


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1702_170205

theorem max_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1702_170205


namespace NUMINAMATH_CALUDE_venus_speed_conversion_l1702_170287

/-- Converts a speed from miles per second to miles per hour -/
def miles_per_second_to_miles_per_hour (speed_mps : ℝ) : ℝ :=
  speed_mps * 3600

/-- The speed of Venus around the sun in miles per second -/
def venus_speed_mps : ℝ := 21.9

theorem venus_speed_conversion :
  miles_per_second_to_miles_per_hour venus_speed_mps = 78840 := by
  sorry

end NUMINAMATH_CALUDE_venus_speed_conversion_l1702_170287


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1702_170281

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l1702_170281


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_5_l1702_170239

theorem tan_alpha_2_implies_expression_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_5_l1702_170239


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1702_170224

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 - 16 * x - 16 * y^2 + 32 * y + 144 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_eq = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l1702_170224


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l1702_170256

theorem smallest_angle_satisfying_equation :
  let f : ℝ → ℝ := λ x => 9 * Real.sin x * Real.cos x ^ 4 - 9 * Real.sin x ^ 4 * Real.cos x
  ∃ x : ℝ, x > 0 ∧ f x = 1/2 ∧ ∀ y : ℝ, y > 0 → f y = 1/2 → x ≤ y ∧ x = π/6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_equation_l1702_170256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1702_170291

/-- Given an arithmetic sequence {a_n} where a₂ + a₄ = 8 and a₁ = 2, prove that a₅ = 6 -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) 
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_sum : a 2 + a 4 = 8)
  (h_first : a 1 = 2) : 
  a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1702_170291


namespace NUMINAMATH_CALUDE_latest_90_degrees_time_l1702_170282

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

-- Define the theorem
theorem latest_90_degrees_time :
  ∃ t : ℝ, t ≤ 17 ∧ temperature t = 90 ∧
  ∀ s : ℝ, s > 17 → temperature s ≠ 90 :=
by sorry

end NUMINAMATH_CALUDE_latest_90_degrees_time_l1702_170282


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1702_170251

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The second circle: x^2 + y^2 - 2mx + m^2 - 1 = 0 -/
def circle2 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 1 = 0}

theorem circles_externally_tangent :
  ∀ m : ℝ, (∃ p : ℝ × ℝ, p ∈ circle1 ∩ circle2 m) ∧
           externally_tangent (0, 0) (m, 0) 2 1 ↔ m = 3 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1702_170251


namespace NUMINAMATH_CALUDE_smallest_value_quadratic_l1702_170272

theorem smallest_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_quadratic_l1702_170272


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1702_170246

theorem boys_to_girls_ratio (total_students girls : ℕ) (h1 : total_students = 780) (h2 : girls = 300) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1702_170246


namespace NUMINAMATH_CALUDE_shane_semester_distance_l1702_170279

/-- Calculates the total distance traveled for round trips during a semester -/
def total_semester_distance (daily_one_way_distance : ℕ) (semester_days : ℕ) : ℕ :=
  2 * daily_one_way_distance * semester_days

/-- Proves that the total distance traveled during the semester is 1600 miles -/
theorem shane_semester_distance :
  total_semester_distance 10 80 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_shane_semester_distance_l1702_170279


namespace NUMINAMATH_CALUDE_circumscribed_equal_triangulation_only_square_l1702_170268

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 3

/-- A polygon is circumscribed if all its sides are tangent to a common circle -/
def IsCircumscribed (P : ConvexPolygon n) : Prop :=
  sorry

/-- A polygon can be dissected into equal triangles by non-intersecting diagonals -/
def HasEqualTriangulation (P : ConvexPolygon n) : Prop :=
  sorry

/-- The main theorem -/
theorem circumscribed_equal_triangulation_only_square
  (n : ℕ) (P : ConvexPolygon n)
  (h_circ : IsCircumscribed P)
  (h_triang : HasEqualTriangulation P) :
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_equal_triangulation_only_square_l1702_170268


namespace NUMINAMATH_CALUDE_teacher_assignment_ways_l1702_170240

/-- The number of ways to assign teachers to classes -/
def assignmentWays (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2

/-- Theorem stating the number of ways to assign 4 teachers to 8 classes -/
theorem teacher_assignment_ways :
  assignmentWays 8 4 = 2520 :=
by sorry

#eval assignmentWays 8 4

end NUMINAMATH_CALUDE_teacher_assignment_ways_l1702_170240


namespace NUMINAMATH_CALUDE_roots_equation_result_l1702_170220

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ + 1 = 0 → δ^2 - 3*δ + 1 = 0 → 8*γ^3 + 15*δ^2 = 179 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_result_l1702_170220


namespace NUMINAMATH_CALUDE_points_per_enemy_l1702_170278

/-- 
Given a video game level with the following conditions:
- There are 11 enemies in total
- Defeating all but 3 enemies results in 72 points
This theorem proves that the number of points earned for defeating one enemy is 9.
-/
theorem points_per_enemy (total_enemies : ℕ) (remaining_enemies : ℕ) (total_points : ℕ) :
  total_enemies = 11 →
  remaining_enemies = 3 →
  total_points = 72 →
  (total_points / (total_enemies - remaining_enemies) : ℚ) = 9 := by
  sorry

#check points_per_enemy

end NUMINAMATH_CALUDE_points_per_enemy_l1702_170278


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_is_31_l1702_170288

theorem sum_of_A_and_B_is_31 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 11 * x + 35) / (x - 3)) →
  A + B = 31 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_is_31_l1702_170288


namespace NUMINAMATH_CALUDE_voldemort_calorie_limit_l1702_170274

/-- Voldemort's daily calorie intake limit -/
def daily_calorie_limit : ℕ := by sorry

/-- Calories from breakfast -/
def breakfast_calories : ℕ := 560

/-- Calories from lunch -/
def lunch_calories : ℕ := 780

/-- Calories from dinner -/
def dinner_calories : ℕ := 110 + 310 + 215

/-- Remaining calories Voldemort can still take -/
def remaining_calories : ℕ := 525

/-- Theorem stating Voldemort's daily calorie intake limit -/
theorem voldemort_calorie_limit :
  daily_calorie_limit = breakfast_calories + lunch_calories + dinner_calories + remaining_calories := by
  sorry

end NUMINAMATH_CALUDE_voldemort_calorie_limit_l1702_170274


namespace NUMINAMATH_CALUDE_square_sum_xy_l1702_170292

theorem square_sum_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1702_170292


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l1702_170225

theorem floor_of_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l1702_170225


namespace NUMINAMATH_CALUDE_teacher_class_choices_l1702_170229

theorem teacher_class_choices (n_teachers : ℕ) (n_classes : ℕ) : 
  n_teachers = 5 → n_classes = 4 → (n_classes : ℕ) ^ n_teachers = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_teacher_class_choices_l1702_170229


namespace NUMINAMATH_CALUDE_max_value_is_b_l1702_170222

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (1/2) (max b (max (2*a*b) (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_b_l1702_170222


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l1702_170273

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-3, -4)

/-- The expected reflected point -/
def expected_reflection : ℝ × ℝ := (-3, 4)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l1702_170273


namespace NUMINAMATH_CALUDE_fair_payment_division_l1702_170236

/-- Represents the payment for digging the trench -/
def total_payment : ℚ := 2

/-- Represents Abraham's digging rate relative to Benjamin's soil throwing rate -/
def abraham_dig_rate : ℚ := 1

/-- Represents Benjamin's digging rate relative to Abraham's soil throwing rate -/
def benjamin_dig_rate : ℚ := 4

/-- Represents the ratio of Abraham's payment to the total payment -/
def abraham_payment_ratio : ℚ := 1/3

/-- Represents the ratio of Benjamin's payment to the total payment -/
def benjamin_payment_ratio : ℚ := 2/3

/-- Theorem stating the fair division of payment between Abraham and Benjamin -/
theorem fair_payment_division :
  abraham_payment_ratio * total_payment = 2/3 ∧
  benjamin_payment_ratio * total_payment = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_fair_payment_division_l1702_170236


namespace NUMINAMATH_CALUDE_springdale_rainfall_l1702_170209

theorem springdale_rainfall (first_week : ℝ) (second_week : ℝ) : 
  second_week = 1.5 * first_week →
  second_week = 24 →
  first_week + second_week = 40 := by
sorry

end NUMINAMATH_CALUDE_springdale_rainfall_l1702_170209


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l1702_170290

/-- Represents the selling price of the product -/
def selling_price : ℝ → ℝ := id

/-- Represents the purchase cost of the product -/
def purchase_cost : ℝ := 40

/-- Represents the number of units sold at a given price -/
def units_sold (x : ℝ) : ℝ := 500 - 20 * (x - 50)

/-- Represents the profit at a given selling price -/
def profit (x : ℝ) : ℝ := (x - purchase_cost) * (units_sold x)

/-- Theorem stating that the profit-maximizing selling price is 57.5 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit x ≥ profit y ∧ x = 57.5 := by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l1702_170290


namespace NUMINAMATH_CALUDE_problem_solution_l1702_170242

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : b < c) :
  (a^2 * b < a^2 * c) ∧ (a^3 < a^2 * b) ∧ (a + b < b + c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1702_170242


namespace NUMINAMATH_CALUDE_books_ratio_l1702_170223

theorem books_ratio (harry_books : ℕ) (total_books : ℕ) : 
  harry_books = 50 →
  total_books = 175 →
  ∃ (flora_books : ℕ),
    flora_books = 2 * harry_books ∧
    harry_books + flora_books + (harry_books / 2) = total_books :=
by sorry

end NUMINAMATH_CALUDE_books_ratio_l1702_170223


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1702_170266

/-- The amount of money left after buying a gift and cake for their mother -/
def money_left (gift_cost cake_cost erika_savings : ℚ) : ℚ :=
  let rick_savings := gift_cost / 2
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  total_savings - total_cost

/-- Theorem stating the amount of money left after buying the gift and cake -/
theorem money_left_after_purchase : 
  money_left 250 25 155 = 5 := by sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1702_170266


namespace NUMINAMATH_CALUDE_college_cost_calculation_l1702_170264

/-- The total cost of Sabina's first year of college -/
def total_cost : ℝ := 30000

/-- Sabina's savings -/
def savings : ℝ := 10000

/-- The percentage of the remainder covered by the grant -/
def grant_percentage : ℝ := 0.40

/-- The amount of the loan Sabina needs -/
def loan_amount : ℝ := 12000

/-- Theorem stating that the total cost is correct given the conditions -/
theorem college_cost_calculation :
  total_cost = savings + grant_percentage * (total_cost - savings) + loan_amount := by
  sorry

end NUMINAMATH_CALUDE_college_cost_calculation_l1702_170264


namespace NUMINAMATH_CALUDE_shopkeeper_loss_per_metre_l1702_170267

/-- Calculates the loss per metre of cloth sold by a shopkeeper -/
theorem shopkeeper_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 36000)
  (h3 : cost_price_per_metre = 70) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_per_metre_l1702_170267


namespace NUMINAMATH_CALUDE_no_winning_strategy_l1702_170262

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)
  (total : ℕ)
  (h_total : total = red + black)
  (h_standard : total = 52 ∧ red = 26 ∧ black = 26)

/-- Represents a strategy for playing the card game -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state and a strategy -/
noncomputable def win_probability (d : Deck) (s : Strategy) : ℝ :=
  d.red / d.total

/-- Theorem stating that no strategy can have a winning probability greater than 0.5 -/
theorem no_winning_strategy (s : Strategy) :
  ∀ d : Deck, win_probability d s ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l1702_170262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1702_170263

theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := 1
  let aₙ : ℤ := 1996
  let n : ℕ := 96
  let s := n * (a₁ + aₙ) / 2
  s = 95856 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1702_170263


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_rationalize_35_cube_root_l1702_170231

theorem rationalize_denominator_cube_root (x : ℝ) (hx : x > 0) :
  (x / x^(1/3)) = x^(2/3) :=
by sorry

theorem rationalize_35_cube_root :
  (35 : ℝ) / (35 : ℝ)^(1/3) = (1225 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_rationalize_35_cube_root_l1702_170231


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1702_170257

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1702_170257


namespace NUMINAMATH_CALUDE_fraction_simplification_l1702_170269

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 11 + 2 / 9) = 4257 / 2345 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1702_170269


namespace NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l1702_170227

theorem residue_of_negative_1235_mod_29 : 
  ∃ k : ℤ, -1235 = 29 * k + 12 ∧ 0 ≤ 12 ∧ 12 < 29 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_1235_mod_29_l1702_170227


namespace NUMINAMATH_CALUDE_sue_initial_savings_proof_l1702_170202

/-- The cost of the perfume in dollars -/
def perfume_cost : ℝ := 50

/-- Christian's initial savings in dollars -/
def christian_initial_savings : ℝ := 5

/-- Number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- Cost per yard mowed in dollars -/
def cost_per_yard : ℝ := 5

/-- Number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- Cost per dog walked in dollars -/
def cost_per_dog : ℝ := 2

/-- Additional amount needed in dollars -/
def additional_needed : ℝ := 6

/-- Sue's initial savings in dollars -/
def sue_initial_savings : ℝ := 7

theorem sue_initial_savings_proof :
  sue_initial_savings = 
    perfume_cost - 
    (christian_initial_savings + 
     (yards_mowed : ℝ) * cost_per_yard + 
     (dogs_walked : ℝ) * cost_per_dog) := by
  sorry

end NUMINAMATH_CALUDE_sue_initial_savings_proof_l1702_170202


namespace NUMINAMATH_CALUDE_age_difference_l1702_170214

/-- Given Sierra's current age and Diaz's age 20 years from now, 
    prove that the difference between (10 times Diaz's current age minus 40) 
    and (10 times Sierra's current age) is 20. -/
theorem age_difference (sierra_age : ℕ) (diaz_future_age : ℕ) : 
  sierra_age = 30 → 
  diaz_future_age = 56 → 
  (10 * (diaz_future_age - 20) - 40) - (10 * sierra_age) = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l1702_170214


namespace NUMINAMATH_CALUDE_median_of_special_list_l1702_170275

def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def isMedian (m : ℕ) : Prop :=
  let N := sumOfSquares 100
  let leftCount := sumOfSquares (m - 1)
  let rightCount := sumOfSquares m
  N / 2 > leftCount ∧ N / 2 ≤ rightCount

theorem median_of_special_list : isMedian 72 := by
  sorry

end NUMINAMATH_CALUDE_median_of_special_list_l1702_170275


namespace NUMINAMATH_CALUDE_sum_of_square_roots_equals_one_l1702_170293

theorem sum_of_square_roots_equals_one (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  Real.sqrt b / (a + b) + Real.sqrt c / (b + c) + Real.sqrt a / (c + a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_equals_one_l1702_170293


namespace NUMINAMATH_CALUDE_comparison_theorems_l1702_170204

theorem comparison_theorems :
  (∀ a : ℝ, a < 0 → a / (a - 1) > 0) ∧
  (∀ x : ℝ, x < -1 → 2 / (x^2 - 1) > (x - 1) / (x^2 - 2*x + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → 2*x*y / (x + y) < (x + y) / 2) :=
by sorry

end NUMINAMATH_CALUDE_comparison_theorems_l1702_170204


namespace NUMINAMATH_CALUDE_divisibility_problem_l1702_170259

def solution_set : Set Int :=
  {-21, -9, -5, -3, -1, 1, 2, 4, 5, 6, 7, 9, 11, 15, 27}

theorem divisibility_problem :
  ∀ x : Int, x ≠ 3 ∧ (x - 3 ∣ x^3 - 3) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1702_170259


namespace NUMINAMATH_CALUDE_cruz_marbles_l1702_170249

/-- Proof that Cruz has 8 marbles given the conditions of the problem -/
theorem cruz_marbles :
  ∀ (atticus jensen cruz : ℕ),
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end NUMINAMATH_CALUDE_cruz_marbles_l1702_170249


namespace NUMINAMATH_CALUDE_ear_muffs_bought_in_december_l1702_170289

theorem ear_muffs_bought_in_december (before_december : ℕ) (total : ℕ) 
  (h1 : before_december = 1346)
  (h2 : total = 7790) :
  total - before_december = 6444 :=
by sorry

end NUMINAMATH_CALUDE_ear_muffs_bought_in_december_l1702_170289


namespace NUMINAMATH_CALUDE_gcd_60_75_l1702_170250

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_60_75_l1702_170250


namespace NUMINAMATH_CALUDE_total_faces_painted_l1702_170297

/-- The number of cuboids painted by Ezekiel -/
def num_cuboids : ℕ := 5

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Ezekiel -/
theorem total_faces_painted :
  num_cuboids * faces_per_cuboid = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_faces_painted_l1702_170297


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1702_170254

theorem no_integer_solutions : ¬∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3*x*y := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1702_170254


namespace NUMINAMATH_CALUDE_matthews_friends_l1702_170215

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 22)
  (h2 : crackers_per_friend = 2) :
  total_crackers / crackers_per_friend = 11 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l1702_170215


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l1702_170211

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 95 → red = 26 → yellow = 37 → blue = total - red - yellow → blue = 32 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l1702_170211


namespace NUMINAMATH_CALUDE_marys_number_proof_l1702_170200

theorem marys_number_proof : ∃! x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 
    3 * x + 11 = 10 * a + b ∧
    10 * b + a ≥ 71 ∧ 
    10 * b + a ≤ 75) ∧
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_marys_number_proof_l1702_170200


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1702_170213

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1)^3 * (z + 1)) ≤ 12 * Real.sqrt 3 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 1)^3 * (w + 1)) = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1702_170213


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_range_l1702_170244

/-- The function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_positive_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_range_l1702_170244


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1702_170283

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  (x - a - 1)^2 + (y - Real.sqrt 3 * a)^2 = 1

-- Define the condition |MA| = 2|MO|
def condition_M (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * (x^2 + y^2)

-- Define the range of a
def range_a (a : ℝ) : Prop :=
  (1/2 ≤ a ∧ a ≤ 3/2) ∨ (-3/2 ≤ a ∧ a ≤ -1/2)

theorem circle_intersection_range :
  ∀ a : ℝ, (∃ x y : ℝ, circle_C a x y ∧ condition_M x y) ↔ range_a a :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1702_170283


namespace NUMINAMATH_CALUDE_tea_consumption_average_l1702_170201

/-- Represents the inverse proportionality between research hours and tea quantity -/
def inverse_prop (k : ℝ) (r t : ℝ) : Prop := r * t = k

theorem tea_consumption_average (k : ℝ) :
  k = 8 * 3 →
  let t1 := k / 5
  let t2 := k / 10
  let t3 := k / 7
  (t1 + t2 + t3) / 3 = 124 / 35 := by
  sorry

end NUMINAMATH_CALUDE_tea_consumption_average_l1702_170201


namespace NUMINAMATH_CALUDE_rational_function_property_l1702_170206

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ  -- Linear function
  q : ℝ → ℝ  -- Quadratic function
  linear_p : ∃ a b : ℝ, ∀ x, p x = a * x + b
  quadratic_q : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_through_point : p 1 / q 1 = 2

theorem rational_function_property (f : RationalFunction) : f.p 0 / f.q 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_property_l1702_170206


namespace NUMINAMATH_CALUDE_square_roots_problem_l1702_170238

theorem square_roots_problem (x : ℝ) :
  (∃ (a : ℝ), a > 0 ∧ (x + 1)^2 = a ∧ (x - 5)^2 = a) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1702_170238


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1702_170245

/-- Given a line L1 with equation 3x + 4y + 5 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4y - 3x - 3 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x + 4 * y + 5 = 0) →
  P = (0, -3) →
  ∃ (L2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 4 * y - 3 * x - 3 = 0) ∧
    P ∈ L2 ∧
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w →
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q →
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1702_170245


namespace NUMINAMATH_CALUDE_fraction_transformation_l1702_170260

theorem fraction_transformation (a b : ℝ) (h : a * b > 0) :
  (3 * a + 2 * (3 * b)) / (2 * (3 * a) * (3 * b)) = (1 / 3) * ((a + 2 * b) / (2 * a * b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1702_170260


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l1702_170217

theorem half_abs_diff_squares : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l1702_170217


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l1702_170285

theorem medicine_supply_duration (pills_per_supply : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) (days_per_month : ℕ) : 
  pills_per_supply = 90 →
  pill_fraction = 1/3 →
  days_between_doses = 3 →
  days_per_month = 30 →
  (pills_per_supply * (days_between_doses / pill_fraction) / days_per_month : ℚ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l1702_170285


namespace NUMINAMATH_CALUDE_unique_solution_abc_l1702_170207

theorem unique_solution_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a+1)^2 / (b+c-1) + (b+2)^2 / (c+a-3) + (c+3)^2 / (a+b-5) = 32) :
  a = 8 ∧ b = 6 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l1702_170207


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1702_170277

theorem smaller_number_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = 3 / 4) (h4 : a + b = 21) (h5 : max a b = 12) : 
  min a b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1702_170277


namespace NUMINAMATH_CALUDE_gcd_280_2155_l1702_170299

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_280_2155_l1702_170299


namespace NUMINAMATH_CALUDE_function_inequality_solution_l1702_170235

-- Define the function f
noncomputable def f (x : ℝ) (p q a b : ℝ) (h : ℝ → ℝ) : ℝ :=
  if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)

-- State the theorem
theorem function_inequality_solution (p q a b : ℝ) (h : ℝ → ℝ) :
  q ≠ 1 →
  (∀ x, q > 0 → h (x + p) ≥ h x) →
  (∀ x, q < 0 → h (x + p) ≥ -h x) →
  (∀ x, f (x + p) p q a b h - q * f x p q a b h ≥ a * x + b) ↔
  (∀ x, f x p q a b h = if q > 0 then
    (a * x) / (1 - q) + h x * q^x + b / (1 - q) - a * p / ((1 - q)^2)
  else
    (a * x) / (1 - q) + h x * (-q)^x + b / (1 - q) - a * p / ((1 - q)^2)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l1702_170235


namespace NUMINAMATH_CALUDE_next_next_perfect_square_l1702_170230

theorem next_next_perfect_square (x : ℕ) (k : ℕ) (h : x = k^2) :
  (k + 2)^2 = x + 4 * Int.sqrt x + 4 :=
sorry

end NUMINAMATH_CALUDE_next_next_perfect_square_l1702_170230


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l1702_170218

/-- The number of coins Ricardo has -/
def total_coins : ℕ := 1980

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of dimes Ricardo has -/
def num_dimes : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => penny_value * num_pennies p + dime_value * num_dimes p

/-- The maximum possible value of Ricardo's coins in cents -/
def max_value : ℕ := total_value 1

/-- The minimum possible value of Ricardo's coins in cents -/
def min_value : ℕ := total_value (total_coins - 1)

theorem ricardo_coin_difference :
  max_value - min_value = 17802 :=
sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l1702_170218


namespace NUMINAMATH_CALUDE_tangent_line_at_one_range_of_m_l1702_170258

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (x - 1) * Real.log x - m * (x + 1)

-- Part 1: Tangent line equation
theorem tangent_line_at_one (m : ℝ) (h : m = 1) :
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
  (∀ x y : ℝ, y = f x m → (a * x + b * y + c = 0 ↔ x = 1)) :=
sorry

-- Part 2: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_range_of_m_l1702_170258


namespace NUMINAMATH_CALUDE_cyclist_speed_proof_l1702_170228

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 28.5

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 6

/-- The distance between City X and City Y in miles -/
def distance_XY : ℝ := 100

/-- The distance C travels before turning back in miles -/
def distance_C_before_turn : ℝ := 80

/-- The distance from City Y where C and D meet after turning back in miles -/
def meeting_distance : ℝ := 15

theorem cyclist_speed_proof :
  speed_C = 28.5 ∧
  speed_D = speed_C + 6 ∧
  (distance_C_before_turn + meeting_distance) / speed_C = 
  (distance_XY + meeting_distance) / speed_D :=
sorry

end NUMINAMATH_CALUDE_cyclist_speed_proof_l1702_170228


namespace NUMINAMATH_CALUDE_largest_number_with_distinct_digits_summing_to_19_l1702_170265

/-- Checks if all digits in a number are different -/
def hasDistinctDigits (n : ℕ) : Bool := sorry

/-- Calculates the sum of digits of a number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The target sum of digits -/
def targetSum : ℕ := 19

/-- The proposed largest number -/
def largestNumber : ℕ := 943210

theorem largest_number_with_distinct_digits_summing_to_19 :
  (∀ m : ℕ, m > largestNumber → 
    ¬(hasDistinctDigits m ∧ digitSum m = targetSum)) ∧
  hasDistinctDigits largestNumber ∧
  digitSum largestNumber = targetSum :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_distinct_digits_summing_to_19_l1702_170265
