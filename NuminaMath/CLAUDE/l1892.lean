import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_l1892_189285

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem quadratic_minimum :
  (∃ (x : ℝ), f x = -25) ∧ (∀ (y : ℝ), f y ≥ -25) ∧ (f (-7) = -25) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1892_189285


namespace NUMINAMATH_CALUDE_smallest_w_l1892_189232

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (11^2) (936 * w) → 
  ∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (11^2) (936 * v) → 
    w ≤ v →
  w = 4356 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l1892_189232


namespace NUMINAMATH_CALUDE_walking_problem_l1892_189288

/-- The problem of two people walking towards each other on a road -/
theorem walking_problem (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) :
  total_distance = 40 ∧ 
  yolanda_speed = 2 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ (meeting_time : ℝ),
    meeting_time > 0 ∧
    head_start * yolanda_speed + meeting_time * yolanda_speed + meeting_time * bob_speed = total_distance ∧
    meeting_time * bob_speed = 25 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l1892_189288


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1892_189233

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ+), x^2 + y^2 = x^3 + 2*y := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1892_189233


namespace NUMINAMATH_CALUDE_smallest_b_value_l1892_189266

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 4) 
  (h2 : Nat.gcd ((a^3 - b^3) / (a - b)) (a * b) = 4) : 
  b = 2 ∧ ∀ (c : ℕ+), c < b → ¬(∃ (d : ℕ+), d - c = 4 ∧ 
    Nat.gcd ((d^3 - c^3) / (d - c)) (d * c) = 4) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1892_189266


namespace NUMINAMATH_CALUDE_existence_implies_upper_bound_l1892_189222

theorem existence_implies_upper_bound (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_existence_implies_upper_bound_l1892_189222


namespace NUMINAMATH_CALUDE_asphalt_cost_per_truckload_l1892_189208

/-- Calculates the cost per truckload of asphalt before tax -/
theorem asphalt_cost_per_truckload
  (road_length : ℝ)
  (road_width : ℝ)
  (coverage_per_truckload : ℝ)
  (tax_rate : ℝ)
  (total_cost_with_tax : ℝ)
  (h1 : road_length = 2000)
  (h2 : road_width = 20)
  (h3 : coverage_per_truckload = 800)
  (h4 : tax_rate = 0.2)
  (h5 : total_cost_with_tax = 4500) :
  (road_length * road_width) / coverage_per_truckload *
  (total_cost_with_tax / (1 + tax_rate)) /
  ((road_length * road_width) / coverage_per_truckload) = 75 := by
sorry

end NUMINAMATH_CALUDE_asphalt_cost_per_truckload_l1892_189208


namespace NUMINAMATH_CALUDE_function_equivalence_and_coefficient_sum_l1892_189231

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 - 4*x - 12) / (x + 3)

def g (x : ℝ) : ℝ := x^2 - 4

def A : ℝ := 1
def B : ℝ := 0
def C : ℝ := -4
def D : ℝ := -3

theorem function_equivalence_and_coefficient_sum :
  (∀ x : ℝ, x ≠ D → f x = g x) ∧
  A + B + C + D = -6 := by sorry

end NUMINAMATH_CALUDE_function_equivalence_and_coefficient_sum_l1892_189231


namespace NUMINAMATH_CALUDE_smaller_number_proof_l1892_189286

theorem smaller_number_proof (x y : ℝ) (sum_eq : x + y = 79) (diff_eq : x - y = 15) :
  y = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l1892_189286


namespace NUMINAMATH_CALUDE_unique_b_value_l1892_189259

theorem unique_b_value : ∃! b : ℚ, ∀ x : ℚ, 5 * (3 * x - b) = 3 * (5 * x - 9) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l1892_189259


namespace NUMINAMATH_CALUDE_apple_difference_l1892_189256

/-- Proves that the difference between green and red apples after delivery is 140 -/
theorem apple_difference (initial_green : ℕ) (initial_red_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 32 →
  initial_red_difference = 200 →
  delivered_green = 340 →
  (initial_green + delivered_green) - (initial_green + initial_red_difference) = 140 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l1892_189256


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1892_189284

theorem fraction_evaluation : 
  let x : ℚ := 5
  (x^6 - 16*x^3 + x^2 + 64) / (x^3 - 8) = 4571 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1892_189284


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1892_189211

theorem complex_fraction_sum (x y : ℝ) : 
  (∃ (z : ℂ), z = (1 + y * Complex.I) / (1 + Complex.I) ∧ (z : ℂ).re = x) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1892_189211


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l1892_189279

theorem complex_expression_evaluation : 
  (((3.2 - 1.7) / 0.003) / ((29 / 35 - 3 / 7) * 4 / 0.2) - 
   ((1 + 13 / 20 - 1.5) * 1.5) / ((2.44 + (1 + 14 / 25)) * (1 / 8))) / (62 + 1 / 20) + 
  (1.364 / 0.124) = 12 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l1892_189279


namespace NUMINAMATH_CALUDE_fourth_grade_students_l1892_189214

/-- The number of students in fourth grade at the end of the year -/
def final_students (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial conditions, the final number of students is 11 -/
theorem fourth_grade_students : final_students 8 5 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l1892_189214


namespace NUMINAMATH_CALUDE_joys_remaining_tape_l1892_189200

/-- Calculates the remaining tape after wrapping a rectangular field once. -/
def remaining_tape (total_tape : ℝ) (width : ℝ) (length : ℝ) : ℝ :=
  total_tape - (2 * (width + length))

/-- Theorem stating the remaining tape for Joy's specific problem. -/
theorem joys_remaining_tape :
  remaining_tape 250 20 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_joys_remaining_tape_l1892_189200


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1892_189267

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 12 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1892_189267


namespace NUMINAMATH_CALUDE_tan_a_values_l1892_189212

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tan_a_values_l1892_189212


namespace NUMINAMATH_CALUDE_hexagon_star_perimeter_constant_l1892_189230

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Fin 6 → Point

/-- Checks if a hexagon is equilateral -/
def isEquilateral (h : Hexagon) : Prop := sorry

/-- Calculates the perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ := sorry

/-- Calculates the perimeter of the star formed by extending the sides of the hexagon -/
def starPerimeter (h : Hexagon) : ℝ := sorry

theorem hexagon_star_perimeter_constant 
  (h : Hexagon) 
  (equilateral : isEquilateral h) 
  (unit_perimeter : perimeter h = 1) :
  ∀ (h' : Hexagon), 
    isEquilateral h' → 
    perimeter h' = 1 → 
    starPerimeter h = starPerimeter h' :=
sorry

end NUMINAMATH_CALUDE_hexagon_star_perimeter_constant_l1892_189230


namespace NUMINAMATH_CALUDE_kim_total_points_l1892_189280

/-- Represents the points awarded for each round in the contest -/
structure RoundPoints where
  easy : Nat
  average : Nat
  hard : Nat

/-- Represents the number of correct answers in each round -/
structure CorrectAnswers where
  easy : Nat
  average : Nat
  hard : Nat

/-- Calculates the total points given the round points and correct answers -/
def calculateTotalPoints (points : RoundPoints) (answers : CorrectAnswers) : Nat :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

/-- Theorem: Given the contest conditions, Kim's total points are 38 -/
theorem kim_total_points :
  let points : RoundPoints := ⟨2, 3, 5⟩
  let answers : CorrectAnswers := ⟨6, 2, 4⟩
  calculateTotalPoints points answers = 38 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_points_l1892_189280


namespace NUMINAMATH_CALUDE_road_repaving_today_distance_l1892_189228

/-- Represents the repaving progress of a road construction project -/
structure RoadRepaving where
  totalRepaved : ℕ
  repavedBefore : ℕ

/-- Calculates the distance repaved today given the total repaved and repaved before -/
def distanceRepavedToday (r : RoadRepaving) : ℕ :=
  r.totalRepaved - r.repavedBefore

/-- Theorem: For the given road repaving project, the distance repaved today is 805 inches -/
theorem road_repaving_today_distance 
  (r : RoadRepaving) 
  (h1 : r.totalRepaved = 4938) 
  (h2 : r.repavedBefore = 4133) : 
  distanceRepavedToday r = 805 := by
  sorry

#eval distanceRepavedToday { totalRepaved := 4938, repavedBefore := 4133 }

end NUMINAMATH_CALUDE_road_repaving_today_distance_l1892_189228


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1892_189287

theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 7 < 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1892_189287


namespace NUMINAMATH_CALUDE_intersection_of_lines_l1892_189243

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space of the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : IntersectionPoint) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

/-- The theorem stating the intersection point of two specific lines -/
theorem intersection_of_lines :
  let line1 : Line := { m := 3, b := -1 }
  let line2 : Line := { m := -6, b := -4 }
  let point : IntersectionPoint := { x := -1/3, y := -2 }
  (pointOnLine point line1) ∧ (pointOnLine point line2) ∧
  (∀ p : IntersectionPoint, (pointOnLine p line1) ∧ (pointOnLine p line2) → p = point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l1892_189243


namespace NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_base8_l1892_189268

/-- The largest integer whose square has exactly 4 digits in base 8 -/
def M : ℕ := 31

/-- Conversion of M to base 8 -/
def M_base8 : ℕ := 37

theorem largest_integer_with_four_digit_square_base8 :
  (∀ n : ℕ, n > M → ¬(8^3 ≤ n^2 ∧ n^2 < 8^4)) ∧
  (8^3 ≤ M^2 ∧ M^2 < 8^4) ∧
  M_base8 = M := by sorry

end NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_base8_l1892_189268


namespace NUMINAMATH_CALUDE_fifth_root_of_unity_l1892_189223

theorem fifth_root_of_unity (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + 1 = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + m + p = 0) :
  m^5 = 1 := by sorry

end NUMINAMATH_CALUDE_fifth_root_of_unity_l1892_189223


namespace NUMINAMATH_CALUDE_cube_plane_angle_l1892_189294

/-- Given a cube with a plane passing through a side of its base, dividing the volume
    in the ratio m:n (where m ≤ n), the angle α between this plane and the base of
    the cube is given by α = arctan(2m / (m + n)). -/
theorem cube_plane_angle (m n : ℝ) (h : 0 < m ∧ m ≤ n) : 
  ∃ (α : ℝ), α = Real.arctan (2 * m / (m + n)) ∧
  ∃ (V₁ V₂ : ℝ), V₁ / V₂ = m / n ∧
  V₁ = (1/2) * (Real.tan α) ∧
  V₂ = 1 - (1/2) * (Real.tan α) := by
sorry

end NUMINAMATH_CALUDE_cube_plane_angle_l1892_189294


namespace NUMINAMATH_CALUDE_albert_earnings_increase_l1892_189239

theorem albert_earnings_increase (E : ℝ) (P : ℝ) 
  (h1 : E * (1 + P) = 693)
  (h2 : E * 1.20 = 660) :
  P = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_albert_earnings_increase_l1892_189239


namespace NUMINAMATH_CALUDE_store_items_cost_price_l1892_189264

/-- The cost price of an item given its profit and loss prices -/
def costPrice (profitPrice lossPrice : ℚ) : ℚ := (profitPrice + lossPrice) / 2

/-- The combined cost price of three items -/
def combinedCostPrice (cpA cpB cpC : ℚ) : ℚ := cpA + cpB + cpC

theorem store_items_cost_price : 
  let cpA := costPrice 110 70
  let cpB := costPrice 90 30
  let cpC := costPrice 150 50
  combinedCostPrice cpA cpB cpC = 250 := by
sorry

#eval costPrice 110 70 -- Expected output: 90
#eval costPrice 90 30  -- Expected output: 60
#eval costPrice 150 50 -- Expected output: 100
#eval combinedCostPrice (costPrice 110 70) (costPrice 90 30) (costPrice 150 50) -- Expected output: 250

end NUMINAMATH_CALUDE_store_items_cost_price_l1892_189264


namespace NUMINAMATH_CALUDE_locus_is_circle_l1892_189218

def locus_of_z (z₀ : ℂ) (z : ℂ) : Prop :=
  z₀ ≠ 0 ∧ z ≠ 0 ∧ ∃ z₁ : ℂ, Complex.abs (z₁ - z₀) = Complex.abs z₁ ∧ z₁ * z = -1

theorem locus_is_circle (z₀ : ℂ) (z : ℂ) :
  locus_of_z z₀ z → Complex.abs (z + 1 / z₀) = 1 / Complex.abs z₀ :=
by sorry

end NUMINAMATH_CALUDE_locus_is_circle_l1892_189218


namespace NUMINAMATH_CALUDE_max_type_c_tubes_l1892_189205

/-- Represents the number of test tubes of each type -/
structure TestTubes where
  a : ℕ  -- Type A (10% solution)
  b : ℕ  -- Type B (20% solution)
  c : ℕ  -- Type C (90% solution)

/-- The problem constraints -/
def validSolution (t : TestTubes) : Prop :=
  -- Total number of test tubes is 1000
  t.a + t.b + t.c = 1000 ∧
  -- The resulting solution is 20.17%
  10 * t.a + 20 * t.b + 90 * t.c = 2017 * (t.a + t.b + t.c) ∧
  -- Two consecutive pourings cannot use test tubes of the same type
  t.a > 0 ∧ t.b > 0

/-- The theorem statement -/
theorem max_type_c_tubes :
  ∃ (t : TestTubes), validSolution t ∧
    (∀ (t' : TestTubes), validSolution t' → t'.c ≤ t.c) ∧
    t.c = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_type_c_tubes_l1892_189205


namespace NUMINAMATH_CALUDE_function_identity_l1892_189206

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_identity_l1892_189206


namespace NUMINAMATH_CALUDE_monica_students_l1892_189281

/-- The number of students Monica sees each day -/
def total_students : ℕ :=
  let first_class := 20
  let second_third_classes := 25 + 25
  let fourth_class := first_class / 2
  let fifth_sixth_classes := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Monica sees 136 students each day -/
theorem monica_students : total_students = 136 := by
  sorry

end NUMINAMATH_CALUDE_monica_students_l1892_189281


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1892_189246

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1892_189246


namespace NUMINAMATH_CALUDE_saras_basketball_games_l1892_189262

theorem saras_basketball_games (won_games lost_games : ℕ) 
  (h1 : won_games = 12) 
  (h2 : lost_games = 4) : 
  won_games + lost_games = 16 := by
  sorry

end NUMINAMATH_CALUDE_saras_basketball_games_l1892_189262


namespace NUMINAMATH_CALUDE_total_jumps_eq_308_l1892_189251

/-- The total number of times Joonyoung and Namyoung jumped rope --/
def total_jumps (joonyoung_freq : ℕ) (joonyoung_months : ℕ) (namyoung_freq : ℕ) (namyoung_months : ℕ) : ℕ :=
  joonyoung_freq * joonyoung_months + namyoung_freq * namyoung_months

/-- Theorem stating that the total jumps for Joonyoung and Namyoung is 308 --/
theorem total_jumps_eq_308 :
  total_jumps 56 3 35 4 = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_eq_308_l1892_189251


namespace NUMINAMATH_CALUDE_min_triangles_in_configuration_l1892_189238

/-- A configuration of lines in a plane. -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersect : Bool

/-- The number of triangular regions formed by a line configuration. -/
def num_triangles (config : LineConfiguration) : ℕ := sorry

/-- Theorem: Given 3000 lines drawn on a plane where no two lines are parallel
    and no three lines intersect at a single point, the number of triangular
    regions formed is at least 2000. -/
theorem min_triangles_in_configuration :
  ∀ (config : LineConfiguration),
    config.num_lines = 3000 →
    config.no_parallel = true →
    config.no_triple_intersect = true →
    num_triangles config ≥ 2000 := by sorry

end NUMINAMATH_CALUDE_min_triangles_in_configuration_l1892_189238


namespace NUMINAMATH_CALUDE_three_color_right_triangle_l1892_189234

/-- A color type representing red, blue, and green --/
inductive Color
  | Red
  | Blue
  | Green

/-- A function that assigns a color to each point with integer coordinates --/
def coloring : ℤ × ℤ → Color := sorry

/-- Predicate to check if three points form a right-angled triangle --/
def is_right_triangle (p1 p2 p3 : ℤ × ℤ) : Prop := sorry

theorem three_color_right_triangle 
  (h1 : ∀ p : ℤ × ℤ, coloring p = Color.Red ∨ coloring p = Color.Blue ∨ coloring p = Color.Green)
  (h2 : ∃ p : ℤ × ℤ, coloring p = Color.Red)
  (h3 : ∃ p : ℤ × ℤ, coloring p = Color.Blue)
  (h4 : ∃ p : ℤ × ℤ, coloring p = Color.Green)
  (h5 : coloring (0, 0) = Color.Red)
  (h6 : coloring (0, 1) = Color.Blue) :
  ∃ p1 p2 p3 : ℤ × ℤ, 
    coloring p1 ≠ coloring p2 ∧ 
    coloring p2 ≠ coloring p3 ∧ 
    coloring p3 ≠ coloring p1 ∧ 
    is_right_triangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_three_color_right_triangle_l1892_189234


namespace NUMINAMATH_CALUDE_lcm_count_theorem_l1892_189215

theorem lcm_count_theorem : 
  ∃ (S : Finset ℕ), 
    S.card = 19 ∧ 
    (∀ k : ℕ, k > 0 → (Nat.lcm (Nat.lcm (9^9) (12^12)) k = 18^18 ↔ k ∈ S)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_count_theorem_l1892_189215


namespace NUMINAMATH_CALUDE_constant_term_implies_a_l1892_189298

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (ax^2 + 1/√x)^5 -/
def constantTerm (a : ℝ) : ℝ := a * (binomial 5 4)

theorem constant_term_implies_a (a : ℝ) :
  constantTerm a = -10 → a = -2 := by sorry

end NUMINAMATH_CALUDE_constant_term_implies_a_l1892_189298


namespace NUMINAMATH_CALUDE_occur_permutations_correct_l1892_189247

/-- The number of unique permutations of the letters in "OCCUR" -/
def occurrPermutations : ℕ := 60

/-- The total number of letters in "OCCUR" -/
def totalLetters : ℕ := 5

/-- The number of times the letter "C" appears in "OCCUR" -/
def cCount : ℕ := 2

/-- Theorem stating that the number of unique permutations of "OCCUR" is correct -/
theorem occur_permutations_correct :
  occurrPermutations = (Nat.factorial totalLetters) / (Nat.factorial cCount) :=
by sorry

end NUMINAMATH_CALUDE_occur_permutations_correct_l1892_189247


namespace NUMINAMATH_CALUDE_N_subset_M_l1892_189258

-- Define the sets M and N
def M : Set ℝ := {x | x < 9}
def N : Set ℝ := {x | x^2 < 9}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l1892_189258


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1892_189224

theorem incorrect_inequality (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1892_189224


namespace NUMINAMATH_CALUDE_car_speed_problem_l1892_189240

/-- Given a car traveling for 2 hours with a speed of 40 km/h in the second hour
    and an average speed of 90 km/h over the entire journey,
    prove that the speed in the first hour must be 140 km/h. -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) (speed_first_hour : ℝ) :
  speed_second_hour = 40 →
  average_speed = 90 →
  average_speed = (speed_first_hour + speed_second_hour) / 2 →
  speed_first_hour = 140 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1892_189240


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l1892_189225

/-- Represents the swimming scenario with two swimmers in a pool. -/
structure SwimmingScenario where
  poolLength : ℝ
  swimmer1Speed : ℝ
  swimmer2Speed : ℝ
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other. -/
def numberOfPasses (scenario : SwimmingScenario) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, the swimmers pass each other 20 times. -/
theorem swimmers_pass_count (scenario : SwimmingScenario) 
  (h1 : scenario.poolLength = 90)
  (h2 : scenario.swimmer1Speed = 3)
  (h3 : scenario.swimmer2Speed = 2)
  (h4 : scenario.totalTime = 12 * 60) : -- 12 minutes in seconds
  numberOfPasses scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l1892_189225


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_plus_b_l1892_189213

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 7]

-- Theorem statement
theorem a_perpendicular_to_a_plus_b :
  (a 0 * (a 0 + b 0) + a 1 * (a 1 + b 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_plus_b_l1892_189213


namespace NUMINAMATH_CALUDE_soccer_league_games_l1892_189260

theorem soccer_league_games (n : ℕ) (h : n = 12) : (n * (n - 1)) / 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1892_189260


namespace NUMINAMATH_CALUDE_soccer_team_penalty_kicks_l1892_189273

/-- Calculates the total number of penalty kicks in a soccer team training exercise. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem: In a soccer team with 24 players, including 4 goalies, 
    where each player shoots once at each goalie, the total number of penalty kicks is 92. -/
theorem soccer_team_penalty_kicks :
  total_penalty_kicks 24 4 = 92 := by
  sorry


end NUMINAMATH_CALUDE_soccer_team_penalty_kicks_l1892_189273


namespace NUMINAMATH_CALUDE_diophantine_equation_equivalence_l1892_189217

theorem diophantine_equation_equivalence (n k : ℕ) (h : n > k) :
  (∃ (x y z : ℕ+), x^n + y^n = z^k) ↔
  (∃ (x y z : ℕ+), x^n + y^n = z^(n-k)) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_equivalence_l1892_189217


namespace NUMINAMATH_CALUDE_fishing_problem_l1892_189293

theorem fishing_problem (blaine_catch : ℕ) (keith_catch : ℕ) : 
  blaine_catch = 5 → 
  keith_catch = 2 * blaine_catch → 
  blaine_catch + keith_catch = 15 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l1892_189293


namespace NUMINAMATH_CALUDE_max_value_of_function_l1892_189276

theorem max_value_of_function (f : ℝ → ℝ) (h : f = λ x => x + Real.sqrt 2 * Real.cos x) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x ∧
  f x = Real.pi / 4 + 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1892_189276


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l1892_189244

/-- Given a function f(x) = ax^2 + 3x - 2, prove that if the slope of the tangent line
    at the point (2, f(2)) is 7, then a = 1. -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 + 3 * x - 2
  (deriv f 2 = 7) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l1892_189244


namespace NUMINAMATH_CALUDE_expression_evaluation_l1892_189207

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1892_189207


namespace NUMINAMATH_CALUDE_tyrone_gives_seven_point_five_verify_final_ratio_l1892_189227

/-- The number of marbles Tyrone gives to Eric to end up with three times as many marbles as Eric, given their initial marble counts. -/
def marbles_given (tyrone_initial : ℚ) (eric_initial : ℚ) : ℚ :=
  let x : ℚ := (tyrone_initial + eric_initial) / 4 - eric_initial
  x

/-- Theorem stating that given the initial conditions, Tyrone gives 7.5 marbles to Eric. -/
theorem tyrone_gives_seven_point_five :
  marbles_given 120 30 = 7.5 := by
  sorry

/-- Verification that after giving marbles, Tyrone has three times as many as Eric. -/
theorem verify_final_ratio 
  (tyrone_initial eric_initial : ℚ) 
  (h : tyrone_initial = 120 ∧ eric_initial = 30) :
  let x := marbles_given tyrone_initial eric_initial
  (tyrone_initial - x) = 3 * (eric_initial + x) := by
  sorry

end NUMINAMATH_CALUDE_tyrone_gives_seven_point_five_verify_final_ratio_l1892_189227


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l1892_189299

/-- The value of a for which the tangents to C₁ and C₂ at their intersection point are perpendicular -/
theorem perpendicular_tangents_intersection (a : ℝ) : 
  a > 0 → 
  ∃ (x y : ℝ), 
    (y = a * x^3 + 1) ∧ 
    (x^2 + y^2 = 5/2) ∧ 
    (∃ (m₁ m₂ : ℝ), 
      (m₁ = 3 * a * x^2) ∧ 
      (m₂ = -x / y) ∧ 
      (m₁ * m₂ = -1)) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l1892_189299


namespace NUMINAMATH_CALUDE_candy_difference_l1892_189297

-- Define the initial variables
def candy_given : ℝ := 6.25
def candy_left : ℝ := 4.75

-- Define the theorem
theorem candy_difference : candy_given - candy_left = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l1892_189297


namespace NUMINAMATH_CALUDE_third_rectangle_area_l1892_189296

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given three rectangles forming a larger rectangle without gaps or overlaps,
    where two rectangles have dimensions 3 cm × 8 cm and 2 cm × 5 cm,
    the area of the third rectangle must be 4 cm². -/
theorem third_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
  r1.length = 3 ∧ r1.width = 8 ∧
  r2.length = 2 ∧ r2.width = 5 →
  r1.area + r2.area + r3.area = (r1.area + r2.area) →
  r3.area = 4 := by
  sorry

#check third_rectangle_area

end NUMINAMATH_CALUDE_third_rectangle_area_l1892_189296


namespace NUMINAMATH_CALUDE_simplify_fraction_1_l1892_189290

theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1 ∧ a ≠ -2) :
  (a^2 - 3*a + 2) / (a^2 + a - 2) = (a - 2) / (a + 2) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_l1892_189290


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_15120_l1892_189270

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_15120 : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_15120_l1892_189270


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1892_189242

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 4 * x + 2 = 0}

theorem unique_solution_condition (a : ℝ) : 
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1892_189242


namespace NUMINAMATH_CALUDE_total_legs_in_collection_l1892_189254

theorem total_legs_in_collection (num_ants num_spiders : ℕ) 
  (ant_legs spider_legs : ℕ) (h1 : num_ants = 12) (h2 : num_spiders = 8) 
  (h3 : ant_legs = 6) (h4 : spider_legs = 8) : 
  num_ants * ant_legs + num_spiders * spider_legs = 136 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_collection_l1892_189254


namespace NUMINAMATH_CALUDE_handshake_count_l1892_189236

theorem handshake_count (n : ℕ) (h : n = 25) : 
  (n * (n - 1) / 2) * 3 = 900 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l1892_189236


namespace NUMINAMATH_CALUDE_math_problem_proof_l1892_189257

theorem math_problem_proof (first_answer : ℕ) (second_answer : ℕ) (third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  first_answer + second_answer + third_answer = 3200 →
  first_answer + second_answer - third_answer = 400 := by
  sorry

end NUMINAMATH_CALUDE_math_problem_proof_l1892_189257


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1892_189271

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1892_189271


namespace NUMINAMATH_CALUDE_distance_calculation_l1892_189277

/-- The distance between Cara's and Don's homes -/
def distance_between_homes : ℝ := 45

/-- Cara's walking speed in km/h -/
def cara_speed : ℝ := 6

/-- Don's walking speed in km/h -/
def don_speed : ℝ := 5

/-- The distance Cara walks before meeting Don in km -/
def cara_distance : ℝ := 30

/-- The time difference between Cara's and Don's start in hours -/
def time_difference : ℝ := 2

theorem distance_calculation :
  distance_between_homes = cara_distance + don_speed * (cara_distance / cara_speed - time_difference) :=
sorry

end NUMINAMATH_CALUDE_distance_calculation_l1892_189277


namespace NUMINAMATH_CALUDE_apple_basket_theorem_l1892_189209

/-- Represents the number of apples in each basket -/
def baskets : List ℕ := [20, 30, 40, 60, 90]

/-- The total number of apples initially -/
def total : ℕ := baskets.sum

/-- Checks if a number is divisible by 3 -/
def divisibleBy3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Checks if removing a basket results in a valid 2:1 ratio -/
def validRemoval (n : ℕ) : Prop :=
  n ∈ baskets ∧ divisibleBy3 (total - n) ∧
  ∃ x y : ℕ, x + y = total - n ∧ x = 2 * y ∧
  (x ∈ baskets.filter (· ≠ n) ∨ y ∈ baskets.filter (· ≠ n))

/-- The main theorem -/
theorem apple_basket_theorem :
  ∀ n : ℕ, validRemoval n → n = 60 ∨ n = 90 := by sorry

end NUMINAMATH_CALUDE_apple_basket_theorem_l1892_189209


namespace NUMINAMATH_CALUDE_third_month_relation_l1892_189261

def freelancer_earnings 
  (first_month : ℕ) 
  (second_month : ℕ) 
  (third_month : ℕ) 
  (total : ℕ) : Prop :=
  first_month = 350 ∧
  second_month = 2 * first_month + 50 ∧
  total = first_month + second_month + third_month ∧
  total = 5500

theorem third_month_relation 
  (first_month second_month third_month total : ℕ) :
  freelancer_earnings first_month second_month third_month total →
  third_month = 4 * (first_month + second_month) :=
by
  sorry

end NUMINAMATH_CALUDE_third_month_relation_l1892_189261


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1892_189291

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1892_189291


namespace NUMINAMATH_CALUDE_prime_power_sum_l1892_189248

theorem prime_power_sum (p : ℕ) (x y z : ℕ) 
  (hp : Prime p) 
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) 
  (heq : x^p + y^p = p^z) : 
  z = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l1892_189248


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_reals_l1892_189220

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def B : Set ℝ := {x | x^2 - 4*x - 5 > 0}

-- Part 1
theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

-- Part 2
theorem range_of_a_when_union_is_reals :
  (∃ a, A a ∪ B = Set.univ) → ∃ a, 1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_range_of_a_when_union_is_reals_l1892_189220


namespace NUMINAMATH_CALUDE_clock_movement_theorem_l1892_189274

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ

/-- Represents a clock -/
structure Clock where
  startTime : Time
  degreeMoved : ℝ
  degreesPer12Hours : ℝ

/-- Calculates the ending time given a clock -/
def endingTime (c : Clock) : Time :=
  sorry

/-- The theorem to prove -/
theorem clock_movement_theorem (c : Clock) : 
  c.startTime = ⟨12, 0⟩ →
  c.degreeMoved = 74.99999999999999 →
  c.degreesPer12Hours = 360 →
  endingTime c = ⟨14, 30⟩ :=
sorry

end NUMINAMATH_CALUDE_clock_movement_theorem_l1892_189274


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1892_189203

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for the number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1892_189203


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l1892_189250

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.12
  let geometry_weight : ℝ := 0.62
  let history_weight : ℝ := 4.25
  let literature_weight : ℝ := 3.8
  let chem_geo_combined : ℝ := chemistry_weight + geometry_weight
  let hist_lit_combined : ℝ := history_weight + literature_weight
  chem_geo_combined - hist_lit_combined = -0.31 :=
by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l1892_189250


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1892_189289

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is the line x - 2y = 0,
    then its eccentricity is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ x - 2*y = 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1892_189289


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l1892_189275

theorem negation_of_exists (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_proposition_l1892_189275


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1892_189219

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 7/49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l1892_189219


namespace NUMINAMATH_CALUDE_value_of_expression_l1892_189272

def smallest_positive_integer : ℕ := 1

def largest_negative_integer : ℤ := -1

def smallest_absolute_rational : ℚ := 0

def rational_at_distance_4 : Set ℚ := {d : ℚ | d = 4 ∨ d = -4}

theorem value_of_expression (a b : ℤ) (c d : ℚ) :
  a = smallest_positive_integer ∧
  b = largest_negative_integer ∧
  c = smallest_absolute_rational ∧
  d ∈ rational_at_distance_4 →
  a - b - c + d = -2 ∨ a - b - c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1892_189272


namespace NUMINAMATH_CALUDE_max_triangle_area_l1892_189269

/-- The maximum area of a triangle ABC with side lengths satisfying the given constraints is 1 -/
theorem max_triangle_area (AB BC CA : ℝ) (h1 : 0 ≤ AB ∧ AB ≤ 1) (h2 : 1 ≤ BC ∧ BC ≤ 2) (h3 : 2 ≤ CA ∧ CA ≤ 3) :
  (∃ (S : ℝ), S = Real.sqrt ((AB + BC + CA) / 2 * ((AB + BC + CA) / 2 - AB) * ((AB + BC + CA) / 2 - BC) * ((AB + BC + CA) / 2 - CA))) →
  (∀ (area : ℝ), area ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1892_189269


namespace NUMINAMATH_CALUDE_camp_wonka_marshmallows_l1892_189204

theorem camp_wonka_marshmallows (total_campers : ℕ) 
  (boys_fraction : ℚ) (girls_fraction : ℚ) 
  (boys_toasting_percentage : ℚ) (girls_toasting_percentage : ℚ) :
  total_campers = 96 →
  boys_fraction = 2/3 →
  girls_fraction = 1/3 →
  boys_toasting_percentage = 1/2 →
  girls_toasting_percentage = 3/4 →
  (boys_fraction * total_campers * boys_toasting_percentage + 
   girls_fraction * total_campers * girls_toasting_percentage : ℚ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_camp_wonka_marshmallows_l1892_189204


namespace NUMINAMATH_CALUDE_ladder_problem_l1892_189221

theorem ladder_problem (ladder_length height base : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12)
  (h3 : ladder_length^2 = height^2 + base^2) : 
  base = 5 := by sorry

end NUMINAMATH_CALUDE_ladder_problem_l1892_189221


namespace NUMINAMATH_CALUDE_largest_n_for_product_4021_l1892_189210

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1 : ℤ) * seq.diff

theorem largest_n_for_product_4021 (a b : ArithmeticSequence)
    (h1 : a.first = 1)
    (h2 : b.first = 1)
    (h3 : a.diff ≤ b.diff)
    (h4 : ∃ n : ℕ, a.nthTerm n * b.nthTerm n = 4021) :
    (∀ m : ℕ, a.nthTerm m * b.nthTerm m = 4021 → m ≤ 11) ∧
    (∃ n : ℕ, n = 11 ∧ a.nthTerm n * b.nthTerm n = 4021) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_product_4021_l1892_189210


namespace NUMINAMATH_CALUDE_geometry_propositions_l1892_189278

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1892_189278


namespace NUMINAMATH_CALUDE_equation_solutions_l1892_189282

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (3 * x₁^2 - 4 * x₁ = 2 * x₁ ∧ x₁ = 0) ∧
                (3 * x₂^2 - 4 * x₂ = 2 * x₂ ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (y₁ * (y₁ + 8) = 16 ∧ y₁ = -4 + 4 * Real.sqrt 2) ∧
                (y₂ * (y₂ + 8) = 16 ∧ y₂ = -4 - 4 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1892_189282


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1892_189292

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem set_intersection_problem : A ∩ B = {8, 10} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1892_189292


namespace NUMINAMATH_CALUDE_football_game_attendance_l1892_189283

/-- Represents the number of adults attending the football game -/
def num_adults : ℕ := sorry

/-- Represents the number of children attending the football game -/
def num_children : ℕ := sorry

/-- The price of an adult ticket in cents -/
def adult_price : ℕ := 60

/-- The price of a child ticket in cents -/
def child_price : ℕ := 25

/-- The total number of attendees -/
def total_attendance : ℕ := 280

/-- The total money collected in cents -/
def total_money : ℕ := 14000

theorem football_game_attendance :
  (num_adults + num_children = total_attendance) ∧
  (num_adults * adult_price + num_children * child_price = total_money) →
  num_adults = 200 := by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l1892_189283


namespace NUMINAMATH_CALUDE_bernie_chocolate_savings_l1892_189263

/-- Calculates the savings over a given number of weeks when buying chocolates at a discounted price --/
def chocolate_savings (chocolates_per_week : ℕ) (regular_price discount_price : ℚ) (weeks : ℕ) : ℚ :=
  (chocolates_per_week * (regular_price - discount_price)) * weeks

/-- The savings over three weeks when buying two chocolates per week at a store with a $2 price instead of a store with a $3 price is equal to $6 --/
theorem bernie_chocolate_savings :
  chocolate_savings 2 3 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_bernie_chocolate_savings_l1892_189263


namespace NUMINAMATH_CALUDE_rope_length_problem_l1892_189249

theorem rope_length_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece / longer_piece = 3 / 4 →
  longer_piece = 20 →
  total_length = shorter_piece + longer_piece →
  total_length = 35 := by
sorry

end NUMINAMATH_CALUDE_rope_length_problem_l1892_189249


namespace NUMINAMATH_CALUDE_circle_center_sum_l1892_189265

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) →  -- Circle equation
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 ≤ (x - a)^2 + (y - b)^2) →  -- Definition of center
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1892_189265


namespace NUMINAMATH_CALUDE_todd_snow_cone_stand_l1892_189201

/-- Todd's snow-cone stand problem -/
theorem todd_snow_cone_stand (borrowed : ℝ) (repay : ℝ) (equipment : ℝ) (ingredients : ℝ) 
  (marketing : ℝ) (snow_cones : ℕ) (price : ℝ) : 
  borrowed = 200 →
  repay = 220 →
  equipment = 100 →
  ingredients = 45 →
  marketing = 30 →
  snow_cones = 350 →
  price = 1.5 →
  snow_cones * price - (equipment + ingredients + marketing) - repay = 130 := by
  sorry

end NUMINAMATH_CALUDE_todd_snow_cone_stand_l1892_189201


namespace NUMINAMATH_CALUDE_symmetric_point_of_A_l1892_189202

def line_equation (x y : ℝ) : Prop := 2*x - 4*y + 9 = 0

def is_symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the given line
  (y₂ - y₁) / (x₂ - x₁) = -1 / (1/2) ∧
  -- The midpoint of the two points lies on the given line
  line_equation ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

theorem symmetric_point_of_A : is_symmetric_point 2 2 1 4 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_A_l1892_189202


namespace NUMINAMATH_CALUDE_middle_term_value_l1892_189226

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence3 where
  a : ℤ  -- First term
  b : ℤ  -- Second term
  c : ℤ  -- Third term
  is_arithmetic : b - a = c - b

/-- The problem statement -/
theorem middle_term_value (seq : ArithmeticSequence3) 
  (h1 : seq.a = 2^3)
  (h2 : seq.c = 2^5) : 
  seq.b = 20 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_value_l1892_189226


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l1892_189237

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l1892_189237


namespace NUMINAMATH_CALUDE_calculator_result_l1892_189295

def special_key (x : ℚ) : ℚ := 1 / (1 - x)

theorem calculator_result :
  (Nat.iterate special_key 100 5 : ℚ) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_calculator_result_l1892_189295


namespace NUMINAMATH_CALUDE_right_triangle_enlargement_l1892_189255

theorem right_triangle_enlargement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) : 
  (5*c)^2 = (5*a)^2 + (5*b)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_enlargement_l1892_189255


namespace NUMINAMATH_CALUDE_operations_to_equality_l1892_189245

def num_operations (a b : ℕ) (subtract_a add_b : ℕ) : ℕ :=
  (a - b) / (subtract_a + add_b)

theorem operations_to_equality : num_operations 365 24 19 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_operations_to_equality_l1892_189245


namespace NUMINAMATH_CALUDE_sandwich_problem_l1892_189229

theorem sandwich_problem (billy_sandwiches katelyn_sandwiches chloe_sandwiches : ℕ) : 
  billy_sandwiches = 49 →
  chloe_sandwiches = katelyn_sandwiches / 4 →
  billy_sandwiches + katelyn_sandwiches + chloe_sandwiches = 169 →
  katelyn_sandwiches > billy_sandwiches →
  katelyn_sandwiches - billy_sandwiches = 47 := by
sorry


end NUMINAMATH_CALUDE_sandwich_problem_l1892_189229


namespace NUMINAMATH_CALUDE_expression_value_l1892_189216

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1892_189216


namespace NUMINAMATH_CALUDE_problem_statement_l1892_189241

/-- Given real numbers x, y, and z satisfying certain conditions, 
    prove that a specific expression equals 13.5 -/
theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -9)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1892_189241


namespace NUMINAMATH_CALUDE_percentage_problem_l1892_189235

theorem percentage_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : 8 = (6 / 100) * a) 
  (h2 : ∃ x, (x / 100) * b = 6) (h3 : b / a = 9 / 2) : 
  ∃ x, (x / 100) * b = 6 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1892_189235


namespace NUMINAMATH_CALUDE_equation_solutions_l1892_189252

theorem equation_solutions : 
  {x : ℝ | (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6)} = {7, -2} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1892_189252


namespace NUMINAMATH_CALUDE_integer_puzzle_l1892_189253

theorem integer_puzzle (x y : ℕ+) (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 := by
  sorry

end NUMINAMATH_CALUDE_integer_puzzle_l1892_189253
