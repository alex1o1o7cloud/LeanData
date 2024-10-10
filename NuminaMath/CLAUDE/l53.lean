import Mathlib

namespace equation_solution_l53_5363

theorem equation_solution : ∃! y : ℚ, y ≠ 2 ∧ (7 * y) / (y - 2) - 4 / (y - 2) = 3 / (y - 2) + 1 := by
  sorry

end equation_solution_l53_5363


namespace expression_simplification_l53_5300

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - 2*a*b + b^2) / (2*a*b) - (2*a*b - b^2) / (3*a*b - 3*a^2) = (a - b)^2 / (2*a*b) := by
  sorry

end expression_simplification_l53_5300


namespace flat_percentage_calculation_l53_5391

/-- The price of each flat -/
def flat_price : ℚ := 675958

/-- The overall gain from the transaction -/
def overall_gain : ℚ := 144 / 100

/-- The percentage of gain or loss on each flat -/
noncomputable def percentage : ℚ := overall_gain / (2 * flat_price) * 100

theorem flat_percentage_calculation :
  ∃ (ε : ℚ), abs (percentage - 1065 / 100000000) < ε ∧ ε > 0 := by
  sorry

end flat_percentage_calculation_l53_5391


namespace max_value_x_1plusx_3minusx_l53_5380

theorem max_value_x_1plusx_3minusx (x : ℝ) (h : x > 0) :
  x * (1 + x) * (3 - x) ≤ (70 + 26 * Real.sqrt 13) / 27 ∧
  ∃ y > 0, y * (1 + y) * (3 - y) = (70 + 26 * Real.sqrt 13) / 27 :=
by sorry

end max_value_x_1plusx_3minusx_l53_5380


namespace sin_sum_to_product_l53_5339

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end sin_sum_to_product_l53_5339


namespace isosceles_right_triangle_area_l53_5319

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
    sorry

#check isosceles_right_triangle_area

end isosceles_right_triangle_area_l53_5319


namespace intern_teacher_arrangements_l53_5394

def num_teachers : ℕ := 5
def num_classes : ℕ := 3

def arrangements (n m : ℕ) : ℕ := sorry

theorem intern_teacher_arrangements :
  let remaining_teachers := num_teachers - 1
  arrangements remaining_teachers num_classes = 50 :=
by sorry

end intern_teacher_arrangements_l53_5394


namespace ali_circles_l53_5342

theorem ali_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ernie_circles : ℕ) : 
  total_boxes = 80 → 
  ali_boxes_per_circle = 8 → 
  ernie_boxes_per_circle = 10 → 
  ernie_circles = 4 → 
  (total_boxes - ernie_boxes_per_circle * ernie_circles) / ali_boxes_per_circle = 5 := by
sorry

end ali_circles_l53_5342


namespace age_difference_l53_5371

/-- Proves that A was half of B's age 10 years ago given the conditions -/
theorem age_difference (a b : ℕ) : 
  (a : ℚ) / b = 3 / 4 →  -- ratio of present ages is 3:4
  a + b = 35 →          -- sum of present ages is 35
  ∃ (y : ℕ), y = 10 ∧ (a - y : ℚ) = (1 / 2) * (b - y) := by
  sorry

end age_difference_l53_5371


namespace matrix_M_property_l53_5366

def matrix_M (x : ℝ × ℝ) : ℝ × ℝ := sorry

theorem matrix_M_property (M : ℝ × ℝ → ℝ × ℝ) 
  (h1 : M (2, -1) = (3, 0))
  (h2 : M (-3, 5) = (-1, -1)) :
  M (5, 1) = (11, -1) := by sorry

end matrix_M_property_l53_5366


namespace beach_seashells_l53_5373

/-- 
Given a person who spends 5 days at the beach and finds 7 seashells each day,
the total number of seashells found during the trip is 35.
-/
theorem beach_seashells : 
  ∀ (days : ℕ) (shells_per_day : ℕ),
  days = 5 → shells_per_day = 7 →
  days * shells_per_day = 35 := by
sorry

end beach_seashells_l53_5373


namespace michael_saved_five_cookies_l53_5327

/-- The number of cookies Michael saved to give Sarah -/
def michaels_cookies (sarahs_initial_cupcakes : ℕ) (sarahs_final_desserts : ℕ) : ℕ :=
  sarahs_final_desserts - (sarahs_initial_cupcakes - sarahs_initial_cupcakes / 3)

theorem michael_saved_five_cookies :
  michaels_cookies 9 11 = 5 :=
by sorry

end michael_saved_five_cookies_l53_5327


namespace daejun_marbles_l53_5362

/-- The number of bags Daejun has -/
def num_bags : ℕ := 20

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 156

/-- The total number of marbles Daejun has -/
def total_marbles : ℕ := num_bags * marbles_per_bag

theorem daejun_marbles : total_marbles = 3120 := by
  sorry

end daejun_marbles_l53_5362


namespace min_value_of_sum_min_value_exists_l53_5392

theorem min_value_of_sum (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∀ x y : ℝ, x > -1 → y > -2 → (x + 1) * (y + 2) = 16 → a + b ≤ x + y :=
sorry

theorem min_value_exists (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∃ x y : ℝ, x > -1 ∧ y > -2 ∧ (x + 1) * (y + 2) = 16 ∧ x + y = 5 :=
sorry

end min_value_of_sum_min_value_exists_l53_5392


namespace image_of_negative_two_three_preimage_of_two_negative_three_l53_5341

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2, 3) = (1, -6) := by sorry

-- Theorem for the pre-image of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end image_of_negative_two_three_preimage_of_two_negative_three_l53_5341


namespace complex_number_in_first_quadrant_l53_5354

theorem complex_number_in_first_quadrant :
  let z : ℂ := 1 / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_in_first_quadrant_l53_5354


namespace min_value_on_interval_l53_5316

def f (x : ℝ) := -x^2 + 4*x - 2

theorem min_value_on_interval :
  ∀ x ∈ Set.Icc 1 4, f x ≥ -2 ∧ ∃ y ∈ Set.Icc 1 4, f y = -2 :=
by sorry

end min_value_on_interval_l53_5316


namespace triangle_area_l53_5306

/-- The area of a triangle with vertices at (2, 2), (2, -3), and (7, 2) is 12.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (2, -3)
  let C : ℝ × ℝ := (7, 2)
  (1/2 : ℝ) * |A.1 - C.1| * |A.2 - B.2| = 12.5 := by sorry

end triangle_area_l53_5306


namespace cos_90_degrees_equals_zero_l53_5382

theorem cos_90_degrees_equals_zero : 
  let cos_def : ℝ → ℝ := λ θ => (Real.cos θ)
  let unit_circle_point : ℝ × ℝ := (0, 1)
  cos_def (π / 2) = 0 := by
  sorry

end cos_90_degrees_equals_zero_l53_5382


namespace sadaf_height_l53_5378

theorem sadaf_height (lily_height : ℝ) (anika_height : ℝ) (sadaf_height : ℝ) 
  (h1 : lily_height = 90)
  (h2 : anika_height = 4/3 * lily_height)
  (h3 : sadaf_height = 5/4 * anika_height) :
  sadaf_height = 150 := by
  sorry

end sadaf_height_l53_5378


namespace extreme_point_of_f_l53_5383

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1)^3 + 2

-- State the theorem
theorem extreme_point_of_f :
  ∃! x : ℝ, ∀ y : ℝ, f y ≥ f x :=
  by sorry

end extreme_point_of_f_l53_5383


namespace lines_do_not_form_triangle_l53_5370

/-- Three lines in a 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → ℝ → Prop
  line3 : ℝ → ℝ → ℝ → Prop

/-- The given three lines -/
def givenLines (m : ℝ) : ThreeLines :=
  { line1 := λ x y => 4 * x + y = 4
  , line2 := λ x y m => m * x + y = 0
  , line3 := λ x y m => 2 * x - 3 * m * y = 4 }

/-- Predicate to check if three lines form a triangle -/
def formsTriangle (lines : ThreeLines) : Prop := sorry

/-- The set of m values for which the lines do not form a triangle -/
def noTriangleValues : Set ℝ := {4, -1/6, -1, 2/3}

/-- Theorem stating the condition for the lines to not form a triangle -/
theorem lines_do_not_form_triangle (m : ℝ) :
  ¬(formsTriangle (givenLines m)) ↔ m ∈ noTriangleValues :=
sorry

end lines_do_not_form_triangle_l53_5370


namespace rectangle_cut_squares_l53_5369

/-- Given a rectangle with length 90 cm and width 42 cm, prove that when cut into the largest possible squares with integer side lengths, the minimum number of squares is 105 and their total perimeter is 2520 cm. -/
theorem rectangle_cut_squares (length width : ℕ) (h1 : length = 90) (h2 : width = 42) :
  let side_length := Nat.gcd length width
  let num_squares := (length / side_length) * (width / side_length)
  let total_perimeter := num_squares * (4 * side_length)
  num_squares = 105 ∧ total_perimeter = 2520 := by
  sorry

end rectangle_cut_squares_l53_5369


namespace partial_fraction_decomposition_l53_5338

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    P = 21/4 ∧ Q = 15 ∧ R = -11/2 ∧
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      5*x + 1 = (x - 4)*(x - 2)^2 * (P/(x - 4) + Q/(x - 2) + R/(x - 2)^2) :=
by sorry

end partial_fraction_decomposition_l53_5338


namespace parabola_latus_rectum_l53_5358

/-- For a parabola y = ax^2 with a > 0, if the length of its latus rectum is 4 units, then a = 1/4 -/
theorem parabola_latus_rectum (a : ℝ) (h1 : a > 0) :
  (1 / a = 4) → a = 1 / 4 := by
  sorry

end parabola_latus_rectum_l53_5358


namespace ratio_of_segments_l53_5321

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end ratio_of_segments_l53_5321


namespace sum_of_squares_130_l53_5368

theorem sum_of_squares_130 (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a^2 + b^2 = 130 → 
  a + b = 16 := by
sorry

end sum_of_squares_130_l53_5368


namespace cube_sum_reciprocal_l53_5389

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 := by
  sorry

end cube_sum_reciprocal_l53_5389


namespace equation_characterizes_triangles_l53_5395

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The equation given in the problem. -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.c^2 - t.a^2) / t.b + (t.b^2 - t.c^2) / t.a = t.b - t.a

/-- A right-angled triangle. -/
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- An isosceles triangle. -/
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- The main theorem. -/
theorem equation_characterizes_triangles (t : Triangle) :
  satisfies_equation t ↔ is_right_angled t ∨ is_isosceles t := by
  sorry

end equation_characterizes_triangles_l53_5395


namespace expected_sedans_is_48_l53_5381

/-- Represents the car dealership's sales plan -/
structure SalesPlan where
  sportsCarRatio : ℕ
  sedanRatio : ℕ
  totalTarget : ℕ
  plannedSportsCars : ℕ

/-- Calculates the number of sedans to be sold based on the sales plan -/
def expectedSedans (plan : SalesPlan) : ℕ :=
  plan.sedanRatio * plan.plannedSportsCars / plan.sportsCarRatio

/-- Theorem stating that the expected number of sedans is 48 given the specified conditions -/
theorem expected_sedans_is_48 (plan : SalesPlan)
  (h1 : plan.sportsCarRatio = 5)
  (h2 : plan.sedanRatio = 8)
  (h3 : plan.totalTarget = 78)
  (h4 : plan.plannedSportsCars = 30)
  (h5 : plan.plannedSportsCars + expectedSedans plan = plan.totalTarget) :
  expectedSedans plan = 48 := by
  sorry

#eval expectedSedans {sportsCarRatio := 5, sedanRatio := 8, totalTarget := 78, plannedSportsCars := 30}

end expected_sedans_is_48_l53_5381


namespace max_product_sides_l53_5310

/-- A convex quadrilateral with side lengths a, b, c, d and diagonal lengths e, f --/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  convex : True  -- Assuming convexity without formal definition
  max_side : max a (max b (max c (max d (max e f)))) = 1

/-- The maximum product of side lengths in a convex quadrilateral with max side length 1 is 2 - √3 --/
theorem max_product_sides (q : ConvexQuadrilateral) : 
  ∃ (m : ℝ), m = q.a * q.b * q.c * q.d ∧ m ≤ 2 - Real.sqrt 3 := by
  sorry

#check max_product_sides

end max_product_sides_l53_5310


namespace complex_expression_simplification_l53_5337

theorem complex_expression_simplification :
  (7 - 3 * Complex.I) - (2 - 5 * Complex.I) - (3 + 2 * Complex.I) = (2 : ℂ) := by
  sorry

end complex_expression_simplification_l53_5337


namespace vegetable_field_area_l53_5377

theorem vegetable_field_area (V W : ℝ) 
  (h1 : (1/2) * V + (1/3) * W = 13)
  (h2 : (1/2) * W + (1/3) * V = 12) : 
  V = 18 := by
sorry

end vegetable_field_area_l53_5377


namespace trailing_zeros_625_factorial_l53_5347

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeros_625_factorial :
  trailingZeros (factorial 625) = 156 := by sorry

end trailing_zeros_625_factorial_l53_5347


namespace stratified_sampling_most_appropriate_l53_5336

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| RandomNumber
| Systematic
| Stratified

/-- Represents a population with two equal-sized subgroups -/
structure Population :=
  (size : ℕ)
  (subgroup1_size : ℕ)
  (subgroup2_size : ℕ)
  (h_equal_size : subgroup1_size = subgroup2_size)
  (h_total_size : subgroup1_size + subgroup2_size = size)

/-- Represents the goal of understanding differences between subgroups -/
def UnderstandDifferences : Prop := True

/-- The most appropriate sampling method for a given population and goal -/
def MostAppropriateSamplingMethod (p : Population) (goal : UnderstandDifferences) : SamplingMethod :=
  SamplingMethod.Stratified

/-- Theorem stating that stratified sampling is the most appropriate method 
    for a population with two equal-sized subgroups when the goal is to 
    understand differences between these subgroups -/
theorem stratified_sampling_most_appropriate 
  (p : Population) (goal : UnderstandDifferences) :
  MostAppropriateSamplingMethod p goal = SamplingMethod.Stratified :=
by
  sorry


end stratified_sampling_most_appropriate_l53_5336


namespace service_center_location_l53_5323

/-- The location of the service center on a highway given the locations of two exits -/
theorem service_center_location 
  (fourth_exit_location : ℝ) 
  (twelfth_exit_location : ℝ) 
  (h1 : fourth_exit_location = 50)
  (h2 : twelfth_exit_location = 190)
  (service_center_location : ℝ) 
  (h3 : service_center_location = fourth_exit_location + (twelfth_exit_location - fourth_exit_location) / 2) :
  service_center_location = 120 := by
sorry

end service_center_location_l53_5323


namespace protein_percentage_of_first_meal_l53_5335

-- Define the constants
def total_weight : ℝ := 280
def mixture_protein_percentage : ℝ := 13
def cornmeal_protein_percentage : ℝ := 7
def first_meal_weight : ℝ := 240
def cornmeal_weight : ℝ := total_weight - first_meal_weight

-- Define the theorem
theorem protein_percentage_of_first_meal :
  let total_protein := total_weight * mixture_protein_percentage / 100
  let cornmeal_protein := cornmeal_weight * cornmeal_protein_percentage / 100
  let first_meal_protein := total_protein - cornmeal_protein
  first_meal_protein / first_meal_weight * 100 = 14 := by
sorry

end protein_percentage_of_first_meal_l53_5335


namespace orchid_seed_weight_scientific_notation_l53_5348

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

theorem orchid_seed_weight_scientific_notation :
  toScientificNotation 0.0000005 = ScientificNotation.mk 5 (-7) (by norm_num) := by sorry

end orchid_seed_weight_scientific_notation_l53_5348


namespace odd_triangle_perimeter_l53_5390

/-- A triangle with two sides of lengths 2 and 3, and the third side being an odd number -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℕ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : Odd side3
  h4 : side3 > 0  -- Ensuring positive length
  h5 : side1 + side2 > side3  -- Triangle inequality
  h6 : side1 + side3 > side2
  h7 : side2 + side3 > side1

/-- The perimeter of an OddTriangle is 8 -/
theorem odd_triangle_perimeter (t : OddTriangle) : t.side1 + t.side2 + t.side3 = 8 :=
by sorry

end odd_triangle_perimeter_l53_5390


namespace parabola_intersection_length_l53_5312

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b ∧ focus ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}}

-- Define the intersection points A and B
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ p ∈ line_through_focus m b}

-- State the theorem
theorem parabola_intersection_length 
  (m b : ℝ) 
  (A B : ℝ × ℝ) 
  (h_A : A ∈ intersection_points m b) 
  (h_B : B ∈ intersection_points m b) 
  (h_midpoint : (A.1 + B.1) / 2 = 3) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end parabola_intersection_length_l53_5312


namespace minimum_amount_spent_on_boxes_l53_5397

/-- The minimum amount spent on boxes for packaging a collection -/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (box_cost : ℝ)
  (total_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : box_cost = 0.5)
  (h5 : total_volume = 2160000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * box_cost = 225 :=
sorry

end minimum_amount_spent_on_boxes_l53_5397


namespace worker_distribution_l53_5320

theorem worker_distribution (total_workers : ℕ) (male_days female_days : ℝ) 
  (h_total : total_workers = 20)
  (h_male_days : male_days = 2)
  (h_female_days : female_days = 3)
  (h_work_rate : ∀ (x y : ℕ), x + y = total_workers → 
    (x : ℝ) / male_days + (y : ℝ) / female_days = 1) :
  ∃ (male_workers female_workers : ℕ),
    male_workers + female_workers = total_workers ∧
    male_workers = 12 ∧
    female_workers = 8 :=
sorry

end worker_distribution_l53_5320


namespace test_questions_count_l53_5353

theorem test_questions_count (total_points : ℕ) (two_point_questions : ℕ) :
  total_points = 100 →
  two_point_questions = 30 →
  ∃ (four_point_questions : ℕ),
    total_points = 2 * two_point_questions + 4 * four_point_questions ∧
    two_point_questions + four_point_questions = 40 :=
by
  sorry

end test_questions_count_l53_5353


namespace x_axis_intercept_l53_5330

/-- The x-axis intercept of the line x + 2y + 1 = 0 is -1. -/
theorem x_axis_intercept :
  ∃ (x : ℝ), x + 2 * 0 + 1 = 0 ∧ x = -1 := by
  sorry

end x_axis_intercept_l53_5330


namespace research_budget_allocation_l53_5334

theorem research_budget_allocation (microphotonics home_electronics gmo industrial_lubricants basic_astrophysics food_additives : ℝ) : 
  microphotonics = 10 →
  home_electronics = 24 →
  gmo = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics / 100 = 50.4 / 360 →
  microphotonics + home_electronics + gmo + industrial_lubricants + basic_astrophysics + food_additives = 100 →
  food_additives = 15 := by
  sorry

end research_budget_allocation_l53_5334


namespace pictures_per_album_l53_5318

/-- Given the number of pictures uploaded from phone and camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

#eval (35 + 5) / 5  -- Expected output: 8

end pictures_per_album_l53_5318


namespace equation_solutions_l53_5396

theorem equation_solutions : 
  let f (r : ℝ) := (r^2 - 6*r + 9) / (r^2 - 9*r + 14)
  let g (r : ℝ) := (r^2 - 4*r - 21) / (r^2 - 2*r - 35)
  ∀ r : ℝ, f r = g r ↔ (r = 3 ∨ r = (-1 + Real.sqrt 69) / 2 ∨ r = (-1 - Real.sqrt 69) / 2) :=
by sorry

end equation_solutions_l53_5396


namespace parabola_chord_sum_constant_l53_5309

/-- Theorem: For a parabola y = x^2, if there exists a constant d such that
    for all chords AB passing through D = (0,d), the sum s = 1/AD^2 + 1/BD^2 is constant,
    then d = 1/2 and s = 4. -/
theorem parabola_chord_sum_constant (d : ℝ) :
  (∃ (s : ℝ), ∀ (A B : ℝ × ℝ),
    A.2 = A.1^2 ∧ B.2 = B.1^2 ∧  -- A and B are on the parabola y = x^2
    (∃ (m : ℝ), A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →  -- AB passes through (0,d)
    1 / ((A.1^2 + (A.2 - d)^2) : ℝ) + 1 / ((B.1^2 + (B.2 - d)^2) : ℝ) = s) →
  d = 1/2 ∧ s = 4 :=
sorry

end parabola_chord_sum_constant_l53_5309


namespace tan_seven_pi_fourths_l53_5328

theorem tan_seven_pi_fourths : Real.tan (7 * π / 4) = -1 := by
  sorry

end tan_seven_pi_fourths_l53_5328


namespace add_3031_minutes_to_initial_equals_final_l53_5385

-- Define a structure for date and time
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

-- Define the function to add minutes to a DateTime
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

-- Define the initial and final DateTimes
def initialDateTime : DateTime :=
  { year := 2020, month := 12, day := 31, hour := 17, minute := 0 }

def finalDateTime : DateTime :=
  { year := 2021, month := 1, day := 2, hour := 19, minute := 31 }

-- Theorem to prove
theorem add_3031_minutes_to_initial_equals_final :
  addMinutes initialDateTime 3031 = finalDateTime :=
sorry

end add_3031_minutes_to_initial_equals_final_l53_5385


namespace jumping_contest_l53_5372

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end jumping_contest_l53_5372


namespace correct_num_kids_l53_5326

/-- The number of kids in a group that can wash whiteboards -/
def num_kids : ℕ := 4

/-- The number of whiteboards the group can wash in 20 minutes -/
def group_whiteboards : ℕ := 3

/-- The time in minutes it takes the group to wash their whiteboards -/
def group_time : ℕ := 20

/-- The number of whiteboards one kid can wash -/
def one_kid_whiteboards : ℕ := 6

/-- The time in minutes it takes one kid to wash their whiteboards -/
def one_kid_time : ℕ := 160

/-- Theorem stating that the number of kids in the group is correct -/
theorem correct_num_kids :
  num_kids * (group_whiteboards * one_kid_time) = (one_kid_whiteboards * group_time) :=
by sorry

end correct_num_kids_l53_5326


namespace base8_653_equals_base10_427_l53_5374

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

theorem base8_653_equals_base10_427 :
  base8ToBase10 653 = 427 := by
  sorry

end base8_653_equals_base10_427_l53_5374


namespace attendance_difference_l53_5324

/-- Proves that the difference in student attendance between the second and first day is 40 --/
theorem attendance_difference (total_students : ℕ) (total_absent : ℕ) 
  (absent_day1 absent_day2 absent_day3 : ℕ) :
  total_students = 280 →
  total_absent = 240 →
  absent_day1 + absent_day2 + absent_day3 = total_absent →
  absent_day2 = 2 * absent_day3 →
  absent_day3 = total_students / 7 →
  absent_day2 < absent_day1 →
  (total_students - absent_day2) - (total_students - absent_day1) = 40 := by
  sorry

end attendance_difference_l53_5324


namespace blocks_count_l53_5325

/-- The number of blocks in Jacob's toy bin --/
def total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Theorem: Given the conditions, the total number of blocks is 75 --/
theorem blocks_count :
  let red : ℕ := 18
  let yellow : ℕ := red + 7
  let blue : ℕ := red + 14
  total_blocks red yellow blue = 75 := by sorry

end blocks_count_l53_5325


namespace probability_science_second_given_arts_first_l53_5317

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of science questions
def science_questions : ℕ := 3

-- Define the number of arts questions
def arts_questions : ℕ := 2

-- Define the probability of drawing an arts question in the first draw
def prob_arts_first : ℚ := arts_questions / total_questions

-- Define the probability of drawing a science question in the second draw given an arts question was drawn first
def prob_science_second_given_arts_first : ℚ := science_questions / (total_questions - 1)

-- Theorem statement
theorem probability_science_second_given_arts_first :
  prob_science_second_given_arts_first = 3/4 :=
sorry

end probability_science_second_given_arts_first_l53_5317


namespace paint_difference_l53_5340

theorem paint_difference (R r : ℝ) (h : R > 0) (h' : r > 0) : 
  (4 / 3 * Real.pi * R^3 - 4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * r^3) = 14.625 →
  (4 * Real.pi * R^2 - 4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 0.84 :=
by sorry

end paint_difference_l53_5340


namespace arithmetic_sequence_formulas_l53_5361

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formulas
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = 20)
  (h_diff : a 11 - a 8 = 18) :
  (∃ (an : ℕ → ℝ), ∀ n, an n = 6 * n - 14 ∧ a n = an n) ∧
  (∃ (bn : ℕ → ℝ), ∀ n, bn n = 2 * n - 10 ∧
    (∀ k, ∃ m, a m = bn (3*k - 2) ∧ a (m+1) = bn (3*k + 1))) :=
by sorry

end arithmetic_sequence_formulas_l53_5361


namespace team_selection_count_l53_5311

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 10

-- Define the team size and minimum number of girls required
def team_size : ℕ := 6
def min_girls : ℕ := 3

-- Define the function to calculate the number of ways to select the team
def select_team : ℕ := 
  (Nat.choose num_girls 3 * Nat.choose num_boys 3) +
  (Nat.choose num_girls 4 * Nat.choose num_boys 2) +
  (Nat.choose num_girls 5 * Nat.choose num_boys 1) +
  (Nat.choose num_girls 6)

-- Theorem statement
theorem team_selection_count : select_team = 4770 := by
  sorry

end team_selection_count_l53_5311


namespace imaginary_part_of_complex_l53_5360

theorem imaginary_part_of_complex (z : ℂ) (h : z = 3 - 4 * I) : z.im = -4 := by
  sorry

end imaginary_part_of_complex_l53_5360


namespace gcd_13m_plus_4_7m_plus_2_max_2_l53_5346

theorem gcd_13m_plus_4_7m_plus_2_max_2 :
  (∀ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end gcd_13m_plus_4_7m_plus_2_max_2_l53_5346


namespace expression_evaluation_l53_5343

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c + c*(c+1)^c)^c = 5750939763536 := by
  sorry

end expression_evaluation_l53_5343


namespace smallest_value_in_range_l53_5322

theorem smallest_value_in_range (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y := by
  sorry

#check smallest_value_in_range

end smallest_value_in_range_l53_5322


namespace train_passing_time_train_passing_man_time_l53_5365

/-- Time taken for a train to pass a stationary point -/
theorem train_passing_time (platform_length : Real) (platform_passing_time : Real) (train_speed_kmh : Real) : Real :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let train_length := platform_passing_time * train_speed_ms - platform_length
  train_length / train_speed_ms

/-- Proof that a train passing a 30.0024-meter platform in 22 seconds at 54 km/hr takes approximately 20 seconds to pass a stationary point -/
theorem train_passing_man_time : 
  abs (train_passing_time 30.0024 22 54 - 20) < 0.01 := by
  sorry

end train_passing_time_train_passing_man_time_l53_5365


namespace tetrahedron_volume_from_midpoint_distances_l53_5301

/-- The volume of a regular tetrahedron, given specific midpoint distances -/
theorem tetrahedron_volume_from_midpoint_distances :
  ∀ (midpoint_to_face midpoint_to_edge : ℝ),
    midpoint_to_face = 2 →
    midpoint_to_edge = Real.sqrt 5 →
    ∃ (volume : ℝ), volume = 27 * Real.sqrt 3 := by
  sorry

end tetrahedron_volume_from_midpoint_distances_l53_5301


namespace total_pokemon_cards_l53_5308

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of dozens of cards each person has -/
def dozens_per_person : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem: The total number of Pokemon cards owned by 4 people is 432,
    given that each person has 9 dozen cards and one dozen equals 12 items. -/
theorem total_pokemon_cards :
  num_people * dozens_per_person * items_per_dozen = 432 := by
  sorry

end total_pokemon_cards_l53_5308


namespace negation_equivalence_l53_5364

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by
  sorry

end negation_equivalence_l53_5364


namespace parabola_equilateral_triangle_p_value_l53_5305

/-- Parabola defined by x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle defined by center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: For a parabola C: x^2 = 2py (p > 0), if there exists a point A on C
    such that A is equidistant from O(0,0) and M(0,9), and triangle ABO is equilateral
    (where B is another point on the circle with center M and radius |OA|),
    then p = 3/4 -/
theorem parabola_equilateral_triangle_p_value
  (C : Parabola)
  (A : Point)
  (h_A_on_C : A.x^2 = 2 * C.p * A.y)
  (h_A_equidistant : A.x^2 + A.y^2 = A.x^2 + (A.y - 9)^2)
  (h_ABO_equilateral : ∃ B : Point, B.x^2 + (B.y - 9)^2 = A.x^2 + A.y^2 ∧
                       A.x^2 + A.y^2 = (A.x - B.x)^2 + (A.y - B.y)^2) :
  C.p = 3/4 := by
  sorry

end parabola_equilateral_triangle_p_value_l53_5305


namespace binomial_expansion_properties_l53_5350

/-- Coefficient of the r-th term in the binomial expansion of (x + 1/(2√x))^n -/
def coeff (n : ℕ) (r : ℕ) : ℚ :=
  (1 / 2^r) * (n.choose r)

/-- The expansion of (x + 1/(2√x))^n has coefficients forming an arithmetic sequence
    for the first three terms -/
def arithmetic_sequence (n : ℕ) : Prop :=
  coeff n 0 - coeff n 1 = coeff n 1 - coeff n 2

/-- The r-th term has the maximum coefficient in the expansion -/
def max_coeff (n : ℕ) (r : ℕ) : Prop :=
  ∀ k, k ≠ r → coeff n r ≥ coeff n k

theorem binomial_expansion_properties :
  ∃ n : ℕ,
    arithmetic_sequence n ∧
    max_coeff n 2 ∧
    max_coeff n 3 ∧
    ∀ r, r ≠ 2 ∧ r ≠ 3 → ¬(max_coeff n r) :=
by sorry

end binomial_expansion_properties_l53_5350


namespace inequality_implies_b_leq_c_l53_5351

theorem inequality_implies_b_leq_c
  (a b c x y : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (h : a * x + b * y ≤ b * x + c * y ∧ b * x + c * y ≤ c * x + a * y) :
  b ≤ c :=
by sorry

end inequality_implies_b_leq_c_l53_5351


namespace five_valid_configurations_l53_5344

/-- Represents a square in the figure -/
structure Square :=
  (label : Char)

/-- Represents the L-shaped figure -/
structure LShape :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 4)

/-- Represents the set of additional squares -/
structure AdditionalSquares :=
  (squares : Finset Square)
  (size : Nat)
  (h_size : size = 8)

/-- Represents a configuration formed by adding one square to the L-shape -/
structure Configuration :=
  (base : LShape)
  (added : Square)

/-- Predicate to determine if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that exactly 5 configurations can be folded into a topless cubical box -/
theorem five_valid_configurations
  (l : LShape)
  (extras : AdditionalSquares) :
  ∃! (validConfigs : Finset Configuration),
    validConfigs.card = 5 ∧
    ∀ (config : Configuration),
      config ∈ validConfigs ↔
        (config.base = l ∧
         config.added ∈ extras.squares ∧
         canFoldIntoCube config) :=
sorry

end five_valid_configurations_l53_5344


namespace integer_solutions_of_equation_l53_5315

theorem integer_solutions_of_equation :
  let S : Set (ℤ × ℤ) := {(x, y) | x * y - 2 * x - 2 * y + 7 = 0}
  S = {(5, 1), (-1, 3), (3, -1), (1, 5)} := by
  sorry

end integer_solutions_of_equation_l53_5315


namespace angle_on_bisector_l53_5302

-- Define the set of integers
variable (k : ℤ)

-- Define the angle in degrees
def angle (k : ℤ) : ℝ := k * 180 + 135

-- Define the property of being on the bisector of the second or fourth quadrant
def on_bisector_2nd_or_4th (θ : ℝ) : Prop :=
  ∃ n : ℤ, θ = 135 + n * 360 ∨ θ = 315 + n * 360

-- Theorem statement
theorem angle_on_bisector :
  ∀ θ : ℝ, on_bisector_2nd_or_4th θ ↔ ∃ k : ℤ, θ = angle k :=
sorry

end angle_on_bisector_l53_5302


namespace squareable_numbers_l53_5356

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val + 1 + i.val = k^2

theorem squareable_numbers : 
  (¬ isSquareable 7) ∧ 
  (isSquareable 9) ∧ 
  (¬ isSquareable 11) ∧ 
  (isSquareable 15) :=
sorry

end squareable_numbers_l53_5356


namespace dogs_and_bunnies_total_l53_5387

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- Defines the conditions of the pet shop problem -/
def pet_shop_problem (shop : PetShop) : Prop :=
  shop.dogs = 51 ∧
  shop.dogs * 5 = shop.cats * 3 ∧
  shop.dogs * 9 = shop.bunnies * 3

/-- Theorem stating the total number of dogs and bunnies in the pet shop -/
theorem dogs_and_bunnies_total (shop : PetShop) :
  pet_shop_problem shop → shop.dogs + shop.bunnies = 204 := by
  sorry


end dogs_and_bunnies_total_l53_5387


namespace percentage_calculation_l53_5376

theorem percentage_calculation (n : ℝ) : n = 4000 → (0.15 * (0.30 * (0.50 * n))) = 90 := by
  sorry

end percentage_calculation_l53_5376


namespace conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l53_5313

/-- The equation y^4 - 9x^4 = 3y^2 - 1 represents two conic sections -/
theorem conic_sections_decomposition (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔
  ((y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2) ∨ (y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2))) :=
by sorry

/-- The first equation represents an ellipse -/
def is_ellipse (x y : ℝ) : Prop :=
  y^2 - 3/2 = 3*x^2 + Real.sqrt 5/2

/-- The second equation represents a hyperbola -/
def is_hyperbola (x y : ℝ) : Prop :=
  y^2 - 3/2 = -(3*x^2 + Real.sqrt 5/2)

/-- The original equation decomposes into an ellipse and a hyperbola -/
theorem decomposition_into_ellipse_and_hyperbola (x y : ℝ) :
  y^4 - 9*x^4 = 3*y^2 - 1 ↔ (is_ellipse x y ∨ is_hyperbola x y) :=
by sorry

end conic_sections_decomposition_decomposition_into_ellipse_and_hyperbola_l53_5313


namespace tangent_line_at_x_1_l53_5331

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 8*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -2*x + 8

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 6*x + 1 :=
by sorry

end tangent_line_at_x_1_l53_5331


namespace sequence_properties_l53_5388

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sequence_properties
  (a : ℕ → ℕ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (b : ℕ → ℕ)
  (h_b : ∀ n, b n = a (a n))
  (c : ℕ → ℕ)
  (h_c : ∀ n, c n = a (a (n + 1)))
  (h_b_value : ∀ n, b n = 3 * n)
  (h_c_arithmetic : is_arithmetic_sequence c ∧ ∀ n, c (n + 1) = c n + 1) :
  a 1 = 2 ∧ c 1 = 6 ∧ is_arithmetic_sequence a :=
by sorry

end sequence_properties_l53_5388


namespace nonagon_diagonal_count_l53_5304

/-- The number of diagonals in a regular nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

theorem nonagon_diagonal_count :
  nonagon_diagonals = (nonagon_vertices.choose 2) - nonagon_sides := by
  sorry

end nonagon_diagonal_count_l53_5304


namespace base_difference_calculation_l53_5345

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem stating the result of the calculation -/
theorem base_difference_calculation :
  base6ToBase10 52143 - base7ToBase10 4310 = 5449 := by sorry

end base_difference_calculation_l53_5345


namespace polynomial_coefficient_sum_l53_5307

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6 - x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 28 := by
sorry

end polynomial_coefficient_sum_l53_5307


namespace max_side_length_of_special_triangle_l53_5355

theorem max_side_length_of_special_triangle :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 30 →
  a < b + c ∧ b < a + c ∧ c < a + b →
  ∀ x : ℕ, x ≤ a ∧ x ≤ b ∧ x ≤ c →
  x ≤ 14 :=
by sorry

end max_side_length_of_special_triangle_l53_5355


namespace fourth_term_of_geometric_progression_l53_5379

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_geometric_progression :
  ∀ (a r : ℝ),
  (geometric_progression a r 1 = 6^(1/2)) →
  (geometric_progression a r 2 = 6^(1/6)) →
  (geometric_progression a r 3 = 6^(1/12)) →
  (geometric_progression a r 4 = 6^0) :=
by sorry

end fourth_term_of_geometric_progression_l53_5379


namespace equilateral_triangle_sum_product_l53_5398

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24,
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768 -/
theorem equilateral_triangle_sum_product (a b c : ℂ) : 
  (∃ (ω : ℂ), ω ^ 3 = 1 ∧ ω ≠ 1 ∧ c - a = (b - a) * ω) →
  Complex.abs (b - a) = 24 →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end equilateral_triangle_sum_product_l53_5398


namespace inequality_range_l53_5375

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x - Real.log x - a > 0) ↔ a < 1 := by
  sorry

end inequality_range_l53_5375


namespace polynomial_positive_reals_l53_5359

def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_reals :
  (∀ x y : ℝ, P x y > 0) ∧
  (∀ c : ℝ, c > 0 → ∃ x y : ℝ, P x y = c) :=
by sorry

end polynomial_positive_reals_l53_5359


namespace cos_alpha_for_point_l53_5384

/-- If the terminal side of angle α passes through the point (-1, 6), then cos α = -√37/37 -/
theorem cos_alpha_for_point (α : Real) : 
  (∃ (t : Real), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 6) →
  Real.cos α = -Real.sqrt 37 / 37 := by
sorry

end cos_alpha_for_point_l53_5384


namespace fraction_inequality_l53_5332

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c / a > d / b :=
by sorry

end fraction_inequality_l53_5332


namespace distance_traveled_l53_5329

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Given a speed of 100 km/hr and a time of 5 hours, the distance traveled is 500 km -/
theorem distance_traveled (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  distance speed time = 500 := by
  sorry

end distance_traveled_l53_5329


namespace symmetric_point_xOy_l53_5367

def xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}

def symmetric_point (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  (p.1, p.2.1, -p.2.2)

theorem symmetric_point_xOy : 
  symmetric_point (2, 3, 4) xOy_plane = (2, 3, -4) := by sorry

end symmetric_point_xOy_l53_5367


namespace inequality_proof_l53_5349

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 ↔ 
  1/3 < x ∧ x < 14/3 :=
by sorry

end inequality_proof_l53_5349


namespace wood_carvings_per_shelf_example_l53_5357

/-- Given a total number of wood carvings and a number of shelves,
    calculate the number of wood carvings per shelf. -/
def woodCarvingsPerShelf (totalCarvings : ℕ) (numShelves : ℕ) : ℕ :=
  totalCarvings / numShelves

/-- Theorem stating that with 56 total wood carvings and 7 shelves,
    each shelf contains 8 wood carvings. -/
theorem wood_carvings_per_shelf_example :
  woodCarvingsPerShelf 56 7 = 8 := by
  sorry

end wood_carvings_per_shelf_example_l53_5357


namespace a_6_equals_25_l53_5352

/-- An increasing geometric sequence -/
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The roots of x^2 - 6x + 5 = 0 -/
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 6*x + 5 = 0

/-- Theorem: For an increasing geometric sequence where a_2 and a_4 are roots of x^2 - 6x + 5 = 0, a_6 = 25 -/
theorem a_6_equals_25 (a : ℕ → ℝ) 
  (h1 : is_increasing_geometric_sequence a)
  (h2 : is_root_of_equation (a 2))
  (h3 : is_root_of_equation (a 4)) :
  a 6 = 25 :=
sorry

end a_6_equals_25_l53_5352


namespace non_officers_count_l53_5399

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Average salary of all employees in the office -/
def avg_salary_all : ℕ := 120

/-- Average salary of officers -/
def avg_salary_officers : ℕ := 430

/-- Average salary of non-officers -/
def avg_salary_non_officers : ℕ := 110

/-- Number of officers -/
def num_officers : ℕ := 15

/-- Theorem stating that the number of non-officers is 465 -/
theorem non_officers_count : num_non_officers = 465 := by
  sorry

end non_officers_count_l53_5399


namespace quarters_collected_per_month_l53_5393

/-- Represents the number of quarters Phil collected each month during the second year -/
def quarters_per_month : ℕ := sorry

/-- The initial number of quarters Phil had -/
def initial_quarters : ℕ := 50

/-- The number of quarters Phil had after doubling his initial collection -/
def after_doubling : ℕ := 2 * initial_quarters

/-- The number of quarters Phil collected in the third year -/
def third_year_quarters : ℕ := 4

/-- The number of quarters Phil had before losing some -/
def before_loss : ℕ := 140

/-- The number of quarters Phil had after losing some -/
def after_loss : ℕ := 105

/-- Theorem stating that the number of quarters collected each month in the second year is 3 -/
theorem quarters_collected_per_month : 
  quarters_per_month = 3 ∧
  after_doubling + 12 * quarters_per_month + third_year_quarters = before_loss ∧
  before_loss * 3 = after_loss * 4 := by sorry

end quarters_collected_per_month_l53_5393


namespace min_A_over_B_l53_5386

theorem min_A_over_B (x A B : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B + 3) :
  A / B ≥ 6 + 2 * Real.sqrt 11 ∧
  (A / B = 6 + 2 * Real.sqrt 11 ↔ B = Real.sqrt 11) :=
sorry

end min_A_over_B_l53_5386


namespace max_quarters_and_dimes_l53_5303

theorem max_quarters_and_dimes (total : ℚ) (h_total : total = 425/100) :
  ∃ (quarters dimes pennies : ℕ),
    quarters = dimes ∧
    quarters * (25 : ℚ)/100 + dimes * (10 : ℚ)/100 + pennies * (1 : ℚ)/100 = total ∧
    ∀ q d p : ℕ, q = d →
      q * (25 : ℚ)/100 + d * (10 : ℚ)/100 + p * (1 : ℚ)/100 = total →
      q ≤ quarters :=
by sorry

end max_quarters_and_dimes_l53_5303


namespace pythagorean_triple_properties_l53_5333

theorem pythagorean_triple_properties (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (Even a ∨ Even b) ∧
  (3 ∣ a ∨ 3 ∣ b) ∧
  (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) := by
  sorry

end pythagorean_triple_properties_l53_5333


namespace same_walking_speed_l53_5314

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 13*x - 48

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 - 5*x - 84

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

theorem same_walking_speed : 
  ∃ x : ℝ, x > 0 ∧ 
    jack_speed x = jill_distance x / jill_time x ∧ 
    jack_speed x = 6 := by
  sorry

end same_walking_speed_l53_5314
