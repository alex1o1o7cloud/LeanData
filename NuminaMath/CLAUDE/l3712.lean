import Mathlib

namespace NUMINAMATH_CALUDE_bakers_cakes_l3712_371263

/-- Baker's cake problem -/
theorem bakers_cakes (total_cakes : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) : 
  total_cakes = 217 → cakes_left = 72 → cakes_sold = total_cakes - cakes_left → cakes_sold = 145 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3712_371263


namespace NUMINAMATH_CALUDE_school_growth_difference_l3712_371223

theorem school_growth_difference
  (total_last_year : ℕ)
  (school_yy_last_year : ℕ)
  (xx_growth_rate : ℚ)
  (yy_growth_rate : ℚ)
  (h1 : total_last_year = 4000)
  (h2 : school_yy_last_year = 2400)
  (h3 : xx_growth_rate = 7 / 100)
  (h4 : yy_growth_rate = 3 / 100) :
  let school_xx_last_year := total_last_year - school_yy_last_year
  let xx_growth := (school_xx_last_year : ℚ) * xx_growth_rate
  let yy_growth := (school_yy_last_year : ℚ) * yy_growth_rate
  ⌊xx_growth - yy_growth⌋ = 40 := by
  sorry

end NUMINAMATH_CALUDE_school_growth_difference_l3712_371223


namespace NUMINAMATH_CALUDE_original_number_l3712_371248

theorem original_number : ∃ x : ℕ, x - (x / 3) = 36 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3712_371248


namespace NUMINAMATH_CALUDE_bruce_fruit_purchase_l3712_371215

/-- Calculates the total amount paid for fruits given their quantities and rates. -/
def totalAmountPaid (grapeQuantity mangoQuantity grapeRate mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 1055 to the shopkeeper for his fruit purchase. -/
theorem bruce_fruit_purchase : totalAmountPaid 8 9 70 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_bruce_fruit_purchase_l3712_371215


namespace NUMINAMATH_CALUDE_sequence_a_equals_fibonacci_6n_l3712_371243

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (18 * sequence_a n + 8 * (Nat.sqrt (5 * (sequence_a n)^2 - 4))) / 2

theorem sequence_a_equals_fibonacci_6n :
  ∀ n : ℕ, sequence_a n = fibonacci (6 * n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_equals_fibonacci_6n_l3712_371243


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l3712_371286

theorem solution_exists_in_interval : ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x + x = 2 := by
  sorry


end NUMINAMATH_CALUDE_solution_exists_in_interval_l3712_371286


namespace NUMINAMATH_CALUDE_nancy_vacation_pictures_l3712_371265

/-- The number of pictures Nancy took at the zoo -/
def zoo_pictures : ℕ := 49

/-- The number of pictures Nancy took at the museum -/
def museum_pictures : ℕ := 8

/-- The number of pictures Nancy deleted -/
def deleted_pictures : ℕ := 38

/-- The total number of pictures Nancy took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Nancy has after deleting some -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem nancy_vacation_pictures : remaining_pictures = 19 := by
  sorry

end NUMINAMATH_CALUDE_nancy_vacation_pictures_l3712_371265


namespace NUMINAMATH_CALUDE_ada_paul_test_scores_l3712_371200

/-- Ada and Paul's test scores problem -/
theorem ada_paul_test_scores 
  (a1 a2 a3 p1 p2 p3 : ℤ) 
  (h1 : a1 = p1 + 10)
  (h2 : a2 = p2 + 4)
  (h3 : (p1 + p2 + p3) / 3 = (a1 + a2 + a3) / 3 + 4) :
  p3 - a3 = 26 := by
sorry

end NUMINAMATH_CALUDE_ada_paul_test_scores_l3712_371200


namespace NUMINAMATH_CALUDE_last_three_digits_of_11_pow_30_l3712_371268

theorem last_three_digits_of_11_pow_30 : 11^30 ≡ 801 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_11_pow_30_l3712_371268


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l3712_371211

theorem paper_folding_thickness (initial_thickness : ℝ) (num_folds : ℕ) (floor_height : ℝ) :
  initial_thickness = 0.1 →
  num_folds = 20 →
  floor_height = 3 →
  ⌊(2^num_folds * initial_thickness / 1000) / floor_height⌋ = 35 :=
by sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l3712_371211


namespace NUMINAMATH_CALUDE_mari_buttons_l3712_371228

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 4 + 5 * kendra_buttons →
  mari_buttons = 64 := by
sorry

end NUMINAMATH_CALUDE_mari_buttons_l3712_371228


namespace NUMINAMATH_CALUDE_conic_section_union_l3712_371245

/-- The equation y^4 - 6x^4 = 3y^2 - 2 represents the union of a hyperbola and an ellipse -/
theorem conic_section_union (x y : ℝ) : 
  (y^4 - 6*x^4 = 3*y^2 - 2) ↔ 
  ((y^2 - 3*x^2 = 2 ∨ y^2 - 2*x^2 = 1) ∨ (y^2 + 3*x^2 = 2 ∨ y^2 + 2*x^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_conic_section_union_l3712_371245


namespace NUMINAMATH_CALUDE_pearl_distribution_l3712_371293

theorem pearl_distribution (n : ℕ) : 
  (∀ m : ℕ, m > n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) → 
  n % 8 = 6 → 
  n % 7 = 5 → 
  n % 9 = 0 → 
  n = 54 := by
sorry

end NUMINAMATH_CALUDE_pearl_distribution_l3712_371293


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3712_371287

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sixth term of a geometric sequence satisfying given conditions. -/
theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : IsGeometric a)
  (h_sum : a 1 + a 2 = -1)
  (h_diff : a 1 - a 3 = -3) :
  a 6 = -32 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3712_371287


namespace NUMINAMATH_CALUDE_journey_time_calculation_l3712_371298

/-- Given a constant speed, if a 200-mile journey takes 5 hours, 
    then a 120-mile journey will take 3 hours. -/
theorem journey_time_calculation (speed : ℝ) 
  (h1 : speed > 0)
  (h2 : 200 = speed * 5) : 
  120 = speed * 3 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l3712_371298


namespace NUMINAMATH_CALUDE_average_grade_year_before_l3712_371217

/-- Calculates the average grade for the year before last given the following conditions:
  * The student took 6 courses last year with an average grade of 100 points
  * The student took 5 courses the year before
  * The average grade for the entire two-year period was 86 points
-/
theorem average_grade_year_before (courses_last_year : Nat) (avg_grade_last_year : ℝ)
  (courses_year_before : Nat) (avg_grade_two_years : ℝ) :
  courses_last_year = 6 →
  avg_grade_last_year = 100 →
  courses_year_before = 5 →
  avg_grade_two_years = 86 →
  (courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) /
    (courses_year_before + courses_last_year) = avg_grade_two_years →
  avg_grade_year_before = 69.2 :=
by
  sorry

#check average_grade_year_before

end NUMINAMATH_CALUDE_average_grade_year_before_l3712_371217


namespace NUMINAMATH_CALUDE_sin_2y_plus_x_l3712_371230

theorem sin_2y_plus_x (x y : Real) 
  (h1 : Real.sin x = 1/3) 
  (h2 : Real.sin (x + y) = 1) : 
  Real.sin (2*y + x) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sin_2y_plus_x_l3712_371230


namespace NUMINAMATH_CALUDE_chloe_picked_42_carrots_l3712_371258

/-- Represents the number of carrots Chloe picked on the second day -/
def carrots_picked_next_day (initial_carrots : ℕ) (carrots_thrown : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - carrots_thrown)

/-- Theorem stating that Chloe picked 42 carrots the next day -/
theorem chloe_picked_42_carrots : 
  carrots_picked_next_day 48 45 45 = 42 := by
  sorry

#eval carrots_picked_next_day 48 45 45

end NUMINAMATH_CALUDE_chloe_picked_42_carrots_l3712_371258


namespace NUMINAMATH_CALUDE_smallest_total_score_l3712_371292

theorem smallest_total_score : 
  ∃ (T : ℕ), T > 0 ∧ 
  (∃ (n m : ℕ), 2 * n + 5 * m = T ∧ (n ≥ m + 3 ∨ m ≥ n + 3)) ∧ 
  (∀ (S : ℕ), S > 0 → S < T → 
    ¬(∃ (n m : ℕ), 2 * n + 5 * m = S ∧ (n ≥ m + 3 ∨ m ≥ n + 3))) ∧
  T = 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_total_score_l3712_371292


namespace NUMINAMATH_CALUDE_geneticallyModifiedMicroorganismsAllocation_l3712_371278

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysics : ℝ
  geneticallyModifiedMicroorganisms : ℝ

/-- The total budget percentage --/
def totalBudgetPercentage : ℝ := 100

/-- The total degrees in a circle --/
def totalDegrees : ℝ := 360

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem geneticallyModifiedMicroorganismsAllocation (budget : BudgetAllocation) : 
  budget.microphotonics = 12 ∧ 
  budget.homeElectronics = 24 ∧ 
  budget.foodAdditives = 15 ∧ 
  budget.industrialLubricants = 8 ∧ 
  budget.basicAstrophysics * (totalBudgetPercentage / totalDegrees) = 12 ∧
  budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
    budget.industrialLubricants + budget.basicAstrophysics + 
    budget.geneticallyModifiedMicroorganisms = totalBudgetPercentage →
  budget.geneticallyModifiedMicroorganisms = 29 := by
  sorry


end NUMINAMATH_CALUDE_geneticallyModifiedMicroorganismsAllocation_l3712_371278


namespace NUMINAMATH_CALUDE_cube_intersection_probability_l3712_371276

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  vertices : Finset (Fin 8)
  faces : Finset (Finset (Fin 4))
  vertex_count : vertices.card = 8
  face_count : faces.card = 6

/-- A function that determines if three vertices form a plane intersecting the cube's interior -/
def plane_intersects_interior (c : Cube) (v1 v2 v3 : Fin 8) : Prop :=
  sorry

/-- The probability of three randomly chosen distinct vertices forming a plane
    that intersects the interior of the cube -/
def intersection_probability (c : Cube) : ℚ :=
  sorry

theorem cube_intersection_probability (c : Cube) :
  intersection_probability c = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_cube_intersection_probability_l3712_371276


namespace NUMINAMATH_CALUDE_andrew_donuts_problem_l3712_371222

theorem andrew_donuts_problem (monday tuesday wednesday : ℕ) : 
  tuesday = monday / 2 →
  wednesday = 4 * monday →
  monday + tuesday + wednesday = 49 →
  monday = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_andrew_donuts_problem_l3712_371222


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3712_371247

theorem triangle_perimeter_bound : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  (∃ n : ℕ, n = 57 ∧ ∀ m : ℕ, m > (s + 7 + 21) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3712_371247


namespace NUMINAMATH_CALUDE_f_increasing_condition_l3712_371219

/-- The quadratic function f(x) = 3x^2 - ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - a * x + 4

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 6 * x - a

/-- The theorem stating the condition for f(x) to be increasing on [-5, +∞) -/
theorem f_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ -5 → (f_deriv a x ≥ 0)) ↔ a ≤ -30 := by sorry

end NUMINAMATH_CALUDE_f_increasing_condition_l3712_371219


namespace NUMINAMATH_CALUDE_percentage_equality_l3712_371236

theorem percentage_equality (x : ℝ) (h : x = 130) : 
  (65 / 100 * x) / 422.50 * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3712_371236


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3712_371257

def age_ratio (a_current : ℕ) (b_current : ℕ) : ℚ :=
  (a_current + 20) / (b_current - 20)

theorem age_ratio_is_two_to_one :
  ∀ (a_current b_current : ℕ),
    b_current = 70 →
    a_current = b_current + 10 →
    age_ratio a_current b_current = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l3712_371257


namespace NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l3712_371212

theorem irrationality_of_sqrt_two_and_rationality_of_others : 
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = Real.sqrt 2) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = 1 / 3) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = 0) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (a : ℚ) / b = -0.7) :=
by sorry

end NUMINAMATH_CALUDE_irrationality_of_sqrt_two_and_rationality_of_others_l3712_371212


namespace NUMINAMATH_CALUDE_factor_polynomial_l3712_371232

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 162 * x^9 = -18 * x^5 * (9 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3712_371232


namespace NUMINAMATH_CALUDE_checkerboard_probability_l3712_371296

/-- The size of one side of the square checkerboard -/
def board_size : ℕ := 10

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * board_size - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not on the perimeter -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l3712_371296


namespace NUMINAMATH_CALUDE_right_triangle_altitude_segment_length_l3712_371260

/-- A right triangle with specific altitude properties -/
structure RightTriangleWithAltitudes where
  -- The lengths of the segments on the hypotenuse
  hypotenuse_segment1 : ℝ
  hypotenuse_segment2 : ℝ
  -- The length of one segment on a leg
  leg_segment : ℝ
  -- Ensure the hypotenuse segments are positive
  hyp_seg1_pos : 0 < hypotenuse_segment1
  hyp_seg2_pos : 0 < hypotenuse_segment2
  -- Ensure the leg segment is positive
  leg_seg_pos : 0 < leg_segment

/-- The theorem stating the length of the unknown segment -/
theorem right_triangle_altitude_segment_length 
  (triangle : RightTriangleWithAltitudes) 
  (h1 : triangle.hypotenuse_segment1 = 4)
  (h2 : triangle.hypotenuse_segment2 = 6)
  (h3 : triangle.leg_segment = 3) :
  ∃ y : ℝ, y = 4.5 ∧ 
    (triangle.leg_segment / triangle.hypotenuse_segment1 = 
     (triangle.leg_segment + y) / (triangle.hypotenuse_segment1 + triangle.hypotenuse_segment2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_segment_length_l3712_371260


namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l3712_371240

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2*p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l3712_371240


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3712_371204

theorem quadratic_rewrite (j : ℝ) : 
  ∃ (c p q : ℝ), 9 * j^2 - 12 * j + 27 = c * (j + p)^2 + q ∧ q / p = -69 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3712_371204


namespace NUMINAMATH_CALUDE_acid_dilution_l3712_371203

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 ∧ 
  initial_concentration = 0.4 ∧ 
  water_added = 30 ∧ 
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_acid_dilution_l3712_371203


namespace NUMINAMATH_CALUDE_sum_of_large_numbers_l3712_371271

theorem sum_of_large_numbers : 800000000000 + 299999999999 = 1099999999999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_large_numbers_l3712_371271


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l3712_371273

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - 4*x - 2*k + 8

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0

-- Define the additional condition
def roots_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^3 * x₂ + x₁ * x₂^3 = 24

-- Theorem statement
theorem quadratic_roots_theorem :
  ∀ k : ℝ, has_two_real_roots k →
  (∃ x₁ x₂ : ℝ, quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 ∧ roots_condition x₁ x₂) →
  k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l3712_371273


namespace NUMINAMATH_CALUDE_shanes_bread_packages_l3712_371251

theorem shanes_bread_packages :
  ∀ (slices_per_bread_package : ℕ) 
    (ham_packages : ℕ) 
    (slices_per_ham_package : ℕ) 
    (bread_slices_per_sandwich : ℕ) 
    (leftover_bread_slices : ℕ),
  slices_per_bread_package = 20 →
  ham_packages = 2 →
  slices_per_ham_package = 8 →
  bread_slices_per_sandwich = 2 →
  leftover_bread_slices = 8 →
  (ham_packages * slices_per_ham_package * bread_slices_per_sandwich + leftover_bread_slices) / slices_per_bread_package = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_shanes_bread_packages_l3712_371251


namespace NUMINAMATH_CALUDE_expression_equals_expected_result_l3712_371250

-- Define the expression
def expression : ℤ := 8 - (-3) + (-5) + (-7)

-- Define the expected result
def expected_result : ℤ := 3 + 8 - 7 - 5

-- Theorem statement
theorem expression_equals_expected_result :
  expression = expected_result :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_expected_result_l3712_371250


namespace NUMINAMATH_CALUDE_generalized_spatial_apollonian_problems_l3712_371249

/-- The number of types of objects (sphere, point, plane) --/
def n : ℕ := 3

/-- The number of objects to be chosen --/
def k : ℕ := 4

/-- Combinations with repetition --/
def combinations_with_repetition (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of generalized spatial Apollonian problems --/
theorem generalized_spatial_apollonian_problems :
  combinations_with_repetition n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_generalized_spatial_apollonian_problems_l3712_371249


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l3712_371235

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the quadrants
def first_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : Point) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : Point) : Prop := p.1 > 0 ∧ p.2 < 0

-- Define the points P and Q
def P (b : ℝ) : Point := (2, b)
def Q (b : ℝ) : Point := (b, -2)

-- State the theorem
theorem point_quadrant_relation (b : ℝ) :
  fourth_quadrant (P b) → third_quadrant (Q b) :=
by
  sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l3712_371235


namespace NUMINAMATH_CALUDE_cards_lost_l3712_371207

def initial_cards : ℕ := 88
def remaining_cards : ℕ := 18

theorem cards_lost : initial_cards - remaining_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_cards_lost_l3712_371207


namespace NUMINAMATH_CALUDE_francie_savings_l3712_371210

/-- Calculates Francie's remaining money after saving and spending --/
def franciesRemainingMoney (
  initialWeeklyAllowance : ℕ) 
  (initialWeeks : ℕ)
  (raisedWeeklyAllowance : ℕ)
  (raisedWeeks : ℕ)
  (videoGameCost : ℕ) : ℕ :=
  let totalSavings := initialWeeklyAllowance * initialWeeks + raisedWeeklyAllowance * raisedWeeks
  let remainingAfterClothes := totalSavings / 2
  remainingAfterClothes - videoGameCost

theorem francie_savings : franciesRemainingMoney 5 8 6 6 35 = 3 := by
  sorry

end NUMINAMATH_CALUDE_francie_savings_l3712_371210


namespace NUMINAMATH_CALUDE_min_side_length_l3712_371283

/-- An isosceles triangle with a perpendicular line from vertex to base -/
structure IsoscelesTriangleWithPerp where
  -- The length of two equal sides
  side : ℝ
  -- The length of CD
  cd : ℕ
  -- Assertion that BD^2 = 77
  h_bd_sq : side^2 - cd^2 = 77

/-- The theorem stating the minimal possible integer value for AC -/
theorem min_side_length (t : IsoscelesTriangleWithPerp) : 
  ∃ (min : ℕ), (∀ (t' : IsoscelesTriangleWithPerp), (t'.side : ℝ) ≥ min) ∧ min = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_l3712_371283


namespace NUMINAMATH_CALUDE_y_squared_value_l3712_371241

theorem y_squared_value (y : ℝ) (h : (y + 16) ^ (1/4) - (y - 16) ^ (1/4) = 2) : y^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l3712_371241


namespace NUMINAMATH_CALUDE_replacement_concentration_l3712_371224

/-- Represents a salt solution with a given concentration -/
structure SaltSolution where
  concentration : ℝ
  concentration_nonneg : 0 ≤ concentration
  concentration_le_one : concentration ≤ 1

/-- The result of mixing two salt solutions -/
def mix_solutions (s1 s2 : SaltSolution) (ratio : ℝ) : SaltSolution where
  concentration := s1.concentration * (1 - ratio) + s2.concentration * ratio
  concentration_nonneg := sorry
  concentration_le_one := sorry

theorem replacement_concentration 
  (original second : SaltSolution)
  (h1 : original.concentration = 0.14)
  (h2 : (mix_solutions original second 0.25).concentration = 0.16) :
  second.concentration = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_replacement_concentration_l3712_371224


namespace NUMINAMATH_CALUDE_brownie_division_l3712_371220

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 30

-- Define the dimensions of each brownie piece
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- Define the number of pieces
def num_pieces : ℕ := 60

-- Theorem statement
theorem brownie_division :
  pan_length * pan_width = num_pieces * piece_length * piece_width :=
by sorry

end NUMINAMATH_CALUDE_brownie_division_l3712_371220


namespace NUMINAMATH_CALUDE_A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l3712_371244

/-- Represents a hexagonal grid game. -/
structure HexGame where
  k : ℕ
  -- Add other necessary components of the game state

/-- Defines a valid move for Player A. -/
def valid_move_A (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player A
  sorry

/-- Defines a valid move for Player B. -/
def valid_move_B (game : HexGame) : Prop :=
  -- Define conditions for a valid move by Player B
  sorry

/-- Defines the winning condition for Player A. -/
def A_wins (game : HexGame) : Prop :=
  -- Define the condition when Player A wins
  sorry

/-- States that Player A can win in a finite number of moves for k = 5. -/
theorem A_can_win_with_5 : 
  ∃ (game : HexGame), game.k = 5 ∧ A_wins game :=
sorry

/-- States that Player A cannot win in a finite number of moves for k ≥ 6. -/
theorem A_cannot_win_with_6_or_more :
  ∀ (game : HexGame), game.k ≥ 6 → ¬(A_wins game) :=
sorry

/-- The main theorem stating that 6 is the minimum value of k for which
    Player A cannot win in a finite number of moves. -/
theorem min_k_for_A_cannot_win : 
  ∃ (k : ℕ), k = 6 ∧ 
  (∀ (game : HexGame), game.k ≥ k → ¬(A_wins game)) ∧
  (∀ (k' : ℕ), k' < k → ∃ (game : HexGame), game.k = k' ∧ A_wins game) :=
sorry

end NUMINAMATH_CALUDE_A_can_win_with_5_A_cannot_win_with_6_or_more_min_k_for_A_cannot_win_l3712_371244


namespace NUMINAMATH_CALUDE_solution_difference_l3712_371246

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 24 * r - 120) →
  ((s - 5) * (s + 5) = 24 * s - 120) →
  r ≠ s →
  r > s →
  r - s = 14 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l3712_371246


namespace NUMINAMATH_CALUDE_port_vessel_ratio_l3712_371284

theorem port_vessel_ratio :
  ∀ (cargo sailboats fishing : ℕ),
    cargo + 4 + sailboats + fishing = 28 →
    sailboats = cargo + 6 →
    sailboats = 7 * fishing →
    cargo = 2 * 4 :=
by sorry

end NUMINAMATH_CALUDE_port_vessel_ratio_l3712_371284


namespace NUMINAMATH_CALUDE_digit_count_700_l3712_371274

def count_digit (d : Nat) (n : Nat) : Nat :=
  (n / 100 + 1) * 10

theorem digit_count_700 : 
  (count_digit 9 700 + count_digit 8 700) = 280 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_700_l3712_371274


namespace NUMINAMATH_CALUDE_percentage_more_than_6_years_is_21_875_l3712_371209

/-- Represents the employee tenure distribution of a company -/
structure EmployeeTenure where
  less_than_3_years : ℕ
  between_3_and_6_years : ℕ
  more_than_6_years : ℕ

/-- Calculates the percentage of employees who have worked for more than 6 years -/
def percentage_more_than_6_years (e : EmployeeTenure) : ℚ :=
  (e.more_than_6_years : ℚ) / (e.less_than_3_years + e.between_3_and_6_years + e.more_than_6_years) * 100

/-- Proves that the percentage of employees who have worked for more than 6 years is 21.875% -/
theorem percentage_more_than_6_years_is_21_875 (e : EmployeeTenure) 
  (h : ∃ (x : ℕ), e.less_than_3_years = 10 * x ∧ 
                   e.between_3_and_6_years = 15 * x ∧ 
                   e.more_than_6_years = 7 * x) : 
  percentage_more_than_6_years e = 21875 / 1000 := by
  sorry

#eval (21875 : ℚ) / 1000  -- To verify that 21875/1000 = 21.875

end NUMINAMATH_CALUDE_percentage_more_than_6_years_is_21_875_l3712_371209


namespace NUMINAMATH_CALUDE_projection_magnitude_l3712_371238

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 1)

theorem projection_magnitude :
  let proj_magnitude := abs ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) * Real.sqrt (b.1^2 + b.2^2)
  proj_magnitude = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_projection_magnitude_l3712_371238


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l3712_371229

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l3712_371229


namespace NUMINAMATH_CALUDE_evaluate_expression_l3712_371227

theorem evaluate_expression : 5 - 9 * (8 - 3 * 2) / 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3712_371227


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3712_371254

theorem trigonometric_equality (θ : Real) (h : Real.sin (3 * Real.pi + θ) = 1/2) :
  (Real.cos (3 * Real.pi + θ)) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) +
  (Real.cos (θ - 4 * Real.pi)) / (Real.cos (θ + 2 * Real.pi) * Real.cos (3 * Real.pi + θ) + Real.cos (-θ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3712_371254


namespace NUMINAMATH_CALUDE_least_x_divisible_by_three_l3712_371289

theorem least_x_divisible_by_three : 
  ∃ (x : ℕ), x < 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(23 * 100 + y * 10 + 57) % 3 = 0) ∧
  (23 * 100 + x * 10 + 57) % 3 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_x_divisible_by_three_l3712_371289


namespace NUMINAMATH_CALUDE_sqrt2_not_all_zeros_in_range_l3712_371214

/-- The nth decimal digit of √2 -/
def d (n : ℕ) : ℕ := sorry

/-- The range of n we're considering -/
def n_range : Set ℕ := {n | 1000001 ≤ n ∧ n ≤ 3000000}

theorem sqrt2_not_all_zeros_in_range : 
  ¬ (∀ n ∈ n_range, d n = 0) := by sorry

end NUMINAMATH_CALUDE_sqrt2_not_all_zeros_in_range_l3712_371214


namespace NUMINAMATH_CALUDE_total_weight_jack_and_sam_l3712_371267

theorem total_weight_jack_and_sam : 
  ∀ (jack_weight sam_weight : ℕ),
  jack_weight = 52 →
  jack_weight = sam_weight + 8 →
  jack_weight + sam_weight = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_jack_and_sam_l3712_371267


namespace NUMINAMATH_CALUDE_pumpkin_relationship_other_orchard_pumpkins_l3712_371213

/-- Represents the number of pumpkins at Sunshine Orchard -/
def sunshine_pumpkins : ℕ := 54

/-- Represents the number of pumpkins at the other orchard -/
def other_pumpkins : ℕ := 14

/-- Theorem stating the relationship between the number of pumpkins at Sunshine Orchard and the other orchard -/
theorem pumpkin_relationship : sunshine_pumpkins = 3 * other_pumpkins + 12 := by
  sorry

/-- Theorem proving that the other orchard has 14 pumpkins given the conditions -/
theorem other_orchard_pumpkins : other_pumpkins = 14 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_relationship_other_orchard_pumpkins_l3712_371213


namespace NUMINAMATH_CALUDE_equidistant_point_l3712_371290

theorem equidistant_point : ∃ x : ℝ, |x - (-2)| = |x - 4| ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_l3712_371290


namespace NUMINAMATH_CALUDE_david_window_washing_time_l3712_371264

/-- Represents the time taken to wash windows -/
def wash_time (windows_per_unit : ℕ) (minutes_per_unit : ℕ) (total_windows : ℕ) : ℕ :=
  (total_windows / windows_per_unit) * minutes_per_unit

/-- Proves that it takes David 160 minutes to wash all windows in his house -/
theorem david_window_washing_time :
  wash_time 4 10 64 = 160 := by
  sorry

#eval wash_time 4 10 64

end NUMINAMATH_CALUDE_david_window_washing_time_l3712_371264


namespace NUMINAMATH_CALUDE_original_flock_size_l3712_371255

/-- Represents the flock size and its changes over time -/
structure FlockDynamics where
  initialSize : ℕ
  yearlyKilled : ℕ
  yearlyBorn : ℕ
  years : ℕ
  joinedFlockSize : ℕ
  finalCombinedSize : ℕ

/-- Theorem stating the original flock size given the conditions -/
theorem original_flock_size (fd : FlockDynamics)
  (h1 : fd.yearlyKilled = 20)
  (h2 : fd.yearlyBorn = 30)
  (h3 : fd.years = 5)
  (h4 : fd.joinedFlockSize = 150)
  (h5 : fd.finalCombinedSize = 300)
  : fd.initialSize = 100 := by
  sorry

#check original_flock_size

end NUMINAMATH_CALUDE_original_flock_size_l3712_371255


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3712_371285

def set_A (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0

def set_B (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x^2 + y^2 ≤ 1

theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ set_A x y ∨ set_B x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3712_371285


namespace NUMINAMATH_CALUDE_diamond_45_15_l3712_371277

/-- The diamond operation on positive real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ :=
  x / y

/-- Axioms for the diamond operation -/
axiom diamond_positive (x y : ℝ) : 0 < x → 0 < y → 0 < diamond x y

axiom diamond_prop1 (x y : ℝ) : 0 < x → 0 < y → diamond (x * y) y = x * diamond y y

axiom diamond_prop2 (x : ℝ) : 0 < x → diamond (diamond x 1) x = diamond x 1

axiom diamond_def (x y : ℝ) : 0 < x → 0 < y → diamond x y = x / y

axiom diamond_one : diamond 1 1 = 1

/-- Theorem: 45 ◇ 15 = 3 -/
theorem diamond_45_15 : diamond 45 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_45_15_l3712_371277


namespace NUMINAMATH_CALUDE_complex_multiplication_l3712_371234

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3712_371234


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l3712_371272

/-- Given Shekar's scores in four subjects and his average marks, prove his score in social studies -/
theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 75)
  (h5 : average_score = 73)
  (h6 : average_score = (math_score + science_score + english_score + biology_score + social_studies_score) / 5) :
  social_studies_score = 82 := by
  sorry

end NUMINAMATH_CALUDE_shekars_social_studies_score_l3712_371272


namespace NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3712_371237

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l3712_371237


namespace NUMINAMATH_CALUDE_masha_number_is_1001_l3712_371205

/-- Represents the possible operations Vasya could have performed -/
inductive Operation
  | Sum
  | Product

/-- Checks if a number is a valid choice for Sasha or Masha -/
def is_valid_choice (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 2002

/-- Checks if Sasha can determine Masha's number -/
def sasha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a + c = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a * c = 2002

/-- Checks if Masha can determine Sasha's number -/
def masha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c + b = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c * b = 2002

theorem masha_number_is_1001 (a b : ℕ) (op : Operation) :
  is_valid_choice a →
  is_valid_choice b →
  (op = Operation.Sum → a + b = 2002) →
  (op = Operation.Product → a * b = 2002) →
  ¬(sasha_can_determine a b op) →
  ¬(masha_can_determine a b op) →
  b = 1001 := by
  sorry


end NUMINAMATH_CALUDE_masha_number_is_1001_l3712_371205


namespace NUMINAMATH_CALUDE_min_value_sin_function_l3712_371218

theorem min_value_sin_function (x : Real) (h : x ∈ Set.Ioo 0 (Real.pi / 2)) :
  (2 * Real.sin x ^ 2 + 1) / Real.sin (2 * x) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l3712_371218


namespace NUMINAMATH_CALUDE_unique_star_solution_l3712_371262

/-- Definition of the ★ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists a unique real number y such that 4 ★ y = 10 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_star_solution_l3712_371262


namespace NUMINAMATH_CALUDE_square_field_area_l3712_371280

theorem square_field_area (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l3712_371280


namespace NUMINAMATH_CALUDE_first_rectangle_width_first_rectangle_width_proof_l3712_371269

/-- Given two rectangles, where the second has width 3 and height 6,
    and the first has height 5 and area 2 square inches more than the second,
    prove that the width of the first rectangle is 4 inches. -/
theorem first_rectangle_width : ℝ → Prop :=
  fun w : ℝ =>
    let first_height : ℝ := 5
    let second_width : ℝ := 3
    let second_height : ℝ := 6
    let first_area : ℝ := w * first_height
    let second_area : ℝ := second_width * second_height
    first_area = second_area + 2 → w = 4

/-- Proof of the theorem -/
theorem first_rectangle_width_proof : first_rectangle_width 4 := by
  sorry

end NUMINAMATH_CALUDE_first_rectangle_width_first_rectangle_width_proof_l3712_371269


namespace NUMINAMATH_CALUDE_election_ratio_l3712_371281

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.64 * X + 0.46 * Y = 0.58 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l3712_371281


namespace NUMINAMATH_CALUDE_salary_change_percentage_l3712_371261

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.51 → x = 70 := by sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l3712_371261


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3712_371288

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + m = 0 → y = x) → 
  m = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3712_371288


namespace NUMINAMATH_CALUDE_computer_price_increase_l3712_371231

theorem computer_price_increase (original_price : ℝ) : 
  original_price + 0.2 * original_price = 351 → 
  2 * original_price = 585 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l3712_371231


namespace NUMINAMATH_CALUDE_largest_valid_selection_l3712_371279

/-- Represents a selection of squares on an n × n grid -/
def Selection (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle contains a selected square -/
def containsSelected (s : Selection n) (x y w h : ℕ) : Prop :=
  ∃ (i j : Fin n), s i j ∧ x ≤ i.val ∧ i.val < x + w ∧ y ≤ j.val ∧ j.val < y + h

/-- Checks if a selection satisfies the condition for all rectangles -/
def validSelection (n : ℕ) (s : Selection n) : Prop :=
  ∀ (x y w h : ℕ), x + w ≤ n → y + h ≤ n → w * h ≥ n → containsSelected s x y w h

/-- The main theorem stating that 7 is the largest n satisfying the condition -/
theorem largest_valid_selection :
  (∀ n : ℕ, n ≤ 7 → ∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) ∧
  (∀ n : ℕ, n > 7 → ¬∃ s : Selection n, (∀ i : Fin n, ∃! j : Fin n, s i j) ∧ validSelection n s) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_selection_l3712_371279


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3712_371253

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at P -/
def k : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | tangent_line x y} ↔
  y - P.2 = k * (x - P.1) ∧ y = f x := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3712_371253


namespace NUMINAMATH_CALUDE_todds_time_is_correct_l3712_371242

/-- Todd's running time around the track -/
def todds_time : ℕ := 88

/-- Brian's running time around the track -/
def brians_time : ℕ := 96

/-- The difference in running time between Brian and Todd -/
def time_difference : ℕ := 8

/-- Theorem stating that Todd's time is correct given the conditions -/
theorem todds_time_is_correct : todds_time = brians_time - time_difference := by
  sorry

end NUMINAMATH_CALUDE_todds_time_is_correct_l3712_371242


namespace NUMINAMATH_CALUDE_river_depth_difference_l3712_371291

/-- River depth problem -/
theorem river_depth_difference (depth_may depth_june depth_july : ℝ) : 
  depth_may = 5 →
  depth_july = 45 →
  depth_july = 3 * depth_june →
  depth_june - depth_may = 10 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_difference_l3712_371291


namespace NUMINAMATH_CALUDE_tampa_bay_bucs_problem_l3712_371221

/-- The Tampa Bay Bucs team composition problem -/
theorem tampa_bay_bucs_problem 
  (initial_football_players : ℕ)
  (initial_cheerleaders : ℕ)
  (quitting_football_players : ℕ)
  (quitting_cheerleaders : ℕ)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : quitting_football_players = 10)
  (h4 : quitting_cheerleaders = 4) :
  (initial_football_players - quitting_football_players) + 
  (initial_cheerleaders - quitting_cheerleaders) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tampa_bay_bucs_problem_l3712_371221


namespace NUMINAMATH_CALUDE_age_ratio_after_years_l3712_371282

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8
def years_elapsed : ℕ := 4

theorem age_ratio_after_years : 
  (suzy_current_age + years_elapsed) / (mary_current_age + years_elapsed) = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_after_years_l3712_371282


namespace NUMINAMATH_CALUDE_conjecture_counterexample_l3712_371208

theorem conjecture_counterexample : ∃ n : ℕ, 
  (n % 2 = 1 ∧ n > 5) ∧ 
  ¬∃ (p k : ℕ), Prime p ∧ n = p + 2 * k^2 :=
sorry

end NUMINAMATH_CALUDE_conjecture_counterexample_l3712_371208


namespace NUMINAMATH_CALUDE_daniels_cats_l3712_371297

theorem daniels_cats (horses dogs turtles goats cats : ℕ) : 
  horses = 2 → 
  dogs = 5 → 
  turtles = 3 → 
  goats = 1 → 
  4 * (horses + dogs + cats + turtles + goats) = 72 → 
  cats = 7 := by
sorry

end NUMINAMATH_CALUDE_daniels_cats_l3712_371297


namespace NUMINAMATH_CALUDE_desk_rearrangement_combinations_l3712_371299

theorem desk_rearrangement_combinations : 
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 3
  let day4_choices : ℕ := 2
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 12 := by
sorry

end NUMINAMATH_CALUDE_desk_rearrangement_combinations_l3712_371299


namespace NUMINAMATH_CALUDE_basketball_passes_l3712_371225

/-- Represents the number of ways the ball can be with player A after n moves -/
def ball_with_A (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * ball_with_A (n - 1) + 3 * ball_with_A (n - 2)

/-- The problem statement -/
theorem basketball_passes :
  ball_with_A 7 = 1094 := by
  sorry


end NUMINAMATH_CALUDE_basketball_passes_l3712_371225


namespace NUMINAMATH_CALUDE_sin_360_minus_alpha_eq_sin_alpha_l3712_371295

theorem sin_360_minus_alpha_eq_sin_alpha (α : ℝ) : 
  Real.sin (2 * Real.pi - α) = Real.sin α := by sorry

end NUMINAMATH_CALUDE_sin_360_minus_alpha_eq_sin_alpha_l3712_371295


namespace NUMINAMATH_CALUDE_sticker_distribution_l3712_371239

theorem sticker_distribution (total_stickers : ℕ) (num_friends : ℕ) 
  (h1 : total_stickers = 72) (h2 : num_friends = 9) :
  total_stickers / num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3712_371239


namespace NUMINAMATH_CALUDE_proposition_a_is_true_l3712_371226

theorem proposition_a_is_true : ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0 := by
  sorry

#check proposition_a_is_true

end NUMINAMATH_CALUDE_proposition_a_is_true_l3712_371226


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l3712_371270

theorem jenny_easter_eggs (red_eggs : ℕ) (orange_eggs : ℕ) (eggs_per_basket : ℕ) 
  (h1 : red_eggs = 21)
  (h2 : orange_eggs = 28)
  (h3 : eggs_per_basket ≥ 5)
  (h4 : red_eggs % eggs_per_basket = 0)
  (h5 : orange_eggs % eggs_per_basket = 0) :
  eggs_per_basket = 7 := by
sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l3712_371270


namespace NUMINAMATH_CALUDE_intersection_and_coefficients_l3712_371252

def A : Set ℝ := {x | x^2 < 9}
def B : Set ℝ := {x | (x-2)*(x+4) < 0}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -3 < x ∧ x < 2}) ∧
  (∃ a b : ℝ, ∀ x : ℝ, (x ∈ A ∪ B) ↔ (2*x^2 + a*x + b < 0) ∧ a = 2 ∧ b = -24) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_coefficients_l3712_371252


namespace NUMINAMATH_CALUDE_total_fingers_folded_l3712_371202

/-- The number of fingers folded by Yoojung -/
def yoojung_fingers : ℕ := 2

/-- The number of fingers folded by Yuna -/
def yuna_fingers : ℕ := 5

/-- The total number of fingers folded by both Yoojung and Yuna -/
def total_fingers : ℕ := yoojung_fingers + yuna_fingers

theorem total_fingers_folded : total_fingers = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_fingers_folded_l3712_371202


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3712_371275

theorem complex_magnitude_problem (z : ℂ) (h : z / (2 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3712_371275


namespace NUMINAMATH_CALUDE_cookies_division_l3712_371216

/-- The number of cookies each person received -/
def cookies_per_person : ℕ := 30

/-- The total number of cookies prepared -/
def total_cookies : ℕ := 420

/-- The number of people Brenda's mother made cookies for -/
def number_of_people : ℕ := total_cookies / cookies_per_person

theorem cookies_division (cookies_per_person : ℕ) (total_cookies : ℕ) :
  cookies_per_person > 0 →
  total_cookies % cookies_per_person = 0 →
  number_of_people = 14 := by
  sorry

end NUMINAMATH_CALUDE_cookies_division_l3712_371216


namespace NUMINAMATH_CALUDE_union_of_sets_l3712_371266

/-- Given sets A and B, prove that their union is [-1, +∞) -/
theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | -3 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3}) →
  (B = {x : ℝ | x > 1}) →
  A ∪ B = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3712_371266


namespace NUMINAMATH_CALUDE_unique_solution_abc_l3712_371201

theorem unique_solution_abc :
  ∀ A B C : ℝ,
  A = 2 * B - 3 * C →
  B = 2 * C - 5 →
  A + B + C = 100 →
  A = 18.75 ∧ B = 52.5 ∧ C = 28.75 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l3712_371201


namespace NUMINAMATH_CALUDE_accounting_class_average_score_l3712_371256

/-- The average score for an accounting class --/
def average_score (total_students : ℕ) 
  (day1_percent day2_percent day3_percent : ℚ)
  (day1_score day2_score day3_score : ℚ) : ℚ :=
  (day1_percent * day1_score + day2_percent * day2_score + day3_percent * day3_score) / 1

theorem accounting_class_average_score :
  let total_students : ℕ := 200
  let day1_percent : ℚ := 60 / 100
  let day2_percent : ℚ := 30 / 100
  let day3_percent : ℚ := 10 / 100
  let day1_score : ℚ := 65 / 100
  let day2_score : ℚ := 75 / 100
  let day3_score : ℚ := 95 / 100
  average_score total_students day1_percent day2_percent day3_percent day1_score day2_score day3_score = 71 / 100 := by
  sorry

end NUMINAMATH_CALUDE_accounting_class_average_score_l3712_371256


namespace NUMINAMATH_CALUDE_physics_marks_correct_l3712_371259

/-- Given a student's marks in four subjects and their average across five subjects,
    calculate the marks in the fifth subject. -/
def calculate_physics_marks (e m c b : ℕ) (avg : ℚ) (n : ℕ) : ℚ :=
  n * avg - (e + m + c + b)

/-- Theorem stating that the calculated physics marks are correct given the problem conditions. -/
theorem physics_marks_correct 
  (e m c b : ℕ) 
  (avg : ℚ) 
  (n : ℕ) 
  (h1 : e = 70) 
  (h2 : m = 60) 
  (h3 : c = 60) 
  (h4 : b = 65) 
  (h5 : avg = 66.6) 
  (h6 : n = 5) : 
  calculate_physics_marks e m c b avg n = 78 := by
sorry

#eval calculate_physics_marks 70 60 60 65 66.6 5

end NUMINAMATH_CALUDE_physics_marks_correct_l3712_371259


namespace NUMINAMATH_CALUDE_roots_sum_minus_product_l3712_371233

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → 
  x₂^2 - 2*x₂ - 1 = 0 → 
  x₁ + x₂ - x₁*x₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_minus_product_l3712_371233


namespace NUMINAMATH_CALUDE_system_solution_unique_l3712_371206

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2^(x + y) = x + 7) ∧ (x + y = 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3712_371206


namespace NUMINAMATH_CALUDE_log_equation_l3712_371294

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : log10 2 * log10 50 + log10 25 - log10 5 * log10 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l3712_371294
