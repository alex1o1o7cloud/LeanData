import Mathlib

namespace NUMINAMATH_CALUDE_plush_bear_distribution_l281_28193

theorem plush_bear_distribution (total_bears : ℕ) (kindergarten_bears : ℕ) (num_classes : ℕ) :
  total_bears = 48 →
  kindergarten_bears = 15 →
  num_classes = 3 →
  (total_bears - kindergarten_bears) / num_classes = 11 :=
by sorry

end NUMINAMATH_CALUDE_plush_bear_distribution_l281_28193


namespace NUMINAMATH_CALUDE_framed_photo_border_area_l281_28167

/-- The area of the border of a framed rectangular photograph -/
theorem framed_photo_border_area 
  (photo_height : ℝ) 
  (photo_width : ℝ) 
  (border_width : ℝ) 
  (h1 : photo_height = 6) 
  (h2 : photo_width = 8) 
  (h3 : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 120 := by
  sorry

end NUMINAMATH_CALUDE_framed_photo_border_area_l281_28167


namespace NUMINAMATH_CALUDE_nine_identical_digits_multiples_l281_28140

theorem nine_identical_digits_multiples (n : ℕ) : 
  n ≥ 1 ∧ n ≤ 9 → 
  ∃ (d : ℕ), d ≥ 1 ∧ d ≤ 9 ∧ 12345679 * (9 * n) = d * 111111111 ∧
  (∀ (m : ℕ), 12345679 * m = d * 111111111 → m = 9 * n) :=
by sorry

end NUMINAMATH_CALUDE_nine_identical_digits_multiples_l281_28140


namespace NUMINAMATH_CALUDE_equal_numbers_theorem_l281_28181

theorem equal_numbers_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) :
  a = b ∨ a = c ∨ b = c :=
sorry

end NUMINAMATH_CALUDE_equal_numbers_theorem_l281_28181


namespace NUMINAMATH_CALUDE_set_B_determination_l281_28176

theorem set_B_determination (A B : Set ℕ) : 
  A = {1, 2} → 
  A ∩ B = {1} → 
  A ∪ B = {0, 1, 2} → 
  B = {0, 1} := by
sorry

end NUMINAMATH_CALUDE_set_B_determination_l281_28176


namespace NUMINAMATH_CALUDE_ryan_study_time_l281_28100

/-- Ryan's daily study hours for English -/
def english_hours : ℕ := 6

/-- Ryan's daily study hours for Chinese -/
def chinese_hours : ℕ := 7

/-- Number of days Ryan studies -/
def study_days : ℕ := 5

/-- Total study hours for both languages over the given period -/
def total_study_hours : ℕ := (english_hours + chinese_hours) * study_days

theorem ryan_study_time : total_study_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_ryan_study_time_l281_28100


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l281_28163

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line)
  (given_point : Point)
  (result_line : Line) : Prop :=
  (given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 5) →
  (given_point.x = -2 ∧ given_point.y = 1) →
  (result_line.a = 2 ∧ result_line.b = -3 ∧ result_line.c = 7) →
  (given_point.liesOn result_line ∧ result_line.isParallelTo given_line)

-- The proof of the theorem
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 2 (-3) 5) 
  (Point.mk (-2) 1) 
  (Line.mk 2 (-3) 7) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l281_28163


namespace NUMINAMATH_CALUDE_final_cell_count_l281_28104

def initial_cells : ℕ := 5
def split_ratio : ℕ := 3
def split_interval : ℕ := 3
def total_days : ℕ := 12

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem final_cell_count :
  geometric_sequence initial_cells split_ratio (total_days / split_interval) = 135 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l281_28104


namespace NUMINAMATH_CALUDE_undefined_value_expression_undefined_l281_28120

theorem undefined_value (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined (x : ℝ) : 
  ¬(∃y : ℝ, y = (3*x^3 + 5) / (x^2 - 24*x + 144)) ↔ (x = 12) := by sorry

end NUMINAMATH_CALUDE_undefined_value_expression_undefined_l281_28120


namespace NUMINAMATH_CALUDE_bisection_method_result_l281_28180

def f (x : ℝ) := x^3 - 3*x + 1

theorem bisection_method_result :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 0 < x₀ ∧ x₀ < 1 →
  ∃ a b : ℝ, 1/4 < a ∧ a < x₀ ∧ x₀ < b ∧ b < 1/2 ∧
    f a * f b < 0 ∧
    ∀ c ∈ Set.Ioo (0 : ℝ) 1, f c * f (1/2) ≤ 0 → c ≤ 1/2 ∧
    ∀ d ∈ Set.Ioo (0 : ℝ) (1/2), f d * f (1/4) ≤ 0 → 1/4 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_result_l281_28180


namespace NUMINAMATH_CALUDE_chocolate_profit_example_l281_28142

/-- Calculates the profit from selling chocolates given the following conditions:
  * Number of chocolate bars
  * Cost price per bar
  * Total selling price
  * Packaging cost per bar
-/
def chocolate_profit (num_bars : ℕ) (cost_price : ℚ) (total_selling_price : ℚ) (packaging_cost : ℚ) : ℚ :=
  total_selling_price - (num_bars * (cost_price + packaging_cost))

theorem chocolate_profit_example :
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_profit_example_l281_28142


namespace NUMINAMATH_CALUDE_maintenance_check_increase_maintenance_check_theorem_l281_28189

theorem maintenance_check_increase (initial_interval : ℝ) 
  (additive_a_percent : ℝ) (additive_b_percent : ℝ) : ℝ :=
  let interval_after_a := initial_interval * (1 + additive_a_percent)
  let interval_after_b := interval_after_a * (1 + additive_b_percent)
  let total_increase_percent := (interval_after_b - initial_interval) / initial_interval * 100
  total_increase_percent

theorem maintenance_check_theorem :
  maintenance_check_increase 45 0.35 0.20 = 62 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_maintenance_check_theorem_l281_28189


namespace NUMINAMATH_CALUDE_lucas_overall_accuracy_l281_28122

theorem lucas_overall_accuracy 
  (emily_individual_accuracy : Real) 
  (emily_overall_accuracy : Real)
  (lucas_individual_accuracy : Real)
  (h1 : emily_individual_accuracy = 0.7)
  (h2 : emily_overall_accuracy = 0.82)
  (h3 : lucas_individual_accuracy = 0.85) :
  lucas_individual_accuracy * 0.5 + (emily_overall_accuracy - emily_individual_accuracy * 0.5) = 0.895 := by
  sorry

end NUMINAMATH_CALUDE_lucas_overall_accuracy_l281_28122


namespace NUMINAMATH_CALUDE_fence_length_for_specific_yard_l281_28196

/-- A rectangular yard with given dimensions and area -/
structure RectangularYard where
  length : ℝ
  width : ℝ
  area : ℝ
  length_positive : 0 < length
  width_positive : 0 < width
  area_eq : area = length * width

/-- The fence length for a rectangular yard -/
def fence_length (yard : RectangularYard) : ℝ :=
  2 * yard.width + yard.length

/-- Theorem: For a rectangular yard with one side of 40 feet and an area of 240 square feet,
    the sum of the lengths of the other three sides is 52 feet. -/
theorem fence_length_for_specific_yard :
  ∃ (yard : RectangularYard),
    yard.length = 40 ∧
    yard.area = 240 ∧
    fence_length yard = 52 := by
  sorry

end NUMINAMATH_CALUDE_fence_length_for_specific_yard_l281_28196


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l281_28127

theorem largest_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ c : ℝ, c = 1/2 ∧ x^6 + y^6 ≥ c * x * y ∧ ∀ d : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d * a * b) → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l281_28127


namespace NUMINAMATH_CALUDE_angle_measure_l281_28126

theorem angle_measure (m1 m2 m3 : ℝ) (h1 : m1 = 80) (h2 : m2 = 35) (h3 : m3 = 25) :
  ∃ m4 : ℝ, m4 = 140 ∧ m1 + m2 + m3 + (180 - m4) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l281_28126


namespace NUMINAMATH_CALUDE_probability_different_colors_7_5_l281_28147

/-- The probability of drawing two chips of different colors from a bag -/
def probability_different_colors (blue_chips yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + yellow_chips
  let prob_blue_then_yellow := (blue_chips : ℚ) / total_chips * yellow_chips / (total_chips - 1)
  let prob_yellow_then_blue := (yellow_chips : ℚ) / total_chips * blue_chips / (total_chips - 1)
  prob_blue_then_yellow + prob_yellow_then_blue

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_7_5 :
  probability_different_colors 7 5 = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_7_5_l281_28147


namespace NUMINAMATH_CALUDE_running_preference_related_to_gender_certainty_running_preference_related_to_gender_l281_28143

/-- Represents the data from the survey about running preferences among university students. -/
structure RunningPreferenceSurvey where
  total_students : ℕ
  boys : ℕ
  boys_not_liking : ℕ
  girls_liking : ℕ

/-- Calculates the chi-square value for the given survey data. -/
def calculate_chi_square (survey : RunningPreferenceSurvey) : ℚ :=
  let girls := survey.total_students - survey.boys
  let boys_liking := survey.boys - survey.boys_not_liking
  let girls_not_liking := girls - survey.girls_liking
  let n := survey.total_students
  let a := boys_liking
  let b := survey.boys_not_liking
  let c := survey.girls_liking
  let d := girls_not_liking
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Theorem stating that the chi-square value for the given survey data is greater than 6.635,
    indicating a 99% certainty that liking running is related to gender. -/
theorem running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  calculate_chi_square survey > 6635 / 1000 :=
sorry

/-- Corollary stating that there is a 99% certainty that liking running is related to gender. -/
theorem certainty_running_preference_related_to_gender (survey : RunningPreferenceSurvey)
  (h1 : survey.total_students = 200)
  (h2 : survey.boys = 120)
  (h3 : survey.boys_not_liking = 50)
  (h4 : survey.girls_liking = 30) :
  ∃ (certainty : ℚ), certainty = 99 / 100 ∧
  calculate_chi_square survey > 6635 / 1000 :=
sorry

end NUMINAMATH_CALUDE_running_preference_related_to_gender_certainty_running_preference_related_to_gender_l281_28143


namespace NUMINAMATH_CALUDE_point_on_line_l281_28117

/-- Prove that for a point P(2, m) lying on the line 3x + y = 2, the value of m is -4. -/
theorem point_on_line (m : ℝ) : (3 * 2 + m = 2) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l281_28117


namespace NUMINAMATH_CALUDE_xyz_product_l281_28116

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l281_28116


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l281_28138

/-- Given a store with coloring books, prove the number of shelves used -/
theorem coloring_book_shelves 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : books_per_shelf = 7)
  : (initial_stock - books_sold) / books_per_shelf = 3 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l281_28138


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l281_28190

/-- The number of ways to distribute n students among k universities,
    with each university receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  sorry

/-- The number of ways to partition n elements into k non-empty subsets. -/
def stirling2 (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distribute_students 5 3 = 150 :=
sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l281_28190


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l281_28146

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1
theorem union_condition (a : ℝ) : A a ∪ B = A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l281_28146


namespace NUMINAMATH_CALUDE_largest_integer_solution_l281_28151

theorem largest_integer_solution : 
  ∃ (x : ℕ), (1/4 : ℚ) + (x/5 : ℚ) < 2 ∧ 
  ∀ (y : ℕ), y > x → (1/4 : ℚ) + (y/5 : ℚ) ≥ 2 :=
by
  use 23
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l281_28151


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_lines_l281_28128

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x^2 + 1) + 1

theorem f_derivative_and_tangent_lines :
  (∃ f' : ℝ → ℝ, ∀ x, deriv f x = f' x ∧ f' x = 3 * x^2 - 2 * x + 1) ∧
  (∃ t₁ t₂ : ℝ → ℝ,
    (∀ x, t₁ x = x) ∧
    (∀ x, t₂ x = 2 * x - 1) ∧
    (t₁ 1 = f 1 ∧ t₂ 1 = f 1) ∧
    (∃ x₀, deriv f x₀ = deriv t₁ x₀ ∧ f x₀ = t₁ x₀) ∧
    (∃ x₁, deriv f x₁ = deriv t₂ x₁ ∧ f x₁ = t₂ x₁)) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_and_tangent_lines_l281_28128


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l281_28188

theorem fraction_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l281_28188


namespace NUMINAMATH_CALUDE_chord_length_squared_l281_28119

/-- Two circles with radii 8 and 6, centers 12 units apart, intersecting at P.
    Q and R are points on the circles such that QP = PR. -/
structure CircleConfiguration where
  circle1_radius : ℝ
  circle2_radius : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h1 : circle1_radius = 8
  h2 : circle2_radius = 6
  h3 : center_distance = 12
  h4 : chord_length > 0

/-- The square of the chord length in the given circle configuration is 130. -/
theorem chord_length_squared (config : CircleConfiguration) : 
  config.chord_length ^ 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l281_28119


namespace NUMINAMATH_CALUDE_nacl_formed_l281_28145

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  product3 : String

-- Define the moles of substances
structure Moles where
  nh4cl : ℝ
  naoh : ℝ
  nacl : ℝ

-- Define the reaction and initial moles
def reaction : Reaction :=
  { reactant1 := "NH4Cl"
  , reactant2 := "NaOH"
  , product1 := "NaCl"
  , product2 := "NH3"
  , product3 := "H2O" }

def initial_moles : Moles :=
  { nh4cl := 2
  , naoh := 2
  , nacl := 0 }

-- Theorem statement
theorem nacl_formed (r : Reaction) (m : Moles) :
  r = reaction ∧ m = initial_moles →
  m.nacl + 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_nacl_formed_l281_28145


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l281_28159

theorem quadratic_root_sum (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∀ x : ℝ, x^2 - p*x + r = 0 → ∃ y : ℝ, y^2 - p*y + r = 0 ∧ x + y = 8) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l281_28159


namespace NUMINAMATH_CALUDE_passing_percentage_is_45_percent_l281_28102

/-- Represents an examination with a passing percentage. -/
structure Examination where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Calculates the passing marks for an examination. -/
def passing_marks (exam : Examination) : ℚ :=
  (exam.passing_percentage / 100) * exam.max_marks

/-- Theorem: The passing percentage is 45% given the conditions. -/
theorem passing_percentage_is_45_percent 
  (max_marks : ℕ) 
  (failing_score : ℕ) 
  (deficit : ℕ) 
  (h1 : max_marks = 500)
  (h2 : failing_score = 180)
  (h3 : deficit = 45)
  : ∃ (exam : Examination), 
    exam.max_marks = max_marks ∧ 
    exam.passing_percentage = 45 ∧
    passing_marks exam = failing_score + deficit :=
  sorry


end NUMINAMATH_CALUDE_passing_percentage_is_45_percent_l281_28102


namespace NUMINAMATH_CALUDE_cylinder_from_constant_radius_l281_28114

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying r = c in cylindrical coordinates -/
def CylindricalSet (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.r = c}

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) : Prop :=
  ∃ c : ℝ, c > 0 ∧ S = CylindricalSet c

theorem cylinder_from_constant_radius (c : ℝ) (h : c > 0) :
  IsCylinder (CylindricalSet c) := by
  sorry

#check cylinder_from_constant_radius

end NUMINAMATH_CALUDE_cylinder_from_constant_radius_l281_28114


namespace NUMINAMATH_CALUDE_total_ways_eq_600_l281_28133

/-- Represents the number of cards in the left pocket -/
def left_cards : ℕ := 30

/-- Represents the number of cards in the right pocket -/
def right_cards : ℕ := 20

/-- Represents the total number of ways to select one card from each pocket -/
def total_ways : ℕ := left_cards * right_cards

/-- Theorem stating that the total number of ways to select one card from each pocket is 600 -/
theorem total_ways_eq_600 : total_ways = 600 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_600_l281_28133


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_A_solution_is_correct_l281_28103

/-- Given a mixture of liquids A and B, this theorem proves the initial amount of liquid A. -/
theorem initial_amount_of_liquid_A
  (initial_ratio : ℚ) -- Initial ratio of A to B
  (replacement_volume : ℚ) -- Volume of mixture replaced with B
  (final_ratio : ℚ) -- Final ratio of A to B
  (h1 : initial_ratio = 4 / 1)
  (h2 : replacement_volume = 40)
  (h3 : final_ratio = 2 / 3)
  : ℚ :=
by
  sorry

#check initial_amount_of_liquid_A

/-- The solution to the problem -/
def solution : ℚ := 32

/-- Proof that the solution is correct -/
theorem solution_is_correct :
  initial_amount_of_liquid_A (4 / 1) 40 (2 / 3) rfl rfl rfl = solution :=
by
  sorry

end NUMINAMATH_CALUDE_initial_amount_of_liquid_A_solution_is_correct_l281_28103


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l281_28101

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) : 
  -3 ≤ x - 2*y ∧ x - 2*y < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l281_28101


namespace NUMINAMATH_CALUDE_right_triangle_properties_l281_28197

/-- A triangle with side lengths 13, 84, and 85 is a right triangle with area 546, semiperimeter 91, and inradius 6 -/
theorem right_triangle_properties : ∃ (a b c : ℝ), 
  a = 13 ∧ b = 84 ∧ c = 85 ∧
  a^2 + b^2 = c^2 ∧
  (1/2 * a * b : ℝ) = 546 ∧
  ((a + b + c) / 2 : ℝ) = 91 ∧
  (546 / 91 : ℝ) = 6 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_properties_l281_28197


namespace NUMINAMATH_CALUDE_zenith_school_reading_fraction_l281_28137

/-- Represents the student body at Zenith Middle School -/
structure StudentBody where
  total : ℕ
  enjoy_reading : ℕ
  dislike_reading : ℕ
  enjoy_and_express : ℕ
  enjoy_but_pretend_dislike : ℕ
  dislike_and_express : ℕ
  dislike_but_pretend_enjoy : ℕ

/-- The conditions of the problem -/
def zenith_school (s : StudentBody) : Prop :=
  s.total > 0 ∧
  s.enjoy_reading = (70 * s.total) / 100 ∧
  s.dislike_reading = s.total - s.enjoy_reading ∧
  s.enjoy_and_express = (70 * s.enjoy_reading) / 100 ∧
  s.enjoy_but_pretend_dislike = s.enjoy_reading - s.enjoy_and_express ∧
  s.dislike_and_express = (75 * s.dislike_reading) / 100 ∧
  s.dislike_but_pretend_enjoy = s.dislike_reading - s.dislike_and_express

/-- The theorem to be proved -/
theorem zenith_school_reading_fraction (s : StudentBody) :
  zenith_school s →
  (s.enjoy_but_pretend_dislike : ℚ) / (s.enjoy_but_pretend_dislike + s.dislike_and_express) = 21 / 43 := by
  sorry


end NUMINAMATH_CALUDE_zenith_school_reading_fraction_l281_28137


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l281_28161

/-- The eccentricity of a hyperbola with equation (y^2 / a^2) - (x^2 / b^2) = 1 and asymptote y = 2x is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (k : ℝ), k = a / b ∧ k = 2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l281_28161


namespace NUMINAMATH_CALUDE_canada_trip_problem_l281_28174

/-- Represents the exchange rate from US dollars to Canadian dollars -/
def exchange_rate : ℚ := 15 / 9

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem canada_trip_problem (d : ℕ) :
  (exchange_rate * d - 120 = d) → 
  d = 180 ∧ sum_of_digits d = 9 := by
  sorry

end NUMINAMATH_CALUDE_canada_trip_problem_l281_28174


namespace NUMINAMATH_CALUDE_distance_C_D_l281_28113

/-- An ellipse with equation 16(x-2)^2 + 4y^2 = 64 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 16 * (p.1 - 2)^2 + 4 * p.2^2 = 64}

/-- The center of the ellipse -/
def center : ℝ × ℝ := (2, 0)

/-- The semi-major axis length -/
def a : ℝ := 4

/-- The semi-minor axis length -/
def b : ℝ := 2

/-- An endpoint of the major axis -/
def C : ℝ × ℝ := (center.1, center.2 + a)

/-- An endpoint of the minor axis -/
def D : ℝ × ℝ := (center.1 + b, center.2)

/-- The theorem stating the distance between C and D -/
theorem distance_C_D : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_C_D_l281_28113


namespace NUMINAMATH_CALUDE_solve_for_x_l281_28153

theorem solve_for_x (x : ℤ) : x + 1315 + 9211 - 1569 = 11901 → x = 2944 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l281_28153


namespace NUMINAMATH_CALUDE_thirty_three_million_equals_33000000_l281_28154

-- Define million
def million : ℕ := 1000000

-- Define 33 million
def thirty_three_million : ℕ := 33 * million

-- Theorem to prove
theorem thirty_three_million_equals_33000000 : 
  thirty_three_million = 33000000 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_million_equals_33000000_l281_28154


namespace NUMINAMATH_CALUDE_optimal_triangle_height_l281_28187

/-- Given two parallel lines with distance b between them, and a segment of length a on one of the lines,
    the sum of areas of two triangles formed by connecting a point on the line segment to a point on the other parallel line
    is minimized when the height of one triangle is b√2/2. -/
theorem optimal_triangle_height (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let height := b * Real.sqrt 2 / 2
  let area (h : ℝ) := a * h / 2 + a * (b - h) ^ 2 / (2 * h)
  ∀ h, 0 < h ∧ h < b → area height ≤ area h := by
sorry

end NUMINAMATH_CALUDE_optimal_triangle_height_l281_28187


namespace NUMINAMATH_CALUDE_imaginary_part_of_f_i_over_i_l281_28149

-- Define the complex function f(x) = x^3 - 1
def f (x : ℂ) : ℂ := x^3 - 1

-- State the theorem
theorem imaginary_part_of_f_i_over_i :
  Complex.im (f Complex.I / Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_f_i_over_i_l281_28149


namespace NUMINAMATH_CALUDE_kishore_savings_l281_28130

def monthly_salary (expenses : ℕ) (savings_rate : ℚ) : ℚ :=
  expenses / (1 - savings_rate)

def savings (salary : ℚ) (savings_rate : ℚ) : ℚ :=
  salary * savings_rate

theorem kishore_savings (expenses : ℕ) (savings_rate : ℚ) :
  expenses = 18000 →
  savings_rate = 1/10 →
  savings (monthly_salary expenses savings_rate) savings_rate = 2000 := by
sorry

end NUMINAMATH_CALUDE_kishore_savings_l281_28130


namespace NUMINAMATH_CALUDE_vertex_not_zero_l281_28125

/-- The vertex of a quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis if and only if
    m = 2 or m = -2 or m = 6 -/
def vertex_on_axis (m : ℝ) : Prop :=
  m = 2 ∨ m = -2 ∨ m = 6

/-- If the vertex of the quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis,
    then m ≠ 0 -/
theorem vertex_not_zero (m : ℝ) (h : vertex_on_axis m) : m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vertex_not_zero_l281_28125


namespace NUMINAMATH_CALUDE_race_time_proof_l281_28191

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The conditions of the specific race described in the problem -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  r.runner_a.speed * r.runner_a.time = r.distance ∧
  r.runner_b.speed * r.runner_b.time = r.distance ∧
  (r.runner_a.speed * r.runner_a.time - r.runner_b.speed * r.runner_a.time = 50 ∨
   r.runner_b.time - r.runner_a.time = 20)

theorem race_time_proof (r : Race) (h : race_conditions r) : r.runner_a.time = 400 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l281_28191


namespace NUMINAMATH_CALUDE_house_price_ratio_l281_28168

def total_price : ℕ := 600000
def first_house_price : ℕ := 200000

theorem house_price_ratio :
  (total_price - first_house_price) / first_house_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_house_price_ratio_l281_28168


namespace NUMINAMATH_CALUDE_different_result_l281_28134

theorem different_result : 
  (-2 - (-3) ≠ 2 - 3) ∧ 
  (-2 - (-3) ≠ -3 + 2) ∧ 
  (-2 - (-3) ≠ -3 - (-2)) ∧ 
  (2 - 3 = -3 + 2) ∧ 
  (2 - 3 = -3 - (-2)) := by
  sorry

end NUMINAMATH_CALUDE_different_result_l281_28134


namespace NUMINAMATH_CALUDE_angle_measure_proof_l281_28175

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = (180 - x) / 2 - 25) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l281_28175


namespace NUMINAMATH_CALUDE_part_one_part_two_l281_28183

open Real

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Theorem for part (1)
theorem part_one (t : Triangle) 
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.C = π/3)
  (h3 : t.b = 1) : 
  t.a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (t : Triangle)
  (h1 : t.a^2 = 4 * Real.sqrt 3 * t.S)
  (h2 : t.c / t.b = 2 + Real.sqrt 3) :
  t.A = π/3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l281_28183


namespace NUMINAMATH_CALUDE_total_weight_loss_is_correct_l281_28150

/-- The total weight loss of Seth, Jerome, Veronica, and Maya -/
def totalWeightLoss (sethLoss : ℝ) : ℝ :=
  let jeromeLoss := 3 * sethLoss
  let veronicaLoss := sethLoss + 1.56
  let sethVeronicaCombined := sethLoss + veronicaLoss
  let mayaLoss := sethVeronicaCombined * 0.75
  sethLoss + jeromeLoss + veronicaLoss + mayaLoss

/-- Theorem stating that the total weight loss is 116.675 pounds -/
theorem total_weight_loss_is_correct :
  totalWeightLoss 17.53 = 116.675 := by
  sorry

#eval totalWeightLoss 17.53

end NUMINAMATH_CALUDE_total_weight_loss_is_correct_l281_28150


namespace NUMINAMATH_CALUDE_sine_function_property_l281_28177

theorem sine_function_property (ω : ℝ) (a : ℝ) (h1 : ω > 0) :
  (∀ x : ℝ, Real.sin (ω * (x - 1/2)) = Real.sin (ω * (x + 1/2))) →
  Real.sin (-ω/4) = a →
  Real.sin (9*ω/4) = -a := by sorry

end NUMINAMATH_CALUDE_sine_function_property_l281_28177


namespace NUMINAMATH_CALUDE_van_capacity_l281_28141

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 33 →
  adults = 9 →
  vans = 6 →
  (students + adults) / vans = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l281_28141


namespace NUMINAMATH_CALUDE_event_C_is_certain_l281_28135

-- Define an enumeration for the events
inductive Event
  | A -- It will rain after thunder
  | B -- Tomorrow will be sunny
  | C -- 1 hour equals 60 minutes
  | D -- There will be a rainbow after the rain

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.C => True
  | _ => False

-- Theorem stating that Event C is certain
theorem event_C_is_certain : isCertain Event.C := by
  sorry

end NUMINAMATH_CALUDE_event_C_is_certain_l281_28135


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l281_28155

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def line (k m x y : ℝ) : Prop := y = k * x + m

def perpendicular_bisector (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - (y₁ + y₂) / 2) / (x - (x₁ + x₂) / 2) = -(x₂ - x₁) / (y₂ - y₁)

theorem ellipse_line_intersection (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
    perpendicular_bisector x₁ y₁ x₂ y₂ 0 (-1/2)) →
  2 * k^2 + 1 = 2 * m := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l281_28155


namespace NUMINAMATH_CALUDE_adidas_cost_l281_28160

/-- The cost of Adidas shoes given the sales information -/
theorem adidas_cost (total_goal : ℝ) (nike_cost reebok_cost : ℝ) 
  (nike_sold adidas_sold reebok_sold : ℕ) (excess : ℝ) 
  (h1 : total_goal = 1000)
  (h2 : nike_cost = 60)
  (h3 : reebok_cost = 35)
  (h4 : nike_sold = 8)
  (h5 : adidas_sold = 6)
  (h6 : reebok_sold = 9)
  (h7 : excess = 65)
  : ∃ (adidas_cost : ℝ), 
    nike_cost * nike_sold + adidas_cost * adidas_sold + reebok_cost * reebok_sold 
    = total_goal + excess ∧ adidas_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_adidas_cost_l281_28160


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l281_28198

/-- Represents the number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- Represents the number of smaller triangles used to construct a large triangle -/
def triangles_per_large : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  let corner_same := num_colors -- All corners same color
  let corner_two_same := num_colors * (num_colors - 1) -- Two corners same, one different
  let corner_all_diff := choose num_colors 3 -- All corners different
  let total_corners := corner_same + corner_two_same + corner_all_diff
  total_corners * num_colors -- Multiply by choices for center triangle

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l281_28198


namespace NUMINAMATH_CALUDE_prob_at_least_two_correct_value_l281_28164

/-- The number of questions Jessica randomly guesses -/
def n : ℕ := 6

/-- The number of possible answers for each question -/
def m : ℕ := 3

/-- The probability of guessing a single question correctly -/
def p : ℚ := 1 / m

/-- The probability of guessing a single question incorrectly -/
def q : ℚ := 1 - p

/-- The probability of getting at least two correct answers out of n randomly guessed questions -/
def prob_at_least_two_correct : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_correct_value : 
  prob_at_least_two_correct = 473 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_correct_value_l281_28164


namespace NUMINAMATH_CALUDE_zebra_chase_time_l281_28111

/-- The time (in hours) it takes for the zebra to catch up with the tiger -/
def catchup_time : ℝ := 6

/-- The speed of the zebra in km/h -/
def zebra_speed : ℝ := 55

/-- The speed of the tiger in km/h -/
def tiger_speed : ℝ := 30

/-- The time (in hours) after which the zebra starts chasing the tiger -/
def chase_start_time : ℝ := 5

theorem zebra_chase_time :
  chase_start_time * tiger_speed + catchup_time * tiger_speed = catchup_time * zebra_speed :=
sorry

end NUMINAMATH_CALUDE_zebra_chase_time_l281_28111


namespace NUMINAMATH_CALUDE_interval_of_increase_l281_28171

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 + 6 * x - 12

-- Theorem stating the interval of increase for f(x)
theorem interval_of_increase (x : ℝ) :
  StrictMonoOn f (Set.Iio (-2) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_interval_of_increase_l281_28171


namespace NUMINAMATH_CALUDE_larger_number_proof_l281_28110

theorem larger_number_proof (a b : ℕ) : 
  (Nat.gcd a b = 25) →
  (Nat.lcm a b = 4550) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  (max a b = 350) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l281_28110


namespace NUMINAMATH_CALUDE_room_area_l281_28129

/-- The area of a rectangular room with length 5 feet and width 2 feet is 10 square feet. -/
theorem room_area : 
  let length : ℝ := 5
  let width : ℝ := 2
  length * width = 10 := by sorry

end NUMINAMATH_CALUDE_room_area_l281_28129


namespace NUMINAMATH_CALUDE_digit2List_2000th_digit_l281_28182

/-- A function that generates the list of positive integers with first digit 2 in increasing order -/
def digit2List : ℕ → ℕ 
| 0 => 2
| n + 1 => 
  let prev := digit2List n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- The number formed by the nth, (n+1)th, and (n+2)th digits in the digit2List -/
def threeDigitNumber (n : ℕ) : ℕ := sorry

theorem digit2List_2000th_digit : threeDigitNumber 1998 = 427 := by sorry

end NUMINAMATH_CALUDE_digit2List_2000th_digit_l281_28182


namespace NUMINAMATH_CALUDE_megan_removed_two_albums_l281_28124

/-- Calculates the number of albums removed from a shopping cart. -/
def albums_removed (initial_albums : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ) : ℕ :=
  initial_albums - (total_songs_bought / songs_per_album)

/-- Proves that Megan removed 2 albums from her shopping cart. -/
theorem megan_removed_two_albums :
  albums_removed 8 7 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_removed_two_albums_l281_28124


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l281_28108

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x₀ : ℝ, x₀^2 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l281_28108


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l281_28106

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 80)
  (h_sum : S = a / (1 - r)) :
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l281_28106


namespace NUMINAMATH_CALUDE_pythagorean_proof_l281_28112

theorem pythagorean_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  b^2 = 13 * (b - a)^2 → a / b = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_proof_l281_28112


namespace NUMINAMATH_CALUDE_matthew_crackers_l281_28199

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 55

/-- The number of cakes Matthew had -/
def cakes : ℕ := 34

/-- The number of friends Matthew gave crackers and cakes to -/
def friends : ℕ := 11

/-- The number of crackers each person ate -/
def crackers_eaten_per_person : ℕ := 2

theorem matthew_crackers :
  (cakes / friends = initial_crackers / friends) ∧
  (friends * crackers_eaten_per_person + friends * (cakes / friends) = initial_crackers) :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l281_28199


namespace NUMINAMATH_CALUDE_inequality_problem_l281_28105

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.log (a*b + 1) ≥ 0) ∧ 
  (Real.sqrt (a + b) ≥ 2) ∧ 
  ¬(∀ (x y : ℝ), x > 0 → y > 0 → x^3 + y^3 ≥ 2*x*y^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l281_28105


namespace NUMINAMATH_CALUDE_extreme_value_at_negative_three_l281_28144

/-- The function f(x) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 3*x - 9

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a + 3

theorem extreme_value_at_negative_three (a : ℝ) :
  (∃ (x : ℝ), f' a x = 0) ∧ f' a (-3) = 0 → a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_negative_three_l281_28144


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l281_28131

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l281_28131


namespace NUMINAMATH_CALUDE_least_digit_sum_multiple_2003_l281_28172

/-- Sum of decimal digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- The least value of S(m) where m is a multiple of 2003 -/
theorem least_digit_sum_multiple_2003 : 
  (∃ m : ℕ, m % 2003 = 0 ∧ S m = 3) ∧ 
  (∀ m : ℕ, m % 2003 = 0 → S m ≥ 3) := by sorry

end NUMINAMATH_CALUDE_least_digit_sum_multiple_2003_l281_28172


namespace NUMINAMATH_CALUDE_max_ab_value_l281_28169

theorem max_ab_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 4 * b = 1) :
  ab ≤ 1/16 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 4 * b₀ = 1 ∧ a₀ * b₀ = 1/16 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l281_28169


namespace NUMINAMATH_CALUDE_total_ways_is_eight_l281_28136

/-- The number of ways an individual can sign up -/
def sign_up_ways : ℕ := 2

/-- The number of individuals signing up -/
def num_individuals : ℕ := 3

/-- The total number of different ways all individuals can sign up -/
def total_ways : ℕ := sign_up_ways ^ num_individuals

/-- Theorem: The total number of different ways all individuals can sign up is 8 -/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_eight_l281_28136


namespace NUMINAMATH_CALUDE_room_width_calculation_l281_28107

theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 28875 →
  rate_per_sqm = 1400 →
  (total_cost / rate_per_sqm) / length = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_room_width_calculation_l281_28107


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l281_28170

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (θ : ℝ) (h1 : d = 12 * Real.sqrt 3) (h2 : θ = π / 3) :
  let r := d / (4 * Real.sqrt 3)
  (4 / 3) * π * r^3 = 288 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l281_28170


namespace NUMINAMATH_CALUDE_otimes_inequality_system_l281_28165

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 2 * b

-- Theorem statement
theorem otimes_inequality_system (a : ℝ) :
  (∀ x : ℝ, x > 6 ↔ (otimes x 3 > 0 ∧ otimes x a > a)) →
  a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_system_l281_28165


namespace NUMINAMATH_CALUDE_embroidery_project_time_l281_28115

/-- Represents the embroidery project details -/
structure EmbroideryProject where
  flower_stitches : ℕ
  flower_speed : ℕ
  unicorn_stitches : ℕ
  unicorn_speed : ℕ
  godzilla_stitches : ℕ
  godzilla_speed : ℕ
  num_flowers : ℕ
  num_unicorns : ℕ
  num_godzilla : ℕ
  break_duration : ℕ
  work_duration : ℕ

/-- Calculates the total time needed for the embroidery project -/
def total_time (project : EmbroideryProject) : ℕ :=
  let total_stitches := project.flower_stitches * project.num_flowers +
                        project.unicorn_stitches * project.num_unicorns +
                        project.godzilla_stitches * project.num_godzilla
  let total_work_time := (total_stitches / project.flower_speed * project.num_flowers +
                          total_stitches / project.unicorn_speed * project.num_unicorns +
                          total_stitches / project.godzilla_speed * project.num_godzilla)
  let num_breaks := total_work_time / project.work_duration
  let total_break_time := num_breaks * project.break_duration
  total_work_time + total_break_time

/-- The main theorem stating the total time for the given embroidery project -/
theorem embroidery_project_time :
  let project : EmbroideryProject := {
    flower_stitches := 60,
    flower_speed := 4,
    unicorn_stitches := 180,
    unicorn_speed := 5,
    godzilla_stitches := 800,
    godzilla_speed := 3,
    num_flowers := 50,
    num_unicorns := 3,
    num_godzilla := 1,
    break_duration := 5,
    work_duration := 30
  }
  total_time project = 1310 := by
  sorry


end NUMINAMATH_CALUDE_embroidery_project_time_l281_28115


namespace NUMINAMATH_CALUDE_boys_at_reunion_l281_28109

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: There are 7 boys at the reunion -/
theorem boys_at_reunion : ∃ n : ℕ, n > 0 ∧ handshakes n = 21 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_at_reunion_l281_28109


namespace NUMINAMATH_CALUDE_exists_valid_triangle_l281_28123

-- Define the necessary structures
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the given elements
variable (p q : Line)
variable (C : Point)
variable (c : ℝ)

-- Define a right triangle
structure RightTriangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (hypotenuse_length : ℝ)

-- Define the conditions for the desired triangle
def is_valid_triangle (t : RightTriangle) : Prop :=
  -- Right angle at C
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0 ∧
  -- Vertex A on line p
  t.A.y = p.slope * t.A.x + p.intercept ∧
  -- Hypotenuse parallel to line q
  (t.A.y - t.C.y) / (t.A.x - t.C.x) = q.slope ∧
  -- Hypotenuse length is c
  t.hypotenuse_length = c ∧
  -- C is the given point
  t.C = C

-- Theorem statement
theorem exists_valid_triangle :
  ∃ (t : RightTriangle), is_valid_triangle p q C c t :=
sorry

end NUMINAMATH_CALUDE_exists_valid_triangle_l281_28123


namespace NUMINAMATH_CALUDE_martha_crayons_l281_28185

def crayons_problem (initial_crayons : ℕ) (total_after_buying : ℕ) : Prop :=
  let lost_crayons := initial_crayons / 2
  let remaining_crayons := initial_crayons - lost_crayons
  let new_set_size := total_after_buying - remaining_crayons
  new_set_size = 20

theorem martha_crayons : crayons_problem 18 29 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayons_l281_28185


namespace NUMINAMATH_CALUDE_problem_solution_l281_28156

theorem problem_solution (x : ℚ) : 
  2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l281_28156


namespace NUMINAMATH_CALUDE_star_properties_l281_28132

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem statement
theorem star_properties :
  (∀ x ∈ T, star x (-1) ≠ x ∨ star (-1) x ≠ x) ∧
  (star 1 (-1/2) = -1 ∧ star (-1/2) 1 = -1) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l281_28132


namespace NUMINAMATH_CALUDE_equilateral_triangle_properties_l281_28184

theorem equilateral_triangle_properties (side : ℝ) (h : side = 20) :
  let height := side * (Real.sqrt 3) / 2
  let half_side := side / 2
  height = 10 * Real.sqrt 3 ∧ half_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_properties_l281_28184


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l281_28121

/-- Given a geometric sequence with common ratio q < 0, prove that a₉S₈ > a₈S₉ -/
theorem geometric_sequence_inequality (a₁ : ℝ) (q : ℝ) (hq : q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let S : ℕ → ℝ := λ n => a₁ * (1 - q^n) / (1 - q)
  (a 9) * (S 8) > (a 8) * (S 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l281_28121


namespace NUMINAMATH_CALUDE_product_of_special_numbers_l281_28178

theorem product_of_special_numbers (m n : ℕ+) 
  (sum_eq : m + n = 20)
  (fraction_sum_eq : (1 : ℚ) / m + (1 : ℚ) / n = 5 / 24) :
  (m * n : ℕ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_numbers_l281_28178


namespace NUMINAMATH_CALUDE_five_points_in_unit_triangle_close_pair_l281_28173

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  is_positive : side_length > 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is inside the triangle
def is_inside_triangle (t : EquilateralTriangle) (p : Point) : Prop :=
  sorry -- Actual implementation would go here

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem five_points_in_unit_triangle_close_pair 
  (t : EquilateralTriangle) 
  (h_side : t.side_length = 1) 
  (points : Fin 5 → Point) 
  (h_inside : ∀ i, is_inside_triangle t (points i)) :
  ∃ i j, i ≠ j ∧ distance (points i) (points j) < 0.5 :=
sorry

end NUMINAMATH_CALUDE_five_points_in_unit_triangle_close_pair_l281_28173


namespace NUMINAMATH_CALUDE_quadratic_relationship_l281_28152

theorem quadratic_relationship (x : ℕ) (z : ℕ) : 
  (x = 1 ∧ z = 5) ∨ 
  (x = 2 ∧ z = 12) ∨ 
  (x = 3 ∧ z = 23) ∨ 
  (x = 4 ∧ z = 38) ∨ 
  (x = 5 ∧ z = 57) → 
  z = 2 * x^2 + x + 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_relationship_l281_28152


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l281_28192

-- Define the points P and Q
def P : ℝ × ℝ := (3, 2)
def Q : ℝ × ℝ := (1, 4)

-- Define the line l as a function ax + by + c = 0
def line_l (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (P Q : ℝ × ℝ) (a b c : ℝ) : Prop :=
  let midpoint := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  line_l a b c midpoint.1 midpoint.2 ∧
  a * (Q.2 - P.2) = b * (P.1 - Q.1)

-- Theorem statement
theorem symmetric_line_equation :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ symmetric_wrt_line P Q a b c ∧ line_l a b c = line_l 1 (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l281_28192


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l281_28179

-- Define the complex number z(a)
def z (a : ℝ) : ℂ := (a - 1) * (a + 2) + (a + 3) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → isPurelyImaginary (z a)) ∧
  ¬(∀ a : ℝ, isPurelyImaginary (z a) → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l281_28179


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l281_28118

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12 -/
theorem two_std_dev_below_mean (d : NormalDistribution) 
    (h1 : d.mean = 15) (h2 : d.std_dev = 1.5) : 
    value_n_std_dev_below d 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l281_28118


namespace NUMINAMATH_CALUDE_gold_value_proof_l281_28158

def gold_problem (legacy_bars : ℕ) (aleena_difference : ℕ) (bar_value : ℕ) : Prop :=
  let aleena_bars := legacy_bars - aleena_difference
  let total_bars := legacy_bars + aleena_bars
  let total_value := total_bars * bar_value
  total_value = 17600

theorem gold_value_proof :
  gold_problem 5 2 2200 := by
  sorry

end NUMINAMATH_CALUDE_gold_value_proof_l281_28158


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l281_28157

/-- The vertex of a parabola y = 3x^2 shifted 2 units left and 3 units up is at (-2,3) -/
theorem shifted_parabola_vertex :
  let f (x : ℝ) := 3 * (x + 2)^2 + 3
  ∃! (a b : ℝ), (∀ x, f x ≥ f a) ∧ f a = b ∧ a = -2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l281_28157


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_reciprocal_sum_equality_condition_l281_28162

theorem min_value_sqrt_sum_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a + b) * (1 / Real.sqrt a + 1 / Real.sqrt b) = 2 * Real.sqrt 2 ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_reciprocal_sum_equality_condition_l281_28162


namespace NUMINAMATH_CALUDE_simplify_powers_of_ten_l281_28195

theorem simplify_powers_of_ten : 
  (10 ^ 0.4) * (10 ^ 0.5) * (10 ^ 0.2) * (10 ^ (-0.6)) * (10 ^ 0.5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_of_ten_l281_28195


namespace NUMINAMATH_CALUDE_smallest_addition_and_quotient_l281_28166

theorem smallest_addition_and_quotient : 
  let n := 897326
  let d := 456
  let x := d - (n % d)
  ∀ y, 0 ≤ y ∧ y < x → ¬(d ∣ (n + y)) ∧
  (d ∣ (n + x)) ∧
  ((n + x) / d = 1968) := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_and_quotient_l281_28166


namespace NUMINAMATH_CALUDE_intersection_and_lines_l281_28194

-- Define the lines
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := x - 3*y + 10 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point
def M : ℝ × ℝ := (-1, 3)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3*x - 4*y + 15 = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4*x + 3*y - 5 = 0

theorem intersection_and_lines :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ (x, y) = M) ∧
  (parallel_line M.1 M.2 ∧ ∀ x y, parallel_line x y → l₃ x y → x = y) ∧
  (perpendicular_line M.1 M.2 ∧ ∀ x y, perpendicular_line x y → l₃ x y → 
    (x - M.1) * 3 + (y - M.2) * (-4) = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l281_28194


namespace NUMINAMATH_CALUDE_prime_value_of_cubic_polynomial_l281_28139

theorem prime_value_of_cubic_polynomial (n : ℕ) (a : ℚ) (b : ℕ) :
  b = n^3 - 4*a*n^2 - 12*n + 144 →
  Nat.Prime b →
  b = 11 := by
  sorry

end NUMINAMATH_CALUDE_prime_value_of_cubic_polynomial_l281_28139


namespace NUMINAMATH_CALUDE_unique_prime_factors_count_l281_28148

theorem unique_prime_factors_count (n : ℕ+) (h : Nat.card (Nat.divisors n) = 12320) :
  Finset.card (Nat.factors n).toFinset = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_factors_count_l281_28148


namespace NUMINAMATH_CALUDE_expand_expressions_l281_28186

theorem expand_expressions (x m n : ℝ) :
  ((-3*x - 5) * (5 - 3*x) = 9*x^2 - 25) ∧
  ((-3*x - 5) * (5 + 3*x) = -9*x^2 - 30*x - 25) ∧
  ((2*m - 3*n + 1) * (2*m + 1 + 3*n) = 4*m^2 + 4*m + 1 - 9*n^2) := by
  sorry

end NUMINAMATH_CALUDE_expand_expressions_l281_28186
