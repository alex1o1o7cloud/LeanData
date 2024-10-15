import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1677_167763

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4)
  (h2 : (2*m + n) / 2 = 5) :
  (m + n) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1677_167763


namespace NUMINAMATH_CALUDE_probability_of_two_positive_roots_l1677_167715

-- Define the interval for a
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + 4*a - 3

-- Define the condition for two positive roots
def has_two_positive_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- Define the probability measure on the interval
noncomputable def probability_measure : MeasureTheory.Measure ℝ :=
  sorry

-- State the theorem
theorem probability_of_two_positive_roots :
  probability_measure {a ∈ interval | has_two_positive_roots a} = 3/8 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_positive_roots_l1677_167715


namespace NUMINAMATH_CALUDE_five_letter_words_count_l1677_167710

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The total number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of five-letter words starting with a vowel and ending with a consonant -/
def num_words : ℕ := num_vowels * num_letters * num_letters * num_letters * num_consonants

theorem five_letter_words_count : num_words = 1844760 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l1677_167710


namespace NUMINAMATH_CALUDE_max_value_constraint_l1677_167772

theorem max_value_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2/2 = 1) :
  a * Real.sqrt (1 + b^2) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1677_167772


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1677_167744

theorem rectangle_circle_area_ratio (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * Real.pi * r) (h2 : l = 2 * w) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l1677_167744


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1677_167734

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 9 * d)
  (h_geom_mean : a k ^ 2 = a 1 * a (2 * k)) :
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l1677_167734


namespace NUMINAMATH_CALUDE_triangle_coloring_theorem_l1677_167743

/-- The number of colors available for coloring the triangle vertices -/
def num_colors : ℕ := 4

/-- The number of vertices in a triangle -/
def num_vertices : ℕ := 3

/-- 
Calculates the number of ways to color the vertices of a triangle
such that no two vertices have the same color
-/
def triangle_coloring_ways : ℕ :=
  num_colors * (num_colors - 1) * (num_colors - 2)

/-- 
Theorem: The number of ways to color the vertices of a triangle
with 4 colors, such that no two vertices have the same color, is 24
-/
theorem triangle_coloring_theorem : 
  triangle_coloring_ways = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_coloring_theorem_l1677_167743


namespace NUMINAMATH_CALUDE_comparison_theorem_l1677_167723

theorem comparison_theorem :
  (5.6 - 7/8 > 4.6) ∧ (638/81 < 271/29) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l1677_167723


namespace NUMINAMATH_CALUDE_ap_has_ten_terms_l1677_167733

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                -- number of terms
  a : ℝ                -- first term
  d : ℝ                -- common difference
  n_even : Even n
  sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 28
  sum_even : (n / 2) * (2 * a + n * d) = 38
  last_first_diff : a + (n - 1) * d - a = 16

/-- Theorem stating that an arithmetic progression with the given properties has 10 terms -/
theorem ap_has_ten_terms (ap : ArithmeticProgression) : ap.n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ap_has_ten_terms_l1677_167733


namespace NUMINAMATH_CALUDE_chess_game_outcome_l1677_167711

theorem chess_game_outcome (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7) :
  let prob_draw := prob_A_not_lose - prob_A_win
  prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by sorry

end NUMINAMATH_CALUDE_chess_game_outcome_l1677_167711


namespace NUMINAMATH_CALUDE_project_hours_difference_l1677_167780

theorem project_hours_difference (total_hours : ℕ) 
  (h1 : total_hours = 216) 
  (kate_hours : ℕ) (pat_hours : ℕ) (mark_hours : ℕ) 
  (h2 : pat_hours = 2 * kate_hours) 
  (h3 : pat_hours * 3 = mark_hours) 
  (h4 : kate_hours + pat_hours + mark_hours = total_hours) : 
  mark_hours - kate_hours = 120 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1677_167780


namespace NUMINAMATH_CALUDE_roads_per_neighborhood_is_four_l1677_167765

/-- The number of roads passing through each neighborhood in a town with the following properties:
  * The town has 10 neighborhoods.
  * Each road has 250 street lights on each opposite side.
  * The total number of street lights in the town is 20000.
-/
def roads_per_neighborhood : ℕ := 4

/-- Theorem stating that the number of roads passing through each neighborhood is 4 -/
theorem roads_per_neighborhood_is_four :
  let neighborhoods : ℕ := 10
  let lights_per_side : ℕ := 250
  let total_lights : ℕ := 20000
  roads_per_neighborhood * neighborhoods * (2 * lights_per_side) = total_lights :=
by sorry

end NUMINAMATH_CALUDE_roads_per_neighborhood_is_four_l1677_167765


namespace NUMINAMATH_CALUDE_y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l1677_167720

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

-- Theorem for the maximum and minimum values of y/x
theorem y_over_x_bounds {x y : ℝ} (h : satisfies_equation x y) (hx : x ≠ 0) :
  y / x ≤ Real.sqrt 3 ∧ y / x ≥ -Real.sqrt 3 :=
sorry

-- Theorem for the maximum and minimum values of y - x
theorem y_minus_x_bounds {x y : ℝ} (h : satisfies_equation x y) :
  y - x ≤ -2 + Real.sqrt 6 ∧ y - x ≥ -2 - Real.sqrt 6 :=
sorry

-- Theorem for the maximum and minimum values of x^2 + y^2
theorem x_squared_plus_y_squared_bounds {x y : ℝ} (h : satisfies_equation x y) :
  x^2 + y^2 ≤ 7 + 4 * Real.sqrt 3 ∧ x^2 + y^2 ≥ 7 - 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_y_over_x_bounds_y_minus_x_bounds_x_squared_plus_y_squared_bounds_l1677_167720


namespace NUMINAMATH_CALUDE_nanoseconds_to_scientific_notation_l1677_167725

/-- Conversion factor from nanoseconds to seconds -/
def nanosecond_to_second : ℝ := 1e-9

/-- The number of nanoseconds we want to convert -/
def nanoseconds : ℝ := 20

/-- The expected result in scientific notation (in seconds) -/
def expected_result : ℝ := 2e-8

theorem nanoseconds_to_scientific_notation :
  nanoseconds * nanosecond_to_second = expected_result := by
  sorry

end NUMINAMATH_CALUDE_nanoseconds_to_scientific_notation_l1677_167725


namespace NUMINAMATH_CALUDE_circle_M_properties_l1677_167732

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define the center of a circle
def is_center (cx cy : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - cx)^2 + (y - cy)^2 = (x - cx)^2 + (y - cy)^2

-- Define a tangent line to a circle
def is_tangent_line (m b : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃! x y, circle x y ∧ y = m*x + b

-- Main theorem
theorem circle_M_properties :
  (is_center (-2) 1 circle_M) ∧
  (∀ m b : ℝ, is_tangent_line m b circle_M ∧ 0 = m*(-3) + b → b = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_M_properties_l1677_167732


namespace NUMINAMATH_CALUDE_special_triangle_properties_l1677_167796

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  side_relation : a = (1/2) * c + b * Real.cos C
  area : (1/2) * a * c * Real.sin ((1/3) * Real.pi) = Real.sqrt 3
  side_b : b = Real.sqrt 13

/-- Properties of the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : 
  t.B = (1/3) * Real.pi ∧ t.a + t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l1677_167796


namespace NUMINAMATH_CALUDE_system_solution_l1677_167750

/-- The system of equations has only two solutions -/
theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1677_167750


namespace NUMINAMATH_CALUDE_range_of_a_squared_minus_2b_l1677_167776

/-- A quadratic function with two real roots in [0, 1] -/
structure QuadraticWithRootsInUnitInterval where
  a : ℝ
  b : ℝ
  has_two_roots_in_unit_interval : ∃ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ 1 ∧ 
    x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- The range of a^2 - 2b for quadratic functions with roots in [0, 1] -/
theorem range_of_a_squared_minus_2b (f : QuadraticWithRootsInUnitInterval) :
  ∃ (z : ℝ), z = f.a^2 - 2*f.b ∧ 0 ≤ z ∧ z ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_squared_minus_2b_l1677_167776


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1677_167705

theorem max_value_on_circle (x y z : ℝ) : 
  x^2 + y^2 = 4 → z = 2*x + y → z ≤ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1677_167705


namespace NUMINAMATH_CALUDE_divisibility_property_l1677_167746

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1677_167746


namespace NUMINAMATH_CALUDE_max_attendance_l1677_167728

/-- Represents the number of students that can attend an event --/
structure EventAttendance where
  boys : ℕ
  girls : ℕ

/-- Represents the capacities of the three auditoriums --/
structure AuditoriumCapacities where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Checks if the attendance satisfies the given conditions --/
def satisfiesConditions (attendance : EventAttendance) (capacities : AuditoriumCapacities) : Prop :=
  -- The ratio of boys to girls is 7:11
  11 * attendance.boys = 7 * attendance.girls
  -- There are 72 more girls than boys
  ∧ attendance.girls = attendance.boys + 72
  -- The total attendance doesn't exceed any individual auditorium's capacity
  ∧ attendance.boys + attendance.girls ≤ capacities.A
  ∧ attendance.boys + attendance.girls ≤ capacities.B
  ∧ attendance.boys + attendance.girls ≤ capacities.C

/-- The main theorem stating the maximum number of students that can attend --/
theorem max_attendance (capacities : AuditoriumCapacities)
    (hA : capacities.A = 180)
    (hB : capacities.B = 220)
    (hC : capacities.C = 150) :
    ∃ (attendance : EventAttendance),
      satisfiesConditions attendance capacities
      ∧ ∀ (other : EventAttendance),
          satisfiesConditions other capacities →
          attendance.boys + attendance.girls ≥ other.boys + other.girls
      ∧ attendance.boys + attendance.girls = 324 :=
  sorry


end NUMINAMATH_CALUDE_max_attendance_l1677_167728


namespace NUMINAMATH_CALUDE_min_cost_notebooks_l1677_167752

/-- Represents the unit price of type A notebooks -/
def price_A : ℝ := 11

/-- Represents the unit price of type B notebooks -/
def price_B : ℝ := price_A + 1

/-- Represents the total number of notebooks to be purchased -/
def total_notebooks : ℕ := 100

/-- Represents the constraint on the quantity of type B notebooks -/
def type_B_constraint (a : ℕ) : Prop := total_notebooks - a ≤ 3 * a

/-- Represents the cost function for purchasing notebooks -/
def cost_function (a : ℕ) : ℝ := price_A * a + price_B * (total_notebooks - a)

/-- Theorem stating that the minimum cost for purchasing 100 notebooks is $1100 -/
theorem min_cost_notebooks : 
  ∃ (a : ℕ), a ≤ total_notebooks ∧ 
  type_B_constraint a ∧ 
  (∀ (b : ℕ), b ≤ total_notebooks → type_B_constraint b → cost_function a ≤ cost_function b) ∧
  cost_function a = 1100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_notebooks_l1677_167752


namespace NUMINAMATH_CALUDE_geq_one_necessary_not_sufficient_for_gt_one_l1677_167788

theorem geq_one_necessary_not_sufficient_for_gt_one :
  (∀ x : ℝ, x > 1 → x ≥ 1) ∧
  (∃ x : ℝ, x ≥ 1 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_geq_one_necessary_not_sufficient_for_gt_one_l1677_167788


namespace NUMINAMATH_CALUDE_fruit_salad_count_l1677_167716

def total_fruit_salads (alaya_salads : ℕ) (angel_multiplier : ℕ) : ℕ :=
  alaya_salads + angel_multiplier * alaya_salads

theorem fruit_salad_count :
  total_fruit_salads 200 2 = 600 :=
by sorry

end NUMINAMATH_CALUDE_fruit_salad_count_l1677_167716


namespace NUMINAMATH_CALUDE_inequality_proof_l1677_167779

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (3 * x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1677_167779


namespace NUMINAMATH_CALUDE_triangle_transformation_l1677_167717

-- Define the initial triangle
def initial_triangle : List (ℝ × ℝ) := [(0, 0), (1, 0), (0, 1)]

-- Define the transformation functions
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def translate_right (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2, p.2)

-- Define the composite transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_right (reflect_x_axis (rotate_180 p))

-- Theorem statement
theorem triangle_transformation :
  List.map transform initial_triangle = [(2, 0), (1, 0), (2, 1)] := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_l1677_167717


namespace NUMINAMATH_CALUDE_extreme_points_property_l1677_167798

theorem extreme_points_property (a : ℝ) (f : ℝ → ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1/2 →
  (∀ x, f x = x * (Real.log x - a * x)) →
  x₁ < x₂ →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₁ - ε) (x₁ + ε), f x ≤ f x₁) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₂ - ε) (x₂ + ε), f x ≤ f x₂) →
  f x₁ < 0 ∧ f x₂ > -1/2 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_property_l1677_167798


namespace NUMINAMATH_CALUDE_log_inequality_l1677_167792

theorem log_inequality (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = Real.log x / Real.log 3) ∧ f a > f 2) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1677_167792


namespace NUMINAMATH_CALUDE_triangle_formation_l1677_167794

/-- Given two sticks of lengths 3 and 5, determine if a third stick of length l can form a triangle with them. -/
def can_form_triangle (l : ℝ) : Prop :=
  l > 0 ∧ l + 3 > 5 ∧ l + 5 > 3 ∧ 3 + 5 > l

theorem triangle_formation :
  can_form_triangle 5 ∧
  ¬can_form_triangle 2 ∧
  ¬can_form_triangle 8 ∧
  ¬can_form_triangle 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1677_167794


namespace NUMINAMATH_CALUDE_max_label_outcomes_l1677_167791

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of times a ball is drawn -/
def num_draws : ℕ := 3

/-- The total number of possible outcomes when drawing num_draws times from num_balls balls -/
def total_outcomes : ℕ := num_balls ^ num_draws

/-- The number of outcomes that don't include the maximum label -/
def outcomes_without_max : ℕ := 8

/-- Theorem: The number of ways to draw a maximum label of 3 when drawing 3 balls 
    (with replacement) from a box containing balls labeled 1, 2, and 3 is equal to 19 -/
theorem max_label_outcomes : 
  total_outcomes - outcomes_without_max = 19 := by sorry

end NUMINAMATH_CALUDE_max_label_outcomes_l1677_167791


namespace NUMINAMATH_CALUDE_pencils_added_l1677_167795

theorem pencils_added (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 41)
  (h2 : final_pencils = 71) :
  final_pencils - initial_pencils = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencils_added_l1677_167795


namespace NUMINAMATH_CALUDE_train_platform_time_l1677_167760

theorem train_platform_time (l t T : ℝ) (v : ℝ) (h1 : v > 0) (h2 : l > 0) (h3 : t > 0) :
  v = l / t →
  v = (l + 2.5 * l) / T →
  T = 3.5 * t := by
sorry

end NUMINAMATH_CALUDE_train_platform_time_l1677_167760


namespace NUMINAMATH_CALUDE_job_completion_proof_l1677_167789

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 36

/-- The number of additional machines added -/
def additional_machines : ℕ := 5

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 27

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 20

theorem job_completion_proof :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines) / reduced_days :=
by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l1677_167789


namespace NUMINAMATH_CALUDE_unique_square_sum_l1677_167742

theorem unique_square_sum : ∃! x : ℕ+, 
  (∃ m : ℕ+, (x : ℕ) + 100 = m^2) ∧ 
  (∃ n : ℕ+, (x : ℕ) + 168 = n^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_sum_l1677_167742


namespace NUMINAMATH_CALUDE_mod_seven_difference_l1677_167700

theorem mod_seven_difference (n : ℕ) : (47^824 - 25^824) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_difference_l1677_167700


namespace NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1677_167738

/-- Given algebraic expressions A and B -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B = 5xy - 2x + 2y -/
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = 5 * x * y - 2 * x + 2 * y := by sorry

/-- Theorem 2: A - 2B = -7 when x = -1 and y = 3 -/
theorem A_minus_2B_specific : A (-1) 3 - 2 * B (-1) 3 = -7 := by sorry

/-- Theorem 3: y = 2/5 when A - 2B is independent of x -/
theorem y_value_when_independent (y : ℝ) :
  (∀ x, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by sorry

end NUMINAMATH_CALUDE_A_minus_2B_A_minus_2B_specific_y_value_when_independent_l1677_167738


namespace NUMINAMATH_CALUDE_ad_eq_bc_necessary_not_sufficient_l1677_167774

/-- A sequence of four non-zero real numbers forms a geometric sequence -/
def IsGeometricSequence (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- The condition ad=bc for four non-zero real numbers -/
def AdEqualsBc (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ a * d = b * c

theorem ad_eq_bc_necessary_not_sufficient :
  (∀ a b c d : ℝ, IsGeometricSequence a b c d → AdEqualsBc a b c d) ∧
  (∃ a b c d : ℝ, AdEqualsBc a b c d ∧ ¬IsGeometricSequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_ad_eq_bc_necessary_not_sufficient_l1677_167774


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1677_167727

theorem set_intersection_problem (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1677_167727


namespace NUMINAMATH_CALUDE_sally_balloons_l1677_167704

/-- 
Given that Sally has x orange balloons initially, finds 2 more orange balloons,
and ends up with 11 orange balloons in total, prove that x = 9.
-/
theorem sally_balloons (x : ℝ) : x + 2 = 11 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_sally_balloons_l1677_167704


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l1677_167773

/-- Represents a number in base 8 --/
def BaseEight : Type := Nat

/-- Converts a base 8 number to its decimal representation --/
def to_decimal (n : BaseEight) : Nat := sorry

/-- Converts a decimal number to its base 8 representation --/
def from_decimal (n : Nat) : BaseEight := sorry

/-- Subtracts two base 8 numbers --/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

theorem base_eight_subtraction :
  base_eight_sub (from_decimal 4765) (from_decimal 2314) = from_decimal 2447 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l1677_167773


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1677_167707

theorem unique_solution_to_equation : ∃! t : ℝ, 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) + 2^t = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1677_167707


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l1677_167745

/-- Given x = (3 + 2√2)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + 2 * Real.sqrt 2) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l1677_167745


namespace NUMINAMATH_CALUDE_binomial_expansion_equality_l1677_167731

theorem binomial_expansion_equality (a b : ℝ) (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k < n ∧
    (Nat.choose n 0) * a^n = (Nat.choose n 2) * a^(n-2) * b^2) →
  a^2 = n * (n - 1) * b :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_equality_l1677_167731


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1677_167721

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1677_167721


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1677_167785

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + 2*a > 4 ∧ 2*x - b < 5) ↔ (0 < x ∧ x < 2)) →
  (a + b)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1677_167785


namespace NUMINAMATH_CALUDE_davids_biology_marks_l1677_167759

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def average_marks : ℕ := 93
def total_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * total_subjects
  all_subjects_total - known_subjects_total = 95 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l1677_167759


namespace NUMINAMATH_CALUDE_student_weight_l1677_167722

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 116) :
  student_weight = 79 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l1677_167722


namespace NUMINAMATH_CALUDE_solution_set_l1677_167787

theorem solution_set (x y z : ℝ) : 
  x = (4 * z^2) / (1 + 4 * z^2) ∧
  y = (4 * x^2) / (1 + 4 * x^2) ∧
  z = (4 * y^2) / (1 + 4 * y^2) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l1677_167787


namespace NUMINAMATH_CALUDE_olivias_cookie_baggies_l1677_167751

def cookies_per_baggie : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

theorem olivias_cookie_baggies :
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cookie_baggies_l1677_167751


namespace NUMINAMATH_CALUDE_fraction_simplification_l1677_167762

theorem fraction_simplification 
  (x y z : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (hyz : y - z / x ≠ 0) : 
  (x + z / y) / (y + z / x) = x / y :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1677_167762


namespace NUMINAMATH_CALUDE_product_factor_adjustment_l1677_167724

theorem product_factor_adjustment (a b c : ℝ) (h1 : a * b = c) (h2 : a / 100 * (b * 100) = c) : 
  b * 100 = b * 100 := by sorry

end NUMINAMATH_CALUDE_product_factor_adjustment_l1677_167724


namespace NUMINAMATH_CALUDE_candy_distribution_l1677_167729

/-- The number of distinct pieces of candy --/
def n : ℕ := 8

/-- The number of bags --/
def k : ℕ := 3

/-- The number of ways to distribute n distinct objects into k groups,
    where each group must have at least one object --/
def distribute_distinct (n k : ℕ) : ℕ :=
  (n - k + 1).choose (k - 1) * n.factorial

theorem candy_distribution :
  distribute_distinct n k = 846720 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1677_167729


namespace NUMINAMATH_CALUDE_shopping_expenditure_l1677_167769

theorem shopping_expenditure (x : ℝ) 
  (emma_spent : x > 0)
  (elsa_spent : ℝ → ℝ)
  (elizabeth_spent : ℝ → ℝ)
  (elsa_condition : elsa_spent x = 2 * x)
  (elizabeth_condition : elizabeth_spent x = 4 * elsa_spent x)
  (total_spent : x + elsa_spent x + elizabeth_spent x = 638) :
  x = 58 := by
sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l1677_167769


namespace NUMINAMATH_CALUDE_dormitory_students_l1677_167739

theorem dormitory_students (T : ℝ) (h1 : T > 0) : 
  let first_year := T / 2
  let second_year := T / 2
  let first_year_undeclared := (4 / 5) * first_year
  let first_year_declared := first_year - first_year_undeclared
  let second_year_declared := 4 * first_year_declared
  let second_year_undeclared := second_year - second_year_declared
  second_year_undeclared / T = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_dormitory_students_l1677_167739


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1677_167771

/-- An arithmetic sequence with index starting from 1 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1677_167771


namespace NUMINAMATH_CALUDE_am_gm_inequality_for_two_l1677_167775

theorem am_gm_inequality_for_two (x : ℝ) (hx : x > 0) :
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_for_two_l1677_167775


namespace NUMINAMATH_CALUDE_potassium_count_in_compound_l1677_167766

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  k : ℕ  -- number of Potassium atoms
  cr : ℕ -- number of Chromium atoms
  o : ℕ  -- number of Oxygen atoms

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (count : AtomCount) (k_weight cr_weight o_weight : ℝ) : ℝ :=
  count.k * k_weight + count.cr * cr_weight + count.o * o_weight

/-- Theorem stating that a compound with 2 Chromium atoms, 7 Oxygen atoms, 
    and a total molecular weight of 296 g/mol must contain 2 Potassium atoms -/
theorem potassium_count_in_compound :
  ∀ (count : AtomCount),
    count.cr = 2 →
    count.o = 7 →
    molecularWeight count 39.1 52.0 16.0 = 296 →
    count.k = 2 := by
  sorry

end NUMINAMATH_CALUDE_potassium_count_in_compound_l1677_167766


namespace NUMINAMATH_CALUDE_sequence_properties_l1677_167747

def a (n : ℕ+) : ℚ := (9 * n^2 - 9 * n + 2) / (9 * n^2 - 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (∀ n : ℕ+, a n ≠ 99 / 100) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) ∧
  (∃! n : ℕ+, 1 / 3 < a n ∧ a n < 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1677_167747


namespace NUMINAMATH_CALUDE_candy_cost_theorem_l1677_167753

def candy_problem (caramel_price : ℚ) : Prop :=
  let candy_bar_price := 2 * caramel_price
  let cotton_candy_price := 2 * candy_bar_price
  6 * candy_bar_price + 3 * caramel_price + cotton_candy_price = 57

theorem candy_cost_theorem : candy_problem 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_theorem_l1677_167753


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1677_167799

theorem sum_of_fourth_powers (a : ℝ) (h : (a + 1/a)^4 = 16) : a^4 + 1/a^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1677_167799


namespace NUMINAMATH_CALUDE_smallest_possible_a_l1677_167797

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (29 * ↑x)) :
  ∃ a_min : ℝ, a_min = 10 * Real.pi - 29 ∧ 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (29 * ↑x)) → a_min ≤ a') :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l1677_167797


namespace NUMINAMATH_CALUDE_light_bulb_conditional_probability_l1677_167736

theorem light_bulb_conditional_probability 
  (p_3000 : ℝ) 
  (p_4500 : ℝ) 
  (h1 : p_3000 = 0.8) 
  (h2 : p_4500 = 0.2) 
  (h3 : p_3000 ≠ 0) : 
  p_4500 / p_3000 = 0.25 := by
sorry

end NUMINAMATH_CALUDE_light_bulb_conditional_probability_l1677_167736


namespace NUMINAMATH_CALUDE_farm_feet_count_l1677_167748

/-- Given a farm with hens and cows, prove the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 50 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 = 144) :=
by
  sorry

#check farm_feet_count

end NUMINAMATH_CALUDE_farm_feet_count_l1677_167748


namespace NUMINAMATH_CALUDE_possible_winning_scores_for_A_l1677_167703

/-- Represents the outcome of a single question for a team -/
inductive QuestionOutcome
  | Correct
  | Incorrect
  | NoBuzz

/-- Calculates the score for a single question based on the outcome -/
def scoreQuestion (outcome : QuestionOutcome) : Int :=
  match outcome with
  | QuestionOutcome.Correct => 1
  | QuestionOutcome.Incorrect => -1
  | QuestionOutcome.NoBuzz => 0

/-- Calculates the total score for a team based on their outcomes for three questions -/
def calculateScore (q1 q2 q3 : QuestionOutcome) : Int :=
  scoreQuestion q1 + scoreQuestion q2 + scoreQuestion q3

/-- Defines a winning condition for team A -/
def teamAWins (scoreA scoreB : Int) : Prop :=
  scoreA > scoreB

/-- The main theorem stating the possible winning scores for team A -/
theorem possible_winning_scores_for_A :
  ∀ (q1A q2A q3A q1B q2B q3B : QuestionOutcome),
    let scoreA := calculateScore q1A q2A q3A
    let scoreB := calculateScore q1B q2B q3B
    teamAWins scoreA scoreB →
    (scoreA = -1 ∨ scoreA = 0 ∨ scoreA = 1 ∨ scoreA = 3) :=
  sorry


end NUMINAMATH_CALUDE_possible_winning_scores_for_A_l1677_167703


namespace NUMINAMATH_CALUDE_min_weighings_is_two_l1677_167737

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A strategy for finding the real medal -/
def Strategy := List WeighResult → Nat

/-- The total number of medals -/
def totalMedals : Nat := 9

/-- The number of real medals -/
def realMedals : Nat := 1

/-- A weighing operation that compares two sets of medals -/
def weigh (leftSet rightSet : List Nat) : WeighResult := sorry

/-- Checks if a strategy correctly identifies the real medal -/
def isValidStrategy (s : Strategy) : Prop := sorry

/-- The minimum number of weighings required to find the real medal -/
def minWeighings : Nat := sorry

theorem min_weighings_is_two :
  minWeighings = 2 := by sorry

end NUMINAMATH_CALUDE_min_weighings_is_two_l1677_167737


namespace NUMINAMATH_CALUDE_cube_tower_surface_area_l1677_167730

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.side ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.side ^ 2

/-- Represents the tower of cubes -/
def CubeTower : List Cube := [
  { side := 8 },
  { side := 7 },
  { side := 6 },
  { side := 5 },
  { side := 4 },
  { side := 3 },
  { side := 2 },
  { side := 1 }
]

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (c : Cube) (isBottom : Bool) : ℕ :=
  if isBottom then
    5 * c.side ^ 2  -- 5 visible faces for bottom cube
  else if c.side = 1 then
    5 * c.side ^ 2  -- 5 visible faces for top cube
  else
    4 * c.side ^ 2  -- 4 visible faces for other cubes (3 full + 2 partial = 4)

/-- Calculates the total visible surface area of the cube tower -/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ :=
  let rec aux (cubes : List Cube) (acc : ℕ) (isFirst : Bool) : ℕ :=
    match cubes with
    | [] => acc
    | c :: rest => aux rest (acc + visibleSurfaceArea c isFirst) false
  aux tower 0 true

/-- The main theorem stating that the total visible surface area of the cube tower is 945 -/
theorem cube_tower_surface_area :
  totalVisibleSurfaceArea CubeTower = 945 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_cube_tower_surface_area_l1677_167730


namespace NUMINAMATH_CALUDE_arithmetic_cube_reciprocal_roots_l1677_167761

theorem arithmetic_cube_reciprocal_roots :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x = (abs x) ^ (1/2)) →
  (∀ x : ℝ, x > 0 → (x ^ (1/3)) ^ 3 = x) →
  (∀ x : ℝ, x ≠ 0 → x * (1/x) = 1) →
  (Real.sqrt ((-81)^2) = 9) ∧
  ((1/27) ^ (1/3) = 1/3) ∧
  (1 / Real.sqrt 2 = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_cube_reciprocal_roots_l1677_167761


namespace NUMINAMATH_CALUDE_lawyer_percentage_l1677_167740

theorem lawyer_percentage (total_members : ℝ) (h1 : total_members > 0) :
  let women_percentage : ℝ := 0.80
  let woman_lawyer_prob : ℝ := 0.32
  let women_lawyers_percentage : ℝ := woman_lawyer_prob / women_percentage
  women_lawyers_percentage = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_lawyer_percentage_l1677_167740


namespace NUMINAMATH_CALUDE_trivia_team_score_l1677_167714

theorem trivia_team_score (total_members : ℕ) (absent_members : ℕ) (total_points : ℕ) :
  total_members = 14 →
  absent_members = 7 →
  total_points = 35 →
  (total_points / (total_members - absent_members) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1677_167714


namespace NUMINAMATH_CALUDE_linear_functions_product_sign_l1677_167781

theorem linear_functions_product_sign (a b c d : ℝ) :
  b < 0 →
  d < 0 →
  ((a > 0 ∧ c < 0) ∨ (a < 0 ∧ c > 0)) →
  a * b * c * d < 0 := by
sorry

end NUMINAMATH_CALUDE_linear_functions_product_sign_l1677_167781


namespace NUMINAMATH_CALUDE_total_coins_is_660_l1677_167764

/-- The number of coins Jayden received -/
def jayden_coins : ℕ := 300

/-- The additional coins Jason received compared to Jayden -/
def jason_extra_coins : ℕ := 60

/-- The total number of coins given to both boys -/
def total_coins : ℕ := jayden_coins + (jayden_coins + jason_extra_coins)

/-- Theorem stating that the total number of coins given to both boys is 660 -/
theorem total_coins_is_660 : total_coins = 660 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_660_l1677_167764


namespace NUMINAMATH_CALUDE_penny_socks_l1677_167709

def sock_problem (initial_amount : ℕ) (sock_cost : ℕ) (hat_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  ∃ (num_socks : ℕ), 
    initial_amount = sock_cost * num_socks + hat_cost + remaining_amount

theorem penny_socks : sock_problem 20 2 7 5 → ∃ (num_socks : ℕ), num_socks = 4 := by
  sorry

end NUMINAMATH_CALUDE_penny_socks_l1677_167709


namespace NUMINAMATH_CALUDE_fraction_simplification_l1677_167793

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^6 + 8*y^3 + 16) / (y^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1677_167793


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1677_167718

theorem absolute_value_simplification : |(-4^3 + 5^2 - 6)| = 45 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1677_167718


namespace NUMINAMATH_CALUDE_total_shaded_area_l1677_167702

/-- Represents the fraction of area shaded at each level of division -/
def shaded_fraction : ℚ := 1 / 4

/-- Represents the ratio between successive terms in the geometric series -/
def common_ratio : ℚ := 1 / 16

/-- Theorem stating that the total shaded area is 4/15 -/
theorem total_shaded_area :
  (shaded_fraction / (1 - common_ratio) : ℚ) = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_total_shaded_area_l1677_167702


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1677_167706

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (0 < a ∧ a < b → (1 / a > 1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a > 1 / b) → (0 < a ∧ a < b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1677_167706


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1677_167758

theorem smallest_m_for_integral_solutions :
  let has_integral_solutions (m : ℤ) := ∃ x y : ℤ, 10 * x^2 - m * x + 780 = 0 ∧ 10 * y^2 - m * y + 780 = 0 ∧ x ≠ y
  ∀ m : ℤ, m > 0 → has_integral_solutions m → m ≥ 190 ∧
  has_integral_solutions 190 :=
by sorry


end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l1677_167758


namespace NUMINAMATH_CALUDE_polynomial_equality_l1677_167749

theorem polynomial_equality (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) → 
  a - n + k = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1677_167749


namespace NUMINAMATH_CALUDE_employed_females_percentage_l1677_167783

/-- Given the employment statistics of town X, calculate the percentage of employed females among all employed people. -/
theorem employed_females_percentage (total_employed : ℝ) (employed_males : ℝ) 
  (h1 : total_employed = 60) 
  (h2 : employed_males = 48) : 
  (total_employed - employed_males) / total_employed * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l1677_167783


namespace NUMINAMATH_CALUDE_least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l1677_167754

theorem least_11_heavy_three_digit : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 106 → n % 11 ≤ 6 :=
by sorry

theorem is_11_heavy_106 : 106 % 11 > 6 :=
by sorry

theorem least_11_heavy_three_digit_is_106 : 
  ∀ n : ℕ, 100 ≤ n ∧ n % 11 > 6 → n ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l1677_167754


namespace NUMINAMATH_CALUDE_beth_book_collection_l1677_167778

theorem beth_book_collection (novels_percent : Real) (graphic_novels : Nat) (comic_books_percent : Real) :
  novels_percent = 0.65 →
  comic_books_percent = 0.2 →
  graphic_novels = 18 →
  ∃ (total_books : Nat), 
    (novels_percent + comic_books_percent + (graphic_novels : Real) / total_books) = 1 ∧
    total_books = 120 := by
  sorry

end NUMINAMATH_CALUDE_beth_book_collection_l1677_167778


namespace NUMINAMATH_CALUDE_three_geometric_sequences_l1677_167786

/-- An arithmetic sequence starting with 1 -/
structure ArithmeticSequence :=
  (d : ℝ)
  (a₁ : ℝ := 1 + d)
  (a₂ : ℝ := 1 + 2*d)
  (a₃ : ℝ := 1 + 3*d)
  (positive : 0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃)

/-- A function that counts the number of geometric sequences that can be formed from 1 and the terms of an arithmetic sequence -/
def countGeometricSequences (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 3 geometric sequences -/
theorem three_geometric_sequences (seq : ArithmeticSequence) : 
  countGeometricSequences seq = 3 :=
sorry

end NUMINAMATH_CALUDE_three_geometric_sequences_l1677_167786


namespace NUMINAMATH_CALUDE_squirrel_acorns_at_spring_l1677_167768

def calculate_acorns_at_spring (initial_stash : ℕ) 
  (first_month_percent second_month_percent third_month_percent : ℚ)
  (first_month_taken second_month_taken third_month_taken : ℚ)
  (first_month_found second_month_lost third_month_found : ℤ) : ℚ :=
  let first_month := (initial_stash : ℚ) * first_month_percent * (1 - first_month_taken) + first_month_found
  let second_month := (initial_stash : ℚ) * second_month_percent * (1 - second_month_taken) - second_month_lost
  let third_month := (initial_stash : ℚ) * third_month_percent * (1 - third_month_taken) + third_month_found
  first_month + second_month + third_month

theorem squirrel_acorns_at_spring :
  calculate_acorns_at_spring 500 (2/5) (3/10) (3/10) (1/5) (1/4) (3/20) 15 10 20 = 425 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_at_spring_l1677_167768


namespace NUMINAMATH_CALUDE_difference_calculation_l1677_167735

theorem difference_calculation (total : ℝ) (h : total = 8000) : 
  (1 / 10 : ℝ) * total - (1 / 20 : ℝ) * (1 / 100 : ℝ) * total = 796 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l1677_167735


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1677_167784

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0,
    if 3 and -2 are roots of the equation, then (b+c)/a = -7 -/
theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 3^3 + b * 3^2 + c * 3 + d = 0)
  (h2 : a * (-2)^3 + b * (-2)^2 + c * (-2) + d = 0) :
  (b + c) / a = -7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1677_167784


namespace NUMINAMATH_CALUDE_roper_lawn_cut_area_l1677_167790

/-- Calculates the average area of grass cut per month for a rectangular lawn --/
def average_area_cut_per_month (length width : ℝ) (cuts_per_month_high cuts_per_month_low : ℕ) (months_high months_low : ℕ) : ℝ :=
  let lawn_area := length * width
  let total_cuts_per_year := cuts_per_month_high * months_high + cuts_per_month_low * months_low
  let average_cuts_per_month := total_cuts_per_year / 12
  lawn_area * average_cuts_per_month

/-- Theorem stating that the average area of grass cut per month for Mr. Roper's lawn is 14175 square meters --/
theorem roper_lawn_cut_area :
  average_area_cut_per_month 45 35 15 3 6 6 = 14175 := by sorry

end NUMINAMATH_CALUDE_roper_lawn_cut_area_l1677_167790


namespace NUMINAMATH_CALUDE_a_closed_form_l1677_167757

def a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (2 * a n + 6) / (a n + 1)

theorem a_closed_form (n : ℕ) :
  a n = (3 * 4^(n+1) + 2 * (-1)^(n+1)) / (4^(n+1) + (-1)^n) := by
  sorry

end NUMINAMATH_CALUDE_a_closed_form_l1677_167757


namespace NUMINAMATH_CALUDE_remainder_after_adding_4032_l1677_167756

theorem remainder_after_adding_4032 (m : ℤ) (h : m % 8 = 3) :
  (m + 4032) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_4032_l1677_167756


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1677_167708

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 2662) :
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1677_167708


namespace NUMINAMATH_CALUDE_f_three_fourths_equals_three_l1677_167713

-- Define g(x)
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f(g(x))
noncomputable def f (y : ℝ) : ℝ :=
  if y ≠ 1 then (1 - (1 - y)) / (1 - y) else 0

-- Theorem statement
theorem f_three_fourths_equals_three : f (3/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_three_fourths_equals_three_l1677_167713


namespace NUMINAMATH_CALUDE_multiples_2_3_not_5_l1677_167719

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_2_3_not_5 (max : ℕ) (h : max = 200) :
  (count_multiples 2 max + count_multiples 3 max - count_multiples 6 max) -
  (count_multiples 10 max + count_multiples 15 max - count_multiples 30 max) = 107 :=
by sorry

end NUMINAMATH_CALUDE_multiples_2_3_not_5_l1677_167719


namespace NUMINAMATH_CALUDE_sqrt_of_negative_six_squared_l1677_167782

theorem sqrt_of_negative_six_squared (x : ℝ) : Real.sqrt ((-6)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_six_squared_l1677_167782


namespace NUMINAMATH_CALUDE_find_other_number_l1677_167755

theorem find_other_number (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 21)
  (h3 : a = 210) :
  b = 462 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l1677_167755


namespace NUMINAMATH_CALUDE_highest_numbered_street_l1677_167767

/-- Represents the length of Apple Street in meters -/
def street_length : ℝ := 3200

/-- Represents the distance between intersecting streets in meters -/
def intersection_distance : ℝ := 200

/-- Represents the number of non-numbered streets (Peach and Cherry) -/
def non_numbered_streets : ℕ := 2

/-- Theorem stating that the highest-numbered street is the 14th street -/
theorem highest_numbered_street :
  ⌊street_length / intersection_distance⌋ - non_numbered_streets = 14 := by
  sorry


end NUMINAMATH_CALUDE_highest_numbered_street_l1677_167767


namespace NUMINAMATH_CALUDE_solution_in_quadrant_II_l1677_167701

theorem solution_in_quadrant_II (k : ℝ) :
  (∃ x y : ℝ, 2 * x + y = 6 ∧ k * x - y = 4 ∧ x < 0 ∧ y > 0) ↔ k < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_II_l1677_167701


namespace NUMINAMATH_CALUDE_unique_triple_l1677_167777

theorem unique_triple : 
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    x^2 + y - z = 100 ∧ 
    x + y^2 - z = 124 ∧
    x = 12 ∧ y = 13 ∧ z = 57 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1677_167777


namespace NUMINAMATH_CALUDE_range_of_p_l1677_167741

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 10

-- Define set A
def A : Set ℝ := {x | f' x ≤ 0}

-- Define set B
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

-- Theorem statement
theorem range_of_p (p : ℝ) : A ∪ B p = A → p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l1677_167741


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1677_167712

/-- Given a cubic function f and two points (a, f(a)) and (b, f(b)), prove that a + b = -2 --/
theorem sum_of_roots_cubic (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x) →
  f a = 1 →
  f b = -9 →
  a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1677_167712


namespace NUMINAMATH_CALUDE_lightbulb_most_suitable_l1677_167726

/-- Represents a survey option --/
inductive SurveyOption
  | SecurityCheck
  | ClassmateExercise
  | JobInterview
  | LightbulbLifespan

/-- Defines what makes a survey suitable for sampling --/
def suitableForSampling (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.SecurityCheck => false
  | SurveyOption.ClassmateExercise => false
  | SurveyOption.JobInterview => false
  | SurveyOption.LightbulbLifespan => true

/-- Theorem stating that the lightbulb lifespan survey is most suitable for sampling --/
theorem lightbulb_most_suitable :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.LightbulbLifespan →
    suitableForSampling SurveyOption.LightbulbLifespan ∧
    ¬(suitableForSampling option) :=
by
  sorry

#check lightbulb_most_suitable

end NUMINAMATH_CALUDE_lightbulb_most_suitable_l1677_167726


namespace NUMINAMATH_CALUDE_enriques_commission_l1677_167770

/-- Represents the commission rate as a real number between 0 and 1 -/
def commission_rate : ℝ := 0.15

/-- Represents the number of suits sold -/
def suits_sold : ℕ := 2

/-- Represents the price of each suit in dollars -/
def suit_price : ℝ := 700.00

/-- Represents the number of shirts sold -/
def shirts_sold : ℕ := 6

/-- Represents the price of each shirt in dollars -/
def shirt_price : ℝ := 50.00

/-- Represents the number of loafers sold -/
def loafers_sold : ℕ := 2

/-- Represents the price of each pair of loafers in dollars -/
def loafer_price : ℝ := 150.00

/-- Calculates the total sales amount -/
def total_sales : ℝ := 
  suits_sold * suit_price + shirts_sold * shirt_price + loafers_sold * loafer_price

/-- Theorem: Enrique's commission is $300.00 -/
theorem enriques_commission : commission_rate * total_sales = 300.00 := by
  sorry

end NUMINAMATH_CALUDE_enriques_commission_l1677_167770
