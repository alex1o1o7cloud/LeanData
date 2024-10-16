import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l874_87489

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 82 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l874_87489


namespace NUMINAMATH_CALUDE_periodic_function_phase_shift_l874_87460

theorem periodic_function_phase_shift (f : ℝ → ℝ) (ω φ : ℝ) :
  (ω > 0) →
  (-π / 2 < φ) →
  (φ < π / 2) →
  (∀ x : ℝ, f x = 2 * Real.sin (ω * x + φ)) →
  (∀ x : ℝ, f (x + π / 6) = f (x - π / 6)) →
  (∀ x : ℝ, f (5 * π / 18 + x) = f (5 * π / 18 - x)) →
  φ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_phase_shift_l874_87460


namespace NUMINAMATH_CALUDE_ratio_problem_l874_87445

theorem ratio_problem (a b c d : ℚ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l874_87445


namespace NUMINAMATH_CALUDE_reflect_point_1_2_l874_87499

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The theorem states that reflecting the point (1,2) across the x-axis results in (1,-2) -/
theorem reflect_point_1_2 : reflect_x (1, 2) = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflect_point_1_2_l874_87499


namespace NUMINAMATH_CALUDE_inequality_solution_set_abs_b_greater_than_two_l874_87463

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part I
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 :=
sorry

-- Theorem for part II
theorem abs_b_greater_than_two (a b : ℝ) :
  |a| > 1 → f (a * b) > |a| * f (b / a) → |b| > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_abs_b_greater_than_two_l874_87463


namespace NUMINAMATH_CALUDE_triangle_height_l874_87453

theorem triangle_height (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 3 ∧ BC = Real.sqrt 13 ∧ AC = 4 →
  ∃ D : ℝ × ℝ, 
    (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) = 0 ∧
    Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 3/2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l874_87453


namespace NUMINAMATH_CALUDE_max_four_digit_sum_l874_87400

def A (s n k : ℕ) : ℕ :=
  if n = 1 then
    if 1 ≤ s ∧ s ≤ k then 1 else 0
  else if s < n then 0
  else if k = 0 then 0
  else A (s - k) (n - 1) (k - 1) + A s n (k - 1)

theorem max_four_digit_sum :
  (∀ s, s ≠ 20 → A s 4 9 ≤ A 20 4 9) ∧
  A 20 4 9 = 12 := by sorry

end NUMINAMATH_CALUDE_max_four_digit_sum_l874_87400


namespace NUMINAMATH_CALUDE_det_B_squared_minus_3B_l874_87431

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by sorry

end NUMINAMATH_CALUDE_det_B_squared_minus_3B_l874_87431


namespace NUMINAMATH_CALUDE_union_equality_implies_a_geq_one_l874_87439

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equality_implies_a_geq_one (a : ℝ) :
  A ∪ B a = B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_geq_one_l874_87439


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l874_87434

structure University :=
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (juniors : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)

def stratified_sample (u : University) : Vector ℕ 4 :=
  let sampling_ratio := u.sample_size / u.total_students
  ⟨[u.freshmen * sampling_ratio,
    u.sophomores * sampling_ratio,
    u.juniors * sampling_ratio,
    u.seniors * sampling_ratio],
   by simp⟩

theorem correct_stratified_sample (u : University) 
  (h1 : u.total_students = 8000)
  (h2 : u.freshmen = 1600)
  (h3 : u.sophomores = 3200)
  (h4 : u.juniors = 2000)
  (h5 : u.seniors = 1200)
  (h6 : u.sample_size = 400)
  (h7 : u.total_students = u.freshmen + u.sophomores + u.juniors + u.seniors) :
  stratified_sample u = ⟨[80, 160, 100, 60], by simp⟩ := by
  sorry

#check correct_stratified_sample

end NUMINAMATH_CALUDE_correct_stratified_sample_l874_87434


namespace NUMINAMATH_CALUDE_olivia_supermarket_spending_l874_87484

/-- The amount of money Olivia spent at the supermarket -/
def money_spent (initial_amount : ℕ) (amount_left : ℕ) : ℕ :=
  initial_amount - amount_left

theorem olivia_supermarket_spending :
  money_spent 128 90 = 38 := by
  sorry

end NUMINAMATH_CALUDE_olivia_supermarket_spending_l874_87484


namespace NUMINAMATH_CALUDE_total_course_hours_l874_87446

/-- Represents the total hours spent on a course over the duration of 24 weeks --/
structure CourseHours where
  weekly : ℕ
  additional : ℕ

/-- Calculates the total hours for a course over 24 weeks --/
def totalHours (c : CourseHours) : ℕ := c.weekly * 24 + c.additional

/-- Data analytics course structure --/
def dataAnalyticsCourse : CourseHours :=
  { weekly := 14,  -- 10 hours class + 4 hours homework
    additional := 90 }  -- 48 hours lab sessions + 42 hours projects

/-- Programming course structure --/
def programmingCourse : CourseHours :=
  { weekly := 18,  -- 4 hours class + 8 hours lab + 6 hours assignments
    additional := 0 }

/-- Statistics course structure --/
def statisticsCourse : CourseHours :=
  { weekly := 11,  -- 6 hours class + 2 hours lab + 3 hours group projects
    additional := 45 }  -- 5 hours/week for 9 weeks for exam study

/-- The main theorem stating the total hours spent on all courses --/
theorem total_course_hours :
  totalHours dataAnalyticsCourse +
  totalHours programmingCourse +
  totalHours statisticsCourse = 1167 := by
  sorry

end NUMINAMATH_CALUDE_total_course_hours_l874_87446


namespace NUMINAMATH_CALUDE_square_sum_value_l874_87423

theorem square_sum_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l874_87423


namespace NUMINAMATH_CALUDE_shared_property_of_shapes_l874_87456

-- Define the basic shape
structure Shape :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of having opposite sides parallel and equal
def has_opposite_sides_parallel_and_equal (s : Shape) : Prop :=
  let v := s.vertices
  (v 0 - v 1 = v 3 - v 2) ∧ (v 1 - v 2 = v 0 - v 3)

-- Define the specific shapes
def is_parallelogram (s : Shape) : Prop :=
  has_opposite_sides_parallel_and_equal s

def is_rectangle (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  (v 1 - v 0) • (v 2 - v 1) = 0

def is_rhombus (s : Shape) : Prop :=
  is_parallelogram s ∧
  let v := s.vertices
  ‖v 1 - v 0‖ = ‖v 2 - v 1‖

def is_square (s : Shape) : Prop :=
  is_rectangle s ∧ is_rhombus s

-- Theorem statement
theorem shared_property_of_shapes (s : Shape) :
  (is_parallelogram s ∨ is_rectangle s ∨ is_rhombus s ∨ is_square s) →
  has_opposite_sides_parallel_and_equal s :=
sorry

end NUMINAMATH_CALUDE_shared_property_of_shapes_l874_87456


namespace NUMINAMATH_CALUDE_sum_of_roots_l874_87454

theorem sum_of_roots (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + 4*x + 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l874_87454


namespace NUMINAMATH_CALUDE_total_balloons_is_370_l874_87459

/-- The number of remaining balloons after some burst -/
def remaining_balloons (bags : ℕ) (per_bag : ℕ) (burst : ℕ) : ℕ :=
  bags * per_bag - burst

/-- The total number of remaining balloons -/
def total_remaining_balloons : ℕ :=
  let round := remaining_balloons 5 25 5
  let long := remaining_balloons 4 35 7
  let heart := remaining_balloons 3 40 3
  round + long + heart

theorem total_balloons_is_370 : total_remaining_balloons = 370 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_is_370_l874_87459


namespace NUMINAMATH_CALUDE_smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l874_87490

/-- The smallest natural number with digit sum 2017 -/
def smallest_num_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_num_with_digit_sum_2017_properties :
  digit_sum smallest_num_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_num_with_digit_sum_2017 = 225 ∧
  smallest_num_with_digit_sum_2017 < 10^225 ∧
  ∀ m : ℕ, m < smallest_num_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

theorem first_digit_times_num_digits :
  (smallest_num_with_digit_sum_2017 / 10^224) * num_digits smallest_num_with_digit_sum_2017 = 225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l874_87490


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l874_87419

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -1 ∧ x - 4 + 9 / (x + 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l874_87419


namespace NUMINAMATH_CALUDE_subtracted_value_l874_87406

theorem subtracted_value (N : ℕ) (V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l874_87406


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l874_87464

theorem min_value_quadratic_form (a b c d : ℝ) (h : 5*a + 6*b - 7*c + 4*d = 1) :
  3*a^2 + 2*b^2 + 5*c^2 + d^2 ≥ 15/782 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 5*a₀ + 6*b₀ - 7*c₀ + 4*d₀ = 1 ∧ 3*a₀^2 + 2*b₀^2 + 5*c₀^2 + d₀^2 = 15/782 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l874_87464


namespace NUMINAMATH_CALUDE_second_quadrant_angles_l874_87437

-- Define a function to check if an angle is in the second quadrant
def is_in_second_quadrant (angle : ℝ) : Prop :=
  90 < angle % 360 ∧ angle % 360 ≤ 180

-- Define the given angles
def angle1 : ℝ := -120
def angle2 : ℝ := -240
def angle3 : ℝ := 180
def angle4 : ℝ := 495

-- Theorem statement
theorem second_quadrant_angles :
  is_in_second_quadrant angle2 ∧
  is_in_second_quadrant angle4 ∧
  ¬is_in_second_quadrant angle1 ∧
  ¬is_in_second_quadrant angle3 :=
sorry

end NUMINAMATH_CALUDE_second_quadrant_angles_l874_87437


namespace NUMINAMATH_CALUDE_sequence_sum_l874_87485

/-- Given a sequence defined by a₁ + b₁ = 1, a² + b² = 3, a³ + b³ = 4, a⁴ + b⁴ = 7, a⁵ + b⁵ = 11,
    and for n ≥ 3, aⁿ + bⁿ = (aⁿ⁻¹ + bⁿ⁻¹) + (aⁿ⁻² + bⁿ⁻²),
    prove that a¹¹ + b¹¹ = 199 -/
theorem sequence_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^11 + b^11 = 199 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_l874_87485


namespace NUMINAMATH_CALUDE_michelle_sandwiches_l874_87412

/-- The number of sandwiches Michelle gave to the first co-worker -/
def sandwiches_given : ℕ := sorry

/-- The total number of sandwiches Michelle originally made -/
def total_sandwiches : ℕ := 20

/-- The number of sandwiches left for other co-workers -/
def sandwiches_left : ℕ := 8

/-- The number of sandwiches Michelle kept for herself -/
def sandwiches_kept : ℕ := 2 * sandwiches_given

theorem michelle_sandwiches :
  sandwiches_given + sandwiches_kept + sandwiches_left = total_sandwiches ∧
  sandwiches_given = 4 := by
  sorry

end NUMINAMATH_CALUDE_michelle_sandwiches_l874_87412


namespace NUMINAMATH_CALUDE_households_using_neither_brand_l874_87428

/-- Given information about household soap usage, prove the number of households using neither brand. -/
theorem households_using_neither_brand (total : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  only_A = 60 →
  both = 40 →
  (total - (only_A + 3 * both + both)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_households_using_neither_brand_l874_87428


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l874_87494

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l874_87494


namespace NUMINAMATH_CALUDE_no_eulerian_path_four_odd_degree_l874_87488

/-- A simple graph represented by its vertex set and a function determining adjacency. -/
structure Graph (V : Type) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def has_odd_degree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph is a path that visits every edge exactly once. -/
def has_eulerian_path (G : Graph V) : Prop := sorry

/-- The main theorem: a graph with four vertices of odd degree does not have an Eulerian path. -/
theorem no_eulerian_path_four_odd_degree (V : Type) (G : Graph V) 
  (h : ∃ (a b c d : V), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    has_odd_degree G a ∧ has_odd_degree G b ∧ has_odd_degree G c ∧ has_odd_degree G d) :
  ¬ has_eulerian_path G := by sorry

end NUMINAMATH_CALUDE_no_eulerian_path_four_odd_degree_l874_87488


namespace NUMINAMATH_CALUDE_longest_side_of_specific_garden_l874_87447

/-- Represents a rectangular garden with given perimeter and area-to-perimeter ratio --/
structure RectangularGarden where
  perimeter : ℝ
  areaToPerimeterRatio : ℝ

/-- Calculates the length of the longest side of a rectangular garden --/
def longestSide (garden : RectangularGarden) : ℝ :=
  sorry

/-- Theorem stating the longest side of the specific garden --/
theorem longest_side_of_specific_garden :
  let garden : RectangularGarden := { perimeter := 225, areaToPerimeterRatio := 8 }
  longestSide garden = 93.175 := by sorry

end NUMINAMATH_CALUDE_longest_side_of_specific_garden_l874_87447


namespace NUMINAMATH_CALUDE_inequality_solution_set_l874_87498

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 5 * abs x + 6 < 0) ↔ ((-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3)) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l874_87498


namespace NUMINAMATH_CALUDE_point_on_curve_l874_87401

noncomputable def tangent_slope (x : ℝ) : ℝ := 1 + Real.log x

theorem point_on_curve (x y : ℝ) (h : y = x * Real.log x) :
  tangent_slope x = 2 → x = Real.exp 1 ∧ y = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_l874_87401


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l874_87403

/-- The perimeter of a quadrilateral with sides x, x + 1, 6, and 10, where x = 3, is 23. -/
theorem quadrilateral_perimeter (x : ℝ) (h : x = 3) : x + (x + 1) + 6 + 10 = 23 := by
  sorry

#check quadrilateral_perimeter

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l874_87403


namespace NUMINAMATH_CALUDE_tom_gave_sixteen_balloons_l874_87426

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial_balloons remaining_balloons : ℕ) : ℕ :=
  initial_balloons - remaining_balloons

/-- Theorem: Tom gave 16 balloons to Fred -/
theorem tom_gave_sixteen_balloons :
  let initial_balloons : ℕ := 30
  let remaining_balloons : ℕ := 14
  balloons_given initial_balloons remaining_balloons = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_sixteen_balloons_l874_87426


namespace NUMINAMATH_CALUDE_arrangements_count_l874_87414

/-- The number of arrangements of 6 people with specific conditions -/
def num_arrangements : ℕ :=
  let total_people : ℕ := 6
  let num_teachers : ℕ := 1
  let num_male_students : ℕ := 2
  let num_female_students : ℕ := 3
  let male_students_arrangements : ℕ := 2  -- A_{2}^{2}
  let female_adjacent_pair_selections : ℕ := 3  -- C_{3}^{2}
  let remaining_people_arrangements : ℕ := 12  -- A_{3}^{3}
  male_students_arrangements * female_adjacent_pair_selections * remaining_people_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : num_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l874_87414


namespace NUMINAMATH_CALUDE_average_book_width_l874_87416

/-- The average width of 7 books with given widths is 4.5 cm -/
theorem average_book_width : 
  let book_widths : List ℝ := [5, 3/4, 1.5, 3, 12, 2, 7.5]
  (book_widths.sum / book_widths.length : ℝ) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_average_book_width_l874_87416


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l874_87471

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : Nat
  divided_squares : Nat
  shaded_triangles : Nat

/-- The fraction of a square covered by a shaded triangle -/
def triangle_coverage : Rat := 1/2

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  (quilt.shaded_triangles : Rat) * triangle_coverage / (quilt.total_squares : Rat)

/-- Theorem stating that the shaded fraction of the quilt block is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (quilt : QuiltBlock),
    quilt.total_squares = 16 ∧
    quilt.divided_squares = 4 ∧
    quilt.shaded_triangles = 4 →
    shaded_fraction quilt = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l874_87471


namespace NUMINAMATH_CALUDE_gcd_problems_l874_87468

theorem gcd_problems : 
  (Nat.gcd 840 1785 = 105) ∧ (Nat.gcd 612 468 = 156) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l874_87468


namespace NUMINAMATH_CALUDE_total_spent_on_presents_l874_87443

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℚ := 35.50
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℚ := 120.75
def leonard_belt_price : ℚ := 44.25
def leonard_discount_rate : ℚ := 0.10

def michael_backpack_price : ℚ := 89.50
def michael_jeans : ℕ := 3
def michael_jeans_price : ℚ := 54.50
def michael_tie_price : ℚ := 24.75
def michael_discount_rate : ℚ := 0.15

def emily_shirts : ℕ := 2
def emily_shirt_price : ℚ := 69.25
def emily_books : ℕ := 4
def emily_book_price : ℚ := 14.80
def emily_tax_rate : ℚ := 0.08

theorem total_spent_on_presents (leonard_total michael_total emily_total : ℚ) :
  leonard_total = (1 - leonard_discount_rate) * (leonard_wallets * leonard_wallet_price + leonard_sneakers * leonard_sneaker_price + leonard_belt_price) →
  michael_total = (1 - michael_discount_rate) * (michael_backpack_price + michael_jeans * michael_jeans_price + michael_tie_price) →
  emily_total = (1 + emily_tax_rate) * (emily_shirts * emily_shirt_price + emily_books * emily_book_price) →
  leonard_total + michael_total + emily_total = 802.64 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_presents_l874_87443


namespace NUMINAMATH_CALUDE_cost_per_watt_hour_is_020_l874_87495

/-- Calculates the cost per watt-hour given the number of bulbs, wattage per bulb,
    number of days, and total monthly expense. -/
def cost_per_watt_hour (num_bulbs : ℕ) (watts_per_bulb : ℕ) (days : ℕ) (total_expense : ℚ) : ℚ :=
  total_expense / (num_bulbs * watts_per_bulb * days : ℚ)

/-- Theorem stating that the cost per watt-hour is $0.20 under the given conditions. -/
theorem cost_per_watt_hour_is_020 :
  cost_per_watt_hour 40 60 30 14400 = 1/5 := by sorry

end NUMINAMATH_CALUDE_cost_per_watt_hour_is_020_l874_87495


namespace NUMINAMATH_CALUDE_at_least_one_truth_teller_not_knight_l874_87425

-- Define the possible types of people
inductive PersonType
  | Knight
  | Liar
  | Normal

-- Define a person with a type and a statement about the other person
structure Person where
  type : PersonType
  statement : PersonType → Prop

-- Define what it means for a person to be telling the truth
def isTellingTruth (p : Person) (otherType : PersonType) : Prop :=
  match p.type with
  | PersonType.Knight => p.statement otherType
  | PersonType.Liar => ¬(p.statement otherType)
  | PersonType.Normal => True

-- Define the specific statements made by A and B
def statementA (typeB : PersonType) : Prop := typeB = PersonType.Knight
def statementB (typeA : PersonType) : Prop := typeA ≠ PersonType.Knight

-- Define A and B
def A : Person := { type := PersonType.Knight, statement := statementA }
def B : Person := { type := PersonType.Knight, statement := statementB }

-- The main theorem
theorem at_least_one_truth_teller_not_knight :
  ∃ p : Person, p ∈ [A, B] ∧ 
    (∃ otherType : PersonType, isTellingTruth p otherType) ∧ 
    p.type ≠ PersonType.Knight :=
sorry

end NUMINAMATH_CALUDE_at_least_one_truth_teller_not_knight_l874_87425


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l874_87470

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 4 →
    (B * x - 17) / (x^2 - 7*x + 12) = A / (x - 3) + 4 / (x - 4)) →
  A = 5/4 ∧ B = 21/4 ∧ A + B = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l874_87470


namespace NUMINAMATH_CALUDE_lee_class_b_students_l874_87466

theorem lee_class_b_students (kipling_total : ℕ) (kipling_b : ℕ) (lee_total : ℕ) 
  (h1 : kipling_total = 12)
  (h2 : kipling_b = 8)
  (h3 : lee_total = 30) :
  ∃ (lee_b : ℕ), (lee_b : ℚ) / lee_total = (kipling_b : ℚ) / kipling_total ∧ lee_b = 20 := by
  sorry


end NUMINAMATH_CALUDE_lee_class_b_students_l874_87466


namespace NUMINAMATH_CALUDE_hexagon_diagonal_length_l874_87455

/-- The length of a diagonal in a regular hexagon --/
theorem hexagon_diagonal_length (side_length : ℝ) (h : side_length = 12) :
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_length_l874_87455


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_three_l874_87483

theorem abs_two_minus_sqrt_three : |2 - Real.sqrt 3| = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_three_l874_87483


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l874_87415

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents the given binary number 1011001₍₂₎ -/
def binary_number : List Bool := [true, false, false, true, true, false, true]

/-- The octal number we want to prove equality with -/
def octal_number : ℕ := 131

theorem binary_to_octal_conversion :
  binary_to_decimal binary_number = octal_number := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l874_87415


namespace NUMINAMATH_CALUDE_food_problem_l874_87475

/-- The number of days food lasts for a group of men -/
def food_duration (initial_men : ℕ) (additional_men : ℕ) (initial_days : ℕ) (additional_days : ℕ) : Prop :=
  initial_men * initial_days = 
  initial_men * 2 + (initial_men + additional_men) * additional_days

theorem food_problem : 
  ∃ (D : ℕ), food_duration 760 760 D 10 ∧ D = 22 := by
  sorry

end NUMINAMATH_CALUDE_food_problem_l874_87475


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l874_87436

-- Define the universal set U
def U : Set Int := {-1, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 2*x - 3 = 0}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l874_87436


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l874_87418

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (3 + Complex.I) * m - (2 + Complex.I)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < 0 := by sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l874_87418


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l874_87476

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 6
  let b := (1 : ℚ) / 12
  let midpoint := (a + b) / 2
  midpoint = (1 : ℚ) / 8 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l874_87476


namespace NUMINAMATH_CALUDE_mike_books_before_sale_l874_87457

/-- The number of books Mike bought at the yard sale -/
def books_bought : ℕ := 21

/-- The total number of books Mike has now -/
def total_books_now : ℕ := 56

/-- The number of books Mike had before the yard sale -/
def books_before : ℕ := total_books_now - books_bought

theorem mike_books_before_sale : books_before = 35 := by
  sorry

end NUMINAMATH_CALUDE_mike_books_before_sale_l874_87457


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l874_87496

theorem average_of_a_and_b (a b c : ℝ) : 
  (a + b) / 2 = 45 ∧ (b + c) / 2 = 60 ∧ c - a = 30 → (a + b) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l874_87496


namespace NUMINAMATH_CALUDE_polynomial_difference_theorem_l874_87404

/-- Given two polynomials that differ in terms of x^2 and y^2, 
    prove the values of m and n and the result of a specific expression. -/
theorem polynomial_difference_theorem (m n : ℝ) : 
  (∀ x y : ℝ, 2 * (m * x^2 - 2 * y^2) - (x - 2 * y) - (x - n * y^2 - 2 * x^2) = 0) →
  m = -1 ∧ n = 4 ∧ (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_theorem_l874_87404


namespace NUMINAMATH_CALUDE_average_age_combined_l874_87480

/-- The average age of a group of fifth-graders, parents, and teachers -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) (num_teachers : ℕ)
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h1 : num_fifth_graders = 40)
  (h2 : num_parents = 60)
  (h3 : num_teachers = 10)
  (h4 : avg_age_fifth_graders = 10)
  (h5 : avg_age_parents = 35)
  (h6 : avg_age_teachers = 45) :
  (num_fifth_graders * avg_age_fifth_graders +
   num_parents * avg_age_parents +
   num_teachers * avg_age_teachers) /
  (num_fifth_graders + num_parents + num_teachers : ℚ) = 295 / 11 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l874_87480


namespace NUMINAMATH_CALUDE_cans_collection_proof_l874_87481

/-- The number of cans collected on a given day -/
def cans_on_day (a b : ℚ) (d : ℕ) : ℚ := a * d^2 + b

theorem cans_collection_proof (a b : ℚ) :
  cans_on_day a b 1 = 4 ∧
  cans_on_day a b 2 = 9 ∧
  cans_on_day a b 3 = 14 →
  a = 5/3 ∧ b = 7/3 ∧ cans_on_day a b 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_cans_collection_proof_l874_87481


namespace NUMINAMATH_CALUDE_insects_in_lab_l874_87449

/-- The number of insects in a laboratory given the total number of insect legs and legs per insect. -/
def num_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: There are 8 insects in the laboratory. -/
theorem insects_in_lab : num_insects 48 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_lab_l874_87449


namespace NUMINAMATH_CALUDE_hyperbola_properties_l874_87486

/-- Given a hyperbola with the equation (x²/a² - y²/b² = 1) where a > 0 and b > 0,
    if a perpendicular line from the right focus to an asymptote has length 2 and slope -1/2,
    then b = 2, the hyperbola equation is x² - y²/4 = 1, and the foot of the perpendicular
    is at (√5/5, 2√5/5). -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y c : ℝ),
    (x^2/a^2 - y^2/b^2 = 1) ∧  -- Equation of the hyperbola
    (c^2 = a^2 + b^2) ∧        -- Relation between c and a, b
    ((a^2/c - c)^2 + (a*b/c)^2 = 4) ∧  -- Length of perpendicular = 2
    (-1/2 = (a*b/c) / (a^2/c - c))) →  -- Slope of perpendicular = -1/2
  (b = 2 ∧ 
   (∀ x y, x^2 - y^2/4 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) ∧
   (∃ x y, x = Real.sqrt 5 / 5 ∧ y = 2 * Real.sqrt 5 / 5 ∧
           b*x - a*y = 0 ∧ y = -a/b * (x - Real.sqrt (a^2 + b^2)))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l874_87486


namespace NUMINAMATH_CALUDE_floor_difference_l874_87410

/-- The number of floors in Building A -/
def floors_A : ℕ := 4

/-- The number of floors in Building B -/
def floors_B : ℕ := sorry

/-- The number of floors in Building C -/
def floors_C : ℕ := 59

/-- The relationship between floors in Building B and C -/
axiom floors_C_relation : floors_C = 5 * floors_B - 6

/-- The difference in floors between Building A and Building B is 9 -/
theorem floor_difference : floors_B - floors_A = 9 := by sorry

end NUMINAMATH_CALUDE_floor_difference_l874_87410


namespace NUMINAMATH_CALUDE_future_age_difference_l874_87451

/-- Represents the age difference between Kaylee and Matt in the future -/
def AgeDifference (x : ℕ) : Prop :=
  (8 + x) = 3 * 5

/-- Proves that the number of years into the future when Kaylee will be 3 times as old as Matt is now is 7 years -/
theorem future_age_difference : ∃ (x : ℕ), AgeDifference x ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_future_age_difference_l874_87451


namespace NUMINAMATH_CALUDE_shaded_area_of_intersecting_rectangles_l874_87465

/-- The area of the shaded region formed by two intersecting perpendicular rectangles -/
theorem shaded_area_of_intersecting_rectangles (rect1_width rect1_height rect2_width rect2_height : ℝ) 
  (h1 : rect1_width = 2 ∧ rect1_height = 10)
  (h2 : rect2_width = 3 ∧ rect2_height = 8)
  (h3 : rect1_width ≤ rect2_height ∧ rect2_width ≤ rect1_height) : 
  rect1_width * rect1_height + rect2_width * rect2_height - rect1_width * rect2_width = 38 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_intersecting_rectangles_l874_87465


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l874_87407

/-- Represents the production rate and time for a machine. -/
structure Machine where
  rate : ℝ  -- Sprockets produced per hour
  time : ℝ  -- Hours to produce 2000 sprockets

/-- Given three machines A, B, and G with specific production relationships,
    prove that machine A produces 200/11 sprockets per hour. -/
theorem machine_a_production_rate 
  (a b g : Machine)
  (total_sprockets : ℝ)
  (h1 : total_sprockets = 2000)
  (h2 : a.time = g.time + 10)
  (h3 : b.time = g.time - 5)
  (h4 : g.rate = 1.1 * a.rate)
  (h5 : b.rate = 1.15 * a.rate)
  (h6 : a.rate * a.time = total_sprockets)
  (h7 : b.rate * b.time = total_sprockets)
  (h8 : g.rate * g.time = total_sprockets) :
  a.rate = 200 / 11 := by
  sorry

#eval (200 : ℚ) / 11

end NUMINAMATH_CALUDE_machine_a_production_rate_l874_87407


namespace NUMINAMATH_CALUDE_sack_of_rice_weight_l874_87492

theorem sack_of_rice_weight (cost : ℝ) (price_per_kg : ℝ) (profit : ℝ) (weight : ℝ) : 
  cost = 50 → 
  price_per_kg = 1.20 → 
  profit = 10 → 
  price_per_kg * weight = cost + profit → 
  weight = 50 := by
sorry

end NUMINAMATH_CALUDE_sack_of_rice_weight_l874_87492


namespace NUMINAMATH_CALUDE_cos_theta_minus_pi_third_l874_87461

theorem cos_theta_minus_pi_third (θ : ℝ) 
  (h : Real.sin (3 * Real.pi - θ) = (Real.sqrt 5 / 2) * Real.sin (Real.pi / 2 + θ)) :
  Real.cos (θ - Real.pi / 3) = (1 / 3 + Real.sqrt 15 / 6) ∨ 
  Real.cos (θ - Real.pi / 3) = -(1 / 3 + Real.sqrt 15 / 6) :=
by sorry

end NUMINAMATH_CALUDE_cos_theta_minus_pi_third_l874_87461


namespace NUMINAMATH_CALUDE_number_wall_solution_l874_87448

/-- Represents a number wall with 4 elements in the bottom row -/
structure NumberWall :=
  (bottom_row : Fin 4 → ℕ)
  (second_row_right : ℕ)
  (top : ℕ)

/-- Checks if a number wall is valid according to the summing rules -/
def is_valid_wall (w : NumberWall) : Prop :=
  w.second_row_right = w.bottom_row 2 + w.bottom_row 3 ∧
  w.top = (w.bottom_row 0 + w.bottom_row 1 + w.bottom_row 2) + w.second_row_right

theorem number_wall_solution (w : NumberWall) 
  (h1 : w.bottom_row 1 = 3)
  (h2 : w.bottom_row 2 = 6)
  (h3 : w.bottom_row 3 = 5)
  (h4 : w.second_row_right = 20)
  (h5 : w.top = 57)
  (h6 : is_valid_wall w) :
  w.bottom_row 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_solution_l874_87448


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l874_87440

def set_A : Set ℝ := {x | x^2 + x - 2 < 0}
def set_B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l874_87440


namespace NUMINAMATH_CALUDE_odd_prime_divisor_condition_l874_87405

theorem odd_prime_divisor_condition (n : ℕ) :
  (n > 0 ∧ ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) ↔ (Nat.Prime n ∧ n % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_odd_prime_divisor_condition_l874_87405


namespace NUMINAMATH_CALUDE_preimage_of_three_l874_87473

def A : Set ℝ := Set.univ
def B : Set ℝ := Set.univ

def f : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem preimage_of_three (h : f 2 = 3) : 
  ∃ x ∈ A, f x = 3 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_preimage_of_three_l874_87473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l874_87433

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_condition : a 2^2 + 2*a 2*a 8 + a 6*a 10 = 16) : 
  a 4 * a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l874_87433


namespace NUMINAMATH_CALUDE_edward_spending_l874_87462

theorem edward_spending (initial : ℝ) : 
  initial > 0 →
  let after_clothes := initial - 250
  let after_food := after_clothes - (0.35 * after_clothes)
  let after_electronics := after_food - (0.5 * after_food)
  after_electronics = 200 →
  initial = 1875 := by sorry

end NUMINAMATH_CALUDE_edward_spending_l874_87462


namespace NUMINAMATH_CALUDE_calvins_weight_loss_l874_87432

/-- Calvin's weight loss problem -/
theorem calvins_weight_loss
  (initial_weight : ℕ)
  (weight_loss_per_month : ℕ)
  (months : ℕ)
  (hw : initial_weight = 250)
  (hl : weight_loss_per_month = 8)
  (hm : months = 12) :
  initial_weight - (weight_loss_per_month * months) = 154 :=
by sorry

end NUMINAMATH_CALUDE_calvins_weight_loss_l874_87432


namespace NUMINAMATH_CALUDE_exponent_calculation_l874_87442

theorem exponent_calculation : (((18^15 / 18^14)^3 * 8^3) / 4^5) = 2916 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l874_87442


namespace NUMINAMATH_CALUDE_t_shaped_area_l874_87450

/-- The area of a T-shaped region formed by subtracting three smaller rectangles
    from a larger rectangle -/
theorem t_shaped_area (total_width total_height : ℝ)
                      (rect1_width rect1_height : ℝ)
                      (rect2_width rect2_height : ℝ)
                      (rect3_width rect3_height : ℝ)
                      (h1 : total_width = 6)
                      (h2 : total_height = 5)
                      (h3 : rect1_width = 1)
                      (h4 : rect1_height = 4)
                      (h5 : rect2_width = 1)
                      (h6 : rect2_height = 4)
                      (h7 : rect3_width = 1)
                      (h8 : rect3_height = 3) :
  total_width * total_height - 
  (rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height) = 19 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_area_l874_87450


namespace NUMINAMATH_CALUDE_fourth_term_equals_eleven_l874_87482

/-- Given a sequence {aₙ} where Sₙ = 2n² - 3n, prove that a₄ = 11 -/
theorem fourth_term_equals_eleven (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n) →
  (∀ n, a n = S n - S (n-1)) →
  a 4 = 11 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_equals_eleven_l874_87482


namespace NUMINAMATH_CALUDE_milk_selling_price_l874_87420

/-- Proves that the selling price of milk per litre is twice the cost price,
    given the mixing ratio of water to milk and the profit percentage. -/
theorem milk_selling_price 
  (x : ℝ) -- cost price of pure milk per litre
  (water_ratio : ℝ) -- ratio of water added to pure milk
  (milk_ratio : ℝ) -- ratio of pure milk
  (profit_percentage : ℝ) -- profit percentage
  (h1 : water_ratio = 2) -- 2 litres of water are added
  (h2 : milk_ratio = 6) -- to every 6 litres of pure milk
  (h3 : profit_percentage = 166.67) -- profit percentage is 166.67%
  : ∃ (selling_price : ℝ), selling_price = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_milk_selling_price_l874_87420


namespace NUMINAMATH_CALUDE_crayons_per_box_l874_87477

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 35) (h2 : num_boxes = 7) :
  total_crayons / num_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_box_l874_87477


namespace NUMINAMATH_CALUDE_area_of_locus_enclosed_l874_87469

/-- The locus of the center of a circle touching y = -x and passing through (0, 1) -/
def locusOfCenter (x y : ℝ) : Prop :=
  x = y + Real.sqrt (4 * y - 2) ∨ x = y - Real.sqrt (4 * y - 2)

/-- The area enclosed by the locus and the line y = 1 -/
noncomputable def enclosedArea : ℝ :=
  ∫ y in (0)..(1), 2 * Real.sqrt (4 * y - 2)

theorem area_of_locus_enclosed : enclosedArea = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_locus_enclosed_l874_87469


namespace NUMINAMATH_CALUDE_min_xy_m_range_l874_87409

-- Define the conditions
def condition (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 1/x + 3/y = 2

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) : 
  ∀ a b : ℝ, condition a b → x * y ≤ a * b ∧ x * y ≥ 3 :=
sorry

-- Theorem for the range of m
theorem m_range (x y : ℝ) (h : condition x y) :
  ∀ m : ℝ, (∀ a b : ℝ, condition a b → 3*a + b ≥ m^2 - m) → 
  -2 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_min_xy_m_range_l874_87409


namespace NUMINAMATH_CALUDE_solution_set_f_exp_pos_l874_87491

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Theorem statement
theorem solution_set_f_exp_pos :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_exp_pos_l874_87491


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l874_87452

/-- Given the initial and final conditions of cucumbers after water evaporation,
    prove that the initial water percentage was 99%. -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (final_water_percentage : ℝ)
  (final_weight : ℝ)
  (h_initial_weight : initial_weight = 100)
  (h_final_water_percentage : final_water_percentage = 96)
  (h_final_weight : final_weight = 25) :
  (initial_weight - (1 - final_water_percentage / 100) * final_weight) / initial_weight * 100 = 99 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l874_87452


namespace NUMINAMATH_CALUDE_move_right_example_l874_87438

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveRight (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

/-- The theorem stating that moving (-1, 3) by 5 units right results in (4, 3) -/
theorem move_right_example :
  let initial := Point.mk (-1) 3
  let final := moveRight initial 5
  final = Point.mk 4 3 := by sorry

end NUMINAMATH_CALUDE_move_right_example_l874_87438


namespace NUMINAMATH_CALUDE_negation_of_at_least_one_even_l874_87479

theorem negation_of_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_at_least_one_even_l874_87479


namespace NUMINAMATH_CALUDE_total_energy_calculation_l874_87497

def light_energy (base_watts : ℕ) (multiplier : ℕ) (hours : ℕ) : ℕ :=
  base_watts * multiplier * hours

theorem total_energy_calculation (base_watts : ℕ) (hours : ℕ) 
  (h1 : base_watts = 6)
  (h2 : hours = 2) :
  light_energy base_watts 1 hours + 
  light_energy base_watts 3 hours + 
  light_energy base_watts 4 hours = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_energy_calculation_l874_87497


namespace NUMINAMATH_CALUDE_sandbox_width_l874_87441

/-- A sandbox is a rectangle with a specific perimeter and length-width relationship -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 30
  length_eq : length = 2 * width

theorem sandbox_width (s : Sandbox) : s.width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l874_87441


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l874_87467

theorem absolute_value_not_positive (y : ℚ) : |5 * y - 3| ≤ 0 ↔ y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l874_87467


namespace NUMINAMATH_CALUDE_exponent_calculation_l874_87429

theorem exponent_calculation (a : ℝ) : a^3 * a * a^4 + (-3 * a^4)^2 = 10 * a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l874_87429


namespace NUMINAMATH_CALUDE_yellow_yarns_count_l874_87402

/-- The number of scarves that can be made from one yarn -/
def scarves_per_yarn : ℕ := 3

/-- The number of red yarns May bought -/
def red_yarns : ℕ := 2

/-- The number of blue yarns May bought -/
def blue_yarns : ℕ := 6

/-- The total number of scarves May can make -/
def total_scarves : ℕ := 36

/-- The number of yellow yarns May bought -/
def yellow_yarns : ℕ := (total_scarves - (red_yarns + blue_yarns) * scarves_per_yarn) / scarves_per_yarn

theorem yellow_yarns_count : yellow_yarns = 4 := by
  sorry

end NUMINAMATH_CALUDE_yellow_yarns_count_l874_87402


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l874_87411

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 36) : 
  x + y ≤ 2 * Real.sqrt 43 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l874_87411


namespace NUMINAMATH_CALUDE_min_value_inequality_l874_87474

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l874_87474


namespace NUMINAMATH_CALUDE_andrews_sticker_fraction_l874_87430

theorem andrews_sticker_fraction 
  (total_stickers : ℕ) 
  (andrews_fraction : ℚ) 
  (bills_fraction : ℚ) 
  (total_given : ℕ) :
  total_stickers = 100 →
  bills_fraction = 3/10 →
  total_given = 44 →
  andrews_fraction * total_stickers + 
    bills_fraction * (total_stickers - andrews_fraction * total_stickers) = total_given →
  andrews_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_andrews_sticker_fraction_l874_87430


namespace NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l874_87422

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 1) (m^2 - m - 2)

-- 1. z is a real number iff m = -1 or m = 2
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = 2 := by sorry

-- 2. z is a complex number iff m ≠ -1 and m ≠ 2
theorem z_is_complex (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ -1 ∧ m ≠ 2 := by sorry

-- 3. z is a pure imaginary number iff m = 1
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 1 := by sorry

-- 4. z is in the fourth quadrant iff 1 < m < 2
theorem z_in_fourth_quadrant (m : ℝ) : 
  (z m).re > 0 ∧ (z m).im < 0 ↔ 1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_z_is_real_z_is_complex_z_is_pure_imaginary_z_in_fourth_quadrant_l874_87422


namespace NUMINAMATH_CALUDE_min_base_sum_l874_87427

theorem min_base_sum : 
  ∃ (a b : ℕ+), 
    (3 * a.val + 5 = 4 * b.val + 2) ∧ 
    (∀ (c d : ℕ+), (3 * c.val + 5 = 4 * d.val + 2) → (a.val + b.val ≤ c.val + d.val)) ∧
    (a.val + b.val = 13) := by
  sorry

end NUMINAMATH_CALUDE_min_base_sum_l874_87427


namespace NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l874_87472

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A based on its complement in U
def A : Finset Nat := U \ {2}

-- Theorem statement
theorem number_of_proper_subsets_of_A :
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_proper_subsets_of_A_l874_87472


namespace NUMINAMATH_CALUDE_container_volume_increase_l874_87487

theorem container_volume_increase (original_volume : ℝ) :
  let new_volume := original_volume * 8
  2 * 2 * 2 * original_volume = new_volume :=
by sorry

end NUMINAMATH_CALUDE_container_volume_increase_l874_87487


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l874_87493

theorem sandy_molly_age_ratio :
  ∀ (sandy_age molly_age : ℕ),
    sandy_age = 78 + 6 →
    (sandy_age + 16) * 2 = (molly_age + 16) * 5 →
    sandy_age * 2 = molly_age * 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l874_87493


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l874_87417

theorem least_three_digit_multiple_of_13 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 13 ∣ n → 104 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_13_l874_87417


namespace NUMINAMATH_CALUDE_count_valid_words_l874_87444

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of total possible words without restrictions -/
def total_words : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

/-- The number of words with fewer than two 'A's -/
def words_with_less_than_two_a : ℕ := 
  ((alphabet_size - 1) ^ 2) + (2 * (alphabet_size - 1)) + 
  ((alphabet_size - 1) ^ 3) + (3 * (alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 4) + (4 * (alphabet_size - 1) ^ 3) + 
  ((alphabet_size - 1) ^ 5) + (5 * (alphabet_size - 1) ^ 4)

/-- The number of valid words in the language -/
def valid_words : ℕ := total_words - words_with_less_than_two_a

theorem count_valid_words : 
  valid_words = (25^1 + 25^2 + 25^3 + 25^4 + 25^5) - 
                (24^2 + 2 * 24 + 24^3 + 3 * 24^2 + 24^4 + 4 * 24^3 + 24^5 + 5 * 24^4) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_words_l874_87444


namespace NUMINAMATH_CALUDE_remainder_sum_l874_87458

theorem remainder_sum (x y : ℤ) 
  (hx : x % 80 = 75) 
  (hy : y % 120 = 117) : 
  (x + y) % 40 = 32 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l874_87458


namespace NUMINAMATH_CALUDE_town_population_problem_l874_87408

/-- The original population of the town -/
def original_population : ℕ := 1200

/-- The increase in population -/
def population_increase : ℕ := 1500

/-- The percentage decrease after the increase -/
def percentage_decrease : ℚ := 15 / 100

/-- The final difference in population compared to the original plus increase -/
def final_difference : ℕ := 45

theorem town_population_problem :
  let increased_population := original_population + population_increase
  let decreased_population := increased_population - (increased_population * percentage_decrease).floor
  decreased_population = original_population + population_increase - final_difference :=
by sorry

end NUMINAMATH_CALUDE_town_population_problem_l874_87408


namespace NUMINAMATH_CALUDE_sum_of_roots_l874_87424

theorem sum_of_roots (p q : ℝ) (hp_neq_q : p ≠ q) : 
  (∃ x : ℝ, x^2 + p*x + q = 0 ∧ x^2 + q*x + p = 0) → p + q = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l874_87424


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l874_87421

/-- Calculates the simple interest rate given loan details and total interest --/
def calculate_interest_rate (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) : ℚ :=
  let total_interest_fraction := (loan1_principal * loan1_time + loan2_principal * loan2_time) / 100
  total_interest / total_interest_fraction

theorem interest_rate_calculation (loan1_principal loan1_time loan2_principal loan2_time total_interest : ℚ) 
  (h1 : loan1_principal = 5000)
  (h2 : loan1_time = 2)
  (h3 : loan2_principal = 3000)
  (h4 : loan2_time = 4)
  (h5 : total_interest = 2200) :
  calculate_interest_rate loan1_principal loan1_time loan2_principal loan2_time total_interest = 10 := by
  sorry

#eval calculate_interest_rate 5000 2 3000 4 2200

end NUMINAMATH_CALUDE_interest_rate_calculation_l874_87421


namespace NUMINAMATH_CALUDE_clarence_oranges_l874_87413

/-- The total number of oranges Clarence has -/
def total_oranges (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Clarence has 8 oranges in total -/
theorem clarence_oranges :
  total_oranges 5 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l874_87413


namespace NUMINAMATH_CALUDE_sum_four_digit_distinct_remainder_l874_87478

def T : ℕ := sorry

theorem sum_four_digit_distinct_remainder (T : ℕ) : T % 1000 = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_four_digit_distinct_remainder_l874_87478


namespace NUMINAMATH_CALUDE_meeting_point_distance_l874_87435

theorem meeting_point_distance (total_distance : ℝ) (speed1 speed2 : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed1 = 2)
  (h3 : speed2 = 4) :
  speed1 * (total_distance / (speed1 + speed2)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l874_87435
