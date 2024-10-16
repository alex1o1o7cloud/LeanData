import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_l654_65495

theorem quadratic_roots (a : ℝ) : 
  (3^2 - 2*3 + a = 0) → 
  ((-1)^2 - 2*(-1) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l654_65495


namespace NUMINAMATH_CALUDE_tangent_line_at_point_2_neg6_l654_65449

-- Define the function f(x) = x³ + x - 16
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_point_2_neg6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := -6
  let m : ℝ := f' x₀
  (∀ x y, y - y₀ = m * (x - x₀) ↔ 13 * x - y - 32 = 0) ∧
  f x₀ = y₀ ∧
  m = 13 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_2_neg6_l654_65449


namespace NUMINAMATH_CALUDE_sum_of_fractions_l654_65482

theorem sum_of_fractions : 
  (2 : ℚ) / 100 + 5 / 1000 + 8 / 10000 + 6 / 100000 = 0.02586 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l654_65482


namespace NUMINAMATH_CALUDE_line_intersection_x_axis_l654_65406

/-- A line passing through two points intersects the x-axis --/
theorem line_intersection_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 8 ∧ y₁ = 2) 
  (h_point2 : x₂ = 4 ∧ y₂ = 6) :
  ∃ x : ℝ, x = 10 ∧ 
    (y₂ - y₁) * (x - x₁) = (x₂ - x₁) * (0 - y₁) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_x_axis_l654_65406


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l654_65473

/-- The number of shirts -/
def num_shirts : ℕ := 8

/-- The number of ties -/
def num_ties : ℕ := 6

/-- The number of shirts that can be paired with the specific tie -/
def specific_shirts : ℕ := 2

/-- The number of different shirt-and-tie combinations -/
def total_combinations : ℕ := (num_shirts - specific_shirts) * (num_ties - 1) + specific_shirts

theorem shirt_tie_combinations : total_combinations = 32 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l654_65473


namespace NUMINAMATH_CALUDE_equation_solution_l654_65463

theorem equation_solution : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l654_65463


namespace NUMINAMATH_CALUDE_class_artworks_count_l654_65428

/-- Represents the number of artworks created by a group of students -/
structure Artworks :=
  (paintings : ℕ)
  (drawings : ℕ)
  (sculptures : ℕ)

/-- Calculates the total number of artworks -/
def total_artworks (a : Artworks) : ℕ :=
  a.paintings + a.drawings + a.sculptures

theorem class_artworks_count :
  let total_students : ℕ := 36
  let group1_students : ℕ := 24
  let group2_students : ℕ := 12
  let total_kits : ℕ := 48
  let group1_sharing_ratio : ℕ := 3  -- 1 kit per 3 students
  let group2_sharing_ratio : ℕ := 2  -- 1 kit per 2 students
  
  let group1_first_half : Artworks := ⟨2, 4, 1⟩
  let group1_second_half : Artworks := ⟨1, 5, 3⟩
  let group2_first_third : Artworks := ⟨3, 6, 3⟩
  let group2_second_third : Artworks := ⟨4, 7, 1⟩
  
  let group1_artworks : Artworks := ⟨
    12 * group1_first_half.paintings + 12 * group1_second_half.paintings,
    12 * group1_first_half.drawings + 12 * group1_second_half.drawings,
    12 * group1_first_half.sculptures + 12 * group1_second_half.sculptures
  ⟩
  
  let group2_artworks : Artworks := ⟨
    4 * group2_first_third.paintings + 8 * group2_second_third.paintings,
    4 * group2_first_third.drawings + 8 * group2_second_third.drawings,
    4 * group2_first_third.sculptures + 8 * group2_second_third.sculptures
  ⟩
  
  let total_class_artworks : Artworks := ⟨
    group1_artworks.paintings + group2_artworks.paintings,
    group1_artworks.drawings + group2_artworks.drawings,
    group1_artworks.sculptures + group2_artworks.sculptures
  ⟩
  
  total_artworks total_class_artworks = 336 := by sorry

end NUMINAMATH_CALUDE_class_artworks_count_l654_65428


namespace NUMINAMATH_CALUDE_sin_cos_identity_l654_65474

theorem sin_cos_identity : 
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.cos (34 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l654_65474


namespace NUMINAMATH_CALUDE_existence_of_xy_l654_65412

theorem existence_of_xy : ∃ x y : ℕ+, 
  (x.val < 30 ∧ y.val < 30) ∧ 
  (x.val + y.val + x.val * y.val = 119) ∧
  (x.val + y.val = 20) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l654_65412


namespace NUMINAMATH_CALUDE_function_max_min_on_interval_l654_65486

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem function_max_min_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc m 0, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc m 0, f x = 3) ∧
  (∀ x ∈ Set.Icc m 0, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc m 0, f x = 2) ↔
  m ∈ Set.Icc (-2) (-1) :=
by sorry

end NUMINAMATH_CALUDE_function_max_min_on_interval_l654_65486


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_l654_65488

/-- Calculates the percentage of blueberry muffins out of the total muffins -/
theorem blueberry_muffin_percentage
  (num_cartons : ℕ)
  (blueberries_per_carton : ℕ)
  (blueberries_per_muffin : ℕ)
  (num_cinnamon_muffins : ℕ)
  (h1 : num_cartons = 3)
  (h2 : blueberries_per_carton = 200)
  (h3 : blueberries_per_muffin = 10)
  (h4 : num_cinnamon_muffins = 60)
  : (((num_cartons * blueberries_per_carton) / blueberries_per_muffin : ℚ) /
     ((num_cartons * blueberries_per_carton) / blueberries_per_muffin + num_cinnamon_muffins)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_l654_65488


namespace NUMINAMATH_CALUDE_stratified_sampling_second_grade_l654_65481

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def total_students (g : GradeDistribution) : ℕ :=
  g.first + g.second + g.third

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total sample size -/
def total_sample (s : SampleDistribution) : ℕ :=
  s.first + s.second + s.third

/-- Checks if the sample distribution is proportional to the grade distribution -/
def is_proportional_sample (g : GradeDistribution) (s : SampleDistribution) : Prop :=
  g.first * s.second = g.second * s.first ∧
  g.second * s.third = g.third * s.second

theorem stratified_sampling_second_grade
  (g : GradeDistribution)
  (s : SampleDistribution)
  (h1 : total_students g = 2000)
  (h2 : g.first = 5 * g.third)
  (h3 : g.second = 3 * g.third)
  (h4 : total_sample s = 20)
  (h5 : is_proportional_sample g s) :
  s.second = 6 := by
  sorry

#check stratified_sampling_second_grade

end NUMINAMATH_CALUDE_stratified_sampling_second_grade_l654_65481


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l654_65432

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) :
  let x := (a^2 - b^2) / (2*a)
  x^2 + b^2 = (a - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l654_65432


namespace NUMINAMATH_CALUDE_part_one_part_two_l654_65455

-- Define the function y
def y (x a : ℝ) : ℝ := 2 * x^2 - (a + 2) * x + a

-- Part 1
theorem part_one : 
  ∀ x : ℝ, y x (-1) > 0 ↔ (x > 1 ∨ x < -1/2) := by sorry

-- Part 2
theorem part_two :
  ∀ a x₁ x₂ : ℝ, 
    (x₁ > 0 ∧ x₂ > 0) →
    (2 * x₁^2 - (a + 2) * x₁ + a = x₁ + 1) →
    (2 * x₂^2 - (a + 2) * x₂ + a = x₂ + 1) →
    (∀ x : ℝ, x > 0 → x₂/x₁ + x₁/x₂ ≥ 6) ∧ 
    (∃ a : ℝ, x₂/x₁ + x₁/x₂ = 6) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l654_65455


namespace NUMINAMATH_CALUDE_cos_165_degrees_l654_65400

theorem cos_165_degrees : 
  Real.cos (165 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_165_degrees_l654_65400


namespace NUMINAMATH_CALUDE_solution_k_value_l654_65403

theorem solution_k_value (x y k : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (heq : 2 * x + k * y = 6) :
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_value_l654_65403


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l654_65477

/-- The number of blue balls in the second urn -/
def M : ℝ := 7.4

/-- The probability of drawing two balls of the same color -/
def same_color_probability : ℝ := 0.65

/-- The number of green balls in the first urn -/
def green_balls_urn1 : ℕ := 3

/-- The number of blue balls in the first urn -/
def blue_balls_urn1 : ℕ := 7

/-- The number of green balls in the second urn -/
def green_balls_urn2 : ℕ := 20

theorem urn_probability_theorem :
  (green_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (green_balls_urn2 / (green_balls_urn2 + M : ℝ)) +
  (blue_balls_urn1 / (green_balls_urn1 + blue_balls_urn1 : ℝ)) * (M / (green_balls_urn2 + M : ℝ)) =
  same_color_probability :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l654_65477


namespace NUMINAMATH_CALUDE_f_extrema_l654_65496

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1) * (x + a)

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem f_extrema (a : ℝ) (h : f_derivative a (-1) = 0) :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = max) ∧
    (∀ x ∈ Set.Icc (-3/2 : ℝ) 1, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc (-3/2 : ℝ) 1, f a x = min) ∧
    max = 6 ∧ min = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l654_65496


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l654_65480

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales based on the given conditions -/
def valid_jelly_sales (s : JellySales) : Prop :=
  s.grape = 2 * s.strawberry ∧
  s.raspberry = 2 * s.plum ∧
  s.raspberry * 3 = s.grape ∧
  s.plum = 6

/-- Theorem stating that given the conditions, 18 jars of strawberry jelly were sold -/
theorem strawberry_jelly_sales (s : JellySales) (h : valid_jelly_sales s) : s.strawberry = 18 := by
  sorry


end NUMINAMATH_CALUDE_strawberry_jelly_sales_l654_65480


namespace NUMINAMATH_CALUDE_bees_after_six_days_l654_65429

/-- The number of bees after n days in the hive process -/
def bees (n : ℕ) : ℕ := 6^n

/-- The process starts with 1 bee and continues for 6 days -/
def days : ℕ := 6

/-- The theorem stating the number of bees after 6 days -/
theorem bees_after_six_days : bees days = 46656 := by sorry

end NUMINAMATH_CALUDE_bees_after_six_days_l654_65429


namespace NUMINAMATH_CALUDE_point_on_circle_l654_65405

theorem point_on_circle (t : ℝ) : 
  let x := (3 - t^3) / (3 + t^3)
  let y := 3*t / (3 + t^3)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_circle_l654_65405


namespace NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l654_65492

/-- The function f(x) defined as x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem stating that 5/4 is the largest value of c such that -5 is in the range of f(x) -/
theorem largest_c_for_negative_five_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -5) ↔ c ≤ 5/4 := by sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l654_65492


namespace NUMINAMATH_CALUDE_line_through_circle_center_l654_65413

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- 
If the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0,
then a = 1
-/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l654_65413


namespace NUMINAMATH_CALUDE_sunday_necklace_production_l654_65423

/-- The number of necklaces made by the first machine on Sunday -/
def first_machine_necklaces : ℕ := 45

/-- The ratio of necklaces made by the second machine compared to the first -/
def second_machine_ratio : ℝ := 2.4

/-- The total number of necklaces made on Sunday -/
def total_necklaces : ℕ := 153

/-- Theorem stating that the total number of necklaces made on Sunday is 153 -/
theorem sunday_necklace_production :
  (first_machine_necklaces : ℝ) + first_machine_necklaces * second_machine_ratio = total_necklaces := by
  sorry

end NUMINAMATH_CALUDE_sunday_necklace_production_l654_65423


namespace NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l654_65441

def is_p_safe (n p : ℕ) : Prop :=
  ∀ k : ℤ, |n - k * p| ≥ 3

def simultaneously_safe (n : ℕ) : Prop :=
  is_p_safe n 5 ∧ is_p_safe n 7 ∧ is_p_safe n 11

theorem no_simultaneously_safe_numbers : 
  ¬ ∃ n : ℕ, n > 0 ∧ n ≤ 500 ∧ simultaneously_safe n := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneously_safe_numbers_l654_65441


namespace NUMINAMATH_CALUDE_range_of_function_l654_65465

theorem range_of_function (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l654_65465


namespace NUMINAMATH_CALUDE_larger_number_is_72_l654_65404

theorem larger_number_is_72 (x y : ℝ) : 
  5 * y = 6 * x → y - x = 12 → y = 72 := by sorry

end NUMINAMATH_CALUDE_larger_number_is_72_l654_65404


namespace NUMINAMATH_CALUDE_abes_age_problem_l654_65430

theorem abes_age_problem (present_age : ℕ) (sum_ages : ℕ) (years_ago : ℕ) :
  present_age = 19 →
  sum_ages = 31 →
  sum_ages = present_age + (present_age - years_ago) →
  years_ago = 7 := by
sorry

end NUMINAMATH_CALUDE_abes_age_problem_l654_65430


namespace NUMINAMATH_CALUDE_age_ratio_is_two_l654_65426

/-- The age difference between Yuan and David -/
def age_difference : ℕ := 7

/-- David's age -/
def david_age : ℕ := 7

/-- Yuan's age -/
def yuan_age : ℕ := david_age + age_difference

/-- The ratio of Yuan's age to David's age -/
def age_ratio : ℚ := yuan_age / david_age

theorem age_ratio_is_two : age_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_l654_65426


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fifth_l654_65410

theorem maximize_x_cubed_y_fifth (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  x^3 * y^5 ≤ (7.5^3) * (12.5^5) ∧ 
  (x^3 * y^5 = (7.5^3) * (12.5^5) ↔ x = 7.5 ∧ y = 12.5) := by
sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fifth_l654_65410


namespace NUMINAMATH_CALUDE_range_of_f_l654_65458

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f :
  Set.range f = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l654_65458


namespace NUMINAMATH_CALUDE_carla_tile_counting_l654_65490

theorem carla_tile_counting (tiles : ℕ) (books : ℕ) (book_counts : ℕ) (total_counts : ℕ)
  (h1 : tiles = 38)
  (h2 : books = 75)
  (h3 : book_counts = 3)
  (h4 : total_counts = 301)
  : ∃ (tile_counts : ℕ), tile_counts * tiles + book_counts * books = total_counts ∧ tile_counts = 2 := by
  sorry

end NUMINAMATH_CALUDE_carla_tile_counting_l654_65490


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l654_65421

-- Problem 1
theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((1) * (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2))) / ((1/3) * a^(1/6) * b^(5/6)) = 6 * a :=
sorry

-- Problem 2
theorem evaluate_expression :
  (2 * (9/16)^(1/2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + 
   Real.log (4 * Real.exp 3) - Real.log 8 / Real.log 9 * Real.log 33 / Real.log 4) = 7/2 :=
sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l654_65421


namespace NUMINAMATH_CALUDE_parallelogram_area_equality_l654_65414

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- Given a triangle ABC, constructs parallelogram ACDE on side AC -/
def constructParallelogramOnAC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC, constructs parallelogram BCFG on side BC -/
def constructParallelogramOnBC (t : Triangle) : Parallelogram := sorry

/-- Given a triangle ABC and point H, constructs parallelogram ABML on side AB 
    such that AL and BM are equal and parallel to HC -/
def constructParallelogramOnAB (t : Triangle) (H : Point) : Parallelogram := sorry

/-- Main theorem statement -/
theorem parallelogram_area_equality 
  (t : Triangle) 
  (H : Point) 
  (ACDE : Parallelogram) 
  (BCFG : Parallelogram) 
  (ABML : Parallelogram) 
  (h1 : ACDE = constructParallelogramOnAC t) 
  (h2 : BCFG = constructParallelogramOnBC t) 
  (h3 : ABML = constructParallelogramOnAB t H) :
  area ABML = area ACDE + area BCFG := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_equality_l654_65414


namespace NUMINAMATH_CALUDE_freken_bok_candies_l654_65499

def initial_candies : ℕ := 111

-- n represents the number of candies before lunch
def candies_before_lunch (n : ℕ) : Prop :=
  n ≤ initial_candies ∧ 
  ∃ (k : ℕ), k * 20 = 11 * n ∧ 
  k ≤ initial_candies

def candies_found_by_freken_bok (n : ℕ) : ℕ :=
  (11 * n) / 60

theorem freken_bok_candies :
  ∃ (n : ℕ), candies_before_lunch n ∧ 
  candies_found_by_freken_bok n = 11 :=
sorry

end NUMINAMATH_CALUDE_freken_bok_candies_l654_65499


namespace NUMINAMATH_CALUDE_sector_central_angle_l654_65451

/-- Given a circular sector with arc length 4 and area 2, 
    prove that its central angle is 4 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (θ : ℝ) :
  arc_length = 4 →
  area = 2 →
  θ = arc_length / (2 * area / arc_length) →
  θ = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l654_65451


namespace NUMINAMATH_CALUDE_distance_is_1760_l654_65459

/-- The distance between Péter's and Károly's houses in meters. -/
def distance_between_houses : ℝ := 1760

/-- The distance from Péter's house to the first meeting point in meters. -/
def first_meeting_distance : ℝ := 720

/-- The distance from Károly's house to the second meeting point in meters. -/
def second_meeting_distance : ℝ := 400

/-- Theorem stating that the distance between the houses is 1760 meters. -/
theorem distance_is_1760 :
  let x := distance_between_houses
  let d1 := first_meeting_distance
  let d2 := second_meeting_distance
  (d1 / (x - d1) = (x - d2) / (x + d2)) →
  x = 1760 := by
  sorry


end NUMINAMATH_CALUDE_distance_is_1760_l654_65459


namespace NUMINAMATH_CALUDE_at_least_one_chinese_book_l654_65476

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem at_least_one_chinese_book :
  ∀ (selection : Finset ℕ),
  selection.card = drawn_books →
  (∀ i ∈ selection, i < total_books) →
  ∃ i ∈ selection, i < chinese_books :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_chinese_book_l654_65476


namespace NUMINAMATH_CALUDE_equilateral_roots_l654_65401

/-- Given complex numbers p and q, and z₁ and z₂ being the roots of z² + pz + q = 0
    such that 0, z₁, and z₂ form an equilateral triangle in the complex plane,
    prove that p²/q = 1 -/
theorem equilateral_roots (p q z₁ z₂ : ℂ) : 
  z₁^2 + p*z₁ + q = 0 ∧ 
  z₂^2 + p*z₂ + q = 0 ∧ 
  ∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_l654_65401


namespace NUMINAMATH_CALUDE_sarah_shirts_l654_65446

/-- The total number of shirts Sarah owns after buying new shirts -/
theorem sarah_shirts (initial_shirts new_shirts : ℕ) 
  (h1 : initial_shirts = 9)
  (h2 : new_shirts = 8) : 
  initial_shirts + new_shirts = 17 := by
  sorry

end NUMINAMATH_CALUDE_sarah_shirts_l654_65446


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l654_65487

theorem remainder_after_adding_2025 (n : ℤ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l654_65487


namespace NUMINAMATH_CALUDE_product_of_sums_powers_l654_65469

theorem product_of_sums_powers : (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 63403380965376 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_powers_l654_65469


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l654_65498

theorem system_of_equations_solution (a b c : ℝ) : 
  (a - b = 3) → 
  (a^2 + b^2 = 31) → 
  (a + 2*b - c = 5) → 
  (a*b - c = 37/2) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l654_65498


namespace NUMINAMATH_CALUDE_intersection_range_l654_65468

/-- The function f(x) = x³ - 3x - 1 --/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Theorem: If the line y = m intersects the graph of f(x) = x³ - 3x - 1
    at three distinct points, then m is in the open interval (-3, 1) --/
theorem intersection_range (m : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) →
  m > -3 ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l654_65468


namespace NUMINAMATH_CALUDE_product_difference_equals_one_l654_65456

theorem product_difference_equals_one : (527 : ℤ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_one_l654_65456


namespace NUMINAMATH_CALUDE_probability_one_of_each_l654_65402

def num_shirts : ℕ := 6
def num_shorts : ℕ := 8
def num_socks : ℕ := 9
def num_hats : ℕ := 4
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
def items_to_select : ℕ := 4

theorem probability_one_of_each :
  (num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1 * num_hats.choose 1) / total_items.choose items_to_select = 96 / 975 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l654_65402


namespace NUMINAMATH_CALUDE_projection_result_l654_65439

/-- Given two vectors a and b, if both are projected onto the same vector v
    resulting in the same vector p, then p is equal to (15/58, 35/58). -/
theorem projection_result (a b v p : ℝ × ℝ) : 
  a = (-3, 2) →
  b = (4, -1) →
  (∃ (k₁ k₂ : ℝ), p = k₁ • v ∧ p = k₂ • v) →
  p = (15/58, 35/58) :=
sorry

end NUMINAMATH_CALUDE_projection_result_l654_65439


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l654_65440

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l654_65440


namespace NUMINAMATH_CALUDE_cubic_system_product_l654_65475

theorem cubic_system_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010 ∧ y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2010 ∧ y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2010 ∧ y₃^3 - 3*x₃^2*y₃ = 2000) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/100 := by
sorry

end NUMINAMATH_CALUDE_cubic_system_product_l654_65475


namespace NUMINAMATH_CALUDE_specific_pyramid_height_l654_65478

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to any vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its square base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  let p := RightPyramid.mk 40 15
  pyramid_height p = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_height_l654_65478


namespace NUMINAMATH_CALUDE_height_comparison_equivalences_l654_65437

-- Define the classes A and B
variable (A B : Type)

-- Define a height function for students
variable (height : A ⊕ B → ℝ)

-- Define the propositions for each question
def tallest_A_taller_than_tallest_B : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def every_B_shorter_than_some_A : Prop :=
  ∀ b : B, ∃ a : A, height (Sum.inl a) > height (Sum.inr b)

def for_any_A_exists_shorter_B : Prop :=
  ∀ a : A, ∃ b : B, height (Sum.inl a) > height (Sum.inr b)

def shortest_B_shorter_than_shortest_A : Prop :=
  ∃ a : A, ∀ b : B, height (Sum.inl a) > height (Sum.inr b)

-- State the theorem
theorem height_comparison_equivalences
  (A B : Type) (height : A ⊕ B → ℝ) :
  (tallest_A_taller_than_tallest_B A B height ↔ every_B_shorter_than_some_A A B height) ∧
  (for_any_A_exists_shorter_B A B height ↔ shortest_B_shorter_than_shortest_A A B height) :=
sorry

end NUMINAMATH_CALUDE_height_comparison_equivalences_l654_65437


namespace NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l654_65417

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x-3) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 3)

/-- Theorem: x(x-3) = 0 is a quadratic equation -/
theorem x_x_minus_3_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_x_minus_3_is_quadratic_l654_65417


namespace NUMINAMATH_CALUDE_margaret_score_is_86_l654_65434

/-- Given an average test score, calculate Margaret's score based on the conditions -/
def margaret_score (average : ℝ) : ℝ :=
  let marco_score := average * 0.9
  marco_score + 5

/-- Theorem stating that Margaret's score is 86 given the conditions -/
theorem margaret_score_is_86 :
  margaret_score 90 = 86 := by
  sorry

end NUMINAMATH_CALUDE_margaret_score_is_86_l654_65434


namespace NUMINAMATH_CALUDE_mollys_age_l654_65433

/-- Given that the ratio of Sandy's age to Molly's age is 4:3,
    and Sandy will be 34 years old in 6 years,
    prove that Molly's current age is 21 years. -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l654_65433


namespace NUMINAMATH_CALUDE_solve_equation_l654_65411

theorem solve_equation (x : ℝ) (h : 8 * (2 + 1 / x) = 18) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l654_65411


namespace NUMINAMATH_CALUDE_factorization_equality_l654_65457

theorem factorization_equality (x : ℝ) : 
  32 * x^4 - 48 * x^7 + 16 * x^2 = 16 * x^2 * (2 * x^2 - 3 * x^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l654_65457


namespace NUMINAMATH_CALUDE_base_equation_solution_l654_65466

/-- Convert a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Given positive integers C and D where D = C + 2, 
    and the equation 253_C - 75_D = 124_(C+D) holds, 
    prove that C + D = 26 -/
theorem base_equation_solution (C D : Nat) 
  (h1 : C > 0) 
  (h2 : D > 0) 
  (h3 : D = C + 2) 
  (h4 : toBase10 [2, 5, 3] C - toBase10 [7, 5] D = toBase10 [1, 2, 4] (C + D)) :
  C + D = 26 := by
  sorry

end NUMINAMATH_CALUDE_base_equation_solution_l654_65466


namespace NUMINAMATH_CALUDE_transformations_of_f_l654_65448

def f (x : ℝ) : ℝ := 3 * x + 4

def shift_left_down (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (x + 1) - 2

def reflect_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = f (-x)

def reflect_y_eq_1 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = 2 - f x

def reflect_y_eq_neg_x (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = (x + 4) / 3

def reflect_point (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, g x = f (2 * a - x) + 2 * (b - a)

theorem transformations_of_f :
  (∃ g : ℝ → ℝ, shift_left_down g ∧ (∀ x, g x = 3 * x + 5)) ∧
  (∃ g : ℝ → ℝ, reflect_y_axis g ∧ (∀ x, g x = -3 * x + 4)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_1 g ∧ (∀ x, g x = -3 * x - 2)) ∧
  (∃ g : ℝ → ℝ, reflect_y_eq_neg_x g) ∧
  (∀ a b : ℝ, ∃ g : ℝ → ℝ, reflect_point g a b ∧ (∀ x, g x = 3 * x + 2 * b - 6 * a - 4)) :=
sorry

end NUMINAMATH_CALUDE_transformations_of_f_l654_65448


namespace NUMINAMATH_CALUDE_quadratic_transformation_l654_65450

/-- Given that ax^2 + bx + c can be expressed as 4(x - 5)^2 + 16, prove that when 5ax^2 + 5bx + 5c 
    is expressed in the form n(x - h)^2 + k, the value of h is 5. -/
theorem quadratic_transformation (a b c : ℝ) 
    (h : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    ∃ n k, ∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = n * (x - 5)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l654_65450


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_line_l654_65484

/-- Given a function f(x) = a*sin(x) + b*cos(x) where x is real,
    if x₀ is an axis of symmetry for f(x) and tan(x₀) = 2,
    then the point (a,b) lies on the line x - 2y = 0. -/
theorem symmetry_axis_implies_line (a b x₀ : ℝ) :
  let f := fun (x : ℝ) ↦ a * Real.sin x + b * Real.cos x
  (∀ x, f (x₀ + x) = f (x₀ - x)) →  -- x₀ is an axis of symmetry
  Real.tan x₀ = 2 →
  a - 2 * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_line_l654_65484


namespace NUMINAMATH_CALUDE_moss_pollen_radius_scientific_notation_l654_65453

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem moss_pollen_radius_scientific_notation :
  let r := 0.0000042
  let sn := toScientificNotation r
  sn.significand = 4.2 ∧ sn.exponent = -6 := by sorry

end NUMINAMATH_CALUDE_moss_pollen_radius_scientific_notation_l654_65453


namespace NUMINAMATH_CALUDE_evaluate_expression_l654_65427

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (2 * z - 5 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l654_65427


namespace NUMINAMATH_CALUDE_plane_equation_l654_65464

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (pt : Point3D) (pl : Plane) : Prop :=
  pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (pl1 pl2 : Plane) : Prop :=
  pl1.a * pl2.a + pl1.b * pl2.b + pl1.c * pl2.c = 0

/-- The main theorem -/
theorem plane_equation : ∃ (pl : Plane),
  pointOnPlane ⟨0, 2, 1⟩ pl ∧
  pointOnPlane ⟨2, 0, 1⟩ pl ∧
  planesArePerpendicular pl ⟨2, -1, 3, -4⟩ ∧
  pl.a > 0 ∧
  Int.gcd (Int.natAbs (Int.floor pl.a)) (Int.gcd (Int.natAbs (Int.floor pl.b)) (Int.gcd (Int.natAbs (Int.floor pl.c)) (Int.natAbs (Int.floor pl.d)))) = 1 ∧
  pl = ⟨1, 1, -1, -1⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l654_65464


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l654_65418

theorem quadratic_equation_roots (k : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
   (k^2 - 1) * x₁^2 - 3*(3*k - 1)*x₁ + 18 = 0 ∧
   (k^2 - 1) * x₂^2 - 3*(3*k - 1)*x₂ + 18 = 0) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l654_65418


namespace NUMINAMATH_CALUDE_z_value_l654_65454

theorem z_value (a : ℕ) (z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 49 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l654_65454


namespace NUMINAMATH_CALUDE_janice_earnings_l654_65485

/-- Janice's weekly earnings calculation --/
theorem janice_earnings 
  (days_per_week : ℕ) 
  (overtime_shifts : ℕ) 
  (overtime_pay : ℝ) 
  (total_earnings : ℝ) 
  (h1 : days_per_week = 5)
  (h2 : overtime_shifts = 3)
  (h3 : overtime_pay = 15)
  (h4 : total_earnings = 195) :
  ∃ (daily_earnings : ℝ), 
    daily_earnings * days_per_week + overtime_pay * overtime_shifts = total_earnings ∧ 
    daily_earnings = 30 := by
  sorry


end NUMINAMATH_CALUDE_janice_earnings_l654_65485


namespace NUMINAMATH_CALUDE_patio_length_l654_65462

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := by
  sorry

end NUMINAMATH_CALUDE_patio_length_l654_65462


namespace NUMINAMATH_CALUDE_dogs_with_tags_and_collars_l654_65494

theorem dogs_with_tags_and_collars (total : ℕ) (tags : ℕ) (collars : ℕ) (neither : ℕ) 
  (h_total : total = 80)
  (h_tags : tags = 45)
  (h_collars : collars = 40)
  (h_neither : neither = 1) :
  total = tags + collars - (tags + collars - total + neither) := by
  sorry

#check dogs_with_tags_and_collars

end NUMINAMATH_CALUDE_dogs_with_tags_and_collars_l654_65494


namespace NUMINAMATH_CALUDE_mary_apple_expense_l654_65409

theorem mary_apple_expense (total_spent berries_cost peaches_cost : ℚ)
  (h1 : total_spent = 34.72)
  (h2 : berries_cost = 11.08)
  (h3 : peaches_cost = 9.31) :
  total_spent - (berries_cost + peaches_cost) = 14.33 := by
sorry

end NUMINAMATH_CALUDE_mary_apple_expense_l654_65409


namespace NUMINAMATH_CALUDE_sin_cos_sum_one_l654_65472

theorem sin_cos_sum_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_one_l654_65472


namespace NUMINAMATH_CALUDE_robot_wear_combinations_l654_65416

/-- Represents the number of ways to wear items on one arm -/
def waysPerArm : ℕ := 1

/-- Represents the number of arms -/
def numArms : ℕ := 2

/-- Represents the number of ways to order items between arms -/
def waysBetweenArms : ℕ := 1

/-- Calculates the total number of ways to wear all items -/
def totalWays : ℕ := waysPerArm ^ numArms * waysBetweenArms

theorem robot_wear_combinations : totalWays = 4 := by
  sorry

end NUMINAMATH_CALUDE_robot_wear_combinations_l654_65416


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_sqrt_five_l654_65431

theorem complex_magnitude_equals_five_sqrt_five (t : ℝ) :
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 5 * Real.sqrt 5 ↔ t = 10) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_sqrt_five_l654_65431


namespace NUMINAMATH_CALUDE_two_trains_problem_l654_65422

/-- The problem of two trains approaching each other -/
theorem two_trains_problem (length1 length2 speed1 clear_time : ℝ) 
  (h1 : length1 = 120)
  (h2 : length2 = 300)
  (h3 : speed1 = 42)
  (h4 : clear_time = 20.99832013438925) : 
  ∃ speed2 : ℝ, speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_trains_problem_l654_65422


namespace NUMINAMATH_CALUDE_birds_on_fence_l654_65425

/-- Given an initial number of birds and a final number of birds on a fence,
    calculate the number of additional birds that joined. -/
def additional_birds (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that given 6 initial birds and 10 final birds on a fence,
    the number of additional birds that joined is 4. -/
theorem birds_on_fence : additional_birds 6 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l654_65425


namespace NUMINAMATH_CALUDE_inequality_proof_l654_65479

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 ∧ ((a + b) / 2)^2 ≥ a * b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l654_65479


namespace NUMINAMATH_CALUDE_equation_substitution_l654_65470

theorem equation_substitution (x y : ℝ) :
  (y = 2 * x - 1) → (2 * x - 3 * y = 5) → (2 * x - 6 * x + 3 = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l654_65470


namespace NUMINAMATH_CALUDE_vector_addition_l654_65497

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![3, 0]

-- State the theorem
theorem vector_addition :
  (a + b) = ![4, 2] := by sorry

end NUMINAMATH_CALUDE_vector_addition_l654_65497


namespace NUMINAMATH_CALUDE_triangle_perimeter_l654_65443

/-- Proves that a triangle with inradius 2.5 cm and area 40 cm² has a perimeter of 32 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) :
  r = 2.5 →
  A = 40 →
  A = r * (p / 2) →
  p = 32 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l654_65443


namespace NUMINAMATH_CALUDE_sin_ten_degrees_root_l654_65467

theorem sin_ten_degrees_root : ∃ x : ℝ, 
  (x = Real.sin (10 * π / 180)) ∧ 
  (8 * x^3 - 6 * x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_ten_degrees_root_l654_65467


namespace NUMINAMATH_CALUDE_box_side_length_l654_65460

/-- Proves that the length of one side of a cubic box is approximately 18.17 inches
    given the cost per box, total volume needed, and total cost. -/
theorem box_side_length (cost_per_box : ℝ) (total_volume : ℝ) (total_cost : ℝ)
  (h1 : cost_per_box = 1.30)
  (h2 : total_volume = 3.06 * 1000000)
  (h3 : total_cost = 663)
  : ∃ (side_length : ℝ), abs (side_length - 18.17) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_box_side_length_l654_65460


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l654_65444

theorem fraction_nonnegative (x : ℝ) (h : x ≠ 3) : x^2 / (x - 3)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l654_65444


namespace NUMINAMATH_CALUDE_mask_production_rates_l654_65415

/-- Represents the daily production rate of masks in millions before equipment change -/
def initial_rate : ℝ := 40

/-- Represents the daily production rate of masks in millions after equipment change -/
def final_rate : ℝ := 56

/-- Represents the number of masks left to produce in millions -/
def remaining_masks : ℝ := 280

/-- Represents the increase in production efficiency as a decimal -/
def efficiency_increase : ℝ := 0.4

/-- Represents the number of days saved due to equipment change -/
def days_saved : ℝ := 2

theorem mask_production_rates :
  (remaining_masks / initial_rate - remaining_masks / (initial_rate * (1 + efficiency_increase)) = days_saved) ∧
  (final_rate = initial_rate * (1 + efficiency_increase)) := by
  sorry

end NUMINAMATH_CALUDE_mask_production_rates_l654_65415


namespace NUMINAMATH_CALUDE_initial_water_percentage_l654_65436

theorem initial_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 36 →
  final_fraction = 3/4 →
  ∃ initial_percentage : ℝ,
    initial_percentage = 30 ∧
    (initial_percentage / 100) * capacity + added_water = final_fraction * capacity :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l654_65436


namespace NUMINAMATH_CALUDE_cone_from_sector_l654_65442

/-- Represents a circular sector -/
structure CircularSector where
  radius : ℝ
  angle : ℝ

/-- Represents a cone -/
structure Cone where
  baseRadius : ℝ
  slantHeight : ℝ
  height : ℝ

/-- Checks if a cone can be formed from a given circular sector -/
def canFormCone (sector : CircularSector) (cone : Cone) : Prop :=
  -- The slant height of the cone equals the radius of the sector
  cone.slantHeight = sector.radius ∧
  -- The arc length of the sector equals the circumference of the cone's base
  (sector.angle / 360) * (2 * Real.pi * sector.radius) = 2 * Real.pi * cone.baseRadius ∧
  -- The Pythagorean theorem holds for the cone's dimensions
  cone.slantHeight ^ 2 = cone.baseRadius ^ 2 + cone.height ^ 2

/-- Theorem stating that a specific cone can be formed from a given sector -/
theorem cone_from_sector :
  let sector := CircularSector.mk 15 300
  let cone := Cone.mk 12 15 9
  canFormCone sector cone := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_l654_65442


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l654_65483

/-- Represents a tetrahedron with four heights -/
structure Tetrahedron where
  h1 : ℝ
  h2 : ℝ
  h3 : ℝ
  h4 : ℝ

/-- The property that the sum of areas of any three faces is greater than the area of the fourth face -/
def validTetrahedron (t : Tetrahedron) : Prop :=
  ∀ (v : ℝ), v > 0 →
    (3 * v / t.h1 < 3 * v / t.h2 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h2 < 3 * v / t.h1 + 3 * v / t.h3 + 3 * v / t.h4) ∧
    (3 * v / t.h3 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h4) ∧
    (3 * v / t.h4 < 3 * v / t.h1 + 3 * v / t.h2 + 3 * v / t.h3)

/-- Theorem stating that no tetrahedron exists with heights 1, 2, 3, and 6 -/
theorem no_tetrahedron_with_heights_1_2_3_6 :
  ¬ ∃ (t : Tetrahedron), t.h1 = 1 ∧ t.h2 = 2 ∧ t.h3 = 3 ∧ t.h4 = 6 ∧ validTetrahedron t :=
sorry

end NUMINAMATH_CALUDE_no_tetrahedron_with_heights_1_2_3_6_l654_65483


namespace NUMINAMATH_CALUDE_second_studio_students_l654_65419

theorem second_studio_students (total : ℕ) (first : ℕ) (third : ℕ) 
  (h1 : total = 376) (h2 : first = 110) (h3 : third = 131) :
  total - first - third = 135 := by
  sorry

end NUMINAMATH_CALUDE_second_studio_students_l654_65419


namespace NUMINAMATH_CALUDE_BA_equals_AB_l654_65491

def matrix_2x2 (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, b; c, d]

theorem BA_equals_AB (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B)
  (h2 : A * B = matrix_2x2 5 2 (-2) 4) :
  B * A = matrix_2x2 5 2 (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_BA_equals_AB_l654_65491


namespace NUMINAMATH_CALUDE_baseball_league_games_l654_65445

theorem baseball_league_games (n m : ℕ) : 
  (∃ (g₁ g₂ : Finset (Finset ℕ)), 
    (g₁.card = 4 ∧ g₂.card = 4) ∧ 
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₁, t₁ ≠ t₂ → (∃ k : ℕ, k = n)) ∧
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₂, (∃ k : ℕ, k = m)) ∧
    n > 2 * m ∧
    m > 4 ∧
    (∃ t ∈ g₁, 3 * n + 4 * m = 76)) →
  n = 48 := by sorry

end NUMINAMATH_CALUDE_baseball_league_games_l654_65445


namespace NUMINAMATH_CALUDE_jose_painting_time_l654_65420

/-- The time it takes for Alex to paint a car alone -/
def alex_time : ℝ := 5

/-- The time it takes for Jose and Alex to paint a car together -/
def combined_time : ℝ := 2.91666666667

/-- The time it takes for Jose to paint a car alone -/
def jose_time : ℝ := 7

/-- Theorem stating that given Alex's time and the combined time, Jose's time is 7 days -/
theorem jose_painting_time : 
  1 / alex_time + 1 / jose_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_jose_painting_time_l654_65420


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l654_65407

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℚ
  workDays : ℕ

theorem daily_wage_of_c (a b c : Worker) 
  (ratio_a_b : a.dailyWage / b.dailyWage = 3 / 4)
  (ratio_b_c : b.dailyWage / c.dailyWage = 4 / 5)
  (work_days : a.workDays = 6 ∧ b.workDays = 9 ∧ c.workDays = 4)
  (total_earning : a.dailyWage * a.workDays + b.dailyWage * b.workDays + c.dailyWage * c.workDays = 1850) :
  c.dailyWage = 625 / 3 := by
  sorry


end NUMINAMATH_CALUDE_daily_wage_of_c_l654_65407


namespace NUMINAMATH_CALUDE_distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l654_65493

/-- The number of ways to distribute 7 different books among 3 people -/
def distribute_books (scenario : Nat) : Nat :=
  match scenario with
  | 1 => 630  -- One person gets 1 book, one gets 2 books, and one gets 4 books
  | 2 => 630  -- One person gets 3 books, and two people each get 2 books
  | _ => 0    -- Invalid scenario

/-- Proof that both distribution scenarios result in 630 ways -/
theorem distribute_books_equal : distribute_books 1 = distribute_books 2 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 1 is 630 -/
theorem distribute_books_scenario1 : distribute_books 1 = 630 := by
  sorry

/-- Proof that the number of ways to distribute books in scenario 2 is 630 -/
theorem distribute_books_scenario2 : distribute_books 2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_equal_distribute_books_scenario1_distribute_books_scenario2_l654_65493


namespace NUMINAMATH_CALUDE_count_multiples_eq_16_l654_65424

/-- The number of positive multiples of 3 less than 150 with units digit 3 or 9 -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 10 = 3 ∨ n % 10 = 9)
    (Finset.filter (fun n => n % 3 = 0) (Finset.range 150))).card

theorem count_multiples_eq_16 : count_multiples = 16 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_eq_16_l654_65424


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_square_expression_l654_65438

theorem absolute_value_sqrt_square_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_square_expression_l654_65438


namespace NUMINAMATH_CALUDE_euler_line_equation_l654_65452

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the property that AC = BC (isosceles triangle)
def is_isosceles (C : ℝ × ℝ) : Prop := dist A C = dist B C

-- Define the Euler line
def euler_line (C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | P.1 - 2 * P.2 + 3 = 0}

-- Theorem statement
theorem euler_line_equation (C : ℝ × ℝ) (h : is_isosceles C) :
  euler_line C = {P : ℝ × ℝ | P.1 - 2 * P.2 + 3 = 0} :=
sorry

end NUMINAMATH_CALUDE_euler_line_equation_l654_65452


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l654_65489

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let p : Point2D := ⟨1, -2⟩
  let q : Point2D := symmetricPoint p
  q.x = -1 ∧ q.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l654_65489


namespace NUMINAMATH_CALUDE_cube_sum_zero_l654_65447

theorem cube_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_zero_l654_65447


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l654_65461

/-- Quadratic function f(x) = x^2 + (2-a)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2-a)*x + 5

theorem quadratic_max_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f a x ≤ f a 1) →
  a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l654_65461


namespace NUMINAMATH_CALUDE_smallest_greater_than_1_1_l654_65471

def S : Set ℚ := {1.4, 9/10, 1.2, 0.5, 13/10}

theorem smallest_greater_than_1_1 : 
  ∃ x ∈ S, x > 1.1 ∧ ∀ y ∈ S, y > 1.1 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_greater_than_1_1_l654_65471


namespace NUMINAMATH_CALUDE_power_of_negative_product_l654_65435

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l654_65435


namespace NUMINAMATH_CALUDE_expand_product_l654_65408

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l654_65408
