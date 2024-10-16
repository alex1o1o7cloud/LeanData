import Mathlib

namespace NUMINAMATH_CALUDE_safe_mountain_climb_l3128_312840

theorem safe_mountain_climb : ∃ t : ℕ,
  t ≥ 0 ∧
  t % 26 ≠ 0 ∧ t % 26 ≠ 1 ∧
  t % 14 ≠ 0 ∧ t % 14 ≠ 1 ∧
  (t + 6) % 26 ≠ 0 ∧ (t + 6) % 26 ≠ 1 ∧
  (t + 6) % 14 ≠ 0 ∧ (t + 6) % 14 ≠ 1 ∧
  t + 24 < 26 * 14 := by
  sorry

end NUMINAMATH_CALUDE_safe_mountain_climb_l3128_312840


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3128_312849

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c) 
    (h4 : a^2 + b^2 = c^2) : 
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') = (5 + 3 * Real.sqrt 2) / (a' + b' + c') := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3128_312849


namespace NUMINAMATH_CALUDE_water_volume_for_four_balls_l3128_312885

/-- The volume of water needed to cover four touching balls in a cylinder -/
theorem water_volume_for_four_balls (r ball_radius container_radius : ℝ) 
  (h_ball_radius : ball_radius = 0.5)
  (h_container_radius : container_radius = 1) :
  let water_height := container_radius + ball_radius
  let cylinder_volume := π * container_radius^2 * water_height
  let ball_volume := (4/3) * π * ball_radius^3
  cylinder_volume - 4 * ball_volume = (2/3) * π := by sorry

end NUMINAMATH_CALUDE_water_volume_for_four_balls_l3128_312885


namespace NUMINAMATH_CALUDE_sixteenth_selected_student_number_l3128_312835

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalStudents : ℕ
  numGroups : ℕ
  interval : ℕ
  firstSelected : ℕ

/-- Calculates the number of the nth selected student in a systematic sampling. -/
def nthSelectedStudent (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.firstSelected + (n - 1) * s.interval

theorem sixteenth_selected_student_number
  (s : SystematicSampling)
  (h1 : s.totalStudents = 800)
  (h2 : s.numGroups = 50)
  (h3 : s.interval = s.totalStudents / s.numGroups)
  (h4 : nthSelectedStudent s 3 = 36) :
  nthSelectedStudent s 16 = 244 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_selected_student_number_l3128_312835


namespace NUMINAMATH_CALUDE_flower_count_l3128_312830

theorem flower_count (vase_capacity : ℝ) (carnation_count : ℝ) (vases_needed : ℝ) :
  vase_capacity = 6.0 →
  carnation_count = 7.0 →
  vases_needed = 6.666666667 →
  (vases_needed * vase_capacity + carnation_count : ℝ) = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l3128_312830


namespace NUMINAMATH_CALUDE_director_sphere_theorem_l3128_312844

/-- The surface S: ax^2 + by^2 + cz^2 = 1 -/
def S (a b c : ℝ) (x y z : ℝ) : Prop :=
  a * x^2 + b * y^2 + c * z^2 = 1

/-- The director sphere K: x^2 + y^2 + z^2 = 1/a + 1/b + 1/c -/
def K (a b c : ℝ) (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1/a + 1/b + 1/c

/-- A plane tangent to S at point (u, v, w) -/
def tangent_plane (a b c : ℝ) (u v w x y z : ℝ) : Prop :=
  a * u * x + b * v * y + c * w * z = 1

/-- Three mutually perpendicular planes -/
def perpendicular_planes (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℝ) : Prop :=
  p₁ * p₂ + q₁ * q₂ + r₁ * r₂ = 0 ∧
  p₁ * p₃ + q₁ * q₃ + r₁ * r₃ = 0 ∧
  p₂ * p₃ + q₂ * q₃ + r₂ * r₃ = 0

theorem director_sphere_theorem (a b c : ℝ) (x₀ y₀ z₀ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ u₁ v₁ w₁ u₂ v₂ w₂ u₃ v₃ w₃ : ℝ,
    S a b c u₁ v₁ w₁ ∧ S a b c u₂ v₂ w₂ ∧ S a b c u₃ v₃ w₃ ∧
    tangent_plane a b c u₁ v₁ w₁ x₀ y₀ z₀ ∧
    tangent_plane a b c u₂ v₂ w₂ x₀ y₀ z₀ ∧
    tangent_plane a b c u₃ v₃ w₃ x₀ y₀ z₀ ∧
    perpendicular_planes 
      (a * u₁) (b * v₁) (c * w₁)
      (a * u₂) (b * v₂) (c * w₂)
      (a * u₃) (b * v₃) (c * w₃)) →
  K a b c x₀ y₀ z₀ := by
  sorry

end NUMINAMATH_CALUDE_director_sphere_theorem_l3128_312844


namespace NUMINAMATH_CALUDE_payment_calculation_l3128_312825

theorem payment_calculation (payment_rate : ℚ) (rooms_cleaned : ℚ) :
  payment_rate = 13 / 3 →
  rooms_cleaned = 8 / 5 →
  payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l3128_312825


namespace NUMINAMATH_CALUDE_circle_condition_l3128_312859

/-- The equation of a potentially circular shape -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

/-- The condition for the equation to represent a circle -/
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating the range of 'a' for which the equation represents a circle -/
theorem circle_condition (a : ℝ) : is_circle a ↔ -2 < a ∧ a < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3128_312859


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3128_312834

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  -- The proof would go here, but we're skipping it as requested
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3128_312834


namespace NUMINAMATH_CALUDE_triangle_properties_l3128_312848

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The angle C in radians -/
def angle_C (t : Triangle) : ℝ :=
  sorry

/-- The angle B in radians -/
def angle_B (t : Triangle) : ℝ :=
  sorry

theorem triangle_properties (t : Triangle) 
  (ha : t.a = 3) 
  (hb : t.b = 5) 
  (hc : t.c = 7) : 
  (angle_C t = 2 * Real.pi / 3) ∧ 
  (Real.sin (angle_B t + Real.pi / 3) = 4 * Real.sqrt 3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3128_312848


namespace NUMINAMATH_CALUDE_initial_cash_calculation_l3128_312895

-- Define the initial cash as a real number
variable (X : ℝ)

-- Define the constants from the problem
def raw_materials : ℝ := 500
def machinery : ℝ := 400
def sales_tax : ℝ := 0.05
def exchange_rate : ℝ := 1.2
def labor_cost_rate : ℝ := 0.1
def inflation_rate : ℝ := 0.02
def years : ℕ := 2
def remaining_amount : ℝ := 900

-- State the theorem
theorem initial_cash_calculation :
  remaining_amount = (X - ((1 + sales_tax) * (raw_materials + machinery) * exchange_rate + labor_cost_rate * X)) / (1 + inflation_rate) ^ years :=
by sorry

end NUMINAMATH_CALUDE_initial_cash_calculation_l3128_312895


namespace NUMINAMATH_CALUDE_mirror_area_l3128_312850

/-- The area of a rectangular mirror fitted inside a frame -/
theorem mirror_area (frame_width frame_height frame_side_width : ℝ) 
  (h1 : frame_width = 100)
  (h2 : frame_height = 70)
  (h3 : frame_side_width = 15) : 
  (frame_width - 2 * frame_side_width) * (frame_height - 2 * frame_side_width) = 2800 :=
by sorry

end NUMINAMATH_CALUDE_mirror_area_l3128_312850


namespace NUMINAMATH_CALUDE_satisfying_function_characterization_l3128_312833

/-- A function from positive reals to reals satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 →
    (f x + f y ≤ f (x + y) / 2) ∧
    (f x / x + f y / y ≥ f (x + y) / (x + y))

/-- The theorem stating that any satisfying function must be of the form f(x) = ax² where a ≤ 0 -/
theorem satisfying_function_characterization (f : ℝ → ℝ) :
  SatisfyingFunction f →
  ∃ a : ℝ, a ≤ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x^2 :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_characterization_l3128_312833


namespace NUMINAMATH_CALUDE_symmetry_condition_l3128_312896

/-- Given a curve y = (ax + b) / (cx - d) where a, b, c, and d are nonzero real numbers,
    if y = x and y = -x are axes of symmetry, then d + b = 0 -/
theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x : ℝ, x = (a * x + b) / (c * x - d)) →
  (∀ x : ℝ, x = (a * (-x) + b) / (c * (-x) - d)) →
  d + b = 0 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_condition_l3128_312896


namespace NUMINAMATH_CALUDE_rental_crossover_point_l3128_312846

/-- Represents the rental rates for a car agency -/
structure AgencyRates where
  dailyRate : ℝ
  mileRate : ℝ

/-- Theorem stating the crossover point for car rental agencies -/
theorem rental_crossover_point (days : ℝ) (agency1 agency2 : AgencyRates) 
  (h1 : agency1.dailyRate = 20.25)
  (h2 : agency1.mileRate = 0.14)
  (h3 : agency2.dailyRate = 18.25)
  (h4 : agency2.mileRate = 0.22)
  : ∃ m : ℝ, m = 25 * days ∧ 
    agency1.dailyRate * days + agency1.mileRate * m = agency2.dailyRate * days + agency2.mileRate * m :=
by sorry

end NUMINAMATH_CALUDE_rental_crossover_point_l3128_312846


namespace NUMINAMATH_CALUDE_taylor_paint_time_l3128_312842

theorem taylor_paint_time (jennifer_time : ℝ) (combined_time : ℝ) (taylor_time : ℝ) : 
  jennifer_time = 10 →
  combined_time = 5.45454545455 →
  (1 / taylor_time) + (1 / jennifer_time) = (1 / combined_time) →
  taylor_time = 12 := by
sorry

end NUMINAMATH_CALUDE_taylor_paint_time_l3128_312842


namespace NUMINAMATH_CALUDE_unique_cube_fraction_l3128_312892

theorem unique_cube_fraction :
  ∃! (n : ℤ), n ≠ 30 ∧ ∃ (k : ℤ), n / (30 - n) = k^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_fraction_l3128_312892


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3128_312894

/-- Proves that for the line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3128_312894


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3128_312860

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 120 = longer_base
  area_ratio : (shorter_base + (shorter_base + 60)) / ((shorter_base + 60) + longer_base) = 3 / 4
  equal_areas : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment ^ 2 / 120⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l3128_312860


namespace NUMINAMATH_CALUDE_large_box_tape_proof_l3128_312867

/-- Amount of tape (in feet) needed to seal a medium box -/
def medium_box_tape : ℝ := 2

/-- Amount of tape (in feet) needed to seal a small box -/
def small_box_tape : ℝ := 1

/-- Amount of tape (in feet) needed for the address label on any box -/
def label_tape : ℝ := 1

/-- Number of large boxes packed -/
def large_boxes : ℕ := 2

/-- Number of medium boxes packed -/
def medium_boxes : ℕ := 8

/-- Number of small boxes packed -/
def small_boxes : ℕ := 5

/-- Total amount of tape (in feet) used -/
def total_tape : ℝ := 44

/-- Amount of tape (in feet) needed to seal a large box -/
def large_box_tape : ℝ := 4

theorem large_box_tape_proof :
  large_box_tape * large_boxes + label_tape * large_boxes +
  medium_box_tape * medium_boxes + label_tape * medium_boxes +
  small_box_tape * small_boxes + label_tape * small_boxes = total_tape :=
by sorry

end NUMINAMATH_CALUDE_large_box_tape_proof_l3128_312867


namespace NUMINAMATH_CALUDE_probability_at_most_one_correct_l3128_312863

theorem probability_at_most_one_correct (pA pB : ℚ) : 
  pA = 3/5 → pB = 2/3 → 
  let p_at_most_one := 
    (1 - pA) * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
    2 * (1 - pA) * (1 - pA) * pB * (1 - pB)
  p_at_most_one = 32/225 := by
sorry

end NUMINAMATH_CALUDE_probability_at_most_one_correct_l3128_312863


namespace NUMINAMATH_CALUDE_travel_ratio_l3128_312853

/-- The ratio of distances in a specific travel scenario -/
theorem travel_ratio (d x : ℝ) (h1 : 0 < d) (h2 : 0 < x) (h3 : x < d) :
  (d - x) / 1 = x / 1 + d / 7 → x / (d - x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_ratio_l3128_312853


namespace NUMINAMATH_CALUDE_playground_to_landscape_ratio_l3128_312817

/-- A rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  length_breadth_relation : length = 4 * breadth
  length_value : length = 120
  playground_size : playground_area = 1200

/-- The ratio of playground area to total landscape area is 1:3 -/
theorem playground_to_landscape_ratio (L : Landscape) :
  L.playground_area / (L.length * L.breadth) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_playground_to_landscape_ratio_l3128_312817


namespace NUMINAMATH_CALUDE_sunflower_height_in_meters_l3128_312864

-- Define constants
def sister_height_feet : ℝ := 4.15
def sister_additional_height_cm : ℝ := 37
def sunflower_height_difference_inches : ℝ := 63

-- Define conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem sunflower_height_in_meters :
  let sister_height_cm := sister_height_feet * inches_per_foot * cm_per_inch + sister_additional_height_cm
  let sunflower_height_cm := sister_height_cm + sunflower_height_difference_inches * cm_per_inch
  sunflower_height_cm / cm_per_meter = 3.23512 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_in_meters_l3128_312864


namespace NUMINAMATH_CALUDE_multiply_fractions_l3128_312899

theorem multiply_fractions : (7 : ℚ) * (1 / 17) * 34 = 14 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l3128_312899


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3128_312861

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3128_312861


namespace NUMINAMATH_CALUDE_pencil_cost_is_two_l3128_312807

/-- Represents the cost of school supplies for Mary --/
structure SchoolSuppliesCost where
  num_classes : ℕ
  folders_per_class : ℕ
  pencils_per_class : ℕ
  erasers_per_pencils : ℕ
  folder_cost : ℚ
  eraser_cost : ℚ
  paint_cost : ℚ
  total_spent : ℚ

/-- Calculates the cost of a single pencil given the school supplies cost structure --/
def pencil_cost (c : SchoolSuppliesCost) : ℚ :=
  let total_folders := c.num_classes * c.folders_per_class
  let total_pencils := c.num_classes * c.pencils_per_class
  let total_erasers := (total_pencils + c.erasers_per_pencils - 1) / c.erasers_per_pencils
  let non_pencil_cost := total_folders * c.folder_cost + total_erasers * c.eraser_cost + c.paint_cost
  let pencil_total_cost := c.total_spent - non_pencil_cost
  pencil_total_cost / total_pencils

/-- Theorem stating that the cost of each pencil is $2 --/
theorem pencil_cost_is_two (c : SchoolSuppliesCost) 
  (h1 : c.num_classes = 6)
  (h2 : c.folders_per_class = 1)
  (h3 : c.pencils_per_class = 3)
  (h4 : c.erasers_per_pencils = 6)
  (h5 : c.folder_cost = 6)
  (h6 : c.eraser_cost = 1)
  (h7 : c.paint_cost = 5)
  (h8 : c.total_spent = 80) :
  pencil_cost c = 2 := by
  sorry


end NUMINAMATH_CALUDE_pencil_cost_is_two_l3128_312807


namespace NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l3128_312871

theorem smallest_y_with_given_remainders : 
  ∃! y : ℕ, 
    y > 0 ∧
    y % 3 = 2 ∧ 
    y % 5 = 4 ∧ 
    y % 7 = 6 ∧
    ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 5 = 4 ∧ z % 7 = 6 → y ≤ z :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l3128_312871


namespace NUMINAMATH_CALUDE_triangle_problem_l3128_312868

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) ∧
  -- Given conditions
  b = 2 ∧
  (1/2) * a * c * (Real.sin B) = Real.sqrt 3 →
  -- Conclusion
  B = π/3 ∧ a = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3128_312868


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3128_312852

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - x - 2 = 0 → x = 2 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 → x^2 - x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3128_312852


namespace NUMINAMATH_CALUDE_particular_number_problem_l3128_312872

theorem particular_number_problem : ∃! x : ℚ, ((x / 23) - 67) * 2 = 102 :=
  by sorry

end NUMINAMATH_CALUDE_particular_number_problem_l3128_312872


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3128_312800

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3128_312800


namespace NUMINAMATH_CALUDE_cos_product_special_angles_l3128_312832

theorem cos_product_special_angles : 
  Real.cos ((2 * π) / 5) * Real.cos ((6 * π) / 5) = -(1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_cos_product_special_angles_l3128_312832


namespace NUMINAMATH_CALUDE_value_of_a_l3128_312883

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 5) 
  (eq3 : c = 3) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3128_312883


namespace NUMINAMATH_CALUDE_bryce_raisins_l3128_312893

theorem bryce_raisins (x : ℕ) : 
  (x - 6 = x / 2) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l3128_312893


namespace NUMINAMATH_CALUDE_picture_placement_l3128_312805

/-- Given a wall of width 27 feet and a centered picture of width 5 feet,
    the distance from the end of the wall to the nearest edge of the picture is 11 feet. -/
theorem picture_placement (wall_width picture_width : ℝ) (h1 : wall_width = 27) (h2 : picture_width = 5) :
  (wall_width - picture_width) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l3128_312805


namespace NUMINAMATH_CALUDE_quadratic_roots_l3128_312839

theorem quadratic_roots : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3128_312839


namespace NUMINAMATH_CALUDE_max_additional_plates_l3128_312873

/-- Represents the sets of letters for each position in the license plate --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)
  (fourth : Finset Char)

/-- The initial license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'A', 'E', 'I', 'O', 'U'},
    second := {'B', 'C', 'D'},
    third := {'L', 'M', 'N', 'P'},
    fourth := {'S', 'T'} }

/-- The number of new letters that can be added --/
def newLettersCount : Nat := 3

/-- The maximum number of letters that can be added to a single set --/
def maxAddToSet : Nat := 2

/-- Calculates the number of possible license plates --/
def calculatePlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card * sets.fourth.card

/-- Theorem: The maximum number of additional license plates is 180 --/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (calculatePlates newSets - calculatePlates initialSets = 180) ∧
    (∀ (otherSets : LicensePlateSets),
      (calculatePlates otherSets - calculatePlates initialSets) ≤ 180) :=
sorry


end NUMINAMATH_CALUDE_max_additional_plates_l3128_312873


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3128_312855

theorem min_value_sqrt_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  Real.sqrt (a^2 + b^2 + c^2) ≥ Real.sqrt 3 ∧ 
  (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3128_312855


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3128_312880

theorem polynomial_evaluation (x : ℝ) (h : x = 4) : x^3 - x^2 + x - 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3128_312880


namespace NUMINAMATH_CALUDE_triangle_property_l3128_312884

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C ∧
  a = 4 →
  A = π / 3 ∧ 
  (∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
    4 * 4 = b' * b' + c' * c' - b' * c' → 
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3128_312884


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3128_312881

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3128_312881


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l3128_312890

theorem sqrt_meaningful (x : ℝ) : ∃ y : ℝ, y ^ 2 = x - 3 ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l3128_312890


namespace NUMINAMATH_CALUDE_two_in_A_implies_a_is_one_or_two_l3128_312886

-- Define the set A
def A (a : ℝ) : Set ℝ := {-2, 2*a, a^2 - a}

-- Theorem statement
theorem two_in_A_implies_a_is_one_or_two :
  ∀ a : ℝ, 2 ∈ A a → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_in_A_implies_a_is_one_or_two_l3128_312886


namespace NUMINAMATH_CALUDE_log_equation_solution_l3128_312803

theorem log_equation_solution :
  ∃! x : ℝ, (Real.log (Real.sqrt (7 * x + 3)) + Real.log (Real.sqrt (4 * x + 5)) = 1 / 2 + Real.log 3) ∧
             (7 * x + 3 > 0) ∧ (4 * x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3128_312803


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_y_equals_4x_l3128_312857

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_to_y_equals_4x :
  ∃! P₀ : ℝ × ℝ, 
    P₀.1 = 1 ∧ 
    P₀.2 = 0 ∧ 
    (∀ x : ℝ, f x = x^3 + x - 2) ∧
    (deriv f P₀.1 = 4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_y_equals_4x_l3128_312857


namespace NUMINAMATH_CALUDE_ab_value_l3128_312847

theorem ab_value (a b : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3128_312847


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3128_312828

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (a ≥ 5) → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧ 
  ∃ b : ℝ, b < 5 ∧ (∀ x ∈ Set.Icc 1 2, x^2 - b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3128_312828


namespace NUMINAMATH_CALUDE_expand_binomials_l3128_312822

theorem expand_binomials (x y : ℝ) : (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3128_312822


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3128_312869

/-- Given a line and a circle in 2D space, prove that the sum of distances from a specific point to the intersection points of the line and circle is √6. -/
theorem intersection_distance_sum (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (P A B : ℝ × ℝ) :
  l = {(x, y) : ℝ × ℝ | x + y = 1} →
  C = {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x + 2*y = 0} →
  P = (1, 0) →
  A ∈ l ∩ C →
  B ∈ l ∩ C →
  A ≠ B →
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l3128_312869


namespace NUMINAMATH_CALUDE_min_value_at_neg_pi_half_l3128_312802

/-- The function f(x) = x + 2cos(x) has its minimum value on the interval [-π/2, 0] at x = -π/2 -/
theorem min_value_at_neg_pi_half :
  let f : ℝ → ℝ := λ x ↦ x + 2 * Real.cos x
  let a : ℝ := -π/2
  let b : ℝ := 0
  ∀ x ∈ Set.Icc a b, f a ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_neg_pi_half_l3128_312802


namespace NUMINAMATH_CALUDE_find_P_value_l3128_312821

theorem find_P_value (P Q R B C y z : ℝ) 
  (eq1 : P = Q + R + 32)
  (eq2 : y = B + C + P + z)
  (eq3 : z = Q - R)
  (eq4 : B = 1/3 * P)
  (eq5 : C = 1/3 * P) :
  P = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_P_value_l3128_312821


namespace NUMINAMATH_CALUDE_consecutive_ones_count_is_3719_l3128_312862

/-- Fibonacci-like sequence for numbers without consecutive 1's -/
def F : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => F (n + 1) + F n

/-- The number of 12-digit integers with digits 1 or 2 and two consecutive 1's -/
def consecutive_ones_count : ℕ := 2^12 - F 11

theorem consecutive_ones_count_is_3719 : consecutive_ones_count = 3719 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_ones_count_is_3719_l3128_312862


namespace NUMINAMATH_CALUDE_candy_division_theorem_l3128_312820

/-- Represents the share of candy each person takes -/
structure CandyShare where
  al : Rat
  bert : Rat
  carl : Rat
  dana : Rat

/-- The function that calculates the remaining candy fraction -/
def remainingCandy (shares : CandyShare) : Rat :=
  1 - (shares.al + shares.bert + shares.carl + shares.dana)

/-- The theorem stating the correct remaining candy fraction -/
theorem candy_division_theorem (x : Rat) (shares : CandyShare) :
  shares.al = 3/7 ∧
  shares.bert = 2/7 * (1 - 3/7) ∧
  shares.carl = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7)) ∧
  shares.dana = 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7) - 1/7 * (1 - 3/7 - 2/7 * (1 - 3/7))) →
  remainingCandy shares = 584/2401 := by
  sorry

#check candy_division_theorem

end NUMINAMATH_CALUDE_candy_division_theorem_l3128_312820


namespace NUMINAMATH_CALUDE_max_runs_in_match_l3128_312897

/-- Represents the number of overs in a cricket match -/
def overs : ℕ := 20

/-- Represents the maximum number of runs a batsman can score -/
def max_runs : ℕ := 663

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored off a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Represents the total number of balls in the match -/
def total_balls : ℕ := overs * balls_per_over

/-- Theorem stating that under certain conditions, the maximum runs a batsman can score in the match is 663 -/
theorem max_runs_in_match : 
  ∃ (balls_faced : ℕ) (runs_per_ball : ℕ), 
    balls_faced ≤ total_balls ∧ 
    runs_per_ball ≤ max_runs_per_ball ∧ 
    balls_faced * runs_per_ball = max_runs :=
sorry

end NUMINAMATH_CALUDE_max_runs_in_match_l3128_312897


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_theorem_l3128_312818

/-- A triangle with an inscribed parallelogram -/
structure InscribedParallelogram where
  -- Triangle side lengths
  side1 : ℝ
  side2 : ℝ
  -- Parallelogram side on triangle base
  para_side : ℝ

/-- Properties of the inscribed parallelogram -/
def inscribed_parallelogram_properties (t : InscribedParallelogram) : Prop :=
  t.side1 = 9 ∧ t.side2 = 15 ∧ t.para_side = 6

/-- Theorem about the inscribed parallelogram -/
theorem inscribed_parallelogram_theorem (t : InscribedParallelogram) 
  (h : inscribed_parallelogram_properties t) :
  ∃ (other_side base : ℝ),
    other_side = 4 * Real.sqrt 2 ∧ 
    base = 18 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_parallelogram_theorem_l3128_312818


namespace NUMINAMATH_CALUDE_triangle_area_l3128_312836

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 → 
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3128_312836


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l3128_312812

theorem negative_two_cubed_equality : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l3128_312812


namespace NUMINAMATH_CALUDE_problem_statement_l3128_312831

theorem problem_statement (x : ℝ) : 
  let a := x^2 - 1
  let b := 2*x + 2
  (a + b ≥ 0) ∧ (max a b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3128_312831


namespace NUMINAMATH_CALUDE_closest_point_proof_l3128_312866

-- Define the cheese location
def cheese : ℝ × ℝ := (10, 10)

-- Define the mouse's path
def mouse_path (x : ℝ) : ℝ := -4 * x + 16

-- Define the point of closest approach
def closest_point : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem closest_point_proof :
  -- The closest point is on the mouse's path
  mouse_path closest_point.1 = closest_point.2 ∧
  -- The closest point is indeed the closest to the cheese
  ∀ x : ℝ, x ≠ closest_point.1 →
    (x - cheese.1)^2 + (mouse_path x - cheese.2)^2 >
    (closest_point.1 - cheese.1)^2 + (closest_point.2 - cheese.2)^2 ∧
  -- The sum of coordinates of the closest point is 10
  closest_point.1 + closest_point.2 = 10 :=
sorry

end NUMINAMATH_CALUDE_closest_point_proof_l3128_312866


namespace NUMINAMATH_CALUDE_positive_cube_sum_inequality_l3128_312815

theorem positive_cube_sum_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) : 
  (a + b) * (a^5 + b^5) ≥ 4 ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_cube_sum_inequality_l3128_312815


namespace NUMINAMATH_CALUDE_caravan_feet_heads_difference_l3128_312843

/-- Represents the number of feet for each animal type -/
def feet_per_animal : Nat → Nat
| 0 => 2  -- Hens
| 1 => 4  -- Goats
| 2 => 4  -- Camels
| 3 => 2  -- Keepers
| _ => 0  -- Other (shouldn't occur)

/-- Calculates the total number of feet for a given animal type and count -/
def total_feet (animal_type : Nat) (count : Nat) : Nat :=
  count * feet_per_animal animal_type

/-- Theorem: In a caravan with 60 hens, 35 goats, 6 camels, and 10 keepers,
    the difference between the total number of feet and the total number of heads is 193 -/
theorem caravan_feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let total_heads := hens + goats + camels + keepers
  let total_feet := total_feet 0 hens + total_feet 1 goats + total_feet 2 camels + total_feet 3 keepers
  total_feet - total_heads = 193 := by
  sorry

end NUMINAMATH_CALUDE_caravan_feet_heads_difference_l3128_312843


namespace NUMINAMATH_CALUDE_bee_multiple_l3128_312838

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l3128_312838


namespace NUMINAMATH_CALUDE_no_winning_strategy_l3128_312826

/-- Represents a strategy for deciding when to stop in the card game. -/
def Strategy : Type := List Bool → Bool

/-- Represents the state of the game at any point. -/
structure GameState :=
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The initial state of the game with a standard deck. -/
def initial_state : GameState :=
  { red_cards := 26, black_cards := 26 }

/-- Calculates the probability of winning given a game state and a strategy. -/
def winning_probability (state : GameState) (strategy : Strategy) : ℚ :=
  sorry

/-- Theorem stating that no strategy can have a winning probability greater than 0.5. -/
theorem no_winning_strategy :
  ∀ (strategy : Strategy),
    winning_probability initial_state strategy ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_no_winning_strategy_l3128_312826


namespace NUMINAMATH_CALUDE_ravi_has_six_nickels_l3128_312841

/-- Represents the number of nickels Ravi has -/
def n : ℕ := sorry

/-- Represents the number of quarters Ravi has -/
def quarters : ℕ := n + 2

/-- Represents the number of dimes Ravi has -/
def dimes : ℕ := quarters + 4

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total value of all coins in dollars -/
def total_value : ℚ := 350 / 100

theorem ravi_has_six_nickels :
  n * nickel_value + quarters * quarter_value + dimes * dime_value = total_value →
  n = 6 := by sorry

end NUMINAMATH_CALUDE_ravi_has_six_nickels_l3128_312841


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3128_312856

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 3 = 2 →
  a 4 + a 5 = 6 →
  a 5 + a 6 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3128_312856


namespace NUMINAMATH_CALUDE_triangle_properties_l3128_312858

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  c : ℝ
  b : ℝ
  B : ℝ

/-- The possible values for angle C in the triangle -/
def possible_C (t : Triangle) : Set ℝ :=
  {60, 120}

/-- The possible areas of the triangle -/
def possible_areas (t : Triangle) : Set ℝ :=
  {Real.sqrt 3 / 2, Real.sqrt 3 / 4}

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h_c : t.c = Real.sqrt 3)
  (h_b : t.b = 1)
  (h_B : t.B = 30) :
  (∃ (C : ℝ), C ∈ possible_C t) ∧
  (∃ (area : ℝ), area ∈ possible_areas t) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3128_312858


namespace NUMINAMATH_CALUDE_unique_star_solution_l3128_312801

/-- Definition of the ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists exactly one real number y such that 4 ⋆ y = 20 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 20 := by
  sorry

end NUMINAMATH_CALUDE_unique_star_solution_l3128_312801


namespace NUMINAMATH_CALUDE_four_n_plus_two_not_in_M_l3128_312888

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- Theorem stating that 4n+2 is not in M for any integer n -/
theorem four_n_plus_two_not_in_M (n : ℤ) : (4*n + 2) ∉ M := by
  sorry

end NUMINAMATH_CALUDE_four_n_plus_two_not_in_M_l3128_312888


namespace NUMINAMATH_CALUDE_exam_questions_count_l3128_312819

theorem exam_questions_count :
  ∀ (a b c : ℕ),
    b = 23 →
    c = 1 →
    a ≥ 1 →
    b ≥ 1 →
    c ≥ 1 →
    a ≥ (6 : ℚ) / 10 * (a + 2 * b + 3 * c) →
    a + b + c = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_questions_count_l3128_312819


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3128_312810

/-- Given a line L1 with equation 3x - y = 7 and a point P (2, -3),
    this theorem proves that the line L2 with equation y = -1/3x - 7/3
    is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 7
  let L2 : ℝ → ℝ → Prop := λ x y => y = -1/3 * x - 7/3
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → L2 x y → (3 * (-1/3) = -1)) ∧ 
  (L2 P.1 P.2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3128_312810


namespace NUMINAMATH_CALUDE_fourth_transaction_is_37_l3128_312878

/-- Represents the balance of class funds after a series of transactions -/
def class_funds (initial_balance : Int) (transactions : List Int) : Int :=
  initial_balance + transactions.sum

/-- Theorem: Given the initial balance and three transactions, 
    the fourth transaction must be 37 to reach the final balance of 82 -/
theorem fourth_transaction_is_37 
  (initial_balance : Int)
  (transaction1 transaction2 transaction3 : Int)
  (h1 : initial_balance = 0)
  (h2 : transaction1 = 230)
  (h3 : transaction2 = -75)
  (h4 : transaction3 = -110) :
  class_funds initial_balance [transaction1, transaction2, transaction3, 37] = 82 := by
  sorry

#eval class_funds 0 [230, -75, -110, 37]

end NUMINAMATH_CALUDE_fourth_transaction_is_37_l3128_312878


namespace NUMINAMATH_CALUDE_frank_reading_time_l3128_312804

theorem frank_reading_time (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 8) (h2 : total_pages = 576) :
  total_pages / pages_per_day = 72 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_time_l3128_312804


namespace NUMINAMATH_CALUDE_census_survey_is_D_census_suitability_criterion_l3128_312891

-- Define the survey options
inductive SurveyOption
| A : SurveyOption  -- West Lake Longjing tea quality
| B : SurveyOption  -- Xiaoshan TV station viewership
| C : SurveyOption  -- Xiaoshan people's happiness index
| D : SurveyOption  -- Classmates' health status

-- Define the property of being suitable for a census
def SuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Define the property of having a small quantity of subjects
def HasSmallQuantity (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.D => True
  | _ => False

-- Theorem stating that the survey suitable for a census is option D
theorem census_survey_is_D :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ option = SurveyOption.D :=
  sorry

-- Theorem stating that a survey is suitable for a census if and only if it has a small quantity of subjects
theorem census_suitability_criterion :
  ∀ (option : SurveyOption),
    SuitableForCensus option ↔ HasSmallQuantity option :=
  sorry

end NUMINAMATH_CALUDE_census_survey_is_D_census_suitability_criterion_l3128_312891


namespace NUMINAMATH_CALUDE_min_buttons_theorem_l3128_312898

def is_valid_button_set (buttons : Finset Nat) : Prop :=
  ∀ n : Nat, n ≥ 1 ∧ n ≤ 99999999 →
    (∃ digits : List Nat, digits.all (· ∈ buttons) ∧ digits.foldl (· * 10 + ·) 0 = n) ∨
    (∃ a b : Nat, 
      (∃ digits_a : List Nat, digits_a.all (· ∈ buttons) ∧ digits_a.foldl (· * 10 + ·) 0 = a) ∧
      (∃ digits_b : List Nat, digits_b.all (· ∈ buttons) ∧ digits_b.foldl (· * 10 + ·) 0 = b) ∧
      a + b = n)

theorem min_buttons_theorem :
  ∃ buttons : Finset Nat,
    buttons.card = 5 ∧
    is_valid_button_set buttons ∧
    ∀ smaller_buttons : Finset Nat, smaller_buttons.card < 5 → ¬is_valid_button_set smaller_buttons :=
by sorry

end NUMINAMATH_CALUDE_min_buttons_theorem_l3128_312898


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3128_312824

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3128_312824


namespace NUMINAMATH_CALUDE_alcohol_mixture_exists_l3128_312808

theorem alcohol_mixture_exists : ∃ (x y z : ℕ), 
  x + y + z = 560 ∧ 
  (70 * x + 64 * y + 50 * z : ℚ) = 60 * 560 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_exists_l3128_312808


namespace NUMINAMATH_CALUDE_waiter_customer_count_l3128_312827

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 → lunch_rush = 20.0 → later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l3128_312827


namespace NUMINAMATH_CALUDE_frequency_below_70kg_l3128_312811

def total_students : ℕ := 50

def weight_groups : List (ℝ × ℝ) := [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]

def frequencies : List ℕ := [6, 8, 15, 18, 3]

def students_below_70kg : ℕ := 29

theorem frequency_below_70kg :
  (students_below_70kg : ℝ) / total_students = 0.58 :=
sorry

end NUMINAMATH_CALUDE_frequency_below_70kg_l3128_312811


namespace NUMINAMATH_CALUDE_triangle_side_value_l3128_312877

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) : 
  (b^2 - c^2 + 2*a = 0) →
  (Real.tan C / Real.tan B = 3) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3128_312877


namespace NUMINAMATH_CALUDE_megan_zoo_pictures_l3128_312876

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- Represents the total number of pictures Megan took -/
def total_pictures : ℕ := zoo_pictures + 18

/-- Represents the number of pictures remaining after deletion -/
def remaining_pictures : ℕ := total_pictures - 31

theorem megan_zoo_pictures : 
  zoo_pictures = 15 ∧ 
  total_pictures = zoo_pictures + 18 ∧ 
  remaining_pictures = 2 :=
sorry

end NUMINAMATH_CALUDE_megan_zoo_pictures_l3128_312876


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3128_312875

theorem smallest_fraction_between (p q : ℕ+) 
  (h1 : (3 : ℚ) / 5 < p / q)
  (h2 : p / q < (5 : ℚ) / 8)
  (h3 : ∀ (r s : ℕ+), (3 : ℚ) / 5 < r / s → r / s < (5 : ℚ) / 8 → s ≤ q) :
  q - p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3128_312875


namespace NUMINAMATH_CALUDE_road_length_probability_l3128_312814

theorem road_length_probability : 
  ∀ (p_ab p_bc : ℝ),
    0 ≤ p_ab ∧ p_ab ≤ 1 →
    0 ≤ p_bc ∧ p_bc ≤ 1 →
    p_ab = 2/3 →
    p_bc = 1/2 →
    1 - (1 - p_ab) * (1 - p_bc) = 5/6 :=
by
  sorry

end NUMINAMATH_CALUDE_road_length_probability_l3128_312814


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_l3128_312882

theorem quadratic_polynomial_root : ∃ (a b c : ℝ), 
  (a = 1) ∧ 
  (∀ x : ℂ, x^2 + b*x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
  (a*x^2 + b*x + c = x^2 - 8*x + 17) :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_l3128_312882


namespace NUMINAMATH_CALUDE_triangle_side_length_l3128_312854

/-- Proves that in a triangle ABC with A = 60°, B = 45°, and c = 20, the length of side a is equal to 30√2 - 10√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → -- 60° in radians
  B = π / 4 → -- 45° in radians
  c = 20 →
  a = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3128_312854


namespace NUMINAMATH_CALUDE_salon_earnings_l3128_312870

/-- Calculates the total earnings from hair salon services -/
def total_earnings (haircut_price style_price coloring_price treatment_price : ℕ)
                   (haircuts styles colorings treatments : ℕ) : ℕ :=
  haircut_price * haircuts +
  style_price * styles +
  coloring_price * colorings +
  treatment_price * treatments

/-- Theorem stating that given specific prices and quantities, the total earnings are 871 -/
theorem salon_earnings :
  total_earnings 12 25 35 50 8 5 10 6 = 871 := by
  sorry

end NUMINAMATH_CALUDE_salon_earnings_l3128_312870


namespace NUMINAMATH_CALUDE_quadratic_roots_relationship_l3128_312889

theorem quadratic_roots_relationship (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, m₁ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, m₂ * x^2 + (1/3) * x + 1 = 0 ↔ x = x₃ ∨ x = x₄) →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relationship_l3128_312889


namespace NUMINAMATH_CALUDE_final_reflection_of_C_l3128_312837

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def C : ℝ × ℝ := (3, 2)

theorem final_reflection_of_C :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) C = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_final_reflection_of_C_l3128_312837


namespace NUMINAMATH_CALUDE_set_A_proof_l3128_312874

def U : Set Nat := {0, 1, 2, 3, 4, 5}

theorem set_A_proof (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ A) ∩ B = {0, 4})
  (h4 : (U \ A) ∩ (U \ B) = {3, 5}) :
  A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_A_proof_l3128_312874


namespace NUMINAMATH_CALUDE_cone_angle_and_ratio_l3128_312865

/-- For a cone with ratio k of total surface area to axial cross-section area,
    prove the angle between height and slant height, and permissible k values. -/
theorem cone_angle_and_ratio (k : ℝ) (α : ℝ) : k > π ∧ α = π/2 - 2 * Real.arctan (π/k) → 
  (π * (Real.sin α + 1)) / Real.cos α = k := by
  sorry

end NUMINAMATH_CALUDE_cone_angle_and_ratio_l3128_312865


namespace NUMINAMATH_CALUDE_square_sum_equals_nine_billion_four_million_l3128_312829

theorem square_sum_equals_nine_billion_four_million : (300000 : ℕ)^2 + (20000 : ℕ)^2 = 9004000000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_nine_billion_four_million_l3128_312829


namespace NUMINAMATH_CALUDE_fraction_subtraction_and_multiplication_l3128_312809

theorem fraction_subtraction_and_multiplication :
  (1 / 2 : ℚ) * ((5 / 6 : ℚ) - (1 / 9 : ℚ)) = 13 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_and_multiplication_l3128_312809


namespace NUMINAMATH_CALUDE_david_did_58_pushups_l3128_312813

/-- The number of push-ups David did -/
def davids_pushups (zachary_pushups : ℕ) (difference : ℕ) : ℕ :=
  zachary_pushups + difference

theorem david_did_58_pushups :
  davids_pushups 19 39 = 58 := by
  sorry

end NUMINAMATH_CALUDE_david_did_58_pushups_l3128_312813


namespace NUMINAMATH_CALUDE_litter_patrol_aluminum_cans_l3128_312879

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The total number of pieces of litter picked up by the Litter Patrol -/
def total_litter : ℕ := 18

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := total_litter - glass_bottles

theorem litter_patrol_aluminum_cans : aluminum_cans = 8 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_aluminum_cans_l3128_312879


namespace NUMINAMATH_CALUDE_edward_final_earnings_l3128_312823

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_final_earnings_l3128_312823


namespace NUMINAMATH_CALUDE_inequality_solution_l3128_312845

theorem inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x ≥ 9 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3128_312845


namespace NUMINAMATH_CALUDE_line_slope_l3128_312816

/-- A line in the xy-plane with y-intercept 20 and passing through (150, 600) has slope 580/150 -/
theorem line_slope (line : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ line ↔ y = (580/150) * x + 20) →
  (0, 20) ∈ line →
  (150, 600) ∈ line →
  ∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ line ↔ y = m * x + 20 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3128_312816


namespace NUMINAMATH_CALUDE_oliver_unwashed_shirts_l3128_312887

theorem oliver_unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 39)
  (h2 : long_sleeve = 47)
  (h3 : washed = 20) :
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end NUMINAMATH_CALUDE_oliver_unwashed_shirts_l3128_312887


namespace NUMINAMATH_CALUDE_line_transformation_l3128_312851

-- Define the original line
def original_line (x : ℝ) : ℝ := x - 2

-- Define the transformation (moving 3 units upwards)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 3

-- Define the new line
def new_line (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- Theorem statement
theorem line_transformation :
  ∃ k b : ℝ, 
    (∀ x : ℝ, transform original_line x = new_line k b x) ∧ 
    k = 1 ∧ 
    b = 1 ∧ 
    (∀ x : ℝ, new_line k b x > 0 → x > -1) :=
by sorry

end NUMINAMATH_CALUDE_line_transformation_l3128_312851


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3128_312806

theorem sufficient_condition_for_inequality (a b : ℝ) :
  Real.sqrt (a - 1) > Real.sqrt (b - 1) → a > b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3128_312806
