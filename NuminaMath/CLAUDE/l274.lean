import Mathlib

namespace NUMINAMATH_CALUDE_f_has_three_zeros_l274_27427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 2*x) * Real.log x + (a - 1/2) * x^2 + 2*(1 - a)*x + a

theorem f_has_three_zeros (a : ℝ) (h : a < -2) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l274_27427


namespace NUMINAMATH_CALUDE_total_cost_calculation_l274_27424

def regular_admission : ℚ := 8
def early_discount_percentage : ℚ := 25 / 100
def student_discount_percentage : ℚ := 10 / 100
def total_people : ℕ := 6
def students : ℕ := 2

def discounted_price : ℚ := regular_admission * (1 - early_discount_percentage)
def student_price : ℚ := discounted_price * (1 - student_discount_percentage)

theorem total_cost_calculation :
  let non_student_cost := (total_people - students) * discounted_price
  let student_cost := students * student_price
  non_student_cost + student_cost = 348 / 10 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l274_27424


namespace NUMINAMATH_CALUDE_shaded_area_of_semicircles_l274_27426

/-- The shaded area of semicircles in a pattern --/
theorem shaded_area_of_semicircles (d : ℝ) (l : ℝ) : 
  d = 3 → l = 24 → (l / d) * (π * d^2 / 8) = 18 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_semicircles_l274_27426


namespace NUMINAMATH_CALUDE_parabola_properties_l274_27412

/-- A parabola with equation x² = 3y is symmetric with respect to the y-axis and passes through
    the intersection points of x - y = 0 and x² + y² - 6y = 0 -/
theorem parabola_properties (x y : ℝ) :
  (x^2 = 3*y) →
  (∀ (x₀ : ℝ), (x₀^2 = 3*y) ↔ ((-x₀)^2 = 3*y)) ∧
  (∃ (x₁ y₁ : ℝ), x₁ - y₁ = 0 ∧ x₁^2 + y₁^2 - 6*y₁ = 0 ∧ x₁^2 = 3*y₁) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l274_27412


namespace NUMINAMATH_CALUDE_inverse_f_composition_l274_27456

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

theorem inverse_f_composition : 
  ∃ (f_inv : ℤ → ℤ), 
    (∀ (y : ℤ), f (f_inv y) = y) ∧ 
    (∀ (x : ℤ), f_inv (f x) = x) ∧
    f_inv (f_inv 122 / f_inv 18 + f_inv 50) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_composition_l274_27456


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l274_27464

/-- Calculate simple interest for a loan where the time period equals the interest rate -/
theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) : 
  principal = 1800 →
  rate = 5.93 →
  let interest := principal * rate * rate / 100
  ∃ ε > 0, |interest - 632.61| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l274_27464


namespace NUMINAMATH_CALUDE_circle_position_l274_27408

def circle_center : ℝ × ℝ := (-3, 4)
def circle_radius : ℝ := 3

theorem circle_position :
  let (x, y) := circle_center
  let r := circle_radius
  (abs y > r) ∧ (abs x = r) := by sorry

end NUMINAMATH_CALUDE_circle_position_l274_27408


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l274_27433

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2 / (2 + a)) + (1 / (a + 2 * b)) = 1) :
  (a + b ≥ Real.sqrt 2 + 1/2) ∧ 
  (a + b = Real.sqrt 2 + 1/2 ↔ a = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l274_27433


namespace NUMINAMATH_CALUDE_product_of_solutions_l274_27423

theorem product_of_solutions (y : ℝ) : (|y| = 3 * (|y| - 2)) → ∃ z : ℝ, (|z| = 3 * (|z| - 2)) ∧ (y * z = -9) := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l274_27423


namespace NUMINAMATH_CALUDE_seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l274_27452

theorem seventeen_to_fourteen_greater_than_thirtyone_to_eleven :
  (17 : ℝ)^14 > (31 : ℝ)^11 := by sorry

end NUMINAMATH_CALUDE_seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l274_27452


namespace NUMINAMATH_CALUDE_max_value_of_a_l274_27478

theorem max_value_of_a (a : ℝ) : 
  (∀ x > 1, a - x + Real.log (x * (x + 1)) ≤ 0) →
  a ≤ (1 + Real.sqrt 3) / 2 - Real.log ((3 / 2) + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l274_27478


namespace NUMINAMATH_CALUDE_base_conversion_l274_27479

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion :
  (binary_to_decimal [true, true, false, true, false, true]) = 43 ∧
  (decimal_to_base7 85) = [1, 5, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_l274_27479


namespace NUMINAMATH_CALUDE_first_project_breadth_l274_27413

/-- Represents a digging project with depth, length, breadth, and duration -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ
  duration : ℝ

/-- The volume of a digging project -/
def volume (p : DiggingProject) : ℝ := p.depth * p.length * p.breadth

/-- The first digging project with unknown breadth -/
def project1 (b : ℝ) : DiggingProject := {
  depth := 100,
  length := 25,
  breadth := b,
  duration := 12
}

/-- The second digging project -/
def project2 : DiggingProject := {
  depth := 75,
  length := 20,
  breadth := 50,
  duration := 12
}

/-- The theorem stating that the breadth of the first project is 30 meters -/
theorem first_project_breadth :
  ∃ b : ℝ, volume (project1 b) = volume project2 ∧ b = 30 := by
  sorry


end NUMINAMATH_CALUDE_first_project_breadth_l274_27413


namespace NUMINAMATH_CALUDE_junior_senior_ratio_l274_27435

theorem junior_senior_ratio (j s : ℕ) 
  (h1 : j > 0) (h2 : s > 0)
  (h3 : (j / 3 : ℚ) = (2 * s / 3 : ℚ)) : 
  j = 2 * s := by
sorry

end NUMINAMATH_CALUDE_junior_senior_ratio_l274_27435


namespace NUMINAMATH_CALUDE_tshirt_socks_price_difference_l274_27476

/-- The price difference between a t-shirt and socks -/
theorem tshirt_socks_price_difference 
  (jeans_price t_shirt_price socks_price : ℝ) 
  (h1 : jeans_price = 2 * t_shirt_price) 
  (h2 : jeans_price = 30) 
  (h3 : socks_price = 5) : 
  t_shirt_price - socks_price = 10 := by
sorry

end NUMINAMATH_CALUDE_tshirt_socks_price_difference_l274_27476


namespace NUMINAMATH_CALUDE_teacher_worksheets_l274_27462

theorem teacher_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 3 →
  graded_worksheets = 7 →
  remaining_problems = 24 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 15 := by
  sorry

end NUMINAMATH_CALUDE_teacher_worksheets_l274_27462


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_80_l274_27403

theorem closest_integer_to_cube_root_80 : 
  ∀ n : ℤ, |n - (80 : ℝ)^(1/3)| ≥ |4 - (80 : ℝ)^(1/3)| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_80_l274_27403


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l274_27439

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q ≤ q') →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l274_27439


namespace NUMINAMATH_CALUDE_inversion_number_reverse_l274_27467

/-- An array of 8 distinct integers -/
def Array8 := Fin 8 → ℤ

/-- The inversion number of an array -/
def inversionNumber (A : Array8) : ℕ :=
  sorry

/-- Theorem: Given an array of 8 distinct integers with inversion number 2,
    the inversion number of its reverse (excluding the last element) is at least 19 -/
theorem inversion_number_reverse (A : Array8) 
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j)
  (h_inv_num : inversionNumber A = 2) :
  inversionNumber (fun i => A (⟨7 - i.val, sorry⟩ : Fin 8)) ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_inversion_number_reverse_l274_27467


namespace NUMINAMATH_CALUDE_sqrt2_irrational_bound_l274_27463

theorem sqrt2_irrational_bound (p q : ℤ) (hq : q ≠ 0) :
  |Real.sqrt 2 - (p : ℝ) / (q : ℝ)| > 1 / (3 * (q : ℝ)^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_bound_l274_27463


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l274_27444

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_divisibility :
  (arithmetic_sequence_sum 1 8 313) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l274_27444


namespace NUMINAMATH_CALUDE_ball_ratio_l274_27477

theorem ball_ratio (total : ℕ) (blue red : ℕ) (green : ℕ := 3 * blue) 
  (yellow : ℕ := total - (blue + red + green)) 
  (h1 : total = 36) (h2 : blue = 6) (h3 : red = 4) :
  yellow / red = 2 :=
by sorry

end NUMINAMATH_CALUDE_ball_ratio_l274_27477


namespace NUMINAMATH_CALUDE_distinct_lunches_l274_27420

/-- The number of main course options available --/
def main_course_options : ℕ := 4

/-- The number of beverage options available --/
def beverage_options : ℕ := 3

/-- The number of snack options available --/
def snack_options : ℕ := 2

/-- The total number of distinct possible lunches --/
def total_lunches : ℕ := main_course_options * beverage_options * snack_options

/-- Theorem stating that the total number of distinct possible lunches is 24 --/
theorem distinct_lunches : total_lunches = 24 := by
  sorry

end NUMINAMATH_CALUDE_distinct_lunches_l274_27420


namespace NUMINAMATH_CALUDE_arccos_one_half_l274_27466

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_l274_27466


namespace NUMINAMATH_CALUDE_carpet_breadth_calculation_l274_27468

theorem carpet_breadth_calculation (b : ℝ) : 
  let first_length := 1.44 * b
  let second_length := 1.4 * first_length
  let second_breadth := 1.25 * b
  let second_area := second_length * second_breadth
  let cost_per_sqm := 45
  let total_cost := 4082.4
  second_area = total_cost / cost_per_sqm →
  b = 6.08 := by
sorry

end NUMINAMATH_CALUDE_carpet_breadth_calculation_l274_27468


namespace NUMINAMATH_CALUDE_dual_polyhedra_equal_radii_l274_27437

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  face_circumscribed_radius : ℝ

/-- Represents a pair of dual regular polyhedra -/
structure DualRegularPolyhedra where
  original : RegularPolyhedron
  dual : RegularPolyhedron

/-- Theorem: For dual regular polyhedra with equal inscribed sphere radii,
    their circumscribed sphere radii and face circumscribed circle radii are equal -/
theorem dual_polyhedra_equal_radii (p : DualRegularPolyhedra) 
    (h : p.original.inscribed_radius = p.dual.inscribed_radius) : 
    p.original.circumscribed_radius = p.dual.circumscribed_radius ∧ 
    p.original.face_circumscribed_radius = p.dual.face_circumscribed_radius := by
  sorry


end NUMINAMATH_CALUDE_dual_polyhedra_equal_radii_l274_27437


namespace NUMINAMATH_CALUDE_total_legs_on_farm_l274_27486

/-- The number of legs for each animal type -/
def duck_legs : ℕ := 2
def dog_legs : ℕ := 4

/-- The farm composition -/
def total_animals : ℕ := 11
def num_ducks : ℕ := 6
def num_dogs : ℕ := total_animals - num_ducks

/-- The theorem to prove -/
theorem total_legs_on_farm : 
  num_ducks * duck_legs + num_dogs * dog_legs = 32 := by sorry

end NUMINAMATH_CALUDE_total_legs_on_farm_l274_27486


namespace NUMINAMATH_CALUDE_saturday_price_calculation_l274_27422

theorem saturday_price_calculation (original_price : ℝ) 
  (h1 : original_price = 180) 
  (sale_discount : ℝ) (h2 : sale_discount = 0.5)
  (saturday_discount : ℝ) (h3 : saturday_discount = 0.2) : 
  original_price * (1 - sale_discount) * (1 - saturday_discount) = 72 := by
  sorry

end NUMINAMATH_CALUDE_saturday_price_calculation_l274_27422


namespace NUMINAMATH_CALUDE_sqrt_inequality_increasing_function_inequality_l274_27407

-- Part 1
theorem sqrt_inequality (x₁ x₂ : ℝ) (h1 : 0 ≤ x₁) (h2 : 0 ≤ x₂) (h3 : x₁ ≠ x₂) :
  (1/2) * (Real.sqrt x₁ + Real.sqrt x₂) < Real.sqrt ((x₁ + x₂) / 2) := by
  sorry

-- Part 2
theorem increasing_function_inequality {f : ℝ → ℝ} (h : Monotone f) 
  {a b : ℝ} (h1 : a + f a ≤ b + f b) : a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_increasing_function_inequality_l274_27407


namespace NUMINAMATH_CALUDE_geometric_sequence_expression_zero_l274_27491

/-- For a geometric sequence, the product of terms equidistant from the ends is constant -/
def geometric_sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 * a n = a 2 * a (n - 1)

/-- The expression (a₁aₙ)² - a₂a₄aₙ₋₁aₙ₋₃ equals zero for any geometric sequence -/
theorem geometric_sequence_expression_zero (a : ℕ → ℝ) (n : ℕ) 
  (h : geometric_sequence_property a) : 
  (a 1 * a n)^2 - (a 2 * a 4 * a (n-1) * a (n-3)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_expression_zero_l274_27491


namespace NUMINAMATH_CALUDE_circle_properties_l274_27481

/-- Given a circle with circumference 36 cm, prove its radius, diameter, and area -/
theorem circle_properties (C : ℝ) (h : C = 36) :
  ∃ (r d A : ℝ),
    r = 18 / Real.pi ∧
    d = 36 / Real.pi ∧
    A = 324 / Real.pi ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l274_27481


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l274_27473

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = 16) :
  a 3^2 + 2 * a 2 * a 6 + a 3 * a 7 = 400 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l274_27473


namespace NUMINAMATH_CALUDE_batsman_average_increase_l274_27434

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the average runs per innings -/
def calculateAverage (totalRuns : ℕ) (innings : ℕ) : ℚ :=
  (totalRuns : ℚ) / (innings : ℚ)

theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    let newTotalRuns := b.totalRuns + 55
    let newInnings := b.innings + 1
    let newAverage := calculateAverage newTotalRuns newInnings
    newAverage = 44 →
    newAverage - b.average = 1 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l274_27434


namespace NUMINAMATH_CALUDE_complex_expression_equals_half_l274_27421

theorem complex_expression_equals_half :
  |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_half_l274_27421


namespace NUMINAMATH_CALUDE_bacteria_count_l274_27432

theorem bacteria_count (original : ℕ) (increase : ℕ) (current : ℕ) : 
  original = 600 → increase = 8317 → current = original + increase → current = 8917 := by
sorry

end NUMINAMATH_CALUDE_bacteria_count_l274_27432


namespace NUMINAMATH_CALUDE_population_change_theorem_l274_27414

/-- Calculates the population after three years of changes --/
def population_after_three_years (initial_population : ℕ) : ℕ :=
  let year1 := (initial_population * 80) / 100
  let year2_increase := (year1 * 110) / 100
  let year2 := (year2_increase * 95) / 100
  let year3_increase := (year2 * 108) / 100
  (year3_increase * 75) / 100

/-- Theorem stating that the population after three years of changes is 10157 --/
theorem population_change_theorem :
  population_after_three_years 15000 = 10157 := by
  sorry

end NUMINAMATH_CALUDE_population_change_theorem_l274_27414


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l274_27453

theorem austin_robot_purchase (num_robots : ℕ) (cost_per_robot tax change : ℚ) : 
  num_robots = 7 →
  cost_per_robot = 8.75 →
  tax = 7.22 →
  change = 11.53 →
  (num_robots : ℚ) * cost_per_robot + tax + change = 80 := by
  sorry

end NUMINAMATH_CALUDE_austin_robot_purchase_l274_27453


namespace NUMINAMATH_CALUDE_bag_of_balls_l274_27419

theorem bag_of_balls (total : ℕ) (blue : ℕ) (green : ℕ) : 
  blue = 6 →
  blue + green = total →
  (blue : ℚ) / total = 1 / 4 →
  green = 18 := by
sorry

end NUMINAMATH_CALUDE_bag_of_balls_l274_27419


namespace NUMINAMATH_CALUDE_polyhedral_angle_sum_lt_360_l274_27436

/-- A polyhedral angle is represented by a list of planar angles (in degrees) -/
def PolyhedralAngle := List Float

/-- The sum of planar angles in a polyhedral angle is less than 360° -/
theorem polyhedral_angle_sum_lt_360 (pa : PolyhedralAngle) : 
  pa.sum < 360 := by sorry

end NUMINAMATH_CALUDE_polyhedral_angle_sum_lt_360_l274_27436


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_conditions_l274_27457

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_conditions (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ)
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2)) ↔
  (x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2 ∧ x₁ = x₂ ∧ z₁ = z₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_conditions_l274_27457


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l274_27497

/-- Given a point A with coordinates (m, n) that are (-3, 2) with respect to the origin, 
    prove that m + n = -1 -/
theorem sum_of_coordinates (m n : ℝ) (h : (m, n) = (-3, 2)) : m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l274_27497


namespace NUMINAMATH_CALUDE_one_rhythm_for_specific_phrase_l274_27460

/-- Represents the duration of a note in terms of fractions of a measure -/
structure NoteDuration where
  numerator : ℕ
  denominator : ℕ+

/-- Represents a musical phrase -/
structure MusicalPhrase where
  measures : ℕ
  note_duration : NoteDuration
  no_rests : Bool

/-- Counts the number of different rhythms possible for a given musical phrase -/
def count_rhythms (phrase : MusicalPhrase) : ℕ :=
  sorry

/-- Theorem stating that a 2-measure phrase with notes lasting 1/8 of 1/4 of a measure and no rests has only one possible rhythm -/
theorem one_rhythm_for_specific_phrase :
  ∀ (phrase : MusicalPhrase),
    phrase.measures = 2 ∧
    phrase.note_duration = { numerator := 1, denominator := 32 } ∧
    phrase.no_rests = true →
    count_rhythms phrase = 1 :=
  sorry

end NUMINAMATH_CALUDE_one_rhythm_for_specific_phrase_l274_27460


namespace NUMINAMATH_CALUDE_john_finishes_ahead_l274_27445

/-- The distance John finishes ahead of Steve in a race --/
def distance_john_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time - initial_distance) - (steve_speed * push_time)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 12
  let push_time : ℝ := 28
  distance_john_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end NUMINAMATH_CALUDE_john_finishes_ahead_l274_27445


namespace NUMINAMATH_CALUDE_school_classrooms_l274_27495

theorem school_classrooms (total_students : ℕ) (desks_type1 : ℕ) (desks_type2 : ℕ) :
  total_students = 400 →
  desks_type1 = 30 →
  desks_type2 = 25 →
  ∃ (num_classrooms : ℕ),
    num_classrooms > 0 ∧
    (num_classrooms / 3) * desks_type1 + (2 * num_classrooms / 3) * desks_type2 = total_students ∧
    num_classrooms = 15 := by
  sorry

end NUMINAMATH_CALUDE_school_classrooms_l274_27495


namespace NUMINAMATH_CALUDE_circle_ratio_l274_27480

theorem circle_ratio (a b : ℝ) (h : a > 0) (h' : b > 0) 
  (h_area : π * b^2 - π * a^2 = 4 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_l274_27480


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l274_27465

theorem arithmetic_evaluation : 6 + 4 / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l274_27465


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l274_27461

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → |x| ≥ |r| :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l274_27461


namespace NUMINAMATH_CALUDE_season_games_calculation_l274_27400

/-- Represents the number of games played by a team in a season -/
def total_games : ℕ := 125

/-- Represents the number of games in the first part of the season -/
def first_games : ℕ := 100

/-- Represents the win percentage for the first part of the season -/
def first_win_percentage : ℚ := 75 / 100

/-- Represents the win percentage for the remaining games -/
def remaining_win_percentage : ℚ := 50 / 100

/-- Represents the overall win percentage for the entire season -/
def overall_win_percentage : ℚ := 70 / 100

theorem season_games_calculation :
  let remaining_games := total_games - first_games
  (first_win_percentage * first_games + remaining_win_percentage * remaining_games) / total_games = overall_win_percentage :=
by sorry

end NUMINAMATH_CALUDE_season_games_calculation_l274_27400


namespace NUMINAMATH_CALUDE_only_negative_four_less_than_negative_three_l274_27483

theorem only_negative_four_less_than_negative_three :
  let numbers : List ℝ := [-4, -2.8, 0, |-4|]
  ∀ x ∈ numbers, x < -3 ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_less_than_negative_three_l274_27483


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l274_27455

/-- Given two vectors a and b in R², where a = (3, 2) and b = (m, -1),
    if a and b are orthogonal, then m = 2/3 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (m, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l274_27455


namespace NUMINAMATH_CALUDE_trolley_passengers_count_l274_27438

/-- Calculates the number of people on a trolley after three stops -/
def trolley_passengers : ℕ :=
  let initial := 1  -- driver
  let first_stop := initial + 10
  let second_stop := first_stop - 3 + (2 * 10)
  let third_stop := second_stop - 18 + 2
  third_stop

theorem trolley_passengers_count : trolley_passengers = 12 := by
  sorry

end NUMINAMATH_CALUDE_trolley_passengers_count_l274_27438


namespace NUMINAMATH_CALUDE_pat_shark_photo_profit_l274_27429

/-- Calculates the expected profit for Pat's shark photo hunting trip. -/
theorem pat_shark_photo_profit :
  let photo_earnings : ℕ → ℚ := λ n => 15 * n
  let sharks_per_hour : ℕ := 6
  let fuel_cost_per_hour : ℚ := 50
  let hunting_hours : ℕ := 5
  let total_sharks : ℕ := sharks_per_hour * hunting_hours
  let total_earnings : ℚ := photo_earnings total_sharks
  let total_fuel_cost : ℚ := fuel_cost_per_hour * hunting_hours
  let profit : ℚ := total_earnings - total_fuel_cost
  profit = 200 := by
sorry


end NUMINAMATH_CALUDE_pat_shark_photo_profit_l274_27429


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l274_27454

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^4 : Polynomial ℝ) + 3 * X^3 - 4 = (X^2 + X - 3) * q + (5 * X - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l274_27454


namespace NUMINAMATH_CALUDE_bug_return_probability_l274_27492

/-- Probability of returning to the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | 1 => 0
  | (n + 2) => (1 / 3) * (1 - Q (n + 1)) + (1 / 3) * Q n

/-- The probability of returning to the starting vertex on the tenth move is 34817/59049 -/
theorem bug_return_probability :
  Q 10 = 34817 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l274_27492


namespace NUMINAMATH_CALUDE_publishing_break_even_l274_27493

/-- A publishing company's break-even point calculation -/
theorem publishing_break_even 
  (fixed_cost : ℝ) 
  (variable_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : fixed_cost = 50000)
  (h2 : variable_cost = 4)
  (h3 : selling_price = 9) :
  ∃ x : ℝ, x = 10000 ∧ selling_price * x = fixed_cost + variable_cost * x :=
sorry

end NUMINAMATH_CALUDE_publishing_break_even_l274_27493


namespace NUMINAMATH_CALUDE_parallel_resistors_l274_27485

theorem parallel_resistors (x y r : ℝ) (hx : x = 3) (hy : y = 5) 
  (hr : 1 / r = 1 / x + 1 / y) : r = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l274_27485


namespace NUMINAMATH_CALUDE_prob_same_length_hexagon_l274_27488

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements in T -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 17 / 35

theorem prob_same_length_hexagon :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) /
  (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end NUMINAMATH_CALUDE_prob_same_length_hexagon_l274_27488


namespace NUMINAMATH_CALUDE_cupcake_count_l274_27470

theorem cupcake_count (initial : ℕ) (sold : ℕ) (additional : ℕ) : 
  initial ≥ sold → initial - sold + additional = (initial - sold) + additional := by
  sorry

end NUMINAMATH_CALUDE_cupcake_count_l274_27470


namespace NUMINAMATH_CALUDE_product_of_roots_l274_27440

theorem product_of_roots (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (4 * y) * Real.sqrt (25 * y) = 50) : 
  x * y = Real.sqrt (25 / 24) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l274_27440


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l274_27458

/-- The volume of a cylinder formed by rotating a square around its vertical line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 20 →
  volume = π * (side_length / 2)^2 * side_length →
  volume = 2000 * π := by
sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l274_27458


namespace NUMINAMATH_CALUDE_triangle_point_movement_l274_27418

theorem triangle_point_movement (AB BC : ℝ) (v_P v_Q : ℝ) (area_PBQ : ℝ) : 
  AB = 6 →
  BC = 8 →
  v_P = 1 →
  v_Q = 2 →
  area_PBQ = 5 →
  ∃ t : ℝ, t = 1 ∧ 
    (1/2) * (AB - t * v_P) * (t * v_Q) = area_PBQ ∧
    t * v_Q ≤ BC :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_movement_l274_27418


namespace NUMINAMATH_CALUDE_fraction_equality_l274_27451

theorem fraction_equality (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l274_27451


namespace NUMINAMATH_CALUDE_book_sale_profit_l274_27404

theorem book_sale_profit (total_cost : ℝ) (loss_percentage : ℝ) (gain_percentage1 : ℝ) (gain_percentage2 : ℝ) 
  (h1 : total_cost = 1080)
  (h2 : loss_percentage = 0.1)
  (h3 : gain_percentage1 = 0.15)
  (h4 : gain_percentage2 = 0.25)
  (h5 : (1 - loss_percentage) * (total_cost / 2) = 
        (1 + gain_percentage1) * (total_cost * 2 / 6) + 
        (1 + gain_percentage2) * (total_cost / 6)) :
  total_cost / 2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l274_27404


namespace NUMINAMATH_CALUDE_window_purchase_savings_l274_27431

/-- The regular price of a window in dollars -/
def regular_price : ℕ := 100

/-- The number of windows Alice needs -/
def alice_windows : ℕ := 9

/-- The number of windows Bob needs -/
def bob_windows : ℕ := 11

/-- The number of windows purchased that qualify for the special deal -/
def special_deal_threshold : ℕ := 10

/-- The number of free windows given in the special deal -/
def free_windows : ℕ := 2

/-- Calculate the cost of windows with the special deal applied -/
def cost_with_deal (n : ℕ) : ℕ :=
  let sets := n / special_deal_threshold
  let remainder := n % special_deal_threshold
  (n - sets * free_windows) * regular_price

/-- The main theorem stating the savings when purchasing together -/
theorem window_purchase_savings :
  cost_with_deal alice_windows + cost_with_deal bob_windows -
  cost_with_deal (alice_windows + bob_windows) = 200 := by
  sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l274_27431


namespace NUMINAMATH_CALUDE_abs_neg_six_l274_27474

theorem abs_neg_six : |(-6 : ℤ)| = 6 := by sorry

end NUMINAMATH_CALUDE_abs_neg_six_l274_27474


namespace NUMINAMATH_CALUDE_angle_bak_is_right_angle_l274_27409

-- Define the tetrahedron and its points
variable (A B C D K : EuclideanSpace ℝ (Fin 3))

-- Define the angles
def angle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

-- State the conditions
variable (h1 : angle B A C + angle B A D = Real.pi)
variable (h2 : angle C A K = angle K A D)

-- State the theorem
theorem angle_bak_is_right_angle : angle B A K = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_angle_bak_is_right_angle_l274_27409


namespace NUMINAMATH_CALUDE_calculation_proof_l274_27490

theorem calculation_proof : 1525 + 140 / 70 - 225 = 1302 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l274_27490


namespace NUMINAMATH_CALUDE_flour_calculation_l274_27405

/-- Calculates the required cups of flour given the original recipe ratio, scaling factor, and amount of butter used. -/
def required_flour (original_butter original_flour scaling_factor butter_used : ℚ) : ℚ :=
  (butter_used / original_butter) * scaling_factor * original_flour

/-- Proves that given the specified conditions, the required amount of flour is 30 cups. -/
theorem flour_calculation (original_butter original_flour scaling_factor butter_used : ℚ) 
  (h1 : original_butter = 2)
  (h2 : original_flour = 5)
  (h3 : scaling_factor = 4)
  (h4 : butter_used = 12) :
  required_flour original_butter original_flour scaling_factor butter_used = 30 := by
sorry

#eval required_flour 2 5 4 12

end NUMINAMATH_CALUDE_flour_calculation_l274_27405


namespace NUMINAMATH_CALUDE_special_rectangle_area_l274_27469

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  x : ℝ  -- Length
  y : ℝ  -- Width
  perimeter_eq : x + y = 30  -- Half perimeter equals 30
  side_diff : x = y + 3

/-- The area of a SpecialRectangle -/
def area (r : SpecialRectangle) : ℝ := r.x * r.y

/-- Theorem stating the area of the SpecialRectangle -/
theorem special_rectangle_area :
  ∀ r : SpecialRectangle, area r = 222.75 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l274_27469


namespace NUMINAMATH_CALUDE_number_of_students_l274_27447

/-- Proves the number of students in a class given certain conditions about average ages --/
theorem number_of_students (n : ℕ) (student_avg : ℚ) (new_avg : ℚ) (staff_age : ℕ) 
  (h1 : student_avg = 16)
  (h2 : new_avg = student_avg + 1)
  (h3 : staff_age = 49)
  (h4 : (n * student_avg + staff_age) / (n + 1) = new_avg) : 
  n = 32 := by
  sorry

#check number_of_students

end NUMINAMATH_CALUDE_number_of_students_l274_27447


namespace NUMINAMATH_CALUDE_waiting_by_tree_only_random_l274_27415

/-- Represents an idiom --/
inductive Idiom
  | CatchingTurtleInJar
  | WaitingByTreeForRabbit
  | RisingTideLiftAllBoats
  | FishingForMoonInWater

/-- Predicate to determine if an idiom describes a random event --/
def is_random_event (i : Idiom) : Prop :=
  match i with
  | Idiom.WaitingByTreeForRabbit => true
  | _ => false

/-- Theorem stating that "Waiting by a tree for a rabbit" is the only idiom
    among the given options that describes a random event --/
theorem waiting_by_tree_only_random :
  ∀ (i : Idiom), is_random_event i ↔ i = Idiom.WaitingByTreeForRabbit :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_tree_only_random_l274_27415


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l274_27406

/-- Proves that given a car traveling for two hours with an average speed of 45 km/h
    and a speed of 60 km/h in the first hour, the speed in the second hour must be 30 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (total_time : ℝ)
  (h_average_speed : average_speed = 45)
  (h_first_hour_speed : first_hour_speed = 60)
  (h_total_time : total_time = 2)
  : (2 * average_speed - first_hour_speed = 30) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l274_27406


namespace NUMINAMATH_CALUDE_pizza_slices_per_child_l274_27472

/-- Calculates the number of pizza slices each child wants given the following conditions:
  * There are 2 adults and 6 children
  * Each adult wants 3 slices
  * They order 3 pizzas with 4 slices each
-/
theorem pizza_slices_per_child 
  (num_adults : Nat) 
  (num_children : Nat) 
  (slices_per_adult : Nat) 
  (num_pizzas : Nat) 
  (slices_per_pizza : Nat) 
  (h1 : num_adults = 2) 
  (h2 : num_children = 6) 
  (h3 : slices_per_adult = 3) 
  (h4 : num_pizzas = 3) 
  (h5 : slices_per_pizza = 4) : 
  (num_pizzas * slices_per_pizza - num_adults * slices_per_adult) / num_children = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_child_l274_27472


namespace NUMINAMATH_CALUDE_bobby_blocks_l274_27441

theorem bobby_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 2 → given_blocks = 6 → initial_blocks + given_blocks = 8 :=
by sorry

end NUMINAMATH_CALUDE_bobby_blocks_l274_27441


namespace NUMINAMATH_CALUDE_expected_steps_l274_27430

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The probability of moving in any direction -/
def moveProbability : ℚ := 1/4

/-- Roger's starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- Function to determine if a point can be reached more quickly by a different route -/
def canReachQuicker (path : List Point) : Bool :=
  sorry

/-- The expected number of additional steps after the initial step -/
def e₁ : ℚ := 8/3

/-- The expected number of additional steps after moving perpendicular -/
def e₂ : ℚ := 2

/-- The main theorem: The expected number of steps Roger takes before he stops is 11/3 -/
theorem expected_steps :
  let totalSteps := 1 + e₁
  totalSteps = 11/3 := by sorry

end NUMINAMATH_CALUDE_expected_steps_l274_27430


namespace NUMINAMATH_CALUDE_xiao_ding_distance_to_school_l274_27449

/-- Proof that Xiao Ding's distance to school is 60 meters -/
theorem xiao_ding_distance_to_school : 
  ∀ (xw xd xc xz : ℝ),
  xw + xd + xc + xz = 705 →  -- Total distance condition
  xw = 4 * xd →              -- Xiao Wang's distance condition
  xc = xw / 2 + 20 →         -- Xiao Chen's distance condition
  xz = 2 * xc - 15 →         -- Xiao Zhang's distance condition
  xd = 60 := by              -- Conclusion: Xiao Ding's distance is 60 meters
sorry

end NUMINAMATH_CALUDE_xiao_ding_distance_to_school_l274_27449


namespace NUMINAMATH_CALUDE_hydrochloric_acid_moles_required_l274_27446

/-- Represents a chemical substance with its coefficient in a chemical equation -/
structure Substance where
  name : String
  coefficient : ℕ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def sodium_bisulfite : Substance := ⟨"NaHSO3", 1⟩
def hydrochloric_acid : Substance := ⟨"HCl", 1⟩
def sodium_chloride : Substance := ⟨"NaCl", 1⟩
def water : Substance := ⟨"H2O", 1⟩
def sulfur_dioxide : Substance := ⟨"SO2", 1⟩

def reaction : Reaction :=
  ⟨[sodium_bisulfite, hydrochloric_acid], [sodium_chloride, water, sulfur_dioxide]⟩

/-- The number of moles of a substance required or produced in a reaction -/
def moles_required (s : Substance) (n : ℕ) : ℕ := s.coefficient * n

theorem hydrochloric_acid_moles_required :
  moles_required hydrochloric_acid 2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_moles_required_l274_27446


namespace NUMINAMATH_CALUDE_marks_interest_earned_l274_27498

/-- Calculates the interest earned on an investment with annual compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Mark's investment -/
theorem marks_interest_earned :
  let principal : ℝ := 1500
  let rate : ℝ := 0.02
  let years : ℕ := 8
  abs (interestEarned principal rate years - 257.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_marks_interest_earned_l274_27498


namespace NUMINAMATH_CALUDE_unique_number_not_in_range_l274_27496

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ -d/c → f x = (a*x + b)/(c*x + d))
  (h11 : f 11 = 11)
  (h41 : f 41 = 41)
  (hinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = a/12 :=
sorry

end NUMINAMATH_CALUDE_unique_number_not_in_range_l274_27496


namespace NUMINAMATH_CALUDE_monochromatic_sequence_exists_l274_27489

def S (n : ℕ) : ℕ := (n * (n^2 + 5)) / 6

def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i < n - 1, a i < a (i + 1)) ∧
  (∀ i < n - 2, a (i + 1) - a i ≤ a (i + 2) - a (i + 1))

theorem monochromatic_sequence_exists (n : ℕ) (h : n ≥ 2) :
  ∀ c : ℕ → Bool,
  ∃ a : ℕ → ℕ, ∃ color : Bool,
    (∀ i < n, a i ≤ S n) ∧
    (∀ i < n, c (a i) = color) ∧
    is_valid_sequence a n :=
sorry

end NUMINAMATH_CALUDE_monochromatic_sequence_exists_l274_27489


namespace NUMINAMATH_CALUDE_clock_hands_separation_l274_27459

/-- Represents the angle between clock hands at a given time -/
def clockHandAngle (m : ℕ) : ℝ :=
  |6 * m - 0.5 * m|

/-- Checks if the angle between clock hands is 1° (or equivalent) -/
def isOneDegreeSeparation (m : ℕ) : Prop :=
  ∃ k : ℤ, clockHandAngle m = 1 + 360 * k ∨ clockHandAngle m = 1 - 360 * k

theorem clock_hands_separation :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 720 →
    (isOneDegreeSeparation m ↔ m = 262 ∨ m = 458) :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_separation_l274_27459


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l274_27484

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 - 4}

-- Theorem statement
theorem complement_intersection_M_N : 
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l274_27484


namespace NUMINAMATH_CALUDE_kamal_math_marks_l274_27401

/-- Proves that given Kamal's marks in English, Physics, Chemistry, and Biology,
    with a specific average for all 5 subjects, his marks in Mathematics can be determined. -/
theorem kamal_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ)
  (h_english : english = 76)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_biology : biology = 85)
  (h_average : average = 75)
  (h_subjects : 5 * average = english + physics + chemistry + biology + mathematics) :
  mathematics = 65 :=
by sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l274_27401


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l274_27410

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 2 if the difference of their
    third terms equals 3 times the difference of their second terms minus their first term. -/
theorem sum_of_common_ratios_is_two
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0) :
  (k * p^2 - k * r^2 = 3 * (k * p - k * r) - k) →
  p + r = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l274_27410


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l274_27402

theorem abby_and_damon_weight (a b c d : ℝ)
  (h1 : a + b = 265)
  (h2 : b + c = 250)
  (h3 : c + d = 280) :
  a + d = 295 := by
  sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l274_27402


namespace NUMINAMATH_CALUDE_total_candies_l274_27487

/-- The total number of candies Linda and Chloe have together is 62, 
    given that Linda has 34 candies and Chloe has 28 candies. -/
theorem total_candies (linda_candies chloe_candies : ℕ) 
  (h1 : linda_candies = 34) 
  (h2 : chloe_candies = 28) : 
  linda_candies + chloe_candies = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l274_27487


namespace NUMINAMATH_CALUDE_first_number_value_l274_27411

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l274_27411


namespace NUMINAMATH_CALUDE_factorize_quadratic_l274_27425

theorem factorize_quadratic (a : ℝ) : a^2 - 8*a + 15 = (a-3)*(a-5) := by
  sorry

end NUMINAMATH_CALUDE_factorize_quadratic_l274_27425


namespace NUMINAMATH_CALUDE_sequence_property_l274_27494

def strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def gcd_property (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, gcd (a m) (a n) = a (gcd m n)

def least_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
  (∃ r s : ℕ, r < k ∧ k < s ∧ a k ^ 2 = a r * a s) ∧
  (∀ k' : ℕ, k' < k → ¬∃ r s : ℕ, r < k' ∧ k' < s ∧ a k' ^ 2 = a r * a s)

theorem sequence_property (a : ℕ → ℕ) (k r s : ℕ) :
  strictly_increasing a →
  gcd_property a →
  least_k a k →
  r < k →
  k < s →
  a k ^ 2 = a r * a s →
  r ∣ k ∧ k ∣ s :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l274_27494


namespace NUMINAMATH_CALUDE_x_range_l274_27482

theorem x_range (x : ℝ) (h : ∀ a > 0, x^2 < 1 + a) : -1 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l274_27482


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l274_27417

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 3 / 5, then the ratio of their areas is 9:25 -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 5) (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l274_27417


namespace NUMINAMATH_CALUDE_pedestrian_speed_ratio_l274_27416

/-- Two pedestrians depart simultaneously from point A in the same direction.
    The first pedestrian meets a tourist 20 minutes after leaving point A.
    The second pedestrian meets the tourist 5 minutes after the first pedestrian.
    The tourist arrives at point A 10 minutes after the second meeting. -/
theorem pedestrian_speed_ratio 
  (v₁ : ℝ) -- speed of the first pedestrian
  (v₂ : ℝ) -- speed of the second pedestrian
  (v : ℝ)  -- speed of the tourist
  (h₁ : v₁ > 0)
  (h₂ : v₂ > 0)
  (h₃ : v > 0)
  (h₄ : (1/3) * v₁ = (1/4) * v) -- first meeting point equation
  (h₅ : (5/12) * v₂ = (1/6) * v) -- second meeting point equation
  : v₁ / v₂ = 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_pedestrian_speed_ratio_l274_27416


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l274_27443

theorem jacket_price_calculation (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (sales_tax : ℝ) :
  initial_price = 120 ∧
  discount1 = 0.20 ∧
  discount2 = 0.25 ∧
  sales_tax = 0.05 →
  initial_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax) = 75.60 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l274_27443


namespace NUMINAMATH_CALUDE_two_digit_number_is_30_l274_27450

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := { n : ℕ × ℕ // n.1 < 10 ∧ n.2 < 10 }

/-- Converts a two-digit number to its decimal representation -/
def to_decimal (n : TwoDigitNumber) : ℚ :=
  n.val.1 * 10 + n.val.2

/-- Represents a repeating decimal of the form 2.xy̅ -/
def repeating_decimal (n : TwoDigitNumber) : ℚ :=
  2 + (to_decimal n) / 99

/-- The main theorem stating that the two-digit number satisfying the equation is 30 -/
theorem two_digit_number_is_30 :
  ∃ (n : TwoDigitNumber), 
    75 * (repeating_decimal n - (2 + (to_decimal n) / 100)) = 2 ∧
    to_decimal n = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_is_30_l274_27450


namespace NUMINAMATH_CALUDE_limit_x_to_x_as_x_approaches_zero_l274_27499

theorem limit_x_to_x_as_x_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |x^x - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_to_x_as_x_approaches_zero_l274_27499


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l274_27448

theorem quadratic_equation_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 3 = 0 ∧ m * y^2 - 2 * y + 3 = 0) ↔ 
  (m < 1/3 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l274_27448


namespace NUMINAMATH_CALUDE_ten_cut_patterns_l274_27471

/-- Represents a grid with cells that can be cut into rectangles and squares. -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (total_cells : ℕ)
  (removed_cells : ℕ)

/-- Represents a way to cut the grid. -/
structure CutPattern :=
  (rectangles : ℕ)
  (squares : ℕ)

/-- The number of valid cut patterns for a given grid. -/
def valid_cut_patterns (g : Grid) (p : CutPattern) : ℕ := sorry

/-- The main theorem stating that there are exactly 10 ways to cut the specific grid. -/
theorem ten_cut_patterns :
  ∃ (g : Grid) (p : CutPattern),
    g.rows = 3 ∧
    g.cols = 6 ∧
    g.total_cells = 17 ∧
    g.removed_cells = 1 ∧
    p.rectangles = 8 ∧
    p.squares = 1 ∧
    valid_cut_patterns g p = 10 := by sorry

end NUMINAMATH_CALUDE_ten_cut_patterns_l274_27471


namespace NUMINAMATH_CALUDE_todd_total_gum_l274_27475

-- Define the initial number of gum pieces Todd had
def initial_gum : ℕ := 38

-- Define the number of gum pieces Steve gave to Todd
def steve_gum : ℕ := 16

-- Theorem statement
theorem todd_total_gum : initial_gum + steve_gum = 54 := by
  sorry

end NUMINAMATH_CALUDE_todd_total_gum_l274_27475


namespace NUMINAMATH_CALUDE_hexagon_sectors_perimeter_l274_27428

/-- The perimeter of a shape formed by removing three equal sectors from a regular hexagon -/
def shaded_perimeter (sector_perimeter : ℝ) : ℝ :=
  3 * sector_perimeter

theorem hexagon_sectors_perimeter :
  ∀ (sector_perimeter : ℝ),
  sector_perimeter = 18 →
  shaded_perimeter sector_perimeter = 54 := by
sorry

end NUMINAMATH_CALUDE_hexagon_sectors_perimeter_l274_27428


namespace NUMINAMATH_CALUDE_gabby_fruit_problem_l274_27442

theorem gabby_fruit_problem (watermelons peaches plums : ℕ) : 
  peaches = watermelons + 12 →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  watermelons = 1 := by
sorry

end NUMINAMATH_CALUDE_gabby_fruit_problem_l274_27442
