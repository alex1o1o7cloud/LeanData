import Mathlib

namespace NUMINAMATH_CALUDE_exam_count_proof_l998_99855

theorem exam_count_proof (prev_avg : ℝ) (desired_avg : ℝ) (next_score : ℝ) :
  prev_avg = 84 →
  desired_avg = 86 →
  next_score = 100 →
  ∃ n : ℕ, n > 0 ∧ (n * desired_avg - (n - 1) * prev_avg = next_score) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_exam_count_proof_l998_99855


namespace NUMINAMATH_CALUDE_triangle_area_l998_99892

theorem triangle_area (base height : ℝ) (h1 : base = 6) (h2 : height = 8) :
  (1 / 2) * base * height = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l998_99892


namespace NUMINAMATH_CALUDE_theater_attendance_l998_99852

/-- Calculates the total number of attendees given ticket prices and revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - child_price * num_children) / adult_price
  num_adults + num_children

/-- Theorem stating that under the given conditions, the total number of attendees is 280 --/
theorem theater_attendance : total_attendees 60 25 14000 80 = 280 := by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l998_99852


namespace NUMINAMATH_CALUDE_inequality_proof_l998_99810

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l998_99810


namespace NUMINAMATH_CALUDE_proposition_relationship_l998_99858

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l998_99858


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l998_99881

theorem arithmetic_mean_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = (a + b + c) / 3 + 5)
  (h2 : (a + c) / 2 = (a + b + c) / 3 - 8) :
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l998_99881


namespace NUMINAMATH_CALUDE_distance_to_origin_l998_99844

theorem distance_to_origin : ∃ (M : ℝ × ℝ), 
  M = (-5, 12) ∧ Real.sqrt ((-5)^2 + 12^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l998_99844


namespace NUMINAMATH_CALUDE_function_property_l998_99834

def evenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem function_property (f : ℝ → ℝ) (heven : evenFunction f) (hdec : decreasingOnNegative f) :
  (∀ a : ℝ, f (1 - a) > f (2 * a - 1) ↔ 0 < a ∧ a < 2/3) :=
sorry

end NUMINAMATH_CALUDE_function_property_l998_99834


namespace NUMINAMATH_CALUDE_prime_factor_puzzle_l998_99870

theorem prime_factor_puzzle (a b c d w x y z : ℕ) : 
  (Nat.Prime w) → 
  (Nat.Prime x) → 
  (Nat.Prime y) → 
  (Nat.Prime z) → 
  (w < x) → 
  (x < y) → 
  (y < z) → 
  ((w^a) * (x^b) * (y^c) * (z^d) = 660) → 
  ((a + b) - (c + d) = 1) → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_puzzle_l998_99870


namespace NUMINAMATH_CALUDE_no_valid_partition_l998_99848

theorem no_valid_partition : ¬∃ (A B C : Set ℕ), 
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 1 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 1 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 1 ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_partition_l998_99848


namespace NUMINAMATH_CALUDE_fraction_to_zero_power_l998_99880

theorem fraction_to_zero_power :
  let a : ℤ := -573293
  let b : ℕ := 7903827
  (a : ℚ) / b ^ (0 : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_to_zero_power_l998_99880


namespace NUMINAMATH_CALUDE_min_t_value_l998_99861

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 2 * Real.sqrt 2) ^ 2 + (y - 1) ^ 2 = 1

-- Define points A and B
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (t : ℝ) : Prop :=
  circle_C P.1 P.2 ∧
  let AP := (P.1 + t, P.2)
  let BP := (P.1 - t, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- State the theorem
theorem min_t_value :
  ∀ t : ℝ, t > 0 →
  (∃ P : ℝ × ℝ, point_P_condition P t) →
  (∀ t' : ℝ, t' > 0 ∧ (∃ P : ℝ × ℝ, point_P_condition P t') → t' ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_t_value_l998_99861


namespace NUMINAMATH_CALUDE_students_left_proof_l998_99817

/-- The number of students who showed up initially -/
def initial_students : ℕ := 16

/-- The number of students who were checked out early -/
def checked_out_students : ℕ := 7

/-- The number of students left at the end of the day -/
def remaining_students : ℕ := initial_students - checked_out_students

theorem students_left_proof : remaining_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_left_proof_l998_99817


namespace NUMINAMATH_CALUDE_min_balls_for_five_same_color_l998_99864

/-- Given a bag with 10 red balls, 10 yellow balls, and 10 white balls,
    the minimum number of balls that must be drawn to ensure
    at least 5 balls of the same color is 13. -/
theorem min_balls_for_five_same_color (red yellow white : ℕ) 
  (h_red : red = 10) (h_yellow : yellow = 10) (h_white : white = 10) :
  ∃ (n : ℕ), n = 13 ∧ 
  ∀ (m : ℕ), m < n → 
  ∃ (r y w : ℕ), r + y + w = m ∧ r < 5 ∧ y < 5 ∧ w < 5 :=
sorry

end NUMINAMATH_CALUDE_min_balls_for_five_same_color_l998_99864


namespace NUMINAMATH_CALUDE_painting_area_calculation_l998_99829

theorem painting_area_calculation (price_per_sqft : ℝ) (total_cost : ℝ) (area : ℝ) :
  price_per_sqft = 15 →
  total_cost = 840 →
  area * price_per_sqft = total_cost →
  area = 56 := by
sorry

end NUMINAMATH_CALUDE_painting_area_calculation_l998_99829


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l998_99887

theorem inscribed_cylinder_height (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 8 → c = 3 →
  h^2 = r^2 - c^2 →
  h = Real.sqrt 55 := by
sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l998_99887


namespace NUMINAMATH_CALUDE_rd_cost_productivity_relation_l998_99897

/-- The R&D costs required to increase the average labor productivity by 1 million rubles per person -/
def rd_cost_per_unit_productivity : ℝ := 4576

/-- The current R&D costs in million rubles -/
def current_rd_cost : ℝ := 3157.61

/-- The change in average labor productivity in million rubles per person -/
def delta_productivity : ℝ := 0.69

/-- Theorem stating that the R&D costs required to increase the average labor productivity
    by 1 million rubles per person is equal to the ratio of current R&D costs to the change
    in average labor productivity -/
theorem rd_cost_productivity_relation :
  rd_cost_per_unit_productivity = current_rd_cost / delta_productivity := by
  sorry

end NUMINAMATH_CALUDE_rd_cost_productivity_relation_l998_99897


namespace NUMINAMATH_CALUDE_existence_of_special_number_l998_99801

/-- A function that returns the decimal representation of a natural number as a list of digits -/
def decimal_representation (n : ℕ) : List ℕ := sorry

/-- A function that counts the occurrences of a digit in a list of digits -/
def count_occurrences (digit : ℕ) (digits : List ℕ) : ℕ := sorry

/-- A function that interchanges two digits at given positions in a list of digits -/
def interchange_digits (digits : List ℕ) (pos1 pos2 : ℕ) : List ℕ := sorry

/-- A function that converts a list of digits back to a natural number -/
def from_digits (digits : List ℕ) : ℕ := sorry

/-- The set of prime divisors of a natural number -/
def prime_divisors (n : ℕ) : Set ℕ := sorry

theorem existence_of_special_number :
  ∃ n : ℕ,
    (∀ d : ℕ, d < 10 → count_occurrences d (decimal_representation n) ≥ 2006) ∧
    (∃ pos1 pos2 : ℕ,
      pos1 ≠ pos2 ∧
      let digits := decimal_representation n
      let m := from_digits (interchange_digits digits pos1 pos2)
      n ≠ m ∧
      prime_divisors n = prime_divisors m) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l998_99801


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l998_99891

/-- The trajectory of a point P satisfying |PF₁| + |PF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_is_line_segment (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-4, 0) → 
  F₂ = (4, 0) → 
  dist P F₁ + dist P F₂ = 8 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l998_99891


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_15_l998_99899

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_sum_of_digits_15 :
  isFirstYearAfter2020WithSumOfDigits15 2049 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_15_l998_99899


namespace NUMINAMATH_CALUDE_f_extrema_l998_99853

def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, f p q x ≤ f p q x₁) ∧ (∀ x : ℝ, f p q x ≥ f p q x₂) ∧ 
                 (f p q x₁ = 4/27) ∧ (f p q x₂ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l998_99853


namespace NUMINAMATH_CALUDE_fraction_less_than_sqrt_l998_99877

theorem fraction_less_than_sqrt (x : ℝ) (h : x > 0) : x / (1 + x) < Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_sqrt_l998_99877


namespace NUMINAMATH_CALUDE_journey_equation_l998_99860

/-- Given a journey with three parts, prove the relationship between total distance, 
    total time, speeds, and time spent on each part. -/
theorem journey_equation 
  (D T x y z t₁ t₂ t₃ : ℝ) 
  (h_total_time : T = t₁ + t₂ + t₃) 
  (h_total_distance : D = x * t₁ + y * t₂ + z * t₃) 
  (h_positive_speed : x > 0 ∧ y > 0 ∧ z > 0)
  (h_positive_time : t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0)
  (h_positive_total : D > 0 ∧ T > 0) :
  D = x * t₁ + y * (T - t₁ - t₃) + z * t₃ :=
sorry

end NUMINAMATH_CALUDE_journey_equation_l998_99860


namespace NUMINAMATH_CALUDE_last_group_markers_theorem_l998_99878

/-- Calculates the number of markers each student in the last group receives --/
def markers_per_last_student (total_students : ℕ) (marker_boxes : ℕ) (markers_per_box : ℕ)
  (first_group_students : ℕ) (first_group_markers_per_student : ℕ)
  (second_group_students : ℕ) (second_group_markers_per_student : ℕ) : ℕ :=
  let total_markers := marker_boxes * markers_per_box
  let first_group_markers := first_group_students * first_group_markers_per_student
  let second_group_markers := second_group_students * second_group_markers_per_student
  let remaining_markers := total_markers - first_group_markers - second_group_markers
  let last_group_students := total_students - first_group_students - second_group_students
  remaining_markers / last_group_students

/-- Theorem stating that under the given conditions, each student in the last group receives 6 markers --/
theorem last_group_markers_theorem :
  markers_per_last_student 30 22 5 10 2 15 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_last_group_markers_theorem_l998_99878


namespace NUMINAMATH_CALUDE_simplify_fraction_l998_99805

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l998_99805


namespace NUMINAMATH_CALUDE_hard_lens_price_l998_99839

/-- Represents the price of contact lenses and sales information -/
structure LensSales where
  soft_price : ℕ
  hard_price : ℕ
  soft_count : ℕ
  hard_count : ℕ
  total_sales : ℕ

/-- Theorem stating the price of hard contact lenses -/
theorem hard_lens_price (sales : LensSales) : 
  sales.soft_price = 150 ∧ 
  sales.soft_count = sales.hard_count + 5 ∧
  sales.soft_count + sales.hard_count = 11 ∧
  sales.total_sales = sales.soft_price * sales.soft_count + sales.hard_price * sales.hard_count ∧
  sales.total_sales = 1455 →
  sales.hard_price = 85 := by
sorry

end NUMINAMATH_CALUDE_hard_lens_price_l998_99839


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l998_99822

theorem min_value_of_complex_expression :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (z : ℂ), Complex.abs z = 2 →
  Complex.abs (z^2 - z + 1) ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l998_99822


namespace NUMINAMATH_CALUDE_quadratic_properties_l998_99840

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (x + a) * (x - a - 1)

-- State the theorem
theorem quadratic_properties (a : ℝ) (h_a : a > 0) :
  -- 1. Axis of symmetry
  (∃ (x : ℝ), x = 1/2 ∧ ∀ (y : ℝ), f a (x - y) = f a (x + y)) ∧
  -- 2. Vertex coordinates when maximum is 4
  (∃ (x_max : ℝ), x_max ∈ Set.Icc (-1) 3 ∧ 
    (∀ (x : ℝ), x ∈ Set.Icc (-1) 3 → f a x ≤ 4) ∧ 
    f a x_max = 4 →
    f a (1/2) = -9/4) ∧
  -- 3. Range of t
  (∀ (t x₁ x₂ y₁ y₂ : ℝ),
    y₁ ≠ y₂ ∧
    t < x₁ ∧ x₁ < t + 1 ∧
    t + 2 < x₂ ∧ x₂ < t + 3 ∧
    f a x₁ = y₁ ∧ f a x₂ = y₂ →
    t ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l998_99840


namespace NUMINAMATH_CALUDE_four_points_on_circle_l998_99841

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def on_same_circle (A B C D : Point) : Prop :=
  ∃ (center : Point) (r : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = r^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = r^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = r^2 ∧
    (center.x - D.x)^2 + (center.y - D.y)^2 = r^2

theorem four_points_on_circle :
  let A : Point := ⟨-1, 5⟩
  let B : Point := ⟨5, 5⟩
  let C : Point := ⟨-3, 1⟩
  let D : Point := ⟨6, -2⟩
  on_same_circle A B C D :=
by
  sorry

end NUMINAMATH_CALUDE_four_points_on_circle_l998_99841


namespace NUMINAMATH_CALUDE_multiply_special_polynomials_l998_99812

theorem multiply_special_polynomials (x : ℝ) : 
  (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^6 - 4096 := by
sorry

end NUMINAMATH_CALUDE_multiply_special_polynomials_l998_99812


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l998_99816

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l998_99816


namespace NUMINAMATH_CALUDE_equal_intersection_areas_l998_99886

/-- A tetrahedron with specific properties -/
structure Tetrahedron where
  opposite_edges_perpendicular : Bool
  vertical_segment : ℝ
  midplane_area : ℝ

/-- A sphere with a specific radius -/
structure Sphere where
  radius : ℝ

/-- The configuration of a tetrahedron and a sphere -/
structure Configuration where
  tetra : Tetrahedron
  sphere : Sphere
  radius_condition : sphere.radius^2 * π = tetra.midplane_area
  vertical_segment_condition : tetra.vertical_segment = 2 * sphere.radius

/-- The area of intersection of a shape with a plane -/
def intersection_area (height : ℝ) : Configuration → ℝ
  | _ => sorry

/-- The main theorem stating that the areas of intersection are equal for all heights -/
theorem equal_intersection_areas (config : Configuration) :
  ∀ h : ℝ, 0 ≤ h ∧ h ≤ config.tetra.vertical_segment →
    intersection_area h config = intersection_area (config.tetra.vertical_segment - h) config :=
  sorry

end NUMINAMATH_CALUDE_equal_intersection_areas_l998_99886


namespace NUMINAMATH_CALUDE_gcd_problem_l998_99827

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 35 n = 7 ∧ n = 77 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l998_99827


namespace NUMINAMATH_CALUDE_fourth_year_afforestation_l998_99803

/-- The area afforested in a given year, starting from an initial area and increasing by a fixed percentage annually. -/
def afforestedArea (initialArea : ℝ) (increaseRate : ℝ) (year : ℕ) : ℝ :=
  initialArea * (1 + increaseRate) ^ (year - 1)

/-- Theorem stating that given an initial afforestation of 10,000 acres and an annual increase of 20%, 
    the area afforested in the fourth year is 17,280 acres. -/
theorem fourth_year_afforestation :
  afforestedArea 10000 0.2 4 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_fourth_year_afforestation_l998_99803


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l998_99896

theorem rectangle_area_increase 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := 1.3 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.495 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l998_99896


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l998_99866

/-- Given that ax^2 + 21x + 9 is the square of a binomial, prove that a = 49/4 -/
theorem quadratic_is_square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 21 * x + 9 = (r * x + s)^2) → 
  a = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l998_99866


namespace NUMINAMATH_CALUDE_find_divisor_l998_99843

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 172 →
  quotient = 10 →
  remainder = 2 →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l998_99843


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l998_99830

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + 3 < 0) →
  a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l998_99830


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l998_99814

def ring_arrangements (total_rings : ℕ) (chosen_rings : ℕ) (fingers : ℕ) : ℕ :=
  (total_rings.choose chosen_rings) * (chosen_rings.factorial) * ((chosen_rings + fingers - 1).choose (fingers - 1))

theorem ring_arrangement_count :
  ring_arrangements 8 5 4 = 376320 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l998_99814


namespace NUMINAMATH_CALUDE_range_of_m_l998_99868

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ m > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l998_99868


namespace NUMINAMATH_CALUDE_max_product_xyz_l998_99874

theorem max_product_xyz (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hsum : x + y + z = 1) (heq : x = y) (hbound : x ≤ z ∧ z ≤ 2*x) :
  ∃ (max_val : ℝ), ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    a + b + c = 1 → a = b → 
    a ≤ c → c ≤ 2*a → 
    a * b * c ≤ max_val ∧ 
    max_val = 1 / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_product_xyz_l998_99874


namespace NUMINAMATH_CALUDE_projection_onto_common_vector_l998_99871

/-- Given two vectors v1 and v2 in ℝ², prove that their projection onto a common vector is v1 if v1 is orthogonal to (v2 - v1). -/
theorem projection_onto_common_vector (v1 v2 : ℝ × ℝ) :
  v1 = (3, 2) ∧ v2 = (1, 5) →
  let diff := (v2.1 - v1.1, v2.2 - v1.2)
  (v1.1 * diff.1 + v1.2 * diff.2 = 0) →
  ∃ (x y : ℝ), (∀ t : ℝ, (v1.1 + t * diff.1, v1.2 + t * diff.2) = (x, y)) :=
by sorry

#check projection_onto_common_vector

end NUMINAMATH_CALUDE_projection_onto_common_vector_l998_99871


namespace NUMINAMATH_CALUDE_gcd_problem_l998_99847

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = 17 * k) :
  Nat.gcd (Int.natAbs (2 * a ^ 2 + 33 * a + 85)) (Int.natAbs (a + 17)) = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l998_99847


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l998_99825

theorem fraction_product_theorem : 
  (7 : ℚ) / 4 * 8 / 14 * 28 / 16 * 24 / 36 * 49 / 35 * 40 / 25 * 63 / 42 * 32 / 48 = 56 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l998_99825


namespace NUMINAMATH_CALUDE_point_y_coordinate_l998_99808

/-- A straight line in the xy-plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given line has slope 2 and y-intercept 2 -/
def given_line : Line :=
  { slope := 2, y_intercept := 2 }

/-- The x-coordinate of the point in question is 239 -/
def given_x : ℝ := 239

/-- A point is on a line if its coordinates satisfy the line equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Theorem: The point on the given line with x-coordinate 239 has y-coordinate 480 -/
theorem point_y_coordinate :
  ∃ p : Point, p.x = given_x ∧ point_on_line p given_line ∧ p.y = 480 :=
sorry

end NUMINAMATH_CALUDE_point_y_coordinate_l998_99808


namespace NUMINAMATH_CALUDE_fourth_root_of_unity_l998_99869

theorem fourth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 7 ∧ 
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) = 
  Complex.exp (Complex.I * (2 * n * π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_unity_l998_99869


namespace NUMINAMATH_CALUDE_tim_tetrises_l998_99898

/-- The number of tetrises Tim scored -/
def num_tetrises (single_points tetris_points num_singles total_points : ℕ) : ℕ :=
  (total_points - num_singles * single_points) / tetris_points

/-- Theorem: Tim scored 4 tetrises -/
theorem tim_tetrises :
  let single_points : ℕ := 1000
  let tetris_points : ℕ := 8 * single_points
  let num_singles : ℕ := 6
  let total_points : ℕ := 38000
  num_tetrises single_points tetris_points num_singles total_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_tim_tetrises_l998_99898


namespace NUMINAMATH_CALUDE_system_equivalence_l998_99867

theorem system_equivalence (x y a b : ℝ) : 
  (2 * x + y = 5 ∧ a * x + 3 * y = -1) ∧
  (x - y = 1 ∧ 4 * x + b * y = 11) →
  a = -2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_system_equivalence_l998_99867


namespace NUMINAMATH_CALUDE_john_ultramarathon_distance_l998_99873

/-- Calculates the total distance John can run after training -/
def johnRunningDistance (initialTime : ℝ) (timeIncrease : ℝ) (initialSpeed : ℝ) (speedIncrease : ℝ) : ℝ :=
  (initialTime * (1 + timeIncrease)) * (initialSpeed + speedIncrease)

theorem john_ultramarathon_distance :
  johnRunningDistance 8 0.75 8 4 = 168 := by
  sorry

end NUMINAMATH_CALUDE_john_ultramarathon_distance_l998_99873


namespace NUMINAMATH_CALUDE_specific_isosceles_triangle_area_l998_99882

/-- Represents an isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ
  leg_difference : ℝ

/-- Calculates the area of the isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles triangle -/
theorem specific_isosceles_triangle_area :
  let t : IsoscelesTriangle := {
    altitude := 10,
    perimeter := 40,
    leg_difference := 2
  }
  area t = 81.2 := by sorry

end NUMINAMATH_CALUDE_specific_isosceles_triangle_area_l998_99882


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_gcd_of_54_87_172_l998_99883

theorem greatest_common_divisor_of_differences : Int → Int → Int → Prop :=
  fun a b c => 
    let diff1 := b - a
    let diff2 := c - b
    let diff3 := c - a
    Nat.gcd (Nat.gcd (Int.natAbs diff1) (Int.natAbs diff2)) (Int.natAbs diff3) = 1

theorem gcd_of_54_87_172 : greatest_common_divisor_of_differences 54 87 172 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_gcd_of_54_87_172_l998_99883


namespace NUMINAMATH_CALUDE_emily_holidays_l998_99856

/-- The number of days Emily takes off each month -/
def days_off_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Emily takes in a year -/
def total_holidays : ℕ := days_off_per_month * months_in_year

theorem emily_holidays : total_holidays = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_holidays_l998_99856


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l998_99879

/-- An isosceles triangle with specific measurements -/
structure IsoscelesTriangle where
  -- Base of the triangle
  base : ℝ
  -- Median from one of the equal sides
  median : ℝ
  -- Length of the equal sides
  side : ℝ
  -- The base is 4√2 cm
  base_eq : base = 4 * Real.sqrt 2
  -- The median is 5 cm
  median_eq : median = 5
  -- The triangle is isosceles (implied by the structure)

/-- 
Theorem: In an isosceles triangle with a base of 4√2 cm and a median of 5 cm 
from one of the equal sides, the length of the equal sides is 6 cm.
-/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l998_99879


namespace NUMINAMATH_CALUDE_pants_count_l998_99885

/-- Represents the number of each type of clothing item in a dresser -/
structure DresserContents where
  pants : ℕ
  shorts : ℕ
  shirts : ℕ

/-- The ratio of pants to shorts to shirts in the dresser -/
def clothingRatio : ℕ × ℕ × ℕ := (7, 7, 10)

/-- The number of shirts in the dresser -/
def shirtCount : ℕ := 20

/-- Checks if the given DresserContents satisfies the ratio condition -/
def satisfiesRatio (contents : DresserContents) : Prop :=
  contents.pants * clothingRatio.2.2 = contents.shirts * clothingRatio.1 ∧
  contents.shorts * clothingRatio.2.2 = contents.shirts * clothingRatio.2.1

theorem pants_count (contents : DresserContents) 
  (h_ratio : satisfiesRatio contents) 
  (h_shirts : contents.shirts = shirtCount) : 
  contents.pants = 14 := by
  sorry

end NUMINAMATH_CALUDE_pants_count_l998_99885


namespace NUMINAMATH_CALUDE_rationalize_denominator_l998_99826

theorem rationalize_denominator :
  ∃ (a b : ℝ), a + b * Real.sqrt 3 = -Real.sqrt 3 - 2 ∧
  (a + b * Real.sqrt 3) * (Real.sqrt 3 - 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l998_99826


namespace NUMINAMATH_CALUDE_inverse_sum_property_l998_99800

-- Define the function f and its properties
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + f (-x) = 2

-- Define the inverse function property
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_sum_property
  (f : ℝ → ℝ)
  (h_inv : has_inverse f)
  (h_prop : f_property f) :
  ∀ x : ℝ, f⁻¹ (2008 - x) + f⁻¹ (x - 2006) = 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l998_99800


namespace NUMINAMATH_CALUDE_degrees_minutes_to_decimal_l998_99818

-- Define the conversion factor from minutes to degrees
def minutes_to_degrees (m : ℚ) : ℚ := m / 60

-- Define the problem
theorem degrees_minutes_to_decimal (d : ℚ) (m : ℚ) :
  d + minutes_to_degrees m = 18.4 → d = 18 ∧ m = 24 :=
by sorry

end NUMINAMATH_CALUDE_degrees_minutes_to_decimal_l998_99818


namespace NUMINAMATH_CALUDE_probability_8_of_hearts_or_spade_l998_99836

def standard_deck : ℕ := 52

def probability_8_of_hearts : ℚ := 1 / standard_deck

def probability_spade : ℚ := 1 / 4

theorem probability_8_of_hearts_or_spade :
  probability_8_of_hearts + probability_spade = 7 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_8_of_hearts_or_spade_l998_99836


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l998_99875

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (x - y) * (y - z) * (z - x)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_powers_l998_99875


namespace NUMINAMATH_CALUDE_harry_buckets_per_round_l998_99889

theorem harry_buckets_per_round 
  (george_buckets : ℕ) 
  (total_buckets : ℕ) 
  (total_rounds : ℕ) 
  (h1 : george_buckets = 2)
  (h2 : total_buckets = 110)
  (h3 : total_rounds = 22) :
  (total_buckets - george_buckets * total_rounds) / total_rounds = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_buckets_per_round_l998_99889


namespace NUMINAMATH_CALUDE_smallest_overlap_coffee_tea_l998_99838

/-- The smallest possible percentage of adults who drink both coffee and tea,
    given that 50% drink coffee and 60% drink tea. -/
theorem smallest_overlap_coffee_tea : ℝ :=
  let coffee_drinkers : ℝ := 50
  let tea_drinkers : ℝ := 60
  let total_percentage : ℝ := 100
  min (coffee_drinkers + tea_drinkers - total_percentage) coffee_drinkers

end NUMINAMATH_CALUDE_smallest_overlap_coffee_tea_l998_99838


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l998_99851

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₀ : ℝ, f a x₀ ≤ |2*a - 1|) → 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l998_99851


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l998_99884

def p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

theorem necessary_but_not_sufficient_condition :
  (∃ (a b : ℝ), (a = -1 ∧ b = 2 ∨ a = -2 ∧ b = 2) ∧
    (∀ x, p x → a < x ∧ x < b) ∧
    (∃ y, a < y ∧ y < b ∧ ¬(p y))) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l998_99884


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l998_99828

/-- The volume of a right rectangular prism with face areas 6, 8, and 12 square inches is 24 cubic inches. -/
theorem right_rectangular_prism_volume (l w h : ℝ) 
  (area1 : l * w = 6)
  (area2 : w * h = 8)
  (area3 : l * h = 12) :
  l * w * h = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l998_99828


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l998_99821

/-- A parabola intersects a line at exactly one point if and only if b = 49/12 -/
theorem parabola_line_intersection (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 5*x + 4 = -2*x + 1) ↔ b = 49/12 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l998_99821


namespace NUMINAMATH_CALUDE_acute_angle_alpha_l998_99862

theorem acute_angle_alpha (α : Real) (h : 0 < α ∧ α < Real.pi / 2) 
  (eq : Real.cos (Real.pi / 6) * Real.sin α = Real.sqrt 3 / 4) : 
  α = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_alpha_l998_99862


namespace NUMINAMATH_CALUDE_valid_arrangements_l998_99802

/-- The number of boys. -/
def num_boys : ℕ := 4

/-- The number of girls. -/
def num_girls : ℕ := 3

/-- The number of people to be selected. -/
def num_selected : ℕ := 3

/-- The number of tasks. -/
def num_tasks : ℕ := 3

/-- The function to calculate the number of permutations. -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating the number of valid arrangements. -/
theorem valid_arrangements : 
  permutations (num_boys + num_girls) num_selected - 
  permutations num_boys num_selected = 186 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l998_99802


namespace NUMINAMATH_CALUDE_compound_line_chart_optimal_l998_99893

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | Bar
  | Pie
  | Line
  | Scatter
  | CompoundLine

/-- Represents the requirements for the chart -/
structure ChartRequirements where
  numStudents : Nat
  showComparison : Bool
  showChangesOverTime : Bool

/-- Determines if a chart type is optimal for given requirements -/
def isOptimalChart (chart : StatisticalChart) (req : ChartRequirements) : Prop :=
  chart = StatisticalChart.CompoundLine ∧
  req.numStudents = 2 ∧
  req.showComparison = true ∧
  req.showChangesOverTime = true

/-- Theorem stating that a compound line chart is optimal for the given scenario -/
theorem compound_line_chart_optimal (req : ChartRequirements) :
  req.numStudents = 2 →
  req.showComparison = true →
  req.showChangesOverTime = true →
  isOptimalChart StatisticalChart.CompoundLine req :=
by sorry

end NUMINAMATH_CALUDE_compound_line_chart_optimal_l998_99893


namespace NUMINAMATH_CALUDE_triangle_side_length_l998_99809

theorem triangle_side_length (A B : Real) (b : Real) (hA : A = 60 * π / 180) (hB : B = 45 * π / 180) (hb : b = Real.sqrt 2) :
  ∃ a : Real, a = Real.sqrt 3 ∧ a * Real.sin B = b * Real.sin A := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l998_99809


namespace NUMINAMATH_CALUDE_probability_at_least_six_heads_l998_99831

/-- A sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- The total number of possible outcomes for 8 coin flips -/
def totalOutcomes : ℕ := 256

/-- Checks if a sequence has at least 6 consecutive heads -/
def hasAtLeastSixConsecutiveHeads (s : CoinFlipSequence) : Prop :=
  ∃ i, i + 5 < 8 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

/-- The number of sequences with at least 6 consecutive heads -/
def favorableOutcomes : ℕ := 13

/-- The probability of getting at least 6 consecutive heads in 8 fair coin flips -/
def probabilityAtLeastSixHeads : ℚ := favorableOutcomes / totalOutcomes

theorem probability_at_least_six_heads :
  probabilityAtLeastSixHeads = 13 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_six_heads_l998_99831


namespace NUMINAMATH_CALUDE_password_probability_l998_99806

def password_space : ℕ := 10 * 52 * 52 * 10

def even_start_space : ℕ := 5 * 52 * 52 * 10

def diff_letters_space : ℕ := 10 * 52 * 51 * 10

def non_zero_end_space : ℕ := 10 * 52 * 52 * 9

def valid_password_space : ℕ := 5 * 52 * 51 * 9

theorem password_probability :
  (valid_password_space : ℚ) / password_space = 459 / 1040 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l998_99806


namespace NUMINAMATH_CALUDE_inequality_proof_l998_99845

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0)
  (h4 : x ≥ y) (h5 : y ≥ z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((a*x + c*z) * (a*z + c*x)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l998_99845


namespace NUMINAMATH_CALUDE_inequality_proof_l998_99837

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 + 1 - 2*x*(x*y^2 - x + z + 1) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l998_99837


namespace NUMINAMATH_CALUDE_farmer_additional_earnings_l998_99895

/-- Represents the farmer's market transactions and wheelbarrow sale --/
def farmer_earnings (duck_price chicken_price : ℕ) (ducks_sold chickens_sold : ℕ) : ℕ :=
  let total_earnings := duck_price * ducks_sold + chicken_price * chickens_sold
  let wheelbarrow_cost := total_earnings / 2
  let wheelbarrow_sale_price := wheelbarrow_cost * 2
  wheelbarrow_sale_price - wheelbarrow_cost

/-- Proves that the farmer's additional earnings from selling the wheelbarrow is $30 --/
theorem farmer_additional_earnings :
  farmer_earnings 10 8 2 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_additional_earnings_l998_99895


namespace NUMINAMATH_CALUDE_power_inequality_l998_99833

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^5 + y^5 - (x^4*y + x*y^4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l998_99833


namespace NUMINAMATH_CALUDE_library_books_remaining_l998_99842

theorem library_books_remaining (initial_books : ℕ) 
  (day1_borrowers : ℕ) (books_per_borrower : ℕ) (day2_borrowed : ℕ) : 
  initial_books = 100 →
  day1_borrowers = 5 →
  books_per_borrower = 2 →
  day2_borrowed = 20 →
  initial_books - (day1_borrowers * books_per_borrower + day2_borrowed) = 70 :=
by sorry

end NUMINAMATH_CALUDE_library_books_remaining_l998_99842


namespace NUMINAMATH_CALUDE_star_37_25_l998_99894

-- Define the star operation
def star (x y : ℝ) : ℝ := x * y + 3

-- State the theorem
theorem star_37_25 :
  (∀ (x : ℝ), x > 0 → star (star x 1) x = star x (star 1 x)) →
  star 1 1 = 4 →
  star 37 25 = 928 := by sorry

end NUMINAMATH_CALUDE_star_37_25_l998_99894


namespace NUMINAMATH_CALUDE_equiangular_hexagon_side_lengths_l998_99876

/-- An equiangular hexagon is a hexagon where all internal angles are equal. -/
structure EquiangularHexagon where
  sides : Fin 6 → ℝ
  is_equiangular : True  -- This is a placeholder for the equiangular property

/-- The theorem stating the side lengths of a specific equiangular hexagon -/
theorem equiangular_hexagon_side_lengths 
  (h : EquiangularHexagon) 
  (h1 : h.sides 0 = 3)
  (h2 : h.sides 2 = 5)
  (h3 : h.sides 3 = 4)
  (h4 : h.sides 4 = 1) :
  h.sides 5 = 6 ∧ h.sides 1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_equiangular_hexagon_side_lengths_l998_99876


namespace NUMINAMATH_CALUDE_stating_triangle_division_theorem_l998_99872

/-- 
Represents the number of parts a triangle is divided into when each vertex
is connected to n points on the opposite side, assuming no three lines intersect
at the same point.
-/
def triangle_division (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- 
Theorem stating that when each vertex of a triangle is connected by straight lines
to n points on the opposite side, and no three lines intersect at the same point,
the triangle is divided into 3n^2 + 3n + 1 parts.
-/
theorem triangle_division_theorem (n : ℕ) :
  triangle_division n = 3 * n^2 + 3 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_stating_triangle_division_theorem_l998_99872


namespace NUMINAMATH_CALUDE_find_c_l998_99819

theorem find_c (m c : ℕ) : 
  m < 10 → c < 10 → m = 2 * c → 
  (10 * m + c : ℚ) / 99 = (c + 4 : ℚ) / (m + 5) → 
  c = 3 :=
sorry

end NUMINAMATH_CALUDE_find_c_l998_99819


namespace NUMINAMATH_CALUDE_sqrt_inequality_l998_99865

theorem sqrt_inequality (x : ℝ) :
  3 - x ≥ 0 → x + 1 ≥ 0 →
  (Real.sqrt (3 - x) - Real.sqrt (x + 1) > 1 / 2 ↔ -1 ≤ x ∧ x < 1 - Real.sqrt 31 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l998_99865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l998_99804

theorem arithmetic_sequence_problem (a₁ a₂ a₃ : ℚ) (x : ℚ) 
  (h1 : a₁ = 1/3)
  (h2 : a₂ = 2*x)
  (h3 : a₃ = x + 4)
  (h_arithmetic : a₃ - a₂ = a₂ - a₁) :
  x = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l998_99804


namespace NUMINAMATH_CALUDE_add_fractions_l998_99857

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_l998_99857


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l998_99824

def is_valid_representation (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 9

def has_at_least_two_of_each (n : ℕ) : Prop :=
  (n.digits 10).count 4 ≥ 2 ∧ (n.digits 10).count 9 ≥ 2

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 4 = 0 ∧
    m % 9 = 0 ∧
    is_valid_representation m ∧
    has_at_least_two_of_each m ∧
    (∀ k : ℕ, k > 0 ∧ k % 4 = 0 ∧ k % 9 = 0 ∧ is_valid_representation k ∧ has_at_least_two_of_each k → m ≤ k) ∧
    last_four_digits m = 9494 :=
  by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l998_99824


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l998_99854

theorem quadratic_equal_roots (m n : ℝ) : 
  (m = 2 ∧ n = 1) → 
  ∃ x : ℝ, x^2 - m*x + n = 0 ∧ 
  ∀ y : ℝ, y^2 - m*y + n = 0 → y = x :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l998_99854


namespace NUMINAMATH_CALUDE_problem_stack_surface_area_l998_99849

/-- Represents a solid formed by stacking unit cubes -/
structure CubeStack where
  base_length : ℕ
  base_width : ℕ
  base_height : ℕ
  top_cube : Bool

/-- Calculates the surface area of a CubeStack -/
def surface_area (stack : CubeStack) : ℕ :=
  sorry

/-- The specific cube stack described in the problem -/
def problem_stack : CubeStack :=
  { base_length := 3
  , base_width := 3
  , base_height := 1
  , top_cube := true }

/-- Theorem stating that the surface area of the problem_stack is 34 square units -/
theorem problem_stack_surface_area :
  surface_area problem_stack = 34 :=
sorry

end NUMINAMATH_CALUDE_problem_stack_surface_area_l998_99849


namespace NUMINAMATH_CALUDE_wax_already_possessed_l998_99813

/-- Given the total amount of wax needed and the additional amount required,
    calculate the amount of wax already possessed. -/
theorem wax_already_possessed
  (total_wax : ℕ)
  (additional_wax : ℕ)
  (h1 : total_wax = 288)
  (h2 : additional_wax = 260)
  : total_wax - additional_wax = 28 :=
by sorry

end NUMINAMATH_CALUDE_wax_already_possessed_l998_99813


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l998_99823

/-- Given two right circular cylinders with radii r₁ and r₂, and heights h₁ and h₂,
    prove that if the volume of the second is twice the first and r₂ = 1.1 * r₁,
    then h₂ ≈ 1.65 * h₁ -/
theorem cylinder_height_relationship (r₁ r₂ h₁ h₂ : ℝ) 
  (volume_relation : π * r₂^2 * h₂ = 2 * π * r₁^2 * h₁)
  (radius_relation : r₂ = 1.1 * r₁)
  (h₁_pos : h₁ > 0) (r₁_pos : r₁ > 0) :
  ∃ ε > 0, abs (h₂ / h₁ - 200 / 121) < ε :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l998_99823


namespace NUMINAMATH_CALUDE_total_buttons_for_order_l998_99846

/-- The number of shirts ordered for each type -/
def shirts_per_type : ℕ := 200

/-- The number of buttons on the first type of shirt -/
def buttons_type1 : ℕ := 3

/-- The number of buttons on the second type of shirt -/
def buttons_type2 : ℕ := 5

/-- Theorem: The total number of buttons needed for the order is 1600 -/
theorem total_buttons_for_order :
  shirts_per_type * buttons_type1 + shirts_per_type * buttons_type2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_for_order_l998_99846


namespace NUMINAMATH_CALUDE_line_through_points_2m_minus_b_l998_99811

/-- Given a line passing through points (1,3) and (4,15), prove that 2m - b = 9 where y = mx + b is the equation of the line. -/
theorem line_through_points_2m_minus_b (m b : ℝ) : 
  (3 : ℝ) = m * 1 + b → 
  (15 : ℝ) = m * 4 + b → 
  2 * m - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_2m_minus_b_l998_99811


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l998_99890

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l998_99890


namespace NUMINAMATH_CALUDE_roberto_outfits_l998_99863

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 6
  let shirts : ℕ := 7
  let jackets : ℕ := 4
  let shoes : ℕ := 2
  number_of_outfits trousers shirts jackets shoes = 336 := by
  sorry


end NUMINAMATH_CALUDE_roberto_outfits_l998_99863


namespace NUMINAMATH_CALUDE_fraction_equality_l998_99807

theorem fraction_equality : (4^3 : ℝ) / (10^2 - 6^2) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l998_99807


namespace NUMINAMATH_CALUDE_fish_tagging_problem_l998_99859

/-- The number of tagged fish in the second catch, given the conditions of the fish tagging problem. -/
def tagged_fish_in_second_catch (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ) : ℕ :=
  (initially_tagged * second_catch) / total_fish

/-- Theorem stating that the number of tagged fish in the second catch is 2 under the given conditions. -/
theorem fish_tagging_problem (total_fish : ℕ) (initially_tagged : ℕ) (second_catch : ℕ)
  (h_total : total_fish = 1750)
  (h_tagged : initially_tagged = 70)
  (h_catch : second_catch = 50) :
  tagged_fish_in_second_catch total_fish initially_tagged second_catch = 2 :=
by sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l998_99859


namespace NUMINAMATH_CALUDE_log7_10_approximation_l998_99815

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.499

-- Define the target approximation
def log7_10_approx : ℝ := 2

-- Theorem statement
theorem log7_10_approximation :
  abs (Real.log 10 / Real.log 7 - log7_10_approx) < 0.1 :=
sorry


end NUMINAMATH_CALUDE_log7_10_approximation_l998_99815


namespace NUMINAMATH_CALUDE_tonys_initial_money_is_87_l998_99835

/-- Calculates Tony's initial amount of money -/
def tonys_initial_money (cheese_cost beef_cost beef_amount cheese_amount money_left : ℕ) : ℕ :=
  cheese_cost * cheese_amount + beef_cost * beef_amount + money_left

/-- Proves that Tony's initial amount of money was $87 -/
theorem tonys_initial_money_is_87 :
  tonys_initial_money 7 5 1 3 61 = 87 := by
  sorry

end NUMINAMATH_CALUDE_tonys_initial_money_is_87_l998_99835


namespace NUMINAMATH_CALUDE_sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l998_99888

theorem sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2 :
  Real.sqrt 50 + 2 / (Real.sqrt 2 + 1) - 4 * Real.sqrt (1/2) + 2 * (Real.sqrt 2 - 1)^0 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l998_99888


namespace NUMINAMATH_CALUDE_incorrect_statement_is_E_l998_99820

theorem incorrect_statement_is_E :
  -- Statement A
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a + c > b + c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a * c > b * c)) ∧
  (∀ (a b c : ℝ), c > 0 → (a > b ↔ a / c > b / c)) ∧
  -- Statement B
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (a + b) / 2 > Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ (s : ℝ), s > 0 → ∃ (x : ℝ), x > 0 ∧ x < s ∧
    ∀ (y : ℝ), y > 0 → y < s → x * (s - x) ≥ y * (s - y)) ∧
  -- Statement D
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b →
    (a^2 + b^2) / 2 > ((a + b) / 2)^2) ∧
  -- Statement E (negation)
  (∃ (p : ℝ), p > 0 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = p ∧ x + y > 2 * Real.sqrt p) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_is_E_l998_99820


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_f_at_specific_angle_l998_99832

-- Problem 1
theorem trigonometric_expression_equals_one :
  Real.sin (-120 * Real.pi / 180) * Real.cos (210 * Real.pi / 180) +
  Real.cos (-300 * Real.pi / 180) * Real.sin (-330 * Real.pi / 180) = 1 := by
sorry

-- Problem 2
noncomputable def f (α : Real) : Real :=
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.cos ((3 * Real.pi) / 2 + α) - Real.sin ((Real.pi / 2 + α) ^ 2))

theorem f_at_specific_angle (h : 1 + 2 * Real.sin (-23 * Real.pi / 6) ≠ 0) :
  f (-23 * Real.pi / 6) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_f_at_specific_angle_l998_99832


namespace NUMINAMATH_CALUDE_text_messages_difference_l998_99850

theorem text_messages_difference (last_week : ℕ) (total : ℕ) : last_week = 111 → total = 283 → total - last_week - last_week = 61 := by
  sorry

end NUMINAMATH_CALUDE_text_messages_difference_l998_99850
