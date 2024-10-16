import Mathlib

namespace NUMINAMATH_CALUDE_average_marks_combined_l48_4849

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 = 20 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 60 →
  let total_marks := n1 * avg1 + n2 * avg2
  let total_students := n1 + n2
  abs ((total_marks / total_students) - 54.29) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l48_4849


namespace NUMINAMATH_CALUDE_equation_system_result_l48_4897

theorem equation_system_result : ∃ (x y z : ℝ), 
  z ≠ 0 ∧ 
  3*x - 4*y - 2*z = 0 ∧ 
  x + 4*y - 20*z = 0 ∧ 
  (x^2 + 4*x*y) / (y^2 + z^2) = 106496/36324 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_result_l48_4897


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l48_4898

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) :
  (a * x^2 + b * x + c = 0) →
  ¬(a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l48_4898


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l48_4826

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l48_4826


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l48_4841

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (4 * d^2 + 8 * d - 48 = 0) →
  (4 * e^2 + 8 * e - 48 = 0) →
  d ≠ e →
  (d - e)^2 = 49 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l48_4841


namespace NUMINAMATH_CALUDE_shelter_ratio_l48_4837

theorem shelter_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  num_cats = 45 →
  (num_cats : ℚ) / (num_dogs + 12 : ℚ) = 15 / 11 →
  (num_cats : ℚ) / (num_dogs : ℚ) = 15 / 7 :=
by sorry

end NUMINAMATH_CALUDE_shelter_ratio_l48_4837


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l48_4827

theorem shaded_area_rectangle (length width : ℝ) (h1 : length = 8) (h2 : width = 4) : 
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * length * width
  let shaded_area := rectangle_area - triangle_area
  shaded_area = 16 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l48_4827


namespace NUMINAMATH_CALUDE_circle_ellipse_intersection_l48_4838

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define the point D
def D : ℝ × ℝ := (1, 0)

-- Define the ellipse E (trajectory of P)
def E (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point C
def C_point : ℝ × ℝ := (-1, 0)

-- Define the perpendicular foot W
def W (x0 y0 : ℝ) : Prop := 
  ∃ (k1 k2 : ℝ), 
    (y0 = k1 * (x0 + 1)) ∧ 
    (y0 = k2 * (x0 - 1)) ∧ 
    (k1 * k2 = -1) ∧
    E x0 y0

-- Define the theorem
theorem circle_ellipse_intersection :
  ∀ (x0 y0 : ℝ), W x0 y0 →
    (x0^2 / 2 + y0^2 < 1) ∧
    (∃ (area : ℝ), area = 16/9 ∧ 
      ∀ (q r s t : ℝ × ℝ), 
        E q.1 q.2 → E r.1 r.2 → E s.1 s.2 → E t.1 t.2 →
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
        area ≤ (abs ((q.1 - s.1) * (r.2 - t.2) - (q.2 - s.2) * (r.1 - t.1))) / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_intersection_l48_4838


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l48_4879

def A : Set ℝ := {-1, 0, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l48_4879


namespace NUMINAMATH_CALUDE_fraction_simplification_l48_4871

theorem fraction_simplification (x y z : ℝ) (h : x + 2*y + z ≠ 0) :
  (x^2 + y^2 - 4*z^2 + 2*x*y) / (x^2 + 4*y^2 - z^2 + 2*x*z) = (x + y - 2*z) / (x + z - 2*y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l48_4871


namespace NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l48_4832

theorem reciprocal_equality_implies_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a = 1 / b → a = b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equality_implies_equality_l48_4832


namespace NUMINAMATH_CALUDE_open_box_volume_l48_4880

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ)
  (h_length : sheet_length = 52)
  (h_width : sheet_width = 36)
  (h_cut : cut_size = 8) :
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5760 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l48_4880


namespace NUMINAMATH_CALUDE_worker_y_defective_rate_l48_4877

-- Define the fractions and percentages
def worker_x_fraction : ℝ := 1 - 0.1666666666666668
def worker_y_fraction : ℝ := 0.1666666666666668
def worker_x_defective_rate : ℝ := 0.005
def total_defective_rate : ℝ := 0.0055

-- Theorem statement
theorem worker_y_defective_rate :
  ∃ (y_rate : ℝ),
    y_rate = 0.008 ∧
    total_defective_rate = worker_x_fraction * worker_x_defective_rate + worker_y_fraction * y_rate :=
by sorry

end NUMINAMATH_CALUDE_worker_y_defective_rate_l48_4877


namespace NUMINAMATH_CALUDE_S_is_circle_l48_4874

-- Define the set of complex numbers satisfying |z-3|=1
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3) = 1}

-- Theorem statement
theorem S_is_circle : 
  ∃ (center : ℂ) (radius : ℝ), S = {z : ℂ | Complex.abs (z - center) = radius} ∧ radius > 0 :=
by sorry

end NUMINAMATH_CALUDE_S_is_circle_l48_4874


namespace NUMINAMATH_CALUDE_sum_six_equals_twentyfour_l48_4819

/-- An arithmetic sequence {a_n} with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.S n

theorem sum_six_equals_twentyfour (seq : ArithmeticSequence) 
  (h2 : sum_n seq 2 = 2) 
  (h4 : sum_n seq 4 = 10) : 
  sum_n seq 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_equals_twentyfour_l48_4819


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l48_4845

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l48_4845


namespace NUMINAMATH_CALUDE_inequality_proof_l48_4851

theorem inequality_proof (S a b c x y z : ℝ) 
  (hS : S > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a + x = S) (eq2 : b + y = S) (eq3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l48_4851


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l48_4840

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 + 2*x) :
  ∀ x < 0, f x = -x^2 + 2*x :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l48_4840


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l48_4823

theorem diophantine_equation_solution (x y z : ℤ) :
  2 * x^2 + 3 * y^2 = z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l48_4823


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l48_4803

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The measure of an angle in a regular octagon in degrees. -/
def regular_octagon_angle : ℝ := 135

/-- The measure of angle ABH in a regular octagon ABCDEFGH in degrees. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := 22.5

/-- Theorem: In a regular octagon ABCDEFGH, the measure of angle ABH is 22.5°. -/
theorem angle_ABH_measure (octagon : RegularOctagon) :
  angle_ABH octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABH_measure_l48_4803


namespace NUMINAMATH_CALUDE_two_digit_gcd_theorem_l48_4899

/-- Represents a two-digit decimal number as a pair of natural numbers (a, b) -/
def TwoDigitNumber := { p : ℕ × ℕ // p.1 ≤ 9 ∧ p.2 ≤ 9 ∧ p.1 ≠ 0 }

/-- Converts a two-digit number (a, b) to its decimal representation ab -/
def toDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.1 + n.val.2

/-- Converts a two-digit number (a, b) to its reversed decimal representation ba -/
def toReversedDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.2 + n.val.1

/-- Checks if a two-digit number satisfies the GCD condition -/
def satisfiesGCDCondition (n : TwoDigitNumber) : Prop :=
  Nat.gcd (toDecimal n) (toReversedDecimal n) = n.val.1^2 - n.val.2^2

theorem two_digit_gcd_theorem :
  ∃ (n1 n2 : TwoDigitNumber),
    satisfiesGCDCondition n1 ∧
    satisfiesGCDCondition n2 ∧
    toDecimal n1 = 21 ∧
    toDecimal n2 = 54 ∧
    (∀ (n : TwoDigitNumber), satisfiesGCDCondition n → (toDecimal n = 21 ∨ toDecimal n = 54)) :=
  sorry

end NUMINAMATH_CALUDE_two_digit_gcd_theorem_l48_4899


namespace NUMINAMATH_CALUDE_rectangle_ratio_l48_4842

/-- Given an arrangement of four congruent rectangles around an inner square,
    prove that the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) : 
  s > 0 →  -- inner square side length is positive
  x > 0 ∧ y > 0 →  -- rectangle sides are positive
  s + 2*y = 3*s →  -- outer square side length
  x + s = 3*s →  -- outer square side length (alternate direction)
  (3*s)^2 = 9*s^2 →  -- area relation
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l48_4842


namespace NUMINAMATH_CALUDE_count_lattice_points_l48_4872

/-- The number of lattice points on the graph of x^2 - y^2 = 36 -/
def lattice_points_count : ℕ := 8

/-- A predicate that checks if a pair of integers satisfies x^2 - y^2 = 36 -/
def satisfies_equation (x y : ℤ) : Prop := x^2 - y^2 = 36

/-- The theorem stating that there are exactly 8 lattice points on the graph of x^2 - y^2 = 36 -/
theorem count_lattice_points :
  (∃! (s : Finset (ℤ × ℤ)), s.card = lattice_points_count ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
by sorry

#check count_lattice_points

end NUMINAMATH_CALUDE_count_lattice_points_l48_4872


namespace NUMINAMATH_CALUDE_shane_photos_january_l48_4828

/-- Calculates the number of photos taken per day in January given the total number of photos
    in the first two months and the number of photos taken each week in February. -/
def photos_per_day_january (total_photos : ℕ) (photos_per_week_feb : ℕ) : ℕ :=
  let photos_feb := photos_per_week_feb * 4
  let photos_jan := total_photos - photos_feb
  photos_jan / 31

/-- Theorem stating that given 146 total photos in the first two months and 21 photos
    per week in February, Shane took 2 photos per day in January. -/
theorem shane_photos_january : photos_per_day_january 146 21 = 2 := by
  sorry

#eval photos_per_day_january 146 21

end NUMINAMATH_CALUDE_shane_photos_january_l48_4828


namespace NUMINAMATH_CALUDE_total_flowers_l48_4868

theorem total_flowers (roses : ℕ) (lilies : ℕ) (tulips : ℕ) : 
  roses = 34 →
  lilies = roses + 13 →
  tulips = lilies - 23 →
  roses + lilies + tulips = 105 := by
sorry

end NUMINAMATH_CALUDE_total_flowers_l48_4868


namespace NUMINAMATH_CALUDE_fourth_drawn_is_92_l48_4820

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (groupNumber : ℕ) : ℕ :=
  firstDrawn + (groupNumber - 1) * (populationSize / sampleSize)

/-- Theorem: The fourth drawn number in the given systematic sampling scenario is 92 -/
theorem fourth_drawn_is_92 :
  systematicSample 600 20 2 4 = 92 := by
  sorry

#eval systematicSample 600 20 2 4

end NUMINAMATH_CALUDE_fourth_drawn_is_92_l48_4820


namespace NUMINAMATH_CALUDE_base_conversion_problem_l48_4885

theorem base_conversion_problem (b : ℕ+) : 
  (b : ℝ)^5 ≤ 125 ∧ 125 < (b : ℝ)^6 ↔ b = 2 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l48_4885


namespace NUMINAMATH_CALUDE_ellipse_inscribed_parallelogram_slope_product_l48_4878

/-- Given an ellipse Γ: x²/3 + y²/2 = 1, with a parallelogram ABCD inscribed in it
    such that BD is a diagonal and B and D are symmetric about the origin,
    the product of the slopes of adjacent sides AB and BC is equal to -2/3. -/
theorem ellipse_inscribed_parallelogram_slope_product
  (Γ : Set (ℝ × ℝ))
  (h_ellipse : Γ = {(x, y) | x^2/3 + y^2/2 = 1})
  (A B C D : ℝ × ℝ)
  (h_inscribed : A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ∧ D ∈ Γ)
  (h_parallelogram : (A.1 + C.1 = B.1 + D.1) ∧ (A.2 + C.2 = B.2 + D.2))
  (h_diagonal : B.1 + D.1 = 0 ∧ B.2 + D.2 = 0)
  (k₁ k₂ : ℝ)
  (h_slope_AB : k₁ = (B.2 - A.2) / (B.1 - A.1))
  (h_slope_BC : k₂ = (C.2 - B.2) / (C.1 - B.1)) :
  k₁ * k₂ = -2/3 := by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_parallelogram_slope_product_l48_4878


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l48_4864

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 410 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 44 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l48_4864


namespace NUMINAMATH_CALUDE_remainder_theorem_l48_4813

-- Define the polynomial P
variable (P : ℝ → ℝ)

-- Define the conditions
axiom P_19 : P 19 = 99
axiom P_99 : P 99 = 19

-- Define the remainder function
def remainder (P : ℝ → ℝ) : ℝ → ℝ :=
  λ x => -x + 118

-- Theorem statement
theorem remainder_theorem :
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = (x - 19) * (x - 99) * Q x + remainder P x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l48_4813


namespace NUMINAMATH_CALUDE_alan_pine_trees_l48_4892

/-- The number of pine cones dropped by each tree -/
def pine_cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def pine_cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def total_roof_weight : ℕ := 1920

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

theorem alan_pine_trees :
  num_trees * (pine_cones_per_tree * roof_percentage).floor * pine_cone_weight = total_roof_weight :=
sorry

end NUMINAMATH_CALUDE_alan_pine_trees_l48_4892


namespace NUMINAMATH_CALUDE_points_collinear_collinear_vectors_l48_4858

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Non-zero vectors e₁ and e₂ are not collinear -/
def not_collinear (e₁ e₂ : V) : Prop :=
  e₁ ≠ 0 ∧ e₂ ≠ 0 ∧ ∀ (r : ℝ), e₁ ≠ r • e₂

variable (e₁ e₂ : V) (h : not_collinear e₁ e₂)

/-- Vector AB -/
def AB : V := e₁ + e₂

/-- Vector BC -/
def BC : V := 2 • e₁ + 8 • e₂

/-- Vector CD -/
def CD : V := 3 • (e₁ - e₂)

/-- Three points are collinear if the vector between any two is a scalar multiple of the vector between the other two -/
def collinear (A B D : V) : Prop :=
  ∃ (r : ℝ), B - A = r • (D - B) ∨ D - B = r • (B - A)

theorem points_collinear :
  collinear (0 : V) (AB e₁ e₂) ((AB e₁ e₂) + (BC e₁ e₂) + (CD e₁ e₂)) :=
sorry

theorem collinear_vectors (k : ℝ) :
  (∃ (r : ℝ), k • e₁ + e₂ = r • (e₁ + k • e₂)) ↔ k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_points_collinear_collinear_vectors_l48_4858


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l48_4830

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_two_digits (n : ℕ) : ℕ :=
  n / 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def is_permutation (a b : ℕ) : Prop :=
  a / 10 = b % 10 ∧ a % 10 = b / 10

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

theorem four_digit_number_theorem :
  ∃! n : ℕ, 
    is_valid_four_digit_number n ∧
    is_permutation (first_two_digits n) (last_two_digits n) ∧
    first_two_digits n - last_two_digits n = sum_of_digits (first_two_digits n) ∧
    n = 5445 :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l48_4830


namespace NUMINAMATH_CALUDE_polynomial_equality_l48_4853

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_equality (a b c d : ℝ) :
  (Q a b c d (-1) = 2) →
  (∀ x, Q a b c d x = 2 + x^2 - x^3) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l48_4853


namespace NUMINAMATH_CALUDE_triangle_measure_l48_4829

/-- Given an equilateral triangle with side length 7.5 meters, 
    prove that three times the square of the side length is 168.75 meters. -/
theorem triangle_measure (side_length : ℝ) : 
  side_length = 7.5 → 3 * (side_length ^ 2) = 168.75 := by
  sorry

#check triangle_measure

end NUMINAMATH_CALUDE_triangle_measure_l48_4829


namespace NUMINAMATH_CALUDE_production_days_calculation_l48_4894

/-- Given the average daily production for n days and the production on an additional day,
    calculate the number of days n. -/
theorem production_days_calculation (n : ℕ) : 
  (n * 70 + 90) / (n + 1) = 75 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l48_4894


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l48_4808

/-- Given a hyperbola with equation y²/4 - x²/9 = 1, its asymptotes are 2x±3y=0 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  y^2/4 - x^2/9 = 1 →
  (∃ (k : ℝ), k = 2 ∨ k = -2) →
  2*x + k*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l48_4808


namespace NUMINAMATH_CALUDE_lianliang_run_distance_l48_4814

/-- The length of the playground in meters -/
def playground_length : ℕ := 110

/-- The difference between the length and width of the playground in meters -/
def length_width_difference : ℕ := 15

/-- The width of the playground in meters -/
def playground_width : ℕ := playground_length - length_width_difference

/-- The perimeter of the playground in meters -/
def playground_perimeter : ℕ := (playground_length + playground_width) * 2

theorem lianliang_run_distance : playground_perimeter = 230 := by
  sorry

end NUMINAMATH_CALUDE_lianliang_run_distance_l48_4814


namespace NUMINAMATH_CALUDE_A_value_l48_4839

/-- Rounds a natural number down to the nearest tens -/
def round_down_to_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

/-- Given a natural number n = A567 where A is unknown, 
    if n rounds down to 2560, then A = 2 -/
theorem A_value (n : ℕ) (h : round_down_to_tens n = 2560) : 
  n / 1000 = 2 := by sorry

end NUMINAMATH_CALUDE_A_value_l48_4839


namespace NUMINAMATH_CALUDE_circle_area_ratio_l48_4811

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) : 
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) → 
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l48_4811


namespace NUMINAMATH_CALUDE_stratified_sample_size_l48_4843

/-- Represents the quantity of each product model in a sample -/
structure ProductSample where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the ratio of quantities for products A, B, and C -/
def quantity_ratio : ProductSample := ⟨2, 3, 5⟩

/-- Calculates the total sample size -/
def total_sample_size (s : ProductSample) : ℕ := s.a + s.b + s.c

/-- Theorem: If a stratified sample with the given ratio contains 16 units of model A, 
    then the total sample size is 80 -/
theorem stratified_sample_size 
  (sample : ProductSample)
  (h_ratio : ∃ k : ℕ, sample.a = k * quantity_ratio.a ∧ 
                      sample.b = k * quantity_ratio.b ∧ 
                      sample.c = k * quantity_ratio.c)
  (h_model_a : sample.a = 16) :
  total_sample_size sample = 80 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l48_4843


namespace NUMINAMATH_CALUDE_ice_bag_cost_calculation_l48_4896

def small_bag_cost : ℚ := 80 / 100
def large_bag_cost : ℚ := 146 / 100
def total_bags : ℕ := 30
def small_bags : ℕ := 18
def discount_rate : ℚ := 12 / 100

theorem ice_bag_cost_calculation :
  let large_bags : ℕ := total_bags - small_bags
  let total_cost_before_discount : ℚ := small_bag_cost * small_bags + large_bag_cost * large_bags
  let discount_amount : ℚ := total_cost_before_discount * discount_rate
  let total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount
  ∃ (rounded_cost : ℚ), (rounded_cost * 100).floor = 2809 ∧ 
    |rounded_cost - total_cost_after_discount| ≤ 1 / 200 :=
by sorry

end NUMINAMATH_CALUDE_ice_bag_cost_calculation_l48_4896


namespace NUMINAMATH_CALUDE_inequality_proof_l48_4863

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  b / a < (c - b) / (c - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l48_4863


namespace NUMINAMATH_CALUDE_president_savings_theorem_l48_4860

/-- The amount saved by the president for his reelection campaign --/
def president_savings (total_funds friends_percentage family_percentage : ℝ) : ℝ :=
  let friends_contribution := friends_percentage * total_funds
  let remaining_after_friends := total_funds - friends_contribution
  let family_contribution := family_percentage * remaining_after_friends
  total_funds - friends_contribution - family_contribution

/-- Theorem stating the amount saved by the president given the campaign fund conditions --/
theorem president_savings_theorem :
  president_savings 10000 0.4 0.3 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_president_savings_theorem_l48_4860


namespace NUMINAMATH_CALUDE_parallel_line_angle_theorem_l48_4824

-- Define the structure for our geometric configuration
structure ParallelLineConfig where
  -- Angle QTV
  angle_QTV : ℝ
  -- Angle SUV
  angle_SUV : ℝ
  -- Angle TVU
  angle_TVU : ℝ
  -- Assumption that PQ and RS are parallel
  parallel_PQ_RS : True
  -- Assumptions about the given angles
  h_QTV : angle_QTV = 30
  h_SUV : angle_SUV = 40

-- Theorem statement
theorem parallel_line_angle_theorem (config : ParallelLineConfig) :
  config.angle_TVU = 70 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_angle_theorem_l48_4824


namespace NUMINAMATH_CALUDE_xy_value_l48_4866

theorem xy_value (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l48_4866


namespace NUMINAMATH_CALUDE_stair_climbing_problem_l48_4818

/-- Calculates the number of steps climbed given the number of flights, height per flight, and step height. -/
def steps_climbed (num_flights : ℕ) (height_per_flight : ℚ) (step_height_inches : ℚ) : ℚ :=
  (num_flights * height_per_flight) / (step_height_inches / 12)

/-- Proves that climbing 9 flights of 10 feet each, with steps of 18 inches, results in 60 steps. -/
theorem stair_climbing_problem :
  steps_climbed 9 10 18 = 60 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_problem_l48_4818


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_45_l48_4856

theorem smallest_divisible_by_18_and_45 : ∃ n : ℕ+, (∀ m : ℕ+, 18 ∣ m ∧ 45 ∣ m → n ≤ m) ∧ 18 ∣ n ∧ 45 ∣ n :=
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_45_l48_4856


namespace NUMINAMATH_CALUDE_binomial_sum_unique_l48_4862

theorem binomial_sum_unique (m : ℤ) : 
  (Nat.choose 25 m.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ m = 13 :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_unique_l48_4862


namespace NUMINAMATH_CALUDE_jackie_free_time_l48_4855

theorem jackie_free_time (total_hours work_hours exercise_hours sleep_hours : ℕ)
  (h1 : total_hours = 24)
  (h2 : work_hours = 8)
  (h3 : exercise_hours = 3)
  (h4 : sleep_hours = 8) :
  total_hours - (work_hours + exercise_hours + sleep_hours) = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackie_free_time_l48_4855


namespace NUMINAMATH_CALUDE_survivor_probability_l48_4848

/-- The number of contestants in the game -/
def total_contestants : ℕ := 18

/-- The number of tribes in the game -/
def number_of_tribes : ℕ := 3

/-- The number of contestants in each tribe -/
def contestants_per_tribe : ℕ := 6

/-- The number of contestants who quit the game -/
def quitters : ℕ := 2

/-- Probability that both quitters are from the same tribe -/
def prob_same_tribe : ℚ := 5/17

theorem survivor_probability :
  (total_contestants = number_of_tribes * contestants_per_tribe) →
  (total_contestants ≥ quitters) →
  (prob_same_tribe = (number_of_tribes * (contestants_per_tribe.choose quitters)) / 
                     (total_contestants.choose quitters)) :=
by sorry

end NUMINAMATH_CALUDE_survivor_probability_l48_4848


namespace NUMINAMATH_CALUDE_workshop_equation_system_l48_4883

/-- Represents the workshop scenario with workers producing bolts and nuts. -/
structure Workshop where
  totalWorkers : ℕ
  boltProduction : ℕ
  nutProduction : ℕ
  boltToNutRatio : ℚ

/-- Checks if a given system of equations correctly represents the workshop scenario. -/
def isCorrectSystem (w : Workshop) (x y : ℕ) : Prop :=
  (x + y = w.totalWorkers) ∧
  (2 * w.boltProduction * x = w.nutProduction * y)

/-- Theorem stating that the given system of equations correctly represents the workshop scenario. -/
theorem workshop_equation_system (w : Workshop) 
    (h1 : w.totalWorkers = 56)
    (h2 : w.boltProduction = 16)
    (h3 : w.nutProduction = 24)
    (h4 : w.boltToNutRatio = 1/2)
    (x y : ℕ) :
    isCorrectSystem w x y ↔ 
    (x + y = 56 ∧ 2 * 16 * x = 24 * y) :=
  sorry

end NUMINAMATH_CALUDE_workshop_equation_system_l48_4883


namespace NUMINAMATH_CALUDE_thomas_monthly_earnings_l48_4895

/-- Thomas's weekly earnings in the factory -/
def weekly_earnings : ℕ := 4550

/-- Number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Calculates monthly earnings based on weekly earnings and number of weeks in a month -/
def monthly_earnings (w : ℕ) (n : ℕ) : ℕ := w * n

/-- Theorem stating that Thomas's monthly earnings are 18200 -/
theorem thomas_monthly_earnings :
  monthly_earnings weekly_earnings weeks_in_month = 18200 := by
  sorry

end NUMINAMATH_CALUDE_thomas_monthly_earnings_l48_4895


namespace NUMINAMATH_CALUDE_expand_product_l48_4889

theorem expand_product (y : ℝ) : 3 * (y - 4) * (y + 9) = 3 * y^2 + 15 * y - 108 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l48_4889


namespace NUMINAMATH_CALUDE_sum_of_monomials_is_monomial_l48_4890

/-- 
Given two monomials 2x^3y^n and -6x^(m+5)y, if their sum is still a monomial,
then m + n = -1.
-/
theorem sum_of_monomials_is_monomial (m n : ℤ) : 
  (∃ (x y : ℝ), ∀ (a b : ℝ), 2 * (x^3) * (y^n) + (-6) * (x^(m+5)) * y = a * (x^b) * y) → 
  m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_monomials_is_monomial_l48_4890


namespace NUMINAMATH_CALUDE_correct_weight_proof_l48_4833

/-- Proves that the correct weight is 65 kg given the problem conditions -/
theorem correct_weight_proof (n : ℕ) (initial_avg : ℝ) (misread_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ initial_avg = 58.4 ∧ misread_weight = 56 ∧ correct_avg = 58.85 →
  ∃ (correct_weight : ℝ),
    correct_weight = 65 ∧
    n * correct_avg = (n * initial_avg - misread_weight + correct_weight) :=
by sorry


end NUMINAMATH_CALUDE_correct_weight_proof_l48_4833


namespace NUMINAMATH_CALUDE_marges_garden_plants_l48_4817

/-- Calculates the final number of plants in Marge's garden --/
def final_plant_count (total_seeds sunflower_seeds marigold_seeds seeds_not_grown : ℕ)
  (marigold_growth_rate sunflower_growth_rate : ℚ)
  (sunflower_wilt_rate marigold_eaten_rate pest_control_rate : ℚ)
  (weed_strangle_rate : ℚ) (weeds_pulled weeds_kept : ℕ) : ℕ :=
  sorry

/-- The theorem stating the final number of plants in Marge's garden --/
theorem marges_garden_plants :
  final_plant_count 23 13 10 5
    (4/10) (6/10) (1/4) (1/2) (3/4)
    (1/3) 2 1 = 6 :=
  sorry

end NUMINAMATH_CALUDE_marges_garden_plants_l48_4817


namespace NUMINAMATH_CALUDE_decagon_vertex_sum_l48_4886

theorem decagon_vertex_sum (π : Fin 10 → Fin 10) 
  (hπ : Function.Bijective π) :
  ∃ k : Fin 10, 
    π k + π ((k + 9) % 10) + π ((k + 1) % 10) ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_decagon_vertex_sum_l48_4886


namespace NUMINAMATH_CALUDE_jerseys_sold_l48_4846

def jersey_profit : ℕ := 165
def total_jersey_sales : ℕ := 25740

theorem jerseys_sold : 
  (total_jersey_sales / jersey_profit : ℕ) = 156 :=
by sorry

end NUMINAMATH_CALUDE_jerseys_sold_l48_4846


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l48_4821

/-- Represents the number of teachers --/
def num_teachers : ℕ := 4

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the condition that each school must receive at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- Calculates the number of ways to allocate teachers to schools --/
def allocation_schemes (n_teachers : ℕ) (n_schools : ℕ) (min_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 36 --/
theorem allocation_schemes_eq_36 : 
  allocation_schemes num_teachers num_schools min_teachers_per_school = 36 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l48_4821


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l48_4888

/-- A continuous monotonous function satisfying the given inequality -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  Monotone f ∧
  f 0 = 1 ∧
  ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1

/-- The main theorem stating that any function satisfying the conditions must be f(x) = x + 1 -/
theorem unique_satisfying_function (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l48_4888


namespace NUMINAMATH_CALUDE_Q_value_at_negative_one_l48_4815

/-- The cubic polynomial P(x) -/
def P (x : ℝ) : ℝ := x^3 + 8*x^2 - x + 3

/-- The roots of P(x) -/
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

/-- Q is a monic polynomial with roots ab - c^2, ac - b^2, bc - a^2 -/
def Q (x : ℝ) : ℝ := x^3 + 67*x^2 + 67*x + 1537

theorem Q_value_at_negative_one :
  P a = 0 → P b = 0 → P c = 0 → Q (-1) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_Q_value_at_negative_one_l48_4815


namespace NUMINAMATH_CALUDE_exam_day_percentage_l48_4802

/-- Proves that 70% of students took the exam on the assigned day given the conditions of the problem -/
theorem exam_day_percentage :
  ∀ (x : ℝ),
  (x ≥ 0) →
  (x ≤ 100) →
  (0.6 * x + 0.9 * (100 - x) = 69) →
  x = 70 := by
  sorry

end NUMINAMATH_CALUDE_exam_day_percentage_l48_4802


namespace NUMINAMATH_CALUDE_eighth_grade_students_l48_4869

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls → 
  boys = total - girls → 
  2 * girls - boys = 16 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l48_4869


namespace NUMINAMATH_CALUDE_tax_discount_order_invariance_l48_4887

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_order_invariance 
  (original_price tax_rate discount_rate : ℝ) 
  (h_tax : 0 ≤ tax_rate) (h_discount : 0 ≤ discount_rate) (h_price : 0 < original_price) :
  original_price * (1 + tax_rate) * (1 - discount_rate) = 
  original_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end NUMINAMATH_CALUDE_tax_discount_order_invariance_l48_4887


namespace NUMINAMATH_CALUDE_mason_savings_l48_4875

theorem mason_savings (savings : ℝ) (total_books : ℕ) (book_price : ℝ) 
  (h1 : savings > 0) 
  (h2 : total_books > 0) 
  (h3 : book_price > 0) 
  (h4 : (1/4) * savings = (2/5) * total_books * book_price) : 
  savings - total_books * book_price = (3/8) * savings := by
sorry

end NUMINAMATH_CALUDE_mason_savings_l48_4875


namespace NUMINAMATH_CALUDE_daily_step_goal_l48_4854

def sunday_steps : ℕ := 9400
def monday_steps : ℕ := 9100
def tuesday_steps : ℕ := 8300
def wednesday_steps : ℕ := 9200
def thursday_steps : ℕ := 8900
def friday_saturday_avg : ℕ := 9050
def days_in_week : ℕ := 7

theorem daily_step_goal :
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + thursday_steps + 
   2 * friday_saturday_avg) / days_in_week = 9000 := by
  sorry

end NUMINAMATH_CALUDE_daily_step_goal_l48_4854


namespace NUMINAMATH_CALUDE_customers_who_tipped_l48_4852

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 29 →
  additional_customers = 20 →
  non_tipping_customers = 34 →
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l48_4852


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l48_4876

theorem negative_integer_equation_solution :
  ∃ (N : ℤ), N < 0 ∧ 3 * N^2 + N = 15 → N = -3 :=
by sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l48_4876


namespace NUMINAMATH_CALUDE_find_number_l48_4891

theorem find_number : ∃ x : ℤ, x + 12 - 27 = 24 ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l48_4891


namespace NUMINAMATH_CALUDE_fish_tank_ratio_l48_4809

theorem fish_tank_ratio : 
  let first_tank := 7 + 8
  let second_tank := 2 * first_tank
  let third_tank := 10
  (third_tank : ℚ) / second_tank = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_tank_ratio_l48_4809


namespace NUMINAMATH_CALUDE_max_value_theorem_l48_4861

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
    3 * a' * b' * Real.sqrt 2 + 6 * b' * c' = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l48_4861


namespace NUMINAMATH_CALUDE_private_schools_in_district_a_l48_4881

theorem private_schools_in_district_a 
  (total_schools : ℕ) 
  (public_schools : ℕ) 
  (parochial_schools : ℕ) 
  (private_schools : ℕ) 
  (district_a_schools : ℕ) 
  (district_b_schools : ℕ) 
  (district_b_private_schools : ℕ) :
  total_schools = 50 →
  public_schools = 25 →
  parochial_schools = 16 →
  private_schools = 9 →
  district_a_schools = 18 →
  district_b_schools = 17 →
  district_b_private_schools = 2 →
  ∃ (district_c_schools : ℕ),
    district_c_schools = total_schools - district_a_schools - district_b_schools ∧
    district_c_schools % 3 = 0 →
    private_schools - district_b_private_schools - (district_c_schools / 3) = 2 :=
by sorry

end NUMINAMATH_CALUDE_private_schools_in_district_a_l48_4881


namespace NUMINAMATH_CALUDE_min_sqrt_equality_characterization_l48_4836

theorem min_sqrt_equality_characterization (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a * b + 1) / (a * b * c)))
       (min (Real.sqrt ((b * c + 1) / (a * b * c)))
            (Real.sqrt ((a * c + 1) / (a * b * c))))
   = Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c))
  ↔ ∃ r : ℝ, r > 0 ∧ 
    ((a = 1 / (1 + r^2) ∧ b = 1 / (1 + 1/r^2) ∧ c = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (b = 1 / (1 + r^2) ∧ c = 1 / (1 + 1/r^2) ∧ a = (r + 1/r)^2 / (1 + (r + 1/r)^2)) ∨
     (c = 1 / (1 + r^2) ∧ a = 1 / (1 + 1/r^2) ∧ b = (r + 1/r)^2 / (1 + (r + 1/r)^2))) :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_characterization_l48_4836


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l48_4847

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 4 = 0 ∧ x₂^2 + m*x₂ + 4 = 0) → 
  m ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l48_4847


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l48_4867

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l48_4867


namespace NUMINAMATH_CALUDE_compound_interest_repayment_l48_4857

-- Define the initial loan amount in yuan
def initial_loan : ℝ := 100000

-- Define the annual interest rate
def interest_rate : ℝ := 0.07

-- Define the repayment function (in ten thousand yuan)
def repayment_amount (years : ℕ) : ℝ :=
  10 * (1 + interest_rate) ^ years

-- Define the total repayment after 5 years (in yuan)
def total_repayment_5_years : ℕ := 140255

-- Define the number of installments
def num_installments : ℕ := 5

-- Define the annual installment amount (in yuan)
def annual_installment : ℕ := 24389

theorem compound_interest_repayment :
  -- 1. Repayment function
  (∀ x : ℕ, repayment_amount x = 10 * (1 + interest_rate) ^ x) ∧
  -- 2. Total repayment after 5 years
  (repayment_amount 5 * 10000 = total_repayment_5_years) ∧
  -- 3. Annual installment calculation
  (annual_installment * (((1 + interest_rate) ^ num_installments - 1) / interest_rate) =
    total_repayment_5_years) :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_repayment_l48_4857


namespace NUMINAMATH_CALUDE_alani_babysitting_earnings_l48_4882

/-- Given Alani's earnings for baby-sitting, calculate her earnings for a different duration -/
theorem alani_babysitting_earnings 
  (initial_earnings : ℝ) 
  (initial_hours : ℝ) 
  (new_hours : ℝ) 
  (h1 : initial_earnings = 45) 
  (h2 : initial_hours = 3) 
  (h3 : new_hours = 5) : 
  (initial_earnings / initial_hours) * new_hours = 75 := by
  sorry

#check alani_babysitting_earnings

end NUMINAMATH_CALUDE_alani_babysitting_earnings_l48_4882


namespace NUMINAMATH_CALUDE_g_of_5_l48_4807

def g (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 28*x^2 - 20*x - 80

theorem g_of_5 : g 5 = -5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l48_4807


namespace NUMINAMATH_CALUDE_johnnys_age_l48_4816

theorem johnnys_age : ∃ (age : ℕ), 
  (age + 2 = 2 * (age - 3)) ∧ age = 8 := by sorry

end NUMINAMATH_CALUDE_johnnys_age_l48_4816


namespace NUMINAMATH_CALUDE_max_value_of_f_l48_4810

/-- The quadratic function f(x) = -x^2 - 3x + 4 -/
def f (x : ℝ) : ℝ := -x^2 - 3*x + 4

/-- The maximum value of f(x) is 25/4 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 25/4 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l48_4810


namespace NUMINAMATH_CALUDE_f_of_x_minus_one_l48_4805

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_minus_one_l48_4805


namespace NUMINAMATH_CALUDE_total_fat_served_l48_4870

/-- The amount of fat in ounces for a single herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for a single eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a single pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type served -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem total_fat_served :
  total_fat = 3600 := by sorry

end NUMINAMATH_CALUDE_total_fat_served_l48_4870


namespace NUMINAMATH_CALUDE_rolling_circle_traces_line_l48_4859

/-- A circle with radius R -/
structure SmallCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = R

/-- A circle with radius 2R -/
structure LargeCircle (R : ℝ) where
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = 2 * R

/-- A point on the circumference of the small circle -/
def PointOnSmallCircle (R : ℝ) (sc : SmallCircle R) : Type :=
  { p : ℝ × ℝ // (p.1 - sc.center.1)^2 + (p.2 - sc.center.2)^2 = R^2 }

/-- The path traced by a point on the small circle as it rolls inside the large circle -/
def TracedPath (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) (p : PointOnSmallCircle R sc) : Set (ℝ × ℝ) :=
  sorry

/-- The statement that the traced path is a straight line -/
def IsStraitLine (path : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem rolling_circle_traces_line (R : ℝ) (sc : SmallCircle R) (lc : LargeCircle R) 
  (p : PointOnSmallCircle R sc) : 
  IsStraitLine (TracedPath R sc lc p) :=
sorry

end NUMINAMATH_CALUDE_rolling_circle_traces_line_l48_4859


namespace NUMINAMATH_CALUDE_treadmill_price_correct_l48_4825

/-- The price of the treadmill at Toby's garage sale. -/
def treadmill_price : ℝ := 133.33

/-- The total sum of money Toby made at the garage sale. -/
def total_money : ℝ := 600

/-- Theorem stating that the treadmill price is correct given the conditions of the garage sale. -/
theorem treadmill_price_correct : 
  treadmill_price + 0.5 * treadmill_price + 3 * treadmill_price = total_money :=
by sorry

end NUMINAMATH_CALUDE_treadmill_price_correct_l48_4825


namespace NUMINAMATH_CALUDE_congruence_solution_l48_4884

theorem congruence_solution (n : ℤ) : 11 * 21 ≡ 16 [ZMOD 43] := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l48_4884


namespace NUMINAMATH_CALUDE_sticker_pages_calculation_l48_4835

/-- Given a total number of stickers and the number of stickers per page,
    calculate the number of pages. -/
def calculate_pages (total_stickers : ℕ) (stickers_per_page : ℕ) : ℕ :=
  total_stickers / stickers_per_page

/-- Theorem stating that with 220 total stickers and 10 stickers per page,
    the number of pages is 22. -/
theorem sticker_pages_calculation :
  calculate_pages 220 10 = 22 := by
  sorry

end NUMINAMATH_CALUDE_sticker_pages_calculation_l48_4835


namespace NUMINAMATH_CALUDE_sum_powers_l48_4865

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  (∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_powers_l48_4865


namespace NUMINAMATH_CALUDE_train_length_calculation_l48_4873

/-- Calculates the length of a train given its speed, the speed of a person walking in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 67 →
  person_speed = 5 →
  passing_time = 6 →
  (train_speed + person_speed) * passing_time * (1000 / 3600) = 120 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l48_4873


namespace NUMINAMATH_CALUDE_jeans_pricing_markup_l48_4812

theorem jeans_pricing_markup (manufacturing_cost : ℝ) (customer_price : ℝ) (retailer_price : ℝ)
  (h1 : customer_price = manufacturing_cost * 1.54)
  (h2 : customer_price = retailer_price * 1.1) :
  (retailer_price - manufacturing_cost) / manufacturing_cost * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_markup_l48_4812


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l48_4893

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l48_4893


namespace NUMINAMATH_CALUDE_bookman_purchase_theorem_l48_4800

theorem bookman_purchase_theorem (hardback_price : ℕ) (paperback_price : ℕ) 
  (hardback_count : ℕ) (total_sold : ℕ) (remaining_value : ℕ) :
  hardback_price = 20 →
  paperback_price = 10 →
  hardback_count = 10 →
  total_sold = 14 →
  remaining_value = 360 →
  ∃ (total_copies : ℕ),
    total_copies = hardback_count + (remaining_value / paperback_price) + (total_sold - hardback_count) ∧
    total_copies = 50 :=
by sorry

end NUMINAMATH_CALUDE_bookman_purchase_theorem_l48_4800


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l48_4806

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l48_4806


namespace NUMINAMATH_CALUDE_greatest_x_4a_value_l48_4831

theorem greatest_x_4a_value : 
  ∀ (x a b c : ℕ), 
    (100 ≤ x) ∧ (x < 1000) →  -- x is a 3-digit integer
    (x = 100*a + 10*b + c) →  -- a, b, c are hundreds, tens, and units digits
    (4*a = 2*b) ∧ (2*b = c) → -- 4a = 2b = c
    (a > 0) →                 -- a > 0
    (∃ (x₁ x₂ : ℕ), (100 ≤ x₁) ∧ (x₁ < 1000) ∧ (100 ≤ x₂) ∧ (x₂ < 1000) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ x₁) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) ∧ (y ≠ x₁) → y ≤ x₂) ∧
      (x₁ - x₂ = 124)) →     -- difference between two greatest values is 124
    (∃ (a_max : ℕ), (100 ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧ 
      (100*a_max + 10*(2*a_max) + 4*a_max < 1000) ∧
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧
      (4*a_max = 8)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_4a_value_l48_4831


namespace NUMINAMATH_CALUDE_equilateral_triangle_circle_areas_l48_4822

theorem equilateral_triangle_circle_areas (s : ℝ) (h : s = 12) :
  let r := s / 2
  let sector_area := (π * r^2) / 3
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let shaded_area := 2 * (sector_area - triangle_area)
  ∃ (a b c : ℝ), shaded_area = a * π - b * Real.sqrt c ∧ a + b + c = 33 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_circle_areas_l48_4822


namespace NUMINAMATH_CALUDE_cubic_factorization_l48_4844

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l48_4844


namespace NUMINAMATH_CALUDE_largest_difference_l48_4834

-- Define the type for our table
def Table := Fin 20 → Fin 20 → Fin 400

-- Define a property that checks if a table is valid
def is_valid_table (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

-- Define the property we want to prove
theorem largest_difference (t : Table) (h : is_valid_table t) :
  (∃ i j k, (i = k ∨ j = k) ∧ 
    (t i j : ℕ).succ.pred - (t i k : ℕ).succ.pred ≥ 209) ∧
  ∀ m, m > 209 → 
    ∃ t', is_valid_table t' ∧ 
      ∀ i j k, (i = k ∨ j = k) → 
        (t' i j : ℕ).succ.pred - (t' i k : ℕ).succ.pred < m :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l48_4834


namespace NUMINAMATH_CALUDE_min_value_and_angle_l48_4801

theorem min_value_and_angle (A : Real) : 
  0 ≤ A ∧ A ≤ 2 * Real.pi →
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    2 * Real.sin (A / 2) + Real.sin A ≤ 2 * Real.sin (θ / 2) + Real.sin θ) →
  2 * Real.sin (A / 2) + Real.sin A = -4 ∧ A = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_angle_l48_4801


namespace NUMINAMATH_CALUDE_lottery_winnings_calculation_l48_4850

/-- Calculates the amount taken home from lottery winnings after tax and processing fee --/
def amountTakenHome (winnings : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  winnings - (winnings * taxRate) - processingFee

/-- Theorem stating that given specific lottery winnings, tax rate, and processing fee, 
    the amount taken home is $35 --/
theorem lottery_winnings_calculation :
  amountTakenHome 50 0.2 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winnings_calculation_l48_4850


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l48_4804

/-- The trajectory of the midpoint Q between a point P on the unit circle and a fixed point M -/
theorem midpoint_trajectory (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (Q.1 = (P.1 + 2) / 2 ∧ Q.2 = P.2 / 2) →  -- Q is the midpoint of PM where M is (2, 0)
  (Q.1 - 1)^2 + Q.2^2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l48_4804
