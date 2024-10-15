import Mathlib

namespace NUMINAMATH_CALUDE_fA_inter_fB_l460_46047

def f (n : ℕ+) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def fA : Set ℕ+ := {n : ℕ+ | f n ∈ A}
def fB : Set ℕ+ := {m : ℕ+ | f m ∈ B}

theorem fA_inter_fB : fA ∩ fB = {1, 2} := by sorry

end NUMINAMATH_CALUDE_fA_inter_fB_l460_46047


namespace NUMINAMATH_CALUDE_center_of_specific_circle_l460_46033

/-- The center coordinates of a circle given its equation -/
def circle_center (a b r : ℝ) : ℝ × ℝ := (a, -b)

/-- Theorem: The center coordinates of the circle (x-2)^2 + (y+1)^2 = 4 are (2, -1) -/
theorem center_of_specific_circle :
  circle_center 2 (-1) 2 = (2, -1) := by sorry

end NUMINAMATH_CALUDE_center_of_specific_circle_l460_46033


namespace NUMINAMATH_CALUDE_last_number_problem_l460_46018

theorem last_number_problem (a b c d : ℝ) 
  (h1 : (a + b + c) / 3 = 6)
  (h2 : (b + c + d) / 3 = 5)
  (h3 : a + d = 11) :
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_last_number_problem_l460_46018


namespace NUMINAMATH_CALUDE_executive_committee_selection_l460_46069

theorem executive_committee_selection (total_members : ℕ) (committee_size : ℕ) (ineligible_members : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  ineligible_members = 4 →
  Nat.choose (total_members - ineligible_members) committee_size = 60770 := by
sorry

end NUMINAMATH_CALUDE_executive_committee_selection_l460_46069


namespace NUMINAMATH_CALUDE_triangle_perimeter_l460_46044

theorem triangle_perimeter : ∀ x : ℝ, 
  (x - 2) * (x - 4) = 0 →
  x + 3 > 6 →
  x + 3 + 6 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l460_46044


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_square_l460_46094

/-- Given a square with side length 70√2 cm, divided into four congruent 45-45-90 triangles
    by its diagonals, the perimeter of one of these triangles is 140√2 + 140 cm. -/
theorem triangle_perimeter_in_square (side_length : ℝ) (h : side_length = 70 * Real.sqrt 2) :
  let diagonal := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal
  triangle_perimeter = 140 * Real.sqrt 2 + 140 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_square_l460_46094


namespace NUMINAMATH_CALUDE_min_product_of_tangent_line_to_unit_circle_l460_46001

theorem min_product_of_tangent_line_to_unit_circle (a b : ℝ) : 
  a > 0 → b > 0 → (∃ x y : ℝ, x^2 + y^2 = 1 ∧ x/a + y/b = 1) → a * b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_product_of_tangent_line_to_unit_circle_l460_46001


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l460_46055

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 3)
  (h_side1 : q.sides 0 = 100)
  (h_side2 : q.sides 1 = 200)
  (h_side3 : q.sides 2 = 300) :
  q.sides 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l460_46055


namespace NUMINAMATH_CALUDE_sector_area_given_arc_length_l460_46016

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4 cm,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area_given_arc_length (r : ℝ) : r * 2 = 4 → r^2 = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_given_arc_length_l460_46016


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_l460_46031

theorem median_of_consecutive_integers (n : ℕ) (a : ℤ) (h : n > 0) :
  (∀ i, 0 ≤ i ∧ i < n → (a + i) + (a + (n - 1) - i) = 120) →
  (n % 2 = 1 → (a + (n - 1) / 2) = 60) ∧
  (n % 2 = 0 → (2 * a + n - 1) / 2 = 60) :=
sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_l460_46031


namespace NUMINAMATH_CALUDE_integer_root_values_l460_46009

def polynomial (x b : ℤ) : ℤ := x^3 + 2*x^2 + b*x + 8

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-81, -26, -12, -6, 4, 9, 47} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l460_46009


namespace NUMINAMATH_CALUDE_point_belongs_to_transformed_plane_l460_46072

/-- Plane equation coefficients -/
structure PlaneCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply similarity transformation to plane equation -/
def transformPlane (p : PlaneCoefficients) (k : ℝ) : PlaneCoefficients :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : PlaneCoefficients) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A belongs to the image of plane a after similarity transformation -/
theorem point_belongs_to_transformed_plane 
  (A : Point3D) 
  (a : PlaneCoefficients) 
  (k : ℝ) 
  (h1 : A.x = 1/2) 
  (h2 : A.y = 1/3) 
  (h3 : A.z = 1) 
  (h4 : a.a = 2) 
  (h5 : a.b = -3) 
  (h6 : a.c = 3) 
  (h7 : a.d = -2) 
  (h8 : k = 1.5) : 
  satisfiesPlane A (transformPlane a k) :=
sorry

end NUMINAMATH_CALUDE_point_belongs_to_transformed_plane_l460_46072


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l460_46056

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3 under certain conditions. -/
theorem sum_of_common_ratios_is_three
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0)
  (h : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l460_46056


namespace NUMINAMATH_CALUDE_circle_tangent_condition_l460_46005

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate for a circle being tangent to the x-axis at the origin -/
def is_tangent_at_origin (c : Circle) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + c.D * x + c.E * y + c.F = 0 → 
    (x = 0 ∧ y = 0) ∨ (x ≠ 0 ∧ y ≠ 0)

theorem circle_tangent_condition (c : Circle) :
  is_tangent_at_origin c → c.E ≠ 0 ∧ c.D = 0 ∧ c.F = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_condition_l460_46005


namespace NUMINAMATH_CALUDE_determinant_equals_zy_l460_46066

def matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  !![1, x, z; 1, x+z, z; 1, y, y+z]

theorem determinant_equals_zy (x y z : ℝ) : 
  Matrix.det (matrix x y z) = z * y := by sorry

end NUMINAMATH_CALUDE_determinant_equals_zy_l460_46066


namespace NUMINAMATH_CALUDE_f_composition_half_f_composition_eq_one_solutions_l460_46030

noncomputable section

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_half : f (f (1/2)) = 1/2 := by sorry

theorem f_composition_eq_one_solutions : 
  {x : ℝ | f (f x) = 1} = {1, Real.exp (Real.exp 1)} := by sorry

end NUMINAMATH_CALUDE_f_composition_half_f_composition_eq_one_solutions_l460_46030


namespace NUMINAMATH_CALUDE_shaded_area_is_24_5_l460_46079

/-- Represents the structure of the grid --/
structure Grid :=
  (rect1 : Int × Int)
  (rect2 : Int × Int)
  (rect3 : Int × Int)

/-- Calculates the area of a rectangle --/
def rectangleArea (dims : Int × Int) : Int :=
  dims.1 * dims.2

/-- Calculates the total area of the grid --/
def totalGridArea (g : Grid) : Int :=
  rectangleArea g.rect1 + rectangleArea g.rect2 + rectangleArea g.rect3

/-- Calculates the area of a right-angled triangle --/
def triangleArea (base height : Int) : Rat :=
  (base * height) / 2

/-- The main theorem stating the area of the shaded region --/
theorem shaded_area_is_24_5 (g : Grid) 
    (h1 : g.rect1 = (3, 4))
    (h2 : g.rect2 = (4, 5))
    (h3 : g.rect3 = (5, 6))
    (h4 : totalGridArea g = 62)
    (h5 : triangleArea 15 5 = 37.5) :
  totalGridArea g - triangleArea 15 5 = 24.5 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_24_5_l460_46079


namespace NUMINAMATH_CALUDE_cake_slices_kept_l460_46048

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (extra_eaten : ℕ) : 
  total_slices = 35 →
  eaten_fraction = 2/5 →
  extra_eaten = 3 →
  total_slices - (eaten_fraction * total_slices + extra_eaten) = 18 :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_kept_l460_46048


namespace NUMINAMATH_CALUDE_discount_order_difference_discount_order_difference_proof_l460_46081

/-- Proves that the difference between applying 25% off then $5 off, and applying $5 off then 25% off, on a $30 item, is 125 cents. -/
theorem discount_order_difference : ℝ → Prop :=
  fun original_price : ℝ =>
    let first_discount : ℝ := 5
    let second_discount_rate : ℝ := 0.25
    let price_25_then_5 := (original_price * (1 - second_discount_rate)) - first_discount
    let price_5_then_25 := (original_price - first_discount) * (1 - second_discount_rate)
    original_price = 30 →
    (price_25_then_5 - price_5_then_25) * 100 = 125

/-- The proof of the theorem. -/
theorem discount_order_difference_proof : discount_order_difference 30 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_discount_order_difference_proof_l460_46081


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l460_46059

theorem trigonometric_equation_solutions :
  ∃ (S : Finset ℝ), 
    (∀ X ∈ S, 0 < X ∧ X < 2 * Real.pi) ∧
    (∀ X ∈ S, 1 + 2 * Real.sin X - 4 * (Real.sin X)^2 - 8 * (Real.sin X)^3 = 0) ∧
    S.card = 4 ∧
    (∀ Y, 0 < Y ∧ Y < 2 * Real.pi → 
      (1 + 2 * Real.sin Y - 4 * (Real.sin Y)^2 - 8 * (Real.sin Y)^3 = 0) → 
      Y ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l460_46059


namespace NUMINAMATH_CALUDE_suzy_final_book_count_l460_46076

/-- Calculates the final number of books Suzy has after a series of transactions -/
def final_book_count (initial_books : ℕ) 
                     (wed_checkout : ℕ) 
                     (thu_return thu_checkout : ℕ) 
                     (fri_return : ℕ) : ℕ :=
  initial_books - wed_checkout + thu_return - thu_checkout + fri_return

/-- Theorem stating that Suzy ends up with 80 books given the specific transactions -/
theorem suzy_final_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_suzy_final_book_count_l460_46076


namespace NUMINAMATH_CALUDE_solve_gumball_problem_l460_46057

def gumball_problem (alicia_gumballs : ℕ) (remaining_gumballs : ℕ) : Prop :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_gumballs := total_gumballs - remaining_gumballs
  (taken_gumballs : ℚ) / (total_gumballs : ℚ) = 2/5

theorem solve_gumball_problem :
  gumball_problem 20 60 := by sorry

end NUMINAMATH_CALUDE_solve_gumball_problem_l460_46057


namespace NUMINAMATH_CALUDE_max_sum_coordinates_l460_46074

/-- Triangle DEF in the cartesian plane with the following properties:
  - Area of triangle DEF is 65
  - Coordinates of D are (10, 15)
  - Coordinates of F are (19, 18)
  - Coordinates of E are (r, s)
  - The line containing the median to side DF has slope -3
-/
def TriangleDEF (r s : ℝ) : Prop :=
  let d := (10, 15)
  let f := (19, 18)
  let e := (r, s)
  let area := 65
  let median_slope := -3
  -- Area condition
  area = (1/2) * abs (r * 15 + 10 * 18 + 19 * s - s * 10 - 15 * 19 - r * 18) ∧
  -- Median slope condition
  median_slope = (s - (33/2)) / (r - (29/2))

theorem max_sum_coordinates (r s : ℝ) :
  TriangleDEF r s → r + s ≤ 1454/15 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_coordinates_l460_46074


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_power_l460_46096

theorem imaginary_part_of_complex_power (i : ℂ) (h : i * i = -1) :
  let z := (1 + i) / (1 - i)
  Complex.im (z ^ 2023) = -Complex.im i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_power_l460_46096


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_zero_one_l460_46026

def A : Set ℝ := {x : ℝ | x^2 - 1 < 0}
def B : Set ℝ := {x : ℝ | x > 0}

theorem A_intersect_B_eq_open_zero_one : A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_zero_one_l460_46026


namespace NUMINAMATH_CALUDE_fraction_power_product_l460_46086

theorem fraction_power_product :
  (3 / 5 : ℚ) ^ 4 * (2 / 9 : ℚ) = 162 / 5625 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l460_46086


namespace NUMINAMATH_CALUDE_multiple_decimals_between_7_5_and_9_5_l460_46071

theorem multiple_decimals_between_7_5_and_9_5 : 
  ∃ (x y : ℝ), 7.5 < x ∧ x < y ∧ y < 9.5 :=
sorry

end NUMINAMATH_CALUDE_multiple_decimals_between_7_5_and_9_5_l460_46071


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l460_46022

/-- A number is a palindrome if it reads the same backward as forward -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The next palindrome after a given number -/
def nextPalindrome (n : ℕ) : ℕ := sorry

theorem greatest_possible_average_speed 
  (initial_reading : ℕ) 
  (drive_duration : ℝ) 
  (speed_limit : ℝ) 
  (h1 : isPalindrome initial_reading)
  (h2 : drive_duration = 2)
  (h3 : speed_limit = 65)
  (h4 : initial_reading = 12321) :
  let final_reading := nextPalindrome initial_reading
  let distance := final_reading - initial_reading
  let max_distance := drive_duration * speed_limit
  let average_speed := distance / drive_duration
  (distance ≤ max_distance ∧ isPalindrome final_reading) →
  average_speed ≤ 50 ∧ 
  ∃ (s : ℝ), s > 50 → 
    ¬∃ (d : ℕ), d > distance ∧ 
      d ≤ max_distance ∧ 
      isPalindrome (initial_reading + d) ∧
      s = d / drive_duration :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l460_46022


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l460_46041

theorem sqrt_expression_simplification :
  let x := Real.sqrt 97
  let y := Real.sqrt 486
  let z := Real.sqrt 125
  let w := Real.sqrt 54
  let v := Real.sqrt 49
  (x + y + z) / (w + v) = (x + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l460_46041


namespace NUMINAMATH_CALUDE_green_upgrade_area_l460_46027

/-- Proves that the actual average annual area of green upgrade is 90 million square meters --/
theorem green_upgrade_area (total_area : ℝ) (planned_years original_plan actual_plan : ℝ) :
  total_area = 180 →
  actual_plan = 2 * original_plan →
  planned_years - (total_area / actual_plan) = 2 →
  actual_plan = 90 := by
  sorry

end NUMINAMATH_CALUDE_green_upgrade_area_l460_46027


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l460_46095

theorem graduating_class_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 466 → diff = 212 → boys + (boys + diff) = total → boys = 127 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l460_46095


namespace NUMINAMATH_CALUDE_some_number_value_l460_46098

theorem some_number_value (x : ℝ) : 65 + 5 * 12 / (180 / x) = 66 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l460_46098


namespace NUMINAMATH_CALUDE_logarithm_product_equality_logarithm_expression_equality_l460_46063

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the lg function (log base 10)
noncomputable def lg (x : ℝ) : ℝ := log 10 x

theorem logarithm_product_equality : 
  log 2 25 * log 3 4 * log 5 9 = 8 := by sorry

theorem logarithm_expression_equality :
  1/2 * lg (32/49) - 4/3 * lg (Real.sqrt 8) + lg (Real.sqrt 245) = 1/2 := by sorry

end NUMINAMATH_CALUDE_logarithm_product_equality_logarithm_expression_equality_l460_46063


namespace NUMINAMATH_CALUDE_binomial_60_3_l460_46013

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l460_46013


namespace NUMINAMATH_CALUDE_intersection_area_of_specific_rectangles_l460_46092

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  angle : ℝ  -- Angle of rotation in radians

/-- Calculates the area of intersection between two rectangles -/
noncomputable def intersectionArea (r1 r2 : Rectangle) : ℝ :=
  sorry

/-- Theorem stating the area of intersection between two specific rectangles -/
theorem intersection_area_of_specific_rectangles :
  let r1 : Rectangle := { width := 4, height := 12, angle := 0 }
  let r2 : Rectangle := { width := 5, height := 10, angle := π/6 }
  intersectionArea r1 r2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_of_specific_rectangles_l460_46092


namespace NUMINAMATH_CALUDE_paint_house_theorem_l460_46061

/-- Represents the time (in hours) it takes to paint a house given the number of people working -/
def paintTime (people : ℕ) : ℚ :=
  24 / people

theorem paint_house_theorem :
  paintTime 4 = 6 →
  paintTime 3 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_house_theorem_l460_46061


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l460_46010

def sticker_counts : List ℕ := [5, 7, 7, 10, 11]

def num_packs : ℕ := 5

theorem average_stickers_per_pack :
  (sticker_counts.sum / num_packs : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l460_46010


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l460_46075

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x ∧ 
  x = 729/144 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l460_46075


namespace NUMINAMATH_CALUDE_binomial_identities_l460_46070

theorem binomial_identities (n k : ℕ) : 
  k * (n.choose k) = n * ((n - 1).choose (k - 1)) ∧ 
  (Finset.range (n + 1)).sum (λ k => k * (n.choose k)) = n * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_identities_l460_46070


namespace NUMINAMATH_CALUDE_min_value_of_f_l460_46091

/-- The quadratic function f(x) = (x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- The minimum value of f(x) is -3 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l460_46091


namespace NUMINAMATH_CALUDE_unique_prime_solution_l460_46060

theorem unique_prime_solution :
  ∀ (p q r : ℕ),
    Prime p ∧ Prime q ∧ Prime r →
    p + q^2 = r^4 →
    p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l460_46060


namespace NUMINAMATH_CALUDE_total_packs_eq_243_l460_46068

/-- The total number of packs sold in six villages -/
def total_packs : ℕ := 23 + 28 + 35 + 43 + 50 + 64

/-- Theorem stating that the total number of packs sold equals 243 -/
theorem total_packs_eq_243 : total_packs = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_eq_243_l460_46068


namespace NUMINAMATH_CALUDE_experiment_sequences_l460_46052

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 6

/-- The number of possible positions for procedure A -/
def a_positions : ℕ := 2

/-- The number of procedures excluding A -/
def remaining_procedures : ℕ := num_procedures - 1

/-- The number of arrangements of B and C -/
def bc_arrangements : ℕ := 2

theorem experiment_sequences :
  (a_positions * remaining_procedures.factorial * bc_arrangements) = 96 := by
  sorry

end NUMINAMATH_CALUDE_experiment_sequences_l460_46052


namespace NUMINAMATH_CALUDE_only_345_is_right_triangle_l460_46093

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- The theorem stating that among the given sets, only (3,4,5) forms a right triangle -/
theorem only_345_is_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
by sorry

end NUMINAMATH_CALUDE_only_345_is_right_triangle_l460_46093


namespace NUMINAMATH_CALUDE_quadratic_roots_l460_46050

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l460_46050


namespace NUMINAMATH_CALUDE_jester_win_prob_constant_l460_46034

/-- The probability of the Jester winning in a game with 2n-1 regular townspeople, one Jester, and one goon -/
def jester_win_probability (n : ℕ+) : ℚ :=
  1 / 3

/-- The game ends immediately if the Jester is sent to jail during the morning -/
axiom morning_jail_win (n : ℕ+) : 
  jester_win_probability n = 1 / (2 * n + 1) + 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

/-- The Jester does not win if sent to jail at night -/
axiom night_jail_no_win (n : ℕ+) :
  jester_win_probability n = 
    ((2 * n - 1) / (2 * n + 1)) * ((2 * n - 2) / (2 * n - 1)) * jester_win_probability (n - 1)

theorem jester_win_prob_constant (n : ℕ+) : 
  jester_win_probability n = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_jester_win_prob_constant_l460_46034


namespace NUMINAMATH_CALUDE_compute_fraction_power_l460_46032

theorem compute_fraction_power : 8 * (2 / 7)^4 = 128 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l460_46032


namespace NUMINAMATH_CALUDE_inequality_relations_l460_46067

theorem inequality_relations (a b : ℝ) (h : a > b) : 
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relations_l460_46067


namespace NUMINAMATH_CALUDE_remainder_sum_l460_46065

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l460_46065


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l460_46025

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums : Finset ℝ := {a + b, a + c, a + d, b + c, b + d, c + d}
  ∃ (x y : ℝ), x ∈ sums ∧ y ∈ sums ∧ 
    sums = {210, 345, 275, 255, x, y} →
    (∀ (u v : ℝ), u ∈ sums ∧ v ∈ sums → u + v ≤ 775) ∧
    (∃ (u v : ℝ), u ∈ sums ∧ v ∈ sums ∧ u + v = 775) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l460_46025


namespace NUMINAMATH_CALUDE_copy_pages_for_ten_dollars_l460_46085

/-- The number of pages that can be copied for a given amount of money, 
    given the cost of copying 5 pages --/
def pages_copied (cost_5_pages : ℚ) (amount : ℚ) : ℚ :=
  (amount / cost_5_pages) * 5

/-- Theorem stating that given the cost of 10 cents for 5 pages, 
    the number of pages that can be copied for $10 is 500 --/
theorem copy_pages_for_ten_dollars :
  pages_copied (10 / 100) (10 : ℚ) = 500 := by
  sorry

#eval pages_copied (10 / 100) 10

end NUMINAMATH_CALUDE_copy_pages_for_ten_dollars_l460_46085


namespace NUMINAMATH_CALUDE_cos_150_degrees_l460_46089

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l460_46089


namespace NUMINAMATH_CALUDE_total_cost_calculation_l460_46090

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10

def num_shirts : ℕ := 3
def num_hats : ℕ := 4
def num_jeans : ℕ := 2

theorem total_cost_calculation :
  shirt_cost * num_shirts + hat_cost * num_hats + jeans_cost * num_jeans = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l460_46090


namespace NUMINAMATH_CALUDE_yellow_ball_count_l460_46049

theorem yellow_ball_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l460_46049


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l460_46088

/-- The area of a regular octagon inscribed in a circle -/
theorem inscribed_octagon_area (r : ℝ) (h : r^2 = 256) :
  2 * (1 + Real.sqrt 2) * (r * Real.sqrt (2 - Real.sqrt 2))^2 = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l460_46088


namespace NUMINAMATH_CALUDE_division_in_ratio_l460_46028

theorem division_in_ratio (total : ℕ) (ratio_b ratio_c : ℕ) (amount_c : ℕ) : 
  total = 2000 →
  ratio_b = 4 →
  ratio_c = 16 →
  amount_c = total * ratio_c / (ratio_b + ratio_c) →
  amount_c = 1600 := by
sorry

end NUMINAMATH_CALUDE_division_in_ratio_l460_46028


namespace NUMINAMATH_CALUDE_person_height_from_shadow_l460_46035

/-- Given a tree and a person under the same light conditions, calculate the person's height -/
theorem person_height_from_shadow (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 60)
  (h2 : tree_shadow = 18)
  (h3 : person_shadow = 3) :
  (tree_height / tree_shadow) * person_shadow = 10 := by
  sorry

end NUMINAMATH_CALUDE_person_height_from_shadow_l460_46035


namespace NUMINAMATH_CALUDE_function_existence_and_properties_l460_46097

/-- A function satisfying the given equation -/
def SatisfiesEquation (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f n m = (1/4) * (f (n-1) m + f (n+1) m + f n (m-1) + f n (m+1))

/-- The function is non-constant -/
def IsNonConstant (f : ℤ → ℤ → ℤ) : Prop :=
  ∃ n₁ m₁ n₂ m₂ : ℤ, f n₁ m₁ ≠ f n₂ m₂

/-- The function takes values both greater and less than any integer -/
def SpansAllIntegers (f : ℤ → ℤ → ℤ) : Prop :=
  ∀ k : ℤ, (∃ n₁ m₁ : ℤ, f n₁ m₁ > k) ∧ (∃ n₂ m₂ : ℤ, f n₂ m₂ < k)

/-- The main theorem -/
theorem function_existence_and_properties :
  ∃ f : ℤ → ℤ → ℤ, SatisfiesEquation f ∧ IsNonConstant f ∧ SpansAllIntegers f := by
  sorry

end NUMINAMATH_CALUDE_function_existence_and_properties_l460_46097


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l460_46084

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 4 → Nat.lcm a b = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l460_46084


namespace NUMINAMATH_CALUDE_box_surface_area_l460_46051

def sheet_length : ℕ := 25
def sheet_width : ℕ := 40
def corner_size : ℕ := 8

def surface_area : ℕ :=
  sheet_length * sheet_width - 4 * (corner_size * corner_size)

theorem box_surface_area :
  surface_area = 744 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l460_46051


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l460_46011

theorem max_product_constrained_sum (x y : ℝ) (h1 : x + y = 40) (h2 : x > 0) (h3 : y > 0) :
  x * y ≤ 400 ∧ ∃ (a b : ℝ), a + b = 40 ∧ a > 0 ∧ b > 0 ∧ a * b = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l460_46011


namespace NUMINAMATH_CALUDE_megan_total_songs_l460_46000

/-- Calculates the total number of songs bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Proves that the total number of songs bought is correct for Megan's scenario. -/
theorem megan_total_songs :
  total_songs 8 2 7 = 42 :=
by sorry

end NUMINAMATH_CALUDE_megan_total_songs_l460_46000


namespace NUMINAMATH_CALUDE_inverse_proportion_point_relation_l460_46021

/-- Given two points A(x₁, 2) and B(x₂, 4) on the graph of y = k/x where k > 0,
    prove that x₁ > x₂ > 0 -/
theorem inverse_proportion_point_relation (k x₁ x₂ : ℝ) 
  (h_k : k > 0)
  (h_A : 2 = k / x₁)
  (h_B : 4 = k / x₂) :
  x₁ > x₂ ∧ x₂ > 0 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_relation_l460_46021


namespace NUMINAMATH_CALUDE_nested_expression_value_l460_46080

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l460_46080


namespace NUMINAMATH_CALUDE_blue_eyed_percentage_l460_46036

/-- Represents the number of kittens with blue eyes for a cat -/
def blue_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 3 else 4

/-- Represents the number of kittens with brown eyes for a cat -/
def brown_eyed_kittens (cat : Nat) : Nat :=
  if cat = 1 then 7 else 6

/-- The total number of kittens -/
def total_kittens : Nat :=
  (blue_eyed_kittens 1 + brown_eyed_kittens 1) + (blue_eyed_kittens 2 + brown_eyed_kittens 2)

/-- The total number of blue-eyed kittens -/
def total_blue_eyed : Nat :=
  blue_eyed_kittens 1 + blue_eyed_kittens 2

/-- Theorem stating that 35% of all kittens have blue eyes -/
theorem blue_eyed_percentage :
  (total_blue_eyed : ℚ) / (total_kittens : ℚ) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_blue_eyed_percentage_l460_46036


namespace NUMINAMATH_CALUDE_unique_solution_l460_46020

theorem unique_solution : ∃! x : ℤ, x^2 + 105 = (x - 20)^2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l460_46020


namespace NUMINAMATH_CALUDE_probability_of_correct_distribution_l460_46082

/-- Represents the types of rolls -/
inductive RollType
  | Nut
  | Cheese
  | Fruit
  | Chocolate

/-- Represents a guest's set of rolls -/
def GuestRolls := Finset RollType

/-- The number of guests -/
def num_guests : Nat := 3

/-- The number of roll types -/
def num_roll_types : Nat := 4

/-- The total number of rolls -/
def total_rolls : Nat := num_guests * num_roll_types

/-- A function to calculate the probability of a specific distribution of rolls -/
noncomputable def probability_of_distribution (distribution : Finset GuestRolls) : ℚ := sorry

/-- The correct distribution where each guest has one of each roll type -/
def correct_distribution : Finset GuestRolls := sorry

/-- Theorem stating that the probability of the correct distribution is 24/1925 -/
theorem probability_of_correct_distribution :
  probability_of_distribution correct_distribution = 24 / 1925 := by sorry

end NUMINAMATH_CALUDE_probability_of_correct_distribution_l460_46082


namespace NUMINAMATH_CALUDE_greater_fourteen_game_count_l460_46023

/-- Represents a basketball league with two divisions -/
structure BasketballLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of scheduled games in the league -/
def total_games (league : BasketballLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  total_teams * games_per_team / 2

/-- The Greater Fourteen Basketball League -/
def greater_fourteen : BasketballLeague :=
  { divisions := 2,
    teams_per_division := 7,
    intra_division_games := 2,
    inter_division_games := 2 }

theorem greater_fourteen_game_count :
  total_games greater_fourteen = 182 := by
  sorry

end NUMINAMATH_CALUDE_greater_fourteen_game_count_l460_46023


namespace NUMINAMATH_CALUDE_profit_at_least_150_cents_l460_46002

-- Define the buying and selling prices
def orange_buy_price : ℚ := 15 / 4
def orange_sell_price : ℚ := 35 / 7
def apple_buy_price : ℚ := 20 / 5
def apple_sell_price : ℚ := 50 / 8

-- Define the profit function
def profit (num_oranges num_apples : ℕ) : ℚ :=
  (orange_sell_price - orange_buy_price) * num_oranges +
  (apple_sell_price - apple_buy_price) * num_apples

-- Theorem statement
theorem profit_at_least_150_cents :
  profit 43 43 ≥ 150 := by sorry

end NUMINAMATH_CALUDE_profit_at_least_150_cents_l460_46002


namespace NUMINAMATH_CALUDE_michael_paid_594_l460_46019

/-- The total amount Michael paid for his purchases after discounts -/
def total_paid (suit_price shoes_price shirt_price tie_price : ℕ) 
  (suit_discount shoes_discount shirt_tie_discount_percent : ℕ) : ℕ :=
  let suit_discounted := suit_price - suit_discount
  let shoes_discounted := shoes_price - shoes_discount
  let shirt_tie_total := shirt_price + tie_price
  let shirt_tie_discount := shirt_tie_total * shirt_tie_discount_percent / 100
  let shirt_tie_discounted := shirt_tie_total - shirt_tie_discount
  suit_discounted + shoes_discounted + shirt_tie_discounted

/-- Theorem stating that Michael paid $594 for his purchases -/
theorem michael_paid_594 :
  total_paid 430 190 80 50 100 30 20 = 594 := by sorry

end NUMINAMATH_CALUDE_michael_paid_594_l460_46019


namespace NUMINAMATH_CALUDE_metal_waste_l460_46012

/-- Given a rectangle with sides a and b (a < b), calculate the total metal wasted
    after cutting out a maximum circular piece and then a maximum square piece from the circle. -/
theorem metal_waste (a b : ℝ) (h : 0 < a ∧ a < b) :
  let circle_area := Real.pi * (a / 2)^2
  let square_side := a / Real.sqrt 2
  let square_area := square_side^2
  ab - square_area = ab - a^2 / 2 := by sorry

end NUMINAMATH_CALUDE_metal_waste_l460_46012


namespace NUMINAMATH_CALUDE_barometric_pressure_proof_l460_46024

/-- Represents the combined gas law equation -/
def combined_gas_law (p1 v1 T1 p2 v2 T2 : ℝ) : Prop :=
  p1 * v1 / T1 = p2 * v2 / T2

/-- Calculates the absolute temperature from Celsius -/
def absolute_temp (celsius : ℝ) : ℝ := celsius + 273

theorem barometric_pressure_proof 
  (well_functioning_pressure : ℝ) 
  (faulty_pressure_15C : ℝ) 
  (faulty_pressure_30C : ℝ) 
  (air_free_space : ℝ) :
  well_functioning_pressure = 762 →
  faulty_pressure_15C = 704 →
  faulty_pressure_30C = 692 →
  air_free_space = 143 →
  ∃ (true_pressure : ℝ),
    true_pressure = 748 ∧
    combined_gas_law 
      (well_functioning_pressure - faulty_pressure_15C) 
      air_free_space 
      (absolute_temp 15)
      (true_pressure - faulty_pressure_30C) 
      (air_free_space + (faulty_pressure_15C - faulty_pressure_30C)) 
      (absolute_temp 30) :=
by sorry

end NUMINAMATH_CALUDE_barometric_pressure_proof_l460_46024


namespace NUMINAMATH_CALUDE_cube_labeling_impossible_l460_46058

/-- Represents a cube with vertices labeled by natural numbers -/
structure LabeledCube :=
  (vertices : Fin 8 → ℕ)
  (is_permutation : Function.Bijective vertices)

/-- The set of edges in a cube -/
def cube_edges : Finset (Fin 8 × Fin 8) := sorry

/-- The sum of labels at the ends of an edge -/
def edge_sum (c : LabeledCube) (e : Fin 8 × Fin 8) : ℕ :=
  c.vertices e.1 + c.vertices e.2

/-- Theorem: It's impossible to label a cube's vertices with 1 to 8 such that all edge sums are different -/
theorem cube_labeling_impossible : 
  ¬ ∃ (c : LabeledCube), (∀ v : Fin 8, c.vertices v ∈ Finset.range 9 \ {0}) ∧ 
    (∀ e₁ e₂ : Fin 8 × Fin 8, e₁ ∈ cube_edges → e₂ ∈ cube_edges → e₁ ≠ e₂ → 
      edge_sum c e₁ ≠ edge_sum c e₂) :=
sorry

end NUMINAMATH_CALUDE_cube_labeling_impossible_l460_46058


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l460_46043

/-- Given a hyperbola with equation x² - my² = 1, where m is a real number,
    if the length of the conjugate axis is three times that of the transverse axis,
    then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 - m*y^2 = 1) → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*b = 3*(2*a) ∧ a^2 = 1 ∧ b^2 = 1/m) →
  m = 1/9 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l460_46043


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_2047_l460_46039

/-- The cycle of the last two digits of powers of 6 -/
def last_two_digits_cycle : List ℕ := [16, 96, 76, 56]

/-- The length of the cycle -/
def cycle_length : ℕ := 4

theorem tens_digit_of_6_pow_2047 (h : last_two_digits_cycle = [16, 96, 76, 56]) :
  (6^2047 / 10) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_2047_l460_46039


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l460_46053

theorem simplify_nested_roots (a : ℝ) : 
  (((a^9)^(1/6))^(1/3))^4 * (((a^9)^(1/3))^(1/6))^4 = a^4 := by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l460_46053


namespace NUMINAMATH_CALUDE_sum_of_fractions_l460_46054

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-5 : ℚ) / 6 = (-5 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l460_46054


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l460_46040

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l460_46040


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l460_46003

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61/60 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l460_46003


namespace NUMINAMATH_CALUDE_survey_min_overlap_l460_46037

/-- Given a survey of 120 people, where 95 like Mozart, 80 like Bach, and 75 like Beethoven,
    the minimum number of people who like both Mozart and Bach but not Beethoven is 45. -/
theorem survey_min_overlap (total : ℕ) (mozart : ℕ) (bach : ℕ) (beethoven : ℕ)
  (h_total : total = 120)
  (h_mozart : mozart = 95)
  (h_bach : bach = 80)
  (h_beethoven : beethoven = 75)
  (h_mozart_le : mozart ≤ total)
  (h_bach_le : bach ≤ total)
  (h_beethoven_le : beethoven ≤ total) :
  ∃ (overlap : ℕ), overlap ≥ 45 ∧
    overlap ≤ mozart ∧
    overlap ≤ bach ∧
    overlap ≤ total - beethoven ∧
    ∀ (x : ℕ), x < overlap →
      ¬(x ≤ mozart ∧ x ≤ bach ∧ x ≤ total - beethoven) :=
by sorry

end NUMINAMATH_CALUDE_survey_min_overlap_l460_46037


namespace NUMINAMATH_CALUDE_equation_solution_l460_46015

theorem equation_solution (r : ℝ) : 
  (r^2 - 6*r + 8)/(r^2 - 9*r + 20) = (r^2 - 3*r - 10)/(r^2 - 2*r - 15) ↔ r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l460_46015


namespace NUMINAMATH_CALUDE_oplus_example_l460_46004

/-- Definition of the ⊕ operation -/
def oplus (a b c : ℝ) (k : ℤ) : ℝ := b^2 - k * (a^2 * c)

/-- Theorem stating that ⊕(2, 5, 3, 3) = -11 -/
theorem oplus_example : oplus 2 5 3 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_oplus_example_l460_46004


namespace NUMINAMATH_CALUDE_outfit_combinations_l460_46064

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 4 → pants = 5 → hats = 3 → shirts * pants * hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l460_46064


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l460_46006

theorem sum_of_multiples_is_even (c d : ℤ) (hc : 6 ∣ c) (hd : 9 ∣ d) : Even (c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l460_46006


namespace NUMINAMATH_CALUDE_shekars_math_marks_l460_46029

def science_marks : ℝ := 65
def social_studies_marks : ℝ := 82
def english_marks : ℝ := 62
def biology_marks : ℝ := 85
def average_marks : ℝ := 74
def number_of_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ (math_marks : ℝ),
    (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / number_of_subjects = average_marks ∧
    math_marks = 76 := by
  sorry

end NUMINAMATH_CALUDE_shekars_math_marks_l460_46029


namespace NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l460_46042

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (g : ℂ → ℂ) 
  (h₁ : ∀ x, g x = x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h₂ : g (-3*I) = 0) 
  (h₃ : g (1 + I) = 0) : 
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_coefficients_l460_46042


namespace NUMINAMATH_CALUDE_campbell_geometry_qualification_l460_46046

/-- Represents the minimum score required in the 4th quarter to achieve a given average -/
def min_fourth_quarter_score (q1 q2 q3 : ℚ) (required_avg : ℚ) : ℚ :=
  4 * required_avg - (q1 + q2 + q3)

/-- Theorem: Given Campbell's scores and the required average, the minimum 4th quarter score is 107% -/
theorem campbell_geometry_qualification (campbell_q1 campbell_q2 campbell_q3 : ℚ)
  (h1 : campbell_q1 = 84/100)
  (h2 : campbell_q2 = 79/100)
  (h3 : campbell_q3 = 70/100)
  (required_avg : ℚ)
  (h4 : required_avg = 85/100) :
  min_fourth_quarter_score campbell_q1 campbell_q2 campbell_q3 required_avg = 107/100 := by
sorry

#eval min_fourth_quarter_score (84/100) (79/100) (70/100) (85/100)

end NUMINAMATH_CALUDE_campbell_geometry_qualification_l460_46046


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l460_46073

/-- An ellipse with the given properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  minor_axis_length : b = Real.sqrt 3
  foci_triangle : ∃ (c : ℝ), a = 2 * c ∧ a^2 = b^2 + c^2

/-- The point P -/
def P : ℝ × ℝ := (0, 2)

/-- Theorem about the special ellipse and its properties -/
theorem special_ellipse_properties (E : SpecialEllipse) :
  -- 1. Standard equation
  E.a^2 = 4 ∧ E.b^2 = 3 ∧
  -- 2. Existence of line l
  ∃ (k : ℝ), 
    -- 3. Equation of line l
    (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) ∧
    -- Line passes through P and intersects the ellipse at two distinct points
    ∃ (M N : ℝ × ℝ), M ≠ N ∧
      M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
      N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
      M.2 = k * M.1 + P.2 ∧
      N.2 = k * N.1 + P.2 ∧
      -- Satisfying the dot product condition
      M.1 * N.1 + M.2 * N.2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l460_46073


namespace NUMINAMATH_CALUDE_pitcher_problem_l460_46008

theorem pitcher_problem (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := (5 / 6) * C
  let juice_per_cup := juice_in_pitcher / 3
  juice_per_cup / C = 5 / 18 := by
sorry

end NUMINAMATH_CALUDE_pitcher_problem_l460_46008


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_two_l460_46038

theorem sum_of_reciprocals_equals_two (a b c : ℝ) 
  (ha : a^3 - 2020*a + 1010 = 0)
  (hb : b^3 - 2020*b + 1010 = 0)
  (hc : c^3 - 2020*c + 1010 = 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  1/a + 1/b + 1/c = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_two_l460_46038


namespace NUMINAMATH_CALUDE_exists_axisymmetric_capital_letter_l460_46087

-- Define a type for capital letters
inductive CapitalLetter
  | A | B | C | D | E | F | G | H | I | J | K | L | M
  | N | O | P | Q | R | S | T | U | V | W | X | Y | Z

-- Define a predicate for axisymmetric figures
def isAxisymmetric (letter : CapitalLetter) : Prop :=
  sorry  -- The actual implementation would depend on how we define axisymmetry

-- Theorem statement
theorem exists_axisymmetric_capital_letter :
  ∃ (letter : CapitalLetter), 
    (letter = CapitalLetter.A ∨ 
     letter = CapitalLetter.B ∨ 
     letter = CapitalLetter.D ∨ 
     letter = CapitalLetter.E) ∧ 
    isAxisymmetric letter :=
by
  sorry


end NUMINAMATH_CALUDE_exists_axisymmetric_capital_letter_l460_46087


namespace NUMINAMATH_CALUDE_quadratic_condition_l460_46007

/-- For the equation (m-2)x^2 + 3mx + 1 = 0 to be a quadratic equation in x, m ≠ 2 must hold. -/
theorem quadratic_condition (m : ℝ) : 
  (∀ x, ∃ y, y = (m - 2) * x^2 + 3 * m * x + 1) → m ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_condition_l460_46007


namespace NUMINAMATH_CALUDE_rook_placement_l460_46062

/-- The number of ways to place 3 rooks on a 6 × 2006 chessboard such that they don't attack each other -/
def placeRooks : ℕ := 20 * 2006 * 2005 * 2004

/-- The width of the chessboard -/
def boardWidth : ℕ := 6

/-- The height of the chessboard -/
def boardHeight : ℕ := 2006

/-- The number of rooks to be placed -/
def numRooks : ℕ := 3

theorem rook_placement :
  placeRooks = (Nat.choose boardWidth numRooks) * 
               boardHeight * (boardHeight - 1) * (boardHeight - 2) := by
  sorry

end NUMINAMATH_CALUDE_rook_placement_l460_46062


namespace NUMINAMATH_CALUDE_solution_value_l460_46083

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/8 is the solution to the equation F(a,3,8) = F(a,5,10) -/
theorem solution_value :
  ∃ a : ℝ, F a 3 8 = F a 5 10 ∧ a = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l460_46083


namespace NUMINAMATH_CALUDE_cubic_equation_result_l460_46077

theorem cubic_equation_result (x : ℝ) (h : x^3 + 4*x^2 = 8) :
  x^5 + 80*x^3 = -376*x^2 - 32*x + 768 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l460_46077


namespace NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l460_46078

theorem four_integers_product_2002_sum_less_40 :
  ∀ (a b c d : ℕ+),
    a * b * c * d = 2002 →
    (a : ℕ) + b + c + d < 40 →
    ((a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨
     (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨
     (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
     (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨
     (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
     (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨
     (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
     (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨
     (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) ∨
     (a = 7 ∧ b = 2 ∧ c = 11 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 2 ∧ d = 13) ∨
     (a = 7 ∧ b = 11 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 2 ∧ c = 7 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 2 ∧ d = 13) ∨
     (a = 11 ∧ b = 7 ∧ c = 13 ∧ d = 2) ∨
     (a = 11 ∧ b = 13 ∧ c = 2 ∧ d = 7) ∨
     (a = 11 ∧ b = 13 ∧ c = 7 ∧ d = 2) ∨
     (a = 13 ∧ b = 2 ∧ c = 7 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 2 ∧ d = 11) ∨
     (a = 13 ∧ b = 7 ∧ c = 11 ∧ d = 2) ∨
     (a = 13 ∧ b = 11 ∧ c = 2 ∧ d = 7) ∨
     (a = 13 ∧ b = 11 ∧ c = 7 ∧ d = 2) ∨
     (a = 14 ∧ b = 1 ∧ c = 11 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 1 ∧ d = 13) ∨
     (a = 14 ∧ b = 11 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 1 ∧ c = 14 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 1 ∧ d = 13) ∨
     (a = 11 ∧ b = 14 ∧ c = 13 ∧ d = 1) ∨
     (a = 11 ∧ b = 13 ∧ c = 1 ∧ d = 14) ∨
     (a = 11 ∧ b = 13 ∧ c = 14 ∧ d = 1) ∨
     (a = 13 ∧ b = 1 ∧ c = 14 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 1 ∧ d = 11) ∨
     (a = 13 ∧ b = 14 ∧ c = 11 ∧ d = 1) ∨
     (a = 13 ∧ b = 11 ∧ c = 1 ∧ d = 14) ∨
     (a = 13 ∧ b = 11 ∧ c = 14 ∧ d = 1)) :=
by sorry

end NUMINAMATH_CALUDE_four_integers_product_2002_sum_less_40_l460_46078


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l460_46045

/-- The cubic equation 5z^3 - 4iz^2 + z - k = 0 has at least one real root for all positive real k -/
theorem cubic_equation_real_root (k : ℝ) (hk : k > 0) : 
  ∃ (z : ℂ), z.im = 0 ∧ 5 * z^3 - 4 * Complex.I * z^2 + z - k = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l460_46045


namespace NUMINAMATH_CALUDE_simplify_fraction_l460_46099

theorem simplify_fraction (a b : ℝ) 
  (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l460_46099


namespace NUMINAMATH_CALUDE_trig_identity_l460_46017

theorem trig_identity (α : ℝ) :
  (Real.sin (2 * α) - Real.sin (3 * α) + Real.sin (4 * α)) /
  (Real.cos (2 * α) - Real.cos (3 * α) + Real.cos (4 * α)) =
  Real.tan (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l460_46017


namespace NUMINAMATH_CALUDE_two_true_propositions_l460_46014

def p (x y : ℝ) : Prop := (x > |y|) → (x > y)

def q (x y : ℝ) : Prop := (x + y > 0) → (x^2 > y^2)

theorem two_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧
  ¬(¬(p x y) ∧ ¬(q x y)) ∧
  (p x y ∧ ¬(q x y)) ∧
  ¬(p x y ∧ q x y) :=
sorry

end NUMINAMATH_CALUDE_two_true_propositions_l460_46014
