import Mathlib

namespace NUMINAMATH_CALUDE_projects_equal_volume_projects_equal_days_l3607_360736

/-- Represents the dimensions of an excavation project -/
structure ProjectDimensions where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth to be dug given project dimensions -/
def calculateVolume (dimensions : ProjectDimensions) : ℝ :=
  dimensions.depth * dimensions.length * dimensions.breadth

/-- The dimensions of Project 1 -/
def project1 : ProjectDimensions := {
  depth := 100,
  length := 25,
  breadth := 30
}

/-- The dimensions of Project 2 -/
def project2 : ProjectDimensions := {
  depth := 75,
  length := 20,
  breadth := 50
}

/-- Theorem stating that the volumes of both projects are equal -/
theorem projects_equal_volume : calculateVolume project1 = calculateVolume project2 := by
  sorry

/-- Corollary stating that the number of days required for both projects is the same -/
theorem projects_equal_days (days1 days2 : ℕ) 
    (h : calculateVolume project1 = calculateVolume project2) : days1 = days2 := by
  sorry

end NUMINAMATH_CALUDE_projects_equal_volume_projects_equal_days_l3607_360736


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_and_complement_l3607_360758

theorem angle_with_special_supplement_and_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_and_complement_l3607_360758


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l3607_360728

theorem complex_equation_real_solution :
  ∀ x : ℝ, (x^2 + Complex.I * x + 6 : ℂ) = (2 * Complex.I + 5 * x : ℂ) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l3607_360728


namespace NUMINAMATH_CALUDE_cone_height_equals_radius_l3607_360719

/-- The height of a cone formed by rolling a semicircular sheet of iron -/
def coneHeight (R : ℝ) : ℝ := R

/-- Theorem stating that the height of the cone is equal to the radius of the semicircular sheet -/
theorem cone_height_equals_radius (R : ℝ) (h : R > 0) : 
  coneHeight R = R := by sorry

end NUMINAMATH_CALUDE_cone_height_equals_radius_l3607_360719


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3607_360781

theorem units_digit_of_fraction (n : ℕ) : n = 1994 → (5^n + 6^n) % 7 = 5 → (5^n + 6^n) % 10 = 1 → (5^n + 6^n) / 7 % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3607_360781


namespace NUMINAMATH_CALUDE_reflection_maps_points_l3607_360751

/-- Reflects a point across the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_maps_points :
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (-2, 5)
  let A' : ℝ × ℝ := (2, -3)
  let B' : ℝ × ℝ := (5, -2)
  reflect_y_eq_x A = A' ∧ reflect_y_eq_x B = B' := by
  sorry


end NUMINAMATH_CALUDE_reflection_maps_points_l3607_360751


namespace NUMINAMATH_CALUDE_product_of_fractions_l3607_360718

theorem product_of_fractions : (1 / 3 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * (3 / 7 : ℚ) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3607_360718


namespace NUMINAMATH_CALUDE_third_group_frequency_count_l3607_360712

theorem third_group_frequency_count :
  ∀ (n₁ n₂ n₃ n₄ n₅ : ℕ),
  n₁ + n₂ + n₃ = 160 →
  n₃ + n₄ + n₅ = 260 →
  (n₃ : ℝ) / (n₁ + n₂ + n₃ + n₄ + n₅ : ℝ) = 0.20 →
  n₃ = 70 :=
by sorry

end NUMINAMATH_CALUDE_third_group_frequency_count_l3607_360712


namespace NUMINAMATH_CALUDE_initial_time_is_six_hours_l3607_360731

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours (distance : ℝ) (speed_new : ℝ) (time_factor : ℝ) :
  distance = 288 →
  speed_new = 32 →
  time_factor = 3 / 2 →
  ∃ (time_initial : ℝ),
    distance = speed_new * (time_factor * time_initial) ∧
    time_initial = 6 := by
  sorry


end NUMINAMATH_CALUDE_initial_time_is_six_hours_l3607_360731


namespace NUMINAMATH_CALUDE_gcd_g_x_l3607_360772

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(4*x+11)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l3607_360772


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3607_360703

theorem quadratic_completing_square (x : ℝ) : 
  (∃ p q : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + p)^2 = q) → 
  (∃ q : ℝ, (∀ x : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + 1)^2 = q) ∧ q = 33) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3607_360703


namespace NUMINAMATH_CALUDE_expression_value_l3607_360710

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3607_360710


namespace NUMINAMATH_CALUDE_remainder_sum_mod_13_l3607_360725

theorem remainder_sum_mod_13 (a b c d : ℤ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_13_l3607_360725


namespace NUMINAMATH_CALUDE_function_value_2008_l3607_360786

theorem function_value_2008 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (4 - x)) 
  (h2 : ∀ x, f (2 - x) + f (x - 2) = 0) : 
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2008_l3607_360786


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3607_360722

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l3607_360722


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3607_360765

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 24*a^2 + 50*a - 14 = 0 →
  b^3 - 24*b^2 + 50*b - 14 = 0 →
  c^3 - 24*c^2 + 50*c - 14 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 476/15 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3607_360765


namespace NUMINAMATH_CALUDE_pencil_box_problem_l3607_360796

structure BoxOfPencils where
  blue : ℕ
  green : ℕ

def Vasya (box : BoxOfPencils) : Prop := box.blue ≥ 4
def Kolya (box : BoxOfPencils) : Prop := box.green ≥ 5
def Petya (box : BoxOfPencils) : Prop := box.blue ≥ 3 ∧ box.green ≥ 4
def Misha (box : BoxOfPencils) : Prop := box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_problem :
  ∃ (box : BoxOfPencils),
    (Vasya box ∧ ¬Kolya box ∧ Petya box ∧ Misha box) ∧
    ¬∃ (other_box : BoxOfPencils),
      ((¬Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ ¬Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ ¬Misha other_box)) :=
by
  sorry

#check pencil_box_problem

end NUMINAMATH_CALUDE_pencil_box_problem_l3607_360796


namespace NUMINAMATH_CALUDE_area_is_24_l3607_360735

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := |3 * x| + |4 * y| = 12

/-- The graph is symmetric with respect to both x-axis and y-axis -/
axiom symmetry : ∀ x y : ℝ, equation x y ↔ equation (-x) y ∧ equation x (-y)

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 24 square units -/
theorem area_is_24 : enclosed_area = 24 :=
sorry

end NUMINAMATH_CALUDE_area_is_24_l3607_360735


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3607_360773

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  perimeter : ℝ
  square_side : ℝ
  perimeter_eq : perimeter = 8 * square_side

/-- The area of a rectangle divided into four identical squares -/
def area (rect : RectangleWithSquares) : ℝ :=
  4 * rect.square_side^2

/-- Theorem: A rectangle with perimeter 160 cm divided into four identical squares has an area of 1600 cm² -/
theorem rectangle_area_theorem (rect : RectangleWithSquares) (h : rect.perimeter = 160) :
  area rect = 1600 := by
  sorry

#check rectangle_area_theorem

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3607_360773


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l3607_360743

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2))
  (h2 : 3 ∣ (a*b + b*c + c*a)) :
  6 ∣ (a^3 + b^3 + c^3) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l3607_360743


namespace NUMINAMATH_CALUDE_data_set_mode_l3607_360791

def data_set : List ℕ := [9, 7, 10, 8, 10, 9, 10]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem data_set_mode :
  mode data_set = 10 := by sorry

end NUMINAMATH_CALUDE_data_set_mode_l3607_360791


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3607_360714

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -a - c - e → 
  3 * ((a + b * I) + (c + d * I) + (e + f * I) + (g + h * I)) = 2 * I → 
  d + f + h = -4/3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3607_360714


namespace NUMINAMATH_CALUDE_divisibility_of_m_l3607_360713

theorem divisibility_of_m (m : ℤ) : m = 76^2006 - 76 → 100 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_m_l3607_360713


namespace NUMINAMATH_CALUDE_proportion_condition_l3607_360746

theorem proportion_condition (a b c : ℝ) (h : b ≠ 0 ∧ c ≠ 0) : 
  (∃ x y : ℝ, x / y = a / b ∧ y / x = b / c ∧ x^2 ≠ y * x) ∧
  (a / b = b / c → b^2 = a * c) ∧
  ¬(b^2 = a * c → a / b = b / c) :=
sorry

end NUMINAMATH_CALUDE_proportion_condition_l3607_360746


namespace NUMINAMATH_CALUDE_function_value_theorem_l3607_360762

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x + 3) :
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_function_value_theorem_l3607_360762


namespace NUMINAMATH_CALUDE_f_properties_l3607_360766

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x - 15

-- Define the theorem
theorem f_properties (a : ℝ) (x : ℝ) (h : |x - a| < 1) :
  (∃ (y : ℝ), |f y| > 5 ↔ (y < -4 ∨ y > 5 ∨ ((1 - Real.sqrt 41) / 2 < y ∧ y < (1 + Real.sqrt 41) / 2))) ∧
  |f x - f a| < 2 * (|a| + 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3607_360766


namespace NUMINAMATH_CALUDE_sqrt_equation_equivalence_l3607_360733

theorem sqrt_equation_equivalence (x : ℝ) (h : x > 6) :
  Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3 ↔ x ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_equivalence_l3607_360733


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3607_360754

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (sum_eq : p + q + r + s + t + u = 8) : 
  1/p + 9/q + 16/r + 25/s + 36/t + 49/u ≥ 84.5 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ),
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 8 ∧
    1/p' + 9/q' + 16/r' + 25/s' + 36/t' + 49/u' = 84.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3607_360754


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3607_360793

theorem solution_to_system_of_equations :
  let solutions : List (ℝ × ℝ) := [
    (Real.sqrt 5, Real.sqrt 6), (Real.sqrt 5, -Real.sqrt 6),
    (-Real.sqrt 5, Real.sqrt 6), (-Real.sqrt 5, -Real.sqrt 6),
    (Real.sqrt 6, Real.sqrt 5), (Real.sqrt 6, -Real.sqrt 5),
    (-Real.sqrt 6, Real.sqrt 5), (-Real.sqrt 6, -Real.sqrt 5)
  ]
  ∀ (x y : ℝ),
    (3 * x^2 + 3 * y^2 - x^2 * y^2 = 3 ∧
     x^4 + y^4 - x^2 * y^2 = 31) ↔
    (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3607_360793


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3607_360704

/-- The area of the shaded region not covered by four circles centered at the vertices of a square -/
theorem shaded_area_square_with_circles (side_length radius : ℝ) (h1 : side_length = 8) (h2 : radius = 3) :
  side_length ^ 2 - 4 * Real.pi * radius ^ 2 = 64 - 36 * Real.pi := by
  sorry

#check shaded_area_square_with_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3607_360704


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_46_l3607_360715

theorem no_primes_divisible_by_46 : ∀ p : ℕ, Nat.Prime p → ¬(46 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_46_l3607_360715


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l3607_360740

/-- Given two employees X and Y with a total pay of 330 and Y's pay of 150,
    prove that X's pay as a percentage of Y's pay is 120%. -/
theorem employee_pay_percentage (total_pay : ℝ) (y_pay : ℝ) :
  total_pay = 330 →
  y_pay = 150 →
  (total_pay - y_pay) / y_pay * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l3607_360740


namespace NUMINAMATH_CALUDE_chihuahua_grooming_time_l3607_360700

/-- The time Karen takes to groom different types of dogs -/
structure GroomingTimes where
  rottweiler : ℕ
  border_collie : ℕ
  chihuahua : ℕ

/-- The number of each type of dog Karen grooms -/
structure DogCounts where
  rottweilers : ℕ
  border_collies : ℕ
  chihuahuas : ℕ

/-- Calculates the total grooming time for all dogs -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.border_collie * counts.border_collies +
  times.chihuahua * counts.chihuahuas

theorem chihuahua_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
  times.rottweiler = 20 →
  times.border_collie = 10 →
  counts.rottweilers = 6 →
  counts.border_collies = 9 →
  counts.chihuahuas = 1 →
  totalGroomingTime times counts = 255 →
  times.chihuahua = 45 := by
  sorry

end NUMINAMATH_CALUDE_chihuahua_grooming_time_l3607_360700


namespace NUMINAMATH_CALUDE_crimson_valley_skirts_l3607_360741

/-- The number of skirts in each valley -/
structure ValleySkirts where
  ember : ℕ
  azure : ℕ
  seafoam : ℕ
  purple : ℕ
  crimson : ℕ

/-- The conditions for the valley skirts problem -/
def valley_conditions (v : ValleySkirts) : Prop :=
  v.crimson = v.purple / 3 ∧
  v.purple = v.seafoam / 4 ∧
  v.seafoam = v.azure * 3 / 5 ∧
  v.azure = v.ember * 2 ∧
  v.ember = 120

/-- Theorem stating that given the conditions, Crimson Valley has 12 skirts -/
theorem crimson_valley_skirts (v : ValleySkirts) 
  (h : valley_conditions v) : v.crimson = 12 := by
  sorry

end NUMINAMATH_CALUDE_crimson_valley_skirts_l3607_360741


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l3607_360744

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 41 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 0

/-- Theorem: The radius of the circle described by the given equation is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y → ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l3607_360744


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_line_through_point_with_segment_l3607_360723

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0
def l₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 3 = 0

-- Define the perpendicular line n
def n (x y : ℝ) : Prop := y = -(Real.sqrt 3 / 3) * x + 2 ∨ y = -(Real.sqrt 3 / 3) * x - 2

-- Define the line m
def m (x y : ℝ) : Prop := x = Real.sqrt 3 ∨ y = (Real.sqrt 3 / 3) * x + 3

-- Theorem for part (1)
theorem perpendicular_line_equation
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_perp : ∀ x y, n x y → (∀ x' y', l₁ x' y' → (y - y') = (Real.sqrt 3 / 3) * (x - x')))
  (h_area : ∃ a b, n a 0 ∧ n 0 b ∧ a * b / 2 = 2 * Real.sqrt 3) :
  ∀ x y, n x y :=
sorry

-- Theorem for part (2)
theorem line_through_point_with_segment
  (h_parallel : ∀ x y, l₁ x y ↔ l₂ x y)
  (h_point : m (Real.sqrt 3) 4)
  (h_segment : ∃ x₁ y₁ x₂ y₂,
    m x₁ y₁ ∧ m x₂ y₂ ∧ l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2) :
  ∀ x y, m x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_line_through_point_with_segment_l3607_360723


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l3607_360734

/-- Proves the number of boxes of the third type of cookies Clara sells -/
theorem clara_cookie_sales (cookies_per_box1 cookies_per_box2 cookies_per_box3 : ℕ)
  (boxes_sold1 boxes_sold2 : ℕ) (total_cookies : ℕ)
  (h1 : cookies_per_box1 = 12)
  (h2 : cookies_per_box2 = 20)
  (h3 : cookies_per_box3 = 16)
  (h4 : boxes_sold1 = 50)
  (h5 : boxes_sold2 = 80)
  (h6 : total_cookies = 3320)
  (h7 : total_cookies = cookies_per_box1 * boxes_sold1 + cookies_per_box2 * boxes_sold2 + cookies_per_box3 * boxes_sold3) :
  boxes_sold3 = 70 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l3607_360734


namespace NUMINAMATH_CALUDE_eight_million_two_hundred_thousand_scientific_notation_l3607_360707

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eight_million_two_hundred_thousand_scientific_notation :
  toScientificNotation 8200000 = ScientificNotation.mk 8.2 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_eight_million_two_hundred_thousand_scientific_notation_l3607_360707


namespace NUMINAMATH_CALUDE_root_power_equality_l3607_360790

theorem root_power_equality (x : ℝ) (h : x > 0) :
  (x^((1:ℝ)/5)) / (x^((1:ℝ)/2)) = x^(-(3:ℝ)/10) := by sorry

end NUMINAMATH_CALUDE_root_power_equality_l3607_360790


namespace NUMINAMATH_CALUDE_volleyball_team_selection_16_6_2_1_l3607_360711

def volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) (c : ℕ) : ℕ :=
  Nat.choose (n - t - c) (k - c) + t * Nat.choose (n - t - c) (k - c - 1)

theorem volleyball_team_selection_16_6_2_1 :
  volleyball_team_selection 16 6 2 1 = 2717 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_16_6_2_1_l3607_360711


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3607_360757

theorem sum_of_coefficients (k : ℝ) (h : k ≠ 0) : ∃ (a b c d : ℤ),
  (8 * k + 9 + 10 * k^2 - 3 * k^3) + (4 * k + 6 + k^2 + k^3) = 
  (a : ℝ) * k^3 + (b : ℝ) * k^2 + (c : ℝ) * k + (d : ℝ) ∧ 
  a + b + c + d = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3607_360757


namespace NUMINAMATH_CALUDE_lcm_18_30_l3607_360702

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l3607_360702


namespace NUMINAMATH_CALUDE_root_of_equation_l3607_360770

theorem root_of_equation (x : ℝ) : 
  (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = -4.5) :=
by sorry

end NUMINAMATH_CALUDE_root_of_equation_l3607_360770


namespace NUMINAMATH_CALUDE_sum_abc_equals_33_l3607_360756

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : N = 5 * a + 3 * b + 5 * c)
  (h_eq2 : N = 4 * a + 5 * b + 4 * c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_33_l3607_360756


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l3607_360753

theorem sum_of_solutions_eq_six :
  ∃ (M₁ M₂ : ℝ), (M₁ * (M₁ - 6) = -5) ∧ (M₂ * (M₂ - 6) = -5) ∧ (M₁ + M₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l3607_360753


namespace NUMINAMATH_CALUDE_marta_textbook_expenses_l3607_360761

/-- The total amount Marta spent on textbooks -/
def total_spent (sale_price : ℕ) (sale_quantity : ℕ) (online_total : ℕ) (bookstore_multiplier : ℕ) : ℕ :=
  sale_price * sale_quantity + online_total + bookstore_multiplier * online_total

/-- Theorem stating the total amount Marta spent on textbooks -/
theorem marta_textbook_expenses : total_spent 10 5 40 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_marta_textbook_expenses_l3607_360761


namespace NUMINAMATH_CALUDE_segment_length_l3607_360771

/-- Given three points on a line, prove that the length of AC is either 7 or 1 -/
theorem segment_length (A B C : ℝ) : 
  (B - A = 4) → (C - B = 3) → (C - A = 7 ∨ C - A = 1) := by sorry

end NUMINAMATH_CALUDE_segment_length_l3607_360771


namespace NUMINAMATH_CALUDE_algebraic_identities_l3607_360755

variable (a b : ℝ)

theorem algebraic_identities :
  ((a - 2*b)^2 - (b - a)*(a + b) = 2*a^2 - 4*a*b + 3*b^2) ∧
  ((2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3607_360755


namespace NUMINAMATH_CALUDE_initial_marbles_l3607_360779

theorem initial_marbles (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 8 → 
  remaining = 79 → 
  initial = 87 := by sorry

end NUMINAMATH_CALUDE_initial_marbles_l3607_360779


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3607_360788

/-- Calculates the number of rectangles in a grid with sides along the grid lines -/
def count_rectangles (m n : ℕ) : ℕ :=
  let horizontal := (m * (m + 1) * (n + 1)) / 2
  let vertical := (n * (n + 1) * (m + 1)) / 2
  horizontal + vertical - (m * n)

/-- The theorem stating that a 5x4 grid contains 24 rectangles -/
theorem rectangles_in_5x4_grid :
  count_rectangles 5 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3607_360788


namespace NUMINAMATH_CALUDE_cleaning_time_theorem_l3607_360738

/-- Represents the grove of trees -/
structure Grove :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the time to clean each tree without help -/
def time_per_tree_without_help (g : Grove) (total_time_with_help : ℕ) : ℚ :=
  let total_trees := g.rows * g.columns
  let time_per_tree_with_help := total_time_with_help / total_trees
  2 * time_per_tree_with_help

theorem cleaning_time_theorem (g : Grove) (h : g.rows = 4 ∧ g.columns = 5) :
  time_per_tree_without_help g 60 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_theorem_l3607_360738


namespace NUMINAMATH_CALUDE_largest_in_systematic_sample_l3607_360794

/-- Represents a systematic sample --/
structure SystematicSample where
  total : Nat
  start : Nat
  interval : Nat

/-- Checks if a number is in the systematic sample --/
def inSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = s.start + k * s.interval ∧ n ≤ s.total

/-- The largest number in the sample --/
def largestInSample (s : SystematicSample) : Nat :=
  s.start + ((s.total - s.start) / s.interval) * s.interval

theorem largest_in_systematic_sample
  (employees : Nat)
  (first : Nat)
  (second : Nat)
  (h1 : employees = 500)
  (h2 : first = 6)
  (h3 : second = 31)
  (h4 : second - first = 31 - 6) :
  let s := SystematicSample.mk employees first (second - first)
  largestInSample s = 481 :=
by
  sorry

#check largest_in_systematic_sample

end NUMINAMATH_CALUDE_largest_in_systematic_sample_l3607_360794


namespace NUMINAMATH_CALUDE_jellybeans_theorem_l3607_360798

def jellybeans_problem (initial_jellybeans : ℕ) (normal_class_size : ℕ) (sick_children : ℕ) (jellybeans_per_child : ℕ) : Prop :=
  let attending_children := normal_class_size - sick_children
  let eaten_jellybeans := attending_children * jellybeans_per_child
  let remaining_jellybeans := initial_jellybeans - eaten_jellybeans
  remaining_jellybeans = 34

theorem jellybeans_theorem :
  jellybeans_problem 100 24 2 3 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_theorem_l3607_360798


namespace NUMINAMATH_CALUDE_maria_budget_excess_l3607_360778

theorem maria_budget_excess : 
  let sweater_price : ℚ := 35
  let scarf_price : ℚ := 25
  let mittens_price : ℚ := 15
  let hat_price : ℚ := 12
  let family_members : ℕ := 15
  let discount_threshold : ℚ := 800
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  let spending_limit : ℚ := 1500

  let set_price := 2 * sweater_price + scarf_price + mittens_price + hat_price
  let total_price := family_members * set_price
  let discounted_price := if total_price > discount_threshold 
                          then total_price * (1 - discount_rate) 
                          else total_price
  let final_price := discounted_price * (1 + sales_tax_rate)

  final_price - spending_limit = 229.35 := by sorry

end NUMINAMATH_CALUDE_maria_budget_excess_l3607_360778


namespace NUMINAMATH_CALUDE_equation_implication_l3607_360792

theorem equation_implication (x : ℝ) : 3 * x + 2 = 11 → 6 * x + 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_implication_l3607_360792


namespace NUMINAMATH_CALUDE_simplify_fraction_l3607_360732

theorem simplify_fraction (a b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2*a + 1) / (1 - b / (2*b - 1)) = (2*a + 1) * (2*b - 1) / (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3607_360732


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_l3607_360705

theorem stratified_sampling_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) :
  total_students = 3600 →
  second_year_students = 900 →
  sample_size = 720 →
  (second_year_students * sample_size) / total_students = 180 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_l3607_360705


namespace NUMINAMATH_CALUDE_monthly_income_problem_l3607_360748

/-- Given the average monthly incomes of three people, prove the income of one person -/
theorem monthly_income_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 4050)
  (h2 : (B + C) / 2 = 5250)
  (h3 : (A + C) / 2 = 4200) :
  A = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_problem_l3607_360748


namespace NUMINAMATH_CALUDE_unique_intersection_l3607_360768

/-- A function f(x) that represents a quadratic or linear equation depending on the value of a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Theorem stating that f(x) intersects the x-axis at only one point iff a = 0, 1, or 9 -/
theorem unique_intersection (a : ℝ) :
  (∃! x, f a x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l3607_360768


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l3607_360797

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter
  (s : ℝ)  -- side length of the equilateral triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : 2 * s + 5 = 45)  -- condition from isosceles triangle
  : 3 * s = 60 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem equilateral_triangle_perimeter_alt
  (s : ℝ)  -- side length of the equilateral triangle
  (P_isosceles : ℝ)  -- perimeter of the isosceles triangle
  (b : ℝ)  -- base of the isosceles triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : P_isosceles = 45)  -- given perimeter of isosceles triangle
  (h3 : b = 5)  -- given base of isosceles triangle
  (h4 : P_isosceles = 2 * s + b)  -- definition of isosceles triangle perimeter
  : 3 * s = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l3607_360797


namespace NUMINAMATH_CALUDE_johnnys_jogging_speed_l3607_360763

/-- Proves that given the specified conditions, Johnny's jogging speed to school is approximately 9.333333333333334 miles per hour -/
theorem johnnys_jogging_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (bus_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 6.461538461538462) 
  (h3 : bus_speed = 21) : 
  ∃ (jogging_speed : ℝ), 
    (distance / jogging_speed + distance / bus_speed = total_time) ∧ 
    (abs (jogging_speed - 9.333333333333334) < 0.000001) := by
  sorry

end NUMINAMATH_CALUDE_johnnys_jogging_speed_l3607_360763


namespace NUMINAMATH_CALUDE_f_symmetry_l3607_360787

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3607_360787


namespace NUMINAMATH_CALUDE_blithe_toy_collection_l3607_360727

/-- Given Blithe's toy collection changes, prove the initial number of toys. -/
theorem blithe_toy_collection (X : ℕ) : 
  X - 6 + 9 + 5 - 3 = 43 → X = 38 := by
  sorry

end NUMINAMATH_CALUDE_blithe_toy_collection_l3607_360727


namespace NUMINAMATH_CALUDE_part1_range_of_m_part2_range_of_m_l3607_360730

-- Define the function f
def f (a m x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x + m

-- Part 1
theorem part1_range_of_m :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f 1 m x = 0 ∧ f 1 m y = 0 ∧ f 1 m z = 0) →
  -1 < m ∧ m < 5/27 :=
sorry

-- Part 2
theorem part2_range_of_m :
  ∀ m : ℝ, (∀ a : ℝ, 3 ≤ a ∧ a ≤ 6 →
    ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a m x ≤ 1) →
  m ≤ -87 :=
sorry

end NUMINAMATH_CALUDE_part1_range_of_m_part2_range_of_m_l3607_360730


namespace NUMINAMATH_CALUDE_angelina_speed_l3607_360749

/-- Angelina's journey from home to gym via grocery store -/
def angelina_journey (v : ℝ) : Prop :=
  let time_home_to_grocery := 200 / v
  let time_grocery_to_gym := 300 / (2 * v)
  time_home_to_grocery = time_grocery_to_gym + 50

theorem angelina_speed : ∃ v : ℝ, angelina_journey v ∧ v > 0 ∧ 2 * v = 2 := by
  sorry

end NUMINAMATH_CALUDE_angelina_speed_l3607_360749


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3607_360767

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x > 2*a - 3 ∧ 2*x ≥ 3*(x-2) + 5))) →
  (1/2 : ℝ) ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3607_360767


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3607_360720

theorem inscribed_squares_ratio : ∀ x y : ℝ,
  (5 : ℝ) ^ 2 + 12 ^ 2 = 13 ^ 2 →
  (12 - x) / 12 = x / 5 →
  y + 2 * (5 * y / 13) = 13 →
  x / y = 1380 / 2873 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3607_360720


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3607_360708

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3607_360708


namespace NUMINAMATH_CALUDE_angle_D_measure_l3607_360716

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity condition (sum of angles = 720°)
  A + B + C + D + E + F = 720 ∧
  -- Angle congruence conditions
  A = B ∧ B = C ∧
  D = E ∧
  F = 2 * D ∧
  -- Relationship between angles A and D
  A + 30 = D

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3607_360716


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l3607_360784

theorem factorial_equation_solution :
  ∃! (a b : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  (5 * 4 * 3 * 2 * 1)^8 + (5 * 4 * 3 * 2 * 1)^7 = 4000000000000000 + a * 100000000000000 + 356000000000000 + 400000000000 + 80000000000 + b * 10000000000 + 80000000000 ∧
  a = 3 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l3607_360784


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3607_360769

theorem trig_expression_equals_one : 
  Real.sqrt 3 * Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + Real.sin (45 * π / 180) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3607_360769


namespace NUMINAMATH_CALUDE_investment_interest_rate_calculation_l3607_360742

theorem investment_interest_rate_calculation 
  (total_investment : ℝ) 
  (known_rate : ℝ) 
  (unknown_investment : ℝ) 
  (income_difference : ℝ) :
  let known_investment := total_investment - unknown_investment
  let unknown_rate := (known_investment * known_rate - income_difference) / unknown_investment
  total_investment = 2000 ∧ 
  known_rate = 0.10 ∧ 
  unknown_investment = 800 ∧ 
  income_difference = 56 → 
  unknown_rate = 0.08 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_calculation_l3607_360742


namespace NUMINAMATH_CALUDE_point_on_line_l3607_360747

/-- Given a line passing through points (3, -5) and (5, 1), 
    prove that any point (7, y) on this line must have y = 7. -/
theorem point_on_line (y : ℝ) : 
  (∀ (x : ℝ), (x - 3) * (1 - (-5)) = (y - (-5)) * (5 - 3) → x = 7) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3607_360747


namespace NUMINAMATH_CALUDE_unitPrice_is_constant_l3607_360764

/-- Represents the data from a fuel dispenser --/
structure FuelDispenser :=
  (amount : ℝ)
  (unitPrice : ℝ)
  (unitPricePerYuanPerLiter : ℝ)

/-- The fuel dispenser data from the problem --/
def fuelData : FuelDispenser :=
  { amount := 116.64,
    unitPrice := 18,
    unitPricePerYuanPerLiter := 6.48 }

/-- Predicate to check if a value is constant in the fuel dispenser context --/
def isConstant (f : FuelDispenser → ℝ) : Prop :=
  ∀ (d1 d2 : FuelDispenser), d1.unitPrice = d2.unitPrice → f d1 = f d2

/-- Theorem stating that the unit price is constant --/
theorem unitPrice_is_constant :
  isConstant (λ d : FuelDispenser => d.unitPrice) :=
sorry

end NUMINAMATH_CALUDE_unitPrice_is_constant_l3607_360764


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l3607_360774

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sin (2 * θ) = a) : 
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l3607_360774


namespace NUMINAMATH_CALUDE_dice_probability_l3607_360737

def num_dice : ℕ := 6
def sides_per_die : ℕ := 8

def probability_at_least_two_same : ℚ := 3781 / 4096

theorem dice_probability :
  probability_at_least_two_same = 1 - (sides_per_die.factorial / (sides_per_die - num_dice).factorial) / sides_per_die ^ num_dice :=
by sorry

end NUMINAMATH_CALUDE_dice_probability_l3607_360737


namespace NUMINAMATH_CALUDE_min_r_for_perfect_square_l3607_360759

theorem min_r_for_perfect_square : 
  ∃ (r : ℕ), r > 0 ∧ 
  (∃ (n : ℕ), 4^3 + 4^r + 4^4 = n^2) ∧
  (∀ (s : ℕ), s > 0 ∧ s < r → ¬∃ (m : ℕ), 4^3 + 4^s + 4^4 = m^2) ∧
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_min_r_for_perfect_square_l3607_360759


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3607_360776

theorem trigonometric_equation_solution (x : Real) : 
  (8.456 * (Real.tan x)^2 * (Real.tan (3*x))^2 * Real.tan (4*x) = 
   (Real.tan x)^2 - (Real.tan (3*x))^2 + Real.tan (4*x)) ↔ 
  (∃ k : Int, x = k * Real.pi ∨ x = (Real.pi / 4) * (2 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3607_360776


namespace NUMINAMATH_CALUDE_correct_algebraic_equation_l3607_360726

theorem correct_algebraic_equation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_equation_l3607_360726


namespace NUMINAMATH_CALUDE_complex_vector_properties_l3607_360717

open Complex

theorem complex_vector_properties (x y : ℝ) : 
  let z₁ : ℂ := (1 + I) / I
  let z₂ : ℂ := x + y * I
  true → 
  (∃ (k : ℝ), z₁.re * k = z₂.re ∧ z₁.im * k = z₂.im → x + y = 0) ∧
  (z₁.re * z₂.re + z₁.im * z₂.im = 0 → abs (z₁ + z₂) = abs (z₁ - z₂)) := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_properties_l3607_360717


namespace NUMINAMATH_CALUDE_average_price_of_books_l3607_360745

/-- The average price of books bought by Rahim -/
theorem average_price_of_books (books_shop1 : ℕ) (price_shop1 : ℕ) 
  (books_shop2 : ℕ) (price_shop2 : ℕ) :
  books_shop1 = 40 →
  price_shop1 = 600 →
  books_shop2 = 20 →
  price_shop2 = 240 →
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2) = 14 := by
  sorry

#check average_price_of_books

end NUMINAMATH_CALUDE_average_price_of_books_l3607_360745


namespace NUMINAMATH_CALUDE_divisibility_by_6p_l3607_360724

theorem divisibility_by_6p (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, 7^p - 5^p - 2 = 6 * p * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_6p_l3607_360724


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3607_360760

theorem circle_diameter_from_area (A : Real) (r : Real) (d : Real) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3607_360760


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3607_360795

theorem sine_cosine_relation (α : Real) (h : Real.sin (α + π/6) = 1/3) : 
  Real.cos (α - π/3) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3607_360795


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_four_a_l3607_360785

theorem factorization_a_squared_minus_four_a (a : ℝ) : a^2 - 4*a = a*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_four_a_l3607_360785


namespace NUMINAMATH_CALUDE_garden_shorter_side_l3607_360752

theorem garden_shorter_side (perimeter : ℝ) (area : ℝ) : perimeter = 60 ∧ area = 200 → ∃ x y : ℝ, x ≤ y ∧ 2*x + 2*y = perimeter ∧ x*y = area ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_garden_shorter_side_l3607_360752


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l3607_360799

theorem fraction_cube_equality : (64000 ^ 3 : ℚ) / (16000 ^ 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l3607_360799


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_l3607_360789

theorem arithmetic_geometric_mean_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 = 2 * Real.sqrt 3 ∧ Real.sqrt (x * y) = Real.sqrt 3 → |x - y| = 6 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_l3607_360789


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3607_360706

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 104) : 
  a^2 + b^2 = 1392 := by sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3607_360706


namespace NUMINAMATH_CALUDE_distinct_results_count_l3607_360739

/-- Represents the possible operators that can replace * in the expression -/
inductive Operator
| Add
| Sub
| Mul
| Div

/-- Represents the expression as a list of operators -/
def Expression := List Operator

/-- Evaluates an expression according to the given rules -/
def evaluate (expr : Expression) : ℚ :=
  sorry

/-- Generates all possible expressions -/
def allExpressions : List Expression :=
  sorry

/-- Counts the number of distinct results -/
def countDistinctResults (exprs : List Expression) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct results is 15 -/
theorem distinct_results_count :
  countDistinctResults allExpressions = 15 := by
  sorry

end NUMINAMATH_CALUDE_distinct_results_count_l3607_360739


namespace NUMINAMATH_CALUDE_matrix_product_abc_l3607_360780

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 3, -1; 0, 5, -4; -2, 5, 2]
def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, -3, 0; 2, 1, -4; 5, 0, 1]
def C : Matrix (Fin 3) (Fin 2) ℝ := !![1, -1; 0, 2; 1, 0]

theorem matrix_product_abc :
  A * B * C = !![(-6 : ℝ), -13; -34, 20; -4, 8] := by sorry

end NUMINAMATH_CALUDE_matrix_product_abc_l3607_360780


namespace NUMINAMATH_CALUDE_extra_charge_per_wand_l3607_360783

def total_wands_bought : ℕ := 3
def cost_per_wand : ℚ := 60
def wands_sold : ℕ := 2
def total_collected : ℚ := 130

theorem extra_charge_per_wand :
  (total_collected / wands_sold) - cost_per_wand = 5 :=
by sorry

end NUMINAMATH_CALUDE_extra_charge_per_wand_l3607_360783


namespace NUMINAMATH_CALUDE_diamond_example_l3607_360721

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := (a + b) * (a - b) + a

/-- Theorem stating that 2 ◊ (3 ◊ 4) = -10 -/
theorem diamond_example : diamond 2 (diamond 3 4) = -10 := by
  sorry

end NUMINAMATH_CALUDE_diamond_example_l3607_360721


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3607_360750

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0) (pos_f : f > 0)
  (sum_eq_8 : a + b + c + d + e + f = 8) :
  (1 / a + 9 / b + 4 / c + 25 / d + 16 / e + 49 / f) ≥ 1352 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3607_360750


namespace NUMINAMATH_CALUDE_locus_of_centers_l3607_360775

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (3 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) ↔ 
  84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l3607_360775


namespace NUMINAMATH_CALUDE_work_completion_time_l3607_360709

/-- Proves that if person A can complete a work in 40 days, and together with person B they can complete 0.25 part of the work in 6 days, then person B can complete the work alone in 60 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 40) (hab : 1 / a + 1 / b = 1 / 24) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3607_360709


namespace NUMINAMATH_CALUDE_dividend_calculation_l3607_360701

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21)
  (h2 : quotient = 14)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 301 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3607_360701


namespace NUMINAMATH_CALUDE_constant_function_l3607_360782

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)

theorem constant_function (f : ℝ → ℝ) (h1 : f 0 = 1/2) (h2 : FunctionalEq f) :
    ∀ x : ℝ, f x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_l3607_360782


namespace NUMINAMATH_CALUDE_town_average_age_l3607_360729

theorem town_average_age (k : ℕ) (h_k : k > 0) : 
  let num_children := 3 * k
  let num_adults := 2 * k
  let avg_age_children := 10
  let avg_age_adults := 40
  let total_population := num_children + num_adults
  let total_age := num_children * avg_age_children + num_adults * avg_age_adults
  (total_age : ℚ) / total_population = 22 :=
by sorry

end NUMINAMATH_CALUDE_town_average_age_l3607_360729


namespace NUMINAMATH_CALUDE_no_prime_solution_l3607_360777

def base_p_to_decimal (digits : List Nat) (p : Nat) : Nat :=
  digits.foldl (fun acc d => acc * p + d) 0

theorem no_prime_solution :
  ¬∃ p : Nat, Prime p ∧
    (base_p_to_decimal [1, 0, 3, 2] p + 
     base_p_to_decimal [5, 0, 7] p + 
     base_p_to_decimal [2, 1, 4] p + 
     base_p_to_decimal [2, 0, 5] p + 
     base_p_to_decimal [1, 0] p = 
     base_p_to_decimal [4, 2, 3] p + 
     base_p_to_decimal [5, 4, 1] p + 
     base_p_to_decimal [6, 6, 0] p) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3607_360777
