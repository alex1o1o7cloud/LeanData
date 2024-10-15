import Mathlib

namespace NUMINAMATH_CALUDE_triangle_ratio_l3681_368100

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3681_368100


namespace NUMINAMATH_CALUDE_rectangle_area_l3681_368199

/-- A rectangle with length thrice its breadth and perimeter 48 meters has an area of 108 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3681_368199


namespace NUMINAMATH_CALUDE_apple_distribution_equation_l3681_368103

def represents_apple_distribution (x : ℕ) : Prop :=
  (x - 1) % 3 = 0 ∧ (x + 2) % 4 = 0

theorem apple_distribution_equation :
  ∀ x : ℕ, represents_apple_distribution x ↔ (x - 1) / 3 = (x + 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_equation_l3681_368103


namespace NUMINAMATH_CALUDE_num_plane_line_pairs_is_48_l3681_368167

/-- A rectangular box -/
structure RectangularBox where
  -- We don't need to define the specifics of the box for this problem

/-- A line determined by two vertices of the box -/
structure BoxLine where
  box : RectangularBox
  -- We don't need to specify how the line is determined

/-- A plane containing four vertices of the box -/
structure BoxPlane where
  box : RectangularBox
  -- We don't need to specify how the plane is determined

/-- A plane-line pair in the box -/
structure PlaneLine where
  box : RectangularBox
  line : BoxLine
  plane : BoxPlane
  is_parallel : Bool -- Indicates if the line and plane are parallel

/-- The number of plane-line pairs in a rectangular box -/
def num_plane_line_pairs (box : RectangularBox) : Nat :=
  -- The actual implementation is not needed for the statement
  sorry

/-- Theorem stating that the number of plane-line pairs in a rectangular box is 48 -/
theorem num_plane_line_pairs_is_48 (box : RectangularBox) :
  num_plane_line_pairs box = 48 := by
  sorry

end NUMINAMATH_CALUDE_num_plane_line_pairs_is_48_l3681_368167


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l3681_368127

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = x / (x + 1)) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l3681_368127


namespace NUMINAMATH_CALUDE_find_m_value_l3681_368152

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l3681_368152


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3681_368159

/-- The distance between the foci of a hyperbola defined by xy = 4 is 4√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / (f₂.1 - f₁.1)^2 - 
                               (y - f₁.2)^2 / (f₂.2 - f₁.2)^2 = 1) ∧
    Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3681_368159


namespace NUMINAMATH_CALUDE_total_animals_in_community_l3681_368154

theorem total_animals_in_community (total_families : ℕ) 
  (families_with_two_dogs : ℕ) (families_with_one_dog : ℕ) 
  (h1 : total_families = 50)
  (h2 : families_with_two_dogs = 15)
  (h3 : families_with_one_dog = 20) :
  (families_with_two_dogs * 2 + families_with_one_dog * 1 + 
   (total_families - families_with_two_dogs - families_with_one_dog) * 2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_in_community_l3681_368154


namespace NUMINAMATH_CALUDE_rotate_rectangle_is_cylinder_l3681_368129

/-- A rectangle is a 2D shape with four sides and four right angles. -/
structure Rectangle where
  width : ℝ
  height : ℝ
  (positive_dimensions : width > 0 ∧ height > 0)

/-- A cylinder is a 3D shape with two circular bases connected by a curved surface. -/
structure Cylinder where
  radius : ℝ
  height : ℝ
  (positive_dimensions : radius > 0 ∧ height > 0)

/-- The result of rotating a rectangle around one of its sides. -/
def rotateRectangle (r : Rectangle) : Cylinder :=
  sorry

/-- Theorem stating that rotating a rectangle around one of its sides results in a cylinder. -/
theorem rotate_rectangle_is_cylinder (r : Rectangle) :
  ∃ (c : Cylinder), c = rotateRectangle r :=
sorry

end NUMINAMATH_CALUDE_rotate_rectangle_is_cylinder_l3681_368129


namespace NUMINAMATH_CALUDE_percentage_relation_l3681_368198

theorem percentage_relation (third_number : ℝ) (first_number : ℝ) (second_number : ℝ)
  (h1 : first_number = 0.08 * third_number)
  (h2 : second_number = 0.16 * third_number)
  (h3 : first_number = 0.5 * second_number) :
  first_number = 0.08 * third_number := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3681_368198


namespace NUMINAMATH_CALUDE_tirzah_handbags_l3681_368145

/-- The number of handbags Tirzah has -/
def num_handbags : ℕ := 24

/-- The total number of purses Tirzah has -/
def total_purses : ℕ := 26

/-- The fraction of fake purses -/
def fake_purses_fraction : ℚ := 1/2

/-- The fraction of fake handbags -/
def fake_handbags_fraction : ℚ := 1/4

/-- The total number of authentic items (purses and handbags) -/
def total_authentic : ℕ := 31

theorem tirzah_handbags :
  num_handbags = 24 ∧
  total_purses = 26 ∧
  fake_purses_fraction = 1/2 ∧
  fake_handbags_fraction = 1/4 ∧
  total_authentic = 31 ∧
  (total_purses : ℚ) * (1 - fake_purses_fraction) + (num_handbags : ℚ) * (1 - fake_handbags_fraction) = total_authentic := by
  sorry

end NUMINAMATH_CALUDE_tirzah_handbags_l3681_368145


namespace NUMINAMATH_CALUDE_nova_monthly_donation_l3681_368151

/-- Nova's monthly donation to charity -/
def monthly_donation : ℕ := 1707

/-- Nova's total annual donation to charity -/
def annual_donation : ℕ := 20484

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: Nova's monthly donation is $1,707 -/
theorem nova_monthly_donation :
  monthly_donation = annual_donation / months_in_year :=
by sorry

end NUMINAMATH_CALUDE_nova_monthly_donation_l3681_368151


namespace NUMINAMATH_CALUDE_line_equation_l3681_368135

/-- A line passing through point A(1,4) with zero sum of intercepts on coordinate axes -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on coordinate axes is zero -/
  zero_sum_intercepts : 1 - (4 - y_intercept) / slope + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3681_368135


namespace NUMINAMATH_CALUDE_ginger_water_usage_l3681_368176

/-- The amount of water Ginger drank and used in her garden --/
def water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (extra_bottles : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (extra_bottles * cups_per_bottle)

/-- Theorem stating the total amount of water Ginger used --/
theorem ginger_water_usage :
  water_used 8 2 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l3681_368176


namespace NUMINAMATH_CALUDE_samuel_travel_distance_l3681_368111

/-- The total distance Samuel needs to travel to reach the hotel -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) (remaining_distance : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + remaining_distance

/-- Theorem stating that Samuel needs to travel 600 miles to reach the hotel -/
theorem samuel_travel_distance :
  total_distance 50 80 3 4 130 = 600 := by
  sorry

end NUMINAMATH_CALUDE_samuel_travel_distance_l3681_368111


namespace NUMINAMATH_CALUDE_rectangle_area_plus_perimeter_l3681_368118

/-- Represents a rectangle with positive integer side lengths -/
structure Rectangle where
  length : ℕ+
  width : ℕ+

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length.val * r.width.val

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length.val + r.width.val)

/-- Predicate to check if a number can be expressed as the sum of area and perimeter -/
def canBeExpressedAsAreaPlusPerimeter (n : ℕ) : Prop :=
  ∃ r : Rectangle, area r + perimeter r = n

theorem rectangle_area_plus_perimeter :
  (canBeExpressedAsAreaPlusPerimeter 100) ∧
  (canBeExpressedAsAreaPlusPerimeter 104) ∧
  (canBeExpressedAsAreaPlusPerimeter 106) ∧
  (canBeExpressedAsAreaPlusPerimeter 108) ∧
  ¬(canBeExpressedAsAreaPlusPerimeter 102) := by sorry

end NUMINAMATH_CALUDE_rectangle_area_plus_perimeter_l3681_368118


namespace NUMINAMATH_CALUDE_novel_sales_ratio_l3681_368102

theorem novel_sales_ratio : 
  let total_copies : ℕ := 440000
  let paperback_copies : ℕ := 363600
  let initial_hardback : ℕ := 36000
  let hardback_copies := total_copies - paperback_copies
  let later_hardback := hardback_copies - initial_hardback
  let later_paperback := paperback_copies
  (later_paperback : ℚ) / (later_hardback : ℚ) = 9 / 1 :=
by sorry

end NUMINAMATH_CALUDE_novel_sales_ratio_l3681_368102


namespace NUMINAMATH_CALUDE_trip_length_l3681_368157

theorem trip_length (total : ℚ) 
  (h1 : (1 / 4 : ℚ) * total + 25 + (1 / 6 : ℚ) * total = total) :
  total = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_length_l3681_368157


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l3681_368183

theorem hemisphere_surface_area (r : Real) : 
  π * r^2 = 3 → 3 * π * r^2 = 9 := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l3681_368183


namespace NUMINAMATH_CALUDE_vacant_seats_l3681_368123

def total_seats : ℕ := 600
def filled_percentage : ℚ := 45 / 100

theorem vacant_seats : 
  ⌊(1 - filled_percentage) * total_seats⌋ = 330 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l3681_368123


namespace NUMINAMATH_CALUDE_random_variables_comparison_l3681_368158

-- Define the random variables ξ and η
def ξ (a b c : ℝ) : ℝ → ℝ := sorry
def η (a b c : ℝ) : ℝ → ℝ := sorry

-- Define the probability measure
def P : Set ℝ → ℝ := sorry

-- Define expected value
def E (X : ℝ → ℝ) : ℝ := sorry

-- Define variance
def D (X : ℝ → ℝ) : ℝ := sorry

theorem random_variables_comparison (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (E (ξ a b c) = E (η a b c)) ∧ (D (ξ a b c) > D (η a b c)) := by
  sorry

end NUMINAMATH_CALUDE_random_variables_comparison_l3681_368158


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3681_368173

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = (11.11111111111111 / 100) * x) → 
  P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3681_368173


namespace NUMINAMATH_CALUDE_sum_of_bases_equals_1188_l3681_368193

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1188 :
  base8ToBase10 537 + base13ToBase10 (4 * 13^2 + C * 13 + 5) = 1188 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_equals_1188_l3681_368193


namespace NUMINAMATH_CALUDE_function_inequality_l3681_368179

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3681_368179


namespace NUMINAMATH_CALUDE_solution_sum_of_squares_l3681_368174

theorem solution_sum_of_squares (x y : ℝ) : 
  x * y = 8 → 
  x^2 * y + x * y^2 + 2*x + 2*y = 108 → 
  x^2 + y^2 = 100.64 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_of_squares_l3681_368174


namespace NUMINAMATH_CALUDE_johns_equation_l3681_368156

theorem johns_equation (a b c d e : ℤ) : 
  a = 2 → b = 3 → c = 4 → d = 5 →
  (a - b - c * d + e = a - (b - (c * (d - e)))) →
  e = 8 := by
sorry

end NUMINAMATH_CALUDE_johns_equation_l3681_368156


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3681_368172

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_origin :
  let p : ℝ × ℝ := (0, 0)  -- The origin point
  let m : ℝ := f' p.1      -- The slope of the tangent line at the origin
  ∀ x y : ℝ, y = m * (x - p.1) + f p.1 → y = 0 :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_at_origin_l3681_368172


namespace NUMINAMATH_CALUDE_point_difference_on_line_l3681_368105

/-- Given two points (m, n) and (m + v, n + 18) on the line x = (y / 6) - (2 / 5),
    prove that v = 3 -/
theorem point_difference_on_line (m n : ℝ) :
  (m = n / 6 - 2 / 5) →
  (m + 3 = (n + 18) / 6 - 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_point_difference_on_line_l3681_368105


namespace NUMINAMATH_CALUDE_greta_hourly_wage_l3681_368148

/-- Greta's hourly wage in dollars -/
def greta_wage : ℝ := 12

/-- The number of hours Greta worked -/
def greta_hours : ℕ := 40

/-- Lisa's hourly wage in dollars -/
def lisa_wage : ℝ := 15

/-- The number of hours Lisa would need to work to equal Greta's earnings -/
def lisa_hours : ℕ := 32

theorem greta_hourly_wage :
  greta_wage * greta_hours = lisa_wage * lisa_hours := by sorry

end NUMINAMATH_CALUDE_greta_hourly_wage_l3681_368148


namespace NUMINAMATH_CALUDE_water_percentage_in_first_liquid_l3681_368120

/-- Given two liquids in a glass, prove the percentage of water in the first liquid --/
theorem water_percentage_in_first_liquid 
  (water_percent_second : ℝ) 
  (parts_first : ℝ) 
  (parts_second : ℝ) 
  (water_percent_mixture : ℝ) : 
  water_percent_second = 0.35 →
  parts_first = 10 →
  parts_second = 4 →
  water_percent_mixture = 0.24285714285714285 →
  ∃ (water_percent_first : ℝ), 
    water_percent_first * parts_first + water_percent_second * parts_second = 
    water_percent_mixture * (parts_first + parts_second) ∧
    water_percent_first = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_first_liquid_l3681_368120


namespace NUMINAMATH_CALUDE_zero_not_read_in_4006530_l3681_368125

/-- Rule for reading a number -/
def readNumber (n : ℕ) : Bool :=
  sorry

/-- Checks if zero is read out in the number -/
def isZeroReadOut (n : ℕ) : Bool :=
  sorry

theorem zero_not_read_in_4006530 :
  ¬(isZeroReadOut 4006530) ∧ 
  (isZeroReadOut 4650003) ∧ 
  (isZeroReadOut 4650300) ∧ 
  (isZeroReadOut 4006053) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_read_in_4006530_l3681_368125


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_exists_l3681_368121

theorem diophantine_equation_solution_exists :
  ∃ (x y z : ℕ+), 
    (z = Nat.gcd x y) ∧ 
    (x + y^2 + z^3 = x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_exists_l3681_368121


namespace NUMINAMATH_CALUDE_square_of_negative_integer_is_positive_l3681_368112

theorem square_of_negative_integer_is_positive (P : Int) (h : P < 0) : P^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_integer_is_positive_l3681_368112


namespace NUMINAMATH_CALUDE_derivative_f_at_half_l3681_368114

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem derivative_f_at_half : 
  deriv f (1/2) = -2 :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_half_l3681_368114


namespace NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l3681_368191

/-- A positive integer n is lovely if there exists a positive integer k and 
    positive integers d₁, d₂, ..., dₖ such that n = d₁d₂...dₖ and d_i² | n+d_i for all i ∈ {1, ..., k}. -/
def IsLovely (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∃ d : Fin k → ℕ+, 
    (n = (Finset.univ.prod (λ i => d i))) ∧ 
    (∀ i : Fin k, (d i)^2 ∣ (n + d i))

/-- There are infinitely many lovely numbers. -/
theorem infinitely_many_lovely_numbers : ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ IsLovely n :=
sorry

/-- There does not exist a lovely number greater than 1 which is a square of an integer. -/
theorem no_lovely_square_greater_than_one : ¬∃ n : ℕ+, n > 1 ∧ ∃ m : ℕ+, n = m^2 ∧ IsLovely n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l3681_368191


namespace NUMINAMATH_CALUDE_triangle_right_angle_l3681_368196

theorem triangle_right_angle (a b : ℝ) (A B : Real) (h : a + b = a / Real.tan A + b / Real.tan B) :
  A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l3681_368196


namespace NUMINAMATH_CALUDE_largest_decimal_l3681_368109

theorem largest_decimal (a b c d e : ℚ) 
  (ha : a = 0.997) 
  (hb : b = 0.9969) 
  (hc : c = 0.99699) 
  (hd : d = 0.9699) 
  (he : e = 0.999) : 
  e = max a (max b (max c (max d e))) := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l3681_368109


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l3681_368162

/-- Triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (t : Triangle) where
  area : ℝ → ℝ

/-- The theorem stating the value of β in the area formula of the inscribed rectangle -/
theorem inscribed_rectangle_area_coefficient (t : Triangle) 
  (r : InscribedRectangle t) : 
  t.a = 12 → t.b = 25 → t.c = 17 → 
  (∃ α β : ℝ, ∀ ω, r.area ω = α * ω - β * ω^2) → 
  (∃ β : ℝ, β = 36 / 125) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l3681_368162


namespace NUMINAMATH_CALUDE_negative_sum_l3681_368101

theorem negative_sum (u v w : ℝ) 
  (hu : -1 < u ∧ u < 0) 
  (hv : 0 < v ∧ v < 1) 
  (hw : -2 < w ∧ w < -1) : 
  v + w < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3681_368101


namespace NUMINAMATH_CALUDE_yellow_face_probability_l3681_368144

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l3681_368144


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3681_368147

/-- A triangle inscribed in a circle with given properties -/
structure InscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The ratio of the triangle's sides -/
  side_ratio : Fin 3 → ℝ
  /-- The side ratio corresponds to a 3:4:5 triangle -/
  ratio_valid : side_ratio = ![3, 4, 5]
  /-- The radius of the circle is 5 -/
  radius_is_5 : radius = 5

/-- The area of an inscribed triangle with the given properties is 24 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : Real.sqrt (
  (t.side_ratio 0 * t.side_ratio 1 * t.side_ratio 2 * (t.side_ratio 0 + t.side_ratio 1 + t.side_ratio 2)) /
  ((t.side_ratio 0 + t.side_ratio 1) * (t.side_ratio 1 + t.side_ratio 2) * (t.side_ratio 2 + t.side_ratio 0))
) * t.radius ^ 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3681_368147


namespace NUMINAMATH_CALUDE_scientific_notation_of_8200000_l3681_368197

theorem scientific_notation_of_8200000 :
  (8200000 : ℝ) = 8.2 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8200000_l3681_368197


namespace NUMINAMATH_CALUDE_article_pricing_l3681_368180

/-- The value of x when the cost price of 20 articles equals the selling price of x articles with a 25% profit -/
theorem article_pricing (C : ℝ) (x : ℝ) (h1 : C > 0) :
  20 * C = x * (C * (1 + 0.25)) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_l3681_368180


namespace NUMINAMATH_CALUDE_platform_length_problem_solution_l3681_368141

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- The length of the platform is 208.8 meters --/
theorem problem_solution : 
  (platform_length 180 70 20) = 208.8 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_problem_solution_l3681_368141


namespace NUMINAMATH_CALUDE_electronic_product_pricing_l3681_368189

/-- The marked price of an electronic product -/
def marked_price : ℝ := 28

/-- The cost price of the electronic product -/
def cost_price : ℝ := 21

/-- The selling price ratio (90% of marked price) -/
def selling_price_ratio : ℝ := 0.9

/-- The profit ratio (20%) -/
def profit_ratio : ℝ := 0.2

theorem electronic_product_pricing :
  selling_price_ratio * marked_price - cost_price = profit_ratio * cost_price :=
by sorry

end NUMINAMATH_CALUDE_electronic_product_pricing_l3681_368189


namespace NUMINAMATH_CALUDE_smallest_inverse_undefined_l3681_368110

theorem smallest_inverse_undefined (a : ℕ) : a = 6 ↔ 
  a > 0 ∧ 
  (∀ k < a, k > 0 → (Nat.gcd k 72 = 1 ∨ Nat.gcd k 90 = 1)) ∧
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 90 > 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_undefined_l3681_368110


namespace NUMINAMATH_CALUDE_identity_proof_l3681_368108

theorem identity_proof (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l3681_368108


namespace NUMINAMATH_CALUDE_remainder_8_pow_1996_mod_5_l3681_368160

theorem remainder_8_pow_1996_mod_5 : 8^1996 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8_pow_1996_mod_5_l3681_368160


namespace NUMINAMATH_CALUDE_circle_center_correct_l3681_368164

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 6*y - 16 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, -3)

/-- Theorem: The center of the circle with equation x^2 + 4x + y^2 + 6y - 16 = 0 is (-2, -3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3681_368164


namespace NUMINAMATH_CALUDE_log_8641_between_consecutive_integers_l3681_368139

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_8641_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ 
  (log10 1000 : ℝ) = 3 ∧
  (log10 10000 : ℝ) = 4 ∧
  1000 < 8641 ∧ 8641 < 10000 ∧
  Monotone log10 ∧
  (c : ℝ) < log10 8641 ∧ log10 8641 < (d : ℝ) ∧
  c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_8641_between_consecutive_integers_l3681_368139


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l3681_368186

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (rate : ℝ) (time : ℝ) : 
  rate = 3 / 10 → time = 40 → rate * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_bike_ride_l3681_368186


namespace NUMINAMATH_CALUDE_tan_sum_identity_l3681_368131

theorem tan_sum_identity (x : ℝ) : 
  Real.tan (18 * π / 180 - x) * Real.tan (12 * π / 180 + x) + 
  Real.sqrt 3 * (Real.tan (18 * π / 180 - x) + Real.tan (12 * π / 180 + x)) = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l3681_368131


namespace NUMINAMATH_CALUDE_male_students_count_l3681_368195

def scienceGroup (x : ℕ) : Prop :=
  ∃ (total : ℕ), total = x + 2 ∧ 
  (Nat.choose x 2) * (Nat.choose 2 1) = 20

theorem male_students_count :
  ∀ x : ℕ, scienceGroup x → x = 5 := by sorry

end NUMINAMATH_CALUDE_male_students_count_l3681_368195


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3681_368163

theorem smallest_m_for_integral_solutions : 
  let f (m : ℕ) (x : ℤ) := 12 * x^2 - m * x + 360
  ∃ (m₀ : ℕ), m₀ > 0 ∧ 
    (∃ (x : ℤ), f m₀ x = 0) ∧ 
    (∀ (m : ℕ), 0 < m ∧ m < m₀ → ∀ (x : ℤ), f m x ≠ 0) ∧
    m₀ = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3681_368163


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3681_368192

-- Define p and q as predicates on real numbers x and y
def p (x y : ℝ) : Prop := x + y ≠ 4
def q (x y : ℝ) : Prop := x ≠ 1 ∨ y ≠ 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3681_368192


namespace NUMINAMATH_CALUDE_arithmetic_problem_l3681_368128

theorem arithmetic_problem : (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l3681_368128


namespace NUMINAMATH_CALUDE_mark_reading_time_l3681_368178

/-- Mark's daily reading time in hours -/
def daily_reading_time : ℕ := 2

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mark's planned increase in weekly reading time in hours -/
def weekly_increase : ℕ := 4

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading_time : ℕ := daily_reading_time * days_per_week + weekly_increase

theorem mark_reading_time :
  desired_weekly_reading_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_mark_reading_time_l3681_368178


namespace NUMINAMATH_CALUDE_percent_juniors_in_sports_l3681_368136

theorem percent_juniors_in_sports (total_students : ℕ) (percent_juniors : ℚ) (juniors_in_sports : ℕ) :
  total_students = 500 →
  percent_juniors = 40 / 100 →
  juniors_in_sports = 140 →
  (juniors_in_sports : ℚ) / (percent_juniors * total_students) * 100 = 70 := by
  sorry


end NUMINAMATH_CALUDE_percent_juniors_in_sports_l3681_368136


namespace NUMINAMATH_CALUDE_disk_arrangement_area_sum_l3681_368185

theorem disk_arrangement_area_sum :
  ∀ (n : ℕ) (r : ℝ) (disk_radius : ℝ),
    n = 15 →
    r = 1 →
    disk_radius = 2 - Real.sqrt 3 →
    (↑n * π * disk_radius^2 : ℝ) = π * (105 - 60 * Real.sqrt 3) ∧
    105 + 60 + 3 = 168 := by
  sorry

end NUMINAMATH_CALUDE_disk_arrangement_area_sum_l3681_368185


namespace NUMINAMATH_CALUDE_crayons_left_l3681_368169

theorem crayons_left (initial : ℝ) (taken : ℝ) (left : ℝ) : 
  initial = 7.5 → taken = 2.25 → left = initial - taken → left = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l3681_368169


namespace NUMINAMATH_CALUDE_merchant_savings_l3681_368142

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.1, 0.3, 0.2]
def option2_discounts : List ℝ := [0.25, 0.15, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem merchant_savings :
  apply_successive_discounts initial_order option2_discounts -
  apply_successive_discounts initial_order option1_discounts = 1524.38 := by
  sorry

end NUMINAMATH_CALUDE_merchant_savings_l3681_368142


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l3681_368146

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →
  C = 60 * π / 180 →
  c = 1 →
  b = Real.sqrt 6 / 3 →
  b < a ∧ b < c :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l3681_368146


namespace NUMINAMATH_CALUDE_city_a_sand_amount_l3681_368175

/-- The amount of sand received by City A, given the total amount and amounts received by other cities -/
theorem city_a_sand_amount (total sand_b sand_c sand_d : ℝ) (h1 : total = 95) 
  (h2 : sand_b = 26) (h3 : sand_c = 24.5) (h4 : sand_d = 28) : 
  total - (sand_b + sand_c + sand_d) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_city_a_sand_amount_l3681_368175


namespace NUMINAMATH_CALUDE_previous_salary_calculation_l3681_368165

-- Define the salary increase rate
def salary_increase_rate : ℝ := 1.05

-- Define the new salary
def new_salary : ℝ := 2100

-- Theorem statement
theorem previous_salary_calculation :
  ∃ (previous_salary : ℝ),
    salary_increase_rate * previous_salary = new_salary ∧
    previous_salary = 2000 := by
  sorry

end NUMINAMATH_CALUDE_previous_salary_calculation_l3681_368165


namespace NUMINAMATH_CALUDE_product_real_part_l3681_368107

/-- Given two complex numbers a and b with magnitudes 3 and 5 respectively,
    prove that the positive real part of their product is 6√6. -/
theorem product_real_part (a b : ℂ) (ha : Complex.abs a = 3) (hb : Complex.abs b = 5) :
  (a * b).re = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_product_real_part_l3681_368107


namespace NUMINAMATH_CALUDE_prime_pairs_divisibility_l3681_368133

theorem prime_pairs_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (6 * p * q ∣ p^3 + q^2 + 38) → 
    ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 13)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_divisibility_l3681_368133


namespace NUMINAMATH_CALUDE_infinite_primes_solution_l3681_368113

theorem infinite_primes_solution (f : ℕ → ℕ) (k : ℕ) 
  (h_inj : Function.Injective f) 
  (h_bound : ∀ n, f n ≤ n^k) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    (∀ q ∈ S, Nat.Prime q ∧ 
      ∃ p, Nat.Prime p ∧ f p ≡ 0 [MOD q]) :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_solution_l3681_368113


namespace NUMINAMATH_CALUDE_trapezoid_inner_quadrilateral_area_l3681_368138

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- A trapezoid is a quadrilateral with two parallel sides -/
structure Trapezoid extends Quadrilateral :=
  (parallel : (A.y - B.y) / (A.x - B.x) = (D.y - C.y) / (D.x - C.x))

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point lies on a line segment -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (P Q R S : Point) : Point := sorry

/-- Theorem: Area of inner quadrilateral is at most 1/4 of trapezoid area -/
theorem trapezoid_inner_quadrilateral_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (F : Point) 
  (H : Point) 
  (G : Point)
  (hE : onSegment ABCD.A ABCD.B E)
  (hF : onSegment ABCD.C ABCD.D F)
  (hH : H = intersectionPoint ABCD.C E ABCD.B F)
  (hG : G = intersectionPoint E ABCD.D ABCD.A F) :
  area ⟨E, H, F, G⟩ ≤ (1/4 : ℝ) * area ABCD.toQuadrilateral := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_inner_quadrilateral_area_l3681_368138


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l3681_368194

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l3681_368194


namespace NUMINAMATH_CALUDE_polynomial_sum_l3681_368122

theorem polynomial_sum (f g : ℝ → ℝ) :
  (∀ x, f x + g x = 3 * x - x^2) →
  (∀ x, f x = x^2 - 4 * x + 3) →
  (∀ x, g x = -2 * x^2 + 7 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3681_368122


namespace NUMINAMATH_CALUDE_analogous_property_is_about_surfaces_l3681_368149

/-- Represents a geometric property in plane geometry -/
structure PlaneProperty where
  description : String

/-- Represents a geometric property in solid geometry -/
structure SolidProperty where
  description : String

/-- Represents the analogy between plane and solid geometry properties -/
def analogy (plane : PlaneProperty) : SolidProperty :=
  sorry

/-- The plane geometry property about equilateral triangles -/
def triangle_property : PlaneProperty :=
  { description := "The sum of distances from any point inside an equilateral triangle to its three sides is constant" }

/-- Theorem stating that the analogous property in solid geometry is about surfaces -/
theorem analogous_property_is_about_surfaces :
  ∃ (surface_prop : SolidProperty),
    (analogy triangle_property).description = "A property about surfaces" :=
  sorry

end NUMINAMATH_CALUDE_analogous_property_is_about_surfaces_l3681_368149


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3681_368150

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the condition c = 2a cos B
def condition (t : Triangle) : Prop :=
  t.c = 2 * t.a * Real.cos t.angleB

-- State the theorem
theorem isosceles_triangle (t : Triangle) (h : condition t) : t.a = t.b := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3681_368150


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l3681_368190

def f (t : ℕ) : ℚ :=
  if t < 15 then (1/3) * t + 8 else -(1/3) * t + 18

def g (t : ℕ) : ℚ := -t + 30

def W (t : ℕ) : ℚ := (f t) * (g t)

theorem max_daily_sales_revenue (t : ℕ) (h : 0 < t ∧ t ≤ 30) : 
  W t ≤ 243 ∧ ∃ t₀ : ℕ, 0 < t₀ ∧ t₀ ≤ 30 ∧ W t₀ = 243 :=
sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l3681_368190


namespace NUMINAMATH_CALUDE_madeline_utilities_l3681_368184

/-- Calculates the amount left for utilities given expenses and income --/
def amount_for_utilities (rent groceries medical emergency hourly_wage hours : ℕ) : ℕ :=
  hourly_wage * hours - (rent + groceries + medical + emergency)

/-- Proves that Madeline's amount left for utilities is $70 --/
theorem madeline_utilities : amount_for_utilities 1200 400 200 200 15 138 = 70 := by
  sorry

end NUMINAMATH_CALUDE_madeline_utilities_l3681_368184


namespace NUMINAMATH_CALUDE_total_pets_is_45_l3681_368177

/-- The total number of pets given the specified conditions -/
def total_pets : ℕ :=
  let taylor_cats := 4
  let friends_with_double_pets := 3
  let friend1_dogs := 3
  let friend1_birds := 1
  let friend2_dogs := 5
  let friend2_cats := 2
  let friend3_reptiles := 2
  let friend3_birds := 3
  let friend3_cats := 1

  let total_cats := taylor_cats + friends_with_double_pets * (2 * taylor_cats) + friend2_cats + friend3_cats
  let total_dogs := friend1_dogs + friend2_dogs
  let total_birds := friend1_birds + friend3_birds
  let total_reptiles := friend3_reptiles

  total_cats + total_dogs + total_birds + total_reptiles

theorem total_pets_is_45 : total_pets = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_45_l3681_368177


namespace NUMINAMATH_CALUDE_profit_maximization_l3681_368116

noncomputable def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then
    -5 * x^2 + 420 * x - 3
  else if 30 < x ∧ x ≤ 110 then
    -2 * x - 20000 / (x + 10) + 597
  else
    0

theorem profit_maximization :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 110 ∧
  g x = 9320 ∧
  ∀ (y : ℝ), 0 < y ∧ y ≤ 110 → g y ≤ g x :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l3681_368116


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l3681_368171

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Represents the ratio condition AC:CB = 5:2 -/
def ratio_condition (a b c : Point) : Prop :=
  (c.x - a.x) / (b.x - c.x) = 5 / 2

theorem parabola_chord_intersection :
  ∀ (a b c : Point),
    parabola a →
    parabola b →
    c.x = 0 →
    c.y = 20 →
    ratio_condition a b c →
    ((a.x = -5 * Real.sqrt 2 ∧ b.x = 2 * Real.sqrt 2) ∨
     (a.x = 5 * Real.sqrt 2 ∧ b.x = -2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l3681_368171


namespace NUMINAMATH_CALUDE_mabels_daisies_l3681_368104

theorem mabels_daisies (petals_per_daisy : ℕ) (remaining_petals : ℕ) (daisies_given_away : ℕ) : 
  petals_per_daisy = 8 →
  daisies_given_away = 2 →
  remaining_petals = 24 →
  ∃ (initial_daisies : ℕ), 
    initial_daisies * petals_per_daisy = 
      remaining_petals + daisies_given_away * petals_per_daisy ∧
    initial_daisies = 5 :=
by sorry

end NUMINAMATH_CALUDE_mabels_daisies_l3681_368104


namespace NUMINAMATH_CALUDE_floor_times_self_eq_120_l3681_368140

theorem floor_times_self_eq_120 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 120 ∧ x = 120 / 11 :=
by sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_120_l3681_368140


namespace NUMINAMATH_CALUDE_min_distance_to_circle_l3681_368119

theorem min_distance_to_circle (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 4 = 0 →
  ∃ (min : ℝ), min = Real.sqrt 13 - 3 ∧
    ∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 →
      Real.sqrt (a^2 + b^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_circle_l3681_368119


namespace NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l3681_368182

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x ≥ 1 ∨ x < 0}

-- Define the solution set for the second inequality based on the value of a
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x > -1}
  else if a > 0 then
    {x | -1 < x ∧ x < 1/a}
  else if a < -1 then
    {x | x < -1 ∨ x > 1/a}
  else if a = -1 then
    {x | x ≠ -1}
  else
    {x | x < 1/a ∨ x > -1}

-- Theorem for the first inequality
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solutionSet1 ↔ (x - 1) / x ≥ 0 ∧ x ≠ 0 :=
sorry

-- Theorem for the second inequality
theorem solution_set2_correct :
  ∀ a x : ℝ, x ∈ solutionSet2 a ↔ a * x^2 + (a - 1) * x - 1 < 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l3681_368182


namespace NUMINAMATH_CALUDE_andy_initial_minks_l3681_368143

/-- The number of mink skins required to make one coat -/
def skins_per_coat : ℕ := 15

/-- The number of babies each mink has -/
def babies_per_mink : ℕ := 6

/-- The fraction of minks set free by activists -/
def fraction_set_free : ℚ := 1/2

/-- The number of coats Andy can make -/
def coats_made : ℕ := 7

/-- Theorem stating that given the conditions, Andy must have bought 30 minks initially -/
theorem andy_initial_minks :
  ∀ x : ℕ,
  (x + x * babies_per_mink) * (1 - fraction_set_free) = coats_made * skins_per_coat →
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_andy_initial_minks_l3681_368143


namespace NUMINAMATH_CALUDE_bottles_left_on_shelf_l3681_368130

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_purchase : ℕ) (harry_purchase : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_purchase = 5)
  (h3 : harry_purchase = 6) :
  initial_bottles - (jason_purchase + harry_purchase) = 24 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_on_shelf_l3681_368130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3681_368155

/-- Three real numbers form an arithmetic sequence if the middle term is the arithmetic mean of the other two terms. -/
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

/-- If 2, m, 6 form an arithmetic sequence, then m = 4 -/
theorem arithmetic_sequence_middle_term : 
  ∀ m : ℝ, is_arithmetic_sequence 2 m 6 → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3681_368155


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_l3681_368170

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two non-adjacent identical digits -/
def count_two_same_not_adjacent : ℕ := 81

/-- The count of three-digit numbers with identical first and last digits -/
def count_first_last_same : ℕ := 81

/-- Theorem stating the count of valid three-digit numbers -/
theorem valid_three_digit_numbers :
  valid_count = total_three_digit_numbers - count_two_same_not_adjacent - count_first_last_same :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_l3681_368170


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3681_368187

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3681_368187


namespace NUMINAMATH_CALUDE_calendar_reuse_2052_l3681_368188

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

def calendar_repeats (year1 year2 : ℕ) : Prop :=
  is_leap_year year1 ∧ is_leap_year year2 ∧ (year2 - year1) % 28 = 0

theorem calendar_reuse_2052 :
  ∀ y : ℕ, y > 1912 → y < 2052 → ¬(calendar_repeats y 2052) →
  calendar_repeats 1912 2052 ∧ is_leap_year 1912 ∧ is_leap_year 2052 :=
sorry

end NUMINAMATH_CALUDE_calendar_reuse_2052_l3681_368188


namespace NUMINAMATH_CALUDE_find_divisor_l3681_368124

theorem find_divisor (n m d : ℕ) (h1 : n - 7 = m) (h2 : m % d = 0) (h3 : ∀ k < 7, (n - k) % d ≠ 0) : d = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3681_368124


namespace NUMINAMATH_CALUDE_constant_point_on_graph_unique_constant_point_l3681_368168

/-- The quadratic function f(x) that passes through a constant point for any real m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The constant point that lies on the graph of f(x) for all real m -/
def constant_point : ℝ × ℝ := (2, 13)

/-- Theorem stating that the constant_point lies on the graph of f(x) for all real m -/
theorem constant_point_on_graph :
  ∀ m : ℝ, f m (constant_point.1) = constant_point.2 :=
by sorry

/-- Theorem stating that constant_point is the unique point satisfying the condition -/
theorem unique_constant_point :
  ∀ p : ℝ × ℝ, (∀ m : ℝ, f m p.1 = p.2) → p = constant_point :=
by sorry

end NUMINAMATH_CALUDE_constant_point_on_graph_unique_constant_point_l3681_368168


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l3681_368115

theorem triangle_area_ratio (BD DC : ℝ) (area_ABD : ℝ) (area_ADC : ℝ) :
  BD / DC = 5 / 2 →
  area_ABD = 40 →
  area_ADC = area_ABD * (DC / BD) →
  area_ADC = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l3681_368115


namespace NUMINAMATH_CALUDE_range_of_q_l3681_368134

def q (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

theorem range_of_q :
  Set.range q = Set.Icc 0 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_q_l3681_368134


namespace NUMINAMATH_CALUDE_area_of_figure_l3681_368161

/-- Given a figure F in the plane of one face of a dihedral angle,
    S is the area of its orthogonal projection onto the other face,
    Q is the area of its orthogonal projection onto the bisector plane.
    This theorem proves that the area T of figure F is equal to (1/2)(√(S² + 8Q²) - S). -/
theorem area_of_figure (S Q : ℝ) (hS : S > 0) (hQ : Q > 0) :
  ∃ T : ℝ, T = (1/2) * (Real.sqrt (S^2 + 8*Q^2) - S) ∧ T > 0 := by
sorry

end NUMINAMATH_CALUDE_area_of_figure_l3681_368161


namespace NUMINAMATH_CALUDE_gift_shop_pricing_l3681_368137

theorem gift_shop_pricing (p : ℝ) (hp : p > 0) : 
  0.7 * (1.1 * p) = 0.77 * p := by sorry

end NUMINAMATH_CALUDE_gift_shop_pricing_l3681_368137


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3681_368126

theorem no_positive_integer_solution : 
  ¬ ∃ (n k : ℕ+), (5 + 3 * Real.sqrt 2) ^ n.val = (3 + 5 * Real.sqrt 2) ^ k.val := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3681_368126


namespace NUMINAMATH_CALUDE_amphibian_count_l3681_368153

/-- The total number of amphibians observed in the pond -/
def total_amphibians (frogs salamanders tadpoles newts : ℕ) : ℕ :=
  frogs + salamanders + tadpoles + newts

/-- Theorem stating that the total number of amphibians is 42 -/
theorem amphibian_count : 
  total_amphibians 7 4 30 1 = 42 := by sorry

end NUMINAMATH_CALUDE_amphibian_count_l3681_368153


namespace NUMINAMATH_CALUDE_not_tileable_rectangles_l3681_368166

/-- A domino is a 1x2 rectangle -/
structure Domino :=
  (width : Nat := 2)
  (height : Nat := 1)

/-- A rectangle with given width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Predicate to check if a rectangle is (1,2)-tileable -/
def is_tileable (r : Rectangle) : Prop := sorry

/-- Theorem stating that 1xk and 2xn (where 4 ∤ n) rectangles are not (1,2)-tileable -/
theorem not_tileable_rectangles :
  ∀ (k n : Nat), 
    (¬ is_tileable ⟨1, k⟩) ∧ 
    ((¬ (4 ∣ n)) → ¬ is_tileable ⟨2, n⟩) :=
by sorry

end NUMINAMATH_CALUDE_not_tileable_rectangles_l3681_368166


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_and_side_range_l3681_368181

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  top : ℝ
  bottom : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.top + t.bottom + t.side1 + t.side2

/-- Theorem stating the relationship between perimeter and side length,
    and the valid range for the variable side length -/
theorem trapezoid_perimeter_and_side_range (x : ℝ) :
  let t := Trapezoid.mk 4 7 12 x
  (perimeter t = x + 23) ∧ (9 < x ∧ x < 15) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_and_side_range_l3681_368181


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3681_368106

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l3681_368106


namespace NUMINAMATH_CALUDE_binary_110110011_to_octal_l3681_368117

def binary_to_octal (b : Nat) : Nat :=
  sorry

theorem binary_110110011_to_octal :
  binary_to_octal 110110011 = 163 := by
  sorry

end NUMINAMATH_CALUDE_binary_110110011_to_octal_l3681_368117


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3681_368132

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 7*I
  z₁ / z₂ = (29:ℝ)/53 - (31:ℝ)/53 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3681_368132
