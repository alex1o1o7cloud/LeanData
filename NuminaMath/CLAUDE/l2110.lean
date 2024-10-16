import Mathlib

namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_l2110_211022

theorem historical_fiction_new_releases 
  (total_inventory : ℝ)
  (historical_fiction_ratio : ℝ)
  (historical_fiction_new_release_ratio : ℝ)
  (other_new_release_ratio : ℝ)
  (h1 : historical_fiction_ratio = 0.3)
  (h2 : historical_fiction_new_release_ratio = 0.3)
  (h3 : other_new_release_ratio = 0.4)
  (h4 : total_inventory > 0) :
  let historical_fiction := total_inventory * historical_fiction_ratio
  let other_books := total_inventory * (1 - historical_fiction_ratio)
  let historical_fiction_new_releases := historical_fiction * historical_fiction_new_release_ratio
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  historical_fiction_new_releases / total_new_releases = 9 / 37 := by
    sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_l2110_211022


namespace NUMINAMATH_CALUDE_permutation_combination_sum_l2110_211070

theorem permutation_combination_sum (m n : ℕ) : 
  (Nat.factorial n) / (Nat.factorial (n - m)) = 272 →
  (Nat.factorial n) / ((Nat.factorial (n - m)) * (Nat.factorial m)) = 136 →
  m + n = 19 := by
sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l2110_211070


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2110_211068

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 4 ∧ b = 8 ∧ c = 8 → -- Two sides are 8, one side is 4
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- Triangle inequality
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2110_211068


namespace NUMINAMATH_CALUDE_three_lines_triangle_l2110_211014

/-- A line in the 2D plane represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y = l1.c ∧
             l2.a * x + l2.b * y = l2.c ∧
             l3.a * x + l3.b * y = l3.c

/-- The set of possible values for m -/
def possible_m_values : Set ℝ :=
  {m : ℝ | m = 4 ∨ m = -1/6 ∨ m = 1 ∨ m = -2/3}

theorem three_lines_triangle (m : ℝ) :
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3*m, 4⟩
  (parallel l1 l2 ∨ parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m ∈ possible_m_values :=
sorry

end NUMINAMATH_CALUDE_three_lines_triangle_l2110_211014


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2110_211013

/-- Perimeter of triangle ABF₂ in an ellipse with given parameters -/
theorem ellipse_triangle_perimeter (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b = 4)
    (h4 : c / a = 3 / 5) (h5 : a^2 = b^2 + c^2) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  ∀ A B : ℝ × ℝ, A ∈ ellipse → B ∈ ellipse →
    ∃ t : ℝ, A = F₁ + t • (1, 0) ∧ B = F₁ + t • (1, 0) →
      dist A B + dist A F₂ + dist B F₂ = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2110_211013


namespace NUMINAMATH_CALUDE_equality_implications_l2110_211021

theorem equality_implications (a b x y : ℝ) (h : a = b) : 
  (a - 3 = b - 3) ∧ 
  (3 * a = 3 * b) ∧ 
  ((a + 3) / 4 = (b + 3) / 4) ∧
  (∃ x y, a * x ≠ b * y) := by
sorry

end NUMINAMATH_CALUDE_equality_implications_l2110_211021


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2110_211051

/-- The curve defined by r = 8 tan(θ)cos(θ) in polar coordinates is a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∃ (x₀ y₀ R : ℝ), ∀ (θ : ℝ) (r : ℝ),
    r = 8 * Real.tan θ * Real.cos θ →
    (r * Real.cos θ - x₀)^2 + (r * Real.sin θ - y₀)^2 = R^2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2110_211051


namespace NUMINAMATH_CALUDE_right_triangle_m_values_l2110_211090

/-- A right triangle in a 2D Cartesian coordinate system -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
             (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 ∨
             (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

/-- The theorem to be proved -/
theorem right_triangle_m_values (t : RightTriangle) 
    (h1 : t.B.1 - t.A.1 = 1 ∧ t.B.2 - t.A.2 = 1)
    (h2 : t.C.1 - t.A.1 = 2 ∧ ∃ m : ℝ, t.C.2 - t.A.2 = m) :
  ∃ m : ℝ, (t.C.2 - t.A.2 = m ∧ (m = -2 ∨ m = 0)) := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_m_values_l2110_211090


namespace NUMINAMATH_CALUDE_min_n_for_integer_sqrt_l2110_211053

theorem min_n_for_integer_sqrt (n : ℕ+) : 
  (∃ k : ℕ, k^2 = 51 + n) → (∀ m : ℕ+, m < n → ¬∃ k : ℕ, k^2 = 51 + m) → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_integer_sqrt_l2110_211053


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l2110_211041

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the area of a specific triangle -/
theorem triangle_area_is_one (t : Triangle) 
  (h1 : (Real.cos t.B / t.b) + (Real.cos t.C / t.c) = (Real.sin t.A / (2 * Real.sin t.C)))
  (h2 : Real.sqrt 3 * t.b * Real.cos t.C = (2 * t.a - Real.sqrt 3 * t.c) * Real.cos t.B)
  (h3 : ∃ (r : ℝ), Real.sin t.A = r * Real.sin t.B ∧ Real.sin t.B = r * Real.sin t.C) :
  (1/2) * t.a * t.c * Real.sin t.B = 1 := by
  sorry

#check triangle_area_is_one

end NUMINAMATH_CALUDE_triangle_area_is_one_l2110_211041


namespace NUMINAMATH_CALUDE_ice_cream_sales_for_games_l2110_211015

theorem ice_cream_sales_for_games (game_cost : ℕ) (ice_cream_price : ℕ) : 
  game_cost = 60 → ice_cream_price = 5 → (2 * game_cost) / ice_cream_price = 24 := by
  sorry

#check ice_cream_sales_for_games

end NUMINAMATH_CALUDE_ice_cream_sales_for_games_l2110_211015


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2110_211097

/-- A rectangular solid with prime edge lengths and volume 429 has surface area 430. -/
theorem rectangular_solid_surface_area :
  ∀ l w h : ℕ,
  Prime l → Prime w → Prime h →
  l * w * h = 429 →
  2 * (l * w + w * h + h * l) = 430 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2110_211097


namespace NUMINAMATH_CALUDE_max_factors_is_231_l2110_211059

/-- The number of positive factors of b^n, where b and n are positive integers -/
def num_factors (b n : ℕ+) : ℕ := sorry

/-- The maximum number of positive factors for b^n given constraints -/
def max_num_factors : ℕ := sorry

theorem max_factors_is_231 :
  ∀ b n : ℕ+, b ≤ 20 → n ≤ 10 → num_factors b n ≤ max_num_factors ∧ max_num_factors = 231 := by sorry

end NUMINAMATH_CALUDE_max_factors_is_231_l2110_211059


namespace NUMINAMATH_CALUDE_three_roots_and_minimum_implies_ratio_l2110_211064

/-- Given positive real numbers a, b, c with a > c, if the equation |x²-ax+b| = cx 
    has exactly three distinct real roots, and the function f(x) = |x²-ax+b| + cx 
    has a minimum value of c², then a/c = 5 -/
theorem three_roots_and_minimum_implies_ratio (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hac : a > c)
  (h_three_roots : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    |x^2 - a*x + b| = c*x ∧ |y^2 - a*y + b| = c*y ∧ |z^2 - a*z + b| = c*z)
  (h_min : ∃ m : ℝ, ∀ x : ℝ, |x^2 - a*x + b| + c*x ≥ c^2 ∧ 
    ∃ x₀ : ℝ, |x₀^2 - a*x₀ + b| + c*x₀ = c^2) :
  a / c = 5 := by
sorry

end NUMINAMATH_CALUDE_three_roots_and_minimum_implies_ratio_l2110_211064


namespace NUMINAMATH_CALUDE_triangular_pyramid_theorem_l2110_211054

/-- A triangular pyramid with face areas S₁, S₂, S₃, S₄, distances H₁, H₂, H₃, H₄ 
    from any internal point to the faces, volume V, and constant k. -/
structure TriangularPyramid where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ
  k : ℝ
  h_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0 ∧ V > 0 ∧ k > 0
  h_ratio : S₁ / 1 = S₂ / 2 ∧ S₂ / 2 = S₃ / 3 ∧ S₃ / 3 = S₄ / 4 ∧ S₄ / 4 = k

/-- The theorem to be proved -/
theorem triangular_pyramid_theorem (p : TriangularPyramid) : 
  1 * p.H₁ + 2 * p.H₂ + 3 * p.H₃ + 4 * p.H₄ = 3 * p.V / p.k := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_theorem_l2110_211054


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2110_211082

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 275) : |x - y| = 14 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2110_211082


namespace NUMINAMATH_CALUDE_sum_of_ages_l2110_211072

/-- The sum of Mario and Maria's ages is 7 years -/
theorem sum_of_ages : 
  ∀ (mario_age maria_age : ℕ),
  mario_age = 4 →
  mario_age = maria_age + 1 →
  mario_age + maria_age = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2110_211072


namespace NUMINAMATH_CALUDE_race_theorem_l2110_211067

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The result of the first race -/
def first_race_result (r : Race) (d : ℝ) : Prop :=
  r.distance / r.runner_b.speed = (r.distance - d) / r.runner_a.speed

/-- The theorem to be proved -/
theorem race_theorem (h d : ℝ) (r : Race) 
  (h_pos : h > 0)
  (d_pos : d > 0)
  (first_race : first_race_result r d)
  (h_eq : r.distance = h) :
  let second_race_time := (h + d/2) / r.runner_a.speed
  let second_race_b_distance := second_race_time * r.runner_b.speed
  h - second_race_b_distance = d * (d + h) / (2 * h) := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l2110_211067


namespace NUMINAMATH_CALUDE_factorization_proof_l2110_211003

theorem factorization_proof (m : ℝ) : 4 - m^2 = (2 + m) * (2 - m) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2110_211003


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l2110_211017

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l2110_211017


namespace NUMINAMATH_CALUDE_teaching_years_difference_l2110_211044

/-- Proves that Virginia has taught 9 fewer years than Dennis given the conditions of the problem --/
theorem teaching_years_difference :
  ∀ (V A D : ℕ),
  V + A + D = 93 →
  V = A + 9 →
  D = 40 →
  V < D →
  D - V = 9 := by
sorry

end NUMINAMATH_CALUDE_teaching_years_difference_l2110_211044


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2110_211084

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = a^2 - 1 + Complex.I * (a - 1)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2110_211084


namespace NUMINAMATH_CALUDE_happy_equation_properties_l2110_211073

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b^2) / (4 * a)

def happy_numbers_to_each_other (a b c p q r : ℤ) : Prop :=
  |r * happy_number a b c - c * happy_number p q r| = 0

theorem happy_equation_properties :
  ∀ (a b c m n p q r : ℤ),
  (a ≠ 0 ∧ p ≠ 0) →
  (∃ (x y : ℤ), a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) →
  (∃ (x y : ℤ), p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ x ≠ y) →
  (happy_number 1 (-2) (-3) = -4) ∧
  (1 < m ∧ m < 6 ∧ 
   ∃ (x y : ℤ), x^2 - (2*m-1)*x + (m^2-2*m-3) = 0 ∧ 
                y^2 - (2*m-1)*y + (m^2-2*m-3) = 0 ∧ 
                x ≠ y →
   m = 3 ∧ happy_number 1 (-5) 0 = -25/4) ∧
  (∃ (x1 y1 x2 y2 : ℤ),
    x1^2 - m*x1 + (m+1) = 0 ∧ y1^2 - m*y1 + (m+1) = 0 ∧ x1 ≠ y1 ∧
    x2^2 - (n+2)*x2 + 2*n = 0 ∧ y2^2 - (n+2)*y2 + 2*n = 0 ∧ x2 ≠ y2 ∧
    happy_numbers_to_each_other 1 (-m) (m+1) 1 (-(n+2)) (2*n) →
    n = 0 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_happy_equation_properties_l2110_211073


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2110_211002

/-- Given a triangle with sides 9, 12, and 15 units and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ) :
  triangle_side1 = 9 ∧ triangle_side2 = 12 ∧ triangle_side3 = 15 ∧ rectangle_width = 6 ∧
  (1/2 * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2110_211002


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2110_211047

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {-4, 5/2}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y : Set ℝ := {38, 31.5}

/-- First parabola equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 5 * x - 6

/-- Second parabola equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 14

theorem parabolas_intersection :
  ∀ x ∈ intersection_x, ∃ y ∈ intersection_y,
    parabola1 x = y ∧ parabola2 x = y :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2110_211047


namespace NUMINAMATH_CALUDE_angle_AO2B_greater_than_90_degrees_l2110_211078

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the center of circle O₂
def O2_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem angle_AO2B_greater_than_90_degrees :
  let angle_AO2B := sorry
  angle_AO2B > 90 := by sorry

end NUMINAMATH_CALUDE_angle_AO2B_greater_than_90_degrees_l2110_211078


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2110_211040

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + Complex.I) * (1 - Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2110_211040


namespace NUMINAMATH_CALUDE_lcm_of_4_6_15_l2110_211065

theorem lcm_of_4_6_15 : Nat.lcm (Nat.lcm 4 6) 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_6_15_l2110_211065


namespace NUMINAMATH_CALUDE_first_month_sale_l2110_211075

def average_sale : ℕ := 7000
def num_months : ℕ := 6
def sale_month2 : ℕ := 6524
def sale_month3 : ℕ := 5689
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6000
def sale_month6 : ℕ := 12557

theorem first_month_sale (sale_month1 : ℕ) : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6 = average_sale * num_months →
  sale_month1 = average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) :=
by
  sorry

#eval average_sale * num_months - (sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6)

end NUMINAMATH_CALUDE_first_month_sale_l2110_211075


namespace NUMINAMATH_CALUDE_distance_after_three_minutes_l2110_211057

/-- The distance between two vehicles after a given time, given their speeds and initial positions. -/
def distanceBetweenVehicles (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed2 - speed1) * time

theorem distance_after_three_minutes :
  let truckSpeed : ℝ := 65
  let carSpeed : ℝ := 85
  let time : ℝ := 3 / 60
  distanceBetweenVehicles truckSpeed carSpeed time = 1 := by
  sorry

#check distance_after_three_minutes

end NUMINAMATH_CALUDE_distance_after_three_minutes_l2110_211057


namespace NUMINAMATH_CALUDE_fraction_equality_l2110_211055

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2110_211055


namespace NUMINAMATH_CALUDE_all_numbers_multiple_of_three_l2110_211086

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def numbers_to_check : List ℕ := [123, 234, 345, 456, 567]

theorem all_numbers_multiple_of_three 
  (h : ∀ n : ℕ, is_multiple_of_three n ↔ is_multiple_of_three (sum_of_digits n)) :
  ∀ n ∈ numbers_to_check, is_multiple_of_three n :=
by sorry

end NUMINAMATH_CALUDE_all_numbers_multiple_of_three_l2110_211086


namespace NUMINAMATH_CALUDE_gcd_consecutive_terms_unbounded_l2110_211020

def a (n : ℕ) : ℤ := n.factorial - n

theorem gcd_consecutive_terms_unbounded :
  ∀ M : ℕ, ∃ n : ℕ, Int.gcd (a n) (a (n + 1)) > M :=
sorry

end NUMINAMATH_CALUDE_gcd_consecutive_terms_unbounded_l2110_211020


namespace NUMINAMATH_CALUDE_polar_equation_is_circle_l2110_211061

/-- The curve represented by the polar equation ρ = sin θ + cos θ is a circle. -/
theorem polar_equation_is_circle :
  ∀ (ρ θ : ℝ), ρ = Real.sin θ + Real.cos θ →
  ∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ),
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - x₀)^2 + (y - y₀)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_equation_is_circle_l2110_211061


namespace NUMINAMATH_CALUDE_expression_evaluation_l2110_211089

theorem expression_evaluation :
  let x : ℚ := -2
  (x - 2)^2 + (2 + x)*(x - 2) - 2*x*(2*x - 1) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2110_211089


namespace NUMINAMATH_CALUDE_complement_of_A_l2110_211049

-- Define the set A
def A : Set ℝ := {x : ℝ | (x + 1) / (x + 2) ≤ 0}

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Ici (-1) ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_l2110_211049


namespace NUMINAMATH_CALUDE_four_planes_max_parts_l2110_211030

/-- The maximum number of parts into which space can be divided by k planes -/
def max_parts (k : ℕ) : ℚ := (k^3 + 5*k + 6) / 6

/-- Theorem: The maximum number of parts into which space can be divided by four planes is 15 -/
theorem four_planes_max_parts : max_parts 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_four_planes_max_parts_l2110_211030


namespace NUMINAMATH_CALUDE_perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2110_211026

/-- A natural number is a perfect square if and only if each prime in its prime factorization appears an even number of times. -/
theorem perfect_square_iff_even_exponents (n : ℕ) :
  (∃ m : ℕ, n = m ^ 2) ↔ (∀ p : ℕ, Prime p → ∃ k : ℕ, n.factorization p = 2 * k) :=
sorry

/-- A natural number is a k-th power if and only if each prime in its prime factorization appears a number of times divisible by k. -/
theorem kth_power_iff_divisible_exponents (n k : ℕ) (hk : k > 0) :
  (∃ m : ℕ, n = m ^ k) ↔ (∀ p : ℕ, Prime p → ∃ l : ℕ, n.factorization p = k * l) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_iff_even_exponents_kth_power_iff_divisible_exponents_l2110_211026


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2110_211077

theorem shaded_area_theorem (r R : ℝ) (h1 : R > 0) (h2 : r > 0) : 
  (π * R^2 = 100 * π) → (r = R / 2) → 
  (π * R^2 / 2 + π * r^2 / 4 = 31.25 * π) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2110_211077


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2110_211033

/-- Represents the number of tokens Alex has --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver : ExchangeRule  -- 3 red → 2 silver + 1 blue
  | BlueToSilver : ExchangeRule -- 2 blue → 1 silver + 1 red

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : Option TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver =>
      if tc.red ≥ 3 then
        some ⟨tc.red - 3, tc.blue + 1, tc.silver + 2⟩
      else
        none
  | ExchangeRule.BlueToSilver =>
      if tc.blue ≥ 2 then
        some ⟨tc.red + 1, tc.blue - 2, tc.silver + 1⟩
      else
        none

/-- Checks if any exchange is possible --/
def canExchange (tc : TokenCount) : Bool :=
  tc.red ≥ 3 ∨ tc.blue ≥ 2

/-- The main theorem to prove --/
theorem max_silver_tokens (initialRed initialBlue : ℕ) 
    (h1 : initialRed = 100) (h2 : initialBlue = 50) :
    ∃ (finalTokens : TokenCount),
      finalTokens.red < 3 ∧ 
      finalTokens.blue < 2 ∧
      finalTokens.silver = 147 ∧
      (∃ (exchanges : List ExchangeRule), 
        finalTokens = exchanges.foldl 
          (fun acc rule => 
            match applyExchange acc rule with
            | some newCount => newCount
            | none => acc) 
          ⟨initialRed, initialBlue, 0⟩) := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2110_211033


namespace NUMINAMATH_CALUDE_complement_of_S_in_U_l2110_211043

def U : Finset Nat := {1,2,3,4,5,6,7}
def S : Finset Nat := {1,3,5}

theorem complement_of_S_in_U :
  (U \ S) = {2,4,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_of_S_in_U_l2110_211043


namespace NUMINAMATH_CALUDE_students_in_class_l2110_211010

def total_pencils : ℕ := 125
def pencils_per_student : ℕ := 5

theorem students_in_class : total_pencils / pencils_per_student = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_in_class_l2110_211010


namespace NUMINAMATH_CALUDE_only_t_squared_valid_l2110_211052

-- Define a type for programming statements
inductive ProgramStatement
  | Input (var : String) (value : String)
  | Assignment (var : String) (expr : String)
  | Print (var : String) (value : String)

-- Define a function to check if a statement is valid
def isValidStatement : ProgramStatement → Bool
  | ProgramStatement.Input var value => false  -- INPUT x=3 is not valid
  | ProgramStatement.Assignment "T" "T*T" => true  -- T=T*T is valid
  | ProgramStatement.Assignment var1 var2 => false  -- A=B=2 is not valid
  | ProgramStatement.Print var value => false  -- PRINT A=4 is not valid

-- Theorem stating that only T=T*T is valid among the given statements
theorem only_t_squared_valid :
  (isValidStatement (ProgramStatement.Input "x" "3") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "A" "B=2") = false) ∧
  (isValidStatement (ProgramStatement.Assignment "T" "T*T") = true) ∧
  (isValidStatement (ProgramStatement.Print "A" "4") = false) := by
  sorry


end NUMINAMATH_CALUDE_only_t_squared_valid_l2110_211052


namespace NUMINAMATH_CALUDE_john_running_days_l2110_211019

/-- The number of days John ran before getting injured -/
def days_ran (daily_distance : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / daily_distance

theorem john_running_days :
  days_ran 1700 10200 = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_running_days_l2110_211019


namespace NUMINAMATH_CALUDE_tank_capacity_l2110_211009

theorem tank_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) : 
  num_trucks = 3 → 
  tanks_per_truck = 3 → 
  total_capacity = 1350 → 
  (total_capacity / (num_trucks * tanks_per_truck) : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l2110_211009


namespace NUMINAMATH_CALUDE_irrigation_canal_construction_l2110_211091

/-- Irrigation Canal Construction Problem -/
theorem irrigation_canal_construction
  (total_length : ℝ)
  (team_b_extra : ℝ)
  (time_ratio : ℝ)
  (cost_a : ℝ)
  (cost_b : ℝ)
  (total_time : ℝ)
  (h_total_length : total_length = 1650)
  (h_team_b_extra : team_b_extra = 30)
  (h_time_ratio : time_ratio = 3/2)
  (h_cost_a : cost_a = 90000)
  (h_cost_b : cost_b = 120000)
  (h_total_time : total_time = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧
    rate_b = 90 ∧
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    (total_length / rate_b) * time_ratio = (total_length / rate_a) ∧
    ∃ (solo_days : ℝ),
      solo_days * rate_a + (total_time - solo_days) * (rate_a + rate_b) = total_length ∧
      total_cost = solo_days * cost_a + total_time * cost_a + (total_time - solo_days) * cost_b :=
by sorry

end NUMINAMATH_CALUDE_irrigation_canal_construction_l2110_211091


namespace NUMINAMATH_CALUDE_pyramid_display_rows_l2110_211023

/-- Represents the number of cans in a pyramid display. -/
def pyramid_display (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that a pyramid display with 210 cans has 14 rows. -/
theorem pyramid_display_rows :
  ∃ (n : ℕ), pyramid_display n = 210 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_display_rows_l2110_211023


namespace NUMINAMATH_CALUDE_airplane_seats_multiple_l2110_211080

theorem airplane_seats_multiple (total_seats first_class_seats : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class_seats = 77)
  (h3 : ∃ m : ℕ, total_seats = first_class_seats + (m * first_class_seats + 2)) :
  ∃ m : ℕ, m = 4 ∧ total_seats = first_class_seats + (m * first_class_seats + 2) :=
by sorry

end NUMINAMATH_CALUDE_airplane_seats_multiple_l2110_211080


namespace NUMINAMATH_CALUDE_line_equation_solution_l2110_211088

theorem line_equation_solution (a b : ℝ) (h_a : a ≠ 0) :
  (∀ x y : ℝ, y = a * x + b) →
  (4 = a * 0 + b) →
  (0 = a * (-3) + b) →
  (∀ x : ℝ, a * x + b = 0 ↔ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_solution_l2110_211088


namespace NUMINAMATH_CALUDE_problem_statement_l2110_211025

theorem problem_statement (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 31 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (h1 : a ≠ 17)
  (h2 : x ≠ 0) :
  a / (a - 17) + b / (b - 31) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2110_211025


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2110_211005

theorem largest_k_for_distinct_roots : 
  ∀ k : ℤ, 
  (∃ x y : ℝ, x ≠ y ∧ 
    (k - 2 : ℝ) * x^2 - 4 * x + 4 = 0 ∧ 
    (k - 2 : ℝ) * y^2 - 4 * y + 4 = 0) →
  k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2110_211005


namespace NUMINAMATH_CALUDE_output_for_twelve_l2110_211092

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 2 else step1 / 2

theorem output_for_twelve : function_machine 12 = 34 := by sorry

end NUMINAMATH_CALUDE_output_for_twelve_l2110_211092


namespace NUMINAMATH_CALUDE_pencil_count_l2110_211008

/-- Given a shop with pencils, pens, and exercise books in a ratio of 10 : 2 : 3,
    and 36 exercise books in total, prove that there are 120 pencils. -/
theorem pencil_count (ratio_pencils : ℕ) (ratio_pens : ℕ) (ratio_books : ℕ) 
    (total_books : ℕ) (h1 : ratio_pencils = 10) (h2 : ratio_pens = 2) 
    (h3 : ratio_books = 3) (h4 : total_books = 36) : 
    ratio_pencils * (total_books / ratio_books) = 120 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2110_211008


namespace NUMINAMATH_CALUDE_jacks_second_half_time_l2110_211024

/-- Proves that Jack's time for the second half of the hill is 6 seconds -/
theorem jacks_second_half_time
  (jack_first_half : ℕ)
  (jack_finishes_before : ℕ)
  (jill_total_time : ℕ)
  (h1 : jack_first_half = 19)
  (h2 : jack_finishes_before = 7)
  (h3 : jill_total_time = 32) :
  jill_total_time - jack_finishes_before - jack_first_half = 6 := by
  sorry

#check jacks_second_half_time

end NUMINAMATH_CALUDE_jacks_second_half_time_l2110_211024


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2110_211007

theorem quadratic_one_solution (q : ℝ) (hq : q ≠ 0) :
  (∃! x, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2110_211007


namespace NUMINAMATH_CALUDE_greatest_integer_third_side_l2110_211058

theorem greatest_integer_third_side (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ a + x > b ∧ b + x > a)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_third_side_l2110_211058


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l2110_211062

theorem diophantine_equation_solvable (n : ℤ) :
  ∃ (x y z : ℤ), 10 * x * y + 17 * y * z + 27 * z * x = n :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l2110_211062


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2110_211060

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -9 ∧ x₂ = 1 ∧ x₁^2 + 8*x₁ - 9 = 0 ∧ x₂^2 + 8*x₂ - 9 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -3 ∧ y₂ = 1 ∧ y₁*(y₁-1) + 3*(y₁-1) = 0 ∧ y₂*(y₂-1) + 3*(y₂-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2110_211060


namespace NUMINAMATH_CALUDE_smallest_k_for_congruence_l2110_211004

theorem smallest_k_for_congruence : 
  (∃ k : ℕ, k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) ∧
  201 % 24 = 9 % 24 →
  (∃ k : ℕ, k = 8 ∧ k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_congruence_l2110_211004


namespace NUMINAMATH_CALUDE_bowling_ball_weights_l2110_211056

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℕ := 36

/-- The number of bowling balls that weigh the same as the canoes -/
def num_bowling_balls : ℕ := 9

/-- The number of canoes that weigh the same as the bowling balls -/
def num_canoes : ℕ := 4

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℕ := canoe_weight * num_canoes / num_bowling_balls

/-- The total weight of five bowling balls in pounds -/
def five_bowling_balls_weight : ℕ := bowling_ball_weight * 5

theorem bowling_ball_weights :
  bowling_ball_weight = 16 ∧ five_bowling_balls_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weights_l2110_211056


namespace NUMINAMATH_CALUDE_min_dials_for_equal_sums_l2110_211081

/-- A type representing a 12-sided dial with numbers from 1 to 12 -/
def Dial := Fin 12 → Fin 12

/-- A stack of dials -/
def Stack := ℕ → Dial

/-- The sum of numbers in a column of the stack -/
def columnSum (s : Stack) (col : Fin 12) (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => (s i col).val + 1)

/-- Whether all column sums have the same remainder modulo 12 -/
def allColumnSumsEqualMod12 (s : Stack) (n : ℕ) : Prop :=
  ∀ (c₁ c₂ : Fin 12), columnSum s c₁ n % 12 = columnSum s c₂ n % 12

/-- The theorem stating that 12 is the minimum number of dials required -/
theorem min_dials_for_equal_sums :
  ∀ (s : Stack), (∃ (n : ℕ), allColumnSumsEqualMod12 s n) →
  (∃ (n : ℕ), n ≥ 12 ∧ allColumnSumsEqualMod12 s n) :=
by sorry

end NUMINAMATH_CALUDE_min_dials_for_equal_sums_l2110_211081


namespace NUMINAMATH_CALUDE_find_x_l2110_211093

theorem find_x (x y z : ℝ) 
  (hxy : x * y / (x + y) = 4)
  (hxz : x * z / (x + z) = 9)
  (hyz : y * z / (y + z) = 16)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hdist : x ≠ y ∧ y ≠ z ∧ x ≠ z) :
  x = 384 / 21 := by
sorry

end NUMINAMATH_CALUDE_find_x_l2110_211093


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2110_211027

/-- Calculates the gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage (true_weight false_weight : ℕ) : 
  true_weight = 1000 → 
  false_weight = 960 → 
  (true_weight - false_weight) * 100 / true_weight = 4 := by
  sorry

#check shopkeeper_gain_percentage

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l2110_211027


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2110_211076

/-- The range of m for which two circles intersect -/
theorem circle_intersection_range :
  let circle1 : ℝ → ℝ → ℝ → Prop := λ x y m ↦ x^2 + y^2 = m
  let circle2 : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 6*x + 8*y - 24 = 0
  ∀ m : ℝ, (∃ x y : ℝ, circle1 x y m ∧ circle2 x y) ↔ 4 < m ∧ m < 144 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2110_211076


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_three_halves_l2110_211095

theorem sqrt_meaningful_iff_x_geq_three_halves (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 2 * x - 3) ↔ x ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_three_halves_l2110_211095


namespace NUMINAMATH_CALUDE_joan_change_l2110_211048

/-- The change Joan received after buying a cat toy and a cage -/
theorem joan_change (cat_toy_cost cage_cost payment : ℚ) : 
  cat_toy_cost = 877/100 →
  cage_cost = 1097/100 →
  payment = 20 →
  payment - (cat_toy_cost + cage_cost) = 26/100 := by
sorry

end NUMINAMATH_CALUDE_joan_change_l2110_211048


namespace NUMINAMATH_CALUDE_inverse_matrices_values_l2110_211018

def Matrix1 (a : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![a, 2],
    ![1, 4]]

def Matrix2 (b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![-2/7, 1/7],
    ![b, 3/14]]

theorem inverse_matrices_values (a b : ℚ) : 
  Matrix1 a * Matrix2 b = 1 → a = -3 ∧ b = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_values_l2110_211018


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2110_211045

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2110_211045


namespace NUMINAMATH_CALUDE_square_of_integer_root_l2110_211031

theorem square_of_integer_root (n : ℕ) : 
  ∃ (m : ℤ), (2 : ℝ) + 2 * Real.sqrt (28 * (n^2 : ℝ) + 1) = m → 
  ∃ (k : ℤ), m = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_of_integer_root_l2110_211031


namespace NUMINAMATH_CALUDE_inequality_comparison_l2110_211046

theorem inequality_comparison (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ 
  (abs a > abs b) ∧ 
  (a^2 > b^2) ∧
  ¬(∀ a b, a < b ∧ b < 0 → 1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2110_211046


namespace NUMINAMATH_CALUDE_specific_square_figure_perimeter_l2110_211011

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.horizontalSegments + f.verticalSegments) * f.squareSideLength * 2

/-- Theorem stating that the perimeter of the specific square figure is 52 -/
theorem specific_square_figure_perimeter :
  ∃ (f : SquareFigure),
    f.squareSideLength = 2 ∧
    f.horizontalSegments = 16 ∧
    f.verticalSegments = 10 ∧
    perimeter f = 52 := by
  sorry

end NUMINAMATH_CALUDE_specific_square_figure_perimeter_l2110_211011


namespace NUMINAMATH_CALUDE_max_equalization_value_l2110_211096

/-- Represents a 3x3 board with numbers --/
def Board := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in the board are equal --/
def all_equal (b : Board) : Prop :=
  ∀ i j k l, b i j = b k l

/-- Represents a valid operation on the board --/
inductive Operation
| row (i : Fin 3) (x : ℝ)
| col (j : Fin 3) (x : ℝ)

/-- Applies an operation to the board --/
def apply_operation (b : Board) (op : Operation) : Board :=
  sorry

/-- Checks if a board can be transformed to have all elements equal to m --/
def can_equalize (b : Board) (m : ℕ) : Prop :=
  ∃ (ops : List Operation), all_equal (ops.foldl apply_operation b) ∧
    ∀ i j, (ops.foldl apply_operation b) i j = m

/-- Initial board configuration --/
def initial_board : Board :=
  λ i j => i.val * 3 + j.val + 1

/-- Main theorem: The maximum value of m for which the board can be equalized is 4 --/
theorem max_equalization_value :
  (∀ m : ℕ, m > 4 → ¬ can_equalize initial_board m) ∧
  can_equalize initial_board 4 :=
sorry

end NUMINAMATH_CALUDE_max_equalization_value_l2110_211096


namespace NUMINAMATH_CALUDE_pairing_probability_l2110_211085

/-- The probability of one student being paired with another specific student
    in a class where some students are absent. -/
theorem pairing_probability
  (total_students : ℕ)
  (absent_students : ℕ)
  (h1 : total_students = 40)
  (h2 : absent_students = 5)
  (h3 : absent_students < total_students) :
  (1 : ℚ) / (total_students - absent_students - 1) = 1 / 34 :=
by sorry

end NUMINAMATH_CALUDE_pairing_probability_l2110_211085


namespace NUMINAMATH_CALUDE_no_real_solutions_for_quadratic_inequality_l2110_211066

theorem no_real_solutions_for_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_quadratic_inequality_l2110_211066


namespace NUMINAMATH_CALUDE_parallel_resistance_calculation_l2110_211012

/-- 
Represents the combined resistance of two resistors connected in parallel.
x: resistance of the first resistor in ohms
y: resistance of the second resistor in ohms
r: combined resistance in ohms
-/
def parallel_resistance (x y : ℝ) (r : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ r > 0 ∧ (1 / r = 1 / x + 1 / y)

theorem parallel_resistance_calculation :
  ∃ (r : ℝ), parallel_resistance 4 6 r ∧ r = 2.4 := by sorry

end NUMINAMATH_CALUDE_parallel_resistance_calculation_l2110_211012


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2110_211050

/-- Arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term :
  arithmeticSequence 150 = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l2110_211050


namespace NUMINAMATH_CALUDE_f_properties_l2110_211069

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Define monotonicity for a function on ℝ
def Monotonic (g : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → g x ≤ g y ∨ ∀ x y, x ≤ y → g y ≤ g x

-- Theorem statement
theorem f_properties :
  (f 2 (f 2 2) = 0) ∧
  (∀ a : ℝ, Monotonic (f a) ↔ 1/7 ≤ a ∧ a < 1/3) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2110_211069


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l2110_211001

theorem lcm_from_hcf_and_product (x y : ℕ+) : 
  Nat.gcd x y = 12 → x * y = 2460 → Nat.lcm x y = 205 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l2110_211001


namespace NUMINAMATH_CALUDE_unique_f_two_l2110_211094

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x + y) + x * f y - 2 * x * y - x + 2

theorem unique_f_two (f : ℝ → ℝ) (h : FunctionalEquation f) : 
  ∃! z : ℝ, f 2 = z ∧ z = 4 := by sorry

end NUMINAMATH_CALUDE_unique_f_two_l2110_211094


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l2110_211037

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, (n / 10^d % 10 ≠ 0 → is_odd_digit (n / 10^d % 10))

def alternating_sum (n : ℕ) : ℤ :=
  let digits := List.reverse (List.map (λ i => n / 10^i % 10) (List.range 4))
  List.foldl (λ sum (i, d) => sum + ((-1)^i : ℤ) * d) 0 (List.enumFrom 0 digits)

theorem largest_odd_digit_multiple_of_11 :
  9393 < 10000 ∧
  has_only_odd_digits 9393 ∧
  alternating_sum 9393 % 11 = 0 ∧
  ∀ n : ℕ, n < 10000 → has_only_odd_digits n → alternating_sum n % 11 = 0 → n ≤ 9393 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l2110_211037


namespace NUMINAMATH_CALUDE_paint_cans_problem_l2110_211038

theorem paint_cans_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) :
  original_rooms = 50 →
  lost_cans = 5 →
  remaining_rooms = 35 →
  (∃ (cans_per_room : ℚ), 
    cans_per_room * (original_rooms - remaining_rooms) = lost_cans ∧
    cans_per_room * remaining_rooms = 12) :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_problem_l2110_211038


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2110_211006

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) : 
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2110_211006


namespace NUMINAMATH_CALUDE_subset_sum_exists_l2110_211036

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 → 
  (∀ n ∈ nums, n ≤ 100) → 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l2110_211036


namespace NUMINAMATH_CALUDE_unique_base_for_625_l2110_211098

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_two_digits_odd (n : ℕ) (b : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ d₄ : ℕ, 
    n = d₁ * b^3 + d₂ * b^2 + d₃ * b^1 + d₄ * b^0 ∧
    d₃ % 2 = 1 ∧ d₄ % 2 = 1

theorem unique_base_for_625 :
  ∃! b : ℕ, b > 1 ∧ is_four_digit 625 b ∧ last_two_digits_odd 625 b :=
sorry

end NUMINAMATH_CALUDE_unique_base_for_625_l2110_211098


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l2110_211034

/-- A hyperbola passing through specific points -/
structure Hyperbola where
  /-- The hyperbola is centered at the origin -/
  center : (ℝ × ℝ) := (0, 0)
  /-- The hyperbola passes through (5, -6) -/
  point1 : (ℝ × ℝ) := (5, -6)
  /-- The hyperbola passes through (3, 0) -/
  point2 : (ℝ × ℝ) := (3, 0)
  /-- The hyperbola passes through (s, -3) for some real s -/
  point3 : (ℝ × ℝ)
  /-- The third point has y-coordinate -3 -/
  h_point3_y : point3.2 = -3

/-- The theorem stating that s² = 12 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.point3.1 ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l2110_211034


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l2110_211035

/-- Represents a combination of soda packs -/
structure SodaPacks where
  pack8 : ℕ
  pack15 : ℕ
  pack32 : ℕ

/-- Calculates the total number of cans for a given combination of packs -/
def totalCans (packs : SodaPacks) : ℕ :=
  8 * packs.pack8 + 15 * packs.pack15 + 32 * packs.pack32

/-- Calculates the total number of packs for a given combination -/
def totalPacks (packs : SodaPacks) : ℕ :=
  packs.pack8 + packs.pack15 + packs.pack32

/-- Theorem: The minimum number of packs to buy exactly 120 cans is 6 -/
theorem min_packs_for_120_cans : 
  ∃ (min_packs : SodaPacks), 
    totalCans min_packs = 120 ∧ 
    totalPacks min_packs = 6 ∧
    ∀ (other_packs : SodaPacks), 
      totalCans other_packs = 120 → 
      totalPacks other_packs ≥ totalPacks min_packs :=
by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l2110_211035


namespace NUMINAMATH_CALUDE_square_plus_one_ge_two_abs_l2110_211099

theorem square_plus_one_ge_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_two_abs_l2110_211099


namespace NUMINAMATH_CALUDE_pink_cubes_after_cutting_l2110_211087

/-- Represents a cube with a given volume and number of colored faces -/
structure ColoredCube where
  volume : ℝ
  coloredFaces : ℕ

/-- Represents the result of cutting a larger cube into smaller cubes -/
structure CutCubeResult where
  totalCubes : ℕ
  coloredCubes : ℕ

/-- Function to calculate the number of colored cubes after cutting -/
def cutAndCountColored (cube : ColoredCube) (cuts : ℕ) : CutCubeResult :=
  sorry

/-- Theorem stating the result for the specific problem -/
theorem pink_cubes_after_cutting :
  let largeCube : ColoredCube := ⟨125, 2⟩
  let result := cutAndCountColored largeCube 125
  result.totalCubes = 125 ∧ result.coloredCubes = 46 := by sorry

end NUMINAMATH_CALUDE_pink_cubes_after_cutting_l2110_211087


namespace NUMINAMATH_CALUDE_stacy_height_proof_l2110_211016

/-- Calculates Stacy's current height given her previous height, James' growth, and the difference between their growth. -/
def stacys_current_height (stacy_previous_height james_growth growth_difference : ℕ) : ℕ :=
  stacy_previous_height + james_growth + growth_difference

/-- Proves that Stacy's current height is 57 inches. -/
theorem stacy_height_proof :
  stacys_current_height 50 1 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_stacy_height_proof_l2110_211016


namespace NUMINAMATH_CALUDE_remainder_problem_l2110_211042

theorem remainder_problem (n : ℕ) (h : 2 * n % 4 = 2) : n % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2110_211042


namespace NUMINAMATH_CALUDE_correct_meiosis_sequence_l2110_211039

-- Define the stages of meiosis
inductive MeiosisStage
  | Replication
  | Synapsis
  | Separation
  | Division

-- Define a sequence type
def Sequence := List MeiosisStage

-- Define the four given sequences
def sequenceA : Sequence := [MeiosisStage.Replication, MeiosisStage.Synapsis, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceB : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Division]
def sequenceC : Sequence := [MeiosisStage.Synapsis, MeiosisStage.Replication, MeiosisStage.Division, MeiosisStage.Separation]
def sequenceD : Sequence := [MeiosisStage.Replication, MeiosisStage.Separation, MeiosisStage.Synapsis, MeiosisStage.Division]

-- Define a function to check if a sequence is correct
def isCorrectSequence (s : Sequence) : Prop :=
  s = sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_meiosis_sequence :
  isCorrectSequence sequenceA ∧
  ¬isCorrectSequence sequenceB ∧
  ¬isCorrectSequence sequenceC ∧
  ¬isCorrectSequence sequenceD :=
sorry

end NUMINAMATH_CALUDE_correct_meiosis_sequence_l2110_211039


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2110_211028

/-- Given a parabola y² = 2px with a point Q(6, y₀) on it, 
    if the distance from Q to the focus is 10, 
    then the distance from the focus to the directrix is 8. -/
theorem parabola_focus_directrix_distance 
  (p : ℝ) (y₀ : ℝ) :
  y₀^2 = 2*p*6 → -- Q(6, y₀) lies on the parabola y² = 2px
  (6 + p/2)^2 + y₀^2 = 10^2 → -- distance from Q to focus is 10
  p = 8 := -- distance from focus to directrix
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2110_211028


namespace NUMINAMATH_CALUDE_change_parity_mismatch_l2110_211063

theorem change_parity_mismatch (bills : List ℕ) (denominations : List ℕ) :
  (bills.length = 10) →
  (∀ d ∈ denominations, d % 2 = 1) →
  (∀ b ∈ bills, b ∈ denominations) →
  (bills.sum ≠ 31) :=
sorry

end NUMINAMATH_CALUDE_change_parity_mismatch_l2110_211063


namespace NUMINAMATH_CALUDE_min_sum_three_integers_l2110_211000

theorem min_sum_three_integers (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (∃ (k₁ k₂ k₃ : ℕ), 
    (1 / a + 1 / b : ℚ) = k₁ * (1 / c : ℚ) ∧
    (1 / a + 1 / c : ℚ) = k₂ * (1 / b : ℚ) ∧
    (1 / b + 1 / c : ℚ) = k₃ * (1 / a : ℚ)) →
  a + b + c ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_three_integers_l2110_211000


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2110_211071

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a₃ = 20 and a₆ = 5, prove that a₉ = 5/4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a3 : a 3 = 20) 
    (h_a6 : a 6 = 5) : 
  a 9 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2110_211071


namespace NUMINAMATH_CALUDE_tv_screen_coverage_l2110_211029

theorem tv_screen_coverage (w1 h1 w2 h2 : ℚ) : 
  w1 / h1 = 16 / 9 →
  w2 / h2 = 4 / 3 →
  (h2 - h1 * (w2 / w1)) / h2 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tv_screen_coverage_l2110_211029


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2110_211079

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 12) :
  x^3 + y^3 = 91 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2110_211079


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2110_211074

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ)
  (h_rate : rate = 0.08)
  (h_time : time = 1)
  (h_interest : interest = 800)
  (h_formula : interest = principal * rate * time) :
  principal = 10000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l2110_211074


namespace NUMINAMATH_CALUDE_susan_stationery_purchase_l2110_211032

theorem susan_stationery_purchase (pencil_cost : ℚ) (pen_cost : ℚ) (total_spent : ℚ) (pencils_bought : ℕ) :
  pencil_cost = 25 / 100 →
  pen_cost = 80 / 100 →
  total_spent = 20 →
  pencils_bought = 16 →
  ∃ (pens_bought : ℕ),
    (pencils_bought : ℚ) * pencil_cost + (pens_bought : ℚ) * pen_cost = total_spent ∧
    pencils_bought + pens_bought = 36 :=
by sorry

end NUMINAMATH_CALUDE_susan_stationery_purchase_l2110_211032


namespace NUMINAMATH_CALUDE_wire_around_square_field_l2110_211083

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) (times_around : ℕ) : 
  area = 69696 →
  wire_length = 15840 →
  times_around = 15 →
  wire_length = times_around * (4 * Real.sqrt area) :=
by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l2110_211083
