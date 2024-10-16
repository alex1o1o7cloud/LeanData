import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l1821_182151

/-- Given a system of equations with parameters a, b, and c, prove that the solutions for x, y, and z are as stated. -/
theorem system_solution (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∃ (x y z : ℝ),
    x ≠ y ∧
    (x - y) / (x + z) = a ∧
    (x^2 - y^2) / (x + z) = b ∧
    (x^3 + x^2*y - x*y^2 - y^3) / (x + z)^2 = b^2 / (a^2 * c) ∧
    x = (a^3 * c + b) / (2 * a) ∧
    y = (b - a^3 * c) / (2 * a) ∧
    z = (2 * a^2 * c - a^3 * c - b) / (2 * a) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1821_182151


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1821_182100

theorem quadratic_roots_sum_and_product (α β : ℝ) : 
  α ≠ β →
  α^2 - 5*α - 2 = 0 →
  β^2 - 5*β - 2 = 0 →
  α + β + α*β = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1821_182100


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1821_182170

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 7}

theorem complement_of_A_in_U :
  U \ A = {3, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1821_182170


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1821_182144

theorem cubic_equation_with_double_root (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 84 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l1821_182144


namespace NUMINAMATH_CALUDE_prism_volume_l1821_182146

/-- A right rectangular prism with face areas 15, 20, and 24 square inches has a volume of 60 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 20) (h3 : l * h = 24) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1821_182146


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1821_182128

theorem probability_nine_heads_in_twelve_flips : 
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l1821_182128


namespace NUMINAMATH_CALUDE_circle_area_through_points_l1821_182190

/-- The area of a circle with center R(1, 2) passing through S(-7, 6) is 80π -/
theorem circle_area_through_points : 
  let R : ℝ × ℝ := (1, 2)
  let S : ℝ × ℝ := (-7, 6)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 80 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l1821_182190


namespace NUMINAMATH_CALUDE_modulus_of_12_plus_5i_l1821_182138

theorem modulus_of_12_plus_5i : Complex.abs (12 + 5 * Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_12_plus_5i_l1821_182138


namespace NUMINAMATH_CALUDE_cats_remaining_after_sale_l1821_182131

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining_after_sale 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13)
  (h2 : house_cats = 5)
  (h3 : cats_sold = 10) :
  siamese_cats + house_cats - cats_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_sale_l1821_182131


namespace NUMINAMATH_CALUDE_polynomial_value_l1821_182189

def f (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) : ℝ :=
  a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 7 * x^5

theorem polynomial_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  f 2004 a₀ a₁ a₂ a₃ a₄ = 72 ∧
  f 2005 a₀ a₁ a₂ a₃ a₄ = -30 ∧
  f 2006 a₀ a₁ a₂ a₃ a₄ = 32 ∧
  f 2007 a₀ a₁ a₂ a₃ a₄ = -24 ∧
  f 2008 a₀ a₁ a₂ a₃ a₄ = 24 →
  f 2009 a₀ a₁ a₂ a₃ a₄ = 847 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1821_182189


namespace NUMINAMATH_CALUDE_dodecahedron_triangles_l1821_182116

/-- The number of vertices in a dodecahedron -/
def dodecahedron_vertices : ℕ := 20

/-- The number of distinct triangles that can be constructed by connecting three different vertices of a dodecahedron -/
def distinct_triangles (n : ℕ) : ℕ := n.choose 3

theorem dodecahedron_triangles :
  distinct_triangles dodecahedron_vertices = 1140 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangles_l1821_182116


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1821_182132

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + 6 → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n ∧ is_prime (n + 6) ∧
  ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m ∧ is_prime (m + 6)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l1821_182132


namespace NUMINAMATH_CALUDE_nth_odd_multiple_of_three_l1821_182159

theorem nth_odd_multiple_of_three (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, k > 0 ∧ k = 6 * n - 3 ∧ k % 2 = 1 ∧ k % 3 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m < k ∧ m % 2 = 1 ∧ m % 3 = 0 → 
   ∃ i : ℕ, i < n ∧ m = 6 * i - 3) :=
by sorry

end NUMINAMATH_CALUDE_nth_odd_multiple_of_three_l1821_182159


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1821_182185

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    y = m * x + b ∧
    (∃ h : x > 0, y = f x) →
    (x = 1 → y = f 1) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x', 0 < |x' - 1| ∧ |x' - 1| < δ →
      |y - (f 1 + (x' - 1) * ((f x' - f 1) / (x' - 1)))| / |x' - 1| < ε) →
    m = 1 ∧ b = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1821_182185


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1821_182115

theorem complex_fraction_simplification :
  let numerator := (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484)
  let denominator := (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484)
  ∀ x : ℕ, x^4 + 484 = (x^2 - 22*x + 22) * (x^2 + 22*x + 22) →
  (numerator / denominator : ℚ) = 3867 / 7 := by
sorry

#eval (3867 : ℚ) / 7

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1821_182115


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l1821_182113

-- Define the yard and flower beds
def yard_length : ℝ := 25
def yard_width : ℝ := 5
def flower_bed_area : ℝ := 50

-- Define the theorem
theorem flower_bed_fraction :
  let total_yard_area := yard_length * yard_width
  let total_flower_bed_area := 2 * flower_bed_area
  (total_flower_bed_area / total_yard_area) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l1821_182113


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1821_182158

-- Problem 1
theorem problem_one : Real.sqrt 9 - (-2023)^(0 : ℤ) + 2^(-1 : ℤ) = 5/2 := by sorry

-- Problem 2
theorem problem_two (a b : ℝ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2*b)) = 2 / (a + b) := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1821_182158


namespace NUMINAMATH_CALUDE_triangle_inequality_l1821_182192

-- Define the points and line segments
variable (A B C D E : ℝ × ℝ)
variable (a b c d : ℝ)

-- Conditions
axiom distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E
axiom points_order : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1 ∧ D.1 < E.1
axiom segment_lengths : 
  dist A B = a ∧ dist B D = b ∧ dist D E = c ∧ dist E C = d

-- Theorem to prove
theorem triangle_inequality (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  b < a + c := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1821_182192


namespace NUMINAMATH_CALUDE_cookie_distribution_l1821_182101

theorem cookie_distribution (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : num_people = 5)
  (h2 : cookies_per_person = 7) : 
  num_people * cookies_per_person = 35 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1821_182101


namespace NUMINAMATH_CALUDE_division_expression_equality_l1821_182133

theorem division_expression_equality : 180 / (8 + 9 * 3 - 4) = 180 / 31 := by
  sorry

end NUMINAMATH_CALUDE_division_expression_equality_l1821_182133


namespace NUMINAMATH_CALUDE_tilly_star_count_l1821_182182

theorem tilly_star_count (east_stars : ℕ) : 
  east_stars + 6 * east_stars = 840 → east_stars = 120 := by
  sorry

end NUMINAMATH_CALUDE_tilly_star_count_l1821_182182


namespace NUMINAMATH_CALUDE_association_sum_constant_l1821_182104

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- Represents a point on a segment -/
structure PointOnSegment where
  segment : Segment
  distanceFromMidpoint : ℝ

/-- The scheme associating points on AB with points on A'B' -/
def associationScheme (p : PointOnSegment) (p' : PointOnSegment) : Prop :=
  3 * p.distanceFromMidpoint - 2 * p'.distanceFromMidpoint = 6

theorem association_sum_constant 
  (ab : Segment)
  (a'b' : Segment)
  (p : PointOnSegment)
  (p' : PointOnSegment)
  (h1 : ab.length = 10)
  (h2 : a'b'.length = 18)
  (h3 : p.segment = ab)
  (h4 : p'.segment = a'b')
  (h5 : associationScheme p p') :
  p.distanceFromMidpoint + p'.distanceFromMidpoint = 12 :=
sorry

end NUMINAMATH_CALUDE_association_sum_constant_l1821_182104


namespace NUMINAMATH_CALUDE_locus_of_centers_l1821_182160

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being externally tangent to C₃ and internally tangent to C₄
def is_tangent_to_C₃_C₄ (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)

-- State the theorem
theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C₃_C₄ a b r) → a^2 + 18*b^2 - 6*a - 440 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1821_182160


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1821_182122

/-- Given that a^3 varies inversely with b^4, prove that if a = 5 when b = 2, then a = 5/2 when b = 4 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (h : ∃ k : ℝ, ∀ a b, a^3 * b^4 = k) :
  (5^3 * 2^4 = a^3 * 4^4) → a = 5/2 := by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1821_182122


namespace NUMINAMATH_CALUDE_cube_root_square_l1821_182118

theorem cube_root_square (x : ℝ) : (x + 5) ^ (1/3 : ℝ) = 3 → (x + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l1821_182118


namespace NUMINAMATH_CALUDE_min_value_theorem_l1821_182187

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1821_182187


namespace NUMINAMATH_CALUDE_x_values_l1821_182157

theorem x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1821_182157


namespace NUMINAMATH_CALUDE_total_triangles_is_twenty_l1821_182199

/-- A rectangle with diagonals and midpoint segments. -/
structure RectangleWithDiagonals where
  /-- The rectangle has different length sides. -/
  different_sides : Bool
  /-- The diagonals intersect at the center. -/
  diagonals_intersect_center : Bool
  /-- Segments join midpoints of opposite sides. -/
  midpoint_segments : Bool

/-- Count the number of triangles in the rectangle configuration. -/
def count_triangles (r : RectangleWithDiagonals) : ℕ :=
  sorry

/-- Theorem stating that the total number of triangles is 20. -/
theorem total_triangles_is_twenty (r : RectangleWithDiagonals) 
  (h1 : r.different_sides = true)
  (h2 : r.diagonals_intersect_center = true)
  (h3 : r.midpoint_segments = true) : 
  count_triangles r = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_twenty_l1821_182199


namespace NUMINAMATH_CALUDE_triangle_properties_l1821_182110

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

def condition2 (t : Triangle) : Prop :=
  2 * t.a - t.c = 2 * t.b * Real.cos t.C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_b : t.b = 2 * Real.sqrt 3)
  (h_cond1 : condition1 t)
  (h_cond2 : condition2 t) :
  (∃ (area : ℝ), t.a = 2 → area = 2 * Real.sqrt 3) ∧
  (2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1821_182110


namespace NUMINAMATH_CALUDE_isabel_country_albums_l1821_182123

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 8

/-- Represents the number of pop albums bought -/
def pop_albums : ℕ := 5

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 72

/-- Represents the number of country albums bought -/
def country_albums : ℕ := (total_songs - pop_albums * songs_per_album) / songs_per_album

theorem isabel_country_albums : country_albums = 4 := by
  sorry

end NUMINAMATH_CALUDE_isabel_country_albums_l1821_182123


namespace NUMINAMATH_CALUDE_valid_numbers_l1821_182178

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 1 ∧ n % 5 = 3

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {13, 28, 43, 58, 73, 88} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1821_182178


namespace NUMINAMATH_CALUDE_tickets_spent_on_beanie_l1821_182161

theorem tickets_spent_on_beanie (initial_tickets : Real) (lost_tickets : Real) (remaining_tickets : Real)
  (h1 : initial_tickets = 49.0)
  (h2 : lost_tickets = 6.0)
  (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by sorry

end NUMINAMATH_CALUDE_tickets_spent_on_beanie_l1821_182161


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1821_182127

/-- Proves that given a car's speed in the first hour is 120 km/h and its average speed over two hours is 95 km/h, the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 120) 
  (h2 : average_speed = 95) : 
  (2 * average_speed - speed_first_hour = 70) := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l1821_182127


namespace NUMINAMATH_CALUDE_complex_pattern_cannot_be_formed_l1821_182167

-- Define the types of shapes
inductive Shape
| Triangle
| Square

-- Define the set of available pieces
def available_pieces : Multiset Shape :=
  Multiset.replicate 8 Shape.Triangle + Multiset.replicate 7 Shape.Square

-- Define the possible figures
inductive Figure
| LargeRectangle
| Triangle
| Square
| ComplexPattern
| LongNarrowRectangle

-- Define a function to check if a figure can be formed
def can_form_figure (pieces : Multiset Shape) (figure : Figure) : Prop :=
  match figure with
  | Figure.LargeRectangle => true
  | Figure.Triangle => true
  | Figure.Square => true
  | Figure.ComplexPattern => false
  | Figure.LongNarrowRectangle => true

-- Theorem statement
theorem complex_pattern_cannot_be_formed :
  ∀ (figure : Figure),
    figure ≠ Figure.ComplexPattern ↔ can_form_figure available_pieces figure :=
by sorry

end NUMINAMATH_CALUDE_complex_pattern_cannot_be_formed_l1821_182167


namespace NUMINAMATH_CALUDE_pascal_triangle_distinct_elements_l1821_182168

theorem pascal_triangle_distinct_elements :
  ∃ (n : ℕ) (k l m p : ℕ),
    0 < k ∧ k < l ∧ l < m ∧ m < p ∧ p < n ∧
    2 * (n.choose k) = n.choose l ∧
    2 * (n.choose m) = n.choose p ∧
    (n.choose k) ≠ (n.choose l) ∧
    (n.choose k) ≠ (n.choose m) ∧
    (n.choose k) ≠ (n.choose p) ∧
    (n.choose l) ≠ (n.choose m) ∧
    (n.choose l) ≠ (n.choose p) ∧
    (n.choose m) ≠ (n.choose p) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_distinct_elements_l1821_182168


namespace NUMINAMATH_CALUDE_multiply_63_57_l1821_182165

theorem multiply_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_57_l1821_182165


namespace NUMINAMATH_CALUDE_inequality_solution_l1821_182152

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 5) ∧ x / (3 * x - 5) < 9 ↔ x > 45 / 26 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1821_182152


namespace NUMINAMATH_CALUDE_fraction_addition_l1821_182163

theorem fraction_addition : (168 : ℚ) / 240 + 100 / 150 = 41 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1821_182163


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1821_182109

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2345 [MOD 26]) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1821_182109


namespace NUMINAMATH_CALUDE_smallest_power_congruence_l1821_182155

theorem smallest_power_congruence (h : 2015 = 5 * 13 * 31) :
  (∃ n : ℕ, n > 0 ∧ 2^n ≡ 1 [ZMOD 2015]) ∧
  (∀ m : ℕ, m > 0 ∧ 2^m ≡ 1 [ZMOD 2015] → m ≥ 60) ∧
  2^60 ≡ 1 [ZMOD 2015] := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_congruence_l1821_182155


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1821_182119

theorem quadratic_equation_solution : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 6*x₁ + 8 = 0 ∧ 
  x₂^2 - 6*x₂ + 8 = 0 ∧ 
  x₁ = 2 ∧ 
  x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1821_182119


namespace NUMINAMATH_CALUDE_camryn_trumpet_practice_l1821_182105

/-- Represents the number of days between Camryn's practices for each instrument -/
structure PracticeSchedule where
  trumpet : ℕ
  flute : ℕ

/-- Checks if the practice schedule satisfies the given conditions -/
def is_valid_schedule (schedule : PracticeSchedule) : Prop :=
  schedule.flute = 3 ∧
  schedule.trumpet > 1 ∧
  schedule.trumpet < 33 ∧
  Nat.lcm schedule.trumpet schedule.flute = 33

theorem camryn_trumpet_practice (schedule : PracticeSchedule) :
  is_valid_schedule schedule → schedule.trumpet = 11 := by
  sorry

#check camryn_trumpet_practice

end NUMINAMATH_CALUDE_camryn_trumpet_practice_l1821_182105


namespace NUMINAMATH_CALUDE_max_value_on_edge_l1821_182111

/-- A 2D grid represented as a function from pairs of integers to real numbers. -/
def Grid := ℤ × ℤ → ℝ

/-- Predicate to check if a cell is on the edge of the grid. -/
def isOnEdge (m n : ℕ) (i j : ℤ) : Prop :=
  i = 0 ∨ i = m - 1 ∨ j = 0 ∨ j = n - 1

/-- The set of valid coordinates in an m × n grid. -/
def validCoords (m n : ℕ) : Set (ℤ × ℤ) :=
  {(i, j) | 0 ≤ i ∧ i < m ∧ 0 ≤ j ∧ j < n}

/-- Predicate to check if a grid satisfies the arithmetic mean property. -/
def satisfiesArithmeticMeanProperty (g : Grid) (m n : ℕ) : Prop :=
  ∀ (i j : ℤ), (i, j) ∈ validCoords m n → ¬isOnEdge m n i j →
    g (i, j) = (g (i-1, j) + g (i+1, j) + g (i, j-1) + g (i, j+1)) / 4

/-- Theorem: The maximum value in a grid satisfying the arithmetic mean property
    must be on the edge. -/
theorem max_value_on_edge (g : Grid) (m n : ℕ) 
    (h_mean : satisfiesArithmeticMeanProperty g m n)
    (h_distinct : ∀ (i j k l : ℤ), (i, j) ≠ (k, l) → 
      (i, j) ∈ validCoords m n → (k, l) ∈ validCoords m n → g (i, j) ≠ g (k, l))
    (h_finite : m > 0 ∧ n > 0) :
    ∃ (i j : ℤ), (i, j) ∈ validCoords m n ∧ isOnEdge m n i j ∧
      ∀ (k l : ℤ), (k, l) ∈ validCoords m n → g (i, j) ≥ g (k, l) :=
  sorry

end NUMINAMATH_CALUDE_max_value_on_edge_l1821_182111


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1821_182102

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B → C) ∧ (C → B))
variable (h3 : (C → D) ∧ ¬(D → C))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l1821_182102


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1821_182134

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 105 / 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l1821_182134


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_explicit_l1821_182150

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 3)⁻¹ + (y + 3)⁻¹ = 1/4 → a + 3*b ≤ x + 3*y :=
by sorry

theorem min_value_explicit (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 3)⁻¹ + (b + 3)⁻¹ = 1/4) : 
  a + 3*b = 4 + 8*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_explicit_l1821_182150


namespace NUMINAMATH_CALUDE_vector_operation_l1821_182164

def vector_a : ℝ × ℝ × ℝ := (2, 0, -1)
def vector_b : ℝ × ℝ × ℝ := (0, 1, -2)

theorem vector_operation :
  (2 : ℝ) • vector_a - vector_b = (4, -1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1821_182164


namespace NUMINAMATH_CALUDE_average_problem_l1821_182140

theorem average_problem (x : ℝ) : (0.4 + x) / 2 = 0.2025 → x = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1821_182140


namespace NUMINAMATH_CALUDE_mentoring_program_fraction_l1821_182126

theorem mentoring_program_fraction (total : ℕ) (s : ℕ) (n : ℕ) : 
  total = s + n →
  s > 0 →
  n > 0 →
  (n : ℚ) / 4 = (s : ℚ) / 3 →
  ((n : ℚ) / 4 + (s : ℚ) / 3) / total = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_mentoring_program_fraction_l1821_182126


namespace NUMINAMATH_CALUDE_scalene_triangle_area_l1821_182107

/-- Given an outer triangle enclosing a regular hexagon, prove the area of one scalene triangle -/
theorem scalene_triangle_area 
  (outer_triangle_area : ℝ) 
  (hexagon_area : ℝ) 
  (num_scalene_triangles : ℕ)
  (h1 : outer_triangle_area = 25)
  (h2 : hexagon_area = 4)
  (h3 : num_scalene_triangles = 6) :
  (outer_triangle_area - hexagon_area) / num_scalene_triangles = 3.5 := by
sorry

end NUMINAMATH_CALUDE_scalene_triangle_area_l1821_182107


namespace NUMINAMATH_CALUDE_line_intersection_l1821_182136

theorem line_intersection :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x - y = -5) ∧ (x = 15/38) ∧ (y = 140/19) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1821_182136


namespace NUMINAMATH_CALUDE_balloon_distribution_l1821_182193

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (remaining_balloons : ℕ) : 
  total_balloons = 243 → 
  num_friends = 10 → 
  remaining_balloons = total_balloons % num_friends → 
  remaining_balloons = 3 := by
sorry

end NUMINAMATH_CALUDE_balloon_distribution_l1821_182193


namespace NUMINAMATH_CALUDE_committee_selection_l1821_182196

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) : 
  Nat.choose n k = 792 :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_l1821_182196


namespace NUMINAMATH_CALUDE_min_value_expression_l1821_182176

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (a / (b + 1)) ≥ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1821_182176


namespace NUMINAMATH_CALUDE_oarsmen_count_l1821_182156

theorem oarsmen_count (weight_old : ℝ) (weight_new : ℝ) (avg_increase : ℝ) :
  weight_old = 53 →
  weight_new = 71 →
  avg_increase = 1.8 →
  ∃ n : ℕ, n > 0 ∧ n * avg_increase = weight_new - weight_old :=
by
  sorry

end NUMINAMATH_CALUDE_oarsmen_count_l1821_182156


namespace NUMINAMATH_CALUDE_terms_before_zero_l1821_182149

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem terms_before_zero (a : ℤ) (d : ℤ) (h1 : a = 102) (h2 : d = -6) :
  ∃ n : ℕ, n = 17 ∧ arithmetic_sequence a d (n + 1) = 0 :=
sorry

end NUMINAMATH_CALUDE_terms_before_zero_l1821_182149


namespace NUMINAMATH_CALUDE_sum_of_digits_1729_base_8_l1821_182191

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits of 1729 in base 8 is equal to 7 -/
theorem sum_of_digits_1729_base_8 :
  sumDigits (toBase8 1729) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1729_base_8_l1821_182191


namespace NUMINAMATH_CALUDE_modulus_of_complex_l1821_182181

theorem modulus_of_complex (i : ℂ) (a : ℝ) : 
  i * i = -1 →
  (1 - i) * (1 - a * i) ∈ {z : ℂ | z.re = 0} →
  Complex.abs (1 - a * i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l1821_182181


namespace NUMINAMATH_CALUDE_rachel_research_time_l1821_182188

/-- Represents the time spent on different activities while writing an essay -/
structure EssayTime where
  writing_speed : ℕ  -- pages per 30 minutes
  total_pages : ℕ
  editing_time : ℕ  -- in minutes
  total_time : ℕ    -- in minutes

/-- Calculates the time spent researching for an essay -/
def research_time (e : EssayTime) : ℕ :=
  e.total_time - (e.total_pages * 30 + e.editing_time)

/-- Theorem stating that Rachel spent 45 minutes researching -/
theorem rachel_research_time :
  let e : EssayTime := {
    writing_speed := 1,
    total_pages := 6,
    editing_time := 75,
    total_time := 300
  }
  research_time e = 45 := by sorry

end NUMINAMATH_CALUDE_rachel_research_time_l1821_182188


namespace NUMINAMATH_CALUDE_weight_of_b_l1821_182162

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l1821_182162


namespace NUMINAMATH_CALUDE_sum_of_100th_bracket_l1821_182197

def sequence_start : ℕ := 3

def cycle_length : ℕ := 4

def numbers_per_cycle : ℕ := 10

def target_bracket : ℕ := 100

theorem sum_of_100th_bracket :
  let total_numbers := (target_bracket - 1) / cycle_length * numbers_per_cycle
  let last_number := sequence_start + 2 * (total_numbers - 1)
  let bracket_numbers := [last_number - 6, last_number - 4, last_number - 2, last_number]
  List.sum bracket_numbers = 1992 := by
sorry

end NUMINAMATH_CALUDE_sum_of_100th_bracket_l1821_182197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l1821_182106

/-- The smallest positive integer n such that 2009 - 7n < 0 -/
def smallest_n : ℕ := 288

theorem arithmetic_sequence_smallest_negative (n : ℕ) :
  n ≥ smallest_n ↔ 2009 - 7 * n < 0 :=
by
  sorry

theorem smallest_n_is_minimal :
  ∀ k : ℕ, k < smallest_n → 2009 - 7 * k ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l1821_182106


namespace NUMINAMATH_CALUDE_even_function_extension_l1821_182166

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- The given function f defined for x ≤ 0 -/
def f_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 0 → f x = x^2 - 2*x

theorem even_function_extension :
  ∀ f : ℝ → ℝ, EvenFunction f → f_nonpositive f →
  ∀ x : ℝ, x > 0 → f x = x^2 + 2*x :=
sorry

end NUMINAMATH_CALUDE_even_function_extension_l1821_182166


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1821_182143

/-- The lateral area of a cylinder with a rectangular front view of area 6 -/
theorem cylinder_lateral_area (h : ℝ) (h_pos : h > 0) : 
  let d := 6 / h
  π * d * h = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1821_182143


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1821_182195

theorem binomial_expansion_sum (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (a - x)^8 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1821_182195


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l1821_182112

theorem cliffs_rock_collection (sedimentary : ℕ) (igneous : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * (igneous / 3) = 40 →
  sedimentary + igneous = 180 := by
sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l1821_182112


namespace NUMINAMATH_CALUDE_diagonals_sum_bounds_l1821_182180

/-- A convex pentagon in a 2D plane -/
structure ConvexPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  convex : Bool

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the perimeter of the pentagon -/
def perimeter (p : ConvexPentagon) : ℝ :=
  distance p.A p.B + distance p.B p.C + distance p.C p.D + distance p.D p.E + distance p.E p.A

/-- Calculate the sum of diagonals of the pentagon -/
def sumDiagonals (p : ConvexPentagon) : ℝ :=
  distance p.A p.C + distance p.B p.D + distance p.C p.E + distance p.D p.A + distance p.B p.E

/-- Theorem: The sum of diagonals is greater than the perimeter but less than twice the perimeter -/
theorem diagonals_sum_bounds (p : ConvexPentagon) (h : p.convex = true) :
  perimeter p < sumDiagonals p ∧ sumDiagonals p < 2 * perimeter p := by sorry

end NUMINAMATH_CALUDE_diagonals_sum_bounds_l1821_182180


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_l1821_182125

/-- Proves that given an original salary of 5000, an initial increase of 10%,
    and a final salary of 5225, the percentage decrease after the initial increase is 5%. -/
theorem salary_decrease_percentage
  (original_salary : ℝ)
  (initial_increase_percentage : ℝ)
  (final_salary : ℝ)
  (h1 : original_salary = 5000)
  (h2 : initial_increase_percentage = 10)
  (h3 : final_salary = 5225)
  : ∃ (decrease_percentage : ℝ),
    decrease_percentage = 5 ∧
    final_salary = original_salary * (1 + initial_increase_percentage / 100) * (1 - decrease_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_l1821_182125


namespace NUMINAMATH_CALUDE_find_x_l1821_182172

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(2*b) = a^b * x^b → x = a^3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1821_182172


namespace NUMINAMATH_CALUDE_correct_practice_times_l1821_182145

/-- Represents the practice schedule and time spent on instruments in a month -/
structure PracticeSchedule where
  piano_daily_minutes : ℕ
  violin_daily_minutes : ℕ
  flute_daily_minutes : ℕ
  violin_days_per_week : ℕ
  flute_days_per_week : ℕ
  weeks_with_6_days : ℕ
  weeks_with_7_days : ℕ

/-- Calculates the total practice time for each instrument in the given month -/
def calculate_practice_time (schedule : PracticeSchedule) :
  ℕ × ℕ × ℕ :=
  let total_days := schedule.weeks_with_6_days * 6 + schedule.weeks_with_7_days * 7
  let violin_total_days := schedule.violin_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  let flute_total_days := schedule.flute_days_per_week * (schedule.weeks_with_6_days + schedule.weeks_with_7_days)
  (schedule.piano_daily_minutes * total_days,
   schedule.violin_daily_minutes * violin_total_days,
   schedule.flute_daily_minutes * flute_total_days)

/-- Theorem stating the correct practice times for each instrument -/
theorem correct_practice_times (schedule : PracticeSchedule)
  (h1 : schedule.piano_daily_minutes = 25)
  (h2 : schedule.violin_daily_minutes = 3 * schedule.piano_daily_minutes)
  (h3 : schedule.flute_daily_minutes = schedule.violin_daily_minutes / 2)
  (h4 : schedule.violin_days_per_week = 5)
  (h5 : schedule.flute_days_per_week = 4)
  (h6 : schedule.weeks_with_6_days = 2)
  (h7 : schedule.weeks_with_7_days = 2) :
  calculate_practice_time schedule = (650, 1500, 600) :=
sorry


end NUMINAMATH_CALUDE_correct_practice_times_l1821_182145


namespace NUMINAMATH_CALUDE_car_wash_earnings_l1821_182184

theorem car_wash_earnings :
  ∀ (total lisa tommy : ℝ),
    lisa = total / 2 →
    tommy = lisa / 2 →
    lisa = tommy + 15 →
    total = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_wash_earnings_l1821_182184


namespace NUMINAMATH_CALUDE_inequality_holds_iff_in_interval_l1821_182135

/-- For fixed positive real numbers a and b, the inequality 
    (1/√x) + (1/√(a+b-x)) < (1/√a) + (1/√b) 
    holds if and only if x is in the open interval (min(a, b), max(a, b)) -/
theorem inequality_holds_iff_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, (1 / Real.sqrt x + 1 / Real.sqrt (a + b - x) < 1 / Real.sqrt a + 1 / Real.sqrt b) ↔ 
    (min a b < x ∧ x < max a b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_in_interval_l1821_182135


namespace NUMINAMATH_CALUDE_next_coincidence_correct_l1821_182177

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Checks if hour and minute hands coincide at given time -/
def handsCoincide (t : Time) : Prop :=
  (t.hours % 12 * 60 + t.minutes) * 11 = t.minutes * 12

/-- The next time after midnight when clock hands coincide -/
def nextCoincidence : Time :=
  { hours := 1, minutes := 5, seconds := 27 }

theorem next_coincidence_correct :
  handsCoincide nextCoincidence ∧
  ∀ t : Time, t.toSeconds < nextCoincidence.toSeconds → ¬handsCoincide t :=
by sorry

end NUMINAMATH_CALUDE_next_coincidence_correct_l1821_182177


namespace NUMINAMATH_CALUDE_expression_evaluation_l1821_182194

theorem expression_evaluation : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1821_182194


namespace NUMINAMATH_CALUDE_g_composite_three_roots_l1821_182171

/-- The function g(x) defined as x^2 - 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

/-- The composite function g(g(x)) -/
def g_composite (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ 
    (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧
    (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 3 -/
theorem g_composite_three_roots (d : ℝ) :
  has_exactly_three_distinct_real_roots (g_composite d) ↔ d = 3 :=
sorry

end NUMINAMATH_CALUDE_g_composite_three_roots_l1821_182171


namespace NUMINAMATH_CALUDE_total_students_in_high_school_l1821_182169

/-- Proves that the total number of students in a high school is 500 given the number of students in different course combinations. -/
theorem total_students_in_high_school : 
  ∀ (music art both neither : ℕ),
    music = 30 →
    art = 10 →
    both = 10 →
    neither = 470 →
    music + art - both + neither = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_in_high_school_l1821_182169


namespace NUMINAMATH_CALUDE_book_distribution_ways_book_distribution_proof_l1821_182117

/-- The number of ways to distribute 6 different books to 3 people, 
    where one person gets 1 book, another gets 2 books, 
    and the last gets 3 books, in any order. -/
theorem book_distribution_ways : ℕ := by
  sorry

/-- The number of ways to choose 1 book from 6 books -/
def choose_one_from_six : ℕ := 6

/-- The number of ways to choose 2 books from 5 books -/
def choose_two_from_five : ℕ := 10

/-- The number of ways to arrange 3 people -/
def arrange_three_people : ℕ := 6

theorem book_distribution_proof : 
  book_distribution_ways = choose_one_from_six * choose_two_from_five * arrange_three_people := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_book_distribution_proof_l1821_182117


namespace NUMINAMATH_CALUDE_gcd_12345_54321_l1821_182147

theorem gcd_12345_54321 : Nat.gcd 12345 54321 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_54321_l1821_182147


namespace NUMINAMATH_CALUDE_oplus_five_two_l1821_182183

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 3 * a + 4 * b

-- State the theorem
theorem oplus_five_two : oplus 5 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_oplus_five_two_l1821_182183


namespace NUMINAMATH_CALUDE_function_shift_and_value_l1821_182179

theorem function_shift_and_value (φ : Real) 
  (h1 : 0 < φ ∧ φ < π / 2) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x, f x = 2 * Real.sin (x + φ)) 
  (g : ℝ → ℝ) 
  (h3 : ∀ x, g x = f (x + π / 3)) 
  (h4 : ∀ x, g x = g (-x)) : 
  f (π / 6) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_shift_and_value_l1821_182179


namespace NUMINAMATH_CALUDE_apartment_number_theorem_l1821_182148

/-- The number of apartments on each floor (actual) -/
def apartments_per_floor : ℕ := 7

/-- The number of apartments Anna initially thought were on each floor -/
def assumed_apartments_per_floor : ℕ := 6

/-- The floor number where Anna's apartment is located -/
def target_floor : ℕ := 4

/-- The set of possible apartment numbers on the target floor when there are 6 apartments per floor -/
def apartment_numbers_6 : Set ℕ := Set.Icc ((target_floor - 1) * assumed_apartments_per_floor + 1) (target_floor * assumed_apartments_per_floor)

/-- The set of possible apartment numbers on the target floor when there are 7 apartments per floor -/
def apartment_numbers_7 : Set ℕ := Set.Icc ((target_floor - 1) * apartments_per_floor + 1) (target_floor * apartments_per_floor)

/-- The set of apartment numbers that exist in both scenarios -/
def possible_apartment_numbers : Set ℕ := apartment_numbers_6 ∩ apartment_numbers_7

theorem apartment_number_theorem : possible_apartment_numbers = {22, 23, 24} := by
  sorry

end NUMINAMATH_CALUDE_apartment_number_theorem_l1821_182148


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l1821_182153

theorem triangle_side_sum_max (a b c : ℝ) (A : ℝ) :
  a = 4 → A = π / 3 → b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l1821_182153


namespace NUMINAMATH_CALUDE_unique_solution_l1821_182137

def A (a b : ℝ) := {x : ℝ | x^2 + a*x + b = 0}
def B (c : ℝ) := {x : ℝ | x^2 + c*x + 15 = 0}

theorem unique_solution :
  ∃! (a b c : ℝ),
    (A a b ∪ B c = {3, 5}) ∧
    (A a b ∩ B c = {3}) ∧
    a = -6 ∧ b = 9 ∧ c = -8 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1821_182137


namespace NUMINAMATH_CALUDE_derivative_x_squared_over_x_plus_three_l1821_182142

/-- The derivative of x^2 / (x+3) is (x^2 + 6x) / (x+3)^2 -/
theorem derivative_x_squared_over_x_plus_three (x : ℝ) :
  deriv (fun x => x^2 / (x + 3)) x = (x^2 + 6*x) / (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_squared_over_x_plus_three_l1821_182142


namespace NUMINAMATH_CALUDE_set_B_equals_l1821_182154

def U : Set Nat := {1, 3, 5, 7, 9}

theorem set_B_equals (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : A ∩ B = {1, 3})
  (h4 : (U \ A) ∩ B = {5}) :
  B = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equals_l1821_182154


namespace NUMINAMATH_CALUDE_factorization_problems_l1821_182173

theorem factorization_problems :
  (∀ a : ℝ, a^3 - 4*a = a*(a+2)*(a-2)) ∧
  (∀ m x y : ℝ, 3*m*x^2 - 6*m*x*y + 3*m*y^2 = 3*m*(x-y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1821_182173


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1821_182121

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1821_182121


namespace NUMINAMATH_CALUDE_a_2012_value_l1821_182124

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => a n / (1 + a n)

-- State the theorem
theorem a_2012_value : a 2012 = 2 / 4025 := by
  sorry

end NUMINAMATH_CALUDE_a_2012_value_l1821_182124


namespace NUMINAMATH_CALUDE_investment_sum_l1821_182141

/-- Given a sum of money invested for 2 years, if increasing the interest rate by 3% 
    results in 300 more rupees of interest, then the original sum invested must be 5000 rupees. -/
theorem investment_sum (P : ℝ) (R : ℝ) : 
  (P * (R + 3) * 2) / 100 = (P * R * 2) / 100 + 300 → P = 5000 := by
sorry

end NUMINAMATH_CALUDE_investment_sum_l1821_182141


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1821_182174

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + x + b

-- Define the solution set of the original inequality
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the new quadratic function
def g (c x : ℝ) := x^2 - (c - 2) * x - 2 * c

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, (∀ x : ℝ, f a b x > 0 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → {x : ℝ | g c x < 0} = ∅) ∧
    (c > -2 → {x : ℝ | g c x < 0} = Set.Ioo (-2) c) ∧
    (c < -2 → {x : ℝ | g c x < 0} = Set.Ioo c (-2))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1821_182174


namespace NUMINAMATH_CALUDE_number_problem_l1821_182186

theorem number_problem (x : ℝ) : (1/3) * x - 5 = 10 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1821_182186


namespace NUMINAMATH_CALUDE_acute_triangle_existence_l1821_182103

theorem acute_triangle_existence (n : ℕ) (hn : n ≥ 13) :
  ∀ (a : Fin n → ℝ), 
  (∀ i, a i > 0) →
  (∀ i j, a i ≤ n * a j) →
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i^2 + a j^2 > a k^2 ∧
    a i^2 + a k^2 > a j^2 ∧
    a j^2 + a k^2 > a i^2 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_existence_l1821_182103


namespace NUMINAMATH_CALUDE_bathing_suits_for_women_l1821_182139

theorem bathing_suits_for_women (total : ℕ) (men : ℕ) (women : ℕ) : 
  total = 19766 → men = 14797 → women = total - men → women = 4969 := by
sorry

end NUMINAMATH_CALUDE_bathing_suits_for_women_l1821_182139


namespace NUMINAMATH_CALUDE_computer_device_properties_l1821_182130

theorem computer_device_properties :
  ∃ f : ℕ → ℕ → ℕ,
    (f 1 1 = 1) ∧
    (∀ m n : ℕ, f m (n + 1) = f m n + 2) ∧
    (∀ m : ℕ, f (m + 1) 1 = 2 * f m 1) ∧
    (∀ n : ℕ, f 1 n = 2 * n - 1) ∧
    (∀ m : ℕ, f m 1 = 2^(m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_computer_device_properties_l1821_182130


namespace NUMINAMATH_CALUDE_count_integer_pairs_l1821_182114

theorem count_integer_pairs : 
  ∃ (count : ℕ), 
    (2^2876 < 3^1250 ∧ 3^1250 < 2^2877) →
    count = (Finset.filter 
      (λ (pair : ℕ × ℕ) => 
        let (m, n) := pair
        1 ≤ m ∧ m ≤ 2875 ∧ 3^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 3^(n+1))
      (Finset.product (Finset.range 2876) (Finset.range (1250 + 1)))).card ∧
    count = 3750 :=
by sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l1821_182114


namespace NUMINAMATH_CALUDE_mens_wages_l1821_182198

/-- Proves that given the conditions in the problem, the total wages for 9 men is Rs. 72 -/
theorem mens_wages (total_earnings : ℕ) (num_men num_boys : ℕ) (W : ℕ) :
  total_earnings = 216 →
  num_men = 9 →
  num_boys = 7 →
  num_men * W = num_men * num_boys →
  (3 * num_men : ℕ) * (total_earnings / (3 * num_men)) = 72 :=
by sorry

end NUMINAMATH_CALUDE_mens_wages_l1821_182198


namespace NUMINAMATH_CALUDE_distance_to_focus_is_six_l1821_182129

/-- A parabola with equation y^2 = 4x -/
structure Parabola where
  equation : ∀ x y, y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

theorem distance_to_focus_is_six (p : Parabola) (P : PointOnParabola p) 
  (h : |P.x - (-3)| = 5) : 
  Real.sqrt ((P.x - focus.1)^2 + (P.y - focus.2)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_six_l1821_182129


namespace NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l1821_182108

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4,
    red_cards := 26,
    black_cards := 26 }

/-- The number of Kings in a standard deck -/
def num_kings (d : Deck) : Nat := d.suits

/-- The number of Queens in a standard deck -/
def num_queens (d : Deck) : Nat := d.suits

/-- The probability of drawing a King then a Queen from a shuffled deck -/
def prob_king_queen (d : Deck) : Rat :=
  (num_kings d * num_queens d) / (d.cards * (d.cards - 1))

/-- Theorem: The probability of drawing a King then a Queen from a standard 52-card deck is 4/663 -/
theorem prob_king_queen_standard_deck :
  prob_king_queen standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l1821_182108


namespace NUMINAMATH_CALUDE_justin_and_tim_same_game_l1821_182120

def total_players : ℕ := 12
def players_per_game : ℕ := 6

theorem justin_and_tim_same_game :
  let total_combinations := Nat.choose total_players players_per_game
  let games_with_justin_and_tim := Nat.choose (total_players - 2) (players_per_game - 2)
  games_with_justin_and_tim = 210 := by
  sorry

end NUMINAMATH_CALUDE_justin_and_tim_same_game_l1821_182120


namespace NUMINAMATH_CALUDE_product_ab_equals_ten_l1821_182175

theorem product_ab_equals_ten (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 21) : 
  a * b = 10 := by
sorry

end NUMINAMATH_CALUDE_product_ab_equals_ten_l1821_182175
