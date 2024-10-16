import Mathlib

namespace NUMINAMATH_CALUDE_not_clear_def_not_set_l3241_324148

-- Define what it means for a collection to have a clear definition
def has_clear_definition (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x) ∨ (¬C x)

-- Define what it means to be a set
def is_set (S : Type → Prop) : Prop :=
  has_clear_definition S

-- Theorem: A collection without a clear definition is not a set
theorem not_clear_def_not_set (C : Type → Prop) :
  ¬(has_clear_definition C) → ¬(is_set C) := by
  sorry

end NUMINAMATH_CALUDE_not_clear_def_not_set_l3241_324148


namespace NUMINAMATH_CALUDE_circumcircle_equation_l3241_324176

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define an equilateral triangle on the parabola
def equilateral_triangle_on_parabola (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  parabola O.1 O.2 ∧ parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) :
  equilateral_triangle_on_parabola A B →
  ∃ x y : ℝ, (x - 4)^2 + y^2 = 16 ∧
            (x - 0)^2 + (y - 0)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
            (x - 0)^2 + (y - 0)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l3241_324176


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l3241_324156

def line_equation (x y : ℝ) : ℝ := 2 * x + y - 3

theorem opposite_sides_of_line :
  (line_equation 0 0 < 0) ∧ (line_equation 2 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l3241_324156


namespace NUMINAMATH_CALUDE_total_people_on_large_seats_l3241_324194

/-- The number of large seats on the Ferris wheel -/
def large_seats : ℕ := 7

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := 12

/-- Theorem: The total number of people who can ride on large seats is 84 -/
theorem total_people_on_large_seats : large_seats * people_per_large_seat = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_people_on_large_seats_l3241_324194


namespace NUMINAMATH_CALUDE_f_minimum_value_range_l3241_324134

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * (x + 1)

theorem f_minimum_value_range (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀ ∧ f a x₀ > a^2 + a) ↔ -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_range_l3241_324134


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3241_324198

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 - 5*y^2 = 1 →
  ∃ n : ℕ, (x + y * Real.sqrt 5 = (9 + 4 * Real.sqrt 5)^n) ∨
           (x + y * Real.sqrt 5 = -(9 + 4 * Real.sqrt 5)^n) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3241_324198


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3241_324118

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3241_324118


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3241_324136

/-- Represents the number of magical herbs -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical stones -/
def num_stones : ℕ := 6

/-- Represents the number of herbs incompatible with one specific stone -/
def incompatible_herbs : ℕ := 3

/-- Calculates the number of valid combinations for the wizard's elixir -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

/-- Proves that the number of valid combinations for the wizard's elixir is 21 -/
theorem wizard_elixir_combinations : valid_combinations = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3241_324136


namespace NUMINAMATH_CALUDE_circle_rotation_l3241_324167

theorem circle_rotation (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = 11 * Real.pi) :
  (d / (2 * Real.pi * r)) % 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_rotation_l3241_324167


namespace NUMINAMATH_CALUDE_min_rubber_bands_specific_l3241_324124

/-- Calculates the minimum number of rubber bands needed to tie matches and cotton swabs into bundles. -/
def min_rubber_bands (total_matches : ℕ) (total_swabs : ℕ) (matches_per_bundle : ℕ) (swabs_per_bundle : ℕ) (bands_per_bundle : ℕ) : ℕ :=
  let match_bundles := total_matches / matches_per_bundle
  let swab_bundles := total_swabs / swabs_per_bundle
  (match_bundles + swab_bundles) * bands_per_bundle

/-- Theorem stating that given the specific conditions, the minimum number of rubber bands needed is 14. -/
theorem min_rubber_bands_specific : 
  min_rubber_bands 40 34 8 12 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_rubber_bands_specific_l3241_324124


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3241_324163

theorem purely_imaginary_complex_fraction (a b : ℝ) (h : b ≠ 0) :
  (∃ (k : ℝ), (Complex.I : ℂ) * k = (a + Complex.I * b) / (4 + Complex.I * 3)) →
  a / b = -3/4 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l3241_324163


namespace NUMINAMATH_CALUDE_album_jumps_l3241_324197

/-- Calculates the number of jumps a person can do while listening to an album --/
theorem album_jumps (jumps_per_second : ℝ) (num_songs : ℕ) (song_length_minutes : ℝ) : 
  jumps_per_second = 1 →
  num_songs = 10 →
  song_length_minutes = 3.5 →
  (jumps_per_second * num_songs * song_length_minutes * 60 : ℝ) = 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_album_jumps_l3241_324197


namespace NUMINAMATH_CALUDE_problem_solution_l3241_324117

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : a ≠ 0)
  (h4 : c * d = 1) :
  2*a + 2*b + (a + b - 3*c*d) - m = 2 ∨ 2*a + 2*b + (a + b - 3*c*d) - m = -6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3241_324117


namespace NUMINAMATH_CALUDE_louies_previous_goals_correct_l3241_324160

/-- Calculates the number of goals Louie scored in previous matches before the last match -/
def louies_previous_goals (
  louies_last_match_goals : ℕ
  ) (
  brother_seasons : ℕ
  ) (
  games_per_season : ℕ
  ) (
  total_goals : ℕ
  ) : ℕ := by
  sorry

theorem louies_previous_goals_correct :
  louies_previous_goals 4 3 50 1244 = 40 := by
  sorry

end NUMINAMATH_CALUDE_louies_previous_goals_correct_l3241_324160


namespace NUMINAMATH_CALUDE_parabola_solution_l3241_324120

/-- Parabola intersecting x-axis at two points -/
structure Parabola where
  a : ℝ
  intersectionA : ℝ × ℝ
  intersectionB : ℝ × ℝ

/-- The parabola y = a(x+1)^2 + 2 intersects the x-axis at A(-3, 0) and B -/
def parabola_problem (p : Parabola) : Prop :=
  p.intersectionA = (-3, 0) ∧
  p.intersectionA.2 = p.a * (p.intersectionA.1 + 1)^2 + 2 ∧
  p.intersectionB.2 = p.a * (p.intersectionB.1 + 1)^2 + 2 ∧
  p.intersectionA.2 = 0 ∧
  p.intersectionB.2 = 0

theorem parabola_solution (p : Parabola) (h : parabola_problem p) :
  p.a = -1/2 ∧ p.intersectionB = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_solution_l3241_324120


namespace NUMINAMATH_CALUDE_stone_width_is_5dm_l3241_324199

-- Define the given parameters
def hall_length : Real := 36
def hall_width : Real := 15
def stone_length : Real := 0.2  -- 2 dm = 0.2 m
def num_stones : Nat := 5400

-- Define the theorem
theorem stone_width_is_5dm :
  ∃ (stone_width : Real),
    stone_width * 10 = 5 ∧  -- Convert to decimeters
    hall_length * hall_width = (stone_length * stone_width) * num_stones := by
  sorry

end NUMINAMATH_CALUDE_stone_width_is_5dm_l3241_324199


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3241_324182

theorem polynomial_evaluation : 
  let x : ℕ := 2
  (x^4 + x^3 + x^2 + x + 1 : ℕ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3241_324182


namespace NUMINAMATH_CALUDE_hex_lattice_equilateral_triangles_l3241_324158

/-- Represents a point in a 2D hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the hexagonal lattice -/
def HexagonalLattice : Type := List LatticePoint

/-- Calculates the distance between two points -/
def distance (p1 p2 : LatticePoint) : ℝ := sorry

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : LatticePoint) : Bool := sorry

/-- Counts the number of equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : Nat := sorry

/-- The hexagonal lattice with 7 points -/
def hexLattice : HexagonalLattice := sorry

theorem hex_lattice_equilateral_triangles :
  countEquilateralTriangles hexLattice = 6 := by sorry

end NUMINAMATH_CALUDE_hex_lattice_equilateral_triangles_l3241_324158


namespace NUMINAMATH_CALUDE_third_circle_radius_l3241_324155

/-- Given two externally tangent circles and a third circle tangent to both and their center line, 
    prove that the radius of the third circle is √46 - 5 -/
theorem third_circle_radius 
  (P Q R : ℝ × ℝ) -- Centers of the three circles
  (r : ℝ) -- Radius of the third circle
  (h1 : dist P Q = 10) -- Distance between centers of first two circles
  (h2 : dist P R = 3 + r) -- Distance from P to R
  (h3 : dist Q R = 7 + r) -- Distance from Q to R
  (h4 : (R.1 - P.1) * (Q.1 - P.1) + (R.2 - P.2) * (Q.2 - P.2) = 0) -- R is on the perpendicular bisector of PQ
  : r = Real.sqrt 46 - 5 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l3241_324155


namespace NUMINAMATH_CALUDE_tangent_lines_max_value_min_value_l3241_324144

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x - 9

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (x₀ y₀ : ℝ), 
    (f' x₀ = -9 ∧ y₀ = f x₀ ∧ (y₀ = -9*x₀ - 3 ∨ y₀ = -9*x₀ + 19)) :=
sorry

-- Theorem for maximum value
theorem max_value :
  ∃ (x : ℝ), f x = 24 ∧ ∀ y, f y ≤ f x :=
sorry

-- Theorem for minimum value
theorem min_value :
  ∃ (x : ℝ), f x = -8 ∧ ∀ y, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_max_value_min_value_l3241_324144


namespace NUMINAMATH_CALUDE_athlete_formation_problem_l3241_324172

theorem athlete_formation_problem :
  ∃ n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    (n + 4) % 10 = 0 ∧
    (n + 5) % 11 = 0 ∧
    n = 226 := by
  sorry

end NUMINAMATH_CALUDE_athlete_formation_problem_l3241_324172


namespace NUMINAMATH_CALUDE_avery_donation_l3241_324168

/-- The number of clothes Avery is donating -/
def total_clothes (shirts pants shorts : ℕ) : ℕ := shirts + pants + shorts

/-- Theorem stating the total number of clothes Avery is donating -/
theorem avery_donation :
  ∀ (shirts pants shorts : ℕ),
    shirts = 4 →
    pants = 2 * shirts →
    shorts = pants / 2 →
    total_clothes shirts pants shorts = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l3241_324168


namespace NUMINAMATH_CALUDE_sin_shift_left_l3241_324103

/-- Shifting a sinusoidal function to the left -/
theorem sin_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π / 6)
  g x = Real.sin (2 * x + π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_left_l3241_324103


namespace NUMINAMATH_CALUDE_complex_magnitude_one_minus_i_l3241_324122

theorem complex_magnitude_one_minus_i :
  Complex.abs (1 - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_minus_i_l3241_324122


namespace NUMINAMATH_CALUDE_triangle_max_third_side_l3241_324174

theorem triangle_max_third_side (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (c : ℕ), c = 14 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_third_side_l3241_324174


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3241_324162

def total_cars : ℕ := 300
def ford_percentage : ℚ := 20 / 100
def nissan_percentage : ℚ := 25 / 100
def volkswagen_percentage : ℚ := 10 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (ford_percentage + nissan_percentage + volkswagen_percentage)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3241_324162


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l3241_324132

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The clock displays '5' instead of '2' -/
  display_error : ℕ → ℕ
  display_error_def : ∀ n, display_error n = if n = 2 then 5 else n

/-- The fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  5/8

theorem error_clock_correct_time_fraction (clock : ErrorClock) :
  correct_time_fraction clock = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l3241_324132


namespace NUMINAMATH_CALUDE_pineapple_problem_l3241_324138

/-- Calculates the number of rotten pineapples given the initial count, sold count, and remaining fresh count. -/
def rottenPineapples (initial sold fresh : ℕ) : ℕ :=
  initial - sold - fresh

/-- Theorem stating that given the specific conditions from the problem, 
    the number of rotten pineapples thrown away is 9. -/
theorem pineapple_problem : rottenPineapples 86 48 29 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_problem_l3241_324138


namespace NUMINAMATH_CALUDE_kids_staying_home_l3241_324153

def total_kids : ℕ := 898051
def kids_at_camp : ℕ := 629424

theorem kids_staying_home : total_kids - kids_at_camp = 268627 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l3241_324153


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_l3241_324178

/-- The area of a region bounded by three identical circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5 -- radius of each arc
  let θ : ℝ := Real.pi / 2 -- central angle in radians (90 degrees)
  let segment_area : ℝ := r^2 * (θ - Real.sin θ) / 2 -- area of one circular segment
  let total_area : ℝ := 3 * segment_area -- area of the entire region
  total_area = (75 * Real.pi - 150) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_l3241_324178


namespace NUMINAMATH_CALUDE_two_digit_primes_with_rearranged_digits_and_square_difference_l3241_324125

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def digits_rearranged (a b : ℕ) : Prop :=
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem two_digit_primes_with_rearranged_digits_and_square_difference :
  ∀ a b : ℕ,
    is_two_digit_prime a ∧
    is_two_digit_prime b ∧
    digits_rearranged a b ∧
    is_perfect_square (a - b) →
    (a = 73 ∧ b = 37) ∨ (a = 37 ∧ b = 73) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_primes_with_rearranged_digits_and_square_difference_l3241_324125


namespace NUMINAMATH_CALUDE_expand_product_l3241_324100

theorem expand_product (x : ℝ) : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3241_324100


namespace NUMINAMATH_CALUDE_magic_sum_divisible_by_three_l3241_324157

/-- Represents a 3x3 magic square with integer entries -/
def MagicSquare : Type := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (m : MagicSquare) : ℤ :=
  m 0 0 + m 0 1 + m 0 2

/-- Predicate to check if a given matrix is a magic square -/
def isMagicSquare (m : MagicSquare) : Prop :=
  (∀ i : Fin 3, (m i 0 + m i 1 + m i 2 = magicSum m)) ∧ 
  (∀ j : Fin 3, (m 0 j + m 1 j + m 2 j = magicSum m)) ∧ 
  (m 0 0 + m 1 1 + m 2 2 = magicSum m) ∧
  (m 0 2 + m 1 1 + m 2 0 = magicSum m)

/-- Theorem: The magic sum of a 3x3 magic square is divisible by 3 -/
theorem magic_sum_divisible_by_three (m : MagicSquare) (h : isMagicSquare m) :
  ∃ k : ℤ, magicSum m = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_magic_sum_divisible_by_three_l3241_324157


namespace NUMINAMATH_CALUDE_sixteen_point_sphere_half_circumscribed_sphere_l3241_324187

/-- A tetrahedron with its associated spheres -/
structure Tetrahedron where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the sixteen-point sphere -/
  r : ℝ

/-- Theorem: There exists a tetrahedron for which the radius of its sixteen-point sphere 
    is equal to half the radius of its circumscribed sphere -/
theorem sixteen_point_sphere_half_circumscribed_sphere : 
  ∃ (t : Tetrahedron), t.r = t.R / 2 := by
  sorry


end NUMINAMATH_CALUDE_sixteen_point_sphere_half_circumscribed_sphere_l3241_324187


namespace NUMINAMATH_CALUDE_triangle_side_length_l3241_324164

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 10 → B = π/3 → C = π/4 → 
  c = 10 * (Real.sqrt 3 - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3241_324164


namespace NUMINAMATH_CALUDE_mountain_climb_time_l3241_324135

/-- Proves that the time to go up the mountain is 2 hours given the specified conditions -/
theorem mountain_climb_time 
  (total_time : ℝ) 
  (uphill_speed downhill_speed : ℝ) 
  (route_difference : ℝ) :
  total_time = 4 →
  uphill_speed = 3 →
  downhill_speed = 4 →
  route_difference = 2 →
  ∃ (uphill_time : ℝ),
    uphill_time = 2 ∧
    ∃ (downhill_time uphill_distance downhill_distance : ℝ),
      uphill_time + downhill_time = total_time ∧
      uphill_distance / uphill_speed = uphill_time ∧
      downhill_distance / downhill_speed = downhill_time ∧
      downhill_distance = uphill_distance + route_difference :=
by sorry

end NUMINAMATH_CALUDE_mountain_climb_time_l3241_324135


namespace NUMINAMATH_CALUDE_trig_expression_evaluation_l3241_324143

theorem trig_expression_evaluation : 
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) / 
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_evaluation_l3241_324143


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_500_l3241_324147

def multiple_of_15 (n : ℤ) : ℤ := 15 * n

theorem closest_multiple_of_15_to_500 :
  ∀ k : ℤ, k ≠ 33 → |500 - multiple_of_15 33| ≤ |500 - multiple_of_15 k| :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_500_l3241_324147


namespace NUMINAMATH_CALUDE_expression_equality_l3241_324107

theorem expression_equality : 
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 + 
  (2013^2 * 2014 - 2015) / Nat.factorial 2014 = 
  1 / Nat.factorial 2009 + 1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 1 / Nat.factorial 2014 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3241_324107


namespace NUMINAMATH_CALUDE_mary_nickels_l3241_324146

theorem mary_nickels (initial : ℕ) (given : ℕ) (total : ℕ) : 
  initial = 7 → given = 5 → total = initial + given → total = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_nickels_l3241_324146


namespace NUMINAMATH_CALUDE_thirteen_times_fifty_in_tens_l3241_324109

theorem thirteen_times_fifty_in_tens : 13 * 50 = 65 * 10 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_times_fifty_in_tens_l3241_324109


namespace NUMINAMATH_CALUDE_extra_apples_l3241_324191

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 25)
  (h2 : green_apples = 17)
  (h3 : students = 10) :
  red_apples + green_apples - students = 32 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l3241_324191


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l3241_324159

/-- Given two digits X and Y in base d > 8, if XY + XX = 182 in base d, then X - Y = d - 8 in base d -/
theorem digit_difference_in_base_d (d : ℕ) (X Y : Fin d) (h_d : d > 8) 
  (h_sum : d * X.val + Y.val + d * X.val + X.val = d^2 + 8*d + 2) : 
  X.val - Y.val = d - 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l3241_324159


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3241_324166

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3241_324166


namespace NUMINAMATH_CALUDE_line_passes_first_third_quadrants_l3241_324150

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_passes_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_first_third_quadrants_l3241_324150


namespace NUMINAMATH_CALUDE_remainder_problem_l3241_324137

theorem remainder_problem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3241_324137


namespace NUMINAMATH_CALUDE_power_zero_plus_power_division_l3241_324115

theorem power_zero_plus_power_division (x y : ℕ) : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_plus_power_division_l3241_324115


namespace NUMINAMATH_CALUDE_product_of_fractions_l3241_324151

theorem product_of_fractions : (1 : ℚ) / 5 * (3 : ℚ) / 7 = (3 : ℚ) / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3241_324151


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_with_second_digit_6_l3241_324102

/-- A function that returns true if all digits in a number are different --/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the digit at a specific position in a number --/
def digitAt (n : ℕ) (pos : ℕ) : ℕ := sorry

theorem smallest_four_digit_number_with_second_digit_6 :
  ∀ n : ℕ,
  (1000 ≤ n ∧ n < 10000) →  -- four-digit number
  (digitAt n 2 = 6) →       -- second digit is 6
  allDigitsDifferent n →    -- all digits are different
  1602 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_with_second_digit_6_l3241_324102


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l3241_324193

/-- Given three distinct single-digit numbers, returns the largest three-digit number that can be formed using these digits. -/
def largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the second largest three-digit number that can be formed using these digits. -/
def second_largest_three_digit (a b c : Nat) : Nat := sorry

/-- Given three distinct single-digit numbers, returns the smallest three-digit number that can be formed using these digits. -/
def smallest_three_digit (a b c : Nat) : Nat := sorry

theorem three_digit_sum_theorem :
  let a := 2
  let b := 5
  let c := 8
  largest_three_digit a b c + smallest_three_digit a b c + second_largest_three_digit a b c = 1935 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l3241_324193


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l3241_324183

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (2 * x - y) / 2 > x - y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l3241_324183


namespace NUMINAMATH_CALUDE_quadratic_sum_l3241_324128

/-- A quadratic function with specified properties -/
structure QuadraticFunction where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  point_x : ℝ
  point_y : ℝ
  vertex_condition : vertex_y = d * vertex_x^2 + e * vertex_x + f
  point_condition : point_y = d * point_x^2 + e * point_x + f
  is_vertex : ∀ x : ℝ, d * x^2 + e * x + f ≥ vertex_y

/-- Theorem: For a quadratic function with given properties, d + e + 2f = 19 -/
theorem quadratic_sum (g : QuadraticFunction) 
  (h1 : g.vertex_x = -2) 
  (h2 : g.vertex_y = 3) 
  (h3 : g.point_x = 0) 
  (h4 : g.point_y = 7) : 
  g.d + g.e + 2 * g.f = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3241_324128


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l3241_324131

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns true if n is a multiple of m -/
def isMultipleOf (n m : ℕ) : Prop := ∃ k, n = m * k

theorem least_multiple_with_digit_product_multiple : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n ≥ 315 := by sorry

theorem three_one_five_satisfies :
  isMultipleOf 315 15 ∧ isMultipleOf (digitProduct 315) 15 := by sorry

theorem least_multiple_with_digit_product_multiple_is_315 : 
  ∀ n : ℕ, n > 0 → isMultipleOf n 15 → isMultipleOf (digitProduct n) 15 → n = 315 ∨ n > 315 := by sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_three_one_five_satisfies_least_multiple_with_digit_product_multiple_is_315_l3241_324131


namespace NUMINAMATH_CALUDE_range_of_a_l3241_324180

-- Define the propositions p and q
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | p x}
def B (a : ℝ) : Set ℝ := {x : ℝ | q x a}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(p x) ∧ q x a)) → 
  ∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3241_324180


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3241_324141

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z / (1 - z) = Complex.I * 2 ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3241_324141


namespace NUMINAMATH_CALUDE_total_turnips_l3241_324181

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139) 
  (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := by
sorry

end NUMINAMATH_CALUDE_total_turnips_l3241_324181


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l3241_324114

-- Define a quadrilateral structure
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  c_positive : 0 < c
  d_positive : 0 < d
  area_positive : 0 < area

-- State the theorem
theorem quadrilateral_area_inequality (q : Quadrilateral) : 2 * q.area ≤ q.a * q.c + q.b * q.d := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l3241_324114


namespace NUMINAMATH_CALUDE_rachel_removed_bottle_caps_l3241_324127

/-- The number of bottle caps Rachel removed from a jar --/
def bottleCapsRemoved (originalCount remainingCount : ℕ) : ℕ :=
  originalCount - remainingCount

/-- Theorem: The number of bottle caps Rachel removed is equal to the difference
    between the original number and the remaining number of bottle caps --/
theorem rachel_removed_bottle_caps :
  bottleCapsRemoved 87 40 = 47 := by
  sorry

end NUMINAMATH_CALUDE_rachel_removed_bottle_caps_l3241_324127


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l3241_324184

-- Problem 1
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  S 2 = S 6 →  -- S₂ = S₆
  a 4 = 1 →    -- a₄ = 1
  a 5 = -1 := by sorry

-- Problem 2
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 4 - a 2 = 24 →  -- a₄ - a₂ = 24
  a 2 + a 3 = 6 →   -- a₂ + a₃ = 6
  a 1 = 1/5 ∧ q = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l3241_324184


namespace NUMINAMATH_CALUDE_dumbbell_weight_l3241_324113

theorem dumbbell_weight (total_dumbbells : ℕ) (total_weight : ℕ) 
  (h1 : total_dumbbells = 6)
  (h2 : total_weight = 120) :
  total_weight / total_dumbbells = 20 := by
sorry

end NUMINAMATH_CALUDE_dumbbell_weight_l3241_324113


namespace NUMINAMATH_CALUDE_rachel_earnings_calculation_l3241_324196

/-- Rachel's earnings in one hour -/
def rachel_earnings (base_wage : ℚ) (num_customers : ℕ) (tip_per_customer : ℚ) : ℚ :=
  base_wage + num_customers * tip_per_customer

/-- Theorem: Rachel's earnings in one hour -/
theorem rachel_earnings_calculation : 
  rachel_earnings 12 20 (5/4) = 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_earnings_calculation_l3241_324196


namespace NUMINAMATH_CALUDE_digit_zero_equality_l3241_324106

-- Define a function to count digits in a number
def countDigits (n : ℕ) : ℕ := sorry

-- Define a function to count zeros in a number
def countZeros (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of digits in a sequence
def sumDigits (n : ℕ) : ℕ := sorry

-- Define a function to sum the count of zeros in a sequence
def sumZeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem digit_zero_equality : sumDigits (10^8) = sumZeros (10^9) := by sorry

end NUMINAMATH_CALUDE_digit_zero_equality_l3241_324106


namespace NUMINAMATH_CALUDE_real_part_zero_necessary_not_sufficient_l3241_324165

/-- A complex number is purely imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "real part is zero" is necessary but not sufficient for a complex number to be purely imaginary. -/
theorem real_part_zero_necessary_not_sufficient :
  ∀ (a b : ℝ), 
    (∀ (z : ℂ), is_purely_imaginary z → z.re = 0) ∧
    ¬(∀ (z : ℂ), z.re = 0 → is_purely_imaginary z) :=
by sorry

end NUMINAMATH_CALUDE_real_part_zero_necessary_not_sufficient_l3241_324165


namespace NUMINAMATH_CALUDE_linear_function_identification_l3241_324123

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

theorem linear_function_identification :
  let f₁ : ℝ → ℝ := λ x ↦ x^3
  let f₂ : ℝ → ℝ := λ x ↦ -2*x + 1
  let f₃ : ℝ → ℝ := λ x ↦ 2/x
  let f₄ : ℝ → ℝ := λ x ↦ 2*x^2 + 1
  is_linear f₂ ∧ ¬is_linear f₁ ∧ ¬is_linear f₃ ∧ ¬is_linear f₄ :=
by sorry

end NUMINAMATH_CALUDE_linear_function_identification_l3241_324123


namespace NUMINAMATH_CALUDE_julie_age_l3241_324170

theorem julie_age (julie aaron : ℕ) 
  (h1 : julie = 4 * aaron) 
  (h2 : julie + 10 = 2 * (aaron + 10)) : 
  julie = 20 := by
sorry

end NUMINAMATH_CALUDE_julie_age_l3241_324170


namespace NUMINAMATH_CALUDE_sqrt_two_irrationality_proof_assumption_l3241_324108

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrationality_proof_assumption :
  (IsIrrational (Real.sqrt 2)) ↔ 
  (¬IsRational (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrationality_proof_assumption_l3241_324108


namespace NUMINAMATH_CALUDE_amount_of_c_l3241_324145

theorem amount_of_c (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_amount_of_c_l3241_324145


namespace NUMINAMATH_CALUDE_smallest_factor_smallest_factor_exists_l3241_324190

theorem smallest_factor (n : ℕ) : n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 → n ≥ 468 := by
  sorry

theorem smallest_factor_exists : ∃ n : ℕ, n > 0 ∧ 936 * n % 2^5 = 0 ∧ 936 * n % 3^3 = 0 ∧ 936 * n % 13^2 = 0 ∧ n = 468 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_smallest_factor_exists_l3241_324190


namespace NUMINAMATH_CALUDE_number_of_violas_proof_l3241_324110

/-- The number of violas in a music store, given the following conditions:
  * There are 800 cellos in the store
  * There are 70 cello-viola pairs made from the same tree
  * The probability of randomly choosing a cello-viola pair from the same tree is 0.00014583333333333335
-/
def number_of_violas : ℕ :=
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  600

theorem number_of_violas_proof :
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  number_of_violas = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_of_violas_proof_l3241_324110


namespace NUMINAMATH_CALUDE_fermat_last_digit_l3241_324112

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The last digit of Fermat numbers for n ≥ 2 is always 7 -/
theorem fermat_last_digit (n : ℕ) (h : n ≥ 2) : F n % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fermat_last_digit_l3241_324112


namespace NUMINAMATH_CALUDE_prob_two_eights_eight_dice_l3241_324121

/-- The probability of rolling exactly two 8's when rolling eight 8-sided dice -/
theorem prob_two_eights_eight_dice : ℝ := by
  -- Define the number of dice
  let n : ℕ := 8
  -- Define the number of sides on each die
  let sides : ℕ := 8
  -- Define the number of desired outcomes (dice showing 8)
  let k : ℕ := 2
  
  -- Calculate the probability
  have prob : ℝ := (n.choose k : ℝ) * (1 / sides) ^ k * ((sides - 1) / sides) ^ (n - k)
  
  -- Assert that this probability is equal to the given fraction
  have h : prob = (28 * 117649) / 16777216 := by sorry
  
  -- Return the probability
  exact prob

end NUMINAMATH_CALUDE_prob_two_eights_eight_dice_l3241_324121


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3241_324152

def total_distance : ℝ := 120
def total_time : ℝ := 7

theorem average_speed_calculation : 
  (total_distance / total_time) = 120 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3241_324152


namespace NUMINAMATH_CALUDE_center_value_is_27_l3241_324173

/-- Represents a 7x7 array where each row and column is an arithmetic sequence -/
def ArithmeticArray := Fin 7 → Fin 7 → ℤ

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : Fin 7 → ℤ) : ℤ :=
  (seq 6 - seq 0) / 6

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (seq : Fin 7 → ℤ) : Prop :=
  ∀ i j : Fin 7, seq j - seq i = (j - i : ℤ) * commonDifference seq

/-- Theorem: The center value of the arithmetic array is 27 -/
theorem center_value_is_27 (A : ArithmeticArray) 
  (h_rows : ∀ i : Fin 7, isArithmeticSequence (λ j ↦ A i j))
  (h_cols : ∀ j : Fin 7, isArithmeticSequence (λ i ↦ A i j))
  (h_first_row : A 0 0 = 3 ∧ A 0 6 = 39)
  (h_last_row : A 6 0 = 10 ∧ A 6 6 = 58) :
  A 3 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_27_l3241_324173


namespace NUMINAMATH_CALUDE_no_common_point_implies_skew_l3241_324188

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relations
variable (parallel : Line → Line → Prop)
variable (determine_plane : Line → Line → Plane → Prop)
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (has_common_point : Line → Line → Prop)

-- Axioms and definitions
axiom parallel_determine_plane (a b : Line) (p : Plane) :
  parallel a b → determine_plane a b p

axiom non_coplanar_non_collinear (p q r s : Point) :
  ¬coplanar p q r s → ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s

axiom skew_perpendicular (l₁ l₂ : Line) (p : Plane) :
  skew l₁ l₂ → ¬(perpendicular l₁ p ∧ perpendicular l₂ p)

-- The statement to be proved false
theorem no_common_point_implies_skew (l₁ l₂ : Line) :
  ¬has_common_point l₁ l₂ → skew l₁ l₂ := by sorry

end NUMINAMATH_CALUDE_no_common_point_implies_skew_l3241_324188


namespace NUMINAMATH_CALUDE_intersection_point_is_two_one_l3241_324189

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first line: x - 2y = 0 -/
def line1 (p : IntersectionPoint) : Prop :=
  p.x - 2 * p.y = 0

/-- Definition of the second line: x + y - 3 = 0 -/
def line2 (p : IntersectionPoint) : Prop :=
  p.x + p.y - 3 = 0

/-- Theorem stating that (2, 1) is the unique intersection point of the two lines -/
theorem intersection_point_is_two_one :
  ∃! p : IntersectionPoint, line1 p ∧ line2 p ∧ p.x = 2 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_two_one_l3241_324189


namespace NUMINAMATH_CALUDE_sum_of_consecutive_primes_has_three_prime_factors_l3241_324169

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ n, p < n → n < q → ¬(is_prime n)

theorem sum_of_consecutive_primes_has_three_prime_factors (p q : ℕ) :
  p > 2 → q > 2 → consecutive_primes p q →
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ p + q = a * b * c :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_primes_has_three_prime_factors_l3241_324169


namespace NUMINAMATH_CALUDE_kerosene_cost_friday_l3241_324140

/-- The cost of a liter of kerosene on Friday given the market conditions --/
theorem kerosene_cost_friday (rice_cost_monday : ℝ) 
  (h1 : rice_cost_monday = 0.36)
  (h2 : ∀ x, x > 0 → x * 12 * rice_cost_monday = x * 8 * (0.5 * rice_cost_monday))
  (h3 : ∀ x, x > 0 → 1.2 * x * rice_cost_monday = x * 1.2 * rice_cost_monday) :
  ∃ (kerosene_cost_friday : ℝ), kerosene_cost_friday = 0.576 :=
by sorry

end NUMINAMATH_CALUDE_kerosene_cost_friday_l3241_324140


namespace NUMINAMATH_CALUDE_bons_winning_probability_l3241_324149

/-- The probability of rolling a six -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six -/
def prob_not_six : ℚ := 5/6

/-- The probability that B. Bons wins the game -/
def prob_bons_wins : ℚ := 5/11

/-- Theorem stating that the probability of B. Bons winning is 5/11 -/
theorem bons_winning_probability : 
  prob_bons_wins = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_wins :=
by sorry

end NUMINAMATH_CALUDE_bons_winning_probability_l3241_324149


namespace NUMINAMATH_CALUDE_difference_of_squares_l3241_324171

theorem difference_of_squares (x : ℝ) : (2 + 3*x) * (2 - 3*x) = 4 - 9*x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3241_324171


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3241_324192

/-- Given an ellipse and a hyperbola with the same a and b parameters,
    prove that if the ellipse has eccentricity 1/2,
    then the hyperbola has eccentricity √7/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (∃ c : ℝ, c/a = 1/2)) :
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ c' : ℝ, c'/a = Real.sqrt 7 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3241_324192


namespace NUMINAMATH_CALUDE_iris_count_after_rose_addition_l3241_324142

/-- Given a garden with an initial ratio of irises to roses of 3:7,
    and an initial count of 35 roses, prove that after adding 30 roses,
    the number of irises that maintains the ratio is 27. -/
theorem iris_count_after_rose_addition 
  (initial_roses : ℕ) 
  (added_roses : ℕ) 
  (iris_ratio : ℕ) 
  (rose_ratio : ℕ) : 
  initial_roses = 35 →
  added_roses = 30 →
  iris_ratio = 3 →
  rose_ratio = 7 →
  (∃ (total_irises : ℕ), 
    total_irises * rose_ratio = (initial_roses + added_roses) * iris_ratio ∧
    total_irises = 27) :=
by sorry

end NUMINAMATH_CALUDE_iris_count_after_rose_addition_l3241_324142


namespace NUMINAMATH_CALUDE_composite_numbers_with_special_divisors_l3241_324185

theorem composite_numbers_with_special_divisors :
  ∀ n : ℕ, n > 1 →
    (∀ d : ℕ, d ∣ n → d ≠ 1 → d ≠ n → n - 20 ≤ d ∧ d ≤ n - 12) →
    n = 21 ∨ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_with_special_divisors_l3241_324185


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_1987_l3241_324101

theorem closest_multiple_of_15_to_1987 :
  ∃ (n : ℤ), n * 15 = 1980 ∧
  ∀ (m : ℤ), m * 15 ≠ 1980 → |1987 - (m * 15)| ≥ |1987 - 1980| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_1987_l3241_324101


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3241_324126

theorem cubic_roots_sum_of_squares_reciprocals (p q r : ℂ) : 
  p^3 - 15*p^2 + 26*p + 3 = 0 →
  q^3 - 15*q^2 + 26*q + 3 = 0 →
  r^3 - 15*r^2 + 26*r + 3 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 766/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocals_l3241_324126


namespace NUMINAMATH_CALUDE_curve_translation_l3241_324111

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  x^2 - y^2 - 2*x - 2*y - 1 = 0

/-- The transformed curve equation -/
def transformed_curve (x' y' : ℝ) : Prop :=
  x'^2 - y'^2 = 1

/-- The translation vector -/
def translation : ℝ × ℝ := (1, -1)

/-- Theorem stating that the given translation transforms the original curve to the transformed curve -/
theorem curve_translation :
  ∀ (x y : ℝ), original_curve x y ↔ transformed_curve (x - translation.1) (y - translation.2) :=
by sorry

end NUMINAMATH_CALUDE_curve_translation_l3241_324111


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l3241_324154

/-- For a right circular cylinder with fixed volume, the surface area is minimized when the diameter equals the height -/
theorem cylinder_min_surface_area (V : ℝ) (h V_pos : V > 0) :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  V = π * r^2 * h ∧
  (∀ (r' h' : ℝ), r' > 0 → h' > 0 → V = π * r'^2 * h' →
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') ∧
  h = 2 * r := by
  sorry

#check cylinder_min_surface_area

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l3241_324154


namespace NUMINAMATH_CALUDE_locus_of_points_l3241_324139

/-- Given two parallel lines e₁ and e₂ in the plane, separated by a distance 2g,
    and a perpendicular line f intersecting them at O₁ and O₂ respectively,
    this theorem characterizes the locus of points P(x, y) such that a line through P
    intersects e₁ at P₁ and e₂ at P₂ with O₁P₁ · O₂P₂ = k. -/
theorem locus_of_points (g : ℝ) (k : ℝ) (x y : ℝ) :
  k = 1 → (y^2 / g^2 ≥ 1 - x^2) ∧
  k = -1 → (y^2 / g^2 ≤ 1 + x^2) := by
  sorry


end NUMINAMATH_CALUDE_locus_of_points_l3241_324139


namespace NUMINAMATH_CALUDE_zero_exponent_eq_one_l3241_324105

theorem zero_exponent_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_eq_one_l3241_324105


namespace NUMINAMATH_CALUDE_unique_common_solution_coefficient_l3241_324116

theorem unique_common_solution_coefficient : 
  ∃! a : ℝ, ∃ x : ℝ, (x^2 + a*x + 1 = 0) ∧ (x^2 - x - a = 0) ∧ (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_common_solution_coefficient_l3241_324116


namespace NUMINAMATH_CALUDE_max_value_complex_l3241_324175

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l3241_324175


namespace NUMINAMATH_CALUDE_board_operations_finite_and_invariant_l3241_324177

/-- Represents the state of the board with n natural numbers -/
def Board := List Nat

/-- Performs one operation on the board, replacing two numbers with their GCD and LCM -/
def performOperation (board : Board) (i j : Nat) : Board :=
  sorry

/-- Checks if the board is in its final state (all pairs are proper) -/
def isFinalState (board : Board) : Bool :=
  sorry

theorem board_operations_finite_and_invariant (initial_board : Board) :
  ∃ (final_board : Board),
    (∀ (sequence : List (Nat × Nat)), 
      isFinalState (sequence.foldl (λ b (i, j) => performOperation b i j) initial_board)) ∧
    (∀ (sequence1 sequence2 : List (Nat × Nat)),
      isFinalState (sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board) ∧
      isFinalState (sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) →
      sequence1.foldl (λ b (i, j) => performOperation b i j) initial_board =
      sequence2.foldl (λ b (i, j) => performOperation b i j) initial_board) :=
by
  sorry

end NUMINAMATH_CALUDE_board_operations_finite_and_invariant_l3241_324177


namespace NUMINAMATH_CALUDE_balls_in_urns_l3241_324195

/-- The number of ways to place k identical balls into n urns with at most one ball per urn -/
def place_balls_limited (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to place k identical balls into n urns with unlimited balls per urn -/
def place_balls_unlimited (n k : ℕ) : ℕ := Nat.choose (k+n-1) (n-1)

theorem balls_in_urns (n k : ℕ) :
  (place_balls_limited n k = Nat.choose n k) ∧
  (place_balls_unlimited n k = Nat.choose (k+n-1) (n-1)) := by
  sorry

end NUMINAMATH_CALUDE_balls_in_urns_l3241_324195


namespace NUMINAMATH_CALUDE_committee_formation_l3241_324186

theorem committee_formation (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l3241_324186


namespace NUMINAMATH_CALUDE_carls_cupcake_goal_l3241_324133

/-- Carl's cupcake selling problem -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) : 
  goal = 96 → days = 2 → payment = 24 → cupcakes_per_day * days = goal + payment → cupcakes_per_day = 60 := by
  sorry

end NUMINAMATH_CALUDE_carls_cupcake_goal_l3241_324133


namespace NUMINAMATH_CALUDE_parallel_condition_l3241_324130

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The condition ab = 1 -/
def condition (l1 l2 : Line) : Prop :=
  l1.a * l2.b = 1

theorem parallel_condition (l1 l2 : Line) :
  (parallel l1 l2 → condition l1 l2) ∧
  ¬(condition l1 l2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3241_324130


namespace NUMINAMATH_CALUDE_equation_solution_l3241_324179

theorem equation_solution : ∃ x : ℝ, 12 * (x - 3) - 1 = 2 * x + 3 ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3241_324179


namespace NUMINAMATH_CALUDE_decimal_digits_divisibility_l3241_324104

def repeatedDigits (a b c : ℕ) : ℕ :=
  a * (10^4006 - 10^2004) / 99 + b * 10^2002 + c * (10^2002 - 1) / 99

theorem decimal_digits_divisibility (a b c : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) 
  (h_div : 37 ∣ repeatedDigits a b c) : 
  b = a + c := by sorry

end NUMINAMATH_CALUDE_decimal_digits_divisibility_l3241_324104


namespace NUMINAMATH_CALUDE_sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l3241_324119

theorem sqrt_two_sin_twenty_equals_cos_minus_sin_theta (θ : Real) :
  θ > 0 ∧ θ < Real.pi / 2 →
  Real.sqrt 2 * Real.sin (20 * Real.pi / 180) = Real.cos θ - Real.sin θ →
  θ = 25 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_sin_twenty_equals_cos_minus_sin_theta_l3241_324119


namespace NUMINAMATH_CALUDE_linear_coefficient_is_zero_l3241_324129

/-- The coefficient of the linear term in the standard form of (2 - x)(3x + 4) = 2x - 1 is 0 -/
theorem linear_coefficient_is_zero : 
  let f : ℝ → ℝ := λ x => (2 - x) * (3 * x + 4) - (2 * x - 1)
  ∃ a c : ℝ, ∀ x, f x = -3 * x^2 + 0 * x + c :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_is_zero_l3241_324129


namespace NUMINAMATH_CALUDE_triangle_properties_l3241_324161

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = 3 * t.b * t.c * Real.cos t.A)
  (h2 : t.B = t.C)
  (h3 : t.a = 2) :
  (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 5) ∧ 
  (Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3241_324161
