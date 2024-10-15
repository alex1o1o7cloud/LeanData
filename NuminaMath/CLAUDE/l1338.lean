import Mathlib

namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_in_hexagon_l1338_133894

/-- A regular hexagon with randomly colored edges -/
structure ColoredHexagon where
  /-- The number of sides in a regular hexagon -/
  numSides : Nat
  /-- The number of diagonals in a regular hexagon -/
  numDiagonals : Nat
  /-- The total number of edges (sides + diagonals) -/
  numEdges : Nat
  /-- The number of possible triangles in a hexagon -/
  numTriangles : Nat
  /-- The probability of an edge being a specific color -/
  probEdgeColor : ℚ
  /-- The probability of a triangle not being monochromatic -/
  probNonMonochromatic : ℚ

/-- The probability of having at least one monochromatic triangle in a colored hexagon -/
def probMonochromaticTriangle (h : ColoredHexagon) : ℚ :=
  1 - (h.probNonMonochromatic ^ h.numTriangles)

/-- Theorem stating the probability of a monochromatic triangle in a randomly colored hexagon -/
theorem prob_monochromatic_triangle_in_hexagon :
  ∃ (h : ColoredHexagon),
    h.numSides = 6 ∧
    h.numDiagonals = 9 ∧
    h.numEdges = 15 ∧
    h.numTriangles = 20 ∧
    h.probEdgeColor = 1/2 ∧
    h.probNonMonochromatic = 3/4 ∧
    probMonochromaticTriangle h = 253/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_in_hexagon_l1338_133894


namespace NUMINAMATH_CALUDE_square_area_rational_l1338_133813

theorem square_area_rational (s : ℚ) : ∃ (a : ℚ), a = s^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_rational_l1338_133813


namespace NUMINAMATH_CALUDE_probability_colored_ball_l1338_133843

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of red balls
def red_balls : ℕ := 2

-- Define the number of blue balls
def blue_balls : ℕ := 5

-- Define the number of white balls
def white_balls : ℕ := 3

-- Theorem: The probability of drawing a colored ball is 7/10
theorem probability_colored_ball :
  (red_balls + blue_balls : ℚ) / total_balls = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_colored_ball_l1338_133843


namespace NUMINAMATH_CALUDE_circles_common_chord_l1338_133888

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y - 40 = 0

-- Define the line
def common_chord (x y : ℝ) : Prop := x + 3*y - 10 = 0

-- Theorem statement
theorem circles_common_chord :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    circle1 p1.1 p1.2 ∧ circle1 p2.1 p2.2 ∧
    circle2 p1.1 p1.2 ∧ circle2 p2.1 p2.2 →
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y :=
sorry

end NUMINAMATH_CALUDE_circles_common_chord_l1338_133888


namespace NUMINAMATH_CALUDE_parabola_symmetry_point_l1338_133837

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on a parabola -/
def onParabola (p : Parabola) (pt : Point) : Prop :=
  pt.y = p.a * (pt.x - p.h)^2 + p.k

theorem parabola_symmetry_point (p : Parabola) :
  ∃ (m : ℝ),
    onParabola p ⟨-1, 2⟩ ∧
    onParabola p ⟨1, -2⟩ ∧
    onParabola p ⟨3, 2⟩ ∧
    onParabola p ⟨-2, m⟩ ∧
    onParabola p ⟨4, m⟩ := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_point_l1338_133837


namespace NUMINAMATH_CALUDE_point_transformation_l1338_133836

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutYEqualX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 2 3
  let (x₂, y₂) := reflectAboutYEqualX x₁ y₁
  (x₂ = 5 ∧ y₂ = -1) → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1338_133836


namespace NUMINAMATH_CALUDE_largest_number_after_removal_l1338_133834

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def concatenated_primes : Nat :=
  first_ten_primes.foldl (fun acc n => acc * (10 ^ (Nat.digits 10 n).length) + n) 0

def remove_six_digits (n : Nat) : Set Nat :=
  { m | ∃ (digits : List Nat), 
    digits.length = (Nat.digits 10 n).length - 6 ∧
    (Nat.digits 10 m) = digits ∧
    (∀ d ∈ digits, d ∈ Nat.digits 10 n) }

theorem largest_number_after_removal :
  7317192329 ∈ remove_six_digits concatenated_primes ∧
  ∀ m ∈ remove_six_digits concatenated_primes, m ≤ 7317192329 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_after_removal_l1338_133834


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1338_133892

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ 2) → ((2 * x + 5) / (x - 2) < 1 ↔ -7 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1338_133892


namespace NUMINAMATH_CALUDE_a_c_inequality_l1338_133864

theorem a_c_inequality (a c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : c > 1) :
  a * c + 1 < a + c := by
  sorry

end NUMINAMATH_CALUDE_a_c_inequality_l1338_133864


namespace NUMINAMATH_CALUDE_absolute_value_solution_set_l1338_133898

theorem absolute_value_solution_set (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_solution_set_l1338_133898


namespace NUMINAMATH_CALUDE_dream_number_k_value_l1338_133802

def is_dream_number (p : ℕ) : Prop :=
  p ≥ 100 ∧ p < 1000 ∧
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  h ≠ 0 ∧ t ≠ 0 ∧ u ≠ 0 ∧
  (h - t : ℤ) = (t - u : ℤ)

def m (p : ℕ) : ℕ :=
  let h := p / 100
  let t := (p / 10) % 10
  let u := p % 10
  (10 * h + t) + (10 * t + u)

def n (p : ℕ) : ℕ :=
  let h := p / 100
  let u := p % 10
  (10 * h + u) + (10 * u + h)

def F (p : ℕ) : ℚ :=
  (m p - n p : ℚ) / 9

def s (x y : ℕ) : ℕ := 10 * x + y + 502

def t (a b : ℕ) : ℕ := 10 * a + b + 200

theorem dream_number_k_value
  (x y a b : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 7)
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 1 ≤ b ∧ b ≤ 9)
  (hs : is_dream_number (s x y))
  (ht : is_dream_number (t a b))
  (h_eq : 2 * F (s x y) + F (t a b) = -1)
  : F (s x y) / F (s x y) = -3 := by
  sorry

end NUMINAMATH_CALUDE_dream_number_k_value_l1338_133802


namespace NUMINAMATH_CALUDE_expression_evaluation_l1338_133867

theorem expression_evaluation (x y : ℕ) (h1 : x = 2) (h2 : y = 3) : 3 * x^y + 4 * y^x = 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1338_133867


namespace NUMINAMATH_CALUDE_boat_journey_distance_l1338_133811

def boat_journey (total_time : ℝ) (stream_velocity : ℝ) (boat_speed : ℝ) : Prop :=
  let downstream_speed : ℝ := boat_speed + stream_velocity
  let upstream_speed : ℝ := boat_speed - stream_velocity
  let distance : ℝ := 180
  (distance / downstream_speed + (distance / 2) / upstream_speed = total_time) ∧
  (downstream_speed > 0) ∧
  (upstream_speed > 0)

theorem boat_journey_distance :
  boat_journey 19 4 14 := by sorry

end NUMINAMATH_CALUDE_boat_journey_distance_l1338_133811


namespace NUMINAMATH_CALUDE_almond_weight_in_mixture_l1338_133886

/-- Given a mixture of nuts where the ratio of almonds to walnuts is 5:1 by weight,
    and the total weight is 140 pounds, the weight of almonds is 116.67 pounds. -/
theorem almond_weight_in_mixture (almond_parts : ℕ) (walnut_parts : ℕ) (total_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 1 →
  total_weight = 140 →
  (almond_parts * total_weight) / (almond_parts + walnut_parts) = 116.67 := by
  sorry

end NUMINAMATH_CALUDE_almond_weight_in_mixture_l1338_133886


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_six_l1338_133818

theorem smallest_digit_divisible_by_six : 
  ∃ (N : ℕ), N < 10 ∧ (1453 * 10 + N) % 6 = 0 ∧ 
  ∀ (M : ℕ), M < N → M < 10 → (1453 * 10 + M) % 6 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_six_l1338_133818


namespace NUMINAMATH_CALUDE_students_playing_at_least_one_sport_l1338_133897

/-- The number of students who like to play basketball -/
def B : ℕ := 7

/-- The number of students who like to play cricket -/
def C : ℕ := 10

/-- The number of students who like to play soccer -/
def S : ℕ := 8

/-- The number of students who like to play all three sports -/
def BCS : ℕ := 2

/-- The number of students who like to play both basketball and cricket -/
def BC : ℕ := 5

/-- The number of students who like to play both basketball and soccer -/
def BS : ℕ := 4

/-- The number of students who like to play both cricket and soccer -/
def CS : ℕ := 3

/-- The theorem stating that the number of students who like to play at least one sport is 21 -/
theorem students_playing_at_least_one_sport : 
  B + C + S - ((BC - BCS) + (BS - BCS) + (CS - BCS)) + BCS = 21 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_at_least_one_sport_l1338_133897


namespace NUMINAMATH_CALUDE_consecutive_cubes_to_consecutive_squares_l1338_133800

theorem consecutive_cubes_to_consecutive_squares (A : ℕ) :
  (∃ k : ℕ, A^2 = (k + 1)^3 - k^3) →
  (∃ m : ℕ, A = m^2 + (m + 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_cubes_to_consecutive_squares_l1338_133800


namespace NUMINAMATH_CALUDE_brick_surface_area_l1338_133847

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a rectangular prism with dimensions 8 cm x 4 cm x 2 cm is 112 square centimeters. -/
theorem brick_surface_area :
  surface_area 8 4 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l1338_133847


namespace NUMINAMATH_CALUDE_platform_length_l1338_133839

/-- Given a train's speed and crossing times, calculate the platform length -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- 72 kmph
  (h2 : platform_crossing_time = 32)  -- 32 seconds
  (h3 : man_crossing_time = 18)  -- 18 seconds
  : ∃ (platform_length : ℝ), platform_length = 280 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l1338_133839


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1338_133874

theorem quadratic_roots_problem (p q : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (p + 5 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (p + 5 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  (q + 3 * Complex.I) ^ 2 - (12 + 8 * Complex.I) * (q + 3 * Complex.I) + (20 + 40 * Complex.I) = 0 →
  p = 10 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1338_133874


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1338_133883

/-- Given two positive integers with LCM 750 and product 18750, their HCF is 25 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h1 : Nat.lcm A B = 750) 
  (h2 : A * B = 18750) : 
  Nat.gcd A B = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1338_133883


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1338_133848

theorem diophantine_equation_solution (a : ℕ+) :
  ∃ (x y : ℕ+), (x^3 + x + a^2 : ℤ) = y^2 ∧
  x = 4 * a^2 * (16 * a^4 + 2) ∧
  y = 2 * a * (16 * a^4 + 2) * (16 * a^4 + 1) - a :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1338_133848


namespace NUMINAMATH_CALUDE_common_root_is_negative_one_l1338_133846

/-- Given two equations with a common root and a condition on the coefficients,
    prove that the common root is -1. -/
theorem common_root_is_negative_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^3 + b*x + a = 0 → x = -1 := by
  sorry

#check common_root_is_negative_one

end NUMINAMATH_CALUDE_common_root_is_negative_one_l1338_133846


namespace NUMINAMATH_CALUDE_product_inequality_l1338_133882

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1338_133882


namespace NUMINAMATH_CALUDE_radian_measure_60_degrees_l1338_133805

/-- The radian measure of a 60° angle is π/3. -/
theorem radian_measure_60_degrees :
  (60 * Real.pi / 180 : ℝ) = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_radian_measure_60_degrees_l1338_133805


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1338_133823

theorem fraction_evaluation : (3 : ℚ) / (1 - 3 / 4) = 12 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1338_133823


namespace NUMINAMATH_CALUDE_exam_venue_problem_l1338_133872

/-- Given a group of students, calculates the number not good at either of two subjects. -/
def students_not_good_at_either (total : ℕ) (good_at_english : ℕ) (good_at_chinese : ℕ) (good_at_both : ℕ) : ℕ :=
  total - (good_at_english + good_at_chinese - good_at_both)

/-- Proves that in a group of 45 students, if 35 are good at English, 31 are good at Chinese,
    and 24 are good at both, then 3 students are not good at either subject. -/
theorem exam_venue_problem :
  students_not_good_at_either 45 35 31 24 = 3 := by
  sorry

end NUMINAMATH_CALUDE_exam_venue_problem_l1338_133872


namespace NUMINAMATH_CALUDE_sequence_inequality_l1338_133862

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n + a (2 * n) ≥ 3 * n)
  (h2 : ∀ n : ℕ, a (n + 1) + n ≤ 2 * Real.sqrt (a n * (n + 1)))
  (h3 : ∀ n : ℕ, 0 ≤ a n) :
  ∀ n : ℕ, a n ≥ n :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1338_133862


namespace NUMINAMATH_CALUDE_product_of_roots_squared_minus_three_l1338_133824

theorem product_of_roots_squared_minus_three (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = -35) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_squared_minus_three_l1338_133824


namespace NUMINAMATH_CALUDE_sum_product_range_l1338_133806

theorem sum_product_range (x y z : ℝ) (h : x + y + z = 3) :
  ∃ S : Set ℝ, S = Set.Iic (9/4) ∧
  ∀ t : ℝ, (∃ a b c : ℝ, a + b + c = 3 ∧ t = a*b + a*c + b*c) ↔ t ∈ S :=
sorry

end NUMINAMATH_CALUDE_sum_product_range_l1338_133806


namespace NUMINAMATH_CALUDE_flower_basket_problem_l1338_133873

theorem flower_basket_problem (o y p : ℕ) 
  (h1 : y + p = 7)   -- All but 7 are orange
  (h2 : o + p = 10)  -- All but 10 are yellow
  (h3 : o + y = 5)   -- All but 5 are purple
  : o + y + p = 11 := by
  sorry

end NUMINAMATH_CALUDE_flower_basket_problem_l1338_133873


namespace NUMINAMATH_CALUDE_binomial_9_choose_5_l1338_133833

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_9_choose_5_l1338_133833


namespace NUMINAMATH_CALUDE_rectangle_area_l1338_133860

/-- Given a rectangle with length L and width W, if increasing the length by 10
    and decreasing the width by 6 doesn't change the area, and the perimeter is 76,
    then the area of the original rectangle is 360 square meters. -/
theorem rectangle_area (L W : ℝ) : 
  (L + 10) * (W - 6) = L * W → 2 * L + 2 * W = 76 → L * W = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1338_133860


namespace NUMINAMATH_CALUDE_line_always_intersects_hyperbola_iff_k_in_range_l1338_133801

/-- A line intersects a hyperbola if their equations have a common solution -/
def intersects (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The main theorem: if a line always intersects the hyperbola, then k is in the open interval (-√2/2, √2/2) -/
theorem line_always_intersects_hyperbola_iff_k_in_range (k : ℝ) :
  (∀ b : ℝ, intersects k b) ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_always_intersects_hyperbola_iff_k_in_range_l1338_133801


namespace NUMINAMATH_CALUDE_inequality_proof_l1338_133842

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1)^2 / b + (b + 1)^2 / a ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1338_133842


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1338_133835

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1338_133835


namespace NUMINAMATH_CALUDE_factor_calculation_l1338_133868

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 18 → 
  factor * (2 * initial_number + 5) = 123 → 
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1338_133868


namespace NUMINAMATH_CALUDE_conference_room_arrangements_count_l1338_133832

/-- The number of distinct arrangements of seats in a conference room. -/
def conference_room_arrangements : ℕ :=
  let total_seats : ℕ := 12
  let armchairs : ℕ := 6
  let benches : ℕ := 4
  let stools : ℕ := 2
  Nat.choose total_seats stools * Nat.choose (total_seats - stools) benches

theorem conference_room_arrangements_count :
  conference_room_arrangements = 13860 := by
  sorry

#eval conference_room_arrangements

end NUMINAMATH_CALUDE_conference_room_arrangements_count_l1338_133832


namespace NUMINAMATH_CALUDE_factorial_1500_trailing_zeros_l1338_133815

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (Finset.range 13).sum fun i => n / (5 ^ (i + 1))

/-- 1500! has 374 trailing zeros -/
theorem factorial_1500_trailing_zeros :
  trailingZeros 1500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1500_trailing_zeros_l1338_133815


namespace NUMINAMATH_CALUDE_seminar_selection_l1338_133845

theorem seminar_selection (boys girls : ℕ) (total_select : ℕ) : 
  boys = 4 → girls = 3 → total_select = 4 →
  (Nat.choose (boys + girls) total_select) - (Nat.choose boys total_select) = 34 := by
sorry

end NUMINAMATH_CALUDE_seminar_selection_l1338_133845


namespace NUMINAMATH_CALUDE_area_difference_is_one_l1338_133853

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define an equilateral triangle with side length 1
def unit_equilateral_triangle : Set (ℝ × ℝ) := sorry

-- Define the region R (union of square and 12 triangles)
def R : Set (ℝ × ℝ) := sorry

-- Define the smallest convex polygon S containing R
def S : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_difference_is_one :
  area (S \ R) = 1 := by sorry

end NUMINAMATH_CALUDE_area_difference_is_one_l1338_133853


namespace NUMINAMATH_CALUDE_first_round_games_count_l1338_133825

/-- A tennis tournament with specific conditions -/
structure TennisTournament where
  total_rounds : Nat
  second_round_games : Nat
  third_round_games : Nat
  final_games : Nat
  cans_per_game : Nat
  balls_per_can : Nat
  total_balls_used : Nat

/-- The number of games in the first round of the tournament -/
def first_round_games (t : TennisTournament) : Nat :=
  ((t.total_balls_used - (t.second_round_games + t.third_round_games + t.final_games) * 
    t.cans_per_game * t.balls_per_can) / (t.cans_per_game * t.balls_per_can))

/-- Theorem stating the number of games in the first round -/
theorem first_round_games_count (t : TennisTournament) 
  (h1 : t.total_rounds = 4)
  (h2 : t.second_round_games = 4)
  (h3 : t.third_round_games = 2)
  (h4 : t.final_games = 1)
  (h5 : t.cans_per_game = 5)
  (h6 : t.balls_per_can = 3)
  (h7 : t.total_balls_used = 225) :
  first_round_games t = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_round_games_count_l1338_133825


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1338_133876

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  5 * x^2 = 15 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (x + 3)^3 = -64 ↔ x = -7 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l1338_133876


namespace NUMINAMATH_CALUDE_tech_class_avg_age_l1338_133820

def avg_age_arts : ℝ := 21
def num_arts_classes : ℕ := 8
def num_tech_classes : ℕ := 5
def overall_avg_age : ℝ := 19.846153846153847

theorem tech_class_avg_age :
  let total_classes := num_arts_classes + num_tech_classes
  let total_age := overall_avg_age * total_classes
  let arts_total_age := avg_age_arts * num_arts_classes
  (total_age - arts_total_age) / num_tech_classes = 990.4000000000002 := by
sorry

end NUMINAMATH_CALUDE_tech_class_avg_age_l1338_133820


namespace NUMINAMATH_CALUDE_james_tylenol_frequency_l1338_133887

/-- Proves that James takes Tylenol tablets every 6 hours given the conditions --/
theorem james_tylenol_frequency 
  (tablets_per_dose : ℕ)
  (mg_per_tablet : ℕ)
  (total_mg_per_day : ℕ)
  (hours_per_day : ℕ)
  (h1 : tablets_per_dose = 2)
  (h2 : mg_per_tablet = 375)
  (h3 : total_mg_per_day = 3000)
  (h4 : hours_per_day = 24) :
  (hours_per_day : ℚ) / ((total_mg_per_day : ℚ) / ((tablets_per_dose : ℚ) * mg_per_tablet)) = 6 := by
  sorry

#check james_tylenol_frequency

end NUMINAMATH_CALUDE_james_tylenol_frequency_l1338_133887


namespace NUMINAMATH_CALUDE_three_legged_dogs_carly_three_legged_dogs_l1338_133877

theorem three_legged_dogs (total_nails : ℕ) (total_dogs : ℕ) : ℕ :=
  let nails_per_paw := 4
  let paws_per_dog := 4
  let nails_per_dog := nails_per_paw * paws_per_dog
  let expected_total_nails := total_dogs * nails_per_dog
  let missing_nails := expected_total_nails - total_nails
  missing_nails / nails_per_paw

theorem carly_three_legged_dogs :
  three_legged_dogs 164 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_legged_dogs_carly_three_legged_dogs_l1338_133877


namespace NUMINAMATH_CALUDE_fifth_term_is_eight_l1338_133812

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∀ n, a (n + 1) = 2 * a n

/-- Theorem: In a geometric sequence with common ratio 2 and a₂a₆ = 16, a₅ = 8 -/
theorem fifth_term_is_eight (seq : GeometricSequence) 
    (h : seq.a 2 * seq.a 6 = 16) : seq.a 5 = 8 := by
  sorry

#check fifth_term_is_eight

end NUMINAMATH_CALUDE_fifth_term_is_eight_l1338_133812


namespace NUMINAMATH_CALUDE_f_properties_l1338_133879

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 3*a

/-- Theorem stating the properties of the function f and its zeros -/
theorem f_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) →
  (3/2 < a ∧ a < 3) ∧
  (2*(Real.sqrt 2 + 1)/3 < 1/x₁ + 1/x₂ + 1/x₃) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1338_133879


namespace NUMINAMATH_CALUDE_line_parallelism_l1338_133881

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallelism 
  (a b : Line) 
  (α β : Plane) 
  (l : Line) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : intersection α β = l) 
  (h4 : parallel_lines a l) 
  (h5 : subset_line_plane b β) 
  (h6 : parallel_line_plane b α) : 
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallelism_l1338_133881


namespace NUMINAMATH_CALUDE_number_line_essential_elements_l1338_133896

/-- Represents the essential elements of a number line -/
inductive NumberLineElement
  | PositiveDirection
  | Origin
  | UnitLength

/-- The set of essential elements of a number line -/
def essentialElements : Set NumberLineElement :=
  {NumberLineElement.PositiveDirection, NumberLineElement.Origin, NumberLineElement.UnitLength}

/-- Theorem stating that the essential elements of a number line are precisely
    positive direction, origin, and unit length -/
theorem number_line_essential_elements :
  ∀ (e : NumberLineElement), e ∈ essentialElements ↔
    (e = NumberLineElement.PositiveDirection ∨
     e = NumberLineElement.Origin ∨
     e = NumberLineElement.UnitLength) :=
by sorry

end NUMINAMATH_CALUDE_number_line_essential_elements_l1338_133896


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l1338_133816

theorem quadratic_roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 7 = 0 →
  x₂^2 - 4*x₂ - 7 = 0 →
  x₁ ≠ 0 →
  x₂ ≠ 0 →
  1/x₁ + 1/x₂ = -4/7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l1338_133816


namespace NUMINAMATH_CALUDE_triple_hash_twenty_l1338_133840

/-- The # operation defined on real numbers -/
def hash (N : ℝ) : ℝ := 0.75 * N + 3

/-- Theorem stating that applying the hash operation three times to 20 results in 15.375 -/
theorem triple_hash_twenty : hash (hash (hash 20)) = 15.375 := by sorry

end NUMINAMATH_CALUDE_triple_hash_twenty_l1338_133840


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1338_133821

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1338_133821


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1338_133809

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -(Real.sqrt 3 / 2) := by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1338_133809


namespace NUMINAMATH_CALUDE_total_price_houses_l1338_133861

/-- The total price of two houses, given the price of the first house and that the second house is twice as expensive. -/
def total_price (price_first_house : ℕ) : ℕ :=
  price_first_house + 2 * price_first_house

/-- Theorem stating that the total price of the two houses is $600,000 when the first house costs $200,000. -/
theorem total_price_houses : total_price 200000 = 600000 := by
  sorry

end NUMINAMATH_CALUDE_total_price_houses_l1338_133861


namespace NUMINAMATH_CALUDE_max_player_salary_l1338_133829

theorem max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  n = 25 →
  min_salary = 15000 →
  total_cap = 800000 →
  (n - 1) * min_salary + (total_cap - (n - 1) * min_salary) = 440000 :=
by sorry

end NUMINAMATH_CALUDE_max_player_salary_l1338_133829


namespace NUMINAMATH_CALUDE_circle_path_in_triangle_l1338_133850

/-- The path length of the center of a circle rolling inside a triangle --/
def circle_path_length (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

/-- Theorem stating the path length of a circle's center rolling inside a specific triangle --/
theorem circle_path_in_triangle : 
  let a : ℝ := 8
  let b : ℝ := 10
  let c : ℝ := 12.5
  let r : ℝ := 1.5
  circle_path_length a b c r = 21.5 := by
  sorry

#check circle_path_in_triangle

end NUMINAMATH_CALUDE_circle_path_in_triangle_l1338_133850


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1338_133803

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 5225) → original_salary = 5000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1338_133803


namespace NUMINAMATH_CALUDE_min_marked_cells_l1338_133830

/-- Represents a marked cell on the board -/
structure MarkedCell :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a board with marked cells -/
structure Board :=
  (size : ℕ)
  (marked_cells : List MarkedCell)

/-- Checks if a sub-board contains a marked cell on both diagonals -/
def subBoardContainsMarkedDiagonals (board : Board) (m : ℕ) (topLeft : MarkedCell) : Prop :=
  ∃ (c1 c2 : MarkedCell),
    c1 ∈ board.marked_cells ∧
    c2 ∈ board.marked_cells ∧
    c1.row - topLeft.row = c1.col - topLeft.col ∧
    c2.row - topLeft.row = topLeft.col + m - 1 - c2.col ∧
    c1.row ≥ topLeft.row ∧ c1.row < topLeft.row + m ∧
    c1.col ≥ topLeft.col ∧ c1.col < topLeft.col + m ∧
    c2.row ≥ topLeft.row ∧ c2.row < topLeft.row + m ∧
    c2.col ≥ topLeft.col ∧ c2.col < topLeft.col + m

/-- The main theorem stating the minimum number of marked cells -/
theorem min_marked_cells (n : ℕ) :
  ∃ (board : Board),
    board.size = n ∧
    board.marked_cells.length = n ∧
    (∀ (m : ℕ) (topLeft : MarkedCell),
      m > n / 2 →
      topLeft.row + m ≤ n →
      topLeft.col + m ≤ n →
      subBoardContainsMarkedDiagonals board m topLeft) ∧
    (∀ (board' : Board),
      board'.size = n →
      board'.marked_cells.length < n →
      ∃ (m : ℕ) (topLeft : MarkedCell),
        m > n / 2 ∧
        topLeft.row + m ≤ n ∧
        topLeft.col + m ≤ n ∧
        ¬subBoardContainsMarkedDiagonals board' m topLeft) := by
  sorry

end NUMINAMATH_CALUDE_min_marked_cells_l1338_133830


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1338_133826

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - 3*x + a = 0}
def B (b : ℝ) : Set ℝ := {x | x^2 + b = 0}

-- State the theorem
theorem union_of_A_and_B (a b : ℝ) :
  (∃ (x : ℝ), A a ∩ B b = {x}) →
  (∃ (y z : ℝ), A a ∪ B b = {y, z, 2}) :=
sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1338_133826


namespace NUMINAMATH_CALUDE_ron_has_two_friends_l1338_133844

/-- The number of Ron's friends eating pizza -/
def num_friends (total_slices : ℕ) (slices_per_person : ℕ) : ℕ :=
  total_slices / slices_per_person - 1

/-- Theorem: Given a 12-slice pizza and 4 slices per person, Ron has 2 friends -/
theorem ron_has_two_friends : num_friends 12 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ron_has_two_friends_l1338_133844


namespace NUMINAMATH_CALUDE_sum_divisors_cube_lt_n_fourth_l1338_133828

def S (n : ℕ) : ℕ := sorry

theorem sum_divisors_cube_lt_n_fourth {n : ℕ} (h_odd : Odd n) (h_gt_one : n > 1) :
  (S n)^3 < n^4 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_cube_lt_n_fourth_l1338_133828


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l1338_133838

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 0.454545

-- Theorem statement
theorem repeating_decimal_value : repeating_decimal = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_value_l1338_133838


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1338_133807

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = 3 ∧ m = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1338_133807


namespace NUMINAMATH_CALUDE_inequality_solution_l1338_133858

theorem inequality_solution (x : ℝ) : 
  (202 * Real.sqrt (x^3 - 2*x - 2/x + 1/x^3 + 4) ≤ 0) ↔ 
  (x = (-1 - Real.sqrt 17 + Real.sqrt (2 * Real.sqrt 17 + 2)) / 4 ∨ 
   x = (-1 - Real.sqrt 17 - Real.sqrt (2 * Real.sqrt 17 + 2)) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1338_133858


namespace NUMINAMATH_CALUDE_train_crossing_time_l1338_133884

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 105 →
  train_speed_kmh = 54 →
  crossing_time = 7 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1338_133884


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l1338_133854

-- Define the probability of a boy or girl being born
def prob_boy_or_girl : ℚ := 1 / 2

-- Define the number of children in the family
def num_children : ℕ := 4

-- Theorem statement
theorem prob_at_least_one_boy_and_girl :
  (1 : ℚ) - (prob_boy_or_girl ^ num_children + prob_boy_or_girl ^ num_children) = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_and_girl_l1338_133854


namespace NUMINAMATH_CALUDE_certain_number_bound_l1338_133875

theorem certain_number_bound (n : ℝ) : 
  (∀ x : ℝ, x ≤ 2 → 6.1 * 10^x < n) → n > 610 := by sorry

end NUMINAMATH_CALUDE_certain_number_bound_l1338_133875


namespace NUMINAMATH_CALUDE_function_inequality_l1338_133808

def f (x : ℝ) := x^2 - 2*x

theorem function_inequality (a : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, f x ≤ a^2 + 2*a) → a ∈ Set.Iic (-2) ∪ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1338_133808


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1338_133880

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem: The smallest number of identical cubes needed to fill a box with 
    dimensions 36x45x18 inches is 40 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨36, 45, 18⟩ = 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1338_133880


namespace NUMINAMATH_CALUDE_students_without_A_l1338_133866

/-- Given a class of students and their grades in three subjects, calculate the number of students who didn't receive an A in any subject. -/
theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (science : ℕ) 
  (history_math : ℕ) (history_science : ℕ) (math_science : ℕ) (all_three : ℕ) : 
  total = 40 →
  history = 10 →
  math = 15 →
  science = 8 →
  history_math = 5 →
  history_science = 3 →
  math_science = 4 →
  all_three = 2 →
  total - (history + math + science - history_math - history_science - math_science + all_three) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_without_A_l1338_133866


namespace NUMINAMATH_CALUDE_always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l1338_133819

-- Define the grid
def Grid := Fin 10 → Fin 10 → Bool

-- Define ship sizes
inductive ShipSize
| one
| two
| three
| four

-- Define the list of ships to be placed
def ships : List ShipSize :=
  [ShipSize.four] ++ List.replicate 2 ShipSize.three ++
  List.replicate 3 ShipSize.two ++ List.replicate 4 ShipSize.one

-- Define a valid placement
def isValidPlacement (g : Grid) (s : ShipSize) (x y : Fin 10) (horizontal : Bool) : Prop :=
  sorry

-- Define the theorem for part a
theorem always_possible_largest_to_smallest :
  ∀ (g : Grid),
  ∃ (g' : Grid),
    (∀ s ∈ ships, ∃ x y h, isValidPlacement g' s x y h) ∧
    (∀ x y, g' x y → g x y) :=
  sorry

-- Define the theorem for part b
theorem exists_impossible_smallest_to_largest :
  ∃ (g : Grid),
    (∀ s ∈ (ships.reverse.take (ships.length - 1)),
      ∃ x y h, isValidPlacement g s x y h) ∧
    (∀ x y h, ¬isValidPlacement g ShipSize.four x y h) :=
  sorry

end NUMINAMATH_CALUDE_always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l1338_133819


namespace NUMINAMATH_CALUDE_tetrahedron_inequality_l1338_133855

/-- Represents a tetrahedron with base edge lengths a, b, c, 
    lateral edge lengths x, y, z, and d being the distance from 
    the top vertex to the centroid of the base. -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  d : ℝ

/-- Theorem stating that for any tetrahedron, the sum of lateral edge lengths
    is less than or equal to the sum of base edge lengths plus three times
    the distance from the top vertex to the centroid of the base. -/
theorem tetrahedron_inequality (t : Tetrahedron) : 
  t.x + t.y + t.z ≤ t.a + t.b + t.c + 3 * t.d := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_inequality_l1338_133855


namespace NUMINAMATH_CALUDE_original_deck_size_is_52_l1338_133893

/-- The number of players among whom the deck is distributed -/
def num_players : ℕ := 3

/-- The number of cards each player receives after distribution -/
def cards_per_player : ℕ := 18

/-- The number of cards added to the original deck -/
def added_cards : ℕ := 2

/-- The original number of cards in the deck -/
def original_deck_size : ℕ := num_players * cards_per_player - added_cards

theorem original_deck_size_is_52 : original_deck_size = 52 := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_is_52_l1338_133893


namespace NUMINAMATH_CALUDE_intersection_slope_l1338_133822

/-- Given two lines that intersect at a point, prove the slope of one line. -/
theorem intersection_slope (m : ℝ) : 
  (∀ x y, y = -2 * x + 3 → y = m * x + 4) → -- Line p: y = -2x + 3, Line q: y = mx + 4
  1 = -2 * 1 + 3 →                         -- Point (1, 1) satisfies line p
  1 = m * 1 + 4 →                          -- Point (1, 1) satisfies line q
  m = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_slope_l1338_133822


namespace NUMINAMATH_CALUDE_boat_production_three_months_l1338_133871

def boat_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else if months = 1 then initial
  else initial + boat_production (initial * 3) (months - 1)

theorem boat_production_three_months :
  boat_production 5 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_boat_production_three_months_l1338_133871


namespace NUMINAMATH_CALUDE_divisibility_property_l1338_133841

theorem divisibility_property (a b c d : ℤ) (h : (a - c) ∣ (a * b + c * d)) :
  (a - c) ∣ (a * d + b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1338_133841


namespace NUMINAMATH_CALUDE_sets_intersection_theorem_l1338_133817

def A (p q : ℝ) : Set ℝ := {x | x^2 + p*x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | q*x^2 + p*x + 1 = 0}

theorem sets_intersection_theorem (p q : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ (A p q ∩ B p q).Nonempty ∧ (-2 ∈ A p q) →
  ((p = 1 ∧ q = -2) ∨ (p = 3 ∧ q = 2) ∨ (p = 5/2 ∧ q = 1)) :=
by sorry

end NUMINAMATH_CALUDE_sets_intersection_theorem_l1338_133817


namespace NUMINAMATH_CALUDE_lucy_cookie_sales_l1338_133878

/-- Given that Robyn sold 16 packs of cookies and together with Lucy they sold 35 packs,
    prove that Lucy sold 19 packs. -/
theorem lucy_cookie_sales (robyn_sales : ℕ) (total_sales : ℕ) (h1 : robyn_sales = 16) (h2 : total_sales = 35) :
  total_sales - robyn_sales = 19 := by
  sorry

end NUMINAMATH_CALUDE_lucy_cookie_sales_l1338_133878


namespace NUMINAMATH_CALUDE_number_selection_game_probability_l1338_133891

/-- The probability of not winning a prize in the number selection game -/
def prob_not_win : ℚ := 2499 / 2500

/-- The number of options to choose from -/
def num_options : ℕ := 50

theorem number_selection_game_probability :
  prob_not_win = 1 - (1 / num_options^2) :=
sorry

end NUMINAMATH_CALUDE_number_selection_game_probability_l1338_133891


namespace NUMINAMATH_CALUDE_cricket_count_l1338_133895

theorem cricket_count (initial : Float) (additional : Float) :
  initial = 7.0 → additional = 11.0 → initial + additional = 18.0 := by sorry

end NUMINAMATH_CALUDE_cricket_count_l1338_133895


namespace NUMINAMATH_CALUDE_expression_value_l1338_133827

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = -3) :
  ((2 * x - y)^2 - (x - y) * (x + y) - 2 * y^2) / x = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1338_133827


namespace NUMINAMATH_CALUDE_total_crackers_bought_l1338_133889

/-- The number of boxes of crackers Darren bought -/
def darren_boxes : ℕ := 4

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The number of boxes Calvin bought -/
def calvin_boxes : ℕ := 2 * darren_boxes - 1

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := darren_boxes * crackers_per_box + calvin_boxes * crackers_per_box

theorem total_crackers_bought :
  total_crackers = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_crackers_bought_l1338_133889


namespace NUMINAMATH_CALUDE_john_skateboard_distance_l1338_133810

/-- Represents John's journey with skateboarding distances -/
structure JourneyDistances where
  to_park_skateboard : ℕ
  to_park_walk : ℕ
  to_park_bike : ℕ
  park_jog : ℕ
  from_park_bike : ℕ
  from_park_swim : ℕ
  from_park_skateboard : ℕ

/-- Calculates the total skateboarding distance for John's journey -/
def total_skateboard_distance (j : JourneyDistances) : ℕ :=
  j.to_park_skateboard + j.from_park_skateboard

/-- Theorem: John's total skateboarding distance is 25 miles -/
theorem john_skateboard_distance (j : JourneyDistances)
  (h1 : j.to_park_skateboard = 16)
  (h2 : j.to_park_walk = 8)
  (h3 : j.to_park_bike = 6)
  (h4 : j.park_jog = 3)
  (h5 : j.from_park_bike = 5)
  (h6 : j.from_park_swim = 1)
  (h7 : j.from_park_skateboard = 9) :
  total_skateboard_distance j = 25 := by
  sorry

end NUMINAMATH_CALUDE_john_skateboard_distance_l1338_133810


namespace NUMINAMATH_CALUDE_cosine_calculation_l1338_133863

theorem cosine_calculation : Real.cos (π/3) - 2⁻¹ + Real.sqrt ((-2)^2) - (π-3)^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_calculation_l1338_133863


namespace NUMINAMATH_CALUDE_simplify_expression_l1338_133869

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (97 * x + 45) = 100 * x + 60 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1338_133869


namespace NUMINAMATH_CALUDE_inequality_condition_l1338_133852

def f (x : ℝ) := x^2 - 4*x + 3

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x + 3| < a) ↔ b^2 + 2*b + 3 ≤ a :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l1338_133852


namespace NUMINAMATH_CALUDE_max_a_for_function_inequality_l1338_133849

theorem max_a_for_function_inequality (f : ℝ → ℝ) (h : ∀ x, x ∈ [3, 5] → f x = 2 * x / (x - 1)) :
  (∃ a : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ a) ∧ 
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) →
  (∃ a : ℝ, a = 5/2 ∧ 
   (∀ x, x ∈ [3, 5] → f x ≥ a) ∧
   (∀ b : ℝ, (∀ x, x ∈ [3, 5] → f x ≥ b) → b ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_function_inequality_l1338_133849


namespace NUMINAMATH_CALUDE_leadership_choices_l1338_133890

/-- The number of ways to choose leadership in an organization --/
def choose_leadership (total_members : ℕ) (president_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  let remaining_after_president := total_members - president_count
  let remaining_after_vps := remaining_after_president - vp_count
  let remaining_after_vp1_managers := remaining_after_vps - managers_per_vp
  total_members *
  remaining_after_president *
  (remaining_after_president - 1) *
  (Nat.choose remaining_after_vps managers_per_vp) *
  (Nat.choose remaining_after_vp1_managers managers_per_vp)

/-- Theorem stating the number of ways to choose leadership in the given organization --/
theorem leadership_choices :
  choose_leadership 12 1 2 2 = 554400 :=
by sorry

end NUMINAMATH_CALUDE_leadership_choices_l1338_133890


namespace NUMINAMATH_CALUDE_fraction_simplification_l1338_133870

theorem fraction_simplification : (5 * 6) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1338_133870


namespace NUMINAMATH_CALUDE_investment_ratio_l1338_133851

/-- Given two investors p and q, where p invested 60000 and the profit is divided in the ratio 4:6,
    prove that q invested 90000. -/
theorem investment_ratio (p q : ℕ) (h1 : p = 60000) (h2 : 4 * q = 6 * p) : q = 90000 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l1338_133851


namespace NUMINAMATH_CALUDE_average_headcount_l1338_133885

def spring_05_06 : ℕ := 11200
def fall_05_06 : ℕ := 11100
def spring_06_07 : ℕ := 10800
def fall_06_07 : ℕ := 11000  -- approximated due to report error

def total_headcount : ℕ := spring_05_06 + fall_05_06 + spring_06_07 + fall_06_07
def num_terms : ℕ := 4

theorem average_headcount : 
  (total_headcount : ℚ) / num_terms = 11025 := by sorry

end NUMINAMATH_CALUDE_average_headcount_l1338_133885


namespace NUMINAMATH_CALUDE_cookout_attendance_l1338_133857

theorem cookout_attendance (kids_2004 kids_2005 kids_2006 : ℕ) : 
  kids_2005 = kids_2004 / 2 →
  kids_2006 = (2 * kids_2005) / 3 →
  kids_2006 = 20 →
  kids_2004 = 60 := by
sorry

end NUMINAMATH_CALUDE_cookout_attendance_l1338_133857


namespace NUMINAMATH_CALUDE_min_value_expression_l1338_133865

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) - x * y * z ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1338_133865


namespace NUMINAMATH_CALUDE_cake_ratio_correct_l1338_133814

/-- The ratio of cakes made each day compared to the previous day -/
def cake_ratio : ℝ := 2

/-- The number of cakes made on the first day -/
def first_day_cakes : ℕ := 10

/-- The number of cakes made on the sixth day -/
def sixth_day_cakes : ℕ := 320

/-- Theorem stating that the cake ratio is correct given the conditions -/
theorem cake_ratio_correct :
  (first_day_cakes : ℝ) * cake_ratio ^ 5 = sixth_day_cakes := by sorry

end NUMINAMATH_CALUDE_cake_ratio_correct_l1338_133814


namespace NUMINAMATH_CALUDE_twenty_fifth_digit_sum_eighths_quarters_l1338_133831

theorem twenty_fifth_digit_sum_eighths_quarters : ∃ (s : ℚ), 
  (s = 1/8 + 1/4) ∧ 
  (∃ (d : ℕ → ℕ), (∀ n, d n < 10) ∧ 
    (s = ∑' n, (d n : ℚ) / 10^(n+1)) ∧ 
    (d 24 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_digit_sum_eighths_quarters_l1338_133831


namespace NUMINAMATH_CALUDE_warehouse_analysis_l1338_133856

/-- Represents the daily record of material movement --/
structure MaterialRecord where
  quantity : Int
  times : Nat

/-- Calculates the net change in material quantity --/
def netChange (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + r.quantity * r.times) 0

/-- Calculates the transportation cost for Option 1 --/
def costOption1 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r =>
    acc + (if r.quantity > 0 then 5 else 8) * r.quantity.natAbs * r.times
  ) 0

/-- Calculates the transportation cost for Option 2 --/
def costOption2 (records : List MaterialRecord) : Int :=
  records.foldl (fun acc r => acc + 6 * r.quantity.natAbs * r.times) 0

theorem warehouse_analysis (records : List MaterialRecord) :
  records = [
    { quantity := -3, times := 2 },
    { quantity := 4, times := 1 },
    { quantity := -1, times := 3 },
    { quantity := 2, times := 3 },
    { quantity := -5, times := 2 }
  ] →
  netChange records = -9 ∧ costOption2 records < costOption1 records := by
  sorry

end NUMINAMATH_CALUDE_warehouse_analysis_l1338_133856


namespace NUMINAMATH_CALUDE_carmichael_family_children_l1338_133804

/-- The Carmichael family problem -/
theorem carmichael_family_children (f : ℝ) (x : ℝ) (y : ℝ) : 
  (45 + f + x * y) / (2 + x) = 25 →   -- average age of the family
  (f + x * y) / (1 + x) = 20 →        -- average age of father and children
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_carmichael_family_children_l1338_133804


namespace NUMINAMATH_CALUDE_complex_equality_l1338_133899

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l1338_133899


namespace NUMINAMATH_CALUDE_subset_ratio_eight_elements_l1338_133859

theorem subset_ratio_eight_elements : 
  let n : ℕ := 8
  let total_subsets : ℕ := 2^n
  let three_elem_subsets : ℕ := n.choose 3
  (three_elem_subsets : ℚ) / total_subsets = 7/32 := by sorry

end NUMINAMATH_CALUDE_subset_ratio_eight_elements_l1338_133859
