import Mathlib

namespace NUMINAMATH_CALUDE_extreme_value_condition_l3464_346461

/-- If f(x) = m cos x + (1/2) sin 2x reaches an extreme value at x = π/4, then m = 0 -/
theorem extreme_value_condition (m : ℝ) : 
  let f := fun (x : ℝ) => m * Real.cos x + (1/2) * Real.sin (2*x)
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (π/4 + h) ≤ f (π/4)) →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l3464_346461


namespace NUMINAMATH_CALUDE_seed_fertilizer_ratio_is_three_to_one_l3464_346451

/-- Given a total amount of seed and fertilizer, and the amount of seed,
    calculate the ratio of seed to fertilizer. -/
def seedFertilizerRatio (total : ℚ) (seed : ℚ) : ℚ :=
  seed / (total - seed)

/-- Theorem stating that given 60 gallons total and 45 gallons of seed,
    the ratio of seed to fertilizer is 3:1. -/
theorem seed_fertilizer_ratio_is_three_to_one :
  seedFertilizerRatio 60 45 = 3 := by
  sorry

#eval seedFertilizerRatio 60 45

end NUMINAMATH_CALUDE_seed_fertilizer_ratio_is_three_to_one_l3464_346451


namespace NUMINAMATH_CALUDE_cornbread_pieces_l3464_346492

-- Define the dimensions of the pan
def pan_length : ℕ := 20
def pan_width : ℕ := 18

-- Define the dimensions of each piece of cornbread
def piece_length : ℕ := 2
def piece_width : ℕ := 2

-- Theorem to prove
theorem cornbread_pieces :
  (pan_length * pan_width) / (piece_length * piece_width) = 90 := by
  sorry


end NUMINAMATH_CALUDE_cornbread_pieces_l3464_346492


namespace NUMINAMATH_CALUDE_negation_equivalence_l3464_346447

theorem negation_equivalence :
  (¬ (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0)) ↔
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3464_346447


namespace NUMINAMATH_CALUDE_irrational_functional_equation_implies_constant_l3464_346428

/-- A function satisfying f(ab) = f(a+b) for all irrational a and b -/
def IrrationalFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, Irrational a → Irrational b → f (a * b) = f (a + b)

/-- Theorem: If a function satisfies the irrational functional equation, then it is constant -/
theorem irrational_functional_equation_implies_constant
  (f : ℝ → ℝ) (h : IrrationalFunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_irrational_functional_equation_implies_constant_l3464_346428


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3464_346443

/-- Given three rectangles with specific side length ratios, prove the ratio of areas -/
theorem rectangle_area_ratio (a b c d e f : ℝ) 
  (h1 : a / c = 3 / 5) 
  (h2 : b / d = 3 / 5) 
  (h3 : a / e = 7 / 4) 
  (h4 : b / f = 7 / 4) 
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) (h10 : f ≠ 0) :
  (a * b) / ((c * d) + (e * f)) = 441 / 1369 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l3464_346443


namespace NUMINAMATH_CALUDE_function_comparison_l3464_346487

theorem function_comparison (a : ℝ) (h_a : a > 1/2) :
  ∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2,
    (1/2 * a * x₁^2 - (2*a + 1) * x₁ + 21) < (x₂^2 - 2*x₂ + Real.exp x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l3464_346487


namespace NUMINAMATH_CALUDE_rectangle_triangle_length_l3464_346484

/-- Given a rectangle PQRS with PQ = 4 cm, QR = 10 cm, and PM = MQ,
    if the area of triangle PMQ is half the area of rectangle PQRS,
    then the length of segment MQ is 2√10 cm. -/
theorem rectangle_triangle_length (P Q R S M : ℝ × ℝ) : 
  let pq := dist P Q
  let qr := dist Q R
  let pm := dist P M
  let mq := dist M Q
  let area_rect := pq * qr
  let area_tri := (1/2) * pm * mq
  pq = 4 →
  qr = 10 →
  pm = mq →
  area_tri = (1/2) * area_rect →
  mq = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_length_l3464_346484


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l3464_346421

/-- The shortest length of the tangent from a point on the line x - y + 2√2 = 0 to the circle x² + y² = 1 is √3 -/
theorem shortest_tangent_length (x y : ℝ) : 
  (x - y + 2 * Real.sqrt 2 = 0) →
  (x^2 + y^2 = 1) →
  ∃ (px py : ℝ), 
    (px - py + 2 * Real.sqrt 2 = 0) ∧
    Real.sqrt ((px - x)^2 + (py - y)^2) ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l3464_346421


namespace NUMINAMATH_CALUDE_shortest_chord_length_l3464_346437

/-- The circle with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- The center of circle1 -/
def center1 : ℝ × ℝ := (1, 2)

/-- The line of symmetry for circle1 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p = t • center1}

/-- The circle with center (0,0) and radius 3 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

/-- The shortest chord length theorem -/
theorem shortest_chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ circle2 ∧ B ∈ circle2 ∧ A ∈ l ∧ B ∈ l ∧
    ∀ (C D : ℝ × ℝ), C ∈ circle2 → D ∈ circle2 → C ∈ l → D ∈ l →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_l3464_346437


namespace NUMINAMATH_CALUDE_infinitely_many_superabundant_l3464_346434

/-- Sum of divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Superabundant number -/
def is_superabundant (m : ℕ+) : Prop :=
  ∀ k : ℕ+, k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

/-- There are infinitely many superabundant numbers -/
theorem infinitely_many_superabundant :
  ∀ N : ℕ, ∃ m : ℕ+, m > N ∧ is_superabundant m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_superabundant_l3464_346434


namespace NUMINAMATH_CALUDE_plaster_cost_per_sq_meter_l3464_346431

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total surface area of a rectangular tank that needs to be plastered -/
def totalPlasterArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.depth + d.width * d.depth) + d.length * d.width

/-- Theorem: Given a rectangular tank with dimensions 25m x 12m x 6m and a total plastering cost of 223.2 paise, 
    the cost per square meter of plastering is 0.3 paise -/
theorem plaster_cost_per_sq_meter (tank : TankDimensions) 
  (h1 : tank.length = 25)
  (h2 : tank.width = 12)
  (h3 : tank.depth = 6)
  (total_cost : ℝ)
  (h4 : total_cost = 223.2) : 
  total_cost / totalPlasterArea tank = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_plaster_cost_per_sq_meter_l3464_346431


namespace NUMINAMATH_CALUDE_hot_dogs_leftover_l3464_346450

theorem hot_dogs_leftover : 20146130 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_leftover_l3464_346450


namespace NUMINAMATH_CALUDE_orange_distribution_difference_l3464_346418

/-- Calculates the difference in oranges per student between initial and final distribution --/
def orange_difference (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) : ℕ :=
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students)

theorem orange_distribution_difference :
  orange_difference 108 36 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_difference_l3464_346418


namespace NUMINAMATH_CALUDE_gcd_equals_2023_l3464_346473

theorem gcd_equals_2023 (a b c : ℕ+) 
  (h : Nat.gcd a b + Nat.gcd a c + Nat.gcd b c = b + c + 2023) : 
  Nat.gcd b c = 2023 := by
  sorry

end NUMINAMATH_CALUDE_gcd_equals_2023_l3464_346473


namespace NUMINAMATH_CALUDE_tony_pills_l3464_346470

/-- The number of pills left in Tony's bottle after his treatment --/
def pills_left : ℕ :=
  let initial_pills : ℕ := 50
  let first_two_days : ℕ := 2 * 3 * 2
  let next_three_days : ℕ := 1 * 3 * 3
  let last_day : ℕ := 2
  initial_pills - (first_two_days + next_three_days + last_day)

theorem tony_pills : pills_left = 27 := by
  sorry

end NUMINAMATH_CALUDE_tony_pills_l3464_346470


namespace NUMINAMATH_CALUDE_sum_inequality_l3464_346478

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 + 2*b^2 + 3) + 1 / (b^2 + 2*c^2 + 3) + 1 / (c^2 + 2*a^2 + 3) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3464_346478


namespace NUMINAMATH_CALUDE_school_experiment_l3464_346423

theorem school_experiment (boys girls : ℕ) (h1 : boys = 100) (h2 : girls = 125) : 
  (girls - boys) / girls * 100 = 20 ∧ (girls - boys) / boys * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_school_experiment_l3464_346423


namespace NUMINAMATH_CALUDE_complement_A_intersection_B_range_l3464_346439

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x : ℝ | |x| > a}

-- Define the complement of A with respect to U
def complementA : Set ℝ := Set.Icc (-1) 3

theorem complement_A_intersection_B_range :
  (∀ a : ℝ, (complementA ∩ B a).Nonempty) ↔ a ∈ Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersection_B_range_l3464_346439


namespace NUMINAMATH_CALUDE_shoe_alteration_cost_l3464_346416

def total_pairs : ℕ := 14
def sneaker_cost : ℕ := 37
def high_heel_cost : ℕ := 44
def boot_cost : ℕ := 52
def sneaker_pairs : ℕ := 5
def high_heel_pairs : ℕ := 4
def boot_pairs : ℕ := total_pairs - sneaker_pairs - high_heel_pairs
def discount_threshold : ℕ := 10
def discount_per_shoe : ℕ := 2

def total_cost : ℕ := 
  sneaker_pairs * 2 * sneaker_cost + 
  high_heel_pairs * 2 * high_heel_cost + 
  boot_pairs * 2 * boot_cost

def discounted_pairs : ℕ := max (total_pairs - discount_threshold) 0

def total_discount : ℕ := discounted_pairs * 2 * discount_per_shoe

theorem shoe_alteration_cost : 
  total_cost - total_discount = 1226 := by sorry

end NUMINAMATH_CALUDE_shoe_alteration_cost_l3464_346416


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_ratio_l3464_346424

/-- An obtuse triangle with sides a, b, and c, where a is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  a_longest : a ≥ b ∧ a ≥ c
  obtuse : a^2 > b^2 + c^2

/-- The ratio of the sum of squares of two shorter sides to the square of the longest side
    in an obtuse triangle is always greater than or equal to 1/2 -/
theorem obtuse_triangle_side_ratio (t : ObtuseTriangle) :
  (t.b^2 + t.c^2) / t.a^2 ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_ratio_l3464_346424


namespace NUMINAMATH_CALUDE_trailing_zeroes_89_factorial_plus_97_factorial_l3464_346404

/-- The number of trailing zeroes in a natural number -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeroes in 89! + 97! is 20 -/
theorem trailing_zeroes_89_factorial_plus_97_factorial :
  trailingZeroes (factorial 89 + factorial 97) = 20 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_89_factorial_plus_97_factorial_l3464_346404


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_even_integers_l3464_346485

theorem greatest_sum_consecutive_even_integers (n : ℕ) :
  n % 2 = 0 →  -- n is even
  n * (n + 2) < 800 →  -- product is less than 800
  ∀ m : ℕ, m % 2 = 0 →  -- for all even m
    m * (m + 2) < 800 →  -- whose product with its consecutive even is less than 800
    n + (n + 2) ≥ m + (m + 2) →  -- n and n+2 have the greatest sum
  n + (n + 2) = 54  -- the greatest sum is 54
:= by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_even_integers_l3464_346485


namespace NUMINAMATH_CALUDE_at_least_one_is_one_l3464_346413

theorem at_least_one_is_one (x y z : ℝ) 
  (h1 : (1 / x) + (1 / y) + (1 / z) = 1) 
  (h2 : 1 / (x + y + z) = 1) : 
  x = 1 ∨ y = 1 ∨ z = 1 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_is_one_l3464_346413


namespace NUMINAMATH_CALUDE_pyramid_height_l3464_346457

/-- The height of a triangular pyramid with a right-angled base and equal lateral edges -/
theorem pyramid_height (a b l : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 0 < l) :
  let h := (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2)
  ∃ (h : ℝ), h > 0 ∧ h = (1 / 2 : ℝ) * Real.sqrt (4 * l^2 - a^2 - b^2) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_l3464_346457


namespace NUMINAMATH_CALUDE_truck_fill_rate_l3464_346475

/-- The rate at which a person can fill a truck with stone blocks per hour -/
def fill_rate : ℕ → Prop :=
  λ r => 
    -- Truck capacity
    let capacity : ℕ := 6000
    -- Number of people working initially
    let initial_workers : ℕ := 2
    -- Number of hours initial workers work
    let initial_hours : ℕ := 4
    -- Total number of workers after more join
    let total_workers : ℕ := 8
    -- Number of hours all workers work together
    let final_hours : ℕ := 2
    -- Total time to fill the truck
    let total_time : ℕ := 6

    -- The truck is filled when the sum of blocks filled in both phases equals the capacity
    (initial_workers * initial_hours * r) + (total_workers * final_hours * r) = capacity

theorem truck_fill_rate : fill_rate 250 := by
  sorry

end NUMINAMATH_CALUDE_truck_fill_rate_l3464_346475


namespace NUMINAMATH_CALUDE_class_b_more_consistent_l3464_346453

/-- Represents the variance of a class's test scores -/
structure ClassVariance where
  value : ℝ
  is_nonneg : value ≥ 0

/-- Determines if one class has more consistent scores than another based on their variances -/
def has_more_consistent_scores (class_a class_b : ClassVariance) : Prop :=
  class_a.value > class_b.value

theorem class_b_more_consistent :
  let class_a : ClassVariance := ⟨2.56, by norm_num⟩
  let class_b : ClassVariance := ⟨1.92, by norm_num⟩
  has_more_consistent_scores class_b class_a := by
  sorry

end NUMINAMATH_CALUDE_class_b_more_consistent_l3464_346453


namespace NUMINAMATH_CALUDE_triangle_right_angled_l3464_346409

theorem triangle_right_angled (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (eq : 2 * (a^8 + b^8 + c^8) = (a^4 + b^4 + c^4)^2) : 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l3464_346409


namespace NUMINAMATH_CALUDE_unique_increasing_function_theorem_l3464_346471

def IncreasingFunction (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x ≤ y → f x ≤ f y

theorem unique_increasing_function_theorem (f : ℕ+ → ℕ+) 
  (h_increasing : IncreasingFunction f)
  (h_inequality : ∀ x : ℕ+, (f x) * (f (f x)) ≤ x^2) :
  ∀ x : ℕ+, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_increasing_function_theorem_l3464_346471


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l3464_346481

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / (3 * 1000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l3464_346481


namespace NUMINAMATH_CALUDE_sequence_product_l3464_346462

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l3464_346462


namespace NUMINAMATH_CALUDE_y_intercept_of_line_a_l3464_346472

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The given line y = 2x + 4 -/
def given_line : Line :=
  { slope := 2, point := (0, 4) }

/-- Line a, which is parallel to the given line and passes through (2, 5) -/
def line_a : Line :=
  { slope := given_line.slope, point := (2, 5) }

theorem y_intercept_of_line_a :
  y_intercept line_a = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_a_l3464_346472


namespace NUMINAMATH_CALUDE_triangles_on_square_sides_l3464_346433

/-- The number of triangles formed by 12 points on the sides of a square -/
def num_triangles_on_square_sides : ℕ := 216

/-- The total number of points on the sides of the square -/
def total_points : ℕ := 12

/-- The number of sides of the square -/
def num_sides : ℕ := 4

/-- The number of points on each side of the square (excluding vertices) -/
def points_per_side : ℕ := 3

/-- Theorem stating the number of triangles formed by points on square sides -/
theorem triangles_on_square_sides :
  num_triangles_on_square_sides = 
    (total_points.choose 3) - (num_sides * points_per_side.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_triangles_on_square_sides_l3464_346433


namespace NUMINAMATH_CALUDE_cube_painting_probability_l3464_346468

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with six faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube has no adjacent red faces -/
def noAdjacentRed (c : Cube) : Prop := sorry

/-- Counts the number of valid cube paintings -/
def validPaintings : ℕ := sorry

/-- Checks if two cubes can be rotated to look identical -/
def canRotateIdentical (c1 c2 : Cube) : Prop := sorry

/-- Counts the number of ways two cubes can be painted to look identical after rotation -/
def identicalAppearances : ℕ := sorry

/-- The main theorem stating the probability of two cubes being painted and rotatable to look identical -/
theorem cube_painting_probability :
  (identicalAppearances : ℚ) / (validPaintings^2 : ℚ) = 1 / 5776 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l3464_346468


namespace NUMINAMATH_CALUDE_older_brother_pocket_money_l3464_346489

theorem older_brother_pocket_money
  (total_money : ℕ)
  (difference : ℕ)
  (h1 : total_money = 12000)
  (h2 : difference = 1000) :
  ∃ (younger older : ℕ),
    younger + older = total_money ∧
    older = younger + difference ∧
    older = 6500 := by
  sorry

end NUMINAMATH_CALUDE_older_brother_pocket_money_l3464_346489


namespace NUMINAMATH_CALUDE_domino_double_cover_l3464_346432

/-- Represents a domino tile placement on a 2×2 square -/
inductive DominoPlacement
  | Horizontal
  | Vertical

/-- Represents a tiling of a 2n × 2m rectangle using 1 × 2 domino tiles -/
def Tiling (n m : ℕ) := Fin n → Fin m → DominoPlacement

/-- Checks if two tilings are complementary (non-overlapping) -/
def complementary (t1 t2 : Tiling n m) : Prop :=
  ∀ i j, t1 i j ≠ t2 i j

theorem domino_double_cover (n m : ℕ) :
  ∃ (t1 t2 : Tiling n m), complementary t1 t2 := by sorry

end NUMINAMATH_CALUDE_domino_double_cover_l3464_346432


namespace NUMINAMATH_CALUDE_range_of_m_l3464_346469

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Define the interval [2, 4]
def I : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∃ x ∈ I, m - f x > 0) → m > 5 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3464_346469


namespace NUMINAMATH_CALUDE_library_visitors_average_l3464_346410

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 4 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 276 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3464_346410


namespace NUMINAMATH_CALUDE_eggs_per_basket_l3464_346452

theorem eggs_per_basket (yellow_eggs : Nat) (pink_eggs : Nat) (min_eggs : Nat) : 
  yellow_eggs = 30 → pink_eggs = 45 → min_eggs = 5 → 
  ∃ (eggs_per_basket : Nat), 
    eggs_per_basket ∣ yellow_eggs ∧ 
    eggs_per_basket ∣ pink_eggs ∧ 
    eggs_per_basket ≥ min_eggs ∧
    ∀ (n : Nat), n ∣ yellow_eggs → n ∣ pink_eggs → n ≥ min_eggs → n ≤ eggs_per_basket :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l3464_346452


namespace NUMINAMATH_CALUDE_tony_school_years_l3464_346459

/-- The total number of years Tony went to school to become an astronaut -/
def total_school_years (first_degree_years : ℕ) (additional_degrees : ℕ) (graduate_degree_years : ℕ) : ℕ :=
  first_degree_years + additional_degrees * first_degree_years + graduate_degree_years

/-- Theorem stating that Tony went to school for 14 years -/
theorem tony_school_years :
  total_school_years 4 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tony_school_years_l3464_346459


namespace NUMINAMATH_CALUDE_power_sum_problem_l3464_346455

theorem power_sum_problem : ∃ (x y n : ℕ+), 
  (x * y = 6) ∧ 
  (x ^ n.val + y ^ n.val = 35) ∧ 
  (∀ m : ℕ+, m < n → x ^ m.val + y ^ m.val ≠ 35) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3464_346455


namespace NUMINAMATH_CALUDE_pierre_birthday_l3464_346427

/-- Represents a date with day and month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a person's age and birthday -/
structure Person where
  age : Nat
  birthday : Date

def nextYear (d : Date) : Date :=
  if d.month = 12 && d.day = 31 then { day := 1, month := 1 }
  else { day := d.day, month := d.month }

def yesterday (d : Date) : Date :=
  if d.day = 1 && d.month = 1 then { day := 31, month := 12 }
  else if d.day = 1 then { day := 31, month := d.month - 1 }
  else { day := d.day - 1, month := d.month }

def dayBeforeYesterday (d : Date) : Date := yesterday (yesterday d)

theorem pierre_birthday (today : Date) (pierre : Person) : 
  pierre.age = 11 → 
  (dayBeforeYesterday today).day = 31 → 
  (dayBeforeYesterday today).month = 12 →
  pierre.birthday = yesterday today →
  (nextYear today).day = 1 → 
  (nextYear today).month = 1 →
  today.day = 1 ∧ today.month = 1 := by
  sorry

#check pierre_birthday

end NUMINAMATH_CALUDE_pierre_birthday_l3464_346427


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_mans_upstream_speed_is_twelve_l3464_346456

/-- Calculates the speed of a man rowing upstream given his speed in still water and his speed downstream. -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that for a given man's speed in still water and downstream, 
    his upstream speed is equal to twice his still water speed minus his downstream speed. -/
theorem upstream_speed_calculation 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0)
  (h2 : speed_downstream > speed_still) :
  speed_upstream speed_still speed_downstream = 2 * speed_still - speed_downstream :=
by sorry

/-- The speed of the man rowing upstream in the given problem. -/
def mans_upstream_speed : ℝ := speed_upstream 25 38

/-- Theorem proving that the man's upstream speed in the given problem is 12 km/h. -/
theorem mans_upstream_speed_is_twelve : 
  mans_upstream_speed = 12 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_calculation_mans_upstream_speed_is_twelve_l3464_346456


namespace NUMINAMATH_CALUDE_paint_calculation_l3464_346480

/-- Given three people painting a wall with a work ratio and total area,
    calculate the area painted by the third person. -/
theorem paint_calculation (ratio_a ratio_b ratio_c total_area : ℕ) 
    (ratio_positive : ratio_a > 0 ∧ ratio_b > 0 ∧ ratio_c > 0)
    (total_positive : total_area > 0) :
    let total_ratio := ratio_a + ratio_b + ratio_c
    ratio_c * total_area / total_ratio = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3464_346480


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3464_346414

theorem polynomial_factorization (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3464_346414


namespace NUMINAMATH_CALUDE_ferry_position_after_202_trips_l3464_346490

/-- Represents the shore where the ferry can be docked --/
inductive Shore : Type
  | South : Shore
  | North : Shore

/-- Determines the shore where the ferry is docked after a given number of trips --/
def ferry_position (start : Shore) (trips : ℕ) : Shore :=
  if trips % 2 = 0 then start else
    match start with
    | Shore.South => Shore.North
    | Shore.North => Shore.South

/-- Theorem stating that after 202 trips starting from the south shore, the ferry ends up on the south shore --/
theorem ferry_position_after_202_trips :
  ferry_position Shore.South 202 = Shore.South := by
  sorry

end NUMINAMATH_CALUDE_ferry_position_after_202_trips_l3464_346490


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3464_346400

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 136) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3464_346400


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3464_346493

theorem cube_volume_from_surface_area (surface_area : ℝ) (h : surface_area = 600) :
  let side_length := Real.sqrt (surface_area / 6)
  side_length ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3464_346493


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l3464_346438

theorem real_part_of_complex_square : Complex.re ((1 + 2 * Complex.I) ^ 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l3464_346438


namespace NUMINAMATH_CALUDE_divisible_by_twenty_l3464_346412

theorem divisible_by_twenty (n : ℕ) : ∃ k : ℤ, 9^(8*n+4) - 7^(8*n+4) = 20*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twenty_l3464_346412


namespace NUMINAMATH_CALUDE_square_difference_quotient_l3464_346474

theorem square_difference_quotient : (347^2 - 333^2) / 14 = 680 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_quotient_l3464_346474


namespace NUMINAMATH_CALUDE_stationery_shop_sales_percentage_l3464_346466

theorem stationery_shop_sales_percentage (pen_sales pencil_sales marker_sales : ℝ) 
  (h_pen : pen_sales = 25)
  (h_pencil : pencil_sales = 30)
  (h_marker : marker_sales = 20)
  (h_total : pen_sales + pencil_sales + marker_sales + (100 - pen_sales - pencil_sales - marker_sales) = 100) :
  100 - pen_sales - pencil_sales - marker_sales = 25 := by
sorry

end NUMINAMATH_CALUDE_stationery_shop_sales_percentage_l3464_346466


namespace NUMINAMATH_CALUDE_mango_profit_percentage_l3464_346449

/-- Represents the rate at which mangoes are bought (number of mangoes per rupee) -/
def buy_rate : ℚ := 6

/-- Represents the rate at which mangoes are sold (number of mangoes per rupee) -/
def sell_rate : ℚ := 3

/-- Calculates the profit percentage given buy and sell rates -/
def profit_percentage (buy : ℚ) (sell : ℚ) : ℚ :=
  ((sell⁻¹ - buy⁻¹) / buy⁻¹) * 100

theorem mango_profit_percentage :
  profit_percentage buy_rate sell_rate = 100 := by
  sorry

end NUMINAMATH_CALUDE_mango_profit_percentage_l3464_346449


namespace NUMINAMATH_CALUDE_car_speed_ratio_l3464_346458

theorem car_speed_ratio : 
  ∀ (speed_A speed_B : ℝ),
    speed_B = 50 →
    speed_A * 6 + speed_B * 2 = 1000 →
    speed_A / speed_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_ratio_l3464_346458


namespace NUMINAMATH_CALUDE_line_x_intercept_x_intercept_is_four_l3464_346454

/-- A line passing through two points (1, 3) and (5, -1) has x-intercept 4 -/
theorem line_x_intercept : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (x_intercept : ℝ) =>
    (slope = ((-1) - 3) / (5 - 1)) ∧
    (3 = slope * (1 - x_intercept)) ∧
    (x_intercept = 4)

/-- The x-intercept of the line passing through (1, 3) and (5, -1) is 4 -/
theorem x_intercept_is_four : ∃ (slope : ℝ), line_x_intercept slope 4 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_x_intercept_is_four_l3464_346454


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3464_346497

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ (k : ℕ), k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3464_346497


namespace NUMINAMATH_CALUDE_prime_power_sum_l3464_346435

theorem prime_power_sum (n : ℕ) : Prime (n^4 + 4^n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l3464_346435


namespace NUMINAMATH_CALUDE_intersection_q_complement_p_l3464_346422

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≥ 9}
def Q : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_q_complement_p :
  Q ∩ (U \ P) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_q_complement_p_l3464_346422


namespace NUMINAMATH_CALUDE_fashion_markup_l3464_346483

theorem fashion_markup (original_price : ℝ) (markup1 markup2 markup3 : ℝ) 
  (h1 : markup1 = 0.35)
  (h2 : markup2 = 0.25)
  (h3 : markup3 = 0.45) :
  let price1 := original_price * (1 + markup1)
  let price2 := price1 * (1 + markup2)
  let final_price := price2 * (1 + markup3)
  (final_price - original_price) / original_price * 100 = 144.69 := by
sorry

end NUMINAMATH_CALUDE_fashion_markup_l3464_346483


namespace NUMINAMATH_CALUDE_five_books_three_bins_l3464_346406

-- Define the Stirling number of the second kind
def stirling2 (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem five_books_three_bins : stirling2 5 3 = 25 := by sorry

end NUMINAMATH_CALUDE_five_books_three_bins_l3464_346406


namespace NUMINAMATH_CALUDE_pigeonhole_mod_three_l3464_346417

theorem pigeonhole_mod_three (s : Finset ℤ) (h : s.card = 6) :
  ∃ (a b c d : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a * b) % 3 = (c * d) % 3 :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_mod_three_l3464_346417


namespace NUMINAMATH_CALUDE_initial_number_of_students_l3464_346429

theorem initial_number_of_students :
  ∀ (n : ℕ) (W : ℝ),
    W = n * 28 →
    W + 10 = (n + 1) * 27.4 →
    n = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_students_l3464_346429


namespace NUMINAMATH_CALUDE_or_true_implications_l3464_346479

theorem or_true_implications (p q : Prop) (h : p ∨ q) :
  ¬(
    ((p ∧ q) = True) ∨
    ((p ∧ q) = False) ∨
    ((¬p ∨ ¬q) = True) ∨
    ((¬p ∨ ¬q) = False)
  ) := by sorry

end NUMINAMATH_CALUDE_or_true_implications_l3464_346479


namespace NUMINAMATH_CALUDE_square_root_of_difference_l3464_346446

theorem square_root_of_difference : 
  Real.sqrt (20212020 * 20202021 - 20212021 * 20202020) = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_difference_l3464_346446


namespace NUMINAMATH_CALUDE_decimal_place_values_l3464_346407

/-- Represents the place value in a decimal number system. -/
inductive PlaceValue
| Ones
| Tens
| Hundreds
| Thousands
| TenThousands
| HundredThousands
| Millions
| TenMillions
| HundredMillions

/-- Returns the position of a place value from right to left. -/
def position (pv : PlaceValue) : Nat :=
  match pv with
  | .Ones => 1
  | .Tens => 2
  | .Hundreds => 3
  | .Thousands => 4
  | .TenThousands => 5
  | .HundredThousands => 6
  | .Millions => 7
  | .TenMillions => 8
  | .HundredMillions => 9

theorem decimal_place_values :
  (position PlaceValue.Hundreds = 3) ∧
  (position PlaceValue.TenThousands = 5) ∧
  (position PlaceValue.Thousands = 4) := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_values_l3464_346407


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3464_346401

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem stating the number of arrangements for 6 books with 3 identical -/
theorem book_arrangement_theorem :
  arrange_books 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3464_346401


namespace NUMINAMATH_CALUDE_range_of_a_l3464_346463

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : (∃ x, x ∈ A ∩ B a) → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3464_346463


namespace NUMINAMATH_CALUDE_prob_rain_holiday_l3464_346460

/-- The probability of rain on Friday without a storm -/
def prob_rain_friday : ℝ := 0.3

/-- The probability of rain on Monday without a storm -/
def prob_rain_monday : ℝ := 0.6

/-- The increase in probability of rain if a storm develops -/
def storm_increase : ℝ := 0.2

/-- The probability of a storm developing -/
def prob_storm : ℝ := 0.5

/-- Assumption that all probabilities are independent -/
axiom probabilities_independent : True

/-- The probability of rain on at least one day during the holiday -/
def prob_rain_at_least_one_day : ℝ := 
  1 - (prob_storm * (1 - (prob_rain_friday + storm_increase)) * (1 - (prob_rain_monday + storm_increase)) + 
       (1 - prob_storm) * (1 - prob_rain_friday) * (1 - prob_rain_monday))

theorem prob_rain_holiday : prob_rain_at_least_one_day = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_prob_rain_holiday_l3464_346460


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3464_346420

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (5 + 2 * z) = 11 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3464_346420


namespace NUMINAMATH_CALUDE_cookie_sale_charity_share_l3464_346494

/-- Calculates the amount each charity receives when John sells cookies and splits the profit. -/
theorem cookie_sale_charity_share :
  let dozen : ℕ := 6
  let cookies_per_dozen : ℕ := 12
  let total_cookies : ℕ := dozen * cookies_per_dozen
  let price_per_cookie : ℚ := 3/2
  let cost_per_cookie : ℚ := 1/4
  let total_revenue : ℚ := total_cookies * price_per_cookie
  let total_cost : ℚ := total_cookies * cost_per_cookie
  let profit : ℚ := total_revenue - total_cost
  let num_charities : ℕ := 2
  let charity_share : ℚ := profit / num_charities
  charity_share = 45
:= by sorry

end NUMINAMATH_CALUDE_cookie_sale_charity_share_l3464_346494


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l3464_346495

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l3464_346495


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l3464_346436

/-- Proves that if a shop owner charges 20% more than the cost price,
    and a customer paid 3600 for an item, then the cost price was 3000. -/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (cost_price : ℝ) : 
  markup_percentage = 0.20 →
  selling_price = 3600 →
  selling_price = cost_price * (1 + markup_percentage) →
  cost_price = 3000 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l3464_346436


namespace NUMINAMATH_CALUDE_camera_pics_count_l3464_346419

/-- Represents the number of pictures in Olivia's photo collection. -/
structure PhotoCollection where
  phone_pics : ℕ
  camera_pics : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The properties of Olivia's photo collection as described in the problem. -/
def olivias_collection : PhotoCollection where
  phone_pics := 5
  camera_pics := 35  -- This is what we want to prove
  albums := 8
  pics_per_album := 5

/-- Theorem stating that the number of pictures from Olivia's camera is 35. -/
theorem camera_pics_count (p : PhotoCollection) 
  (h1 : p.phone_pics = 5)
  (h2 : p.albums = 8)
  (h3 : p.pics_per_album = 5)
  (h4 : p.phone_pics + p.camera_pics = p.albums * p.pics_per_album) :
  p.camera_pics = 35 := by
  sorry

#check camera_pics_count olivias_collection

end NUMINAMATH_CALUDE_camera_pics_count_l3464_346419


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l3464_346496

theorem inverse_variation_sqrt (y : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, x > 0 → y x * Real.sqrt x = k) →  -- y varies inversely as √x
  (y 4 = 8) →                                -- when x = 4, y = 8
  (y 8 = 4 * Real.sqrt 2) :=                 -- y = 4√2 when x = 8
by sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l3464_346496


namespace NUMINAMATH_CALUDE_min_r_for_B_subset_C_l3464_346426

open Set Real

-- Define the sets A, B, and C(r)
def A : Set ℝ := {t | 0 < t ∧ t < 2 * π}

def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = sin t ∧ p.2 = 2 * sin t * cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

-- State the theorem
theorem min_r_for_B_subset_C : 
  (∀ r, B ⊆ C r → r ≥ 5/4) ∧ B ⊆ C (5/4) := by sorry

end NUMINAMATH_CALUDE_min_r_for_B_subset_C_l3464_346426


namespace NUMINAMATH_CALUDE_monomial_exponent_difference_l3464_346486

theorem monomial_exponent_difference (a b : ℤ) : 
  ((-1 : ℚ) * X^3 * Y^1 = X^a * Y^(b-1)) → (a - b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_difference_l3464_346486


namespace NUMINAMATH_CALUDE_cube_three_minus_seven_equals_square_four_plus_four_l3464_346491

theorem cube_three_minus_seven_equals_square_four_plus_four :
  3^3 - 7 = 4^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_three_minus_seven_equals_square_four_plus_four_l3464_346491


namespace NUMINAMATH_CALUDE_cars_without_features_l3464_346430

theorem cars_without_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (both : ℕ)
  (h_total : total = 65)
  (h_airbags : airbags = 45)
  (h_power_windows : power_windows = 30)
  (h_both : both = 12) :
  total - (airbags + power_windows - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cars_without_features_l3464_346430


namespace NUMINAMATH_CALUDE_gulliver_kefir_consumption_l3464_346445

/-- Represents the total number of bottles of kefir Gulliver drinks -/
def total_kefir_bottles (initial_money : ℕ) (initial_price : ℕ) : ℕ :=
  initial_money * 6 / (7 * initial_price)

/-- Theorem stating the total number of kefir bottles Gulliver drinks -/
theorem gulliver_kefir_consumption :
  total_kefir_bottles 7000000 7 = 1166666 := by
  sorry

#eval total_kefir_bottles 7000000 7

end NUMINAMATH_CALUDE_gulliver_kefir_consumption_l3464_346445


namespace NUMINAMATH_CALUDE_engineers_teachers_ratio_l3464_346408

theorem engineers_teachers_ratio (e t : ℕ) (he : e > 0) (ht : t > 0) :
  (40 * e + 55 * t : ℚ) / (e + t) = 46 →
  e / t = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_engineers_teachers_ratio_l3464_346408


namespace NUMINAMATH_CALUDE_multiply_72_28_l3464_346465

theorem multiply_72_28 : 72 * 28 = 4896 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72_28_l3464_346465


namespace NUMINAMATH_CALUDE_binary_sum_equals_158_l3464_346467

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 1001001₂ -/
def binary2 : List Bool := [true, false, false, true, false, false, true]

/-- Theorem stating that the sum of 1010101₂ and 1001001₂ is 158 in decimal -/
theorem binary_sum_equals_158 :
  binaryToDecimal binary1 + binaryToDecimal binary2 = 158 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_158_l3464_346467


namespace NUMINAMATH_CALUDE_sequence_inequality_l3464_346425

/-- A sequence of positive real numbers satisfying the given inequality -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → a i > 0 ∧ i * (a i)^2 ≥ (i + 1) * (a (i - 1)) * (a (i + 1))

/-- Definition of the sequence b in terms of a -/
def b (a : ℕ → ℝ) (x y : ℝ) : ℕ → ℝ :=
  λ i => x * (a i) + y * (a (i - 1))

theorem sequence_inequality (a : ℕ → ℝ) (x y : ℝ) 
    (h_pos : PositiveSequence a) (h_x : x > 0) (h_y : y > 0) :
    ∀ i : ℕ, i ≥ 2 → i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3464_346425


namespace NUMINAMATH_CALUDE_calculation_proof_l3464_346405

theorem calculation_proof : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3464_346405


namespace NUMINAMATH_CALUDE_grid_transform_iff_even_l3464_346402

/-- Represents a grid operation that changes adjacent entries' signs -/
def GridOperation (n : ℕ) := Fin n → Fin n → Unit

/-- Represents the state of the grid -/
def GridState (n : ℕ) := Fin n → Fin n → Int

/-- Initial grid state with all entries 1 -/
def initialGrid (n : ℕ) : GridState n :=
  λ _ _ => 1

/-- Final grid state with all entries -1 -/
def finalGrid (n : ℕ) : GridState n :=
  λ _ _ => -1

/-- Predicate to check if a sequence of operations can transform the grid -/
def canTransform (n : ℕ) : Prop :=
  ∃ (seq : List (GridOperation n)), 
    ∃ (result : GridState n), 
      result = finalGrid n

/-- Main theorem: Grid can be transformed iff n is even -/
theorem grid_transform_iff_even (n : ℕ) (h : n ≥ 2) : 
  canTransform n ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_grid_transform_iff_even_l3464_346402


namespace NUMINAMATH_CALUDE_perpendicular_tangents_sum_l3464_346488

/-- The problem statement -/
theorem perpendicular_tangents_sum (a b : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    -- Point (x₀, y₀) is on both curves
    (y₀ = x₀^2 - 2*x₀ + 2 ∧ y₀ = -x₀^2 + a*x₀ + b) ∧ 
    -- Tangents are perpendicular
    (2*x₀ - 2) * (-2*x₀ + a) = -1) → 
  a + b = 5/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_sum_l3464_346488


namespace NUMINAMATH_CALUDE_taylor_family_reunion_l3464_346477

theorem taylor_family_reunion (kids : ℕ) (adults : ℕ) (tables : ℕ) 
  (h1 : kids = 45)
  (h2 : adults = 123)
  (h3 : tables = 14)
  : (kids + adults) / tables = 12 := by
  sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_l3464_346477


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3464_346476

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432) :
  ∃ (rate : ℝ), 
    rate > 0 ∧ 
    rate = (interest_paid * 100) / (principal * rate) ∧ 
    rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3464_346476


namespace NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l3464_346499

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines determine a unique plane -/
theorem two_intersecting_lines_determine_plane (l1 l2 : Line3D) :
  intersect l1 l2 → ∃! p : Plane3D, lineOnPlane l1 p ∧ lineOnPlane l2 p :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l3464_346499


namespace NUMINAMATH_CALUDE_andy_sock_ratio_l3464_346448

/-- The ratio of white socks to black socks -/
def sock_ratio (white : ℕ) (black : ℕ) : ℚ := white / black

theorem andy_sock_ratio :
  ∀ white : ℕ,
  let black := 6
  white / 2 = black + 6 →
  sock_ratio white black = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_andy_sock_ratio_l3464_346448


namespace NUMINAMATH_CALUDE_age_difference_l3464_346411

def sachin_age : ℕ := 49

theorem age_difference (rahul_age : ℕ) 
  (h1 : sachin_age < rahul_age)
  (h2 : sachin_age * 9 = rahul_age * 7) : 
  rahul_age - sachin_age = 14 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3464_346411


namespace NUMINAMATH_CALUDE_parabola_focus_l3464_346444

/-- The focus of a parabola y^2 = 8x with directrix x + 2 = 0 is at (2,0) -/
theorem parabola_focus (x y : ℝ) : 
  (y^2 = 8*x) →  -- point (x,y) is on the parabola
  (∀ (a b : ℝ), (a + 2 = 0) → ((x - a)^2 + (y - b)^2 = 4)) → -- distance to directrix equals distance to (2,0)
  (x = 2 ∧ y = 0) -- focus is at (2,0)
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3464_346444


namespace NUMINAMATH_CALUDE_division_simplification_l3464_346482

theorem division_simplification (x y : ℝ) (h : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3464_346482


namespace NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l3464_346403

theorem triangle_abc_is_obtuse (A B C : Real) (a b c : Real) :
  B = Real.pi / 6 →  -- 30 degrees in radians
  b = Real.sqrt 2 →
  c = 2 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  c / Real.sin C = a / Real.sin A →
  A > Real.pi / 2 ∨ B > Real.pi / 2 ∨ C > Real.pi / 2 := by
  sorry

#check triangle_abc_is_obtuse

end NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l3464_346403


namespace NUMINAMATH_CALUDE_leesburg_population_l3464_346464

theorem leesburg_population (salem_factor : ℕ) (moved_out : ℕ) (women_ratio : ℚ) (women_count : ℕ) :
  salem_factor = 15 →
  moved_out = 130000 →
  women_ratio = 1/2 →
  women_count = 377050 →
  ∃ (leesburg_pop : ℕ), leesburg_pop = 58940 ∧ 
    salem_factor * leesburg_pop = 2 * women_count + moved_out :=
sorry

end NUMINAMATH_CALUDE_leesburg_population_l3464_346464


namespace NUMINAMATH_CALUDE_seating_arrangements_l3464_346498

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with k specific people consecutive. -/
def consecutiveArrangements (n k : ℕ) : ℕ := 
  (Nat.factorial (n - k + 1)) * (Nat.factorial k)

/-- The number of people to be seated. -/
def totalPeople : ℕ := 10

/-- The number of specific individuals who refuse to sit consecutively. -/
def specificIndividuals : ℕ := 4

/-- The number of ways to arrange people with restrictions. -/
def arrangementsWithRestrictions : ℕ := 
  totalArrangements totalPeople - consecutiveArrangements totalPeople specificIndividuals

theorem seating_arrangements :
  arrangementsWithRestrictions = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3464_346498


namespace NUMINAMATH_CALUDE_problem_solution_l3464_346415

theorem problem_solution (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3464_346415


namespace NUMINAMATH_CALUDE_harmonic_geometric_sequence_ratio_l3464_346442

theorem harmonic_geometric_sequence_ratio (x y z : ℝ) :
  (1 / y - 1 / x) / (1 / x - 1 / z) = 1 →  -- harmonic sequence condition
  (5 * y * z) / (3 * x * y) = (7 * z * x) / (5 * y * z) →  -- geometric sequence condition
  y / z + z / y = 58 / 21 := by sorry

end NUMINAMATH_CALUDE_harmonic_geometric_sequence_ratio_l3464_346442


namespace NUMINAMATH_CALUDE_never_return_to_initial_l3464_346441

def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

def iterate_transform (q : ℝ × ℝ × ℝ × ℝ) (n : ℕ) : ℝ × ℝ × ℝ × ℝ :=
  match n with
  | 0 => q
  | n + 1 => transform (iterate_transform q n)

theorem never_return_to_initial (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hna : a ≠ 1) (hnb : b ≠ 1) (hnc : c ≠ 1) (hnd : d ≠ 1) :
  ∀ n : ℕ, iterate_transform (a, b, c, d) n ≠ (a, b, c, d) :=
sorry

end NUMINAMATH_CALUDE_never_return_to_initial_l3464_346441


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_plus_x_l3464_346440

theorem factorization_of_x_squared_plus_x (x : ℝ) : x^2 + x = x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_plus_x_l3464_346440
