import Mathlib

namespace NUMINAMATH_CALUDE_horner_method_f_2_l484_48470

def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 7

theorem horner_method_f_2 : f 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l484_48470


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l484_48404

/-- The inverse relationship between y^5 and z^(1/5) -/
def inverse_relation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^5 * z^(1/5) = k

theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_relation y₁ z₁)
  (h2 : inverse_relation y₂ z₂)
  (h3 : y₁ = 3)
  (h4 : z₁ = 8)
  (h5 : y₂ = 6) :
  z₂ = 1 / 1048576 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l484_48404


namespace NUMINAMATH_CALUDE_min_value_expression_l484_48437

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (y^2 / (x + 1)) + (x^2 / (y + 1)) ≥ 9/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l484_48437


namespace NUMINAMATH_CALUDE_right_angle_sufficiency_not_necessity_l484_48487

theorem right_angle_sufficiency_not_necessity (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < A) ∧ (0 < B) ∧ (0 < C) ∧ (A + B + C = π) →
  -- 1. If angle C is 90°, then cos A + sin A = cos B + sin B
  (C = π / 2 → Real.cos A + Real.sin A = Real.cos B + Real.sin B) ∧
  -- 2. There exists a triangle where cos A + sin A = cos B + sin B, but angle C ≠ 90°
  ∃ (A' B' C' : ℝ), (0 < A') ∧ (0 < B') ∧ (0 < C') ∧ (A' + B' + C' = π) ∧
    (Real.cos A' + Real.sin A' = Real.cos B' + Real.sin B') ∧ (C' ≠ π / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_angle_sufficiency_not_necessity_l484_48487


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l484_48454

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 50) 
  (h3 : b * c = 75) : 
  a * b * c = 150 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l484_48454


namespace NUMINAMATH_CALUDE_theater_revenue_l484_48478

theorem theater_revenue (n : ℕ) (cost total_revenue actual_revenue : ℝ) :
  (total_revenue = cost * 1.2) →
  (actual_revenue = total_revenue * 0.95) →
  (actual_revenue = cost * 1.14) :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l484_48478


namespace NUMINAMATH_CALUDE_common_divisors_9240_8000_l484_48499

theorem common_divisors_9240_8000 : ∃ n : ℕ, n = (Nat.divisors (Nat.gcd 9240 8000)).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_9240_8000_l484_48499


namespace NUMINAMATH_CALUDE_product_of_roots_l484_48442

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → 
  ∃ x₁ x₂ : ℝ, x₁ * x₂ = -34 ∧ (x₁ + 3) * (x₁ - 4) = 22 ∧ (x₂ + 3) * (x₂ - 4) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l484_48442


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l484_48461

theorem inequality_not_always_true (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  (∃ c, ¬(a * c > b * c)) ∧ 
  (∀ c, a * c + c > b * c + c) ∧ 
  (∀ c, a - c^2 > b - c^2) ∧ 
  (∀ c, a + c^3 > b + c^3) ∧ 
  (∀ c, a * c^3 > b * c^3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l484_48461


namespace NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_l484_48422

/-- Represents the total number of products in the box -/
def total_products : ℕ := 5

/-- Represents the number of first-class products in the box -/
def first_class_products : ℕ := 3

/-- Represents the number of second-class products in the box -/
def second_class_products : ℕ := 2

/-- Represents the probability of drawing a second-class item on the first draw -/
def prob_second_class_first : ℚ := second_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a second-class item was drawn first -/
def prob_first_class_second_given_second_first : ℚ := first_class_products / (total_products - 1)

/-- Represents the probability of drawing a first-class item on the first draw -/
def prob_first_class_first : ℚ := first_class_products / total_products

/-- Represents the probability of drawing a first-class item on the second draw,
    given that a first-class item was drawn first -/
def prob_first_class_second_given_first_first : ℚ := (first_class_products - 1) / (total_products - 1)

theorem probability_second_class_first_given_first_class_second :
  (prob_second_class_first * prob_first_class_second_given_second_first) /
  (prob_second_class_first * prob_first_class_second_given_second_first +
   prob_first_class_first * prob_first_class_second_given_first_first) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_class_first_given_first_class_second_l484_48422


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l484_48416

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 : 
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l484_48416


namespace NUMINAMATH_CALUDE_total_practice_time_is_135_l484_48485

/-- The number of minutes Daniel practices basketball each day during the school week -/
def school_day_practice : ℕ := 15

/-- The number of days in a school week -/
def school_week_days : ℕ := 5

/-- The number of days in a weekend -/
def weekend_days : ℕ := 2

/-- The total number of minutes Daniel practices during a whole week -/
def total_practice_time : ℕ :=
  (school_day_practice * school_week_days) +
  (2 * school_day_practice * weekend_days)

theorem total_practice_time_is_135 :
  total_practice_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_135_l484_48485


namespace NUMINAMATH_CALUDE_tiled_polygon_sides_l484_48474

/-- A tile is either a square or an equilateral triangle with side length 1 -/
inductive Tile
| Square
| EquilateralTriangle

/-- A convex polygon formed by tiles -/
structure TiledPolygon where
  sides : ℕ
  tiles : List Tile
  is_convex : Bool
  no_gaps : Bool
  no_overlap : Bool

/-- The theorem stating the possible number of sides for a convex polygon formed by tiles -/
theorem tiled_polygon_sides (p : TiledPolygon) (h_convex : p.is_convex = true) 
  (h_no_gaps : p.no_gaps = true) (h_no_overlap : p.no_overlap = true) : 
  3 ≤ p.sides ∧ p.sides ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_tiled_polygon_sides_l484_48474


namespace NUMINAMATH_CALUDE_range_of_a_l484_48401

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically increasing on [0, +∞)
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) 
  (h_even : IsEven f) 
  (h_mono : IsMonoIncreasing f) 
  (h_ineq : f (a - 3) < f 4) : 
  -1 < a ∧ a < 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l484_48401


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l484_48448

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 20) →
  p + q = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l484_48448


namespace NUMINAMATH_CALUDE_batch_size_proof_l484_48400

theorem batch_size_proof (n : ℕ) : 
  500 ≤ n ∧ n ≤ 600 ∧ 
  n % 20 = 13 ∧ 
  n % 27 = 20 → 
  n = 533 :=
sorry

end NUMINAMATH_CALUDE_batch_size_proof_l484_48400


namespace NUMINAMATH_CALUDE_ellipse_equation_l484_48435

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    is_ellipse x1 y1 a b ∧ 
    is_ellipse x2 y2 a b ∧ 
    on_line x1 y1 ∧ 
    on_line x2 y2 ∧ 
    ((x1 = a ∧ y1 = 0) ∨ (x2 = a ∧ y2 = 0)) ∧ 
    ((x1 = 0 ∧ y1 = b) ∨ (x2 = 0 ∧ y2 = b))) →
  (∀ x y : ℝ, is_ellipse x y a b ↔ x^2 / 16 + y^2 / 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l484_48435


namespace NUMINAMATH_CALUDE_square_difference_thirteen_twelve_l484_48473

theorem square_difference_thirteen_twelve : (13 + 12)^2 - (13 - 12)^2 = 624 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_thirteen_twelve_l484_48473


namespace NUMINAMATH_CALUDE_road_sign_ratio_l484_48495

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ

/-- The conditions of the road sign problem -/
def road_sign_conditions (r : RoadSigns) : Prop :=
  r.s₁ = 40 ∧
  r.s₂ > r.s₁ ∧
  r.s₃ = 2 * r.s₂ ∧
  r.s₄ = r.s₃ - 20 ∧
  r.s₁ + r.s₂ + r.s₃ + r.s₄ = 270

/-- The theorem stating that under the given conditions, 
    the ratio of road signs at the second intersection to the first is 5:4 -/
theorem road_sign_ratio (r : RoadSigns) : 
  road_sign_conditions r → r.s₂ * 4 = r.s₁ * 5 := by
  sorry

end NUMINAMATH_CALUDE_road_sign_ratio_l484_48495


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l484_48463

/-- A linear function y = mx + b is decreasing on ℝ if and only if m < 0 -/
theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l484_48463


namespace NUMINAMATH_CALUDE_rectangle_count_l484_48476

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Check if a point lies on a line segment between two other points -/
def pointOnSegment (P Q R : Point2D) : Prop := sorry

/-- Check if two line segments are perpendicular -/
def perpendicular (P Q R S : Point2D) : Prop := sorry

/-- Check if a line segment forms a 30° angle with another line segment -/
def angle30Degrees (P Q R S : Point2D) : Prop := sorry

/-- Check if a rectangle satisfies the given conditions -/
def validRectangle (rect : Rectangle) (P1 P2 P3 : Point2D) : Prop :=
  (rect.A = P1 ∨ rect.A = P2 ∨ rect.A = P3) ∧
  (pointOnSegment P1 rect.A rect.B ∨ pointOnSegment P2 rect.A rect.B ∨ pointOnSegment P3 rect.A rect.B) ∧
  (pointOnSegment P1 rect.B rect.C ∨ pointOnSegment P2 rect.B rect.C ∨ pointOnSegment P3 rect.B rect.C) ∧
  (pointOnSegment P1 rect.C rect.D ∨ pointOnSegment P2 rect.C rect.D ∨ pointOnSegment P3 rect.C rect.D) ∧
  (pointOnSegment P1 rect.D rect.A ∨ pointOnSegment P2 rect.D rect.A ∨ pointOnSegment P3 rect.D rect.A) ∧
  perpendicular rect.A rect.B rect.B rect.C ∧
  perpendicular rect.B rect.C rect.C rect.D ∧
  (angle30Degrees rect.A rect.C rect.A rect.B ∨ angle30Degrees rect.A rect.C rect.C rect.D)

theorem rectangle_count (P1 P2 P3 : Point2D) 
  (h_distinct : P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3) :
  ∃! (s : Finset Rectangle), (∀ rect ∈ s, validRectangle rect P1 P2 P3) ∧ s.card = 60 :=
sorry

end NUMINAMATH_CALUDE_rectangle_count_l484_48476


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l484_48479

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 35, prove that x + y = 58 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (n : ℕ), n ≥ 5 ∧ 
    (∀ k : ℕ, k ≤ n → 
      (if k = 1 then 3
       else if k = 2 then 7
       else if k = 3 then 11
       else if k = n - 1 then x
       else if k = n then y
       else if k = n + 1 then 35
       else 0) = 3 + (k - 1) * 4)) →
  x + y = 58 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l484_48479


namespace NUMINAMATH_CALUDE_largest_base_8_three_digit_in_base_10_l484_48491

/-- The largest three-digit number in a given base -/
def largest_three_digit (base : ℕ) : ℕ :=
  (base - 1) * base^2 + (base - 1) * base^1 + (base - 1) * base^0

/-- Theorem: The largest three-digit base-8 number in base-10 is 511 -/
theorem largest_base_8_three_digit_in_base_10 :
  largest_three_digit 8 = 511 := by sorry

end NUMINAMATH_CALUDE_largest_base_8_three_digit_in_base_10_l484_48491


namespace NUMINAMATH_CALUDE_circle_equation_after_translation_l484_48453

/-- Given a circle C with parametric equations x = 2cos(θ) and y = 2 + 2sin(θ),
    and a translation of the origin to (1, 2), prove that the standard equation
    of the circle in the new coordinate system is (x' - 1)² + (y' - 4)² = 4 -/
theorem circle_equation_after_translation (θ : ℝ) (x y x' y' : ℝ) :
  (x = 2 * Real.cos θ) →
  (y = 2 + 2 * Real.sin θ) →
  (x' = x - 1) →
  (y' = y - 2) →
  (x' - 1)^2 + (y' - 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_after_translation_l484_48453


namespace NUMINAMATH_CALUDE_deepak_age_l484_48430

/-- Given that the ratio of Rahul's age to Deepak's age is 5:2,
    and Rahul will be 26 years old after 6 years,
    prove that Deepak's current age is 8 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) 
  (h_ratio : rahul_age * 2 = deepak_age * 5)
  (h_future : rahul_age + 6 = 26) : 
  deepak_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l484_48430


namespace NUMINAMATH_CALUDE_leading_coefficient_of_specific_polynomial_l484_48455

/-- A polynomial function from ℝ to ℝ -/
noncomputable def PolynomialFunction := ℝ → ℝ

/-- The leading coefficient of a polynomial function -/
noncomputable def leadingCoefficient (g : PolynomialFunction) : ℝ := sorry

theorem leading_coefficient_of_specific_polynomial 
  (g : PolynomialFunction)
  (h : ∀ x : ℝ, g (x + 1) - g x = 8 * x + 6) :
  leadingCoefficient g = 4 := by sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_specific_polynomial_l484_48455


namespace NUMINAMATH_CALUDE_sqrt_16_minus_pi_minus_3_pow_0_l484_48443

theorem sqrt_16_minus_pi_minus_3_pow_0 : Real.sqrt 16 - (π - 3)^0 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_16_minus_pi_minus_3_pow_0_l484_48443


namespace NUMINAMATH_CALUDE_profit_percentage_approx_l484_48458

/-- Calculates the profit percentage for a given purchase and sale scenario. -/
def profit_percentage (items_bought : ℕ) (price_paid : ℕ) (discount : ℚ) : ℚ :=
  let cost := price_paid
  let selling_price := (items_bought : ℚ) * (1 - discount)
  let profit := selling_price - (cost : ℚ)
  (profit / cost) * 100

/-- Theorem stating that the profit percentage for the given scenario is approximately 11.91%. -/
theorem profit_percentage_approx (ε : ℚ) (h_ε : ε > 0) :
  ∃ δ : ℚ, abs (profit_percentage 52 46 (1/100) - 911/7650) < ε :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_approx_l484_48458


namespace NUMINAMATH_CALUDE_min_value_theorem_l484_48411

theorem min_value_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, |x - a| + |x + b| ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l484_48411


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l484_48415

/-- Given three consecutive odd integers whose sum is 75 and whose largest and smallest differ by 6, the largest is 27 -/
theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ k : ℤ, c = 2*k + 1) →  -- c is odd
  b = a + 2 →              -- b is the next consecutive odd after a
  c = b + 2 →              -- c is the next consecutive odd after b
  a + b + c = 75 →         -- sum is 75
  c - a = 6 →              -- difference between largest and smallest is 6
  c = 27 :=                -- largest number is 27
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odds_l484_48415


namespace NUMINAMATH_CALUDE_faster_river_longer_time_l484_48432

/-- Proves that the total travel time in a faster river is greater than in a slower river -/
theorem faster_river_longer_time
  (v : ℝ) (v₁ v₂ S : ℝ) 
  (h_v : v > 0) 
  (h_v₁ : v₁ > 0) 
  (h_v₂ : v₂ > 0) 
  (h_S : S > 0)
  (h_v₁_gt_v₂ : v₁ > v₂) 
  (h_v_gt_v₁ : v > v₁) 
  (h_v_gt_v₂ : v > v₂) :
  (2 * S * v) / (v^2 - v₁^2) > (2 * S * v) / (v^2 - v₂^2) :=
by sorry

end NUMINAMATH_CALUDE_faster_river_longer_time_l484_48432


namespace NUMINAMATH_CALUDE_scalene_triangle_area_l484_48418

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

end NUMINAMATH_CALUDE_scalene_triangle_area_l484_48418


namespace NUMINAMATH_CALUDE_sum_p_q_r_l484_48483

/-- The largest real solution to the given equation -/
noncomputable def n : ℝ := 
  Real.sqrt (53 + Real.sqrt 249) + 13

/-- The equation that n satisfies -/
axiom n_eq : (4 / (n - 4)) + (6 / (n - 6)) + (18 / (n - 18)) + (20 / (n - 20)) = n^2 - 13*n - 6

/-- The existence of positive integers p, q, and r -/
axiom exists_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r)

/-- The theorem to be proved -/
theorem sum_p_q_r : ∃ (p q r : ℕ+), n = p + Real.sqrt (q + Real.sqrt r) ∧ p + q + r = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_p_q_r_l484_48483


namespace NUMINAMATH_CALUDE_tigers_home_games_l484_48472

/-- The number of home games played by the Tigers -/
def total_home_games (losses ties wins : ℕ) : ℕ := losses + ties + wins

/-- The number of losses in Tiger's home games -/
def losses : ℕ := 12

/-- The number of ties in Tiger's home games -/
def ties : ℕ := losses / 2

/-- The number of wins in Tiger's home games -/
def wins : ℕ := 38

theorem tigers_home_games : total_home_games losses ties wins = 56 := by
  sorry

end NUMINAMATH_CALUDE_tigers_home_games_l484_48472


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l484_48427

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_diff : a 3 - 3 * a 2 = 2)
  (h_mean : 5 * a 4 = (12 * a 3 + 2 * a 5) / 2) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l484_48427


namespace NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l484_48424

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside the given ranges

theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l484_48424


namespace NUMINAMATH_CALUDE_tangent_difference_l484_48446

theorem tangent_difference (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - 1 / Real.tan (θ + π/8) = -14 := by
  sorry

end NUMINAMATH_CALUDE_tangent_difference_l484_48446


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l484_48492

theorem sin_sum_of_complex_exponentials
  (γ δ : ℝ)
  (h1 : Complex.exp (γ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (δ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = -(63 / 65) :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l484_48492


namespace NUMINAMATH_CALUDE_flagpole_break_height_l484_48426

theorem flagpole_break_height (h : ℝ) (d : ℝ) (x : ℝ) :
  h = 10 ∧ d = 2 ∧ x * x + d * d = (h - x) * (h - x) →
  x = 2 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l484_48426


namespace NUMINAMATH_CALUDE_f_property_l484_48494

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- This case should never occur in our problem

-- State the theorem
theorem f_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l484_48494


namespace NUMINAMATH_CALUDE_union_of_sets_l484_48464

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l484_48464


namespace NUMINAMATH_CALUDE_numeralia_license_plate_probability_l484_48408

/-- Represents the set of possible symbols for each position in a Numeralia license plate -/
structure NumeraliaLicensePlate :=
  (vowels : Finset Char)
  (nonVowels : Finset Char)
  (digits : Finset Char)

/-- The probability of a specific valid license plate configuration in Numeralia -/
def licensePlateProbability (plate : NumeraliaLicensePlate) : ℚ :=
  1 / ((plate.vowels.card : ℚ) * (plate.nonVowels.card : ℚ) * ((plate.nonVowels.card - 1) : ℚ) * 
       (plate.digits.card + plate.vowels.card : ℚ))

/-- Theorem stating the probability of a specific valid license plate configuration in Numeralia -/
theorem numeralia_license_plate_probability 
  (plate : NumeraliaLicensePlate)
  (h1 : plate.vowels.card = 5)
  (h2 : plate.nonVowels.card = 21)
  (h3 : plate.digits.card = 10) :
  licensePlateProbability plate = 1 / 31500 := by
  sorry


end NUMINAMATH_CALUDE_numeralia_license_plate_probability_l484_48408


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l484_48414

open Real

theorem trigonometric_inequality : 
  let a := sin (3 * π / 5)
  let b := cos (2 * π / 5)
  let c := tan (2 * π / 5)
  b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l484_48414


namespace NUMINAMATH_CALUDE_max_working_groups_is_18_more_than_18_groups_impossible_l484_48407

/-- Represents a working group formation problem -/
structure WorkingGroupProblem where
  totalTeachers : ℕ
  groupSize : ℕ
  maxGroupsPerTeacher : ℕ

/-- Calculates the maximum number of working groups that can be formed -/
def maxWorkingGroups (problem : WorkingGroupProblem) : ℕ :=
  min
    (problem.totalTeachers * problem.maxGroupsPerTeacher / problem.groupSize)
    ((problem.totalTeachers * problem.maxGroupsPerTeacher) / problem.groupSize)

/-- The specific problem instance -/
def specificProblem : WorkingGroupProblem :=
  { totalTeachers := 36
    groupSize := 4
    maxGroupsPerTeacher := 2 }

/-- Theorem stating that the maximum number of working groups is 18 -/
theorem max_working_groups_is_18 :
  maxWorkingGroups specificProblem = 18 := by
  sorry

/-- Theorem proving that more than 18 groups is impossible -/
theorem more_than_18_groups_impossible (n : ℕ) :
  n > 18 → n * specificProblem.groupSize > specificProblem.totalTeachers * specificProblem.maxGroupsPerTeacher := by
  sorry

end NUMINAMATH_CALUDE_max_working_groups_is_18_more_than_18_groups_impossible_l484_48407


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l484_48417

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ := sorry

/-- Theorem stating the area of the region -/
theorem area_between_circles_and_x_axis :
  let c1 : Circle := { center := (3, 5), radius := 5 }
  let c2 : Circle := { center := (15, 5), radius := 3 }
  areaRegion c1 c2 = 60 - 17 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l484_48417


namespace NUMINAMATH_CALUDE_unique_solution_equation_l484_48475

theorem unique_solution_equation : ∃! (x : ℕ), x > 0 ∧ (x - x) + x * x + x / x = 50 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l484_48475


namespace NUMINAMATH_CALUDE_inequality_solution_set_l484_48402

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, (x > 4 ∧ x < b) ↔ (Real.sqrt x > a * x + 3/2)) →
  (a = 1/8 ∧ b = 36) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l484_48402


namespace NUMINAMATH_CALUDE_lemon_juice_fraction_l484_48496

theorem lemon_juice_fraction (total_members : ℕ) (orange_juice_orders : ℕ) : 
  total_members = 30 →
  orange_juice_orders = 6 →
  ∃ (lemon_fraction : ℚ),
    lemon_fraction = 7 / 10 ∧
    lemon_fraction * total_members +
    (1 / 3) * (total_members - lemon_fraction * total_members) +
    orange_juice_orders = total_members :=
by sorry

end NUMINAMATH_CALUDE_lemon_juice_fraction_l484_48496


namespace NUMINAMATH_CALUDE_system_solution_l484_48484

theorem system_solution (x y a : ℝ) : 
  3 * x + y = a → 
  2 * x + 5 * y = 2 * a → 
  x = 3 → 
  a = 13 := by sorry

end NUMINAMATH_CALUDE_system_solution_l484_48484


namespace NUMINAMATH_CALUDE_variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l484_48428

-- Define a type for data sets
def DataSet := List ℝ

-- Define measures
def mean (data : DataSet) : ℝ := sorry
def variance (data : DataSet) : ℝ := sorry
def median (data : DataSet) : ℝ := sorry
def mode (data : DataSet) : ℝ := sorry

-- Define a predicate for measures of dispersion
def isDispersionMeasure (measure : DataSet → ℝ) : Prop := sorry

-- Theorem stating that variance is a measure of dispersion
theorem variance_is_dispersion_measure : isDispersionMeasure variance := sorry

-- Theorems stating that mean, median, and mode are not measures of dispersion
theorem mean_is_not_dispersion_measure : ¬ isDispersionMeasure mean := sorry
theorem median_is_not_dispersion_measure : ¬ isDispersionMeasure median := sorry
theorem mode_is_not_dispersion_measure : ¬ isDispersionMeasure mode := sorry

end NUMINAMATH_CALUDE_variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l484_48428


namespace NUMINAMATH_CALUDE_cos_A_in_special_triangle_l484_48468

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if 2S = a² - (b-c)² where S is the area of the triangle, then cos A = 3/5.
-/
theorem cos_A_in_special_triangle (a b c : ℝ) (A : Real) :
  0 < A → A < Real.pi / 2 →  -- A is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  2 * (1/2 * b * c * Real.sin A) = a^2 - (b - c)^2 →  -- area condition
  Real.cos A = 3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_A_in_special_triangle_l484_48468


namespace NUMINAMATH_CALUDE_initial_participants_count_l484_48413

/-- The number of participants in the social event -/
def n : ℕ := 15

/-- The number of people who left early -/
def early_leavers : ℕ := 4

/-- The number of handshakes each early leaver performed -/
def handshakes_per_leaver : ℕ := 2

/-- The total number of handshakes that occurred -/
def total_handshakes : ℕ := 60

/-- Theorem stating that n is the correct number of initial participants -/
theorem initial_participants_count :
  Nat.choose n 2 - (early_leavers * handshakes_per_leaver - Nat.choose early_leavers 2) = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_initial_participants_count_l484_48413


namespace NUMINAMATH_CALUDE_cat_food_weight_l484_48423

/-- Given the conditions of Mrs. Anderson's pet food purchase, prove that each bag of cat food weighs 3 pounds. -/
theorem cat_food_weight (cat_bags dog_bags : ℕ) (dog_extra_weight : ℕ) (ounces_per_pound : ℕ) (total_ounces : ℕ) :
  cat_bags = 2 ∧ 
  dog_bags = 2 ∧ 
  dog_extra_weight = 2 ∧
  ounces_per_pound = 16 ∧
  total_ounces = 256 →
  ∃ (cat_weight : ℕ), 
    cat_weight = 3 ∧
    total_ounces = ounces_per_pound * (cat_bags * cat_weight + dog_bags * (cat_weight + dog_extra_weight)) :=
by sorry

end NUMINAMATH_CALUDE_cat_food_weight_l484_48423


namespace NUMINAMATH_CALUDE_total_reading_materials_l484_48480

def magazines : ℕ := 425
def newspapers : ℕ := 275

theorem total_reading_materials : magazines + newspapers = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_reading_materials_l484_48480


namespace NUMINAMATH_CALUDE_part1_correct_part2_correct_part3_correct_l484_48431

/-- Represents a coupon type with its discount amount -/
structure CouponType where
  discount : ℕ

/-- The available coupon types -/
def couponTypes : Fin 3 → CouponType
  | 0 => ⟨100⟩  -- A Type
  | 1 => ⟨68⟩   -- B Type
  | 2 => ⟨20⟩   -- C Type

/-- Calculate the total discount from using multiple coupons -/
def totalDiscount (coupons : Fin 3 → ℕ) : ℕ :=
  (coupons 0) * (couponTypes 0).discount +
  (coupons 1) * (couponTypes 1).discount +
  (coupons 2) * (couponTypes 2).discount

/-- Theorem for part 1 -/
theorem part1_correct :
  totalDiscount ![1, 5, 4] = 520 := by sorry

/-- Theorem for part 2 -/
theorem part2_correct :
  totalDiscount ![2, 3, 0] = 404 := by sorry

/-- Helper function to check if a combination is valid -/
def isValidCombination (a b c : ℕ) : Prop :=
  a ≤ 16 ∧ b ≤ 16 ∧ c ≤ 16 ∧
  ((a > 0 ∧ b > 0 ∧ c = 0) ∨
   (a > 0 ∧ b = 0 ∧ c > 0) ∨
   (a = 0 ∧ b > 0 ∧ c > 0))

/-- Theorem for part 3 -/
theorem part3_correct :
  (∀ a b c : ℕ,
    isValidCombination a b c ∧ totalDiscount ![a, b, c] = 708 →
    (a = 3 ∧ b = 6 ∧ c = 0) ∨ (a = 0 ∧ b = 6 ∧ c = 15)) ∧
  isValidCombination 3 6 0 ∧
  totalDiscount ![3, 6, 0] = 708 ∧
  isValidCombination 0 6 15 ∧
  totalDiscount ![0, 6, 15] = 708 := by sorry

end NUMINAMATH_CALUDE_part1_correct_part2_correct_part3_correct_l484_48431


namespace NUMINAMATH_CALUDE_valid_solutions_characterization_l484_48497

/-- A number is a valid solution if it's a four-digit number,
    divisible by 28, and can be expressed as the sum of squares
    of three consecutive even numbers. -/
def is_valid_solution (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  n % 28 = 0 ∧
  ∃ k : ℕ, n = 12 * k^2 + 8

/-- The set of all valid solutions -/
def solution_set : Set ℕ := {1736, 3080, 4340, 6356, 8120}

/-- Theorem stating that the solution_set contains exactly
    the numbers satisfying is_valid_solution -/
theorem valid_solutions_characterization :
  ∀ n : ℕ, is_valid_solution n ↔ n ∈ solution_set :=
by sorry

#check valid_solutions_characterization

end NUMINAMATH_CALUDE_valid_solutions_characterization_l484_48497


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l484_48481

theorem angle_in_third_quadrant (α : Real) : 
  (Real.sin α * Real.tan α < 0) → 
  (Real.cos α / Real.tan α < 0) → 
  (α > Real.pi ∧ α < 3 * Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l484_48481


namespace NUMINAMATH_CALUDE_complement_of_M_l484_48477

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 < 2*x}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l484_48477


namespace NUMINAMATH_CALUDE_alice_weekly_distance_l484_48488

/-- Represents the walking distances for a single day --/
structure DailyWalk where
  morning : ℕ
  evening : ℕ

/-- Alice's walking schedule for the week --/
def aliceSchedule : List DailyWalk := [
  ⟨21, 0⟩,  -- Monday
  ⟨14, 0⟩,  -- Tuesday
  ⟨22, 0⟩,  -- Wednesday
  ⟨19, 0⟩,  -- Thursday
  ⟨20, 0⟩   -- Friday
]

/-- Calculates the total walking distance for a day --/
def totalDailyDistance (day : DailyWalk) : ℕ :=
  day.morning + day.evening

/-- Calculates the total walking distance for the week --/
def totalWeeklyDistance (schedule : List DailyWalk) : ℕ :=
  schedule.map totalDailyDistance |>.sum

/-- Theorem: Alice's total walking distance for the week is 96 miles --/
theorem alice_weekly_distance :
  totalWeeklyDistance aliceSchedule = 96 := by
  sorry

end NUMINAMATH_CALUDE_alice_weekly_distance_l484_48488


namespace NUMINAMATH_CALUDE_set_equality_proof_l484_48425

def U : Set Nat := {0, 1, 2}

theorem set_equality_proof (A : Set Nat) (h : (U \ A) = {2}) : A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_proof_l484_48425


namespace NUMINAMATH_CALUDE_sector_radius_for_cone_l484_48420

/-- 
Given a sector with central angle 120° used to form a cone with base radius 2 cm,
prove that the radius of the sector is 6 cm.
-/
theorem sector_radius_for_cone (θ : Real) (r_base : Real) (r_sector : Real) : 
  θ = 120 → r_base = 2 → (θ / 360) * (2 * Real.pi * r_sector) = 2 * Real.pi * r_base → r_sector = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_for_cone_l484_48420


namespace NUMINAMATH_CALUDE_total_water_filled_jars_l484_48409

/-- Represents the number of jars of each size -/
def num_jars_per_size : ℚ := 20

/-- Represents the total number of jar sizes -/
def num_jar_sizes : ℕ := 3

/-- Represents the total volume of water in gallons -/
def total_water : ℚ := 35

/-- Theorem stating the total number of water-filled jars -/
theorem total_water_filled_jars : 
  (1/4 + 1/2 + 1) * num_jars_per_size = total_water ∧ 
  num_jars_per_size * num_jar_sizes = 60 := by
  sorry

#check total_water_filled_jars

end NUMINAMATH_CALUDE_total_water_filled_jars_l484_48409


namespace NUMINAMATH_CALUDE_problem_solution_l484_48421

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 18 → 4 * (x - 3) = -188 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l484_48421


namespace NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l484_48489

theorem walk_a_thon_miles_difference 
  (last_year_rate : ℝ) 
  (this_year_rate : ℝ) 
  (last_year_total : ℝ) 
  (h1 : last_year_rate = 4)
  (h2 : this_year_rate = 2.75)
  (h3 : last_year_total = 44) :
  (last_year_total / this_year_rate) - (last_year_total / last_year_rate) = 5 :=
by sorry

end NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l484_48489


namespace NUMINAMATH_CALUDE_largest_reciprocal_l484_48462

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/7 → b = 1/2 → c = 3/4 → d = 4 → e = 100 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l484_48462


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l484_48465

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l484_48465


namespace NUMINAMATH_CALUDE_three_number_set_range_l484_48459

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered set
  (a + b + c) / 3 = 6 ∧  -- Mean is 6
  b = 6 ∧  -- Median is 6
  a = 2  -- Smallest number is 2
  →
  c - a = 8 :=  -- Range is 8
by sorry

end NUMINAMATH_CALUDE_three_number_set_range_l484_48459


namespace NUMINAMATH_CALUDE_small_kite_area_l484_48456

/-- A kite is defined by its four vertices -/
structure Kite where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The specific kite from the problem -/
def smallKite : Kite :=
  { v1 := (3, 0)
    v2 := (0, 5)
    v3 := (3, 7)
    v4 := (6, 5) }

/-- Theorem stating that the area of the small kite is 21 square inches -/
theorem small_kite_area : kiteArea smallKite = 21 := by sorry

end NUMINAMATH_CALUDE_small_kite_area_l484_48456


namespace NUMINAMATH_CALUDE_prime_pair_unique_solution_l484_48434

theorem prime_pair_unique_solution (p q : ℕ) : 
  Prime p → Prime q → p > q →
  (∃ k : ℤ, ((p + q : ℤ)^(p + q) * (p - q : ℤ)^(p - q) - 1) / 
            ((p + q : ℤ)^(p - q) * (p - q : ℤ)^(p + q) - 1) = k) →
  p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_pair_unique_solution_l484_48434


namespace NUMINAMATH_CALUDE_sphere_water_volume_calculation_l484_48452

/-- The volume of water in a sphere container that can be transferred to 
    a given number of hemisphere containers of a specific volume. -/
def sphere_water_volume (num_hemispheres : ℕ) (hemisphere_volume : ℝ) : ℝ :=
  (num_hemispheres : ℝ) * hemisphere_volume

/-- Theorem stating the volume of water in a sphere container
    given the number of hemisphere containers and their volume. -/
theorem sphere_water_volume_calculation :
  sphere_water_volume 2744 4 = 10976 := by
  sorry

end NUMINAMATH_CALUDE_sphere_water_volume_calculation_l484_48452


namespace NUMINAMATH_CALUDE_average_problem_l484_48445

theorem average_problem (x : ℝ) : 
  (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l484_48445


namespace NUMINAMATH_CALUDE_medium_sized_fir_trees_l484_48490

theorem medium_sized_fir_trees (total : ℕ) (oaks : ℕ) (saplings : ℕ) 
  (h1 : total = 96) 
  (h2 : oaks = 15) 
  (h3 : saplings = 58) : 
  total - oaks - saplings = 23 := by
  sorry

end NUMINAMATH_CALUDE_medium_sized_fir_trees_l484_48490


namespace NUMINAMATH_CALUDE_sorting_problem_l484_48482

/-- The number of parcels sorted by a machine per hour relative to a person -/
def machine_efficiency : ℕ := 20

/-- The number of machines used in the comparison -/
def num_machines : ℕ := 5

/-- The number of people used in the comparison -/
def num_people : ℕ := 20

/-- The number of parcels sorted in the comparison -/
def parcels_sorted : ℕ := 6000

/-- The time difference in hours between machines and people sorting -/
def time_difference : ℕ := 4

/-- The number of hours machines work per day -/
def machine_work_hours : ℕ := 16

/-- The number of parcels that need to be sorted per day -/
def daily_parcels : ℕ := 100000

/-- The number of parcels sorted manually by one person per hour -/
def parcels_per_person (x : ℕ) : Prop :=
  (parcels_sorted / (num_people * x)) - (parcels_sorted / (num_machines * machine_efficiency * x)) = time_difference

/-- The minimum number of machines needed to sort the daily parcels -/
def machines_needed (y : ℕ) : Prop :=
  y = (daily_parcels + machine_work_hours * machine_efficiency * 60 - 1) / (machine_work_hours * machine_efficiency * 60)

theorem sorting_problem :
  ∃ (x y : ℕ), parcels_per_person x ∧ machines_needed y ∧ x = 60 ∧ y = 6 :=
sorry

end NUMINAMATH_CALUDE_sorting_problem_l484_48482


namespace NUMINAMATH_CALUDE_triangle_properties_l484_48460

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a + t.c)^2 = t.b^2 + 2 * Real.sqrt 3 * t.a * t.c * Real.sin t.C)
  (h2 : t.b = 8)
  (h3 : t.a > t.c)
  (h4 : 1/2 * t.a * t.c * Real.sin t.B = 3 * Real.sqrt 3) :
  t.B = π/3 ∧ t.a = 5 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l484_48460


namespace NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l484_48419

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

end NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l484_48419


namespace NUMINAMATH_CALUDE_quadratic_roots_l484_48406

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 2) * x + 2 * k

theorem quadratic_roots (k : ℝ) :
  (quadratic k 1 = 0 → k = 1 ∧ quadratic k 2 = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l484_48406


namespace NUMINAMATH_CALUDE_circle_properties_l484_48466

/-- Proves properties of a circle with circumference 24 cm -/
theorem circle_properties :
  ∀ (r : ℝ), 2 * π * r = 24 →
  (2 * r = 24 / π ∧ π * r^2 = 144 / π) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l484_48466


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_l484_48493

theorem exponential_function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_l484_48493


namespace NUMINAMATH_CALUDE_real_part_of_z_l484_48447

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l484_48447


namespace NUMINAMATH_CALUDE_starting_number_proof_l484_48436

theorem starting_number_proof (x : Int) : 
  (∃ (l : List Int), l.length = 15 ∧ 
    (∀ n ∈ l, Even n ∧ x ≤ n ∧ n ≤ 40) ∧
    (∀ n, x ≤ n ∧ n ≤ 40 ∧ Even n → n ∈ l)) →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l484_48436


namespace NUMINAMATH_CALUDE_positive_real_inequality_l484_48440

theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2000 + b^2000 = a^1998 + b^1998) : a^2 + b^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l484_48440


namespace NUMINAMATH_CALUDE_dog_food_consumption_l484_48433

theorem dog_food_consumption (num_dogs : ℕ) (total_food : ℝ) (h1 : num_dogs = 2) (h2 : total_food = 0.25) :
  let food_per_dog := total_food / num_dogs
  food_per_dog = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l484_48433


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_22_5_l484_48498

/-- The area of a quadrilateral with vertices at (1, 2), (1, -1), (4, -1), and (7, 8) -/
def quadrilateral_area : ℝ :=
  let A := (1, 2)
  let B := (1, -1)
  let C := (4, -1)
  let D := (7, 8)
  -- Area calculation goes here
  0 -- Placeholder

theorem quadrilateral_area_is_22_5 : quadrilateral_area = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_22_5_l484_48498


namespace NUMINAMATH_CALUDE_sheet_area_difference_l484_48439

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (length1 width1 length2 width2 : ℝ) : ℝ :=
  combinedArea length1 width1 - combinedArea length2 width2

theorem sheet_area_difference :
  areaDifference 11 9 4.5 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l484_48439


namespace NUMINAMATH_CALUDE_triangle_base_length_l484_48486

/-- Given a triangle with area 24.36 and height 5.8, its base length is 8.4 -/
theorem triangle_base_length : 
  ∀ (base : ℝ), 
    (24.36 = (base * 5.8) / 2) → 
    base = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l484_48486


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l484_48450

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + b*x + 2016 = (x+r)*(x+s)) → b ≥ 260) ∧
  (∃ (r s : ℕ), r > 0 ∧ s > 0 ∧ 8 ∣ s ∧ (∀ x : ℤ, x^2 + 260*x + 2016 = (x+r)*(x+s))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l484_48450


namespace NUMINAMATH_CALUDE_abs_sum_cases_l484_48410

theorem abs_sum_cases (x : ℝ) (h : x < 2) :
  (|x - 2| + |2 + x| = 4 ∧ -2 ≤ x) ∨ (|x - 2| + |2 + x| = -2*x ∧ x < -2) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_cases_l484_48410


namespace NUMINAMATH_CALUDE_three_samples_in_interval_l484_48438

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.population / s.sample_size

/-- Counts the number of sampled elements within the given interval -/
def count_sampled_in_interval (s : SystematicSample) : ℕ :=
  let k := sampling_interval s
  let first_sample := k * ((s.interval_start - 1) / k + 1)
  (s.interval_end - first_sample) / k + 1

/-- Theorem stating that for the given systematic sample, 
    exactly 3 sampled numbers fall within the interval [61, 120] -/
theorem three_samples_in_interval : 
  let s : SystematicSample := {
    population := 840
    sample_size := 42
    interval_start := 61
    interval_end := 120
  }
  count_sampled_in_interval s = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_samples_in_interval_l484_48438


namespace NUMINAMATH_CALUDE_least_n_divisibility_l484_48405

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k = 0) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k = 0) ∨
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k ≠ 0)) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisibility_l484_48405


namespace NUMINAMATH_CALUDE_similar_rectangles_width_l484_48467

theorem similar_rectangles_width (area_ratio : ℚ) (small_width : ℚ) (large_width : ℚ) : 
  area_ratio = 1 / 9 →
  small_width = 2 →
  (large_width / small_width) ^ 2 = 1 / area_ratio →
  large_width = 6 := by
sorry

end NUMINAMATH_CALUDE_similar_rectangles_width_l484_48467


namespace NUMINAMATH_CALUDE_binomial_p_value_l484_48449

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (X : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with E(X) = 4 and D(X) = 3, p = 1/4 -/
theorem binomial_p_value (X : BinomialDistribution) 
  (h2 : expectation X = 4) 
  (h3 : variance X = 3) : 
  X.p = 1/4 := by sorry

end NUMINAMATH_CALUDE_binomial_p_value_l484_48449


namespace NUMINAMATH_CALUDE_intersecting_lines_a_value_l484_48412

/-- Three lines intersect at one point if and only if their equations are satisfied simultaneously -/
def intersect_at_one_point (a : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + 2 * y + 8 = 0 ∧ 4 * x + 3 * y = 10 ∧ 2 * x - y = 10

/-- The theorem stating that if the three given lines intersect at one point, then a = -1 -/
theorem intersecting_lines_a_value :
  ∀ a : ℝ, intersect_at_one_point a → a = -1 :=
by
  sorry

#check intersecting_lines_a_value

end NUMINAMATH_CALUDE_intersecting_lines_a_value_l484_48412


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l484_48469

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  B = π / 3 →  -- 60° in radians
  a = Real.sqrt 6 →
  b = 3 →
  A + B + C = π →  -- sum of angles in a triangle
  a / (Real.sin A) = b / (Real.sin B) →  -- law of sines
  A < B →  -- larger side opposite larger angle
  A = π / 4  -- 45° in radians
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l484_48469


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l484_48403

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statement
theorem complement_of_M_in_U : 
  U \ M = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l484_48403


namespace NUMINAMATH_CALUDE_quadratic_function_relationship_l484_48441

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  a : ℝ
  b : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_a : b = -5 * a^2 + p * a + q
  eq_b : y₁ = q
  eq_c : b = -5 * (4 - a)^2 + p * (4 - a) + q
  eq_d : y₂ = -5 + p + q
  eq_e : y₃ = -80 + 4 * p + q

/-- The relationship between y₁, y₂, and y₃ for the given quadratic function -/
theorem quadratic_function_relationship (f : QuadraticFunction) : f.y₃ = f.y₁ ∧ f.y₁ < f.y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_relationship_l484_48441


namespace NUMINAMATH_CALUDE_cookies_left_after_ted_l484_48444

/-- Calculates the number of cookies left after Frank's baking and consumption, and Ted's visit -/
def cookies_left (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
                 (frank_daily_consumption : ℕ) (ted_consumption : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_daily_consumption - ted_consumption

/-- Proves that 134 cookies are left after 6 days of Frank's baking and Ted's visit -/
theorem cookies_left_after_ted : cookies_left 6 2 12 1 4 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_ted_l484_48444


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l484_48471

/-- Given that the binomial coefficient of the 7th term in the expansion of (a+b)^n is the largest, prove that n = 12. -/
theorem largest_binomial_coefficient_seventh_term (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → Nat.choose n k ≤ Nat.choose n 6) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_seventh_term_l484_48471


namespace NUMINAMATH_CALUDE_fibSeriesSum_l484_48429

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of the infinite series of Fibonacci numbers divided by powers of 2 -/
noncomputable def fibSeries : ℝ := ∑' n, (fib n : ℝ) / 2^n

/-- The sum of the infinite series of Fibonacci numbers divided by powers of 2 equals 2 -/
theorem fibSeriesSum : fibSeries = 2 := by sorry

end NUMINAMATH_CALUDE_fibSeriesSum_l484_48429


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l484_48451

/-- The set M represents an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N represents a family of lines parameterized by m and b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- The theorem states that if M intersects N for all m, then b is in the specified range -/
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l484_48451


namespace NUMINAMATH_CALUDE_sampledInInterval_eq_three_l484_48457

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : ℕ
  sampleSize : ℕ
  intervalStart : ℕ
  intervalEnd : ℕ

/-- Calculates the number of sampled individuals within a given interval -/
def sampledInInterval (s : SystematicSampling) : ℕ :=
  let stride := s.totalPopulation / s.sampleSize
  let firstSample := s.intervalStart + (stride - s.intervalStart % stride) % stride
  if firstSample > s.intervalEnd then 0
  else (s.intervalEnd - firstSample) / stride + 1

/-- Theorem stating that for the given systematic sampling scenario, 
    the number of sampled individuals in the interval [61, 120] is 3 -/
theorem sampledInInterval_eq_three :
  let s : SystematicSampling := {
    totalPopulation := 840,
    sampleSize := 42,
    intervalStart := 61,
    intervalEnd := 120
  }
  sampledInInterval s = 3 := by sorry

end NUMINAMATH_CALUDE_sampledInInterval_eq_three_l484_48457
