import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_conditions_l3346_334635

/-- Given a system of equations:
    a x + b y = c z
    a √(1 - x²) + b √(1 - y²) = c √(1 - z²)
    where x, y, z are real variables,
    prove that for a real solution to exist:
    1. a, b, c must satisfy the triangle inequalities
    2. At least one of a or b must have the same sign as c -/
theorem system_solution_conditions (a b c : ℝ) : 
  (∃ x y z : ℝ, a * x + b * y = c * z ∧ 
   a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) →
  (abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b) ∧
  (a * c ≥ 0 ∨ b * c ≥ 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l3346_334635


namespace NUMINAMATH_CALUDE_circle_equation_l3346_334638

theorem circle_equation (x y : ℝ) : 
  (∃ (R : ℝ), (x - 3)^2 + (y - 1)^2 = R^2) ∧ 
  (0 - 3)^2 + (0 - 1)^2 = 10 →
  (x - 3)^2 + (y - 1)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3346_334638


namespace NUMINAMATH_CALUDE_inequality_proof_l3346_334684

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3346_334684


namespace NUMINAMATH_CALUDE_total_holidays_in_year_l3346_334633

def holidays_per_month : ℕ := 2
def months_per_year : ℕ := 12

theorem total_holidays_in_year : 
  holidays_per_month * months_per_year = 24 := by sorry

end NUMINAMATH_CALUDE_total_holidays_in_year_l3346_334633


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3346_334603

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) :
  -3 < a - b ∧ a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3346_334603


namespace NUMINAMATH_CALUDE_probability_red_from_box2_is_11_27_l3346_334654

/-- Represents a box containing balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- The probability of drawing a red ball from box 2 after the described process -/
def probability_red_from_box2 (box1 box2 : Box) : ℚ :=
  let total_balls1 := box1.white + box1.red
  let total_balls2 := box2.white + box2.red + 1
  let prob_white_from_box1 := box1.white / total_balls1
  let prob_red_from_box1 := box1.red / total_balls1
  let prob_red_if_white_moved := prob_white_from_box1 * (box2.red / total_balls2)
  let prob_red_if_red_moved := prob_red_from_box1 * ((box2.red + 1) / total_balls2)
  prob_red_if_white_moved + prob_red_if_red_moved

theorem probability_red_from_box2_is_11_27 :
  let box1 : Box := { white := 2, red := 4 }
  let box2 : Box := { white := 5, red := 3 }
  probability_red_from_box2 box1 box2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_from_box2_is_11_27_l3346_334654


namespace NUMINAMATH_CALUDE_lauren_tuesday_earnings_l3346_334677

/-- Represents Lauren's earnings from her social media channel on Tuesday -/
def LaurenEarnings : ℝ → ℝ → ℕ → ℕ → ℝ :=
  λ commercial_rate subscription_rate commercial_views subscriptions =>
    commercial_rate * (commercial_views : ℝ) + subscription_rate * (subscriptions : ℝ)

/-- Theorem stating Lauren's earnings on Tuesday -/
theorem lauren_tuesday_earnings :
  LaurenEarnings 0.5 1 100 27 = 77 := by
  sorry

end NUMINAMATH_CALUDE_lauren_tuesday_earnings_l3346_334677


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3346_334678

/-- Given a triangle ABC, prove that a hexagon with sides parallel to the triangle's sides
    and equal length d satisfies the equation: d = (abc) / (ab + bc + ca) -/
theorem hexagon_side_length (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  d = (a * b * c) / (a * b + b * c + c * a) ↔
  (1 / d = 1 / a + 1 / b + 1 / c) :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3346_334678


namespace NUMINAMATH_CALUDE_triple_composition_even_l3346_334637

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l3346_334637


namespace NUMINAMATH_CALUDE_truck_speed_l3346_334609

/-- Proves that a truck traveling 600 meters in 20 seconds has a speed of 108 kilometers per hour. -/
theorem truck_speed (distance : ℝ) (time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ) : 
  distance = 600 →
  time = 20 →
  speed_ms = distance / time →
  speed_kmh = speed_ms * 3.6 →
  speed_kmh = 108 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_l3346_334609


namespace NUMINAMATH_CALUDE_min_value_cyclic_sum_l3346_334636

theorem min_value_cyclic_sum (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (k * a / b) + (k * b / c) + (k * c / a) ≥ 3 * k ∧
  ((k * a / b) + (k * b / c) + (k * c / a) = 3 * k ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_cyclic_sum_l3346_334636


namespace NUMINAMATH_CALUDE_green_probability_is_half_l3346_334680

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a specific colored cube is 1/2 -/
theorem green_probability_is_half :
  let cube : ColoredCube := {
    total_faces := 6,
    green_faces := 3,
    yellow_faces := 2,
    red_faces := 1
  }
  green_probability cube = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_green_probability_is_half_l3346_334680


namespace NUMINAMATH_CALUDE_inequality_solution_l3346_334695

noncomputable def solution_set : Set ℝ :=
  { x | x ∈ Set.Ioo (-4) (-14/3) ∪ Set.Ioi (6 + 3 * Real.sqrt 2) }

theorem inequality_solution :
  { x : ℝ | (2*x + 3) / (x + 4) > (5*x + 6) / (3*x + 14) } = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3346_334695


namespace NUMINAMATH_CALUDE_arithmetic_mean_example_l3346_334606

theorem arithmetic_mean_example : 
  let numbers : List ℕ := [12, 24, 36, 48]
  (numbers.sum / numbers.length : ℚ) = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_example_l3346_334606


namespace NUMINAMATH_CALUDE_john_index_cards_purchase_l3346_334628

/-- Calculates the total number of index card packs bought for all students -/
def total_packs_bought (num_classes : ℕ) (students_per_class : ℕ) (packs_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * packs_per_student

/-- Proves that given 6 classes with 30 students each, and 2 packs per student, the total packs bought is 360 -/
theorem john_index_cards_purchase :
  total_packs_bought 6 30 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_john_index_cards_purchase_l3346_334628


namespace NUMINAMATH_CALUDE_smallest_n_for_m_disjoint_monochromatic_edges_l3346_334634

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m pairwise disjoint edges of the same color -/
def HasMDisjointMonochromaticEdges (n m : ℕ) (coloring : TwoColoring n) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i : Fin m, (edges i).1 ≠ (edges i).2) ∧
    (∀ i j : Fin m, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                            (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∃ c : Fin 2, ∀ i : Fin m, coloring (edges i).1 (edges i).2 = c)

/-- The main theorem -/
theorem smallest_n_for_m_disjoint_monochromatic_edges (m : ℕ) (hm : m > 0) :
  (∀ n : ℕ, n ≥ 3 * m - 1 → ∀ coloring : TwoColoring n, HasMDisjointMonochromaticEdges n m coloring) ∧
  (∀ n : ℕ, n < 3 * m - 1 → ∃ coloring : TwoColoring n, ¬HasMDisjointMonochromaticEdges n m coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_m_disjoint_monochromatic_edges_l3346_334634


namespace NUMINAMATH_CALUDE_simplify_fraction_l3346_334615

theorem simplify_fraction : (88 : ℚ) / 7744 = 1 / 88 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3346_334615


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solution_existence_l3346_334645

-- Define the function f
def f (x : ℝ) : ℝ := 45 * abs (2 * x + 3) + abs (2 * x - 1)

-- Theorem for part I
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} :=
sorry

-- Theorem for part II
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3 * m + 1|} = {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solution_existence_l3346_334645


namespace NUMINAMATH_CALUDE_non_red_cubes_count_total_small_cubes_correct_l3346_334640

/-- Represents the number of small cubes without red faces in a 6x6x6 cube with three faces painted red -/
def non_red_cubes : Set ℕ :=
  {n : ℕ | n = 120 ∨ n = 125}

/-- The main theorem stating that the number of non-red cubes is either 120 or 125 -/
theorem non_red_cubes_count :
  ∀ n : ℕ, n ∈ non_red_cubes ↔ (n = 120 ∨ n = 125) :=
by
  sorry

/-- The cube is 6x6x6 -/
def cube_size : ℕ := 6

/-- The number of small cubes the large cube is cut into -/
def total_small_cubes : ℕ := 216

/-- The number of faces painted red -/
def painted_faces : ℕ := 3

/-- The size of each small cube -/
def small_cube_size : ℕ := 1

/-- Theorem stating that the total number of small cubes is correct -/
theorem total_small_cubes_correct :
  cube_size ^ 3 = total_small_cubes :=
by
  sorry

end NUMINAMATH_CALUDE_non_red_cubes_count_total_small_cubes_correct_l3346_334640


namespace NUMINAMATH_CALUDE_no_valid_base_6_digit_for_divisibility_by_7_l3346_334620

theorem no_valid_base_6_digit_for_divisibility_by_7 :
  ∀ d : ℕ, d ≤ 5 → ¬(∃ k : ℤ, 652 + 42 * d = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_base_6_digit_for_divisibility_by_7_l3346_334620


namespace NUMINAMATH_CALUDE_odd_function_properties_l3346_334631

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →  -- f(-√2) = √2
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →  -- f(2√2) = 10√2
  (∃ (f' : ℝ → ℝ),
    (∀ x, f a b c x = x^3 - 3*x) ∧  -- f(x) = x³ - 3x
    (∀ x, x < -1 → f' x > 0) ∧  -- f is increasing on (-∞, -1)
    (∀ x, -1 < x ∧ x < 1 → f' x < 0) ∧  -- f is decreasing on (-1, 1)
    (∀ x, 1 < x → f' x > 0) ∧  -- f is increasing on (1, +∞)
    (∀ m, (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
           f a b c x + m = 0 ∧ f a b c y + m = 0 ∧ f a b c z + m = 0) ↔
          -2 < m ∧ m < 2)) -- f(x) + m = 0 has three distinct roots iff m ∈ (-2, 2)
  := by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l3346_334631


namespace NUMINAMATH_CALUDE_mutual_fund_share_price_increase_l3346_334602

theorem mutual_fund_share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * 1.25
  let second_quarter_price := initial_price * 1.55
  (second_quarter_price - first_quarter_price) / first_quarter_price * 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mutual_fund_share_price_increase_l3346_334602


namespace NUMINAMATH_CALUDE_age_puzzle_l3346_334689

theorem age_puzzle (A : ℕ) (h : A = 32) : ∃ N : ℚ, N * (A + 4) - 4 * (A - 4) = A ∧ N = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3346_334689


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3346_334657

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem parallel_line_through_point (A B C : Point) : 
  let AC : Line := { a := C.y - A.y, b := A.x - C.x, c := A.y * C.x - A.x * C.y }
  let L : Line := { a := 1, b := -2, c := -7 }
  A.x = 5 ∧ A.y = 2 ∧ B.x = -1 ∧ B.y = -4 ∧ C.x = -5 ∧ C.y = -3 →
  B.liesOn L ∧ L.isParallelTo AC := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l3346_334657


namespace NUMINAMATH_CALUDE_polygon_side_length_theorem_l3346_334679

/-- A convex polygon that can be divided into unit equilateral triangles and unit squares -/
structure ConvexPolygon where
  sides : List ℕ
  is_convex : Bool

/-- The number of ways to divide a ConvexPolygon into unit equilateral triangles and unit squares -/
def divisionWays (M : ConvexPolygon) : ℕ := sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) (h_prime : Nat.Prime p) :
  divisionWays M = p → ∃ (side : ℕ), side ∈ M.sides ∧ side = p - 1 := by sorry

end NUMINAMATH_CALUDE_polygon_side_length_theorem_l3346_334679


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3346_334691

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3346_334691


namespace NUMINAMATH_CALUDE_laundry_day_lcm_l3346_334661

theorem laundry_day_lcm : Nat.lcm 6 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_laundry_day_lcm_l3346_334661


namespace NUMINAMATH_CALUDE_zero_product_theorem_l3346_334601

theorem zero_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (sum_condition : x₁ + x₂ + x₃ + x₄ = 0)
  (power_sum_condition : x₁^7 + x₂^7 + x₃^7 + x₄^7 = 0) :
  x₄ * (x₄ + x₁) * (x₄ + x₂) * (x₄ + x₃) = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_product_theorem_l3346_334601


namespace NUMINAMATH_CALUDE_min_value_theorem_l3346_334607

theorem min_value_theorem (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  (x / (x^2 + 1)) + (y / (y^2 + 1)) + (z / (z^2 + 1)) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3346_334607


namespace NUMINAMATH_CALUDE_circles_common_point_l3346_334632

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in the plane -/
def Point := ℝ × ℝ

/-- Returns true if the given point lies on the given circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Returns true if two circles intersect -/
def circles_intersect (c1 c2 : Circle) : Prop :=
  ∃ p : Point, point_on_circle p c1 ∧ point_on_circle p c2

/-- Returns the orthocenter of a triangle given its vertices -/
noncomputable def orthocenter (p1 p2 p3 : Point) : Point :=
  sorry

/-- The main theorem -/
theorem circles_common_point (k1 k2 k3 k : Circle) : 
  (point_on_circle k1.center k) → 
  (point_on_circle k2.center k) → 
  (point_on_circle k3.center k) → 
  (∃ p12 : Point, point_on_circle p12 k1 ∧ point_on_circle p12 k2 ∧ point_on_circle p12 k) →
  (∃ p23 : Point, point_on_circle p23 k2 ∧ point_on_circle p23 k3 ∧ point_on_circle p23 k) →
  (∃ p31 : Point, point_on_circle p31 k3 ∧ point_on_circle p31 k1 ∧ point_on_circle p31 k) →
  (∃ p : Point, point_on_circle p k1 ∧ point_on_circle p k2 ∧ point_on_circle p k3) ↔
  (point_on_circle (orthocenter k1.center k2.center k3.center) k1 ∧ 
   point_on_circle (orthocenter k1.center k2.center k3.center) k2 ∧ 
   point_on_circle (orthocenter k1.center k2.center k3.center) k3) ∨
  (∃ p : Point, point_on_circle p k ∧ point_on_circle p k1 ∧ point_on_circle p k2 ∧ point_on_circle p k3) :=
by sorry

end NUMINAMATH_CALUDE_circles_common_point_l3346_334632


namespace NUMINAMATH_CALUDE_sneaker_coupon_value_l3346_334671

/-- Proves that the coupon value is $10 given the conditions of the sneaker purchase problem -/
theorem sneaker_coupon_value (original_price : ℝ) (membership_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 120)
  (h2 : membership_discount = 0.1)
  (h3 : final_price = 99) :
  ∃ (coupon_value : ℝ), 
    (1 - membership_discount) * (original_price - coupon_value) = final_price ∧
    coupon_value = 10 :=
by sorry

end NUMINAMATH_CALUDE_sneaker_coupon_value_l3346_334671


namespace NUMINAMATH_CALUDE_candy_store_revenue_l3346_334652

/-- Represents the revenue calculation for a candy store sale --/
theorem candy_store_revenue : 
  let fudge_pounds : ℝ := 37
  let fudge_price : ℝ := 2.5
  let truffle_count : ℝ := 82
  let truffle_price : ℝ := 1.5
  let pretzel_count : ℝ := 48
  let pretzel_price : ℝ := 2
  let fudge_discount : ℝ := 0.1
  let sales_tax : ℝ := 0.05

  let fudge_revenue := fudge_pounds * fudge_price
  let truffle_revenue := truffle_count * truffle_price
  let pretzel_revenue := pretzel_count * pretzel_price

  let total_before_discount := fudge_revenue + truffle_revenue + pretzel_revenue
  let fudge_discount_amount := fudge_revenue * fudge_discount
  let total_after_discount := total_before_discount - fudge_discount_amount
  let tax_amount := total_after_discount * sales_tax
  let final_revenue := total_after_discount + tax_amount

  final_revenue = 317.36
  := by sorry


end NUMINAMATH_CALUDE_candy_store_revenue_l3346_334652


namespace NUMINAMATH_CALUDE_opposite_of_one_is_negative_one_opposite_property_l3346_334650

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of 1 is -1
theorem opposite_of_one_is_negative_one :
  opposite 1 = -1 :=
by
  -- The proof would go here
  sorry

-- Axiom that defines the property of opposite
axiom opposite_sum_zero (a : ℤ) : a + (opposite a) = 0

-- Theorem to prove that our definition of opposite satisfies the axiom
theorem opposite_property : ∀ a : ℤ, a + (opposite a) = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_is_negative_one_opposite_property_l3346_334650


namespace NUMINAMATH_CALUDE_total_chips_bags_l3346_334653

theorem total_chips_bags (total_bags : ℕ) (doritos_bags : ℕ) : 
  (4 * doritos_bags = total_bags) →  -- One quarter of the bags are Doritos
  (4 * 5 = doritos_bags) →           -- Doritos bags can be split into 4 equal piles with 5 bags in each
  total_bags = 80 :=                 -- Prove that the total number of bags is 80
by
  sorry

end NUMINAMATH_CALUDE_total_chips_bags_l3346_334653


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3346_334622

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) :
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.8 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3346_334622


namespace NUMINAMATH_CALUDE_complex_magnitude_l3346_334660

theorem complex_magnitude (z₁ z₂ : ℂ) 
  (h1 : z₁ + z₂ = Complex.I * z₁) 
  (h2 : z₂^2 = 2 * Complex.I) : 
  Complex.abs z₁ = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3346_334660


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3346_334623

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {2, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3346_334623


namespace NUMINAMATH_CALUDE_impurity_reduction_proof_l3346_334681

/-- Represents the reduction factor of impurities after each filtration -/
def reduction_factor : ℝ := 0.8

/-- Represents the target impurity level as a fraction of the original -/
def target_impurity : ℝ := 0.05

/-- The minimum number of filtrations required to reduce impurities below the target level -/
def min_filtrations : ℕ := 14

theorem impurity_reduction_proof :
  (reduction_factor ^ min_filtrations : ℝ) < target_impurity ∧
  ∀ n : ℕ, n < min_filtrations → (reduction_factor ^ n : ℝ) ≥ target_impurity :=
sorry

end NUMINAMATH_CALUDE_impurity_reduction_proof_l3346_334681


namespace NUMINAMATH_CALUDE_customers_who_tipped_l3346_334672

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) : 
  initial_customers = 39 →
  additional_customers = 12 →
  non_tipping_customers = 49 →
  initial_customers + additional_customers - non_tipping_customers = 2 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l3346_334672


namespace NUMINAMATH_CALUDE_student_take_home_pay_l3346_334673

/-- Calculates the take-home pay for a well-performing student at a fast-food chain --/
def takeHomePay (baseSalary : ℝ) (bonus : ℝ) (taxRate : ℝ) : ℝ :=
  let totalEarnings := baseSalary + bonus
  let taxAmount := totalEarnings * taxRate
  totalEarnings - taxAmount

/-- Theorem: The take-home pay for a well-performing student is 26,100 rubles --/
theorem student_take_home_pay :
  takeHomePay 25000 5000 0.13 = 26100 := by
  sorry

#eval takeHomePay 25000 5000 0.13

end NUMINAMATH_CALUDE_student_take_home_pay_l3346_334673


namespace NUMINAMATH_CALUDE_right_triangle_cone_volume_l3346_334664

theorem right_triangle_cone_volume (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / 3 : ℝ) * π * y^2 * x = 1500 * π ∧
  (1 / 3 : ℝ) * π * x^2 * y = 540 * π →
  Real.sqrt (x^2 + y^2) = 5 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cone_volume_l3346_334664


namespace NUMINAMATH_CALUDE_max_consecutive_digit_sums_l3346_334694

/-- Given a natural number, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Returns true if the given list of natural numbers contains n consecutive numbers. -/
def isConsecutive (l : List ℕ) (n : ℕ) : Prop := sorry

/-- Theorem: 18 is the maximum value of n for which there exists a sequence of n consecutive 
    natural numbers whose digit sums form another sequence of n consecutive numbers. -/
theorem max_consecutive_digit_sums : 
  ∀ n : ℕ, n > 18 → 
  ¬∃ (start : ℕ), 
    let numbers := List.range n |>.map (λ i => start + i)
    let digitSums := numbers.map sumOfDigits
    isConsecutive numbers n ∧ isConsecutive digitSums n :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_digit_sums_l3346_334694


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3346_334613

/-- Given a geometric sequence {a_n} where a_5 = -16 and a_8 = 8, prove that a_11 = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n)^m) →  -- geometric sequence property
  a 5 = -16 →
  a 8 = 8 →
  a 11 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3346_334613


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3346_334627

theorem x_minus_y_value (x y : ℤ) (hx : x = -3) (hy : |y| = 4) :
  x - y = 1 ∨ x - y = -7 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3346_334627


namespace NUMINAMATH_CALUDE_fruit_arrangement_theorem_l3346_334690

def number_of_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) (group3 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2 * Nat.factorial group3)

theorem fruit_arrangement_theorem :
  number_of_arrangements 7 4 2 1 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_theorem_l3346_334690


namespace NUMINAMATH_CALUDE_age_ratio_l3346_334687

def kul_age : ℕ := 22
def saras_age : ℕ := 33

theorem age_ratio : 
  (saras_age : ℚ) / (kul_age : ℚ) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_l3346_334687


namespace NUMINAMATH_CALUDE_arithmetic_mean_squares_l3346_334655

theorem arithmetic_mean_squares (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ((((x + a)^2) / x + ((x - a)^2) / x) / 2) = x + a^2 / x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_squares_l3346_334655


namespace NUMINAMATH_CALUDE_unique_solution_abc_l3346_334644

theorem unique_solution_abc : ∃! (a b c : ℝ),
  a > 2 ∧ b > 2 ∧ c > 2 ∧
  ((a + 1)^2) / (b + c - 1) + ((b + 3)^2) / (c + a - 3) + ((c + 5)^2) / (a + b - 5) = 27 ∧
  a = 9 ∧ b = 7 ∧ c = 2 := by
  sorry

#check unique_solution_abc

end NUMINAMATH_CALUDE_unique_solution_abc_l3346_334644


namespace NUMINAMATH_CALUDE_jerry_walking_distance_l3346_334621

theorem jerry_walking_distance (monday_miles tuesday_miles : ℝ) 
  (h1 : monday_miles = tuesday_miles)
  (h2 : monday_miles + tuesday_miles = 18) : 
  monday_miles = 9 :=
by sorry

end NUMINAMATH_CALUDE_jerry_walking_distance_l3346_334621


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3346_334642

theorem absolute_value_equation : 
  {x : ℤ | |(-5 + x)| = 11} = {16, -6} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3346_334642


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3346_334619

theorem fraction_equivalence : 
  ∃ n : ℤ, (2 + n : ℚ) / (7 + n) = 3 / 4 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3346_334619


namespace NUMINAMATH_CALUDE_two_pump_filling_time_l3346_334682

/-- Given two pumps with filling rates of 1/3 tank per hour and 4 tanks per hour respectively,
    the time taken to fill a tank when both pumps work together is 3/13 hours. -/
theorem two_pump_filling_time :
  let small_pump_rate : ℚ := 1/3  -- Rate of small pump in tanks per hour
  let large_pump_rate : ℚ := 4    -- Rate of large pump in tanks per hour
  let combined_rate : ℚ := small_pump_rate + large_pump_rate
  let filling_time : ℚ := 1 / combined_rate
  filling_time = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_two_pump_filling_time_l3346_334682


namespace NUMINAMATH_CALUDE_equilateral_triangle_complex_l3346_334688

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768. -/
theorem equilateral_triangle_complex (a b c : ℂ) :
  (∃ (w : ℂ), w^3 = 1 ∧ w ≠ 1 ∧ b - a = 24 * w ∧ c - a = 24 * w^2) →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_complex_l3346_334688


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l3346_334667

theorem range_of_2a_minus_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) :
  ∀ x, x ∈ Set.Ioo (-4 : ℝ) 2 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = 2*a - b :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l3346_334667


namespace NUMINAMATH_CALUDE_quadratic_roots_l3346_334630

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The theorem stating that the quadratic equation has specific roots -/
theorem quadratic_roots (a b c : ℝ) : 
  f a b c (-2) = 21 ∧ 
  f a b c (-1) = 12 ∧ 
  f a b c 1 = 0 ∧ 
  f a b c 2 = -3 ∧ 
  f a b c 4 = -3 → 
  (∀ x, f a b c x = 0 ↔ x = 1 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3346_334630


namespace NUMINAMATH_CALUDE_octagon_diagonal_length_l3346_334646

/-- The length of a diagonal in a regular octagon inscribed in a circle -/
theorem octagon_diagonal_length (r : ℝ) (h : r = 12) :
  let diagonal_length := Real.sqrt (288 + 144 * Real.sqrt 2)
  ∃ (AC : ℝ), AC = diagonal_length := by sorry

end NUMINAMATH_CALUDE_octagon_diagonal_length_l3346_334646


namespace NUMINAMATH_CALUDE_glass_to_sand_ratio_l3346_334683

/-- Represents the number of items in each container --/
structure BeachTreasures where
  bucket : ℕ  -- number of seashells in the bucket
  jar : ℕ     -- number of glass pieces in the jar
  bag : ℕ     -- number of sand dollars in the bag

/-- The conditions of Simon's beach treasure collection --/
def simons_treasures : BeachTreasures → Prop
  | t => t.bucket = 5 * t.jar ∧ 
         t.jar = t.bag ∧ 
         t.bag = 10 ∧ 
         t.bucket + t.jar + t.bag = 190

/-- The theorem stating the ratio of glass pieces to sand dollars --/
theorem glass_to_sand_ratio (t : BeachTreasures) 
  (h : simons_treasures t) : t.jar / t.bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_glass_to_sand_ratio_l3346_334683


namespace NUMINAMATH_CALUDE_some_number_is_four_l3346_334641

theorem some_number_is_four : ∃ n : ℚ, (27 / n) * 12 - 18 = 3 * 12 + 27 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_some_number_is_four_l3346_334641


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3346_334612

def is_odd (n : ℕ) : Prop := n % 2 = 1

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def valid_number (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000000 ∧ is_odd n ∧ is_odd (digit_sum n) ∧ is_odd (digit_product n)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 39 ∧
  ∀ m, valid_number m → m ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3346_334612


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_l3346_334616

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l2 (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l3 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (x, y) where
  x := -2
  y := 2

-- Define perpendicularity of lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (m b : ℝ), 
    (l1 P.1 P.2) ∧ 
    (l2 P.1 P.2) ∧ 
    (perpendicular m ((1 : ℝ) / 2)) ∧ 
    (∀ (x y : ℝ), m * x + y + b = 0 ↔ 2 * x + y + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_l3346_334616


namespace NUMINAMATH_CALUDE_x_less_than_y_l3346_334692

theorem x_less_than_y : 123456789 * 123456786 < 123456788 * 123456787 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l3346_334692


namespace NUMINAMATH_CALUDE_probability_even_sum_l3346_334697

def wheel1 : List ℕ := [1, 1, 2, 3, 3, 4]
def wheel2 : List ℕ := [2, 4, 5, 5, 6]

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even (l : List ℕ) : ℕ := (l.filter is_even).length

def total_outcomes : ℕ := wheel1.length * wheel2.length

def favorable_outcomes : ℕ := 
  (wheel1.filter is_even).length * (wheel2.filter is_even).length +
  (wheel1.filter (fun x => ¬(is_even x))).length * (wheel2.filter (fun x => ¬(is_even x))).length

theorem probability_even_sum : 
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_even_sum_l3346_334697


namespace NUMINAMATH_CALUDE_train_length_l3346_334662

/-- Given a train that passes a pole in 11 seconds and a 120 m long platform in 22 seconds, 
    its length is 120 meters. -/
theorem train_length (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 11)
    (h2 : platform_time = 22)
    (h3 : platform_length = 120) : ℝ :=
  by sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3346_334662


namespace NUMINAMATH_CALUDE_singer_arrangements_eq_18_l3346_334685

/-- The number of different arrangements for 5 singers with restrictions -/
def singer_arrangements : ℕ :=
  let total_singers : ℕ := 5
  let singers_to_arrange : ℕ := total_singers - 1  -- excluding the last singer
  let first_position_choices : ℕ := singers_to_arrange - 1  -- excluding the singer who can't be first
  first_position_choices * Nat.factorial (singers_to_arrange - 1)

theorem singer_arrangements_eq_18 : singer_arrangements = 18 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangements_eq_18_l3346_334685


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l3346_334696

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 28 →
  leaving_age = 22 →
  remaining_people = 5 →
  (initial_people * initial_average - leaving_age) / remaining_people = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l3346_334696


namespace NUMINAMATH_CALUDE_cube_sum_equals_407_l3346_334605

theorem cube_sum_equals_407 (x y : ℝ) (h1 : x + y = 11) (h2 : x^2 * y = 36) :
  x^3 + y^3 = 407 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_407_l3346_334605


namespace NUMINAMATH_CALUDE_base_eight_31_equals_25_l3346_334670

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 31 is equal to the base-ten number 25 -/
theorem base_eight_31_equals_25 : base_eight_to_ten 3 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_31_equals_25_l3346_334670


namespace NUMINAMATH_CALUDE_toys_per_day_l3346_334626

-- Define the weekly toy production
def weekly_production : ℕ := 4340

-- Define the number of working days per week
def working_days : ℕ := 2

-- Define the daily toy production
def daily_production : ℕ := weekly_production / working_days

-- Theorem to prove
theorem toys_per_day : daily_production = 2170 := by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l3346_334626


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3346_334675

/-- The equation of the tangent line to y = xe^(x-1) at (1, 1) is y = 2x - 1 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (x - 1)) → -- Curve equation
  (1 = 1 * Real.exp (1 - 1)) → -- Point (1, 1) satisfies the curve equation
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ 
    (y - 1 = m * (x - 1)) ∧   -- Point-slope form of tangent line
    (m = (1 + 1) * Real.exp (1 - 1)) ∧ -- Slope at x = 1
    (y = 2 * x - 1)) -- Equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3346_334675


namespace NUMINAMATH_CALUDE_product_equality_l3346_334639

theorem product_equality : 250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3346_334639


namespace NUMINAMATH_CALUDE_coefficient_is_40_l3346_334618

/-- The coefficient of x^3y^2 in the expansion of (x-2y)^5 -/
def coefficient : ℤ := 
  (Nat.choose 5 2) * (-2)^2

/-- Theorem stating that the coefficient of x^3y^2 in (x-2y)^5 is 40 -/
theorem coefficient_is_40 : coefficient = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_is_40_l3346_334618


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3346_334676

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 →
  L = 1636 →
  L = 6 * S + R →
  R < S →
  R = 10 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3346_334676


namespace NUMINAMATH_CALUDE_bus_length_calculation_l3346_334600

/-- Calculates the length of a bus given its speed, the speed of a person moving in the opposite direction, and the time it takes for the bus to pass the person. -/
theorem bus_length_calculation (bus_speed : ℝ) (skater_speed : ℝ) (passing_time : ℝ) :
  bus_speed = 40 ∧ skater_speed = 8 ∧ passing_time = 1.125 →
  (bus_speed + skater_speed) * passing_time * (5 / 18) = 45 :=
by sorry

end NUMINAMATH_CALUDE_bus_length_calculation_l3346_334600


namespace NUMINAMATH_CALUDE_fred_grew_four_carrots_l3346_334649

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := 6

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := total_carrots - sally_carrots

theorem fred_grew_four_carrots : fred_carrots = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_four_carrots_l3346_334649


namespace NUMINAMATH_CALUDE_five_pointed_star_angle_sum_l3346_334651

/-- An irregular five-pointed star -/
structure FivePointedStar where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The angles at the vertices of a five-pointed star -/
structure StarAngles where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  ε : ℝ

/-- The sum of angles in a five-pointed star is 180 degrees -/
theorem five_pointed_star_angle_sum (star : FivePointedStar) (angles : StarAngles) :
  angles.α + angles.β + angles.γ + angles.δ + angles.ε = 180 := by
  sorry

#check five_pointed_star_angle_sum

end NUMINAMATH_CALUDE_five_pointed_star_angle_sum_l3346_334651


namespace NUMINAMATH_CALUDE_base3_sum_equality_l3346_334629

/-- Converts a base 3 number represented as a list of digits to a natural number. -/
def base3ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- The sum of 2₃, 121₃, 1212₃, and 12121₃ equals 2111₃ in base 3. -/
theorem base3_sum_equality : 
  base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1] = 
  base3ToNat [1, 1, 1, 2] := by
  sorry

#eval base3ToNat [2] + base3ToNat [1, 2, 1] + base3ToNat [2, 1, 2, 1] + base3ToNat [1, 2, 1, 2, 1]
#eval base3ToNat [1, 1, 1, 2]

end NUMINAMATH_CALUDE_base3_sum_equality_l3346_334629


namespace NUMINAMATH_CALUDE_four_dice_same_number_l3346_334656

-- Define a standard six-sided die
def standard_die := Finset.range 6

-- Define the probability of getting the same number on all four dice
def same_number_probability : ℚ :=
  (1 : ℚ) / (standard_die.card ^ 4)

-- Theorem statement
theorem four_dice_same_number :
  same_number_probability = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_l3346_334656


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l3346_334693

/-- The area of Andrea's living room floor, given a carpet covering 20% of it --/
theorem andreas_living_room_area 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) 
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage_percent = 20) : 
  carpet_length * carpet_width / (carpet_coverage_percent / 100) = 180 := by
sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l3346_334693


namespace NUMINAMATH_CALUDE_triangle_area_with_squares_l3346_334624

/-- Given a scalene triangle with adjoining squares, prove its area -/
theorem triangle_area_with_squares (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 = 100 → b^2 = 64 → c^2 = 49 → h^2 = 81 →
  (1/2 : ℝ) * a * h = 45 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_with_squares_l3346_334624


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3346_334663

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, s n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If s_30 = s_60 for an arithmetic sequence, then s_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.s 30 = seq.s 60) : seq.s 90 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3346_334663


namespace NUMINAMATH_CALUDE_find_other_number_l3346_334674

theorem find_other_number (x y : ℤ) (h1 : 4 * x + 3 * y = 154) (h2 : x = 14 ∨ y = 14) : x = 28 ∨ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3346_334674


namespace NUMINAMATH_CALUDE_coin_flipping_theorem_l3346_334648

theorem coin_flipping_theorem :
  ∀ (initial_state : Fin 2015 → Bool),
  ∃! (final_state : Bool),
    (∀ (i : Fin 2015), final_state = initial_state i) ∨
    (∀ (i : Fin 2015), final_state ≠ initial_state i) :=
by
  sorry


end NUMINAMATH_CALUDE_coin_flipping_theorem_l3346_334648


namespace NUMINAMATH_CALUDE_system_solution_l3346_334665

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, x + y = a ∧ 2 * x + y = 16 ∧ x = 6 ∧ y = b) → 
  a = 10 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3346_334665


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_proof_l3346_334611

def number_of_pencils : ℕ := 48
def pencil_pen_difference : ℕ := 8

def number_of_pens : ℕ := number_of_pencils - pencil_pen_difference

def pen_pencil_ratio : ℚ × ℚ := (5, 6)

theorem pen_pencil_ratio_proof :
  (number_of_pens : ℚ) / (number_of_pencils : ℚ) = pen_pencil_ratio.1 / pen_pencil_ratio.2 :=
by sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_proof_l3346_334611


namespace NUMINAMATH_CALUDE_puppies_given_sandy_friend_puppies_l3346_334666

/-- Given the initial number of puppies and the total number of puppies after receiving more,
    calculate the number of puppies Sandy's friend gave her. -/
theorem puppies_given (initial : ℝ) (total : ℕ) : ℝ :=
  total - initial

/-- Prove that the number of puppies Sandy's friend gave her is 4. -/
theorem sandy_friend_puppies : puppies_given 8 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_sandy_friend_puppies_l3346_334666


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3346_334669

theorem diophantine_equation_solution (a b : ℕ+) 
  (h1 : (b ^ 619 : ℕ) ∣ (a ^ 1000 : ℕ) + 1)
  (h2 : (a ^ 619 : ℕ) ∣ (b ^ 1000 : ℕ) + 1) :
  a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3346_334669


namespace NUMINAMATH_CALUDE_problem_solution_l3346_334698

theorem problem_solution (x y : ℝ) (h : |x - Real.sqrt 3 + 1| + Real.sqrt (y - 2) = 0) :
  (x = Real.sqrt 3 - 1 ∧ y = 2) ∧ x^2 + 2*x - 3*y = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3346_334698


namespace NUMINAMATH_CALUDE_sum_integers_11_to_24_l3346_334608

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_integers_11_to_24 : sum_integers 11 24 = 245 := by sorry

end NUMINAMATH_CALUDE_sum_integers_11_to_24_l3346_334608


namespace NUMINAMATH_CALUDE_problem_solution_l3346_334643

theorem problem_solution :
  -- Part 1
  let n : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * i + 1)
  let m : ℕ := Finset.sum (Finset.range 16) (λ i => 2 * (i + 1))
  m - n = 16 ∧
  -- Part 2
  let trapezium_area (a b h : ℝ) := (a + b) * h / 2
  trapezium_area 4 16 16 = 160 ∧
  -- Part 3
  let isosceles_triangle (side angle : ℝ) := side > 0 ∧ 0 < angle ∧ angle < π
  ∀ side angle, isosceles_triangle side angle → angle = π / 3 → 3 = 3 ∧
  -- Part 4
  let f (x : ℝ) := 3 * x^(2/3) - 8 * x^(1/3) + 4
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ x = 8/27 ∧ ∀ y, y > 0 → f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3346_334643


namespace NUMINAMATH_CALUDE_largest_integer_x_l3346_334617

theorem largest_integer_x : ∀ x : ℤ, (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7 ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_x_l3346_334617


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3346_334699

/-- In a Cartesian coordinate system, the coordinates of a point with respect to the origin are equal to the point's coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : (x, y) = (x, y) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3346_334699


namespace NUMINAMATH_CALUDE_inner_hexagon_area_lower_bound_l3346_334647

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular hexagon -/
structure PointInHexagon (h : RegularHexagon) where
  point : ℝ × ℝ
  is_inside : sorry

/-- The hexagon formed by connecting a point to the vertices of a regular hexagon -/
def inner_hexagon (h : RegularHexagon) (p : PointInHexagon h) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner hexagon is at least 2/3 of the original hexagon -/
theorem inner_hexagon_area_lower_bound (h : RegularHexagon) (p : PointInHexagon h) :
  area (inner_hexagon h p) ≥ (2/3) * area (Set.range h.vertices) :=
sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_lower_bound_l3346_334647


namespace NUMINAMATH_CALUDE_f_max_min_l3346_334610

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- State the theorem
theorem f_max_min :
  (∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5) ∧
  (∃ (x : ℝ), f x = -27 ∧ ∀ (y : ℝ), f y ≥ -27) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_l3346_334610


namespace NUMINAMATH_CALUDE_parabola_parameter_distance_l3346_334686

/-- Parabola type representing y = ax^2 -/
structure Parabola where
  a : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to calculate the distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (p : Parabola) (pt : Point) : ℝ :=
  if p.a > 0 then
    abs (pt.y + 1 / (4 * p.a))
  else
    abs (pt.y - 1 / (4 * p.a))

/-- Theorem stating the relationship between the parabola parameter and the distance to directrix -/
theorem parabola_parameter_distance (p : Parabola) :
  let m : Point := ⟨2, 1⟩
  distance_to_directrix p m = 2 →
  p.a = 1/4 ∨ p.a = -1/12 :=
sorry

end NUMINAMATH_CALUDE_parabola_parameter_distance_l3346_334686


namespace NUMINAMATH_CALUDE_acute_angle_inequalities_l3346_334659

open Real

theorem acute_angle_inequalities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π/2) 
  (h_acute_β : 0 < β ∧ β < π/2) 
  (h_α_lt_β : α < β) : 
  (α - sin α < β - sin β) ∧ (tan α - α < tan β - β) := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_inequalities_l3346_334659


namespace NUMINAMATH_CALUDE_set_d_forms_triangle_l3346_334604

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem set_d_forms_triangle :
  can_form_triangle 6 6 6 := by
  sorry

end NUMINAMATH_CALUDE_set_d_forms_triangle_l3346_334604


namespace NUMINAMATH_CALUDE_train_speed_l3346_334614

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 120) (h2 : time = 16) :
  length / time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3346_334614


namespace NUMINAMATH_CALUDE_platform_length_l3346_334668

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length + platform_length) / platform_time = train_length / pole_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3346_334668


namespace NUMINAMATH_CALUDE_find_a_value_l3346_334625

theorem find_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x^3 + 9 * x^2 + 6 * x - 7) →
  (((fun x ↦ 3 * a * x^2 + 18 * x + 6) (-1)) = 4) →
  a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3346_334625


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l3346_334658

theorem quadratic_roots_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ 
    x₂^2 + p*x₂ + q = 0 ∧ 
    |x₁ - x₂| = 2) →
  p = Real.sqrt (4*q + 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l3346_334658
