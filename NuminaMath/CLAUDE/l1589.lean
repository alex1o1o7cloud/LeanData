import Mathlib

namespace NUMINAMATH_CALUDE_painting_wall_coverage_percentage_l1589_158958

/-- Represents the dimensions of a rectangular painting -/
structure PaintingDimensions where
  length : ℚ
  width : ℚ

/-- Represents the dimensions of an irregular pentagonal wall -/
structure WallDimensions where
  side1 : ℚ
  side2 : ℚ
  side3 : ℚ
  side4 : ℚ
  side5 : ℚ
  height : ℚ

/-- Calculates the area of a rectangular painting -/
def paintingArea (p : PaintingDimensions) : ℚ :=
  p.length * p.width

/-- Calculates the approximate area of the irregular pentagonal wall -/
def wallArea (w : WallDimensions) : ℚ :=
  (w.side3 * w.height) / 2

/-- Calculates the percentage of the wall covered by the painting -/
def coveragePercentage (p : PaintingDimensions) (w : WallDimensions) : ℚ :=
  (paintingArea p / wallArea w) * 100

/-- Theorem stating that the painting covers approximately 39.21% of the wall -/
theorem painting_wall_coverage_percentage :
  let painting := PaintingDimensions.mk (13/4) (38/5)
  let wall := WallDimensions.mk 4 12 14 10 8 9
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
    abs (coveragePercentage painting wall - 39.21) < ε :=
sorry

end NUMINAMATH_CALUDE_painting_wall_coverage_percentage_l1589_158958


namespace NUMINAMATH_CALUDE_images_per_card_l1589_158900

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The total amount John spent on memory cards in dollars -/
def total_spent : ℕ := 13140

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem images_per_card : 
  (years * days_per_year * pictures_per_day) / (total_spent / cost_per_card) = 50 := by
  sorry

end NUMINAMATH_CALUDE_images_per_card_l1589_158900


namespace NUMINAMATH_CALUDE_line_segment_length_l1589_158983

theorem line_segment_length (volume : ℝ) (radius : ℝ) (length : ℝ) : 
  volume = 432 * Real.pi →
  radius = 4 →
  volume = (Real.pi * radius^2 * length) + (2 * (2/3) * Real.pi * radius^3) →
  length = 50/3 := by
sorry

end NUMINAMATH_CALUDE_line_segment_length_l1589_158983


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l1589_158990

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_line_triangle_area : 
  let f (x : ℝ) := Real.exp x
  let x₀ : ℝ := 2
  let y₀ : ℝ := Real.exp x₀
  let m : ℝ := Real.exp x₀  -- slope of the tangent line
  let b : ℝ := y₀ - m * x₀  -- y-intercept of the tangent line
  let x_intercept : ℝ := -b / m  -- x-intercept of the tangent line
  Real.exp 2 / 2 = (x_intercept * y₀) / 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l1589_158990


namespace NUMINAMATH_CALUDE_square_field_area_l1589_158938

/-- Given a square field where the diagonal can be traversed at 8 km/hr in 0.5 hours,
    the area of the field is 8 square kilometers. -/
theorem square_field_area (speed : ℝ) (time : ℝ) (diagonal : ℝ) (side : ℝ) (area : ℝ) : 
  speed = 8 →
  time = 0.5 →
  diagonal = speed * time →
  diagonal^2 = 2 * side^2 →
  area = side^2 →
  area = 8 := by
sorry

end NUMINAMATH_CALUDE_square_field_area_l1589_158938


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1589_158937

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5) + (c+6)) / 7) →
  ((d + (d+1) + (d+2) + (d+3) + (d+4) + (d+5) + (d+6)) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1589_158937


namespace NUMINAMATH_CALUDE_dot_product_equals_three_l1589_158976

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (3, x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_equals_three (x : ℝ) :
  dot_product vector_a (vector_b x) = 3 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_dot_product_equals_three_l1589_158976


namespace NUMINAMATH_CALUDE_numeric_methods_count_l1589_158981

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 225

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control_students : ℕ := 450

/-- The number of second-year students studying both subjects -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 676

/-- The approximate percentage of second-year students -/
def second_year_percentage : ℚ := 80 / 100

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 541

theorem numeric_methods_count : 
  numeric_methods_students = 
    total_second_year_students + both_subjects_students - automatic_control_students :=
by sorry

end NUMINAMATH_CALUDE_numeric_methods_count_l1589_158981


namespace NUMINAMATH_CALUDE_order_of_expressions_l1589_158985

theorem order_of_expressions :
  let a : ℝ := (1/2)^(1/3)
  let b : ℝ := (1/3)^(1/2)
  let c : ℝ := Real.log (3/Real.pi)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l1589_158985


namespace NUMINAMATH_CALUDE_trace_equation_equiv_equal_distance_l1589_158935

/-- 
For any point P(x, y) in a 2D coordinate system, the trace equation y = |x| 
is equivalent to the condition that the distance from P to the x-axis 
is equal to the distance from P to the y-axis.
-/
theorem trace_equation_equiv_equal_distance (x y : ℝ) : 
  y = |x| ↔ |y| = |x| :=
sorry

end NUMINAMATH_CALUDE_trace_equation_equiv_equal_distance_l1589_158935


namespace NUMINAMATH_CALUDE_correct_recommendation_plans_l1589_158920

def total_students : ℕ := 7
def students_to_recommend : ℕ := 4
def sports_talented : ℕ := 2
def artistic_talented : ℕ := 2
def other_talented : ℕ := 3

def recommendation_plans : ℕ := sorry

theorem correct_recommendation_plans : recommendation_plans = 25 := by sorry

end NUMINAMATH_CALUDE_correct_recommendation_plans_l1589_158920


namespace NUMINAMATH_CALUDE_largest_square_from_rectangle_l1589_158943

theorem largest_square_from_rectangle (width length : ℕ) 
  (h_width : width = 63) (h_length : length = 42) :
  Nat.gcd width length = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_from_rectangle_l1589_158943


namespace NUMINAMATH_CALUDE_handshakes_eight_people_l1589_158956

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 28. -/
theorem handshakes_eight_people : handshakes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_eight_people_l1589_158956


namespace NUMINAMATH_CALUDE_perfect_square_coefficient_l1589_158968

theorem perfect_square_coefficient (x : ℝ) : ∃ (r s : ℝ), 
  (81/16) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_coefficient_l1589_158968


namespace NUMINAMATH_CALUDE_total_cost_proof_l1589_158975

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem total_cost_proof :
  (roses_bought + daisies_bought) * flower_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1589_158975


namespace NUMINAMATH_CALUDE_sophie_coin_distribution_l1589_158936

/-- The minimum number of additional coins needed for Sophie's distribution. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins Sophie needs. -/
theorem sophie_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 10) (h2 : initial_coins = 40) : 
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

#eval min_additional_coins 10 40

end NUMINAMATH_CALUDE_sophie_coin_distribution_l1589_158936


namespace NUMINAMATH_CALUDE_monthly_parking_fee_l1589_158967

/-- Proves that the monthly parking fee is $40 given the specified conditions -/
theorem monthly_parking_fee (weekly_fee : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_fee = 10 →
  yearly_savings = 40 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_fee : ℕ), monthly_fee = 40 ∧ weeks_per_year * weekly_fee - months_per_year * monthly_fee = yearly_savings :=
by sorry

end NUMINAMATH_CALUDE_monthly_parking_fee_l1589_158967


namespace NUMINAMATH_CALUDE_equation_root_l1589_158909

theorem equation_root : ∃ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_equation_root_l1589_158909


namespace NUMINAMATH_CALUDE_product_correction_l1589_158908

theorem product_correction (a b : ℕ) : 
  a > 9 ∧ a < 100 ∧ (a - 3) * b = 224 → a * b = 245 := by
  sorry

end NUMINAMATH_CALUDE_product_correction_l1589_158908


namespace NUMINAMATH_CALUDE_marked_circles_alignment_l1589_158940

/-- Two identical circles, each marked with k arcs -/
structure MarkedCircle where
  k : ℕ
  arcs : Fin k → ℝ
  arc_measure : ∀ i, arcs i < 180 / (k^2 - k + 1)
  alignment : ∃ r : ℝ, ∀ i, ∃ j, arcs i = (fun x => (x + r) % 360) (arcs j)

/-- The theorem statement -/
theorem marked_circles_alignment (c1 c2 : MarkedCircle) (h : c1 = c2) :
  ∃ r : ℝ, ∀ i, ∀ j, c1.arcs i ≠ (fun x => (x + r) % 360) (c2.arcs j) := by
  sorry

end NUMINAMATH_CALUDE_marked_circles_alignment_l1589_158940


namespace NUMINAMATH_CALUDE_chocolate_difference_l1589_158905

theorem chocolate_difference (friend1 friend2 friend3 : ℚ)
  (h1 : friend1 = 5/6)
  (h2 : friend2 = 2/3)
  (h3 : friend3 = 7/9) :
  max friend1 (max friend2 friend3) - min friend1 (min friend2 friend3) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_difference_l1589_158905


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1589_158986

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  x : ℕ
  is_even : Even x

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.x + (t.x + 2) + (t.x + 4)

/-- Checks if the triangle inequality holds for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.x + (t.x + 2) > t.x + 4 ∧
  t.x + (t.x + 4) > t.x + 2 ∧
  (t.x + 2) + (t.x + 4) > t.x

/-- The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t' ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l1589_158986


namespace NUMINAMATH_CALUDE_coupon_value_l1589_158984

def total_price : ℕ := 67
def num_people : ℕ := 3
def individual_contribution : ℕ := 21

theorem coupon_value :
  total_price - (num_people * individual_contribution) = 4 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l1589_158984


namespace NUMINAMATH_CALUDE_total_sheets_used_l1589_158921

theorem total_sheets_used (total_classes : ℕ) (first_class_count : ℕ) (last_class_count : ℕ)
  (first_class_students : ℕ) (last_class_students : ℕ)
  (first_class_sheets_per_student : ℕ) (last_class_sheets_per_student : ℕ) :
  total_classes = first_class_count + last_class_count →
  first_class_count = 3 →
  last_class_count = 3 →
  first_class_students = 22 →
  last_class_students = 18 →
  first_class_sheets_per_student = 6 →
  last_class_sheets_per_student = 4 →
  (first_class_count * first_class_students * first_class_sheets_per_student) +
  (last_class_count * last_class_students * last_class_sheets_per_student) = 612 :=
by sorry

end NUMINAMATH_CALUDE_total_sheets_used_l1589_158921


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l1589_158972

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l1589_158972


namespace NUMINAMATH_CALUDE_cube_root_of_quarter_l1589_158962

theorem cube_root_of_quarter (t s : ℝ) : t = 15 * s^3 ∧ t = 3.75 → s = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_quarter_l1589_158962


namespace NUMINAMATH_CALUDE_system_solution_l1589_158961

theorem system_solution :
  ∃ (x y z : ℝ), 
    (x + 2*y = 4) ∧ 
    (2*x + 5*y - 2*z = 11) ∧ 
    (3*x - 5*y + 2*z = -1) ∧
    (x = 2) ∧ (y = 1) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1589_158961


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1589_158907

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 1)^2 + 1) < 0 ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1589_158907


namespace NUMINAMATH_CALUDE_complex_geometric_sequence_l1589_158913

theorem complex_geometric_sequence (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2*a + 2*Complex.I
  let z₃ : ℂ := 3*a + 4*Complex.I
  (∃ r : ℝ, r > 0 ∧ Complex.abs z₂ = r * Complex.abs z₁ ∧ Complex.abs z₃ = r * Complex.abs z₂) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_geometric_sequence_l1589_158913


namespace NUMINAMATH_CALUDE_fraction_equality_l1589_158970

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : 
  a / b = 0.8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1589_158970


namespace NUMINAMATH_CALUDE_max_t_value_l1589_158919

theorem max_t_value (t : ℝ) (h : t > 0) :
  (∀ u v : ℝ, (u + 5 - 2*v)^2 + (u - v^2)^2 ≥ t^2) →
  t ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l1589_158919


namespace NUMINAMATH_CALUDE_subtraction_multiplication_problem_l1589_158955

theorem subtraction_multiplication_problem : 
  let initial_value : ℚ := 555.55
  let subtracted_value : ℚ := 111.11
  let multiplier : ℚ := 2
  let result : ℚ := (initial_value - subtracted_value) * multiplier
  result = 888.88 := by sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_problem_l1589_158955


namespace NUMINAMATH_CALUDE_inequality_proof_l1589_158957

theorem inequality_proof (a b c A B C u v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hu : 0 < u) (hv : 0 < v)
  (h1 : a * u^2 - b * u + c ≤ 0)
  (h2 : A * v^2 - B * v + C ≤ 0) :
  (a * u + A * v) * (c / u + C / v) ≤ ((b + B) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1589_158957


namespace NUMINAMATH_CALUDE_sixth_segment_length_l1589_158942

def segment_lengths (a : Fin 7 → ℕ) : Prop :=
  a 0 = 1 ∧ a 6 = 21 ∧ 
  (∀ i j, i < j → a i < a j) ∧
  (∀ i j k, i < j ∧ j < k → a i + a j ≤ a k)

theorem sixth_segment_length (a : Fin 7 → ℕ) (h : segment_lengths a) : a 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sixth_segment_length_l1589_158942


namespace NUMINAMATH_CALUDE_jelly_bean_division_l1589_158963

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 72 →
  eaten_amount = 12 →
  num_piles = 5 →
  (initial_amount - eaten_amount) / num_piles = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l1589_158963


namespace NUMINAMATH_CALUDE_fraction_equality_l1589_158922

theorem fraction_equality : (1-2+4-8+16-32+64)/(2-4+8-16+32-64+128) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1589_158922


namespace NUMINAMATH_CALUDE_fish_ratio_l1589_158906

/-- Proves that the ratio of blue fish to the total number of fish is 1:2 -/
theorem fish_ratio (blue orange green : ℕ) : 
  blue + orange + green = 80 →  -- Total number of fish
  orange = blue - 15 →          -- 15 fewer orange than blue
  green = 15 →                  -- Number of green fish
  blue * 2 = 80                 -- Ratio of blue to total is 1:2
    := by sorry

end NUMINAMATH_CALUDE_fish_ratio_l1589_158906


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l1589_158959

theorem sqrt_expression_equals_one :
  (Real.sqrt 6 - Real.sqrt 2) / Real.sqrt 2 + |Real.sqrt 3 - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l1589_158959


namespace NUMINAMATH_CALUDE_volume_surface_area_radius_relation_l1589_158965

/-- A convex polyhedron with an inscribed sphere -/
class ConvexPolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- The theorem stating the relationship between volume, surface area, and inscribed sphere radius -/
theorem volume_surface_area_radius_relation (P : ConvexPolyhedron) : 
  P.volume = (1 / 3) * P.surface_area * P.inscribed_radius :=
sorry

end NUMINAMATH_CALUDE_volume_surface_area_radius_relation_l1589_158965


namespace NUMINAMATH_CALUDE_triangular_number_difference_l1589_158998

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2008th triangular numbers is 4019 -/
theorem triangular_number_difference : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l1589_158998


namespace NUMINAMATH_CALUDE_matrix_power_10_l1589_158944

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 1, 1]

theorem matrix_power_10 : A ^ 10 = !![512, 512; 512, 512] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_10_l1589_158944


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1589_158914

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1589_158914


namespace NUMINAMATH_CALUDE_multiplication_mistake_difference_l1589_158993

theorem multiplication_mistake_difference : 
  let correct_multiplicand : Nat := 136
  let correct_multiplier : Nat := 43
  let mistaken_multiplier : Nat := 34
  (correct_multiplicand * correct_multiplier) - (correct_multiplicand * mistaken_multiplier) = 1224 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_difference_l1589_158993


namespace NUMINAMATH_CALUDE_balloon_final_height_l1589_158996

/-- Represents the sequence of balloon movements -/
def BalloonMovements : List Int := [6, -2, 3, -2]

/-- Calculates the final height of the balloon after a sequence of movements -/
def finalHeight (movements : List Int) : Int :=
  movements.foldl (· + ·) 0

/-- Theorem stating that the final height of the balloon is 5 meters -/
theorem balloon_final_height :
  finalHeight BalloonMovements = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_final_height_l1589_158996


namespace NUMINAMATH_CALUDE_distinct_pairs_count_l1589_158995

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents a marble with a color and quantity --/
structure Marble where
  color : Color
  quantity : Nat

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def countDistinctPairs (marbles : List Marble) : Nat :=
  sorry

/-- Theorem: Given the specific set of marbles, the number of distinct pairs is 7 --/
theorem distinct_pairs_count :
  let marbles : List Marble := [
    ⟨Color.Red, 1⟩,
    ⟨Color.Green, 1⟩,
    ⟨Color.Blue, 2⟩,
    ⟨Color.Yellow, 2⟩
  ]
  countDistinctPairs marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_count_l1589_158995


namespace NUMINAMATH_CALUDE_zoo_rabbits_l1589_158934

/-- Given a zoo with parrots and rabbits, where the ratio of parrots to rabbits
    is 3:4 and there are 21 parrots, prove that there are 28 rabbits. -/
theorem zoo_rabbits (parrots : ℕ) (rabbits : ℕ) : 
  parrots = 21 → 3 * rabbits = 4 * parrots → rabbits = 28 := by
  sorry

end NUMINAMATH_CALUDE_zoo_rabbits_l1589_158934


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1589_158987

/-- Given that Rahul's age after 6 years will be 26 and Deepak's current age is 15,
    prove that the ratio of their current ages is 4:3. -/
theorem age_ratio_problem (rahul_future_age : ℕ) (deepak_age : ℕ) : 
  rahul_future_age = 26 → 
  deepak_age = 15 → 
  (rahul_future_age - 6) / deepak_age = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1589_158987


namespace NUMINAMATH_CALUDE_solve_for_y_l1589_158932

theorem solve_for_y (x y : ℝ) (h1 : x^(y+1) = 16) (h2 : x = 8) : y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1589_158932


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l1589_158903

theorem easter_egg_distribution (total_people : ℕ) (eggs_per_person : ℕ) (num_baskets : ℕ) :
  total_people = 20 →
  eggs_per_person = 9 →
  num_baskets = 15 →
  (total_people * eggs_per_person) / num_baskets = 12 := by
sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l1589_158903


namespace NUMINAMATH_CALUDE_volume_ratio_cubes_l1589_158988

/-- Given two cubes with edge lengths in the ratio 3:1, if the volume of the smaller cube is 1 unit,
    then the volume of the larger cube is 27 units. -/
theorem volume_ratio_cubes (e : ℝ) (h1 : e > 0) (h2 : e^3 = 1) :
  (3*e)^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cubes_l1589_158988


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1589_158901

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y - 3 = 0 →
  let m := -1 / Real.sqrt 3
  let α := Real.arctan m
  α = 150 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1589_158901


namespace NUMINAMATH_CALUDE_range_of_function_l1589_158924

theorem range_of_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ a * x - b ∧ a * x - b ≤ 1) →
  ∃ y : ℝ, y ∈ Set.Icc (-4/5) (2/7) ∧
    y = (3 * a + b + 1) / (a + 2 * b - 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_function_l1589_158924


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1589_158948

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_103_l1589_158948


namespace NUMINAMATH_CALUDE_bathroom_width_l1589_158915

/-- Proves that the width of Mrs. Garvey's bathroom is 6 feet -/
theorem bathroom_width : 
  ∀ (length width : ℝ) (tile_side : ℝ) (num_tiles : ℕ),
  length = 10 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side^2 * num_tiles) →
  width = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_bathroom_width_l1589_158915


namespace NUMINAMATH_CALUDE_smallest_survey_size_l1589_158912

theorem smallest_survey_size (n : ℕ) : n > 0 ∧ 
  (∃ k₁ : ℕ, n * (140 : ℚ) / 360 = k₁) ∧
  (∃ k₂ : ℕ, n * (108 : ℚ) / 360 = k₂) ∧
  (∃ k₃ : ℕ, n * (72 : ℚ) / 360 = k₃) ∧
  (∃ k₄ : ℕ, n * (40 : ℚ) / 360 = k₄) →
  n ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_survey_size_l1589_158912


namespace NUMINAMATH_CALUDE_basketball_points_distribution_l1589_158945

theorem basketball_points_distribution (x : ℝ) (y : ℕ) : 
  (1/3 : ℝ) * x + (3/8 : ℝ) * x + 18 + y = x →
  y ≤ 15 →
  (∀ i ∈ Finset.range 5, (y : ℝ) / 5 ≤ 3) →
  y = 14 :=
by sorry

end NUMINAMATH_CALUDE_basketball_points_distribution_l1589_158945


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1589_158917

theorem x_intercept_of_line (x y : ℝ) : 
  (5 * x - 2 * y - 10 = 0) → (y = 0 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1589_158917


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1589_158925

/-- Given a right-angled triangle ABC with ∠A = π/2, 
    prove that arctan(b/(c+a)) + arctan(c/(b+a)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h_right_angle : a^2 = b^2 + c^2) :
  Real.arctan (b / (c + a)) + Real.arctan (c / (b + a)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1589_158925


namespace NUMINAMATH_CALUDE_proportion_solution_l1589_158926

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 7) → x = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1589_158926


namespace NUMINAMATH_CALUDE_hat_shop_pricing_l1589_158910

theorem hat_shop_pricing (original_price : ℝ) (increase_rate : ℝ) (additional_charge : ℝ) (discount_rate : ℝ) : 
  original_price = 40 ∧ 
  increase_rate = 0.3 ∧ 
  additional_charge = 5 ∧ 
  discount_rate = 0.25 → 
  (1 - discount_rate) * (original_price * (1 + increase_rate) + additional_charge) = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_hat_shop_pricing_l1589_158910


namespace NUMINAMATH_CALUDE_utensils_per_pack_l1589_158950

/-- Given that packs have an equal number of knives, forks, and spoons,
    and 5 packs contain 50 spoons, prove that each pack contains 30 utensils. -/
theorem utensils_per_pack (total_packs : ℕ) (total_spoons : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_spoons = 50) :
  let spoons_per_pack := total_spoons / total_packs
  let utensils_per_pack := 3 * spoons_per_pack
  utensils_per_pack = 30 := by
sorry

end NUMINAMATH_CALUDE_utensils_per_pack_l1589_158950


namespace NUMINAMATH_CALUDE_range_of_m_for_true_proposition_l1589_158918

theorem range_of_m_for_true_proposition (m : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x + 1) + m = 0) →
  m ≤ 1 ∧ ∀ y : ℝ, y < m → ∃ x : ℝ, 4^x - 2^(x + 1) + y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_true_proposition_l1589_158918


namespace NUMINAMATH_CALUDE_meaningful_sqrt_fraction_range_l1589_158991

theorem meaningful_sqrt_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (4 - x)) / (Real.sqrt (x - 1))) ↔ (1 < x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_fraction_range_l1589_158991


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1589_158930

/-- The function f(x) = a^(x+2) - 3 passes through the point (-2, -2) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 3
  f (-2) = -2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1589_158930


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1589_158966

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l1589_158966


namespace NUMINAMATH_CALUDE_hexagon_vertex_recovery_erased_vertex_recoverable_l1589_158977

/-- Represents a hexagon with numbers on its vertices -/
structure Hexagon where
  -- Vertex numbers
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Theorem: Any vertex number in a hexagon can be determined from the other five -/
theorem hexagon_vertex_recovery (h : Hexagon) :
  h.a = h.b + h.d + h.f - h.c - h.e :=
by sorry

/-- Corollary: It's possible to recover an erased vertex number in the hexagon -/
theorem erased_vertex_recoverable (h : Hexagon) :
  ∃ (x : ℝ), x = h.b + h.d + h.f - h.c - h.e :=
by sorry

end NUMINAMATH_CALUDE_hexagon_vertex_recovery_erased_vertex_recoverable_l1589_158977


namespace NUMINAMATH_CALUDE_ribbon_cutting_theorem_l1589_158997

/-- Represents the cutting time for a pair of centimeters -/
structure CutTimePair :=
  (first : Nat)
  (second : Nat)

/-- Calculates the total cutting time for the ribbon -/
def totalCutTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  (ribbonLength / 2) * (cutTimePair.first + cutTimePair.second)

/-- Calculates the length of ribbon cut in half the total time -/
def ribbonCutInHalfTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  ((totalCutTime ribbonLength cutTimePair) / 2) / (cutTimePair.first + cutTimePair.second) * 2

theorem ribbon_cutting_theorem (ribbonLength : Nat) (cutTimePair : CutTimePair) :
  ribbonLength = 200 →
  cutTimePair = { first := 35, second := 40 } →
  totalCutTime ribbonLength cutTimePair = 3750 ∧
  ribbonLength - ribbonCutInHalfTime ribbonLength cutTimePair = 150 :=
by sorry

#eval totalCutTime 200 { first := 35, second := 40 }
#eval 200 - ribbonCutInHalfTime 200 { first := 35, second := 40 }

end NUMINAMATH_CALUDE_ribbon_cutting_theorem_l1589_158997


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1589_158929

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 166) (h2 : divisor = 20) (h3 : quotient = 8) :
  dividend % divisor = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1589_158929


namespace NUMINAMATH_CALUDE_range_of_a_l1589_158904

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > a then x + 2 * x else x^2 + 5 * x + 2

/-- Function g(x) defined as f(x) - 2x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * x

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    (∀ w : ℝ, g a w = 0 → w = x ∨ w = y ∨ w = z)) →
  a ∈ Set.Icc (-1) 2 ∧ a ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1589_158904


namespace NUMINAMATH_CALUDE_all_diagonal_triangles_multiplicative_l1589_158911

/-- A regular polygon with n sides, all of length 1 -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (side_length : ℝ := 1)

/-- A triangle formed by diagonals in a regular polygon -/
structure DiagonalTriangle (n : ℕ) (p : RegularPolygon n) where
  (vertex1 : ℝ × ℝ)
  (vertex2 : ℝ × ℝ)
  (vertex3 : ℝ × ℝ)

/-- A triangle is multiplicative if the product of the lengths of two sides equals the length of the third side -/
def is_multiplicative (t : DiagonalTriangle n p) : Prop :=
  ∀ (i j k : Fin 3), i ≠ j → j ≠ k → i ≠ k →
    let sides := [dist t.vertex1 t.vertex2, dist t.vertex2 t.vertex3, dist t.vertex3 t.vertex1]
    sides[i] * sides[j] = sides[k]

/-- The main theorem: all triangles formed by diagonals in a regular polygon are multiplicative -/
theorem all_diagonal_triangles_multiplicative (n : ℕ) (p : RegularPolygon n) :
  ∀ t : DiagonalTriangle n p, is_multiplicative t :=
sorry

end NUMINAMATH_CALUDE_all_diagonal_triangles_multiplicative_l1589_158911


namespace NUMINAMATH_CALUDE_spinner_final_direction_l1589_158992

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation type
inductive RotationType
| Clockwise
| Counterclockwise

-- Define a function to represent a rotation
def rotate (initial : Direction) (amount : Rat) (type : RotationType) : Direction :=
  sorry

-- Define the problem statement
theorem spinner_final_direction 
  (initial : Direction)
  (rotation1 : Rat)
  (type1 : RotationType)
  (rotation2 : Rat)
  (type2 : RotationType)
  (h1 : initial = Direction.South)
  (h2 : rotation1 = 19/4)
  (h3 : type1 = RotationType.Clockwise)
  (h4 : rotation2 = 13/2)
  (h5 : type2 = RotationType.Counterclockwise) :
  rotate (rotate initial rotation1 type1) rotation2 type2 = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l1589_158992


namespace NUMINAMATH_CALUDE_job_completion_time_l1589_158980

/-- The time taken for three workers to complete a job together, given their individual completion times -/
theorem job_completion_time (time_A time_B time_C : ℝ) 
  (hA : time_A = 7) 
  (hB : time_B = 10) 
  (hC : time_C = 12) : 
  1 / (1 / time_A + 1 / time_B + 1 / time_C) = 420 / 137 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1589_158980


namespace NUMINAMATH_CALUDE_odd_function_composition_even_l1589_158989

-- Define an odd function
def OddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_composition_even
  (g : ℝ → ℝ)
  (h : OddFunction g) :
  EvenFunction (fun x ↦ g (g (g (g x)))) :=
sorry

end NUMINAMATH_CALUDE_odd_function_composition_even_l1589_158989


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1589_158974

/-- Given inversely proportional variables x and y, if x + y = 30 and x - y = 10,
    then y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (D : ℝ) (h1 : x * y = D)
    (h2 : x + y = 30) (h3 : x - y = 10) :
  (x = 7) → (y = 200 / 7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1589_158974


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1589_158954

/-- A circle with center (h, k) and radius r is represented by the equation (x - h)² + (y - k)² = r² --/
def is_circle (h k r : ℝ) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A circle is tangent to the x-axis if its distance from the x-axis equals its radius --/
def tangent_to_x_axis (h k r : ℝ) : Prop := k = r

theorem circle_equation_proof (x y : ℝ) :
  let h : ℝ := 2
  let k : ℝ := 1
  let f : ℝ → ℝ → Prop := λ x y ↦ (x - 2)^2 + (y - 1)^2 = 1
  is_circle h k 1 f ∧ tangent_to_x_axis h k 1 := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1589_158954


namespace NUMINAMATH_CALUDE_optimal_pricing_and_profit_daily_profit_function_l1589_158982

/-- Represents the daily profit function for a product --/
def daily_profit (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

/-- Represents the constraint on the selling price --/
def price_constraint (x : ℝ) : Prop := 30 ≤ x ∧ x ≤ 54

/-- The theorem stating the optimal selling price and maximum profit --/
theorem optimal_pricing_and_profit :
  ∃ (x : ℝ), price_constraint x ∧ 
    (∀ y, price_constraint y → daily_profit y ≤ daily_profit x) ∧
    x = 42 ∧ daily_profit x = 432 := by
  sorry

/-- The theorem stating the form of the daily profit function --/
theorem daily_profit_function (x : ℝ) :
  daily_profit x = (x - 30) * (162 - 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_optimal_pricing_and_profit_daily_profit_function_l1589_158982


namespace NUMINAMATH_CALUDE_profit_percentage_l1589_158960

theorem profit_percentage (C S : ℝ) (h : 315 * C = 250 * S) : 
  (S - C) / C * 100 = 26 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1589_158960


namespace NUMINAMATH_CALUDE_class_average_height_l1589_158969

theorem class_average_height (total_girls : ℕ) (group1_girls : ℕ) (group2_girls : ℕ) 
  (group1_avg_height : ℝ) (group2_avg_height : ℝ) :
  total_girls = group1_girls + group2_girls →
  group1_girls = 30 →
  group2_girls = 10 →
  group1_avg_height = 160 →
  group2_avg_height = 156 →
  (group1_girls * group1_avg_height + group2_girls * group2_avg_height) / total_girls = 159 := by
sorry

end NUMINAMATH_CALUDE_class_average_height_l1589_158969


namespace NUMINAMATH_CALUDE_workout_ratio_theorem_l1589_158973

/-- Represents the workout schedule for Rayman, Junior, and Wolverine -/
structure WorkoutSchedule where
  junior_hours : ℝ
  rayman_hours : ℝ
  wolverine_hours : ℝ
  ratio : ℝ

/-- Theorem stating the relationship between workout hours -/
theorem workout_ratio_theorem (w : WorkoutSchedule) 
  (h1 : w.rayman_hours = w.junior_hours / 2)
  (h2 : w.wolverine_hours = 60)
  (h3 : w.wolverine_hours = w.ratio * (w.rayman_hours + w.junior_hours)) :
  w.ratio = 40 / w.junior_hours :=
sorry

end NUMINAMATH_CALUDE_workout_ratio_theorem_l1589_158973


namespace NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l1589_158999

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let initial_blue := (3 / 5) * total
  let initial_red := total - initial_blue
  let new_red := 2 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l1589_158999


namespace NUMINAMATH_CALUDE_juan_saw_eight_pickup_trucks_l1589_158953

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := sorry

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

theorem juan_saw_eight_pickup_trucks : num_pickup_trucks = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_saw_eight_pickup_trucks_l1589_158953


namespace NUMINAMATH_CALUDE_max_large_planes_is_seven_l1589_158952

/-- Calculates the maximum number of planes that can fit in a hangar -/
def max_planes (hangar_length : ℕ) (plane_length : ℕ) (safety_gap : ℕ) : ℕ :=
  (hangar_length) / (plane_length + safety_gap)

/-- Theorem: The maximum number of large planes in the hangar is 7 -/
theorem max_large_planes_is_seven :
  max_planes 900 110 10 = 7 := by
  sorry

#eval max_planes 900 110 10

end NUMINAMATH_CALUDE_max_large_planes_is_seven_l1589_158952


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1589_158994

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let small_hole_diameter : ℝ := 1.5
  let large_hole_diameter : ℝ := 2.5
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 2287.875 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l1589_158994


namespace NUMINAMATH_CALUDE_train_distance_problem_l1589_158946

theorem train_distance_problem (speed1 speed2 distance_difference : ℝ) 
  (h1 : speed1 = 20)
  (h2 : speed2 = 25)
  (h3 : distance_difference = 65)
  : speed1 * (distance_difference / (speed2 - speed1)) + 
    speed2 * (distance_difference / (speed2 - speed1)) = 585 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1589_158946


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1589_158949

/-- Definition of equivalent rational number pair -/
def is_equivalent_pair (m n : ℚ) : Prop := m + n = m * n

/-- Part 1: Prove that (3, 3/2) is an equivalent rational number pair -/
theorem part_one : is_equivalent_pair 3 (3/2) := by sorry

/-- Part 2: If (x+1, 4) is an equivalent rational number pair, then x = 1/3 -/
theorem part_two (x : ℚ) : is_equivalent_pair (x + 1) 4 → x = 1/3 := by sorry

/-- Part 3: If (m, n) is an equivalent rational number pair, 
    then 12 - 6mn + 6m + 6n = 12 -/
theorem part_three (m n : ℚ) : 
  is_equivalent_pair m n → 12 - 6*m*n + 6*m + 6*n = 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1589_158949


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1589_158927

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1589_158927


namespace NUMINAMATH_CALUDE_polynomial_value_l1589_158928

theorem polynomial_value (a b : ℝ) (h1 : a * b = 7) (h2 : a + b = 2) :
  a^2 * b + a * b^2 - 20 = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l1589_158928


namespace NUMINAMATH_CALUDE_heart_diamond_spade_probability_l1589_158902

/-- Probability of drawing a heart, then a diamond, then a spade from a standard 52-card deck -/
theorem heart_diamond_spade_probability : 
  let total_cards : ℕ := 52
  let hearts : ℕ := 13
  let diamonds : ℕ := 13
  let spades : ℕ := 13
  (hearts : ℚ) / total_cards * 
  (diamonds : ℚ) / (total_cards - 1) * 
  (spades : ℚ) / (total_cards - 2) = 2197 / 132600 := by
sorry

end NUMINAMATH_CALUDE_heart_diamond_spade_probability_l1589_158902


namespace NUMINAMATH_CALUDE_largest_n_is_correct_l1589_158971

/-- The largest positive integer n for which the system of equations has integer solutions -/
def largest_n : ℕ := 3

/-- Predicate to check if a given n has integer solutions for the system of equations -/
def has_integer_solution (n : ℕ) : Prop :=
  ∃ x : ℤ, ∃ y : Fin n → ℤ,
    ∀ i j : Fin n, (x + i.val + 1)^2 + y i^2 = (x + j.val + 1)^2 + y j^2

/-- Theorem stating that largest_n is indeed the largest n with integer solutions -/
theorem largest_n_is_correct :
  (has_integer_solution largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬(has_integer_solution m)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_is_correct_l1589_158971


namespace NUMINAMATH_CALUDE_rational_equation_system_l1589_158978

theorem rational_equation_system (x y z : ℚ) 
  (eq1 : x - y + 2 * z = 1)
  (eq2 : x + y + 4 * z = 3) : 
  x + 2 * y + 5 * z = 4 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_system_l1589_158978


namespace NUMINAMATH_CALUDE_girls_in_class_l1589_158951

theorem girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 →
  girls + boys = 41 →
  girls = 22 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l1589_158951


namespace NUMINAMATH_CALUDE_function_properties_l1589_158931

open Real

theorem function_properties (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < -1) :
  (f (-2) > f 2 + 4) ∧ 
  (∀ x : ℝ, f x > f (x + 1) + 1) ∧ 
  (∃ x : ℝ, x ≥ 0 ∧ f (sqrt x) + sqrt x < f 0) ∧
  (∀ a : ℝ, a ≠ 0 → f (|a| + 1 / |a|) + |a| + 1 / |a| < f 2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1589_158931


namespace NUMINAMATH_CALUDE_function_properties_l1589_158947

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the given conditions
variable (h1 : ∀ x, f (x + 3) = g (-x) + 2)
variable (h2 : ∀ x, f' (x - 1) = g' x)
variable (h3 : ∀ x, g (-x + 1) = -g (x + 1))

-- State the properties to be proved
theorem function_properties :
  (g 1 = 0) ∧
  (∀ x, g' (x + 1) = -g' (3 - x)) ∧
  (∀ x, g (x + 1) = g (3 - x)) ∧
  (∀ x, g (x + 4) = g x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1589_158947


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1589_158941

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A week has 7 days -/
def daysInWeek : ℕ := 7

/-- A leap year has 52 complete weeks and 2 extra days -/
def leapYearWeeks : ℕ := 52
def leapYearExtraDays : ℕ := 2

/-- The probability of a randomly selected leap year having 53 Sundays -/
def probLeapYear53Sundays : ℚ := 2 / 7

theorem leap_year_53_sundays_probability :
  probLeapYear53Sundays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1589_158941


namespace NUMINAMATH_CALUDE_unique_solution_f_f_eq_zero_l1589_158939

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x + 4 else 3*x - 6

-- Theorem statement
theorem unique_solution_f_f_eq_zero :
  ∃! x : ℝ, f (f x) = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_f_f_eq_zero_l1589_158939


namespace NUMINAMATH_CALUDE_next_common_term_l1589_158933

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem next_common_term
  (a₁ b₁ d₁ d₂ : ℤ)
  (h₁ : a₁ = 3)
  (h₂ : b₁ = 16)
  (h₃ : d₁ = 17)
  (h₄ : d₂ = 11)
  (h₅ : ∃ (n m : ℕ), arithmetic_sequence a₁ d₁ n = 71 ∧ arithmetic_sequence b₁ d₂ m = 71)
  : ∃ (k l : ℕ), 
    arithmetic_sequence a₁ d₁ k = arithmetic_sequence b₁ d₂ l ∧
    arithmetic_sequence a₁ d₁ k > 71 ∧
    arithmetic_sequence a₁ d₁ k = 258 :=
sorry

end NUMINAMATH_CALUDE_next_common_term_l1589_158933


namespace NUMINAMATH_CALUDE_sams_walking_speed_l1589_158964

/-- Proves that Sam's walking speed is 5 miles per hour given the problem conditions -/
theorem sams_walking_speed (initial_distance : ℝ) (freds_speed : ℝ) (sams_distance : ℝ) : 
  initial_distance = 35 →
  freds_speed = 2 →
  sams_distance = 25 →
  (initial_distance - sams_distance) / freds_speed = sams_distance / 5 := by
  sorry

#check sams_walking_speed

end NUMINAMATH_CALUDE_sams_walking_speed_l1589_158964


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1589_158916

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  a 2016 = a 2015 + 2 * a 2014 →
  a m * a n = 16 * (a 1)^2 →
  (4 : ℝ) / m + 1 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1589_158916


namespace NUMINAMATH_CALUDE_area_at_stage_6_l1589_158979

/-- The side length of each square in inches -/
def square_side : ℕ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℕ :=
  (num_squares stage) * (square_side * square_side)

/-- Theorem: The area of the rectangle at Stage 6 is 96 square inches -/
theorem area_at_stage_6 : rectangle_area 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_6_l1589_158979


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1589_158923

theorem max_sum_of_squares (a b c d : ℕ+) (h : a^2 + b^2 + c^2 + d^2 = 70) :
  a + b + c + d ≤ 16 ∧ ∃ (a' b' c' d' : ℕ+), a'^2 + b'^2 + c'^2 + d'^2 = 70 ∧ a' + b' + c' + d' = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1589_158923
