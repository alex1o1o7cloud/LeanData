import Mathlib

namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1634_163479

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 14)
  (h3 : Complex.abs (w - 2*z) = 2) :
  Complex.abs (w^3 + z^3) = 38 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1634_163479


namespace NUMINAMATH_CALUDE_mortgage_repayment_duration_l1634_163450

theorem mortgage_repayment_duration (a : ℝ) (r : ℝ) (S : ℝ) (h1 : a = 400) (h2 : r = 2) (h3 : S = 819200) :
  ∃ n : ℕ, n = 11 ∧ S = a * (1 - r^n) / (1 - r) ∧ ∀ m : ℕ, m < n → S > a * (1 - r^m) / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_mortgage_repayment_duration_l1634_163450


namespace NUMINAMATH_CALUDE_area_triangle_abc_is_ten_l1634_163402

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Area of triangle ABC is 10 -/
theorem area_triangle_abc_is_ten
  (M₁ : Parabola)
  (M₂ : Parabola)
  (A : Point)
  (B : Point)
  (C : Point)
  (h₁ : M₂.b = -2 * M₂.a) -- M₂ is a horizontal translation of M₁
  (h₂ : A.y = M₂.a * A.x^2 + M₂.b * A.x + M₂.c) -- A is on M₂
  (h₃ : B.x = C.x) -- B and C are on the axis of symmetry of M₂
  (h₄ : C.x = 2 ∧ C.y = M₁.c - 5) -- Coordinates of C
  (h₅ : B.y = M₁.a * B.x^2 + M₁.c) -- B is on M₁
  (h₆ : C.y = M₂.a * C.x^2 + M₂.b * C.x + M₂.c) -- C is on M₂
  : (1/2 : ℝ) * |C.x - A.x| * |C.y - B.y| = 10 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_abc_is_ten_l1634_163402


namespace NUMINAMATH_CALUDE_subset_star_inclusion_l1634_163497

/-- Given non-empty sets of real numbers M and P, where M ⊆ P, prove that P* ⊆ M* -/
theorem subset_star_inclusion {M P : Set ℝ} (hM : M.Nonempty) (hP : P.Nonempty) (h_subset : M ⊆ P) :
  {y : ℝ | ∀ x ∈ P, y ≥ x} ⊆ {y : ℝ | ∀ x ∈ M, y ≥ x} := by
  sorry

end NUMINAMATH_CALUDE_subset_star_inclusion_l1634_163497


namespace NUMINAMATH_CALUDE_red_on_third_prob_l1634_163458

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a specific outcome on the RedDie -/
def roll_prob (d : RedDie) (is_red : Bool) : ℚ :=
  if is_red then d.red_sides / d.sides else (d.sides - d.red_sides) / d.sides

/-- The probability of the die landing with a red side up for the first time on the third roll -/
def red_on_third (d : RedDie) : ℚ :=
  (roll_prob d false) * (roll_prob d false) * (roll_prob d true)

theorem red_on_third_prob (d : RedDie) : 
  red_on_third d = 147 / 1000 := by sorry

end NUMINAMATH_CALUDE_red_on_third_prob_l1634_163458


namespace NUMINAMATH_CALUDE_statements_correctness_l1634_163415

-- Define the statements
def statement_A (l : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, l = {(x, y) | x + y = c} ∧ (-2, -3) ∈ l ∧ c = -5

def statement_B (m : ℝ) : Prop :=
  (1, 3) ∈ {(x, y) | 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0}

def statement_C (θ : ℝ) : Prop :=
  ∀ x y : ℝ, y - 1 = Real.tan θ * (x - 1) ↔ (x, y) ∈ {(x, y) | y - 1 = Real.tan θ * (x - 1)}

def statement_D (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∀ x y : ℝ, (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁) ↔
    (x, y) ∈ {(x, y) | (x₂ - x₁) * (y - y₁) = (y₂ - y₁) * (x - x₁)}

-- Theorem stating which statements are correct and incorrect
theorem statements_correctness :
  (∃ l : Set (ℝ × ℝ), ¬statement_A l) ∧
  (∀ m : ℝ, statement_B m) ∧
  (∃ θ : ℝ, ¬statement_C θ) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, statement_D x₁ y₁ x₂ y₂) := by
  sorry


end NUMINAMATH_CALUDE_statements_correctness_l1634_163415


namespace NUMINAMATH_CALUDE_total_slices_is_seven_l1634_163491

/-- The number of slices of pie sold yesterday -/
def slices_yesterday : ℕ := 5

/-- The number of slices of pie served today -/
def slices_today : ℕ := 2

/-- The total number of slices of pie sold -/
def total_slices : ℕ := slices_yesterday + slices_today

theorem total_slices_is_seven : total_slices = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_seven_l1634_163491


namespace NUMINAMATH_CALUDE_function_properties_l1634_163418

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x + b

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

theorem function_properties (a b : ℝ) :
  f a b 0 = 2 →
  f' a 1 = 0 →
  (∃ (x : ℝ), f 3 2 x = f a b x) ∧
  (∀ (x : ℝ), x < -3 → (f' 3 x > 0)) ∧
  (∀ (x : ℝ), -3 < x ∧ x < 1 → (f' 3 x < 0)) ∧
  (∀ (x : ℝ), x > 1 → (f' 3 x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1634_163418


namespace NUMINAMATH_CALUDE_courtyard_paving_l1634_163439

/-- Calculate the number of bricks required to pave a courtyard -/
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width : ℝ) 
  (h1 : courtyard_length = 35)
  (h2 : courtyard_width = 24)
  (h3 : brick_length = 0.15)
  (h4 : brick_width = 0.08) :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 70000 := by
  sorry

#eval (35 * 24) / (0.15 * 0.08)

end NUMINAMATH_CALUDE_courtyard_paving_l1634_163439


namespace NUMINAMATH_CALUDE_expressions_correctness_l1634_163493

theorem expressions_correctness (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) :
  (∃ x : ℝ, x * x = a / b) ∧ 
  (∃ y : ℝ, y * y = b / a) ∧
  (∃ z : ℝ, z * z = a * b) ∧
  (∃ w : ℝ, w * w = a / b) ∧
  (Real.sqrt (a / b) * Real.sqrt (b / a) = 1) ∧
  (Real.sqrt (a * b) / Real.sqrt (a / b) = -b) := by
  sorry

end NUMINAMATH_CALUDE_expressions_correctness_l1634_163493


namespace NUMINAMATH_CALUDE_right_triangle_area_l1634_163453

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  angle = 45 * π / 180 →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1634_163453


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l1634_163484

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 40) 
  (h2 : s.downstream = 60) : 
  upstreamSpeed s = 20 := by
  sorry

#check upstream_speed_calculation

end NUMINAMATH_CALUDE_upstream_speed_calculation_l1634_163484


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l1634_163494

theorem other_solution_quadratic (x : ℚ) : 
  (48 * (3/4)^2 + 25 = 77 * (3/4) + 4) → 
  (48 * x^2 + 25 = 77 * x + 4) → 
  x = 3/4 ∨ x = 7/12 := by
sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l1634_163494


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1634_163446

/-- The quadratic equation 2qx^2 - 20x + 5 = 0 has only one solution when q = 10 -/
theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, 2 * q * x^2 - 20 * x + 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1634_163446


namespace NUMINAMATH_CALUDE_one_acute_triangle_in_1997_gon_l1634_163404

/-- A convex regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A decomposition of a polygon into triangles using non-intersecting diagonals -/
structure TriangularDecomposition (n : ℕ) where
  (polygon : RegularPolygon n)

/-- An acute triangle -/
structure AcuteTriangle

/-- The number of acute triangles in a triangular decomposition of a regular polygon -/
def num_acute_triangles (n : ℕ) (decomp : TriangularDecomposition n) : ℕ :=
  sorry

/-- The main theorem: In a regular 1997-gon, there is exactly one acute triangle
    in its triangular decomposition -/
theorem one_acute_triangle_in_1997_gon :
  ∀ (decomp : TriangularDecomposition 1997),
    num_acute_triangles 1997 decomp = 1 :=
  sorry

end NUMINAMATH_CALUDE_one_acute_triangle_in_1997_gon_l1634_163404


namespace NUMINAMATH_CALUDE_substitution_result_l1634_163459

theorem substitution_result (x y : ℝ) :
  (4 * x + 5 * y = 7) ∧ (y = 2 * x - 1) →
  4 * x + 10 * x - 5 = 7 := by
sorry

end NUMINAMATH_CALUDE_substitution_result_l1634_163459


namespace NUMINAMATH_CALUDE_area_code_count_l1634_163438

/-- The number of uppercase letters available -/
def uppercaseLetters : Nat := 26

/-- The number of lowercase letters and digits available for the second character -/
def secondCharOptions : Nat := 36

/-- The number of special characters available -/
def specialChars : Nat := 10

/-- The number of digits available -/
def digits : Nat := 10

/-- The total number of unique area codes that can be created -/
def totalAreaCodes : Nat := 
  (uppercaseLetters * secondCharOptions) + 
  (uppercaseLetters * secondCharOptions * specialChars) + 
  (uppercaseLetters * secondCharOptions * specialChars * digits)

theorem area_code_count : totalAreaCodes = 103896 := by
  sorry

end NUMINAMATH_CALUDE_area_code_count_l1634_163438


namespace NUMINAMATH_CALUDE_weight_equivalence_l1634_163447

/-- The weight ratio between small and large circles -/
def weight_ratio : ℚ := 2 / 5

/-- The number of small circles -/
def num_small_circles : ℕ := 15

/-- Theorem stating the equivalence in weight between small and large circles -/
theorem weight_equivalence :
  (num_small_circles : ℚ) * weight_ratio = 6 := by sorry

end NUMINAMATH_CALUDE_weight_equivalence_l1634_163447


namespace NUMINAMATH_CALUDE_trapezium_area_l1634_163417

/-- The area of a trapezium with given dimensions -/
theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 10) (hh : h = 10) :
  (a + b) * h / 2 = 150 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l1634_163417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1634_163414

theorem arithmetic_sequence_length (a₁ aₙ d : ℕ) (h : a₁ = 6) (h' : aₙ = 206) (h'' : d = 4) :
  (aₙ - a₁) / d + 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1634_163414


namespace NUMINAMATH_CALUDE_probability_white_ball_l1634_163440

/-- The probability of drawing a white ball from a bag with red, white, and black balls -/
theorem probability_white_ball (red white black : ℕ) (h : red = 5 ∧ white = 2 ∧ black = 3) :
  (white : ℚ) / (red + white + black : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l1634_163440


namespace NUMINAMATH_CALUDE_parabola_coefficients_l1634_163474

/-- A parabola with equation y = ax^2 + bx + c, vertex at (5, -1), 
    vertical axis of symmetry, and passing through (2, 8) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (a * 5^2 + b * 5 + c = -1) ∧
  (∀ x : ℝ, a * (x - 5)^2 + (a * 5^2 + b * 5 + c) = a * x^2 + b * x + c) ∧
  (a * 2^2 + b * 2 + c = 8)

/-- The values of a, b, and c for the given parabola are 1, -10, and 24 respectively -/
theorem parabola_coefficients : 
  ∃ a b c : ℝ, Parabola a b c ∧ a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l1634_163474


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l1634_163432

/-- A parabola with equation y^2 = 4x and two points on it -/
structure Parabola where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁^2 = 4 * x₁
  eq₂ : y₂^2 = 4 * x₂
  dist : (x₁ - x₂)^2 + (y₁ - y₂)^2 = 49  -- |AB|^2 = 7^2

/-- The distance from the midpoint of AB to the directrix of the parabola is 7/2 -/
theorem parabola_midpoint_to_directrix (p : Parabola) : 
  (p.x₁ + p.x₂) / 2 + 1 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_l1634_163432


namespace NUMINAMATH_CALUDE_distance_AB_is_600_l1634_163411

-- Define the cities
structure City where
  name : String

-- Define the travelers
structure Traveler where
  name : String
  start : City
  destination : City
  travelTime : ℝ
  averageSpeed : ℝ

-- Define the problem setup
def cityA : City := ⟨"A"⟩
def cityB : City := ⟨"B"⟩
def cityC : City := ⟨"C"⟩

def eddy : Traveler := ⟨"Eddy", cityA, cityB, 3, 2⟩
def freddy : Traveler := ⟨"Freddy", cityA, cityC, 3, 1⟩

-- Define the distances
def distanceAC : ℝ := 300

-- Theorem statement
theorem distance_AB_is_600 :
  let distanceAB := eddy.averageSpeed * eddy.travelTime
  distanceAC = freddy.averageSpeed * freddy.travelTime →
  eddy.averageSpeed = 2 * freddy.averageSpeed →
  distanceAB = 600 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_600_l1634_163411


namespace NUMINAMATH_CALUDE_prime_sum_difference_l1634_163416

theorem prime_sum_difference (m n p : ℕ) 
  (hm : Nat.Prime m) (hn : Nat.Prime n) (hp : Nat.Prime p)
  (h_pos : 0 < p ∧ 0 < n ∧ 0 < m)
  (h_order : m > n ∧ n > p)
  (h_sum : m + n + p = 74)
  (h_diff : m - n - p = 44) :
  m = 59 ∧ n = 13 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_difference_l1634_163416


namespace NUMINAMATH_CALUDE_largest_of_three_negatives_l1634_163424

theorem largest_of_three_negatives (a b c : ℝ) 
  (neg_a : a < 0) (neg_b : b < 0) (neg_c : c < 0)
  (h : c / (a + b) < a / (b + c) ∧ a / (b + c) < b / (c + a)) :
  c > a ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_largest_of_three_negatives_l1634_163424


namespace NUMINAMATH_CALUDE_smallest_y_value_l1634_163431

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 21 * z + 18 = z * (2 * z + 12) → y ≤ z) ∧
  (3 * y^2 + 21 * y + 18 = y * (2 * y + 12)) ∧
  y = -6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_value_l1634_163431


namespace NUMINAMATH_CALUDE_map_distance_conversion_l1634_163471

/-- Calculates the actual distance given map distance and scale --/
def actual_distance (map_distance : ℝ) (map_scale : ℝ) : ℝ :=
  map_distance * map_scale

/-- Theorem: Given a map scale where 312 inches represent 136 km,
    a distance of 25 inches on the map corresponds to approximately 10.897425 km
    in actual distance. --/
theorem map_distance_conversion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (actual_dist : ℝ),
    abs (actual_distance 25 (136 / 312) - actual_dist) < ε ∧
    abs (actual_dist - 10.897425) < ε :=
by sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l1634_163471


namespace NUMINAMATH_CALUDE_sum_possible_angles_l1634_163413

/-- An isosceles triangle with one angle of 80 degrees -/
structure IsoscelesTriangle80 where
  /-- The measure of one of the angles in degrees -/
  angle1 : ℝ
  /-- The measure of another angle in degrees -/
  angle2 : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One of the angles is 80 degrees -/
  has80Angle : angle1 = 80 ∨ angle2 = 80
  /-- The sum of all angles in a triangle is 180 degrees -/
  angleSum : angle1 + angle2 + (180 - angle1 - angle2) = 180

/-- The theorem to be proved -/
theorem sum_possible_angles (t : IsoscelesTriangle80) :
  ∃ (y1 y2 y3 : ℝ), (y1 = t.angle1 ∧ y1 ≠ 80) ∨ 
                    (y1 = t.angle2 ∧ y1 ≠ 80) ∨
                    (y1 = 180 - t.angle1 - t.angle2 ∧ y1 ≠ 80) ∧
                    (y2 = t.angle1 ∧ y2 ≠ 80) ∨ 
                    (y2 = t.angle2 ∧ y2 ≠ 80) ∨
                    (y2 = 180 - t.angle1 - t.angle2 ∧ y2 ≠ 80) ∧
                    (y3 = t.angle1 ∧ y3 ≠ 80) ∨ 
                    (y3 = t.angle2 ∧ y3 ≠ 80) ∨
                    (y3 = 180 - t.angle1 - t.angle2 ∧ y3 ≠ 80) ∧
                    y1 + y2 + y3 = 150 :=
  sorry

end NUMINAMATH_CALUDE_sum_possible_angles_l1634_163413


namespace NUMINAMATH_CALUDE_highest_power_of_three_N_l1634_163465

/-- Concatenates a list of integers into a single integer -/
def concatenate_integers (list : List Int) : Int :=
  sorry

/-- Generates a list of 2-digit integers from 73 to 29 in descending order -/
def generate_list : List Int :=
  sorry

/-- The number N formed by concatenating 2-digit integers from 73 to 29 in descending order -/
def N : Int := concatenate_integers generate_list

/-- The highest power of 3 that divides a given integer -/
def highest_power_of_three (n : Int) : Int :=
  sorry

theorem highest_power_of_three_N :
  highest_power_of_three N = 0 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_N_l1634_163465


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1634_163478

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 5*x > 22) → x ≤ -4 ∧ 7 - 5*(-4) > 22 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1634_163478


namespace NUMINAMATH_CALUDE_stellas_antique_shop_profit_l1634_163442

/-- Calculates the profit for Stella's antique shop given the inventory and prices --/
theorem stellas_antique_shop_profit :
  let dolls : ℕ := 6
  let clocks : ℕ := 4
  let glasses : ℕ := 8
  let vases : ℕ := 3
  let postcards : ℕ := 10
  let doll_price : ℕ := 8
  let clock_price : ℕ := 25
  let glass_price : ℕ := 6
  let vase_price : ℕ := 12
  let postcard_price : ℕ := 3
  let purchase_cost : ℕ := 250
  let revenue := dolls * doll_price + clocks * clock_price + glasses * glass_price + 
                 vases * vase_price + postcards * postcard_price
  let profit := revenue - purchase_cost
  profit = 12 := by sorry

end NUMINAMATH_CALUDE_stellas_antique_shop_profit_l1634_163442


namespace NUMINAMATH_CALUDE_trigonometric_equation_equivalence_l1634_163483

theorem trigonometric_equation_equivalence (α : ℝ) : 
  (1 - 2 * (Real.cos α) ^ 2) / (2 * Real.tan (2 * α - π / 4) * (Real.sin (π / 4 + 2 * α)) ^ 2) = 
  -(Real.cos (2 * α)) / ((Real.cos (2 * α - π / 4) + Real.sin (2 * α - π / 4)) ^ 2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_equivalence_l1634_163483


namespace NUMINAMATH_CALUDE_f_range_implies_a_range_l1634_163499

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x + 1| - |4*x + a|

-- State the theorem
theorem f_range_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ -5) → a ∈ Set.Iic (-4) ∪ Set.Ici 6 :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_a_range_l1634_163499


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1634_163457

theorem cos_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (α - π/4))
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.cos (2 * α) = Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1634_163457


namespace NUMINAMATH_CALUDE_cabbage_area_l1634_163486

theorem cabbage_area (garden_area_this_year garden_area_last_year : ℝ) 
  (cabbages_this_year cabbages_last_year : ℕ) :
  (garden_area_this_year = cabbages_this_year) →
  (garden_area_this_year = garden_area_last_year + 199) →
  (cabbages_this_year = 10000) →
  (∃ x y : ℝ, garden_area_last_year = x^2 ∧ garden_area_this_year = y^2) →
  (garden_area_this_year / cabbages_this_year = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cabbage_area_l1634_163486


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1634_163496

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_a4 : a 4 = 4)
  (h_a6 : a 6 = 16) :
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ+, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1634_163496


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1634_163426

theorem absolute_value_inequality (x : ℝ) : 
  (abs x + abs (abs x - 1) = 1) → (x + 1) * (x - 1) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1634_163426


namespace NUMINAMATH_CALUDE_mean_temperature_l1634_163490

theorem mean_temperature (temperatures : List ℝ) : 
  temperatures = [75, 77, 76, 80, 82] → 
  (temperatures.sum / temperatures.length : ℝ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1634_163490


namespace NUMINAMATH_CALUDE_book_club_snack_fee_l1634_163409

theorem book_club_snack_fee (members : ℕ) (hardcover_price paperback_price : ℚ)
  (hardcover_count paperback_count : ℕ) (total_collected : ℚ) :
  members = 6 →
  hardcover_price = 30 →
  paperback_price = 12 →
  hardcover_count = 6 →
  paperback_count = 6 →
  total_collected = 2412 →
  (total_collected - members * (hardcover_price * hardcover_count + paperback_price * paperback_count)) / members = 150 := by
  sorry

#check book_club_snack_fee

end NUMINAMATH_CALUDE_book_club_snack_fee_l1634_163409


namespace NUMINAMATH_CALUDE_shares_distribution_l1634_163449

/-- Proves that given the conditions, the shares of A, B, C, D, and E are 50, 100, 300, 150, and 600 respectively. -/
theorem shares_distribution (total : ℝ) (a b c d e : ℝ) 
  (h_total : total = 1200)
  (h_ab : a = (1/2) * b)
  (h_bc : b = (1/3) * c)
  (h_cd : c = 2 * d)
  (h_de : d = (1/4) * e)
  (h_sum : a + b + c + d + e = total) :
  a = 50 ∧ b = 100 ∧ c = 300 ∧ d = 150 ∧ e = 600 := by
  sorry

#check shares_distribution

end NUMINAMATH_CALUDE_shares_distribution_l1634_163449


namespace NUMINAMATH_CALUDE_quadrilateral_area_sum_l1634_163435

/-- Represents a quadrilateral PQRS -/
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

/-- Checks if a quadrilateral is convex -/
def is_convex (quad : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def area (quad : Quadrilateral) : ℝ := sorry

/-- Checks if a number has no perfect square factors greater than 1 -/
def no_perfect_square_factors (n : ℝ) : Prop := sorry

theorem quadrilateral_area_sum (quad : Quadrilateral) (a b c : ℝ) :
  is_convex quad →
  distance quad.P quad.Q = 7 →
  distance quad.Q quad.R = 3 →
  distance quad.R quad.S = 9 →
  distance quad.S quad.P = 9 →
  angle quad.R quad.S quad.P = π / 3 →
  ∃ (a b c : ℝ), area quad = Real.sqrt a + b * Real.sqrt c ∧
                  no_perfect_square_factors a ∧
                  no_perfect_square_factors c →
  a + b + c = 608.25 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_sum_l1634_163435


namespace NUMINAMATH_CALUDE_f_one_is_zero_five_zeros_symmetric_center_l1634_163454

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x - 1) = f (x + 1)
axiom decreasing_property : ∀ x₁ x₂, x₁ ∈ Set.Ioo 0 1 → x₂ ∈ Set.Ioo 0 1 → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

-- Theorem statements
theorem f_one_is_zero : f 1 = 0 := by sorry

theorem five_zeros : 
  f (-2) = 0 ∧ f (-1) = 0 ∧ f 0 = 0 ∧ f 1 = 0 ∧ f 2 = 0 := by sorry

theorem symmetric_center : 
  ∀ x, f (2014 + x) = -f (2014 - x) := by sorry

end NUMINAMATH_CALUDE_f_one_is_zero_five_zeros_symmetric_center_l1634_163454


namespace NUMINAMATH_CALUDE_root_shift_polynomial_l1634_163400

theorem root_shift_polynomial (a b c : ℂ) : 
  (a^3 - 6*a^2 + 11*a - 6 = 0) ∧ 
  (b^3 - 6*b^2 + 11*b - 6 = 0) ∧ 
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  ((a - 3)^3 + 3*(a - 3)^2 + 2*(a - 3) = 0) ∧
  ((b - 3)^3 + 3*(b - 3)^2 + 2*(b - 3) = 0) ∧
  ((c - 3)^3 + 3*(c - 3)^2 + 2*(c - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_polynomial_l1634_163400


namespace NUMINAMATH_CALUDE_joan_next_birthday_age_l1634_163460

theorem joan_next_birthday_age
  (joan larry kim : ℝ)
  (h1 : joan = 1.3 * larry)
  (h2 : larry = 0.75 * kim)
  (h3 : joan + larry + kim = 39)
  : ⌊joan⌋ + 1 = 15 :=
sorry

end NUMINAMATH_CALUDE_joan_next_birthday_age_l1634_163460


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1634_163429

theorem geometric_sequence_sum (a r : ℝ) (h1 : a + a*r + a*r^2 = 13) (h2 : a * (1 - r^7) / (1 - r) = 183) : 
  ∃ (ε : ℝ), abs (a + a*r + a*r^2 + a*r^3 + a*r^4 - 75.764) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1634_163429


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l1634_163408

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y ∈ {y : ℝ | 0 < y ∧ y < x}, ¬(Real.sin (2*y) * Real.sin (3*y) = Real.cos (2*y) * Real.cos (3*y))) ∧
  (Real.sin (2*x) * Real.sin (3*x) = Real.cos (2*x) * Real.cos (3*x)) ∧
  (x * (180 / Real.pi) = 18) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l1634_163408


namespace NUMINAMATH_CALUDE_sine_tangent_inequality_l1634_163461

theorem sine_tangent_inequality (x y : ℝ) : 
  (0 < Real.sin (50 * π / 180) ∧ Real.sin (50 * π / 180) < 1) →
  Real.tan (50 * π / 180) > 1 →
  (Real.sin (50 * π / 180))^x - (Real.tan (50 * π / 180))^x ≤ 
  (Real.sin (50 * π / 180))^(-y) - (Real.tan (50 * π / 180))^(-y) →
  x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_tangent_inequality_l1634_163461


namespace NUMINAMATH_CALUDE_toy_box_paths_l1634_163481

/-- Represents a rectangular grid --/
structure Grid :=
  (length : ℕ)
  (width : ℕ)

/-- Calculates the number of paths in a grid from one corner to the opposite corner,
    moving only right and up, covering a specific total distance --/
def numPaths (g : Grid) (totalDistance : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that for a 50x40 grid with total distance 90,
    there are 12 possible paths --/
theorem toy_box_paths :
  let g : Grid := { length := 50, width := 40 }
  numPaths g 90 = 12 := by
  sorry

end NUMINAMATH_CALUDE_toy_box_paths_l1634_163481


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_2022_l1634_163423

def unit_digit (n : ℕ) : ℕ := n % 10

def power_of_3_unit_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem unit_digit_of_3_to_2022 :
  unit_digit (3^2022) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_2022_l1634_163423


namespace NUMINAMATH_CALUDE_neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l1634_163477

-- Define the color type
inductive Color : Type
  | Black : Color
  | Red : Color
  | White : Color

-- Define the coloring function
def coloring : ℤ → Color := sorry

-- Define the coloring rules
axiom neg_black_is_white : ∀ n : ℤ, coloring n = Color.Black → coloring (-n) = Color.White
axiom sum_white_is_black : ∀ a b : ℤ, coloring a = Color.White → coloring b = Color.White → coloring (a + b) = Color.Black

-- Theorems to prove
theorem neg_white_is_black : ∀ n : ℤ, coloring n = Color.White → coloring (-n) = Color.Black := sorry

theorem sum_black_is_white : ∀ a b : ℤ, coloring a = Color.Black → coloring b = Color.Black → coloring (a + b) = Color.White := sorry

theorem zero_is_red : coloring 0 = Color.Red := sorry

theorem nonzero_black_or_white : ∀ n : ℤ, n ≠ 0 → (coloring n = Color.Black ∨ coloring n = Color.White) := sorry

theorem neg_opposite_color : ∀ n : ℤ, n ≠ 0 → 
  (coloring n = Color.Black → coloring (-n) = Color.White) ∧ 
  (coloring n = Color.White → coloring (-n) = Color.Black) := sorry

end NUMINAMATH_CALUDE_neg_white_is_black_sum_black_is_white_zero_is_red_nonzero_black_or_white_neg_opposite_color_l1634_163477


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1634_163407

theorem min_value_quadratic_sum (x y s : ℝ) (h : x + y = s) :
  (∀ a b : ℝ, a + b = s → 3 * a^2 + 2 * b^2 ≥ (6/5) * s^2) ∧
  ∃ x₀ y₀ : ℝ, x₀ + y₀ = s ∧ 3 * x₀^2 + 2 * y₀^2 = (6/5) * s^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1634_163407


namespace NUMINAMATH_CALUDE_number_of_students_l1634_163421

theorem number_of_students (x : ℕ) 
  (h1 : 3600 = (3600 / x) * x)  -- Retail price for x tools
  (h2 : 3600 = (3600 / (x + 60)) * (x + 60))  -- Wholesale price for x + 60 tools
  (h3 : (3600 / x) * 50 = (3600 / (x + 60)) * 60)  -- Cost equality condition
  : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1634_163421


namespace NUMINAMATH_CALUDE_smallest_possible_d_l1634_163427

theorem smallest_possible_d (c d : ℝ) : 
  (2 < c) → 
  (c < d) → 
  (2 + c ≤ d) → 
  (2/c + 2/d ≤ 2) → 
  d ≥ 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l1634_163427


namespace NUMINAMATH_CALUDE_max_subsets_100_l1634_163434

/-- Given a set S of n elements, f(n) returns the maximum number of non-empty subsets
    that can be chosen from S such that any two chosen subsets are either disjoint
    or one contains the other. -/
def max_subsets (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem stating that for a set of 100 elements, the maximum number of non-empty subsets
    that can be chosen, such that any two chosen subsets are either disjoint or one contains
    the other, is 199. -/
theorem max_subsets_100 : max_subsets 100 = 199 := by
  sorry

end NUMINAMATH_CALUDE_max_subsets_100_l1634_163434


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l1634_163401

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- State the theorem
theorem tangent_line_at_origin (a : ℝ) 
  (h : ∀ x, f' a x = f' a (-x)) : 
  ∃ m : ℝ, m = -2 ∧ ∀ x, f a x = m * x + f a 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l1634_163401


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l1634_163428

theorem cube_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  6 * (x^3 + y^3 + z^3)^2 ≤ (x^2 + y^2 + z^2)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l1634_163428


namespace NUMINAMATH_CALUDE_trailingZeros_100_factorial_l1634_163441

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_100_factorial_l1634_163441


namespace NUMINAMATH_CALUDE_other_person_age_l1634_163469

/-- Given two people where one (Marco) is 1 year older than twice the age of the other,
    and the sum of their ages is 37, prove that the younger person is 12 years old. -/
theorem other_person_age (x : ℕ) : x + (2 * x + 1) = 37 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_person_age_l1634_163469


namespace NUMINAMATH_CALUDE_S_has_maximum_l1634_163476

def S (n : ℕ+) : ℤ := -2 * n.val ^ 3 + 21 * n.val ^ 2 + 23 * n.val

theorem S_has_maximum : ∃ (m : ℕ+), ∀ (n : ℕ+), S n ≤ S m ∧ S m = 504 := by
  sorry

end NUMINAMATH_CALUDE_S_has_maximum_l1634_163476


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1634_163492

/-- Given two parallel 2D vectors a and b, where a = (2, 3) and b = (x, 6), prove that x = 4. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1634_163492


namespace NUMINAMATH_CALUDE_min_value_abc_l1634_163422

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l1634_163422


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_values_solution_set_correct_l1634_163482

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values (a : ℝ) :
  M ∩ N a = ∅ → a = -6 ∨ a = -2 := by
  sorry

-- Define the solution set
def solution_set : Set ℝ := {-6, -2}

-- State the theorem for the solution set
theorem solution_set_correct :
  ∀ a : ℝ, (M ∩ N a = ∅) → a ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_values_solution_set_correct_l1634_163482


namespace NUMINAMATH_CALUDE_pool_width_l1634_163437

theorem pool_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 30 ∧ length = 10 ∧ area = length * width → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_pool_width_l1634_163437


namespace NUMINAMATH_CALUDE_function_properties_l1634_163430

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≠ 0

/-- The quadratic function we're considering -/
def f (m n : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + n * x + 3 * m + n

/-- The theorem stating the properties of the function and its maximum value -/
theorem function_properties :
  ∃ (m n : ℝ),
    EvenFunction (f m n) ∧
    Domain (f m n) (m - 1) (2 * m) ∧
    m = 1/3 ∧
    n = 0 ∧
    (∀ x, m - 1 ≤ x ∧ x ≤ 2 * m → f m n x ≤ 31/27) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1634_163430


namespace NUMINAMATH_CALUDE_angle_cde_is_eleven_degrees_l1634_163462

/-- Given a configuration in a rectangle where:
    - Angle ACB = 80°
    - Angle FEG = 64°
    - Angle DCE = 86°
    - Angle DEC = 83°
    Prove that angle CDE (θ) is equal to 11°. -/
theorem angle_cde_is_eleven_degrees 
  (angle_ACB : ℝ) (angle_FEG : ℝ) (angle_DCE : ℝ) (angle_DEC : ℝ)
  (h1 : angle_ACB = 80)
  (h2 : angle_FEG = 64)
  (h3 : angle_DCE = 86)
  (h4 : angle_DEC = 83) :
  180 - angle_DCE - angle_DEC = 11 := by
  sorry

end NUMINAMATH_CALUDE_angle_cde_is_eleven_degrees_l1634_163462


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1634_163498

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1634_163498


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l1634_163480

theorem mystery_book_shelves (total_books : ℕ) (books_per_shelf : ℕ) (picture_shelves : ℕ) :
  total_books = 72 →
  books_per_shelf = 9 →
  picture_shelves = 5 →
  (total_books - picture_shelves * books_per_shelf) / books_per_shelf = 3 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l1634_163480


namespace NUMINAMATH_CALUDE_ball_travel_distance_l1634_163436

/-- Represents an elliptical billiard table -/
structure EllipticalTable where
  majorAxis : ℝ
  focalDistance : ℝ

/-- Possible distances traveled by a ball on an elliptical table -/
def possibleDistances (table : EllipticalTable) : Set ℝ :=
  {4, 3, 1}

/-- Theorem: The distance traveled by a ball on a specific elliptical table -/
theorem ball_travel_distance (table : EllipticalTable) 
  (h1 : table.majorAxis = 2)
  (h2 : table.focalDistance = 1) :
  ∃ d ∈ possibleDistances table, d = 4 ∨ d = 3 ∨ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_ball_travel_distance_l1634_163436


namespace NUMINAMATH_CALUDE_inequality_proof_l1634_163463

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1634_163463


namespace NUMINAMATH_CALUDE_largest_divisor_with_remainders_l1634_163410

theorem largest_divisor_with_remainders : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℤ), 69 = k * n + 5) ∧ 
  (∃ (l : ℤ), 86 = l * n + 6) ∧ 
  (∀ (m : ℕ), m > n → 
    (¬∃ (k : ℤ), 69 = k * m + 5) ∨ 
    (¬∃ (l : ℤ), 86 = l * m + 6)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_with_remainders_l1634_163410


namespace NUMINAMATH_CALUDE_principal_is_2000_l1634_163448

/-- Given an interest rate, time period, and total interest, 
    calculates the principal amount borrowed. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, 
    the principal amount borrowed is 2000. -/
theorem principal_is_2000 : 
  let rate : ℚ := 5
  let time : ℕ := 13
  let interest : ℚ := 1300
  calculate_principal rate time interest = 2000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_2000_l1634_163448


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1634_163452

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (Real.sqrt 2 / 2) ∧ x ≠ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1634_163452


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1634_163420

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y) ∧ f = fun x ↦ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1634_163420


namespace NUMINAMATH_CALUDE_square_perimeter_l1634_163473

/-- Given a square with area 720 square meters, its perimeter is 48√5 meters. -/
theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 720 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 48 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1634_163473


namespace NUMINAMATH_CALUDE_class_ratio_proof_l1634_163470

theorem class_ratio_proof (B G : ℝ) 
  (h1 : B > 0) 
  (h2 : G > 0) 
  (h3 : 0.80 * B + 0.75 * G = 0.78 * (B + G)) : 
  B / G = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_proof_l1634_163470


namespace NUMINAMATH_CALUDE_parking_theorem_l1634_163451

/-- The number of ways to park 5 trains on 5 tracks with one restriction -/
def parking_arrangements (n : ℕ) (restricted_train : ℕ) (restricted_track : ℕ) : ℕ :=
  (n - 1) * Nat.factorial (n - 1)

theorem parking_theorem :
  parking_arrangements 5 1 1 = 96 :=
by sorry

end NUMINAMATH_CALUDE_parking_theorem_l1634_163451


namespace NUMINAMATH_CALUDE_chess_game_probability_l1634_163403

theorem chess_game_probability (p_win p_not_lose : ℝ) :
  p_win = 0.3 → p_not_lose = 0.8 → p_win + (p_not_lose - p_win) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l1634_163403


namespace NUMINAMATH_CALUDE_profit_increase_may_to_june_l1634_163406

theorem profit_increase_may_to_june
  (march_to_april : Real)
  (april_to_may : Real)
  (march_to_june : Real)
  (h1 : march_to_april = 0.30)
  (h2 : april_to_may = -0.20)
  (h3 : march_to_june = 0.5600000000000001)
  : ∃ may_to_june : Real,
    (1 + march_to_april) * (1 + april_to_may) * (1 + may_to_june) = 1 + march_to_june ∧
    may_to_june = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_may_to_june_l1634_163406


namespace NUMINAMATH_CALUDE_mary_warmth_hours_l1634_163455

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : Nat
  table : Nat
  cabinet : Nat
  stool : Nat

/-- Represents the quantity of each furniture type Mary chops -/
structure ChoppedFurniture where
  chairs : Nat
  tables : Nat
  cabinets : Nat
  stools : Nat

/-- Calculates the total number of sticks of wood produced -/
def totalWood (fw : FurnitureWood) (cf : ChoppedFurniture) : Nat :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.cabinet * cf.cabinets + fw.stool * cf.stools

/-- Theorem stating that Mary can keep warm for 64 hours with the chopped firewood -/
theorem mary_warmth_hours (fw : FurnitureWood) (cf : ChoppedFurniture) (sticksPerHour : Nat) :
  fw.chair = 8 →
  fw.table = 12 →
  fw.cabinet = 16 →
  fw.stool = 3 →
  cf.chairs = 25 →
  cf.tables = 12 →
  cf.cabinets = 5 →
  cf.stools = 8 →
  sticksPerHour = 7 →
  totalWood fw cf / sticksPerHour = 64 := by
  sorry

#check mary_warmth_hours

end NUMINAMATH_CALUDE_mary_warmth_hours_l1634_163455


namespace NUMINAMATH_CALUDE_max_value_a4a7_l1634_163487

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The maximum value of a_4 * a_7 in an arithmetic sequence where a_6 = 4 -/
theorem max_value_a4a7 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h6 : a 6 = 4) :
  (∀ d : ℝ, a 4 * a 7 ≤ 18) ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_max_value_a4a7_l1634_163487


namespace NUMINAMATH_CALUDE_min_cuts_to_touch_coin_l1634_163468

/-- Represents a circular object with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a straight cut on the pancake -/
structure Cut where
  width : ℝ

/-- The pancake -/
def pancake : Circle := { radius := 10 }

/-- The coin -/
def coin : Circle := { radius := 1 }

/-- The width of the area covered by a single cut -/
def cut_width : ℝ := 2

/-- The minimum number of cuts needed -/
def min_cuts : ℕ := 10

theorem min_cuts_to_touch_coin : 
  ∀ (cuts : ℕ), 
    cuts < min_cuts → 
    ∃ (coin_position : ℝ × ℝ), 
      coin_position.1^2 + coin_position.2^2 ≤ pancake.radius^2 ∧ 
      ∀ (cut : Cut), cut.width = cut_width → 
        ∃ (d : ℝ), d > coin.radius ∧ 
          ∀ (p : ℝ × ℝ), p.1^2 + p.2^2 ≤ coin.radius^2 → 
            (p.1 - coin_position.1)^2 + (p.2 - coin_position.2)^2 ≤ d^2 := by
  sorry

#check min_cuts_to_touch_coin

end NUMINAMATH_CALUDE_min_cuts_to_touch_coin_l1634_163468


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1634_163475

theorem a_gt_one_sufficient_not_necessary_for_a_gt_zero :
  (∃ a : ℝ, a > 0 ∧ ¬(a > 1)) ∧
  (∀ a : ℝ, a > 1 → a > 0) :=
sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1634_163475


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1634_163472

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x + y = 1 → x * y ≤ 1/4) ∧
  (∃ x y : ℝ, x * y ≤ 1/4 ∧ x + y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1634_163472


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocals_equals_sqrt1111_over_112_l1634_163464

theorem sqrt_sum_reciprocals_equals_sqrt1111_over_112 :
  Real.sqrt (1 / 25 + 1 / 36 + 1 / 49) = Real.sqrt 1111 / 112 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocals_equals_sqrt1111_over_112_l1634_163464


namespace NUMINAMATH_CALUDE_circle_equal_circumference_area_l1634_163489

theorem circle_equal_circumference_area (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equal_circumference_area_l1634_163489


namespace NUMINAMATH_CALUDE_family_income_problem_l1634_163456

theorem family_income_problem (initial_members : ℕ) (deceased_income new_average : ℚ) 
  (h1 : initial_members = 4)
  (h2 : deceased_income = 1170)
  (h3 : new_average = 590) :
  let initial_average := (initial_members * new_average + deceased_income) / initial_members
  initial_average = 735 := by
sorry

end NUMINAMATH_CALUDE_family_income_problem_l1634_163456


namespace NUMINAMATH_CALUDE_correlation_relationships_l1634_163488

-- Define the types of relationships
inductive Relationship
  | PointCoordinate
  | AppleYieldClimate
  | TreeDiameterHeight
  | StudentID

-- Define a function to determine if a relationship involves correlation
def involvesCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleYieldClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

-- Theorem statement
theorem correlation_relationships :
  (involvesCorrelation Relationship.PointCoordinate = False) ∧
  (involvesCorrelation Relationship.AppleYieldClimate = True) ∧
  (involvesCorrelation Relationship.TreeDiameterHeight = True) ∧
  (involvesCorrelation Relationship.StudentID = False) :=
sorry

end NUMINAMATH_CALUDE_correlation_relationships_l1634_163488


namespace NUMINAMATH_CALUDE_inequality_proof_l1634_163443

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) (h4 : a + b + c = 3) : 
  a * b^2 + b * c^2 + c * a^2 ≤ 27/8 ∧ 
  (a * b^2 + b * c^2 + c * a^2 = 27/8 ↔ a = 3/2 ∧ b = 3/2 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1634_163443


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1634_163466

theorem geometric_arithmetic_progression_sum : ∃ x y : ℝ, 
  (5 < x ∧ x < y ∧ y < 12) ∧ 
  (∃ r : ℝ, r > 1 ∧ x = 5 * r ∧ y = 5 * r^2) ∧
  (∃ d : ℝ, d > 0 ∧ y = x + d ∧ 12 = y + d) ∧
  (abs (x + y - 16.2788) < 0.0001) := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l1634_163466


namespace NUMINAMATH_CALUDE_regular_nonagon_perimeter_l1634_163485

/-- A regular polygon with 9 sides, each 2 centimeters long -/
structure RegularNonagon where
  side_length : ℝ
  num_sides : ℕ
  h1 : side_length = 2
  h2 : num_sides = 9

/-- The perimeter of a regular nonagon -/
def perimeter (n : RegularNonagon) : ℝ :=
  n.side_length * n.num_sides

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem regular_nonagon_perimeter (n : RegularNonagon) : perimeter n = 18 := by
  sorry

#check regular_nonagon_perimeter

end NUMINAMATH_CALUDE_regular_nonagon_perimeter_l1634_163485


namespace NUMINAMATH_CALUDE_base_b_not_perfect_square_l1634_163425

theorem base_b_not_perfect_square (b : ℕ) (h : b ≥ 3) :
  ¬∃ (n : ℕ), 2 * b^2 + 2 * b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_perfect_square_l1634_163425


namespace NUMINAMATH_CALUDE_mikes_video_games_l1634_163419

theorem mikes_video_games (non_working : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  non_working = 9 → price_per_game = 5 → total_earnings = 30 →
  non_working + (total_earnings / price_per_game) = 15 :=
by sorry

end NUMINAMATH_CALUDE_mikes_video_games_l1634_163419


namespace NUMINAMATH_CALUDE_min_trees_for_three_types_l1634_163433

/-- Represents the four types of trees in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove where
  trees : Finset TreeType
  total_count : ℕ
  four_types_in_85 : ∀ (subset : Finset TreeType), subset.card = 85 → (∀ t : TreeType, t ∈ subset)

/-- The theorem to be proved -/
theorem min_trees_for_three_types (g : Grove) (h1 : g.total_count = 100) :
  ∃ (n : ℕ), n = 69 ∧ 
  (∀ (subset : Finset TreeType), subset.card ≥ n → 
    ∃ (t1 t2 t3 : TreeType), t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset) ∧
  (∃ (subset : Finset TreeType), subset.card = n - 1 ∧ 
    ∀ (t1 t2 t3 : TreeType), t1 ≠ t2 → t1 ≠ t3 → t2 ≠ t3 → 
      ¬(t1 ∈ subset ∧ t2 ∈ subset ∧ t3 ∈ subset)) :=
by sorry

end NUMINAMATH_CALUDE_min_trees_for_three_types_l1634_163433


namespace NUMINAMATH_CALUDE_fraction_calculation_l1634_163444

theorem fraction_calculation (w x y : ℝ) 
  (h1 : w / y = 1 / 5)
  (h2 : (x + y) / y = 2.2) :
  w / x = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1634_163444


namespace NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l1634_163445

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The average point of a dataset -/
structure AveragePoint where
  x : ℝ
  y : ℝ

/-- Theorem: Two regression lines with the same average point intersect at that point -/
theorem regression_lines_intersect_at_average_point 
  (l₁ l₂ : RegressionLine) 
  (avg : AveragePoint) : 
  (avg.x * l₁.slope + l₁.intercept = avg.y) ∧ 
  (avg.x * l₂.slope + l₂.intercept = avg.y) := by
  sorry

#check regression_lines_intersect_at_average_point

end NUMINAMATH_CALUDE_regression_lines_intersect_at_average_point_l1634_163445


namespace NUMINAMATH_CALUDE_equation_solution_l1634_163467

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : (7*x - 4) / (x - 2) = 5 / (x - 2)) : x = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1634_163467


namespace NUMINAMATH_CALUDE_routes_from_bristol_to_carlisle_l1634_163412

/-- The number of routes from Bristol to Birmingham -/
def bristol_to_birmingham : ℕ := 6

/-- The number of routes from Birmingham to Sheffield -/
def birmingham_to_sheffield : ℕ := 3

/-- The number of routes from Sheffield to Carlisle -/
def sheffield_to_carlisle : ℕ := 2

/-- The total number of routes from Bristol to Carlisle -/
def total_routes : ℕ := bristol_to_birmingham * birmingham_to_sheffield * sheffield_to_carlisle

theorem routes_from_bristol_to_carlisle : total_routes = 36 := by
  sorry

end NUMINAMATH_CALUDE_routes_from_bristol_to_carlisle_l1634_163412


namespace NUMINAMATH_CALUDE_bus_cyclist_speeds_l1634_163405

/-- The speed of the buses in km/h -/
def bus_speed : ℝ := 42

/-- The speed of the cyclist in km/h -/
def cyclist_speed : ℝ := 18

/-- The distance between points A and B in km -/
def distance : ℝ := 37

/-- The time in minutes from the start of the first bus to meeting the cyclist -/
def time_bus1_to_meeting : ℝ := 40

/-- The time in minutes from the start of the second bus to meeting the cyclist -/
def time_bus2_to_meeting : ℝ := 31

/-- The time in minutes from the start of the cyclist to meeting the first bus -/
def time_cyclist_to_bus1 : ℝ := 30

/-- The time in minutes from the start of the cyclist to meeting the second bus -/
def time_cyclist_to_bus2 : ℝ := 51

theorem bus_cyclist_speeds : 
  bus_speed * (time_bus1_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus1 / 60) = distance ∧
  bus_speed * (time_bus2_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus2 / 60) = distance :=
by sorry

end NUMINAMATH_CALUDE_bus_cyclist_speeds_l1634_163405


namespace NUMINAMATH_CALUDE_beta_conditions_l1634_163495

theorem beta_conditions (β : ℂ) (h1 : β ≠ -1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  Complex.abs (β^3 + 1) = 3 ∧ Complex.abs (β^6 + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_beta_conditions_l1634_163495
