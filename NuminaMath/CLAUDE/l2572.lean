import Mathlib

namespace NUMINAMATH_CALUDE_janet_ticket_count_l2572_257298

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
                  (roller_coaster_rides : ℕ) (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides

/-- Proof that Janet needs 47 tickets for her planned rides -/
theorem janet_ticket_count : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_janet_ticket_count_l2572_257298


namespace NUMINAMATH_CALUDE_total_arc_length_is_900_l2572_257215

/-- A triangle with its circumcircle -/
structure CircumscribedTriangle where
  /-- The radius of the circumcircle -/
  radius : ℝ
  /-- The perimeter of the triangle -/
  perimeter : ℝ

/-- The total length of arcs XX', YY', and ZZ' in a circumscribed triangle -/
def total_arc_length (t : CircumscribedTriangle) : ℝ := sorry

/-- Theorem: The total length of arcs XX', YY', and ZZ' is 900° -/
theorem total_arc_length_is_900 (t : CircumscribedTriangle) 
  (h1 : t.radius = 5) 
  (h2 : t.perimeter = 24) : 
  total_arc_length t = 900 := by sorry

end NUMINAMATH_CALUDE_total_arc_length_is_900_l2572_257215


namespace NUMINAMATH_CALUDE_combination_sum_equals_84_l2572_257279

theorem combination_sum_equals_84 : Nat.choose 8 2 + Nat.choose 8 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_84_l2572_257279


namespace NUMINAMATH_CALUDE_three_integers_problem_l2572_257237

theorem three_integers_problem :
  ∃ (x y z : ℤ),
    (x + y) / 2 + z = 42 ∧
    (y + z) / 2 + x = 13 ∧
    (x + z) / 2 + y = 37 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_problem_l2572_257237


namespace NUMINAMATH_CALUDE_triangle_side_length_l2572_257214

/-- Given a triangle ABC with area √3, angle B = 60°, and a² + c² = 3ac, prove that the length of side b is 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- Angle B is 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Side length b is 2√2
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2572_257214


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l2572_257291

/-- Represents a school with a certain number of male and female teachers --/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The total number of teachers in a school --/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- The schools in the problem --/
def school_A : School := { male_teachers := 2, female_teachers := 1 }
def school_B : School := { male_teachers := 1, female_teachers := 2 }

/-- The total number of teachers in both schools --/
def total_teachers : ℕ := school_A.total_teachers + school_B.total_teachers

/-- Theorem for the probability of selecting two teachers of the same gender --/
theorem same_gender_probability :
  (school_A.male_teachers * school_B.male_teachers + school_A.female_teachers * school_B.female_teachers) / 
  (school_A.total_teachers * school_B.total_teachers) = 4 / 9 := by sorry

/-- Theorem for the probability of selecting two teachers from the same school --/
theorem same_school_probability :
  (school_A.total_teachers * (school_A.total_teachers - 1) + school_B.total_teachers * (school_B.total_teachers - 1)) / 
  (total_teachers * (total_teachers - 1)) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l2572_257291


namespace NUMINAMATH_CALUDE_equation_solution_l2572_257235

theorem equation_solution : ∃ x : ℚ, 300 * 2 + (12 + 4) * x / 8 = 602 :=
  by
    use 1
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2572_257235


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2572_257234

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2572_257234


namespace NUMINAMATH_CALUDE_area_of_S_l2572_257272

-- Define the set S
def S : Set (ℝ × ℝ) := {(a, b) | ∀ x, x^2 + 2*b*x + 1 ≠ 2*a*(x + b)}

-- State the theorem
theorem area_of_S : MeasureTheory.volume S = π := by sorry

end NUMINAMATH_CALUDE_area_of_S_l2572_257272


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2572_257240

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3) ∧
  (∃ a b : ℝ, a + b > 3 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2572_257240


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2572_257266

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2572_257266


namespace NUMINAMATH_CALUDE_number_difference_l2572_257255

theorem number_difference (x y : ℤ) (h1 : x + y = 62) (h2 : y = 25) : |x - y| = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2572_257255


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l2572_257202

theorem addition_preserves_inequality (a b c d : ℝ) : 
  a < b → c < d → a + c < b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l2572_257202


namespace NUMINAMATH_CALUDE_sum_absolute_value_l2572_257295

theorem sum_absolute_value (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : x₁ + 1 = x₂ + 2 ∧ x₂ + 2 = x₃ + 3 ∧ x₃ + 3 = x₄ + 4 ∧ x₄ + 4 = x₅ + 5 ∧ 
       x₅ + 5 = x₁ + x₂ + x₃ + x₄ + x₅ + 6) : 
  |x₁ + x₂ + x₃ + x₄ + x₅| = 3.75 := by
sorry

end NUMINAMATH_CALUDE_sum_absolute_value_l2572_257295


namespace NUMINAMATH_CALUDE_sugar_recipe_problem_l2572_257232

/-- The number of recipes that can be accommodated given a certain amount of sugar and recipe requirement -/
def recipes_accommodated (total_sugar : ℚ) (sugar_per_recipe : ℚ) : ℚ :=
  total_sugar / sugar_per_recipe

/-- The problem statement -/
theorem sugar_recipe_problem :
  let total_sugar : ℚ := 56 / 3  -- 18⅔ cups
  let sugar_per_recipe : ℚ := 3 / 2  -- 1½ cups
  recipes_accommodated total_sugar sugar_per_recipe = 112 / 9 :=
by
  sorry

#eval (112 : ℚ) / 9  -- Should output 12⁴⁄₉

end NUMINAMATH_CALUDE_sugar_recipe_problem_l2572_257232


namespace NUMINAMATH_CALUDE_total_pieces_is_11403_l2572_257282

/-- Calculates the total number of pieces in John's puzzles -/
def totalPuzzlePieces : ℕ :=
  let puzzle1 : ℕ := 1000
  let puzzle2 : ℕ := puzzle1 + (puzzle1 * 20 / 100)
  let puzzle3 : ℕ := puzzle2 + (puzzle2 * 50 / 100)
  let puzzle4 : ℕ := puzzle3 + (puzzle3 * 75 / 100)
  let puzzle5 : ℕ := puzzle4 + (puzzle4 * 35 / 100)
  puzzle1 + puzzle2 + puzzle3 + puzzle4 + puzzle5

theorem total_pieces_is_11403 : totalPuzzlePieces = 11403 := by
  sorry

end NUMINAMATH_CALUDE_total_pieces_is_11403_l2572_257282


namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l2572_257242

/-- Given a line y = b - 2x where 0 < b < 6, intersecting the y-axis at P and the line x=6 at S,
    if the ratio of the area of triangle QRS to the area of triangle QOP is 4:9,
    then b = √(1296/11). -/
theorem line_intersection_area_ratio (b : ℝ) : 
  0 < b → b < 6 → 
  let line := fun x => b - 2 * x
  let P := (0, b)
  let S := (6, line 6)
  let Q := (b / 2, 0)
  let R := (6, 0)
  let area_QOP := (1 / 2) * (b / 2) * b
  let area_QRS := (1 / 2) * (6 - b / 2) * |b - 12|
  area_QRS / area_QOP = 4 / 9 →
  b = Real.sqrt (1296 / 11) := by
sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l2572_257242


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_expression_l2572_257287

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (k : ℕ) : ℚ :=
  let n : ℕ := 2 * k - 1
  let a₁ : ℚ := k^2 - 1
  let d : ℚ := 1
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating the sum of the arithmetic sequence equals the given expression -/
theorem arithmetic_sum_equals_expression (k : ℕ) :
  arithmetic_sum k = 2 * k^3 + k^2 - 4 * k + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_expression_l2572_257287


namespace NUMINAMATH_CALUDE_smallest_zero_201_l2572_257276

/-- A sequence defined by the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 134
  | 1 => 150
  | (k + 2) => a k - (k + 1) / a (k + 1)

/-- The property that a_n = 0 -/
def sequence_zero (n : ℕ) : Prop := a n = 0

/-- Theorem stating that 201 is the smallest positive integer n for which a_n = 0 -/
theorem smallest_zero_201 : 
  (∀ m : ℕ, m < 201 → ¬ sequence_zero m) ∧ sequence_zero 201 := by sorry

end NUMINAMATH_CALUDE_smallest_zero_201_l2572_257276


namespace NUMINAMATH_CALUDE_fraction_simplification_l2572_257206

theorem fraction_simplification :
  (1 - 1/3) / (1 - 1/2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2572_257206


namespace NUMINAMATH_CALUDE_hex_conversion_sum_l2572_257239

/-- Converts a hexadecimal number to decimal --/
def hex_to_decimal (hex : String) : ℕ := sorry

/-- Converts a decimal number to radix 7 --/
def decimal_to_radix7 (n : ℕ) : String := sorry

/-- Converts a radix 7 number to decimal --/
def radix7_to_decimal (r7 : String) : ℕ := sorry

/-- Converts a decimal number to hexadecimal --/
def decimal_to_hex (n : ℕ) : String := sorry

/-- Adds two hexadecimal numbers and returns the result in hexadecimal --/
def add_hex (hex1 : String) (hex2 : String) : String := sorry

theorem hex_conversion_sum :
  let initial_hex := "E78"
  let decimal := hex_to_decimal initial_hex
  let radix7 := decimal_to_radix7 decimal
  let back_to_decimal := radix7_to_decimal radix7
  let final_hex := decimal_to_hex back_to_decimal
  add_hex initial_hex final_hex = "1CF0" := by sorry

end NUMINAMATH_CALUDE_hex_conversion_sum_l2572_257239


namespace NUMINAMATH_CALUDE_expression_result_l2572_257293

theorem expression_result : 
  (7899665 : ℝ) - 12 * 3 * 2 + (7^3) / Real.sqrt 144 = 7899621.5833 := by
sorry

end NUMINAMATH_CALUDE_expression_result_l2572_257293


namespace NUMINAMATH_CALUDE_book_price_calculation_l2572_257211

theorem book_price_calculation (P : ℝ) : 
  P * 0.85 * 1.40 = 476 → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_book_price_calculation_l2572_257211


namespace NUMINAMATH_CALUDE_felix_tree_chopping_l2572_257229

theorem felix_tree_chopping (trees_per_sharpen : ℕ) (sharpen_cost : ℕ) (total_spent : ℕ) : 
  trees_per_sharpen = 13 → 
  sharpen_cost = 5 → 
  total_spent = 35 → 
  ∃ (trees_chopped : ℕ), trees_chopped ≥ 91 ∧ trees_chopped ≥ (total_spent / sharpen_cost) * trees_per_sharpen :=
by
  sorry

end NUMINAMATH_CALUDE_felix_tree_chopping_l2572_257229


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2572_257273

theorem complex_expression_simplification :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - 
  (0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884) - 
  (0.2956 * 0.3412 * 0.6573) = -0.3902 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2572_257273


namespace NUMINAMATH_CALUDE_squares_below_line_count_l2572_257297

/-- The number of squares below the line 5x + 45y = 225 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 45
  let y_intercept : ℕ := 5
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

theorem squares_below_line_count : squares_below_line = 88 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_count_l2572_257297


namespace NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l2572_257228

theorem prime_divisor_of_fermat_number (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_divides : p ∣ 2^(2^n) + 1) : 2^(n+1) ∣ p - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_of_fermat_number_l2572_257228


namespace NUMINAMATH_CALUDE_minimum_selling_price_chocolate_manufacturer_l2572_257209

/-- Calculates the minimum selling price per unit to achieve a desired monthly profit -/
def minimum_selling_price (units : ℕ) (cost_per_unit : ℚ) (desired_profit : ℚ) : ℚ :=
  (units * cost_per_unit + desired_profit) / units

theorem minimum_selling_price_chocolate_manufacturer :
  let units : ℕ := 400
  let cost_per_unit : ℚ := 40
  let desired_profit : ℚ := 40000
  minimum_selling_price units cost_per_unit desired_profit = 140 := by
  sorry

end NUMINAMATH_CALUDE_minimum_selling_price_chocolate_manufacturer_l2572_257209


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l2572_257231

theorem candy_box_price_increase (P : ℝ) : P + 0.25 * P = 10 → P = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l2572_257231


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l2572_257219

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n → n ≥ 104 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l2572_257219


namespace NUMINAMATH_CALUDE_biggest_collection_l2572_257204

def yoongi_collection : ℕ := 4
def jungkook_collection : ℕ := 6 * 3
def yuna_collection : ℕ := 5

theorem biggest_collection :
  max yoongi_collection (max jungkook_collection yuna_collection) = jungkook_collection :=
by sorry

end NUMINAMATH_CALUDE_biggest_collection_l2572_257204


namespace NUMINAMATH_CALUDE_domino_partition_exists_l2572_257256

/-- Represents a domino piece with two numbers -/
structure Domino :=
  (a b : Nat)
  (h1 : a ≤ 6)
  (h2 : b ≤ 6)

/-- The set of all domino pieces in a standard double-six set -/
def dominoSet : Finset Domino :=
  sorry

/-- The sum of points on all domino pieces -/
def totalSum : Nat :=
  sorry

/-- A partition of the domino set into 4 groups -/
def Partition := Fin 4 → Finset Domino

theorem domino_partition_exists :
  ∃ (p : Partition),
    (∀ i j, i ≠ j → Disjoint (p i) (p j)) ∧
    (∀ i, (p i).sum (λ d => d.a + d.b) = 21) ∧
    (∀ d ∈ dominoSet, ∃ i, d ∈ p i) :=
  sorry

end NUMINAMATH_CALUDE_domino_partition_exists_l2572_257256


namespace NUMINAMATH_CALUDE_quadratic_minimum_at_positive_x_l2572_257233

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem quadratic_minimum_at_positive_x :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_at_positive_x_l2572_257233


namespace NUMINAMATH_CALUDE_send_more_money_solution_l2572_257280

def is_valid_assignment (S E N D M O R Y : Nat) : Prop :=
  S ≠ 0 ∧ M ≠ 0 ∧
  S < 10 ∧ E < 10 ∧ N < 10 ∧ D < 10 ∧ M < 10 ∧ O < 10 ∧ R < 10 ∧ Y < 10 ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ M ∧ S ≠ O ∧ S ≠ R ∧ S ≠ Y ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ M ∧ E ≠ O ∧ E ≠ R ∧ E ≠ Y ∧
  N ≠ D ∧ N ≠ M ∧ N ≠ O ∧ N ≠ R ∧ N ≠ Y ∧
  D ≠ M ∧ D ≠ O ∧ D ≠ R ∧ D ≠ Y ∧
  M ≠ O ∧ M ≠ R ∧ M ≠ Y ∧
  O ≠ R ∧ O ≠ Y ∧
  R ≠ Y

theorem send_more_money_solution :
  ∃ (S E N D M O R Y : Nat),
    is_valid_assignment S E N D M O R Y ∧
    1000 * S + 100 * E + 10 * N + D + 1000 * M + 100 * O + 10 * R + E =
    10000 * M + 1000 * O + 100 * N + 10 * E + Y :=
by sorry

end NUMINAMATH_CALUDE_send_more_money_solution_l2572_257280


namespace NUMINAMATH_CALUDE_not_perfect_square_l2572_257258

theorem not_perfect_square : 
  (∃ x : ℝ, (6:ℝ)^210 = x^2) ∧
  (∀ x : ℝ, (7:ℝ)^301 ≠ x^2) ∧
  (∃ x : ℝ, (8:ℝ)^402 = x^2) ∧
  (∃ x : ℝ, (9:ℝ)^302 = x^2) ∧
  (∃ x : ℝ, (10:ℝ)^404 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2572_257258


namespace NUMINAMATH_CALUDE_locus_C_equation_point_N_coordinates_l2572_257270

-- Define the circle and its properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Define the locus C
def LocusC (p : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), Circle p r ∧ 
  (p.1 - 1)^2 + p.2^2 = r^2 ∧  -- Tangent to F(1,0)
  (p.1 + 1)^2 = r^2            -- Tangent to x = -1

-- Define point A
def PointA : ℝ × ℝ := (4, 4)

-- Define point B
def PointB : ℝ × ℝ := (0, 4)

-- Define point M
def PointM : ℝ × ℝ := (0, 2)

-- Define point F
def PointF : ℝ × ℝ := (1, 0)

-- Theorem for the equation of locus C
theorem locus_C_equation : 
  ∀ p : ℝ × ℝ, LocusC p ↔ p.2^2 = 4 * p.1 := by sorry

-- Theorem for the coordinates of point N
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, N.1 = 8/5 ∧ N.2 = 4/5 ∧
  (N.2 - PointM.2) / (N.1 - PointM.1) = -3/4 ∧  -- MN perpendicular to FA
  (PointA.2 - PointF.2) / (PointA.1 - PointF.1) = 4/3 := by sorry

end NUMINAMATH_CALUDE_locus_C_equation_point_N_coordinates_l2572_257270


namespace NUMINAMATH_CALUDE_f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l2572_257218

/-- The quadratic function f(x) = x^2 + 15x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 15*x + 3

/-- Theorem stating that f(x) is minimized when x = -15/2 -/
theorem f_min_at_neg_15_div_2 :
  ∀ x : ℝ, f (-15/2) ≤ f x :=
by
  sorry

/-- Theorem stating that -15/2 is the unique minimizer of f(x) -/
theorem f_unique_min_at_neg_15_div_2 :
  ∀ x : ℝ, x ≠ -15/2 → f (-15/2) < f x :=
by
  sorry

end NUMINAMATH_CALUDE_f_min_at_neg_15_div_2_f_unique_min_at_neg_15_div_2_l2572_257218


namespace NUMINAMATH_CALUDE_blue_surface_area_fraction_l2572_257230

theorem blue_surface_area_fraction (edge_length : ℕ) (small_cube_count : ℕ) 
  (green_count : ℕ) (blue_count : ℕ) :
  edge_length = 4 →
  small_cube_count = 64 →
  green_count = 44 →
  blue_count = 20 →
  (∃ (blue_exposed : ℕ), 
    blue_exposed ≤ blue_count ∧ 
    blue_exposed * 1 = (edge_length ^ 2 * 6) / 8) :=
by sorry

end NUMINAMATH_CALUDE_blue_surface_area_fraction_l2572_257230


namespace NUMINAMATH_CALUDE_three_propositions_are_true_l2572_257252

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define the relation of two lines being skew
def are_skew (a b : Line) : Prop := sorry

-- Define the relation of a line intersecting another line at a point
def intersects_at (l1 l2 : Line) (p : Point) : Prop := sorry

-- Define the relation of two lines being parallel
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the relation of two lines determining a plane
def determine_plane (l1 l2 : Line) (p : Plane) : Prop := sorry

theorem three_propositions_are_true :
  -- Proposition 1
  (∀ (a b c d : Line) (E F G H : Point),
    are_skew a b ∧
    intersects_at c a E ∧ intersects_at c b F ∧
    intersects_at d a G ∧ intersects_at d b H ∧
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    are_skew c d) ∧
  -- Proposition 2
  (∀ (a b l : Line),
    are_skew a b →
    are_parallel l a →
    ¬(are_parallel l b)) ∧
  -- Proposition 3
  (∀ (a b l : Line),
    are_skew a b →
    (∃ (P Q : Point), intersects_at l a P ∧ intersects_at l b Q) →
    ∃ (p1 p2 : Plane), determine_plane a l p1 ∧ determine_plane b l p2) :=
by sorry

end NUMINAMATH_CALUDE_three_propositions_are_true_l2572_257252


namespace NUMINAMATH_CALUDE_negative_real_inequality_l2572_257288

theorem negative_real_inequality (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
  x * y * z / ((1 + 5*x) * (4*x + 3*y) * (5*y + 6*z) * (z + 18)) ≤ 1 / 5120 := by
  sorry

end NUMINAMATH_CALUDE_negative_real_inequality_l2572_257288


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2572_257217

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l2572_257217


namespace NUMINAMATH_CALUDE_right_triangle_square_equal_area_l2572_257286

theorem right_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) : 
  (1/2 * s * h = s^2) → h = 2*s := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_square_equal_area_l2572_257286


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2572_257264

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter of 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 8 →
  perimeter = 26 →
  perimeter = 2 * congruent_side + base →
  base = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2572_257264


namespace NUMINAMATH_CALUDE_junior_teachers_sampled_count_l2572_257271

/-- Represents the number of teachers in each category -/
structure TeacherCounts where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the sample size for stratified sampling -/
def SampleSize : Nat := 50

/-- Calculates the number of junior teachers in a stratified sample -/
def juniorTeachersSampled (counts : TeacherCounts) (sampleSize : Nat) : Nat :=
  (sampleSize * counts.junior) / counts.total

/-- Theorem: The number of junior teachers sampled is 20 -/
theorem junior_teachers_sampled_count 
  (counts : TeacherCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.senior = 20)
  (h3 : counts.intermediate = 100)
  (h4 : counts.junior = 80) :
  juniorTeachersSampled counts SampleSize = 20 := by
  sorry

#eval juniorTeachersSampled { total := 200, senior := 20, intermediate := 100, junior := 80 } SampleSize

end NUMINAMATH_CALUDE_junior_teachers_sampled_count_l2572_257271


namespace NUMINAMATH_CALUDE_gain_percent_when_cost_equals_sell_l2572_257200

/-- Proves that if the cost price of 50 articles equals the selling price of 25 articles, 
    then the gain percent is 100%. -/
theorem gain_percent_when_cost_equals_sell (C S : ℝ) 
  (h : 50 * C = 25 * S) : (S - C) / C * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_when_cost_equals_sell_l2572_257200


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l2572_257241

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_xaxis :
  let c1 : Circle := { center := (3, 3), radius := 3 }
  let c2 : Circle := { center := (9, 3), radius := 3 }
  areaRegion c1 c2 = 18 - (9 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l2572_257241


namespace NUMINAMATH_CALUDE_simplify_expression_l2572_257284

theorem simplify_expression : (256 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2572_257284


namespace NUMINAMATH_CALUDE_chocolate_distribution_chocolate_squares_per_student_l2572_257281

theorem chocolate_distribution (gerald_bars : Nat) (squares_per_bar : Nat) (teacher_multiplier : Nat) (num_students : Nat) : Nat :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

-- The main theorem
theorem chocolate_squares_per_student :
  chocolate_distribution 7 8 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_chocolate_squares_per_student_l2572_257281


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l2572_257274

def is_valid_increment (n m : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ),
    n = 10000 * d₁ + 1000 * d₂ + 100 * d₃ + 10 * d₄ + d₅ ∧
    m = 10000 * (d₁ + 2) + 1000 * (d₂ + 4) + 100 * (d₃ + 2) + 10 * (d₄ + 4) + (d₅ + 4) ∧
    d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10 ∧ d₅ < 10

theorem unique_five_digit_number :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 →
    (∃ m : ℕ, is_valid_increment n m ∧ m = 4 * n) →
    n = 14074 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l2572_257274


namespace NUMINAMATH_CALUDE_number_of_divisors_30030_l2572_257225

def number_to_factorize : Nat := 30030

/-- The number of positive divisors of 30030 is 64 -/
theorem number_of_divisors_30030 : 
  (Nat.divisors number_to_factorize).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_30030_l2572_257225


namespace NUMINAMATH_CALUDE_rational_function_with_infinite_integer_values_is_polynomial_l2572_257249

/-- A rational function is a quotient of two real polynomials -/
def RationalFunction (f : ℝ → ℝ) : Prop :=
  ∃ p q : Polynomial ℝ, q ≠ 0 ∧ ∀ x, f x = (p.eval x) / (q.eval x)

/-- A function that takes integer values at infinitely many integer points -/
def IntegerValuesAtInfinitelyManyPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m > n, ∃ k : ℤ, f k = m

/-- Main theorem: If f is a rational function and takes integer values at infinitely many
    integer points, then f is a polynomial -/
theorem rational_function_with_infinite_integer_values_is_polynomial
  (f : ℝ → ℝ) (hf : RationalFunction f) (hi : IntegerValuesAtInfinitelyManyPoints f) :
  ∃ p : Polynomial ℝ, ∀ x, f x = p.eval x :=
sorry

end NUMINAMATH_CALUDE_rational_function_with_infinite_integer_values_is_polynomial_l2572_257249


namespace NUMINAMATH_CALUDE_students_passed_both_tests_l2572_257216

theorem students_passed_both_tests
  (total : ℕ)
  (passed_chinese : ℕ)
  (passed_english : ℕ)
  (failed_both : ℕ)
  (h1 : total = 50)
  (h2 : passed_chinese = 40)
  (h3 : passed_english = 31)
  (h4 : failed_both = 4) :
  total - failed_both = passed_chinese + passed_english - (passed_chinese + passed_english - (total - failed_both)) :=
by sorry

end NUMINAMATH_CALUDE_students_passed_both_tests_l2572_257216


namespace NUMINAMATH_CALUDE_altitude_segment_length_l2572_257257

/-- Represents an acute triangle with two altitudes dividing the sides. -/
structure AcuteTriangleWithAltitudes where
  -- The lengths of the segments created by the altitudes
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ y > 0
  a_val : a = 7
  b_val : b = 4
  c_val : c = 3

/-- The theorem stating that y = 12/7 in the given triangle configuration. -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l2572_257257


namespace NUMINAMATH_CALUDE_cube_root_inequality_l2572_257265

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*Real.sqrt 3)) := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l2572_257265


namespace NUMINAMATH_CALUDE_expression_value_l2572_257268

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x ^ 2)) = 59052 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2572_257268


namespace NUMINAMATH_CALUDE_function_parity_l2572_257236

noncomputable def f (x : ℝ) : ℝ := 1/x - 3^x
noncomputable def g (x : ℝ) : ℝ := 2^x - 2^(-x)
def h (x : ℝ) : ℝ := x^2 + |x|
noncomputable def k (x : ℝ) : ℝ := Real.log ((x+1)/(x-1))

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem function_parity :
  (¬ is_odd f ∧ ¬ is_even f) ∧
  (is_odd g ∨ is_even g) ∧
  (is_odd h ∨ is_even h) ∧
  (is_odd k ∨ is_even k) :=
sorry

end NUMINAMATH_CALUDE_function_parity_l2572_257236


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l2572_257223

/-- The number of rectangles on a 4x4 grid with sides parallel to axes -/
def num_rectangles : ℕ := 36

/-- The number of ways to choose 2 items from 4 -/
def choose_two_from_four : ℕ := 6

theorem rectangles_on_4x4_grid :
  num_rectangles = choose_two_from_four * choose_two_from_four :=
by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l2572_257223


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l2572_257244

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.exp x + x

def line (x : ℝ) : ℝ := 3 * x - 1

theorem min_distance_curve_line :
  ∃ (d : ℝ), d = (3 * Real.sqrt 10) / 10 ∧
  ∀ (x₁ x₂ : ℝ), 
    let y₁ := curve x₁
    let y₂ := line x₂
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l2572_257244


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l2572_257253

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l2572_257253


namespace NUMINAMATH_CALUDE_hyperbola_equation_tangent_line_perpendicular_intersection_l2572_257299

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2 * Real.sqrt 3

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the modified hyperbola C' for part 3
def hyperbola_C' (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2

-- Define the line for part 3
def line_part3 (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Theorem statements
theorem hyperbola_equation :
  ∀ x y : ℝ, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1 := by sorry

theorem tangent_line :
  ∀ k : ℝ, (∃! p : ℝ × ℝ, hyperbola_C p.1 p.2 ∧ line_l k p.1 p.2) ↔
    k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∨ k = 2 ∨ k = -2 := by sorry

theorem perpendicular_intersection :
  ∀ k : ℝ, (∃ A B : ℝ × ℝ,
    hyperbola_C' A.1 A.2 ∧ hyperbola_C' B.1 B.2 ∧
    line_part3 k A.1 A.2 ∧ line_part3 k B.1 B.2 ∧
    A.1 * B.1 + A.2 * B.2 = 0) ↔
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_tangent_line_perpendicular_intersection_l2572_257299


namespace NUMINAMATH_CALUDE_max_value_of_function_l2572_257294

theorem max_value_of_function (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  (x*y + 2*y*z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2572_257294


namespace NUMINAMATH_CALUDE_betty_books_l2572_257238

theorem betty_books : ∀ (b : ℕ), 
  (b + (b + b / 4) = 45) → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_betty_books_l2572_257238


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2572_257292

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (4 - 5*x) = 9 - x := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l2572_257292


namespace NUMINAMATH_CALUDE_wooden_block_length_l2572_257201

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Define the initial length in meters
def initial_length_m : ℝ := 31

-- Define the additional length in centimeters
def additional_length_cm : ℝ := 30

-- Theorem to prove
theorem wooden_block_length :
  (initial_length_m * meters_to_cm + additional_length_cm) = 3130 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_length_l2572_257201


namespace NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_and_tangent_l2572_257246

theorem sine_of_sum_inverse_sine_and_tangent :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_sum_inverse_sine_and_tangent_l2572_257246


namespace NUMINAMATH_CALUDE_mathematics_players_l2572_257227

/-- Theorem: Number of players taking mathematics in Riverdale Academy volleyball team -/
theorem mathematics_players (total : ℕ) (physics : ℕ) (both : ℕ) : 
  total = 15 → physics = 9 → both = 4 → (∃ (math : ℕ), math = 10) :=
by sorry

end NUMINAMATH_CALUDE_mathematics_players_l2572_257227


namespace NUMINAMATH_CALUDE_total_paint_used_l2572_257224

/-- The amount of paint Joe uses at two airports over two weeks -/
def paint_used (paint1 paint2 : ℝ) (week1_ratio1 week2_ratio1 week1_ratio2 week2_ratio2 : ℝ) : ℝ :=
  let remaining1 := paint1 * (1 - week1_ratio1)
  let used1 := paint1 * week1_ratio1 + remaining1 * week2_ratio1
  let remaining2 := paint2 * (1 - week1_ratio2)
  let used2 := paint2 * week1_ratio2 + remaining2 * week2_ratio2
  used1 + used2

/-- Theorem stating the total amount of paint Joe uses at both airports -/
theorem total_paint_used :
  paint_used 360 600 (1/4) (1/6) (1/3) (1/5) = 415 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_used_l2572_257224


namespace NUMINAMATH_CALUDE_smallest_d_correct_l2572_257296

/-- The smallest possible value of d satisfying the triangle and square perimeter conditions -/
def smallest_d : ℕ :=
  let d : ℕ := 675
  d

theorem smallest_d_correct :
  let d := smallest_d
  -- The perimeter of the equilateral triangle exceeds the perimeter of the square by 2023 cm
  ∀ s : ℝ, 3 * (s + d) - 4 * s = 2023 →
  -- The square has a perimeter greater than 0 cm
  (s > 0) →
  -- d is a multiple of 3
  (d % 3 = 0) →
  -- d is the smallest value satisfying these conditions
  ∀ d' : ℕ, d' < d →
    (∀ s : ℝ, 3 * (s + d') - 4 * s = 2023 → s > 0 → d' % 3 = 0 → False) :=
by sorry

#eval smallest_d

end NUMINAMATH_CALUDE_smallest_d_correct_l2572_257296


namespace NUMINAMATH_CALUDE_jordan_max_points_l2572_257243

structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  freeThrowAttempts : ℕ
  threePointSuccess : ℚ
  twoPointSuccess : ℚ
  freeThrowSuccess : ℚ

def totalShots (game : BasketballGame) : ℕ :=
  game.threePointAttempts + game.twoPointAttempts + game.freeThrowAttempts

def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccess * game.threePointAttempts +
  2 * game.twoPointSuccess * game.twoPointAttempts +
  game.freeThrowSuccess * game.freeThrowAttempts

theorem jordan_max_points :
  ∀ (game : BasketballGame),
  game.threePointSuccess = 1/4 →
  game.twoPointSuccess = 2/5 →
  game.freeThrowSuccess = 4/5 →
  totalShots game = 50 →
  totalPoints game ≤ 39 :=
by sorry

end NUMINAMATH_CALUDE_jordan_max_points_l2572_257243


namespace NUMINAMATH_CALUDE_pancake_cooking_theorem_l2572_257251

/-- Represents the minimum time needed to cook a given number of pancakes -/
def min_cooking_time (num_pancakes : ℕ) : ℕ :=
  sorry

/-- The pancake cooking theorem -/
theorem pancake_cooking_theorem :
  let pan_capacity : ℕ := 2
  let cooking_time_per_pancake : ℕ := 2
  let num_pancakes : ℕ := 3
  min_cooking_time num_pancakes = 3 :=
sorry

end NUMINAMATH_CALUDE_pancake_cooking_theorem_l2572_257251


namespace NUMINAMATH_CALUDE_quadratic_function_condition_l2572_257213

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x^2 + m

-- Theorem statement
theorem quadratic_function_condition (m : ℝ) : 
  (∀ x, ∃ a b c, f m x = a * x^2 + b * x + c ∧ a ≠ 0) ↔ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_condition_l2572_257213


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_to_lines_l2572_257250

/-- The minimum sum of distances from a point on the parabola y^2 = 4x to two lines -/
theorem min_distance_sum_parabola_to_lines : 
  let l₁ := {(x, y) : ℝ × ℝ | 4 * x - 3 * y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x = -1}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4 * x}
  let dist_to_l₁ (a : ℝ) := |4 * a^2 - 6 * a + 6| / 5
  let dist_to_l₂ (a : ℝ) := |a^2 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (a : ℝ), (dist_to_l₁ a + dist_to_l₂ a) ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_min_distance_sum_parabola_to_lines_l2572_257250


namespace NUMINAMATH_CALUDE_total_cinnamon_swirls_l2572_257267

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: If there are 3 people eating an equal number of cinnamon swirls, 
    and one person ate 4 pieces, then the total number of pieces is 12. -/
theorem total_cinnamon_swirls : 
  num_people * janes_pieces = 12 := by sorry

end NUMINAMATH_CALUDE_total_cinnamon_swirls_l2572_257267


namespace NUMINAMATH_CALUDE_binomial_expansion_term_sum_l2572_257247

theorem binomial_expansion_term_sum (n : ℕ) (b : ℝ) : 
  n ≥ 2 → 
  b ≠ 0 → 
  (Nat.choose n 3 : ℝ) * b^(n-3) + (Nat.choose n 4 : ℝ) * b^(n-4) = 0 → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_term_sum_l2572_257247


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l2572_257226

/-- Represents a four-digit number in the form x28x --/
def fourDigitNumber (x : ℕ) : ℕ := x * 1000 + 280 + x

/-- Checks if a natural number is a single digit (0-9) --/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem unique_divisible_by_18 :
  ∃! x : ℕ, isSingleDigit x ∧ (fourDigitNumber x % 18 = 0) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l2572_257226


namespace NUMINAMATH_CALUDE_pencil_sales_problem_l2572_257289

theorem pencil_sales_problem (eraser_price regular_price short_price : ℚ)
  (eraser_quantity short_quantity : ℕ) (total_revenue : ℚ)
  (h1 : eraser_price = 0.8)
  (h2 : regular_price = 0.5)
  (h3 : short_price = 0.4)
  (h4 : eraser_quantity = 200)
  (h5 : short_quantity = 35)
  (h6 : total_revenue = 194)
  (h7 : eraser_price * eraser_quantity + regular_price * x + short_price * short_quantity = total_revenue) :
  x = 40 := by
  sorry

#check pencil_sales_problem

end NUMINAMATH_CALUDE_pencil_sales_problem_l2572_257289


namespace NUMINAMATH_CALUDE_right_triangle_x_coordinate_l2572_257278

/-- Given points P, Q, and R forming a right triangle with ∠PQR = 90°, prove that the x-coordinate of R is 13. -/
theorem right_triangle_x_coordinate :
  let P : ℝ × ℝ := (2, 0)
  let Q : ℝ × ℝ := (11, -3)
  let R : ℝ × ℝ := (x, 3)
  ∀ x : ℝ,
  (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0 →
  x = 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_x_coordinate_l2572_257278


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2572_257275

/-- The set of possible slopes for a line with y-intercept (0, -3) intersecting the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt (1/20) ∨ m ≥ Real.sqrt (1/20)}

/-- The equation of the line with slope m and y-intercept (0, -3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) : 
  m ∈ possible_slopes ↔ 
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2572_257275


namespace NUMINAMATH_CALUDE_quanxing_max_difference_l2572_257285

/-- Represents the mass of a bottle of Quanxing mineral water in mL -/
structure QuanxingBottle where
  mass : ℝ
  h : abs (mass - 450) ≤ 1

/-- The maximum difference in mass between any two Quanxing bottles is 2 mL -/
theorem quanxing_max_difference (bottle1 bottle2 : QuanxingBottle) :
  abs (bottle1.mass - bottle2.mass) ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_quanxing_max_difference_l2572_257285


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l2572_257261

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3.5 : ℝ) * n = 28)
  (h2 : (90 : ℝ) - 62 = 28) : 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l2572_257261


namespace NUMINAMATH_CALUDE_difference_of_squares_simplification_l2572_257259

theorem difference_of_squares_simplification : (164^2 - 148^2) / 16 = 312 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_simplification_l2572_257259


namespace NUMINAMATH_CALUDE_claire_gift_card_value_l2572_257260

/-- The value of Claire's gift card -/
def gift_card_value : ℚ := 100

/-- Cost of a latte -/
def latte_cost : ℚ := 3.75

/-- Cost of a croissant -/
def croissant_cost : ℚ := 3.50

/-- Cost of a cookie -/
def cookie_cost : ℚ := 1.25

/-- Number of days Claire buys coffee and pastry -/
def days : ℕ := 7

/-- Number of cookies Claire buys -/
def num_cookies : ℕ := 5

/-- Amount left on the gift card after spending -/
def amount_left : ℚ := 43

/-- Theorem stating the value of Claire's gift card -/
theorem claire_gift_card_value :
  gift_card_value = 
    (latte_cost + croissant_cost) * days + 
    cookie_cost * num_cookies + 
    amount_left :=
by sorry

end NUMINAMATH_CALUDE_claire_gift_card_value_l2572_257260


namespace NUMINAMATH_CALUDE_line_and_symmetric_point_l2572_257248

/-- Given a line with inclination angle 135° passing through (1,1), 
    prove its equation and find the symmetric point of (3,4) with respect to it. -/
theorem line_and_symmetric_point :
  let l : Set (ℝ × ℝ) := {(x, y) | x + y - 2 = 0}
  let P : ℝ × ℝ := (1, 1)
  let A : ℝ × ℝ := (3, 4)
  let inclination_angle : ℝ := 135 * (π / 180)
  -- Line l passes through P
  (P ∈ l) →
  -- The slope of l is tan(135°)
  (∀ (x y : ℝ), (x, y) ∈ l → y - P.2 = Real.tan inclination_angle * (x - P.1)) →
  -- The equation of l is x + y - 2 = 0
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y - 2 = 0) ∧
  -- The symmetric point A' of A with respect to l has coordinates (-2, -1)
  (∃ (A' : ℝ × ℝ), 
    -- A' is on the opposite side of l from A
    (A'.1 + A'.2 - 2) * (A.1 + A.2 - 2) < 0 ∧
    -- The midpoint of AA' is on l
    ((A.1 + A'.1) / 2 + (A.2 + A'.2) / 2 - 2 = 0) ∧
    -- AA' is perpendicular to l
    ((A'.2 - A.2) / (A'.1 - A.1)) * Real.tan inclination_angle = -1 ∧
    -- A' has coordinates (-2, -1)
    A' = (-2, -1)) := by
  sorry

end NUMINAMATH_CALUDE_line_and_symmetric_point_l2572_257248


namespace NUMINAMATH_CALUDE_curve_C_range_l2572_257203

/-- The curve C is defined by the equation x^2 + y^2 + 2ax - 4ay + 5a^2 - 4 = 0 -/
def C (a x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: If all points on curve C are in the second quadrant, then a > 2 -/
theorem curve_C_range (a : ℝ) :
  (∀ x y : ℝ, C a x y → second_quadrant x y) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_range_l2572_257203


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l2572_257212

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    x ≥ 13^6 * 5^7 ∧
    (∀ (x' : ℕ+) (a' b' c' d' : ℕ), 5 * x'^7 = 13 * y^11 → x' = a'^c' * b'^d' → x' ≥ x) ∧
    a + b + c + d = 31 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l2572_257212


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l2572_257245

theorem complex_quadratic_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = Complex.I * 2 ∧ 
  z₂ = -2 - Complex.I * 2 ∧ 
  z₁^2 + 2*z₁ = -3 + Complex.I * 4 ∧
  z₂^2 + 2*z₂ = -3 + Complex.I * 4 := by
sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l2572_257245


namespace NUMINAMATH_CALUDE_distance_from_origin_of_complex_fraction_l2572_257222

theorem distance_from_origin_of_complex_fraction : 
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_of_complex_fraction_l2572_257222


namespace NUMINAMATH_CALUDE_square_difference_given_system_l2572_257254

theorem square_difference_given_system (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 20) 
  (eq2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_system_l2572_257254


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2572_257207

/-- The distance from the focus to the directrix of a parabola y^2 = 8x is 4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 8*x → 
  ∃ (f d : ℝ × ℝ), 
    (f.1 - d.1)^2 + (f.2 - d.2)^2 = 4^2 ∧
    (∀ (p : ℝ × ℝ), p.2^2 = 8*p.1 → 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - d.1)^2 + (p.2 - d.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2572_257207


namespace NUMINAMATH_CALUDE_min_perimeter_special_triangle_l2572_257210

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem min_perimeter_special_triangle :
  ∃ (c : ℕ), 
    is_valid_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_valid_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_special_triangle_l2572_257210


namespace NUMINAMATH_CALUDE_probability_of_two_in_pascal_triangle_l2572_257208

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the occurrences of a specific number in Pascal's Triangle -/
def countOccurrences (triangle : List (List ℕ)) (target : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- The main theorem: probability of selecting 2 from first 20 rows of Pascal's Triangle -/
theorem probability_of_two_in_pascal_triangle :
  let triangle := PascalTriangle 20
  let occurrences := countOccurrences triangle 2
  let total := totalElements triangle
  (occurrences : ℚ) / total = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_in_pascal_triangle_l2572_257208


namespace NUMINAMATH_CALUDE_euler_totient_power_of_two_l2572_257263

theorem euler_totient_power_of_two (n : ℕ) : 
  Odd n → 
  ∃ k m : ℕ, Nat.totient n = 2^k ∧ Nat.totient (n+1) = 2^m → 
  ∃ p : ℕ, n + 1 = 2^p ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_euler_totient_power_of_two_l2572_257263


namespace NUMINAMATH_CALUDE_square_of_1031_l2572_257283

theorem square_of_1031 : (1031 : ℕ)^2 = 1062961 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1031_l2572_257283


namespace NUMINAMATH_CALUDE_chicken_egg_production_l2572_257220

/-- Given that 6 chickens lay 30 eggs in 5 days, prove that 10 chickens will lay 80 eggs in 8 days. -/
theorem chicken_egg_production 
  (initial_chickens : ℕ) 
  (initial_eggs : ℕ) 
  (initial_days : ℕ)
  (new_chickens : ℕ) 
  (new_days : ℕ)
  (h1 : initial_chickens = 6)
  (h2 : initial_eggs = 30)
  (h3 : initial_days = 5)
  (h4 : new_chickens = 10)
  (h5 : new_days = 8) :
  (new_chickens * new_days * initial_eggs) / (initial_chickens * initial_days) = 80 :=
by sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l2572_257220


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2572_257205

/-- Given a parabola with equation y^2 = 8x, prove that its latus rectum has equation x = -2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y^2 = 8*x → (∃ (a : ℝ), a = -2 ∧ ∀ (x₀ y₀ : ℝ), y₀^2 = 8*x₀ → x₀ = a → 
    (x₀, y₀) ∈ {p : ℝ × ℝ | p.1 = a ∧ p.2^2 = 8*p.1}) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l2572_257205


namespace NUMINAMATH_CALUDE_constant_value_l2572_257277

theorem constant_value (x : ℝ) (some_constant a k n : ℝ) :
  (3 * x + some_constant) * (2 * x - 7) = a * x^2 + k * x + n →
  a - n + k = 3 →
  some_constant = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l2572_257277


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2572_257221

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 105 rupees at 25 paise per meter has an area of 10800 square meters -/
theorem rectangular_field_area (length width : ℝ) (fencing_cost : ℝ) : 
  length / width = 4 / 3 →
  fencing_cost = 105 →
  (2 * (length + width)) * 0.25 = fencing_cost * 100 →
  length * width = 10800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2572_257221


namespace NUMINAMATH_CALUDE_negation_absolute_value_inequality_l2572_257290

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 2| < 3) ↔ (∃ x : ℝ, |x - 2| ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_absolute_value_inequality_l2572_257290


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2572_257262

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b = 9 →
  a = 2 * c →
  B = π / 3 →
  a + b + c = 9 + 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2572_257262


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_l2572_257269

theorem divisibility_of_factorial (n : ℕ+) :
  (2011^2011 ∣ n!) → (2011^2012 ∣ n!) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_l2572_257269
