import Mathlib

namespace NUMINAMATH_CALUDE_max_value_expression_l2212_221294

theorem max_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  (1 / (a^2 - 4*a + 9)) + (1 / (b^2 - 4*b + 9)) + (1 / (c^2 - 4*c + 9)) ≤ 7/18 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2212_221294


namespace NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l2212_221221

def cost (n : ℕ) : ℕ := 20 * n + 500

def goldfish_points : Set (ℕ × ℕ) :=
  {p | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ p = (n, cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_points ∧ 
  (∀ p q : ℕ × ℕ, p ∈ goldfish_points → q ∈ goldfish_points → p ≠ q → p.2 ≠ q.2) :=
sorry

end NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l2212_221221


namespace NUMINAMATH_CALUDE_uphill_divisible_by_nine_count_l2212_221274

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def UphillInteger (n : ℕ) : Prop := sorry

/-- Check if a natural number ends with 6 -/
def EndsWithSix (n : ℕ) : Prop := sorry

/-- Count the number of uphill integers ending in 6 that are divisible by 9 -/
def CountUphillDivisibleBySix : ℕ := sorry

theorem uphill_divisible_by_nine_count : CountUphillDivisibleBySix = 2 := by sorry

end NUMINAMATH_CALUDE_uphill_divisible_by_nine_count_l2212_221274


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2212_221206

/-- An arithmetic sequence with non-zero common difference -/
def arithmetic_seq (a : ℕ → ℝ) := ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_seq (b : ℕ → ℝ) := ∃ r ≠ 0, ∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_seq a)
  (h_geom : geometric_seq b)
  (h_relation : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_equal : b 7 = a 7) :
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2212_221206


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l2212_221269

theorem third_root_of_cubic (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ↔ x = -3 ∨ x = 4 ∨ x = -1/2) →
  ∃ x, x ≠ -3 ∧ x ≠ 4 ∧ a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (10 - a) = 0 ∧ x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l2212_221269


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2212_221234

def is_valid_number (n : ℕ) : Prop :=
  ∃ k : ℕ, 
    n = 5 * 10^(k-1) + (n % 10^(k-1)) ∧ 
    10 * (n % 10^(k-1)) + 5 = n / 4

theorem smallest_valid_number : 
  is_valid_number 512820 ∧ 
  ∀ m : ℕ, m < 512820 → ¬(is_valid_number m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2212_221234


namespace NUMINAMATH_CALUDE_unique_prime_divisor_l2212_221258

theorem unique_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k ∈ Finset.range n,
    ∃ p : ℕ, Nat.Prime p ∧ 
      (p ∣ (n.factorial + k + 1)) ∧
      (∀ j ∈ Finset.range n, j ≠ k → ¬(p ∣ (n.factorial + j + 1))) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_divisor_l2212_221258


namespace NUMINAMATH_CALUDE_max_value_expression_l2212_221252

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-5 : ℝ) 5)
  (hb : b ∈ Set.Icc (-5 : ℝ) 5)
  (hc : c ∈ Set.Icc (-5 : ℝ) 5)
  (hd : d ∈ Set.Icc (-5 : ℝ) 5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 110 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ),
    a₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    b₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    c₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    d₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀ = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2212_221252


namespace NUMINAMATH_CALUDE_award_eligibility_l2212_221227

theorem award_eligibility (x : ℕ) : 
  x ≤ 8 → 2 * x + (8 - x) ≥ 12 ↔ 
    x ≥ 4 ∧ x ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_award_eligibility_l2212_221227


namespace NUMINAMATH_CALUDE_polygon_formation_and_perimeter_l2212_221262

-- Define the structures
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Parallelogram where
  O : Point
  X : Point
  Y : Point
  Z : Point

-- Define the function that creates a parallelogram from two points and O
def createParallelogram (O X Y : Point) : Parallelogram := sorry

-- Define the function that checks if a point is inside a triangle
def isPointInTriangle (p : Point) (t : Triangle) : Prop := sorry

-- Define the function that calculates the perimeter of a triangle
def trianglePerimeter (t : Triangle) : ℝ := sorry

-- Define the function that calculates the perimeter of a polygon
def polygonPerimeter (vertices : List Point) : ℝ := sorry

-- Main theorem
theorem polygon_formation_and_perimeter 
  (ABC DEF : Triangle) (O : Point) : 
  ∃ (polygon : List Point),
    (∀ X Y, isPointInTriangle X ABC → isPointInTriangle Y DEF →
      let p := createParallelogram O X Y
      (p.O ∈ polygon ∧ p.X ∈ polygon ∧ p.Y ∈ polygon ∧ p.Z ∈ polygon)) ∧
    (polygon.length = 6) ∧
    (polygonPerimeter polygon = trianglePerimeter ABC + trianglePerimeter DEF) :=
sorry

end NUMINAMATH_CALUDE_polygon_formation_and_perimeter_l2212_221262


namespace NUMINAMATH_CALUDE_road_repair_length_l2212_221290

theorem road_repair_length : 
  ∀ (total_length : ℝ),
  (200 : ℝ) + 0.4 * total_length + 700 = total_length →
  total_length = 1500 := by
sorry

end NUMINAMATH_CALUDE_road_repair_length_l2212_221290


namespace NUMINAMATH_CALUDE_ellipse_trajectory_l2212_221286

-- Define the focal points
def F1 : ℝ × ℝ := (3, 0)
def F2 : ℝ × ℝ := (-3, 0)

-- Define the distance sum constant
def distanceSum : ℝ := 10

-- Define the equation of the ellipse
def isOnEllipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 25) + (P.2^2 / 16) = 1

-- Theorem statement
theorem ellipse_trajectory :
  ∀ P : ℝ × ℝ, 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = distanceSum →
  isOnEllipse P :=
sorry

end NUMINAMATH_CALUDE_ellipse_trajectory_l2212_221286


namespace NUMINAMATH_CALUDE_min_value_h_l2212_221232

theorem min_value_h (x : ℝ) (hx : x > 0) : x + 1/x + 1/(x + 1/x)^2 ≥ 2.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_h_l2212_221232


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l2212_221223

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := x^2 + 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation1 x1 ∧ equation1 x2 ∧ 
  (∀ x : ℝ, equation1 x → x = x1 ∨ x = x2) ∧
  x1 = 1 ∧ x2 = -3 :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation2 x1 ∧ equation2 x2 ∧ 
  (∀ x : ℝ, equation2 x → x = x1 ∨ x = x2) ∧
  x1 = -5 ∧ x2 = 1 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l2212_221223


namespace NUMINAMATH_CALUDE_divisibility_implication_l2212_221296

theorem divisibility_implication (a b : ℤ) : (17 ∣ (2*a + 3*b)) → (17 ∣ (9*a + 5*b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2212_221296


namespace NUMINAMATH_CALUDE_total_candies_l2212_221204

theorem total_candies (linda_candies chloe_candies michael_candies : ℕ) 
  (h1 : linda_candies = 340)
  (h2 : chloe_candies = 280)
  (h3 : michael_candies = 450) :
  linda_candies + chloe_candies + michael_candies = 1070 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2212_221204


namespace NUMINAMATH_CALUDE_special_determinant_l2212_221219

open Matrix

/-- The determinant of an n×n matrix with diagonal elements b and all other elements a
    is equal to [b+(n-1)a](b-a)^(n-1) -/
theorem special_determinant (n : ℕ) (a b : ℝ) :
  let M : Matrix (Fin n) (Fin n) ℝ := λ i j => if i = j then b else a
  det M = (b + (n - 1) * a) * (b - a) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_determinant_l2212_221219


namespace NUMINAMATH_CALUDE_sum_of_squares_divided_by_365_l2212_221210

theorem sum_of_squares_divided_by_365 : (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divided_by_365_l2212_221210


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l2212_221255

/-- The annual cost of Mary Anne's sparkling water consumption -/
def annual_sparkling_water_cost (daily_consumption : ℚ) (bottle_cost : ℚ) : ℚ :=
  (365 : ℚ) * daily_consumption * bottle_cost

/-- Theorem: Mary Anne's annual sparkling water cost is $146.00 -/
theorem mary_anne_sparkling_water_cost :
  annual_sparkling_water_cost (1/5) 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_cost_l2212_221255


namespace NUMINAMATH_CALUDE_units_digit_17_cubed_times_24_l2212_221243

theorem units_digit_17_cubed_times_24 : (17^3 * 24) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_cubed_times_24_l2212_221243


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l2212_221259

theorem rectangular_plot_dimensions (length breadth : ℝ) : 
  length = 55 →
  breadth + (length - breadth) = length →
  4 * breadth + 2 * (length - breadth) = 5300 / 26.5 →
  length - breadth = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l2212_221259


namespace NUMINAMATH_CALUDE_medians_intersect_l2212_221225

/-- Definition of a triangle -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Definition of a point being the midpoint of a line segment -/
def isMidpoint (M : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = (P.2 + Q.2) / 2

/-- Definition of a median -/
def isMedian (P Q R S : ℝ × ℝ) : Prop :=
  isMidpoint S Q R

/-- Theorem: The medians of a triangle intersect at a single point -/
theorem medians_intersect (t : Triangle) 
  (A' : ℝ × ℝ) (B' : ℝ × ℝ) (C' : ℝ × ℝ)
  (h1 : isMidpoint A' t.B t.C)
  (h2 : isMidpoint B' t.C t.A)
  (h3 : isMidpoint C' t.A t.B)
  (h4 : isMedian t.A t.B t.C A')
  (h5 : isMedian t.B t.C t.A B')
  (h6 : isMedian t.C t.A t.B C') :
  ∃ G : ℝ × ℝ, (∃ k₁ k₂ k₃ : ℝ, 
    G = k₁ • t.A + (1 - k₁) • A' ∧
    G = k₂ • t.B + (1 - k₂) • B' ∧
    G = k₃ • t.C + (1 - k₃) • C') :=
  sorry


end NUMINAMATH_CALUDE_medians_intersect_l2212_221225


namespace NUMINAMATH_CALUDE_women_employees_l2212_221267

theorem women_employees (total : ℕ) 
  (h1 : total > 0)
  (h2 : (60 : ℚ) / 100 * total = ↑(total - (40 : ℚ) / 100 * total))
  (h3 : (75 : ℚ) / 100 * ((40 : ℚ) / 100 * total) + 8 = (40 : ℚ) / 100 * total) :
  (60 : ℚ) / 100 * total = 48 := by
  sorry

end NUMINAMATH_CALUDE_women_employees_l2212_221267


namespace NUMINAMATH_CALUDE_common_difference_is_two_l2212_221226

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 2016 / 2016 = seq.S 2015 / 2015 + 1) : 
    ∃ d, d = 2 ∧ ∀ n, seq.a (n + 1) - seq.a n = d :=
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l2212_221226


namespace NUMINAMATH_CALUDE_iodine131_electrons_l2212_221250

structure Atom where
  atomicMass : ℕ
  protonNumber : ℕ

def numberOfNeutrons (a : Atom) : ℕ := a.atomicMass - a.protonNumber

def numberOfElectrons (a : Atom) : ℕ := a.protonNumber

def iodine131 : Atom := ⟨131, 53⟩

theorem iodine131_electrons : numberOfElectrons iodine131 = 53 := by
  sorry

end NUMINAMATH_CALUDE_iodine131_electrons_l2212_221250


namespace NUMINAMATH_CALUDE_sin_cos_product_l2212_221279

theorem sin_cos_product (α : ℝ) (h : Real.sin α - Real.cos α = -5/4) :
  Real.sin α * Real.cos α = -9/32 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l2212_221279


namespace NUMINAMATH_CALUDE_fifth_month_sales_l2212_221247

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_3 : ℕ := 6855
def sales_4 : ℕ := 7230
def sales_6 : ℕ := 7991
def average_sales : ℕ := 7000
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sales ∧
    sales_5 = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l2212_221247


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2212_221281

theorem triangle_max_perimeter (A B C : ℝ) (b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  (1 - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C →
  1 + b + c ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2212_221281


namespace NUMINAMATH_CALUDE_f_inequality_l2212_221283

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_def (x : ℝ) : f (Real.tan (2 * x)) = Real.tan x ^ 4 + (1 / Real.tan x) ^ 4

theorem f_inequality : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) ≥ 196 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2212_221283


namespace NUMINAMATH_CALUDE_glued_cubes_faces_l2212_221297

/-- The number of faces of a cube -/
def cube_faces : ℕ := 6

/-- The number of new faces contributed by each glued cube -/
def new_faces_per_cube : ℕ := 5

/-- The number of faces in the resulting solid when a cube is glued to each face of an original cube -/
def resulting_solid_faces : ℕ := cube_faces + cube_faces * new_faces_per_cube

theorem glued_cubes_faces : resulting_solid_faces = 36 := by
  sorry

end NUMINAMATH_CALUDE_glued_cubes_faces_l2212_221297


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l2212_221233

/-- The line equation 2y + 5x = 15 -/
def line_equation (x y : ℝ) : Prop := 2 * y + 5 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (3, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l2212_221233


namespace NUMINAMATH_CALUDE_series_duration_l2212_221245

theorem series_duration (episode1 episode2 episode3 episode4 : ℕ) 
  (h1 : episode1 = 58)
  (h2 : episode2 = 62)
  (h3 : episode3 = 65)
  (h4 : episode4 = 55) :
  (episode1 + episode2 + episode3 + episode4) / 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_series_duration_l2212_221245


namespace NUMINAMATH_CALUDE_prob_two_red_is_two_fifths_l2212_221271

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The number of balls drawn from the bag -/
def num_drawn : ℕ := 2

/-- The probability of drawing two red balls -/
def prob_two_red : ℚ := (num_red_balls.choose num_drawn : ℚ) / (total_balls.choose num_drawn)

theorem prob_two_red_is_two_fifths : prob_two_red = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_is_two_fifths_l2212_221271


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2212_221287

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2212_221287


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2212_221242

theorem simplify_sqrt_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 10 + (2 * Real.sqrt 500) / Real.sqrt 5 = Real.sqrt 2205 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2212_221242


namespace NUMINAMATH_CALUDE_find_fraction_l2212_221288

theorem find_fraction (x : ℝ) (f : ℝ) : x - f * x = 58 → x = 145 → f = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_find_fraction_l2212_221288


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2212_221295

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/2
  let n : ℕ := 7
  geometric_sum a r n = 127/256 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2212_221295


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2212_221257

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (3 : ℂ) - 2 * i * z = -4 + 5 * i * z ∧ z = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2212_221257


namespace NUMINAMATH_CALUDE_greatest_number_of_bouquets_l2212_221282

theorem greatest_number_of_bouquets (white_tulips red_tulips : ℕ) 
  (h1 : white_tulips = 21) (h2 : red_tulips = 91) : 
  (Nat.gcd white_tulips red_tulips) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_of_bouquets_l2212_221282


namespace NUMINAMATH_CALUDE_income_comparison_l2212_221289

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.4)) :
  mary = juan * 0.84 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l2212_221289


namespace NUMINAMATH_CALUDE_urn_problem_l2212_221241

theorem urn_problem (M : ℕ) : M = 111 ↔ 
  (5 : ℝ) / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l2212_221241


namespace NUMINAMATH_CALUDE_gcd_8917_4273_l2212_221292

theorem gcd_8917_4273 : Nat.gcd 8917 4273 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8917_4273_l2212_221292


namespace NUMINAMATH_CALUDE_log_1560_base_5_rounded_l2212_221230

theorem log_1560_base_5_rounded (ε : ℝ) (h : ε > 0) :
  ∃ (n : ℤ), n = 5 ∧ |Real.log 1560 / Real.log 5 - n| < 1/2 + ε :=
sorry

end NUMINAMATH_CALUDE_log_1560_base_5_rounded_l2212_221230


namespace NUMINAMATH_CALUDE_triangle_area_l2212_221261

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 13/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2212_221261


namespace NUMINAMATH_CALUDE_det_of_matrix_is_one_l2212_221293

-- Define the determinant formula for a 2x2 matrix
def det_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define our specific matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]

-- Theorem statement
theorem det_of_matrix_is_one :
  det_2x2 (matrix 0 0) (matrix 0 1) (matrix 1 0) (matrix 1 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_is_one_l2212_221293


namespace NUMINAMATH_CALUDE_total_spider_legs_l2212_221254

/-- The number of spiders in Zoey's room -/
def num_spiders : ℕ := 5

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Zoey's room is 40 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l2212_221254


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l2212_221270

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * (N - 2)) / Nat.factorial (N + 2) = 
  (Nat.factorial N * (N - 2)) / (N + 2) := by
sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l2212_221270


namespace NUMINAMATH_CALUDE_prob_over_60_and_hypertension_is_9_percent_l2212_221248

/-- The probability of a person being over 60 years old in the region -/
def prob_over_60 : ℝ := 0.2

/-- The probability of a person having hypertension given they are over 60 -/
def prob_hypertension_given_over_60 : ℝ := 0.45

/-- The probability of a person being both over 60 and having hypertension -/
def prob_over_60_and_hypertension : ℝ := prob_over_60 * prob_hypertension_given_over_60

theorem prob_over_60_and_hypertension_is_9_percent :
  prob_over_60_and_hypertension = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_prob_over_60_and_hypertension_is_9_percent_l2212_221248


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2212_221246

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + (-11) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l2212_221246


namespace NUMINAMATH_CALUDE_nancy_quarters_l2212_221291

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The total amount Nancy saved in dollars -/
def total_saved : ℚ := 3

/-- The number of quarters Nancy saved -/
def num_quarters : ℕ := 12

theorem nancy_quarters :
  (quarter_value * num_quarters : ℚ) = total_saved := by sorry

end NUMINAMATH_CALUDE_nancy_quarters_l2212_221291


namespace NUMINAMATH_CALUDE_difference_of_squares_502_498_l2212_221256

theorem difference_of_squares_502_498 : 502^2 - 498^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_502_498_l2212_221256


namespace NUMINAMATH_CALUDE_edge_projection_max_sum_l2212_221220

theorem edge_projection_max_sum (a b : ℝ) : 
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 7 ∧ x^2 + y^2 = 6 ∧ 
   a^2 = x^2 + 1 ∧ b^2 = y^2 + 1) →
  a + b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_edge_projection_max_sum_l2212_221220


namespace NUMINAMATH_CALUDE_cube_painting_probability_l2212_221253

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a painted cube -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def total_single_cube_paintings : ℕ := 729

/-- The total number of ways to paint two cubes -/
def total_two_cube_paintings : ℕ := 531441

/-- The number of ways to paint two cubes so they are identical after rotation -/
def identical_after_rotation : ℕ := 1178

/-- The probability that two independently painted cubes are identical after rotation -/
def probability_identical_after_rotation : ℚ := 1178 / 531441

theorem cube_painting_probability :
  probability_identical_after_rotation = identical_after_rotation / total_two_cube_paintings :=
by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l2212_221253


namespace NUMINAMATH_CALUDE_gcd_property_l2212_221236

theorem gcd_property (a b c : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd a.natAbs (b * c).natAbs = Nat.gcd a.natAbs c.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l2212_221236


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2212_221212

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x + 2| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2212_221212


namespace NUMINAMATH_CALUDE_constant_sum_through_P_l2212_221251

/-- The function f(x) = x³ + 3x² + x -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + x

/-- The point P on the graph of f -/
def P : ℝ × ℝ := (-1, f (-1))

theorem constant_sum_through_P :
  ∃ (y : ℝ), ∀ (x₁ x₂ : ℝ),
    x₁ ≠ -1 → x₂ ≠ -1 →
    (x₂ - (-1)) * (f x₁ - f (-1)) = (x₁ - (-1)) * (f x₂ - f (-1)) →
    f x₁ + f x₂ = y :=
  sorry

end NUMINAMATH_CALUDE_constant_sum_through_P_l2212_221251


namespace NUMINAMATH_CALUDE_average_age_increase_l2212_221260

theorem average_age_increase (n : ℕ) (m : ℕ) (avg_29 : ℝ) (age_30 : ℕ) :
  n = 30 →
  m = 29 →
  avg_29 = 12 →
  age_30 = 80 →
  let total_29 := m * avg_29
  let new_total := total_29 + age_30
  let new_avg := new_total / n
  abs (new_avg - avg_29 - 2.27) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_average_age_increase_l2212_221260


namespace NUMINAMATH_CALUDE_function_inequality_l2212_221244

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + 2 * (deriv f x) > 0) : 
  f 1 > f 0 / Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2212_221244


namespace NUMINAMATH_CALUDE_volume_per_gram_l2212_221205

/-- Given a substance with specific properties, calculate its volume per gram -/
theorem volume_per_gram (density : ℝ) (mass_per_cubic_meter : ℝ) (grams_per_kg : ℝ) (cc_per_cubic_meter : ℝ) :
  density = mass_per_cubic_meter ∧ 
  grams_per_kg = 1000 ∧ 
  cc_per_cubic_meter = 1000000 →
  (1 / density) * grams_per_kg * cc_per_cubic_meter = 10 := by
  sorry

#check volume_per_gram

end NUMINAMATH_CALUDE_volume_per_gram_l2212_221205


namespace NUMINAMATH_CALUDE_machine_quality_l2212_221208

/-- Represents a packaging machine --/
structure PackagingMachine where
  weight : Real → Real  -- Random variable representing packaging weight

/-- Defines the expected value of a random variable --/
def expectedValue (X : Real → Real) : Real :=
  sorry

/-- Defines the variance of a random variable --/
def variance (X : Real → Real) : Real :=
  sorry

/-- Determines if a packaging machine has better quality --/
def betterQuality (m1 m2 : PackagingMachine) : Prop :=
  expectedValue m1.weight = expectedValue m2.weight ∧
  variance m1.weight > variance m2.weight →
  sorry  -- This represents that m2 has better quality

/-- Theorem stating which machine has better quality --/
theorem machine_quality (A B : PackagingMachine) :
  betterQuality A B → sorry  -- This represents that B has better quality
:= by sorry

end NUMINAMATH_CALUDE_machine_quality_l2212_221208


namespace NUMINAMATH_CALUDE_range_of_a_l2212_221229

/-- Custom multiplication operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2212_221229


namespace NUMINAMATH_CALUDE_sum_in_base5_l2212_221201

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 -/
structure Base5 where
  value : ℕ

theorem sum_in_base5 : 
  let a := Base5.mk 132
  let b := Base5.mk 214
  let c := Base5.mk 341
  let sum := base10ToBase5 (base5ToBase10 a.value + base5ToBase10 b.value + base5ToBase10 c.value)
  sum = 1242 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l2212_221201


namespace NUMINAMATH_CALUDE_no_common_root_l2212_221299

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ∀ x : ℝ, (x^2 + b*x + c = 0) → (x^2 + a*x + d = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_l2212_221299


namespace NUMINAMATH_CALUDE_root_product_equals_sixteen_l2212_221240

theorem root_product_equals_sixteen :
  (16 : ℝ)^(1/4) * (64 : ℝ)^(1/3) * (4 : ℝ)^(1/2) = 16 := by sorry

end NUMINAMATH_CALUDE_root_product_equals_sixteen_l2212_221240


namespace NUMINAMATH_CALUDE_trendy_quotient_l2212_221268

/-- A number is trendy if it contains the digit sequence 2016 -/
def IsTrendy (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a * 10000 + 2016 * 100 + b * 10 + c

/-- Main theorem: Any natural number can be expressed as the quotient of two trendy numbers -/
theorem trendy_quotient (N : ℕ) : 
  ∃ A D : ℕ, IsTrendy A ∧ IsTrendy D ∧ N = A / D :=
sorry

end NUMINAMATH_CALUDE_trendy_quotient_l2212_221268


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2212_221264

def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_intersection_theorem :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2212_221264


namespace NUMINAMATH_CALUDE_total_selling_price_theorem_l2212_221298

def calculate_selling_price (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let price_before_discount := cost * (1 + profit_percent)
  price_before_discount * (1 - discount_percent)

theorem total_selling_price_theorem :
  let item1 := calculate_selling_price 192 0.25 0.10
  let item2 := calculate_selling_price 350 0.15 0.05
  let item3 := calculate_selling_price 500 0.30 0.15
  item1 + item2 + item3 = 1150.875 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_theorem_l2212_221298


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2212_221275

theorem consecutive_integers_product_sum :
  ∀ x y z : ℤ,
  (y = x + 1) →
  (z = y + 1) →
  (x * y * z = 336) →
  (x + y + z = 21) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l2212_221275


namespace NUMINAMATH_CALUDE_left_handed_rock_fans_count_l2212_221280

/-- Represents a club with members having different characteristics -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  rockFans : ℕ
  rightHandedNonRockFans : ℕ

/-- The number of left-handed rock music fans in the club -/
def leftHandedRockFans (c : Club) : ℕ :=
  c.leftHanded + c.rockFans - (c.total - c.rightHandedNonRockFans)

/-- Theorem stating the number of left-handed rock music fans in the given club -/
theorem left_handed_rock_fans_count (c : Club) 
  (h1 : c.total = 25)
  (h2 : c.leftHanded = 10)
  (h3 : c.rockFans = 18)
  (h4 : c.rightHandedNonRockFans = 4) :
  leftHandedRockFans c = 7 := by
  sorry

#eval leftHandedRockFans { total := 25, leftHanded := 10, rockFans := 18, rightHandedNonRockFans := 4 }

end NUMINAMATH_CALUDE_left_handed_rock_fans_count_l2212_221280


namespace NUMINAMATH_CALUDE_job_completion_time_l2212_221278

/-- Given two people A and B who can complete a job individually in 9 and 18 days respectively,
    this theorem proves that they can complete the job together in 6 days. -/
theorem job_completion_time (a_time b_time combined_time : ℚ) 
  (ha : a_time = 9)
  (hb : b_time = 18)
  (hc : combined_time = 6)
  (h_combined : (1 / a_time + 1 / b_time)⁻¹ = combined_time) : 
  combined_time = 6 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2212_221278


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_ratio_3_l2212_221238

theorem tan_alpha_2_implies_ratio_3 (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_ratio_3_l2212_221238


namespace NUMINAMATH_CALUDE_second_discount_percentage_prove_discount_percentage_l2212_221235

/-- Calculates the second discount percentage given the original price, first discount percentage, and final price --/
theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_decimal := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_decimal * 100

/-- Proves that the second discount percentage is approximately 2% given the problem conditions --/
theorem prove_discount_percentage : 
  let original_price : ℝ := 65
  let first_discount_percent : ℝ := 10
  let final_price : ℝ := 57.33
  let result := second_discount_percentage original_price first_discount_percent final_price
  abs (result - 2) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_prove_discount_percentage_l2212_221235


namespace NUMINAMATH_CALUDE_miser_knight_theorem_l2212_221239

theorem miser_knight_theorem (N : ℕ) (h2 : ∀ (a b : ℕ), a + b = 2 → N % a = 0 ∧ N % b = 0)
  (h3 : ∀ (a b c : ℕ), a + b + c = 3 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0)
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = 4 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0)
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = 5 → N % a = 0 ∧ N % b = 0 ∧ N % c = 0 ∧ N % d = 0 ∧ N % e = 0) :
  N % 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_miser_knight_theorem_l2212_221239


namespace NUMINAMATH_CALUDE_ellipse_properties_l2212_221209

structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_b_lt_a : b < a
  h_e_eq : e = (a^2 - b^2).sqrt / a

def standard_equation (E : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / E.a^2 + y^2 / E.b^2 = 1

def vertices (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a, 0), (E.a, 0)}

def foci (E : Ellipse) : Set (ℝ × ℝ) :=
  {(-E.a * E.e, 0), (E.a * E.e, 0)}

def major_axis_length (E : Ellipse) : ℝ := 2 * E.a

def focal_distance (E : Ellipse) : ℝ := 2 * E.a * E.e

theorem ellipse_properties (E : Ellipse) (h_a : E.a = 5) (h_e : E.e = 4/5) :
  standard_equation E ∧
  vertices E = {(-5, 0), (5, 0)} ∧
  foci E = {(-4, 0), (4, 0)} ∧
  major_axis_length E = 10 ∧
  focal_distance E = 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2212_221209


namespace NUMINAMATH_CALUDE_circumcircle_equation_l2212_221263

-- Define the points and line
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def area_ABC : ℝ := 10

-- Define the possible equations of the circumcircle
def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 - 1/2 * x - 5 * y - 3/2 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 - 25/6 * x - 89/9 * y + 347/18 = 0

-- Theorem statement
theorem circumcircle_equation :
  ∃ (C : ℝ × ℝ), line_C C.1 C.2 ∧
  (∀ (x y : ℝ), circle_eq1 x y ∨ circle_eq2 x y) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l2212_221263


namespace NUMINAMATH_CALUDE_absolute_value_of_complex_product_l2212_221249

open Complex

theorem absolute_value_of_complex_product : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) * (1 + 3*i)
  Complex.abs z = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_complex_product_l2212_221249


namespace NUMINAMATH_CALUDE_set_intersection_union_theorem_l2212_221237

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem set_intersection_union_theorem (a b : ℝ) :
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a = -3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_theorem_l2212_221237


namespace NUMINAMATH_CALUDE_lisa_notebook_savings_l2212_221266

/-- Calculates the savings when buying notebooks with discounts -/
def notebook_savings (
  quantity : ℕ
  ) (original_price : ℚ
  ) (discount_rate : ℚ
  ) (bulk_discount : ℚ
  ) (bulk_threshold : ℕ
  ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_discount := quantity * discounted_price
  let final_total := 
    if quantity > bulk_threshold
    then total_with_discount - bulk_discount
    else total_with_discount
  total_without_discount - final_total

/-- Theorem stating the savings for Lisa's notebook purchase -/
theorem lisa_notebook_savings :
  notebook_savings 8 3 (30/100) 5 7 = 61/5 := by
  sorry

end NUMINAMATH_CALUDE_lisa_notebook_savings_l2212_221266


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l2212_221265

-- Define the function representing the curve
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4 * x

-- Theorem statement
theorem tangent_slope_at_zero : f' 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l2212_221265


namespace NUMINAMATH_CALUDE_jake_has_fewer_than_19_peaches_l2212_221200

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jill : ℕ
  jake : ℕ

/-- The given conditions -/
def peach_conditions (p : PeachCount) : Prop :=
  p.steven = 19 ∧
  p.jill = 6 ∧
  p.steven = p.jill + 13 ∧
  p.jake < p.steven

/-- Theorem: Jake has fewer than 19 peaches -/
theorem jake_has_fewer_than_19_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.jake < 19 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_fewer_than_19_peaches_l2212_221200


namespace NUMINAMATH_CALUDE_exam_thresholds_l2212_221222

theorem exam_thresholds (T : ℝ) 
  (hA : 0.25 * T + 30 = 130) 
  (hB : 0.35 * T - 10 = 130) 
  (hC : 0.40 * T = 160) : 
  (130 : ℝ) = 130 ∧ (160 : ℝ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_exam_thresholds_l2212_221222


namespace NUMINAMATH_CALUDE_jim_diving_hours_l2212_221276

/-- The number of gold coins Jim finds per hour -/
def gold_coins_per_hour : ℕ := 25

/-- The number of gold coins in the treasure chest -/
def chest_coins : ℕ := 100

/-- The number of smaller bags Jim found -/
def num_smaller_bags : ℕ := 2

/-- The number of gold coins in each smaller bag -/
def coins_per_smaller_bag : ℕ := chest_coins / 2

/-- The total number of gold coins Jim found -/
def total_coins : ℕ := chest_coins + num_smaller_bags * coins_per_smaller_bag

/-- Theorem: Jim spent 8 hours scuba diving -/
theorem jim_diving_hours : total_coins / gold_coins_per_hour = 8 := by
  sorry

end NUMINAMATH_CALUDE_jim_diving_hours_l2212_221276


namespace NUMINAMATH_CALUDE_x_value_l2212_221215

theorem x_value : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ∧ x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2212_221215


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l2212_221224

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0) ∧
    (x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0) ∧
    (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l2212_221224


namespace NUMINAMATH_CALUDE_xy_equals_nine_l2212_221277

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2*y) = x^2 + 18) : x * y = 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_nine_l2212_221277


namespace NUMINAMATH_CALUDE_quadrilateral_property_l2212_221231

-- Define the quadrilateral and its properties
structure Quadrilateral :=
  (area : ℝ)
  (pq : ℝ)
  (rs : ℝ)
  (d : ℝ)
  (m : ℕ)
  (n : ℕ)
  (p : ℕ)

-- Define the theorem
theorem quadrilateral_property (q : Quadrilateral) : 
  q.area = 15 ∧ q.pq = 6 ∧ q.rs = 8 ∧ q.d^2 = q.m + q.n * Real.sqrt q.p → 
  q.m + q.n + q.p = 81 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l2212_221231


namespace NUMINAMATH_CALUDE_triangle_property_l2212_221217

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (∀ b' c' : Real, 0 < b' ∧ 0 < c' → a + b' + c' ≤ 3 + 2 * Real.sqrt 3) ∧
  (∃ b' c' : Real, 0 < b' ∧ 0 < c' ∧ a + b' + c' = 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2212_221217


namespace NUMINAMATH_CALUDE_sufficient_condition_for_perpendicular_l2212_221202

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the perpendicular relation between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the perpendicular relation between two planes
def perp_plane_plane (p1 : Plane) (p2 : Plane) : Prop := sorry

-- Theorem statement
theorem sufficient_condition_for_perpendicular 
  (α β : Plane) (m n : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_perpendicular_l2212_221202


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l2212_221216

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

theorem ellipse_fixed_point_intersection
  (C : Ellipse)
  (h_point : (1 : ℝ)^2 / C.a^2 + (Real.sqrt 6 / 3)^2 / C.b^2 = 1)
  (h_eccentricity : Real.sqrt (C.a^2 - C.b^2) / C.a = Real.sqrt 6 / 3)
  (l : Line)
  (P Q : Point)
  (h_intersect_P : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_intersect_Q : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1)
  (h_on_line_P : P.y = l.m * P.x + l.c)
  (h_on_line_Q : Q.y = l.m * Q.x + l.c)
  (h_perpendicular : P.x * Q.x + P.y * Q.y = 0)
  (h_not_vertex : l.m ≠ 0 ∨ l.c ≠ 1) :
  l.m * 0 + l.c = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l2212_221216


namespace NUMINAMATH_CALUDE_charge_difference_l2212_221273

/-- The charge for a single color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The charge for a single color copy at print shop Y -/
def charge_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 80

/-- The theorem stating the difference in charges between print shops Y and X for 80 color copies -/
theorem charge_difference : (num_copies : ℚ) * charge_Y - (num_copies : ℚ) * charge_X = 120 := by
  sorry

end NUMINAMATH_CALUDE_charge_difference_l2212_221273


namespace NUMINAMATH_CALUDE_factors_of_48_l2212_221272

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by sorry

end NUMINAMATH_CALUDE_factors_of_48_l2212_221272


namespace NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l2212_221228

/-- The percentage of children who play soccer -/
def soccer_players : ℝ := 0.55

/-- The percentage of children who swim -/
def swimmers : ℝ := 0.45

/-- The percentage of soccer players who swim -/
def soccer_swimmers : ℝ := 0.50

/-- The percentage of campers who play basketball -/
def basketball_players : ℝ := 0.35

/-- The percentage of basketball players who play soccer but do not swim -/
def basketball_soccer_no_swim : ℝ := 0.20

/-- Theorem stating that 50% of non-swimmers play soccer -/
theorem non_swimmers_playing_soccer : 
  (soccer_players - soccer_players * soccer_swimmers) / (1 - swimmers) = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_non_swimmers_playing_soccer_l2212_221228


namespace NUMINAMATH_CALUDE_exist_non_adjacent_colors_l2212_221285

/-- Represents a coloring of a 50x50 square --/
def Coloring := Fin 50 → Fin 50 → Fin 100

/-- No single-color domino exists in the coloring --/
def NoSingleColorDomino (c : Coloring) : Prop :=
  ∀ i j, (i < 49 → c i j ≠ c (i+1) j) ∧ (j < 49 → c i j ≠ c i (j+1))

/-- All colors are present in the coloring --/
def AllColorsPresent (c : Coloring) : Prop :=
  ∀ color, ∃ i j, c i j = color

/-- Two colors are not adjacent if they don't appear next to each other anywhere --/
def ColorsNotAdjacent (c : Coloring) (color1 color2 : Fin 100) : Prop :=
  ∀ i j, (i < 49 → (c i j ≠ color1 ∨ c (i+1) j ≠ color2) ∧ (c i j ≠ color2 ∨ c (i+1) j ≠ color1)) ∧
         (j < 49 → (c i j ≠ color1 ∨ c i (j+1) ≠ color2) ∧ (c i j ≠ color2 ∨ c i (j+1) ≠ color1))

/-- Main theorem: There exist two non-adjacent colors in any valid coloring --/
theorem exist_non_adjacent_colors (c : Coloring) 
  (h1 : NoSingleColorDomino c) (h2 : AllColorsPresent c) : 
  ∃ color1 color2, ColorsNotAdjacent c color1 color2 := by
  sorry

end NUMINAMATH_CALUDE_exist_non_adjacent_colors_l2212_221285


namespace NUMINAMATH_CALUDE_boxes_per_hand_for_ten_people_l2212_221214

/-- Given a group of people and the total number of boxes they can hold,
    calculate the number of boxes a single person can hold in each hand. -/
def boxes_per_hand (group_size : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / group_size) / 2

/-- Theorem stating that for a group of 10 people holding 20 boxes in total,
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_ten_people :
  boxes_per_hand 10 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_hand_for_ten_people_l2212_221214


namespace NUMINAMATH_CALUDE_bug_path_theorem_l2212_221211

/-- Represents a rectangular garden paved with square pavers -/
structure PavedGarden where
  width : ℕ  -- width in feet
  length : ℕ  -- length in feet
  paver_size : ℕ  -- size of square paver in feet

/-- Calculates the number of pavers a bug visits when walking diagonally across the garden -/
def pavers_visited (garden : PavedGarden) : ℕ :=
  let width_pavers := garden.width / garden.paver_size
  let length_pavers := (garden.length + garden.paver_size - 1) / garden.paver_size
  width_pavers + length_pavers - Nat.gcd width_pavers length_pavers

/-- Theorem stating that a bug walking diagonally across a 14x19 garden with 2-foot pavers visits 16 pavers -/
theorem bug_path_theorem :
  let garden : PavedGarden := { width := 14, length := 19, paver_size := 2 }
  pavers_visited garden = 16 := by sorry

end NUMINAMATH_CALUDE_bug_path_theorem_l2212_221211


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l2212_221284

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The loss percentage for a radio with cost price 1900 and selling price 1330 is 30% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1900
  let selling_price : ℚ := 1330
  loss_percentage cost_price selling_price = 30 := by
sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l2212_221284


namespace NUMINAMATH_CALUDE_sequence_sum_expression_l2212_221218

/-- Given a sequence {a_n} with sum of first n terms S_n, where a_1 = 1 and S_n = 2a_{n+1},
    prove that S_n = (3/2)^(n-1) for n > 1 -/
theorem sequence_sum_expression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  (∀ k, S k = 2 * a (k + 1)) →
  n > 1 →
  S n = (3/2)^(n-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_expression_l2212_221218


namespace NUMINAMATH_CALUDE_max_profit_is_120_l2212_221203

def profit_A (x : ℕ) : ℚ := -x^2 + 21*x
def profit_B (x : ℕ) : ℚ := 2*x
def total_profit (x : ℕ) : ℚ := profit_A x + profit_B (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x > 0 ∧ x ≤ 15 ∧
  total_profit x = 120 ∧
  ∀ y : ℕ, y > 0 → y ≤ 15 → total_profit y ≤ total_profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l2212_221203


namespace NUMINAMATH_CALUDE_no_rin_is_bin_l2212_221213

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Bin, Fin, and Rin
variable (Bin Fin Rin : U → Prop)

-- Premise I: All Bins are Fins
axiom all_bins_are_fins : ∀ x, Bin x → Fin x

-- Premise II: Some Rins are not Fins
axiom some_rins_not_fins : ∃ x, Rin x ∧ ¬Fin x

-- Theorem to prove
theorem no_rin_is_bin : (∀ x, Bin x → Fin x) → (∃ x, Rin x ∧ ¬Fin x) → (∀ x, Rin x → ¬Bin x) :=
sorry

end NUMINAMATH_CALUDE_no_rin_is_bin_l2212_221213


namespace NUMINAMATH_CALUDE_sum_A_B_eq_x_squared_sum_A_2B_eq_24_when_x_neg_2_l2212_221207

-- Define variables
variable (x : ℝ)

-- Define B as a function of x
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 6

-- Define A as a function of x, given A-B
def A (x : ℝ) : ℝ := (-7 * x^2 + 10 * x + 12) + B x

-- Theorem 1: A+B = x^2
theorem sum_A_B_eq_x_squared (x : ℝ) : A x + B x = x^2 := by sorry

-- Theorem 2: A+2B = 24 when x=-2
theorem sum_A_2B_eq_24_when_x_neg_2 : A (-2) + 2 * B (-2) = 24 := by sorry

end NUMINAMATH_CALUDE_sum_A_B_eq_x_squared_sum_A_2B_eq_24_when_x_neg_2_l2212_221207
