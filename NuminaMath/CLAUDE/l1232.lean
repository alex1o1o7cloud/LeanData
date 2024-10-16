import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_constant_sequence_l1232_123247

/-- An irreducible polynomial with integer coefficients -/
def IrreducibleIntPoly := Polynomial ℤ

/-- The number of solutions to p(x) ≡ 0 mod q^n -/
def num_solutions (p : IrreducibleIntPoly) (q : ℕ) (n : ℕ) : ℕ := sorry

theorem existence_of_constant_sequence 
  (p : IrreducibleIntPoly) 
  (q : ℕ) 
  (h_q : Nat.Prime q) :
  ∃ M : ℕ, ∀ n ≥ M, num_solutions p q n = num_solutions p q M := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_sequence_l1232_123247


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1232_123256

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1232_123256


namespace NUMINAMATH_CALUDE_tangent_ellipse_d_value_l1232_123234

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis with foci at (3,7) and (d,7) -/
structure TangentEllipse where
  d : ℝ
  focus1 : ℝ × ℝ := (3, 7)
  focus2 : ℝ × ℝ := (d, 7)
  in_first_quadrant : d > 3
  tangent_to_axes : True  -- This is a simplification, as we can't directly represent tangency in this structure

/-- The value of d for the given ellipse is 49/3 -/
theorem tangent_ellipse_d_value (e : TangentEllipse) : e.d = 49/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ellipse_d_value_l1232_123234


namespace NUMINAMATH_CALUDE_sum_of_repeated_addition_and_multiplication_l1232_123283

theorem sum_of_repeated_addition_and_multiplication (m n : ℕ+) :
  (m.val * 2) + (3 ^ n.val) = 2 * m.val + 3 ^ n.val := by sorry

end NUMINAMATH_CALUDE_sum_of_repeated_addition_and_multiplication_l1232_123283


namespace NUMINAMATH_CALUDE_ones_digit_of_6_to_34_l1232_123246

theorem ones_digit_of_6_to_34 : ∃ k : ℕ, 6^34 = 10 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_to_34_l1232_123246


namespace NUMINAMATH_CALUDE_largest_coin_distribution_l1232_123297

theorem largest_coin_distribution (n : ℕ) : n ≤ 108 ∧ n < 120 ∧ ∃ (k : ℕ), n = 15 * k + 3 →
  ∀ m : ℕ, m < 120 ∧ ∃ (k : ℕ), m = 15 * k + 3 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_l1232_123297


namespace NUMINAMATH_CALUDE_age_problem_l1232_123209

theorem age_problem (billy joe sarah : ℕ) 
  (h1 : billy = 3 * joe)
  (h2 : billy + joe = 48)
  (h3 : joe + sarah = 30) :
  billy = 36 ∧ joe = 12 ∧ sarah = 18 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1232_123209


namespace NUMINAMATH_CALUDE_count_special_numbers_in_range_l1232_123227

def count_multiples_of_three (n : ℕ) : ℕ :=
  (n / 3 : ℕ)

def count_special_numbers (n : ℕ) : ℕ :=
  ((n + 2) / 12 : ℕ)

theorem count_special_numbers_in_range : 
  count_multiples_of_three 2015 = 671 ∧ 
  count_special_numbers 2015 = 167 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_in_range_l1232_123227


namespace NUMINAMATH_CALUDE_entry_fee_reduction_l1232_123230

theorem entry_fee_reduction (original_fee : ℝ) (sale_increase : ℝ) (visitor_increase : ℝ) :
  original_fee = 1 ∧ 
  sale_increase = 0.2 ∧ 
  visitor_increase = 0.6 →
  ∃ (reduced_fee : ℝ),
    reduced_fee = 1 - 0.375 ∧
    (1 + visitor_increase) * reduced_fee * original_fee = (1 + sale_increase) * original_fee :=
by sorry

end NUMINAMATH_CALUDE_entry_fee_reduction_l1232_123230


namespace NUMINAMATH_CALUDE_equation_solution_l1232_123240

theorem equation_solution : 
  ∃! x : ℝ, (3 / x = 2 / (x - 2)) ∧ (x ≠ 0) ∧ (x ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1232_123240


namespace NUMINAMATH_CALUDE_product_over_sum_minus_four_l1232_123217

theorem product_over_sum_minus_four :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) /
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 - 4) = 362880 / 41 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_minus_four_l1232_123217


namespace NUMINAMATH_CALUDE_total_contribution_proof_l1232_123286

/-- Proves that the total contribution is $1040 given the specified conditions --/
theorem total_contribution_proof (niraj brittany angela : ℕ) : 
  niraj = 80 ∧ 
  brittany = 3 * niraj ∧ 
  angela = 3 * brittany → 
  niraj + brittany + angela = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_contribution_proof_l1232_123286


namespace NUMINAMATH_CALUDE_part_one_part_two_l1232_123224

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one (x : ℝ) :
  let a : ℝ := 1
  let b : ℝ := 1
  f x a b > 8 ↔ (x < -1 ∨ x > 1.5) :=
sorry

-- Part II
theorem part_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) →
  (1/a + 1/b ≥ (3 + 2*Real.sqrt 2) / 2) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ (∀ x, f x a b ≥ 5) ∧ (∃ x, f x a b = 5) ∧ 1/a + 1/b = (3 + 2*Real.sqrt 2) / 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1232_123224


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1232_123290

/-- The focal length of a hyperbola with equation x²/2 - y²/2 = 1 is 2√2 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), x^2/2 - y^2/2 = 1 → 
  f = 2 * Real.sqrt ((x^2/2) + (y^2/2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1232_123290


namespace NUMINAMATH_CALUDE_not_all_exponential_increasing_l1232_123288

theorem not_all_exponential_increasing :
  ¬ (∀ a : ℝ, a > 0 ∧ a ≠ 1 → (∀ x y : ℝ, x < y → a^x < a^y)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_exponential_increasing_l1232_123288


namespace NUMINAMATH_CALUDE_planted_area_fraction_l1232_123205

/-- A right triangle with legs of length 3 and 4 units -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : leg1 = 3 ∧ leg2 = 4

/-- A square placed in the right angle corner of the triangle -/
structure CornerSquare (t : RightTriangle) where
  side_length : ℝ
  in_corner : side_length > 0
  distance_to_hypotenuse : ℝ
  is_correct_distance : distance_to_hypotenuse = 2

theorem planted_area_fraction (t : RightTriangle) (s : CornerSquare t) :
  (t.leg1 * t.leg2 / 2 - s.side_length ^ 2) / (t.leg1 * t.leg2 / 2) = 145 / 147 := by
  sorry

end NUMINAMATH_CALUDE_planted_area_fraction_l1232_123205


namespace NUMINAMATH_CALUDE_group_size_proof_l1232_123265

theorem group_size_proof (total : ℕ) 
  (h1 : (2 : ℚ) / 5 * total = (28 : ℚ) / 100 * total + 96) 
  (h2 : (28 : ℚ) / 100 * total = total - ((2 : ℚ) / 5 * total - 96)) : 
  total = 800 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l1232_123265


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1232_123220

/-- A trapezoid with sides A, B, C, and D -/
structure Trapezoid where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.A + t.B + t.C + t.D

/-- Theorem: The perimeter of the given trapezoid ABCD is 180 units -/
theorem trapezoid_perimeter : 
  ∀ (ABCD : Trapezoid), 
  ABCD.B = 50 → 
  ABCD.A = 30 → 
  ABCD.C = 25 → 
  ABCD.D = 75 → 
  perimeter ABCD = 180 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l1232_123220


namespace NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1232_123211

/-- A seven-digit palindrome is a number of the form abcdcba where a, b, c, d are digits and a ≠ 0 -/
def SevenDigitPalindrome : Type := ℕ

/-- The count of valid digits for the first position of a seven-digit palindrome -/
def FirstDigitCount : ℕ := 9

/-- The count of valid digits for each of the second, third, and fourth positions of a seven-digit palindrome -/
def OtherDigitCount : ℕ := 10

/-- The total number of seven-digit palindromes -/
def TotalSevenDigitPalindromes : ℕ := FirstDigitCount * OtherDigitCount * OtherDigitCount * OtherDigitCount

theorem seven_digit_palindrome_count : TotalSevenDigitPalindromes = 9000 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindrome_count_l1232_123211


namespace NUMINAMATH_CALUDE_max_quadrilateral_intersections_l1232_123248

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Type := Unit

/-- The number of sides in a quadrilateral -/
def num_sides (q : Quadrilateral) : ℕ := 4

/-- The maximum number of intersection points between two quadrilaterals -/
def max_intersection_points (q1 q2 : Quadrilateral) : ℕ :=
  num_sides q1 * num_sides q2

theorem max_quadrilateral_intersections :
  ∀ (q1 q2 : Quadrilateral), max_intersection_points q1 q2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_quadrilateral_intersections_l1232_123248


namespace NUMINAMATH_CALUDE_remainder_problem_l1232_123261

theorem remainder_problem (N : ℕ) (h1 : N = 184) (h2 : N % 15 = 4) : N % 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1232_123261


namespace NUMINAMATH_CALUDE_pentagon_area_difference_l1232_123278

/-- Given a rectangle with dimensions 48 mm and 64 mm, when folded along its diagonal
    to form a pentagon, the area difference between the original rectangle and
    the resulting pentagon is 1200 mm². -/
theorem pentagon_area_difference (a b : ℝ) (ha : a = 48) (hb : b = 64) :
  let rect_area := a * b
  let diag := Real.sqrt (a^2 + b^2)
  let overlap_height := Real.sqrt ((diag/2)^2 - ((b - (b^2 - a^2) / (2 * b))^2))
  let overlap_area := (1/2) * diag * overlap_height
  rect_area - (rect_area - overlap_area) = 1200 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_area_difference_l1232_123278


namespace NUMINAMATH_CALUDE_small_circle_radius_l1232_123221

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of large circle is 10 meters
  (3 * (2 * r) = 2 * R) →  -- three diameters of small circles equal diameter of large circle
  r = 10 / 3 :=  -- radius of small circle is 10/3 meters
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l1232_123221


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l1232_123222

/-- Given a polynomial ax^4 + bx^3 + 45x^2 - 18x + 10 with a factor of 5x^2 - 3x + 2,
    prove that a = 151.25 and b = -98.25 -/
theorem polynomial_factor_coefficients :
  ∀ (a b : ℝ),
  (∃ (c d : ℝ), ∀ (x : ℝ),
    a * x^4 + b * x^3 + 45 * x^2 - 18 * x + 10 =
    (5 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 5)) →
  a = 151.25 ∧ b = -98.25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l1232_123222


namespace NUMINAMATH_CALUDE_royal_family_children_l1232_123255

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The initial age of the king and queen -/
def initial_parent_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The number of sons -/
def num_sons : ℕ := 3

/-- The maximum allowed number of children -/
def max_children : ℕ := 20

theorem royal_family_children :
  (initial_parent_age * 2 + 2 * n = initial_children_age + (d + num_sons) * n) ∧
  (d + num_sons ≤ max_children) →
  (d + num_sons = 7) ∨ (d + num_sons = 9) := by
  sorry

end NUMINAMATH_CALUDE_royal_family_children_l1232_123255


namespace NUMINAMATH_CALUDE_sum_intersections_four_lines_l1232_123269

/-- The number of intersections for a given number of lines -/
def intersections (k : ℕ) : ℕ := 
  if k ≤ 1 then 0
  else Nat.choose k 2

/-- The sum of all possible numbers of intersections for up to 4 lines -/
def sum_intersections : ℕ :=
  (List.range 5).map intersections |>.sum

/-- Theorem: The sum of all possible numbers of intersections for four distinct lines in a plane is 19 -/
theorem sum_intersections_four_lines :
  sum_intersections = 19 := by sorry

end NUMINAMATH_CALUDE_sum_intersections_four_lines_l1232_123269


namespace NUMINAMATH_CALUDE_fraction_operations_l1232_123266

theorem fraction_operations : (3 / 7 : ℚ) / 4 * (1 / 2) = 3 / 56 := by sorry

end NUMINAMATH_CALUDE_fraction_operations_l1232_123266


namespace NUMINAMATH_CALUDE_pizza_pieces_l1232_123281

theorem pizza_pieces (total_people : Nat) (half_eaters : Nat) (three_quarter_eaters : Nat) (pieces_left : Nat) :
  total_people = 4 →
  half_eaters = 2 →
  three_quarter_eaters = 2 →
  pieces_left = 6 →
  ∃ (pieces_per_pizza : Nat),
    pieces_per_pizza * (half_eaters * (1/2) + three_quarter_eaters * (1/4)) = pieces_left ∧
    pieces_per_pizza = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_l1232_123281


namespace NUMINAMATH_CALUDE_parabola_p_value_l1232_123236

/-- Represents a parabola with equation y^2 = 2px and directrix x = -2 -/
structure Parabola where
  p : ℝ
  eq : ∀ x y : ℝ, y^2 = 2 * p * x
  directrix : ∀ x : ℝ, x = -2

/-- The value of p for the given parabola is 4 -/
theorem parabola_p_value (par : Parabola) : par.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l1232_123236


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1232_123202

/-- Given a geometric sequence where the first term is m and the fourth term is n,
    prove that the product of the second and fifth terms is equal to n⋅∛(m⋅n²). -/
theorem geometric_sequence_product (m n : ℝ) (h : m > 0) (h' : n > 0) : 
  let q := (n / m) ^ (1/3)
  let second_term := m * q
  let fifth_term := m * q^4
  second_term * fifth_term = n * (m * n^2)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1232_123202


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l1232_123250

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b * e ≠ 0) (h2 : a / b < c / b - d / e) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (a = x ∨ a = y ∨ a = z) :=
sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l1232_123250


namespace NUMINAMATH_CALUDE_small_sphere_diameter_small_sphere_diameter_value_l1232_123228

/-- The diameter of small spheres fitting in the corners of a cube with a larger sphere inside --/
theorem small_sphere_diameter 
  (cube_side : ℝ) 
  (large_sphere_diameter : ℝ) 
  (h_cube : cube_side = 32) 
  (h_large_sphere : large_sphere_diameter = 30) : 
  ℝ :=
let large_sphere_radius := large_sphere_diameter / 2
let small_sphere_radius := (cube_side * Real.sqrt 3 / 2 - large_sphere_radius) / (Real.sqrt 3 + 1)
2 * small_sphere_radius

theorem small_sphere_diameter_value :
  small_sphere_diameter 32 30 rfl rfl = 63 - 31 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_small_sphere_diameter_small_sphere_diameter_value_l1232_123228


namespace NUMINAMATH_CALUDE_solution_inequality_l1232_123274

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

-- Define the theorem
theorem solution_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x ((1 / m) + (1 / (2 * n))) ≤ 1 + |x - 3| ↔ 1 ≤ x ∧ x ≤ 3) →
  m + 2 * n ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_inequality_l1232_123274


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1232_123204

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∃ x : ℝ, x*(x-3) = 7*(3-x) ↔ x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1232_123204


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_630_l1232_123291

theorem sin_n_eq_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = -180 ∨ n = 180) := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_630_l1232_123291


namespace NUMINAMATH_CALUDE_calculation_proof_l1232_123215

theorem calculation_proof : 2014 * (1 / 19 - 1 / 53) = 68 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1232_123215


namespace NUMINAMATH_CALUDE_lara_flowers_to_mom_l1232_123239

theorem lara_flowers_to_mom (total_flowers grandma_flowers mom_flowers vase_flowers : ℕ) :
  total_flowers = 52 →
  grandma_flowers = mom_flowers + 6 →
  vase_flowers = 16 →
  total_flowers = mom_flowers + grandma_flowers + vase_flowers →
  mom_flowers = 15 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_to_mom_l1232_123239


namespace NUMINAMATH_CALUDE_cricket_average_l1232_123260

theorem cricket_average (A : ℝ) : 
  (11 * (A + 4) = 10 * A + 86) → A = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l1232_123260


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1232_123207

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def vector_dot_product (v w : ℝ × ℝ) : ℝ := sorry

def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := sorry

def vector_length (v : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral) 
  (h_convex : is_convex q) 
  (h_bd : vector_length (vector_add q.B (vector_add q.D (-q.B))) = 2) 
  (h_perp : vector_dot_product (vector_add q.A (vector_add q.C (-q.A))) 
                               (vector_add q.B (vector_add q.D (-q.B))) = 0) 
  (h_sum : vector_dot_product (vector_add (vector_add q.A (vector_add q.B (-q.A))) 
                                          (vector_add q.D (vector_add q.C (-q.D)))) 
                              (vector_add (vector_add q.B (vector_add q.C (-q.B))) 
                                          (vector_add q.A (vector_add q.D (-q.A)))) = 5) : 
  area q = 3 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1232_123207


namespace NUMINAMATH_CALUDE_vasya_drove_two_fifths_l1232_123295

/-- Represents the fraction of total distance driven by each person -/
structure DriverDistances where
  anton : ℝ
  vasya : ℝ
  sasha : ℝ
  dima : ℝ

/-- The conditions of the driving problem -/
def drivingConditions (d : DriverDistances) : Prop :=
  d.anton = d.vasya / 2 ∧
  d.sasha = d.anton + d.dima ∧
  d.dima = 1 / 10 ∧
  d.anton + d.vasya + d.sasha + d.dima = 1

theorem vasya_drove_two_fifths :
  ∀ d : DriverDistances, drivingConditions d → d.vasya = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vasya_drove_two_fifths_l1232_123295


namespace NUMINAMATH_CALUDE_string_length_problem_l1232_123284

/-- The length of strings problem -/
theorem string_length_problem (red white blue : ℝ) : 
  red = 8 → 
  white = 5 * red → 
  blue = 8 * white → 
  blue = 320 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l1232_123284


namespace NUMINAMATH_CALUDE_acute_angle_equality_l1232_123287

theorem acute_angle_equality (α : Real) (h1 : 0 < α ∧ α < Real.pi / 2) 
  (h2 : 1 + (Real.sqrt 3 / Real.tan (80 * Real.pi / 180)) = 1 / Real.sin α) : 
  α = 50 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_equality_l1232_123287


namespace NUMINAMATH_CALUDE_min_value_of_f_l1232_123241

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_of_f :
  ∃ (y : ℝ), (∀ (x : ℝ), x ≥ 0 → f x ≥ y) ∧ (∃ (x : ℝ), x ≥ 0 ∧ f x = y) ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1232_123241


namespace NUMINAMATH_CALUDE_max_profit_multimedia_devices_l1232_123264

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the number of devices -/
def device_constraint (x : ℝ) : Prop := 10 ≤ x ∧ x ≤ 50

theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), device_constraint x ∧
    (∀ y, device_constraint y → profit_function x ≥ profit_function y) ∧
    profit_function x = 19 ∧
    x = 10 := by sorry

end NUMINAMATH_CALUDE_max_profit_multimedia_devices_l1232_123264


namespace NUMINAMATH_CALUDE_rectangle_dimensions_and_area_l1232_123201

theorem rectangle_dimensions_and_area (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 7) →
  (x = (17 + Real.sqrt 349) / 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_and_area_l1232_123201


namespace NUMINAMATH_CALUDE_jane_ice_cream_pudding_cost_difference_l1232_123231

theorem jane_ice_cream_pudding_cost_difference :
  let ice_cream_cones : ℕ := 15
  let pudding_cups : ℕ := 5
  let ice_cream_cost_per_cone : ℕ := 5
  let pudding_cost_per_cup : ℕ := 2
  let total_ice_cream_cost := ice_cream_cones * ice_cream_cost_per_cone
  let total_pudding_cost := pudding_cups * pudding_cost_per_cup
  total_ice_cream_cost - total_pudding_cost = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_ice_cream_pudding_cost_difference_l1232_123231


namespace NUMINAMATH_CALUDE_descending_eight_digit_numbers_count_l1232_123206

/-- The number of eight-digit numbers where each digit (except the last one) 
    is greater than the following digit. -/
def count_descending_eight_digit_numbers : ℕ :=
  Nat.choose 10 2

/-- Theorem stating that the count of eight-digit numbers with descending digits
    is equal to choosing 2 from 10. -/
theorem descending_eight_digit_numbers_count :
  count_descending_eight_digit_numbers = 45 := by
  sorry

end NUMINAMATH_CALUDE_descending_eight_digit_numbers_count_l1232_123206


namespace NUMINAMATH_CALUDE_inverse_variation_l1232_123200

/-- Given that p and q vary inversely, prove that when p = 400, q = 1, 
    given that when p = 800, q = 0.5 -/
theorem inverse_variation (p q : ℝ) (h : p * q = 800 * 0.5) :
  p = 400 → q = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_l1232_123200


namespace NUMINAMATH_CALUDE_decreasing_interval_of_symmetric_quadratic_l1232_123213

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem decreasing_interval_of_symmetric_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x ∈ Set.range (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  a ≠ 0 →
  ∃ (l r : ℝ), l = -2/3 ∧ r = 0 ∧
    ∀ x y, l ≤ x ∧ x < y ∧ y ≤ r → f a b y < f a b x :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_symmetric_quadratic_l1232_123213


namespace NUMINAMATH_CALUDE_yw_approx_6_32_l1232_123243

/-- Triangle XYZ with W on XY -/
structure TriangleXYZW where
  /-- Point X -/
  X : ℝ × ℝ
  /-- Point Y -/
  Y : ℝ × ℝ
  /-- Point Z -/
  Z : ℝ × ℝ
  /-- Point W on XY -/
  W : ℝ × ℝ
  /-- XZ = YZ = 10 -/
  xz_eq_yz : dist X Z = dist Y Z ∧ dist X Z = 10
  /-- W is on XY -/
  w_on_xy : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ W = (1 - t) • X + t • Y
  /-- XW = 5 -/
  xw_eq_5 : dist X W = 5
  /-- ZW = 6 -/
  zw_eq_6 : dist Z W = 6

/-- The length of YW is approximately 6.32 -/
theorem yw_approx_6_32 (t : TriangleXYZW) : 
  abs (dist t.Y t.W - 6.32) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_yw_approx_6_32_l1232_123243


namespace NUMINAMATH_CALUDE_vector_perpendicular_and_acute_angle_l1232_123210

def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

def acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem vector_perpendicular_and_acute_angle (m : ℝ) :
  (perpendicular (λ i => (1/2) * (a i) + (b i)) (λ i => (a i) + m * (b i)) ↔ m = -5/12) ∧
  (acute_angle (λ i => (1/2) * (a i) + (b i)) (λ i => (a i) + m * (b i)) ↔ m > -5/12 ∧ m ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_and_acute_angle_l1232_123210


namespace NUMINAMATH_CALUDE_shaniqua_haircut_price_l1232_123216

/-- The amount Shaniqua makes for each haircut -/
def haircut_price : ℚ := sorry

/-- The amount Shaniqua makes for each style -/
def style_price : ℚ := 25

/-- The total number of haircuts Shaniqua gave -/
def num_haircuts : ℕ := 8

/-- The total number of styles Shaniqua gave -/
def num_styles : ℕ := 5

/-- The total amount Shaniqua made -/
def total_amount : ℚ := 221

theorem shaniqua_haircut_price : 
  haircut_price * num_haircuts + style_price * num_styles = total_amount ∧ 
  haircut_price = 12 := by sorry

end NUMINAMATH_CALUDE_shaniqua_haircut_price_l1232_123216


namespace NUMINAMATH_CALUDE_original_recipe_yield_l1232_123258

/-- Represents a cookie recipe -/
structure Recipe where
  butter : ℝ
  cookies : ℝ

/-- Proves that given a recipe that uses 4 pounds of butter, 
    if 1 pound of butter makes 4 dozen cookies, 
    then the original recipe makes 16 dozen cookies. -/
theorem original_recipe_yield 
  (original : Recipe) 
  (h1 : original.butter = 4) 
  (h2 : ∃ (scaled : Recipe), scaled.butter = 1 ∧ scaled.cookies = 4) : 
  original.cookies = 16 := by
sorry

end NUMINAMATH_CALUDE_original_recipe_yield_l1232_123258


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l1232_123292

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_25_and_n_plus_3_l1232_123292


namespace NUMINAMATH_CALUDE_min_a_for_no_zeros_l1232_123277

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x

noncomputable def g (x : ℝ) : ℝ := x * exp (1 - x)

theorem min_a_for_no_zeros (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1/2 → f a x > 0) ↔ a ≥ 2 - 4 * log 2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_no_zeros_l1232_123277


namespace NUMINAMATH_CALUDE_coin_stack_theorem_l1232_123235

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

def coin_arrangements (n : ℕ) : ℕ := fibonacci (n + 2)

theorem coin_stack_theorem :
  coin_arrangements 10 = 233 :=
by sorry

end NUMINAMATH_CALUDE_coin_stack_theorem_l1232_123235


namespace NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_eq_neg_one_l1232_123233

/-- Given complex numbers z₁ and z₂, prove that if z₁ · z₂ is purely imaginary, then a = -1 -/
theorem product_pure_imaginary_implies_a_eq_neg_one (a : ℝ) :
  let z₁ : ℂ := a - Complex.I
  let z₂ : ℂ := 1 + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_implies_a_eq_neg_one_l1232_123233


namespace NUMINAMATH_CALUDE_z1_div_z2_equals_one_minus_two_i_l1232_123279

-- Define complex numbers z1 and z2
def z1 : ℂ := Complex.mk 2 1
def z2 : ℂ := Complex.mk 0 1

-- Theorem statement
theorem z1_div_z2_equals_one_minus_two_i :
  z1 / z2 = Complex.mk 1 (-2) :=
sorry

end NUMINAMATH_CALUDE_z1_div_z2_equals_one_minus_two_i_l1232_123279


namespace NUMINAMATH_CALUDE_xiaohong_school_distance_l1232_123285

/-- The distance between Xiaohong's home and school -/
def distance : ℝ := 2880

/-- The scheduled arrival time in minutes -/
def scheduled_time : ℝ := 29

theorem xiaohong_school_distance :
  (∃ t : ℝ, 
    distance = 120 * (t - 5) ∧
    distance = 90 * (t + 3)) →
  distance = 2880 :=
by sorry

end NUMINAMATH_CALUDE_xiaohong_school_distance_l1232_123285


namespace NUMINAMATH_CALUDE_nuts_division_proof_l1232_123271

/-- The number of boys dividing nuts -/
def num_boys : ℕ := 4

/-- The number of nuts each boy receives at the end -/
def nuts_per_boy : ℕ := 3 * num_boys

/-- The number of nuts taken by the nth boy -/
def nuts_taken (n : ℕ) : ℕ := 3 * n

/-- The remaining nuts after the nth boy's turn -/
def remaining_nuts (n : ℕ) : ℕ :=
  if n = num_boys then 0
  else 5 * (nuts_per_boy - nuts_taken n)

theorem nuts_division_proof :
  (∀ n : ℕ, n ≤ num_boys → nuts_per_boy = nuts_taken n + remaining_nuts n / 5) ∧
  remaining_nuts num_boys = 0 :=
sorry

end NUMINAMATH_CALUDE_nuts_division_proof_l1232_123271


namespace NUMINAMATH_CALUDE_area_and_inequality_l1232_123294

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x - a) - x + a

-- State the theorem
theorem area_and_inequality (a : ℝ) (h : a > 0) :
  (∃ A : ℝ, A = (8:ℝ)/3 ∧ A = ∫ x in (2*a/3)..(2*a), (f a x - a)) → a = 2 ∧
  (∀ x : ℝ, f a x > x ↔ x < 3*a/4) :=
sorry

end NUMINAMATH_CALUDE_area_and_inequality_l1232_123294


namespace NUMINAMATH_CALUDE_range_of_p_l1232_123237

def h (x : ℝ) : ℝ := 2 * x + 1

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → -1 ≤ p x ∧ p x ≤ 31 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l1232_123237


namespace NUMINAMATH_CALUDE_thirty_divisor_numbers_l1232_123242

def is_valid_number (n : ℕ) : Prop :=
  (n % 30 = 0) ∧ (Nat.divisors n).card = 30

def valid_numbers : Finset ℕ := {720, 1200, 1620, 4050, 7500, 11250}

theorem thirty_divisor_numbers :
  ∀ n : ℕ, is_valid_number n ↔ n ∈ valid_numbers := by
  sorry

end NUMINAMATH_CALUDE_thirty_divisor_numbers_l1232_123242


namespace NUMINAMATH_CALUDE_transform_result_l1232_123275

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90 degrees counterclockwise around (1, 5) -/
def rotate90 (p : Point) : Point :=
  Point.mk (-(p.y - 5) + 1) ((p.x - 1) + 5)

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  Point.mk (-p.y) (-p.x)

/-- The final transformation applied to the initial point -/
def transform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

theorem transform_result (a b : ℝ) : 
  transform (Point.mk a b) = Point.mk (-6) 3 → b - a = 7 := by
  sorry

end NUMINAMATH_CALUDE_transform_result_l1232_123275


namespace NUMINAMATH_CALUDE_grid_rectangles_l1232_123219

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a gridSize × gridSize grid -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem grid_rectangles :
  numRectangles = 36 := by sorry

end NUMINAMATH_CALUDE_grid_rectangles_l1232_123219


namespace NUMINAMATH_CALUDE_even_power_difference_divisible_l1232_123225

theorem even_power_difference_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 → ∃ m : ℤ, x^(2*k) - y^(2*k) = (x + y) * m :=
by sorry

end NUMINAMATH_CALUDE_even_power_difference_divisible_l1232_123225


namespace NUMINAMATH_CALUDE_bobs_total_bushels_l1232_123254

/-- Calculates the number of bushels from a row of corn, rounding down to the nearest whole bushel -/
def bushelsFromRow (stalks : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  stalks / stalksPerBushel

/-- Represents Bob's corn harvest -/
structure CornHarvest where
  row1 : (ℕ × ℕ)
  row2 : (ℕ × ℕ)
  row3 : (ℕ × ℕ)
  row4 : (ℕ × ℕ)
  row5 : (ℕ × ℕ)
  row6 : (ℕ × ℕ)
  row7 : (ℕ × ℕ)

/-- Calculates the total bushels of corn from Bob's harvest -/
def totalBushels (harvest : CornHarvest) : ℕ :=
  bushelsFromRow harvest.row1.1 harvest.row1.2 +
  bushelsFromRow harvest.row2.1 harvest.row2.2 +
  bushelsFromRow harvest.row3.1 harvest.row3.2 +
  bushelsFromRow harvest.row4.1 harvest.row4.2 +
  bushelsFromRow harvest.row5.1 harvest.row5.2 +
  bushelsFromRow harvest.row6.1 harvest.row6.2 +
  bushelsFromRow harvest.row7.1 harvest.row7.2

/-- Bob's actual corn harvest -/
def bobsHarvest : CornHarvest :=
  { row1 := (82, 8)
    row2 := (94, 9)
    row3 := (78, 7)
    row4 := (96, 12)
    row5 := (85, 10)
    row6 := (91, 13)
    row7 := (88, 11) }

theorem bobs_total_bushels :
  totalBushels bobsHarvest = 62 := by
  sorry

end NUMINAMATH_CALUDE_bobs_total_bushels_l1232_123254


namespace NUMINAMATH_CALUDE_longest_chord_in_quarter_circle_l1232_123208

theorem longest_chord_in_quarter_circle (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let chord_length := (2 * r ^ 2) ^ (1/2)
  chord_length ^ 2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_longest_chord_in_quarter_circle_l1232_123208


namespace NUMINAMATH_CALUDE_coin_toss_probability_l1232_123253

-- Define the probability of landing heads
def p : ℚ := 3/5

-- Define the number of tosses
def n : ℕ := 4

-- Define the number of desired heads
def k : ℕ := 2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the probability of getting exactly k heads in n tosses
def probability (p : ℚ) (n k : ℕ) : ℚ :=
  binomial_coeff n k * p^k * (1 - p)^(n - k)

-- State the theorem
theorem coin_toss_probability : probability p n k = 216/625 := by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l1232_123253


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1232_123212

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + m = 0 ∧ x₂^2 - 3*x₂ + m = 0) → 
  m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1232_123212


namespace NUMINAMATH_CALUDE_max_fraction_value_l1232_123218

def is_odd_integer (y : ℝ) : Prop := ∃ (k : ℤ), y = 2 * k + 1

theorem max_fraction_value (x y : ℝ) 
  (hx : -5 ≤ x ∧ x ≤ -3) 
  (hy : 3 ≤ y ∧ y ≤ 5) 
  (hy_odd : is_odd_integer y) : 
  (∀ z, -5 ≤ z ∧ z ≤ -3 → ∀ w, 3 ≤ w ∧ w ≤ 5 → is_odd_integer w → (x + y) / x ≥ (z + w) / z) ∧ 
  (x + y) / x ≤ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l1232_123218


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1232_123293

/-- Represents a geometric sequence with first term a and common ratio q -/
def GeometricSequence (a : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a * q ^ (n - 1)

/-- The common ratio of a geometric sequence satisfying given conditions is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℝ) (q : ℝ) (h_pos : q > 0) :
  let seq := GeometricSequence a q
  (seq 3 - 3 * seq 2 = 2) ∧ 
  (5 * seq 4 = (12 * seq 3 + 2 * seq 5) / 2) →
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1232_123293


namespace NUMINAMATH_CALUDE_simplify_expression_l1232_123203

theorem simplify_expression (x : ℝ) :
  3*x^3 + 4*x + 5*x^2 + 2 - (7 - 3*x^3 - 4*x - 5*x^2) = 6*x^3 + 10*x^2 + 8*x - 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1232_123203


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1232_123263

theorem power_fraction_simplification : 
  (12 : ℕ)^10 / (144 : ℕ)^4 = (144 : ℕ) :=
by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1232_123263


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1232_123262

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties 
  (a₁ : ℤ) 
  (d : ℤ) 
  (h1 : a₁ = 23)
  (h2 : ∀ n : ℕ, n ≤ 6 → arithmetic_sequence a₁ d n > 0)
  (h3 : arithmetic_sequence a₁ d 7 < 0) :
  (d = -4) ∧ 
  (∃ n : ℕ, sum_arithmetic_sequence a₁ d n = 78 ∧ 
    ∀ m : ℕ, sum_arithmetic_sequence a₁ d m ≤ 78) ∧
  (∃ n : ℕ, n = 12 ∧ sum_arithmetic_sequence a₁ d n > 0 ∧ 
    ∀ m : ℕ, m > 12 → sum_arithmetic_sequence a₁ d m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1232_123262


namespace NUMINAMATH_CALUDE_paper_distribution_l1232_123259

theorem paper_distribution (total_sheets : ℕ) (num_printers : ℕ) 
  (h1 : total_sheets = 221) (h2 : num_printers = 31) :
  (total_sheets / num_printers : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paper_distribution_l1232_123259


namespace NUMINAMATH_CALUDE_percentage_problem_l1232_123226

theorem percentage_problem (X : ℝ) : 
  (0.2 * 40 + 0.25 * X = 23) → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1232_123226


namespace NUMINAMATH_CALUDE_product_inequality_l1232_123249

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1232_123249


namespace NUMINAMATH_CALUDE_store_pricing_l1232_123276

/-- Represents the price structure of a store selling chairs, tables, and shelves. -/
structure StorePrice where
  chair : ℝ
  table : ℝ
  shelf : ℝ

/-- Defines the properties of the store's pricing and discount policy. -/
def ValidStorePrice (p : StorePrice) : Prop :=
  p.chair + p.table = 72 ∧
  2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table) ∧
  p.chair + p.table + p.shelf = 95

/-- Calculates the discounted price for a combination of items. -/
def DiscountedPrice (p : StorePrice) : ℝ :=
  0.9 * (p.chair + 2 * p.table + p.shelf)

/-- Theorem stating the correct prices for the store items and the discounted combination. -/
theorem store_pricing (p : StorePrice) (h : ValidStorePrice p) :
  p.table = 63 ∧ DiscountedPrice p = 142.2 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l1232_123276


namespace NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1232_123238

theorem complex_square_in_fourth_quadrant :
  let z : ℂ := 2 - I
  (z^2).re > 0 ∧ (z^2).im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1232_123238


namespace NUMINAMATH_CALUDE_sum_of_squares_given_means_l1232_123270

theorem sum_of_squares_given_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 2 * Real.sqrt 5 →
  a^2 + b^2 = 216 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_given_means_l1232_123270


namespace NUMINAMATH_CALUDE_school_selections_l1232_123298

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem school_selections : 
  (choose 6 3) * (choose 5 2) = 200 := by
sorry

end NUMINAMATH_CALUDE_school_selections_l1232_123298


namespace NUMINAMATH_CALUDE_negation_of_existence_of_real_roots_l1232_123268

theorem negation_of_existence_of_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_of_real_roots_l1232_123268


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l1232_123289

theorem smallest_value_in_range (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  (1 / x ≤ x) ∧ (1 / x ≤ x^2) ∧ (1 / x ≤ 2*x) ∧ (1 / x ≤ Real.sqrt x) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_in_range_l1232_123289


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l1232_123252

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 8 * y = 2 ∧ 4 * y - x = 6 ∧ x = 14 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l1232_123252


namespace NUMINAMATH_CALUDE_dvds_in_book_l1232_123296

/-- Given a DVD book with a total capacity and some empty spaces,
    calculate the number of DVDs already in the book. -/
theorem dvds_in_book (total_capacity : ℕ) (empty_spaces : ℕ)
    (h1 : total_capacity = 126)
    (h2 : empty_spaces = 45) :
    total_capacity - empty_spaces = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvds_in_book_l1232_123296


namespace NUMINAMATH_CALUDE_principal_amount_satisfies_conditions_l1232_123280

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := 6400

/-- The annual interest rate -/
def interest_rate : ℝ := 0.05

/-- The time period in years -/
def time_period : ℝ := 2

/-- The difference between compound interest and simple interest -/
def interest_difference : ℝ := 16

/-- Theorem stating that the principal amount satisfies the given conditions -/
theorem principal_amount_satisfies_conditions :
  let compound_interest := principal_amount * (1 + interest_rate) ^ time_period - principal_amount
  let simple_interest := principal_amount * interest_rate * time_period
  compound_interest - simple_interest = interest_difference :=
by sorry

end NUMINAMATH_CALUDE_principal_amount_satisfies_conditions_l1232_123280


namespace NUMINAMATH_CALUDE_fraction_calculation_l1232_123273

theorem fraction_calculation : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) * (5 / 1 : ℚ) = (34 / 100 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1232_123273


namespace NUMINAMATH_CALUDE_product_equals_143_l1232_123272

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

-- Define a function to convert ternary to decimal
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldr (fun (i, digit) acc => acc + digit * 3^i) 0

-- Define the binary number 1101₂
def binary_1101 : List Bool := [true, false, true, true]

-- Define the ternary number 102₃
def ternary_102 : List Nat := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_1101) * (ternary_to_decimal ternary_102) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_143_l1232_123272


namespace NUMINAMATH_CALUDE_no_solution_mod_seven_l1232_123229

theorem no_solution_mod_seven (m : ℤ) : 
  (0 ≤ m ∧ m ≤ 6) →
  (m = 4 ↔ ∀ x y : ℤ, (3 * x^2 - 10 * x * y - 8 * y^2) % 7 ≠ m % 7) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_mod_seven_l1232_123229


namespace NUMINAMATH_CALUDE_M_subset_N_l1232_123257

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1232_123257


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l1232_123245

theorem tailor_cut_difference : 
  let skirt_cut : ℚ := 7/8
  let pants_cut : ℚ := 5/6
  skirt_cut - pants_cut = 1/24 := by sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l1232_123245


namespace NUMINAMATH_CALUDE_unique_solution_for_A_l1232_123223

/-- Given an equation 1A + 4B3 = 469, where A and B are single digits and 4B3 is a three-digit number,
    prove that A = 6 is the unique solution for A. -/
theorem unique_solution_for_A : ∃! (A : ℕ), ∃ (B : ℕ),
  (A < 10) ∧ (B < 10) ∧ (400 ≤ 4 * 10 * B + 3) ∧ (4 * 10 * B + 3 < 1000) ∧
  (10 * A + 4 * 10 * B + 3 = 469) ∧ A = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_A_l1232_123223


namespace NUMINAMATH_CALUDE_y1_greater_y2_l1232_123232

/-- Given two points A(m-1, y₁) and B(m, y₂) on the line y = -2x + 1, prove that y₁ > y₂ -/
theorem y1_greater_y2 (m : ℝ) (y₁ y₂ : ℝ) 
  (hA : y₁ = -2 * (m - 1) + 1) 
  (hB : y₂ = -2 * m + 1) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l1232_123232


namespace NUMINAMATH_CALUDE_circumcenter_equidistant_closest_vertex_l1232_123244

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices of the triangle
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

-- Theorem: Any point in the plane is closest to one of the three vertices
theorem closest_vertex (t : Triangle) (p : ℝ × ℝ) :
  (distance p t.A ≤ distance p t.B ∧ distance p t.A ≤ distance p t.C) ∨
  (distance p t.B ≤ distance p t.A ∧ distance p t.B ≤ distance p t.C) ∨
  (distance p t.C ≤ distance p t.A ∧ distance p t.C ≤ distance p t.B) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_equidistant_closest_vertex_l1232_123244


namespace NUMINAMATH_CALUDE_alice_age_l1232_123299

/-- The ages of Alice, Bob, and Claire satisfy the given conditions -/
structure AgeRelationship where
  alice : ℕ
  bob : ℕ
  claire : ℕ
  alice_younger_than_bob : alice = bob - 3
  bob_older_than_claire : bob = claire + 5
  claire_age : claire = 12

/-- Alice's age is 14 years old given the age relationships -/
theorem alice_age (ar : AgeRelationship) : ar.alice = 14 := by
  sorry

end NUMINAMATH_CALUDE_alice_age_l1232_123299


namespace NUMINAMATH_CALUDE_not_recurring_decimal_example_l1232_123267

def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ (a b : ℕ) (c : ℕ+), x = (a : ℝ) / b + (c : ℝ) / (10^b * 9)

theorem not_recurring_decimal_example : ¬ is_recurring_decimal 0.89898989 := by
  sorry

end NUMINAMATH_CALUDE_not_recurring_decimal_example_l1232_123267


namespace NUMINAMATH_CALUDE_exists_sequence_satisfying_conditions_l1232_123282

/-- The number of distinct prime factors shared by two positive integers -/
def d (m n : ℕ+) : ℕ := sorry

/-- The existence of a sequence satisfying the given conditions -/
theorem exists_sequence_satisfying_conditions :
  ∃ (a : ℕ+ → ℕ+),
    (a 1 ≥ 2018^2018) ∧
    (∀ m n, m ≤ n → a m ≤ a n) ∧
    (∀ m n, m ≠ n → d m n = d (a m) (a n)) :=
  sorry

end NUMINAMATH_CALUDE_exists_sequence_satisfying_conditions_l1232_123282


namespace NUMINAMATH_CALUDE_talking_segment_duration_l1232_123214

/-- Represents the duration of a radio show in minutes -/
def show_duration : ℕ := 3 * 60

/-- Represents the number of talking segments in the show -/
def num_talking_segments : ℕ := 3

/-- Represents the number of ad breaks in the show -/
def num_ad_breaks : ℕ := 5

/-- Represents the duration of each ad break in minutes -/
def ad_break_duration : ℕ := 5

/-- Represents the total duration of songs played in the show in minutes -/
def song_duration : ℕ := 125

/-- Theorem stating that each talking segment lasts 10 minutes -/
theorem talking_segment_duration :
  (show_duration - num_ad_breaks * ad_break_duration - song_duration) / num_talking_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_talking_segment_duration_l1232_123214


namespace NUMINAMATH_CALUDE_work_completion_indeterminate_l1232_123251

structure WorkScenario where
  men : ℕ
  days : ℕ
  hours_per_day : ℝ

def total_work (scenario : WorkScenario) : ℝ :=
  scenario.men * scenario.days * scenario.hours_per_day

theorem work_completion_indeterminate 
  (scenario1 scenario2 : WorkScenario)
  (h1 : scenario1.men = 8)
  (h2 : scenario1.days = 24)
  (h3 : scenario2.men = 12)
  (h4 : scenario2.days = 16)
  (h5 : scenario1.hours_per_day = scenario2.hours_per_day)
  (h6 : total_work scenario1 = total_work scenario2) :
  ∀ (h : ℝ), ∃ (scenario1' scenario2' : WorkScenario),
    scenario1'.men = scenario1.men ∧
    scenario1'.days = scenario1.days ∧
    scenario2'.men = scenario2.men ∧
    scenario2'.days = scenario2.days ∧
    scenario1'.hours_per_day = h ∧
    scenario2'.hours_per_day = h ∧
    total_work scenario1' = total_work scenario2' :=
sorry

end NUMINAMATH_CALUDE_work_completion_indeterminate_l1232_123251
