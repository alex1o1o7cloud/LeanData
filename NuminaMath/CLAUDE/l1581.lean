import Mathlib

namespace NUMINAMATH_CALUDE_book_division_l1581_158196

theorem book_division (total_books : ℕ) (first_division : ℕ) (second_division : ℕ) (books_per_category : ℕ) 
  (h1 : total_books = 1200)
  (h2 : first_division = 3)
  (h3 : second_division = 4)
  (h4 : books_per_category = 15) :
  (total_books / first_division / second_division / books_per_category) * 
  first_division * second_division = 84 :=
by sorry

end NUMINAMATH_CALUDE_book_division_l1581_158196


namespace NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1581_158183

/-- A hyperbola with focal length 4√3 and one branch intersected by the line y = x - 3 at two points -/
structure Hyperbola where
  /-- The focal length of the hyperbola -/
  focal_length : ℝ
  /-- The line that intersects one branch of the hyperbola at two points -/
  intersecting_line : ℝ → ℝ
  /-- Condition that the focal length is 4√3 -/
  focal_length_cond : focal_length = 4 * Real.sqrt 3
  /-- Condition that the line y = x - 3 intersects one branch at two points -/
  intersecting_line_cond : intersecting_line = fun x => x - 3

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 6 - y^2 / 6 = 1

/-- Theorem stating that the given hyperbola has the equation x²/6 - y²/6 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 6 - y^2 / 6 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_theorem_l1581_158183


namespace NUMINAMATH_CALUDE_parabola_symmetric_point_l1581_158195

/-- Parabola type -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_positive : p > 0
  h_equation : ∀ x y, equation x y ↔ y^2 = 2*p*x
  h_focus : focus = (p/2, 0)

/-- Line type -/
structure Line where
  angle : ℝ
  point : ℝ × ℝ

/-- Symmetric points with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem parabola_symmetric_point
  (C : Parabola)
  (l : Line)
  (h_angle : l.angle = π/6)
  (h_passes : l.point = C.focus)
  (P : ℝ × ℝ)
  (h_on_parabola : C.equation P.1 P.2)
  (h_symmetric : symmetric P (5, 0) l) :
  P.1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_point_l1581_158195


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l1581_158118

/-- In a triangle ABC with ∠C = 120°, given sides a and b, and angle bisector lc,
    the equation 1/a + 1/b = 1/lc holds. -/
theorem triangle_angle_bisector_theorem (a b lc : ℝ) (ha : a > 0) (hb : b > 0) (hlc : lc > 0) :
  let angle_C : ℝ := 120 * Real.pi / 180
  1 / a + 1 / b = 1 / lc :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_theorem_l1581_158118


namespace NUMINAMATH_CALUDE_smallest_four_digit_not_dividing_l1581_158125

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_not_dividing :
  ∃ (n : ℕ), is_four_digit n ∧
    ¬(sum_of_first_n n ∣ product_of_first_n n) ∧
    (∀ m, is_four_digit m ∧ m < n →
      sum_of_first_n m ∣ product_of_first_n m) ∧
    n = 1002 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_not_dividing_l1581_158125


namespace NUMINAMATH_CALUDE_number_of_values_in_calculation_l1581_158115

theorem number_of_values_in_calculation 
  (initial_average : ℝ)
  (correct_average : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 51)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 75) :
  ∃ (n : ℕ), n > 0 ∧ 
    n * initial_average + (correct_value - incorrect_value) = n * correct_average ∧
    n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_values_in_calculation_l1581_158115


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1581_158160

/-- Given a line segment with midpoint (6, -10) and one endpoint (8, 0),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (x + 8) / 2 = 6 ∧ (y + 0) / 2 = -10 → 
  x + y = -16 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l1581_158160


namespace NUMINAMATH_CALUDE_concert_revenue_l1581_158189

theorem concert_revenue (total_attendees : ℕ) (reserved_price unreserved_price : ℚ)
  (reserved_sold unreserved_sold : ℕ) :
  total_attendees = reserved_sold + unreserved_sold →
  reserved_price = 25 →
  unreserved_price = 20 →
  reserved_sold = 246 →
  unreserved_sold = 246 →
  (reserved_sold : ℚ) * reserved_price + (unreserved_sold : ℚ) * unreserved_price = 11070 :=
by sorry

end NUMINAMATH_CALUDE_concert_revenue_l1581_158189


namespace NUMINAMATH_CALUDE_total_days_2004_to_2008_l1581_158198

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_2004_to_2008 :
  totalDaysInRange 2004 2008 = 1827 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2004_to_2008_l1581_158198


namespace NUMINAMATH_CALUDE_root_product_cubic_l1581_158104

theorem root_product_cubic (p q r : ℝ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) ∧ 
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) ∧ 
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by
sorry

end NUMINAMATH_CALUDE_root_product_cubic_l1581_158104


namespace NUMINAMATH_CALUDE_negative_seven_plus_three_l1581_158105

theorem negative_seven_plus_three : (-7 : ℤ) + 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_plus_three_l1581_158105


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1581_158109

theorem diophantine_equation_solution : ∃ (a b c : ℕ+), a^3 + b^4 = c^5 ∧ a = 256 ∧ b = 64 ∧ c = 32 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1581_158109


namespace NUMINAMATH_CALUDE_power_sum_difference_l1581_158169

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) - 3^5 = 58686 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1581_158169


namespace NUMINAMATH_CALUDE_poly5_with_negative_integer_roots_l1581_158199

/-- A polynomial of degree 5 with integer coefficients -/
structure Poly5 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ
  t : ℤ

/-- The polynomial function corresponding to a Poly5 -/
def poly5_func (g : Poly5) : ℝ → ℝ :=
  λ x => x^5 + g.p * x^4 + g.q * x^3 + g.r * x^2 + g.s * x + g.t

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (g : Poly5) : Prop :=
  ∀ x : ℝ, poly5_func g x = 0 → (∃ n : ℤ, x = -n ∧ n > 0)

theorem poly5_with_negative_integer_roots
  (g : Poly5)
  (h1 : all_roots_negative_integers g)
  (h2 : g.p + g.q + g.r + g.s + g.t = 3024) :
  g.t = 1600 := by
  sorry

end NUMINAMATH_CALUDE_poly5_with_negative_integer_roots_l1581_158199


namespace NUMINAMATH_CALUDE_basketball_free_throws_l1581_158177

theorem basketball_free_throws (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →  -- Points from three-point shots are twice the points from two-point shots
  (free_throws = 2 * two_points - 1) →   -- Number of free throws is twice the number of two-point shots minus one
  (2 * two_points + 3 * three_points + free_throws = 89) →  -- Total score is 89 points
  free_throws = 29 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l1581_158177


namespace NUMINAMATH_CALUDE_road_paving_length_l1581_158186

/-- The length of road paved in April, in meters -/
def april_length : ℕ := 480

/-- The difference between March and April paving lengths, in meters -/
def length_difference : ℕ := 160

/-- The total length of road paved in March and April -/
def total_length : ℕ := april_length + (april_length + length_difference)

theorem road_paving_length : total_length = 1120 := by sorry

end NUMINAMATH_CALUDE_road_paving_length_l1581_158186


namespace NUMINAMATH_CALUDE_linear_system_solution_l1581_158134

theorem linear_system_solution (a b : ℝ) (h1 : 2*a + 3*b = 7) (h2 : 3*a + 2*b = 8) :
  ∃ (m n : ℝ), a*(m+n) + b*(m-n) = 7 ∧ b*(m+n) + a*(m-n) = 8 ∧ m = 5/2 ∧ n = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1581_158134


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1581_158133

theorem ellipse_eccentricity (b : ℝ) : 
  b > 0 → 
  (∀ x y : ℝ, x^2 + y^2 / (b^2 + 1) = 1 → 
    b / Real.sqrt (b^2 + 1) = Real.sqrt 10 / 10) → 
  b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1581_158133


namespace NUMINAMATH_CALUDE_regular_heptagon_diagonal_relation_l1581_158148

/-- Regular heptagon with side length a, diagonal spanning two sides c, and diagonal spanning three sides d -/
structure RegularHeptagon where
  a : ℝ  -- side length
  c : ℝ  -- length of diagonal spanning two sides
  d : ℝ  -- length of diagonal spanning three sides

/-- Theorem: In a regular heptagon, d^2 = c^2 + a^2 -/
theorem regular_heptagon_diagonal_relation (h : RegularHeptagon) : h.d^2 = h.c^2 + h.a^2 := by
  sorry

end NUMINAMATH_CALUDE_regular_heptagon_diagonal_relation_l1581_158148


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1581_158122

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1581_158122


namespace NUMINAMATH_CALUDE_cupboard_sale_percentage_l1581_158192

def cost_price : ℝ := 6875
def additional_amount : ℝ := 1650
def profit_percentage : ℝ := 12

theorem cupboard_sale_percentage (selling_price : ℝ) 
  (h1 : selling_price + additional_amount = cost_price * (1 + profit_percentage / 100)) :
  (cost_price - selling_price) / cost_price * 100 = profit_percentage := by
sorry

end NUMINAMATH_CALUDE_cupboard_sale_percentage_l1581_158192


namespace NUMINAMATH_CALUDE_minus_six_otimes_minus_two_l1581_158119

-- Define the new operation ⊗
def otimes (a b : ℚ) : ℚ := a^2 + b

-- Theorem statement
theorem minus_six_otimes_minus_two : otimes (-6) (-2) = 34 := by sorry

end NUMINAMATH_CALUDE_minus_six_otimes_minus_two_l1581_158119


namespace NUMINAMATH_CALUDE_line_intersects_circle_iff_abs_b_le_sqrt2_l1581_158162

/-- The line y=x+b has common points with the circle x²+y²=1 if and only if |b| ≤ √2. -/
theorem line_intersects_circle_iff_abs_b_le_sqrt2 (b : ℝ) : 
  (∃ (x y : ℝ), y = x + b ∧ x^2 + y^2 = 1) ↔ |b| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_iff_abs_b_le_sqrt2_l1581_158162


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1581_158168

/-- The proposition "If a and b are both even, then the sum of a and b is even" -/
def original_proposition (a b : ℤ) : Prop :=
  (Even a ∧ Even b) → Even (a + b)

/-- The contrapositive of the original proposition -/
def contrapositive (a b : ℤ) : Prop :=
  ¬Even (a + b) → ¬(Even a ∧ Even b)

/-- Theorem stating that the contrapositive is equivalent to "If the sum of a and b is not even, then a and b are not both even" -/
theorem contrapositive_equivalence :
  ∀ a b : ℤ, contrapositive a b ↔ (¬Even (a + b) → ¬(Even a ∧ Even b)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1581_158168


namespace NUMINAMATH_CALUDE_f_at_neg_one_l1581_158170

/-- The function f(x) = x^3 + x^2 - 2x -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x

/-- Theorem: f(-1) = 2 -/
theorem f_at_neg_one : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_neg_one_l1581_158170


namespace NUMINAMATH_CALUDE_other_x_intercept_l1581_158135

/-- Definition of an ellipse with given foci and one x-intercept -/
def ellipse (f1 f2 x1 : ℝ × ℝ) :=
  f1 = (0, 3) ∧ f2 = (4, 0) ∧ x1 = (0, 0)

/-- The sum of distances from any point on the ellipse to the foci is constant -/
def ellipse_property (p : ℝ × ℝ) (f1 f2 : ℝ × ℝ) :=
  Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
  Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) =
  Real.sqrt (f1.1^2 + f1.2^2) + Real.sqrt (f2.1^2 + f2.2^2)

/-- Theorem: The other x-intercept of the ellipse is at (56/11, 0) -/
theorem other_x_intercept (f1 f2 x1 : ℝ × ℝ) :
  ellipse f1 f2 x1 →
  ∃ x2 : ℝ × ℝ, x2 = (56/11, 0) ∧
    ellipse_property x2 f1 f2 ∧
    x2.2 = 0 ∧ x2 ≠ x1 :=
sorry

end NUMINAMATH_CALUDE_other_x_intercept_l1581_158135


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l1581_158187

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l1581_158187


namespace NUMINAMATH_CALUDE_scientific_notation_of_189100_l1581_158156

/-- The scientific notation of 189100 is 1.891 × 10^5 -/
theorem scientific_notation_of_189100 :
  (189100 : ℝ) = 1.891 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_189100_l1581_158156


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l1581_158141

/-- Given a triangle PQR with side lengths, prove that the area of triangle MOI is 11/4 -/
theorem area_of_triangle_MOI (P Q R O I M : ℝ × ℝ) : 
  let pq : ℝ := 10
  let pr : ℝ := 8
  let qr : ℝ := 6
  -- O is the circumcenter
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = (O.1 - Q.1)^2 + (O.2 - Q.2)^2 ∧
  (O.1 - Q.1)^2 + (O.2 - Q.2)^2 = (O.1 - R.1)^2 + (O.2 - R.2)^2 →
  -- I is the incenter
  (I.1 - P.1) / pq + (I.1 - Q.1) / qr + (I.1 - R.1) / pr = 0 ∧
  (I.2 - P.2) / pq + (I.2 - Q.2) / qr + (I.2 - R.2) / pr = 0 →
  -- M is the center of a circle tangent to PR, QR, and the circumcircle
  ∃ (r : ℝ), 
    r = (M.1 - P.1)^2 + (M.2 - P.2)^2 ∧
    r = (M.1 - R.1)^2 + (M.2 - R.2)^2 ∧
    r + ((O.1 - M.1)^2 + (O.2 - M.2)^2).sqrt = (O.1 - P.1)^2 + (O.2 - P.2)^2 →
  -- Area of triangle MOI is 11/4
  abs ((O.1 * (I.2 - M.2) + I.1 * (M.2 - O.2) + M.1 * (O.2 - I.2)) / 2) = 11/4 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l1581_158141


namespace NUMINAMATH_CALUDE_fraction_not_whole_number_l1581_158144

theorem fraction_not_whole_number : 
  (∃ n : ℕ, 60 / 12 = n) ∧ 
  (∀ n : ℕ, 60 / 8 ≠ n) ∧ 
  (∃ n : ℕ, 60 / 5 = n) ∧ 
  (∃ n : ℕ, 60 / 4 = n) ∧ 
  (∃ n : ℕ, 60 / 3 = n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_whole_number_l1581_158144


namespace NUMINAMATH_CALUDE_big_dig_copper_production_l1581_158173

/-- Represents a mine with its daily ore production and copper percentage -/
structure Mine where
  daily_production : ℝ
  copper_percentage : ℝ

/-- Calculates the total daily copper production from all mines -/
def total_copper_production (mines : List Mine) : ℝ :=
  mines.foldl (fun acc mine => acc + mine.daily_production * mine.copper_percentage) 0

/-- Theorem stating the total daily copper production from all four mines -/
theorem big_dig_copper_production :
  let mine_a : Mine := { daily_production := 4500, copper_percentage := 0.055 }
  let mine_b : Mine := { daily_production := 6000, copper_percentage := 0.071 }
  let mine_c : Mine := { daily_production := 5000, copper_percentage := 0.147 }
  let mine_d : Mine := { daily_production := 3500, copper_percentage := 0.092 }
  let all_mines : List Mine := [mine_a, mine_b, mine_c, mine_d]
  total_copper_production all_mines = 1730.5 := by
  sorry


end NUMINAMATH_CALUDE_big_dig_copper_production_l1581_158173


namespace NUMINAMATH_CALUDE_fraction_ordering_l1581_158123

theorem fraction_ordering : (6 : ℚ) / 23 < 8 / 25 ∧ 8 / 25 < 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1581_158123


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1581_158126

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1581_158126


namespace NUMINAMATH_CALUDE_equation_solution_l1581_158108

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1581_158108


namespace NUMINAMATH_CALUDE_total_holidays_in_year_l1581_158181

def holidays : List Nat := [4, 3, 5, 3, 4, 2, 5, 3, 4, 3, 5, 4]

theorem total_holidays_in_year : holidays.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_holidays_in_year_l1581_158181


namespace NUMINAMATH_CALUDE_max_surrounding_sum_l1581_158174

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List ℕ) : Prop := l.Nodup

/-- Checks if the product of three numbers equals 3240 -/
def product_is_3240 (a b c : ℕ) : Prop := a * b * c = 3240

/-- Checks if a grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  g 1 1 = 45 ∧
  (∀ i j k, (i = 0 ∧ j = k) ∨ (i = 2 ∧ j = k) ∨ (j = 0 ∧ i = k) ∨ (j = 2 ∧ i = k) ∨
            (i + j = 2 ∧ k = 1) ∨ (i = j ∧ k = 1) →
            product_is_3240 (g i j) (g i k) (g j k)) ∧
  all_distinct [g 0 0, g 0 1, g 0 2, g 1 0, g 1 2, g 2 0, g 2 1, g 2 2]

/-- Sum of the eight numbers surrounding the center in a grid -/
def surrounding_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 1 + g 0 2 + g 1 0 + g 1 2 + g 2 0 + g 2 1 + g 2 2

/-- The theorem stating the maximum sum of surrounding numbers -/
theorem max_surrounding_sum :
  ∀ g : Grid, valid_grid g → surrounding_sum g ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_max_surrounding_sum_l1581_158174


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1581_158190

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two vectors are parallel -/
def parallel (v1 v2 : Vector3D) : Prop := sorry

/-- A vector is a unit vector -/
def is_unit_vector (v : Vector3D) : Prop := sorry

/-- Two vectors are collinear -/
def collinear (v1 v2 : Vector3D) : Prop := sorry

/-- The zero vector -/
def zero_vector : Vector3D := ⟨0, 0, 0⟩

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  ¬(∀ (v1 v2 : Vector3D) (p : Point3D), is_unit_vector v1 → is_unit_vector v2 → v1.x = p.x ∧ v1.y = p.y ∧ v1.z = p.z → v2.x = p.x ∧ v2.y = p.y ∧ v2.z = p.z) ∧ 
  ¬(∀ (A B C D : Point3D), parallel ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩ ⟨D.x - C.x, D.y - C.y, D.z - C.z⟩ → 
    ∃ (t : ℝ), C = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  ¬(∀ (a b c : Vector3D), parallel a b → parallel b c → b ≠ zero_vector → parallel a c) ∧
  ¬(∀ (v1 v2 : Vector3D) (A B C D : Point3D), 
    collinear v1 v2 → 
    v1.x = B.x - A.x ∧ v1.y = B.y - A.y ∧ v1.z = B.z - A.z →
    v2.x = D.x - C.x ∧ v2.y = D.y - C.y ∧ v2.z = D.z - C.z →
    A ≠ C → B ≠ D) :=
by sorry

end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l1581_158190


namespace NUMINAMATH_CALUDE_starship_sales_l1581_158171

theorem starship_sales (starship_price mech_price ultimate_price : ℕ)
                       (total_items total_revenue : ℕ) :
  starship_price = 8 →
  mech_price = 26 →
  ultimate_price = 33 →
  total_items = 31 →
  total_revenue = 370 →
  ∃ (x y : ℕ),
    x + y ≤ total_items ∧
    (total_items - x - y) % 2 = 0 ∧
    x * starship_price + y * mech_price + 
      ((total_items - x - y) / 2) * ultimate_price = total_revenue ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_starship_sales_l1581_158171


namespace NUMINAMATH_CALUDE_photo_arrangements_l1581_158150

/-- The number of arrangements for four students (two boys and two girls) in a row,
    where the two girls must stand next to each other. -/
def arrangements_count : ℕ := 12

/-- The number of ways to arrange two girls next to each other. -/
def girls_arrangement : ℕ := 2

/-- The number of ways to arrange three entities (two boys and the pair of girls). -/
def entities_arrangement : ℕ := 6

theorem photo_arrangements :
  arrangements_count = girls_arrangement * entities_arrangement :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1581_158150


namespace NUMINAMATH_CALUDE_prob_ace_king_heart_value_l1581_158113

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Probability of drawing Ace of clubs, King of clubs, and any heart in that order -/
def prob_ace_king_heart : ℚ :=
  1 / StandardDeck *
  1 / (StandardDeck - 1) *
  NumHearts / (StandardDeck - 2)

/-- Theorem stating the probability of drawing Ace of clubs, King of clubs, and any heart -/
theorem prob_ace_king_heart_value : prob_ace_king_heart = 13 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_heart_value_l1581_158113


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1581_158149

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l1581_158149


namespace NUMINAMATH_CALUDE_partition_equal_product_l1581_158165

def numbers : List Nat := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

def target_product : Nat := 349188840

theorem partition_equal_product :
  ∃ (A B : List Nat),
    A.length = 5 ∧
    B.length = 5 ∧
    A ∪ B = numbers ∧
    A ∩ B = [] ∧
    A.prod = target_product ∧
    B.prod = target_product :=
by
  sorry

end NUMINAMATH_CALUDE_partition_equal_product_l1581_158165


namespace NUMINAMATH_CALUDE_system_solution_l1581_158110

theorem system_solution :
  ∀ (a b c d n m : ℚ),
    a / 7 + b / 8 = n →
    b = 3 * a - 2 →
    c / 9 + d / 10 = m →
    d = 4 * c + 1 →
    a = 3 →
    c = 2 →
    n = 73 / 56 ∧ m = 101 / 90 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1581_158110


namespace NUMINAMATH_CALUDE_card_partition_theorem_l1581_158194

/-- Represents a card with a number written on it -/
structure Card where
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- The sum of numbers on a stack of cards -/
def stackSum (s : Stack) : Nat :=
  s.map (λ c => c.number) |>.sum

theorem card_partition_theorem (n k : Nat) (cards : List Card) :
  (∀ c ∈ cards, c.number ≤ n) →
  (cards.map (λ c => c.number)).sum = k * n.factorial →
  ∃ (partition : List Stack),
    partition.length = k ∧
    partition.all (λ s => stackSum s = n.factorial) ∧
    partition.join = cards :=
  sorry

end NUMINAMATH_CALUDE_card_partition_theorem_l1581_158194


namespace NUMINAMATH_CALUDE_polygon_with_30_degree_exterior_angles_has_12_sides_l1581_158128

/-- A polygon with exterior angles each measuring 30° has 12 sides -/
theorem polygon_with_30_degree_exterior_angles_has_12_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_30_degree_exterior_angles_has_12_sides_l1581_158128


namespace NUMINAMATH_CALUDE_quadratic_sum_l1581_158166

/-- A quadratic function with specific properties -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function g(x) = ax^2 + bx + c with vertex at (-2, 6) 
    and passing through (0, 2), the value of a + 2b + c is -7 -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, g a b c x = a * x^2 + b * x + c) →  -- Definition of g
  g a b c (-2) = 6 →                        -- Vertex at (-2, 6)
  (∀ x, g a b c x ≤ 6) →                   -- (-2, 6) is the maximum point
  g a b c 0 = 2 →                          -- Point (0, 2) on the graph
  a + 2*b + c = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1581_158166


namespace NUMINAMATH_CALUDE_grain_storage_capacity_l1581_158106

theorem grain_storage_capacity (total_bins : ℕ) (large_bin_capacity : ℕ) (total_capacity : ℕ) (num_large_bins : ℕ) :
  total_bins = 30 →
  large_bin_capacity = 20 →
  total_capacity = 510 →
  num_large_bins = 12 →
  ∃ (small_bin_capacity : ℕ),
    small_bin_capacity * (total_bins - num_large_bins) + large_bin_capacity * num_large_bins = total_capacity ∧
    small_bin_capacity = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_grain_storage_capacity_l1581_158106


namespace NUMINAMATH_CALUDE_max_n_value_l1581_158102

theorem max_n_value (A B : ℤ) (h : A * B = 54) : 
  ∃ (n : ℤ), n = 3 * B + A ∧ ∀ (m : ℤ), m = 3 * B + A → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l1581_158102


namespace NUMINAMATH_CALUDE_power_difference_equals_one_third_l1581_158151

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 200 ∧ ∀ k > x, ¬(2^k ∣ 200)

def is_greatest_power_of_5_factor (y : ℕ) : Prop :=
  5^y ∣ 200 ∧ ∀ k > y, ¬(5^k ∣ 200)

theorem power_difference_equals_one_third
  (x y : ℕ)
  (h2 : is_greatest_power_of_2_factor x)
  (h5 : is_greatest_power_of_5_factor y) :
  (1/3 : ℚ)^(x - y) = 1/3 := by sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_third_l1581_158151


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1581_158155

/-- A geometric sequence with a_3 = 4 and a_6 = 1/2 has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence property
  a 3 = 4 →                         -- given condition
  a 6 = 1 / 2 →                     -- given condition
  q = 1 / 2 :=                      -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1581_158155


namespace NUMINAMATH_CALUDE_philips_banana_groups_l1581_158180

/-- Given Philip's fruit collection, prove the number of banana groups -/
theorem philips_banana_groups
  (total_oranges : ℕ) (total_bananas : ℕ)
  (orange_groups : ℕ) (oranges_per_group : ℕ)
  (h1 : total_oranges = 384)
  (h2 : total_bananas = 192)
  (h3 : orange_groups = 16)
  (h4 : oranges_per_group = 24)
  (h5 : total_oranges = orange_groups * oranges_per_group)
  : total_bananas / oranges_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l1581_158180


namespace NUMINAMATH_CALUDE_power_of_two_in_product_l1581_158182

theorem power_of_two_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = k * 3^3 * 11^2) →  -- The product has 3^3 and 11^2 as factors
  (∀ x : ℕ+, x < 132 → ¬∃ k : ℕ, 936 * x = k * 3^3 * 11^2) →  -- 132 is the smallest possible w
  (∃ m : ℕ, 936 * w = 2^5 * m ∧ m % 2 ≠ 0) :=  -- The highest power of 2 dividing the product is 2^5
by sorry

end NUMINAMATH_CALUDE_power_of_two_in_product_l1581_158182


namespace NUMINAMATH_CALUDE_sqrt_power_eight_equals_390625_l1581_158127

theorem sqrt_power_eight_equals_390625 :
  (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 8 = 390625 := by sorry

end NUMINAMATH_CALUDE_sqrt_power_eight_equals_390625_l1581_158127


namespace NUMINAMATH_CALUDE_correct_swap_l1581_158130

-- Define the initial values
def a : Int := 2
def b : Int := -6

-- Define the swap operation
def swap (x y : Int) : (Int × Int) :=
  let c := x
  let new_x := y
  let new_y := c
  (new_x, new_y)

-- Theorem statement
theorem correct_swap :
  swap a b = (-6, 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_swap_l1581_158130


namespace NUMINAMATH_CALUDE_benzene_formation_enthalpy_l1581_158111

-- Define the substances
def C : Type := Unit
def H₂ : Type := Unit
def C₂H₂ : Type := Unit
def C₆H₆ : Type := Unit

-- Define the states
inductive State
| Gas
| Liquid
| Graphite

-- Define a reaction
structure Reaction :=
  (reactants : List (Type × State × ℕ))
  (products : List (Type × State × ℕ))
  (heat_effect : ℝ)

-- Given reactions
def reaction1 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 1)], [(C, State.Graphite, 2), (H₂, State.Gas, 1)], 226.7⟩

def reaction2 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 3)], [(C₆H₆, State.Liquid, 1)], 631.1⟩

def reaction3 : Reaction :=
  ⟨[(C₆H₆, State.Liquid, 1)], [(C₆H₆, State.Liquid, 1)], -33.9⟩

-- Standard enthalpy of formation
def standard_enthalpy_of_formation (substance : Type) (state : State) : ℝ := sorry

-- Theorem statement
theorem benzene_formation_enthalpy :
  standard_enthalpy_of_formation C₆H₆ State.Liquid = -82.9 :=
sorry

end NUMINAMATH_CALUDE_benzene_formation_enthalpy_l1581_158111


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1581_158124

-- Define the properties of function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_nonneg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f (2 * x - 1) > 0} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1581_158124


namespace NUMINAMATH_CALUDE_race_head_start_l1581_158132

/-- Proof of head start time in a race --/
theorem race_head_start 
  (race_distance : ℝ) 
  (cristina_speed : ℝ) 
  (nicky_speed : ℝ) 
  (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : catch_up_time = 30) :
  let distance_covered := nicky_speed * catch_up_time
  let cristina_time := distance_covered / cristina_speed
  let head_start := catch_up_time - cristina_time
  head_start = 12 := by sorry

end NUMINAMATH_CALUDE_race_head_start_l1581_158132


namespace NUMINAMATH_CALUDE_g_sum_zero_l1581_158131

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l1581_158131


namespace NUMINAMATH_CALUDE_line_equations_l1581_158121

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Function to check if two lines are perpendicular
def linesPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem line_equations :
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨1, 1⟩
  let l1 : Line2D := ⟨1, 1, -1⟩  -- x + y - 1 = 0
  let l2 : Line2D := ⟨3, 1, -1⟩  -- 3x + y - 1 = 0
  let result1 : Line2D := ⟨1, 1, -3⟩  -- x + y - 3 = 0
  let result2 : Line2D := ⟨1, -3, 2⟩  -- x - 3y + 2 = 0
  (pointOnLine p1 result1 ∧ linesParallel result1 l1) ∧
  (pointOnLine p2 result2 ∧ linesPerpendicular result2 l2) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l1581_158121


namespace NUMINAMATH_CALUDE_total_chips_eaten_l1581_158188

/-- 
Given that John eats x bags of chips for dinner and 2x bags after dinner,
prove that the total number of bags eaten is 3x.
-/
theorem total_chips_eaten (x : ℕ) : x + 2*x = 3*x := by
  sorry

end NUMINAMATH_CALUDE_total_chips_eaten_l1581_158188


namespace NUMINAMATH_CALUDE_books_sold_in_garage_sale_l1581_158184

theorem books_sold_in_garage_sale :
  ∀ (initial_books given_to_friend remaining_books : ℕ),
    initial_books = 108 →
    given_to_friend = 35 →
    remaining_books = 62 →
    initial_books - given_to_friend - remaining_books = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_books_sold_in_garage_sale_l1581_158184


namespace NUMINAMATH_CALUDE_b_work_days_l1581_158147

/-- Proves that B worked for 10 days before leaving the job -/
theorem b_work_days (a_total : ℕ) (b_total : ℕ) (a_remaining : ℕ) : 
  a_total = 21 → b_total = 15 → a_remaining = 7 → 
  ∃ (b_days : ℕ), b_days = 10 ∧ 
    (b_days : ℚ) / b_total + a_remaining / a_total = 1 :=
by sorry

end NUMINAMATH_CALUDE_b_work_days_l1581_158147


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1581_158185

/-- A hyperbola with parameter b > 0 -/
structure Hyperbola (b : ℝ) : Prop where
  pos : b > 0

/-- A line with equation x + 3y - 1 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + 3 * p.2 - 1 = 0}

/-- The left branch of the hyperbola -/
def LeftBranch (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 < 0 ∧ p.1^2 / 4 - p.2^2 / b^2 = 1}

/-- Predicate for line intersecting the left branch of hyperbola -/
def Intersects (b : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ Line ∩ LeftBranch b

/-- Theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem sufficient_not_necessary (h : Hyperbola b) :
    (b > 1 → Intersects b) ∧ ¬(Intersects b → b > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1581_158185


namespace NUMINAMATH_CALUDE_goldbach_2024_l1581_158116

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  p + q = 2024 ∧ 
  is_prime p ∧ 
  is_prime q ∧ 
  (p > 1000 ∨ q > 1000) :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l1581_158116


namespace NUMINAMATH_CALUDE_price_reduction_l1581_158175

theorem price_reduction (initial_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  first_reduction = 0.15 →
  second_reduction = 0.20 →
  let price_after_first := initial_price * (1 - first_reduction)
  let price_after_second := price_after_first * (1 - second_reduction)
  initial_price > 0 →
  (initial_price - price_after_second) / initial_price = 0.32 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l1581_158175


namespace NUMINAMATH_CALUDE_complex_power_sum_l1581_158164

/-- Given that z = (i - 1) / √2, prove that z^100 + z^50 + 1 = -i -/
theorem complex_power_sum (z : ℂ) : z = (Complex.I - 1) / Real.sqrt 2 → z^100 + z^50 + 1 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1581_158164


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1581_158143

theorem triangle_angle_c (A B C : Real) (h1 : 2 * Real.sin A + 5 * Real.cos B = 5) 
  (h2 : 5 * Real.sin B + 2 * Real.cos A = 2) 
  (h3 : A + B + C = Real.pi) : 
  C = Real.arcsin (1/5) ∨ C = Real.pi - Real.arcsin (1/5) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1581_158143


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l1581_158103

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem ninth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 3 = 20 →
  arithmetic_sequence a d 6 = 26 →
  arithmetic_sequence a d 9 = 32 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l1581_158103


namespace NUMINAMATH_CALUDE_trig_identity_l1581_158154

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : 0 < a) (h' : 0 < b) 
  (h_identity : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = 1 / (a + b)^5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1581_158154


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1581_158120

theorem lcm_gcd_problem :
  let a₁ := 5^2 * 7^4
  let b₁ := 490 * 175
  let a₂ := 2^5 * 3 * 7
  let b₂ := 3^4 * 5^4 * 7^2
  let c₂ := 10000
  (Nat.gcd a₁ b₁ = 8575 ∧ Nat.lcm a₁ b₁ = 600250) ∧
  (Nat.gcd a₂ (Nat.gcd b₂ c₂) = 1 ∧ Nat.lcm a₂ (Nat.lcm b₂ c₂) = 793881600) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1581_158120


namespace NUMINAMATH_CALUDE_mysoon_ornament_collection_l1581_158191

theorem mysoon_ornament_collection :
  ∀ (O : ℕ), 
    (O / 6 + 10 : ℕ) = (O / 3 : ℕ) * 2 →  -- Condition 1 and 2 combined
    (O / 3 : ℕ) = O / 3 →                 -- Condition 3
    O = 20 := by
  sorry

end NUMINAMATH_CALUDE_mysoon_ornament_collection_l1581_158191


namespace NUMINAMATH_CALUDE_special_number_subtraction_units_digit_l1581_158179

/-- Represents a three-digit number with specific digit relationships -/
structure SpecialNumber where
  units : ℕ
  tens : ℕ
  hundreds : ℕ
  tens_double_units : tens = 2 * units
  hundreds_three_more : hundreds = units + 3

/-- Calculates the numeric value of a SpecialNumber -/
def value (n : SpecialNumber) : ℕ := 100 * n.hundreds + 10 * n.tens + n.units

/-- Calculates the numeric value of the reversed SpecialNumber -/
def reversed_value (n : SpecialNumber) : ℕ := 100 * n.units + 10 * n.tens + n.hundreds

/-- The main theorem to be proved -/
theorem special_number_subtraction_units_digit (n : SpecialNumber) :
  (value n - reversed_value n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_special_number_subtraction_units_digit_l1581_158179


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_l1581_158139

theorem sqrt_sum_equals_sqrt (n : ℕ+) :
  (∃ x y : ℕ+, Real.sqrt x + Real.sqrt y = Real.sqrt n) ↔
  (∃ p q : ℕ, p > 1 ∧ n = p^2 * q) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_l1581_158139


namespace NUMINAMATH_CALUDE_range_of_a_l1581_158158

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 1)^2 > 4
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1 ↔ (∀ x : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a)))) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1581_158158


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_ten_l1581_158167

theorem cubic_fraction_equals_ten (a b : ℝ) (ha : a = 7) (hb : b = 3) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_ten_l1581_158167


namespace NUMINAMATH_CALUDE_divisible_by_117_and_2_less_than_2011_l1581_158157

theorem divisible_by_117_and_2_less_than_2011 : 
  (Finset.filter (fun n => n < 2011 ∧ n % 117 = 0 ∧ n % 2 = 0) (Finset.range 2011)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_117_and_2_less_than_2011_l1581_158157


namespace NUMINAMATH_CALUDE_sarah_brings_nine_photos_l1581_158152

/-- The number of photos Sarah brings to fill a photo album -/
def sarahs_photos (total_slots : ℕ) (cristina_photos : ℕ) (john_photos : ℕ) (clarissa_photos : ℕ) : ℕ :=
  total_slots - (cristina_photos + john_photos + clarissa_photos)

/-- Theorem stating that Sarah brings 9 photos given the conditions in the problem -/
theorem sarah_brings_nine_photos :
  sarahs_photos 40 7 10 14 = 9 := by
  sorry

#eval sarahs_photos 40 7 10 14

end NUMINAMATH_CALUDE_sarah_brings_nine_photos_l1581_158152


namespace NUMINAMATH_CALUDE_vector_collinearity_l1581_158159

/-- Two vectors in R² -/
def PA : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def PB (x : ℝ) : Fin 2 → ℝ := ![2, x]

/-- Collinearity condition for three points in R² -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 - v 1 * w 0 = 0

theorem vector_collinearity (x : ℝ) : 
  collinear PA (PB x) → x = -4 := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1581_158159


namespace NUMINAMATH_CALUDE_tangent_condition_l1581_158145

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

/-- The circle equation -/
def circle_equation (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 2

/-- The line is tangent to the circle -/
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_equation x y ∧ circle_equation x y a b ∧
  ∀ x' y' : ℝ, line_equation x' y' → circle_equation x' y' a b → (x', y') = (x, y)

theorem tangent_condition (a b : ℝ) :
  (a + b = 1 → is_tangent a b) ∧
  (∃ a' b' : ℝ, is_tangent a' b' ∧ a' + b' ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l1581_158145


namespace NUMINAMATH_CALUDE_extra_fruits_l1581_158193

def red_apples_ordered : ℕ := 60
def green_apples_ordered : ℕ := 34
def bananas_ordered : ℕ := 25
def oranges_ordered : ℕ := 45

def red_apple_students : ℕ := 3
def green_apple_students : ℕ := 2
def banana_students : ℕ := 5
def orange_students : ℕ := 10

def red_apples_per_student : ℕ := 2
def green_apples_per_student : ℕ := 2
def bananas_per_student : ℕ := 2
def oranges_per_student : ℕ := 1

theorem extra_fruits :
  red_apples_ordered - red_apple_students * red_apples_per_student +
  green_apples_ordered - green_apple_students * green_apples_per_student +
  bananas_ordered - banana_students * bananas_per_student +
  oranges_ordered - orange_students * oranges_per_student = 134 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruits_l1581_158193


namespace NUMINAMATH_CALUDE_min_cost_at_zero_min_cost_value_l1581_158142

/-- Represents a transportation plan for machines between two locations --/
structure TransportPlan where
  x : ℕ  -- Number of machines transported from B to A
  h : x ≤ 6  -- Constraint on x

/-- Calculates the total cost of a transport plan --/
def totalCost (plan : TransportPlan) : ℕ :=
  200 * plan.x + 8600

/-- Theorem: The minimum cost occurs when no machines are moved from B to A --/
theorem min_cost_at_zero :
  ∀ plan : TransportPlan, totalCost plan ≥ 8600 := by
  sorry

/-- Theorem: The minimum cost is 8600 yuan --/
theorem min_cost_value :
  (∃ plan : TransportPlan, totalCost plan = 8600) ∧
  (∀ plan : TransportPlan, totalCost plan ≥ 8600) := by
  sorry

end NUMINAMATH_CALUDE_min_cost_at_zero_min_cost_value_l1581_158142


namespace NUMINAMATH_CALUDE_number_equation_solution_l1581_158161

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1581_158161


namespace NUMINAMATH_CALUDE_sum_squared_equals_sixteen_l1581_158129

theorem sum_squared_equals_sixteen (a b : ℝ) (h : a + b = 4) : a^2 + 2*a*b + b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_sixteen_l1581_158129


namespace NUMINAMATH_CALUDE_intersection_union_problem_l1581_158137

theorem intersection_union_problem (m : ℝ) : 
  let A : Set ℝ := {3, 4, m^2 - 3*m - 1}
  let B : Set ℝ := {2*m, -3}
  (A ∩ B = {-3}) → (m = 1 ∧ A ∪ B = {-3, 2, 3, 4}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_problem_l1581_158137


namespace NUMINAMATH_CALUDE_weight_of_rod_l1581_158136

/-- Represents the weight of a uniform rod -/
structure UniformRod where
  /-- Weight per meter of the rod -/
  weight_per_meter : ℝ
  /-- The rod is uniform (constant weight per meter) -/
  uniform : True

/-- Calculate the weight of a given length of a uniform rod -/
def weight_of_length (rod : UniformRod) (length : ℝ) : ℝ :=
  rod.weight_per_meter * length

/-- Theorem: Given a uniform rod where 8 m weighs 30.4 kg, the weight of 11.25 m is 42.75 kg -/
theorem weight_of_rod (rod : UniformRod) 
  (h : weight_of_length rod 8 = 30.4) : 
  weight_of_length rod 11.25 = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_rod_l1581_158136


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1581_158114

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1581_158114


namespace NUMINAMATH_CALUDE_circle_inequality_max_k_l1581_158117

theorem circle_inequality_max_k : 
  (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) ∧ 
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) → k ≤ -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_max_k_l1581_158117


namespace NUMINAMATH_CALUDE_correct_statements_l1581_158140

theorem correct_statements (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (3*a + 2*b) / (2*a + 3*b) < a / b ∧
  Real.log ((a + b) / 2) > (Real.log a + Real.log b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_statements_l1581_158140


namespace NUMINAMATH_CALUDE_height_difference_calculation_l1581_158101

/-- The combined height difference between an uncle and his two relatives -/
def combined_height_difference (uncle_height james_initial_ratio growth_spurt younger_sibling_height : ℝ) : ℝ :=
  let james_new_height := uncle_height * james_initial_ratio + growth_spurt
  let diff_uncle_james := uncle_height - james_new_height
  let diff_uncle_younger := uncle_height - younger_sibling_height
  diff_uncle_james + diff_uncle_younger

/-- Theorem stating the combined height difference given specific measurements -/
theorem height_difference_calculation :
  combined_height_difference 72 (2/3) 10 38 = 48 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_calculation_l1581_158101


namespace NUMINAMATH_CALUDE_f_not_monotonic_implies_k_range_l1581_158153

noncomputable def f (x : ℝ) : ℝ := x^2 - (1/2) * Real.log x + 1

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ f x > f y

theorem f_not_monotonic_implies_k_range (k : ℝ) :
  (∀ x, x > 0 → f x = x^2 - (1/2) * Real.log x + 1) →
  (¬ is_monotonic f (k - 1) (k + 1)) →
  k ∈ Set.Icc 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_implies_k_range_l1581_158153


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1581_158197

theorem simplify_trig_expression (x : ℝ) (h : 1 + Real.sin x + Real.cos x ≠ 0) :
  (1 + Real.sin x - Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1581_158197


namespace NUMINAMATH_CALUDE_ship_speed_comparison_ship_time_comparison_l1581_158163

/-- Prove that the harmonic mean of two speeds is less than their arithmetic mean -/
theorem ship_speed_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * speed_forward * speed_return) / (speed_forward + speed_return) < 
  (speed_forward + speed_return) / 2 := by
  sorry

/-- Prove that a ship with varying speeds takes longer than a ship with constant average speed -/
theorem ship_time_comparison 
  (distance : ℝ) 
  (speed_forward : ℝ) 
  (speed_return : ℝ) 
  (h1 : 0 < distance)
  (h2 : 0 < speed_forward)
  (h3 : 0 < speed_return)
  (h4 : speed_forward ≠ speed_return) :
  (2 * distance) / ((2 * speed_forward * speed_return) / (speed_forward + speed_return)) > 
  (2 * distance) / ((speed_forward + speed_return) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ship_speed_comparison_ship_time_comparison_l1581_158163


namespace NUMINAMATH_CALUDE_average_difference_l1581_158178

def total_students : ℕ := 150
def total_teachers : ℕ := 6
def class_sizes : List ℕ := [60, 40, 30, 10, 5, 5]

def t : ℚ := (total_students : ℚ) / total_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / total_students

theorem average_difference :
  t - s = -16.68 :=
sorry

end NUMINAMATH_CALUDE_average_difference_l1581_158178


namespace NUMINAMATH_CALUDE_at_least_one_genuine_product_l1581_158176

theorem at_least_one_genuine_product (total : Nat) (genuine : Nat) (defective : Nat) (selected : Nat) :
  total = genuine + defective →
  total = 12 →
  genuine = 10 →
  defective = 2 →
  selected = 3 →
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_product_l1581_158176


namespace NUMINAMATH_CALUDE_sons_age_l1581_158172

/-- Given a father and son where the father is 26 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 24 years. -/
theorem sons_age (son_age father_age : ℕ) 
  (h1 : father_age = son_age + 26)
  (h2 : father_age + 2 = 2 * (son_age + 2)) : 
  son_age = 24 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l1581_158172


namespace NUMINAMATH_CALUDE_triangle_area_angle_l1581_158100

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / 4
  S = (1/2) * a * b * Real.sin (π/4) →
  ∃ A B C : ℝ,
    A + B + C = π ∧
    a = BC ∧ b = AC ∧ c = AB ∧
    C = π/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l1581_158100


namespace NUMINAMATH_CALUDE_base8_addition_problem_l1581_158138

-- Define the base
def base : ℕ := 8

-- Define the addition operation in base 8
def add_base8 (a b : ℕ) : ℕ := (a + b) % base

-- Define the carry operation in base 8
def carry_base8 (a b : ℕ) : ℕ := (a + b) / base

-- The theorem to prove
theorem base8_addition_problem (square : ℕ) :
  square < base →
  add_base8 (add_base8 square square) 4 = 6 →
  add_base8 (add_base8 3 5) square = square →
  add_base8 (add_base8 4 square) (carry_base8 3 5) = 3 →
  square = 1 := by
sorry

end NUMINAMATH_CALUDE_base8_addition_problem_l1581_158138


namespace NUMINAMATH_CALUDE_james_potato_problem_l1581_158146

/-- The problem of calculating the number of people James made potatoes for. -/
theorem james_potato_problem (pounds_per_person : ℝ) (bag_weight : ℝ) (bag_cost : ℝ) (total_spent : ℝ) :
  pounds_per_person = 1.5 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_spent = 15 →
  (total_spent / bag_cost) * bag_weight / pounds_per_person = 40 := by
  sorry

#check james_potato_problem

end NUMINAMATH_CALUDE_james_potato_problem_l1581_158146


namespace NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1581_158112

theorem square_pentagon_side_ratio :
  ∀ (s_s s_p : ℝ),
  s_s > 0 → s_p > 0 →
  s_s^2 = (5 * s_p^2 * (Real.sqrt 5 + 1)) / 8 →
  s_p / s_s = Real.sqrt (8 / (5 * (Real.sqrt 5 + 1))) :=
by sorry

end NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1581_158112


namespace NUMINAMATH_CALUDE_equation_solution_difference_l1581_158107

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (2*x₁ + 15) = 3 ∧
  (x₂ + 3)^2 / (2*x₂ + 15) = 3 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l1581_158107
