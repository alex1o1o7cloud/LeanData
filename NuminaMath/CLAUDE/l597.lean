import Mathlib

namespace NUMINAMATH_CALUDE_parabola_parameters_correct_l597_59759

/-- Two parabolas with common focus and passing through two points -/
structure TwoParabolas where
  F : ℝ × ℝ
  P₁ : ℝ × ℝ
  P₂ : ℝ × ℝ
  h₁ : F = (2, 2)
  h₂ : P₁ = (4, 2)
  h₃ : P₂ = (-2, 5)

/-- The parameters of the two parabolas -/
def parabola_parameters (tp : TwoParabolas) : ℝ × ℝ :=
  (2, 3.6)

/-- Theorem stating that the parameters of the two parabolas are 2 and 3.6 -/
theorem parabola_parameters_correct (tp : TwoParabolas) :
  parabola_parameters tp = (2, 3.6) := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameters_correct_l597_59759


namespace NUMINAMATH_CALUDE_painting_area_l597_59777

/-- The area of a rectangular painting inside a border -/
theorem painting_area (outer_width outer_height border_width : ℕ) : 
  outer_width = 100 ∧ outer_height = 150 ∧ border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 8400 :=
by sorry

end NUMINAMATH_CALUDE_painting_area_l597_59777


namespace NUMINAMATH_CALUDE_exists_valid_coloring_for_all_k_l597_59763

/-- A point on an infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A set of black squares on an infinite white grid -/
def BlackSquares := Set GridPoint

/-- A line on the grid (vertical, horizontal, or diagonal) -/
structure GridLine where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The number of black squares on a given line -/
def blackSquaresOnLine (blacks : BlackSquares) (line : GridLine) : ℕ :=
  sorry

/-- A valid coloring of the grid for a given k -/
def validColoring (k : ℕ) (blacks : BlackSquares) : Prop :=
  (blacks.Nonempty) ∧
  (∀ line : GridLine, blackSquaresOnLine blacks line = k ∨ blackSquaresOnLine blacks line = 0)

/-- The main theorem: for any positive k, there exists a valid coloring -/
theorem exists_valid_coloring_for_all_k :
  ∀ k : ℕ, k > 0 → ∃ blacks : BlackSquares, validColoring k blacks :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_for_all_k_l597_59763


namespace NUMINAMATH_CALUDE_expression_evaluation_l597_59744

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  3 * y^2 - x^2 + (2 * x - y) - (x^2 + 3 * y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l597_59744


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_problem_solution_l597_59775

/-- An arithmetic sequence with the property that the sequence of products of consecutive terms
    forms a geometric progression, and the first term is 1, is constant with all terms equal to 1. -/
theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  ∀ n, a n = 1 :=  -- all terms are 1
by sorry

/-- The 2017th term of the sequence described in the problem is 1. -/
theorem problem_solution (a : ℕ → ℝ) :
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  a 2017 = 1 :=  -- 2017th term is 1
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_problem_solution_l597_59775


namespace NUMINAMATH_CALUDE_distribute_students_count_l597_59779

/-- The number of ways to distribute 5 students into 3 groups -/
def distribute_students : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let k : ℕ := 3  -- Number of groups
  let min_a : ℕ := 2  -- Minimum number of students in Group A
  let min_bc : ℕ := 1  -- Minimum number of students in Groups B and C
  sorry

/-- Theorem stating that the number of distribution schemes is 80 -/
theorem distribute_students_count : distribute_students = 80 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l597_59779


namespace NUMINAMATH_CALUDE_vector_sum_as_complex_sum_l597_59785

theorem vector_sum_as_complex_sum :
  let z₁ : ℂ := 1 + 4*I
  let z₂ : ℂ := -3 + 2*I
  z₁ + z₂ = -2 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_as_complex_sum_l597_59785


namespace NUMINAMATH_CALUDE_rainfall_ratio_l597_59781

theorem rainfall_ratio (total_rainfall : ℝ) (second_week_rainfall : ℝ) 
  (h1 : total_rainfall = 40)
  (h2 : second_week_rainfall = 24) :
  (second_week_rainfall) / (total_rainfall - second_week_rainfall) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l597_59781


namespace NUMINAMATH_CALUDE_inequality_always_holds_l597_59735

theorem inequality_always_holds (a b c : ℝ) (h : a < b ∧ b < c) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l597_59735


namespace NUMINAMATH_CALUDE_x_minus_y_power_2007_l597_59713

theorem x_minus_y_power_2007 (x y : ℝ) :
  5 * x^2 - 4 * x * y + y^2 - 2 * x + 1 = 0 →
  (x - y)^2007 = -1 := by sorry

end NUMINAMATH_CALUDE_x_minus_y_power_2007_l597_59713


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l597_59722

/-- The equation 3(5 + dx) = 15x + 15 has infinitely many solutions for x if and only if d = 5 -/
theorem infinitely_many_solutions (d : ℝ) : 
  (∀ x, 3 * (5 + d * x) = 15 * x + 15) ↔ d = 5 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l597_59722


namespace NUMINAMATH_CALUDE_num_regions_convex_ngon_l597_59792

/-- A convex n-gon is a polygon with n sides where all interior angles are less than 180 degrees. -/
structure ConvexNGon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- The number of regions formed by the diagonals of a convex n-gon where no three diagonals intersect at a single interior point. -/
def num_regions (n : ℕ) : ℕ := Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Theorem stating that the number of regions formed by the diagonals of a convex n-gon
    where no three diagonals intersect at a single interior point is C_n^4 + C_{n-1}^2. -/
theorem num_regions_convex_ngon (n : ℕ) (polygon : ConvexNGon n) :
  num_regions n = Nat.choose n 4 + Nat.choose (n - 1) 2 := by
  sorry

#check num_regions_convex_ngon

end NUMINAMATH_CALUDE_num_regions_convex_ngon_l597_59792


namespace NUMINAMATH_CALUDE_percentage_boys_school_A_l597_59731

/-- Proves that the percentage of boys from school A in a camp is 20% -/
theorem percentage_boys_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (percent_A_science : ℚ) :
  total_boys = 250 →
  boys_A_not_science = 35 →
  percent_A_science = 30 / 100 →
  (boys_A_not_science : ℚ) / ((1 - percent_A_science) * total_boys) = 20 / 100 := by
  sorry

#check percentage_boys_school_A

end NUMINAMATH_CALUDE_percentage_boys_school_A_l597_59731


namespace NUMINAMATH_CALUDE_solve_for_a_l597_59721

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (a : ℝ) : Prop :=
  (a * i) / (1 - i) = -1 + i

-- Theorem statement
theorem solve_for_a : ∃ (a : ℝ), equation a ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l597_59721


namespace NUMINAMATH_CALUDE_factorization_problem_l597_59784

-- Define the expressions
def expr_A (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2
def expr_A_factored (a b : ℝ) : ℝ := (a + b)^2

def expr_B (x y : ℝ) : ℝ := x*y - 4*x + y - 4
def expr_B_factored (x y : ℝ) : ℝ := (x + 1) * (y - 4)

def expr_C (x : ℝ) : ℝ := x^2 + 6*x - 9
def expr_C_factored (x : ℝ) : ℝ := (x + 3) * (x - 3) + 6*x

def expr_D (x : ℝ) : ℝ := x^2 + 3*x - 10
def expr_D_factored (x : ℝ) : ℝ := (x + 5) * (x - 2)

-- Theorem stating that C is not a factorization while others are
theorem factorization_problem :
  (∀ a b, expr_A a b = expr_A_factored a b) ∧
  (∀ x y, expr_B x y = expr_B_factored x y) ∧
  (∃ x, expr_C x ≠ expr_C_factored x) ∧
  (∀ x, expr_D x = expr_D_factored x) :=
sorry

end NUMINAMATH_CALUDE_factorization_problem_l597_59784


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l597_59737

theorem quadratic_equation_properties (m : ℝ) :
  -- Part 1: The equation always has real roots
  ∃ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 ∧ x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 ∧
  -- Part 2: If x₁² + x₂² = 5, then m = 0 or m = -2
  (∀ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 → x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 → x₁^2 + x₂^2 = 5 → m = 0 ∨ m = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l597_59737


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l597_59774

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l597_59774


namespace NUMINAMATH_CALUDE_largest_n_value_l597_59745

/-- Represents a number in a given base -/
structure BaseRepresentation (base : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < base

/-- The value of n in base 10 given its representation in another base -/
def toBase10 (base : ℕ) (repr : BaseRepresentation base) : ℕ :=
  repr.digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_n_value (n : ℕ) 
  (base8_repr : BaseRepresentation 8)
  (base12_repr : BaseRepresentation 12)
  (h1 : toBase10 8 base8_repr = n)
  (h2 : toBase10 12 base12_repr = n)
  (h3 : base8_repr.digits.length = 3)
  (h4 : base12_repr.digits.length = 3)
  (h5 : base8_repr.digits.reverse = base12_repr.digits) :
  n ≤ 509 := by
  sorry

#check largest_n_value

end NUMINAMATH_CALUDE_largest_n_value_l597_59745


namespace NUMINAMATH_CALUDE_rationalize_denominator_l597_59711

theorem rationalize_denominator :
  36 / Real.sqrt 7 = (36 * Real.sqrt 7) / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l597_59711


namespace NUMINAMATH_CALUDE_triangle_properties_l597_59783

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 - t.a*t.b + t.b^2 = t.c^2) 
  (h2 : t.c = 2) 
  (area : ℝ) 
  (h3 : area = 1/2 * t.a * t.b * Real.sin t.C) :
  (t.C = Real.pi/2) ∧ (t.a + t.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l597_59783


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l597_59725

theorem solve_exponential_equation (x : ℝ) : 
  (12 : ℝ)^x * 6^4 / 432 = 432 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l597_59725


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l597_59743

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 1995 * x - 1996
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -1995) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l597_59743


namespace NUMINAMATH_CALUDE_student_average_age_l597_59767

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (new_average : ℝ) 
  (h1 : num_students = 10)
  (h2 : teacher_age = 26)
  (h3 : new_average = 16)
  (h4 : (num_students : ℝ) * new_average = (num_students + 1 : ℝ) * new_average - teacher_age) :
  (num_students : ℝ) * new_average - teacher_age = num_students * 15 := by
sorry

end NUMINAMATH_CALUDE_student_average_age_l597_59767


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l597_59707

theorem inequality_not_always_hold (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ c : ℝ, ¬(a * c > b * c) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l597_59707


namespace NUMINAMATH_CALUDE_clothing_sales_properties_l597_59708

/-- Represents the sales pattern of a new clothing item in July -/
structure ClothingSales where
  /-- The day number in July when maximum sales occurred -/
  max_day : ℕ
  /-- The maximum number of pieces sold in a day -/
  max_sales : ℕ
  /-- The number of days the clothing was popular -/
  popular_days : ℕ

/-- Calculates the sales for a given day in July -/
def daily_sales (day : ℕ) : ℕ :=
  if day ≤ 13 then 3 * day else 65 - 2 * day

/-- Calculates the cumulative sales up to a given day in July -/
def cumulative_sales (day : ℕ) : ℕ :=
  if day ≤ 13 
  then (3 + 3 * day) * day / 2
  else 273 + (51 - day) * (day - 13)

/-- Theorem stating the properties of the clothing sales in July -/
theorem clothing_sales_properties : ∃ (s : ClothingSales),
  s.max_day = 13 ∧ 
  s.max_sales = 39 ∧ 
  s.popular_days = 11 ∧
  daily_sales 1 = 3 ∧
  daily_sales 31 = 3 ∧
  (∀ d : ℕ, d < s.max_day → daily_sales (d + 1) = daily_sales d + 3) ∧
  (∀ d : ℕ, s.max_day < d ∧ d ≤ 31 → daily_sales d = daily_sales (d - 1) - 2) ∧
  (∃ d : ℕ, d ≥ 12 ∧ cumulative_sales d ≥ 200 ∧ cumulative_sales (d - 1) < 200) ∧
  (∃ d : ℕ, d ≤ 22 ∧ daily_sales d ≥ 20 ∧ daily_sales (d + 1) < 20) := by
  sorry

end NUMINAMATH_CALUDE_clothing_sales_properties_l597_59708


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l597_59787

/-- Edward's lawn mowing earnings problem -/
theorem lawn_mowing_earnings 
  (total_earnings summer_earnings spring_earnings supplies_cost end_amount : ℕ)
  (h1 : total_earnings = summer_earnings + spring_earnings)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5)
  (h4 : end_amount = 24)
  (h5 : total_earnings = end_amount + supplies_cost) :
  spring_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l597_59787


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_l597_59782

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary concepts
variable (is_cyclic_quadrilateral : Point → Point → Point → Point → Prop)
variable (diagonals_intersect : Point → Point → Point → Point → Point → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (circles_intersect : Circle → Circle → Point → Prop)
variable (center_of_circle : Circle → Point)
variable (distinct : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Point → Point → Prop)

-- State the theorem
theorem cyclic_quadrilateral_perpendicular
  (A B C D X Y O : Point)
  (circle_ABCD : Circle)
  (circle_ABX circle_CDX : Circle)
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : diagonals_intersect A B C D X)
  (h3 : circle_ABX = circumcircle A B X)
  (h4 : circle_CDX = circumcircle C D X)
  (h5 : circles_intersect circle_ABX circle_CDX Y)
  (h6 : O = center_of_circle circle_ABCD)
  (h7 : distinct O X Y)
  : perpendicular O Y X Y :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_l597_59782


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l597_59790

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 789 % 26 ∧ 
  ∀ (m : ℕ), m > 0 ∧ (5 * m) % 26 = 789 % 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l597_59790


namespace NUMINAMATH_CALUDE_intersection_implies_values_l597_59758

/-- Sets T and S in the xy-plane -/
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

/-- The main theorem -/
theorem intersection_implies_values (a b : ℝ) :
  T a ∩ S b = {(2, 1)} → a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_values_l597_59758


namespace NUMINAMATH_CALUDE_prime_quadruples_sum_882_l597_59728

theorem prime_quadruples_sum_882 : 
  ∀ p₁ p₂ p₃ p₄ : ℕ, 
    Prime p₁ → Prime p₂ → Prime p₃ → Prime p₄ →
    p₁ < p₂ → p₂ < p₃ → p₃ < p₄ →
    p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882 →
    ((p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
     (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
     (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29)) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_quadruples_sum_882_l597_59728


namespace NUMINAMATH_CALUDE_painted_cubes_count_l597_59704

/-- Represents a cube constructed from unit cubes -/
structure LargeCube where
  side_length : ℕ
  unpainted_cubes : ℕ

/-- Calculates the number of cubes with at least one face painted -/
def painted_cubes (c : LargeCube) : ℕ :=
  c.side_length ^ 3 - c.unpainted_cubes

/-- The theorem states that for a cube with 22 unpainted cubes,
    42 cubes have at least one face painted red -/
theorem painted_cubes_count (c : LargeCube) 
  (h1 : c.unpainted_cubes = 22) 
  (h2 : c.side_length = 4) : 
  painted_cubes c = 42 := by
  sorry

#check painted_cubes_count

end NUMINAMATH_CALUDE_painted_cubes_count_l597_59704


namespace NUMINAMATH_CALUDE_room_perimeter_is_16_l597_59717

/-- A rectangular room with specific properties -/
structure Room where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_eq : length = 3 * breadth
  area_eq : area = length * breadth

/-- The perimeter of a rectangular room -/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: The perimeter of a room with given properties is 16 meters -/
theorem room_perimeter_is_16 (r : Room) (h : r.area = 12) : perimeter r = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_perimeter_is_16_l597_59717


namespace NUMINAMATH_CALUDE_sum_of_corners_is_164_l597_59741

/-- Represents a square on the checkerboard -/
structure Square where
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- Function to get the number in a given square -/
def getNumber (s : Square) : Nat :=
  (s.row - 1) * boardSize + s.col

/-- The four corner squares of the board -/
def corners : List Square := [
  { row := 1, col := 1 },       -- Top left
  { row := 1, col := boardSize },  -- Top right
  { row := boardSize, col := 1 },  -- Bottom left
  { row := boardSize, col := boardSize }  -- Bottom right
]

/-- Theorem stating that the sum of numbers in the corners is 164 -/
theorem sum_of_corners_is_164 :
  (corners.map getNumber).sum = 164 := by sorry

end NUMINAMATH_CALUDE_sum_of_corners_is_164_l597_59741


namespace NUMINAMATH_CALUDE_hyperbola_equation_l597_59746

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  h_asymptote : ∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt 3 * x
  h_focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ x = -6

/-- The theorem stating the specific equation of the hyperbola -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 9 ∧ h.b^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l597_59746


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l597_59709

-- Define an even function on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_incr : increasing_on_nonneg f) :
  ∀ k, f k > f 2 ↔ k > 2 ∨ k < -2 :=
sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l597_59709


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l597_59765

/-- Given a square banner with side length 12 feet, one large shaded square
    with side length S, and twelve smaller congruent shaded squares with
    side length T, where 12:S = S:T = 4, the total shaded area is 15.75 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S = 12 / 4 →
  T = S / 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l597_59765


namespace NUMINAMATH_CALUDE_monotone_special_function_characterization_l597_59791

/-- A monotone function on real numbers satisfying f(x) + 2x = f(f(x)) -/
def MonotoneSpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ ∀ x, f x + 2 * x = f (f x)

/-- The theorem stating that a MonotoneSpecialFunction must be either f(x) = -x or f(x) = 2x -/
theorem monotone_special_function_characterization (f : ℝ → ℝ) 
  (hf : MonotoneSpecialFunction f) : 
  (∀ x, f x = -x) ∨ (∀ x, f x = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_monotone_special_function_characterization_l597_59791


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l597_59710

def M : Set ℕ := {x : ℕ | 0 < x ∧ x < 4}
def N : Set ℕ := {x : ℕ | 1 < x ∧ x ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l597_59710


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l597_59751

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l597_59751


namespace NUMINAMATH_CALUDE_modulus_of_w_l597_59795

theorem modulus_of_w (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_w_l597_59795


namespace NUMINAMATH_CALUDE_sum_of_possible_k_is_95_l597_59761

/-- Given a quadratic equation x^2 + 10x + k = 0 with two distinct negative integer solutions,
    this function returns the sum of all possible values of k. -/
def sumOfPossibleK : ℤ := by
  sorry

/-- The theorem states that the sum of all possible values of k is 95. -/
theorem sum_of_possible_k_is_95 : sumOfPossibleK = 95 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_k_is_95_l597_59761


namespace NUMINAMATH_CALUDE_lg_45_equals_1_minus_m_plus_2n_l597_59769

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_45_equals_1_minus_m_plus_2n (m n : ℝ) (h1 : lg 2 = m) (h2 : lg 3 = n) :
  lg 45 = 1 - m + 2 * n := by
  sorry

end NUMINAMATH_CALUDE_lg_45_equals_1_minus_m_plus_2n_l597_59769


namespace NUMINAMATH_CALUDE_root_ratio_implies_k_value_l597_59756

theorem root_ratio_implies_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 10*x + k = 0 ∧ 
   y^2 + 10*y + k = 0 ∧
   x / y = 3) →
  k = 18.75 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_implies_k_value_l597_59756


namespace NUMINAMATH_CALUDE_largest_arithmetic_mean_of_special_pairs_l597_59789

theorem largest_arithmetic_mean_of_special_pairs : ∃ (a b : ℕ),
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  (a + b) / 2 = (25 / 24) * Real.sqrt (a * b) ∧
  ∀ (c d : ℕ),
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    (c + d) / 2 = (25 / 24) * Real.sqrt (c * d) →
    (a + b) / 2 ≥ (c + d) / 2 ∧
  (a + b) / 2 = 75 :=
sorry

end NUMINAMATH_CALUDE_largest_arithmetic_mean_of_special_pairs_l597_59789


namespace NUMINAMATH_CALUDE_friendship_theorem_l597_59780

/-- A graph representing friendships in a city --/
structure FriendshipGraph where
  vertices : Finset ℕ
  edges : Finset (Finset ℕ)
  edge_size : ∀ e ∈ edges, Finset.card e = 2
  vertex_bound : Finset.card vertices = 2000000

/-- Property that every subgraph of 2000 vertices contains a triangle --/
def has_triangle_in_subgraphs (G : FriendshipGraph) : Prop :=
  ∀ S : Finset ℕ, S ⊆ G.vertices → Finset.card S = 2000 →
    ∃ T : Finset ℕ, T ⊆ S ∧ Finset.card T = 3 ∧ T ∈ G.edges

/-- Theorem stating the existence of K₄ in the graph --/
theorem friendship_theorem (G : FriendshipGraph) 
  (h : has_triangle_in_subgraphs G) : 
  ∃ K : Finset ℕ, K ⊆ G.vertices ∧ Finset.card K = 4 ∧ 
    ∀ e : Finset ℕ, e ⊆ K → Finset.card e = 2 → e ∈ G.edges :=
sorry

end NUMINAMATH_CALUDE_friendship_theorem_l597_59780


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l597_59753

def f (x : ℤ) : ℤ := x^3 - 6*x^2 - 4*x + 24

theorem integer_roots_of_cubic :
  ∀ x : ℤ, f x = 0 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l597_59753


namespace NUMINAMATH_CALUDE_append_digit_twice_divisible_by_three_l597_59726

theorem append_digit_twice_divisible_by_three (N d : ℕ) 
  (hN : N % 3 ≠ 0) (hd : d % 3 ≠ 0) (hd_last : d < 10) :
  ∃ k, N * 100 + d * 10 + d = 3 * k :=
sorry

end NUMINAMATH_CALUDE_append_digit_twice_divisible_by_three_l597_59726


namespace NUMINAMATH_CALUDE_farmers_harvest_l597_59754

/-- Farmer's harvest problem -/
theorem farmers_harvest
  (total_potatoes : ℕ)
  (potatoes_per_bundle : ℕ)
  (potato_bundle_price : ℚ)
  (total_carrots : ℕ)
  (carrot_bundle_price : ℚ)
  (total_revenue : ℚ)
  (h1 : total_potatoes = 250)
  (h2 : potatoes_per_bundle = 25)
  (h3 : potato_bundle_price = 190/100)
  (h4 : total_carrots = 320)
  (h5 : carrot_bundle_price = 2)
  (h6 : total_revenue = 51)
  : (total_carrots / ((total_revenue - (total_potatoes / potatoes_per_bundle * potato_bundle_price)) / carrot_bundle_price) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_farmers_harvest_l597_59754


namespace NUMINAMATH_CALUDE_shifted_line_y_intercept_l597_59730

/-- A line in the form y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- Shift a line vertically -/
def shiftLine (l : Line) (shift : ℝ) : Line :=
  { m := l.m, c := l.c + shift }

/-- The y-intercept of a line -/
def yIntercept (l : Line) : ℝ := l.c

theorem shifted_line_y_intercept :
  let original_line : Line := { m := 1, c := -1 }
  let shifted_line := shiftLine original_line 2
  yIntercept shifted_line = 1 := by sorry

end NUMINAMATH_CALUDE_shifted_line_y_intercept_l597_59730


namespace NUMINAMATH_CALUDE_congruence_problem_l597_59752

theorem congruence_problem (x : ℤ) 
  (h1 : (2 + x) % 3 = 2^2 % 3)
  (h2 : (4 + x) % 5 = 3^2 % 5)
  (h3 : (6 + x) % 7 = 5^2 % 7) :
  x % 105 = 5 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l597_59752


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l597_59701

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 3) 
  (h6 : seq.S 6 = 15) : 
  seq.a 10 + seq.a 11 + seq.a 12 = 30 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_seq_sum_l597_59701


namespace NUMINAMATH_CALUDE_sum_even_102_to_200_proof_l597_59705

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

/-- The first even integer in the range 102 to 200 -/
def first_even_102_to_200 : ℕ := 102

/-- The last even integer in the range 102 to 200 -/
def last_even_102_to_200 : ℕ := 200

theorem sum_even_102_to_200_proof :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (first_even_102_to_200 + last_even_102_to_200) :=
by sorry

end NUMINAMATH_CALUDE_sum_even_102_to_200_proof_l597_59705


namespace NUMINAMATH_CALUDE_divisible_by_six_l597_59797

theorem divisible_by_six (x : ℤ) : 
  (∃ k : ℤ, x^2 + 5*x - 12 = 6*k) ↔ (∃ t : ℤ, x = 3*t ∨ x = 3*t + 1) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_six_l597_59797


namespace NUMINAMATH_CALUDE_problem_solution_l597_59738

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l597_59738


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l597_59703

/-- Given that the arithmetic mean of six expressions is 30, prove that x = 18.5 and y = 10. -/
theorem arithmetic_mean_problem (x y : ℝ) :
  ((2*x - y) + 20 + (3*x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30 →
  x = 18.5 ∧ y = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l597_59703


namespace NUMINAMATH_CALUDE_parabola_directrix_through_ellipse_focus_l597_59712

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

-- Define the focus of the ellipse
def ellipse_focus : ℝ × ℝ := (2, 0)

-- Define the directrix of the parabola
def parabola_directrix (p : ℝ) : ℝ → Prop := λ x ↦ x = -p/2

-- Theorem statement
theorem parabola_directrix_through_ellipse_focus :
  ∀ p : ℝ, (∃ x y : ℝ, parabola p x y ∧ ellipse x y ∧ 
    parabola_directrix p (ellipse_focus.1)) →
  parabola_directrix p = λ x ↦ x = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_through_ellipse_focus_l597_59712


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l597_59788

theorem green_shirt_pairs (total_students : ℕ) (red_shirts : ℕ) (green_shirts : ℕ) 
  (total_pairs : ℕ) (red_red_pairs : ℕ) :
  total_students = 180 →
  red_shirts = 83 →
  green_shirts = 97 →
  total_pairs = 90 →
  red_red_pairs = 35 →
  red_shirts + green_shirts = total_students →
  2 * total_pairs = total_students →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 42 ∧ 
    green_green_pairs + red_red_pairs + (green_shirts - 2 * green_green_pairs) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l597_59788


namespace NUMINAMATH_CALUDE_min_sum_xy_l597_59755

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x - y - 10 = 0) :
  x + y ≥ 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ + x₀ - y₀ - 10 = 0 ∧ x₀ + y₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xy_l597_59755


namespace NUMINAMATH_CALUDE_exists_valid_expression_l597_59794

def Expression := List (Fin 4 → ℕ)

def applyOps (nums : Fin 4 → ℕ) (ops : Fin 3 → Char) : ℕ :=
  let e1 := match ops 0 with
    | '+' => nums 0 + nums 1
    | '-' => nums 0 - nums 1
    | '×' => nums 0 * nums 1
    | _ => 0
  let e2 := match ops 1 with
    | '+' => e1 + nums 2
    | '-' => e1 - nums 2
    | '×' => e1 * nums 2
    | _ => 0
  match ops 2 with
    | '+' => e2 + nums 3
    | '-' => e2 - nums 3
    | '×' => e2 * nums 3
    | _ => 0

def isValidOps (ops : Fin 3 → Char) : Prop :=
  (ops 0 = '+' ∨ ops 0 = '-' ∨ ops 0 = '×') ∧
  (ops 1 = '+' ∨ ops 1 = '-' ∨ ops 1 = '×') ∧
  (ops 2 = '+' ∨ ops 2 = '-' ∨ ops 2 = '×') ∧
  (ops 0 ≠ ops 1) ∧ (ops 1 ≠ ops 2) ∧ (ops 0 ≠ ops 2)

theorem exists_valid_expression : ∃ (ops : Fin 3 → Char),
  isValidOps ops ∧ applyOps (λ i => [5, 4, 6, 3][i]) ops = 19 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_expression_l597_59794


namespace NUMINAMATH_CALUDE_two_digit_times_99_l597_59715

theorem two_digit_times_99 (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : A ≠ 0) :
  (10 * A + B) * 99 = 100 * (10 * A + B - 1) + (100 - (10 * A + B)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_times_99_l597_59715


namespace NUMINAMATH_CALUDE_basketball_probabilities_l597_59772

/-- Probability of A making a shot -/
def prob_A_makes : ℝ := 0.8

/-- Probability of B missing a shot -/
def prob_B_misses : ℝ := 0.1

/-- Probability of B making a shot -/
def prob_B_makes : ℝ := 1 - prob_B_misses

theorem basketball_probabilities :
  (prob_A_makes * prob_B_makes = 0.72) ∧
  (prob_A_makes * (1 - prob_B_makes) + (1 - prob_A_makes) * prob_B_makes = 0.26) := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l597_59772


namespace NUMINAMATH_CALUDE_certain_number_proof_l597_59757

theorem certain_number_proof (N : ℝ) : 
  (1/2)^22 * N^11 = 1/(18^22) → N = 81 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l597_59757


namespace NUMINAMATH_CALUDE_increase_in_position_for_given_slope_l597_59739

/-- The increase in position for a person moving along a slope --/
def increase_in_position (slope_ratio : ℚ) (total_distance : ℝ) : ℝ :=
  sorry

/-- The theorem stating the increase in position for the given problem --/
theorem increase_in_position_for_given_slope : 
  increase_in_position (1/2) (100 * Real.sqrt 5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_increase_in_position_for_given_slope_l597_59739


namespace NUMINAMATH_CALUDE_red_ball_certain_event_l597_59793

/-- Represents a bag of balls -/
structure Bag where
  balls : Set Color

/-- Represents the color of a ball -/
inductive Color where
  | Red

/-- Represents an event -/
structure Event where
  occurs : Prop

/-- Defines a certain event -/
def CertainEvent (e : Event) : Prop :=
  e.occurs = True

/-- Defines the event of drawing a ball from a bag -/
def DrawBall (b : Bag) (c : Color) : Event where
  occurs := c ∈ b.balls

/-- Theorem: Drawing a red ball from a bag containing only red balls is a certain event -/
theorem red_ball_certain_event (b : Bag) (h : b.balls = {Color.Red}) :
  CertainEvent (DrawBall b Color.Red) := by
  sorry

end NUMINAMATH_CALUDE_red_ball_certain_event_l597_59793


namespace NUMINAMATH_CALUDE_hat_pairs_l597_59729

theorem hat_pairs (total : ℕ) (hat_wearers : ℕ) (h1 : total = 12) (h2 : hat_wearers = 4) :
  (total.choose 2) - ((total - hat_wearers).choose 2) = 38 := by
  sorry

end NUMINAMATH_CALUDE_hat_pairs_l597_59729


namespace NUMINAMATH_CALUDE_a_squared_minus_b_squared_eq_zero_l597_59776

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def first_three_multiples_of_fourteen : List ℕ := [14, 28, 42]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def b : ℕ := first_three_multiples_of_fourteen[1]

theorem a_squared_minus_b_squared_eq_zero : a^2 - (b^2 : ℚ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_minus_b_squared_eq_zero_l597_59776


namespace NUMINAMATH_CALUDE_log_inequality_l597_59768

theorem log_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let f : ℝ → ℝ := fun x ↦ |Real.log x / Real.log a|
  f (1/4) > f (1/3) ∧ f (1/3) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l597_59768


namespace NUMINAMATH_CALUDE_inequality_proof_l597_59766

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (1 / x + 1 / y ≥ 4 / (x + y)) ∧
  (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l597_59766


namespace NUMINAMATH_CALUDE_bernardo_wins_l597_59764

theorem bernardo_wins (N : ℕ) : N = 63 ↔ 
  N ≤ 1999 ∧ 
  (∀ m : ℕ, m < N → 
    (3*m < 3000 ∧
     3*m + 100 < 3000 ∧
     9*m + 300 < 3000 ∧
     9*m + 400 < 3000 ∧
     27*m + 1200 < 3000 ∧
     27*m + 1300 < 3000)) ∧
  (3*N < 3000 ∧
   3*N + 100 < 3000 ∧
   9*N + 300 < 3000 ∧
   9*N + 400 < 3000 ∧
   27*N + 1200 < 3000 ∧
   27*N + 1300 ≥ 3000) :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_l597_59764


namespace NUMINAMATH_CALUDE_definite_integral_2x_plus_exp_l597_59702

theorem definite_integral_2x_plus_exp : ∫ x in (0:ℝ)..1, (2 * x + Real.exp x) = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_2x_plus_exp_l597_59702


namespace NUMINAMATH_CALUDE_mildred_initial_oranges_l597_59770

/-- The number of oranges Mildred's father gave her -/
def oranges_from_father : ℕ := 2

/-- The total number of oranges Mildred has after receiving oranges from her father -/
def total_oranges : ℕ := 79

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := total_oranges - oranges_from_father

theorem mildred_initial_oranges :
  initial_oranges = 77 :=
by sorry

end NUMINAMATH_CALUDE_mildred_initial_oranges_l597_59770


namespace NUMINAMATH_CALUDE_negative_square_two_l597_59799

theorem negative_square_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_two_l597_59799


namespace NUMINAMATH_CALUDE_image_of_two_three_l597_59760

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

-- State the theorem
theorem image_of_two_three :
  f (2, 3) = (6, 5) := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_three_l597_59760


namespace NUMINAMATH_CALUDE_more_apples_than_oranges_l597_59734

theorem more_apples_than_oranges :
  ∀ (apples oranges : ℕ),
    apples + oranges = 301 →
    apples = 164 →
    apples > oranges →
    apples - oranges = 27 := by
  sorry

end NUMINAMATH_CALUDE_more_apples_than_oranges_l597_59734


namespace NUMINAMATH_CALUDE_mailman_junk_mail_l597_59762

/-- Given a total number of mail pieces and a number of magazines, 
    calculate the number of junk mail pieces. -/
def junk_mail (total : ℕ) (magazines : ℕ) : ℕ :=
  total - magazines

theorem mailman_junk_mail :
  junk_mail 11 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mailman_junk_mail_l597_59762


namespace NUMINAMATH_CALUDE_money_sharing_l597_59706

theorem money_sharing (emani howard jamal : ℕ) (h1 : emani = 150) (h2 : emani = howard + 30) (h3 : jamal = 75) :
  (emani + howard + jamal) / 3 = 115 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l597_59706


namespace NUMINAMATH_CALUDE_two_correct_relations_l597_59720

theorem two_correct_relations : 
  (0 ∈ ({0} : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {0}) ∧ 
  ¬({0, 1} ⊆ ({(0, 1)} : Set (ℕ × ℕ))) ∧ 
  ∀ a b : ℕ, {(a, b)} ≠ ({(b, a)} : Set (ℕ × ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_two_correct_relations_l597_59720


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_l597_59714

theorem largest_root_of_cubic (p q r : ℝ) : 
  p + q + r = 3 → 
  p * q + p * r + q * r = -8 → 
  p * q * r = -18 → 
  ∃ (x : ℝ), x = Real.sqrt 6 ∧ 
    x = max p (max q r) ∧
    x^3 - 3*x^2 - 8*x + 18 = 0 := by
sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_l597_59714


namespace NUMINAMATH_CALUDE_ruth_gave_two_sandwiches_to_brother_l597_59749

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches the first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of other cousins -/
def other_cousins : ℕ := 2

/-- The number of sandwiches each other cousin ate -/
def each_other_cousin_ate : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- The number of sandwiches Ruth gave to her brother -/
def sandwiches_to_brother : ℕ := total_sandwiches - (ruth_ate + first_cousin_ate + other_cousins * each_other_cousin_ate + sandwiches_left)

theorem ruth_gave_two_sandwiches_to_brother : sandwiches_to_brother = 2 := by
  sorry

end NUMINAMATH_CALUDE_ruth_gave_two_sandwiches_to_brother_l597_59749


namespace NUMINAMATH_CALUDE_not_all_fractions_integer_l597_59718

theorem not_all_fractions_integer 
  (a b c r s t : ℕ+) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (eq1 : a * b + 1 = r^2)
  (eq2 : a * c + 1 = s^2)
  (eq3 : b * c + 1 = t^2) :
  ¬(∃ (x y z : ℕ), (r * t : ℚ) / s = x ∧ (r * s : ℚ) / t = y ∧ (s * t : ℚ) / r = z) :=
sorry

end NUMINAMATH_CALUDE_not_all_fractions_integer_l597_59718


namespace NUMINAMATH_CALUDE_book_sale_price_l597_59748

def book_sale (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ) : Prop :=
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  (2 : ℚ) / 3 * total_books = sold_books ∧
  unsold_books = 36 ∧
  total_amount = 252 ∧
  price_per_book = (7 : ℚ) / 2

theorem book_sale_price :
  ∃ (total_books : ℕ) (unsold_books : ℕ) (total_amount : ℚ),
    book_sale total_books unsold_books total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_price_l597_59748


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_000815_l597_59773

def scientific_notation (n : ℝ) (coefficient : ℝ) (exponent : ℤ) : Prop :=
  1 ≤ coefficient ∧ coefficient < 10 ∧ n = coefficient * (10 : ℝ) ^ exponent

theorem scientific_notation_of_0_000815 :
  scientific_notation 0.000815 8.15 (-4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_000815_l597_59773


namespace NUMINAMATH_CALUDE_inequality_proof_l597_59747

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + y^2 + z^2) * ((x^2 + y^2 + z^2)^2 - (x*y + y*z + z*x)^2) ≥ 
  (x + y + z)^2 * ((x^2 + y^2 + z^2) - (x*y + y*z + z*x))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l597_59747


namespace NUMINAMATH_CALUDE_adult_books_count_l597_59750

theorem adult_books_count (total : ℕ) (children_percent : ℚ) (h1 : total = 160) (h2 : children_percent = 35 / 100) :
  (total : ℚ) * (1 - children_percent) = 104 := by
  sorry

end NUMINAMATH_CALUDE_adult_books_count_l597_59750


namespace NUMINAMATH_CALUDE_sum_of_inscribed_circles_limit_l597_59740

/-- The sum of areas of inscribed circles in a rectangle --/
def sum_of_circle_areas (m : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- The limit of the sum as n approaches infinity --/
def limit_of_sum (m : ℝ) : ℝ :=
  sorry

/-- Theorem: The limit of the sum of areas of inscribed circles approaches 5πm^2 --/
theorem sum_of_inscribed_circles_limit (m : ℝ) (h : m > 0) :
  limit_of_sum m = 5 * Real.pi * m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inscribed_circles_limit_l597_59740


namespace NUMINAMATH_CALUDE_inequality_proof_l597_59716

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b + c * d) * (a * d + b * c) / ((a + c) * (b + d)) ≥ Real.sqrt (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l597_59716


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l597_59736

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 1/2

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia sent carnations to -/
def num_teachers : ℕ := 5

/-- The number of friends Georgia bought carnations for -/
def num_friends : ℕ := 14

/-- The total cost of carnations Georgia would spend -/
def total_cost : ℚ := num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost

theorem georgia_carnation_cost : total_cost = 27 := by sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l597_59736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l597_59723

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a4 : a 4 = 13)
  (h_a7 : a 7 = 25) :
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l597_59723


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l597_59742

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle C has the equation x^2 + (y + 1)^2 = 1
theorem symmetric_circle_equation :
  ∀ x y : ℝ,
  (∃ x' y' : ℝ, original_circle x' y' ∧ 
   symmetry_line ((x + x') / 2) ((y + y') / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l597_59742


namespace NUMINAMATH_CALUDE_quadratic_factorization_l597_59727

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 9 * a = a * (x - 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l597_59727


namespace NUMINAMATH_CALUDE_club_officer_selection_l597_59700

theorem club_officer_selection (n : ℕ) (e : ℕ) (h1 : n = 12) (h2 : e = 5) (h3 : e ≤ n) :
  (n * (n - 1) * e * (n - 2)) = 6600 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l597_59700


namespace NUMINAMATH_CALUDE_optimal_reading_distribution_l597_59771

theorem optimal_reading_distribution 
  (total_time : ℕ) 
  (disc_capacity : ℕ) 
  (max_unused_space : ℕ) 
  (h1 : total_time = 630) 
  (h2 : disc_capacity = 80) 
  (h3 : max_unused_space = 4) :
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * (disc_capacity - max_unused_space) ≥ total_time ∧
    (num_discs - 1) * disc_capacity < total_time ∧
    total_time / num_discs = 70 :=
sorry

end NUMINAMATH_CALUDE_optimal_reading_distribution_l597_59771


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l597_59778

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 12)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_size : ℝ) * avg_age_group1 + (group2_size : ℝ) * avg_age_group2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l597_59778


namespace NUMINAMATH_CALUDE_correct_costs_l597_59732

/-- Represents the costs of a pen, pencil, and ink refill -/
structure ItemCosts where
  pen : ℚ
  pencil : ℚ
  ink_refill : ℚ

/-- Checks if the given costs satisfy the problem conditions -/
def satisfies_conditions (costs : ItemCosts) : Prop :=
  costs.pen + costs.pencil + costs.ink_refill = 2.4 ∧
  costs.pen = costs.ink_refill + 1.5 ∧
  costs.pencil = costs.ink_refill - 0.4

/-- Theorem stating the correct costs for the items -/
theorem correct_costs :
  ∃ (costs : ItemCosts),
    satisfies_conditions costs ∧
    costs.pen = 1.93 ∧
    costs.pencil = 0.03 ∧
    costs.ink_refill = 0.43 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_costs_l597_59732


namespace NUMINAMATH_CALUDE_chef_michel_pies_l597_59719

/-- Calculates the total number of pies sold given the number of slices per pie and the number of slices ordered --/
def total_pies_sold (shepherds_pie_slices : ℕ) (chicken_pot_pie_slices : ℕ) 
                    (shepherds_pie_ordered : ℕ) (chicken_pot_pie_ordered : ℕ) : ℕ :=
  (shepherds_pie_ordered / shepherds_pie_slices) + (chicken_pot_pie_ordered / chicken_pot_pie_slices)

/-- Proves that Chef Michel sold 29 pies in total --/
theorem chef_michel_pies : 
  total_pies_sold 4 5 52 80 = 29 := by
  sorry

end NUMINAMATH_CALUDE_chef_michel_pies_l597_59719


namespace NUMINAMATH_CALUDE_ticket_sales_problem_l597_59724

/-- Proves that the total number of tickets sold is 42 given the conditions of the ticket sales problem. -/
theorem ticket_sales_problem (adult_price child_price total_sales child_tickets : ℕ)
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : child_tickets = 16) :
  ∃ (adult_tickets : ℕ), adult_price * adult_tickets + child_price * child_tickets = total_sales ∧
                          adult_tickets + child_tickets = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_problem_l597_59724


namespace NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l597_59796

theorem abs_five_implies_plus_minus_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_implies_plus_minus_five_l597_59796


namespace NUMINAMATH_CALUDE_sum_of_valid_b_is_six_l597_59733

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The sum of all positive integer values of b for which the quadratic equation 3x^2 + 7x + b = 0 has rational roots -/
def sum_of_valid_b : ℕ := sorry

/-- The main theorem stating that the sum of valid b values is 6 -/
theorem sum_of_valid_b_is_six : sum_of_valid_b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_b_is_six_l597_59733


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l597_59786

/-- Given a geometric sequence where the second term is 18 and the fifth term is 1458,
    prove that the first term is 6. -/
theorem geometric_sequence_first_term
  (a : ℝ)  -- First term of the sequence
  (r : ℝ)  -- Common ratio of the sequence
  (h1 : a * r = 18)  -- Second term is 18
  (h2 : a * r^4 = 1458)  -- Fifth term is 1458
  : a = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l597_59786


namespace NUMINAMATH_CALUDE_walter_works_five_days_l597_59798

/-- Calculates the number of days Walter works per week given his hourly rate, daily hours, allocation percentage, and allocated amount for school. -/
def calculate_work_days (hourly_rate : ℚ) (daily_hours : ℚ) (allocation_percentage : ℚ) (school_allocation : ℚ) : ℚ :=
  let daily_earnings := hourly_rate * daily_hours
  let weekly_earnings := school_allocation / allocation_percentage
  weekly_earnings / daily_earnings

/-- Theorem stating that Walter works 5 days a week given the specified conditions. -/
theorem walter_works_five_days 
  (hourly_rate : ℚ) 
  (daily_hours : ℚ) 
  (allocation_percentage : ℚ) 
  (school_allocation : ℚ) 
  (h1 : hourly_rate = 5)
  (h2 : daily_hours = 4)
  (h3 : allocation_percentage = 3/4)
  (h4 : school_allocation = 75) :
  calculate_work_days hourly_rate daily_hours allocation_percentage school_allocation = 5 := by
  sorry

end NUMINAMATH_CALUDE_walter_works_five_days_l597_59798
