import Mathlib

namespace NUMINAMATH_CALUDE_modified_rectangle_area_l2497_249740

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem about the area of a modified rectangle --/
theorem modified_rectangle_area 
  (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (modified : Rectangle), 
    (modified.length = original.length ∧ modified.width = original.width - 2) ∨
    (modified.length = original.length - 2 ∧ modified.width = original.width) ∧
    area modified = 15) :
  ∃ (final : Rectangle), 
    ((h2.choose.length = original.length ∧ h2.choose.width = original.width - 2) →
      final.length = original.length - 2 ∧ final.width = original.width) ∧
    ((h2.choose.length = original.length - 2 ∧ h2.choose.width = original.width) →
      final.length = original.length ∧ final.width = original.width - 2) ∧
    area final = 7 := by
  sorry

end NUMINAMATH_CALUDE_modified_rectangle_area_l2497_249740


namespace NUMINAMATH_CALUDE_divisibility_condition_l2497_249703

theorem divisibility_condition (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n) ∣ (3^m + 1) → 
  (m * n) ∣ (3^n + 1) → 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2497_249703


namespace NUMINAMATH_CALUDE_cube_difference_equals_negative_875_l2497_249710

theorem cube_difference_equals_negative_875 (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 2 * x + y = 20) : 
  x^3 - y^3 = -875 := by sorry

end NUMINAMATH_CALUDE_cube_difference_equals_negative_875_l2497_249710


namespace NUMINAMATH_CALUDE_tournament_players_l2497_249789

-- Define the number of Asian players
variable (n : ℕ)

-- Define the number of European players as 2n
def european_players := 2 * n

-- Define the total number of matches
def total_matches := n * (n - 1) / 2 + (2 * n) * (2 * n - 1) / 2 + 2 * n^2

-- Define the number of matches won by Europeans
def european_wins (x : ℕ) := (2 * n) * (2 * n - 1) / 2 + x

-- Define the number of matches won by Asians
def asian_wins (x : ℕ) := n * (n - 1) / 2 + 2 * n^2 - x

-- State the theorem
theorem tournament_players :
  ∃ x : ℕ, european_wins n x = (5 / 7) * asian_wins n x ∧ n = 3 ∧ n + european_players n = 9 := by
  sorry


end NUMINAMATH_CALUDE_tournament_players_l2497_249789


namespace NUMINAMATH_CALUDE_wall_length_proof_l2497_249762

/-- Given a square mirror and a rectangular wall, prove the wall's length. -/
theorem wall_length_proof (mirror_side : ℕ) (wall_width : ℕ) : 
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side : ℕ) * 2 = wall_width * (wall_width * 2 - wall_width % 2) →
  wall_width * 2 - wall_width % 2 = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_length_proof_l2497_249762


namespace NUMINAMATH_CALUDE_square_area_problem_l2497_249714

theorem square_area_problem (x : ℝ) (h : 4 * x^2 = 240) : x^2 + (2*x)^2 + x^2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l2497_249714


namespace NUMINAMATH_CALUDE_grape_rate_specific_grape_rate_l2497_249774

/-- The rate of grapes per kg given the following conditions:
  1. 8 kg of grapes were purchased at an unknown rate
  2. 9 kg of mangoes were purchased at 50 rupees per kg
  3. The total amount paid was 1010 rupees -/
theorem grape_rate : ℕ → ℕ → ℕ → ℕ → Prop :=
  λ grape_quantity mango_quantity mango_rate total_paid =>
    ∃ (G : ℕ),
      grape_quantity * G + mango_quantity * mango_rate = total_paid ∧
      G = 70

/-- The specific instance of the problem -/
theorem specific_grape_rate : grape_rate 8 9 50 1010 := by
  sorry

end NUMINAMATH_CALUDE_grape_rate_specific_grape_rate_l2497_249774


namespace NUMINAMATH_CALUDE_barry_head_standing_duration_l2497_249736

def head_standing_duration (total_time minutes_between_turns num_turns : ℕ) : ℚ :=
  (total_time - minutes_between_turns * (num_turns - 1)) / num_turns

theorem barry_head_standing_duration :
  ∃ (x : ℕ), x ≥ 11 ∧ x < 12 ∧ head_standing_duration 120 5 8 < x :=
sorry

end NUMINAMATH_CALUDE_barry_head_standing_duration_l2497_249736


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l2497_249709

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l2497_249709


namespace NUMINAMATH_CALUDE_rectangle_x_value_l2497_249717

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -4), and (x, -4) and area 30, prove that x = -5 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -4), (x, -4)]
  let width := 1 - (-4)
  let area := 30
  let length := area / width
  x = 1 - length → x = -5 := by sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l2497_249717


namespace NUMINAMATH_CALUDE_quadratic_sum_l2497_249770

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 3 3 2 →
  a + b + 2 * c = 2 := by
  sorry

#check quadratic_sum

end NUMINAMATH_CALUDE_quadratic_sum_l2497_249770


namespace NUMINAMATH_CALUDE_selectThreePeopleIs600_l2497_249799

/-- The number of ways to select 3 people from a 5×5 matrix,
    such that no two selected people are in the same row or column. -/
def selectThreePeople : ℕ :=
  let numColumns : ℕ := 5
  let numRows : ℕ := 5
  let numPeopleToSelect : ℕ := 3
  let waysToChooseColumns : ℕ := Nat.choose numColumns numPeopleToSelect
  let waysToChooseFirstPerson : ℕ := numRows
  let waysToChooseSecondPerson : ℕ := numRows - 1
  let waysToChooseThirdPerson : ℕ := numRows - 2
  waysToChooseColumns * waysToChooseFirstPerson * waysToChooseSecondPerson * waysToChooseThirdPerson

/-- Theorem stating that the number of ways to select 3 people
    from a 5×5 matrix, such that no two selected people are in
    the same row or column, is equal to 600. -/
theorem selectThreePeopleIs600 : selectThreePeople = 600 := by
  sorry

end NUMINAMATH_CALUDE_selectThreePeopleIs600_l2497_249799


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2497_249796

theorem cube_roots_of_unity_sum (ω ω_conj : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_conj = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_conj^3 = 1 →
  ω^4 + ω_conj^4 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2497_249796


namespace NUMINAMATH_CALUDE_negative_two_thousand_ten_plus_two_l2497_249729

theorem negative_two_thousand_ten_plus_two :
  (-2010 : ℤ) + 2 = -2008 := by sorry

end NUMINAMATH_CALUDE_negative_two_thousand_ten_plus_two_l2497_249729


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2497_249754

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt x + Real.sqrt (x + 3) = 12) → x = 19881 / 576 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2497_249754


namespace NUMINAMATH_CALUDE_average_student_headcount_l2497_249735

def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

theorem average_student_headcount : 
  (student_headcount.sum / student_headcount.length : ℚ) = 11029 := by
  sorry

end NUMINAMATH_CALUDE_average_student_headcount_l2497_249735


namespace NUMINAMATH_CALUDE_election_probabilities_l2497_249725

theorem election_probabilities 
  (pA pB pC : ℝ)
  (hA : pA = 4/5)
  (hB : pB = 3/5)
  (hC : pC = 7/10) :
  let p_exactly_one := pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC
  let p_at_most_two := 1 - pA * pB * pC
  (p_exactly_one = 47/250) ∧ (p_at_most_two = 83/125) := by
  sorry

end NUMINAMATH_CALUDE_election_probabilities_l2497_249725


namespace NUMINAMATH_CALUDE_sine_inequality_l2497_249760

theorem sine_inequality : 
  (∀ x y, x ∈ Set.Icc 0 (π/2) → y ∈ Set.Icc 0 (π/2) → x < y → Real.sin x < Real.sin y) →
  3*π/7 > 2*π/5 →
  3*π/7 ∈ Set.Icc 0 (π/2) →
  2*π/5 ∈ Set.Icc 0 (π/2) →
  Real.sin (3*π/7) > Real.sin (2*π/5) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2497_249760


namespace NUMINAMATH_CALUDE_swimmers_pass_21_times_l2497_249706

/-- Represents the swimming pool setup and swimmer characteristics --/
structure SwimmingSetup where
  poolLength : ℝ
  swimmerASpeed : ℝ
  swimmerBSpeed : ℝ
  totalTime : ℝ

/-- Calculates the number of times swimmers pass each other --/
def calculatePassings (setup : SwimmingSetup) : ℕ :=
  sorry

/-- Theorem stating that the swimmers pass each other 21 times --/
theorem swimmers_pass_21_times :
  let setup : SwimmingSetup := {
    poolLength := 120,
    swimmerASpeed := 4,
    swimmerBSpeed := 3,
    totalTime := 15 * 60  -- 15 minutes in seconds
  }
  calculatePassings setup = 21 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_21_times_l2497_249706


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2497_249776

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n ≥ 3 → 
  exterior_angle = 45 →
  (360 : ℝ) / exterior_angle = n →
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2497_249776


namespace NUMINAMATH_CALUDE_train_length_l2497_249734

/-- The length of a train that overtakes a motorbike -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 18 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 180 :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_l2497_249734


namespace NUMINAMATH_CALUDE_num_faces_after_transformation_l2497_249751

/-- Represents the number of steps in the transformation process -/
def num_steps : ℕ := 5

/-- The initial number of vertices in a cube -/
def initial_vertices : ℕ := 8

/-- The initial number of edges in a cube -/
def initial_edges : ℕ := 12

/-- The factor by which vertices and edges increase in each step -/
def increase_factor : ℕ := 3

/-- Calculates the number of vertices after the transformation -/
def final_vertices : ℕ := initial_vertices * increase_factor ^ num_steps

/-- Calculates the number of edges after the transformation -/
def final_edges : ℕ := initial_edges * increase_factor ^ num_steps

/-- Theorem stating the number of faces after the transformation -/
theorem num_faces_after_transformation : 
  final_vertices - final_edges + 974 = 2 :=
sorry

end NUMINAMATH_CALUDE_num_faces_after_transformation_l2497_249751


namespace NUMINAMATH_CALUDE_sneezing_fit_proof_l2497_249745

/-- Calculates the number of sneezes given the duration of a sneezing fit in minutes
    and the interval between sneezes in seconds. -/
def number_of_sneezes (duration_minutes : ℕ) (interval_seconds : ℕ) : ℕ :=
  (duration_minutes * 60) / interval_seconds

/-- Proves that a 2-minute sneezing fit with sneezes every 3 seconds results in 40 sneezes. -/
theorem sneezing_fit_proof :
  number_of_sneezes 2 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sneezing_fit_proof_l2497_249745


namespace NUMINAMATH_CALUDE_ellipse_y_equation_ellipse_x_equation_l2497_249749

/-- An ellipse with foci on the y-axis -/
structure EllipseY where
  c : ℝ
  e : ℝ

/-- An ellipse passing through a point on the x-axis -/
structure EllipseX where
  x : ℝ
  e : ℝ

/-- Standard equation of an ellipse -/
def standardEquation (a b : ℝ) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_y_equation (E : EllipseY) (h1 : E.c = 6) (h2 : E.e = 2/3) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 9 (Real.sqrt 45) :=
sorry

theorem ellipse_x_equation (E : EllipseX) (h1 : E.x = 2) (h2 : E.e = Real.sqrt 3 / 2) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 2 1) ∨
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ standardEquation a b = standardEquation 4 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_y_equation_ellipse_x_equation_l2497_249749


namespace NUMINAMATH_CALUDE_triangle_line_equations_l2497_249753

/-- Triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (-5, 0)
    B := (3, -3)
    C := (0, 2) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of line AC -/
def lineAC : LineEquation :=
  { a := 2
    b := -5
    c := 10 }

/-- The equation of the median to side BC -/
def medianBC : LineEquation :=
  { a := 1
    b := 13
    c := 5 }

theorem triangle_line_equations (t : Triangle) :
  t = givenTriangle →
  (lineAC.a * t.A.1 + lineAC.b * t.A.2 + lineAC.c = 0 ∧
   lineAC.a * t.C.1 + lineAC.b * t.C.2 + lineAC.c = 0) ∧
  (medianBC.a * t.A.1 + medianBC.b * t.A.2 + medianBC.c = 0 ∧
   medianBC.a * ((t.B.1 + t.C.1) / 2) + medianBC.b * ((t.B.2 + t.C.2) / 2) + medianBC.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l2497_249753


namespace NUMINAMATH_CALUDE_triangle_with_squares_sum_l2497_249738

/-- A right-angled triangle with two inscribed squares -/
structure TriangleWithSquares where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The areas of the inscribed squares
  area_s1 : ℝ
  area_s2 : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  inscribed_square1 : area_s1 = 40 * b + 1
  inscribed_square2 : area_s2 = 40 * b
  sum_sides : c = a + b

theorem triangle_with_squares_sum (t : TriangleWithSquares) : t.c = 462 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_squares_sum_l2497_249738


namespace NUMINAMATH_CALUDE_inequality_proof_l2497_249744

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2497_249744


namespace NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l2497_249780

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem z_minus_two_purely_imaginary (z : ℂ) (h : z = 2 - I) : 
  is_purely_imaginary (z - 2) := by
  sorry

end NUMINAMATH_CALUDE_z_minus_two_purely_imaginary_l2497_249780


namespace NUMINAMATH_CALUDE_student_count_pedro_grade_count_l2497_249711

/-- If a student is ranked both n-th best and n-th worst in a grade,
    then the total number of students in that grade is 2n - 1. -/
theorem student_count (n : ℕ) (h : n > 0) :
  ∃ (total : ℕ), total = 2 * n - 1 := by sorry

/-- There are 59 students in Pedro's grade. -/
theorem pedro_grade_count :
  ∃ (total : ℕ), total = 59 := by
  apply student_count 30
  norm_num

end NUMINAMATH_CALUDE_student_count_pedro_grade_count_l2497_249711


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l2497_249778

def is_solution (x y z : ℕ) : Prop :=
  (x + y) / z = (Nat.factorial x + Nat.factorial y) / Nat.factorial z

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, is_solution x y z →
    ((x = 1 ∧ y = 1 ∧ z = 2) ∨
     (x = 2 ∧ y = 2 ∧ z = 1) ∨
     (x = y ∧ y = z ∧ z ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l2497_249778


namespace NUMINAMATH_CALUDE_no_same_color_in_large_rectangle_l2497_249772

/-- A coloring of the plane is a function from pairs of integers to colors. -/
def Coloring (Color : Type) := ℤ × ℤ → Color

/-- A rectangle in the plane is defined by its top-left and bottom-right corners. -/
structure Rectangle :=
  (top_left : ℤ × ℤ)
  (bottom_right : ℤ × ℤ)

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℤ :=
  2 * (r.bottom_right.1 - r.top_left.1 + r.top_left.2 - r.bottom_right.2)

/-- A predicate that checks if a coloring satisfies the condition that
    no rectangle with perimeter 100 contains two squares of the same color. -/
def valid_coloring (c : Coloring (Fin 1201)) : Prop :=
  ∀ r : Rectangle, r.perimeter = 100 →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y

/-- The main theorem: if a coloring is valid, then no 1×1201 or 1201×1 rectangle
    contains two squares of the same color. -/
theorem no_same_color_in_large_rectangle
  (c : Coloring (Fin 1201)) (h : valid_coloring c) :
  (∀ r : Rectangle,
    (r.bottom_right.1 - r.top_left.1 = 1200 ∧ r.top_left.2 - r.bottom_right.2 = 0) ∨
    (r.bottom_right.1 - r.top_left.1 = 0 ∧ r.top_left.2 - r.bottom_right.2 = 1200) →
    ∀ x y : ℤ × ℤ, x ≠ y →
      x.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      x.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      y.1 ∈ Set.Icc r.top_left.1 r.bottom_right.1 →
      y.2 ∈ Set.Icc r.bottom_right.2 r.top_left.2 →
      c x ≠ c y) :=
by sorry

end NUMINAMATH_CALUDE_no_same_color_in_large_rectangle_l2497_249772


namespace NUMINAMATH_CALUDE_fraction_equality_l2497_249732

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a - b) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2497_249732


namespace NUMINAMATH_CALUDE_solve_equation_l2497_249786

theorem solve_equation : ∃ x : ℝ, 60 + 5 * x / (180 / 3) = 61 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2497_249786


namespace NUMINAMATH_CALUDE_investment_growth_l2497_249791

/-- The initial investment amount that grows to $563.35 after 5 years at 12% annual interest rate compounded yearly. -/
def initial_investment : ℝ := 319.77

/-- The final amount after 5 years of investment. -/
def final_amount : ℝ := 563.35

/-- The annual interest rate as a decimal. -/
def interest_rate : ℝ := 0.12

/-- The number of years the money is invested. -/
def years : ℕ := 5

/-- Theorem stating that the initial investment grows to the final amount after the specified time and interest rate. -/
theorem investment_growth :
  final_amount = initial_investment * (1 + interest_rate) ^ years := by
  sorry

#eval initial_investment

end NUMINAMATH_CALUDE_investment_growth_l2497_249791


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2023_l2497_249792

theorem sum_of_last_two_digits_of_9_pow_2023 : ∃ (a b : ℕ), 
  (9^2023 : ℕ) % 100 = 10 * a + b ∧ a + b = 11 := by sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_9_pow_2023_l2497_249792


namespace NUMINAMATH_CALUDE_circle_radius_l2497_249794

theorem circle_radius (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ 
  (∃ r : ℝ, r > 0 ∧ x = π * r^2 ∧ y = 2 * π * r) ∧
  x + y = 72 * π →
  ∃ r : ℝ, r = 6 ∧ x = π * r^2 ∧ y = 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2497_249794


namespace NUMINAMATH_CALUDE_average_of_last_three_l2497_249767

theorem average_of_last_three (A B C D : ℝ) : 
  A = 33 →
  D = 18 →
  (A + B + C) / 3 = 20 →
  (B + C + D) / 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l2497_249767


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2497_249771

theorem interest_rate_calculation (total amount : ℕ) (first_part : ℕ) (first_rate : ℚ) (yearly_income : ℕ) : 
  total = 2500 →
  first_part = 2000 →
  first_rate = 5/100 →
  yearly_income = 130 →
  ∃ second_rate : ℚ,
    (first_part * first_rate + (total - first_part) * second_rate = yearly_income) ∧
    second_rate = 6/100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2497_249771


namespace NUMINAMATH_CALUDE_robin_initial_distance_l2497_249759

/-- The distance Robin walked before realizing he forgot his bag -/
def initial_distance : ℝ := sorry

/-- The distance between Robin's house and the city center -/
def house_to_center : ℝ := 500

/-- The total distance Robin walked -/
def total_distance : ℝ := 900

theorem robin_initial_distance :
  initial_distance = 200 :=
by
  have journey_equation : 2 * initial_distance + house_to_center = total_distance := by sorry
  sorry

end NUMINAMATH_CALUDE_robin_initial_distance_l2497_249759


namespace NUMINAMATH_CALUDE_Q_no_real_roots_l2497_249715

def Q (x : ℝ) : ℝ := x^6 - 3*x^5 + 6*x^4 - 6*x^3 - x + 8

theorem Q_no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_Q_no_real_roots_l2497_249715


namespace NUMINAMATH_CALUDE_remaining_garden_area_is_48_l2497_249708

/-- The area of a rectangle with given length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- The dimensions of the large garden -/
def largeGardenLength : ℝ := 10
def largeGardenWidth : ℝ := 6

/-- The dimensions of the small plot -/
def smallPlotLength : ℝ := 4
def smallPlotWidth : ℝ := 3

/-- The remaining garden area after removing the small plot -/
def remainingGardenArea : ℝ :=
  rectangleArea largeGardenLength largeGardenWidth -
  rectangleArea smallPlotLength smallPlotWidth

theorem remaining_garden_area_is_48 :
  remainingGardenArea = 48 := by sorry

end NUMINAMATH_CALUDE_remaining_garden_area_is_48_l2497_249708


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2497_249773

theorem complex_expression_evaluation : 
  (0.027)^(-1/3 : ℝ) - (-1/7)^(-2 : ℝ) + (25/9 : ℝ)^(1/2 : ℝ) - (Real.sqrt 2 - 1)^(0 : ℝ) = -45 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2497_249773


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2497_249758

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 1711 ∧ ∀ x, 8 * x^2 - 24 * x + 1729 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2497_249758


namespace NUMINAMATH_CALUDE_cubic_polynomial_unique_l2497_249716

/-- A monic cubic polynomial with real coefficients -/
def cubic_polynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_unique 
  (q : ℝ → ℂ) 
  (h_monic : ∀ x, q x = x^3 + (q 1 - 1) * x^2 + (q 1 - q 0 - 1) * x + q 0)
  (h_root : q (5 - 3*I) = 0)
  (h_const : q 0 = 81) :
  ∀ x, q x = x^3 - (79/16)*x^2 - (17/8)*x + 81 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_unique_l2497_249716


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l2497_249733

theorem complex_in_second_quadrant : 
  let z : ℂ := Complex.mk (Real.cos 3) (Real.sin 3)
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l2497_249733


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2497_249798

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i :=
sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2497_249798


namespace NUMINAMATH_CALUDE_equation_solution_l2497_249768

theorem equation_solution : ∃! x : ℝ, 13 + Real.sqrt (-4 + 5 * x * 3) = 14 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2497_249768


namespace NUMINAMATH_CALUDE_blue_butterflies_count_l2497_249726

-- Define the variables
def total_butterflies : ℕ := 11
def black_butterflies : ℕ := 5

-- Define the theorem
theorem blue_butterflies_count :
  ∃ (blue yellow : ℕ),
    blue = 2 * yellow ∧
    blue + yellow + black_butterflies = total_butterflies ∧
    blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_butterflies_count_l2497_249726


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2497_249728

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- Define the variables
variables {a t x₁ x₂ : ℝ}

-- State the theorem
theorem isosceles_right_triangle_property
  (h1 : f a x₁ = 0)
  (h2 : f a x₂ = 0)
  (h3 : x₁ < x₂)
  (h4 : ∃ (c : ℝ), f a c = (x₂ - x₁) / 2 ∧ c = (x₁ + x₂) / 2)
  (h5 : t = Real.sqrt ((x₂ - 1) / (x₁ - 1)))
  : a * t - (a + t) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2497_249728


namespace NUMINAMATH_CALUDE_function_inequality_and_range_l2497_249784

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a + 2)*x + 4

-- Define the theorem
theorem function_inequality_and_range (a : ℝ) :
  -- Part 1: Solution sets for f(x) ≤ -2a + 4
  (a < 2 → {x : ℝ | f a x ≤ -2*a + 4} = {x : ℝ | a ≤ x ∧ x ≤ 2}) ∧
  (a = 2 → {x : ℝ | f a x ≤ -2*a + 4} = {x : ℝ | x = 2}) ∧
  (a > 2 → {x : ℝ | f a x ≤ -2*a + 4} = {x : ℝ | 2 ≤ x ∧ x ≤ a}) ∧
  -- Part 2: Range of a when f(x) + a + 1 ≥ 0 for x ∈ [1, 4]
  (∀ x ∈ Set.Icc 1 4, f a x + a + 1 ≥ 0) → a ∈ Set.Iic 4 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_range_l2497_249784


namespace NUMINAMATH_CALUDE_largest_value_l2497_249722

theorem largest_value (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b (max c d)) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2497_249722


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l2497_249756

theorem sum_of_squares_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → a*x^2 + b*x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

theorem sum_of_squares_of_specific_equation :
  let r₁ := (-(-14) + Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  let r₂ := (-(-14) - Real.sqrt ((-14)^2 - 4*1*8)) / (2*1)
  r₁^2 + r₂^2 = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_equation_l2497_249756


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l2497_249766

theorem modulo_residue_problem : (325 + 3 * 66 + 8 * 187 + 6 * 23) % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l2497_249766


namespace NUMINAMATH_CALUDE_movie_theater_total_movies_l2497_249727

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies_shown (num_screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  num_screens * (open_hours / movie_duration)

/-- Proves that a movie theater with 6 screens, open for 8 hours, showing 2-hour movies, shows 24 movies total. -/
theorem movie_theater_total_movies :
  total_movies_shown 6 8 2 = 24 := by
  sorry

#eval total_movies_shown 6 8 2

end NUMINAMATH_CALUDE_movie_theater_total_movies_l2497_249727


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2497_249763

/-- Parabola with vertex at origin and focus on positive x-axis -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_x_axis : focus.2 = 0 ∧ focus.1 > 0

/-- The curve E: x²+y²-6x+4y-3=0 -/
def curve_E (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 4*y - 3 = 0

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : point.2^2 = 2 * p.focus.1 * point.1

theorem parabola_point_coordinates (p : Parabola) 
  (h1 : ∃! x y, curve_E x y ∧ x = -p.focus.1) 
  (A : PointOnParabola p) 
  (h2 : A.point.1 * (A.point.1 - p.focus.1) + A.point.2 * A.point.2 = -4) :
  A.point = (1, 2) ∨ A.point = (1, -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2497_249763


namespace NUMINAMATH_CALUDE_multiples_count_multiples_count_equals_188_l2497_249782

theorem multiples_count : ℕ :=
  let range_start := 1
  let range_end := 600
  let count_multiples_of (n : ℕ) := (range_end / n : ℕ)
  let multiples_of_5 := count_multiples_of 5
  let multiples_of_7 := count_multiples_of 7
  let multiples_of_35 := count_multiples_of 35
  multiples_of_5 + multiples_of_7 - multiples_of_35

theorem multiples_count_equals_188 : multiples_count = 188 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_multiples_count_equals_188_l2497_249782


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2497_249700

theorem floor_equation_solution (x : ℝ) :
  ⌊⌊3 * x⌋ + (1 : ℝ) / 2⌋ = ⌊x + 3⌋ ↔ 4 / 3 ≤ x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2497_249700


namespace NUMINAMATH_CALUDE_max_area_and_optimal_length_l2497_249775

/-- Represents the dimensions and cost of a rectangular house. -/
structure House where
  x : ℝ  -- Length of front wall
  y : ℝ  -- Length of side wall
  h : ℝ  -- Height of the house
  coloredSteelPrice : ℝ  -- Price per meter of colored steel
  compositeSteelPrice : ℝ  -- Price per meter of composite steel
  roofPrice : ℝ  -- Price per square meter of roof

/-- Calculates the material cost of the house. -/
def materialCost (h : House) : ℝ :=
  2 * h.x * h.coloredSteelPrice * h.h +
  2 * h.y * h.compositeSteelPrice * h.h +
  h.x * h.y * h.roofPrice

/-- Calculates the area of the house. -/
def area (h : House) : ℝ := h.x * h.y

/-- Theorem stating the maximum area and optimal front wall length. -/
theorem max_area_and_optimal_length (h : House)
    (height_constraint : h.h = 2.5)
    (colored_steel_price : h.coloredSteelPrice = 450)
    (composite_steel_price : h.compositeSteelPrice = 200)
    (roof_price : h.roofPrice = 200)
    (cost_constraint : materialCost h ≤ 32000) :
    (∃ (max_area : ℝ) (optimal_x : ℝ),
      max_area = 100 ∧
      optimal_x = 20 / 3 ∧
      area h ≤ max_area ∧
      (area h = max_area ↔ h.x = optimal_x)) := by
  sorry


end NUMINAMATH_CALUDE_max_area_and_optimal_length_l2497_249775


namespace NUMINAMATH_CALUDE_water_displacement_squared_l2497_249746

def cube_side_length : ℝ := 12
def tank_radius : ℝ := 6
def tank_height : ℝ := 15

theorem water_displacement_squared :
  let cube_volume := cube_side_length ^ 3
  let cube_diagonal := cube_side_length * Real.sqrt 3
  cube_diagonal ≤ tank_height →
  (cube_volume ^ 2 : ℝ) = 2985984 := by sorry

end NUMINAMATH_CALUDE_water_displacement_squared_l2497_249746


namespace NUMINAMATH_CALUDE_unique_n_for_prime_roots_l2497_249724

/-- Determines if a natural number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The quadratic equation as a function of x and n -/
def quadraticEq (x n : ℕ) : ℤ :=
  2 * x^2 - 8*n*x + 10*x - n^2 + 35*n - 76

theorem unique_n_for_prime_roots :
  ∃! n : ℕ, ∃ x₁ x₂ : ℕ,
    x₁ ≠ x₂ ∧
    isPrime x₁ ∧
    isPrime x₂ ∧
    quadraticEq x₁ n = 0 ∧
    quadraticEq x₂ n = 0 ∧
    n = 3 ∧
    x₁ = 2 ∧
    x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_roots_l2497_249724


namespace NUMINAMATH_CALUDE_xiao_dong_jump_distance_l2497_249721

/-- Given a standard jump distance and a recorded result, calculate the actual jump distance. -/
def actual_jump_distance (standard : ℝ) (recorded : ℝ) : ℝ :=
  standard + recorded

/-- Theorem: For a standard jump distance of 4.00 meters and a recorded result of -0.32,
    the actual jump distance is 3.68 meters. -/
theorem xiao_dong_jump_distance :
  let standard : ℝ := 4.00
  let recorded : ℝ := -0.32
  actual_jump_distance standard recorded = 3.68 := by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_jump_distance_l2497_249721


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l2497_249723

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| over all real y is 4√2 -/
theorem min_max_abs_quadratic_minus_linear :
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| ≥ 4 * Real.sqrt 2) ∧
  (∃ y : ℝ, ∀ x : ℝ, 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l2497_249723


namespace NUMINAMATH_CALUDE_ln_lower_bound_l2497_249797

theorem ln_lower_bound (n : ℕ) (k : ℕ) (h : k = (Nat.factorization n).support.card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ln_lower_bound_l2497_249797


namespace NUMINAMATH_CALUDE_race_theorem_l2497_249719

def race_problem (john_speed : ℝ) (race_distance : ℝ) (winning_margin : ℝ) : Prop :=
  let john_time := race_distance / john_speed * 60
  let next_fastest_time := john_time + winning_margin
  next_fastest_time = 23

theorem race_theorem :
  race_problem 15 5 3 := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l2497_249719


namespace NUMINAMATH_CALUDE_scientific_notation_3930_billion_l2497_249739

-- Define billion as 10^9
def billion : ℕ := 10^9

-- Theorem to prove the equality
theorem scientific_notation_3930_billion :
  (3930 : ℝ) * billion = 3.93 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3930_billion_l2497_249739


namespace NUMINAMATH_CALUDE_triangle_dot_product_l2497_249788

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area √3, prove AB · AC = ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  (AB.1 * AC.1 + AB.2 * AC.2 = 2) ∨ (AB.1 * AC.1 + AB.2 * AC.2 = -2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l2497_249788


namespace NUMINAMATH_CALUDE_second_question_percentage_l2497_249783

theorem second_question_percentage 
  (first_correct : Real) 
  (neither_correct : Real) 
  (both_correct : Real) 
  (h1 : first_correct = 0.75) 
  (h2 : neither_correct = 0.2) 
  (h3 : both_correct = 0.65) : 
  ∃ (second_correct : Real), second_correct = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_second_question_percentage_l2497_249783


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2497_249795

theorem least_value_quadratic (x : ℝ) :
  (4 * x^2 + 7 * x + 3 = 5) → x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2497_249795


namespace NUMINAMATH_CALUDE_sandbox_width_l2497_249757

theorem sandbox_width (perimeter : ℝ) (width : ℝ) (length : ℝ) : 
  perimeter = 30 →
  length = 2 * width →
  perimeter = 2 * width + 2 * length →
  width = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l2497_249757


namespace NUMINAMATH_CALUDE_welders_proof_l2497_249741

/-- Represents the initial number of welders -/
def initial_welders : ℕ := 12

/-- Represents the number of days initially needed to complete the order -/
def initial_days : ℕ := 3

/-- Represents the number of welders that leave after the first day -/
def welders_left : ℕ := 9

/-- Represents the additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 8

/-- Proves that the initial number of welders is correct given the conditions -/
theorem welders_proof :
  (initial_welders - welders_left) * additional_days = initial_welders * (initial_days - 1) :=
by sorry

end NUMINAMATH_CALUDE_welders_proof_l2497_249741


namespace NUMINAMATH_CALUDE_max_min_kangaroo_weight_l2497_249787

theorem max_min_kangaroo_weight :
  ∀ (a b c : ℕ),
    a > 0 → b > 0 → c > 0 →
    a + b + c = 97 →
    min a (min b c) ≤ 32 ∧
    ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 97 ∧ min x (min y z) = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_min_kangaroo_weight_l2497_249787


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2497_249702

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 5 * (x - 2 * x^3) - 4 * (2 * x^2 - x^3 + 3 * x^6) + 3 * (5 * x^2 - 2 * x^8)

-- Theorem stating that the coefficient of x^2 in the polynomial is 7
theorem x_squared_coefficient : (deriv (deriv poly)) 0 / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2497_249702


namespace NUMINAMATH_CALUDE_average_of_sqrt_equation_l2497_249777

theorem average_of_sqrt_equation (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, Real.sqrt (3 * x^2 + 4 * x + 1) = Real.sqrt 37 ↔ x = x₁ ∨ x = x₂) ∧
  (x₁ + x₂) / 2 = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_average_of_sqrt_equation_l2497_249777


namespace NUMINAMATH_CALUDE_cans_left_to_load_l2497_249793

/-- Given a packing scenario for canned juice, prove the number of cans left to be loaded. -/
theorem cans_left_to_load 
  (cans_per_carton : ℕ) 
  (total_cartons : ℕ) 
  (loaded_cartons : ℕ) 
  (h1 : cans_per_carton = 20)
  (h2 : total_cartons = 50)
  (h3 : loaded_cartons = 40) :
  (total_cartons - loaded_cartons) * cans_per_carton = 200 :=
by sorry

end NUMINAMATH_CALUDE_cans_left_to_load_l2497_249793


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l2497_249704

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (a - 2) * x + 1 ≠ 0

def prop_q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + a * x + 1)

def range_of_a : Set ℝ := Set.Iic (-2) ∪ Set.Ioo 1 2 ∪ Set.Ici 3

theorem range_of_a_theorem :
  (∀ a : ℝ, (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a)) →
  {a : ℝ | prop_p a ∨ prop_q a} = range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l2497_249704


namespace NUMINAMATH_CALUDE_room_width_is_seven_l2497_249705

/-- Represents the dimensions and features of a room -/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  doorCount : ℕ
  doorArea : ℝ
  largeWindowCount : ℕ
  largeWindowArea : ℝ
  smallWindowCount : ℕ
  smallWindowArea : ℝ
  paintCostPerSqM : ℝ
  totalPaintCost : ℝ

/-- Calculates the paintable area of the room -/
def paintableArea (r : Room) : ℝ :=
  2 * (r.height * r.length + r.height * r.width) -
  (r.doorCount * r.doorArea + r.largeWindowCount * r.largeWindowArea + r.smallWindowCount * r.smallWindowArea)

/-- Theorem stating that the width of the room is 7 meters -/
theorem room_width_is_seven (r : Room) 
  (h1 : r.length = 10)
  (h2 : r.height = 5)
  (h3 : r.doorCount = 2)
  (h4 : r.doorArea = 3)
  (h5 : r.largeWindowCount = 1)
  (h6 : r.largeWindowArea = 3)
  (h7 : r.smallWindowCount = 2)
  (h8 : r.smallWindowArea = 1.5)
  (h9 : r.paintCostPerSqM = 3)
  (h10 : r.totalPaintCost = 474)
  (h11 : paintableArea r * r.paintCostPerSqM = r.totalPaintCost) :
  r.width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_seven_l2497_249705


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2497_249790

theorem quadratic_equation_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 4*x₁ - 4 = 0) ∧ (x₂^2 + 4*x₂ - 4 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2497_249790


namespace NUMINAMATH_CALUDE_problem_solution_l2497_249742

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2497_249742


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2497_249712

theorem quadratic_equation_completion_square :
  ∃ (d e : ℤ), (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + d)^2 = e) ∧ d + e = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2497_249712


namespace NUMINAMATH_CALUDE_simplify_expression_l2497_249750

theorem simplify_expression (x : ℝ) : (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2497_249750


namespace NUMINAMATH_CALUDE_creature_perimeter_l2497_249743

/-- The perimeter of a circular creature with an open mouth -/
theorem creature_perimeter (r : ℝ) (central_angle : ℝ) : 
  r = 2 → central_angle = 270 → 
  (central_angle / 360) * (2 * π * r) + 2 * r = 3 * π + 4 :=
by sorry

end NUMINAMATH_CALUDE_creature_perimeter_l2497_249743


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l2497_249713

theorem scientific_notation_conversion :
  (2.61 * 10^(-5) = 0.0000261) ∧ (0.00068 = 6.8 * 10^(-4)) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l2497_249713


namespace NUMINAMATH_CALUDE_birth_death_rate_interval_birth_death_rate_problem_l2497_249755

theorem birth_death_rate_interval (birth_rate : ℕ) (death_rate : ℕ) (daily_increase : ℕ) : ℕ :=
  let net_rate := birth_rate - death_rate
  let intervals_per_day := daily_increase / net_rate
  let minutes_per_day := 24 * 60
  minutes_per_day / intervals_per_day

theorem birth_death_rate_problem :
  birth_death_rate_interval 10 2 345600 = 48 := by
  sorry

end NUMINAMATH_CALUDE_birth_death_rate_interval_birth_death_rate_problem_l2497_249755


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l2497_249765

-- Define the angle A
def angle_A : ℝ := 42

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 48 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l2497_249765


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2497_249720

/-- A linear function that decreases as x increases and satisfies kb > 0 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant
  (k b : ℝ) -- k and b are real numbers
  (h1 : k * b > 0) -- condition: kb > 0
  (h2 : k < 0) -- condition: y decreases as x increases
  : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l2497_249720


namespace NUMINAMATH_CALUDE_polynomial_degree_is_8_l2497_249730

def polynomial_degree (x : ℝ) : ℕ :=
  let expr1 := x^7
  let expr2 := x + 1/x
  let expr3 := 1 + 3/x + 5/(x^2)
  let result := expr1 * expr2 * expr3
  8  -- The degree of the resulting polynomial

theorem polynomial_degree_is_8 : 
  ∀ x : ℝ, x ≠ 0 → polynomial_degree x = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_is_8_l2497_249730


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2497_249747

theorem arithmetic_expression_evaluation : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l2497_249747


namespace NUMINAMATH_CALUDE_logical_judgment_structures_l2497_249761

-- Define the basic structures of algorithms
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures that require logical judgment
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => True
  | AlgorithmStructure.Loop => True
  | _ => False

-- Theorem statement
theorem logical_judgment_structures :
  ∀ s : AlgorithmStructure,
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end NUMINAMATH_CALUDE_logical_judgment_structures_l2497_249761


namespace NUMINAMATH_CALUDE_orange_shirt_cost_l2497_249701

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def yellow_shirt_cost : ℕ := 500  -- $5.00
def blue_shirt_cost : ℕ := 560    -- $5.60
def green_shirt_cost : ℕ := 525   -- $5.25

-- Define the total amount spent by P.T.O. (in cents)
def total_spent : ℕ := 231700  -- $2,317.00

-- Theorem to prove
theorem orange_shirt_cost :
  (total_spent
    - (first_graders * yellow_shirt_cost
    + second_graders * blue_shirt_cost
    + third_graders * green_shirt_cost))
  / kindergartners = 580 := by
  sorry

end NUMINAMATH_CALUDE_orange_shirt_cost_l2497_249701


namespace NUMINAMATH_CALUDE_alpha_value_l2497_249731

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - Real.pi / 18) = Real.sqrt 3 / 2) : 
  α = Real.pi * 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2497_249731


namespace NUMINAMATH_CALUDE_total_heads_count_l2497_249769

/-- The number of feet per hen -/
def henFeet : ℕ := 2

/-- The number of feet per cow -/
def cowFeet : ℕ := 4

/-- Theorem: Given a group of hens and cows, if the total number of feet is 140
    and there are 26 hens, then the total number of heads is 48. -/
theorem total_heads_count (totalFeet : ℕ) (henCount : ℕ) : 
  totalFeet = 140 → henCount = 26 → henCount * henFeet + (totalFeet - henCount * henFeet) / cowFeet = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_count_l2497_249769


namespace NUMINAMATH_CALUDE_prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l2497_249737

/-- Represents the probability distribution of a discrete random variable -/
def ProbabilityDistribution (α : Type*) := α → ℚ

/-- The total number of students in the group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one boy when choosing two students -/
def prob_one_boy : ℚ := 3/5

/-- Represents the number of boys selected -/
inductive X where
  | zero : X
  | one : X
  | two : X

/-- The probability distribution of X (number of boys selected) -/
def dist_X : ProbabilityDistribution X :=
  fun x => match x with
    | X.zero => 1/10
    | X.one  => 3/5
    | X.two  => 3/10

/-- Theorem stating the probability of selecting exactly one boy is correct -/
theorem prob_one_boy_correct :
  prob_one_boy = 3/5 := by sorry

/-- Theorem stating the probability distribution of X is correct -/
theorem dist_X_correct :
  dist_X X.zero = 1/10 ∧
  dist_X X.one  = 3/5  ∧
  dist_X X.two  = 3/10 := by sorry

/-- Theorem stating the sum of probabilities in the distribution equals 1 -/
theorem dist_X_sum_to_one :
  dist_X X.zero + dist_X X.one + dist_X X.two = 1 := by sorry

end NUMINAMATH_CALUDE_prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l2497_249737


namespace NUMINAMATH_CALUDE_cos_minus_sin_2pi_non_decreasing_l2497_249764

def T_non_decreasing (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) ≥ f x

theorem cos_minus_sin_2pi_non_decreasing :
  T_non_decreasing (fun x => Real.cos x - Real.sin x) (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_2pi_non_decreasing_l2497_249764


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2497_249785

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2497_249785


namespace NUMINAMATH_CALUDE_sum_of_coefficients_P_l2497_249752

/-- The polynomial in question -/
def P (x : ℝ) : ℝ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

/-- Theorem stating that the sum of coefficients of P is 2019 -/
theorem sum_of_coefficients_P : (P 1) = 2019 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_P_l2497_249752


namespace NUMINAMATH_CALUDE_prime_factors_of_N_l2497_249707

def N : ℕ := (10^2011 - 1) / 9

theorem prime_factors_of_N (p : ℕ) (hp : p.Prime) (hdiv : p ∣ N) :
  ∃ j : ℕ, p = 4022 * j + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_N_l2497_249707


namespace NUMINAMATH_CALUDE_simplify_expression_l2497_249779

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 4*(2^(n+1))) / (4*(2^(n+4))) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2497_249779


namespace NUMINAMATH_CALUDE_min_production_avoid_losses_l2497_249718

/-- The minimum daily production of gloves to avoid losses -/
def min_production : ℕ := 800

/-- The total daily production cost (in yuan) as a function of daily production volume (in pairs) -/
def total_cost (x : ℕ) : ℕ := 5 * x + 4000

/-- The factory price per pair of gloves (in yuan) -/
def price_per_pair : ℕ := 10

/-- The daily revenue (in yuan) as a function of daily production volume (in pairs) -/
def revenue (x : ℕ) : ℕ := price_per_pair * x

/-- Theorem stating that the minimum daily production to avoid losses is 800 pairs -/
theorem min_production_avoid_losses :
  ∀ x : ℕ, x ≥ min_production ↔ revenue x ≥ total_cost x :=
sorry

end NUMINAMATH_CALUDE_min_production_avoid_losses_l2497_249718


namespace NUMINAMATH_CALUDE_equation_solutions_l2497_249748

theorem equation_solutions :
  (∃ x : ℚ, (5*x - 1)/4 = (3*x + 1)/2 - (2 - x)/3 ↔ x = -1/7) ∧
  (∃ x : ℚ, (3*x + 2)/2 - 1 = (2*x - 1)/4 - (2*x + 1)/5 ↔ x = -9/28) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2497_249748


namespace NUMINAMATH_CALUDE_triangle_shape_l2497_249781

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that if 
    a/cos(A) = b/cos(B) = c/cos(C) and sin(A) = 2sin(B)cos(C), then A = B = C. -/
theorem triangle_shape (a b c A B C : ℝ) 
    (h1 : a / Real.cos A = b / Real.cos B) 
    (h2 : b / Real.cos B = c / Real.cos C)
    (h3 : Real.sin A = 2 * Real.sin B * Real.cos C)
    (h4 : 0 < A ∧ A < π)
    (h5 : 0 < B ∧ B < π)
    (h6 : 0 < C ∧ C < π)
    (h7 : A + B + C = π) : 
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l2497_249781
