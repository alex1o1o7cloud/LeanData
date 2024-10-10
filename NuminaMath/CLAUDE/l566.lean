import Mathlib

namespace fourth_quarter_profits_l566_56698

/-- Proves that given the annual profits, first quarter profits, and third quarter profits,
    the fourth quarter profits are equal to the difference between the annual profits
    and the sum of the first and third quarter profits. -/
theorem fourth_quarter_profits
  (annual_profits : ℕ)
  (first_quarter_profits : ℕ)
  (third_quarter_profits : ℕ)
  (h1 : annual_profits = 8000)
  (h2 : first_quarter_profits = 1500)
  (h3 : third_quarter_profits = 3000) :
  annual_profits - (first_quarter_profits + third_quarter_profits) = 3500 :=
by sorry

end fourth_quarter_profits_l566_56698


namespace arithmetic_geometric_inequality_two_variables_l566_56689

theorem arithmetic_geometric_inequality_two_variables 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ 
  ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end arithmetic_geometric_inequality_two_variables_l566_56689


namespace age_system_properties_l566_56639

/-- Represents the ages and aging rates of four people -/
structure AgeSystem where
  a : ℝ  -- Age of person A
  b : ℝ  -- Age of person B
  c : ℝ  -- Age of person C
  d : ℝ  -- Age of person D
  x : ℝ  -- Age difference between A and C
  y : ℝ  -- Number of years passed
  rA : ℝ  -- Aging rate of A relative to C
  rB : ℝ  -- Aging rate of B relative to C
  rD : ℝ  -- Aging rate of D relative to C

/-- The age system satisfies the given conditions -/
def satisfiesConditions (s : AgeSystem) : Prop :=
  s.a + s.b = 13 + (s.b + s.c) ∧
  s.c = s.a - s.x ∧
  s.a + s.d = 2 * (s.b + s.c) ∧
  (s.a + s.rA * s.y) + (s.b + s.rB * s.y) = 25 + (s.b + s.rB * s.y) + (s.c + s.y)

/-- Theorem stating the properties of the age system -/
theorem age_system_properties (s : AgeSystem) 
  (h : satisfiesConditions s) : 
  s.x = 13 ∧ s.d = 2 * s.b + s.a - 26 ∧ s.rA * s.y = 12 + s.y := by
  sorry


end age_system_properties_l566_56639


namespace greatest_triangle_perimeter_l566_56641

theorem greatest_triangle_perimeter : ∃ (a b c : ℕ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧  -- positive integer side lengths
  (b = 4 * a) ∧              -- one side is four times as long as a second side
  (c = 20) ∧                 -- the third side has length 20
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧  -- triangle inequality
  (∀ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) → 
    (y = 4 * x) → (z = 20) → 
    (x + y > z ∧ y + z > x ∧ z + x > y) →
    (x + y + z ≤ a + b + c)) ∧
  (a + b + c = 50) :=
by sorry

end greatest_triangle_perimeter_l566_56641


namespace fun_run_no_shows_fun_run_no_shows_solution_l566_56687

/-- Fun Run Attendance Problem -/
theorem fun_run_no_shows (signed_up_last_year : ℕ) (runners_this_year : ℕ) : ℕ :=
  let runners_last_year := runners_this_year / 2
  signed_up_last_year - runners_last_year

/-- The number of people who did not show up to run last year is 40 -/
theorem fun_run_no_shows_solution : fun_run_no_shows 200 320 = 40 := by
  sorry

end fun_run_no_shows_fun_run_no_shows_solution_l566_56687


namespace logarithm_expression_evaluation_l566_56699

theorem logarithm_expression_evaluation :
  Real.log 5 / Real.log 10 + Real.log 2 / Real.log 10 + (3/5)^0 + Real.log (Real.exp (1/2)) = 5/2 := by
  sorry

end logarithm_expression_evaluation_l566_56699


namespace trigonometric_problem_l566_56659

theorem trigonometric_problem (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α = 7/5) ∧
  (Real.sin (2 * α + π/3) = -12/25 - 7 * Real.sqrt 3 / 50) := by
  sorry

end trigonometric_problem_l566_56659


namespace adam_first_year_students_l566_56630

/-- The number of students Adam teaches per year after the first year -/
def students_per_year : ℕ := 50

/-- The total number of years Adam teaches -/
def total_years : ℕ := 10

/-- The total number of students Adam teaches in 10 years -/
def total_students : ℕ := 490

/-- The number of students Adam taught in the first year -/
def first_year_students : ℕ := total_students - (students_per_year * (total_years - 1))

theorem adam_first_year_students :
  first_year_students = 40 := by sorry

end adam_first_year_students_l566_56630


namespace functional_equation_problem_l566_56661

/-- The functional equation problem -/
theorem functional_equation_problem (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f (y * f x - 1) = x^2 * f y - f x) ↔ (∀ x : ℝ, f x = x) :=
sorry

end functional_equation_problem_l566_56661


namespace count_squares_below_line_l566_56629

/-- The number of 1x1 squares in the first quadrant entirely below the line 6x + 143y = 858 -/
def squares_below_line : ℕ :=
  355

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop :=
  6 * x + 143 * y = 858

theorem count_squares_below_line :
  squares_below_line = 355 :=
sorry

end count_squares_below_line_l566_56629


namespace min_value_x_l566_56614

theorem min_value_x (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
  (h4 : ∀ a b, a > 0 → b > 0 → (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))
  (h5 : ∀ a b, a > 0 → b > 0 → (4*a + b*(1 - a) = 0)) :
  x ≥ 1 :=
sorry

end min_value_x_l566_56614


namespace range_of_a_l566_56683

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 2^x - a = 1/(x-1)) → 0 < a ∧ a < 2 := by
  sorry

end range_of_a_l566_56683


namespace solution_in_interval_l566_56652

-- Define the function f(x) = x^3 + x - 4
def f (x : ℝ) : ℝ := x^3 + x - 4

-- State the theorem
theorem solution_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 (3/2) ∧ f x = 0 :=
by
  sorry

end solution_in_interval_l566_56652


namespace percentage_parents_without_full_time_jobs_l566_56601

theorem percentage_parents_without_full_time_jobs :
  ∀ (total_parents : ℕ) (mothers fathers : ℕ),
    mothers + fathers = total_parents →
    mothers = (2 / 5 : ℚ) * total_parents →
    (3 / 4 : ℚ) * mothers + (9 / 10 : ℚ) * fathers = (21 / 25 : ℚ) * total_parents :=
by
  sorry

end percentage_parents_without_full_time_jobs_l566_56601


namespace complex_fraction_simplification_l566_56692

theorem complex_fraction_simplification :
  (5 + 3*Complex.I) / (2 + 3*Complex.I) = 19/13 - 9/13 * Complex.I :=
by sorry

end complex_fraction_simplification_l566_56692


namespace sand_in_partial_bag_l566_56610

theorem sand_in_partial_bag (total_sand : ℝ) (bag_capacity : ℝ) (h1 : total_sand = 1254.75) (h2 : bag_capacity = 73.5) :
  total_sand - (bag_capacity * ⌊total_sand / bag_capacity⌋) = 5.25 := by
  sorry

end sand_in_partial_bag_l566_56610


namespace sin_sixty_degrees_l566_56681

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l566_56681


namespace nth_equation_l566_56636

theorem nth_equation (n : ℕ+) : (10 * n + 5)^2 = n * (n + 1) * 100 + 5^2 := by
  sorry

end nth_equation_l566_56636


namespace exists_same_color_four_directions_l566_56674

/-- A color in the grid -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- A position in the grid -/
structure Position where
  x : Fin 50
  y : Fin 50

/-- A coloring of the grid -/
def Coloring := Position → Color

/-- A position has a same-colored square above it -/
def has_same_color_above (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y > p.y ∧ c q = c p

/-- A position has a same-colored square below it -/
def has_same_color_below (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.x = p.x ∧ q.y < p.y ∧ c q = c p

/-- A position has a same-colored square to its left -/
def has_same_color_left (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x < p.x ∧ c q = c p

/-- A position has a same-colored square to its right -/
def has_same_color_right (c : Coloring) (p : Position) : Prop :=
  ∃ q : Position, q.y = p.y ∧ q.x > p.x ∧ c q = c p

/-- Main theorem: There exists a position with same-colored squares in all four directions -/
theorem exists_same_color_four_directions (c : Coloring) : 
  ∃ p : Position, 
    has_same_color_above c p ∧ 
    has_same_color_below c p ∧ 
    has_same_color_left c p ∧ 
    has_same_color_right c p := by
  sorry

end exists_same_color_four_directions_l566_56674


namespace frame_ratio_l566_56606

theorem frame_ratio (x : ℝ) (h : x > 0) : 
  (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 →
  (20 + 2*x) / (30 + 6*x) = 1/2 := by
sorry

end frame_ratio_l566_56606


namespace polynomial_roots_l566_56609

theorem polynomial_roots : 
  let p : ℝ → ℝ := fun x ↦ x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by sorry

end polynomial_roots_l566_56609


namespace right_triangle_perimeter_equals_area_l566_56603

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (a * b) / 2 →
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨
   (a = 12 ∧ b = 5 ∧ c = 13) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end right_triangle_perimeter_equals_area_l566_56603


namespace probability_of_green_ball_l566_56617

theorem probability_of_green_ball (total_balls green_balls : ℕ) 
  (h1 : total_balls = 10)
  (h2 : green_balls = 4) : 
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end probability_of_green_ball_l566_56617


namespace sehnenviereck_ungleichung_infinitely_many_equality_cases_l566_56693

theorem sehnenviereck_ungleichung (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) (hd1 : d < 1)
  (sum : a + b + c + d = 2) :
  Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (a * c + b * d) / 2 :=
sorry

theorem infinitely_many_equality_cases :
  ∃ S : Set (ℝ × ℝ × ℝ × ℝ), Cardinal.mk S = Cardinal.mk ℝ ∧
  ∀ (a b c d : ℝ), (a, b, c, d) ∈ S →
    0 < a ∧ a < 1 ∧
    0 < b ∧ b < 1 ∧
    0 < c ∧ c < 1 ∧
    0 < d ∧ d < 1 ∧
    a + b + c + d = 2 ∧
    Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) = (a * c + b * d) / 2 :=
sorry

end sehnenviereck_ungleichung_infinitely_many_equality_cases_l566_56693


namespace orthic_triangle_smallest_perimeter_l566_56657

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Checks if a point is on a line segment between two other points -/
def isOnSegment (P : Point) (A : Point) (B : Point) : Prop := sorry

/-- Checks if a triangle is inscribed in another triangle -/
def isInscribed (inner outer : Triangle) : Prop := 
  isOnSegment inner.A outer.B outer.C ∧
  isOnSegment inner.B outer.A outer.C ∧
  isOnSegment inner.C outer.A outer.B

/-- Constructs the orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle := sorry

/-- The main theorem: the orthic triangle has the smallest perimeter among all inscribed triangles -/
theorem orthic_triangle_smallest_perimeter (ABC : Triangle) 
  (h_acute : isAcuteAngled ABC) :
  let PQR := orthicTriangle ABC
  ∀ XYZ : Triangle, isInscribed XYZ ABC → perimeter PQR ≤ perimeter XYZ := by
  sorry

end orthic_triangle_smallest_perimeter_l566_56657


namespace vasyas_capital_decreases_l566_56672

/-- Represents the change in Vasya's capital after a series of trading days -/
def vasyas_capital_change (num_unsuccessful_days : ℕ) : ℝ :=
  (1.1^2 * 0.8)^num_unsuccessful_days

/-- Theorem stating that Vasya's capital decreases -/
theorem vasyas_capital_decreases (num_unsuccessful_days : ℕ) :
  vasyas_capital_change num_unsuccessful_days < 1 := by
  sorry

#check vasyas_capital_decreases

end vasyas_capital_decreases_l566_56672


namespace non_shaded_perimeter_16_l566_56670

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter_16 (outer : Rectangle) (inner : Rectangle) (shaded_area : ℝ) :
  outer.width = 12 ∧ 
  outer.height = 10 ∧ 
  inner.width = 5 ∧ 
  inner.height = 3 ∧
  shaded_area = 120 →
  perimeter inner = 16 := by
sorry

end non_shaded_perimeter_16_l566_56670


namespace product_of_two_digit_numbers_l566_56646

theorem product_of_two_digit_numbers (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a ≤ 99) 
  (h2 : 10 ≤ b ∧ b ≤ 99) 
  (h3 : a * b = 4500) 
  (h4 : a ≤ b) : 
  a = 50 := by
sorry

end product_of_two_digit_numbers_l566_56646


namespace inequality_solution_set_l566_56618

theorem inequality_solution_set : 
  {x : ℝ | 2 * (x^2 - x) < 4} = {x : ℝ | -1 < x ∧ x < 2} := by
sorry

end inequality_solution_set_l566_56618


namespace probability_two_absent_one_present_l566_56615

theorem probability_two_absent_one_present :
  let p_absent : ℚ := 1 / 30
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 
    3 * p_absent * p_absent * p_present
  p_two_absent_one_present = 29 / 9000 := by
  sorry

end probability_two_absent_one_present_l566_56615


namespace inverse_sum_equals_negative_two_l566_56668

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_equals_negative_two :
  (Function.invFun g) 8 + (Function.invFun g) (-64) = -2 := by
  sorry

end inverse_sum_equals_negative_two_l566_56668


namespace smallest_positive_integer_with_remainders_l566_56649

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ, 
  x > 0 ∧ 
  x % 5 = 4 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y :=
by
  sorry

end smallest_positive_integer_with_remainders_l566_56649


namespace friendly_numbers_solution_l566_56647

/-- Two rational numbers are friendly if their sum is 66 -/
def friendly (m n : ℚ) : Prop := m + n = 66

/-- Given that 7x and -18 are friendly numbers, prove that x = 12 -/
theorem friendly_numbers_solution (x : ℚ) (h : friendly (7 * x) (-18)) : x = 12 := by
  sorry

end friendly_numbers_solution_l566_56647


namespace tank_length_proof_l566_56640

/-- Proves that a tank with given dimensions and plastering cost has a specific length -/
theorem tank_length_proof (width depth L : ℝ) (plastering_rate : ℝ) (total_cost : ℝ) : 
  width = 12 →
  depth = 6 →
  plastering_rate = 75 / 100 →
  total_cost = 558 →
  (2 * depth * L + 2 * depth * width + width * L) * plastering_rate = total_cost →
  L = 25 := by
sorry


end tank_length_proof_l566_56640


namespace imaginary_part_of_z_l566_56663

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) = 2) : 
  Complex.im z = -4/5 := by sorry

end imaginary_part_of_z_l566_56663


namespace train_speed_problem_l566_56650

theorem train_speed_problem (v : ℝ) : 
  v > 0 →  -- The speed of the first train is positive
  (∃ t : ℝ, t > 0 ∧  -- There exists a positive time t when the trains meet
    v * t = 25 * t + 60 ∧  -- One train travels 60 km more than the other
    v * t + 25 * t = 540) →  -- Total distance traveled equals the distance between stations
  v = 31.25 := by
sorry

end train_speed_problem_l566_56650


namespace inequality_one_inequality_two_l566_56688

-- Part (1)
theorem inequality_one (x : ℝ) :
  (2 < |2*x - 5| ∧ |2*x - 5| ≤ 7) ↔ ((-1 ≤ x ∧ x < 3/2) ∨ (7/2 < x ∧ x ≤ 6)) :=
sorry

-- Part (2)
theorem inequality_two (x : ℝ) :
  (1 / (x - 1) > x + 1) ↔ (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) :=
sorry

end inequality_one_inequality_two_l566_56688


namespace vector_position_at_negative_two_l566_56653

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  position : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfying the problem conditions -/
def given_line : ParametricLine :=
  { position := sorry }

theorem vector_position_at_negative_two :
  let l := given_line
  (l.position 1 = (2, 0, -3)) →
  (l.position 2 = (7, -2, 1)) →
  (l.position 4 = (17, -6, 9)) →
  l.position (-2) = (-1, 3, -9) := by
  sorry

end vector_position_at_negative_two_l566_56653


namespace comic_book_frames_l566_56643

/-- The number of frames per page in Julian's comic book -/
def frames_per_page : ℝ := 143.0

/-- The number of pages in Julian's comic book -/
def pages : ℝ := 11.0

/-- The total number of frames in Julian's comic book -/
def total_frames : ℝ := frames_per_page * pages

theorem comic_book_frames :
  total_frames = 1573.0 := by sorry

end comic_book_frames_l566_56643


namespace mean_height_is_60_l566_56694

/-- Represents the stem and leaf plot of player heights --/
def stemAndLeaf : List (Nat × List Nat) := [
  (4, [9]),
  (5, [2, 3, 5, 8, 8, 9]),
  (6, [0, 1, 1, 2, 6, 8, 9, 9])
]

/-- Calculates the total sum of heights from the stem and leaf plot --/
def sumHeights (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (stem, leaves) => 
    acc + stem * 10 * leaves.length + leaves.sum
  ) 0

/-- Calculates the number of players from the stem and leaf plot --/
def countPlayers (plot : List (Nat × List Nat)) : Nat :=
  plot.foldl (fun acc (_, leaves) => acc + leaves.length) 0

/-- The mean height of the players --/
def meanHeight : ℚ := (sumHeights stemAndLeaf : ℚ) / (countPlayers stemAndLeaf : ℚ)

theorem mean_height_is_60 : meanHeight = 60 := by
  sorry

end mean_height_is_60_l566_56694


namespace expansion_coefficient_l566_56628

/-- The coefficient of a^2 * b^3 * c^3 in the expansion of (a + b + c)^8 -/
def coefficient_a2b3c3 : ℕ :=
  Nat.choose 8 5 * Nat.choose 5 3

theorem expansion_coefficient :
  coefficient_a2b3c3 = 560 := by sorry

end expansion_coefficient_l566_56628


namespace union_M_N_l566_56664

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by sorry

end union_M_N_l566_56664


namespace special_sequence_sum_2017_l566_56605

/-- A sequence with special properties -/
def SpecialSequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → S (n + 1) - S n = 3^n / a n

/-- The sum of the first 2017 terms of the special sequence -/
theorem special_sequence_sum_2017 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SpecialSequence a S) : S 2017 = 3^1009 - 2 := by
  sorry

end special_sequence_sum_2017_l566_56605


namespace right_triangle_hypotenuse_l566_56608

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a = 1994 →         -- One cathetus is 1994
  c = 994010         -- Hypotenuse is 994010
  := by sorry

end right_triangle_hypotenuse_l566_56608


namespace complex_division_result_l566_56619

theorem complex_division_result : Complex.I / (1 - Complex.I) = -1/2 + Complex.I/2 := by
  sorry

end complex_division_result_l566_56619


namespace michael_small_balls_l566_56645

/-- Represents the number of rubber bands in a pack --/
def total_rubber_bands : ℕ := 5000

/-- Represents the number of rubber bands needed for a small ball --/
def small_ball_rubber_bands : ℕ := 50

/-- Represents the number of rubber bands needed for a large ball --/
def large_ball_rubber_bands : ℕ := 300

/-- Represents the number of large balls that can be made with remaining rubber bands --/
def remaining_large_balls : ℕ := 13

/-- Calculates the number of small balls Michael made --/
def small_balls_made : ℕ :=
  (total_rubber_bands - remaining_large_balls * large_ball_rubber_bands) / small_ball_rubber_bands

theorem michael_small_balls :
  small_balls_made = 22 :=
sorry

end michael_small_balls_l566_56645


namespace walter_fall_distance_l566_56675

/-- The distance Walter fell before passing David -/
def distance_fallen (d : ℝ) : ℝ := 2 * d

theorem walter_fall_distance (d : ℝ) (h_positive : d > 0) :
  distance_fallen d = 2 * d :=
by sorry

end walter_fall_distance_l566_56675


namespace y_share_is_27_l566_56690

/-- Given a sum divided among x, y, and z, where y gets 45 paisa and z gets 50 paisa for each rupee x gets, 
    and the total amount is Rs. 117, prove that y's share is Rs. 27. -/
theorem y_share_is_27 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 117) 
  (h2 : y_share = 0.45 * x_share) 
  (h3 : z_share = 0.50 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  y_share = 27 := by
sorry


end y_share_is_27_l566_56690


namespace faye_bought_30_songs_l566_56612

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end faye_bought_30_songs_l566_56612


namespace right_triangle_inradius_l566_56634

/-- The inradius of a right triangle with sides 9, 40, and 41 is 4 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end right_triangle_inradius_l566_56634


namespace largest_prime_satisfying_inequality_l566_56655

theorem largest_prime_satisfying_inequality :
  ∃ (m : ℕ), m.Prime ∧ m^2 - 11*m + 28 < 0 ∧
  ∀ (n : ℕ), n.Prime → n^2 - 11*n + 28 < 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end largest_prime_satisfying_inequality_l566_56655


namespace meat_price_proof_l566_56621

/-- The price of meat per ounce in cents -/
def meat_price : ℝ := 6

theorem meat_price_proof :
  (∃ (paid_16 paid_8 : ℝ),
    16 * meat_price = paid_16 - 30 ∧
    8 * meat_price = paid_8 + 18) :=
by sorry

end meat_price_proof_l566_56621


namespace weaker_correlation_as_r_approaches_zero_l566_56658

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 < r ∧ r < 1 }

/-- A measure of correlation strength -/
def correlationStrength (r : CorrelationCoefficient) : ℝ := |r.val|

/-- Theorem: As the absolute value of the correlation coefficient approaches 0, 
    the correlation between two variables becomes weaker -/
theorem weaker_correlation_as_r_approaches_zero 
  (r : CorrelationCoefficient) : 
  ∀ ε > 0, ∃ δ > 0, ∀ r' : CorrelationCoefficient, 
    correlationStrength r' < δ → correlationStrength r' < ε :=
sorry

end weaker_correlation_as_r_approaches_zero_l566_56658


namespace total_distance_rowed_total_distance_is_15_19_l566_56620

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (total_time * upstream_speed * downstream_speed) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance

/-- Proves that the total distance traveled is approximately 15.19 km -/
theorem total_distance_is_15_19 :
  ∃ ε > 0, |total_distance_rowed 8 1.8 2 - 15.19| < ε :=
by
  sorry

end total_distance_rowed_total_distance_is_15_19_l566_56620


namespace line_through_three_points_l566_56626

/-- Given a line containing points (0, 5), (7, k), and (25, 2), prove that k = 104/25 -/
theorem line_through_three_points (k : ℝ) : 
  (∃ m b : ℝ, (0 = m * 0 + b ∧ 5 = m * 0 + b) ∧ 
              (k = m * 7 + b) ∧ 
              (2 = m * 25 + b)) → 
  k = 104 / 25 := by
sorry

end line_through_three_points_l566_56626


namespace simplify_fraction_l566_56648

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + a / (1 - a)) = a := by
  sorry

end simplify_fraction_l566_56648


namespace inverse_proposition_true_l566_56660

theorem inverse_proposition_true : 
  (∃ x y : ℝ, x ≤ y ∧ x ≤ |y|) ∧ 
  ¬(∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) ∧ 
  ¬(∃ x : ℝ, x = 1 ∧ x^2 + x - 2 ≠ 0) ∧ 
  ¬(∀ x : ℝ, x ≤ 1 → x^2 ≤ x) := by
  sorry

end inverse_proposition_true_l566_56660


namespace smallest_divisor_k_l566_56682

def f (z : ℂ) : ℂ := z^10 + z^9 + z^8 + z^6 + z^5 + z^4 + z + 1

theorem smallest_divisor_k : 
  (∀ z : ℂ, f z = 0 → z^84 = 1) ∧ 
  (∀ k : ℕ, k < 84 → ∃ z : ℂ, f z = 0 ∧ z^k ≠ 1) :=
sorry

end smallest_divisor_k_l566_56682


namespace integer_root_b_values_l566_56623

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 4*x^2 + b*x + 12 = 0

def valid_b_values : Set ℤ :=
  {-193, -97, -62, -35, -25, -18, -17, -14, -3, -1, 2, 9}

theorem integer_root_b_values :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
sorry

end integer_root_b_values_l566_56623


namespace system_solution_l566_56665

theorem system_solution (x y : ℝ) (h1 : 3 * x + 2 * y = 2) (h2 : 2 * x + 3 * y = 8) : x + y = 2 := by
  sorry

end system_solution_l566_56665


namespace trigonometric_identity_proof_l566_56686

theorem trigonometric_identity_proof :
  Real.cos (13 * π / 180) * Real.sin (58 * π / 180) - 
  Real.sin (13 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end trigonometric_identity_proof_l566_56686


namespace jerry_lawsuit_compensation_l566_56669

def annual_salary : ℕ := 50000
def years_lost : ℕ := 30
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def award_percentage : ℚ := 80 / 100

theorem jerry_lawsuit_compensation :
  let total_salary := annual_salary * years_lost
  let direct_damages := total_salary + medical_bills
  let punitive_damages := direct_damages * punitive_multiplier
  let total_asked := direct_damages + punitive_damages
  let awarded_amount := (total_asked : ℚ) * award_percentage
  awarded_amount = 5440000 := by sorry

end jerry_lawsuit_compensation_l566_56669


namespace handshake_arrangements_mod_1000_l566_56678

/-- Represents the number of ways N people can shake hands with exactly two others each -/
def handshake_arrangements (N : ℕ) : ℕ :=
  sorry

/-- The number of ways 9 people can shake hands with exactly two others each -/
def N : ℕ := handshake_arrangements 9

/-- Theorem stating that the number of handshake arrangements for 9 people is congruent to 152 modulo 1000 -/
theorem handshake_arrangements_mod_1000 : N ≡ 152 [ZMOD 1000] := by
  sorry

end handshake_arrangements_mod_1000_l566_56678


namespace rectangle_area_irrational_l566_56642

-- Define a rectangle with rational length and irrational width
structure Rectangle where
  length : ℚ
  width : ℝ
  width_irrational : Irrational width

-- Define the area of the rectangle
def area (rect : Rectangle) : ℝ := (rect.length : ℝ) * rect.width

-- Theorem statement
theorem rectangle_area_irrational (rect : Rectangle) : Irrational (area rect) := by
  sorry

end rectangle_area_irrational_l566_56642


namespace function_inequality_l566_56667

open Real

theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (hpos_f : ∀ x, f x > 0) (hpos_g : ∀ x, g x > 0)
  (h_inequality : ∀ x, (deriv^[2] f) x * g x - f x * (deriv^[2] g) x < 0)
  (a b x : ℝ) (hx : b < x ∧ x < a) :
  f x * g a > f a * g x :=
by sorry

end function_inequality_l566_56667


namespace full_price_revenue_l566_56602

/-- Represents the price of a full-price ticket -/
def full_price : ℝ := sorry

/-- Represents the number of full-price tickets sold -/
def full_price_tickets : ℕ := sorry

/-- Represents the number of discounted tickets sold -/
def discounted_tickets : ℕ := sorry

/-- The total number of tickets sold is 160 -/
axiom total_tickets : full_price_tickets + discounted_tickets = 160

/-- The total revenue is $2400 -/
axiom total_revenue : full_price * full_price_tickets + (full_price / 3) * discounted_tickets = 2400

/-- Theorem stating that the revenue from full-price tickets is $400 -/
theorem full_price_revenue : full_price * full_price_tickets = 400 := by sorry

end full_price_revenue_l566_56602


namespace forest_tree_density_l566_56685

/-- Calculates the tree density in a rectangular forest given the logging parameters --/
theorem forest_tree_density
  (forest_length : ℕ)
  (forest_width : ℕ)
  (loggers : ℕ)
  (months : ℕ)
  (days_per_month : ℕ)
  (trees_per_logger_per_day : ℕ)
  (h1 : forest_length = 4)
  (h2 : forest_width = 6)
  (h3 : loggers = 8)
  (h4 : months = 10)
  (h5 : days_per_month = 30)
  (h6 : trees_per_logger_per_day = 6) :
  (loggers * months * days_per_month * trees_per_logger_per_day) / (forest_length * forest_width) = 600 := by
  sorry

#check forest_tree_density

end forest_tree_density_l566_56685


namespace prob_odd_divisor_15_factorial_l566_56654

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0) (Finset.range (n + 1))).card

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := (Finset.filter (λ d => n % d = 0 ∧ d % 2 ≠ 0) (Finset.range (n + 1))).card

/-- The probability of choosing an odd divisor of n -/
def probOddDivisor (n : ℕ) : ℚ := numOddDivisors n / numDivisors n

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 6 := by
  sorry

end prob_odd_divisor_15_factorial_l566_56654


namespace poems_sally_can_recite_l566_56616

theorem poems_sally_can_recite (initial_poems : ℕ) (forgotten_poems : ℕ) : 
  initial_poems = 8 → forgotten_poems = 5 → initial_poems - forgotten_poems = 3 := by
sorry

end poems_sally_can_recite_l566_56616


namespace second_bag_weight_is_10_l566_56604

/-- The weight of the second bag of dog food Elise bought -/
def second_bag_weight (initial_weight first_bag_weight final_weight : ℕ) : ℕ :=
  final_weight - (initial_weight + first_bag_weight)

theorem second_bag_weight_is_10 :
  second_bag_weight 15 15 40 = 10 := by
  sorry

end second_bag_weight_is_10_l566_56604


namespace correct_calculation_l566_56691

theorem correct_calculation (x : ℝ) (h : x ≠ 0) : (x^2 + x) / x = x + 1 := by
  sorry

end correct_calculation_l566_56691


namespace sum_of_five_consecutive_integers_l566_56684

theorem sum_of_five_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 5 * n + 10 := by
  sorry

end sum_of_five_consecutive_integers_l566_56684


namespace coin_packing_l566_56635

theorem coin_packing (n : ℕ) (r R : ℝ) (hn : n > 0) (hr : r > 0) (hR : R > r) :
  (1 / 2 : ℝ) * (R / r - 1) ≤ Real.sqrt n ∧ Real.sqrt n ≤ R / r :=
by sorry

end coin_packing_l566_56635


namespace dr_jones_remaining_money_l566_56611

theorem dr_jones_remaining_money :
  let monthly_earnings : ℕ := 6000
  let house_rental : ℕ := 640
  let food_expense : ℕ := 380
  let electric_water_bill : ℕ := monthly_earnings / 4
  let insurance_cost : ℕ := monthly_earnings / 5
  let total_expenses : ℕ := house_rental + food_expense + electric_water_bill + insurance_cost
  let remaining_money : ℕ := monthly_earnings - total_expenses
  remaining_money = 2280 := by
  sorry

end dr_jones_remaining_money_l566_56611


namespace function_value_at_one_l566_56666

/-- Given a function f where f(x-3) = 2x^2 - 3x + 1, prove that f(1) = 21 -/
theorem function_value_at_one (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 3) = 2 * x^2 - 3 * x + 1) : 
  f 1 = 21 := by
  sorry

end function_value_at_one_l566_56666


namespace justine_coloring_ratio_l566_56631

/-- Given a total number of sheets, number of binders, and sheets used by Justine,
    prove that the ratio of sheets Justine colored to total sheets in her binder is 1:2 -/
theorem justine_coloring_ratio 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_used : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_used = 245)
  : (sheets_used : ℚ) / (total_sheets / num_binders) = 1 / 2 := by
  sorry

end justine_coloring_ratio_l566_56631


namespace cylinder_minus_cones_volume_l566_56607

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cone : ℝ) 
  (h_cyl : ℝ) 
  (h_r : r = 10) 
  (h_cone_height : h_cone = 15) 
  (h_cyl_height : h_cyl = 30) : 
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end cylinder_minus_cones_volume_l566_56607


namespace cube_with_hole_volume_is_384_l566_56656

/-- The volume of a cube with a square hole cut through its center. -/
def cube_with_hole_volume (cube_side : ℝ) (hole_side : ℝ) : ℝ :=
  cube_side ^ 3 - hole_side ^ 2 * cube_side

/-- Theorem stating that a cube with side length 8 cm and a square hole
    with side length 4 cm cut through its center has a volume of 384 cm³. -/
theorem cube_with_hole_volume_is_384 :
  cube_with_hole_volume 8 4 = 384 := by
  sorry

#eval cube_with_hole_volume 8 4

end cube_with_hole_volume_is_384_l566_56656


namespace fraction_problem_l566_56632

theorem fraction_problem : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5100 = 765.0000000000001 := by
  sorry

end fraction_problem_l566_56632


namespace unique_monotonic_involutive_function_l566_56651

-- Define the properties of the function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) = x

-- Theorem statement
theorem unique_monotonic_involutive_function :
  ∀ f : ℝ → ℝ, Monotonic f → Involutive f → ∀ x : ℝ, f x = x :=
by
  sorry


end unique_monotonic_involutive_function_l566_56651


namespace eight_person_round_robin_matches_l566_56677

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: An 8-person round-robin tennis tournament has 28 matches -/
theorem eight_person_round_robin_matches :
  roundRobinMatches 8 = 28 := by
  sorry

#eval roundRobinMatches 8  -- This should output 28

end eight_person_round_robin_matches_l566_56677


namespace smallest_divisible_number_l566_56679

theorem smallest_divisible_number (n : ℕ) : 
  n = 719 + 288721 → 
  (∀ m : ℕ, 719 < m ∧ m < n → ¬(618 ∣ m ∧ 3648 ∣ m ∧ 60 ∣ m)) ∧ 
  (618 ∣ n ∧ 3648 ∣ n ∧ 60 ∣ n) :=
by sorry

end smallest_divisible_number_l566_56679


namespace line_points_k_value_l566_56644

/-- A line contains the points (3, 10), (1, k), and (-7, 2). Prove that k = 8.4. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (3 * m + b = 10) ∧ 
    (1 * m + b = k) ∧ 
    (-7 * m + b = 2)) → 
  k = 8.4 := by
sorry

end line_points_k_value_l566_56644


namespace right_triangle_30_hypotenuse_l566_56622

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of side XZ -/
  xz : ℝ
  /-- XZ is positive -/
  xz_pos : 0 < xz

/-- The length of the hypotenuse in a right triangle with a 30-degree angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.xz

/-- Theorem: In a right triangle XYZ with right angle at X, if angle YZX = 30° and XZ = 15, then XY = 30 -/
theorem right_triangle_30_hypotenuse :
  ∀ t : RightTriangle30, t.xz = 15 → hypotenuse t = 30 := by
  sorry

end right_triangle_30_hypotenuse_l566_56622


namespace star_three_four_l566_56627

-- Define the * operation
def star (a b : ℝ) : ℝ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_three_four : star 3 4 = 8 := by
  sorry

end star_three_four_l566_56627


namespace algebraic_expression_value_l566_56613

theorem algebraic_expression_value (m n : ℝ) (h : m^2 + 3*n - 1 = 2) :
  2*m^2 + 6*n + 1 = 7 := by
  sorry

end algebraic_expression_value_l566_56613


namespace dan_placed_13_scissors_l566_56633

/-- The number of scissors Dan placed in the drawer -/
def scissors_placed (initial_count final_count : ℕ) : ℕ :=
  final_count - initial_count

/-- Proof that Dan placed 13 scissors in the drawer -/
theorem dan_placed_13_scissors (initial_count final_count : ℕ) 
  (h1 : initial_count = 39)
  (h2 : final_count = 52) : 
  scissors_placed initial_count final_count = 13 := by
  sorry

#eval scissors_placed 39 52  -- Should output 13

end dan_placed_13_scissors_l566_56633


namespace least_positive_integer_with_given_remainders_l566_56625

theorem least_positive_integer_with_given_remainders : ∃! N : ℕ,
  (N % 11 = 10) ∧
  (N % 12 = 11) ∧
  (N % 13 = 12) ∧
  (N % 14 = 13) ∧
  (∀ M : ℕ, M < N →
    ¬((M % 11 = 10) ∧
      (M % 12 = 11) ∧
      (M % 13 = 12) ∧
      (M % 14 = 13))) ∧
  N = 12011 :=
by sorry

end least_positive_integer_with_given_remainders_l566_56625


namespace larger_solution_quadratic_l566_56600

theorem larger_solution_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 → x ≤ 11 ∧ (∃ y, y^2 - 9*y - 22 = 0 ∧ y ≠ x) :=
by
  sorry

end larger_solution_quadratic_l566_56600


namespace train_length_l566_56671

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 65 * 1000 / 3600) 
    (h2 : t = 15.506451791548985) (h3 : bridge_length = 150) : 
    v * t - bridge_length = 130 := by
  sorry

end train_length_l566_56671


namespace modular_inverse_of_two_mod_127_l566_56673

theorem modular_inverse_of_two_mod_127 : ∃ x : ℕ, x < 127 ∧ (2 * x) % 127 = 1 :=
  by
    use 64
    sorry

end modular_inverse_of_two_mod_127_l566_56673


namespace linear_function_uniqueness_l566_56676

/-- A linear function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f x < f y -/
def IsIncreasingLinear (f : ℝ → ℝ) : Prop :=
  (∃ a b : ℝ, ∀ x, f x = a * x + b) ∧ (∀ x y, x < y → f x < f y)

/-- The main theorem -/
theorem linear_function_uniqueness (f : ℝ → ℝ) 
  (h_increasing : IsIncreasingLinear f)
  (h_composition : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 := by
  sorry

end linear_function_uniqueness_l566_56676


namespace min_additional_squares_for_axisymmetry_l566_56697

/-- Represents a rectangle with shaded squares -/
structure ShadedRectangle where
  width : ℕ
  height : ℕ
  shadedSquares : Finset (ℕ × ℕ)

/-- Checks if a ShadedRectangle is axisymmetric with two lines of symmetry -/
def isAxisymmetric (rect : ShadedRectangle) : Prop :=
  ∀ (x y : ℕ), x < rect.width ∧ y < rect.height →
    ((x, y) ∈ rect.shadedSquares ↔ (rect.width - 1 - x, y) ∈ rect.shadedSquares) ∧
    ((x, y) ∈ rect.shadedSquares ↔ (x, rect.height - 1 - y) ∈ rect.shadedSquares)

/-- The theorem to be proved -/
theorem min_additional_squares_for_axisymmetry 
  (rect : ShadedRectangle) 
  (h : rect.shadedSquares.card = 3) : 
  ∃ (additionalSquares : Finset (ℕ × ℕ)),
    additionalSquares.card = 6 ∧
    isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ additionalSquares⟩ ∧
    ∀ (smallerSet : Finset (ℕ × ℕ)), 
      smallerSet.card < 6 → 
      ¬isAxisymmetric ⟨rect.width, rect.height, rect.shadedSquares ∪ smallerSet⟩ :=
sorry

end min_additional_squares_for_axisymmetry_l566_56697


namespace remainder_113_pow_113_plus_113_mod_137_l566_56696

theorem remainder_113_pow_113_plus_113_mod_137 
  (h1 : Prime 113) 
  (h2 : Prime 137) 
  (h3 : 113 < 137) : 
  (113^113 + 113) % 137 = 89 := by
sorry

end remainder_113_pow_113_plus_113_mod_137_l566_56696


namespace limit_of_f_at_one_l566_56680

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - x - 2) / (4 * x^2 - 5 * x + 1)

theorem limit_of_f_at_one :
  ∃ (L : ℝ), L = 5/3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x - L| < ε :=
sorry

end limit_of_f_at_one_l566_56680


namespace probability_penny_dime_heads_l566_56624

-- Define the coin flip experiment
def coin_flip_experiment : ℕ := 5

-- Define the number of coins we're interested in (penny and dime)
def target_coins : ℕ := 2

-- Define the probability of a single coin coming up heads
def prob_heads : ℚ := 1/2

-- Theorem statement
theorem probability_penny_dime_heads :
  (prob_heads ^ target_coins) * (2 ^ (coin_flip_experiment - target_coins)) / (2 ^ coin_flip_experiment) = 1/4 := by
sorry

end probability_penny_dime_heads_l566_56624


namespace not_necessary_not_sufficient_l566_56662

-- Define the conditions
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

-- Theorem statement
theorem not_necessary_not_sufficient :
  (∃ θ a, condition_A θ a ∧ ¬condition_B θ a) ∧
  (∃ θ a, condition_B θ a ∧ ¬condition_A θ a) :=
sorry

end not_necessary_not_sufficient_l566_56662


namespace min_pool_cost_is_5400_l566_56638

/-- Represents the specifications of a rectangular pool -/
structure PoolSpecs where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the minimum cost of constructing a rectangular pool given its specifications -/
def minPoolCost (specs : PoolSpecs) : ℝ :=
  sorry

/-- Theorem stating that the minimum cost of constructing the specified pool is 5400 yuan -/
theorem min_pool_cost_is_5400 :
  let specs : PoolSpecs := {
    volume := 18,
    depth := 2,
    bottomCost := 200,
    wallCost := 150
  }
  minPoolCost specs = 5400 :=
by sorry

end min_pool_cost_is_5400_l566_56638


namespace problem_solution_l566_56695

theorem problem_solution (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  (x^2 + y^2 = 14) ∧ (x / y - y / x = 8 * Real.sqrt 3) := by
  sorry

end problem_solution_l566_56695


namespace range_of_m_l566_56637

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) / (10 - x) ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, q x m → p x) →  -- p is a necessary condition for q
  (m < 0) →             -- Given condition
  m ≥ -3 ∧ m < 0        -- Conclusion: range of m is [-3, 0)
  := by sorry

end range_of_m_l566_56637
