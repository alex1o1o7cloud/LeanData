import Mathlib

namespace NUMINAMATH_CALUDE_max_amount_is_7550_l1831_183189

-- Define the total value of chips bought
def total_value : ℕ := 10000

-- Define the chip denominations
def chip_50_value : ℕ := 50
def chip_200_value : ℕ := 200

-- Define the total number of chips lost
def total_chips_lost : ℕ := 30

-- Define the relationship between lost chips
axiom lost_chips_relation : ∃ (x y : ℕ), x = 3 * y ∧ x + y = total_chips_lost

-- Define the function to calculate the maximum amount received back
def max_amount_received : ℕ := 
  total_value - (7 * chip_200_value + 21 * chip_50_value)

-- Theorem to prove
theorem max_amount_is_7550 : max_amount_received = 7550 := by
  sorry

end NUMINAMATH_CALUDE_max_amount_is_7550_l1831_183189


namespace NUMINAMATH_CALUDE_surface_area_of_solid_with_square_views_l1831_183110

/-- A solid with three square views -/
structure Solid where
  /-- The side length of the square views -/
  side_length : ℝ
  /-- The three views are squares -/
  square_views : Prop

/-- The surface area of a solid -/
def surface_area (s : Solid) : ℝ := sorry

/-- Theorem: The surface area of a solid with three square views of side length 2 is 24 -/
theorem surface_area_of_solid_with_square_views (s : Solid) 
  (h1 : s.side_length = 2) 
  (h2 : s.square_views) : 
  surface_area s = 24 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_solid_with_square_views_l1831_183110


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1831_183146

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  4 * (n % 10 * 10 + n / 10) = 2 * n

theorem smallest_valid_number : 
  (∃ (n : ℕ), is_valid n) ∧ 
  (∀ (m : ℕ), is_valid m → m ≥ 52) ∧
  is_valid 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1831_183146


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1831_183192

def U : Finset ℕ := {1,2,3,4,5,6,7}
def A : Finset ℕ := {1,3,5}
def B : Finset ℕ := {2,3,6}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1,3,4,5,7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1831_183192


namespace NUMINAMATH_CALUDE_inequality_property_l1831_183159

theorem inequality_property (a b : ℝ) (h : a < 0 ∧ 0 < b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l1831_183159


namespace NUMINAMATH_CALUDE_complex_number_range_l1831_183113

theorem complex_number_range (a : ℝ) (z : ℂ) : 
  z = a + Complex.I ∧ 
  (z.re < 0 ∧ z.im > 0) ∧ 
  Complex.abs (z * (1 + Complex.I)) > 2 → 
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_range_l1831_183113


namespace NUMINAMATH_CALUDE_division_count_correct_l1831_183176

def num_couples : ℕ := 5
def first_group_size : ℕ := 6
def min_couples_in_first_group : ℕ := 2

/-- The number of ways to divide 5 couples into two groups, 
    where the first group contains 6 people including at least two couples. -/
def num_divisions : ℕ := 130

theorem division_count_correct : 
  ∀ (n : ℕ) (k : ℕ) (m : ℕ),
  n = num_couples → 
  k = first_group_size → 
  m = min_couples_in_first_group →
  num_divisions = (Nat.choose n 2 * (Nat.choose ((n - 2) * 2) 2 - Nat.choose (n - 2) 1)) + 
                   Nat.choose n 3 :=
by sorry

end NUMINAMATH_CALUDE_division_count_correct_l1831_183176


namespace NUMINAMATH_CALUDE_total_cantelopes_l1831_183150

theorem total_cantelopes (fred_cantelopes tim_cantelopes : ℕ) 
  (h1 : fred_cantelopes = 38) 
  (h2 : tim_cantelopes = 44) : 
  fred_cantelopes + tim_cantelopes = 82 := by
sorry

end NUMINAMATH_CALUDE_total_cantelopes_l1831_183150


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l1831_183119

def f (x : ℝ) := x^3 - 12*x

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l1831_183119


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1831_183168

/-- An isosceles triangle with two sides of lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) →
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1831_183168


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1831_183184

/-- An ellipse with semi-major axis 2√2 and semi-minor axis 2√2 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 8 = 1

/-- A square inscribed in the ellipse with sides parallel to the axes -/
def InscribedSquare (s : ℝ) : Prop :=
  ∃ (x y : ℝ), Ellipse x y ∧ s = 2 * x ∧ s = 2 * y

/-- The area of the inscribed square is 32/3 -/
theorem inscribed_square_area :
  ∃ (s : ℝ), InscribedSquare s ∧ s^2 = 32/3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1831_183184


namespace NUMINAMATH_CALUDE_win_sectors_area_l1831_183163

theorem win_sectors_area (r : ℝ) (p : ℝ) (n : ℕ) : 
  r = 15 → p = 1/6 → n = 2 → n * p * (π * r^2) = 75 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sectors_area_l1831_183163


namespace NUMINAMATH_CALUDE_bob_homework_time_l1831_183126

theorem bob_homework_time (alice_time bob_time : ℕ) : 
  alice_time = 40 → bob_time = (3 * alice_time) / 8 → bob_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_homework_time_l1831_183126


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1831_183143

theorem not_p_sufficient_not_necessary_for_not_q :
  (∃ x : ℝ, ¬(abs x ≥ 1) ∧ ¬(x^2 + x - 6 ≥ 0)) ∧
  (∃ x : ℝ, ¬(x^2 + x - 6 ≥ 0) ∧ (abs x ≥ 1)) :=
by sorry


end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1831_183143


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l1831_183179

theorem smallest_k_for_inequality : 
  ∃ (k : ℕ), k = 4 ∧ 
  (∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l1831_183179


namespace NUMINAMATH_CALUDE_battery_current_l1831_183133

/-- Given a battery with voltage 48V, prove that when connected to a 12Ω resistance, 
    the resulting current is 4A. -/
theorem battery_current (V R I : ℝ) : 
  V = 48 → R = 12 → I = V / R → I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_l1831_183133


namespace NUMINAMATH_CALUDE_fraction_value_l1831_183106

theorem fraction_value : (1 * 2 * 3 * 4) / (1 + 2 + 3 + 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1831_183106


namespace NUMINAMATH_CALUDE_fraction_inequality_l1831_183169

theorem fraction_inequality (x : ℝ) : x / (x + 1) < 0 ↔ -1 < x ∧ x < 0 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1831_183169


namespace NUMINAMATH_CALUDE_parabola_equation_l1831_183181

/-- Represents a parabola with vertex at the origin and coordinate axes as axes of symmetry -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -2*a*x

/-- The parabola passes through the given point -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

theorem parabola_equation : 
  ∃ (p : Parabola), passes_through p (-2) (-4) ∧ p.a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l1831_183181


namespace NUMINAMATH_CALUDE_juvenile_female_percentage_l1831_183167

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  adult_females : ℕ
  juvenile_females : ℕ

/-- Conditions for the Lagoon Island alligator population -/
def lagoon_conditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.total / 2 ∧
  pop.males = 25 ∧
  pop.adult_females = 15 ∧
  pop.juvenile_females = pop.total / 2 - pop.adult_females

/-- Theorem: The percentage of juvenile female alligators is 40% -/
theorem juvenile_female_percentage (pop : AlligatorPopulation) 
  (h : lagoon_conditions pop) : 
  (pop.juvenile_females : ℚ) / (pop.total / 2 : ℚ) = 2/5 := by
  sorry

#check juvenile_female_percentage

end NUMINAMATH_CALUDE_juvenile_female_percentage_l1831_183167


namespace NUMINAMATH_CALUDE_train_length_calculation_l1831_183191

/-- Given a train passing a bridge, calculate its length -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 140 →
  time_to_pass = 52 →
  (train_speed * time_to_pass - bridge_length) = 510 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1831_183191


namespace NUMINAMATH_CALUDE_triangular_pyramid_no_circular_cross_section_l1831_183122

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere
  | TriangularPyramid

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.Cone => True
  | Solid.Cylinder => True
  | Solid.Sphere => True
  | Solid.TriangularPyramid => False

-- Theorem statement
theorem triangular_pyramid_no_circular_cross_section :
  ∀ s : Solid, ¬(has_circular_cross_section s) ↔ s = Solid.TriangularPyramid :=
by sorry


end NUMINAMATH_CALUDE_triangular_pyramid_no_circular_cross_section_l1831_183122


namespace NUMINAMATH_CALUDE_abs_fraction_inequality_l1831_183104

theorem abs_fraction_inequality (x : ℝ) :
  x ≠ 0 → (|(x - 2) / x| > (x - 2) / x ↔ 0 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_abs_fraction_inequality_l1831_183104


namespace NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_is_zero_l1831_183185

open Real

theorem min_max_abs_x_squared_minus_2xy_is_zero :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - 2*x*y| ≤ z) →
    (∀ y' : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 - 2*x*y'| ≥ z) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_is_zero_l1831_183185


namespace NUMINAMATH_CALUDE_f_range_l1831_183134

-- Define the function f
def f (x : ℝ) : ℝ := -(x - 5)^2 + 1

-- Define the domain
def domain : Set ℝ := {x | 2 < x ∧ x < 6}

-- Define the range
def range : Set ℝ := {y | -8 < y ∧ y ≤ 1}

-- Theorem statement
theorem f_range : 
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x ∈ domain, f x ∈ range :=
sorry

end NUMINAMATH_CALUDE_f_range_l1831_183134


namespace NUMINAMATH_CALUDE_three_leaf_clover_count_l1831_183197

theorem three_leaf_clover_count :
  ∀ (total_leaves : ℕ) (three_leaf_count : ℕ),
    total_leaves = 1000 →
    3 * three_leaf_count + 4 = total_leaves →
    three_leaf_count = 332 := by
  sorry

end NUMINAMATH_CALUDE_three_leaf_clover_count_l1831_183197


namespace NUMINAMATH_CALUDE_team_selection_probability_l1831_183194

/-- The probability of randomly selecting a team that includes three specific players -/
theorem team_selection_probability 
  (total_players : ℕ) 
  (team_size : ℕ) 
  (specific_players : ℕ) 
  (h1 : total_players = 12) 
  (h2 : team_size = 6) 
  (h3 : specific_players = 3) :
  (Nat.choose (total_players - specific_players) (team_size - specific_players)) / 
  (Nat.choose total_players team_size) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_probability_l1831_183194


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1831_183129

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The sum of coordinates of a point -/
def sumCoordinates (p : Point) : ℝ := p.x + p.y

/-- Theorem: Sum of coordinates of vertex C in the given parallelogram is 7 -/
theorem parallelogram_vertex_sum : 
  ∀ (ABCD : Parallelogram),
    ABCD.A = ⟨2, 3⟩ →
    ABCD.B = ⟨-1, 0⟩ →
    ABCD.D = ⟨5, -4⟩ →
    sumCoordinates ABCD.C = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l1831_183129


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l1831_183175

/-- A complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- The theorem statement -/
theorem complex_ratio_theorem (a b c : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a / b = b / c) (hbc : b / c = c / a) :
  (a + b - c) / (a - b + c) = 1 ∨
  (a + b - c) / (a - b + c) = ω ∨
  (a + b - c) / (a - b + c) = ω^2 :=
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l1831_183175


namespace NUMINAMATH_CALUDE_truck_distance_before_meeting_l1831_183183

/-- The distance between two trucks one minute before they meet, given their initial separation and speeds -/
theorem truck_distance_before_meeting
  (initial_distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (h1 : initial_distance = 4)
  (h2 : speed_A = 45)
  (h3 : speed_B = 60)
  : ∃ (d : ℝ), d = 250 / 1000 ∧ d = initial_distance + speed_A * (1 / 60) - speed_B * (1 / 60) :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_before_meeting_l1831_183183


namespace NUMINAMATH_CALUDE_system_solution_l1831_183157

theorem system_solution (m n : ℝ) 
  (eq1 : m * 3 + (-7) = 5)
  (eq2 : 2 * (7/2) - n * (-2) = 13)
  : ∃ (x y : ℝ), m * x + y = 5 ∧ 2 * x - n * y = 13 ∧ x = 2 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1831_183157


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l1831_183131

theorem shaded_triangle_probability 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles > 4) 
  (h2 : total_triangles = 10) 
  (h3 : shaded_triangles = 4) : 
  (shaded_triangles : ℚ) / total_triangles = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_shaded_triangle_probability_l1831_183131


namespace NUMINAMATH_CALUDE_complex_product_real_l1831_183127

theorem complex_product_real (x : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := x - I
  (z₁ * z₂).im = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1831_183127


namespace NUMINAMATH_CALUDE_problem_solution_l1831_183109

theorem problem_solution : (-1)^2023 + |2 * Real.sqrt 2 - 3| + (8 : ℝ)^(1/3) = 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1831_183109


namespace NUMINAMATH_CALUDE_test_questions_count_l1831_183147

theorem test_questions_count (sections : Nat) (correct_answers : Nat) 
  (h1 : sections = 4)
  (h2 : correct_answers = 20)
  (h3 : ∀ x : Nat, x > 0 → (60 : Real) / 100 < (correct_answers : Real) / x → (correct_answers : Real) / x < (70 : Real) / 100 → x % sections = 0 → x = 32) :
  ∃ total_questions : Nat, 
    total_questions > 0 ∧ 
    (60 : Real) / 100 < (correct_answers : Real) / total_questions ∧ 
    (correct_answers : Real) / total_questions < (70 : Real) / 100 ∧ 
    total_questions % sections = 0 ∧
    total_questions = 32 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_count_l1831_183147


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l1831_183108

/-- The number of cupcakes when shared equally among children -/
def cupcakes_per_child : ℕ := 12

/-- The number of children sharing the cupcakes -/
def number_of_children : ℕ := 8

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := cupcakes_per_child * number_of_children

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l1831_183108


namespace NUMINAMATH_CALUDE_factor_2x_squared_minus_8_l1831_183120

theorem factor_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_2x_squared_minus_8_l1831_183120


namespace NUMINAMATH_CALUDE_num_distinct_representations_eq_six_l1831_183125

/-- Represents a digit configuration using matchsticks -/
def DigitConfig := Nat

/-- The maximum number of matchsticks in the original configuration -/
def max_sticks : Nat := 7

/-- The set of all possible digit configurations -/
def all_configs : Finset DigitConfig := sorry

/-- The number of distinct digit representations -/
def num_distinct_representations : Nat := Finset.card all_configs

/-- Theorem stating that the number of distinct representations is 6 -/
theorem num_distinct_representations_eq_six :
  num_distinct_representations = 6 := by sorry

end NUMINAMATH_CALUDE_num_distinct_representations_eq_six_l1831_183125


namespace NUMINAMATH_CALUDE_point_on_line_through_two_points_l1831_183187

/-- A point lies on a line if it satisfies the line equation --/
def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem statement --/
theorem point_on_line_through_two_points :
  point_on_line 1 2 5 10 3 6 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_through_two_points_l1831_183187


namespace NUMINAMATH_CALUDE_sum_range_for_cube_sum_two_l1831_183165

theorem sum_range_for_cube_sum_two (x y : ℝ) (h : x^3 + y^3 = 2) :
  0 < x + y ∧ x + y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_for_cube_sum_two_l1831_183165


namespace NUMINAMATH_CALUDE_abc_book_cost_l1831_183136

/-- The cost of the best-selling book "TOP" -/
def top_cost : ℝ := 8

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℝ := 12

/-- The cost of the "ABC" book -/
def abc_cost : ℝ := 23

theorem abc_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_abc_book_cost_l1831_183136


namespace NUMINAMATH_CALUDE_max_excellent_boys_100_l1831_183102

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Defines the "not worse than" relation between two people -/
def notWorseThan (a b : Person) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Defines an "excellent boy" as someone who is not worse than all others -/
def excellentBoy (p : Person) (group : Finset Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notWorseThan p q

/-- The main theorem: The maximum number of excellent boys in a group of 100 is 100 -/
theorem max_excellent_boys_100 :
  ∃ (group : Finset Person), group.card = 100 ∧
  ∃ (excellent : Finset Person), excellent ⊆ group ∧ excellent.card = 100 ∧
  ∀ p ∈ excellent, excellentBoy p group :=
sorry

end NUMINAMATH_CALUDE_max_excellent_boys_100_l1831_183102


namespace NUMINAMATH_CALUDE_f_of_one_equals_one_l1831_183121

theorem f_of_one_equals_one (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = Real.cos x) : f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_one_l1831_183121


namespace NUMINAMATH_CALUDE_survey_respondents_l1831_183118

theorem survey_respondents : 
  ∀ (x y : ℕ), 
    x = 60 → -- Number of people who prefer brand X
    x = 3 * y → -- Ratio of preference for X to Y is 3:1
    x + y = 80 -- Total number of respondents
    := by sorry

end NUMINAMATH_CALUDE_survey_respondents_l1831_183118


namespace NUMINAMATH_CALUDE_bisection_next_point_l1831_183196

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Define the initial interval
def a : ℝ := 0
def b : ℝ := 2

-- Define the first midpoint
def m₁ : ℝ := 1

-- Theorem statement
theorem bisection_next_point :
  f a < 0 ∧ f b > 0 ∧ f m₁ < 0 →
  (a + b) / 2 = 1.5 := by sorry

end NUMINAMATH_CALUDE_bisection_next_point_l1831_183196


namespace NUMINAMATH_CALUDE_newspaper_delivery_patterns_l1831_183135

/-- Represents the number of valid newspaper delivery patterns for n houses -/
def D : ℕ → ℕ
| 0 => 1  -- Base case, one way to deliver to zero houses
| 1 => 2  -- Two ways to deliver to one house (deliver or not)
| 2 => 4  -- Four ways to deliver to two houses
| n + 3 => D (n + 2) + D (n + 1) + D n  -- Recurrence relation

/-- The condition that the last house must receive a newspaper -/
def lastHouseDelivery (n : ℕ) : ℕ := D (n - 1)

/-- The number of houses on the lane -/
def numHouses : ℕ := 12

/-- The theorem stating the number of valid delivery patterns for 12 houses -/
theorem newspaper_delivery_patterns :
  lastHouseDelivery numHouses = 927 := by sorry

end NUMINAMATH_CALUDE_newspaper_delivery_patterns_l1831_183135


namespace NUMINAMATH_CALUDE_min_value_problem_l1831_183117

theorem min_value_problem (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hab : a * b = 1/4) :
  (∀ x y : ℝ, 0 < x ∧ x < 1 → 0 < y ∧ y < 1 → x * y = 1/4 →
    1 / (1 - x) + 2 / (1 - y) ≥ 4 + 4 * Real.sqrt 2 / 3) ∧
  (∃ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 ∧
    1 / (1 - x) + 2 / (1 - y) = 4 + 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1831_183117


namespace NUMINAMATH_CALUDE_cost_of_seven_sandwiches_six_sodas_l1831_183101

/-- Calculates the total cost of purchasing sandwiches and sodas at Sally's Snack Shop -/
def snack_shop_cost (sandwich_count : ℕ) (soda_count : ℕ) : ℕ :=
  let sandwich_price := 4
  let soda_price := 3
  let bulk_discount := 10
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_price + soda_count * soda_price
  if total_items > 10 then total_cost - bulk_discount else total_cost

/-- Theorem stating that purchasing 7 sandwiches and 6 sodas costs $36 -/
theorem cost_of_seven_sandwiches_six_sodas :
  snack_shop_cost 7 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_seven_sandwiches_six_sodas_l1831_183101


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1831_183177

theorem sum_of_fractions : (3 : ℚ) / 5 + 5 / 11 + 1 / 3 = 229 / 165 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1831_183177


namespace NUMINAMATH_CALUDE_sum_of_distinct_roots_l1831_183174

theorem sum_of_distinct_roots (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_roots_l1831_183174


namespace NUMINAMATH_CALUDE_luke_weed_eating_earnings_luke_weed_eating_proof_l1831_183155

theorem luke_weed_eating_earnings : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mowing_earnings weeks_lasting weekly_spending total_earnings =>
    mowing_earnings = 9 →
    weeks_lasting = 9 →
    weekly_spending = 3 →
    total_earnings = weeks_lasting * weekly_spending →
    total_earnings - mowing_earnings = 18

-- The proof would go here, but we'll use sorry as instructed
theorem luke_weed_eating_proof : luke_weed_eating_earnings 9 9 3 27 := by
  sorry

end NUMINAMATH_CALUDE_luke_weed_eating_earnings_luke_weed_eating_proof_l1831_183155


namespace NUMINAMATH_CALUDE_equilateral_triangle_construction_l1831_183100

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def rotatePoint (p : ℝ × ℝ) (center : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ := sorry

theorem equilateral_triangle_construction 
  (A : ℝ × ℝ) (S₁ S₂ : Circle) : 
  ∃ (B C : ℝ × ℝ), 
    (∃ (t : ℝ), B = rotatePoint C A (-π/3)) ∧
    (∃ (t : ℝ), C = rotatePoint B A (π/3)) ∧
    (∃ (t : ℝ), B ∈ {p | (p.1 - S₁.center.1)^2 + (p.2 - S₁.center.2)^2 = S₁.radius^2}) ∧
    (∃ (t : ℝ), C ∈ {p | (p.1 - S₂.center.1)^2 + (p.2 - S₂.center.2)^2 = S₂.radius^2}) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_construction_l1831_183100


namespace NUMINAMATH_CALUDE_terminal_side_first_quadrant_l1831_183193

-- Define the angle in degrees
def angle : ℝ := -330

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (θ : ℝ) : Quadrant :=
  sorry

-- Theorem statement
theorem terminal_side_first_quadrant :
  angle_quadrant angle = Quadrant.first :=
sorry

end NUMINAMATH_CALUDE_terminal_side_first_quadrant_l1831_183193


namespace NUMINAMATH_CALUDE_shoe_selection_probability_l1831_183170

def total_pairs : ℕ := 16
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 2
def red_pairs : ℕ := 2

theorem shoe_selection_probability :
  let total_shoes := total_pairs * 2
  let prob_same_color_diff_foot : ℚ :=
    (black_pairs * black_pairs + brown_pairs * brown_pairs + 
     gray_pairs * gray_pairs + red_pairs * red_pairs) / 
    (total_shoes * (total_shoes - 1))
  prob_same_color_diff_foot = 11 / 62 := by sorry

end NUMINAMATH_CALUDE_shoe_selection_probability_l1831_183170


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1831_183195

/-- Given a right-angled triangle PQR with the right angle at R, PR = 8, PQ = 15, and QR = 6,
    prove that the perimeter of the triangle is 24. -/
theorem triangle_perimeter (P Q R : ℝ × ℝ) : 
  (R.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (R.1 - P.1) →  -- Right angle at R
  dist P R = 8 →
  dist P Q = 15 →
  dist Q R = 6 →
  dist P R + dist P Q + dist Q R = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1831_183195


namespace NUMINAMATH_CALUDE_problem_statements_l1831_183171

theorem problem_statements (a b : ℝ) :
  (ab < 0 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → a / b = -1) ∧
  (a + b < 0 ∧ ab > 0 → |2*a + 3*b| = -(2*a + 3*b)) ∧
  ¬(∀ a b : ℝ, |a - b| + a - b = 0 → b > a) ∧
  ¬(∀ a b : ℝ, |a| > |b| → (a + b) * (a - b) < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1831_183171


namespace NUMINAMATH_CALUDE_length_PS_l1831_183156

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), P = (0, 0) ∧ Q = (x, y) ∧ R = (z, 0)

-- Define a right angle at P
def RightAngleAtP (P Q R : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the lengths of PR and PQ
def LengthPR (P R : ℝ × ℝ) : ℝ := 3
def LengthPQ (P Q : ℝ × ℝ) : ℝ := 4

-- Define S as the point where the angle bisector of ∠QPR meets QR
def AngleBisectorS (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ S = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2) ∧
  (S.1 - P.1) * (Q.2 - P.2) = (S.2 - P.2) * (Q.1 - P.1) ∧
  (S.1 - P.1) * (R.2 - P.2) = (S.2 - P.2) * (R.1 - P.1)

-- Main theorem
theorem length_PS (P Q R S : ℝ × ℝ) :
  Triangle P Q R →
  RightAngleAtP P Q R →
  LengthPR P R = 3 →
  LengthPQ P Q = 4 →
  AngleBisectorS P Q R S →
  Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_length_PS_l1831_183156


namespace NUMINAMATH_CALUDE_gain_percentage_calculation_l1831_183107

theorem gain_percentage_calculation (selling_price gain : ℝ) : 
  selling_price = 225 → gain = 75 → (gain / (selling_price - gain)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gain_percentage_calculation_l1831_183107


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1831_183112

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 3)
  (h_a8 : a 8 = 8) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l1831_183112


namespace NUMINAMATH_CALUDE_circle_center_distance_to_line_l1831_183190

/-- A circle passing through (1, 2) and tangent to both coordinate axes has its center
    at distance 2√5/5 from the line 2x - y - 3 = 0 -/
theorem circle_center_distance_to_line :
  ∀ (a : ℝ), 
    (∃ (x y : ℝ), (x - a)^2 + (y - a)^2 = a^2 ∧ x = 1 ∧ y = 2) →  -- Circle passes through (1, 2)
    (∃ (x : ℝ), (x - a)^2 + a^2 = a^2) →                          -- Circle is tangent to x-axis
    (∃ (y : ℝ), a^2 + (y - a)^2 = a^2) →                          -- Circle is tangent to y-axis
    (|a - 3| / Real.sqrt 5 : ℝ) = 2 * Real.sqrt 5 / 5 :=
by sorry


end NUMINAMATH_CALUDE_circle_center_distance_to_line_l1831_183190


namespace NUMINAMATH_CALUDE_total_games_played_l1831_183124

theorem total_games_played (total_teams : Nat) (rivalry_groups : Nat) (teams_per_group : Nat) (additional_games_per_team : Nat) : 
  total_teams = 50 → 
  rivalry_groups = 10 → 
  teams_per_group = 5 → 
  additional_games_per_team = 2 → 
  (total_teams * (total_teams - 1) / 2) + (rivalry_groups * teams_per_group * additional_games_per_team / 2) = 1325 := by
  sorry

end NUMINAMATH_CALUDE_total_games_played_l1831_183124


namespace NUMINAMATH_CALUDE_min_value_theorem_l1831_183111

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 ∧
  ∀ x, x = (a^2 + b^2) / (a - b) → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1831_183111


namespace NUMINAMATH_CALUDE_diamond_value_l1831_183132

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem diamond_value (diamond : Digit) :
  (9 * diamond.val + 6 = 10 * diamond.val + 3) → diamond.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l1831_183132


namespace NUMINAMATH_CALUDE_tangent_line_parallel_proof_l1831_183178

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to the parabola at point (a, a^2) -/
def tangent_slope (a : ℝ) : ℝ := 2 * a

/-- The slope of the line 2x - y + 4 = 0 -/
def given_line_slope : ℝ := 2

/-- The equation of the tangent line at point (a, a^2) -/
def tangent_line_eq (a : ℝ) (x y : ℝ) : Prop :=
  y - a^2 = tangent_slope a * (x - a)

theorem tangent_line_parallel_proof (a : ℝ) :
  tangent_slope a = given_line_slope →
  a = 1 ∧
  ∀ x y : ℝ, tangent_line_eq a x y ↔ 2*x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_proof_l1831_183178


namespace NUMINAMATH_CALUDE_number_pattern_equality_l1831_183186

theorem number_pattern_equality (n : ℕ) (h : n > 1) :
  3 * (6 * (10^n - 1) / 9)^3 = 
    8 * ((10^n - 1) / 9) * 10^(2*n+1) + 
    6 * 10^(2*n) + 
    2 * ((10^n - 1) / 9) * 10^(n+1) + 
    4 * 10^n + 
    8 * ((10^n - 1) / 9) := by
  sorry

end NUMINAMATH_CALUDE_number_pattern_equality_l1831_183186


namespace NUMINAMATH_CALUDE_unique_common_point_modulo25_l1831_183137

/-- Given two congruences on modulo 25 graph paper, prove there's exactly one common point with x-coordinate 1 --/
theorem unique_common_point_modulo25 : ∃! p : ℕ × ℕ, 
  p.1 < 25 ∧ 
  p.2 < 25 ∧
  p.2 ≡ 10 * p.1 + 3 [ZMOD 25] ∧ 
  p.2 ≡ p.1^2 + 15 * p.1 + 20 [ZMOD 25] ∧
  p.1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_point_modulo25_l1831_183137


namespace NUMINAMATH_CALUDE_french_students_count_l1831_183115

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 79)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 25)
  : ∃ french : ℕ, french = 41 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_french_students_count_l1831_183115


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1831_183128

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 5 → m ≤ n → n = 93 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1831_183128


namespace NUMINAMATH_CALUDE_min_value_and_range_l1831_183152

variable (x y z : ℝ)

def t (x y z : ℝ) : ℝ := x^2 + y^2 + 2*z^2

theorem min_value_and_range :
  (x + y + 2*z = 1) →
  (∃ (min : ℝ), ∀ x y z, t x y z ≥ min ∧ ∃ x y z, t x y z = min) ∧
  (t x y z = 1/2 → 0 ≤ z ∧ z ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_range_l1831_183152


namespace NUMINAMATH_CALUDE_m_range_l1831_183153

def A : Set ℝ := {x | (x + 1) / (x - 3) < 0}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : 
  (∀ x, x ∈ B m → x ∈ A) ∧ 
  (∃ x, x ∈ A ∧ x ∉ B m) → 
  m > 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1831_183153


namespace NUMINAMATH_CALUDE_goldfish_feeding_cost_specific_goldfish_problem_l1831_183142

/-- The cost to feed goldfish that need special food -/
theorem goldfish_feeding_cost 
  (total_goldfish : ℕ) 
  (food_per_fish : ℚ) 
  (special_food_percentage : ℚ) 
  (special_food_cost : ℚ) : ℚ :=
  let special_fish_count := (total_goldfish : ℚ) * special_food_percentage
  let special_food_amount := special_fish_count * food_per_fish
  let daily_cost := special_food_amount * special_food_cost
  daily_cost

/-- Proof of the specific goldfish feeding cost problem -/
theorem specific_goldfish_problem : 
  goldfish_feeding_cost 50 (3/2) (1/5) 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_feeding_cost_specific_goldfish_problem_l1831_183142


namespace NUMINAMATH_CALUDE_expression_evaluation_l1831_183199

theorem expression_evaluation : 
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1831_183199


namespace NUMINAMATH_CALUDE_only_al_tells_truth_l1831_183141

structure Pirate where
  name : String
  gold : ℕ
  silver : ℕ
  bronze : ℕ

def pirates : List Pirate := [
  ⟨"Tom", 10, 8, 11⟩,
  ⟨"Al", 9, 11, 10⟩,
  ⟨"Pit", 10, 10, 9⟩,
  ⟨"Jim", 8, 10, 11⟩
]

def totalCoins : ℕ := 30

theorem only_al_tells_truth :
  ∃! p : Pirate, p ∈ pirates ∧ p.gold + p.silver + p.bronze = totalCoins ∧
    ∀ q : Pirate, q ∈ pirates → q ≠ p → 
      (q.gold ≠ p.gold ∧ q.silver ≠ p.silver ∧ q.bronze ≠ p.bronze) :=
by
  sorry

#check only_al_tells_truth

end NUMINAMATH_CALUDE_only_al_tells_truth_l1831_183141


namespace NUMINAMATH_CALUDE_roots_equation_l1831_183198

theorem roots_equation (c d r s : ℝ) : 
  (c^2 - 7*c + 12 = 0) →
  (d^2 - 7*d + 12 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 169/12 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_l1831_183198


namespace NUMINAMATH_CALUDE_remainder_sum_l1831_183180

theorem remainder_sum (x y : ℤ) (hx : x % 80 = 75) (hy : y % 120 = 115) :
  (x + y) % 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1831_183180


namespace NUMINAMATH_CALUDE_polynomial_transformation_l1831_183182

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l1831_183182


namespace NUMINAMATH_CALUDE_set_intersection_difference_l1831_183138

theorem set_intersection_difference (S T : Set ℕ) (a b : ℕ) :
  S = {1, 2, a} →
  T = {2, 3, 4, b} →
  S ∩ T = {1, 2, 3} →
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_difference_l1831_183138


namespace NUMINAMATH_CALUDE_proportion_solution_l1831_183103

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1831_183103


namespace NUMINAMATH_CALUDE_sum_reciprocals_equal_two_point_five_l1831_183173

theorem sum_reciprocals_equal_two_point_five
  (a b c d e : ℝ)
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1) (he : e ≠ -1)
  (ω : ℂ)
  (hω1 : ω^3 = 1)
  (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) + (1 / (e + ω)) = 5 / (2*ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) + (1 / (e + 1)) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equal_two_point_five_l1831_183173


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l1831_183188

theorem fahrenheit_celsius_conversion (F C : ℝ) : C = (5 / 9) * (F - 30) → C = 30 → F = 84 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l1831_183188


namespace NUMINAMATH_CALUDE_course_selection_theorem_l1831_183144

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 + 
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l1831_183144


namespace NUMINAMATH_CALUDE_roots_pure_imaginary_l1831_183161

theorem roots_pure_imaginary (k : ℝ) (hk : k > 0) :
  ∃ (b c : ℝ), ∀ (z : ℂ), 8 * z^2 - 5 * I * z - k = 0 → z = b * I ∨ z = c * I :=
by sorry

end NUMINAMATH_CALUDE_roots_pure_imaginary_l1831_183161


namespace NUMINAMATH_CALUDE_tan_equality_solution_l1831_183151

theorem tan_equality_solution (n : ℤ) :
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1340 * π / 180) →
  n = 80 ∨ n = -100 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l1831_183151


namespace NUMINAMATH_CALUDE_problem_solution_l1831_183123

/-- Given m ≥ 0 and f(x) = 2|x - 1| - |2x + m| with a maximum value of 3,
    prove that m = 1 and min(a² + b² + c²) = 1/6 where a - 2b + c = m -/
theorem problem_solution (m : ℝ) (h_m : m ≥ 0)
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * |x - 1| - |2*x + m|)
  (h_max : ∀ x, f x ≤ 3) (h_exists : ∃ x, f x = 3) :
  m = 1 ∧ (∃ a b c : ℝ, a - 2*b + c = m ∧
    a^2 + b^2 + c^2 = 1/6 ∧
    ∀ a' b' c' : ℝ, a' - 2*b' + c' = m → a'^2 + b'^2 + c'^2 ≥ 1/6) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1831_183123


namespace NUMINAMATH_CALUDE_min_side_length_l1831_183158

theorem min_side_length (AB AC DC BD BC : ℕ) : 
  AB = 7 → AC = 15 → DC = 10 → BD = 25 → BC > 0 →
  (AB + BC > AC) → (AC + BC > AB) →
  (BD + DC > BC) → (BD + BC > DC) →
  BC ≥ 15 ∧ ∃ (BC : ℕ), BC = 15 ∧ 
    AB + BC > AC ∧ AC + BC > AB ∧
    BD + DC > BC ∧ BD + BC > DC :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l1831_183158


namespace NUMINAMATH_CALUDE_vector_perpendicular_l1831_183166

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, -3)

theorem vector_perpendicular : 
  let diff := (a.1 - b.1, a.2 - b.2)
  a.1 * diff.1 + a.2 * diff.2 = 0 := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l1831_183166


namespace NUMINAMATH_CALUDE_second_attempt_score_l1831_183148

/-- Represents the score of a dart throw attempt -/
structure DartScore where
  score : ℕ
  darts : ℕ
  min_per_dart : ℕ
  max_per_dart : ℕ

/-- The relationship between three dart throw attempts -/
structure ThreeAttempts where
  first : DartScore
  second : DartScore
  third : DartScore
  second_twice_first : first.score * 2 = second.score
  third_1_5_second : second.score * 3 = third.score * 2

/-- The theorem stating the score of the second attempt -/
theorem second_attempt_score (attempts : ThreeAttempts) 
  (h1 : attempts.first.darts = 8)
  (h2 : attempts.second.darts = 8)
  (h3 : attempts.third.darts = 8)
  (h4 : attempts.first.min_per_dart = 3)
  (h5 : attempts.first.max_per_dart = 9)
  (h6 : attempts.second.min_per_dart = 3)
  (h7 : attempts.second.max_per_dart = 9)
  (h8 : attempts.third.min_per_dart = 3)
  (h9 : attempts.third.max_per_dart = 9)
  : attempts.second.score = 48 := by
  sorry

end NUMINAMATH_CALUDE_second_attempt_score_l1831_183148


namespace NUMINAMATH_CALUDE_club_members_neither_subject_l1831_183172

theorem club_members_neither_subject (total : ℕ) (cs : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 80)
  (h3 : bio = 50)
  (h4 : both = 15) :
  total - (cs + bio - both) = 35 := by
  sorry

end NUMINAMATH_CALUDE_club_members_neither_subject_l1831_183172


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1831_183162

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - x - 20) / ((x - 3) * (x + 1) * (x - 2)) = 
    A / (x - 3) + B / (x + 1) + C / (x - 2) →
  A * B * C = 0 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1831_183162


namespace NUMINAMATH_CALUDE_equation_solution_l1831_183149

def solution_set : Set (ℤ × ℤ) := {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)}

theorem equation_solution (x y : ℤ) : 
  (x + y ≠ 0) → ((x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ solution_set) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1831_183149


namespace NUMINAMATH_CALUDE_cookie_problem_l1831_183145

/-- The number of guests who did not come to Brenda's mother's cookie event -/
def guests_not_came (total_guests : ℕ) (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_guests - (total_cookies / cookies_per_guest)

theorem cookie_problem :
  guests_not_came 10 18 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l1831_183145


namespace NUMINAMATH_CALUDE_has_extremum_if_a_less_than_neg_one_l1831_183105

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

/-- Theorem stating that if a < -1, then f has an extremum -/
theorem has_extremum_if_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∃ x : ℝ, f_derivative a x = 0 :=
sorry

end NUMINAMATH_CALUDE_has_extremum_if_a_less_than_neg_one_l1831_183105


namespace NUMINAMATH_CALUDE_complex_coordinate_proof_l1831_183116

theorem complex_coordinate_proof (z : ℂ) : (z - 2*I) * (1 + I) = I → z = 1/2 + 5/2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_proof_l1831_183116


namespace NUMINAMATH_CALUDE_jen_final_distance_l1831_183130

/-- Calculates the final distance from the starting point for a person walking
    at a constant rate, forward for a certain time, then back for another time. -/
def final_distance (rate : ℝ) (forward_time : ℝ) (back_time : ℝ) : ℝ :=
  rate * forward_time - rate * back_time

/-- Theorem stating that given the specific conditions of Jen's walk,
    her final distance from the starting point is 4 miles. -/
theorem jen_final_distance :
  let rate : ℝ := 4
  let forward_time : ℝ := 2
  let back_time : ℝ := 1
  final_distance rate forward_time back_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_jen_final_distance_l1831_183130


namespace NUMINAMATH_CALUDE_nancy_picked_three_apples_l1831_183154

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Keith ate -/
def keith_ate : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 4.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- Theorem stating that Nancy picked 3.0 apples -/
theorem nancy_picked_three_apples : 
  mike_apples + nancy_apples - keith_ate = apples_left :=
by sorry

end NUMINAMATH_CALUDE_nancy_picked_three_apples_l1831_183154


namespace NUMINAMATH_CALUDE_inequality_proof_l1831_183164

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) (h5 : n > 0) : 
  x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n)) ≥ 3^n / (3^(n+2) - 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1831_183164


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1831_183140

/-- Given a person who misses the bus by 4 minutes when walking at 4/5 of their usual speed,
    their usual time to catch the bus is 16 minutes. -/
theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4/5 * usual_speed) * (usual_time + 4) = usual_speed * usual_time → usual_time = 16 := by
  sorry

#check usual_time_to_catch_bus

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l1831_183140


namespace NUMINAMATH_CALUDE_reciprocal_of_one_third_l1831_183114

theorem reciprocal_of_one_third (x : ℚ) : x * (1/3) = 1 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_third_l1831_183114


namespace NUMINAMATH_CALUDE_solve_system_l1831_183160

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 260) 
  (h2 : x - y = 200) : 
  y = 30 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1831_183160


namespace NUMINAMATH_CALUDE_scholarship_problem_l1831_183139

theorem scholarship_problem (total_students : ℕ) 
  (full_merit_percent half_merit_percent sports_percent need_based_percent : ℚ)
  (full_merit_and_sports_percent half_merit_and_need_based_percent : ℚ)
  (h1 : total_students = 300)
  (h2 : full_merit_percent = 5 / 100)
  (h3 : half_merit_percent = 10 / 100)
  (h4 : sports_percent = 3 / 100)
  (h5 : need_based_percent = 7 / 100)
  (h6 : full_merit_and_sports_percent = 1 / 100)
  (h7 : half_merit_and_need_based_percent = 2 / 100) :
  ↑total_students - 
  (↑total_students * (full_merit_percent + half_merit_percent + sports_percent + need_based_percent) -
   ↑total_students * (full_merit_and_sports_percent + half_merit_and_need_based_percent)) = 234 := by
  sorry

end NUMINAMATH_CALUDE_scholarship_problem_l1831_183139
