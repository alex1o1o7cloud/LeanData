import Mathlib

namespace NUMINAMATH_CALUDE_maximum_at_one_implies_a_greater_than_neg_one_l1852_185267

/-- The function f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1 -/
def has_maximum_at_one (a b : ℝ) : Prop :=
  ∀ x, x > 0 → (Real.log x - (1/2) * a * x^2 - b * x) ≤ (Real.log 1 - (1/2) * a * 1^2 - b * 1)

/-- If f(x) = ln x - (1/2)ax² - bx has a maximum at x = 1, then a > -1 -/
theorem maximum_at_one_implies_a_greater_than_neg_one (a b : ℝ) :
  has_maximum_at_one a b → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_maximum_at_one_implies_a_greater_than_neg_one_l1852_185267


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1852_185241

/-- The eccentricity of a hyperbola with the given conditions is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let right_vertex := (a, 0)
  let line := fun (x : ℝ) => -x + a
  let asymptote1 := fun (x : ℝ) => (b / a) * x
  let asymptote2 := fun (x : ℝ) => -(b / a) * x
  let B := (a^2 / (a + b), a * b / (a + b))
  let C := (a^2 / (a - b), -a * b / (a - b))
  let vector_AB := (B.1 - right_vertex.1, B.2 - right_vertex.2)
  let vector_BC := (C.1 - B.1, C.2 - B.2)
  vector_AB = (1/2 : ℝ) • vector_BC →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1852_185241


namespace NUMINAMATH_CALUDE_eliminate_y_condition_l1852_185225

/-- Represents a system of two linear equations in two variables -/
structure LinearSystem (α : Type*) [Field α] :=
  (a₁ b₁ c₁ : α)
  (a₂ b₂ c₂ : α)

/-- Checks if y can be directly eliminated when subtracting the second equation from the first -/
def canEliminateY {α : Type*} [Field α] (sys : LinearSystem α) : Prop :=
  sys.b₁ + sys.b₂ = 0

/-- The specific linear system from the problem -/
def problemSystem (α : Type*) [Field α] (m n : α) : LinearSystem α :=
  { a₁ := 6, b₁ := m, c₁ := 3,
    a₂ := 2, b₂ := -n, c₂ := -6 }

theorem eliminate_y_condition (α : Type*) [Field α] (m n : α) :
  canEliminateY (problemSystem α m n) ↔ m + n = 0 :=
sorry

end NUMINAMATH_CALUDE_eliminate_y_condition_l1852_185225


namespace NUMINAMATH_CALUDE_sequence_inequality_l1852_185261

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a 1 + a (n + 1)) / a n > 1 + 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1852_185261


namespace NUMINAMATH_CALUDE_analogical_reasoning_is_specific_to_specific_l1852_185262

-- Define the types of reasoning
inductive ReasoningType
  | Reasonable
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | GeneralToSpecific
  | SpecificToGeneral
  | SpecificToSpecific

-- Define the property of a reasoning type
def reasoning_direction (r : ReasoningType) : ReasoningDirection :=
  match r with
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific
  | _ => ReasoningDirection.GeneralToSpecific -- Default for other types, not relevant for this problem

-- Theorem statement
theorem analogical_reasoning_is_specific_to_specific :
  reasoning_direction ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific :=
by sorry

end NUMINAMATH_CALUDE_analogical_reasoning_is_specific_to_specific_l1852_185262


namespace NUMINAMATH_CALUDE_solve_for_B_l1852_185235

theorem solve_for_B : ∃ B : ℝ, (4 * B + 5 = 25) ∧ (B = 5) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_B_l1852_185235


namespace NUMINAMATH_CALUDE_line_equation_l1852_185266

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number has the equation y = (5/3)x - 17 -/
theorem line_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t - 7
  y = (5/3) * x - 17 := by sorry

end NUMINAMATH_CALUDE_line_equation_l1852_185266


namespace NUMINAMATH_CALUDE_min_value_expression_l1852_185218

theorem min_value_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2021 ≥ 2020 ∧
  ∃ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + 2021 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1852_185218


namespace NUMINAMATH_CALUDE_octagon_area_l1852_185231

-- Define the octagon's vertices
def octagon_vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 3), (2.5, 4), (4.5, 4), (6, 1), (4.5, -2), (2.5, -3), (1, -3)]

-- Define the function to calculate the area of a polygon given its vertices
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area :
  polygon_area octagon_vertices = 34 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_l1852_185231


namespace NUMINAMATH_CALUDE_exp_ge_linear_l1852_185286

theorem exp_ge_linear (x : ℝ) : x + 1 ≤ Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_linear_l1852_185286


namespace NUMINAMATH_CALUDE_initial_machines_count_l1852_185293

/-- Given a number of machines working at a constant rate, this theorem proves
    the number of machines initially working based on their production output. -/
theorem initial_machines_count (x : ℝ) (N : ℕ) : 
  (N : ℝ) * x / 4 = 20 * 3 * x / 6 → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1852_185293


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_even_l1852_185237

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def consecutive_even (a b c d : ℤ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ is_even d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_of_four_consecutive_even (a b c d : ℤ) :
  consecutive_even a b c d → a + b + c + d = 92 → d = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_even_l1852_185237


namespace NUMINAMATH_CALUDE_b_2023_value_l1852_185212

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem b_2023_value (b : ℕ → ℝ) :
  RecurrenceSequence b →
  b 1 = 2 + Real.sqrt 5 →
  b 2010 = 12 + Real.sqrt 5 →
  b 2023 = (4 + 10 * Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_b_2023_value_l1852_185212


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1852_185249

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The given arithmetic sequence condition -/
def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  arithmetic_sequence_condition a →
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1852_185249


namespace NUMINAMATH_CALUDE_solve_equation_l1852_185274

theorem solve_equation (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1852_185274


namespace NUMINAMATH_CALUDE_negation_implication_true_l1852_185298

theorem negation_implication_true (a b c : ℝ) : 
  ¬(a > b → a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_implication_true_l1852_185298


namespace NUMINAMATH_CALUDE_division_remainder_l1852_185226

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 13 →
  divisor = 7 →
  quotient = 1 →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1852_185226


namespace NUMINAMATH_CALUDE_triangle_line_equations_l1852_185254

/-- Triangle with vertices A(0,-5), B(-3,3), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Given triangle -/
def givenTriangle : Triangle :=
  { A := (0, -5)
  , B := (-3, 3)
  , C := (2, 0) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) satisfies the line equation ax + by + c = 0 -/
def satisfiesLineEquation (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem triangle_line_equations (t : Triangle) :
  (t = givenTriangle) →
  (∃ (lab : LineEquation), lab.a = 8 ∧ lab.b = 3 ∧ lab.c = 15 ∧
    satisfiesLineEquation t.A lab ∧ satisfiesLineEquation t.B lab) ∧
  (∃ (lac : LineEquation), lac.a = 5 ∧ lac.b = -2 ∧ lac.c = -10 ∧
    satisfiesLineEquation t.A lac ∧ satisfiesLineEquation t.C lac) :=
by sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l1852_185254


namespace NUMINAMATH_CALUDE_gus_total_eggs_l1852_185256

/-- The number of eggs Gus ate for breakfast -/
def breakfast_eggs : ℕ := 2

/-- The number of eggs Gus ate for lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs Gus ate for dinner -/
def dinner_eggs : ℕ := 1

/-- The total number of eggs Gus ate -/
def total_eggs : ℕ := breakfast_eggs + lunch_eggs + dinner_eggs

theorem gus_total_eggs : total_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gus_total_eggs_l1852_185256


namespace NUMINAMATH_CALUDE_triangle_area_is_correct_l1852_185280

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the bounding line -/
def boundingLine (x y : ℝ) : Prop := 3 * x + y = 9

theorem triangle_area_is_correct : 
  triangleArea = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_correct_l1852_185280


namespace NUMINAMATH_CALUDE_no_multiple_of_five_l1852_185294

theorem no_multiple_of_five (n : ℕ) : 
  2 ≤ n → n ≤ 100 → ¬(5 ∣ (2 + 5*n + n^2 + 5*n^3 + 2*n^4)) :=
by sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_l1852_185294


namespace NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_five_l1852_185242

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem f_neg_three_gt_f_neg_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : is_decreasing_on_nonneg f) :
  f (-3) > f (-5) :=
sorry

end NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_five_l1852_185242


namespace NUMINAMATH_CALUDE_arcade_tickets_difference_l1852_185265

theorem arcade_tickets_difference (tickets_won tickets_left : ℕ) 
  (h1 : tickets_won = 48) 
  (h2 : tickets_left = 32) : 
  tickets_won - tickets_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_difference_l1852_185265


namespace NUMINAMATH_CALUDE_scatter_plot_always_possible_l1852_185263

/-- Represents statistical data for two variables -/
structure StatisticalData where
  variable1 : List ℝ
  variable2 : List ℝ
  length_eq : variable1.length = variable2.length

/-- Represents a scatter plot -/
structure ScatterPlot where
  points : List (ℝ × ℝ)

/-- Given statistical data for two variables, it is always possible to create a scatter plot -/
theorem scatter_plot_always_possible (data : StatisticalData) : 
  ∃ (plot : ScatterPlot), true := by sorry

end NUMINAMATH_CALUDE_scatter_plot_always_possible_l1852_185263


namespace NUMINAMATH_CALUDE_eagle_pairs_count_l1852_185297

/-- The number of nesting pairs of bald eagles in 1963 -/
def pairs_1963 : ℕ := 417

/-- The increase in nesting pairs since 1963 -/
def increase : ℕ := 6649

/-- The current number of nesting pairs of bald eagles in the lower 48 states -/
def current_pairs : ℕ := pairs_1963 + increase

theorem eagle_pairs_count : current_pairs = 7066 := by
  sorry

end NUMINAMATH_CALUDE_eagle_pairs_count_l1852_185297


namespace NUMINAMATH_CALUDE_exam_score_problem_l1852_185209

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℤ) :
  total_questions = 75 ∧ correct_score = 4 ∧ wrong_penalty = 1 ∧ total_score = 125 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_score - (total_questions - correct_answers) * wrong_penalty = total_score ∧
    correct_answers = 40 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1852_185209


namespace NUMINAMATH_CALUDE_pages_after_break_l1852_185277

theorem pages_after_break (total_pages : ℕ) (break_percentage : ℚ) 
  (h1 : total_pages = 30) 
  (h2 : break_percentage = 7/10) : 
  total_pages - (total_pages * break_percentage).floor = 9 := by
  sorry

end NUMINAMATH_CALUDE_pages_after_break_l1852_185277


namespace NUMINAMATH_CALUDE_three_pi_irrational_l1852_185279

/-- π is an irrational number -/
axiom pi_irrational : Irrational Real.pi

/-- The product of an irrational number and a non-zero rational number is irrational -/
axiom irrational_mul_rational {x : ℝ} (hx : Irrational x) {q : ℚ} (hq : q ≠ 0) :
  Irrational (x * ↑q)

/-- 3π is an irrational number -/
theorem three_pi_irrational : Irrational (3 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_three_pi_irrational_l1852_185279


namespace NUMINAMATH_CALUDE_expected_remaining_balls_l1852_185227

/-- Represents the number of red balls initially in the bag -/
def redBalls : ℕ := 100

/-- Represents the number of blue balls initially in the bag -/
def blueBalls : ℕ := 100

/-- Represents the total number of balls initially in the bag -/
def totalBalls : ℕ := redBalls + blueBalls

/-- Represents the process of drawing balls without replacement until all red balls are drawn -/
def drawUntilAllRed (red blue : ℕ) : ℝ := sorry

/-- Theorem stating the expected number of remaining balls after drawing all red balls -/
theorem expected_remaining_balls :
  drawUntilAllRed redBalls blueBalls = blueBalls / (totalBalls : ℝ) := by sorry

end NUMINAMATH_CALUDE_expected_remaining_balls_l1852_185227


namespace NUMINAMATH_CALUDE_smallest_y_for_81_power_gt_7_power_42_l1852_185206

theorem smallest_y_for_81_power_gt_7_power_42 :
  ∃ y : ℕ, (∀ z : ℕ, 81^z ≤ 7^42 → z < y) ∧ 81^y > 7^42 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_81_power_gt_7_power_42_l1852_185206


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1852_185221

def cyclic_system (n : ℕ) (p q : ℝ) (x y z : ℝ) : Prop :=
  y = x^n + p*x + q ∧ z = y^n + p*y + q ∧ x = z^n + p*z + q

theorem cyclic_inequality (n : ℕ) (p q : ℝ) (x y z : ℝ) 
  (h_sys : cyclic_system n p q x y z) 
  (h_n : n = 2 ∨ n = 2010) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) :
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1852_185221


namespace NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l1852_185210

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def path_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem rectangular_field_path_area_and_cost 
  (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 725 ∧ 
  path_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#check rectangular_field_path_area_and_cost

end NUMINAMATH_CALUDE_rectangular_field_path_area_and_cost_l1852_185210


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1852_185214

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 360360 :=
by sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1852_185214


namespace NUMINAMATH_CALUDE_bobby_jump_difference_l1852_185234

/-- The number of jumps Bobby can do per minute as a child -/
def child_jumps_per_minute : ℕ := 30

/-- The number of jumps Bobby can do per second as an adult -/
def adult_jumps_per_second : ℕ := 1

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
theorem bobby_jump_difference : 
  adult_jumps_per_second * seconds_per_minute - child_jumps_per_minute = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_jump_difference_l1852_185234


namespace NUMINAMATH_CALUDE_yellow_red_difference_l1852_185200

/-- The number of houses Isabella has -/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses -/
def isabellaHouses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.green = 90 ∧
  h.green + h.red = 160

/-- Theorem: Isabella has 40 fewer yellow houses than red houses -/
theorem yellow_red_difference (h : Houses) (hcond : isabellaHouses h) :
  h.red - h.yellow = 40 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_l1852_185200


namespace NUMINAMATH_CALUDE_percentage_students_owning_only_cats_l1852_185201

/-- Proves that the percentage of students owning only cats is 10% -/
theorem percentage_students_owning_only_cats
  (total_students : ℕ)
  (students_with_dogs : ℕ)
  (students_with_cats : ℕ)
  (students_with_both : ℕ)
  (h1 : total_students = 500)
  (h2 : students_with_dogs = 200)
  (h3 : students_with_cats = 100)
  (h4 : students_with_both = 50) :
  (students_with_cats - students_with_both) / total_students = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_students_owning_only_cats_l1852_185201


namespace NUMINAMATH_CALUDE_biology_physics_ratio_l1852_185243

/-- The ratio of students in Biology class to Physics class -/
theorem biology_physics_ratio :
  let girls_biology : ℕ := 3 * 25
  let boys_biology : ℕ := 25
  let students_biology : ℕ := girls_biology + boys_biology
  let students_physics : ℕ := 200
  (students_biology : ℚ) / students_physics = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_biology_physics_ratio_l1852_185243


namespace NUMINAMATH_CALUDE_female_democrat_ratio_l1852_185247

theorem female_democrat_ratio (total_participants male_participants female_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 750 →
  total_participants = male_participants + female_participants →
  male_democrats = male_participants / 4 →
  female_democrats = 125 →
  male_democrats + female_democrats = total_participants / 3 →
  2 * female_democrats = female_participants :=
by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_l1852_185247


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l1852_185252

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.53247247247

/-- The fraction representation we're aiming for -/
def fraction : ℚ := 53171 / 99900

/-- Theorem stating that the decimal equals the fraction -/
theorem decimal_equals_fraction : decimal = fraction := by sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l1852_185252


namespace NUMINAMATH_CALUDE_tricia_age_correct_l1852_185202

-- Define the ages of each person as natural numbers
def Vincent_age : ℕ := 22
def Rupert_age : ℕ := Vincent_age - 2
def Khloe_age : ℕ := Rupert_age - 10
def Eugene_age : ℕ := 3 * Khloe_age
def Yorick_age : ℕ := 2 * Eugene_age
def Amilia_age : ℕ := Yorick_age / 4
def Tricia_age : ℕ := 5

-- State the theorem
theorem tricia_age_correct : 
  Vincent_age = 22 ∧ 
  Rupert_age = Vincent_age - 2 ∧
  Khloe_age = Rupert_age - 10 ∧
  Khloe_age * 3 = Eugene_age ∧
  Eugene_age * 2 = Yorick_age ∧
  Yorick_age / 4 = Amilia_age ∧
  ∃ (n : ℕ), n * Tricia_age = Amilia_age →
  Tricia_age = 5 :=
by sorry

end NUMINAMATH_CALUDE_tricia_age_correct_l1852_185202


namespace NUMINAMATH_CALUDE_average_calls_is_40_l1852_185248

/-- Represents the number of calls answered each day for a week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average number of calls per day --/
def averageCalls (w : WeekCalls) : ℚ :=
  (w.monday + w.tuesday + w.wednesday + w.thursday + w.friday) / 5

/-- Theorem stating that for the given week of calls, the average is 40 --/
theorem average_calls_is_40 (w : WeekCalls) 
    (h1 : w.monday = 35)
    (h2 : w.tuesday = 46)
    (h3 : w.wednesday = 27)
    (h4 : w.thursday = 61)
    (h5 : w.friday = 31) :
    averageCalls w = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_calls_is_40_l1852_185248


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1852_185207

theorem polynomial_simplification (p : ℝ) :
  (5 * p^3 - 7 * p^2 + 3 * p + 8) + (-3 * p^3 + 9 * p^2 - 4 * p + 2) =
  2 * p^3 + 2 * p^2 - p + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1852_185207


namespace NUMINAMATH_CALUDE_divisor_is_six_l1852_185290

def original_number : ℕ := 427398
def subtracted_number : ℕ := 6

theorem divisor_is_six : ∃ (d : ℕ), d > 0 ∧ d = subtracted_number ∧ (original_number - subtracted_number) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_is_six_l1852_185290


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l1852_185282

theorem reciprocal_product_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a + b = 3 * a * b) : 
  (1 / a) * (1 / b) = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l1852_185282


namespace NUMINAMATH_CALUDE_intersection_A_B_l1852_185240

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 1}
def B : Set ℝ := {-2, -1, 0, 1}

theorem intersection_A_B : A ∩ B = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1852_185240


namespace NUMINAMATH_CALUDE_min_positive_temperatures_l1852_185270

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n * (n - 1) = pos_products + neg_products →
  pos_products = 68 →
  neg_products = 64 →
  ∃ k : ℕ, k ≤ n ∧ k * (k - 1) = pos_products ∧ k ≥ 4 ∧ 
  ∀ m : ℕ, m < k → m * (m - 1) ≠ pos_products :=
by sorry

end NUMINAMATH_CALUDE_min_positive_temperatures_l1852_185270


namespace NUMINAMATH_CALUDE_original_number_problem_l1852_185269

theorem original_number_problem (x : ℚ) : 2 * x + 5 = x / 2 + 20 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l1852_185269


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l1852_185238

theorem nested_square_root_equality : 
  Real.sqrt (49 * Real.sqrt (25 * Real.sqrt 9)) = 5 * Real.sqrt 7 * Real.sqrt (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l1852_185238


namespace NUMINAMATH_CALUDE_no_x_squared_term_l1852_185251

/-- Given an algebraic expression (x^2 + mx)(x - 3), if the simplified form does not contain the term x^2, then m = 3 -/
theorem no_x_squared_term (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x - 3) = x^3 + (m - 3)*x^2 - 3*m*x) →
  (m - 3 = 0) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l1852_185251


namespace NUMINAMATH_CALUDE_g_composition_of_three_l1852_185287

def g (x : ℝ) : ℝ := 7 * x + 3

theorem g_composition_of_three : g (g (g 3)) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l1852_185287


namespace NUMINAMATH_CALUDE_merchant_markup_l1852_185213

theorem merchant_markup (C : ℝ) (x : ℝ) : 
  (C * (1 + x / 100) * 0.7 = C * 1.225) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_merchant_markup_l1852_185213


namespace NUMINAMATH_CALUDE_min_value_theorem_l1852_185224

/-- The function f(x) defined as |x-a| + |x+b| -/
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

/-- The theorem stating the minimum value of (a^2/b + b^2/a) given conditions on f(x) -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 3) (hequal : ∃ x, f a b x = 3) :
  ∀ c d, c > 0 → d > 0 → c^2 / d + d^2 / c ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1852_185224


namespace NUMINAMATH_CALUDE_curve_circle_intersection_l1852_185245

theorem curve_circle_intersection :
  ∃ (x y : ℝ), (2 * x - y + 1 = 0) ∧ (x^2 + (y - Real.sqrt 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_curve_circle_intersection_l1852_185245


namespace NUMINAMATH_CALUDE_min_translation_for_even_function_l1852_185268

theorem min_translation_for_even_function (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, Real.sin (3 * (x + m) + π / 4) = Real.sin (3 * (-x + m) + π / 4)) →
  m ≥ π / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_even_function_l1852_185268


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l1852_185291

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (f n + m) ∣ (n^2 + f n * f m)

theorem unique_satisfying_function :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l1852_185291


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l1852_185211

def g (x : ℝ) : ℝ := -3 * x^3 - 2 * x^2 + x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l1852_185211


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l1852_185273

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Wednesday -/
def wednesday_rainfall : ℝ := 0.08

/-- Theorem stating that the rainfall on Tuesday is 0.42 cm -/
theorem tuesday_rainfall : 
  total_rainfall - (monday_rainfall + wednesday_rainfall) = 0.42 := by sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l1852_185273


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1852_185220

theorem contrapositive_equivalence (x : ℝ) :
  (x > 1 → x^2 > 1) ↔ (x^2 ≤ 1 → x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1852_185220


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1852_185271

theorem min_value_cyclic_fraction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / b + b / c + c / d + d / a ≥ 4 ∧
  (a / b + b / c + c / d + d / a = 4 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_l1852_185271


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1852_185203

theorem smallest_divisible_by_1_to_10 : ∀ n : ℕ, n > 0 → (∀ i : ℕ, 1 ≤ i → i ≤ 10 → i ∣ n) → n ≥ 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1852_185203


namespace NUMINAMATH_CALUDE_tonyas_christmas_gifts_l1852_185246

/-- Tonya's Christmas gift problem -/
theorem tonyas_christmas_gifts (num_sisters : ℕ) (num_dolls : ℕ) (doll_cost : ℕ) (lego_cost : ℕ) 
  (h1 : num_sisters = 2)
  (h2 : num_dolls = 4)
  (h3 : doll_cost = 15)
  (h4 : lego_cost = 20) :
  (num_dolls * doll_cost) / lego_cost = 3 := by
  sorry

#check tonyas_christmas_gifts

end NUMINAMATH_CALUDE_tonyas_christmas_gifts_l1852_185246


namespace NUMINAMATH_CALUDE_interesting_numbers_characterization_l1852_185236

def is_interesting (n : ℕ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
    n = ⌊1/a⌋ + ⌊1/b⌋ + ⌊1/c⌋

theorem interesting_numbers_characterization :
  ∀ n : ℕ, is_interesting n ↔ n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_interesting_numbers_characterization_l1852_185236


namespace NUMINAMATH_CALUDE_closet_probability_l1852_185292

def shirts : ℕ := 5
def shorts : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + shorts + socks
def articles_picked : ℕ := 4

theorem closet_probability : 
  (Nat.choose shirts 2 * Nat.choose shorts 1 * Nat.choose socks 1) / 
  Nat.choose total_articles articles_picked = 112 / 969 := by
  sorry

end NUMINAMATH_CALUDE_closet_probability_l1852_185292


namespace NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1852_185215

/-- The total surface area of a solid figure formed by placing a hemisphere on top of a cone -/
theorem hemisphere_on_cone_surface_area
  (hemisphere_radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_slant_height : ℝ)
  (hemisphere_radius_eq : hemisphere_radius = 5)
  (cone_base_radius_eq : cone_base_radius = 7)
  (cone_slant_height_eq : cone_slant_height = 14) :
  2 * π * hemisphere_radius^2 + π * hemisphere_radius^2 + π * cone_base_radius * cone_slant_height = 173 * π :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_on_cone_surface_area_l1852_185215


namespace NUMINAMATH_CALUDE_expression_evaluation_l1852_185232

theorem expression_evaluation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^8 * y^9 = 5^9 / (2 * 3^9) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1852_185232


namespace NUMINAMATH_CALUDE_sum_equals_three_fourths_l1852_185260

theorem sum_equals_three_fourths : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18 + 1/21
  let removed_terms := (1/12 : ℚ) + 1/21
  original_sum - removed_terms = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_fourths_l1852_185260


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l1852_185223

theorem nth_equation_pattern (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l1852_185223


namespace NUMINAMATH_CALUDE_company_workers_count_l1852_185204

theorem company_workers_count (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) * (1 / 10 : ℚ) + (2 * total / 3 : ℚ) * (3 / 5 : ℚ) = men →
  men = 120 →
  total - men = 280 :=
by sorry

end NUMINAMATH_CALUDE_company_workers_count_l1852_185204


namespace NUMINAMATH_CALUDE_regression_line_prediction_l1852_185296

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

theorem regression_line_prediction 
  (slope : ℝ) 
  (center_x center_y : ℝ) 
  (h_slope : slope = 1.23) 
  (h_center : center_y = slope * center_x + intercept) 
  (h_center_x : center_x = 4) 
  (h_center_y : center_y = 5) :
  let line : RegressionLine := {
    slope := slope,
    intercept := center_y - slope * center_x
  }
  line.predict 2 = 2.54 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_prediction_l1852_185296


namespace NUMINAMATH_CALUDE_magic_square_sum_l1852_185276

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (top_left : ℕ := 30)
  (top_right : ℕ := 27)
  (middle_left : ℕ := 33)
  (bottom_middle : ℕ := 18)
  (sum : ℕ)
  (row_sums : sum = top_left + b + top_right)
  (col_sums : sum = top_left + middle_left + a)
  (diag_sums : sum = top_left + c + e)
  (middle_row : sum = middle_left + c + d)
  (bottom_row : sum = a + bottom_middle + e)

/-- The sum of a and d in the magic square is 38 -/
theorem magic_square_sum (ms : MagicSquare) : ms.a + ms.d = 38 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1852_185276


namespace NUMINAMATH_CALUDE_x_value_proof_l1852_185253

theorem x_value_proof (x : ℚ) (h : (1/4 : ℚ) - (1/5 : ℚ) + (1/10 : ℚ) = 4/x) : x = 80/3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1852_185253


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1852_185233

/-- The first parabola -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- The second parabola -/
def g (x : ℝ) : ℝ := 2 * x^2 - 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-2, 3), (1/2, -4.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, f p.1 = g p.1 ↔ p ∈ intersection_points := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1852_185233


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1852_185289

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q > 1 →
  (4 * (a 2011)^2 - 8 * (a 2011) + 3 = 0) →
  (4 * (a 2012)^2 - 8 * (a 2012) + 3 = 0) →
  a 2013 + a 2014 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1852_185289


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1852_185255

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 + (5 - 3)^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1852_185255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1852_185284

def isArithmeticSequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  isArithmeticSequence a →
  isArithmeticSequence b →
  a 1 = 25 →
  b 1 = 125 →
  a 2 + b 2 = 150 →
  (a + b) 2006 = 150 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1852_185284


namespace NUMINAMATH_CALUDE_ratio_sum_squares_l1852_185208

theorem ratio_sum_squares : 
  ∀ (x y z : ℝ), 
    y = 2 * x → 
    z = 3 * x → 
    x + y + z = 12 → 
    x^2 + y^2 + z^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_l1852_185208


namespace NUMINAMATH_CALUDE_math_problem_time_l1852_185229

/-- Proves that the time to solve each math problem is 2 minutes -/
theorem math_problem_time (
  math_problems : ℕ)
  (social_studies_problems : ℕ)
  (science_problems : ℕ)
  (social_studies_time : ℚ)
  (science_time : ℚ)
  (total_time : ℚ)
  (h1 : math_problems = 15)
  (h2 : social_studies_problems = 6)
  (h3 : science_problems = 10)
  (h4 : social_studies_time = 1/2)
  (h5 : science_time = 3/2)
  (h6 : total_time = 48) :
  ∃ (math_time : ℚ), math_time * math_problems + social_studies_time * social_studies_problems + science_time * science_problems = total_time ∧ math_time = 2 := by
  sorry

#check math_problem_time

end NUMINAMATH_CALUDE_math_problem_time_l1852_185229


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1852_185230

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1852_185230


namespace NUMINAMATH_CALUDE_prime_divisibility_l1852_185285

theorem prime_divisibility (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1852_185285


namespace NUMINAMATH_CALUDE_subtract_from_percentage_l1852_185281

theorem subtract_from_percentage (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → percentage = 40 → subtrahend = 30 →
  percentage / 100 * number - subtrahend = 50 := by
sorry

end NUMINAMATH_CALUDE_subtract_from_percentage_l1852_185281


namespace NUMINAMATH_CALUDE_rectangle_perimeter_squares_l1852_185283

def rectangle_length : ℕ := 47
def rectangle_width : ℕ := 65
def square_sides : List ℕ := [3, 5, 6, 11, 17, 19, 22, 23, 24, 25]
def perimeter_squares : List ℕ := [17, 19, 22, 23, 24, 25]

theorem rectangle_perimeter_squares :
  (2 * (rectangle_length + rectangle_width) = 
   2 * (perimeter_squares[3] + perimeter_squares[4] + perimeter_squares[5] + perimeter_squares[2]) + 
   perimeter_squares[0] + perimeter_squares[1]) ∧
  (∀ s ∈ perimeter_squares, s ∈ square_sides) ∧
  (perimeter_squares.length = 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_squares_l1852_185283


namespace NUMINAMATH_CALUDE_chairs_built_in_ten_days_l1852_185216

/-- Calculates the number of chairs a worker can build in a given number of days -/
def chairs_built (shift_hours : ℕ) (build_time : ℕ) (days : ℕ) : ℕ :=
  let chairs_per_shift := min 1 (shift_hours / build_time)
  chairs_per_shift * days

/-- Theorem stating that a worker who works 8-hour shifts and takes 5 hours to build 1 chair
    can build 10 chairs in 10 days -/
theorem chairs_built_in_ten_days :
  chairs_built 8 5 10 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chairs_built_in_ten_days_l1852_185216


namespace NUMINAMATH_CALUDE_sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l1852_185295

-- Define the continued fraction representation for √2 - 1
def sqrt2_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => 1 / (2 + sqrt2_minus1_cf n)

-- Define the continued fraction representation for √3 - 1
def sqrt3_minus1_cf (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => 1 / (1 + 1 / (2 + sqrt3_minus1_cf n))

-- Define the fourth convergent of √2 - 1
def sqrt2_minus1_4th_convergent : ℚ := 12 / 29

-- Define the fourth convergent of √3 - 1
def sqrt3_minus1_4th_convergent : ℚ := 8 / 11

theorem sqrt2_minus1_cf_infinite :
  ∀ n : ℕ, sqrt2_minus1_cf n ≠ sqrt2_minus1_cf (n+1) :=
sorry

theorem sqrt3_minus1_cf_infinite :
  ∀ n : ℕ, sqrt3_minus1_cf n ≠ sqrt3_minus1_cf (n+1) :=
sorry

theorem sqrt2_minus1_4th_convergent_error :
  |Real.sqrt 2 - 1 - sqrt2_minus1_4th_convergent| < 1 / 2000 :=
sorry

theorem sqrt3_minus1_4th_convergent_error :
  |Real.sqrt 3 - 1 - sqrt3_minus1_4th_convergent| < 1 / 209 :=
sorry

end NUMINAMATH_CALUDE_sqrt2_minus1_cf_infinite_sqrt3_minus1_cf_infinite_sqrt2_minus1_4th_convergent_error_sqrt3_minus1_4th_convergent_error_l1852_185295


namespace NUMINAMATH_CALUDE_swordtails_count_l1852_185244

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The amount of food each Goldfish gets (in teaspoons) -/
def goldfish_food : ℚ := 1

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets (in teaspoons) -/
def guppy_food : ℚ := 1/2

/-- The amount of food each Swordtail gets (in teaspoons) -/
def swordtail_food : ℚ := 2

/-- The total amount of food given to all fish (in teaspoons) -/
def total_food : ℚ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

theorem swordtails_count : 
  (num_goldfish : ℚ) * goldfish_food + 
  (num_guppies : ℚ) * guppy_food + 
  (num_swordtails : ℚ) * swordtail_food = total_food :=
sorry

end NUMINAMATH_CALUDE_swordtails_count_l1852_185244


namespace NUMINAMATH_CALUDE_smallest_divisible_by_3_5_7_13_greater_than_1000_l1852_185222

theorem smallest_divisible_by_3_5_7_13_greater_than_1000 : ∃ n : ℕ,
  n > 1000 ∧
  n % 3 = 0 ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n % 13 = 0 ∧
  (∀ m : ℕ, m > 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1365 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_3_5_7_13_greater_than_1000_l1852_185222


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1852_185239

theorem cube_equation_solution (x y z : ℕ) (h : x^3 = 3*y^3 + 9*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1852_185239


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l1852_185264

/-- Represents the ages of Tom and Jerry -/
structure Ages where
  tom : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.tom - 3 = 4 * (ages.jerry - 3)) ∧ 
  (ages.tom - 8 = 5 * (ages.jerry - 8))

/-- The future age ratio condition -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.jerry + years) = ages.tom + years

/-- The main theorem to prove -/
theorem age_ratio_theorem : 
  ∃ (ages : Ages), age_conditions ages → future_ratio ages 7 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l1852_185264


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1852_185228

theorem perpendicular_lines (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -9]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, v1 i * v2 i = 0) → b = 27/4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1852_185228


namespace NUMINAMATH_CALUDE_digits_of_2_pow_120_l1852_185275

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ m : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_120_l1852_185275


namespace NUMINAMATH_CALUDE_circle_perimeter_special_radius_l1852_185258

/-- The perimeter of a circle with radius 4 / π cm is 8 cm. -/
theorem circle_perimeter_special_radius :
  let r : ℝ := 4 / Real.pi
  2 * Real.pi * r = 8 := by sorry

end NUMINAMATH_CALUDE_circle_perimeter_special_radius_l1852_185258


namespace NUMINAMATH_CALUDE_integral_proof_l1852_185278

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  let f : ℝ → ℝ := λ x => (x^3 + 6*x^2 + 14*x + 10) / ((x+1)*(x+2)^3)
  let F : ℝ → ℝ := λ x => log (abs (x+1)) - 1 / (x+2)^2
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_integral_proof_l1852_185278


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1852_185288

theorem no_integer_solutions : ¬∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - z^2 = 45) ∧ 
  (-x^2 + 5*y*z + 3*z^2 = 28) ∧ 
  (x^2 - x*y + 9*z^2 = 140) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1852_185288


namespace NUMINAMATH_CALUDE_two_digit_integers_count_l1852_185299

def available_digits : Finset Nat := {1, 2, 3, 8, 9}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def count_two_digit_integers (digits : Finset Nat) : Nat :=
  (digits.filter (λ d ↦ d ≤ 9)).card * (digits.filter (λ d ↦ d ≤ 9)).card

theorem two_digit_integers_count :
  count_two_digit_integers available_digits = 25 := by sorry

end NUMINAMATH_CALUDE_two_digit_integers_count_l1852_185299


namespace NUMINAMATH_CALUDE_rosa_flowers_total_l1852_185219

theorem rosa_flowers_total (initial : ℝ) (gift : ℝ) (total : ℝ) 
    (h1 : initial = 67.5) 
    (h2 : gift = 90.75) 
    (h3 : total = initial + gift) : 
  total = 158.25 := by
sorry

end NUMINAMATH_CALUDE_rosa_flowers_total_l1852_185219


namespace NUMINAMATH_CALUDE_cyclist_hiker_catch_up_l1852_185217

/-- Proves that the time the cyclist travels after passing the hiker before stopping
    is equal to the time it takes the hiker to catch up to the cyclist while waiting. -/
theorem cyclist_hiker_catch_up (hiker_speed cyclist_speed : ℝ) (wait_time : ℝ) :
  hiker_speed > 0 →
  cyclist_speed > hiker_speed →
  wait_time > 0 →
  cyclist_speed = 4 * hiker_speed →
  (cyclist_speed / hiker_speed - 1) * wait_time = wait_time :=
by
  sorry

#check cyclist_hiker_catch_up

end NUMINAMATH_CALUDE_cyclist_hiker_catch_up_l1852_185217


namespace NUMINAMATH_CALUDE_race_track_distance_squared_l1852_185259

theorem race_track_distance_squared (inner_radius outer_radius : ℝ) 
  (h_inner : inner_radius = 11) 
  (h_outer : outer_radius = 12) 
  (separation_angle : ℝ) 
  (h_angle : separation_angle = 30 * π / 180) : 
  (inner_radius^2 + outer_radius^2 - 2 * inner_radius * outer_radius * Real.cos separation_angle) 
  = 265 - 132 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_race_track_distance_squared_l1852_185259


namespace NUMINAMATH_CALUDE_smallest_tree_height_l1852_185272

/-- Proves that the height of the smallest tree is 12 feet given the conditions of the problem -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 ∧ 
  middle = tallest / 2 - 6 ∧ 
  smallest = middle / 4 → 
  smallest = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l1852_185272


namespace NUMINAMATH_CALUDE_son_age_l1852_185205

theorem son_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_l1852_185205


namespace NUMINAMATH_CALUDE_ratio_problem_l1852_185250

/-- Given two numbers with a 20:1 ratio where the first number is 200, 
    the second number is 10. -/
theorem ratio_problem (a b : ℝ) : 
  (a / b = 20) → (a = 200) → (b = 10) := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1852_185250


namespace NUMINAMATH_CALUDE_school_population_l1852_185257

theorem school_population (t : ℕ) : 
  let g := 4 * t          -- number of girls
  let b := 6 * g          -- number of boys
  let s := t / 2          -- number of staff members
  b + g + t + s = 59 * t / 2 := by
sorry

end NUMINAMATH_CALUDE_school_population_l1852_185257
