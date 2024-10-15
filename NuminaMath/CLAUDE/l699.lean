import Mathlib

namespace NUMINAMATH_CALUDE_answer_choices_per_mc_question_l699_69999

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def true_false_combinations : ℕ := 6

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- The number of multiple-choice questions -/
def num_mc_questions : ℕ := 2

theorem answer_choices_per_mc_question :
  ∃ n : ℕ, n > 0 ∧ true_false_combinations * n^num_mc_questions = total_combinations :=
sorry

end NUMINAMATH_CALUDE_answer_choices_per_mc_question_l699_69999


namespace NUMINAMATH_CALUDE_even_function_m_value_l699_69998

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- Given f(x) = x^2 + (m+2)x + 3 is an even function, prove that m = -2 -/
theorem even_function_m_value (m : ℝ) :
  IsEven (fun x => x^2 + (m+2)*x + 3) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l699_69998


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l699_69915

theorem max_sum_of_goods (a b : ℕ+) : 
  7 * a + 19 * b = 213 →
  ∀ x y : ℕ+, 7 * x + 19 * y = 213 → a + b ≥ x + y →
  a + b = 27 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l699_69915


namespace NUMINAMATH_CALUDE_complex_square_multiply_i_l699_69905

theorem complex_square_multiply_i : (1 - Complex.I)^2 * Complex.I = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_multiply_i_l699_69905


namespace NUMINAMATH_CALUDE_remove_layer_from_10x10x10_cube_l699_69981

/-- Represents a cube made of smaller cubes -/
structure Cube where
  side_length : ℕ
  total_cubes : ℕ

/-- Calculates the number of remaining cubes after removing one layer -/
def remaining_cubes (c : Cube) : ℕ :=
  c.total_cubes - (c.side_length * c.side_length)

/-- Theorem: For a 10x10x10 cube, removing one layer leaves 900 cubes -/
theorem remove_layer_from_10x10x10_cube :
  let c : Cube := { side_length := 10, total_cubes := 1000 }
  remaining_cubes c = 900 := by
  sorry

end NUMINAMATH_CALUDE_remove_layer_from_10x10x10_cube_l699_69981


namespace NUMINAMATH_CALUDE_six_number_list_product_l699_69993

theorem six_number_list_product (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (h_order : a₁ ≤ a₂ ∧ a₂ ≤ a₃ ∧ a₃ ≤ a₄ ∧ a₄ ≤ a₅ ∧ a₅ ≤ a₆)
  (h_remove_largest : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 - 1)
  (h_remove_smallest : (a₂ + a₃ + a₄ + a₅ + a₆) / 5 = (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 + 1)
  (h_remove_both : (a₂ + a₃ + a₄ + a₅) / 4 = 20) :
  a₁ * a₆ = 375 := by
sorry

end NUMINAMATH_CALUDE_six_number_list_product_l699_69993


namespace NUMINAMATH_CALUDE_ramsey_r33_l699_69950

-- Define a type for the colors of the edges
inductive Color
| Red
| Blue

-- Define the graph type
def Graph := Fin 6 → Fin 6 → Color

-- Define what it means for three vertices to form a monochromatic triangle
def IsMonochromaticTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  g v1 v2 = g v2 v3 ∧ g v2 v3 = g v3 v1

-- State the theorem
theorem ramsey_r33 (g : Graph) :
  (∀ (v1 v2 : Fin 6), v1 ≠ v2 → g v1 v2 = g v2 v1) →  -- Symmetry condition
  (∃ (v1 v2 v3 : Fin 6), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ IsMonochromaticTriangle g v1 v2 v3) :=
by
  sorry

end NUMINAMATH_CALUDE_ramsey_r33_l699_69950


namespace NUMINAMATH_CALUDE_bridge_length_l699_69912

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time_s : Real) :
  train_length = 130 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  ∃ (bridge_length : Real),
    bridge_length = 245 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time_s :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l699_69912


namespace NUMINAMATH_CALUDE_complex_product_equality_l699_69946

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (Complex.I * π / 9)) : 
  (2*x + x^3) * (2*x^3 + x^9) * (2*x^6 + x^18) * (2*x^9 + x^27) * (2*x^12 + x^36) * (2*x^15 + x^45) = 549 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equality_l699_69946


namespace NUMINAMATH_CALUDE_abcd_over_hife_value_l699_69957

theorem abcd_over_hife_value (a b c d e f g h i : ℝ) 
  (hab : a / b = 1 / 3)
  (hbc : b / c = 2)
  (hcd : c / d = 1 / 2)
  (hde : d / e = 3)
  (hef : e / f = 1 / 10)
  (hfg : f / g = 3 / 4)
  (hgh : g / h = 1 / 5)
  (hhi : h / i = 5)
  (h_nonzero : b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧ i ≠ 0) :
  a * b * c * d / (h * i * f * e) = 432 / 25 := by
  sorry

end NUMINAMATH_CALUDE_abcd_over_hife_value_l699_69957


namespace NUMINAMATH_CALUDE_polynomial_factorization_l699_69901

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 6*x + 8) + (x^2 + 5*x - 7) = (x^2 + 5*x + 2) * (x^2 + 5*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l699_69901


namespace NUMINAMATH_CALUDE_system_solution_l699_69980

theorem system_solution (x y : ℝ) : 
  (x + y = 1 ∧ 2*x + y = 5) → (x = 4 ∧ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l699_69980


namespace NUMINAMATH_CALUDE_barbed_wire_height_l699_69923

theorem barbed_wire_height (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  area = 3136 →
  cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let wire_length := perimeter - (↑num_gates * gate_width)
  let wire_cost := wire_length * cost_per_meter
  let height := (total_cost - wire_cost) / wire_length
  height = 2 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_height_l699_69923


namespace NUMINAMATH_CALUDE_min_value_expression_l699_69918

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 3) :
  a^2 + 4*a*b + 8*b^2 + 10*b*c + 3*c^2 ≥ 27 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 4*a₀*b₀ + 8*b₀^2 + 10*b₀*c₀ + 3*c₀^2 = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l699_69918


namespace NUMINAMATH_CALUDE_no_real_solutions_l699_69951

theorem no_real_solutions : ∀ x : ℝ, (x^2000 / 2001 + 2 * Real.sqrt 3 * x^2 - 2 * Real.sqrt 5 * x + Real.sqrt 3) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l699_69951


namespace NUMINAMATH_CALUDE_second_year_students_l699_69908

/-- Represents the number of students in each year and the total number of students. -/
structure SchoolPopulation where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ
  total : ℕ

/-- Represents the sample size and the number of first-year students in the sample. -/
structure Sample where
  size : ℕ
  firstYearSample : ℕ

/-- 
Proves that given the conditions of the problem, the number of second-year students is 300.
-/
theorem second_year_students 
  (school : SchoolPopulation)
  (sample : Sample)
  (h1 : school.firstYear = 450)
  (h2 : school.thirdYear = 250)
  (h3 : sample.size = 60)
  (h4 : sample.firstYearSample = 27)
  (h5 : (school.firstYear : ℚ) / school.total = sample.firstYearSample / sample.size) :
  school.secondYear = 300 := by
  sorry

#check second_year_students

end NUMINAMATH_CALUDE_second_year_students_l699_69908


namespace NUMINAMATH_CALUDE_range_of_m_l699_69994

/-- A function f(x) = x^2 - 2x + m where x is a real number -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The theorem stating the range of m given the conditions -/
theorem range_of_m (m : ℝ) : 
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) → 
  (∀ x, f m (1 - x) ≥ -1) → 
  m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l699_69994


namespace NUMINAMATH_CALUDE_tetrahedron_stripe_probability_l699_69935

/-- Represents the orientation of a stripe on a face of a tetrahedron -/
inductive StripeOrientation
  | First
  | Second
  | Third

/-- Represents the configuration of stripes on a tetrahedron -/
def TetrahedronStripes := Fin 4 → StripeOrientation

/-- Predicate to check if a given configuration of stripes forms a continuous stripe around the tetrahedron -/
def isContinuousStripe (config : TetrahedronStripes) : Prop := sorry

/-- The total number of possible stripe configurations -/
def totalConfigurations : ℕ := 3^4

/-- The number of configurations that form a continuous stripe -/
def favorableConfigurations : ℕ := 18

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem tetrahedron_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 2 / 9 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_stripe_probability_l699_69935


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l699_69906

theorem inscribed_circle_radius (XY XZ YZ : ℝ) (h1 : XY = 26) (h2 : XZ = 15) (h3 : YZ = 17) :
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  area / s = 2 * Real.sqrt 42 / 29 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l699_69906


namespace NUMINAMATH_CALUDE_lcm_of_180_and_504_l699_69919

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_180_and_504_l699_69919


namespace NUMINAMATH_CALUDE_leila_marathon_distance_l699_69917

/-- Represents the total distance covered in marathons -/
structure MarathonDistance where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance covered in multiple marathons -/
def totalDistance (numMarathons : ℕ) (marathonMiles : ℕ) (marathonYards : ℕ) (yardsPerMile : ℕ) : MarathonDistance :=
  sorry

/-- Theorem stating the total distance covered by Leila in her marathons -/
theorem leila_marathon_distance :
  let numMarathons : ℕ := 15
  let marathonMiles : ℕ := 26
  let marathonYards : ℕ := 385
  let yardsPerMile : ℕ := 1760
  let result := totalDistance numMarathons marathonMiles marathonYards yardsPerMile
  result.miles = 393 ∧ result.yards = 495 ∧ result.yards < yardsPerMile :=
by sorry

end NUMINAMATH_CALUDE_leila_marathon_distance_l699_69917


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l699_69967

theorem salary_increase_percentage (initial_salary final_salary : ℝ) 
  (increase_percentage decrease_percentage : ℝ) :
  initial_salary = 5000 →
  final_salary = 5225 →
  decrease_percentage = 5 →
  final_salary = initial_salary * (1 + increase_percentage / 100) * (1 - decrease_percentage / 100) →
  increase_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l699_69967


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l699_69988

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 1| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l699_69988


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l699_69907

theorem rectangle_area_increase (length : ℝ) (breadth : ℝ) 
  (h1 : length = 40)
  (h2 : breadth = 20)
  (h3 : length = 2 * breadth) :
  (length - 5) * (breadth + 5) - length * breadth = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l699_69907


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l699_69938

/-- Represents a number in a given base -/
def representIn (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a representation in a given base to a natural number -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  representIn 8 a = [1, 1] ∧
  representIn 8 b = [2, 2] ∧
  (∀ (n : ℕ) (a' b' : ℕ), a' > 2 → b' > 2 →
    representIn n a' = [1, 1] →
    representIn n b' = [2, 2] →
    n ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l699_69938


namespace NUMINAMATH_CALUDE_rocky_first_round_knockouts_l699_69995

def total_fights : ℕ := 190
def knockout_percentage : ℚ := 1/2
def first_round_knockout_percentage : ℚ := 1/5

theorem rocky_first_round_knockouts :
  (total_fights : ℚ) * knockout_percentage * first_round_knockout_percentage = 19 := by
  sorry

end NUMINAMATH_CALUDE_rocky_first_round_knockouts_l699_69995


namespace NUMINAMATH_CALUDE_sqrt_problem_proportional_function_l699_69961

-- Problem 1
theorem sqrt_problem : Real.sqrt 18 - Real.sqrt 24 / Real.sqrt 3 = Real.sqrt 2 := by sorry

-- Problem 2
theorem proportional_function (f : ℝ → ℝ) (h1 : ∀ x y, f (x + y) = f x + f y) (h2 : f 1 = 2) :
  ∀ x, f x = 2 * x := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_proportional_function_l699_69961


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l699_69954

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : 
  ∀ (x y : ℝ), x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l699_69954


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l699_69924

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope k
def line (k x y : ℝ) : Prop := y = k*(x - 2)

-- Define the intersection points
def intersection (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line k p.1 p.2}

-- Define the condition AF = 2FB
def point_condition (A B : ℝ × ℝ) : Prop :=
  (4 - A.1, -A.2) = (2*(B.1 - 4), 2*B.2)

-- Theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  ∃ (A B : ℝ × ℝ), A ∈ intersection k ∧ B ∈ intersection k ∧
  point_condition A B → |k| = 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l699_69924


namespace NUMINAMATH_CALUDE_x_value_in_set_l699_69986

theorem x_value_in_set (x : ℝ) : -2 ∈ ({3, 5, x, x^2 + 3*x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l699_69986


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l699_69991

/-- The speed of a train excluding stoppages, given its speed including stoppages and stop duration. -/
theorem train_speed_excluding_stoppages 
  (speed_with_stops : ℝ) 
  (stop_duration : ℝ) 
  (h1 : speed_with_stops = 32) 
  (h2 : stop_duration = 20) : 
  speed_with_stops * 60 / (60 - stop_duration) = 48 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l699_69991


namespace NUMINAMATH_CALUDE_line_one_point_not_always_tangent_l699_69941

-- Define a curve as a set of points in 2D space
def Curve := Set (ℝ × ℝ)

-- Define a line as a set of points in 2D space
def Line := Set (ℝ × ℝ)

-- Define what it means for a line to be tangent to a curve
def IsTangent (l : Line) (c : Curve) : Prop := sorry

-- Define what it means for a line to have only one common point with a curve
def HasOneCommonPoint (l : Line) (c : Curve) : Prop := sorry

-- Theorem statement
theorem line_one_point_not_always_tangent :
  ∃ (l : Line) (c : Curve), HasOneCommonPoint l c ∧ ¬IsTangent l c := by sorry

end NUMINAMATH_CALUDE_line_one_point_not_always_tangent_l699_69941


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l699_69963

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem f_extrema_on_interval :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc 1 3, f x ≥ min ∧ f x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc 1 3, f x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc 1 3, f x₂ = max) ∧
    min = 1 ∧ max = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l699_69963


namespace NUMINAMATH_CALUDE_fraction_comparison_l699_69947

theorem fraction_comparison (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l699_69947


namespace NUMINAMATH_CALUDE_problem_statement_l699_69914

theorem problem_statement : 
  (∃ x : ℝ, x - x + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a^2 < b^2 → a < b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l699_69914


namespace NUMINAMATH_CALUDE_janes_drawing_paper_l699_69936

/-- The number of old, brown sheets of drawing paper Jane has. -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has. -/
def yellow_sheets : ℕ := 27

/-- The total number of sheets of drawing paper Jane has. -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem janes_drawing_paper : total_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_janes_drawing_paper_l699_69936


namespace NUMINAMATH_CALUDE_daily_tylenol_intake_l699_69989

def tablets_per_dose : ℕ := 2
def mg_per_tablet : ℕ := 375
def hours_between_doses : ℕ := 6
def hours_per_day : ℕ := 24

theorem daily_tylenol_intake :
  tablets_per_dose * mg_per_tablet * (hours_per_day / hours_between_doses) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_daily_tylenol_intake_l699_69989


namespace NUMINAMATH_CALUDE_cube_difference_1234567_l699_69934

theorem cube_difference_1234567 : ∃ a b : ℤ, a^3 - b^3 = 1234567 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_1234567_l699_69934


namespace NUMINAMATH_CALUDE_polynomial_remainder_l699_69921

/-- The polynomial p(x) = x^3 - 2x^2 + x + 1 -/
def p (x : ℝ) : ℝ := x^3 - 2*x^2 + x + 1

/-- The remainder when p(x) is divided by (x-4) -/
def remainder : ℝ := p 4

theorem polynomial_remainder : remainder = 37 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l699_69921


namespace NUMINAMATH_CALUDE_spending_problem_solution_l699_69940

def spending_problem (initial_money : ℝ) : Prop :=
  let remaining_after_first := initial_money - (initial_money / 2 + 200)
  let spent_at_second := remaining_after_first / 2 + 300
  initial_money - (initial_money / 2 + 200) - spent_at_second = 350

theorem spending_problem_solution :
  spending_problem 3000 := by sorry

end NUMINAMATH_CALUDE_spending_problem_solution_l699_69940


namespace NUMINAMATH_CALUDE_handshakes_at_reunion_l699_69978

/-- Represents a family reunion with married couples -/
structure FamilyReunion where
  couples : ℕ
  people_per_couple : ℕ := 2

/-- Calculates the total number of handshakes at a family reunion -/
def total_handshakes (reunion : FamilyReunion) : ℕ :=
  let total_people := reunion.couples * reunion.people_per_couple
  let handshakes_per_person := total_people - 1 - 1 - (3 * reunion.people_per_couple)
  (total_people * handshakes_per_person) / 2

/-- Theorem: The total number of handshakes at a specific family reunion is 64 -/
theorem handshakes_at_reunion :
  let reunion : FamilyReunion := { couples := 8 }
  total_handshakes reunion = 64 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_at_reunion_l699_69978


namespace NUMINAMATH_CALUDE_table_formula_proof_l699_69960

theorem table_formula_proof : 
  (∀ (x y : ℕ), (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ 
   (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) → y = x^2 + x + 1) :=
by sorry

end NUMINAMATH_CALUDE_table_formula_proof_l699_69960


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l699_69965

-- Define a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  is_right_triangle : angle1 + angle2 = 90

-- Define the condition that the ratio of the two acute angles is 3:1
def angle_ratio (t : RightTriangle) : Prop :=
  t.angle1 / t.angle2 = 3

-- Theorem statement
theorem right_triangle_acute_angles 
  (t : RightTriangle) 
  (h : angle_ratio t) : 
  (t.angle1 = 67.5 ∧ t.angle2 = 22.5) ∨ (t.angle1 = 22.5 ∧ t.angle2 = 67.5) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l699_69965


namespace NUMINAMATH_CALUDE_plane_angle_in_right_triangle_l699_69970

/-- Given a right triangle and a plane through its hypotenuse, 
    this theorem relates the angles the plane makes with the triangle and its legs. -/
theorem plane_angle_in_right_triangle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π / 2) 
  (h_β : 0 < β ∧ β < π / 2) : 
  ∃ γ, γ = Real.arcsin (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) ∧ 
           0 ≤ γ ∧ γ ≤ π / 2 := by
  sorry


end NUMINAMATH_CALUDE_plane_angle_in_right_triangle_l699_69970


namespace NUMINAMATH_CALUDE_blue_socks_count_l699_69928

/-- Represents the number of pairs of socks Luis bought -/
structure SockPurchase where
  red : ℕ
  blue : ℕ

/-- Represents the cost of socks in dollars -/
structure SockCost where
  red : ℕ
  blue : ℕ

/-- Calculates the total cost of the sock purchase -/
def totalCost (purchase : SockPurchase) (cost : SockCost) : ℕ :=
  purchase.red * cost.red + purchase.blue * cost.blue

theorem blue_socks_count (purchase : SockPurchase) (cost : SockCost) :
  purchase.red = 4 →
  cost.red = 3 →
  cost.blue = 5 →
  totalCost purchase cost = 42 →
  purchase.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_socks_count_l699_69928


namespace NUMINAMATH_CALUDE_min_distance_to_curve_l699_69982

theorem min_distance_to_curve :
  let f (x y : ℝ) := Real.sqrt (x^2 + y^2)
  let g (x y : ℝ) := 6*x + 8*y - 4*x^2
  ∃ (min : ℝ), min = Real.sqrt 2061 / 8 ∧
    (∀ x y : ℝ, g x y = 48 → f x y ≥ min) ∧
    (∃ x y : ℝ, g x y = 48 ∧ f x y = min) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_curve_l699_69982


namespace NUMINAMATH_CALUDE_clue_represents_8671_l699_69975

/-- Represents a mapping from characters to digits -/
def CharToDigitMap := Char → Nat

/-- Creates a mapping from the string "BEST OF LUCK" to digits 0-9 in order -/
def createBestOfLuckMap : CharToDigitMap :=
  fun c => match c with
    | 'B' => 0
    | 'E' => 1
    | 'S' => 2
    | 'T' => 3
    | 'O' => 4
    | 'F' => 5
    | 'L' => 6
    | 'U' => 7
    | 'C' => 8
    | 'K' => 9
    | _ => 0  -- Default case, should not be reached for valid inputs

/-- Converts a string to a number using the given character-to-digit mapping -/
def stringToNumber (map : CharToDigitMap) (s : String) : Nat :=
  s.foldl (fun acc c => 10 * acc + map c) 0

/-- Theorem: The code word "CLUE" represents the number 8671 -/
theorem clue_represents_8671 :
  stringToNumber createBestOfLuckMap "CLUE" = 8671 := by
  sorry

#eval stringToNumber createBestOfLuckMap "CLUE"

end NUMINAMATH_CALUDE_clue_represents_8671_l699_69975


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l699_69996

theorem fraction_sum_equation (x : ℝ) : 
  (7 / (x - 2) + x / (2 - x) = 4) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l699_69996


namespace NUMINAMATH_CALUDE_final_passengers_count_l699_69966

/-- The number of people on the bus after all stops -/
def final_passengers : ℕ :=
  let initial := 110
  let stop1 := initial - 20 + 15
  let stop2 := stop1 - 34 + 17
  let stop3 := stop2 - 18 + 7
  let stop4 := stop3 - 29 + 19
  let stop5 := stop4 - 11 + 13
  let stop6 := stop5 - 15 + 8
  let stop7 := stop6 - 13 + 5
  let stop8 := stop7 - 6 + 0
  stop8

/-- Theorem stating that the final number of passengers is 48 -/
theorem final_passengers_count : final_passengers = 48 := by
  sorry

end NUMINAMATH_CALUDE_final_passengers_count_l699_69966


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l699_69903

/-- A positive geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 3 →
  a 3 + a 4 = 12 →
  a 4 + a 5 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l699_69903


namespace NUMINAMATH_CALUDE_average_with_added_number_l699_69916

theorem average_with_added_number (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by sorry

end NUMINAMATH_CALUDE_average_with_added_number_l699_69916


namespace NUMINAMATH_CALUDE_adam_red_balls_l699_69972

/-- The number of red balls in Adam's collection --/
def red_balls (total blue pink orange : ℕ) : ℕ :=
  total - (blue + pink + orange)

/-- Theorem stating the number of red balls in Adam's collection --/
theorem adam_red_balls :
  ∀ (total blue pink orange : ℕ),
    total = 50 →
    blue = 10 →
    orange = 5 →
    pink = 3 * orange →
    red_balls total blue pink orange = 20 := by
  sorry

end NUMINAMATH_CALUDE_adam_red_balls_l699_69972


namespace NUMINAMATH_CALUDE_train_speed_calculation_l699_69977

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 250 ∧ bridge_length = 120 ∧ time = 20 →
  (train_length + bridge_length) / time = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l699_69977


namespace NUMINAMATH_CALUDE_reptile_house_count_l699_69922

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 3 * rain_forest_animals - 5

theorem reptile_house_count : reptile_house_animals = 16 := by
  sorry

end NUMINAMATH_CALUDE_reptile_house_count_l699_69922


namespace NUMINAMATH_CALUDE_conditions_implications_l699_69958

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between A, B, C, and D
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_C : C → B
axiom C_nec_not_suff_D : (D → C) ∧ ¬(C → D)

-- State the theorem to be proved
theorem conditions_implications :
  -- B is a necessary but not sufficient condition for A
  ((B → A) ∧ ¬(A → B)) ∧
  -- A is a sufficient but not necessary condition for C
  ((A → C) ∧ ¬(C → A)) ∧
  -- D is neither a sufficient nor necessary condition for A
  (¬(D → A) ∧ ¬(A → D)) := by
  sorry

end NUMINAMATH_CALUDE_conditions_implications_l699_69958


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_sum_l699_69969

theorem cubic_and_quadratic_sum (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (prod_eq : x * y = 12) : 
  x^3 + y^3 = 224 ∧ x^2 + y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_sum_l699_69969


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l699_69910

/-- Given two lines l₁ and l₂ in the xy-plane:
    l₁: mx + y - 1 = 0
    l₂: x - 2y + 5 = 0
    If l₁ is perpendicular to l₂, then m = 2. -/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∀ x y, mx + y - 1 = 0 → x - 2*y + 5 = 0 → (mx + y - 1 = 0 ∧ x - 2*y + 5 = 0) → m = 2) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l699_69910


namespace NUMINAMATH_CALUDE_unique_linear_equation_solution_l699_69930

theorem unique_linear_equation_solution (m n : ℕ+) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ+),
    (a * x.val + b * y.val = c) ↔ (x = m ∧ y = n) :=
sorry

end NUMINAMATH_CALUDE_unique_linear_equation_solution_l699_69930


namespace NUMINAMATH_CALUDE_tourist_walking_speed_l699_69979

/-- Represents the problem of calculating tourist walking speed -/
def TouristWalkingSpeedProblem (scheduled_arrival : ℝ) (actual_arrival : ℝ) (early_arrival : ℝ) (bus_speed : ℝ) : Prop :=
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    scheduled_arrival > actual_arrival ∧
    early_arrival > 0 ∧
    bus_speed > 0 ∧
    let time_diff := scheduled_arrival - actual_arrival
    let encounter_time := time_diff - early_arrival
    let bus_travel_time := early_arrival / 2
    let distance := bus_speed * bus_travel_time
    walking_speed = distance / encounter_time ∧
    walking_speed = 5

/-- The main theorem stating the solution to the tourist walking speed problem -/
theorem tourist_walking_speed :
  TouristWalkingSpeedProblem 5 3.25 0.25 60 :=
by
  sorry


end NUMINAMATH_CALUDE_tourist_walking_speed_l699_69979


namespace NUMINAMATH_CALUDE_function_value_theorem_l699_69942

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2 where f(-5) = m, prove that f(5) = -m + 4 -/
theorem function_value_theorem (a b c m : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^7 - b * x^5 + c * x^3 + 2
  f (-5) = m → f 5 = -m + 4 := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l699_69942


namespace NUMINAMATH_CALUDE_smallest_value_for_x_greater_than_one_l699_69984

theorem smallest_value_for_x_greater_than_one (x : ℝ) (hx : x > 1) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_greater_than_one_l699_69984


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l699_69955

/-- Parametric curve defined by x = 3t² + 4 and y = t² - 2, where 0 ≤ t ≤ 3 -/
def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + 4, t^2 - 2)

/-- The range of the parameter t -/
def t_range : Set ℝ := {t : ℝ | 0 ≤ t ∧ t ≤ 3}

/-- The set of points on the curve -/
def curve_points : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ t_range, p = parametric_curve t}

/-- Theorem: The curve is a line segment -/
theorem curve_is_line_segment :
  ∃ a b c : ℝ, a ≠ 0 ∧ curve_points = {p : ℝ × ℝ | a * p.1 + b * p.2 = c} ∩
    {p : ℝ × ℝ | ∃ t ∈ t_range, p = parametric_curve t} :=
by sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l699_69955


namespace NUMINAMATH_CALUDE_subset_sum_exists_l699_69926

theorem subset_sum_exists (nums : List ℕ) : 
  nums.length = 100 ∧ 
  (∀ n ∈ nums, n < 100) ∧ 
  nums.sum = 200 → 
  ∃ subset : List ℕ, subset ⊆ nums ∧ subset.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_exists_l699_69926


namespace NUMINAMATH_CALUDE_set_intersection_equals_greater_equal_one_l699_69927

-- Define the sets S and T
def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem set_intersection_equals_greater_equal_one :
  S ∩ T = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equals_greater_equal_one_l699_69927


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l699_69956

theorem reporters_covering_local_politics
  (total_reporters : ℕ)
  (h1 : total_reporters > 0)
  (politics_not_local : Real)
  (h2 : politics_not_local = 0.4)
  (not_politics : Real)
  (h3 : not_politics = 0.7) :
  (1 - politics_not_local) * (1 - not_politics) * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l699_69956


namespace NUMINAMATH_CALUDE_simplify_square_roots_l699_69959

theorem simplify_square_roots : 
  (Real.sqrt 288 / Real.sqrt 32) - (Real.sqrt 242 / Real.sqrt 121) = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l699_69959


namespace NUMINAMATH_CALUDE_greatest_square_with_nine_factors_l699_69920

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem greatest_square_with_nine_factors :
  ∃ n : ℕ, n = 196 ∧
    n < 200 ∧
    is_perfect_square n ∧
    count_factors n = 9 ∧
    ∀ m : ℕ, m < 200 → is_perfect_square m → count_factors m = 9 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_square_with_nine_factors_l699_69920


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l699_69931

theorem equation_one_integral_root :
  ∃! (x : ℤ), x - 9 / (x - 5 : ℚ) = 4 - 9 / (x - 5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l699_69931


namespace NUMINAMATH_CALUDE_picnic_men_count_l699_69992

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic attendance -/
def picnicConditions (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.adults = p.men + p.women ∧
  p.total = p.men + p.women + p.children

/-- Theorem: Given the conditions, the number of men at the picnic is 65 -/
theorem picnic_men_count (p : PicnicAttendance) :
  picnicConditions p → p.men = 65 := by
  sorry

end NUMINAMATH_CALUDE_picnic_men_count_l699_69992


namespace NUMINAMATH_CALUDE_moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l699_69997

/-- The moment of inertia of a system of points -/
noncomputable def moment_of_inertia {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) : ℝ :=
  let total_mass := (Finset.univ.sum m)
  (1 / total_mass) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2)))

/-- Theorem: Moment of inertia for unit masses -/
theorem moment_of_inertia_unit_mass {n : ℕ} (a : Fin n → Fin n → ℝ) :
  moment_of_inertia a (λ _ => 1) = 
  (1 / n) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => (a i j)^2))) :=
sorry

/-- Theorem: Moment of inertia for arbitrary masses -/
theorem moment_of_inertia_arbitrary_mass {n : ℕ} (a : Fin n → Fin n → ℝ) (m : Fin n → ℝ) :
  moment_of_inertia a m = 
  (1 / (Finset.univ.sum m)) * (Finset.sum (Finset.univ.filter (λ i => i.val < n)) 
    (λ i => Finset.sum (Finset.univ.filter (λ j => i.val < j.val)) 
      (λ j => m i * m j * (a i j)^2))) :=
sorry

end NUMINAMATH_CALUDE_moment_of_inertia_unit_mass_moment_of_inertia_arbitrary_mass_l699_69997


namespace NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l699_69943

theorem right_triangle_and_multiplicative_inverse :
  (35^2 + 312^2 = 313^2) ∧ 
  (520 * 2026 % 4231 = 1) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_and_multiplicative_inverse_l699_69943


namespace NUMINAMATH_CALUDE_shoe_difference_l699_69900

/-- Scott's number of shoe pairs -/
def scott_shoes : ℕ := 7

/-- Anthony's number of shoe pairs -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- Jim's number of shoe pairs -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- The difference between Anthony's and Jim's shoe pairs -/
theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l699_69900


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l699_69973

/-- Represents a unit cube -/
structure UnitCube where
  volume : ℝ := 1
  surfaceArea : ℝ := 6

/-- Represents the custom shape described in the problem -/
structure CustomShape where
  baseCubes : Fin 5 → UnitCube
  topCube : UnitCube
  bottomCube : UnitCube

/-- Calculates the total volume of the CustomShape -/
def totalVolume (shape : CustomShape) : ℝ :=
  7  -- 5 base cubes + 1 top cube + 1 bottom cube

/-- Calculates the total surface area of the CustomShape -/
def totalSurfaceArea (shape : CustomShape) : ℝ :=
  28  -- As calculated in the problem

/-- The main theorem to be proved -/
theorem volume_to_surface_area_ratio (shape : CustomShape) :
  totalVolume shape / totalSurfaceArea shape = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l699_69973


namespace NUMINAMATH_CALUDE_percentage_of_160_l699_69983

theorem percentage_of_160 : (3 / 8 : ℚ) / 100 * 160 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_160_l699_69983


namespace NUMINAMATH_CALUDE_boys_in_class_l699_69952

/-- Given a class with 10 girls, prove that if there are 780 ways to select 1 girl and 2 boys
    when choosing 3 students at random, then the number of boys in the class is 13. -/
theorem boys_in_class (num_girls : ℕ) (num_ways : ℕ) : 
  num_girls = 10 →
  num_ways = 780 →
  (∃ num_boys : ℕ, 
    num_ways = (num_girls.choose 1) * (num_boys.choose 2) ∧
    num_boys = 13) :=
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l699_69952


namespace NUMINAMATH_CALUDE_five_students_four_lectures_l699_69964

/-- The number of ways students can choose lectures --/
def number_of_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 5 students choosing from 4 lectures results in 4^5 choices --/
theorem five_students_four_lectures :
  number_of_choices 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_four_lectures_l699_69964


namespace NUMINAMATH_CALUDE_tan_half_product_squared_l699_69953

theorem tan_half_product_squared (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) : 
  (Real.tan (a / 2) * Real.tan (b / 2))^2 = 26 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_squared_l699_69953


namespace NUMINAMATH_CALUDE_circle_radius_is_zero_l699_69990

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem stating that the radius of the circle is 0 -/
theorem circle_radius_is_zero :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_zero_l699_69990


namespace NUMINAMATH_CALUDE_translation_office_staff_count_l699_69933

/-- The number of people working at a translation office -/
def translation_office_staff : ℕ :=
  let english_only : ℕ := 8
  let german_only : ℕ := 8
  let russian_only : ℕ := 8
  let english_german : ℕ := 1
  let german_russian : ℕ := 2
  let english_russian : ℕ := 3
  let all_three : ℕ := 1
  english_only + german_only + russian_only + english_german + german_russian + english_russian + all_three

/-- Theorem stating the number of people working at the translation office -/
theorem translation_office_staff_count : translation_office_staff = 31 := by
  sorry

end NUMINAMATH_CALUDE_translation_office_staff_count_l699_69933


namespace NUMINAMATH_CALUDE_N_properties_l699_69904

def N : ℕ := 2^2022 + 1

theorem N_properties :
  (∃ k : ℕ, N = 65 * k) ∧
  (∃ a b c d : ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ N = a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_N_properties_l699_69904


namespace NUMINAMATH_CALUDE_triangle_cannot_have_two_right_angles_l699_69948

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angles 0 + t.angles 1 + t.angles 2

-- Define a right angle
def rightAngle : ℝ := 90

-- Theorem: A triangle cannot have two right angles
theorem triangle_cannot_have_two_right_angles (t : Triangle) :
  (t.angles 0 = rightAngle ∧ t.angles 1 = rightAngle) →
  t.sumOfAngles ≠ 180 :=
sorry

end NUMINAMATH_CALUDE_triangle_cannot_have_two_right_angles_l699_69948


namespace NUMINAMATH_CALUDE_coefficient_of_y_l699_69974

theorem coefficient_of_y (b : ℝ) : 
  (5 * (2 : ℝ)^2 - b * 2 + 55 = 59) → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l699_69974


namespace NUMINAMATH_CALUDE_candy_packing_problem_l699_69925

theorem candy_packing_problem (n : ℕ) : 
  n % 10 = 6 ∧ 
  n % 15 = 11 ∧ 
  200 ≤ n ∧ n ≤ 250 → 
  n = 206 ∨ n = 236 := by
sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l699_69925


namespace NUMINAMATH_CALUDE_video_game_earnings_l699_69962

def total_games : ℕ := 16
def non_working_games : ℕ := 8
def price_per_game : ℕ := 7

theorem video_game_earnings : 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l699_69962


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l699_69939

theorem larger_divided_by_smaller (L S : ℕ) (h1 : L - S = 2395) (h2 : S = 476) (h3 : L % S = 15) :
  L / S = 6 := by
  sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l699_69939


namespace NUMINAMATH_CALUDE_amber_max_ounces_l699_69985

/-- Represents the amount of money Amber has to spend -/
def amberMoney : ℚ := 7

/-- Represents the cost of a bag of candy in dollars -/
def candyCost : ℚ := 1

/-- Represents the number of ounces in a bag of candy -/
def candyOunces : ℚ := 12

/-- Represents the cost of a bag of chips in dollars -/
def chipsCost : ℚ := 1.4

/-- Represents the number of ounces in a bag of chips -/
def chipsOunces : ℚ := 17

/-- Calculates the maximum number of ounces Amber can get -/
def maxOunces : ℚ := max (amberMoney / candyCost * candyOunces) (amberMoney / chipsCost * chipsOunces)

theorem amber_max_ounces : maxOunces = 85 := by sorry

end NUMINAMATH_CALUDE_amber_max_ounces_l699_69985


namespace NUMINAMATH_CALUDE_circle_chord_problem_l699_69976

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*a*y + a^2 - 24 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  2*x - y = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem statement
theorem circle_chord_problem :
  ∃ (a : ℝ),
    (∀ x y, circle_C x y a → ∃ x₀ y₀, center_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = 25) ∧
    a = 2 ∧
    (∀ m, ∃ chord_length,
      chord_length = Real.sqrt (4 * (25 - 5)) ∧
      (∀ x y, circle_C x y a ∧ line_l x y m →
        ∃ l, l ≤ chord_length ∧ l^2 = (x - 3)^2 + (y - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_problem_l699_69976


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l699_69971

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference / 2 →
  friend_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l699_69971


namespace NUMINAMATH_CALUDE_bus_passengers_l699_69937

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (got_off : ℕ) (final : ℕ) : 
  got_on = 7 → got_off = 9 → final = 26 → initial + got_on - got_off = final → initial = 28 := by
sorry

end NUMINAMATH_CALUDE_bus_passengers_l699_69937


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_range_of_m_l699_69911

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m}

-- State the theorems
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

theorem union_complement_A_B : (Set.univ \ A) ∪ B = {x | -2 < x ∧ x < 5} := by sorry

theorem range_of_m (m : ℝ) : B ∩ C m = C m → m < -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_A_B_range_of_m_l699_69911


namespace NUMINAMATH_CALUDE_ball_costs_and_max_purchase_l699_69929

/-- Represents the cost of basketballs and soccer balls -/
structure BallCosts where
  basketball : ℕ
  soccer : ℕ

/-- Represents the purchase constraints -/
structure PurchaseConstraints where
  total_balls : ℕ
  max_cost : ℕ

/-- Theorem stating the correct costs and maximum number of basketballs -/
theorem ball_costs_and_max_purchase 
  (costs : BallCosts) 
  (constraints : PurchaseConstraints) : 
  (2 * costs.basketball + 3 * costs.soccer = 310) → 
  (5 * costs.basketball + 2 * costs.soccer = 500) → 
  (constraints.total_balls = 60) → 
  (constraints.max_cost = 4000) → 
  (costs.basketball = 80 ∧ costs.soccer = 50 ∧ 
   (∀ m : ℕ, m * costs.basketball + (constraints.total_balls - m) * costs.soccer ≤ constraints.max_cost → m ≤ 33)) := by
  sorry

end NUMINAMATH_CALUDE_ball_costs_and_max_purchase_l699_69929


namespace NUMINAMATH_CALUDE_shaded_area_between_triangles_l699_69945

/-- The area of the shaded region between two back-to-back isosceles triangles -/
theorem shaded_area_between_triangles (b h x₀ : ℝ) :
  b > 0 → h > 0 →
  let x₁ := x₀ - b / 2
  let x₂ := x₀ + b / 2
  let y := h
  (x₂ - x₁) * y = 280 :=
by
  sorry

#check shaded_area_between_triangles 12 10 10

end NUMINAMATH_CALUDE_shaded_area_between_triangles_l699_69945


namespace NUMINAMATH_CALUDE_combined_work_time_l699_69913

theorem combined_work_time (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a = 21 → b = 6 → c = 12 → (1 / (1/a + 1/b + 1/c) : ℝ) = 84/25 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_l699_69913


namespace NUMINAMATH_CALUDE_hyperbola_equation_l699_69949

/-- Represents a hyperbola with a given asymptote and a point it passes through -/
structure Hyperbola where
  asymptote_slope : ℝ
  point : ℝ × ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of a hyperbola given its asymptote and a point -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 1/2)
    (h_point : h.point = (2 * Real.sqrt 2, 1)) :
    standard_equation 4 1 (h.point.1) (h.point.2) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l699_69949


namespace NUMINAMATH_CALUDE_smallest_exponent_sum_l699_69987

theorem smallest_exponent_sum (p q r s : ℕ+) 
  (h_eq : (3^(p:ℕ))^2 + (3^(q:ℕ))^3 + (3^(r:ℕ))^5 = (3^(s:ℕ))^7) : 
  (p:ℕ) + q + r + s ≥ 106 := by
  sorry

end NUMINAMATH_CALUDE_smallest_exponent_sum_l699_69987


namespace NUMINAMATH_CALUDE_fence_painting_problem_l699_69944

/-- Given a fence of 360 square feet to be painted by three people in the ratio 3:5:2,
    prove that the person with the smallest share paints 72 square feet. -/
theorem fence_painting_problem (total_area : ℝ) (ratio_a ratio_b ratio_c : ℕ) :
  total_area = 360 →
  ratio_a = 3 →
  ratio_b = 5 →
  ratio_c = 2 →
  (ratio_a + ratio_b + ratio_c : ℝ) * (total_area / (ratio_a + ratio_b + ratio_c : ℝ) * ratio_c) = 72 :=
by sorry

end NUMINAMATH_CALUDE_fence_painting_problem_l699_69944


namespace NUMINAMATH_CALUDE_circle_and_tangents_l699_69909

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line tangent to the circle
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 4 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, 3)

-- Define the two possible tangent lines through P
def tangent_line1 (x : ℝ) : Prop := x = 2
def tangent_line2 (x y : ℝ) : Prop := 5 * x - 12 * y + 26 = 0

theorem circle_and_tangents :
  -- The circle is tangent to the given line
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line x y) ∧
  -- The circle passes through only one point of the line
  (∀ (x y : ℝ), circle_equation x y → tangent_line x y → 
    ∀ (x' y' : ℝ), x' ≠ x ∨ y' ≠ y → circle_equation x' y' → ¬tangent_line x' y') ∧
  -- The two tangent lines pass through P and are tangent to the circle
  (tangent_line1 point_P.1 ∨ tangent_line2 point_P.1 point_P.2) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line1 x) ∧
  (∃ (x y : ℝ), circle_equation x y ∧ tangent_line2 x y) ∧
  -- There are no other tangent lines through P
  (∀ (f : ℝ → ℝ), f point_P.1 = point_P.2 → 
    (∃ (x y : ℝ), circle_equation x y ∧ y = f x) →
    (∀ x, f x = point_P.2 + (x - point_P.1) * 5 / 12 ∨ f x = point_P.2)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l699_69909


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l699_69968

theorem smallest_fraction_greater_than_three_fourths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (a : ℚ) / b > 3 / 4 →
    (73 : ℚ) / 97 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fourths_l699_69968


namespace NUMINAMATH_CALUDE_ellipse_equation_l699_69902

/-- Given an ellipse C₁ and a circle C₂, prove that C₁ has the equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- a > b > 0
  P = (0, -1) ∧  -- P(0,-1) is a vertex of C₁
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 →  -- Equation of C₁
  2 * a = 4 ∧  -- Major axis of C₁ is diameter of C₂
  ∀ x y : ℝ, x^2 + y^2 = 4 →  -- Equation of C₂
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1  -- Equation of C₁ we want to prove
  := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l699_69902


namespace NUMINAMATH_CALUDE_least_with_twelve_factors_l699_69932

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- n is the least positive integer with exactly k positive factors -/
def is_least_with_factors (n : ℕ+) (k : ℕ) : Prop :=
  num_factors n = k ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 96 12 := by sorry

end NUMINAMATH_CALUDE_least_with_twelve_factors_l699_69932
