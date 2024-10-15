import Mathlib

namespace NUMINAMATH_CALUDE_percentage_difference_l702_70299

theorem percentage_difference (x : ℝ) : 
  (x / 100) * 170 - 0.35 * 300 = 31 → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l702_70299


namespace NUMINAMATH_CALUDE_sum_squares_interior_8th_row_l702_70278

/-- Pascal's Triangle row function -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Function to get interior numbers of a row -/
def interior_numbers (row : List ℕ) : List ℕ := sorry

/-- Sum of squares function -/
def sum_of_squares (list : List ℕ) : ℕ := sorry

/-- Theorem: Sum of squares of interior numbers in 8th row of Pascal's Triangle is 3430 -/
theorem sum_squares_interior_8th_row : 
  sum_of_squares (interior_numbers (pascal_row 8)) = 3430 := by sorry

end NUMINAMATH_CALUDE_sum_squares_interior_8th_row_l702_70278


namespace NUMINAMATH_CALUDE_power_product_equality_l702_70221

theorem power_product_equality (a b : ℝ) : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l702_70221


namespace NUMINAMATH_CALUDE_expand_product_l702_70207

theorem expand_product (x : ℝ) : -3 * (2 * x + 4) * (x - 7) = -6 * x^2 + 30 * x + 84 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l702_70207


namespace NUMINAMATH_CALUDE_probability_of_triangle_in_15_gon_l702_70297

/-- Definition of a regular 15-gon -/
def regular_15_gon : Set (ℝ × ℝ) := sorry

/-- Function to check if three segments can form a triangle with positive area -/
def can_form_triangle (s1 s2 s3 : ℝ × ℝ × ℝ × ℝ) : Prop := sorry

/-- Total number of ways to choose 3 distinct segments from a 15-gon -/
def total_choices : ℕ := Nat.choose (Nat.choose 15 2) 3

/-- Number of ways to choose 3 distinct segments that form a triangle -/
def valid_choices : ℕ := sorry

theorem probability_of_triangle_in_15_gon :
  (valid_choices : ℚ) / total_choices = 163 / 455 := by sorry

end NUMINAMATH_CALUDE_probability_of_triangle_in_15_gon_l702_70297


namespace NUMINAMATH_CALUDE_equal_pay_implies_harry_worked_33_hours_l702_70270

/-- Payment structure for an employee -/
structure PaymentStructure where
  base_rate : ℝ
  base_hours : ℕ
  overtime_multiplier : ℝ

/-- Calculate the total pay for an employee given their payment structure and hours worked -/
def calculate_pay (ps : PaymentStructure) (hours_worked : ℕ) : ℝ :=
  let base_pay := ps.base_rate * (min ps.base_hours hours_worked)
  let overtime_hours := max 0 (hours_worked - ps.base_hours)
  let overtime_pay := ps.base_rate * ps.overtime_multiplier * overtime_hours
  base_pay + overtime_pay

theorem equal_pay_implies_harry_worked_33_hours 
  (x : ℝ) 
  (harry_structure : PaymentStructure)
  (james_structure : PaymentStructure)
  (h_harry : harry_structure = { base_rate := x, base_hours := 15, overtime_multiplier := 1.5 })
  (h_james : james_structure = { base_rate := x, base_hours := 40, overtime_multiplier := 2 })
  (james_hours : ℕ)
  (h_james_hours : james_hours = 41)
  (harry_hours : ℕ)
  (h_equal_pay : calculate_pay harry_structure harry_hours = calculate_pay james_structure james_hours) :
  harry_hours = 33 := by
  sorry

end NUMINAMATH_CALUDE_equal_pay_implies_harry_worked_33_hours_l702_70270


namespace NUMINAMATH_CALUDE_a_value_proof_l702_70285

theorem a_value_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l702_70285


namespace NUMINAMATH_CALUDE_bacteria_increase_l702_70226

theorem bacteria_increase (original : ℕ) (current : ℕ) (increase : ℕ) : 
  original = 600 → current = 8917 → increase = current - original → increase = 8317 := by
sorry

end NUMINAMATH_CALUDE_bacteria_increase_l702_70226


namespace NUMINAMATH_CALUDE_line_l_equation_tangent_circle_a_l702_70248

-- Define the lines and circle
def l1 (x y : ℝ) : Prop := 2 * x - y = 1
def l2 (x y : ℝ) : Prop := x + 2 * y = 3
def l3 (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8 ∧ a > 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statements
theorem line_l_equation :
  (∀ x y : ℝ, l1 x y ∧ l2 x y → (x, y) = P) →
  (∀ x y : ℝ, l x y → l3 ((x + 2) / 2) ((2 - x) / 2)) →
  ∀ x y : ℝ, l x y ↔ x + y - 2 = 0 := by sorry

theorem tangent_circle_a :
  (∀ x y : ℝ, l x y → C x y 6) →
  (∀ x y a : ℝ, l x y → C x y a → a = 6) := by sorry

end NUMINAMATH_CALUDE_line_l_equation_tangent_circle_a_l702_70248


namespace NUMINAMATH_CALUDE_cubes_fill_box_l702_70272

/-- Proves that 2-inch cubes fill 100% of a 8×6×12 inch box -/
theorem cubes_fill_box (box_length box_width box_height cube_side: ℕ) 
  (h1: box_length = 8)
  (h2: box_width = 6)
  (h3: box_height = 12)
  (h4: cube_side = 2)
  (h5: box_length % cube_side = 0)
  (h6: box_width % cube_side = 0)
  (h7: box_height % cube_side = 0) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side)) * cube_side^3) / (box_length * box_width * box_height) = 1 :=
by sorry

end NUMINAMATH_CALUDE_cubes_fill_box_l702_70272


namespace NUMINAMATH_CALUDE_sqrt_sixteen_times_sqrt_sixteen_equals_eight_l702_70203

theorem sqrt_sixteen_times_sqrt_sixteen_equals_eight : Real.sqrt (16 * Real.sqrt 16) = 2^3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sixteen_times_sqrt_sixteen_equals_eight_l702_70203


namespace NUMINAMATH_CALUDE_isosceles_triangle_apex_angle_l702_70279

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle : ℝ
  apex_angle : ℝ
  is_isosceles : base_angle ≥ 0 ∧ apex_angle ≥ 0
  angle_sum : 2 * base_angle + apex_angle = 180

-- Theorem statement
theorem isosceles_triangle_apex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 42) : 
  triangle.apex_angle = 96 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_apex_angle_l702_70279


namespace NUMINAMATH_CALUDE_parabola_locus_l702_70292

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P lies on parabola C -/
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola_C P.1 P.2

/-- Vector relation between P, Q, and F -/
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (P.1 - Q.1, P.2 - Q.2) = (2*(focus.1 - Q.1), 2*(focus.2 - Q.2))

/-- Curve E: 9y² = 12x - 8 -/
def curve_E (x y : ℝ) : Prop := 9*y^2 = 12*x - 8

theorem parabola_locus :
  ∀ Q : ℝ × ℝ,
  (∃ P : ℝ × ℝ, point_on_parabola P ∧ vector_relation P Q) →
  curve_E Q.1 Q.2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_locus_l702_70292


namespace NUMINAMATH_CALUDE_constant_speed_journey_time_l702_70296

/-- Given a constant speed journey, prove the total travel time -/
theorem constant_speed_journey_time 
  (total_distance : ℝ) 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 400) 
  (h2 : initial_distance = 100) 
  (h3 : initial_time = 1) 
  (h4 : initial_distance / initial_time = (total_distance - initial_distance) / (total_time - initial_time)) : 
  total_time = 4 :=
by
  sorry

#check constant_speed_journey_time

end NUMINAMATH_CALUDE_constant_speed_journey_time_l702_70296


namespace NUMINAMATH_CALUDE_smallest_cookie_boxes_l702_70227

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m) ∧ 
  (15 * n - 1) % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cookie_boxes_l702_70227


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l702_70263

/-- The number of ways to arrange comics as described in the problem -/
def comic_arrangements (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) : ℕ :=
  (Nat.factorial (batman + xmen)) * (Nat.factorial calvin_hobbes) * 2

/-- Theorem stating the correct number of arrangements for the given comic counts -/
theorem comic_arrangement_count :
  comic_arrangements 7 6 5 = 1494084992000 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l702_70263


namespace NUMINAMATH_CALUDE_no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l702_70257

-- Define a type for figures on a plane
structure PlaneFigure where
  -- Add necessary fields here
  isBounded : Bool
  hasParallelAxes : Bool

-- Define a predicate for having two parallel, non-coincident symmetry axes
def hasParallelSymmetryAxes (f : PlaneFigure) : Prop :=
  f.hasParallelAxes

theorem no_bounded_figure_with_parallel_axes :
  ¬ ∃ (f : PlaneFigure), f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

theorem exists_unbounded_figure_with_parallel_axes :
  ∃ (f : PlaneFigure), ¬f.isBounded ∧ hasParallelSymmetryAxes f := by
  sorry

end NUMINAMATH_CALUDE_no_bounded_figure_with_parallel_axes_exists_unbounded_figure_with_parallel_axes_l702_70257


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l702_70236

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1/a' + 1/b' > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l702_70236


namespace NUMINAMATH_CALUDE_f_of_5_equals_20_l702_70277

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- Theorem statement
theorem f_of_5_equals_20 : f 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_20_l702_70277


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l702_70247

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a1_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4) :
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_value_l702_70247


namespace NUMINAMATH_CALUDE_cars_to_double_earnings_l702_70206

def base_salary : ℕ := 1000
def commission_per_car : ℕ := 200
def january_earnings : ℕ := 1800

theorem cars_to_double_earnings : 
  ∃ (february_cars : ℕ), 
    base_salary + february_cars * commission_per_car = 2 * january_earnings ∧ 
    february_cars = 13 :=
by sorry

end NUMINAMATH_CALUDE_cars_to_double_earnings_l702_70206


namespace NUMINAMATH_CALUDE_lottery_win_probability_l702_70265

def megaBallCount : ℕ := 27
def winnerBallCount : ℕ := 44
def winnerBallPick : ℕ := 5

theorem lottery_win_probability :
  (1 : ℚ) / megaBallCount * (1 : ℚ) / Nat.choose winnerBallCount winnerBallPick = 1 / 29322216 :=
by sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l702_70265


namespace NUMINAMATH_CALUDE_runner_stops_on_start_quarter_l702_70280

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | X : Quarter
  | Y : Quarter
  | Z : Quarter
  | W : Quarter

/-- The circular track -/
structure Track :=
  (circumference : ℝ)
  (quarters : Fin 4 → Quarter)

/-- Represents a runner on the track -/
structure Runner :=
  (start_quarter : Quarter)
  (distance_run : ℝ)

/-- Function to determine the quarter where a runner stops -/
def stop_quarter (track : Track) (runner : Runner) : Quarter :=
  runner.start_quarter

/-- Theorem stating that a runner stops on the same quarter they started on
    when running a multiple of the track's circumference -/
theorem runner_stops_on_start_quarter 
  (track : Track) 
  (runner : Runner) 
  (h1 : track.circumference = 200)
  (h2 : runner.distance_run = 3000) :
  stop_quarter track runner = runner.start_quarter :=
sorry

end NUMINAMATH_CALUDE_runner_stops_on_start_quarter_l702_70280


namespace NUMINAMATH_CALUDE_x_eq_1_sufficient_not_necessary_for_quadratic_l702_70237

theorem x_eq_1_sufficient_not_necessary_for_quadratic : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_1_sufficient_not_necessary_for_quadratic_l702_70237


namespace NUMINAMATH_CALUDE_tracy_candies_l702_70259

theorem tracy_candies : ∃ (x : ℕ), 
  (x % 4 = 0) ∧ 
  ((3 * x / 4) % 3 = 0) ∧ 
  (x / 2 - 29 = 10) ∧ 
  x = 78 := by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_l702_70259


namespace NUMINAMATH_CALUDE_min_ones_is_one_l702_70274

/-- Represents the count of squares of each size --/
structure SquareCounts where
  threes : Nat
  twos : Nat
  ones : Nat

/-- Checks if the given square counts fit within a 7x7 square --/
def fitsIn7x7 (counts : SquareCounts) : Prop :=
  9 * counts.threes + 4 * counts.twos + counts.ones = 49

/-- Defines a valid square division --/
def isValidDivision (counts : SquareCounts) : Prop :=
  fitsIn7x7 counts ∧ counts.threes ≥ 0 ∧ counts.twos ≥ 0 ∧ counts.ones ≥ 0

/-- The main theorem stating that the minimum number of 1x1 squares is 1 --/
theorem min_ones_is_one :
  ∃ (counts : SquareCounts), isValidDivision counts ∧ counts.ones = 1 ∧
  (∀ (other : SquareCounts), isValidDivision other → other.ones ≥ counts.ones) :=
sorry

end NUMINAMATH_CALUDE_min_ones_is_one_l702_70274


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l702_70220

theorem cubic_equation_solutions : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3*I*Real.sqrt 3)/2
  let z₃ : ℂ := (3/2) - (3*I*Real.sqrt 3)/2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  (∀ z : ℂ, z^3 = -27 → z = z₁ ∨ z = z₂ ∨ z = z₃) := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l702_70220


namespace NUMINAMATH_CALUDE_quadratic_solution_and_sum_l702_70246

theorem quadratic_solution_and_sum (x : ℝ) : 
  x^2 + 14*x = 96 → 
  ∃ (a b : ℕ), 
    (x = Real.sqrt a - b) ∧ 
    (x > 0) ∧ 
    (a = 145) ∧ 
    (b = 7) ∧ 
    (a + b = 152) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_sum_l702_70246


namespace NUMINAMATH_CALUDE_population_change_l702_70256

/-- Proves that given an initial population of 15000, a 12% increase in the first year,
    and a final population of 14784 after two years, the percentage decrease in the second year is 12%. -/
theorem population_change (initial_population : ℝ) (first_year_increase : ℝ) (final_population : ℝ)
  (h1 : initial_population = 15000)
  (h2 : first_year_increase = 0.12)
  (h3 : final_population = 14784) :
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let second_year_decrease := (population_after_first_year - final_population) / population_after_first_year
  second_year_decrease = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l702_70256


namespace NUMINAMATH_CALUDE_lillians_candies_l702_70251

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end NUMINAMATH_CALUDE_lillians_candies_l702_70251


namespace NUMINAMATH_CALUDE_terminating_decimal_of_fraction_l702_70225

theorem terminating_decimal_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 65 / 1000 →
  ∃ (a b : ℕ), (n : ℚ) / d = (a : ℚ) / (10 ^ b) ∧ (a : ℚ) / (10 ^ b) = 0.065 :=
by sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_fraction_l702_70225


namespace NUMINAMATH_CALUDE_problem_solution_l702_70234

noncomputable def equation (x a : ℝ) : Prop := Real.arctan (x / 2) + Real.arctan (2 - x) = a

theorem problem_solution :
  (∀ x : ℝ, equation x (π / 4) → Real.arccos (x / 2) = 2*π/3 ∨ Real.arccos (x / 2) = 0) ∧
  (∀ a : ℝ, (∃ x : ℝ, equation x a) → a ∈ Set.Icc (Real.arctan (1 / (-2 * Real.sqrt 10 - 6))) (Real.arctan (1 / (2 * Real.sqrt 10 - 6)))) ∧
  (∀ a : ℝ, (∃ α β : ℝ, α ≠ β ∧ α ∈ Set.Icc 5 15 ∧ β ∈ Set.Icc 5 15 ∧ equation α a ∧ equation β a) →
    (∀ γ δ : ℝ, γ ≠ δ ∧ γ ∈ Set.Icc 5 15 ∧ δ ∈ Set.Icc 5 15 ∧ equation γ a ∧ equation δ a → γ + δ ≤ 19)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l702_70234


namespace NUMINAMATH_CALUDE_truck_fuel_efficiency_l702_70261

theorem truck_fuel_efficiency 
  (distance : ℝ) 
  (current_gas : ℝ) 
  (additional_gas : ℝ) 
  (h1 : distance = 90) 
  (h2 : current_gas = 12) 
  (h3 : additional_gas = 18) : 
  distance / (current_gas + additional_gas) = 3 := by
sorry

end NUMINAMATH_CALUDE_truck_fuel_efficiency_l702_70261


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l702_70266

/-- Represents a parabola in the form y = a(x-h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 3 →
  let p' := shift p 4 (-4)
  p'.a * X ^ 2 + p'.a * p'.h ^ 2 - 2 * p'.a * p'.h * X + p'.k = 3 * X ^ 2 - 1 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l702_70266


namespace NUMINAMATH_CALUDE_stone_value_proof_l702_70218

/-- Represents the worth of a precious stone based on its weight and a proportionality constant -/
def stone_worth (weight : ℝ) (k : ℝ) : ℝ := k * weight^2

/-- Calculates the total worth of two pieces of a stone -/
def pieces_worth (weight1 : ℝ) (weight2 : ℝ) (k : ℝ) : ℝ :=
  stone_worth weight1 k + stone_worth weight2 k

theorem stone_value_proof (k : ℝ) :
  let original_weight : ℝ := 35
  let smaller_piece : ℝ := 2 * (original_weight / 7)
  let larger_piece : ℝ := 5 * (original_weight / 7)
  let loss : ℝ := 5000
  stone_worth original_weight k - pieces_worth smaller_piece larger_piece k = loss →
  stone_worth original_weight k = 12250 := by
sorry

end NUMINAMATH_CALUDE_stone_value_proof_l702_70218


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l702_70271

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l702_70271


namespace NUMINAMATH_CALUDE_middle_number_is_twelve_l702_70249

/-- Given three distinct integers x, y, z satisfying the given conditions,
    prove that the middle number y equals 12. -/
theorem middle_number_is_twelve (x y z : ℤ)
  (h_distinct : x < y ∧ y < z)
  (h_sum1 : x + y = 21)
  (h_sum2 : x + z = 25)
  (h_sum3 : y + z = 28) :
  y = 12 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_twelve_l702_70249


namespace NUMINAMATH_CALUDE_football_team_right_handed_count_l702_70282

theorem football_team_right_handed_count (total_players throwers : ℕ) : 
  total_players = 70 →
  throwers = 37 →
  (total_players - throwers) % 3 = 0 →
  (throwers + (total_players - throwers) * 2 / 3 = 59) :=
by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_count_l702_70282


namespace NUMINAMATH_CALUDE_equation_solution_l702_70245

theorem equation_solution :
  ∃! x : ℚ, x ≠ -2 ∧ (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l702_70245


namespace NUMINAMATH_CALUDE_max_a_value_l702_70275

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -24) → 
  (a > 0) → 
  a ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l702_70275


namespace NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l702_70258

theorem four_digit_cubes_divisible_by_16 : 
  (∃! (list : List ℕ), 
    (∀ n ∈ list, 1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0) ∧ 
    list.length = 3) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_cubes_divisible_by_16_l702_70258


namespace NUMINAMATH_CALUDE_lemonade_proportion_l702_70210

/-- Given that 24 lemons make 32 gallons of lemonade, proves that 3 lemons make 4 gallons -/
theorem lemonade_proportion :
  (24 : ℚ) / 32 = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l702_70210


namespace NUMINAMATH_CALUDE_pencil_sharpener_time_l702_70202

/-- Represents the time in minutes for which we're solving -/
def t : ℝ := 6

/-- Time (in seconds) for hand-crank sharpener to sharpen one pencil -/
def hand_crank_time : ℝ := 45

/-- Time (in seconds) for electric sharpener to sharpen one pencil -/
def electric_time : ℝ := 20

/-- The difference in number of pencils sharpened -/
def pencil_difference : ℕ := 10

theorem pencil_sharpener_time :
  (60 * t / electric_time) = (60 * t / hand_crank_time) + pencil_difference :=
sorry

end NUMINAMATH_CALUDE_pencil_sharpener_time_l702_70202


namespace NUMINAMATH_CALUDE_triangle_proof_l702_70241

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The median from vertex A to side BC -/
def median (t : Triangle) : ℝ := sorry

/-- The area of the triangle -/
def area (t : Triangle) : ℝ := sorry

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A - Real.sqrt 3 * t.c * Real.cos t.A = Real.sqrt 3 * t.a * Real.cos t.C)
  (h2 : t.B = π / 6)
  (h3 : median t = Real.sqrt 7) :
  t.A = π / 6 ∧ area t = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_proof_l702_70241


namespace NUMINAMATH_CALUDE_convergence_of_difference_series_l702_70290

/-- Given two real sequences (a_i) and (b_i) where the series of their squares converge,
    prove that the series of |a_i - b_i|^p converges for all p ≥ 2. -/
theorem convergence_of_difference_series
  (a b : ℕ → ℝ)
  (ha : Summable (λ i => (a i)^2))
  (hb : Summable (λ i => (b i)^2))
  (p : ℝ)
  (hp : p ≥ 2) :
  Summable (λ i => |a i - b i|^p) :=
sorry

end NUMINAMATH_CALUDE_convergence_of_difference_series_l702_70290


namespace NUMINAMATH_CALUDE_test_scores_theorem_l702_70211

def is_valid_sequence (s : List Nat) : Prop :=
  s.length > 0 ∧ 
  s.Nodup ∧ 
  s.sum = 119 ∧ 
  (s.take 3).sum = 23 ∧ 
  (s.reverse.take 3).sum = 49

theorem test_scores_theorem (s : List Nat) (h : is_valid_sequence s) : 
  s.length = 10 ∧ s.maximum? = some 18 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_theorem_l702_70211


namespace NUMINAMATH_CALUDE_pencil_price_l702_70200

theorem pencil_price (x y : ℚ) 
  (eq1 : 3 * x + 5 * y = 345)
  (eq2 : 4 * x + 2 * y = 280) :
  y = 540 / 14 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l702_70200


namespace NUMINAMATH_CALUDE_fraction_simplification_positive_integer_solutions_l702_70283

-- Problem 1
theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) := by
  sorry

-- Problem 2
def inequality_system (x : ℝ) : Prop :=
  (2*x + 1) / 3 - (5*x - 1) / 2 < 1 ∧ 5*x - 1 < 3*(x + 2)

theorem positive_integer_solutions :
  {x : ℕ | inequality_system x} = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_positive_integer_solutions_l702_70283


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l702_70253

/-- The function f(x) = x ln x - (a/2)x^2 - x is monotonic on (0, +∞) if and only if a ∈ [1/e, +∞) -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x > 0, Monotone (fun x => x * Real.log x - a / 2 * x^2 - x)) ↔ a ≥ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l702_70253


namespace NUMINAMATH_CALUDE_susan_decade_fraction_l702_70213

/-- Represents the collection of quarters Susan has -/
structure QuarterCollection where
  total : ℕ
  decade_count : ℕ

/-- The fraction of quarters representing states that joined the union in a specific decade -/
def decade_fraction (c : QuarterCollection) : ℚ :=
  c.decade_count / c.total

/-- Susan's collection of quarters -/
def susan_collection : QuarterCollection :=
  { total := 22, decade_count := 7 }

theorem susan_decade_fraction :
  decade_fraction susan_collection = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_susan_decade_fraction_l702_70213


namespace NUMINAMATH_CALUDE_expression_factorization_l702_70291

theorem expression_factorization (a : ℝ) :
  (8 * a^4 + 92 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 5 * a^2 + 2) = 
  a^2 * (10 * a^2 + 89 * a - 10) - 1 := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l702_70291


namespace NUMINAMATH_CALUDE_probability_three_yellow_one_white_l702_70262

/-- The probability of drawing 3 yellow balls followed by 1 white ball from a box
    containing 5 yellow balls and 4 white balls, where yellow balls are returned
    after being drawn. -/
theorem probability_three_yellow_one_white (yellow_balls : ℕ) (white_balls : ℕ)
    (h_yellow : yellow_balls = 5) (h_white : white_balls = 4) :
    (yellow_balls / (yellow_balls + white_balls : ℚ))^3 *
    (white_balls / (yellow_balls + white_balls : ℚ)) =
    (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_three_yellow_one_white_l702_70262


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l702_70201

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l702_70201


namespace NUMINAMATH_CALUDE_triangle_formation_theorem_l702_70264

/-- Given three positive real numbers a, b, and c, they can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem states that among the given combinations, only (4, 5, 6)
    satisfies the triangle inequality and thus can form a triangle. -/
theorem triangle_formation_theorem :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 3 6 ∧
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 4 10 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_theorem_l702_70264


namespace NUMINAMATH_CALUDE_dog_catch_ball_time_l702_70231

/-- The time it takes for a dog to catch up to a thrown ball -/
theorem dog_catch_ball_time (ball_speed : ℝ) (ball_time : ℝ) (dog_speed : ℝ) :
  ball_speed = 20 →
  ball_time = 8 →
  dog_speed = 5 →
  (ball_speed * ball_time) / dog_speed = 32 := by
  sorry

#check dog_catch_ball_time

end NUMINAMATH_CALUDE_dog_catch_ball_time_l702_70231


namespace NUMINAMATH_CALUDE_remainder_h_x_10_divided_by_h_x_l702_70205

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

-- State the theorem
theorem remainder_h_x_10_divided_by_h_x : 
  ∃ (q : ℝ → ℝ), h (x^10) = q x * h x + 7 := by sorry

end NUMINAMATH_CALUDE_remainder_h_x_10_divided_by_h_x_l702_70205


namespace NUMINAMATH_CALUDE_count_nonadjacent_permutations_l702_70204

/-- The number of permutations of n distinct elements where two specific elements are not adjacent -/
def nonadjacent_permutations (n : ℕ) : ℕ :=
  (n - 2) * Nat.factorial (n - 1)

/-- Theorem stating that the number of permutations of n distinct elements 
    where two specific elements are not adjacent is (n-2)(n-1)! -/
theorem count_nonadjacent_permutations (n : ℕ) (h : n ≥ 2) :
  nonadjacent_permutations n = (n - 2) * Nat.factorial (n - 1) := by
  sorry

#check count_nonadjacent_permutations

end NUMINAMATH_CALUDE_count_nonadjacent_permutations_l702_70204


namespace NUMINAMATH_CALUDE_distance_from_negative_two_l702_70281

-- Define the distance function on the real number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem distance_from_negative_two :
  ∀ x : ℝ, distance x (-2) = 3 ↔ x = -5 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_negative_two_l702_70281


namespace NUMINAMATH_CALUDE_angelinas_speed_to_gym_l702_70224

-- Define the distances and time difference
def distance_home_to_grocery : ℝ := 1200
def distance_grocery_to_gym : ℝ := 480
def time_difference : ℝ := 40

-- Define the relationship between speeds
def speed_grocery_to_gym (v : ℝ) : ℝ := 2 * v

-- Theorem statement
theorem angelinas_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧
  distance_home_to_grocery / v - distance_grocery_to_gym / (speed_grocery_to_gym v) = time_difference ∧
  speed_grocery_to_gym v = 48 := by
  sorry

end NUMINAMATH_CALUDE_angelinas_speed_to_gym_l702_70224


namespace NUMINAMATH_CALUDE_product_xyzw_l702_70269

theorem product_xyzw (x y z w : ℝ) (h1 : x + 1/y = 1) (h2 : y + 1/z + w = 1) (h3 : w = 2) (h4 : y ≠ 0) :
  x * y * z * w = -2 * y^2 + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_product_xyzw_l702_70269


namespace NUMINAMATH_CALUDE_triangle_side_length_l702_70268

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l702_70268


namespace NUMINAMATH_CALUDE_bug_flower_problem_l702_70250

theorem bug_flower_problem (total_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) :
  total_bugs = 3 →
  total_flowers = 6 →
  total_flowers = total_bugs * flowers_per_bug →
  flowers_per_bug = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bug_flower_problem_l702_70250


namespace NUMINAMATH_CALUDE_cuboids_intersecting_diagonal_l702_70267

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  sideLength : ℕ

/-- Counts the number of cuboids intersecting the diagonal of a cube -/
def countIntersectingCuboids (cuboid : Cuboid) (cube : Cube) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem cuboids_intersecting_diagonal
  (smallCuboid : Cuboid)
  (largeCube : Cube)
  (h1 : smallCuboid.length = 2)
  (h2 : smallCuboid.width = 3)
  (h3 : smallCuboid.height = 5)
  (h4 : largeCube.sideLength = 90)
  (h5 : largeCube.sideLength % smallCuboid.length = 0)
  (h6 : largeCube.sideLength % smallCuboid.width = 0)
  (h7 : largeCube.sideLength % smallCuboid.height = 0) :
  countIntersectingCuboids smallCuboid largeCube = 65 := by
  sorry


end NUMINAMATH_CALUDE_cuboids_intersecting_diagonal_l702_70267


namespace NUMINAMATH_CALUDE_samuel_homework_time_l702_70284

theorem samuel_homework_time (sarah_time : Real) (time_difference : Nat) : 
  sarah_time = 1.3 → time_difference = 48 → 
  ⌊sarah_time * 60 - time_difference⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_samuel_homework_time_l702_70284


namespace NUMINAMATH_CALUDE_quilt_cost_calculation_l702_70222

/-- The cost of a rectangular quilt -/
def quilt_cost (length width cost_per_sqft : ℝ) : ℝ :=
  length * width * cost_per_sqft

/-- Theorem: The cost of a 7ft by 8ft quilt at $40 per square foot is $2240 -/
theorem quilt_cost_calculation :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_calculation_l702_70222


namespace NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l702_70235

theorem max_cube_sum_on_sphere (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) :
  x^3 + y^3 + z^3 ≤ 27 ∧ ∃ a b c : ℝ, a^2 + b^2 + c^2 = 9 ∧ a^3 + b^3 + c^3 = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_sphere_l702_70235


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l702_70260

theorem polynomial_irreducibility (a b c : ℤ) : 
  (0 < |c| ∧ |c| < |b| ∧ |b| < |a|) →
  (∀ x : ℤ, Irreducible (x * (x - a) * (x - b) * (x - c) + 1)) ↔
  (a ≠ 1 ∨ b ≠ 2 ∨ c ≠ 3) ∧ (a ≠ -1 ∨ b ≠ -2 ∨ c ≠ -3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l702_70260


namespace NUMINAMATH_CALUDE_find_number_l702_70232

theorem find_number : ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l702_70232


namespace NUMINAMATH_CALUDE_subset_sum_modulo_l702_70286

theorem subset_sum_modulo (N : ℕ) (A : Finset ℕ) :
  A.card = N →
  A ⊆ Finset.range (N^2) →
  ∃ (B : Finset ℕ), 
    B.card = N ∧ 
    B ⊆ Finset.range (N^2) ∧ 
    ((A.product B).image (λ (p : ℕ × ℕ) => (p.1 + p.2) % (N^2))).card ≥ N^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_modulo_l702_70286


namespace NUMINAMATH_CALUDE_range_of_a_l702_70240

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | |x - 1| > 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : (A a ∩ B = A a) ↔ (a ≤ -1 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l702_70240


namespace NUMINAMATH_CALUDE_multiple_problem_l702_70295

theorem multiple_problem (S L : ℝ) (h1 : S = 10) (h2 : S + L = 24) :
  ∃ M : ℝ, 7 * S = M * L ∧ M = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l702_70295


namespace NUMINAMATH_CALUDE_prime_plus_two_implies_divisible_by_six_l702_70208

theorem prime_plus_two_implies_divisible_by_six (p : ℤ) : 
  Prime p → p > 3 → Prime (p + 2) → (6 : ℤ) ∣ (p + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_plus_two_implies_divisible_by_six_l702_70208


namespace NUMINAMATH_CALUDE_intersection_chord_length_l702_70244

/-- The line L: 3x - y - 6 = 0 -/
def line_L (x y : ℝ) : Prop := 3 * x - y - 6 = 0

/-- The circle C: x^2 + y^2 - 2x - 4y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

/-- The length of the chord AB formed by the intersection of line L and circle C -/
noncomputable def chord_length : ℝ := Real.sqrt 10

theorem intersection_chord_length :
  chord_length = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l702_70244


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l702_70255

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : x*(x+2) + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l702_70255


namespace NUMINAMATH_CALUDE_expression_value_l702_70215

theorem expression_value : (100 - (3000 - 300)) + (3000 - (300 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l702_70215


namespace NUMINAMATH_CALUDE_proportion_property_l702_70223

theorem proportion_property (a b c d : ℝ) (h : a / b = c / d) : b * c - a * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_proportion_property_l702_70223


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l702_70288

theorem sum_of_coefficients_zero 
  (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ a₁₄ : ℝ) :
  (∀ x : ℝ, (1 + x - x^2)^3 * (1 - 2*x^2)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12 + a₁₃*x^13 + a₁₄*x^14) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ + a₁₄ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l702_70288


namespace NUMINAMATH_CALUDE_function_composition_sqrt2_l702_70216

theorem function_composition_sqrt2 (a : ℝ) (f : ℝ → ℝ) (h1 : 0 < a) :
  (∀ x, f x = a * x^2 - Real.sqrt 2) →
  f (f (Real.sqrt 2)) = -Real.sqrt 2 →
  a = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_function_composition_sqrt2_l702_70216


namespace NUMINAMATH_CALUDE_S_subset_T_l702_70217

-- Define set S
def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}

-- Define set T
def T : Set ℕ := {x | ∃ n : ℕ, x = 3*n}

-- Theorem stating S is a subset of T
theorem S_subset_T : S ⊆ T := by
  sorry

end NUMINAMATH_CALUDE_S_subset_T_l702_70217


namespace NUMINAMATH_CALUDE_brad_age_l702_70228

/-- Given the ages and relationships between Jaymee, Shara, and Brad, prove Brad's age -/
theorem brad_age (shara_age : ℕ) (jaymee_age : ℕ) (brad_age : ℕ) : 
  shara_age = 10 →
  jaymee_age = 2 * shara_age + 2 →
  brad_age = (shara_age + jaymee_age) / 2 - 3 →
  brad_age = 13 :=
by sorry

end NUMINAMATH_CALUDE_brad_age_l702_70228


namespace NUMINAMATH_CALUDE_factor_expression_l702_70229

theorem factor_expression (b : ℝ) : 26 * b^2 + 78 * b = 26 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l702_70229


namespace NUMINAMATH_CALUDE_girls_attending_event_l702_70214

theorem girls_attending_event (total_students : ℕ) (total_attendees : ℕ) 
  (girls : ℕ) (boys : ℕ) (h1 : total_students = 1500) 
  (h2 : total_attendees = 900) (h3 : girls + boys = total_students) 
  (h4 : (3 * girls) / 5 + (2 * boys) / 3 = total_attendees) : 
  (3 * girls) / 5 = 900 := by
  sorry

end NUMINAMATH_CALUDE_girls_attending_event_l702_70214


namespace NUMINAMATH_CALUDE_convex_shape_volume_is_half_l702_70252

/-- A cube with midlines of each face divided in a 1:3 ratio -/
structure DividedCube where
  /-- The volume of the original cube -/
  volume : ℝ
  /-- The ratio in which the midlines are divided -/
  divisionRatio : ℝ
  /-- Assumption that the division ratio is 1:3 -/
  ratio_is_one_three : divisionRatio = 1/3

/-- The volume of the convex shape formed by the points dividing the midlines -/
def convexShapeVolume (c : DividedCube) : ℝ := sorry

/-- Theorem stating that the volume of the convex shape is half the volume of the cube -/
theorem convex_shape_volume_is_half (c : DividedCube) : 
  convexShapeVolume c = c.volume / 2 := by sorry

end NUMINAMATH_CALUDE_convex_shape_volume_is_half_l702_70252


namespace NUMINAMATH_CALUDE_min_value_of_function_l702_70287

theorem min_value_of_function (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) ≥ 3 ∧ ∃ y > 1, y + 1 / (y - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l702_70287


namespace NUMINAMATH_CALUDE_num_triangles_in_polygon_l702_70219

/-- 
A polygon with n sides, where n is at least 3.
-/
structure Polygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- 
The number of triangles formed by non-intersecting diagonals in an n-gon.
-/
def num_triangles (p : Polygon) : ℕ := p.n - 2

/-- 
Theorem: The number of triangles formed by non-intersecting diagonals 
in an n-gon is equal to n-2.
-/
theorem num_triangles_in_polygon (p : Polygon) : 
  num_triangles p = p.n - 2 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_in_polygon_l702_70219


namespace NUMINAMATH_CALUDE_winner_third_difference_l702_70230

/-- Represents the vote count for each candidate in the election. -/
structure ElectionResult where
  total_votes : Nat
  num_candidates : Nat
  winner_votes : Nat
  second_votes : Nat
  third_votes : Nat
  fourth_votes : Nat

/-- Theorem stating the difference between the winner's votes and the third opponent's votes. -/
theorem winner_third_difference (e : ElectionResult) 
  (h1 : e.total_votes = 963)
  (h2 : e.num_candidates = 4)
  (h3 : e.winner_votes = 195)
  (h4 : e.second_votes = 142)
  (h5 : e.third_votes = 116)
  (h6 : e.fourth_votes = 90)
  : e.winner_votes - e.third_votes = 79 := by
  sorry


end NUMINAMATH_CALUDE_winner_third_difference_l702_70230


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l702_70294

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 ∧
    m % 11 = 10 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l702_70294


namespace NUMINAMATH_CALUDE_lap_time_calculation_l702_70233

/-- Represents the field and boy's running conditions -/
structure FieldConditions where
  side_length : ℝ
  normal_speed : ℝ
  sandy_length : ℝ
  sandy_speed_reduction : ℝ
  hurdle_count_low : ℕ
  hurdle_count_high : ℕ
  hurdle_time_low : ℝ
  hurdle_time_high : ℝ
  corner_slowdown : ℝ

/-- Calculates the total time to complete one lap around the field -/
def total_lap_time (conditions : FieldConditions) : ℝ :=
  sorry

/-- Theorem stating the total time to complete one lap -/
theorem lap_time_calculation (conditions : FieldConditions) 
  (h1 : conditions.side_length = 50)
  (h2 : conditions.normal_speed = 9 * 1000 / 3600)
  (h3 : conditions.sandy_length = 20)
  (h4 : conditions.sandy_speed_reduction = 0.25)
  (h5 : conditions.hurdle_count_low = 2)
  (h6 : conditions.hurdle_count_high = 2)
  (h7 : conditions.hurdle_time_low = 2)
  (h8 : conditions.hurdle_time_high = 3)
  (h9 : conditions.corner_slowdown = 2) :
  total_lap_time conditions = 138.68 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_calculation_l702_70233


namespace NUMINAMATH_CALUDE_mirror_area_l702_70254

/-- The area of a rectangular mirror with a frame -/
theorem mirror_area (overall_length overall_width frame_width : ℝ) 
  (h1 : overall_length = 100)
  (h2 : overall_width = 50)
  (h3 : frame_width = 8) : 
  (overall_length - 2 * frame_width) * (overall_width - 2 * frame_width) = 2856 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l702_70254


namespace NUMINAMATH_CALUDE_expression_values_l702_70238

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l702_70238


namespace NUMINAMATH_CALUDE_bus_passenger_count_l702_70273

/-- The number of passengers who got on at the first stop -/
def passengers_first_stop : ℕ := 16

theorem bus_passenger_count : passengers_first_stop = 16 :=
  let initial_passengers : ℕ := 50
  let final_passengers : ℕ := 49
  let passengers_off : ℕ := 22
  let passengers_on_other_stops : ℕ := 5
  have h : initial_passengers + passengers_first_stop - (passengers_off - passengers_on_other_stops) = final_passengers :=
    by sorry
  by sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l702_70273


namespace NUMINAMATH_CALUDE_complex_number_subtraction_l702_70242

theorem complex_number_subtraction (i : ℂ) (h : i * i = -1) :
  (7 - 3 * i) - 3 * (2 + 5 * i) = 1 - 18 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_number_subtraction_l702_70242


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l702_70298

theorem quadratic_equation_roots (α β : ℝ) (h1 : α + β = 5) (h2 : α * β = 6) :
  (α ^ 2 - 5 * α + 6 = 0) ∧ (β ^ 2 - 5 * β + 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l702_70298


namespace NUMINAMATH_CALUDE_exactly_one_false_l702_70209

theorem exactly_one_false :
  (∀ a b : ℝ, a ≥ b ∧ b > -1 → a / (1 + a) ≥ b / (1 + b)) ∧
  (∀ m n : ℕ+, m ≤ n → Real.sqrt (m * (n - m)) ≤ n / 2) ∧
  ¬(∀ a b x₁ y₁ : ℝ, x₁^2 + y₁^2 = 9 ∧ (a - x₁)^2 + (b - y₁)^2 = 1 →
    ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 9 ∧ (p.1 - a)^2 + (p.2 - b)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_false_l702_70209


namespace NUMINAMATH_CALUDE_polynomial_equality_l702_70276

theorem polynomial_equality (a b c d e : ℝ) :
  (∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = (2*x - 1)^4) →
  a + c = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l702_70276


namespace NUMINAMATH_CALUDE_ivan_remaining_money_l702_70239

def initial_amount : ℚ := 10
def cupcake_fraction : ℚ := 1/5
def milkshake_cost : ℚ := 5

theorem ivan_remaining_money :
  let cupcake_cost : ℚ := initial_amount * cupcake_fraction
  let remaining_after_cupcakes : ℚ := initial_amount - cupcake_cost
  let final_remaining : ℚ := remaining_after_cupcakes - milkshake_cost
  final_remaining = 3 := by sorry

end NUMINAMATH_CALUDE_ivan_remaining_money_l702_70239


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l702_70243

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 3 * r) = 300)
  (h2 : P * (1 + 8 * r) = 400) :
  r = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l702_70243


namespace NUMINAMATH_CALUDE_min_formula_l702_70293

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_formula_l702_70293


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l702_70289

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 a2 b2 : ℝ) : Prop := a1 / b1 = a2 / b2

/-- Definition of the first line l1 -/
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m

/-- Definition of the second line l2 -/
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

/-- Additional condition for m -/
def additional_condition (m : ℝ) : Prop := (3 + m) / 2 ≠ (5 - 3 * m) / 8

theorem parallel_lines_m_value :
  ∃ (m : ℝ), parallel_lines (3 + m) 4 2 (5 + m) ∧ 
             additional_condition m ∧
             m = -7 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l702_70289


namespace NUMINAMATH_CALUDE_ratio_expression_value_l702_70212

theorem ratio_expression_value (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/6) :
  (4*A - 3*B) / (5*C + 2*A) = 1/4 := by sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l702_70212
