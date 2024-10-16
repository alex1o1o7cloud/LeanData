import Mathlib

namespace NUMINAMATH_CALUDE_proposition_implication_l3242_324265

theorem proposition_implication (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (¬p ∧ q) ∨ (¬p ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l3242_324265


namespace NUMINAMATH_CALUDE_point_coordinates_l3242_324247

/-- A point in the second quadrant with given distances from axes has coordinates (-1, 2) -/
theorem point_coordinates (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  |P.2| = 2 →            -- Distance from P to x-axis is 2
  |P.1| = 1 →            -- Distance from P to y-axis is 1
  P = (-1, 2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3242_324247


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3242_324250

/-- Proves that the original stock price is 100 given the conditions --/
theorem stock_price_calculation (X : ℝ) : 
  X * 0.95 + 0.001 * (X * 0.95) = 95.2 → X = 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l3242_324250


namespace NUMINAMATH_CALUDE_range_of_m_l3242_324202

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - m) + y^2 / (m - 1) = 1 → m > 2) →
  (∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) →
  ((m > 2) ∨ (1 < m ∧ m < 3)) →
  ¬(1 < m ∧ m < 3) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3242_324202


namespace NUMINAMATH_CALUDE_race_head_start_l3242_324297

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (51 / 44) * Vb) :
  let H := L * (7 / 51)
  (L / Va) = ((L - H) / Vb) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l3242_324297


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3242_324216

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (h_radius : r = 5) :
  2 * π * r^2 + 2 * π * r * h = 170 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3242_324216


namespace NUMINAMATH_CALUDE_johns_age_l3242_324235

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80,
    prove that John is 25 years old. -/
theorem johns_age (john dad : ℕ) 
  (h1 : john = dad - 30)
  (h2 : john + dad = 80) : 
  john = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3242_324235


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3242_324275

/-- Represents an assignment statement --/
inductive AssignmentStatement
  | Constant : ℕ → String → AssignmentStatement
  | Variable : String → String → AssignmentStatement
  | Expression : String → String → AssignmentStatement
  | SelfAssignment : String → AssignmentStatement

/-- Checks if an assignment statement is valid --/
def isValidAssignment (stmt : AssignmentStatement) : Bool :=
  match stmt with
  | AssignmentStatement.Constant _ _ => false
  | AssignmentStatement.Variable _ _ => true
  | AssignmentStatement.Expression _ _ => false
  | AssignmentStatement.SelfAssignment _ => true

/-- The list of given assignment statements --/
def givenStatements : List AssignmentStatement :=
  [AssignmentStatement.Constant 2 "A",
   AssignmentStatement.Expression "x_+_y" "2",
   AssignmentStatement.Expression "A_-_B" "-2",
   AssignmentStatement.SelfAssignment "A"]

/-- Counts the number of valid assignment statements in a list --/
def countValidAssignments (stmts : List AssignmentStatement) : ℕ :=
  (stmts.filter isValidAssignment).length

theorem correct_assignment_count :
  countValidAssignments givenStatements = 1 := by sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3242_324275


namespace NUMINAMATH_CALUDE_factorization_proof_l3242_324271

theorem factorization_proof (a : ℝ) : a^2 - 4*a + 4 = (a - 2)^2 := by
  sorry

#check factorization_proof

end NUMINAMATH_CALUDE_factorization_proof_l3242_324271


namespace NUMINAMATH_CALUDE_reflection_y_axis_l3242_324239

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

theorem reflection_y_axis : 
  let A : ℝ × ℝ := (-3, 4)
  reflect_y A = (3, 4) := by sorry

end NUMINAMATH_CALUDE_reflection_y_axis_l3242_324239


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3242_324295

-- Define the fixed circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the moving circle M
def M (x y r : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = r^2

-- Define tangency condition
def isTangent (c₁ c₂ : ℝ → ℝ → Prop) (m : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y r, m x y r → (c₁ x y ∨ c₂ x y)

-- Main theorem
theorem trajectory_of_moving_circle :
  ∀ x y : ℝ, isTangent C₁ C₂ M → (x = 0 ∨ x^2 - y^2 / 3 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3242_324295


namespace NUMINAMATH_CALUDE_evaluate_expression_l3242_324248

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3242_324248


namespace NUMINAMATH_CALUDE_extremum_at_one_l3242_324224

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extremum_at_one (a b : ℝ) :
  f a b 1 = 10 ∧ (deriv (f a b)) 1 = 0 → a = -4 ∧ b = 11 :=
by sorry

end NUMINAMATH_CALUDE_extremum_at_one_l3242_324224


namespace NUMINAMATH_CALUDE_equation_solution_l3242_324280

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) - 5 / x = 0.5) ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3242_324280


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3242_324296

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 2
  let rectangle2_width : ℝ := 4
  let semicircle_diameter : ℝ := 2
  
  let volume_cylinder1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume_cylinder2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let volume_hemisphere : ℝ := (2/3) * π * (semicircle_diameter/2)^3
  
  let total_volume : ℝ := volume_cylinder1 + volume_cylinder2 + volume_hemisphere
  
  total_volume = 52 * (2/3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3242_324296


namespace NUMINAMATH_CALUDE_kantana_chocolates_l3242_324273

/-- The number of chocolates Kantana buys for herself and her sister every Saturday -/
def regular_chocolates : ℕ := 3

/-- The number of additional chocolates Kantana bought for Charlie on the last Saturday -/
def additional_chocolates : ℕ := 10

/-- The number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- The total number of chocolates Kantana bought for the month -/
def total_chocolates : ℕ := (saturdays_in_month - 1) * regular_chocolates + 
                            (regular_chocolates + additional_chocolates)

theorem kantana_chocolates : total_chocolates = 22 := by
  sorry

end NUMINAMATH_CALUDE_kantana_chocolates_l3242_324273


namespace NUMINAMATH_CALUDE_basketball_game_properties_l3242_324234

/-- Represents the score of player A in a single round -/
inductive Score
  | Minus : Score  -- A loses the round
  | Zero : Score   -- Tie in the round
  | Plus : Score   -- A wins the round

/-- Represents the number of rounds played -/
inductive Rounds
  | Two : Rounds
  | Three : Rounds
  | Four : Rounds

/-- The basketball shooting game between A and B -/
structure BasketballGame where
  a_accuracy : ℝ
  b_accuracy : ℝ
  max_rounds : ℕ
  win_difference : ℕ

/-- The probability distribution of the score in a single round -/
def score_distribution (game : BasketballGame) : Score → ℝ
  | Score.Minus => game.b_accuracy * (1 - game.a_accuracy)
  | Score.Zero => game.a_accuracy * game.b_accuracy + (1 - game.a_accuracy) * (1 - game.b_accuracy)
  | Score.Plus => game.a_accuracy * (1 - game.b_accuracy)

/-- The probability of a tie in the game -/
def tie_probability (game : BasketballGame) : ℝ := sorry

/-- The probability distribution of the number of rounds played -/
def rounds_distribution (game : BasketballGame) : Rounds → ℝ
  | Rounds.Two => sorry
  | Rounds.Three => sorry
  | Rounds.Four => sorry

/-- The expected number of rounds played -/
def expected_rounds (game : BasketballGame) : ℝ := sorry

theorem basketball_game_properties (game : BasketballGame) 
  (h1 : game.a_accuracy = 0.5)
  (h2 : game.b_accuracy = 0.6)
  (h3 : game.max_rounds = 4)
  (h4 : game.win_difference = 4) :
  score_distribution game Score.Minus = 0.3 ∧ 
  score_distribution game Score.Zero = 0.5 ∧
  score_distribution game Score.Plus = 0.2 ∧
  tie_probability game = 0.2569 ∧
  rounds_distribution game Rounds.Two = 0.13 ∧
  rounds_distribution game Rounds.Three = 0.13 ∧
  rounds_distribution game Rounds.Four = 0.74 ∧
  expected_rounds game = 3.61 := by sorry

end NUMINAMATH_CALUDE_basketball_game_properties_l3242_324234


namespace NUMINAMATH_CALUDE_largest_multiple_eleven_l3242_324288

theorem largest_multiple_eleven (n : ℤ) : 
  (n * 11 = -209) → 
  (-n * 11 > -210) ∧ 
  ∀ m : ℤ, (m > n) → (-m * 11 ≤ -210) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_eleven_l3242_324288


namespace NUMINAMATH_CALUDE_sum_of_abs_coeff_equals_729_l3242_324294

/-- Given a polynomial p(x) = a₆x⁶ + a₅x⁵ + ... + a₁x + a₀ that equals (2x-1)⁶,
    the sum of the absolute values of its coefficients is 729. -/
theorem sum_of_abs_coeff_equals_729 (a : Fin 7 → ℤ) : 
  (∀ x, (2*x - 1)^6 = a 6 * x^6 + a 5 * x^5 + a 4 * x^4 + a 3 * x^3 + a 2 * x^2 + a 1 * x + a 0) →
  (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 729) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_coeff_equals_729_l3242_324294


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l3242_324262

/-- Calculates the number of dots in a pentagon given its layer number -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else dots_in_pentagon (n - 1) + 5 * n

theorem fourth_pentagon_dots :
  dots_in_pentagon 3 = 31 := by
  sorry

#eval dots_in_pentagon 3

end NUMINAMATH_CALUDE_fourth_pentagon_dots_l3242_324262


namespace NUMINAMATH_CALUDE_complex_inequality_l3242_324206

theorem complex_inequality (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
  Complex.abs (z - w) ≥ (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ∧
  (Complex.abs (z - w) = (1/2 : ℝ) * (Complex.abs z + Complex.abs w) * Complex.abs ((z / Complex.abs z) - (w / Complex.abs w)) ↔
   (z / w).re < 0 ∨ Complex.abs z = Complex.abs w) :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_l3242_324206


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3242_324266

def line1 (a : ℝ) (x y : ℝ) : Prop := 2*x + a*y = 0

def line2 (a : ℝ) (x y : ℝ) : Prop := x - (a+1)*y = 0

def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, line1 a x₁ y₁ → line2 a x₂ y₂ → 
    (x₁ * x₂ + y₁ * y₂ = 0 ∨ (x₁ = 0 ∧ y₁ = 0) ∨ (x₂ = 0 ∧ y₂ = 0))

theorem perpendicular_lines (a : ℝ) : perpendicular a → (a = -2 ∨ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3242_324266


namespace NUMINAMATH_CALUDE_shaniqua_style_price_l3242_324230

/-- Proves that Shaniqua makes $25 for every style given the conditions -/
theorem shaniqua_style_price 
  (haircut_price : ℕ) 
  (total_earned : ℕ) 
  (num_haircuts : ℕ) 
  (num_styles : ℕ) 
  (h1 : haircut_price = 12)
  (h2 : total_earned = 221)
  (h3 : num_haircuts = 8)
  (h4 : num_styles = 5)
  (h5 : total_earned = num_haircuts * haircut_price + num_styles * (total_earned - num_haircuts * haircut_price) / num_styles) : 
  (total_earned - num_haircuts * haircut_price) / num_styles = 25 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_style_price_l3242_324230


namespace NUMINAMATH_CALUDE_parabola_single_intersection_l3242_324284

/-- A parabola defined by y = x^2 + 2x + c + 1 -/
def parabola (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c + 1

/-- A horizontal line defined by y = 1 -/
def line : ℝ → ℝ := λ _ => 1

/-- The condition for the parabola to intersect the line at only one point -/
def single_intersection (c : ℝ) : Prop :=
  ∃! x, parabola c x = line x

theorem parabola_single_intersection :
  ∀ c : ℝ, single_intersection c ↔ c = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_single_intersection_l3242_324284


namespace NUMINAMATH_CALUDE_olivia_friday_hours_l3242_324223

/-- Calculates the number of hours Olivia worked on Friday given her hourly rate, work hours on Monday and Wednesday, and total earnings for the week. -/
def fridayHours (hourlyRate : ℚ) (mondayHours wednesdayHours : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * (mondayHours + wednesdayHours)) / hourlyRate

/-- Proves that Olivia worked 6 hours on Friday given the specified conditions. -/
theorem olivia_friday_hours :
  fridayHours 9 4 3 117 = 6 := by
  sorry

end NUMINAMATH_CALUDE_olivia_friday_hours_l3242_324223


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3242_324291

open Real

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem f_max_min_on_interval :
  let a := 0
  let b := 2 * Real.pi / 3
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 1 ∧
    min = -(1/2) * Real.exp (2 * Real.pi / 3) - 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3242_324291


namespace NUMINAMATH_CALUDE_problem_solution_l3242_324233

-- Define the solution set for x(x-2) < 0
def solution_set := {x : ℝ | x * (x - 2) < 0}

-- Define the proposed incorrect solution set
def incorrect_set := {x : ℝ | x < 0 ∨ x > 2}

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem statement
theorem problem_solution :
  (solution_set ≠ incorrect_set) ∧
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3242_324233


namespace NUMINAMATH_CALUDE_tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l3242_324220

theorem tangent_sum_twelve_eighteen_equals_sqrt_three_over_three :
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l3242_324220


namespace NUMINAMATH_CALUDE_triangle_min_value_l3242_324253

theorem triangle_min_value (a b c : ℝ) (A : ℝ) (area : ℝ) : 
  A = Real.pi / 3 →
  area = Real.sqrt 3 →
  area = (1 / 2) * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  (∀ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) ≥ 5) ∧
  (∃ x y : ℝ, (4 * x ^ 2 + 4 * y ^ 2 - 3 * a ^ 2) / (x + y) = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_min_value_l3242_324253


namespace NUMINAMATH_CALUDE_broken_lines_count_l3242_324290

/-- The number of paths on a grid with 2n steps, n horizontal and n vertical -/
def grid_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of broken lines of length 2n on a grid
    with cell side length 1 and vertices at intersections is (C_{2n}^{n})^2 -/
theorem broken_lines_count (n : ℕ) :
  grid_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_broken_lines_count_l3242_324290


namespace NUMINAMATH_CALUDE_rose_price_theorem_l3242_324210

/-- The price of an individual rose -/
def individual_rose_price : ℝ := 7.5

/-- The cost of one dozen roses -/
def dozen_price : ℝ := 36

/-- The cost of two dozen roses -/
def two_dozen_price : ℝ := 50

/-- The maximum number of roses that can be purchased for $680 -/
def max_roses : ℕ := 316

/-- The total budget available -/
def total_budget : ℝ := 680

theorem rose_price_theorem :
  (dozen_price = 12 * individual_rose_price) ∧
  (two_dozen_price = 24 * individual_rose_price) ∧
  (∀ n : ℕ, n * individual_rose_price ≤ total_budget → n ≤ max_roses) ∧
  (max_roses * individual_rose_price ≤ total_budget) :=
by sorry

end NUMINAMATH_CALUDE_rose_price_theorem_l3242_324210


namespace NUMINAMATH_CALUDE_no_perfect_square_with_only_six_and_zero_l3242_324254

theorem no_perfect_square_with_only_six_and_zero : 
  ¬ ∃ (n : ℕ), (∃ (m : ℕ), n = m^2) ∧ 
  (∀ (d : ℕ), d ∈ n.digits 10 → d = 6 ∨ d = 0) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_square_with_only_six_and_zero_l3242_324254


namespace NUMINAMATH_CALUDE_monotonic_f_implies_m_range_inequality_implies_a_range_l3242_324207

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + m * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a * x - 3

theorem monotonic_f_implies_m_range (m : ℝ) :
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) → m ≤ -1 := by sorry

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x, x > 0 → 2 * (f 0 x) ≥ g a x) → a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_m_range_inequality_implies_a_range_l3242_324207


namespace NUMINAMATH_CALUDE_axis_of_symmetry_cos_minus_sin_l3242_324229

/-- The axis of symmetry for the function y = cos(2x) - sin(2x) is x = -π/8 -/
theorem axis_of_symmetry_cos_minus_sin (x : ℝ) : 
  (∀ y, y = Real.cos (2 * x) - Real.sin (2 * x)) → 
  (∃ k : ℤ, x = (k : ℝ) * π / 2 - π / 8) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_cos_minus_sin_l3242_324229


namespace NUMINAMATH_CALUDE_brick_width_is_10cm_l3242_324282

/-- Proves that the width of a brick is 10 cm given the specified conditions -/
theorem brick_width_is_10cm 
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_length : ℝ) (brick_height : ℝ)
  (num_bricks : ℕ)
  (h_wall_length : wall_length = 29)
  (h_wall_width : wall_width = 2)
  (h_wall_height : wall_height = 0.75)
  (h_brick_length : brick_length = 20)
  (h_brick_height : brick_height = 7.5)
  (h_num_bricks : num_bricks = 29000)
  : ∃ (brick_width : ℝ), 
    wall_length * wall_width * wall_height * 1000000 = 
    num_bricks * brick_length * brick_width * brick_height ∧ 
    brick_width = 10 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_10cm_l3242_324282


namespace NUMINAMATH_CALUDE_maria_assembly_time_l3242_324292

/-- Represents the time taken to assemble furniture items -/
structure AssemblyTime where
  chairs : Nat
  tables : Nat
  bookshelf : Nat
  tv_stand : Nat

/-- Calculates the total assembly time for all furniture items -/
def total_assembly_time (time : AssemblyTime) (num_chairs num_tables : Nat) : Nat :=
  num_chairs * time.chairs + num_tables * time.tables + time.bookshelf + time.tv_stand

/-- Theorem: The total assembly time for Maria's furniture is 100 minutes -/
theorem maria_assembly_time :
  let time : AssemblyTime := { chairs := 8, tables := 12, bookshelf := 25, tv_stand := 35 }
  total_assembly_time time 2 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_maria_assembly_time_l3242_324292


namespace NUMINAMATH_CALUDE_sum_representation_l3242_324240

def sum_of_complex_exponentials (z₁ z₂ : ℂ) : ℂ := z₁ + z₂

theorem sum_representation (z₁ z₂ : ℂ) :
  let sum := sum_of_complex_exponentials z₁ z₂
  let r := 30 * Real.cos (π / 10)
  let θ := 9 * π / 20
  z₁ = 15 * Complex.exp (Complex.I * π / 5) ∧
  z₂ = 15 * Complex.exp (Complex.I * 7 * π / 10) →
  sum = r * Complex.exp (Complex.I * θ) :=
by sorry

end NUMINAMATH_CALUDE_sum_representation_l3242_324240


namespace NUMINAMATH_CALUDE_impossibility_of_transformation_l3242_324218

def operation (a b : ℤ) : ℤ × ℤ := (5*a - 2*b, 3*a - 4*b)

def initial_set : Set ℤ := {n | 1 ≤ n ∧ n ≤ 2018}

def target_sequence : Set ℤ := {n | ∃ k, 1 ≤ k ∧ k ≤ 2018 ∧ n = 3*k}

theorem impossibility_of_transformation :
  ∀ (S : Set ℤ), S = initial_set →
  ¬∃ (n : ℕ), ∃ (S' : Set ℤ),
    (∀ k ≤ n, ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧
      S' = (S \ {a, b}) ∪ {(operation a b).1, (operation a b).2}) →
    target_sequence ⊆ S' :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_transformation_l3242_324218


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3242_324270

theorem angle_sum_is_pi_over_two (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (eq1 : 3 * Real.sin a ^ 2 + 2 * Real.sin b ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * a) - 2 * Real.sin (2 * b) = 0) :
  a + 2 * b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3242_324270


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l3242_324289

theorem least_n_for_inequality : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 8 → k ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 8) ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l3242_324289


namespace NUMINAMATH_CALUDE_twelve_point_polygons_l3242_324298

/-- The number of distinct convex polygons with 3 or more sides that can be formed from n points on a circle -/
def convex_polygons (n : ℕ) : ℕ :=
  2^n - 1 - n - (n.choose 2)

theorem twelve_point_polygons :
  convex_polygons 12 = 4017 := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_polygons_l3242_324298


namespace NUMINAMATH_CALUDE_danielle_apartment_rooms_l3242_324236

theorem danielle_apartment_rooms : 
  ∀ (heidi grant danielle : ℕ),
  heidi = 3 * danielle →
  grant * 9 = heidi →
  grant = 2 →
  danielle = 6 := by
sorry

end NUMINAMATH_CALUDE_danielle_apartment_rooms_l3242_324236


namespace NUMINAMATH_CALUDE_complex_counterexample_l3242_324268

theorem complex_counterexample : ∃ z₁ z₂ : ℂ, (Complex.abs z₁ = Complex.abs z₂) ∧ (z₁^2 ≠ z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_complex_counterexample_l3242_324268


namespace NUMINAMATH_CALUDE_remainder_problem_l3242_324251

theorem remainder_problem (N : ℤ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3242_324251


namespace NUMINAMATH_CALUDE_largest_power_of_18_dividing_30_factorial_l3242_324242

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_of_18_dividing_30_factorial :
  (∃ n : ℕ, 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) →
  (∃ n : ℕ, n = 7 ∧ 18^n ∣ factorial 30 ∧ ∀ m : ℕ, m > n → ¬(18^m ∣ factorial 30)) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_18_dividing_30_factorial_l3242_324242


namespace NUMINAMATH_CALUDE_equation_solutions_l3242_324222

theorem equation_solutions :
  (∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5) ∧
  (∃ x : ℝ, 25 * x^2 + 3 = 12 ∧ (x = 3/5 ∨ x = -3/5)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3242_324222


namespace NUMINAMATH_CALUDE_policemen_cover_all_streets_l3242_324278

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) := 
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the chosen intersections for policemen
def chosenIntersections : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem: The chosen intersections cover all streets
theorem policemen_cover_all_streets : 
  ∀ street ∈ allStreets, ∃ intersection ∈ chosenIntersections, intersection ∈ street :=
sorry

end NUMINAMATH_CALUDE_policemen_cover_all_streets_l3242_324278


namespace NUMINAMATH_CALUDE_intersection_area_is_sqrt_80_l3242_324261

/-- Represents a square pyramid -/
structure SquarePyramid where
  base_side : ℝ
  edge_length : ℝ

/-- Represents a plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : SquarePyramid
  -- The plane passes through midpoints of one lateral edge and two base edges

/-- The area of intersection between the plane and the pyramid -/
noncomputable def intersection_area (plane : IntersectingPlane) : ℝ := sorry

theorem intersection_area_is_sqrt_80 (plane : IntersectingPlane) :
  plane.pyramid.base_side = 4 →
  plane.pyramid.edge_length = 4 →
  intersection_area plane = Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_is_sqrt_80_l3242_324261


namespace NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3242_324244

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem even_function_derivative_is_odd 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x) -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_is_odd_l3242_324244


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3242_324214

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}

-- Define the set N
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^x ∧ 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3242_324214


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3242_324219

theorem min_value_of_sum_of_squares (x y : ℝ) : 
  (x + 5)^2 + (y - 12)^2 = 14^2 → ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≤ a^2 + b^2 := by
  sorry

#check min_value_of_sum_of_squares

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3242_324219


namespace NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3242_324217

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define a property for symmetry
def is_symmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.IsoscelesTrapezoid => True
  | Shape.Parallelogram => False

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ∃ (s : Shape), ¬(is_symmetrical s) ∧ s = Shape.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3242_324217


namespace NUMINAMATH_CALUDE_polynomial_primes_theorem_l3242_324208

def is_valid_polynomial (Q : ℤ → ℤ) : Prop :=
  ∃ (a b c : ℤ), ∀ x, Q x = a * x^2 + b * x + c

def satisfies_condition (Q : ℤ → ℤ) : Prop :=
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    |Q p₁| = 11 ∧ |Q p₂| = 11 ∧ |Q p₃| = 11

def is_solution (Q : ℤ → ℤ) : Prop :=
  (∀ x, Q x = 11) ∨
  (∀ x, Q x = x^2 - 13*x + 11) ∨
  (∀ x, Q x = 2*x^2 - 32*x + 67) ∨
  (∀ x, Q x = 11*x^2 - 77*x + 121)

theorem polynomial_primes_theorem :
  ∀ Q : ℤ → ℤ, is_valid_polynomial Q → satisfies_condition Q → is_solution Q :=
sorry

end NUMINAMATH_CALUDE_polynomial_primes_theorem_l3242_324208


namespace NUMINAMATH_CALUDE_distance_product_range_l3242_324237

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 8

-- Define a point P on C₁
structure PointOnC₁ where
  x : ℝ
  y : ℝ
  on_C₁ : C₁ x y

-- Define the line l with 45° inclination passing through P
def line_l (P : PointOnC₁) (x y : ℝ) : Prop :=
  y - P.y = (x - P.x)

-- Define the intersection points Q and R
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_C₂ : C₂ x y
  on_l : line_l P x y

-- Define the product of distances |PQ| · |PR|
def distance_product (P : PointOnC₁) (Q R : IntersectionPoint) : ℝ :=
  ((Q.x - P.x)^2 + (Q.y - P.y)^2) * ((R.x - P.x)^2 + (R.y - P.y)^2)

-- State the theorem
theorem distance_product_range (P : PointOnC₁) (Q R : IntersectionPoint) 
  (h_distinct : Q ≠ R) :
  ∃ (d : ℝ), distance_product P Q R = d ∧ (d ∈ Set.Icc 4 8 ∨ d ∈ Set.Ioo 8 200) :=
sorry

end NUMINAMATH_CALUDE_distance_product_range_l3242_324237


namespace NUMINAMATH_CALUDE_hillary_activities_lcm_l3242_324274

theorem hillary_activities_lcm : Nat.lcm (Nat.lcm 6 4) 16 = 48 := by
  sorry

end NUMINAMATH_CALUDE_hillary_activities_lcm_l3242_324274


namespace NUMINAMATH_CALUDE_bucket_capacity_l3242_324276

theorem bucket_capacity (tank_capacity : ℝ) (first_scenario_buckets : ℕ) (second_scenario_buckets : ℕ) (second_scenario_capacity : ℝ) :
  first_scenario_buckets = 30 →
  second_scenario_buckets = 45 →
  second_scenario_capacity = 9 →
  tank_capacity = first_scenario_buckets * (tank_capacity / first_scenario_buckets) →
  tank_capacity = second_scenario_buckets * second_scenario_capacity →
  tank_capacity / first_scenario_buckets = 13.5 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_l3242_324276


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3242_324264

-- Define the displacement function
def h (t : ℝ) : ℝ := 14 * t - t^2

-- Define the instantaneous velocity function (derivative of h)
def v (t : ℝ) : ℝ := 14 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3242_324264


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3242_324258

/-- Given two functions f and g defined as:
    f(x) = -2|x-a| + b
    g(x) = 2|x-c| + d
    If f(5) = g(5) = 10 and f(11) = g(11) = 6,
    then a + c = 16 -/
theorem intersection_implies_sum (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d → x = 5 ∨ x = 11) →
  -2 * |5 - a| + b = 10 →
  2 * |5 - c| + d = 10 →
  -2 * |11 - a| + b = 6 →
  2 * |11 - c| + d = 6 →
  a + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3242_324258


namespace NUMINAMATH_CALUDE_average_weight_decrease_l3242_324255

theorem average_weight_decrease (initial_count : ℕ) (initial_avg : ℝ) (new_weight : ℝ) : 
  initial_count = 20 → 
  initial_avg = 57 → 
  new_weight = 48 → 
  let new_avg := (initial_count * initial_avg + new_weight) / (initial_count + 1)
  initial_avg - new_avg = 0.43 := by sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l3242_324255


namespace NUMINAMATH_CALUDE_intersection_characterization_l3242_324286

-- Define set A
def A : Set ℝ := {x | Real.log (2 * x) < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 3 * x + 2}

-- Define the intersection of A and B
def A_intersect_B : Set (ℝ × ℝ) := {p | p.1 ∈ A ∧ p.2 ∈ B ∧ p.2 = 3 * p.1 + 2}

-- Theorem statement
theorem intersection_characterization : 
  A_intersect_B = {p : ℝ × ℝ | 2 < p.2 ∧ p.2 < 14} := by
  sorry

end NUMINAMATH_CALUDE_intersection_characterization_l3242_324286


namespace NUMINAMATH_CALUDE_x_range_l3242_324249

theorem x_range (x : ℝ) (h1 : 1/x < 4) (h2 : 1/x > -6) (h3 : x < 0) :
  -1/6 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3242_324249


namespace NUMINAMATH_CALUDE_car_sale_percentage_l3242_324267

theorem car_sale_percentage (P x : ℝ) : 
  P - 2500 = 30000 →
  x / 100 * P = 30000 - 4000 →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_car_sale_percentage_l3242_324267


namespace NUMINAMATH_CALUDE_max_intersection_points_ellipse_three_lines_l3242_324241

/-- Represents a line in a 2D plane -/
structure Line :=
  (a b c : ℝ)

/-- Represents an ellipse in a 2D plane -/
structure Ellipse :=
  (a b c d e f : ℝ)

/-- Counts the maximum number of intersection points between an ellipse and a line -/
def maxIntersectionPointsEllipseLine : ℕ := 2

/-- Counts the maximum number of intersection points between two distinct lines -/
def maxIntersectionPointsTwoLines : ℕ := 1

/-- The number of distinct pairs of lines given 3 lines -/
def numLinePairs : ℕ := 3

/-- The number of lines -/
def numLines : ℕ := 3

theorem max_intersection_points_ellipse_three_lines :
  ∀ (e : Ellipse) (l₁ l₂ l₃ : Line),
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ →
    (maxIntersectionPointsEllipseLine * numLines) + 
    (maxIntersectionPointsTwoLines * numLinePairs) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_ellipse_three_lines_l3242_324241


namespace NUMINAMATH_CALUDE_digit_product_over_21_l3242_324203

theorem digit_product_over_21 (c d : ℕ) : 
  (c < 10 ∧ d < 10) → -- c and d are base-10 digits
  (7 * 7 * 7 + 6 * 7 + 5 = 400 + 10 * c + d) → -- 765₇ = 4cd₁₀
  (c * d : ℚ) / 21 = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_over_21_l3242_324203


namespace NUMINAMATH_CALUDE_vector_subtraction_l3242_324283

/-- Given vectors a and b in ℝ², prove that a - 2b = (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3242_324283


namespace NUMINAMATH_CALUDE_sector_max_area_l3242_324272

/-- Given a sector with perimeter 16, its maximum area is 16. -/
theorem sector_max_area (r θ : ℝ) (h : 2 * r + r * θ = 16) : 
  ∀ r' θ' : ℝ, 2 * r' + r' * θ' = 16 → (1/2) * r' * r' * θ' ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3242_324272


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l3242_324285

def num_pigs : ℕ := 5
def num_rabbits : ℕ := 3
def num_dogs : ℕ := 2
def num_chickens : ℕ := 6

def total_animals : ℕ := num_pigs + num_rabbits + num_dogs + num_chickens

def num_animal_types : ℕ := 4

theorem animal_arrangement_count :
  (Nat.factorial num_animal_types) *
  (Nat.factorial num_pigs) *
  (Nat.factorial num_rabbits) *
  (Nat.factorial num_dogs) *
  (Nat.factorial num_chickens) = 12441600 :=
by sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l3242_324285


namespace NUMINAMATH_CALUDE_not_always_possible_to_make_all_white_l3242_324243

/-- Represents a smaller equilateral triangle within the larger triangle -/
structure SmallTriangle where
  color : Bool  -- true for white, false for black

/-- Represents the entire configuration of the divided equilateral triangle -/
structure TriangleConfiguration where
  smallTriangles : List SmallTriangle
  numRows : Nat  -- number of rows in the triangle

/-- Represents a repainting operation -/
def repaint (config : TriangleConfiguration) (lineIndex : Nat) : TriangleConfiguration :=
  sorry

/-- Checks if all small triangles in the configuration are white -/
def allWhite (config : TriangleConfiguration) : Bool :=
  sorry

/-- Theorem stating that there exists a configuration where it's impossible to make all triangles white -/
theorem not_always_possible_to_make_all_white :
  ∃ (initialConfig : TriangleConfiguration),
    ∀ (repaintSequence : List Nat),
      let finalConfig := repaintSequence.foldl repaint initialConfig
      ¬(allWhite finalConfig) :=
sorry

end NUMINAMATH_CALUDE_not_always_possible_to_make_all_white_l3242_324243


namespace NUMINAMATH_CALUDE_early_winner_emerges_l3242_324212

/-- The number of participants in the tournament -/
def n : ℕ := 10

/-- The number of matches each participant plays -/
def matches_per_participant : ℕ := n - 1

/-- The total number of matches in the tournament -/
def total_matches : ℕ := n * matches_per_participant / 2

/-- The number of matches per round -/
def matches_per_round : ℕ := n / 2

/-- The maximum points a participant can score in one round -/
def max_points_per_round : ℚ := 1

/-- The minimum number of rounds required for an early winner to emerge -/
def min_rounds_for_winner : ℕ := 7

theorem early_winner_emerges (
  winner_points : ℚ → ℚ → Prop) 
  (other_max_points : ℚ → ℚ → Prop) : 
  (∀ r : ℕ, r < min_rounds_for_winner → 
    ¬(winner_points r > other_max_points r)) ∧
  (winner_points min_rounds_for_winner > 
    other_max_points min_rounds_for_winner) := by
  sorry

end NUMINAMATH_CALUDE_early_winner_emerges_l3242_324212


namespace NUMINAMATH_CALUDE_family_size_l3242_324245

/-- Represents the number of slices per tomato -/
def slices_per_tomato : ℕ := 8

/-- Represents the number of slices needed for one person's meal -/
def slices_per_meal : ℕ := 20

/-- Represents the number of tomatoes Thelma needs -/
def total_tomatoes : ℕ := 20

/-- Theorem: Given the conditions, the family has 8 people -/
theorem family_size :
  (total_tomatoes * slices_per_tomato) / slices_per_meal = 8 := by
  sorry

end NUMINAMATH_CALUDE_family_size_l3242_324245


namespace NUMINAMATH_CALUDE_problem_solution_l3242_324225

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + a*x^2 - 1

-- Define the function g
def g (b : ℝ) (x : ℝ) : ℝ := b*x^2 - 1

theorem problem_solution :
  -- Part 1
  (∀ x y, (0 ≤ x ∧ x < y ∧ y ≤ 1) → f 4 x < f 4 y) ∧
  (∀ x y, (1 ≤ x ∧ x < y ∧ y ≤ 2) → f 4 x > f 4 y) →
  -- Part 2
  (∃ b₁ b₂, b₁ ≠ b₂ ∧
    (∀ b, (∃! x₁ x₂, x₁ ≠ x₂ ∧ f 4 x₁ = g b x₁ ∧ f 4 x₂ = g b x₂) ↔ (b = b₁ ∨ b = b₂))) ∧
  -- Part 3
  (∀ m n, m ∈ Set.Icc (-6 : ℝ) (-2) →
    ((∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f 4 x ≤ m*x^3 + 2*x^2 - n) →
      n ∈ Set.Iic (-4 : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3242_324225


namespace NUMINAMATH_CALUDE_three_not_in_range_of_g_l3242_324226

/-- The function g(x) defined as x^2 - bx + c -/
def g (b c x : ℝ) : ℝ := x^2 - b*x + c

/-- Theorem stating the conditions for 3 to not be in the range of g(x) -/
theorem three_not_in_range_of_g (b c : ℝ) :
  (∀ x, g b c x ≠ 3) ↔ (c ≥ 3 ∧ b > -Real.sqrt (4*c - 12) ∧ b < Real.sqrt (4*c - 12)) :=
sorry

end NUMINAMATH_CALUDE_three_not_in_range_of_g_l3242_324226


namespace NUMINAMATH_CALUDE_coin_value_increase_l3242_324277

def coins_bought : ℕ := 20
def initial_price : ℚ := 15
def coins_sold : ℕ := 12

def original_investment : ℚ := coins_bought * initial_price
def selling_price : ℚ := original_investment

theorem coin_value_increase :
  (selling_price / coins_sold - initial_price) / initial_price = 2/3 :=
sorry

end NUMINAMATH_CALUDE_coin_value_increase_l3242_324277


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3242_324246

/-- The minimum distance from the origin to the line 2x - y + 1 = 0 is √5/5 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 2 * x - y + 1 = 0}
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧ 
    (∀ (P : ℝ × ℝ), P ∈ line → d ≤ Real.sqrt (P.1^2 + P.2^2)) ∧
    (∃ (P : ℝ × ℝ), P ∈ line ∧ d = Real.sqrt (P.1^2 + P.2^2)) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l3242_324246


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3242_324279

def f (x : ℝ) : ℝ := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3242_324279


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3242_324252

theorem trigonometric_identity (α β γ n : ℝ) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) :
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3242_324252


namespace NUMINAMATH_CALUDE_three_eighths_percent_of_240_l3242_324263

theorem three_eighths_percent_of_240 : (3 / 8 / 100) * 240 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_percent_of_240_l3242_324263


namespace NUMINAMATH_CALUDE_guard_max_demand_l3242_324257

/-- Represents the outcome of the outsider's decision -/
inductive Outcome
| Pay
| Refuse

/-- Represents the guard's demand and the outsider's decision -/
structure Scenario where
  guardDemand : ℕ
  outsiderDecision : Outcome

/-- Calculates the outsider's loss based on the scenario -/
def outsiderLoss (s : Scenario) : ℤ :=
  match s.outsiderDecision with
  | Outcome.Pay => s.guardDemand - 100
  | Outcome.Refuse => 100

/-- Determines if the outsider will pay based on personal benefit -/
def willPay (guardDemand : ℕ) : Prop :=
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Pay } <
  outsiderLoss { guardDemand := guardDemand, outsiderDecision := Outcome.Refuse }

/-- The maximum number of coins the guard can demand -/
def maxGuardDemand : ℕ := 199

theorem guard_max_demand :
  (∀ n : ℕ, n ≤ maxGuardDemand → willPay n) ∧
  (∀ n : ℕ, n > maxGuardDemand → ¬willPay n) :=
sorry

end NUMINAMATH_CALUDE_guard_max_demand_l3242_324257


namespace NUMINAMATH_CALUDE_inequality_theorems_l3242_324211

theorem inequality_theorems :
  (∀ a b : ℝ, a > b → (1 / a < 1 / b → a * b > 0)) ∧
  (∀ a b : ℝ, a > b → (1 / a > 1 / b → a > 0 ∧ 0 > b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorems_l3242_324211


namespace NUMINAMATH_CALUDE_f_at_two_l3242_324269

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f' is the derivative of f
axiom is_derivative : ∀ x, deriv f x = f' x

-- f(x) = 2xf'(2) + ln(x-1)
axiom f_def : ∀ x, f x = 2 * x * (f' 2) + Real.log (x - 1)

theorem f_at_two : f 2 = -4 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l3242_324269


namespace NUMINAMATH_CALUDE_joel_stuffed_animals_l3242_324281

/-- The number of stuffed animals Joel collected -/
def stuffed_animals : ℕ := 18

/-- The number of action figures Joel collected -/
def action_figures : ℕ := 42

/-- The number of board games Joel collected -/
def board_games : ℕ := 2

/-- The number of puzzles Joel collected -/
def puzzles : ℕ := 13

/-- The total number of toys Joel donated -/
def total_toys : ℕ := 108

/-- The number of toys that were Joel's own -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister gave him -/
def sisters_toys : ℕ := (joels_toys / 2)

theorem joel_stuffed_animals :
  stuffed_animals + action_figures + board_games + puzzles + sisters_toys + joels_toys = total_toys :=
by sorry

end NUMINAMATH_CALUDE_joel_stuffed_animals_l3242_324281


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l3242_324215

-- Define the concept of opposite
def opposite (x : Int) : Int := -x

-- State the theorem
theorem opposite_of_negative_eight :
  opposite (-8) = 8 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l3242_324215


namespace NUMINAMATH_CALUDE_event_properties_l3242_324256

-- Define the types of events
inductive Event
| Train
| Shooting

-- Define the type of outcomes
inductive Outcome
| Success
| Failure

-- Define a function to get the number of trials for each event
def num_trials (e : Event) : ℕ :=
  match e with
  | Event.Train => 3
  | Event.Shooting => 2

-- Define a function to get the possible outcomes for each event
def possible_outcomes (e : Event) : List Outcome :=
  [Outcome.Success, Outcome.Failure]

-- Theorem statement
theorem event_properties :
  (∀ e : Event, num_trials e > 0) ∧
  (∀ e : Event, possible_outcomes e = [Outcome.Success, Outcome.Failure]) :=
by sorry

end NUMINAMATH_CALUDE_event_properties_l3242_324256


namespace NUMINAMATH_CALUDE_total_lives_calculation_l3242_324221

/-- Given 7 initial friends, 2 additional players, and 7 lives per player,
    the total number of lives for all players is 63. -/
theorem total_lives_calculation (initial_friends : ℕ) (additional_players : ℕ) (lives_per_player : ℕ)
    (h1 : initial_friends = 7)
    (h2 : additional_players = 2)
    (h3 : lives_per_player = 7) :
    (initial_friends + additional_players) * lives_per_player = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l3242_324221


namespace NUMINAMATH_CALUDE_mikis_sandcastle_height_l3242_324287

/-- The height of Miki's sister's sandcastle in feet -/
def sisters_height : ℝ := 0.5

/-- The difference in height between Miki's and her sister's sandcastles in feet -/
def height_difference : ℝ := 0.33

/-- The height of Miki's sandcastle in feet -/
def mikis_height : ℝ := sisters_height + height_difference

theorem mikis_sandcastle_height : mikis_height = 0.83 := by
  sorry

end NUMINAMATH_CALUDE_mikis_sandcastle_height_l3242_324287


namespace NUMINAMATH_CALUDE_greatest_divisor_l3242_324231

theorem greatest_divisor (G : ℕ) : G = 127 ↔ 
  G > 0 ∧ 
  (∀ n : ℕ, n > G → ¬(1657 % n = 6 ∧ 2037 % n = 5)) ∧
  1657 % G = 6 ∧ 
  2037 % G = 5 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_l3242_324231


namespace NUMINAMATH_CALUDE_toy_sales_earnings_difference_l3242_324293

theorem toy_sales_earnings_difference :
  let bert_initial_price : ℝ := 18
  let bert_initial_quantity : ℕ := 10
  let bert_discount_percentage : ℝ := 0.15
  let bert_discounted_quantity : ℕ := 3

  let tory_initial_price : ℝ := 20
  let tory_initial_quantity : ℕ := 15
  let tory_discount_percentage : ℝ := 0.10
  let tory_discounted_quantity : ℕ := 7

  let tax_rate : ℝ := 0.05

  let bert_earnings : ℝ := 
    (bert_initial_price * bert_initial_quantity - 
     bert_discount_percentage * bert_initial_price * bert_discounted_quantity) * 
    (1 + tax_rate)

  let tory_earnings : ℝ := 
    (tory_initial_price * tory_initial_quantity - 
     tory_discount_percentage * tory_initial_price * tory_discounted_quantity) * 
    (1 + tax_rate)

  tory_earnings - bert_earnings = 119.805 :=
by sorry

end NUMINAMATH_CALUDE_toy_sales_earnings_difference_l3242_324293


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3242_324232

theorem arithmetic_computation : -10 * 5 - (-8 * -4) + (-12 * -6) + 2 * 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3242_324232


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l3242_324299

def f (x : ℝ) := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l3242_324299


namespace NUMINAMATH_CALUDE_field_trip_difference_l3242_324238

/-- Given the number of vans, buses, people per van, and people per bus,
    prove that the difference between the number of people traveling by bus
    and the number of people traveling by van is 108.0 --/
theorem field_trip_difference (num_vans : ℝ) (num_buses : ℝ) 
                               (people_per_van : ℝ) (people_per_bus : ℝ) :
  num_vans = 6.0 →
  num_buses = 8.0 →
  people_per_van = 6.0 →
  people_per_bus = 18.0 →
  num_buses * people_per_bus - num_vans * people_per_van = 108.0 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_difference_l3242_324238


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3242_324201

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 56 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3242_324201


namespace NUMINAMATH_CALUDE_election_votes_l3242_324213

theorem election_votes (total_votes : ℕ) 
  (h1 : (70 : ℚ) / 100 * total_votes - (30 : ℚ) / 100 * total_votes = 182) : 
  total_votes = 455 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3242_324213


namespace NUMINAMATH_CALUDE_calculate_expression_l3242_324200

theorem calculate_expression (y : ℝ) (h : y = 3) : y + y * (y^y)^2 = 2190 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3242_324200


namespace NUMINAMATH_CALUDE_abs_diff_of_abs_l3242_324205

theorem abs_diff_of_abs : ∀ a b : ℝ, 
  (abs a = 3 ∧ abs b = 5) → abs (abs (a + b) - abs (a - b)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_of_abs_l3242_324205


namespace NUMINAMATH_CALUDE_henrikh_walk_time_per_block_l3242_324259

/-- The time it takes Henrikh to walk one block to work -/
def walkTimePerBlock : ℝ := 60

/-- The number of blocks from Henrikh's home to his office -/
def distanceInBlocks : ℕ := 12

/-- The time it takes Henrikh to ride his bicycle for one block -/
def bikeTimePerBlock : ℝ := 20

/-- The additional time it takes to walk compared to riding a bicycle for the entire distance -/
def additionalWalkTime : ℝ := 8 * 60  -- 8 minutes in seconds

theorem henrikh_walk_time_per_block :
  walkTimePerBlock * distanceInBlocks = 
  bikeTimePerBlock * distanceInBlocks + additionalWalkTime :=
by sorry

end NUMINAMATH_CALUDE_henrikh_walk_time_per_block_l3242_324259


namespace NUMINAMATH_CALUDE_radio_show_music_commercial_ratio_l3242_324228

/-- Represents a segment of a radio show -/
structure Segment where
  total_time : ℕ
  commercial_time : ℕ

/-- Calculates the greatest common divisor of two natural numbers -/
def gcd (a b : ℕ) : ℕ := sorry

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (a b : ℕ) : ℕ × ℕ := sorry

theorem radio_show_music_commercial_ratio 
  (segment1 : Segment)
  (segment2 : Segment)
  (segment3 : Segment)
  (h1 : segment1.total_time = 56 ∧ segment1.commercial_time = 22)
  (h2 : segment2.total_time = 84 ∧ segment2.commercial_time = 28)
  (h3 : segment3.total_time = 128 ∧ segment3.commercial_time = 34) :
  simplify_ratio 
    ((segment1.total_time - segment1.commercial_time) + 
     (segment2.total_time - segment2.commercial_time) + 
     (segment3.total_time - segment3.commercial_time))
    (segment1.commercial_time + segment2.commercial_time + segment3.commercial_time) = (46, 21) := by
  sorry

end NUMINAMATH_CALUDE_radio_show_music_commercial_ratio_l3242_324228


namespace NUMINAMATH_CALUDE_kelly_glue_bottles_l3242_324209

theorem kelly_glue_bottles (students : ℕ) (paper_per_student : ℕ) (added_paper : ℕ) (final_supplies : ℕ) :
  students = 8 →
  paper_per_student = 3 →
  added_paper = 5 →
  final_supplies = 20 →
  ∃ (initial_supplies : ℕ) (glue_bottles : ℕ),
    initial_supplies = students * paper_per_student + glue_bottles ∧
    initial_supplies / 2 + added_paper = final_supplies ∧
    glue_bottles = 6 :=
by sorry

end NUMINAMATH_CALUDE_kelly_glue_bottles_l3242_324209


namespace NUMINAMATH_CALUDE_find_y_l3242_324260

theorem find_y (c d : ℝ) (y : ℝ) (h1 : d > 0) : 
  ((3 * c) ^ (3 * d) = c^d * y^d) → y = 27 * c^2 := by
sorry

end NUMINAMATH_CALUDE_find_y_l3242_324260


namespace NUMINAMATH_CALUDE_quadratic_real_root_l3242_324227

theorem quadratic_real_root (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l3242_324227


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3242_324204

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | a * x - 1 > 0} = {x : ℝ | x < 1 / a} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3242_324204
