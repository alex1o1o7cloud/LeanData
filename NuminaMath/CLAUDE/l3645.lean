import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l3645_364587

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def SymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if
    for all points (x, g(x)), the point (3-x, g(x)) is also on the graph of g -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  SymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g :=
by
  sorry

#check symmetry_implies_axis

end NUMINAMATH_CALUDE_symmetry_implies_axis_l3645_364587


namespace NUMINAMATH_CALUDE_positive_sum_l3645_364511

theorem positive_sum (x y z : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l3645_364511


namespace NUMINAMATH_CALUDE_custom_equation_solution_l3645_364570

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℚ, star 3 (star 6 x) = -2 ∧ x = 17/2 := by sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l3645_364570


namespace NUMINAMATH_CALUDE_unique_modular_solution_l3645_364597

theorem unique_modular_solution (m : ℤ) : 
  (5 ≤ m ∧ m ≤ 9) → (m ≡ 5023 [ZMOD 6]) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l3645_364597


namespace NUMINAMATH_CALUDE_symmetry_line_probability_l3645_364519

/-- Represents a point on a grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a rectangle with a uniform grid --/
structure GridRectangle where
  width : Nat
  height : Nat

/-- The total number of points in the grid rectangle --/
def totalPoints (rect : GridRectangle) : Nat :=
  rect.width * rect.height

/-- The center point of the rectangle --/
def centerPoint (rect : GridRectangle) : GridPoint :=
  { x := rect.width / 2, y := rect.height / 2 }

/-- Checks if a given point is on a line of symmetry --/
def isOnSymmetryLine (p : GridPoint) (center : GridPoint) (rect : GridRectangle) : Bool :=
  p.x = center.x ∨ p.y = center.y

/-- Counts the number of points on lines of symmetry, excluding the center --/
def countSymmetryPoints (rect : GridRectangle) : Nat :=
  rect.width + rect.height - 2

/-- The main theorem --/
theorem symmetry_line_probability (rect : GridRectangle) : 
  rect.width = 10 ∧ rect.height = 10 →
  (countSymmetryPoints rect : Rat) / ((totalPoints rect - 1 : Nat) : Rat) = 2 / 11 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_line_probability_l3645_364519


namespace NUMINAMATH_CALUDE_log_stack_sum_l3645_364533

/-- Given an arithmetic sequence with 11 terms, starting at 5 and ending at 15,
    prove that the sum of all terms is 110. -/
theorem log_stack_sum : 
  let n : ℕ := 11  -- number of terms
  let a : ℕ := 5   -- first term
  let l : ℕ := 15  -- last term
  n * (a + l) / 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3645_364533


namespace NUMINAMATH_CALUDE_casino_game_max_guaranteed_money_l3645_364536

/-- Represents the outcome of a single bet -/
inductive BetOutcome
| Win
| Lose

/-- Represents the state of the game after each bet -/
structure GameState :=
  (money : ℕ)
  (bets_made : ℕ)
  (consecutive_losses : ℕ)

/-- The betting strategy function type -/
def BettingStrategy := GameState → ℕ

/-- The game rules function type -/
def GameRules := GameState → BetOutcome → GameState

theorem casino_game_max_guaranteed_money 
  (initial_money : ℕ) 
  (max_bets : ℕ) 
  (max_bet_amount : ℕ) 
  (consolation_win_threshold : ℕ) 
  (strategy : BettingStrategy) 
  (rules : GameRules) :
  initial_money = 100 →
  max_bets = 5 →
  max_bet_amount = 17 →
  consolation_win_threshold = 4 →
  ∃ (final_money : ℕ), final_money ≥ 98 ∧
    ∀ (outcomes : List BetOutcome), 
      outcomes.length = max_bets →
      let final_state := outcomes.foldl rules { money := initial_money, bets_made := 0, consecutive_losses := 0 }
      final_state.money ≥ final_money :=
by sorry

end NUMINAMATH_CALUDE_casino_game_max_guaranteed_money_l3645_364536


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l3645_364595

theorem fruit_condition_percentage (oranges bananas : ℕ) 
  (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 8 / 100 →
  let total_fruits := oranges + bananas
  let rotten_oranges := (rotten_oranges_percent * oranges).num
  let rotten_bananas := (rotten_bananas_percent * bananas).num
  let total_rotten := rotten_oranges + rotten_bananas
  let good_fruits := total_fruits - total_rotten
  (good_fruits : ℚ) / total_fruits * 100 = 87.8 := by
sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l3645_364595


namespace NUMINAMATH_CALUDE_intersection_sum_l3645_364526

/-- Given two lines y = nx + 5 and y = 4x + c that intersect at (8, 9),
    prove that n + c = -22.5 -/
theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 5 ∨ y = 4 * x + c) →
  9 = n * 8 + 5 →
  9 = 4 * 8 + c →
  n + c = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3645_364526


namespace NUMINAMATH_CALUDE_rebecca_camping_items_l3645_364529

/-- Represents the number of tent stakes Rebecca bought. -/
def tent_stakes : ℕ := sorry

/-- Represents the number of packets of drink mix Rebecca bought. -/
def drink_mix : ℕ := 3 * tent_stakes

/-- Represents the number of bottles of water Rebecca bought. -/
def water_bottles : ℕ := tent_stakes + 2

/-- The total number of items Rebecca bought. -/
def total_items : ℕ := 22

theorem rebecca_camping_items : tent_stakes = 4 :=
  by sorry

end NUMINAMATH_CALUDE_rebecca_camping_items_l3645_364529


namespace NUMINAMATH_CALUDE_star_sharing_problem_l3645_364591

theorem star_sharing_problem (stars : ℝ) (students_per_star : ℝ) 
  (h1 : stars = 3.0) 
  (h2 : students_per_star = 41.33333333) : 
  ⌊stars * students_per_star⌋ = 124 := by
  sorry

end NUMINAMATH_CALUDE_star_sharing_problem_l3645_364591


namespace NUMINAMATH_CALUDE_train_length_l3645_364508

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 8 → speed * time * (1000 / 3600) = 400 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3645_364508


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_l3645_364521

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (1, 1)
  f2 : ℝ × ℝ := (1, 3)
  -- Point on the ellipse
  p : ℝ × ℝ := (-4, 2)
  -- Constants in the ellipse equation
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- Positivity of a and b
  a_pos : a > 0
  b_pos : b > 0
  -- Ellipse equation
  eq : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - f1.1)^2 + (p.2 - f1.2)^2 + (p.1 - f2.1)^2 + (p.2 - f2.2)^2 = (2*a)^2}

/-- Theorem: For the given ellipse, a + k = 7 -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 7 := by sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_l3645_364521


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3645_364512

/-- Given complex numbers z₁, z₂, z₃ satisfying (z₃ - z₁) / (z₂ - z₁) = ai 
    where a ∈ ℝ and a ≠ 0, the angle between vectors ⃗Z₁Z₂ and ⃗Z₁Z₃ is π/2. -/
theorem angle_between_vectors (z₁ z₂ z₃ : ℂ) (a : ℝ) 
    (h : (z₃ - z₁) / (z₂ - z₁) = Complex.I * a) 
    (ha : a ≠ 0) : 
  Complex.arg ((z₃ - z₁) / (z₂ - z₁)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3645_364512


namespace NUMINAMATH_CALUDE_lawn_mowing_solution_l3645_364560

/-- Represents the lawn mowing problem -/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap mowing_speed : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let strips := lawn_width / effective_width
  let total_distance := strips * lawn_length
  let mowing_time := total_distance / mowing_speed
  0.94 < mowing_time ∧ mowing_time < 0.96

/-- Theorem stating the solution to the lawn mowing problem -/
theorem lawn_mowing_solution :
  LawnMowingProblem 72 120 (30/12) (6/12) 4500 :=
sorry

end NUMINAMATH_CALUDE_lawn_mowing_solution_l3645_364560


namespace NUMINAMATH_CALUDE_expression_value_at_nine_l3645_364501

theorem expression_value_at_nine :
  let x : ℝ := 9
  let f (x : ℝ) := (x^9 - 27*x^6 + 19683) / (x^6 - 27)
  f x = 492804 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_nine_l3645_364501


namespace NUMINAMATH_CALUDE_geometric_sequence_angle_l3645_364534

theorem geometric_sequence_angle (a : ℕ → ℝ) (α : ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 8 = -Real.sqrt 3 * Real.sin α) →  -- root product condition
  (a 1 + a 8 = 2 * Real.sin α) →  -- root sum condition
  ((a 1 + a 8)^2 = 2 * a 3 * a 6 + 6) →  -- given equation
  (0 < α ∧ α < π / 2) →  -- acute angle condition
  α = π / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_angle_l3645_364534


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3645_364538

/-- Given a line segment with one endpoint at (-3, -15) and midpoint at (2, -5),
    the sum of coordinates of the other endpoint is 12 -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (2 : ℝ) = (-3 + x) / 2 → 
    (-5 : ℝ) = (-15 + y) / 2 → 
    x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3645_364538


namespace NUMINAMATH_CALUDE_domain_of_f_given_range_l3645_364527

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem domain_of_f_given_range :
  (∀ y ∈ Set.Ioo 2 3, ∃ x, f x = y) ∧ f 2 = 3 →
  {x : ℝ | ∃ y ∈ Set.Ioo 2 3, f x = y} ∪ {2} = Set.Ioo 1 2 ∪ {2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_given_range_l3645_364527


namespace NUMINAMATH_CALUDE_arrangements_count_is_24_l3645_364573

/-- The number of ways to arrange 5 people in a line, where two specific people
    must stand next to each other but not at the ends. -/
def arrangements_count : ℕ :=
  /- Number of ways to arrange A and B together -/
  (2 * 1) *
  /- Number of positions for A and B together (excluding ends) -/
  3 *
  /- Number of ways to arrange the other 3 people -/
  (3 * 2 * 1)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_is_24_l3645_364573


namespace NUMINAMATH_CALUDE_largest_angle_is_75_l3645_364592

-- Define the angles of the triangle
def triangle_angles (a b c : ℝ) : Prop :=
  -- The sum of all angles in a triangle is 180°
  a + b + c = 180 ∧
  -- Two angles sum to 7/6 of a right angle (90°)
  b + c = 7/6 * 90 ∧
  -- One angle is 10° more than twice the other
  c = 2 * b + 10 ∧
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c

-- Theorem statement
theorem largest_angle_is_75 (a b c : ℝ) :
  triangle_angles a b c → max a (max b c) = 75 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_75_l3645_364592


namespace NUMINAMATH_CALUDE_box_volume_increase_l3645_364558

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l3645_364558


namespace NUMINAMATH_CALUDE_justin_jerseys_l3645_364559

theorem justin_jerseys (long_sleeve_cost : ℕ) (striped_cost : ℕ) (long_sleeve_count : ℕ) (total_spent : ℕ) :
  long_sleeve_cost = 15 →
  striped_cost = 10 →
  long_sleeve_count = 4 →
  total_spent = 80 →
  (total_spent - long_sleeve_cost * long_sleeve_count) / striped_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_justin_jerseys_l3645_364559


namespace NUMINAMATH_CALUDE_triangle_tangent_sum_l3645_364590

theorem triangle_tangent_sum (A B C : Real) : 
  A + B + C = π →  -- angle sum property of triangle
  A + C = 2 * B →  -- given condition
  Real.tan (A / 2) + Real.tan (C / 2) + Real.sqrt 3 * Real.tan (A / 2) * Real.tan (C / 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_sum_l3645_364590


namespace NUMINAMATH_CALUDE_school_picnic_volunteers_l3645_364599

theorem school_picnic_volunteers (total_parents : ℕ) (supervise : ℕ) (both : ℕ) (refresh_ratio : ℚ) : 
  total_parents = 84 →
  supervise = 25 →
  both = 11 →
  refresh_ratio = 3/2 →
  ∃ (refresh : ℕ) (neither : ℕ),
    refresh = refresh_ratio * neither ∧
    total_parents = (supervise - both) + (refresh - both) + both + neither ∧
    refresh = 42 := by
  sorry

end NUMINAMATH_CALUDE_school_picnic_volunteers_l3645_364599


namespace NUMINAMATH_CALUDE_fruit_salad_composition_l3645_364507

/-- Fruit salad composition problem -/
theorem fruit_salad_composition (total : ℕ) (b r g c : ℕ) : 
  total = 360 ∧ 
  r = 3 * b ∧ 
  g = 4 * c ∧ 
  c = 5 * r ∧ 
  total = b + r + g + c → 
  c = 68 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_composition_l3645_364507


namespace NUMINAMATH_CALUDE_sanda_exercise_days_l3645_364545

theorem sanda_exercise_days 
  (javier_daily_minutes : ℕ) 
  (javier_days : ℕ) 
  (sanda_daily_minutes : ℕ) 
  (total_minutes : ℕ) :
  javier_daily_minutes = 50 →
  javier_days = 7 →
  sanda_daily_minutes = 90 →
  total_minutes = 620 →
  (javier_daily_minutes * javier_days + sanda_daily_minutes * (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = total_minutes) →
  (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = 3 :=
by sorry

end NUMINAMATH_CALUDE_sanda_exercise_days_l3645_364545


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3645_364547

theorem trigonometric_equation_solutions (x : ℝ) :
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * (Real.cos (3 * x / 2))^2 ↔
  (∃ k : ℤ, x = π / 8 * (2 * k + 1)) ∨ (∃ n : ℤ, x = π / 4 * (4 * n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3645_364547


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l3645_364566

/-- Represents the steps in linear regression analysis -/
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | ComputeCorrelation
  | PlotScatterDiagram

/-- Represents a sequence of regression steps -/
def RegressionSequence := List RegressionStep

/-- The correct sequence of regression steps -/
def correctSequence : RegressionSequence := [
  RegressionStep.CollectData,
  RegressionStep.PlotScatterDiagram,
  RegressionStep.ComputeCorrelation,
  RegressionStep.CalculateEquation,
  RegressionStep.InterpretEquation
]

/-- Predicate to check if a sequence is valid for determining linear relationship -/
def isValidSequence (seq : RegressionSequence) : Prop := 
  seq = correctSequence

/-- Theorem stating that the correct sequence is valid for linear regression analysis -/
theorem correct_regression_sequence : 
  isValidSequence correctSequence := by sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l3645_364566


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3645_364540

/-- Converts a number from base 7 to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 7 -/
def unitsDigitBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem units_digit_of_expression :
  let a := 43
  let b := 124
  let c := 15
  unitsDigitBase7 ((toBase7 (toBase10 a + toBase10 b)) * c) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3645_364540


namespace NUMINAMATH_CALUDE_principal_is_7000_l3645_364555

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  time : ℚ
  gainPerYear : ℚ

/-- Theorem stating that given the conditions, the principal is 7000 -/
theorem principal_is_7000 (t : Transaction) 
  (h1 : t.time = 2)
  (h2 : t.borrowRate = 4)
  (h3 : t.lendRate = 6)
  (h4 : t.gainPerYear = 140)
  (h5 : t.gainPerYear = (simpleInterest t.principal t.lendRate t.time - 
                         simpleInterest t.principal t.borrowRate t.time) / t.time) :
  t.principal = 7000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_7000_l3645_364555


namespace NUMINAMATH_CALUDE_boarding_students_count_total_boarding_students_l3645_364553

theorem boarding_students_count (total_students : ℕ) (male_students : ℕ) 
  (female_youth_league : ℕ) (female_boarding : ℕ) (non_boarding_youth_league : ℕ) 
  (male_boarding_youth_league : ℕ) (male_non_youth_league_non_boarding : ℕ) 
  (female_non_youth_league_non_boarding : ℕ) : ℕ :=
  sorry

/-- Given the following conditions:
    - There are 50 students in total
    - There are 33 male students
    - There are 7 female members of the Youth League
    - There are 9 female boarding students
    - There are 15 non-boarding members of the Youth League
    - There are 6 male boarding members of the Youth League
    - There are 8 male students who are non-members of the Youth League and non-boarding
    - There are 3 female students who are non-members of the Youth League and non-boarding
    The total number of boarding students is 28. -/
theorem total_boarding_students :
  boarding_students_count 50 33 7 9 15 6 8 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_boarding_students_count_total_boarding_students_l3645_364553


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l3645_364583

/-- Given that (1-2x)^5 = a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove the following statements about the coefficients. -/
theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ 
   a₁ + a₂ + a₃ + a₄ + a₅ = -2 ∧
   a₁ + a₃ + a₅ = -122) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l3645_364583


namespace NUMINAMATH_CALUDE_x_value_proof_l3645_364586

theorem x_value_proof (x y : ℝ) (h1 : x > y) 
  (h2 : x^2 * y^2 + x^2 + y^2 + 2*x*y = 40) 
  (h3 : x*y + x + y = 8) : x = 3 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3645_364586


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3645_364572

/-- For a hyperbola with equation x²/9 - y²/m = 1 and eccentricity e = 2, m = 27 -/
theorem hyperbola_eccentricity (x y m : ℝ) (e : ℝ) 
  (h1 : x^2 / 9 - y^2 / m = 1)
  (h2 : e = 2)
  (h3 : e = Real.sqrt (1 + m / 9)) : 
  m = 27 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3645_364572


namespace NUMINAMATH_CALUDE_preimage_of_two_neg_one_l3645_364598

/-- A mapping f from ℝ² to ℝ² defined by f(a,b) = (a+b, a-b) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (a, b) ↦ (a + b, a - b)

/-- The theorem stating that the preimage of (2, -1) under f is (1/2, 3/2) -/
theorem preimage_of_two_neg_one : 
  f (1/2, 3/2) = (2, -1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_two_neg_one_l3645_364598


namespace NUMINAMATH_CALUDE_point_B_in_third_quadrant_l3645_364539

/-- Given that point A (-x, y-1) is in the fourth quadrant, 
    prove that point B (y-1, x) is in the third quadrant. -/
theorem point_B_in_third_quadrant 
  (x y : ℝ) 
  (h_fourth : -x > 0 ∧ y - 1 < 0) : 
  y - 1 < 0 ∧ x < 0 := by
sorry

end NUMINAMATH_CALUDE_point_B_in_third_quadrant_l3645_364539


namespace NUMINAMATH_CALUDE_divisibility_by_1947_l3645_364589

theorem divisibility_by_1947 (n : ℕ) (h : Odd n) :
  (46^n + 296 * 13^n) % 1947 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1947_l3645_364589


namespace NUMINAMATH_CALUDE_cubes_fill_box_l3645_364524

def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_size : ℕ := 2

theorem cubes_fill_box : 
  (box_length / cube_size) * (box_width / cube_size) * (box_height / cube_size) * (cube_size^3) = 
  box_length * box_width * box_height := by
  sorry

end NUMINAMATH_CALUDE_cubes_fill_box_l3645_364524


namespace NUMINAMATH_CALUDE_subset_range_l3645_364541

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m - 3}

-- Theorem statement
theorem subset_range (m : ℝ) : B m ⊆ A ↔ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_subset_range_l3645_364541


namespace NUMINAMATH_CALUDE_digits_left_of_264_divisible_by_4_l3645_364506

theorem digits_left_of_264_divisible_by_4 : 
  (∀ n : ℕ, n < 10 → (n * 1000 + 264) % 4 = 0) ∧ 
  (∃ (S : Finset ℕ), S.card = 10 ∧ ∀ n ∈ S, n < 10 ∧ (n * 1000 + 264) % 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_digits_left_of_264_divisible_by_4_l3645_364506


namespace NUMINAMATH_CALUDE_second_warehouse_more_profitable_l3645_364575

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_first : ℝ := 80

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_second : ℝ := 20

/-- Represents the probability of the bank repossessing the second warehouse -/
def repossession_probability : ℝ := 0.5

/-- Represents the number of months after which repossession might occur -/
def repossession_month : ℕ := 5

/-- Represents the moving expenses in thousands of rubles -/
def moving_expenses : ℝ := 150

/-- Represents the lease duration in months -/
def lease_duration : ℕ := 12

/-- Calculates the expected cost of renting the second warehouse for one year -/
def expected_cost_second : ℝ :=
  let cost_no_repossession := monthly_rent_second * lease_duration
  let cost_repossession := monthly_rent_second * repossession_month +
                           monthly_rent_first * (lease_duration - repossession_month) +
                           moving_expenses
  (1 - repossession_probability) * cost_no_repossession +
  repossession_probability * cost_repossession

/-- Calculates the cost of renting the first warehouse for one year -/
def cost_first : ℝ := monthly_rent_first * lease_duration

theorem second_warehouse_more_profitable :
  expected_cost_second < cost_first :=
sorry

end NUMINAMATH_CALUDE_second_warehouse_more_profitable_l3645_364575


namespace NUMINAMATH_CALUDE_fruit_store_problem_l3645_364548

-- Define the types of fruits
inductive FruitType
| A
| B

-- Define the purchase data
structure PurchaseData where
  typeA : ℕ
  typeB : ℕ
  totalCost : ℕ

-- Define the problem parameters
def firstPurchase : PurchaseData := ⟨60, 40, 1520⟩
def secondPurchase : PurchaseData := ⟨30, 50, 1360⟩
def thirdPurchaseTotal : ℕ := 200
def thirdPurchaseMaxCost : ℕ := 3360
def typeASellingPrice : ℕ := 17
def typeBSellingPrice : ℕ := 30
def minProfit : ℕ := 800

-- Define the theorem
theorem fruit_store_problem :
  ∃ (priceA priceB : ℕ) (m : ℕ),
    -- Conditions for the first two purchases
    priceA * firstPurchase.typeA + priceB * firstPurchase.typeB = firstPurchase.totalCost ∧
    priceA * secondPurchase.typeA + priceB * secondPurchase.typeB = secondPurchase.totalCost ∧
    -- Conditions for the third purchase
    ∀ (x : ℕ),
      x ≤ thirdPurchaseTotal →
      priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost →
      (typeASellingPrice - priceA) * (x - m) + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m) ≥ minProfit →
      -- Conclusion
      priceA = 12 ∧ priceB = 20 ∧ m ≤ 22 ∧
      ∀ (m' : ℕ), m' > m → 
        ¬(∃ (x : ℕ),
          x ≤ thirdPurchaseTotal ∧
          priceA * x + priceB * (thirdPurchaseTotal - x) ≤ thirdPurchaseMaxCost ∧
          (typeASellingPrice - priceA) * (x - m') + (typeBSellingPrice - priceB) * (thirdPurchaseTotal - x - 3 * m') ≥ minProfit) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l3645_364548


namespace NUMINAMATH_CALUDE_inverse_point_theorem_l3645_364577

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that f(1) + 1 = 2
axiom condition : f 1 + 1 = 2

-- Theorem to prove
theorem inverse_point_theorem : f_inv 1 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_point_theorem_l3645_364577


namespace NUMINAMATH_CALUDE_additional_workers_for_wall_project_l3645_364585

/-- Calculates the number of additional workers needed to complete a project on time -/
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed_percentage : ℚ) : ℕ :=
  let total_work := total_days * initial_workers
  let remaining_work := total_work * (1 - work_completed_percentage)
  let remaining_days := total_days - days_passed
  let work_by_existing := initial_workers * remaining_days
  let additional_work_needed := remaining_work - work_by_existing
  (additional_work_needed / remaining_days).ceil.toNat

/-- Proves that given the initial conditions, 12 additional workers are needed -/
theorem additional_workers_for_wall_project : 
  additional_workers_needed 50 60 25 (2/5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_workers_for_wall_project_l3645_364585


namespace NUMINAMATH_CALUDE_range_of_m_l3645_364594

def f (x : ℝ) := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 10) ∧
  (∀ x ∈ Set.Icc (-1) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3645_364594


namespace NUMINAMATH_CALUDE_total_amount_received_l3645_364557

def lottery_winnings : ℚ := 555850
def num_students : ℕ := 500
def fraction : ℚ := 3 / 10000

theorem total_amount_received :
  (lottery_winnings * fraction * num_students : ℚ) = 833775 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_received_l3645_364557


namespace NUMINAMATH_CALUDE_unique_poly3_satisfying_conditions_l3645_364517

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  f : ℝ → ℝ
  degree_3 : ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The conditions that the polynomial function must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x : ℝ, p.f (x^2) = (p.f x)^2 ∧
            p.f (x^2) = p.f (p.f x) ∧
            p.f 1 = 1

/-- Theorem stating the uniqueness of the polynomial function -/
theorem unique_poly3_satisfying_conditions :
  ∃! p : Poly3, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_unique_poly3_satisfying_conditions_l3645_364517


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l3645_364528

theorem complex_modulus_squared (z : ℂ) (h : z * Complex.abs z = 3 + 12*I) : Complex.abs z ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l3645_364528


namespace NUMINAMATH_CALUDE_chess_board_pawn_arrangements_l3645_364580

theorem chess_board_pawn_arrangements (n : ℕ) (h : n = 5) : 
  (Finset.range n).card.factorial = 120 := by sorry

end NUMINAMATH_CALUDE_chess_board_pawn_arrangements_l3645_364580


namespace NUMINAMATH_CALUDE_min_sum_of_fractions_l3645_364523

def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D : Nat) :
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (∀ A' B' C' D' : Nat,
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    B' ≠ 0 → D' ≠ 0 →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_fractions_l3645_364523


namespace NUMINAMATH_CALUDE_quadratic_function_example_l3645_364550

theorem quadratic_function_example : ∃ (a b c : ℝ),
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_example_l3645_364550


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3645_364588

theorem tangent_line_problem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x + a / x
  let f' : ℝ → ℝ := λ x => 1 / x - a / (x^2)
  (∀ y, 4 * y - 1 - b = 0 ↔ y = f 1 + f' 1 * (1 - 1)) →
  a * b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l3645_364588


namespace NUMINAMATH_CALUDE_complex_magnitude_l3645_364542

theorem complex_magnitude (z : ℂ) : (1 - 2*Complex.I)*z = 3 + 4*Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3645_364542


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3645_364581

theorem max_value_of_expression (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a ∈ ({2, 3, 6} : Set ℕ) →
  b ∈ ({2, 3, 6} : Set ℕ) →
  c ∈ ({2, 3, 6} : Set ℕ) →
  (a : ℚ) / ((b : ℚ) / (c : ℚ)) ≤ 9 →
  (∃ a' b' c' : ℕ, 
    a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
    a' ∈ ({2, 3, 6} : Set ℕ) ∧
    b' ∈ ({2, 3, 6} : Set ℕ) ∧
    c' ∈ ({2, 3, 6} : Set ℕ) ∧
    (a' : ℚ) / ((b' : ℚ) / (c' : ℚ)) = 9) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3645_364581


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3645_364576

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3645_364576


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3645_364567

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + x^3 - 6 * x^2 + 9 * x - 5) + 
  (-x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 2) + 
  (3 * x^4 - 3 * x^3 + x^2 - x + 1) = 
  4 * x^4 - 8 * x^2 + 12 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3645_364567


namespace NUMINAMATH_CALUDE_range_of_z_l3645_364564

-- Define the variables and their constraints
def a : ℝ := sorry
def b : ℝ := sorry

-- Define the function z
def z (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem range_of_z :
  (2 < a ∧ a < 3) → (-2 < b ∧ b < -1) →
  ∀ z₀ : ℝ, (∃ a₀ b₀ : ℝ, (2 < a₀ ∧ a₀ < 3) ∧ (-2 < b₀ ∧ b₀ < -1) ∧ z a₀ b₀ = z₀) ↔ (5 < z₀ ∧ z₀ < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l3645_364564


namespace NUMINAMATH_CALUDE_coefficient_x2_implies_a_eq_2_l3645_364565

/-- The coefficient of x^2 in the expansion of (x+a)^5 -/
def coefficient_x2 (a : ℝ) : ℝ := 10 * a^3

/-- Theorem stating that if the coefficient of x^2 in (x+a)^5 is 80, then a = 2 -/
theorem coefficient_x2_implies_a_eq_2 :
  coefficient_x2 2 = 80 ∧ (∀ a : ℝ, coefficient_x2 a = 80 → a = 2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x2_implies_a_eq_2_l3645_364565


namespace NUMINAMATH_CALUDE_equation_solution_l3645_364556

theorem equation_solution (y : ℚ) : 
  (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3645_364556


namespace NUMINAMATH_CALUDE_circle_symmetry_l3645_364571

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 l : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 ∧
  (x1 + x2) / 2 + 2 * ((y1 + y2) / 2) - 5 = 0 ∧
  (y2 - y1) / (x2 - x1) * (-1/2) = -1

-- Define circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line C1 C2 l → ∀ x y, C2 x y ↔ x^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3645_364571


namespace NUMINAMATH_CALUDE_min_detectors_for_gold_coins_l3645_364514

/-- Represents a grid of unit squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a subgrid within a larger grid -/
structure Subgrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the minimum number of detectors needed -/
def min_detectors (g : Grid) (s : Subgrid) : ℕ := sorry

/-- The main theorem stating the minimum number of detectors needed -/
theorem min_detectors_for_gold_coins (g : Grid) (s : Subgrid) :
  g.rows = 2017 ∧ g.cols = 2017 ∧ s.rows = 1500 ∧ s.cols = 1500 →
  min_detectors g s = 1034 := by sorry

end NUMINAMATH_CALUDE_min_detectors_for_gold_coins_l3645_364514


namespace NUMINAMATH_CALUDE_banana_difference_l3645_364515

theorem banana_difference (total : ℕ) (lydia_bananas : ℕ) (donna_bananas : ℕ)
  (h1 : total = 200)
  (h2 : lydia_bananas = 60)
  (h3 : donna_bananas = 40) :
  total - donna_bananas - lydia_bananas - lydia_bananas = 40 := by
  sorry

end NUMINAMATH_CALUDE_banana_difference_l3645_364515


namespace NUMINAMATH_CALUDE_wall_photo_area_l3645_364504

theorem wall_photo_area (paper_width paper_length frame_width : ℕ) 
  (hw : paper_width = 8)
  (hl : paper_length = 12)
  (hf : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 := by
  sorry

end NUMINAMATH_CALUDE_wall_photo_area_l3645_364504


namespace NUMINAMATH_CALUDE_sequence_relationship_l3645_364552

def y (x : ℕ) : ℕ := x^2 + x + 1

theorem sequence_relationship (x : ℕ) :
  x > 0 →
  (y (x + 1) - y x = 2 * x + 2) ∧
  (y (x + 2) - y (x + 1) = 2 * x + 4) ∧
  (y (x + 3) - y (x + 2) = 2 * x + 6) ∧
  (y (x + 4) - y (x + 3) = 2 * x + 8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_relationship_l3645_364552


namespace NUMINAMATH_CALUDE_log_sum_simplification_l3645_364578

theorem log_sum_simplification :
  let f (a b : ℝ) := 1 / (Real.log a / Real.log b + 1)
  f 3 12 + f 2 8 + f 7 9 = 1 - Real.log 7 / Real.log 1008 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l3645_364578


namespace NUMINAMATH_CALUDE_coin_value_difference_l3645_364531

/-- Represents the total number of coins Alice has -/
def total_coins : ℕ := 3030

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value in cents given the number of dimes -/
def total_value (dimes : ℕ) : ℕ :=
  dime_value * dimes + nickel_value * (total_coins - dimes)

/-- Represents the constraint that Alice has at least three times as many nickels as dimes -/
def nickel_constraint (dimes : ℕ) : Prop :=
  3 * dimes ≤ total_coins - dimes

theorem coin_value_difference :
  ∃ (max_dimes min_dimes : ℕ),
    nickel_constraint max_dimes ∧
    nickel_constraint min_dimes ∧
    (∀ d, nickel_constraint d → total_value d ≤ total_value max_dimes) ∧
    (∀ d, nickel_constraint d → total_value min_dimes ≤ total_value d) ∧
    total_value max_dimes - total_value min_dimes = 3780 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_difference_l3645_364531


namespace NUMINAMATH_CALUDE_investment_ratio_l3645_364546

/-- Prove that the investment ratio between A and C is 3:1 --/
theorem investment_ratio (a b c : ℕ) (total_profit c_profit : ℕ) : 
  a = 3 * b → -- A and B invested in a ratio of 3:1
  total_profit = 60000 → -- The total profit was 60000
  c_profit = 20000 → -- C received 20000 from the profit
  3 * c = a := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l3645_364546


namespace NUMINAMATH_CALUDE_weight_of_a_l3645_364502

-- Define the people
structure Person where
  weight : ℝ
  height : ℝ
  age : ℝ

-- Define the group of A, B, C
def group_abc (a b c : Person) : Prop :=
  (a.weight + b.weight + c.weight) / 3 = 84 ∧
  (a.height + b.height + c.height) / 3 = 170 ∧
  (a.age + b.age + c.age) / 3 = 30

-- Define the group with D added
def group_abcd (a b c d : Person) : Prop :=
  (a.weight + b.weight + c.weight + d.weight) / 4 = 80 ∧
  (a.height + b.height + c.height + d.height) / 4 = 172 ∧
  (a.age + b.age + c.age + d.age) / 4 = 28

-- Define the group with E replacing A
def group_bcde (b c d e : Person) : Prop :=
  (b.weight + c.weight + d.weight + e.weight) / 4 = 79 ∧
  (b.height + c.height + d.height + e.height) / 4 = 173 ∧
  (b.age + c.age + d.age + e.age) / 4 = 27

-- Define the relationship between D and E
def d_e_relation (d e a : Person) : Prop :=
  e.weight = d.weight + 7 ∧
  e.age = a.age - 3

-- Theorem statement
theorem weight_of_a 
  (a b c d e : Person)
  (h1 : group_abc a b c)
  (h2 : group_abcd a b c d)
  (h3 : group_bcde b c d e)
  (h4 : d_e_relation d e a) :
  a.weight = 79 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_a_l3645_364502


namespace NUMINAMATH_CALUDE_area_decreasing_map_l3645_364505

open Set
open MeasureTheory

-- Define a distance function for ℝ²
noncomputable def distance (x y : ℝ × ℝ) : ℝ := Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

-- Define the properties of function f
def is_distance_decreasing (f : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ x y : ℝ × ℝ, distance x y ≥ distance (f x) (f y)

-- Theorem statement
theorem area_decreasing_map
  (f : ℝ × ℝ → ℝ × ℝ)
  (h_inj : Function.Injective f)
  (h_surj : Function.Surjective f)
  (h_dist : is_distance_decreasing f)
  (A : Set (ℝ × ℝ)) :
  MeasureTheory.volume A ≥ MeasureTheory.volume (f '' A) :=
sorry

end NUMINAMATH_CALUDE_area_decreasing_map_l3645_364505


namespace NUMINAMATH_CALUDE_triangle_construction_possible_l3645_364596

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def OnLine (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def AngleBisector (A B C : Point) (l : Line) : Prop :=
  -- This is a simplified representation of an angle bisector
  OnLine A l ∧ ∃ (P : Point), OnLine P l ∧ P ≠ A

-- Theorem statement
theorem triangle_construction_possible (l : Line) :
  ∃ (A B C N M : Point),
    Midpoint A C N ∧
    Midpoint B C M ∧
    AngleBisector A B C l :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_possible_l3645_364596


namespace NUMINAMATH_CALUDE_greatest_x_value_l3645_364522

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_x_value (x y : ℕ) (a b : ℝ) :
  x > 0 →
  y > 0 →
  is_prime y →
  a > 1 →
  b > 1 →
  a = 2.75 →
  b = 4.26 →
  ((a * x^2) / (y^3 : ℝ)) + b < 800000 →
  Nat.gcd x y = 1 →
  (∀ x' y' : ℕ, x' > 0 → y' > 0 → is_prime y' → Nat.gcd x' y' = 1 →
    ((a * x'^2) / (y'^3 : ℝ)) + b < 800000 → x' + y' < x + y) →
  x ≤ 2801 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3645_364522


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3645_364518

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-41) 3 2 = -437 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l3645_364518


namespace NUMINAMATH_CALUDE_cube_of_99999_l3645_364579

theorem cube_of_99999 : 
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_99999_l3645_364579


namespace NUMINAMATH_CALUDE_triangle_properties_l3645_364551

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a - c) / (a * Real.cos C + c * Real.cos A) = (b - c) / (a + c) →
  a + b + c ≤ 3 * Real.sqrt 3 →
  A = π / 3 ∧ a / (2 * Real.sin (π / 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3645_364551


namespace NUMINAMATH_CALUDE_chef_cakes_l3645_364532

/-- Given a total number of eggs, eggs put in the fridge, and eggs needed per cake,
    calculate the number of cakes that can be made. -/
def cakes_made (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

/-- Prove that given 60 total eggs, with 10 eggs put in the fridge,
    and 5 eggs needed for one cake, the number of cakes the chef can make is 10. -/
theorem chef_cakes :
  cakes_made 60 10 5 = 10 := by
sorry

end NUMINAMATH_CALUDE_chef_cakes_l3645_364532


namespace NUMINAMATH_CALUDE_pyramid_x_value_l3645_364535

/-- Pyramid represents a numerical pyramid where each number below the top row
    is the product of the number to the right and the number to the left in the row immediately above it. -/
structure Pyramid where
  top_left : ℕ
  middle : ℕ
  bottom_left : ℕ
  x : ℕ

/-- Given a Pyramid, this theorem proves that x must be 4 -/
theorem pyramid_x_value (p : Pyramid) (h1 : p.top_left = 35) (h2 : p.middle = 700) (h3 : p.bottom_left = 5)
  (h4 : p.middle = p.top_left * (p.middle / p.top_left))
  (h5 : p.middle / p.top_left = p.bottom_left * p.x) :
  p.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_x_value_l3645_364535


namespace NUMINAMATH_CALUDE_water_needed_for_recipe_l3645_364549

/-- Represents the ratio of ingredients in the fruit punch recipe -/
structure PunchRatio where
  water : ℕ
  orange : ℕ
  lemon : ℕ

/-- Calculates the amount of water needed for a given punch recipe and total volume -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℕ) : ℚ :=
  let total_parts := ratio.water + ratio.orange + ratio.lemon
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 15/2 quarts -/
theorem water_needed_for_recipe : 
  let recipe := PunchRatio.mk 5 2 1
  let total_gallons := 3
  let quarts_per_gallon := 4
  water_needed recipe total_gallons quarts_per_gallon = 15/2 := by
  sorry


end NUMINAMATH_CALUDE_water_needed_for_recipe_l3645_364549


namespace NUMINAMATH_CALUDE_shift_cosine_to_sine_l3645_364554

theorem shift_cosine_to_sine (x : ℝ) :
  let original := λ x : ℝ => 2 * Real.cos (2 * x)
  let shifted := λ x : ℝ => original (x - π / 8)
  let target := λ x : ℝ => 2 * Real.sin (2 * x + π / 4)
  0 < π / 8 ∧ π / 8 < π / 2 →
  shifted = target := by sorry

end NUMINAMATH_CALUDE_shift_cosine_to_sine_l3645_364554


namespace NUMINAMATH_CALUDE_roots_sum_product_l3645_364530

theorem roots_sum_product (p q r : ℂ) : 
  (2 * p ^ 3 - 5 * p ^ 2 + 7 * p - 3 = 0) →
  (2 * q ^ 3 - 5 * q ^ 2 + 7 * q - 3 = 0) →
  (2 * r ^ 3 - 5 * r ^ 2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l3645_364530


namespace NUMINAMATH_CALUDE_michael_monica_ratio_l3645_364563

/-- The ages of three people satisfy certain conditions -/
structure AgesProblem where
  /-- Patrick's age -/
  p : ℕ
  /-- Michael's age -/
  m : ℕ
  /-- Monica's age -/
  mo : ℕ
  /-- The ages of Patrick and Michael are in the ratio of 3:5 -/
  patrick_michael_ratio : 3 * m = 5 * p
  /-- The sum of their ages is 88 -/
  sum_of_ages : p + m + mo = 88
  /-- The difference between Monica and Patrick's ages is 22 -/
  monica_patrick_diff : mo - p = 22

/-- The ratio of Michael's age to Monica's age is 3:4 -/
theorem michael_monica_ratio (prob : AgesProblem) : 3 * prob.mo = 4 * prob.m := by
  sorry

end NUMINAMATH_CALUDE_michael_monica_ratio_l3645_364563


namespace NUMINAMATH_CALUDE_merry_go_round_ride_times_l3645_364516

theorem merry_go_round_ride_times 
  (dave_time : ℝ) 
  (erica_time : ℝ) 
  (erica_longer_percent : ℝ) :
  dave_time = 10 →
  erica_time = 65 →
  erica_longer_percent = 0.30 →
  ∃ (chuck_time : ℝ),
    erica_time = chuck_time * (1 + erica_longer_percent) ∧
    chuck_time / dave_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_ride_times_l3645_364516


namespace NUMINAMATH_CALUDE_fraction_power_simplification_l3645_364543

theorem fraction_power_simplification :
  (77777 : ℕ) = 7 * 11111 →
  (77777 ^ 6 : ℕ) / (11111 ^ 6) = 117649 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_simplification_l3645_364543


namespace NUMINAMATH_CALUDE_sum_of_products_power_inequality_l3645_364520

theorem sum_of_products_power_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b) ^ (5/4 : ℝ) + (b * c) ^ (5/4 : ℝ) + (c * a) ^ (5/4 : ℝ) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_power_inequality_l3645_364520


namespace NUMINAMATH_CALUDE_sandy_turnips_count_undetermined_l3645_364582

/-- Represents the number of vegetables grown by a person -/
structure VegetableCount where
  carrots : ℕ
  turnips : ℕ

/-- The given information about Sandy and Mary's vegetable growth -/
def given : Prop :=
  ∃ (sandy : VegetableCount) (mary : VegetableCount),
    sandy.carrots = 8 ∧
    mary.carrots = 6 ∧
    sandy.carrots + mary.carrots = 14

/-- The statement that Sandy's turnip count cannot be determined -/
def sandy_turnips_undetermined : Prop :=
  ∀ (n : ℕ),
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips = n) →
    (∃ (sandy : VegetableCount) (mary : VegetableCount),
      sandy.carrots = 8 ∧
      mary.carrots = 6 ∧
      sandy.carrots + mary.carrots = 14 ∧
      sandy.turnips ≠ n)

theorem sandy_turnips_count_undetermined :
  given → sandy_turnips_undetermined :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_turnips_count_undetermined_l3645_364582


namespace NUMINAMATH_CALUDE_equation_solution_l3645_364544

theorem equation_solution (y : ℝ) (h : y ≠ 0) : 
  (2 / y) + (3 / y) / (6 / y) = 1.5 → y = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3645_364544


namespace NUMINAMATH_CALUDE_square_area_difference_l3645_364503

theorem square_area_difference : 
  ∀ (smaller_length greater_length : ℝ),
    greater_length = 7 →
    greater_length = smaller_length + 2 →
    (greater_length ^ 2 - smaller_length ^ 2 : ℝ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_area_difference_l3645_364503


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l3645_364574

theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 120) (h2 : num_packs = 6) :
  total_cost / num_packs = 20 := by
sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l3645_364574


namespace NUMINAMATH_CALUDE_total_weight_is_63_l3645_364513

/-- The weight of beeswax used in each candle, in ounces -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes -/
def num_candles : ℕ := 10 - 3

/-- The total weight of all candles made by Ethan, in ounces -/
def total_weight : ℕ := num_candles * (beeswax_weight + coconut_oil_weight)

/-- Theorem stating that the total weight of candles is 63 ounces -/
theorem total_weight_is_63 : total_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_63_l3645_364513


namespace NUMINAMATH_CALUDE_number_of_larger_planes_l3645_364537

/-- Represents the number of airplanes --/
def total_planes : ℕ := 4

/-- Represents the capacity of smaller tanks in liters --/
def smaller_tank_capacity : ℕ := 60

/-- Represents the fuel cost per liter in cents --/
def fuel_cost_per_liter : ℕ := 50

/-- Represents the service charge per plane in cents --/
def service_charge : ℕ := 10000

/-- Represents the total cost to fill all planes in cents --/
def total_cost : ℕ := 55000

/-- Calculates the capacity of larger tanks --/
def larger_tank_capacity : ℕ := smaller_tank_capacity + smaller_tank_capacity / 2

/-- Calculates the fuel cost for a smaller plane in cents --/
def smaller_plane_fuel_cost : ℕ := smaller_tank_capacity * fuel_cost_per_liter

/-- Calculates the fuel cost for a larger plane in cents --/
def larger_plane_fuel_cost : ℕ := larger_tank_capacity * fuel_cost_per_liter

/-- Calculates the total cost for a smaller plane in cents --/
def smaller_plane_total_cost : ℕ := smaller_plane_fuel_cost + service_charge

/-- Calculates the total cost for a larger plane in cents --/
def larger_plane_total_cost : ℕ := larger_plane_fuel_cost + service_charge

/-- Proves that the number of larger planes is 2 --/
theorem number_of_larger_planes : 
  ∃ (n : ℕ), n + (total_planes - n) = total_planes ∧ 
             n * larger_plane_total_cost + (total_planes - n) * smaller_plane_total_cost = total_cost ∧
             n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_larger_planes_l3645_364537


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3645_364525

theorem fraction_evaluation : 
  (20-18+16-14+12-10+8-6+4-2) / (2-4+6-8+10-12+14-16+18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3645_364525


namespace NUMINAMATH_CALUDE_set_operations_l3645_364569

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x < -2 ∨ x > 5}

def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem set_operations :
  (Aᶜ : Set ℝ) = {x | -2 ≤ x ∧ x ≤ 5} ∧
  (Bᶜ : Set ℝ) = {x | x < 4 ∨ x > 6} ∧
  (A ∩ B : Set ℝ) = {x | 5 < x ∧ x ≤ 6} ∧
  ((A ∪ B)ᶜ : Set ℝ) = {x | -2 ≤ x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3645_364569


namespace NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l3645_364510

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof goes here
theorem bob_pennies_proof : bob_pennies 9 31 := by
  sorry

end NUMINAMATH_CALUDE_bob_pennies_bob_pennies_proof_l3645_364510


namespace NUMINAMATH_CALUDE_greatest_good_set_l3645_364561

def is_good (k : ℕ) (S : Set ℕ) : Prop :=
  ∃ (color : ℕ → Fin k),
    ∀ s ∈ S, ∀ x y : ℕ, x + y = s → color x ≠ color y

theorem greatest_good_set (k : ℕ) (h : k > 1) :
  (∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k - 1}) ∧
  ¬(∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k}) :=
sorry

end NUMINAMATH_CALUDE_greatest_good_set_l3645_364561


namespace NUMINAMATH_CALUDE_park_area_theorem_l3645_364509

/-- Represents a rectangular park with a given perimeter where the width is one-third of the length -/
structure RectangularPark where
  perimeter : ℝ
  width : ℝ
  length : ℝ
  width_length_relation : width = length / 3
  perimeter_constraint : perimeter = 2 * (width + length)

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.width * park.length

/-- Theorem stating that a rectangular park with a perimeter of 90 meters and width one-third of its length has an area of 379.6875 square meters -/
theorem park_area_theorem (park : RectangularPark) (h : park.perimeter = 90) : 
  parkArea park = 379.6875 := by
  sorry

end NUMINAMATH_CALUDE_park_area_theorem_l3645_364509


namespace NUMINAMATH_CALUDE_remainder_problem_l3645_364593

theorem remainder_problem : 123456789012 % 360 = 108 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3645_364593


namespace NUMINAMATH_CALUDE_football_team_right_handed_players_l3645_364584

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + ((total_players - throwers) * 2) / 3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_football_team_right_handed_players_l3645_364584


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3645_364568

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 8 →
  total_paid = 1000 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3645_364568


namespace NUMINAMATH_CALUDE_bob_alice_difference_l3645_364500

/-- The difference in final amounts between two investors, given their initial investment
    and respective returns. -/
def investment_difference (initial_investment : ℕ) (alice_multiplier bob_multiplier : ℕ) : ℕ :=
  (initial_investment * bob_multiplier + initial_investment) - (initial_investment * alice_multiplier)

/-- Theorem stating that given the problem conditions, Bob ends up with $8000 more than Alice. -/
theorem bob_alice_difference : investment_difference 2000 2 5 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_bob_alice_difference_l3645_364500


namespace NUMINAMATH_CALUDE_parabola_directrix_l3645_364562

theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 → (∃ (k : ℝ), y = -1/(4*k) ∧ k = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3645_364562
