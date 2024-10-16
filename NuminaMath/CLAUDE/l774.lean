import Mathlib

namespace NUMINAMATH_CALUDE_tensor_inequality_implies_a_range_l774_77482

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem tensor_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → tensor (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_tensor_inequality_implies_a_range_l774_77482


namespace NUMINAMATH_CALUDE_min_value_expression_l774_77402

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = (1 / 2 : ℝ) ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 →
    (1 / a) - (4 * b / (b + 1)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l774_77402


namespace NUMINAMATH_CALUDE_cube_root_floor_equality_l774_77429

theorem cube_root_floor_equality (n : ℕ) :
  ⌊(n : ℝ)^(1/3) + (n + 1 : ℝ)^(1/3)⌋ = ⌊(8*n + 3 : ℝ)^(1/3)⌋ :=
sorry

end NUMINAMATH_CALUDE_cube_root_floor_equality_l774_77429


namespace NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l774_77463

/-- Represents a polyhedron with vertices, edges, and faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra: V - E + F = 2 -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- Each edge is shared by exactly two faces -/
axiom edge_face_relation (p : Polyhedron) : 2 * p.edges = 3 * p.faces

/-- Theorem: There is no polyhedron with exactly 7 edges -/
theorem no_polyhedron_with_seven_edges :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l774_77463


namespace NUMINAMATH_CALUDE_percent_of_x_l774_77401

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l774_77401


namespace NUMINAMATH_CALUDE_winning_strategy_works_l774_77494

/-- Represents the game state with blue and white balls --/
structure GameState where
  blue : ℕ
  white : ℕ

/-- Represents a player's move --/
inductive Move
  | TakeBlue
  | TakeWhite

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeBlue => { blue := state.blue - 3, white := state.white }
  | Move.TakeWhite => { blue := state.blue, white := state.white - 2 }

/-- Checks if the game is over (no balls left) --/
def isGameOver (state : GameState) : Prop :=
  state.blue = 0 ∧ state.white = 0

/-- Represents the winning strategy --/
def winningStrategy (state : GameState) : Prop :=
  3 * state.white = 2 * state.blue

/-- The main theorem to prove --/
theorem winning_strategy_works (initialState : GameState)
  (h_initial : initialState.blue = 15 ∧ initialState.white = 12) :
  ∃ (firstMove : Move),
    let stateAfterFirstMove := applyMove initialState firstMove
    winningStrategy stateAfterFirstMove ∧
    (∀ (opponentMove : Move),
      let stateAfterOpponent := applyMove stateAfterFirstMove opponentMove
      ∃ (response : Move),
        let stateAfterResponse := applyMove stateAfterOpponent response
        winningStrategy stateAfterResponse ∨ isGameOver stateAfterResponse) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_works_l774_77494


namespace NUMINAMATH_CALUDE_divisibility_by_thirty_l774_77445

theorem divisibility_by_thirty (n : ℕ) (h_prime : Nat.Prime n) (h_geq_7 : n ≥ 7) :
  30 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirty_l774_77445


namespace NUMINAMATH_CALUDE_sibling_ages_equations_l774_77466

/-- Represents the ages of two siblings -/
structure SiblingAges where
  x : ℕ  -- Age of the older brother
  y : ℕ  -- Age of the younger sister

/-- The conditions for the sibling ages problem -/
def SiblingAgesProblem (ages : SiblingAges) : Prop :=
  (ages.x = 4 * ages.y) ∧ 
  (ages.x + 3 = 3 * (ages.y + 3))

/-- The theorem stating that the given system of equations is correct -/
theorem sibling_ages_equations (ages : SiblingAges) :
  SiblingAgesProblem ages ↔ 
  (ages.x + 3 = 3 * (ages.y + 3)) ∧ (ages.x = 4 * ages.y) :=
sorry

end NUMINAMATH_CALUDE_sibling_ages_equations_l774_77466


namespace NUMINAMATH_CALUDE_expression_values_l774_77490

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  let e := x / |x| + y / |y| + z / |z| + (x*y*z) / |x*y*z|
  e = 4 ∨ e = 0 ∨ e = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l774_77490


namespace NUMINAMATH_CALUDE_helium_pressure_change_l774_77456

/-- Boyle's Law for ideal gases at constant temperature -/
axiom boyles_law {V1 P1 V2 P2 : ℝ} (hV1 : V1 > 0) (hP1 : P1 > 0) (hV2 : V2 > 0) (hP2 : P2 > 0) :
  V1 * P1 = V2 * P2

theorem helium_pressure_change (V1 P1 V2 P2 : ℝ) 
  (hV1 : V1 = 3.4) (hP1 : P1 = 8) (hV2 : V2 = 8.5) 
  (hV1pos : V1 > 0) (hP1pos : P1 > 0) (hV2pos : V2 > 0) (hP2pos : P2 > 0) :
  P2 = 3.2 := by
  sorry

#check helium_pressure_change

end NUMINAMATH_CALUDE_helium_pressure_change_l774_77456


namespace NUMINAMATH_CALUDE_f_value_at_one_l774_77488

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 120*x + c

-- State the theorem
theorem f_value_at_one (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c 1 = -3682.25 :=
sorry

end NUMINAMATH_CALUDE_f_value_at_one_l774_77488


namespace NUMINAMATH_CALUDE_line_segments_not_in_proportion_l774_77410

theorem line_segments_not_in_proportion :
  let a : ℝ := 4
  let b : ℝ := 5
  let c : ℝ := 6
  let d : ℝ := 10
  (a / b) ≠ (c / d) :=
by sorry

end NUMINAMATH_CALUDE_line_segments_not_in_proportion_l774_77410


namespace NUMINAMATH_CALUDE_tangent_line_circle_parabola_l774_77420

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the xy-plane -/
structure Parabola where
  vertex : ℝ × ℝ
  a : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle at a given point -/
def isTangentToCircle (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Checks if a line is tangent to a parabola at a given point -/
def isTangentToParabola (l : Line) (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_parabola (c : Circle) (p : Parabola) (l : Line) (point : ℝ × ℝ) :
  c.center = (1, 2) →
  c.radius^2 = 1^2 + 2^2 + a →
  p.vertex = (0, 0) →
  p.a = 1/4 →
  isTangentToCircle l c point →
  isTangentToParabola l p point →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_parabola_l774_77420


namespace NUMINAMATH_CALUDE_distinct_roots_isosceles_triangle_k_values_l774_77427

/-- The quadratic equation x^2 - (2k+1)x + k^2 + k = 0 has two distinct real roots for all k -/
theorem distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*k+1)*x₁ + k^2 + k = 0 ∧ x₂^2 - (2*k+1)*x₂ + k^2 + k = 0 :=
sorry

/-- When two roots of x^2 - (2k+1)x + k^2 + k = 0 form two sides of an isosceles triangle 
    with the third side of length 5, k = 4 or k = 5 -/
theorem isosceles_triangle_k_values :
  ∃ x₁ x₂ : ℝ, 
    x₁^2 - (2*4+1)*x₁ + 4^2 + 4 = 0 ∧
    x₂^2 - (2*4+1)*x₂ + 4^2 + 4 = 0 ∧
    ((x₁ = 5 ∧ x₂ = x₁) ∨ (x₂ = 5 ∧ x₁ = x₂))
  ∧
  ∃ y₁ y₂ : ℝ,
    y₁^2 - (2*5+1)*y₁ + 5^2 + 5 = 0 ∧
    y₂^2 - (2*5+1)*y₂ + 5^2 + 5 = 0 ∧
    ((y₁ = 5 ∧ y₂ = y₁) ∨ (y₂ = 5 ∧ y₁ = y₂))
  ∧
  ∀ k : ℝ, k ≠ 4 → k ≠ 5 →
    ¬∃ z₁ z₂ : ℝ,
      z₁^2 - (2*k+1)*z₁ + k^2 + k = 0 ∧
      z₂^2 - (2*k+1)*z₂ + k^2 + k = 0 ∧
      ((z₁ = 5 ∧ z₂ = z₁) ∨ (z₂ = 5 ∧ z₁ = z₂)) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_isosceles_triangle_k_values_l774_77427


namespace NUMINAMATH_CALUDE_diagonal_cut_color_distribution_l774_77484

/-- Represents the color distribution of a scarf --/
structure ColorDistribution where
  white : ℚ
  grey : ℚ
  black : ℚ

/-- Represents a square scarf --/
structure SquareScarf where
  side_length : ℚ
  black_area : ℚ
  grey_area : ℚ

/-- Represents a triangular scarf obtained by cutting a square scarf diagonally --/
structure TriangularScarf where
  color_distribution : ColorDistribution

def diagonal_cut (s : SquareScarf) : (TriangularScarf × TriangularScarf) :=
  sorry

theorem diagonal_cut_color_distribution 
  (s : SquareScarf) 
  (h1 : s.black_area = 1/6) 
  (h2 : s.grey_area = 1/3) :
  let (t1, t2) := diagonal_cut s
  t1.color_distribution = { white := 3/4, grey := 2/9, black := 1/36 } ∧
  t2.color_distribution = { white := 1/4, grey := 4/9, black := 11/36 } :=
sorry

end NUMINAMATH_CALUDE_diagonal_cut_color_distribution_l774_77484


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l774_77415

theorem complex_modulus_problem (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l774_77415


namespace NUMINAMATH_CALUDE_eunice_total_seeds_l774_77406

/-- The number of eggplant seeds Eunice planted in the first pot -/
def seeds_in_first_pot : ℕ := 3

/-- The number of eggplant seeds Eunice planted in the fourth pot -/
def seeds_in_fourth_pot : ℕ := 1

/-- The total number of eggplant seeds Eunice has -/
def total_seeds : ℕ := seeds_in_first_pot + seeds_in_fourth_pot

/-- Theorem stating that the total number of eggplant seeds Eunice has is 4 -/
theorem eunice_total_seeds : total_seeds = 4 := by
  sorry

end NUMINAMATH_CALUDE_eunice_total_seeds_l774_77406


namespace NUMINAMATH_CALUDE_symmetry_line_l774_77467

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry condition
def is_symmetric (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (4 - x)

-- Define the line of symmetry
def line_of_symmetry (g : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, g (k + (x - k)) = g (k - (x - k))

-- Theorem statement
theorem symmetry_line (g : ℝ → ℝ) (h : is_symmetric g) :
  line_of_symmetry g 2 :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_l774_77467


namespace NUMINAMATH_CALUDE_sin_beta_value_l774_77422

theorem sin_beta_value (α β : Real) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l774_77422


namespace NUMINAMATH_CALUDE_city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l774_77480

/-- Represents a survey scenario -/
inductive SurveyScenario
  | class_myopia
  | grade_morning_exercise
  | class_body_temperature
  | city_extracurricular_reading

/-- Determines if a survey scenario is suitable for sampling -/
def suitable_for_sampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.city_extracurricular_reading => True
  | _ => False

/-- Theorem stating that the city-wide extracurricular reading survey is suitable for sampling -/
theorem city_reading_survey_suitable :
  suitable_for_sampling SurveyScenario.city_extracurricular_reading :=
by
  sorry

/-- Theorem stating that the class myopia survey is not suitable for sampling -/
theorem class_myopia_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_myopia :=
by
  sorry

/-- Theorem stating that the grade morning exercise survey is not suitable for sampling -/
theorem grade_exercise_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.grade_morning_exercise :=
by
  sorry

/-- Theorem stating that the class body temperature survey is not suitable for sampling -/
theorem class_temperature_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_body_temperature :=
by
  sorry

end NUMINAMATH_CALUDE_city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l774_77480


namespace NUMINAMATH_CALUDE_solve_parking_lot_l774_77440

def parking_lot (num_bikes : ℕ) (total_wheels : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : Prop :=
  ∃ (num_cars : ℕ), 
    num_cars * wheels_per_car + num_bikes * wheels_per_bike = total_wheels

theorem solve_parking_lot : 
  parking_lot 5 66 4 2 → ∃ (num_cars : ℕ), num_cars = 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_parking_lot_l774_77440


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l774_77458

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (a1 a2 b1 b2 : ℝ),
    circle1 a1 a2 ∧ circle1 b1 b2 ∧
    circle2 a1 a2 ∧ circle2 b1 b2 ∧
    (∀ x y : ℝ, perp_bisector x y ↔ 
      ((x - a1)^2 + (y - a2)^2 = (x - b1)^2 + (y - b2)^2)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l774_77458


namespace NUMINAMATH_CALUDE_decimal_representation_theorem_l774_77470

theorem decimal_representation_theorem (n m : ℕ) (h1 : n > m) (h2 : m ≥ 1) 
  (h3 : ∃ k : ℕ, ∃ p : ℕ, 0 < p ∧ p < n ∧ 
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 ≥ 143 ∧
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 < 144) :
  n > 125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_theorem_l774_77470


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l774_77414

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the complement of B in the universal set (real numbers)
def C_U_B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ C_U_B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l774_77414


namespace NUMINAMATH_CALUDE_base9_85_to_decimal_l774_77432

/-- Converts a two-digit number in base 9 to its decimal representation -/
def base9ToDecimal (tens : Nat) (ones : Nat) : Nat :=
  tens * 9^1 + ones * 9^0

/-- Theorem stating that 85 in base 9 is equal to 77 in decimal -/
theorem base9_85_to_decimal : base9ToDecimal 8 5 = 77 := by
  sorry

end NUMINAMATH_CALUDE_base9_85_to_decimal_l774_77432


namespace NUMINAMATH_CALUDE_contractor_fine_proof_l774_77430

/-- Calculates the daily fine for absence given contract parameters -/
def calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_pay
  (total_earned - total_received) / days_absent

/-- Proves that the daily fine is 7.50 given the contract parameters -/
theorem contractor_fine_proof :
  calculate_daily_fine 30 25 620 4 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_proof_l774_77430


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l774_77421

def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (15 - x)) →
  p a b c 4 = -4 →
  p a b c 11 = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l774_77421


namespace NUMINAMATH_CALUDE_graduating_class_male_percentage_l774_77452

theorem graduating_class_male_percentage :
  ∀ (M F : ℝ),
  M + F = 100 →
  0.5 * M + 0.7 * F = 62 →
  M = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_graduating_class_male_percentage_l774_77452


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l774_77476

theorem arithmetic_sequence_problem :
  ∀ a d : ℝ,
  (a - d) + a + (a + d) = 6 →
  (a - d) * a * (a + d) = -10 →
  ((a - d = 5 ∧ a = 2 ∧ a + d = -1) ∨ (a - d = -1 ∧ a = 2 ∧ a + d = 5)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l774_77476


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l774_77453

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize - populationSize % sampleSize) / sampleSize

theorem systematic_sampling_interval :
  samplingInterval 1003 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l774_77453


namespace NUMINAMATH_CALUDE_long_division_puzzle_l774_77462

theorem long_division_puzzle :
  (631938 : ℚ) / 625 = 1011.1008 := by
  sorry

end NUMINAMATH_CALUDE_long_division_puzzle_l774_77462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l774_77400

/-- The number of terms in an arithmetic sequence from -3 to 53 -/
theorem arithmetic_sequence_length : ∀ (a d : ℤ), 
  a = -3 → 
  d = 4 → 
  ∃ n : ℕ, n > 0 ∧ a + (n - 1) * d = 53 → 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l774_77400


namespace NUMINAMATH_CALUDE_no_product_equality_l774_77411

def a : ℕ → ℤ
  | 0 => 2
  | 1 => 5
  | (n + 2) => (2 - n^2) * a (n + 1) + (2 + n^2) * a n

theorem no_product_equality : ¬∃ (p q r : ℕ+), a p.val * a q.val = a r.val := by
  sorry

end NUMINAMATH_CALUDE_no_product_equality_l774_77411


namespace NUMINAMATH_CALUDE_factorial_sum_equals_1190_l774_77460

theorem factorial_sum_equals_1190 : 
  (Nat.factorial 16) / ((Nat.factorial 6) * (Nat.factorial 10)) + 
  (Nat.factorial 11) / ((Nat.factorial 6) * (Nat.factorial 5)) = 1190 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_1190_l774_77460


namespace NUMINAMATH_CALUDE_intersection_on_diagonal_l774_77478

-- Define the basic geometric objects
variable (A B C D K L M P Q : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define points K, L, M on sides or their extensions
def on_side_or_extension (X Y Z : EuclideanPlane) : Prop := sorry

-- Define the intersection of two lines
def intersect (W X Y Z : EuclideanPlane) : EuclideanPlane := sorry

-- Define a point lying on a line
def lies_on (X Y Z : EuclideanPlane) : Prop := sorry

-- Theorem statement
theorem intersection_on_diagonal 
  (h_quad : is_quadrilateral A B C D)
  (h_K : on_side_or_extension K A B)
  (h_L : on_side_or_extension L B C)
  (h_M : on_side_or_extension M C D)
  (h_P : P = intersect K L A C)
  (h_Q : Q = intersect L M B D) :
  lies_on (intersect K Q M P) A D := by sorry

end NUMINAMATH_CALUDE_intersection_on_diagonal_l774_77478


namespace NUMINAMATH_CALUDE_angle_complement_supplement_relation_l774_77424

/-- 
Given an angle x in degrees, if its complement (90° - x) is 75% of its supplement (180° - x), 
then x = 180°.
-/
theorem angle_complement_supplement_relation (x : ℝ) : 
  (90 - x) = 0.75 * (180 - x) → x = 180 := by sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_relation_l774_77424


namespace NUMINAMATH_CALUDE_cos_sin_sum_l774_77457

theorem cos_sin_sum (x : ℝ) (h : Real.cos (x - π/3) = 1/3) :
  Real.cos (2*x - 5*π/3) + Real.sin (π/3 - x)^2 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l774_77457


namespace NUMINAMATH_CALUDE_smallest_factor_product_l774_77495

theorem smallest_factor_product (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 1764 → ¬(2^5 ∣ 936 * m ∧ 3^3 ∣ 936 * m ∧ 14^2 ∣ 936 * m)) ∧
  (2^5 ∣ 936 * 1764 ∧ 3^3 ∣ 936 * 1764 ∧ 14^2 ∣ 936 * 1764) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_product_l774_77495


namespace NUMINAMATH_CALUDE_table_price_is_84_l774_77426

/-- Represents the price of items in a store --/
structure StorePrice where
  chair : ℝ
  table : ℝ
  lamp : ℝ

/-- Conditions for the store pricing problem --/
def StorePricingConditions (p : StorePrice) : Prop :=
  (2 * p.chair + p.table = 0.6 * (p.chair + 2 * p.table)) ∧
  (p.chair + p.table = 96) ∧
  (p.lamp + p.chair = 0.5 * (2 * p.table + p.lamp))

/-- Theorem stating that under the given conditions, the price of a table is $84 --/
theorem table_price_is_84 (p : StorePrice) 
  (h : StorePricingConditions p) : p.table = 84 := by
  sorry

end NUMINAMATH_CALUDE_table_price_is_84_l774_77426


namespace NUMINAMATH_CALUDE_tank_capacity_l774_77475

/-- Proves that the capacity of a tank filled by two buckets of 4 and 3 liters,
    where the 3-liter bucket is used 4 more times than the 4-liter bucket, is 48 liters. -/
theorem tank_capacity (x : ℕ) : 
  (4 * x = 3 * (x + 4)) → (4 * x = 48) := by sorry

end NUMINAMATH_CALUDE_tank_capacity_l774_77475


namespace NUMINAMATH_CALUDE_parabola_inequality_l774_77477

/-- Prove that for a parabola y = ax^2 + bx + c with a < 0, passing through points (-1, 0) and (m, 0) where 3 < m < 4, the inequality 3a + c > 0 holds. -/
theorem parabola_inequality (a b c m : ℝ) : 
  a < 0 → 
  3 < m → 
  m < 4 → 
  a * (-1)^2 + b * (-1) + c = 0 → 
  a * m^2 + b * m + c = 0 → 
  3 * a + c > 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_inequality_l774_77477


namespace NUMINAMATH_CALUDE_automobile_dealer_revenue_l774_77498

/-- Represents the revenue calculation for an automobile dealer's sale --/
theorem automobile_dealer_revenue :
  ∀ (num_suvs : ℕ),
    num_suvs + (num_suvs + 50) + (2 * num_suvs) = 150 →
    20000 * (num_suvs + 50) + 30000 * (2 * num_suvs) + 40000 * num_suvs = 4000000 :=
by
  sorry

end NUMINAMATH_CALUDE_automobile_dealer_revenue_l774_77498


namespace NUMINAMATH_CALUDE_christmas_decorations_l774_77469

theorem christmas_decorations (boxes : ℕ) (used : ℕ) (given_away : ℕ) : 
  boxes = 4 → used = 35 → given_away = 25 → (used + given_away) / boxes = 15 := by
  sorry

end NUMINAMATH_CALUDE_christmas_decorations_l774_77469


namespace NUMINAMATH_CALUDE_clock_rings_107_times_in_january_l774_77471

/-- Calculates the number of times a clock rings in January -/
def clock_rings_in_january (ring_interval : ℕ) (days_in_january : ℕ) : ℕ :=
  let hours_in_january := days_in_january * 24
  (hours_in_january / ring_interval) + 1

/-- Theorem: A clock that rings every 7 hours will ring 107 times in January -/
theorem clock_rings_107_times_in_january :
  clock_rings_in_january 7 31 = 107 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_107_times_in_january_l774_77471


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_power_sum_l774_77489

theorem prime_pairs_dividing_power_sum :
  ∀ p q : ℕ,
  Nat.Prime p → Nat.Prime q →
  (p * q ∣ 2^p + 2^q) ↔ ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_power_sum_l774_77489


namespace NUMINAMATH_CALUDE_bromine_only_liquid_l774_77408

-- Define the set of elements
inductive Element : Type
| Bromine : Element
| Krypton : Element
| Phosphorus : Element
| Xenon : Element

-- Define the state of matter
inductive State : Type
| Solid : State
| Liquid : State
| Gas : State

-- Define the function to determine the state of an element at given temperature and pressure
def stateAtConditions (e : Element) (temp : ℝ) (pressure : ℝ) : State := sorry

-- Define the temperature and pressure conditions
def roomTemp : ℝ := 25
def atmPressure : ℝ := 1.0

-- Theorem statement
theorem bromine_only_liquid :
  ∀ e : Element, 
    stateAtConditions e roomTemp atmPressure = State.Liquid ↔ e = Element.Bromine :=
sorry

end NUMINAMATH_CALUDE_bromine_only_liquid_l774_77408


namespace NUMINAMATH_CALUDE_sons_ages_l774_77428

def father_age : ℕ := 33
def youngest_son_age : ℕ := 2
def years_until_sum_equal : ℕ := 12

def is_valid_ages (middle_son_age oldest_son_age : ℕ) : Prop :=
  (father_age + years_until_sum_equal = 
   (youngest_son_age + years_until_sum_equal) + 
   (middle_son_age + years_until_sum_equal) + 
   (oldest_son_age + years_until_sum_equal)) ∧
  (middle_son_age > youngest_son_age) ∧
  (oldest_son_age > middle_son_age)

theorem sons_ages : 
  ∃ (middle_son_age oldest_son_age : ℕ),
    is_valid_ages middle_son_age oldest_son_age ∧
    middle_son_age = 3 ∧ oldest_son_age = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sons_ages_l774_77428


namespace NUMINAMATH_CALUDE_swimming_frequency_l774_77454

def runs_every : ℕ := 4
def cycles_every : ℕ := 16
def all_activities_every : ℕ := 48

theorem swimming_frequency :
  ∃ (swims_every : ℕ),
    swims_every > 0 ∧
    (Nat.lcm swims_every runs_every = Nat.lcm (Nat.lcm swims_every runs_every) cycles_every) ∧
    Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = all_activities_every ∧
    swims_every = 3 := by
  sorry

end NUMINAMATH_CALUDE_swimming_frequency_l774_77454


namespace NUMINAMATH_CALUDE_carousel_horse_ratio_l774_77441

/-- The carousel problem -/
theorem carousel_horse_ratio :
  let blue_horses : ℕ := 3
  let purple_horses : ℕ := 3 * blue_horses
  let green_horses : ℕ := 2 * purple_horses
  let total_horses : ℕ := 33
  let gold_horses : ℕ := total_horses - (blue_horses + purple_horses + green_horses)
  (gold_horses : ℚ) / green_horses = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_carousel_horse_ratio_l774_77441


namespace NUMINAMATH_CALUDE_oranges_to_put_back_correct_l774_77431

/-- Represents the number of oranges to put back -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruits : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℕ :=
  6

theorem oranges_to_put_back_correct 
  (apple_price : ℚ) 
  (orange_price : ℚ) 
  (total_fruits : ℕ) 
  (initial_avg_price : ℚ) 
  (desired_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : initial_avg_price = 54/100)
  (h5 : desired_avg_price = 45/100) :
  ∃ (A O : ℕ), 
    A + O = total_fruits ∧ 
    (apple_price * A + orange_price * O) / total_fruits = initial_avg_price ∧
    (apple_price * A + orange_price * (O - oranges_to_put_back apple_price orange_price total_fruits initial_avg_price desired_avg_price)) / 
      (total_fruits - oranges_to_put_back apple_price orange_price total_fruits initial_avg_price desired_avg_price) = desired_avg_price :=
by sorry

#check oranges_to_put_back_correct

end NUMINAMATH_CALUDE_oranges_to_put_back_correct_l774_77431


namespace NUMINAMATH_CALUDE_fantasy_ball_handshakes_l774_77496

/-- The number of goblins attending the Fantasy Creatures Ball -/
def num_goblins : ℕ := 30

/-- The number of pixies attending the Fantasy Creatures Ball -/
def num_pixies : ℕ := 10

/-- Represents whether pixies can shake hands with a given number of goblins -/
def pixie_can_shake (n : ℕ) : Prop := Even n

/-- Calculates the number of handshakes between goblins -/
def goblin_handshakes (n : ℕ) : ℕ := n.choose 2

/-- Calculates the number of handshakes between goblins and pixies -/
def goblin_pixie_handshakes (g p : ℕ) : ℕ := g * p

/-- The total number of handshakes at the Fantasy Creatures Ball -/
def total_handshakes : ℕ := goblin_handshakes num_goblins + goblin_pixie_handshakes num_goblins num_pixies

theorem fantasy_ball_handshakes :
  pixie_can_shake num_goblins →
  total_handshakes = 735 := by
  sorry

end NUMINAMATH_CALUDE_fantasy_ball_handshakes_l774_77496


namespace NUMINAMATH_CALUDE_min_value_of_f_l774_77443

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l774_77443


namespace NUMINAMATH_CALUDE_binary_op_three_seven_l774_77465

def binary_op (c d : ℤ) : ℤ := 4 * c + 3 * d - c * d

theorem binary_op_three_seven : binary_op 3 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binary_op_three_seven_l774_77465


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l774_77403

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is approximately 74.094 g/mol -/
theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 74.094| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l774_77403


namespace NUMINAMATH_CALUDE_sarahs_waist_cm_l774_77451

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℝ := 2.54

-- Define Sarah's waist size in inches
def sarahs_waist_inches : ℝ := 27

-- Theorem to prove Sarah's waist size in centimeters
theorem sarahs_waist_cm : 
  ∃ (waist_cm : ℝ), abs (waist_cm - (sarahs_waist_inches * inches_to_cm)) < 0.05 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_waist_cm_l774_77451


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_problem_l774_77446

/-- Calculates the number of pens given to Sharon -/
def pens_given_to_sharon (initial_pens : ℕ) (mike_pens : ℕ) (final_pens : ℕ) : ℕ :=
  2 * (initial_pens + mike_pens) - final_pens

theorem pens_given_to_sharon_problem :
  let initial_pens : ℕ := 5
  let mike_pens : ℕ := 20
  let final_pens : ℕ := 31
  pens_given_to_sharon initial_pens mike_pens final_pens = 19 := by
  sorry

#eval pens_given_to_sharon 5 20 31

end NUMINAMATH_CALUDE_pens_given_to_sharon_problem_l774_77446


namespace NUMINAMATH_CALUDE_plumber_pipe_cost_l774_77444

/-- The total cost of copper and plastic pipe given specific quantities and prices -/
theorem plumber_pipe_cost (copper_length : ℕ) (plastic_length : ℕ) 
  (copper_price : ℕ) (plastic_price : ℕ) : 
  copper_length = 10 → 
  plastic_length = 15 → 
  copper_price = 5 → 
  plastic_price = 3 → 
  copper_length * copper_price + plastic_length * plastic_price = 95 := by
  sorry

#check plumber_pipe_cost

end NUMINAMATH_CALUDE_plumber_pipe_cost_l774_77444


namespace NUMINAMATH_CALUDE_constant_term_implies_a_equals_one_l774_77436

theorem constant_term_implies_a_equals_one :
  ∀ (a : ℝ),
  (∃ (c : ℝ), c = 80 ∧ 
    c = (5 : ℕ).choose 4 * 2^4 * a * (x : ℝ)^(10 - (5 * 4) / 2) ∧
    (∀ (r : ℕ), r ≠ 4 → 
      (5 : ℕ).choose r * 2^r * a^(5-r) * (x : ℝ)^(10 - (5 * r) / 2) ≠ c)) →
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_constant_term_implies_a_equals_one_l774_77436


namespace NUMINAMATH_CALUDE_influenza_virus_diameter_l774_77425

theorem influenza_virus_diameter (n : ℤ) : 0.000000203 = 2.03 * (10 : ℝ) ^ n → n = -7 := by
  sorry

end NUMINAMATH_CALUDE_influenza_virus_diameter_l774_77425


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l774_77461

theorem triangle_cosine_theorem (A : ℝ) (a m : ℝ) (θ : ℝ) 
  (h1 : A = 40) 
  (h2 : a = 12) 
  (h3 : m = 10) 
  (h4 : A = 1/2 * a * m * Real.sin θ) 
  (h5 : 0 < θ) 
  (h6 : θ < π/2) : 
  Real.cos θ = Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l774_77461


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l774_77434

theorem cone_lateral_surface_area 
  (r : Real) 
  (l : Real) 
  (h_r : r = Real.sqrt 2) 
  (h_l : l = 3 * Real.sqrt 2) : 
  r * l * Real.pi = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l774_77434


namespace NUMINAMATH_CALUDE_orchids_cut_l774_77492

theorem orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 16)
  (h2 : initial_orchids = 3)
  (h3 : final_roses = 13)
  (h4 : final_orchids = 7) :
  final_orchids - initial_orchids = 4 := by
  sorry

end NUMINAMATH_CALUDE_orchids_cut_l774_77492


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l774_77468

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) :
  (∀ n : ℕ, a * r^n = 3 * (a * r^(n+1) + a * r^(n+2))) →
  (∀ n : ℕ, a * r^n > 0) →
  r = (-1 + Real.sqrt (7/3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l774_77468


namespace NUMINAMATH_CALUDE_complex_modulus_l774_77442

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = 4 + 3 * Complex.I) : 
  Complex.abs (z - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l774_77442


namespace NUMINAMATH_CALUDE_exists_n_with_1000_steps_l774_77437

def largest_prime_le (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).max' sorry

def reduction_process (n : ℕ) : ℕ → ℕ
| 0 => 0
| (k + 1) => 
  let n' := n - largest_prime_le n
  if n' ≤ 1 then n' else reduction_process n' k

theorem exists_n_with_1000_steps : 
  ∃ N : ℕ, reduction_process N 1000 = 0 ∧ ∀ k < 1000, reduction_process N k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_exists_n_with_1000_steps_l774_77437


namespace NUMINAMATH_CALUDE_exponent_unchanged_l774_77479

/-- Represents a term in an algebraic expression -/
structure Term where
  coefficient : ℝ
  letter : Char
  exponent : ℕ

/-- Combines two like terms -/
def combineLikeTerms (t1 t2 : Term) : Term :=
  { coefficient := t1.coefficient + t2.coefficient,
    letter := t1.letter,
    exponent := t1.exponent }

/-- Theorem stating that the exponent remains unchanged when combining like terms -/
theorem exponent_unchanged (t1 t2 : Term) (h : t1.letter = t2.letter) :
  (combineLikeTerms t1 t2).exponent = t1.exponent :=
by sorry

end NUMINAMATH_CALUDE_exponent_unchanged_l774_77479


namespace NUMINAMATH_CALUDE_circle_center_is_neg_two_three_l774_77485

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y + 1 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle with the given equation is (-2, 3) -/
theorem circle_center_is_neg_two_three :
  ∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 12 :=
sorry

end NUMINAMATH_CALUDE_circle_center_is_neg_two_three_l774_77485


namespace NUMINAMATH_CALUDE_product_sequence_equals_32_l774_77412

theorem product_sequence_equals_32 : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_equals_32_l774_77412


namespace NUMINAMATH_CALUDE_product_of_constrained_integers_l774_77435

theorem product_of_constrained_integers (a b : ℕ) 
  (h1 : 90 < a + b ∧ a + b < 99)
  (h2 : (9 : ℚ)/10 < (a : ℚ)/(b : ℚ) ∧ (a : ℚ)/(b : ℚ) < (91 : ℚ)/100) :
  a * b = 2346 := by
  sorry

end NUMINAMATH_CALUDE_product_of_constrained_integers_l774_77435


namespace NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l774_77419

/-- The sum of exterior angles of a pentagon is 360 degrees -/
theorem sum_exterior_angles_pentagon : ℝ :=
  360

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Pentagon) : ℝ := 360

theorem sum_exterior_angles_pentagon_is_360 (p : Pentagon) :
  sum_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_pentagon_sum_exterior_angles_pentagon_is_360_l774_77419


namespace NUMINAMATH_CALUDE_commercial_reduction_percentage_l774_77483

theorem commercial_reduction_percentage 
  (original_length : ℝ) 
  (shortened_length : ℝ) 
  (h1 : original_length = 30) 
  (h2 : shortened_length = 21) : 
  (original_length - shortened_length) / original_length * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_commercial_reduction_percentage_l774_77483


namespace NUMINAMATH_CALUDE_quadratic_transformation_l774_77438

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 3 * (x - 5)^2 + 9) →
  ∃ m k, ∀ x, 2 * (a * x^2 + b * x + c) = m * (x - 5)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l774_77438


namespace NUMINAMATH_CALUDE_two_zeros_cubic_l774_77486

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) → 
  c = -2 ∨ c = 2 := by
sorry

end NUMINAMATH_CALUDE_two_zeros_cubic_l774_77486


namespace NUMINAMATH_CALUDE_no_real_solutions_l774_77405

theorem no_real_solutions :
  ¬∃ (x y z u : ℝ), x^4 - 17 = y^4 - 7 ∧ 
                    x^4 - 17 = z^4 + 19 ∧ 
                    x^4 - 17 = u^4 + 5 ∧ 
                    x^4 - 17 = x * y * z * u :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l774_77405


namespace NUMINAMATH_CALUDE_special_function_value_l774_77474

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ x < y → f x ≤ f y) ∧ 
  (∀ x, 0 < x → f (f x - 1/x) = 2)

theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) : f (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l774_77474


namespace NUMINAMATH_CALUDE_sum_x_y_equals_nine_l774_77459

theorem sum_x_y_equals_nine (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 4) : 
  x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_nine_l774_77459


namespace NUMINAMATH_CALUDE_final_digit_is_nine_l774_77455

/-- Represents the sequence of digits formed by concatenating numbers from 1 to 1995 -/
def initial_sequence : List Nat := sorry

/-- Removes digits at even positions from a list of digits -/
def remove_even_positions (digits : List Nat) : List Nat := sorry

/-- Removes digits at odd positions from a list of digits -/
def remove_odd_positions (digits : List Nat) : List Nat := sorry

/-- Applies the alternating removal process until one digit remains -/
def process_sequence (digits : List Nat) : Nat := sorry

theorem final_digit_is_nine : 
  process_sequence initial_sequence = 9 := by sorry

end NUMINAMATH_CALUDE_final_digit_is_nine_l774_77455


namespace NUMINAMATH_CALUDE_percent_teachers_without_conditions_l774_77416

theorem percent_teachers_without_conditions (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : high_bp = 90)
  (h3 : heart_trouble = 50)
  (h4 : both = 30) :
  (((total - (high_bp + heart_trouble - both)) : ℚ) / total) * 100 = 2667 / 100 :=
sorry

end NUMINAMATH_CALUDE_percent_teachers_without_conditions_l774_77416


namespace NUMINAMATH_CALUDE_expression_equivalence_l774_77450

theorem expression_equivalence :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l774_77450


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l774_77493

theorem age_ratio_in_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 14 →
  man_age = son_age + 16 →
  ∃ k : ℕ, (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l774_77493


namespace NUMINAMATH_CALUDE_race_distance_l774_77417

/-- The race distance in meters -/
def d : ℝ := 75

/-- The speed of runner X -/
def x : ℝ := sorry

/-- The speed of runner Y -/
def y : ℝ := sorry

/-- The speed of runner Z -/
def z : ℝ := sorry

/-- Theorem stating that d is the correct race distance -/
theorem race_distance : 
  (d / x = (d - 25) / y) ∧ 
  (d / y = (d - 15) / z) ∧ 
  (d / x = (d - 35) / z) → 
  d = 75 := by sorry

end NUMINAMATH_CALUDE_race_distance_l774_77417


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l774_77409

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1/2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧ 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    1/2 * a' * b' * Real.sin C ≤ 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l774_77409


namespace NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l774_77433

theorem equal_area_dividing_line_slope (r : ℝ) (c1 c2 p : ℝ × ℝ) (m : ℝ) : 
  r = 4 ∧ 
  c1 = (0, 20) ∧ 
  c2 = (6, 12) ∧ 
  p = (4, 0) ∧
  (∀ (x y : ℝ), y = m * (x - p.1) + p.2) ∧
  (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 → 
    (m * x - y + (p.2 - m * p.1))^2 / (m^2 + 1) = 
    (m * c2.1 - c2.2 + (p.2 - m * p.1))^2 / (m^2 + 1)) →
  |m| = 4/3 := by
sorry

end NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l774_77433


namespace NUMINAMATH_CALUDE_expand_expression_l774_77491

theorem expand_expression (x y z : ℝ) : 
  (2 * x + 5) * (3 * y + 4 * z + 15) = 6 * x * y + 8 * x * z + 30 * x + 15 * y + 20 * z + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l774_77491


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_holds_l774_77497

/-- In a parallelogram, given that one angle measures 85 degrees, 
    the difference between this angle and its adjacent angle is 10 degrees. -/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun angle_difference : ℝ =>
    ∀ (smaller_angle larger_angle : ℝ),
      smaller_angle = 85 ∧
      smaller_angle + larger_angle = 180 →
      larger_angle - smaller_angle = angle_difference ∧
      angle_difference = 10

/-- The theorem holds for the given angle difference. -/
theorem parallelogram_angle_difference_holds : parallelogram_angle_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_holds_l774_77497


namespace NUMINAMATH_CALUDE_apple_distribution_l774_77447

theorem apple_distribution (total_apples : ℕ) (ratio_1_2 ratio_1_3 ratio_2_3 : ℚ) :
  total_apples = 169 →
  ratio_1_2 = 1 / 2 →
  ratio_1_3 = 1 / 3 →
  ratio_2_3 = 1 / 2 →
  ∃ (boy1 boy2 boy3 : ℕ),
    boy1 + boy2 + boy3 = total_apples ∧
    boy1 = 78 ∧
    boy2 = 52 ∧
    boy3 = 39 ∧
    (boy1 : ℚ) / (boy2 : ℚ) = ratio_1_2 ∧
    (boy1 : ℚ) / (boy3 : ℚ) = ratio_1_3 ∧
    (boy2 : ℚ) / (boy3 : ℚ) = ratio_2_3 :=
by
  sorry

#check apple_distribution

end NUMINAMATH_CALUDE_apple_distribution_l774_77447


namespace NUMINAMATH_CALUDE_smallest_positive_t_l774_77473

theorem smallest_positive_t (x₁ x₂ x₃ x₄ x₅ t : ℝ) : 
  (x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) →
  (x₁ + x₂ + x₃ + x₄ + x₅ > 0) →
  (x₁ + x₃ = 2 * t * x₂) →
  (x₂ + x₄ = 2 * t * x₃) →
  (x₃ + x₅ = 2 * t * x₄) →
  t > 0 →
  t ≥ 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_t_l774_77473


namespace NUMINAMATH_CALUDE_expected_value_is_three_l774_77487

/-- Represents the outcome of rolling a six-sided dice -/
inductive DiceOutcome
  | Two
  | Five
  | Other

/-- The probability of each dice outcome -/
def probability (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 1/4
  | DiceOutcome.Five => 1/2
  | DiceOutcome.Other => 1/12

/-- The payoff for each dice outcome in dollars -/
def payoff (outcome : DiceOutcome) : ℚ :=
  match outcome with
  | DiceOutcome.Two => 4
  | DiceOutcome.Five => 6
  | DiceOutcome.Other => -3

/-- The expected value of rolling the dice once -/
def expectedValue : ℚ :=
  (probability DiceOutcome.Two * payoff DiceOutcome.Two) +
  (probability DiceOutcome.Five * payoff DiceOutcome.Five) +
  (4 * probability DiceOutcome.Other * payoff DiceOutcome.Other)

theorem expected_value_is_three :
  expectedValue = 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_three_l774_77487


namespace NUMINAMATH_CALUDE_factorial_divisibility_l774_77404

theorem factorial_divisibility (m n : ℕ) : 
  (m.factorial * n.factorial * (m + n).factorial) ∣ ((2 * m).factorial * (2 * n).factorial) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l774_77404


namespace NUMINAMATH_CALUDE_system_equation_ratio_l774_77481

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l774_77481


namespace NUMINAMATH_CALUDE_inequalities_with_negative_numbers_l774_77448

theorem inequalities_with_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  (a^2 > b^2) ∧ (a*b > b^2) ∧ (1/a > 1/b) ∧ (1/(a+b) > 1/a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_negative_numbers_l774_77448


namespace NUMINAMATH_CALUDE_fifty_seventh_digit_of_1_13_l774_77472

def decimal_rep_1_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem fifty_seventh_digit_of_1_13 : 
  (decimal_rep_1_13[(57 - 1) % decimal_rep_1_13.length] = 6) := by
  sorry

end NUMINAMATH_CALUDE_fifty_seventh_digit_of_1_13_l774_77472


namespace NUMINAMATH_CALUDE_combined_cost_price_theorem_l774_77423

def stock_price_1 : ℝ := 100
def stock_price_2 : ℝ := 150
def stock_price_3 : ℝ := 200

def discount_1 : ℝ := 0.06
def discount_2 : ℝ := 0.10
def discount_3 : ℝ := 0.07

def brokerage_1 : ℝ := 0.015
def brokerage_2 : ℝ := 0.02
def brokerage_3 : ℝ := 0.025

def taxation_rate : ℝ := 0.15

def combined_cost_price : ℝ :=
  let discounted_price_1 := stock_price_1 * (1 - discount_1)
  let discounted_price_2 := stock_price_2 * (1 - discount_2)
  let discounted_price_3 := stock_price_3 * (1 - discount_3)
  let cost_price_1 := discounted_price_1 * (1 + brokerage_1)
  let cost_price_2 := discounted_price_2 * (1 + brokerage_2)
  let cost_price_3 := discounted_price_3 * (1 + brokerage_3)
  let total_investing_amount := cost_price_1 + cost_price_2 + cost_price_3
  total_investing_amount * (1 + taxation_rate)

theorem combined_cost_price_theorem : combined_cost_price = 487.324 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_price_theorem_l774_77423


namespace NUMINAMATH_CALUDE_intercepts_sum_l774_77413

theorem intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 42 → y₀ < 42 → 
  (5 * x₀) % 42 = 40 → 
  (3 * y₀) % 42 = 2 → 
  x₀ + y₀ = 36 := by
sorry

end NUMINAMATH_CALUDE_intercepts_sum_l774_77413


namespace NUMINAMATH_CALUDE_number_difference_l774_77418

theorem number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 4 * x = 10) (h3 : x ≤ y) : y - x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l774_77418


namespace NUMINAMATH_CALUDE_rectangle_area_preservation_l774_77499

theorem rectangle_area_preservation (L W : ℝ) (x : ℝ) (h : x > 0) :
  L * W = L * (1 - x / 100) * W * (1 + 11.111111111111107 / 100) →
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_preservation_l774_77499


namespace NUMINAMATH_CALUDE_complex_equation_solution_l774_77449

theorem complex_equation_solution (z : ℂ) :
  (Complex.I / (z - 1) = (1 : ℂ) / 2) → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l774_77449


namespace NUMINAMATH_CALUDE_pet_store_cats_l774_77407

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_left : ℕ) :
  initial_siamese = 12 →
  cats_sold = 20 →
  cats_left = 12 →
  ∃ initial_house : ℕ, initial_house = 20 ∧ initial_siamese + initial_house = cats_sold + cats_left :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l774_77407


namespace NUMINAMATH_CALUDE_usual_time_calculation_l774_77464

/-- Represents the scenario of a person catching a bus -/
structure BusScenario where
  usual_speed : ℝ
  usual_time : ℝ
  faster_speed : ℝ
  missed_time : ℝ

/-- The theorem stating the relationship between usual time and missed time -/
theorem usual_time_calculation (scenario : BusScenario) 
  (h1 : scenario.faster_speed = (5/4) * scenario.usual_speed)
  (h2 : scenario.missed_time = scenario.usual_time + 5)
  (h3 : scenario.usual_speed * scenario.usual_time = scenario.faster_speed * scenario.missed_time) :
  scenario.usual_time = 25 := by
  sorry

#check usual_time_calculation

end NUMINAMATH_CALUDE_usual_time_calculation_l774_77464


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l774_77439

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := 3^n - c

/-- The nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S (n + 1) c - S n c

/-- A sequence is geometric if the ratio between consecutive terms is constant -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  (c = 1 ↔ IsGeometric (a · c)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l774_77439
