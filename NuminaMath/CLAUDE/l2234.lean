import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2234_223472

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≥ 4}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2234_223472


namespace NUMINAMATH_CALUDE_volleyball_club_girls_count_l2234_223453

theorem volleyball_club_girls_count :
  ∀ (total_members : ℕ) (meeting_attendees : ℕ) (girls : ℕ) (boys : ℕ),
    total_members = 32 →
    meeting_attendees = 20 →
    total_members = girls + boys →
    meeting_attendees = boys + girls / 3 →
    girls = 18 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_club_girls_count_l2234_223453


namespace NUMINAMATH_CALUDE_price_crossover_year_l2234_223434

def price_X (year : ℕ) : ℚ :=
  4.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  6.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year : 
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧ 
  price_X 2010 > price_Y 2010 :=
by sorry

end NUMINAMATH_CALUDE_price_crossover_year_l2234_223434


namespace NUMINAMATH_CALUDE_vector_operation_l2234_223433

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 2) → b = (-1, 3) → 2 • a - b = (5, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l2234_223433


namespace NUMINAMATH_CALUDE_square_difference_l2234_223481

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2234_223481


namespace NUMINAMATH_CALUDE_power_sum_and_quotient_l2234_223499

theorem power_sum_and_quotient : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_and_quotient_l2234_223499


namespace NUMINAMATH_CALUDE_parallelogram_division_max_parts_l2234_223497

/-- Given a parallelogram divided into a grid of M by N parts, with one additional line drawn,
    the maximum number of parts the parallelogram can be divided into is MN + M + N - 1. -/
theorem parallelogram_division_max_parts (M N : ℕ) :
  let initial_parts := M * N
  let additional_parts := M + N - 1
  initial_parts + additional_parts = M * N + M + N - 1 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_division_max_parts_l2234_223497


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2234_223409

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (hr : r = 2) 
  (hh : h = 2) : 
  2 * Real.pi * r * h = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2234_223409


namespace NUMINAMATH_CALUDE_worker_b_time_l2234_223456

/-- Given workers a, b, and c, and their work rates, prove that b alone takes 6 hours to complete the work. -/
theorem worker_b_time (a b c : ℝ) : 
  a = 1/3 →                -- a can do the work in 3 hours
  b + c = 1/3 →            -- b and c together can do the work in 3 hours
  a + c = 1/2 →            -- a and c together can do the work in 2 hours
  1/b = 6                  -- b alone takes 6 hours to do the work
:= by sorry

end NUMINAMATH_CALUDE_worker_b_time_l2234_223456


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2234_223450

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  p/(2/p + q*r) + q/(2/q + r*p) + r/(2/r + p*q) = 175/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2234_223450


namespace NUMINAMATH_CALUDE_purple_cell_count_l2234_223436

/-- Represents the state of a cell on the board -/
inductive CellState
| Unpainted
| Blue
| Red
| Purple

/-- Represents a 2x2 square on the board -/
structure Square :=
  (topLeft : Nat × Nat)

/-- Represents the game board -/
def Board := Fin 2022 → Fin 2022 → CellState

/-- Represents a move in the game -/
structure Move :=
  (square : Square)
  (color : CellState)

/-- The game state -/
structure GameState :=
  (board : Board)
  (moves : List Move)

/-- Count the number of purple cells on the board -/
def countPurpleCells (board : Board) : Nat :=
  sorry

/-- Check if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

theorem purple_cell_count (finalState : GameState) 
  (h1 : ∀ move ∈ finalState.moves, isValidMove (applyMove finalState move) move)
  (h2 : ∀ i j, finalState.board i j ≠ CellState.Unpainted) :
  countPurpleCells finalState.board = 2022 * 2020 ∨ 
  countPurpleCells finalState.board = 2020 * 2020 :=
sorry

end NUMINAMATH_CALUDE_purple_cell_count_l2234_223436


namespace NUMINAMATH_CALUDE_vector_operation_l2234_223427

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

theorem vector_operation : 
  (2 : ℝ) • a - b = (-1, 6) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2234_223427


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2234_223443

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 3 adults and 3 children in 6 seats,
    such that no two people of the same type sit together. -/
def seating_arrangements : ℕ :=
  2 * arrangements 3 * arrangements 3

/-- Theorem stating that the number of seating arrangements is 72. -/
theorem seating_arrangements_count :
  seating_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2234_223443


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_l2234_223413

-- Proposition 1
theorem proposition_1 (x y : ℝ) :
  (xy = 0 → x = 0 ∨ y = 0) ↔
  ((x = 0 ∨ y = 0) → xy = 0) ∧
  (xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  ((x ≠ 0 ∧ y ≠ 0) → xy ≠ 0) :=
sorry

-- Proposition 2
theorem proposition_2 (x y : ℝ) :
  ((x > 0 ∧ y > 0) → xy > 0) ↔
  (xy > 0 → x > 0 ∧ y > 0) ∧
  ((x ≤ 0 ∨ y ≤ 0) → xy ≤ 0) ∧
  (xy ≤ 0 → x ≤ 0 ∨ y ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_l2234_223413


namespace NUMINAMATH_CALUDE_min_intersection_cardinality_l2234_223420

-- Define the cardinality of a set
def card (S : Set α) : ℕ := sorry

-- Define the number of subsets of a set
def n (S : Set α) : ℕ := 2^(card S)

-- Define the theorem
theorem min_intersection_cardinality 
  (A B C : Set α) 
  (h1 : n A + n B + n C = n (A ∪ B ∪ C))
  (h2 : card A = 100)
  (h3 : card B = 101)
  (h4 : card (A ∩ B) ≥ 95) :
  96 ≤ card (A ∩ B ∩ C) := by
  sorry

end NUMINAMATH_CALUDE_min_intersection_cardinality_l2234_223420


namespace NUMINAMATH_CALUDE_symmetry_and_line_equation_l2234_223422

/-- The curve on which points P and Q lie -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line of symmetry for points P and Q -/
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- The condition satisfied by the coordinates of P and Q -/
def coordinate_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The theorem stating the value of m and the equation of line PQ -/
theorem symmetry_and_line_equation 
  (x₁ y₁ x₂ y₂ m : ℝ) 
  (h_curve_P : curve x₁ y₁)
  (h_curve_Q : curve x₂ y₂)
  (h_symmetry : symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂)
  (h_condition : coordinate_condition x₁ y₁ x₂ y₂) :
  m = -1 ∧ ∀ (x y : ℝ), y = -x + 1 ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_line_equation_l2234_223422


namespace NUMINAMATH_CALUDE_zero_in_A_l2234_223468

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_zero_in_A_l2234_223468


namespace NUMINAMATH_CALUDE_egg_supply_solution_l2234_223477

/-- Represents the egg supply problem for Mark's farm --/
def egg_supply_problem (daily_supply_store1 : ℕ) (weekly_total : ℕ) : Prop :=
  ∃ (daily_supply_store2 : ℕ),
    daily_supply_store1 = 5 * 12 ∧
    weekly_total = 7 * (daily_supply_store1 + daily_supply_store2) ∧
    daily_supply_store2 = 30

/-- Theorem stating the solution to the egg supply problem --/
theorem egg_supply_solution : 
  egg_supply_problem 60 630 := by
  sorry

end NUMINAMATH_CALUDE_egg_supply_solution_l2234_223477


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2234_223498

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/3 ∧ y = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2234_223498


namespace NUMINAMATH_CALUDE_max_value_P_l2234_223469

open Real

/-- Given positive real numbers a, b, and c satisfying abc + a + c = b,
    the maximum value of P = 2/(a² + 1) - 2/(b² + 1) + 3/(c² + 1) is 1. -/
theorem max_value_P (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a * b * c + a + c = b) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → x * y * z + x + z = y →
    2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_P_l2234_223469


namespace NUMINAMATH_CALUDE_expand_expression_l2234_223447

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2234_223447


namespace NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l2234_223442

theorem quasi_pythagorean_prime_divisor 
  (a b c : ℕ+) 
  (h : c.val ^ 2 = a.val ^ 2 + a.val * b.val + b.val ^ 2) : 
  ∃ (p : ℕ), p > 5 ∧ Nat.Prime p ∧ p ∣ c.val :=
sorry

end NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l2234_223442


namespace NUMINAMATH_CALUDE_compound_ratio_proof_l2234_223475

theorem compound_ratio_proof : 
  let r1 : ℚ := 2/3
  let r2 : ℚ := 6/7
  let r3 : ℚ := 1/3
  let r4 : ℚ := 3/8
  (r1 * r2 * r3 * r4 : ℚ) = 0.07142857142857142 :=
by sorry

end NUMINAMATH_CALUDE_compound_ratio_proof_l2234_223475


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2234_223454

/-- Given an arithmetic sequence with common difference 3 where a₁, a₃, a₄ form a geometric sequence, a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2234_223454


namespace NUMINAMATH_CALUDE_project_completion_time_l2234_223458

/-- The number of days B takes to complete the project -/
def B_days : ℕ := 30

/-- The total number of days when A and B work together with A quitting 5 days before completion -/
def total_days : ℕ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℕ := 5

/-- The number of days A can complete the project alone -/
def A_days : ℕ := 20

theorem project_completion_time :
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l2234_223458


namespace NUMINAMATH_CALUDE_hendricks_guitar_price_l2234_223489

theorem hendricks_guitar_price 
  (gerald_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : gerald_price = 250) 
  (h2 : discount_percentage = 20) :
  gerald_price * (1 - discount_percentage / 100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_hendricks_guitar_price_l2234_223489


namespace NUMINAMATH_CALUDE_discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2234_223461

/-- A quadratic equation ax^2 + bx + c = 0 has exactly one solution if and only if its discriminant is zero -/
theorem discriminant_zero_iff_unique_solution (a b c : ℝ) (ha : a ≠ 0) :
  (b^2 - 4*a*c = 0) ↔ (∃! x, a*x^2 + b*x + c = 0) :=
sorry

/-- The quadratic equation 3x^2 - 6x + m = 0 has exactly one solution if and only if m = 3 -/
theorem unique_solution_iff_m_eq_three :
  (∃! x, 3*x^2 - 6*x + m = 0) ↔ m = 3 :=
sorry

end NUMINAMATH_CALUDE_discriminant_zero_iff_unique_solution_unique_solution_iff_m_eq_three_l2234_223461


namespace NUMINAMATH_CALUDE_martinez_family_height_l2234_223431

def chiquitaHeight : ℝ := 5

def mrMartinezHeight : ℝ := chiquitaHeight + 2

def mrsMartinezHeight : ℝ := chiquitaHeight - 1

def sonHeight : ℝ := chiquitaHeight + 3

def combinedFamilyHeight : ℝ := chiquitaHeight + mrMartinezHeight + mrsMartinezHeight + sonHeight

theorem martinez_family_height : combinedFamilyHeight = 24 := by
  sorry

end NUMINAMATH_CALUDE_martinez_family_height_l2234_223431


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2234_223430

theorem cube_equation_solution : ∃ x : ℝ, (x - 1)^3 = 64 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2234_223430


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2234_223425

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where C = π/6 and 2acosB = c, prove that A = 5π/12. -/
theorem triangle_angle_proof (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  A + B + C = π →  -- Sum of angles in a triangle
  C = π / 6 →  -- Given condition
  2 * a * Real.cos B = c →  -- Given condition
  A = 5 * π / 12 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2234_223425


namespace NUMINAMATH_CALUDE_job_completion_time_l2234_223437

/-- The time taken for two workers to complete a job together -/
def job_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

theorem job_completion_time 
  (rate_A rate_B rate_C : ℚ) 
  (h1 : rate_A + rate_B = 1 / 6)  -- A and B can do the job in 6 days
  (h2 : rate_B + rate_C = 1 / 10) -- B and C can do the job in 10 days
  (h3 : rate_A + rate_B + rate_C = 1 / 5) -- A, B, and C can do the job in 5 days
  : job_time rate_A rate_C = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2234_223437


namespace NUMINAMATH_CALUDE_triangle_intersection_height_l2234_223485

theorem triangle_intersection_height (t : ℝ) : 
  let A : ℝ × ℝ := (0, 8)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (8, 0)
  let T : ℝ × ℝ := ((8 - t) / 4, t)
  let U : ℝ × ℝ := (8 - t, t)
  let area_ATU : ℝ := (1 / 2) * (U.1 - T.1) * (A.2 - T.2)
  (0 ≤ t) ∧ (t ≤ 8) ∧ (area_ATU = 13.5) → t = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_height_l2234_223485


namespace NUMINAMATH_CALUDE_same_color_difference_l2234_223438

/-- The set of colors used for coloring integers. -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A function that colors integers with one of four colors. -/
def ColoringFunction := ℤ → Color

/-- Theorem stating the existence of two integers with the same color and specific difference. -/
theorem same_color_difference (f : ColoringFunction) (x y : ℤ) 
  (h_x_odd : Odd x) (h_y_odd : Odd y) (h_x_y_diff : |x| ≠ |y|) :
  ∃ a b : ℤ, f a = f b ∧ (b - a = x ∨ b - a = y ∨ b - a = x + y ∨ b - a = x - y) := by
  sorry

end NUMINAMATH_CALUDE_same_color_difference_l2234_223438


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2234_223462

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- The point where the function is minimized -/
def min_point : ℝ := 3

theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ f min_point :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2234_223462


namespace NUMINAMATH_CALUDE_age_height_not_function_l2234_223479

-- Define a type for age and height
def Age := ℕ
def Height := ℝ

-- Define a relation between age and height
def AgeHeightRelation := Age → Set Height

-- Define what it means for a relation to be a function
def IsFunction (R : α → Set β) : Prop :=
  ∀ x : α, ∃! y : β, y ∈ R x

-- State the theorem
theorem age_height_not_function :
  ∃ R : AgeHeightRelation, ¬ IsFunction R :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l2234_223479


namespace NUMINAMATH_CALUDE_trains_meet_at_360km_l2234_223424

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : ℕ  -- Departure time in hours after midnight
  speed : ℕ         -- Speed in km/h
  deriving Repr

/-- Calculates the meeting point of three trains -/
def meetingPoint (trainA trainB trainC : Train) : ℕ :=
  let t : ℕ := 18  -- 6 p.m. in 24-hour format
  let distanceA : ℕ := trainA.speed * (t - trainA.departureTime) 
  let distanceB : ℕ := trainB.speed * (t - trainB.departureTime)
  let timeAfterC : ℕ := (distanceB - distanceA) / (trainA.speed - trainB.speed)
  trainC.speed * timeAfterC

theorem trains_meet_at_360km :
  let trainA : Train := { departureTime := 9, speed := 30 }
  let trainB : Train := { departureTime := 15, speed := 40 }
  let trainC : Train := { departureTime := 18, speed := 60 }
  meetingPoint trainA trainB trainC = 360 := by
  sorry

#eval meetingPoint { departureTime := 9, speed := 30 } { departureTime := 15, speed := 40 } { departureTime := 18, speed := 60 }

end NUMINAMATH_CALUDE_trains_meet_at_360km_l2234_223424


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2234_223432

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_value_reciprocal_sum_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ (1 / m₀ + 1 / n₀ = 3 + 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2234_223432


namespace NUMINAMATH_CALUDE_divisibility_condition_l2234_223478

theorem divisibility_condition (p : Nat) (α : Nat) (x : Int) :
  Prime p → p > 2 → α > 0 →
  (∃ k : Int, x^2 - 1 = k * p^α) ↔
  (∃ t : Int, x = t * p^α + 1 ∨ x = t * p^α - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2234_223478


namespace NUMINAMATH_CALUDE_range_of_m_l2234_223457

theorem range_of_m (x y m : ℝ) (h1 : x^2 + 4*y^2*(m^2 + 3*m)*x*y = 0) (h2 : x*y ≠ 0) : -4 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2234_223457


namespace NUMINAMATH_CALUDE_complex_fraction_sum_complex_equation_solution_l2234_223467

-- Define the complex number i
def i : ℂ := Complex.I

-- Problem 1
theorem complex_fraction_sum : 
  (1 + i)^2 / (1 + 2*i) + (1 - i)^2 / (2 - i) = 6/5 - 2/5 * i := by sorry

-- Problem 2
theorem complex_equation_solution (x y : ℝ) :
  x / (1 + i) + y / (1 + 2*i) = 10 / (1 + 3*i) → x = -2 ∧ y = 10 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_complex_equation_solution_l2234_223467


namespace NUMINAMATH_CALUDE_game_outcome_theorem_l2234_223493

/-- Represents the outcome of the game for a player -/
inductive Outcome
  | Points (n : ℕ)
  | NoPoints

/-- Represents a player's choice in the game -/
structure PlayerChoice where
  value : ℕ
  is_valid : 0 ≤ value ∧ value ≤ 10

/-- Determines the outcome for a player based on their choice and whether it's unique -/
def gameOutcome (choice : PlayerChoice) (is_unique : Bool) : Outcome :=
  if is_unique then Outcome.Points choice.value else Outcome.NoPoints

/-- Theorem stating that the outcome is either the chosen points or zero -/
theorem game_outcome_theorem (choice : PlayerChoice) (is_unique : Bool) :
  (gameOutcome choice is_unique = Outcome.Points choice.value) ∨
  (gameOutcome choice is_unique = Outcome.NoPoints) := by
  sorry

#check game_outcome_theorem

end NUMINAMATH_CALUDE_game_outcome_theorem_l2234_223493


namespace NUMINAMATH_CALUDE_estimate_student_population_l2234_223495

theorem estimate_student_population (first_survey : ℕ) (second_survey : ℕ) (overlap : ℕ) 
  (h1 : first_survey = 80)
  (h2 : second_survey = 100)
  (h3 : overlap = 20) :
  (first_survey * second_survey) / overlap = 400 := by
  sorry

end NUMINAMATH_CALUDE_estimate_student_population_l2234_223495


namespace NUMINAMATH_CALUDE_sum_110_is_neg_110_l2234_223404

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 10 terms -/
  sum_10 : ℤ
  /-- Sum of the first 100 terms -/
  sum_100 : ℤ
  /-- Property: sum of first 10 terms is 100 -/
  prop_10 : sum_10 = 100
  /-- Property: sum of first 100 terms is 10 -/
  prop_100 : sum_100 = 10

/-- Theorem: For the given arithmetic sequence, the sum of the first 110 terms is -110 -/
theorem sum_110_is_neg_110 (seq : ArithmeticSequence) : ℤ :=
  -110

#check sum_110_is_neg_110

end NUMINAMATH_CALUDE_sum_110_is_neg_110_l2234_223404


namespace NUMINAMATH_CALUDE_quadratic_root_conjugate_l2234_223440

theorem quadratic_root_conjugate (a b c : ℚ) :
  (a ≠ 0) →
  (a * (3 + Real.sqrt 2)^2 + b * (3 + Real.sqrt 2) + c = 0) →
  (a * (3 - Real.sqrt 2)^2 + b * (3 - Real.sqrt 2) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_conjugate_l2234_223440


namespace NUMINAMATH_CALUDE_rain_probability_implies_very_likely_l2234_223401

-- Define what "very likely" means in terms of probability
def very_likely (p : ℝ) : Prop := p ≥ 0.7

-- Theorem statement
theorem rain_probability_implies_very_likely (p : ℝ) (h : p = 0.8) : very_likely p := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_implies_very_likely_l2234_223401


namespace NUMINAMATH_CALUDE_range_of_a_l2234_223460

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the range of a
def range_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

-- State the theorem
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2234_223460


namespace NUMINAMATH_CALUDE_min_remainders_consecutive_numbers_l2234_223452

theorem min_remainders_consecutive_numbers : ∃ (x a r : ℕ), 
  (100 ≤ x) ∧ (x < 1000) ∧
  (10 ≤ a) ∧ (a < 100) ∧
  (r < a) ∧ (a + r ≥ 100) ∧
  (∀ i : Fin 4, (x + i) % (a + i) = r) :=
by sorry

end NUMINAMATH_CALUDE_min_remainders_consecutive_numbers_l2234_223452


namespace NUMINAMATH_CALUDE_minimum_bundle_price_l2234_223492

/- Define the costs of items -/
def water_cost : ℚ := 0.50
def fruit_cost : ℚ := 0.25
def snack_cost : ℚ := 1.00

/- Define the bundle composition -/
def water_per_bundle : ℕ := 1
def snacks_per_bundle : ℕ := 3
def fruits_per_bundle : ℕ := 2

/- Define the special offer -/
def special_bundle_interval : ℕ := 5
def special_bundle_price : ℚ := 2.00
def complimentary_snacks : ℕ := 1

/- Theorem statement -/
theorem minimum_bundle_price (P : ℚ) : 
  (P ≥ 4.75) ↔ 
  (4 * P + special_bundle_price ≥ 
    5 * (water_cost * water_per_bundle + 
         snack_cost * snacks_per_bundle + 
         fruit_cost * fruits_per_bundle) + 
    snack_cost * complimentary_snacks) := by
  sorry

end NUMINAMATH_CALUDE_minimum_bundle_price_l2234_223492


namespace NUMINAMATH_CALUDE_happy_boys_count_l2234_223463

theorem happy_boys_count (total_children happy_children sad_children neutral_children
                          total_boys total_girls sad_girls neutral_boys : ℕ)
                         (h1 : total_children = 60)
                         (h2 : happy_children = 30)
                         (h3 : sad_children = 10)
                         (h4 : neutral_children = 20)
                         (h5 : total_boys = 17)
                         (h6 : total_girls = 43)
                         (h7 : sad_girls = 4)
                         (h8 : neutral_boys = 5)
                         (h9 : total_children = happy_children + sad_children + neutral_children)
                         (h10 : total_children = total_boys + total_girls) :
  total_boys - (sad_children - sad_girls) - neutral_boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_happy_boys_count_l2234_223463


namespace NUMINAMATH_CALUDE_inequality_proof_l2234_223414

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 1)) + (1 / (5 * b^2 - 4 * b + 1)) + (1 / (5 * c^2 - 4 * c + 1)) ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2234_223414


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2234_223435

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x + 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2234_223435


namespace NUMINAMATH_CALUDE_phone_price_reduction_l2234_223411

theorem phone_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1280)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) :
  x = 0.18 := by sorry

end NUMINAMATH_CALUDE_phone_price_reduction_l2234_223411


namespace NUMINAMATH_CALUDE_calculator_game_result_l2234_223441

def calculator_game (n : Nat) (a b c : Int) : Int :=
  let f1 := fun x => x^3
  let f2 := fun x => x^2
  let f3 := fun x => -x
  (f1^[n] a) + (f2^[n] b) + (f3^[n] c)

theorem calculator_game_result :
  calculator_game 45 1 0 (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_result_l2234_223441


namespace NUMINAMATH_CALUDE_lees_initial_money_l2234_223465

def friends_money : ℕ := 8
def meal_cost : ℕ := 15
def total_paid : ℕ := 18

theorem lees_initial_money :
  ∃ (lees_money : ℕ), lees_money + friends_money = total_paid ∧ lees_money = 10 := by
sorry

end NUMINAMATH_CALUDE_lees_initial_money_l2234_223465


namespace NUMINAMATH_CALUDE_proper_subsets_count_l2234_223473

def U : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {1,2}
def B : Finset Nat := {3,4}

theorem proper_subsets_count :
  (Finset.powerset (A ∩ (U \ B))).card - 1 = 3 := by sorry

end NUMINAMATH_CALUDE_proper_subsets_count_l2234_223473


namespace NUMINAMATH_CALUDE_theta_range_theorem_l2234_223408

-- Define the set of valid θ values
def ValidTheta : Set ℝ := { θ | -Real.pi ≤ θ ∧ θ ≤ Real.pi }

-- Define the inequality condition
def InequalityCondition (θ : ℝ) : Prop :=
  Real.cos (θ + Real.pi / 4) < 3 * (Real.sin θ ^ 5 - Real.cos θ ^ 5)

-- Define the solution set
def SolutionSet : Set ℝ := 
  { θ | (-Real.pi ≤ θ ∧ θ < -3 * Real.pi / 4) ∨ (Real.pi / 4 < θ ∧ θ ≤ Real.pi) }

-- Theorem statement
theorem theta_range_theorem :
  ∀ θ ∈ ValidTheta, InequalityCondition θ ↔ θ ∈ SolutionSet :=
sorry

end NUMINAMATH_CALUDE_theta_range_theorem_l2234_223408


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2234_223491

/-- A geometric sequence with first term 1 and common ratio q ≠ -1 -/
def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  q^(n-1)

theorem geometric_sequence_fifth_term
  (q : ℝ)
  (h1 : q ≠ -1)
  (h2 : geometric_sequence q 5 + geometric_sequence q 4 = 3 * (geometric_sequence q 3 + geometric_sequence q 2)) :
  geometric_sequence q 5 = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2234_223491


namespace NUMINAMATH_CALUDE_polygon_area_is_1800_l2234_223402

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the polygon -/
def vertices : List Point := [
  ⟨0, 0⟩, ⟨15, 0⟩, ⟨45, 30⟩, ⟨45, 45⟩, ⟨30, 45⟩, ⟨0, 15⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vs : List Point) : ℝ :=
  sorry

/-- The theorem stating that the area of the given polygon is 1800 square units -/
theorem polygon_area_is_1800 : polygonArea vertices = 1800 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_1800_l2234_223402


namespace NUMINAMATH_CALUDE_select_three_from_eight_l2234_223486

theorem select_three_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_three_from_eight_l2234_223486


namespace NUMINAMATH_CALUDE_min_groups_for_children_l2234_223400

/-- Given a total of 30 children and a maximum of 7 children per group,
    prove that the minimum number of equal-sized groups needed is 5. -/
theorem min_groups_for_children (total_children : Nat) (max_per_group : Nat) 
    (h1 : total_children = 30) (h2 : max_per_group = 7) : 
    (∃ (group_size : Nat), group_size ≤ max_per_group ∧ 
    total_children % group_size = 0 ∧ 
    total_children / group_size = 5 ∧
    ∀ (other_size : Nat), other_size ≤ max_per_group ∧ 
    total_children % other_size = 0 → 
    total_children / other_size ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_children_l2234_223400


namespace NUMINAMATH_CALUDE_pencil_price_l2234_223419

theorem pencil_price (joy_pencils colleen_pencils : ℕ) (price_difference : ℚ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_difference = 80 →
  ∃ (price : ℚ), 
    colleen_pencils * price = joy_pencils * price + price_difference ∧
    price = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l2234_223419


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2234_223471

theorem cubic_expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2234_223471


namespace NUMINAMATH_CALUDE_lucca_basketball_percentage_proof_l2234_223466

/-- The percentage of Lucca's balls that are basketballs -/
def lucca_basketball_percentage : ℝ := 10

theorem lucca_basketball_percentage_proof :
  let lucca_total_balls : ℕ := 100
  let lucien_total_balls : ℕ := 200
  let lucien_basketball_percentage : ℝ := 20
  let total_basketballs : ℕ := 50
  lucca_basketball_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_lucca_basketball_percentage_proof_l2234_223466


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_probability_l2234_223494

/-- Given a sequence of positive integers, the probability that the sum of three
    independently and randomly selected elements is divisible by 3 is at least 1/4. -/
theorem sum_divisible_by_three_probability (n : ℕ) (seq : Fin n → ℕ+) :
  ∃ (p q r : ℝ), p + q + r = 1 ∧ p ≥ 0 ∧ q ≥ 0 ∧ r ≥ 0 ∧
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_probability_l2234_223494


namespace NUMINAMATH_CALUDE_inequality_proof_l2234_223410

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ 0) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2234_223410


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2234_223418

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -4) :
  (1 - (x + 1) / (x^2 - 2*x + 1)) / ((x - 3) / (x - 1)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2234_223418


namespace NUMINAMATH_CALUDE_vanessa_large_orders_l2234_223412

/-- The number of grams of packing peanuts needed for a large order -/
def large_order_peanuts : ℕ := 200

/-- The number of grams of packing peanuts needed for a small order -/
def small_order_peanuts : ℕ := 50

/-- The total number of grams of packing peanuts used -/
def total_peanuts_used : ℕ := 800

/-- The number of small orders sent -/
def num_small_orders : ℕ := 4

/-- The number of large orders sent -/
def num_large_orders : ℕ := 3

theorem vanessa_large_orders :
  num_large_orders * large_order_peanuts + num_small_orders * small_order_peanuts = total_peanuts_used :=
by sorry

end NUMINAMATH_CALUDE_vanessa_large_orders_l2234_223412


namespace NUMINAMATH_CALUDE_income_remainder_relation_l2234_223487

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children : ℝ
  wife : ℝ
  bills : ℝ
  savings : ℝ
  remainder : ℝ

/-- Theorem stating the relationship between income and remainder --/
theorem income_remainder_relation (d : IncomeDistribution) :
  d.children = 0.18 * d.total ∧
  d.wife = 0.28 * d.total ∧
  d.bills = 0.12 * d.total ∧
  d.savings = 0.15 * d.total ∧
  d.remainder = 35000 →
  0.27 * d.total = 35000 := by
  sorry

end NUMINAMATH_CALUDE_income_remainder_relation_l2234_223487


namespace NUMINAMATH_CALUDE_unique_function_theorem_l2234_223417

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- Condition: f(n) is a perfect square for all n -/
def IsPerfectSquare (f : PositiveIntFunction) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, f n = k * k

/-- Condition: f(m+n) = f(m) + f(n) + 2mn for all m, n -/
def SatisfiesFunctionalEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (m + n) = f m + f n + 2 * m * n

/-- Theorem: The only function satisfying both conditions is f(n) = n² -/
theorem unique_function_theorem (f : PositiveIntFunction) 
  (h1 : IsPerfectSquare f) (h2 : SatisfiesFunctionalEquation f) :
  ∀ n : ℕ+, f n = n * n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l2234_223417


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_221_l2234_223476

theorem modular_inverse_of_5_mod_221 : ∃ x : ℕ, x < 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_221_l2234_223476


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2234_223483

/-- Represents a country --/
inductive Country
| Italy
| Germany

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties

/-- The price of a stamp in cents --/
def stampPrice (c : Country) : ℕ :=
  match c with
  | Country.Italy => 7
  | Country.Germany => 5

/-- The number of stamps Juan has from a given country and decade --/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 8
  | Country.Germany, Decade.Fifties => 7
  | Country.Germany, Decade.Sixties => 6

/-- The total cost of Juan's European stamps from Italy and Germany issued before the 70's --/
def totalCost : ℚ :=
  let italyTotal := (stampCount Country.Italy Decade.Fifties + stampCount Country.Italy Decade.Sixties) * stampPrice Country.Italy
  let germanyTotal := (stampCount Country.Germany Decade.Fifties + stampCount Country.Germany Decade.Sixties) * stampPrice Country.Germany
  (italyTotal + germanyTotal : ℚ) / 100

theorem total_cost_is_correct : totalCost = 156 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2234_223483


namespace NUMINAMATH_CALUDE_evaluate_expression_l2234_223455

theorem evaluate_expression : 
  3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^5 + 3^7 = 6^5 + 3^7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2234_223455


namespace NUMINAMATH_CALUDE_barrel_contents_l2234_223480

theorem barrel_contents :
  ∀ (x : ℝ),
  (x > 0) →
  (x / 6 = x - 5 * x / 6) →
  (5 * x / 30 = 5 * x / 6 - 2 * x / 3) →
  (x / 6 = 2 * x / 3 - x / 2) →
  ((x + 120) + (5 * x / 6 + 120) = 4 * (x / 2)) →
  (x = 1440 ∧ 
   5 * x / 6 = 1200 ∧ 
   2 * x / 3 = 960 ∧ 
   x / 2 = 720) :=
by sorry

end NUMINAMATH_CALUDE_barrel_contents_l2234_223480


namespace NUMINAMATH_CALUDE_room_length_calculation_l2234_223451

theorem room_length_calculation (area : Real) (width : Real) (length : Real) :
  area = 10 ∧ width = 2 ∧ area = length * width → length = 5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l2234_223451


namespace NUMINAMATH_CALUDE_wilsons_theorem_l2234_223439

theorem wilsons_theorem (p : ℕ) (hp : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l2234_223439


namespace NUMINAMATH_CALUDE_carries_hourly_rate_l2234_223429

/-- Represents Carrie's cake-making scenario -/
structure CakeScenario where
  hoursPerDay : ℕ
  daysWorked : ℕ
  suppliesCost : ℕ
  profit : ℕ

/-- Calculates Carrie's hourly rate given the scenario -/
def hourlyRate (scenario : CakeScenario) : ℚ :=
  (scenario.profit + scenario.suppliesCost) / (scenario.hoursPerDay * scenario.daysWorked)

/-- Theorem stating that Carrie's hourly rate was $22 -/
theorem carries_hourly_rate :
  let scenario : CakeScenario := {
    hoursPerDay := 2,
    daysWorked := 4,
    suppliesCost := 54,
    profit := 122
  }
  hourlyRate scenario = 22 := by sorry

end NUMINAMATH_CALUDE_carries_hourly_rate_l2234_223429


namespace NUMINAMATH_CALUDE_fraction_simplification_l2234_223444

theorem fraction_simplification (b y : ℝ) (h : b^2 ≠ y^2) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 - y^2) = (b^2 + y^2) / (b^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2234_223444


namespace NUMINAMATH_CALUDE_parentheses_multiplication_l2234_223482

theorem parentheses_multiplication : (4 - 3) * 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_multiplication_l2234_223482


namespace NUMINAMATH_CALUDE_det_A_plus_three_eq_two_l2234_223415

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 3, 4]

theorem det_A_plus_three_eq_two :
  Matrix.det A + 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_A_plus_three_eq_two_l2234_223415


namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l2234_223484

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : Prop :=
  x^2 + p * x + q = 0

-- State the theorem
theorem root_sum_of_coefficients (p q : ℝ) :
  quadratic p q (1 + i) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l2234_223484


namespace NUMINAMATH_CALUDE_sector_area_l2234_223496

/-- The area of a circular sector with radius 6 cm and central angle 120° is 12π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let θ : ℝ := 120
  let π : ℝ := Real.pi
  (θ / 360) * π * r^2 = 12 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l2234_223496


namespace NUMINAMATH_CALUDE_dividend_calculation_l2234_223416

theorem dividend_calculation (dividend quotient remainder : ℕ) : 
  dividend / 3 = quotient ∧ 
  dividend % 3 = remainder ∧ 
  quotient = 16 ∧ 
  remainder = 4 → 
  dividend = 52 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2234_223416


namespace NUMINAMATH_CALUDE_tree_planting_problem_l2234_223403

-- Define the types for our numbers
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

-- Function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

-- Define our theorem
theorem tree_planting_problem 
  (poplars : ThreeDigitNumber) 
  (lindens : TwoDigitNumber) 
  (h1 : poplars.val + lindens.val = 144)
  (h2 : reverseDigits poplars.val + reverseDigits lindens.val = 603) :
  poplars.val = 105 ∧ lindens.val = 39 := by
  sorry


end NUMINAMATH_CALUDE_tree_planting_problem_l2234_223403


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2234_223464

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (x₀ + 1) ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2234_223464


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2234_223470

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (43 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (43 * π / 180) * Real.sin (17 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_over_2_l2234_223470


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2234_223423

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3) : 
  (a * b) / (a + b + 1) + (a * c) / (a + c + 1) + (b * c) / (b + c + 1) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l2234_223423


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l2234_223488

theorem smallest_number_with_remainder (n : ℕ) : 
  n = 1996 ↔ 
  (n > 1992 ∧ 
   n % 9 = 7 ∧ 
   ∀ m, m > 1992 ∧ m % 9 = 7 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l2234_223488


namespace NUMINAMATH_CALUDE_library_shelves_problem_l2234_223407

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (large_books small_books shelf_capacity : ℕ) : ℕ :=
  let total_units := 2 * large_books + small_books
  (total_units + shelf_capacity - 1) / shelf_capacity

theorem library_shelves_problem :
  let initial_large_books := 18
  let initial_small_books := 18
  let removed_large_books := 4
  let removed_small_books := 2
  let shelf_capacity := 6
  let remaining_large_books := initial_large_books - removed_large_books
  let remaining_small_books := initial_small_books - removed_small_books
  shelves_needed remaining_large_books remaining_small_books shelf_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_problem_l2234_223407


namespace NUMINAMATH_CALUDE_three_number_average_l2234_223459

theorem three_number_average : 
  ∀ (x y z : ℝ),
  y = 2 * x →
  z = 4 * y →
  x = 45 →
  (x + y + z) / 3 = 165 := by
sorry

end NUMINAMATH_CALUDE_three_number_average_l2234_223459


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2234_223448

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (hn : n = 5264) (hd : d = 17) :
  ∃ (k : ℕ), k ≤ d - 1 ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2234_223448


namespace NUMINAMATH_CALUDE_distribution_ways_for_problem_l2234_223405

/-- Represents a hotel with a fixed number of rooms -/
structure Hotel where
  numRooms : Nat
  maxPerRoom : Nat

/-- Represents a group of friends -/
structure FriendGroup where
  numFriends : Nat

/-- Calculates the number of ways to distribute friends in rooms -/
def distributionWays (h : Hotel) (f : FriendGroup) : Nat :=
  sorry

/-- The specific hotel in the problem -/
def problemHotel : Hotel :=
  { numRooms := 5, maxPerRoom := 2 }

/-- The specific friend group in the problem -/
def problemFriendGroup : FriendGroup :=
  { numFriends := 5 }

theorem distribution_ways_for_problem :
  distributionWays problemHotel problemFriendGroup = 2220 :=
sorry

end NUMINAMATH_CALUDE_distribution_ways_for_problem_l2234_223405


namespace NUMINAMATH_CALUDE_hostel_mess_expenditure_l2234_223490

/-- The original expenditure of a hostel mess given specific conditions -/
theorem hostel_mess_expenditure :
  ∀ (initial_students : ℕ) 
    (student_increase : ℕ) 
    (expense_increase : ℕ) 
    (avg_expenditure_decrease : ℕ),
  initial_students = 35 →
  student_increase = 7 →
  expense_increase = 42 →
  avg_expenditure_decrease = 1 →
  ∃ (original_expenditure : ℕ),
    original_expenditure = 420 ∧
    (initial_students + student_increase) * 
      ((original_expenditure / initial_students) - avg_expenditure_decrease) =
    original_expenditure + expense_increase :=
by sorry

end NUMINAMATH_CALUDE_hostel_mess_expenditure_l2234_223490


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2234_223445

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  z₁ / z₂ - z₂ / z₁ = 24 * I / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2234_223445


namespace NUMINAMATH_CALUDE_bart_firewood_calculation_l2234_223449

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of days Bart burns logs (Nov 1 through Feb 28) -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_cut : ℕ := 8

/-- The number of pieces of firewood Bart gets from one tree -/
def firewood_per_tree : ℕ := total_days * logs_per_day / trees_cut

theorem bart_firewood_calculation :
  firewood_per_tree = 75 :=
sorry

end NUMINAMATH_CALUDE_bart_firewood_calculation_l2234_223449


namespace NUMINAMATH_CALUDE_power_calculation_l2234_223406

theorem power_calculation : 3^18 / 27^3 * 9 = 177147 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2234_223406


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2234_223474

-- Define the parallelogram properties
def parallelogram_area : ℝ := 360
def parallelogram_height : ℝ := 12

-- Theorem statement
theorem parallelogram_base_length :
  parallelogram_area / parallelogram_height = 30 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2234_223474


namespace NUMINAMATH_CALUDE_special_sequence_1000th_term_l2234_223428

def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ 
  a 2 = 1015 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem special_sequence_1000th_term (a : ℕ → ℕ) 
  (h : special_sequence a) : a 1000 = 1676 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_1000th_term_l2234_223428


namespace NUMINAMATH_CALUDE_arithmetic_equation_proof_l2234_223421

theorem arithmetic_equation_proof : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 * 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_proof_l2234_223421


namespace NUMINAMATH_CALUDE_marble_jar_problem_l2234_223426

theorem marble_jar_problem (M : ℕ) : 
  (∀ (x : ℕ), x = M / 16 → x - 1 = M / 18) → M = 144 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l2234_223426


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2234_223446

/-- Given an initial angle of 50 degrees that is rotated 540 degrees clockwise,
    the resulting new acute angle is also 50 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) (h1 : initial_angle = 50)
    (h2 : rotation = 540) : 
    (initial_angle + rotation) % 360 = 50 ∨ 
    (360 - (initial_angle + rotation) % 360) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2234_223446
