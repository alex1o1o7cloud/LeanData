import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_integers_l1043_104305

/-- A table of integers -/
def Table := Matrix (Fin 16) (Fin 16) Int

/-- The property that each row of a table contains at most 4 distinct integers -/
def row_constraint (t : Table) : Prop :=
  ∀ i, Finset.card (Finset.image (λ j ↦ t i j) Finset.univ) ≤ 4

/-- The property that each column of a table contains at most 4 distinct integers -/
def col_constraint (t : Table) : Prop :=
  ∀ j, Finset.card (Finset.image (λ i ↦ t i j) Finset.univ) ≤ 4

/-- The number of distinct integers in a table -/
def distinct_count (t : Table) : Nat :=
  Finset.card (Finset.image (λ (p : Fin 16 × Fin 16) ↦ t p.1 p.2) (Finset.univ.product Finset.univ))

/-- The theorem stating that the maximum number of distinct integers in a 16x16 table
    with the given constraints is 49 -/
theorem max_distinct_integers (t : Table) 
    (h_row : row_constraint t) (h_col : col_constraint t) : 
  distinct_count t ≤ 49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_integers_l1043_104305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l1043_104382

/-- Conversion factor from m/s to km/h -/
noncomputable def conversion_factor : ℝ := 3.6

/-- Speed in m/s -/
noncomputable def speed_ms : ℝ := 15 / 36

/-- Speed in km/h -/
noncomputable def speed_kmh : ℝ := 1.5

/-- Theorem: Converting 15/36 m/s to km/h results in 1.5 km/h -/
theorem speed_conversion :
  speed_ms * conversion_factor = speed_kmh := by
  -- Unfold the definitions
  unfold speed_ms conversion_factor speed_kmh
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_l1043_104382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1043_104378

-- Define the cube
def cube_edge_length : ℝ := 10

-- Define the reflection point coordinates
def reflection_point_x : ℝ := 3
def reflection_point_y : ℝ := 6

-- Define the number of reflections needed to reach a vertex
def num_reflections : ℕ := 10

-- Define the path length
noncomputable def path_length : ℝ := num_reflections * Real.sqrt (cube_edge_length^2 + reflection_point_x^2 + reflection_point_y^2)

-- Theorem statement
theorem light_path_length :
  path_length = 10 * Real.sqrt 145 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1043_104378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1043_104366

/-- Given vectors a and b in ℝ², and c = a + t*b, prove that if <a, c> = <b, c>, then t = 5 -/
theorem vector_equality (a b : ℝ × ℝ) (t : ℝ) (h1 : a = (3, 4)) (h2 : b = (1, 0)) :
  let c := (a.1 + t * b.1, a.2 + t * b.2)
  (a.1 * c.1 + a.2 * c.2 = b.1 * c.1 + b.2 * c.2) → t = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1043_104366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABCD_l1043_104323

-- Define the quadrilateral ABCD as a set of four points in ℝ²
def QuadrilateralABCD : Set (ℝ × ℝ) := sorry

-- Define the vectors AC and BD
def AC : ℝ × ℝ := (-2, 1)
def BD : ℝ × ℝ := (2, 4)

-- Function to calculate the area of a quadrilateral given its four vertices
noncomputable def areaQuadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral_ABCD : 
  ∃ (A B C D : ℝ × ℝ), 
    A ∈ QuadrilateralABCD ∧ B ∈ QuadrilateralABCD ∧ 
    C ∈ QuadrilateralABCD ∧ D ∈ QuadrilateralABCD ∧
    C - A = AC ∧ D - B = BD ∧
    areaQuadrilateral A B C D = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABCD_l1043_104323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_l1043_104360

noncomputable section

open Set

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def diameter (C : Set (ℝ × ℝ)) : ℝ := sorry

def area (C : Set (ℝ × ℝ)) : ℝ := sorry

theorem inscribed_circle_diameter 
  (D C : Set (ℝ × ℝ)) 
  (center : ℝ × ℝ)
  (R r : ℝ)
  (h_D : D = Circle center R)
  (h_C : C = Circle center r)
  (h_inscribed : C ⊆ D) 
  (h_D_diameter : diameter D = 24) 
  (h_area_ratio : (area D - area C) / area C = 4) : 
  diameter C = 24 * Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_diameter_l1043_104360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_min_tangent_length_extremum_sum_distances_l1043_104320

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Statement for part (I)
theorem min_c_value : 
  (∀ x y : ℝ, C x y → x - y + (2 * Real.sqrt 2 + 1) ≥ 0) ∧
  (∀ ε > 0, ∃ x y : ℝ, C x y ∧ x - y + (2 * Real.sqrt 2 + 1 - ε) < 0) := 
by sorry

-- Statement for part (II)
theorem min_tangent_length : 
  ∃ x₀ y₀ : ℝ, tangent_line x₀ y₀ ∧
  (∀ x y : ℝ, tangent_line x y → 
    Real.sqrt ((x - 3)^2 + (y - 4)^2) - 2 ≥ Real.sqrt ((x₀ - 3)^2 + (y₀ - 4)^2) - 2) ∧
  Real.sqrt ((x₀ - 3)^2 + (y₀ - 4)^2) - 2 = 2 * Real.sqrt 7 :=
by sorry

-- Helper function for distance calculation
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Statement for part (III)
theorem extremum_sum_distances :
  (∃ x y : ℝ, C x y ∧ 
    dist_squared (x, y) A + dist_squared (x, y) B = 42 ∧
    (∀ x' y' : ℝ, C x' y' → 
      dist_squared (x', y') A + dist_squared (x', y') B ≥ 42)) ∧
  (∃ x y : ℝ, C x y ∧ 
    dist_squared (x, y) A + dist_squared (x, y) B = 202 ∧
    (∀ x' y' : ℝ, C x' y' → 
      dist_squared (x', y') A + dist_squared (x', y') B ≤ 202)) ∧
  (∃ x y : ℝ, C x y ∧ 
    dist_squared (x, y) A + dist_squared (x, y) B = 42 ∧
    x = 9/5 ∧ y = 12/5) ∧
  (∃ x y : ℝ, C x y ∧ 
    dist_squared (x, y) A + dist_squared (x, y) B = 202 ∧
    x = 21/5 ∧ y = 28/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_min_tangent_length_extremum_sum_distances_l1043_104320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_problem_l1043_104356

/-- Converts grams to kilograms -/
noncomputable def gramsToKg (g : ℝ) : ℝ := g / 1000

/-- Calculates the weight of each pile of jelly beans -/
noncomputable def jellyBeanPileWeight (initialWeight : ℝ) (eatenWeight : ℝ) (numPiles : ℕ) : ℝ :=
  (initialWeight - gramsToKg eatenWeight) / (numPiles : ℝ)

theorem jelly_bean_problem :
  let initialWeight := (4.5 : ℝ)
  let eatenWeight := (850 : ℝ)
  let numPiles := 7
  abs (jellyBeanPileWeight initialWeight eatenWeight numPiles - 0.52) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_problem_l1043_104356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_equals_seven_l1043_104397

-- Define the function f as noncomputable
noncomputable def f : ℝ → ℝ := λ x => (x^2 + 5*x + 13) / 9

-- State the theorem
theorem f_five_equals_seven :
  (∀ x : ℝ, f (3*x - 1) = x^2 + x + 1) → f 5 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_five_equals_seven_l1043_104397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_l1043_104304

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (x - 5/2) - Real.log x + 1/x

-- State the theorem
theorem f_positive (x : ℝ) (hx : x > 0) : f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_l1043_104304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_5pi_over_6_l1043_104341

theorem sin_alpha_minus_5pi_over_6 (α : ℝ) 
  (h : Real.cos (α - π/6) + Real.sin α = 4*Real.sqrt 3/5) : 
  Real.sin (α - 5*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_5pi_over_6_l1043_104341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_solution_replacement_l1043_104367

theorem chemical_solution_replacement 
  (original_concentration : ℝ) 
  (replaced_portion : ℝ) 
  (final_concentration : ℝ) 
  (replacement_concentration : ℝ) 
  (h1 : original_concentration = 0.5)
  (h2 : replaced_portion = 0.5)
  (h3 : final_concentration = 0.55)
  (h4 : (1 - replaced_portion) * original_concentration + 
        replaced_portion * replacement_concentration = final_concentration)
  : replacement_concentration = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemical_solution_replacement_l1043_104367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_tangent_circle_radius_theorem_sum_abcd_theorem_l1043_104335

/-- A rhombus with side length 7 and angles at A and C equal to 60° -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_A : ℝ)
  (angle_C : ℝ)

/-- Circles centered at vertices of the rhombus -/
structure RhombusCircles :=
  (radius_A : ℝ)
  (radius_B : ℝ)
  (radius_C : ℝ)
  (radius_D : ℝ)

/-- The radius of the inner tangent circle -/
noncomputable def inner_tangent_circle_radius (r : Rhombus) (c : RhombusCircles) : ℝ := 
  (-15 + 8 * Real.sqrt 7) / 4

/-- Theorem stating the radius of the inner tangent circle -/
theorem inner_tangent_circle_radius_theorem (r : Rhombus) (c : RhombusCircles) :
  r.side_length = 7 ∧ 
  r.angle_A = 60 ∧ 
  r.angle_C = 60 ∧
  c.radius_A = 4 ∧
  c.radius_B = 3 ∧
  c.radius_C = 3 ∧
  c.radius_D = 4 →
  inner_tangent_circle_radius r c = (-15 + 8 * Real.sqrt 7) / 4 :=
by
  sorry

/-- The sum of a, b, c, and d in the expression (a + b√c)/d -/
def sum_abcd : ℕ := 4

/-- Theorem stating that the sum of a, b, c, and d is correct -/
theorem sum_abcd_theorem :
  sum_abcd = 4 :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_tangent_circle_radius_theorem_sum_abcd_theorem_l1043_104335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unique_solution_iff_d_eq_36_l1043_104380

/-- Represents a system of two linear equations in two variables -/
structure LinearSystem where
  a₁ : ℝ
  b₁ : ℝ
  c₁ : ℝ
  a₂ : ℝ
  b₂ : ℝ
  c₂ : ℝ

/-- Checks if a LinearSystem has a unique solution -/
def hasUniqueSolution (sys : LinearSystem) : Prop :=
  sys.a₁ * sys.b₂ ≠ sys.a₂ * sys.b₁

/-- The specific system of equations from the problem -/
def problemSystem (d : ℝ) : LinearSystem :=
  { a₁ := 9, b₁ := 12, c₁ := 36,
    a₂ := 9, b₂ := 12, c₂ := d }

theorem no_unique_solution_iff_d_eq_36 :
  ∀ d : ℝ, ¬(hasUniqueSolution (problemSystem d)) ↔ d = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_unique_solution_iff_d_eq_36_l1043_104380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_roll_area_theorem_l1043_104350

/-- Regular hexagon with side length s -/
structure RegularHexagon where
  s : ℝ
  s_pos : s > 0

/-- The area of a regular hexagon -/
noncomputable def area_hexagon (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.s^2

/-- The circumradius of a regular hexagon -/
noncomputable def circumradius (h : RegularHexagon) : ℝ :=
  h.s / Real.sqrt 3

/-- The area enclosed by the path of a vertex when the hexagon rolls along a straight line -/
noncomputable def area_enclosed_by_roll (h : RegularHexagon) : ℝ :=
  area_hexagon h + 2 * Real.pi * (circumradius h)^2

theorem hexagon_roll_area_theorem (h : RegularHexagon) :
  area_enclosed_by_roll h = area_hexagon h + 2 * Real.pi * (circumradius h)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_roll_area_theorem_l1043_104350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_condition_l1043_104363

-- Define the floor function
noncomputable def floor (r : ℝ) : ℤ := Int.floor r

-- Define the set A(x)
def A (x : ℝ) : Set ℤ := {n : ℤ | ∃ (m : ℕ+), n = floor (m * x)}

-- State the theorem
theorem irrational_condition (α : ℝ) (h_irr : Irrational α) (h_gt1 : α > 1) :
  (∀ β : ℝ, β > 0 → A α ⊃ A β → ∃ (n : ℤ), β = n * α) ↔ α > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_condition_l1043_104363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_fund_ratio_l1043_104330

theorem class_fund_ratio : 
  ∀ (num_ten_bills num_twenty_bills : ℕ),
  num_twenty_bills = 3 →
  10 * num_ten_bills + 20 * num_twenty_bills = 120 →
  num_ten_bills = 2 * num_twenty_bills :=
λ num_ten_bills num_twenty_bills h1 h2 ↦
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_fund_ratio_l1043_104330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1043_104344

noncomputable def f (x : ℝ) := Real.sqrt (2 * x^2 - x + 3) + (2 : ℝ)^(Real.sqrt (x^2 - x))

theorem f_minimum_value (x : ℝ) (h : Real.arcsin x > Real.arccos x) :
  f x ≥ 3 ∧ ∃ y, f y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l1043_104344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_product_l1043_104387

/-- The Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Twin primes property -/
def is_twin_primes (p q : ℕ) : Prop := Nat.Prime p ∧ Nat.Prime q ∧ p - q = 2

theorem twin_primes_product (p q : ℕ) :
  is_twin_primes p q →
  φ (p * q) = 120 →
  p * q = 143 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twin_primes_product_l1043_104387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weeks_played_is_two_l1043_104395

/-- Represents the game completion scenario --/
structure GameCompletion where
  initialDailyHours : ℝ
  initialCompletionPercentage : ℝ
  increasedDailyHours : ℝ
  remainingDays : ℝ

/-- Calculates the initial number of weeks played --/
noncomputable def initialWeeksPlayed (gc : GameCompletion) : ℝ :=
  (gc.initialCompletionPercentage * gc.increasedDailyHours * gc.remainingDays) /
  ((1 - gc.initialCompletionPercentage) * 7 * gc.initialDailyHours)

/-- Theorem stating that the initial number of weeks played is 2 --/
theorem initial_weeks_played_is_two (gc : GameCompletion) 
  (h1 : gc.initialDailyHours = 4)
  (h2 : gc.initialCompletionPercentage = 0.4)
  (h3 : gc.increasedDailyHours = 7)
  (h4 : gc.remainingDays = 12) :
  initialWeeksPlayed gc = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_weeks_played_is_two_l1043_104395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l1043_104317

noncomputable def f (m : ℕ+) (x : ℝ) : ℝ := x^(m.val - 3 : ℤ)

theorem power_function_range (m : ℕ+) 
  (h_sym : ∀ x, f m x = f m (-x))
  (h_decr : ∀ x y, 0 < x → x < y → f m y < f m x) :
  {a : ℝ | f m (a + 1 - m.val) < f m (3 - 2*a - m.val)} = 
  {a : ℝ | 2/3 < a ∧ a < 1 ∨ 1 < a ∧ a < 2} := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_range_l1043_104317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l1043_104338

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

-- Theorem 1
theorem solution_part1 :
  ∀ x : ℝ, f 2 x ≥ 0 ↔ x ≤ 1 ∨ x ≥ 3/2 :=
sorry

-- Theorem 2
theorem solution_part2 :
  ∀ x : ℝ, (∀ a : ℝ, a ∈ Set.Icc (-2) 2 → f a x < 0) → x ∈ Set.Ioo 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l1043_104338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l1043_104332

theorem probability_factor_of_36 : 
  (Finset.filter (fun n : ℕ => n ∣ 36) (Finset.range 37)).card / 36 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_36_l1043_104332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1043_104321

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

-- State the theorem
theorem f_properties :
  -- Part 1: Tangent line at x = 0
  (∀ y, y = f 0 + (Real.exp 0 - 2 * 0) * (y - 0) ↔ y = x) ∧
  -- Part 2: Lower bound for f(x)
  (∀ x, f x ≥ -x^2 + x) ∧
  -- Part 3: Condition for f(x) > kx
  (∀ k, (∀ x, x > 0 → f x > k * x) ↔ k < Real.exp 1 - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1043_104321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_average_salary_main_theorem_l1043_104306

/-- Calculates the average salary of employees given the problem conditions -/
def average_salary (num_employees : ℕ) (manager_salary : ℕ) (average_increase : ℕ) : ℚ := 
  ((num_employees + 1 : ℕ) * average_increase * num_employees - manager_salary : ℚ) / num_employees

/-- Proves that the average salary of employees is 1500, given the conditions of the problem -/
theorem employee_average_salary
  (num_employees : ℕ)
  (manager_salary : ℕ)
  (average_increase : ℕ)
  (h1 : num_employees = 20)
  (h2 : manager_salary = 4650)
  (h3 : average_increase = 150)
  : (num_employees * average_salary num_employees manager_salary average_increase + manager_salary : ℚ) / (num_employees + 1) = 
    average_salary num_employees manager_salary average_increase + average_increase :=
by
  sorry

/-- The main theorem proving that the average salary of employees is 1500 -/
theorem main_theorem 
  (num_employees : ℕ)
  (manager_salary : ℕ)
  (average_increase : ℕ)
  (h1 : num_employees = 20)
  (h2 : manager_salary = 4650)
  (h3 : average_increase = 150)
  : average_salary num_employees manager_salary average_increase = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_average_salary_main_theorem_l1043_104306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1043_104343

theorem power_equation_solution (y : ℝ) : (9 : ℝ)^y = (3 : ℝ)^14 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1043_104343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_redistribution_l1043_104374

/-- Represents a chessboard with an odd number of squares. -/
structure Chessboard where
  size : Nat
  size_odd : Odd size

/-- Represents the color of a cell on the chessboard. -/
inductive CellColor
  | Black
  | White

/-- Represents a cell on the chessboard. -/
structure Cell where
  row : Nat
  col : Nat
  color : CellColor

/-- Defines when two cells are adjacent. -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ (c1.col + 1 = c2.col ∨ c2.col + 1 = c1.col)) ∨
  (c1.col = c2.col ∧ (c1.row + 1 = c2.row ∨ c2.row + 1 = c1.row))

/-- Represents a pawn on the chessboard. -/
structure Pawn where
  initial_position : Cell
  final_position : Cell

/-- Represents a valid redistribution of pawns. -/
def valid_redistribution (board : Chessboard) (pawns : List Pawn) : Prop :=
  pawns.length = board.size * board.size ∧
  ∀ p : Pawn, p ∈ pawns → adjacent p.initial_position p.final_position

/-- The main theorem stating that a valid redistribution is impossible. -/
theorem no_valid_redistribution (board : Chessboard) :
  ¬∃ pawns : List Pawn, valid_redistribution board pawns := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_redistribution_l1043_104374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_fractions_l1043_104355

theorem tan_sum_pi_fractions : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_pi_fractions_l1043_104355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_production_february_l1043_104326

/-- Represents the month of production --/
inductive Month
| October
| November
| December
| January
| February
deriving Repr

/-- Convert Month to a natural number --/
def Month.toNat : Month → Nat
| October => 0
| November => 1
| December => 2
| January => 3
| February => 4

/-- Calculates the number of carrot cakes produced in February --/
def carrot_cakes_february (initial : Nat) (monthly_increase : Nat) (start_month : Month) : Nat :=
  initial + monthly_increase * (Month.February.toNat - start_month.toNat)

/-- Calculates the number of dozens of chocolate chip cookies produced in February --/
def cookies_february (initial : Nat) (monthly_increase : Nat) (start_month : Month) : Nat :=
  initial + monthly_increase * (Month.February.toNat - start_month.toNat)

/-- Calculates the number of cinnamon rolls produced in February --/
def cinnamon_rolls_february (initial : Nat) (doubling_months : Nat) : Nat :=
  initial * (2 ^ doubling_months)

/-- Theorem stating the production in February --/
theorem bakery_production_february :
  carrot_cakes_february 19 2 Month.October = 27 ∧
  cookies_february 50 10 Month.November = 80 ∧
  cinnamon_rolls_february 30 2 = 120 := by
  sorry

#eval carrot_cakes_february 19 2 Month.October
#eval cookies_february 50 10 Month.November
#eval cinnamon_rolls_february 30 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_production_february_l1043_104326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l1043_104336

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = Real.sqrt 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l1043_104336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l1043_104307

noncomputable def m (ω : ℝ) (x : ℝ) : ℝ × ℝ := (2 * Real.cos (ω * x), 1)

noncomputable def n (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (ω * x) - Real.cos (ω * x), a)

noncomputable def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := (m ω x).1 * (n ω a x).1 + (m ω x).2 * (n ω a x).2

def is_periodic (g : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, g (x + T) = g x

def smallest_positive_period (g : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic g T ∧ T > 0 ∧ ∀ T' > 0, is_periodic g T' → T ≤ T'

theorem vector_dot_product_properties (ω : ℝ) (a : ℝ) :
  (ω > 0) →
  (smallest_positive_period (f ω a) π) →
  (∀ x, f ω a x ≤ 3) →
  (∃ x, f ω a x = 3) →
  (ω = 1 ∧ a = 2) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    (k : ℝ) * π - π / 6 ≤ x ∧ x ≤ (k : ℝ) * π + π / 3 →
    ∀ y : ℝ, x < y → f ω a x < f ω a y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_properties_l1043_104307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1043_104300

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: For an ellipse with equation x²/3 + y²/m = 1 and eccentricity 1/2, m is either 9/4 or 4 -/
theorem ellipse_m_values (m : ℝ) (h_m_pos : 0 < m) :
  (∃ e : Ellipse, (e.a^2 = 3 ∧ e.b^2 = m) ∨ (e.a^2 = m ∧ e.b^2 = 3)) →
  (∃ e : Ellipse, eccentricity e = 1/2) →
  m = 9/4 ∨ m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l1043_104300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_video_production_time_l1043_104375

/-- Represents the time taken for various activities in video production -/
structure VideoProductionTime where
  setup : ℚ
  paintingPerVideo : ℚ
  cleanup : ℚ
  editingPerVideo : ℚ

/-- Calculates the time taken to produce a single speed painting video -/
def timePerVideo (t : VideoProductionTime) (batchSize : ℕ) : ℚ :=
  (t.setup + t.cleanup + t.paintingPerVideo * batchSize + t.editingPerVideo * batchSize) / batchSize

/-- Theorem stating that Rachel's time to produce a single speed painting video is 3 hours -/
theorem rachel_video_production_time :
  let t : VideoProductionTime := {
    setup := 1,
    paintingPerVideo := 1,
    cleanup := 1,
    editingPerVideo := 3/2
  }
  let batchSize : ℕ := 4
  timePerVideo t batchSize = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_video_production_time_l1043_104375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1043_104390

/-- The number of sides of the first polygon -/
def n : ℕ := 3

/-- The number of sides of the second polygon -/
def m : ℕ := n + 1

/-- The measure of each angle in the first polygon -/
noncomputable def angle_P1 (n : ℕ) : ℝ := 180 - 360 / n

/-- The measure of each angle in the second polygon -/
noncomputable def angle_P2 (n : ℕ) : ℝ := 180 - 360 / (n + 1)

/-- The theorem stating that n = 3 is the only solution -/
theorem unique_solution : 
  (∀ n : ℕ, n > 2 → angle_P2 n = 1.5 * angle_P1 n) → n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1043_104390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_formula_l1043_104303

/-- The area of the figure located outside three mutually tangent circles of radius r
    and bounded by the arcs between the points of tangency -/
noncomputable def areaOutsideCircles (r : ℝ) : ℝ :=
  r^2 * (Real.sqrt 3 - Real.pi / 2)

/-- Theorem stating that the area of the figure located outside three mutually tangent circles
    of radius r and bounded by the arcs between the points of tangency is r^2(√3 - π/2) -/
theorem area_outside_circles_formula (r : ℝ) (hr : r > 0) :
  areaOutsideCircles r = r^2 * (Real.sqrt 3 - Real.pi / 2) := by
  -- Unfold the definition of areaOutsideCircles
  unfold areaOutsideCircles
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_formula_l1043_104303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_seed_problem_corn_seed_problem_solution_l1043_104337

theorem corn_seed_problem (num_kids : ℕ) (ears_per_row : ℕ) (seeds_per_ear : ℕ) 
  (pay_per_row : ℚ) (dinner_cost : ℚ) (bags_used : ℕ) : ℕ :=
  let total_pay := 2 * dinner_cost
  let rows_planted := total_pay / pay_per_row
  let seeds_per_row := ears_per_row * seeds_per_ear
  let total_seeds := rows_planted * seeds_per_row
  let seeds_per_bag := total_seeds / bags_used
  (seeds_per_bag.num / seeds_per_bag.den).natAbs

theorem corn_seed_problem_solution :
  corn_seed_problem 4 70 2 (3/2) 36 140 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_seed_problem_corn_seed_problem_solution_l1043_104337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_inequality_existence_l1043_104340

theorem unique_inequality_existence (a : ℕ → ℕ) 
  (h_pos : ∀ n, a n > 0) 
  (h_incr : ∀ n, a n < a (n + 1)) : 
  ∃! n : ℕ, n ≥ 1 ∧ 
    a n < (Finset.range (n + 1)).sum (λ i ↦ a i) / n ∧ 
    (Finset.range (n + 1)).sum (λ i ↦ a i) / n ≤ a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_inequality_existence_l1043_104340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_selection_schemes_l1043_104302

/-- Represents the number of people in the selection pool -/
def total_people : ℕ := 5

/-- Represents the number of roles to be filled -/
def total_roles : ℕ := 4

/-- Represents the number of people restricted to certain roles -/
def restricted_people : ℕ := 2

/-- Represents the number of roles that restricted people can take -/
def restricted_roles : ℕ := 2

/-- Represents the number of unrestricted people -/
def unrestricted_people : ℕ := total_people - restricted_people

/-- The total number of selection schemes -/
def total_schemes : ℕ := 36

theorem volunteer_selection_schemes :
  (Nat.choose restricted_people 1 * Nat.choose restricted_roles 1 * (Nat.factorial unrestricted_people / Nat.factorial (unrestricted_people - (total_roles - 1)))) +
  (Nat.factorial restricted_people * (Nat.factorial unrestricted_people / Nat.factorial (unrestricted_people - (total_roles - restricted_people)))) = total_schemes := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_selection_schemes_l1043_104302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_right_angle_l1043_104342

/-- The angular speed of the hour hand in degrees per minute -/
noncomputable def hour_hand_speed : ℝ := 1 / 2

/-- The angular speed of the minute hand in degrees per minute -/
noncomputable def minute_hand_speed : ℝ := 6

/-- The angle between the hands when they form a right angle -/
def right_angle : ℝ := 90

/-- The time after noon when the clock hands first form a right angle -/
noncomputable def right_angle_time : ℝ := 180 / 11

theorem clock_hands_right_angle :
  (minute_hand_speed * right_angle_time - hour_hand_speed * right_angle_time) = right_angle :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_right_angle_l1043_104342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_l1043_104362

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_perp : x * 1 + 1 * (y - 1) = 0) : 
  1 / x + 4 / y ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_l1043_104362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_3_25th_occurrence_l1043_104347

/-- Count the occurrences of a digit in a natural number -/
def countDigitOccurrences (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Count the occurrences of a digit in a range of natural numbers -/
def countDigitOccurrencesInRange (start finish d : ℕ) : ℕ := sorry

/-- Find the number where a digit appears for the nth time in the sequence of natural numbers -/
def findNthOccurrence (d : ℕ) (n : ℕ) : ℕ := sorry

theorem digit_3_25th_occurrence :
  findNthOccurrence 3 25 = 131 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_3_25th_occurrence_l1043_104347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_halves_l1043_104391

/-- The function f(x) = (x + 2) / (6x - 9) has a vertical asymptote at x = 3/2 -/
theorem vertical_asymptote_at_three_halves :
  ∃ (f : ℝ → ℝ) (ε : ℝ), 
    (∀ x : ℝ, f x = (x + 2) / (6 * x - 9)) ∧
    ε > 0 ∧ 
    ∀ (δ : ℝ), 0 < δ → δ < ε →
      (|f (3/2 + δ)| > 1/δ ∧ |f (3/2 - δ)| > 1/δ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_three_halves_l1043_104391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_is_159_l1043_104393

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The first term of our sequence is 3. -/
def FirstTerm : ℕ := 3

/-- The second term of our sequence is 7. -/
def SecondTerm : ℕ := 7

/-- Our sequence. -/
def OurSequence : ℕ → ℕ := sorry

/-- The 40th term of our sequence. -/
def FortiethTerm : ℕ := OurSequence 40

theorem fortieth_term_is_159 :
  ArithmeticSequence OurSequence ∧ 
  OurSequence 1 = FirstTerm ∧ 
  OurSequence 2 = SecondTerm →
  FortiethTerm = 159 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_is_159_l1043_104393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_approximation_l1043_104315

-- Define the approximation relation
def approx (a b : ℝ) : Prop := ∃ ε > 0, |a - b| < ε

-- State the theorem
theorem cube_root_approximation (x : ℝ) :
  approx (Real.rpow 326 (1/3)) 6.882 →
  approx (Real.rpow x (1/3)) 68.82 →
  approx x 326000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_approximation_l1043_104315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_2013_l1043_104364

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 60 / 100
def max_capacity : ℕ := 2000
def years_passed : ℕ := 3  -- From 2010 to 2013

def participants_uncapped (n : ℕ) : ℚ :=
  initial_participants * (1 + annual_increase_rate) ^ n

def participants (n : ℕ) : ℕ :=
  min (Int.toNat ⌊participants_uncapped n⌋) max_capacity

theorem participants_2013 :
  participants years_passed = 2000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_2013_l1043_104364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_is_120_l1043_104308

/-- The coefficient of x^2 in the expansion of (x + 1/x + 2)^5 -/
def coefficientX2 : ℕ :=
  (Finset.range 6).sum (fun k => 
    Nat.choose 5 k * (3^(5-k)) * (2^k) * 
    (Finset.range (k+1)).sum (fun j => Nat.choose k j))

theorem coefficient_x2_is_120 : coefficientX2 = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x2_is_120_l1043_104308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_period_l1043_104386

-- Define the vectors and function
noncomputable def m : ℝ × ℝ := (2, -2 * Real.sqrt 3)
noncomputable def n (B : ℝ) : ℝ × ℝ := (Real.cos B, Real.sin B)
noncomputable def a (x : ℝ) : ℝ × ℝ := (1 + Real.sin (2 * x), Real.cos (2 * x))
noncomputable def f (x B : ℝ) : ℝ := (a x).1 * (n B).1 + (a x).2 * (n B).2

-- State the theorem
theorem triangle_angle_and_function_period :
  ∀ (A B C : ℝ),
  -- A, B, C are interior angles of a triangle
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π →
  -- m ⊥ n
  m.1 * (n B).1 + m.2 * (n B).2 = 0 →
  -- Prove that B = π/6
  B = π / 6 ∧
  -- Prove that the smallest positive period of f(x) is π
  (∀ (x : ℝ), f (x + π) B = f x B) ∧
  (∀ (T : ℝ), 0 < T ∧ T < π → ∃ (x : ℝ), f (x + T) B ≠ f x B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_period_l1043_104386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_boxes_volume_l1043_104379

def box_edge_length : ℝ := 5

def box_volume (edge : ℝ) : ℝ := edge^3

def total_volume (num_boxes : ℕ) (edge : ℝ) : ℝ :=
  (box_volume edge) * (num_boxes : ℝ)

theorem four_boxes_volume :
  total_volume 4 box_edge_length = 500 := by
  -- Unfold definitions
  unfold total_volume
  unfold box_volume
  unfold box_edge_length
  -- Simplify
  simp
  -- Check equality
  norm_num

#eval total_volume 4 box_edge_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_boxes_volume_l1043_104379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1043_104351

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point on a hyperbola -/
def PointOnHyperbola (h : Hyperbola) := { p : ℝ × ℝ // (p.1^2 / h.a^2) - (p.2^2 / h.b^2) = 1 }

/-- Foci of a hyperbola -/
noncomputable def foci (h : Hyperbola) : ℝ × ℝ := 
  let c := Real.sqrt (h.a^2 + h.b^2)
  (-c, c)

/-- Theorem: Eccentricity of a hyperbola under specific conditions -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (p : PointOnHyperbola h)
  (h_perp : (p.val.2 - (foci h).2) / (p.val.1 - (foci h).2) = 0)
  (h_sin : Real.sin (Real.arctan ((p.val.2 - (foci h).1) / (p.val.1 - (foci h).1))) = 1/3) :
  let c := Real.sqrt (h.a^2 + h.b^2)
  c / h.a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1043_104351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1043_104372

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : tan α = 4/3) : 
  ((sin α)^2 + sin (2*α)) / ((cos α)^2 + cos (2*α)) = 20 ∧ 
  sin (2*π/3 - α) = (2 + 3*Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1043_104372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_given_points_l1043_104329

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) is (y₂ - y₁) / (x₂ - x₁) -/
def line_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ := (y₂ - y₁) / (x₂ - x₁)

/-- Theorem: The slope of the line passing through (-4, 6) and (3, -4) is -10/7 -/
theorem slope_through_given_points :
  line_slope (-4) 6 3 (-4) = -10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_through_given_points_l1043_104329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_equation_l1043_104392

/-- A material point moves with velocity proportional to distance traveled. -/
def velocity_proportional_to_distance (s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, deriv s t = k * s t

/-- The equation of motion for a material point -/
def equation_of_motion (s : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, s t = Real.exp (t / 2)

/-- Theorem stating the equation of motion for the given conditions -/
theorem motion_equation 
  (s : ℝ → ℝ)
  (h1 : velocity_proportional_to_distance s)
  (h2 : s 0 = 1)
  (h3 : s 2 = Real.exp 1) :
  equation_of_motion s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motion_equation_l1043_104392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1043_104325

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {2, 3}
def B : Set Int := {-1, 0}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {2, 3} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l1043_104325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_over_two_years_l1043_104354

/-- 
Given a price that increases by 25% in the first year and decreases by 15% in the second year,
prove that the final price is 106.25% of the original price.
-/
theorem price_change_over_two_years (initial_price : ℝ) :
  initial_price * (1 + 0.25) * (1 - 0.15) = initial_price * 1.0625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_over_two_years_l1043_104354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noncongruent_triangles_count_l1043_104333

-- Define the points
variable (A B C M N O : EuclideanSpace ℝ (Fin 2))

-- Define the isosceles triangle condition
variable (h_isosceles : dist A B = dist A C)

-- Define the midpoint conditions
variable (h_M_midpoint : M = (A + B) / 2)
variable (h_N_midpoint : N = (A + C) / 2)
variable (h_O_midpoint : O = (B + C) / 2)

-- Define a function to count noncongruent triangles
def count_noncongruent_triangles (A B C M N O : EuclideanSpace ℝ (Fin 2)) : ℕ := sorry

-- State the theorem
theorem noncongruent_triangles_count :
  count_noncongruent_triangles A B C M N O = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_noncongruent_triangles_count_l1043_104333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_cubed_minus_two_l1043_104309

-- Define the complex number ω
noncomputable def ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I

-- State the theorem
theorem omega_cubed_minus_two : ω^3 - 2 = -1 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_cubed_minus_two_l1043_104309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sequence_l1043_104318

def sequenceA (a₀ : ℝ) : ℕ → ℝ
  | 0 => a₀
  | n + 1 => -3 * sequenceA a₀ n + 2^n

theorem strictly_increasing_sequence (a₀ : ℝ) :
  a₀ = 1/5 → ∀ n : ℕ, sequenceA a₀ n < sequenceA a₀ (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_sequence_l1043_104318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_theorem_l1043_104339

/-- Polynomials with nonnegative real coefficients -/
def NonnegPoly := {p : Polynomial ℝ // ∀ i, 0 ≤ p.coeff i}

/-- The minimum degree of polynomial f in the equation x^2 - cx + 1 = f(x) / g(x) -/
noncomputable def min_degree (c : ℝ) : ℕ :=
  if c ≥ 2 then 0
  else Int.ceil (Real.pi / Real.arccos (c / 2)) |>.toNat

theorem min_degree_theorem (c : ℝ) (hc : c > 0) :
  (c ≥ 2 → ¬∃ (f g : NonnegPoly), f.1 ≠ 0 ∧ g.1 ≠ 0 ∧
    ∀ x, x^2 - c*x + 1 = f.1.eval x / g.1.eval x) ∧
  (0 < c ∧ c < 2 → ∀ (f g : NonnegPoly), f.1 ≠ 0 → g.1 ≠ 0 →
    (∀ x, x^2 - c*x + 1 = f.1.eval x / g.1.eval x) →
    f.1.natDegree ≥ min_degree c ∧
    ∃ (f' g' : NonnegPoly), f'.1 ≠ 0 ∧ g'.1 ≠ 0 ∧
      (∀ x, x^2 - c*x + 1 = f'.1.eval x / g'.1.eval x) ∧
      f'.1.natDegree = min_degree c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_theorem_l1043_104339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1043_104370

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 11) / Real.log (1/2)

-- State the theorem
theorem monotonic_increase_interval :
  ∀ x y, x < y → x < 3 → y < 3 → StrictMonoOn f (Set.Ioo x y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1043_104370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_4_45_l1043_104334

/-- The angle of the minute hand on a clock face at a given minute -/
def minuteHandAngle (minute : ℕ) : ℚ :=
  (minute : ℚ) * 6

/-- The angle of the hour hand on a clock face at a given hour and minute -/
def hourHandAngle (hour : ℕ) (minute : ℕ) : ℚ :=
  ((hour % 12 : ℚ) * 30 + (minute : ℚ) * (1/2))

/-- The smaller angle between two angles on a circle -/
def smallerAngle (angle1 : ℚ) (angle2 : ℚ) : ℚ :=
  min (abs (angle1 - angle2)) (360 - abs (angle1 - angle2))

theorem clock_angle_at_4_45 :
  smallerAngle (minuteHandAngle 45) (hourHandAngle 4 45) = 255/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_4_45_l1043_104334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_thirds_l1043_104388

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (6 * x - 8)

-- Theorem statement
theorem vertical_asymptote_at_four_thirds :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 4/3| ∧ |x - 4/3| < δ → |f x| > 1/ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_thirds_l1043_104388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l1043_104312

theorem contrapositive_sine_equality (A B : ℝ) :
  (A = B → Real.sin A = Real.sin B) ↔ (Real.sin A ≠ Real.sin B → A ≠ B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sine_equality_l1043_104312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_in_A_l1043_104327

def A : Set (ℕ × ℕ) := {p | p.1^2 + p.2^2 ≤ 3}

theorem count_elements_in_A : Finset.card (Finset.filter (fun p => p.1^2 + p.2^2 ≤ 3) (Finset.product (Finset.range 2) (Finset.range 2))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_in_A_l1043_104327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_25_is_thursday_l1043_104369

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => dayAfter (nextDay start) n

theorem january_25_is_thursday (h : DayOfWeek) 
  (h_christmas : h = DayOfWeek.Monday) : 
  dayAfter h 31 = DayOfWeek.Thursday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_25_is_thursday_l1043_104369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sides_ratio_bound_l1043_104316

/-- In a triangle ABC, if sides a, b, c opposite to angles A, B, C form a geometric sequence,
    then (sin A cot C + cos A) / (sin B cot C + cos B) is bounded by (√5 - 1)/2 and (√5 + 1)/2 -/
theorem triangle_geometric_sides_ratio_bound (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a * Real.sin A = b * Real.sin B →
  a * Real.sin A = c * Real.sin C →
  ∃ q : ℝ, b = a * q ∧ c = a * q^2 →
  a + b > c ∧ b + c > a ∧ a + c > b →
  let ratio := (Real.sin A * Real.cos C / Real.sin C + Real.cos A) / 
               (Real.sin B * Real.cos C / Real.sin C + Real.cos B)
  (Real.sqrt 5 - 1) / 2 < ratio ∧ ratio < (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_geometric_sides_ratio_bound_l1043_104316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_weights_l1043_104352

-- Define the type for weights
def Weight := Fin 7

-- Define the mass function
def mass (w : Weight) : ℕ := sorry

-- Define the weighing function
def weigh (w₁ w₂ : List Weight) : Ordering := sorry

-- Theorem statement
theorem identify_weights :
  ∀ (m : Weight → ℕ),
  (∀ w, m w ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ)) →
  (∀ w₁ w₂, w₁ ≠ w₂ → m w₁ ≠ m w₂) →
  ∃ (w₁ w₂ w₃ w₄ w₅ w₆ : List Weight),
    (weigh w₁ w₂ = Ordering.lt ∧
     weigh w₃ w₄ = Ordering.lt ∧
     weigh w₅ w₆ = Ordering.gt) →
    ∀ w, m w = mass w :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identify_weights_l1043_104352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1043_104384

-- Define the circles and points
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 100
def circle_P (x y : ℝ) : Prop := (x - 14)^2 + y^2 = 16

-- Define the center points
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (14, 0)

-- Define the tangent point Q
def Q : ℝ × ℝ := (10, 0)

-- Define the tangent points T and S
noncomputable def T : ℝ × ℝ := sorry
noncomputable def S : ℝ × ℝ := sorry

-- State the theorem
theorem external_tangent_length :
  circle_O T.1 T.2 → 
  circle_P S.1 S.2 →
  (T.1 - S.1)^2 + (T.2 - S.2)^2 = (4 * Real.sqrt 10)^2 →
  (O.1 - S.1)^2 + (O.2 - S.2)^2 = 260 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_length_l1043_104384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l1043_104310

noncomputable def z₁ (m : ℝ) : ℂ := m^2 + 2*m - 3 + (m - 1)*Complex.I

noncomputable def z₂ (z₁ : ℂ) : ℂ := (4 - 2*Complex.I) / ((1 + (1/4)*z₁) * Complex.I)

theorem complex_problem (m : ℝ) 
  (h₁ : z₁ m = Complex.I * Complex.im (z₁ m)) :
  m = -3 ∧ Complex.abs (z₂ (z₁ m)) = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l1043_104310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1043_104365

def sequence_formula (n : ℕ) : ℚ :=
  1 + (1/2)^(n-1)

theorem sequence_theorem (a : ℕ → ℚ) (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = (1/2) * a n + 1/2) : 
    ∀ n, a n = sequence_formula n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l1043_104365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_angle_l1043_104359

-- Define a sphere inscribed in a cone
structure InscribedSphereInCone where
  R : ℝ  -- radius of the sphere
  h : ℝ  -- height of the cone
  touching : h > R  -- condition for surfaces touching

-- Define the volumes
noncomputable def sphereVolume (s : InscribedSphereInCone) : ℝ := (4/3) * Real.pi * s.R^3
noncomputable def coneVolume (s : InscribedSphereInCone) : ℝ := (1/3) * Real.pi * s.R^2 * s.h
noncomputable def betweenVolume (s : InscribedSphereInCone) : ℝ := sphereVolume s - coneVolume s

-- State the theorem
theorem inscribed_sphere_cone_angle 
  (s : InscribedSphereInCone) 
  (h : betweenVolume s = (1/8) * sphereVolume s) : 
  2 * Real.arctan (s.R / (s.h - s.R)) = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_angle_l1043_104359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1043_104311

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | (n + 2) => 1 / (1 - sequence_a (n + 1))

theorem a_2019_value : sequence_a 2019 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_value_l1043_104311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_inscribed_circle_l1043_104301

/-- The area of a regular hexadecagon inscribed in a circle of radius r -/
noncomputable def hexadecagonArea (r : ℝ) : ℝ := 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)

/-- The ratio of the area of a circle to the area of its inscribed regular hexadecagon -/
noncomputable def circleHexadecagonRatio : ℝ := Real.pi / (4 * Real.sqrt (2 - Real.sqrt 2))

theorem hexadecagon_inscribed_circle (r : ℝ) (r_pos : r > 0) :
  (hexadecagonArea r = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) ∧
  (circleHexadecagonRatio = Real.pi / (4 * Real.sqrt (2 - Real.sqrt 2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_inscribed_circle_l1043_104301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_for_450_order_l1043_104313

/-- Calculates the service charge percentage based on the order amount -/
noncomputable def serviceChargePercentage (orderAmount : ℝ) : ℝ :=
  if orderAmount < 500 then 0.04
  else if orderAmount < 1000 then 0.05
  else 0.06

/-- Calculates the sales tax percentage based on the order amount -/
noncomputable def salesTaxPercentage (orderAmount : ℝ) : ℝ :=
  if orderAmount < 500 then 0.05
  else if orderAmount < 1000 then 0.06
  else 0.07

/-- Calculates the discount percentage based on the total amount after taxes -/
noncomputable def discountPercentage (totalAfterTaxes : ℝ) : ℝ :=
  if totalAfterTaxes < 600 then 0.05
  else if totalAfterTaxes < 800 then 0.10
  else 0.15

/-- Calculates the final amount paid after applying service charge, sales tax, and discount -/
noncomputable def finalAmountPaid (orderAmount : ℝ) : ℝ :=
  let serviceCharge := orderAmount * serviceChargePercentage orderAmount
  let salesTax := orderAmount * salesTaxPercentage orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := totalBeforeDiscount * discountPercentage totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_amount_for_450_order :
  ∃ (ε : ℝ), abs (finalAmountPaid 450 - 465.98) < ε ∧ ε < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_for_450_order_l1043_104313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_sum_l1043_104319

/-- Represents a face of the die -/
structure Face where
  value : Nat
  valid : 1 ≤ value ∧ value ≤ 6

/-- Represents the die -/
structure Die where
  faces : Finset Face
  all_unique : ∀ f1 f2, f1 ∈ faces → f2 ∈ faces → f1.value = f2.value → f1 = f2
  opposite_sum : ∀ f, f ∈ faces → ∃ f', f' ∈ faces ∧ f.value + f'.value = 10
  face_count : faces.card = 6

/-- Represents three faces meeting at a vertex -/
structure Vertex (d : Die) where
  f1 : Face
  f2 : Face
  f3 : Face
  in_die : f1 ∈ d.faces ∧ f2 ∈ d.faces ∧ f3 ∈ d.faces
  distinct : f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3

/-- The sum of values on faces meeting at a vertex -/
def vertexSum (d : Die) (v : Vertex d) : Nat :=
  v.f1.value + v.f2.value + v.f3.value

/-- The theorem to be proved -/
theorem max_vertex_sum (d : Die) : 
  ∃ v : Vertex d, vertexSum d v = 12 ∧ ∀ v' : Vertex d, vertexSum d v' ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertex_sum_l1043_104319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1043_104385

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem tangent_line_and_monotonicity 
  (a b : ℝ) 
  (h1 : f a b (π/3) = 0) 
  (h2 : HasDerivAt (f a b) 1 (π/3)) :
  (a = 1/2 ∧ b = -Real.sqrt 3 / 2) ∧
  ∀ x ∈ Set.Icc 0 (2*π/3), 
    HasDerivAt (λ y => 1/2 * y - f a b y) ((λ y => -1/2 + Real.sin (y + π/6)) x) x ∧
    (λ y => -1/2 + Real.sin (y + π/6)) x < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_monotonicity_l1043_104385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l1043_104349

/-- Represents a triangle with its properties -/
structure Triangle where
  leg1 : ℝ
  leg2 : ℝ
  isRight : Prop

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Returns the coordinates of the circumcenter of the triangle -/
noncomputable def Triangle.circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Returns the coordinates of the incenter of the triangle -/
noncomputable def Triangle.incenter (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Given a right triangle with legs 3 and 4, the distance between the centers of its circumscribed and inscribed circles is √5/2 -/
theorem distance_between_circle_centers (t : Triangle) 
  (h1 : t.isRight)
  (h2 : t.leg1 = 3)
  (h3 : t.leg2 = 4) :
  distance t.circumcenter t.incenter = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_circle_centers_l1043_104349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remove_color_maintain_property_l1043_104389

/-- Represents a point on the circle -/
structure Point where
  color : Nat

/-- Represents an arc on the circle -/
structure Arc where
  points : Finset Point

/-- Represents the circle with colored points -/
structure ColoredCircle where
  n : Nat
  points : Finset Point
  colors : Finset Nat

/-- The property that needs to be maintained -/
def hasUniqueColorProperty (circle : ColoredCircle) : Prop :=
  ∀ (arc : Arc), 1 ≤ arc.points.card ∧ arc.points.card ≤ 2 * circle.n - 1 →
    ∃ (color : Nat), (arc.points.filter (λ p => p.color = color)).card = 1

/-- The main theorem -/
theorem remove_color_maintain_property (circle : ColoredCircle) 
  (h1 : circle.n ≥ 2)
  (h2 : circle.points.card = 2 * circle.n)
  (h3 : circle.colors.card = circle.n)
  (h4 : ∀ c ∈ circle.colors, (circle.points.filter (λ p => p.color = c)).card = 2)
  (h5 : hasUniqueColorProperty circle) :
  ∃ (color : Nat), hasUniqueColorProperty 
    { n := circle.n - 1
    , points := circle.points.filter (λ p => p.color ≠ color)
    , colors := circle.colors.erase color } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remove_color_maintain_property_l1043_104389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1043_104394

/-- Given a hyperbola with equation x²/4 - y² = 1, its asymptotes are y = ±x/2 -/
theorem hyperbola_asymptotes :
  let hyperbola := {p : ℝ × ℝ | p.1^2/4 - p.2^2 = 1}
  let asymptotes := {p : ℝ × ℝ | p.2 = p.1/2 ∨ p.2 = -p.1/2}
  ∀ p : ℝ × ℝ, p ∈ asymptotes ↔ (∃ t : ℝ, t ≠ 0 ∧ (t * p.1, t * p.2) ∈ hyperbola) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1043_104394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_prime_consecutive_l1043_104368

theorem largest_non_prime_consecutive : ∃ (n : ℕ), 
  (n < 50) ∧ 
  (n > 9) ∧
  (∀ k : ℕ, k ∈ Finset.range 5 → ¬ Nat.Prime (n - k)) ∧
  (∀ m : ℕ, m > n → 
    ¬(∀ k : ℕ, k ∈ Finset.range 5 → ¬ Nat.Prime (m - k))) ∧
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_prime_consecutive_l1043_104368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_configuration_exists_l1043_104399

/-- A ray in a plane --/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- An intersection point of rays --/
structure Intersection where
  point : ℝ × ℝ
  rays : Finset Ray

/-- A configuration of rays --/
structure RayConfiguration where
  rays : Finset Ray
  intersections : Finset Intersection

/-- Predicate to check if a configuration is valid --/
def is_valid_configuration (config : RayConfiguration) : Prop :=
  Finset.card config.rays = 6 ∧
  Finset.card config.intersections = 4 ∧
  ∀ i ∈ config.intersections, Finset.card i.rays = 3 ∧
  ∀ r ∈ config.rays, ∀ s ∈ config.rays, r ≠ s → r.start ≠ s.start

theorem ray_configuration_exists : ∃ config : RayConfiguration, is_valid_configuration config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_configuration_exists_l1043_104399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_problem_l1043_104328

theorem unique_divisor_problem :
  ∃ (n : ℕ) (d : Fin 12 → ℕ),
    n > 0 ∧
    (∀ i : Fin 12, d i > 0) ∧
    (∀ i j : Fin 12, i < j → d i < d j) ∧
    d 0 = 1 ∧
    d 11 = n ∧
    (∀ k : ℕ, k > 0 → (k ∣ n ↔ ∃ i : Fin 12, d i = k)) ∧
    (let m := d 3 - 1
     d m = (d 0 + d 1 + d 3) * d 7) ∧
    n = 1989 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_divisor_problem_l1043_104328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1043_104371

-- Define the angle α
noncomputable def α : ℝ := 7 * Real.pi / 3

-- Define the point (√m, ∛m) on the terminal side of α
noncomputable def point (m : ℝ) : ℝ × ℝ := (Real.sqrt m, m ^ (1/3))

-- Theorem statement
theorem angle_terminal_side (m : ℝ) :
  point m = (Real.sqrt m, m ^ (1/3)) ∧ α = 7 * Real.pi / 3 → m = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l1043_104371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_counterfeit_weight_relation_l1043_104358

/-- Represents the outcome of a weighing -/
inductive WeighingResult
  | LighterThan
  | EqualTo
  | HeavierThan

/-- Represents a coin -/
structure Coin where
  weight : ℝ

/-- Represents a set of 8 coins, two of which are counterfeit -/
structure CoinSet where
  coins : Fin 8 → Coin
  genuine_weight : ℝ
  counterfeit_light : Fin 8
  counterfeit_heavy : Fin 8
  h_light : (coins counterfeit_light).weight < genuine_weight
  h_heavy : (coins counterfeit_heavy).weight > genuine_weight
  h_distinct : counterfeit_light ≠ counterfeit_heavy
  h_others : ∀ i, i ≠ counterfeit_light → i ≠ counterfeit_heavy → (coins i).weight = genuine_weight

/-- Represents a weighing action -/
def Weighing := (Fin 8 → Bool) → (Fin 8 → Bool) → WeighingResult

/-- The main theorem to be proved -/
theorem determine_counterfeit_weight_relation (cs : CoinSet) :
  ∃ (w₁ w₂ w₃ : Weighing),
    let f := λ (r₁ r₂ r₃ : WeighingResult) ↦ 
      ((cs.coins cs.counterfeit_light).weight + (cs.coins cs.counterfeit_heavy).weight) <
      (2 * cs.genuine_weight) ∨
      ((cs.coins cs.counterfeit_light).weight + (cs.coins cs.counterfeit_heavy).weight) =
      (2 * cs.genuine_weight) ∨
      ((cs.coins cs.counterfeit_light).weight + (cs.coins cs.counterfeit_heavy).weight) >
      (2 * cs.genuine_weight)
    ∀ s₁ s₂ s₃ t₁ t₂ t₃,
      (∀ i, (s₁ i → t₁ i) → (s₂ i → t₂ i) → (s₃ i → t₃ i) → i.val < 8) →
      f (w₁ s₁ t₁) (w₂ s₂ t₂) (w₃ s₃ t₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_counterfeit_weight_relation_l1043_104358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1043_104346

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := (x^2 - 8*x + 15) / 8

/-- The directrix of the parabola -/
noncomputable def directrix : ℝ := -5/8

/-- Theorem: The directrix of the given parabola is y = -5/8 -/
theorem parabola_directrix : 
  ∀ x : ℝ, ∃ y : ℝ, y = parabola_equation x → directrix = -5/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1043_104346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l1043_104322

/-- A line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → Prop

/-- The point (2, π/4) in polar coordinates -/
noncomputable def point : ℝ × ℝ := (2, Real.pi / 4)

/-- A line is parallel to the polar axis if it has a constant y-coordinate in Cartesian coordinates -/
def parallel_to_polar_axis (l : PolarLine) : Prop :=
  ∃ (k : ℝ), ∀ (ρ θ : ℝ), l.equation ρ θ → ρ * Real.sin θ = k

theorem polar_line_equation (l : PolarLine) :
  (∀ (ρ θ : ℝ), l.equation ρ θ ↔ ρ * Real.sin θ = Real.sqrt 2) →
  l.equation point.1 point.2 →
  parallel_to_polar_axis l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_line_equation_l1043_104322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1043_104377

/-- The value of k for which the line x = k intersects the parabola x = -3y^2 - 4y + 7 at exactly one point -/
noncomputable def intersection_k : ℝ := 25 / 3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∃! p : ℝ × ℝ, p.1 = intersection_k ∧ p.1 = parabola p.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1043_104377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_addition_for_divisibility_l1043_104398

theorem smallest_addition_for_divisibility (n m : Nat) (h : Nat.Prime m) :
  ∃ x : Nat, x = m - (n % m) ∧ 
    (∀ y : Nat, y < x → ¬(m ∣ (n + y))) ∧
    (m ∣ (n + x)) := by
  sorry

#eval Nat.gcd 956734 751

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_addition_for_divisibility_l1043_104398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1043_104345

-- Define the base a
variable (a : ℝ)

-- Define the conditions
variable (h1 : a > 0)
variable (h2 : a ≠ 1)
variable (h3 : Real.log 3 / Real.log a > Real.log 2 / Real.log a)

-- Define the function f
noncomputable def f (x : ℝ) := Real.log x / Real.log a

-- Define the condition for the difference between max and min values of f
variable (h4 : ∃ (x y : ℝ), x ∈ Set.Icc a (2 * a) ∧ y ∈ Set.Icc a (2 * a) ∧ f x - f y = 1)

-- Define the function g
noncomputable def g (x : ℝ) := |Real.log x / Real.log a - 1|

-- Theorem statement
theorem problem_solution :
  -- Part 1: a = 2
  a = 2 ∧
  -- Part 2: Solution set of log_(1/3)(x-1) > log_(1/3)(a-x)
  {x : ℝ | Real.log (x-1) / Real.log (1/3) > Real.log (a-x) / Real.log (1/3)} = Set.Ioo 1 (3/2) ∧
  -- Part 3: Monotonicity intervals of g
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → g x > g y) ∧
  (∀ x y, 2 < x ∧ x < y → g x < g y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1043_104345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l1043_104348

/-- Define the nested radical function recursively -/
noncomputable def nestedRadical : ℕ → ℝ
  | 0 => Real.sqrt (1 + 2018 * 2020)
  | n + 1 => Real.sqrt (1 + (2017 - n) * nestedRadical n)

/-- The main theorem stating that the nested radical equals 3 -/
theorem nested_radical_equals_three :
  nestedRadical 2017 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_radical_equals_three_l1043_104348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_coord_at_x_6_l1043_104381

/-- Given a line passing through points (3, -1, 4) and (7, 3, 0),
    the y-coordinate is 2 when the x-coordinate is 6. -/
theorem line_y_coord_at_x_6 :
  let p₁ : Fin 3 → ℝ := ![3, -1, 4]
  let p₂ : Fin 3 → ℝ := ![7, 3, 0]
  let direction : Fin 3 → ℝ := λ i => p₂ i - p₁ i
  let line (t : ℝ) : Fin 3 → ℝ := λ i => p₁ i + t * direction i
  let t₀ : ℝ := (6 - p₁ 0) / direction 0
  line t₀ 1 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_coord_at_x_6_l1043_104381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_inscribed_triangle_l1043_104324

/-- Represents a line in a plane -/
structure Line : Type :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Represents a circle in a plane -/
structure Circle : Type :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents a triangle in a plane -/
structure Triangle : Type :=
  (v1 v2 v3 : ℝ × ℝ)

/-- Checks if a triangle is inscribed in a circle -/
def inscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

/-- Gets a side of a triangle as a line -/
def side (t : Triangle) (i : Fin 3) : Line :=
  sorry

/-- Gets a line from the given three lines based on the permutation index -/
def getLine (ℓ₁ ℓ₂ ℓ₃ : Line) : Fin 3 → Line
| 0 => ℓ₁
| 1 => ℓ₂
| 2 => ℓ₃

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Given three distinct non-parallel lines and a circle, 
    it is possible to construct a triangle inscribed in the circle 
    with sides parallel to the given lines using only a straightedge 
    and parallel line construction. -/
theorem construct_inscribed_triangle 
  (ℓ₁ ℓ₂ ℓ₃ : Line) 
  (ω : Circle) 
  (h_distinct : ℓ₁ ≠ ℓ₂ ∧ ℓ₂ ≠ ℓ₃ ∧ ℓ₃ ≠ ℓ₁) 
  (h_nonparallel : ℓ₁.slope ≠ ℓ₂.slope ∧ ℓ₂.slope ≠ ℓ₃.slope ∧ ℓ₃.slope ≠ ℓ₁.slope) :
  ∃ (t : Triangle), 
    (inscribed t ω) ∧ 
    (∃ (p : Equiv.Perm (Fin 3)), 
      parallel (side t 0) (getLine ℓ₁ ℓ₂ ℓ₃ (p 0)) ∧
      parallel (side t 1) (getLine ℓ₁ ℓ₂ ℓ₃ (p 1)) ∧
      parallel (side t 2) (getLine ℓ₁ ℓ₂ ℓ₃ (p 2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_inscribed_triangle_l1043_104324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l1043_104357

/-- The volume of a cone with slant height 2 cm and base circumference 2π cm is (√3/3)π cm³ -/
theorem cone_volume (s r h : ℝ) (hs : s = 2) (hr : r = 1) (hh : h = Real.sqrt 3) :
  (1/3) * Real.pi * r^2 * h = (Real.sqrt 3 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l1043_104357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1043_104353

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem f_properties :
  -- Period
  (∀ x, f (x + π) = f x) ∧
  -- Maximum value
  (∃ x, f x = Real.sqrt 2 + 1) ∧ (∀ y, f y ≤ Real.sqrt 2 + 1) ∧
  -- Minimum value
  (∃ x, f x = 1 - Real.sqrt 2) ∧ (∀ y, f y ≥ 1 - Real.sqrt 2) ∧
  -- Axis of symmetry
  (∀ k : ℤ, ∀ x, f (x + (k * π / 2 + π / 8)) = f (k * π / 2 + π / 8 - x)) ∧
  -- Center of symmetry
  (∀ k : ℤ, ∀ x, f (k * π / 2 - π / 8 + x) = 2 - f (k * π / 2 - π / 8 - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1043_104353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_approximation_l1043_104331

-- Define the points
def P : ℝ × ℝ := (1, 3)
def Q : ℝ × ℝ := (3, 8)
def R : ℝ × ℝ := (8, 3)
def S : ℝ × ℝ := (10, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter of PQRS
noncomputable def perimeter : ℝ :=
  distance P Q + distance Q R + distance R S + distance S P

-- Theorem statement
theorem perimeter_approximation :
  ∃ (c d : ℤ), c + d = 7 ∧ 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |perimeter - (c * Real.sqrt 2 + d * Real.sqrt 10)| < ε := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_approximation_l1043_104331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_real_power_of_two_leq_zero_l1043_104396

theorem negation_of_exists_real_power_of_two_leq_zero :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_real_power_of_two_leq_zero_l1043_104396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_ratio_problem_l1043_104383

theorem sum_and_ratio_problem (u v : ℝ) 
  (sum_eq : u + v = 360)
  (ratio_eq : u / v = 1 / 1.1) :
  ∃ ε > 0, |v - u - 17| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_and_ratio_problem_l1043_104383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_sqrt3_neg1_l1043_104376

/-- Converts Cartesian coordinates to polar coordinates -/
noncomputable def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 && y < 0 
           then 2 * Real.pi - Real.arctan (abs y / abs x)
           else Real.arctan (y / x)
  (ρ, θ)

theorem cartesian_to_polar_sqrt3_neg1 :
  cartesian_to_polar (Real.sqrt 3) (-1) = (2, 11 * Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_to_polar_sqrt3_neg1_l1043_104376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1043_104314

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x^3 - 2*x^2 - x + 2)

def p : ℕ := 0  -- number of holes

def q : ℕ := 3  -- number of vertical asymptotes

def r : ℕ := 1  -- number of horizontal asymptotes

def s : ℕ := 0  -- number of oblique asymptotes

theorem asymptote_sum : p + 2*q + 3*r + 4*s = 9 := by
  rfl

#eval p + 2*q + 3*r + 4*s

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l1043_104314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_B_over_A_l1043_104373

/-- The function f(Z) that we want to maximize -/
def f (Z : ℂ) (A B C : ℝ) : ℝ := A * Complex.normSq Z + 2 * B * Z.re + C

/-- Theorem stating that the maximum of f occurs at Z = -B/A -/
theorem f_max_at_neg_B_over_A (A B C : ℝ) (hA : A < 0) :
  ∃ (Z : ℂ), ∀ (W : ℂ), f Z A B C ≥ f W A B C ∧ Z = (-B / A : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_neg_B_over_A_l1043_104373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_integers_l1043_104361

theorem divisibility_in_chosen_integers (n : ℕ) (h : n ≥ 1) :
  ∀ (S : Finset ℕ), S.card = n + 1 → (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2*n) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_integers_l1043_104361
