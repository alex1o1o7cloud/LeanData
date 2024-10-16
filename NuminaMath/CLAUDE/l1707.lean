import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_puzzle_l1707_170754

-- Define the lengths of the segments
def top_segment1 : ℝ := 2
def top_segment2 : ℝ := 3
def top_segment4 : ℝ := 4
def bottom_segment1 : ℝ := 3
def bottom_segment2 : ℝ := 5

-- Define X as a real number
def X : ℝ := sorry

-- State the theorem
theorem rectangle_puzzle :
  top_segment1 + top_segment2 + X + top_segment4 = bottom_segment1 + bottom_segment2 + (X + 1) →
  X = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_puzzle_l1707_170754


namespace NUMINAMATH_CALUDE_system1_neither_necessary_nor_sufficient_l1707_170715

-- Define the two systems of inequalities
def system1 (x y a b : ℝ) : Prop := x > a ∧ y > b
def system2 (x y a b : ℝ) : Prop := x + y > a + b ∧ x * y > a * b

-- Theorem stating that system1 is neither necessary nor sufficient for system2
theorem system1_neither_necessary_nor_sufficient :
  ¬(∀ x y a b : ℝ, system1 x y a b → system2 x y a b) ∧
  ¬(∀ x y a b : ℝ, system2 x y a b → system1 x y a b) :=
sorry

end NUMINAMATH_CALUDE_system1_neither_necessary_nor_sufficient_l1707_170715


namespace NUMINAMATH_CALUDE_vector_equation_l1707_170777

def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

theorem vector_equation (m n t : ℝ) :
  c t = m • a + n • b → t = 11 ∧ m + n = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_l1707_170777


namespace NUMINAMATH_CALUDE_average_marks_l1707_170723

def english_marks : ℕ := 86
def mathematics_marks : ℕ := 85
def physics_marks : ℕ := 92
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 95

def total_marks : ℕ := english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks :
  (total_marks : ℚ) / num_subjects = 89 := by sorry

end NUMINAMATH_CALUDE_average_marks_l1707_170723


namespace NUMINAMATH_CALUDE_hours_per_day_l1707_170799

/-- Given that there are 8760 hours in a year and 365 days in a year,
    prove that there are 24 hours in a day. -/
theorem hours_per_day :
  let hours_per_year : ℕ := 8760
  let days_per_year : ℕ := 365
  (hours_per_year / days_per_year : ℚ) = 24 := by
sorry

end NUMINAMATH_CALUDE_hours_per_day_l1707_170799


namespace NUMINAMATH_CALUDE_circle_center_correct_l1707_170710

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (2, 1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1707_170710


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l1707_170725

/-- The coordinates of the foci of the hyperbola x²/3 - y² = 1 are (-2, 0) and (2, 0) -/
theorem hyperbola_foci_coordinates :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 3 - y^2 = 1
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁ = (-2, 0) ∧ f₂ = (2, 0)) ∧
    (∀ x y, h x y ↔ (x - f₁.1)^2 / (f₂.1 - f₁.1)^2 - (y - f₁.2)^2 / ((f₂.1 - f₁.1)^2 / 3 - (f₂.2 - f₁.2)^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l1707_170725


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1707_170791

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →  -- probability of rolling a 2 on a fair 8-sided die
  (5 : ℝ) * p^4 * (1 - p) = 35 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l1707_170791


namespace NUMINAMATH_CALUDE_jills_salary_solution_l1707_170751

def jills_salary_problem (net_salary : ℝ) : Prop :=
  let discretionary_income := net_salary / 5
  let vacation_fund := 0.30 * discretionary_income
  let savings := 0.20 * discretionary_income
  let eating_out := 0.35 * discretionary_income
  let fitness_classes := 0.05 * discretionary_income
  let gifts_and_charity := 99
  vacation_fund + savings + eating_out + fitness_classes + gifts_and_charity = discretionary_income ∧
  net_salary = 4950

theorem jills_salary_solution :
  ∃ (net_salary : ℝ), jills_salary_problem net_salary :=
sorry

end NUMINAMATH_CALUDE_jills_salary_solution_l1707_170751


namespace NUMINAMATH_CALUDE_solve_equation_l1707_170788

theorem solve_equation (x : ℝ) (h : x + 1 = 5) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1707_170788


namespace NUMINAMATH_CALUDE_inequality_always_true_l1707_170734

theorem inequality_always_true (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1707_170734


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1707_170707

/-- The equation of the tangent line to y = x^3 - 1 at x = 1 is y = 3x - 3 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3 - 1) → 
  (∃ m b : ℝ, ∀ x' y' : ℝ, y' = m * x' + b ∧ 
    (y' = (x')^3 - 1 → x' = 1 → y' = m * x' + b) ∧
    (1 = 1^3 - 1 → 1 = m * 1 + b) ∧
    m = 3 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1707_170707


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l1707_170712

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -(Real.sqrt 3 / 2) * y

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

theorem parabola_and_line_intersection :
  -- Conditions
  (∀ x y, parabola x y → parabola (-x) y) → -- Symmetry about y-axis
  parabola 0 0 → -- Vertex at origin
  parabola (Real.sqrt 3) (-2 * Real.sqrt 3) → -- Passes through (√3, -2√3)
  -- Conclusion
  (∀ m, (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m) ↔ 
    m < Real.sqrt 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l1707_170712


namespace NUMINAMATH_CALUDE_speed_increase_percentage_l1707_170740

theorem speed_increase_percentage (distance : ℝ) (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  distance = 96 →
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 50 ∧
    distance / (current_speed * (1 + increase_percentage / 100)) = 
    distance / (current_speed - speed_reduction) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_speed_increase_percentage_l1707_170740


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1707_170732

theorem cubic_equation_solution (w : ℝ) :
  (w + 5)^3 = (w + 2) * (3 * w^2 + 13 * w + 14) →
  w^3 = -2 * w^2 + (35 / 2) * w + 97 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1707_170732


namespace NUMINAMATH_CALUDE_triangle_area_l1707_170784

theorem triangle_area (a b c : ℝ) (h₁ : a = 15) (h₂ : b = 36) (h₃ : c = 39) :
  (1/2) * a * b = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1707_170784


namespace NUMINAMATH_CALUDE_probability_one_of_each_l1707_170733

/-- The number of forks in the drawer -/
def num_forks : ℕ := 8

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 9

/-- The number of knives in the drawer -/
def num_knives : ℕ := 10

/-- The number of teaspoons in the drawer -/
def num_teaspoons : ℕ := 7

/-- The total number of silverware pieces in the drawer -/
def total_silverware : ℕ := num_forks + num_spoons + num_knives + num_teaspoons

/-- The number of pieces to be drawn -/
def draw_count : ℕ := 4

/-- The probability of drawing one of each type of silverware -/
theorem probability_one_of_each :
  (num_forks * num_spoons * num_knives * num_teaspoons : ℚ) / Nat.choose total_silverware draw_count = 40 / 367 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l1707_170733


namespace NUMINAMATH_CALUDE_manager_average_salary_l1707_170737

/-- Represents the salary distribution in Plutarch Enterprises -/
structure SalaryDistribution where
  total_employees : ℝ
  marketer_ratio : ℝ
  engineer_ratio : ℝ
  marketer_salary : ℝ
  engineer_salary : ℝ
  average_salary : ℝ

/-- Theorem stating the average salary of managers in Plutarch Enterprises -/
theorem manager_average_salary (sd : SalaryDistribution) 
  (h1 : sd.marketer_ratio = 0.7)
  (h2 : sd.engineer_ratio = 0.1)
  (h3 : sd.marketer_salary = 50000)
  (h4 : sd.engineer_salary = 80000)
  (h5 : sd.average_salary = 80000) :
  (sd.average_salary * sd.total_employees - 
   (sd.marketer_ratio * sd.marketer_salary * sd.total_employees + 
    sd.engineer_ratio * sd.engineer_salary * sd.total_employees)) / 
   ((1 - sd.marketer_ratio - sd.engineer_ratio) * sd.total_employees) = 185000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l1707_170737


namespace NUMINAMATH_CALUDE_pizza_combinations_l1707_170758

theorem pizza_combinations (n : ℕ) (h : n = 7) : 
  n + (n.choose 2) + (n.choose 3) = 63 := by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l1707_170758


namespace NUMINAMATH_CALUDE_slower_train_time_l1707_170774

/-- Represents a train traveling between two stations -/
structure Train where
  speed : ℝ
  remainingDistance : ℝ

/-- The problem setup -/
def trainProblem (fasterTrain slowerTrain : Train) : Prop :=
  fasterTrain.speed = 3 * slowerTrain.speed ∧
  fasterTrain.remainingDistance = slowerTrain.remainingDistance ∧
  fasterTrain.remainingDistance = 4 * fasterTrain.speed

/-- The theorem to prove -/
theorem slower_train_time
    (fasterTrain slowerTrain : Train)
    (h : trainProblem fasterTrain slowerTrain) :
    slowerTrain.remainingDistance / slowerTrain.speed = 12 := by
  sorry

#check slower_train_time

end NUMINAMATH_CALUDE_slower_train_time_l1707_170774


namespace NUMINAMATH_CALUDE_b_value_in_discriminant_l1707_170785

/-- For a quadratic equation ax^2 + bx + c = 0, 
    the discriminant is defined as b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 2x - 3 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

theorem b_value_in_discriminant :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x ↔ a*x^2 + b*x + c = 0) ∧
    b = -2 :=
sorry

end NUMINAMATH_CALUDE_b_value_in_discriminant_l1707_170785


namespace NUMINAMATH_CALUDE_relay_team_permutations_l1707_170747

theorem relay_team_permutations (n : ℕ) (h : n = 4) : Nat.factorial (n - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l1707_170747


namespace NUMINAMATH_CALUDE_infinite_divisible_by_76_and_unique_centers_l1707_170779

/-- Represents a cell in the spiral grid -/
structure Cell where
  x : ℤ
  y : ℤ

/-- The value at a node of the grid -/
def node_value (c : Cell) : ℕ := sorry

/-- The value at the center of a cell -/
def center_value (c : Cell) : ℕ := sorry

/-- The set of all cells in the infinite grid -/
def all_cells : Set Cell := sorry

theorem infinite_divisible_by_76_and_unique_centers :
  (∃ (S : Set Cell), Set.Infinite S ∧ ∀ c ∈ S, 76 ∣ center_value c) ∧
  (∀ c₁ c₂ : Cell, c₁ ≠ c₂ → center_value c₁ ≠ center_value c₂) := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisible_by_76_and_unique_centers_l1707_170779


namespace NUMINAMATH_CALUDE_sum_of_selected_numbers_l1707_170761

def set1 := Finset.Icc 10 19
def set2 := Finset.Icc 90 99

def is_valid_selection (s1 s2 : Finset ℕ) : Prop :=
  s1.card = 5 ∧ s2.card = 5 ∧ 
  s1 ⊆ set1 ∧ s2 ⊆ set2 ∧
  ∀ x ∈ s1, ∀ y ∈ s1, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s2, ∀ y ∈ s2, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s1, ∀ y ∈ s2, (x - y) % 10 ≠ 0

theorem sum_of_selected_numbers (s1 s2 : Finset ℕ) 
  (h : is_valid_selection s1 s2) : 
  (s1.sum id + s2.sum id) = 545 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_numbers_l1707_170761


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l1707_170753

theorem fenced_area_calculation (length width cutout_side : ℝ) 
  (h1 : length = 18.5)
  (h2 : width = 14)
  (h3 : cutout_side = 3.5) : 
  length * width - cutout_side * cutout_side = 246.75 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l1707_170753


namespace NUMINAMATH_CALUDE_abs_fraction_less_than_one_l1707_170798

theorem abs_fraction_less_than_one (x y : ℝ) 
  (hx : |x| < 1) (hy : |y| < 1) : 
  |((x - y) / (1 - x * y))| < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_fraction_less_than_one_l1707_170798


namespace NUMINAMATH_CALUDE_eagles_win_probability_l1707_170772

/-- The number of games in the series -/
def n : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 1/2

/-- The probability of winning exactly k games out of n -/
def prob_win (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The probability of winning at least 3 games out of 5 -/
def prob_win_at_least_three : ℚ :=
  prob_win 3 + prob_win 4 + prob_win 5

theorem eagles_win_probability : prob_win_at_least_three = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_eagles_win_probability_l1707_170772


namespace NUMINAMATH_CALUDE_differentiable_implies_continuous_l1707_170764

theorem differentiable_implies_continuous (f : ℝ → ℝ) (x₀ : ℝ) :
  DifferentiableAt ℝ f x₀ → ContinuousAt f x₀ := by
  sorry

end NUMINAMATH_CALUDE_differentiable_implies_continuous_l1707_170764


namespace NUMINAMATH_CALUDE_min_value_h_l1707_170757

theorem min_value_h (x : ℝ) (hx : x > 0) :
  x^2 + 1/x^2 + 1/(x^2 + 1/x^2) ≥ 2.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_h_l1707_170757


namespace NUMINAMATH_CALUDE_exists_minimal_period_greater_than_l1707_170744

/-- Definition of the sequence family F(x) -/
def F (x : ℝ) : (ℕ → ℝ) → Prop :=
  λ a => ∀ n, a (n + 1) = x - 1 / a n

/-- Definition of periodicity for a sequence -/
def IsPeriodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, a (n + p) = a n

/-- Definition of minimal period for the family F(x) -/
def IsMinimalPeriod (x : ℝ) (p : ℕ) : Prop :=
  (∀ a, F x a → IsPeriodic a p) ∧
  (∀ q, 0 < q → q < p → ∃ a, F x a ∧ ¬IsPeriodic a q)

/-- Main theorem statement -/
theorem exists_minimal_period_greater_than (P : ℕ) :
  ∃ x : ℝ, ∃ p : ℕ, p > P ∧ IsMinimalPeriod x p :=
sorry

end NUMINAMATH_CALUDE_exists_minimal_period_greater_than_l1707_170744


namespace NUMINAMATH_CALUDE_max_sections_with_five_lines_l1707_170701

/-- The number of sections created by drawing n line segments through a rectangle -/
def num_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else num_sections (n - 1) + n

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_with_five_lines :
  num_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_with_five_lines_l1707_170701


namespace NUMINAMATH_CALUDE_basket_probability_l1707_170730

-- Define the total number of shots
def total_shots : ℕ := 5

-- Define the number of successful shots
def successful_shots : ℕ := 3

-- Define the number of unsuccessful shots
def unsuccessful_shots : ℕ := 2

-- Define the probability of second and third shots being successful
def prob_second_third_successful : ℚ := 3/10

-- Theorem statement
theorem basket_probability :
  (total_shots = successful_shots + unsuccessful_shots) →
  (prob_second_third_successful = 3/10) :=
by sorry

end NUMINAMATH_CALUDE_basket_probability_l1707_170730


namespace NUMINAMATH_CALUDE_set_equality_l1707_170763

def positive_naturals : Set ℕ := {n : ℕ | n > 0}

def set_A : Set ℕ := {x ∈ positive_naturals | x - 3 < 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_A = set_B := by sorry

end NUMINAMATH_CALUDE_set_equality_l1707_170763


namespace NUMINAMATH_CALUDE_vasya_always_wins_l1707_170766

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents the state of the game -/
structure GameState where
  coloredPoints : Point → Option Color

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a winning condition is met -/
def isWinningState (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1 = some c ∧
    state.coloredPoints p2 = some c ∧
    state.coloredPoints p3 = some c

/-- The main theorem stating that Vasya always has a winning strategy -/
theorem vasya_always_wins :
  ∀ (initialState : GameState),
  ∃ (finalState : GameState),
    (∀ p, Option.isSome (finalState.coloredPoints p)) ∧
    isWinningState finalState :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l1707_170766


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1707_170770

/-- A geometric sequence {a_n} where a_3 = 2 and a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 6 = 16

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 2)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = general_term n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1707_170770


namespace NUMINAMATH_CALUDE_power_multiplication_l1707_170775

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1707_170775


namespace NUMINAMATH_CALUDE_quadrilateral_rhombus_l1707_170760

-- Define the points
variable (A B C D P Q R S : ℝ × ℝ)

-- Define the properties of the quadrilaterals
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def are_external_similar_isosceles_triangles (A B C D P Q R S : ℝ × ℝ) : Prop := sorry

def is_rectangle (P Q R S : ℝ × ℝ) : Prop := sorry

def sides_not_equal (P Q R S : ℝ × ℝ) : Prop := sorry

def is_rhombus (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_rhombus 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : are_external_similar_isosceles_triangles A B C D P Q R S)
  (h3 : is_rectangle P Q R S)
  (h4 : sides_not_equal P Q R S) :
  is_rhombus A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_rhombus_l1707_170760


namespace NUMINAMATH_CALUDE_set_inclusion_implies_parameter_range_l1707_170752

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0}

theorem set_inclusion_implies_parameter_range :
  ∀ a : ℝ, A ⊆ B a → a ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_parameter_range_l1707_170752


namespace NUMINAMATH_CALUDE_sum_of_extrema_l1707_170709

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 7) : 
  ∃ (n N : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 7) → n ≤ x ∧ x ≤ N) ∧ n + N = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extrema_l1707_170709


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1707_170735

theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 432 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 96 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1707_170735


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l1707_170708

theorem garage_sale_pricing (total_items : ℕ) (radio_highest_rank : ℕ) (h1 : total_items = 34) (h2 : radio_highest_rank = 14) :
  total_items - radio_highest_rank + 1 = 22 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l1707_170708


namespace NUMINAMATH_CALUDE_system_solution_l1707_170728

theorem system_solution : 
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -2) ∧ 
    (5 * x + 2 * y = 8) ∧ 
    (x = 20 / 23) ∧ 
    (y = 42 / 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1707_170728


namespace NUMINAMATH_CALUDE_seedling_survival_probability_l1707_170721

/-- Represents the data for a single sample of transplanted ginkgo seedlings -/
structure SeedlingData where
  transplanted : ℕ
  survived : ℕ
  survival_rate : ℚ
  transplanted_positive : transplanted > 0
  survived_le_transplanted : survived ≤ transplanted
  rate_calculation : survival_rate = survived / transplanted

/-- The data set of ginkgo seedling transplantation experiments -/
def seedling_samples : List SeedlingData := [
  ⟨100, 84, 84/100, by norm_num, by norm_num, by norm_num⟩,
  ⟨300, 279, 279/300, by norm_num, by norm_num, by norm_num⟩,
  ⟨600, 505, 505/600, by norm_num, by norm_num, by norm_num⟩,
  ⟨1000, 847, 847/1000, by norm_num, by norm_num, by norm_num⟩,
  ⟨7000, 6337, 6337/7000, by norm_num, by norm_num, by norm_num⟩,
  ⟨15000, 13581, 13581/15000, by norm_num, by norm_num, by norm_num⟩
]

/-- The estimated probability of ginkgo seedling survival -/
def estimated_probability : ℚ := 9/10

/-- Theorem stating that the estimated probability approaches 0.9 as sample size increases -/
theorem seedling_survival_probability :
  ∀ ε > 0, ∃ N, ∀ sample ∈ seedling_samples,
    sample.transplanted ≥ N →
    |sample.survival_rate - estimated_probability| < ε :=
sorry

end NUMINAMATH_CALUDE_seedling_survival_probability_l1707_170721


namespace NUMINAMATH_CALUDE_janet_practice_days_l1707_170716

def total_miles : ℕ := 72
def miles_per_day : ℕ := 8

theorem janet_practice_days : 
  total_miles / miles_per_day = 9 := by sorry

end NUMINAMATH_CALUDE_janet_practice_days_l1707_170716


namespace NUMINAMATH_CALUDE_correct_philosophies_l1707_170778

-- Define the philosophies
inductive Philosophy
  | GraspMeasure
  | ComprehensiveView
  | AnalyzeSpecifically
  | EmphasizeKeyPoints

-- Define the conditions
structure IodineScenario where
  iodineEssential : Bool
  oneSizeFitsAllRisky : Bool
  nonIodineDeficientArea : Bool
  increasedNonIodizedSalt : Bool
  allowAdjustment : Bool

-- Define the function to check if a philosophy is reflected
def reflectsPhilosophy (scenario : IodineScenario) (philosophy : Philosophy) : Prop :=
  match philosophy with
  | Philosophy.GraspMeasure => scenario.oneSizeFitsAllRisky
  | Philosophy.ComprehensiveView => scenario.iodineEssential ∧ scenario.oneSizeFitsAllRisky
  | Philosophy.AnalyzeSpecifically => scenario.nonIodineDeficientArea ∧ scenario.increasedNonIodizedSalt ∧ scenario.allowAdjustment
  | Philosophy.EmphasizeKeyPoints => False

-- Theorem to prove
theorem correct_philosophies (scenario : IodineScenario) 
  (h1 : scenario.iodineEssential = true)
  (h2 : scenario.oneSizeFitsAllRisky = true)
  (h3 : scenario.nonIodineDeficientArea = true)
  (h4 : scenario.increasedNonIodizedSalt = true)
  (h5 : scenario.allowAdjustment = true) :
  reflectsPhilosophy scenario Philosophy.GraspMeasure ∧
  reflectsPhilosophy scenario Philosophy.ComprehensiveView ∧
  reflectsPhilosophy scenario Philosophy.AnalyzeSpecifically ∧
  ¬reflectsPhilosophy scenario Philosophy.EmphasizeKeyPoints :=
sorry

end NUMINAMATH_CALUDE_correct_philosophies_l1707_170778


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l1707_170713

theorem equidistant_point_on_x_axis : ∃ x : ℝ, 
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (0, 5)
  let P : ℝ × ℝ := (x, 0)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l1707_170713


namespace NUMINAMATH_CALUDE_exactly_two_talents_l1707_170722

theorem exactly_two_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talent : ℕ) :
  total_students = 120 →
  cannot_sing = 50 →
  cannot_dance = 75 →
  cannot_act = 45 →
  no_talent = 15 →
  (∃ (two_talents : ℕ), two_talents = 70 ∧ 
    two_talents = total_students - 
      (cannot_sing + cannot_dance + cannot_act - 2 * no_talent)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_talents_l1707_170722


namespace NUMINAMATH_CALUDE_max_sum_is_3972_l1707_170711

/-- A function that generates all possible permutations of 9 digits -/
def generatePermutations : List (List Nat) := sorry

/-- A function that splits a list of 9 digits into three numbers -/
def splitIntoThreeNumbers (perm : List Nat) : (Nat × Nat × Nat) := sorry

/-- A function that calculates the sum of three numbers -/
def sumThreeNumbers (nums : Nat × Nat × Nat) : Nat := sorry

/-- The maximum sum achievable using digits 1 to 9 -/
def maxSum : Nat := 3972

theorem max_sum_is_3972 :
  ∀ perm ∈ generatePermutations,
    let (n1, n2, n3) := splitIntoThreeNumbers perm
    sumThreeNumbers (n1, n2, n3) ≤ maxSum :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_3972_l1707_170711


namespace NUMINAMATH_CALUDE_A_equals_B_l1707_170714

/-- The number of digits written when listing integers from 1 to 10^(n-1) -/
def A (n : ℕ) : ℕ := sorry

/-- The number of zeros written when listing integers from 1 to 10^n -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A(n) equals B(n) for all positive integers n -/
theorem A_equals_B (n : ℕ) (h : n > 0) : A n = B n := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l1707_170714


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1707_170743

theorem smallest_n_congruence (k : ℕ) (h : k > 0) :
  (7 ^ k) % 3 = (k ^ 7) % 3 → k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1707_170743


namespace NUMINAMATH_CALUDE_sum_of_solutions_abs_eq_l1707_170706

theorem sum_of_solutions_abs_eq (x : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |3 * x₁ - 12| = 6 ∧ |3 * x₂ - 12| = 6 ∧ x₁ + x₂ = 8) ∧ (∀ x : ℝ, |3 * x - 12| = 6 → x = 2 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_abs_eq_l1707_170706


namespace NUMINAMATH_CALUDE_committee_formation_count_l1707_170781

/-- The number of ways to choose a committee from a basketball team -/
def choose_committee (total_players : ℕ) (committee_size : ℕ) (total_guards : ℕ) : ℕ :=
  total_guards * (Nat.choose (total_players - total_guards) (committee_size - 1))

/-- Theorem: The number of ways to form the committee is 112 -/
theorem committee_formation_count :
  choose_committee 12 3 4 = 112 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1707_170781


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l1707_170700

/-- For positive real numbers a and b where a > b, the sum of the infinite series
    1/(ba) + 1/(a(2a + b)) + 1/((2a + b)(3a + 2b)) + 1/((3a + 2b)(4a + 3b)) + ...
    is equal to 1/((a + b)b) -/
theorem infinite_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n : ℕ => 1 / ((n * a + (n - 1) * b) * ((n + 1) * a + n * b))
  tsum series = 1 / ((a + b) * b) := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l1707_170700


namespace NUMINAMATH_CALUDE_infinitely_many_twin_pretty_numbers_l1707_170767

-- Define what it means for a number to be "pretty"
def isPrettyNumber (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

-- Define a pair of twin pretty numbers
def isTwinPrettyPair (n m : ℕ) : Prop :=
  isPrettyNumber n ∧ isPrettyNumber m ∧ m = n + 1

-- Theorem statement
theorem infinitely_many_twin_pretty_numbers :
  ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ isTwinPrettyPair n m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_twin_pretty_numbers_l1707_170767


namespace NUMINAMATH_CALUDE_pink_to_orange_ratio_l1707_170748

theorem pink_to_orange_ratio :
  -- Define the total number of balls
  let total_balls : ℕ := 50
  -- Define the number of red balls
  let red_balls : ℕ := 20
  -- Define the number of blue balls
  let blue_balls : ℕ := 10
  -- Define the number of orange balls
  let orange_balls : ℕ := 5
  -- Define the number of pink balls
  let pink_balls : ℕ := 15
  -- Ensure that the sum of all balls equals the total
  red_balls + blue_balls + orange_balls + pink_balls = total_balls →
  -- Prove that the ratio of pink to orange balls is 3:1
  (pink_balls : ℚ) / (orange_balls : ℚ) = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_pink_to_orange_ratio_l1707_170748


namespace NUMINAMATH_CALUDE_vertical_pairwise_sets_l1707_170793

-- Definition of a vertical pairwise set
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1 ∧ p.1 ≠ 0}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 ∧ p.1 > 0}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem stating which sets are vertical pairwise sets
theorem vertical_pairwise_sets :
  ¬(is_vertical_pairwise_set M₁) ∧
  ¬(is_vertical_pairwise_set M₂) ∧
  (is_vertical_pairwise_set M₃) ∧
  (is_vertical_pairwise_set M₄) := by
  sorry

end NUMINAMATH_CALUDE_vertical_pairwise_sets_l1707_170793


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l1707_170769

theorem rectangular_parallelepiped_volume
  (l α β : ℝ)
  (h_l : l > 0)
  (h_α : 0 < α ∧ α < π / 2)
  (h_β : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ,
    V = l^3 * Real.sin α * Real.sin β * Real.sqrt (Real.cos (α + β) * Real.cos (α - β)) ∧
    V > 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l1707_170769


namespace NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l1707_170741

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution_is_four_minus_sqrt_two :
  ∃ (x : ℝ), (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) ∧
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l1707_170741


namespace NUMINAMATH_CALUDE_grandchildren_probability_l1707_170796

def num_children : ℕ := 12

theorem grandchildren_probability :
  let total_outcomes := 2^num_children
  let equal_boys_girls := Nat.choose num_children (num_children / 2)
  let all_same_gender := 2
  (total_outcomes - (equal_boys_girls + all_same_gender)) / total_outcomes = 3170 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_grandchildren_probability_l1707_170796


namespace NUMINAMATH_CALUDE_ratio_product_l1707_170702

theorem ratio_product (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_product_l1707_170702


namespace NUMINAMATH_CALUDE_sarah_candy_count_l1707_170736

/-- The number of candy pieces Sarah received for Halloween -/
def total_candy : ℕ := sorry

/-- The number of candy pieces Sarah ate -/
def eaten_candy : ℕ := 36

/-- The number of piles Sarah made with the remaining candy -/
def number_of_piles : ℕ := 8

/-- The number of candy pieces in each pile -/
def pieces_per_pile : ℕ := 9

/-- Theorem stating that the total number of candy pieces Sarah received is 108 -/
theorem sarah_candy_count : total_candy = 108 := by
  sorry

end NUMINAMATH_CALUDE_sarah_candy_count_l1707_170736


namespace NUMINAMATH_CALUDE_fishing_problem_l1707_170704

theorem fishing_problem (total fish_jason fish_ryan fish_jeffery : ℕ) : 
  total = 100 ∧ 
  fish_ryan = 3 * fish_jason ∧ 
  fish_jeffery = 2 * fish_ryan ∧ 
  total = fish_jason + fish_ryan + fish_jeffery →
  fish_jeffery = 60 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l1707_170704


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l1707_170771

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x / x^(1/2))^(1/4) = x^(1/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l1707_170771


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1707_170738

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := m + 1 + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1707_170738


namespace NUMINAMATH_CALUDE_square_of_negative_square_l1707_170750

theorem square_of_negative_square (a : ℝ) : (-a^2)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l1707_170750


namespace NUMINAMATH_CALUDE_first_month_sale_l1707_170794

def average_sale : ℝ := 5400
def num_months : ℕ := 6
def sale_month2 : ℝ := 5366
def sale_month3 : ℝ := 5808
def sale_month4 : ℝ := 5399
def sale_month5 : ℝ := 6124
def sale_month6 : ℝ := 4579

theorem first_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 5124 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l1707_170794


namespace NUMINAMATH_CALUDE_sqrt_sum_expression_l1707_170783

theorem sqrt_sum_expression (a : ℝ) (h : a ≥ 1) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) =
    if 1 ≤ a ∧ a ≤ 2 then 2 else 2 * Real.sqrt (a - 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_expression_l1707_170783


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l1707_170745

theorem exactly_one_positive_integer_satisfies_condition :
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l1707_170745


namespace NUMINAMATH_CALUDE_inequality_range_l1707_170718

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (Real.sin x)^2 + a * Real.cos x - a^2 ≤ 1 + Real.cos x) ↔ 
  (a ≤ -1 ∨ a ≥ 1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1707_170718


namespace NUMINAMATH_CALUDE_tan_beta_rational_iff_p_sq_plus_q_sq_perfect_square_l1707_170797

theorem tan_beta_rational_iff_p_sq_plus_q_sq_perfect_square
  (p q : ℤ) (α β : ℝ) (h_q_nonzero : q ≠ 0) (h_tan_alpha : Real.tan α = p / q)
  (h_tan_2beta : Real.tan (2 * β) = Real.tan (3 * α)) :
  (∃ (r : ℚ), Real.tan β = r) ↔ ∃ (n : ℤ), p^2 + q^2 = n^2 := by sorry

end NUMINAMATH_CALUDE_tan_beta_rational_iff_p_sq_plus_q_sq_perfect_square_l1707_170797


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l1707_170786

theorem rectangular_plot_width
  (length : ℝ)
  (num_poles : ℕ)
  (pole_spacing : ℝ)
  (h1 : length = 90)
  (h2 : num_poles = 14)
  (h3 : pole_spacing = 20)
  : ∃ width : ℝ, width = 40 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_spacing :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l1707_170786


namespace NUMINAMATH_CALUDE_x_plus_y_values_l1707_170742

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l1707_170742


namespace NUMINAMATH_CALUDE_apple_pies_theorem_l1707_170755

def total_apples : ℕ := 128
def unripe_apples : ℕ := 23
def apples_per_pie : ℕ := 7

theorem apple_pies_theorem : 
  (total_apples - unripe_apples) / apples_per_pie = 15 :=
by sorry

end NUMINAMATH_CALUDE_apple_pies_theorem_l1707_170755


namespace NUMINAMATH_CALUDE_remaining_nails_l1707_170795

def initial_nails : ℕ := 400

def kitchen_repair (n : ℕ) : ℕ := n - (n * 35 / 100)

def fence_repair (n : ℕ) : ℕ := n - (n * 75 / 100)

def table_repair (n : ℕ) : ℕ := n - (n * 55 / 100)

def floorboard_repair (n : ℕ) : ℕ := n - (n * 30 / 100)

theorem remaining_nails :
  floorboard_repair (table_repair (fence_repair (kitchen_repair initial_nails))) = 21 :=
by sorry

end NUMINAMATH_CALUDE_remaining_nails_l1707_170795


namespace NUMINAMATH_CALUDE_intersection_theorem_l1707_170782

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_theorem : M ∩ N = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1707_170782


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1707_170719

/-- The number of arrangements of six people in a row,
    where A and B must be adjacent with B to the left of A -/
def arrangements_count : ℕ := 120

/-- Theorem stating that the number of arrangements is 120 -/
theorem six_people_arrangement :
  arrangements_count = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1707_170719


namespace NUMINAMATH_CALUDE_inequality_proof_l1707_170746

theorem inequality_proof : (-abs (abs (-20 : ℝ))) / 2 > -4.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1707_170746


namespace NUMINAMATH_CALUDE_december_ear_muff_sales_l1707_170789

/-- The number of type B ear muffs sold in December -/
def type_b_count : ℕ := 3258

/-- The price of each type B ear muff -/
def type_b_price : ℚ := 69/10

/-- The number of type C ear muffs sold in December -/
def type_c_count : ℕ := 3186

/-- The price of each type C ear muff -/
def type_c_price : ℚ := 74/10

/-- The total amount spent on ear muffs in December -/
def total_spent : ℚ := type_b_count * type_b_price + type_c_count * type_c_price

theorem december_ear_muff_sales :
  total_spent = 460566/10 := by sorry

end NUMINAMATH_CALUDE_december_ear_muff_sales_l1707_170789


namespace NUMINAMATH_CALUDE_partition_remainder_l1707_170739

theorem partition_remainder (S : Finset ℕ) : 
  S.card = 15 → 
  (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000 = 406 := by
  sorry

#eval (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000

end NUMINAMATH_CALUDE_partition_remainder_l1707_170739


namespace NUMINAMATH_CALUDE_club_members_after_four_years_l1707_170726

/-- Calculates the number of people in the club after a given number of years -/
def club_members (initial_regular_members : ℕ) (years : ℕ) : ℕ :=
  initial_regular_members * (2 ^ years)

/-- Theorem stating the number of people in the club after 4 years -/
theorem club_members_after_four_years :
  let initial_total := 9
  let initial_board_members := 3
  let initial_regular_members := initial_total - initial_board_members
  club_members initial_regular_members 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l1707_170726


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l1707_170703

/-- The number of hamburgers served by the restaurant -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := 6

/-- The total number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := hamburgers_served + hamburgers_leftover

/-- Theorem stating that the total number of hamburgers is 9 -/
theorem restaurant_hamburgers : total_hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l1707_170703


namespace NUMINAMATH_CALUDE_helen_cookies_l1707_170720

/-- The number of chocolate chip cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 527

/-- The number of chocolate chip cookies Helen baked this morning -/
def cookies_today : ℕ := 554

/-- The total number of chocolate chip cookies Helen baked -/
def total_cookies : ℕ := cookies_yesterday + cookies_today

theorem helen_cookies : total_cookies = 1081 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l1707_170720


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1707_170729

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1707_170729


namespace NUMINAMATH_CALUDE_tile_perimeter_increase_l1707_170756

/-- Represents a configuration of square tiles --/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration --/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { tiles := initial.tiles + added, perimeter := initial.perimeter + 3 }

/-- The theorem to be proved --/
theorem tile_perimeter_increase :
  ∃ (initial final : TileConfiguration),
    initial.tiles = 10 ∧
    initial.perimeter = 16 ∧
    final = add_tiles initial 3 ∧
    final.perimeter = 19 := by
  sorry

end NUMINAMATH_CALUDE_tile_perimeter_increase_l1707_170756


namespace NUMINAMATH_CALUDE_smallest_number_l1707_170768

/-- Converts a number from base b to decimal (base 10) --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The given numbers in their respective bases --/
def number_A : List Nat := [2, 0, 0, 1]  -- 1002 in base 3
def number_B : List Nat := [0, 1, 2]     -- 210 in base 6
def number_C : List Nat := [0, 0, 0, 1]  -- 1000 in base 4
def number_D : List Nat := [1, 1, 1, 1, 1, 1]  -- 111111 in base 2

/-- The bases of the given numbers --/
def base_A : Nat := 3
def base_B : Nat := 6
def base_C : Nat := 4
def base_D : Nat := 2

theorem smallest_number :
  (to_decimal number_A base_A < to_decimal number_B base_B) ∧
  (to_decimal number_A base_A < to_decimal number_C base_C) ∧
  (to_decimal number_A base_A < to_decimal number_D base_D) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1707_170768


namespace NUMINAMATH_CALUDE_cats_not_eating_l1707_170759

theorem cats_not_eating (total : ℕ) (likes_apples : ℕ) (likes_fish : ℕ) (likes_both : ℕ) 
  (h1 : total = 75)
  (h2 : likes_apples = 15)
  (h3 : likes_fish = 55)
  (h4 : likes_both = 8) :
  total - (likes_apples + likes_fish - likes_both) = 13 :=
by sorry

end NUMINAMATH_CALUDE_cats_not_eating_l1707_170759


namespace NUMINAMATH_CALUDE_lotus_pollen_diameter_scientific_notation_l1707_170731

theorem lotus_pollen_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0025 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.5 ∧ n = -3 := by
  sorry

end NUMINAMATH_CALUDE_lotus_pollen_diameter_scientific_notation_l1707_170731


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1707_170705

-- Define the condition for a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ Set.Ioo (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1707_170705


namespace NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l1707_170773

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem units_digit_of_first_four_composites_product :
  (product_of_list first_four_composites) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_first_four_composites_product_l1707_170773


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1707_170749

/-- 
Given two vectors a and b in R^2, where a = (1, m) and b = (-1, 2m+1),
prove that if a and b are parallel, then m = -1/3.
-/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 2*m+1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1707_170749


namespace NUMINAMATH_CALUDE_max_integer_value_of_fraction_l1707_170717

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4*x^2 + 12*x + 23) / (4*x^2 + 12*x + 9) ≤ 8 ∧ 
  ∃ y : ℝ, (4*y^2 + 12*y + 23) / (4*y^2 + 12*y + 9) > 7 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_value_of_fraction_l1707_170717


namespace NUMINAMATH_CALUDE_shortest_path_on_specific_frustum_l1707_170765

/-- Represents a truncated circular right cone (frustum) -/
structure Frustum where
  lower_circumference : ℝ
  upper_circumference : ℝ
  inclination_angle : ℝ

/-- Calculates the shortest path on the surface of a frustum -/
def shortest_path (f : Frustum) (upper_travel : ℝ) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem shortest_path_on_specific_frustum :
  let f : Frustum := {
    lower_circumference := 10,
    upper_circumference := 9,
    inclination_angle := Real.pi / 3  -- 60 degrees in radians
  }
  shortest_path f 3 = 5 * Real.sqrt 3 / Real.pi :=
sorry

end NUMINAMATH_CALUDE_shortest_path_on_specific_frustum_l1707_170765


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1707_170787

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1, where a > 0 and b > 0
    - O is the origin
    - F₁ and F₂ are the left and right foci
    - P is a point on the left branch
    - M is the midpoint of F₂P
    - |OM| = c/5, where c is the focal distance
    Then the eccentricity e of the hyperbola satisfies 1 < e ≤ 5/3 -/
theorem hyperbola_eccentricity_range 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (O : ℝ × ℝ) (F₁ F₂ P M : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1})
  (O_origin : O = (0, 0))
  (F₁_left : F₁.1 < 0)
  (F₂_right : F₂.1 > 0)
  (P_left_branch : P.1 < 0)
  (M_midpoint : M = ((F₂.1 + P.1)/2, (F₂.2 + P.2)/2))
  (OM_length : Real.sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2) = c/5)
  (e_def : e = c/a) :
  1 < e ∧ e ≤ 5/3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1707_170787


namespace NUMINAMATH_CALUDE_binomial_coefficient_product_l1707_170792

theorem binomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) * (a₁ + a₃ + a₅) = -256 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_product_l1707_170792


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt33_l1707_170780

theorem consecutive_integers_around_sqrt33 (a b : ℤ) :
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 33) →  -- a < √33
  (Real.sqrt 33 < b) →  -- √33 < b
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt33_l1707_170780


namespace NUMINAMATH_CALUDE_line_equation_l1707_170776

-- Define the line l
def Line := ℝ → ℝ → Prop

-- Define the point type
def Point := ℝ × ℝ

-- Define the distance function between a point and a line
def distance (p : Point) (l : Line) : ℝ := sorry

-- Define the condition that line l passes through point P(1,2)
def passes_through (l : Line) : Prop :=
  l 1 2

-- Define the condition that line l is equidistant from A(2,3) and B(0,-5)
def equidistant (l : Line) : Prop :=
  distance (2, 3) l = distance (0, -5) l

-- State the theorem
theorem line_equation (l : Line) 
  (h1 : passes_through l) 
  (h2 : equidistant l) : 
  (∀ x y, l x y ↔ 4*x - y - 2 = 0) ∨ 
  (∀ x y, l x y ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1707_170776


namespace NUMINAMATH_CALUDE_marble_selection_problem_l1707_170762

theorem marble_selection_problem (n : ℕ) (k : ℕ) (total : ℕ) (red : ℕ) :
  n = 10 →
  k = 4 →
  total = Nat.choose n k →
  red = Nat.choose (n - 1) k →
  total - red = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l1707_170762


namespace NUMINAMATH_CALUDE_first_number_proof_l1707_170724

theorem first_number_proof (x : ℝ) : 
  (((10 + 60 + 35) / 3) + 5 = (x + 40 + 60) / 3) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1707_170724


namespace NUMINAMATH_CALUDE_remaining_length_is_twelve_l1707_170790

/-- Represents a rectangle with specific dimensions -/
structure Rectangle :=
  (left : ℝ)
  (top1 : ℝ)
  (top2 : ℝ)
  (top3 : ℝ)

/-- Calculates the total length of remaining segments after removing sides -/
def remaining_length (r : Rectangle) : ℝ :=
  r.left + r.top1 + r.top2 + r.top3

/-- Theorem stating that for a rectangle with given dimensions, 
    the remaining length after removing sides is 12 units -/
theorem remaining_length_is_twelve (r : Rectangle) 
  (h1 : r.left = 8)
  (h2 : r.top1 = 2)
  (h3 : r.top2 = 1)
  (h4 : r.top3 = 1) :
  remaining_length r = 12 := by
  sorry

#check remaining_length_is_twelve

end NUMINAMATH_CALUDE_remaining_length_is_twelve_l1707_170790


namespace NUMINAMATH_CALUDE_total_toys_l1707_170727

/-- Given that Annie has three times more toys than Mike, Annie has two less toys than Tom,
    and Mike has 6 toys, prove that the total number of toys Annie, Mike, and Tom have is 56. -/
theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ)
  (h1 : annie_toys = 3 * mike_toys)
  (h2 : tom_toys = annie_toys + 2)
  (h3 : mike_toys = 6) :
  annie_toys + mike_toys + tom_toys = 56 :=
by sorry

end NUMINAMATH_CALUDE_total_toys_l1707_170727
