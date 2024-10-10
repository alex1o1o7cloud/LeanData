import Mathlib

namespace scale_model_height_l1547_154777

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 984

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height / scale_ratio

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 39 := by
  sorry

end scale_model_height_l1547_154777


namespace simplify_expression_l1547_154793

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 4*b^2 = 9*b^3 + 2*b^2 := by
  sorry

end simplify_expression_l1547_154793


namespace system_solution_l1547_154750

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 0 ∧ 3*x - 4*y = 5) ↔ (x = 1 ∧ y = -1/2) := by
sorry

end system_solution_l1547_154750


namespace rhombus_perimeter_l1547_154738

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by
  sorry

#check rhombus_perimeter

end rhombus_perimeter_l1547_154738


namespace tangent_line_to_circle_l1547_154703

/-- The value of r for which the line x + y = 4 is tangent to the circle (x-2)^2 + (y+1)^2 = r -/
theorem tangent_line_to_circle (x y : ℝ) :
  (x + y = 4) →
  ((x - 2)^2 + (y + 1)^2 = (9:ℝ)/2) →
  ∃ (r : ℝ), r = (9:ℝ)/2 ∧ 
    (∀ (x' y' : ℝ), (x' + y' = 4) → ((x' - 2)^2 + (y' + 1)^2 ≤ r)) ∧
    (∃ (x₀ y₀ : ℝ), (x₀ + y₀ = 4) ∧ ((x₀ - 2)^2 + (y₀ + 1)^2 = r)) :=
by sorry

end tangent_line_to_circle_l1547_154703


namespace peach_count_theorem_l1547_154729

def audrey_initial : ℕ := 26
def paul_initial : ℕ := 48
def maya_initial : ℕ := 57

def audrey_multiplier : ℕ := 3
def paul_multiplier : ℕ := 2
def maya_additional : ℕ := 20

def total_peaches : ℕ := 
  (audrey_initial + audrey_initial * audrey_multiplier) +
  (paul_initial + paul_initial * paul_multiplier) +
  (maya_initial + maya_additional)

theorem peach_count_theorem : total_peaches = 325 := by
  sorry

end peach_count_theorem_l1547_154729


namespace integral_tangent_sine_l1547_154781

open Real MeasureTheory

theorem integral_tangent_sine (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π/4) (arctan 3), 1 / ((3 * tan x + 5) * sin (2 * x))) = (1/10) * log (12/7) := by
  sorry

end integral_tangent_sine_l1547_154781


namespace exponent_division_23_l1547_154756

theorem exponent_division_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end exponent_division_23_l1547_154756


namespace delta_sum_bound_l1547_154723

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of δ(n)/n from 1 to x -/
def delta_sum (x : ℕ+) : ℚ :=
  sorry

/-- Theorem: For any positive integer x, |∑(n=1 to x) [δ(n)/n] - (2/3)x| < 1 -/
theorem delta_sum_bound (x : ℕ+) :
  |delta_sum x - (2/3 : ℚ) * x.val| < 1 :=
sorry

end delta_sum_bound_l1547_154723


namespace solution_set_f_greater_than_two_range_of_T_l1547_154796

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ 1 < x} := by sorry

-- Theorem for the range of T
theorem range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 5/2*T - 1) →
  (T ≤ -3 ∨ T ≥ 1/2) := by sorry

end solution_set_f_greater_than_two_range_of_T_l1547_154796


namespace watermelon_theorem_l1547_154764

def watermelon_problem (initial_watermelons : ℕ) (consumption_pattern : List ℕ) : Prop :=
  let total_consumption := consumption_pattern.sum
  let complete_cycles := initial_watermelons / total_consumption
  let remaining_watermelons := initial_watermelons % total_consumption
  complete_cycles * consumption_pattern.length = 3 ∧
  remaining_watermelons < consumption_pattern.head!

theorem watermelon_theorem :
  watermelon_problem 30 [7, 8, 9] :=
by sorry

end watermelon_theorem_l1547_154764


namespace square_side_length_l1547_154725

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = 9) (h₁ : A = s^2) :
  s = Real.sqrt 9 := by
  sorry

end square_side_length_l1547_154725


namespace loss_percentage_calculation_l1547_154797

/-- Calculate the percentage of loss given the cost price and selling price -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  ((costPrice - sellingPrice) / costPrice) * 100

theorem loss_percentage_calculation (costPrice sellingPrice : ℚ) 
  (h1 : costPrice = 1750)
  (h2 : sellingPrice = 1610) :
  percentageLoss costPrice sellingPrice = 8 := by
  sorry

#eval percentageLoss 1750 1610

end loss_percentage_calculation_l1547_154797


namespace stating_two_cookies_per_guest_l1547_154740

/-- 
Given a total number of cookies and guests, calculates the number of cookies per guest,
assuming each guest receives the same number of cookies.
-/
def cookiesPerGuest (totalCookies guests : ℕ) : ℚ :=
  totalCookies / guests

/-- 
Theorem stating that when there are 10 cookies and 5 guests,
each guest receives 2 cookies.
-/
theorem two_cookies_per_guest :
  cookiesPerGuest 10 5 = 2 := by
  sorry

end stating_two_cookies_per_guest_l1547_154740


namespace vote_intersection_l1547_154775

theorem vote_intersection (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 172 →
  Finset.card B = 143 →
  Finset.card (U \ (A ∪ B)) = 37 →
  Finset.card (A ∩ B) = 102 := by
sorry

end vote_intersection_l1547_154775


namespace monochromatic_triangle_exists_l1547_154708

/-- A coloring of the edges of a complete graph on 10 vertices using two colors -/
def TwoColoring : Type := Fin 10 → Fin 10 → Bool

/-- A triangle in a graph is represented by three distinct vertices -/
structure Triangle (n : Nat) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- A triangle is monochromatic if all its edges have the same color -/
def isMonochromatic (c : TwoColoring) (t : Triangle 10) : Prop :=
  c t.v1 t.v2 = c t.v2 t.v3 ∧ c t.v2 t.v3 = c t.v3 t.v1

/-- The main theorem: every two-coloring of K_10 contains a monochromatic triangle -/
theorem monochromatic_triangle_exists (c : TwoColoring) : 
  ∃ t : Triangle 10, isMonochromatic c t := by
  sorry


end monochromatic_triangle_exists_l1547_154708


namespace sum_of_products_l1547_154732

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 25)
  (eq3 : z^2 + x*z + x^2 = 52) :
  x*y + y*z + x*z = 30 := by
sorry

end sum_of_products_l1547_154732


namespace solve_equation_l1547_154799

-- Define the functions
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def f : ℝ → ℝ := λ x => g (-x)

-- State the theorem
theorem solve_equation (m : ℝ) : f m = -1 → m = -1 / Real.exp 1 := by
  sorry

end solve_equation_l1547_154799


namespace exists_m_divisible_by_1997_l1547_154731

def f (x : ℤ) : ℤ := 3 * x + 2

def f_iter : ℕ → (ℤ → ℤ)
| 0 => id
| n + 1 => f ∘ f_iter n

theorem exists_m_divisible_by_1997 : 
  ∃ m : ℕ+, (1997 : ℤ) ∣ f_iter 99 m.val :=
sorry

end exists_m_divisible_by_1997_l1547_154731


namespace count_valid_triples_l1547_154789

def is_valid_triple (a b c : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 2 * (a * b + b * c + c * a)

theorem count_valid_triples :
  ∃! n : ℕ, ∃ S : Finset (ℕ × ℕ × ℕ),
    S.card = n ∧
    (∀ t ∈ S, is_valid_triple t.1 t.2.1 t.2.2) ∧
    (∀ a b c : ℕ, is_valid_triple a b c → (a, b, c) ∈ S) ∧
    n = 5 :=
sorry

end count_valid_triples_l1547_154789


namespace range_of_a_l1547_154752

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (a ∈ Set.Icc (-1) 1 ∨ a > 3) :=
by sorry

end range_of_a_l1547_154752


namespace slower_painter_start_time_painting_scenario_conditions_l1547_154794

/-- Proves that the slower painter starts at 6.6 hours past noon given the painting scenario conditions -/
theorem slower_painter_start_time :
  ∀ (start_time : ℝ),
    (start_time + 6 = start_time + 7) →  -- Both painters finish at the same time
    (start_time + 7 = 12.6) →            -- They finish at 0.6 past midnight
    start_time = 6.6 := by
  sorry

/-- Defines the time the slower painter starts in hours past noon -/
def slower_painter_start : ℝ := 6.6

/-- Defines the time the faster painter starts in hours past noon -/
def faster_painter_start : ℝ := slower_painter_start + 3

/-- Defines the time both painters finish in hours past noon -/
def finish_time : ℝ := 12.6

/-- Proves that the painting scenario conditions are satisfied -/
theorem painting_scenario_conditions :
  slower_painter_start + 6 = finish_time ∧
  faster_painter_start + 4 = finish_time ∧
  faster_painter_start = slower_painter_start + 3 := by
  sorry

end slower_painter_start_time_painting_scenario_conditions_l1547_154794


namespace square_difference_l1547_154717

theorem square_difference (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 := by
  sorry

end square_difference_l1547_154717


namespace dinner_pizzas_count_l1547_154722

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := 15

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := total_pizzas - lunch_pizzas

theorem dinner_pizzas_count : dinner_pizzas = 6 := by
  sorry

end dinner_pizzas_count_l1547_154722


namespace fractional_equation_solution_l1547_154790

theorem fractional_equation_solution :
  ∃ (x : ℚ), (x ≠ 0 ∧ x ≠ 3) → (3 / (x^2 - 3*x) + (x - 1) / (x - 3) = 1) ∧ x = -3/2 := by
  sorry

end fractional_equation_solution_l1547_154790


namespace grid_division_exists_l1547_154765

/-- Represents a figure cut from the grid -/
structure Figure where
  area : ℕ
  externalPerimeter : ℕ
  internalPerimeter : ℕ

/-- Represents the division of the 9x9 grid -/
structure GridDivision where
  a : Figure
  b : Figure
  c : Figure

/-- The proposition to be proved -/
theorem grid_division_exists : ∃ (d : GridDivision),
  -- The grid is 9x9
  (9 * 9 = d.a.area + d.b.area + d.c.area) ∧
  -- All figures have equal area
  (d.a.area = d.b.area) ∧ (d.b.area = d.c.area) ∧
  -- The perimeter of c equals the sum of perimeters of a and b
  (d.c.externalPerimeter + d.c.internalPerimeter = 
   d.a.externalPerimeter + d.a.internalPerimeter + 
   d.b.externalPerimeter + d.b.internalPerimeter) ∧
  -- The sum of external perimeters is the perimeter of the 9x9 grid
  (d.a.externalPerimeter + d.b.externalPerimeter + d.c.externalPerimeter = 4 * 9) ∧
  -- The sum of a and b's internal perimeters equals c's internal perimeter
  (d.a.internalPerimeter + d.b.internalPerimeter = d.c.internalPerimeter) :=
sorry

end grid_division_exists_l1547_154765


namespace bugs_eating_flowers_l1547_154760

/-- Given 2.5 bugs eating 4.5 flowers in total, the number of flowers consumed per bug is 1.8 -/
theorem bugs_eating_flowers (num_bugs : ℝ) (total_flowers : ℝ) 
    (h1 : num_bugs = 2.5) 
    (h2 : total_flowers = 4.5) : 
  total_flowers / num_bugs = 1.8 := by
  sorry

end bugs_eating_flowers_l1547_154760


namespace quadratic_real_roots_condition_l1547_154747

/-- 
For a quadratic equation (k-2)x^2 - 2kx + k = 6 to have real roots,
k must satisfy k ≥ 1.5 and k ≠ 2.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
sorry

end quadratic_real_roots_condition_l1547_154747


namespace vector_magnitude_l1547_154715

-- Define the vectors a and b
def a (t : ℝ) : Fin 2 → ℝ := ![t - 2, 3]
def b : Fin 2 → ℝ := ![3, -1]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

-- State the theorem
theorem vector_magnitude (t : ℝ) :
  (parallel (λ i => a t i + 2 * b i) b) →
  Real.sqrt ((a t 0) ^ 2 + (a t 1) ^ 2) = 3 * Real.sqrt 10 := by
  sorry

end vector_magnitude_l1547_154715


namespace calculate_expression_l1547_154785

theorem calculate_expression : 2 * 9 - Real.sqrt 36 + 1 = 13 := by
  sorry

end calculate_expression_l1547_154785


namespace visitors_in_scientific_notation_l1547_154770

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number of visitors -/
def visitors : ℕ := 876000

/-- The scientific notation representation of the number of visitors -/
def visitors_scientific : ScientificNotation :=
  { coefficient := 8.76
  , exponent := 5
  , h1 := by sorry }

theorem visitors_in_scientific_notation :
  (visitors : ℝ) = visitors_scientific.coefficient * (10 : ℝ) ^ visitors_scientific.exponent :=
by sorry

end visitors_in_scientific_notation_l1547_154770


namespace two_students_per_section_l1547_154774

/-- Represents a school bus with a given number of rows and total capacity. -/
structure SchoolBus where
  rows : ℕ
  capacity : ℕ

/-- Calculates the number of students allowed per section in a school bus. -/
def studentsPerSection (bus : SchoolBus) : ℚ :=
  bus.capacity / (2 * bus.rows)

/-- Theorem stating that for a bus with 13 rows and capacity of 52 students,
    the number of students per section is 2. -/
theorem two_students_per_section :
  let bus : SchoolBus := { rows := 13, capacity := 52 }
  studentsPerSection bus = 2 := by
  sorry


end two_students_per_section_l1547_154774


namespace union_equals_real_when_m_is_one_sufficient_necessary_condition_l1547_154778

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - m - 1) ≥ 0}

theorem union_equals_real_when_m_is_one :
  A ∪ B 1 = Set.univ := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≤ -2 ∨ m ≥ 3 := by sorry

end union_equals_real_when_m_is_one_sufficient_necessary_condition_l1547_154778


namespace circle_equation_l1547_154772

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the line x - y = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = p.2}

-- State the theorem
theorem circle_equation :
  -- C passes through the origin
  (0, 0) ∈ C ∧
  -- The center of C is on the positive x-axis
  (∃ a : ℝ, a > 0 ∧ (a, 0) ∈ C) ∧
  -- The chord intercepted by the line x-y=0 on C has a length of 2√2
  (∃ p q : ℝ × ℝ, p ∈ C ∧ q ∈ C ∧ p ∈ L ∧ q ∈ L ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 2) →
  -- Then the equation of C is (x-2)^2 + y^2 = 4
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4} :=
by
  sorry

end circle_equation_l1547_154772


namespace parallel_vectors_magnitude_l1547_154792

/-- Given two vectors a and b in R², if a is parallel to b, then |2a - b| = 4√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a.1 = 1 ∧ a.2 = 2 ∧ b.1 = -2 → a.1 * b.2 = a.2 * b.1 → 
  ‖(2 • a - b : ℝ × ℝ)‖ = 4 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l1547_154792


namespace topsoil_cost_l1547_154746

-- Define the cost per cubic foot of topsoil
def cost_per_cubic_foot : ℝ := 8

-- Define the conversion factor from cubic yards to cubic feet
def cubic_yards_to_cubic_feet : ℝ := 27

-- Define the volume in cubic yards
def volume_in_cubic_yards : ℝ := 7

-- Theorem statement
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1512 := by
  sorry

end topsoil_cost_l1547_154746


namespace sqrt_equation_solution_l1547_154735

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 * x - 2) + 12 / Real.sqrt (3 * x - 2) = 8 ↔ x = 2 ∨ x = 38 / 3 :=
by sorry

end sqrt_equation_solution_l1547_154735


namespace complex_equation_solution_l1547_154728

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + 2 * Complex.I → z = -2 - Complex.I := by
  sorry

end complex_equation_solution_l1547_154728


namespace equation_solution_l1547_154745

theorem equation_solution : 
  ∃ x : ℚ, (17 / 60 + 7 / x = 21 / x + 1 / 15) ∧ (x = 840 / 13) := by
  sorry

end equation_solution_l1547_154745


namespace point_in_fourth_quadrant_l1547_154761

/-- A point in a 2D Cartesian plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant in a Cartesian plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that the point (1, -1) lies in the fourth quadrant. -/
theorem point_in_fourth_quadrant :
  let A : Point := ⟨1, -1⟩
  FourthQuadrant A := by
  sorry

end point_in_fourth_quadrant_l1547_154761


namespace circle_area_with_diameter_10_l1547_154769

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10  -- diameter in meters
  let r : ℝ := d / 2  -- radius in meters
  let area : ℝ := π * r^2  -- area formula
  area = 25 * π :=
by
  sorry

end circle_area_with_diameter_10_l1547_154769


namespace seven_digit_increasing_numbers_l1547_154753

theorem seven_digit_increasing_numbers (n : ℕ) (h : n = 7) :
  (Nat.choose (9 + n - 1) n) % 1000 = 435 := by
  sorry

end seven_digit_increasing_numbers_l1547_154753


namespace polynomial_division_remainder_l1547_154787

theorem polynomial_division_remainder :
  ∃ (q r : Polynomial ℤ),
    (X^5 - X^4 + X^3 - X + 1 : Polynomial ℤ) = (X^3 - X + 1) * q + r ∧
    r = -X^2 + 4*X - 1 ∧
    r.degree < (X^3 - X + 1).degree :=
by sorry

end polynomial_division_remainder_l1547_154787


namespace opposite_numbers_absolute_value_l1547_154727

theorem opposite_numbers_absolute_value (a b : ℝ) : 
  a + b = 0 → |a - 2014 + b| = 2014 := by sorry

end opposite_numbers_absolute_value_l1547_154727


namespace words_with_e_count_l1547_154786

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def word_length : Nat := 4

def total_words : Nat := alphabet.card ^ word_length

def words_without_e : Nat := (alphabet.card - 1) ^ word_length

theorem words_with_e_count : 
  total_words - words_without_e = 369 := by sorry

end words_with_e_count_l1547_154786


namespace root_equation_implies_d_equals_eight_l1547_154749

theorem root_equation_implies_d_equals_eight 
  (a b c d : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) (hd : d > 1) 
  (h : ∀ (M : ℝ), M ≠ 1 → (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d))) = M^(17/24)) : 
  d = 8 := by
sorry

end root_equation_implies_d_equals_eight_l1547_154749


namespace bella_steps_l1547_154783

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℕ := 4

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 1056

theorem bella_steps :
  distance * (speed_ratio + 1) / speed_ratio / feet_per_step = steps := by
  sorry

end bella_steps_l1547_154783


namespace fault_line_current_movement_l1547_154768

/-- Represents the movement of a fault line over two years -/
structure FaultLineMovement where
  total : ℝ  -- Total movement over two years
  previous : ℝ  -- Movement in the previous year
  current : ℝ  -- Movement in the current year

/-- Theorem stating the movement of the fault line in the current year -/
theorem fault_line_current_movement (f : FaultLineMovement)
  (h1 : f.total = 6.5)
  (h2 : f.previous = 5.25)
  (h3 : f.total = f.previous + f.current) :
  f.current = 1.25 := by
  sorry

end fault_line_current_movement_l1547_154768


namespace expression_simplification_l1547_154714

theorem expression_simplification (x : ℝ) : 7*x + 9 - 3*x + 15 * 2 = 4*x + 39 := by
  sorry

end expression_simplification_l1547_154714


namespace tire_usage_l1547_154733

/-- Proves that each tire is used for 32,000 miles given the conditions of the problem -/
theorem tire_usage (total_miles : ℕ) (total_tires : ℕ) (tires_in_use : ℕ) 
  (h1 : total_miles = 40000)
  (h2 : total_tires = 5)
  (h3 : tires_in_use = 4)
  (h4 : tires_in_use < total_tires) :
  (total_miles * tires_in_use) / total_tires = 32000 := by
  sorry

end tire_usage_l1547_154733


namespace max_tied_teams_seven_team_tournament_l1547_154701

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : Nat)
  (no_draws : Bool)
  (round_robin : Bool)

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (t : Tournament) : Nat :=
  (t.num_teams * (t.num_teams - 1)) / 2

/-- Represents the maximum number of teams that can be tied for the most wins --/
def max_tied_teams (t : Tournament) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem max_tied_teams_seven_team_tournament :
  ∀ t : Tournament, t.num_teams = 7 → t.no_draws = true → t.round_robin = true →
  max_tied_teams t = 6 :=
sorry

end max_tied_teams_seven_team_tournament_l1547_154701


namespace combined_salaries_l1547_154724

/-- Given the salary of E and the average salary of five individuals including E,
    calculate the combined salaries of the other four individuals. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) :
  salary_E = 9000 →
  average_salary = 8800 →
  num_individuals = 5 →
  (num_individuals * average_salary) - salary_E = 35000 := by
  sorry

end combined_salaries_l1547_154724


namespace perimeter_plus_area_sum_l1547_154716

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : IntegerParallelogram :=
  { v1 := (2, 3)
    v2 := (5, 7)
    v3 := (11, 7)
    v4 := (8, 3) }

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  sorry

/-- Calculate the area of the parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) :
  p = specificParallelogram → perimeter p + area p = 46 :=
sorry

end perimeter_plus_area_sum_l1547_154716


namespace tangent_curves_alpha_l1547_154734

theorem tangent_curves_alpha (f g : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, g x = α * x^2) →
  (∃ x₀, f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀) →
  α = Real.exp 2 / 4 :=
sorry

end tangent_curves_alpha_l1547_154734


namespace exists_multi_illuminated_point_l1547_154706

/-- Represents a street light in City A -/
structure StreetLight where
  position : ℝ × ℝ
  batteryReplacementTime : ℝ

/-- The city configuration -/
structure CityA where
  streetLights : Set StreetLight
  cityRadius : ℝ
  newBatteryRadius : ℝ
  radiusDecreaseRate : ℝ
  batteryLifespan : ℝ
  dailyBatteryUsage : ℕ

/-- The illumination area of a street light at a given time -/
def illuminationArea (light : StreetLight) (time : ℝ) (city : CityA) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem: There exists a point illuminated by multiple street lights -/
theorem exists_multi_illuminated_point (city : CityA) 
  (h1 : city.cityRadius = 10000)
  (h2 : city.newBatteryRadius = 200)
  (h3 : city.radiusDecreaseRate = 10)
  (h4 : city.batteryLifespan = 20)
  (h5 : city.dailyBatteryUsage = 18000) :
  ∃ (point : ℝ × ℝ) (time : ℝ), 
    ∃ (light1 light2 : StreetLight), light1 ≠ light2 ∧ 
    point ∈ illuminationArea light1 time city ∧ 
    point ∈ illuminationArea light2 time city :=
  sorry

end exists_multi_illuminated_point_l1547_154706


namespace crew_member_count_l1547_154763

/-- The number of crew members working on all islands in a country -/
def total_crew_members (num_islands : ℕ) (ships_per_island : ℕ) (crew_per_ship : ℕ) : ℕ :=
  num_islands * ships_per_island * crew_per_ship

/-- Theorem stating the total number of crew members in the given scenario -/
theorem crew_member_count :
  total_crew_members 3 12 24 = 864 := by
  sorry

end crew_member_count_l1547_154763


namespace tissue_cost_with_discount_l1547_154711

/-- Calculate the total cost of tissues with discount --/
theorem tissue_cost_with_discount
  (num_boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (discount_rate : ℚ)
  (h_num_boxes : num_boxes = 25)
  (h_packs_per_box : packs_per_box = 18)
  (h_tissues_per_pack : tissues_per_pack = 150)
  (h_price_per_tissue : price_per_tissue = 6 / 100)
  (h_discount_rate : discount_rate = 1 / 10) :
  (num_boxes : ℚ) * (packs_per_box : ℚ) * (tissues_per_pack : ℚ) * price_per_tissue *
    (1 - discount_rate) = 3645 := by
  sorry

#check tissue_cost_with_discount

end tissue_cost_with_discount_l1547_154711


namespace alphabet_composition_l1547_154718

theorem alphabet_composition (total : ℕ) (both : ℕ) (line_only : ℕ) (dot_only : ℕ) : 
  total = 40 →
  both = 8 →
  line_only = 24 →
  total = both + line_only + dot_only →
  dot_only = 8 := by
sorry

end alphabet_composition_l1547_154718


namespace inscribed_circle_larger_than_sphere_l1547_154705

structure Tetrahedron where
  inscribedSphereRadius : ℝ
  faceInscribedCircleRadius : ℝ
  inscribedSphereRadiusPositive : 0 < inscribedSphereRadius
  faceInscribedCircleRadiusPositive : 0 < faceInscribedCircleRadius

theorem inscribed_circle_larger_than_sphere (t : Tetrahedron) :
  t.faceInscribedCircleRadius > t.inscribedSphereRadius := by
  sorry

end inscribed_circle_larger_than_sphere_l1547_154705


namespace equation_solution_l1547_154707

theorem equation_solution :
  ∃ y : ℚ, (3 / y - (5 / y) / (7 / y) = 1.2) ∧ (y = 105 / 67) := by
  sorry

end equation_solution_l1547_154707


namespace hyperbola_equation_l1547_154726

/-- Represents a hyperbola with center at the origin -/
structure Hyperbola where
  /-- Distance from center to focus -/
  c : ℝ
  /-- Ratio of b to a in the standard equation -/
  b_over_a : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    h.b_over_a = b / a ∧
    h.c^2 = a^2 + b^2 ∧
    x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_equation (h : Hyperbola) (h_focus : h.c = 10) (h_asymptote : h.b_over_a = 4/3) :
  standard_equation h x y ↔ x^2 / 36 - y^2 / 64 = 1 :=
sorry

end hyperbola_equation_l1547_154726


namespace prob_different_ranks_value_l1547_154720

/-- The number of cards in a standard deck --/
def deck_size : ℕ := 52

/-- The number of ranks in a standard deck --/
def num_ranks : ℕ := 13

/-- The number of suits in a standard deck --/
def num_suits : ℕ := 4

/-- The probability of drawing two cards of different ranks from a standard deck --/
def prob_different_ranks : ℚ :=
  (deck_size * (deck_size - 1) - num_ranks * (num_suits * (num_suits - 1))) /
  (deck_size * (deck_size - 1))

theorem prob_different_ranks_value : prob_different_ranks = 208 / 221 := by
  sorry

end prob_different_ranks_value_l1547_154720


namespace money_left_calculation_l1547_154712

theorem money_left_calculation (initial_amount spent_on_sweets given_to_each_friend : ℚ) 
  (number_of_friends : ℕ) (h1 : initial_amount = 200.50) 
  (h2 : spent_on_sweets = 35.25) (h3 : given_to_each_friend = 25.20) 
  (h4 : number_of_friends = 2) : 
  initial_amount - spent_on_sweets - (given_to_each_friend * number_of_friends) = 114.85 := by
  sorry

end money_left_calculation_l1547_154712


namespace sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l1547_154780

theorem sixth_root_of_six (x : ℝ) (h : x > 0) : 
  (x^(1/2)) / (x^(1/3)) = x^(1/6) := by
  sorry

-- The specific case for x = 6
theorem sqrt_2_over_sqrt_3_of_6 : 
  (6^(1/2)) / (6^(1/3)) = 6^(1/6) := by
  sorry

end sixth_root_of_six_sqrt_2_over_sqrt_3_of_6_l1547_154780


namespace market_price_calculation_l1547_154739

/-- Proves that given a reduction in sales tax from 3.5% to 3 1/3% resulting in a
    difference of Rs. 12.99999999999999 in tax amount, the market price of the article is Rs. 7800. -/
theorem market_price_calculation (initial_tax : ℚ) (reduced_tax : ℚ) (tax_difference : ℚ) 
  (h1 : initial_tax = 7/200)  -- 3.5%
  (h2 : reduced_tax = 1/30)   -- 3 1/3%
  (h3 : tax_difference = 12999999999999999/1000000000000000) : -- 12.99999999999999
  ∃ (market_price : ℕ), 
    (initial_tax - reduced_tax) * market_price = tax_difference ∧ 
    market_price = 7800 := by
sorry

end market_price_calculation_l1547_154739


namespace book_donations_mode_l1547_154788

/-- Represents the distribution of book donations -/
def book_donations : List (ℕ × ℕ) := [
  (30, 40), (22, 30), (16, 25), (8, 50), (6, 20), (4, 35)
]

/-- Calculates the mode of a list of pairs (value, frequency) -/
def mode (l : List (ℕ × ℕ)) : ℕ :=
  let max_frequency := l.map Prod.snd |>.maximum?
  match max_frequency with
  | none => 0
  | some max => (l.filter (fun p => p.2 = max)).map Prod.fst |>.minimum?
                |>.getD 0

/-- Theorem: The mode of the book donations is 8 -/
theorem book_donations_mode :
  mode book_donations = 8 := by
  sorry

end book_donations_mode_l1547_154788


namespace at_least_one_root_exists_l1547_154767

theorem at_least_one_root_exists (c m a n : ℝ) : 
  (m^2 + 4*a*c ≥ 0) ∨ (n^2 - 4*a*c ≥ 0) := by sorry

end at_least_one_root_exists_l1547_154767


namespace rook_placement_on_colored_board_l1547_154758

theorem rook_placement_on_colored_board :
  let board_size : ℕ := 64
  let num_rooks : ℕ := 8
  let num_colors : ℕ := 32
  let cells_per_color : ℕ := 2

  let total_placements : ℕ := num_rooks.factorial
  let same_color_placements : ℕ := num_colors * (num_rooks - 2).factorial

  total_placements > same_color_placements :=
by sorry

end rook_placement_on_colored_board_l1547_154758


namespace functional_polynomial_form_l1547_154710

/-- A polynomial that satisfies the given functional equation. -/
structure FunctionalPolynomial where
  P : ℝ → ℝ
  nonzero : P ≠ 0
  satisfies_equation : ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The theorem stating the form of polynomials satisfying the functional equation. -/
theorem functional_polynomial_form (fp : FunctionalPolynomial) :
  ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, fp.P x = (x + 1)^n :=
sorry

end functional_polynomial_form_l1547_154710


namespace perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l1547_154709

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorems
theorem perpendicular_planes_through_perpendicular_line 
  (P Q : Plane) (l : Line) :
  passes_through Q l → perpendicular_line l P → perpendicular P Q := by sorry

theorem non_perpendicular_line_in_perpendicular_planes 
  (P Q : Plane) (l : Line) :
  perpendicular P Q → 
  in_plane l P → 
  ¬ perpendicular_lines l (line_of_intersection P Q) → 
  ¬ perpendicular_line l Q := by sorry

end perpendicular_planes_through_perpendicular_line_non_perpendicular_line_in_perpendicular_planes_l1547_154709


namespace square_binomial_coefficient_l1547_154744

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end square_binomial_coefficient_l1547_154744


namespace units_produced_today_l1547_154748

theorem units_produced_today (past_average : ℝ) (new_average : ℝ) (past_days : ℕ) :
  past_average = 40 →
  new_average = 45 →
  past_days = 9 →
  (past_days + 1) * new_average - past_days * past_average = 90 := by
  sorry

end units_produced_today_l1547_154748


namespace x_value_proof_l1547_154713

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 27) : x = 3 * (3 ^ (1/4)) := by
  sorry

end x_value_proof_l1547_154713


namespace function_not_in_third_quadrant_l1547_154721

theorem function_not_in_third_quadrant
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b > -1) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = a^x + b :=
by sorry

end function_not_in_third_quadrant_l1547_154721


namespace galaxy_first_chinese_supercomputer_l1547_154741

/-- Represents a supercomputer -/
structure Supercomputer where
  name : String
  country : String
  performance : ℕ  -- calculations per second
  year_introduced : ℕ
  month_introduced : ℕ

/-- The Galaxy supercomputer -/
def galaxy : Supercomputer :=
  { name := "Galaxy"
  , country := "China"
  , performance := 100000000  -- 100 million
  , year_introduced := 1983
  , month_introduced := 12 }

/-- Predicate to check if a supercomputer meets the criteria -/
def meets_criteria (sc : Supercomputer) : Prop :=
  sc.country = "China" ∧
  sc.performance ≥ 100000000 ∧
  sc.year_introduced = 1983 ∧
  sc.month_introduced = 12

/-- Theorem stating that Galaxy was China's first supercomputer meeting the criteria -/
theorem galaxy_first_chinese_supercomputer :
  meets_criteria galaxy ∧
  ∀ (sc : Supercomputer), meets_criteria sc → sc.name = galaxy.name :=
by sorry


end galaxy_first_chinese_supercomputer_l1547_154741


namespace weight_qualification_l1547_154776

/-- A weight is qualified if it falls within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

/-- The labeled weight of the flour -/
def labeled_weight : ℝ := 25

/-- The tolerance of the weight -/
def tolerance : ℝ := 0.25

theorem weight_qualification (weight : ℝ) :
  is_qualified weight ↔ labeled_weight - tolerance ≤ weight ∧ weight ≤ labeled_weight + tolerance :=
by sorry

end weight_qualification_l1547_154776


namespace distance_between_foci_l1547_154730

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 10)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 10)

-- Theorem: The distance between foci is √149
theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = Real.sqrt 149 := by
  sorry

#check distance_between_foci

end distance_between_foci_l1547_154730


namespace markov_equation_solution_l1547_154762

/-- Markov equation -/
def markov_equation (x y z : ℕ+) : Prop :=
  x^2 + y^2 + z^2 = 3*x*y*z

/-- Definition of coprime positive integers -/
def coprime (a b : ℕ+) : Prop :=
  Nat.gcd a.val b.val = 1

/-- Definition of sum of squares of two coprime integers -/
def sum_of_coprime_squares (a : ℕ+) : Prop :=
  ∃ (p q : ℕ+), coprime p q ∧ a = p^2 + q^2

/-- Main theorem -/
theorem markov_equation_solution :
  ∀ (a b c : ℕ+), markov_equation a b c →
    (coprime a b ∧ coprime b c ∧ coprime a c) ∧
    (a ≠ 1 → sum_of_coprime_squares a) :=
sorry

end markov_equation_solution_l1547_154762


namespace bad_carrots_count_l1547_154737

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  nancy_carrots = 38 → mom_carrots = 47 → good_carrots = 71 →
  nancy_carrots + mom_carrots - good_carrots = 14 := by
  sorry

end bad_carrots_count_l1547_154737


namespace courier_packages_l1547_154751

theorem courier_packages (x : ℕ) (h1 : x + 2*x = 240) : x = 80 := by
  sorry

end courier_packages_l1547_154751


namespace joana_shopping_problem_l1547_154742

theorem joana_shopping_problem :
  ∃! (b c : ℕ), 15 * b + 17 * c = 143 :=
by sorry

end joana_shopping_problem_l1547_154742


namespace square_EC_dot_ED_l1547_154736

/-- Square ABCD with side length 2 and E as midpoint of AB -/
structure Square2D where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  is_square : A.1 = B.1 ∧ A.2 = D.2 ∧ C.1 = D.1 ∧ C.2 = B.2
  side_length : ‖B - A‖ = 2
  E_midpoint : E = (A + B) / 2

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem square_EC_dot_ED (s : Square2D) :
  dot_product (s.C - s.E) (s.D - s.E) = 3 := by
  sorry

end square_EC_dot_ED_l1547_154736


namespace area_of_ABC_l1547_154759

-- Define the triangle ABC and point P
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def P : ℝ × ℝ := sorry

-- Define the conditions
def is_scalene_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def point_on_hypotenuse (A C P : ℝ × ℝ) : Prop := sorry
def angle_ABP_45 (A B P : ℝ × ℝ) : Prop := sorry
def AP_equals_1 (A P : ℝ × ℝ) : Prop := sorry
def CP_equals_2 (C P : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_ABC :
  is_scalene_right_triangle A B C →
  point_on_hypotenuse A C P →
  angle_ABP_45 A B P →
  AP_equals_1 A P →
  CP_equals_2 C P →
  area A B C = 9/5 := by
  sorry

end area_of_ABC_l1547_154759


namespace speed_calculation_l1547_154779

/-- Proves that given a distance of 600 meters and a time of 5 minutes, the speed is 7.2 km/hour -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 5) :
  (distance / 1000) / (time / 60) = 7.2 := by
  sorry

end speed_calculation_l1547_154779


namespace first_expression_value_l1547_154702

theorem first_expression_value (a : ℝ) (E : ℝ) : 
  a = 28 → (E + (3 * a - 8)) / 2 = 74 → E = 72 := by
  sorry

end first_expression_value_l1547_154702


namespace field_goal_missed_fraction_l1547_154755

theorem field_goal_missed_fraction 
  (total_attempts : ℕ) 
  (wide_right_percentage : ℚ) 
  (wide_right_count : ℕ) 
  (h1 : total_attempts = 60) 
  (h2 : wide_right_percentage = 1/5) 
  (h3 : wide_right_count = 3) : 
  (wide_right_count / wide_right_percentage) / total_attempts = 1/4 :=
sorry

end field_goal_missed_fraction_l1547_154755


namespace rainstorm_multiple_rainstorm_multiple_proof_l1547_154766

/-- Given the conditions of a rainstorm, prove that the multiple of the first hour's
    rain amount that determines the second hour's rain (minus 7 inches) is equal to 2. -/
theorem rainstorm_multiple : ℝ → Prop :=
  fun x =>
    let first_hour_rain := 5
    let second_hour_rain := x * first_hour_rain + 7
    let total_rain := 22
    first_hour_rain + second_hour_rain = total_rain →
    x = 2

/-- Proof of the rainstorm_multiple theorem -/
theorem rainstorm_multiple_proof : rainstorm_multiple 2 := by
  sorry

end rainstorm_multiple_rainstorm_multiple_proof_l1547_154766


namespace pig_count_l1547_154700

theorem pig_count (initial_pigs joining_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joining_pigs = 22) : 
  initial_pigs + joining_pigs = 86 := by
  sorry

end pig_count_l1547_154700


namespace digit_sum_power_equality_l1547_154719

-- Define the sum of digits function
def S (m : ℕ) : ℕ := sorry

-- Define the set of solutions
def solution_set : Set (ℕ × ℕ) :=
  {p | ∃ (b : ℕ), p = (1, b + 1)} ∪ {(3, 2), (9, 1)}

-- State the theorem
theorem digit_sum_power_equality :
  ∀ a b : ℕ, a > 0 → b > 0 →
  (S (a^(b+1)) = a^b ↔ (a, b) ∈ solution_set) := by sorry

end digit_sum_power_equality_l1547_154719


namespace investment_interest_proof_l1547_154791

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

theorem investment_interest_proof :
  let principal : ℝ := 1500
  let rate : ℝ := 0.03
  let time : ℕ := 10
  ⌊compound_interest principal rate time⌋ = 516 := by
  sorry

end investment_interest_proof_l1547_154791


namespace divisibility_conditions_solutions_l1547_154771

theorem divisibility_conditions_solutions (a b : ℕ+) : 
  (a ∣ b^2) → (b ∣ a^2) → ((a + 1) ∣ (b^2 + 1)) → 
  (∃ q : ℕ+, (a = q^2 ∧ b = q) ∨ 
             (a = q^2 ∧ b = q^3) ∨ 
             (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by sorry

end divisibility_conditions_solutions_l1547_154771


namespace tobys_breakfast_calories_l1547_154704

-- Define the calorie content of bread and peanut butter
def bread_calories : ℕ := 100
def peanut_butter_calories : ℕ := 200

-- Define Toby's breakfast composition
def bread_pieces : ℕ := 1
def peanut_butter_servings : ℕ := 2

-- Theorem to prove
theorem tobys_breakfast_calories :
  bread_calories * bread_pieces + peanut_butter_calories * peanut_butter_servings = 500 := by
  sorry

end tobys_breakfast_calories_l1547_154704


namespace container_volume_ratio_l1547_154754

theorem container_volume_ratio (C D : ℚ) 
  (h : C > 0 ∧ D > 0) 
  (transfer : (3 / 4 : ℚ) * C = (2 / 3 : ℚ) * D) : 
  C / D = 8 / 9 := by
sorry

end container_volume_ratio_l1547_154754


namespace rectangle_other_vertices_x_sum_l1547_154798

/-- Given a rectangle with two opposite vertices at (2, 23) and (8, -2),
    the sum of the x-coordinates of the other two vertices is 10. -/
theorem rectangle_other_vertices_x_sum :
  ∀ (A B : ℝ × ℝ),
  let v1 : ℝ × ℝ := (2, 23)
  let v2 : ℝ × ℝ := (8, -2)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  (A.1 + B.1) / 2 = midpoint.1 →
  A.1 + B.1 = 10 :=
by
  sorry

end rectangle_other_vertices_x_sum_l1547_154798


namespace time_after_3250_minutes_l1547_154782

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 1, day := 1, hour := 3, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3250

/-- The resulting date and time -/
def resultDateTime : DateTime :=
  { year := 2020, month := 1, day := 3, hour := 9, minute := 10 }

theorem time_after_3250_minutes :
  addMinutes startDateTime minutesToAdd = resultDateTime :=
sorry

end time_after_3250_minutes_l1547_154782


namespace nina_total_problems_l1547_154795

/-- Given the homework assignments for Ruby and the relative amounts for Nina,
    calculate the total number of problems Nina has to complete. -/
theorem nina_total_problems (ruby_math ruby_reading ruby_science : ℕ)
  (nina_math_factor nina_reading_factor nina_science_factor : ℕ) :
  ruby_math = 12 →
  ruby_reading = 4 →
  ruby_science = 5 →
  nina_math_factor = 5 →
  nina_reading_factor = 9 →
  nina_science_factor = 3 →
  ruby_math * nina_math_factor +
  ruby_reading * nina_reading_factor +
  ruby_science * nina_science_factor = 111 := by
sorry

end nina_total_problems_l1547_154795


namespace cows_eating_husk_l1547_154757

/-- The number of bags of husk eaten by a group of cows in 30 days -/
def bags_eaten (num_cows : ℕ) (bags_per_cow : ℕ) : ℕ :=
  num_cows * bags_per_cow

/-- Theorem: 30 cows eat 30 bags of husk in 30 days -/
theorem cows_eating_husk :
  bags_eaten 30 1 = 30 := by
  sorry

end cows_eating_husk_l1547_154757


namespace tan_alpha_equals_three_implies_ratio_equals_five_l1547_154743

theorem tan_alpha_equals_three_implies_ratio_equals_five (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end tan_alpha_equals_three_implies_ratio_equals_five_l1547_154743


namespace function_range_theorem_l1547_154773

/-- Given a function f(x) = |2x - 1| + |x - 2a|, if for all x ∈ [1, 2], f(x) ≤ 4,
    then the range of real values for a is [1/2, 3/2]. -/
theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f x = |2 * x - 1| + |x - 2 * a|) →
  (∀ x ∈ Set.Icc 1 2, f x ≤ 4) →
  a ∈ Set.Icc (1/2) (3/2) := by
  sorry

end function_range_theorem_l1547_154773


namespace fraction_equality_solution_l1547_154784

theorem fraction_equality_solution :
  ∀ m n : ℕ+, 
  (m : ℚ) / ((n : ℚ) + m) = (n : ℚ) / ((n : ℚ) - m) →
  (∃ h : ℕ, m = (2*h + 1)*h ∧ n = (2*h + 1)*(h + 1)) ∨
  (∃ h : ℕ+, m = 2*h*(4*h^2 - 1) ∧ n = 2*h*(4*h^2 + 1)) :=
by sorry

end fraction_equality_solution_l1547_154784
