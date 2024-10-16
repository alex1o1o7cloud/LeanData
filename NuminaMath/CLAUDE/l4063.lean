import Mathlib

namespace NUMINAMATH_CALUDE_longest_tape_measure_l4063_406356

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 2400) 
  (hb : b = 3600) 
  (hc : c = 5400) : 
  Nat.gcd a (Nat.gcd b c) = 300 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l4063_406356


namespace NUMINAMATH_CALUDE_exist_consecutive_sum_digits_div_13_l4063_406354

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Theorem: There exist two consecutive natural numbers such that
    the sum of the digits of each of them is divisible by 13 -/
theorem exist_consecutive_sum_digits_div_13 :
  ∃ n : ℕ, 13 ∣ sum_of_digits n ∧ 13 ∣ sum_of_digits (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_sum_digits_div_13_l4063_406354


namespace NUMINAMATH_CALUDE_max_power_under_500_l4063_406305

theorem max_power_under_500 :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 1 ∧ x^y < 500 ∧
    (∀ (a b : ℕ), a > 0 → b > 1 → a^b < 500 → a^b ≤ x^y) ∧
    x = 22 ∧ y = 2 ∧ x + y = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_power_under_500_l4063_406305


namespace NUMINAMATH_CALUDE_seedling_difference_seedling_difference_proof_l4063_406322

theorem seedling_difference : ℕ → ℕ → ℕ → Prop :=
  fun pine_seedlings poplar_multiplier difference =>
    pine_seedlings = 180 →
    poplar_multiplier = 4 →
    difference = poplar_multiplier * pine_seedlings - pine_seedlings →
    difference = 540

-- Proof
theorem seedling_difference_proof : seedling_difference 180 4 540 := by
  sorry

end NUMINAMATH_CALUDE_seedling_difference_seedling_difference_proof_l4063_406322


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l4063_406323

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l4063_406323


namespace NUMINAMATH_CALUDE_volume_of_cut_cube_l4063_406336

/-- Represents a three-dimensional solid --/
structure Solid :=
  (volume : ℝ)

/-- Represents a cube --/
def Cube (edge_length : ℝ) : Solid :=
  { volume := edge_length ^ 3 }

/-- Represents the result of cutting parts off a cube --/
def CutCube (c : Solid) (cut_volume : ℝ) : Solid :=
  { volume := c.volume - cut_volume }

/-- Theorem stating that the volume of the resulting solid is 9 --/
theorem volume_of_cut_cube : 
  ∃ (cut_volume : ℝ), 
    (CutCube (Cube 3) cut_volume).volume = 9 :=
sorry

end NUMINAMATH_CALUDE_volume_of_cut_cube_l4063_406336


namespace NUMINAMATH_CALUDE_snow_total_l4063_406311

theorem snow_total (monday_snow tuesday_snow : Real) 
  (h1 : monday_snow = 0.32)
  (h2 : tuesday_snow = 0.21) : 
  monday_snow + tuesday_snow = 0.53 := by
sorry

end NUMINAMATH_CALUDE_snow_total_l4063_406311


namespace NUMINAMATH_CALUDE_bear_cubs_count_l4063_406326

/-- Represents the bear's hunting scenario -/
structure BearHunt where
  totalMeat : ℕ  -- Total meat needed per week
  cubMeat : ℕ    -- Meat needed per cub per week
  rabbitWeight : ℕ -- Weight of each rabbit
  dailyCatch : ℕ  -- Number of rabbits caught daily

/-- Calculates the number of cubs based on the hunting scenario -/
def numCubs (hunt : BearHunt) : ℕ :=
  let weeklyHunt := hunt.dailyCatch * hunt.rabbitWeight * 7
  (weeklyHunt - hunt.totalMeat) / hunt.cubMeat

/-- Theorem stating that the number of cubs is 4 given the specific hunting scenario -/
theorem bear_cubs_count (hunt : BearHunt) 
  (h1 : hunt.totalMeat = 210)
  (h2 : hunt.cubMeat = 35)
  (h3 : hunt.rabbitWeight = 5)
  (h4 : hunt.dailyCatch = 10) :
  numCubs hunt = 4 := by
  sorry

end NUMINAMATH_CALUDE_bear_cubs_count_l4063_406326


namespace NUMINAMATH_CALUDE_max_ac_without_racing_stripes_l4063_406370

/-- Represents the properties of a car group -/
structure CarGroup where
  total : ℕ
  without_ac : ℕ
  with_racing_stripes : ℕ
  (total_valid : total = 100)
  (without_ac_valid : without_ac = 37)
  (racing_stripes_valid : with_racing_stripes ≥ 41)

/-- Theorem: The greatest number of cars that could have air conditioning but not racing stripes -/
theorem max_ac_without_racing_stripes (group : CarGroup) : 
  (group.total - group.without_ac) - group.with_racing_stripes ≤ 22 :=
sorry

end NUMINAMATH_CALUDE_max_ac_without_racing_stripes_l4063_406370


namespace NUMINAMATH_CALUDE_min_regions_for_12_intersections_l4063_406338

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A set of circles in a plane --/
def CircleSet := Set Circle

/-- The number of intersection points between circles in a set --/
def intersectionPoints (s : CircleSet) : ℕ := sorry

/-- The number of regions into which a set of circles divides the plane --/
def regions (s : CircleSet) : ℕ := sorry

/-- The theorem stating the minimum number of regions --/
theorem min_regions_for_12_intersections (s : CircleSet) :
  intersectionPoints s = 12 → regions s ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_regions_for_12_intersections_l4063_406338


namespace NUMINAMATH_CALUDE_integers_between_negative_two_and_three_l4063_406352

theorem integers_between_negative_two_and_three :
  {x : ℤ | x > -2 ∧ x ≤ 3} = {-1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_integers_between_negative_two_and_three_l4063_406352


namespace NUMINAMATH_CALUDE_tan_70_cos_10_expression_l4063_406385

theorem tan_70_cos_10_expression : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_expression_l4063_406385


namespace NUMINAMATH_CALUDE_min_value_on_circle_l4063_406318

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' + 6*y' + 12 = 0 →
    |2*x' - y' - 2| ≥ min) ∧ min = 5 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l4063_406318


namespace NUMINAMATH_CALUDE_scarves_per_box_l4063_406313

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  mittens_per_box = 6 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * mittens_per_box) / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_scarves_per_box_l4063_406313


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_231_4620_l4063_406365

theorem gcd_lcm_sum_231_4620 : Nat.gcd 231 4620 + Nat.lcm 231 4620 = 4851 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_231_4620_l4063_406365


namespace NUMINAMATH_CALUDE_triangle_side_equations_l4063_406369

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The equation of a line given two points -/
def lineEquation (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  { a := a, b := b, c := c }

theorem triangle_side_equations (A B C : Point)
  (hA : A = { x := -5, y := 0 })
  (hB : B = { x := 3, y := -3 })
  (hC : C = { x := 0, y := 2 }) :
  let AB := lineEquation A B
  let AC := lineEquation A C
  let BC := lineEquation B C
  AB = { a := 3, b := 8, c := 15 } ∧
  AC = { a := 2, b := -5, c := 10 } ∧
  BC = { a := 5, b := 3, c := -6 } :=
sorry

end NUMINAMATH_CALUDE_triangle_side_equations_l4063_406369


namespace NUMINAMATH_CALUDE_range_of_a_correct_l4063_406346

/-- Proposition p: The solution set of a^x > 1 (a > 0 and a ≠ 1) is {x | x < 0} -/
def p (a : ℝ) : Prop :=
  0 < a ∧ a ≠ 1 ∧ ∀ x, a^x > 1 ↔ x < 0

/-- Proposition q: The domain of y = log(x^2 - x + a) is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x, x^2 - x + a > 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ :=
  {a | (0 < a ∧ a ≤ 1/4) ∨ a ≥ 1}

theorem range_of_a_correct :
  ∀ a, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_correct_l4063_406346


namespace NUMINAMATH_CALUDE_equation_solution_l4063_406316

theorem equation_solution :
  ∀ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ↔ x = -16 / 7 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4063_406316


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l4063_406366

theorem fraction_inequality_solution (y : ℝ) : 
  1 / (y * (y + 2)) - 1 / ((y + 2) * (y + 4)) < 1 / 4 ↔ 
  y < -4 ∨ (-2 < y ∧ y < 0) ∨ y > 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l4063_406366


namespace NUMINAMATH_CALUDE_special_polyhedron_ratio_l4063_406363

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ
  x : ℝ
  y : ℝ
  all_faces_isosceles : Prop
  edge_lengths : Prop
  vertex_degrees : Prop
  equal_dihedral_angles : Prop

/-- The theorem statement -/
theorem special_polyhedron_ratio 
  (P : SpecialPolyhedron)
  (h_faces : P.faces = 12)
  (h_edges : P.edges = 18)
  (h_vertices : P.vertices = 8)
  (h_isosceles : P.all_faces_isosceles)
  (h_edge_lengths : P.edge_lengths)
  (h_vertex_degrees : P.vertex_degrees)
  (h_dihedral_angles : P.equal_dihedral_angles)
  : P.x / P.y = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_special_polyhedron_ratio_l4063_406363


namespace NUMINAMATH_CALUDE_P_superset_Q_l4063_406377

-- Define the sets P and Q
def P : Set ℝ := {x | x < 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem stating the relationship between P and Q
theorem P_superset_Q : P ⊃ Q := by sorry

end NUMINAMATH_CALUDE_P_superset_Q_l4063_406377


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4063_406325

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h2 : a 1 - a 2 = 3) 
  (h3 : a 1 - a 3 = 2) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = -1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4063_406325


namespace NUMINAMATH_CALUDE_total_profit_is_100_l4063_406319

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_ratio := a_investment * a_months
  let b_investment_ratio := b_investment * b_months
  let total_investment_ratio := a_investment_ratio + b_investment_ratio
  (a_profit_share * total_investment_ratio) / a_investment_ratio

/-- Proves that the total profit is $100 given the specified investments and A's profit share -/
theorem total_profit_is_100 :
  calculate_total_profit 100 12 200 6 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l4063_406319


namespace NUMINAMATH_CALUDE_fraction_value_l4063_406300

theorem fraction_value (x y : ℝ) (h : (1 / x) + (1 / y) = 2) :
  (-2 * y + x * y - 2 * x) / (3 * x + x * y + 3 * y) = -3 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l4063_406300


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l4063_406341

/-- Represents a parabola with equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (0, 1)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_focus_and_directrix (p : Parabola) :
  (focus p = (0, 1)) ∧ (directrix p = fun y => y = -1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l4063_406341


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l4063_406355

/-- Given a parabola x² = 2py where p > 0, with a point M(4, y₀) on the parabola,
    and the distance between M and the focus F being |MF| = 5/4 * y₀,
    prove that the coordinates of the focus F are (0, 1). -/
theorem parabola_focus_coordinates (p : ℝ) (y₀ : ℝ) (h_p : p > 0) :
  x^2 = 2*p*y →
  4^2 = 2*p*y₀ →
  (4^2 + (y₀ - p/2)^2)^(1/2) = 5/4 * y₀ →
  (0, 1) = (0, p/2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l4063_406355


namespace NUMINAMATH_CALUDE_octal_subtraction_l4063_406340

/-- Convert a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 * 64 + d2 * 8 + d3

/-- Convert a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ :=
  let d1 := n / 64
  let d2 := (n / 8) % 8
  let d3 := n % 8
  d1 * 100 + d2 * 10 + d3

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 526 - octal_to_decimal 321) = 205 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_l4063_406340


namespace NUMINAMATH_CALUDE_sin_pi_6_plus_cos_pi_3_simplification_l4063_406353

theorem sin_pi_6_plus_cos_pi_3_simplification (α : ℝ) : 
  Real.sin (π/6 + α) + Real.cos (π/3 + α) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_6_plus_cos_pi_3_simplification_l4063_406353


namespace NUMINAMATH_CALUDE_sqrt_simplification_complex_expression_simplification_square_difference_simplification_l4063_406373

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 27 + Real.sqrt 3 - Real.sqrt 12 = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression_simplification :
  1 / Real.sqrt 24 + |Real.sqrt 6 - 3| + (1 / 2)⁻¹ - 2016^0 = 4 - 13 * Real.sqrt 6 / 12 := by sorry

-- Problem 3
theorem square_difference_simplification :
  (Real.sqrt 3 + Real.sqrt 2)^2 - (Real.sqrt 3 - Real.sqrt 2)^2 = 4 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_simplification_complex_expression_simplification_square_difference_simplification_l4063_406373


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l4063_406342

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  (a * Real.cos B + b * Real.cos A) * Real.cos (2 * C) = c * Real.cos C →
  b = 2 * a →
  S = (Real.sqrt 3 / 2) * Real.sin A * Real.sin B →
  C = 2 * Real.pi / 3 ∧
  Real.sin A = Real.sqrt 21 / 14 ∧
  c = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l4063_406342


namespace NUMINAMATH_CALUDE_apples_in_first_group_l4063_406304

-- Define the cost of an apple
def apple_cost : ℚ := 21/100

-- Define the equation for the first group
def first_group (x : ℚ) (orange_cost : ℚ) : Prop :=
  x * apple_cost + 3 * orange_cost = 177/100

-- Define the equation for the second group
def second_group (orange_cost : ℚ) : Prop :=
  2 * apple_cost + 5 * orange_cost = 127/100

-- Theorem stating that the number of apples in the first group is 6
theorem apples_in_first_group :
  ∃ (orange_cost : ℚ), first_group 6 orange_cost ∧ second_group orange_cost :=
sorry

end NUMINAMATH_CALUDE_apples_in_first_group_l4063_406304


namespace NUMINAMATH_CALUDE_vector_decomposition_l4063_406381

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 1, 3]
def p : Fin 3 → ℝ := ![2, 1, 0]
def q : Fin 3 → ℝ := ![1, 0, 1]
def r : Fin 3 → ℝ := ![4, 2, 1]

/-- The decomposition of x in terms of p, q, and r -/
theorem vector_decomposition : x = (-3 : ℝ) • p + q + (2 : ℝ) • r := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l4063_406381


namespace NUMINAMATH_CALUDE_problem_statement_l4063_406332

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4063_406332


namespace NUMINAMATH_CALUDE_neither_a_nor_b_probability_l4063_406391

def prob_a : ℝ := 0.20
def prob_b : ℝ := 0.40
def prob_a_and_b : ℝ := 0.15

theorem neither_a_nor_b_probability :
  1 - (prob_a + prob_b - prob_a_and_b) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_neither_a_nor_b_probability_l4063_406391


namespace NUMINAMATH_CALUDE_plumber_distribution_l4063_406309

/-- The number of ways to distribute n plumbers to k residences,
    where all plumbers are assigned, each plumber goes to only one residence,
    and each residence has at least one plumber. -/
def distributionSchemes (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 plumbers to 4 residences
    results in 240 different distribution schemes. -/
theorem plumber_distribution :
  distributionSchemes 5 4 = 240 := by sorry

end NUMINAMATH_CALUDE_plumber_distribution_l4063_406309


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l4063_406372

/-- The amount Sandy spent on clothes, in cents -/
def total_spent : ℕ := 3356

/-- The cost of shorts, in cents -/
def shorts_cost : ℕ := 1399

/-- The cost of jacket, in cents -/
def jacket_cost : ℕ := 743

/-- The cost of shirt, in cents -/
def shirt_cost : ℕ := total_spent - (shorts_cost + jacket_cost)

theorem sandy_shirt_cost : shirt_cost = 1214 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l4063_406372


namespace NUMINAMATH_CALUDE_train_speed_l4063_406315

/-- Given a train of length 360 meters passing a bridge of length 140 meters in 36 seconds,
    prove that its speed is 50 km/h. -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 36 →
  (train_length + bridge_length) / time * 3.6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4063_406315


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4063_406330

/-- Given vectors a and b, if there exist real numbers m and n such that 
    m*a + n*b = (5,-5), then m - n = -2 -/
theorem vector_equation_solution (a b : ℝ × ℝ) 
    (h : a = (2, 1) ∧ b = (1, -2)) :
  ∃ (m n : ℝ), m • a + n • b = (5, -5) → m - n = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4063_406330


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_five_l4063_406350

theorem largest_five_digit_divisible_by_five : 
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 5 = 0 → n ≤ 99995 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_five_l4063_406350


namespace NUMINAMATH_CALUDE_sum_lent_is_1000_l4063_406317

/-- Proves that the sum lent is $1000 given the specified conditions -/
theorem sum_lent_is_1000 
  (interest_rate : ℝ) 
  (loan_duration : ℝ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.06)
  (h2 : loan_duration = 8)
  (h3 : interest_difference = 520)
  (simple_interest : ℝ → ℝ → ℝ → ℝ)
  (h4 : ∀ P r t, simple_interest P r t = P * r * t) :
  ∃ P : ℝ, P = 1000 ∧ simple_interest P interest_rate loan_duration = P - interest_difference :=
by sorry

end NUMINAMATH_CALUDE_sum_lent_is_1000_l4063_406317


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l4063_406376

theorem square_ratio_side_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a / b = 75 / 128) → (Real.sqrt (a / b) = 5 * Real.sqrt 6 / 16) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l4063_406376


namespace NUMINAMATH_CALUDE_quadratic_integer_conjugate_theorem_l4063_406329

/-- A structure representing a quadratic integer of the form a + b√d -/
structure QuadraticInteger (d : ℕ) where
  a : ℤ
  b : ℤ

/-- The conjugate of a quadratic integer -/
def conjugate {d : ℕ} (z : QuadraticInteger d) : QuadraticInteger d :=
  ⟨z.a, -z.b⟩

theorem quadratic_integer_conjugate_theorem
  (d : ℕ) (x₀ y₀ x y X Y : ℤ) (r : ℕ) 
  (h_d : ¬ ∃ (n : ℕ), n ^ 2 = d)
  (h_pos : x₀ > 0 ∧ y₀ > 0 ∧ x > 0 ∧ y > 0)
  (h_eq : X + Y * d^(1/2) = (x + y * d^(1/2)) * (x₀ - y₀ * d^(1/2))^r) :
  X - Y * d^(1/2) = (x - y * d^(1/2)) * (x₀ + y₀ * d^(1/2))^r := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_conjugate_theorem_l4063_406329


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_8_l4063_406337

def binary_number : ℕ := 110110111010

-- Define a function to get the last three bits of a binary number
def last_three_bits (n : ℕ) : ℕ := n % 8

-- Theorem statement
theorem remainder_of_binary_div_8 :
  binary_number % 8 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_8_l4063_406337


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l4063_406379

theorem min_value_fraction_sum (a b : ℤ) (h : a > b) :
  (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) ≥ 13 / 6 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (a b : ℤ), a > b ∧ (((2 * a + b) : ℚ) / (a - b : ℚ)) + ((a - b : ℚ) / ((2 * a + b) : ℚ)) = 13 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_min_value_fraction_sum_achievable_l4063_406379


namespace NUMINAMATH_CALUDE_afternoon_rowing_count_l4063_406380

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowing (morning_rowing hiking total : ℕ) : ℕ :=
  total - (morning_rowing + hiking)

/-- Theorem stating that 26 campers went rowing in the afternoon -/
theorem afternoon_rowing_count :
  afternoon_rowing 41 4 71 = 26 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowing_count_l4063_406380


namespace NUMINAMATH_CALUDE_distinct_permutations_with_repetitions_l4063_406314

-- Define the number of elements
def n : ℕ := 5

-- Define the number of repetitions for the first digit
def r1 : ℕ := 3

-- Define the number of repetitions for the second digit
def r2 : ℕ := 2

-- State the theorem
theorem distinct_permutations_with_repetitions :
  (n.factorial) / (r1.factorial * r2.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_permutations_with_repetitions_l4063_406314


namespace NUMINAMATH_CALUDE_gcd_7654321_6789012_l4063_406333

theorem gcd_7654321_6789012 : Nat.gcd 7654321 6789012 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7654321_6789012_l4063_406333


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4063_406389

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l4063_406389


namespace NUMINAMATH_CALUDE_solution_value_l4063_406394

theorem solution_value (r s : ℝ) : 
  (3 * r^2 - 5 * r = 7) → 
  (3 * s^2 - 5 * s = 7) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l4063_406394


namespace NUMINAMATH_CALUDE_determine_investment_l4063_406349

/-- Represents the investment and profit share of a person -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with a specific profit sharing ratio and one known investment,
    prove that the other investor's investment can be determined -/
theorem determine_investment (p q : Investor) (h1 : p.profitShare = 2)
    (h2 : q.profitShare = 4) (h3 : p.investment = 500000) :
    q.investment = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_determine_investment_l4063_406349


namespace NUMINAMATH_CALUDE_cube_surface_area_l4063_406399

/-- The surface area of a cube with edge length 6 cm is 216 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 6
  let surface_area := 6 * edge_length^2
  surface_area = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l4063_406399


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_l4063_406393

def volleyball_team_size : ℕ := 16
def num_twins : ℕ := 2
def num_starters : ℕ := 8

theorem volleyball_lineup_count :
  (Nat.choose (volleyball_team_size - num_twins) num_starters) +
  (num_twins * Nat.choose (volleyball_team_size - num_twins) (num_starters - 1)) = 9867 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_l4063_406393


namespace NUMINAMATH_CALUDE_weight_sum_abby_damon_l4063_406331

/-- Given the weights of four people in pairs, prove that the sum of the weights of the first and fourth person is 300 pounds. -/
theorem weight_sum_abby_damon (a b c d : ℕ) : 
  a + b = 270 → 
  b + c = 250 → 
  c + d = 280 → 
  a + c = 300 → 
  a + d = 300 := by
sorry

end NUMINAMATH_CALUDE_weight_sum_abby_damon_l4063_406331


namespace NUMINAMATH_CALUDE_total_peanuts_l4063_406348

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l4063_406348


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l4063_406335

theorem triangle_third_side_length : ∀ (x : ℝ),
  (x > 0 ∧ 5 + 9 > x ∧ x + 5 > 9 ∧ x + 9 > 5) → x = 8 ∨ (x < 8 ∨ x > 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l4063_406335


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4063_406359

/-- A polynomial is exactly divisible by (x-1)^3 if and only if its coefficients satisfy specific conditions -/
theorem polynomial_divisibility (a b c : ℤ) : 
  (∃ q : Polynomial ℤ, x^4 + a*x^2 + b*x + c = (x - 1)^3 * q) ↔ 
  (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry


end NUMINAMATH_CALUDE_polynomial_divisibility_l4063_406359


namespace NUMINAMATH_CALUDE_pens_per_student_l4063_406312

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) : 
  total_pens = 1001 → total_pencils = 910 → num_students = 91 →
  total_pens / num_students = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_per_student_l4063_406312


namespace NUMINAMATH_CALUDE_sports_only_count_l4063_406324

theorem sports_only_count (total employees : ℕ) (sports_fans : ℕ) (art_fans : ℕ) (neither_fans : ℕ) :
  total = 60 →
  sports_fans = 28 →
  art_fans = 26 →
  neither_fans = 12 →
  sports_fans - (total - neither_fans - art_fans) = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_only_count_l4063_406324


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l4063_406382

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a + b + c ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l4063_406382


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4063_406395

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4063_406395


namespace NUMINAMATH_CALUDE_square_area_11cm_l4063_406351

/-- The area of a square with side length 11 cm is 121 cm². -/
theorem square_area_11cm (side_length : ℝ) (h : side_length = 11) :
  side_length * side_length = 121 := by
  sorry

end NUMINAMATH_CALUDE_square_area_11cm_l4063_406351


namespace NUMINAMATH_CALUDE_matrix_power_2023_l4063_406306

theorem matrix_power_2023 :
  let A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 0, 1]
  A ^ 2023 = !![1, 2023; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l4063_406306


namespace NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l4063_406360

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_pure_imaginary_l4063_406360


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l4063_406387

/-- A function f is increasing on an interval [a, b) if for any x, y in [a, b) with x < y, f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y < b → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 2a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 * a

theorem range_of_a_for_increasing_f :
  {a : ℝ | IsIncreasing (f a) (-1) 2} = Set.Icc (-1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l4063_406387


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l4063_406321

theorem ratio_sum_problem (a b c : ℝ) 
  (ratio : a / 4 = b / 5 ∧ b / 5 = c / 7)
  (sum : a + b + c = 240) :
  2 * b - a + c = 195 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l4063_406321


namespace NUMINAMATH_CALUDE_garden_plant_count_l4063_406371

/-- The number of plants in a garden with given rows and columns -/
def garden_plants (rows : ℕ) (columns : ℕ) : ℕ := rows * columns

/-- Theorem: A garden with 52 rows and 15 columns has 780 plants -/
theorem garden_plant_count : garden_plants 52 15 = 780 := by
  sorry

end NUMINAMATH_CALUDE_garden_plant_count_l4063_406371


namespace NUMINAMATH_CALUDE_concert_ticket_price_l4063_406390

theorem concert_ticket_price (num_tickets : ℕ) (total_spent : ℚ) (h1 : num_tickets = 8) (h2 : total_spent = 32) : 
  total_spent / num_tickets = 4 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l4063_406390


namespace NUMINAMATH_CALUDE_somu_age_problem_l4063_406384

/-- Proves that Somu was one-fifth of his father's age 8 years ago -/
theorem somu_age_problem (somu_age father_age years_ago : ℕ) : 
  somu_age = 16 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 8 := by
  sorry


end NUMINAMATH_CALUDE_somu_age_problem_l4063_406384


namespace NUMINAMATH_CALUDE_quadratic_maximum_l4063_406347

theorem quadratic_maximum : 
  (∀ r : ℝ, -5 * r^2 + 40 * r - 12 ≤ 68) ∧ 
  (∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l4063_406347


namespace NUMINAMATH_CALUDE_new_average_weight_l4063_406386

theorem new_average_weight (initial_count : ℕ) (initial_average : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_average = 28 →
  new_student_weight = 22 →
  (initial_count * initial_average + new_student_weight) / (initial_count + 1) = 27.8 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l4063_406386


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4063_406303

theorem constant_term_expansion (x : ℝ) : 
  let expansion := (x + 1/x - 2)^5
  ∃ (p : ℝ → ℝ), expansion = p x ∧ p 0 = -252 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4063_406303


namespace NUMINAMATH_CALUDE_remaining_region_area_l4063_406364

/-- Represents a rectangle divided into five regions -/
structure DividedRectangle where
  total_area : ℝ
  region1_area : ℝ
  region2_area : ℝ
  region3_area : ℝ
  region4_area : ℝ
  region5_area : ℝ
  area_sum : total_area = region1_area + region2_area + region3_area + region4_area + region5_area

/-- The theorem stating that one of the remaining regions has an area of 27 square units -/
theorem remaining_region_area (rect : DividedRectangle) 
    (h1 : rect.total_area = 72)
    (h2 : rect.region1_area = 15)
    (h3 : rect.region2_area = 12)
    (h4 : rect.region3_area = 18) :
    rect.region4_area = 27 ∨ rect.region5_area = 27 :=
  sorry

end NUMINAMATH_CALUDE_remaining_region_area_l4063_406364


namespace NUMINAMATH_CALUDE_work_rate_problem_l4063_406378

theorem work_rate_problem (a b : ℝ) (h1 : a = (1/2) * b) (h2 : (a + b) * 20 = 1) :
  1 / b = 30 := by sorry

end NUMINAMATH_CALUDE_work_rate_problem_l4063_406378


namespace NUMINAMATH_CALUDE_sqrt_three_expression_l4063_406361

theorem sqrt_three_expression : Real.sqrt 3 * (Real.sqrt 3 - 1 / Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_l4063_406361


namespace NUMINAMATH_CALUDE_binomial_expansion_special_case_l4063_406398

theorem binomial_expansion_special_case : 
  98^3 + 3*(98^2)*2 + 3*98*(2^2) + 2^3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_special_case_l4063_406398


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4063_406334

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def GeometricSequence.sum (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a 1 else g.a 1 * (1 - g.q ^ n) / (1 - g.q)

theorem geometric_sequence_fourth_term 
  (g : GeometricSequence) 
  (h1 : g.a 1 - g.a 5 = -15/2) 
  (h2 : g.sum 4 = -5) : 
  g.a 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4063_406334


namespace NUMINAMATH_CALUDE_shares_of_z_l4063_406392

/-- Represents the number of shares for each stock --/
structure Shares :=
  (v w x y z : ℕ)

/-- Calculates the range of shares --/
def range (s : Shares) : ℕ := max s.v (max s.w (max s.x (max s.y s.z))) - min s.v (min s.w (min s.x (min s.y s.z)))

/-- Initial shares --/
def initial : Shares := ⟨68, 112, 56, 94, 0⟩  -- z is set to 0 as a placeholder

/-- Shares after the transaction --/
def after (s : Shares) : Shares := ⟨s.v, s.w, s.x - 20, s.y + 23, s.z⟩

theorem shares_of_z (z : ℕ) : 
  initial.z = z →
  range (after ⟨initial.v, initial.w, initial.x, initial.y, z⟩) = range initial + 14 →
  z = 47 := by sorry

end NUMINAMATH_CALUDE_shares_of_z_l4063_406392


namespace NUMINAMATH_CALUDE_solve_slurpee_problem_l4063_406374

def slurpee_problem (initial_amount : ℕ) (slurpee_cost : ℕ) (change : ℕ) : Prop :=
  let amount_spent : ℕ := initial_amount - change
  let num_slurpees : ℕ := amount_spent / slurpee_cost
  num_slurpees = 6

theorem solve_slurpee_problem :
  slurpee_problem 20 2 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_slurpee_problem_l4063_406374


namespace NUMINAMATH_CALUDE_v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l4063_406339

-- Define the set v as cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^3}

-- Theorem stating that v is closed under multiplication and division
theorem v_closed_under_mult_and_div :
  (∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v) ∧
  (∀ a b : ℕ, a ∈ v → b ∈ v → b ≠ 0 → (a / b) ∈ v) :=
sorry

-- Theorem stating that v is not closed under addition
theorem v_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (a + b) ∉ v :=
sorry

-- Theorem stating that v is not closed under negative powers
theorem v_not_closed_under_negative_powers :
  ∃ a : ℕ, a ∈ v ∧ a ≠ 0 ∧ (1 / a) ∉ v :=
sorry

end NUMINAMATH_CALUDE_v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l4063_406339


namespace NUMINAMATH_CALUDE_f_not_differentiable_at_zero_l4063_406383

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.sin (x * Real.sin (3 / x)) else 0

theorem f_not_differentiable_at_zero :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end NUMINAMATH_CALUDE_f_not_differentiable_at_zero_l4063_406383


namespace NUMINAMATH_CALUDE_stair_climbing_comparison_l4063_406397

/-- Given two people climbing stairs at different speeds, this theorem calculates
    how many steps the faster person climbs when the slower person reaches a certain height. -/
theorem stair_climbing_comparison
  (matt_speed : ℕ)  -- Matt's speed in steps per minute
  (tom_speed_diff : ℕ)  -- How many more steps per minute Tom climbs compared to Matt
  (matt_steps : ℕ)  -- Number of steps Matt has climbed
  (matt_speed_pos : 0 < matt_speed)  -- Matt's speed is positive
  (h_matt_speed : matt_speed = 20)  -- Matt's actual speed
  (h_tom_speed_diff : tom_speed_diff = 5)  -- Tom's speed difference
  (h_matt_steps : matt_steps = 220)  -- Steps Matt has climbed
  : (matt_steps + (matt_steps / matt_speed) * tom_speed_diff : ℕ) = 275 := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_comparison_l4063_406397


namespace NUMINAMATH_CALUDE_existence_of_special_set_l4063_406396

theorem existence_of_special_set : ∃ (A : Set ℕ), 
  ∀ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) → (Set.Infinite S) →
    ∃ (k : ℕ) (m n : ℕ),
      k ≥ 2 ∧
      m ∈ A ∧
      n ∉ A ∧
      (∃ (factors_m factors_n : Finset ℕ),
        factors_m.card = k ∧
        factors_n.card = k ∧
        (∀ p ∈ factors_m, p ∈ S) ∧
        (∀ p ∈ factors_n, p ∈ S) ∧
        (Finset.prod factors_m id = m) ∧
        (Finset.prod factors_n id = n)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l4063_406396


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4063_406362

-- Define the quadratic function
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) (h : f b c 0 = f b c 2) :
  f b c (3/2) < f b c 0 ∧ f b c 0 < f b c (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4063_406362


namespace NUMINAMATH_CALUDE_squares_containing_a_l4063_406357

/-- Represents a square in a grid -/
structure Square where
  size : Nat
  contains_a : Bool

/-- Represents a 4x4 grid -/
def Grid := Array (Array Square)

/-- Creates a 4x4 grid with A in one cell -/
def create_grid : Grid := sorry

/-- Counts the total number of squares in the grid -/
def total_squares (grid : Grid) : Nat := sorry

/-- Counts the number of squares containing A -/
def squares_with_a (grid : Grid) : Nat := sorry

/-- Theorem stating that there are 13 squares containing A in a 4x4 grid with A in one cell -/
theorem squares_containing_a (grid : Grid) :
  total_squares grid = 20 → squares_with_a grid = 13 := by sorry

end NUMINAMATH_CALUDE_squares_containing_a_l4063_406357


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l4063_406307

theorem polynomial_evaluation : (4 : ℝ)^4 + (4 : ℝ)^3 + (4 : ℝ)^2 + (4 : ℝ) + 1 = 341 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l4063_406307


namespace NUMINAMATH_CALUDE_apollonius_circle_minimum_l4063_406358

/-- Given points A, B, D, and a moving point P in a 2D plane, 
    prove that the minimum value of 2|PD|+|PB| is 2√10 when |PA|/|PB| = 1/2 -/
theorem apollonius_circle_minimum (A B D P : EuclideanSpace ℝ (Fin 2)) :
  A = ![(1 : ℝ), 0] →
  B = ![4, 0] →
  D = ![0, 3] →
  dist P A / dist P B = (1 : ℝ) / 2 →
  ∃ (P : EuclideanSpace ℝ (Fin 2)), 2 * dist P D + dist P B ≥ 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_apollonius_circle_minimum_l4063_406358


namespace NUMINAMATH_CALUDE_tech_group_selection_l4063_406327

theorem tech_group_selection (total : ℕ) (select : ℕ) (ways_with_girl : ℕ) :
  total = 6 →
  select = 3 →
  ways_with_girl = 16 →
  (Nat.choose total select - Nat.choose (total - (total - (Nat.choose total select - ways_with_girl))) select = ways_with_girl) →
  total - (Nat.choose total select - ways_with_girl) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tech_group_selection_l4063_406327


namespace NUMINAMATH_CALUDE_evaluate_expression_l4063_406302

theorem evaluate_expression : 2 - (-3) * 2 - 4 - (-5) * 3 - 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4063_406302


namespace NUMINAMATH_CALUDE_sin_240_degrees_l4063_406328

theorem sin_240_degrees : 
  Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l4063_406328


namespace NUMINAMATH_CALUDE_fraction_addition_l4063_406343

theorem fraction_addition : (7 : ℚ) / 12 + (3 : ℚ) / 8 = (23 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4063_406343


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_is_seven_tenths_l4063_406301

/-- Represents the composition of a research team -/
structure ResearchTeam where
  total : Nat
  males : Nat
  females : Nat

/-- Calculates the probability of at least one female being selected
    when choosing two representatives from a research team -/
def probAtLeastOneFemale (team : ResearchTeam) : Rat :=
  sorry

/-- The main theorem stating the probability for the given team composition -/
theorem prob_at_least_one_female_is_seven_tenths :
  let team : ResearchTeam := ⟨5, 3, 2⟩
  probAtLeastOneFemale team = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_is_seven_tenths_l4063_406301


namespace NUMINAMATH_CALUDE_combination_equality_l4063_406375

theorem combination_equality (x : ℕ+) : 
  (Nat.choose 10 x.val = Nat.choose 10 2) → (x.val = 2 ∨ x.val = 8) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l4063_406375


namespace NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l4063_406310

theorem unique_magnitude_of_complex_roots : ∃! r : ℝ, ∃ z : ℂ, z^2 - 8*z + 45 = 0 ∧ Complex.abs z = r := by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l4063_406310


namespace NUMINAMATH_CALUDE_population_percentage_l4063_406344

theorem population_percentage : 
  let total_population : ℕ := 40000
  let part_population : ℕ := 32000
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_population_percentage_l4063_406344


namespace NUMINAMATH_CALUDE_line_counting_theorem_l4063_406368

theorem line_counting_theorem (n : ℕ) : 
  n > 0 → 
  n % 4 = 3 → 
  (∀ k : ℕ, k ≤ n → k % 4 = (if k % 4 = 0 then 4 else k % 4)) → 
  n = 47 := by
sorry

end NUMINAMATH_CALUDE_line_counting_theorem_l4063_406368


namespace NUMINAMATH_CALUDE_pradeep_failed_marks_l4063_406345

def total_marks : ℕ := 550
def passing_percentage : ℚ := 40 / 100
def pradeep_marks : ℕ := 200

theorem pradeep_failed_marks : 
  (total_marks * passing_percentage).floor - pradeep_marks = 20 := by
  sorry

end NUMINAMATH_CALUDE_pradeep_failed_marks_l4063_406345


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4063_406388

/-- The equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- General form of hyperbola
  (∃ x y : ℝ, y^2 = -4*x) →  -- Parabola equation
  ((-1 : ℝ) = a) →  -- Real axis endpoint coincides with parabola focus
  ((a + b) / a = 2) →  -- Eccentricity is 2
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1) :=  -- Resulting hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4063_406388


namespace NUMINAMATH_CALUDE_set_operations_l4063_406308

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def B : Set ℕ := {4, 7, 8, 9}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = {4, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4063_406308


namespace NUMINAMATH_CALUDE_parabola_c_value_l4063_406320

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_c_value (p : Parabola) :
  p.y_at 3 = -5 →   -- Vertex at (3, -5)
  p.y_at 4 = -3 →   -- Passes through (4, -3)
  p.c = 13 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l4063_406320


namespace NUMINAMATH_CALUDE_dot_product_AO_AB_l4063_406367

/-- The circle O with equation x^2 + y^2 = 4 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The theorem statement -/
theorem dot_product_AO_AB (O A B : ℝ × ℝ) :
  A ∈ Circle → B ∈ Circle →
  ‖(A - O) + (B - O)‖ = ‖(A - O) - (B - O)‖ →
  (A - O) • (A - B) = 4 := by
sorry

end NUMINAMATH_CALUDE_dot_product_AO_AB_l4063_406367
