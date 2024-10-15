import Mathlib

namespace NUMINAMATH_CALUDE_cube_less_than_self_l3922_392229

theorem cube_less_than_self (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : a^3 < a := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_self_l3922_392229


namespace NUMINAMATH_CALUDE_pamela_skittles_l3922_392259

theorem pamela_skittles (initial_skittles : ℕ) (given_skittles : ℕ) : 
  initial_skittles = 50 → given_skittles = 7 → initial_skittles - given_skittles = 43 := by
sorry

end NUMINAMATH_CALUDE_pamela_skittles_l3922_392259


namespace NUMINAMATH_CALUDE_min_attempts_to_open_safe_l3922_392233

/-- Represents a sequence of 7 digits -/
def Code := Fin 7 → Fin 10

/-- Checks if all digits in a code are different -/
def all_different (c : Code) : Prop :=
  ∀ i j : Fin 7, i ≠ j → c i ≠ c j

/-- Checks if at least one digit in the attempt matches the secret code in the same position -/
def has_match (secret : Code) (attempt : Code) : Prop :=
  ∃ i : Fin 7, secret i = attempt i

/-- Represents a sequence of attempts to open the safe -/
def AttemptSequence (n : ℕ) := Fin n → Code

/-- Checks if a sequence of attempts guarantees opening the safe for any possible secret code -/
def guarantees_opening (attempts : AttemptSequence n) : Prop :=
  ∀ secret : Code, all_different secret →
    ∃ attempt ∈ Set.range attempts, all_different attempt ∧ has_match secret attempt

/-- The main theorem: 6 attempts are sufficient and necessary to guarantee opening the safe -/
theorem min_attempts_to_open_safe :
  (∃ attempts : AttemptSequence 6, guarantees_opening attempts) ∧
  (∀ n < 6, ¬∃ attempts : AttemptSequence n, guarantees_opening attempts) :=
sorry

end NUMINAMATH_CALUDE_min_attempts_to_open_safe_l3922_392233


namespace NUMINAMATH_CALUDE_solution_of_quadratic_equations_l3922_392226

theorem solution_of_quadratic_equations :
  let eq1 : ℝ → Prop := λ x ↦ 2 * x^2 = 3 * (2 * x + 1)
  let eq2 : ℝ → Prop := λ x ↦ 3 * x * (x + 2) = 4 * x + 8
  let sol1 : Set ℝ := {(3 + Real.sqrt 15) / 2, (3 - Real.sqrt 15) / 2}
  let sol2 : Set ℝ := {-2, 4/3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_quadratic_equations_l3922_392226


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_eight_l3922_392295

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Define the triangle inequality
def is_valid_triangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the quadratic equation
def is_root_of_equation (x : ℝ) : Prop :=
  x^2 - 4*x + 3 = 0

-- Theorem statement
theorem triangle_perimeter_is_eight :
  ∃ (t : Triangle), t.a = 2 ∧ t.b = 3 ∧ 
  is_root_of_equation t.c ∧ 
  is_valid_triangle t ∧
  perimeter t = 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_eight_l3922_392295


namespace NUMINAMATH_CALUDE_win_by_fourth_round_prob_l3922_392201

/-- The probability of winning a single round in Rock, Paper, Scissors -/
def win_prob : ℚ := 1 / 3

/-- The number of rounds needed to win the game -/
def rounds_to_win : ℕ := 3

/-- The total number of rounds played -/
def total_rounds : ℕ := 4

/-- The probability of winning by the fourth round in a "best of five" Rock, Paper, Scissors game -/
theorem win_by_fourth_round_prob :
  (Nat.choose (total_rounds - 1) (rounds_to_win - 1) : ℚ) *
  win_prob ^ (rounds_to_win - 1) *
  (1 - win_prob) ^ (total_rounds - rounds_to_win) *
  win_prob = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_win_by_fourth_round_prob_l3922_392201


namespace NUMINAMATH_CALUDE_line_x_intercept_x_intercept_is_four_l3922_392267

/-- A line passing through two points (1, 3) and (5, -1) has x-intercept 4 -/
theorem line_x_intercept : ℝ → ℝ → Prop :=
  fun (slope : ℝ) (x_intercept : ℝ) =>
    (slope = ((-1) - 3) / (5 - 1)) ∧
    (3 = slope * (1 - x_intercept)) ∧
    (x_intercept = 4)

/-- The x-intercept of the line passing through (1, 3) and (5, -1) is 4 -/
theorem x_intercept_is_four : ∃ (slope : ℝ), line_x_intercept slope 4 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_x_intercept_is_four_l3922_392267


namespace NUMINAMATH_CALUDE_composite_numbers_l3922_392247

theorem composite_numbers (n : ℕ) (h : n = 3^2001) : 
  (∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 2^n + 1 = a * b) ∧ 
  (∃ (c d : ℕ), c > 1 ∧ d > 1 ∧ 2^n - 1 = c * d) := by
sorry


end NUMINAMATH_CALUDE_composite_numbers_l3922_392247


namespace NUMINAMATH_CALUDE_grid_d4_is_5_l3922_392261

/-- Represents a 5x5 grid of numbers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a row contains all different numbers -/
def row_all_different (g : Grid) (r : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g r i ≠ g r j

/-- Checks if a column contains all different numbers -/
def col_all_different (g : Grid) (c : Fin 5) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i c ≠ g j c

/-- Checks if all rows and columns contain different numbers -/
def all_different (g : Grid) : Prop :=
  (∀ r : Fin 5, row_all_different g r) ∧ (∀ c : Fin 5, col_all_different g c)

/-- Checks if the sum of numbers in the 4th column is 9 -/
def fourth_column_sum_9 (g : Grid) : Prop :=
  (g 1 3).val + (g 3 3).val = 9

/-- Checks if the sum of numbers in white cells of row C is 7 -/
def row_c_white_sum_7 (g : Grid) : Prop :=
  (g 2 0).val + (g 2 2).val + (g 2 4).val = 7

/-- Checks if the sum of numbers in white cells of 2nd column is 8 -/
def second_column_white_sum_8 (g : Grid) : Prop :=
  (g 0 1).val + (g 2 1).val + (g 4 1).val = 8

/-- Checks if the sum of numbers in white cells of row B is less than row D -/
def row_b_less_than_row_d (g : Grid) : Prop :=
  (g 1 1).val + (g 1 3).val < (g 3 1).val + (g 3 3).val

theorem grid_d4_is_5 (g : Grid) 
  (h1 : all_different g)
  (h2 : fourth_column_sum_9 g)
  (h3 : row_c_white_sum_7 g)
  (h4 : second_column_white_sum_8 g)
  (h5 : row_b_less_than_row_d g) :
  g 3 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_grid_d4_is_5_l3922_392261


namespace NUMINAMATH_CALUDE_rubber_elongation_improvement_l3922_392275

def n : ℕ := 10

def z_bar : ℝ := 11

def s_squared : ℝ := 61

def significant_improvement (z_bar s_squared : ℝ) : Prop :=
  z_bar ≥ 2 * Real.sqrt (s_squared / n)

theorem rubber_elongation_improvement :
  significant_improvement z_bar s_squared :=
sorry

end NUMINAMATH_CALUDE_rubber_elongation_improvement_l3922_392275


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3922_392248

/-- An isosceles triangle with given altitude and perimeter has area 75 -/
theorem isosceles_triangle_area (base altitudeToBase equalSide : ℝ) : 
  base > 0 → 
  altitudeToBase > 0 → 
  equalSide > 0 → 
  altitudeToBase = 10 → 
  base + 2 * equalSide = 40 → 
  (1/2) * base * altitudeToBase = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3922_392248


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3922_392282

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3922_392282


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l3922_392212

theorem smallest_four_digit_mod_8_5 : ∃ (n : ℕ), 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 8 = 5) → m ≥ n) ∧
  (n = 1005) := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_5_l3922_392212


namespace NUMINAMATH_CALUDE_f_is_cubic_l3922_392238

/-- A polynomial function of degree 4 -/
def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄

/-- The function reaches its maximum at x = -1 -/
def max_at_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x ≤ f a₀ a₁ a₂ a₃ a₄ (-1)

/-- The graph of y = f(x + 1) is symmetric about (-1, 0) -/
def symmetric_about_neg_one (a₀ a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∀ x, f a₀ a₁ a₂ a₃ a₄ (x + 1) = f a₀ a₁ a₂ a₃ a₄ (-x + 1)

theorem f_is_cubic (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  max_at_neg_one a₀ a₁ a₂ a₃ a₄ →
  symmetric_about_neg_one a₀ a₁ a₂ a₃ a₄ →
  ∀ x, f a₀ a₁ a₂ a₃ a₄ x = x^3 - x :=
sorry

end NUMINAMATH_CALUDE_f_is_cubic_l3922_392238


namespace NUMINAMATH_CALUDE_debby_deleted_pictures_l3922_392280

theorem debby_deleted_pictures (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 24)
  (h2 : museum_pics = 12)
  (h3 : remaining_pics = 22) :
  zoo_pics + museum_pics - remaining_pics = 14 := by
  sorry

end NUMINAMATH_CALUDE_debby_deleted_pictures_l3922_392280


namespace NUMINAMATH_CALUDE_total_coins_always_odd_never_equal_coins_l3922_392264

/-- Represents the state of Laura's coins -/
structure CoinState where
  red : Nat
  green : Nat

/-- Represents the slot machine operation -/
def slotMachine (state : CoinState) (insertRed : Bool) : CoinState :=
  if insertRed then
    { red := state.red - 1, green := state.green + 5 }
  else
    { red := state.red + 5, green := state.green - 1 }

/-- The initial state of Laura's coins -/
def initialState : CoinState := { red := 0, green := 1 }

/-- Theorem stating that the total number of coins is always odd -/
theorem total_coins_always_odd (state : CoinState) (n : Nat) :
  (state.red + state.green) % 2 = 1 := by
  sorry

/-- Theorem stating that Laura can never have an equal number of red and green coins -/
theorem never_equal_coins (state : CoinState) :
  state.red ≠ state.green := by
  sorry

end NUMINAMATH_CALUDE_total_coins_always_odd_never_equal_coins_l3922_392264


namespace NUMINAMATH_CALUDE_car_arrival_delay_l3922_392239

/-- Proves that a car traveling 225 km at 50 kmph instead of 60 kmph arrives 45 minutes later -/
theorem car_arrival_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance = 225 →
  speed1 = 60 →
  speed2 = 50 →
  (distance / speed2 - distance / speed1) * 60 = 45 := by
sorry

end NUMINAMATH_CALUDE_car_arrival_delay_l3922_392239


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3922_392241

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3922_392241


namespace NUMINAMATH_CALUDE_stationery_sales_equation_l3922_392205

/-- Represents the sales equation for a stationery store during a promotional event. -/
theorem stationery_sales_equation (x : ℝ) : 
  (1.2 * 0.8 * x + 2 * 0.9 * (60 - x) = 87) ↔ 
  (x ≥ 0 ∧ x ≤ 60 ∧ 
   1.2 * (1 - 0.2) * x + 2 * (1 - 0.1) * (60 - x) = 87) := by
  sorry

#check stationery_sales_equation

end NUMINAMATH_CALUDE_stationery_sales_equation_l3922_392205


namespace NUMINAMATH_CALUDE_mary_marbles_left_l3922_392255

/-- The number of yellow marbles Mary has left after a series of exchanges -/
def marblesLeft (initial : ℝ) (giveJoan : ℝ) (receiveJoan : ℝ) (giveSam : ℝ) : ℝ :=
  initial - giveJoan + receiveJoan - giveSam

/-- Theorem stating that Mary will have 4.7 yellow marbles left -/
theorem mary_marbles_left :
  marblesLeft 9.5 2.3 1.1 3.6 = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_left_l3922_392255


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l3922_392235

/-- The number of five-digit odd numbers -/
def A : ℕ := 9 * 10 * 10 * 10 * 5

/-- The number of five-digit multiples of 5 that are also odd -/
def B : ℕ := 9 * 10 * 10 * 10 * 1

/-- The sum of A and B is equal to 45,000 -/
theorem sum_of_A_and_B : A + B = 45000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l3922_392235


namespace NUMINAMATH_CALUDE_annieka_made_14_throws_l3922_392245

/-- The number of free-throws made by DeShawn -/
def deshawn_throws : ℕ := 12

/-- The number of free-throws made by Kayla -/
def kayla_throws : ℕ := (deshawn_throws * 3) / 2

/-- The number of free-throws made by Annieka -/
def annieka_throws : ℕ := kayla_throws - 4

/-- Theorem: Annieka made 14 free-throws -/
theorem annieka_made_14_throws : annieka_throws = 14 := by
  sorry

end NUMINAMATH_CALUDE_annieka_made_14_throws_l3922_392245


namespace NUMINAMATH_CALUDE_volume_maximized_at_two_l3922_392299

/-- The volume function of the box -/
def volume (x : ℝ) : ℝ := 4 * x * (6 - x)^2

/-- The side length of the original square sheet -/
def original_side : ℝ := 12

/-- Theorem stating that the volume is maximized when x = 2 -/
theorem volume_maximized_at_two :
  ∃ (max_x : ℝ), max_x = 2 ∧
  ∀ (x : ℝ), 0 < x ∧ x < original_side / 2 → volume x ≤ volume max_x :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_two_l3922_392299


namespace NUMINAMATH_CALUDE_exists_dividable_polyhedron_l3922_392221

/-- A face of a polyhedron -/
structure Face where
  -- Add necessary properties of a face

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  -- Add necessary properties to ensure convexity

/-- A function that checks if a set of faces can form a convex polyhedron -/
def can_form_convex_polyhedron (faces : Set Face) : Prop :=
  ∃ (p : ConvexPolyhedron), p.faces = faces

/-- Theorem: There exists a convex polyhedron whose faces can be divided into two sets,
    each of which can form a convex polyhedron -/
theorem exists_dividable_polyhedron :
  ∃ (p : ConvexPolyhedron) (s₁ s₂ : Set Face),
    s₁ ∪ s₂ = p.faces ∧
    s₁ ∩ s₂ = ∅ ∧
    can_form_convex_polyhedron s₁ ∧
    can_form_convex_polyhedron s₂ :=
sorry

end NUMINAMATH_CALUDE_exists_dividable_polyhedron_l3922_392221


namespace NUMINAMATH_CALUDE_function_comparison_l3922_392223

theorem function_comparison (a : ℝ) (h_a : a > 1/2) :
  ∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2,
    (1/2 * a * x₁^2 - (2*a + 1) * x₁ + 21) < (x₂^2 - 2*x₂ + Real.exp x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l3922_392223


namespace NUMINAMATH_CALUDE_remaining_area_calculation_l3922_392243

theorem remaining_area_calculation : 
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_base : ℝ := 1
  let triangle_height : ℝ := 3
  large_square_side ^ 2 - (small_square_side ^ 2 + (triangle_base * triangle_height / 2)) = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_calculation_l3922_392243


namespace NUMINAMATH_CALUDE_inductive_reasoning_not_comparison_l3922_392234

/-- Represents different types of reasoning --/
inductive ReasoningType
| Deductive
| Inductive
| Analogical
| Plausibility

/-- Represents the process of reasoning --/
structure Reasoning where
  type : ReasoningType
  process : String
  conclusion_certainty : Bool

/-- Definition of deductive reasoning --/
def deductive_reasoning : Reasoning :=
  { type := ReasoningType.Deductive,
    process := "from general to specific",
    conclusion_certainty := true }

/-- Definition of inductive reasoning --/
def inductive_reasoning : Reasoning :=
  { type := ReasoningType.Inductive,
    process := "from specific to general",
    conclusion_certainty := false }

/-- Definition of analogical reasoning --/
def analogical_reasoning : Reasoning :=
  { type := ReasoningType.Analogical,
    process := "comparing characteristics of different things",
    conclusion_certainty := false }

/-- Theorem stating that inductive reasoning is not about comparing characteristics of two types of things --/
theorem inductive_reasoning_not_comparison : 
  inductive_reasoning.process ≠ "reasoning between the characteristics of two types of things" := by
  sorry


end NUMINAMATH_CALUDE_inductive_reasoning_not_comparison_l3922_392234


namespace NUMINAMATH_CALUDE_smallest_x_squared_l3922_392236

/-- Represents a trapezoid ABCD with a circle tangent to its sides --/
structure TrapezoidWithCircle where
  AB : ℝ
  CD : ℝ
  x : ℝ
  circle_center_distance : ℝ

/-- The smallest possible value of x in the trapezoid configuration --/
def smallest_x (t : TrapezoidWithCircle) : ℝ := sorry

/-- Main theorem: The square of the smallest possible x is 256 --/
theorem smallest_x_squared (t : TrapezoidWithCircle) 
  (h1 : t.AB = 70)
  (h2 : t.CD = 25)
  (h3 : t.circle_center_distance = 10) :
  (smallest_x t)^2 = 256 := by sorry

end NUMINAMATH_CALUDE_smallest_x_squared_l3922_392236


namespace NUMINAMATH_CALUDE_equation_solution_l3922_392214

theorem equation_solution :
  let f (x : ℂ) := (x^2 + x + 1) / (x + 1)
  let g (x : ℂ) := x^2 + 2*x + 3
  ∀ x : ℂ, f x = g x ↔ x = -2 ∨ x = Complex.I * Real.sqrt 2 ∨ x = -Complex.I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3922_392214


namespace NUMINAMATH_CALUDE_distance_between_points_l3922_392251

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (4, -6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3922_392251


namespace NUMINAMATH_CALUDE_max_value_expression_l3922_392298

/-- The maximum value of (x + y) / z given the conditions -/
theorem max_value_expression (x y z : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ 
  y ≥ 10 ∧ y ≤ 99 ∧ 
  z ≥ 10 ∧ z ≤ 99 ∧ 
  (x + y + z) / 3 = 60 → 
  (x + y : ℚ) / z ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3922_392298


namespace NUMINAMATH_CALUDE_quadratic_function_max_a_l3922_392289

theorem quadratic_function_max_a (a b c m : ℝ) : 
  a < 0 →
  a * m^2 + b * m + c = b →
  a * (m + 1)^2 + b * (m + 1) + c = a →
  b ≥ a →
  m < 0 →
  (∀ x, a * x^2 + b * x + c ≤ -2) →
  (∀ a', a' < 0 → 
    (∀ x, a' * x^2 + (-a' * m) * x + (-a' * m) ≤ -2) → 
    a' ≤ a) →
  a = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_a_l3922_392289


namespace NUMINAMATH_CALUDE_special_number_property_l3922_392209

/-- The greatest integer less than 100 for which the greatest common factor with 18 is 3 -/
def special_number : ℕ := 93

/-- Theorem stating that special_number satisfies the required conditions -/
theorem special_number_property : 
  special_number < 100 ∧ 
  Nat.gcd special_number 18 = 3 ∧ 
  ∀ n : ℕ, n < 100 → Nat.gcd n 18 = 3 → n ≤ special_number := by
  sorry

end NUMINAMATH_CALUDE_special_number_property_l3922_392209


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l3922_392203

theorem inscribed_circles_area_ratio : 
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (∃ (s : ℝ), s > 0 ∧ R = (s * Real.sqrt 2) / 2 ∧ r = s / 2) →
  (π * r^2) / (π * R^2) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l3922_392203


namespace NUMINAMATH_CALUDE_combined_new_weight_theorem_l3922_392294

/-- Calculates the new weight of fruit after water loss -/
def new_weight (initial_weight : ℝ) (initial_water_percent : ℝ) (evaporation_loss : ℝ) (skin_loss : ℝ) : ℝ :=
  let initial_water := initial_weight * initial_water_percent
  let pulp := initial_weight - initial_water
  let water_loss := initial_water * (evaporation_loss + skin_loss)
  let new_water := initial_water - water_loss
  pulp + new_water

/-- The combined new weight of oranges and apples after water loss -/
theorem combined_new_weight_theorem (orange_weight : ℝ) (apple_weight : ℝ) 
    (orange_water_percent : ℝ) (apple_water_percent : ℝ)
    (orange_evaporation_loss : ℝ) (orange_skin_loss : ℝ)
    (apple_evaporation_loss : ℝ) (apple_skin_loss : ℝ) :
  orange_weight = 5 →
  apple_weight = 3 →
  orange_water_percent = 0.95 →
  apple_water_percent = 0.90 →
  orange_evaporation_loss = 0.05 →
  orange_skin_loss = 0.02 →
  apple_evaporation_loss = 0.03 →
  apple_skin_loss = 0.01 →
  (new_weight orange_weight orange_water_percent orange_evaporation_loss orange_skin_loss +
   new_weight apple_weight apple_water_percent apple_evaporation_loss apple_skin_loss) = 7.5595 := by
  sorry

end NUMINAMATH_CALUDE_combined_new_weight_theorem_l3922_392294


namespace NUMINAMATH_CALUDE_ce_length_l3922_392242

/-- Given a triangle ABC, this function returns true if the triangle is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Given three points A, B, C, this function returns the measure of angle ABC in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given two points A and B, this function returns the distance between them -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem ce_length (A B C D E : ℝ × ℝ) 
  (h1 : is_right_triangle A B E)
  (h2 : is_right_triangle B C E)
  (h3 : is_right_triangle C D E)
  (h4 : angle_measure A E B = 60)
  (h5 : angle_measure B E C = 60)
  (h6 : angle_measure C E D = 60)
  (h7 : distance A E = 36) :
  distance C E = 9 := by sorry

end NUMINAMATH_CALUDE_ce_length_l3922_392242


namespace NUMINAMATH_CALUDE_min_common_perimeter_of_specific_isosceles_triangles_l3922_392202

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Checks if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_common_perimeter_of_specific_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 1180 :=
  sorry

end NUMINAMATH_CALUDE_min_common_perimeter_of_specific_isosceles_triangles_l3922_392202


namespace NUMINAMATH_CALUDE_three_propositions_l3922_392252

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_l3922_392252


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l3922_392215

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base octagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Main theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_areas : cs1.area = 300 * Real.sqrt 2 ∧ cs2.area = 675 * Real.sqrt 2)
  (h_distance : |cs1.distance_from_apex - cs2.distance_from_apex| = 10)
  (h_order : cs1.area < cs2.area) :
  cs2.distance_from_apex = 30 := by
sorry

end NUMINAMATH_CALUDE_larger_cross_section_distance_l3922_392215


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3922_392219

-- Define the inequality system
def inequality_system (a b x : ℝ) : Prop :=
  x - a > 2 ∧ x + 1 < b

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -1 < x ∧ x < 1

-- Theorem statement
theorem inequality_system_solution (a b : ℝ) :
  (∀ x, inequality_system a b x ↔ solution_set x) →
  (a + b)^2023 = -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3922_392219


namespace NUMINAMATH_CALUDE_g_of_5_equals_15_l3922_392271

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5_equals_15 : g 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_15_l3922_392271


namespace NUMINAMATH_CALUDE_perpendicular_tangents_sum_l3922_392224

/-- The problem statement -/
theorem perpendicular_tangents_sum (a b : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    -- Point (x₀, y₀) is on both curves
    (y₀ = x₀^2 - 2*x₀ + 2 ∧ y₀ = -x₀^2 + a*x₀ + b) ∧ 
    -- Tangents are perpendicular
    (2*x₀ - 2) * (-2*x₀ + a) = -1) → 
  a + b = 5/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_sum_l3922_392224


namespace NUMINAMATH_CALUDE_sports_day_theorem_l3922_392273

/-- Represents the score awarded to a class in a single event -/
structure EventScore where
  first : ℕ
  second : ℕ
  third : ℕ
  first_gt_second : first > second
  second_gt_third : second > third

/-- Represents the total scores of all classes -/
structure TotalScores where
  scores : List ℕ
  four_classes : scores.length = 4

/-- The Sports Day competition setup -/
structure SportsDay where
  event_score : EventScore
  total_scores : TotalScores
  events_count : ℕ
  events_count_eq_five : events_count = 5
  scores_sum_eq_events_total : total_scores.scores.sum = events_count * (event_score.first + event_score.second + event_score.third)

theorem sports_day_theorem (sd : SportsDay) 
  (h_scores : sd.total_scores.scores = [21, 6, 9, 4]) : 
  sd.event_score.first + sd.event_score.second + sd.event_score.third = 8 ∧ 
  sd.event_score.first = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_day_theorem_l3922_392273


namespace NUMINAMATH_CALUDE_class_b_more_consistent_l3922_392266

/-- Represents the variance of a class's test scores -/
structure ClassVariance where
  value : ℝ
  is_nonneg : value ≥ 0

/-- Determines if one class has more consistent scores than another based on their variances -/
def has_more_consistent_scores (class_a class_b : ClassVariance) : Prop :=
  class_a.value > class_b.value

theorem class_b_more_consistent :
  let class_a : ClassVariance := ⟨2.56, by norm_num⟩
  let class_b : ClassVariance := ⟨1.92, by norm_num⟩
  has_more_consistent_scores class_b class_a := by
  sorry

end NUMINAMATH_CALUDE_class_b_more_consistent_l3922_392266


namespace NUMINAMATH_CALUDE_octagon_arc_length_l3922_392283

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (s : ℝ) (h : s = 4) :
  let R := s / (2 * Real.sin (π / 8))
  let C := 2 * π * R
  C / 8 = (Real.sqrt 2 * π) / 2 := by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l3922_392283


namespace NUMINAMATH_CALUDE_simplify_expression_l3922_392253

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 6) - (2*x + 7)*(3*x - 2) = -7*x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3922_392253


namespace NUMINAMATH_CALUDE_victor_remaining_lives_l3922_392213

def calculate_lives_remaining (initial_lives : ℕ) 
                               (first_level_loss : ℕ) 
                               (second_level_gain_rate : ℕ) 
                               (second_level_duration : ℕ) 
                               (third_level_loss_rate : ℕ) 
                               (third_level_duration : ℕ) : ℕ :=
  let lives_after_first := initial_lives - first_level_loss
  let second_level_intervals := second_level_duration / 45
  let lives_after_second := lives_after_first + second_level_gain_rate * second_level_intervals
  let third_level_intervals := third_level_duration / 20
  lives_after_second - third_level_loss_rate * third_level_intervals

theorem victor_remaining_lives : 
  calculate_lives_remaining 246 14 3 135 4 80 = 225 := by
  sorry

end NUMINAMATH_CALUDE_victor_remaining_lives_l3922_392213


namespace NUMINAMATH_CALUDE_degree_to_radian_90_l3922_392256

theorem degree_to_radian_90 : 
  (90 : ℝ) * (π / 180) = π / 2 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_90_l3922_392256


namespace NUMINAMATH_CALUDE_line_slope_is_two_l3922_392237

/-- A line in the xy-plane with y-intercept 2 and passing through (239, 480) has slope 2 -/
theorem line_slope_is_two :
  ∀ (m : ℝ) (f : ℝ → ℝ),
  (∀ x, f x = m * x + 2) →  -- Line equation with y-intercept 2
  f 239 = 480 →            -- Line passes through (239, 480)
  m = 2 :=                 -- Slope is 2
by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l3922_392237


namespace NUMINAMATH_CALUDE_eggs_per_basket_l3922_392265

theorem eggs_per_basket (yellow_eggs : Nat) (pink_eggs : Nat) (min_eggs : Nat) : 
  yellow_eggs = 30 → pink_eggs = 45 → min_eggs = 5 → 
  ∃ (eggs_per_basket : Nat), 
    eggs_per_basket ∣ yellow_eggs ∧ 
    eggs_per_basket ∣ pink_eggs ∧ 
    eggs_per_basket ≥ min_eggs ∧
    ∀ (n : Nat), n ∣ yellow_eggs → n ∣ pink_eggs → n ≥ min_eggs → n ≤ eggs_per_basket :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l3922_392265


namespace NUMINAMATH_CALUDE_regular_polygon_15_diagonals_l3922_392287

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 15 diagonals has 7 sides -/
theorem regular_polygon_15_diagonals :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 15 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_15_diagonals_l3922_392287


namespace NUMINAMATH_CALUDE_pens_distribution_eq_six_l3922_392227

/-- The number of ways to distribute n identical items among k distinct groups,
    where each group gets at least m items. -/
def distribute_with_minimum (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The number of ways to distribute 8 pens among 3 friends,
    where each friend gets at least 2 pens. -/
def pens_distribution : ℕ :=
  distribute_with_minimum 8 3 2

theorem pens_distribution_eq_six :
  pens_distribution = 6 := by
  sorry

end NUMINAMATH_CALUDE_pens_distribution_eq_six_l3922_392227


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3922_392220

/-- Proves that the principal amount is 20000 given the specified conditions --/
theorem compound_interest_problem (P : ℝ) : 
  P * (1 + 0.2 / 2)^4 - P * (1 + 0.2)^2 = 482 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3922_392220


namespace NUMINAMATH_CALUDE_print_shop_pricing_l3922_392277

/-- The price per color copy at print shop Y -/
def price_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X for 40 copies -/
def additional_charge : ℚ := 60

/-- The price per color copy at print shop X -/
def price_X : ℚ := 1.25

theorem print_shop_pricing :
  price_Y * num_copies = price_X * num_copies + additional_charge :=
sorry

end NUMINAMATH_CALUDE_print_shop_pricing_l3922_392277


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3922_392290

theorem perfect_square_trinomial_condition (a : ℝ) :
  (∃ b c : ℝ, ∀ x y : ℝ, 4*x^2 - (a-1)*x*y + 9*y^2 = (b*x + c*y)^2) →
  (a = 13 ∨ a = -11) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3922_392290


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_l3922_392254

theorem sum_of_squared_differences : (302^2 - 298^2) + (152^2 - 148^2) = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_l3922_392254


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3922_392270

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 0 →
    exterior_angle = 20 →
    360 / exterior_angle = n →
    (n - 2) * 180 = 2880 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3922_392270


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3922_392218

theorem complex_equation_solution (z : ℂ) :
  (-3 + 4 * Complex.I) * z = 25 * Complex.I → z = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3922_392218


namespace NUMINAMATH_CALUDE_games_played_calculation_l3922_392246

/-- Represents the gambler's poker game statistics -/
structure GamblerStats where
  gamesPlayed : ℝ
  initialWinRate : ℝ
  newWinRate : ℝ
  targetWinRate : ℝ
  additionalGames : ℝ

/-- Theorem stating the number of games played given the conditions -/
theorem games_played_calculation (stats : GamblerStats)
  (h1 : stats.initialWinRate = 0.4)
  (h2 : stats.newWinRate = 0.8)
  (h3 : stats.targetWinRate = 0.6)
  (h4 : stats.additionalGames = 19.999999999999993)
  (h5 : stats.initialWinRate * stats.gamesPlayed + stats.newWinRate * stats.additionalGames = 
        stats.targetWinRate * (stats.gamesPlayed + stats.additionalGames)) :
  stats.gamesPlayed = 20 := by
  sorry

end NUMINAMATH_CALUDE_games_played_calculation_l3922_392246


namespace NUMINAMATH_CALUDE_problem_solution_l3922_392292

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 25) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 82.1762 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3922_392292


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3922_392262

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 8 * X^3 - 35 * X^2 - 45 * X + 52 = 
  (X^2 + 5 * X - 3) * q + (-21 * X + 79) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3922_392262


namespace NUMINAMATH_CALUDE_transformed_area_is_63_l3922_392279

/-- The transformation matrix --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 1, -1]

/-- The original region's area --/
def original_area : ℝ := 9

/-- Theorem stating the area of the transformed region --/
theorem transformed_area_is_63 : 
  |A.det| * original_area = 63 := by sorry

end NUMINAMATH_CALUDE_transformed_area_is_63_l3922_392279


namespace NUMINAMATH_CALUDE_parking_spots_fourth_level_l3922_392276

theorem parking_spots_fourth_level 
  (total_levels : Nat) 
  (first_level_spots : Nat) 
  (second_level_diff : Nat) 
  (third_level_diff : Nat) 
  (total_spots : Nat) :
  total_levels = 4 →
  first_level_spots = 4 →
  second_level_diff = 7 →
  third_level_diff = 6 →
  total_spots = 46 →
  let second_level_spots := first_level_spots + second_level_diff
  let third_level_spots := second_level_spots + third_level_diff
  let fourth_level_spots := total_spots - (first_level_spots + second_level_spots + third_level_spots)
  fourth_level_spots = 14 := by
sorry

end NUMINAMATH_CALUDE_parking_spots_fourth_level_l3922_392276


namespace NUMINAMATH_CALUDE_total_households_l3922_392240

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  both : ℕ
  gasOnly : ℕ
  elecOnly : ℕ
  neither : ℕ

/-- The conditions of the survey -/
def surveyCounts : HouseholdCounts where
  both := 120
  gasOnly := 60
  elecOnly := 4 * 24
  neither := 24

/-- The theorem stating the total number of households surveyed -/
theorem total_households : 
  surveyCounts.both + surveyCounts.gasOnly + surveyCounts.elecOnly + surveyCounts.neither = 300 := by
  sorry


end NUMINAMATH_CALUDE_total_households_l3922_392240


namespace NUMINAMATH_CALUDE_periodic_function_extension_l3922_392284

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_smallest_period : ∀ p, 0 < p → p < 2 → ¬ is_periodic f p)
  (h_def : ∀ x, 0 ≤ x → x < 2 → f x = x^3 - x) :
  ∀ x, -2 ≤ x → x < 0 → f x = x^3 + 6*x^2 + 11*x + 6 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_extension_l3922_392284


namespace NUMINAMATH_CALUDE_min_difference_f_g_l3922_392258

noncomputable def f (x : ℝ) := Real.exp x

noncomputable def g (x : ℝ) := Real.log (x / 2) + 1 / 2

theorem min_difference_f_g :
  ∀ a : ℝ, ∃ b : ℝ, b > 0 ∧ f a = g b ∧
  (∀ c : ℝ, c > 0 ∧ f a = g c → b - a ≤ c - a) ∧
  b - a = 2 + Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_difference_f_g_l3922_392258


namespace NUMINAMATH_CALUDE_pet_shop_inventory_l3922_392217

/-- Represents the pet shop inventory problem --/
theorem pet_shop_inventory (num_kittens : ℕ) (puppy_cost kitten_cost total_value : ℕ) :
  num_kittens = 4 →
  puppy_cost = 20 →
  kitten_cost = 15 →
  total_value = 100 →
  ∃ (num_puppies : ℕ), num_puppies = 2 ∧ num_puppies * puppy_cost + num_kittens * kitten_cost = total_value :=
by
  sorry

end NUMINAMATH_CALUDE_pet_shop_inventory_l3922_392217


namespace NUMINAMATH_CALUDE_opposite_numbers_theorem_l3922_392274

theorem opposite_numbers_theorem (a b c d : ℤ) : 
  (a + b = 0) → 
  (c = -1) → 
  (d = 1 ∨ d = -1) → 
  (2*a + 2*b - c*d = 1 ∨ 2*a + 2*b - c*d = -1) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_theorem_l3922_392274


namespace NUMINAMATH_CALUDE_purchase_plan_monthly_payment_l3922_392225

theorem purchase_plan_monthly_payment 
  (purchase_price : ℝ) 
  (down_payment : ℝ) 
  (num_payments : ℕ) 
  (interest_rate : ℝ) 
  (h1 : purchase_price = 118)
  (h2 : down_payment = 18)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 0.15254237288135593) :
  let total_interest : ℝ := purchase_price * interest_rate
  let total_paid : ℝ := purchase_price + total_interest
  let monthly_payment : ℝ := (total_paid - down_payment) / num_payments
  monthly_payment = 9.833333333333334 := by sorry

end NUMINAMATH_CALUDE_purchase_plan_monthly_payment_l3922_392225


namespace NUMINAMATH_CALUDE_sixth_term_value_l3922_392260

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the roots of the quadratic equation
def roots_of_equation (a : ℕ → ℝ) : Prop :=
  3 * (a 3)^2 - 11 * (a 3) + 9 = 0 ∧ 3 * (a 9)^2 - 11 * (a 9) + 9 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) :
  geometric_sequence a → roots_of_equation a → (a 6)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_value_l3922_392260


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l3922_392204

theorem quadruple_equation_solutions :
  {q : ℕ × ℕ × ℕ × ℕ | let (x, y, z, n) := q; x^2 + y^2 + z^2 + 1 = 2^n} =
  {(0,0,0,0), (1,0,0,1), (0,1,0,1), (0,0,1,1), (1,1,1,2)} := by
  sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l3922_392204


namespace NUMINAMATH_CALUDE_second_discount_percentage_prove_discount_percentage_l3922_392207

/-- Calculates the second discount percentage given the original price, first discount percentage, and final price --/
theorem second_discount_percentage 
  (original_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_decimal := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_decimal * 100

/-- Proves that the second discount percentage is approximately 2% given the problem conditions --/
theorem prove_discount_percentage : 
  let original_price : ℝ := 65
  let first_discount_percent : ℝ := 10
  let final_price : ℝ := 57.33
  let result := second_discount_percentage original_price first_discount_percent final_price
  abs (result - 2) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_prove_discount_percentage_l3922_392207


namespace NUMINAMATH_CALUDE_range_of_even_quadratic_l3922_392269

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the property of being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem range_of_even_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*b) (3*b - 1), f a b x ∈ Set.Icc 1 5) ∧
  is_even (f a b) →
  Set.range (f a b) = Set.Icc 1 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_even_quadratic_l3922_392269


namespace NUMINAMATH_CALUDE_cow_count_is_six_l3922_392288

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Theorem: If the total number of legs is 12 more than twice the number of heads,
    then the number of cows is 6 -/
theorem cow_count_is_six (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 12 → g.cows = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_six_l3922_392288


namespace NUMINAMATH_CALUDE_x_value_l3922_392228

theorem x_value :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 90 →
    x = 137 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3922_392228


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l3922_392297

theorem chord_length_concentric_circles (area_ring : ℝ) (r_small : ℝ) (r_large : ℝ) (chord_length : ℝ) : 
  area_ring = 18.75 * Real.pi ∧ 
  r_large = 2 * r_small ∧ 
  area_ring = Real.pi * (r_large^2 - r_small^2) ∧
  chord_length^2 = 4 * (r_large^2 - r_small^2) →
  chord_length = Real.sqrt 75 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l3922_392297


namespace NUMINAMATH_CALUDE_largest_decimal_l3922_392293

def circle_digits : List Nat := [1, 3, 9, 5, 7, 9, 1, 3, 9, 5, 7, 1]

def is_valid_decimal (d : ℚ) : Prop :=
  ∃ (n : ℕ) (l : List ℕ),
    l.length = 6 ∧
    l.all (λ x => x ∈ circle_digits) ∧
    d = n + (l.foldl (λ acc x => (acc + x) / 10) 0 : ℚ)

def is_largest_decimal (d : ℚ) : Prop :=
  is_valid_decimal d ∧
  ∀ d', is_valid_decimal d' → d' ≤ d

theorem largest_decimal :
  is_largest_decimal (9 + 579139 / 1000000 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_largest_decimal_l3922_392293


namespace NUMINAMATH_CALUDE_range_of_a_l3922_392257

/-- Custom multiplication operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3922_392257


namespace NUMINAMATH_CALUDE_sum_probability_is_thirteen_sixteenths_l3922_392249

/-- Represents an n-sided die -/
structure Die (n : ℕ) where
  sides : Fin n → ℕ
  valid : ∀ i, sides i ∈ Finset.range n.succ

/-- The 8-sided die -/
def eight_sided_die : Die 8 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The 6-sided die -/
def six_sided_die : Die 6 :=
  { sides := λ i => i.val + 1,
    valid := by sorry }

/-- The set of all possible outcomes when rolling two dice -/
def outcomes : Finset (Fin 8 × Fin 6) :=
  Finset.product (Finset.univ : Finset (Fin 8)) (Finset.univ : Finset (Fin 6))

/-- The set of favorable outcomes (sum ≤ 10) -/
def favorable_outcomes : Finset (Fin 8 × Fin 6) :=
  outcomes.filter (λ p => eight_sided_die.sides p.1 + six_sided_die.sides p.2 ≤ 10)

/-- The probability of the sum being less than or equal to 10 -/
def probability : ℚ :=
  favorable_outcomes.card / outcomes.card

theorem sum_probability_is_thirteen_sixteenths :
  probability = 13 / 16 := by sorry

end NUMINAMATH_CALUDE_sum_probability_is_thirteen_sixteenths_l3922_392249


namespace NUMINAMATH_CALUDE_negative_five_times_three_l3922_392268

theorem negative_five_times_three : (-5 : ℤ) * 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_times_three_l3922_392268


namespace NUMINAMATH_CALUDE_returning_players_count_l3922_392232

/-- The number of returning players in a baseball team -/
def returning_players (new_players : ℕ) (group_size : ℕ) (num_groups : ℕ) : ℕ :=
  group_size * num_groups - new_players

/-- Theorem stating the number of returning players in the given scenario -/
theorem returning_players_count : returning_players 4 5 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_returning_players_count_l3922_392232


namespace NUMINAMATH_CALUDE_gcd_property_l3922_392208

theorem gcd_property (a b c : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd a.natAbs (b * c).natAbs = Nat.gcd a.natAbs c.natAbs := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l3922_392208


namespace NUMINAMATH_CALUDE_star_operation_result_l3922_392222

-- Define the sets M and N
def M : Set ℝ := {y | y ≥ 0}
def N : Set ℝ := {y | -3 ≤ y ∧ y ≤ 3}

-- Define the set difference operation
def setDifference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Define the * operation
def starOperation (A B : Set ℝ) : Set ℝ := (setDifference A B) ∪ (setDifference B A)

-- State the theorem
theorem star_operation_result :
  starOperation M N = {x : ℝ | -3 ≤ x ∧ x < 0 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_star_operation_result_l3922_392222


namespace NUMINAMATH_CALUDE_binomial_sum_distinct_values_l3922_392285

theorem binomial_sum_distinct_values :
  ∃ (S : Finset ℕ), (∀ r : ℤ, 7 ≤ r ∧ r ≤ 9 →
    (Nat.choose 10 (r.toNat + 1) + Nat.choose 10 (17 - r.toNat)) ∈ S) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_distinct_values_l3922_392285


namespace NUMINAMATH_CALUDE_travel_distance_calculation_l3922_392200

theorem travel_distance_calculation (total_distance sea_distance : ℕ) 
  (h1 : total_distance = 601)
  (h2 : sea_distance = 150) :
  total_distance - sea_distance = 451 :=
by sorry

end NUMINAMATH_CALUDE_travel_distance_calculation_l3922_392200


namespace NUMINAMATH_CALUDE_annas_pencils_l3922_392291

theorem annas_pencils (anna_pencils : ℕ) (harry_pencils : ℕ) : 
  (harry_pencils = 2 * anna_pencils) → -- Harry has twice Anna's pencils initially
  (harry_pencils - 19 = 81) → -- Harry lost 19 pencils and now has 81 left
  anna_pencils = 50 := by
sorry

end NUMINAMATH_CALUDE_annas_pencils_l3922_392291


namespace NUMINAMATH_CALUDE_cloth_cost_unchanged_l3922_392272

/-- Represents the scenario of a cloth purchase with changing length and price --/
structure ClothPurchase where
  originalCost : ℝ  -- Total cost in rupees
  originalLength : ℝ  -- Length in meters
  lengthIncrease : ℝ  -- Increase in length in meters
  priceDecrease : ℝ  -- Decrease in price per meter in rupees

/-- The total cost remains unchanged after increasing length and decreasing price --/
def costUnchanged (cp : ClothPurchase) : Prop :=
  cp.originalCost = (cp.originalLength + cp.lengthIncrease) * 
    ((cp.originalCost / cp.originalLength) - cp.priceDecrease)

/-- Theorem stating that for the given conditions, the cost remains unchanged when length increases by 4 meters --/
theorem cloth_cost_unchanged : 
  ∃ (cp : ClothPurchase), 
    cp.originalCost = 35 ∧ 
    cp.originalLength = 10 ∧ 
    cp.priceDecrease = 1 ∧ 
    cp.lengthIncrease = 4 ∧ 
    costUnchanged cp := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_unchanged_l3922_392272


namespace NUMINAMATH_CALUDE_prime_octuple_sum_product_relation_l3922_392231

theorem prime_octuple_sum_product_relation :
  ∀ (p₁ p₂ p₃ p₄ p₅ p₆ p₇ p₈ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
    Prime p₅ ∧ Prime p₆ ∧ Prime p₇ ∧ Prime p₈ →
    (p₁^2 + p₂^2 + p₃^2 + p₄^2 + p₅^2 + p₆^2 + p₇^2 + p₈^2 = 4 * (p₁ * p₂ * p₃ * p₄ * p₅ * p₆ * p₇ * p₈) - 992) →
    p₁ = 2 ∧ p₂ = 2 ∧ p₃ = 2 ∧ p₄ = 2 ∧ p₅ = 2 ∧ p₆ = 2 ∧ p₇ = 2 ∧ p₈ = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_prime_octuple_sum_product_relation_l3922_392231


namespace NUMINAMATH_CALUDE_markup_percentage_l3922_392230

/-- Proves that given a cost price of 540, a selling price of 459, and a discount percentage
    of 26.08695652173913%, the percentage marked above the cost price is 15%. -/
theorem markup_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (discount_percentage : ℝ)
  (h_cost_price : cost_price = 540)
  (h_selling_price : selling_price = 459)
  (h_discount_percentage : discount_percentage = 26.08695652173913) :
  let marked_price := selling_price / (1 - discount_percentage / 100)
  (marked_price - cost_price) / cost_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_l3922_392230


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3922_392210

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 5) 
  (eq2 : x + 2*y = 6) : 
  7*x^2 + 10*x*y + 7*y^2 = 85 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3922_392210


namespace NUMINAMATH_CALUDE_smallest_number_minus_three_divisible_by_fifteen_l3922_392244

theorem smallest_number_minus_three_divisible_by_fifteen : 
  ∃ N : ℕ, (N ≥ 18) ∧ (N - 3) % 15 = 0 ∧ ∀ M : ℕ, M < N → (M - 3) % 15 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_minus_three_divisible_by_fifteen_l3922_392244


namespace NUMINAMATH_CALUDE_simplify_fraction_l3922_392286

theorem simplify_fraction : 5 * (18 / 7) * (21 / -54) = -5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3922_392286


namespace NUMINAMATH_CALUDE_fourth_number_unit_digit_l3922_392296

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit 
  (a b c : ℕ) 
  (ha : a = 7858) 
  (hb : b = 1086) 
  (hc : c = 4582) : 
  ∃ d : ℕ, unit_digit (a * b * c * d) = 8 ↔ unit_digit d = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_unit_digit_l3922_392296


namespace NUMINAMATH_CALUDE_pascal_theorem_l3922_392278

-- Define the conic section
structure ConicSection where
  -- Add necessary fields to define a conic section
  -- This is a placeholder and should be replaced with actual definition

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ -- ax + by + c = 0

-- Define the hexagon
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

-- Function to check if a point lies on a conic section
def pointOnConic (p : Point) (c : ConicSection) : Prop :=
  sorry -- Define the condition for a point to lie on the conic section

-- Function to check if two lines intersect at a point
def linesIntersectAt (l1 l2 : Line) (p : Point) : Prop :=
  sorry -- Define the condition for two lines to intersect at a given point

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Define the condition for three points to be collinear

-- Theorem statement
theorem pascal_theorem (c : ConicSection) (h : Hexagon) 
  (hInscribed : pointOnConic h.A c ∧ pointOnConic h.B c ∧ pointOnConic h.C c ∧ 
                pointOnConic h.D c ∧ pointOnConic h.E c ∧ pointOnConic h.F c)
  (M N P : Point)
  (hM : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) M) -- AB and DE
  (hN : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) N) -- BC and EF
  (hP : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) P) -- CD and FA
  : areCollinear M N P :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_pascal_theorem_l3922_392278


namespace NUMINAMATH_CALUDE_profit_for_450_pieces_l3922_392211

/-- The price function for the clothing factory -/
def price (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else 62 - x / 50

/-- The profit function for the clothing factory -/
def profit (x : ℕ) : ℚ :=
  (price x - 40) * x

/-- The theorem stating the profit for an order of 450 pieces -/
theorem profit_for_450_pieces :
  0 < 450 ∧ 450 ≤ 500 → profit 450 = 5850 := by sorry

end NUMINAMATH_CALUDE_profit_for_450_pieces_l3922_392211


namespace NUMINAMATH_CALUDE_ninas_pet_eyes_l3922_392281

/-- The total number of eyes among Nina's pet insects -/
theorem ninas_pet_eyes : 
  let spider_count : ℕ := 3
  let ant_count : ℕ := 50
  let eyes_per_spider : ℕ := 8
  let eyes_per_ant : ℕ := 2
  let total_eyes : ℕ := spider_count * eyes_per_spider + ant_count * eyes_per_ant
  total_eyes = 124 := by sorry

end NUMINAMATH_CALUDE_ninas_pet_eyes_l3922_392281


namespace NUMINAMATH_CALUDE_lottery_is_systematic_sampling_l3922_392206

-- Define the lottery range
def lottery_range : Set ℕ := {n | 0 ≤ n ∧ n < 100000}

-- Define the winning number criteria
def is_winning_number (n : ℕ) : Prop :=
  n ∈ lottery_range ∧ (n % 100 = 88 ∨ n % 100 = 68)

-- Define systematic sampling
def systematic_sampling (S : Set ℕ) (f : ℕ → Prop) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n ∈ S, f n ↔ ∃ m : ℕ, n = m * k

-- Theorem statement
theorem lottery_is_systematic_sampling :
  systematic_sampling lottery_range is_winning_number := by
  sorry


end NUMINAMATH_CALUDE_lottery_is_systematic_sampling_l3922_392206


namespace NUMINAMATH_CALUDE_unique_triplet_l3922_392250

theorem unique_triplet : 
  ∃! (a b c : ℕ), 2 ≤ a ∧ a < b ∧ b < c ∧ 
  (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) ∧
  a = 4 ∧ b = 5 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_triplet_l3922_392250


namespace NUMINAMATH_CALUDE_square_of_integer_proof_l3922_392263

theorem square_of_integer_proof (n : ℕ+) (h : ∃ (k : ℤ), k^2 = 1 + 12 * (n : ℤ)^2) :
  ∃ (m : ℤ), (2 : ℤ) + 2 * Int.sqrt (1 + 12 * (n : ℤ)^2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_proof_l3922_392263


namespace NUMINAMATH_CALUDE_maurice_rides_before_is_10_l3922_392216

/-- The number of times Maurice had been horseback riding before visiting Matt -/
def maurice_rides_before : ℕ := 10

/-- The number of different horses Maurice rode before his visit -/
def maurice_horses_before : ℕ := 2

/-- The number of different horses Matt has ridden -/
def matt_horses : ℕ := 4

/-- The number of times Maurice rode during his visit -/
def maurice_rides_visit : ℕ := 8

/-- The number of additional times Matt rode on his other horses -/
def matt_additional_rides : ℕ := 16

/-- The number of horses Matt rode each time with Maurice -/
def matt_horses_per_ride : ℕ := 2

theorem maurice_rides_before_is_10 :
  maurice_rides_before = 10 ∧
  maurice_horses_before = 2 ∧
  matt_horses = 4 ∧
  maurice_rides_visit = 8 ∧
  matt_additional_rides = 16 ∧
  matt_horses_per_ride = 2 ∧
  maurice_rides_visit = maurice_rides_before ∧
  (maurice_rides_visit * matt_horses_per_ride + matt_additional_rides) = 3 * maurice_rides_before :=
by sorry

end NUMINAMATH_CALUDE_maurice_rides_before_is_10_l3922_392216
