import Mathlib

namespace NUMINAMATH_CALUDE_gifted_subscribers_l3531_353104

/-- Calculates the number of gifted subscribers for a Twitch streamer --/
theorem gifted_subscribers
  (initial_subscribers : ℕ)
  (income_per_subscriber : ℕ)
  (current_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : income_per_subscriber = 9)
  (h3 : current_monthly_income = 1800) :
  current_monthly_income / income_per_subscriber - initial_subscribers = 50 :=
by sorry

end NUMINAMATH_CALUDE_gifted_subscribers_l3531_353104


namespace NUMINAMATH_CALUDE_volume_removed_percentage_l3531_353139

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Theorem: The percentage of volume removed from a box with dimensions 20x15x10,
    by removing a 4cm cube from each of its 8 corners, is equal to (512/3000) * 100% -/
theorem volume_removed_percentage :
  let originalBox : BoxDimensions := ⟨20, 15, 10⟩
  let removedCubeSide : ℝ := 4
  let numCorners : ℕ := 8
  let originalVolume := boxVolume originalBox
  let removedVolume := numCorners * (cubeVolume removedCubeSide)
  (removedVolume / originalVolume) * 100 = (512 / 3000) * 100 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_percentage_l3531_353139


namespace NUMINAMATH_CALUDE_f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l3531_353185

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 8

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4 * a * x^2 - 12 * a^2 * x + 3 * a^3 - 8

-- Theorem 1: f(x) < 0 for all x ∈ [1, 2] iff a > 10
theorem f_negative_iff_a_greater_than_ten (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x < 0) ↔ a > 10 := by sorry

-- Theorem 2: No integer a exists such that g(x) has a local minimum in (0, 1)
theorem no_integer_a_for_g_local_minimum :
  ¬ ∃ a : ℤ, ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (g (a : ℝ)) x := by sorry

end NUMINAMATH_CALUDE_f_negative_iff_a_greater_than_ten_no_integer_a_for_g_local_minimum_l3531_353185


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3531_353193

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2) 
  (h_x_pos : x > 0) 
  (h_cos : Real.cos (θ / 3) = Real.sqrt ((x + 2) / (3 * x))) : 
  Real.tan θ = 
    (Real.sqrt (1 - ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) ^ 2)) / 
    ((4 * (x + 2) ^ (3/2) - 3 * Real.sqrt (3 * x) * Real.sqrt (x + 2)) / 
      (3 * Real.sqrt (3 * x ^ 3))) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3531_353193


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l3531_353111

theorem bus_seat_capacity (left_seats right_seats back_seat_capacity total_capacity : ℕ) 
  (h1 : left_seats = 15)
  (h2 : right_seats = left_seats - 3)
  (h3 : back_seat_capacity = 8)
  (h4 : total_capacity = 89) :
  ∃ (seat_capacity : ℕ), 
    seat_capacity * (left_seats + right_seats) + back_seat_capacity = total_capacity ∧ 
    seat_capacity = 3 := by
sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l3531_353111


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l3531_353122

theorem arithmetic_mean_of_integers (n : ℕ) (start : ℕ) :
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  n ≠ 0 → sum / n = (2 * start + n - 1) / 2 := by
  sorry

theorem arithmetic_mean_of_52_integers_from_2 :
  let n := 52
  let start := 2
  let seq := fun i => start + i - 1
  let sum := (n * (2 * start + n - 1)) / 2
  sum / n = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_integers_arithmetic_mean_of_52_integers_from_2_l3531_353122


namespace NUMINAMATH_CALUDE_max_value_of_2xy_l3531_353129

theorem max_value_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 2 * x * y ≤ 2 * a * b → 2 * x * y ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2xy_l3531_353129


namespace NUMINAMATH_CALUDE_shirt_cost_l3531_353190

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 76) →
  shirt_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3531_353190


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l3531_353188

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 3 * x + 2 = 2 * y ∧ x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l3531_353188


namespace NUMINAMATH_CALUDE_force_resultant_arithmetic_mean_l3531_353144

/-- Given two forces p₁ and p₂ forming an angle α, if their resultant is equal to their arithmetic mean, 
    then the angle α is between 120° and 180°, and the ratio of the forces is between 1/3 and 3. -/
theorem force_resultant_arithmetic_mean 
  (p₁ p₂ : ℝ) 
  (α : Real) 
  (h_positive : p₁ > 0 ∧ p₂ > 0) 
  (h_resultant : Real.sqrt (p₁^2 + p₂^2 + 2*p₁*p₂*(Real.cos α)) = (p₁ + p₂)/2) : 
  (2*π/3 ≤ α ∧ α ≤ π) ∧ (1/3 ≤ p₁/p₂ ∧ p₁/p₂ ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_force_resultant_arithmetic_mean_l3531_353144


namespace NUMINAMATH_CALUDE_weight_range_proof_l3531_353121

theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) : 
  tracy_weight = 52 →
  jake_weight = tracy_weight + 8 →
  tracy_weight + john_weight + jake_weight = 158 →
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
sorry

end NUMINAMATH_CALUDE_weight_range_proof_l3531_353121


namespace NUMINAMATH_CALUDE_unit_circle_solutions_eq_parameterized_solutions_l3531_353158

noncomputable section

variable (F : Type*) [Field F]

/-- The set of solutions to x^2 + y^2 = 1 in a field F where 1 + 1 ≠ 0 -/
def UnitCircleSolutions (F : Type*) [Field F] (h : (1 : F) + 1 ≠ 0) : Set (F × F) :=
  {p : F × F | p.1^2 + p.2^2 = 1}

/-- The parameterized set of solutions -/
def ParameterizedSolutions (F : Type*) [Field F] : Set (F × F) :=
  {p : F × F | ∃ r : F, r^2 ≠ -1 ∧ 
    p = ((r^2 - 1) / (r^2 + 1), 2*r / (r^2 + 1))} ∪ {(1, 0)}

/-- Theorem stating that the solutions to x^2 + y^2 = 1 are exactly the parameterized solutions -/
theorem unit_circle_solutions_eq_parameterized_solutions 
  (h : (1 : F) + 1 ≠ 0) : 
  UnitCircleSolutions F h = ParameterizedSolutions F :=
by sorry

end

end NUMINAMATH_CALUDE_unit_circle_solutions_eq_parameterized_solutions_l3531_353158


namespace NUMINAMATH_CALUDE_camden_swim_count_l3531_353195

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Susannah went swimming in March -/
def susannah_swims : ℕ := 24

/-- The difference in weekly swims between Susannah and Camden -/
def weekly_swim_difference : ℕ := 2

/-- Camden's total number of swims in March -/
def camden_swims : ℕ := 16

theorem camden_swim_count :
  (susannah_swims / weeks_in_march - weekly_swim_difference) * weeks_in_march = camden_swims := by
  sorry

end NUMINAMATH_CALUDE_camden_swim_count_l3531_353195


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3531_353186

-- Define a structure for the partitioned triangle
structure PartitionedTriangle where
  -- Areas of the three smaller triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  areaQuad : ℝ
  -- Total area of the original triangle
  totalArea : ℝ
  -- Condition: The sum of all areas equals the total area
  sum_areas : area1 + area2 + area3 + areaQuad = totalArea

-- Theorem statement
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 12) : 
  t.areaQuad = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3531_353186


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3531_353112

def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2) (h₂ : a₂ = 5) (h₃ : a₃ = 8) :
  arithmeticSequence a₁ (a₂ - a₁) 30 = 89 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3531_353112


namespace NUMINAMATH_CALUDE_two_heads_five_coins_l3531_353150

/-- The probability of getting exactly k heads when tossing n fair coins -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- Theorem: The probability of getting exactly two heads when tossing five fair coins is 5/16 -/
theorem two_heads_five_coins : coinTossProbability 5 2 = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_five_coins_l3531_353150


namespace NUMINAMATH_CALUDE_sector_area_l3531_353166

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 3) (h2 : θ = 120 * π / 180) :
  (θ / (2 * π)) * π * r^2 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3531_353166


namespace NUMINAMATH_CALUDE_shifted_function_passes_through_origin_l3531_353118

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Represents a vertical shift of a function -/
structure VerticalShift where
  shift : ℝ

/-- Checks if a linear function passes through the origin -/
def passes_through_origin (f : LinearFunction) : Prop :=
  f.slope * 0 + f.intercept = 0

/-- Applies a vertical shift to a linear function -/
def apply_shift (f : LinearFunction) (s : VerticalShift) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - s.shift }

/-- The original linear function y = 3x + 5 -/
def original_function : LinearFunction :=
  { slope := 3, intercept := 5 }

/-- The vertical shift of 5 units down -/
def shift_down : VerticalShift :=
  { shift := 5 }

theorem shifted_function_passes_through_origin :
  passes_through_origin (apply_shift original_function shift_down) := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_passes_through_origin_l3531_353118


namespace NUMINAMATH_CALUDE_additional_license_plates_l3531_353172

def initial_first_letter : Nat := 5
def initial_second_letter : Nat := 3
def initial_first_number : Nat := 5
def initial_second_number : Nat := 5

def new_first_letter : Nat := 5
def new_second_letter : Nat := 4
def new_first_number : Nat := 7
def new_second_number : Nat := 5

def initial_combinations : Nat := initial_first_letter * initial_second_letter * initial_first_number * initial_second_number

def new_combinations : Nat := new_first_letter * new_second_letter * new_first_number * new_second_number

theorem additional_license_plates :
  new_combinations - initial_combinations = 325 := by
  sorry

end NUMINAMATH_CALUDE_additional_license_plates_l3531_353172


namespace NUMINAMATH_CALUDE_quadratic_trinomial_theorem_l3531_353136

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_single_root_when_replaced (q : QuadraticTrinomial) : Prop :=
  (1^2 - 4*q.b*q.c = 0) ∧ (q.b^2 - 4*1*q.c = 0) ∧ (q.b^2 - 4*q.a*1 = 0)

/-- Theorem: If a quadratic trinomial satisfies the condition, then its coefficients are a = c = 1/2 and b = ±√2 -/
theorem quadratic_trinomial_theorem (q : QuadraticTrinomial) :
  has_single_root_when_replaced q →
  (q.a = 1/2 ∧ q.c = 1/2 ∧ (q.b = Real.sqrt 2 ∨ q.b = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_theorem_l3531_353136


namespace NUMINAMATH_CALUDE_haley_growth_rate_l3531_353176

/-- Represents Haley's growth over time -/
structure Growth where
  initial_height : ℝ
  final_height : ℝ
  time_period : ℝ
  growth_rate : ℝ

/-- Theorem stating that given the initial conditions, Haley's growth rate is 3 inches per year -/
theorem haley_growth_rate (g : Growth) 
  (h1 : g.initial_height = 20)
  (h2 : g.final_height = 50)
  (h3 : g.time_period = 10)
  (h4 : g.growth_rate = (g.final_height - g.initial_height) / g.time_period) :
  g.growth_rate = 3 := by
  sorry

#check haley_growth_rate

end NUMINAMATH_CALUDE_haley_growth_rate_l3531_353176


namespace NUMINAMATH_CALUDE_xyz_product_l3531_353161

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3531_353161


namespace NUMINAMATH_CALUDE_erased_number_proof_l3531_353110

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x : ℚ) / (n - 1 : ℚ) = 35 + 7/17 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l3531_353110


namespace NUMINAMATH_CALUDE_set_equality_unordered_elements_l3531_353108

theorem set_equality_unordered_elements : 
  let M : Set ℕ := {4, 5}
  let N : Set ℕ := {5, 4}
  M = N :=
by sorry

end NUMINAMATH_CALUDE_set_equality_unordered_elements_l3531_353108


namespace NUMINAMATH_CALUDE_simplify_expression_l3531_353179

theorem simplify_expression (a : ℝ) : (-a^2)^3 * 3*a = -3*a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3531_353179


namespace NUMINAMATH_CALUDE_last_card_in_box_three_l3531_353126

/-- Represents the number of boxes --/
def num_boxes : ℕ := 7

/-- Represents the total number of cards --/
def total_cards : ℕ := 2015

/-- Represents the length of one complete cycle --/
def cycle_length : ℕ := 12

/-- Calculates the box number for a given card number --/
def box_for_card (card_num : ℕ) : ℕ :=
  let position_in_cycle := card_num % cycle_length
  if position_in_cycle ≤ num_boxes then
    position_in_cycle
  else
    num_boxes - (position_in_cycle - num_boxes)

/-- Theorem stating that the last card (2015th) will be placed in box 3 --/
theorem last_card_in_box_three :
  box_for_card total_cards = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_card_in_box_three_l3531_353126


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l3531_353137

/-- A color type with exactly three colors -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- A rectangle in the plane -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (r : Rectangle) : Prop := sorry

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  c r.p1 = c r.p2 ∧ c r.p1 = c r.p3 ∧ c r.p1 = c r.p4

/-- Theorem: In a plane colored with 3 colors, there exists a rectangle whose vertices are all the same color -/
theorem exists_same_color_rectangle (c : Coloring) :
  ∃ (r : Rectangle), IsRectangle r ∧ SameColorVertices r c := by sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l3531_353137


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3531_353177

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / 4 - y^2 / a = 1

-- Define the asymptotes of the hyperbola
def asymptote (a : ℝ) (x y : ℝ) : Prop := y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x

-- Theorem statement
theorem hyperbola_asymptote_a_value :
  ∀ a : ℝ, a > 1 →
  asymptote a 2 (Real.sqrt 3) →
  hyperbola a 2 (Real.sqrt 3) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_value_l3531_353177


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3531_353148

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3531_353148


namespace NUMINAMATH_CALUDE_sequence_a_formula_l3531_353194

def sequence_a (n : ℕ+) : ℝ := sorry

def sum_S (n : ℕ+) : ℝ := sorry

axiom sum_S_2 : sum_S 2 = 4

axiom sequence_a_next (n : ℕ+) : sequence_a (n + 1) = 2 * sum_S n + 1

theorem sequence_a_formula (n : ℕ+) : sequence_a n = 3^(n.val - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l3531_353194


namespace NUMINAMATH_CALUDE_strokes_over_par_tom_strokes_over_par_l3531_353191

theorem strokes_over_par (rounds : ℕ) (avg_strokes : ℕ) (par_value : ℕ) : ℕ :=
  let total_strokes := rounds * avg_strokes
  let total_par := rounds * par_value
  total_strokes - total_par

theorem tom_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_strokes_over_par_tom_strokes_over_par_l3531_353191


namespace NUMINAMATH_CALUDE_smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l3531_353141

theorem smallest_b_undefined_inverse (b : ℕ) : b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) → 
  b ≥ 330 := by
  sorry

theorem b_330_satisfies_conditions : 
  (∀ x : ℕ, x * 330 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 330 % 55 ≠ 1) := by
  sorry

theorem smallest_b_is_330 : 
  ∃ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) ∧ 
  b = 330 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l3531_353141


namespace NUMINAMATH_CALUDE_log_one_half_decreasing_l3531_353113

-- Define the logarithm function with base a
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our specific function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := log (1/2) x

-- State the theorem
theorem log_one_half_decreasing :
  0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1 →
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f y < f x :=
sorry

end NUMINAMATH_CALUDE_log_one_half_decreasing_l3531_353113


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3531_353169

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Represents the probability of selecting a specific combination of marbles -/
def selectProbability (m : MarbleCount) (r w b g : ℕ) : ℚ :=
  (m.red.choose r * m.white.choose w * m.blue.choose b * m.green.choose g : ℚ) /
  (totalMarbles m).choose 5

/-- The conditions for the marble selection probabilities to be equal -/
def equalProbabilities (m : MarbleCount) : Prop :=
  selectProbability m 5 0 0 0 = selectProbability m 4 1 0 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 3 1 1 0 ∧
  selectProbability m 5 0 0 0 = selectProbability m 2 1 1 1 ∧
  selectProbability m 5 0 0 0 = selectProbability m 1 1 1 1

/-- The theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    m.yellow = 4 ∧ 
    equalProbabilities m ∧
    totalMarbles m = 27 ∧
    (∀ (m' : MarbleCount), m'.yellow = 4 → equalProbabilities m' → totalMarbles m' ≥ 27) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3531_353169


namespace NUMINAMATH_CALUDE_geometric_sequence_bounded_l3531_353125

theorem geometric_sequence_bounded (n k : ℕ) (a : ℕ → ℝ) : 
  n > 0 → k > 0 → 
  (∀ i ∈ Finset.range (k+1), n^k ≤ a i ∧ a i ≤ (n+1)^k) →
  (∀ i ∈ Finset.range k, ∃ q : ℝ, a (i+1) = a i * q) →
  (∀ i ∈ Finset.range (k+1), a i = n^k * ((n+1)/n)^i ∨ a i = (n+1)^k * (n/(n+1))^i) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_bounded_l3531_353125


namespace NUMINAMATH_CALUDE_modulus_complex_l3531_353156

theorem modulus_complex (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = 2 * Real.cos (α/2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_complex_l3531_353156


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3531_353157

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ - 2 = 0 ∧ x₁ = -1/2) ∧
                (2 * x₂^2 - 3 * x₂ - 2 = 0 ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ - 1 = 0 ∧ y₁ = (3 + Real.sqrt 17) / 4) ∧
                (2 * y₂^2 - 3 * y₂ - 1 = 0 ∧ y₂ = (3 - Real.sqrt 17) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3531_353157


namespace NUMINAMATH_CALUDE_zigzag_outward_angle_regular_polygon_l3531_353159

/-- The number of degrees at each outward point of a zigzag extension of a regular polygon -/
def outward_angle (n : ℕ) : ℚ :=
  720 / n

theorem zigzag_outward_angle_regular_polygon (n : ℕ) (h : n > 4) :
  outward_angle n = 720 / n :=
by sorry

end NUMINAMATH_CALUDE_zigzag_outward_angle_regular_polygon_l3531_353159


namespace NUMINAMATH_CALUDE_difference_of_squares_2018_ways_l3531_353180

theorem difference_of_squares_2018_ways :
  ∃ (n : ℕ), n = 5^(2 * 2018) ∧
  (∃! (ways : Finset (ℕ × ℕ)), ways.card = 2018 ∧
    ∀ (a b : ℕ), (a, b) ∈ ways ↔ n = a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_2018_ways_l3531_353180


namespace NUMINAMATH_CALUDE_day300_is_saturday_l3531_353192

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in the year 2004 -/
structure Date2004 where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Function to advance a date by a given number of days -/
def advanceDate (d : Date2004) (days : Nat) : Date2004 :=
  sorry

/-- The 50th day of 2004 is a Monday -/
def day50 : Date2004 :=
  { dayNumber := 50, dayOfWeek := DayOfWeek.Monday }

theorem day300_is_saturday :
  (advanceDate day50 250).dayOfWeek = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_day300_is_saturday_l3531_353192


namespace NUMINAMATH_CALUDE_deceased_member_income_l3531_353197

theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (final_members : ℕ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : initial_average = 735)
  (h3 : final_members = 3)
  (h4 : final_average = 590)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1170 :=
by sorry

end NUMINAMATH_CALUDE_deceased_member_income_l3531_353197


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3531_353147

theorem quadratic_factorization (x : ℝ) : 6*x^2 - 24*x + 18 = 6*(x - 1)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3531_353147


namespace NUMINAMATH_CALUDE_A_suff_not_nec_D_l3531_353119

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_iff_C : B ↔ C
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
by sorry

end NUMINAMATH_CALUDE_A_suff_not_nec_D_l3531_353119


namespace NUMINAMATH_CALUDE_amount_left_after_purchase_l3531_353174

/-- Represents the price of a single lollipop in dollars -/
def lollipop_price : ℚ := 3/2

/-- Represents the price of a pack of gummies in dollars -/
def gummies_price : ℚ := 2

/-- Represents the number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- Represents the number of packs of gummies bought -/
def num_gummies : ℕ := 2

/-- Represents the initial amount of money Chastity had in dollars -/
def initial_amount : ℚ := 15

/-- Theorem stating that the amount left after purchasing the candies is $5 -/
theorem amount_left_after_purchase : 
  initial_amount - (↑num_lollipops * lollipop_price + ↑num_gummies * gummies_price) = 5 := by
  sorry

end NUMINAMATH_CALUDE_amount_left_after_purchase_l3531_353174


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l3531_353171

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The optimal price maximizes the revenue --/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → R p ≥ R q ∧
  p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l3531_353171


namespace NUMINAMATH_CALUDE_triangle_ratio_equation_l3531_353130

/-- In a triangle ABC, given the ratios of sides to heights, prove the equation. -/
theorem triangle_ratio_equation (a b c h_a h_b h_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) 
  (h_triangle : h_a * b = h_b * a ∧ h_b * c = h_c * b ∧ h_c * a = h_a * c) 
  (x y z : ℝ) (h_x : x = a / h_a) (h_y : y = b / h_b) (h_z : z = c / h_c) : 
  x^2 + y^2 + z^2 - 2*x*y - 2*y*z - 2*z*x + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_equation_l3531_353130


namespace NUMINAMATH_CALUDE_rod_length_l3531_353116

theorem rod_length (pieces : ℕ) (piece_length : ℝ) (h1 : pieces = 50) (h2 : piece_length = 0.85) :
  pieces * piece_length = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l3531_353116


namespace NUMINAMATH_CALUDE_equation_implies_equal_variables_l3531_353164

theorem equation_implies_equal_variables (a b : ℝ) 
  (h : (1 / (3 * a)) + (2 / (3 * b)) = 3 / (a + 2 * b)) : a = b :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_equal_variables_l3531_353164


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3531_353170

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3531_353170


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3531_353165

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => x^2 - 3*x + 2
  {x : ℝ | f x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3531_353165


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l3531_353145

open Real

theorem min_value_of_trig_function (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (∀ y, 0 < y ∧ y < π / 2 → 
    (1 + cos (2 * y) + 8 * sin y ^ 2) / sin (2 * y) ≥ 
    (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x)) →
  (1 + cos (2 * x) + 8 * sin x ^ 2) / sin (2 * x) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l3531_353145


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3531_353198

theorem rectangle_area_equals_perimeter (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (length > 0 ∧ width > 0 ∧ area = perimeter) → x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l3531_353198


namespace NUMINAMATH_CALUDE_f_2015_equals_one_l3531_353127

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2015_equals_one (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : ∀ x, f (x + 2) * f x = 1)
  (h3 : ∀ x, f x > 0) : 
  f 2015 = 1 := by sorry

end NUMINAMATH_CALUDE_f_2015_equals_one_l3531_353127


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3531_353135

/-- Arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0  -- Arithmetic property
  h_sum : ∀ n, S n = n * (a 0 + a (n-1)) / 2  -- Sum formula

/-- The main theorem -/
theorem arithmetic_sequence_difference
  (seq : ArithmeticSequence)
  (h : seq.S 10 / 10 - seq.S 9 / 9 = 1) :
  seq.a 1 - seq.a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3531_353135


namespace NUMINAMATH_CALUDE_triangle_properties_l3531_353178

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  area : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * Real.sin t.B = 2 * Real.sin t.C)
  (h3 : t.area = 2 + 2 * Real.sqrt 3) :
  (t.b = Real.sqrt 2 * t.a) ∧ (t.b = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3531_353178


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_quadratic_equations_all_solutions_l3531_353120

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x - 1 = 0) ∧
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) :=
by
  constructor
  · use 1 + Real.sqrt 2
    sorry
  · use 2
    sorry

theorem quadratic_equations_all_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  (∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ (x = 2 ∨ x = 4)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_quadratic_equations_all_solutions_l3531_353120


namespace NUMINAMATH_CALUDE_total_albums_is_2835_l3531_353151

/-- The total number of albums owned by six people given certain relationships between their album counts. -/
def total_albums (adele_albums : ℕ) : ℕ :=
  let bridget_albums := adele_albums - 15
  let katrina_albums := 6 * bridget_albums
  let miriam_albums := 7 * katrina_albums
  let carlos_albums := 3 * miriam_albums
  let diane_albums := 2 * katrina_albums
  adele_albums + bridget_albums + katrina_albums + miriam_albums + carlos_albums + diane_albums

/-- Theorem stating that the total number of albums is 2835 given the conditions in the problem. -/
theorem total_albums_is_2835 : total_albums 30 = 2835 := by
  sorry

end NUMINAMATH_CALUDE_total_albums_is_2835_l3531_353151


namespace NUMINAMATH_CALUDE_cube_split_with_31_l3531_353109

/-- For a natural number m > 1, if 31 is one of the odd numbers in the sum that equals m^3, then m = 6. -/
theorem cube_split_with_31 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ) (l : List ℕ), 
    (∀ n ∈ l, Odd n) ∧ 
    (List.sum l = m^3) ∧
    (31 ∈ l) ∧
    (List.length l = m)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_cube_split_with_31_l3531_353109


namespace NUMINAMATH_CALUDE_evaluate_expression_l3531_353103

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 1/3 → z = 12 → x^3 * y^4 * z = 1/432 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3531_353103


namespace NUMINAMATH_CALUDE_pencil_cost_l3531_353105

/-- Proves that the cost of each pencil Cindi bought is $0.50 --/
theorem pencil_cost (cindi_pencils : ℕ) (marcia_pencils : ℕ) (donna_pencils : ℕ) 
  (h1 : marcia_pencils = 2 * cindi_pencils)
  (h2 : donna_pencils = 3 * marcia_pencils)
  (h3 : donna_pencils + marcia_pencils = 480)
  (h4 : cindi_pencils * (cost_per_pencil : ℚ) = 30) : 
  cost_per_pencil = 1/2 := by
  sorry

#check pencil_cost

end NUMINAMATH_CALUDE_pencil_cost_l3531_353105


namespace NUMINAMATH_CALUDE_solve_percentage_problem_l3531_353107

theorem solve_percentage_problem (x : ℝ) : (0.7 * x = (1/3) * x + 110) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_problem_l3531_353107


namespace NUMINAMATH_CALUDE_cos_2alpha_special_value_l3531_353152

theorem cos_2alpha_special_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (α - π/4) = 1/3) : 
  Real.cos (2*α) = -4*Real.sqrt 2/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_special_value_l3531_353152


namespace NUMINAMATH_CALUDE_rational_numbers_four_units_from_origin_l3531_353146

theorem rational_numbers_four_units_from_origin :
  {x : ℚ | |x| = 4} = {-4, 4} := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_four_units_from_origin_l3531_353146


namespace NUMINAMATH_CALUDE_factor_x8_minus_81_l3531_353123

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x^2 - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_x8_minus_81_l3531_353123


namespace NUMINAMATH_CALUDE_floor_of_6_8_l3531_353101

theorem floor_of_6_8 : ⌊(6.8 : ℝ)⌋ = 6 := by sorry

end NUMINAMATH_CALUDE_floor_of_6_8_l3531_353101


namespace NUMINAMATH_CALUDE_m_divided_by_8_l3531_353173

theorem m_divided_by_8 (m : ℕ) (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l3531_353173


namespace NUMINAMATH_CALUDE_binary_sum_equals_141_l3531_353184

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 111000₂ -/
def binary2 : List Bool := [false, false, false, true, true, true]

/-- The sum of the two binary numbers in decimal -/
def sum_decimal : ℕ := binary_to_decimal binary1 + binary_to_decimal binary2

theorem binary_sum_equals_141 : sum_decimal = 141 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_141_l3531_353184


namespace NUMINAMATH_CALUDE_seven_rows_five_seats_l3531_353124

-- Define a movie ticket as a pair of natural numbers
def MovieTicket : Type := ℕ × ℕ

-- Define a function to create a movie ticket representation
def createTicket (rows : ℕ) (seats : ℕ) : MovieTicket := (rows, seats)

-- Theorem statement
theorem seven_rows_five_seats :
  createTicket 7 5 = (7, 5) := by sorry

end NUMINAMATH_CALUDE_seven_rows_five_seats_l3531_353124


namespace NUMINAMATH_CALUDE_rectangle_to_square_dissection_l3531_353132

theorem rectangle_to_square_dissection :
  ∃ (a b c d : ℝ),
    -- Rectangle dimensions
    16 * 9 = a * b + c * d ∧
    -- Two parts form a square
    12 * 12 = a * b + c * d ∧
    -- Dimensions are positive
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    -- One dimension of each part matches the square
    (a = 12 ∨ b = 12 ∨ c = 12 ∨ d = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_dissection_l3531_353132


namespace NUMINAMATH_CALUDE_polynomial_value_at_zero_l3531_353160

theorem polynomial_value_at_zero (p : Polynomial ℝ) : 
  (Polynomial.degree p = 7) →
  (∀ n : Nat, n ≤ 7 → p.eval (3^n) = (3^n)⁻¹) →
  p.eval 0 = 19682 / 6561 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_zero_l3531_353160


namespace NUMINAMATH_CALUDE_smallest_number_l3531_353140

theorem smallest_number (S : Set ℤ) : S = {-2, -1, 0, 1} → ∀ x ∈ S, -2 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3531_353140


namespace NUMINAMATH_CALUDE_radio_profit_percentage_is_approximately_6_8_percent_l3531_353143

/-- Calculates the profit percentage for a radio sale given the following parameters:
    * initial_cost: The initial cost of the radio
    * overhead: Overhead expenses
    * purchase_tax_rate: Purchase tax rate
    * luxury_tax_rate: Luxury tax rate
    * exchange_discount_rate: Exchange offer discount rate
    * sales_tax_rate: Sales tax rate
    * selling_price: Final selling price
-/
def calculate_profit_percentage (
  initial_cost : ℝ
  ) (overhead : ℝ
  ) (purchase_tax_rate : ℝ
  ) (luxury_tax_rate : ℝ
  ) (exchange_discount_rate : ℝ
  ) (sales_tax_rate : ℝ
  ) (selling_price : ℝ
  ) : ℝ :=
  sorry

/-- The profit percentage for the radio sale is approximately 6.8% -/
theorem radio_profit_percentage_is_approximately_6_8_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_profit_percentage 225 28 0.08 0.05 0.10 0.12 300 - 6.8| < ε :=
sorry

end NUMINAMATH_CALUDE_radio_profit_percentage_is_approximately_6_8_percent_l3531_353143


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3531_353154

/-- Calculates the total money taken in on ticket sales given the prices and number of tickets sold. -/
def totalTicketSales (adultPrice childPrice : ℕ) (totalTickets adultTickets : ℕ) : ℕ :=
  adultPrice * adultTickets + childPrice * (totalTickets - adultTickets)

/-- Theorem stating that given the specific ticket prices and sales, the total money taken in is $206. -/
theorem theater_ticket_sales :
  totalTicketSales 8 5 34 12 = 206 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3531_353154


namespace NUMINAMATH_CALUDE_sequence_problem_l3531_353187

/-- Given a sequence of positive integers x₁, x₂, ..., x₇ satisfying
    x₆ = 144 and x_{n+3} = x_{n+2}(x_{n+1} + x_n) for n = 1, 2, 3, 4,
    prove that x₇ = 3456. -/
theorem sequence_problem (x : Fin 7 → ℕ+) 
    (h1 : x 6 = 144)
    (h2 : ∀ n : Fin 4, x (n + 3) = x (n + 2) * (x (n + 1) + x n)) :
  x 7 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3531_353187


namespace NUMINAMATH_CALUDE_solve_for_n_l3531_353162

theorem solve_for_n (Q s r k : ℝ) (h : Q = (s * r) / (1 + k) ^ n) :
  n = Real.log ((s * r) / Q) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_solve_for_n_l3531_353162


namespace NUMINAMATH_CALUDE_lassis_from_mangoes_l3531_353133

/-- Given that 20 lassis can be made from 4 mangoes, prove that 80 lassis can be made from 16 mangoes. -/
theorem lassis_from_mangoes (make_lassis : ℕ → ℕ) 
  (h1 : make_lassis 4 = 20) 
  (h2 : ∀ x y : ℕ, make_lassis (x + y) = make_lassis x + make_lassis y) : 
  make_lassis 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lassis_from_mangoes_l3531_353133


namespace NUMINAMATH_CALUDE_two_color_theorem_l3531_353100

/-- A type representing the two colors used for coloring regions -/
inductive Color
| Blue
| Red

/-- A type representing a circle in the plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A type representing a region in the plane -/
structure Region where
  -- We don't need to define the internal structure of a region for this problem

/-- A function type for coloring regions -/
def ColoringFunction := Region → Color

/-- Predicate to check if two regions are adjacent (separated by a circle arc) -/
def are_adjacent (r1 r2 : Region) : Prop := sorry

/-- Theorem stating the existence of a valid two-color coloring for n circles -/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region) (coloring : ColoringFunction),
    (circles.card = n) ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → are_adjacent r1 r2 →
      coloring r1 ≠ coloring r2) :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3531_353100


namespace NUMINAMATH_CALUDE_x_squared_in_set_l3531_353149

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, -1, x} : Set ℝ) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l3531_353149


namespace NUMINAMATH_CALUDE_max_product_953_l3531_353175

/-- A type representing a valid digit for our problem -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  d1 : Digit
  d2 : Digit
  d3 : Digit
  d4 : Digit
  d5 : Digit
  all_different : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
                  d3 ≠ d4 ∧ d3 ≠ d5 ∧
                  d4 ≠ d5

/-- Function to calculate the product of a three-digit and two-digit number from a DigitCombination -/
def calculate_product (dc : DigitCombination) : ℕ :=
  (100 * digit_to_nat dc.d1 + 10 * digit_to_nat dc.d2 + digit_to_nat dc.d3) *
  (10 * digit_to_nat dc.d4 + digit_to_nat dc.d5)

/-- The main theorem stating that 953 yields the maximum product -/
theorem max_product_953 :
  ∀ dc : DigitCombination,
  calculate_product dc ≤ calculate_product
    { d1 := Digit.nine, d2 := Digit.five, d3 := Digit.three,
      d4 := Digit.eight, d5 := Digit.six,
      all_different := by simp } :=
sorry

end NUMINAMATH_CALUDE_max_product_953_l3531_353175


namespace NUMINAMATH_CALUDE_law_firm_associates_tenure_l3531_353106

theorem law_firm_associates_tenure (total : ℝ) (first_year : ℝ) (second_year : ℝ) (more_than_two_years : ℝ)
  (h1 : second_year / total = 0.3)
  (h2 : (total - first_year) / total = 0.6) :
  more_than_two_years / total = 0.6 - 0.3 := by
sorry

end NUMINAMATH_CALUDE_law_firm_associates_tenure_l3531_353106


namespace NUMINAMATH_CALUDE_pear_sales_ratio_l3531_353163

/-- Given the total pears sold and the amount sold in the afternoon, 
    prove the ratio of afternoon sales to morning sales. -/
theorem pear_sales_ratio 
  (total_pears : ℕ) 
  (afternoon_pears : ℕ) 
  (h1 : total_pears = 480)
  (h2 : afternoon_pears = 320) :
  afternoon_pears / (total_pears - afternoon_pears) = 2 := by
  sorry

end NUMINAMATH_CALUDE_pear_sales_ratio_l3531_353163


namespace NUMINAMATH_CALUDE_scrabble_middle_letter_value_l3531_353182

/-- Given a three-letter word in Scrabble with known conditions, 
    prove the value of the middle letter. -/
theorem scrabble_middle_letter_value 
  (first_letter_value : ℕ) 
  (third_letter_value : ℕ) 
  (total_score : ℕ) 
  (h1 : first_letter_value = 1)
  (h2 : third_letter_value = 1)
  (h3 : total_score = 30)
  (h4 : ∃ (middle_letter_value : ℕ), 
    3 * (first_letter_value + middle_letter_value + third_letter_value) = total_score) :
  ∃ (middle_letter_value : ℕ), middle_letter_value = 8 := by
  sorry

end NUMINAMATH_CALUDE_scrabble_middle_letter_value_l3531_353182


namespace NUMINAMATH_CALUDE_mityas_age_l3531_353199

theorem mityas_age (shura_age mitya_age : ℝ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by
sorry

end NUMINAMATH_CALUDE_mityas_age_l3531_353199


namespace NUMINAMATH_CALUDE_people_per_bus_l3531_353138

/-- Given a field trip with vans and buses, calculate the number of people per bus -/
theorem people_per_bus 
  (total_people : ℕ) 
  (num_vans : ℕ) 
  (people_per_van : ℕ) 
  (num_buses : ℕ) 
  (h1 : total_people = 342)
  (h2 : num_vans = 9)
  (h3 : people_per_van = 8)
  (h4 : num_buses = 10)
  : (total_people - num_vans * people_per_van) / num_buses = 27 := by
  sorry

end NUMINAMATH_CALUDE_people_per_bus_l3531_353138


namespace NUMINAMATH_CALUDE_monotonicity_of_f_range_of_b_l3531_353183

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + b * x + 1)

theorem monotonicity_of_f :
  let f := f 1 1
  ∀ x₁ x₂, (x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂) → f x₁ < f x₂ ∧
           (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1) → f x₁ > f x₂ ∧
           (1 < x₁ ∧ x₁ < x₂) → f x₁ < f x₂ :=
sorry

theorem range_of_b :
  ∀ b : ℝ, (∀ x : ℝ, x ≥ 1 → f 0 b x ≥ 1) ↔ (0 ≤ b ∧ b ≤ Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_range_of_b_l3531_353183


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3531_353117

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the origin for two points. -/
def symmetricToOrigin (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = -a.y

/-- Theorem stating that if point A(5, -1) is symmetric to point B with respect to the origin,
    then the coordinates of point B are (-5, 1). -/
theorem symmetric_point_coordinates :
  let a : Point := ⟨5, -1⟩
  let b : Point := ⟨-5, 1⟩
  symmetricToOrigin a b :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3531_353117


namespace NUMINAMATH_CALUDE_triangle_area_l3531_353167

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3531_353167


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l3531_353155

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -1) :
  (1 : ℝ) / (a + 1) + a / (a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l3531_353155


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l3531_353131

/-- Calculates the total points earned by Wendy for recycling cans and newspapers -/
def total_points (cans_recycled : ℕ) (newspapers_recycled : ℕ) : ℕ :=
  cans_recycled * 5 + newspapers_recycled * 10

/-- Proves that Wendy's total points earned is 75 given the problem conditions -/
theorem wendy_recycling_points :
  let cans_total : ℕ := 11
  let cans_recycled : ℕ := 9
  let newspapers_recycled : ℕ := 3
  total_points cans_recycled newspapers_recycled = 75 := by
  sorry

#eval total_points 9 3

end NUMINAMATH_CALUDE_wendy_recycling_points_l3531_353131


namespace NUMINAMATH_CALUDE_total_pears_is_fifteen_l3531_353134

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 8

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 7

/-- The total number of pears picked -/
def total_pears : ℕ := mike_pears + jason_pears

theorem total_pears_is_fifteen : total_pears = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_is_fifteen_l3531_353134


namespace NUMINAMATH_CALUDE_jet_ski_time_to_dock_b_l3531_353196

/-- Represents the scenario of a jet ski and a canoe traveling on a river --/
structure RiverTravel where
  distance : ℝ  -- Distance between dock A and dock B
  speed_difference : ℝ  -- Speed difference between jet ski and current
  total_time : ℝ  -- Total time until jet ski meets canoe

/-- 
Calculates the time taken by the jet ski to reach dock B.
Returns the time in hours.
-/
def time_to_dock_b (rt : RiverTravel) : ℝ :=
  sorry

/-- Theorem stating that the time taken by the jet ski to reach dock B is 3 hours --/
theorem jet_ski_time_to_dock_b (rt : RiverTravel) 
  (h1 : rt.distance = 60) 
  (h2 : rt.speed_difference = 10) 
  (h3 : rt.total_time = 8) : 
  time_to_dock_b rt = 3 :=
  sorry

end NUMINAMATH_CALUDE_jet_ski_time_to_dock_b_l3531_353196


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3531_353102

/-- A line y = x - b intersects a circle (x-2)^2 + y^2 = 1 at two distinct points
    if and only if b is in the open interval (2 - √2, 2 + √2) -/
theorem line_circle_intersection (b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ - b ∧ y₂ = x₂ - b ∧
    (x₁ - 2)^2 + y₁^2 = 1 ∧
    (x₂ - 2)^2 + y₂^2 = 1) ↔ 
  (2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3531_353102


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l3531_353189

theorem positive_numbers_inequalities 
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l3531_353189


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3531_353114

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1056 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 23 ≠ 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3531_353114


namespace NUMINAMATH_CALUDE_sum_of_roots_l3531_353181

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 6

-- State the theorem
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = -5) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3531_353181


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3531_353153

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating that the intersection point of lines AB and CD is (-4/3, 35, 3/2) --/
theorem intersection_of_lines :
  let A : ℝ × ℝ × ℝ := (3, -5, 4)
  let B : ℝ × ℝ × ℝ := (13, -15, 9)
  let C : ℝ × ℝ × ℝ := (-6, 6, -12)
  let D : ℝ × ℝ × ℝ := (-4, -2, 8)
  intersection_point A B C D = (-4/3, 35, 3/2) := by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3531_353153


namespace NUMINAMATH_CALUDE_min_score_for_average_l3531_353115

def total_tests : ℕ := 7
def max_score : ℕ := 100
def target_average : ℕ := 80

def first_four_scores : List ℕ := [82, 90, 78, 85]

theorem min_score_for_average (scores : List ℕ) 
  (h1 : scores.length = 4)
  (h2 : ∀ s ∈ scores, s ≤ max_score) :
  ∃ (x y z : ℕ),
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / total_tests = target_average ∧
    (∀ a b c : ℕ, 
      a ≤ max_score → b ≤ max_score → c ≤ max_score →
      (scores.sum + a + b + c) / total_tests = target_average →
      min x y ≤ min a b ∧ min x y ≤ c) ∧
    25 = min x (min y z) := by
  sorry

#check min_score_for_average first_four_scores

end NUMINAMATH_CALUDE_min_score_for_average_l3531_353115


namespace NUMINAMATH_CALUDE_coordinates_of_point_A_l3531_353128

def point_A (a : ℝ) : ℝ × ℝ := (a - 1, 3 * a - 2)

theorem coordinates_of_point_A :
  ∀ a : ℝ, (point_A a).1 = (point_A a).2 + 3 → point_A a = (-2, -5) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_A_l3531_353128


namespace NUMINAMATH_CALUDE_exam_comparison_l3531_353142

/-- Proves that Lyssa has 3 fewer correct answers than Precious in an exam with 75 items,
    where Lyssa answers 20% incorrectly and Precious makes 12 mistakes. -/
theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 75)
  (h2 : lyssa_incorrect_percent = 1/5)
  (h3 : precious_mistakes = 12) :
  (total_items - (lyssa_incorrect_percent * total_items).floor) = 
  (total_items - precious_mistakes) - 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_comparison_l3531_353142


namespace NUMINAMATH_CALUDE_age_sum_proof_l3531_353168

theorem age_sum_proof (patrick michael monica : ℕ) : 
  3 * michael = 5 * patrick →
  3 * monica = 5 * michael →
  monica - patrick = 80 →
  patrick + michael + monica = 245 := by
sorry

end NUMINAMATH_CALUDE_age_sum_proof_l3531_353168
