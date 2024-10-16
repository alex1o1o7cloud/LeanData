import Mathlib

namespace NUMINAMATH_CALUDE_two_numbers_product_equals_remaining_sum_l762_76217

theorem two_numbers_product_equals_remaining_sum : ∃ x y : ℕ,
  x ∈ Finset.range 33 ∧ 
  y ∈ Finset.range 33 ∧
  x ≠ y ∧
  x * y = 484 ∧
  (Finset.sum (Finset.range 33) id) - x - y = 484 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_product_equals_remaining_sum_l762_76217


namespace NUMINAMATH_CALUDE_solid_with_rectangular_views_is_cuboid_l762_76227

/-- A solid is a three-dimensional geometric object. -/
structure Solid :=
  (shape : Type)

/-- A view is a two-dimensional projection of a solid. -/
inductive View
  | Front
  | Top
  | Side

/-- A rectangle is a quadrilateral with four right angles. -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

/-- A cuboid is a three-dimensional solid with six rectangular faces. -/
structure Cuboid :=
  (length : ℝ)
  (width : ℝ)
  (height : ℝ)

/-- The projection of a solid onto a view. -/
def projection (s : Solid) (v : View) : Type :=
  sorry

/-- Theorem: If a solid's three views are all rectangles, then the solid is a cuboid. -/
theorem solid_with_rectangular_views_is_cuboid (s : Solid) :
  (∀ v : View, projection s v = Rectangle) → (s.shape = Cuboid) :=
sorry

end NUMINAMATH_CALUDE_solid_with_rectangular_views_is_cuboid_l762_76227


namespace NUMINAMATH_CALUDE_round_trip_distance_l762_76287

/-- Calculates the distance of a round trip given upstream speed, downstream speed, and total time -/
theorem round_trip_distance 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : total_time > 0)
  (h4 : upstream_speed = 3)
  (h5 : downstream_speed = 9)
  (h6 : total_time = 8) :
  (let distance := (upstream_speed * downstream_speed * total_time) / (upstream_speed + downstream_speed)
   distance = 18) := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_l762_76287


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l762_76293

theorem sum_of_roots_quadratic : ∃ (x₁ x₂ : ℝ), 
  x₁^2 - 7*x₁ + 10 = 0 ∧ 
  x₂^2 - 7*x₂ + 10 = 0 ∧ 
  x₁ + x₂ = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l762_76293


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l762_76281

theorem nested_fraction_evaluation :
  (2 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l762_76281


namespace NUMINAMATH_CALUDE_equivalent_statements_l762_76256

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_statements_l762_76256


namespace NUMINAMATH_CALUDE_train_average_speed_l762_76230

def train_distance_1 : ℝ := 240
def train_time_1 : ℝ := 3
def train_distance_2 : ℝ := 450
def train_time_2 : ℝ := 5

theorem train_average_speed :
  (train_distance_1 + train_distance_2) / (train_time_1 + train_time_2) = 86.25 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l762_76230


namespace NUMINAMATH_CALUDE_substring_012_occurrences_l762_76221

/-- Base-3 representation of an integer without leading zeroes -/
def base3Repr (n : ℕ) : List ℕ := sorry

/-- Continuous string formed by joining base-3 representations of integers from 1 to 729 -/
def continuousString : List ℕ := sorry

/-- Count occurrences of a substring in a list -/
def countSubstring (list : List ℕ) (substring : List ℕ) : ℕ := sorry

theorem substring_012_occurrences :
  countSubstring continuousString [0, 1, 2] = 148 := by sorry

end NUMINAMATH_CALUDE_substring_012_occurrences_l762_76221


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l762_76204

/-- Given a line with equation x - 2y + 1 = 0, its symmetric line with respect to the y-axis
    has the equation x + 2y - 1 = 0 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), x - 2*y + 1 = 0 → ∃ (x' y' : ℝ), x' + 2*y' - 1 = 0 ∧ x' = -x ∧ y' = y :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l762_76204


namespace NUMINAMATH_CALUDE_anode_reaction_in_saturated_brine_electrolysis_l762_76236

-- Define the types of particles and reactions
inductive Particle
| Chloride
| Electron
| ChlorineMolecule

inductive Reaction
| Oxidation (reactants : List Particle) (products : List Particle)

-- Define the electrolysis process
def saturatedBrineElectrolysis : Reaction :=
  Reaction.Oxidation [Particle.Chloride, Particle.Chloride] [Particle.ChlorineMolecule]

-- Define the anode reaction
def anodeReaction (r : Reaction) : Prop :=
  match r with
  | Reaction.Oxidation reactants products =>
    reactants = [Particle.Chloride, Particle.Chloride] ∧
    products = [Particle.ChlorineMolecule]

-- Theorem stating that the anode reaction in saturated brine electrolysis
-- is the oxidation of chloride ions to chlorine gas
theorem anode_reaction_in_saturated_brine_electrolysis :
  anodeReaction saturatedBrineElectrolysis :=
by
  sorry

end NUMINAMATH_CALUDE_anode_reaction_in_saturated_brine_electrolysis_l762_76236


namespace NUMINAMATH_CALUDE_correct_division_result_l762_76260

theorem correct_division_result (wrong_divisor correct_divisor student_answer : ℕ) 
  (h1 : wrong_divisor = 840)
  (h2 : correct_divisor = 420)
  (h3 : student_answer = 36) :
  (wrong_divisor * student_answer) / correct_divisor = 72 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_result_l762_76260


namespace NUMINAMATH_CALUDE_smallest_fraction_divisible_l762_76298

theorem smallest_fraction_divisible (f1 f2 f3 : Rat) (h1 : f1 = 6/7) (h2 : f2 = 5/14) (h3 : f3 = 10/21) :
  (∀ q : Rat, (∃ n1 n2 n3 : ℤ, f1 * q = n1 ∧ f2 * q = n2 ∧ f3 * q = n3) →
    (1 : Rat) / 42 ≤ q) ∧
  (∃ n1 n2 n3 : ℤ, f1 * (1/42 : Rat) = n1 ∧ f2 * (1/42 : Rat) = n2 ∧ f3 * (1/42 : Rat) = n3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_divisible_l762_76298


namespace NUMINAMATH_CALUDE_maximize_annual_average_profit_l762_76271

/-- Represents the problem of maximizing annual average profit for equipment purchase --/
theorem maximize_annual_average_profit :
  let initial_cost : ℕ := 90000
  let first_year_cost : ℕ := 20000
  let annual_cost_increase : ℕ := 20000
  let annual_revenue : ℕ := 110000
  let total_cost (n : ℕ) : ℕ := initial_cost + n * first_year_cost + (n * (n - 1) * annual_cost_increase) / 2
  let total_revenue (n : ℕ) : ℕ := n * annual_revenue
  let total_profit (n : ℕ) : ℤ := (total_revenue n : ℤ) - (total_cost n : ℤ)
  let annual_average_profit (n : ℕ) : ℚ := (total_profit n : ℚ) / n
  ∀ m : ℕ, m > 0 → annual_average_profit 3 ≥ annual_average_profit m :=
by
  sorry


end NUMINAMATH_CALUDE_maximize_annual_average_profit_l762_76271


namespace NUMINAMATH_CALUDE_simplify_expression_l762_76210

theorem simplify_expression : 
  1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l762_76210


namespace NUMINAMATH_CALUDE_simplify_fraction_l762_76261

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  ((a + 1) / (a - 1) + 1) / (2 * a / (a^2 - 1)) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l762_76261


namespace NUMINAMATH_CALUDE_inequality_proof_l762_76262

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 0 < a / b ∧ a / b < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l762_76262


namespace NUMINAMATH_CALUDE_system_solution_l762_76258

theorem system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l762_76258


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l762_76268

theorem product_xyz_equals_negative_two
  (x y z : ℝ)
  (h1 : x + 2 / y = 2)
  (h2 : y + 2 / z = 2) :
  x * y * z = -2 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_two_l762_76268


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l762_76257

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l762_76257


namespace NUMINAMATH_CALUDE_integer_fraction_triples_l762_76255

theorem integer_fraction_triples :
  ∀ a b c : ℕ+,
    (a = 1 ∧ b = 20 ∧ c = 1) ∨
    (a = 1 ∧ b = 4 ∧ c = 1) ∨
    (a = 3 ∧ b = 4 ∧ c = 1) ↔
    ∃ k : ℤ, (32 * a.val + 3 * b.val + 48 * c.val) = 4 * k * a.val * b.val * c.val := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_triples_l762_76255


namespace NUMINAMATH_CALUDE_raghu_investment_l762_76290

theorem raghu_investment
  (vishal_investment : ℝ)
  (trishul_investment : ℝ)
  (raghu_investment : ℝ)
  (vishal_more_than_trishul : vishal_investment = 1.1 * trishul_investment)
  (trishul_less_than_raghu : trishul_investment = 0.9 * raghu_investment)
  (total_investment : vishal_investment + trishul_investment + raghu_investment = 6936) :
  raghu_investment = 2400 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l762_76290


namespace NUMINAMATH_CALUDE_total_streets_patrolled_in_one_hour_l762_76248

/-- Represents the patrol rate of a police officer -/
structure PatrolRate where
  streets : ℕ
  hours : ℕ

/-- Calculates the number of streets patrolled per hour -/
def streetsPerHour (rate : PatrolRate) : ℚ :=
  rate.streets / rate.hours

/-- The patrol rates of three officers -/
def officerA : PatrolRate := { streets := 36, hours := 4 }
def officerB : PatrolRate := { streets := 55, hours := 5 }
def officerC : PatrolRate := { streets := 42, hours := 6 }

/-- The total number of streets patrolled by all three officers in one hour -/
def totalStreetsPerHour : ℚ :=
  streetsPerHour officerA + streetsPerHour officerB + streetsPerHour officerC

theorem total_streets_patrolled_in_one_hour :
  totalStreetsPerHour = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_streets_patrolled_in_one_hour_l762_76248


namespace NUMINAMATH_CALUDE_a2_value_l762_76280

theorem a2_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + a₄*(x-2)^4) →
  a₂ = 24 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l762_76280


namespace NUMINAMATH_CALUDE_power_multiplication_l762_76272

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l762_76272


namespace NUMINAMATH_CALUDE_complex_number_problem_l762_76223

theorem complex_number_problem (a b c : ℂ) 
  (h_a_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l762_76223


namespace NUMINAMATH_CALUDE_arithmetic_sequence_characterization_l762_76231

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_characterization (a : ℕ+ → ℝ) :
  is_arithmetic_sequence a ↔ ∀ n : ℕ+, 2 * a (n + 1) = a n + a (n + 2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_characterization_l762_76231


namespace NUMINAMATH_CALUDE_parallelogram_count_l762_76213

/-- The number of parallelograms formed by lines passing through each grid point in a triangle -/
def f (n : ℕ) : ℕ := 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24

/-- Theorem stating that f(n) correctly calculates the number of parallelograms -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l762_76213


namespace NUMINAMATH_CALUDE_sum_simplification_l762_76224

theorem sum_simplification : -2^3 + (-2)^4 + 2^2 - 2^3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l762_76224


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l762_76237

-- Part 1
theorem simplify_expression (a b : ℝ) :
  2 * (a + b)^2 - 8 * (a + b)^2 + 3 * (a + b)^2 = -3 * (a + b)^2 := by sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (h : x^2 + 2*y = 4) :
  -3*x^2 - 6*y + 17 = 5 := by sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 3*b = 3) (h2 : 2*b - c = -5) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_complex_expression_l762_76237


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_solution_set_for_any_a_l762_76284

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem solution_set_for_any_a (a : ℝ) :
  ({x : ℝ | f a x < 0} = ∅ ∧ a = 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | a < x ∧ x < 2*a} ∧ a > 0) ∨
  ({x : ℝ | f a x < 0} = {x : ℝ | 2*a < x ∧ x < a} ∧ a < 0) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_solution_set_for_any_a_l762_76284


namespace NUMINAMATH_CALUDE_boy_running_duration_l762_76206

theorem boy_running_duration (initial_speed initial_time second_distance second_speed : ℝ) 
  (h1 : initial_speed = 15)
  (h2 : initial_time = 3)
  (h3 : second_distance = 190)
  (h4 : second_speed = 19) : 
  initial_time + second_distance / second_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_boy_running_duration_l762_76206


namespace NUMINAMATH_CALUDE_line_slope_l762_76291

/-- The slope of the line given by the equation x/4 + y/3 = 2 is -3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 + y / 3 = 2) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l762_76291


namespace NUMINAMATH_CALUDE_exponential_models_for_rapid_change_l762_76228

/-- Represents an exponential function model -/
structure ExponentialModel where
  -- Add necessary fields here
  rapidChange : Bool
  largeChangeInShortTime : Bool

/-- Represents a practical problem with rapid changes and large amounts of change in short periods -/
structure RapidChangeProblem where
  -- Add necessary fields here
  hasRapidChange : Bool
  hasLargeChangeInShortTime : Bool

/-- States that exponential models are generally used for rapid change problems -/
theorem exponential_models_for_rapid_change 
  (model : ExponentialModel) 
  (problem : RapidChangeProblem) : 
  model.rapidChange ∧ model.largeChangeInShortTime → 
  problem.hasRapidChange ∧ problem.hasLargeChangeInShortTime →
  (∃ (usage : Bool), usage = true) :=
by
  sorry

#check exponential_models_for_rapid_change

end NUMINAMATH_CALUDE_exponential_models_for_rapid_change_l762_76228


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l762_76229

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l762_76229


namespace NUMINAMATH_CALUDE_hawks_score_l762_76208

/-- The number of touchdowns scored by the Hawks -/
def touchdowns : ℕ := 3

/-- The number of points awarded for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total points scored by the Hawks -/
def total_points : ℕ := touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score : total_points = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l762_76208


namespace NUMINAMATH_CALUDE_hex_B1F4_equals_45556_l762_76242

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Convert a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Convert a list of HexDigits to its decimal value --/
def hexListToDecimal (hexList : List HexDigit) : Nat :=
  hexList.foldr (fun digit acc => hexToDecimal digit + 16 * acc) 0

theorem hex_B1F4_equals_45556 :
  hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4] = 45556 := by
  sorry

#eval hexListToDecimal [HexDigit.B, HexDigit.D1, HexDigit.F, HexDigit.D4]

end NUMINAMATH_CALUDE_hex_B1F4_equals_45556_l762_76242


namespace NUMINAMATH_CALUDE_angle_A_measure_l762_76241

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem angle_A_measure (t : Triangle) (h1 : t.a = 7) (h2 : t.b = 8) (h3 : Real.cos t.B = 1/7) :
  t.A = π/3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l762_76241


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l762_76283

/-- Given an equation (x² + cx + d) / (2x - e) = (n - 2) / (n + 2),
    if the roots are numerically equal but have opposite signs,
    then n = (-4 - 2c) / (c - 2) -/
theorem roots_opposite_signs (c d e : ℝ) :
  let f (x : ℝ) := (x^2 + c*x + d) / (2*x - e)
  ∃ (n : ℝ), (∀ x, f x = (n - 2) / (n + 2)) →
  (∃ (r : ℝ), (f r = (n - 2) / (n + 2) ∧ f (-r) = (n - 2) / (n + 2))) →
  n = (-4 - 2*c) / (c - 2) := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l762_76283


namespace NUMINAMATH_CALUDE_degree_of_h_is_4_l762_76274

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 8*x^4 + x^5

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem degree_of_h_is_4 : degree (h 0) = 4 := by sorry

end NUMINAMATH_CALUDE_degree_of_h_is_4_l762_76274


namespace NUMINAMATH_CALUDE_vase_discount_percentage_l762_76289

/-- Proves that the discount percentage on a vase is 25% given the conditions -/
theorem vase_discount_percentage : 
  let original_price : ℝ := 200
  let tax_rate : ℝ := 0.10
  let total_paid : ℝ := 165
  let sale_price : ℝ := total_paid / (1 + tax_rate)
  let discount_amount : ℝ := original_price - sale_price
  let discount_percentage : ℝ := (discount_amount / original_price) * 100
  discount_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_vase_discount_percentage_l762_76289


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l762_76209

theorem addition_preserves_inequality (a b : ℝ) (h : a > b) : a + 2 > b + 2 := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l762_76209


namespace NUMINAMATH_CALUDE_point_B_coordinates_l762_76288

def point := ℝ × ℝ

def vector := ℝ × ℝ

def point_A : point := (-1, 5)

def vector_AB : vector := (6, 9)

def point_B : point := (5, 14)

def vector_between (p q : point) : vector :=
  (q.1 - p.1, q.2 - p.2)

theorem point_B_coordinates :
  vector_between point_A point_B = vector_AB :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l762_76288


namespace NUMINAMATH_CALUDE_mittens_per_box_l762_76276

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  scarves_per_box = 4 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 := by
sorry

end NUMINAMATH_CALUDE_mittens_per_box_l762_76276


namespace NUMINAMATH_CALUDE_train_speed_l762_76275

/-- The speed of a train given crossing times and platform length -/
theorem train_speed (platform_length : ℝ) (platform_crossing_time : ℝ) (man_crossing_time : ℝ) :
  platform_length = 300 →
  platform_crossing_time = 33 →
  man_crossing_time = 18 →
  (platform_length / (platform_crossing_time - man_crossing_time)) * 3.6 = 72 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l762_76275


namespace NUMINAMATH_CALUDE_age_difference_l762_76244

/-- The age difference between A and C, given the condition about total ages -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : A = C + 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l762_76244


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l762_76238

/-- Given a quadratic function f(x) = ax^2 + bx + 7, 
    if f(x+1) - f(x) = 8x - 2 for all x, then a = 4 and b = -6 -/
theorem quadratic_function_coefficients 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 7)
  (h2 : ∀ x, f (x + 1) - f x = 8 * x - 2) : 
  a = 4 ∧ b = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l762_76238


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l762_76251

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l762_76251


namespace NUMINAMATH_CALUDE_speed_in_still_water_l762_76214

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 27 →
  downstream_speed = 35 →
  (upstream_speed + downstream_speed) / 2 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l762_76214


namespace NUMINAMATH_CALUDE_quadrilateral_prism_properties_l762_76278

structure QuadrilateralPrism where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

theorem quadrilateral_prism_properties :
  ∃ (qp : QuadrilateralPrism), qp.vertices = 8 ∧ qp.edges = 12 ∧ qp.faces = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_properties_l762_76278


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l762_76252

/-- The eccentricity of an ellipse with a perpendicular bisector through a point on the ellipse --/
theorem ellipse_eccentricity_range (a b : ℝ) (h_pos : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    x^2 + y^2 = (a^2 - b^2)) → 
    Real.sqrt 2 / 2 ≤ e ∧ e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l762_76252


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l762_76232

theorem unique_four_digit_number :
  ∃! x : ℕ, 
    1000 ≤ x ∧ x < 10000 ∧
    x + (x % 10) = 5574 ∧
    x + ((x / 10) % 10) = 557 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l762_76232


namespace NUMINAMATH_CALUDE_smallest_third_side_of_right_triangle_l762_76286

theorem smallest_third_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 4 →
  c > 0 →
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 →
  3 ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_side_of_right_triangle_l762_76286


namespace NUMINAMATH_CALUDE_triangle_point_trajectory_l762_76239

theorem triangle_point_trajectory (A B C D : ℝ × ℝ) : 
  B = (-2, 0) →
  C = (2, 0) →
  D = (0, 0) →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = 3^2 →
  A.2 ≠ 0 →
  A.1^2 + A.2^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_trajectory_l762_76239


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_l762_76277

/-- Represents the number of employees with birthdays on each day of the week -/
structure BirthdayDistribution where
  wednesday : ℕ
  other : ℕ

/-- The conditions of the problem -/
def validDistribution (d : BirthdayDistribution) : Prop :=
  d.wednesday > d.other ∧
  d.wednesday + 6 * d.other = 50

/-- The theorem to prove -/
theorem min_wednesday_birthdays :
  ∀ d : BirthdayDistribution,
  validDistribution d →
  d.wednesday ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_wednesday_birthdays_l762_76277


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l762_76254

theorem factorization_of_polynomial (z : ℝ) : 
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l762_76254


namespace NUMINAMATH_CALUDE_max_residents_in_block_l762_76259

/-- Represents a block of flats -/
structure BlockOfFlats where
  totalFloors : ℕ
  apartmentsPerFloorType1 : ℕ
  apartmentsPerFloorType2 : ℕ
  maxResidentsPerApartment : ℕ

/-- Calculates the maximum number of residents in a block of flats -/
def maxResidents (block : BlockOfFlats) : ℕ :=
  let floorsType1 := block.totalFloors / 2
  let floorsType2 := block.totalFloors - floorsType1
  let totalApartments := floorsType1 * block.apartmentsPerFloorType1 + floorsType2 * block.apartmentsPerFloorType2
  totalApartments * block.maxResidentsPerApartment

/-- Theorem stating the maximum number of residents in the given block of flats -/
theorem max_residents_in_block :
  let block : BlockOfFlats := {
    totalFloors := 12,
    apartmentsPerFloorType1 := 6,
    apartmentsPerFloorType2 := 5,
    maxResidentsPerApartment := 4
  }
  maxResidents block = 264 := by
  sorry

end NUMINAMATH_CALUDE_max_residents_in_block_l762_76259


namespace NUMINAMATH_CALUDE_sum_of_xyz_l762_76297

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) : 
  x + y + z = 14 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l762_76297


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l762_76201

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l762_76201


namespace NUMINAMATH_CALUDE_f_monotonicity_and_positivity_l762_76233

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x - k * Real.log x

-- State the theorem
theorem f_monotonicity_and_positivity (k : ℝ) (h_k : k > 0) :
  (∀ x > k, ∀ y > k, x < y → f k x < f k y) ∧ 
  (∀ x ∈ Set.Ioo 0 k, ∀ y ∈ Set.Ioo 0 k, x < y → f k x > f k y) ∧
  (∀ x ≥ 1, f k x > 0) → 
  0 < k ∧ k < Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_positivity_l762_76233


namespace NUMINAMATH_CALUDE_max_y_intercept_of_even_function_l762_76263

def f (x a b : ℝ) : ℝ := x^2 + (a^2 + b^2 - 1)*x + a^2 + 2*a*b - b^2

theorem max_y_intercept_of_even_function
  (h : ∀ x, f x a b = f (-x) a b) :
  ∃ C, (∀ a b, f 0 a b ≤ C) ∧ (∃ a b, f 0 a b = C) ∧ C = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_intercept_of_even_function_l762_76263


namespace NUMINAMATH_CALUDE_binary_110110_is_54_l762_76279

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110110_is_54 :
  binary_to_decimal [true, true, false, true, true, false] = 54 := by
  sorry

end NUMINAMATH_CALUDE_binary_110110_is_54_l762_76279


namespace NUMINAMATH_CALUDE_length_of_PC_l762_76219

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the conditions
def is_right_triangle_with_internal_right_angle (t : Triangle) : Prop :=
  -- Right angle at B
  (t.A.1 - t.B.1) * (t.C.1 - t.B.1) + (t.A.2 - t.B.2) * (t.C.2 - t.B.2) = 0 ∧
  -- ∠BPC = 90°
  (t.B.1 - t.P.1) * (t.C.1 - t.P.1) + (t.B.2 - t.P.2) * (t.C.2 - t.P.2) = 0

def satisfies_length_conditions (t : Triangle) : Prop :=
  -- PA = 12
  Real.sqrt ((t.A.1 - t.P.1)^2 + (t.A.2 - t.P.2)^2) = 12 ∧
  -- PB = 8
  Real.sqrt ((t.B.1 - t.P.1)^2 + (t.B.2 - t.P.2)^2) = 8

-- The theorem to prove
theorem length_of_PC (t : Triangle) 
  (h1 : is_right_triangle_with_internal_right_angle t)
  (h2 : satisfies_length_conditions t) :
  Real.sqrt ((t.C.1 - t.P.1)^2 + (t.C.2 - t.P.2)^2) = Real.sqrt 464 :=
sorry

end NUMINAMATH_CALUDE_length_of_PC_l762_76219


namespace NUMINAMATH_CALUDE_complement_union_A_B_l762_76235

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l762_76235


namespace NUMINAMATH_CALUDE_rectangle_width_l762_76292

theorem rectangle_width (length area : ℚ) (h1 : length = 3/5) (h2 : area = 1/3) :
  area / length = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l762_76292


namespace NUMINAMATH_CALUDE_units_digit_of_composite_product_l762_76203

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_composite_product :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_composite_product_l762_76203


namespace NUMINAMATH_CALUDE_bags_filled_on_saturday_l762_76211

theorem bags_filled_on_saturday (bags_sunday : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ) : 
  bags_sunday = 4 →
  cans_per_bag = 9 →
  total_cans = 63 →
  ∃ (bags_saturday : ℕ), 
    bags_saturday * cans_per_bag + bags_sunday * cans_per_bag = total_cans ∧
    bags_saturday = 3 := by
  sorry

end NUMINAMATH_CALUDE_bags_filled_on_saturday_l762_76211


namespace NUMINAMATH_CALUDE_new_speed_calculation_l762_76265

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 252)
  (h2 : original_time = 6)
  (h3 : time_factor = 3/2) :
  let new_time := original_time * time_factor
  let new_speed := distance / new_time
  new_speed = 28 := by sorry

end NUMINAMATH_CALUDE_new_speed_calculation_l762_76265


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l762_76250

theorem initial_birds_on_fence :
  ∀ (initial_birds additional_birds total_birds : ℕ),
    additional_birds = 4 →
    total_birds = 6 →
    total_birds = initial_birds + additional_birds →
    initial_birds = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l762_76250


namespace NUMINAMATH_CALUDE_base_10_89_equals_base_4_1121_l762_76267

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n < 4 then [n]
  else (n % 4) :: toBase4 (n / 4)

/-- Converts a list of base-4 digits to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 4 * acc) 0

/-- Theorem stating that 89 in base 10 is equal to 1121 in base 4 -/
theorem base_10_89_equals_base_4_1121 :
  fromBase4 [1, 2, 1, 1] = 89 := by
  sorry

#eval toBase4 89  -- Should output [1, 2, 1, 1]
#eval fromBase4 [1, 2, 1, 1]  -- Should output 89

end NUMINAMATH_CALUDE_base_10_89_equals_base_4_1121_l762_76267


namespace NUMINAMATH_CALUDE_parabola_equation_l762_76247

/-- Given a parabola y = 2px (p > 0) and a point M on it with abscissa 3,
    if |MF| = 2p, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) :
  ∃ (M : ℝ × ℝ),
    M.1 = 3 ∧
    M.2 = 2 * p * M.1 ∧
    |M.1 - (-p/2)| + M.2 = 2 * p →
  ∀ (x y : ℝ), y = 2 * p * x ↔ y^2 = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l762_76247


namespace NUMINAMATH_CALUDE_CaCO3_molecular_weight_l762_76285

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Calcium atoms in CaCO3 -/
def num_Ca : ℕ := 1

/-- The number of Carbon atoms in CaCO3 -/
def num_C : ℕ := 1

/-- The number of Oxygen atoms in CaCO3 -/
def num_O : ℕ := 3

/-- The molecular weight of CaCO3 in g/mol -/
def molecular_weight_CaCO3 : ℝ :=
  num_Ca * atomic_weight_Ca + num_C * atomic_weight_C + num_O * atomic_weight_O

theorem CaCO3_molecular_weight :
  molecular_weight_CaCO3 = 100.09 := by sorry

end NUMINAMATH_CALUDE_CaCO3_molecular_weight_l762_76285


namespace NUMINAMATH_CALUDE_total_instruments_eq_113_l762_76249

/-- The total number of musical instruments owned by Charlie, Carli, Nick, and Daisy -/
def total_instruments (charlie_flutes charlie_horns charlie_harps charlie_drums : ℕ)
  (carli_flute_ratio carli_horn_ratio carli_drum_ratio : ℕ)
  (nick_flute_offset nick_horn_offset nick_drum_ratio nick_drum_offset : ℕ)
  (daisy_horn_denominator : ℕ) : ℕ :=
  let carli_flutes := charlie_flutes * carli_flute_ratio
  let carli_horns := charlie_horns / carli_horn_ratio
  let carli_drums := charlie_drums * carli_drum_ratio

  let nick_flutes := carli_flutes * 2 - nick_flute_offset
  let nick_horns := charlie_horns + carli_horns
  let nick_drums := carli_drums * nick_drum_ratio - nick_drum_offset

  let daisy_flutes := nick_flutes ^ 2
  let daisy_horns := (nick_horns - carli_horns) / daisy_horn_denominator
  let daisy_harps := charlie_harps
  let daisy_drums := (charlie_drums + carli_drums + nick_drums) / 3

  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_drums +
  nick_flutes + nick_horns + nick_drums +
  daisy_flutes + daisy_horns + daisy_harps + daisy_drums

theorem total_instruments_eq_113 :
  total_instruments 1 2 1 5 3 2 2 1 0 4 2 2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_total_instruments_eq_113_l762_76249


namespace NUMINAMATH_CALUDE_circle_selection_theorem_l762_76273

/-- A figure with circles arranged in a specific pattern -/
structure CircleFigure where
  total_circles : ℕ
  horizontal_lines : ℕ
  diagonal_directions : ℕ

/-- The number of ways to choose three consecutive circles in a given direction -/
def consecutive_choices (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of ways to choose three consecutive circles in the figure -/
def total_choices (fig : CircleFigure) : ℕ :=
  consecutive_choices fig.horizontal_lines +
  fig.diagonal_directions * consecutive_choices (fig.horizontal_lines - 1)

/-- The main theorem stating the number of ways to choose three consecutive circles -/
theorem circle_selection_theorem (fig : CircleFigure) 
  (h1 : fig.total_circles = 33)
  (h2 : fig.horizontal_lines = 7)
  (h3 : fig.diagonal_directions = 2) :
  total_choices fig = 57 := by
  sorry

end NUMINAMATH_CALUDE_circle_selection_theorem_l762_76273


namespace NUMINAMATH_CALUDE_sugar_per_chocolate_bar_l762_76243

/-- Given a company that produces chocolate bars, this theorem proves
    the amount of sugar needed per bar based on production rate and sugar usage. -/
theorem sugar_per_chocolate_bar
  (bars_per_minute : ℕ)
  (sugar_per_two_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_two_minutes = 108) :
  (sugar_per_two_minutes : ℚ) / ((bars_per_minute : ℚ) * 2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_per_chocolate_bar_l762_76243


namespace NUMINAMATH_CALUDE_plane_perp_from_line_perp_and_parallel_l762_76264

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem plane_perp_from_line_perp_and_parallel
  (α β : Plane) (l : Line)
  (h1 : linePerpPlane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_plane_perp_from_line_perp_and_parallel_l762_76264


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l762_76253

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 3

def probability_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem joe_fruit_probability :
  1 - probability_same_fruit = 15/16 := by sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l762_76253


namespace NUMINAMATH_CALUDE_product_digit_sum_l762_76207

theorem product_digit_sum : 
  let product := 2 * 3 * 5 * 7 * 11 * 13 * 17
  ∃ (digits : List Nat), 
    (∀ d ∈ digits, d < 10) ∧ 
    (product.repr.toList.map (λ c => c.toNat - '0'.toNat) = digits) ∧
    (digits.sum = 12) := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l762_76207


namespace NUMINAMATH_CALUDE_one_face_colored_cubes_125_l762_76202

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_divisions : ℕ
  total_small_cubes : ℕ
  colored_faces : ℕ

/-- The number of small cubes with exactly one colored face -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  c.colored_faces * (c.edge_divisions - 2) ^ 2

/-- Theorem stating the number of cubes with one colored face for a specific case -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube,
    c.edge_divisions = 5 →
    c.total_small_cubes = 125 →
    c.colored_faces = 6 →
    one_face_colored_cubes c = 54 := by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_cubes_125_l762_76202


namespace NUMINAMATH_CALUDE_smallest_multiple_l762_76212

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n % 101 = 3 ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m % 101 = 3)) → 
  n = 306 := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l762_76212


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l762_76299

theorem rectangle_perimeter_problem :
  ∀ (a b : ℕ),
    a ≠ b →
    a * b = 2 * (2 * a + 2 * b) →
    2 * (a + b) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l762_76299


namespace NUMINAMATH_CALUDE_three_times_x_greater_than_four_l762_76266

theorem three_times_x_greater_than_four (x : ℝ) : 
  (3 * x > 4) ↔ (∀ y : ℝ, y = 3 * x → y > 4) :=
by sorry

end NUMINAMATH_CALUDE_three_times_x_greater_than_four_l762_76266


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l762_76240

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 12*x →                           -- Point (x, y) is on the parabola y^2 = 12x
  (x - 3)^2 + y^2 = 9^2 →                -- Point is 9 units away from the focus (3, 0)
  (x = 6 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l762_76240


namespace NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l762_76294

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the 20th and 30th triangular numbers is 675 -/
theorem sum_of_20th_and_30th_triangular : triangular_number 20 + triangular_number 30 = 675 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_20th_and_30th_triangular_l762_76294


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l762_76269

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- State the theorem
theorem sum_of_polynomials :
  ∀ x : ℝ, p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l762_76269


namespace NUMINAMATH_CALUDE_therapy_hours_calculation_l762_76220

/-- Represents the pricing structure and patient charges for a psychologist's therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ  -- Cost of the first hour
  additional_hour : ℕ  -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  second_patient_total : ℕ  -- Total charge for the second patient (3 hours)

/-- Calculates the number of therapy hours for the first patient given the pricing structure -/
def calculate_hours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem therapy_hours_calculation (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 20)
  (h2 : pricing.second_patient_total = pricing.first_hour + 2 * pricing.additional_hour)
  (h3 : pricing.second_patient_total = 188)
  (h4 : pricing.first_patient_total = 300) :
  calculate_hours pricing = 5 :=
sorry

end NUMINAMATH_CALUDE_therapy_hours_calculation_l762_76220


namespace NUMINAMATH_CALUDE_equal_sum_black_white_cells_l762_76246

/-- Represents a cell in the Pythagorean multiplication table frame -/
structure Cell where
  row : ℕ
  col : ℕ
  value : ℕ
  isBlack : Bool

/-- Represents a rectangular frame in the Pythagorean multiplication table -/
structure Frame where
  width : ℕ
  height : ℕ
  cells : List Cell

def isPythagoreanMultiplicationTable (frame : Frame) : Prop :=
  ∀ cell ∈ frame.cells, cell.value = cell.row * cell.col

def hasOddSidedFrame (frame : Frame) : Prop :=
  Odd frame.width ∧ Odd frame.height

def hasAlternatingColors (frame : Frame) : Prop :=
  ∀ i j, i + j ≡ 0 [MOD 2] → 
    (∃ cell ∈ frame.cells, cell.row = i ∧ cell.col = j ∧ cell.isBlack)

def hasBlackCorners (frame : Frame) : Prop :=
  ∀ cell ∈ frame.cells, (cell.row = 1 ∨ cell.row = frame.height) ∧ 
                        (cell.col = 1 ∨ cell.col = frame.width) → 
                        cell.isBlack

def sumOfBlackCells (frame : Frame) : ℕ :=
  (frame.cells.filter (·.isBlack)).map (·.value) |> List.sum

def sumOfWhiteCells (frame : Frame) : ℕ :=
  (frame.cells.filter (¬·.isBlack)).map (·.value) |> List.sum

theorem equal_sum_black_white_cells (frame : Frame) 
  (h1 : isPythagoreanMultiplicationTable frame)
  (h2 : hasOddSidedFrame frame)
  (h3 : hasAlternatingColors frame)
  (h4 : hasBlackCorners frame) :
  sumOfBlackCells frame = sumOfWhiteCells frame :=
sorry

end NUMINAMATH_CALUDE_equal_sum_black_white_cells_l762_76246


namespace NUMINAMATH_CALUDE_triangle_expression_simplification_l762_76245

theorem triangle_expression_simplification
  (a b c : ℝ)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a + c - b| - |a + b + c| + |2*b + c| = c :=
sorry

end NUMINAMATH_CALUDE_triangle_expression_simplification_l762_76245


namespace NUMINAMATH_CALUDE_cubic_identity_l762_76226

theorem cubic_identity (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l762_76226


namespace NUMINAMATH_CALUDE_total_white_pieces_l762_76222

/-- The total number of pieces -/
def total_pieces : ℕ := 300

/-- The number of piles -/
def num_piles : ℕ := 100

/-- The number of pieces in each pile -/
def pieces_per_pile : ℕ := 3

/-- The number of piles with exactly one white piece -/
def piles_with_one_white : ℕ := 27

/-- The number of piles with 2 or 3 black pieces -/
def piles_with_two_or_three_black : ℕ := 42

theorem total_white_pieces :
  ∃ (piles_with_three_white : ℕ) 
    (piles_with_two_white : ℕ)
    (total_white : ℕ),
  piles_with_three_white = num_piles - piles_with_one_white - piles_with_two_or_three_black + piles_with_one_white ∧
  piles_with_two_white = num_piles - piles_with_one_white - 2 * piles_with_three_white ∧
  total_white = piles_with_one_white * 1 + piles_with_three_white * 3 + piles_with_two_white * 2 ∧
  total_white = 158 :=
by sorry

end NUMINAMATH_CALUDE_total_white_pieces_l762_76222


namespace NUMINAMATH_CALUDE_eventually_constant_function_l762_76216

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem eventually_constant_function 
  (f : ℕ+ → ℕ+) 
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f := by
sorry

end NUMINAMATH_CALUDE_eventually_constant_function_l762_76216


namespace NUMINAMATH_CALUDE_starting_lineup_count_l762_76270

-- Define the total number of players
def total_players : ℕ := 12

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the size of the starting lineup
def lineup_size : ℕ := 5

-- Theorem statement
theorem starting_lineup_count :
  (num_twins * Nat.choose (total_players - 1) (lineup_size - 1)) = 660 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l762_76270


namespace NUMINAMATH_CALUDE_avg_children_with_children_example_l762_76218

/-- The average number of children in families with children, given:
  - total_families: The total number of families
  - avg_children: The average number of children per family (including all families)
  - childless_families: The number of childless families
-/
def avg_children_with_children (total_families : ℕ) (avg_children : ℚ) (childless_families : ℕ) : ℚ :=
  (total_families : ℚ) * avg_children / ((total_families : ℚ) - (childless_families : ℚ))

/-- Theorem stating that given 15 families with an average of 3 children per family,
    and exactly 3 childless families, the average number of children in the families
    with children is 3.75 -/
theorem avg_children_with_children_example :
  avg_children_with_children 15 3 3 = 45 / 12 :=
sorry

end NUMINAMATH_CALUDE_avg_children_with_children_example_l762_76218


namespace NUMINAMATH_CALUDE_ball_trajectory_5x5_table_l762_76296

/-- Represents a square pool table --/
structure PoolTable :=
  (size : Nat)

/-- Represents a ball's trajectory on the pool table --/
structure BallTrajectory :=
  (table : PoolTable)
  (start_corner : Nat × Nat)
  (angle : Real)

/-- Represents the final state of the ball --/
structure FinalState :=
  (end_pocket : String)
  (edge_hits : Nat)
  (diagonal_squares : Nat)

/-- Main theorem about the ball's trajectory on a 5x5 pool table --/
theorem ball_trajectory_5x5_table :
  ∀ (t : PoolTable) (b : BallTrajectory),
    t.size = 5 →
    b.table = t →
    b.start_corner = (0, 0) →
    b.angle = 45 →
    ∃ (f : FinalState),
      f.end_pocket = "upper-left" ∧
      f.edge_hits = 5 ∧
      f.diagonal_squares = 23 :=
sorry

end NUMINAMATH_CALUDE_ball_trajectory_5x5_table_l762_76296


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_tangent_line_through_point_l762_76215

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 2

theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x : ℝ), (f' 0) * x = m * x ∧ m = 2 :=
sorry

theorem tangent_line_through_point :
  ∃ (m b : ℝ), ∀ (x : ℝ),
    (∃ (x₀ : ℝ), f x₀ = m * x₀ + b ∧ f' x₀ = m) ∧
    (-1 * m + b = -3) ∧
    m = 5 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_tangent_line_through_point_l762_76215


namespace NUMINAMATH_CALUDE_ruler_cost_l762_76234

/-- The cost of the ruler given the costs of other items and payment details -/
theorem ruler_cost (book_cost pen_cost total_paid change : ℕ) : 
  book_cost = 25 →
  pen_cost = 4 →
  total_paid = 50 →
  change = 20 →
  total_paid - change - (book_cost + pen_cost) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ruler_cost_l762_76234


namespace NUMINAMATH_CALUDE_jack_cookies_needed_l762_76295

/-- Represents the sales data and goals for Jack's bake sale -/
structure BakeSale where
  brownies_sold : Nat
  brownies_price : Nat
  lemon_squares_sold : Nat
  lemon_squares_price : Nat
  cookie_price : Nat
  bulk_pack_size : Nat
  bulk_pack_price : Nat
  sales_goal : Nat

/-- Calculates the minimum number of cookies needed to reach the sales goal -/
def min_cookies_needed (sale : BakeSale) : Nat :=
  sorry

/-- Theorem stating that Jack needs to sell 8 cookies to reach his goal -/
theorem jack_cookies_needed (sale : BakeSale) 
  (h1 : sale.brownies_sold = 4)
  (h2 : sale.brownies_price = 3)
  (h3 : sale.lemon_squares_sold = 5)
  (h4 : sale.lemon_squares_price = 2)
  (h5 : sale.cookie_price = 4)
  (h6 : sale.bulk_pack_size = 5)
  (h7 : sale.bulk_pack_price = 17)
  (h8 : sale.sales_goal = 50) :
  min_cookies_needed sale = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_cookies_needed_l762_76295


namespace NUMINAMATH_CALUDE_student_sums_correct_l762_76282

theorem student_sums_correct (wrong_sums correct_sums total_sums : ℕ) : 
  wrong_sums = 2 * correct_sums →
  total_sums = 36 →
  wrong_sums + correct_sums = total_sums →
  correct_sums = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_correct_l762_76282


namespace NUMINAMATH_CALUDE_equation_solution_l762_76205

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 3), (2, 3, 2), (3, 2, 2), (5, 1, 4), (5, 4, 1), (4, 1, 5), (4, 5, 1),
   (1, 4, 5), (1, 5, 4), (8, 1, 3), (8, 3, 1), (3, 1, 8), (3, 8, 1), (1, 3, 8), (1, 8, 3)}

def satisfies_equation (x y z : ℕ) : Prop :=
  (x + 1) * (y + 1) * (z + 1) = 3 * x * y * z

theorem equation_solution :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l762_76205


namespace NUMINAMATH_CALUDE_chips_left_uneaten_l762_76200

def cookies_per_dozen : ℕ := 12
def dozens_made : ℕ := 4
def chips_per_cookie : ℕ := 7
def fraction_eaten : ℚ := 1/2

theorem chips_left_uneaten : 
  (dozens_made * cookies_per_dozen * chips_per_cookie) * (1 - fraction_eaten) = 168 := by
  sorry

end NUMINAMATH_CALUDE_chips_left_uneaten_l762_76200


namespace NUMINAMATH_CALUDE_certain_seconds_proof_l762_76225

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes given in the problem -/
def given_minutes : ℕ := 6

/-- The first ratio number given in the problem -/
def ratio_1 : ℕ := 18

/-- The second ratio number given in the problem -/
def ratio_2 : ℕ := 9

/-- The certain number of seconds we need to find -/
def certain_seconds : ℕ := 720

theorem certain_seconds_proof : 
  (ratio_1 : ℚ) / certain_seconds = ratio_2 / (given_minutes * seconds_per_minute) :=
sorry

end NUMINAMATH_CALUDE_certain_seconds_proof_l762_76225
