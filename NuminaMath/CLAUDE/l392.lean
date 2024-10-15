import Mathlib

namespace NUMINAMATH_CALUDE_pizza_slices_l392_39210

theorem pizza_slices (total_slices : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) :
  total_slices = 16 →
  num_pizzas = 2 →
  total_slices = num_pizzas * slices_per_pizza →
  slices_per_pizza = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_l392_39210


namespace NUMINAMATH_CALUDE_initial_water_percentage_l392_39249

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 70)
  (h2 : added_water = 14)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) = final_water_percentage / 100) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l392_39249


namespace NUMINAMATH_CALUDE_wire_length_ratio_l392_39284

/-- The length of one piece of wire used in Bonnie's cube frame -/
def bonnie_wire_length : ℝ := 8

/-- The number of wire pieces used in Bonnie's cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of one piece of wire used in Roark's unit cube frames -/
def roark_wire_length : ℝ := 2

/-- The volume of Bonnie's cube -/
def bonnie_cube_volume : ℝ := bonnie_wire_length ^ 3

/-- The volume of one of Roark's unit cubes -/
def roark_unit_cube_volume : ℝ := roark_wire_length ^ 3

/-- The number of wire pieces needed for one of Roark's unit cube frames -/
def roark_wire_count_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_count * bonnie_wire_length) / 
  (((bonnie_cube_volume / roark_unit_cube_volume) : ℝ) * 
   (roark_wire_count_per_cube : ℝ) * roark_wire_length) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l392_39284


namespace NUMINAMATH_CALUDE_f_max_min_difference_l392_39274

noncomputable def f (x : ℝ) : ℝ := x * |3 - x| - (x - 3) * |x|

theorem f_max_min_difference :
  ∃ (max min : ℝ), (∀ x, f x ≤ max) ∧ (∀ x, f x ≥ min) ∧ (max - min = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l392_39274


namespace NUMINAMATH_CALUDE_range_of_m_l392_39252

open Set

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
  x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ¬∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the range of m
def range_m : Set ℝ := Ioc 1 2 ∪ Ici 3

-- Theorem statement
theorem range_of_m : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ 
  (∀ m : ℝ, m ∈ range_m ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l392_39252


namespace NUMINAMATH_CALUDE_proposition_implication_l392_39243

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l392_39243


namespace NUMINAMATH_CALUDE_computer_profit_calculation_l392_39241

theorem computer_profit_calculation (C : ℝ) :
  (C + 0.4 * C = 2240) → (C + 0.6 * C = 2560) := by
  sorry

end NUMINAMATH_CALUDE_computer_profit_calculation_l392_39241


namespace NUMINAMATH_CALUDE_hiking_trip_solution_l392_39283

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  men_count : ℕ
  women_count : ℕ
  total_weight : ℝ
  men_backpack_weight : ℝ
  women_backpack_weight : ℝ

/-- Checks if the hiking trip satisfies the given conditions -/
def is_valid_hiking_trip (trip : HikingTrip) : Prop :=
  trip.men_count = 2 ∧
  trip.women_count = 3 ∧
  trip.total_weight = 44 ∧
  trip.men_count * trip.men_backpack_weight + trip.women_count * trip.women_backpack_weight = trip.total_weight ∧
  trip.men_backpack_weight + trip.women_backpack_weight + trip.women_backpack_weight / 2 = 
    trip.women_backpack_weight + trip.men_backpack_weight / 2

theorem hiking_trip_solution (trip : HikingTrip) :
  is_valid_hiking_trip trip → trip.women_backpack_weight = 8 ∧ trip.men_backpack_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_hiking_trip_solution_l392_39283


namespace NUMINAMATH_CALUDE_exact_rolls_probability_l392_39248

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def dice : ℕ := 8

/-- The number of dice we want to show a specific number -/
def target : ℕ := 4

/-- The probability of rolling exactly 'target' number of twos 
    when rolling 'dice' number of 'sides'-sided dice -/
def probability : ℚ := 168070 / 16777216

theorem exact_rolls_probability : 
  (Nat.choose dice target * (1 / sides) ^ target * ((sides - 1) / sides) ^ (dice - target)) = probability := by
  sorry

end NUMINAMATH_CALUDE_exact_rolls_probability_l392_39248


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l392_39232

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The additional length needed to cross the river in inches -/
def additional_length : ℕ := 192

/-- The current length of the bridge in inches -/
def bridge_length : ℕ := river_width - additional_length

theorem bridge_length_calculation :
  bridge_length = 295 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l392_39232


namespace NUMINAMATH_CALUDE_sine_bounds_l392_39286

theorem sine_bounds (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  (Real.sqrt 2 / 2) * x ≤ Real.sin x ∧ Real.sin x ≤ x := by
  sorry

end NUMINAMATH_CALUDE_sine_bounds_l392_39286


namespace NUMINAMATH_CALUDE_base4_1212_is_102_l392_39291

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_1212_is_102 : base4_to_decimal [2, 1, 2, 1] = 102 := by
  sorry

end NUMINAMATH_CALUDE_base4_1212_is_102_l392_39291


namespace NUMINAMATH_CALUDE_disjunction_and_negation_imply_right_true_l392_39202

theorem disjunction_and_negation_imply_right_true (p q : Prop) :
  (p ∨ q) → ¬p → q := by sorry

end NUMINAMATH_CALUDE_disjunction_and_negation_imply_right_true_l392_39202


namespace NUMINAMATH_CALUDE_blind_students_count_l392_39280

theorem blind_students_count (total : ℕ) (deaf_ratio : ℕ) : 
  total = 180 → deaf_ratio = 3 → 
  ∃ (blind : ℕ), blind = 45 ∧ total = blind + deaf_ratio * blind :=
by
  sorry

end NUMINAMATH_CALUDE_blind_students_count_l392_39280


namespace NUMINAMATH_CALUDE_curve_arc_length_l392_39272

noncomputable def arcLength (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem curve_arc_length :
  let ρ := fun φ => 3 * Real.exp (3 * φ / 4)
  let φ₁ := -π / 2
  let φ₂ := π / 2
  arcLength ρ φ₁ φ₂ = 10 * Real.sinh (3 * π / 8) := by
  sorry

end NUMINAMATH_CALUDE_curve_arc_length_l392_39272


namespace NUMINAMATH_CALUDE_monotonically_decreasing_quadratic_l392_39219

/-- A function f is monotonically decreasing on an interval [a, b] if for all x, y in [a, b] with x ≤ y, we have f(x) ≥ f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

/-- The theorem statement -/
theorem monotonically_decreasing_quadratic (a : ℝ) :
  MonotonicallyDecreasing (fun x => a * x^2 - 2 * x + 1) 1 10 ↔ a ≤ 1/10 :=
sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_quadratic_l392_39219


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l392_39299

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ

/-- The diagonal length of an isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal length of the specific isosceles trapezoid is 2√52 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 27, base2 := 11, leg := 12 }
  diagonal_length t = 2 * Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonal_l392_39299


namespace NUMINAMATH_CALUDE_max_areas_formula_l392_39203

/-- Represents a circular disk configuration -/
structure DiskConfiguration where
  n : ℕ
  radii_count : ℕ
  has_secant : Bool
  has_chord : Bool
  chord_intersects_secant : Bool

/-- The maximum number of non-overlapping areas in a disk configuration -/
def max_areas (config : DiskConfiguration) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (config : DiskConfiguration) 
  (h1 : config.n > 0)
  (h2 : config.radii_count = 2 * config.n)
  (h3 : config.has_secant = true)
  (h4 : config.has_chord = true)
  (h5 : config.chord_intersects_secant = false) :
  max_areas config = 4 * config.n - 1 :=
sorry

end NUMINAMATH_CALUDE_max_areas_formula_l392_39203


namespace NUMINAMATH_CALUDE_A_intersect_B_l392_39221

def A : Set ℝ := {-3, -2, 0, 2}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l392_39221


namespace NUMINAMATH_CALUDE_original_number_problem_l392_39293

theorem original_number_problem (x : ℝ) : x * 1.2 = 480 → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l392_39293


namespace NUMINAMATH_CALUDE_departure_representation_l392_39235

/-- Represents the change in grain quantity -/
inductive GrainChange
| Arrival (amount : ℕ)
| Departure (amount : ℕ)

/-- Records the change in grain quantity -/
def record (change : GrainChange) : ℤ :=
  match change with
  | GrainChange.Arrival amount => amount
  | GrainChange.Departure amount => -amount

/-- Theorem: If arrival of 30 tons is recorded as +30, then -30 represents departure of 30 tons -/
theorem departure_representation :
  (record (GrainChange.Arrival 30) = 30) →
  (record (GrainChange.Departure 30) = -30) :=
by
  sorry

end NUMINAMATH_CALUDE_departure_representation_l392_39235


namespace NUMINAMATH_CALUDE_business_subscription_problem_l392_39207

/-- Proves that given the conditions of the business subscription problem, 
    the total amount subscribed is 50,000 Rs. -/
theorem business_subscription_problem 
  (a b c : ℕ) 
  (h1 : a = b + 4000)
  (h2 : b = c + 5000)
  (total_profit : ℕ)
  (h3 : total_profit = 36000)
  (a_profit : ℕ)
  (h4 : a_profit = 15120)
  (h5 : a_profit * (a + b + c) = a * total_profit) :
  a + b + c = 50000 := by
  sorry

end NUMINAMATH_CALUDE_business_subscription_problem_l392_39207


namespace NUMINAMATH_CALUDE_mary_stickers_l392_39205

/-- The number of stickers Mary brought to class -/
def stickers_brought : ℕ := 50

/-- The number of Mary's friends -/
def num_friends : ℕ := 5

/-- The number of stickers Mary gave to each friend -/
def stickers_per_friend : ℕ := 4

/-- The number of stickers Mary gave to each other student -/
def stickers_per_other : ℕ := 2

/-- The number of stickers Mary has left over -/
def stickers_leftover : ℕ := 8

/-- The total number of students in the class, including Mary -/
def total_students : ℕ := 17

theorem mary_stickers :
  stickers_brought =
    num_friends * stickers_per_friend +
    (total_students - 1 - num_friends) * stickers_per_other +
    stickers_leftover :=
by sorry

end NUMINAMATH_CALUDE_mary_stickers_l392_39205


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l392_39251

-- Define the function y = mx^2 - mx - 1
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | y (1/2) x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Part 2
theorem solution_set_part2 (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1/m}
    else
      {x : ℝ | x < 1/m ∨ x > 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l392_39251


namespace NUMINAMATH_CALUDE_star_properties_l392_39227

/-- The operation "*" for any two numbers -/
noncomputable def star (m : ℝ) (x y : ℝ) : ℝ := (4 * x * y) / (m * x + 3 * y)

/-- Theorem stating the properties of the "*" operation -/
theorem star_properties :
  ∃ m : ℝ, (star m 1 2 = 1) ∧ (m = 2) ∧ (star m 3 12 = 24/7) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l392_39227


namespace NUMINAMATH_CALUDE_runner_speed_l392_39245

/-- Calculates the speed of a runner overtaking a parade -/
theorem runner_speed (parade_length : ℝ) (parade_speed : ℝ) (runner_time : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_time = 0.222222222222 →
  parade_length / runner_time = 9 :=
by sorry

end NUMINAMATH_CALUDE_runner_speed_l392_39245


namespace NUMINAMATH_CALUDE_max_value_theorem_l392_39211

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*(Real.sqrt 2) ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l392_39211


namespace NUMINAMATH_CALUDE_sqrt_15_minus_one_over_three_lt_one_l392_39240

theorem sqrt_15_minus_one_over_three_lt_one :
  (Real.sqrt 15 - 1) / 3 < 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_one_over_three_lt_one_l392_39240


namespace NUMINAMATH_CALUDE_ad_value_l392_39254

/-- Given two-digit numbers ab and cd, and that 1ab is a three-digit number -/
def two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

/-- The theorem statement -/
theorem ad_value (a b c d : ℕ) 
  (h1 : two_digit (10 * a + b))
  (h2 : two_digit (10 * c + d))
  (h3 : three_digit (100 + 10 * a + b))
  (h4 : 10 * a + b = 10 * c + d + 24)
  (h5 : 100 + 10 * a + b = 100 * c + 10 * d + 1 + 15) :
  10 * a + d = 32 := by
sorry

end NUMINAMATH_CALUDE_ad_value_l392_39254


namespace NUMINAMATH_CALUDE_function_properties_l392_39250

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (x : ℝ) := -a * x + b + a * x * Real.log x

theorem function_properties :
  a ≠ 0 →
  f e = 2 →
  (b = 2) ∧
  (a > 0 →
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂)) ∧
  (a < 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂)) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l392_39250


namespace NUMINAMATH_CALUDE_cube_surface_area_l392_39279

/-- The surface area of a cube with edge length 20 cm is 2400 cm². -/
theorem cube_surface_area : 
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length * edge_length
  surface_area = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l392_39279


namespace NUMINAMATH_CALUDE_exists_greater_term_l392_39216

/-- Two sequences of positive reals satisfying given recurrence relations -/
def SequencePair (x y : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0 ∧ y n > 0) ∧
  (∀ n, x (n + 2) = x n + (x (n + 1))^2) ∧
  (∀ n, y (n + 2) = (y n)^2 + y (n + 1)) ∧
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

/-- There exists a k such that x_k > y_k -/
theorem exists_greater_term (x y : ℕ → ℝ) (h : SequencePair x y) :
  ∃ k, x k > y k := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_term_l392_39216


namespace NUMINAMATH_CALUDE_container_volume_ratio_l392_39242

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℚ), v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l392_39242


namespace NUMINAMATH_CALUDE_cube_strictly_increasing_l392_39270

theorem cube_strictly_increasing (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_strictly_increasing_l392_39270


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l392_39260

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x + 15 = 51 ↔ x = p ∨ x = q) →
  p ≥ q →
  3*p + 2*q = 15 + 3*Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l392_39260


namespace NUMINAMATH_CALUDE_sixth_root_of_unity_product_l392_39297

theorem sixth_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_root_of_unity_product_l392_39297


namespace NUMINAMATH_CALUDE_special_function_property_l392_39298

/-- A function satisfying the given property for all real numbers -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, d^2 * g c = c^2 * g d

theorem special_function_property (g : ℝ → ℝ) (h : SpecialFunction g) (h3 : g 3 ≠ 0) :
  (g 6 + g 2) / g 3 = 40/9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l392_39298


namespace NUMINAMATH_CALUDE_inequality_holds_l392_39209

theorem inequality_holds (a b : ℕ+) : a^3 + (a+b)^2 + b ≠ b^3 + a + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l392_39209


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l392_39256

/-- The cost of pens and pencils -/
def CostProblem (pen_cost : ℚ) (pencil_cost : ℚ) : Prop :=
  -- Condition 1: The cost of 3 pens and 5 pencils is Rs. 260
  3 * pen_cost + 5 * pencil_cost = 260 ∧
  -- Condition 2: The cost ratio of one pen to one pencil is 5:1
  pen_cost = 5 * pencil_cost

/-- The cost of one dozen pens is Rs. 780 -/
theorem cost_of_dozen_pens 
  (pen_cost : ℚ) (pencil_cost : ℚ) 
  (h : CostProblem pen_cost pencil_cost) : 
  12 * pen_cost = 780 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l392_39256


namespace NUMINAMATH_CALUDE_tank_drainage_rate_l392_39262

/-- Prove that given the conditions of the tank filling problem, 
    the drainage rate of pipe C is 20 liters per minute. -/
theorem tank_drainage_rate 
  (tank_capacity : ℕ) 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (total_time : ℕ) 
  (h1 : tank_capacity = 800)
  (h2 : fill_rate_A = 40)
  (h3 : fill_rate_B = 30)
  (h4 : total_time = 48)
  : ∃ (drain_rate_C : ℕ), 
    drain_rate_C = 20 ∧ 
    (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) = tank_capacity :=
by sorry

end NUMINAMATH_CALUDE_tank_drainage_rate_l392_39262


namespace NUMINAMATH_CALUDE_triangle_area_approx_l392_39225

/-- The area of a triangle with sides 30, 26, and 10 is approximately 126.72 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 26
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  126.71 < area ∧ area < 126.73 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l392_39225


namespace NUMINAMATH_CALUDE_citric_acid_weight_l392_39259

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_molecular_weight : ℝ := 192.12

/-- Theorem stating that the molecular weight of Citric acid is 192.12 g/mol -/
theorem citric_acid_weight : citric_acid_molecular_weight = 192.12 := by sorry

end NUMINAMATH_CALUDE_citric_acid_weight_l392_39259


namespace NUMINAMATH_CALUDE_sales_growth_equation_correct_l392_39201

/-- Represents the sales growth scenario of a product over two years -/
structure SalesGrowth where
  initialSales : ℝ
  salesIncrease : ℝ
  growthRate : ℝ

/-- The equation for the sales growth scenario is correct -/
def isCorrectEquation (sg : SalesGrowth) : Prop :=
  20 * (1 + sg.growthRate)^2 - 20 = 3.12

/-- The given sales data matches the equation -/
theorem sales_growth_equation_correct (sg : SalesGrowth) 
  (h1 : sg.initialSales = 200000)
  (h2 : sg.salesIncrease = 31200) :
  isCorrectEquation sg := by
  sorry

end NUMINAMATH_CALUDE_sales_growth_equation_correct_l392_39201


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l392_39215

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  ArithmeticSequence b →
  a 1 + b 1 = 7 →
  a 3 + b 3 = 21 →
  a 5 + b 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l392_39215


namespace NUMINAMATH_CALUDE_find_k_l392_39271

theorem find_k (k : ℚ) (h : 56 / k = 4) : k = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l392_39271


namespace NUMINAMATH_CALUDE_ellipse_condition_l392_39206

/-- 
Given the equation x^2 + 9y^2 - 6x + 18y = k, 
this theorem states that it represents a non-degenerate ellipse 
if and only if k > -18.
-/
theorem ellipse_condition (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 9*y^2 - 6*x + 18*y = k) → 
  (∃ a b h1 h2 : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, (x - h1)^2 / a^2 + (y - h2)^2 / b^2 = 1) ↔ 
  k > -18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l392_39206


namespace NUMINAMATH_CALUDE_largest_factor_and_smallest_multiple_of_18_l392_39247

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ n : ℕ, n ≤ 18 ∧ 18 % n = 0 ∧ ∀ m : ℕ, m ≤ 18 ∧ 18 % m = 0 → m ≤ n) ∧
  (∃ k : ℕ, 18 ∣ k ∧ ∀ j : ℕ, 18 ∣ j → k ≤ j) :=
by sorry

end NUMINAMATH_CALUDE_largest_factor_and_smallest_multiple_of_18_l392_39247


namespace NUMINAMATH_CALUDE_wells_garden_rows_l392_39226

/-- The number of rows in Mr. Wells' garden -/
def num_rows : ℕ := 50

/-- The number of flowers in each row -/
def flowers_per_row : ℕ := 400

/-- The percentage of flowers cut -/
def cut_percentage : ℚ := 60 / 100

/-- The number of flowers remaining after cutting -/
def remaining_flowers : ℕ := 8000

/-- Theorem stating that the number of rows is correct given the conditions -/
theorem wells_garden_rows :
  num_rows * flowers_per_row * (1 - cut_percentage) = remaining_flowers :=
sorry

end NUMINAMATH_CALUDE_wells_garden_rows_l392_39226


namespace NUMINAMATH_CALUDE_intersecting_line_properties_l392_39229

/-- A line that intersects both positive x-axis and positive y-axis -/
structure IntersectingLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line intersects the positive x-axis -/
  pos_x_intersect : ∃ x : ℝ, x > 0 ∧ m * x + b = 0
  /-- The line intersects the positive y-axis -/
  pos_y_intersect : b > 0

/-- Theorem: An intersecting line has negative slope and positive y-intercept -/
theorem intersecting_line_properties (l : IntersectingLine) : l.m < 0 ∧ l.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_line_properties_l392_39229


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_l392_39253

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (planesIntersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skewLines : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient
  (α β : Plane) (m n : Line)
  (h1 : ¬ planesIntersect α β)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  (∀ α' β' m' n', planesIntersect α' β' → skewLines m' n' → perpendicular m' α' → perpendicular n' β' → True) ∧
  (∃ α' β' m' n', planesIntersect α' β' ∧ ¬ skewLines m' n' ∧ perpendicular m' α' ∧ perpendicular n' β') :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_l392_39253


namespace NUMINAMATH_CALUDE_square_root_sum_simplification_l392_39237

theorem square_root_sum_simplification :
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) - 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_simplification_l392_39237


namespace NUMINAMATH_CALUDE_total_students_in_classes_l392_39290

theorem total_students_in_classes (class_a class_b : ℕ) : 
  (80 * class_a = 90 * (class_a - 8) + 20 * 8) →
  (70 * class_b = 85 * (class_b - 6) + 30 * 6) →
  class_a + class_b = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_classes_l392_39290


namespace NUMINAMATH_CALUDE_probability_concentric_circles_l392_39264

/-- The probability of a randomly chosen point from a circle with radius 3 
    lying within a concentric circle with radius 1 is 1/9. -/
theorem probability_concentric_circles : 
  let outer_radius : ℝ := 3
  let inner_radius : ℝ := 1
  let outer_area := π * outer_radius^2
  let inner_area := π * inner_radius^2
  (inner_area / outer_area : ℝ) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_probability_concentric_circles_l392_39264


namespace NUMINAMATH_CALUDE_intersecting_lines_regions_l392_39281

/-- The number of regions created by n intersecting lines -/
def num_regions (n : ℕ) : ℕ := (n * n - n + 2) / 2 + 1

/-- Theorem stating that for any n ≥ 5, there exists a configuration of n intersecting lines
    that divides the plane into at least n regions -/
theorem intersecting_lines_regions (n : ℕ) (h : n ≥ 5) :
  num_regions n ≥ n :=
by sorry

end NUMINAMATH_CALUDE_intersecting_lines_regions_l392_39281


namespace NUMINAMATH_CALUDE_rain_is_random_event_l392_39295

/-- An event is random if its probability is strictly between 0 and 1 -/
def is_random_event (p : ℝ) : Prop := 0 < p ∧ p < 1

/-- The probability of rain in Xiangyang tomorrow -/
def rain_probability : ℝ := 0.75

theorem rain_is_random_event : is_random_event rain_probability := by
  sorry

end NUMINAMATH_CALUDE_rain_is_random_event_l392_39295


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l392_39282

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The number of peaches Mike has now -/
def current_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := current_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l392_39282


namespace NUMINAMATH_CALUDE_jason_pears_l392_39267

theorem jason_pears (total pears_keith pears_mike : ℕ) 
  (h_total : total = 105)
  (h_keith : pears_keith = 47)
  (h_mike : pears_mike = 12) :
  total - (pears_keith + pears_mike) = 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_pears_l392_39267


namespace NUMINAMATH_CALUDE_log_sum_squares_primes_l392_39288

theorem log_sum_squares_primes (a b : ℕ) (ha : Prime a) (hb : Prime b) 
  (hab : a ≠ b) (ha_gt_2 : a > 2) (hb_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 := by
sorry

end NUMINAMATH_CALUDE_log_sum_squares_primes_l392_39288


namespace NUMINAMATH_CALUDE_inequality_proof_l392_39239

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l392_39239


namespace NUMINAMATH_CALUDE_two_fifths_divided_by_three_l392_39261

theorem two_fifths_divided_by_three : (2 : ℚ) / 5 / 3 = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_divided_by_three_l392_39261


namespace NUMINAMATH_CALUDE_sons_age_l392_39273

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 7 * (son_age - 8) →
  father_age / 4 = 14 →
  son_age = 16 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l392_39273


namespace NUMINAMATH_CALUDE_circles_separated_l392_39231

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (-2, -1)
def center₂ : ℝ × ℝ := (2, 1)

-- Define the radius of the circles
def radius : ℝ := 2

-- Theorem: The circles C₁ and C₂ are separated
theorem circles_separated : 
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) → 
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 > (radius + radius)^2 :=
sorry

end NUMINAMATH_CALUDE_circles_separated_l392_39231


namespace NUMINAMATH_CALUDE_convex_figure_inequalities_isoperimetric_inequality_l392_39222

/-- A convex figure in a plane -/
class ConvexFigure where
  -- Perimeter of the convex figure
  perimeter : ℝ
  -- Area of the convex figure
  area : ℝ
  -- Radius of the inscribed circle
  inscribed_radius : ℝ
  -- Radius of the circumscribed circle
  circumscribed_radius : ℝ
  -- Assumption that the figure is convex
  convex : True

/-- The main theorem stating the inequalities for convex figures -/
theorem convex_figure_inequalities (F : ConvexFigure) :
  let P := F.perimeter
  let S := F.area
  let r := F.inscribed_radius
  let R := F.circumscribed_radius
  (P^2 - 4 * Real.pi * S ≥ (P - 2 * Real.pi * r)^2) ∧
  (P^2 - 4 * Real.pi * S ≥ (2 * Real.pi * R - P)^2) := by
  sorry

/-- Corollary: isoperimetric inequality for planar convex figures -/
theorem isoperimetric_inequality (F : ConvexFigure) :
  F.area / F.perimeter^2 ≤ 1 / (4 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_convex_figure_inequalities_isoperimetric_inequality_l392_39222


namespace NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l392_39230

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral in the plane -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A quadrilateral is a rectangle if its diagonals bisect each other -/
def isRectangle (quad : Quadrilateral) : Prop :=
  let midpointAC := Point.mk ((quad.A.x + quad.C.x) / 2) ((quad.A.y + quad.C.y) / 2)
  let midpointBD := Point.mk ((quad.B.x + quad.D.x) / 2) ((quad.B.y + quad.D.y) / 2)
  midpointAC = midpointBD

/-- Main theorem -/
theorem quadrilateral_is_rectangle (quad : Quadrilateral) :
  (∀ M N P : Point, ¬collinear M N P →
    distanceSquared M quad.A + distanceSquared M quad.C =
    distanceSquared M quad.B + distanceSquared M quad.D) →
  isRectangle quad :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l392_39230


namespace NUMINAMATH_CALUDE_certain_amount_problem_l392_39234

theorem certain_amount_problem : ∃ x : ℤ, 7 * 5 - 15 = 2 * 5 + x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l392_39234


namespace NUMINAMATH_CALUDE_concert_attendance_problem_l392_39224

theorem concert_attendance_problem (total_attendance : ℕ) (adult_cost child_cost total_receipts : ℚ) 
  (h1 : total_attendance = 578)
  (h2 : adult_cost = 2)
  (h3 : child_cost = 3/2)
  (h4 : total_receipts = 985) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_cost * adults + child_cost * children = total_receipts ∧
    adults = 236 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_problem_l392_39224


namespace NUMINAMATH_CALUDE_expression_a_result_l392_39257

theorem expression_a_result : 
  (7 * (2 / 3) + 16 * (5 / 12)) = 34 / 3 := by sorry

end NUMINAMATH_CALUDE_expression_a_result_l392_39257


namespace NUMINAMATH_CALUDE_equality_of_expressions_l392_39244

theorem equality_of_expressions :
  (-2^7 = (-2)^7) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-3 * 2^3 ≠ -3^2 * 2) ∧
  (-((-3)^2) ≠ -((-2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l392_39244


namespace NUMINAMATH_CALUDE_solve_for_b_and_c_l392_39277

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_for_b_and_c (a b c : ℝ) : 
  A a ≠ B b c →
  A a ∪ B b c = {-3, 4} →
  A a ∩ B b c = {-3} →
  b = 3 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_solve_for_b_and_c_l392_39277


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l392_39263

theorem ceiling_sqrt_count : 
  (Finset.range 325 \ Finset.range 290).card = 35 := by sorry

#check ceiling_sqrt_count

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l392_39263


namespace NUMINAMATH_CALUDE_symmetric_sequence_theorem_l392_39213

/-- A symmetric sequence of 7 terms -/
def SymmetricSequence (b : Fin 7 → ℝ) : Prop :=
  ∀ k, k < 7 → b k = b (6 - k)

/-- The first 4 terms form an arithmetic sequence -/
def ArithmeticSequence (b : Fin 7 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ k, k < 3 → b (k + 1) - b k = d

/-- The theorem statement -/
theorem symmetric_sequence_theorem (b : Fin 7 → ℝ) 
  (h_symmetric : SymmetricSequence b)
  (h_arithmetic : ArithmeticSequence b)
  (h_b1 : b 0 = 2)
  (h_sum : b 1 + b 3 = 16) :
  b = ![2, 5, 8, 11, 8, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_theorem_l392_39213


namespace NUMINAMATH_CALUDE_tv_watch_time_two_weeks_l392_39217

/-- Calculates the total hours of TV watched in two weeks -/
def tvWatchTimeInTwoWeeks (minutesPerDay : ℕ) (daysPerWeek : ℕ) : ℚ :=
  let minutesPerWeek : ℕ := minutesPerDay * daysPerWeek
  let hoursPerWeek : ℚ := minutesPerWeek / 60
  hoursPerWeek * 2

/-- Theorem: Children watching 45 minutes of TV per day, 4 days a week, watch 6 hours in two weeks -/
theorem tv_watch_time_two_weeks :
  tvWatchTimeInTwoWeeks 45 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tv_watch_time_two_weeks_l392_39217


namespace NUMINAMATH_CALUDE_diamond_value_l392_39223

theorem diamond_value :
  ∀ (diamond : ℕ),
  diamond < 10 →
  diamond * 6 + 5 = diamond * 9 + 2 →
  diamond = 1 := by
sorry

end NUMINAMATH_CALUDE_diamond_value_l392_39223


namespace NUMINAMATH_CALUDE_fraction_modification_l392_39285

theorem fraction_modification (a b c d k : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : d ≠ 0) (h4 : k ≠ 0) (h5 : k ≠ 1) :
  let x := (b * c - a * d) / (k * d - c)
  (a + k * x) / (b + x) = c / d :=
by sorry

end NUMINAMATH_CALUDE_fraction_modification_l392_39285


namespace NUMINAMATH_CALUDE_sam_coupons_l392_39258

/-- Calculates the number of coupons Sam used when buying tuna cans. -/
def calculate_coupons (num_cans : ℕ) (can_cost : ℕ) (coupon_value : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let total_cost := num_cans * can_cost
  let savings := total_cost - total_spent
  savings / coupon_value

/-- Proves that Sam had 5 coupons given the problem conditions. -/
theorem sam_coupons :
  calculate_coupons 9 175 25 2000 550 = 5 := by
  sorry

#eval calculate_coupons 9 175 25 2000 550

end NUMINAMATH_CALUDE_sam_coupons_l392_39258


namespace NUMINAMATH_CALUDE_principal_correct_l392_39236

/-- Calculates the final amount after compound interest with varying rates and additional investments -/
def final_amount (principal : ℝ) (initial_rate : ℝ) (rate_increase : ℝ) (years : ℝ) (annual_investment : ℝ) : ℝ :=
  let first_year := principal * (1 + initial_rate) + annual_investment
  let second_year := first_year * (1 + (initial_rate + rate_increase)) + annual_investment
  second_year * (1 + (initial_rate + 2 * rate_increase) * (years - 2))

/-- The principal amount is correct if it results in the expected final amount -/
theorem principal_correct (principal : ℝ) : 
  abs (final_amount principal 0.07 0.02 2.4 200 - 1120) < 0.01 → 
  abs (principal - 556.25) < 0.01 := by
  sorry

#eval final_amount 556.25 0.07 0.02 2.4 200

end NUMINAMATH_CALUDE_principal_correct_l392_39236


namespace NUMINAMATH_CALUDE_batsman_highest_score_l392_39218

-- Define the given conditions
def total_innings : ℕ := 46
def average : ℚ := 62
def score_difference : ℕ := 150
def average_excluding_extremes : ℚ := 58

-- Define the theorem
theorem batsman_highest_score :
  ∃ (highest lowest : ℕ),
    (highest - lowest = score_difference) ∧
    (highest + lowest = total_innings * average - (total_innings - 2) * average_excluding_extremes) ∧
    (highest = 225) := by
  sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l392_39218


namespace NUMINAMATH_CALUDE_golf_distance_ratio_l392_39294

/-- Proves that the ratio of the distance traveled on the second turn to the distance traveled on the first turn is 1/2 in a golf scenario. -/
theorem golf_distance_ratio
  (total_distance : ℝ)
  (first_turn_distance : ℝ)
  (overshoot_distance : ℝ)
  (h1 : total_distance = 250)
  (h2 : first_turn_distance = 180)
  (h3 : overshoot_distance = 20)
  : (total_distance - first_turn_distance + overshoot_distance) / first_turn_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_golf_distance_ratio_l392_39294


namespace NUMINAMATH_CALUDE_circle_properties_l392_39233

/-- Given a circle with equation x^2 - 8x - y^2 + 2y = 6, prove its properties. -/
theorem circle_properties :
  let E : Set (ℝ × ℝ) := {p | let (x, y) := p; x^2 - 8*x - y^2 + 2*y = 6}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ E ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c = 4 ∧
    d = 1 ∧
    s^2 = 11 ∧
    c + d + s = 5 + Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l392_39233


namespace NUMINAMATH_CALUDE_product_sum_theorem_l392_39265

theorem product_sum_theorem (a b c : ℝ) (h : a * b * c = 1) :
  a / (a * b + a + 1) + b / (b * c + b + 1) + c / (c * a + c + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l392_39265


namespace NUMINAMATH_CALUDE_function_comparison_l392_39204

open Set

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (h_eq : f a = g a)
  (h_deriv : ∀ x ∈ Set.Ioo a b, deriv f x > deriv g x) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l392_39204


namespace NUMINAMATH_CALUDE_last_digit_base_5_l392_39275

theorem last_digit_base_5 (n : ℕ) (h : n = 119) : n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_base_5_l392_39275


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l392_39287

theorem chess_tournament_participants (n : ℕ) (m : ℕ) : 
  (2 : ℕ) + n = number_of_participants →
  8 = points_scored_by_7th_graders →
  m * n = points_scored_by_8th_graders →
  m * n + 8 = total_points_scored →
  (n + 2) * (n + 1) / 2 = total_games_played →
  total_points_scored = total_games_played →
  n = 7 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l392_39287


namespace NUMINAMATH_CALUDE_negative_square_l392_39289

theorem negative_square : -3^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_l392_39289


namespace NUMINAMATH_CALUDE_not_divisible_by_qplus1_l392_39255

theorem not_divisible_by_qplus1 (q : ℕ) (hodd : Odd q) (hq : q > 2) :
  ¬ (q + 1 ∣ (q + 1)^((q - 1)/2) + 2) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_qplus1_l392_39255


namespace NUMINAMATH_CALUDE_circumcenter_distance_theorem_l392_39200

-- Define a structure for a triangle with its circumcircle properties
structure TriangleWithCircumcircle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles opposite to sides a, b, c respectively
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Radius of the circumscribed circle
  r : ℝ
  -- Distances from circumcenter to sides a, b, c respectively
  pa : ℝ
  pb : ℝ
  pc : ℝ
  -- Conditions for a valid triangle
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Relationship between sides and angles
  sine_law_a : a = 2 * r * Real.sin α
  sine_law_b : b = 2 * r * Real.sin β
  sine_law_c : c = 2 * r * Real.sin γ

-- Theorem statement
theorem circumcenter_distance_theorem (t : TriangleWithCircumcircle) :
  t.pa * Real.sin t.α + t.pb * Real.sin t.β + t.pc * Real.sin t.γ =
  2 * t.r * Real.sin t.α * Real.sin t.β * Real.sin t.γ :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_distance_theorem_l392_39200


namespace NUMINAMATH_CALUDE_graph_is_three_lines_lines_not_concurrent_l392_39228

/-- The equation representing the graph -/
def graph_equation (x y : ℝ) : Prop :=
  x^2 * (x + y + 2) = y^2 * (x + y + 2)

/-- Definition of a line in 2D space -/
def is_line (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ 
  S = {(x, y) | a * x + b * y + c = 0}

/-- The graph consists of three distinct lines -/
theorem graph_is_three_lines :
  ∃ (L₁ L₂ L₃ : Set (ℝ × ℝ)),
    (is_line L₁ ∧ is_line L₂ ∧ is_line L₃) ∧
    (L₁ ≠ L₂ ∧ L₁ ≠ L₃ ∧ L₂ ≠ L₃) ∧
    (∀ x y, graph_equation x y ↔ (x, y) ∈ L₁ ∪ L₂ ∪ L₃) :=
sorry

/-- The three lines do not all pass through a common point -/
theorem lines_not_concurrent :
  ¬∃ (p : ℝ × ℝ), ∀ (L : Set (ℝ × ℝ)),
    (is_line L ∧ (∀ x y, graph_equation x y → (x, y) ∈ L)) → p ∈ L :=
sorry

end NUMINAMATH_CALUDE_graph_is_three_lines_lines_not_concurrent_l392_39228


namespace NUMINAMATH_CALUDE_decreasing_exponential_function_range_l392_39246

theorem decreasing_exponential_function_range :
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 →
  (∀ x y : ℝ, x < y → a^x > a^y) →
  a ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_exponential_function_range_l392_39246


namespace NUMINAMATH_CALUDE_ratio_sum_to_y_l392_39296

theorem ratio_sum_to_y (w x y : ℚ) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) : 
  (x + y) / y = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_to_y_l392_39296


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l392_39269

def A : Set ℝ := {-1, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l392_39269


namespace NUMINAMATH_CALUDE_cafe_pricing_theorem_l392_39278

/-- Represents the pricing structure of a café -/
structure CafePrices where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ

/-- The café's pricing satisfies the given conditions -/
def satisfies_conditions (p : CafePrices) : Prop :=
  4 * p.sandwich + 9 * p.coffee + p.pie = 4.30 ∧
  7 * p.sandwich + 14 * p.coffee + p.pie = 7.00

/-- Calculates the total cost for a given order -/
def order_cost (p : CafePrices) (sandwiches coffees pies : ℕ) : ℝ :=
  p.sandwich * sandwiches + p.coffee * coffees + p.pie * pies

/-- Theorem stating that the cost of 11 sandwiches, 23 coffees, and 2 pies is $18.87 -/
theorem cafe_pricing_theorem (p : CafePrices) :
  satisfies_conditions p →
  order_cost p 11 23 2 = 18.87 := by
  sorry

end NUMINAMATH_CALUDE_cafe_pricing_theorem_l392_39278


namespace NUMINAMATH_CALUDE_age_ratio_proof_l392_39220

def sachin_age : ℚ := 38.5
def age_difference : ℕ := 7

def rahul_age : ℚ := sachin_age - age_difference

theorem age_ratio_proof :
  (sachin_age * 2 / rahul_age * 2).num = 11 ∧
  (sachin_age * 2 / rahul_age * 2).den = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l392_39220


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l392_39214

theorem lcm_gcd_problem (a b : ℕ+) (h1 : Nat.lcm a b = 7700) (h2 : Nat.gcd a b = 11) (h3 : a = 308) : b = 275 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l392_39214


namespace NUMINAMATH_CALUDE_lauren_tuesday_earnings_l392_39266

/-- Calculates Lauren's earnings from her social media channel -/
def laurens_earnings (commercial_rate : ℚ) (subscription_rate : ℚ) (commercial_views : ℕ) (subscriptions : ℕ) : ℚ :=
  commercial_rate * commercial_views + subscription_rate * subscriptions

theorem lauren_tuesday_earnings :
  let commercial_rate : ℚ := 1/2
  let subscription_rate : ℚ := 1
  let commercial_views : ℕ := 100
  let subscriptions : ℕ := 27
  laurens_earnings commercial_rate subscription_rate commercial_views subscriptions = 77 := by
sorry

end NUMINAMATH_CALUDE_lauren_tuesday_earnings_l392_39266


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l392_39292

/-- Represents a point in a hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a hexagonal lattice with a secondary layer -/
structure HexagonalLattice where
  inner : List LatticePoint
  outer : List LatticePoint

/-- Represents an equilateral triangle in the lattice -/
structure EquilateralTriangle where
  vertices : List LatticePoint
  sideLength : ℝ

/-- Function to count equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles with side lengths 1 or √7 is 6 -/
theorem equilateral_triangle_count (lattice : HexagonalLattice) :
  countEquilateralTriangles lattice = 6 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l392_39292


namespace NUMINAMATH_CALUDE_felipe_house_building_time_l392_39268

/-- Felipe and Emilio's house building problem -/
theorem felipe_house_building_time :
  ∀ (felipe_time emilio_time : ℝ),
  felipe_time + emilio_time = 7.5 →
  felipe_time = (1/2) * emilio_time →
  felipe_time * 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_felipe_house_building_time_l392_39268


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l392_39238

theorem quadratic_discriminant :
  let a : ℝ := 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := 1/2
  (b^2 - 4*a*c) = 2 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l392_39238


namespace NUMINAMATH_CALUDE_zero_in_set_implies_m_equals_two_l392_39212

theorem zero_in_set_implies_m_equals_two (m : ℝ) :
  0 ∈ ({m, m^2 - 2*m} : Set ℝ) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_set_implies_m_equals_two_l392_39212


namespace NUMINAMATH_CALUDE_gcd_50403_40302_l392_39208

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50403_40302_l392_39208


namespace NUMINAMATH_CALUDE_brown_beads_count_l392_39276

theorem brown_beads_count (green red taken_out left_in : ℕ) : 
  green = 1 → 
  red = 3 → 
  taken_out = 2 → 
  left_in = 4 → 
  green + red + (taken_out + left_in - (green + red)) = taken_out + left_in → 
  taken_out + left_in - (green + red) = 2 :=
by sorry

end NUMINAMATH_CALUDE_brown_beads_count_l392_39276
