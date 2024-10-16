import Mathlib

namespace NUMINAMATH_CALUDE_minimize_distance_sum_l1001_100182

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the angle at vertex B of a triangle -/
def angle_at_vertex (t : Triangle) (v : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : Point) (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem about the point that minimizes the sum of distances -/
theorem minimize_distance_sum (t : Triangle) : 
  (∀ v, angle_at_vertex t v < 120) → 
    ∃ O, is_inside O t ∧ 
      ∀ P, is_inside P t → 
        distance O t.A + distance O t.B + distance O t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C 
  ∧ 
  (∃ v, angle_at_vertex t v ≥ 120) → 
    ∃ v, angle_at_vertex t v ≥ 120 ∧ 
      ∀ P, is_inside P t → 
        distance v t.A + distance v t.B + distance v t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C :=
sorry

end NUMINAMATH_CALUDE_minimize_distance_sum_l1001_100182


namespace NUMINAMATH_CALUDE_lcm_36_225_l1001_100132

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_225_l1001_100132


namespace NUMINAMATH_CALUDE_valid_purchase_plans_l1001_100157

/-- Represents a purchasing plan for basketballs and footballs -/
structure PurchasePlan where
  basketballs : ℕ
  footballs : ℕ

/-- Checks if a purchase plan is valid according to the given conditions -/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.basketballs + p.footballs = 20 ∧
  p.basketballs > p.footballs ∧
  80 * p.basketballs + 50 * p.footballs ≤ 1400

theorem valid_purchase_plans :
  ∀ (p : PurchasePlan), isValidPlan p ↔ 
    (p = ⟨11, 9⟩ ∨ p = ⟨12, 8⟩ ∨ p = ⟨13, 7⟩) :=
by sorry

end NUMINAMATH_CALUDE_valid_purchase_plans_l1001_100157


namespace NUMINAMATH_CALUDE_expression_bounds_l1001_100142

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l1001_100142


namespace NUMINAMATH_CALUDE_light_toggle_theorem_l1001_100109

/-- Represents the state of a light (on or off) -/
inductive LightState
| Off
| On

/-- Represents a position in the 5x5 grid -/
structure Position where
  row : Fin 5
  col : Fin 5

/-- The type of the 5x5 grid of lights -/
def Grid := Fin 5 → Fin 5 → LightState

/-- Toggles a light and its adjacent lights in the same row and column -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if exactly one light is on in the grid -/
def exactlyOneOn (grid : Grid) : Prop := sorry

/-- The set of possible positions for the single on light -/
def possiblePositions : Set Position :=
  {⟨2, 2⟩, ⟨2, 4⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨4, 4⟩}

/-- The initial grid with all lights off -/
def initialGrid : Grid := fun _ _ => LightState.Off

theorem light_toggle_theorem :
  ∀ (finalGrid : Grid),
    (∃ (toggleSequence : List Position),
      finalGrid = toggleSequence.foldl toggle initialGrid) →
    exactlyOneOn finalGrid →
    ∃ (pos : Position), finalGrid pos.row pos.col = LightState.On ∧ pos ∈ possiblePositions :=
sorry

end NUMINAMATH_CALUDE_light_toggle_theorem_l1001_100109


namespace NUMINAMATH_CALUDE_train_length_calculation_l1001_100164

/-- The length of a train given its speed, a man's walking speed, and the time it takes to cross the man. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 41.9966402687785 →
  ∃ (train_length : ℝ), abs (train_length - 700) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1001_100164


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l1001_100130

/-- The number of poles needed to enclose a rectangular plot -/
def num_poles (length width long_spacing short_spacing : ℕ) : ℕ :=
  2 * ((length / long_spacing - 1) + (width / short_spacing - 1))

/-- Theorem stating the number of poles needed for the given rectangular plot -/
theorem rectangular_plot_poles : 
  num_poles 120 80 5 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l1001_100130


namespace NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_zero_l1001_100137

theorem absolute_value_not_always_greater_than_zero : 
  ¬ (∀ x : ℝ, |x| > 0) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_zero_l1001_100137


namespace NUMINAMATH_CALUDE_value_of_e_l1001_100147

theorem value_of_e : (14 : ℕ)^2 * 5^3 * 568 = 13916000 := by
  sorry

end NUMINAMATH_CALUDE_value_of_e_l1001_100147


namespace NUMINAMATH_CALUDE_stream_speed_l1001_100149

/-- Proves that given a boat's travel times and distances upstream and downstream, the speed of the stream is 1 km/h -/
theorem stream_speed (b : ℝ) (s : ℝ) 
  (h1 : (b + s) * 10 = 100) 
  (h2 : (b - s) * 25 = 200) : 
  s = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1001_100149


namespace NUMINAMATH_CALUDE_min_sum_three_digit_numbers_l1001_100105

def is_valid_triple (a b c : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ 
  b ≥ 100 ∧ b < 1000 ∧ 
  c ≥ 100 ∧ c < 1000 ∧ 
  a + b = c

def uses_distinct_digits (a b c : Nat) : Prop :=
  let digits := a.digits 10 ++ b.digits 10 ++ c.digits 10
  digits.length = 9 ∧ digits.toFinset.card = 9 ∧ 
  ∀ d ∈ digits, d ≥ 1 ∧ d ≤ 9

theorem min_sum_three_digit_numbers :
  ∃ a b c : Nat, is_valid_triple a b c ∧ 
  uses_distinct_digits a b c ∧
  (∀ x y z : Nat, is_valid_triple x y z → uses_distinct_digits x y z → 
    a + b + c ≤ x + y + z) ∧
  a + b + c = 459 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_three_digit_numbers_l1001_100105


namespace NUMINAMATH_CALUDE_petya_catches_up_l1001_100172

/-- Represents the race scenario between Petya and Vasya -/
structure RaceScenario where
  total_distance : ℝ
  vasya_speed : ℝ
  petya_first_half_speed : ℝ

/-- Calculates Petya's required speed for the second half of the race -/
def petya_second_half_speed (race : RaceScenario) : ℝ :=
  2 * race.vasya_speed - race.petya_first_half_speed

/-- Theorem stating that Petya's speed for the second half must be 18 km/h -/
theorem petya_catches_up (race : RaceScenario) 
  (h1 : race.total_distance > 0)
  (h2 : race.vasya_speed = 12)
  (h3 : race.petya_first_half_speed = 9) :
  petya_second_half_speed race = 18 := by
  sorry

#eval petya_second_half_speed { total_distance := 100, vasya_speed := 12, petya_first_half_speed := 9 }

end NUMINAMATH_CALUDE_petya_catches_up_l1001_100172


namespace NUMINAMATH_CALUDE_plot_length_is_61_l1001_100112

def rectangular_plot_length (breadth : ℝ) (length_difference : ℝ) (fencing_cost_per_meter : ℝ) (total_fencing_cost : ℝ) : ℝ :=
  breadth + length_difference

theorem plot_length_is_61 (breadth : ℝ) :
  let length_difference : ℝ := 22
  let fencing_cost_per_meter : ℝ := 26.50
  let total_fencing_cost : ℝ := 5300
  let length := rectangular_plot_length breadth length_difference fencing_cost_per_meter total_fencing_cost
  let perimeter := 2 * (length + breadth)
  fencing_cost_per_meter * perimeter = total_fencing_cost →
  length = 61 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_61_l1001_100112


namespace NUMINAMATH_CALUDE_multiplication_result_l1001_100161

theorem multiplication_result : 72515 * 10005 = 724787425 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l1001_100161


namespace NUMINAMATH_CALUDE_dereks_score_l1001_100144

/-- Given a basketball team's score and the performance of other players, 
    calculate Derek's score. -/
theorem dereks_score 
  (total_score : ℕ) 
  (other_players : ℕ) 
  (avg_score_others : ℕ) 
  (h1 : total_score = 65) 
  (h2 : other_players = 8) 
  (h3 : avg_score_others = 5) : 
  total_score - (other_players * avg_score_others) = 25 := by
sorry

end NUMINAMATH_CALUDE_dereks_score_l1001_100144


namespace NUMINAMATH_CALUDE_total_yen_calculation_l1001_100145

theorem total_yen_calculation (checking_account savings_account : ℕ) 
  (h1 : checking_account = 6359)
  (h2 : savings_account = 3485) :
  checking_account + savings_account = 9844 := by
  sorry

end NUMINAMATH_CALUDE_total_yen_calculation_l1001_100145


namespace NUMINAMATH_CALUDE_price_after_decrease_l1001_100150

/-- The original price of an article given its reduced price after a percentage decrease -/
def original_price (reduced_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  reduced_price / (1 - decrease_percentage)

/-- Theorem stating that if an article's price after a 56% decrease is Rs. 4400, 
    then its original price was Rs. 10000 -/
theorem price_after_decrease (reduced_price : ℚ) (h : reduced_price = 4400) :
  original_price reduced_price (56/100) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_price_after_decrease_l1001_100150


namespace NUMINAMATH_CALUDE_ratio_problem_l1001_100135

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a / b = 1 / 4)
  (h6 : c / d = 5 / 13)
  (h7 : a / d = 0.1388888888888889) :
  b / c = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1001_100135


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1001_100181

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 2*(x^2 - 2*x^3 + x) + 4*(x + 3*x^3 - 2*x^2 + 2*x^5 + 2*x^3) - 3*(2 + x - 5*x^3 - x^2)
  ∃ (a b c d : ℝ), expression = a*x^5 + b*x^4 + 31*x^3 + c*x^2 + d*x + (2 * 1 - 3 * 2) :=
by sorry

#check coefficient_of_x_cubed

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l1001_100181


namespace NUMINAMATH_CALUDE_factorial_base_representation_823_l1001_100194

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_coeff (n k : ℕ) : ℕ :=
  (n / factorial k) % (k + 1)

theorem factorial_base_representation_823 :
  factorial_base_coeff 823 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_base_representation_823_l1001_100194


namespace NUMINAMATH_CALUDE_power_three_2023_mod_seven_l1001_100125

theorem power_three_2023_mod_seven : 3^2023 % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_three_2023_mod_seven_l1001_100125


namespace NUMINAMATH_CALUDE_concurrent_lines_theorem_l1001_100136

/-- A line that intersects opposite sides of a square -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_opposite_sides : Bool

/-- A configuration of lines dividing a square -/
structure SquareDivision where
  lines : Finset DividingLine
  square : Set (ℝ × ℝ)

/-- The number of concurrent lines in a square division -/
def num_concurrent (sd : SquareDivision) : ℕ := sorry

theorem concurrent_lines_theorem (sd : SquareDivision) 
  (h1 : sd.lines.card = 2005)
  (h2 : ∀ l ∈ sd.lines, l.divides_square)
  (h3 : ∀ l ∈ sd.lines, l.area_ratio = 2 / 3)
  (h4 : ∀ l ∈ sd.lines, l.intersects_opposite_sides) :
  num_concurrent sd ≥ 502 := by sorry

end NUMINAMATH_CALUDE_concurrent_lines_theorem_l1001_100136


namespace NUMINAMATH_CALUDE_number_of_clients_l1001_100122

/-- Proves that the number of clients who visited the garage is 15, given the specified conditions. -/
theorem number_of_clients (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) : 
  num_cars = 10 → selections_per_client = 2 → selections_per_car = 3 → 
  (num_cars * selections_per_car) / selections_per_client = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_clients_l1001_100122


namespace NUMINAMATH_CALUDE_simplify_expressions_l1001_100141

/-- Prove the simplification of two algebraic expressions -/
theorem simplify_expressions (x y : ℝ) :
  (7 * x + 3 * (x^2 - 2) - 3 * (1/2 * x^2 - x + 3) = 3/2 * x^2 + 10 * x - 15) ∧
  (3 * (2 * x^2 * y - x * y^2) - 4 * (-x * y^2 + 3 * x^2 * y) = -6 * x^2 * y + x * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1001_100141


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l1001_100146

/-- The number of ways to arrange books on a shelf --/
def arrange_books (num_math_books : ℕ) (num_history_books : ℕ) : ℕ :=
  let math_book_arrangements := (num_math_books.choose 2) * (2 * 2)
  let history_book_arrangements := num_history_books.factorial
  math_book_arrangements * history_book_arrangements

/-- Theorem: The number of ways to arrange 4 math books and 6 history books with 2 math books on each end is 17280 --/
theorem book_arrangement_theorem :
  arrange_books 4 6 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l1001_100146


namespace NUMINAMATH_CALUDE_fraction_numerator_is_twelve_l1001_100185

theorem fraction_numerator_is_twelve :
  ∀ (numerator : ℚ),
    (∃ (denominator : ℚ),
      denominator = 2 * numerator + 4 ∧
      numerator / denominator = 3 / 7) →
    numerator = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_is_twelve_l1001_100185


namespace NUMINAMATH_CALUDE_complex_number_value_l1001_100187

theorem complex_number_value (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l1001_100187


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1001_100166

/-- Calculates the total distance traveled by a car given its initial speed, acceleration, 
    acceleration time, constant speed, and constant speed time. -/
def total_distance (initial_speed : ℝ) (acceleration : ℝ) (accel_time : ℝ) 
                   (constant_speed : ℝ) (const_time : ℝ) : ℝ :=
  -- Distance covered during acceleration
  (initial_speed * accel_time + 0.5 * acceleration * accel_time^2) +
  -- Distance covered at constant speed
  (constant_speed * const_time)

/-- Theorem stating that a car with given parameters travels 250 miles in total -/
theorem car_distance_theorem : 
  total_distance 30 5 2 60 3 = 250 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1001_100166


namespace NUMINAMATH_CALUDE_inverse_217_mod_397_l1001_100152

theorem inverse_217_mod_397 : ∃ a : ℤ, 0 ≤ a ∧ a < 397 ∧ (217 * a) % 397 = 1 :=
by
  use 161
  sorry

end NUMINAMATH_CALUDE_inverse_217_mod_397_l1001_100152


namespace NUMINAMATH_CALUDE_scalar_projection_a_onto_b_l1001_100176

/-- The scalar projection of vector a (1, 2) onto vector b (3, 4) is 11/5 -/
theorem scalar_projection_a_onto_b :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 4)
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_scalar_projection_a_onto_b_l1001_100176


namespace NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l1001_100173

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 25 := by
  sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x/y + y/z + z/x + y/x + z/y + x/z = 10) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a/b + b/c + c/a + b/a + c/b + a/c = 10 ∧
    (a/b + b/c + c/a) * (b/a + c/b + a/c) = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l1001_100173


namespace NUMINAMATH_CALUDE_highest_powers_sum_12_factorial_l1001_100165

theorem highest_powers_sum_12_factorial : 
  let n := 12
  let factorial_n := n.factorial
  let highest_power_of_10 := (factorial_n.factorization 2).min (factorial_n.factorization 5)
  let highest_power_of_6 := (factorial_n.factorization 2).min (factorial_n.factorization 3)
  highest_power_of_10 + highest_power_of_6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_highest_powers_sum_12_factorial_l1001_100165


namespace NUMINAMATH_CALUDE_complement_of_A_l1001_100171

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0 ∨ x ≥ 1}

theorem complement_of_A : Set.compl A = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1001_100171


namespace NUMINAMATH_CALUDE_fencemaker_problem_l1001_100177

/-- Given a rectangular yard with one side of 40 feet and an area of 480 square feet,
    the perimeter minus one side is equal to 64 feet. -/
theorem fencemaker_problem (length width : ℝ) : 
  length = 40 ∧ 
  length * width = 480 ∧ 
  width > 0 → 
  2 * width + length = 64 := by
  sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l1001_100177


namespace NUMINAMATH_CALUDE_second_number_value_l1001_100156

theorem second_number_value (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : a / b = 3 / 4)
  (ratio_bc : b / c = 2 / 5)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) : 
  b = 480 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1001_100156


namespace NUMINAMATH_CALUDE_angle_measure_l1001_100118

theorem angle_measure (θ φ : ℝ) : 
  (90 - θ) = 0.4 * (180 - θ) →  -- complement is 40% of supplement
  φ = 180 - θ →                 -- θ and φ form a linear pair
  φ = 2 * θ →                   -- φ is twice the size of θ
  θ = 30 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_l1001_100118


namespace NUMINAMATH_CALUDE_first_train_length_is_30_l1001_100175

/-- The length of the second train in meters -/
def second_train_length : ℝ := 180

/-- The time taken by the first train to cross the stationary second train in seconds -/
def time_cross_stationary : ℝ := 18

/-- The length of the platform crossed by the first train in meters -/
def platform_length_first : ℝ := 250

/-- The time taken by the first train to cross its platform in seconds -/
def time_cross_platform_first : ℝ := 24

/-- The length of the platform crossed by the second train in meters -/
def platform_length_second : ℝ := 200

/-- The time taken by the second train to cross its platform in seconds -/
def time_cross_platform_second : ℝ := 22

/-- The length of the first train in meters -/
def first_train_length : ℝ := 30

theorem first_train_length_is_30 :
  (first_train_length + second_train_length) / time_cross_stationary =
  (first_train_length + platform_length_first) / time_cross_platform_first ∧
  first_train_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_train_length_is_30_l1001_100175


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1001_100178

/-- Given x = 4 * 21 * 63, the smallest positive integer y such that xy is a perfect cube is 14 -/
theorem smallest_y_for_perfect_cube (x : ℕ) (hx : x = 4 * 21 * 63) :
  ∃ y : ℕ, y > 0 ∧ 
    (∃ z : ℕ, x * y = z^3) ∧
    (∀ w : ℕ, w > 0 ∧ w < y → ¬∃ z : ℕ, x * w = z^3) ∧
    y = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l1001_100178


namespace NUMINAMATH_CALUDE_roots_opposite_signs_n_value_l1001_100116

/-- 
Given an equation of the form (x^2 - (a+1)x) / ((b+1)x - d) = (n-2) / (n+2),
if the roots of this equation are numerically equal but of opposite signs,
then n = 2(b-a) / (a+b+2).
-/
theorem roots_opposite_signs_n_value 
  (a b d n : ℝ) 
  (eq : ∀ x, (x^2 - (a+1)*x) / ((b+1)*x - d) = (n-2) / (n+2)) 
  (roots_opposite : ∃ r : ℝ, (r^2 - (a+1)*r) / ((b+1)*r - d) = (n-2) / (n+2) ∧ 
                              ((-r)^2 - (a+1)*(-r)) / ((b+1)*(-r) - d) = (n-2) / (n+2)) :
  n = 2*(b-a) / (a+b+2) := by
sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_n_value_l1001_100116


namespace NUMINAMATH_CALUDE_prob_divisible_by_4_5_or_7_l1001_100186

/-- The probability of selecting a number from 1 to 200 that is divisible by 4, 5, or 7 -/
theorem prob_divisible_by_4_5_or_7 : 
  let S := Finset.range 200
  let divisible_by_4_5_or_7 := fun n => n % 4 = 0 ∨ n % 5 = 0 ∨ n % 7 = 0
  (S.filter divisible_by_4_5_or_7).card / S.card = 97 / 200 := by
  sorry

end NUMINAMATH_CALUDE_prob_divisible_by_4_5_or_7_l1001_100186


namespace NUMINAMATH_CALUDE_money_left_after_candy_purchase_l1001_100188

def lollipop_cost : ℚ := 1.5
def gummy_pack_cost : ℚ := 2
def lollipop_count : ℕ := 4
def gummy_pack_count : ℕ := 2
def initial_money : ℚ := 15

def total_spent : ℚ := lollipop_cost * lollipop_count + gummy_pack_cost * gummy_pack_count

theorem money_left_after_candy_purchase : 
  initial_money - total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_candy_purchase_l1001_100188


namespace NUMINAMATH_CALUDE_steve_earnings_l1001_100127

def total_copies : ℕ := 1000000
def advance_copies : ℕ := 100000
def price_per_copy : ℚ := 2
def agent_percentage : ℚ := 1/10

theorem steve_earnings : 
  (total_copies - advance_copies) * price_per_copy * (1 - agent_percentage) = 1620000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l1001_100127


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1001_100192

/-- A trapezoid with given dimensions -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.base1 + t.base2 + t.side1 + t.side2

/-- Theorem: The perimeter of the specific trapezoid is 42 -/
theorem trapezoid_perimeter : 
  let t : Trapezoid := { base1 := 10, base2 := 14, side1 := 9, side2 := 9 }
  perimeter t = 42 := by sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1001_100192


namespace NUMINAMATH_CALUDE_hyperbola_center_l1001_100123

/-- The equation of a hyperbola in general form -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola -/
def HyperbolaCenter : ℝ × ℝ := (3, 4)

/-- Theorem: The center of the hyperbola defined by the given equation is (3, 4) -/
theorem hyperbola_center :
  ∀ (x y : ℝ), HyperbolaEquation x y →
  ∃ (a b : ℝ), (x - HyperbolaCenter.1)^2 / a^2 - (y - HyperbolaCenter.2)^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1001_100123


namespace NUMINAMATH_CALUDE_east_west_southwest_angle_l1001_100104

/-- Represents the directions of rays in the decagon arrangement -/
inductive Direction
| North
| East
| WestSouthWest

/-- Represents a regular decagon with rays -/
structure DecagonArrangement where
  rays : Fin 10 → Direction
  north_ray : ∃ i, rays i = Direction.North

/-- Calculates the number of sectors between two directions -/
def sectors_between (d1 d2 : Direction) : ℕ := sorry

/-- Calculates the angle in degrees between two rays -/
def angle_between (d1 d2 : Direction) : ℝ :=
  (sectors_between d1 d2 : ℝ) * 36

theorem east_west_southwest_angle (arrangement : DecagonArrangement) :
  angle_between Direction.East Direction.WestSouthWest = 180 := by sorry

end NUMINAMATH_CALUDE_east_west_southwest_angle_l1001_100104


namespace NUMINAMATH_CALUDE_evaluate_expression_l1001_100102

theorem evaluate_expression (x : ℝ) (h : x = 6) : 
  (x^9 - 24*x^6 + 144*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1001_100102


namespace NUMINAMATH_CALUDE_three_digit_equation_solution_l1001_100128

/-- Represents a three-digit number ABC --/
def threeDigitNumber (A B C : ℕ) : ℕ := 100 * A + 10 * B + C

/-- Represents a two-digit number AB --/
def twoDigitNumber (A B : ℕ) : ℕ := 10 * A + B

/-- Checks if three numbers are distinct digits --/
def areDistinctDigits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1

theorem three_digit_equation_solution :
  ∀ A B C : ℕ,
    areDistinctDigits A B C →
    threeDigitNumber A B C = twoDigitNumber A B * C + twoDigitNumber B C * A + twoDigitNumber C A * B →
    ((A = 7 ∧ B = 8 ∧ C = 1) ∨ (A = 5 ∧ B = 1 ∧ C = 7)) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_equation_solution_l1001_100128


namespace NUMINAMATH_CALUDE_cube_product_theorem_l1001_100167

theorem cube_product_theorem : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) * (f 8) = 73 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_theorem_l1001_100167


namespace NUMINAMATH_CALUDE_min_formula_l1001_100115

theorem min_formula (a b : ℝ) : min a b = (a + b - Real.sqrt ((a - b)^2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_formula_l1001_100115


namespace NUMINAMATH_CALUDE_correct_result_l1001_100155

theorem correct_result (x : ℤ) (h : x - 27 + 19 = 84) : x - 19 + 27 = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l1001_100155


namespace NUMINAMATH_CALUDE_power_equation_l1001_100131

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 5) : a^(2*m + n) = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1001_100131


namespace NUMINAMATH_CALUDE_storm_deposit_calculation_l1001_100169

-- Define the reservoir capacity and initial content
def reservoir_capacity : ℝ := 400000000000
def initial_content : ℝ := 220000000000

-- Define the initial and final fill percentages
def initial_fill_percentage : ℝ := 0.5500000000000001
def final_fill_percentage : ℝ := 0.85

-- Define the amount of water deposited by the storm
def storm_deposit : ℝ := 120000000000

-- Theorem statement
theorem storm_deposit_calculation :
  initial_content = initial_fill_percentage * reservoir_capacity ∧
  storm_deposit = final_fill_percentage * reservoir_capacity - initial_content :=
by sorry

end NUMINAMATH_CALUDE_storm_deposit_calculation_l1001_100169


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1001_100111

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
def min_at_2 (a b : ℝ) : Prop := ∀ x, f a b x ≥ f a b 2

def intercept_length_2 (a b : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ x₂ - x₁ = 2

-- Define g(x)
def g (a b m : ℝ) (x : ℝ) : ℝ := f a b x - m * x

-- Define the conditions for g(x)
def g_zeros_in_intervals (a b m : ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3 ∧ g a b m x₁ = 0 ∧ g a b m x₂ = 0

-- Define the minimum value condition
def min_value_condition (a b : ℝ) (t : ℝ) : Prop :=
  ∀ x ∈ Set.Icc t (t + 1), f a b x ≥ -1/2 ∧ ∃ x₀ ∈ Set.Icc t (t + 1), f a b x₀ = -1/2

-- State the theorem
theorem quadratic_function_properties :
  ∀ a b : ℝ, min_at_2 a b → intercept_length_2 a b →
  (∃ m : ℝ, g_zeros_in_intervals a b m ∧ -1/2 < m ∧ m < 0) ∧
  (∃ t : ℝ, (min_value_condition a b t ∧ t = 1 - Real.sqrt 2 / 2) ∨
            (min_value_condition a b t ∧ t = 2 + Real.sqrt 2 / 2)) ∧
  a = 1 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1001_100111


namespace NUMINAMATH_CALUDE_projectile_max_height_l1001_100179

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 175

/-- Theorem stating that the maximum height reached by the projectile is 175 meters -/
theorem projectile_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1001_100179


namespace NUMINAMATH_CALUDE_population_growth_rate_l1001_100124

/-- Proves that given a population of 10,000 that grows to 12,100 in 2 years
    with a constant annual growth rate, the annual percentage increase is 10%. -/
theorem population_growth_rate (initial_population : ℕ) (final_population : ℕ) 
  (years : ℕ) (growth_rate : ℝ) :
  initial_population = 10000 →
  final_population = 12100 →
  years = 2 →
  final_population = initial_population * (1 + growth_rate) ^ years →
  growth_rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l1001_100124


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l1001_100126

theorem min_x_prime_factorization (x y : ℕ+) (h : 13 * x^4 = 29 * y^12) :
  ∃ (a b c d : ℕ), 
    (x = (29^3 : ℕ+) * (13^3 : ℕ+)) ∧
    (∀ z : ℕ+, 13 * z^4 = 29 * y^12 → x ≤ z) ∧
    (Nat.Prime a ∧ Nat.Prime b) ∧
    (x = a^c * b^d) ∧
    (a + b + c + d = 48) := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l1001_100126


namespace NUMINAMATH_CALUDE_range_of_m_l1001_100189

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) 
  (h : ∀ m : ℝ, (4/x) + (16/y) > m^2 - 3*m + 5) :
  ∀ m : ℝ, -1 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1001_100189


namespace NUMINAMATH_CALUDE_max_garden_area_l1001_100119

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- The perimeter of the garden -/
def Garden.perimeter (g : Garden) : ℝ := 2 * (g.length + g.width)

/-- The area of the garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a rectangular garden with given constraints -/
theorem max_garden_area (g : Garden) 
  (h_perimeter : g.perimeter = 400) 
  (h_min_length : g.length ≥ 100) : 
  g.area ≤ 10000 ∧ (g.area = 10000 ↔ g.length = 100 ∧ g.width = 100) := by
  sorry

#check max_garden_area

end NUMINAMATH_CALUDE_max_garden_area_l1001_100119


namespace NUMINAMATH_CALUDE_income_mean_difference_l1001_100117

/-- The number of families --/
def num_families : ℕ := 1200

/-- The correct highest income --/
def correct_highest_income : ℕ := 150000

/-- The incorrect highest income --/
def incorrect_highest_income : ℕ := 1500000

/-- The sum of all incomes except the highest --/
def S : ℕ := sorry

/-- The difference between the mean of incorrect data and actual data --/
def mean_difference : ℚ :=
  (S + incorrect_highest_income : ℚ) / num_families -
  (S + correct_highest_income : ℚ) / num_families

theorem income_mean_difference :
  mean_difference = 1125 := by sorry

end NUMINAMATH_CALUDE_income_mean_difference_l1001_100117


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1001_100129

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 3 → a = Real.sqrt 2 →
  (b = Real.sqrt 3 → A = π / 4) ∧
  (1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 → b = Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1001_100129


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1001_100139

-- Define the types for lines and planes
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relationships
variable (parallel : L → P → Prop)
variable (perpendicular : L → P → Prop)
variable (plane_perpendicular : P → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (ι : L) (α β : P) (h1 : parallel ι α) (h2 : perpendicular ι β) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1001_100139


namespace NUMINAMATH_CALUDE_complex_number_equal_parts_l1001_100191

theorem complex_number_equal_parts (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equal_parts_l1001_100191


namespace NUMINAMATH_CALUDE_discount_is_forty_percent_l1001_100133

def normal_cost : ℕ := 15
def package_size : ℕ := 20
def package_price : ℕ := 180

def discount_percentage : ℚ :=
  (package_size * normal_cost - package_price) / (package_size * normal_cost) * 100

theorem discount_is_forty_percent : discount_percentage = 40 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_forty_percent_l1001_100133


namespace NUMINAMATH_CALUDE_correct_calculation_l1001_100114

theorem correct_calculation (x : ℝ) : x / 12 = 8 → x * 12 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1001_100114


namespace NUMINAMATH_CALUDE_runners_on_circular_track_l1001_100101

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  speed : ℝ

/-- Theorem about two runners on a circular track -/
theorem runners_on_circular_track
  (track_length : ℝ)
  (troye daniella : Runner)
  (h1 : track_length > 0)
  (h2 : troye.lap_time = 56)
  (h3 : troye.speed = track_length / troye.lap_time)
  (h4 : daniella.speed = track_length / daniella.lap_time)
  (h5 : troye.speed + daniella.speed = track_length / 24) :
  daniella.lap_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_runners_on_circular_track_l1001_100101


namespace NUMINAMATH_CALUDE_min_value_expression_l1001_100184

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 4) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 5 ∧
  ((x + 4) / Real.sqrt (x - 1) = 2 * Real.sqrt 5 ↔ x = 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1001_100184


namespace NUMINAMATH_CALUDE_dinner_bill_ratio_l1001_100197

/-- Given a dinner bill split between three people, this theorem proves
    the ratio of two people's payments given certain conditions. -/
theorem dinner_bill_ratio (total bill : ℚ) (daniel clarence matthew : ℚ) :
  bill = 20.20 →
  daniel = 6.06 →
  daniel = (1 / 2) * clarence →
  bill = daniel + clarence + matthew →
  clarence / matthew = 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dinner_bill_ratio_l1001_100197


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l1001_100193

/-- Given the total number of doughnuts made, the number of boxes sold, and the number of doughnuts
given away, prove that the number of doughnuts in each box is equal to
(total doughnuts made - doughnuts given away) divided by the number of boxes sold. -/
theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : total_doughnuts ≥ doughnuts_given_away)
  (h2 : boxes_sold > 0)
  (h3 : total_doughnuts - doughnuts_given_away = boxes_sold * (total_doughnuts - doughnuts_given_away) / boxes_sold) :
  (total_doughnuts - doughnuts_given_away) / boxes_sold =
  (total_doughnuts - doughnuts_given_away) / boxes_sold :=
by sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l1001_100193


namespace NUMINAMATH_CALUDE_sequence_proof_l1001_100199

def arithmetic_sequence (a b c : ℝ) : Prop := b - a = c - b

def geometric_sequence (f : ℕ → ℝ) : Prop := ∀ n : ℕ, f (n + 1) / f n = f 2 / f 1

def sum_sequence (f : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_sequence f n + f (n + 1)

theorem sequence_proof 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : arithmetic_sequence a b c)
  (h_sum : a + b + c = 15)
  (b_n : ℕ → ℝ)
  (h_geometric : geometric_sequence (λ n => b_n (n + 2)))
  (h_relation : b_n 3 = a + 2 ∧ b_n 4 = b + 5 ∧ b_n 5 = c + 13) :
  (∀ n : ℕ, b_n n = (5/4) * 2^(n-1)) ∧
  (geometric_sequence (λ n => sum_sequence b_n n + 5/4) ∧
   (sum_sequence b_n 1 + 5/4 = 5/2) ∧
   (∀ n : ℕ, (sum_sequence b_n (n+1) + 5/4) / (sum_sequence b_n n + 5/4) = 2)) :=
sorry

end NUMINAMATH_CALUDE_sequence_proof_l1001_100199


namespace NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l1001_100103

-- Define the concept of a relationship between two variables
def Relationship (X Y : Type) := X → Y → Prop

-- Define functional relationship
def FunctionalRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ∀ (x : X), ∃! (y : Y), r x y

-- Define correlation relationship
def CorrelationRelationship (X Y : Type) (r : Relationship X Y) : Prop :=
  ¬FunctionalRelationship X Y r ∧ ∃ (x₁ x₂ : X) (y₁ y₂ : Y), r x₁ y₁ ∧ r x₂ y₂

-- Define the relationships for each option
def CubeVolumeEdgeLength : Relationship ℝ ℝ := sorry
def AngleSine : Relationship ℝ ℝ := sorry
def SunlightRiceYield : Relationship ℝ ℝ := sorry
def HeightVision : Relationship ℝ ℝ := sorry

-- State the theorem
theorem sunlight_rice_yield_correlation :
  CorrelationRelationship ℝ ℝ SunlightRiceYield ∧
  ¬CorrelationRelationship ℝ ℝ CubeVolumeEdgeLength ∧
  ¬CorrelationRelationship ℝ ℝ AngleSine ∧
  ¬CorrelationRelationship ℝ ℝ HeightVision :=
sorry

end NUMINAMATH_CALUDE_sunlight_rice_yield_correlation_l1001_100103


namespace NUMINAMATH_CALUDE_inequality_proof_l1001_100160

theorem inequality_proof (x y z : ℝ) (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1001_100160


namespace NUMINAMATH_CALUDE_cube_surface_area_l1001_100159

/-- The surface area of a cube, given the distance between non-intersecting diagonals of adjacent faces -/
theorem cube_surface_area (d : ℝ) (h : d = 8) : 
  let a := d * 3 / Real.sqrt 3
  6 * a^2 = 1152 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1001_100159


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1001_100170

/-- The area of the union of a rectangle and a circle with specified dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_width : ℝ := 12
  let rectangle_height : ℝ := 15
  let circle_radius : ℝ := 15
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let circle_area : ℝ := π * circle_radius^2
  let overlap_area : ℝ := (1/4) * circle_area
  let union_area : ℝ := rectangle_area + circle_area - overlap_area
  union_area = 180 + 168.75 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1001_100170


namespace NUMINAMATH_CALUDE_book_loss_percentage_l1001_100196

/-- Given that the cost price of 30 books equals the selling price of 40 books,
    prove that the loss percentage is 25%. -/
theorem book_loss_percentage (C S : ℝ) (h : C > 0) (h1 : 30 * C = 40 * S) : 
  (C - S) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_loss_percentage_l1001_100196


namespace NUMINAMATH_CALUDE_hcf_lcm_problem_l1001_100153

theorem hcf_lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : b = 220) : a = 36 := by
  sorry

end NUMINAMATH_CALUDE_hcf_lcm_problem_l1001_100153


namespace NUMINAMATH_CALUDE_math_books_probability_math_books_probability_is_one_sixth_l1001_100138

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books -/
theorem math_books_probability : ℚ :=
  let total_books : ℕ := 4
  let math_books : ℕ := 2
  let books_to_pick : ℕ := 2
  let total_combinations := Nat.choose total_books books_to_pick
  let favorable_combinations := Nat.choose math_books books_to_pick
  (favorable_combinations : ℚ) / total_combinations

/-- The probability of selecting 2 math books from a shelf with 2 math books and 2 physics books is 1/6 -/
theorem math_books_probability_is_one_sixth : math_books_probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_math_books_probability_math_books_probability_is_one_sixth_l1001_100138


namespace NUMINAMATH_CALUDE_circle_radii_sum_l1001_100143

theorem circle_radii_sum : 
  ∀ s : ℝ, 
  (s > 0) →
  (s^2 - 12*s + 12 = 0) →
  (∃ t : ℝ, s^2 - 12*s + 12 = 0 ∧ s ≠ t) →
  (s + (12 - s) = 12) := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l1001_100143


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1001_100198

/-- A rectangle with length thrice its breadth and area 75 square meters has a perimeter of 40 meters. -/
theorem rectangle_perimeter (breadth : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  length = 3 * breadth →
  area = 75 →
  area = length * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1001_100198


namespace NUMINAMATH_CALUDE_x_greater_y_greater_z_l1001_100107

theorem x_greater_y_greater_z (α b x y z : Real) 
  (h_α : α ∈ Set.Ioo (π / 4) (π / 2))
  (h_b : b ∈ Set.Ioo 0 1)
  (h_x : Real.log x = (Real.log (Real.sin α))^2 / Real.log b)
  (h_y : Real.log y = (Real.log (Real.cos α))^2 / Real.log b)
  (h_z : Real.log z = (Real.log (Real.sin α * Real.cos α))^2 / Real.log b) :
  x > y ∧ y > z := by
  sorry

end NUMINAMATH_CALUDE_x_greater_y_greater_z_l1001_100107


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1001_100113

theorem quadratic_minimum (x : ℝ) :
  let y := 4 * x^2 + 8 * x + 16
  ∀ x', 4 * x'^2 + 8 * x' + 16 ≥ 12 ∧ (4 * (-1)^2 + 8 * (-1) + 16 = 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1001_100113


namespace NUMINAMATH_CALUDE_seeds_planted_wednesday_l1001_100158

/-- The number of seeds planted on Wednesday -/
def seeds_wednesday : ℕ := sorry

/-- The number of seeds planted on Thursday -/
def seeds_thursday : ℕ := 2

/-- The total number of seeds planted -/
def total_seeds : ℕ := 22

/-- Theorem stating that the number of seeds planted on Wednesday is 20 -/
theorem seeds_planted_wednesday :
  seeds_wednesday = total_seeds - seeds_thursday ∧ seeds_wednesday = 20 := by sorry

end NUMINAMATH_CALUDE_seeds_planted_wednesday_l1001_100158


namespace NUMINAMATH_CALUDE_production_today_is_90_l1001_100151

/-- Calculates the production for today given the previous average, new average, and number of previous days. -/
def todayProduction (prevAvg newAvg : ℚ) (prevDays : ℕ) : ℚ :=
  (newAvg * (prevDays + 1) : ℚ) - (prevAvg * prevDays : ℚ)

/-- Proves that the production today is 90 units, given the specified conditions. -/
theorem production_today_is_90 :
  todayProduction 60 62 14 = 90 := by
  sorry

#eval todayProduction 60 62 14

end NUMINAMATH_CALUDE_production_today_is_90_l1001_100151


namespace NUMINAMATH_CALUDE_n_equals_t_plus_2_l1001_100174

theorem n_equals_t_plus_2 (t : ℝ) (h : t ≠ 3) :
  let n := (4*t^2 - 10*t - 2 - 3*(t^2 - t + 3) + t^2 + 5*t - 1) / ((t + 7) + (t - 13))
  n = t + 2 := by sorry

end NUMINAMATH_CALUDE_n_equals_t_plus_2_l1001_100174


namespace NUMINAMATH_CALUDE_jasons_red_marbles_indeterminate_l1001_100190

theorem jasons_red_marbles_indeterminate (jason_blue : ℕ) (tom_blue : ℕ) (total_blue : ℕ) 
  (h1 : jason_blue = 44)
  (h2 : tom_blue = 24)
  (h3 : total_blue = jason_blue + tom_blue)
  (h4 : total_blue = 68) :
  ∃ (x y : ℕ), x ≠ y ∧ (jason_blue + x = jason_blue + y) :=
sorry

end NUMINAMATH_CALUDE_jasons_red_marbles_indeterminate_l1001_100190


namespace NUMINAMATH_CALUDE_race_distance_l1001_100154

/-- Given two runners in a race, prove the total distance of the race -/
theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) (h3 : lead = 33.333333333333336) :
  ∃ (distance : ℝ), distance = 100 ∧ distance / time_A - distance / time_B = lead / time_A := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1001_100154


namespace NUMINAMATH_CALUDE_abc_product_l1001_100183

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 152)
  (h2 : b * (c + a) = 162)
  (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l1001_100183


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_minimum_value_achieved_l1001_100134

theorem minimum_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4 * a^2 + b^2)).sqrt) / (a * b) ≥ Real.sqrt 6 :=
sorry

theorem minimum_value_achieved (a : ℝ) (ha : a > 0) :
  (((a^2 + a^2) * (4 * a^2 + a^2)).sqrt) / (a * a) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_minimum_value_achieved_l1001_100134


namespace NUMINAMATH_CALUDE_calculator_minimum_operations_l1001_100108

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesTwo => n * 2

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The theorem to be proved --/
theorem calculator_minimum_operations :
  ∃ (ops : List Operation),
    isValidSequence ops 400 ∧
    ops.length = 10 ∧
    (∀ (other_ops : List Operation),
      isValidSequence other_ops 400 → other_ops.length ≥ 10) := by
  sorry


end NUMINAMATH_CALUDE_calculator_minimum_operations_l1001_100108


namespace NUMINAMATH_CALUDE_composite_sum_l1001_100140

theorem composite_sum (a b c d m n : ℕ) 
  (ha : a > b) (hb : b > c) (hc : c > d) 
  (hdiv : (a + b - c + d) ∣ (a * c + b * d))
  (hm : m > 0) (hn : Odd n) : 
  ∃ k > 1, k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end NUMINAMATH_CALUDE_composite_sum_l1001_100140


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1001_100110

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Icc 0 (π / 2)) 
  (h2 : Real.cos (α + π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1001_100110


namespace NUMINAMATH_CALUDE_ferris_wheel_seat_count_l1001_100168

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  (total_people + people_per_seat - 1) / people_per_seat

/-- Theorem: The Ferris wheel has 3 seats -/
theorem ferris_wheel_seat_count : ferris_wheel_seats 8 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seat_count_l1001_100168


namespace NUMINAMATH_CALUDE_base_k_addition_l1001_100106

/-- Represents a digit in base k -/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k -/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k -/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Checks if two lists of digits are equal -/
def digitListEq (a b : List (Digit k)) : Prop :=
  sorry

theorem base_k_addition :
  ∃ k : ℕ, k > 1 ∧
    digitListEq
      (addBaseK (toBaseK 8374 k) (toBaseK 9423 k))
      (toBaseK 20397 k) ∧
    k = 18 :=
  sorry

end NUMINAMATH_CALUDE_base_k_addition_l1001_100106


namespace NUMINAMATH_CALUDE_equation_equality_relationship_l1001_100148

-- Define what an equality is
def IsEquality (s : String) : Prop := true  -- All mathematical statements of the form a = b are equalities

-- Define what an equation is
def IsEquation (s : String) : Prop := IsEquality s ∧ ∃ x, s.contains x  -- An equation is an equality that contains unknowns

-- The statement we want to prove false
def statement : Prop :=
  (∀ s, IsEquation s → IsEquality s) ∧ (∀ s, IsEquality s → IsEquation s)

-- Theorem: The statement is false
theorem equation_equality_relationship : ¬statement := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_relationship_l1001_100148


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l1001_100100

theorem bake_sale_group_composition (total : ℕ) (boys : ℕ) : 
  (boys : ℚ) / total = 35 / 100 →
  ((boys - 3 : ℚ) / total) = 40 / 100 →
  boys = 21 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l1001_100100


namespace NUMINAMATH_CALUDE_fixed_point_below_x_axis_triangle_area_one_l1001_100121

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + k - 1

-- Part 1: Prove that the line passes through (-1, -1) for all k
theorem fixed_point (k : ℝ) : line_l k (-1) = -1 := by sorry

-- Part 2: Prove the range of k for which all points are below x-axis
theorem below_x_axis (k : ℝ) :
  (∀ x, -4 < x ∧ x < 4 → line_l k x < 0) ↔ -1/3 ≤ k ∧ k ≤ 1/5 := by sorry

-- Part 3: Prove the values of k for which the triangle area is 1
theorem triangle_area_one (k : ℝ) :
  (∃ x y, x > 0 ∧ y > 0 ∧ line_l k x = 0 ∧ line_l k 0 = y ∧ 1/2 * x * y = 1) ↔
  k = 2 + Real.sqrt 3 ∨ k = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_below_x_axis_triangle_area_one_l1001_100121


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1001_100195

theorem smallest_number_of_eggs : ∀ (n : ℕ), 
  (n > 150) → 
  (∃ (c : ℕ), n = 15 * c - 5) → 
  (∀ (m : ℕ), m > 150 ∧ (∃ (d : ℕ), m = 15 * d - 5) → m ≥ n) → 
  n = 160 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1001_100195


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1001_100180

noncomputable section

-- Define the function f
def f (x a b : ℝ) : ℝ := Real.exp x * (x^2 - (a + 2) * x + b)

-- Define the derivative of f
def f' (x a b : ℝ) : ℝ := Real.exp x * (x^2 - a * x + b - (a + 2))

theorem tangent_line_and_minimum_value (a b : ℝ) :
  (f' 0 a b = -2 * a^2) →
  (b = a + 2 - 2 * a^2) ∧
  (∀ a < 0, ∃ M ≥ 2, ∀ x > 0, f x a b < M) :=
by sorry

end

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1001_100180


namespace NUMINAMATH_CALUDE_cyclic_permutation_sum_equality_l1001_100120

def is_cyclic_shift (a : Fin n → ℕ) : Prop :=
  ∃ i, ∀ j, a j = ((j.val + i - 1) % n) + 1

def is_permutation (b : Fin n → ℕ) : Prop :=
  Function.Bijective b ∧ ∀ i, b i ≤ n

theorem cyclic_permutation_sum_equality (n : ℕ) :
  (∃ (a b : Fin n → ℕ),
    is_cyclic_shift a ∧
    is_permutation b ∧
    ∀ i j : Fin n, i.val + 1 + a i + b i = j.val + 1 + a j + b j) ↔
  Odd n :=
sorry

end NUMINAMATH_CALUDE_cyclic_permutation_sum_equality_l1001_100120


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_containing_interval_l1001_100162

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part 1
theorem solution_set_when_a_is_one :
  let a := 1
  ∃ S : Set ℝ, S = {x | f a x ≥ g x} ∧ 
    S = Set.Icc (-1) (((-1 : ℝ) + Real.sqrt 17) / 2) :=
sorry

-- Theorem for part 2
theorem range_of_a_containing_interval :
  ∃ R : Set ℝ, R = {a | ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x} ∧
    R = Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_containing_interval_l1001_100162


namespace NUMINAMATH_CALUDE_range_of_m_l1001_100163

/-- Given the conditions of P and Q, prove that the range of m is [9, +∞) -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|(4 - x) / 3| ≤ 2 → (x + m - 1) * (x - m - 1) ≤ 0)) ∧
  (∃ x : ℝ, |(4 - x) / 3| > 2 ∧ (x + m - 1) * (x - m - 1) > 0) ∧
  (m > 0) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1001_100163
