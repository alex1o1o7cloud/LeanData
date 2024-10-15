import Mathlib

namespace NUMINAMATH_CALUDE_equilateral_center_triangles_properties_l3089_308996

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an equilateral triangle constructed on a side of another triangle -/
structure EquilateralTriangle where
  base : ℝ × ℝ
  apex : ℝ × ℝ

/-- The triangle formed by centers of equilateral triangles -/
def CenterTriangle (T : Triangle) (outward : Bool) : Triangle := sorry

/-- The centroid of a triangle -/
def centroid (T : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Main theorem about properties of triangles formed by centers of equilateral triangles -/
theorem equilateral_center_triangles_properties (T : Triangle) :
  let Δ := CenterTriangle T true
  let δ := CenterTriangle T false
  -- 1) Δ and δ are equilateral
  (∀ (X Y Z : ℝ × ℝ), (X = Δ.A ∧ Y = Δ.B ∧ Z = Δ.C) ∨ (X = δ.A ∧ Y = δ.B ∧ Z = δ.C) →
    dist X Y = dist Y Z ∧ dist Y Z = dist Z X) ∧
  -- 2) Centers of Δ and δ coincide with the centroid of T
  (centroid Δ = centroid T ∧ centroid δ = centroid T) ∧
  -- 3) Area(Δ) - Area(δ) = Area(T)
  (area Δ - area δ = area T) := by
  sorry


end NUMINAMATH_CALUDE_equilateral_center_triangles_properties_l3089_308996


namespace NUMINAMATH_CALUDE_book_profit_percentage_l3089_308918

/-- Given a book's cost price and additional information about its profit, 
    calculate the initial profit percentage. -/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (additional_profit : ℝ) 
  (new_profit_percentage : ℝ) :
  cost_price = 2400 →
  additional_profit = 120 →
  new_profit_percentage = 15 →
  ∃ (initial_profit_percentage : ℝ),
    initial_profit_percentage = 10 ∧
    cost_price * (1 + new_profit_percentage / 100) = 
      cost_price * (1 + initial_profit_percentage / 100) + additional_profit :=
by sorry

end NUMINAMATH_CALUDE_book_profit_percentage_l3089_308918


namespace NUMINAMATH_CALUDE_place_value_ratio_l3089_308958

/-- The number we're analyzing -/
def number : ℚ := 90347.6208

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ := (digit : ℚ) * 10 ^ position

/-- The position of the digit 0 in the number (counting from right, with decimal point at 0) -/
def zero_position : ℤ := 4

/-- The position of the digit 6 in the number (counting from right, with decimal point at 0) -/
def six_position : ℤ := -1

theorem place_value_ratio :
  place_value 1 zero_position / place_value 1 six_position = 100000 := by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l3089_308958


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3089_308977

theorem point_in_fourth_quadrant :
  let P : ℝ × ℝ := (Real.tan (2015 * π / 180), Real.cos (2015 * π / 180))
  (0 < P.1) ∧ (P.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3089_308977


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_mod_4_l3089_308932

def binary_number : ℕ := 111100010111

theorem remainder_of_binary_number_mod_4 :
  binary_number % 4 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_mod_4_l3089_308932


namespace NUMINAMATH_CALUDE_maggies_tractor_rate_l3089_308963

/-- Maggie's weekly income calculation --/
theorem maggies_tractor_rate (office_rate : ℝ) (office_hours tractor_hours total_income : ℝ) :
  office_rate = 10 →
  office_hours = 2 * tractor_hours →
  tractor_hours = 13 →
  total_income = 416 →
  total_income = office_rate * office_hours + tractor_hours * (total_income - office_rate * office_hours) / tractor_hours →
  (total_income - office_rate * office_hours) / tractor_hours = 12 := by
sorry


end NUMINAMATH_CALUDE_maggies_tractor_rate_l3089_308963


namespace NUMINAMATH_CALUDE_expression_evaluation_l3089_308931

theorem expression_evaluation : 
  let x : ℚ := -2
  let y : ℚ := 1/2
  ((x + 2*y)^2 - (x + y)*(3*x - y) - 5*y^2) / (2*x) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3089_308931


namespace NUMINAMATH_CALUDE_pencil_length_l3089_308968

theorem pencil_length (black_fraction : Real) (white_fraction : Real) (blue_length : Real) :
  black_fraction = 1/8 →
  white_fraction = 1/2 →
  blue_length = 3.5 →
  ∃ (total_length : Real),
    total_length * black_fraction +
    (total_length - total_length * black_fraction) * white_fraction +
    blue_length = total_length ∧
    total_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3089_308968


namespace NUMINAMATH_CALUDE_combination_equation_solution_l3089_308923

theorem combination_equation_solution (n : ℕ) : 
  Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l3089_308923


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3089_308941

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 10 players where each player plays every other player once,
    the total number of games played is 45. --/
theorem chess_tournament_games :
  num_games 10 = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3089_308941


namespace NUMINAMATH_CALUDE_num_divisors_5400_multiple_of_5_l3089_308950

/-- The number of positive divisors of 5400 that are multiples of 5 -/
def num_divisors_multiple_of_5 (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 5400 that are multiples of 5 is 24 -/
theorem num_divisors_5400_multiple_of_5 :
  num_divisors_multiple_of_5 5400 = 24 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_5400_multiple_of_5_l3089_308950


namespace NUMINAMATH_CALUDE_ellipse_circle_relation_l3089_308922

/-- An ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  r : ℝ
  h_pos : r > 0

/-- A line in the Cartesian coordinate system -/
structure Line where
  k : ℝ
  m : ℝ

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 = c.r^2

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  l.m^2 = c.r^2 * (1 + l.k^2)

/-- Checks if three points form a right angle -/
def is_right_angle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem ellipse_circle_relation 
  (e : Ellipse) (c : Circle) (l : Line) (A B : Point) :
  c.r < e.b →
  is_tangent l c →
  on_ellipse A e ∧ on_ellipse B e →
  on_circle A c ∧ on_circle B c →
  is_right_angle A B (Point.mk 0 0) →
  1 / e.a^2 + 1 / e.b^2 = 1 / c.r^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_relation_l3089_308922


namespace NUMINAMATH_CALUDE_am_hm_difference_bound_l3089_308912

theorem am_hm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  ((x - y)^2) / (2*(x + y)) < ((y - x)^2) / (8*x) := by
  sorry

end NUMINAMATH_CALUDE_am_hm_difference_bound_l3089_308912


namespace NUMINAMATH_CALUDE_max_sum_of_squares_difference_l3089_308982

theorem max_sum_of_squares_difference (x y : ℕ+) : 
  x^2 - y^2 = 2016 → x + y ≤ 1008 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_difference_l3089_308982


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3089_308993

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 511 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3089_308993


namespace NUMINAMATH_CALUDE_bus_trip_speed_l3089_308917

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 450 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l3089_308917


namespace NUMINAMATH_CALUDE_product_equality_l3089_308949

theorem product_equality : 3.6 * 0.4 * 1.5 = 2.16 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3089_308949


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3089_308928

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "Individual A gets the red card"
def EventA (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Individual B gets the red card"
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3089_308928


namespace NUMINAMATH_CALUDE_triangle_property_l3089_308938

/-- Given a triangle ABC where a^2 + c^2 = b^2 + √2*a*c, prove that:
    1. The size of angle B is π/4
    2. The maximum value of √2*cos(A) + cos(C) is 1 -/
theorem triangle_property (a b c : ℝ) (h : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  (B = π / 4) ∧
  (∃ (x : ℝ), Real.sqrt 2 * Real.cos A + Real.cos C ≤ x ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3089_308938


namespace NUMINAMATH_CALUDE_smallest_n_g_greater_than_20_l3089_308937

/-- The sum of digits to the right of the decimal point in 1/(3^n) -/
def g (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 4 is the smallest positive integer n such that g(n) > 20 -/
theorem smallest_n_g_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 := by sorry

end NUMINAMATH_CALUDE_smallest_n_g_greater_than_20_l3089_308937


namespace NUMINAMATH_CALUDE_series_duration_l3089_308988

theorem series_duration (episode1 episode2 episode3 episode4 : ℕ) 
  (h1 : episode1 = 58)
  (h2 : episode2 = 62)
  (h3 : episode3 = 65)
  (h4 : episode4 = 55) :
  (episode1 + episode2 + episode3 + episode4) / 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_series_duration_l3089_308988


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l3089_308991

/-- The perimeter of a hexagon ABCDEF where five sides are of length 1 and the sixth side is √5 -/
theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AF : ℝ) 
  (h1 : AB = 1) (h2 : BC = 1) (h3 : CD = 1) (h4 : DE = 1) (h5 : EF = 1)
  (h6 : AF = Real.sqrt 5) : AB + BC + CD + DE + EF + AF = 5 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l3089_308991


namespace NUMINAMATH_CALUDE_computer_cost_is_400_l3089_308944

/-- The cost of the computer Delores bought -/
def computer_cost : ℕ := sorry

/-- The initial amount of money Delores had -/
def initial_money : ℕ := 450

/-- The combined cost of the computer and printer -/
def total_purchase : ℕ := 40

/-- The amount of money Delores had left after the purchase -/
def money_left : ℕ := 10

/-- Theorem stating that the computer cost $400 -/
theorem computer_cost_is_400 :
  computer_cost = 400 :=
by sorry

end NUMINAMATH_CALUDE_computer_cost_is_400_l3089_308944


namespace NUMINAMATH_CALUDE_workshop_pairing_probability_l3089_308948

/-- The probability of a specific pairing in a group of participants. -/
def specific_pairing_probability (total_participants : ℕ) : ℚ :=
  if total_participants ≤ 1 then 0
  else 1 / (total_participants - 1 : ℚ)

/-- Theorem: In a workshop with 24 participants, the probability of John pairing with Alice is 1/23. -/
theorem workshop_pairing_probability :
  specific_pairing_probability 24 = 1 / 23 := by
  sorry


end NUMINAMATH_CALUDE_workshop_pairing_probability_l3089_308948


namespace NUMINAMATH_CALUDE_south_opposite_of_north_l3089_308967

/-- Represents the direction of movement --/
inductive Direction
  | North
  | South

/-- Represents a distance with direction --/
structure DirectedDistance where
  distance : ℝ
  direction : Direction

/-- Denotes a distance in kilometers with a sign --/
def denote (d : DirectedDistance) : ℝ :=
  match d.direction with
  | Direction.North => d.distance
  | Direction.South => -d.distance

theorem south_opposite_of_north 
  (h : denote { distance := 3, direction := Direction.North } = 3) :
  denote { distance := 5, direction := Direction.South } = -5 := by
  sorry


end NUMINAMATH_CALUDE_south_opposite_of_north_l3089_308967


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3089_308919

theorem trigonometric_identities (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2) : 
  (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 5 ∧ 
  Real.sin α * Real.cos α + 2 = 23 / 10 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3089_308919


namespace NUMINAMATH_CALUDE_rope_length_increase_l3089_308995

/-- Proves that increasing a circular area with initial radius 10 m by 942.8571428571429 m² results in a new radius of 20 m -/
theorem rope_length_increase (π : Real) (initial_radius : Real) (area_increase : Real) (new_radius : Real) : 
  π > 0 → 
  initial_radius = 10 → 
  area_increase = 942.8571428571429 → 
  new_radius = 20 → 
  π * new_radius^2 = π * initial_radius^2 + area_increase := by
  sorry

#check rope_length_increase

end NUMINAMATH_CALUDE_rope_length_increase_l3089_308995


namespace NUMINAMATH_CALUDE_a_plus_b_values_l3089_308940

/-- A strictly increasing sequence of positive integers -/
def StrictlyIncreasingPositiveSeq (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

/-- The theorem statement -/
theorem a_plus_b_values
  (a b : ℕ → ℕ)
  (h_a_incr : StrictlyIncreasingPositiveSeq a)
  (h_b_incr : StrictlyIncreasingPositiveSeq b)
  (h_eq : a 10 = b 10)
  (h_lt_2017 : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  (a 1 + b 1 = 13) ∨ (a 1 + b 1 = 20) :=
sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l3089_308940


namespace NUMINAMATH_CALUDE_function_inequality_l3089_308987

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + 2 * (deriv f x) > 0) : 
  f 1 > f 0 / Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3089_308987


namespace NUMINAMATH_CALUDE_negation_equivalence_l3089_308971

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔
  (∀ x : ℝ, x ≥ Real.sin x ∧ x ≤ Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3089_308971


namespace NUMINAMATH_CALUDE_vector_sum_proof_l3089_308970

/-- Given two vectors in a plane, prove that their sum with specific coefficients equals a certain vector. -/
theorem vector_sum_proof (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (a • b = 0) → 
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, 7] :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l3089_308970


namespace NUMINAMATH_CALUDE_car_speeds_satisfy_conditions_l3089_308936

/-- Represents the scenario of two cars meeting on a road --/
structure CarMeetingScenario where
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ
  speed_increase1 : ℝ
  speed_increase2 : ℝ
  time_difference : ℝ

/-- Checks if the given car speeds satisfy the meeting conditions --/
def satisfies_conditions (s : CarMeetingScenario) : Prop :=
  s.distance / (s.speed1 - s.speed2) - s.distance / ((s.speed1 + s.speed_increase1) - (s.speed2 + s.speed_increase2)) = s.time_difference

/-- The theorem stating that the given speeds satisfy the conditions --/
theorem car_speeds_satisfy_conditions : ∃ (s : CarMeetingScenario),
  s.distance = 60 ∧
  s.speed_increase1 = 10 ∧
  s.speed_increase2 = 8 ∧
  s.time_difference = 1 ∧
  s.speed1 = 50 ∧
  s.speed2 = 40 ∧
  satisfies_conditions s := by
  sorry


end NUMINAMATH_CALUDE_car_speeds_satisfy_conditions_l3089_308936


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3089_308933

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) :
  let set := List.replicate (n - 2) 1 ++ List.replicate 2 (1 - 1 / n)
  (List.sum set) / n = 1 - 2 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3089_308933


namespace NUMINAMATH_CALUDE_average_first_ten_even_numbers_l3089_308914

theorem average_first_ten_even_numbers :
  let first_ten_even : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_ten_even.sum / first_ten_even.length : ℚ) = 11 := by
sorry

end NUMINAMATH_CALUDE_average_first_ten_even_numbers_l3089_308914


namespace NUMINAMATH_CALUDE_root_cubic_expression_l3089_308934

theorem root_cubic_expression (m : ℝ) : 
  m^2 + 3*m - 2023 = 0 → m^3 + 2*m^2 - 2026*m - 2023 = -4046 := by
  sorry

end NUMINAMATH_CALUDE_root_cubic_expression_l3089_308934


namespace NUMINAMATH_CALUDE_complex_magnitude_l3089_308935

theorem complex_magnitude (z : ℂ) (h : Complex.I - z = 1 + 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3089_308935


namespace NUMINAMATH_CALUDE_height_prediction_at_10_l3089_308990

/-- Regression model for child height based on age -/
def height_model (x : ℝ) : ℝ := 7.2 * x + 74

/-- The model is valid for children aged 3 to 9 years -/
def valid_age_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

/-- Prediction is considered approximate if within 1cm of the calculated value -/
def is_approximate (predicted : ℝ) (actual : ℝ) : Prop := abs (predicted - actual) ≤ 1

theorem height_prediction_at_10 :
  is_approximate (height_model 10) 146 :=
sorry

end NUMINAMATH_CALUDE_height_prediction_at_10_l3089_308990


namespace NUMINAMATH_CALUDE_intersection_A_B_l3089_308999

def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3089_308999


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l3089_308905

-- Define constants
def serving_size : ℚ := 1/2
def servings_per_day : ℕ := 3
def quart_in_ounces : ℕ := 32
def container_size_difference : ℕ := 2

-- Define the container size
def container_size : ℕ := quart_in_ounces - container_size_difference

-- Define daily usage
def daily_usage : ℚ := serving_size * servings_per_day

-- Theorem to prove
theorem hot_sauce_duration :
  (container_size : ℚ) / daily_usage = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l3089_308905


namespace NUMINAMATH_CALUDE_reimu_win_probability_l3089_308942

/-- Represents the result of a single coin toss -/
inductive CoinSide
| Red
| Green

/-- Represents the state of a single coin -/
structure Coin :=
  (side1 : CoinSide)
  (side2 : CoinSide)

/-- Represents the game state -/
structure GameState :=
  (coins : Finset Coin)

/-- The number of coins in the game -/
def numCoins : Nat := 4

/-- A game is valid if it has the correct number of coins -/
def validGame (g : GameState) : Prop :=
  g.coins.card = numCoins

/-- The probability of Reimu winning the game -/
def reimuWinProbability (g : GameState) : ℚ :=
  sorry

/-- The main theorem: probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (g : GameState) (h : validGame g) : 
  reimuWinProbability g = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_reimu_win_probability_l3089_308942


namespace NUMINAMATH_CALUDE_expression_evaluation_l3089_308930

theorem expression_evaluation : 
  Real.sqrt ((-3)^2) + (π - 3)^0 - 8^(2/3) + ((-4)^(1/3))^3 = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3089_308930


namespace NUMINAMATH_CALUDE_degree_of_our_monomial_l3089_308913

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : String) : ℕ :=
  sorry

/-- The monomial -2/5 * x^2 * y -/
def our_monomial : String := "-2/5x^2y"

theorem degree_of_our_monomial :
  degree_of_monomial our_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_our_monomial_l3089_308913


namespace NUMINAMATH_CALUDE_square_of_three_tenths_plus_one_tenth_l3089_308965

theorem square_of_three_tenths_plus_one_tenth (ε : Real) :
  (0.3 : Real)^2 + 0.1 = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_square_of_three_tenths_plus_one_tenth_l3089_308965


namespace NUMINAMATH_CALUDE_diamond_two_four_l3089_308926

-- Define the Diamond operation
def Diamond (a b : ℤ) : ℤ := a * b^3 - b + 2

-- Theorem statement
theorem diamond_two_four : Diamond 2 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_four_l3089_308926


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_seven_l3089_308945

theorem complex_expression_equals_negative_seven :
  Real.sqrt 8 + (1/2)⁻¹ - 4 * Real.cos (45 * π / 180) - 2 / (1/2) * 2 - (2009 - Real.sqrt 3)^0 = -7 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_seven_l3089_308945


namespace NUMINAMATH_CALUDE_divisor_count_problem_l3089_308959

def τ (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_count_problem :
  (Finset.filter (fun n => τ n > 2 ∧ τ (τ n) = 2) (Finset.range 1001)).card = 184 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_problem_l3089_308959


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l3089_308961

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : travel_time = 4) : 
  boat_speed + stream_speed * travel_time = 108 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l3089_308961


namespace NUMINAMATH_CALUDE_two_negative_factors_l3089_308969

theorem two_negative_factors
  (a b c : ℚ)
  (h : a * b * c > 0) :
  (a < 0 ∧ b < 0 ∧ c > 0) ∨
  (a < 0 ∧ b > 0 ∧ c < 0) ∨
  (a > 0 ∧ b < 0 ∧ c < 0) :=
sorry

end NUMINAMATH_CALUDE_two_negative_factors_l3089_308969


namespace NUMINAMATH_CALUDE_certain_number_proof_l3089_308983

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 6) 
  (h2 : p - q = 0.3) : 
  3 / q = 15 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3089_308983


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3089_308994

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3089_308994


namespace NUMINAMATH_CALUDE_christian_future_age_l3089_308903

/-- The age of Brian in years -/
def brian_age : ℕ := sorry

/-- The age of Christian in years -/
def christian_age : ℕ := sorry

/-- The number of years in the future we're considering -/
def years_future : ℕ := 8

/-- Brian's age in the future -/
def brian_future_age : ℕ := 40

theorem christian_future_age :
  christian_age + years_future = 72 :=
by
  have h1 : christian_age = 2 * brian_age := sorry
  have h2 : brian_age + years_future = brian_future_age := sorry
  sorry

end NUMINAMATH_CALUDE_christian_future_age_l3089_308903


namespace NUMINAMATH_CALUDE_zachary_pushups_l3089_308929

/-- Given the information about David and Zachary's push-ups and crunches, 
    prove that Zachary did 28 push-ups. -/
theorem zachary_pushups (david_pushups zachary_pushups david_crunches zachary_crunches : ℕ) :
  david_pushups = zachary_pushups + 40 →
  david_crunches + 17 = zachary_crunches →
  david_crunches = 45 →
  zachary_crunches = 62 →
  zachary_pushups = 28 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l3089_308929


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3089_308984

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3089_308984


namespace NUMINAMATH_CALUDE_f_shifted_l3089_308904

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_shifted (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 3) :
  ∃ y, f (y - 1) = 2 * y - 1 ∧ 2 ≤ y ∧ y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_f_shifted_l3089_308904


namespace NUMINAMATH_CALUDE_employee_salary_l3089_308920

theorem employee_salary (total_salary : ℝ) (m_salary_percentage : ℝ) 
  (h1 : total_salary = 616)
  (h2 : m_salary_percentage = 120 / 100) : 
  ∃ (n_salary : ℝ), 
    n_salary + m_salary_percentage * n_salary = total_salary ∧ 
    n_salary = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_l3089_308920


namespace NUMINAMATH_CALUDE_complex_real_condition_l3089_308964

theorem complex_real_condition (z : ℂ) : (z + 2*I).im = 0 ↔ z.im = -2 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3089_308964


namespace NUMINAMATH_CALUDE_matrix_determinant_solution_l3089_308974

theorem matrix_determinant_solution (a : ℝ) (ha : a ≠ 0) :
  let matrix (x : ℝ) := !![x + a, 2*x, 2*x; 2*x, x + a, 2*x; 2*x, 2*x, x + a]
  ∀ x : ℝ, Matrix.det (matrix x) = 0 ↔ x = -a ∨ x = a/3 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_solution_l3089_308974


namespace NUMINAMATH_CALUDE_max_female_to_male_ratio_l3089_308960

/-- Proves that the maximum ratio of female to male students is 4:1 given the problem conditions -/
theorem max_female_to_male_ratio :
  ∀ (female_count male_count bench_count : ℕ),
  male_count = 29 →
  bench_count = 29 →
  ∃ (x : ℕ), female_count = x * male_count →
  female_count + male_count ≤ bench_count * 5 →
  x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_female_to_male_ratio_l3089_308960


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l3089_308901

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Proves that under the given exchange rate and spending conditions, 
    the sum of digits of the initial U.S. dollars is 8 -/
theorem exchange_rate_problem (d : ℕ) : 
  (8 * d) / 5 - 75 = d → sum_of_digits d = 8 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l3089_308901


namespace NUMINAMATH_CALUDE_twenty_fives_sum_1000_l3089_308951

/-- A list of integers representing a grouping of fives -/
def Grouping : Type := List Nat

/-- The number of fives in a grouping -/
def count_fives : Grouping → Nat
  | [] => 0
  | (x::xs) => (x.digits 10).length + count_fives xs

/-- The sum of a grouping -/
def sum_grouping : Grouping → Nat
  | [] => 0
  | (x::xs) => x + sum_grouping xs

/-- A valid grouping of 20 fives that sums to 1000 -/
theorem twenty_fives_sum_1000 : ∃ (g : Grouping), 
  (count_fives g = 20) ∧ (sum_grouping g = 1000) := by
  sorry

end NUMINAMATH_CALUDE_twenty_fives_sum_1000_l3089_308951


namespace NUMINAMATH_CALUDE_shaded_area_of_square_l3089_308976

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    prove that its area is 50 square cm. -/
theorem shaded_area_of_square (d : ℝ) (n : ℕ) (h1 : d = 10) (h2 : n = 25) :
  (d^2 / 2 : ℝ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_l3089_308976


namespace NUMINAMATH_CALUDE_student_base_choices_l3089_308909

/-- The number of bases available for students to choose from -/
def num_bases : ℕ := 4

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

theorem student_base_choices : total_ways = 256 := by
  sorry

end NUMINAMATH_CALUDE_student_base_choices_l3089_308909


namespace NUMINAMATH_CALUDE_first_bell_weight_l3089_308907

theorem first_bell_weight (w : ℝ) 
  (h1 : w > 0)  -- Ensuring positive weight
  (h2 : 2 * w > 0)  -- Weight of second bell
  (h3 : 4 * (2 * w) > 0)  -- Weight of third bell
  (h4 : w + 2 * w + 4 * (2 * w) = 550)  -- Total weight condition
  : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_bell_weight_l3089_308907


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3089_308902

theorem arithmetic_calculation : 6^2 - 4*5 + 2^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3089_308902


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3089_308939

/-- Given a rectangle with integer length and width, and a perimeter of 120 units,
    the minimum possible area is 59 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 120) → (l * w ≥ 59) := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3089_308939


namespace NUMINAMATH_CALUDE_bakers_cakes_l3089_308943

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes sold_cakes final_cakes : ℕ) 
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111) :
  final_cakes - (initial_cakes - sold_cakes) = 76 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3089_308943


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3089_308981

theorem fraction_to_decimal : 58 / 125 = 0.464 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3089_308981


namespace NUMINAMATH_CALUDE_projection_matrix_condition_l3089_308927

/-- A 2x2 matrix is a projection matrix if and only if its square equals itself. -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix we're working with -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 7/17; c, 10/17]

/-- The main theorem: P is a projection matrix if and only if a = 9/17 and c = 10/17 -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 9/17 ∧ c = 10/17 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_condition_l3089_308927


namespace NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l3089_308900

/-- Represents a table with crosses --/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if a row has an odd number of crosses --/
def rowHasOddCrosses (t : CrossTable m n) (i : Fin m) : Prop :=
  (Finset.filter (λ j => t.crosses i j) (Finset.univ : Finset (Fin n))).card % 2 = 1

/-- Checks if a column has an odd number of crosses --/
def colHasOddCrosses (t : CrossTable m n) (j : Fin n) : Prop :=
  (Finset.filter (λ i => t.crosses i j) (Finset.univ : Finset (Fin m))).card % 2 = 1

/-- Checks if all rows and columns have odd number of crosses --/
def allOddCrosses (t : CrossTable m n) : Prop :=
  (∀ i, rowHasOddCrosses t i) ∧ (∀ j, colHasOddCrosses t j)

/-- Counts the total number of crosses in the table --/
def totalCrosses (t : CrossTable m n) : ℕ :=
  (Finset.filter (λ (i, j) => t.crosses i j) (Finset.univ : Finset (Fin m × Fin n))).card

/-- Theorem: The maximum number of crosses in a 4x10 table with odd crosses in each row and column is 30 --/
theorem max_crosses_4x10 :
  (∃ t : CrossTable 4 10, allOddCrosses t ∧ totalCrosses t = 30) ∧
  (∀ t : CrossTable 4 10, allOddCrosses t → totalCrosses t ≤ 30) := by sorry

/-- Theorem: It's impossible to place crosses in a 5x10 table with odd crosses in each row and column --/
theorem impossible_5x10 :
  ¬ ∃ t : CrossTable 5 10, allOddCrosses t := by sorry

end NUMINAMATH_CALUDE_max_crosses_4x10_impossible_5x10_l3089_308900


namespace NUMINAMATH_CALUDE_friends_receiving_pebbles_l3089_308957

def pebbles_per_dozen : ℕ := 12

theorem friends_receiving_pebbles 
  (total_dozens : ℕ) 
  (pebbles_per_friend : ℕ) 
  (h1 : total_dozens = 3) 
  (h2 : pebbles_per_friend = 4) : 
  (total_dozens * pebbles_per_dozen) / pebbles_per_friend = 9 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pebbles_l3089_308957


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3089_308989

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a :=
sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + (-11) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3089_308989


namespace NUMINAMATH_CALUDE_area_ratio_of_rectangles_l3089_308973

/-- Given two rectangles A and B with specified dimensions, prove that the ratio of their areas is 12/21 -/
theorem area_ratio_of_rectangles (length_A width_A length_B width_B : ℕ) 
  (h1 : length_A = 36) (h2 : width_A = 20) (h3 : length_B = 42) (h4 : width_B = 30) :
  (length_A * width_A : ℚ) / (length_B * width_B) = 12 / 21 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_rectangles_l3089_308973


namespace NUMINAMATH_CALUDE_fraction_sum_constraint_l3089_308980

theorem fraction_sum_constraint (n : ℕ) (hn : n > 0) :
  (1 : ℚ) / 2 + 1 / 3 + 1 / 10 + 1 / n < 1 → n > 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_constraint_l3089_308980


namespace NUMINAMATH_CALUDE_sin_max_implies_even_l3089_308906

theorem sin_max_implies_even (f : ℝ → ℝ) (φ a : ℝ) 
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ f a) :
  ∀ x, f (x + a) = f (-x + a) := by
sorry

end NUMINAMATH_CALUDE_sin_max_implies_even_l3089_308906


namespace NUMINAMATH_CALUDE_M_perfect_square_divisors_l3089_308915

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The product M as defined in the problem -/
def M : ℕ := (factorial 1) * (factorial 2) * (factorial 3) * (factorial 4) * 
              (factorial 5) * (factorial 6) * (factorial 7) * (factorial 8) * (factorial 9)

/-- Count of perfect square divisors of a natural number -/
def count_perfect_square_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that M has 672 perfect square divisors -/
theorem M_perfect_square_divisors : count_perfect_square_divisors M = 672 := by sorry

end NUMINAMATH_CALUDE_M_perfect_square_divisors_l3089_308915


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l3089_308954

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + 2*b = 0) → 
  (∃ x : ℝ, x^2 + 2*b*x + a = 0) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x : ℝ, x^2 + a'*x + 2*b' = 0) → 
    (∃ x : ℝ, x^2 + 2*b'*x + a' = 0) → 
    a' + b' ≥ a + b) → 
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l3089_308954


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3089_308947

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100 + (n.1 : ℚ) / 1000 + (n.2 : ℚ) / 10000

theorem two_digit_number_problem (cd : TwoDigitNumber) :
  72 * toRepeatingDecimal cd - 72 * (1 + toDecimal cd) = 0.8 → cd = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3089_308947


namespace NUMINAMATH_CALUDE_points_collinear_l3089_308916

-- Define the function for log base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- Define the function for log base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the line passing through the origin
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the theorem
theorem points_collinear (k a b : ℝ) 
  (ha : line_through_origin k a = log8 a)
  (hb : line_through_origin k b = log8 b)
  (hc : ∃ c, c = (a, log2 a))
  (hd : ∃ d, d = (b, log2 b)) :
  ∃ m, line_through_origin m a = log2 a ∧ 
       line_through_origin m b = log2 b ∧
       line_through_origin m 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_points_collinear_l3089_308916


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3089_308956

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 3 * x - 4 * y = -2 ∧ 4 * x + 5 * y = 23 ∧ x = 82/31 ∧ y = 77/31 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3089_308956


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3089_308952

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

theorem ellipse_fixed_point_intersection
  (C : Ellipse)
  (h_point : (1 : ℝ)^2 / C.a^2 + (Real.sqrt 6 / 3)^2 / C.b^2 = 1)
  (h_eccentricity : Real.sqrt (C.a^2 - C.b^2) / C.a = Real.sqrt 6 / 3)
  (l : Line)
  (P Q : Point)
  (h_intersect_P : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_intersect_Q : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1)
  (h_on_line_P : P.y = l.m * P.x + l.c)
  (h_on_line_Q : Q.y = l.m * Q.x + l.c)
  (h_perpendicular : P.x * Q.x + P.y * Q.y = 0)
  (h_not_vertex : l.m ≠ 0 ∨ l.c ≠ 1) :
  l.m * 0 + l.c = -1/2 := by sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_intersection_l3089_308952


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l3089_308986

theorem stratified_sampling_problem (total_population : ℕ) 
  (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 55) 
  (h2 : stratum_size = 15) (h3 : stratum_sample = 3) :
  (stratum_sample : ℚ) * total_population / stratum_size = 11 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l3089_308986


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3089_308978

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l3089_308978


namespace NUMINAMATH_CALUDE_every_nat_sum_of_two_three_powers_l3089_308946

def is_power_of_two_three (n : ℕ) : Prop :=
  ∃ (α β : ℕ), n = 2^α * 3^β

def summands_not_multiples (s : List ℕ) : Prop :=
  ∀ i j, i ≠ j → i < s.length → j < s.length →
    ¬(s.get ⟨i, by sorry⟩ ∣ s.get ⟨j, by sorry⟩) ∧
    ¬(s.get ⟨j, by sorry⟩ ∣ s.get ⟨i, by sorry⟩)

theorem every_nat_sum_of_two_three_powers :
  ∀ n : ℕ, n > 0 →
    ∃ s : List ℕ,
      (∀ m ∈ s.toFinset, is_power_of_two_three m) ∧
      (s.sum = n) ∧
      summands_not_multiples s :=
sorry

end NUMINAMATH_CALUDE_every_nat_sum_of_two_three_powers_l3089_308946


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3089_308921

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l3089_308921


namespace NUMINAMATH_CALUDE_class_size_l3089_308998

theorem class_size (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : num_groups = 5) 
  (h2 : students_per_group = 6) : 
  num_groups * students_per_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3089_308998


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l3089_308992

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ), n > 2 →
  (∃ (angle : ℝ), angle = 150 ∧ angle * n = 180 * (n - 2)) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l3089_308992


namespace NUMINAMATH_CALUDE_px_length_l3089_308953

-- Define the quadrilateral CDXW
structure Quadrilateral :=
  (C D W X P : ℝ × ℝ)
  (cd_parallel_wx : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (p_on_cx : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (X.1 - C.1), C.2 + t * (X.2 - C.2)))
  (p_on_dw : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (D.1 + s * (W.1 - D.1), D.2 + s * (W.2 - D.2)))
  (cx_length : Real.sqrt ((X.1 - C.1)^2 + (X.2 - C.2)^2) = 30)
  (dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 15)
  (pw_length : Real.sqrt ((W.1 - P.1)^2 + (W.2 - P.2)^2) = 45)

-- Theorem statement
theorem px_length (q : Quadrilateral) : 
  Real.sqrt ((q.X.1 - q.P.1)^2 + (q.X.2 - q.P.2)^2) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_px_length_l3089_308953


namespace NUMINAMATH_CALUDE_syrup_dilution_l3089_308955

theorem syrup_dilution (x : ℝ) : 
  (0 < x) ∧ 
  (x < 1000) ∧ 
  ((1000 - 2*x) * (1000 - x) = 120000) → 
  x = 400 :=
by sorry

end NUMINAMATH_CALUDE_syrup_dilution_l3089_308955


namespace NUMINAMATH_CALUDE_length_of_AB_l3089_308972

/-- Given a line segment AB with points P and Q, prove that AB has length 70 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are in AB and on the same side of midpoint
  (P - A) / (B - P) = 2 / 3 →        -- P divides AB in ratio 2:3
  (Q - A) / (B - Q) = 3 / 4 →        -- Q divides AB in ratio 3:4
  Q - P = 2 →                        -- PQ = 2
  B - A = 70 := by                   -- AB has length 70
sorry


end NUMINAMATH_CALUDE_length_of_AB_l3089_308972


namespace NUMINAMATH_CALUDE_new_ratio_is_two_to_three_l3089_308910

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numer : ℚ)
  (denom : ℚ)

/-- The initial ratio of acid to base -/
def initialRatio : Ratio := ⟨4, 1⟩

/-- The initial volume of acid in litres -/
def initialAcidVolume : ℚ := 16

/-- The volume of mixture taken out in litres -/
def volumeTakenOut : ℚ := 10

/-- The volume of base added in litres -/
def volumeBaseAdded : ℚ := 10

/-- Calculate the new ratio of acid to base after the replacement -/
def newRatio : Ratio :=
  let initialBaseVolume := initialAcidVolume / initialRatio.numer * initialRatio.denom
  let totalInitialVolume := initialAcidVolume + initialBaseVolume
  let acidRemoved := volumeTakenOut * (initialRatio.numer / (initialRatio.numer + initialRatio.denom))
  let baseRemoved := volumeTakenOut * (initialRatio.denom / (initialRatio.numer + initialRatio.denom))
  let remainingAcid := initialAcidVolume - acidRemoved
  let remainingBase := initialBaseVolume - baseRemoved + volumeBaseAdded
  ⟨remainingAcid, remainingBase⟩

theorem new_ratio_is_two_to_three :
  newRatio = ⟨2, 3⟩ := by sorry


end NUMINAMATH_CALUDE_new_ratio_is_two_to_three_l3089_308910


namespace NUMINAMATH_CALUDE_biology_score_is_85_l3089_308997

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def average_score : ℕ := 71
def total_subjects : ℕ := 5

def biology_score : ℕ := 
  average_score * total_subjects - (mathematics_score + science_score + social_studies_score + english_score)

theorem biology_score_is_85 : biology_score = 85 := by sorry

end NUMINAMATH_CALUDE_biology_score_is_85_l3089_308997


namespace NUMINAMATH_CALUDE_xiaohong_mother_age_l3089_308979

/-- Xiaohong's age when her mother was her current age -/
def xiaohong_age_then : ℕ := 3

/-- Xiaohong's mother's future age when Xiaohong will be her mother's current age -/
def mother_age_future : ℕ := 78

/-- The age difference between Xiaohong and her mother -/
def age_difference : ℕ := mother_age_future - xiaohong_age_then

/-- Xiaohong's current age -/
def xiaohong_age_now : ℕ := age_difference + xiaohong_age_then

/-- Xiaohong's mother's current age -/
def mother_age_now : ℕ := mother_age_future - age_difference

theorem xiaohong_mother_age : mother_age_now = 53 := by
  sorry

#eval mother_age_now

end NUMINAMATH_CALUDE_xiaohong_mother_age_l3089_308979


namespace NUMINAMATH_CALUDE_a_investment_l3089_308925

/-- A partnership business with three partners A, B, and C. -/
structure Partnership where
  investA : ℕ
  investB : ℕ
  investC : ℕ
  totalProfit : ℕ
  cShareProfit : ℕ

/-- The partnership satisfies the given conditions -/
def validPartnership (p : Partnership) : Prop :=
  p.investB = 8000 ∧
  p.investC = 9000 ∧
  p.totalProfit = 88000 ∧
  p.cShareProfit = 36000

/-- The theorem stating A's investment amount -/
theorem a_investment (p : Partnership) (h : validPartnership p) : 
  p.investA = 5000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_l3089_308925


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3089_308924

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 7 + y + 9) / 6 = 12 → y = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3089_308924


namespace NUMINAMATH_CALUDE_problem_solution_l3089_308985

def δ (x : ℝ) : ℝ := 3 * x + 8
def φ (x : ℝ) : ℝ := 9 * x + 7

theorem problem_solution (x : ℝ) : δ (φ x) = 11 → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3089_308985


namespace NUMINAMATH_CALUDE_grass_seed_cost_l3089_308911

/-- The cost of a 5-pound bag of grass seed -/
def cost_5lb : ℝ := 13.80

/-- The cost of a 10-pound bag of grass seed -/
def cost_10lb : ℝ := 20.43

/-- The cost of a 25-pound bag of grass seed -/
def cost_25lb : ℝ := 32.25

/-- The minimum amount of grass seed the customer must buy (in pounds) -/
def min_amount : ℝ := 65

/-- The maximum amount of grass seed the customer can buy (in pounds) -/
def max_amount : ℝ := 80

/-- The least possible cost for the customer -/
def least_cost : ℝ := 98.73

theorem grass_seed_cost : 
  2 * cost_25lb + cost_10lb + cost_5lb = least_cost ∧ 
  2 * 25 + 10 + 5 ≥ min_amount ∧
  2 * 25 + 10 + 5 ≤ max_amount := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l3089_308911


namespace NUMINAMATH_CALUDE_nabla_calculation_l3089_308908

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3089_308908


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l3089_308962

theorem smallest_n_for_irreducible_fractions : ∃ (n : ℕ), 
  (n = 95) ∧ 
  (∀ (k : ℕ), 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ (m : ℕ), m < n → ∃ (k : ℕ), 19 ≤ k ∧ k ≤ 91 ∧ Nat.gcd k (m + k + 2) ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l3089_308962


namespace NUMINAMATH_CALUDE_ajax_initial_weight_ajax_initial_weight_is_80_l3089_308966

/-- Proves that Ajax's initial weight is 80 kg given the exercise and weight conditions --/
theorem ajax_initial_weight : ℝ → Prop :=
  fun (initial_weight : ℝ) =>
    let pounds_per_kg : ℝ := 2.2
    let weight_loss_per_hour : ℝ := 1.5
    let hours_per_day : ℝ := 2
    let days : ℝ := 14
    let final_weight_pounds : ℝ := 134
    
    let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
    let initial_weight_pounds : ℝ := final_weight_pounds + total_weight_loss
    
    initial_weight = initial_weight_pounds / pounds_per_kg ∧ initial_weight = 80

theorem ajax_initial_weight_is_80 : ajax_initial_weight 80 := by
  sorry

end NUMINAMATH_CALUDE_ajax_initial_weight_ajax_initial_weight_is_80_l3089_308966


namespace NUMINAMATH_CALUDE_dean_transactions_l3089_308975

theorem dean_transactions (mabel anthony cal jade dean : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 14 →
  dean = jade + (jade / 4) →
  dean = 100 := by
sorry

end NUMINAMATH_CALUDE_dean_transactions_l3089_308975
