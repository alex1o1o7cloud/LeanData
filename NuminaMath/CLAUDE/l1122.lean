import Mathlib

namespace NUMINAMATH_CALUDE_meal_price_calculation_meal_price_correct_l1122_112202

/-- Calculate the entire price of a meal given individual costs, tax rate, and tip rate -/
theorem meal_price_calculation (appetizer : ℚ) (buffy_entree : ℚ) (oz_entree : ℚ) 
  (side1 : ℚ) (side2 : ℚ) (dessert : ℚ) (drink_price : ℚ) 
  (tax_rate : ℚ) (tip_rate : ℚ) : ℚ :=
  let total_before_tax := appetizer + buffy_entree + oz_entree + side1 + side2 + dessert + 2 * drink_price
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  let tip := total_with_tax * tip_rate
  let total_price := total_with_tax + tip
  total_price

/-- The entire price of the meal is $120.66 -/
theorem meal_price_correct : 
  meal_price_calculation 9 20 25 6 8 11 (13/2) (3/40) (11/50) = 12066/100 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_meal_price_correct_l1122_112202


namespace NUMINAMATH_CALUDE_median_length_l1122_112259

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The length of the median to the longest side of the triangle -/
def median_to_longest_side (t : RightTriangle) : ℝ := 5

/-- Theorem: The length of the median to the longest side is 5 -/
theorem median_length (t : RightTriangle) : median_to_longest_side t = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_length_l1122_112259


namespace NUMINAMATH_CALUDE_exists_non_acute_triangle_with_two_acute_angles_l1122_112224

-- Define what an acute angle is
def is_acute_angle (angle : Real) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define what a right angle is
def is_right_angle (angle : Real) : Prop := angle = Real.pi / 2

-- Define a triangle structure
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_angles : angle1 + angle2 + angle3 = Real.pi

-- Define what an acute triangle is
def is_acute_triangle (t : Triangle) : Prop :=
  is_acute_angle t.angle1 ∧ is_acute_angle t.angle2 ∧ is_acute_angle t.angle3

-- Theorem statement
theorem exists_non_acute_triangle_with_two_acute_angles :
  ∃ (t : Triangle), (is_acute_angle t.angle1 ∧ is_acute_angle t.angle2) ∧ ¬is_acute_triangle t :=
sorry

end NUMINAMATH_CALUDE_exists_non_acute_triangle_with_two_acute_angles_l1122_112224


namespace NUMINAMATH_CALUDE_greatest_number_neither_swimming_nor_soccer_l1122_112200

theorem greatest_number_neither_swimming_nor_soccer 
  (total_students : ℕ) 
  (swimming_fans : ℕ) 
  (soccer_fans : ℕ) 
  (h1 : total_students = 1460) 
  (h2 : swimming_fans = 33) 
  (h3 : soccer_fans = 36) : 
  ∃ (neither_fans : ℕ), 
    neither_fans ≤ total_students - (swimming_fans + soccer_fans) ∧ 
    neither_fans = 1391 :=
sorry

end NUMINAMATH_CALUDE_greatest_number_neither_swimming_nor_soccer_l1122_112200


namespace NUMINAMATH_CALUDE_sphere_radii_problem_l1122_112231

theorem sphere_radii_problem (r₁ r₂ r₃ : ℝ) : 
  -- Three spheres touch each other externally
  2 * Real.sqrt (r₁ * r₂) = 2 ∧
  2 * Real.sqrt (r₁ * r₃) = Real.sqrt 3 ∧
  2 * Real.sqrt (r₂ * r₃) = 1 ∧
  -- The spheres touch a plane at the vertices of a right triangle
  -- One leg of the triangle has length 1
  -- The angle opposite to the leg of length 1 is 30°
  -- (These conditions are implicitly satisfied by the equations above)
  -- The radii are positive
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0
  →
  -- The radii of the spheres are √3, 1/√3, and √3/4
  (r₁ = Real.sqrt 3 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = Real.sqrt 3 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radii_problem_l1122_112231


namespace NUMINAMATH_CALUDE_cousin_calls_l1122_112248

/-- Represents the number of days in a leap year -/
def leapYearDays : ℕ := 366

/-- Represents the calling frequencies of the four cousins -/
def callingFrequencies : List ℕ := [2, 3, 4, 6]

/-- Calculates the number of days with at least one call in a leap year -/
def daysWithCalls (frequencies : List ℕ) (totalDays : ℕ) : ℕ :=
  sorry

theorem cousin_calls :
  daysWithCalls callingFrequencies leapYearDays = 244 :=
sorry

end NUMINAMATH_CALUDE_cousin_calls_l1122_112248


namespace NUMINAMATH_CALUDE_central_cell_value_l1122_112266

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are adjacent -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, adjacent (i, j) (k, l) → (g i j).val + 1 = (g k l).val ∨ (g k l).val + 1 = (g i j).val

/-- Returns the sum of corner cell values in the grid -/
def corner_sum (g : Grid) : ℕ :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- The main theorem to be proved -/
theorem central_cell_value (g : Grid) 
  (h_consec : consecutive_condition g) 
  (h_corner_sum : corner_sum g = 18) :
  (g 1 1).val = 2 :=
sorry

end NUMINAMATH_CALUDE_central_cell_value_l1122_112266


namespace NUMINAMATH_CALUDE_derek_age_is_20_l1122_112212

-- Define the ages as natural numbers
def aunt_beatrice_age : ℕ := 54

-- Define Emily's age in terms of Aunt Beatrice's age
def emily_age : ℕ := aunt_beatrice_age / 2

-- Define Derek's age in terms of Emily's age
def derek_age : ℕ := emily_age - 7

-- Theorem statement
theorem derek_age_is_20 : derek_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_derek_age_is_20_l1122_112212


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1122_112216

/-- Given a geometric sequence {a_n} with a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                     -- First term condition
  a 1 + a 3 + a 5 = 21 →        -- Sum of odd terms condition
  a 3 + a 5 + a 7 = 42 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1122_112216


namespace NUMINAMATH_CALUDE_f_properties_l1122_112267

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := x^3 - (3*(t+1)/2)*x^2 + 3*t*x + 1

theorem f_properties (t : ℝ) (h : t > 0) :
  (∃ (max : ℝ), t = 2 → ∀ x, f t x ≤ max ∧ ∃ y, f t y = max) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∃ x₀, 0 < x₀ ∧ x₀ < 2 ∧ ∀ x, 0 ≤ x ∧ x ≤ 2 → f t' x₀ ≤ f t' x)) ∧
  (∃ (a b : ℝ), a < b ∧ a > 0 ∧ b ≤ 1/3 ∧
    (∀ t', a < t' ∧ t' ≤ b →
      ∀ x, x ≥ 0 → f t' x ≤ x * Real.exp x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1122_112267


namespace NUMINAMATH_CALUDE_final_score_calculation_l1122_112278

theorem final_score_calculation (innovation_score comprehensive_score language_score : ℝ)
  (innovation_weight comprehensive_weight language_weight : ℝ) :
  innovation_score = 88 →
  comprehensive_score = 80 →
  language_score = 75 →
  innovation_weight = 5 →
  comprehensive_weight = 3 →
  language_weight = 2 →
  (innovation_score * innovation_weight + comprehensive_score * comprehensive_weight + language_score * language_weight) /
    (innovation_weight + comprehensive_weight + language_weight) = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_score_calculation_l1122_112278


namespace NUMINAMATH_CALUDE_x_equals_five_l1122_112245

theorem x_equals_five (x : ℝ) (h : x - 2 = 3) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_five_l1122_112245


namespace NUMINAMATH_CALUDE_roses_in_garden_l1122_112275

theorem roses_in_garden (total_pink : ℕ) (roses_per_row : ℕ) 
  (h1 : roses_per_row = 20)
  (h2 : total_pink = 40) : 
  (total_pink / (roses_per_row * (1 - 1/2) * (1 - 3/5))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_garden_l1122_112275


namespace NUMINAMATH_CALUDE_gym_membership_duration_is_three_years_l1122_112272

/-- Calculates the duration of a gym membership in years given the monthly cost,
    down payment, and total cost. -/
def gym_membership_duration (monthly_cost : ℚ) (down_payment : ℚ) (total_cost : ℚ) : ℚ :=
  ((total_cost - down_payment) / monthly_cost) / 12

/-- Proves that given the specific costs, the gym membership duration is 3 years. -/
theorem gym_membership_duration_is_three_years :
  gym_membership_duration 12 50 482 = 3 := by
  sorry

#eval gym_membership_duration 12 50 482

end NUMINAMATH_CALUDE_gym_membership_duration_is_three_years_l1122_112272


namespace NUMINAMATH_CALUDE_taxi_speed_is_60_l1122_112236

/-- The speed of the taxi in mph -/
def taxi_speed : ℝ := 60

/-- The speed of the bus in mph -/
def bus_speed : ℝ := taxi_speed - 30

/-- The time difference between the bus and taxi departure in hours -/
def time_difference : ℝ := 3

/-- The time it takes for the taxi to overtake the bus in hours -/
def overtake_time : ℝ := 3

theorem taxi_speed_is_60 :
  (taxi_speed * overtake_time = bus_speed * (time_difference + overtake_time)) →
  taxi_speed = 60 := by
  sorry

#check taxi_speed_is_60

end NUMINAMATH_CALUDE_taxi_speed_is_60_l1122_112236


namespace NUMINAMATH_CALUDE_initial_state_is_losing_l1122_112237

/-- Represents a game state with two piles of matches -/
structure GameState :=
  (pile1 : Nat) (pile2 : Nat)

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (move : Nat) (fromPile : Bool) : Prop :=
  if fromPile then
    move > 0 ∧ move ≤ state.pile1 ∧ state.pile2 % move = 0
  else
    move > 0 ∧ move ≤ state.pile2 ∧ state.pile1 % move = 0

/-- Defines a losing position in the game -/
def isLosingPosition (state : GameState) : Prop :=
  ∃ (k m n : Nat),
    state.pile1 = 2^k * (2*m + 1) ∧
    state.pile2 = 2^k * (2*n + 1)

/-- The main theorem stating that the initial position (100, 252) is a losing position -/
theorem initial_state_is_losing :
  isLosingPosition (GameState.mk 100 252) :=
sorry

#check initial_state_is_losing

end NUMINAMATH_CALUDE_initial_state_is_losing_l1122_112237


namespace NUMINAMATH_CALUDE_percentage_calculation_l1122_112258

theorem percentage_calculation (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1122_112258


namespace NUMINAMATH_CALUDE_expression_simplification_l1122_112281

theorem expression_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (2 * x) / (x + 1) - (2 * x + 4) / (x^2 - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 2 / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1122_112281


namespace NUMINAMATH_CALUDE_fraction_problem_l1122_112290

theorem fraction_problem (x : ℚ) : 
  x / (4 * x - 4) = 3 / 7 → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1122_112290


namespace NUMINAMATH_CALUDE_square_perimeter_l1122_112226

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 468 → perimeter = 24 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1122_112226


namespace NUMINAMATH_CALUDE_solution_set_for_a_neg_one_range_of_a_l1122_112225

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |3*x - 1|

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x ≤ |3*x + 1|}

-- Statement for part 1
theorem solution_set_for_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | 1/4 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Statement for part 2
theorem range_of_a (a : ℝ) :
  (Set.Icc (1/4 : ℝ) 1 ⊆ M a) → -7/3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_neg_one_range_of_a_l1122_112225


namespace NUMINAMATH_CALUDE_min_buttons_for_adjacency_l1122_112253

/-- Represents a color of a button -/
inductive Color
| A | B | C | D | E | F

/-- Represents a sequence of buttons -/
def ButtonSequence := List Color

/-- Checks if two colors are adjacent in a button sequence -/
def areColorsAdjacent (seq : ButtonSequence) (c1 c2 : Color) : Prop :=
  ∃ i, (seq.get? i = some c1 ∧ seq.get? (i+1) = some c2) ∨
       (seq.get? i = some c2 ∧ seq.get? (i+1) = some c1)

/-- Checks if a button sequence satisfies the adjacency condition for all color pairs -/
def satisfiesCondition (seq : ButtonSequence) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → areColorsAdjacent seq c1 c2

/-- The main theorem stating the minimum number of buttons required -/
theorem min_buttons_for_adjacency :
  ∃ (seq : ButtonSequence),
    seq.length = 18 ∧
    satisfiesCondition seq ∧
    ∀ (seq' : ButtonSequence), satisfiesCondition seq' → seq'.length ≥ 18 :=
sorry

end NUMINAMATH_CALUDE_min_buttons_for_adjacency_l1122_112253


namespace NUMINAMATH_CALUDE_oranges_from_third_tree_l1122_112254

/-- The number of oranges picked from the third tree -/
def oranges_third_tree (total : ℕ) (first : ℕ) (second : ℕ) : ℕ :=
  total - (first + second)

/-- Theorem stating that the number of oranges picked from the third tree is 120 -/
theorem oranges_from_third_tree :
  oranges_third_tree 260 80 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_third_tree_l1122_112254


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_3087_l1122_112263

theorem smallest_prime_factor_of_3087 : Nat.minFac 3087 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_3087_l1122_112263


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1122_112217

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1122_112217


namespace NUMINAMATH_CALUDE_inequality_proof_l1122_112291

theorem inequality_proof (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ b) 
  (h3 : a^4 - 2019*a = c) (h4 : b^4 - 2019*b = c) : 
  -Real.sqrt c < a * b ∧ a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1122_112291


namespace NUMINAMATH_CALUDE_tenth_pebble_count_l1122_112201

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 4 => pebble_sequence (n + 3) + (3 * (n + 4) - 2)

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pebble_count_l1122_112201


namespace NUMINAMATH_CALUDE_snack_eaters_left_eq_30_l1122_112279

/-- Represents the number of snack eaters who left after the second group of outsiders joined -/
def snack_eaters_left (initial_people : ℕ) (initial_snackers : ℕ) (first_outsiders : ℕ) (second_outsiders : ℕ) (final_snackers : ℕ) : ℕ :=
  let total_after_first := initial_snackers + first_outsiders
  let remaining_after_half_left := total_after_first / 2
  let total_after_second := remaining_after_half_left + second_outsiders
  let before_final_half_left := final_snackers * 2
  total_after_second - before_final_half_left

theorem snack_eaters_left_eq_30 :
  snack_eaters_left 200 100 20 10 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_snack_eaters_left_eq_30_l1122_112279


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_l1122_112234

-- 1. 2x^2 + 2x = 2x(x+1)
theorem factorization_1 (x : ℝ) : 2*x^2 + 2*x = 2*x*(x+1) := by sorry

-- 2. a^3 - a = a(a+1)(a-1)
theorem factorization_2 (a : ℝ) : a^3 - a = a*(a+1)*(a-1) := by sorry

-- 3. (x-y)^2 - 4(x-y) + 4 = (x-y-2)^2
theorem factorization_3 (x y : ℝ) : (x-y)^2 - 4*(x-y) + 4 = (x-y-2)^2 := by sorry

-- 4. x^2 + 2xy + y^2 - 9 = (x+y+3)(x+y-3)
theorem factorization_4 (x y : ℝ) : x^2 + 2*x*y + y^2 - 9 = (x+y+3)*(x+y-3) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_factorization_3_factorization_4_l1122_112234


namespace NUMINAMATH_CALUDE_positive_quadratic_expression_l1122_112211

theorem positive_quadratic_expression (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_quadratic_expression_l1122_112211


namespace NUMINAMATH_CALUDE_exists_valid_sequence_l1122_112296

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (s : List Nat) : Prop :=
  s.length > 10 ∧
  s.sum = 20 ∧
  3 ∉ s ∧
  ∀ i j, i ≤ j → j < s.length → (s.take (j + 1)).drop i ≠ [3]

/-- Theorem stating the existence of a valid sequence -/
theorem exists_valid_sequence : ∃ s : List Nat, ValidSequence s := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_sequence_l1122_112296


namespace NUMINAMATH_CALUDE_percentage_not_receiving_muffin_l1122_112286

theorem percentage_not_receiving_muffin (total_percentage : ℝ) (muffin_percentage : ℝ) 
  (h1 : total_percentage = 100) 
  (h2 : muffin_percentage = 38) : 
  total_percentage - muffin_percentage = 62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_receiving_muffin_l1122_112286


namespace NUMINAMATH_CALUDE_frog_hop_probability_l1122_112271

/-- Represents a position on the 4x4 grid -/
inductive Position
| Inner : Fin 2 → Fin 2 → Position
| Edge : Fin 4 → Fin 4 → Position

/-- Represents a possible hop direction -/
inductive Direction
| Up | Down | Left | Right

/-- The grid size -/
def gridSize : Nat := 4

/-- The maximum number of hops -/
def maxHops : Nat := 5

/-- Function to determine if a position is on the edge -/
def isEdge (p : Position) : Bool :=
  match p with
  | Position.Edge _ _ => true
  | _ => false

/-- Function to perform a single hop -/
def hop (p : Position) (d : Direction) : Position :=
  sorry

/-- Function to calculate the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : Rat :=
  sorry

/-- The starting position (second square in the second row) -/
def startPosition : Position := Position.Inner 1 1

/-- The main theorem to prove -/
theorem frog_hop_probability :
  probReachEdge startPosition maxHops = 94 / 256 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l1122_112271


namespace NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l1122_112262

theorem sum_of_19th_powers_zero (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_zero : a^3 + b^3 + c^3 = 0) : 
  a^19 + b^19 + c^19 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_19th_powers_zero_l1122_112262


namespace NUMINAMATH_CALUDE_junior_score_l1122_112282

theorem junior_score (total_students : ℕ) (junior_percent senior_percent : ℚ) 
  (overall_avg senior_avg junior_score : ℚ) : 
  junior_percent = 1/5 →
  senior_percent = 4/5 →
  junior_percent + senior_percent = 1 →
  overall_avg = 85 →
  senior_avg = 83 →
  (junior_percent * junior_score + senior_percent * senior_avg = overall_avg) →
  junior_score = 93 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l1122_112282


namespace NUMINAMATH_CALUDE_median_mode_difference_l1122_112214

def data : List ℕ := [12, 13, 14, 15, 15, 21, 21, 21, 32, 32, 38, 39, 40, 41, 42, 43, 53, 58, 59]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : |median data - mode data| = 11 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l1122_112214


namespace NUMINAMATH_CALUDE_complex_multiplication_l1122_112298

theorem complex_multiplication : (1 - 2*Complex.I) * (3 + 4*Complex.I) * (-1 + Complex.I) = -9 + 13*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1122_112298


namespace NUMINAMATH_CALUDE_vector_to_point_coordinates_l1122_112207

/-- Given a vector AB = (-2, 4), if point A is at the origin (0, 0), 
    then the coordinates of point B are (-2, 4). -/
theorem vector_to_point_coordinates (A B : ℝ × ℝ) : 
  (A.1 - B.1 = 2 ∧ A.2 - B.2 = -4) → 
  (A = (0, 0) → B = (-2, 4)) := by
  sorry

end NUMINAMATH_CALUDE_vector_to_point_coordinates_l1122_112207


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1122_112235

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalScore : Nat
  averageIncrease : Nat
  lastInningsScore : Nat

/-- Calculates the average score of a batsman -/
def calculateAverage (performance : BatsmanPerformance) : Rat :=
  performance.totalScore / performance.innings

theorem batsman_average_after_17th_innings
  (performance : BatsmanPerformance)
  (h1 : performance.innings = 17)
  (h2 : performance.lastInningsScore = 85)
  (h3 : calculateAverage performance - calculateAverage { performance with
    innings := performance.innings - 1
    totalScore := performance.totalScore - performance.lastInningsScore
  } = performance.averageIncrease)
  (h4 : performance.averageIncrease = 3) :
  calculateAverage performance = 37 := by
  sorry

#eval calculateAverage {
  innings := 17,
  totalScore := 17 * 37,
  averageIncrease := 3,
  lastInningsScore := 85
}

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1122_112235


namespace NUMINAMATH_CALUDE_equation_solution_l1122_112294

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4.5 ∧ x₂ = -3) ∧ 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2 ↔ (x = x₁ ∨ x = x₂))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1122_112294


namespace NUMINAMATH_CALUDE_trigonometric_shift_l1122_112273

/-- Proves that √3 * sin(2x) - cos(2x) is equivalent to 2 * sin(2(x + π/12)) --/
theorem trigonometric_shift (x : ℝ) : 
  Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 12)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_shift_l1122_112273


namespace NUMINAMATH_CALUDE_room_painting_problem_l1122_112285

/-- The total area of a room painted by two painters working together --/
def room_area (painter1_rate : ℝ) (painter2_rate : ℝ) (slowdown : ℝ) (time : ℝ) : ℝ :=
  time * (painter1_rate + painter2_rate - slowdown)

theorem room_painting_problem :
  let painter1_rate := 1 / 6
  let painter2_rate := 1 / 8
  let slowdown := 5
  let time := 4
  room_area painter1_rate painter2_rate slowdown time = 120 := by
sorry

end NUMINAMATH_CALUDE_room_painting_problem_l1122_112285


namespace NUMINAMATH_CALUDE_combustible_ice_volume_scientific_notation_l1122_112210

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem combustible_ice_volume_scientific_notation :
  toScientificNotation 19400000000 = ScientificNotation.mk 1.94 10 (by norm_num) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_combustible_ice_volume_scientific_notation_l1122_112210


namespace NUMINAMATH_CALUDE_expected_smallest_seven_from_sixtythree_l1122_112240

/-- The expected value of the smallest number when randomly selecting r numbers from a set of n numbers. -/
def expected_smallest (n : ℕ) (r : ℕ) : ℚ :=
  (n + 1 : ℚ) / (r + 1 : ℚ)

/-- The set size -/
def n : ℕ := 63

/-- The sample size -/
def r : ℕ := 7

theorem expected_smallest_seven_from_sixtythree :
  expected_smallest n r = 8 := by
  sorry

end NUMINAMATH_CALUDE_expected_smallest_seven_from_sixtythree_l1122_112240


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1122_112204

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 18) (h2 : a * b = 45) :
  min a b = 3 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1122_112204


namespace NUMINAMATH_CALUDE_min_value_of_product_l1122_112295

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l1122_112295


namespace NUMINAMATH_CALUDE_division_remainder_l1122_112222

theorem division_remainder : 
  let dividend : ℕ := 23
  let divisor : ℕ := 5
  let quotient : ℕ := 4
  dividend % divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1122_112222


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1122_112249

/-- 
Given a geometric sequence {b_n} where b_n > 0 for all n and the common ratio q > 1,
prove that b₄ + b₈ > b₅ + b₇.
-/
theorem geometric_sequence_inequality (b : ℕ → ℝ) (q : ℝ) 
  (h_positive : ∀ n, b n > 0)
  (h_geometric : ∀ n, b (n + 1) = q * b n)
  (h_q_gt_one : q > 1) :
  b 4 + b 8 > b 5 + b 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1122_112249


namespace NUMINAMATH_CALUDE_magical_card_stack_l1122_112241

/-- 
Given a stack of 2n cards numbered 1 to 2n, with the top n cards forming pile A 
and the rest forming pile B, prove that when restacked by alternating from 
piles B and A, the total number of cards where card 161 retains its original 
position is 482.
-/
theorem magical_card_stack (n : ℕ) : 
  (∃ (total : ℕ), 
    total = 2 * n ∧ 
    161 ≤ n ∧ 
    (∀ (k : ℕ), k ≤ total → k = 161 → (k - 1) / 2 = (n - 161))) → 
  2 * n = 482 :=
by sorry

end NUMINAMATH_CALUDE_magical_card_stack_l1122_112241


namespace NUMINAMATH_CALUDE_remainder_problem_l1122_112246

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1083 % d = r) (h3 : 1455 % d = r) (h4 : 2345 % d = r) : 
  d - r = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1122_112246


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1122_112239

theorem polynomial_factorization :
  (∀ x : ℝ, 2 * x^4 - 2 = 2 * (x^2 + 1) * (x + 1) * (x - 1)) ∧
  (∀ x : ℝ, x^4 - 18 * x^2 + 81 = (x + 3)^2 * (x - 3)^2) ∧
  (∀ y : ℝ, (y^2 - 1)^2 + 11 * (1 - y^2) + 24 = (y + 2) * (y - 2) * (y + 3) * (y - 3)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1122_112239


namespace NUMINAMATH_CALUDE_find_n_l1122_112209

theorem find_n : ∃ n : ℤ, (15 : ℝ) ^ (2 * n) = (1 / 15 : ℝ) ^ (3 * n - 30) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1122_112209


namespace NUMINAMATH_CALUDE_max_pairs_remaining_l1122_112260

/-- Represents the total number of shoe pairs -/
def total_pairs : ℕ := 27

/-- Represents the number of individual shoes lost -/
def shoes_lost : ℕ := 9

/-- Theorem stating the maximum number of complete pairs remaining after losing shoes -/
theorem max_pairs_remaining (total : ℕ) (lost : ℕ) : 
  total = total_pairs → lost = shoes_lost → total - lost ≤ 18 := by
  sorry

#check max_pairs_remaining

end NUMINAMATH_CALUDE_max_pairs_remaining_l1122_112260


namespace NUMINAMATH_CALUDE_find_number_l1122_112218

theorem find_number : ∃ x : ℚ, (4 * x) / 7 + 12 = 36 ∧ x = 42 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1122_112218


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1122_112287

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1122_112287


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l1122_112261

/-- A triangle with given altitudes has a specific area -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h_pos₁ : h₁ > 0) (h_pos₂ : h₂ > 0) (h_pos₃ : h₃ > 0) :
  h₁ = 12 → h₂ = 15 → h₃ = 20 → ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * 150) ∧ (b * h₂ = 2 * 150) ∧ (c * h₃ = 2 * 150) :=
by sorry

#check triangle_area_from_altitudes

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l1122_112261


namespace NUMINAMATH_CALUDE_select_five_from_eight_l1122_112220

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l1122_112220


namespace NUMINAMATH_CALUDE_mn_m_plus_n_is_even_l1122_112213

theorem mn_m_plus_n_is_even (m n : ℤ) : 2 ∣ (m * n * (m + n)) := by
  sorry

end NUMINAMATH_CALUDE_mn_m_plus_n_is_even_l1122_112213


namespace NUMINAMATH_CALUDE_ribbon_cost_comparison_l1122_112206

/-- Represents the cost and quantity of ribbons --/
structure RibbonPurchase where
  cost : ℕ
  quantity : ℕ

/-- Determines if one ribbon is cheaper than another --/
def isCheaper (r1 r2 : RibbonPurchase) : Prop :=
  r1.cost * r2.quantity < r2.cost * r1.quantity

theorem ribbon_cost_comparison 
  (yellow blue : RibbonPurchase)
  (h_yellow : yellow.cost = 24)
  (h_blue : blue.cost = 36) :
  (∃ y b, isCheaper {cost := 24, quantity := y} {cost := 36, quantity := b}) ∧
  (∃ y b, isCheaper {cost := 36, quantity := b} {cost := 24, quantity := y}) ∧
  (∃ y b, yellow.cost * b = blue.cost * y) :=
sorry

end NUMINAMATH_CALUDE_ribbon_cost_comparison_l1122_112206


namespace NUMINAMATH_CALUDE_certain_number_problem_l1122_112244

theorem certain_number_problem (a : ℕ) (certain_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * 45 * certain_number) :
  certain_number = 49 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1122_112244


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1122_112280

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {-2, -1, 0}

theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1122_112280


namespace NUMINAMATH_CALUDE_total_score_is_248_l1122_112289

/-- Calculates the total score across 4 subjects given 3 scores and the 4th as their average -/
def totalScoreAcross4Subjects (geography math english : ℕ) : ℕ :=
  let history := (geography + math + english) / 3
  geography + math + english + history

/-- Proves that given the specific scores, the total across 4 subjects is 248 -/
theorem total_score_is_248 :
  totalScoreAcross4Subjects 50 70 66 = 248 := by
  sorry

#eval totalScoreAcross4Subjects 50 70 66

end NUMINAMATH_CALUDE_total_score_is_248_l1122_112289


namespace NUMINAMATH_CALUDE_yimin_orchard_tree_count_l1122_112228

/-- The number of trees in Yimin Orchard -/
theorem yimin_orchard_tree_count : 
  let pear_rows : ℕ := 15
  let apple_rows : ℕ := 34
  let trees_per_row : ℕ := 21
  (pear_rows + apple_rows) * trees_per_row = 1029 := by
sorry

end NUMINAMATH_CALUDE_yimin_orchard_tree_count_l1122_112228


namespace NUMINAMATH_CALUDE_seven_points_triangle_l1122_112288

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The angle between three points --/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of seven points on a plane --/
def SevenPoints : Type := Fin 7 → Point

theorem seven_points_triangle (points : SevenPoints) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (angle (points i) (points j) (points k) > 2 * π / 3 ∨
     angle (points j) (points k) (points i) > 2 * π / 3 ∨
     angle (points k) (points i) (points j) > 2 * π / 3) :=
  sorry

end NUMINAMATH_CALUDE_seven_points_triangle_l1122_112288


namespace NUMINAMATH_CALUDE_probability_four_green_marbles_l1122_112243

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of draws -/
def num_draws : ℕ := 8

/-- The number of green marbles we want to draw -/
def target_green : ℕ := 4

/-- The probability of drawing exactly 'target_green' green marbles in 'num_draws' draws -/
def probability_exact_green : ℚ :=
  (Nat.choose num_draws target_green : ℚ) *
  (green_marbles ^ target_green * purple_marbles ^ (num_draws - target_green)) /
  (total_marbles ^ num_draws)

theorem probability_four_green_marbles :
  probability_exact_green = 1120 / 6561 :=
sorry

end NUMINAMATH_CALUDE_probability_four_green_marbles_l1122_112243


namespace NUMINAMATH_CALUDE_stone_width_is_five_dm_l1122_112229

/-- Proves that the width of stones used to pave a hall is 5 decimeters -/
theorem stone_width_is_five_dm (hall_length : ℝ) (hall_width : ℝ) 
  (stone_length : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  hall_width = 15 →
  stone_length = 0.4 →
  num_stones = 2700 →
  ∃ (stone_width : ℝ),
    stone_width = 0.5 ∧
    hall_length * hall_width * 100 = num_stones * stone_length * stone_width :=
by sorry

end NUMINAMATH_CALUDE_stone_width_is_five_dm_l1122_112229


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1122_112250

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l1122_112250


namespace NUMINAMATH_CALUDE_bills_remaining_money_bills_remaining_money_proof_l1122_112277

/-- Calculates the amount of money Bill is left with after selling fool's gold and paying a fine -/
theorem bills_remaining_money (ounces_sold : ℕ) (price_per_ounce : ℕ) (fine : ℕ) : ℕ :=
  let total_earned := ounces_sold * price_per_ounce
  total_earned - fine

/-- Proves that Bill is left with $22 given the specific conditions -/
theorem bills_remaining_money_proof :
  bills_remaining_money 8 9 50 = 22 := by
  sorry

end NUMINAMATH_CALUDE_bills_remaining_money_bills_remaining_money_proof_l1122_112277


namespace NUMINAMATH_CALUDE_formula_describes_relationship_l1122_112268

/-- The formula y = 80 - 10x describes the relationship between x and y for a given set of points -/
theorem formula_describes_relationship : ∀ (x y : ℝ), 
  ((x = 0 ∧ y = 80) ∨ 
   (x = 1 ∧ y = 70) ∨ 
   (x = 2 ∧ y = 60) ∨ 
   (x = 3 ∧ y = 50) ∨ 
   (x = 4 ∧ y = 40)) → 
  y = 80 - 10 * x := by
sorry

end NUMINAMATH_CALUDE_formula_describes_relationship_l1122_112268


namespace NUMINAMATH_CALUDE_queen_high_school_teachers_queen_high_school_teachers_correct_l1122_112223

theorem queen_high_school_teachers (num_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_classes := num_students * classes_per_student
  let unique_classes := total_classes / students_per_class
  unique_classes / classes_per_teacher

theorem queen_high_school_teachers_correct : 
  queen_high_school_teachers 1500 6 5 25 = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_high_school_teachers_queen_high_school_teachers_correct_l1122_112223


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1122_112265

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (max_val : ℝ), max_val = 15 ∧ ∀ (x' y' : ℝ), 2 * x'^2 - 6 * x' + y'^2 = 0 →
    x'^2 + y'^2 + 2 * x' ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1122_112265


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1122_112252

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) :
  Complex.abs (1 + z) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1122_112252


namespace NUMINAMATH_CALUDE_prime_factors_equation_l1122_112242

/-- Given an expression (4^x) * (7^5) * (11^2) with 29 prime factors, prove x = 11 -/
theorem prime_factors_equation (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end NUMINAMATH_CALUDE_prime_factors_equation_l1122_112242


namespace NUMINAMATH_CALUDE_geese_count_l1122_112215

/-- The number of ducks in the marsh -/
def num_ducks : ℝ := 37.0

/-- The difference between the number of geese and ducks -/
def geese_duck_difference : ℕ := 21

/-- The number of geese in the marsh -/
def num_geese : ℝ := num_ducks + geese_duck_difference

theorem geese_count : num_geese = 58 := by
  sorry

end NUMINAMATH_CALUDE_geese_count_l1122_112215


namespace NUMINAMATH_CALUDE_nabla_problem_l1122_112238

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + b^(2*a)

-- Theorem statement
theorem nabla_problem : nabla (nabla 2 1) 2 = 259 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1122_112238


namespace NUMINAMATH_CALUDE_all_statements_correct_l1122_112233

/-- The volume of a rectangle with sides a and b, considered as a 3D object of unit height -/
def volume (a b : ℝ) : ℝ := a * b

theorem all_statements_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (volume (2 * a) b = 2 * volume a b) ∧
  (volume a (3 * b) = 3 * volume a b) ∧
  (volume (2 * a) (3 * b) = 6 * volume a b) ∧
  (volume (a / 2) (2 * b) = volume a b) ∧
  (volume (3 * a) (b / 2) = (3 / 2) * volume a b) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l1122_112233


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1122_112293

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 150 → badminton = 75 → tennis = 85 → neither = 15 → 
  badminton + tennis - (total - neither) = 25 := by
sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1122_112293


namespace NUMINAMATH_CALUDE_pumpkin_pie_degrees_l1122_112274

/-- Represents the preference distribution of pies in a class --/
structure PiePreference where
  total : ℕ
  peach : ℕ
  apple : ℕ
  blueberry : ℕ
  pumpkin : ℕ
  banana : ℕ

/-- Calculates the degrees for a given pie in a pie chart --/
def degreesForPie (pref : PiePreference) (pieCount : ℕ) : ℚ :=
  (pieCount : ℚ) / (pref.total : ℚ) * 360

/-- Theorem stating the degrees for pumpkin pie in Jeremy's class --/
theorem pumpkin_pie_degrees (pref : PiePreference) 
  (h1 : pref.total = 40)
  (h2 : pref.peach = 14)
  (h3 : pref.apple = 9)
  (h4 : pref.blueberry = 7)
  (h5 : pref.pumpkin = pref.banana)
  (h6 : pref.pumpkin + pref.banana = pref.total - (pref.peach + pref.apple + pref.blueberry)) :
  degreesForPie pref pref.pumpkin = 45 := by
  sorry


end NUMINAMATH_CALUDE_pumpkin_pie_degrees_l1122_112274


namespace NUMINAMATH_CALUDE_intersection_point_of_perpendicular_lines_l1122_112208

/-- Given a line l: 2x + y = 10 and a point (-10, 0), this theorem proves that the 
    intersection point of l and the line l' passing through (-10, 0) and perpendicular 
    to l is (2, 6). -/
theorem intersection_point_of_perpendicular_lines 
  (l : Set (ℝ × ℝ)) 
  (h_l : l = {(x, y) | 2 * x + y = 10}) 
  (p : ℝ × ℝ) 
  (h_p : p = (-10, 0)) :
  ∃ (q : ℝ × ℝ), q ∈ l ∧ 
    (∃ (l' : Set (ℝ × ℝ)), p ∈ l' ∧ 
      (∀ (x y : ℝ), (x, y) ∈ l' ↔ (x - p.1) * 2 + (y - p.2) = 0) ∧
      q ∈ l' ∧
      q = (2, 6)) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_perpendicular_lines_l1122_112208


namespace NUMINAMATH_CALUDE_range_of_expression_l1122_112219

open Real

theorem range_of_expression (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 ≤ β ∧ β ≤ π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
sorry

end NUMINAMATH_CALUDE_range_of_expression_l1122_112219


namespace NUMINAMATH_CALUDE_function_q_polynomial_l1122_112227

/-- Given a function q(x) satisfying the equation
    q(x) + (2x^6 + 4x^4 + 5x^2 + 7) = (3x^4 + 18x^3 + 15x^2 + 8x + 3),
    prove that q(x) = -2x^6 - x^4 + 18x^3 + 10x^2 + 8x - 4 -/
theorem function_q_polynomial (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 5*x^2 + 7) = (3*x^4 + 18*x^3 + 15*x^2 + 8*x + 3)) →
  (∀ x, q x = -2*x^6 - x^4 + 18*x^3 + 10*x^2 + 8*x - 4) := by
  sorry

end NUMINAMATH_CALUDE_function_q_polynomial_l1122_112227


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1122_112284

/-- Given a hyperbola with equation x²/4 - y²/9 = 1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = 1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1122_112284


namespace NUMINAMATH_CALUDE_max_divisible_integers_l1122_112203

theorem max_divisible_integers (n : ℕ) : ℕ := by
  -- Let S be the set of 2n consecutive integers
  -- Let D be the set of divisors {n+1, n+2, ..., 2n}
  -- max_divisible is the maximum number of integers in S divisible by at least one number in D
  -- We want to prove that max_divisible = n + ⌊n/2⌋
  sorry

#check max_divisible_integers

end NUMINAMATH_CALUDE_max_divisible_integers_l1122_112203


namespace NUMINAMATH_CALUDE_vector_properties_l1122_112283

open Real

/-- Given vectors satisfying certain conditions, prove parallelism and angle between vectors -/
theorem vector_properties (a b c : ℝ × ℝ) : 
  (3 • a - 2 • b = (2, 6)) → 
  (a + 2 • b = (6, 2)) → 
  (c = (1, 1)) → 
  (∃ (k : ℝ), a = k • c) ∧ 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l1122_112283


namespace NUMINAMATH_CALUDE_second_row_starts_with_531_l1122_112299

-- Define the grid type
def Grid := Fin 3 → Fin 3 → Nat

-- Define the valid range of numbers
def ValidNumber (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 5

-- No repetition in rows
def NoRowRepetition (grid : Grid) : Prop :=
  ∀ i j k, j ≠ k → grid i j ≠ grid i k

-- No repetition in columns
def NoColumnRepetition (grid : Grid) : Prop :=
  ∀ i j k, i ≠ k → grid i j ≠ grid k j

-- Divisibility condition
def DivisibilityCondition (grid : Grid) : Prop :=
  ∀ i j, i > 0 → grid i j % grid (i-1) j = 0 ∧
  ∀ i j, j > 0 → grid i j % grid i (j-1) = 0

-- All numbers are valid
def AllValidNumbers (grid : Grid) : Prop :=
  ∀ i j, ValidNumber (grid i j)

-- Main theorem
theorem second_row_starts_with_531 (grid : Grid) 
  (h1 : NoRowRepetition grid)
  (h2 : NoColumnRepetition grid)
  (h3 : DivisibilityCondition grid)
  (h4 : AllValidNumbers grid) :
  grid 1 0 = 5 ∧ grid 1 1 = 1 ∧ grid 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_row_starts_with_531_l1122_112299


namespace NUMINAMATH_CALUDE_annie_cookies_l1122_112230

/-- The number of cookies Annie ate over three days -/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem: Annie ate 29 cookies over three days -/
theorem annie_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday = tuesday + (tuesday * 2 / 5) ∧
  total_cookies monday tuesday wednesday = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_l1122_112230


namespace NUMINAMATH_CALUDE_adam_final_score_l1122_112270

def trivia_game (first_half_correct : ℕ) (second_half_correct : ℕ) 
                (first_half_points : ℕ) (second_half_points : ℕ) 
                (bonus_points : ℕ) (penalty : ℕ) (total_questions : ℕ) : ℕ :=
  let correct_points := first_half_correct * first_half_points + second_half_correct * second_half_points
  let total_correct := first_half_correct + second_half_correct
  let bonus := (total_correct / 3) * bonus_points
  let incorrect := total_questions - total_correct
  let penalty_points := incorrect * penalty
  correct_points + bonus - penalty_points

theorem adam_final_score : 
  trivia_game 15 12 3 5 2 1 35 = 115 := by sorry

end NUMINAMATH_CALUDE_adam_final_score_l1122_112270


namespace NUMINAMATH_CALUDE_proportion_equality_l1122_112292

/-- Given a proportion x : 6 :: 2 : 0.19999999999999998, prove that x = 60 -/
theorem proportion_equality : 
  ∀ x : ℝ, (x / 6 = 2 / 0.19999999999999998) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1122_112292


namespace NUMINAMATH_CALUDE_brown_rabbit_hop_distance_l1122_112297

/-- Proves that given a white rabbit hopping 15 meters per minute and a total distance of 135 meters
    hopped by both rabbits in 5 minutes, the brown rabbit hops 12 meters per minute. -/
theorem brown_rabbit_hop_distance
  (white_rabbit_speed : ℝ)
  (total_distance : ℝ)
  (time : ℝ)
  (h1 : white_rabbit_speed = 15)
  (h2 : total_distance = 135)
  (h3 : time = 5) :
  (total_distance - white_rabbit_speed * time) / time = 12 := by
  sorry

#check brown_rabbit_hop_distance

end NUMINAMATH_CALUDE_brown_rabbit_hop_distance_l1122_112297


namespace NUMINAMATH_CALUDE_yellow_two_days_ago_white_tomorrow_dandelion_counts_l1122_112247

/-- Represents the state of dandelions in the meadow on a given day -/
structure DandelionState where
  yellow : ℕ
  white : ℕ

/-- The lifecycle of a dandelion -/
def dandelionLifecycle : ℕ := 3

/-- The state of dandelions yesterday -/
def yesterdayState : DandelionState := { yellow := 20, white := 14 }

/-- The state of dandelions today -/
def todayState : DandelionState := { yellow := 15, white := 11 }

/-- Theorem: The number of yellow dandelions the day before yesterday -/
theorem yellow_two_days_ago : ℕ := 25

/-- Theorem: The number of white dandelions tomorrow -/
theorem white_tomorrow : ℕ := 9

/-- Main theorem combining both results -/
theorem dandelion_counts : 
  (yellow_two_days_ago = yesterdayState.white + todayState.white) ∧
  (white_tomorrow = yesterdayState.yellow - todayState.white) := by
  sorry

end NUMINAMATH_CALUDE_yellow_two_days_ago_white_tomorrow_dandelion_counts_l1122_112247


namespace NUMINAMATH_CALUDE_evaluate_expression_l1122_112257

theorem evaluate_expression : 8^6 * 27^6 * 8^27 * 27^8 = 2^99 * 3^42 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1122_112257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1122_112251

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  (∃ (a : ℕ → ℝ), 
    (∀ n, a n = arithmetic_sequence a₁ d n) ∧
    ((Real.sin (a 3))^2 * (Real.cos (a 6))^2 - (Real.sin (a 6))^2 * (Real.cos (a 3))^2) / 
      Real.sin (a 4 + a 5) = 1 ∧
    d ∈ Set.Ioo (-1 : ℝ) 0 ∧
    (∀ n : ℕ, n ≠ 9 → 
      (n * a₁ + n * (n - 1) / 2 * d) ≤ (9 * a₁ + 9 * 8 / 2 * d))) →
  a₁ = 17 * Real.pi / 12 ∧ a₁ ∈ Set.Ioo (4 * Real.pi / 3) (3 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1122_112251


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1122_112276

/-- Given two parallel vectors p and q, prove that their sum has a magnitude of √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) :
  p = (2, -3) →
  q.1 = x ∧ q.2 = 6 →
  (∃ (k : ℝ), q = k • p) →
  ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l1122_112276


namespace NUMINAMATH_CALUDE_rebecca_eggs_l1122_112256

/-- The number of marbles Rebecca has -/
def marbles : ℕ := 6

/-- The difference between the number of eggs and marbles -/
def egg_marble_difference : ℕ := 14

/-- The number of eggs Rebecca has -/
def eggs : ℕ := marbles + egg_marble_difference

theorem rebecca_eggs : eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l1122_112256


namespace NUMINAMATH_CALUDE_households_B_and_C_eq_22_l1122_112205

/-- A residential building where each household subscribes to exactly two different newspapers. -/
structure Building where
  /-- The number of subscriptions for newspaper A -/
  subscriptions_A : ℕ
  /-- The number of subscriptions for newspaper B -/
  subscriptions_B : ℕ
  /-- The number of subscriptions for newspaper C -/
  subscriptions_C : ℕ
  /-- The total number of households in the building -/
  total_households : ℕ
  /-- Each household subscribes to exactly two different newspapers -/
  two_subscriptions : subscriptions_A + subscriptions_B + subscriptions_C = 2 * total_households

/-- The number of households subscribing to both newspaper B and C in a given building -/
def households_B_and_C (b : Building) : ℕ :=
  b.total_households - b.subscriptions_A

theorem households_B_and_C_eq_22 (b : Building) 
  (h_A : b.subscriptions_A = 30)
  (h_B : b.subscriptions_B = 34)
  (h_C : b.subscriptions_C = 40) :
  households_B_and_C b = 22 := by
  sorry

#eval households_B_and_C ⟨30, 34, 40, 52, by norm_num⟩

end NUMINAMATH_CALUDE_households_B_and_C_eq_22_l1122_112205


namespace NUMINAMATH_CALUDE_circles_symmetry_implies_sin_cos_theta_l1122_112269

/-- Circle C₁ -/
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + a*x = 0

/-- Circle C₂ -/
def C₂ (a θ : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + y*Real.tan θ = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- Main theorem -/
theorem circles_symmetry_implies_sin_cos_theta (a θ : ℝ) :
  (∀ x y, C₁ a x y ↔ C₁ a ((2*x-1)/2) (2*x-1-y)) →
  (∀ x y, C₂ a θ x y ↔ C₂ a θ ((2*x-1)/2) (2*x-1-y)) →
  Real.sin θ * Real.cos θ = -2/5 := by sorry

end NUMINAMATH_CALUDE_circles_symmetry_implies_sin_cos_theta_l1122_112269


namespace NUMINAMATH_CALUDE_cookie_distribution_l1122_112264

/-- The number of cookies Uncle Jude gave to Tim -/
def cookies_to_tim : ℕ := 15

/-- The total number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike : ℕ := 23

/-- The number of cookies Uncle Jude kept in the fridge -/
def cookies_in_fridge : ℕ := 188

theorem cookie_distribution :
  cookies_to_tim + cookies_to_mike + cookies_in_fridge + 2 * cookies_to_tim = total_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1122_112264


namespace NUMINAMATH_CALUDE_min_sum_factors_l1122_112221

def S (n : ℕ) : ℕ := (3 + 7 + 13 + (2*n + 2*n - 1))

theorem min_sum_factors (a b c : ℕ+) (h : S 10 = a * b * c) :
  ∃ (x y z : ℕ+), S 10 = x * y * z ∧ x + y + z ≤ a + b + c ∧ x + y + z = 68 :=
sorry

end NUMINAMATH_CALUDE_min_sum_factors_l1122_112221


namespace NUMINAMATH_CALUDE_star_example_l1122_112232

def star (x y : ℝ) : ℝ := 5 * x - 2 * y

theorem star_example : (star 3 4) + (star 2 2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l1122_112232


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l1122_112255

def systematic_sampling (population : ℕ) (sample_size : ℕ) (first_drawn : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ := 
  let interval := population / sample_size
  let sequence := fun n => first_drawn + (n - 1) * interval
  let n := (range_start - first_drawn + interval - 1) / interval
  sequence n

theorem systematic_sampling_result :
  systematic_sampling 960 32 9 401 430 = 429 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l1122_112255
