import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_expression_proof_l3397_339731

-- Define the condition
theorem algebraic_expression_proof (a b : ℝ) (h : a - b + 3 = Real.sqrt 2) :
  (2*a - 2*b + 6)^4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_proof_l3397_339731


namespace NUMINAMATH_CALUDE_fraction_simplification_l3397_339764

theorem fraction_simplification (a b c : ℝ) :
  ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c)) ∧
  ((a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3397_339764


namespace NUMINAMATH_CALUDE_complex_multiplication_l3397_339757

theorem complex_multiplication (i : ℂ) : i * i = -1 → -i * (1 - 2*i) = -2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3397_339757


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3397_339782

-- Define the circles and line
def circle_M (x y : ℝ) := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0
def circle_N (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 6 = 0
def line_l (x y : ℝ) := x + y - 9 = 0

-- Define the angle condition
def angle_BAC : ℝ := 45

-- Theorem statement
theorem circle_and_line_properties :
  ∃ (x y : ℝ),
    -- 1. Equation of circle through intersection of M and N, and origin
    (x^2 + y^2 - (50/11)*x - (50/11)*y = 0) ∧
    -- 2a. Equations of line AC when x-coordinate of A is 4
    ((5*x + y - 25 = 0) ∨ (x - 5*y + 21 = 0)) ∧
    -- 2b. Range of possible x-coordinates for point A
    (∀ (m : ℝ), (m ∈ Set.Icc 3 6) ↔ 
      (∃ (y : ℝ), line_l m y ∧ 
        ∃ (B C : ℝ × ℝ), 
          circle_M B.1 B.2 ∧ 
          circle_M C.1 C.2 ∧
          (angle_BAC = 45))) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3397_339782


namespace NUMINAMATH_CALUDE_volume_ratio_is_two_l3397_339708

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a closed cycle of an ideal gas -/
structure GasCycle where
  initial : GasState
  state2 : GasState
  state3 : GasState

/-- The conditions of the gas cycle -/
class CycleConditions (cycle : GasCycle) where
  isobaric_1_2 : cycle.state2.pressure = cycle.initial.pressure
  volume_increase_1_2 : cycle.state2.volume = 4 * cycle.initial.volume
  isothermal_2_3 : cycle.state3.temperature = cycle.state2.temperature
  pressure_increase_2_3 : cycle.state3.pressure > cycle.state2.pressure
  compression_3_1 : ∃ (γ : ℝ), cycle.initial.temperature = γ * cycle.initial.volume^2

/-- The theorem to be proved -/
theorem volume_ratio_is_two 
  (cycle : GasCycle) 
  [conditions : CycleConditions cycle] : 
  cycle.state3.volume = 2 * cycle.initial.volume := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_two_l3397_339708


namespace NUMINAMATH_CALUDE_inequality_reversal_l3397_339748

theorem inequality_reversal (x y : ℝ) (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l3397_339748


namespace NUMINAMATH_CALUDE_tom_payment_l3397_339733

/-- The total amount Tom paid for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 965 for his purchase -/
theorem tom_payment : total_amount 8 70 9 45 = 965 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l3397_339733


namespace NUMINAMATH_CALUDE_correct_relative_pronouns_l3397_339751

/-- A type representing relative pronouns -/
inductive RelativePronoun
  | What
  | Where
  | That
  | Which

/-- A function that checks if a relative pronoun introduces a defining clause without an antecedent -/
def introduces_defining_clause_without_antecedent (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.What => True
  | _ => False

/-- A function that checks if a relative pronoun introduces a clause describing a location or circumstance -/
def introduces_location_clause (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.Where => True
  | _ => False

theorem correct_relative_pronouns :
  ∃ (rp1 rp2 : RelativePronoun),
    introduces_defining_clause_without_antecedent rp1 ∧
    introduces_location_clause rp2 ∧
    rp1 = RelativePronoun.What ∧
    rp2 = RelativePronoun.Where :=
by
  sorry

end NUMINAMATH_CALUDE_correct_relative_pronouns_l3397_339751


namespace NUMINAMATH_CALUDE_probability_two_black_cards_l3397_339799

theorem probability_two_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) :
  (black_cards * (black_cards - 1)) / (total_cards * (total_cards - 1)) = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_cards_l3397_339799


namespace NUMINAMATH_CALUDE_nested_sqrt_fifteen_l3397_339713

theorem nested_sqrt_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fifteen_l3397_339713


namespace NUMINAMATH_CALUDE_total_students_l3397_339775

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) : 
  boys + girls = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l3397_339775


namespace NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3397_339707

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- One angle of the trapezoid is 60 degrees -/
  has_60_degree_angle : ∃ θ, θ = 60
  /-- A circle can be inscribed in the trapezoid -/
  has_inscribed_circle : Bool
  /-- A circle can be circumscribed around the trapezoid -/
  has_circumscribed_circle : Bool

/-- The ratio of bases in a special trapezoid -/
def base_ratio (t : SpecialTrapezoid) : ℚ × ℚ :=
  (1, 3)

/-- Theorem stating the ratio of bases in a special trapezoid -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  base_ratio t = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l3397_339707


namespace NUMINAMATH_CALUDE_power_zero_equals_one_l3397_339744

theorem power_zero_equals_one (x : ℝ) : x ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equals_one_l3397_339744


namespace NUMINAMATH_CALUDE_coefficients_of_specific_quadratic_l3397_339761

/-- Given a quadratic equation ax^2 + bx + c = 0, this function returns the tuple (a, b, c) of its coefficients -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

/-- The coefficients of the quadratic equation x^2 - x + 3 = 0 are (1, -1, 3) -/
theorem coefficients_of_specific_quadratic :
  quadratic_coefficients 1 (-1) 3 = (1, -1, 3) := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_specific_quadratic_l3397_339761


namespace NUMINAMATH_CALUDE_square_sum_value_l3397_339777

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -9) :
  x^2 + y^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3397_339777


namespace NUMINAMATH_CALUDE_solution_count_l3397_339785

/-- The number of distinct solutions to the system of equations:
    x = x^2 + y^2
    y = 3x^2y - y^3 -/
theorem solution_count : 
  (Set.ncard {p : ℝ × ℝ | let (x, y) := p; x = x^2 + y^2 ∧ y = 3*x^2*y - y^3} : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l3397_339785


namespace NUMINAMATH_CALUDE_yarn_crochet_length_l3397_339730

def yarn_problem (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ) : Prop :=
  total_length = 10 ∧ 
  num_parts = 5 ∧ 
  parts_used = 3 ∧ 
  (total_length / num_parts) * parts_used = 6

theorem yarn_crochet_length : 
  ∀ (total_length : ℝ) (num_parts : ℕ) (parts_used : ℕ),
  yarn_problem total_length num_parts parts_used :=
by
  sorry

end NUMINAMATH_CALUDE_yarn_crochet_length_l3397_339730


namespace NUMINAMATH_CALUDE_shape_area_theorem_l3397_339742

/-- Represents a shape in a grid --/
structure GridShape where
  wholeSquares : ℕ
  halfSquares : ℕ

/-- Calculates the area of a GridShape --/
def calculateArea (shape : GridShape) : ℚ :=
  shape.wholeSquares + shape.halfSquares / 2

theorem shape_area_theorem (shape : GridShape) :
  shape.wholeSquares = 5 → shape.halfSquares = 6 → calculateArea shape = 8 := by
  sorry

end NUMINAMATH_CALUDE_shape_area_theorem_l3397_339742


namespace NUMINAMATH_CALUDE_abc_product_l3397_339781

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) :
  a * b * c = 762 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l3397_339781


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l3397_339702

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal 1 - Complex.I) * (Complex.ofReal a + Complex.I) ∈ {z : ℂ | z.re = 0 ∧ z.im ≠ 0} → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l3397_339702


namespace NUMINAMATH_CALUDE_money_spent_on_blades_l3397_339753

def total_earned : ℕ := 42
def game_price : ℕ := 8
def num_games : ℕ := 4

theorem money_spent_on_blades : 
  total_earned - (game_price * num_games) = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_spent_on_blades_l3397_339753


namespace NUMINAMATH_CALUDE_expected_bullets_remaining_l3397_339774

/-- The probability of hitting the target -/
def hit_probability : ℝ := 0.6

/-- The total number of bullets -/
def total_bullets : ℕ := 4

/-- The expected number of bullets remaining after stopping the shooting -/
def expected_remaining_bullets : ℝ := 2.376

/-- Theorem stating that the expected number of bullets remaining is 2.376 -/
theorem expected_bullets_remaining :
  let p := hit_probability
  let n := total_bullets
  let E := expected_remaining_bullets
  E = (0 * (1 - p)^3 + 1 * p * (1 - p)^2 + 2 * p * (1 - p) + 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_expected_bullets_remaining_l3397_339774


namespace NUMINAMATH_CALUDE_min_button_presses_to_escape_l3397_339773

/-- Represents the state of the room with doors and mines -/
structure RoomState where
  armed_mines : ℕ
  closed_doors : ℕ

/-- Represents the actions of pressing buttons -/
structure ButtonPresses where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the final state of the room after pressing buttons -/
def final_state (initial : RoomState) (presses : ButtonPresses) : RoomState :=
  { armed_mines := initial.armed_mines + presses.red - 2 * presses.yellow,
    closed_doors := initial.closed_doors + presses.yellow - 2 * presses.green }

/-- Checks if all mines are disarmed and all doors are open -/
def is_solved (state : RoomState) : Prop :=
  state.armed_mines = 0 ∧ state.closed_doors = 0

/-- The main theorem to prove -/
theorem min_button_presses_to_escape : 
  ∃ (presses : ButtonPresses),
    is_solved (final_state { armed_mines := 3, closed_doors := 3 } presses) ∧
    presses.red + presses.yellow + presses.green = 9 ∧
    ∀ (other_presses : ButtonPresses),
      is_solved (final_state { armed_mines := 3, closed_doors := 3 } other_presses) →
      other_presses.red + other_presses.yellow + other_presses.green ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_button_presses_to_escape_l3397_339773


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l3397_339701

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions in the problem,
    the man's speed with the current is 20 kmph. -/
theorem mans_speed_with_current :
  let speed_against_current := 18
  let current_speed := 1
  speed_with_current speed_against_current current_speed = 20 := by
  sorry

#eval speed_with_current 18 1

end NUMINAMATH_CALUDE_mans_speed_with_current_l3397_339701


namespace NUMINAMATH_CALUDE_marian_cookies_l3397_339736

theorem marian_cookies (cookies_per_tray : ℕ) (num_trays : ℕ) (h1 : cookies_per_tray = 12) (h2 : num_trays = 23) :
  cookies_per_tray * num_trays = 276 := by
  sorry

end NUMINAMATH_CALUDE_marian_cookies_l3397_339736


namespace NUMINAMATH_CALUDE_square_root_expression_values_l3397_339756

theorem square_root_expression_values :
  ∀ (x y z : ℝ),
  (x^2 = 25) →
  (y = 4) →
  (z^2 = 9) →
  (2*x + y - 5*z = -1) ∨ (2*x + y - 5*z = 29) :=
by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_values_l3397_339756


namespace NUMINAMATH_CALUDE_max_discount_rate_l3397_339795

/-- Represents the maximum discount rate problem -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (original_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4) 
  (h2 : original_price = 5) 
  (h3 : min_profit_margin = 0.1) : 
  ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount → 
      original_price * (1 - discount / 100) - cost_price ≥ min_profit_margin * cost_price :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l3397_339795


namespace NUMINAMATH_CALUDE_bricks_per_course_l3397_339755

/-- Proves that the number of bricks in each course is 400 --/
theorem bricks_per_course (initial_courses : ℕ) (added_courses : ℕ) (total_bricks : ℕ) :
  initial_courses = 3 →
  added_courses = 2 →
  total_bricks = 1800 →
  ∃ (bricks_per_course : ℕ),
    bricks_per_course * (initial_courses + added_courses) - bricks_per_course / 2 = total_bricks ∧
    bricks_per_course = 400 := by
  sorry

end NUMINAMATH_CALUDE_bricks_per_course_l3397_339755


namespace NUMINAMATH_CALUDE_factory_output_increase_l3397_339768

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * 1.30 * (1 - 30.07 / 100) = 1 → P = 10 := by
sorry

end NUMINAMATH_CALUDE_factory_output_increase_l3397_339768


namespace NUMINAMATH_CALUDE_trip_time_difference_l3397_339727

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 160)
  (h2 : distance2 = 280)
  (h3 : speed = 40)
  : distance2 / speed - distance1 / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l3397_339727


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l3397_339788

theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) (h_q : q ≠ 1) :
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧
    ∀ u v : ℕ, u ≠ 0 → v ≠ 0 → a u * a v = (a 5)^2 →
      4/s + 1/(4*t) ≤ 4/u + 1/(4*v)) →
  (∃ s t : ℕ, s ≠ 0 ∧ t ≠ 0 ∧ a s * a t = (a 5)^2 ∧ 4/s + 1/(4*t) = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l3397_339788


namespace NUMINAMATH_CALUDE_chord_length_l3397_339739

/-- In a circle with radius 15 units, a chord that is a perpendicular bisector of the radius has a length of 26√3 units. -/
theorem chord_length (r : ℝ) (c : ℝ) : 
  r = 15 → -- The radius is 15 units
  c^2 = 4 * (r^2 - (r/2)^2) → -- The chord is a perpendicular bisector of the radius
  c = 26 * Real.sqrt 3 := by -- The length of the chord is 26√3 units
sorry

end NUMINAMATH_CALUDE_chord_length_l3397_339739


namespace NUMINAMATH_CALUDE_distance_and_speed_l3397_339779

-- Define the variables
def distance : ℝ := sorry
def speed_second_car : ℝ := sorry
def speed_first_car : ℝ := sorry
def speed_third_car : ℝ := sorry

-- Define the relationships between the speeds
axiom speed_diff_first_second : speed_first_car = speed_second_car + 4
axiom speed_diff_second_third : speed_second_car = speed_third_car + 6

-- Define the time differences
axiom time_diff_first_second : distance / speed_first_car = distance / speed_second_car - 3 / 60
axiom time_diff_second_third : distance / speed_second_car = distance / speed_third_car - 5 / 60

-- Theorem to prove
theorem distance_and_speed : distance = 120 ∧ speed_second_car = 96 := by
  sorry

end NUMINAMATH_CALUDE_distance_and_speed_l3397_339779


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l3397_339780

theorem largest_integer_inequality :
  ∃ (n : ℕ), (∀ (m : ℕ), (1/4 : ℚ) + (m : ℚ)/8 < 7/8 → m ≤ n) ∧
             ((1/4 : ℚ) + (n : ℚ)/8 < 7/8) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l3397_339780


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l3397_339723

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

-- Theorem 1: f(x) ≥ 0 if and only if a = 1
theorem f_nonnegative_iff_a_eq_one :
  (∀ x, f a x ≥ 0) ↔ a = 1 :=
sorry

-- Theorem 2: For a ≥ 1, f(x) > x ln x - sin x for all x > 0
theorem f_greater_than_x_ln_x_minus_sin_x
  (a : ℝ) (h : a ≥ 1) :
  ∀ x > 0, f a x > x * Real.log x - Real.sin x :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l3397_339723


namespace NUMINAMATH_CALUDE_inequality_of_means_l3397_339758

theorem inequality_of_means (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > a * b ∧ a * b > 2 * a^2 * b^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_means_l3397_339758


namespace NUMINAMATH_CALUDE_factorization_equality_l3397_339762

theorem factorization_equality (a b : ℝ) : 
  a^2 - 4*b^2 - 2*a + 4*b = (a + 2*b - 2) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3397_339762


namespace NUMINAMATH_CALUDE_smallest_total_hits_is_twelve_l3397_339752

/-- Represents a baseball player's batting statistics -/
structure BattingStats where
  initialHits : ℕ
  initialAtBats : ℕ
  newHits : ℕ
  newAtBats : ℕ
  initialAverage : ℚ
  newAverage : ℚ

/-- Calculates the smallest number of total hits given initial and new batting averages -/
def smallestTotalHits (stats : BattingStats) : ℕ :=
  stats.initialHits + stats.newHits

/-- Theorem: The smallest number of total hits is 12 given the specified conditions -/
theorem smallest_total_hits_is_twelve :
  ∃ (stats : BattingStats),
    stats.initialAverage = 360 / 1000 ∧
    stats.newAverage = 400 / 1000 ∧
    stats.newAtBats = stats.initialAtBats + 5 ∧
    smallestTotalHits stats = 12 ∧
    ∀ (otherStats : BattingStats),
      otherStats.initialAverage = 360 / 1000 ∧
      otherStats.newAverage = 400 / 1000 ∧
      otherStats.newAtBats = otherStats.initialAtBats + 5 →
      smallestTotalHits otherStats ≥ 12 :=
by sorry


end NUMINAMATH_CALUDE_smallest_total_hits_is_twelve_l3397_339752


namespace NUMINAMATH_CALUDE_equation_equivalence_l3397_339794

theorem equation_equivalence (a b c : ℝ) (h : b > 0) :
  (a / Real.sqrt (18 * b)) * (c / Real.sqrt (72 * b)) = 1 →
  a * c = 36 * b :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3397_339794


namespace NUMINAMATH_CALUDE_integer_representation_l3397_339735

theorem integer_representation (n : ℤ) : 
  ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 + d^2 ∨ n = a^2 + b^2 + c^2 - d^2 ∨
                    n = a^2 + b^2 - c^2 - d^2 ∨ n = a^2 - b^2 - c^2 - d^2 :=
sorry

example : ∃ (a b c : ℤ), 1947 = a^2 - b^2 - c^2 :=
sorry

end NUMINAMATH_CALUDE_integer_representation_l3397_339735


namespace NUMINAMATH_CALUDE_percentage_sum_l3397_339784

theorem percentage_sum : (0.15 * 25) + (0.12 * 45) = 9.15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l3397_339784


namespace NUMINAMATH_CALUDE_triangle_vector_representation_l3397_339703

/-- Given a triangle ABC and a point P on line AB, prove that CP can be represented
    in terms of CA and CB under certain conditions. -/
theorem triangle_vector_representation (A B C P : EuclideanSpace ℝ (Fin 3))
    (a b : EuclideanSpace ℝ (Fin 3)) : 
    (C - A = a) →  -- CA = a
    (C - B = b) →  -- CB = b
    (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
    (A - P = 2 • (P - B)) →  -- AP = 2PB
    (C - P = (1/3) • a + (2/3) • b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_representation_l3397_339703


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_950_degrees_l3397_339765

theorem angle_with_same_terminal_side_as_negative_950_degrees :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ ∃ k : ℤ, θ = -950 + 360 * k ∧ θ = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_950_degrees_l3397_339765


namespace NUMINAMATH_CALUDE_initial_guppies_count_l3397_339749

/-- Represents the fish tank scenario --/
structure FishTank where
  initialGuppies : ℕ
  initialAngelfish : ℕ
  initialTigerSharks : ℕ
  initialOscarFish : ℕ
  soldGuppies : ℕ
  soldAngelfish : ℕ
  soldTigerSharks : ℕ
  soldOscarFish : ℕ
  remainingFish : ℕ

/-- Theorem stating the initial number of guppies in Danny's fish tank --/
theorem initial_guppies_count (tank : FishTank)
    (h1 : tank.initialAngelfish = 76)
    (h2 : tank.initialTigerSharks = 89)
    (h3 : tank.initialOscarFish = 58)
    (h4 : tank.soldGuppies = 30)
    (h5 : tank.soldAngelfish = 48)
    (h6 : tank.soldTigerSharks = 17)
    (h7 : tank.soldOscarFish = 24)
    (h8 : tank.remainingFish = 198)
    (h9 : tank.remainingFish = 
      (tank.initialGuppies - tank.soldGuppies) +
      (tank.initialAngelfish - tank.soldAngelfish) +
      (tank.initialTigerSharks - tank.soldTigerSharks) +
      (tank.initialOscarFish - tank.soldOscarFish)) :
    tank.initialGuppies = 94 := by
  sorry

end NUMINAMATH_CALUDE_initial_guppies_count_l3397_339749


namespace NUMINAMATH_CALUDE_remainder_101_power_50_mod_100_l3397_339754

theorem remainder_101_power_50_mod_100 : 101^50 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_power_50_mod_100_l3397_339754


namespace NUMINAMATH_CALUDE_smallest_circle_area_l3397_339790

/-- The smallest area of a circle passing through two given points -/
theorem smallest_circle_area (x₁ y₁ x₂ y₂ : ℝ) : 
  let d := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  let r := d / 2
  let A := π * r^2
  x₁ = -3 ∧ y₁ = -2 ∧ x₂ = 2 ∧ y₂ = 4 →
  A = (61 * π) / 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_circle_area_l3397_339790


namespace NUMINAMATH_CALUDE_correct_calculation_l3397_339724

theorem correct_calculation (a : ℝ) : 2 * a * (1 - a) = 2 * a - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3397_339724


namespace NUMINAMATH_CALUDE_product_and_sum_of_integers_l3397_339750

theorem product_and_sum_of_integers : ∃ (n m : ℕ), 
  m = n + 2 ∧ 
  n * m = 2720 ∧ 
  n > 0 ∧ 
  n + m = 104 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_integers_l3397_339750


namespace NUMINAMATH_CALUDE_distance_from_origin_l3397_339722

/-- Given a point (x,y) satisfying certain conditions, prove that its distance from the origin is √(286 + 2√221) -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 8) (h2 : x > 1) 
  (h3 : Real.sqrt ((x - 1)^2 + 2^2) = 15) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (286 + 2 * Real.sqrt 221) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3397_339722


namespace NUMINAMATH_CALUDE_larger_denomination_proof_l3397_339759

theorem larger_denomination_proof (total_bills : ℕ) (total_value : ℕ) 
  (ten_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 30 →
  total_value = 330 →
  ten_bills = 27 →
  larger_bills = 3 →
  ten_bills + larger_bills = total_bills →
  10 * ten_bills + larger_bills * (total_value - 10 * ten_bills) / larger_bills = total_value →
  (total_value - 10 * ten_bills) / larger_bills = 20 := by
  sorry

end NUMINAMATH_CALUDE_larger_denomination_proof_l3397_339759


namespace NUMINAMATH_CALUDE_special_hyperbola_eccentricity_l3397_339706

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A line segment PQ -/
structure LineSegment where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: Eccentricity of a special hyperbola -/
theorem special_hyperbola_eccentricity (h : Hyperbola) (pq : LineSegment) :
  (∃ (r : ℝ), pq.P.2 = r ∧ pq.Q.2 = r) →  -- PQ is perpendicular to the real axis
  (pq.P.1 = h.F₂.1 ∧ pq.Q.1 = h.F₂.1) →  -- PQ passes through F₂
  (∃ (t : ℝ), (pq.P.1 - h.F₁.1)^2 + (pq.P.2 - h.F₁.2)^2 = t^2 ∧
              (pq.Q.1 - h.F₁.1)^2 + (pq.Q.2 - h.F₁.2)^2 = t^2) →  -- P and Q are on the hyperbola
  ((pq.P.1 - h.F₁.1) * (pq.Q.1 - h.F₁.1) + (pq.P.2 - h.F₁.2) * (pq.Q.2 - h.F₁.2) = 0) →  -- ∠PF₁Q = π/2
  eccentricity h = Real.sqrt 2 + 1 := by
    sorry

end NUMINAMATH_CALUDE_special_hyperbola_eccentricity_l3397_339706


namespace NUMINAMATH_CALUDE_inequality_proof_l3397_339738

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
  1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) ≤ 2 ∧
  (1 / (b * c + c * d + d * a - 1) + 1 / (a * b + c * d + d * a - 1) + 
   1 / (a * b + b * c + d * a - 1) + 1 / (a * b + b * c + c * d - 1) = 2 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3397_339738


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3397_339796

/-- 
For a polynomial of the form x^2 - 18x + k to be a perfect square binomial,
k must equal 81.
-/
theorem perfect_square_condition (k : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 18*x + k = (x + a)^2 + b) ↔ k = 81 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3397_339796


namespace NUMINAMATH_CALUDE_parallelogram_area_l3397_339714

/-- The area of a parallelogram with a diagonal of length 30 meters and an altitude of 20 meters to that diagonal is 600 square meters. -/
theorem parallelogram_area (d : ℝ) (h : ℝ) (h1 : d = 30) (h2 : h = 20) :
  d * h = 600 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3397_339714


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3397_339743

-- Define the solution set of ax^2 - 5x + b > 0
def solution_set_1 : Set ℝ := {x | -3 < x ∧ x < -2}

-- Define the quadratic expression ax^2 - 5x + b
def quadratic_1 (a b x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the quadratic expression bx^2 - 5x + a
def quadratic_2 (a b x : ℝ) : ℝ := b * x^2 - 5 * x + a

-- Define the solution set of bx^2 - 5x + a < 0
def solution_set_2 : Set ℝ := {x | x < -1/2 ∨ x > -1/3}

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) :
  (∀ x, x ∈ solution_set_1 ↔ quadratic_1 a b x > 0) →
  (∀ x, x ∈ solution_set_2 ↔ quadratic_2 a b x < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3397_339743


namespace NUMINAMATH_CALUDE_bug_return_probability_l3397_339772

-- Define the tetrahedron structure
structure Tetrahedron where
  vertices : Fin 4 → Point
  edge_length : ℝ
  is_regular : Bool

-- Define the bug's movement
def bug_move (t : Tetrahedron) (current_vertex : Fin 4) : Fin 4 := sorry

-- Define the probability of returning to the starting vertex after n steps
def return_probability (t : Tetrahedron) (n : ℕ) : ℚ := sorry

-- Main theorem
theorem bug_return_probability (t : Tetrahedron) :
  t.is_regular = true →
  t.edge_length = 1 →
  return_probability t 9 = 4920 / 19683 := by sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3397_339772


namespace NUMINAMATH_CALUDE_friends_drawing_cards_l3397_339778

theorem friends_drawing_cards (n : ℕ) (h : n = 3) :
  let total_outcomes := n.factorial
  let favorable_outcomes := (n - 1).factorial
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_friends_drawing_cards_l3397_339778


namespace NUMINAMATH_CALUDE_circular_bed_circumference_circular_bed_specific_circumference_l3397_339745

/-- The circumference of a circular bed containing a given number of plants -/
theorem circular_bed_circumference (num_plants : Real) (area_per_plant : Real) : Real :=
  let total_area := num_plants * area_per_plant
  let radius := (total_area / Real.pi).sqrt
  2 * Real.pi * radius

/-- Proof that the circular bed with given specifications has the expected circumference -/
theorem circular_bed_specific_circumference : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.000001 ∧ 
  |circular_bed_circumference 22.997889276778874 4 - 34.007194| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_bed_circumference_circular_bed_specific_circumference_l3397_339745


namespace NUMINAMATH_CALUDE_girls_equal_barefoot_children_l3397_339798

/-- Given a lawn with boys and girls, some of whom are barefoot and some wearing shoes,
    prove that the number of girls equals the number of barefoot children
    when the number of barefoot boys equals the number of girls with shoes. -/
theorem girls_equal_barefoot_children
  (num_barefoot_boys : ℕ)
  (num_girls_with_shoes : ℕ)
  (num_barefoot_girls : ℕ)
  (h : num_barefoot_boys = num_girls_with_shoes) :
  num_girls_with_shoes + num_barefoot_girls = num_barefoot_boys + num_barefoot_girls :=
by sorry

end NUMINAMATH_CALUDE_girls_equal_barefoot_children_l3397_339798


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l3397_339791

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 1800) 
  (h2 : males_below_50 = 756) : 
  (males_below_50 : ℝ) / (0.7 * total_employees) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l3397_339791


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3397_339769

-- Define the polynomial (x+a)^2 * (x-1)^3
def p (x a : ℝ) : ℝ := (x + a)^2 * (x - 1)^3

-- Define the coefficient of x^4 in the expansion of p(x,a)
def coeff_x4 (a : ℝ) : ℝ := -3 + 2*a

-- Theorem statement
theorem expansion_coefficient (a : ℝ) : coeff_x4 a = 1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3397_339769


namespace NUMINAMATH_CALUDE_remainder_of_division_l3397_339783

theorem remainder_of_division (n : ℕ) : 
  (3^302 + 302) % (3^151 + 3^101 + 1) = 302 := by
  sorry

#check remainder_of_division

end NUMINAMATH_CALUDE_remainder_of_division_l3397_339783


namespace NUMINAMATH_CALUDE_sally_quarters_problem_l3397_339760

theorem sally_quarters_problem (initial_quarters : ℕ) 
  (first_purchase : ℕ) (second_purchase : ℕ) :
  initial_quarters = 760 →
  first_purchase = 418 →
  second_purchase = 215 →
  initial_quarters - first_purchase - second_purchase = 127 :=
by sorry

end NUMINAMATH_CALUDE_sally_quarters_problem_l3397_339760


namespace NUMINAMATH_CALUDE_function_properties_l3397_339741

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f (1/2) = 0) 
  (h3 : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (1/2 + x) = -f (1/2 - x)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3397_339741


namespace NUMINAMATH_CALUDE_N_is_k_times_sum_of_digits_l3397_339734

/-- A number consisting of k nines -/
def N (k : ℕ) : ℕ := 10^k - 1

/-- The sum of digits of a number consisting of k nines -/
def sum_of_digits (k : ℕ) : ℕ := 9 * k

/-- Theorem stating that N(k) is k times greater than the sum of its digits for all natural k -/
theorem N_is_k_times_sum_of_digits (k : ℕ) :
  N k = k * (sum_of_digits k) :=
sorry

end NUMINAMATH_CALUDE_N_is_k_times_sum_of_digits_l3397_339734


namespace NUMINAMATH_CALUDE_cos_value_given_sin_l3397_339718

theorem cos_value_given_sin (θ : ℝ) (h : Real.sin (θ - π/6) = Real.sqrt 3 / 3) :
  Real.cos (π/3 - 2*θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_given_sin_l3397_339718


namespace NUMINAMATH_CALUDE_digits_of_3_power_20_times_5_power_15_l3397_339793

theorem digits_of_3_power_20_times_5_power_15 : ∃ n : ℕ, 
  (10 ^ (n - 1) ≤ 3^20 * 5^15) ∧ (3^20 * 5^15 < 10^n) ∧ (n = 16) := by sorry

end NUMINAMATH_CALUDE_digits_of_3_power_20_times_5_power_15_l3397_339793


namespace NUMINAMATH_CALUDE_alba_oranges_theorem_l3397_339716

/-- Represents the orange production and sale scenario of the Morales sisters -/
structure OrangeScenario where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  maricela_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℕ
  total_revenue : ℕ

/-- Calculates the number of oranges Alba's trees produce per tree -/
def alba_oranges_per_tree (scenario : OrangeScenario) : ℕ :=
  let total_cups := scenario.total_revenue / scenario.price_per_cup
  let total_oranges := total_cups * scenario.oranges_per_cup
  let gabriela_oranges := scenario.gabriela_oranges_per_tree * scenario.trees_per_sister
  let maricela_oranges := scenario.maricela_oranges_per_tree * scenario.trees_per_sister
  let alba_total_oranges := total_oranges - gabriela_oranges - maricela_oranges
  alba_total_oranges / scenario.trees_per_sister

/-- The main theorem stating that given the scenario conditions, Alba's trees produce 400 oranges per tree -/
theorem alba_oranges_theorem (scenario : OrangeScenario) 
  (h1 : scenario.trees_per_sister = 110)
  (h2 : scenario.gabriela_oranges_per_tree = 600)
  (h3 : scenario.maricela_oranges_per_tree = 500)
  (h4 : scenario.oranges_per_cup = 3)
  (h5 : scenario.price_per_cup = 4)
  (h6 : scenario.total_revenue = 220000) :
  alba_oranges_per_tree scenario = 400 := by
  sorry

end NUMINAMATH_CALUDE_alba_oranges_theorem_l3397_339716


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l3397_339767

/-- Given points A, B, C, and O in a 2D plane, prove that the intersection point P
    of line segments AC and OB has coordinates (3, 3) -/
theorem intersection_point_coordinates :
  let A : Fin 2 → ℝ := ![4, 0]
  let B : Fin 2 → ℝ := ![4, 4]
  let C : Fin 2 → ℝ := ![2, 6]
  let O : Fin 2 → ℝ := ![0, 0]
  ∃ P : Fin 2 → ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (fun i => t * (C i - A i) + A i)) ∧
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (fun i => s * (B i - O i) + O i)) ∧
    P = ![3, 3] :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l3397_339767


namespace NUMINAMATH_CALUDE_min_perimeter_of_divided_rectangle_l3397_339776

/-- Represents the side lengths of the two main squares in the rectangle -/
structure MainSquares where
  a : ℕ
  b : ℕ

/-- Represents the dimensions of the rectangle -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (rect : Rectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- Checks if the given main square side lengths satisfy the rectangle division conditions -/
def satisfiesConditions (squares : MainSquares) : Prop :=
  5 * squares.a + 2 * squares.b = 20 * squares.a - 3 * squares.b

/-- Calculates the rectangle dimensions from the main square side lengths -/
def calculateRectangle (squares : MainSquares) : Rectangle :=
  { width := 2 * squares.a + 2 * squares.b
  , height := 3 * squares.a + 2 * squares.b }

theorem min_perimeter_of_divided_rectangle :
  ∃ (squares : MainSquares),
    satisfiesConditions squares ∧
    ∀ (other : MainSquares),
      satisfiesConditions other →
      perimeter (calculateRectangle squares) ≤ perimeter (calculateRectangle other) ∧
      perimeter (calculateRectangle squares) = 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_divided_rectangle_l3397_339776


namespace NUMINAMATH_CALUDE_building_shadow_length_l3397_339789

/-- Given a flagstaff and a building with their respective heights and shadow lengths,
    prove that the length of the shadow cast by the building is as calculated. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ)
  (building_height : ℝ) :
  flagstaff_height = 17.5 →
  flagstaff_shadow = 40.25 →
  building_height = 12.5 →
  ∃ (building_shadow : ℝ),
    building_shadow = 28.75 ∧
    flagstaff_height / flagstaff_shadow = building_height / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_building_shadow_length_l3397_339789


namespace NUMINAMATH_CALUDE_intersection_values_l3397_339726

/-- The function f(x) = mx² - 6x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 2

/-- The graph of f intersects the x-axis at only one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! x, f m x = 0

theorem intersection_values (m : ℝ) :
  single_intersection m → m = 0 ∨ m = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_values_l3397_339726


namespace NUMINAMATH_CALUDE_cylinder_cube_surface_equality_l3397_339710

theorem cylinder_cube_surface_equality (r h s K : ℝ) : 
  r = 3 → h = 4 → 
  2 * π * r * h = 6 * s^2 → 
  s^3 = 48 / Real.sqrt K → 
  K = 36 / π^3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cube_surface_equality_l3397_339710


namespace NUMINAMATH_CALUDE_integer_root_count_l3397_339746

theorem integer_root_count : ∃! (S : Finset ℝ), 
  (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ 
  (∀ x : ℝ, (∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) → x ∈ S) ∧ 
  Finset.card S = 12 := by sorry

end NUMINAMATH_CALUDE_integer_root_count_l3397_339746


namespace NUMINAMATH_CALUDE_sport_preference_related_to_gender_l3397_339740

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![40, 20],
    ![20, 30]]

-- Define the calculated K^2 value
def calculated_k_squared : ℝ := 7.82

-- Define the critical values and their corresponding probabilities
def critical_values : List (ℝ × ℝ) :=
  [(2.706, 0.10), (3.841, 0.05), (6.635, 0.01), (7.879, 0.005), (10.828, 0.001)]

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.99

-- Theorem statement
theorem sport_preference_related_to_gender :
  ∃ (lower_k upper_k : ℝ) (lower_p upper_p : ℝ),
    (lower_k, lower_p) ∈ critical_values ∧
    (upper_k, upper_p) ∈ critical_values ∧
    lower_k < calculated_k_squared ∧
    calculated_k_squared < upper_k ∧
    lower_p > 1 - target_confidence ∧
    upper_p < 1 - target_confidence :=
by sorry


end NUMINAMATH_CALUDE_sport_preference_related_to_gender_l3397_339740


namespace NUMINAMATH_CALUDE_no_fermat_solutions_with_constraints_l3397_339792

theorem no_fermat_solutions_with_constraints (n : ℕ) (hn : n > 1) :
  ¬∃ (x y z : ℕ), x^n + y^n = z^n ∧ x ≤ n ∧ y ≤ n := by
  sorry

end NUMINAMATH_CALUDE_no_fermat_solutions_with_constraints_l3397_339792


namespace NUMINAMATH_CALUDE_travel_time_difference_l3397_339766

def speed_A : ℝ := 60
def speed_B : ℝ := 45
def distance : ℝ := 360

theorem travel_time_difference :
  (distance / speed_B - distance / speed_A) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_l3397_339766


namespace NUMINAMATH_CALUDE_red_tetrahedron_volume_l3397_339737

/-- The volume of a tetrahedron formed by red vertices in a cube with alternately colored vertices --/
theorem red_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := (1 / 3) * cube_side_length ^ 3 / 2
  let red_tetrahedron_volume := cube_volume - 4 * blue_tetrahedron_volume
  red_tetrahedron_volume = 512 - (4 * 256 / 3) := by
  sorry

#eval 512 - (4 * 256 / 3)  -- To verify the numerical result

end NUMINAMATH_CALUDE_red_tetrahedron_volume_l3397_339737


namespace NUMINAMATH_CALUDE_max_container_weight_l3397_339720

/-- Represents the maximum weight of a container in tons -/
def k : ℕ := 26

/-- Total weight of goods in the warehouse in tons -/
def total_weight : ℕ := 1500

/-- Number of platforms on the train -/
def num_platforms : ℕ := 25

/-- Load capacity of each platform in tons -/
def platform_capacity : ℕ := 80

/-- Represents that containers have integer weights -/
def container_weight_is_integer (weight : ℕ) : Prop := weight > 0

/-- Represents that a container's weight does not exceed k tons -/
def container_weight_limit (weight : ℕ) : Prop := weight ≤ k

/-- Represents that all goods can be transported -/
def can_transport (max_weight : ℕ) : Prop :=
  ∀ (weights : List ℕ),
    (∀ w ∈ weights, container_weight_is_integer w ∧ container_weight_limit w) →
    weights.sum = total_weight →
    ∃ (distribution : List (List ℕ)),
      distribution.length ≤ num_platforms ∧
      (∀ platform ∈ distribution, platform.sum ≤ platform_capacity) ∧
      distribution.join.sum = total_weight

theorem max_container_weight :
  k = 26 ∧ can_transport k ∧ ¬can_transport (k + 1) := by sorry

end NUMINAMATH_CALUDE_max_container_weight_l3397_339720


namespace NUMINAMATH_CALUDE_equal_sum_product_difference_N_l3397_339712

theorem equal_sum_product_difference_N (N : ℕ) :
  ∃ (a b c d : ℕ), (a + b = c + d) ∧ (c * d - a * b = N) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_product_difference_N_l3397_339712


namespace NUMINAMATH_CALUDE_aubreys_garden_aubreys_garden_proof_l3397_339787

/-- Aubrey's Garden Planting Problem -/
theorem aubreys_garden (tomato_cucumber_ratio : Nat) (plants_per_row : Nat) (tomatoes_per_plant : Nat) (total_tomatoes : Nat) : Nat :=
  let tomato_rows := total_tomatoes / (plants_per_row * tomatoes_per_plant)
  let cucumber_rows := tomato_rows * tomato_cucumber_ratio
  tomato_rows + cucumber_rows

/-- Proof of Aubrey's Garden Planting Problem -/
theorem aubreys_garden_proof :
  aubreys_garden 2 8 3 120 = 15 := by
  sorry

end NUMINAMATH_CALUDE_aubreys_garden_aubreys_garden_proof_l3397_339787


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3397_339770

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 4*x + 3
  (f 1 = 0) ∧ (f 3 = 0) ∧ (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3397_339770


namespace NUMINAMATH_CALUDE_nested_sqrt_eighteen_l3397_339771

theorem nested_sqrt_eighteen (y : ℝ) : y = Real.sqrt (18 + y) → y = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_eighteen_l3397_339771


namespace NUMINAMATH_CALUDE_triangle_problem_l3397_339786

theorem triangle_problem (a b c A B C : Real) 
  (h1 : 2 * Real.sqrt 3 * a * b * Real.sin C = a^2 + b^2 - c^2)
  (h2 : a * Real.sin B = b * Real.cos A)
  (h3 : a = 2) :
  C = π/6 ∧ (1/2 * a * c * Real.sin B = (Real.sqrt 3 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3397_339786


namespace NUMINAMATH_CALUDE_evaluate_expression_l3397_339700

theorem evaluate_expression : (24^36) / (72^18) = 8^18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3397_339700


namespace NUMINAMATH_CALUDE_leah_saves_fifty_cents_l3397_339747

/-- Represents the daily savings of Leah in dollars -/
def leah_daily_savings : ℝ := sorry

/-- Represents Josiah's total savings in dollars -/
def josiah_total_savings : ℝ := 0.25 * 24

/-- Represents Leah's total savings in dollars -/
def leah_total_savings : ℝ := leah_daily_savings * 20

/-- Represents Megan's total savings in dollars -/
def megan_total_savings : ℝ := 2 * leah_daily_savings * 12

/-- The total amount saved by all three children -/
def total_savings : ℝ := 28

/-- Theorem stating that Leah's daily savings amount to $0.50 -/
theorem leah_saves_fifty_cents :
  josiah_total_savings + leah_total_savings + megan_total_savings = total_savings →
  leah_daily_savings = 0.50 := by sorry

end NUMINAMATH_CALUDE_leah_saves_fifty_cents_l3397_339747


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l3397_339711

theorem consecutive_even_integers (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- consecutive even integers
    (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) ∧  -- all are even
    (a + c = 128))  -- sum of first and third is 128
  → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l3397_339711


namespace NUMINAMATH_CALUDE_violet_family_ticket_cost_l3397_339729

/-- The cost of separate tickets for Violet's family -/
def separate_ticket_cost (adult_price children_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + children_price * num_children

/-- Theorem: The total cost of separate tickets for Violet's family is $155 -/
theorem violet_family_ticket_cost :
  separate_ticket_cost 35 20 1 6 = 155 :=
by sorry

end NUMINAMATH_CALUDE_violet_family_ticket_cost_l3397_339729


namespace NUMINAMATH_CALUDE_coin_value_problem_l3397_339709

theorem coin_value_problem :
  ∃ (n d q : ℕ),
    n + d + q = 30 ∧
    5 * n + 10 * d + 25 * q = 315 ∧
    10 * n + 25 * d + 5 * q = 5 * n + 10 * d + 25 * q + 120 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_problem_l3397_339709


namespace NUMINAMATH_CALUDE_inequality_proof_l3397_339719

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2*|a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3397_339719


namespace NUMINAMATH_CALUDE_no_triple_exists_l3397_339797

theorem no_triple_exists : ¬∃ (a b c : ℕ+), 
  let p := (a.val - 2) * (b.val - 2) * (c.val - 2) + 12
  Nat.Prime p ∧ 
  ∃ (k : ℕ+), k * p = a.val^2 + b.val^2 + c.val^2 + a.val * b.val * c.val - 2017 := by
  sorry

end NUMINAMATH_CALUDE_no_triple_exists_l3397_339797


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3397_339721

theorem inverse_variation_problem (x y : ℝ) : 
  (x > 0) →
  (y > 0) →
  (∃ k : ℝ, ∀ x y, x^3 * y = k) →
  (2^3 * 8 = x^3 * 512) →
  (y = 512) →
  x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3397_339721


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l3397_339715

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l3397_339715


namespace NUMINAMATH_CALUDE_min_length_3rd_order_repeatable_last_term_value_l3397_339728

/-- Definition of a kth-order repeatable sequence -/
def is_kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, 1 ≤ i ∧ i + k - 1 ≤ m ∧ 1 ≤ j ∧ j + k - 1 ≤ m ∧ i ≠ j ∧
  ∀ t, 0 ≤ t ∧ t < k → a (i + t) = a (j + t)

theorem min_length_3rd_order_repeatable :
  ∀ m : ℕ, m ≥ 3 →
  ((∀ a : ℕ → Fin 2, is_kth_order_repeatable a m 3) ↔ m ≥ 11) :=
sorry

theorem last_term_value (a : ℕ → Fin 2) (m : ℕ) :
  m ≥ 3 →
  a 4 ≠ 1 →
  (¬ is_kth_order_repeatable a m 5) →
  (∃ b : Fin 2, is_kth_order_repeatable (Function.update a (m + 1) b) (m + 1) 5) →
  a m = 0 :=
sorry

end NUMINAMATH_CALUDE_min_length_3rd_order_repeatable_last_term_value_l3397_339728


namespace NUMINAMATH_CALUDE_student_allowance_equation_l3397_339717

/-- The student's weekly allowance satisfies the given equation. -/
theorem student_allowance_equation (A : ℝ) : A > 0 → (3/4 : ℝ) * (1/3 : ℝ) * ((2/5 : ℝ) * A + 4) - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_student_allowance_equation_l3397_339717


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3397_339763

theorem sum_of_four_consecutive_even_integers :
  ∀ a : ℤ,
  (∃ b c d : ℤ, 
    b = a + 2 ∧ 
    c = a + 4 ∧ 
    d = a + 6 ∧ 
    a + d = 136) →
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l3397_339763


namespace NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l3397_339705

/-- The ratio of profits is equal to the ratio of investments -/
theorem profit_ratio_equals_investment_ratio (p q : ℕ) (h : p = 60000 ∧ q = 90000) : 
  (p : ℚ) / q = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_profit_ratio_equals_investment_ratio_l3397_339705


namespace NUMINAMATH_CALUDE_carton_weight_is_three_l3397_339732

/-- The weight of one crate of vegetables in kilograms. -/
def crate_weight : ℝ := 4

/-- The number of crates in the load. -/
def num_crates : ℕ := 12

/-- The number of cartons in the load. -/
def num_cartons : ℕ := 16

/-- The total weight of the load in kilograms. -/
def total_weight : ℝ := 96

/-- The weight of one carton of vegetables in kilograms. -/
def carton_weight : ℝ := 3

/-- Theorem stating that the weight of one carton of vegetables is 3 kilograms. -/
theorem carton_weight_is_three :
  crate_weight * num_crates + carton_weight * num_cartons = total_weight :=
by sorry

end NUMINAMATH_CALUDE_carton_weight_is_three_l3397_339732


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3397_339725

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let perimeter : ℝ := 3 * side_length
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3397_339725


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3397_339704

/-- Represents a pair of integers in the sequence -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Generates the nth pair in the sequence -/
def generatePair (n : ℕ) : IntPair :=
  sorry

/-- The main theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_five_seven : generatePair 60 = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l3397_339704
