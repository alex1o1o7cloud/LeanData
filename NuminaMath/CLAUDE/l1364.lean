import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1364_136401

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1364_136401


namespace NUMINAMATH_CALUDE_lopez_seating_theorem_l1364_136472

/-- Represents the number of family members -/
def family_members : ℕ := 5

/-- Represents the number of front seats in the car -/
def front_seats : ℕ := 2

/-- Represents the number of back seats in the car -/
def back_seats : ℕ := 3

/-- Represents the number of possible drivers (Mr. or Mrs. Lopez) -/
def possible_drivers : ℕ := 2

/-- Calculates the number of possible seating arrangements for the Lopez family -/
def seating_arrangements : ℕ :=
  possible_drivers * (family_members - 1) * Nat.factorial (family_members - 2)

theorem lopez_seating_theorem :
  seating_arrangements = 48 :=
sorry

end NUMINAMATH_CALUDE_lopez_seating_theorem_l1364_136472


namespace NUMINAMATH_CALUDE_factor_and_divisor_properties_l1364_136438

theorem factor_and_divisor_properties :
  (∃ n : ℤ, 24 = 4 * n) ∧
  (∃ n : ℤ, 209 = 19 * n) ∧ ¬(∃ m : ℤ, 63 = 19 * m) ∧
  (∃ k : ℤ, 180 = 9 * k) := by
sorry

end NUMINAMATH_CALUDE_factor_and_divisor_properties_l1364_136438


namespace NUMINAMATH_CALUDE_exists_sequence_to_target_state_l1364_136420

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the replacement rules
inductive ReplacementRule
| Rule1 -- 3 black → 1 black + 2 white
| Rule2 -- 2 black + 1 white → 3 black
| Rule3 -- 1 black + 2 white → 2 white
| Rule4 -- 3 white → 2 white + 1 black

-- Define a function to apply a rule to an urn state
def applyRule (state : UrnState) (rule : ReplacementRule) : UrnState :=
  match rule with
  | ReplacementRule.Rule1 => 
      if state.black ≥ 3 then UrnState.mk (state.white + 2) (state.black - 2) else state
  | ReplacementRule.Rule2 => 
      if state.black ≥ 2 ∧ state.white ≥ 1 then UrnState.mk (state.white - 1) (state.black + 1) else state
  | ReplacementRule.Rule3 => 
      if state.black ≥ 1 ∧ state.white ≥ 2 then UrnState.mk state.white (state.black - 1) else state
  | ReplacementRule.Rule4 => 
      if state.white ≥ 3 then UrnState.mk (state.white - 1) (state.black + 1) else state

-- Define the initial state
def initialState : UrnState := UrnState.mk 50 50

-- Define the target state
def targetState : UrnState := UrnState.mk 2 0

-- Theorem to prove
theorem exists_sequence_to_target_state : 
  ∃ (sequence : List ReplacementRule), 
    (sequence.foldl applyRule initialState) = targetState :=
sorry

end NUMINAMATH_CALUDE_exists_sequence_to_target_state_l1364_136420


namespace NUMINAMATH_CALUDE_sportswear_price_reduction_l1364_136450

/-- Given two equal percentage reductions that reduce a price from 560 to 315,
    prove that the equation 560(1-x)^2 = 315 holds true, where x is the decimal
    form of the percentage reduction. -/
theorem sportswear_price_reduction (x : ℝ) : 
  (∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 560 * (1 - x)^2 = 315) :=
by sorry

end NUMINAMATH_CALUDE_sportswear_price_reduction_l1364_136450


namespace NUMINAMATH_CALUDE_zero_sum_points_for_m_3_unique_zero_sum_point_condition_l1364_136439

/-- Definition of a "zero-sum point" in the Cartesian coordinate system -/
def is_zero_sum_point (x y : ℝ) : Prop := x + y = 0

/-- The quadratic function y = x^2 + 3x + m -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + m

theorem zero_sum_points_for_m_3 :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_zero_sum_point x₁ y₁ ∧
    is_zero_sum_point x₂ y₂ ∧
    quadratic_function 3 x₁ = y₁ ∧
    quadratic_function 3 x₂ = y₂ ∧
    x₁ = -1 ∧ y₁ = 1 ∧
    x₂ = -3 ∧ y₂ = 3 :=
sorry

theorem unique_zero_sum_point_condition (m : ℝ) :
  (∃! (x y : ℝ), is_zero_sum_point x y ∧ quadratic_function m x = y) ↔ m = 4 :=
sorry

end NUMINAMATH_CALUDE_zero_sum_points_for_m_3_unique_zero_sum_point_condition_l1364_136439


namespace NUMINAMATH_CALUDE_correct_probability_l1364_136412

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
  | Clubs
  | Diamonds
  | Hearts
  | Spades

/-- Represents the rank of a card -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten
  | Jack | Queen | King

/-- A card is a face card if it's a Jack, Queen, or King -/
def isFaceCard (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- The probability of drawing a club as the first card and a face card diamond as the second card -/
def consecutiveDrawProbability (d : Deck) : Rat :=
  (13 : Rat) / 884

theorem correct_probability (d : Deck) :
  consecutiveDrawProbability d = 13 / 884 := by
  sorry

end NUMINAMATH_CALUDE_correct_probability_l1364_136412


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1364_136473

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1364_136473


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_two_l1364_136459

theorem arctan_sum_equals_pi_over_two (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/10) + Real.arctan (1/30) + Real.arctan (1/y) = π/2 →
  y = 547/620 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_two_l1364_136459


namespace NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l1364_136454

theorem sum_of_squares_in_ratio (a b c : ℚ) : 
  (a : ℚ) + b + c = 9 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 1701 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l1364_136454


namespace NUMINAMATH_CALUDE_plates_in_second_purchase_is_20_l1364_136498

/-- The cost of one paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of one paper cup -/
def cup_cost : ℝ := sorry

/-- The number of plates in the second purchase -/
def plates_in_second_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $7.50 -/
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

/-- The total cost of some plates and 40 cups is $1.50 -/
axiom second_purchase : plates_in_second_purchase * plate_cost + 40 * cup_cost = 1.50

theorem plates_in_second_purchase_is_20 : plates_in_second_purchase = 20 := by sorry

end NUMINAMATH_CALUDE_plates_in_second_purchase_is_20_l1364_136498


namespace NUMINAMATH_CALUDE_flagpole_height_l1364_136413

/-- The height of a flagpole given specific measurements of surrounding stakes -/
theorem flagpole_height (AB OC OD OH : ℝ) (hAB : AB = 120) 
  (hHC : OH^2 + OC^2 = 170^2) (hHD : OH^2 + OD^2 = 100^2) (hCD : OC^2 + OD^2 = AB^2) :
  OH = 50 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l1364_136413


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_169_l1364_136486

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_169_l1364_136486


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1364_136469

theorem consecutive_integers_average (a : ℤ) (b : ℚ) : 
  (a > 0) →
  (b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4)) / 5) →
  ((b + (b + 10) + (b + 20)) / 3 = a + 12) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1364_136469


namespace NUMINAMATH_CALUDE_odd_area_rectangles_count_l1364_136436

/-- Represents a 3x3 grid of rectangles with integer side lengths -/
structure Grid :=
  (horizontal_lengths : Fin 4 → ℕ)
  (vertical_lengths : Fin 4 → ℕ)

/-- Counts the number of rectangles with odd area in the grid -/
def count_odd_area_rectangles (g : Grid) : ℕ :=
  sorry

/-- Theorem stating that the number of rectangles with odd area is either 0 or 4 -/
theorem odd_area_rectangles_count (g : Grid) : 
  count_odd_area_rectangles g = 0 ∨ count_odd_area_rectangles g = 4 :=
sorry

end NUMINAMATH_CALUDE_odd_area_rectangles_count_l1364_136436


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l1364_136411

theorem cos_2alpha_plus_pi_3 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 12) = 3 / 5) :
  Real.cos (2 * α + π / 3) = -24 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l1364_136411


namespace NUMINAMATH_CALUDE_unique_solution_l1364_136430

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  (x + 1 = z + w + z*w*x) ∧
  (y - 1 = w + x + w*x*y) ∧
  (z + 2 = x + y + x*y*z) ∧
  (w - 2 = y + z + y*z*w)

-- Theorem statement
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1364_136430


namespace NUMINAMATH_CALUDE_reflection_result_l1364_136467

/-- Reflects a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point across the line y = -x + 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (-p'.2, -p'.1) -- Reflect across y = -x
  (p''.1, p''.2 + 2)        -- Translate back up by 2

/-- The final position of point R after two reflections -/
def R_final : ℝ × ℝ :=
  reflect_line (reflect_x_axis (6, 1))

theorem reflection_result :
  R_final = (-3, -4) :=
by sorry

end NUMINAMATH_CALUDE_reflection_result_l1364_136467


namespace NUMINAMATH_CALUDE_polynomial_square_b_value_l1364_136488

theorem polynomial_square_b_value (a b : ℚ) :
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) →
  b = 25/64 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_b_value_l1364_136488


namespace NUMINAMATH_CALUDE_time_after_1876_minutes_l1364_136437

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time and wraps around to the next day if necessary -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

def startTime : Time := { hours := 15, minutes := 0 }  -- 3:00 PM
def minutesToAdd : Nat := 1876

theorem time_after_1876_minutes :
  addMinutes startTime minutesToAdd = { hours := 10, minutes := 16 } := by
  sorry

end NUMINAMATH_CALUDE_time_after_1876_minutes_l1364_136437


namespace NUMINAMATH_CALUDE_average_sitting_time_l1364_136442

def num_students : ℕ := 6
def num_seats : ℕ := 4
def travel_time_hours : ℕ := 3
def travel_time_minutes : ℕ := 12

theorem average_sitting_time :
  let total_minutes : ℕ := travel_time_hours * 60 + travel_time_minutes
  let total_sitting_time : ℕ := num_seats * total_minutes
  let avg_sitting_time : ℕ := total_sitting_time / num_students
  avg_sitting_time = 128 := by
sorry

end NUMINAMATH_CALUDE_average_sitting_time_l1364_136442


namespace NUMINAMATH_CALUDE_hexagon_area_l1364_136443

/-- A regular hexagon with vertices P and R -/
structure RegularHexagon where
  P : ℝ × ℝ
  R : ℝ × ℝ

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon with P at (0,0) and R at (10,2) is 156√3 -/
theorem hexagon_area :
  let h : RegularHexagon := { P := (0, 0), R := (10, 2) }
  area h = 156 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l1364_136443


namespace NUMINAMATH_CALUDE_next_perfect_square_with_two_twos_l1364_136402

/-- A number begins with two 2s if its first two digits are 2 when written in base 10. -/
def begins_with_two_twos (n : ℕ) : Prop :=
  n ≥ 220 ∧ n < 230

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- A perfect square is a natural number that is the square of another natural number. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem next_perfect_square_with_two_twos : 
  (∀ n : ℕ, is_perfect_square n ∧ begins_with_two_twos n ∧ n < 2500 → n ≤ 225) ∧
  is_perfect_square 2500 ∧
  begins_with_two_twos 2500 ∧
  sum_of_digits 2500 = 7 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_square_with_two_twos_l1364_136402


namespace NUMINAMATH_CALUDE_set_A_equals_roster_l1364_136455

def A : Set ℤ := {x | ∃ (n : ℕ+), 6 / (5 - x) = n}

theorem set_A_equals_roster : A = {-1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_roster_l1364_136455


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1364_136419

theorem problem_1 : (-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21 := by sorry

theorem problem_2 : -99*(8/9) * 8 = -799*(1/9) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1364_136419


namespace NUMINAMATH_CALUDE_aquarium_visitors_l1364_136493

theorem aquarium_visitors (total : ℕ) (ill_percentage : ℚ) : 
  total = 500 → ill_percentage = 40 / 100 → 
  (total : ℚ) * (1 - ill_percentage) = 300 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l1364_136493


namespace NUMINAMATH_CALUDE_transformed_variance_l1364_136476

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_variance
  (h : variance x = 3) :
  variance (fun i => 2 * x i + 4) = 12 := by sorry

end NUMINAMATH_CALUDE_transformed_variance_l1364_136476


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1364_136482

/-- The parabola function f(x) = x^2 - 10x + 20 --/
def f (x : ℝ) : ℝ := x^2 - 10*x + 20

/-- A square inscribed between a parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ × ℝ
  side_length : ℝ
  h1 : center.1 = 5 -- The x-coordinate of the center is at the vertex of the parabola
  h2 : center.2 = side_length / 2 -- The y-coordinate of the center is half the side length
  h3 : f (center.1 + side_length / 2) = side_length -- The top right corner lies on the parabola

/-- The theorem stating that the area of the inscribed square is 400 --/
theorem inscribed_square_area (s : InscribedSquare) : s.side_length^2 = 400 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_l1364_136482


namespace NUMINAMATH_CALUDE_triangle_tan_half_angles_inequality_l1364_136409

theorem triangle_tan_half_angles_inequality (A B C : ℝ) (h₁ : A + B + C = π) :
  Real.tan (A / 2) * Real.tan (B / 2) * Real.tan (C / 2) ≤ Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_half_angles_inequality_l1364_136409


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1364_136458

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a1 : a 1 = -16) 
    (h_a4 : a 4 = 8) : 
  a 7 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1364_136458


namespace NUMINAMATH_CALUDE_equation_two_real_roots_l1364_136424

/-- The equation √(x + 9) - 2√(x - 8) + 3 = 0 has exactly two real roots in the domain x ≥ 8 -/
theorem equation_two_real_roots :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ x ∈ s, x ≥ 8 ∧ Real.sqrt (x + 9) - 2 * Real.sqrt (x - 8) + 3 = 0) ∧
  (∀ x ≥ 8, Real.sqrt (x + 9) - 2 * Real.sqrt (x - 8) + 3 = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_equation_two_real_roots_l1364_136424


namespace NUMINAMATH_CALUDE_double_vinegar_theorem_l1364_136407

/-- Represents the ratio of oil to vinegar in a salad dressing -/
structure SaladDressing where
  oil : ℚ
  vinegar : ℚ

/-- The initial ratio of oil to vinegar -/
def initial_ratio : SaladDressing :=
  { oil := 3, vinegar := 1 }

/-- Doubles the amount of vinegar in a salad dressing -/
def double_vinegar (sd : SaladDressing) : SaladDressing :=
  { oil := sd.oil, vinegar := 2 * sd.vinegar }

/-- Calculates the ratio of oil to vinegar -/
def ratio (sd : SaladDressing) : ℚ :=
  sd.oil / sd.vinegar

/-- Theorem: Doubling the vinegar in the initial 3:1 ratio results in a 3:2 ratio -/
theorem double_vinegar_theorem :
  ratio (double_vinegar initial_ratio) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_vinegar_theorem_l1364_136407


namespace NUMINAMATH_CALUDE_total_jellybeans_needed_l1364_136457

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_beans : ℕ := 50

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_needed_l1364_136457


namespace NUMINAMATH_CALUDE_borrowed_amount_is_2500_l1364_136429

/-- Proves that the borrowed amount is 2500 given the problem conditions --/
theorem borrowed_amount_is_2500 
  (borrowed_rate : ℚ) 
  (lent_rate : ℚ) 
  (time : ℚ) 
  (yearly_gain : ℚ) 
  (h1 : borrowed_rate = 4 / 100)
  (h2 : lent_rate = 6 / 100)
  (h3 : time = 2)
  (h4 : yearly_gain = 100) : 
  ∃ (P : ℚ), P = 2500 ∧ 
    (lent_rate * P * time) - (borrowed_rate * P * time) = yearly_gain * time :=
by sorry

end NUMINAMATH_CALUDE_borrowed_amount_is_2500_l1364_136429


namespace NUMINAMATH_CALUDE_abs_eq_self_implies_nonnegative_l1364_136484

theorem abs_eq_self_implies_nonnegative (a : ℝ) : |a| = a → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_implies_nonnegative_l1364_136484


namespace NUMINAMATH_CALUDE_election_vote_difference_l1364_136422

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 10000 → 
  candidate_percentage = 30/100 → 
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
sorry

end NUMINAMATH_CALUDE_election_vote_difference_l1364_136422


namespace NUMINAMATH_CALUDE_min_value_of_ab_min_value_is_6_plus_4sqrt2_l1364_136483

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y + 2 = 2 * (x + y) → a * b ≤ x * y :=
by sorry

theorem min_value_is_6_plus_4sqrt2 (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a * b + 2 = 2 * (a + b)) :
  a * b = 6 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_ab_min_value_is_6_plus_4sqrt2_l1364_136483


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l1364_136474

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 →
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l1364_136474


namespace NUMINAMATH_CALUDE_point_coordinates_l1364_136466

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the third quadrant with distance 3 to the x-axis
    and distance 5 to the y-axis has coordinates (-5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : ThirdQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 5) : 
  p = Point.mk (-5) (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1364_136466


namespace NUMINAMATH_CALUDE_no_three_squares_l1364_136479

theorem no_three_squares (x : ℤ) : ¬(∃ (a b c : ℤ), (2*x - 1 = a^2) ∧ (5*x - 1 = b^2) ∧ (13*x - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_three_squares_l1364_136479


namespace NUMINAMATH_CALUDE_function_negative_on_interval_l1364_136490

theorem function_negative_on_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 1), x^2 + m*x - 1 < 0) → 
  -Real.sqrt 2 / 2 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_function_negative_on_interval_l1364_136490


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l1364_136406

theorem simplify_sqrt_product : Real.sqrt 18 * Real.sqrt 32 * Real.sqrt 2 = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l1364_136406


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1364_136414

/-- Given an arithmetic sequence with first term a₁, common difference d, and n-th term aₙ,
    this function calculates the n-th term. -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem fifth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 5 = 19 := by
  sorry

#check fifth_term_of_sequence

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1364_136414


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2021_mod_19_l1364_136434

theorem residue_of_11_pow_2021_mod_19 :
  (11 : ℤ) ^ 2021 ≡ 17 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2021_mod_19_l1364_136434


namespace NUMINAMATH_CALUDE_student_weight_loss_l1364_136446

theorem student_weight_loss (student_weight sister_weight : ℝ) 
  (h1 : student_weight = 90)
  (h2 : student_weight + sister_weight = 132) : 
  ∃ (weight_loss : ℝ), 
    weight_loss = 6 ∧ 
    student_weight - weight_loss = 2 * sister_weight :=
by sorry

end NUMINAMATH_CALUDE_student_weight_loss_l1364_136446


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1364_136477

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I - 2) / (1 + Complex.I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1364_136477


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_1_sufficient_necessary_condition_l1364_136403

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 6 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1: Intersection of A and B when m = 1
theorem intersection_A_B_when_m_1 : A ∩ B 1 = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2: Condition for x ∈ A to be sufficient and necessary for x ∈ B
theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_1_sufficient_necessary_condition_l1364_136403


namespace NUMINAMATH_CALUDE_intersection_point_ratio_l1364_136447

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type with 60° inclination passing through (1, 0) -/
structure Line where
  x : ℝ
  y : ℝ
  eq : y = Real.sqrt 3 * (x - 1)

/-- Intersection point of the parabola and the line -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : Parabola
  on_line : Line

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem intersection_point_ratio 
  (A B : IntersectionPoint) 
  (h1 : A.x + 1 > B.x + 1) : 
  (A.x + 1) / (B.x + 1) = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_ratio_l1364_136447


namespace NUMINAMATH_CALUDE_andy_stencils_l1364_136452

/-- Calculates the number of stencils painted given the following conditions:
  * Hourly wage
  * Pay per racquet strung
  * Pay per grommet change
  * Pay per stencil painted
  * Hours worked
  * Number of racquets strung
  * Number of grommet sets changed
  * Total earnings -/
def stencils_painted (hourly_wage : ℚ) (pay_per_racquet : ℚ) (pay_per_grommet : ℚ) 
  (pay_per_stencil : ℚ) (hours_worked : ℚ) (racquets_strung : ℕ) (grommets_changed : ℕ) 
  (total_earnings : ℚ) : ℕ :=
  sorry

theorem andy_stencils : 
  stencils_painted 9 15 10 1 8 7 2 202 = 5 :=
sorry

end NUMINAMATH_CALUDE_andy_stencils_l1364_136452


namespace NUMINAMATH_CALUDE_hypotenuse_sum_of_two_triangles_l1364_136448

theorem hypotenuse_sum_of_two_triangles : 
  let triangle1_leg1 : ℝ := 120
  let triangle1_leg2 : ℝ := 160
  let triangle2_leg1 : ℝ := 30
  let triangle2_leg2 : ℝ := 40
  let hypotenuse1 := Real.sqrt (triangle1_leg1^2 + triangle1_leg2^2)
  let hypotenuse2 := Real.sqrt (triangle2_leg1^2 + triangle2_leg2^2)
  hypotenuse1 + hypotenuse2 = 250 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_sum_of_two_triangles_l1364_136448


namespace NUMINAMATH_CALUDE_turtle_position_and_distance_l1364_136445

def turtle_movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

theorem turtle_position_and_distance :
  (List.sum turtle_movements = 5) ∧
  (List.sum (List.map Int.natAbs turtle_movements) = 47) := by
  sorry

end NUMINAMATH_CALUDE_turtle_position_and_distance_l1364_136445


namespace NUMINAMATH_CALUDE_complex_power_24_l1364_136433

theorem complex_power_24 : (((1 - Complex.I) / Real.sqrt 2) ^ 24 : ℂ) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_24_l1364_136433


namespace NUMINAMATH_CALUDE_messages_per_member_per_day_l1364_136492

theorem messages_per_member_per_day :
  let initial_members : ℕ := 150
  let removed_members : ℕ := 20
  let remaining_members : ℕ := initial_members - removed_members
  let total_weekly_messages : ℕ := 45500
  let messages_per_day : ℕ := total_weekly_messages / 7
  let messages_per_member_per_day : ℕ := messages_per_day / remaining_members
  messages_per_member_per_day = 50 :=
by sorry

end NUMINAMATH_CALUDE_messages_per_member_per_day_l1364_136492


namespace NUMINAMATH_CALUDE_outdoor_players_count_l1364_136408

/-- Represents the number of players in different categories -/
structure PlayerCounts where
  total : ℕ
  indoor : ℕ
  both : ℕ
  outdoor : ℕ

/-- Theorem stating the number of outdoor players given the conditions -/
theorem outdoor_players_count (p : PlayerCounts)
  (h_total : p.total = 400)
  (h_indoor : p.indoor = 110)
  (h_both : p.both = 60)
  (h_valid : p.total ≥ p.indoor + p.outdoor - p.both) :
  p.outdoor = 350 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_players_count_l1364_136408


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1364_136417

/-- A line passing through (3, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (3, 2)
  point_condition : 2 = m * 3 + b
  -- The x and y intercepts are equal and non-zero
  intercept_condition : ∃ (a : ℝ), a ≠ 0 ∧ a = b ∧ a = -b/m

/-- The equation of the line with equal intercepts passing through (3, 2) is x + y = 5 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  l.m = -1 ∧ l.b = 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1364_136417


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt2_minus3_l1364_136428

theorem quadratic_root_sqrt2_minus3 :
  let f : ℝ → ℝ := λ x => x^2 + 6*x + 7
  f (Real.sqrt 2 - 3) = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt2_minus3_l1364_136428


namespace NUMINAMATH_CALUDE_three_X_seven_l1364_136432

/-- Operation X is defined as a X b = b + 10*a - a^2 + 2*a*b -/
def X (a b : ℤ) : ℤ := b + 10*a - a^2 + 2*a*b

/-- The value of 3X7 is 70 -/
theorem three_X_seven : X 3 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_three_X_seven_l1364_136432


namespace NUMINAMATH_CALUDE_existence_of_unsolvable_linear_system_l1364_136404

theorem existence_of_unsolvable_linear_system :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, a₁ * x + b₁ * y ≠ c₁ ∨ a₂ * x + b₂ * y ≠ c₂) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unsolvable_linear_system_l1364_136404


namespace NUMINAMATH_CALUDE_remainder_problem_l1364_136481

theorem remainder_problem (x y : ℤ) (k : ℤ) : 
  x = 159 * k + 37 → 
  y = 5 * x^2 + 18 * x + 22 → 
  y % 13 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1364_136481


namespace NUMINAMATH_CALUDE_break_room_seating_capacity_l1364_136449

/-- Given a break room with tables and total seating capacity, 
    calculate the number of people each table can seat. -/
def people_per_table (num_tables : ℕ) (total_capacity : ℕ) : ℕ :=
  total_capacity / num_tables

theorem break_room_seating_capacity 
  (num_tables : ℕ) (total_capacity : ℕ) 
  (h1 : num_tables = 4) 
  (h2 : total_capacity = 32) : 
  people_per_table num_tables total_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_break_room_seating_capacity_l1364_136449


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1364_136460

theorem complex_number_in_second_quadrant : ∃ (z : ℂ), 
  z = (1 + 2*I) - (3 - 4*I) ∧ 
  (z.re < 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1364_136460


namespace NUMINAMATH_CALUDE_larger_number_problem_l1364_136499

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 35) : L = 1631 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1364_136499


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1364_136489

theorem nested_fraction_evaluation :
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 ∧ 8 / 21 ≠ 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1364_136489


namespace NUMINAMATH_CALUDE_final_numbers_l1364_136405

def process (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i j : ℕ), i ≤ n ∧ j ≤ n ∧ m = i * j}

theorem final_numbers (n : ℕ) :
  process n = {m : ℕ | ∃ (k : ℕ), k ≤ n ∧ m = k^2} :=
by sorry

#check final_numbers 2009

end NUMINAMATH_CALUDE_final_numbers_l1364_136405


namespace NUMINAMATH_CALUDE_A_equals_B_l1364_136475

def A : Set ℤ := {x | ∃ n : ℤ, x = 2*n - 1}
def B : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l1364_136475


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l1364_136464

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l1364_136464


namespace NUMINAMATH_CALUDE_trig_identity_l1364_136410

theorem trig_identity (α : ℝ) : 
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + π/6) / Real.sin (4 * α - π/6) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l1364_136410


namespace NUMINAMATH_CALUDE_frequency_of_score_range_l1364_136451

theorem frequency_of_score_range (total_students : ℕ) (high_scorers : ℕ) 
  (h1 : total_students = 50) (h2 : high_scorers = 10) : 
  (high_scorers : ℚ) / total_students = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_score_range_l1364_136451


namespace NUMINAMATH_CALUDE_log_problem_l1364_136441

theorem log_problem (y : ℝ) : y = (Real.log 3 / Real.log 9) ^ (Real.log 16 / Real.log 4) → Real.log y / Real.log 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1364_136441


namespace NUMINAMATH_CALUDE_library_books_checkout_l1364_136478

theorem library_books_checkout (fiction_books : ℕ) (nonfiction_ratio fiction_ratio : ℕ) : 
  fiction_books = 24 → 
  nonfiction_ratio = 7 →
  fiction_ratio = 6 →
  ∃ (total_books : ℕ), total_books = fiction_books + (fiction_books * nonfiction_ratio) / fiction_ratio ∧ total_books = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_checkout_l1364_136478


namespace NUMINAMATH_CALUDE_odd_function_condition_l1364_136423

/-- The function f(x) defined as (3^(x+1) - 1) / (3^x - 1) + a * (sin x + cos x)^2 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (3^(x+1) - 1) / (3^x - 1) + a * (Real.sin x + Real.cos x)^2

/-- Theorem stating that for f to be an odd function, a must equal -2 --/
theorem odd_function_condition (a : ℝ) : 
  (∀ x, f a x = -f a (-x)) ↔ a = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l1364_136423


namespace NUMINAMATH_CALUDE_no_max_min_value_l1364_136453

/-- The function f(x) = x³ - (3/2)x² + 1 has neither a maximum value nor a minimum value -/
theorem no_max_min_value (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - (3/2)*x^2 + 1) :
  (¬ ∃ y, ∀ x, f x ≤ f y) ∧ (¬ ∃ y, ∀ x, f x ≥ f y) := by
  sorry

end NUMINAMATH_CALUDE_no_max_min_value_l1364_136453


namespace NUMINAMATH_CALUDE_fourth_root_equation_l1364_136435

theorem fourth_root_equation (m : ℝ) : (m^4)^(1/4) = 2 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l1364_136435


namespace NUMINAMATH_CALUDE_opposite_of_one_fourth_l1364_136496

theorem opposite_of_one_fourth : -(1 / 4 : ℚ) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_fourth_l1364_136496


namespace NUMINAMATH_CALUDE_four_Y_three_equals_twentyfive_l1364_136416

-- Define the Y operation
def Y (a b : ℝ) : ℝ := (2 * a^2 - 3 * a * b + b^2)^2

-- Theorem statement
theorem four_Y_three_equals_twentyfive : Y 4 3 = 25 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_twentyfive_l1364_136416


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1364_136495

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 45 * k + 31 → ∃ m : ℤ, N = 15 * m + 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1364_136495


namespace NUMINAMATH_CALUDE_johns_local_taxes_l1364_136491

/-- Proves that given John's hourly wage and local tax rate, the amount of local taxes paid in cents per hour is 60 cents. -/
theorem johns_local_taxes (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_local_taxes_l1364_136491


namespace NUMINAMATH_CALUDE_function_through_points_l1364_136421

/-- Given a function f(x) = a^x - k passing through (1,3) and (0,2), prove f(x) = 2^x + 1 -/
theorem function_through_points (a k : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a^x - k) 
    (h2 : f 1 = 3) 
    (h3 : f 0 = 2) : 
    ∀ x, f x = 2^x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_through_points_l1364_136421


namespace NUMINAMATH_CALUDE_soda_duration_is_40_l1364_136425

/-- The number of days soda bottles last given the total number of bottles and daily consumption. -/
def sodaDuration (totalBottles : ℕ) (dailyConsumption : ℕ) : ℕ :=
  totalBottles / dailyConsumption

/-- Theorem stating that the soda bottles will last 40 days. -/
theorem soda_duration_is_40 :
  sodaDuration 360 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_soda_duration_is_40_l1364_136425


namespace NUMINAMATH_CALUDE_white_marbles_count_l1364_136461

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 20 →
  blue = 6 →
  red = 9 →
  prob_red_or_white = 7/10 →
  total - blue - red = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l1364_136461


namespace NUMINAMATH_CALUDE_polygon_with_720_degrees_is_hexagon_l1364_136400

/-- A polygon with a sum of interior angles of 720° has 6 sides. -/
theorem polygon_with_720_degrees_is_hexagon :
  ∀ n : ℕ,
  (180 * (n - 2) = 720) →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_720_degrees_is_hexagon_l1364_136400


namespace NUMINAMATH_CALUDE_ratio_PC_PB_is_zero_l1364_136465

/-- A square with side length 6, where N is the midpoint of AB and P is the intersection of BD and CN -/
structure SquareABCD where
  -- Define the square
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Conditions
  is_square : A = (0, 0) ∧ B = (6, 0) ∧ C = (6, 6) ∧ D = (0, 6)
  -- Define N as midpoint of AB
  N : ℝ × ℝ
  N_is_midpoint : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Define P as intersection of BD and CN
  P : ℝ × ℝ
  P_on_BD : (P.2 - D.2) = ((B.2 - D.2) / (B.1 - D.1)) * (P.1 - D.1)
  P_on_CN : (P.2 - N.2) = ((C.2 - N.2) / (C.1 - N.1)) * (P.1 - N.1)

/-- The ratio of PC to PB is 0 -/
theorem ratio_PC_PB_is_zero (square : SquareABCD) : 
  let PC := Real.sqrt ((square.P.1 - square.C.1)^2 + (square.P.2 - square.C.2)^2)
  let PB := Real.sqrt ((square.P.1 - square.B.1)^2 + (square.P.2 - square.B.2)^2)
  PC / PB = 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_PC_PB_is_zero_l1364_136465


namespace NUMINAMATH_CALUDE_inequality_proof_l1364_136497

theorem inequality_proof (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1364_136497


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1364_136480

/-- The number of players in the chess tournament -/
def num_players : ℕ := 7

/-- The total number of games played in the tournament -/
def total_games : ℕ := 42

/-- The number of times each player plays against each opponent -/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  (num_players * (num_players - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1364_136480


namespace NUMINAMATH_CALUDE_error_percentage_l1364_136494

theorem error_percentage (x : ℝ) (h : x > 0) :
  ∃ ε > 0, abs ((x^2 - x/8) / x^2 * 100 - 88) < ε :=
sorry

end NUMINAMATH_CALUDE_error_percentage_l1364_136494


namespace NUMINAMATH_CALUDE_value_of_M_l1364_136440

theorem value_of_M (m n p M : ℝ) 
  (h1 : M = m / (n + p))
  (h2 : M = n / (p + m))
  (h3 : M = p / (m + n)) :
  M = 1/2 ∨ M = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_M_l1364_136440


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l1364_136418

-- Define the sets A, B, C, and D
def A : Set ℝ := {x | x^2 + 3*x - 4 ≥ 0}
def B : Set ℝ := {x | (x-2)/x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < 1+a}
def D (m : ℝ) : Set ℝ := {x | x^2 - (2*m+1/2)*x + m*(m+1/2) ≤ 0}

-- Part 1
theorem range_of_a :
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ a ≥ 1/2 :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ A ∩ B) ∧
           (∃ x : ℝ, x ∈ A ∩ B ∧ x ∉ D m) ↔
  1 ≤ m ∧ m ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l1364_136418


namespace NUMINAMATH_CALUDE_number_difference_l1364_136485

theorem number_difference (a b : ℕ) : 
  a + b = 25800 →
  ∃ k : ℕ, b = 12 * k →
  a = k →
  b - a = 21824 :=
by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1364_136485


namespace NUMINAMATH_CALUDE_gumball_price_l1364_136456

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_price (num_gumballs : ℕ) (total_cents : ℕ) (price_per_gumball : ℕ) 
  (h1 : num_gumballs = 4)
  (h2 : total_cents = 32)
  (h3 : price_per_gumball * num_gumballs = total_cents) :
  price_per_gumball = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumball_price_l1364_136456


namespace NUMINAMATH_CALUDE_a_range_l1364_136427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

def is_increasing (seq : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → seq n < seq m

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, is_increasing (λ n => f a n)) →
  a ∈ Set.Ioo 2 3 :=
sorry

end NUMINAMATH_CALUDE_a_range_l1364_136427


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l1364_136431

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  let adult_total_cost := num_adults * adult_ticket_price
  let child_total_cost := num_children * child_ticket_price
  adult_total_cost - child_total_cost = 50 := by
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l1364_136431


namespace NUMINAMATH_CALUDE_find_divisor_l1364_136487

theorem find_divisor (n m d : ℕ) (h1 : n = 2304) (h2 : m = 2319) 
  (h3 : m > n) (h4 : m % d = 0) 
  (h5 : ∀ k, n < k ∧ k < m → k % d ≠ 0) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1364_136487


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1364_136470

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x ≤ y → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l1364_136470


namespace NUMINAMATH_CALUDE_find_a_l1364_136471

/-- The system of equations -/
def system (a b m : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y = 2 ∧ m * x - 7 * y = -8

/-- Xiao Li's solution -/
def solution_li (a b : ℝ) : Prop :=
  a * (-2) + b * 3 = 2

/-- Xiao Zhang's solution -/
def solution_zhang (a b : ℝ) : Prop :=
  a * (-2) + b * 2 = 2

/-- Theorem stating that if both solutions satisfy the first equation, then a = -1 -/
theorem find_a (a b m : ℝ) : solution_li a b ∧ solution_zhang a b → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l1364_136471


namespace NUMINAMATH_CALUDE_number_of_advertisements_number_of_advertisements_proof_l1364_136463

/-- The number of advertisements shown during a race, given their duration, 
    cost per minute, and total transmission cost. -/
theorem number_of_advertisements (ad_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) : ℕ :=
  5
where
  ad_duration := 3
  cost_per_minute := 4000
  total_cost := 60000

/-- Proof of the theorem -/
theorem number_of_advertisements_proof :
  number_of_advertisements 3 4000 60000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_advertisements_number_of_advertisements_proof_l1364_136463


namespace NUMINAMATH_CALUDE_uncovered_fraction_of_plates_l1364_136468

/-- The fraction of a circular plate with diameter 12 inches that is not covered
    by a smaller circular plate with diameter 10 inches placed on top of it is 11/36. -/
theorem uncovered_fraction_of_plates (small_diameter large_diameter : ℝ) 
  (h_small : small_diameter = 10)
  (h_large : large_diameter = 12) :
  (large_diameter^2 - small_diameter^2) / large_diameter^2 = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_fraction_of_plates_l1364_136468


namespace NUMINAMATH_CALUDE_subset_iff_positive_l1364_136426

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x : ℝ | x + a > 0}

theorem subset_iff_positive (a : ℝ) : A ⊆ B a ↔ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_positive_l1364_136426


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1364_136462

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) → 
  (10 * x₂^2 + 15 * x₂ - 20 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1364_136462


namespace NUMINAMATH_CALUDE_gcd_count_for_product_360_l1364_136415

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ d, d ∈ S ↔ ∃ (x y : ℕ+), Nat.gcd x y = d ∧ Nat.gcd x y * Nat.lcm x y = 360)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_360_l1364_136415


namespace NUMINAMATH_CALUDE_max_sum_in_S_l1364_136444

/-- The set of ordered pairs of integers (x,y) satisfying x^2 + y^2 = 50 -/
def S : Set (ℤ × ℤ) := {p | p.1^2 + p.2^2 = 50}

/-- The theorem stating that the maximum sum of x+y for (x,y) in S is 10 -/
theorem max_sum_in_S : (⨆ p ∈ S, (p.1 + p.2 : ℤ)) = 10 := by sorry

end NUMINAMATH_CALUDE_max_sum_in_S_l1364_136444
