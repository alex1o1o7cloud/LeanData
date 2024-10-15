import Mathlib

namespace NUMINAMATH_CALUDE_series_convergence_l4065_406541

/-- The infinite sum of the given series converges to 2 -/
theorem series_convergence : 
  ∑' k : ℕ, (8 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l4065_406541


namespace NUMINAMATH_CALUDE_alex_silk_distribution_l4065_406591

/-- The amount of silk each friend receives when Alex distributes his remaining silk -/
def silk_per_friend (total_silk : ℕ) (silk_per_dress : ℕ) (num_dresses : ℕ) (num_friends : ℕ) : ℕ :=
  (total_silk - silk_per_dress * num_dresses) / num_friends

/-- Theorem stating that each friend receives 20 meters of silk -/
theorem alex_silk_distribution :
  silk_per_friend 600 5 100 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_alex_silk_distribution_l4065_406591


namespace NUMINAMATH_CALUDE_find_divisor_l4065_406504

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 13787)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 155 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 155 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l4065_406504


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l4065_406531

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l4065_406531


namespace NUMINAMATH_CALUDE_congruence_product_l4065_406562

theorem congruence_product (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a * c) ≡ (b * d) [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_product_l4065_406562


namespace NUMINAMATH_CALUDE_paityn_red_hats_l4065_406508

/-- Proves that Paityn has 20 red hats given the problem conditions -/
theorem paityn_red_hats :
  ∀ (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ),
  paityn_blue = 24 →
  zola_red = (4 * paityn_red) / 5 →
  zola_blue = 2 * paityn_blue →
  paityn_red + paityn_blue + zola_red + zola_blue = 108 →
  paityn_red = 20 := by
sorry


end NUMINAMATH_CALUDE_paityn_red_hats_l4065_406508


namespace NUMINAMATH_CALUDE_car_speed_problem_l4065_406595

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) 
  (h1 : speed_second_hour = 55)
  (h2 : average_speed = 72.5) : 
  ∃ speed_first_hour : ℝ, 
    speed_first_hour = 90 ∧ 
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4065_406595


namespace NUMINAMATH_CALUDE_number_line_relations_l4065_406580

theorem number_line_relations (a b : ℝ) (h1 : 1/2 < a) (h2 : a < 1) (h3 : 1/2 < b) (h4 : b < 1) :
  (1 < a + b ∧ a + b < 2) ∧ (a - b < 0) ∧ (1/4 < a * b ∧ a * b < 1) := by
  sorry

end NUMINAMATH_CALUDE_number_line_relations_l4065_406580


namespace NUMINAMATH_CALUDE_geometric_sequence_complex_l4065_406561

def z₁ (a : ℝ) : ℂ := a + Complex.I
def z₂ (a : ℝ) : ℂ := 2*a + 2*Complex.I
def z₃ (a : ℝ) : ℂ := 3*a + 4*Complex.I

theorem geometric_sequence_complex (a : ℝ) :
  (Complex.abs (z₂ a))^2 = (Complex.abs (z₁ a)) * (Complex.abs (z₃ a)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_complex_l4065_406561


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l4065_406542

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_12th_term 
  (seq : ArithmeticSequence) 
  (sum7 : sum_n seq 7 = 7)
  (term79 : seq.a 7 + seq.a 9 = 16) : 
  seq.a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l4065_406542


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l4065_406560

theorem cubic_sum_theorem (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^3 + y^3 = 217 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l4065_406560


namespace NUMINAMATH_CALUDE_negation_of_proposition_l4065_406500

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4065_406500


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_negation_and_disjunction_correct_propositions_l4065_406557

-- Proposition ②
theorem contrapositive_real_roots (q : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1 :=
sorry

-- Proposition ③
theorem negation_and_disjunction (p q : Prop) :
  ¬p ∧ (p ∨ q) → q :=
sorry

-- Main theorem combining both propositions
theorem correct_propositions :
  (∃ q : ℝ, (∀ x : ℝ, x^2 + 2*x + q ≠ 0) → q > 1) ∧
  (∀ p q : Prop, ¬p ∧ (p ∨ q) → q) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_negation_and_disjunction_correct_propositions_l4065_406557


namespace NUMINAMATH_CALUDE_second_to_last_digit_is_five_l4065_406513

def is_power_of_prime (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

theorem second_to_last_digit_is_five (N : ℕ) 
  (h1 : N % 10 = 0) 
  (h2 : ∃ d : ℕ, d < N ∧ d ∣ N ∧ is_power_of_prime d ∧ ∀ m : ℕ, m < N → m ∣ N → m ≤ d)
  (h3 : N > 10) :
  (N / 10) % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_is_five_l4065_406513


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_exists_l4065_406570

theorem polynomial_coefficient_sum_exists : ∃ (a b c d : ℤ),
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 3*x^2 + 12*x - 8) ∧
  (∃ (s : ℤ), s = a + b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_exists_l4065_406570


namespace NUMINAMATH_CALUDE_max_divisor_of_prime_sum_l4065_406581

theorem max_divisor_of_prime_sum (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a > 3 → b > 3 → c > 3 →
  2 * a + 5 * b = c →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ n) →
  (∃ (n : ℕ), n ∣ (a + b + c) ∧ ∀ (m : ℕ), m ∣ (a + b + c) → m ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_max_divisor_of_prime_sum_l4065_406581


namespace NUMINAMATH_CALUDE_max_volume_of_prism_l4065_406596

/-- A right prism with a rectangular base -/
structure RectPrism where
  height : ℝ
  base_length : ℝ
  base_width : ℝ

/-- The surface area constraint for the prism -/
def surface_area_constraint (p : RectPrism) : Prop :=
  p.height * p.base_length + p.height * p.base_width + p.base_length * p.base_width = 36

/-- The constraint that base sides are twice the height -/
def base_height_constraint (p : RectPrism) : Prop :=
  p.base_length = 2 * p.height ∧ p.base_width = 2 * p.height

/-- The volume of the prism -/
def volume (p : RectPrism) : ℝ :=
  p.height * p.base_length * p.base_width

/-- The theorem stating the maximum volume of the prism -/
theorem max_volume_of_prism :
  ∃ (p : RectPrism), surface_area_constraint p ∧ base_height_constraint p ∧
    (∀ (q : RectPrism), surface_area_constraint q → base_height_constraint q →
      volume q ≤ volume p) ∧
    volume p = 27 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_max_volume_of_prism_l4065_406596


namespace NUMINAMATH_CALUDE_continuity_at_two_l4065_406588

/-- The function f(x) = -2x^2 - 5 is continuous at x₀ = 2 -/
theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |(-2 * x^2 - 5) - (-2 * 2^2 - 5)| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_two_l4065_406588


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l4065_406583

/-- The focal length of a hyperbola with equation x²/m - y² = 1 (m > 0) 
    and asymptote √3x + my = 0 is 4 -/
theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / m - y^2 = 1
  let asymptote : ℝ → ℝ → Prop := λ x y => Real.sqrt 3 * x + m * y = 0
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧
    (∀ x y, C x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x y, asymptote x y ↔ y = -(Real.sqrt 3 / m) * x) ∧
    c^2 = a^2 + b^2 ∧
    2 * c = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l4065_406583


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l4065_406516

theorem consecutive_integers_average (n m : ℤ) : 
  (n > 0) →
  (m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) →
  ((m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 = n + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l4065_406516


namespace NUMINAMATH_CALUDE_lines_are_parallel_l4065_406577

/-- Two lines are parallel if they have the same slope -/
def parallel (m₁ b₁ m₂ b₂ : ℝ) : Prop := m₁ = m₂

/-- The first line: y = -2x + 1 -/
def line1 (x : ℝ) : ℝ := -2 * x + 1

/-- The second line: y = -2x + 3 -/
def line2 (x : ℝ) : ℝ := -2 * x + 3

/-- Theorem: The line y = -2x + 1 is parallel to the line y = -2x + 3 -/
theorem lines_are_parallel : parallel (-2) 1 (-2) 3 := by sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l4065_406577


namespace NUMINAMATH_CALUDE_curve_self_intersection_l4065_406518

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^3 - 3*t + 1

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^4 - 4*t^2 + 4

/-- The curve crosses itself at (1, 1) -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 1 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l4065_406518


namespace NUMINAMATH_CALUDE_shaded_probability_l4065_406551

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  total_count : Nat

/-- The probability of selecting a shaded triangle -/
def probability_shaded (d : Diagram) : ℚ :=
  d.shaded_count / d.total_count

theorem shaded_probability (d : Diagram) 
  (h1 : d.total_count > 4)
  (h2 : d.shaded_count = d.total_count / 2)
  (h3 : d.shaded_count = (d.triangles.filter Triangle.shaded).length)
  (h4 : d.total_count = d.triangles.length) :
  probability_shaded d = 1 / 2 := by
  sorry

#check shaded_probability

end NUMINAMATH_CALUDE_shaded_probability_l4065_406551


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l4065_406556

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α : Plane) (h : subset n α) :
  (perp_line m n → perp_plane m α) ∧ 
  ¬(perp_line m n ↔ perp_plane m α) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l4065_406556


namespace NUMINAMATH_CALUDE_circle_B_radius_l4065_406582

/-- The configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfig where
  /-- Radius of circle A -/
  radius_A : ℝ
  /-- Radius of circle B -/
  radius_B : ℝ
  /-- Radius of circle C -/
  radius_C : ℝ
  /-- Radius of circle D -/
  radius_D : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externally_tangent : radius_A + radius_B + radius_C = radius_D
  /-- Circles B and C are congruent -/
  B_C_congruent : radius_B = radius_C
  /-- Circle A passes through the center of D -/
  A_through_D_center : radius_A = radius_D / 2
  /-- Circle A has a radius of 2 -/
  A_radius_2 : radius_A = 2

/-- The main theorem stating that given the circle configuration, the radius of circle B is approximately 0.923 -/
theorem circle_B_radius (config : CircleConfig) : 
  0.922 < config.radius_B ∧ config.radius_B < 0.924 := by
  sorry


end NUMINAMATH_CALUDE_circle_B_radius_l4065_406582


namespace NUMINAMATH_CALUDE_base_8_digit_count_l4065_406533

/-- The count of numbers among the first 512 positive integers in base 8 
    that contain either 5 or 6 -/
def count_with_5_or_6 : ℕ := 296

/-- The count of numbers among the first 512 positive integers in base 8 
    that don't contain 5 or 6 -/
def count_without_5_or_6 : ℕ := 6^3

/-- The total count of numbers considered -/
def total_count : ℕ := 512

theorem base_8_digit_count : 
  count_with_5_or_6 = total_count - count_without_5_or_6 := by sorry

end NUMINAMATH_CALUDE_base_8_digit_count_l4065_406533


namespace NUMINAMATH_CALUDE_steve_pie_difference_l4065_406526

/-- The number of days Steve bakes apple pies in a week -/
def apple_pie_days : ℕ := 3

/-- The number of days Steve bakes cherry pies in a week -/
def cherry_pie_days : ℕ := 2

/-- The number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The number of apple pies Steve bakes in a week -/
def apple_pies_per_week : ℕ := apple_pie_days * pies_per_day

/-- The number of cherry pies Steve bakes in a week -/
def cherry_pies_per_week : ℕ := cherry_pie_days * pies_per_day

theorem steve_pie_difference : apple_pies_per_week - cherry_pies_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_difference_l4065_406526


namespace NUMINAMATH_CALUDE_resistor_value_l4065_406594

/-- Given two identical resistors connected in series to a DC voltage source,
    if the voltage across one resistor is 2 V and the current through the circuit is 4 A,
    then the resistance of each resistor is 0.5 Ω. -/
theorem resistor_value (R₀ : ℝ) (U V I : ℝ) : 
  U = 2 → -- Voltage across one resistor
  V = 2 * U → -- Total voltage
  I = 4 → -- Current through the circuit
  V = I * (2 * R₀) → -- Ohm's law
  R₀ = 0.5 := by
  sorry

#check resistor_value

end NUMINAMATH_CALUDE_resistor_value_l4065_406594


namespace NUMINAMATH_CALUDE_marbles_cost_l4065_406509

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : total_spent - (football_cost + baseball_cost) = 9.05 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_l4065_406509


namespace NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l4065_406584

theorem mayoral_election_vote_ratio :
  let votes_Z : ℕ := 25000
  let votes_X : ℕ := 22500
  let votes_Y : ℕ := 2 * votes_X / 3
  let fewer_votes : ℕ := votes_Z - votes_Y
  (fewer_votes : ℚ) / votes_Z = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_mayoral_election_vote_ratio_l4065_406584


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4065_406550

theorem complex_fraction_equality : 
  let a := 3 + 1/3 + 2.5
  let b := 2.5 - (1 + 1/3)
  let c := 4.6 - (2 + 1/3)
  let d := 4.6 + (2 + 1/3)
  let e := 5.2
  let f := 0.05 / (1/7 - 0.125) + 5.7
  (a / b * c / d * e) / f = 5/34 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4065_406550


namespace NUMINAMATH_CALUDE_average_age_first_fifth_dog_l4065_406590

/-- The age of the nth fastest dog -/
def dog_age (n : ℕ) : ℕ :=
  match n with
  | 1 => 10
  | 2 => dog_age 1 - 2
  | 3 => dog_age 2 + 4
  | 4 => dog_age 3 / 2
  | 5 => dog_age 4 + 20
  | _ => 0

theorem average_age_first_fifth_dog :
  (dog_age 1 + dog_age 5) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_age_first_fifth_dog_l4065_406590


namespace NUMINAMATH_CALUDE_solve_chicken_problem_l4065_406576

/-- Represents the problem of calculating the number of chickens sold -/
def chicken_problem (selling_price feed_cost feed_weight feed_per_chicken total_profit : ℚ) : Prop :=
  let cost_per_chicken := (feed_per_chicken / feed_weight) * feed_cost
  let profit_per_chicken := selling_price - cost_per_chicken
  let num_chickens := total_profit / profit_per_chicken
  num_chickens = 50

/-- Theorem stating the solution to the chicken problem -/
theorem solve_chicken_problem :
  chicken_problem 1.5 2 20 2 65 := by
  sorry

#check solve_chicken_problem

end NUMINAMATH_CALUDE_solve_chicken_problem_l4065_406576


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l4065_406535

theorem cubic_expression_equality : 103^3 - 3 * 103^2 + 3 * 103 - 1 = 1061208 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l4065_406535


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l4065_406501

/-- The perimeter of a rhombus with diagonals 18 and 12 is 12√13 -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 12) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 12 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l4065_406501


namespace NUMINAMATH_CALUDE_meeting_point_2015_l4065_406536

/-- Represents a point on a line segment --/
structure Point where
  position : ℝ
  deriving Inhabited

/-- Represents an object moving on a line segment --/
structure MovingObject where
  startPoint : Point
  speed : ℝ
  startTime : ℝ
  deriving Inhabited

/-- Calculates the meeting point of two objects --/
def meetingPoint (obj1 obj2 : MovingObject) : Point :=
  sorry

/-- Theorem: The 2015th meeting point is the same as the 1st meeting point --/
theorem meeting_point_2015 (A B : Point) (obj1 obj2 : MovingObject) :
  obj1.startPoint = A ∧ obj2.startPoint = B →
  obj1.speed > 0 ∧ obj2.speed > 0 →
  meetingPoint obj1 obj2 = meetingPoint obj1 obj2 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_2015_l4065_406536


namespace NUMINAMATH_CALUDE_fraction_1991_1949_position_l4065_406589

/-- Represents a fraction in the table -/
structure Fraction where
  numerator : ℕ
  denominator : ℕ

/-- Represents a row in the table -/
def Row := List Fraction

/-- Generates a row of the table given its index -/
def generateRow (n : ℕ) : Row :=
  sorry

/-- Checks if a fraction appears in a given row -/
def appearsInRow (f : Fraction) (r : Row) : Prop :=
  sorry

/-- The row number where 1991/1949 appears -/
def targetRow : ℕ := 3939

/-- The position of 1991/1949 in its row -/
def targetPosition : ℕ := 1949

theorem fraction_1991_1949_position : 
  let f := Fraction.mk 1991 1949
  let r := generateRow targetRow
  appearsInRow f r ∧ 
  (∃ (l1 l2 : List Fraction), r = l1 ++ [f] ++ l2 ∧ l1.length = targetPosition - 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_1991_1949_position_l4065_406589


namespace NUMINAMATH_CALUDE_intersection_point_P_equation_l4065_406505

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3
def C₂ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2

-- Theorem for the intersection point
theorem intersection_point :
  ∃ ρ θ, C₁ ρ θ ∧ C₂ ρ θ ∧ ρ = 2 * Real.sqrt 3 ∧ θ = Real.pi / 6 :=
sorry

-- Define the relationship between Q and P
def Q_P_relation (ρ_Q θ_Q ρ_P θ_P : ℝ) : Prop :=
  C₂ ρ_Q θ_Q ∧ ρ_Q = (2/3) * ρ_P ∧ θ_Q = θ_P

-- Theorem for the polar coordinate equation of P
theorem P_equation :
  ∀ ρ_P θ_P, (∃ ρ_Q θ_Q, Q_P_relation ρ_Q θ_Q ρ_P θ_P) →
  ρ_P = 10 * Real.cos θ_P ∧ 0 ≤ θ_P ∧ θ_P < Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_P_equation_l4065_406505


namespace NUMINAMATH_CALUDE_smallest_k_for_zero_l4065_406585

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n ≤ M then n + a else n - b

def iterate_f (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterate_f a b M k n)

theorem smallest_k_for_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, (∀ j < k, iterate_f a b M j 0 ≠ 0) ∧ 
            iterate_f a b M k 0 = 0 ∧
            k = (a + b) / Nat.gcd a b :=
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_zero_l4065_406585


namespace NUMINAMATH_CALUDE_exercise_band_resistance_l4065_406597

/-- The resistance added by each exercise band -/
def band_resistance : ℝ := sorry

/-- The number of exercise bands -/
def num_bands : ℕ := 2

/-- The weight of the dumbbell in pounds -/
def dumbbell_weight : ℝ := 10

/-- The total squat weight with both sets of bands doubled and the dumbbell -/
def total_squat_weight : ℝ := 30

/-- Theorem stating that each band adds 10 pounds of resistance -/
theorem exercise_band_resistance :
  band_resistance = 10 :=
by sorry

end NUMINAMATH_CALUDE_exercise_band_resistance_l4065_406597


namespace NUMINAMATH_CALUDE_system_solution_proof_l4065_406563

theorem system_solution_proof :
  ∃ (x y : ℝ), 
    (4 * x + y = 12) ∧ 
    (3 * x - 2 * y = -2) ∧ 
    (x = 2) ∧ 
    (y = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l4065_406563


namespace NUMINAMATH_CALUDE_boys_pass_percentage_l4065_406546

/-- Proves that 28% of boys passed the examination given the problem conditions -/
theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 702 / 1000 →
  let boys := total_candidates - girls
  let total_pass := total_candidates * (1 - total_fail_rate)
  let girls_pass := girls * girls_pass_rate
  let boys_pass := total_pass - girls_pass
  (boys_pass / boys : ℚ) = 28 / 100 := by sorry

end NUMINAMATH_CALUDE_boys_pass_percentage_l4065_406546


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l4065_406529

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 3

-- State the theorem
theorem remainder_theorem (p : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + p a := by sorry

-- State the problem
theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x + 27 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l4065_406529


namespace NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l4065_406575

/-- A complex number z is on the negative y-axis if its real part is 0 and its imaginary part is negative -/
def on_negative_y_axis (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im < 0

/-- The main theorem -/
theorem complex_square_on_negative_y_axis (a : ℝ) :
  on_negative_y_axis ((a + Complex.I) ^ 2) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l4065_406575


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l4065_406522

def isPythagoreanTriple (a b c : ℕ) : Prop := a^2 + b^2 = c^2

def isPrimitivePythagoreanTriple (a b c : ℕ) : Prop :=
  isPythagoreanTriple a b c ∧ Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1

theorem pythagorean_triple_properties
  (a b c : ℕ) (h : isPrimitivePythagoreanTriple a b c) :
  (Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) ∧
  (a % 4 = 0 ∨ b % 4 = 0) ∧
  (a % 3 = 0 ∨ b % 3 = 0) ∧
  (a % 5 = 0 ∨ b % 5 = 0 ∨ c % 5 = 0) ∧
  (∃ k : ℕ, c = 4*k + 1 ∧ c % 3 ≠ 0 ∧ c % 7 ≠ 0 ∧ c % 11 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l4065_406522


namespace NUMINAMATH_CALUDE_mrs_jane_total_coins_l4065_406599

def total_coins (jayden_coins jason_coins : ℕ) : ℕ :=
  jayden_coins + jason_coins

theorem mrs_jane_total_coins : 
  let jayden_coins : ℕ := 300
  let jason_coins : ℕ := jayden_coins + 60
  total_coins jayden_coins jason_coins = 660 := by
  sorry

end NUMINAMATH_CALUDE_mrs_jane_total_coins_l4065_406599


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l4065_406573

/-- Given a hemisphere with base area 144π and a cylindrical extension of height 10 units
    with the same radius as the hemisphere, the total surface area of the combined object is 672π. -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : ℝ) :
  r^2 * Real.pi = 144 * Real.pi →
  h = 10 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h + Real.pi * r^2 = 672 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l4065_406573


namespace NUMINAMATH_CALUDE_floor_painting_overlap_l4065_406539

theorem floor_painting_overlap (red green blue : ℝ) 
  (h_red : red = 0.75) 
  (h_green : green = 0.7) 
  (h_blue : blue = 0.65) : 
  1 - (1 - red + 1 - green + 1 - blue) ≥ 0.1 := by sorry

end NUMINAMATH_CALUDE_floor_painting_overlap_l4065_406539


namespace NUMINAMATH_CALUDE_fraction_equality_l4065_406507

theorem fraction_equality (a b : ℝ) (h : a ≠ b) : (-a + b) / (a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4065_406507


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4065_406537

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 4)^2 + (y + 3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l4065_406537


namespace NUMINAMATH_CALUDE_trailer_homes_calculation_l4065_406558

/-- The number of new trailer homes added to Maple Drive -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes on Maple Drive -/
def initial_homes : ℕ := 25

/-- The number of years that have passed -/
def years_passed : ℕ := 3

/-- The initial average age of the trailer homes -/
def initial_avg_age : ℚ := 15

/-- The current average age of all trailer homes -/
def current_avg_age : ℚ := 12

theorem trailer_homes_calculation :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end NUMINAMATH_CALUDE_trailer_homes_calculation_l4065_406558


namespace NUMINAMATH_CALUDE_sum_zero_implies_opposites_l4065_406566

theorem sum_zero_implies_opposites (a b : ℝ) : a + b = 0 → a = -b := by sorry

end NUMINAMATH_CALUDE_sum_zero_implies_opposites_l4065_406566


namespace NUMINAMATH_CALUDE_trailing_zeroes_of_sum_factorials_l4065_406544

/-- The number of trailing zeroes in a natural number -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of trailing zeroes in 70! + 140! is 16 -/
theorem trailing_zeroes_of_sum_factorials :
  trailingZeroes (factorial 70 + factorial 140) = 16 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_of_sum_factorials_l4065_406544


namespace NUMINAMATH_CALUDE_triangle_theorem_l4065_406569

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 2 ∧
  (Real.cos t.C) * t.a * t.b = 1 ∧
  1/2 * t.a * t.b * (Real.sin t.C) = 1/2 ∧
  (Real.sin t.A) * (Real.cos t.A) = Real.sqrt 3 / 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 4 ∧ (t.c = Real.sqrt 6 ∨ t.c = 2 * Real.sqrt 6 / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l4065_406569


namespace NUMINAMATH_CALUDE_eliza_cookies_l4065_406586

theorem eliza_cookies (x : ℚ) 
  (h1 : x + 3*x + 4*(3*x) + 6*(4*(3*x)) = 234) : x = 117/44 := by
  sorry

end NUMINAMATH_CALUDE_eliza_cookies_l4065_406586


namespace NUMINAMATH_CALUDE_problem_solution_l4065_406559

theorem problem_solution :
  ∀ (x a b c : ℤ),
    x ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
    ((a * x^4) / b * c)^3 = x^3 →
    a + b + c = 9 →
    ((x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4065_406559


namespace NUMINAMATH_CALUDE_sum_equals_negative_seven_and_half_l4065_406520

/-- Given that p + 2 = q + 3 = r + 4 = s + 5 = t + 6 = p + q + r + s + t + 10,
    prove that p + q + r + s + t = -7.5 -/
theorem sum_equals_negative_seven_and_half
  (p q r s t : ℚ)
  (h : p + 2 = q + 3 ∧ 
       q + 3 = r + 4 ∧ 
       r + 4 = s + 5 ∧ 
       s + 5 = t + 6 ∧ 
       t + 6 = p + q + r + s + t + 10) :
  p + q + r + s + t = -7.5 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_seven_and_half_l4065_406520


namespace NUMINAMATH_CALUDE_book_cost_l4065_406540

theorem book_cost (n₅ n₃ : ℕ) : 
  (n₅ + n₃ > 10) → 
  (n₅ + n₃ < 20) → 
  (5 * n₅ = 3 * n₃) → 
  (5 * n₅ = 30) := by
sorry

end NUMINAMATH_CALUDE_book_cost_l4065_406540


namespace NUMINAMATH_CALUDE_prob_sum_14_four_dice_l4065_406502

/-- The number of faces on a standard die -/
def faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 14

/-- The total number of possible outcomes when rolling four dice -/
def total_outcomes : ℕ := faces ^ num_dice

/-- The number of favorable outcomes (sum of 14) -/
def favorable_outcomes : ℕ := 54

/-- The probability of rolling a sum of 14 with four standard six-faced dice -/
theorem prob_sum_14_four_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 54 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_sum_14_four_dice_l4065_406502


namespace NUMINAMATH_CALUDE_inverse_mod_53_l4065_406564

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 11) : (36⁻¹ : ZMod 53) = 42 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l4065_406564


namespace NUMINAMATH_CALUDE_hydrochloric_acid_solution_l4065_406554

/-- Represents the volume of pure hydrochloric acid needed to be added -/
def x : ℝ := sorry

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 60

/-- The initial concentration of hydrochloric acid as a decimal -/
def initial_concentration : ℝ := 0.10

/-- The target concentration of hydrochloric acid as a decimal -/
def target_concentration : ℝ := 0.15

theorem hydrochloric_acid_solution :
  initial_concentration * initial_volume + x = target_concentration * (initial_volume + x) := by
  sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_solution_l4065_406554


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l4065_406510

/-- Given a shopkeeper sells cloth at a loss, calculate the cost price per meter. -/
theorem shopkeeper_cloth_cost_price
  (total_meters : ℕ)
  (total_selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : total_selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  total_selling_price / total_meters + loss_per_meter = 50 := by
  sorry

#check shopkeeper_cloth_cost_price

end NUMINAMATH_CALUDE_shopkeeper_cloth_cost_price_l4065_406510


namespace NUMINAMATH_CALUDE_max_value_of_sum_l4065_406549

theorem max_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = 1 ∧
    1 / (a₀ + 9 * b₀) + 1 / (9 * a₀ + b₀) = 5 / 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l4065_406549


namespace NUMINAMATH_CALUDE_N_cannot_be_2_7_l4065_406534

def M : Set ℕ := {1, 4, 7}

theorem N_cannot_be_2_7 (N : Set ℕ) (h : M ∪ N = M) : N ≠ {2, 7} := by
  sorry

end NUMINAMATH_CALUDE_N_cannot_be_2_7_l4065_406534


namespace NUMINAMATH_CALUDE_birdhouse_volume_difference_l4065_406543

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem: The difference in volume between Sara's and Jake's birdhouses is 1152 cubic inches -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_volume_difference_l4065_406543


namespace NUMINAMATH_CALUDE_vector_inequalities_l4065_406548

theorem vector_inequalities (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_inequalities_l4065_406548


namespace NUMINAMATH_CALUDE_equation_solution_l4065_406511

theorem equation_solution : ∃! x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4065_406511


namespace NUMINAMATH_CALUDE_range_of_a_l4065_406538

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1 / (x - 2) ≥ 1 → |x - a| < 1) ∧ 
   ∃ x, (|x - a| < 1 ∧ 1 / (x - 2) < 1)) →
  a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4065_406538


namespace NUMINAMATH_CALUDE_half_power_inequality_l4065_406524

theorem half_power_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by sorry

end NUMINAMATH_CALUDE_half_power_inequality_l4065_406524


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4065_406521

theorem quadratic_equation_roots : 
  let equation := fun (x : ℂ) => x^2 + x + 2
  ∃ (r₁ r₂ : ℂ), r₁ = (-1 + Complex.I * Real.sqrt 7) / 2 ∧ 
                  r₂ = (-1 - Complex.I * Real.sqrt 7) / 2 ∧ 
                  equation r₁ = 0 ∧ 
                  equation r₂ = 0 ∧
                  ∀ (x : ℂ), equation x = 0 → x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4065_406521


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l4065_406512

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -2 * x^2

/-- The equation of the axis of symmetry -/
def axis_of_symmetry (y : ℝ) : Prop := y = 1/8

/-- Theorem: The axis of symmetry for the parabola y = -2x^2 is y = 1/8 -/
theorem parabola_axis_of_symmetry :
  ∀ x y : ℝ, parabola_equation x y → axis_of_symmetry y := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l4065_406512


namespace NUMINAMATH_CALUDE_distinct_domino_arrangements_l4065_406503

/-- Represents a grid with width and height -/
structure Grid :=
  (width : Nat)
  (height : Nat)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Calculates the number of distinct paths on a grid using a given number of dominoes -/
def countDistinctPaths (g : Grid) (d : Domino) (numDominoes : Nat) : Nat :=
  Nat.choose (g.width + g.height - 2) (g.width - 1)

/-- Theorem: The number of distinct domino arrangements on a 6x5 grid with 5 dominoes is 126 -/
theorem distinct_domino_arrangements :
  let g : Grid := ⟨6, 5⟩
  let d : Domino := ⟨2, 1⟩
  countDistinctPaths g d 5 = 126 := by
  sorry

#eval countDistinctPaths ⟨6, 5⟩ ⟨2, 1⟩ 5

end NUMINAMATH_CALUDE_distinct_domino_arrangements_l4065_406503


namespace NUMINAMATH_CALUDE_max_elements_sum_to_target_l4065_406587

/-- The sequence of consecutive odd numbers from 1 to 101 -/
def oddSequence : List Nat := List.range 51 |>.map (fun n => 2 * n + 1)

/-- The sum of selected numbers should be 2013 -/
def targetSum : Nat := 2013

/-- The maximum number of elements that can be selected -/
def maxElements : Nat := 43

theorem max_elements_sum_to_target :
  ∃ (selected : List Nat),
    selected.length = maxElements ∧
    selected.all (· ∈ oddSequence) ∧
    selected.sum = targetSum ∧
    ∀ (other : List Nat),
      other.all (· ∈ oddSequence) →
      other.sum = targetSum →
      other.length ≤ maxElements :=
sorry

end NUMINAMATH_CALUDE_max_elements_sum_to_target_l4065_406587


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4065_406571

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + 4 * a 7 + a 12 = 96

/-- Theorem stating the relationship in the arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 
  2 * seq.a 3 + seq.a 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4065_406571


namespace NUMINAMATH_CALUDE_prove_sales_tax_percentage_l4065_406523

def total_spent : ℝ := 184.80
def tip_percentage : ℝ := 20
def food_price : ℝ := 140

def sales_tax_percentage : ℝ := 10

theorem prove_sales_tax_percentage :
  let price_with_tax := food_price * (1 + sales_tax_percentage / 100)
  let total_with_tip := price_with_tax * (1 + tip_percentage / 100)
  total_with_tip = total_spent :=
by sorry

end NUMINAMATH_CALUDE_prove_sales_tax_percentage_l4065_406523


namespace NUMINAMATH_CALUDE_total_age_reaches_target_in_10_years_l4065_406519

/-- Represents the number of years between each sibling's birth -/
def age_gap : ℕ := 5

/-- Represents the current age of the eldest sibling -/
def eldest_current_age : ℕ := 20

/-- Represents the target total age of all siblings -/
def target_total_age : ℕ := 75

/-- Calculates the total age of the siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_current_age + years) + 
  (eldest_current_age - age_gap + years) + 
  (eldest_current_age - 2 * age_gap + years)

/-- Theorem stating that it takes 10 years for the total age to reach the target -/
theorem total_age_reaches_target_in_10_years : 
  total_age_after 10 = target_total_age :=
sorry

end NUMINAMATH_CALUDE_total_age_reaches_target_in_10_years_l4065_406519


namespace NUMINAMATH_CALUDE_triangle_area_equivalence_l4065_406532

/-- Given a triangle with angles α, β, γ, side length a opposite to angle α,
    and circumradius R, prove that the two expressions for the area S are equivalent. -/
theorem triangle_area_equivalence (α β γ a R : ℝ) (h_angles : α + β + γ = π)
    (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < a ∧ 0 < R) :
  (a^2 * Real.sin β * Real.sin γ) / (2 * Real.sin α) =
  2 * R^2 * Real.sin α * Real.sin β * Real.sin γ := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equivalence_l4065_406532


namespace NUMINAMATH_CALUDE_tree_climbing_average_height_l4065_406568

theorem tree_climbing_average_height : 
  let first_tree_height : ℝ := 1000
  let second_tree_height : ℝ := first_tree_height / 2
  let third_tree_height : ℝ := first_tree_height / 2
  let fourth_tree_height : ℝ := first_tree_height + 200
  let total_height : ℝ := first_tree_height + second_tree_height + third_tree_height + fourth_tree_height
  let num_trees : ℝ := 4
  (total_height / num_trees) = 800 := by sorry

end NUMINAMATH_CALUDE_tree_climbing_average_height_l4065_406568


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l4065_406506

/-- Represents the systematic sampling of students -/
def systematic_sampling (total_students : ℕ) (students_to_select : ℕ) : List ℕ :=
  sorry

/-- The theorem stating the correct systematic sampling for the given problem -/
theorem correct_systematic_sampling :
  systematic_sampling 50 5 = [5, 15, 25, 35, 45] := by
  sorry

end NUMINAMATH_CALUDE_correct_systematic_sampling_l4065_406506


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4065_406592

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  ∃ x y, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4065_406592


namespace NUMINAMATH_CALUDE_g_triple_equality_l4065_406598

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 3 * x - 50

theorem g_triple_equality (a : ℝ) :
  a < 0 → (g (g (g 15)) = g (g (g a)) ↔ a = -55 / 3) := by
  sorry

end NUMINAMATH_CALUDE_g_triple_equality_l4065_406598


namespace NUMINAMATH_CALUDE_three_common_points_l4065_406547

def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0

def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

def is_common_point (x y : ℝ) : Prop := equation1 x y ∧ equation2 x y

def distinct_points (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

theorem three_common_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_common_point p1.1 p1.2 ∧
    is_common_point p2.1 p2.2 ∧
    is_common_point p3.1 p3.2 ∧
    distinct_points p1 p2 ∧
    distinct_points p1 p3 ∧
    distinct_points p2 p3 ∧
    (∀ (p : ℝ × ℝ), is_common_point p.1 p.2 → p = p1 ∨ p = p2 ∨ p = p3) :=
by sorry

end NUMINAMATH_CALUDE_three_common_points_l4065_406547


namespace NUMINAMATH_CALUDE_inequality_range_l4065_406545

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l4065_406545


namespace NUMINAMATH_CALUDE_smallest_angle_for_sin_polar_graph_l4065_406530

def completes_intrinsic_pattern (t : Real) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ r, r = Real.sin θ ∧ 
  (∀ ϕ, ϕ > t → ∃ ψ, 0 ≤ ψ ∧ ψ ≤ t ∧ Real.sin ϕ = Real.sin ψ)

theorem smallest_angle_for_sin_polar_graph :
  (∀ t < 2 * Real.pi, ¬ completes_intrinsic_pattern t) ∧
  completes_intrinsic_pattern (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_for_sin_polar_graph_l4065_406530


namespace NUMINAMATH_CALUDE_store_opening_cost_l4065_406553

/-- The cost to open Kim's store -/
def openingCost (monthlyRevenue : ℕ) (monthlyExpenses : ℕ) (monthsToPayback : ℕ) : ℕ :=
  (monthlyRevenue - monthlyExpenses) * monthsToPayback

/-- Theorem stating the cost to open Kim's store -/
theorem store_opening_cost : openingCost 4000 1500 10 = 25000 := by
  sorry

end NUMINAMATH_CALUDE_store_opening_cost_l4065_406553


namespace NUMINAMATH_CALUDE_slope_intercept_sum_horizontal_line_l4065_406565

/-- Given two points with the same y-coordinate and different x-coordinates,
    the sum of the slope and y-intercept of the line containing both points is 20. -/
theorem slope_intercept_sum_horizontal_line (C D : ℝ × ℝ) :
  C.2 = 20 →
  D.2 = 20 →
  C.1 ≠ D.1 →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_horizontal_line_l4065_406565


namespace NUMINAMATH_CALUDE_solution_comparison_l4065_406515

theorem solution_comparison (c d p q : ℝ) (hc : c ≠ 0) (hp : p ≠ 0) :
  -d / c < -q / p ↔ q / p < d / c := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l4065_406515


namespace NUMINAMATH_CALUDE_emily_remaining_toys_l4065_406525

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

theorem emily_remaining_toys : remaining_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_remaining_toys_l4065_406525


namespace NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l4065_406567

/-- Given a quadratic function f(x) = ax² + bx + c with a > b > c and a + b + c = 0,
    prove that f(x) < 0 for all x in the open interval (0, 1). -/
theorem quadratic_negative_on_unit_interval
  (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, x ∈ Set.Ioo 0 1 → a * x^2 + b * x + c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_negative_on_unit_interval_l4065_406567


namespace NUMINAMATH_CALUDE_orange_beads_count_l4065_406552

/-- Represents the number of beads of each color in a necklace -/
structure NecklaceComposition where
  green : ℕ
  white : ℕ
  orange : ℕ

/-- Represents the total number of beads available for each color -/
def TotalBeads : ℕ := 45

/-- The composition of beads in each necklace -/
def necklace : NecklaceComposition := {
  green := 9,
  white := 6,
  orange := 9  -- This is what we want to prove
}

/-- The maximum number of necklaces that can be made -/
def maxNecklaces : ℕ := 5

theorem orange_beads_count :
  necklace.orange = 9 ∧
  necklace.green * maxNecklaces = TotalBeads ∧
  necklace.white * maxNecklaces ≤ TotalBeads ∧
  necklace.orange * maxNecklaces = TotalBeads :=
by sorry

end NUMINAMATH_CALUDE_orange_beads_count_l4065_406552


namespace NUMINAMATH_CALUDE_hcf_problem_l4065_406517

theorem hcf_problem (a b : ℕ+) (h1 : max a b = 414) 
  (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 18) : Nat.gcd a b = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l4065_406517


namespace NUMINAMATH_CALUDE_abs_half_minus_three_eighths_i_l4065_406579

theorem abs_half_minus_three_eighths_i : Complex.abs (1/2 - 3/8 * Complex.I) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_abs_half_minus_three_eighths_i_l4065_406579


namespace NUMINAMATH_CALUDE_total_profit_calculation_l4065_406527

/-- Prove that the total profit is 60000 given the investment ratios and C's profit share -/
theorem total_profit_calculation (a b c : ℕ) (total_profit : ℕ) : 
  a * 2 = c * 3 →  -- A and C invested in ratio 3:2
  a = b * 3 →      -- A and B invested in ratio 3:1
  c * total_profit = 20000 * (a + b + c) →  -- C's profit share
  total_profit = 60000 := by
  sorry

#check total_profit_calculation

end NUMINAMATH_CALUDE_total_profit_calculation_l4065_406527


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4065_406528

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 3 * x = 5) : y = 3 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l4065_406528


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangency_l4065_406574

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/4 = 1

-- Define the latus rectum of the parabola
def latus_rectum (p : ℝ) (y : ℝ) : Prop := y = -p/2

-- Theorem statement
theorem parabola_ellipse_tangency :
  ∃ (p : ℝ), ∃ (x y : ℝ),
    parabola p x y ∧
    ellipse x y ∧
    latus_rectum p y ∧
    p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangency_l4065_406574


namespace NUMINAMATH_CALUDE_at_least_one_subgraph_not_planar_l4065_406514

/-- A complete graph with 11 vertices where each edge is colored either red or blue. -/
def CompleteGraph11 : Type := Unit

/-- The red subgraph of the complete graph. -/
def RedSubgraph (G : CompleteGraph11) : Type := Unit

/-- The blue subgraph of the complete graph. -/
def BlueSubgraph (G : CompleteGraph11) : Type := Unit

/-- Predicate to check if a graph is planar. -/
def IsPlanar (G : Type) : Prop := sorry

/-- Theorem stating that at least one of the monochromatic subgraphs is not planar. -/
theorem at_least_one_subgraph_not_planar (G : CompleteGraph11) : 
  ¬(IsPlanar (RedSubgraph G) ∧ IsPlanar (BlueSubgraph G)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_subgraph_not_planar_l4065_406514


namespace NUMINAMATH_CALUDE_rational_product_sum_implies_negative_l4065_406578

theorem rational_product_sum_implies_negative (a b : ℚ) 
  (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_product_sum_implies_negative_l4065_406578


namespace NUMINAMATH_CALUDE_area_difference_l4065_406555

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem area_difference (sheet1_length sheet1_width sheet2_length sheet2_width : ℝ) 
  (h1 : sheet1_length = 11) 
  (h2 : sheet1_width = 13) 
  (h3 : sheet2_length = 6.5) 
  (h4 : sheet2_width = 11) : 
  2 * (sheet1_length * sheet1_width) - 2 * (sheet2_length * sheet2_width) = 143 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_l4065_406555


namespace NUMINAMATH_CALUDE_number_puzzle_l4065_406593

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ x / 5 + 7 = x / 4 - 7 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l4065_406593


namespace NUMINAMATH_CALUDE_min_coins_is_four_l4065_406572

/-- The minimum number of coins Ana can have -/
def min_coins : ℕ :=
  let initial_coins := 22
  let operations := [6, 18, -12]
  sorry

/-- Theorem: The minimum number of coins Ana can have is 4 -/
theorem min_coins_is_four : min_coins = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_is_four_l4065_406572
