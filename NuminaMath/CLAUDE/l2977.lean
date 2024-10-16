import Mathlib

namespace NUMINAMATH_CALUDE_rotate_180_equals_optionC_l2977_297793

/-- Represents a geometric shape --/
structure Shape :=
  (id : ℕ)

/-- Represents a rotation operation --/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  { id := s.id }

/-- The original T-like shape --/
def original : Shape :=
  { id := 0 }

/-- Option C from the problem --/
def optionC : Shape :=
  { id := 1 }

/-- Theorem stating that rotating the original shape 180 degrees results in option C --/
theorem rotate_180_equals_optionC : 
  rotate original 180 = optionC := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_equals_optionC_l2977_297793


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2977_297738

theorem fraction_sum_equality : 
  let a : ℕ := 1
  let b : ℕ := 6
  let c : ℕ := 7
  let d : ℕ := 3
  let e : ℕ := 5
  let f : ℕ := 2
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) →
  (Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1) →
  (a : ℚ) / b + (c : ℚ) / d = (e : ℚ) / f :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2977_297738


namespace NUMINAMATH_CALUDE_train_length_proof_l2977_297799

/-- Given a train with a speed of 40 km/hr that crosses a post in 18 seconds,
    prove that its length is approximately 200 meters. -/
theorem train_length_proof (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 18 → -- time in seconds
  length = speed * (1000 / 3600) * time →
  ∃ ε > 0, |length - 200| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2977_297799


namespace NUMINAMATH_CALUDE_estimated_weight_not_exact_weight_estimated_weight_is_approximation_l2977_297787

/-- Represents the linear regression model for height and weight --/
structure HeightWeightModel where
  slope : ℝ
  intercept : ℝ

/-- The estimated weight based on the linear regression model --/
def estimated_weight (model : HeightWeightModel) (height : ℝ) : ℝ :=
  model.slope * height + model.intercept

/-- The given linear regression model for the problem --/
def given_model : HeightWeightModel :=
  { slope := 0.85, intercept := -85.71 }

/-- Theorem stating that the estimated weight for a 160cm tall girl is not necessarily her exact weight --/
theorem estimated_weight_not_exact_weight :
  ∃ (actual_weight : ℝ), 
    estimated_weight given_model 160 ≠ actual_weight ∧ 
    actual_weight > 0 := by
  sorry

/-- Theorem stating that the estimated weight is just an approximation --/
theorem estimated_weight_is_approximation (height : ℝ) :
  ∃ (ε : ℝ), ε > 0 ∧ 
    ∀ (actual_weight : ℝ), 
      actual_weight > 0 →
      |estimated_weight given_model height - actual_weight| < ε := by
  sorry

end NUMINAMATH_CALUDE_estimated_weight_not_exact_weight_estimated_weight_is_approximation_l2977_297787


namespace NUMINAMATH_CALUDE_function_properties_l2977_297702

/-- Given a function f and a real number a, prove properties of f --/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x + 1) = 3 * a^(x + 1) - 4) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) :
  (∀ x, f x = 3 * a^((x + 1) / 2) - 4) ∧ 
  (f (-1) = -1) ∧
  (a > 1 → ∀ x, f (x - 3/4) ≥ 3 / a^(x^2 / 2) - 4) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2977_297702


namespace NUMINAMATH_CALUDE_playground_area_is_22500_l2977_297712

/-- Represents a rectangular playground --/
structure Playground where
  width : ℝ
  length : ℝ

/-- Properties of the playground --/
def PlaygroundProperties (p : Playground) : Prop :=
  p.length = 2 * p.width + 25 ∧
  2 * (p.length + p.width) = 650

/-- The area of the playground --/
def playgroundArea (p : Playground) : ℝ :=
  p.length * p.width

/-- Theorem: The area of the playground with given properties is 22,500 square feet --/
theorem playground_area_is_22500 :
  ∀ p : Playground, PlaygroundProperties p → playgroundArea p = 22500 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_is_22500_l2977_297712


namespace NUMINAMATH_CALUDE_field_length_l2977_297729

theorem field_length (width : ℝ) (length : ℝ) : 
  width = 13.5 ∧ length = 2 * width - 3 → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_field_length_l2977_297729


namespace NUMINAMATH_CALUDE_output_value_2003_l2977_297701

/-- The annual growth rate of the company's output value -/
def growth_rate : ℝ := 0.10

/-- The initial output value of the company in 2000 (in millions of yuan) -/
def initial_value : ℝ := 10

/-- The number of years between 2000 and 2003 -/
def years : ℕ := 3

/-- The expected output value of the company in 2003 (in millions of yuan) -/
def expected_value : ℝ := 13.31

/-- Theorem stating that the company's output value in 2003 will be 13.31 million yuan -/
theorem output_value_2003 : 
  initial_value * (1 + growth_rate) ^ years = expected_value := by
  sorry

end NUMINAMATH_CALUDE_output_value_2003_l2977_297701


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2977_297786

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2977_297786


namespace NUMINAMATH_CALUDE_division_multiplication_example_l2977_297727

theorem division_multiplication_example : (180 / 6) * 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_example_l2977_297727


namespace NUMINAMATH_CALUDE_volleyball_practice_start_time_l2977_297743

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition of minutes to Time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

theorem volleyball_practice_start_time 
  (start_time : Time) 
  (homework_duration : Nat) 
  (break_duration : Nat) : 
  start_time = { hour := 13, minute := 59 } → 
  homework_duration = 96 → 
  break_duration = 25 → 
  addMinutes (addMinutes start_time homework_duration) break_duration = { hour := 16, minute := 0 } :=
by
  sorry


end NUMINAMATH_CALUDE_volleyball_practice_start_time_l2977_297743


namespace NUMINAMATH_CALUDE_circle_center_x_coordinate_range_l2977_297746

/-- The problem statement as a theorem in Lean 4 -/
theorem circle_center_x_coordinate_range :
  ∀ (O A C M : ℝ × ℝ) (l : ℝ → ℝ) (a : ℝ),
    O = (0, 0) →
    A = (0, 3) →
    (∀ x, l x = x + 1) →
    C.2 = l C.1 →
    C.1 = a →
    ∃ r : ℝ, r = 1 ∧ ∀ p : ℝ × ℝ, (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2 →
      ∃ M : ℝ × ℝ, (M.1 - C.1)^2 + (M.2 - C.2)^2 = r^2 ∧
        (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4 * ((M.1 - O.1)^2 + (M.2 - O.2)^2) →
          -1 - Real.sqrt 7 / 2 ≤ a ∧ a ≤ -1 + Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_x_coordinate_range_l2977_297746


namespace NUMINAMATH_CALUDE_bank_balance_after_five_years_l2977_297754

/-- Calculates the compound interest for a given principal, rate, and time -/
def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents the bank account balance after each year -/
def bankBalance : ℕ → ℝ
  | 0 => 5600
  | 1 => compoundInterest 5600 0.03 1
  | 2 => compoundInterest (bankBalance 1) 0.035 1
  | 3 => compoundInterest (bankBalance 2 + 2000) 0.04 1
  | 4 => compoundInterest (bankBalance 3) 0.045 1
  | 5 => compoundInterest (bankBalance 4) 0.05 1
  | _ => 0  -- For years beyond 5, return 0

theorem bank_balance_after_five_years :
  bankBalance 5 = 9094.20 := by
  sorry


end NUMINAMATH_CALUDE_bank_balance_after_five_years_l2977_297754


namespace NUMINAMATH_CALUDE_expanded_garden_perimeter_l2977_297733

/-- Given a square garden with an area of 49 square meters, if each side is expanded by 4 meters
    to form a new square garden, the perimeter of the new garden is 44 meters. -/
theorem expanded_garden_perimeter : ∀ (original_side : ℝ),
  original_side^2 = 49 →
  (4 * (original_side + 4) = 44) :=
by
  sorry

end NUMINAMATH_CALUDE_expanded_garden_perimeter_l2977_297733


namespace NUMINAMATH_CALUDE_probability_same_length_segments_l2977_297765

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides + diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of diagonals of the first length in a regular hexagon -/
def num_diagonals_length1 : ℕ := 3

/-- The number of diagonals of the second length in a regular hexagon -/
def num_diagonals_length2 : ℕ := 6

/-- The probability of selecting two segments of the same length from a regular hexagon -/
theorem probability_same_length_segments :
  (Nat.choose num_sides 2 + Nat.choose num_diagonals_length1 2 + Nat.choose num_diagonals_length2 2) /
  Nat.choose total_segments 2 = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_length_segments_l2977_297765


namespace NUMINAMATH_CALUDE_line_relationships_exhaustive_l2977_297781

-- Define the possible spatial relationships between lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- Define a line in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as it's not relevant for the statement of the theorem

-- Define the relationship between two lines
def relationshipBetweenLines (l1 l2 : Line3D) : LineRelationship :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem line_relationships_exhaustive (l1 l2 : Line3D) :
  ∃ (r : LineRelationship), relationshipBetweenLines l1 l2 = r :=
sorry

end NUMINAMATH_CALUDE_line_relationships_exhaustive_l2977_297781


namespace NUMINAMATH_CALUDE_total_wheels_is_47_l2977_297783

/-- The total number of wheels in Jordan's neighborhood -/
def total_wheels : ℕ :=
  let jordans_driveway := 
    2 * 4 + -- Two cars with 4 wheels each
    1 +     -- One car has a spare wheel
    3 * 2 + -- Three bikes with 2 wheels each
    1 +     -- One bike missing a rear wheel
    3 +     -- One bike with 2 main wheels and one training wheel
    2 +     -- Trash can with 2 wheels
    3 +     -- Tricycle with 3 wheels
    4 +     -- Wheelchair with 2 main wheels and 2 small front wheels
    4 +     -- Wagon with 4 wheels
    3       -- Pair of old roller skates with 3 wheels (one missing)
  let neighbors_driveway :=
    4 +     -- Pickup truck with 4 wheels
    2 +     -- Boat trailer with 2 wheels
    2 +     -- Motorcycle with 2 wheels
    4       -- ATV with 4 wheels
  jordans_driveway + neighbors_driveway

theorem total_wheels_is_47 : total_wheels = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_47_l2977_297783


namespace NUMINAMATH_CALUDE_fraction_ordering_l2977_297734

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 11 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2977_297734


namespace NUMINAMATH_CALUDE_apples_per_pie_l2977_297782

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  initial_apples = 62 →
  handed_out = 8 →
  num_pies = 6 →
  (initial_apples - handed_out) / num_pies = 9 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l2977_297782


namespace NUMINAMATH_CALUDE_stratified_sampling_third_grade_l2977_297772

/-- Represents the number of students to be sampled from each grade in a stratified sampling -/
structure StratifiedSample where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the stratified sample given the total sample size and the ratio of students in each grade -/
def calculateStratifiedSample (totalSample : ℕ) (ratio1 : ℕ) (ratio2 : ℕ) (ratio3 : ℕ) : StratifiedSample :=
  let totalRatio := ratio1 + ratio2 + ratio3
  { first := (ratio1 * totalSample) / totalRatio,
    second := (ratio2 * totalSample) / totalRatio,
    third := (ratio3 * totalSample) / totalRatio }

theorem stratified_sampling_third_grade 
  (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) 
  (h1 : totalSample = 50)
  (h2 : ratio1 = 3)
  (h3 : ratio2 = 3)
  (h4 : ratio3 = 4) :
  (calculateStratifiedSample totalSample ratio1 ratio2 ratio3).third = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_grade_l2977_297772


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2977_297713

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2977_297713


namespace NUMINAMATH_CALUDE_third_score_calculation_l2977_297731

theorem third_score_calculation (score1 score2 score4 : ℕ) (average : ℚ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  average = 75 →
  ∃ score3 : ℕ, (score1 + score2 + score3 + score4) / 4 = average ∧ score3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_third_score_calculation_l2977_297731


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2977_297724

theorem binomial_coefficient_divisibility (m n : ℕ) (h1 : m > 0) (h2 : n > 1) :
  (∀ k : ℕ, 1 ≤ k ∧ k < m → n ∣ Nat.choose m k) →
  ∃ (p u : ℕ), Prime p ∧ u > 0 ∧ m = p^u ∧ n = p :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l2977_297724


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l2977_297720

theorem simplify_algebraic_expression (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l2977_297720


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2977_297796

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l2977_297796


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2977_297790

/-- Proves that given an original price of $199.99999999999997, a final sale price of $144
    after two successive discounts, where the second discount is 20%,
    the first discount percentage is 10%. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 199.99999999999997)
  (h2 : final_price = 144)
  (h3 : second_discount = 0.2)
  : (original_price - final_price / (1 - second_discount)) / original_price = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2977_297790


namespace NUMINAMATH_CALUDE_descent_problem_l2977_297744

/-- A function that calculates the final elevation after descending --/
def final_elevation (initial_elevation rate_of_descent duration : ℝ) : ℝ :=
  initial_elevation - rate_of_descent * duration

/-- Theorem stating that descending from 400 feet at 10 feet per minute for 5 minutes results in an elevation of 350 feet --/
theorem descent_problem :
  final_elevation 400 10 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_descent_problem_l2977_297744


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2977_297728

theorem polynomial_division_quotient : 
  let dividend := fun (z : ℝ) => 5*z^5 - 3*z^4 + 4*z^3 - 7*z^2 + 2*z - 1
  let divisor := fun (z : ℝ) => 3*z^2 + 4*z + 1
  let quotient := fun (z : ℝ) => (5/3)*z^3 - (29/9)*z^2 + (71/27)*z - 218/81
  ∀ z : ℝ, dividend z = (divisor z) * (quotient z) + (dividend z % divisor z) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2977_297728


namespace NUMINAMATH_CALUDE_shaded_area_is_four_point_five_l2977_297737

/-- The area of a shape composed of a large isosceles right triangle and a crescent (lune) -/
theorem shaded_area_is_four_point_five 
  (large_triangle_leg : ℝ) 
  (semicircle_diameter : ℝ) 
  (π : ℝ) 
  (h1 : large_triangle_leg = 2)
  (h2 : semicircle_diameter = 2)
  (h3 : π = 3) : 
  (1/2 * large_triangle_leg * large_triangle_leg) + 
  ((1/2 * π * (semicircle_diameter/2)^2) - (1/2 * (semicircle_diameter/2) * (semicircle_diameter/2))) = 4.5 := by
  sorry

#check shaded_area_is_four_point_five

end NUMINAMATH_CALUDE_shaded_area_is_four_point_five_l2977_297737


namespace NUMINAMATH_CALUDE_max_value_expression_l2977_297767

theorem max_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq : a + b + c = 3) :
  a + Real.sqrt (a * b) + (a * b * c) ^ (1/4) ≤ 10/3 ∧
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a' + Real.sqrt (a' * b') + (a' * b' * c') ^ (1/4) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2977_297767


namespace NUMINAMATH_CALUDE_equation_solution_l2977_297773

theorem equation_solution : 
  ∀ m n : ℕ, 19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2977_297773


namespace NUMINAMATH_CALUDE_probability_two_slate_rocks_l2977_297779

/-- The probability of selecting two slate rocks from a field with given rock counts -/
theorem probability_two_slate_rocks (slate_count pumice_count granite_count : ℕ) :
  slate_count = 12 →
  pumice_count = 16 →
  granite_count = 8 →
  let total_count := slate_count + pumice_count + granite_count
  (slate_count : ℚ) / total_count * ((slate_count - 1) : ℚ) / (total_count - 1) = 11 / 105 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_slate_rocks_l2977_297779


namespace NUMINAMATH_CALUDE_smallest_winning_m_l2977_297705

/-- Represents the state of a square on the board -/
inductive Color
| White
| Green

/-- Represents the game board -/
def Board := Array Color

/-- Represents a player in the game -/
inductive Player
| Ana
| Banana

/-- Ana's strategy function type -/
def AnaStrategy := Board → Fin 2024 → Bool

/-- Banana's strategy function type -/
def BananaStrategy := Board → Nat → Nat

/-- Simulates a single game with given strategies and m -/
def playGame (m : Nat) (anaStrat : AnaStrategy) (bananaStrat : BananaStrategy) : Bool :=
  sorry

/-- Checks if Ana has a winning strategy for a given m -/
def anaHasWinningStrategy (m : Nat) : Bool :=
  sorry

/-- The main theorem stating the smallest m for which Ana can guarantee winning -/
theorem smallest_winning_m :
  (∀ m : Nat, m < 88 → ¬ anaHasWinningStrategy m) ∧
  anaHasWinningStrategy 88 :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_m_l2977_297705


namespace NUMINAMATH_CALUDE_unique_integer_complex_sixth_power_l2977_297759

def complex_sixth_power_is_integer (n : ℤ) : Prop :=
  ∃ m : ℤ, (n + Complex.I) ^ 6 = m

theorem unique_integer_complex_sixth_power :
  ∃! n : ℤ, complex_sixth_power_is_integer n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_complex_sixth_power_l2977_297759


namespace NUMINAMATH_CALUDE_value_of_expression_l2977_297797

theorem value_of_expression (x : ℝ) (h : x = 5) : 4 * x - 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2977_297797


namespace NUMINAMATH_CALUDE_method_of_continuous_subtraction_equiv_euclid_algorithm_l2977_297707

/-- The Method of Continuous Subtraction as used in ancient Chinese mathematics -/
def methodOfContinuousSubtraction (a b : ℕ) : ℕ :=
  sorry

/-- Euclid's algorithm for finding the greatest common divisor -/
def euclidAlgorithm (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the Method of Continuous Subtraction is equivalent to Euclid's algorithm -/
theorem method_of_continuous_subtraction_equiv_euclid_algorithm :
  ∀ a b : ℕ, methodOfContinuousSubtraction a b = euclidAlgorithm a b :=
sorry

end NUMINAMATH_CALUDE_method_of_continuous_subtraction_equiv_euclid_algorithm_l2977_297707


namespace NUMINAMATH_CALUDE_negation_equivalence_l2977_297708

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2977_297708


namespace NUMINAMATH_CALUDE_quarters_fraction_is_three_fifths_l2977_297768

/-- The total number of quarters Ella has -/
def total_quarters : ℕ := 30

/-- The number of quarters representing states that joined between 1790 and 1809 -/
def quarters_1790_1809 : ℕ := 18

/-- The fraction of quarters representing states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := quarters_1790_1809 / total_quarters

theorem quarters_fraction_is_three_fifths : 
  fraction_1790_1809 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_quarters_fraction_is_three_fifths_l2977_297768


namespace NUMINAMATH_CALUDE_function_properties_l2977_297730

def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

def g (a c x : ℝ) : ℝ := f a c x - 2 * x - 3 + |x - 1|

theorem function_properties :
  ∀ a c : ℕ+,
  f a c 1 = 5 →
  6 < f a c 2 ∧ f a c 2 < 11 →
  (a = 1 ∧ c = 2) ∧
  (∀ x : ℝ, g a c x ≥ -1/4) ∧
  (∃ x : ℝ, g a c x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2977_297730


namespace NUMINAMATH_CALUDE_liam_needed_one_more_correct_answer_l2977_297748

/-- Represents the number of questions Liam answered correctly in each category -/
structure CorrectAnswers where
  programming : ℕ
  dataStructures : ℕ
  algorithms : ℕ

/-- Calculates the total number of correct answers -/
def totalCorrect (answers : CorrectAnswers) : ℕ :=
  answers.programming + answers.dataStructures + answers.algorithms

/-- Represents the examination structure and Liam's performance -/
structure Examination where
  totalQuestions : ℕ
  programmingQuestions : ℕ
  dataStructuresQuestions : ℕ
  algorithmsQuestions : ℕ
  passingPercentage : ℚ
  correctAnswers : CorrectAnswers

/-- Theorem stating that Liam needed 1 more correct answer to pass -/
theorem liam_needed_one_more_correct_answer (exam : Examination)
  (h1 : exam.totalQuestions = 50)
  (h2 : exam.programmingQuestions = 15)
  (h3 : exam.dataStructuresQuestions = 20)
  (h4 : exam.algorithmsQuestions = 15)
  (h5 : exam.passingPercentage = 65 / 100)
  (h6 : exam.correctAnswers.programming = 12)
  (h7 : exam.correctAnswers.dataStructures = 10)
  (h8 : exam.correctAnswers.algorithms = 10) :
  ⌈exam.totalQuestions * exam.passingPercentage⌉ - totalCorrect exam.correctAnswers = 1 := by
  sorry


end NUMINAMATH_CALUDE_liam_needed_one_more_correct_answer_l2977_297748


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2977_297758

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : initial_distance = 180) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l2977_297758


namespace NUMINAMATH_CALUDE_min_sum_squares_l2977_297723

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2977_297723


namespace NUMINAMATH_CALUDE_polynomial_coefficient_identity_l2977_297709

theorem polynomial_coefficient_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_identity_l2977_297709


namespace NUMINAMATH_CALUDE_range_of_s_l2977_297760

-- Define the type for composite positive integers
def CompositePositiveInteger := {n : ℕ | n > 1 ∧ ¬ Prime n}

-- Define the function s
def s : CompositePositiveInteger → ℕ :=
  sorry -- Definition of s as sum of distinct prime factors

-- State the theorem about the range of s
theorem range_of_s :
  ∀ m : ℕ, m ≥ 2 ↔ ∃ n : CompositePositiveInteger, s n = m :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l2977_297760


namespace NUMINAMATH_CALUDE_emily_earnings_is_twenty_l2977_297714

/-- The number of chocolate bars in a box -/
def total_bars : ℕ := 8

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℕ := 4

/-- The number of unsold bars -/
def unsold_bars : ℕ := 3

/-- Emily's earnings from selling chocolate bars -/
def emily_earnings : ℕ := (total_bars - unsold_bars) * cost_per_bar

/-- Theorem stating that Emily's earnings are $20 -/
theorem emily_earnings_is_twenty : emily_earnings = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_earnings_is_twenty_l2977_297714


namespace NUMINAMATH_CALUDE_jump_rope_total_l2977_297794

theorem jump_rope_total (taehyung_jumps_per_day : ℕ) (taehyung_days : ℕ) 
                        (namjoon_jumps_per_day : ℕ) (namjoon_days : ℕ) :
  taehyung_jumps_per_day = 56 →
  taehyung_days = 3 →
  namjoon_jumps_per_day = 35 →
  namjoon_days = 4 →
  taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end NUMINAMATH_CALUDE_jump_rope_total_l2977_297794


namespace NUMINAMATH_CALUDE_union_complement_equal_set_l2977_297706

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_equal_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_set_l2977_297706


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2977_297704

/-- Proves that adding 750 mL of 30% alcohol solution to 250 mL of 10% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 750
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  
  total_alcohol / total_volume = target_concentration := by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2977_297704


namespace NUMINAMATH_CALUDE_box_weights_sum_l2977_297736

theorem box_weights_sum (heavy_box light_box sum : ℚ) : 
  heavy_box = 14/15 → 
  light_box = heavy_box - 1/10 → 
  sum = heavy_box + light_box → 
  sum = 53/30 := by sorry

end NUMINAMATH_CALUDE_box_weights_sum_l2977_297736


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2977_297776

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2977_297776


namespace NUMINAMATH_CALUDE_female_democrats_count_l2977_297718

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 750 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total / 3 : ℚ) →
  female / 2 = 125 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2977_297718


namespace NUMINAMATH_CALUDE_line_parametric_equation_l2977_297766

/-- The standard parametric equation of a line passing through a point with a given slope angle. -/
theorem line_parametric_equation (P : ℝ × ℝ) (θ : ℝ) :
  P = (1, -1) → θ = π / 3 →
  ∃ f g : ℝ → ℝ, 
    (∀ t, f t = 1 + (1/2) * t) ∧ 
    (∀ t, g t = -1 + (Real.sqrt 3 / 2) * t) ∧
    (∀ t, (f t, g t) ∈ {(x, y) | y - P.2 = Real.tan θ * (x - P.1)}) :=
sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l2977_297766


namespace NUMINAMATH_CALUDE_remainder_three_power_101_plus_five_mod_eleven_l2977_297763

theorem remainder_three_power_101_plus_five_mod_eleven :
  (3^101 + 5) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_101_plus_five_mod_eleven_l2977_297763


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l2977_297719

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l2977_297719


namespace NUMINAMATH_CALUDE_mn_minus_n_value_l2977_297711

theorem mn_minus_n_value (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 5/2) (h3 : m * n < 0) : 
  m * n - n = -7.5 ∨ m * n - n = -12.5 := by
sorry

end NUMINAMATH_CALUDE_mn_minus_n_value_l2977_297711


namespace NUMINAMATH_CALUDE_negative_six_div_three_l2977_297771

theorem negative_six_div_three : (-6) / 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_six_div_three_l2977_297771


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2977_297753

theorem trig_identity_proof (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : -Real.sin α = 2 * Real.cos α) :
  2 * (Real.sin α)^2 - Real.sin α * Real.cos α + (Real.cos α)^2 = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2977_297753


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2977_297740

/-- Represents the number of days needed to complete the work -/
def total_days_x : ℝ := 30

/-- Represents the number of days needed to complete the work -/
def total_days_y : ℝ := 15

/-- Represents the number of days x needs to finish the remaining work -/
def remaining_days_x : ℝ := 10.000000000000002

/-- Represents the number of days y worked before leaving -/
def days_y_worked : ℝ := 10

theorem work_completion_theorem :
  days_y_worked * (1 / total_days_y) + remaining_days_x * (1 / total_days_x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2977_297740


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l2977_297721

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents the number of painted faces on a unit cube -/
def painted_faces (c : Cube 4) (unit_cube : Fin 64) : ℕ := sorry

theorem unpainted_cubes_count (c : Cube 4) :
  (Finset.univ.filter (fun unit_cube => painted_faces c unit_cube = 0)).card = 58 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l2977_297721


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2977_297750

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def linesParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem parallel_line_through_point :
  let A : Point2D := ⟨-1, 0⟩
  let l1 : Line2D := ⟨2, -1, 1⟩
  let l2 : Line2D := ⟨2, -1, 2⟩
  pointOnLine A l2 ∧ linesParallel l1 l2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2977_297750


namespace NUMINAMATH_CALUDE_gcd_140_396_l2977_297785

theorem gcd_140_396 : Nat.gcd 140 396 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_140_396_l2977_297785


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_expected_result_l2977_297752

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | (1 - x) / x < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | -1 < x ∧ x < 0} ∪ {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem A_intersect_B_eq_expected_result : A_intersect_B = expected_result := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_expected_result_l2977_297752


namespace NUMINAMATH_CALUDE_color_film_fraction_l2977_297769

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 40 * x
  let total_color := 4 * y
  let selected_bw := (y / x) * (40 * x) / 100
  let selected_color := total_color
  (selected_color) / (selected_bw + selected_color) = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2977_297769


namespace NUMINAMATH_CALUDE_distance_at_speed1_proof_l2977_297778

-- Define the total distance
def total_distance : ℝ := 250

-- Define the two speeds
def speed1 : ℝ := 40
def speed2 : ℝ := 60

-- Define the total time
def total_time : ℝ := 5.2

-- Define the distance covered at speed1 (40 kmph)
def distance_at_speed1 : ℝ := 124

-- Theorem statement
theorem distance_at_speed1_proof :
  let distance_at_speed2 := total_distance - distance_at_speed1
  (distance_at_speed1 / speed1) + (distance_at_speed2 / speed2) = total_time :=
by sorry

end NUMINAMATH_CALUDE_distance_at_speed1_proof_l2977_297778


namespace NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l2977_297715

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_A_in_fourth_quadrant :
  is_in_fourth_quadrant 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_fourth_quadrant_l2977_297715


namespace NUMINAMATH_CALUDE_bijection_image_l2977_297710

def B : Set ℤ := {-3, 3, 5}

def f (x : ℤ) : ℤ := 2 * x - 1

theorem bijection_image (A : Set ℤ) :
  (Function.Bijective f) → (f '' A = B) → A = {-1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_bijection_image_l2977_297710


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_one_l2977_297747

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b (y : ℝ) : ℝ × ℝ := (1, -2*y)

/-- Definition of parallel vectors in ℝ² -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Theorem: If a and b(y) are parallel, then y = 1 -/
theorem parallel_vectors_imply_y_equals_one :
  parallel a (b y) → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_y_equals_one_l2977_297747


namespace NUMINAMATH_CALUDE_probability_sum_six_l2977_297751

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

def total_outcomes : Finset (ℕ × ℕ) :=
  cards.product cards

theorem probability_sum_six :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_l2977_297751


namespace NUMINAMATH_CALUDE_expression_equals_four_l2977_297791

theorem expression_equals_four :
  (8 : ℝ) ^ (1/3) + (1/3)⁻¹ - 2 * Real.cos (30 * π / 180) + |1 - Real.sqrt 3| = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_four_l2977_297791


namespace NUMINAMATH_CALUDE_investment_sum_l2977_297770

/-- Proves that given a sum P invested at 18% p.a. for two years generates Rs. 600 more interest
    than if invested at 12% p.a. for the same period, then P = 5000. -/
theorem investment_sum (P : ℚ) : 
  (P * 18 * 2 / 100) - (P * 12 * 2 / 100) = 600 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2977_297770


namespace NUMINAMATH_CALUDE_target_breaking_orders_l2977_297756

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem target_breaking_orders : 
  let n : ℕ := 9
  let ks : List ℕ := [4, 3, 2]
  multinomial n ks = 1260 := by sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l2977_297756


namespace NUMINAMATH_CALUDE_normal_distribution_equality_l2977_297789

-- Define the random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the normal distribution parameters
variable (μ σ : ℝ)

-- Define the probability measure
variable (P : Set ℝ → ℝ)

-- State the theorem
theorem normal_distribution_equality (h1 : μ = 2) 
  (h2 : P {x | ξ x ≤ 4 - a} = P {x | ξ x ≥ 2 + 3 * a}) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_equality_l2977_297789


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2977_297732

/-- Given 54 animal drawings distributed equally among 6 neighbors, prove that each neighbor receives 9 drawings. -/
theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) (drawings_per_neighbor : ℕ) : 
  total_drawings = 54 → 
  num_neighbors = 6 → 
  total_drawings = num_neighbors * drawings_per_neighbor →
  drawings_per_neighbor = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2977_297732


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2977_297717

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 20)
  (h2 : zero_books = 2)
  (h3 : one_book = 8)
  (h4 : two_books = 3)
  (h5 : (total_students - (zero_books + one_book + two_books)) * 3 ≤ 
        total_students * 2 - (one_book * 1 + two_books * 2)) :
  ∃ (max_books : ℕ), max_books = 8 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l2977_297717


namespace NUMINAMATH_CALUDE_not_circle_iff_a_eq_zero_l2977_297700

/-- The equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The condition for the equation to represent a circle -/
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the equation does not represent a circle iff a = 0 -/
theorem not_circle_iff_a_eq_zero (a : ℝ) :
  ¬(is_circle a) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_not_circle_iff_a_eq_zero_l2977_297700


namespace NUMINAMATH_CALUDE_marathon_practice_average_distance_l2977_297764

/-- Given a person who practices running for a certain number of days and covers a total distance,
    this function calculates the average distance run per day. -/
def average_distance_per_day (total_days : ℕ) (total_distance : ℕ) : ℚ :=
  total_distance / total_days

/-- Theorem stating that for a 9-day practice covering 72 miles, the average distance per day is 8 miles. -/
theorem marathon_practice_average_distance :
  average_distance_per_day 9 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_marathon_practice_average_distance_l2977_297764


namespace NUMINAMATH_CALUDE_bicycle_sale_price_l2977_297703

/-- Given a cost price and two consecutive percentage markups, 
    calculate the final selling price. -/
def final_price (cost_price : ℚ) (markup_percent : ℚ) : ℚ :=
  let first_sale := cost_price * (1 + markup_percent / 100)
  first_sale * (1 + markup_percent / 100)

/-- Theorem: The final selling price of a bicycle with an initial cost of 144,
    after two consecutive 25% markups, is 225. -/
theorem bicycle_sale_price : final_price 144 25 = 225 := by
  sorry

#eval final_price 144 25

end NUMINAMATH_CALUDE_bicycle_sale_price_l2977_297703


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2977_297784

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 5) * (-2*x) - 6*x^2 = -2*x^3 - 10*x) ↔ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2977_297784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2977_297745

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2977_297745


namespace NUMINAMATH_CALUDE_inequality_proof_l2977_297761

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hne : a ≠ b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt 2 ∧ Real.sqrt 2 < 1 / (2^a) + 1 / (2^b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2977_297761


namespace NUMINAMATH_CALUDE_john_ray_difference_l2977_297792

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution -/
def valid_distribution (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.ray = c.mary - 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chicken count -/
theorem john_ray_difference (c : ChickenCount) (h : valid_distribution c) : 
  c.john - c.ray = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_ray_difference_l2977_297792


namespace NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l2977_297716

/-- Given a sinusoidal function y = a * sin(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 3 and the minimum value is -1, then d = 1. -/
theorem sinusoidal_vertical_shift 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (b * x + c) + d)
  (hmax : ∀ x, f x ≤ 3)
  (hmin : ∀ x, f x ≥ -1)
  (hex_max : ∃ x, f x = 3)
  (hex_min : ∃ x, f x = -1) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_sinusoidal_vertical_shift_l2977_297716


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2977_297777

def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum :
  arithmetic_sum (-41) 1 2 = -440 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2977_297777


namespace NUMINAMATH_CALUDE_set_union_problem_l2977_297788

def M (a : ℕ) : Set ℕ := {3, 4^a}
def N (a b : ℕ) : Set ℕ := {a, b}

theorem set_union_problem (a b : ℕ) :
  M a ∩ N a b = {1} → M a ∪ N a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2977_297788


namespace NUMINAMATH_CALUDE_rectangle_properties_l2977_297742

/-- The equation representing the roots of the rectangle's sides -/
def side_equation (m x : ℝ) : Prop := x^2 - m*x + m/2 - 1/4 = 0

/-- The condition for the rectangle to be a square -/
def is_square (m : ℝ) : Prop := ∃ x : ℝ, side_equation m x ∧ ∀ y : ℝ, side_equation m y → y = x

/-- The perimeter of the rectangle given one side length -/
def perimeter (ab bc : ℝ) : ℝ := 2 * (ab + bc)

theorem rectangle_properties :
  (∃ m : ℝ, is_square m ∧ ∃ x : ℝ, side_equation m x ∧ x = 1/2) ∧
  (∃ m : ℝ, side_equation m 2 ∧ ∃ bc : ℝ, side_equation m bc ∧ perimeter 2 bc = 5) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l2977_297742


namespace NUMINAMATH_CALUDE_probability_three_suits_standard_deck_l2977_297762

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The probability of drawing one card each from three specific suits in the top three cards -/
def probability_three_suits (d : Deck) : Rat :=
  sorry

/-- The standard 52-card deck -/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4 }

theorem probability_three_suits_standard_deck :
  probability_three_suits standard_deck = 2197 / 22100 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_suits_standard_deck_l2977_297762


namespace NUMINAMATH_CALUDE_fraction_equality_l2977_297775

theorem fraction_equality (x y : ℝ) (h : x / y = 4 / 3) : (x - y) / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2977_297775


namespace NUMINAMATH_CALUDE_cards_per_page_l2977_297735

theorem cards_per_page
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (num_pages : ℕ)
  (h1 : num_packs = 60)
  (h2 : cards_per_pack = 7)
  (h3 : num_pages = 42)
  : (num_packs * cards_per_pack) / num_pages = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l2977_297735


namespace NUMINAMATH_CALUDE_wire_cutting_l2977_297755

theorem wire_cutting (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) (n : ℕ) :
  total_length = 50 →
  used_parts = 3 →
  unused_length = 20 →
  total_length = n * (total_length - unused_length) / used_parts →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l2977_297755


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2977_297749

theorem divisibility_theorem (a b c d u : ℤ) 
  (h1 : u ∣ (a * c)) 
  (h2 : u ∣ (b * c + a * d)) 
  (h3 : u ∣ (b * d)) : 
  (u ∣ (b * c)) ∧ (u ∣ (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2977_297749


namespace NUMINAMATH_CALUDE_range_of_piecewise_function_l2977_297795

/-- Given two linear functions f and g, and a piecewise function r,
    prove that the range of r is [a/2 + b, c + d] -/
theorem range_of_piecewise_function
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (r : ℝ → ℝ)
  (ha : a < 0)
  (hc : c > 0)
  (hf : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (hg : ∀ x, 0 ≤ x ∧ x ≤ 1 → g x = c * x + d)
  (hr : ∀ x, 0 ≤ x ∧ x ≤ 1 → r x = if x ≤ 0.5 then f x else g x) :
  Set.range r = Set.Icc (a / 2 + b) (c + d) :=
sorry

end NUMINAMATH_CALUDE_range_of_piecewise_function_l2977_297795


namespace NUMINAMATH_CALUDE_periodic_binomial_remainder_l2977_297774

theorem periodic_binomial_remainder (K : ℕ+) : 
  (∃ (p : ℕ+), ∀ (n : ℕ), n ≥ p → 
    (∃ (T : ℕ+), ∀ (m : ℕ), m ≥ p → 
      (Nat.choose (2*(n+m)) (n+m)) % K = (Nat.choose (2*n) n) % K)) ↔ 
  (K = 1 ∨ K = 2) :=
sorry

end NUMINAMATH_CALUDE_periodic_binomial_remainder_l2977_297774


namespace NUMINAMATH_CALUDE_trululu_nonexistence_l2977_297798

structure Individual where
  statement : Prop

def is_weekday (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 5

def Barmaglot_lies (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 3

theorem trululu_nonexistence (day : Nat) 
  (h1 : is_weekday day)
  (h2 : ∃ (i1 i2 : Individual), i1.statement = (∃ Trululu : Type, Nonempty Trululu) ∧ i2.statement = True)
  (h3 : ∀ (i : Individual), i.statement = True → i.statement)
  (h4 : Barmaglot_lies day → ¬(∃ Trululu : Type, Nonempty Trululu))
  (h5 : ¬(Barmaglot_lies day))
  : ¬(∃ Trululu : Type, Nonempty Trululu) := by
  sorry

#check trululu_nonexistence

end NUMINAMATH_CALUDE_trululu_nonexistence_l2977_297798


namespace NUMINAMATH_CALUDE_f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l2977_297725

/-- The original quadratic function -/
def f (x : ℝ) := x^2 - 2*x + 1

/-- The shifted quadratic function -/
def g (x : ℝ) := (x - 1)^2

/-- Theorem stating that f and g are equivalent -/
theorem f_eq_g : ∀ x, f x = g x := by sorry

/-- Theorem stating that g is a right shift of x^2 by 1 unit -/
theorem g_is_right_shift : ∀ x, g x = (x - 1)^2 := by sorry

/-- Main theorem: f is a right shift of x^2 by 1 unit -/
theorem f_is_right_shift_of_x_squared : 
  ∃ h : ℝ, h > 0 ∧ (∀ x, f x = (x - h)^2) := by sorry

end NUMINAMATH_CALUDE_f_eq_g_g_is_right_shift_f_is_right_shift_of_x_squared_l2977_297725


namespace NUMINAMATH_CALUDE_sequence_properties_l2977_297741

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) (k : ℝ) : ℝ := k * n^2 + n

/-- The nth term of the sequence -/
def a (n : ℕ+) (k : ℝ) : ℝ := k * (2 * n - 1) + 1

theorem sequence_properties (k : ℝ) :
  (∀ n : ℕ+, S n k - S (n-1) k = a n k) ∧
  (∀ m : ℕ+, (a (2*m) k)^2 = (a m k) * (a (4*m) k)) →
  k = 1/3 := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2977_297741


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2977_297780

theorem min_value_trig_expression (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≥ (2/3) * ((Real.sin x)^2 + (Real.cos x)^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2977_297780


namespace NUMINAMATH_CALUDE_trig_simplification_l2977_297757

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2977_297757


namespace NUMINAMATH_CALUDE_converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l2977_297726

-- Define the concept of vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- Define the concept of consecutive interior angles
def consecutive_interior_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a b : Angle) : Prop := sorry

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem converse_parallel_supplementary_true :
  ∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b := by sorry

theorem converse_vertical_angles_false :
  ∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b) := by sorry

theorem converse_squares_equal_false :
  ∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b := by sorry

theorem converse_sum_squares_positive_false :
  ∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0) := by sorry

theorem only_parallel_supplementary_has_true_converse :
  (∀ (l1 l2 : Line) (a b : Angle),
    parallel l1 l2 → consecutive_interior_angles a b l1 l2 → supplementary a b) ∧
  (∃ (a b : Angle), a = b ∧ ¬(vertical_angles a b)) ∧
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  (∃ (a b : ℝ), a^2 + b^2 > 0 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end NUMINAMATH_CALUDE_converse_parallel_supplementary_true_converse_vertical_angles_false_converse_squares_equal_false_converse_sum_squares_positive_false_only_parallel_supplementary_has_true_converse_l2977_297726


namespace NUMINAMATH_CALUDE_bus_speed_with_stoppages_l2977_297722

/-- Calculates the speed of a bus including stoppages -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 54 →
  stoppage_time = 10 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time) / total_time) = 45 :=
by
  sorry

#check bus_speed_with_stoppages

end NUMINAMATH_CALUDE_bus_speed_with_stoppages_l2977_297722


namespace NUMINAMATH_CALUDE_negative_real_inequality_l2977_297739

theorem negative_real_inequality (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  a > b ↔ a - 1 / a > b - 1 / b := by sorry

end NUMINAMATH_CALUDE_negative_real_inequality_l2977_297739
