import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2139_213962

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2139_213962


namespace NUMINAMATH_CALUDE_marias_average_balance_l2139_213983

/-- Given Maria's savings account balances for four months, prove that the average monthly balance is $300. -/
theorem marias_average_balance (jan feb mar apr : ℕ) 
  (h_jan : jan = 150)
  (h_feb : feb = 300)
  (h_mar : mar = 450)
  (h_apr : apr = 300) :
  (jan + feb + mar + apr) / 4 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marias_average_balance_l2139_213983


namespace NUMINAMATH_CALUDE_smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l2139_213910

theorem smallest_four_digit_perfect_square_multiple :
  ∀ m : ℕ, m ≥ 1000 → m < 1029 → ¬∃ n : ℕ, 21 * m = n * n :=
by
  sorry

theorem smallest_four_digit_perfect_square_multiple_exists :
  ∃ n : ℕ, 21 * 1029 = n * n :=
by
  sorry

#check smallest_four_digit_perfect_square_multiple
#check smallest_four_digit_perfect_square_multiple_exists

end NUMINAMATH_CALUDE_smallest_four_digit_perfect_square_multiple_smallest_four_digit_perfect_square_multiple_exists_l2139_213910


namespace NUMINAMATH_CALUDE_square_divides_power_plus_one_l2139_213918

theorem square_divides_power_plus_one (n : ℕ) : 
  n ^ 2 ∣ 2 ^ n + 1 ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_square_divides_power_plus_one_l2139_213918


namespace NUMINAMATH_CALUDE_lisas_marbles_problem_l2139_213961

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem :
  min_additional_marbles 12 40 = 38 := by
  sorry

#eval min_additional_marbles 12 40

end NUMINAMATH_CALUDE_lisas_marbles_problem_l2139_213961


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2139_213932

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2139_213932


namespace NUMINAMATH_CALUDE_greeting_card_distribution_four_l2139_213957

def greeting_card_distribution (n : ℕ) : ℕ :=
  if n = 4 then 9 else 0

theorem greeting_card_distribution_four :
  greeting_card_distribution 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_greeting_card_distribution_four_l2139_213957


namespace NUMINAMATH_CALUDE_ratio_problem_l2139_213938

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2139_213938


namespace NUMINAMATH_CALUDE_integer_root_values_l2139_213923

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + a*x^2 + 3*x + 7 = 0) ↔ (a = -11 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2139_213923


namespace NUMINAMATH_CALUDE_function_equality_implies_n_value_l2139_213935

/-- The function f(x) = 2x^2 - 3x + n -/
def f (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + n

/-- The function g(x) = 2x^2 - 3x + 5n -/
def g (n : ℚ) (x : ℚ) : ℚ := 2 * x^2 - 3 * x + 5 * n

/-- Theorem stating that if 3f(3) = 2g(3), then n = 9/7 -/
theorem function_equality_implies_n_value :
  ∀ n : ℚ, 3 * (f n 3) = 2 * (g n 3) → n = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_n_value_l2139_213935


namespace NUMINAMATH_CALUDE_smallest_stairs_l2139_213991

theorem smallest_stairs (n : ℕ) : 
  (n > 20 ∧ n % 6 = 4 ∧ n % 7 = 3) → n ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_smallest_stairs_l2139_213991


namespace NUMINAMATH_CALUDE_product_square_of_sum_and_diff_l2139_213999

theorem product_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 23) 
  (diff_eq : x - y = 7) : 
  (x * y)^2 = 14400 := by
sorry

end NUMINAMATH_CALUDE_product_square_of_sum_and_diff_l2139_213999


namespace NUMINAMATH_CALUDE_max_b_value_l2139_213955

theorem max_b_value (a b c : ℕ) : 
  1 < c → c < b → b < a → a * b * c = 360 → b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l2139_213955


namespace NUMINAMATH_CALUDE_perimeter_of_z_shape_l2139_213916

-- Define the complex number z
variable (z : ℂ)

-- Define the condition that z satisfies
def satisfies_condition (z : ℂ) : Prop :=
  Complex.arg z = Complex.arg (Complex.I * z + Complex.I)

-- Define the shape corresponding to z
def shape_of_z (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ t : ℝ, w = t * z ∧ 0 ≤ t ∧ t ≤ 1}

-- Define the perimeter of the shape
def perimeter_of_shape (s : Set ℂ) : ℝ := sorry

-- State the theorem
theorem perimeter_of_z_shape (h : satisfies_condition z) :
  perimeter_of_shape (shape_of_z z) = π / 2 :=
sorry

end NUMINAMATH_CALUDE_perimeter_of_z_shape_l2139_213916


namespace NUMINAMATH_CALUDE_opposite_of_two_and_two_thirds_l2139_213947

theorem opposite_of_two_and_two_thirds :
  -(2 + 2/3 : ℚ) = -2 - 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_and_two_thirds_l2139_213947


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2139_213936

/-- The sum of the repeating decimals 0.3̄ and 0.6̄ is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (∀ n : ℕ, x * 10^n - ⌊x * 10^n⌋ = 0.3) ∧ 
                (∀ n : ℕ, y * 10^n - ⌊y * 10^n⌋ = 0.6) ∧ 
                x + y = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2139_213936


namespace NUMINAMATH_CALUDE_min_beacons_proof_l2139_213956

/-- Represents a room in the maze --/
structure Room :=
  (x : Nat) (y : Nat)

/-- Represents the maze structure --/
def Maze := List Room

/-- Calculates the distance between two rooms --/
def distance (r1 r2 : Room) : Nat :=
  sorry

/-- Checks if a set of beacon positions allows unambiguous location determination --/
def is_unambiguous (maze : Maze) (beacons : List Room) : Prop :=
  sorry

/-- The minimum number of beacons required --/
def min_beacons : Nat := 3

/-- The specific beacon positions that work --/
def beacon_positions : List Room :=
  [⟨1, 1⟩, ⟨4, 3⟩, ⟨1, 5⟩]  -- Representing a1, d3, a5

theorem min_beacons_proof (maze : Maze) :
  (∀ beacons : List Room, beacons.length < min_beacons → ¬ is_unambiguous maze beacons) ∧
  is_unambiguous maze beacon_positions :=
sorry

end NUMINAMATH_CALUDE_min_beacons_proof_l2139_213956


namespace NUMINAMATH_CALUDE_triangle_max_area_l2139_213953

/-- Given a triangle ABC with side lengths a, b, c opposite angles A, B, C respectively,
    prove that the maximum area is √3 under the given conditions. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  (Real.sin A - Real.sin B) / Real.sin C = (c - b) / (2 + b) →
  (∀ a' b' c' A' B' C',
    a' = 2 →
    (Real.sin A' - Real.sin B') / Real.sin C' = (c' - b') / (2 + b') →
    (1/2) * a' * b' * Real.sin C' ≤ Real.sqrt 3) ∧
  (∃ b' c',
    (Real.sin A - Real.sin B) / Real.sin C = (c' - b') / (2 + b') →
    (1/2) * a * b' * Real.sin C = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2139_213953


namespace NUMINAMATH_CALUDE_inequality_proof_l2139_213912

theorem inequality_proof (x y : ℝ) : x^2 + y^2 + 1 ≥ x*y + x + y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2139_213912


namespace NUMINAMATH_CALUDE_monkey_climb_time_l2139_213980

/-- A monkey climbing a tree with specific conditions -/
def monkey_climb (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) : ℕ :=
  let effective_climb := hop_distance - slip_distance
  let full_climbs := (tree_height - 1) / effective_climb
  let remaining_distance := (tree_height - 1) % effective_climb
  full_climbs + if remaining_distance > 0 then 1 else 0

/-- Theorem stating the time taken by the monkey to climb the tree -/
theorem monkey_climb_time :
  monkey_climb 17 3 2 = 17 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_time_l2139_213980


namespace NUMINAMATH_CALUDE_arrival_time_difference_l2139_213914

/-- Represents the distance to the pool in miles -/
def distance_to_pool : ℝ := 3

/-- Represents Jill's speed in miles per hour -/
def jill_speed : ℝ := 12

/-- Represents Jack's speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculates the time difference in minutes between Jill and Jack's arrival at the pool -/
theorem arrival_time_difference : 
  hours_to_minutes (distance_to_pool / jill_speed - distance_to_pool / jack_speed) = 45 := by
  sorry

end NUMINAMATH_CALUDE_arrival_time_difference_l2139_213914


namespace NUMINAMATH_CALUDE_all_statements_incorrect_l2139_213997

/-- Represents a type of reasoning -/
inductive ReasoningType
| Analogical
| Inductive

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| GeneralToSpecific
| SpecificToGeneral
| SpecificToSpecific

/-- Represents a statement about analogical reasoning -/
structure AnalogicalReasoningStatement where
  always_correct : Bool
  direction : ReasoningDirection
  can_prove_math : Bool
  same_as_inductive : Bool

/-- Definition of analogical reasoning -/
def analogical_reasoning : ReasoningType := ReasoningType.Analogical

/-- Definition of inductive reasoning -/
def inductive_reasoning : ReasoningType := ReasoningType.Inductive

/-- Inductive reasoning is a form of analogical reasoning -/
axiom inductive_is_analogical : inductive_reasoning = analogical_reasoning

/-- The correct properties of analogical reasoning -/
def correct_properties : AnalogicalReasoningStatement :=
  { always_correct := false
  , direction := ReasoningDirection.SpecificToSpecific
  , can_prove_math := false
  , same_as_inductive := false }

/-- Theorem stating that all given statements about analogical reasoning are incorrect -/
theorem all_statements_incorrect (statement : AnalogicalReasoningStatement) :
  statement.always_correct = true ∨
  statement.direction = ReasoningDirection.GeneralToSpecific ∨
  statement.can_prove_math = true ∨
  statement.same_as_inductive = true →
  statement ≠ correct_properties :=
sorry

end NUMINAMATH_CALUDE_all_statements_incorrect_l2139_213997


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l2139_213950

theorem inequality_holds_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3*m) ↔ 
  (m ≤ -1 ∨ m ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l2139_213950


namespace NUMINAMATH_CALUDE_two_aces_probability_l2139_213939

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of Aces in a standard deck
def numAces : ℕ := 4

-- Define the probability of drawing two Aces
def probTwoAces : ℚ := 1 / 221

-- Theorem statement
theorem two_aces_probability :
  (numAces / totalCards) * ((numAces - 1) / (totalCards - 1)) = probTwoAces := by
  sorry

end NUMINAMATH_CALUDE_two_aces_probability_l2139_213939


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2139_213945

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem interior_angle_regular_octagon :
  (sum_interior_angles octagon_sides) / octagon_sides = 135 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2139_213945


namespace NUMINAMATH_CALUDE_max_distance_difference_l2139_213928

def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

def circle_C2 (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 9

def on_x_axis (x y : ℝ) : Prop := y = 0

theorem max_distance_difference :
  ∃ (max : ℝ),
    (∀ (Mx My Nx Ny Px Py : ℝ),
      circle_C1 Mx My →
      circle_C2 Nx Ny →
      on_x_axis Px Py →
      Real.sqrt ((Nx - Px)^2 + (Ny - Py)^2) -
      Real.sqrt ((Mx - Px)^2 + (My - Py)^2) ≤ max) ∧
    max = 4 + Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_difference_l2139_213928


namespace NUMINAMATH_CALUDE_t_has_six_values_l2139_213976

/-- A type representing single-digit positive integers -/
def SingleDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The theorem stating that t can have 6 distinct values -/
theorem t_has_six_values 
  (p q r s t : SingleDigit) 
  (h1 : p.val - q.val = r.val)
  (h2 : r.val - s.val = t.val)
  (h3 : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
        q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
        r ≠ s ∧ r ≠ t ∧ 
        s ≠ t) :
  ∃ (values : Finset ℕ), values.card = 6 ∧ t.val ∈ values ∧ 
  ∀ x, x ∈ values → ∃ (p' q' r' s' t' : SingleDigit), 
    p'.val - q'.val = r'.val ∧ 
    r'.val - s'.val = t'.val ∧ 
    t'.val = x ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ 
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ 
    r' ≠ s' ∧ r' ≠ t' ∧ 
    s' ≠ t' := by
  sorry

end NUMINAMATH_CALUDE_t_has_six_values_l2139_213976


namespace NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_proof_l2139_213974

/-- The probability of cutting a 1-meter rope at a random point such that the longer piece is at least three times as large as the shorter piece is 1/2. -/
theorem rope_cutting_probability : Real → Prop :=
  fun p => p = 1/2 ∧ 
    ∀ c : Real, 0 ≤ c ∧ c ≤ 1 →
      (((1 - c ≥ 3 * c) ∨ (c ≥ 3 * (1 - c))) ↔ (c ≤ 1/4 ∨ c ≥ 3/4)) ∧
      p = (1/4 - 0) + (1 - 3/4)

/-- Proof of the rope cutting probability theorem -/
theorem rope_cutting_probability_proof : ∃ p, rope_cutting_probability p :=
  sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_proof_l2139_213974


namespace NUMINAMATH_CALUDE_election_votes_total_l2139_213951

theorem election_votes_total (winner_percentage : ℚ) (vote_majority : ℕ) : 
  winner_percentage = 7/10 →
  vote_majority = 280 →
  ∃ (total_votes : ℕ), 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = vote_majority ∧
    total_votes = 700 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_total_l2139_213951


namespace NUMINAMATH_CALUDE_ram_money_l2139_213993

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Krishan's amount,
    calculate the amount of money Ram has. -/
theorem ram_money (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  krishan = 3468 →
  ram = 588 := by
sorry

end NUMINAMATH_CALUDE_ram_money_l2139_213993


namespace NUMINAMATH_CALUDE_problem_l2139_213907

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x / Real.cos x + 3

theorem problem (a : ℝ) (h : f (Real.log a) = 4) : f (Real.log (1 / a)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_l2139_213907


namespace NUMINAMATH_CALUDE_sqrt_3_powers_l2139_213988

theorem sqrt_3_powers (n : ℕ) (h : n ≥ 1) :
  ∃ (k w : ℕ), 
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) + ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (2 * k : ℝ)) ∧
    (((2 : ℝ) + Real.sqrt 3) ^ (2 * n) - ((2 : ℝ) - Real.sqrt 3) ^ (2 * n) = (w : ℝ) * Real.sqrt 3) ∧
    w > 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_3_powers_l2139_213988


namespace NUMINAMATH_CALUDE_m_values_l2139_213905

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values (m : ℝ) : (A ∪ B m = A) → (m = 0 ∨ m = -1 ∨ m = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_m_values_l2139_213905


namespace NUMINAMATH_CALUDE_function_properties_l2139_213986

-- Define the function f(x) = ax³ + bx
def f (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem function_properties (a b : ℝ) :
  -- Condition 1: Tangent line at x=3 is parallel to 24x - y + 1 = 0
  f' a b 3 = 24 →
  -- Condition 2: Function has an extremum at x=1
  f' a b 1 = 0 →
  -- Condition 3: a = 1
  a = 1 →
  -- Conclusion 1: f(x) = x³ - 3x
  (∀ x, f a b x = x^3 - 3*x) ∧
  -- Conclusion 2: Interval of monotonic decrease is [-1, 1]
  (∀ x, x ∈ Set.Icc (-1) 1 → f' a b x ≤ 0) ∧
  -- Conclusion 3: For f(x) to be decreasing on [-1, 1], b ≤ -3
  (∀ x, x ∈ Set.Icc (-1) 1 → f' 1 b x ≤ 0) → b ≤ -3 :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l2139_213986


namespace NUMINAMATH_CALUDE_window_purchase_savings_l2139_213933

/-- Represents the store's window sale offer -/
structure WindowOffer where
  regularPrice : ℕ
  buyCount : ℕ
  freeCount : ℕ

/-- Calculates the cost of purchasing a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buyCount + offer.freeCount)
  let remainder := windowsNeeded % (offer.buyCount + offer.freeCount)
  let windowsPaidFor := fullSets * offer.buyCount + min remainder offer.buyCount
  windowsPaidFor * offer.regularPrice

/-- The main theorem stating the savings when Dave and Doug purchase windows together -/
theorem window_purchase_savings : 
  let offer : WindowOffer := ⟨100, 3, 1⟩
  let davesWindows : ℕ := 9
  let dougsWindows : ℕ := 10
  let totalWindows : ℕ := davesWindows + dougsWindows
  let separateCost : ℕ := calculateCost offer davesWindows + calculateCost offer dougsWindows
  let combinedCost : ℕ := calculateCost offer totalWindows
  let savings : ℕ := separateCost - combinedCost
  savings = 600 := by sorry

end NUMINAMATH_CALUDE_window_purchase_savings_l2139_213933


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2139_213901

theorem quadratic_inequality_equivalence (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x - k + 8 ≥ 0) ↔ 
  (k ≥ -2 * Real.sqrt 7 ∧ k ≤ 2 * Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2139_213901


namespace NUMINAMATH_CALUDE_cube_surface_area_l2139_213920

/-- Given three points A, B, and C as vertices of a cube, prove that its surface area is 294 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (1, 4, 2) → B = (2, 0, -7) → C = (5, -5, 1) → 
  (let surface_area := 6 * (dist A B)^2
   surface_area = 294) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2139_213920


namespace NUMINAMATH_CALUDE_complex_is_real_iff_m_eq_neg_one_l2139_213948

theorem complex_is_real_iff_m_eq_neg_one (m : ℝ) :
  (∃ (x : ℝ), Complex.mk (m - 1) (m + 1) = x) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_is_real_iff_m_eq_neg_one_l2139_213948


namespace NUMINAMATH_CALUDE_largest_integer_k_for_real_roots_l2139_213925

theorem largest_integer_k_for_real_roots : ∃ (k : ℤ),
  (∀ (j : ℤ), (∀ (x : ℝ), ∃ (y : ℝ), x * (j * x + 1) - x^2 + 3 = 0) → j ≤ k) ∧
  (∀ (x : ℝ), ∃ (y : ℝ), x * (k * x + 1) - x^2 + 3 = 0) ∧
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_k_for_real_roots_l2139_213925


namespace NUMINAMATH_CALUDE_store_profit_percentage_l2139_213959

/-- Given a store that purchases an item and the conditions of a potential price change,
    this theorem proves that the original profit percentage was 35%. -/
theorem store_profit_percentage
  (original_cost : ℝ)
  (cost_decrease_percentage : ℝ)
  (profit_increase_percentage : ℝ)
  (h1 : original_cost = 200)
  (h2 : cost_decrease_percentage = 10)
  (h3 : profit_increase_percentage = 15)
  (h4 : ∃ (sale_price : ℝ) (original_profit_percentage : ℝ),
        sale_price = original_cost * (1 + original_profit_percentage / 100) ∧
        sale_price = (original_cost * (1 - cost_decrease_percentage / 100)) *
                     (1 + (original_profit_percentage + profit_increase_percentage) / 100)) :
  ∃ (original_profit_percentage : ℝ), original_profit_percentage = 35 :=
sorry

end NUMINAMATH_CALUDE_store_profit_percentage_l2139_213959


namespace NUMINAMATH_CALUDE_marks_per_correct_answer_l2139_213989

/-- Proves that the number of marks scored for each correct answer is 4 -/
theorem marks_per_correct_answer 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_marks : ℤ) 
  (wrong_answer_penalty : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_answers = 36)
  (h3 : total_marks = 130)
  (h4 : wrong_answer_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℤ), 
    marks_per_correct * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by sorry

end NUMINAMATH_CALUDE_marks_per_correct_answer_l2139_213989


namespace NUMINAMATH_CALUDE_number_relationship_l2139_213971

theorem number_relationship (x : ℝ) : 
  (5 * x = 2 * x + 10) → (5 * x - 2 * x = 10) := by sorry

end NUMINAMATH_CALUDE_number_relationship_l2139_213971


namespace NUMINAMATH_CALUDE_apple_production_solution_l2139_213967

/-- Represents the apple production of a tree over three years -/
structure AppleProduction where
  first_year : ℕ
  second_year : ℕ := 2 * first_year + 8
  third_year : ℕ := (3 * second_year) / 4

/-- Theorem stating the solution to the apple production problem -/
theorem apple_production_solution :
  ∃ (prod : AppleProduction),
    prod.first_year + prod.second_year + prod.third_year = 194 ∧
    prod.first_year = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_production_solution_l2139_213967


namespace NUMINAMATH_CALUDE_trig_identity_l2139_213902

theorem trig_identity : (1 / (2 * Real.sin (10 * π / 180))) - 2 * Real.sin (70 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2139_213902


namespace NUMINAMATH_CALUDE_certain_point_on_circle_l2139_213929

/-- A point on the parabola y^2 = 8x -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x

/-- A circle with center on the parabola y^2 = 8x and tangent to x = -2 -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  is_tangent : center.x + radius = 2  -- Distance from center to x = -2 is equal to radius

theorem certain_point_on_circle (c : TangentCircle) : 
  (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

#check certain_point_on_circle

end NUMINAMATH_CALUDE_certain_point_on_circle_l2139_213929


namespace NUMINAMATH_CALUDE_principal_calculation_l2139_213965

/-- Calculates the principal amount given two interest rates, time period, and interest difference --/
def calculate_principal (rate1 rate2 : ℚ) (time : ℕ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (rate1 - rate2) * 100 / time

/-- Theorem stating that the calculated principal is approximately 7142.86 --/
theorem principal_calculation :
  let rate1 : ℚ := 22
  let rate2 : ℚ := 15
  let time : ℕ := 5
  let interest_diff : ℚ := 2500
  let principal := calculate_principal rate1 rate2 time interest_diff
  abs (principal - 7142.86) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2139_213965


namespace NUMINAMATH_CALUDE_B_power_97_l2139_213944

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -2; 0, 2, 0]

theorem B_power_97 : 
  B^97 = !![1, 0, 0; 0, 0, -2 * 16^24; 0, 2 * 16^24, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_97_l2139_213944


namespace NUMINAMATH_CALUDE_fraction_removal_sum_one_l2139_213909

theorem fraction_removal_sum_one :
  let fractions : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let removed : List ℚ := [1/8, 1/10]
  let remaining : List ℚ := fractions.filter (fun x => x ∉ removed)
  remaining.sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_removal_sum_one_l2139_213909


namespace NUMINAMATH_CALUDE_alex_upside_down_hours_l2139_213946

/-- The number of hours Alex needs to hang upside down each month to be tall enough for the roller coaster --/
def hours_upside_down (
  required_height : ℚ)
  (current_height : ℚ)
  (growth_rate_upside_down : ℚ)
  (natural_growth_rate : ℚ)
  (months_in_year : ℕ) : ℚ :=
  let height_difference := required_height - current_height
  let natural_growth := natural_growth_rate * months_in_year
  let additional_growth_needed := height_difference - natural_growth
  (additional_growth_needed / growth_rate_upside_down) / months_in_year

/-- Theorem stating that Alex needs to hang upside down for 2 hours each month --/
theorem alex_upside_down_hours :
  hours_upside_down 54 48 (1/12) (1/3) 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_upside_down_hours_l2139_213946


namespace NUMINAMATH_CALUDE_area_ratio_in_square_l2139_213981

/-- Given a unit square ABCD with points X on BC and Y on CD such that
    triangles ABX, XCY, and YDA have equal areas, the ratio of the area of
    triangle AXY to the area of triangle XCY is √5. -/
theorem area_ratio_in_square (A B C D X Y : ℝ × ℝ) : 
  let square_side_length : ℝ := 1
  let on_side (P Q R : ℝ × ℝ) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • Q + t • R
  let area (P Q R : ℝ × ℝ) : ℝ := abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2
  square_side_length = 1 →
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = 1 ∧ B.2 = 0) →
  (C.1 = 1 ∧ C.2 = 1) →
  (D.1 = 0 ∧ D.2 = 1) →
  on_side X B C →
  on_side Y C D →
  area A B X = area X C Y →
  area X C Y = area Y D A →
  area A X Y / area X C Y = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_in_square_l2139_213981


namespace NUMINAMATH_CALUDE_planted_fraction_is_thirteen_fifteenths_l2139_213979

/-- Represents a right triangle field with an unplanted rectangle at the right angle -/
structure FieldWithUnplantedRectangle where
  /-- Length of the first leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle -/
  leg2 : ℝ
  /-- Width of the unplanted rectangle -/
  rect_width : ℝ
  /-- Height of the unplanted rectangle -/
  rect_height : ℝ
  /-- Shortest distance from the unplanted rectangle to the hypotenuse -/
  dist_to_hypotenuse : ℝ

/-- Calculates the fraction of the field that is planted -/
def planted_fraction (field : FieldWithUnplantedRectangle) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the given field configuration -/
theorem planted_fraction_is_thirteen_fifteenths :
  let field := FieldWithUnplantedRectangle.mk 5 12 1 4 3
  planted_fraction field = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_is_thirteen_fifteenths_l2139_213979


namespace NUMINAMATH_CALUDE_next_235_time_91_minutes_l2139_213927

def is_valid_time (h m : ℕ) : Prop :=
  h < 24 ∧ m < 60

def uses_digits_235_once (h m : ℕ) : Prop :=
  let digits := h.digits 10 ++ m.digits 10
  digits.count 2 = 1 ∧ digits.count 3 = 1 ∧ digits.count 5 = 1

def minutes_from_352_to (h m : ℕ) : ℕ :=
  if h < 3 ∨ (h = 3 ∧ m ≤ 52) then
    (h + 24 - 3) * 60 + (m - 52)
  else
    (h - 3) * 60 + (m - 52)

theorem next_235_time_91_minutes :
  ∃ (h m : ℕ), 
    is_valid_time h m ∧
    uses_digits_235_once h m ∧
    minutes_from_352_to h m = 91 ∧
    (∀ (h' m' : ℕ), 
      is_valid_time h' m' →
      uses_digits_235_once h' m' →
      minutes_from_352_to h' m' ≥ 91) :=
sorry

end NUMINAMATH_CALUDE_next_235_time_91_minutes_l2139_213927


namespace NUMINAMATH_CALUDE_f_negative_three_l2139_213930

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin x)^3 + b * Real.tan x + 1

theorem f_negative_three (a b : ℝ) :
  f a b 3 = 6 → f a b (-3) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_three_l2139_213930


namespace NUMINAMATH_CALUDE_triangle_area_l2139_213943

theorem triangle_area (a b c : ℝ) (h_perimeter : a + b + c = 10 + 2 * Real.sqrt 7)
  (h_ratio : ∃ (k : ℝ), a = 2 * k ∧ b = 3 * k ∧ c = k * Real.sqrt 7) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2139_213943


namespace NUMINAMATH_CALUDE_parallelogram_area_calculation_l2139_213919

-- Define the parallelogram
def parallelogram_base : ℝ := 32
def parallelogram_height : ℝ := 14

-- Define the area formula for a parallelogram
def parallelogram_area (base height : ℝ) : ℝ := base * height

-- Theorem statement
theorem parallelogram_area_calculation :
  parallelogram_area parallelogram_base parallelogram_height = 448 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_calculation_l2139_213919


namespace NUMINAMATH_CALUDE_pyramid_height_is_two_main_theorem_l2139_213969

/-- A right square pyramid with given properties -/
structure RightSquarePyramid where
  top_side : ℝ
  bottom_side : ℝ
  lateral_area : ℝ
  height : ℝ

/-- The theorem stating the height of the pyramid is 2 -/
theorem pyramid_height_is_two (p : RightSquarePyramid) : p.height = 2 :=
  by
  have h1 : p.top_side = 3 := by sorry
  have h2 : p.bottom_side = 6 := by sorry
  have h3 : p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 := by sorry
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem : ∃ (p : RightSquarePyramid), 
  p.top_side = 3 ∧ 
  p.bottom_side = 6 ∧ 
  p.lateral_area = p.top_side ^ 2 + p.bottom_side ^ 2 ∧
  p.height = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_is_two_main_theorem_l2139_213969


namespace NUMINAMATH_CALUDE_positive_real_floor_product_48_l2139_213906

theorem positive_real_floor_product_48 (x : ℝ) :
  x > 0 ∧ x * ⌊x⌋ = 48 → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_positive_real_floor_product_48_l2139_213906


namespace NUMINAMATH_CALUDE_dog_food_weight_l2139_213963

/-- Proves that given the conditions, each sack of dog food weighs 50 kilograms -/
theorem dog_food_weight 
  (num_dogs : ℕ) 
  (meals_per_day : ℕ) 
  (food_per_meal : ℕ) 
  (num_sacks : ℕ) 
  (days_lasting : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : meals_per_day = 2)
  (h3 : food_per_meal = 250)
  (h4 : num_sacks = 2)
  (h5 : days_lasting = 50) :
  (num_dogs * meals_per_day * food_per_meal * days_lasting) / (1000 * num_sacks) = 50 := by
  sorry

#check dog_food_weight

end NUMINAMATH_CALUDE_dog_food_weight_l2139_213963


namespace NUMINAMATH_CALUDE_inequality_proof_l2139_213973

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  24 * x * y * z ≤ 3 * (x + y) * (y + z) * (z + x) ∧ 
  3 * (x + y) * (y + z) * (z + x) ≤ 8 * (x^3 + y^3 + z^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2139_213973


namespace NUMINAMATH_CALUDE_expression_value_l2139_213998

theorem expression_value (a b : ℝ) (h : 2 * a - b = -1) : 
  b * 2 - a * 2^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2139_213998


namespace NUMINAMATH_CALUDE_trapezoid_area_in_hexagon_triangle_l2139_213922

/-- Given a regular hexagon with an inscribed equilateral triangle, this theorem calculates the area of one of the six congruent trapezoids formed between the hexagon and the triangle. -/
theorem trapezoid_area_in_hexagon_triangle (hexagon_area : ℝ) (triangle_area : ℝ) :
  hexagon_area = 24 →
  triangle_area = 4 →
  (hexagon_area - triangle_area) / 6 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_hexagon_triangle_l2139_213922


namespace NUMINAMATH_CALUDE_parabola_vertex_l2139_213958

/-- The vertex coordinates of the parabola y = x^2 - 6x + 1 are (3, -8) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 1
  ∃ a b : ℝ, a = 3 ∧ b = -8 ∧ ∀ x : ℝ, f x = (x - a)^2 + b :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2139_213958


namespace NUMINAMATH_CALUDE_opposite_reciprocal_abs_sum_l2139_213931

theorem opposite_reciprocal_abs_sum (a b c d m : ℝ) : 
  (a + b = 0) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (abs m = 2) →  -- absolute value of m is 2
  (m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1) :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_abs_sum_l2139_213931


namespace NUMINAMATH_CALUDE_unique_A_value_l2139_213966

/-- A function that checks if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- Theorem stating that 2 is the only single-digit value for A that satisfies the equation -/
theorem unique_A_value : 
  ∃! (A : ℕ), isSingleDigit A ∧ 
    (∃ (B C D : ℕ), isSingleDigit B ∧ isSingleDigit C ∧ isSingleDigit D ∧
      isFourDigit (A * 1000 + 2 * 100 + B * 10 + 2) ∧
      isFourDigit (1000 + C * 100 + 10 + D) ∧
      (A * 1000 + 2 * 100 + B * 10 + 2) + (1000 + C * 100 + 10 + D) = 3333) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_A_value_l2139_213966


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_power_of_three_l2139_213984

theorem infinitely_many_divisible_by_power_of_three (k : ℕ) (hk : k > 0) :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (3^k : ℕ) ∣ (f n)^3 + 10 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_power_of_three_l2139_213984


namespace NUMINAMATH_CALUDE_first_non_divisor_is_seven_l2139_213908

def is_valid_integer (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧ n % 3 ≠ 0 ∧ n % 5 ≠ 0

theorem first_non_divisor_is_seven :
  ∃ (S : Finset ℕ), 
    Finset.card S = 26 ∧ 
    (∀ n ∈ S, is_valid_integer n) ∧
    (∀ k > 5, k < 7 → ∃ n ∈ S, n % k = 0) ∧
    (∀ n ∈ S, n % 7 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_first_non_divisor_is_seven_l2139_213908


namespace NUMINAMATH_CALUDE_monotonicity_condition_l2139_213960

theorem monotonicity_condition (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo 0 (π / 3), Monotone (fun x => Real.sin (ω * x + π / 6))) ↔ ω ∈ Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonicity_condition_l2139_213960


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l2139_213900

/-- Custom operation ⊗ on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' for which the inequality holds for all real x -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l2139_213900


namespace NUMINAMATH_CALUDE_triangle_area_relation_l2139_213985

/-- Given a triangle T with area Δ, and two triangles T' and T'' formed by successive altitudes
    with areas Δ' and Δ'' respectively, prove that if Δ' = 30 and Δ'' = 20, then Δ = 45. -/
theorem triangle_area_relation (Δ Δ' Δ'' : ℝ) : Δ' = 30 → Δ'' = 20 → Δ = 45 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_relation_l2139_213985


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2139_213952

theorem triangle_is_right_angled (a b c : ℝ) : 
  a = 3 ∧ b = 5 ∧ (3 * c^2 - 10 * c = 8) ∧ c > 0 → 
  a^2 + c^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l2139_213952


namespace NUMINAMATH_CALUDE_cone_hemisphere_intersection_volume_l2139_213990

/-- The volume of the common part of a right circular cone and an inscribed hemisphere -/
theorem cone_hemisphere_intersection_volume 
  (m r : ℝ) 
  (h : m > r) 
  (h_pos : r > 0) : 
  ∃ V : ℝ, V = (2 * Real.pi * r^3 / 3) * (1 - (2 * m * r^3) / (m^2 + r^2)^2) := by
  sorry

end NUMINAMATH_CALUDE_cone_hemisphere_intersection_volume_l2139_213990


namespace NUMINAMATH_CALUDE_total_fruit_count_l2139_213970

theorem total_fruit_count (orange_crates : Nat) (oranges_per_crate : Nat)
                          (nectarine_boxes : Nat) (nectarines_per_box : Nat) :
  orange_crates = 12 →
  oranges_per_crate = 150 →
  nectarine_boxes = 16 →
  nectarines_per_box = 30 →
  orange_crates * oranges_per_crate + nectarine_boxes * nectarines_per_box = 2280 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_count_l2139_213970


namespace NUMINAMATH_CALUDE_brianna_cd_purchase_l2139_213924

theorem brianna_cd_purchase (m : ℚ) (c : ℚ) (n : ℚ) (h1 : m > 0) (h2 : c > 0) (h3 : n > 0) :
  (1 / 4 : ℚ) * m = (1 / 4 : ℚ) * n * c →
  m - n * c = 0 :=
by sorry

end NUMINAMATH_CALUDE_brianna_cd_purchase_l2139_213924


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2139_213904

/-- Given a function f(x) = 2x^3 - 3x^2 + a, prove that if its maximum value is 6, then a = 6 -/
theorem max_value_implies_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 - 3 * x^2 + a)
  (h2 : ∃ M, M = 6 ∧ ∀ x, f x ≤ M) : 
  a = 6 := by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2139_213904


namespace NUMINAMATH_CALUDE_connie_grandmother_brother_birth_year_l2139_213978

/-- The year Connie's grandmother's older brother was born -/
def older_brother_birth_year : ℕ := sorry

/-- The year Connie's grandmother's older sister was born -/
def older_sister_birth_year : ℕ := 1936

/-- The year Connie's grandmother was born -/
def grandmother_birth_year : ℕ := 1944

theorem connie_grandmother_brother_birth_year :
  (grandmother_birth_year - older_sister_birth_year = 2 * (older_sister_birth_year - older_brother_birth_year)) →
  older_brother_birth_year = 1932 := by
  sorry

end NUMINAMATH_CALUDE_connie_grandmother_brother_birth_year_l2139_213978


namespace NUMINAMATH_CALUDE_shared_friends_l2139_213975

theorem shared_friends (james_friends : ℕ) (john_friends : ℕ) (combined_list : ℕ) :
  james_friends = 75 →
  john_friends = 3 * james_friends →
  combined_list = 275 →
  james_friends + john_friends - combined_list = 25 := by
sorry

end NUMINAMATH_CALUDE_shared_friends_l2139_213975


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_ten_l2139_213940

theorem no_solution_iff_k_eq_ten (k : ℝ) : 
  (∀ x : ℝ, (3*x + 1 ≠ 0 ∧ 5*x + 4 ≠ 0) → ((2*x - 4)/(3*x + 1) ≠ (2*x - k)/(5*x + 4))) ↔ 
  k = 10 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_ten_l2139_213940


namespace NUMINAMATH_CALUDE_no_hexagon_tiling_l2139_213972

-- Define a grid hexagon
structure GridHexagon where
  -- Add necessary fields to define the hexagon
  -- This is a placeholder and should be adjusted based on the specific hexagon properties
  side_length : ℝ
  diagonal_length : ℝ

-- Define a grid rectangle
structure GridRectangle where
  width : ℕ
  height : ℕ

-- Define the tiling property
def can_tile (r : GridRectangle) (h : GridHexagon) : Prop :=
  -- This is a placeholder for the actual tiling condition
  -- It should represent that the rectangle can be tiled with the hexagons
  sorry

-- The main theorem
theorem no_hexagon_tiling (r : GridRectangle) (h : GridHexagon) : 
  ¬(can_tile r h) := by
  sorry

end NUMINAMATH_CALUDE_no_hexagon_tiling_l2139_213972


namespace NUMINAMATH_CALUDE_expression_evaluation_l2139_213937

theorem expression_evaluation :
  let a : ℤ := -4
  (4 * a^2 - 3*a) - (2 * a^2 + a - 1) + (2 - a^2 + 4*a) = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2139_213937


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2139_213977

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2139_213977


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2139_213954

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2139_213954


namespace NUMINAMATH_CALUDE_no_integer_sqrt_difference_150_l2139_213917

theorem no_integer_sqrt_difference_150 :
  (∃ (x : ℤ), x - Real.sqrt x = 20) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 30) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 110) ∧
  (∀ (x : ℤ), x - Real.sqrt x ≠ 150) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 600) := by
  sorry

#check no_integer_sqrt_difference_150

end NUMINAMATH_CALUDE_no_integer_sqrt_difference_150_l2139_213917


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l2139_213964

/-- Calculates the expected value of a coin flip experiment -/
def expected_value (coin_values : List ℚ) (probability : ℚ) : ℚ :=
  (coin_values.sum * probability)

/-- The main theorem: expected value of the coin flip experiment -/
theorem coin_flip_expected_value :
  let coin_values : List ℚ := [1, 5, 10, 50, 100]
  let probability : ℚ := 1/2
  expected_value coin_values probability = 83 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l2139_213964


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2139_213949

/-- Given a parallelogram with adjacent sides of lengths 3s and s units, 
    forming a 60-degree angle, and having an area of 9√3 square units, 
    prove that s = √6. -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →
  let adjacent_side1 := 3 * s
  let adjacent_side2 := s
  let angle := Real.pi / 3  -- 60 degrees in radians
  let area := 9 * Real.sqrt 3
  area = adjacent_side1 * adjacent_side2 * Real.sin angle →
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2139_213949


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2139_213982

-- Problem 1
theorem problem_1 : Real.sqrt 32 + 3 * Real.sqrt (1/2) - Real.sqrt 2 = (9 * Real.sqrt 2) / 2 := by sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 50 * Real.sqrt 32) / Real.sqrt 8 - 4 * Real.sqrt 2 = 6 * Real.sqrt 2 := by sorry

-- Problem 3
theorem problem_3 : (Real.sqrt 5 - 3)^2 + (Real.sqrt 11 + 3) * (Real.sqrt 11 - 3) = 16 - 6 * Real.sqrt 5 := by sorry

-- Problem 4
theorem problem_4 : (2 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt 3 - 12 * Real.sqrt (1/2) = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2139_213982


namespace NUMINAMATH_CALUDE_ball_arrangements_l2139_213996

-- Define the word structure
def Word := String

-- Define a function to count distinct arrangements
def countDistinctArrangements (w : Word) : ℕ := sorry

-- Theorem statement
theorem ball_arrangements :
  let ball : Word := "BALL"
  countDistinctArrangements ball = 12 := by sorry

end NUMINAMATH_CALUDE_ball_arrangements_l2139_213996


namespace NUMINAMATH_CALUDE_three_digit_sum_27_l2139_213992

theorem three_digit_sum_27 : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n / 100 + (n / 10) % 10 + n % 10 = 27) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_27_l2139_213992


namespace NUMINAMATH_CALUDE_max_value_of_x2_plus_y2_l2139_213911

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2*x - 2*y + 2) :
  x^2 + y^2 ≤ 6 + 4 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 6 + 4 * Real.sqrt 2 ∧ x₀^2 + y₀^2 = 2*x₀ - 2*y₀ + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_x2_plus_y2_l2139_213911


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l2139_213987

/-- Given three points in 2D space -/
def point1 : ℝ × ℝ := (4, -3)
def point2 (b : ℝ) : ℝ × ℝ := (2*b + 1, 5)
def point3 (b : ℝ) : ℝ × ℝ := (-b + 3, 1)

/-- Function to check if three points are collinear -/
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

/-- Theorem stating that b = 1/4 when the given points are collinear -/
theorem collinear_points_b_value :
  collinear point1 (point2 b) (point3 b) → b = 1/4 := by sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l2139_213987


namespace NUMINAMATH_CALUDE_probability_geometry_second_draw_l2139_213903

/-- Represents the set of questions in the problem -/
structure QuestionSet where
  total : ℕ
  algebra : ℕ
  geometry : ℕ
  algebra_first_draw : Prop

/-- The probability of selecting a geometry question on the second draw,
    given an algebra question was selected on the first draw -/
def conditional_probability (qs : QuestionSet) : ℚ :=
  qs.geometry / (qs.total - 1)

/-- The main theorem to prove -/
theorem probability_geometry_second_draw 
  (qs : QuestionSet) 
  (h1 : qs.total = 5) 
  (h2 : qs.algebra = 3) 
  (h3 : qs.geometry = 2) 
  (h4 : qs.algebra_first_draw) : 
  conditional_probability qs = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_geometry_second_draw_l2139_213903


namespace NUMINAMATH_CALUDE_solution_set_properties_l2139_213934

-- Define the set M
def M : Set ℝ := {x | 3 - 2*x < 0}

-- Theorem statement
theorem solution_set_properties :
  0 ∉ M ∧ 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_solution_set_properties_l2139_213934


namespace NUMINAMATH_CALUDE_even_difference_of_coefficients_l2139_213941

theorem even_difference_of_coefficients (a₁ a₂ b₁ b₂ m n : ℤ) : 
  a₁ ≠ a₂ →
  m ≠ n →
  (m^2 + a₁*m + b₁ = n^2 + a₂*n + b₂) →
  (m^2 + a₂*m + b₂ = n^2 + a₁*n + b₁) →
  ∃ k : ℤ, a₁ - a₂ = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_even_difference_of_coefficients_l2139_213941


namespace NUMINAMATH_CALUDE_jerry_trays_capacity_l2139_213913

def jerry_trays (trays_table1 trays_table2 num_trips : ℕ) : ℕ :=
  (trays_table1 + trays_table2) / num_trips

theorem jerry_trays_capacity :
  jerry_trays 9 7 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_trays_capacity_l2139_213913


namespace NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2139_213915

theorem smallest_difference_in_triangle (XZ XY YZ : ℕ) : 
  XZ + XY + YZ = 3030 →
  XZ < XY →
  XY ≤ YZ →
  ∃ k : ℕ, XY = 5 * k →
  ∀ XZ' XY' YZ' : ℕ, 
    XZ' + XY' + YZ' = 3030 →
    XZ' < XY' →
    XY' ≤ YZ' →
    (∃ k' : ℕ, XY' = 5 * k') →
    XY - XZ ≤ XY' - XZ' :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_in_triangle_l2139_213915


namespace NUMINAMATH_CALUDE_minsu_marbles_left_l2139_213994

/-- Calculates the number of marbles left after distribution -/
def marblesLeft (totalMarbles : ℕ) (largeBulk smallBulk : ℕ) (largeBoxes smallBoxes : ℕ) : ℕ :=
  totalMarbles - (largeBulk * largeBoxes + smallBulk * smallBoxes)

/-- Theorem stating the number of marbles left after Minsu's distribution -/
theorem minsu_marbles_left :
  marblesLeft 240 35 6 4 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_minsu_marbles_left_l2139_213994


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2139_213995

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + 5 / 99 + 3 / 9999 = 910 / 3333 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2139_213995


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2139_213968

-- Define the property of being a nonprime integer greater than 1 with no prime factor less than 15
def is_valid_number (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n ∧ ∀ p : ℕ, Nat.Prime p → p < 15 → ¬ p ∣ n

-- State the theorem
theorem smallest_valid_number :
  ∃ n : ℕ, is_valid_number n ∧ ∀ m : ℕ, is_valid_number m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2139_213968


namespace NUMINAMATH_CALUDE_three_tangent_lines_l2139_213926

/-- A line that passes through the point (0, 2) and has only one common point with the parabola y^2 = 8x -/
structure TangentLine where
  -- The slope of the line (None if vertical)
  slope : Option ℝ
  -- Condition that the line passes through (0, 2)
  passes_through_point : True
  -- Condition that the line has only one common point with y^2 = 8x
  single_intersection : True

/-- The number of lines passing through (0, 2) with only one common point with y^2 = 8x -/
def count_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem three_tangent_lines : count_tangent_lines = 3 := by sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l2139_213926


namespace NUMINAMATH_CALUDE_max_b_plus_c_l2139_213921

theorem max_b_plus_c (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c = b + 2) :
  b + c ≤ 18 ∧ ∃ (b' c' : ℕ), b' + c' = 18 ∧ a > b' ∧ a + b' = 18 ∧ c' = b' + 2 := by
  sorry

end NUMINAMATH_CALUDE_max_b_plus_c_l2139_213921


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l2139_213942

theorem sin_plus_cos_value (A : Real) (h : Real.sin (2 * A) = 2/3) :
  Real.sin A + Real.cos A = Real.sqrt (5/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l2139_213942
