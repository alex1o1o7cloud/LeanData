import Mathlib

namespace NUMINAMATH_CALUDE_berry_cobbler_problem_l2974_297427

theorem berry_cobbler_problem (total_needed : ℕ) (blueberries : ℕ) (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : blueberries = 8)
  (h3 : to_buy = 9) :
  total_needed - (blueberries + to_buy) = 4 := by
  sorry

end NUMINAMATH_CALUDE_berry_cobbler_problem_l2974_297427


namespace NUMINAMATH_CALUDE_consecutive_digit_product_is_square_l2974_297454

/-- Represents a 16-digit positive integer -/
def SixteenDigitInteger := { n : ℕ // 10^15 ≤ n ∧ n < 10^16 }

/-- Extracts a consecutive sequence of digits from a natural number -/
def extractDigitSequence (n : ℕ) (start : ℕ) (len : ℕ) : ℕ := sorry

/-- Checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- Main theorem: For any 16-digit positive integer, there exists a consecutive
    sequence of digits whose product is a perfect square -/
theorem consecutive_digit_product_is_square (A : SixteenDigitInteger) : 
  ∃ start len, isPerfectSquare (extractDigitSequence A.val start len) :=
sorry

end NUMINAMATH_CALUDE_consecutive_digit_product_is_square_l2974_297454


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l2974_297497

/-- Proves that if selling at 75% of cost price results in Rs. 600, 
    then selling at 125% of cost price results in Rs. 1000 -/
theorem shopkeeper_pricing (CP : ℝ) : 
  CP * 0.75 = 600 → CP * 1.25 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_pricing_l2974_297497


namespace NUMINAMATH_CALUDE_primitive_decomposition_existence_l2974_297413

/-- A decomposition of a square into rectangles. -/
structure SquareDecomposition :=
  (n : ℕ)  -- number of rectangles
  (is_finite : n > 0)
  (parallel_sides : Bool)
  (is_primitive : Bool)

/-- Predicate for a valid primitive square decomposition. -/
def valid_primitive_decomposition (d : SquareDecomposition) : Prop :=
  d.parallel_sides ∧ d.is_primitive

/-- Theorem stating for which n a primitive decomposition exists. -/
theorem primitive_decomposition_existence :
  ∀ n : ℕ, (∃ d : SquareDecomposition, d.n = n ∧ valid_primitive_decomposition d) ↔ (n = 5 ∨ n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_primitive_decomposition_existence_l2974_297413


namespace NUMINAMATH_CALUDE_expression_value_l2974_297485

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)    -- absolute value of m is 2
  : m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2974_297485


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2974_297479

/-- A pocket containing balls of two colors -/
structure Pocket where
  red : ℕ
  white : ℕ

/-- The possible outcomes when drawing two balls -/
inductive Outcome
  | TwoRed
  | OneRedOneWhite
  | TwoWhite

/-- Define the events -/
def ExactlyOneWhite (o : Outcome) : Prop :=
  o = Outcome.OneRedOneWhite

def ExactlyTwoWhite (o : Outcome) : Prop :=
  o = Outcome.TwoWhite

/-- The probability of an outcome given a pocket -/
def probability (p : Pocket) (o : Outcome) : ℚ :=
  match o with
  | Outcome.TwoRed => (p.red * (p.red - 1)) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.OneRedOneWhite => (2 * p.red * p.white) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.TwoWhite => (p.white * (p.white - 1)) / ((p.red + p.white) * (p.red + p.white - 1))

theorem mutually_exclusive_not_contradictory (p : Pocket) (h : p.red = 2 ∧ p.white = 2) :
  (∀ o : Outcome, ¬(ExactlyOneWhite o ∧ ExactlyTwoWhite o)) ∧ 
  (probability p Outcome.OneRedOneWhite + probability p Outcome.TwoWhite < 1) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2974_297479


namespace NUMINAMATH_CALUDE_promotional_price_calculation_l2974_297428

/-- The cost of one chocolate at the store with the promotion -/
def promotional_price : ℚ := 2

theorem promotional_price_calculation :
  let chocolates_per_week : ℕ := 2
  let weeks : ℕ := 3
  let local_price : ℚ := 3
  let total_savings : ℚ := 6
  promotional_price = (chocolates_per_week * weeks * local_price - total_savings) / (chocolates_per_week * weeks) :=
by sorry

end NUMINAMATH_CALUDE_promotional_price_calculation_l2974_297428


namespace NUMINAMATH_CALUDE_ball_max_height_l2974_297453

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem statement
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 40 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l2974_297453


namespace NUMINAMATH_CALUDE_existence_of_suitable_set_l2974_297478

theorem existence_of_suitable_set (ε : Real) (h_ε : 0 < ε ∧ ε < 1) :
  ∃ N₀ : ℕ, ∀ N ≥ N₀, ∃ S : Finset ℕ,
    (S.card : ℝ) ≥ ε * N ∧
    (∀ x ∈ S, x ≤ N) ∧
    (∀ x ∈ S, Nat.gcd x (S.sum id) > 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_suitable_set_l2974_297478


namespace NUMINAMATH_CALUDE_smallest_mustang_is_12_inches_l2974_297439

/-- The length of the smallest Mustang model given the full-size and scaling factors -/
def smallest_mustang_length (full_size : ℝ) (mid_size_factor : ℝ) (smallest_factor : ℝ) : ℝ :=
  full_size * mid_size_factor * smallest_factor

/-- Theorem stating that the smallest Mustang model is 12 inches long -/
theorem smallest_mustang_is_12_inches :
  smallest_mustang_length 240 (1/10) (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_mustang_is_12_inches_l2974_297439


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_consecutive_list_l2974_297404

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_consecutive_list :
  let D := consecutive_integers (-4) 12
  let positives := positive_integers D
  range positives = 6 := by sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_consecutive_list_l2974_297404


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2974_297420

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1a3 : a 1 * a 3 = 36)
  (h_a4 : a 4 = 54) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2974_297420


namespace NUMINAMATH_CALUDE_response_change_difference_l2974_297417

/-- Represents the percentages of student responses --/
structure ResponsePercentages where
  yes : ℝ
  no : ℝ
  undecided : ℝ

/-- The problem statement --/
theorem response_change_difference
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 30)
  (h_initial_undecided : initial.undecided = 30)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 10)
  (h_final_undecided : final.undecided = 30) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 20 :=
sorry

end NUMINAMATH_CALUDE_response_change_difference_l2974_297417


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2974_297437

-- Define set A
def A : Set ℝ := {x | x^2 - x = 0}

-- Define set B
def B : Set ℝ := {-1, 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2974_297437


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2974_297499

-- Problem 1
theorem problem_1 : 
  (Real.sqrt 24 - Real.sqrt (1/2)) - (Real.sqrt (1/8) + Real.sqrt 6) = Real.sqrt 6 - (3 * Real.sqrt 2) / 4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2974_297499


namespace NUMINAMATH_CALUDE_mrs_hilt_books_l2974_297475

/-- The number of books Mrs. Hilt bought -/
def num_books : ℕ := sorry

/-- The cost per book when buying -/
def cost_per_book : ℕ := 11

/-- The price per book when selling -/
def price_per_book : ℕ := 25

/-- The difference between total sold amount and total paid amount -/
def profit : ℕ := 210

theorem mrs_hilt_books :
  num_books * price_per_book - num_books * cost_per_book = profit ∧ num_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_books_l2974_297475


namespace NUMINAMATH_CALUDE_back_seat_capacity_is_nine_l2974_297471

/-- Represents the seating capacity of a bus -/
structure BusSeats where
  leftSeats : Nat
  rightSeats : Nat
  peoplePerSeat : Nat
  totalCapacity : Nat

/-- Calculates the number of people who can sit at the back seat of the bus -/
def backSeatCapacity (bus : BusSeats) : Nat :=
  bus.totalCapacity - (bus.leftSeats + bus.rightSeats) * bus.peoplePerSeat

/-- Theorem stating the back seat capacity of the given bus configuration -/
theorem back_seat_capacity_is_nine :
  let bus : BusSeats := {
    leftSeats := 15,
    rightSeats := 12,
    peoplePerSeat := 3,
    totalCapacity := 90
  }
  backSeatCapacity bus = 9 := by sorry

end NUMINAMATH_CALUDE_back_seat_capacity_is_nine_l2974_297471


namespace NUMINAMATH_CALUDE_g_3_value_l2974_297474

/-- A linear function satisfying certain conditions -/
def g (x : ℝ) : ℝ := sorry

/-- The inverse of g -/
def g_inv (x : ℝ) : ℝ := sorry

/-- g is a linear function -/
axiom g_linear : ∃ (c d : ℝ), ∀ x, g x = c * x + d

/-- g satisfies the equation g(x) = 5g^(-1)(x) + 3 -/
axiom g_equation : ∀ x, g x = 5 * g_inv x + 3

/-- g(2) = 5 -/
axiom g_2_eq_5 : g 2 = 5

/-- Main theorem: g(3) = 3√5 + (3√5)/(√5 + 5) -/
theorem g_3_value : g 3 = 3 * Real.sqrt 5 + (3 * Real.sqrt 5) / (Real.sqrt 5 + 5) := by sorry

end NUMINAMATH_CALUDE_g_3_value_l2974_297474


namespace NUMINAMATH_CALUDE_gamma_value_l2974_297468

theorem gamma_value (γ δ : ℂ) : 
  (γ + δ).re > 0 →
  (Complex.I * (γ - 3 * δ)).re > 0 →
  δ = 4 + 3 * Complex.I →
  γ = 16 - 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_gamma_value_l2974_297468


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l2974_297481

theorem mariela_get_well_cards (total : ℝ) (from_home : ℝ) (from_country : ℝ)
  (h1 : total = 403.0)
  (h2 : from_home = 287.0)
  (h3 : total = from_home + from_country) :
  from_country = 116.0 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l2974_297481


namespace NUMINAMATH_CALUDE_reciprocal_quotient_not_always_one_l2974_297467

theorem reciprocal_quotient_not_always_one :
  ¬ (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → a * b = 1 → a / b = 1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_quotient_not_always_one_l2974_297467


namespace NUMINAMATH_CALUDE_runner_journey_time_l2974_297477

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfSpeed : ℝ
  secondHalfSpeed : ℝ
  firstHalfTime : ℝ
  secondHalfTime : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem runner_journey_time (j : RunnerJourney) 
  (h1 : j.totalDistance = 40)
  (h2 : j.secondHalfSpeed = j.firstHalfSpeed / 2)
  (h3 : j.secondHalfTime = j.firstHalfTime + 5)
  (h4 : j.firstHalfTime = (j.totalDistance / 2) / j.firstHalfSpeed)
  (h5 : j.secondHalfTime = (j.totalDistance / 2) / j.secondHalfSpeed) :
  j.secondHalfTime = 10 := by
  sorry

end NUMINAMATH_CALUDE_runner_journey_time_l2974_297477


namespace NUMINAMATH_CALUDE_circle_circumference_ratio_l2974_297418

theorem circle_circumference_ratio (r_large r_small : ℝ) (h : r_large / r_small = 3 / 2) :
  (2 * Real.pi * r_large) / (2 * Real.pi * r_small) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_ratio_l2974_297418


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2974_297462

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (x + 2) * (x - 1) - 3 * x * (x + 3) = -2 * x^2 - 8 * x - 2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : (a + 3) * (a^2 + 9) * (a - 3) = a^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2974_297462


namespace NUMINAMATH_CALUDE_car_speed_second_hour_car_speed_second_hour_value_l2974_297407

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 100)
  (h2 : average_speed = 90) : 
  (speed_first_hour + (2 * average_speed - speed_first_hour)) / 2 = average_speed := by
  sorry

/-- The speed of the car in the second hour is 80 km/h. -/
theorem car_speed_second_hour_value : 
  ∃ (speed_second_hour : ℝ), 
    speed_second_hour = 80 ∧ 
    (100 + speed_second_hour) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_car_speed_second_hour_value_l2974_297407


namespace NUMINAMATH_CALUDE_ordered_pair_satisfies_equation_l2974_297403

theorem ordered_pair_satisfies_equation :
  let a : ℝ := 9
  let b : ℝ := -4
  (Real.sqrt (25 - 16 * Real.cos (π / 3)) = a - b * (1 / Real.cos (π / 3))) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pair_satisfies_equation_l2974_297403


namespace NUMINAMATH_CALUDE_age_problem_l2974_297400

theorem age_problem :
  ∃ (x y z w v : ℕ),
    x + y + z = 74 ∧
    x = 7 * w ∧
    y = 2 * w + 2 * v ∧
    z = 2 * w + 3 * v ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ v > 0 ∧
    x = 28 ∧ y = 20 ∧ z = 26 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l2974_297400


namespace NUMINAMATH_CALUDE_equation_solution_l2974_297441

theorem equation_solution (a b c : ℤ) : 
  (∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  ((a = 10 ∧ b = -11 ∧ c = -11) ∨ (a = 14 ∧ b = -13 ∧ c = -13)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2974_297441


namespace NUMINAMATH_CALUDE_consecutive_non_primes_under_50_l2974_297409

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem consecutive_non_primes_under_50 :
  ∃ (a b c d e : ℕ),
    a < 50 ∧ b < 50 ∧ c < 50 ∧ d < 50 ∧ e < 50 ∧
    ¬(is_prime a) ∧ ¬(is_prime b) ∧ ¬(is_prime c) ∧ ¬(is_prime d) ∧ ¬(is_prime e) ∧
    b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
    e = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_under_50_l2974_297409


namespace NUMINAMATH_CALUDE_compound_weight_l2974_297484

/-- Given a compound with a molecular weight of 1098 and 9 moles of this compound,
    prove that the total weight is 9882 grams. -/
theorem compound_weight (molecular_weight : ℕ) (moles : ℕ) : 
  molecular_weight = 1098 → moles = 9 → molecular_weight * moles = 9882 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l2974_297484


namespace NUMINAMATH_CALUDE_largest_sample_is_433_l2974_297464

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  totalItems : ℕ
  sampleSize : ℕ
  knownSample : ℕ
  firstItem : ℕ
  interval : ℕ

/-- Calculates the largest sampled number in a systematic sampling scheme. -/
def largestSampledNumber (s : SystematicSampling) : ℕ :=
  ((s.firstItem - 1 + (s.sampleSize - 1) * s.interval) % s.totalItems) + 1

/-- Theorem stating that for the given systematic sampling scheme,
    the largest sampled number is 433. -/
theorem largest_sample_is_433 :
  ∃ s : SystematicSampling,
    s.totalItems = 360 ∧
    s.sampleSize = 30 ∧
    s.knownSample = 105 ∧
    s.firstItem = 97 ∧
    s.interval = 12 ∧
    largestSampledNumber s = 433 :=
  sorry

end NUMINAMATH_CALUDE_largest_sample_is_433_l2974_297464


namespace NUMINAMATH_CALUDE_angle_AFE_measure_l2974_297488

-- Define the points
variable (A B C D E F : Point)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the relationship AB = 2BC
def AB_twice_BC (A B C : Point) : Prop := sorry

-- Define E on the opposite half-plane from A with respect to CD
def E_opposite_halfplane (A C D E : Point) : Prop := sorry

-- Define angle CDE = 120°
def angle_CDE_120 (C D E : Point) : Prop := sorry

-- Define F as midpoint of AD
def F_midpoint_AD (A D F : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_AFE_measure
  (h_rect : is_rectangle A B C D)
  (h_AB_BC : AB_twice_BC A B C)
  (h_E_opp : E_opposite_halfplane A C D E)
  (h_CDE : angle_CDE_120 C D E)
  (h_F_mid : F_midpoint_AD A D F) :
  angle_measure A F E = 150 := by sorry

end NUMINAMATH_CALUDE_angle_AFE_measure_l2974_297488


namespace NUMINAMATH_CALUDE_exponential_inequality_l2974_297412

theorem exponential_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  2^x * x + 2^y * y ≥ 2^y * x + 2^x * y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2974_297412


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2974_297489

theorem complex_modulus_problem (x y : ℝ) (h : (Complex.I : ℂ) / (1 + Complex.I) = x + y * Complex.I) : 
  Complex.abs (x - y * Complex.I) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2974_297489


namespace NUMINAMATH_CALUDE_volume_ratio_is_19_89_l2974_297423

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  origin : Point3D
  edge_length : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point1 : Point3D
  point2 : Point3D
  point3 : Point3D

def cube : Cube := {
  origin := { x := 0, y := 0, z := 0 }
  edge_length := 6
}

def point_A : Point3D := { x := 0, y := 0, z := 0 }
def point_H : Point3D := { x := 6, y := 6, z := 2 }
def point_F : Point3D := { x := 6, y := 6, z := 3 }

def cutting_plane : Plane := {
  point1 := point_A
  point2 := point_H
  point3 := point_F
}

/-- Calculates the volume of a part of the cube cut by the plane -/
def volume_of_part (c : Cube) (p : Plane) : ℝ := sorry

/-- Theorem: The volume ratio of the two parts is 19:89 -/
theorem volume_ratio_is_19_89 (c : Cube) (p : Plane) : 
  let v1 := volume_of_part c p
  let v2 := c.edge_length ^ 3 - v1
  v1 / v2 = 19 / 89 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_19_89_l2974_297423


namespace NUMINAMATH_CALUDE_f_six_value_l2974_297498

/-- A function f from integers to integers satisfying f(n) = f(n-1) - n for all n -/
def RecursiveFunction (f : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, f n = f (n - 1) - n

theorem f_six_value (f : ℤ → ℤ) (k : ℤ) 
  (h1 : RecursiveFunction f) 
  (h2 : f k = 14) : 
  f 6 = f (k - (k - 6)) - 7 := by
  sorry

end NUMINAMATH_CALUDE_f_six_value_l2974_297498


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2974_297469

theorem inequality_equivalence (x y : ℝ) : 
  Real.sqrt (x^2 - 2*x*y) > Real.sqrt (1 - y^2) ↔ 
  ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2974_297469


namespace NUMINAMATH_CALUDE_multiplication_of_decimals_l2974_297463

theorem multiplication_of_decimals : 3.6 * 0.3 = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_decimals_l2974_297463


namespace NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_bounds_l2974_297470

theorem y_squared_plus_7y_plus_12_bounds (y : ℝ) (h : y^2 - 7*y + 12 < 0) : 
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
sorry

end NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_bounds_l2974_297470


namespace NUMINAMATH_CALUDE_alpha_range_l2974_297447

theorem alpha_range (α : Real) (k : Int) 
  (h1 : Real.sin α > 0)
  (h2 : Real.cos α < 0)
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k, (α / 3 ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + Real.pi / 3)) ∨
       (α / 3 ∈ Set.Ioo (2 * k * Real.pi + 5 * Real.pi / 6) (2 * k * Real.pi + Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_range_l2974_297447


namespace NUMINAMATH_CALUDE_intersection_points_count_l2974_297448

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

def g (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem intersection_points_count :
  ∃ (a b : ℝ), a ≠ b ∧ f a = g a ∧ f b = g b ∧
  ∀ (x : ℝ), f x = g x → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2974_297448


namespace NUMINAMATH_CALUDE_union_equality_implies_a_equals_three_l2974_297495

-- Define the sets A and B
def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

-- State the theorem
theorem union_equality_implies_a_equals_three (a : ℝ) : A ∪ B a = A → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_equals_three_l2974_297495


namespace NUMINAMATH_CALUDE_inheritance_problem_l2974_297444

theorem inheritance_problem (total_inheritance : ℕ) (additional_amount : ℕ) : 
  total_inheritance = 46800 →
  additional_amount = 1950 →
  ∃ (original_children : ℕ),
    original_children > 2 ∧
    (total_inheritance / original_children + additional_amount = total_inheritance / (original_children - 2)) ∧
    original_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l2974_297444


namespace NUMINAMATH_CALUDE_novel_reading_time_difference_l2974_297487

/-- The number of pages in the novel -/
def pages : ℕ := 760

/-- The time in seconds Bob takes to read one page -/
def bob_time : ℕ := 45

/-- The time in seconds Chandra takes to read one page -/
def chandra_time : ℕ := 30

/-- The difference in reading time between Bob and Chandra for the entire novel -/
def reading_time_difference : ℕ := pages * bob_time - pages * chandra_time

theorem novel_reading_time_difference :
  reading_time_difference = 11400 := by
  sorry

end NUMINAMATH_CALUDE_novel_reading_time_difference_l2974_297487


namespace NUMINAMATH_CALUDE_highest_winner_number_l2974_297457

/-- Represents a single-elimination tournament with wrestlers having qualification numbers. -/
structure WrestlingTournament where
  num_wrestlers : ℕ
  can_win : ℕ → ℕ → Prop

/-- The conditions of our specific tournament. -/
def our_tournament : WrestlingTournament where
  num_wrestlers := 512
  can_win := fun a b => b ≤ a + 2

/-- The number of rounds in a single-elimination tournament. -/
def num_rounds (t : WrestlingTournament) : ℕ :=
  Nat.log 2 t.num_wrestlers

/-- The highest possible qualification number for the winner. -/
def max_winner_number (t : WrestlingTournament) : ℕ :=
  1 + 2 * num_rounds t

theorem highest_winner_number (t : WrestlingTournament) :
  t = our_tournament →
  max_winner_number t = 18 :=
by sorry

end NUMINAMATH_CALUDE_highest_winner_number_l2974_297457


namespace NUMINAMATH_CALUDE_new_class_mean_approx_67_percent_l2974_297410

/-- Represents the class statistics for Mr. Thompson's chemistry class -/
structure ChemistryClass where
  total_students : ℕ
  group1_students : ℕ
  group1_average : ℚ
  group2_students : ℕ
  group2_average : ℚ
  group3_students : ℕ
  group3_average : ℚ

/-- Calculates the new class mean for Mr. Thompson's chemistry class -/
def new_class_mean (c : ChemistryClass) : ℚ :=
  (c.group1_students * c.group1_average + 
   c.group2_students * c.group2_average + 
   c.group3_students * c.group3_average) / c.total_students

/-- Theorem stating that the new class mean is approximately 67% -/
theorem new_class_mean_approx_67_percent (c : ChemistryClass) 
  (h1 : c.total_students = 60)
  (h2 : c.group1_students = 50)
  (h3 : c.group1_average = 65/100)
  (h4 : c.group2_students = 8)
  (h5 : c.group2_average = 85/100)
  (h6 : c.group3_students = 2)
  (h7 : c.group3_average = 55/100) :
  ∃ ε > 0, |new_class_mean c - 67/100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_new_class_mean_approx_67_percent_l2974_297410


namespace NUMINAMATH_CALUDE_fraction_simplification_l2974_297438

theorem fraction_simplification :
  ((3^1005)^2 - (3^1003)^2) / ((3^1004)^2 - (3^1002)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2974_297438


namespace NUMINAMATH_CALUDE_division_problem_l2974_297405

theorem division_problem : (120 : ℚ) / ((6 / 2) + 4) = 17 + 1/7 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2974_297405


namespace NUMINAMATH_CALUDE_max_min_constrained_optimization_l2974_297459

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 2) + Real.sqrt (y - 3) = 3

-- Define the objective function
def objective (x y : ℝ) : ℝ :=
  x - 2*y

-- Theorem statement
theorem max_min_constrained_optimization :
  ∃ (x_max y_max x_min y_min : ℝ),
    constraint x_max y_max ∧
    constraint x_min y_min ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    (∀ x y, constraint x y → objective x_min y_min ≤ objective x y) ∧
    x_max = 11 ∧ y_max = 3 ∧
    x_min = 2 ∧ y_min = 12 ∧
    objective x_max y_max = 5 ∧
    objective x_min y_min = -22 :=
  sorry

end NUMINAMATH_CALUDE_max_min_constrained_optimization_l2974_297459


namespace NUMINAMATH_CALUDE_equation_solution_l2974_297455

theorem equation_solution : ∃! x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 1 ∧ x = -11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2974_297455


namespace NUMINAMATH_CALUDE_cornbread_pieces_count_l2974_297445

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a piece of cornbread --/
structure PieceDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of whole pieces that can be cut from a pan --/
def maxPieces (pan : PanDimensions) (piece : PieceDimensions) (margin : ℕ) : ℕ :=
  ((pan.length / piece.length) * ((pan.width - margin) / piece.width))

/-- Theorem stating that the maximum number of pieces for the given dimensions is 72 --/
theorem cornbread_pieces_count :
  let pan := PanDimensions.mk 24 20
  let piece := PieceDimensions.mk 3 2
  let margin := 1
  maxPieces pan piece margin = 72 := by
  sorry


end NUMINAMATH_CALUDE_cornbread_pieces_count_l2974_297445


namespace NUMINAMATH_CALUDE_probability_three_primes_is_correct_l2974_297446

def num_dice : ℕ := 7
def faces_per_die : ℕ := 10
def num_primes_per_die : ℕ := 4

def probability_exactly_three_primes : ℚ :=
  (num_dice.choose 3) *
  (num_primes_per_die / faces_per_die) ^ 3 *
  ((faces_per_die - num_primes_per_die) / faces_per_die) ^ (num_dice - 3)

theorem probability_three_primes_is_correct :
  probability_exactly_three_primes = 9072 / 31250 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_is_correct_l2974_297446


namespace NUMINAMATH_CALUDE_mumblian_language_word_count_l2974_297451

/-- The number of letters in the Mumblian alphabet -/
def alphabet_size : ℕ := 5

/-- The maximum word length in the Mumblian language -/
def max_word_length : ℕ := 3

/-- The number of words of a given length in the Mumblian language -/
def words_of_length (n : ℕ) : ℕ := 
  if n > 0 ∧ n ≤ max_word_length then alphabet_size ^ n else 0

/-- The total number of words in the Mumblian language -/
def total_words : ℕ := 
  (words_of_length 1) + (words_of_length 2) + (words_of_length 3)

theorem mumblian_language_word_count : total_words = 155 := by
  sorry

end NUMINAMATH_CALUDE_mumblian_language_word_count_l2974_297451


namespace NUMINAMATH_CALUDE_number_scientific_notation_equality_l2974_297449

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff : 1 ≤ coefficient
  coeff_lt_ten : coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 11090000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.109
    exponent := 7
    one_le_coeff := by sorry
    coeff_lt_ten := by sorry }

theorem number_scientific_notation_equality :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by
  sorry

end NUMINAMATH_CALUDE_number_scientific_notation_equality_l2974_297449


namespace NUMINAMATH_CALUDE_nine_qualified_products_possible_l2974_297419

/-- The probability of success (pass rate) -/
def p : ℝ := 0.9

/-- The number of trials (products inspected) -/
def n : ℕ := 10

/-- The number of successes (qualified products) we're interested in -/
def k : ℕ := 9

/-- The binomial probability of k successes in n trials with probability p -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem nine_qualified_products_possible : binomialProbability n k p > 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_qualified_products_possible_l2974_297419


namespace NUMINAMATH_CALUDE_valid_x_values_l2974_297406

-- Define the property for x
def is_valid_x (x : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    x ^ 2 = 2525000000 + a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f * 1 + 89 ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10

-- State the theorem
theorem valid_x_values : 
  ∀ x : ℕ, is_valid_x x ↔ (x = 502567 ∨ x = 502583) :=
sorry

end NUMINAMATH_CALUDE_valid_x_values_l2974_297406


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l2974_297421

theorem prob_at_least_one_woman (men women selected : ℕ) :
  men = 9 →
  women = 5 →
  selected = 3 →
  (1 - (Nat.choose men selected) / (Nat.choose (men + women) selected) : ℚ) = 23/30 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l2974_297421


namespace NUMINAMATH_CALUDE_rectangular_box_dimension_sum_square_l2974_297472

/-- Given a rectangular box with dimensions a, b, c, where a = b + c + 10,
    prove that the square of the sum of dimensions is equal to 4(b+c)^2 + 40(b+c) + 100 -/
theorem rectangular_box_dimension_sum_square (b c : ℝ) :
  let a : ℝ := b + c + 10
  let D : ℝ := a + b + c
  D^2 = 4*(b+c)^2 + 40*(b+c) + 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_dimension_sum_square_l2974_297472


namespace NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l2974_297426

/-- Given two perpendicular lines l₁ and l₂, prove that the minimum value of |ab| is 2 -/
theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, a^2 * x + y + 2 = 0 → b * x - (a^2 + 1) * y - 1 = 0 → 
   (a^2 * 1) * (b / (a^2 + 1)) = -1) →
  ∃ (min : ℝ), min = 2 ∧ ∀ a' b' : ℝ, 
    (∀ x y : ℝ, (a')^2 * x + y + 2 = 0 → b' * x - ((a')^2 + 1) * y - 1 = 0 → 
     ((a')^2 * 1) * (b' / ((a')^2 + 1)) = -1) →
    |a' * b'| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_abs_ab_for_perpendicular_lines_l2974_297426


namespace NUMINAMATH_CALUDE_log_approximation_l2974_297435

-- Define the base of the logarithm
def base : ℝ := 8

-- Define the given logarithmic value
def log_value : ℝ := 2.75

-- Define the approximate result
def approx_result : ℝ := 215

-- Define a tolerance for the approximation
def tolerance : ℝ := 0.1

-- Theorem statement
theorem log_approximation (y : ℝ) (h : Real.log y / Real.log base = log_value) :
  |y - approx_result| < tolerance :=
sorry

end NUMINAMATH_CALUDE_log_approximation_l2974_297435


namespace NUMINAMATH_CALUDE_diameter_circle_equation_l2974_297450

/-- A circle passing through two points, where the line segment between the points is a diameter -/
structure DiameterCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The standard equation of a circle -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the given circle has the specified equation -/
theorem diameter_circle_equation (C : DiameterCircle) 
  (h₁ : C.A = (1, 2)) 
  (h₂ : C.B = (3, 1)) : 
  ∀ x y, circle_equation 2 (3/2) (5/4) x y :=
sorry

end NUMINAMATH_CALUDE_diameter_circle_equation_l2974_297450


namespace NUMINAMATH_CALUDE_rowing_speed_problem_l2974_297422

/-- Given a man who can row upstream at 26 kmph and downstream at 40 kmph,
    prove that his speed in still water is 33 kmph and the speed of the river current is 7 kmph. -/
theorem rowing_speed_problem (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 26)
  (h_downstream : downstream_speed = 40) :
  ∃ (still_water_speed river_current_speed : ℝ),
    still_water_speed = 33 ∧
    river_current_speed = 7 ∧
    upstream_speed = still_water_speed - river_current_speed ∧
    downstream_speed = still_water_speed + river_current_speed :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_problem_l2974_297422


namespace NUMINAMATH_CALUDE_circleplus_properties_l2974_297402

-- Define the ⊕ operation
def circleplus (x y : ℚ) : ℚ := x * y + 1

-- Theorem statement
theorem circleplus_properties :
  (circleplus 2 4 = 9) ∧
  (∀ x : ℚ, circleplus 3 (2*x - 1) = 4 → x = 1) := by
sorry

end NUMINAMATH_CALUDE_circleplus_properties_l2974_297402


namespace NUMINAMATH_CALUDE_no_extreme_points_l2974_297414

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

def f_derivative (x : ℝ) : ℝ := 3*(x - 1)^2

theorem no_extreme_points (h : ∀ x, f_derivative x ≥ 0) :
  ∀ x, ¬ (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_l2974_297414


namespace NUMINAMATH_CALUDE_rain_probability_l2974_297473

/-- Probability of rain on at least one of two days -/
theorem rain_probability (p1 p2 p2_given_r1 : ℝ) 
  (h1 : p1 = 0.3) 
  (h2 : p2 = 0.4) 
  (h3 : p2_given_r1 = 0.7) 
  (h4 : 0 ≤ p1 ∧ p1 ≤ 1)
  (h5 : 0 ≤ p2 ∧ p2 ≤ 1)
  (h6 : 0 ≤ p2_given_r1 ∧ p2_given_r1 ≤ 1) : 
  1 - ((1 - p1) * (1 - p2) + p1 * (1 - p2_given_r1)) = 0.49 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_l2974_297473


namespace NUMINAMATH_CALUDE_median_line_equation_circle_equation_l2974_297486

/-- Triangle ABC with vertices A(-3,0), B(2,0), and C(0,-4) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Define the specific triangle ABC -/
def triangleABC : Triangle :=
  { A := (-3, 0),
    B := (2, 0),
    C := (0, -4) }

/-- General form of a line equation: ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- General form of a circle equation: x^2 + y^2 + dx + ey + f = 0 -/
structure Circle :=
  (d : ℝ)
  (e : ℝ)
  (f : ℝ)

/-- Theorem: The median line of side BC in triangle ABC has equation x + 2y + 3 = 0 -/
theorem median_line_equation (t : Triangle) (l : Line) : t = triangleABC → l = { a := 1, b := 2, c := 3 } := by sorry

/-- Theorem: The circle passing through points A, B, and C has equation x^2 + y^2 + x + (5/2)y - 6 = 0 -/
theorem circle_equation (t : Triangle) (c : Circle) : t = triangleABC → c = { d := 1, e := 5/2, f := -6 } := by sorry

end NUMINAMATH_CALUDE_median_line_equation_circle_equation_l2974_297486


namespace NUMINAMATH_CALUDE_max_product_sum_200_l2974_297429

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_200_l2974_297429


namespace NUMINAMATH_CALUDE_card_selection_theorem_l2974_297416

/-- Given n ≥ 4 and 2n + 4 cards with real numbers, prove that we can select 4 cards
    such that the difference between the sums of pairs is less than 1 / (n - √(n/2)). -/
theorem card_selection_theorem (n : ℕ) (h_n : n ≥ 4) 
    (a : ℕ → ℝ) (h_a : ∀ m, m ≤ 2*n + 4 → ⌊a m⌋ = m) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    i ≤ 2*n + 4 ∧ j ≤ 2*n + 4 ∧ k ≤ 2*n + 4 ∧ l ≤ 2*n + 4 ∧
    |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) := by
  sorry

end NUMINAMATH_CALUDE_card_selection_theorem_l2974_297416


namespace NUMINAMATH_CALUDE_f_composition_equals_negative_four_thirds_l2974_297415

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x else 1/(x-1)

theorem f_composition_equals_negative_four_thirds :
  f (f 2) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_negative_four_thirds_l2974_297415


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l2974_297466

-- Define a regular polygon with n sides
def RegularPolygon (n : ℕ) : Prop :=
  n > 2  -- A polygon must have at least 3 sides

-- Define the number of diagonals in a polygon with n sides
def NumDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Theorem statement
theorem regular_polygon_diagonals_sides (n : ℕ) : 
  RegularPolygon n → (n - NumDiagonals n = 0) → n = 5 := by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_diagonals_sides_l2974_297466


namespace NUMINAMATH_CALUDE_angle_equation_l2974_297442

theorem angle_equation (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : (Real.sin α + Real.cos α) * (Real.sin β + Real.cos β) = 2) :
  (Real.sin (2 * α) + Real.cos (3 * β))^2 + (Real.sin (2 * β) + Real.cos (3 * α))^2 = 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_l2974_297442


namespace NUMINAMATH_CALUDE_students_in_class_l2974_297493

theorem students_in_class (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n + 2 = 4 * k) ∧
  (∃ l : ℕ, n + 3 = 5 * l) ∧
  (∃ m : ℕ, n + 4 = 6 * m) ↔ 
  (n = 122 ∨ n = 182) :=
by sorry

end NUMINAMATH_CALUDE_students_in_class_l2974_297493


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2974_297424

theorem min_value_reciprocal_sum (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) ≥ 3 + 2*Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (h : 0 < x ∧ x < 1) :
  1/x + 2/(1-x) = 3 + 2*Real.sqrt 2 ↔ x = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_equality_condition_l2974_297424


namespace NUMINAMATH_CALUDE_student_age_proof_l2974_297458

theorem student_age_proof (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (teacher_age : ℕ) 
  (h1 : n = 30)
  (h2 : initial_avg = 10)
  (h3 : new_avg = 11)
  (h4 : teacher_age = 41) :
  ∃ (student_age : ℕ), 
    (n : ℚ) * initial_avg - student_age + teacher_age = (n : ℚ) * new_avg ∧ 
    student_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_student_age_proof_l2974_297458


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l2974_297430

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the property of being invertible on [c, ∞)
def is_invertible_on_range (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → f x = f y → x = y

-- Theorem statement
theorem smallest_invertible_domain : 
  (∀ c < 3, ¬(is_invertible_on_range f c)) ∧ 
  (is_invertible_on_range f 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l2974_297430


namespace NUMINAMATH_CALUDE_basketball_score_proof_l2974_297452

theorem basketball_score_proof :
  ∀ (two_pointers three_pointers free_throws : ℕ),
    2 * two_pointers = 3 * three_pointers →
    free_throws = 2 * two_pointers →
    2 * two_pointers + 3 * three_pointers + free_throws = 78 →
    free_throws = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l2974_297452


namespace NUMINAMATH_CALUDE_quadratic_range_l2974_297411

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

-- Theorem statement
theorem quadratic_range :
  Set.range f = {y : ℝ | y ≥ 4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_range_l2974_297411


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l2974_297436

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 20 → speed2 = 30 → (speed1 + speed2) / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l2974_297436


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2974_297401

/-- The number of apples initially in the cafeteria. -/
def initial_apples : ℕ := sorry

/-- The number of apples handed out to students. -/
def apples_handed_out : ℕ := 19

/-- The number of pies that can be made. -/
def pies_made : ℕ := 7

/-- The number of apples required for each pie. -/
def apples_per_pie : ℕ := 8

/-- The number of apples used for making pies. -/
def apples_for_pies : ℕ := pies_made * apples_per_pie

theorem cafeteria_apples : initial_apples = apples_handed_out + apples_for_pies := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2974_297401


namespace NUMINAMATH_CALUDE_neighbor_field_size_l2974_297456

/-- Represents the yield of a cornfield over a period of time -/
structure CornYield where
  amount : ℕ
  months : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield_per_period : CornYield

def total_yield (field : Cornfield) (months : ℕ) : ℕ :=
  field.hectares * field.yield_per_period.amount * (months / field.yield_per_period.months)

def johnson_field : Cornfield :=
  { hectares := 1
  , yield_per_period := { amount := 80, months := 2 }
  }

def neighbor_field (hectares : ℕ) : Cornfield :=
  { hectares := hectares
  , yield_per_period := { amount := 160, months := 2 }
  }

theorem neighbor_field_size :
  ∃ (x : ℕ), total_yield johnson_field 6 + total_yield (neighbor_field x) 6 = 1200 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_field_size_l2974_297456


namespace NUMINAMATH_CALUDE_cubic_not_always_square_l2974_297440

theorem cubic_not_always_square (a b c : ℤ) : ∃ n : ℕ+, ¬∃ m : ℤ, (n : ℤ)^3 + a*(n : ℤ)^2 + b*(n : ℤ) + c = m^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_always_square_l2974_297440


namespace NUMINAMATH_CALUDE_group_frequency_l2974_297434

theorem group_frequency (sample_capacity : ℕ) (group_frequency_ratio : ℚ) : 
  sample_capacity = 20 →
  group_frequency_ratio = 1/4 →
  (sample_capacity : ℚ) * group_frequency_ratio = 5 := by
sorry

end NUMINAMATH_CALUDE_group_frequency_l2974_297434


namespace NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l2974_297433

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_alcohol_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel2_capacity = 5)
  (h3 : vessel2_alcohol_percentage = 40)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_alcohol_percentage = 27.5) :
  ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 25 ∧
    (vessel1_alcohol_percentage / 100) * vessel1_capacity +
    (vessel2_alcohol_percentage / 100) * vessel2_capacity =
    (final_alcohol_percentage / 100) * final_vessel_capacity :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l2974_297433


namespace NUMINAMATH_CALUDE_min_xy_value_l2974_297482

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) : 
  (x * y : ℕ) ≥ 108 := by
  sorry

end NUMINAMATH_CALUDE_min_xy_value_l2974_297482


namespace NUMINAMATH_CALUDE_continuity_at_two_delta_epsilon_relation_l2974_297480

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 25 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_delta_epsilon_relation_l2974_297480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2974_297431

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 1 = 3 ∧
  a 2 + a 5 = 36

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 6 * n - 3

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  GeneralTermFormula a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2974_297431


namespace NUMINAMATH_CALUDE_numbers_with_five_in_range_l2974_297492

def count_numbers_with_five (n : ℕ) : ℕ :=
  n - (6 * 9 * 9)

theorem numbers_with_five_in_range :
  count_numbers_with_five 700 = 214 := by
  sorry

end NUMINAMATH_CALUDE_numbers_with_five_in_range_l2974_297492


namespace NUMINAMATH_CALUDE_total_selling_price_proof_l2974_297483

def calculate_selling_price (cost : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost + cost * profit_percentage / 100

theorem total_selling_price_proof (cost_A cost_B cost_C : ℕ)
  (profit_percentage_A profit_percentage_B profit_percentage_C : ℕ)
  (h1 : cost_A = 400)
  (h2 : cost_B = 600)
  (h3 : cost_C = 800)
  (h4 : profit_percentage_A = 40)
  (h5 : profit_percentage_B = 35)
  (h6 : profit_percentage_C = 25) :
  calculate_selling_price cost_A profit_percentage_A +
  calculate_selling_price cost_B profit_percentage_B +
  calculate_selling_price cost_C profit_percentage_C = 2370 :=
by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_proof_l2974_297483


namespace NUMINAMATH_CALUDE_area_of_ADE_l2974_297408

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def Triangle.area (t : Triangle) : ℝ := sorry

def Triangle.isRightAngle (t : Triangle) (vertex : ℝ × ℝ) : Prop := sorry

def angle_bisector (A B C D : ℝ × ℝ) (E : ℝ × ℝ) : Prop := sorry

theorem area_of_ADE (A B C D E : ℝ × ℝ) : 
  let abc := Triangle.mk A B C
  let abd := Triangle.mk A B D
  (Triangle.area abc = 24) →
  (Triangle.isRightAngle abc B) →
  (Triangle.isRightAngle abd B) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 64) →
  ((B.1 - C.1)^2 + (B.2 - C.2)^2 = 36) →
  ((A.1 - D.1)^2 + (A.2 - D.2)^2 = 64) →
  (angle_bisector A C A D E) →
  Triangle.area (Triangle.mk A D E) = 20 := by sorry

end NUMINAMATH_CALUDE_area_of_ADE_l2974_297408


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2974_297491

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_diff_1 : a 2 - a 1 = 1)
  (h_diff_2 : a 5 - a 4 = 8) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2974_297491


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l2974_297490

theorem tan_product_pi_ninths : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l2974_297490


namespace NUMINAMATH_CALUDE_tracis_road_trip_l2974_297494

/-- Traci's road trip problem -/
theorem tracis_road_trip (total_distance : ℝ) (remaining_distance : ℝ) (x : ℝ) : 
  total_distance = 600 →
  remaining_distance = 300 →
  remaining_distance = total_distance - x * total_distance - (1/4) * (total_distance - x * total_distance) →
  x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tracis_road_trip_l2974_297494


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2974_297465

theorem fraction_subtraction : (3/8 : ℚ) + (5/12 : ℚ) - (1/6 : ℚ) = (5/8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2974_297465


namespace NUMINAMATH_CALUDE_abcd_sum_l2974_297496

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -2)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_l2974_297496


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2974_297460

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 = 1173 / 5000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2974_297460


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l2974_297432

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 72 → x - y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l2974_297432


namespace NUMINAMATH_CALUDE_can_form_123_l2974_297461

/-- A type representing the allowed arithmetic operations -/
inductive Operation
| Add
| Subtract
| Multiply

/-- A type representing an arithmetic expression -/
inductive Expr
| Num (n : ℕ)
| Op (op : Operation) (e1 e2 : Expr)

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℤ
| Expr.Num n => n
| Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
| Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
| Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses each of the numbers 1, 2, 3, 4, 5 exactly once -/
def usesAllNumbers : Expr → Bool := sorry

/-- The main theorem stating that 123 can be formed using the given rules -/
theorem can_form_123 : ∃ e : Expr, usesAllNumbers e ∧ eval e = 123 := by sorry

end NUMINAMATH_CALUDE_can_form_123_l2974_297461


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2974_297425

/-- Given two 2D vectors a and b, where a = (1,2) and b = (-1,x), 
    if a is parallel to b, then the magnitude of b is √5. -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-1, x]
  (∃ (k : ℝ), ∀ i, b i = k * a i) →
  Real.sqrt ((b 0)^2 + (b 1)^2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2974_297425


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l2974_297443

theorem sum_a_b_equals_negative_one (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l2974_297443


namespace NUMINAMATH_CALUDE_root_equation_value_l2974_297476

theorem root_equation_value (a b m : ℝ) : 
  a * m^2 + b * m + 5 = 0 → a * m^2 + b * m - 7 = -12 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2974_297476
