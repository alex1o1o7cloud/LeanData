import Mathlib

namespace NUMINAMATH_CALUDE_dennis_initial_money_l257_25791

def shirt_cost : ℕ := 27
def ten_dollar_bills : ℕ := 2
def loose_coins : ℕ := 3

theorem dennis_initial_money : 
  shirt_cost + ten_dollar_bills * 10 + loose_coins = 50 := by
  sorry

end NUMINAMATH_CALUDE_dennis_initial_money_l257_25791


namespace NUMINAMATH_CALUDE_similarity_coefficient_bounds_l257_25777

/-- Two triangles are similar if their corresponding sides are proportional -/
def similar_triangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

/-- The interval for the similarity coefficient -/
def similarity_coefficient_interval (k : ℝ) : Prop :=
  k > Real.sqrt 5 / 2 - 1 / 2 ∧ k < Real.sqrt 5 / 2 + 1 / 2

/-- Theorem: The similarity coefficient of two similar triangles lies within a specific interval -/
theorem similarity_coefficient_bounds (x y z p : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ p > 0) 
  (h_similar : similar_triangles x y z p) : 
  ∃ k : ℝ, similarity_coefficient_interval k ∧ x = k * y ∧ y = k * z ∧ z = k * p :=
by
  sorry

end NUMINAMATH_CALUDE_similarity_coefficient_bounds_l257_25777


namespace NUMINAMATH_CALUDE_angle_PSQ_measure_l257_25727

-- Define the points
variable (K L M N P Q S : Point) (ω : Circle)

-- Define the trapezoid
def is_trapezoid (K L M N : Point) : Prop := sorry

-- Define the circle passing through L and M
def circle_through (ω : Circle) (L M : Point) : Prop := sorry

-- Define the circle intersecting KL at P and MN at Q
def circle_intersects (ω : Circle) (K L M N P Q : Point) : Prop := sorry

-- Define the circle tangent to KN at S
def circle_tangent_at (ω : Circle) (K N S : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_PSQ_measure 
  (h_trapezoid : is_trapezoid K L M N)
  (h_circle_through : circle_through ω L M)
  (h_circle_intersects : circle_intersects ω K L M N P Q)
  (h_circle_tangent : circle_tangent_at ω K N S)
  (h_angle_LSM : angle_measure L S M = 50)
  (h_angle_equal : angle_measure K L S = angle_measure S N M) :
  angle_measure P S Q = 65 := by sorry

end NUMINAMATH_CALUDE_angle_PSQ_measure_l257_25727


namespace NUMINAMATH_CALUDE_division_remainder_proof_l257_25741

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 689 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l257_25741


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l257_25739

theorem least_positive_integer_with_given_remainders : 
  ∃ (d : ℕ), d > 0 ∧ d % 7 = 1 ∧ d % 5 = 2 ∧ d % 3 = 2 ∧ 
  ∀ (n : ℕ), n > 0 ∧ n % 7 = 1 ∧ n % 5 = 2 ∧ n % 3 = 2 → d ≤ n :=
by
  use 92
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l257_25739


namespace NUMINAMATH_CALUDE_turtle_conservation_count_l257_25743

theorem turtle_conservation_count :
  let green_turtles : ℕ := 800
  let hawksbill_turtles : ℕ := 2 * green_turtles
  let total_turtles : ℕ := green_turtles + hawksbill_turtles
  total_turtles = 2400 :=
by sorry

end NUMINAMATH_CALUDE_turtle_conservation_count_l257_25743


namespace NUMINAMATH_CALUDE_sin_45_degrees_l257_25770

/-- The sine of 45 degrees is equal to √2/2 -/
theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l257_25770


namespace NUMINAMATH_CALUDE_penalty_kicks_required_l257_25789

theorem penalty_kicks_required (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 18) (h2 : goalies = 4) : 
  (total_players - goalies) * goalies = 68 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_required_l257_25789


namespace NUMINAMATH_CALUDE_train_length_proof_l257_25735

/-- Proves that a train crossing a 550-meter platform in 51 seconds and a signal pole in 18 seconds has a length of 300 meters. -/
theorem train_length_proof (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 550)
  (h2 : platform_time = 51)
  (h3 : pole_time = 18) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l257_25735


namespace NUMINAMATH_CALUDE_boys_in_third_group_l257_25703

/-- Represents the work rate of a single person --/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers --/
structure WorkGroup where
  boys : ℕ
  girls : ℕ

/-- Calculates the total work done by a group in a given number of days --/
def totalWork (group : WorkGroup) (boyRate girlRate : WorkRate) (days : ℕ) : ℝ :=
  (group.boys : ℝ) * boyRate.rate * (days : ℝ) + (group.girls : ℝ) * girlRate.rate * (days : ℝ)

/-- The main theorem stating that the number of boys in the third group is 26 --/
theorem boys_in_third_group : 
  ∀ (x : ℕ) (boyRate girlRate : WorkRate),
  let group1 := WorkGroup.mk x 20
  let group2 := WorkGroup.mk 6 8
  let group3 := WorkGroup.mk 26 48
  totalWork group1 boyRate girlRate 4 = totalWork group2 boyRate girlRate 10 ∧
  totalWork group1 boyRate girlRate 4 = totalWork group3 boyRate girlRate 2 →
  group3.boys = 26 := by
sorry

end NUMINAMATH_CALUDE_boys_in_third_group_l257_25703


namespace NUMINAMATH_CALUDE_puzzle_spells_bach_l257_25719

/-- Represents a musical symbol --/
inductive MusicalSymbol
  | DoubleFlatSolKey
  | ATenorClef
  | CAltoClef
  | BNaturalSolKey

/-- Represents the interpretation rules --/
def interpretSymbol (s : MusicalSymbol) : Char :=
  match s with
  | MusicalSymbol.DoubleFlatSolKey => 'B'
  | MusicalSymbol.ATenorClef => 'A'
  | MusicalSymbol.CAltoClef => 'C'
  | MusicalSymbol.BNaturalSolKey => 'H'

/-- The sequence of symbols in the puzzle --/
def puzzleSequence : List MusicalSymbol := [
  MusicalSymbol.DoubleFlatSolKey,
  MusicalSymbol.ATenorClef,
  MusicalSymbol.CAltoClef,
  MusicalSymbol.BNaturalSolKey
]

/-- The theorem stating that the puzzle sequence spells "BACH" --/
theorem puzzle_spells_bach :
  puzzleSequence.map interpretSymbol = ['B', 'A', 'C', 'H'] := by
  sorry


end NUMINAMATH_CALUDE_puzzle_spells_bach_l257_25719


namespace NUMINAMATH_CALUDE_product_equality_l257_25744

theorem product_equality (a b c : ℝ) (h : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) :
  6 * 15 * 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l257_25744


namespace NUMINAMATH_CALUDE_square_of_integer_ending_in_five_l257_25713

theorem square_of_integer_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_ending_in_five_l257_25713


namespace NUMINAMATH_CALUDE_valid_tiling_exists_l257_25732

/-- Represents a point on the infinite 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a domino piece on the grid -/
inductive Domino
  | Horizontal (topLeft : GridPoint)
  | Vertical (topLeft : GridPoint)

/-- Represents a tiling of the infinite grid with dominos -/
def Tiling := GridPoint → Domino

/-- Checks if a given point is covered by a domino in the tiling -/
def isCovered (t : Tiling) (p : GridPoint) : Prop := 
  ∃ d : Domino, d ∈ Set.range t ∧ 
    match d with
    | Domino.Horizontal tl => p.x = tl.x ∧ (p.y = tl.y ∨ p.y = tl.y + 1)
    | Domino.Vertical tl => p.y = tl.y ∧ (p.x = tl.x ∨ p.x = tl.x + 1)

/-- Checks if a horizontal line intersects a finite number of dominos -/
def finiteHorizontalIntersections (t : Tiling) : Prop :=
  ∀ y : ℤ, ∃ n : ℕ, ∀ x : ℤ, x > n → 
    (t ⟨x, y⟩ = t ⟨x - 1, y⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- Checks if a vertical line intersects a finite number of dominos -/
def finiteVerticalIntersections (t : Tiling) : Prop :=
  ∀ x : ℤ, ∃ n : ℕ, ∀ y : ℤ, y > n → 
    (t ⟨x, y⟩ = t ⟨x, y - 1⟩ ∨ t ⟨x, y⟩ = t ⟨x - 1, y - 1⟩)

/-- The main theorem stating that a valid tiling with the required properties exists -/
theorem valid_tiling_exists : 
  ∃ t : Tiling, 
    (∀ p : GridPoint, isCovered t p) ∧ 
    finiteHorizontalIntersections t ∧
    finiteVerticalIntersections t := by
  sorry

end NUMINAMATH_CALUDE_valid_tiling_exists_l257_25732


namespace NUMINAMATH_CALUDE_kristin_income_l257_25792

/-- Represents the tax structure and Kristin's income --/
structure TaxSystem where
  p : ℝ  -- base tax rate in decimal form
  income : ℝ  -- Kristin's annual income

/-- Calculates the total tax paid based on the given tax structure --/
def totalTax (ts : TaxSystem) : ℝ :=
  ts.p * 28000 + (ts.p + 0.02) * (ts.income - 28000)

/-- Theorem stating that Kristin's income is $32000 given the tax conditions --/
theorem kristin_income (ts : TaxSystem) :
  (totalTax ts = (ts.p + 0.0025) * ts.income) → ts.income = 32000 := by
  sorry


end NUMINAMATH_CALUDE_kristin_income_l257_25792


namespace NUMINAMATH_CALUDE_swap_digits_theorem_l257_25754

/-- Represents a two-digit number with digits a and b -/
structure TwoDigitNumber where
  a : ℕ
  b : ℕ
  a_less_than_ten : a < 10
  b_less_than_ten : b < 10

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ := 10 * n.a + n.b

/-- The value of a two-digit number with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : ℕ := 10 * n.b + n.a

/-- Theorem stating that swapping digits in a two-digit number results in 10b + a -/
theorem swap_digits_theorem (n : TwoDigitNumber) : 
  n.swapped_value = 10 * n.b + n.a := by sorry

end NUMINAMATH_CALUDE_swap_digits_theorem_l257_25754


namespace NUMINAMATH_CALUDE_vehicle_tire_usage_l257_25793

/-- Calculates the miles each tire is used given the total miles traveled, 
    number of tires, and number of tires used at a time. -/
def milesPerTire (totalMiles : ℕ) (numTires : ℕ) (tiresUsed : ℕ) : ℚ :=
  (totalMiles * tiresUsed : ℚ) / numTires

/-- Proves that given the conditions of the problem, each tire is used for 32,000 miles -/
theorem vehicle_tire_usage :
  let totalMiles : ℕ := 48000
  let numTires : ℕ := 6
  let tiresUsed : ℕ := 4
  milesPerTire totalMiles numTires tiresUsed = 32000 := by
  sorry

#eval milesPerTire 48000 6 4

end NUMINAMATH_CALUDE_vehicle_tire_usage_l257_25793


namespace NUMINAMATH_CALUDE_alphabet_value_problem_l257_25740

theorem alphabet_value_problem (T M A H E : ℤ) : 
  T = 15 →
  M + A + T + H = 47 →
  T + E + A + M = 58 →
  M + E + E + T = 45 →
  M = 8 := by
sorry

end NUMINAMATH_CALUDE_alphabet_value_problem_l257_25740


namespace NUMINAMATH_CALUDE_kendy_transfer_proof_l257_25709

-- Define the initial balance
def initial_balance : ℚ := 190

-- Define the remaining balance
def remaining_balance : ℚ := 100

-- Define the amount transferred to mom
def amount_to_mom : ℚ := 60

-- Define the amount transferred to sister
def amount_to_sister : ℚ := amount_to_mom / 2

-- Theorem statement
theorem kendy_transfer_proof :
  initial_balance - (amount_to_mom + amount_to_sister) = remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_kendy_transfer_proof_l257_25709


namespace NUMINAMATH_CALUDE_solve_selinas_shirt_sales_l257_25783

/-- Represents the problem of determining how many shirts Selina sold. -/
def SelinasShirtSales : Prop :=
  let pants_price : ℕ := 5
  let shorts_price : ℕ := 3
  let shirt_price : ℕ := 4
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_bought : ℕ := 2
  let shirt_buy_price : ℕ := 10
  let remaining_money : ℕ := 30
  ∃ (shirts_sold : ℕ),
    shirts_sold * shirt_price + 
    pants_sold * pants_price + 
    shorts_sold * shorts_price = 
    remaining_money + shirts_bought * shirt_buy_price ∧
    shirts_sold = 5

theorem solve_selinas_shirt_sales : SelinasShirtSales := by
  sorry

#check solve_selinas_shirt_sales

end NUMINAMATH_CALUDE_solve_selinas_shirt_sales_l257_25783


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l257_25737

theorem simplify_sqrt_expression :
  Real.sqrt (75 - 30 * Real.sqrt 5) = 5 - 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l257_25737


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_prove_dormitory_to_city_distance_l257_25764

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun D : ℝ =>
    (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40

-- The proof is omitted
theorem prove_dormitory_to_city_distance :
  ∃ D : ℝ, dormitory_to_city_distance D :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_prove_dormitory_to_city_distance_l257_25764


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l257_25768

theorem interest_equality_implies_second_sum (total : ℚ) 
  (h1 : total = 2665) 
  (h2 : ∃ x : ℚ, x * (3/100) * 8 = (total - x) * (5/100) * 3) : 
  ∃ second : ℚ, second = total - 2460 :=
sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l257_25768


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l257_25733

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : 
  r = (1 : ℝ) / 4 →
  S = 16 →
  S = a / (1 - r) →
  a * r = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l257_25733


namespace NUMINAMATH_CALUDE_foot_of_perpendicular_to_yOz_plane_l257_25772

/-- The foot of a perpendicular from a point to a plane -/
def foot_of_perpendicular (P : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  sorry

/-- The yOz plane in ℝ³ -/
def yOz_plane : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 = 0}

theorem foot_of_perpendicular_to_yOz_plane :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let Q := foot_of_perpendicular P yOz_plane
  Q = (0, Real.sqrt 2, Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_foot_of_perpendicular_to_yOz_plane_l257_25772


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l257_25702

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l257_25702


namespace NUMINAMATH_CALUDE_money_left_over_l257_25784

theorem money_left_over (hourly_rate : ℕ) (hours_worked : ℕ) (game_cost : ℕ) (candy_cost : ℕ) : 
  hourly_rate = 8 → 
  hours_worked = 9 → 
  game_cost = 60 → 
  candy_cost = 5 → 
  hourly_rate * hours_worked - (game_cost + candy_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l257_25784


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l257_25706

def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence
  (a : ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d 3 = 23)
  (h2 : arithmetic_sequence a d 7 = 35) :
  arithmetic_sequence a d 10 = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l257_25706


namespace NUMINAMATH_CALUDE_smallest_disjoint_r_l257_25728

def A : Set ℤ := {n | ∃ k : ℕ, (n = 3 + 10 * k) ∨ (n = 6 + 26 * k) ∨ (n = 5 + 29 * k)}

def is_disjoint (r b : ℤ) : Prop :=
  ∀ k l : ℕ, (b + r * k) ∉ A

theorem smallest_disjoint_r : 
  (∃ b : ℤ, is_disjoint 290 b) ∧ 
  (∀ r : ℕ, r < 290 → ¬∃ b : ℤ, is_disjoint r b) :=
sorry

end NUMINAMATH_CALUDE_smallest_disjoint_r_l257_25728


namespace NUMINAMATH_CALUDE_largest_prime_factor_l257_25718

theorem largest_prime_factor : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l257_25718


namespace NUMINAMATH_CALUDE_log_2_base_10_l257_25765

theorem log_2_base_10 (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) (h3 : 2^9 = 512) (h4 : 2^12 = 4096) :
  Real.log 2 / Real.log 10 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_l257_25765


namespace NUMINAMATH_CALUDE_smallest_integers_satisfying_equation_l257_25775

theorem smallest_integers_satisfying_equation :
  ∃ (a b : ℕ+),
    (7 * a^3 = 11 * b^5) ∧
    (∀ (a' b' : ℕ+), 7 * a'^3 = 11 * b'^5 → a ≤ a' ∧ b ≤ b') ∧
    a = 41503 ∧
    b = 539 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integers_satisfying_equation_l257_25775


namespace NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l257_25714

theorem coefficient_m5n5_in_expansion : (Nat.choose 10 5) = 252 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n5_in_expansion_l257_25714


namespace NUMINAMATH_CALUDE_four_digit_number_not_divisible_by_11_l257_25704

def is_not_divisible_by_11 (n : ℕ) : Prop := ¬(n % 11 = 0)

theorem four_digit_number_not_divisible_by_11 :
  ∀ B : ℕ, B < 10 →
  (∃ A : ℕ, A < 10 ∧ 
    (∀ B : ℕ, B < 10 → is_not_divisible_by_11 (9000 + 100 * A + 10 * B))) ↔ 
  (∃ A : ℕ, A = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_not_divisible_by_11_l257_25704


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l257_25736

theorem point_in_first_quadrant (m : ℝ) : 
  let P : ℝ × ℝ := (3, m^2 + 1)
  P.1 > 0 ∧ P.2 > 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l257_25736


namespace NUMINAMATH_CALUDE_solution_is_negative_two_l257_25779

-- Define the equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 2 ∧ (4 / (x - 2) = 2 / x)

-- Theorem statement
theorem solution_is_negative_two :
  ∃ (x : ℝ), fractional_equation x ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_solution_is_negative_two_l257_25779


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l257_25749

def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x + 1

theorem max_value_of_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 10 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l257_25749


namespace NUMINAMATH_CALUDE_stratified_sample_size_l257_25717

/-- Represents the ratio of students in three schools -/
structure SchoolRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the number of students sampled from the smallest school -/
def totalSampleSize (ratio : SchoolRatio) (smallestSchoolSample : ℕ) : ℕ :=
  smallestSchoolSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For schools with ratio 2:3:5, if 10 students are sampled from the smallest school, the total sample is 50 -/
theorem stratified_sample_size (ratio : SchoolRatio) (h1 : ratio.a = 2) (h2 : ratio.b = 3) (h3 : ratio.c = 5) :
  totalSampleSize ratio 10 = 50 := by
  sorry

#eval totalSampleSize ⟨2, 3, 5⟩ 10

end NUMINAMATH_CALUDE_stratified_sample_size_l257_25717


namespace NUMINAMATH_CALUDE_billy_video_count_l257_25787

/-- The number of videos suggested in each round -/
def suggestions_per_round : ℕ := 15

/-- The number of rounds Billy goes through without liking any videos -/
def unsuccessful_rounds : ℕ := 5

/-- The position of the video Billy watches in the final round -/
def final_video_position : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos_watched : ℕ := suggestions_per_round * unsuccessful_rounds + 1

theorem billy_video_count :
  total_videos_watched = 76 :=
sorry

end NUMINAMATH_CALUDE_billy_video_count_l257_25787


namespace NUMINAMATH_CALUDE_system_solution_l257_25759

theorem system_solution : ∃! (x y : ℚ), 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -241/29 ∧ y = -32/29 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l257_25759


namespace NUMINAMATH_CALUDE_remaining_money_after_tickets_l257_25766

/-- Calculates the remaining money after buying tickets -/
def remaining_money (olivia_money : ℕ) (nigel_money : ℕ) (num_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  olivia_money + nigel_money - num_tickets * ticket_price

/-- Proves that Olivia and Nigel have $83 left after buying tickets -/
theorem remaining_money_after_tickets : remaining_money 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_tickets_l257_25766


namespace NUMINAMATH_CALUDE_map_scale_theorem_l257_25734

/-- Represents the scale of a map in inches per foot -/
def scale : ℚ := 2 / 500

/-- Represents the length of a line segment on the map in inches -/
def map_length : ℚ := 29 / 4

/-- Calculates the actual length in feet represented by a length on the map -/
def actual_length (map_len : ℚ) : ℚ := map_len / scale

theorem map_scale_theorem :
  actual_length map_length = 1812.5 := by sorry

end NUMINAMATH_CALUDE_map_scale_theorem_l257_25734


namespace NUMINAMATH_CALUDE_johnson_yield_l257_25722

/-- Represents the yield of corn per hectare every two months -/
structure CornYield where
  amount : ℕ

/-- Represents a cornfield -/
structure Cornfield where
  hectares : ℕ
  yield : CornYield

def total_yield (field : Cornfield) (periods : ℕ) : ℕ :=
  field.hectares * field.yield.amount * periods

theorem johnson_yield (johnson : Cornfield) (neighbor : Cornfield) 
    (h1 : johnson.hectares = 1)
    (h2 : neighbor.hectares = 2)
    (h3 : neighbor.yield.amount = 2 * johnson.yield.amount)
    (h4 : total_yield johnson 3 + total_yield neighbor 3 = 1200) :
  johnson.yield.amount = 80 := by
  sorry

#check johnson_yield

end NUMINAMATH_CALUDE_johnson_yield_l257_25722


namespace NUMINAMATH_CALUDE_minimal_fence_posts_l257_25757

/-- Calculates the number of fence posts required for a rectangular park --/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := width / post_spacing
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the minimal number of fence posts required for the given park --/
theorem minimal_fence_posts :
  fence_posts 90 45 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_minimal_fence_posts_l257_25757


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_ten_l257_25723

theorem units_digit_of_seven_to_ten (n : ℕ) : 7^10 ≡ 9 [MOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_ten_l257_25723


namespace NUMINAMATH_CALUDE_function_value_at_three_l257_25773

/-- Given a function f : ℝ → ℝ satisfying f(x) + 2f(1 - x) = 3x^2 for all real x,
    prove that f(3) = -1 -/
theorem function_value_at_three (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2) : 
    f 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l257_25773


namespace NUMINAMATH_CALUDE_proposition_truth_values_l257_25796

-- Define proposition p
def p : Prop := ∀ a : ℝ, (∀ x : ℝ, (x^2 + |x - a| = (-x)^2 + |(-x) - a|)) → a = 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Theorem stating the truth values of the propositions
theorem proposition_truth_values : 
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬((¬p) ∧ q) ∧ 
  ((¬p) ∨ (¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l257_25796


namespace NUMINAMATH_CALUDE_hannahs_purchase_cost_l257_25788

/-- The total cost of purchasing sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that the total cost of 3 sweatshirts at $15 each and 2 T-shirts at $10 each is $65 -/
theorem hannahs_purchase_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_purchase_cost_l257_25788


namespace NUMINAMATH_CALUDE_max_m_value_min_quadratic_sum_l257_25710

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_quadratic_sum {a b c : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  2*a^2 + 3*b^2 + 4*c^2 ≥ 12/13 ∧ 
  (2*a^2 + 3*b^2 + 4*c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_quadratic_sum_l257_25710


namespace NUMINAMATH_CALUDE_race_finish_order_l257_25755

theorem race_finish_order (total_students : ℕ) (before_yoongi : ℕ) (after_yoongi : ℕ) : 
  total_students = 20 → before_yoongi = 11 → after_yoongi = total_students - (before_yoongi + 1) → after_yoongi = 8 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_order_l257_25755


namespace NUMINAMATH_CALUDE_train_speed_calculation_l257_25771

/-- Represents the speed of a train in various conditions -/
structure TrainSpeed where
  /-- Speed of the train including stoppages (in kmph) -/
  average_speed : ℝ
  /-- Time the train stops per hour (in minutes) -/
  stop_time : ℝ
  /-- Speed of the train when not stopping (in kmph) -/
  actual_speed : ℝ

/-- Theorem stating the relationship between average speed, stop time, and actual speed -/
theorem train_speed_calculation (t : TrainSpeed) (h1 : t.average_speed = 36) 
    (h2 : t.stop_time = 24) : t.actual_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l257_25771


namespace NUMINAMATH_CALUDE_equation_solutions_l257_25782

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x - 8) + 1 / (x^2 + 2*x - 8) + 1 / (x^2 - 13*x - 8) = 0)} = 
  {8, 1, -1, -8} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l257_25782


namespace NUMINAMATH_CALUDE_min_chips_for_adjacency_l257_25746

/-- Represents a color of a chip -/
def Color : Type := Fin 6

/-- Represents a row of chips -/
def ChipRow := List Color

/-- Checks if two colors are adjacent in a row -/
def areAdjacent (c1 c2 : Color) (row : ChipRow) : Prop :=
  ∃ i, (row.get? i = some c1 ∧ row.get? (i+1) = some c2) ∨
       (row.get? i = some c2 ∧ row.get? (i+1) = some c1)

/-- Checks if all pairs of colors are adjacent in a row -/
def allPairsAdjacent (row : ChipRow) : Prop :=
  ∀ c1 c2 : Color, c1 ≠ c2 → areAdjacent c1 c2 row

/-- The main theorem stating the minimum number of chips required -/
theorem min_chips_for_adjacency :
  ∃ (row : ChipRow), allPairsAdjacent row ∧ row.length = 18 ∧
  (∀ (row' : ChipRow), allPairsAdjacent row' → row'.length ≥ 18) :=
sorry

end NUMINAMATH_CALUDE_min_chips_for_adjacency_l257_25746


namespace NUMINAMATH_CALUDE_grandson_age_l257_25762

/-- Given the ages of three family members satisfying certain conditions,
    prove that the youngest member (grandson) is 20 years old. -/
theorem grandson_age (grandson_age son_age markus_age : ℕ) : 
  son_age = 2 * grandson_age →
  markus_age = 2 * son_age →
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_grandson_age_l257_25762


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l257_25797

def standard_deck_size : ℕ := 52
def heart_or_king_count : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - heart_or_king_count / standard_deck_size) ^ 2
  p = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l257_25797


namespace NUMINAMATH_CALUDE_magnitude_of_b_l257_25731

/-- Given two vectors a and b in ℝ², prove that the magnitude of b is 2√2 -/
theorem magnitude_of_b (a b : ℝ × ℝ) : 
  let angle := 3 * π / 4
  a = (-3, 4) →
  a.fst * b.fst + a.snd * b.snd = -10 →
  Real.sqrt (b.fst ^ 2 + b.snd ^ 2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_b_l257_25731


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_square_l257_25721

theorem sum_and_reciprocal_square (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + m + 1/m = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_square_l257_25721


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l257_25748

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 3

theorem monotonic_quadratic (m : ℝ) :
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → (f m x₁ < f m x₂ ∨ f m x₁ > f m x₂)) →
  m ≤ 1 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l257_25748


namespace NUMINAMATH_CALUDE_median_angle_relation_l257_25701

/-- Represents a triangle with sides a, b, c, angle γ opposite to side c, and median sc to side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  γ : ℝ
  sc : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_γ : 0 < γ
  pos_sc : 0 < sc
  tri_ineq : a + b > c ∧ b + c > a ∧ c + a > b

theorem median_angle_relation (t : Triangle) :
  (t.γ < 90 ↔ t.sc > t.c / 2) ∧
  (t.γ = 90 ↔ t.sc = t.c / 2) ∧
  (t.γ > 90 ↔ t.sc < t.c / 2) :=
sorry

end NUMINAMATH_CALUDE_median_angle_relation_l257_25701


namespace NUMINAMATH_CALUDE_manager_selection_problem_l257_25799

theorem manager_selection_problem (n m k : ℕ) (h1 : n = 7) (h2 : m = 4) (h3 : k = 2) :
  (Nat.choose n m) - (Nat.choose (n - k) (m - k)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_manager_selection_problem_l257_25799


namespace NUMINAMATH_CALUDE_deposit_calculation_l257_25785

theorem deposit_calculation (total_cost : ℝ) (deposit : ℝ) : 
  deposit = 0.1 * total_cost ∧ 
  total_cost - deposit = 1080 → 
  deposit = 120 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l257_25785


namespace NUMINAMATH_CALUDE_min_value_of_expression_l257_25756

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 3/2) :
  ∃ (m : ℝ), m = 16/9 ∧ ∀ x, 1 < x ∧ x < 3/2 → (1/(3-2*x) + 2/(x-1)) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l257_25756


namespace NUMINAMATH_CALUDE_savings_calculation_l257_25753

theorem savings_calculation (savings : ℚ) : 
  (1 / 2 : ℚ) * savings = 300 → savings = 600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l257_25753


namespace NUMINAMATH_CALUDE_max_cables_used_l257_25708

/-- Represents the number of brand A computers -/
def brand_A_count : Nat := 25

/-- Represents the number of brand B computers -/
def brand_B_count : Nat := 15

/-- Represents the total number of employees -/
def total_employees : Nat := brand_A_count + brand_B_count

/-- Theorem stating the maximum number of cables that can be used -/
theorem max_cables_used : 
  ∀ (cables : Nat), 
    (cables ≤ brand_A_count * brand_B_count) → 
    (∀ (b : Nat), b < brand_B_count → ∃ (a : Nat), a < brand_A_count ∧ True) → 
    cables ≤ 375 :=
by sorry

end NUMINAMATH_CALUDE_max_cables_used_l257_25708


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l257_25700

theorem min_value_trig_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 20)^2 ≥ 236 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l257_25700


namespace NUMINAMATH_CALUDE_total_seeds_l257_25790

/-- The number of seeds Emily planted in the big garden. -/
def big_garden_seeds : ℕ := 36

/-- The number of small gardens Emily had. -/
def small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden. -/
def seeds_per_small_garden : ℕ := 2

/-- Theorem stating the total number of seeds Emily started with. -/
theorem total_seeds : 
  big_garden_seeds + small_gardens * seeds_per_small_garden = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_l257_25790


namespace NUMINAMATH_CALUDE_alexis_dresses_l257_25776

theorem alexis_dresses (isabella_total : ℕ) (alexis_pants : ℕ) 
  (h1 : isabella_total = 13)
  (h2 : alexis_pants = 21) : 
  3 * isabella_total - alexis_pants = 18 := by
  sorry

end NUMINAMATH_CALUDE_alexis_dresses_l257_25776


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l257_25726

/-- An ellipse intersecting a line with a specific midpoint property -/
structure EllipseLineIntersection where
  m : ℝ
  n : ℝ
  -- Ellipse equation: mx^2 + ny^2 = 1
  -- Line equation: x + y - 1 = 0
  -- Intersection points exist (implicit)
  -- Line through midpoint and origin has slope √2/2 (implicit)

/-- The ratio m/n equals √2/2 for the given ellipse-line intersection -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l257_25726


namespace NUMINAMATH_CALUDE_increase_both_averages_l257_25798

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ x ∈ group1,
    average (group1.filter (· ≠ x)) > average group1 ∧
    average (x :: group2) > average group2 :=
by sorry

end NUMINAMATH_CALUDE_increase_both_averages_l257_25798


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_17_gt_200_l257_25730

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_17_gt_200 :
  ∃ p : ℕ,
    is_prime p ∧
    digit_sum p = 17 ∧
    p > 200 ∧
    (∀ q : ℕ, is_prime q → digit_sum q = 17 → q > 200 → p ≤ q) ∧
    p = 197 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_17_gt_200_l257_25730


namespace NUMINAMATH_CALUDE_equal_bills_at_20_minutes_l257_25780

/-- Represents a telephone company with a base rate and per-minute charge. -/
structure TelephoneCompany where
  base_rate : ℝ
  per_minute_charge : ℝ

/-- Calculates the total cost for a given number of minutes. -/
def total_cost (company : TelephoneCompany) (minutes : ℝ) : ℝ :=
  company.base_rate + company.per_minute_charge * minutes

/-- The three telephone companies with their respective rates. -/
def united_telephone : TelephoneCompany := ⟨11, 0.25⟩
def atlantic_call : TelephoneCompany := ⟨12, 0.20⟩
def global_connect : TelephoneCompany := ⟨13, 0.15⟩

theorem equal_bills_at_20_minutes :
  ∃ (m : ℝ),
    m = 20 ∧
    total_cost united_telephone m = total_cost atlantic_call m ∧
    total_cost atlantic_call m = total_cost global_connect m :=
  sorry

end NUMINAMATH_CALUDE_equal_bills_at_20_minutes_l257_25780


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_half_l257_25769

/-- Given an ellipse and a hyperbola with shared foci, prove the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity_half 
  (a b m n c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hab : a > b)
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hyperbola_eq : ∀ (x y : ℝ), x^2 / m^2 - y^2 / n^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (foci : c > 0 ∧ {(-c, 0), (c, 0)} ⊆ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} ∩ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (geom_mean : c^2 = a * m)
  (arith_mean : n^2 = m^2 + c^2 / 2) :
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_half_l257_25769


namespace NUMINAMATH_CALUDE_integer_pair_problem_l257_25767

theorem integer_pair_problem (a b q r : ℕ) (h1 : a > b) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + 2*r = 2020) :
  ((a = 53 ∧ b = 29) ∨ (a = 53 ∧ b = 15)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_problem_l257_25767


namespace NUMINAMATH_CALUDE_min_value_expression_l257_25751

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^3 + b^3 + 1/a^3 + b/a ≥ 53/27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l257_25751


namespace NUMINAMATH_CALUDE_fraction_problem_l257_25712

theorem fraction_problem (f : ℚ) : f = 1/3 → 0.75 * 264 = f * 264 + 110 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l257_25712


namespace NUMINAMATH_CALUDE_square_perimeter_with_area_9_l257_25725

theorem square_perimeter_with_area_9 (s : ℝ) (h1 : s^2 = 9) (h2 : ∃ k : ℕ, 4 * s = 4 * k) : 4 * s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_with_area_9_l257_25725


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l257_25724

theorem quadratic_equation_solution (h : (63 * (5/7)^2 + 36) = (100 * (5/7) - 9)) :
  (63 * 1^2 + 36) = (100 * 1 - 9) ∧ 
  ∀ x : ℚ, x ≠ 5/7 → x ≠ 1 → (63 * x^2 + 36) ≠ (100 * x - 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l257_25724


namespace NUMINAMATH_CALUDE_prime_power_plus_144_square_l257_25742

theorem prime_power_plus_144_square (p n m : ℕ) : 
  p.Prime → p > 0 → n > 0 → m > 0 → p^n + 144 = m^2 → 
  (p = 5 ∧ n = 2 ∧ m = 13) ∨ (p = 2 ∧ n = 8 ∧ m = 20) ∨ (p = 3 ∧ n = 4 ∧ m = 15) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_plus_144_square_l257_25742


namespace NUMINAMATH_CALUDE_max_value_at_two_l257_25715

/-- The function f(x) = x(x-m)² -/
def f (m : ℝ) (x : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) -/
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*m*x + m^2

theorem max_value_at_two (m : ℝ) :
  (∀ x : ℝ, f m x ≤ f m 2) →
  (m = 6 ∧
   ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                             f m x₁ = a ∧ f m x₂ = a ∧ f m x₃ = a) ↔
             (0 < a ∧ a < 32)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_two_l257_25715


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l257_25794

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

#eval count_disjoint_subsets S % 1000

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l257_25794


namespace NUMINAMATH_CALUDE_vicente_meat_purchase_l257_25786

theorem vicente_meat_purchase
  (rice_kg : ℕ)
  (rice_price : ℚ)
  (meat_price : ℚ)
  (total_spent : ℚ)
  (h1 : rice_kg = 5)
  (h2 : rice_price = 2)
  (h3 : meat_price = 5)
  (h4 : total_spent = 25)
  : (total_spent - rice_kg * rice_price) / meat_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_vicente_meat_purchase_l257_25786


namespace NUMINAMATH_CALUDE_parking_garage_floors_l257_25720

/-- Represents a parking garage with the given properties -/
structure ParkingGarage where
  floors : ℕ
  drive_time : ℕ
  id_check_time : ℕ
  total_time : ℕ

/-- Calculates the number of ID checks required -/
def id_checks (g : ParkingGarage) : ℕ := (g.floors - 1) / 3

/-- Calculates the total time to traverse the garage -/
def calculate_total_time (g : ParkingGarage) : ℕ :=
  g.drive_time * (g.floors - 1) + g.id_check_time * id_checks g

/-- Theorem stating that a parking garage with the given properties has 13 floors -/
theorem parking_garage_floors :
  ∃ (g : ParkingGarage), 
    g.drive_time = 80 ∧ 
    g.id_check_time = 120 ∧ 
    g.total_time = 1440 ∧ 
    calculate_total_time g = g.total_time ∧ 
    g.floors = 13 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_floors_l257_25720


namespace NUMINAMATH_CALUDE_sqrt_twelve_simplification_l257_25745

theorem sqrt_twelve_simplification : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_simplification_l257_25745


namespace NUMINAMATH_CALUDE_similar_triangle_area_reduction_l257_25707

/-- Given a right-angled triangle with area A and hypotenuse H, if a smaller similar triangle
    is formed by cutting parallel to the hypotenuse such that the new hypotenuse H' = 0.65H,
    then the area A' of the smaller triangle is equal to A * (0.65)^2. -/
theorem similar_triangle_area_reduction (A H H' A' : ℝ) 
    (h1 : A > 0) 
    (h2 : H > 0) 
    (h3 : H' = 0.65 * H) 
    (h4 : A' / A = (H' / H)^2) : 
  A' = A * (0.65)^2 := by
  sorry

#check similar_triangle_area_reduction

end NUMINAMATH_CALUDE_similar_triangle_area_reduction_l257_25707


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l257_25774

/-- Given three points A, B, C on a line, with AB = 3 km and BC = 4 km,
    a cyclist starting from A towards C, and a pedestrian starting from B towards A,
    prove that they meet at a point 2.1 km from A if they arrive at their
    destinations simultaneously. -/
theorem cyclist_pedestrian_meeting_point
  (A B C : ℝ) -- Points represented as real numbers
  (h_order : A < B ∧ B < C) -- Points are in order
  (h_AB : B - A = 3) -- Distance AB is 3 km
  (h_BC : C - B = 4) -- Distance BC is 4 km
  (cyclist_speed pedestrian_speed : ℝ) -- Speeds of cyclist and pedestrian
  (h_speeds_positive : cyclist_speed > 0 ∧ pedestrian_speed > 0) -- Speeds are positive
  (h_simultaneous_arrival : (C - A) / cyclist_speed = (B - A) / pedestrian_speed) -- Simultaneous arrival
  : ∃ (D : ℝ), D - A = 21/10 ∧ A < D ∧ D < B :=
sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l257_25774


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l257_25729

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l257_25729


namespace NUMINAMATH_CALUDE_garden_bed_area_l257_25747

/-- Represents the dimensions of a rectangular garden bed -/
structure GardenBed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a garden bed -/
def area (bed : GardenBed) : ℝ := bed.length * bed.width

/-- Theorem: Given the conditions, prove that the area of each unknown garden bed is 9 sq ft -/
theorem garden_bed_area 
  (known_bed : GardenBed)
  (unknown_bed : GardenBed)
  (h1 : known_bed.length = 4)
  (h2 : known_bed.width = 3)
  (h3 : area known_bed + area known_bed + area unknown_bed + area unknown_bed = 42) :
  area unknown_bed = 9 := by
  sorry

end NUMINAMATH_CALUDE_garden_bed_area_l257_25747


namespace NUMINAMATH_CALUDE_tan_intersection_distance_l257_25716

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem tan_intersection_distance (a : ℝ) :
  ∃ (d : ℝ), d > 0 ∧
  ∀ (x : ℝ), tan x = a → tan (x + d) = a ∧
  ∀ (y : ℝ), 0 < y ∧ y < d → tan (x + y) ≠ a :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_tan_intersection_distance_l257_25716


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l257_25711

/-- A parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an intersection point between a parabola and a circle -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is a tangent point between a parabola and a circle -/
def is_tangent_point (p : Parabola) (c : Circle) (point : IntersectionPoint) : Prop :=
  sorry

/-- Theorem stating that if a circle and a parabola intersect at exactly two points,
    and one is a tangent point, then the other must also be a tangent point -/
theorem parabola_circle_tangency
  (p : Parabola) (c : Circle) 
  (i1 i2 : IntersectionPoint) 
  (h_distinct : i1 ≠ i2)
  (h_only_two : ∀ i : IntersectionPoint, i = i1 ∨ i = i2)
  (h_tangent : is_tangent_point p c i1) :
  is_tangent_point p c i2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l257_25711


namespace NUMINAMATH_CALUDE_recipe_total_ingredients_l257_25795

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar -/
def totalIngredients (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and sugar amount, the total ingredients is 28 cups -/
theorem recipe_total_ingredients :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  totalIngredients ratio 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_ingredients_l257_25795


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l257_25763

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+8) = (8 : ℝ)^(3*x+7) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l257_25763


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_divisors_greater_than_sqrt_l257_25752

theorem arithmetic_mean_of_divisors_greater_than_sqrt (n : ℕ) (hn : n > 1) :
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  (divisors.sum / divisors.length : ℝ) > Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_divisors_greater_than_sqrt_l257_25752


namespace NUMINAMATH_CALUDE_revenue_growth_exists_l257_25761

/-- Represents the revenue growth rate in a supermarket over three months -/
def revenue_growth_equation (x : ℝ) : Prop :=
  let january_revenue : ℝ := 90
  let total_revenue : ℝ := 144
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = total_revenue

/-- Theorem stating that the revenue growth equation holds for some growth rate x -/
theorem revenue_growth_exists : ∃ x : ℝ, revenue_growth_equation x := by
  sorry

end NUMINAMATH_CALUDE_revenue_growth_exists_l257_25761


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l257_25760

/-- A rectangular parallelepiped with length and width twice the height and sum of edge lengths 100 cm has surface area 400 cm² -/
theorem rectangular_parallelepiped_surface_area 
  (h : ℝ) 
  (sum_edges : 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100) : 
  2 * (2 * h) * (2 * h) + 2 * (2 * h) * h + 2 * (2 * h) * h = 400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l257_25760


namespace NUMINAMATH_CALUDE_barber_loss_l257_25781

/-- Represents the monetary transactions in the barbershop scenario -/
structure BarbershopScenario where
  haircut_price : ℕ
  counterfeit_bill : ℕ
  change_given : ℕ
  replacement_bill : ℕ

/-- Calculates the total loss for the barber in the given scenario -/
def calculate_loss (scenario : BarbershopScenario) : ℕ :=
  scenario.haircut_price + scenario.change_given + scenario.replacement_bill - scenario.counterfeit_bill

/-- Theorem stating that the barber's loss in the given scenario is $25 -/
theorem barber_loss (scenario : BarbershopScenario) 
  (h1 : scenario.haircut_price = 15)
  (h2 : scenario.counterfeit_bill = 20)
  (h3 : scenario.change_given = 5)
  (h4 : scenario.replacement_bill = 20) :
  calculate_loss scenario = 25 := by
  sorry

end NUMINAMATH_CALUDE_barber_loss_l257_25781


namespace NUMINAMATH_CALUDE_subset_quadratic_linear_l257_25758

theorem subset_quadratic_linear (a : ℝ) : 
  let M : Set ℝ := {x | x^2 + x - 6 = 0}
  let N : Set ℝ := {x | a*x - 1 = 0}
  N ⊆ M → (a = 1/2 ∨ a = -1/3) :=
by
  sorry

#check subset_quadratic_linear

end NUMINAMATH_CALUDE_subset_quadratic_linear_l257_25758


namespace NUMINAMATH_CALUDE_base4_10203_equals_291_l257_25738

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_10203_equals_291 :
  base4_to_decimal [3, 0, 2, 0, 1] = 291 := by
  sorry

end NUMINAMATH_CALUDE_base4_10203_equals_291_l257_25738


namespace NUMINAMATH_CALUDE_range_of_a_l257_25778

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) → 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l257_25778


namespace NUMINAMATH_CALUDE_coefficient_x2y2_l257_25750

/-- The coefficient of x²y² in the expansion of (x+y)⁵(c+1/c)⁸ is 700 -/
theorem coefficient_x2y2 : 
  (Finset.sum Finset.univ (fun (k : Fin 6) => 
    Nat.choose 5 k.val * Nat.choose 8 4 * k.val.choose 2)) = 700 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_l257_25750


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l257_25705

/-- Given vectors a and b in ℝ², prove that if a + b is perpendicular to b, then the second component of a is 8. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ b = (3, -2)) :
  (a + b) • b = 0 → a.2 = 8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l257_25705
