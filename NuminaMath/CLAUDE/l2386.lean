import Mathlib

namespace NUMINAMATH_CALUDE_missing_ratio_l2386_238620

theorem missing_ratio (x y : ℚ) (h : x / y * (6 / 11) * (11 / 2) = 2) : x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_ratio_l2386_238620


namespace NUMINAMATH_CALUDE_complex_magnitude_l2386_238680

theorem complex_magnitude (z : ℂ) (h : z + Complex.I = z * Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2386_238680


namespace NUMINAMATH_CALUDE_simplify_fraction_l2386_238649

theorem simplify_fraction : 18 * (8 / 15) * (1 / 12) = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2386_238649


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l2386_238611

theorem set_equality_implies_sum_of_powers (x y : ℝ) : 
  ({x, x * y, x + y} : Set ℝ) = ({0, |x|, y} : Set ℝ) → x^2018 + y^2018 = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_of_powers_l2386_238611


namespace NUMINAMATH_CALUDE_one_cubic_yard_equals_27_cubic_feet_l2386_238678

-- Define the conversion rate between yards and feet
def yard_to_feet : ℝ := 3

-- Define a cubic yard in terms of cubic feet
def cubic_yard_to_cubic_feet : ℝ := yard_to_feet ^ 3

-- Theorem statement
theorem one_cubic_yard_equals_27_cubic_feet :
  cubic_yard_to_cubic_feet = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_cubic_yard_equals_27_cubic_feet_l2386_238678


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l2386_238626

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l2386_238626


namespace NUMINAMATH_CALUDE_prime_odd_sum_l2386_238630

theorem prime_odd_sum (x y : ℕ) : 
  Nat.Prime x → 
  Odd y → 
  x^2 + y = 2009 → 
  x + y = 2007 := by
sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l2386_238630


namespace NUMINAMATH_CALUDE_first_term_value_l2386_238625

/-- A geometric sequence with five terms -/
structure GeometricSequence :=
  (a b c d e : ℝ)
  (is_geometric : ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r)

/-- Theorem: In a geometric sequence where the fourth term is 81 and the fifth term is 243, the first term is 3 -/
theorem first_term_value (seq : GeometricSequence) 
  (h1 : seq.d = 81)
  (h2 : seq.e = 243) : 
  seq.a = 3 := by
sorry

end NUMINAMATH_CALUDE_first_term_value_l2386_238625


namespace NUMINAMATH_CALUDE_smallest_angle_is_180_div_7_l2386_238623

/-- An isosceles triangle that can be cut into two isosceles triangles -/
structure CuttableIsoscelesTriangle where
  /-- The measure of one of the equal angles in the original triangle -/
  α : ℝ
  /-- The original triangle is isosceles -/
  isosceles : α ≤ 90
  /-- The triangle can be cut into two isosceles triangles -/
  cuttable : ∃ (β γ : ℝ), (β = α ∧ γ = (180 - α) / 2) ∨ 
                           (β = (180 - α) / 2 ∧ γ = (180 - α) / 2) ∨
                           (β = 90 - α / 2 ∧ γ = α)

/-- The smallest angle in a CuttableIsoscelesTriangle is 180/7 -/
theorem smallest_angle_is_180_div_7 : 
  ∀ (t : CuttableIsoscelesTriangle), 
    min t.α (180 - 2 * t.α) ≥ 180 / 7 ∧ 
    ∃ (t : CuttableIsoscelesTriangle), min t.α (180 - 2 * t.α) = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_180_div_7_l2386_238623


namespace NUMINAMATH_CALUDE_m_range_l2386_238618

-- Define the propositions p and q
def p (m : ℝ) : Prop := m + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2386_238618


namespace NUMINAMATH_CALUDE_combination_equality_l2386_238682

theorem combination_equality (x : ℕ) : 
  (Nat.choose 14 x = Nat.choose 14 (2*x - 4)) → (x = 4 ∨ x = 6) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2386_238682


namespace NUMINAMATH_CALUDE_books_returned_percentage_l2386_238675

-- Define the initial number of books
def initial_books : ℕ := 75

-- Define the final number of books
def final_books : ℕ := 63

-- Define the number of books loaned out (rounded to 40)
def loaned_books : ℕ := 40

-- Define the percentage of books returned
def percentage_returned : ℚ := 70

-- Theorem statement
theorem books_returned_percentage :
  (((initial_books - final_books : ℚ) / loaned_books) * 100 = percentage_returned) :=
sorry

end NUMINAMATH_CALUDE_books_returned_percentage_l2386_238675


namespace NUMINAMATH_CALUDE_even_cube_diff_iff_even_sum_l2386_238639

theorem even_cube_diff_iff_even_sum (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end NUMINAMATH_CALUDE_even_cube_diff_iff_even_sum_l2386_238639


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2386_238660

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  let givenLine : Line := { a := 2, b := 1, c := -5 }
  let point : Point := { x := 3, y := 0 }
  let perpendicularLine : Line := { a := 1, b := -2, c := 3 }
  perpendicular givenLine perpendicularLine ∧ 
  pointOnLine point perpendicularLine := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2386_238660


namespace NUMINAMATH_CALUDE_gift_splitting_l2386_238686

theorem gift_splitting (initial_cost : ℝ) (dropout_count : ℕ) (extra_cost : ℝ) : 
  initial_cost = 120 ∧ 
  dropout_count = 4 ∧ 
  extra_cost = 8 →
  ∃ (n : ℕ), 
    n > dropout_count ∧
    initial_cost / (n - dropout_count : ℝ) = initial_cost / n + extra_cost ∧
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_gift_splitting_l2386_238686


namespace NUMINAMATH_CALUDE_meter_to_step_conversion_l2386_238676

-- Define our units of measurement
variable (hops skips jumps steps meters : ℚ)

-- Define the relationships between units
variable (hop_skip_relation : 2 * hops = 3 * skips)
variable (jump_hop_relation : 4 * jumps = 6 * hops)
variable (jump_meter_relation : 5 * jumps = 20 * meters)
variable (skip_step_relation : 15 * skips = 10 * steps)

-- State the theorem
theorem meter_to_step_conversion :
  1 * meters = 3/8 * steps :=
sorry

end NUMINAMATH_CALUDE_meter_to_step_conversion_l2386_238676


namespace NUMINAMATH_CALUDE_bridge_length_l2386_238647

/-- Given a train of length 120 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 255 meters. -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 120 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2386_238647


namespace NUMINAMATH_CALUDE_rectangle_tiling_l2386_238642

/-- A rectangle can be perfectly tiled by unit-width strips if and only if
    at least one of its dimensions is an integer. -/
theorem rectangle_tiling (a b : ℝ) :
  (∃ (n : ℕ), a * b = n) →
  (∃ (k : ℕ), a = k ∨ b = k) := by sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l2386_238642


namespace NUMINAMATH_CALUDE_eighth_term_value_l2386_238640

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- Seventh term is 8
  seventh_term : a + 6*d = 8

/-- The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : seq.a + 7*seq.d = 65/7 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2386_238640


namespace NUMINAMATH_CALUDE_n_has_24_digits_l2386_238655

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 12 -/
axiom n_div_12 : 12 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n^4 is a perfect fifth power -/
axiom n_fourth_fifth : ∃ k : ℕ, n^4 = k^5

/-- n is the smallest positive integer satisfying all conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (12 ∣ m) → (∃ k : ℕ, m^2 = k^3) → 
  (∃ k : ℕ, m^3 = k^2) → (∃ k : ℕ, m^4 = k^5) → m ≥ n

/-- Function to count the number of digits in a natural number -/
def digit_count (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 24 digits -/
theorem n_has_24_digits : digit_count n = 24 := sorry

end NUMINAMATH_CALUDE_n_has_24_digits_l2386_238655


namespace NUMINAMATH_CALUDE_roots_negative_of_each_other_l2386_238637

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s, 
    if r = -s, then b = 0 -/
theorem roots_negative_of_each_other 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) 
  (h4 : r = -s) : 
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_negative_of_each_other_l2386_238637


namespace NUMINAMATH_CALUDE_gcd_840_1764_l2386_238628

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l2386_238628


namespace NUMINAMATH_CALUDE_cannot_reach_54_from_12_l2386_238614

def Operation := Nat → Nat

def isValidOperation (op : Operation) : Prop :=
  ∀ n, (op n = 2 * n) ∨ (op n = 3 * n) ∨ (op n = n / 2) ∨ (op n = n / 3)

def applyOperations (ops : List Operation) (start : Nat) : Nat :=
  ops.foldl (λ acc op => op acc) start

theorem cannot_reach_54_from_12 :
  ¬ ∃ (ops : List Operation),
    (ops.length = 60) ∧
    (∀ op ∈ ops, isValidOperation op) ∧
    (applyOperations ops 12 = 54) :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_54_from_12_l2386_238614


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2386_238673

theorem fraction_evaluation :
  ⌈(23 / 11 : ℚ) - ⌈(37 / 19 : ℚ)⌉⌉ / ⌈(35 / 11 : ℚ) + ⌈(11 * 19 / 37 : ℚ)⌉⌉ = (1 / 10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2386_238673


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_2007_to_1024_minus_1_l2386_238683

theorem largest_power_of_two_dividing_2007_to_1024_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (2007^1024 - 1)) ∧
  (∀ (m : ℕ), m > 14 → ¬(2^m ∣ (2007^1024 - 1))) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_2007_to_1024_minus_1_l2386_238683


namespace NUMINAMATH_CALUDE_birds_in_tree_l2386_238687

/-- Given a tree with an initial number of birds and additional birds that fly up to it,
    prove that the total number of birds is the sum of the initial and additional birds. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2386_238687


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_inequality_l2386_238696

theorem right_triangle_sine_cosine_inequality 
  (A B C : Real) 
  (h_right_angle : C = Real.pi / 2) 
  (h_acute_A : 0 < A ∧ A < Real.pi / 4) :
  Real.sin B > Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_inequality_l2386_238696


namespace NUMINAMATH_CALUDE_unknown_interest_rate_l2386_238661

/-- Proves that given the conditions of the problem, the unknown interest rate is 6% -/
theorem unknown_interest_rate (total : ℚ) (part1 : ℚ) (part2 : ℚ) (rate1 : ℚ) (rate2 : ℚ) (yearly_income : ℚ) :
  total = 2600 →
  part1 = 1600 →
  part2 = total - part1 →
  rate1 = 5 / 100 →
  yearly_income = 140 →
  yearly_income = part1 * rate1 + part2 * rate2 →
  rate2 = 6 / 100 := by
sorry

#eval (6 : ℚ) / 100

end NUMINAMATH_CALUDE_unknown_interest_rate_l2386_238661


namespace NUMINAMATH_CALUDE_smallest_independent_after_reorganization_l2386_238654

/-- Represents a faction of deputies -/
structure Faction :=
  (size : ℕ)

/-- Represents the parliament configuration -/
structure Parliament :=
  (factions : List Faction)
  (independent : ℕ)

def initialParliament : Parliament :=
  { factions := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].map (λ n => ⟨n⟩),
    independent := 0 }

def totalDeputies (p : Parliament) : ℕ :=
  p.factions.foldl (λ acc f => acc + f.size) p.independent

def isValidReorganization (initial final : Parliament) : Prop :=
  totalDeputies initial = totalDeputies final ∧
  final.factions.all (λ f => f.size ≤ initial.factions.length) ∧
  final.factions.all (λ f => f.size ≥ 5)

theorem smallest_independent_after_reorganization :
  ∀ (final : Parliament),
    isValidReorganization initialParliament final →
    final.independent ≥ 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_independent_after_reorganization_l2386_238654


namespace NUMINAMATH_CALUDE_football_game_attendance_l2386_238651

/-- Football game attendance problem -/
theorem football_game_attendance 
  (saturday_attendance : ℕ)
  (monday_attendance : ℕ)
  (wednesday_attendance : ℕ)
  (friday_attendance : ℕ)
  (expected_total : ℕ)
  (actual_total : ℕ)
  (h1 : saturday_attendance = 80)
  (h2 : monday_attendance = saturday_attendance - 20)
  (h3 : wednesday_attendance > monday_attendance)
  (h4 : friday_attendance = saturday_attendance + monday_attendance)
  (h5 : expected_total = 350)
  (h6 : actual_total = expected_total + 40)
  (h7 : actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance) :
  wednesday_attendance - monday_attendance = 50 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l2386_238651


namespace NUMINAMATH_CALUDE_vector_relation_in_right_triangular_prism_l2386_238632

/-- A right triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightTriangularPrism (V : Type*) [AddCommGroup V] :=
  (A B C A₁ B₁ C₁ : V)

/-- The theorem stating the relation between vectors in a right triangular prism -/
theorem vector_relation_in_right_triangular_prism
  {V : Type*} [AddCommGroup V] (prism : RightTriangularPrism V)
  (a b c : V)
  (h1 : prism.C - prism.A = a)
  (h2 : prism.C - prism.B = b)
  (h3 : prism.C - prism.C₁ = c) :
  prism.A₁ - prism.B = -a - c + b := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_in_right_triangular_prism_l2386_238632


namespace NUMINAMATH_CALUDE_quadratic_inequality_unique_solution_l2386_238663

theorem quadratic_inequality_unique_solution (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 + a*x + 5 ∧ x^2 + a*x + 5 ≤ 4) ↔ (a = 2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_unique_solution_l2386_238663


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l2386_238633

theorem crushing_load_calculation (T H K : ℝ) (hT : T = 3) (hH : H = 9) (hK : K = 2) :
  (50 * T^5) / (K * H^3) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l2386_238633


namespace NUMINAMATH_CALUDE_line_slope_through_point_with_x_intercept_l2386_238627

/-- Given a line passing through the point (3, 4) with an x-intercept of 1, 
    its slope is 2. -/
theorem line_slope_through_point_with_x_intercept : 
  ∀ (f : ℝ → ℝ), 
    (∃ m b : ℝ, ∀ x, f x = m * x + b) →  -- f is a linear function
    f 3 = 4 →                           -- f passes through (3, 4)
    f 1 = 0 →                           -- x-intercept is 1
    ∃ m : ℝ, (∀ x, f x = m * x + b) ∧ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_line_slope_through_point_with_x_intercept_l2386_238627


namespace NUMINAMATH_CALUDE_cookie_ratio_anna_to_tim_l2386_238650

/-- Represents the distribution of cookies among recipients --/
structure CookieDistribution where
  total : Nat
  tim : Nat
  mike : Nat
  fridge : Nat

/-- Calculates the number of cookies given to Anna --/
def cookiesForAnna (d : CookieDistribution) : Nat :=
  d.total - (d.tim + d.mike + d.fridge)

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : Nat
  denominator : Nat

/-- Theorem stating the ratio of cookies given to Anna to cookies given to Tim --/
theorem cookie_ratio_anna_to_tim (d : CookieDistribution)
  (h1 : d.total = 256)
  (h2 : d.tim = 15)
  (h3 : d.mike = 23)
  (h4 : d.fridge = 188) :
  Ratio.mk (cookiesForAnna d) d.tim = Ratio.mk 2 1 := by
  sorry

#check cookie_ratio_anna_to_tim

end NUMINAMATH_CALUDE_cookie_ratio_anna_to_tim_l2386_238650


namespace NUMINAMATH_CALUDE_max_triangle_area_in_three_squares_l2386_238631

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in the plane -/
structure Square where
  center : Point
  side : ℝ

/-- Definition of a unit square -/
def isUnitSquare (s : Square) : Prop := s.side = 1

/-- Definition of a point being contained in a square -/
def isContainedIn (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : Point) : ℝ :=
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

/-- The main theorem -/
theorem max_triangle_area_in_three_squares 
  (s₁ s₂ s₃ : Square) 
  (h₁ : isUnitSquare s₁) 
  (h₂ : isUnitSquare s₂) 
  (h₃ : isUnitSquare s₃) 
  (X : Point) 
  (hX₁ : isContainedIn X s₁) 
  (hX₂ : isContainedIn X s₂) 
  (hX₃ : isContainedIn X s₃) 
  (A B C : Point) 
  (hA : isContainedIn A s₁ ∨ isContainedIn A s₂ ∨ isContainedIn A s₃)
  (hB : isContainedIn B s₁ ∨ isContainedIn B s₂ ∨ isContainedIn B s₃)
  (hC : isContainedIn C s₁ ∨ isContainedIn C s₂ ∨ isContainedIn C s₃) :
  triangleArea A B C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_three_squares_l2386_238631


namespace NUMINAMATH_CALUDE_melissa_bananas_l2386_238636

/-- Calculates the number of bananas Melissa has left after sharing some. -/
def bananas_left (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Proves that Melissa has 84 bananas left after sharing 4 out of her initial 88 bananas. -/
theorem melissa_bananas : bananas_left 88 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_melissa_bananas_l2386_238636


namespace NUMINAMATH_CALUDE_final_lives_correct_tiffany_final_lives_l2386_238656

/-- Given an initial number of lives, lives lost, and a bonus multiplier,
    calculate the final number of lives after completing the bonus stage. -/
def finalLives (initialLives lostLives bonusMultiplier : ℕ) : ℕ :=
  let remainingLives := initialLives - lostLives
  remainingLives + bonusMultiplier * remainingLives

/-- Theorem: The final number of lives after the bonus stage is correct. -/
theorem final_lives_correct (initialLives lostLives bonusMultiplier : ℕ) 
    (h : lostLives ≤ initialLives) :
    finalLives initialLives lostLives bonusMultiplier = 
    (initialLives - lostLives) + bonusMultiplier * (initialLives - lostLives) := by
  sorry

/-- Corollary: For the specific case in the problem. -/
theorem tiffany_final_lives : 
    finalLives 250 58 3 = 768 := by
  sorry

end NUMINAMATH_CALUDE_final_lives_correct_tiffany_final_lives_l2386_238656


namespace NUMINAMATH_CALUDE_assignment_n_plus_one_increases_by_one_l2386_238615

/-- Represents a variable in a programming language -/
structure Variable where
  name : String
  value : Int

/-- Represents an expression in a programming language -/
inductive Expression where
  | Const : Int → Expression
  | Var : Variable → Expression
  | Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language -/
structure AssignmentStatement where
  lhs : Variable
  rhs : Expression

/-- Evaluates an expression given the current state of variables -/
def evalExpression (expr : Expression) (state : List Variable) : Int :=
  match expr with
  | Expression.Const n => n
  | Expression.Var v => v.value
  | Expression.Add e1 e2 => evalExpression e1 state + evalExpression e2 state

/-- Executes an assignment statement and returns the updated state -/
def executeAssignment (stmt : AssignmentStatement) (state : List Variable) : List Variable :=
  let newValue := evalExpression stmt.rhs state
  state.map fun v => if v.name = stmt.lhs.name then { v with value := newValue } else v

/-- Theorem: N=N+1 increases the value of N by 1 -/
theorem assignment_n_plus_one_increases_by_one (n : Variable) (state : List Variable) :
  let stmt : AssignmentStatement := { lhs := n, rhs := Expression.Add (Expression.Var n) (Expression.Const 1) }
  let newState := executeAssignment stmt state
  let oldValue := (state.find? fun v => v.name = n.name).map (fun v => v.value)
  let newValue := (newState.find? fun v => v.name = n.name).map (fun v => v.value)
  (oldValue.isSome ∧ newValue.isSome) →
  newValue = oldValue.map (fun v => v + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_assignment_n_plus_one_increases_by_one_l2386_238615


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l2386_238665

theorem quadratic_inequality_implication (x : ℝ) :
  x^2 - 5*x + 6 < 0 → 20 < x^2 + 5*x + 6 ∧ x^2 + 5*x + 6 < 30 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l2386_238665


namespace NUMINAMATH_CALUDE_problem_solution_l2386_238601

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 42)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2386_238601


namespace NUMINAMATH_CALUDE_office_canteen_round_tables_l2386_238652

theorem office_canteen_round_tables :
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let total_chairs : ℕ := 26
  
  ∃ (round_tables : ℕ),
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table = total_chairs ∧
    round_tables = 2 :=
by sorry

end NUMINAMATH_CALUDE_office_canteen_round_tables_l2386_238652


namespace NUMINAMATH_CALUDE_donation_is_45_l2386_238634

/-- The total donation to the class funds given the number of stuffed animals and selling prices -/
def total_donation (barbara_stuffed_animals : ℕ) (barbara_price : ℚ) (trish_price : ℚ) : ℚ :=
  let trish_stuffed_animals := 2 * barbara_stuffed_animals
  barbara_stuffed_animals * barbara_price + trish_stuffed_animals * trish_price

/-- Proof that the total donation is $45 given the specific conditions -/
theorem donation_is_45 :
  total_donation 9 2 (3/2) = 45 := by
  sorry

#eval total_donation 9 2 (3/2)

end NUMINAMATH_CALUDE_donation_is_45_l2386_238634


namespace NUMINAMATH_CALUDE_chess_competition_result_l2386_238667

/-- Represents the number of 8th-grade students in the chess competition. -/
def n : ℕ := sorry

/-- Represents the score of each 8th-grade student. -/
def σ : ℚ := sorry

/-- The total points scored by the two 7th-grade students. -/
def seventh_grade_total : ℕ := 8

/-- The theorem stating the conditions and the result of the chess competition. -/
theorem chess_competition_result :
  (∃ (n : ℕ) (σ : ℚ),
    n > 0 ∧
    σ = (2 * n - 7 : ℚ) / n ∧
    σ = 2 - 7 / n ∧
    (σ = 1 ∨ σ = (3 : ℚ) / 2) ∧
    n = 7) :=
sorry

end NUMINAMATH_CALUDE_chess_competition_result_l2386_238667


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2386_238662

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_value_implies_a (a : ℝ) :
  (∀ x, a - 2 ≤ x ∧ x ≤ a + 1 → f x ≤ 3) ∧
  (∃ x, a - 2 ≤ x ∧ x ≤ a + 1 ∧ f x = 3) →
  a = 0 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2386_238662


namespace NUMINAMATH_CALUDE_limit_of_a_is_one_l2386_238669

def a : ℕ+ → ℚ
  | n => if n < 10000 then (2^(n.val+1)) / (2^n.val+1) else ((n.val+1)^2) / (n.val^2+1)

theorem limit_of_a_is_one :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |a n - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_a_is_one_l2386_238669


namespace NUMINAMATH_CALUDE_sin_beta_value_l2386_238646

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5 / 13)
  (h4 : Real.sin α = 4 / 5) :
  Real.sin β = -56 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2386_238646


namespace NUMINAMATH_CALUDE_problem_solution_l2386_238616

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 12 < n.val + 3)
  (h2 : (m.val + (m.val + 6) + (m.val + 12) + (n.val + 3) + (n.val + 6) + 3 * n.val) / 6 = n.val + 3)
  (h3 : (m.val + 12 + n.val + 3) / 2 = n.val + 3) : 
  m.val + n.val = 57 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2386_238616


namespace NUMINAMATH_CALUDE_imaginary_power_minus_fraction_l2386_238679

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_minus_fraction : i^7 - 2 / i = i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_minus_fraction_l2386_238679


namespace NUMINAMATH_CALUDE_reading_difference_l2386_238648

/-- The number of pages Janet reads per day -/
def janet_pages_per_day : ℕ := 80

/-- The number of pages Belinda reads per day -/
def belinda_pages_per_day : ℕ := 30

/-- The number of weeks in the reading period -/
def reading_weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem reading_difference :
  (janet_pages_per_day - belinda_pages_per_day) * (reading_weeks * days_per_week) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_reading_difference_l2386_238648


namespace NUMINAMATH_CALUDE_infinite_fraction_reciprocal_l2386_238681

theorem infinite_fraction_reciprocal (y : ℝ) : 
  y = 1 + (Real.sqrt 3) / (1 + (Real.sqrt 3) / (1 + y)) → 
  1 / ((y + 1) * (y - 2)) = -(Real.sqrt 3) - 2 :=
by sorry

end NUMINAMATH_CALUDE_infinite_fraction_reciprocal_l2386_238681


namespace NUMINAMATH_CALUDE_inequality_proof_l2386_238604

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b)^2 + b * c / (b + c)^2 + c * a / (c + a)^2) + 
  3 * (a^2 + b^2 + c^2) / (a + b + c)^2 ≥ 7/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2386_238604


namespace NUMINAMATH_CALUDE_laura_shirt_count_l2386_238684

def pants_count : ℕ := 2
def pants_price : ℕ := 54
def shirt_price : ℕ := 33
def money_given : ℕ := 250
def change_received : ℕ := 10

theorem laura_shirt_count :
  (money_given - change_received - pants_count * pants_price) / shirt_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_laura_shirt_count_l2386_238684


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l2386_238621

/-- Given a geometric sequence {a_n} where a₄ = 7 and a₈ = 63, prove that a₆ = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 = 7 →                                  -- given a₄ = 7
  a 8 = 63 →                                 -- given a₈ = 63
  a 6 = 21 :=                                -- prove a₆ = 21
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l2386_238621


namespace NUMINAMATH_CALUDE_fred_final_card_count_l2386_238600

/-- Calculates the final number of baseball cards Fred has after a series of transactions. -/
def final_card_count (initial : ℕ) (given_away : ℕ) (traded : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Proves that Fred ends up with 6 baseball cards after the given transactions. -/
theorem fred_final_card_count :
  final_card_count 5 2 1 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_card_count_l2386_238600


namespace NUMINAMATH_CALUDE_mouse_cheese_distance_sum_l2386_238658

/-- The point where the mouse begins moving away from the cheese -/
def mouse_turn_point (c d : ℝ) : Prop :=
  ∃ (k : ℝ), 
    d = -3 * c + 18 ∧  -- Mouse path
    d - 5 = k * (c - 20) ∧  -- Perpendicular line
    -3 * k = -1  -- Perpendicular condition

/-- The theorem stating the sum of coordinates where the mouse turns -/
theorem mouse_cheese_distance_sum : 
  ∃ (c d : ℝ), mouse_turn_point c d ∧ c + d = 9.4 :=
sorry

end NUMINAMATH_CALUDE_mouse_cheese_distance_sum_l2386_238658


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2386_238695

theorem rectangle_dimensions (vertical_side : ℝ) (square_side : ℝ) (horizontal_side : ℝ) : 
  vertical_side = 28 →
  square_side = 10 →
  (vertical_side - square_side) ^ 2 + (horizontal_side - square_side) ^ 2 = vertical_side ^ 2 →
  horizontal_side = 45 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2386_238695


namespace NUMINAMATH_CALUDE_mrs_brown_other_bills_value_l2386_238622

/-- Represents the utility bill payment scenario for Mrs. Brown -/
structure UtilityPayment where
  totalBill : ℕ
  tenDollarBillsUsed : ℕ
  tenDollarBillValue : ℕ

/-- Calculates the value of other bills used in the utility payment -/
def otherBillsValue (payment : UtilityPayment) : ℕ :=
  payment.totalBill - (payment.tenDollarBillsUsed * payment.tenDollarBillValue)

/-- Theorem stating that the value of other bills used by Mrs. Brown is $150 -/
theorem mrs_brown_other_bills_value :
  let payment : UtilityPayment := {
    totalBill := 170,
    tenDollarBillsUsed := 2,
    tenDollarBillValue := 10
  }
  otherBillsValue payment = 150 := by
  sorry

end NUMINAMATH_CALUDE_mrs_brown_other_bills_value_l2386_238622


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2386_238653

/-- Given a hyperbola with equation x²/8 - y²/2 = 1, its asymptotes have the equations y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 8 - y^2 / 2 = 1 →
  ∃ (k : ℝ), k = 1/2 ∧ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2386_238653


namespace NUMINAMATH_CALUDE_intersection_condition_implies_m_leq_neg_one_l2386_238672

/-- Given sets A and B, prove that if A ∩ B = A, then m ≤ -1 -/
theorem intersection_condition_implies_m_leq_neg_one (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| < 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∩ B = A → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_m_leq_neg_one_l2386_238672


namespace NUMINAMATH_CALUDE_coin_experiment_results_l2386_238624

/-- A fair coin is flipped 100 times with 48 heads observed. -/
structure CoinExperiment where
  total_flips : ℕ
  heads_count : ℕ
  is_fair : Bool
  h_total : total_flips = 100
  h_heads : heads_count = 48
  h_fair : is_fair = true

/-- The frequency of heads in a coin experiment. -/
def frequency (e : CoinExperiment) : ℚ :=
  e.heads_count / e.total_flips

/-- The theoretical probability of heads for a fair coin. -/
def fair_coin_probability : ℚ := 1 / 2

theorem coin_experiment_results (e : CoinExperiment) :
  frequency e = 48 / 100 ∧ fair_coin_probability = 1 / 2 := by
  sorry

#eval (48 : ℚ) / 100  -- To show that 48/100 evaluates to 0.48
#eval (1 : ℚ) / 2     -- To show that 1/2 evaluates to 0.5

end NUMINAMATH_CALUDE_coin_experiment_results_l2386_238624


namespace NUMINAMATH_CALUDE_ratio_change_l2386_238699

theorem ratio_change (x y : ℚ) : 
  x / y = 3 / 4 → 
  y = 40 → 
  (x + 10) / (y + 10) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_change_l2386_238699


namespace NUMINAMATH_CALUDE_contractor_absence_l2386_238659

theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧
  daily_pay = 25 ∧
  daily_fine = (15/2) ∧
  total_amount = 685 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_pay * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 2 :=
by sorry

end NUMINAMATH_CALUDE_contractor_absence_l2386_238659


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l2386_238677

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → y ∈ Set.Ioo (-9 : ℝ) (-8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l2386_238677


namespace NUMINAMATH_CALUDE_remainder_difference_l2386_238698

theorem remainder_difference (m n : ℕ) (hm : m % 6 = 2) (hn : n % 6 = 3) (h_gt : m > n) :
  (m - n) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_difference_l2386_238698


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2386_238674

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the "not contained in" relation between a line and a plane
variable (notContainedIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane (l : Line) (α : Plane) :
  notContainedIn l α → parallel l α := by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2386_238674


namespace NUMINAMATH_CALUDE_hexagon_theorem_l2386_238638

/-- Regular hexagon with side length 4 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ := 4)

/-- Intersection point of diagonals CE and DF -/
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Point K defined by vector equation -/
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Predicate for a point being outside a hexagon -/
def is_outside (p : ℝ × ℝ) (hex : RegularHexagon) : Prop := sorry

theorem hexagon_theorem (hex : RegularHexagon) :
  is_outside (K hex) hex ∧ distance (K hex) hex.A = (4 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_theorem_l2386_238638


namespace NUMINAMATH_CALUDE_true_discount_example_l2386_238664

/-- Given a banker's discount and face value, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (face_value : ℚ) : ℚ :=
  (face_value * bankers_discount) / (face_value + bankers_discount)

/-- Theorem stating that for given values, the true discount is 480 -/
theorem true_discount_example : true_discount 576 2880 = 480 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_example_l2386_238664


namespace NUMINAMATH_CALUDE_rhombus_properties_l2386_238657

structure Rhombus where
  points : Fin 4 → ℝ × ℝ
  is_rhombus : ∀ i j : Fin 4, i ≠ j → dist (points i) (points j) = dist (points ((i+1) % 4)) (points ((j+1) % 4))

def diagonal1 (r : Rhombus) : ℝ × ℝ := r.points 0 - r.points 2
def diagonal2 (r : Rhombus) : ℝ × ℝ := r.points 1 - r.points 3

theorem rhombus_properties (r : Rhombus) :
  (∃ m : ℝ, diagonal1 r = m • (diagonal2 r)) ∧ 
  (diagonal1 r • diagonal2 r = 0) ∧
  (∀ i : Fin 4, dist (r.points i) (r.points ((i+1) % 4)) = dist (r.points ((i+1) % 4)) (r.points ((i+2) % 4))) ∧
  (¬ ∀ r : Rhombus, ‖diagonal1 r‖ = ‖diagonal2 r‖) := by
  sorry

#check rhombus_properties

end NUMINAMATH_CALUDE_rhombus_properties_l2386_238657


namespace NUMINAMATH_CALUDE_equation_solution_l2386_238690

theorem equation_solution : 
  ∃ y : ℚ, (40 / 70)^2 = Real.sqrt (y / 70) → y = 17920 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2386_238690


namespace NUMINAMATH_CALUDE_stickers_used_for_decoration_l2386_238693

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 26
def birthday_stickers : ℕ := 20
def given_away_stickers : ℕ := 6
def left_stickers : ℕ := 2

theorem stickers_used_for_decoration :
  initial_stickers + bought_stickers + birthday_stickers - given_away_stickers - left_stickers = 58 :=
by sorry

end NUMINAMATH_CALUDE_stickers_used_for_decoration_l2386_238693


namespace NUMINAMATH_CALUDE_suresh_work_hours_l2386_238691

/-- Proves that Suresh worked for 9 hours given the conditions of the problem -/
theorem suresh_work_hours 
  (suresh_rate : ℚ) 
  (ashutosh_rate : ℚ) 
  (ashutosh_remaining_hours : ℚ) 
  (h1 : suresh_rate = 1 / 15)
  (h2 : ashutosh_rate = 1 / 20)
  (h3 : ashutosh_remaining_hours = 8)
  : ∃ x : ℚ, x * suresh_rate + ashutosh_remaining_hours * ashutosh_rate = 1 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_suresh_work_hours_l2386_238691


namespace NUMINAMATH_CALUDE_production_rates_l2386_238645

/-- The rate at which A makes parts per hour -/
def rate_A : ℝ := sorry

/-- The rate at which B makes parts per hour -/
def rate_B : ℝ := sorry

/-- The time it takes for A to make 90 parts equals the time for B to make 120 parts -/
axiom time_ratio : (90 / rate_A) = (120 / rate_B)

/-- A and B together make 35 parts per hour -/
axiom total_rate : rate_A + rate_B = 35

theorem production_rates : rate_A = 15 ∧ rate_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_production_rates_l2386_238645


namespace NUMINAMATH_CALUDE_bookshelf_problem_l2386_238670

/-- Bookshelf purchasing problem -/
theorem bookshelf_problem (price_A price_B : ℕ) 
  (h1 : 3 * price_A + 2 * price_B = 1020)
  (h2 : 4 * price_A + 3 * price_B = 1440)
  (total_bookshelves : ℕ) (h3 : total_bookshelves = 20)
  (max_budget : ℕ) (h4 : max_budget = 4320) :
  (price_A = 180 ∧ price_B = 240) ∧ 
  (∃ (m : ℕ), 
    (m = 8 ∨ m = 9 ∨ m = 10) ∧
    (total_bookshelves - m ≥ m) ∧
    (price_A * m + price_B * (total_bookshelves - m) ≤ max_budget)) := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l2386_238670


namespace NUMINAMATH_CALUDE_inequality_solution_l2386_238612

def inequality (a x : ℝ) : Prop :=
  (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

theorem inequality_solution :
  (∀ a : ℝ, inequality a x) ↔ x = -2 ∨ x = 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2386_238612


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2386_238602

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
  p^2 + q^2 + r^2 = 34/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2386_238602


namespace NUMINAMATH_CALUDE_rational_numbers_include_zero_not_only_positive_and_negative_rationals_l2386_238635

theorem rational_numbers_include_zero : ∃ (x : ℚ), x ≠ 0 ∧ x ≥ 0 ∧ x ≤ 0 := by
  sorry

theorem not_only_positive_and_negative_rationals : 
  ¬(∀ (x : ℚ), x > 0 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_include_zero_not_only_positive_and_negative_rationals_l2386_238635


namespace NUMINAMATH_CALUDE_boys_between_rajan_and_vinay_l2386_238617

theorem boys_between_rajan_and_vinay (total_boys : ℕ) (rajan_position : ℕ) (vinay_position : ℕ)
  (h1 : total_boys = 24)
  (h2 : rajan_position = 6)
  (h3 : vinay_position = 10) :
  total_boys - (rajan_position - 1 + vinay_position - 1 + 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_between_rajan_and_vinay_l2386_238617


namespace NUMINAMATH_CALUDE_replaced_men_age_sum_l2386_238619

/-- Given a group of 8 men where replacing two of them with two women increases the average age by 2 years,
    and the average age of the women is 32 years, prove that the combined age of the two replaced men is 48 years. -/
theorem replaced_men_age_sum (n : ℕ) (A : ℝ) (women_avg_age : ℝ) :
  n = 8 ∧ women_avg_age = 32 →
  ∃ (older_man_age younger_man_age : ℝ),
    n * (A + 2) = (n - 2) * A + 2 * women_avg_age ∧
    older_man_age + younger_man_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_replaced_men_age_sum_l2386_238619


namespace NUMINAMATH_CALUDE_complex_imaginary_part_eq_two_l2386_238605

theorem complex_imaginary_part_eq_two (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_eq_two_l2386_238605


namespace NUMINAMATH_CALUDE_parallel_tangents_and_function_inequality_l2386_238608

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2 * Real.log x

/-- The function g(x) defined in the problem -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*a*x - (2*a + 1) + 2/x

theorem parallel_tangents_and_function_inequality (a : ℝ) (h_a : a > 0) :
  (f_deriv a 1 = f_deriv a 3 → a = 1/12) ∧
  (∀ x₁ ∈ Set.Ioc 0 2, ∃ x₂ ∈ Set.Ioc 0 2, f a x₁ < g x₂) ↔ 
  (0 < a ∧ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_parallel_tangents_and_function_inequality_l2386_238608


namespace NUMINAMATH_CALUDE_maddie_makeup_palettes_l2386_238685

/-- The number of makeup palettes Maddie bought -/
def num_palettes : ℕ := 3

/-- The cost of each makeup palette in dollars -/
def palette_cost : ℚ := 15

/-- The total cost of lipsticks in dollars -/
def lipstick_cost : ℚ := 10

/-- The total cost of hair color boxes in dollars -/
def hair_color_cost : ℚ := 12

/-- The total amount Maddie paid in dollars -/
def total_paid : ℚ := 67

/-- Theorem stating that the number of makeup palettes Maddie bought is correct -/
theorem maddie_makeup_palettes : 
  (num_palettes : ℚ) * palette_cost + lipstick_cost + hair_color_cost = total_paid := by
  sorry

#check maddie_makeup_palettes

end NUMINAMATH_CALUDE_maddie_makeup_palettes_l2386_238685


namespace NUMINAMATH_CALUDE_perfect_squares_and_multiple_of_40_l2386_238607

theorem perfect_squares_and_multiple_of_40 :
  ∃ n : ℤ, ∃ a b : ℤ,
    (2 * n + 1 = a^2) ∧
    (3 * n + 1 = b^2) ∧
    (∃ k : ℤ, n = 40 * k) :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_and_multiple_of_40_l2386_238607


namespace NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l2386_238603

theorem unique_number_with_special_divisor_property : 
  ∃! (N : ℕ), 
    N > 0 ∧ 
    (∃ (k : ℕ), 
      N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisor_property_l2386_238603


namespace NUMINAMATH_CALUDE_min_value_of_f_l2386_238671

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f (a : ℝ) (h1 : a > 2) 
  (h2 : ∀ x > 2, f x ≥ f a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2386_238671


namespace NUMINAMATH_CALUDE_town_population_proof_l2386_238692

theorem town_population_proof (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_population : ℕ) :
  new_people = 100 →
  moved_out = 400 →
  years = 4 →
  final_population = 60 →
  (∃ original_population : ℕ,
    original_population = 1260 ∧
    final_population = ((original_population + new_people - moved_out) / 2^years)) :=
by sorry

end NUMINAMATH_CALUDE_town_population_proof_l2386_238692


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2386_238606

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the angle C to be 60°
def angle_C_60 (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for ∠C = 60° here
  True

-- Define D as the point where altitude from C meets AB
def altitude_C_D (A B C D : ℝ × ℝ) : Prop :=
  -- Add condition for D being on altitude from C here
  True

-- Define that the sides of triangle ABC are integers
def integer_sides (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for integer sides here
  True

-- Define BD = 17³
def BD_17_cubed (B D : ℝ × ℝ) : Prop :=
  -- Add condition for BD = 17³ here
  True

-- Define cos B = m/n where m and n are relatively prime positive integers
def cos_B_frac (B : ℝ × ℝ) (m n : ℕ) : Prop :=
  -- Add conditions for cos B = m/n and m, n coprime here
  True

theorem triangle_cosine_sum (A B C D : ℝ × ℝ) (m n : ℕ) :
  triangle_ABC A B C →
  angle_C_60 A B C →
  altitude_C_D A B C D →
  integer_sides A B C →
  BD_17_cubed B D →
  cos_B_frac B m n →
  m + n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2386_238606


namespace NUMINAMATH_CALUDE_bridge_length_l2386_238610

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 → 
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2386_238610


namespace NUMINAMATH_CALUDE_park_visitors_l2386_238666

theorem park_visitors (hikers bike_riders total : ℕ) : 
  hikers = 427 →
  hikers = bike_riders + 178 →
  total = hikers + bike_riders →
  total = 676 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l2386_238666


namespace NUMINAMATH_CALUDE_adam_marbles_l2386_238689

theorem adam_marbles (greg_marbles : ℕ) (greg_more_than_adam : ℕ) 
  (h1 : greg_marbles = 43)
  (h2 : greg_more_than_adam = 14) :
  greg_marbles - greg_more_than_adam = 29 := by
  sorry

end NUMINAMATH_CALUDE_adam_marbles_l2386_238689


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l2386_238688

theorem chord_length_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0 ∧ a*x + y + 1 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 - 2*a*x₁ + a = 0 ∧ 
    x₂^2 + y₂^2 - 2*a*x₂ + a = 0 ∧
    a*x₁ + y₁ + 1 = 0 ∧ 
    a*x₂ + y₂ + 1 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_value_l2386_238688


namespace NUMINAMATH_CALUDE_problem_solution_l2386_238697

theorem problem_solution (x y : ℝ) : 
  x = 0.7 * y →
  x = 210 →
  y = 300 ∧ ¬(∃ k : ℤ, y = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2386_238697


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l2386_238641

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l2386_238641


namespace NUMINAMATH_CALUDE_train_crossing_time_l2386_238694

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time : 
  ∀ (train_length : ℝ) (train_speed : ℝ),
  train_length = 120 →
  train_speed = 27 →
  (2 * train_length) / (2 * train_speed * (1000 / 3600)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2386_238694


namespace NUMINAMATH_CALUDE_popcorn_yield_two_tablespoons_yield_l2386_238629

/-- Represents the ratio of cups of popcorn to tablespoons of kernels -/
def popcorn_ratio (cups : ℚ) (tablespoons : ℚ) : Prop :=
  cups / tablespoons = 2

theorem popcorn_yield (cups : ℚ) (tablespoons : ℚ) 
  (h : popcorn_ratio 16 8) : 
  popcorn_ratio cups tablespoons → cups = 2 * tablespoons :=
by
  sorry

/-- Shows that 2 tablespoons of kernels make 4 cups of popcorn -/
theorem two_tablespoons_yield (h : popcorn_ratio 16 8) : 
  popcorn_ratio 4 2 :=
by
  sorry

end NUMINAMATH_CALUDE_popcorn_yield_two_tablespoons_yield_l2386_238629


namespace NUMINAMATH_CALUDE_clients_using_radio_l2386_238609

/-- The number of clients using radio in an advertising agency with given client distribution. -/
theorem clients_using_radio (total : ℕ) (tv : ℕ) (mag : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) :
  total = 180 →
  tv = 115 →
  mag = 130 →
  tv_mag = 85 →
  tv_radio = 75 →
  radio_mag = 95 →
  all_three = 80 →
  ∃ radio : ℕ, radio = 30 ∧ 
    total = tv + radio + mag - tv_mag - tv_radio - radio_mag + all_three :=
by
  sorry


end NUMINAMATH_CALUDE_clients_using_radio_l2386_238609


namespace NUMINAMATH_CALUDE_percentage_of_a_l2386_238643

-- Define the four numbers
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := a = 0.12 * b
def condition2 : Prop := b = 0.40 * c
def condition3 : Prop := c = 0.75 * d
def condition4 : Prop := d = 1.50 * (a + b)

-- Define the theorem
theorem percentage_of_a (h1 : condition1 a b) (h2 : condition2 b c) 
                        (h3 : condition3 c d) (h4 : condition4 a b d) :
  (a / (b + c + d)) * 100 = (1 / 43.166) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_a_l2386_238643


namespace NUMINAMATH_CALUDE_valid_pairs_l2386_238613

def is_valid_pair (a b : ℕ) : Prop :=
  (a^2 + b) % (b^2 - a) = 0 ∧ (b^2 + a) % (a^2 - b) = 0

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔ 
    ((a = 1 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 1) ∨ 
     (a = 2 ∧ b = 2) ∨ 
     (a = 2 ∧ b = 3) ∨ 
     (a = 3 ∧ b = 2) ∨ 
     (a = 3 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2386_238613


namespace NUMINAMATH_CALUDE_race_permutations_l2386_238668

theorem race_permutations (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_permutations_l2386_238668


namespace NUMINAMATH_CALUDE_square_less_than_triple_l2386_238644

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l2386_238644
