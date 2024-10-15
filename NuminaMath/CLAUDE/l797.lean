import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_line_slope_l797_79786

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define a line l
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Theorem statement
theorem ellipse_line_slope (x₁ y₁ x₂ y₂ k m : ℝ) :
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
  line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
  is_midpoint x₁ y₁ x₂ y₂ 1 1 →
  k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_l797_79786


namespace NUMINAMATH_CALUDE_intersection_M_N_l797_79784

def U := ℝ

def M : Set ℝ := {-1, 1, 2}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : N ∩ M = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l797_79784


namespace NUMINAMATH_CALUDE_two_complex_roots_iff_k_values_l797_79771

/-- The equation has exactly two complex roots if and only if k is 0, 2i, or -2i -/
theorem two_complex_roots_iff_k_values (k : ℂ) : 
  (∃! (r₁ r₂ : ℂ), ∀ (x : ℂ), x ≠ -3 ∧ x ≠ -4 → 
    (x / (x + 3) + x / (x + 4) = k * x ↔ x = 0 ∨ x = r₁ ∨ x = r₂)) ↔ 
  (k = 0 ∨ k = 2*I ∨ k = -2*I) :=
sorry

end NUMINAMATH_CALUDE_two_complex_roots_iff_k_values_l797_79771


namespace NUMINAMATH_CALUDE_math_basketball_count_l797_79774

/-- Represents the number of students in a school with various club and team memberships -/
structure SchoolMembership where
  total : ℕ
  science_club : ℕ
  math_club : ℕ
  football_team : ℕ
  basketball_team : ℕ
  science_football : ℕ

/-- Conditions for the school membership problem -/
def school_conditions (s : SchoolMembership) : Prop :=
  s.total = 60 ∧
  s.science_club + s.math_club = s.total ∧
  s.football_team + s.basketball_team = s.total ∧
  s.science_football = 20 ∧
  s.math_club = 36 ∧
  s.basketball_team = 22

/-- Theorem stating the number of students in both math club and basketball team -/
theorem math_basketball_count (s : SchoolMembership) 
  (h : school_conditions s) : 
  s.math_club + s.basketball_team - s.total = 18 := by
  sorry

#check math_basketball_count

end NUMINAMATH_CALUDE_math_basketball_count_l797_79774


namespace NUMINAMATH_CALUDE_joggers_speed_ratio_l797_79764

theorem joggers_speed_ratio (v₁ v₂ : ℝ) (h1 : v₁ > v₂) (h2 : (v₁ + v₂) * 2 = 8) (h3 : (v₁ - v₂) * 4 = 8) : v₁ / v₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_joggers_speed_ratio_l797_79764


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_triangle_is_equilateral_l797_79752

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfiesCondition (t : Triangle) : Prop :=
  t.b^2 + t.c^2 = t.a^2 + t.b * t.c

-- Define the law of cosines
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C

-- Define the equilateral triangle condition
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Theorem 1: Prove that angle A is 60 degrees
theorem angle_A_is_60_degrees (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : satisfiesCondition t) 
  (h3 : lawOfCosines t) : 
  t.A = Real.pi / 3 :=
sorry

-- Theorem 2: Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : isValidTriangle t)
  (h2 : satisfiesCondition t)
  (h3 : lawOfSines t)
  (h4 : Real.sin t.B * Real.sin t.C = Real.sin t.A ^ 2) :
  isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_triangle_is_equilateral_l797_79752


namespace NUMINAMATH_CALUDE_students_after_yoongi_l797_79755

/-- Given a group of students waiting for a bus, this theorem proves
    the number of students who came after a specific student. -/
theorem students_after_yoongi 
  (total : ℕ) 
  (before_jungkook : ℕ) 
  (h1 : total = 20) 
  (h2 : before_jungkook = 11) 
  (h3 : ∃ (before_yoongi : ℕ), before_yoongi + 1 = before_jungkook) : 
  ∃ (after_yoongi : ℕ), after_yoongi = total - (before_jungkook - 1) - 1 ∧ after_yoongi = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_after_yoongi_l797_79755


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l797_79757

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l797_79757


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l797_79787

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    prove that the third term is 3/4. -/
theorem third_term_of_geometric_series 
  (a : ℝ) -- First term of the series
  (h1 : 0 < (1 : ℝ) - (1/4 : ℝ)) -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - (1/4 : ℝ)) = 16) -- Sum formula for infinite geometric series
  : a * (1/4)^2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l797_79787


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_in_arithmetic_sequence_l797_79796

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The given sequence 3, 9, x, y, 27 -/
def givenSequence (x y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9
  | 2 => x
  | 3 => y
  | 4 => 27
  | _ => 0  -- For indices beyond 4, we return 0 (this part is not relevant to our problem)

theorem sum_of_x_and_y_in_arithmetic_sequence (x y : ℝ) 
    (h : isArithmeticSequence (givenSequence x y)) : x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_in_arithmetic_sequence_l797_79796


namespace NUMINAMATH_CALUDE_moving_circle_center_path_l797_79783

/-- A moving circle M with center (x, y) passes through (3, 2) and is tangent to y = 1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = (y - 1)^2

/-- The equation of the path traced by the center of the moving circle -/
def CenterPath (x y : ℝ) : Prop :=
  x^2 - 6*x + 2*y + 12 = 0

/-- Theorem: The equation of the path traced by the center of the moving circle
    is x^2 - 6x + 2y + 12 = 0 -/
theorem moving_circle_center_path :
  ∀ x y : ℝ, MovingCircle x y → CenterPath x y := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_center_path_l797_79783


namespace NUMINAMATH_CALUDE_peach_price_to_friends_peach_price_proof_l797_79710

/-- The price of peaches sold to friends, given the following conditions:
  * Lilia has 15 peaches
  * She sold 10 peaches to friends
  * She sold 4 peaches to relatives for $1.25 each
  * She kept 1 peach for herself
  * She earned $25 in total from selling 14 peaches
-/
theorem peach_price_to_friends : ℝ :=
  let total_peaches : ℕ := 15
  let peaches_to_friends : ℕ := 10
  let peaches_to_relatives : ℕ := 4
  let peaches_kept : ℕ := 1
  let price_to_relatives : ℝ := 1.25
  let total_earned : ℝ := 25
  let price_to_friends : ℝ := (total_earned - peaches_to_relatives * price_to_relatives) / peaches_to_friends
  2

theorem peach_price_proof (total_peaches : ℕ) (peaches_to_friends : ℕ) (peaches_to_relatives : ℕ) 
    (peaches_kept : ℕ) (price_to_relatives : ℝ) (total_earned : ℝ) :
    total_peaches = peaches_to_friends + peaches_to_relatives + peaches_kept →
    total_earned = peaches_to_friends * peach_price_to_friends + peaches_to_relatives * price_to_relatives →
    peach_price_to_friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_peach_price_to_friends_peach_price_proof_l797_79710


namespace NUMINAMATH_CALUDE_juan_number_problem_l797_79741

theorem juan_number_problem (n : ℚ) : 
  ((n + 3) * 3 - 5) / 3 = 10 → n = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_juan_number_problem_l797_79741


namespace NUMINAMATH_CALUDE_cindy_added_25_pens_l797_79792

/-- Calculates the number of pens Cindy added given the initial count, pens received, pens given away, and final count. -/
def pens_added_by_cindy (initial : ℕ) (received : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received - given_away)

/-- Theorem stating that Cindy added 25 pens given the specific conditions of the problem. -/
theorem cindy_added_25_pens :
  pens_added_by_cindy 5 20 10 40 = 25 := by
  sorry

#eval pens_added_by_cindy 5 20 10 40

end NUMINAMATH_CALUDE_cindy_added_25_pens_l797_79792


namespace NUMINAMATH_CALUDE_distance_probability_l797_79719

/-- Represents a city in our problem -/
inductive City : Type
| Bangkok : City
| CapeTown : City
| Honolulu : City
| London : City

/-- The distance between two cities in miles -/
def distance (c1 c2 : City) : ℕ :=
  match c1, c2 with
  | City.Bangkok, City.CapeTown => 6300
  | City.Bangkok, City.Honolulu => 6609
  | City.Bangkok, City.London => 5944
  | City.CapeTown, City.Honolulu => 11535
  | City.CapeTown, City.London => 5989
  | City.Honolulu, City.London => 7240
  | _, _ => 0  -- Same city or reverse order

/-- The total number of unique city pairs -/
def totalPairs : ℕ := 6

/-- The number of city pairs with distance less than 8000 miles -/
def pairsLessThan8000 : ℕ := 5

/-- The probability of selecting two cities with distance less than 8000 miles -/
def probability : ℚ := 5 / 6

theorem distance_probability :
  (probability : ℚ) = (pairsLessThan8000 : ℚ) / (totalPairs : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_distance_probability_l797_79719


namespace NUMINAMATH_CALUDE_reading_speed_ratio_l797_79772

/-- Given that Emery takes 20 days to read a book and the average number of days
    for Emery and Serena to read the book is 60, prove that the ratio of
    Emery's reading speed to Serena's reading speed is 5:1 -/
theorem reading_speed_ratio
  (emery_days : ℕ)
  (average_days : ℚ)
  (h_emery : emery_days = 20)
  (h_average : average_days = 60) :
  ∃ (emery_speed serena_speed : ℚ), 
    emery_speed / serena_speed = 5 / 1 :=
by sorry

end NUMINAMATH_CALUDE_reading_speed_ratio_l797_79772


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_3x_minus_y_equals_0_l797_79715

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x^3 - 1

theorem tangent_line_parallel_to_3x_minus_y_equals_0 :
  let P : ℝ × ℝ := (1, 0)
  f P.1 = P.2 ∧ f' P.1 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_3x_minus_y_equals_0_l797_79715


namespace NUMINAMATH_CALUDE_bee_puzzle_l797_79706

theorem bee_puzzle (B : ℕ) 
  (h1 : B > 0)
  (h2 : B % 5 = 0)
  (h3 : B % 3 = 0)
  (h4 : B = B / 5 + B / 3 + 3 * (B / 3 - B / 5) + 1) :
  B = 15 := by
sorry

end NUMINAMATH_CALUDE_bee_puzzle_l797_79706


namespace NUMINAMATH_CALUDE_equation_implies_conditions_l797_79737

theorem equation_implies_conditions (x y z w : ℝ) 
  (h : (2*x + y) / (y + z) = (z + w) / (w + 2*x)) :
  x = z/2 ∨ 2*x + y + z + w = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_implies_conditions_l797_79737


namespace NUMINAMATH_CALUDE_inverse_f_90_l797_79745

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_90 : f⁻¹ 90 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_90_l797_79745


namespace NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l797_79782

theorem sqrt_inequality_equivalence :
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 6 + Real.sqrt 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l797_79782


namespace NUMINAMATH_CALUDE_bus_arrival_time_difference_l797_79778

/-- Proves that a person walking to a bus stand will arrive 10 minutes early when doubling their speed -/
theorem bus_arrival_time_difference (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (miss_time : ℝ) : 
  distance = 2.2 →
  speed1 = 3 →
  speed2 = 6 →
  miss_time = 12 →
  (distance / speed2 * 60) = ((distance / speed1 * 60) - miss_time) - 10 :=
by sorry

end NUMINAMATH_CALUDE_bus_arrival_time_difference_l797_79778


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l797_79740

theorem solution_satisfies_equations :
  let a : ℚ := 4/7
  let b : ℚ := 19/7
  let c : ℚ := 29/19
  let d : ℚ := -6/19
  (8*a^2 - 3*b^2 + 5*c^2 + 16*d^2 - 10*a*b + 42*c*d + 18*a + 22*b - 2*c - 54*d = 42) ∧
  (15*a^2 - 3*b^2 + 21*c^2 - 5*d^2 + 4*a*b + 32*c*d - 28*a + 14*b - 54*c - 52*d = -22) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l797_79740


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l797_79762

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l797_79762


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l797_79763

theorem perfect_square_quadratic (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 18*x + k = (a*x + b)^2) ↔ k = 81 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l797_79763


namespace NUMINAMATH_CALUDE_alicia_satisfaction_l797_79743

/-- Represents the satisfaction equation for Alicia's activities --/
def satisfaction (reading : ℝ) (painting : ℝ) : ℝ := reading * painting

/-- Represents the constraint that t should be positive and less than 4 --/
def valid_t (t : ℝ) : Prop := 0 < t ∧ t < 4

theorem alicia_satisfaction (t : ℝ) : 
  valid_t t →
  satisfaction (12 - t) t = satisfaction (2*t + 2) (4 - t) →
  t = 2 :=
by sorry

end NUMINAMATH_CALUDE_alicia_satisfaction_l797_79743


namespace NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l797_79768

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (purple : ℕ)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 27 socks are needed to guarantee 12 pairs in the given drawer -/
theorem min_socks_for_twelve_pairs :
  let drawer : SockDrawer := { red := 90, green := 70, blue := 50, purple := 30 }
  minSocksForPairs drawer 12 = 27 := by sorry

end NUMINAMATH_CALUDE_min_socks_for_twelve_pairs_l797_79768


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l797_79704

theorem probability_nine_heads_in_twelve_flips : 
  (Nat.choose 12 9 : ℚ) / (2^12 : ℚ) = 220 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l797_79704


namespace NUMINAMATH_CALUDE_vector_subtraction_l797_79734

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := ![2, -x]
def b : Fin 2 → ℝ := ![-1, 3]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem vector_subtraction (x : ℝ) : 
  dot_product (a x) b = 4 → 
  (λ i : Fin 2 => (a x) i - 2 * (b i)) = ![4, -4] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l797_79734


namespace NUMINAMATH_CALUDE_triangle_side_length_l797_79717

/-- Represents a triangle with side lengths x, y, z and angles X, Y, Z --/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  X : ℝ
  Y : ℝ
  Z : ℝ

/-- The theorem stating the properties of the specific triangle and its side length y --/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.Z = 4 * t.X) 
  (h2 : t.x = 36) 
  (h3 : t.z = 72) : 
  t.y = 72 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l797_79717


namespace NUMINAMATH_CALUDE_knight_count_l797_79756

def is_correct_arrangement (knights liars : ℕ) : Prop :=
  knights + liars = 2019 ∧
  knights > 2 * liars ∧
  knights ≤ 2 * liars + 1

theorem knight_count : ∃ (knights liars : ℕ), 
  is_correct_arrangement knights liars ∧ knights = 1346 := by
  sorry

end NUMINAMATH_CALUDE_knight_count_l797_79756


namespace NUMINAMATH_CALUDE_roots_of_composite_quadratic_l797_79702

/-- A quadratic function with real coefficients -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given point -/
def evaluate (f : QuadraticFunction) (x : ℂ) : ℂ :=
  f.a * x^2 + f.b * x + f.c

/-- Predicate stating that a complex number is purely imaginary -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Predicate stating that all roots of the equation f(x) = 0 are purely imaginary -/
def hasPurelyImaginaryRoots (f : QuadraticFunction) : Prop :=
  ∀ x : ℂ, evaluate f x = 0 → isPurelyImaginary x

/-- Theorem stating the nature of roots for f(f(x)) = 0 -/
theorem roots_of_composite_quadratic
  (f : QuadraticFunction)
  (h : hasPurelyImaginaryRoots f) :
  ∀ x : ℂ, evaluate f (evaluate f x) = 0 →
    (¬ x.im = 0) ∧ ¬ isPurelyImaginary x :=
sorry

end NUMINAMATH_CALUDE_roots_of_composite_quadratic_l797_79702


namespace NUMINAMATH_CALUDE_factorial_division_l797_79746

theorem factorial_division : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l797_79746


namespace NUMINAMATH_CALUDE_milkshake_production_theorem_l797_79794

/-- Represents the milkshake production scenario -/
structure MilkshakeProduction where
  augustus_rate : ℕ
  luna_rate : ℕ
  neptune_rate : ℕ
  total_hours : ℕ
  neptune_start : ℕ
  break_interval : ℕ
  extra_break : ℕ
  break_consumption : ℕ

/-- Calculates the total number of milkshakes produced -/
def total_milkshakes (prod : MilkshakeProduction) : ℕ :=
  sorry

/-- The main theorem stating that given the conditions, 93 milkshakes are produced -/
theorem milkshake_production_theorem (prod : MilkshakeProduction)
  (h1 : prod.augustus_rate = 3)
  (h2 : prod.luna_rate = 7)
  (h3 : prod.neptune_rate = 5)
  (h4 : prod.total_hours = 12)
  (h5 : prod.neptune_start = 3)
  (h6 : prod.break_interval = 3)
  (h7 : prod.extra_break = 7)
  (h8 : prod.break_consumption = 18) :
  total_milkshakes prod = 93 :=
sorry

end NUMINAMATH_CALUDE_milkshake_production_theorem_l797_79794


namespace NUMINAMATH_CALUDE_ann_has_36_blocks_l797_79722

/-- The number of blocks Ann has at the end, given her initial blocks, 
    blocks found, and blocks lost. -/
def anns_final_blocks (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost

/-- Theorem stating that Ann ends up with 36 blocks -/
theorem ann_has_36_blocks : anns_final_blocks 9 44 17 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_has_36_blocks_l797_79722


namespace NUMINAMATH_CALUDE_animal_ratio_proof_l797_79761

/-- Given ratios between animals, prove the final ratio of all animals -/
theorem animal_ratio_proof 
  (chicken_pig_ratio : ℚ × ℚ)
  (sheep_horse_ratio : ℚ × ℚ)
  (pig_horse_ratio : ℚ × ℚ)
  (h1 : chicken_pig_ratio = (26, 5))
  (h2 : sheep_horse_ratio = (25, 9))
  (h3 : pig_horse_ratio = (10, 3)) :
  ∃ (k : ℚ), k > 0 ∧ 
    k * 156 = chicken_pig_ratio.1 * pig_horse_ratio.2 ∧
    k * 30 = chicken_pig_ratio.2 * pig_horse_ratio.2 ∧
    k * 9 = pig_horse_ratio.2 ∧
    k * 25 = sheep_horse_ratio.1 * pig_horse_ratio.2 / sheep_horse_ratio.2 :=
by
  sorry

end NUMINAMATH_CALUDE_animal_ratio_proof_l797_79761


namespace NUMINAMATH_CALUDE_cycle_selling_price_l797_79705

/-- Calculates the selling price of a cycle given its original price and loss percentage. -/
theorem cycle_selling_price (original_price loss_percentage : ℝ) :
  original_price = 2300 →
  loss_percentage = 30 →
  original_price * (1 - loss_percentage / 100) = 1610 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l797_79705


namespace NUMINAMATH_CALUDE_division_remainder_l797_79791

def largest_three_digit : Nat := 975
def smallest_two_digit : Nat := 23

theorem division_remainder :
  largest_three_digit % smallest_two_digit = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l797_79791


namespace NUMINAMATH_CALUDE_equidistant_line_proof_l797_79769

-- Define the two given lines
def line1 (x y : ℝ) : ℝ := 3 * x + 2 * y - 6
def line2 (x y : ℝ) : ℝ := 6 * x + 4 * y - 3

-- Define the proposed equidistant line
def equidistant_line (x y : ℝ) : ℝ := 12 * x + 8 * y - 15

-- Theorem statement
theorem equidistant_line_proof :
  ∀ (x y : ℝ), |equidistant_line x y| = |line1 x y| ∧ |equidistant_line x y| = |line2 x y| :=
sorry

end NUMINAMATH_CALUDE_equidistant_line_proof_l797_79769


namespace NUMINAMATH_CALUDE_physics_marks_l797_79711

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 70)
  (physics_math_avg : (P + M) / 2 = 90)
  (physics_chem_avg : (P + C) / 2 = 70) :
  P = 110 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l797_79711


namespace NUMINAMATH_CALUDE_total_crayons_l797_79758

theorem total_crayons (billy_crayons jane_crayons : ℝ) 
  (h1 : billy_crayons = 62.0) 
  (h2 : jane_crayons = 52.0) : 
  billy_crayons + jane_crayons = 114.0 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l797_79758


namespace NUMINAMATH_CALUDE_pinterest_group_pins_l797_79724

theorem pinterest_group_pins 
  (num_members : ℕ) 
  (initial_pins : ℕ) 
  (daily_contribution : ℕ) 
  (weekly_deletion : ℕ) 
  (days_in_month : ℕ) 
  (h1 : num_members = 20)
  (h2 : initial_pins = 1000)
  (h3 : daily_contribution = 10)
  (h4 : weekly_deletion = 5)
  (h5 : days_in_month = 30) :
  let total_new_pins := num_members * daily_contribution * days_in_month
  let total_deleted_pins := num_members * weekly_deletion * (days_in_month / 7)
  initial_pins + total_new_pins - total_deleted_pins = 6600 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_pins_l797_79724


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_equality_l797_79785

/-- A regular dodecagon -/
structure RegularDodecagon where
  /-- Side length of the dodecagon -/
  a : ℝ
  /-- Length of shortest diagonal spanning three sides -/
  b : ℝ
  /-- Length of longest diagonal spanning six sides -/
  d : ℝ
  /-- Positive side length -/
  a_pos : 0 < a

/-- In a regular dodecagon, the length of the shortest diagonal spanning three sides
    is equal to the length of the longest diagonal spanning six sides -/
theorem dodecagon_diagonal_equality (poly : RegularDodecagon) : poly.b = poly.d := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonal_equality_l797_79785


namespace NUMINAMATH_CALUDE_both_runners_in_picture_probability_zero_l797_79732

/-- Represents a runner on the circular track -/
structure Runner where
  direction : Bool  -- true for counterclockwise, false for clockwise
  lap_time : ℕ      -- time to complete one lap in seconds

/-- Represents the photographer's picture -/
structure Picture where
  coverage : ℝ      -- fraction of the track covered by the picture
  center : ℝ        -- position of the center of the picture on the track (0 ≤ center < 1)

/-- Calculates the probability of both runners being in the picture -/
def probability_both_in_picture (lydia : Runner) (lucas : Runner) (pic : Picture) : ℝ :=
  sorry

/-- Theorem stating that the probability of both runners being in the picture is 0 -/
theorem both_runners_in_picture_probability_zero 
  (lydia : Runner) 
  (lucas : Runner) 
  (pic : Picture) 
  (h1 : lydia.direction = true) 
  (h2 : lydia.lap_time = 120) 
  (h3 : lucas.direction = false) 
  (h4 : lucas.lap_time = 100) 
  (h5 : pic.coverage = 1/3) 
  (h6 : pic.center = 0) : 
  probability_both_in_picture lydia lucas pic = 0 :=
sorry

end NUMINAMATH_CALUDE_both_runners_in_picture_probability_zero_l797_79732


namespace NUMINAMATH_CALUDE_divisibility_by_43_l797_79788

theorem divisibility_by_43 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  43 ∣ (7^p - 6^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_43_l797_79788


namespace NUMINAMATH_CALUDE_total_earnings_l797_79739

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem total_earnings :
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l797_79739


namespace NUMINAMATH_CALUDE_expression_value_l797_79767

theorem expression_value :
  let x : ℝ := 1
  let y : ℝ := 1
  let z : ℝ := 3
  let p : ℝ := 2
  let q : ℝ := 4
  let r : ℝ := 2
  let s : ℝ := 3
  let t : ℝ := 3
  (p + x)^2 * y * z - q * r * (x * y * z)^2 + s^t = -18 := by sorry

end NUMINAMATH_CALUDE_expression_value_l797_79767


namespace NUMINAMATH_CALUDE_max_principals_l797_79760

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the minimum term length for a principal -/
def minTerm : ℕ := 3

/-- Represents the maximum term length for a principal -/
def maxTerm : ℕ := 5

/-- Represents a valid principal term length -/
def ValidTerm (t : ℕ) : Prop := minTerm ≤ t ∧ t ≤ maxTerm

/-- 
Theorem: The maximum number of principals during a continuous 10-year period is 3,
given that each principal's term can be between 3 and 5 years.
-/
theorem max_principals :
  ∃ (n : ℕ) (terms : List ℕ),
    (∀ t ∈ terms, ValidTerm t) ∧ 
    (terms.sum ≥ period) ∧
    (terms.length = n) ∧
    (∀ m : ℕ, m > n → 
      ¬∃ (terms' : List ℕ), 
        (∀ t ∈ terms', ValidTerm t) ∧ 
        (terms'.sum ≥ period) ∧
        (terms'.length = m)) ∧
    n = 3 :=
  sorry

end NUMINAMATH_CALUDE_max_principals_l797_79760


namespace NUMINAMATH_CALUDE_spinner_probability_divisible_by_3_l797_79750

/-- A spinner with 8 equal sections numbered from 1 to 8 -/
def Spinner := Finset (Fin 8)

/-- The set of numbers on the spinner that are divisible by 3 -/
def DivisibleBy3 (s : Spinner) : Finset (Fin 8) :=
  s.filter (fun n => n % 3 = 0)

/-- The probability of an event on the spinner -/
def Probability (event : Finset (Fin 8)) (s : Spinner) : ℚ :=
  event.card / s.card

theorem spinner_probability_divisible_by_3 (s : Spinner) :
  Probability (DivisibleBy3 s) s = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_spinner_probability_divisible_by_3_l797_79750


namespace NUMINAMATH_CALUDE_bus_driver_compensation_theorem_l797_79795

/-- Represents the compensation structure and work hours of a bus driver -/
structure BusDriverCompensation where
  regular_rate : ℝ
  overtime_rate : ℝ
  total_compensation : ℝ
  total_hours : ℝ
  regular_hours_limit : ℝ

/-- Calculates the overtime rate based on the regular rate -/
def overtime_rate (regular_rate : ℝ) : ℝ :=
  regular_rate * 1.75

/-- Theorem stating the conditions and the result to be proved -/
theorem bus_driver_compensation_theorem (driver : BusDriverCompensation) :
  driver.regular_rate = 16 ∧
  driver.overtime_rate = overtime_rate driver.regular_rate ∧
  driver.total_compensation = 920 ∧
  driver.total_hours = 50 →
  driver.regular_hours_limit = 40 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_theorem_l797_79795


namespace NUMINAMATH_CALUDE_compute_expression_l797_79720

theorem compute_expression : 9 * (2 / 3) ^ 4 = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l797_79720


namespace NUMINAMATH_CALUDE_twentieth_number_in_sequence_l797_79708

theorem twentieth_number_in_sequence : ∃ (n : ℕ), 
  (n % 8 = 5) ∧ 
  (n % 3 = 2) ∧ 
  (∃ (k : ℕ), k = 19 ∧ n = 5 + 24 * k) ∧
  n = 461 := by
sorry

end NUMINAMATH_CALUDE_twentieth_number_in_sequence_l797_79708


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l797_79728

/-- Given a right triangle with one leg of length 5 and an altitude to the hypotenuse of length 4,
    the length of the other leg is 20/3. -/
theorem right_triangle_leg_length (a b c h : ℝ) : 
  a = 5 →                        -- One leg has length 5
  h = 4 →                        -- Altitude to hypotenuse has length 4
  a^2 + b^2 = c^2 →              -- Pythagorean theorem
  (1/2) * a * b = (1/2) * c * h → -- Area equality
  b = 20/3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l797_79728


namespace NUMINAMATH_CALUDE_total_snowfall_l797_79733

theorem total_snowfall (morning_snow afternoon_snow : ℝ) 
  (h1 : morning_snow = 0.125)
  (h2 : afternoon_snow = 0.5) :
  morning_snow + afternoon_snow = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_total_snowfall_l797_79733


namespace NUMINAMATH_CALUDE_history_books_count_l797_79751

theorem history_books_count (total : ℕ) (reading : ℕ) (math : ℕ) (science : ℕ) (history : ℕ) : 
  total = 10 →
  reading = 2 * total / 5 →
  math = 3 * total / 10 →
  science = math - 1 →
  history = total - (reading + math + science) →
  history = 1 := by
sorry

end NUMINAMATH_CALUDE_history_books_count_l797_79751


namespace NUMINAMATH_CALUDE_total_apples_l797_79780

theorem total_apples (kayla_apples kylie_apples : ℕ) : 
  kayla_apples = 40 → 
  kayla_apples = kylie_apples / 4 → 
  kayla_apples + kylie_apples = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_total_apples_l797_79780


namespace NUMINAMATH_CALUDE_function_increasing_iff_m_in_range_l797_79759

/-- The function f(x) = (1/3)x³ - mx² - 3m²x + 1 is increasing on (1, 2) if and only if m is in [-1, 1/3] -/
theorem function_increasing_iff_m_in_range (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, StrictMono (fun x => (1/3) * x^3 - m * x^2 - 3 * m^2 * x + 1)) ↔
  m ∈ Set.Icc (-1) (1/3) :=
sorry

end NUMINAMATH_CALUDE_function_increasing_iff_m_in_range_l797_79759


namespace NUMINAMATH_CALUDE_problem_statement_l797_79775

theorem problem_statement (a b c : ℝ) 
  (h1 : a + 2*b + 3*c = 12)
  (h2 : a^2 + b^2 + c^2 = a*b + a*c + b*c) :
  a + b^2 + c^3 = 14 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l797_79775


namespace NUMINAMATH_CALUDE_expand_quadratic_l797_79723

theorem expand_quadratic (a : ℝ) : a * (a - 3) = a^2 - 3*a := by sorry

end NUMINAMATH_CALUDE_expand_quadratic_l797_79723


namespace NUMINAMATH_CALUDE_third_member_reels_six_l797_79747

/-- Represents a fishing competition with three team members -/
structure FishingCompetition where
  days : ℕ
  fish_per_day_1 : ℕ
  fish_per_day_2 : ℕ
  total_fish : ℕ

/-- Calculates the number of fish the third member reels per day -/
def third_member_fish_per_day (comp : FishingCompetition) : ℕ :=
  (comp.total_fish - (comp.fish_per_day_1 + comp.fish_per_day_2) * comp.days) / comp.days

/-- Theorem stating that in the given conditions, the third member reels 6 fish per day -/
theorem third_member_reels_six (comp : FishingCompetition) 
  (h1 : comp.days = 5)
  (h2 : comp.fish_per_day_1 = 4)
  (h3 : comp.fish_per_day_2 = 8)
  (h4 : comp.total_fish = 90) : 
  third_member_fish_per_day comp = 6 := by
  sorry

#eval third_member_fish_per_day ⟨5, 4, 8, 90⟩

end NUMINAMATH_CALUDE_third_member_reels_six_l797_79747


namespace NUMINAMATH_CALUDE_fourth_power_sum_l797_79770

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a^2 + b^2 + c^2 = 5) 
  (h3 : a^3 + b^3 + c^3 = 8) : 
  a^4 + b^4 + c^4 = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l797_79770


namespace NUMINAMATH_CALUDE_square_sum_equality_l797_79712

theorem square_sum_equality : 108 * 108 + 92 * 92 = 20128 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l797_79712


namespace NUMINAMATH_CALUDE_intersection_line_equation_l797_79701

/-- Given two lines that intersect at (2, 3), prove that the line passing through
    the points defined by their coefficients has the equation 2x + 3y - 1 = 0 -/
theorem intersection_line_equation (A₁ B₁ A₂ B₂ : ℝ) :
  (A₁ * 2 + B₁ * 3 = 1) →
  (A₂ * 2 + B₂ * 3 = 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (A₁ - A₂) * 2 + (B₁ - B₂) * 3 = k * (2 * (A₁ - A₂) + 3 * (B₁ - B₂) - 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l797_79701


namespace NUMINAMATH_CALUDE_fraction_ordering_l797_79744

theorem fraction_ordering : 19/15 < 17/13 ∧ 17/13 < 15/11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l797_79744


namespace NUMINAMATH_CALUDE_man_swimming_speed_l797_79781

/-- The speed of a man in still water, given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_distance : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 48) 
  (h2 : downstream_time = 3) 
  (h3 : upstream_distance = 34) 
  (h4 : upstream_time = 4) : 
  ∃ (man_speed stream_speed : ℝ), 
    man_speed = 12.25 ∧ 
    downstream_distance = (man_speed + stream_speed) * downstream_time ∧
    upstream_distance = (man_speed - stream_speed) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_man_swimming_speed_l797_79781


namespace NUMINAMATH_CALUDE_abs_sum_inequality_iff_l797_79736

theorem abs_sum_inequality_iff (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| > a) ↔ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_iff_l797_79736


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l797_79738

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 4
def q (x y : ℝ) : Prop := x + y ≠ 6

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l797_79738


namespace NUMINAMATH_CALUDE_john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l797_79727

/-- Calculates how many meters ahead John finishes compared to Steve in a race --/
theorem john_finish_distance_ahead_of_steve 
  (initial_distance_behind : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_push_time : ℝ) : ℝ :=
  let john_distance := john_speed * final_push_time
  let steve_distance := steve_speed * final_push_time
  let steve_effective_distance := steve_distance + initial_distance_behind
  john_distance - steve_effective_distance

/-- Proves that John finishes 2 meters ahead of Steve given the race conditions --/
theorem john_finishes_two_meters_ahead : 
  john_finish_distance_ahead_of_steve 15 4.2 3.8 42.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_finish_distance_ahead_of_steve_john_finishes_two_meters_ahead_l797_79727


namespace NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l797_79725

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- Represents a division of a polygon into triangular regions using diagonals -/
structure PolygonDivision (n : ℕ) where
  polygon : RegularPolygon n
  num_triangles : ℕ
  num_diagonals : ℕ
  diagonals_dont_intersect : Bool

/-- Represents the number of isosceles triangles in a polygon division -/
def num_isosceles_triangles (d : PolygonDivision n) : ℕ :=
  sorry

/-- Theorem: The maximum number of isosceles triangles in a specific polygon division -/
theorem max_isosceles_triangles_2017gon :
  ∀ (d : PolygonDivision 2017),
    d.num_triangles = 2015 →
    d.num_diagonals = 2014 →
    d.diagonals_dont_intersect = true →
    num_isosceles_triangles d ≤ 2010 :=
  sorry

end NUMINAMATH_CALUDE_max_isosceles_triangles_2017gon_l797_79725


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_120_l797_79776

/-- Given a sequence of consecutive natural numbers with sum 120, 
    the largest number in the sequence is 26 -/
theorem largest_consecutive_sum_120 (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : (n : ℝ) * (2 * a + n - 1) / 2 = 120) :
  a + n - 1 ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_120_l797_79776


namespace NUMINAMATH_CALUDE_min_cost_pool_l797_79713

/-- Represents the dimensions and cost parameters of a rectangular pool -/
structure PoolParams where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of the pool given its length and width -/
def totalCost (p : PoolParams) (length width : ℝ) : ℝ :=
  p.bottomCost * length * width + p.wallCost * (2 * length * p.depth + 2 * width * p.depth)

/-- Theorem stating the minimum cost and dimensions of the pool -/
theorem min_cost_pool (p : PoolParams) 
    (hv : p.volume = 16)
    (hd : p.depth = 4)
    (hb : p.bottomCost = 110)
    (hw : p.wallCost = 90) :
    ∃ (length width : ℝ),
      length * width * p.depth = p.volume ∧
      length = 2 ∧
      width = 2 ∧
      totalCost p length width = 1880 ∧
      ∀ (l w : ℝ), l * w * p.depth = p.volume → totalCost p l w ≥ totalCost p length width :=
  sorry

end NUMINAMATH_CALUDE_min_cost_pool_l797_79713


namespace NUMINAMATH_CALUDE_exists_x_fx_equals_four_l797_79748

open Real

theorem exists_x_fx_equals_four :
  ∃ x₀ ∈ Set.Ioo 0 (3 * π), 3 + cos (2 * x₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_fx_equals_four_l797_79748


namespace NUMINAMATH_CALUDE_intersection_with_ratio_l797_79718

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given parallel lines and points, there exists a line through B intersecting the parallel lines at X and Y with the given ratio -/
theorem intersection_with_ratio 
  (a c : Line) 
  (A B C : Point) 
  (m n : ℝ) 
  (h_parallel : Line.parallel a c)
  (h_A_on_a : A.on_line a)
  (h_C_on_c : C.on_line c)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0) :
  ∃ (X Y : Point) (l : Line),
    X.on_line a ∧
    Y.on_line c ∧
    B.on_line l ∧
    X.on_line l ∧
    Y.on_line l ∧
    ∃ (k : ℝ), k > 0 ∧ 
      (X.x - A.x)^2 + (X.y - A.y)^2 = k * m^2 ∧
      (Y.x - C.x)^2 + (Y.y - C.y)^2 = k * n^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_with_ratio_l797_79718


namespace NUMINAMATH_CALUDE_quartic_integer_roots_l797_79714

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure QuarticPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  d_nonzero : d ≠ 0

/-- The number of integer roots of a quartic polynomial, counting multiplicities -/
def num_integer_roots (p : QuarticPolynomial) : ℕ :=
  sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem quartic_integer_roots (p : QuarticPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_quartic_integer_roots_l797_79714


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l797_79749

/-- Given that the set {1, a, b/a} equals the set {0, a², a+b}, prove that a²⁰¹³ + b²⁰¹² = -1 -/
theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2013 + b^2012 = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l797_79749


namespace NUMINAMATH_CALUDE_variance_of_data_l797_79773

/-- Given a list of 5 real numbers with an average of 5 and an average of squares of 33,
    prove that the variance of the list is 8. -/
theorem variance_of_data (x : List ℝ) (hx : x.length = 5)
  (h_avg : x.sum / 5 = 5)
  (h_avg_sq : (x.map (λ xi => xi^2)).sum / 5 = 33) :
  (x.map (λ xi => (xi - 5)^2)).sum / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_data_l797_79773


namespace NUMINAMATH_CALUDE_three_tangents_condition_l797_79726

/-- The curve function f(x) = x³ + 3x² + ax + a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a*x + a - 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*x + a

/-- The tangent line equation passing through (0, 2) -/
def tangent_line (a : ℝ) (x₀ : ℝ) (x : ℝ) : ℝ :=
  f_deriv a x₀ * (x - x₀) + f a x₀

/-- The condition for a point x₀ to be on a tangent line passing through (0, 2) -/
def tangent_condition (a : ℝ) (x₀ : ℝ) : Prop :=
  tangent_line a x₀ 0 = 2

/-- The main theorem stating the condition for exactly three tangent lines -/
theorem three_tangents_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    tangent_condition a x₁ ∧ tangent_condition a x₂ ∧ tangent_condition a x₃ ∧
    (∀ x : ℝ, tangent_condition a x → x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  4 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_three_tangents_condition_l797_79726


namespace NUMINAMATH_CALUDE_art_collection_remaining_l797_79721

/-- Calculates the remaining number of art pieces after a donation --/
def remaining_art_pieces (initial : ℕ) (donated : ℕ) : ℕ :=
  initial - donated

/-- Theorem: Given 70 initial pieces and 46 donated pieces, 24 pieces remain --/
theorem art_collection_remaining :
  remaining_art_pieces 70 46 = 24 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_remaining_l797_79721


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l797_79729

theorem smallest_integer_satisfying_conditions : ∃ x : ℤ, 
  (∀ y : ℤ, (|3*y - 4| ≤ 25 ∧ 3 ∣ y) → x ≤ y) ∧ 
  |3*x - 4| ≤ 25 ∧ 
  3 ∣ x ∧
  x = -6 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l797_79729


namespace NUMINAMATH_CALUDE_rectangular_triangular_field_equal_area_l797_79766

/-- Proves that a rectangular field with width 4 m and length 6.3 m has the same area
    as a triangular field with base 7.2 m and height 7 m. -/
theorem rectangular_triangular_field_equal_area :
  let triangle_base : ℝ := 7.2
  let triangle_height : ℝ := 7
  let triangle_area : ℝ := (triangle_base * triangle_height) / 2
  let rectangle_width : ℝ := 4
  let rectangle_length : ℝ := 6.3
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  triangle_area = rectangle_area := by sorry

end NUMINAMATH_CALUDE_rectangular_triangular_field_equal_area_l797_79766


namespace NUMINAMATH_CALUDE_percentage_calculation_l797_79779

theorem percentage_calculation (P : ℝ) : 
  P * 5600 = 126 → 
  (0.3 * 0.5 * 5600 : ℝ) = 840 → 
  P = 0.0225 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l797_79779


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l797_79707

/-- Given a glucose solution where 45 cubic centimeters contain 6.75 grams of glucose,
    prove that the volume containing 15 grams of glucose is 100 cubic centimeters. -/
theorem glucose_solution_volume (volume : ℝ) (glucose_mass : ℝ) :
  (45 : ℝ) / volume = 6.75 / glucose_mass →
  glucose_mass = 15 →
  volume = 100 := by
  sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l797_79707


namespace NUMINAMATH_CALUDE_girls_in_club_l797_79797

theorem girls_in_club (total : Nat) (girls : Nat) (boys : Nat) : 
  total = 36 →
  girls + boys = total →
  (∀ (group : Nat), group = 33 → girls > group / 2) →
  (∃ (group : Nat), group = 31 ∧ boys ≥ group / 2) →
  girls = 20 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_club_l797_79797


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l797_79742

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l797_79742


namespace NUMINAMATH_CALUDE_twelve_chairs_subsets_l797_79765

/-- The number of chairs in the circle -/
def n : ℕ := 12

/-- A function that calculates the number of subsets with at least four adjacent chairs -/
def subsetsWithAdjacentChairs (n : ℕ) : ℕ := sorry

/-- Theorem stating that for 12 chairs, the number of subsets with at least four adjacent chairs is 1776 -/
theorem twelve_chairs_subsets : subsetsWithAdjacentChairs n = 1776 := by sorry

end NUMINAMATH_CALUDE_twelve_chairs_subsets_l797_79765


namespace NUMINAMATH_CALUDE_algorithm_output_l797_79789

def sum_odd_numbers (n : Nat) : Nat :=
  List.sum (List.range n |>.filter (λ x => x % 2 = 1))

theorem algorithm_output : 1 + sum_odd_numbers 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l797_79789


namespace NUMINAMATH_CALUDE_square_grid_divisible_four_parts_l797_79798

/-- A square grid of cells that can be divided into four equal parts -/
structure SquareGrid where
  n : ℕ
  n_even : Even n
  n_ge_2 : n ≥ 2

/-- The number of cells in each part when the grid is divided into four equal parts -/
def cells_per_part (grid : SquareGrid) : ℕ := grid.n * grid.n / 4

/-- Theorem stating that a square grid can be divided into four equal parts -/
theorem square_grid_divisible_four_parts (grid : SquareGrid) :
  ∃ (part_size : ℕ), part_size = cells_per_part grid ∧ 
  grid.n * grid.n = 4 * part_size :=
sorry

end NUMINAMATH_CALUDE_square_grid_divisible_four_parts_l797_79798


namespace NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l797_79700

theorem smallest_y_with_given_remainders : ∃ y : ℕ, 
  y > 0 ∧ 
  y % 3 = 2 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧ 
  ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_with_given_remainders_l797_79700


namespace NUMINAMATH_CALUDE_parabola_right_angle_l797_79703

theorem parabola_right_angle (a : ℝ) : 
  let f (x : ℝ) := -(x + 3) * (2 * x + a)
  let x₁ := -3
  let x₂ := -a / 2
  let y_c := f 0
  let A := (x₁, 0)
  let B := (x₂, 0)
  let C := (0, y_c)
  f x₁ = 0 ∧ f x₂ = 0 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  a = -1/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_right_angle_l797_79703


namespace NUMINAMATH_CALUDE_solution_set_l797_79709

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0

theorem solution_set : 
  {x : ℝ | equation x} = {-9, -3, 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_l797_79709


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l797_79735

theorem rectangular_solid_volume (a b c : ℝ) 
  (side_area : a * b = 20)
  (front_area : b * c = 15)
  (bottom_area : a * c = 12)
  (dimension_relation : a = 2 * b) : 
  a * b * c = 12 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l797_79735


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l797_79790

theorem sqrt_fourth_power_equals_256 (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l797_79790


namespace NUMINAMATH_CALUDE_square_has_perpendicular_diagonals_but_parallelogram_not_l797_79753

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a parallelogram
structure Parallelogram :=
  (base : ℝ)
  (height : ℝ)
  (base_positive : base > 0)
  (height_positive : height > 0)

-- Define the property of perpendicular diagonals
def has_perpendicular_diagonals (S : Type) : Prop :=
  ∀ s : S, ∃ d₁ d₂ : ℝ × ℝ, d₁.1 * d₂.1 + d₁.2 * d₂.2 = 0

-- Theorem statement
theorem square_has_perpendicular_diagonals_but_parallelogram_not :
  (has_perpendicular_diagonals Square) ∧ ¬(has_perpendicular_diagonals Parallelogram) :=
sorry

end NUMINAMATH_CALUDE_square_has_perpendicular_diagonals_but_parallelogram_not_l797_79753


namespace NUMINAMATH_CALUDE_harry_seed_purchase_cost_l797_79730

/-- Given the prices of seed packets and the quantities Harry wants to buy, 
    prove that the total cost is $18.00 -/
theorem harry_seed_purchase_cost : 
  let pumpkin_price : ℚ := 25/10
  let tomato_price : ℚ := 15/10
  let chili_price : ℚ := 9/10
  let pumpkin_quantity : ℕ := 3
  let tomato_quantity : ℕ := 4
  let chili_quantity : ℕ := 5
  (pumpkin_price * pumpkin_quantity + 
   tomato_price * tomato_quantity + 
   chili_price * chili_quantity) = 18
:= by sorry

end NUMINAMATH_CALUDE_harry_seed_purchase_cost_l797_79730


namespace NUMINAMATH_CALUDE_inequality_always_holds_l797_79777

theorem inequality_always_holds (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l797_79777


namespace NUMINAMATH_CALUDE_jerrys_age_l797_79793

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 14 → 
  mickey_age = 3 * jerry_age - 4 → 
  jerry_age = 6 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l797_79793


namespace NUMINAMATH_CALUDE_expression_equals_eight_l797_79754

theorem expression_equals_eight : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l797_79754


namespace NUMINAMATH_CALUDE_system_solution_l797_79716

theorem system_solution (x y : ℝ) 
  (eq1 : 2 * x + y = 4) 
  (eq2 : x + 2 * y = 5) : 
  (x - y = -1 ∧ x + y = 3) ∧ 
  (1/3 * x^2 - 1/3 * y^2) * (x^2 - 2*x*y + y^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l797_79716


namespace NUMINAMATH_CALUDE_vector_collinearity_l797_79731

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • b) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l797_79731


namespace NUMINAMATH_CALUDE_lee_cookies_l797_79799

/-- Given an initial ratio of flour to cookies and a new amount of flour, 
    calculate the number of cookies that can be made. -/
def cookies_from_flour (initial_flour : ℚ) (initial_cookies : ℚ) (new_flour : ℚ) : ℚ :=
  (initial_cookies / initial_flour) * new_flour

/-- Theorem stating that with the given initial ratio and remaining flour, 
    Lee can make 36 cookies. -/
theorem lee_cookies : 
  let initial_flour : ℚ := 2
  let initial_cookies : ℚ := 18
  let initial_available : ℚ := 5
  let spilled : ℚ := 1
  let remaining_flour : ℚ := initial_available - spilled
  cookies_from_flour initial_flour initial_cookies remaining_flour = 36 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l797_79799
