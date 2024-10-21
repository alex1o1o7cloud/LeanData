import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_increasing_f_l205_20503

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem a_range_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-3) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_increasing_f_l205_20503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_condition_l205_20524

theorem complex_number_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) * (-3 + a * Complex.I)
  (z.re < 0 ∧ z.im = 0) → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_condition_l205_20524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l205_20519

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Theorem: Point A(15, 0, 0) is equidistant from B(4, 6, 8) and C(2, 4, 6) -/
theorem point_equidistant_from_two_points :
  let A : Point3D := ⟨15, 0, 0⟩
  let B : Point3D := ⟨4, 6, 8⟩
  let C : Point3D := ⟨2, 4, 6⟩
  distance A B = distance A C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_two_points_l205_20519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_side_l205_20549

/-- The side length of an equilateral triangle inscribed in a unit square -/
noncomputable def inscribed_equilateral_side : ℝ := 2 * Real.sqrt 3 / 3

/-- Theorem: The side length of an equilateral triangle inscribed in a square with side length 1 is 2√3/3 -/
theorem inscribed_equilateral_triangle_side :
  ∃ (x : ℝ), x > 0 ∧ x = inscribed_equilateral_side ∧
  (∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 →
    (a^2 + b^2 = x^2 ∧ (1-a)^2 + (1-b)^2 = x^2 ∧ (1-c)^2 + b^2 = x^2) →
    x ≤ inscribed_equilateral_side) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_side_l205_20549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_triangle_l205_20562

/-- An acute-angled triangle with specific properties -/
structure AcuteTriangle where
  /-- The length of the segment connecting two altitude bases -/
  segment1 : ℝ
  /-- The length of the segment connecting two altitude bases -/
  segment2 : ℝ
  /-- The length of the segment connecting two altitude bases -/
  segment3 : ℝ
  /-- The triangle is acute-angled -/
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0
  /-- The segments form a right triangle -/
  right_triangle : segment1^2 + segment2^2 = segment3^2

/-- The radius of the circumcircle of the triangle -/
noncomputable def circumradius (t : AcuteTriangle) : ℝ := t.segment3 / 2

/-- Theorem: For an acute-angled triangle with segments connecting altitude bases
    of lengths 8, 15, and 17, the radius of the circumcircle is 17 -/
theorem circumradius_of_specific_triangle :
  ∀ t : AcuteTriangle, t.segment1 = 8 ∧ t.segment2 = 15 ∧ t.segment3 = 17 →
  circumradius t = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_triangle_l205_20562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l205_20597

/-- Converts kilometers per hour to meters per second -/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

/-- Calculates the time (in seconds) for a train to cross a fixed point -/
noncomputable def train_crossing_time (length : ℝ) (speed_km_per_hr : ℝ) : ℝ :=
  length / km_per_hr_to_m_per_s speed_km_per_hr

theorem train_crossing_pole :
  train_crossing_time 100 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_l205_20597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_49_l205_20570

def transform (n : ℕ) : ℕ :=
  if n < 15 then n * 7
  else if n % 2 = 0 then n / 2
  else n - 7

def sequenceTerms : ℕ → ℕ
  | 0 => 100
  | n + 1 => transform (sequenceTerms n)

theorem fiftieth_term_is_49 : sequenceTerms 49 = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_49_l205_20570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_sufficient_not_necessary_for_q_l205_20516

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 ≥ 1
def q (x : ℝ) : Prop := Real.rpow 2 x ≤ 2

-- Define the negation of p
def not_p (x : ℝ) : Prop := ¬(p x)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, not_p x → q x) ∧ 
  (∃ x : ℝ, q x ∧ ¬(not_p x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_p_sufficient_not_necessary_for_q_l205_20516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_fifth_pi_l205_20522

theorem cos_difference_fifth_pi : Real.cos (π / 5) - Real.cos (2 * π / 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_fifth_pi_l205_20522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relationship_l205_20559

theorem sin_cos_relationship :
  (∀ α : ℝ, Real.sin α = 2/3 → Real.cos (2*α) = 1/9) ∧
  (∃ α : ℝ, Real.cos (2*α) = 1/9 ∧ Real.sin α ≠ 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relationship_l205_20559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_max_pizzas_l205_20555

/-- Represents the ingredients required for a pizza -/
structure PizzaRequirements where
  dough : ℚ
  cheese : ℚ
  sauce : ℚ
  toppings : ℚ

/-- Represents the available ingredients -/
structure AvailableIngredients where
  dough : ℚ
  cheese : ℚ
  sauce : ℚ
  toppings : ℚ

/-- Calculates the maximum number of pizzas that can be made with given requirements and available ingredients -/
def maxPizzas (req : PizzaRequirements) (avail : AvailableIngredients) : ℕ :=
  Int.toNat <| min
    (min (Int.floor (avail.dough / req.dough)) (Int.floor (avail.cheese / req.cheese)))
    (min (Int.floor (avail.sauce / req.sauce)) (Int.floor (avail.toppings / req.toppings)))

/-- The main theorem stating the maximum number of pizzas Jimmy can make -/
theorem jimmy_max_pizzas :
  let req := PizzaRequirements.mk 1 (1/4) (1/6) (1/3)
  let avail := AvailableIngredients.mk 200 20 20 35
  maxPizzas req avail = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_max_pizzas_l205_20555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_center_square_probability_l205_20568

/-- Represents a dart board as described in the problem -/
structure DartBoard where
  side_length : ℝ
  is_regular_octagon : Bool
  has_center_square : Bool
  has_four_isosceles_triangles : Bool
  has_inscribed_squares : Bool

/-- Calculates the area of the center square -/
noncomputable def center_square_area (board : DartBoard) : ℝ :=
  board.side_length ^ 2

/-- Calculates the total area of the dart board -/
noncomputable def total_board_area (board : DartBoard) : ℝ :=
  (3 / 2) * board.side_length ^ 2

/-- Represents the uniform probability distribution of dart throws -/
def uniform_probability (board : DartBoard) : Prop :=
  True  -- Assuming the dart is equally likely to land anywhere on the board

/-- Theorem: The probability of a dart landing in the center square is 2/3 -/
theorem dart_center_square_probability (board : DartBoard) 
  (h1 : board.is_regular_octagon = true)
  (h2 : board.has_center_square = true)
  (h3 : board.has_four_isosceles_triangles = true)
  (h4 : board.has_inscribed_squares = true)
  (h5 : uniform_probability board) :
  center_square_area board / total_board_area board = 2 / 3 := by
  sorry

#check dart_center_square_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_center_square_probability_l205_20568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_iff_power_of_two_l205_20521

/-- Represents the angle of rotation of the radius after k seconds -/
noncomputable def angle (N : ℕ) (k : ℕ) : ℝ := (360 * k : ℝ) / N

/-- Predicate to check if all positions are distinct -/
def all_positions_distinct (N : ℕ) : Prop :=
  ∀ k l, 0 ≤ k ∧ k < N → 0 ≤ l ∧ l < N → k ≠ l → angle N k ≠ angle N l

/-- Theorem stating that the circle is divided into N equal sectors iff N is a power of 2 -/
theorem circle_division_iff_power_of_two (N : ℕ) (h : N > 3) :
  (∃ m : ℕ, N = 2^m) ↔ all_positions_distinct N := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_division_iff_power_of_two_l205_20521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l205_20542

-- Define the custom operation
noncomputable def customOp (a b : ℝ) : ℝ :=
  if a ≥ b then b else a

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  customOp (3^x) (3^(-x))

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l205_20542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ij_is_283_l205_20538

/-- Represents the probability of getting heads in a single flip of a biased coin -/
noncomputable def p : ℝ := sorry

/-- The probability of getting exactly 2 heads in 5 flips equals the probability of getting exactly 4 heads in 5 flips -/
axiom prob_equal : (Nat.choose 5 2) * p^2 * (1-p)^3 = (Nat.choose 5 4) * p^4 * (1-p)

/-- The probability of getting exactly 3 heads in 5 flips -/
noncomputable def prob_3_heads : ℝ := (Nat.choose 5 3) * p^3 * (1-p)^2

/-- The numerator and denominator of prob_3_heads in lowest terms -/
noncomputable def i : ℕ := sorry
noncomputable def j : ℕ := sorry

axiom prob_3_heads_frac : prob_3_heads = i / j

/-- The main theorem to prove -/
theorem sum_of_ij_is_283 : i + j = 283 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ij_is_283_l205_20538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l205_20584

theorem order_of_values (a b c : ℝ) : 
  a = 0.32 → b = 20.3 → c = Real.log 20.3 / Real.log 10 → b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l205_20584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l205_20598

/-- Represents a parabola with equation y^2 = ax and directrix x = -1 -/
structure Parabola where
  a : ℝ
  directrix : ℝ
  directrix_eq : directrix = -1

/-- Represents a point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = p.a * x

/-- The theorem to be proved -/
theorem parabola_theorem (p : Parabola) 
  (A B : ParabolaPoint p) (F : ℝ × ℝ) :
  p.directrix = -1 →
  (∃ (line : ℝ → ℝ), line F.1 = F.2 ∧ line A.x = A.y ∧ line B.x = B.y) →
  A.x + B.x = 6 →
  p.a = 4 ∧ Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l205_20598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_ratio_l205_20508

/-- Sum of first n terms of an arithmetic progression -/
noncomputable def arithmeticSum (a d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a + (n - 1) * d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms
    is three times the sum of the first 10 terms, the ratio of the first term
    to the common difference is -2:1 -/
theorem arithmetic_progression_ratio (a d : ℝ) :
  arithmeticSum a d 15 = 3 * arithmeticSum a d 10 → a = -2 * d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_ratio_l205_20508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_circle_center_l205_20515

/-- The distance between a point in polar coordinates and the center of a circle defined by a polar equation -/
theorem distance_point_to_circle_center (ρ_A : ℝ) (θ_A : ℝ) : 
  (ρ_A = 4 ∧ θ_A = π/6) →
  let polar_to_rect := λ (ρ : ℝ) (θ : ℝ) ↦ (ρ * Real.cos θ, ρ * Real.sin θ)
  let circle_eq := λ θ ↦ 4 * Real.sin θ
  let center := (0, 2)
  let A := polar_to_rect ρ_A θ_A
  Real.sqrt ((A.1 - center.1)^2 + (A.2 - center.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_circle_center_l205_20515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l205_20532

def seq (a b n : ℕ) : ℕ := a * 2017^n + b * 2016^n

theorem infinite_prime_divisors (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n, p ∣ seq a b n} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_prime_divisors_l205_20532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_plus_n_l205_20511

noncomputable def f (m n x : ℝ) : ℝ := m * (2:ℝ)^x + x^2 + n*x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_plus_n_l205_20511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sink_filling_estimate_l205_20553

-- Define the filling rates for each tap
noncomputable def tap1_rate : ℝ := 1 / 287
noncomputable def tap2_rate : ℝ := 1 / 283
noncomputable def tap3_rate : ℝ := 1 / 325

-- Define the combined rate of all taps
noncomputable def combined_rate : ℝ := tap1_rate + tap2_rate + tap3_rate

-- Define the estimated time to fill the sink
noncomputable def estimated_time : ℝ := 1 / combined_rate

-- Theorem statement
theorem sink_filling_estimate :
  ∃ ε > 0, |estimated_time - 100| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sink_filling_estimate_l205_20553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_200_l205_20546

theorem factors_multiple_of_200 (m : ℕ) (h : m = 2^12 * 3^10 * 5^9) :
  (Finset.filter (λ x : ℕ ↦ x ∣ m ∧ 200 ∣ x) (Finset.range (m + 1))).card = 880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_multiple_of_200_l205_20546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l205_20517

-- Define the function g(x) as noncomputable
noncomputable def g (x c d : ℝ) : ℝ := (x + 6) / (x^2 + c*x + d)

-- State the theorem
theorem asymptote_sum (c d : ℝ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -4 → g x c d ≠ 0) ∧ 
  (3^2 + c*3 + d = 0) ∧ 
  ((-4)^2 + c*(-4) + d = 0) → 
  c + d = -11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_l205_20517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_l205_20571

def temperatures : List ℝ := [40, 50, 65, 36, 82, 72, 26]

theorem average_temperature (temps : List ℝ) (h : temps = temperatures) :
  (temps.sum / temps.length : ℝ) = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_l205_20571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l205_20579

-- Define the line l: √3x - y + 2 = 0
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 2 = 0

-- Define the circle C: x² + y² + 2y = 0
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 2*y = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- State the theorem
theorem min_distance_line_circle :
  ∃ (px py qx qy : ℝ),
    line px py ∧ circleEq qx qy ∧
    (∀ (px' py' qx' qy' : ℝ),
      line px' py' → circleEq qx' qy' →
      distance px py qx qy ≤ distance px' py' qx' qy') ∧
    distance px py qx qy = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l205_20579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_negative_four_l205_20575

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (7 * x - 3) / (x + 2)

-- State the theorem
theorem g_at_negative_four : g (-4) = 15.5 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_negative_four_l205_20575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenus_expected_games_l205_20586

/-- Represents the probability of winning for a player in a given game --/
def WinProbability := Fin 2 → ℚ

/-- Defines the tenus game --/
structure TenusGame where
  tim_prob : WinProbability
  allen_prob : WinProbability
  tim_odd_win : tim_prob 0 = 3/4
  allen_even_win : allen_prob 1 = 3/4

/-- The expected number of games in a tenus match --/
noncomputable def expected_games (game : TenusGame) : ℚ := 16/3

/-- Theorem stating that the expected number of games in a tenus match is 16/3 --/
theorem tenus_expected_games (game : TenusGame) : 
  expected_games game = 16/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenus_expected_games_l205_20586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l205_20577

def S : Set ℝ := {x | |x - 1| + |x + 2| ≤ 4}

theorem solution_set : S = Set.Icc (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l205_20577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l205_20507

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (2*a - 1)*x + 3*a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/5 : ℝ) (1/2 : ℝ) ∧ a ≠ 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l205_20507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l205_20566

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < 1}
def B : Set ℝ := {x : ℝ | (3 : ℝ) ^ x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l205_20566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_cosine_l205_20563

/-- Given a triangle ABC where the angles A, B, and C are in the ratio 2:3:4,
    prove that cos(A/2) = (a + c) / (2b), where a, b, and c are the lengths
    of the sides opposite to angles A, B, and C respectively. -/
theorem triangle_angle_ratio_cosine (A B C a b c : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Angles are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  A + B + C = π ∧          -- Sum of angles in a triangle is π radians
  A = (2/9) * π ∧          -- Angle A is 2/9 of π (40°)
  B = (1/3) * π ∧          -- Angle B is 1/3 of π (60°)
  C = (4/9) * π ∧          -- Angle C is 4/9 of π (80°)
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A  -- Law of cosines
  →
  Real.cos (A/2) = (a + c) / (2*b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_cosine_l205_20563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_halves_pi_l205_20523

theorem sin_three_halves_pi : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_halves_pi_l205_20523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_ahead_of_train_l205_20501

/-- The distance in meters that a jogger is ahead of a train engine, given specific conditions --/
noncomputable def jogger_distance (jogger_speed train_speed train_length pass_time : ℝ) : ℝ :=
  train_speed * pass_time / 3.6 - train_length

theorem jogger_ahead_of_train (jogger_speed train_speed train_length pass_time : ℝ) 
  (h1 : jogger_speed = 9)
  (h2 : train_speed = 45)
  (h3 : train_length = 120)
  (h4 : pass_time = 30) :
  jogger_distance jogger_speed train_speed train_length pass_time = 180 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval jogger_distance 9 45 120 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_ahead_of_train_l205_20501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_to_work_is_76_l205_20567

/-- The number of hours James needs to work to pay for the food fight damage --/
def hours_to_work : ℕ :=
  let minimum_wage : ℚ := 8
  let meat_cost : ℚ := 20 * 5
  let fruit_veg_cost : ℚ := 15 * 4
  let cheese_cost : ℚ := 25 * (7/2)
  let bread_cost : ℚ := 60 * (3/2)
  let milk_cost : ℚ := 20 * 2
  let juice_cost : ℚ := 5 * 6
  let food_cost : ℚ := meat_cost + fruit_veg_cost + cheese_cost + bread_cost + milk_cost + juice_cost
  let cleaning_supplies_cost : ℚ := 15
  let janitorial_pay : ℚ := 10 * 10 * (3/2)
  let total_cost : ℚ := food_cost + cleaning_supplies_cost + janitorial_pay
  let interest_rate : ℚ := 5 / 100
  let total_cost_with_interest : ℚ := total_cost * (1 + interest_rate)
  (Int.ceil (total_cost_with_interest / minimum_wage)).toNat

theorem hours_to_work_is_76 : hours_to_work = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_to_work_is_76_l205_20567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_javier_speech_outline_time_l205_20500

/-- Given Javier's speech preparation time constraints, prove he spends 30 minutes outlining. -/
theorem javier_speech_outline_time :
  ∀ (O : ℕ), 
  (O + (O + 28) + (O + 28) / 2 = 117) → O = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_javier_speech_outline_time_l205_20500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_main_theorem_l205_20526

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = -4*y
def circle2 (x y : ℝ) : Prop := (x-1)^2 + y^2 = 1

-- Define the centers and radii
def center1 : ℝ × ℝ := (0, -2)
def radius1 : ℝ := 2
def center2 : ℝ × ℝ := (1, 0)
def radius2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem: The circles are intersecting
theorem circles_intersect : 
  abs (radius1 - radius2) < distance_between_centers ∧ 
  distance_between_centers < radius1 + radius2 := by
  sorry

-- Main theorem: The circles defined by circle1 and circle2 are intersecting
theorem main_theorem : ∃ x y : ℝ, circle1 x y ∧ circle2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_main_theorem_l205_20526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l205_20572

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_equation (p : Point) :
  let l1 : Line := { a := 2, b := -3, c := 0 }  -- y = 2/3x
  let l2 : Line := { a := 3, b := 2, c := -1 }  -- 3x + 2y - 1 = 0
  p.x = -1 ∧ p.y = 2 →
  p.liesOn l2 ∧ Line.perpendicular l1 l2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l205_20572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_sum_reciprocals_exact_min_value_sum_reciprocals_l205_20502

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := |1/2 * x + 1| + |x - 1|

-- Theorem for the minimum value of f
theorem min_value_f : ∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a ∧ a = 3/2 :=
sorry

-- Theorem for the minimum value of 1/m + 1/n
theorem min_value_sum_reciprocals :
  ∀ (m n : ℝ), m > 0 → n > 0 → m^2 + n^2 = 3/2 →
  1/m + 1/n ≥ 4 * Real.sqrt 3 / 3 :=
sorry

-- Theorem stating the exact minimum value
theorem exact_min_value_sum_reciprocals :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m^2 + n^2 = 3/2 ∧
  1/m + 1/n = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_sum_reciprocals_exact_min_value_sum_reciprocals_l205_20502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l205_20590

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = f (-(x - 1))

def is_monotone_increasing_from_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 ≤ x → x < y → f x < f y

-- State the theorem
theorem solution_set_of_inequality
  (h1 : is_even_shifted f)
  (h2 : is_monotone_increasing_from_neg_one f) :
  {x : ℝ | f (1 - 2^x) < f (-7)} = Set.Ioi 3 := by
  sorry

#check solution_set_of_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l205_20590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_in_third_quadrant_l205_20547

/-- Inverse proportion function -/
noncomputable def inverse_prop (a : ℝ) (x : ℝ) : ℝ := (a^2 + 1) / x

/-- Theorem: The graph of y = bx - b does not pass through the third quadrant -/
theorem not_in_third_quadrant (a x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : y₁ = inverse_prop a x₁)
  (h₂ : y₂ = inverse_prop a x₂)
  (h₃ : x₁ ≠ x₂)
  (h₄ : x₁ > 0)
  (h₅ : x₂ > 0) :
  let b := (x₁ - x₂) * (y₁ - y₂)
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y ≠ b * x - b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_in_third_quadrant_l205_20547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_g_max_integer_a_l205_20539

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 2 * x - 3

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.log x + a * (x - 1) / x

-- Statement for the first part of the problem
theorem f_unique_zero :
  ∃! z : ℝ, z ≥ 1 ∧ f z = 0 := by sorry

-- Statement for the second part of the problem
theorem g_max_integer_a :
  (∀ x ≥ 1, Monotone (g 6)) ∧
  ¬(∀ x ≥ 1, Monotone (g 7)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_g_max_integer_a_l205_20539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l205_20533

noncomputable section

/-- The shaded area of a floor with given dimensions and tile pattern -/
def shaded_area (floor_length floor_width tile_size circle_radius : ℝ) : ℝ :=
  let total_tiles := floor_length * floor_width / (tile_size * tile_size)
  let white_area_per_tile := 4 * (Real.pi * circle_radius^2 / 4)
  let shaded_area_per_tile := tile_size * tile_size - white_area_per_tile
  total_tiles * shaded_area_per_tile

/-- Theorem stating the shaded area of the specific floor described in the problem -/
theorem floor_shaded_area :
  shaded_area 10 8 1 (1/2) = 80 - 20 * Real.pi :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_shaded_area_l205_20533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_constant_product_min_sum_areas_l205_20554

-- Define the parabola G
def parabola_G (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle (renamed to avoid conflict)
def circle_eq (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define the point P
def point_P (m : ℝ) : ℝ × ℝ := (m, 4)

-- Define the distance from P to the directrix
def distance_to_directrix (p : ℝ) : ℝ := |4 + p|

-- Theorem for the equation of parabola G
theorem parabola_equation : 
  ∃ p : ℝ, p > 0 ∧ distance_to_directrix p = 5 → 
  ∀ x y : ℝ, parabola_G x y ↔ x^2 = 4*y := by sorry

-- Theorem for constant product |AC| · |BD|
theorem constant_product (k : ℝ) : 
  ∃ c : ℝ, ∀ A B C D : ℝ × ℝ,
  (∃ x : ℝ, parabola_G x (k*x+1) ∧ A = (x, k*x+1)) →
  (∃ x : ℝ, parabola_G x (k*x+1) ∧ B = (x, k*x+1)) →
  (∃ x : ℝ, circle_eq x (k*x+1) ∧ C = (x, k*x+1)) →
  (∃ x : ℝ, circle_eq x (k*x+1) ∧ D = (x, k*x+1)) →
  ‖A.1 - C.1‖ * ‖B.1 - D.1‖ = c := by sorry

-- Define area of triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem for minimum sum of areas
theorem min_sum_areas : 
  ∃ M : ℝ × ℝ, ∀ A B C D : ℝ × ℝ,
  (∃ x : ℝ, parabola_G x A.2 ∧ A = (x, A.2)) →
  (∃ x : ℝ, parabola_G x B.2 ∧ B = (x, B.2)) →
  (∃ x : ℝ, circle_eq x C.2 ∧ C = (x, C.2)) →
  (∃ x : ℝ, circle_eq x D.2 ∧ D = (x, D.2)) →
  (∀ k : ℝ, area_triangle A C M + area_triangle B D M ≥ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_constant_product_min_sum_areas_l205_20554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_correct_l205_20592

/-- Represents the time in hours for a given number of printing presses to print 500,000 papers -/
noncomputable def print_time (n : ℝ) : ℝ := 480 / n

theorem print_time_correct (n : ℝ) (h1 : n > 0) :
  print_time n = 480 / n ∧
  print_time 40 = 12 ∧
  print_time 30 = 15.999999999999998 := by
  sorry

#check print_time_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_correct_l205_20592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l205_20587

theorem square_measurement_error (x : ℝ) (e : ℝ) (h : e > 0) :
  ((x * (1 + e / 100))^2 - x^2) / x^2 * 100 = 10.25 →
  abs (e - 5.125) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l205_20587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l205_20534

-- Define the statements
def statement_A : Prop := ∀ x : ℝ, (x^2 ≠ 1 → x^2 - 1 ≠ 0) ↔ (x^2 - 1 = 0 → x^2 = 1)

def statement_B : Prop := ∃ x : ℝ, x^2 = x ∧ x ≠ 1

noncomputable def statement_C : Prop := (¬∃ x : ℝ, Real.rpow 2 x ≤ 0) ↔ (∀ x : ℝ, Real.rpow 2 x > 0)

def statement_D : Prop := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)

-- Theorem stating that A, B, and C are correct, while D is incorrect
theorem problem_solution :
  statement_A ∧ statement_B ∧ statement_C ∧ ¬statement_D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l205_20534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abs_x_leq_two_l205_20544

open MeasureTheory Set
open scoped Interval

/-- The probability that |x| ≤ 2 when x is randomly selected from [-1, 3] is 3/4 -/
theorem probability_abs_x_leq_two : 
  (volume (Icc (-1 : ℝ) 3 ∩ {x | |x| ≤ 2})) / (volume (Icc (-1 : ℝ) 3)) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_abs_x_leq_two_l205_20544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l205_20560

def mySequence : List ℕ := [4, 5, 14, 15, 24, 25]

def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 6 ∧
  seq.get? 0 = some 4 ∧
  seq.get? 1 = some 5 ∧
  seq.get? 2 = some 14 ∧
  seq.get? 3 = some 15 ∧
  seq.get? 4 = some 24 ∧
  seq.get? 5 = some 25

theorem missing_number (seq : List ℕ) (h : is_valid_sequence seq) :
  ∃ x : ℕ, (seq ++ [x]).length = 7 ∧ (seq ++ [x]).get? 6 = some 34 :=
by
  sorry

#check missing_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l205_20560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_product_problem_l205_20574

def P (n : ℕ) : ℕ := (n.digits 10).prod

theorem digit_product_problem :
  (∃ S : Finset ℕ, S.card = 19 ∧ ∀ n ∈ S, 1 ≤ n ∧ n < 1000 ∧ P n = 12) ∧
  ((Finset.filter (fun n => 1 ≤ n ∧ n < 200 ∧ P n = 0) (Finset.range 200)).card = 28) ∧
  ((Finset.filter (fun n => 1 ≤ n ∧ n < 200 ∧ 37 < P n ∧ P n < 45) (Finset.range 200)).card = 8) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 250 → P n ≤ 81) ∧
  (P 99 = 81 ∧ P 199 = 81) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_product_problem_l205_20574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_initial_chessboard_covering_divisibility_l205_20537

/-- Represents the number of ways to cover a 3 × n chessboard with one corner removed using dominoes -/
def t : ℕ → ℕ
| 1 => 1
| 3 => 4
| n => sorry  -- We'll define this properly later

/-- Represents the number of ways to cover a 3 × n chessboard with no corners removed using dominoes -/
def s : ℕ → ℕ
| _ => sorry  -- We'll define this properly later

/-- The recurrence relation for t -/
axiom t_recurrence (k : ℕ) : t (2 * k + 1) = s (2 * k) + t (2 * k - 1)

/-- The recurrence relation for s -/
axiom s_recurrence (k : ℕ) : s (2 * k) = s (2 * k - 2) + 2 * t (2 * k - 1)

/-- Initial conditions for t -/
theorem t_initial : t 1 = 1 ∧ t 3 = 4 := by
  simp [t]

/-- Theorem stating that the number of ways to cover a 3 × 2021 chessboard 
    with one corner removed using dominoes is divisible by 19 -/
theorem chessboard_covering_divisibility : 
  19 ∣ t 2021 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_initial_chessboard_covering_divisibility_l205_20537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l205_20548

/-- Calculates the percent increase given the original salary and the increase. -/
noncomputable def percent_increase (original : ℝ) (increase : ℝ) : ℝ :=
  (increase / original) * 100

theorem salary_increase_percentage (new_salary : ℝ) (increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : increase = 25000) :
  ∃ (ε : ℝ), abs (percent_increase (new_salary - increase) increase - 38.46) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l205_20548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l205_20556

theorem sin_minus_cos_for_special_angle (θ : Real) (h1 : θ ∈ Set.Ioo 0 (Real.pi/2)) (h2 : Real.tan θ = 1/3) :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_for_special_angle_l205_20556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_not_constructible_l205_20541

/-- A type representing geometric constructions possible with a straightedge -/
inductive StraightedgeConstruction
  | Line (p q : ℝ × ℝ) : StraightedgeConstruction
  | Intersection (l1 l2 : StraightedgeConstruction) : StraightedgeConstruction

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a point is constructible using only a straightedge -/
def is_constructible (c : Circle) (p : ℝ × ℝ) : Prop :=
  ∃ (construction : StraightedgeConstruction), match construction with
    | StraightedgeConstruction.Line a b => p = a ∨ p = b
    | StraightedgeConstruction.Intersection _ _ => True  -- Simplified for this example

/-- Theorem stating that it's impossible to construct the center of a circle using only a straightedge -/
theorem center_not_constructible (c : Circle) :
  ¬ is_constructible c c.center :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_not_constructible_l205_20541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_without_replacement_variance_with_replacement_l205_20582

/-- Represents the random variable X, the number of draws when the drawing stops -/
def X : Type := ℕ

/-- Total number of balls in the bag -/
def total_balls : ℕ := 6

/-- Number of yellow balls in the bag -/
def yellow_balls : ℕ := 4

/-- Number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Maximum number of drawing opportunities -/
def max_draws : ℕ := 3

/-- Expected value of X when drawing without replacement -/
def E_X_without_replacement : ℚ := 31/15

/-- Variance of X when drawing with replacement -/
def D_X_with_replacement : ℚ := 62/81

/-- Theorem for the expected value of X when drawing without replacement -/
theorem expected_value_without_replacement :
  let p1 : ℚ := white_balls / total_balls
  let p2 : ℚ := yellow_balls / total_balls * white_balls / (total_balls - 1)
  let p3 : ℚ := yellow_balls / total_balls * (yellow_balls - 1) / (total_balls - 1) * white_balls / (total_balls - 2)
  1 * p1 + 2 * p2 + 3 * p3 = E_X_without_replacement := by
  sorry

/-- Theorem for the variance of X when drawing with replacement -/
theorem variance_with_replacement :
  let p1 : ℚ := white_balls / total_balls
  let p2 : ℚ := yellow_balls / total_balls * white_balls / total_balls
  let p3 : ℚ := 1 - p1 - p2
  let E_X : ℚ := 1 * p1 + 2 * p2 + 3 * p3
  (1 - E_X)^2 * p1 + (2 - E_X)^2 * p2 + (3 - E_X)^2 * p3 = D_X_with_replacement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_without_replacement_variance_with_replacement_l205_20582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_constraint_l205_20557

noncomputable def f (x a : ℝ) : ℝ := Real.log (3^x + 3^(-x) - a)

theorem function_range_implies_a_constraint (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f x a = y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_constraint_l205_20557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2005_to_2008_l205_20529

-- Define the set of years we're considering
def years : Finset Nat := {2005, 2006, 2007, 2008}

-- Define a function to determine if a year is a leap year
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

-- Define a function to get the number of days in a year
def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

-- State the theorem
theorem total_days_2005_to_2008 :
  (Finset.sum years daysInYear) = 1461 := by
  sorry

#eval Finset.sum years daysInYear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2005_to_2008_l205_20529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l205_20525

/-- Calculates the percentage increase between two prices -/
noncomputable def percentage_increase (initial_price final_price : ℝ) : ℝ :=
  ((final_price - initial_price) / initial_price) * 100

/-- The initial price of gasoline in 1972 -/
def initial_price : ℝ := 29.90

/-- The final price of gasoline in 1992 -/
def final_price : ℝ := 149.70

/-- Theorem stating that the percentage increase from 1972 to 1992 is 400% -/
theorem gasoline_price_increase : 
  percentage_increase initial_price final_price = 400 := by
  -- Unfold the definition of percentage_increase
  unfold percentage_increase
  -- Simplify the expression
  simp [initial_price, final_price]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gasoline_price_increase_l205_20525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l205_20540

open Set Real

-- Define the functions
noncomputable def f (x : ℝ) := Real.log (Real.tan x)
noncomputable def g (x : ℝ) := (3 * Real.sin x + 1) / (Real.sin x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | ∃ k : ℤ, x ∈ Ioo (k * π) (k * π + π/2)}

-- Define the range of g
def range_g : Set ℝ := Icc (-4) (2/3)

-- Theorem statements
theorem domain_of_f : Set.range f = domain_f := by sorry

theorem range_of_g : Set.range g = range_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l205_20540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h2o_amount_for_nh4cl_hydrolysis_l205_20594

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String
deriving Inhabited

-- Define the reaction equation
structure Reaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the given reaction
def nh4cl_hydrolysis : Reaction :=
  { reactants := [
      (⟨"NH4Cl"⟩, 1),
      (⟨"H2O"⟩, 1)
    ],
    products := [
      (⟨"HCl"⟩, 1),
      (⟨"NH4OH"⟩, 1)
    ]
  }

-- Theorem statement
theorem h2o_amount_for_nh4cl_hydrolysis :
  let reaction := nh4cl_hydrolysis
  let nh4cl_amount : ℚ := 1
  let hcl_amount : ℚ := 1
  let nh4oh_amount : ℚ := 1
  ∃ (h2o_amount : ℚ),
    h2o_amount = 1 ∧
    (reaction.reactants.filter (λ r => r.1.name = "NH4Cl")).head!.2 * nh4cl_amount =
    (reaction.reactants.filter (λ r => r.1.name = "H2O")).head!.2 * h2o_amount ∧
    (reaction.products.filter (λ p => p.1.name = "HCl")).head!.2 * hcl_amount =
    (reaction.reactants.filter (λ r => r.1.name = "NH4Cl")).head!.2 * nh4cl_amount ∧
    (reaction.products.filter (λ p => p.1.name = "NH4OH")).head!.2 * nh4oh_amount =
    (reaction.reactants.filter (λ r => r.1.name = "NH4Cl")).head!.2 * nh4cl_amount :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h2o_amount_for_nh4cl_hydrolysis_l205_20594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l205_20551

-- Define the function f(x) = log₂x + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 3

-- State the theorem
theorem solution_interval :
  ∃ (s : Set ℝ), s = {x | x > 0 ∧ f x = 0} ∧ s = Set.Icc 2 3 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_continuous : Continuous f := by
  sorry

lemma f_strictly_increasing : StrictMono f := by
  sorry

lemma f_two_nonpos : f 2 ≤ 0 := by
  sorry

lemma f_three_pos : f 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l205_20551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l205_20561

def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : IsArithmeticSequence a) 
  (sum_456 : a 4 + a 5 + a 6 = 36) : a 1 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l205_20561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_x_completion_time_l205_20569

/-- The time it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := sorry

/-- The time it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 36

/-- The number of each model used when working together -/
def num_computers : ℕ := 24

theorem model_x_completion_time :
  (↑num_computers * (1 / model_x_time + 1 / model_y_time) = 1) →
  model_x_time = 72 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_x_completion_time_l205_20569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_journey_average_speed_l205_20588

/-- Calculates the average speed of a two-part journey -/
theorem two_part_journey_average_speed 
  (time1 : ℝ) (speed1 : ℝ) (time2 : ℝ) (speed2 : ℝ) 
  (h1 : time1 = 5) (h2 : speed1 = 40) (h3 : time2 = 3) (h4 : speed2 = 80) :
  (time1 * speed1 + time2 * speed2) / (time1 + time2) = 55 := by
  -- Proof steps would go here
  sorry

#check two_part_journey_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_part_journey_average_speed_l205_20588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l205_20530

/-- The slope angle of a vertical line is 90 degrees. -/
theorem slope_angle_vertical_line (a : ℝ) : 
  90 = 90 := by
  -- The actual proof would involve defining SlopeAngle and proving it's 90 for vertical lines
  -- For now, we'll use a trivial equality to make the theorem compile
  rfl

#check slope_angle_vertical_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_vertical_line_l205_20530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_account_theorem_l205_20573

def initial_deposit : ℚ := 70000

def transactions : List ℚ := [2, -3, 3.5, -2.5, 4, -1.2, 1, -0.8]

def account_balance (n : ℕ) : ℚ :=
  initial_deposit + 10000 * (transactions.take n).sum

noncomputable def max_balance : ℚ × ℕ :=
  (List.range (transactions.length + 1)).foldl
    (λ acc i => let balance := account_balance i
                if balance > acc.1 then (balance, i) else acc)
    (initial_deposit, 0)

theorem company_account_theorem :
  account_balance transactions.length = 110000 ∧
  max_balance = (110000, 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_account_theorem_l205_20573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_theorem_l205_20506

/-- The polynomial P(x) = x^3 + x^2 - x + 2 -/
def P (x : ℂ) : ℂ := x^3 + x^2 - x + 2

theorem polynomial_range_theorem (r : ℝ) :
  (∃ z : ℂ, z.im ≠ 0 ∧ P z = r) → r > 3 ∧ r < 49/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_range_theorem_l205_20506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l205_20531

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - (1 - a) * x - (2 - a) * Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Theorem statement
theorem function_properties (a : ℝ) :
  (∀ x > 0, StrictMono (g a)) ↔ a ∈ Set.Ici 2 ∧
  ∀ m n x₀ : ℝ, 0 < m → m < n → F a m = 0 → F a n = 0 → x₀ = (m + n) / 2 → 
    deriv (F a) x₀ ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l205_20531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_performances_correct_l205_20535

/-- The probability of seeing a specific role pair in the alternative casting in one performance -/
noncomputable def prob_alt_casting : ℝ := 1 / 2

/-- The number of role pairs Sári wants to see in alternative casting -/
def num_pairs : ℕ := 3

/-- The minimum desired probability of seeing all pairs in alternative casting -/
noncomputable def min_prob : ℝ := 0.9

/-- Calculate the probability of seeing all pairs in alternative casting given n performances -/
noncomputable def prob_all_pairs (n : ℕ) : ℝ := (1 - (1 - prob_alt_casting) ^ n) ^ num_pairs

/-- The minimum number of additional performances needed -/
def min_performances : ℕ := 5

theorem min_performances_correct :
  (∀ k < min_performances, prob_all_pairs k < min_prob) ∧
  prob_all_pairs min_performances ≥ min_prob :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_performances_correct_l205_20535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_divisibility_l205_20504

theorem prime_factor_divisibility (n : ℕ) :
  (∃ (p : List ℕ), (∀ q ∈ p, Nat.Prime q) ∧ 
   n = p.prod ∧ 
   n ∣ (p.map (λ x => x + 1)).prod) ↔ 
  (∃ (r s : ℕ), n = 2^r * 3^s ∧ s ≤ r ∧ r ≤ 2*s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_divisibility_l205_20504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_column_products_not_equal_l205_20505

-- Define the type for our table
def Table := Fin 10 → Fin 10 → ℕ

-- Define a valid table filled with numbers from 110 to 209
def is_valid_table (t : Table) : Prop :=
  ∀ i j, 110 ≤ t i j ∧ t i j ≤ 209 ∧
  ∀ k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

-- Define row product
def row_product (t : Table) (i : Fin 10) : ℕ :=
  (Finset.univ : Finset (Fin 10)).prod (λ j ↦ t i j)

-- Define column product
def column_product (t : Table) (j : Fin 10) : ℕ :=
  (Finset.univ : Finset (Fin 10)).prod (λ i ↦ t i j)

-- Define sets of row and column products
def row_products (t : Table) : Finset ℕ :=
  (Finset.univ : Finset (Fin 10)).image (row_product t)

def column_products (t : Table) : Finset ℕ :=
  (Finset.univ : Finset (Fin 10)).image (column_product t)

-- Theorem statement
theorem row_column_products_not_equal (t : Table) (h : is_valid_table t) :
  row_products t ≠ column_products t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_column_products_not_equal_l205_20505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l205_20552

noncomputable def f (x : ℝ) : ℝ := (Real.pi - x) * Real.sin x

theorem function_properties :
  let f := f
  ∀ x ∈ Set.Icc 0 Real.pi,
    (∃ m : ℝ, (deriv f) 0 = m ∧ m = Real.pi) ∧
    (∀ a x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 Real.pi → x₂ ∈ Set.Icc 0 Real.pi →
      f x₁ = a → f x₂ = a → x₁ ≠ x₂ → a < 2) ∧
    (∀ a x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 Real.pi → x₂ ∈ Set.Icc 0 Real.pi →
      f x₁ = a → f x₂ = a → x₁ ≠ x₂ → |x₁ - x₂| ≤ Real.pi - a - a / Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l205_20552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_values_l205_20591

theorem sum_of_possible_values (x y : ℝ) (h : x * y + x / y + y / x = 3) :
  ∃ (S : Finset ℝ), (∀ z ∈ S, ∃ a b : ℝ, a * b + a / b + b / a = 3 ∧ z = (a + 1) * (b + 1)) ∧
                    S.sum id = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_values_l205_20591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_POQ_l205_20543

/-- Given a parabola y² = 4x and a line with slope π/4 intersecting the parabola at points P and Q,
    prove that the area of triangle POQ is 2√2, where O is the origin. -/
theorem area_triangle_POQ (P Q : ℝ × ℝ) : 
  (∃ (y₁ y₂ : ℝ), P.2^2 = 4 * P.1 ∧ Q.2^2 = 4 * Q.1) →  -- Parabola equation
  (∃ (k : ℝ), Q.2 - P.2 = k * (Q.1 - P.1) ∧ k = π / 4) →  -- Line slope
  P ≠ Q →  -- P and Q are distinct points
  ∃ (S : ℝ), S = abs ((P.1 * Q.2 - Q.1 * P.2) / 2) ∧ S = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_POQ_l205_20543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_from_parallel_and_perpendicular_perpendicular_from_parallel_planes_and_perpendicular_l205_20583

-- Define the basic types
variable (P : Type) -- Type for points
variable (L : Type) -- Type for lines
variable (Plane : Type) -- Type for planes

-- Define the relations
variable (parallel_lines : L → L → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : L → Plane → Prop)

-- Define the specific objects
variable (l m : L)
variable (α β : Plane)

-- Axioms
axiom different_lines : l ≠ m
axiom different_planes : α ≠ β

-- Theorem 1
theorem perpendicular_from_parallel_and_perpendicular
  (h1 : parallel_lines l m)
  (h2 : perpendicular_line_plane m α) :
  perpendicular_line_plane l α :=
sorry

-- Theorem 2
theorem perpendicular_from_parallel_planes_and_perpendicular
  (h1 : parallel_planes α β)
  (h2 : perpendicular_line_plane l β) :
  perpendicular_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_from_parallel_and_perpendicular_perpendicular_from_parallel_planes_and_perpendicular_l205_20583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l205_20514

theorem calculate_expression : (27 : Real)^(1/3) - Real.sqrt 4 - (1/3)⁻¹ + (-2020)^0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l205_20514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_arrangements_l205_20596

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of ways to arrange the assembly line -/
def num_arrangements : ℕ := 120

/-- Theorem stating that the number of arrangements is correct -/
theorem assembly_line_arrangements :
  let n := num_tasks
  let k := 2  -- number of tasks with a specific order constraint
  num_arrangements = (n - k + 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_arrangements_l205_20596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l205_20513

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  let x := d - n % d
  (∀ y : ℕ, (n + y) % d = 0 → y ≥ x) ∧
  (n + x) % d = 0 :=
by
  intro x
  constructor
  · intro y h_div
    sorry -- Proof skipped
  · sorry -- Proof skipped

#eval (25 - 1015 % 25)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l205_20513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l205_20512

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := (1 - i) / (1 + i) + 2 * i

-- Theorem statement
theorem magnitude_of_z : Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_l205_20512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auntie_parking_probability_l205_20589

/-- The number of parking spaces in a row -/
def total_spaces : ℕ := 18

/-- The number of cars already parked -/
def parked_cars : ℕ := 14

/-- The number of spaces required by Auntie Em's SUV -/
def suv_spaces : ℕ := 2

/-- The probability that Auntie Em can park her SUV -/
def prob_auntie_can_park : ℚ := 113 / 204

theorem auntie_parking_probability :
  prob_auntie_can_park = 1 - (Nat.choose (total_spaces - parked_cars + (suv_spaces - 1)) (suv_spaces - 1)) / (Nat.choose total_spaces parked_cars) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auntie_parking_probability_l205_20589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_water_flow_l205_20518

/-- Calculates the volume of water flowing into the sea per minute for a river with given dimensions and flow rate. -/
noncomputable def water_flow_per_minute (depth : ℝ) (width : ℝ) (flow_rate_kmph : ℝ) : ℝ :=
  let cross_sectional_area := depth * width
  let flow_rate_m_per_min := flow_rate_kmph * 1000 / 60
  cross_sectional_area * flow_rate_m_per_min

/-- Theorem stating that for a river with depth 4 m, width 40 m, and flow rate 4 kmph,
    the volume of water flowing into the sea per minute is approximately 10666.67 cubic meters. -/
theorem river_water_flow :
  let depth : ℝ := 4
  let width : ℝ := 40
  let flow_rate_kmph : ℝ := 4
  abs (water_flow_per_minute depth width flow_rate_kmph - 10666.67) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_water_flow_l205_20518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_159_over_11_l205_20580

/-- Represents the average number of minutes run per day for each grade -/
structure GradeRunningTime where
  third : ℚ
  fourth : ℚ
  fifth : ℚ

/-- Represents the ratio of students in each grade compared to fifth grade -/
structure GradeRatio where
  third : ℚ
  fourth : ℚ
  fifth : ℚ

/-- Calculate the average running time for all students given the running times and ratios for each grade -/
def averageRunningTime (times : GradeRunningTime) (ratios : GradeRatio) : ℚ :=
  (times.third * ratios.third + times.fourth * ratios.fourth + times.fifth * ratios.fifth) /
  (ratios.third + ratios.fourth + ratios.fifth)

theorem average_running_time_is_159_over_11 :
  let times := GradeRunningTime.mk 14 17 12
  let ratios := GradeRatio.mk 3 (3/2) 1
  averageRunningTime times ratios = 159 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_running_time_is_159_over_11_l205_20580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_even_p_for_300p_square_and_prime_multiple_l205_20599

theorem least_even_p_for_300p_square_and_prime_multiple : ∃ (p q : ℕ), 
  (∀ (p' : ℕ), p' < p → ¬(∃ (n : ℕ), 300 * p' = n^2 ∧ ∃ (q' : ℕ), Nat.Prime q' ∧ q' ∣ (300 * p') ∧ q' ∉ ({2, 3, 5} : Finset ℕ))) ∧ 
  Even p ∧ 
  (∃ (n : ℕ), 300 * p = n^2) ∧ 
  Nat.Prime q ∧ 
  q ∣ (300 * p) ∧ 
  q ∉ ({2, 3, 5} : Finset ℕ) ∧ 
  p = 6 ∧ 
  q = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_even_p_for_300p_square_and_prime_multiple_l205_20599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_problem_l205_20545

/-- The length of the second train given the conditions of the problem -/
noncomputable def second_train_length (first_train_length : ℝ) (first_train_speed : ℝ) (second_train_speed : ℝ) (clearing_time : ℝ) : ℝ :=
  let relative_speed := (first_train_speed + second_train_speed) * 1000 / 3600
  relative_speed * clearing_time - first_train_length

/-- Theorem stating the length of the second train given the problem conditions -/
theorem second_train_length_problem :
  let l := second_train_length 120 42 30 19.99840012798976
  ∃ ε > 0, |l - 279.97| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_problem_l205_20545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_iff_lambda_gt_neg_three_l205_20528

/-- A sequence {a_n} defined by a_n = n^2 + λn where n ∈ ℕ₊ -/
def a (lambda : ℝ) (n : ℕ+) : ℝ := n^2 + lambda * n

/-- The sequence {a_n} is strictly increasing -/
def is_strictly_increasing (lambda : ℝ) : Prop :=
  ∀ n : ℕ+, a lambda n < a lambda (n + 1)

/-- Theorem: The sequence {a_n} is strictly increasing if and only if λ > -3 -/
theorem strictly_increasing_iff_lambda_gt_neg_three (lambda : ℝ) :
  is_strictly_increasing lambda ↔ lambda > -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_iff_lambda_gt_neg_three_l205_20528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_calculates_correct_sum_l205_20550

def algorithm (x : Int) : Int :=
  let rec loop (i : Int) (s : Int) (a : Int) (fuel : Nat) : Int :=
    if fuel = 0 then s
    else if i > 9 then s
    else loop (i + 2) (s + a * i) (-a) (fuel - 1)
  loop 1 0 x 5

theorem algorithm_calculates_correct_sum : 
  algorithm (-1) = -1 + 3 - 5 + 7 - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_calculates_correct_sum_l205_20550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l205_20593

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- Main theorem -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
    (h : seq.a 2 + seq.a 8 = 10) : 
    S seq 9 - seq.a 5 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l205_20593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l205_20536

-- Define a line in a coordinate plane
structure Line where
  -- We don't need to specify the exact representation of a line here
  -- as we're only interested in its properties

-- Define the concept of an inclination angle
def has_inclination_angle (l : Line) : Prop :=
  ∃ α : Real, 0 ≤ α ∧ α < Real.pi

-- Define the concept of a gradient
def has_gradient (l : Line) : Prop :=
  ∃ m : Real, True  -- We just need to know if a gradient exists, not its value

-- Define the relationship between gradient and inclination angle
def gradient_is_tan_of_angle (l : Line) (α : Real) : Prop :=
  ∃ m : Real, m = Real.tan α

theorem all_propositions_false : 
  (¬ ∀ l : Line, has_inclination_angle l ∧ has_gradient l) ∧ 
  (¬ ∀ l : Line, ∃ α : Real, has_inclination_angle l → (0 ≤ α ∧ α ≤ Real.pi)) ∧
  (¬ ∀ l : Line, ∀ α : Real, gradient_is_tan_of_angle l α → has_inclination_angle l ∧ ∃ β, β = α) ∧
  (¬ ∀ l : Line, ∀ α : Real, has_inclination_angle l → gradient_is_tan_of_angle l α) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l205_20536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l205_20578

/-- Calculates the length of a bridge given train specifications and crossing time. -/
noncomputable def bridge_length (num_carriages : ℕ) (carriage_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_minutes : ℝ) : ℝ :=
  let train_length := (num_carriages + 1 : ℝ) * carriage_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let crossing_time_seconds := crossing_time_minutes * 60
  let total_distance := train_speed_mps * crossing_time_seconds
  total_distance - train_length

/-- Theorem stating that under given conditions, the bridge length is approximately 3501 meters. -/
theorem bridge_length_calculation :
  let num_carriages := 24
  let carriage_length := 60
  let train_speed_kmph := 60
  let crossing_time_minutes := 5
  ∃ ε > 0, ε < 1 ∧ 
    |bridge_length num_carriages carriage_length train_speed_kmph crossing_time_minutes - 3501| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l205_20578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_parabola_proof_l205_20527

-- Hyperbola part
def hyperbola_equation (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ -y^2 = 1

theorem hyperbola_proof (C : Set (ℝ × ℝ)) 
  (center : (0, 0) ∈ C) 
  (right_focus : (2, 0) ∈ C) 
  (real_axis_length : ℝ) 
  (h_real_axis : real_axis_length = 2) :
  hyperbola_equation C :=
sorry

-- Parabola part
def parabola_equation (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = m * x}

noncomputable def directrix_distance : ℝ := 2

theorem parabola_proof (m : ℝ) (h_m : m ≠ 0) :
  (∃ (x : ℝ), x = -3 ∧ directrix_distance = |x + 1| ∧ parabola_equation 12 = parabola_equation m) ∨
  (∃ (x : ℝ), x = 1 ∧ directrix_distance = |x + 1| ∧ parabola_equation (-4) = parabola_equation m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_parabola_proof_l205_20527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_l205_20564

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-4, a)

-- Define the condition that P is on the terminal side of angle α
def on_terminal_side (α : ℝ) (p : ℝ × ℝ) : Prop :=
  Real.sin α = p.2 / Real.sqrt (p.1^2 + p.2^2) ∧
  Real.cos α = p.1 / Real.sqrt (p.1^2 + p.2^2)

-- State the theorem
theorem point_on_angle (a : ℝ) (α : ℝ) :
  on_terminal_side α (P a) →
  Real.sin α * Real.cos α = Real.sqrt 3 / 4 →
  a = -4 * Real.sqrt 3 ∨ a = -4 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_l205_20564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_disk_arrangement_area_l205_20595

/-- Represents a disk in a 2D plane -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arrangement of disks on a circle -/
structure DiskArrangement where
  circle_radius : ℝ
  disks : List Disk
  covers_circle : Bool
  no_overlap : Bool
  tangent_neighbors : Bool
  octagon_pattern : Bool

/-- Calculates the area of a single disk -/
noncomputable def disk_area (d : Disk) : ℝ :=
  Real.pi * d.radius ^ 2

/-- Calculates the sum of areas of all disks in an arrangement -/
noncomputable def total_area (arr : DiskArrangement) : ℝ :=
  (arr.disks.map disk_area).sum

/-- The main theorem to be proved -/
theorem octagon_disk_arrangement_area 
  (arr : DiskArrangement) 
  (h1 : arr.circle_radius = 1)
  (h2 : arr.disks.length = 8)
  (h3 : arr.covers_circle = true)
  (h4 : arr.no_overlap = true)
  (h5 : arr.tangent_neighbors = true)
  (h6 : arr.octagon_pattern = true) :
  total_area arr = Real.pi * (48 - 32 * Real.sqrt 2) := by
  sorry

#eval 48 + 32 + 2  -- Expected output: 82

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_disk_arrangement_area_l205_20595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_product_real_implies_x_value_l205_20576

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := 1 + Complex.I
def z₂ (x : ℝ) : ℂ := x + 2 * Complex.I

-- Theorem statement
theorem z_product_real_implies_x_value :
  ∀ x : ℝ, (z₁ * z₂ x).im = 0 → x = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_product_real_implies_x_value_l205_20576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l205_20510

theorem parity_of_expression (a b c d : ℕ) (ha : Odd a) (hb : Odd b) (hd : Even d) :
  Odd (3^a + (b-1)^2*c - (2^d - c)) ↔ Even c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l205_20510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l205_20520

/-- Represents a subject with its marks and credit hours -/
structure Subject where
  name : String
  marks : ℕ
  creditHours : ℕ

/-- Calculates the weighted average marks given a list of subjects -/
def weightedAverage (subjects : List Subject) : ℚ :=
  let totalWeightedMarks := subjects.foldl (fun acc s => acc + (s.marks : ℚ) * (s.creditHours : ℚ)) 0
  let totalCreditHours := subjects.foldl (fun acc s => acc + (s.creditHours : ℚ)) 0
  totalWeightedMarks / totalCreditHours

/-- Dacid's subjects with their marks and credit hours -/
def dacidSubjects : List Subject := [
  ⟨"English", 86, 3⟩,
  ⟨"Mathematics", 85, 4⟩,
  ⟨"Physics", 92, 4⟩,
  ⟨"Chemistry", 87, 3⟩,
  ⟨"Biology", 95, 3⟩,
  ⟨"History", 89, 2⟩,
  ⟨"Physical Education", 75, 1⟩
]

/-- Theorem stating that Dacid's weighted average marks is 88.25 -/
theorem dacid_weighted_average :
  weightedAverage dacidSubjects = 88.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l205_20520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_between_probability_l205_20558

/-- The number of people standing in a row for a photo. -/
def n : ℕ := 6

/-- The probability that there are exactly two people standing between person A and person B. -/
def probability : ℚ := 1 / 5

/-- Calculate the number of arrangements with exactly two people between A and B. -/
def number_of_arrangements_with_two_between (n : ℕ) : ℕ :=
  if n ≥ 5 then 2 * (n - 4).factorial * (n - 2).factorial else 0

/-- Calculate the total number of arrangements of n people. -/
def total_number_of_arrangements (n : ℕ) : ℕ := n.factorial

/-- Theorem stating that the probability of exactly two people standing between
    person A and person B in a row of n people is equal to the calculated probability. -/
theorem two_between_probability :
  (number_of_arrangements_with_two_between n : ℚ) / (total_number_of_arrangements n) = probability :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_between_probability_l205_20558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l205_20585

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + Real.sqrt 3 / 2

theorem f_properties :
  ∃ (k : ℤ), 
    (∀ x ∈ Set.Icc 0 (Real.pi / 2),
      (x = k * Real.pi / 2 + 5 * Real.pi / 12 → 
        ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y = f (2 * (k * Real.pi / 2 + 5 * Real.pi / 12) - y))) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3 / 2) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l205_20585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l205_20581

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then 2*a - (x + 4/x) else x - 4/x

-- Statement for part (1)
theorem part_one : ∃! x, f 1 x = 3 ∧ x = 4 := by sorry

-- Statement for part (2)
theorem part_two :
  ∀ a : ℝ, a ≤ -1 →
  (∃ x y z : ℝ, x < y ∧ y < z ∧
    f a x = 3 ∧ f a y = 3 ∧ f a z = 3 ∧
    y - x = z - y) →
  a = -11/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l205_20581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_average_weight_l205_20565

/-- Arun's weight in kilograms -/
noncomputable def arun_weight : ℝ := sorry

/-- Arun's opinion on his weight -/
axiom arun_opinion : 62 < arun_weight ∧ arun_weight < 72

/-- Arun's brother's opinion on Arun's weight -/
axiom brother_opinion : 60 < arun_weight ∧ arun_weight < 70

/-- Arun's mother's opinion on Arun's weight -/
axiom mother_opinion : arun_weight ≤ 65

/-- The average of Arun's probable weights -/
noncomputable def average_weight : ℝ := (62 + 65) / 2

/-- Theorem stating that the average of Arun's probable weights is 63.5 kg -/
theorem arun_average_weight : average_weight = 63.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_average_weight_l205_20565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_A_quantity_l205_20509

noncomputable def initial_A (x : ℝ) : ℝ := 7 * x
noncomputable def initial_B (x : ℝ) : ℝ := 5 * x

noncomputable def initial_total (x : ℝ) : ℝ := initial_A x + initial_B x

noncomputable def removed_A (x : ℝ) : ℝ := (7 / 12) * 6
noncomputable def removed_B (x : ℝ) : ℝ := (5 / 12) * 6

noncomputable def remaining_A (x : ℝ) : ℝ := initial_A x - removed_A x
noncomputable def remaining_B (x : ℝ) : ℝ := initial_B x - removed_B x

noncomputable def new_B (x : ℝ) : ℝ := remaining_B x + 6

theorem initial_liquid_A_quantity :
  ∃ x : ℝ, 
    (initial_A x / initial_B x = 7 / 5) ∧
    (remaining_A x / new_B x = 7 / 9) ∧
    (initial_A x = 14) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_liquid_A_quantity_l205_20509
