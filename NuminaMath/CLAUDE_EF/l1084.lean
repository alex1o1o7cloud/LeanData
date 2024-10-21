import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_seating_arrangements_l1084_108465

/-- Represents the Lopez family members -/
inductive FamilyMember
  | MrLopez
  | MrsLopez
  | Child1
  | Child2
deriving Fintype, DecidableEq

/-- Represents a seating arrangement in the car -/
structure SeatingArrangement where
  driver : FamilyMember
  frontPassenger : FamilyMember
  backChild : FamilyMember
deriving Fintype, DecidableEq

/-- The set of all possible seating arrangements with the dog in the car -/
def seatingArrangements : Set SeatingArrangement :=
  { arr | (arr.driver = FamilyMember.MrLopez ∨ arr.driver = FamilyMember.MrsLopez) ∧
          arr.driver ≠ arr.frontPassenger ∧
          arr.driver ≠ arr.backChild ∧
          arr.frontPassenger ≠ arr.backChild ∧
          (arr.backChild = FamilyMember.Child1 ∨ arr.backChild = FamilyMember.Child2) }

/-- Predicate to check if a seating arrangement is valid -/
def isValidArrangement (arr : SeatingArrangement) : Prop :=
  (arr.driver = FamilyMember.MrLopez ∨ arr.driver = FamilyMember.MrsLopez) ∧
  arr.driver ≠ arr.frontPassenger ∧
  arr.driver ≠ arr.backChild ∧
  arr.frontPassenger ≠ arr.backChild ∧
  (arr.backChild = FamilyMember.Child1 ∨ arr.backChild = FamilyMember.Child2)

instance : DecidablePred isValidArrangement := by
  intro arr
  apply And.decidable
  all_goals { apply instDecidable }

theorem lopez_seating_arrangements :
  Finset.card (Finset.filter isValidArrangement Finset.univ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lopez_seating_arrangements_l1084_108465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_great_white_shark_teeth_l1084_108479

/-- The number of teeth in different shark species -/
structure SharkTeeth where
  tiger : ℕ
  hammerhead : ℕ
  great_white : ℕ

/-- Given conditions about shark teeth -/
def shark_teeth_conditions (s : SharkTeeth) : Prop :=
  s.tiger = 180 ∧
  s.hammerhead = s.tiger / 6 ∧
  s.great_white = 2 * (s.tiger + s.hammerhead)

/-- Theorem stating the number of teeth in a great white shark -/
theorem great_white_shark_teeth (s : SharkTeeth) 
  (h : shark_teeth_conditions s) : s.great_white = 420 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_great_white_shark_teeth_l1084_108479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_completes_in_40_days_l1084_108461

noncomputable section

/-- The number of days it takes q to complete the work alone -/
def q_days : ℝ := 24

/-- The total number of days the work lasted -/
def total_days : ℝ := 25

/-- The number of days p worked alone before q joined -/
def p_alone_days : ℝ := 16

/-- The total amount of work to be done -/
def total_work : ℝ := 1

/-- The rate at which p completes work per day -/
noncomputable def p_rate (x : ℝ) : ℝ := total_work / x

/-- The rate at which q completes work per day -/
noncomputable def q_rate : ℝ := total_work / q_days

/-- The theorem stating that p can complete the work alone in 40 days -/
theorem p_completes_in_40_days :
  ∃ x : ℝ, x = 40 ∧
  total_work = p_alone_days * p_rate x + (total_days - p_alone_days) * (p_rate x + q_rate) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_completes_in_40_days_l1084_108461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1084_108402

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sin (2 * x) + Real.cos (2 * x)) / Real.log (1/3)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

theorem f_monotone_decreasing (k : ℤ) :
  monotone_decreasing_on f (Set.Ioo (k * Real.pi - Real.pi/8) (k * Real.pi + Real.pi/8)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1084_108402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l1084_108429

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the general form of a line equation
def general_line_equation (a b c : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ a * x + b * y + c = 0

-- Theorem statement
theorem line_equation_through_points :
  ∀ x y : ℝ, line_through_points 1 0 0 2 x y ↔ general_line_equation 2 1 (-2) x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_points_l1084_108429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l1084_108451

/-- Given a polynomial P_n of degree n and a constant c, prove the division theorem -/
theorem polynomial_division_theorem 
  (n : ℕ) 
  (c : ℝ) 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_an_nonzero : a n ≠ 0)
  (h_bn : b n = a n)
  (h_bk : ∀ k, k < n → b k = c * b (k + 1) + a k) :
  let P_n := fun (x : ℝ) => (Finset.range (n + 1)).sum (fun i => a i * x ^ i)
  let Q := fun (x : ℝ) => (Finset.range n).sum (fun i => b (i + 1) * x ^ i)
  P_n = fun x => (x - c) * Q x + b 0 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l1084_108451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1084_108493

theorem triangle_cosine_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧ 
  b = (5/8) * a ∧ 
  A = 2 * B → 
  Real.cos A = 7/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1084_108493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1084_108414

-- Define the function g
noncomputable def g (x : ℝ) := Real.arcsin x + x^3

-- State the theorem
theorem inequality_solution_set (x : ℝ) :
  (0 < x ∧ x ≤ 1) ↔ Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0 :=
by
  sorry

-- Define the properties of g
axiom g_increasing : ∀ x y, x < y → x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → g x < g y
axiom g_odd : ∀ x, x ∈ Set.Icc (-1) 1 → g (-x) = -g x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1084_108414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_player_wins_prob_l1084_108495

/-- Represents a coin flipping game with four players -/
structure CoinGame where
  /-- The probability of flipping heads -/
  p_heads : ℝ
  /-- Assumption that the coin is fair -/
  fair_coin : p_heads = 1 / 2

/-- The probability that the third player wins the game -/
noncomputable def third_player_wins (game : CoinGame) : ℝ :=
  (game.p_heads * (1 - game.p_heads)^4) / (1 - (1 - game.p_heads)^4)

/-- Theorem stating that the probability of the third player winning is 1/31 -/
theorem third_player_wins_prob (game : CoinGame) :
    third_player_wins game = 1 / 31 := by
  sorry

#check third_player_wins_prob

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_player_wins_prob_l1084_108495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_3967149_1847234_l1084_108496

noncomputable def original_number : ℝ := 3967149.1847234

noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

theorem round_to_nearest_tenth_3967149_1847234 :
  round_to_nearest_tenth original_number = 3967149.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_3967149_1847234_l1084_108496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_is_two_l1084_108471

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The minimum sum of distances from chord endpoints to y-axis is 2 -/
theorem min_distance_sum_is_two :
  ∀ (A B : Point) (C D : Point),
    A ∈ Parabola →
    B ∈ Parabola →
    (∃ t : ℝ, A = Point.mk (t * focus.x) (t * focus.y) ∧
              B = Point.mk ((1 - t) * focus.x) ((1 - t) * focus.y)) →
    C = Point.mk 0 A.y →
    D = Point.mk 0 B.y →
    (∀ A' B' C' D' : Point,
      A' ∈ Parabola →
      B' ∈ Parabola →
      (∃ t' : ℝ, A' = Point.mk (t' * focus.x) (t' * focus.y) ∧
                 B' = Point.mk ((1 - t') * focus.x) ((1 - t') * focus.y)) →
      C' = Point.mk 0 A'.y →
      D' = Point.mk 0 B'.y →
      distance A C + distance B D ≤ distance A' C' + distance B' D') →
    distance A C + distance B D = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_is_two_l1084_108471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_l1084_108435

-- Define the complex number Z as a function of m
noncomputable def Z (m : ℝ) : ℂ := Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)

-- Theorem for Z being a real number
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -2 ∨ m = -1 := by
  sorry

-- Theorem for Z being a pure imaginary number
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ↔ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_is_real_Z_is_pure_imaginary_l1084_108435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1084_108475

theorem equation_solution : 
  ∃ t : ℝ, 6 * (3 : ℝ)^(2*t) + Real.sqrt (4 * 9 * (9 : ℝ)^t) = 90 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1084_108475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_function_l1084_108440

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

theorem domain_of_tan_function :
  ∀ x : ℝ, ¬(∃ k : ℤ, x = k * Real.pi / 2 + 5 * Real.pi / 12) ↔ 
    ∀ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_tan_function_l1084_108440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beths_average_speed_l1084_108425

/-- Represents a trip with distance and time -/
structure Trip where
  distance : ℝ
  time : ℝ

/-- Calculates the average speed of a trip -/
noncomputable def averageSpeed (trip : Trip) : ℝ :=
  trip.distance / trip.time

theorem beths_average_speed :
  let johns_speed : ℝ := 40
  let johns_time : ℝ := 0.5
  let johns_trip := Trip.mk (johns_speed * johns_time) johns_time
  let beths_trip := Trip.mk (johns_trip.distance + 5) (johns_trip.time + 1/3)
  averageSpeed beths_trip = 30 := by
    -- Proof steps would go here
    sorry

#eval "Theorem statement typecheck"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beths_average_speed_l1084_108425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_one_in_base_three_l1084_108455

theorem square_has_one_in_base_three (n : ℤ) : 
  ∃ k : ℕ, ∃ m : ℤ, n^2 = 3^k * (3*m + 1) + 3^k * r ∧ r < 3^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_has_one_in_base_three_l1084_108455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_m_range_l1084_108464

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3|
def g (x m : ℝ) : ℝ := -|x + 4| + m

-- Define the constant a
variable (a : ℝ)

-- Theorem for part (I)
theorem solution_set (h : a < 2) :
  {x : ℝ | f x + a - 2 > 0} = Set.Ioi (5 - a) ∪ Set.Iic (a + 1) :=
sorry

-- Theorem for part (II)
theorem m_range (m : ℝ) :
  (∀ x, f x > g x m) → m < 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_m_range_l1084_108464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1084_108459

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x - 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) → a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1084_108459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_properties_l1084_108408

/-- A function satisfying the given properties -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) ∧
  (∀ x : ℝ, x > 1 → f x > 0)

theorem functional_equation_properties
  (f : ℝ → ℝ)
  (h : FunctionalEquation f) :
  (f 1 = 0) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x < f y) ∧
  (f 2 = 1 →
    {x : ℝ | f (-x) + f (3 - x) ≥ -2} = {x : ℝ | x ≤ (3 - Real.sqrt 10) / 2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_properties_l1084_108408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_theorem_l1084_108481

-- Define the basic types and structures
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points A, B, C and lengths a, b, c
variable (A B C : V) (a b c : ℝ)

-- Define circles Sa, Sb, Sc
def Sa (A : V) (a : ℝ) := Metric.closedBall A a
def Sb (B : V) (b : ℝ) := Metric.closedBall B b
def Sc (C : V) (c : ℝ) := Metric.closedBall C c

-- Define the radical center
noncomputable def radical_center (Sa Sb Sc : Set V) : V := sorry

-- Define the tangent length from a point to a circle
noncomputable def tangent_length (p : V) (S : Set V) : ℝ := sorry

-- State the theorem
theorem circle_construction_theorem (A B C : V) (a b c : ℝ) :
  ∃ (O : V) (r : ℝ),
    let S := Metric.closedBall O r
    tangent_length A S = a ∧
    tangent_length B S = b ∧
    tangent_length C S = c ∧
    O = radical_center (Sa A a) (Sb B b) (Sc C c) ∧
    r = tangent_length O (Sa A a) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_theorem_l1084_108481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_gcd_l1084_108426

theorem cubic_polynomial_gcd (p q r a b c : ℤ) : 
  (∃ (P : ℤ → ℤ), P = (fun x ↦ x^3 - p*x^2 + q*x - r)) → 
  (∃ P : ℤ → ℤ, ∀ x : ℤ, P x = 0 ↔ x = a ∨ x = b ∨ x = c) →
  Int.gcd a (Int.gcd b c) = 1 →
  Int.gcd p (Int.gcd q r) = 1 := by
  intro hP hRoots hGcdRoots
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_gcd_l1084_108426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_equality_l1084_108448

def A : ℂ := -3 + 2*Complex.I
def O : ℂ := 3*Complex.I
def P : ℂ := 1 + 3*Complex.I
def S : ℂ := -2 - Complex.I

theorem complex_sum_equality : 2*A - O + 3*P + S = -5 + 9*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_equality_l1084_108448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_increasing_l1084_108423

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - Real.pi / 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem f_period_and_g_increasing (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + 2*Real.pi/3) = f ω x) :
  ω = 3/2 ∧
  ∀ k : ℤ, is_increasing (g ω) (2*Real.pi*↑k/3 + Real.pi/4) (2*Real.pi*↑k/3 + 7*Real.pi/12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_increasing_l1084_108423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_series_property_l1084_108487

/-- A series with a special property -/
structure SpecialSeries (α : Type*) [AddGroup α] where
  elements : Set α
  zero_elem : α
  zero_in_series : zero_elem ∈ elements
  equidistant_property : ∀ (x y : α), x ∈ elements → y ∈ elements → 
    ∃ (z : α), z ∈ elements ∧ z = 0 ∧ 
    (∃ d : α, x = z + d ∧ y = z + d ∨ x = z - d ∧ y = z - d)

/-- The main theorem -/
theorem special_series_property {α : Type*} [AddGroup α] (Φ : SpecialSeries α) :
  ∀ (x y : α), x ∈ Φ.elements → y ∈ Φ.elements → 
    (∃ (z : α), z ∈ Φ.elements ∧ z = 0 ∧ 
    (∃ d : α, x = z + d ∧ y = z + d ∨ x = z - d ∧ y = z - d)) →
  x + y = 0 ∨ x - y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_series_property_l1084_108487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_l1084_108401

/-- Represents a digit in our number system -/
def Digit := Fin 6

/-- Converts a base 6 number to its decimal representation -/
def toDecimal (d₂ d₁ d₀ : Digit) : ℕ :=
  d₂.val * 36 + d₁.val * 6 + d₀.val

/-- The equation BAN + AND + SAN = ASK in our number system -/
def equationHolds (B A N D S K : Digit) : Prop :=
  toDecimal B A N + toDecimal A N D + toDecimal S A N = toDecimal A S K

theorem lock_combination :
  ∃ (B A N D S K : Digit),
    B = ⟨1, by norm_num⟩ ∧ 
    A = ⟨2, by norm_num⟩ ∧ 
    N = ⟨3, by norm_num⟩ ∧ 
    D = ⟨4, by norm_num⟩ ∧ 
    S = ⟨5, by norm_num⟩ ∧
    equationHolds B A N D S K ∧
    toDecimal S A N = 523 := by
  sorry

#eval toDecimal ⟨5, by norm_num⟩ ⟨2, by norm_num⟩ ⟨3, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_l1084_108401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_n_l1084_108446

noncomputable def a : ℝ := Real.pi / 2010

noncomputable def series_term (k : ℕ) : ℝ → ℝ := λ x => 2 * (Real.cos (k^2 * x) * Real.sin (k * x))

noncomputable def series_sum (n : ℕ) : ℝ → ℝ := λ x => (Finset.range n).sum (λ k => series_term (k + 1) x)

theorem smallest_integer_n : 
  (∀ m : ℕ, m < 201 → ¬(∃ z : ℤ, series_sum m a = ↑z)) ∧ 
  (∃ z : ℤ, series_sum 201 a = ↑z) := by
  sorry

#check smallest_integer_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_n_l1084_108446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_initial_investment_l1084_108485

/-- Represents the initial investment of B in Rupees -/
def B : ℕ := sorry

/-- The total profit at the end of the year in Rupees -/
def total_profit : ℕ := 756

/-- A's share of the profit in Rupees -/
def A_profit : ℕ := 288

/-- A's initial investment in Rupees -/
def A_initial : ℕ := 3000

/-- Amount A withdraws after 8 months in Rupees -/
def A_withdraw : ℕ := 1000

/-- Amount B advances after 8 months in Rupees -/
def B_advance : ℕ := 1000

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Month when investment changes occur -/
def change_month : ℕ := 8

theorem B_initial_investment :
  (A_initial * change_month + (A_initial - A_withdraw) * (months_in_year - change_month)) * 
    (total_profit - A_profit) = 
  (B * change_month + (B + B_advance) * (months_in_year - change_month)) * A_profit ∧
  B = 4000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_initial_investment_l1084_108485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l1084_108412

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the river banks as parallel lines
def parallel_banks (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Define the bridge as perpendicular to the banks
def perpendicular_bridge (bridge : Line) (bank : Line) : Prop :=
  bridge.a * bank.a + bridge.b * bank.b = 0

-- Define the parallel translation
def parallel_translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Define the intersection point of a line and a line segment
noncomputable def intersection_point (l : Line) (p1 p2 : Point) : Point :=
  sorry

-- State the theorem
theorem shortest_path (A B M N : Point) (bank1 bank2 bridge : Line) :
  parallel_banks bank1 bank2 →
  perpendicular_bridge bridge bank1 →
  let A' := parallel_translate A (Point.mk (N.x - M.x) (N.y - M.y))
  let N_optimal := intersection_point bank1 A' B
  N = N_optimal →
  ∀ (N' : Point), N' ≠ N → 
    distance A M + distance M N + distance N B ≤ 
    distance A M + distance M N' + distance N' B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l1084_108412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_double_when_speed_halved_l1084_108450

/-- Calculates the time needed for a trip given the distance and speed -/
noncomputable def trip_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem trip_time_double_when_speed_halved (initial_speed initial_time new_speed : ℝ) 
    (h1 : initial_speed > 0)
    (h2 : new_speed > 0)
    (h3 : initial_time > 0)
    (h4 : new_speed = initial_speed / 2)
    (h5 : initial_speed = 80)
    (h6 : initial_time = 6.75)
    (h7 : new_speed = 40) :
  trip_time (initial_speed * initial_time) new_speed = 2 * initial_time := by
  sorry

def round_to_hundredth (x : Float) : Float :=
  (x * 100).round / 100

#eval round_to_hundredth (2 * 6.75) -- Expected output: 13.50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_double_when_speed_halved_l1084_108450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_over_x_for_complex_number_l1084_108410

/-- Given a complex number z = x + yi where x and y are real numbers, 
    and |z - 2| = √3, the maximum value of y/x is √3 -/
theorem max_y_over_x_for_complex_number (x y : ℝ) : 
  let z : ℂ := Complex.ofReal x + Complex.I * Complex.ofReal y
  (Complex.abs (z - 2) = Real.sqrt 3) →
  (∃ (m : ℝ), ∀ (x' y' : ℝ), 
    let z' : ℂ := Complex.ofReal x' + Complex.I * Complex.ofReal y'
    (Complex.abs (z' - 2) = Real.sqrt 3) → 
    (y' / x' ≤ m) ∧ 
    (m = Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_over_x_for_complex_number_l1084_108410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_seventh_root_l1084_108463

theorem power_of_two_product_seventh_root : (((2 : ℝ) ^ 14 * (2 : ℝ) ^ 21) ^ (1 / 7)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_seventh_root_l1084_108463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_men_fraction_is_correct_l1084_108447

/-- Represents the fraction of employees who are women -/
noncomputable def women_fraction : ℝ := 0.76

/-- Represents the fraction of employees who are married -/
noncomputable def married_fraction : ℝ := 0.60

/-- Represents the fraction of women who are married -/
noncomputable def married_women_fraction : ℝ := 0.6842

/-- Calculates the fraction of men who are single -/
noncomputable def single_men_fraction : ℝ :=
  let married_women := married_women_fraction * women_fraction
  let married_men := married_fraction - married_women
  let men_fraction := 1 - women_fraction
  (men_fraction - married_men) / men_fraction

theorem single_men_fraction_is_correct :
  ∃ ε > 0, |single_men_fraction - 0.6683| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_men_fraction_is_correct_l1084_108447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_false_statement_4_false_l1084_108404

-- Define the type for planes and lines
variable (Plane Line : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)

-- Define the parallelism and perpendicularity relations between lines and planes
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Define the parallelism and perpendicularity relations between lines
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- Define the property of a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the property of two lines intersecting
variable (lines_intersect : Line → Line → Prop)

-- Define two non-coincident planes
variable (α β : Plane)
variable (h_non_coincident : α ≠ β)

-- Statement ①
theorem statement_1 (l1 l2 m1 m2 : Line) :
  line_in_plane l1 α →
  line_in_plane l2 α →
  lines_intersect l1 l2 →
  line_in_plane m1 β →
  line_in_plane m2 β →
  line_parallel l1 m1 →
  line_parallel l2 m2 →
  parallel α β :=
by sorry

-- Statement ②
theorem statement_2 (l m : Line) :
  ¬line_in_plane l α →
  line_in_plane m α →
  line_parallel l m →
  line_parallel_plane l α :=
by sorry

-- Statement ③ (false)
theorem statement_3_false :
  ∃ (l m : Line),
    line_in_plane l α ∧
    line_in_plane l β ∧
    line_in_plane m α ∧
    line_perpendicular l m ∧
    ¬perpendicular α β :=
by sorry

-- Statement ④ (false)
theorem statement_4_false :
  ∃ (l m1 m2 : Line),
    lines_intersect m1 m2 ∧
    line_in_plane m1 α ∧
    line_in_plane m2 α ∧
    line_perpendicular l m1 ∧
    line_perpendicular l m2 ∧
    ¬line_perpendicular_plane l α :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_false_statement_4_false_l1084_108404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pond_depth_l1084_108445

/-- Represents a rectangular pond with given dimensions and volume --/
structure RectangularPond where
  length : ℝ
  width : ℝ
  volume : ℝ

/-- Calculates the depth of a rectangular pond --/
noncomputable def pondDepth (p : RectangularPond) : ℝ :=
  p.volume / (p.length * p.width)

/-- Theorem stating that the depth of the specific pond is 5 meters --/
theorem specific_pond_depth :
  let p : RectangularPond := { length := 20, width := 15, volume := 1500 }
  pondDepth p = 5 := by
  -- Unfold the definition of pondDepth
  unfold pondDepth
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pond_depth_l1084_108445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nigella_earnings_l1084_108415

def base_salary : ℚ := 3000
def commission_rate : ℚ := 2 / 100
def num_houses : ℕ := 3
def house_a_cost : ℚ := 60000
def house_b_cost : ℚ := 3 * house_a_cost
def house_c_cost : ℚ := 2 * house_a_cost - 110000

def total_sales : ℚ := house_a_cost + house_b_cost + house_c_cost
def commission : ℚ := commission_rate * total_sales
def total_earnings : ℚ := base_salary + commission

theorem nigella_earnings : total_earnings = 8000 := by
  -- Expand definitions
  unfold total_earnings
  unfold commission
  unfold total_sales
  unfold house_b_cost
  unfold house_c_cost
  -- Simplify expressions
  simp [base_salary, commission_rate, house_a_cost]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nigella_earnings_l1084_108415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1084_108421

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define a point on the ellipse
def P (a b : ℝ) := {p : ℝ × ℝ // ellipse a b p.1 p.2}

-- Define the angle F₁PF₂
noncomputable def angle_F₁PF₂ (a b : ℝ) (p : P a b) : ℝ := sorry

-- Define the radius of the inscribed circle
noncomputable def radius_inscribed (a b : ℝ) (p : P a b) : ℝ := sorry

-- Define the ratio of areas of circumscribed to inscribed circles
noncomputable def area_ratio (a b : ℝ) (p : P a b) : ℝ := sorry

-- Theorem statement
theorem ellipse_theorem (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ p : P a b,
    angle_F₁PF₂ a b p = 30 * π / 180 ∧ 
    radius_inscribed a b p = 2 - Real.sqrt 3) →
  (a = 2 ∧ b = Real.sqrt 3 ∧ 
   ∀ q : P a b, area_ratio a b q ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l1084_108421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l1084_108453

/-- The displacement function of a particle moving along a straight line -/
noncomputable def y (t : ℝ) : ℝ := 3 * t^2 + 4

/-- The instantaneous velocity of the particle at time t -/
noncomputable def v (t : ℝ) : ℝ := deriv y t

/-- Theorem stating that the instantaneous velocity at t = 2 is 12 m/s -/
theorem instantaneous_velocity_at_2 : v 2 = 12 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_l1084_108453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_conic_is_circle_l1084_108499

/-- Represents a conic section equation in the form (x-h)^2 + (k(y-v))^2 = r^2 -/
structure ConicSection where
  h : ℚ  -- x-coordinate of the center
  k : ℚ  -- scaling factor for y
  v : ℚ  -- y-coordinate of the center (scaled)
  r : ℚ  -- radius or semi-major axis

/-- Defines when a conic section is considered a circle -/
def is_circle (c : ConicSection) : Prop :=
  c.k^2 = 1 ∧ c.r > 0

/-- The specific conic section from the problem -/
def problem_conic : ConicSection :=
  { h := 2
  , k := 3
  , v := -1/3
  , r := 3 }

/-- Theorem stating that the problem's conic section is a circle -/
theorem problem_conic_is_circle : is_circle problem_conic := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_conic_is_circle_l1084_108499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1084_108418

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) that intersects
    with the line y = 2x, its eccentricity e is greater than √5. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_intersect : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 2*x) :
  Real.sqrt (1 + (b/a)^2) > Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1084_108418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_two_distinct_real_roots_l1084_108467

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) : 
  (b^2 - 4*a*c > 0) → ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 := by
  sorry

theorem two_distinct_real_roots : 
  ∃ x y : ℝ, x ≠ y ∧ 2*x^2 + 3*x - 4 = 0 ∧ 2*y^2 + 3*y - 4 = 0 := by
  apply quadratic_equation_roots 2 3 (-4) (by norm_num)
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_two_distinct_real_roots_l1084_108467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_l1084_108420

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x

-- State that f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Define the integral bounds
def a : ℝ := -3
def b : ℝ := 3

-- State that the interval is symmetric about the origin
axiom interval_symmetric : a = -b

-- Theorem statement
theorem integral_f_zero : ∫ x in a..b, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_l1084_108420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_area_ratio_probability_is_correct_l1084_108443

/-- The probability of two randomly chosen numbers in [0, 1] satisfying both sum ≤ 1 and product ≤ 2/9 -/
def probability_sum_and_product (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x + y ≤ 1 ∧ x * y ≤ 2/9

/-- The area of the region defined by the inequalities -/
noncomputable def area_of_region : ℝ := 0.467

/-- The total area of the unit square -/
def total_area : ℝ := 1

/-- Theorem stating that the probability is equal to the ratio of the areas -/
theorem probability_equals_area_ratio :
  (area_of_region / total_area) = 0.467 := by
  sorry

/-- Theorem stating that the calculated probability is correct -/
theorem probability_is_correct :
  ∀ x y : ℝ, probability_sum_and_product x y ↔ (x, y) ∈ {(x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x + y ≤ 1 ∧ x * y ≤ 2/9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_area_ratio_probability_is_correct_l1084_108443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zuminglish_12_letter_words_l1084_108416

/-- Represents the number of n-letter words ending in two consonants -/
def a : ℕ → ℕ := sorry

/-- Represents the number of n-letter words ending in consonant-vowel -/
def b : ℕ → ℕ := sorry

/-- Represents the number of n-letter words ending in vowel-consonant -/
def c : ℕ → ℕ := sorry

/-- Represents the number of n-letter words ending in consonant-vowel-consonant -/
def d : ℕ → ℕ := sorry

/-- The recursive relation for a -/
axiom a_rec : ∀ n : ℕ, a (n + 1) = 2 * (a n + c n + d n)

/-- The recursive relation for b -/
axiom b_rec : ∀ n : ℕ, b (n + 1) = a n

/-- The recursive relation for c -/
axiom c_rec : ∀ n : ℕ, c (n + 1) = 2 * b n

/-- The recursive relation for d -/
axiom d_rec : ∀ n : ℕ, d (n + 1) = 2 * c n

/-- Initial values for 2-letter words -/
axiom initial_values : a 2 = 4 ∧ b 2 = 2 ∧ c 2 = 2 ∧ d 2 = 0

/-- The main theorem stating the result for 12-letter words -/
theorem zuminglish_12_letter_words : (a 12 + b 12 + c 12 + d 12) % 1000 = 382 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zuminglish_12_letter_words_l1084_108416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_twelve_l1084_108437

/-- Sequence b defined recursively -/
def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | n+2 => (1/2) * b (n+1) + (1/6) * b n

/-- Sum of the infinite series -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the infinite series equals 12 -/
theorem series_sum_equals_twelve : seriesSum = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_twelve_l1084_108437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_formula_l1084_108417

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem gauss_formula : frac 3.8 + frac (-1.7) - frac 1 = 1.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gauss_formula_l1084_108417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_always_constant_l1084_108431

noncomputable def line (x y : ℝ) : Prop := y = 4 * x - 3

noncomputable def vector_on_line (v : ℝ × ℝ) : Prop :=
  line v.1 v.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  let scalar := dot_product / magnitude_squared
  (scalar * w.1, scalar * w.2)

theorem projection_always_constant (w : ℝ × ℝ) :
  ∀ v : ℝ × ℝ, vector_on_line v → projection v w = (12/17, -3/17) :=
by
  sorry

#check projection_always_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_always_constant_l1084_108431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1084_108444

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 4

-- Define the distance between foci
noncomputable def distance_between_foci (eq : (ℝ → ℝ → Prop)) : ℝ :=
  3 * Real.sqrt 3

-- Theorem statement
theorem ellipse_foci_distance :
  distance_between_foci ellipse_equation = 3 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1084_108444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_process_terminates_l1084_108424

/-- Represents a list of positive integers -/
def IntegerList := List Nat

/-- Represents a single replacement step -/
inductive ReplacementStep
  | DecreaseLeft : ReplacementStep  -- represents (x-1, x)
  | IncreaseRight : ReplacementStep -- represents (y+1, x)

/-- Defines the condition for a valid replacement -/
def canReplace (list : IntegerList) (i : Nat) : Prop :=
  i + 1 < list.length ∧ list.get? i > list.get? (i+1)

/-- Applies a single replacement step to the list -/
def applyReplacement (list : IntegerList) (i : Nat) (step : ReplacementStep) : IntegerList :=
  sorry -- Implementation details omitted

/-- Theorem stating that the replacement process must terminate -/
theorem replacement_process_terminates (initial_list : IntegerList) :
  ∃ (n : Nat), ∀ (sequence : Nat → IntegerList),
    (sequence 0 = initial_list) →
    (∀ k, ∃ i step, canReplace (sequence k) i ∧ 
      sequence (k+1) = applyReplacement (sequence k) i step) →
    (∃ m ≤ n, ¬∃ i, canReplace (sequence m) i) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_process_terminates_l1084_108424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_minus_cones_volume_l1084_108484

/-- The volume of a cylinder not occupied by two congruent cones -/
theorem cylinder_minus_cones_volume 
  (r : ℝ) (h_cylinder : ℝ) (h_cone : ℝ)
  (hr : r = 10)
  (hh_cylinder : h_cylinder = 40)
  (hh_cone : h_cone = 15) :
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 3000 * π := by
  sorry

#check cylinder_minus_cones_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_minus_cones_volume_l1084_108484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_in_triangle_l1084_108439

theorem minimum_distance_in_triangle :
  ∀ (A B C E : ℝ × ℝ),
    let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
    d A B = 3 →
    d A C = 4 →
    d B C = Real.sqrt 13 →
    (∀ F : ℝ × ℝ, 2 * (d E A + d E B + d E C) ≤ 2 * (d F A + d F B + d F C)) →
    2 * (d E A + d E B + d E C) = 2 * Real.sqrt 37 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_in_triangle_l1084_108439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_from_ellipse_focus_l1084_108434

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the left focus of the ellipse
noncomputable def left_focus : ℝ × ℝ := (-Real.sqrt 5, 0)

-- Define the center of the ellipse
def ellipse_center : ℝ × ℝ := (0, 0)

-- Define the parabola
noncomputable def parabola (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 5 * x

-- Define the focus of the parabola (same as left focus of ellipse)
noncomputable def parabola_focus : ℝ × ℝ := left_focus

-- Define the vertex of the parabola (same as center of ellipse)
def parabola_vertex : ℝ × ℝ := ellipse_center

-- Theorem statement
theorem parabola_from_ellipse_focus :
  ∀ (x y : ℝ),
  (ellipse x y ∧
   (left_focus = parabola_focus) ∧
   (ellipse_center = parabola_vertex)) →
  parabola x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_from_ellipse_focus_l1084_108434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1084_108433

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem setup
structure GeometricSetup where
  circle : Circle
  line_a : Line
  line_b : Line
  line_i : Line

-- Define the condition that lines a and b intersect the circle
def intersects_circle (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), (y = l.slope * x + l.intercept) ∧ 
  ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

-- Define a parallel line
def parallel_line (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define the equal segment condition
def equal_segment_condition (l1 l2 : Line) (c : Circle) (p : Line) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ((x1 - c.center.1)^2 + (y1 - c.center.2)^2 = c.radius^2) ∧
    ((x2 - c.center.1)^2 + (y2 - c.center.2)^2 = c.radius^2) ∧
    (y1 = p.slope * x1 + p.intercept) ∧
    (y2 = p.slope * x2 + p.intercept) ∧
    ((x1 - (y1 - l1.intercept) / l1.slope)^2 + (y1 - y1)^2 =
     (x2 - (y2 - l2.intercept) / l2.slope)^2 + (y2 - y2)^2)

-- Theorem statement
theorem solution_count (setup : GeometricSetup) 
  (h1 : intersects_circle setup.line_a setup.circle)
  (h2 : intersects_circle setup.line_b setup.circle) :
  ∃ (n : ℕ), n ∈ ({0, 1, 2, 3} : Set ℕ) ∧
  (∃ (lines : Finset Line), lines.card = n ∧
    ∀ l ∈ lines, parallel_line l setup.line_i ∧
    equal_segment_condition setup.line_a setup.line_b setup.circle l) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1084_108433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_quadrilateral_area_l1084_108462

-- Define the hyperbola
def hyperbola : ℝ × ℝ → Prop := λ p => p.1^2 - p.2^2/3 = 1

-- Define the foci
def is_focus (F : ℝ × ℝ) (C : (ℝ × ℝ) → Prop) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ F.1^2 - F.2^2 = a^2 ∧ 
  ∀ (P : ℝ × ℝ), C P → |P.1 - F.1| - |P.2 - F.2| = 2*a

-- Define symmetry with respect to origin
def symmetric_wrt_origin (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (P F₁ Q F₂ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem hyperbola_quadrilateral_area
  (F₁ F₂ P Q : ℝ × ℝ)
  (h_F₁ : is_focus F₁ hyperbola)
  (h_F₂ : is_focus F₂ hyperbola)
  (h_P : hyperbola P)
  (h_Q : hyperbola Q)
  (h_sym : symmetric_wrt_origin P Q)
  (h_angle : angle P F₂ Q = 2*π/3)  -- 120° in radians
  : area_quadrilateral P F₁ Q F₂ = 6 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_quadrilateral_area_l1084_108462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_three_odd_l1084_108491

noncomputable section

-- Define the function f
def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (Real.pi / 2 * x + φ)

-- State the theorem
theorem f_minus_three_odd (A : ℝ) (φ : ℝ) (h1 : A > 0) (h2 : f A φ 1 = 0) :
  ∀ x, f A φ (x - 3) = -f A φ (-x + 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_three_odd_l1084_108491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_f_max_value_l1084_108442

/-- The function f(x) defined as sin(2x)cos(π/5) - cos(2x)sin(π/5) -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (Real.pi / 5) - Real.cos (2 * x) * Real.sin (Real.pi / 5)

/-- The smallest positive period of f(x) is π -/
theorem f_period : ∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x := by
  sorry

/-- The axis of symmetry of f(x) is x = 7π/20 + kπ/2, where k ∈ ℤ -/
theorem f_symmetry_axis : ∀ (k : ℤ), ∀ (x : ℝ), 
  f (7 * Real.pi / 20 + k * Real.pi / 2 + x) = f (7 * Real.pi / 20 + k * Real.pi / 2 - x) := by
  sorry

/-- The maximum value of f(x) on the interval [0, π/2] is 1 -/
theorem f_max_value : ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_f_symmetry_axis_f_max_value_l1084_108442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1084_108476

noncomputable section

open Real

/-- Helper definition for IsTriangle -/
def IsTriangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

/-- Helper definition for OppositeAngle -/
def OppositeAngle (side : ℝ) (angle : ℝ) (other1 other2 : ℝ) : Prop :=
  side^2 = other1^2 + other2^2 - 2 * other1 * other2 * cos angle

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC exists
  IsTriangle A B C →
  -- Sides a, b, c are opposite to angles A, B, C
  OppositeAngle a A B C ∧ OppositeAngle b B A C ∧ OppositeAngle c C A B →
  -- Given equation
  2 * cos (2 * A) + 4 * cos (B + C) + 3 = 0 →
  -- Given conditions
  a = sqrt 3 →
  b + c = 3 →
  -- Conclusions
  A = π / 3 ∧ ((b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1084_108476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_l1084_108405

-- Define the fixed points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the distance function as noncomputable
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the theorem
theorem apollonius_circle (x y : ℝ) :
  (distance (x, y) A) / (distance (x, y) B) = 2 ↔ (x - 5)^2 + y^2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_l1084_108405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_l1084_108432

theorem coin_distribution : 
  (Nat.choose (25 + 4 - 1) (4 - 1) : ℕ) = 3276 := by
  -- number of identical coins: 25
  -- number of schoolchildren: 4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_l1084_108432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_currents_l1084_108400

/-- Represents a Bunsen cell -/
structure BunsenCell where
  emf : ℝ
  internalResistance : ℝ

/-- Represents a wire -/
structure Wire where
  resistance : ℝ

/-- Represents the circuit configuration -/
structure Circuit where
  cells : List BunsenCell
  wireA : Wire
  wireB : Wire

noncomputable def totalEMF (c : Circuit) : ℝ :=
  c.cells.map (λ cell => cell.emf) |> List.sum

noncomputable def totalInternalResistance (c : Circuit) : ℝ :=
  c.cells.map (λ cell => cell.internalResistance) |> List.sum

noncomputable def parallelResistance (r1 r2 : ℝ) : ℝ :=
  1 / (1 / r1 + 1 / r2)

noncomputable def totalResistance (c : Circuit) : ℝ :=
  totalInternalResistance c + parallelResistance c.wireA.resistance c.wireB.resistance

noncomputable def totalCurrent (c : Circuit) : ℝ :=
  totalEMF c / totalResistance c

noncomputable def currentInWireA (c : Circuit) (totalI : ℝ) : ℝ :=
  totalI * c.wireB.resistance / (c.wireA.resistance + c.wireB.resistance)

noncomputable def currentInWireB (c : Circuit) (totalI : ℝ) : ℝ :=
  totalI * c.wireA.resistance / (c.wireA.resistance + c.wireB.resistance)

theorem circuit_currents (c : Circuit) 
  (h1 : c.cells.length = 6)
  (h2 : ∀ cell ∈ c.cells, cell.emf = 1.8 ∧ cell.internalResistance = 0.2)
  (h3 : c.wireA.resistance = 6)
  (h4 : c.wireB.resistance = 3) :
  let totalI := totalCurrent c
  abs (totalI - 3.38) < 0.01 ∧ 
  abs (currentInWireA c totalI - 1.13) < 0.01 ∧ 
  abs (currentInWireB c totalI - 2.25) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_currents_l1084_108400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_theorem_l1084_108483

/-- Represents the ladder scenario -/
structure LadderScenario where
  length : ℝ
  initial_base_distance : ℝ
  top_slip : ℝ

/-- Calculates the distance the foot of the ladder slides -/
noncomputable def foot_slide (scenario : LadderScenario) : ℝ :=
  let initial_height := (scenario.length^2 - scenario.initial_base_distance^2).sqrt
  let new_height := initial_height - scenario.top_slip
  (scenario.length^2 - new_height^2).sqrt - scenario.initial_base_distance

/-- The main theorem stating the result of the ladder problem -/
theorem ladder_slide_theorem (ε : ℝ) (hε : ε > 0) : 
  ∃ (scenario : LadderScenario), 
    scenario.length = 30 ∧ 
    scenario.initial_base_distance = 8 ∧ 
    scenario.top_slip = 3 ∧ 
    |foot_slide scenario - 7.1| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ladder_slide_theorem_l1084_108483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_production_is_7x_l1084_108494

/-- The number of boxes produced by machines A and B working simultaneously for 14 minutes -/
noncomputable def simultaneous_production (x : ℝ) : ℝ :=
  let rate_A := x / 10
  let rate_B := 2 * x / 5
  (rate_A + rate_B) * 14

/-- Theorem stating that the simultaneous production of machines A and B for 14 minutes is 7x boxes -/
theorem simultaneous_production_is_7x (x : ℝ) : simultaneous_production x = 7 * x := by
  unfold simultaneous_production
  -- Expand the definition and simplify
  simp [mul_add, mul_div_assoc, mul_comm, mul_assoc]
  -- Perform algebraic manipulations
  ring

#check simultaneous_production_is_7x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_production_is_7x_l1084_108494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_condition_l1084_108419

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4
def circle_C2 (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define the centers and radii
def center_C1 : ℝ × ℝ := (0, 2)
def center_C2 : ℝ × ℝ := (1, -1)
def radius_C1 : ℝ := 2
noncomputable def radius_C2 : ℝ := Real.sqrt 2

-- Theorem stating that C1 and C2 intersect
theorem circles_intersect :
  ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y :=
by
  sorry

-- Additional theorem to show the relationship between radii and distance
theorem circles_intersect_condition :
  (radius_C1 + radius_C2)^2 > (center_C1.1 - center_C2.1)^2 + (center_C1.2 - center_C2.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_condition_l1084_108419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_and_solution_set_l1084_108482

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m * Real.exp (Real.log 3 * abs (x - 1))

/-- Proposition stating the main result -/
theorem unique_zero_implies_m_and_solution_set (m : ℝ) :
  (∃! x, f m x = 0) →
  (m = 1 ∧ Set.Ioo 0 2 = {x | f m x < 3 * m}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_implies_m_and_solution_set_l1084_108482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1084_108422

def sequence_a : ℕ → ℚ
| 0 => 1
| n + 1 => 3 * sequence_a n / (3 + sequence_a n)

theorem sequence_a_formula (n : ℕ) : sequence_a n = 3 / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l1084_108422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1084_108490

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.05
  let time : ℕ := 7
  round_to_nearest (compound_interest principal rate time) = 11257 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1084_108490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leah_earnings_l1084_108427

/-- Represents Leah's initial earnings in dollars -/
def initial_earnings : ℚ := 28

/-- Represents the amount spent on a milkshake -/
def milkshake_cost : ℚ := (1 / 7) * initial_earnings

/-- Represents the amount remaining after buying the milkshake -/
def remaining_after_milkshake : ℚ := initial_earnings - milkshake_cost

/-- Represents the amount put into savings -/
def savings_amount : ℚ := (1 / 2) * remaining_after_milkshake

/-- Represents the amount left in the wallet -/
def wallet_amount : ℚ := remaining_after_milkshake - savings_amount

/-- Represents the amount of money shredded by the dog -/
def shredded_amount : ℚ := 11

/-- Represents the amount left intact in the wallet -/
def intact_amount : ℚ := 1

theorem leah_earnings : initial_earnings = 28 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leah_earnings_l1084_108427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1084_108478

theorem problem_statement :
  (∀ x : ℝ, (2 : ℝ)^x + (2 : ℝ)^(-x) > 1) ∧ (∀ x : ℝ, Real.log (x^2 + 2*x + 3) > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1084_108478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_sum_bound_l1084_108413

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

/-- The smallest upper bound for the sum of fractional parts -/
noncomputable def M : ℝ := 2 + 2024 / 2025

/-- Theorem stating the bound on the sum of fractional parts and its tightness -/
theorem fractional_parts_sum_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 2024) :
  frac a + frac b + frac c ≤ M ∧ 
  ∀ ε > 0, ∃ a' b' c' : ℝ, 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ a' * b' * c' = 2024 ∧ 
    frac a' + frac b' + frac c' > M - ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractional_parts_sum_bound_l1084_108413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rancher_problem_l1084_108449

/-- Represents the rancher's problem of buying steers and cows --/
theorem rancher_problem (budget steer_cost cow_cost min_steers : ℕ) :
  ∃ (s c : ℕ),
    s * steer_cost + c * cow_cost = budget ∧
    s ≥ min_steers ∧
    c ≥ s / 2 ∧
    s = 12 ∧ c = 11 ∧
    ∀ (s' c' : ℕ),
      s' * steer_cost + c' * cow_cost = budget →
      s' ≥ min_steers →
      c' ≥ s' / 2 →
      s + c ≥ s' + c' := by
  sorry

#check rancher_problem 800 30 40 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rancher_problem_l1084_108449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_mass_from_unbalanced_scale_l1084_108477

/-- Represents a balance scale with unequal arm lengths -/
structure UnbalancedScale where
  left_arm : ℝ
  right_arm : ℝ
  left_arm_pos : 0 < left_arm
  right_arm_pos : 0 < right_arm
  unequal_arms : left_arm ≠ right_arm

/-- The accurate mass of an object given its measured masses on both pans of an unbalanced scale -/
noncomputable def accurateMass (scale : UnbalancedScale) (m₁ m₂ : ℝ) : ℝ :=
  Real.sqrt (m₁ * m₂)

theorem accurate_mass_from_unbalanced_scale (scale : UnbalancedScale) (m₁ m₂ : ℝ) 
    (h₁ : m₁ > 0) (h₂ : m₂ > 0) :
    let x := accurateMass scale m₁ m₂
    scale.left_arm * x = scale.right_arm * m₁ ∧ 
    scale.right_arm * x = scale.left_arm * m₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_mass_from_unbalanced_scale_l1084_108477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcenter_vector_l1084_108473

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 2 ∧ AC = 1 ∧ BC = Real.sqrt 7

-- Define the circumcenter
def Circumcenter (A B C O : ℝ × ℝ) : Prop :=
  let OA := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  let OB := Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2)
  let OC := Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2)
  OA = OB ∧ OB = OC

-- Define the vector equation
def VectorEquation (A B C O : ℝ × ℝ) (lambda mu : ℝ) : Prop :=
  O.1 - A.1 = lambda * (B.1 - A.1) + mu * (C.1 - A.1) ∧
  O.2 - A.2 = lambda * (B.2 - A.2) + mu * (C.2 - A.2)

-- The main theorem
theorem triangle_circumcenter_vector (A B C O : ℝ × ℝ) (lambda mu : ℝ) :
  Triangle A B C →
  Circumcenter A B C O →
  VectorEquation A B C O lambda mu →
  lambda + mu = 13/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcenter_vector_l1084_108473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1084_108428

/-- The length of a platform crossed by a train -/
noncomputable def platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  train_speed_mps * crossing_time - train_length

/-- Theorem stating the length of the platform -/
theorem platform_length_calculation :
  platform_length 450 108 25 = 300 := by
  -- Unfold the definition of platform_length
  unfold platform_length
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1084_108428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l1084_108409

theorem inequalities_proof (a b c : ℝ) 
  (h1 : a < 0) (h2 : b > 0) (h3 : c > 0) (h4 : a < b) (h5 : b < c) :
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a + c < b + c) ∧ 
  (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l1084_108409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1084_108430

/-- The hyperbola C: x²/4 - y² = 1 --/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The right branch of the hyperbola --/
def right_branch (x y : ℝ) : Prop := hyperbola x y ∧ x > 0

/-- The foci of the hyperbola --/
noncomputable def focus_left : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def focus_right : ℝ × ℝ := (Real.sqrt 5, 0)

/-- A point P on the right branch of the hyperbola --/
def point_on_hyperbola (x₀ y₀ : ℝ) : Prop := right_branch x₀ y₀ ∧ y₀ ≥ 1

/-- The x-coordinate of the intersection of the angle bisector with the x-axis --/
noncomputable def m (x₀ : ℝ) : ℝ := 4 / x₀

/-- Placeholder for triangle area check --/
def is_triangle_area (S : ℝ) : Prop := sorry

/-- The statement to be proved --/
theorem hyperbola_theorem (x₀ y₀ : ℝ) (h : point_on_hyperbola x₀ y₀) :
  (0 < m x₀ ∧ m x₀ ≤ Real.sqrt 2) ∧
  (∃ (S : ℝ), S = 4 * Real.sqrt 30 ∧ ∀ (S' : ℝ), is_triangle_area S' → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1084_108430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_specific_value_l1084_108472

/-- The diamond operation for real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + x*y)

/-- Theorem stating the result of the specific diamond operation -/
theorem diamond_specific_value : 
  diamond (diamond 6 15) (diamond (-15) (-6)) = 351 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_specific_value_l1084_108472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ratio_inverse_proportion_range_l1084_108466

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Part 1
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) (h2 : f x₂ = y₂) (h3 : x₁ / x₂ = 2) :
  y₁ / y₂ = 1 / 2 := by
  sorry

-- Part 2
theorem inverse_proportion_range (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) (h2 : f x₂ = y₂) (h3 : x₁ = x₂ + 2) (h4 : y₁ = 3 * y₂) :
  ∀ x > x₁ + x₂, f x < -3/2 ∨ f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ratio_inverse_proportion_range_l1084_108466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_of_square_l1084_108456

/-- Given a rectangle ABCD and a square DCEF that share side DC, prove that the diagonal DE of the square is 8 units long when AB = 8 and AD = 4. -/
theorem diagonal_of_square (A B C D E F : ℝ × ℝ) : 
  let rectangle_area := (B.1 - A.1) * (D.2 - A.2)
  let square_side := (E.1 - D.1)
  (B.1 - A.1 = 8) →  -- AB = 8
  (D.2 - A.2 = 4) →  -- AD = 4
  (rectangle_area = square_side^2) →  -- Areas are equal
  (E.1 - D.1 = F.2 - D.2) →  -- DCEF is a square
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 8  -- DE = 8
  := by
    intro h1 h2 h3 h4
    -- The proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_of_square_l1084_108456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_l1084_108474

-- Define polynomials f, g, and h
variable (f g h : Polynomial ℝ)

-- Define the relationship between h, f, and g
def h_def (f g h : Polynomial ℝ) : Prop := 
  ∀ x, h.eval x = (f.comp g).eval x + x * g.eval x

-- Theorem statement
theorem degree_of_g (f g h : Polynomial ℝ) : 
  h_def f g h →           -- Condition 2
  h.degree = 7 →          -- Condition 3
  f.degree = 3 →          -- Condition 4
  g.degree = 6 :=         -- Conclusion
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_g_l1084_108474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_weight_calculation_l1084_108460

/-- The weight in grams that a dishonest shopkeeper uses for a kg, given their gain percentage -/
def shopkeeperWeight (gainPercent : Float) : Float :=
  1000 - (gainPercent / 100) * 1000

theorem shopkeeper_weight_calculation :
  let actualGainPercent : Float := 2.0408163265306145
  let calculatedWeight : Float := shopkeeperWeight actualGainPercent
  (calculatedWeight - 979.59).abs < 0.01 := by sorry

#eval shopkeeperWeight 2.0408163265306145

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_weight_calculation_l1084_108460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l1084_108407

/-- The number of solutions to sin x + a = b x -/
def num_solutions (a b : ℝ) : ℕ := sorry

theorem system_has_solution (a b : ℝ) 
  (h : num_solutions a b = 2) : 
  ∃ x : ℝ, Real.sin x + a = b * x ∧ Real.cos x = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_solution_l1084_108407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_rounded_l1084_108411

def fall_03_04_headcount : ℕ := 11500
def fall_04_05_headcount : ℕ := 11600
def fall_05_06_headcount : ℕ := 11300

def average_headcount : ℚ := (fall_03_04_headcount + fall_04_05_headcount + fall_05_06_headcount) / 3

theorem average_headcount_rounded :
  round average_headcount = 11467 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_headcount_rounded_l1084_108411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_property_l1084_108489

variable (N : Matrix (Fin 2) (Fin 2) ℝ)

def v1 : Fin 2 → ℝ := ![1, 2]
def v2 : Fin 2 → ℝ := ![4, -1]
def v3 : Fin 2 → ℝ := ![6, 3]

theorem matrix_N_property (h1 : N.mulVec v1 = ![(-2), 4])
                          (h2 : N.mulVec v2 = ![3, -6]) :
  N.mulVec v3 = ![(-1), 2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_property_l1084_108489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1084_108469

/-- The present value of a machine given its future value and depletion rate. -/
noncomputable def present_value (future_value : ℝ) (depletion_rate : ℝ) (years : ℕ) : ℝ :=
  future_value / ((1 - depletion_rate) ^ years)

/-- Theorem: The present value of a machine with a 10% annual depletion rate
    and a value of $891 after 2 years is $1100. -/
theorem machine_present_value :
  let future_value : ℝ := 891
  let depletion_rate : ℝ := 0.1
  let years : ℕ := 2
  present_value future_value depletion_rate years = 1100 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_present_value_l1084_108469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_f_equiv_g_l1084_108454

/-- The cosine function with a phase shift -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 6)

/-- The sine function with a horizontal shift -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * (x + Real.pi / 6))

/-- Trigonometric identity for cosine in terms of sine -/
theorem cos_to_sin (α β : ℝ) : Real.cos (α - β) = Real.sin (α - β + Real.pi / 2) := by sorry

/-- Theorem stating that f and g are equivalent functions -/
theorem f_equiv_g : ∀ x, f x = g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_to_sin_f_equiv_g_l1084_108454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_l1084_108458

-- Define the investment amounts for P and Q
variable (P_i Q_i : ℝ)

-- Define the investment durations for P and Q
noncomputable def P_duration : ℝ := 5
noncomputable def Q_duration : ℝ := 14

-- Define the profit ratio
noncomputable def profit_ratio : ℝ := 7 / 14

-- Theorem statement
theorem investment_ratio (h : P_i * P_duration / (Q_i * Q_duration) = profit_ratio) :
  P_i / Q_i = 7 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_l1084_108458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_unchanged_l1084_108486

/-- Represents the percentage of profit -/
noncomputable def profit_percentage : ℝ := 32

/-- Represents the discount percentage -/
noncomputable def discount_percentage : ℝ := 4

/-- Calculates the selling price given the cost price and profit percentage -/
noncomputable def selling_price (cost_price : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

/-- Calculates the discounted price given the original price and discount percentage -/
noncomputable def discounted_price (original_price : ℝ) : ℝ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem: The profit percentage remains the same whether a discount is offered or not -/
theorem profit_percentage_unchanged (cost_price : ℝ) (cost_price_positive : cost_price > 0) :
  let sp := selling_price cost_price
  let dsp := discounted_price sp
  (dsp - cost_price) / cost_price * 100 = profit_percentage := by
  sorry

#check profit_percentage_unchanged

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_unchanged_l1084_108486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_of_differential_equation_l1084_108497

-- Define the function y(x) as noncomputable
noncomputable def y (x : ℝ) : ℝ := (x^3 / 3) - x + (14 / 3)

-- State the theorem
theorem particular_solution_of_differential_equation :
  -- The derivative of y with respect to x is equal to x^2 - 1
  (∀ x : ℝ, deriv y x = x^2 - 1) ∧
  -- The function y satisfies the initial condition y(1) = 4
  y 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_of_differential_equation_l1084_108497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1084_108436

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1084_108436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planar_graphs_l1084_108438

-- Define the graph types
def Graph := Type

def CompleteGraph (n : ℕ) : Graph := sorry

def CompleteBipartiteGraph (m n : ℕ) : Graph := sorry

-- Define planarity
def IsPlanar (G : Graph) : Prop := sorry

-- State the theorem
theorem planar_graphs :
  IsPlanar (CompleteGraph 4) ∧
  ¬IsPlanar (CompleteBipartiteGraph 3 3) ∧
  ¬IsPlanar (CompleteGraph 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planar_graphs_l1084_108438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_representation_l1084_108457

/-- Represents a financial transaction amount -/
def TransactionAmount := ℤ

/-- Indicates whether a transaction is a gain or loss -/
inductive TransactionType
| Gain
| Loss

/-- Represents a financial transaction -/
structure Transaction where
  amount : ℕ
  type : TransactionType

/-- Converts a Transaction to a TransactionAmount -/
def transactionToAmount (t : Transaction) : TransactionAmount :=
  match t.type with
  | TransactionType.Gain => (t.amount : ℤ)
  | TransactionType.Loss => -(t.amount : ℤ)

theorem loss_representation (loss_amount : ℕ) :
  transactionToAmount { amount := loss_amount, type := TransactionType.Loss } = -(loss_amount : ℤ) := by
  sorry

#eval transactionToAmount { amount := 300, type := TransactionType.Loss }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_representation_l1084_108457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_bought_l1084_108452

/-- The number of articles bought at the cost price -/
def N : ℕ := 50

/-- The cost price of one article -/
def C : ℝ := sorry

/-- The gain percent as a fraction -/
def gain_percent : ℚ := 2/3

theorem articles_bought : N = 50 := by
  have h1 : N * C = 30 * (C * (1 + gain_percent)) := by sorry
  have h2 : gain_percent = 2/3 := rfl
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_bought_l1084_108452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_symmetry_l1084_108470

/-- A face of a polyhedron -/
structure Face where
  has_center_of_symmetry : Bool

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  is_convex : Bool

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point about a center -/
def reflect (center : Point) (point : Point) : Point :=
  { x := 2 * center.x - point.x,
    y := 2 * center.y - point.y,
    z := 2 * center.z - point.z }

/-- Definition of a point belonging to a polyhedron -/
def Point.belongsTo (p : Point) (poly : ConvexPolyhedron) : Prop :=
  sorry -- Define this based on your specific requirements

/-- Definition of center of symmetry for a polyhedron -/
def has_center_of_symmetry (p : ConvexPolyhedron) : Prop :=
  ∃ (center : Point), ∀ (point : Point), point.belongsTo p → (reflect center point).belongsTo p

/-- Theorem: If each face of a convex polyhedron has a center of symmetry, 
    then the polyhedron itself has a center of symmetry -/
theorem convex_polyhedron_symmetry 
  (p : ConvexPolyhedron) 
  (h : p.is_convex = true) 
  (h_faces : ∀ f ∈ p.faces, f.has_center_of_symmetry = true) : 
  has_center_of_symmetry p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_symmetry_l1084_108470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l1084_108488

noncomputable def shift_sin (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)
noncomputable def original_sin (x : ℝ) := 4 * Real.sin (2 * x)

noncomputable def cos_symmetry (φ : ℝ) (x : ℝ) := 4 * Real.cos (2 * x + φ)

noncomputable def tan_function (x : ℝ) := (4 * Real.tan x) / (1 - Real.tan x ^ 2)

noncomputable def simplify_expression := Real.sqrt (1 + Real.sin 2) - Real.sqrt (1 - Real.sin 2)

theorem exactly_two_correct : 
  (∃ x : ℝ, shift_sin x ≠ original_sin (x + Real.pi / 3)) ∧ 
  (∀ φ : ℝ, (∃ k : ℤ, φ = k * Real.pi + Real.pi / 6) ↔ 
    (∀ x : ℝ, cos_symmetry φ (Real.pi / 3 - x) = -cos_symmetry φ x)) ∧
  (∃ T : ℝ, T = Real.pi / 2 ∧ ∀ x : ℝ, tan_function (x + T) = tan_function x) ∧
  (simplify_expression ≠ 2 * Real.sin 1) := by
  sorry

#check exactly_two_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l1084_108488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projections_l1084_108468

/-- Represents a tetrahedron with its orthogonal projections onto face planes. -/
structure Tetrahedron where
  /-- The tetrahedron has an orthogonal projection onto one face plane that is a trapezoid with area 1. -/
  has_trapezoid_projection : Prop
  /-- Function that returns the area of the orthogonal projection onto a given face. -/
  projection_area : Fin 4 → ℝ

/-- Predicate to check if a projection is a square. -/
def is_square (area : ℝ) : Prop :=
  ∃ (side : ℝ), side > 0 ∧ side * side = area

/-- Theorem stating properties of the tetrahedron's projections. -/
theorem tetrahedron_projections (t : Tetrahedron) :
  t.has_trapezoid_projection →
  (∃ (f : Fin 4), t.projection_area f = 1 ∧ 
    ¬∃ (g : Fin 4), g ≠ f ∧ t.projection_area g = 1 ∧ is_square (t.projection_area g)) ∧
  (∃ (h : Fin 4), t.projection_area h = 1 / 2019 ∧ is_square (t.projection_area h)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projections_l1084_108468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_30_not_75_l1084_108441

theorem three_digit_multiples_of_30_not_75 :
  Finset.card (Finset.filter (λ n : ℕ => 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (Finset.range 1000)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_30_not_75_l1084_108441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_pop_probability_l1084_108406

theorem ice_pop_probability : 
  let total_pops : ℕ := 11
  let cherry_pops : ℕ := 4
  let orange_pops : ℕ := 3
  let lemon_lime_pops : ℕ := 4
  (cherry_pops + orange_pops + lemon_lime_pops = total_pops) →
  ((8 : ℚ) / 11) = 1 - (
    ((cherry_pops * (cherry_pops - 1) : ℚ) / (total_pops * (total_pops - 1))) +
    ((orange_pops * (orange_pops - 1) : ℚ) / (total_pops * (total_pops - 1))) +
    ((lemon_lime_pops * (lemon_lime_pops - 1) : ℚ) / (total_pops * (total_pops - 1)))
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ice_pop_probability_l1084_108406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l1084_108480

/-- Represents a player in the game -/
inductive Player : Type where
  | Bela : Player
  | Jenn : Player
deriving Repr, DecidableEq

/-- The game state -/
structure GameState where
  chosen : List ℝ
  current_player : Player

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ 10 ∧
  ∀ x ∈ state.chosen, |move - x| ≥ 2

/-- Defines a winning strategy for the first player -/
def has_winning_strategy (player : Player) : Prop :=
  ∃ (strategy : GameState → ℝ),
    ∀ (state : GameState),
      state.current_player = player →
      is_valid_move state (strategy state) →
      ¬∃ (opponent_move : ℝ),
        is_valid_move
          { chosen := strategy state :: state.chosen,
            current_player := if player = Player.Bela then Player.Jenn else Player.Bela }
          opponent_move

/-- The main theorem stating that Bela (the first player) has a winning strategy -/
theorem bela_always_wins : has_winning_strategy Player.Bela := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_always_wins_l1084_108480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1084_108492

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = -2/3) : 
  Real.cos (π - 2*α) = -1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_2alpha_l1084_108492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1084_108403

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*a*x + 3) / Real.log (1/2)

-- State the theorem
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1084_108403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1084_108498

theorem sin_theta_value (θ : ℝ) (a : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) :
  Real.sin θ = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1084_108498
