import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l766_76677

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - Real.log (a * x + 2 * a + 1) + 2

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x ≥ -2, f a x ≥ 0) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l766_76677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_represents_basketball_quantity_l766_76699

/-- Represents the quantity of basketballs -/
def x : ℕ := sorry

/-- Represents the quantity of soccer balls -/
def soccer_balls : ℕ := 2 * x

/-- Represents the total cost of soccer balls in yuan -/
def soccer_cost : ℕ := 5000

/-- Represents the total cost of basketballs in yuan -/
def basketball_cost : ℕ := 4000

/-- Represents the difference in unit price between basketballs and soccer balls in yuan -/
def price_difference : ℕ := 30

/-- States that the equation holds true -/
axiom equation_holds : (soccer_cost : ℚ) / soccer_balls = (basketball_cost : ℚ) / x - price_difference

theorem x_represents_basketball_quantity :
  x = x := by
  rfl

/-- Interpretation of what x represents -/
def x_interpretation : String := "quantity of basketballs"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_represents_basketball_quantity_l766_76699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_inverse_pairs_l766_76617

-- Define the matrix type
def Matrix2x2 (α : Type) := Fin 2 → Fin 2 → α

-- Define the identity matrix
def identityMatrix : Matrix2x2 ℝ := λ i j => if i = j then 1 else 0

-- Define the matrix multiplication
def matrixMul (A B : Matrix2x2 ℝ) : Matrix2x2 ℝ :=
  λ i j => (Finset.univ.sum λ k => A i k * B k j)

-- Define our specific matrix
def ourMatrix (a b : ℝ) : Matrix2x2 ℝ :=
  λ i j => match i, j with
    | 0, 0 => a
    | 0, 1 => 5
    | 1, 0 => -12
    | 1, 1 => b

-- Theorem statement
theorem two_inverse_pairs :
  ∃! n : ℕ, ∃ S : Finset (ℝ × ℝ),
    S.card = n ∧
    (∀ p : ℝ × ℝ, p ∈ S ↔ matrixMul (ourMatrix p.1 p.2) (ourMatrix p.1 p.2) = identityMatrix) ∧
    n = 2 := by
  sorry

#check two_inverse_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_inverse_pairs_l766_76617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_even_and_odd_in_sequence_l766_76653

/-- The sequence a_n defined as ⌊n√2⌋ + ⌊n√3⌋ for n ≥ 0 -/
noncomputable def a (n : ℕ) : ℤ := ⌊(n : ℝ) * Real.sqrt 2⌋ + ⌊(n : ℝ) * Real.sqrt 3⌋

/-- There are infinitely many even and infinitely many odd numbers in the sequence a_n -/
theorem infinite_even_and_odd_in_sequence :
  (∀ N : ℕ, ∃ n ≥ N, Even (a n)) ∧
  (∀ N : ℕ, ∃ n ≥ N, Odd (a n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_even_and_odd_in_sequence_l766_76653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_existence_l766_76628

-- Define a circle K
def Circle (K : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point on a circle
def PointOnCircle (p : ℝ × ℝ) (K : Set (ℝ × ℝ)) : Prop := sorry

-- Define a quadrilateral
def Quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define an inscribed circle in a quadrilateral
def HasInscribedCircle (A B C D : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem inscribed_circle_existence 
  (K : Set (ℝ × ℝ)) 
  (A B C : ℝ × ℝ) 
  (hK : Circle K)
  (hA : PointOnCircle A K)
  (hB : PointOnCircle B K)
  (hC : PointOnCircle C K) :
  ∃ D : ℝ × ℝ, PointOnCircle D K ∧ 
    Quadrilateral A B C D ∧ 
    HasInscribedCircle A B C D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_existence_l766_76628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_line_l766_76660

theorem equal_intercepts_line (l : ℝ) : 
  (∀ x y : ℝ, (2*l+1)*x + (l+2)*y + 2*l + 2 = 0) →
  (∃ a : ℝ, a ≠ 0 ∧ 
    (2*l+1)*a + 2*l + 2 = 0 ∧ 
    (l+2)*a + 2*l + 2 = 0) →
  l = 1 ∨ l = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercepts_line_l766_76660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_even_f_sin_periodic_f_sin_inv_even_f_sin_inv_not_periodic_l766_76679

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 1 + x else 1 - x

-- Define the composition of f with sin
noncomputable def f_sin (x : ℝ) : ℝ := f (Real.sin x)

-- Define the composition of f with sin(1/x)
noncomputable def f_sin_inv (x : ℝ) : ℝ := f (Real.sin (1 / x))

-- Theorem 1: f_sin is an even function
theorem f_sin_even : ∀ x, f_sin (-x) = f_sin x := by sorry

-- Theorem 2: f_sin is periodic with period 2π
theorem f_sin_periodic : ∀ x, f_sin (x + 2 * Real.pi) = f_sin x := by sorry

-- Theorem 3: f_sin_inv is an even function
theorem f_sin_inv_even : ∀ x, f_sin_inv (-x) = f_sin_inv x := by sorry

-- Theorem 4: f_sin_inv is not periodic
theorem f_sin_inv_not_periodic : ¬ ∃ p, p ≠ 0 ∧ ∀ x, f_sin_inv (x + p) = f_sin_inv x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_even_f_sin_periodic_f_sin_inv_even_f_sin_inv_not_periodic_l766_76679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_in_2004_l766_76622

def initial_participation : ℕ := 300
def annual_increase_rate : ℚ := 40 / 100
def years_passed : ℕ := 4

noncomputable def expected_participants : ℕ :=
  ⌊(initial_participation : ℚ) * (1 + annual_increase_rate) ^ years_passed⌋.toNat

theorem participants_in_2004 :
  expected_participants = 1152 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_participants_in_2004_l766_76622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l766_76643

-- Define set A
def A : Set ℤ := {x | (x + 2) * (x - 5) < 0}

-- Define set B (now as a set of integers)
def B : Set ℤ := {x | x > -5 ∧ x < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l766_76643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l766_76631

/-- The curve on which point P lies -/
def curve (x y : ℝ) : Prop := x^2 - y - Real.log x = 0

/-- The line to which we're measuring the distance -/
def line (x y : ℝ) : Prop := y = x - 3

/-- The minimum distance from a point on the curve to the line -/
noncomputable def min_distance : ℝ := 3 * Real.sqrt 2 / 2

theorem min_distance_theorem :
  ∀ (x y : ℝ), curve x y →
  ∃ (d : ℝ), d ≥ 0 ∧
    (∀ (x' y' : ℝ), line x' y' → d ≤ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    d = min_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l766_76631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_intervals_l766_76694

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 2 * Real.cos x * (1 - 2 * Real.sin x)

-- Define the domain
def domain : Set ℝ := Set.Ioo 0 (2 * Real.pi)

-- Define the intervals of monotonic decrease
def decrease_intervals : Set (Set ℝ) := {Set.Ioo (Real.pi / 6) (Real.pi / 2), Set.Ioo (5 * Real.pi / 6) (3 * Real.pi / 2)}

-- Theorem statement
theorem monotonic_decrease_intervals :
  ∀ x ∈ domain, (∃ I ∈ decrease_intervals, x ∈ I) ↔ f' x < 0 := by
  sorry

#check monotonic_decrease_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_intervals_l766_76694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_216_l766_76626

/-- A kite with given side lengths and one diagonal -/
structure Kite where
  side1 : ℝ
  side2 : ℝ
  diagonal1 : ℝ
  rightAngle : Bool

/-- The area of a kite given its diagonals -/
noncomputable def kiteArea (d1 d2 : ℝ) : ℝ := (1/2) * d1 * d2

/-- Theorem: The area of the specific kite is 216 square units -/
theorem kite_area_is_216 (k : Kite) (h1 : k.side1 = 15) (h2 : k.side2 = 20) 
    (h3 : k.diagonal1 = 24) (h4 : k.rightAngle = true) : 
    ∃ (x : ℝ), kiteArea k.diagonal1 x = 216 := by
  sorry

#check kite_area_is_216

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_area_is_216_l766_76626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l766_76614

theorem exponent_problem : (625 : ℝ) ^ (12 / 100 : ℝ) * (625 : ℝ) ^ (38 / 100 : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l766_76614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_satisfies_conditions_l766_76664

noncomputable def sequenceTerm (n : ℕ) : ℚ := (2011 * 6^(n-1)) / n + n

theorem sequence_satisfies_conditions :
  (sequenceTerm 1 = 2012) ∧
  (∀ n : ℕ, n ≥ 2 → n ≤ 3 → ∃ m : ℤ, sequenceTerm n = ↑m) ∧
  (∃ (c q : ℚ), q > 0 ∧ q ≤ 10 ∧
    ∀ n : ℕ, n > 0 → n * (sequenceTerm n) - n^2 = c * q^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_satisfies_conditions_l766_76664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_X_fill_time_l766_76661

-- Define the rates of pumps X and Y
def rate_X : ℚ := 1 / 40
def rate_Y : ℚ := 1 / 48

-- Define the time it takes for pump X alone to fill the tank
def time_X : ℚ := 40

-- Define the time it takes for both pumps to fill the tank
def time_both : ℚ := 6 * time_X

-- Axioms based on the problem conditions
axiom pump_Y_empty_time : rate_Y = 1 / 48

axiom both_pumps_relation : time_both = 6 * time_X

axiom fill_rate_relation : rate_X - rate_Y = (1 / 6) * rate_X

-- Theorem to prove
theorem pump_X_fill_time : time_X = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pump_X_fill_time_l766_76661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l766_76641

def A : Set ℕ := {2, 4, 6, 8, 10}
def B : Set ℕ := {x : ℕ | 3 ≤ x ∧ x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l766_76641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_square_condition_l766_76691

/-- 
For a quadratic equation x² + px + q = 0, this theorem states the 
necessary and sufficient condition for one root to be the square of the other.
-/
theorem quadratic_root_square_condition (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧ x₂ = x₁^2) ↔ 
  p = -(Real.rpow q (1/3) + Real.rpow q (2/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_square_condition_l766_76691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_two_l766_76692

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2) + x

-- State the theorem
theorem integral_equals_pi_over_two :
  ∫ x in Set.Icc (-1) 1, f x = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_pi_over_two_l766_76692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_arrangement_l766_76686

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Represents the equation (a) × (bc) × (def) = (ghij) -/
def satisfies_equation (a b c d e f g h i j : Digit) : Prop :=
  (a.val : ℕ) * (10 * b.val + c.val : ℕ) * (100 * d.val + 10 * e.val + f.val : ℕ) = 
  (1000 * g.val + 100 * h.val + 10 * i.val + j.val : ℕ)

/-- All digits are different -/
def all_different (a b c d e f g h i j : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem unique_digit_arrangement :
  ∃! (a b c d e f g h i j : Digit),
    satisfies_equation a b c d e f g h i j ∧
    all_different a b c d e f g h i j :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_arrangement_l766_76686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_calculation_l766_76611

theorem floor_abs_calculation : 
  (Int.floor (abs (-5.7 : ℝ)))^2 + (Int.natAbs (Int.floor (-5.7 : ℝ))) - (1/2 : ℚ) = 61/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_abs_calculation_l766_76611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_installation_cost_calculation_l766_76658

noncomputable def purchase_price : ℝ := 16500
noncomputable def discount_rate : ℝ := 0.2
noncomputable def transport_cost : ℝ := 125
noncomputable def desired_selling_price : ℝ := 23100
noncomputable def profit_rate : ℝ := 0.1

noncomputable def labelled_price : ℝ := purchase_price / (1 - discount_rate)

noncomputable def installation_cost : ℝ := desired_selling_price - purchase_price - transport_cost

theorem installation_cost_calculation :
  installation_cost = 6350 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_installation_cost_calculation_l766_76658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l766_76685

-- Define the points and the orthocenter
noncomputable def A (x : ℝ) : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
def B : ℝ × ℝ := (-3, -1)
def C : ℝ × ℝ := (2, -1)
def H : ℝ → ℝ → ℝ × ℝ := λ x y ↦ (x, y)

-- Define the orthocenter equation
def orthocenter_equation (x y : ℝ) : Prop :=
  y = (6 - x - x^2) / (1 + Real.sqrt (1 - x^2)) - 1

-- Define what it means for a point to be the orthocenter of a triangle
def is_orthocenter_of (h a b c : ℝ × ℝ) : Prop :=
  sorry -- We'll leave this undefined for now, as it's not essential for the build

-- State the theorem
theorem orthocenter_locus :
  ∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 →
  (is_orthocenter_of (H x y) (A x) B C) →
  orthocenter_equation x y :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l766_76685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_ticket_receipts_l766_76616

/-- Calculates the total receipts from ticket sales at a rally --/
theorem rally_ticket_receipts 
  (total_attendance : ℕ) 
  (pre_rally_price : ℚ) 
  (door_price : ℚ) 
  (pre_rally_tickets : ℕ) 
  (h1 : total_attendance = 750)
  (h2 : pre_rally_price = 2)
  (h3 : door_price = 2.75)
  (h4 : pre_rally_tickets = 475)
  : total_receipts = 1706.25 := by
  have door_tickets : ℕ := total_attendance - pre_rally_tickets
  have pre_rally_revenue : ℚ := pre_rally_tickets * pre_rally_price
  have door_revenue : ℚ := door_tickets * door_price
  have total_receipts : ℚ := pre_rally_revenue + door_revenue
  sorry

where
  total_receipts : ℚ := (pre_rally_tickets * pre_rally_price) + 
                        ((total_attendance - pre_rally_tickets) * door_price)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rally_ticket_receipts_l766_76616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_inequality_implies_m_greater_than_one_l766_76669

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.exp x) / (Real.exp x + a)

theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) → a = 1 :=
by sorry

theorem inequality_implies_m_greater_than_one (m : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f 1 (2^(x+1) - 4^x) + f 1 (1-m) > 0) → m > 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_inequality_implies_m_greater_than_one_l766_76669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l766_76600

noncomputable def a (x m : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, m + Real.cos x)

noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, -m + Real.cos x)

noncomputable def f (x m : ℝ) : ℝ := (a x m).1 * (b x m).1 + (a x m).2 * (b x m).2

theorem max_value_of_f (m : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc (-π/6) (π/3) ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-π/6) (π/3) → f y m ≤ f x m) ∧
  f x m = -3/2 ∧ x = π/6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l766_76600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_when_area_equals_perimeter_l766_76607

theorem rectangle_width_when_area_equals_perimeter :
  ∀ w : ℝ,
    w > 0 →
    let l := 2 * w
    l * w = 2 * (l + w) →
    w = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_when_area_equals_perimeter_l766_76607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l766_76621

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : f ω (Real.pi / 6) = f ω (Real.pi / 3))
  (h3 : ∃ (x_min : ℝ), x_min ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3) ∧ 
    ∀ (x : ℝ), x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3) → f ω x_min ≤ f ω x)
  (h4 : ∀ (x : ℝ), x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3) → 
    ∃ (y : ℝ), y ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3) ∧ f ω y > f ω x) :
  ω = 14 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l766_76621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saline_mixture_volume_l766_76638

/-- The total volume of a saline solution mixture -/
noncomputable def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

/-- The concentration of a saline solution mixture -/
noncomputable def mixture_concentration (c1 c2 v1 v2 : ℝ) : ℝ :=
  (c1 * v1 + c2 * v2) / (v1 + v2)

theorem saline_mixture_volume :
  let v1 : ℝ := 3.6  -- Volume of 1% solution
  let v2 : ℝ := 1.4  -- Volume of 9% solution
  let c1 : ℝ := 0.01 -- Concentration of 1% solution
  let c2 : ℝ := 0.09 -- Concentration of 9% solution
  let desired_concentration : ℝ := 0.0324 -- 3.24% desired concentration
  (mixture_concentration c1 c2 v1 v2 = desired_concentration) →
  (total_volume v1 v2 = 5.0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saline_mixture_volume_l766_76638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_outside_approx_three_l766_76639

/-- The probability of a value falling outside μ ± 3σ in a normal distribution -/
def prob_outside_3sigma : ℝ := 0.0027

/-- The sample size -/
def sample_size : ℕ := 1000

/-- The expected number of samples falling outside (μ - 3σ, μ + 3σ) -/
noncomputable def expected_outside : ℝ := prob_outside_3sigma * (sample_size : ℝ)

theorem expected_outside_approx_three :
  ⌈expected_outside⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_outside_approx_three_l766_76639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l766_76665

/-- Represents a candle with its burn time in minutes -/
structure Candle where
  burn_time : ℝ

/-- Calculates the length of a candle's stub after a given time -/
noncomputable def stub_length (c : Candle) (initial_length : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_length * (1 - elapsed_time / c.burn_time)

theorem candle_lighting_time 
  (candle1 candle2 : Candle)
  (h1 : candle1.burn_time = 4 * 60)
  (h2 : candle2.burn_time = 6 * 60)
  (initial_length : ℝ)
  (h3 : initial_length > 0)
  (elapsed_time : ℝ)
  (h4 : elapsed_time > 0)
  (h5 : elapsed_time < min candle1.burn_time candle2.burn_time)
  (h6 : stub_length candle2 initial_length elapsed_time = 
        3 * stub_length candle1 initial_length elapsed_time) :
  elapsed_time = 3 * 60 + 26 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_lighting_time_l766_76665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l766_76672

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 9 / (x + 1)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -1 ∧ f x = y) ↔ y ∈ Set.Iic (-7) ∪ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l766_76672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_signup_combinations_l766_76673

def number_of_signup_combinations (n k : ℕ) : ℕ := k ^ n

theorem student_signup_combinations (n k : ℕ) :
  n > 0 → k > 0 → (k ^ n : ℕ) = number_of_signup_combinations n k :=
by
  intros hn hk
  unfold number_of_signup_combinations
  rfl

#eval number_of_signup_combinations 5 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_signup_combinations_l766_76673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l766_76657

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x else x^2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l766_76657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3b_minus_a_minus_c_l766_76625

theorem sqrt_3b_minus_a_minus_c : 
  ∀ (a b c : ℤ),
  (Real.sqrt (2 * a - 1 : ℝ) = 3) →
  (b = Int.floor (Real.sqrt 10)) →
  (c = -8) →  -- Changed from Real.cubeRoot to direct equality
  Real.sqrt (3 * b - a - c : ℝ) = 2 * Real.sqrt 3 ∨
  Real.sqrt (3 * b - a - c : ℝ) = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3b_minus_a_minus_c_l766_76625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_approx_8_2_l766_76695

/-- A line with slope 1 passing through a point on the negative x-axis and intersecting y = x^2 -/
structure IntersectingLine where
  b : ℝ
  hp : b > 0

/-- The y-intercept of the line PR -/
def y_intercept (l : IntersectingLine) : ℝ := l.b

/-- The x-coordinates of the intersection points Q and R -/
noncomputable def intersection_x (l : IntersectingLine) : ℝ × ℝ :=
  ((1 - Real.sqrt (1 + 4 * l.b)) / 2, (1 + Real.sqrt (1 + 4 * l.b)) / 2)

/-- The condition that PQ = QR -/
def equal_distances (l : IntersectingLine) : Prop :=
  let (xq, xr) := intersection_x l
  |xq + l.b| = |xr - xq|

theorem y_intercept_approx_8_2 (l : IntersectingLine) (h : equal_distances l) :
  ∃ ε > 0, |y_intercept l - 8.2| < ε := by
  sorry

#check y_intercept_approx_8_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_approx_8_2_l766_76695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l766_76670

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the area of the figure
noncomputable def area : ℝ := ∫ x in (Set.Icc 0 1), (1 - curve x)

-- Theorem statement
theorem area_enclosed_by_curve : area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curve_l766_76670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l766_76674

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line that intersects the circle
def intersecting_line (x y : ℝ) : Prop := 4*x + 3*y - 12 = 0

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

-- Define the point outside the circle
def point_outside : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (R : ℝ),
    -- The circle equation
    (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 1)^2 = R^2) ∧
    -- The radius is correct
    R^2 = 4 ∧
    -- The chord length is correct
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
      intersecting_line x₁ y₁ ∧ intersecting_line x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = chord_length^2 ∧
    -- The tangent lines are correct
    (∀ x y, (x = -1 ∨ 3*x - 4*y + 11 = 0) →
      -- The line passes through the given point
      (x = point_outside.1 ∧ y = point_outside.2 ∨
       -- Or it's tangent to the circle
       ∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧
         (x - x₀)^2 + (y - y₀)^2 = R^2 ∧
         (∀ x' y', circle_C x' y' →
           (x' - x₀)^2 + (y' - y₀)^2 ≥ (x - x₀)^2 + (y - y₀)^2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangent_lines_l766_76674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l766_76682

theorem starting_number_proof (n : ℕ) : 
  (∃ (seq : List ℕ), 
    seq.length = 12 ∧ 
    (∀ m ∈ seq, m % 3 = 0) ∧
    seq.maximum ≤ some 46 ∧ 
    (∀ i j, i < j → seq.get? i < seq.get? j) ∧
    (∀ k, k ∈ seq → k ≥ n) ∧
    (∀ k, k % 3 = 0 → k < n ∨ k > (seq.maximum.getD 0))) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_starting_number_proof_l766_76682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_total_l766_76601

theorem restaurant_bill_total 
  (alice_tip bob_tip carol_tip : ℝ)
  (alice_percent bob_percent carol_percent : ℝ)
  (h1 : alice_tip = 5)
  (h2 : bob_tip = 3)
  (h3 : carol_tip = 9)
  (h4 : alice_percent = 0.20)
  (h5 : bob_percent = 0.15)
  (h6 : carol_percent = 0.30)
  (h7 : alice_tip = alice_percent * (alice_tip / alice_percent))
  (h8 : bob_tip = bob_percent * (bob_tip / bob_percent))
  (h9 : carol_tip = carol_percent * (carol_tip / carol_percent)) :
  (alice_tip / alice_percent) + (bob_tip / bob_percent) + (carol_tip / carol_percent) = 75 := by
  sorry

#check restaurant_bill_total

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_total_l766_76601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l766_76667

/-- The length of the longest segment in a cylinder --/
noncomputable def longest_segment (radius : ℝ) (height : ℝ) : ℝ :=
  Real.sqrt ((2 * radius) ^ 2 + height ^ 2)

/-- Theorem: The longest segment in a cylinder with radius 5 cm and height 10 cm is 10√2 cm --/
theorem longest_segment_in_specific_cylinder :
  longest_segment 5 10 = 10 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_segment_in_specific_cylinder_l766_76667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_eq_cos_3x_l766_76623

theorem sin_2x_eq_cos_3x (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (3 * x) ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * π * k ∨
            x = 3 * π / 2 + 2 * π * k ∨
            x = π / 10 + 2 * π * k ∨
            x = 9 * π / 10 + 2 * π * k ∨
            x = 13 * π / 10 + 2 * π * k ∨
            x = 17 * π / 10 + 2 * π * k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_eq_cos_3x_l766_76623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l766_76619

-- Define the function as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - 2*x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l766_76619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PTXM_eq_18_l766_76647

/-- A 10-sided polygon with specific properties -/
structure DecagonPQRSTVWXYZ where
  /-- Each side of the polygon is 3 units long -/
  side_length : ℝ
  side_length_eq : side_length = 3
  /-- Each angle is 120°, except for angles at P, T, and Y which are 90° -/
  angle : Fin 10 → ℝ
  angle_eq : ∀ i, angle i = if i = 0 ∨ i = 3 ∨ i = 6 then 90 else 120

/-- The quadrilateral PTXM formed by the intersection of PV and TX -/
structure QuadrilateralPTXM (d : DecagonPQRSTVWXYZ) where
  P : ℝ × ℝ
  T : ℝ × ℝ
  X : ℝ × ℝ
  M : ℝ × ℝ

/-- The area of a quadrilateral -/
noncomputable def area (d : DecagonPQRSTVWXYZ) (q : QuadrilateralPTXM d) : ℝ := sorry

/-- Theorem stating that the area of quadrilateral PTXM is 18 units² -/
theorem area_PTXM_eq_18 (d : DecagonPQRSTVWXYZ) (q : QuadrilateralPTXM d) :
  area d q = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PTXM_eq_18_l766_76647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l766_76612

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 7 - 3 * x

-- Define the proposed inverse function g
noncomputable def g (x : ℝ) : ℝ := (7 - x) / 3

-- Theorem statement
theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l766_76612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_capacity_of_scaled_box_l766_76627

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ := d.height * d.width * d.length

/-- Represents the clay capacity of a box -/
noncomputable def clayCapacity (d : BoxDimensions) : ℝ := 
  48 * (boxVolume d) / (boxVolume { height := 2, width := 4, length := 6 })

theorem clay_capacity_of_scaled_box :
  let box1 : BoxDimensions := { height := 2, width := 4, length := 6 }
  let box2 : BoxDimensions := { height := 3 * box1.height, width := 2 * box1.width, length := 1.5 * box1.length }
  clayCapacity box2 = 432 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clay_capacity_of_scaled_box_l766_76627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l766_76676

/-- A line in the xy-plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- The x-intercept of a line, if it exists. -/
noncomputable def x_intercept (l : Line) : Option ℝ :=
  if l.b = 0 then none else some (-l.c / l.a)

/-- The y-intercept of a line, if it exists. -/
noncomputable def y_intercept (l : Line) : Option ℝ :=
  if l.a = 0 then none else some (-l.c / l.b)

/-- Theorem: The line 4x - 3y + 12 = 0 has x-intercept -3 and y-intercept 4. -/
theorem line_intercepts : 
  let l : Line := ⟨4, -3, 12, Or.inl (by norm_num)⟩
  x_intercept l = some (-3) ∧ y_intercept l = some 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l766_76676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_domain_range_l766_76633

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f2 (x : ℝ) : ℝ := x⁻¹
noncomputable def f3 (x : ℝ) : ℝ := x^(1/3)  -- Changed from Real.cbrt
noncomputable def f4 (x : ℝ) : ℝ := x^2

-- Define the domains and ranges
def domain1 : Set ℝ := {x | x ≥ 0}
def range1 : Set ℝ := {y | y ≥ 0}
def domain2 : Set ℝ := {x | x ≠ 0}
def range2 : Set ℝ := {y | y ≠ 0}
def domain3 : Set ℝ := Set.univ
def range3 : Set ℝ := Set.univ
def domain4 : Set ℝ := Set.univ
def range4 : Set ℝ := {y | y ≥ 0}

-- Theorem statement
theorem different_domain_range :
  (domain4 ≠ domain1 ∨ range4 ≠ range1) ∧
  (domain4 ≠ domain2 ∨ range4 ≠ range2) ∧
  (domain4 ≠ domain3 ∨ range4 ≠ range3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_domain_range_l766_76633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprockets_per_hour_machine_x_l766_76604

/-- Given two machines X and B that manufacture sprockets, this theorem proves
    the number of sprockets produced per hour by machine X. -/
theorem sprockets_per_hour_machine_x 
  (total_sprockets time_difference : ℕ) 
  (production_ratio : ℚ) 
  (h1 : total_sprockets = 660)
  (h2 : time_difference = 10)
  (h3 : production_ratio = 11/10) :
  ∃ sprockets_per_hour_x : ℕ, sprockets_per_hour_x = 6 := by
  -- Proof that machine X produces 6 sprockets per hour
  sorry

#check sprockets_per_hour_machine_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprockets_per_hour_machine_x_l766_76604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_FAC_value_l766_76620

/-- A rectangular prism with specific dimensions -/
structure RectangularPrism where
  AB : ℝ
  AD : ℝ
  AE : ℝ
  is_rectangular : True

/-- The sine of angle FAC in the given rectangular prism -/
noncomputable def sine_FAC (prism : RectangularPrism) : ℝ :=
  Real.sqrt (49 / 50)

/-- Theorem stating that for a rectangular prism with AB = 1, AD = 2, and AE = 3, 
    the sine of angle FAC is equal to √(49/50) -/
theorem sine_FAC_value (prism : RectangularPrism) 
  (h1 : prism.AB = 1) (h2 : prism.AD = 2) (h3 : prism.AE = 3) : 
  sine_FAC prism = Real.sqrt (49 / 50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_FAC_value_l766_76620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l766_76690

/-- A rectangular field with one uncovered side -/
structure RectangularField where
  uncovered_side : ℝ
  area : ℝ

/-- Calculate the fencing required for a rectangular field -/
noncomputable def fencing_required (field : RectangularField) : ℝ :=
  let width := field.area / field.uncovered_side
  2 * width + field.uncovered_side

/-- Theorem: The fencing required for a field with area 720 sq. feet and uncovered side 20 feet is 92 feet -/
theorem fencing_for_given_field :
  fencing_required { uncovered_side := 20, area := 720 } = 92 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_for_given_field_l766_76690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l766_76606

/-- A triangular pyramid with specific properties -/
structure TriangularPyramid where
  a : ℝ  -- Length of lateral edges
  right_angle : Bool  -- One angle at the vertex is a right angle
  other_angles : Bool  -- Other two angles at the vertex are 60°

/-- The volume of a triangular pyramid with specific properties -/
noncomputable def pyramid_volume (p : TriangularPyramid) : ℝ :=
  (p.a^3 * Real.sqrt 6) / 12

/-- Theorem: The volume of the specified triangular pyramid is (a³√6) / 12 -/
theorem pyramid_volume_formula (p : TriangularPyramid) 
  (h1 : p.a > 0) 
  (h2 : p.right_angle) 
  (h3 : p.other_angles) : 
  pyramid_volume p = (p.a^3 * Real.sqrt 6) / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_formula_l766_76606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l766_76649

noncomputable def f (m : ℤ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

theorem power_function_increasing (m : ℤ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x < f m y) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l766_76649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_farm_boxes_packed_l766_76680

/-- A fruit farm packs oranges in boxes. -/
structure FruitFarm where
  /-- The number of oranges each box holds -/
  oranges_per_box : ℕ
  /-- The total number of oranges -/
  total_oranges : ℕ
  /-- The number of days to pack the boxes -/
  days_to_pack : ℕ

/-- Calculates the number of boxes packed in a day -/
def boxes_packed_per_day (farm : FruitFarm) : ℕ :=
  farm.total_oranges / farm.oranges_per_box

/-- Theorem: The fruit farm packs 2650 boxes in a day -/
theorem fruit_farm_boxes_packed (farm : FruitFarm) 
  (h1 : farm.oranges_per_box = 10)
  (h2 : farm.total_oranges = 26500)
  (h3 : farm.days_to_pack = 1) :
  boxes_packed_per_day farm = 2650 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_farm_boxes_packed_l766_76680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_group_frequency_and_rate_l766_76637

/-- Given a sample of 30 data points divided into 4 groups with a ratio of 2:4:3:1,
    the frequency of the third group is 9 and its frequency rate is 0.3. -/
theorem third_group_frequency_and_rate :
  let total_sample : ℕ := 30
  let num_groups : ℕ := 4
  let group_ratio : List ℕ := [2, 4, 3, 1]
  let third_group_index : ℕ := 2  -- 0-based index
  let total_ratio : ℕ := group_ratio.sum
  let third_group_frequency : ℕ := total_sample * (group_ratio.get! third_group_index) / total_ratio
  let third_group_frequency_rate : ℚ := third_group_frequency / total_sample
  third_group_frequency = 9 ∧ third_group_frequency_rate = 3/10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_group_frequency_and_rate_l766_76637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_expression_l766_76671

/-- The number of distinct values that the expression 3^(3^(3^3)) can have when parenthesized differently -/
def num_distinct_values : ℕ := 2

/-- The original expression 3^(3^(3^3)) -/
def original_expression : ℕ → ℕ → ℕ → ℕ := λ a b c ↦ a^(b^(c^3))

theorem distinct_values_of_expression :
  ∃ (f g : ℕ → ℕ → ℕ → ℕ),
    f 3 3 3 ≠ g 3 3 3 ∧
    (∀ h : ℕ → ℕ → ℕ → ℕ, h 3 3 3 = f 3 3 3 ∨ h 3 3 3 = g 3 3 3) ∧
    (f 3 3 3 = original_expression 3 3 3 ∨ g 3 3 3 = original_expression 3 3 3) := by
  sorry

#check distinct_values_of_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_values_of_expression_l766_76671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_problems_l766_76640

theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) 
  (h1 : total_problems = 40)
  (h2 : pass_percentage = 85 / 100) : 
  (total_problems : ℤ) - Int.floor (pass_percentage * total_problems) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missed_problems_l766_76640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_6_times_β_l766_76693

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 5; 0, -2]
def β : Matrix (Fin 2) (Fin 1) ℝ := !![-1; 1]

theorem A_power_6_times_β : A ^ 6 * β = !![-64; 64] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_6_times_β_l766_76693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l766_76615

noncomputable section

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus F
def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define point A
def point_a : ℝ × ℝ := (0, -2)

-- Define the distance from major vertex to point A
def distance_to_a : ℝ := 2 * Real.sqrt 2

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 2

-- Define the area of triangle OMN
noncomputable def triangle_area (k : ℝ) : ℝ := 4 * Real.sqrt (4 * k^2 - 3) / (1 + 4 * k^2)

theorem ellipse_and_max_area_line :
  -- The equation of the ellipse
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 4 + y^2 = 1) ∧
  -- The equation of the line when the area is maximized
  (∃ k : ℝ, k = Real.sqrt 7 / 2 ∧
    (∀ k' : ℝ, k' ≠ k → triangle_area k ≥ triangle_area k') ∧
    (∀ x y : ℝ, line k x y ↔ 2 * y - Real.sqrt 7 * x + 4 = 0)) ∧
  (∃ k : ℝ, k = -Real.sqrt 7 / 2 ∧
    (∀ k' : ℝ, k' ≠ k → triangle_area k ≥ triangle_area k') ∧
    (∀ x y : ℝ, line k x y ↔ 2 * y + Real.sqrt 7 * x + 4 = 0)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_max_area_line_l766_76615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_intersection_l766_76696

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define a line with slope 1
def line_slope_1 (x y t : ℝ) : Prop := y = x + t

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ t : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
  line_slope_1 x₁ y₁ t ∧ line_slope_1 x₂ y₂ t ∧
  x₁ ≠ x₂

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem max_distance_intersection :
  ∃ (x₁ y₁ x₂ y₂ t : ℝ),
    intersection_points x₁ y₁ x₂ y₂ t ∧
    ∀ (a₁ b₁ a₂ b₂ s : ℝ),
      intersection_points a₁ b₁ a₂ b₂ s →
      distance x₁ y₁ x₂ y₂ ≥ distance a₁ b₁ a₂ b₂ ∧
      distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_intersection_l766_76696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_true_l766_76659

/-- Represents a 4x4 grid of signs -/
def Grid := Fin 4 → Fin 4 → Bool

/-- An operation that flips signs in a row, column, or diagonal -/
def Operation := Grid → Grid

/-- Counts the number of True values in a grid -/
def count_true (g : Grid) : Nat :=
  Finset.sum (Finset.univ : Finset (Fin 4)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 4)) (λ j =>
      if g i j then 1 else 0))

/-- Checks if a number is odd -/
def is_odd (n : Nat) : Prop := n % 2 = 1

/-- Initial grid configuration -/
def initial_grid : Grid :=
  λ i j => ¬(i = 0 ∧ j = 3)

/-- An operation preserves the parity of True values -/
axiom parity_preserving (op : Operation) (g : Grid) :
  is_odd (count_true g) ↔ is_odd (count_true (op g))

/-- The theorem to be proved -/
theorem impossible_all_true : 
  ∀ (ops : List Operation), 
    is_odd (count_true initial_grid) → 
    count_true (ops.foldl (λ g op => op g) initial_grid) ≠ 16 := by
  sorry

#eval count_true initial_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_all_true_l766_76659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_value_l766_76698

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- The value of cd for an equilateral triangle with vertices (0,0), (c,15), and (d,43) -/
noncomputable def equilateral_triangle_cd : ℝ :=
  (5051 - 3195 * Real.sqrt 3) / (18 * Real.sqrt 3)

/-- Calculate the distance between two complex points -/
noncomputable def distance (p1 p2 : ComplexPoint) : ℝ :=
  Real.sqrt ((p1.re - p2.re)^2 + (p1.im - p2.im)^2)

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : ComplexPoint) : Prop :=
  let d12 := distance p1 p2
  let d23 := distance p2 p3
  let d31 := distance p3 p1
  d12 = d23 ∧ d23 = d31

/-- Theorem: The product cd for an equilateral triangle with vertices (0,0), (c,15), and (d,43) 
    is equal to (5051 - 3195√3) / (18√3) -/
theorem equilateral_triangle_cd_value 
  (origin : ComplexPoint)
  (point1 : ComplexPoint)
  (point2 : ComplexPoint)
  (h1 : origin.re = 0 ∧ origin.im = 0)
  (h2 : point1.im = 15)
  (h3 : point2.im = 43)
  (h_equilateral : IsEquilateralTriangle origin point1 point2) :
  point1.re * point2.re = equilateral_triangle_cd :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_value_l766_76698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l766_76642

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  B : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 3 ∧ t.c = 3 ∧ t.B = Real.pi / 6

-- State the theorem
theorem triangle_side_a (t : Triangle) 
  (h : triangle_conditions t) : 
  t.a = Real.sqrt 3 ∨ t.a = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_a_l766_76642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_sum_distances_l766_76610

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 6*y

-- Define the focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F.2 = 3/2

-- Define that a point lies on the parabola
def on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

theorem triangle_on_parabola_sum_distances
  (A B C F : ℝ × ℝ)
  (h_A : on_parabola A)
  (h_B : on_parabola B)
  (h_C : on_parabola C)
  (h_F : focus F)
  (h_centroid : F - A = (1/3 : ℝ) • ((B - A) + (C - A))) :
  ‖A - F‖ + ‖B - F‖ + ‖C - F‖ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_on_parabola_sum_distances_l766_76610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_equals_l766_76681

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 5

noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.sqrt (f x)) - 2

noncomputable def h (x : ℝ) : ℝ := f (g x)

-- State the theorem
theorem h_of_two_equals : h 2 = 2 * Real.exp (2 * Real.sqrt 17) - 6 * Real.exp (Real.sqrt 17) + 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_two_equals_l766_76681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_theorem_l766_76687

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x + 1

-- Define the set of x values
def X : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

-- Define the set of y values
def Y : Set ℝ := {y | -3 ≤ y ∧ y ≤ 3}

-- Define the set of valid a values
def A : Set ℝ := {a | a ∈ Set.Icc (-1/2) 0 ∪ Set.Ioo 0 1}

-- Theorem statement
theorem quadratic_range_theorem :
  ∀ a : ℝ, (∀ x ∈ X, f a x ∈ Y) ↔ a ∈ A := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_theorem_l766_76687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_theorem_l766_76668

/-- Represents a stock with a dividend rate and yield -/
structure Stock where
  dividendRate : ℚ
  yield : ℚ

/-- Calculates the quoted price of a stock given its dividend rate and yield -/
def quotedPrice (s : Stock) : ℚ := s.dividendRate / s.yield

/-- Theorem stating that a stock with 20% dividend rate and 8% yield has a quoted price of $2.50 -/
theorem stock_price_theorem (s : Stock) 
  (h1 : s.dividendRate = 1/5) 
  (h2 : s.yield = 2/25) : 
  quotedPrice s = 5/2 := by
  simp [quotedPrice, h1, h2]
  norm_num

-- Example calculation
#eval quotedPrice { dividendRate := 1/5, yield := 2/25 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_theorem_l766_76668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l766_76645

theorem min_value_expression (a b : ℤ) (ha : 0 < a ∧ a < 8) (hb : 0 < b ∧ b < 8) :
  (∀ x y : ℤ, 0 < x ∧ x < 8 → 0 < y ∧ y < 8 → 3 * x - 2 * x * y ≥ 3 * a - 2 * a * b) →
  3 * a - 2 * a * b = -77 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l766_76645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_pipe_fill_time_l766_76663

/-- Represents the time it takes for a pipe to fill a pool -/
def FillTime := ℝ

/-- Represents the rate at which a pipe fills a pool (in pools per hour) -/
def FillRate := ℝ

/-- The time it takes for all three pipes to fill the pool together -/
def allPipesTime : ℝ := 3

/-- The time it takes for the third pipe to fill the pool alone -/
def thirdPipeTime : ℝ := 8

/-- The ratio of the first pipe's speed to the second pipe's speed -/
def firstToSecondRatio : ℝ := 1.25

theorem faster_pipe_fill_time :
  ∃ (r₁ r₂ r₃ : ℝ),
    r₁ = firstToSecondRatio * r₂ ∧
    r₃ = 1 / thirdPipeTime ∧
    r₁ + r₂ + r₃ = 1 / allPipesTime →
    1 / r₁ = 8.64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_pipe_fill_time_l766_76663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flips_count_l766_76636

/-- Represents a fair coin with probability of tails equal to 1/2 -/
structure FairCoin where
  prob_tails : ℚ
  fair : prob_tails = 1/2

/-- The number of times the coin is flipped -/
def num_flips : ℕ → ℕ := id

/-- The probability of getting tails on the first flip and heads on the last two flips
    when flipping a fair coin n times -/
noncomputable def prob_tails_first_heads_last_two (n : ℕ) (c : FairCoin) : ℚ :=
  c.prob_tails * (1 - c.prob_tails)^2 * (1/2)^(n - 3)

theorem coin_flips_count (c : FairCoin) :
  ∃ n : ℕ, n > 0 ∧ prob_tails_first_heads_last_two n c = 1/8 → num_flips n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flips_count_l766_76636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l766_76689

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - 2*x + 3) / Real.log a

-- State the theorem
theorem log_function_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, |f a x| < 1) →
  (a > 6 ∨ (a > 0 ∧ a < 1/6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_range_l766_76689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_can_reach_all_naturals_l766_76609

/-- Represents a jump of the flea -/
structure Jump where
  length : ℕ
  direction : Bool  -- True for right, False for left

/-- The sequence of jumps the flea makes -/
def JumpSequence := List Jump

/-- The position of the flea after a sequence of jumps -/
def position (jumps : JumpSequence) : ℤ :=
  jumps.foldl (λ pos jump => 
    if jump.direction then
      pos + jump.length
    else
      pos - jump.length) 0

/-- The length of the k-th jump -/
def jumpLength (k : ℕ) : ℕ := 2^k + 1

/-- Theorem stating that the flea can reach any natural number -/
theorem flea_can_reach_all_naturals :
  ∀ n : ℕ, ∃ jumps : JumpSequence, 
    (∀ k, k < jumps.length → (jumps.get ⟨k, by sorry⟩).length = jumpLength k) ∧
    position jumps = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flea_can_reach_all_naturals_l766_76609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_l766_76697

noncomputable section

/-- The set of possible α values -/
def alpha_set : Set ℝ := {-2, -1, -1/2, 1/2, 1, 2, 3}

/-- The power function -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- A function is decreasing on (0, +∞) if f(x) > f(y) for all 0 < x < y -/
def is_decreasing_on_pos (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → g x > g y

theorem unique_alpha :
  ∃! α, α ∈ alpha_set ∧ is_odd_function (f α) ∧ is_decreasing_on_pos (f α) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_l766_76697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l766_76605

noncomputable def f (x : ℝ) : ℝ := x^(-(5 : ℤ)) + 3 * Real.sin x

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = -5 * x^(-(6 : ℤ)) + 3 * Real.cos x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l766_76605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_range_of_x_plus_2y_l766_76630

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (m t : ℝ) : ℝ × ℝ := (m + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem for part (1)
theorem intersection_points_m_value 
  (A B : ℝ × ℝ) (m : ℝ) 
  (h1 : ∃ t1 t2 : ℝ, line_l m t1 = A ∧ line_l m t2 = B)
  (h2 : ∃ θ1 θ2 : ℝ, (curve_C θ1 * Real.cos θ1, curve_C θ1 * Real.sin θ1) = A ∧ 
                      (curve_C θ2 * Real.cos θ2, curve_C θ2 * Real.sin θ2) = B)
  (h3 : distance A B = Real.sqrt 14) :
  m = 1 ∨ m = 3 := by sorry

-- Theorem for part (2)
theorem range_of_x_plus_2y 
  (x y : ℝ) 
  (h : ∃ θ : ℝ, x = 2 + 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :
  2 - 2 * Real.sqrt 5 ≤ x + 2 * y ∧ x + 2 * y ≤ 2 + 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_m_value_range_of_x_plus_2y_l766_76630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_proof_l766_76602

/-- Calculates the correct average marks for a class given the following conditions:
  * The number of students in the class
  * The incorrect average marks
  * The incorrectly recorded marks for one student
  * The correct marks for that student
-/
def correct_average (num_students : ℕ) (incorrect_avg : ℚ) (incorrect_marks correct_marks : ℕ) : ℚ :=
  let incorrect_total := incorrect_avg * num_students
  let difference := (correct_marks : ℚ) - (incorrect_marks : ℚ)
  let correct_total := incorrect_total + difference
  correct_total / num_students

/-- Proves that the correct average marks for the given conditions is approximately 71.71 -/
theorem correct_average_proof :
  let result := correct_average 35 72 46 56
  (⌊result * 100⌋ : ℚ) / 100 = 71.71 := by
  sorry

#eval correct_average 35 72 46 56

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_average_proof_l766_76602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l766_76644

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Check if a triangle is right-angled -/
def isRightTriangle (p1 p2 p3 : Point) : Prop :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- The main theorem -/
theorem right_triangle_locations (A B : Point) (h : distance A B = 10) :
  ∃! (s : Finset Point), s.card = 8 ∧ 
    ∀ C ∈ s, isRightTriangle A B C ∧ triangleArea A B C = 20 := by
  sorry

#check right_triangle_locations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l766_76644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_distinct_six_dice_l766_76652

def throw_dice (n : ℕ) : ℕ := 6^n

def four_distinct_outcomes : ℕ :=
  -- Case 1: One number shows 3 times, three others show once each
  (6 * (Nat.choose 6 3) * (Nat.choose 5 3) * Nat.factorial 3) +
  -- Case 2: Two numbers show twice each, two others show once each
  ((Nat.choose 6 2) * (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 4 2) * Nat.factorial 2)

theorem probability_four_distinct_six_dice :
  (four_distinct_outcomes : ℚ) / (throw_dice 6 : ℚ) = 325 / 648 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_distinct_six_dice_l766_76652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_has_ten_pigs_l766_76678

/-- Represents Casey's water pumping scenario -/
structure WaterPumping where
  pump_rate : ℚ
  corn_rows : ℕ
  corn_plants_per_row : ℕ
  water_per_corn : ℚ
  water_per_pig : ℚ
  num_ducks : ℕ
  water_per_duck : ℚ
  pumping_time : ℚ

/-- Calculates the number of pigs Casey can water -/
def num_pigs (w : WaterPumping) : ℕ :=
  let total_water := w.pump_rate * w.pumping_time
  let corn_water := (w.corn_rows * w.corn_plants_per_row : ℚ) * w.water_per_corn
  let duck_water := (w.num_ducks : ℚ) * w.water_per_duck
  let remaining_water := total_water - corn_water - duck_water
  (remaining_water / w.water_per_pig).floor.toNat

/-- Theorem stating that Casey has 10 pigs -/
theorem casey_has_ten_pigs :
  let w : WaterPumping := {
    pump_rate := 3,
    corn_rows := 4,
    corn_plants_per_row := 15,
    water_per_corn := 1/2,
    water_per_pig := 4,
    num_ducks := 20,
    water_per_duck := 1/4,
    pumping_time := 25
  }
  num_pigs w = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_casey_has_ten_pigs_l766_76678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_initial_bag_red_balls_removed_correct_l766_76648

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  red : ℕ
  white : ℕ

/-- Calculates the probability of drawing a red ball -/
def probRed (bag : ColoredBalls) : ℚ :=
  bag.red / (bag.red + bag.white)

/-- Calculates the probability of drawing a white ball -/
def probWhite (bag : ColoredBalls) : ℚ :=
  bag.white / (bag.red + bag.white)

/-- The initial bag of balls -/
def initialBag : ColoredBalls := ⟨12, 6⟩

/-- The number of red balls removed to achieve the new probability -/
def redBallsRemoved : ℕ := 3

/-- The bag after removing some red balls -/
def newBag : ColoredBalls := ⟨initialBag.red - redBallsRemoved, initialBag.white⟩

theorem prob_red_initial_bag :
  probRed initialBag = 2/3 := by sorry

theorem red_balls_removed_correct :
  probWhite newBag = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_initial_bag_red_balls_removed_correct_l766_76648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l766_76624

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point (-1, 2) to the line 2x + y - 10 = 0 is 2√5 -/
theorem distance_point_to_line_example : distance_point_to_line (-1) 2 2 1 (-10) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l766_76624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_ratio_l766_76655

theorem rectangle_area_ratio (x : ℝ) (x_pos : x > 0) : 
  (x * (2 * x)) / ((3 * x) * (6 * x)) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_ratio_l766_76655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_prizes_l766_76651

/-- Given the price conditions for prizes A and B, prove the minimum total cost for 80 prizes --/
theorem min_cost_for_prizes (price_A price_B : ℚ) (total_prizes : ℕ) 
  (h1 : price_A + 2 * price_B = 64)
  (h2 : 2 * price_A + price_B = 56) 
  (h3 : total_prizes = 80) :
  ∃ min_cost : ℚ, min_cost = 1440 ∧
  ∀ a : ℚ, 0 < a → a < total_prizes → a ≤ 3 * (total_prizes - a) →
  price_A * a + price_B * (total_prizes - a) ≥ min_cost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_for_prizes_l766_76651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_l766_76662

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_cosine_l766_76662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_angles_l766_76603

/-- An isometry in 3D space -/
structure Isometry3D where
  apply : Point3D → Point3D
  -- Add necessary properties (e.g., distance preservation)

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Angle measure between three points -/
noncomputable def angle_measure (A B C : Point3D) : ℝ :=
  sorry

theorem tetrahedron_angles 
  (A B C D : Point3D)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (iso1 : Isometry3D)
  (iso2 : Isometry3D)
  (h1 : iso1.apply A = B ∧ iso1.apply B = A ∧ iso1.apply C = C ∧ iso1.apply D = D)
  (h2 : iso2.apply A = B ∧ iso2.apply B = C ∧ iso2.apply C = D ∧ iso2.apply D = A) :
  angle_measure A B C = 60 ∧ angle_measure D A C = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_angles_l766_76603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_bus_251_l766_76688

/-- The probability of event B occurring before event A, given their periods -/
noncomputable def probability_B_before_A (period_A period_B : ℝ) : ℝ :=
  (min period_A period_B)^2 / (2 * period_A * period_B)

/-- The periods of the two bus routes -/
def period_152 : ℝ := 5
def period_251 : ℝ := 7

/-- The theorem stating the probability of taking bus No. 251 -/
theorem probability_of_bus_251 :
  probability_B_before_A period_152 period_251 = 5 / 14 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_bus_251_l766_76688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_sum_range_l766_76608

/-- Given real numbers m and n, if the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle
    (x-1)² + (y-1)² = 1, then m + n is in the set (-∞, 2-2√2] ∪ [2+2√2, +∞) -/
theorem tangent_line_circle_sum_range (m n : ℝ) 
  (h : ∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 = 1) :
  m + n ∈ Set.Iic (2 - 2 * Real.sqrt 2) ∪ Set.Ici (2 + 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_sum_range_l766_76608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_of_g_l766_76650

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem two_zeros_of_g :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x when x ≥ 0
  ∃! (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ g z₁ = 0 ∧ g z₂ = 0 ∧ ∀ x, g x = 0 → x = z₁ ∨ x = z₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_of_g_l766_76650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_transform_f_l766_76634

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 2*x + 2)

-- Define the Fourier transform
noncomputable def fourier_transform (f : ℝ → ℝ) : ℝ → ℂ := sorry

-- State the given Fourier transform pair
axiom fourier_pair : fourier_transform (fun x => 1 / (x^2 + 1)) = 
  fun p => Complex.ofReal (Real.sqrt (Real.pi / 2)) * Complex.exp (-Complex.abs (Complex.ofReal p))

-- State the shifting property of Fourier transform
axiom fourier_shift (f : ℝ → ℝ) (a : ℝ) :
  fourier_transform (fun x => f (x - a)) = 
  fun p => Complex.exp (-i * Complex.ofReal p * Complex.ofReal a) * fourier_transform f p

-- Theorem to prove
theorem fourier_transform_f :
  fourier_transform f = 
  fun p => Complex.ofReal (Real.sqrt (Real.pi / 2)) * Complex.exp (-Complex.abs (Complex.ofReal p) + i * Complex.ofReal p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_transform_f_l766_76634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l766_76683

theorem max_t_value (x y z t : ℝ) :
  2*x^2 + 4*x*y + 3*y^2 - 2*x*z - 2*y*z + z^2 + 1 = t + Real.sqrt (y + z - t) →
  t ≤ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_t_value_l766_76683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_element_is_zero_l766_76654

/-- Represents the sequence described in the problem -/
def my_sequence : ℕ → ℕ
| 0 => 0
| n + 1 => if n % (Nat.sqrt (n + 1) + 1) = 0 then 1 else my_sequence n + 1

/-- The theorem states that the first element of the sequence is 0 -/
theorem first_element_is_zero : my_sequence 0 = 0 := by
  rfl

#eval my_sequence 0  -- This will evaluate to 0
#eval my_sequence 1  -- This will evaluate to 1
#eval my_sequence 2  -- This will evaluate to 2
#eval my_sequence 3  -- This will evaluate to 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_element_is_zero_l766_76654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l766_76675

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 120 % (k^2) = 24) : 150 % k = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l766_76675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_travel_time_l766_76632

/-- Calculates the total travel time given distance, speed, and break times -/
noncomputable def total_travel_time (distance : ℝ) (speed : ℝ) (lunch_break : ℝ) (bathroom_breaks : ℝ) : ℝ :=
  distance / speed + (lunch_break + bathroom_breaks) / 60

/-- Proves that the total travel time for Joan's trip is 9 hours -/
theorem joan_travel_time :
  let distance := (480 : ℝ)
  let speed := (60 : ℝ)
  let lunch_break := (30 : ℝ)
  let bathroom_breaks := (2 * 15 : ℝ)
  total_travel_time distance speed lunch_break bathroom_breaks = 9 := by
  unfold total_travel_time
  simp
  -- The actual proof steps would go here
  sorry

#check joan_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_travel_time_l766_76632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_high_voucher_range_of_a_l766_76666

/-- Representation of a lottery scheme -/
structure LotteryScheme where
  voucher_600 : ℚ  -- Probability of receiving a 600 voucher
  voucher_high : ℚ  -- Probability of receiving a voucher higher than 600
  voucher_low : ℚ   -- Probability of receiving a voucher lower than 600

/-- Definition of scheme one -/
def scheme_one : LotteryScheme :=
  { voucher_600 := 1/6,
    voucher_high := 1/9,
    voucher_low := 13/18 }

/-- Definition of scheme two -/
def scheme_two (a : ℚ) : LotteryScheme :=
  { voucher_600 := 1/4,
    voucher_high := 0,
    voucher_low := 3/4 }

/-- Theorem stating the probability of receiving a voucher of at least 600 -/
theorem probability_high_voucher :
  ∀ a : ℚ, 200 ≤ a → a < 600 →
  (1/2 * (scheme_one.voucher_600 + scheme_one.voucher_high +
          (scheme_two a).voucher_600 + (scheme_two a).voucher_high)) = 19/72 :=
by
  sorry

/-- Theorem stating the range of 'a' that satisfies E(X) ≥ 400 in scheme two -/
theorem range_of_a :
  ∀ a : ℚ, 200 ≤ a → a < 600 →
  (175 + a/2 ≥ 400 ↔ 450 ≤ a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_high_voucher_range_of_a_l766_76666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l766_76635

/-- A parabola in a 2D plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Three pairwise distinct parabolas in a plane --/
def ThreeParabolas : Type := { p : Fin 3 → Parabola // ∀ i j, i ≠ j → p i ≠ p j }

/-- The set of intersection points of two or more parabolas --/
def IntersectionPoints (p : ThreeParabolas) : Set (ℝ × ℝ) :=
  { point | ∃ i j, i ≠ j ∧ point ∈ { (x, y) | y = (p.val i).a * x^2 + (p.val i).b * x + (p.val i).c } ∩
                               { (x, y) | y = (p.val j).a * x^2 + (p.val j).b * x + (p.val j).c } }

/-- The theorem stating the maximum number of intersection points --/
theorem max_intersection_points :
  ∃ p : ThreeParabolas, ∀ q : ThreeParabolas, (IntersectionPoints q).ncard ≤ (IntersectionPoints p).ncard ∧
                                              (IntersectionPoints p).ncard = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l766_76635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owlna_luxuries_l766_76613

def minimum_with_all_luxuries (refrigerator television computer air_conditioner : ℝ) : Prop :=
  0 ≤ refrigerator ∧ refrigerator ≤ 1 ∧
  0 ≤ television ∧ television ≤ 1 ∧
  0 ≤ computer ∧ computer ≤ 1 ∧
  0 ≤ air_conditioner ∧ air_conditioner ≤ 1 →
  (max refrigerator (max television (max computer air_conditioner))) =
  (max refrigerator (max television (max computer air_conditioner)))

theorem owlna_luxuries :
  minimum_with_all_luxuries 0.7 0.75 0.65 0.95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_owlna_luxuries_l766_76613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l766_76684

theorem smallest_positive_angle (α : Real) : 
  (∃ (x y : Real), x = Real.sin (2 * π / 3) ∧ y = Real.cos (2 * π / 3) ∧ 
   x = Real.sin α ∧ y = Real.cos α ∧ 
   ∀ β, 0 < β ∧ β < 2 * π → (Real.sin β = x ∧ Real.cos β = y → α ≤ β)) → 
  α = 11 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_l766_76684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l766_76629

/-- Given an isosceles triangle ABC and a point P, prove the length of BC and the dot product of PA and PB -/
theorem isosceles_triangle_properties (A B C P : ℝ × ℝ × ℝ) : 
  (‖A - B‖ = 2) →  -- AB = 2
  (‖A - C‖ = 2) →  -- AC = 2
  ((B - A) • (B - C) = 2) →  -- BA • BC = 2
  (C - P = (1/2) • (C - A) - 2 • (C - B)) →  -- CP = (1/2)CA - 2CB
  (‖B - C‖ = 2 ∧ (P - A) • (P - B) = 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l766_76629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l766_76656

def x : ℕ → ℚ
  | 0 => 0  -- Add a case for 0
  | 1 => 6
  | 2 => 4
  | (n + 3) => (x (n + 2))^2 / x (n + 1) - 4 / x (n + 1)

theorem x_general_term : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → x n = 8 - 2 * n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l766_76656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l766_76646

/-- In a triangle ABC, if the ratio of cosines of angles B and C is equal to the negative ratio of side b to (2a + c), and B is between 0 and π, then B equals 2π/3. -/
theorem triangle_angle_measure (A B C : Real) (a b c : Real) : 
  0 < B → B < Real.pi →
  a > 0 → b > 0 → c > 0 →
  Real.cos B / Real.cos C = -b / (2*a + c) →
  B = 2*Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l766_76646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l766_76618

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 - 3 * x) - (x + 1) ^ 0

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -1 ∨ (x > -1 ∧ x ≤ 2/3)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l766_76618
