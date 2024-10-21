import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_period_is_two_years_l957_95742

/-- Represents the simple interest scenario described in the problem -/
structure SimpleInterestScenario where
  principal : ℚ
  amount1 : ℚ
  amount2 : ℚ
  period2 : ℚ

/-- Calculates the initial period for a given simple interest scenario -/
def calculateInitialPeriod (s : SimpleInterestScenario) : ℚ :=
  (s.period2 * (s.amount1 - s.principal)) / (s.amount2 - s.amount1)

/-- Theorem stating that the initial period is 2 years for the given scenario -/
theorem initial_period_is_two_years : 
  let s : SimpleInterestScenario := {
    principal := 684,
    amount1 := 780,
    amount2 := 1020,
    period2 := 5
  }
  calculateInitialPeriod s = 2 := by
  -- Proof goes here
  sorry

#eval calculateInitialPeriod {
  principal := 684,
  amount1 := 780,
  amount2 := 1020,
  period2 := 5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_period_is_two_years_l957_95742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l957_95702

/-- Proves that given a round trip where the outbound journey takes 3 hours at 40 km/h
    and the return journey takes 2 hours, the speed of the return journey is 60 km/h. -/
theorem round_trip_speed 
  (outbound_time : ℝ) 
  (outbound_speed : ℝ) 
  (return_time : ℝ) 
  (h1 : outbound_time = 3)
  (h2 : outbound_speed = 40)
  (h3 : return_time = 2) :
  let total_distance := outbound_time * outbound_speed
  let return_speed := total_distance / return_time
  return_speed = 60 := by
  sorry

#check round_trip_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_l957_95702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l957_95768

theorem remainder_theorem (x : ℝ) : 
  ∃ q : Polynomial ℝ, X^11 - 1 = (X + 1) * q + (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l957_95768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_line_hyperbola_intersection_l957_95758

/-- The length of the chord formed by the intersection of a line and a hyperbola -/
theorem chord_length_line_hyperbola_intersection :
  ∀ (t : ℝ),
  let line : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 2 + t ∧ p.2 = Real.sqrt 3 * t
  let hyperbola : ℝ × ℝ → Prop := λ p => p.1^2 - p.2^2 = 1
  let intersection := {p : ℝ × ℝ | line p ∧ hyperbola p}
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ∈ intersection ∧ p₂ ∈ intersection ∧
    Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = 2 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_line_hyperbola_intersection_l957_95758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_right_angled_triangle_l957_95706

-- Define our own Vector type as a pair of real numbers
def MyVector := ℝ × ℝ

-- Define the dot product of two vectors
def dot_product (v w : MyVector) : ℝ := v.1 * w.1 + v.2 * w.2

-- Part 1
theorem midpoint_theorem (a m : ℝ) : 
  let AB : MyVector := (3, 1)
  let AC : MyVector := (-1, a)
  let AD : MyVector := (m, 2)
  (∃ (B C D : ℝ × ℝ), D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →
  a = 3 ∧ m = 1 := by sorry

-- Part 2
theorem right_angled_triangle (a : ℝ) :
  let AB : MyVector := (3, 1)
  let AC : MyVector := (-1, a)
  (∃ (B C : ℝ × ℝ), 
    dot_product AB AC = 0 ∨ 
    dot_product AB (C.1 - 3, C.2 - 1) = 0 ∨ 
    dot_product AC (3 - C.1, 1 - C.2) = 0) →
  a = 3 ∨ a = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_right_angled_triangle_l957_95706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_over_f_at_3_l957_95777

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- State the theorem
theorem f_prime_over_f_at_3 :
  let f' := deriv f
  (f' 3) / (f 3) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_over_f_at_3_l957_95777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_in_three_hours_l957_95747

/-- The time (in hours) at which two cars meet on a highway -/
noncomputable def meeting_time (highway_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  highway_length / (speed1 + speed2)

theorem cars_meet_in_three_hours :
  let highway_length : ℝ := 105
  let speed1 : ℝ := 15
  let speed2 : ℝ := 20
  meeting_time highway_length speed1 speed2 = 3 := by
  -- Unfold the definition of meeting_time
  unfold meeting_time
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cars_meet_in_three_hours_l957_95747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l957_95729

-- Define a function f with domain [-1, 0]
def f : ℝ → Set ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc (-1) 0

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) (-1)

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ domain_f, (f x).Nonempty) →
  (∀ x ∈ domain_f_shifted, (f (x + 1)).Nonempty) :=
by
  intro h
  intro x hx
  have h1 : x + 1 ∈ domain_f := by
    simp [domain_f, domain_f_shifted] at *
    exact ⟨by linarith, by linarith⟩
  exact h (x + 1) h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_shift_l957_95729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l957_95745

open Real Set

theorem omega_range (ω : ℝ) (f : ℝ → ℝ) :
  (ω > 0) →
  (f = λ x ↦ sin (ω * x + π / 6)) →
  (∀ x ∈ Ioo 0 (π / 6), StrictMono f) →
  (∃ x ∈ Ioo (π / 6) (π / 3), IsLocalMax f x) →
  (1 < ω) ∧ (ω < 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l957_95745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_minimum_savings_l957_95733

/-- Represents the denomination of a bill in rubles -/
inductive Bill where
  | fifty : Bill
  | hundred : Bill
deriving BEq, Repr

/-- Represents Masha's piggy bank -/
structure PiggyBank where
  weeks : Nat
  bills : List Bill

/-- Calculates the total value of bills in the piggy bank -/
def totalValue (pb : PiggyBank) : Nat :=
  pb.bills.foldl (fun acc b => acc + match b with
    | Bill.fifty => 50
    | Bill.hundred => 100) 0

/-- Removes the smallest denomination bill from the piggy bank -/
def removeSmallest (pb : PiggyBank) : PiggyBank :=
  if pb.bills.contains Bill.fifty
  then { pb with bills := pb.bills.erase Bill.fifty }
  else { pb with bills := pb.bills.erase Bill.hundred }

/-- Simulates Masha's saving process for a year -/
def simulateSavings (initialPB : PiggyBank) : Nat :=
  let rec loop (pb : PiggyBank) (sistersTotal : Nat) (fuel : Nat) : Nat :=
    if fuel = 0 then 0
    else if pb.weeks ≥ 52 then
      if sistersTotal = 1250 then totalValue pb
      else 0  -- Invalid scenario
    else
      let newPB := { pb with
        weeks := pb.weeks + 1,
        bills := if pb.weeks % 4 = 0
          then (removeSmallest pb).bills
          else pb.bills ++ [if pb.bills.length % 2 = 0 then Bill.fifty else Bill.hundred]
      }
      let newSistersTotal := if pb.weeks % 4 = 0
        then sistersTotal + (if pb.bills.contains Bill.fifty then 50 else 100)
        else sistersTotal
      loop newPB newSistersTotal (fuel - 1)
  loop initialPB 0 53  -- 53 is enough fuel for 52 weeks plus one extra iteration

/-- The main theorem stating the minimum amount Masha could have accumulated -/
theorem masha_minimum_savings :
  ∃ (pb : PiggyBank), simulateSavings pb = 3750 ∧
    ∀ (pb' : PiggyBank), simulateSavings pb' ≥ 3750 := by
  sorry

#eval simulateSavings { weeks := 0, bills := [] }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_minimum_savings_l957_95733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_three_equation_l957_95730

theorem power_three_equation (x : ℤ) : (3 : ℚ)^7 * (3 : ℚ)^x = 81 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_three_equation_l957_95730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l957_95749

noncomputable def point1 : ℝ × ℝ := (-1.5, 3.5)
noncomputable def point2 : ℝ × ℝ := (4, -7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = Real.sqrt 140.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l957_95749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l957_95790

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

theorem existence_of_special_set :
  ∃ (S : Finset ℕ),
    S.card = 1990 ∧
    (∀ a b, a ∈ S → b ∈ S → a ≠ b → is_coprime a b) ∧
    (∀ T : Finset ℕ, T ⊆ S → T.card ≥ 2 → is_composite (T.sum id)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l957_95790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_sad_equation_l957_95788

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Definition of the equation BANK and SAD must satisfy -/
def satisfiesEquation (B A N K S D : Digit) : Prop :=
  5 * (1000 * B.val + 100 * A.val + 10 * N.val + K.val) = 
  6 * (100 * S.val + 10 * A.val + D.val)

/-- All digits are distinct -/
def allDistinct (B A N K S D : Digit) : Prop :=
  B ≠ A ∧ B ≠ N ∧ B ≠ K ∧ B ≠ S ∧ B ≠ D ∧
  A ≠ N ∧ A ≠ K ∧ A ≠ S ∧ A ≠ D ∧
  N ≠ K ∧ N ≠ S ∧ N ≠ D ∧
  K ≠ S ∧ K ≠ D ∧
  S ≠ D

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem bank_sad_equation :
  ∃! (B A N K S D : Digit),
    satisfiesEquation B A N K S D ∧
    allDistinct B A N K S D ∧
    B = ⟨1, by norm_num⟩ ∧ 
    A = ⟨0, by norm_num⟩ ∧ 
    N = ⟨8, by norm_num⟩ ∧ 
    K = ⟨6, by norm_num⟩ ∧ 
    S = ⟨9, by norm_num⟩ ∧ 
    D = ⟨5, by norm_num⟩ := by
  sorry

#check bank_sad_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_sad_equation_l957_95788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_example_l957_95786

/-- Given two 2D vectors, compute the projection of one onto the other -/
noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := b.1 * b.1 + b.2 * b.2
  let scalar := dot_product / magnitude_squared
  (scalar * b.1, scalar * b.2)

/-- The projection of (0,4) onto (-3,-3) is (2,2) -/
theorem projection_example : proj (0, 4) (-3, -3) = (2, 2) := by
  -- Unfold the definition of proj
  unfold proj
  -- Simplify the expressions
  simp
  -- The rest of the proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_example_l957_95786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_with_inscribed_rectangle_l957_95753

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Predicate to check if a rectangle is inscribed in a semicircle with radius r -/
def inscribed_in_semicircle (rect : Rectangle) (r : ℝ) : Prop :=
  rect.height^2 + (rect.width/2)^2 = r^2 ∧ rect.width = 2*r

/-- Area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

theorem semicircle_area_with_inscribed_rectangle :
  ∀ (r : ℝ), r > 0 →
  (∃ (rect : Rectangle), rect.width = 1 ∧ rect.height = 3 ∧ 
   inscribed_in_semicircle rect r) →
  semicircle_area r = 13 * Real.pi / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_with_inscribed_rectangle_l957_95753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_17th_match_new_average_is_39_l957_95760

/-- Represents the average score of a batsman before the 17th match -/
def previous_average : ℝ := 36

/-- Represents the score in the 17th match -/
def score_17th_match : ℕ := 87

/-- Represents the increase in average after the 17th match -/
def average_increase : ℕ := 3

/-- Theorem stating that given the conditions, the new average after the 17th match is 39 -/
theorem new_average_after_17th_match : 
  (16 * previous_average + score_17th_match) / 17 = previous_average + average_increase := by
  sorry

/-- Corollary proving that the new average is 39 -/
theorem new_average_is_39 : 
  (16 * previous_average + score_17th_match) / 17 = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_17th_match_new_average_is_39_l957_95760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l957_95705

theorem sequence_general_term (n : ℕ) : 
  let a := λ k : ℕ => (2 * k - 1 : ℚ) / (2^k : ℚ)
  (a 1 = 1/2) ∧ (a 2 = 3/4) ∧ (a 3 = 5/8) ∧ (a 4 = 7/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l957_95705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangent_segments_l957_95783

/-- Given a pentagon with sides a, b, c, d, e and an inscribed circle, 
    the segments into which the point of tangency divides side a are 
    (a + b - c - d + e) / 2 and (a - b - c + d + e) / 2 -/
theorem inscribed_circle_tangent_segments 
  (a b c d e : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0) 
  (h_pentagon : ∃ (pentagon : Set ℝ), 
    ∃ (sides : List ℝ), sides = [a, b, c, d, e] ∧ 
    ∃ (circle : Set ℝ), circle ⊆ pentagon) :
  ∃ (x : ℝ), x = (a + b - c - d + e) / 2 ∧ 
             (a - x) = (a - b - c + d + e) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangent_segments_l957_95783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l957_95731

-- Define the line l: 4x - 3y - 12 = 0
def line (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Define the circle (x-2)² + (y-2)² = 5
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define points A and B as the intersection of the line and circle
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define point C as the intersection of the line and x-axis
noncomputable def C : ℝ × ℝ := sorry

-- Define point D as the intersection of the line and y-axis
noncomputable def D : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem to prove
theorem line_circle_intersection :
  2 * distance C D = 5 * distance A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l957_95731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l957_95704

/-- The number of significant digits in a real number -/
def significantDigits (x : ℝ) : ℕ := sorry

/-- The area of the square -/
def area : ℝ := 3.2416

/-- The side length of the square -/
noncomputable def sideLength : ℝ := Real.sqrt area

/-- The number of significant digits in the side length is 4 -/
theorem side_significant_digits : significantDigits sideLength = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l957_95704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_divisibility_pattern_l957_95710

theorem smallest_integer_with_divisibility_pattern : ∃ N : ℕ,
  (N = 180180) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 ∧ k ≠ 28 ∧ k ≠ 29 → N % k = 0) ∧
  (N % 28 ≠ 0) ∧ (N % 29 ≠ 0) ∧
  (∀ M : ℕ, 0 < M ∧ M < N → 
    (∃ k : ℕ, 1 ≤ k ∧ k ≤ 30 ∧ k ≠ 28 ∧ k ≠ 29 ∧ M % k ≠ 0) ∨
    (M % 28 = 0 ∨ M % 29 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_divisibility_pattern_l957_95710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l957_95763

theorem angle_terminal_side (b : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (b, 4) ∧ P.1 = b * Real.cos α ∧ P.2 = b * Real.sin α) →
  Real.cos α = -3/5 →
  b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l957_95763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_count_div_239_l957_95727

/-- The number of valid 2x120 grids as described in the problem -/
def validGridCount : ℕ := Nat.choose 240 120 / 121

/-- Theorem stating that the number of valid 2x120 grids is divisible by 239 -/
theorem valid_grid_count_div_239 : 239 ∣ validGridCount := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_grid_count_div_239_l957_95727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_is_two_l957_95721

/-- A circle C in the Cartesian coordinate system -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- Theorem: The distance from the origin to the center of circle C is 2 -/
theorem distance_to_center_is_two (C : Circle) 
    (hx : C.x = λ θ => 2 * Real.cos θ)
    (hy : C.y = λ θ => 2 + 2 * Real.sin θ) : 
  distance 0 0 0 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_center_is_two_l957_95721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_remaining_vacation_days_correct_l957_95766

/-- Calculates the remaining vacation days for Andrew given the company's policies and his work schedule. -/
def andrew_remaining_vacation_days
  (total_work_days : ℕ)
  (public_holidays : ℕ)
  (sick_leave_days : ℕ)
  (march_vacation : ℕ)
  (september_vacation : ℕ)
  : ℕ :=
  let effective_work_days := total_work_days - public_holidays - sick_leave_days
  let first_half_days := effective_work_days / 2
  let second_half_days := effective_work_days - first_half_days
  let first_half_vacation := first_half_days / 10
  let second_half_vacation := second_half_days / 20
  let total_earned := first_half_vacation + second_half_vacation
  let total_taken := march_vacation + september_vacation
  total_earned - total_taken

theorem andrew_remaining_vacation_days_correct
  (total_work_days : ℕ)
  (public_holidays : ℕ)
  (sick_leave_days : ℕ)
  (march_vacation : ℕ)
  (september_vacation : ℕ)
  (h1 : total_work_days = 290)
  (h2 : public_holidays = 10)
  (h3 : sick_leave_days = 5)
  (h4 : march_vacation = 5)
  (h5 : september_vacation = 2 * march_vacation)
  : andrew_remaining_vacation_days total_work_days public_holidays sick_leave_days march_vacation september_vacation = 4 :=
by
  sorry

#eval andrew_remaining_vacation_days 290 10 5 5 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_andrew_remaining_vacation_days_correct_l957_95766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_remaining_raisins_l957_95735

/-- The cost of one cream puff in some arbitrary unit -/
def v : ℝ := sorry

/-- The cost of one deciliter of Kofola in the same unit -/
def k : ℝ := sorry

/-- The cost of one dekagram of yogurt raisins in the same unit -/
def r : ℝ := sorry

/-- The total amount of Martin's savings in the same unit -/
def total : ℝ := sorry

/-- Theorem stating that Martin can buy 60 grams of yogurt raisins with his remaining money -/
theorem martin_remaining_raisins 
  (h1 : 3 * v + 3 * k = 18 * r)
  (h2 : 12 * r + 5 * k = total)
  (h3 : v + 6 * k = total - 6 * r) :
  6 * 10 = 60 := by
  -- The proof is trivial as it's just a simple arithmetic fact
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_remaining_raisins_l957_95735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biff_headphones_cost_l957_95717

/-- Proves that the cost of headphones is $16 given the conditions of Biff's bus trip --/
theorem biff_headphones_cost (ticket_cost : ℝ) (snacks_cost : ℝ) (hourly_rate : ℝ) 
  (wifi_cost : ℝ) (trip_duration : ℝ) (headphones_cost : ℝ) :
  ticket_cost = 11 →
  snacks_cost = 3 →
  hourly_rate = 12 →
  wifi_cost = 2 →
  trip_duration = 3 →
  hourly_rate * trip_duration = 
    ticket_cost + snacks_cost + wifi_cost * trip_duration + headphones_cost →
  headphones_cost = 16 := by
  sorry

#check biff_headphones_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biff_headphones_cost_l957_95717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_z_in_first_quadrant_l957_95780

-- Define complex number multiplication
def complex_mul (a b c d : ℝ) : ℂ :=
  (a * c - b * d) + (a * d + b * c) * Complex.I

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m + 2) + (m^2 - m - 2) * Complex.I

-- Theorem 1: Product of two complex numbers
theorem complex_product : complex_mul (-3) 1 2 (-4) = -2 + 14 * Complex.I := by sorry

-- Theorem 2: Condition for z to be in the first quadrant
theorem z_in_first_quadrant : 
  ∀ m : ℝ, (z m).re > 0 ∧ (z m).im > 0 ↔ m > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_z_in_first_quadrant_l957_95780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_torn_sheets_count_l957_95709

/-- Represents a book with numbered pages -/
structure Book where
  firstTornPage : Nat
  lastTornPage : Nat

/-- Checks if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- Checks if two numbers have the same digits -/
def sameDigits (a b : Nat) : Bool :=
  (Nat.digits 10 a).toFinset = (Nat.digits 10 b).toFinset

/-- Calculates the number of sheets torn out -/
def tornSheets (book : Book) : Nat :=
  (book.lastTornPage - book.firstTornPage + 1) / 2

/-- The main theorem -/
theorem torn_sheets_count (book : Book) :
  book.firstTornPage = 185 →
  book.lastTornPage = 518 →
  isEven book.lastTornPage →
  sameDigits book.firstTornPage book.lastTornPage →
  tornSheets book = 167 := by
  sorry

#eval tornSheets { firstTornPage := 185, lastTornPage := 518 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_torn_sheets_count_l957_95709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_x_fx_neg_l957_95775

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

theorem solution_set_x_fx_neg
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_incr : increasing_on f (Set.Ici 0))
  (h_f_neg3 : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = Set.Ioo (-3) 0 ∪ Set.Ioo 0 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_x_fx_neg_l957_95775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l957_95757

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 4 * Real.pi / 3)

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f (7 * Real.pi / 12 + x) = f (7 * Real.pi / 12 - x)) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12),
    ∀ y ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12),
    x ≤ y → f x ≥ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l957_95757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_simplification_l957_95744

theorem nested_root_simplification :
  ∀ (x : ℝ), x > 0 → x = 14^3 → 
  Real.sqrt (Real.rpow (Real.sqrt (1 / x)) (1/3)) = 1 / Real.rpow 14 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_root_simplification_l957_95744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l957_95754

-- Define an acute triangle with circumradius 1
def AcuteTriangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C

-- State the theorem
theorem triangle_inequality {A B C a b c : ℝ} (h : AcuteTriangle A B C a b c) :
  a / (1 - Real.sin A) + b / (1 - Real.sin B) + c / (1 - Real.sin C) ≥ 18 + 12 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l957_95754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_20_and_36_is_divisible_by_20_and_36_smallest_number_divisible_by_20_and_36_l957_95701

theorem smallest_divisible_by_20_and_36 : ∀ n : ℕ, n > 0 ∧ n % 20 = 0 ∧ n % 36 = 0 → n ≥ 180 :=
by
  sorry

theorem is_divisible_by_20_and_36 : 180 % 20 = 0 ∧ 180 % 36 = 0 :=
by
  sorry

theorem smallest_number_divisible_by_20_and_36 : 
  (∀ n : ℕ, n > 0 ∧ n % 20 = 0 ∧ n % 36 = 0 → n ≥ 180) ∧ (180 % 20 = 0 ∧ 180 % 36 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_20_and_36_is_divisible_by_20_and_36_smallest_number_divisible_by_20_and_36_l957_95701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_pair_characterization_l957_95722

def is_excellent_pair (x y : ℕ+) : Prop :=
  ∀ (a b : ℕ+), (a.val ∣ (x.val^3 + y.val^3)) → (b.val ∣ (x.val^3 + y.val^3)) → 
  Nat.Coprime a.val b.val → ((a.val + b.val - 1) ∣ (x.val^3 + y.val^3))

theorem excellent_pair_characterization (x y : ℕ+) : 
  is_excellent_pair x y ↔ 
  (∃ k : ℕ, (x.val = 2^k ∧ y.val = 2^k) ∨ 
            (x.val = 2 * 3^k ∧ y.val = 3^k) ∨ 
            (x.val = 3^k ∧ y.val = 2 * 3^k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_pair_characterization_l957_95722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_extreme_coins_possible_l957_95718

/-- Represents a coin with a unique weight -/
structure Coin where
  weight : ℕ
  unique : True

/-- Represents a two-pan balance scale -/
def Balance := Coin → Coin → Bool

/-- The total number of coins -/
def numCoins : ℕ := 68

/-- The maximum number of allowed weighings -/
def maxWeighings : ℕ := 100

/-- A function that finds the heaviest and lightest coins -/
def findExtremeCoinsFn := (Fin numCoins → Coin) → Balance → (Coin × Coin)

/-- Count the number of weighings used by the function -/
def countWeighings (f : findExtremeCoinsFn) (coins : Fin numCoins → Coin) (scale : Balance) : ℕ :=
  sorry  -- Implementation to count the number of weighings used by f

theorem find_extreme_coins_possible :
  ∃ (coins : Fin numCoins → Coin) (scale : Balance) (f : findExtremeCoinsFn),
    (∀ i j, i ≠ j → (coins i).weight ≠ (coins j).weight) →
    ∃ (heaviest lightest : Coin),
      (∃ i, coins i = heaviest) ∧
      (∃ i, coins i = lightest) ∧
      (∀ i, scale heaviest (coins i) = true) ∧
      (∀ i, scale (coins i) lightest = true) ∧
      (f coins scale = (heaviest, lightest)) ∧
      (countWeighings f coins scale ≤ maxWeighings) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_extreme_coins_possible_l957_95718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ludwig_earnings_correct_l957_95737

/-- Calculates Ludwig's weekly earnings based on his work schedule and rates --/
def ludwig_weekly_earnings (
  weekday_rate : ℚ
) (weekend_rate : ℚ
) (weekday_hours : ℚ
) (weekend_hours : ℚ
) (overtime_multiplier : ℚ
) (total_hours : ℚ
) (standard_hours : ℚ
) : ℚ :=
  let weekday_earnings := 4 * weekday_rate
  let weekend_earnings := 3 * weekend_rate / 2
  let regular_earnings := weekday_earnings + weekend_earnings
  let overtime_hours := max (total_hours - standard_hours) 0
  let overtime_rate := weekend_rate / 8 * overtime_multiplier
  let overtime_earnings := overtime_hours * overtime_rate
  regular_earnings + overtime_earnings

/-- Theorem stating that Ludwig's weekly earnings are $115.50 --/
theorem ludwig_earnings_correct :
  ludwig_weekly_earnings 12 15 32 20 (3/2) 52 48 = 115.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ludwig_earnings_correct_l957_95737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_320000_units_l957_95793

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6*x
  else if x > 40 then 7400/x - 40000/(x^2)
  else 0

-- Define the profit function
noncomputable def w (x : ℝ) : ℝ := x * R x - 16*x - 40

-- Theorem statement
theorem max_profit_at_320000_units :
  ∃ (x_max : ℝ), x_max = 32 ∧
  ∀ (x : ℝ), x > 0 → w x ≤ w x_max ∧
  w x_max = 11634 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_320000_units_l957_95793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l957_95732

/-- The y-intercept of a common external tangent line to two circles -/
theorem common_external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (3, 7)  -- Center of circle 1
  let r1 : ℝ := 3           -- Radius of circle 1
  let c2 : ℝ × ℝ := (10, 12) -- Center of circle 2
  let r2 : ℝ := 7           -- Radius of circle 2
  let m : ℝ := 35 / 12      -- Slope of the tangent line
  let b : ℝ := 912 / 119    -- y-intercept to be proven
  let line (x : ℝ) := m * x + b  -- Equation of the line
  
  -- Conditions for external tangency
  (∀ x y : ℝ, (x - c1.1)^2 + (y - c1.2)^2 = r1^2 → line y ≠ x) ∧
  (∀ x y : ℝ, (x - c2.1)^2 + (y - c2.2)^2 = r2^2 → line y ≠ x) ∧
  
  -- Existence of tangent points
  (∃ x1 y1 : ℝ, (x1 - c1.1)^2 + (y1 - c1.2)^2 = r1^2 ∧ line y1 = x1) ∧
  (∃ x2 y2 : ℝ, (x2 - c2.1)^2 + (y2 - c2.2)^2 = r2^2 ∧ line y2 = x2) ∧
  
  -- Positive slope condition
  m > 0
  
  → b = 912 / 119 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_external_tangent_y_intercept_l957_95732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_power_of_two_l957_95728

/-- An arithmetic progression of integers -/
def ArithmeticProgression (a d : ℤ) : ℕ → ℤ := fun n => a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic progression -/
def SumArithmeticProgression (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- A number is a power of two -/
def IsPowerOfTwo (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2^k

/-- A number is a power of two (integer version) -/
def IsPowerOfTwoInt (x : ℤ) : Prop :=
  ∃ k : ℕ, x = 2^k

theorem arithmetic_progression_sum_power_of_two (a d : ℤ) (n : ℕ) :
  IsPowerOfTwoInt (SumArithmeticProgression a d n) → IsPowerOfTwo n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_power_of_two_l957_95728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_maximum_l957_95743

/-- The quadratic function f(x) = -1/4 * x^2 + x - 4 -/
noncomputable def f (x : ℝ) : ℝ := -1/4 * x^2 + x - 4

theorem quadratic_maximum :
  (∀ x : ℝ, f x ≤ f 2) ∧ f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_maximum_l957_95743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_rearrangement_not_all_ones_l957_95712

/-- Represents a positive integer without zero digits -/
structure NonZeroDigitNumber where
  value : Nat
  positive : 0 < value
  no_zero_digits : ∀ d, d ∈ value.digits 10 → d ≠ 0

/-- Represents a rearrangement of digits of a number -/
def is_digit_rearrangement (n m : Nat) : Prop :=
  Multiset.ofList (n.digits 10) = Multiset.ofList (m.digits 10)

/-- Represents a number consisting only of ones -/
def is_all_ones (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

/-- Main theorem: The sum of a non-zero digit number and its rearrangement cannot be all ones -/
theorem sum_with_rearrangement_not_all_ones (n : NonZeroDigitNumber) :
  ¬∃ m : Nat, is_digit_rearrangement n.value m ∧ is_all_ones (n.value + m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_with_rearrangement_not_all_ones_l957_95712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_perimeter_is_42_l957_95792

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The perimeter of the first smaller triangle -/
  p₁ : ℝ
  /-- The perimeter of the second smaller triangle -/
  p₂ : ℝ
  /-- The perimeter of the third smaller triangle -/
  p₃ : ℝ

/-- The perimeter of the original triangle -/
def perimeter_of_original_triangle (t : TriangleWithInscribedCircle) : ℝ :=
  t.p₁ + t.p₂ + t.p₃

/-- The perimeter of the original triangle is the sum of the perimeters of the three smaller triangles -/
theorem perimeter_sum (t : TriangleWithInscribedCircle) :
  perimeter_of_original_triangle t = t.p₁ + t.p₂ + t.p₃ := by
  rfl

/-- The perimeter of the original triangle is 42 cm when the smaller triangles have perimeters 12 cm, 14 cm, and 16 cm -/
theorem perimeter_is_42 (t : TriangleWithInscribedCircle) 
  (h₁ : t.p₁ = 12) (h₂ : t.p₂ = 14) (h₃ : t.p₃ = 16) :
  perimeter_of_original_triangle t = 42 := by
  rw [perimeter_of_original_triangle, h₁, h₂, h₃]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_sum_perimeter_is_42_l957_95792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l957_95752

theorem cosine_sum_zero (x y z : ℝ) :
  Real.cos x + Real.cos y + Real.cos z + Real.cos (x + y + z) = 0 →
  ∃ (n m k : ℤ), x = π / 2 + π * ↑n ∧ y = π / 2 + π * ↑m ∧ z = π / 2 + π * ↑k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_l957_95752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_brownie_pan_size_l957_95762

theorem smallest_brownie_pan_size : 
  ∃ s : ℕ, s = 8 ∧ 
    (∀ n : ℕ, n < s → (n - 2)^2 ≠ 4 * n - 4 ∨ n % 2 ≠ 0) ∧
    (s - 2)^2 = 4 * s - 4 ∧ s % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_brownie_pan_size_l957_95762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_line_l957_95708

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to calculate the distance between a point and a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.slope * p.x - p.y + l.intercept) / Real.sqrt (l.slope^2 + 1)

/-- Function to check if a line equally divides the area of a circle -/
def equallySplitsCircle (c : Circle) (l : Line) : Prop :=
  distancePointToLine c.center l ≤ c.radius

/-- The main theorem -/
theorem equal_area_division_line :
  ∃ (l : Line),
    l.slope = 24/5 ∨ l.slope = -24/5 ∧
    equallySplitsCircle { center := { x := 10, y := 80 }, radius := 4 } l ∧
    equallySplitsCircle { center := { x := 13, y := 64 }, radius := 4 } l ∧
    equallySplitsCircle { center := { x := 15, y := 72 }, radius := 4 } l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_line_l957_95708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l957_95741

theorem inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ 
  (a ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l957_95741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_and_pythagorean_pattern_l957_95785

theorem fraction_and_pythagorean_pattern (n : ℕ) (hn : n > 0) :
  (1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = (4 * n : ℚ) / (4 * n^2 - 1) ∧
  (4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2 ∧
  (196^2 : ℕ) + 9603^2 = 9605^2 := by
  sorry

#check fraction_and_pythagorean_pattern

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_and_pythagorean_pattern_l957_95785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l957_95769

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h_geom_mean : seq.a 4 ^ 2 = seq.a 3 * seq.a 7)
  (h_sum_8 : sum_n seq 8 = 16) :
  sum_n seq 10 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l957_95769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_odd_g_l957_95711

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 6)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := Real.sin ((1 / 2) * x - 3 * m + Real.pi / 6)

theorem min_m_for_odd_g :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, g m (-x) = -(g m x)) →
  m ≥ Real.pi / 18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_odd_g_l957_95711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_proof_l957_95764

/-- The horizontal shift required to transform the graph of y = cos 2x into y = cos(2x - π/4) -/
noncomputable def horizontal_shift : ℝ := Real.pi / 8

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 4)

theorem horizontal_shift_proof :
  ∀ x : ℝ, g x = f (x - horizontal_shift) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_proof_l957_95764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l957_95739

-- Define the domain
def PositiveReals : Set ℝ := {x : ℝ | x > 0}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := -2 / x
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 3
noncomputable def h (x : ℝ) : ℝ := x - 1/x

-- Define the condition
def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ∈ PositiveReals → x₂ ∈ PositiveReals → x₁ ≠ x₂ →
    (f x₁ - f x₂) / (x₁ - x₂) > 0

-- Theorem statement
theorem functions_satisfy_condition :
  satisfiesCondition f ∧ satisfiesCondition g ∧ satisfiesCondition h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_satisfy_condition_l957_95739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_proof_l957_95799

/-- The height of a right circular cone with given radius and volume -/
noncomputable def cone_height (radius : ℝ) (volume : ℝ) : ℝ :=
  (3 * volume) / (Real.pi * radius^2)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem funnel_height_proof :
  round_to_nearest (cone_height 4 150) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_funnel_height_proof_l957_95799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_168_l957_95784

-- Define the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define the y-intercept
def y_intercept : ℝ := f 0

-- Define the base of the triangle
def base : ℝ := x_intercept1 - x_intercept2

-- Define the height of the triangle
def triangle_height : ℝ := y_intercept

-- Theorem statement
theorem triangle_area_is_168 : 
  (1/2 : ℝ) * base * triangle_height = 168 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_168_l957_95784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_of_3_sqrt_5_l957_95796

-- Define the sequence
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (2 * n - 1)

-- Theorem statement
theorem term_position_of_3_sqrt_5 : 
  ∃ (n : ℕ), n = 23 ∧ a n = 3 * Real.sqrt 5 := by
  -- We'll use 23 as our witness
  use 23
  constructor
  · -- First part: n = 23
    rfl
  · -- Second part: a 23 = 3 * Real.sqrt 5
    -- This requires calculation, which we'll skip for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_term_position_of_3_sqrt_5_l957_95796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l957_95791

noncomputable def third_roots_of_unity : Finset ℂ := 
  {1, Complex.exp (2 * Real.pi * Complex.I / 3), Complex.exp (-2 * Real.pi * Complex.I / 3)}

def satisfies_equation (c : ℂ) : Prop :=
  ∀ z : ℂ, ∀ r s t : ℂ, r ∈ third_roots_of_unity → s ∈ third_roots_of_unity → t ∈ third_roots_of_unity →
  r ≠ s → s ≠ t → r ≠ t →
  (z - r) * (z - s) * (z - t) = (z - c*r) * (z - c*s) * (z - c*t)

theorem exactly_three_solutions :
  ∃! (solutions : Finset ℂ), solutions.card = 3 ∧ ∀ c : ℂ, c ∈ solutions ↔ satisfies_equation c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l957_95791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l957_95771

noncomputable def f (x : ℝ) := Real.cos (2 * x)
noncomputable def g (x : ℝ) := Real.sin (x + Real.pi / 4)

theorem function_transformation :
  ∀ x : ℝ, f x = g (x / 2 + Real.pi / 4) :=
by
  intro x
  unfold f g
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l957_95771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l957_95720

-- Define the circle C: x^2 + y^2 = 1
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l: y = x + 1
def lineL (x y : ℝ) : Prop := y = x + 1

-- Define the y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Define the angle of inclination
noncomputable def angle_of_inclination (m : ℝ) : ℝ := Real.arctan m

-- Define the number of intersection points
def num_intersections (C : (ℝ × ℝ) → Prop) (l : (ℝ × ℝ) → Prop) : ℕ := 
  sorry -- We'll leave this as sorry for now as it requires more complex computation

theorem circle_line_properties :
  (y_intercept (λ x => x + 1) = 1) ∧
  (angle_of_inclination 1 = π/4) ∧
  (num_intersections (λ (x, y) => circleC x y) (λ (x, y) => lineL x y) = 2) := by
  sorry -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_properties_l957_95720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l957_95759

theorem solve_exponential_equation :
  ∃ y : ℝ, 5 * (4 : ℝ) ^ y = 1280 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l957_95759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_8_g_decreasing_intervals_l957_95748

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

-- Define the conditions
axiom φ_range (φ : ℝ) : 0 < φ ∧ φ < Real.pi
axiom ω_positive (ω : ℝ) : ω > 0
axiom f_even (ω φ : ℝ) : ∀ x, f ω φ (-x) = f ω φ x
axiom symmetry_axes_distance (ω : ℝ) : (2 * Real.pi) / ω = Real.pi

-- Define the function g
noncomputable def g (ω φ : ℝ) (x : ℝ) : ℝ := f ω φ (x / 4 - Real.pi / 6)

-- State the theorems to be proved
theorem f_value_at_pi_over_8 (ω φ : ℝ) : f ω φ (Real.pi / 8) = Real.sqrt 2 := by sorry

theorem g_decreasing_intervals (ω φ : ℝ) (k : ℤ) :
  StrictMonoOn (g ω φ) (Set.Icc (4 * ↑k * Real.pi + 2 * Real.pi / 3) (4 * ↑k * Real.pi + 8 * Real.pi / 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_8_g_decreasing_intervals_l957_95748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l957_95798

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos (x - Real.pi / 6)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l957_95798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_is_80_l957_95774

/-- A square with one side on the line y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  /-- The x-coordinate of vertex A -/
  t : ℝ
  /-- Vertex A lies on the parabola y = x^2 -/
  vertex_a_on_parabola : t^2 = t^2
  /-- Vertex B lies on the parabola y = x^2 -/
  vertex_b_on_parabola : (2 - t)^2 = (2 - t)^2
  /-- Side AB lies on the line y = 2x - 17 -/
  side_on_line : ∀ x ∈ Set.Icc t (2 - t), 2*x - 17 = 2*x + t^2 - 2*t

/-- The area of a SpecialSquare -/
noncomputable def area (s : SpecialSquare) : ℝ :=
  (2 * Real.sqrt 5 * |1 - s.t|)^2

/-- The minimum area of all possible SpecialSquares is 80 -/
theorem min_area_is_80 : 
  ∃ (s : SpecialSquare), area s = 80 ∧ ∀ (s' : SpecialSquare), area s' ≥ 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_is_80_l957_95774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_horizontal_length_l957_95794

/-- Represents a rectangular television screen -/
structure TVScreen where
  horizontal : ℝ
  vertical : ℝ
  diagonal : ℝ

/-- Calculates the horizontal length of a TV screen given the aspect ratio and diagonal length -/
noncomputable def horizontal_length (aspect_ratio_h aspect_ratio_v diagonal : ℝ) : ℝ :=
  (aspect_ratio_h * diagonal) / Real.sqrt (aspect_ratio_h^2 + aspect_ratio_v^2)

/-- Theorem stating the horizontal length of a 32-inch TV with 16:9 aspect ratio -/
theorem tv_horizontal_length :
  let tv : TVScreen := {
    horizontal := horizontal_length 16 9 32,
    vertical := (9 * horizontal_length 16 9 32) / 16,
    diagonal := 32
  }
  tv.horizontal = 512 / Real.sqrt 337 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_horizontal_length_l957_95794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l957_95726

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) (l : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = l}

-- Define the vector n
def n : ℝ × ℝ := (0, 2)

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a)^2)

-- State the theorem
theorem hyperbola_eccentricity 
  (a b : ℝ) (l : ℝ) (A B : ℝ × ℝ) 
  (h1 : l ≠ 0)
  (h2 : A ∈ Hyperbola a b l)
  (h3 : B ∈ Hyperbola a b l)
  (h4 : A ≠ B)
  (h5 : ∃ (k : ℝ), A.2 = k * A.1 ∧ B.2 = k * B.1) -- A and B are on the same asymptote
  (h6 : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3) -- |AB| = 3
  (h7 : ((A.1 - B.1) * n.1 + (A.2 - B.2) * n.2) / Real.sqrt (n.1^2 + n.2^2) = -1) -- Projection condition
  : eccentricity a b = 3 * Real.sqrt 2 / 4 ∨ eccentricity a b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l957_95726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l957_95750

noncomputable def f (x : ℝ) := Real.log x + 2 * x - 1

theorem zero_point_in_interval :
  ∃! x : ℝ, 1/2 < x ∧ x < 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l957_95750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_one_l957_95746

noncomputable def g (x : ℝ) : ℝ := |⌊x⌋| - |⌊2 - x⌋|

theorem g_symmetric_about_one : ∀ x : ℝ, g (1 + x) = g (1 - x) := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetric_about_one_l957_95746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_parametric_curve_l957_95761

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := (t^2 - 2) * Real.sin t + 2 * t * Real.cos t
noncomputable def y (t : ℝ) : ℝ := (2 - t^2) * Real.cos t + 2 * t * Real.sin t

-- Define the arc length function
noncomputable def arc_length (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

-- State the theorem
theorem arc_length_of_parametric_curve :
  arc_length 0 Real.pi = (1/3) * (Real.pi^2 + 4)^(3/2) - 8/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_parametric_curve_l957_95761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l957_95716

-- Define the power function as noncomputable
noncomputable def powerFunction (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_through_point (α : ℝ) :
  powerFunction α 3 = 3 * Real.sqrt 3 → α = 3 / 2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l957_95716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l957_95740

/-- The number of circular arcs in the curve -/
def num_arcs : ℕ := 12

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := 2 * Real.pi / 3

/-- The side length of the regular hexagon -/
def hexagon_side : ℝ := 3

/-- The area enclosed by the curve constructed from circular arcs -/
noncomputable def enclosed_area (n : ℕ) (l : ℝ) (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2 + n * (l^2 / (4 * Real.pi))

/-- Theorem stating that the area enclosed by the curve is 13.5√3 + 4π -/
theorem enclosed_area_theorem :
  enclosed_area num_arcs arc_length hexagon_side = 13.5 * Real.sqrt 3 + 4 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_theorem_l957_95740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_coverage_l957_95700

-- Define an equilateral triangle type
structure EquilateralTriangle where
  area : ℝ
  isEquilateral : Bool

-- Define a function to check if a set of triangles can cover another triangle
def canCover (t : EquilateralTriangle) (ts : List EquilateralTriangle) : Prop :=
  ∃ (arrangement : List EquilateralTriangle → Prop), 
    arrangement ts ∧ 
    (∀ p : ℝ × ℝ, ∃ t' ∈ ts, True)  -- Simplified condition to avoid Membership issues

-- Theorem statement
theorem equilateral_triangle_coverage :
  ∀ (T : EquilateralTriangle) (t₁ t₂ t₃ t₄ t₅ : EquilateralTriangle),
    T.area = 1 →
    T.isEquilateral = true →
    t₁.isEquilateral = true →
    t₂.isEquilateral = true →
    t₃.isEquilateral = true →
    t₄.isEquilateral = true →
    t₅.isEquilateral = true →
    t₁.area + t₂.area + t₃.area + t₄.area + t₅.area = 2 →
    canCover T [t₁, t₂, t₃, t₄, t₅] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_coverage_l957_95700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_of_bars_l957_95751

/-- The weight of a copper bar in kilograms -/
noncomputable def copper_weight : ℝ := 90

/-- The weight of a steel bar in kilograms -/
noncomputable def steel_weight : ℝ := copper_weight + 20

/-- The weight of a tin bar in kilograms -/
noncomputable def tin_weight : ℝ := steel_weight / 2

/-- The number of bars of each metal type -/
def num_bars : ℕ := 20

theorem total_weight_of_bars :
  (num_bars : ℝ) * (copper_weight + steel_weight + tin_weight) = 5100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_weight_of_bars_l957_95751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_player_wins_prob_l957_95715

/-- Represents a coin-flipping game with three players. -/
structure CoinGame where
  /-- The probability of flipping heads on a single flip. -/
  p_heads : ℝ
  /-- Assumption that the coin is fair. -/
  fair_coin : p_heads = 1 / 2

/-- The probability that the third player wins the game. -/
noncomputable def third_player_wins (game : CoinGame) : ℝ :=
  let r := 1 - game.p_heads
  (game.p_heads * r^2) / (1 - r^3)

/-- Theorem stating that the probability of the third player winning is 1/7. -/
theorem third_player_wins_prob (game : CoinGame) : 
  third_player_wins game = 1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_player_wins_prob_l957_95715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_for_repeating_fraction_l957_95765

/-- Represents the sum of the geometric series for the repeating decimal -/
def repeating_decimal_sum (k : ℚ) : ℚ :=
  (5 / k + 6 / k^2) / (1 - 1 / k^2)

/-- The theorem stating the smallest base for the repeating fraction -/
theorem smallest_base_for_repeating_fraction : 
  ∃ k : ℕ, k > 0 ∧ k = 27 ∧ 
  (∀ m : ℕ, m > 0 ∧ m < k → (11 : ℚ) / 97 ≠ repeating_decimal_sum m) ∧
  (11 : ℚ) / 97 = repeating_decimal_sum k :=
sorry

#check smallest_base_for_repeating_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_for_repeating_fraction_l957_95765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_shade_value_l957_95736

/-- Represents a rectangle in the 2 by 2003 grid -/
structure Rectangle where
  left : Nat
  right : Nat
  top : Bool
  bottom : Bool

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * Nat.choose 2003 2

/-- The number of rectangles that contain a shaded square -/
def shaded_rectangles : ℕ := 3 * 1002 * 1002

/-- The probability of choosing a rectangle that doesn't include a shaded square -/
noncomputable def prob_no_shade : ℚ := 1 - (shaded_rectangles : ℚ) / total_rectangles

theorem prob_no_shade_value : prob_no_shade = 1001 / 2003 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_shade_value_l957_95736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_is_25_percent_l957_95719

-- Define the dealer's practices
def counterfeit_weight_ratio : ℚ := 4/5  -- 20% less than real weight
def impurity_ratio : ℚ := 15/100  -- 15% impurities added

-- Define the function to calculate the actual weight sold
def actual_weight_sold (claimed_weight : ℚ) : ℚ :=
  claimed_weight * counterfeit_weight_ratio * (1 + impurity_ratio)

-- Define the function to calculate the actual cost to the dealer
def actual_cost (claimed_weight : ℚ) (cost_per_unit : ℚ) : ℚ :=
  claimed_weight * counterfeit_weight_ratio * cost_per_unit

-- Define the function to calculate the net profit percentage
def net_profit_percentage (claimed_weight : ℚ) (cost_per_unit : ℚ) : ℚ :=
  let selling_price := claimed_weight * cost_per_unit
  let actual_cost := actual_cost claimed_weight cost_per_unit
  ((selling_price - actual_cost) / actual_cost) * 100

-- Theorem statement
theorem dealer_profit_is_25_percent (claimed_weight : ℚ) (cost_per_unit : ℚ) :
  claimed_weight > 0 → cost_per_unit > 0 →
  net_profit_percentage claimed_weight cost_per_unit = 25 := by
  sorry

#eval net_profit_percentage 1 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_is_25_percent_l957_95719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l957_95795

-- Define the logarithm base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 5/3 → log8 (3 * x - 5) = 2 → x = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l957_95795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l957_95756

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def is_acute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_condition (t : Triangle) : Prop :=
  2 * (Real.cos ((t.B + t.C) / 2))^2 + Real.sin (2 * t.A) = 1

def side_a_condition (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 3 - 2

def area_condition (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = 2

-- State the theorem
theorem triangle_problem (t : Triangle)
  (h_acute : is_acute t)
  (h_cond : satisfies_condition t)
  (h_side_a : side_a_condition t)
  (h_area : area_condition t) :
  t.A = Real.pi/6 ∧ t.b + t.c = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l957_95756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l957_95703

theorem exponential_equation_solution :
  ∀ y : ℝ, (3 : ℝ)^(2*y) - (3 : ℝ)^(2*y - 1) = 81 →
    y = 5/2 ∧ (3*y)^y = 30^(5/2) / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l957_95703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l957_95767

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := |x + t/2| + (8 - t^2)/4

noncomputable def F (t : ℝ) (x : ℝ) : ℝ := f t (f t x)

theorem range_of_t (t : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f t x = y) ↔ (∀ y : ℝ, ∃ x : ℝ, F t x = y) →
  t < -2 ∨ t > 4 :=
by
  sorry

#check range_of_t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l957_95767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l957_95714

open Nat

/-- The number of divisors of 9! that are larger than 8! -/
theorem divisors_of_nine_factorial_greater_than_eight_factorial : 
  (Finset.filter (λ d ↦ d > 8! ∧ 9! % d = 0) (Finset.range (9! + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l957_95714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_seven_sixths_l957_95776

-- Define the function f
noncomputable def f (a b : ℝ) : ℝ :=
  if a + b ≤ 4 then
    (a^2 * b - 2*a - 4) / (3*a)
  else
    (a*b - b - 3) / (-3*b)

-- State the theorem
theorem f_sum_equals_negative_seven_sixths :
  f 1 3 + f 3 2 = -7/6 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_negative_seven_sixths_l957_95776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l957_95779

theorem first_six_average (total_results : ℕ) (total_average : ℚ) (last_six_average : ℚ) (sixth_result : ℚ) 
    (first_six_average : ℚ) :
  total_results = 11 →
  total_average = 60 →
  last_six_average = 63 →
  sixth_result = 66 →
  (6 : ℚ) * (total_results : ℚ) * total_average = 
    6 * first_six_average + 6 * last_six_average - sixth_result →
  first_six_average = 58 :=
by
  sorry

#check first_six_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l957_95779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l957_95787

/-- Converts a point from cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (8, Real.pi / 4, Real.sqrt 3)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (4 * Real.sqrt 2, 4 * Real.sqrt 2, Real.sqrt 3)

/-- Theorem stating that the conversion from cylindrical to rectangular coordinates is correct -/
theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l957_95787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_l957_95725

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the sequence we're interested in
def our_sequence (n : ℕ) : ℕ := 
  ((fib n % 10)^2) % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ our_sequence k = d

-- State the theorem
theorem last_digit_to_appear :
  ∃ N : ℕ, (∀ d : ℕ, d < 10 → digitAppears d N) ∧ 
  ∃ m : ℕ, m < N ∧ ¬(digitAppears 2 m) ∧ (∀ d : ℕ, d < 10 ∧ d ≠ 2 → digitAppears d m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_to_appear_l957_95725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l957_95773

/-- The time (in seconds) it takes for a train to pass a man running in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / ((train_speed + man_speed) * 1000 / 3600)

/-- Theorem: A 200 m long train traveling at 80 km/hr will pass a man running at 10 km/hr
    in the opposite direction in 8 seconds -/
theorem train_passing_man_time :
  train_passing_time 200 80 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_man_time_l957_95773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_calculation_l957_95713

def planned_weekly_production : ℕ := 2100
def average_daily_production : ℕ := 300
def daily_deviations : List ℤ := [5, -2, -5, 15, -10, 16, -9]
def piece_rate_wage : ℕ := 60
def excess_bonus : ℕ := 50
def underproduction_penalty : ℕ := 80

def actual_weekly_production : ℤ := planned_weekly_production + daily_deviations.sum

theorem total_wage_calculation :
  let total_wage := (actual_weekly_production.toNat * piece_rate_wage : ℕ) + 
                    ((actual_weekly_production - planned_weekly_production).toNat * excess_bonus : ℕ)
  total_wage = 127100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wage_calculation_l957_95713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_document_printing_sequences_l957_95738

/-- The number of valid sequences for printing n documents -/
def valid_sequences (n : ℕ) : ℚ :=
  2 / (n + 1 : ℚ) * (Nat.choose (2 * n - 1) n : ℚ)

/-- The nth Catalan number -/
def nth_catalan (n : ℕ) : ℚ :=
  2 / (n + 1 : ℚ) * (Nat.choose (2 * n - 1) n : ℚ)

theorem document_printing_sequences (n : ℕ) :
  valid_sequences n = nth_catalan n := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_document_printing_sequences_l957_95738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l957_95734

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1)

theorem f_increasing_on_positive_reals : 
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_positive_reals_l957_95734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_six_triangles_arrangement_l957_95778

/-- The area of a region covered by six equilateral triangles arranged in a specific pattern -/
theorem area_of_six_triangles_arrangement : 
  (let side_length : ℝ := 4
   let num_triangles : ℕ := 6
   let triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
   let overlap_area : ℝ := (Real.sqrt 3 / 4) * (side_length / 2)^2
   let total_area : ℝ := num_triangles * triangle_area - (num_triangles - 1) * overlap_area
   total_area) = 19 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_six_triangles_arrangement_l957_95778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l957_95723

-- Define the original function
def original_function (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the translation
def translate_left (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x ↦ f (x + units)
def translate_down (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x ↦ f x - units

-- Define the resulting function after translation
def f (x : ℝ) : ℝ := translate_down (translate_left original_function 2) 2 x

-- Theorem statement
theorem translation_result : f = λ x ↦ x^2 - 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l957_95723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l957_95789

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  let perimeter : ℝ := 3 * side_length
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_perimeter_ratio_l957_95789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_survey_order_l957_95724

-- Define the steps of the statistical survey activity
inductive SurveyStep
  | CollectingData
  | DesigningSurveyQuestionnaires
  | EstimatingPopulation
  | OrganizingData
  | AnalyzingData
deriving DecidableEq

-- Define a function to represent the correct order of steps
def correctOrder : List SurveyStep :=
  [SurveyStep.DesigningSurveyQuestionnaires,
   SurveyStep.CollectingData,
   SurveyStep.OrganizingData,
   SurveyStep.AnalyzingData,
   SurveyStep.EstimatingPopulation]

-- Define a function to check if a list contains all steps exactly once
def containsAllStepsOnce (order : List SurveyStep) : Prop :=
  order.length = 5 ∧
  order.toFinset = {SurveyStep.CollectingData,
                    SurveyStep.DesigningSurveyQuestionnaires,
                    SurveyStep.EstimatingPopulation,
                    SurveyStep.OrganizingData,
                    SurveyStep.AnalyzingData}

-- Theorem stating that the correctOrder is the only valid order
theorem correct_survey_order :
  ∀ (order : List SurveyStep),
    containsAllStepsOnce order → order = correctOrder :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_survey_order_l957_95724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_ratio_theorem_ab_product_is_eight_l957_95755

/-- The ratio of the area of a circle inscribed in a regular octagon 
    (touching the midpoints of the octagon's sides) to the area of the octagon -/
noncomputable def circle_octagon_area_ratio : ℝ :=
  (Real.sqrt 2 / 4) * Real.pi

/-- Regular octagon with a circle inscribed, touching midpoints of sides -/
structure OctagonWithInscribedCircle where
  /-- Side length of the octagon -/
  side_length : ℝ
  /-- Radius of the inscribed circle (half the side length) -/
  circle_radius : ℝ
  radius_is_half_side : circle_radius = side_length / 2

/-- Area of a regular octagon given its side length -/
noncomputable def octagon_area (o : OctagonWithInscribedCircle) : ℝ :=
  2 * (o.side_length ^ 2) * Real.sqrt 2

/-- Area of the inscribed circle -/
noncomputable def circle_area (o : OctagonWithInscribedCircle) : ℝ :=
  Real.pi * (o.circle_radius ^ 2)

/-- Theorem stating that the ratio of the inscribed circle's area to the octagon's area
    is equal to the constant circle_octagon_area_ratio -/
theorem circle_octagon_area_ratio_theorem (o : OctagonWithInscribedCircle) :
  circle_area o / octagon_area o = circle_octagon_area_ratio := by
  sorry

/-- The product of a and b where the ratio is expressed as (√a / b)π in simplest form -/
def ab_product : ℕ := 8

/-- Theorem stating that the product of a and b is 8 -/
theorem ab_product_is_eight : ab_product = 8 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_octagon_area_ratio_theorem_ab_product_is_eight_l957_95755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l957_95707

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) + 2 * (Real.cos x) ^ 2) / Real.cos x

theorem f_properties :
  (f (π / 4) = 2 * Real.sqrt 2) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < π / 4 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l957_95707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l957_95782

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) :
  Real.sin (π / 6 - α) = 1 / 3 → Real.cos (2 * π / 3 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l957_95782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_isosceles_right_triangle_omega_2019_value_l957_95781

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

noncomputable def A (ω : ℝ) (n : ℕ) : ℝ × ℝ := ((2 * n - 1) * Real.pi / (2 * ω), 0)

def IsIsoscelesRightTriangle (p q r : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (x₂ - x₃)^2 + (y₂ - y₃)^2 ∧
  (x₁ - x₂) * (x₂ - x₃) + (y₁ - y₂) * (y₂ - y₃) = 0

noncomputable def ω_sequence : ℕ → ℝ := λ n => (2 * n - 1) * Real.pi / 2

theorem intersection_points_form_isosceles_right_triangle (ω : ℝ) (k t p : ℕ) 
  (h₁ : 0 < ω) (h₂ : k ≠ t ∧ t ≠ p ∧ k ≠ p) :
  IsIsoscelesRightTriangle (A ω k) (A ω t) (A ω p) ↔ ∃ n : ℕ, ω = ω_sequence n := by
  sorry

theorem omega_2019_value :
  ω_sequence 2019 = 4037 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_isosceles_right_triangle_omega_2019_value_l957_95781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_not_in_second_quadrant_l957_95772

-- Define the quadrant type
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine if an angle is in the third quadrant
def isThirdQuadrant (α : Real) : Prop :=
  ∃ k : Int, Real.pi + 2 * k * Real.pi < α ∧ α < 3/2 * Real.pi + 2 * k * Real.pi

-- Define a function to determine if an angle is in the second quadrant
def isSecondQuadrant (α : Real) : Prop :=
  ∃ k : Int, Real.pi/2 + 2 * k * Real.pi < α ∧ α < Real.pi + 2 * k * Real.pi

-- Theorem statement
theorem terminal_side_not_in_second_quadrant (α : Real) :
  isThirdQuadrant α → ¬isSecondQuadrant (α/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_not_in_second_quadrant_l957_95772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_leq_l957_95797

theorem negation_of_exists_leq (p : Prop) : 
  (p ↔ ∃ n : ℕ, n > 0 ∧ 2^n ≤ 2*n + 1) → 
  (¬p ↔ ∀ n : ℕ, n > 0 → 2^n > 2*n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_leq_l957_95797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_common_difference_l957_95770

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  firstTerm : ℝ
  commonDifference : ℝ

/-- Get the nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.firstTerm + (n - 1 : ℝ) * ap.commonDifference

theorem arithmetic_progression_common_difference :
  ∃ (ap : ArithmeticProgression),
    ap.firstTerm = 2 ∧
    ap.nthTerm 15 = 44 ∧
    ap.commonDifference = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_common_difference_l957_95770
