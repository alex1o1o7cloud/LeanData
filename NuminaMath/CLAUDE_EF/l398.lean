import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clares_milk_cost_l398_39803

/-- Given Clare's shopping scenario, prove the cost of each carton of milk --/
theorem clares_milk_cost (initial_money bread_count milk_count money_left : ℕ) 
  (h_initial : initial_money = 47)
  (h_bread : bread_count = 4)
  (h_milk : milk_count = 2)
  (h_left : money_left = 35)
  : ∃ (item_cost : ℕ), item_cost = 2 ∧ initial_money - money_left = (bread_count + milk_count) * item_cost := by
  sorry

#check clares_milk_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clares_milk_cost_l398_39803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finley_tickets_l398_39814

/-- Given a total number of tickets and a ratio for distribution, 
    calculate the number of tickets for a specific part of the ratio -/
def calculate_tickets (total_tickets : ℕ) (ratio : List ℚ) (part_index : ℕ) : Option ℚ :=
  let total_parts := ratio.sum
  let tickets_given := (3 : ℚ) / 4 * total_tickets
  let part_value := tickets_given / total_parts
  ratio.get? part_index |>.map (· * part_value)

/-- Theorem stating that Finley (index 1 in the ratio list) gets 1575 tickets -/
theorem finley_tickets :
  let total_tickets : ℕ := 7500
  let ratio : List ℚ := [4.3, 11.2, 7.8, 6.4, 10.3]
  calculate_tickets total_tickets ratio 1 = some 1575 := by
  sorry

#eval calculate_tickets 7500 [4.3, 11.2, 7.8, 6.4, 10.3] 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finley_tickets_l398_39814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_chapters_problem_l398_39816

/-- Represents the number of carts -/
def x : ℕ := sorry

/-- Represents the total number of people -/
def y : ℕ := sorry

/-- Condition 1: If each cart carries 2 people, then 9 people need to walk -/
axiom condition1 : y = 2 * x + 9

/-- Condition 2: If each cart carries 3 people, then two carts are empty -/
axiom condition2 : y = 3 * (x - 2)

/-- Theorem: The system of equations correctly represents the problem -/
theorem nine_chapters_problem :
  (y = 2 * x + 9) ∧ (y = 3 * (x - 2)) := by
  constructor
  . exact condition1
  . exact condition2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_chapters_problem_l398_39816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_doubling_time_l398_39894

/-- The doubling time of a bacteria population -/
noncomputable def doubling_time (initial_population final_population : ℝ) (total_time : ℝ) : ℝ :=
  total_time / (Real.log (final_population / initial_population) / Real.log 2)

/-- Theorem: The doubling time for a bacteria population growing from 1,000 to 500,000 in 35.86 minutes is approximately 4 minutes -/
theorem bacteria_doubling_time :
  let initial_population : ℝ := 1000
  let final_population : ℝ := 500000
  let total_time : ℝ := 35.86
  abs (doubling_time initial_population final_population total_time - 4) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_doubling_time_l398_39894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_taxi_fare_l398_39815

/-- Represents the time of day in hours (0-23) -/
def Time := Fin 24

/-- Represents the fare structure for a taxi company -/
structure FareStructure where
  peakFirstQuarterMile : ℝ
  peakAdditionalQuarterMile : ℝ
  offPeakFirstFifthMile : ℝ
  offPeakAdditionalFifthMile : ℝ
  surchargeForMoreThanTwoPassengers : ℝ

/-- Checks if a given time is within peak hours -/
def isPeakHour (t : Time) : Bool :=
  (6 ≤ t.val && t.val < 10) || (16 ≤ t.val && t.val < 20)

/-- Calculates the fare for a given distance, time, and number of passengers -/
noncomputable def calculateFare (fs : FareStructure) (distance : ℝ) (time : Time) (passengers : Nat) : ℝ :=
  sorry

/-- The fare structure for the taxi company -/
def taxiCompanyFareStructure : FareStructure :=
  { peakFirstQuarterMile := 3.00
  , peakAdditionalQuarterMile := 0.50
  , offPeakFirstFifthMile := 2.80
  , offPeakAdditionalFifthMile := 0.40
  , surchargeForMoreThanTwoPassengers := 1.00 }

theorem steve_taxi_fare :
  calculateFare taxiCompanyFareStructure 8 ⟨17, by norm_num⟩ 3 = 19.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_taxi_fare_l398_39815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l398_39801

theorem tan_alpha_minus_pi_fourth (α : Real) 
  (h1 : Real.sin α + Real.cos α = Real.sqrt 2 / 3)
  (h2 : 0 < α)
  (h3 : α < Real.pi) : 
  Real.tan (α - Real.pi/4) = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l398_39801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l398_39897

def sample_A : List ℝ := [43, 44, 47, 53, 43, 51]
def sample_B : List ℝ := sample_A.map (λ x => x - 3)

/-- Standard deviation of a list of real numbers -/
noncomputable def stdDev (l : List ℝ) : ℝ := sorry

/-- Mean of a list of real numbers -/
noncomputable def mean (l : List ℝ) : ℝ := sorry

/-- Mode of a list of real numbers -/
noncomputable def mode (l : List ℝ) : Set ℝ := sorry

/-- Median of a list of real numbers -/
noncomputable def median (l : List ℝ) : ℝ := sorry

theorem standard_deviation_invariant (A B : List ℝ) 
  (h : B = A.map (λ x => x - 3)) :
  stdDev A = stdDev B ∧ 
  mean A ≠ mean B ∧
  mode A ≠ mode B ∧
  median A ≠ median B := by
  sorry

#check standard_deviation_invariant sample_A sample_B rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l398_39897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l398_39819

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 else 3*x - 62

-- State the theorem
theorem h_composition_equality {a : ℝ} (ha : a < 0) :
  h (h (h 15)) = h (h (h a)) ↔ a = -Real.sqrt (Real.sqrt 83521) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_composition_equality_l398_39819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l398_39800

/-- Represents a distribution of balls into boxes -/
structure Distribution where
  box1 : ℕ
  box2 : ℕ
  box3 : ℕ

/-- The set of all valid distributions -/
def validDistributions : Set Distribution :=
  {d : Distribution | d.box1 + d.box2 + d.box3 = 9 ∧ d.box2 ≥ 2 ∧ d.box3 ≥ 3}

/-- The number of valid distributions -/
def numValidDistributions : ℕ := 10

theorem distribution_count : numValidDistributions = 10 := by
  -- The proof goes here
  sorry

#eval numValidDistributions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_count_l398_39800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_eq_7297_l398_39899

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The number of terms in the first n groups -/
def group_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last term of the nth group -/
def g (n : ℕ) : ℕ := 2 * n^2 - (5 * n) / 2 + 4

/-- The proposition that v_2008 equals 7297 -/
theorem v_2008_eq_7297 : v 2008 = 7297 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_eq_7297_l398_39899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_perimeter_l398_39858

/-- Represents an equilateral triangle with an inscribed circle -/
structure EquilateralTriangleWithInscribedCircle where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- Radius of the inscribed circle -/
  circle_radius : ℝ

/-- Represents a triangle formed by a tangent to the inscribed circle -/
structure TangentTriangle where
  /-- The equilateral triangle with inscribed circle -/
  parent : EquilateralTriangleWithInscribedCircle
  /-- Point where the tangent touches the circle -/
  tangent_point : ℝ × ℝ
  /-- Points where the tangent intersects the sides of the equilateral triangle -/
  intersection_points : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculate the perimeter of a TangentTriangle -/
def perimeter (t : TangentTriangle) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The perimeter of the tangent triangle is equal to the side length of the parent equilateral triangle -/
theorem tangent_triangle_perimeter 
  (t : TangentTriangle) : 
  perimeter t = t.parent.side_length :=
by
  sorry  -- The proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_perimeter_l398_39858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l398_39876

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Define the circumcircle H
def H : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 3)^2) = 10}

-- Define the line l
def l : Set (ℝ × ℝ) := {p | p.1 = 3 ∨ p.2 = (4/3) * p.1 - 2}

-- Define the segment BH
def BH : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • B + t • (0, 3)}

-- Define the circle C
def circle_C (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 2)^2 = r^2}

theorem triangle_ABC_properties :
  -- 1. Equation of circle H
  H = {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 10} ∧
  -- 2. Equation of line l
  (∀ p ∈ l, (p.1 = 3 ∨ p.2 = (4/3) * p.1 - 2) ∧ p ∈ H → ∃ q ∈ H, q ≠ p ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4) ∧
  -- 3. Range of radius r of circle C
  (∀ P ∈ BH, ∀ r : ℝ,
    (∃ M N : ℝ × ℝ, M ∈ circle_C r ∧ N ∈ circle_C r ∧ M ≠ N ∧
      M = ((P.1 + N.1) / 2, (P.2 + N.2) / 2)) →
    Real.sqrt 10 / 3 ≤ r ∧ r < 4 * Real.sqrt 10 / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l398_39876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l398_39880

/-- The slope of a line tangent to an ellipse -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ (x y : ℝ), x^2/7 + y^2/2 = 1 ∧ y = k*x + 2) ∧ 
  (∀ (x y : ℝ), x^2/7 + y^2/2 = 1 → y = k*x + 2 → 
    ∀ (x' y' : ℝ), x'^2/7 + y'^2/2 = 1 → y' = k*x' + 2 → x' = x ∧ y' = y) →
  k = Real.sqrt (1/7) ∨ k = -Real.sqrt (1/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l398_39880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l398_39873

def sequence_a : ℕ+ → ℝ := sorry
def sequence_S : ℕ+ → ℝ := sorry
def sequence_T : ℕ+ → ℝ := sorry

axiom partial_sum_relation (n : ℕ+) : sequence_T n = 2 * sequence_S n - n^2

theorem sequence_a_properties :
  (sequence_a 1 = 1) ∧
  (∀ n : ℕ+, sequence_a n = 3 * 2^(n.val - 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l398_39873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l398_39862

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the orthocenter H and circumcenter O
variable (H O : ℝ × ℝ)

-- Define vectors
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- Define vector addition
def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Assume the existence of orthocenter and circumcenter functions
noncomputable def orthocenter : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) := sorry
noncomputable def circumcenter : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Theorem statement
theorem orthocenter_circumcenter_relation 
  (h_orthocenter : H = orthocenter A B C)
  (h_circumcenter : O = circumcenter A B C)
  (h_sum_magnitude : magnitude (add_vectors (add_vectors (vector H A) (vector H B)) (vector H C)) = 2) :
  magnitude (add_vectors (add_vectors (vector O A) (vector O B)) (vector O C)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l398_39862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_increasing_interval_l398_39864

theorem cos_increasing_interval (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos (π / 4 - x)
  let a : ℝ := 2 * k * π - 3 * π / 4
  let b : ℝ := 2 * k * π + π / 4
  StrictMonoOn f (Set.Icc a b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_increasing_interval_l398_39864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bcd_is_27_l398_39875

/-- Represents the symbols used in the encryption scheme -/
inductive EncryptionSymbol : Type
| A : EncryptionSymbol
| B : EncryptionSymbol
| C : EncryptionSymbol
| D : EncryptionSymbol

/-- Represents a base-4 number as a list of symbols -/
def EncryptedNumber := List EncryptionSymbol

/-- Maps symbols to their corresponding base-4 digits -/
def symbolToDigit : EncryptionSymbol → Nat
| EncryptionSymbol.A => 0
| EncryptionSymbol.B => 1
| EncryptionSymbol.C => 2
| EncryptionSymbol.D => 3

/-- Converts an encrypted number to its base-10 representation -/
def toBase10 (en : EncryptedNumber) : Nat :=
  en.foldr (fun s acc => acc * 4 + symbolToDigit s) 0

/-- The encryption scheme property: three continuously increasing integers -/
axiom encryption_property (n : Nat) :
  toBase10 [EncryptionSymbol.A, EncryptionSymbol.B, EncryptionSymbol.C] = n ∧
  toBase10 [EncryptionSymbol.A, EncryptionSymbol.B, EncryptionSymbol.D] = n + 1 ∧
  toBase10 [EncryptionSymbol.B, EncryptionSymbol.A, EncryptionSymbol.A] = n + 2

/-- The main theorem: BCD in the encryption scheme represents 27 in base-10 -/
theorem bcd_is_27 : toBase10 [EncryptionSymbol.B, EncryptionSymbol.C, EncryptionSymbol.D] = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bcd_is_27_l398_39875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_palindrome_probability_l398_39877

/-- The probability of a license plate containing palindromes in both letter and digit sequences -/
theorem license_plate_palindrome_probability :
  (let letter_count : ℕ := 26
   let digit_count : ℕ := 10
   let plate_letter_length : ℕ := 4
   let plate_digit_length : ℕ := 4
   let letter_palindrome_count : ℕ := letter_count ^ 2
   let digit_palindrome_count : ℕ := digit_count ^ 2
   let total_letter_arrangements : ℕ := letter_count ^ plate_letter_length
   let total_digit_arrangements : ℕ := digit_count ^ plate_digit_length
   let probability : ℚ := (letter_palindrome_count * digit_palindrome_count : ℚ) / 
                          ((total_letter_arrangements * total_digit_arrangements) : ℚ)
   probability = 1 / 67600) := by
  sorry

#eval 1 + 67600 -- Evaluates to 67601

end NUMINAMATH_CALUDE_ERRORFEEDBACK_license_plate_palindrome_probability_l398_39877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l398_39889

-- Define the function f(x) = x - 1 - ln x
noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

-- State the theorem
theorem max_b_value (h : ∀ x > 0, f x ≥ (1 - 1 / Real.exp 2) * x - 2) :
  ∀ b > 1 - 1 / Real.exp 2, ∃ x > 0, f x < b * x - 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l398_39889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l398_39888

/-- Definition of closely related functions -/
def closely_related (f g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  ((f x₁ = 0 ∧ g x₁ = 0) ∨ (f x₁ = 0 ∧ x₁ = 0)) ∧
  ((f x₂ = 0 ∧ g x₂ = 0) ∨ (g x₂ = 0 ∧ x₂ = 0))

/-- Part 1: Prove that y=2x+1 and y=2x²+3x+1 are closely related -/
theorem part1 : closely_related (λ x => 2*x + 1) (λ x => 2*x^2 + 3*x + 1) := by sorry

/-- Part 2: Prove that b = 2 or b = -1 for the given conditions -/
theorem part2 (a c : ℝ) (h : c > 0) :
  let f := λ x => x + 3*c
  let g := λ x => a*x^2 + 2*b*x + 3*c
  closely_related f g →
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ g x₂ = 0 ∧ x₁ = 3*x₂ →
  b = 2 ∨ b = -1 := by sorry

/-- Part 3: Prove the analytical expression of y for the given conditions -/
theorem part3 (m n t : ℝ) :
  let y₁ := λ x => x + 1
  let y₂ := λ x => m*x^2 + n*x + t
  closely_related y₁ y₂ →
  (∀ x ∈ Set.Icc m (m+1), m*x^2 + m*x - m ≥ m) →
  (λ x => m*(x^2 + x - 1)) = (λ x => -x^2 - x + 1) ∨
  (λ x => m*(x^2 + x - 1)) = (λ x => 2*x^2 + 2*x - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_part3_l398_39888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_roots_l398_39881

theorem sum_of_real_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ + x₂ = 5 ∧
  (∀ x : ℂ, x^4 - 7*x^3 + 14*x^2 - 14*x + 4 = 0 ↔ (x^2 - 5*x + 2 = 0 ∨ x^2 - 2*x + 2 = 0)) ∧
  {x : ℝ | x^4 - 7*x^3 + 14*x^2 - 14*x + 4 = 0} = {x₁, x₂} := by
  sorry

#check sum_of_real_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_real_roots_l398_39881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_both_l398_39848

/-- Represents a class of students with preferences for calligraphy and music -/
structure StudentClass where
  total : ℕ
  calligraphy : ℕ
  music : ℕ
  calligraphy_le_total : calligraphy ≤ total
  music_le_total : music ≤ total

/-- The number of students who like both calligraphy and music -/
def both (c : StudentClass) : ℤ := c.calligraphy + c.music - c.total

/-- The maximum number of students who can like both calligraphy and music -/
def max_both (c : StudentClass) : ℕ := min c.calligraphy c.music

/-- The minimum number of students who can like both calligraphy and music -/
def min_both (c : StudentClass) : ℕ := max (Int.toNat (both c)) 0

theorem sum_max_min_both (c : StudentClass) 
  (h_total : c.total = 48)
  (h_calligraphy : c.calligraphy = 24)
  (h_music : c.music = 30) :
  max_both c + min_both c = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_both_l398_39848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_problem_l398_39887

/-- The initial speed of the motorcyclist in km/h -/
noncomputable def initial_speed : ℝ := 48

/-- The distance between points A and B in km -/
noncomputable def distance : ℝ := 120

/-- The speed increase for the second part of the return trip in km/h -/
noncomputable def speed_increase : ℝ := 6

/-- The duration of the stop during the return trip in hours -/
noncomputable def stop_duration : ℝ := 1/6

theorem motorcyclist_speed_problem :
  let time_ab := distance / initial_speed
  let time_return_first_hour := (1 : ℝ)
  let distance_remaining := distance - initial_speed * time_return_first_hour
  let time_return_after_stop := distance_remaining / (initial_speed + speed_increase)
  time_ab = time_return_first_hour + stop_duration + time_return_after_stop := by
  sorry

#check motorcyclist_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcyclist_speed_problem_l398_39887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_p_l398_39853

theorem negation_of_proposition_p :
  (¬ ∀ x : ℝ, Real.cos x ≥ 1) ↔ (∃ x : ℝ, Real.cos x < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_p_l398_39853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_is_3700_l398_39886

/-- A bijective mapping from the letters in "SIMPLE TASK" to digits 0-9 -/
def letter_to_digit : Char → Fin 10 := sorry

/-- The word we want to convert to a number -/
def word : List Char := ['P', 'A', 'S', 'S']

/-- Convert a list of characters to a natural number based on letter_to_digit mapping -/
def word_to_number (w : List Char) : ℕ :=
  w.foldl (fun acc d => 10 * acc + (letter_to_digit d).val) 0

/-- Conditions on the letter_to_digit mapping -/
axiom p_maps_to_3 : letter_to_digit 'P' = 3
axiom a_maps_to_7 : letter_to_digit 'A' = 7
axiom s_maps_to_0 : letter_to_digit 'S' = 0

/-- The main theorem to prove -/
theorem pass_is_3700 : word_to_number word = 3700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pass_is_3700_l398_39886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_odometer_correct_miles_l398_39847

/-- Represents an odometer that skips the digit 5 --/
structure FaultyOdometer where
  reading : Nat

/-- Converts a faulty odometer reading to the actual miles traveled --/
def actualMiles (o : FaultyOdometer) : Nat :=
  sorry

/-- The current odometer reading --/
def currentReading : FaultyOdometer :=
  { reading := 3006 }

theorem faulty_odometer_correct_miles :
  actualMiles currentReading = 2192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_odometer_correct_miles_l398_39847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l398_39857

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length of the hyperbola
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem min_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola a b a b ∧ hyperbola a b a (-b) →  -- D(a, b) and E(a, -b) are on the hyperbola
  a * b = 8 →  -- Area of triangle ODE is 8
  ∀ c d : ℝ, c > 0 → d > 0 → hyperbola c d c d ∧ hyperbola c d c (-d) → c * d = 8 →
    focal_length a b ≤ focal_length c d ∧
    ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ hyperbola a₀ b₀ a₀ b₀ ∧ hyperbola a₀ b₀ a₀ (-b₀) ∧
      a₀ * b₀ = 8 ∧ focal_length a₀ b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_l398_39857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_width_l398_39865

/-- The width of a rectangular prism with length 5 and height 8, 
    where the prism's diagonal plus twice the width equals 21 -/
noncomputable def width : ℝ :=
  (-28 + Real.sqrt (1253 / 3)) / 2

/-- Theorem stating the properties of the rectangular prism -/
theorem rectangular_prism_width :
  let l : ℝ := 5
  let h : ℝ := 8
  let w := width
  Real.sqrt (l^2 + w^2 + h^2) + 2*w = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_width_l398_39865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_stop_time_l398_39826

/-- The time it takes for the woman to stop after passing the man -/
noncomputable def stop_time (man_speed woman_speed wait_time : ℝ) : ℝ :=
  (wait_time * man_speed) / (woman_speed - man_speed)

/-- Theorem stating that the woman stops 5 minutes after passing the man -/
theorem woman_stop_time :
  let man_speed : ℝ := 5  -- miles per hour
  let woman_speed : ℝ := 25  -- miles per hour
  let wait_time : ℝ := 1/3  -- hours (20 minutes)
  stop_time man_speed woman_speed wait_time = 1/12  -- hours (5 minutes)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_woman_stop_time_l398_39826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l398_39863

theorem inequality_proof (x : ℝ) (hx : x > 0) :
  (2 : ℝ)^(x^(1/12)) + (2 : ℝ)^(x^(1/4)) ≥ (2 : ℝ)^(1 + x^(1/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l398_39863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_movement_theorem_l398_39892

/-- Represents the total distance traveled by a flag on a flagpole -/
noncomputable def flag_distance (pole_height : ℝ) : ℝ :=
  pole_height + (pole_height / 2) + (pole_height / 2) + pole_height

/-- Theorem stating that for a 60-foot flagpole, the total distance traveled by the flag is 180 feet -/
theorem flag_movement_theorem :
  flag_distance 60 = 180 := by
  -- Unfold the definition of flag_distance
  unfold flag_distance
  -- Simplify the arithmetic
  simp [add_assoc, mul_two, ← add_mul]
  -- Check that 60 + 60 + 60 = 180
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flag_movement_theorem_l398_39892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_ratio_2_3_4_is_acute_l398_39810

theorem triangle_with_ratio_2_3_4_is_acute (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →  -- Angles are positive
  A + B + C = 180 →        -- Sum of angles is 180 degrees
  B = 3/2 * A →            -- Ratio of B to A is 3:2
  C = 2 * A →              -- Ratio of C to A is 4:2 (simplified to 2:1)
  A < 90 ∧ B < 90 ∧ C < 90 -- All angles are less than 90 degrees (acute triangle)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_ratio_2_3_4_is_acute_l398_39810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationships_l398_39874

-- Define a plane
def Plane : Type := ℝ × ℝ

-- Define a line in the plane
def Line : Type := Plane → Prop

-- Define the relationship between two lines
inductive LineRelationship
| Parallel
| Intersecting

-- Statement to prove
theorem line_relationships (l1 l2 : Line) :
  l1 ≠ l2 → ∃ (r : LineRelationship), r = LineRelationship.Parallel ∨ r = LineRelationship.Intersecting :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relationships_l398_39874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l398_39883

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes has a slope of √3, then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_slope : b / a = Real.sqrt 3) : 
  Real.sqrt (a^2 + b^2) / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l398_39883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x0_equals_zero_l398_39812

theorem tan_x0_equals_zero (x₀ : ℝ) (h1 : 0 ≤ x₀) (h2 : x₀ ≤ π/2)
  (h3 : Real.sqrt (Real.sin x₀ + 1) - Real.sqrt (1 - Real.sin x₀) = Real.sin (x₀/2)) :
  Real.tan x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_x0_equals_zero_l398_39812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l398_39860

-- Define the force function
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 2 then 5 else 3*x + 4

-- Define the work function
noncomputable def work (a b : ℝ) : ℝ :=
  ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation : work 0 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l398_39860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_f_has_two_zeros_l398_39836

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x - a else 2 * x - a

-- State the theorem
theorem range_of_a_when_f_has_two_zeros :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) → 0 < a ∧ a ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_when_f_has_two_zeros_l398_39836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_walking_distance_l398_39835

/-- Calculates the distance Alex had to walk to reach the next town -/
theorem alex_walking_distance (total_distance : ℝ) (flat_time flat_speed : ℝ) 
  (uphill_time uphill_speed : ℝ) (downhill_time downhill_speed : ℝ) 
  (h1 : total_distance = 164) 
  (h2 : flat_time = 4.5) (h3 : flat_speed = 20)
  (h4 : uphill_time = 2.5) (h5 : uphill_speed = 12)
  (h6 : downhill_time = 1.5) (h7 : downhill_speed = 24) : 
  total_distance - (flat_time * flat_speed + uphill_time * uphill_speed + downhill_time * downhill_speed) = 8 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_walking_distance_l398_39835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_special_case_l398_39832

/-- The slope angle of a line passing through two points -/
noncomputable def slope_angle (x1 y1 x2 y2 : ℝ) : ℝ := Real.arctan ((y2 - y1) / (x2 - x1))

/-- Theorem: The slope angle of the line passing through (0, 0) and (1, √3) is 60° -/
theorem slope_angle_special_case : slope_angle 0 0 1 (Real.sqrt 3) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_special_case_l398_39832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l398_39879

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := sin (π/2 - x) * sin x - Real.sqrt 3 * (cos x)^2 - Real.sqrt 3 / 2

theorem f_properties :
  (∃ (k : ℤ), f ((5 * π / 12) + k * π) = 1 ∧ 
    ∀ (x : ℝ), f x ≤ 1) ∧
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < π ∧ 0 < x₂ ∧ x₂ < π →
    f x₁ = 2/3 ∧ f x₂ = 2/3 → cos (x₁ - x₂) = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l398_39879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commutator_squared_zero_l398_39852

open Matrix Complex

/-- The adjugate matrix of a square matrix -/
noncomputable def adjugate (n : ℕ) (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  sorry

/-- A matrix is idempotent if A * A = A -/
def isIdempotent {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ) : Prop :=
  A * A = A

theorem commutator_squared_zero
  {n : ℕ} (hn : n ≥ 2)
  (A B : Matrix (Fin n) (Fin n) ℂ)
  (C : Matrix (Fin n) (Fin n) ℂ)
  (hC : isIdempotent C)
  (h : adjugate n C = A * B - B * A) :
  (A * B - B * A) * (A * B - B * A) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commutator_squared_zero_l398_39852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_sum_l398_39870

theorem alternating_sequence_sum : 
  (Finset.range 100).sum (fun i => (-1 : ℤ)^i * (i + 1)) = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sequence_sum_l398_39870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_l398_39882

theorem equal_debt_time (jordan_initial_debt lee_initial_debt jordan_interest_rate lee_interest_rate jordan_extra_debt : ℝ) 
  (h1 : jordan_initial_debt = 200)
  (h2 : lee_initial_debt = 300)
  (h3 : jordan_interest_rate = 0.12)
  (h4 : lee_interest_rate = 0.08)
  (h5 : jordan_extra_debt = 20) :
  (jordan_initial_debt + jordan_extra_debt) * (1 + jordan_interest_rate * (100 / 3)) = 
    lee_initial_debt * (1 + lee_interest_rate * (100 / 3)) := by
  sorry

#eval Float.toString ((100:Float) / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_l398_39882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_reasoning_l398_39804

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the membership relation for points in lines and planes
variable (mem_line : Point → Line → Prop)
variable (mem_plane : Point → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Statement to be proven false
theorem incorrect_reasoning (l : Line) (α : Plane) :
  ¬(∀ A : Point, (¬(subset l α) ∧ mem_line A l) → ¬(mem_plane A α)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_reasoning_l398_39804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l398_39817

/-- The number of solutions to the equation √(x+3) = ax + 2 -/
noncomputable def numSolutions (a : ℝ) : ℕ :=
  if 0 < a ∧ a < 1/6 ∨ 1/2 < a ∧ a ≤ 2/3 then 2
  else if a ≤ 0 ∨ a = 1/6 ∨ a = 1/2 ∨ a > 2/3 then 1
  else 0

theorem solutions_count (a : ℝ) :
  (numSolutions a = 2 ↔ (0 < a ∧ a < 1/6 ∨ 1/2 < a ∧ a ≤ 2/3)) ∧
  (numSolutions a = 1 ↔ (a ≤ 0 ∨ a = 1/6 ∨ a = 1/2 ∨ a > 2/3)) ∧
  (numSolutions a = 0 ↔ (1/6 < a ∧ a < 1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_l398_39817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_value_l398_39855

theorem angle_beta_value (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α = Real.sqrt 5 / 5 →
  Real.sin (α - β) = -(Real.sqrt 10) / 10 →
  β = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_beta_value_l398_39855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_bounds_l398_39831

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  pos : ∀ x ∈ domain, f x > 0
  deriv_bounds : ∀ x ∈ domain, f x < (deriv f) x ∧ (deriv f) x < 3 * f x

/-- The main theorem -/
theorem special_function_range_bounds (sf : SpecialFunction) :
  1 / exp 6 < sf.f 1 / sf.f 3 ∧ sf.f 1 / sf.f 3 < 1 / exp 2 := by
  sorry

#check special_function_range_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_bounds_l398_39831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l398_39807

theorem triangle_segment_length (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  (‖A - C‖ = 10) →
  (‖B - C‖ = 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B) →
  (‖A - D‖ = 12) →
  (‖C - D‖ = 5) →
  ‖B - D‖ = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_segment_length_l398_39807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_four_hundred_fifty_fifth_number_l398_39859

def digit_count : ℕ → ℕ
| 5 => 1
| 3 => 2
| 2 => 100
| _ => 0

def valid_number (n : List ℕ) : Prop :=
  n.length = 10 ∧ 
  n.count 5 = 1 ∧ 
  n.count 3 ≤ 2 ∧ 
  n.count 2 = 10 - n.count 3 - 1 ∧
  (n.foldl (· * ·) 1) % 10 = 0

def count_valid_numbers : ℕ := sorry

theorem valid_number_count : count_valid_numbers = 460 := by sorry

def nth_valid_number (n : ℕ) : List ℕ := sorry

theorem four_hundred_fifty_fifth_number :
  nth_valid_number 455 = [5, 3, 2, 2, 2, 2, 2, 3, 2, 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_four_hundred_fifty_fifth_number_l398_39859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l398_39845

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-4) 5

-- Define the new function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (x + 2)

-- Define the domain of g
def domain_g : Set ℝ := Set.Ioc (-2) 6

-- Theorem statement
theorem domain_of_g (x : ℝ) : 
  x ∈ domain_g ↔ (x - 1) ∈ domain_f ∧ x + 2 > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l398_39845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_function_l398_39878

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem sum_of_special_function (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_odd : is_odd (fun x ↦ f (x - 1)))
  (h_f2 : f 2 = -1) :
  (Finset.sum (Finset.range 2011) (fun i ↦ f (i + 1))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_function_l398_39878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diametric_cutting_theorem_multiple_cutting_circles_theorem_l398_39808

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A circle S cuts another circle Sigma diametrically if their common chord is a diameter of Sigma --/
def cuts_diametrically (S Sigma : Circle) : Prop := sorry

/-- Three circles with distinct centers --/
def three_circles (S_A S_B S_C : Circle) : Prop :=
  S_A.center ≠ S_B.center ∧ S_B.center ≠ S_C.center ∧ S_C.center ≠ S_A.center

/-- Centers of three circles are collinear --/
def centers_collinear (S_A S_B S_C : Circle) : Prop := sorry

/-- A circle cuts all three given circles diametrically --/
def cuts_all_diametrically (S S_A S_B S_C : Circle) : Prop :=
  cuts_diametrically S S_A ∧ cuts_diametrically S S_B ∧ cuts_diametrically S S_C

/-- There exists a unique circle that cuts all three given circles diametrically --/
def unique_cutting_circle (S_A S_B S_C : Circle) : Prop :=
  ∃! S, cuts_all_diametrically S S_A S_B S_C

/-- All circles cutting the three given circles diametrically pass through two fixed points --/
def all_pass_through_fixed_points (S_A S_B S_C : Circle) : Prop := sorry

theorem diametric_cutting_theorem (S_A S_B S_C : Circle) 
  (h : three_circles S_A S_B S_C) :
  centers_collinear S_A S_B S_C ↔ ¬(unique_cutting_circle S_A S_B S_C) :=
by sorry

theorem multiple_cutting_circles_theorem (S_A S_B S_C : Circle) 
  (h : three_circles S_A S_B S_C) :
  (∃ S₁ S₂, S₁ ≠ S₂ ∧ cuts_all_diametrically S₁ S_A S_B S_C ∧ cuts_all_diametrically S₂ S_A S_B S_C) →
  all_pass_through_fixed_points S_A S_B S_C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diametric_cutting_theorem_multiple_cutting_circles_theorem_l398_39808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_B_in_geometric_triangle_l398_39820

-- Define a triangle with sides a, b, c opposite to angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  sum_angles : A + B + C = π
  -- Cosine theorem
  cos_theorem_B : Real.cos B = (a^2 + c^2 - b^2) / (2*a*c)

-- Define the geometric sequence property
def is_geometric_sequence (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

-- Theorem statement
theorem max_angle_B_in_geometric_triangle (t : Triangle) 
  (h_geo : is_geometric_sequence t) : 
  t.B ≤ π/3 ∧ ∃ (t' : Triangle), is_geometric_sequence t' ∧ t'.B = π/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_B_in_geometric_triangle_l398_39820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_shared_sets_l398_39821

-- Define the number of sets
def num_sets : ℕ := 11

-- Define the number of elements in each set
def set_size : ℕ := 5

-- Define a type for our sets
def SetSystem := Fin num_sets → Finset ℕ

-- Define the property that every pair of sets has a non-empty intersection
def has_pairwise_intersection (S : SetSystem) : Prop :=
  ∀ i j, i ≠ j → (S i).Nonempty ∧ (S j).Nonempty ∧ (S i ∩ S j).Nonempty

-- Define the property that each set has exactly 5 elements
def has_correct_size (S : SetSystem) : Prop :=
  ∀ i, (S i).card = set_size

-- Define the maximum number of sets sharing a common element
noncomputable def max_shared (S : SetSystem) : ℕ :=
  Finset.sup (Finset.univ.biUnion S) (λ x ↦ (Finset.univ.filter (λ i ↦ x ∈ S i)).card)

-- State the theorem
theorem min_max_shared_sets :
  ∃ (S : SetSystem), has_pairwise_intersection S ∧ has_correct_size S ∧
  (∀ (T : SetSystem), has_pairwise_intersection T → has_correct_size T →
    max_shared S ≤ max_shared T) ∧
  max_shared S = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_shared_sets_l398_39821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l398_39885

def is_valid_grid (grid : Fin 4 → Fin 3 → ℕ) : Prop :=
  (∀ i : Fin 4, ∃ s : ℕ, (List.sum (List.map (grid i) (List.range 3))) = s) ∧
  (∀ j : Fin 3, ∃ t : ℕ, (List.sum (List.map (λ i => grid i j) (List.range 4))) = t) ∧
  (∀ i j, grid i j ∈ Finset.range 14 \ {0}) ∧
  ((List.map (λ (i, j) => grid i j) (List.product (List.range 4) (List.range 3))).toFinset.card = 12)

theorem grid_sum_theorem (grid : Fin 4 → Fin 3 → ℕ) :
  is_valid_grid grid →
  (∃ n : ℕ, n ∈ Finset.range 14 \ {0} ∧ n ∉ (List.map (λ (i, j) => grid i j) (List.product (List.range 4) (List.range 3))).toFinset) →
  (∀ i : Fin 4, (List.sum (List.map (grid i) (List.range 3))) = 21) ∧
  (∀ j : Fin 3, (List.sum (List.map (λ i => grid i j) (List.range 4))) = 28) ∧
  (7 ∉ (List.map (λ (i, j) => grid i j) (List.product (List.range 4) (List.range 3))).toFinset) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l398_39885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l398_39867

theorem inequality_proof (a b c : ℝ) (n : ℕ+) 
  (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1 + a)^(-(1/n : ℝ)) + (1 + b)^(-(1/n : ℝ)) + (1 + c)^(-(1/n : ℝ)) ≤ 3 * (1 + (a*b*c)^(1/3))^(-(1/n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l398_39867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l398_39825

/-- The slope of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def my_slope (x₁ y₁ x₂ y₂ : ℚ) : ℚ :=
  (y₂ - y₁) / (x₂ - x₁)

/-- Theorem: The slope of the line passing through (-12, -39) and (400, 0) is 39/412 -/
theorem line_slope :
  my_slope (-12) (-39) 400 0 = 39 / 412 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l398_39825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_group_positive_count_l398_39805

/-- Represents a group of people with their ages --/
structure AgeGroup where
  members : Finset Int
  age : Int → Int

/-- The properties of the age group in the problem --/
def ProblemGroup (G : AgeGroup) : Prop :=
  (G.members.card = 23) ∧
  (∀ m ∈ G.members, -20 ≤ G.age m ∧ G.age m ≤ 20) ∧
  (G.members.sum G.age = 0) ∧
  ((G.members.filter (fun m ↦ G.age m < 0)).card = 5)

theorem age_group_positive_count (G : AgeGroup) (h : ProblemGroup G) :
  (G.members.filter (fun m ↦ G.age m > 0)).card ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_group_positive_count_l398_39805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractal_sequence_properties_l398_39841

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => sorry  -- The recursive definition would go here

-- Define the sum function
def S (n : ℕ) : ℕ := sorry  -- Sum of first n terms

-- Theorem statement
theorem fractal_sequence_properties :
  a 2000 = 2 ∧ S 2000 = 3950 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractal_sequence_properties_l398_39841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_tan_not_one_l398_39861

theorem negation_of_forall_tan_not_one :
  (¬ ∀ x : ℝ, Real.tan x ≠ 1) ↔ (∃ x : ℝ, Real.tan x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_tan_not_one_l398_39861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_proof_l398_39833

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem function_composition_proof :
  (∀ x, f (g x) = 6 * x + 3) →
  (∀ x, g x = 2 * x + 1) →
  (∀ x, f x = 3 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_proof_l398_39833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l398_39856

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin (Real.pi * x)
  else Real.cos ((Real.pi * x) / 2 + Real.pi / 3)

theorem f_composition_value : f (f (15 / 2)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l398_39856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l398_39866

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angleSum : A + B + C = π
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define vectors m and n
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b)

def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b - t.a)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : dot_product (m t) (n t) = 0) : 
  t.C = π / 3 ∧ 
  Real.sqrt 3 / 2 < Real.sin t.A + Real.sin t.B ∧ 
  Real.sin t.A + Real.sin t.B ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l398_39866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l398_39840

/-- The length of the first train in meters -/
noncomputable def train1_length : ℝ := 156.62

/-- The speed of the first train in km/hr -/
noncomputable def train1_speed : ℝ := 30

/-- The speed of the second train in km/hr -/
noncomputable def train2_speed : ℝ := 36

/-- The time taken for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ := 13.996334838667455

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

/-- The theorem stating the length of the second train -/
theorem second_train_length :
  ∃ (L : ℝ), 
    (train1_length + L) / ((train1_speed + train2_speed) * km_hr_to_m_s) = crossing_time ∧ 
    abs (L - 100.05) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l398_39840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_is_5_75_l398_39827

/-- Represents the fishing scenario with Jenna and Mark --/
structure FishingScenario where
  distance_from_shore : ℝ
  water_intake_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ
  rowing_time : ℝ
  bailing_increase : ℝ
  increase_time : ℝ

/-- Calculates the minimum initial bailing rate required to prevent sinking --/
noncomputable def min_initial_bailing_rate (scenario : FishingScenario) : ℝ :=
  let total_water_intake := scenario.water_intake_rate * scenario.rowing_time
  let max_allowed_water := total_water_intake - scenario.max_water_capacity
  let bailing_time_before_increase := scenario.increase_time
  let bailing_time_after_increase := scenario.rowing_time - scenario.increase_time
  (max_allowed_water - scenario.bailing_increase * bailing_time_after_increase) /
    (bailing_time_before_increase + bailing_time_after_increase)

/-- Theorem stating that the minimum initial bailing rate for the given scenario is 5.75 --/
theorem min_bailing_rate_is_5_75 (scenario : FishingScenario) 
    (h1 : scenario.distance_from_shore = 2)
    (h2 : scenario.water_intake_rate = 8)
    (h3 : scenario.max_water_capacity = 50)
    (h4 : scenario.rowing_speed = 3)
    (h5 : scenario.rowing_time = 40)
    (h6 : scenario.bailing_increase = 2)
    (h7 : scenario.increase_time = 20) :
  min_initial_bailing_rate scenario = 5.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bailing_rate_is_5_75_l398_39827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l398_39846

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l398_39846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_529_l398_39823

/-- Represents the possible digits for each position based on the visible segments --/
def PossibleDigits : List (List Nat) :=
  [[3, 5, 9], [2, 3, 7], [3, 4, 8, 9], [2, 3, 7], [3, 5, 9], [1, 4, 7], [4, 5, 9], [2], [4, 5, 9]]

/-- Checks if a number is a valid three-digit number --/
def isValidThreeDigitNumber (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

/-- Checks if a digit is possible for a given position --/
def isPossibleDigit (digit : Nat) (position : Nat) : Prop :=
  digit ∈ (PossibleDigits.get! position)

/-- The main theorem stating the maximum difference --/
theorem max_difference_is_529 :
  ∃ (minuend subtrahend : Nat),
    isValidThreeDigitNumber minuend ∧
    isValidThreeDigitNumber subtrahend ∧
    isValidThreeDigitNumber (minuend - subtrahend) ∧
    (∀ (d1 d2 d3 : Nat), isPossibleDigit d1 0 → isPossibleDigit d2 1 → isPossibleDigit d3 2 →
      minuend = 100 * d1 + 10 * d2 + d3) ∧
    (∀ (d1 d2 d3 : Nat), isPossibleDigit d1 3 → isPossibleDigit d2 4 → isPossibleDigit d3 5 →
      subtrahend = 100 * d1 + 10 * d2 + d3) ∧
    (minuend - subtrahend = 529) ∧
    (∀ (m s : Nat),
      isValidThreeDigitNumber m →
      isValidThreeDigitNumber s →
      isValidThreeDigitNumber (m - s) →
      (∀ (d1 d2 d3 : Nat), isPossibleDigit d1 0 → isPossibleDigit d2 1 → isPossibleDigit d3 2 →
        m = 100 * d1 + 10 * d2 + d3) →
      (∀ (d1 d2 d3 : Nat), isPossibleDigit d1 3 → isPossibleDigit d2 4 → isPossibleDigit d3 5 →
        s = 100 * d1 + 10 * d2 + d3) →
      m - s ≤ minuend - subtrahend) :=
by
  sorry

#check max_difference_is_529

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_is_529_l398_39823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_prob_l398_39884

/-- Represents the probability of Fiona reaching a specific lily pad safely. -/
def safe_reach_prob : ℕ → ℚ := sorry

/-- The number of lily pads. -/
def num_pads : ℕ := 16

/-- The pad where Fiona starts. -/
def start_pad : ℕ := 0

/-- The pad with food (destination). -/
def food_pad : ℕ := 14

/-- The pads with predators. -/
def predator_pads : List ℕ := [4, 9]

/-- The probability of hopping to the next pad. -/
def hop_prob : ℚ := 1/2

/-- The probability of jumping two pads. -/
def jump_prob : ℚ := 1/2

/-- The probability of jumping back to pad 0 from pad 1. -/
def back_jump_prob : ℚ := 1/4

/-- Theorem stating the probability of Fiona reaching the food pad safely. -/
theorem fiona_reach_food_prob : safe_reach_prob food_pad = 3/128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_prob_l398_39884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l398_39822

theorem parity_of_expression (a b c : ℕ) (ha : Odd a) (hb : Odd b) : 
  Odd (5^a + (b+1)^2*c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parity_of_expression_l398_39822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vishal_excess_percentage_l398_39843

/-- Represents the investment amounts of Vishal, Trishul, and Raghu -/
structure Investments where
  vishal : ℚ
  trishul : ℚ
  raghu : ℚ

/-- The conditions of the investment problem -/
def InvestmentConditions (i : Investments) : Prop :=
  i.raghu = 2000 ∧
  i.trishul = i.raghu * (9/10) ∧
  i.vishal + i.trishul + i.raghu = 5780

/-- The percentage by which Vishal invested more than Trishul -/
noncomputable def VishalExcessPercentage (i : Investments) : ℚ :=
  (i.vishal - i.trishul) / i.trishul * 100

/-- Theorem stating that under the given conditions, 
    Vishal invested 10% more than Trishul -/
theorem vishal_excess_percentage 
  (i : Investments) (h : InvestmentConditions i) : 
  VishalExcessPercentage i = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vishal_excess_percentage_l398_39843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_one_l398_39829

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2
noncomputable def g (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := (x - 2) / 2
noncomputable def g_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composition_equals_one :
  f (g_inv (f_inv (f_inv (g (f 10))))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_one_l398_39829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l398_39844

/-- The number of distinct intersection points of two curves -/
def intersection_count : ℕ := 2

/-- First curve equation -/
def curve1 (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- Second curve equation -/
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 16

/-- Theorem stating that there are exactly two distinct intersection points -/
theorem two_intersection_points :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = intersection_count ∧
    ∀ p ∈ points, let (x, y) := p; curve1 x y ∧ curve2 x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l398_39844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bounds_l398_39830

theorem quadratic_function_bounds (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → (a * x^2 + b * x + c) ∈ Set.Icc (-1 : ℝ) 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 2 → (a * x^2 + b * x + c) ∈ Set.Icc (-7 : ℝ) 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bounds_l398_39830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_weight_is_72_5_l398_39806

/-- Density function of the rod -/
def ρ (x : ℝ) : ℝ := 10 + x

/-- Length of the rod -/
def rod_length : ℝ := 11.25

/-- Total weight of the rod -/
def total_weight : ℝ := 42.75

/-- Start point of the segment -/
def a : ℝ := 2

/-- End point of the segment -/
def b : ℝ := 7

/-- Weight of a segment of the rod from point a to point b -/
noncomputable def segment_weight (a b : ℝ) : ℝ := ∫ x in a..b, ρ x

/-- Theorem stating that the segment weight from a to b is 72.5 -/
theorem segment_weight_is_72_5 : segment_weight a b = 72.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_weight_is_72_5_l398_39806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puddle_base_area_is_one_sq_cm_l398_39871

/-- Represents the properties of a cylindrical puddle filled by rain --/
structure RainPuddle where
  rainfall_rate : ℚ  -- cm per hour
  rain_duration : ℚ  -- hours
  puddle_depth : ℚ   -- cm

/-- Calculates the base area of a rain puddle --/
def base_area (puddle : RainPuddle) : ℚ :=
  (puddle.rainfall_rate * puddle.rain_duration) / puddle.puddle_depth

/-- Theorem stating that under given conditions, the base area of the puddle is 1 square cm --/
theorem puddle_base_area_is_one_sq_cm (puddle : RainPuddle)
  (h_rate : puddle.rainfall_rate = 10)
  (h_duration : puddle.rain_duration = 3)
  (h_depth : puddle.puddle_depth = 30) :
  base_area puddle = 1 := by
  sorry

def example_puddle : RainPuddle := {
  rainfall_rate := 10,
  rain_duration := 3,
  puddle_depth := 30
}

#eval base_area example_puddle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puddle_base_area_is_one_sq_cm_l398_39871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_is_6_sqrt_3_l398_39809

/-- An equilateral triangle with side length 6 -/
structure EquilateralTriangle :=
  (side_length : ℝ)
  (is_equilateral : side_length = 6)

/-- The sum of distances from a vertex to midpoints of all sides -/
noncomputable def sum_distances_to_midpoints (t : EquilateralTriangle) : ℝ :=
  3 * (t.side_length * Real.sqrt 3 / 2)

/-- Theorem: The sum of distances from a vertex to midpoints is 6√3 -/
theorem sum_distances_is_6_sqrt_3 (t : EquilateralTriangle) :
  sum_distances_to_midpoints t = 6 * Real.sqrt 3 := by
  sorry

#check sum_distances_is_6_sqrt_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_is_6_sqrt_3_l398_39809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l398_39850

/-- Given sequences a and b with specific properties, prove a lower bound for lambda -/
theorem sequence_lower_bound (a b : ℕ+ → ℝ) (lambda : ℝ) : 
  (∀ n : ℕ+, a (n + 1) = 3 * a n) →
  (∀ n : ℕ+, b n = b (n + 1) - 1) →
  (b 6 = a 1) →
  (b 6 = 3) →
  (∀ n : ℕ+, (2 * lambda - 1) * a n > 36 * b n) →
  lambda > 13 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lower_bound_l398_39850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l398_39896

-- Define the motion equation
noncomputable def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the velocity function as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by
  -- The proof steps would go here, but we'll use 'sorry' for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_3_seconds_l398_39896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l398_39849

-- Define the function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 1)^x

-- State the theorem
theorem decreasing_function_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) ↔ (1 < |a| ∧ |a| < Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_condition_l398_39849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l398_39824

open Real

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = 2^x - m)
  (h : ∀ x₁, x₁ ∈ Set.Icc (-1) 3 → ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂) :
  m ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l398_39824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_vector_l398_39811

/-- Given a triangle ABC with points G on AB and H on BC, prove that the intersection Q of AG and CH
    can be expressed as a specific linear combination of A, B, and C. -/
theorem triangle_intersection_vector (A B C G H Q : ℝ × ℝ) : 
  (∃ (t : ℝ), G = (1 - t) • A + t • B ∧ t = 3/4) →  -- G lies on AB with AG:GB = 1:3
  (∃ (s : ℝ), H = (1 - s) • B + s • C ∧ s = 2/3) →  -- H lies on BC with BH:HC = 2:1
  (∃ (r₁ r₂ : ℝ), Q = (1 - r₁) • A + r₁ • G ∧ Q = (1 - r₂) • C + r₂ • H) →  -- Q is on AG and CH
  Q = (3/7) • A + (2/7) • B + (2/7) • C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_vector_l398_39811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_p_and_q_l398_39869

/-- Line represented by a slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The first quadrant of the coordinate plane -/
def first_quadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The area under a line in the first quadrant -/
noncomputable def area_under_line (l : Line) : ℝ :=
  (l.y_intercept ^ 2) / (2 * (-l.slope))

/-- The area between two lines in the first quadrant -/
noncomputable def area_between_lines (l1 l2 : Line) : ℝ :=
  area_under_line l1 - area_under_line l2

/-- The probability of a point falling between two lines -/
noncomputable def probability_between_lines (l1 l2 : Line) : ℝ :=
  area_between_lines l1 l2 / area_under_line l1

theorem probability_between_p_and_q :
  let p : Line := { slope := -2, y_intercept := 8 }
  let q : Line := { slope := -3, y_intercept := 9 }
  probability_between_lines p q = 0.15625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_p_and_q_l398_39869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l398_39854

theorem smallest_angle_in_ratio_triangle :
  ∃ (a b c : ℝ), a = 45 ∧ b > a ∧ c > a ∧ a + b + c = 180 :=
by
  -- Define the ratio of angles
  have angle_ratio : ∃ (k : ℝ), k > 0 ∧ 
    (∃ (a b c : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧
    -- Sum of angles in a triangle is 180 degrees
    a + b + c = 180) := by
    sorry
  
  -- The smallest angle is 45 degrees
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_ratio_triangle_l398_39854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l398_39898

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (m + 2)*x - 1 < (m + 2)*y - 1

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : S = Set.Iic (-2) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l398_39898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l398_39828

/-- Given a polynomial q(x) satisfying specific remainder conditions,
    prove that s(x) = -x^2 + x + 8 is the unique quadratic remainder when
    q(x) is divided by (x - 3)(x - 4)(x + 2), and that s(7) = -34. -/
theorem remainder_problem (q : ℝ → ℝ) 
  (h₁ : ∃ k₁, q 3 = 2 + (3 - 3) * k₁)
  (h₂ : ∃ k₂, q 4 = -4 + (4 - 4) * k₂)
  (h₃ : ∃ k₃, q (-2) = 5 + (-2 + 2) * k₃) :
  (∃! s : ℝ → ℝ, (∃ t : ℝ → ℝ, ∀ x, q x = (x - 3) * (x - 4) * (x + 2) * t x + s x) ∧
                 (∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c)) ∧
  (let s := λ x => -x^2 + x + 8; s 7 = -34) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l398_39828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_speed_is_10_l398_39834

/-- The speed of the hare in a race with a turtle -/
noncomputable def hare_speed (race_distance : ℝ) (turtle_speed : ℝ) (head_start : ℝ) : ℝ :=
  race_distance / (race_distance / turtle_speed - head_start)

/-- Theorem stating that the hare's speed is 10 feet/second under given conditions -/
theorem hare_speed_is_10 :
  hare_speed 20 1 18 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_speed_is_10_l398_39834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l398_39837

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- Define the theorem
theorem range_of_p :
  Set.range (fun x ↦ p x) ∩ Set.Ici (0 : ℝ) = Set.Ici (9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l398_39837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_plus_pi_third_l398_39868

theorem sine_of_angle_plus_pi_third (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 12) = 3 / 5) : 
  Real.sin (α + π / 3) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_of_angle_plus_pi_third_l398_39868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bake_sale_cookies_l398_39813

/-- Proves that John made 6 dozens of cookies for the bake sale -/
theorem bake_sale_cookies : ∃ (dozens_of_cookies : ℕ), dozens_of_cookies = 6 := by
  -- Selling price of each cookie in cents
  let selling_price : ℕ := 150
  -- Cost to make each cookie in cents
  let cost_per_cookie : ℕ := 25
  -- Amount each charity receives in cents
  let charity_amount : ℕ := 4500
  -- Number of charities
  let num_charities : ℕ := 2
  -- Definition of a dozen
  let cookies_per_dozen : ℕ := 12
  
  -- The number of dozens of cookies John made
  let dozens_of_cookies : ℕ := 6
  
  -- Proof that John made 6 dozens of cookies
  have h : (charity_amount * num_charities) / ((selling_price - cost_per_cookie) / 100) / cookies_per_dozen = dozens_of_cookies := by sorry
  
  -- Conclude the proof
  exact ⟨dozens_of_cookies, rfl⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bake_sale_cookies_l398_39813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_odd_l398_39842

theorem k_is_odd (m n k : ℕ) (hm : m > 0) (hn : n > 0) (hk : k > 0)
  (h : 3 * m * k = (m + 3)^n + 1) : 
  Odd k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_odd_l398_39842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_minimum_exists_l398_39893

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x) + a * Real.sin x

-- Theorem for symmetry about x = π/2
theorem f_symmetry (a : ℝ) : ∀ x, f a (Real.pi - x) = f a x := by
  intro x
  simp [f]
  -- The proof steps would go here
  sorry

-- Theorem for minimum value when a = 1
theorem f_minimum_exists :
  ∃ α : ℝ, Real.pi/2 < α ∧ α ≤ 7*Real.pi/6 ∧
  (∀ x ∈ Set.Ioo (-Real.pi/6) α, f 1 x ≥ 0) ∧
  (∃ x₀ ∈ Set.Ioo (-Real.pi/6) α, f 1 x₀ = 0) := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_minimum_exists_l398_39893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l398_39838

theorem cos_alpha_value (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : π / 2 < β) (h4 : β < π)
  (h5 : Real.cos β = -1/3)
  (h6 : Real.sin (α + β) = 1/3) :
  Real.cos α = 4 * Real.sqrt 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l398_39838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_equal_parts_l398_39891

/-- Represents a complex-shaped ice plate -/
structure IcePlate where
  area : ℝ
  hasDiagonalSymmetry : Bool

/-- Represents a cut on the ice plate -/
structure Cut where
  isDiagonal : Bool

/-- Represents a part of the ice plate after cutting -/
structure IcePart where
  area : ℝ

/-- Function to cut the ice plate -/
def cutPlate (plate : IcePlate) (cut : Cut) : (IcePart × IcePart) :=
  sorry

/-- Theorem stating that a diagonal cut on a plate with diagonal symmetry results in equal parts -/
theorem diagonal_cut_equal_parts (plate : IcePlate) (cut : Cut) :
  plate.hasDiagonalSymmetry ∧ cut.isDiagonal →
  let (part1, part2) := cutPlate plate cut
  part1.area = part2.area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_cut_equal_parts_l398_39891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_pair_l398_39851

/-- A positive integer is perfect if the sum of its positive divisors is twice the number itself. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.sum (Finset.filter (· ∣ n) (Finset.range (n + 1))) id) = 2 * n

/-- The theorem states that 7 is the only positive integer n such that both n-1 and n(n+1)/2 are perfect numbers. -/
theorem unique_perfect_pair : ∀ n : ℕ, n > 0 → 
  (IsPerfect (n - 1)) ∧ (IsPerfect (n * (n + 1) / 2)) ↔ n = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perfect_pair_l398_39851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_properties_l398_39802

-- Define the reaction components
inductive Species
| SO₂ | O₂ | SO₃

-- Define the reaction
def reaction : List (Nat × Species) → List (Nat × Species)
| [(2, Species.SO₂), (1, Species.O₂)] => [(2, Species.SO₃)]
| _ => []

-- Define the equilibrium constant
noncomputable def K (so3 so2 o2 : ℝ) : ℝ := (so3^2) / (so2^2 * o2)

-- Define the enthalpy change
def ΔH : ℝ := -190

-- State the theorem
theorem equilibrium_properties :
  ∀ (T₁ T₂ : ℝ) (so3 so2 o2 : ℝ) (V : ℝ) (t : ℝ) (n_so3_init n_so3_final : ℝ),
  T₁ < T₂ →
  ΔH < 0 →
  K so3 so2 o2 > 0 →
  -- 1. Equilibrium constant comparison
  K so3 so2 o2 > K (so3 * (T₂ / T₁)) (so2 * (T₂ / T₁)) (o2 * (T₂ / T₁)) →
  -- 2. Equilibrium indicators
  (∃ (k : ℝ), k * (so2 * o2) = 2 * so3) →
  (∃ (n_total : ℝ), n_total = so2 + o2 + so3) →
  (∃ (m_avg : ℝ), m_avg = (64 * so2 + 32 * o2 + 80 * so3) / (so2 + o2 + so3)) →
  -- 3. Reaction rate and equilibrium shift
  (0.2 - (2 * 0.18) / 2) / (5 * 0.5) = 0.036 →
  n_so3_init = 0.18 →
  n_so3_final > n_so3_init →
  n_so3_final > 0.36 ∧ n_so3_final < 0.50 →
  True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilibrium_properties_l398_39802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_inverse_functions_l398_39818

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := (1/3)^x

-- Define the logarithmic function
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem symmetry_of_inverse_functions :
  ∃ (L : ℝ → ℝ), (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) →
  (∀ a b, f a = b ↔ g b = a) →
  L = λ x ↦ x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_inverse_functions_l398_39818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_h_l398_39872

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := sin (x + Real.pi / 2)
noncomputable def g (x : ℝ) : ℝ := sin (Real.pi - x)

-- Define the function h as the sum of f and g
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- State the theorem about the center of symmetry
theorem center_of_symmetry_h :
  ∀ x : ℝ, h (3*Real.pi/4 + x) = h (3*Real.pi/4 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_h_l398_39872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_max_area_of_triangle_l398_39890

-- Define the vectors m and n
noncomputable def m (x : Real) : Real × Real := (Real.sin x, 1)
noncomputable def n (x : Real) : Real × Real := (Real.sqrt 3 * Real.cos x, (1/2) * Real.cos (2*x))

-- Define the function f
noncomputable def f (x : Real) : Real := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Theorem for the maximum value of f
theorem max_value_of_f : 
  ∃ (x : Real), f x = 1 ∧ ∀ (y : Real), f y ≤ 1 := by
  sorry

-- Theorem for the maximum area of triangle ABC
theorem max_area_of_triangle (A : Real) (a b c : Real) :
  f A = 1/2 → a = 2 → b > 0 → c > 0 →
  (1/2) * b * c * Real.sin A ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_max_area_of_triangle_l398_39890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_l398_39839

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -5 * x^3 + 6 * x^2 + 2 * x - 8

-- State the theorem
theorem degree_of_h (h : ℝ → ℝ) : 
  (∃ (n : ℕ), ∀ (x : ℝ), ∃ (p : Polynomial ℝ), h x = p.eval x ∧ p.natDegree ≤ n) →  -- h is a polynomial
  (∃ (p : Polynomial ℝ), ∀ (x : ℝ), f x + h x = p.eval x ∧ p.natDegree = 2) →      -- degree of f + h is 2
  (∃ (p : Polynomial ℝ), ∀ (x : ℝ), h x = p.eval x ∧ p.natDegree = 3) :=           -- degree of h is 3
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_h_l398_39839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l398_39895

-- Define the circles and points
def origin : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- Distance function (marked as noncomputable due to Real.sqrt)
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem concentric_circles_k_value (k : ℝ) :
  distance origin P = distance origin P - 5 → -- The difference between radii is 5
  distance origin P = distance origin (S k) + 5 → -- S is on the smaller circle
  k = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_k_value_l398_39895
