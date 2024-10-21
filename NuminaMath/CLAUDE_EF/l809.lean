import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_l809_80952

-- Define the arrival time range
def arrivalTimeRange : ℝ := 30

-- Define the probability of overlap
def overlapProbability : ℝ := 0.5

-- Define the structure of n
noncomputable def n (d e f : ℕ) : ℝ := d - e * Real.sqrt (f : ℝ)

-- Define the condition for f
def isPrimeSquareFree (f : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ f)

-- Main theorem
theorem mathematicians_meeting
  (d e f : ℕ)
  (hd : d > 0)
  (he : e > 0)
  (hf : f > 0)
  (hf_prime : isPrimeSquareFree f)
  (hn : n d e f = arrivalTimeRange - arrivalTimeRange * Real.sqrt (1 - overlapProbability)) :
  d + e + f = 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_l809_80952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eleven_pi_sixths_l809_80925

theorem cos_eleven_pi_sixths : Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_eleven_pi_sixths_l809_80925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_explosion_calculation_l809_80968

/-- Parameters for the explosion problem -/
structure ExplosionParams where
  a : ℝ -- Door width in meters
  b : ℝ -- Door height in meters
  m : ℝ -- Door mass in kg
  s : ℝ -- Horizontal distance traveled in meters
  h : ℝ -- Height of the floor in meters
  t : ℝ -- Temperature of exploding gas in Celsius
  α : ℝ -- Expansion coefficient of gas
  P₀ : ℝ -- Initial pressure in mm Hg
  T₀ : ℝ -- Initial temperature in Kelvin

/-- Results of the explosion calculation -/
structure ExplosionResults where
  force : ℝ -- Force exerted on the door in kg-weight
  duration : ℝ -- Duration of the explosion in seconds

/-- Theorem stating the relationship between the parameters and the results -/
theorem explosion_calculation (params : ExplosionParams) 
  (h1 : params.a = 2.20)
  (h2 : params.b = 1.15)
  (h3 : params.m = 30)
  (h4 : params.s = 80)
  (h5 : params.h = 6)
  (h6 : params.t = 1500)
  (h7 : params.α = 1 / 273)
  (h8 : params.P₀ = 760)
  (h9 : params.T₀ = 273) :
  ∃ (results : ExplosionResults), 
    (abs (results.force - 143200) < 1) ∧ 
    (abs (results.duration - 0.003) < 0.0001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_explosion_calculation_l809_80968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_between_parallel_lines_l809_80923

-- Define the setup
structure Geometry where
  Line : Type
  Point : Type
  Angle : Type
  parallel : Line → Line → Prop
  angle_measure : Angle → ℝ

variable (G : Geometry)

-- Define the problem statement
theorem angle_measure_between_parallel_lines 
  (p q : G.Line) 
  (X Y Z : G.Angle) 
  (h_parallel : G.parallel p q)
  (h_X : G.angle_measure X = 100)
  (h_Y : G.angle_measure Y = 140) :
  G.angle_measure Z = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_between_parallel_lines_l809_80923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l809_80936

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := -(3^(-x))

-- Theorem statement
theorem symmetry_about_origin (x : ℝ) : f x = -g (-x) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l809_80936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40pi_l809_80950

/-- The area between a circle circumscribing two externally tangent circles of radii 4 and 5 -/
noncomputable def shadedArea : ℝ := 40 * Real.pi

/-- The radius of the larger circumscribing circle -/
def largeRadius : ℝ := 4 + 5

theorem shaded_area_is_40pi :
  let smallCircle1Area := Real.pi * 4^2
  let smallCircle2Area := Real.pi * 5^2
  let largeCircleArea := Real.pi * largeRadius^2
  largeCircleArea - smallCircle1Area - smallCircle2Area = shadedArea := by
  -- Expand definitions
  unfold shadedArea largeRadius
  -- Simplify expressions
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40pi_l809_80950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l809_80910

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 9) + 1 / (x^2 - 6*x + 21) + Real.cos (2 * Real.pi * x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 1.25 := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l809_80910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_one_fifty_l809_80909

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieRoll
| One
| Two
| Three
| Four
| Five
| Six
| Seven
| Eight

/-- The probability of each outcome for a fair 8-sided die -/
noncomputable def dieProbability : ℝ := 1 / 8

/-- Determines if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- Calculates the winnings for a given roll -/
def winnings (roll : DieRoll) : ℝ :=
  match roll with
  | DieRoll.One => 0
  | DieRoll.Two => 2
  | DieRoll.Three => 3
  | DieRoll.Four => -4
  | DieRoll.Five => 5
  | DieRoll.Six => -6
  | DieRoll.Seven => 7
  | DieRoll.Eight => 5

/-- The expected value of winnings for one die toss -/
noncomputable def expectedValue : ℝ :=
  dieProbability * (winnings DieRoll.One +
                    winnings DieRoll.Two +
                    winnings DieRoll.Three +
                    winnings DieRoll.Four +
                    winnings DieRoll.Five +
                    winnings DieRoll.Six +
                    winnings DieRoll.Seven +
                    winnings DieRoll.Eight)

/-- Theorem stating that the expected value of winnings is $1.50 -/
theorem expected_value_is_one_fifty :
  expectedValue = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_one_fifty_l809_80909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l809_80904

/-- Given a point P(x,y,z) in the first octant of 3D space, where z = 2y, and the angles α, β, and γ
    between the line OP and the x-, y-, and z-axes respectively, if cos α = 2/5 and cos β = 3/5,
    then cos γ = 2√3/5. -/
theorem angle_cosine_relation (x y z : ℝ) (α β γ : ℝ) : 
  x > 0 → y > 0 → z > 0 →
  z = 2 * y →
  Real.cos α = 2/5 →
  Real.cos β = 3/5 →
  Real.cos α = x / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos β = y / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos γ = z / Real.sqrt (x^2 + y^2 + z^2) →
  Real.cos γ = 2 * Real.sqrt 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l809_80904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_actions_to_same_types_l809_80974

/-- Represents a box containing fruits -/
structure FruitBox where
  bananas : ℕ
  mangoes : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  black : FruitBox
  white : FruitBox

/-- Initial state of the boxes -/
def initialState : BoxState := {
  black := { bananas := 8, mangoes := 10 }
  white := { bananas := 0, mangoes := 12 }
}

/-- Checks if both boxes contain the same types of fruits -/
def sameTypes (state : BoxState) : Prop :=
  (state.black.bananas > 0 ↔ state.white.bananas > 0) ∧
  (state.black.mangoes > 0 ↔ state.white.mangoes > 0)

/-- Represents a single action -/
inductive FruitAction
  | eat : FruitAction
  | transfer : FruitAction

/-- Applies an action to the current state -/
def applyAction (state : BoxState) (action : FruitAction) : BoxState :=
  sorry

/-- Theorem stating the minimum number of actions required -/
theorem min_actions_to_same_types :
  ∀ (actions : List FruitAction),
    (∃ (finalState : BoxState),
      finalState = actions.foldl applyAction initialState ∧
      sameTypes finalState) →
    actions.length ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_actions_to_same_types_l809_80974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completion_time_l809_80915

/-- The number of days it takes for A to complete the job alone -/
noncomputable def x : ℝ := 15

/-- The fraction of the job that B can complete in one day -/
noncomputable def b_rate : ℝ := 1 / 20

/-- The number of days A and B work together -/
noncomputable def days_together : ℝ := 3

/-- The fraction of the job left after A and B work together -/
noncomputable def fraction_left : ℝ := 0.65

/-- Theorem stating the relationship between the variables and that x is positive -/
theorem a_completion_time :
  (days_together * (1 / x + b_rate) = 1 - fraction_left) ∧
  (x > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_completion_time_l809_80915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_seven_times_l809_80986

def roll_probability : ℚ := 1 / 2

def total_rolls : ℕ := 8

def min_successful_rolls : ℕ := 7

theorem probability_at_least_four_seven_times :
  (Finset.sum (Finset.range 2) (λ k => 
    (Nat.choose total_rolls (total_rolls - k)) * 
    (roll_probability ^ (total_rolls - k)) * 
    ((1 - roll_probability) ^ k))) = 9 / 256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_seven_times_l809_80986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_l_distance_to_l_l809_80964

-- Define the line l: √3x + y + √3 - 2 = 0
noncomputable def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y + Real.sqrt 3 - 2 = 0

-- Define the direction vector
noncomputable def direction_vector : ℝ × ℝ := (1, -Real.sqrt 3)

-- Define the point
def point : ℝ × ℝ := (-1, 0)

-- Theorem 1: (1, -√3) is a direction vector of l
theorem direction_vector_of_l :
  let (dx, dy) := direction_vector
  ∀ x y, line_l x y → line_l (x + dx) (y + dy) := by sorry

-- Theorem 2: The distance from the point (-1, 0) to l is 1
theorem distance_to_l :
  let (x₀, y₀) := point
  let d := |Real.sqrt 3 * x₀ + y₀ + Real.sqrt 3 - 2| / Real.sqrt (3 + 1)
  d = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_l_distance_to_l_l809_80964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_bobbi_same_number_probability_l809_80908

/-- The upper bound for the chosen numbers -/
def upperBound : ℕ := 500

/-- Billy's divisor -/
def billyDivisor : ℕ := 30

/-- Bobbi's divisor -/
def bobbiDivisor : ℕ := 45

/-- The probability of Billy and Bobbi selecting the same number -/
def sameProbability : ℚ := 5 / 176

theorem billy_bobbi_same_number_probability :
  let billyChoices := Finset.filter (fun n => n > 0 ∧ n < upperBound ∧ billyDivisor ∣ n) (Finset.range upperBound)
  let bobbiChoices := Finset.filter (fun n => n > 0 ∧ n < upperBound ∧ bobbiDivisor ∣ n) (Finset.range upperBound)
  let sameChoices := billyChoices ∩ bobbiChoices
  (sameChoices.card : ℚ) / ((billyChoices.card : ℚ) * (bobbiChoices.card : ℚ)) = sameProbability :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_bobbi_same_number_probability_l809_80908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_and_tangent_line_l809_80932

-- Define the circle on which point A moves
def circle_A (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 16

-- Define the midpoint of a segment
def midpoint_AB (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

-- Define the trajectory of point M
def trajectory_M (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define the line passing through C and intercepting equal lengths on axes
def line_through_C (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y = 0

-- Define the tangent line
def tangent_line (a : ℝ) (x y : ℝ) : Prop := x + y = a + 2

-- Main theorem
theorem midpoint_trajectory_and_tangent_line :
  ∃ (a : ℝ), a > 0 ∧
  (∀ (x y : ℝ), 
    (∃ (mx my : ℝ), midpoint_AB x y (-1) 0 mx my ∧ circle_A x y) → 
    trajectory_M mx my) ∧
  (∃ (x y : ℝ), 
    trajectory_M x y ∧ 
    line_through_C a x y ∧ 
    tangent_line a x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_and_tangent_line_l809_80932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_ratio_l809_80965

-- Define the square ABCD and triangle XYZ
variable (A B C D X Y Z : ℝ × ℝ)

-- Define the areas
noncomputable def area_square (A B C D : ℝ × ℝ) : ℝ := sorry
noncomputable def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define the condition that ABCD is a square
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the condition that the area of ABCD is 7/32 of the area of XYZ
def area_ratio (A B C D X Y Z : ℝ × ℝ) : Prop :=
  area_square A B C D = (7/32) * area_triangle X Y Z

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem square_triangle_ratio
  (h_square : is_square A B C D)
  (h_ratio : area_ratio A B C D X Y Z) :
  (distance X A) / (distance X Y) = 7/8 ∨ (distance X A) / (distance X Y) = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_ratio_l809_80965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ananthu_work_rate_l809_80967

/-- Represents the number of days a person takes to complete the work alone -/
structure WorkRate where
  days : ℚ
  days_positive : days > 0

/-- Represents the portion of work completed -/
noncomputable def work_completed (rate : WorkRate) (days : ℚ) : ℚ :=
  days / rate.days

theorem ananthu_work_rate (amit_rate : WorkRate) 
  (h_amit : amit_rate.days = 10)
  (amit_days : ℚ)
  (h_amit_days : amit_days = 2)
  (total_days : ℚ)
  (h_total_days : total_days = 18) : 
  ∃ (ananthu_rate : WorkRate), 
    work_completed amit_rate amit_days + 
    work_completed ananthu_rate (total_days - amit_days) = 1 ∧ 
    ananthu_rate.days = 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ananthu_work_rate_l809_80967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l809_80961

theorem trig_identity_proof (α : ℝ) : 
  (Real.cos (2 * α)) / (4 * (Real.sin (π / 4 + α))^2 * Real.tan (π / 4 - α)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_proof_l809_80961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_not_in_set_A_f₂_in_set_A_l809_80940

noncomputable def set_A (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x ≠ y → f x + 2 * f y > 3 * f ((x + 2 * y) / 3)

noncomputable def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 2

def f₂ (x : ℝ) : ℝ := (x + 1) ^ 2

theorem f₁_not_in_set_A : ¬(set_A f₁) := by sorry

theorem f₂_in_set_A : set_A f₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_not_in_set_A_f₂_in_set_A_l809_80940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_height_is_six_sevenths_l809_80972

/-- Represents a tetrahedron formed by three perpendicular rods -/
structure Tetrahedron where
  a : ℝ  -- Length of the first rod
  b : ℝ  -- Length of the second rod
  c : ℝ  -- Length of the third rod

/-- Calculates the height of the tetrahedron -/
noncomputable def tetrahedron_height (t : Tetrahedron) : ℝ :=
  let base_area := (Real.sqrt (t.a^2 + t.b^2) * Real.sqrt (t.a^2 + t.c^2) * 
    Real.sqrt (1 - 1 / (t.a^2 + t.b^2 + t.c^2))) / 2
  3 * (t.a * t.b / 2) * t.c / base_area

theorem tetrahedron_height_is_six_sevenths :
  let t : Tetrahedron := ⟨1, 2, 3⟩
  tetrahedron_height t = 6/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_height_is_six_sevenths_l809_80972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_payment_is_180_l809_80992

/-- Represents the rental arrangement for a pasture -/
structure PastureRental where
  totalRent : ℕ
  aHorses : ℕ
  aMonths : ℕ
  bHorses : ℕ
  bMonths : ℕ
  cHorses : ℕ
  cMonths : ℕ

/-- Calculates the payment for person B given a PastureRental arrangement -/
def calculateBPayment (rental : PastureRental) : ℕ :=
  let totalHorseMonths := rental.aHorses * rental.aMonths + 
                          rental.bHorses * rental.bMonths + 
                          rental.cHorses * rental.cMonths
  let costPerHorseMonth := rental.totalRent / totalHorseMonths
  rental.bHorses * rental.bMonths * costPerHorseMonth

/-- Theorem stating that B's payment for the given arrangement is 180 Rs -/
theorem b_payment_is_180 (rental : PastureRental) 
  (h1 : rental.totalRent = 435)
  (h2 : rental.aHorses = 12) (h3 : rental.aMonths = 8)
  (h4 : rental.bHorses = 16) (h5 : rental.bMonths = 9)
  (h6 : rental.cHorses = 18) (h7 : rental.cMonths = 6) :
  calculateBPayment rental = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_payment_is_180_l809_80992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_sin_squared_alpha_l809_80994

theorem cos_2alpha_plus_sin_squared_alpha 
  (α : ℝ) 
  (h : Real.cos α = 3/5) : 
  Real.cos (2*α) + Real.sin α^2 = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_sin_squared_alpha_l809_80994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_coefficients_l809_80956

/-- A system of linear equations with unknown coefficients otimes and oplus -/
structure EquationSystem where
  eq1 : ℝ → ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ → ℝ
  solution_x : ℝ
  solution_y : ℝ

/-- The specific equation system from the problem -/
def problemSystem : EquationSystem where
  eq1 := fun x y otimes => x + otimes * y - 3
  eq2 := fun x y oplus => 3 * x - oplus * y - 1
  solution_x := 1
  solution_y := 1

/-- Theorem stating the values of otimes and oplus -/
theorem find_coefficients (system : EquationSystem) :
  system.eq1 system.solution_x system.solution_y 2 = 0 ∧
  system.eq2 system.solution_x system.solution_y 1 = 0 ∧
  system.solution_x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_coefficients_l809_80956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1999_prime_arithmetic_progression_below_12345_l809_80970

theorem no_1999_prime_arithmetic_progression_below_12345 :
  ∀ (a d : ℕ), (∀ k : Fin 1999, Prime (a + k * d)) →
  (∀ i j : Fin 1999, i ≠ j → a + i * d ≠ a + j * d) →
  ∃ k : Fin 1999, a + k * d ≥ 12345 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1999_prime_arithmetic_progression_below_12345_l809_80970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_preferred_function_l809_80984

open MeasureTheory Measure Set Real

theorem min_integral_preferred_function (f : ℝ → ℝ) 
  (hf_cont : ContinuousOn f (Icc 0 1))
  (hf_cond : ∀ x y, x ∈ Icc 0 1 → y ∈ Icc 0 1 → f x + f y ≥ |x - y|) :
  ∫ x in Icc 0 1, f x ≥ 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_preferred_function_l809_80984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_digit_erasure_l809_80985

/-- A 100-digit natural number -/
def HundredDigitNumber : Type := { n : ℕ // 10^99 ≤ n ∧ n < 10^100 }

/-- Represents the result of erasing two adjacent digits from a number -/
def EraseAdjacentDigits (N : ℕ) : Set ℕ :=
  { N' | ∃ (k : ℕ) (a : ℕ), 0 ≤ a ∧ a < 100 ∧
    ∃ (m n : ℕ), m < 10^k ∧
    N = m + 10^k * a + 10^(k+2) * n ∧
    N' = m + 10^k * n }

/-- The main theorem -/
theorem adjacent_digit_erasure (N : HundredDigitNumber) :
  (∃ N' ∈ EraseAdjacentDigits N.val, N.val = 87 * N') →
  (N.val = 435 * 10^97 ∨ N.val = 1305 * 10^96 ∨ N.val = 2175 * 10^96 ∨ N.val = 3045 * 10^96) := by
  sorry

#check adjacent_digit_erasure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_digit_erasure_l809_80985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_ratio_l809_80933

-- Define the circle and its properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points on the circle
def Point : Type := ℝ × ℝ

-- Define the circle S
noncomputable def S : Circle := { center := (0, 0), radius := 1 }

-- Define the diameters and points
noncomputable def A : Point := (-1, 0)
noncomputable def B : Point := (1, 0)
noncomputable def C : Point := (0, 1)
noncomputable def D : Point := (0, -1)

-- Define the intersection points
noncomputable def K : Point := (0, -1/3)
noncomputable def L : Point := (1/2, 0)

-- Define the ratio function
noncomputable def ratio (p q r : Point) : ℝ := sorry

-- Define a function to check if a point is on the circle
def onCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_chord_ratio (E : Point) :
  onCircle S E →
  ratio C K D = 2 / 1 →
  ratio A L B = 3 / 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_ratio_l809_80933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equals_one_l809_80975

-- Define the parabola P
noncomputable def P (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Define the vertex and focus of P
noncomputable def V1 : ℝ × ℝ := (1, 1)
noncomputable def F1 : ℝ × ℝ := (1, 5/4)

-- Define points A and B on P
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the condition that tangent lines at A and B are perpendicular
def tangents_perpendicular (A B : ℝ × ℝ) : Prop := sorry

-- Define the locus Q (midpoint of AB)
noncomputable def Q (A B : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the vertex and focus of Q
noncomputable def V2 : ℝ × ℝ := (0, 1)
noncomputable def F2 : ℝ × ℝ := (0, 5/4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem ratio_equals_one :
  ∀ (A B : ℝ × ℝ),
  A.2 = P A.1 →
  B.2 = P B.1 →
  tangents_perpendicular A B →
  (distance F1 F2) / (distance V1 V2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_equals_one_l809_80975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l809_80938

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  major_axis : ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

/-- A line intersecting the ellipse -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 2

/-- The length of the chord formed by the intersection of the line and the ellipse -/
noncomputable def chord_length (e : Ellipse) : ℝ :=
  16 * Real.sqrt 2 / 5

/-- Theorem stating the properties of the ellipse and its intersection -/
theorem ellipse_properties (e : Ellipse) :
  e.center = (0, 0) →
  e.focus1 = (-2 * Real.sqrt 3, 0) →
  e.focus2 = (2 * Real.sqrt 3, 0) →
  e.major_axis = 8 →
  (∀ x y : ℝ, ellipse_equation e x y ↔ x^2 / 16 + y^2 / 4 = 1) ∧
  chord_length e = 16 * Real.sqrt 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l809_80938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l809_80937

theorem trigonometric_identities (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin α = 3/5) : 
  Real.sin (π/4 + α) = -Real.sqrt 2/10 ∧ 
  Real.cos (π/6 - 2*α) = (7*Real.sqrt 3 - 24)/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l809_80937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2490_and_cos_52pi_over_3_l809_80991

theorem sin_2490_and_cos_52pi_over_3 :
  Real.sin (2490 * Real.pi / 180) = -1/2 ∧ Real.cos (-52/3 * Real.pi) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2490_and_cos_52pi_over_3_l809_80991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l809_80954

/-- Given a triangle ABC with the following properties:
  * The measure of angle A is π/3
  * The length of side b is 2a cos B, where a is the length of side opposite to angle A
  * The length of side c is 1
  Prove that the area of triangle ABC is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  A = π/3 → 
  b = 2*a*(Real.cos B) → 
  c = 1 → 
  (1/2) * b * c * (Real.sin A) = Real.sqrt 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l809_80954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_implies_positive_exponent_l809_80948

-- Define the power function
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ Real.rpow x a

-- Define what it means for a function to be increasing on (0, +∞)
def is_increasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- Theorem statement
theorem power_function_increasing_implies_positive_exponent :
  ∀ a : ℝ, is_increasing_on_positive_reals (power_function a) → a > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_implies_positive_exponent_l809_80948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_theorem_l809_80901

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The radius of the circle in feet -/
def r : ℝ := 50

/-- The total distance traveled when connecting all non-adjacent points on a circle -/
noncomputable def total_distance (n : ℕ) (r : ℝ) : ℝ :=
  n * (2 * r * 2 + 2 * (r * 2 * (Real.sqrt 2)))

theorem circle_distance_theorem :
  total_distance n r = 1600 + 800 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_theorem_l809_80901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l809_80946

/-- Represents a triangle with vertices A, B, and C. -/
def Triangle (A B C : Point) : Prop := sorry

/-- Represents cyclic order for triangle side division. -/
inductive CyclicOrder
| Cyclic
| ReverseCyclic

/-- Represents the division of sides of one triangle to form another triangle
    with a given ratio and order. -/
def DivideTriangleSides (A B C A₁ B₁ C₁ : Point) (ratio : ℝ) (order : CyclicOrder) : Prop := sorry

/-- Represents that two triangles are similar. -/
def SimilarTriangles (A B C A₁ B₁ C₁ : Point) : Prop := sorry

/-- Represents that two triangles are similarly oriented. -/
def SimilarlyOriented (A B C A₁ B₁ C₁ : Point) : Prop := sorry

/-- Given a triangle ABC and a ratio λ, if the sides of ABC are divided in ratio λ cyclically 
    to form A₁B₁C₁, and the sides of A₁B₁C₁ are divided in ratio λ in reverse cyclic order 
    to form A₂B₂C₂, then A₂B₂C₂ is similar and similarly oriented to ABC. -/
theorem triangle_similarity (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point) (ratio : ℝ) 
  (h_abc : Triangle A B C)
  (h_a₁b₁c₁ : DivideTriangleSides A B C A₁ B₁ C₁ ratio CyclicOrder.Cyclic)
  (h_a₂b₂c₂ : DivideTriangleSides A₁ B₁ C₁ A₂ B₂ C₂ ratio CyclicOrder.ReverseCyclic) :
  SimilarTriangles A B C A₂ B₂ C₂ ∧ SimilarlyOriented A B C A₂ B₂ C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_l809_80946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_fourth_quadrant_m_value_for_coefficient_z_value_when_m_is_5_l809_80945

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (3*m - 2) + (m - 1)*Complex.I

-- Statement 1: z is purely imaginary when m = 2/3
theorem z_purely_imaginary : z (2/3) = 0 + ((-1/3) : ℝ)*Complex.I := by sorry

-- Statement 2: z is in the fourth quadrant when 2/3 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) :
  0 < (z m).re ∧ (z m).im < 0 := by sorry

-- Statement 3: m = 5 when the coefficient of the third term in (1+2x)^m is 40
theorem m_value_for_coefficient (m : ℕ) (h : m > 0) :
  (Nat.choose m 2) * 2^2 = 40 → m = 5 := by sorry

-- Statement 4: z = 13 + 4i when m = 5
theorem z_value_when_m_is_5 : z 5 = 13 + 4*Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_fourth_quadrant_m_value_for_coefficient_z_value_when_m_is_5_l809_80945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_min_value_on_interval_max_value_on_interval_inequality_for_g_l809_80957

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.cos x

-- Define the derivative of f as g
noncomputable def g (x : ℝ) : ℝ := 2*x - Real.sin x

theorem tangent_line_at_zero (x : ℝ) :
  (fun y => y = 1) = (fun y => y - f 0 = g 0 * (x - 0)) :=
by sorry

theorem min_value_on_interval :
  ∀ x ∈ Set.Icc (-2*Real.pi) (2*Real.pi), f x ≥ 1 :=
by sorry

theorem max_value_on_interval :
  ∀ x ∈ Set.Icc (-2*Real.pi) (2*Real.pi), f x ≤ 4*Real.pi^2 + 1 :=
by sorry

theorem inequality_for_g (s t : ℝ) (h : s > t) :
  g s - g t < 3*s - 3*t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_min_value_on_interval_max_value_on_interval_inequality_for_g_l809_80957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_is_four_l809_80947

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Theorem: The rate percent is 4% when the simple interest is 320 for 2 years -/
theorem rate_percent_is_four (principal : ℝ) (h : simple_interest principal 4 2 = 320) :
  4 = (320 * 100) / (principal * 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rate_percent_is_four_l809_80947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_constrained_l809_80943

/-- The maximum number of positive factors for b^n given constraints -/
theorem max_factors_constrained (b n : ℕ) (hb : b ≤ 20) (hn : n ≤ 15) :
  (∃ (k : ℕ), k = b^n ∧ (∀ (m : ℕ), m = b^n → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors k))) →
  Nat.card (Nat.divisors (b^n)) ≤ 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_factors_constrained_l809_80943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l809_80958

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 160) : ℝ := by
  let side_length := perimeter / 8
  let area := 4 * side_length ^ 2
  have h2 : area = 1600 := by
    -- Proof steps would go here
    sorry
  exact area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l809_80958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_x_greater_than_e_g_has_two_distinct_roots_l809_80928

-- Define the function f(x) = ae^x - x^2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x^2

-- Part 1: Prove that when a = 1 and x > e, f(x) > 0
theorem f_positive_when_x_greater_than_e (x : ℝ) (h : x > Real.exp 1) : f 1 x > 0 := by
  sorry

-- Define g(x) = f(x) + x^2 - x = ae^x - x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x

-- Part 2: Prove that g(x) has two distinct real roots iff 0 < a < 1/e
theorem g_has_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0) ↔ (0 < a ∧ a < 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_when_x_greater_than_e_g_has_two_distinct_roots_l809_80928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l809_80995

noncomputable def move (z : ℂ) : ℂ := z * Complex.exp (Complex.I * Real.pi / 6) + 8

noncomputable def final_position (initial_position : ℂ) (num_moves : ℕ) : ℂ :=
  (move^[num_moves]) initial_position

theorem particle_returns_to_start :
  final_position 6 120 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_returns_to_start_l809_80995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_extra_fee_l809_80918

/-- Represents the data plan and usage for a 4-week period -/
structure DataPlan where
  weeklyLimits : Fin 4 → ℚ
  weeklyFees : Fin 4 → ℚ
  weeklyUsage : Fin 4 → ℚ

/-- Calculates the extra fee for a given week -/
def extraFee (plan : DataPlan) (week : Fin 4) : ℚ :=
  max 0 ((plan.weeklyUsage week - plan.weeklyLimits week) * plan.weeklyFees week)

/-- Calculates the total extra fee for the 4-week period -/
def totalExtraFee (plan : DataPlan) : ℚ :=
  (Finset.sum Finset.univ fun i => extraFee plan i)

/-- The specific data plan and usage for Charlie -/
def charliePlan : DataPlan := {
  weeklyLimits := fun i => match i with
    | 0 => 2 | 1 => 3 | 2 => 2 | 3 => 1
  weeklyFees := fun i => match i with
    | 0 => 12 | 1 => 10 | 2 => 8 | 3 => 6
  weeklyUsage := fun i => match i with
    | 0 => 5/2 | 1 => 4 | 2 => 3 | 3 => 5
}

/-- Theorem stating that Charlie's total extra fee is $48 -/
theorem charlie_extra_fee : totalExtraFee charliePlan = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlie_extra_fee_l809_80918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_numbers_in_A_P_l809_80912

-- Define a polynomial of degree 8
def MyPolynomial (α : Type*) [Ring α] := α → α

-- Define the set A_P
def A_P (P : MyPolynomial ℝ) (c : ℝ) : Set ℝ := {x | P x = c}

-- Define a predicate for odd integers
def IsOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- State the theorem
theorem min_odd_numbers_in_A_P (P : MyPolynomial ℝ) (c : ℝ) :
  (∃ (x : ℝ), x ∈ A_P P c ∧ x = 8) →
  (∃ (y : ℝ), y ∈ A_P P c ∧ ∃ (n : Int), ↑n = y ∧ IsOdd n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_odd_numbers_in_A_P_l809_80912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_power_eight_l809_80924

open Real MeasureTheory Interval

theorem integral_sin_power_eight : 
  ∫ x in (0 : ℝ)..π, (2^4 * sin x^8) = (35 * π) / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sin_power_eight_l809_80924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l809_80935

/-- A line passing through the second and fourth quadrants -/
structure QuadrantLine where
  passes_through_second_fourth : Bool

/-- The inclination angle of a line -/
def inclination_angle (l : QuadrantLine) : ℝ := sorry

/-- Theorem: The inclination angle of a line passing through the second and fourth quadrants 
    is between 90° and 180° -/
theorem inclination_angle_range (l : QuadrantLine) 
  (h : l.passes_through_second_fourth = true) : 
  90 < inclination_angle l ∧ inclination_angle l < 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l809_80935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_mile_speed_l809_80969

-- Define the problem parameters
def total_distance : ℝ := 2
def normal_speed : ℝ := 4
def first_mile_speed : ℝ := 3

-- Define the theorem
theorem last_mile_speed : 
  (1 / (total_distance / normal_speed - 1 / first_mile_speed)) = 6 :=
by
  -- Unfold the definitions
  unfold total_distance normal_speed first_mile_speed
  
  -- Simplify the expression
  simp
  
  -- Perform algebraic manipulations
  ring
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_mile_speed_l809_80969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_4001_approx_greater_l809_80999

-- Define the square root function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the linear approximation function
noncomputable def linear_approx (x x₀ : ℝ) : ℝ := f x₀ + (deriv f x₀) * (x - x₀)

-- State the theorem
theorem sqrt_4001_approx_greater :
  linear_approx 4.001 4 > f 4.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_4001_approx_greater_l809_80999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nStaircaseDissectionExists_l809_80953

/-- Definition of an n-staircase -/
def nStaircase (n : ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ≤ n ∧ p.2 ≤ p.1}

/-- The size of an n-staircase -/
def nStaircaseSize (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- A dissection of an n-staircase into smaller n-staircases -/
def nStaircaseDissection (n : ℕ) : 
  Set (Set (ℕ × ℕ)) :=
  {S | ∃ k : ℕ, k < n ∧ S = nStaircase k}

/-- The theorem stating that an n-staircase can be dissected into smaller n-staircases -/
theorem nStaircaseDissectionExists (n : ℕ) (h : n > 0) :
  ∃ D : Set (Set (ℕ × ℕ)), 
    (∀ S ∈ D, S ∈ nStaircaseDissection n) ∧ 
    (⋃₀ D = nStaircase n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nStaircaseDissectionExists_l809_80953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_l809_80989

theorem cosine_value_from_sine (α : ℝ) :
  Real.sin (π / 5 - α) = 1 / 3 →
  Real.cos ((3 * π) / 10 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_from_sine_l809_80989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l809_80951

theorem set_elements_problem (P Q : Finset ℕ) 
  (h1 : P.card = 3 * Q.card)
  (h2 : (P ∪ Q).card = 4500)
  (h3 : (P ∩ Q).card = 1500) :
  P.card = 3375 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_elements_problem_l809_80951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_atlas_population_closest_to_target_l809_80959

-- Define the population growth function
def population (initial : ℕ) (year : ℕ) : ℕ :=
  initial * (4 ^ ((year - 2000) / 20))

-- Define a function to calculate the difference from the target population
def populationDifference (initial : ℕ) (year : ℕ) (target : ℕ) : ℕ :=
  Int.natAbs (population initial year - target)

-- Theorem statement
theorem atlas_population_closest_to_target (initial : ℕ) (target : ℕ) :
  initial = 400 →
  target = 12800 →
  (populationDifference initial 2060 target < populationDifference initial 2040 target) ∧
  (populationDifference initial 2060 target < populationDifference initial 2080 target) ∧
  (populationDifference initial 2060 target < populationDifference initial 2100 target) :=
by
  intros h_initial h_target
  sorry

#eval population 400 2060
#eval populationDifference 400 2060 12800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_atlas_population_closest_to_target_l809_80959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l809_80955

-- Define sets A and B
def A : Set ℝ := {x | 2 * x^2 - 5 * x < 0}
def B : Set ℝ := {x | 3^(x - 1) ≥ Real.sqrt 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (3/2) (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l809_80955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assassination_assignment_l809_80927

-- Define the assassination attempts
inductive Attempt
| Christmas
| NewYears
| JulyFourteenth

-- Define the terrorist groups
inductive TerroristGroup
| Corsican
| Breton
| Basque

-- Define a function type that assigns a group to each attempt
def Assignment := Attempt → TerroristGroup

-- Define the statements
def statement1 (a : Assignment) : Prop :=
  a Attempt.Christmas = TerroristGroup.Basque ∧ a Attempt.NewYears ≠ TerroristGroup.Basque

def statement2 (a : Assignment) : Prop :=
  a Attempt.JulyFourteenth ≠ TerroristGroup.Breton

-- Define the condition that exactly one statement is true
def oneStatementTrue (a : Assignment) : Prop :=
  (statement1 a ∧ ¬statement2 a) ∨
  (¬statement1 a ∧ statement2 a)

-- Define the correct assignment
def correctAssignment : Assignment :=
  fun
  | Attempt.Christmas => TerroristGroup.Basque
  | Attempt.NewYears => TerroristGroup.Corsican
  | Attempt.JulyFourteenth => TerroristGroup.Breton

-- The theorem to prove
theorem assassination_assignment :
  ∀ a : Assignment,
    oneStatementTrue a →
    a = correctAssignment :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_assassination_assignment_l809_80927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l809_80916

theorem cos_sin_equation (x : ℝ) : 
  (Real.cos x - 3 * Real.sin x = 2) → 
  (3 * Real.sin x + Real.cos x = 0 ∨ 3 * Real.sin x + Real.cos x = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_l809_80916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l809_80913

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- The quadrilateral vertices -/
def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨2006, 2007⟩

theorem quadrilateral_area : 
  triangleArea A B C + triangleArea A C D = 2007 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l809_80913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_gas_cost_l809_80934

/-- Represents a road trip with given parameters -/
structure RoadTrip where
  speed1 : ℚ  -- Speed for the first part of the trip (mph)
  time1 : ℚ   -- Time for the first part of the trip (hours)
  speed2 : ℚ  -- Speed for the second part of the trip (mph)
  time2 : ℚ   -- Time for the second part of the trip (hours)
  mpg : ℚ     -- Miles per gallon
  total_cost : ℚ -- Total cost of gas for the trip

/-- Calculates the cost per gallon of gas for a given road trip -/
def cost_per_gallon (trip : RoadTrip) : ℚ :=
  let total_distance := trip.speed1 * trip.time1 + trip.speed2 * trip.time2
  let gallons_used := total_distance / trip.mpg
  trip.total_cost / gallons_used

/-- Theorem stating that for the given road trip parameters, the cost per gallon is $2 -/
theorem road_trip_gas_cost :
  let trip := RoadTrip.mk 60 2 50 3 30 18
  cost_per_gallon trip = 2 := by
  -- Proof goes here
  sorry

#eval cost_per_gallon (RoadTrip.mk 60 2 50 3 30 18)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_gas_cost_l809_80934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l809_80962

theorem problem_statement (x : ℝ) : 
  let N : ℝ := (2 : ℝ)^((2 : ℝ)^2)
  (N * N^(N^N) = (2 : ℝ)^((2 : ℝ)^x)) → x = 66 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l809_80962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_315_base2_l809_80900

/-- Convert a natural number to a list of its binary digits -/
def to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 2) :: aux (m / 2)
  aux n |>.reverse

/-- The sum of the digits in the base-2 representation of 315 (base 10) is 6. -/
theorem sum_of_digits_315_base2 : 
  (to_binary 315).sum = 6 := by
  sorry

#eval (to_binary 315).sum -- This will evaluate to 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_315_base2_l809_80900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l809_80977

/-- Conversion factor from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s (x : ℝ) : ℝ := x * (1000 / 3600)

/-- Time taken for a train to cross an electric pole -/
noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_per_hr : ℝ) : ℝ :=
  train_length / (km_per_hr_to_m_per_s train_speed_km_per_hr)

/-- Theorem: A train 100 meters long traveling at 180 km/hr takes 2 seconds to cross an electric pole -/
theorem train_crossing_time :
  time_to_cross 100 180 = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l809_80977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l809_80987

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r satisfies the equation S = a / (1 - r) -/
noncomputable def infinite_geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: An infinite geometric series with first term 500 and sum 2500 has a common ratio of 4/5 -/
theorem infinite_geometric_series_ratio : 
  ∃ (r : ℝ), infinite_geometric_series_sum 500 r = 2500 ∧ r = 4/5 := by
  use 4/5
  constructor
  · simp [infinite_geometric_series_sum]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_ratio_l809_80987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_sum_l809_80926

/-- Given positive real numbers a and b satisfying the specified conditions, prove that ab = 10^117 -/
theorem product_of_logarithmic_sum (a b : ℝ) : 
  a > 0 → b > 0 →
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    Real.sqrt (Real.log a / Real.log 10) = m ∧
    Real.sqrt (Real.log b / Real.log 10) = n ∧
    Real.log (Real.sqrt a) / Real.log 10 = m^2 / 2 ∧
    Real.log (Real.sqrt b) / Real.log 10 = n^2 / 2 ∧
    m + n + m^2 / 2 + n^2 / 2 = 108) →
  a * b = 10^117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_logarithmic_sum_l809_80926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_distance_l809_80920

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def line (k m x y : ℝ) : Prop := y = k * x + m

noncomputable def distance_point_to_line (k m : ℝ) : ℝ := |m| / Real.sqrt (1 + k^2)

theorem ellipse_line_intersection_distance (k m : ℝ) :
  (∃ E F : ℝ × ℝ, 
    ellipse E.1 E.2 ∧ 
    ellipse F.1 F.2 ∧
    line k m E.1 E.2 ∧ 
    line k m F.1 F.2 ∧
    (E.1 * F.1 + E.2 * F.2 = 0)) →
  distance_point_to_line k m = 2 * Real.sqrt 5 / 5 := by
  sorry

#check ellipse_line_intersection_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_distance_l809_80920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l809_80981

theorem trigonometric_identities :
  (Real.cos (15 * π / 180))^2 - (Real.sin (15 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (π / 8) * Real.cos (π / 8) = Real.sqrt 2 / 4 ∧
  Real.tan (15 * π / 180) = 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l809_80981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_tan_α_value_l809_80973

-- Define the angle α and point P(m, 1)
variable (α : Real)
variable (m : Real)

-- Define the conditions
axiom cos_α : Real.cos α = -1/3
axiom point_on_terminal_side : m^2 + 1^2 ≠ 0 ∧ Real.cos α = m / Real.sqrt (m^2 + 1)

-- Theorem for the value of m
theorem m_value : m = -Real.sqrt 2 / 4 := by sorry

-- Theorem for the value of tan α
theorem tan_α_value : Real.tan α = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_tan_α_value_l809_80973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l809_80988

-- Define the curves
def C₁ (x y : ℝ) : Prop := y^2 = 8*x
def C₂ (x y : ℝ) : Prop := (x-2)^2 + y^2 = 1

-- Define the distance ratio
noncomputable def distanceRatio (x y : ℝ) : ℝ :=
  (Real.sqrt (x^2 + y^2)) / (Real.sqrt ((x-2)^2 + y^2))

theorem max_distance_ratio :
  ∃ (max : ℝ), max = (4 * Real.sqrt 7) / 7 ∧
  ∀ (x y : ℝ), C₁ x y → C₂ x y →
  distanceRatio x y ≤ max := by
  sorry

#check max_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ratio_l809_80988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_is_minimum_count_lowest_score_ways_is_correct_l809_80966

/-- Represents the scoring function for a card game -/
def lowest_score (A B C : ℕ) : ℕ :=
  min (A * C) (min (3 * A * B) (2 * B * C))

/-- Theorem stating that the lowest possible score is given by the lowest_score function -/
theorem lowest_score_is_minimum (A B C : ℕ) :
  ∀ score : ℕ, score ≥ lowest_score A B C := by
  sorry

/-- Function to count the number of ways to achieve the lowest score -/
def count_lowest_score_ways (A B C : ℕ) : ℕ :=
  if 2 * B > A ∧ 3 * B > C then 1
  else if 2 * B = A ∧ 2 * B > C then C + 1
  else if 3 * B = C ∧ 3 * B > A then A + 1
  else if A = C ∧ A > 2 * B then B + 1
  else if 2 * B = A ∧ 3 * B = C then A + B + C
  else 1

/-- Theorem stating that count_lowest_score_ways correctly counts the number of ways -/
theorem count_lowest_score_ways_is_correct (A B C : ℕ) :
  count_lowest_score_ways A B C = 
    (count_lowest_score_ways A B C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_score_is_minimum_count_lowest_score_ways_is_correct_l809_80966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l809_80982

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sqrt 3 * Real.sin x

noncomputable def axis_of_symmetry (k : ℤ) : ℝ := k * Real.pi - Real.pi / 3

def range_of_x (k : ℤ) : Set ℝ := Set.Icc (- 2 * Real.pi / 3 + 2 * k * Real.pi) (2 * k * Real.pi)

theorem function_properties :
  (∀ x : ℝ, ∃ k : ℤ, f (x + axis_of_symmetry k) = f (axis_of_symmetry k - x)) ∧
  (∀ x : ℝ, f x ≥ 1 ↔ ∃ k : ℤ, x ∈ range_of_x k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l809_80982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_problem_l809_80930

/-- Definition of sister conic sections -/
def sister_conics (C₁ C₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, C₁ = {p : ℝ × ℝ | (p.1^2/a^2) + (p.2^2/b^2) = 1} ∧
             C₂ = {p : ℝ × ℝ | (p.1^2/a^2) - (p.2^2/b^2) = 1}

/-- Definition of eccentricity for ellipse and hyperbola -/
noncomputable def eccentricity (C : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Definition of vertex of a conic section -/
def is_vertex (p : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Definition of slope of a line through two points -/
noncomputable def line_slope (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

theorem conic_sections_problem
  (C₁ : Set (ℝ × ℝ))
  (b : ℝ)
  (h₁ : C₁ = {p : ℝ × ℝ | (p.1^2/4) + (p.2^2/b^2) = 1})
  (h₂ : 0 < b)
  (h₃ : b < 2)
  (C₂ : Set (ℝ × ℝ))
  (h₄ : sister_conics C₁ C₂)
  (e₁ : ℝ)
  (h₅ : e₁ = eccentricity C₁)
  (e₂ : ℝ)
  (h₆ : e₂ = eccentricity C₂)
  (h₇ : e₁ * e₂ = Real.sqrt 15 / 4)
  (M N : ℝ × ℝ)
  (h₈ : is_vertex M C₁)
  (h₉ : is_vertex N C₁)
  (G : ℝ × ℝ)
  (h₁₀ : G = (4, 0))
  (A B : ℝ × ℝ)
  (h₁₁ : A ∈ C₂)
  (h₁₂ : B ∈ C₂)
  (h₁₃ : ∃ t : ℝ, (1 - t) • G + t • A = B)
  (k_AM k_BN : ℝ)
  (h₁₄ : k_AM = line_slope A M)
  (h₁₅ : k_BN = line_slope B N)
  : (C₂ = {p : ℝ × ℝ | (p.1^2/4) - p.2^2 = 1}) ∧
    (k_AM / k_BN = -1/3) ∧
    (∀ w, w = k_AM^2 + (2/3)*k_BN →
      (-3/4 < w ∧ w < -11/36) ∨ (13/36 < w ∧ w < 5/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_problem_l809_80930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l809_80997

theorem perpendicular_vectors (x : ℝ) : 
  let m : Fin 2 → ℝ := ![3, -2]
  let n : Fin 2 → ℝ := ![1, x]
  (m 0 * (m 0 + n 0) + m 1 * (m 1 + n 1) = 0) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l809_80997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l809_80976

theorem min_value_of_expression (y : ℝ) (hy : y > 0) : 9 * y^4 + 4 * y^(-5 : ℝ) ≥ 13 ∧ 
  (9 * y^4 + 4 * y^(-5 : ℝ) = 13 ↔ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l809_80976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_rate_problem_l809_80939

/-- Calculates the dividend rate given the face value, market value, and desired return rate of a share. -/
noncomputable def dividend_rate (face_value : ℝ) (market_value : ℝ) (desired_return_rate : ℝ) : ℝ :=
  (desired_return_rate * market_value / face_value) * 100

/-- Theorem stating that for a share with face value 52, market value 39, and desired return rate 12%,
    the dividend rate is 9%. -/
theorem dividend_rate_problem :
  let face_value : ℝ := 52
  let market_value : ℝ := 39
  let desired_return_rate : ℝ := 0.12
  dividend_rate face_value market_value desired_return_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dividend_rate_problem_l809_80939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l809_80960

noncomputable section

-- Define the side lengths of the triangles
def large_side : ℝ := 18
def small_side : ℝ := 6

-- Define the area of an equilateral triangle
noncomputable def equilateral_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

-- Define the areas of the large and small triangles
noncomputable def large_triangle_area : ℝ := equilateral_area large_side
noncomputable def small_triangle_area : ℝ := equilateral_area small_side

-- Define the area of the pentagonal region
noncomputable def pentagonal_area : ℝ := large_triangle_area - small_triangle_area

-- State the theorem
theorem triangle_to_pentagon_area_ratio :
  small_triangle_area / pentagonal_area = 1 / 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_pentagon_area_ratio_l809_80960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l809_80914

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else a n + 1

def b (n : ℕ) : ℕ := a (2 * n)

theorem b_formula (n : ℕ) (h : n ≥ 1) : b n = 3 * n - 1 := by
  sorry

theorem sum_first_20_a : (Finset.range 20).sum a = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_sum_first_20_a_l809_80914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_collection_time_l809_80996

/-- The time taken to collect all balls given the following conditions:
  * There are 45 balls to collect in total
  * Dad collects 4 balls every 40 seconds
  * Luke removes 3 balls every 40 seconds
  * The process continues until all 45 balls are collected for the first time
-/
theorem ball_collection_time 
  (total_balls : ℕ)
  (dad_collection_rate : ℕ)
  (luke_removal_rate : ℕ)
  (cycle_duration : ℕ)
  (h1 : total_balls = 45)
  (h2 : dad_collection_rate = 4)
  (h3 : luke_removal_rate = 3)
  (h4 : cycle_duration = 40) :
  (((total_balls - 2) / (dad_collection_rate - luke_removal_rate)) + 1) * cycle_duration = 1760 := by
  sorry

#check ball_collection_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_collection_time_l809_80996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radii_l809_80907

-- Define the volumes of the cones
noncomputable def volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Define the parameters
theorem cone_radii (h₁ h₂ r₁ r₂ : ℝ) 
  (h_positive : h₁ > 0 ∧ h₂ > 0)
  (h_relation : h₂ = 4 * h₁)
  (r₁_value : r₁ = 5)
  (vol_equal : volume r₁ h₂ = volume r₂ h₁) :
  r₂ = 10 := by
  sorry

#check cone_radii

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radii_l809_80907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_for_divisibility_l809_80998

theorem smallest_increase_for_divisibility : ∃! x : ℕ, 
  (∀ n : ℕ, n ∈ ({12, 18, 24, 32, 40} : Set ℕ) → (1441 + x) % n = 0) ∧
  (∀ y : ℕ, y < x → ∃ m : ℕ, m ∈ ({12, 18, 24, 32, 40} : Set ℕ) ∧ (1441 + y) % m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_increase_for_divisibility_l809_80998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l809_80931

/-- The time (in seconds) it takes for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / ((train_speed + man_speed) * 1000 / 3600)

/-- Theorem stating that the time for a 550-meter long train moving at 60 kmph to pass a man
    moving at 6 kmph in the opposite direction is approximately 30 seconds -/
theorem train_passing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_passing_time 550 60 6 - 30| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l809_80931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_single_point_l809_80917

-- Define the plane
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the fixed points A and B
variable (A B : V)

-- Define the radius b
variable (b : ℝ)

-- Define a circle
structure Circle (V : Type*) [NormedAddCommGroup V] where
  center : V
  radius : ℝ

-- Define the set of circles passing through A and B with radius b
def CirclesPassingThroughAB (A B : V) (b : ℝ) : Set (Circle V) :=
  {c : Circle V | ‖A - c.center‖ = c.radius ∧ ‖B - c.center‖ = c.radius ∧ c.radius = b}

-- State the theorem
theorem locus_is_single_point (A B : V) (b : ℝ) :
  ‖A - B‖ = 2 * b →
  ∃! center : V, ∀ c ∈ CirclesPassingThroughAB A B b, c.center = center :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_single_point_l809_80917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equal_intercepts_l809_80906

/-- A line passing through (1,2) with equal absolute x and y intercepts has a y-intercept of either 3 or 1 -/
theorem line_through_point_equal_intercepts :
  ∀ l : Set (ℝ × ℝ),
  ((1, 2) ∈ l) →
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x, y) ∈ l ↔ x/a + y/b = 1)) →
  (∃ c : ℝ, c > 0 ∧ (∀ x y : ℝ, (x, y) ∈ l ↔ x/c + y/c = 1)) →
  (∃ y_intercept : ℝ, y_intercept = 3 ∨ y_intercept = 1 ∧ (0, y_intercept) ∈ l) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equal_intercepts_l809_80906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_approximation_l809_80903

/-- Calculates the second discount percentage given the list price, final price, and first discount percentage. -/
noncomputable def calculate_second_discount (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) : ℝ :=
  let price_after_first_discount := list_price * (1 - first_discount / 100)
  let second_discount_decimal := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_decimal * 100

/-- The second discount is approximately 6.86% given the problem conditions. -/
theorem second_discount_approximation :
  let list_price : ℝ := 67
  let final_price : ℝ := 56.16
  let first_discount : ℝ := 10
  let second_discount := calculate_second_discount list_price final_price first_discount
  abs (second_discount - 6.86) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_approximation_l809_80903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_case1_line_equation_case2_l809_80921

-- Define a line type
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to calculate the slope from an angle in degrees
noncomputable def slopeFromAngle (angle : ℝ) : ℝ :=
  Real.tan (angle * Real.pi / 180)

-- Function to calculate the sum of intercepts
noncomputable def sumOfIntercepts (l : Line) : ℝ :=
  -l.intercept / l.slope + l.intercept

-- Theorem for case 1
theorem line_equation_case1 (l : Line) (A : Point) :
  A.x = -2 ∧ A.y = 3 ∧ slopeFromAngle 135 = l.slope ∧ pointOnLine A l →
  l.slope * -1 = 1 ∧ l.intercept = 1 := by
  sorry

-- Theorem for case 2
theorem line_equation_case2 (l1 l2 : Line) (A : Point) :
  A.x = -2 ∧ A.y = 3 ∧ pointOnLine A l1 ∧ pointOnLine A l2 ∧
  sumOfIntercepts l1 = 2 ∧ sumOfIntercepts l2 = 2 →
  (l1.slope = 1 ∧ l1.intercept = 1) ∨ (l2.slope = 3/2 ∧ l2.intercept = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_case1_line_equation_case2_l809_80921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_finishes_first_l809_80929

noncomputable section

-- Define the garden areas
def alice_garden_area : ℝ := 1
def bob_garden_area : ℝ := 1 / 3
def charlie_garden_area : ℝ := 1 / 2

-- Define the mowing speeds
def alice_mowing_speed : ℝ := 1
def bob_mowing_speed : ℝ := 1 / 2
def charlie_mowing_speed : ℝ := 1 / 4

-- Define the mowing times
def alice_mowing_time : ℝ := alice_garden_area / alice_mowing_speed
def bob_mowing_time : ℝ := bob_garden_area / bob_mowing_speed
def charlie_mowing_time : ℝ := charlie_garden_area / charlie_mowing_speed

theorem bob_finishes_first :
  bob_mowing_time < alice_mowing_time ∧ bob_mowing_time < charlie_mowing_time := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_finishes_first_l809_80929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l809_80919

open Complex

-- Define the locus of z given z₁, z₂, and λ
def locus (z₁ z₂ : ℂ) (lambda : ℝ) : Set ℂ :=
  {z : ℂ | abs (z - z₁) = lambda * abs (z - z₂)}

-- Define the perpendicular bisector of a line segment
def perpendicularBisector (z₁ z₂ : ℂ) : Set ℂ :=
  {z : ℂ | abs (z - z₁) = abs (z - z₂)}

-- Define the circle with given center and radius
def circleSet (center : ℂ) (radius : ℝ) : Set ℂ :=
  {z : ℂ | abs (z - center) = radius}

-- Theorem statement
theorem locus_characterization (z₁ z₂ : ℂ) (lambda : ℝ) 
  (h₁ : z₁ ≠ z₂) (h₂ : lambda > 0) :
  (lambda = 1 → locus z₁ z₂ lambda = perpendicularBisector z₁ z₂) ∧
  (lambda ≠ 1 → ∃ (center : ℂ) (radius : ℝ),
    center = (z₁ - lambda^2 * z₂) / (1 - lambda^2) ∧
    radius = Real.sqrt (abs (center - z₁)^2 - lambda^2 * abs (center - z₂)^2) ∧
    locus z₁ z₂ lambda = circleSet center radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_characterization_l809_80919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_of_f_eq_zero_l809_80944

-- Define the piece-wise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x ≤ 3 then 5 - x
  else x - 4

-- Theorem statement
theorem two_solutions_for_f_of_f_eq_zero :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f (f x) = 0 ∧
    ∀ y : ℝ, f (f y) = 0 → y ∈ s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_f_of_f_eq_zero_l809_80944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l809_80983

theorem equation_solution (x : ℝ) : 
  (16:ℝ)^x + (81:ℝ)^x = (2/3) * ((8:ℝ)^x + (27:ℝ)^x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l809_80983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_is_twentyeight_l809_80941

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + a 4 + a 6 = 12

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: The sum of the first 7 terms of the arithmetic sequence is 28 -/
theorem sum_seven_is_twentyeight (seq : ArithmeticSequence) : sum_n seq 7 = 28 := by
  -- Proof steps would go here
  sorry

#check sum_seven_is_twentyeight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_seven_is_twentyeight_l809_80941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_age_approx_l809_80902

-- Define the weights and ages
noncomputable def mother_weight : ℝ := sorry
noncomputable def daughter_weight : ℝ := sorry
noncomputable def grandchild_weight : ℝ := sorry
noncomputable def son_in_law_weight : ℝ := sorry
noncomputable def daughter_age : ℝ := sorry
def grandchild_age : ℝ := 6

-- Define the conditions
axiom total_weight : mother_weight + daughter_weight + grandchild_weight + son_in_law_weight = 230
axiom daughter_grandchild_weight : daughter_weight + grandchild_weight = 60
axiom son_in_law_weight_relation : son_in_law_weight = 2 * mother_weight
axiom grandchild_weight_relation : grandchild_weight = mother_weight / 5
axiom weight_age_proportion : ∃ k : ℝ, k > 0 ∧ daughter_weight = k * daughter_age ∧ grandchild_weight = k * grandchild_age

-- State the theorem
theorem daughter_age_approx :
  ∃ ε > 0, |daughter_age - 25.76| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_age_approx_l809_80902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_incorrect_l809_80980

-- Define the propositions
def proposition1 : Prop := ∀ f : ℝ → ℝ, ∀ x₀ : ℝ, (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x ≤ f x₀ ∨ f x ≥ f x₀) → (HasDerivAt f 0 x₀)

def proposition2 : Prop := ∀ a b : ℝ × ℝ, (∃ θ : ℝ, θ > Real.pi / 2 ∧ θ < Real.pi ∧ a.1 * b.1 + a.2 * b.2 = Real.cos θ * Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) → a.1 * b.1 + a.2 * b.2 < 0

def proposition3 : Prop := ∀ x : ℝ, (1 / (x - 1) > 0) ↔ ¬(1 / (x - 1) ≤ 0)

def proposition4 : Prop := (∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 > 0)

-- Theorem statement
theorem exactly_three_incorrect :
  (¬proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4) ∨
  (¬proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4) ∨
  (¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4) ∨
  (proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_incorrect_l809_80980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l809_80971

theorem no_solutions_in_interval (x : ℝ) : 
  x ∈ Set.Icc 0 (π / 2) → 
  Real.cos ((π / 3) * Real.sin x) ≠ Real.sin ((π / 3) * Real.cos x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l809_80971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l809_80978

/-- Given a three-digit number between 100 and 999, if its digits and the digits of its difference from 1000 are the same but in a different order, then the sum of its digits is 14. -/
theorem digit_sum_property (A : ℕ) : 
  100 ≤ A → A < 1000 → 
  (∃ a b c : ℕ, A = 100 * a + 10 * b + c ∧ 
               (1000 - A) ∈ ({100 * a + 10 * c + b, 100 * b + 10 * a + c, 100 * b + 10 * c + a, 
                             100 * c + 10 * a + b, 100 * c + 10 * b + a} : Set ℕ)) →
  ∃ a b c : ℕ, A = 100 * a + 10 * b + c ∧ a + b + c = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l809_80978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_productivity_day2_l809_80911

/-- Represents the productivity in lines of code per hour -/
def productivity : ℝ → ℝ := sorry

/-- Represents the number of coffee breaks -/
def coffee_breaks : ℝ → ℝ := sorry

/-- The constant of proportionality between productivity and coffee breaks -/
def k : ℝ := sorry

/-- Productivity is inversely proportional to the number of coffee breaks -/
axiom inverse_prop (d : ℝ) : productivity d * coffee_breaks d = k

/-- On day 1, there were 3 coffee breaks -/
axiom day1_breaks : coffee_breaks 1 = 3

/-- On day 1, productivity was 200 lines per hour -/
axiom day1_productivity : productivity 1 = 200

/-- On day 2, there were 5 coffee breaks -/
axiom day2_breaks : coffee_breaks 2 = 5

theorem productivity_day2 : productivity 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_productivity_day2_l809_80911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_on_circle_l809_80949

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - m*y - 25 = 0

-- Define the fixed points
noncomputable def fixed_point_1 : ℝ × ℝ := (Real.sqrt 5, 2 * Real.sqrt 5)
noncomputable def fixed_point_2 : ℝ × ℝ := (-Real.sqrt 5, -2 * Real.sqrt 5)

-- Theorem statement
theorem fixed_points_on_circle :
  ∀ m : ℝ, 
    circle_equation fixed_point_1.1 fixed_point_1.2 m ∧
    circle_equation fixed_point_2.1 fixed_point_2.2 m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_on_circle_l809_80949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l809_80963

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log (x + 2) / Real.log 2

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (M : ℝ), M = 3 ∧ 
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ M) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = M) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_on_interval_l809_80963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_5_l809_80990

def sequenceList : List Nat := List.range 20 |>.map (fun i => 4 + 10 * i)

theorem product_remainder_mod_5 : 
  sequenceList.prod % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_5_l809_80990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_closed_form_l809_80905

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The positive solution of X² = X + 1 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The negative solution of X² = X + 1 -/
noncomputable def φ' : ℝ := (1 - Real.sqrt 5) / 2

/-- Theorem: Closed form of Fibonacci sequence -/
theorem fibonacci_closed_form (n : ℕ) :
  (fib n : ℝ) = (φ ^ n - φ' ^ n) / Real.sqrt 5 := by
  sorry

#check fibonacci_closed_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_closed_form_l809_80905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l809_80993

theorem parallel_line_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let m := -a / b
  ∃ k : ℝ × ℝ, a * k.1 + b * k.2 = c ∧
  (∀ l : ℝ × ℝ, (a * l.1 + b * l.2 = c) → (l.2 - k.2) = m * (l.1 - k.1)) ∧
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_slope_l809_80993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nickel_count_in_wallet_l809_80942

/-- Represents the types of coins in the wallet --/
inductive Coin
  | penny
  | nickel
  | dime
  | quarter
deriving BEq

/-- The value of each coin type in cents --/
def coin_value : Coin → Nat
  | Coin.penny => 1
  | Coin.nickel => 5
  | Coin.dime => 10
  | Coin.quarter => 25

/-- The wallet containing coins --/
structure Wallet :=
  (coins : List Coin)

/-- Calculate the total value of coins in the wallet --/
def total_value (w : Wallet) : Nat :=
  w.coins.foldl (fun acc c => acc + coin_value c) 0

/-- Calculate the average value of coins in the wallet --/
def average_value (w : Wallet) : ℚ :=
  (total_value w : ℚ) / w.coins.length

/-- Add a dime to the wallet --/
def add_dime (w : Wallet) : Wallet :=
  ⟨w.coins ++ [Coin.dime]⟩

/-- Count the number of nickels in the wallet --/
def count_nickels (w : Wallet) : Nat :=
  w.coins.filter (· == Coin.nickel) |>.length

/-- The main theorem --/
theorem nickel_count_in_wallet (w : Wallet) :
  average_value w = 15 ∧ average_value (add_dime w) = 16 →
  count_nickels w = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nickel_count_in_wallet_l809_80942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_division_impossibility_l809_80922

/-- Represents a prism -/
structure Prism where
  volume : ℝ
  base_area : ℝ
  height : ℝ

/-- Represents a pyramid within the prism -/
structure Pyramid where
  base_area : ℝ
  height : ℝ
  base_on_prism_base : Bool
  apex_on_opposite_base : Bool

/-- Represents the central cross-section of the prism -/
structure CentralCrossSection where
  area : ℝ

/-- The theorem stating the impossibility of dividing a prism into non-overlapping pyramids
    with the specified properties -/
theorem prism_division_impossibility (p : Prism) (cs : CentralCrossSection) :
  ¬ ∃ (pyramids : List Pyramid),
    (∀ pyr, pyr ∈ pyramids → pyr.base_on_prism_base ∧ pyr.apex_on_opposite_base) ∧
    (∀ pyr1 pyr2, pyr1 ∈ pyramids → pyr2 ∈ pyramids → pyr1 ≠ pyr2 → pyr1.base_area ≠ pyr2.base_area) ∧
    ((pyramids.map (·.base_area)).sum = 2 * p.base_area) ∧
    ((pyramids.map (·.base_area / 4)).sum < cs.area) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_division_impossibility_l809_80922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_ball_volume_l809_80979

/-- The volume of an iron ball given the dimensions of iron bars and the number of bars and balls -/
theorem iron_ball_volume
  (length width height : ℝ)
  (num_bars num_balls : ℕ)
  (h1 : length = 12)
  (h2 : width = 8)
  (h3 : height = 6)
  (h4 : num_bars = 10)
  (h5 : num_balls = 720) :
  (length * width * height * (num_bars : ℝ)) / (num_balls : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_ball_volume_l809_80979
