import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weekly_gas_consumption_l1062_106290

/-- Calculates the weekly gas consumption given car efficiency, work travel, and leisure travel. -/
noncomputable def weekly_gas_consumption (car_efficiency : ℝ) (work_distance : ℝ) (work_days : ℕ) (leisure_distance : ℝ) : ℝ :=
  (work_distance * (work_days : ℝ) + leisure_distance) / car_efficiency

/-- Theorem: John's weekly gas consumption is 8 gallons -/
theorem john_weekly_gas_consumption :
  weekly_gas_consumption 30 40 5 40 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_weekly_gas_consumption_l1062_106290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1062_106229

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt x

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 1/2 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1062_106229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleOneGreen_l1062_106260

/-- Represents the color of a bead -/
inductive Color
| Black
| Blue
| Green
deriving Repr, DecidableEq

/-- Represents the state of the bead circle -/
def BeadCircle := Vector Color 2016

/-- The replacement rule for a single bead -/
def replaceRule (left right : Color) : Color :=
  if left = right then left
  else match left, right with
    | Color.Black, Color.Blue | Color.Blue, Color.Black => Color.Green
    | Color.Black, Color.Green | Color.Green, Color.Black => Color.Blue
    | Color.Blue, Color.Green | Color.Green, Color.Blue => Color.Black
    | _, _ => left  -- This case should never occur in our problem

/-- Applies the replacement rule to the entire circle -/
def applyReplacement (circle : BeadCircle) : BeadCircle :=
  Vector.ofFn fun i => replaceRule (circle.get ((i - 1 + 2016) % 2016)) (circle.get ((i + 1) % 2016))

/-- Checks if the circle has exactly one green bead and all others blue -/
def hasOneGreenRestBlue (circle : BeadCircle) : Prop :=
  (circle.toList.filter (· == Color.Green)).length = 1 ∧
  (circle.toList.filter (· == Color.Blue)).length = 2015

/-- Initial state with two adjacent black beads and all others blue -/
def initialState : BeadCircle :=
  Vector.ofFn fun i => if i = 0 ∨ i = 1 then Color.Black else Color.Blue

/-- Theorem: It's impossible to reach a state with one green bead and all others blue -/
theorem impossibleOneGreen :
  ¬∃ (n : ℕ), hasOneGreenRestBlue (n.iterate applyReplacement initialState) := by
  sorry

#eval initialState
#eval applyReplacement initialState

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleOneGreen_l1062_106260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guilty_defendant_l1062_106221

-- Define the set of defendants
inductive Defendant : Type where
| A : Defendant
| B : Defendant
| C : Defendant

-- Define a function to represent accusations
def accuses : Defendant → Defendant → Prop := sorry

-- Define a function to represent whether a defendant is telling the truth
def isTellingTruth : Defendant → Prop := sorry

-- Define a function to represent whether a defendant is guilty
def isGuilty : Defendant → Prop := sorry

-- Theorem statement
theorem guilty_defendant 
  (h1 : ∀ d : Defendant, ∃ d' : Defendant, d ≠ d' ∧ accuses d d')
  (h2 : isTellingTruth Defendant.A)
  (h3 : ∀ d : Defendant, isTellingTruth d → ∃ d' : Defendant, accuses d d' ∧ isGuilty d')
  (h4 : ∀ d : Defendant, ¬isTellingTruth d → ∃ d' : Defendant, accuses d d' ∧ ¬isGuilty d')
  (h5 : ∀ d : Defendant, isGuilty d → ∀ d' : Defendant, d ≠ d' → ¬isGuilty d')
  : isGuilty Defendant.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guilty_defendant_l1062_106221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_2006th_term_l1062_106287

/-- A sequence where each term is a multiple of 3 and when added to 1 results in a perfect square -/
def special_sequence : ℕ → ℕ
  | n => if n % 2 = 0 
         then (3 * (n / 2) + 1)^2 - 1 
         else (3 * (n / 2) + 2)^2 - 1

/-- The 2006th term of the special sequence is 3009² - 1 -/
theorem special_sequence_2006th_term : 
  special_sequence 2006 = 3009^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_2006th_term_l1062_106287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tied_moments_hockey_l1062_106247

def n : ℕ := 5

def probability_tied (k : ℕ) : ℚ :=
  (Nat.choose (2 * k) k : ℚ) / (2 ^ (2 * k))

def expected_tied_moments : ℚ :=
  Finset.sum (Finset.range n) (λ k => probability_tied (k + 1))

theorem expected_tied_moments_hockey :
  ∃ ε > 0, |expected_tied_moments - 1707/1000| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_tied_moments_hockey_l1062_106247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1062_106250

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc 0 1

-- Define the domain of f(2^x - 2)
def domain_f_2_pow_x_minus_2 : Set ℝ := Set.Icc (Real.log 3 / Real.log 2) 2

-- Theorem statement
theorem domain_equivalence :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) = f (x + 1)) →
  (∀ x ∈ domain_f_2_pow_x_minus_2, f (2^x - 2) = f (2^x - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l1062_106250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_N2O5_approx_l1062_106296

/-- The mass percentage of nitrogen in dinitrogen pentoxide (N2O5) -/
noncomputable def mass_percentage_N_in_N2O5 : ℝ :=
  let atomic_mass_N : ℝ := 14.01
  let atomic_mass_O : ℝ := 16.00
  let molar_mass_N2O5 : ℝ := 2 * atomic_mass_N + 5 * atomic_mass_O
  let mass_N_in_N2O5 : ℝ := 2 * atomic_mass_N
  (mass_N_in_N2O5 / molar_mass_N2O5) * 100

/-- Theorem stating that the mass percentage of nitrogen in N2O5 is approximately 25.94% -/
theorem mass_percentage_N_in_N2O5_approx :
  ∃ ε > 0, |mass_percentage_N_in_N2O5 - 25.94| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_N_in_N2O5_approx_l1062_106296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_function_l1062_106205

theorem symmetric_sine_function (φ : ℝ) : 
  φ ∈ Set.Ioo (-π/2) (π/2) →
  (∀ x : ℝ, Real.sin (3*x + φ) = Real.sin (3*(2*(3*π/5) - x) + φ)) →
  φ = -3*π/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_function_l1062_106205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orchid_price_is_50_l1062_106285

/-- The price of each orchid sold by a plant supplier --/
noncomputable def orchid_price : ℚ :=
  let num_orchids : ℕ := 20
  let num_money_plants : ℕ := 15
  let money_plant_price : ℚ := 25
  let worker_wage : ℚ := 40
  let num_workers : ℕ := 2
  let new_pots_cost : ℚ := 150
  let remaining_money : ℚ := 1145
  (remaining_money + new_pots_cost + (num_workers * worker_wage) - (num_money_plants * money_plant_price)) / num_orchids

theorem orchid_price_is_50 : orchid_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orchid_price_is_50_l1062_106285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1062_106216

theorem find_x (a x : ℝ) (h1 : (4 : ℝ)^a = 2) (h2 : Real.log x / Real.log a = 2*a) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l1062_106216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1062_106256

theorem tan_difference (α β : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : Real.tan (π - β) = 1/2) : 
  Real.tan (α - β) = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l1062_106256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l1062_106215

theorem sin_angle_sum (a : Real) (h1 : Real.sin (π / 3 + a) = 5 / 13) (h2 : a ∈ Set.Ioo (π / 6) (2 * π / 3)) :
  Real.sin (π / 12 + a) = 17 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_angle_sum_l1062_106215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sequence_x_value_l1062_106252

/-- A sequence of five terms where the second term is unknown -/
def Sequence : Type := ℤ × ℤ × ℤ × ℤ × ℤ

/-- The differences between consecutive terms in the sequence -/
def differences (s : Sequence) : ℤ × ℤ × ℤ × ℤ :=
  match s with
  | (a, x, c, d, e) => (x - a, c - x, d - c, e - d)

/-- The differences between consecutive differences -/
def secondDifferences (d : ℤ × ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ :=
  match d with
  | (a, b, c, d) => (b - a, c - b, d - c)

/-- Theorem stating that if the sequence follows a quadratic pattern, then x = 64 -/
theorem quadratic_sequence_x_value (s : Sequence) :
  s.1 = 8 ∧ s.2.1 = 62 ∧ s.2.2.1 = -4 ∧ s.2.2.2 = -12 →
  let d := differences s
  let sd := secondDifferences d
  sd.2.1 = sd.2.2 →
  s.2.1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sequence_x_value_l1062_106252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_practice_hours_is_22_l1062_106266

/-- Represents the practice hours for each day of the week -/
def regular_hours : Fin 7 → ℕ
  | 0 => 4  -- Monday
  | 1 => 5  -- Tuesday
  | 2 => 6  -- Wednesday
  | 3 => 5  -- Thursday
  | 4 => 3  -- Friday
  | 5 => 4  -- Saturday
  | 6 => 0  -- Sunday (Rest day)

/-- Represents the adjustment to practice hours due to weather events -/
def weather_adjustment : Fin 7 → ℤ
  | 1 => -1  -- Tuesday: shortened by 1 hour
  | 2 => -6  -- Wednesday: canceled entirely
  | 4 => 2   -- Friday: extended by 2 hours
  | _ => 0   -- No changes for other days

/-- Calculates the adjusted practice hours for a given day -/
def adjusted_hours (day : Fin 7) : ℕ :=
  (regular_hours day + weather_adjustment day).natAbs

/-- Theorem: The total adjusted practice hours for the week is 22 -/
theorem total_practice_hours_is_22 :
  (Finset.sum Finset.univ adjusted_hours) = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_practice_hours_is_22_l1062_106266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l1062_106281

theorem manufacturing_percentage (circle_degrees : ℝ) (manufacturing_degrees : ℝ) 
  (h1 : circle_degrees = 360) 
  (h2 : manufacturing_degrees = 180) : 
  manufacturing_degrees / circle_degrees = 1/2 := by
  rw [h1, h2]
  norm_num

#check manufacturing_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_percentage_l1062_106281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_remaining_money_l1062_106203

def weekly_earnings : List ℚ := [1200, 1300, 1100, 1400]
def recharge_rates : List ℚ := [6/10, 6/10, 4/10, 3/10]
def weekly_expenses : List ℚ := [200, 150, 250, 300]
def additional_costs : ℚ := 500 + 1500

def total_earnings : ℚ := weekly_earnings.sum
def total_recharged : ℚ := (List.zip weekly_earnings recharge_rates).map (λ (x, y) => x * y) |>.sum
def total_expenses : ℚ := weekly_expenses.sum + additional_costs

theorem brendans_remaining_money :
  total_earnings - (total_recharged + total_expenses) = -260 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendans_remaining_money_l1062_106203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_condition_inequality_condition_l1062_106230

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / x
def g (x : ℝ) : ℝ := x + Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x + g x

-- Statement for part (1)
theorem extremum_point_condition (a : ℝ) (h_a : a > 0) :
  (∀ x, x > 0 → (deriv (h a)) x = 0 → x = 1) → a = Real.sqrt 3 := by sorry

-- Statement for part (2)
theorem inequality_condition (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂, 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → f a x₁ ≥ g x₂) →
  a ≥ (Real.exp 1 + 1) / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_point_condition_inequality_condition_l1062_106230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_emails_for_18_children_min_emails_general_min_emails_is_optimal_l1062_106262

/-- Represents the minimum number of emails required for all children to know all solutions -/
def min_emails (n : ℕ) : ℕ := 2 * (n - 1)

/-- Theorem stating that for 18 children, the minimum number of emails is 34 -/
theorem min_emails_for_18_children : 
  min_emails 18 = 34 := by
  rfl

/-- Proves that the minimum number of emails for n children is always 2(n-1) -/
theorem min_emails_general (n : ℕ) (h : n ≥ 2) : 
  min_emails n = 2 * (n - 1) := by
  rfl

/-- Definition: A child knows another child's solution -/
def knows_solution {n : ℕ} (i j : Fin n) : Prop := sorry

/-- Proves that the solution is optimal -/
theorem min_emails_is_optimal {n : ℕ} (h : n ≥ 2) :
  ∀ m : ℕ, m < min_emails n → 
  ∃ i j : Fin n, i ≠ j ∧ (∀ k : Fin n, ¬(knows_solution i k ∧ knows_solution j k)) := by
  sorry

/-- Axiom: A child knows their own solution -/
axiom knows_own_solution {n : ℕ} (i : Fin n) : knows_solution i i

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_emails_for_18_children_min_emails_general_min_emails_is_optimal_l1062_106262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_value_l1062_106276

theorem function_minimum_value (b : ℝ) :
  let f := λ x : ℝ => (1/3) * x^3 - (1 + b/2) * x^2 + 2 * b * x
  (∃ x y, x ∈ Set.Icc (-3) 1 ∧ y ∈ Set.Icc (-3) 1 ∧ x < y ∧ f x > f y) →
  (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 2 * b - 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_value_l1062_106276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_along_squares_l1062_106278

theorem path_length_along_squares (PQ : ℝ) (n : ℕ) (h : PQ = 73) :
  let segments := List.replicate n (PQ / n)
  let path_length := List.sum (List.map (λ s => 3 * s) segments)
  path_length = 219 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_along_squares_l1062_106278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_properties_l1062_106282

structure CoinToss where
  sampleSpace : Finset (Bool × Bool)
  probA : ℚ
  probB : ℚ
  probAB : ℚ

noncomputable def uniformCoinToss : CoinToss where
  sampleSpace := {(true, true), (true, false), (false, true), (false, false)}
  probA := 1/2
  probB := 1/2
  probAB := 1/4

theorem coin_toss_properties (ct : CoinToss) (h : ct = uniformCoinToss) :
  Finset.card ct.sampleSpace = 4 ∧ 
  ct.probAB = 1/4 ∧
  ct.probAB = ct.probA * ct.probB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_properties_l1062_106282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1062_106280

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧
  t.b = Real.sqrt 3 ∧
  ∃ (d : Real), t.A + d = t.B ∧ t.B + d = t.C ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi

-- Theorem statement
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2 : Real) * t.a * t.b = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1062_106280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1062_106210

theorem inequality_proof (x y z l m n : ℝ) 
  (h1 : l ≥ 0) (h2 : m ≥ 0) (h3 : n ≥ 0) (h4 : l + m + n > 0) : 
  x^2 / (l*x + m*y + n*z) + y^2 / (l*y + m*z + n*x) + z^2 / (l*z + m*x + n*y) 
  ≥ (x + y + z) / (l + m + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1062_106210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_sheet_drawings_l1062_106261

def total_drawings : ℕ := 12 * 36 * 8

def sheets_per_notebook : ℕ := 36

def drawings_per_sheet_new : ℕ := 5

def complete_notebooks_after_reorganization : ℕ := 6

def full_sheets_in_last_notebook : ℕ := 29

theorem last_sheet_drawings : 
  total_drawings - 
  (complete_notebooks_after_reorganization * sheets_per_notebook + full_sheets_in_last_notebook) * 
  drawings_per_sheet_new = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_sheet_drawings_l1062_106261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ps_length_is_sqrt_66_33_l1062_106234

/-- A quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (P Q R S T : ℝ × ℝ)
  (pt : ℝ)
  (tr : ℝ)
  (qt : ℝ)
  (ts : ℝ)
  (pq : ℝ)
  (intersect : (P.1 - R.1) * (Q.2 - S.2) = (Q.1 - S.1) * (P.2 - R.2))

/-- The length of PS in the quadrilateral -/
noncomputable def ps_length (quad : Quadrilateral) : ℝ :=
  Real.sqrt ((quad.P.1 - quad.S.1)^2 + (quad.P.2 - quad.S.2)^2)

theorem ps_length_is_sqrt_66_33 (quad : Quadrilateral) 
  (h1 : quad.pt = 5)
  (h2 : quad.tr = 7)
  (h3 : quad.qt = 9)
  (h4 : quad.ts = 4)
  (h5 : quad.pq = 7) :
  ps_length quad = Real.sqrt 66.33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ps_length_is_sqrt_66_33_l1062_106234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_time_calculation_l1062_106299

/-- Represents the shadow length in feet -/
noncomputable def shadow_length (hours : ℝ) : ℝ := 5 * hours

/-- Converts inches to feet -/
noncomputable def inches_to_feet (inches : ℝ) : ℝ := inches / 12

theorem shadow_time_calculation (current_length_inches : ℝ) 
  (h1 : current_length_inches = 360) : 
  ∃ (hours : ℝ), hours = 6 ∧ shadow_length hours = inches_to_feet current_length_inches :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_time_calculation_l1062_106299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1062_106284

open Real
open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def not_collinear (a b : V) : Prop := ¬ ∃ (k : ℝ), b = k • a

def collinear_opposite (c d : V) : Prop := ∃ (m : ℝ), m < 0 ∧ c = m • d

theorem vector_problem (a b : V) (l : ℝ) 
  (h1 : not_collinear a b)
  (h2 : collinear_opposite (l • a + b) (a + (2 * l - 1) • b)) :
  l = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1062_106284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1062_106291

def jo_sum : ℤ := (List.range 50).sum

def round_to_nearest_ten (n : ℕ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

def kate_sum : ℤ := (List.range 50).map (round_to_nearest_ten ∘ (·+1)) |>.sum

theorem sum_difference : |jo_sum - kate_sum| = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_l1062_106291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_sum_l1062_106271

def die1 : List Nat := [1, 2, 3, 3, 4, 4]
def die2 : List Nat := [2, 3, 4, 7, 7, 10]

def is_target_sum (pair : Nat × Nat) : Bool :=
  let sum := pair.1 + pair.2
  sum = 7 || sum = 9 || sum = 11

def count_target_outcomes : Nat :=
  (List.product die1 die2).filter is_target_sum |>.length

theorem probability_of_target_sum :
  (count_target_outcomes : Rat) / ((die1.length * die2.length) : Rat) = 11 / 36 := by
  sorry

#eval count_target_outcomes
#eval die1.length * die2.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_target_sum_l1062_106271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l1062_106264

theorem sin_translation_to_cos (φ : ℝ) : 
  (0 ≤ φ) ∧ (φ < 2 * Real.pi) →
  (∀ x, Real.sin (x + φ) = Real.cos (x - Real.pi / 6)) →
  φ = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l1062_106264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l1062_106226

noncomputable def v : Fin 3 → ℝ := ![
  (3 - Real.sqrt 6) / 4,
  0,
  -Real.sqrt 6 / 2
]

noncomputable def a : Fin 3 → ℝ := ![2, 2, -1]
noncomputable def b : Fin 3 → ℝ := ![0, 1, -1]

theorem vector_satisfies_conditions :
  (v 0 = (3 - Real.sqrt 6) / 4 ∧ v 1 = 0 ∧ v 2 = -Real.sqrt 6 / 2) ∧  -- v lies in xz-plane
  Finset.sum (Finset.range 3) (fun i => (v i) ^ 2) = 1 ∧  -- v is a unit vector
  (Finset.sum (Finset.range 3) (fun i => (v i) * (a i))) / 
    (Real.sqrt (Finset.sum (Finset.range 3) (fun i => (v i) ^ 2)) * 
     Real.sqrt (Finset.sum (Finset.range 3) (fun i => (a i) ^ 2))) = 1 / 2 ∧  -- angle with a is 60°
  (Finset.sum (Finset.range 3) (fun i => (v i) * (b i))) / 
    (Real.sqrt (Finset.sum (Finset.range 3) (fun i => (v i) ^ 2)) * 
     Real.sqrt (Finset.sum (Finset.range 3) (fun i => (b i) ^ 2))) = Real.sqrt 3 / 2  -- angle with b is 30°
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_conditions_l1062_106226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_ratio_proof_l1062_106295

/-- The number of gnomes in Ravenswood forest -/
def ravenswood_gnomes : ℚ := 80

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℚ := 20

/-- The ratio of gnomes in Ravenswood forest to Westerville woods -/
def gnome_ratio : ℚ := 4

theorem gnome_ratio_proof :
  (ravenswood_gnomes = westerville_gnomes) →
  (westerville_gnomes = 20) →
  (ravenswood_gnomes * (1 - 0.4) = 48) →
  ravenswood_gnomes / westerville_gnomes = gnome_ratio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_ratio_proof_l1062_106295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1062_106206

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: Given an ellipse with specific properties, its eccentricity is √2/2 -/
theorem ellipse_eccentricity (e : Ellipse) (p f₁ f₂ : Point) :
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1 →
  distance origin p = (1/2) * distance f₁ f₂ →
  distance p f₁ * distance p f₂ = e.a^2 →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1062_106206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_gain_loss_percentage_setA_zero_gain_loss_setB_zero_gain_loss_setC_zero_gain_loss_l1062_106200

structure BookSet where
  costBooks : ℕ
  sellBooks : ℕ
  totalCost : ℚ
  totalSell : ℚ
  hEqualTotals : totalCost = totalSell

noncomputable def gainLossPercentage (bs : BookSet) : ℚ :=
  (bs.totalSell - bs.totalCost) / bs.totalCost * 100

theorem zero_gain_loss_percentage (bs : BookSet) : gainLossPercentage bs = 0 := by
  sorry

-- Define the three sets
def setA : BookSet := ⟨50, 60, 1000, 1000, rfl⟩
def setB : BookSet := ⟨30, 40, 800, 800, rfl⟩
def setC : BookSet := ⟨20, 25, 500, 500, rfl⟩

-- Prove that each set has 0% gain/loss
theorem setA_zero_gain_loss : gainLossPercentage setA = 0 := by
  sorry

theorem setB_zero_gain_loss : gainLossPercentage setB = 0 := by
  sorry

theorem setC_zero_gain_loss : gainLossPercentage setC = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_gain_loss_percentage_setA_zero_gain_loss_setB_zero_gain_loss_setC_zero_gain_loss_l1062_106200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mindmaster_secret_codes_l1062_106207

/-- The number of colors available for pegs -/
def num_colors : ℕ := 10

/-- The total number of slots -/
def total_slots : ℕ := 5

/-- The maximum number of slots that can be filled -/
def max_filled_slots : ℕ := total_slots - 1

/-- The number of possible secret codes in the Mindmaster variant -/
def num_secret_codes : ℕ := 
  Finset.sum (Finset.range max_filled_slots) (λ k => (Nat.choose total_slots (k + 1)) * (num_colors ^ (k + 1)))

theorem mindmaster_secret_codes :
  num_secret_codes = 61050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mindmaster_secret_codes_l1062_106207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l1062_106209

/-- The volume of a cone with radius r and height h --/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The height of a cone given its volume and base radius --/
noncomputable def cone_height (v r : ℝ) : ℝ := (3 * v) / (Real.pi * r^2)

theorem cone_height_calculation (v : ℝ) (h : ℝ) :
  v = 16384 * Real.pi →
  cone_volume h h = v →
  ‖h - 36.8‖ < 0.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l1062_106209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_of_angles_l1062_106222

theorem tan_difference_of_angles (α β : Real) : 
  0 < α ∧ α < Real.pi/2 →
  0 < β ∧ β < Real.pi/2 →
  Real.sin α - Real.sin β = -1/2 →
  Real.cos α - Real.cos β = 1/2 →
  Real.tan (α - β) = -Real.sqrt 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_of_angles_l1062_106222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1062_106273

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem range_of_sum (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 2 < a + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l1062_106273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_cos_value_when_f_is_negative_one_l1062_106233

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

-- Theorem for the value of cos(2π/3 - 2x) when f(x) = -1
theorem cos_value_when_f_is_negative_one :
  ∀ (x : ℝ), f x = -1 → Real.cos (2 * Real.pi / 3 - 2 * x) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_cos_value_when_f_is_negative_one_l1062_106233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_propositions_true_l1062_106217

-- Define the real number a
variable (a : ℝ)

-- Define the condition 1 < a < 2
def a_condition (a : ℝ) : Prop := 1 < a ∧ a < 2

-- Define the logarithm function
noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- Define the proposition p
def proposition_p (a : ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → log_function a x > log_function a y

-- Define the proposition q
def proposition_q (a : ℝ) : Prop := (∀ x : ℝ, |x| < 1 → x < a) ∧ ¬(∀ x : ℝ, x < a → |x| < 1)

-- State the theorem
theorem both_propositions_true {a : ℝ} (h : a_condition a) : proposition_p a ∧ proposition_q a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_propositions_true_l1062_106217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_net_profit_l1062_106259

/-- Represents the tiered pricing structure for patches --/
structure TierPricing where
  tier1 : ℕ → ℚ  -- Price for 1-10 patches
  tier2 : ℕ → ℚ  -- Price for 11-30 patches
  tier3 : ℕ → ℚ  -- Price for 31-50 patches
  tier4 : ℕ → ℚ  -- Price for 51-100 patches

/-- Represents the customer breakdown for the month --/
structure CustomerBreakdown where
  tier1_customers : ℕ
  tier1_patches : ℕ
  tier2_customers : List ℕ
  tier3_customers : ℕ
  tier3_patches : ℕ
  tier4_customers : ℕ
  tier4_patches : ℕ

/-- Calculates Sean's net profit for the month --/
def calculate_net_profit (
  patch_cost : ℚ
) (shipping_fee : ℚ
) (unit_size : ℕ
) (pricing : TierPricing
) (customers : CustomerBreakdown
) : ℚ :=
  sorry  -- Proof omitted

/-- Theorem stating Sean's net profit for the given scenario --/
theorem sean_net_profit :
  let patch_cost : ℚ := 5/4
  let shipping_fee : ℚ := 20
  let unit_size : ℕ := 100
  let pricing : TierPricing := {
    tier1 := fun _ => 12,
    tier2 := fun _ => 23/2,
    tier3 := fun _ => 11,
    tier4 := fun _ => 21/2
  }
  let customers : CustomerBreakdown := {
    tier1_customers := 4,
    tier1_patches := 5,
    tier2_customers := [20, 15, 12],
    tier3_customers := 2,
    tier3_patches := 35,
    tier4_customers := 1,
    tier4_patches := 100
  }
  calculate_net_profit patch_cost shipping_fee unit_size pricing customers = 4331/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_net_profit_l1062_106259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_ratio_l1062_106274

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and eccentricity 5/3,
    a line passing through one focus and tangent to the circle x² + y² = a²,
    prove that the ratio of the distance from the focus to a point on the hyperbola
    to the distance from the focus to the tangent point is 4. -/
theorem hyperbola_focus_distance_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e : ℝ := 5/3  -- eccentricity
  let C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1^2/a^2) - (p.2^2/b^2) = 1}  -- hyperbola
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}  -- circle
  let F₁ : ℝ × ℝ := (-a*e, 0)  -- left focus
  ∃ (l : Set (ℝ × ℝ)) (T P : ℝ × ℝ),  -- line l, tangent point T, intersection point P
    T ∈ circle ∧ P ∈ C ∧ F₁ ∈ l ∧ T ∈ l ∧ P ∈ l ∧
    (∀ x ∈ circle, x ≠ T → x ∉ l) →  -- l is tangent to circle at T
    ‖P - F₁‖ / ‖T - F₁‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_ratio_l1062_106274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_convergence_l1062_106239

/-- Represents a cell in the grid -/
structure Cell where
  row : ℕ
  col : ℕ

/-- Represents the grid -/
def Grid (n : ℕ) := Cell → Int

/-- The operation that updates the grid -/
def update_grid (n : ℕ) (g : Grid n) : Grid n :=
  λ c ↦ (g (Cell.mk ((c.row - 1 + n) % n) c.col)) *
       (g (Cell.mk ((c.row + 1) % n) c.col)) *
       (g (Cell.mk c.row ((c.col - 1 + n) % n))) *
       (g (Cell.mk c.row ((c.col + 1) % n)))

/-- Checks if all cells in the grid are +1 -/
def all_positive_one (n : ℕ) (g : Grid n) : Prop :=
  ∀ c : Cell, c.row < n ∧ c.col < n → g c = 1

/-- The main theorem -/
theorem grid_convergence (n : ℕ) (h : n ≥ 2) :
  (∃ k : ℕ, n = 2^k) ↔
  (∀ g : Grid n, ∃ m : ℕ, all_positive_one n ((update_grid n)^[m] g)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_convergence_l1062_106239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_101_l1062_106251

/-- Represents a four-digit integer -/
structure FourDigitInt where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- A sequence of four-digit integers satisfying the given conditions -/
def ValidSequence (seq : List FourDigitInt) : Prop :=
  ∀ i, i + 1 < seq.length →
    (seq.get ⟨i, by sorry⟩).value / 100 % 100 = (seq.get ⟨i+1, by sorry⟩).value / 1000 % 10 ∧
    (seq.get ⟨i, by sorry⟩).value / 10 % 10 = (seq.get ⟨i+1, by sorry⟩).value / 100 % 10

/-- The sum of all terms in the sequence -/
def SequenceSum (seq : List FourDigitInt) : ℕ :=
  seq.foldl (λ sum term => sum + term.value) 0

theorem sum_divisible_by_101 (seq : List FourDigitInt) (h : ValidSequence seq) :
  101 ∣ SequenceSum seq := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_101_l1062_106251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fraction_less_than_two_unique_zero_point_condition_l1062_106240

-- Problem 1
theorem min_fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by sorry

-- Problem 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / x

theorem unique_zero_point_condition (a : ℝ) (ha : a > 0) :
  (∃! x, x > 0 ∧ f a x = 0) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fraction_less_than_two_unique_zero_point_condition_l1062_106240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_perpendicular_l1062_106254

/-- The minimum distance between a point on an ellipse and a perpendicular line -/
theorem min_distance_ellipse_perpendicular (x y : ℝ) :
  (x^2 / 25 + y^2 / 16 = 1) →  -- P(x,y) is on the ellipse
  ∃ (m : ℝ × ℝ),
    (‖m - (3, 0)‖ = 1) ∧  -- |AM| = 1
    ((x - m.1) * (3 - m.1) + (y - m.2) * (0 - m.2) = 0) →  -- PM · AM = 0
    (∀ (x' y' : ℝ), 
      (x'^2 / 25 + y'^2 / 16 = 1) →
      ‖(x', y') - m‖^2 ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_perpendicular_l1062_106254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l1062_106211

/-- A palindrome is a positive integer that reads the same backward as forward. -/
def IsPalindrome (k : ℕ) : Prop := sorry

/-- The number of digits in a natural number -/
def NumDigits (k : ℕ) : ℕ := sorry

/-- The first digit of a natural number -/
def FirstDigit (k : ℕ) : ℕ := sorry

theorem count_palindromes (n : ℕ) :
  (∃ (s : Finset ℕ), (∀ k ∈ s, IsPalindrome k ∧ NumDigits k = 2*n+1 ∧ FirstDigit k ≠ 0) ∧
                     s.card = 9 * (10^n : ℕ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_palindromes_l1062_106211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1062_106218

/-- The focus of a parabola with equation x = ay^2 (a ≠ 0) -/
noncomputable def parabola_focus (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

/-- Theorem: The coordinates of the focus of the parabola x = ay^2 (a ≠ 0) are (1/(4a), 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  parabola_focus a h = (1 / (4 * a), 0) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1062_106218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l1062_106255

-- Define A' as a series
noncomputable def A' : ℝ := ∑' n, if (n % 4 ≠ 0 ∧ n % 2 = 0) then ((-1)^((n / 2) + 1) / n^2) else 0

-- Define B' as a series
noncomputable def B' : ℝ := ∑' n, if (n % 8 = 0) then ((-1)^((n / 4) - 1) / n^2) else 0

-- Theorem stating that A' / B' = 17
theorem A_div_B_eq_17 : A' / B' = 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_div_B_eq_17_l1062_106255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l1062_106288

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x^2 - Real.cos x

-- State the theorem
theorem solution_set_f (x : ℝ) :
  f (x - 1) > f (-1) ↔ x ∈ Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l1062_106288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1062_106212

-- Define a type for colors
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- State the theorem
theorem same_color_unit_distance (f : Coloring) :
  ∃ (p q : Point), f p = f q ∧ distance p q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_unit_distance_l1062_106212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_power_relations_l1062_106269

-- Part 1
theorem expression_evaluation :
  (2/3 : ℝ)^0 + 2^(-2 : ℤ) * (16/9 : ℝ)^(1/2 : ℝ) + (Real.log 8 / Real.log 10 + Real.log 125 / Real.log 10) = 13/3 := by sorry

-- Part 2
theorem power_relations (a : ℝ) (h : a + a^(-1 : ℝ) = 5) :
  (a^2 + a^(-2 : ℝ) = 23) ∧ (a^(1/2 : ℝ) + a^(-1/2 : ℝ) = Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_power_relations_l1062_106269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1062_106235

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^3 - 3*x^2 + 2*x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 0 ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ∨ 2 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1062_106235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_parabola_equation_l1062_106277

/-- A parabola in a sequence of parabolas generated by centroids -/
structure Parabola where
  n : ℕ
  m : ℝ
  equation : ℝ → ℝ → Prop

/-- The initial parabola P with equation y^2 = mx -/
def initial_parabola (m : ℝ) : Parabola where
  n := 0
  m := m
  equation := λ x y ↦ y^2 = m * x

/-- Generate the next parabola in the sequence -/
def next_parabola (P : Parabola) : Parabola where
  n := P.n + 1
  m := P.m
  equation := λ x y ↦ y^2 = (1 / 3^(P.n + 1)) * P.m * (x - (P.m / 4) * (1 - 1 / 3^(P.n + 1)))

/-- The nth parabola in the sequence -/
def nth_parabola (m : ℝ) : ℕ → Parabola
  | 0 => initial_parabola m
  | n + 1 => next_parabola (nth_parabola m n)

/-- The main theorem: the equation of the nth parabola -/
theorem nth_parabola_equation (m : ℝ) (n : ℕ) (x y : ℝ) :
  (nth_parabola m n).equation x y ↔
    y^2 = (1 / 3^n) * m * (x - (m / 4) * (1 - 1 / 3^n)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_parabola_equation_l1062_106277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_is_12_hours_l1062_106219

/-- The time it takes for Pipe A to fill the tank with a leak present -/
noncomputable def fillTimeWithLeak (fillTime : ℝ) (emptyTime : ℝ) : ℝ :=
  1 / (1 / fillTime - 1 / emptyTime)

/-- Theorem stating that the time to fill the tank with a leak is 12 hours -/
theorem fill_time_with_leak_is_12_hours :
  fillTimeWithLeak 8 24 = 12 := by
  -- Unfold the definition of fillTimeWithLeak
  unfold fillTimeWithLeak
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_with_leak_is_12_hours_l1062_106219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinable_with_sqrt_3_l1062_106214

theorem combinable_with_sqrt_3 :
  ∀ (x : ℝ), x ∈ ({Real.sqrt (3/2), Real.sqrt 8, Real.sqrt 0.5, Real.sqrt 12} : Set ℝ) →
  (∃ (q : ℚ), x = q * Real.sqrt 3) ↔ x = Real.sqrt 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinable_with_sqrt_3_l1062_106214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_side_length_l1062_106292

/-- Represents a square sheet of paper. -/
structure Paper where
  side : ℝ
  thickness : ℝ

/-- Represents a folding operation on a square sheet of paper. -/
noncomputable def fold (p : Paper) : Paper :=
  { side := p.side / Real.sqrt 2,
    thickness := p.thickness * 2 }

/-- Theorem stating the relationship between the original and final paper states after folding. -/
theorem original_side_length 
  (initial : Paper) 
  (final : Paper) 
  (n : ℕ) 
  (h1 : final = (fold^[n] initial)) 
  (h2 : final.side = 3) 
  (h3 : final.thickness = 16) : 
  initial.side = 12 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_side_length_l1062_106292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l1062_106241

/-- Given a curve f(x) = x^4 - x, if there exists a point P where the tangent line
    is perpendicular to the line x + 3y = 0, then the x-coordinate of P is 1. -/
theorem tangent_perpendicular_point (f : ℝ → ℝ) (x : ℝ) :
  f = (λ x => x^4 - x) →
  (∃ P : ℝ × ℝ, (deriv f P.1 = 3) ∧ P.1 = x) →
  x = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l1062_106241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_appropriate_for_low_carbon_lifestyle_option_c_is_correct_l1062_106267

/-- Represents a survey method -/
inductive SurveyMethod
| Sampling
| Complete

/-- Represents a characteristic of a population -/
structure PopulationCharacteristic where
  name : String

/-- Represents a city -/
structure City where
  name : String
  population : ℕ

/-- Defines the appropriateness of a survey method for a given characteristic in a city -/
def is_appropriate_method (method : SurveyMethod) (characteristic : PopulationCharacteristic) (city : City) : Prop :=
  match method with
  | SurveyMethod.Sampling => characteristic.name = "low-carbon lifestyle" ∧ city.population > 10000
  | SurveyMethod.Complete => city.population ≤ 10000

/-- The main theorem stating that sampling survey methods are appropriate for understanding 
    the low-carbon lifestyle of people in an entire city -/
theorem sampling_appropriate_for_low_carbon_lifestyle (city : City) 
    (h : city.population > 10000) : 
    is_appropriate_method SurveyMethod.Sampling 
      { name := "low-carbon lifestyle" } city := by
  sorry

/-- Custom definitions for list operations not available in the standard library -/
def List.mode (l : List α) [DecidableEq α] : Option α :=
  sorry

def List.median (l : List α) [Ord α] : Option α :=
  sorry

def List.variance (l : List ℤ) : ℚ :=
  sorry

def List.isMoreStable (l1 l2 : List ℤ) : Prop :=
  sorry

/-- Theorem stating that option C is the only correct statement among the four options -/
theorem option_c_is_correct : 
    ∃ (city : City), 
      is_appropriate_method SurveyMethod.Sampling 
        { name := "low-carbon lifestyle" } city ∧ 
      (∀ (other_city : City), 
        ¬(other_city.population ≤ 2 → 
          (∃ (n : ℕ), n ≤ 2 ∧ other_city.population = 2 * n)) ∧
        ¬(∃ (data : List ℕ), data = [2, 2, 3, 6] ∧ 
          data.mode = some 2 ∧ data.median = some 2) ∧
        ¬(∀ (scores_a scores_b : List ℤ), 
          scores_a.length = 5 ∧ scores_b.length = 5 →
          scores_a.sum / scores_a.length = 90 ∧ 
          scores_b.sum / scores_b.length = 90 →
          scores_a.variance = 5 ∧ scores_b.variance = 12 →
          scores_b.isMoreStable scores_a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sampling_appropriate_for_low_carbon_lifestyle_option_c_is_correct_l1062_106267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1062_106213

-- Define the parabola E
noncomputable def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the directrix l
noncomputable def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define point T where l intersects the x-axis
def T : ℝ × ℝ := (-1, 0)

-- Define a point A on E
variable (A : ℝ × ℝ)

-- Define A₁ as the foot of the perpendicular from A to l
noncomputable def A₁ (A : ℝ × ℝ) : ℝ × ℝ := (-1, A.2)

-- Define S as the intersection of A₁F and the y-axis
noncomputable def S (A : ℝ × ℝ) : ℝ × ℝ := (0, A.2 / 2)

theorem parabola_focus_distance (A : ℝ × ℝ) 
  (hA : A ∈ E) 
  (hParallel : (S A).2 - T.2 = A.2 - F.2) : 
  ‖A - F‖ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l1062_106213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_isosceles_condition_l1062_106232

/-- Represents an ellipse with semi-major axis a and semi-minor axis √2 -/
structure Ellipse (a : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 2 = 1

/-- Represents a line with slope e and y-intercept k -/
structure Line (e k : ℝ) where
  eq : ∀ x y : ℝ, y = e * x + k

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a : ℝ) : ℝ := 
  Real.sqrt (a^2 - 2) / a

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the relationship between the ellipse parameters and the isosceles triangle condition -/
theorem ellipse_isosceles_condition (a : ℝ) 
  (h1 : a > Real.sqrt 2) 
  (C : Ellipse a) 
  (e : ℝ) 
  (h2 : e = eccentricity a) 
  (l : Line e 0) 
  (F1 F2 P : Point) 
  (h3 : F1.x = -Real.sqrt (a^2 - 2) ∧ F1.y = 0) 
  (h4 : F2.x = Real.sqrt (a^2 - 2) ∧ F2.y = 0) 
  (h5 : P.y - F1.y = e * (P.x - F1.x)) 
  (h6 : distance P F2 = distance F1 F2) : 
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_isosceles_condition_l1062_106232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l1062_106275

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * Real.exp x

-- State the theorem
theorem max_difference_of_f (k : ℝ) (x₁ x₂ : ℝ) 
  (hk : k ∈ Set.Icc (-3) (-1))
  (hx₁ : x₁ ∈ Set.Icc k (k + 2))
  (hx₂ : x₂ ∈ Set.Icc k (k + 2)) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := by
  sorry

#check max_difference_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l1062_106275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_at_min_focal_length_l1062_106297

/-- Represents a hyperbola with parameter m -/
structure Hyperbola (m : ℝ) where
  eq : ∀ x y : ℝ, x^2 / (m^2 + 8) - y^2 / (6 - 2*m) = 1

/-- The focal length of the hyperbola -/
noncomputable def focal_length (m : ℝ) : ℝ := 2 * Real.sqrt (m^2 - 2*m + 14)

/-- The value of m that minimizes the focal length -/
def m_min : ℝ := 1

/-- The equation of the asymptotes when focal length is minimum -/
def asymptote_eq (x y : ℝ) : Prop := y = 2/3 * x ∨ y = -2/3 * x

theorem hyperbola_asymptotes_at_min_focal_length :
  ∀ m : ℝ, 
  (∀ m' : ℝ, focal_length m ≤ focal_length m') →
  ∀ x y : ℝ, asymptote_eq x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_at_min_focal_length_l1062_106297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1062_106265

noncomputable def f (x : ℝ) := Real.log (2 * x^2 - 3)

theorem f_decreasing_interval :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < -Real.sqrt 6 / 2 → f x₁ > f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l1062_106265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1062_106268

theorem sin_double_angle_special_case (α : ℝ) 
  (h1 : Real.cos (5 * π / 12 + α) = 1 / 3)
  (h2 : -π < α ∧ α < -π / 2) : 
  Real.sin (2 * (5 * π / 12 + α)) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1062_106268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_per_cup_l1062_106279

/-- Given a bird feeder scenario, calculate the number of birds that can be fed by one cup of birdseed -/
theorem birds_per_cup 
  (total_capacity : ℝ) 
  (stolen_amount : ℝ) 
  (birds_fed : ℕ) 
  (h1 : total_capacity = 2) 
  (h2 : stolen_amount = 0.5) 
  (h3 : birds_fed = 21) : 
  (birds_fed : ℝ) / (total_capacity - stolen_amount) = 14 := by
  sorry

#check birds_per_cup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_per_cup_l1062_106279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_doll_size_l1062_106245

/-- Represents the size of a doll in a Russian nesting doll set -/
noncomputable def doll_size (n : ℕ) : ℝ :=
  243 * (2/3)^(n-1)

/-- Theorem stating that the 6th doll in the sequence is 32 cm tall -/
theorem sixth_doll_size : doll_size 6 = 32 := by
  -- Unfold the definition of doll_size
  unfold doll_size
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_doll_size_l1062_106245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_decreasing_h_for_increasing_f_l1062_106201

-- Define the necessary functions and properties
def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f y < f x

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

def NonNegativeReals : Set ℝ := { x : ℝ | 0 ≤ x }

-- State the theorem
theorem no_decreasing_h_for_increasing_f :
  ¬ ∃ h : ℝ → ℝ, 
    DecreasingOn h NonNegativeReals ∧ 
    IncreasingOn (fun x => (x^2 - x + 1) * h x) NonNegativeReals :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_decreasing_h_for_increasing_f_l1062_106201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l1062_106246

/-- The function f(x) defined as ln x + ax + 1/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x + 1 / x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / x + a - 1 / (x ^ 2)

/-- The set of possible values for a -/
def A : Set ℝ := {a | a ≤ -1/4 ∨ a ≥ 0}

theorem monotonically_increasing_condition (a : ℝ) :
  (∀ x ≥ 1, f_derivative a x ≥ 0) ↔ a ∈ A := by
  sorry

#check monotonically_increasing_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_increasing_condition_l1062_106246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l1062_106224

-- Define the structure for a line segment
structure LineSegment where
  length : ℝ

-- Define the structure for a distance
structure Distance where
  value : ℝ

-- Define the parallel lines
def AB : LineSegment := { length := 180 }
def CD : LineSegment := { length := 120 }
def EF : LineSegment := { length := 180 } -- We'll prove this

-- Define the heights
def h_CD : Distance := { value := 1 } -- Arbitrary value, as we only need the ratio
def h_AB : Distance := { value := 1.5 * h_CD.value }

-- Theorem statement
theorem EF_length :
  AB.length = 180 →
  CD.length = 120 →
  h_AB.value = 1.5 * h_CD.value →
  EF.length = 180 := by
  intros hAB hCD hHeight
  -- The proof would go here
  sorry

#check EF_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_EF_length_l1062_106224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1062_106231

/-- Arithmetic sequence sum -/
def S (n : ℕ) (a : ℕ) : ℕ := n * a + n * (n - 1)

/-- Geometric sequence sum -/
def T (n : ℕ) (a : ℕ) : ℕ := a * (a^n - 1) / (a - 1)

theorem arithmetic_geometric_sequence_problem :
  (∃ (a : ℕ), a > 0 ∧ (∃ (x y : ℕ), ({2*a, 1, a^2 + 3} : Multiset ℕ) = {x, x + y, x + 2*y})) ∧
  (∀ (a : ℕ), a > 0 → (∃ (x y : ℕ), ({2*a, 1, a^2 + 3} : Multiset ℕ) = {x, x + y, x + 2*y}) → a = 2) ∧
  (∀ (n : ℕ), n > 0 → ((T n 2 + 2 : ℚ) / 2^n > (S n 2 : ℚ) - 130 ↔ n ≤ 10)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_problem_l1062_106231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_E_radius_l1062_106208

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  vertices : List (ℝ × ℝ)

/-- Predicate to check if a triangle is inscribed in a circle -/
def inscribed_in (T : EquilateralTriangle) (C : Circle) : Prop :=
  sorry

/-- Predicate to check if one circle is internally tangent to another at a specific point -/
def internally_tangent_to (C1 : Circle) (C2 : Circle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if one circle is externally tangent to another -/
def externally_tangent_to (C1 : Circle) (C2 : Circle) : Prop :=
  sorry

/-- Main theorem statement -/
theorem circle_E_radius
  (A : Circle)
  (B : Circle)
  (C : Circle)
  (D : Circle)
  (E : Circle)
  (T : EquilateralTriangle)
  (h1 : A.radius = 10)
  (h2 : B.radius = 4)
  (h3 : C.radius = 2)
  (h4 : D.radius = 2)
  (h5 : T.vertices.length = 3)
  (h6 : inscribed_in T A)
  (h7 : internally_tangent_to B A (T.vertices.get! 0))
  (h8 : internally_tangent_to C A (T.vertices.get! 1))
  (h9 : internally_tangent_to D A (T.vertices.get! 2))
  (h10 : externally_tangent_to B E)
  (h11 : externally_tangent_to C E)
  (h12 : externally_tangent_to D E) :
  E.radius = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_E_radius_l1062_106208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_repeating_decimals_l1062_106286

/-- A function that determines if a fraction n/(n+2) is a repeating decimal -/
def isRepeatingDecimal (n : Nat) : Bool :=
  let denominator := n + 2
  let simplifiedDenominator := denominator / (Nat.gcd n denominator)
  ¬ (simplifiedDenominator.factors.all (λ p => p = 2 ∨ p = 5))

/-- The count of integers n where 1 ≤ n ≤ 150 such that n/(n+2) is a repeating decimal -/
def countRepeatingDecimals : Nat :=
  (Finset.range 150).filter (λ n => isRepeatingDecimal (n + 1)) |>.card

theorem count_repeating_decimals :
  countRepeatingDecimals = 134 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_repeating_decimals_l1062_106286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qda_area_l1062_106204

/-- The area of triangle QDA given the coordinates of points Q, A, D, and conditions on p and q -/
theorem triangle_qda_area (q p : ℝ) (hq : q > 0) (hp : p < 15) : 
  (1/2 : ℝ) * q * (15 - p) = 
  (1/2 : ℝ) * (q - 0) * (15 - p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qda_area_l1062_106204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_value_9_98_to_fifth_l1062_106294

theorem approx_value_9_98_to_fifth : 
  ∃ (y : ℤ), abs ((10 - 0.02 : ℝ)^5 - (10^5 - 5 * 10^4 * 0.02 + 10 * 10^3 * 0.02^2) - y) < 0.5 ∧ y = 99004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_value_9_98_to_fifth_l1062_106294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_ratio_l1062_106225

theorem cube_surface_area_ratio (x : ℝ) (h : x > 0) :
  (6 * (6 * x)^2) / (6 * x^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_ratio_l1062_106225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1062_106270

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
noncomputable def train_length (speed1 speed2 : ℝ) (crossing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  let combined_length := relative_speed * crossing_time
  combined_length - other_train_length

/-- The length of the first train is 220 meters. -/
theorem first_train_length :
  train_length 120 80 9 280.04 = 220 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l1062_106270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_calculation_l1062_106242

theorem saree_price_calculation (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 222.904 ∧ discount1 = 0.12 ∧ discount2 = 0.15 →
  ∃ original_price : ℝ, 
    final_price = original_price * (1 - discount1) * (1 - discount2) ∧
    abs (original_price - 297.86) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_price_calculation_l1062_106242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1062_106283

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 4| + |x - 2*a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 ≤ 13} = Set.Icc (-13/2) (13/2) := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f x a ≥ a^2 + 5*a} = Set.Icc ((-7-Real.sqrt 33)/2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_a_part2_l1062_106283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_constraint_l1062_106257

theorem max_value_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) :
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_constraint_l1062_106257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_zeros_reciprocal_l1062_106228

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x + m * (x - 1)

-- Theorem for part 1
theorem f_monotonic_increasing :
  Monotone (f 1) := by sorry

-- Theorem for part 2
theorem f_zeros_reciprocal (m : ℝ) (hm : m < -2) :
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧
    f m x₁ = 0 ∧ f m x₂ = 0 ∧ x₁ * x₂ = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_increasing_f_zeros_reciprocal_l1062_106228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1062_106263

noncomputable section

variables (a b : ℝ)

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x + 3 else a * x + b

theorem function_property :
  (∀ x₁ : ℝ, ∃! x₂ : ℝ, f a b x₁ = f a b x₂) →
  (f a b (2 * a) = f a b (3 * b)) →
  a + b = -Real.sqrt 6 / 2 + 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l1062_106263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l1062_106272

variable (f g : ℝ → ℝ) (a : ℝ)

-- Condition 1
def condition1 : Prop := ∀ x, f x = 2 * a^x * g x ∧ a > 0 ∧ a ≠ 1

-- Condition 2
def condition2 : Prop := ∀ x, g x ≠ 0

-- Condition 3
noncomputable def condition3 : Prop := ∀ x, f x * (deriv g x) < (deriv f x) * g x

-- Condition 4
def condition4 : Prop := f 1 / g 1 + f (-1) / g (-1) = 5

-- Theorem statement
theorem determine_a (h1 : condition1 f g a) (h2 : condition2 g) (h3 : condition3 f g) (h4 : condition4 f g) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l1062_106272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_eq_two_l1062_106223

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-1) 1 then 2 else x

theorem range_of_f_f_eq_two :
  {x : ℝ | f (f x) = 2} = {2} ∪ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_f_eq_two_l1062_106223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ac_mn_is_90_degrees_l1062_106249

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle (a b : ℝ) where
  side_a : a > 0
  side_b : b > 0
  b_greater_a : b > a

/-- Represents a fold line in the rectangle -/
structure FoldLine (a b : ℝ) where
  rectangle : Rectangle a b
  m_on_ab : ℝ
  n_on_cd : ℝ
  m_in_range : 0 ≤ m_on_ab ∧ m_on_ab ≤ b
  n_in_range : 0 ≤ n_on_cd ∧ n_on_cd ≤ b

/-- Represents a 2D line -/
structure Line2D where
  -- Define properties of a 2D line
  slope : ℝ
  intercept : ℝ

/-- Helper function to represent the line A'C -/
def line_ac (a b : ℝ) (fold : FoldLine a b) : Line2D := sorry

/-- Helper function to represent the line MN -/
def line_mn (fold : FoldLine a b) : Line2D := sorry

/-- Helper function to calculate the angle between two lines -/
noncomputable def angle_between_lines (l1 l2 : Line2D) : ℝ := sorry

/-- Theorem: The angle between A'C and MN is 90 degrees -/
theorem angle_ac_mn_is_90_degrees
  (a b : ℝ)
  (rect : Rectangle a b)
  (fold : FoldLine a b)
  (dihedral_angle : ℝ)
  (h_dihedral : dihedral_angle = 18) :
  angle_between_lines (line_ac a b fold) (line_mn fold) = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ac_mn_is_90_degrees_l1062_106249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1062_106237

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / (|x + 3| - 3)

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)}

-- Theorem statement
theorem f_is_odd : ∀ x ∈ dom_f, f (-x) = -f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l1062_106237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_16km_l1062_106248

/-- Represents the scenario of two hunters and a dog meeting -/
structure HuntersDogScenario where
  distance : ℝ  -- Distance between the two villages
  speed1 : ℝ    -- Speed of the first hunter
  speed2 : ℝ    -- Speed of the second hunter
  speedDog : ℝ  -- Speed of the dog

/-- Calculates the total distance run by the dog -/
noncomputable def dogDistance (scenario : HuntersDogScenario) : ℝ :=
  (scenario.speedDog * scenario.distance) / (scenario.speed1 + scenario.speed2)

/-- Theorem stating that the dog runs 16 km given the specific scenario -/
theorem dog_distance_16km :
  let scenario : HuntersDogScenario := {
    distance := 18,
    speed1 := 5,
    speed2 := 4,
    speedDog := 8
  }
  dogDistance scenario = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_distance_16km_l1062_106248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_l1062_106298

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cos_angle_between_vectors (p q : V) 
  (hp : ‖p‖ = 7)
  (hq : ‖q‖ = 10)
  (hpq : ‖p + q‖ = 13) :
  Real.cos (Real.arccos (inner p q / (‖p‖ * ‖q‖))) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_between_vectors_l1062_106298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l1062_106202

theorem problem_1 : Real.sqrt 36 - 3 * (-1) ^ 2023 + ((-8) ^ (1/3 : ℝ)) = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l1062_106202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_four_out_of_five_l1062_106289

/-- Represents a chessboard size -/
structure BoardSize where
  rows : Nat
  cols : Nat

/-- Defines the game rules and winning condition -/
def canFirstPlayerWin (board : BoardSize) : Bool :=
  board.rows % 2 == 0 || board.cols % 2 == 0

/-- List of given board sizes -/
def givenBoards : List BoardSize :=
  [⟨6, 7⟩, ⟨6, 8⟩, ⟨7, 7⟩, ⟨7, 8⟩, ⟨8, 8⟩]

/-- The main theorem to be proved -/
theorem first_player_wins_four_out_of_five :
  (givenBoards.filter canFirstPlayerWin).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_four_out_of_five_l1062_106289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_split_l1062_106244

theorem restaurant_bill_split (total_bill : ℚ) (num_people : ℕ) (smallest_unit : ℚ) :
  total_bill = 514.16 →
  num_people = 9 →
  smallest_unit = 1/100 →
  (total_bill / num_people).floor * smallest_unit + smallest_unit = 57.13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_split_l1062_106244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_regular_polygon_l1062_106253

/-- A regular polygon with 1998 sides -/
structure RegularPolygon1998 where
  vertices : Fin 1998 → ℝ × ℝ
  is_regular : ∀ i j : Fin 1998, 
    dist (vertices i) (vertices ((i + 1) % 1998)) = dist (vertices j) (vertices ((j + 1) % 1998))

/-- The trajectory of the ball -/
structure Trajectory (A : RegularPolygon1998) where
  points : Fin 1998 → ℝ × ℝ
  start_midpoint : points 0 = (A.vertices 0 + A.vertices 1) / 2
  on_sides : ∀ i : Fin 1998, 
    ∃ t : ℝ, points i = (1 - t) • (A.vertices i) + t • (A.vertices ((i + 1) % 1998))
  reflection_law : ∀ i : Fin 1998, 
    (points ((i - 1) % 1998) - points i) • (A.vertices ((i + 1) % 1998) - points i) =
    (A.vertices i - points i) • (points ((i + 1) % 1998) - points i)
  closes : points 1997 = points 0

/-- The main theorem -/
theorem trajectory_is_regular_polygon 
  (A : RegularPolygon1998) (B : Trajectory A) : 
  ∃ (C : RegularPolygon1998), ∀ i : Fin 1998, C.vertices i = B.points i :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_regular_polygon_l1062_106253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milas_hourly_rate_l1062_106293

/-- Proves that Mila's hourly rate is $10 given the specified conditions -/
theorem milas_hourly_rate (agnes_rate : ℝ) (agnes_hours_per_week : ℝ) 
  (weeks_per_month : ℝ) (mila_hours_per_month : ℝ) 
  (h1 : agnes_rate = 15)
  (h2 : agnes_hours_per_week = 8)
  (h3 : weeks_per_month = 4)
  (h4 : mila_hours_per_month = 48) :
  agnes_rate * agnes_hours_per_week * weeks_per_month / mila_hours_per_month = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milas_hourly_rate_l1062_106293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_k_values_l1062_106243

theorem possible_k_values (a b c k : ℕ+) 
  (h : (a + b + c)^2 = k * (a * b * c)) :
  (k : ℕ) ∈ ({1, 2, 3, 4, 5, 6, 8, 9} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_k_values_l1062_106243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_f_monotonic_increase_l1062_106236

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem f_properties :
  -- 1. Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  -- 2. Maximum and minimum values on [0, π/3]
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi/3 → f x ≤ 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi/3 ∧ f x = 2) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi/3 → f x ≥ 0) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi/3 ∧ f x = 0) ∧
  -- 3. Equivalent form of f(x)
  (∀ (x : ℝ), f x = 2 * Real.sin (2*x + Real.pi/3)) := by
  sorry

-- Additional theorem for the period
theorem f_period :
  ∃ (T : ℝ), T = Real.pi ∧ T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x := by
  sorry

-- Additional theorem for the monotonic increase interval
theorem f_monotonic_increase :
  ∀ (k : ℤ), ∀ (x : ℝ), -5*Real.pi/12 + k*Real.pi ≤ x ∧ x ≤ Real.pi/12 + k*Real.pi → 
    ∀ (y : ℝ), -5*Real.pi/12 + k*Real.pi ≤ y ∧ y ≤ x → f y ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_f_monotonic_increase_l1062_106236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_is_two_std_dev_below_mean_l1062_106220

/-- Calculates the number of standard deviations a value is from the mean -/
noncomputable def standardDeviationsFromMean (mean stdDev value : ℝ) : ℝ :=
  (value - mean) / stdDev

theorem value_is_two_std_dev_below_mean 
  (mean : ℝ) (stdDev : ℝ) (value : ℝ)
  (h_mean : mean = 12)
  (h_stdDev : stdDev = 1.2)
  (h_value : value = 9.6) :
  standardDeviationsFromMean mean stdDev value = -2 := by
  -- Unfold the definition of standardDeviationsFromMean
  unfold standardDeviationsFromMean
  -- Substitute the given values
  rw [h_mean, h_stdDev, h_value]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_is_two_std_dev_below_mean_l1062_106220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_faces_same_edges_l1062_106227

/-- Represents a face of a polyhedron -/
structure Face where
  edges : ℕ
  edges_ge_three : edges ≥ 3

/-- Represents a polyhedron -/
structure Polyhedron where
  faces : Finset Face
  nonempty : faces.Nonempty

/-- Theorem: In any polyhedron, there exist at least two faces with the same number of edges -/
theorem two_faces_same_edges (P : Polyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.edges = f₂.edges := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_faces_same_edges_l1062_106227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_length_l1062_106258

/-- A square with side length 2 is divided into a trapezoid and a triangle with equal areas.
    The division is made by connecting the center of the square to a midpoint of one side
    and to a point on the adjacent side at distance 1 from the center.
    This theorem proves that the length of the longer parallel side of the trapezoid is 3. -/
theorem trapezoid_longer_side_length (square_side : ℝ) (center_to_point_dist : ℝ) (longer_side : ℝ) : 
  square_side = 2 →
  center_to_point_dist = 1 →
  (longer_side + center_to_point_dist) / 2 = square_side^2 / 4 - (longer_side + center_to_point_dist) / 2 →
  longer_side = 3 := by
  intros h1 h2 h3
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_length_l1062_106258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l1062_106238

theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * (x - π/6) + π/3) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_equivalence_l1062_106238
