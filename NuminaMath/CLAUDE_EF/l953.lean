import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_value_of_g_l953_95307

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x + 1)

-- Define the function g
noncomputable def g (a x : ℝ) : ℝ := f x - a * Real.exp x - x

-- Theorem for the tangent line
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (∀ x, m * x + b = 2 * x) ∧
  (m * 0 + b = f 0) ∧
  HasDerivAt f 2 0 :=
sorry

-- Theorem for the maximum value of g
theorem max_value_of_g (a : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧
    ∀ y ∈ Set.Icc 1 2, g a y ≤ g a x) ∧
  ((a ≥ (2 * Real.exp 1 - 1) / (Real.exp 1 - 1) →
    ∃ x ∈ Set.Icc 1 2, g a x = (1 - a) * Real.exp 1) ∧
   (a < (2 * Real.exp 1 - 1) / (Real.exp 1 - 1) →
    ∃ x ∈ Set.Icc 1 2, g a x = (2 - a) * Real.exp 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_max_value_of_g_l953_95307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_circle_l953_95309

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem midpoint_locus_is_circle (K : Circle) (P : Point) :
  K.radius = 10 →
  distance P K.center = 4 →
  ∃ (C : Circle), C.radius = 2 ∧
    ∀ (Q : Point),
      (∃ (A B : Point), distance A K.center ≤ K.radius ∧
                        distance B K.center ≤ K.radius ∧
                        Q = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
                        distance P A = distance P B) →
      distance Q C.center = C.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_is_circle_l953_95309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l953_95317

/-- A sequence satisfying the given recurrence relation -/
def a : ℕ → ℤ
  | 0 => 11
  | 1 => 11
  | n + 2 => (a (2 * (n / 2)) + a (2 * (n % 2))) / 2 - ((n / 2) - (n % 2))^2

/-- The theorem stating that the 45th term of the sequence is 1991 -/
theorem a_45_equals_1991 : a 45 = 1991 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_45_equals_1991_l953_95317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bellas_siblings_l953_95345

-- Define the characteristics
inductive EyeColor
| Green
| Gray
deriving BEq, Repr

inductive HairColor
| Red
| Brown
deriving BEq, Repr

inductive AgeGroup
| Older
| Younger
deriving BEq, Repr

-- Define a child's characteristics
structure ChildCharacteristics where
  eyes : EyeColor
  hair : HairColor
  age : AgeGroup
deriving BEq, Repr

-- Define the children
def Bella : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Red, AgeGroup.Older⟩
def Derek : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Red, AgeGroup.Younger⟩
def Olivia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Brown, AgeGroup.Older⟩
def Lucas : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Brown, AgeGroup.Younger⟩
def Emma : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Red, AgeGroup.Older⟩
def Ryan : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Red, AgeGroup.Older⟩
def Sophia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Brown, AgeGroup.Younger⟩
def Ethan : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Brown, AgeGroup.Older⟩

-- Function to count shared characteristics
def sharedCharacteristics (c1 c2 : ChildCharacteristics) : Nat :=
  (if c1.eyes == c2.eyes then 1 else 0) +
  (if c1.hair == c2.hair then 1 else 0) +
  (if c1.age == c2.age then 1 else 0)

-- Define what it means to be siblings
def areSiblings (c1 c2 c3 : ChildCharacteristics) : Prop :=
  sharedCharacteristics c1 c2 ≥ 2 ∧
  sharedCharacteristics c1 c3 ≥ 2 ∧
  sharedCharacteristics c2 c3 ≥ 2

-- Theorem statement
theorem bellas_siblings :
  areSiblings Bella Emma Olivia ∧
  ∀ (c1 c2 : ChildCharacteristics),
    c1 ≠ Emma → c2 ≠ Olivia →
    ¬(areSiblings Bella c1 c2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bellas_siblings_l953_95345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_initial_savings_l953_95370

/-- Represents Julie's savings and interest calculations --/
structure JulieSavings where
  simple_interest_rate : ℝ
  compound_interest_rate : ℝ
  time : ℝ
  simple_interest_earned : ℝ
  compound_interest_earned : ℝ

/-- Calculates the initial savings based on the given conditions --/
noncomputable def calculate_initial_savings (s : JulieSavings) : ℝ :=
  let simple_principal := s.simple_interest_earned / (s.simple_interest_rate * s.time)
  let compound_principal := s.compound_interest_earned / ((1 + s.compound_interest_rate) ^ s.time - 1)
  simple_principal + compound_principal

/-- Theorem stating that Julie's initial savings were approximately $3773.85 --/
theorem julie_initial_savings :
  let s : JulieSavings := {
    simple_interest_rate := 0.04,
    compound_interest_rate := 0.05,
    time := 5,
    simple_interest_earned := 400,
    compound_interest_earned := 490
  }
  abs (calculate_initial_savings s - 3773.85) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_initial_savings_l953_95370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AE_length_l953_95313

/-- Given a lattice with points one unit apart, and points A(0,4), B(7,0), C(5,3), D(3,0),
    prove that the length of segment AE is 4√65/7, where E is the intersection of AB and CD. -/
theorem segment_AE_length :
  let A : ℝ × ℝ := (0, 4)
  let B : ℝ × ℝ := (7, 0)
  let C : ℝ × ℝ := (5, 3)
  let D : ℝ × ℝ := (3, 0)
  let E : ℝ × ℝ := 
    (((A.1 * B.2 - A.2 * B.1) * (C.1 - D.1) - (A.1 - B.1) * (C.1 * D.2 - C.2 * D.1)) / 
    ((A.1 - B.1) * (C.2 - D.2) - (A.2 - B.2) * (C.1 - D.1)),
    ((A.1 * B.2 - A.2 * B.1) * (C.2 - D.2) - (A.2 - B.2) * (C.1 * D.2 - C.2 * D.1)) / 
    ((A.1 - B.1) * (C.2 - D.2) - (A.2 - B.2) * (C.1 - D.1)))
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 4 * Real.sqrt 65 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AE_length_l953_95313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l953_95343

theorem cos_five_theta (θ : Real) (h : Real.cos θ = 1/4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_theta_l953_95343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l953_95374

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  5 / (4 * x^(-4 : ℝ)) + (4 * x^3) / 5 = (x^3 * (25 * x + 16)) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l953_95374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_typing_hours_per_day_l953_95352

/-- Given Carl's typing speed and total output over a week, prove how many hours he types per day -/
theorem carl_typing_hours_per_day 
  (typing_speed : ℕ) -- Carl's typing speed in words per minute
  (total_words : ℕ) -- Total words typed in a week
  (total_days : ℕ) -- Number of days in a week
  (h1 : typing_speed = 50)
  (h2 : total_words = 84000)
  (h3 : total_days = 7) :
  (total_words / total_days) / typing_speed / 60 = 4 := by
  -- Proof steps would go here
  sorry

#check carl_typing_hours_per_day

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_typing_hours_per_day_l953_95352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l953_95306

-- Define the hyperbola
def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def is_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 3

-- Define the asymptote of the hyperbola
def is_asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = (b / a) * x ∨ y = -(b / a) * x

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), is_asymptote a b x y ∧ is_circle x y

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) :
  (∀ x y, is_hyperbola a b x y) →
  is_tangent a b →
  eccentricity a b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l953_95306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l953_95346

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x

-- Define the point of tangency
def point : ℝ × ℝ := (1, -1)

-- State the theorem
theorem tangent_line_equation :
  let slope := (deriv f) point.fst
  (λ (x y : ℝ) => x - y - 2 = 0) = 
  (λ (x y : ℝ) => y - point.snd = slope * (x - point.fst)) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l953_95346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_b_equals_3_g_monotone_increasing_t_range_when_g_sum_negative_l953_95366

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-2)*x + 4
def g (a b : ℝ) (x : ℝ) : ℝ := (x + b - 3) / (a*x^2 + 2)

-- Theorem 1: If f is even on [-b, b^2-b-3], then b = 3
theorem even_function_implies_b_equals_3 (a b : ℝ) :
  (∀ x ∈ Set.Icc (-b) (b^2 - b - 3), f a x = f a (-x)) →
  b = 3 := by
  sorry

-- Theorem 2: g is monotonically increasing on (-1,1)
theorem g_monotone_increasing :
  StrictMonoOn (g 2 3) (Set.Ioo (-1) 1) := by
  sorry

-- Theorem 3: If g(t-1) + g(2t) < 0, then t ∈ (0, 1/3)
theorem t_range_when_g_sum_negative :
  ∀ t : ℝ, g 2 3 (t-1) + g 2 3 (2*t) < 0 → t ∈ Set.Ioo 0 (1/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_b_equals_3_g_monotone_increasing_t_range_when_g_sum_negative_l953_95366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_price_theorem_l953_95384

/-- The price per kilogram of pears given the conditions of the problem -/
noncomputable def price_per_kg_pears : ℚ :=
  let apple_kg : ℚ := 45
  let pear_kg : ℚ := 36
  let price_difference : ℚ := 32.4
  32.4 / (apple_kg - pear_kg)

/-- Theorem stating that the price per kilogram of pears is 32.4 / 9 yuan -/
theorem pear_price_theorem : price_per_kg_pears = 32.4 / 9 := by
  -- Unfold the definition of price_per_kg_pears
  unfold price_per_kg_pears
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pear_price_theorem_l953_95384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_premises_l953_95308

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Chinese : U → Prop)
variable (StrongAndUnyielding : U → Prop)
variable (PeopleOfYaAn : U → Prop)

-- Define the syllogism statements
def statement1 (U : Type) (PeopleOfYaAn StrongAndUnyielding : U → Prop) : Prop :=
  ∀ x, PeopleOfYaAn x → StrongAndUnyielding x

def statement2 (U : Type) (PeopleOfYaAn Chinese : U → Prop) : Prop :=
  ∀ x, PeopleOfYaAn x → Chinese x

def statement3 (U : Type) (Chinese StrongAndUnyielding : U → Prop) : Prop :=
  ∀ x, Chinese x → StrongAndUnyielding x

-- Define what it means to be a major premise and a minor premise
def isMajorPremise (U : Type) (s : Prop) : Prop :=
  ∃ (P Q : U → Prop), s = ∀ x, P x → Q x

def isMinorPremise (U : Type) (s : Prop) (major : Prop) : Prop := 
  ∃ (P Q R : U → Prop), major = (∀ x, Q x → R x) ∧ s = (∀ x, P x → Q x)

-- Theorem statement
theorem syllogism_premises (U : Type) (Chinese StrongAndUnyielding PeopleOfYaAn : U → Prop) :
  isMajorPremise U (statement3 U Chinese StrongAndUnyielding) ∧
  isMinorPremise U (statement2 U PeopleOfYaAn Chinese) (statement3 U Chinese StrongAndUnyielding) :=
by
  constructor
  . -- Proof that statement3 is the major premise
    use Chinese, StrongAndUnyielding
    rfl
  . -- Proof that statement2 is the minor premise
    use PeopleOfYaAn, Chinese, StrongAndUnyielding
    constructor
    . rfl
    . rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_premises_l953_95308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_distance_l953_95302

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distance (runner : Runner) (t : ℝ) : ℝ :=
  runner.speed * t

/-- Represents a race between two runners -/
structure Race where
  length : ℝ
  runner_a : Runner
  runner_b : Runner
  time_difference : ℝ

theorem race_winner_distance (race : Race) 
  (h1 : race.length = 1000) -- Race length is 1 kilometer (1000 meters)
  (h2 : race.runner_a.time = 490) -- A completes the race in 490 seconds
  (h3 : race.time_difference = 10) -- A beats B by 10 seconds
  : ∃ ε > 0, |distance race.runner_a race.time_difference - 20.408| < ε := by
  sorry

#eval Float.ceil (1000 / 490 * 10 * 1000) / 1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_winner_distance_l953_95302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_proof_l953_95351

theorem salary_change_proof (initial_salary : ℝ) (initial_salary_pos : initial_salary > 0) :
  let after_first_increase := initial_salary * 1.25
  let after_first_decrease := after_first_increase * 0.9
  let after_second_increase := after_first_decrease * 1.3
  let final_salary := after_second_increase * 0.8
  (final_salary - initial_salary) / initial_salary = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_change_proof_l953_95351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_m_l953_95337

open Real

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x - m / x
noncomputable def g (x : ℝ) : ℝ := 3 * log x

-- Part 1: Tangent line equation
theorem tangent_line_equation :
  let f₄ := f 4
  let x₀ := 2
  let y₀ := f₄ x₀
  let m := (deriv f₄) x₀
  (fun x ↦ m * (x - x₀) + y₀) = (fun x ↦ 5 * x - 4) := by sorry

-- Part 2: Range of m
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 1 (sqrt ℯ), f m x - g x < 3) →
  m < (9 * sqrt ℯ) / (2 * (ℯ - 1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_m_l953_95337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_l953_95310

theorem largest_of_three : (∀ x ∈ ({5, 8, 4} : Set ℕ), x ≤ 8) ∧ 8 ∈ ({5, 8, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_l953_95310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l953_95339

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (Real.sin x)

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (5 * Real.pi / 6),
  ∃ y ∈ Set.Icc (1/2) 1, f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (1/2) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l953_95339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l953_95334

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a) / 2 * x^2 + a * x - Real.log x

theorem f_properties :
  (∀ x > 0, ∃ y, f 3 x ≤ 2 ∧ f 3 y = 2) ∧
  (∀ x > 0, ∃ y, f 3 x ≥ 5/4 + Real.log 2 ∧ f 3 y = 5/4 + Real.log 2) ∧
  (∀ a > 1, 
    (a < 2 → 
      (∀ x ∈ Set.Ioo 0 1 ∪ Set.Ioi (1/(a-1)), ∀ y ∈ Set.Ioo x (max x y), f a x ≥ f a y) ∧
      (∀ x ∈ Set.Ioo 1 (1/(a-1)), ∀ y ∈ Set.Ioo x (1/(a-1)), f a x ≤ f a y)) ∧
    (a = 2 → 
      (∀ x > 0, ∀ y > x, f a x ≥ f a y)) ∧
    (a > 2 → 
      (∀ x ∈ Set.Ioo 0 (1/(a-1)) ∪ Set.Ioi 1, ∀ y ∈ Set.Ioo x (max x y), f a x ≥ f a y) ∧
      (∀ x ∈ Set.Ioo (1/(a-1)) 1, ∀ y ∈ Set.Ioo x 1, f a x ≤ f a y))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l953_95334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l953_95350

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  equation : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1

/-- The asymptote of the hyperbola -/
noncomputable def asymptote (h : Hyperbola) : ℝ → ℝ := λ x => (Real.sqrt 3 / 3) * x

/-- The distance from a point to a line -/
noncomputable def distanceToLine (x y : ℝ) (m c : ℝ) : ℝ :=
  abs (y - m * x - c) / Real.sqrt (1 + m^2)

/-- Theorem stating the conditions and the resulting equation of the hyperbola -/
theorem hyperbola_equation (h : Hyperbola) :
  (∀ x, asymptote h x = (Real.sqrt 3 / 3) * x) →
  (distanceToLine h.a 0 (Real.sqrt 3 / 3) 0 = 1) →
  (∀ x y, h.equation x y ↔ x^2 / 4 - 3 * y^2 / 4 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l953_95350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l953_95348

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x - x - a

theorem negation_of_proposition :
  (¬∀ a : ℝ, a > 0 → a ≠ 1 → ∃ x : ℝ, f a x = 0) ↔
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f a x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l953_95348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l953_95378

/-- Calculate compound interest given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Calculate simple interest given principal, rate, and time -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ) / 100

theorem compound_interest_calculation 
  (P : ℝ) 
  (h1 : simple_interest P 5 2 = 58) :
  compound_interest P 5 2 = 59.45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l953_95378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l953_95305

-- Define Triangle as a structure with three points
structure Triangle (A B C : ℝ × ℝ) : Type where
  is_triangle : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

-- Define angle as a function that takes three points and returns a real number
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_angles (A B C : ℝ × ℝ) (M N K : ℝ × ℝ) :
  Triangle A B C →
  angle B A C = 60 →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ M = (1 - t) • B + t • C) →
  (∃ s : ℝ, 0 < s ∧ s < 1 ∧ N = (1 - s) • A + s • C) →
  (∃ r : ℝ, 0 < r ∧ r < 1 ∧ K = (1 - r) • A + r • B) →
  ‖B - K‖ = ‖K - M‖ ∧ ‖K - M‖ = ‖M - N‖ ∧ ‖M - N‖ = ‖N - C‖ →
  ‖A - N‖ = 2 * ‖A - K‖ →
  angle A B C = 75 ∧ angle A C B = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angles_l953_95305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l953_95323

-- Define the curve C
noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the polar equation of line l
def line_l_polar (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- State the theorem
theorem curve_line_intersection (m : ℝ) :
  (∀ x y, y = -Real.sqrt 3 * x - 2 * m →
    ∃ t, curve_C t = (x, y)) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l953_95323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_eight_l953_95383

/-- The number whose multiples are considered for calculating the average -/
noncomputable def x : ℝ := sorry

/-- The positive integer n -/
def n : ℕ := 16

/-- The average of the first 7 positive multiples of x -/
noncomputable def a : ℝ := (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7

/-- The median of the first 3 positive multiples of n -/
def b : ℝ := 2 * n

/-- Theorem stating that x = 8 given the conditions -/
theorem x_equals_eight : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_eight_l953_95383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_robins_l953_95367

theorem chase_robins (gabrielle_robins gabrielle_cardinals gabrielle_bluejays : ℕ)
                     (chase_bluejays chase_cardinals : ℕ)
                     (gabrielle_total chase_total : ℕ)
                     (chase_robins : ℕ)
                     (h1 : gabrielle_robins = 5)
                     (h2 : gabrielle_cardinals = 4)
                     (h3 : gabrielle_bluejays = 3)
                     (h4 : chase_bluejays = 3)
                     (h5 : chase_cardinals = 5)
                     (h6 : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_bluejays)
                     (h7 : chase_total = chase_robins + chase_bluejays + chase_cardinals)
                     (h8 : gabrielle_total = (120 : ℕ) * chase_total / 100) :
  chase_robins = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_robins_l953_95367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_revolution_volume_l953_95376

/-- A regular octahedron with side length s -/
structure RegularOctahedron where
  s : ℝ
  s_pos : 0 < s

/-- The volume of the solid created by revolving a regular octahedron about the line
    connecting the barycenters of two parallel faces -/
noncomputable def revolution_volume (octahedron : RegularOctahedron) : ℝ :=
  (5 * Real.pi * octahedron.s^3) / (9 * Real.sqrt 6)

/-- Theorem stating that the volume of the solid created by revolving a regular octahedron
    about the line connecting the barycenters of two parallel faces is (5π * s³) / (9√6) -/
theorem octahedron_revolution_volume (octahedron : RegularOctahedron) :
  revolution_volume octahedron = (5 * Real.pi * octahedron.s^3) / (9 * Real.sqrt 6) := by
  -- Unfold the definition of revolution_volume
  unfold revolution_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_revolution_volume_l953_95376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_factorization_l953_95379

theorem smallest_b_for_factorization : 
  ∀ b : ℕ+, 
  (∃ p q : ℤ, (X : Polynomial ℤ)^2 + b.val • X + 1512 = (X + p) * (X + q)) → 
  b ≥ 82 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_for_factorization_l953_95379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l953_95396

-- Define the function f
noncomputable def f (a : ℝ) (g : ℝ → ℝ) : ℝ → ℝ := 
  λ x => if x ≥ 0 then 2^x + a else g x

-- State the theorem
theorem odd_function_properties (a : ℝ) (g : ℝ → ℝ) 
  (h_odd : ∀ x, f a g x = -f a g (-x)) :
  (a = -1) ∧ (∀ x, f a g x + 3 = 0 ↔ x = -2) := by
  sorry

#check odd_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l953_95396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_storage_ratio_l953_95375

theorem bakery_storage_ratio :
  -- Define the given ratios and quantities
  let sugar_to_flour_ratio : ℚ := 5 / 4
  let sugar_amount : ℕ := 3000
  let flour_amount : ℕ := (sugar_amount * 4) / 5
  let hypothetical_flour_to_baking_soda_ratio : ℚ := 8 / 1
  let additional_baking_soda : ℕ := 60
  
  -- Define the current amount of baking soda
  let current_baking_soda : ℚ := 
    (flour_amount : ℚ) / hypothetical_flour_to_baking_soda_ratio - additional_baking_soda
  
  -- Define the actual flour to baking soda ratio
  let actual_flour_to_baking_soda_ratio : ℚ := (flour_amount : ℚ) / current_baking_soda
  
  -- Theorem statement
  actual_flour_to_baking_soda_ratio = 10 / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bakery_storage_ratio_l953_95375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l953_95391

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 5 * x + 6
noncomputable def g (x : ℝ) : ℝ := 4 * x - 7
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function
noncomputable def h_inv (x : ℝ) : ℝ := (x + 29) / 20

-- Theorem statement
theorem h_inverse_correct : 
  ∀ x : ℝ, h (h_inv x) = x ∧ h_inv (h x) = x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l953_95391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_trip_time_l953_95327

/-- Represents Julia's trip to the mountain retreat -/
structure MountainTrip where
  interstate_distance : ℝ
  mountain_road_distance : ℝ
  dirt_track_distance : ℝ
  interstate_speed : ℝ
  mountain_road_speed : ℝ
  dirt_track_speed : ℝ
  interstate_time : ℝ

/-- The conditions of Julia's trip -/
def julia_trip : MountainTrip where
  interstate_distance := 120
  mountain_road_distance := 40
  dirt_track_distance := 5
  interstate_speed := 2  -- We'll calculate this based on the given time
  mountain_road_speed := 1  -- This is half of the interstate speed
  dirt_track_speed := 0.5  -- This is a quarter of the interstate speed
  interstate_time := 60

/-- Calculate the total trip time -/
noncomputable def total_trip_time (trip : MountainTrip) : ℝ :=
  trip.interstate_time +
  trip.mountain_road_distance / trip.mountain_road_speed +
  trip.dirt_track_distance / trip.dirt_track_speed

/-- Theorem stating that Julia's total trip time is 110 minutes -/
theorem julia_trip_time :
  total_trip_time julia_trip = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_trip_time_l953_95327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_class_size_l953_95355

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents a class of children -/
structure ClassRoom where
  boys : ℕ
  girls : ℕ

/-- Condition: No two boys share the same birth day of the week -/
def valid_boys (c : ClassRoom) : Prop :=
  c.boys ≤ days_in_week

/-- Condition: No two girls share the same birth month -/
def valid_girls (c : ClassRoom) : Prop :=
  c.girls ≤ months_in_year

/-- Condition: Adding one more child would violate one of the conditions -/
def max_reached (c : ClassRoom) : Prop :=
  c.boys = days_in_week ∧ c.girls = months_in_year

/-- The theorem to be proved -/
theorem max_class_size :
  ∀ c : ClassRoom,
    valid_boys c →
    valid_girls c →
    max_reached c →
    c.boys + c.girls = 19 :=
by
  intro c hb hg hm
  cases hm with
  | intro hboys hgirls =>
    rw [hboys, hgirls]
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_class_size_l953_95355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l953_95321

-- Define the circle C: (x+1)^2 + y^2 = 2
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define point P
def point_P (n : ℝ) : ℝ × ℝ := (-2, n)

theorem circle_tangent_line (n : ℝ) (h1 : n > 0) (h2 : circle_C (-2) n) :
  -- 1. The y-coordinate of P is 1
  n = 1 ∧
  -- 2. The equation of the tangent line to circle C passing through P is x - y + 3 = 0
  ∀ (x y : ℝ), (x - y + 3 = 0) ↔ (∃ (t : ℝ), x = -2 + t ∧ y = 1 + t ∧ circle_C x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l953_95321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l953_95329

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x ∈ Set.Icc 3 6}
def C : Set ℝ := {x | x < 5}

theorem set_operations :
  (A ∩ B : Set ℝ) = {3, 4, 5, 6} ∧
  (Aᶜ ∪ C : Set ℝ) = {x | x < 5 ∨ x ≥ 7} := by
  sorry

#check set_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l953_95329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersects_triangle_sides_once_l953_95386

-- Define the triangle and its properties
structure AcuteTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isAcute : Prop

-- Define the orthocenter, circumcenter, and circumradius
noncomputable def orthocenter (t : AcuteTriangle) : ℝ × ℝ := sorry
noncomputable def circumcenter (t : AcuteTriangle) : ℝ × ℝ := sorry
noncomputable def circumradius (t : AcuteTriangle) : ℝ := sorry

-- Define an ellipse
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  majorAxis : ℝ

-- Define a function to check if a point is on a line segment
def isOnLineSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop := sorry

-- Define a function to count intersections between an ellipse and a line segment
def countIntersections (e : Ellipse) (a b : ℝ × ℝ) : ℕ := sorry

-- The main theorem
theorem ellipse_intersects_triangle_sides_once 
  (t : AcuteTriangle) : 
  let M := orthocenter t
  let O := circumcenter t
  let r := circumradius t
  let e := Ellipse.mk O M r
  (countIntersections e t.A t.B = 1) ∧ 
  (countIntersections e t.B t.C = 1) ∧ 
  (countIntersections e t.C t.A = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersects_triangle_sides_once_l953_95386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l953_95325

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧
    ∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1/2) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l953_95325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_angles_l953_95320

theorem min_sum_of_angles (α β : Real) (t : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.tan α = 2/t →
  Real.tan β = t/15 →
  (∀ t' : Real, 10 * (2/t') + 3 * (t'/15) ≥ 10 * (2/t) + 3 * (t/15)) →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_angles_l953_95320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l953_95316

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation holds for given positive integers a and b -/
def equation_holds (a b : ℕ+) : Prop :=
  floor ((a : ℝ)^2 / b) + floor ((b : ℝ)^2 / a) = 
  floor (((a : ℝ)^2 + (b : ℝ)^2) / (a * b)) + a * b

/-- The theorem stating the characterization of solutions -/
theorem solution_characterization :
  ∀ a b : ℕ+, equation_holds a b ↔ 
  (∃ n : ℕ, (a = n ∧ b = n^2 + 1) ∨ (a = n^2 + 1 ∧ b = n)) :=
by
  sorry

#check solution_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l953_95316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l953_95392

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) 
  (h₁ : a₁ = 4) 
  (h₂ : arithmeticSum a₁ d 2 = 6) : 
  d = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l953_95392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l953_95311

/-- Represents a segment of the cyclist's journey -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def time_for_segment (s : Segment) : ℝ := s.distance / s.speed

/-- Calculates the average speed given total distance and total time -/
noncomputable def average_speed (total_distance total_time : ℝ) : ℝ := total_distance / total_time

theorem cyclist_average_speed :
  let segments : List Segment := [
    { distance := 5, speed := 8 },
    { distance := 3, speed := 6 },
    { distance := 9, speed := 14 },
    { distance := 12, speed := 11 }
  ]
  let total_distance := segments.foldl (fun acc s => acc + s.distance) 0
  let total_time := segments.foldl (fun acc s => acc + time_for_segment s) 0
  let avg_speed := average_speed total_distance total_time
  abs (avg_speed - 10.14) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclist_average_speed_l953_95311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cut_five_pieces_l953_95358

/-- Represents the time needed to cut a piece of wood into a given number of pieces -/
noncomputable def cutting_time (num_pieces : ℕ) (base_time : ℝ) (base_pieces : ℕ) : ℝ :=
  base_time * (num_pieces - 1) / (base_pieces - 1)

/-- Theorem stating that it takes 32 seconds to cut a piece of wood into 5 pieces,
    given that it takes 24 seconds to cut it into 4 pieces at a constant speed -/
theorem time_to_cut_five_pieces :
  cutting_time 5 24 4 = 32 := by
  -- Unfold the definition of cutting_time
  unfold cutting_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_cut_five_pieces_l953_95358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_proof_l953_95315

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (6 * x + Real.pi / 4)

noncomputable def transformed_function (x : ℝ) : ℝ := Real.sin (2 * x)

theorem symmetry_center_proof :
  let x₀ : ℝ := Real.pi / 2
  let y₀ : ℝ := 0
  (∀ (t : ℝ), transformed_function (x₀ + t) = transformed_function (x₀ - t)) ∧
  (transformed_function x₀ = y₀) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_proof_l953_95315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_xy_max_value_is_one_max_achieved_l953_95330

theorem max_product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 : ℝ)^x * (2 : ℝ)^y = 4) :
  ∀ (a b : ℝ), a > 0 → b > 0 → (2 : ℝ)^a * (2 : ℝ)^b = 4 → x * y ≥ a * b :=
by sorry

theorem max_value_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 : ℝ)^x * (2 : ℝ)^y = 4) :
  x * y ≤ 1 :=
by sorry

theorem max_achieved (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 : ℝ)^x * (2 : ℝ)^y = 4) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 : ℝ)^a * (2 : ℝ)^b = 4 ∧ a * b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_xy_max_value_is_one_max_achieved_l953_95330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l953_95385

-- Define the variables and conditions
variable (a : ℝ)
variable (ha : a > -38)

-- Define P and Q
noncomputable def P (a : ℝ) : ℝ := Real.sqrt (a + 41) - Real.sqrt (a + 40)
noncomputable def Q (a : ℝ) : ℝ := Real.sqrt (a + 39) - Real.sqrt (a + 38)

-- State the theorem
theorem P_less_than_Q (a : ℝ) (ha : a > -38) : P a < Q a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l953_95385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l953_95365

variable (a b : ℝ)
variable (x : ℝ)
variable (p : ℝ → ℝ)
variable (f : ℝ → ℝ)

theorem inequality_solution (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) (h3 : x > 0) (h4 : ∀ y, p y ≥ 0) :
  let f := λ x ↦ (1 / (a^2 - b^2)) * (a * Real.sin x - b * x^2 * Real.sin (1/x) + a * p x - b * x^2 * p (1/x))
  a * f x + b * x^2 * f (1/x) ≥ Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l953_95365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l953_95381

noncomputable def scores : List ℝ := [30, 26, 32, 27, 35]

noncomputable def mean (xs : List ℝ) : ℝ :=
  (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs) ^ 2)).sum / xs.length

theorem variance_of_scores :
  variance scores = 54 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l953_95381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l953_95394

def f (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i => 1 / ((i + 1 : ℕ) : ℚ) ^ 3) + 1

def g (n : ℕ+) : ℚ :=
  3 / 2 - 1 / (2 * (n : ℚ) ^ 2)

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l953_95394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_barycentric_coordinates_l953_95398

/-- Triangle PQR with side lengths p, q, r -/
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ

/-- Incenter of a triangle -/
noncomputable def incenter (t : Triangle) : ℝ × ℝ × ℝ :=
  (t.p / (t.p + t.q + t.r), t.q / (t.p + t.q + t.r), t.r / (t.p + t.q + t.r))

/-- The theorem to be proved -/
theorem incenter_barycentric_coordinates (t : Triangle) 
  (h1 : t.p = 6) (h2 : t.q = 8) (h3 : t.r = 3) : 
  incenter t = (6/17, 8/17, 3/17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_barycentric_coordinates_l953_95398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_price_theorem_l953_95318

noncomputable def price_mixture (price1 price2 : ℝ) (ratio : ℝ) : ℝ :=
  ((ratio * price1) + price2) / (ratio + 1)

theorem mixture_price_theorem (price1 price2 ratio : ℝ) 
  (h1 : price1 = 16)
  (h2 : price2 = 24)
  (h3 : ratio = 3) :
  price_mixture price1 price2 ratio = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_price_theorem_l953_95318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_10km_l953_95324

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialWorkers : ℕ
  completedLength : ℚ
  completedDays : ℕ
  additionalWorkers : ℕ

/-- Calculates the total length of the road given the project parameters -/
noncomputable def calculateRoadLength (project : RoadProject) : ℚ :=
  let workPerManDay := project.completedLength / (project.initialWorkers * project.completedDays)
  let remainingDays := project.totalDays - project.completedDays
  let totalWorkersAfterAdjustment := project.initialWorkers + project.additionalWorkers
  let remainingLength := (totalWorkersAfterAdjustment * workPerManDay * remainingDays)
  project.completedLength + remainingLength

/-- Theorem stating that the total length of the road is 10 km -/
theorem road_length_is_10km (project : RoadProject) 
    (h1 : project.totalDays = 15)
    (h2 : project.initialWorkers = 30)
    (h3 : project.completedLength = 2)
    (h4 : project.completedDays = 5)
    (h5 : project.additionalWorkers = 30) :
    calculateRoadLength project = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_10km_l953_95324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l953_95341

/-- Predicate to check if three lengths form a triangle. -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of a triangle given its side lengths. -/
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_inequality (a b c m_a m_b m_c : ℝ) 
  (h_triangle : IsTriangle a b c)
  (h_ma : m_a = 2 * (area a b c) / a)
  (h_mb : m_b = 2 * (area a b c) / b)
  (h_mc : m_c = 2 * (area a b c) / c) :
  (a / m_a)^2 + (b / m_b)^2 + (c / m_c)^2 ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l953_95341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l953_95371

/-- Represents a line in 2D space with equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  not_both_zero : A ≠ 0 ∨ B ≠ 0

theorem line_properties (l : Line) :
  (l.A * l.B ≠ 0 → (∃ x : ℝ, l.A * x + l.C = 0) ∧ (∃ y : ℝ, l.B * y + l.C = 0)) ∧
  (l.A = 0 → ∃ k : ℝ, ∀ x y : ℝ, l.B * y + l.C = 0 → y = k) ∧
  (l.B = 0 ∧ l.C = 0 → ∀ y : ℝ, l.A * 0 + l.B * y + l.C = 0) ∧
  (l.C = 0 → l.A * 0 + l.B * 0 + l.C = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l953_95371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_implies_ratio_l953_95303

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 0, 2]

theorem matrix_commutation_implies_ratio {x y z w : ℝ} 
  (h1 : A * !![x, y; z, w] = !![x, y; z, w] * A)
  (h2 : z ≠ 3 * y) :
  (x - w) / (z - 3 * y) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_commutation_implies_ratio_l953_95303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_iff_a_range_l953_95377

-- Define the curves
noncomputable def C₁ (x : ℝ) : ℝ := x^2
noncomputable def C₂ (a x : ℝ) : ℝ := Real.exp x / a

-- Define the derivatives of the curves
noncomputable def C₁' (x : ℝ) : ℝ := 2 * x
noncomputable def C₂' (a x : ℝ) : ℝ := Real.exp x / a

-- Define the condition for a common tangent
def has_common_tangent (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, C₁' x₁ = C₂' a x₂ ∧ C₁ x₁ - C₂ a x₂ = C₁' x₁ * (x₁ - x₂)

-- State the theorem
theorem common_tangent_iff_a_range (a : ℝ) :
  a > 0 → (has_common_tangent a ↔ a ≥ Real.exp 2 / 4) := by
  sorry

#check common_tangent_iff_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_iff_a_range_l953_95377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_specific_remainders_l953_95395

theorem count_integers_with_specific_remainders : 
  let S : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 10000) (Finset.range 10001)
  let P : Finset ℕ := Finset.filter (λ n => n % 3 = 2 ∧ n % 5 = 3 ∧ n % 7 = 4) (Finset.range 10001)
  (S ∩ P).card = 95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_specific_remainders_l953_95395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_implies_n_equals_two_l953_95360

noncomputable def Quadrilateral (n : ℕ) : List (ℝ × ℝ) := 
  [(n, Real.exp (n : ℝ)), (n + 2, Real.exp ((n + 2) : ℝ)), 
   (n + 4, Real.exp ((n + 4) : ℝ)), (n + 6, Real.exp ((n + 6) : ℝ))]

noncomputable def QuadrilateralArea (n : ℕ) : ℝ :=
  let vertices := Quadrilateral n
  abs (
    vertices[0]!.1 * vertices[1]!.2 + 
    vertices[1]!.1 * vertices[2]!.2 + 
    vertices[2]!.1 * vertices[3]!.2 + 
    vertices[3]!.1 * vertices[0]!.2 - 
    vertices[1]!.1 * vertices[0]!.2 - 
    vertices[2]!.1 * vertices[1]!.2 - 
    vertices[3]!.1 * vertices[2]!.2 - 
    vertices[0]!.1 * vertices[3]!.2
  ) / 2

theorem quadrilateral_area_implies_n_equals_two (n : ℕ) :
  n > 0 → QuadrilateralArea n = Real.exp 6 - Real.exp 2 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_implies_n_equals_two_l953_95360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_S_is_infinite_l953_95388

theorem set_S_is_infinite (S : Set ℕ) 
  (h : ∀ a, a ∈ S → ∃ b c, b ∈ S ∧ c ∈ S ∧ a = (b * (3 * c - 5)) / 15) : 
  Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_S_is_infinite_l953_95388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_2_solution_set_a_pos_solution_set_a_eq_0_solution_set_a_neg_l953_95301

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 + 2*x + 1 - a^2 ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | inequality x a}

-- Theorem for a = 2
theorem solution_set_a_eq_2 :
  solution_set 2 = Set.Icc (-3) 1 := by sorry

-- Theorem for a > 0
theorem solution_set_a_pos (a : ℝ) (h : a > 0) :
  solution_set a = Set.Icc (-1 - a) (-1 + a) := by sorry

-- Theorem for a = 0
theorem solution_set_a_eq_0 :
  solution_set 0 = {-1} := by sorry

-- Theorem for a < 0
theorem solution_set_a_neg (a : ℝ) (h : a < 0) :
  solution_set a = Set.Icc (-1 + a) (-1 - a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_a_eq_2_solution_set_a_pos_solution_set_a_eq_0_solution_set_a_neg_l953_95301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_2_sufficient_not_necessary_l953_95359

-- Define the binomial expansion of (x-a)^6
def binomial_expansion (x a : ℝ) : ℕ → ℝ
  | 0 => x^6
  | 1 => 6*x^5*(-a)
  | 2 => 15*x^4*a^2
  | 3 => 20*x^3*(-a)^3
  | 4 => 15*x^2*a^4
  | 5 => 6*x*(-a)^5
  | 6 => a^6
  | _ => 0

-- Define the condition for the third term
def third_term_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, binomial_expansion x a 2 = 60*x^4

-- Theorem statement
theorem a_equals_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → third_term_condition a) ∧
  ¬(∀ a : ℝ, third_term_condition a → a = 2) :=
by
  constructor
  · intro a h_a
    intro x
    simp [third_term_condition, binomial_expansion]
    rw [h_a]
    ring
  · push_neg
    use -2
    constructor
    · intro x
      simp [third_term_condition, binomial_expansion]
      ring
    · intro h
      exact absurd h (by norm_num)

#check a_equals_2_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_2_sufficient_not_necessary_l953_95359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l953_95319

-- Define the points A, B, and D
def A : ℝ × ℝ := (8, 7)
def B : ℝ × ℝ := (-1, 2)
def D : ℝ × ℝ := (-4, 5)

-- Define the function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the function to check if a point is on the altitude from A
def isOnAltitude (p : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (B.1 - A.1) + (p.2 - A.2) * (B.2 - A.2) = 0

-- State the theorem
theorem point_C_coordinates :
  ∃ C : ℝ × ℝ,
    distance A B = distance A C ∧
    isOnAltitude D ∧
    C = (-7, 8) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l953_95319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_in_second_quadrant_l953_95333

theorem sine_value_in_second_quadrant (α : Real) :
  (π / 2 < α) ∧ (α < π) →  -- α is in the second quadrant
  Real.sin (π / 2 - α) = -1 / 3 →
  Real.sin α = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_value_in_second_quadrant_l953_95333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_sin_plus_sqrt3_cos_l953_95393

/-- The function f(x) = sin(x) + √3 * cos(x) attains its extreme values when x = π/6 + kπ, where k is an integer. -/
theorem extreme_values_of_sin_plus_sqrt3_cos (x : ℝ) :
  ∃ (k : ℤ), (∀ y, |Real.sin y + Real.sqrt 3 * Real.cos y| ≤ |Real.sin x + Real.sqrt 3 * Real.cos x|) ↔ 
  x = π / 6 + k * π := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_of_sin_plus_sqrt3_cos_l953_95393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_equals_twelve_l953_95373

noncomputable def x : ℝ := Real.sqrt (12 - 3 * Real.sqrt 7) - Real.sqrt (12 + 3 * Real.sqrt 7)
noncomputable def y : ℝ := Real.sqrt (7 - 4 * Real.sqrt 3) - Real.sqrt (7 + 4 * Real.sqrt 3)
noncomputable def z : ℝ := Real.sqrt (2 + Real.sqrt 3) - Real.sqrt (2 - Real.sqrt 3)

theorem xyz_equals_twelve : x * y * z = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_equals_twelve_l953_95373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l953_95380

open Real

noncomputable def f (x : ℝ) : ℝ := 3*x + sin x

theorem range_of_x (h1 : ∀ x ∈ Set.Ioo (-1) 1, (deriv f x) = 3 + cos x)
                   (h2 : f 0 = 0)
                   (h3 : ∀ x, f (1 - x) + f (1 - x^2) < 0) :
  ∃ x, x ∈ Set.Ioo 1 (Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l953_95380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_l953_95356

/-- Represents the daily temperatures in Orlando for a month plus three days -/
def temperatures : List ℚ := [
  55, 62, 58, 65, 54, 60, 56,
  70, 74, 71, 77, 64, 68, 72,
  82, 85, 89, 73, 65, 63, 67,
  75, 72, 60, 57, 50, 55, 58,
  69, 67, 70
]

/-- The number of days for which we have temperature data -/
def numDays : ℕ := 31

/-- Theorem: The average temperature is equal to the sum of all temperatures divided by the number of days -/
theorem average_temperature : 
  (temperatures.sum / numDays : ℚ) = (temperatures.sum / numDays) := by
  rfl

/-- Compute the average temperature -/
def computeAverage : ℚ := temperatures.sum / numDays

#eval computeAverage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_temperature_l953_95356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l953_95335

def A : Set ℝ := {x : ℝ | x > -2}
def B : Set ℝ := {x : ℝ | x > 2}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo (-2) 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l953_95335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l953_95397

-- Define the function f(x) = 4cos(x) - 1
noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x - 1

-- State the theorem
theorem f_min_max :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    -1 ≤ f x ∧ f x ≤ 3 ∧
    (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₂ = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l953_95397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_tin_height_l953_95344

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The height of a cylinder given its volume and radius -/
noncomputable def cylinderHeight (v r : ℝ) : ℝ := v / (Real.pi * r^2)

theorem cylinder_tin_height :
  let diameter := (14 : ℝ)
  let volume := (245 : ℝ)
  let radius := diameter / 2
  let height := cylinderHeight volume radius
  ∃ ε > 0, |height - 1.59155| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_tin_height_l953_95344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fencing_cost_is_810_l953_95322

/-- Represents a rectangular flowerbed with width and length -/
structure Flowerbed where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a flowerbed -/
def perimeter (f : Flowerbed) : ℝ := 2 * (f.width + f.length)

/-- Calculates the cost of fencing a flowerbed with a given price per meter -/
def fencingCost (f : Flowerbed) (pricePerMeter : ℝ) : ℝ := perimeter f * pricePerMeter

/-- Theorem: The minimum cost of fencing three flowerbeds with given dimensions and fencing options is $810 -/
theorem min_fencing_cost_is_810 
  (flowerbed1 : Flowerbed)
  (flowerbed2 : Flowerbed)
  (flowerbed3 : Flowerbed)
  (woodenFencePrice : ℝ)
  (metalFencePrice : ℝ)
  (bambooFencePrice : ℝ)
  (h1 : flowerbed1.width = 4)
  (h2 : flowerbed1.length = 2 * flowerbed1.width - 1)
  (h3 : flowerbed2.length = flowerbed1.length + 3)
  (h4 : flowerbed2.width = flowerbed1.width - 2)
  (h5 : flowerbed3.width = (flowerbed1.width + flowerbed2.width) / 2)
  (h6 : flowerbed3.length = (flowerbed1.length + flowerbed2.length) / 2)
  (h7 : woodenFencePrice = 10)
  (h8 : metalFencePrice = 15)
  (h9 : bambooFencePrice = 12)
  : min (fencingCost flowerbed1 woodenFencePrice + fencingCost flowerbed2 metalFencePrice + fencingCost flowerbed3 woodenFencePrice)
        (fencingCost flowerbed1 woodenFencePrice + fencingCost flowerbed2 metalFencePrice + fencingCost flowerbed3 bambooFencePrice) = 810 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_fencing_cost_is_810_l953_95322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_6_l953_95357

-- Define the line ℓ
def line_ℓ (k : ℝ) (x y : ℝ) : Prop := k * x + y + 4 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 6 = 0

-- Define the line m
def line_m (k : ℝ) (x y : ℝ) : Prop := y = x + k

-- Define the axis of symmetry property
def is_axis_of_symmetry (k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_C x y → line_ℓ k (-x) y

-- Define the chord length function
noncomputable def chord_length (k : ℝ) : ℝ :=
  let center_x := -2
  let center_y := 2
  let d := |center_x - center_y + k| / Real.sqrt 2
  2 * Real.sqrt (4 - d^2)

-- Main theorem
theorem chord_length_is_sqrt_6 :
  ∃ k : ℝ, is_axis_of_symmetry k ∧ chord_length k = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_6_l953_95357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_line_prime_tangent_to_parabola_l953_95342

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the circle (renamed to avoid conflict with built-in circle)
def problem_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 8

-- Define the point P on the y-axis
def point_P (m : ℝ) : ℝ × ℝ := (0, m)

-- Define the line l'
def line_l_prime (m : ℝ) (x y : ℝ) : Prop := y = -x - m

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := x^2 = 4 * y

theorem circle_tangent_to_line (m : ℝ) :
  let P := point_P m
  line_l m P.1 P.2 ∧ problem_circle P.1 P.2 := by sorry

theorem line_prime_tangent_to_parabola (m : ℝ) :
  (∃ x y, line_l_prime m x y ∧ parabola_C x y ∧
    (∀ x' y', line_l_prime m x' y' → parabola_C x' y' → (x', y') = (x, y))) ↔
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_line_prime_tangent_to_parabola_l953_95342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l953_95338

-- Define the functions f and g
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m^2/2 + 2*m - 3

-- Part 1
theorem part_one (m : ℝ) (a : ℝ) :
  (∀ x, g m x < m^2/2 + 1 ↔ (1 < x ∧ x < a)) →
  a = 2 := by
  sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 0 1, ∃ x₂ ∈ Set.Icc 1 2, f m x₁ > g m x₂) →
  -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l953_95338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_factor_of_10_factorial_l953_95349

theorem largest_square_factor_of_10_factorial :
  ∀ k : ℕ, k^2 ∣ Nat.factorial 10 → k ≤ 720 ∧ 720^2 ∣ Nat.factorial 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_factor_of_10_factorial_l953_95349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l953_95387

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := (x - 1)^2 / (x - 5)^2

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x < 5 ∨ x > 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l953_95387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_change_of_basis_l953_95382

def A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 2; 3, -1, 0; 1, 1, -2]

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, -1, 1; -1, 1, -2; -1, 2, 1]

def A' : Matrix (Fin 3) (Fin 3) ℝ := !![9, -1, 27; 5, -1, 14; -3, 1, -8]

theorem transformation_matrix_change_of_basis :
  IsUnit (Matrix.det B) →
  A' = B⁻¹ * A * B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_change_of_basis_l953_95382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l953_95353

noncomputable section

def point := ℝ × ℝ

def reflect_over_y_axis (p : point) : point :=
  (-p.1, p.2)

def reflect_over_y_eq_neg_x (p : point) : point :=
  (-p.2, -p.1)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def triangle_area (a b c : point) : ℝ :=
  let s := (distance a b + distance b c + distance c a) / 2
  Real.sqrt (s * (s - distance a b) * (s - distance b c) * (s - distance c a))

theorem triangle_abc_area :
  let a : point := (3, 4)
  let b : point := reflect_over_y_axis a
  let c : point := reflect_over_y_eq_neg_x b
  triangle_area a b c = 35 * Real.sqrt 2 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l953_95353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_theorem_l953_95328

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define membership for points in a circle
instance : Membership (ℝ × ℝ) Circle where
  mem p c := dist p c.center = c.radius

-- Define the given conditions
variable (S₁ S₂ : Circle)
variable (A : ℝ × ℝ)

-- A is a common point of S₁ and S₂
axiom A_on_S₁ : A ∈ S₁
axiom A_on_S₂ : A ∈ S₂

-- S is the reflection of S₁ through point A
noncomputable def S : Circle := sorry

-- B is the intersection point of S and S₂ (different from A)
variable (B : ℝ × ℝ)
axiom B_on_S : B ∈ S
axiom B_on_S₂ : B ∈ S₂
axiom B_ne_A : B ≠ A

-- Define a line through two points
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define a chord of a circle
noncomputable def chord (c : Circle) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- The theorem to prove
theorem equal_chords_theorem :
  ∃ l : Set (ℝ × ℝ), A ∈ l ∧ chord S₁ l = chord S₂ l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_chords_theorem_l953_95328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_angle_l953_95368

/-- Represents that a line is tangent to a circle at a point. -/
def IsTangent (O A B : Point) : Prop := sorry

/-- Represents a triangle formed by three tangents to a circle. -/
structure TangentTriangle (O : Point) (Q C D : Point) : Prop where
  is_tangent_QC : IsTangent O Q C
  is_tangent_QD : IsTangent O Q D
  is_tangent_CD : IsTangent O C D

/-- Represents the measure of an angle in degrees. -/
def AngleMeasure (A B C : Point) : ℝ := sorry

/-- Given a circle O and a triangle QCD formed by three tangents to the circle,
    if ∠CQD = 50°, then ∠COD = 65°. -/
theorem tangent_triangle_angle (O : Point) (Q C D : Point) :
  TangentTriangle O Q C D →
  AngleMeasure C Q D = 50 →
  AngleMeasure C O D = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_angle_l953_95368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l953_95399

/-- Calculates the time (in seconds) required for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_crossing_time :
  time_to_cross_bridge 145 45 230 = 30 := by
  -- Unfold the definition of time_to_cross_bridge
  unfold time_to_cross_bridge
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l953_95399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_l953_95336

-- Define the pizza and pepperoni properties
noncomputable def pizza_diameter : ℝ := 16
def pepperoni_count : ℕ := 32
def pepperoni_across_diameter : ℕ := 8

-- Calculate the pizza radius
noncomputable def pizza_radius : ℝ := pizza_diameter / 2

-- Calculate the pepperoni radius
noncomputable def pepperoni_radius : ℝ := pizza_diameter / (2 * pepperoni_across_diameter)

-- Define the theorem
theorem pepperoni_coverage :
  (pepperoni_count * Real.pi * pepperoni_radius^2) / (Real.pi * pizza_radius^2) = 1 / 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pepperoni_coverage_l953_95336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l953_95362

-- Define the train's length in meters
noncomputable def train_length : ℝ := 50

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 144

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 5 / 18

-- Define the time taken to cross the pole in seconds
noncomputable def crossing_time : ℝ := 1.25

-- Theorem statement
theorem train_crossing_time :
  crossing_time = train_length / (train_speed_kmh * kmh_to_ms) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l953_95362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_races_for_top3_correct_l953_95363

/-- Represents a horse -/
structure Horse where
  id : Nat

/-- Represents race conditions -/
structure RaceConditions where
  -- Add relevant fields

/-- Represents track layout -/
structure TrackLayout where
  -- Add relevant fields

/-- Represents a race with varying conditions and asymmetric layouts -/
structure Race where
  horses : Finset Horse
  conditions : RaceConditions
  layout : TrackLayout

/-- Represents the result of a race -/
structure RaceResult where
  race : Race
  rankings : List Horse

/-- A strategy for determining the top 3 fastest horses -/
structure RacingStrategy where
  races : List Race
  determineTop3 : List RaceResult → Option (Horse × Horse × Horse)

/-- The minimum number of races needed to determine the top 3 fastest horses -/
def minRacesForTop3 (totalHorses : Nat) (maxHorsesPerRace : Nat) : Nat :=
  7

theorem min_races_for_top3_correct :
  ∀ (s : RacingStrategy),
    s.races.length ≥ minRacesForTop3 30 6 →
    (∀ (results : List RaceResult),
      results.length = s.races.length →
      (∀ (r : RaceResult), r ∈ results → r.race ∈ s.races) →
      (∀ (race : Race), race ∈ s.races → race.horses.card ≤ 6) →
      (∃ (top3 : Horse × Horse × Horse), s.determineTop3 results = some top3)) :=
by
  sorry

#check min_races_for_top3_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_races_for_top3_correct_l953_95363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l953_95364

theorem angle_difference (α β : Real) : 
  0 < α → α < π/2 → 0 < β → β < π/2 →
  Real.cos α = 2 * Real.sqrt 5 / 5 →
  Real.sin β = 3 * Real.sqrt 10 / 10 →
  α - β = -π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_difference_l953_95364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l953_95369

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_sqrt3_over_2_l953_95369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l953_95389

/-- Represents the marathon distance in kilometers -/
noncomputable def marathon_distance : ℝ := 40

/-- Represents Jill's marathon time in hours -/
noncomputable def jill_time : ℝ := 4.0

/-- Represents the ratio of Jack's speed to Jill's speed -/
noncomputable def speed_ratio : ℝ := 0.888888888888889

/-- Calculates Jack's marathon time given the marathon distance, Jill's time, and the speed ratio -/
noncomputable def jack_time (d : ℝ) (t : ℝ) (r : ℝ) : ℝ := d / (r * (d / t))

/-- Theorem stating that Jack's marathon time is 4.5 hours -/
theorem jack_marathon_time :
  jack_time marathon_distance jill_time speed_ratio = 4.5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_marathon_time_l953_95389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_value_l953_95372

noncomputable def semi_annual_rate : ℝ := 0.02
noncomputable def quarterly_rate : ℝ := 0.04
noncomputable def investment_period : ℝ := 1
noncomputable def first_investment : ℝ := 80804

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

theorem second_investment_value :
  ∃ (second_investment : ℝ),
    compound_interest first_investment semi_annual_rate 2 investment_period =
    compound_interest second_investment quarterly_rate 4 investment_period ∧
    second_investment = 79200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_investment_value_l953_95372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l953_95347

theorem trigonometric_identities 
  (α β : ℝ)
  (h1 : Real.cos α = 5/13)
  (h2 : Real.cos (α - β) = 4/5)
  (h3 : 0 < β)
  (h4 : β < α)
  (h5 : α < π/2) :
  Real.tan (2*α) = -120/119 ∧ Real.cos β = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l953_95347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_result_l953_95300

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l in parametric form
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * t, -1 + t)

-- Define point P in polar coordinates
def point_P : ℝ × ℝ := (1, 3 * Real.pi / 2)

-- Define the function to calculate the result
def calculate_result (A B : ℝ × ℝ) : ℝ :=
  let PA := Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)
  let PB := Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)
  (PA + 1) * (PB + 1)

-- Theorem statement
theorem intersection_result :
  ∃ (A B : ℝ × ℝ),
    (∀ θ, curve_C θ = Real.sqrt ((A.1)^2 + (A.2)^2)) ∧
    (∃ t, line_l t = A) ∧
    (∀ θ, curve_C θ = Real.sqrt ((B.1)^2 + (B.2)^2)) ∧
    (∃ t, line_l t = B) ∧
    calculate_result A B = 3 + Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_result_l953_95300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_after_transformation_l953_95340

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_value_after_transformation (φ : ℝ) 
  (h : abs φ < Real.pi / 2) :
  ∃ (g : ℝ → ℝ), 
    (∀ x, g x = f (x - Real.pi / 12) (-x))
    ∧ (∀ x ∈ Set.Icc 0 (Real.pi / 2), g x ≥ -Real.sqrt 3 / 2)
    ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), g x = -Real.sqrt 3 / 2) := by
  sorry

#check min_value_after_transformation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_after_transformation_l953_95340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_numenor_population_thrice_gondor_l953_95390

/-- The average life expectancy in Gondor -/
def gondor_life_expectancy : ℝ := 64

/-- The average life expectancy in Numenor -/
def numenor_life_expectancy : ℝ := 92

/-- The average life expectancy in both countries combined -/
def combined_life_expectancy : ℝ := 85

/-- Theorem stating that the population of Numenor is 3 times the population of Gondor -/
theorem numenor_population_thrice_gondor (gondor_population numenor_population : ℝ) :
  (gondor_life_expectancy * gondor_population + numenor_life_expectancy * numenor_population) / (gondor_population + numenor_population) = combined_life_expectancy →
  numenor_population = 3 * gondor_population :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_numenor_population_thrice_gondor_l953_95390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_is_eight_l953_95312

/-- The smallest number of 250 mL bottles needed to buy at least 60 fluid ounces of milk -/
def min_bottles : ℕ :=
  let fl_oz_per_liter : ℚ := 33.8
  let ml_per_bottle : ℚ := 250
  let min_fl_oz : ℚ := 60
  (((min_fl_oz / fl_oz_per_liter) * 1000 / ml_per_bottle).ceil.toNat)

theorem min_bottles_is_eight : min_bottles = 8 := by
  sorry

#eval min_bottles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_is_eight_l953_95312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_divides_m_solve_equation_l953_95314

theorem n_divides_m (m n : ℕ) (h1 : m > n) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := by
  sorry

theorem solve_equation (m n : ℕ) (h1 : m > n) (h2 : m > 0) (h3 : n > 0)
  (h4 : Nat.gcd m n + Nat.lcm m n = m + n) (h5 : m - n = 10) :
  (m = 11 ∧ n = 1) ∨ (m = 12 ∧ n = 2) ∨ (m = 15 ∧ n = 5) ∨ (m = 20 ∧ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_divides_m_solve_equation_l953_95314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_radius_reduces_difference_l953_95304

/-- Represents a circle on a chessboard -/
structure ChessboardCircle where
  center : ℝ × ℝ  -- Center coordinates
  radius : ℝ      -- Radius of the circle

/-- Calculates the total arc length on light squares -/
noncomputable def lightArcLength (c : ChessboardCircle) : ℝ := sorry

/-- Calculates the total arc length on dark squares -/
noncomputable def darkArcLength (c : ChessboardCircle) : ℝ := sorry

/-- Counts the number of intersected squares -/
def intersectedSquares (c : ChessboardCircle) : ℕ := sorry

/-- Theorem stating that decreasing the radius reduces the difference in arc lengths -/
theorem decrease_radius_reduces_difference
  (c : ChessboardCircle)
  (h1 : c.center.1 ∈ Set.Ioo 0 8 ∧ c.center.2 ∈ Set.Ioo 0 8)  -- Center is in an interior square
  (h2 : c.radius = 1.9)                                      -- Initial radius is 1.9
  (h3 : |lightArcLength c - darkArcLength c| < 0.01)         -- Arc lengths are approximately equal
  (ε : ℝ)
  (h4 : 0 < ε)
  (h5 : ε < 0.1)
  : ∃ (c' : ChessboardCircle),
    c'.center = c.center ∧
    c'.radius = c.radius - ε ∧
    intersectedSquares c' = intersectedSquares c ∧
    |lightArcLength c' - darkArcLength c'| < |lightArcLength c - darkArcLength c| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decrease_radius_reduces_difference_l953_95304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l953_95331

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if three real numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem ellipse_equation (e : Ellipse) (f1 f2 : Point) (p : Point) :
  -- The ellipse has its center at the origin
  e.a > 0 ∧ e.b > 0 ∧ e.a > e.b →
  -- The foci F₁ and F₂ are on the x-axis
  f1.y = 0 ∧ f2.y = 0 →
  -- Point P(2, √3) lies on the ellipse
  p.x = 2 ∧ p.y = Real.sqrt 3 ∧ pointOnEllipse p e →
  -- |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence
  isArithmeticSequence (distance p f1) (distance f1 f2) (distance p f2) →
  -- The equation of the ellipse is x²/8 + y²/6 = 1
  e.a^2 = 8 ∧ e.b^2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l953_95331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_l953_95332

theorem tan_sum_of_roots (α β : ℝ) : 
  (2 * (Real.tan α)^2 + 3 * (Real.tan α) - 7 = 0) → 
  (2 * (Real.tan β)^2 + 3 * (Real.tan β) - 7 = 0) → 
  Real.tan α ≠ Real.tan β →
  Real.tan (α + β) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_of_roots_l953_95332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l953_95354

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def ω₁ : Circle := { center := (0, 0), radius := 961 }
def ω₂ : Circle := { center := (672, 0), radius := 625 }
noncomputable def ω : Circle := { center := (0, 0), radius := 0 }  -- Placeholder values

-- Define the points of intersection and where AB intersects ω
noncomputable def A : ℝ × ℝ := (0, 0)  -- Placeholder values
noncomputable def B : ℝ × ℝ := (0, 0)  -- Placeholder values
noncomputable def P : ℝ × ℝ := (0, 0)  -- Placeholder values
noncomputable def Q : ℝ × ℝ := (0, 0)  -- Placeholder values

-- State the theorem
theorem distance_between_centers : 
  -- ω₁ and ω₂ intersect at distinct points A and B
  A ≠ B ∧
  -- ω is externally tangent to both ω₁ and ω₂
  (∀ p : ℝ × ℝ, (p.1 - ω.center.1)^2 + (p.2 - ω.center.2)^2 = ω.radius^2 →
    ((p.1 - ω₁.center.1)^2 + (p.2 - ω₁.center.2)^2 = (ω₁.radius + ω.radius)^2 ∨
     (p.1 - ω₂.center.1)^2 + (p.2 - ω₂.center.2)^2 = (ω₂.radius + ω.radius)^2)) ∧
  -- Line AB intersects ω at points P and Q
  (∃ t₁ t₂ : ℝ, 
    P = (A.1 + t₁ * (B.1 - A.1), A.2 + t₁ * (B.2 - A.2)) ∧
    Q = (A.1 + t₂ * (B.1 - A.1), A.2 + t₂ * (B.2 - A.2)) ∧
    (P.1 - ω.center.1)^2 + (P.2 - ω.center.2)^2 = ω.radius^2 ∧
    (Q.1 - ω.center.1)^2 + (Q.2 - ω.center.2)^2 = ω.radius^2) ∧
  -- The measure of minor arc PQ on ω is 120°
  Real.cos (120 * π / 180) = 
    ((P.1 - ω.center.1) * (Q.1 - ω.center.1) + (P.2 - ω.center.2) * (Q.2 - ω.center.2)) / 
    (ω.radius^2) →
  -- The distance between the centers of ω₁ and ω₂ is 672
  (ω₂.center.1 - ω₁.center.1)^2 + (ω₂.center.2 - ω₁.center.2)^2 = 672^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_centers_l953_95354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_odd_iff_a_eq_one_l953_95326

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Statement 1: f is monotonically increasing
theorem f_monotone_increasing (a : ℝ) : 
  ∀ x y : ℝ, x < y → f a x < f a y := by sorry

-- Statement 2: f is an odd function iff a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_odd_iff_a_eq_one_l953_95326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_earnings_l953_95361

/-- Represents the different tasks Jerry can perform --/
inductive TaskType
  | A
  | B
  | C

/-- Returns the rate for a given task --/
def taskRate (t : TaskType) : ℕ :=
  match t with
  | TaskType.A => 40
  | TaskType.B => 50
  | TaskType.C => 60

/-- Represents Jerry's work schedule for a day --/
structure DaySchedule where
  hours : ℕ
  tasks : List TaskType

/-- Represents Jerry's work schedule for the week --/
def weekSchedule : List DaySchedule := [
  ⟨10, [TaskType.A, TaskType.B]⟩,
  ⟨8, [TaskType.A, TaskType.C]⟩,
  ⟨6, [TaskType.B]⟩,
  ⟨12, [TaskType.A, TaskType.B, TaskType.C]⟩,
  ⟨4, [TaskType.C]⟩,
  ⟨10, [TaskType.A, TaskType.B]⟩,
  ⟨0, []⟩
]

/-- Calculates the total earnings for the week --/
def totalEarnings (schedule : List DaySchedule) : ℕ :=
  sorry

theorem jerry_earnings : totalEarnings weekSchedule = 1220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_earnings_l953_95361
