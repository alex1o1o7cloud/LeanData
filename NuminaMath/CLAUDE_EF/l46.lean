import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_congruence_l46_4667

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  100 ≤ n ∧ n < 1000 ∧ (75 * n) % 450 = 300 →
  n ≤ 994 ∧
  (75 * 994) % 450 = 300 ∧
  994 < 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_congruence_l46_4667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_application_methods_l46_4692

theorem application_methods (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  Fintype.card (Fin n → Fin m) = m ^ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_application_methods_l46_4692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_satisfies_conditions_l46_4696

/-- Given two lines in 2D space and points on these lines, prove that the projection
    of the vector between these points onto a specific vector satisfies given conditions. -/
theorem projection_satisfies_conditions :
  -- Define the parametric equations for lines l and m
  let l : ℝ → ℝ × ℝ := λ t ↦ (2 + 2*t, 3 + t)
  let m : ℝ → ℝ × ℝ := λ s ↦ (-3 + 2*s, 5 + s)
  
  -- Define points A and B on lines l and m respectively
  let A : ℝ × ℝ := l 1
  let B : ℝ × ℝ := m 2
  
  -- Define the vector v
  let v : ℝ × ℝ := (3, -6)
  
  -- State the conditions
  (v.1 + v.2 = 3) →
  
  -- The conclusion we want to prove
  ∃ (P : ℝ × ℝ), ∃ (t : ℝ), P = m t ∧
  (∃ (k : ℝ), (A.1 - P.1, A.2 - P.2) = (k * v.1, k * v.2)) ∧
  ((A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_satisfies_conditions_l46_4696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_cos_value_when_f_eq_eight_fifths_l46_4674

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (1 + Real.sin (2 * x)) * 1 + (Real.sin x - Real.cos x) * (Real.sin x + Real.cos x)

-- Statement 1: Maximum value of f
theorem f_max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = Real.sqrt 2 + 1 := by
  sorry

-- Statement 2: Value of cos(2(π/4 - 2θ)) when f(θ) = 8/5
theorem cos_value_when_f_eq_eight_fifths (θ : ℝ) (h : f θ = 8/5) :
  Real.cos (2 * (Real.pi/4 - 2*θ)) = 16/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_cos_value_when_f_eq_eight_fifths_l46_4674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_valid_arrangement_l46_4676

/-- A type representing the possible positions in the arrangement --/
inductive Position
| first | second | third | fourth | fifth

/-- A function type representing an arrangement of the numbers --/
def Arrangement := Position → Int

/-- The set of numbers to be arranged --/
def numbers : Finset Int := {-3, 1, 7, 8, 14}

/-- The largest number in the set --/
def largest : Int := 14

/-- The smallest number in the set --/
def smallest : Int := -3

/-- The median of the set --/
def median : Int := 7

/-- Predicate to check if an arrangement is valid according to the rules --/
def isValidArrangement (arr : Arrangement) : Prop :=
  (arr Position.first ≠ largest ∧ arr Position.second ≠ largest) ∧
  (arr Position.fifth ≠ smallest ∧ arr Position.fourth ≠ smallest) ∧
  (arr Position.first ≠ median ∧ arr Position.fifth ≠ median) ∧
  (∀ n : Int, n ∈ numbers ↔ ∃ p : Position, arr p = n)

/-- Theorem stating that for any valid arrangement, the average of the first and last numbers is 5.5 --/
theorem average_of_valid_arrangement (arr : Arrangement) (h : isValidArrangement arr) :
  (arr Position.first + arr Position.fifth : ℚ) / 2 = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_valid_arrangement_l46_4676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_ending_in_2_l46_4622

/-- The sequence of integers between 100 and 600 that end in 2 -/
def intSequence : List Nat := List.range 50 |>.map (λ n => 102 + 10 * n)

/-- The sum of the sequence -/
def sequenceSum : Nat := intSequence.sum

theorem sum_of_integers_ending_in_2 : sequenceSum = 17350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_integers_ending_in_2_l46_4622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l46_4604

open Real
open BigOperators
open Finset

-- Define the quadrilateral ABCD and point P
variable (A B C D P : ℝ × ℝ)

-- Define the vector from P to each vertex
def PA (A P : ℝ × ℝ) : ℝ × ℝ := (A.1 - P.1, A.2 - P.2)
def PB (B P : ℝ × ℝ) : ℝ × ℝ := (B.1 - P.1, B.2 - P.2)
def PC (C P : ℝ × ℝ) : ℝ × ℝ := (C.1 - P.1, C.2 - P.2)
def PD (D P : ℝ × ℝ) : ℝ × ℝ := (D.1 - P.1, D.2 - P.2)

-- Define the condition that P is inside ABCD
def P_inside (A B C D P : ℝ × ℝ) : Prop :=
  ∃ (α β γ δ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧
  α + β + γ + δ = 1 ∧
  P = (α * A.1 + β * B.1 + γ * C.1 + δ * D.1,
       α * A.2 + β * B.2 + γ * C.2 + δ * D.2)

-- Define the vector equation
def vector_equation (A B C D P : ℝ × ℝ) : Prop :=
  PA A P + 3 • PB B P + 2 • PC C P + 4 • PD D P = (0, 0)

-- Define the area of a triangle
noncomputable def triangle_area (X Y Z : ℝ × ℝ) : ℝ :=
  abs ((X.1 - Z.1) * (Y.2 - Z.2) - (Y.1 - Z.1) * (X.2 - Z.2)) / 2

-- Define the area of a quadrilateral as the sum of two triangles
noncomputable def quadrilateral_area (A B C D : ℝ × ℝ) : ℝ :=
  triangle_area A B C + triangle_area A C D

-- State the theorem
theorem area_ratio_theorem (A B C D P : ℝ × ℝ) 
  (h_inside : P_inside A B C D P) 
  (h_vector : vector_equation A B C D P) :
  quadrilateral_area A B C D / triangle_area A P D = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l46_4604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_difference_l46_4654

/-- Given a rectangle with points (5, 5), (9, 2), (a, 13), and (15, b),
    prove that a - b = 1 -/
theorem rectangle_point_difference (a b : ℝ) : 
  (∃ (rect : Set (ℝ × ℝ)), 
    (5, 5) ∈ rect ∧ 
    (9, 2) ∈ rect ∧ 
    (a, 13) ∈ rect ∧ 
    (15, b) ∈ rect) → 
  a - b = 1 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_point_difference_l46_4654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_perpendicular_parallel_l46_4687

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations as variables
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (line_in : Line → Plane → Prop)

-- Notation
local notation:50 a " ⟂ " b => perpendicular a b
local notation:50 a " ∥ " b => parallel a b
local notation:50 l " ⟂ " p => line_perpendicular l p
local notation:50 l " ∥ " p => line_parallel l p
local notation:50 l " ⊆ " p => line_in l p

-- Theorem statement
theorem plane_line_perpendicular_parallel 
  (α β γ : Plane) (l : Line) : 
  ((line_perpendicular l α ∧ line_parallel l β) → perpendicular α β) ∧
  ((parallel α β ∧ ¬(line_in l β) ∧ line_parallel l α) → line_parallel l β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_line_perpendicular_parallel_l46_4687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l46_4677

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 5 * Real.cos x ^ 2 + 3 * Real.cos x + 2 * Real.sin x ^ 2 + 1) / (Real.cos x + 2)

theorem range_of_g :
  Set.range g = Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l46_4677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l46_4672

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x - 1) / (2^x + a)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem odd_function_implies_a_equals_one (a : ℝ) :
  is_odd_function (f a) → a = 1 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l46_4672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l46_4650

theorem circle_triangle_areas (X Y Z : ℝ) : 
  let a : ℝ := 15
  let b : ℝ := 20
  let c : ℝ := 25
  let triangle_area : ℝ := (a * b) / 2
  let circle_radius : ℝ := c / 2
  let circle_area : ℝ := π * circle_radius^2
  X > 0 ∧ Y > 0 ∧ Z > 0 →
  X + Y + Z = circle_area - triangle_area →
  Z ≥ X ∧ Z ≥ Y →
  X + Y + triangle_area = Z := by
  intro h1 h2 h3
  sorry

#check circle_triangle_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l46_4650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l46_4679

/-- The function f(x) = (a/x + √x)^9 where a is a real constant -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / x + Real.sqrt x) ^ 9

/-- The coefficient of x^3 in the expansion of f(x) -/
noncomputable def coeff_x3 (a : ℝ) : ℝ := 9 / 4

theorem problem_solution :
  (∀ a : ℝ, coeff_x3 a = 9 / 4 → a = 1 / 4) ∧
  (∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → f a x ≥ 27) ↔ a ≥ 4 / 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l46_4679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_l46_4609

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- Theorem: For an ellipse with equation x²/m + y²/4 = 1, focus on x-axis, and eccentricity 1/2, m = 16/3 -/
theorem ellipse_eccentricity_m (m : ℝ) :
  m > 4 →  -- Condition: focus on x-axis implies m > 4
  (∀ x y : ℝ, x^2/m + y^2/4 = 1 → 
    eccentricity (Real.sqrt (max m 4)) 2 = 1/2) →  -- Condition: equation of ellipse and eccentricity
  m = 16/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_m_l46_4609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lauras_distance_to_school_l46_4665

/-- The distance between Laura's house and her school in miles -/
def distance_to_school : ℝ := sorry

/-- The number of days Laura drives to school per week -/
def school_days : ℕ := 5

/-- The number of days Laura drives to the supermarket per week -/
def supermarket_days : ℕ := 2

/-- The additional distance to the supermarket compared to the school in miles -/
def extra_distance_to_supermarket : ℝ := 10

/-- The total distance Laura drives per week in miles -/
def total_weekly_distance : ℝ := 220

theorem lauras_distance_to_school :
  distance_to_school = 10 :=
by
  have h1 : 2 * distance_to_school * school_days +
            2 * (distance_to_school + extra_distance_to_supermarket) * supermarket_days =
            total_weekly_distance :=
    sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lauras_distance_to_school_l46_4665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_in_area_l46_4646

-- Define propositions p and q
variable (p q : Prop)

-- Define the meaning of p and q (we'll use these as axioms instead of definitions)
axiom p_meaning : p ↔ (∃ x : String, x = "A lands within the designated area")
axiom q_meaning : q ↔ (∃ x : String, x = "B lands within the designated area")

-- Theorem to prove
theorem at_least_one_not_in_area : 
  (∃ x : String, x = "At least one of the trainees does not land within the designated area") ↔ (¬p ∨ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_not_in_area_l46_4646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_difference_is_ten_minutes_l46_4601

-- Define the constants
noncomputable def distance_to_park : ℝ := 2
noncomputable def alice_speed : ℝ := 12
noncomputable def bob_speed : ℝ := 6

-- Define the arrival time difference function
noncomputable def arrival_time_difference : ℝ := 
  (distance_to_park / bob_speed - distance_to_park / alice_speed) * 60

-- Theorem to prove
theorem arrival_difference_is_ten_minutes : arrival_time_difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_difference_is_ten_minutes_l46_4601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_l46_4695

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 10.2

-- Define the cost function
def G (x : ℝ) : ℝ := 2 + x

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := R x - G x

theorem max_profit_at_400_units :
  ∃ (max_profit : ℝ),
    max_profit = f 4 ∧
    max_profit = 3.6 ∧
    ∀ x, f x ≤ max_profit := by
  sorry

#check max_profit_at_400_units

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_400_units_l46_4695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_functions_difference_finitely_many_zeros_l46_4632

-- Define the set of all functions from ℤ to ℝ
def S := ℤ → ℝ

-- Define the function g
def g (f : S) : S := λ x => f (x + 1) - f x

-- Define what it means for a function to be good
def is_good (f : S) : Prop := ∃ n : ℕ, ∀ x, (g^[n] f) x = 0

-- State the theorem
theorem good_functions_difference_finitely_many_zeros
  (s t : S) (hs : is_good s) (ht : is_good t) (h_distinct : s ≠ t) :
  (Set.Finite {m : ℤ | s m = t m}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_functions_difference_finitely_many_zeros_l46_4632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l46_4682

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the curve C1
def curve_C1 (x y : ℝ) : Prop := y^2 = 2*(x-1)

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    curve_C1 A.1 A.2 ∧ curve_C1 B.1 B.2 ∧
    distance point_P.1 point_P.2 A.1 A.2 + distance point_P.1 point_P.2 B.1 B.2 = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l46_4682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l46_4649

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * Real.exp x

-- State the theorem
theorem max_difference_of_f (k : ℝ) (h_k : k ∈ Set.Icc (-3) (-1)) :
  ∃ (C : ℝ), C = 4 * Real.exp 1 ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc k (k + 2) → x₂ ∈ Set.Icc k (k + 2) →
  |f x₁ - f x₂| ≤ C) ∧
  (∃ (y₁ y₂ : ℝ), y₁ ∈ Set.Icc k (k + 2) ∧ y₂ ∈ Set.Icc k (k + 2) ∧
  |f y₁ - f y₂| = C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_of_f_l46_4649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l46_4643

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.log (8 + 2*x - x^2) / Real.log 10
noncomputable def g (x : ℝ) := Real.sqrt (1 - 2/(x-1))

-- Define the domains M and N
def M : Set ℝ := {x | -2 < x ∧ x < 4}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- State the theorem
theorem domain_intersection :
  (M : Set ℝ) = {x | -2 < x ∧ x < 4} ∧
  (N : Set ℝ) = {x | x < 1 ∨ x ≥ 3} ∧
  (M ∩ N : Set ℝ) = {x | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_l46_4643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_win_finite_guaranteed_win_infinite_l46_4666

/-- Represents a spell with self-reduction a and opponent-reduction b -/
structure Spell where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ a < b

/-- Represents the state of the wizard duel -/
structure DuelState where
  height1 : ℝ
  height2 : ℝ

/-- Applies a spell to the duel state -/
def applySpell (state : DuelState) (spell : Spell) (isFirst : Bool) : DuelState :=
  if isFirst then
    { height1 := state.height1 - spell.a, height2 := state.height2 - spell.b }
  else
    { height1 := state.height1 - spell.b, height2 := state.height2 - spell.a }

/-- Checks if the given wizard has won -/
noncomputable def hasWon (state : DuelState) (isFirst : Bool) : Bool :=
  if isFirst then
    state.height1 > 0 ∧ state.height2 ≤ 0
  else
    state.height2 > 0 ∧ state.height1 ≤ 0

/-- Theorem: With a finite set of spells, the second wizard cannot guarantee a win -/
theorem no_guaranteed_win_finite (spells : Finset Spell) : 
  ∃ (strategy : ℕ → Spell), ∀ (n : ℕ), strategy n ∈ spells → 
    ¬∃ (counter_strategy : ℕ → Spell), ∀ (m : ℕ), counter_strategy m ∈ spells → 
      ∃ (k : ℕ), hasWon (applySpell (applySpell (DuelState.mk 100 100) (strategy k) true) (counter_strategy k) false) false := by
  sorry

/-- Theorem: With an infinite set of spells, the second wizard can guarantee a win -/
theorem guaranteed_win_infinite (spells : Set Spell) (h_infinite : Set.Infinite spells) : 
  ∃ (strategy : ℕ → Spell), ∀ (n : ℕ), strategy n ∈ spells → 
    ∀ (counter_strategy : ℕ → Spell), ∀ (m : ℕ), counter_strategy m ∈ spells → 
      ∃ (k : ℕ), hasWon (applySpell (applySpell (DuelState.mk 100 100) (counter_strategy k) true) (strategy k) false) false := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_guaranteed_win_finite_guaranteed_win_infinite_l46_4666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_l46_4681

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  first_positive : a 1 > 0
  condition : 3 * a 8 = 5 * a 13

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map (fun i => seq.a (i + 1)) |>.sum

/-- The theorem stating that S_20 is the largest sum -/
theorem largest_sum (seq : ArithmeticSequence) :
  ∀ n : ℕ, S seq n ≤ S seq 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_l46_4681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zero_values_l46_4664

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_not_constant : ∃ x y : ℝ, f x ≠ f y
axiom g_not_constant : ∃ x y : ℝ, g x ≠ g y

axiom f_functional_equation : ∀ x y : ℝ, f (x + y) = f x * g y + g x * f y
axiom g_functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y - f x * f y

-- State the theorem
theorem f_g_zero_values : f 0 = 0 ∧ g 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_zero_values_l46_4664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l46_4684

theorem pure_imaginary_condition (a : ℝ) : 
  ∃ b : ℝ, (Complex.I + 1) * (a - Complex.I) = Complex.I * b ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l46_4684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l46_4673

/-- The function for which we need to find the center of symmetry -/
noncomputable def f (x : ℝ) : ℝ := 2 - 1 / (x + 1)

/-- The proposed center of symmetry -/
def center : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the center of symmetry for the graph of f is (-1, 2) -/
theorem center_of_symmetry :
  ∀ (x : ℝ), x ≠ -1 → f (center.fst + (x - center.fst)) = f (center.fst - (x - center.fst)) := by
  sorry

#check center_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l46_4673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l46_4621

noncomputable def curve_x (t : Real) : Real := 6 * (t - Real.sin t)
noncomputable def curve_y (t : Real) : Real := 6 * (1 - Real.cos t)

noncomputable def lower_bound : Real := 9
noncomputable def x_lower_bound : Real := 0
noncomputable def x_upper_bound : Real := 12 * Real.pi

theorem area_of_bounded_figure :
  ∃ (a b : Real),
    a < b ∧
    curve_x a = x_lower_bound ∧
    curve_x b = x_upper_bound ∧
    (∀ t ∈ Set.Icc a b, curve_y t ≥ lower_bound) →
    (∫ (t : Real) in a..b, curve_y t * (deriv curve_x t)) = 36 * Real.pi + 72 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_figure_l46_4621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l46_4641

theorem order_of_values :
  let a := 0.1 * Real.exp 0.1
  let b := 1/9
  let c := -Real.log 0.9
  b > a ∧ a > c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l46_4641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_3_l46_4624

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 4

-- State the theorem
theorem f_of_g_3 : f (g 3) = (66 / 17) * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_3_l46_4624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l46_4623

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 1 = 0

-- Define the slope of a line
def line_slope (m : ℝ) : Prop := ∀ x y, line_equation x y → y = m * x + Real.sqrt 3 / 3

-- Define the x-intercept
def x_intercept (a : ℝ) : Prop := line_equation a 0

-- Theorem statement
theorem line_properties :
  ∃ (m a : ℝ), line_slope m ∧ x_intercept a ∧ m = -Real.sqrt 3 / 3 ∧ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l46_4623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popsicle_melting_ratio_l46_4611

/-- The number of popsicles in the box -/
def num_popsicles : ℕ := 6

/-- The ratio of melting speed between the last and first popsicle -/
noncomputable def last_to_first_ratio : ℝ := 32

/-- The ratio of melting speed between each subsequent popsicle and the previous one -/
noncomputable def melting_ratio : ℝ := (last_to_first_ratio ^ (1 / (num_popsicles - 1 : ℝ)))

/-- Theorem stating that the melting ratio raised to the power of (num_popsicles - 1) equals the last_to_first_ratio -/
theorem popsicle_melting_ratio :
  melting_ratio ^ (num_popsicles - 1 : ℝ) = last_to_first_ratio :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popsicle_melting_ratio_l46_4611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l46_4614

-- Define the bag contents
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

-- Define the number of balls drawn
def balls_drawn : ℕ := 2

-- Define the events
def exactly_one_white (outcome : Finset ℕ) : Prop :=
  outcome.card = balls_drawn ∧ (outcome.filter (λ x ↦ x ≤ white_balls)).card = 1

def exactly_two_white (outcome : Finset ℕ) : Prop :=
  outcome.card = balls_drawn ∧ (outcome.filter (λ x ↦ x ≤ white_balls)).card = 2

-- Theorem statement
theorem mutually_exclusive_not_contradictory :
  (∃ outcome : Finset ℕ, exactly_one_white outcome) ∧
  (∃ outcome : Finset ℕ, exactly_two_white outcome) ∧
  (∀ outcome : Finset ℕ, ¬(exactly_one_white outcome ∧ exactly_two_white outcome)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_contradictory_l46_4614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l46_4635

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  h : ∀ n, a (n + 1) = a n * q

/-- Sum of first n terms of a geometric sequence -/
def geometricSum (g : GeometricSequence) (n : ℕ) : ℚ :=
  if g.q = 1 then n * g.a 1
  else g.a 1 * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_ratio (g : GeometricSequence) 
  (h1 : g.a 3 = 6)
  (h2 : geometricSum g 3 = 18) :
  g.q = 1 ∨ g.q = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l46_4635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_median_l46_4645

def is_median (L : List ℕ) (m : ℕ) : Prop :=
  L.length % 2 = 1 ∧
  (L.filter (· < m)).length = (L.length - 1) / 2 ∧
  (L.filter (· > m)).length = (L.length - 1) / 2

theorem largest_possible_median (L : List ℕ) : 
  L.length = 11 ∧ 
  (∀ x ∈ L, x > 0) ∧
  3 ∈ L ∧ 5 ∈ L ∧ 7 ∈ L ∧ 9 ∈ L ∧ 1 ∈ L ∧ 8 ∈ L →
  ∃ M : ℕ, M ≤ 9 ∧ (∀ M' : ℕ, is_median L M' → M' ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_median_l46_4645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l46_4686

noncomputable def f (A ω φ x : ℝ) := A * Real.sin (ω * x + φ)

theorem function_properties :
  ∀ (A ω φ : ℝ),
    A > 0 → ω > 0 → 0 < φ → φ < Real.pi / 2 →
    (∀ x, f A ω φ x ≥ -4) →
    f A ω φ 0 = 2 * Real.sqrt 2 →
    (∀ x₁ x₂, x₁ < x₂ → f A ω φ x₁ = f A ω φ x₂ → x₂ - x₁ = Real.pi / ω) →
    (A = 4 ∧ φ = Real.pi / 4 ∧ ω = 1) ∧
    (∀ x, -Real.pi / 2 ≤ x → x ≤ Real.pi / 2 → f A ω φ x ≤ 4) ∧
    (∀ x, -Real.pi / 2 ≤ x → x ≤ Real.pi / 2 → f A ω φ x ≥ -2 * Real.sqrt 2) ∧
    (∀ x, Real.pi / 2 < x → x < Real.pi → f A ω φ x = 1 →
      Real.cos (x + 5 * Real.pi / 12) = (-3 * Real.sqrt 5 - 1) / 8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l46_4686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l46_4662

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define point M
def M : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem intersection_distance_sum : 
  ∃ (t₁ t₂ : ℝ), 
    C (l t₁).1 (l t₁).2 ∧ 
    C (l t₂).1 (l t₂).2 ∧ 
    t₁ ≠ t₂ ∧
    distance M (l t₁) + distance M (l t₂) = Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l46_4662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_digit_sum_18_l46_4653

def digits_sum (n : Nat) : Nat :=
  (n.repr.toList.map (fun c => c.toNat - 48)).sum

def all_digits_different (n : Nat) : Prop :=
  n.repr.toList.toFinset.card = n.repr.length

theorem largest_number_with_digit_sum_18 : 
  (∀ m : Nat, m ≠ 543210 → 
    (digits_sum m = 18 ∧ all_digits_different m) → 
    m < 543210) ∧
  digits_sum 543210 = 18 ∧ 
  all_digits_different 543210 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_digit_sum_18_l46_4653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_theorem_l46_4626

/-- Represents the number of slices in different pizza sizes -/
def SlicesPerPizza : Type := Nat

/-- Represents the number of pizzas ordered for each size -/
def PizzasOrdered : Type := Nat

/-- Calculates the total number of pizzas ordered -/
def total_pizzas (small medium large : Nat) : Nat :=
  small + medium + large

/-- Calculates the total number of slices from all pizzas -/
def total_slices (small medium large : Nat) 
  (small_slices medium_slices large_slices : Nat) : Nat :=
  small * small_slices + medium * medium_slices + large * large_slices

theorem pizza_order_theorem 
  (small_slices : Nat) 
  (medium_slices : Nat) 
  (large_slices : Nat) 
  (small_ordered : Nat) 
  (medium_ordered : Nat) :
  small_slices = 6 →
  medium_slices = 8 →
  large_slices = 12 →
  small_ordered = 4 →
  medium_ordered = 5 →
  total_slices small_ordered medium_ordered 
    ((136 - total_slices small_ordered medium_ordered 0 small_slices medium_slices 0) / large_slices)
    small_slices medium_slices large_slices = 136 →
  total_pizzas small_ordered medium_ordered 
    ((136 - total_slices small_ordered medium_ordered 0 small_slices medium_slices 0) / large_slices) = 15 :=
by sorry

#check pizza_order_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_order_theorem_l46_4626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_shortest_segment_length_shortest_segment_line_l46_4637

/-- The line l with parameter m -/
def line (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - y - 8 * m - 3 = 0

/-- The circle C -/
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 6)^2 = 25

/-- The line l always intersects with circle C -/
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, line m x y ∧ circle_C x y :=
by sorry

/-- The shortest segment length is 2√6 -/
theorem shortest_segment_length :
  ∃ m : ℝ, ∀ x y : ℝ,
    line m x y ∧ circle_C x y →
    ∃ x' y' : ℝ, line m x' y' ∧ circle_C x' y' ∧
    ∀ u v : ℝ, line m u v ∧ circle_C u v →
      (x - x')^2 + (y - y')^2 ≤ (u - v)^2 + (v - y)^2 ∧
      (x - x')^2 + (y - y')^2 = 24 :=
by sorry

/-- The line equation for the shortest segment -/
theorem shortest_segment_line :
  ∃ m : ℝ, ∀ x y : ℝ,
    line m x y ↔ x + 3 * y + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_shortest_segment_length_shortest_segment_line_l46_4637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l46_4629

-- Define a straight line
structure StraightLine where
  slope : Option ℝ
  inclinationAngle : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Helper definition for symmetric point
def IsSymmetricPoint (p q : Point) (f : ℝ → ℝ) : Prop :=
  (q.x - p.x) * (q.y - f ((p.x + q.x) / 2)) = -(q.y - p.y) * (q.x - (p.x + q.x) / 2) ∧
  f ((p.x + q.x) / 2) = (p.y + q.y) / 2

-- Define the theorem
theorem line_properties :
  -- 1. Existence of a line with inclination angle but no slope
  ∃ (l : StraightLine), l.slope = none ∧ l.inclinationAngle = 90 ∧
  -- 2. Existence of lines where greater inclination doesn't imply greater slope
  ∃ (l1 l2 : StraightLine), l1.inclinationAngle > l2.inclinationAngle ∧
    (∀ m1 m2, l1.slope = some m1 → l2.slope = some m2 → m1 < m2) ∧
  -- 3. Line with direction vector (3, √3) has 30° inclination
  ∃ (l : StraightLine), l.slope = some (Real.sqrt 3 / 3) → l.inclinationAngle = 30 ∧
  -- 4. Symmetry point calculation
  ∀ (p : Point), p.x = 0 ∧ p.y = 2 →
    ∃ (q : Point), q.x = 1 ∧ q.y = 3 ∧
      IsSymmetricPoint p q (fun x => x + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l46_4629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l46_4608

theorem complex_number_equality : (2 : ℂ) / (Complex.I + 1) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_equality_l46_4608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_from_P_l46_4603

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points
variable (P Q K B : ℝ × ℝ)

-- Define the circles
variable (circle1 circle2 : Circle)

-- Helper functions
def on_circle (point : ℝ × ℝ) (circle : Circle) : Prop := sorry

def is_external_similarity_center (point : ℝ × ℝ) (c1 c2 : Circle) : Prop := sorry

def is_internal_similarity_center (point : ℝ × ℝ) (c1 c2 : Circle) : Prop := sorry

def is_right_angle (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem right_angle_from_P (hDiffRadii : circle1.radius ≠ circle2.radius)
  (hCommonChord : P ≠ Q ∧ on_circle P circle1 ∧ on_circle P circle2 ∧ 
                           on_circle Q circle1 ∧ on_circle Q circle2)
  (hExternalCenter : is_external_similarity_center K circle1 circle2)
  (hInternalCenter : is_internal_similarity_center B circle1 circle2) :
  is_right_angle P B K :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_from_P_l46_4603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l46_4659

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 250 → time = 6 → ∃ length : ℝ, 
  (abs (length - 416.67) < 0.01) ∧ 
  (length = speed * 1000 / 3600 * time) := by
  intros h_speed h_time
  use speed * 1000 / 3600 * time
  constructor
  · rw [h_speed, h_time]
    norm_num
    apply abs_lt.mpr
    constructor <;> norm_num
  · rfl

#check train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_l46_4659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_chords_l46_4651

noncomputable def is_radius (r chord1 chord2 midpoint_distance : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (point : ℝ × ℝ),
    let chord1_midpoint := ((center.1 + point.1) / 2, (center.2 + point.2) / 2)
    let chord2_endpoint := (point.1 + 2 * (chord2 / 2) * (center.2 - point.2) / r, 
                            point.2 - 2 * (chord2 / 2) * (center.1 - point.1) / r)
    let chord2_midpoint := ((point.1 + chord2_endpoint.1) / 2, (point.2 + chord2_endpoint.2) / 2)
    (center.1 - point.1)^2 + (center.2 - point.2)^2 = r^2 ∧
    (center.1 - chord2_endpoint.1)^2 + (center.2 - chord2_endpoint.2)^2 = r^2 ∧
    (point.1 - chord2_endpoint.1)^2 + (point.2 - chord2_endpoint.2)^2 = chord2^2 ∧
    (chord1_midpoint.1 - chord2_midpoint.1)^2 + (chord1_midpoint.2 - chord2_midpoint.2)^2 = midpoint_distance^2

theorem circle_radius_from_chords (chord1 chord2 midpoint_distance : ℝ) 
  (h1 : chord1 = 9)
  (h2 : chord2 = 17)
  (h3 : midpoint_distance = 5) :
  ∃ (r : ℝ), r = 85 / 8 ∧ 
  is_radius r chord1 chord2 midpoint_distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_chords_l46_4651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_path_max_length_l46_4630

/-- Represents a right rectangular prism -/
structure Prism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the length of a space diagonal in a right rectangular prism -/
noncomputable def spaceDiagonalLength (p : Prism) : ℝ :=
  Real.sqrt (p.length ^ 2 + p.width ^ 2 + p.height ^ 2)

/-- Calculates the length of the longest face diagonal in a right rectangular prism -/
noncomputable def longestFaceDiagonalLength (p : Prism) : ℝ :=
  Real.sqrt (max (p.length ^ 2 + p.width ^ 2) (max (p.width ^ 2 + p.height ^ 2) (p.height ^ 2 + p.length ^ 2)))

/-- Represents a path through all corners of a prism -/
structure CornerPath (p : Prism) where
  length : ℝ
  visitsAllCorners : Prop
  startsAndEndsAtSameCorner : Prop
  usesOnlyStraightLines : Prop

/-- The theorem to be proved -/
theorem mosquito_path_max_length (p : Prism) (h1 : p.length = 1) (h2 : p.width = 2) (h3 : p.height = 3) :
  ∃ path : CornerPath p, 
    path.visitsAllCorners ∧ 
    path.startsAndEndsAtSameCorner ∧ 
    path.usesOnlyStraightLines ∧
    path.length = 4 * spaceDiagonalLength p + 2 * longestFaceDiagonalLength p ∧
    ∀ other_path : CornerPath p, other_path.length ≤ path.length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_path_max_length_l46_4630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_is_generalized_distance_l46_4620

def is_generalized_distance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0 ∧ (f x y = 0 ↔ x = y)) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

theorem abs_diff_is_generalized_distance :
  is_generalized_distance (fun x y ↦ |x - y|) := by
  sorry

#check abs_diff_is_generalized_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_diff_is_generalized_distance_l46_4620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_additional_inputs_l46_4636

/-- Represents the washing machine and laundry scenario -/
structure LaundryScenario where
  totalCapacity : ℚ
  clothesWeight : ℚ
  initialDetergent : ℚ
  detergentScoopWeight : ℚ
  optimalDetergentRatio : ℚ

/-- Calculates the additional detergent and water needed for optimal washing -/
noncomputable def additionalLaundryInputs (scenario : LaundryScenario) : ℚ × ℚ :=
  let totalDetergent := scenario.initialDetergent + scenario.detergentScoopWeight * 2
  let remainingCapacity := scenario.totalCapacity - scenario.clothesWeight - totalDetergent
  let optimalWater := remainingCapacity / (1 + scenario.optimalDetergentRatio)
  let optimalDetergent := optimalWater * scenario.optimalDetergentRatio
  (optimalDetergent - totalDetergent, optimalWater)

/-- Theorem stating that the calculated additional inputs are correct -/
theorem correct_additional_inputs (scenario : LaundryScenario) 
  (h1 : scenario.totalCapacity = 20)
  (h2 : scenario.clothesWeight = 5)
  (h3 : scenario.initialDetergent = 0)
  (h4 : scenario.detergentScoopWeight = 1/50)
  (h5 : scenario.optimalDetergentRatio = 1/250) :
  additionalLaundryInputs scenario = (1/50, 747/50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_additional_inputs_l46_4636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_squares_parallelograms_l46_4647

-- Define the points in the complex plane
variable (A B C A' B' C' A₁ A₂ B₁ B₂ C₁ C₂ : ℂ)

-- Define the triangle ABC
def triangle_ABC (A B C : ℂ) : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the squares on the sides of triangle ABC
def squares_on_sides (A B C : ℂ) : Prop :=
  ∃ (A₁ A₂ B₁ B₂ C₁ C₂ : ℂ),
    (B - C)^2 = (A₁ - B)^2 ∧ (A₂ - C)^2 = (A₁ - B)^2 ∧
    (C - A)^2 = (B₁ - C)^2 ∧ (B₂ - A)^2 = (B₁ - C)^2 ∧
    (A - B)^2 = (C₁ - A)^2 ∧ (C₂ - B)^2 = (C₁ - A)^2

-- Define the parallelograms
def parallelograms (A B C A' B' C' A₁ A₂ B₁ B₂ C₁ C₂ : ℂ) : Prop :=
  A' - A = C₂ - B₁ ∧
  B' - B = A₂ - C₁ ∧
  C' - C = B₂ - A₁

-- Define the theorem
theorem triangle_squares_parallelograms
  (h_triangle : triangle_ABC A B C)
  (h_squares : squares_on_sides A B C)
  (h_parallelograms : parallelograms A B C A' B' C' A₁ A₂ B₁ B₂ C₁ C₂) :
  (Complex.I * (B - C) * (A' - A)).im = 0 ∧  -- BC ⊥ AA'
  ((A + B + C) / 3 = (A' + B' + C') / 3)  -- Common centroid
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_squares_parallelograms_l46_4647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l46_4617

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the property that the polynomial satisfies
def SatisfiesProperty (f : RealPolynomial) : Prop :=
  ∀ (x y z : ℝ), x + y + z = 0 → f (x * y) + f (y * z) + f (z * x) = f (x * y + y * z + z * x)

-- State the theorem
theorem polynomial_property :
  ∀ (f : RealPolynomial), SatisfiesProperty f → ∃ (c : ℝ), ∀ (x : ℝ), f x = c * x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_property_l46_4617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_whole_number_of_7523_4987_l46_4648

noncomputable def round_to_nearest_whole (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem nearest_whole_number_of_7523_4987 :
  round_to_nearest_whole 7523.4987 = 7523 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearest_whole_number_of_7523_4987_l46_4648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_OMNB₁_l46_4652

noncomputable section

-- Define the cube
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

-- Define points
def A : Fin 3 → ℝ := ![0, 0, 0]
def B : Fin 3 → ℝ := ![1, 0, 0]
def C : Fin 3 → ℝ := ![1, 1, 0]
def D : Fin 3 → ℝ := ![0, 1, 0]
def A₁ : Fin 3 → ℝ := ![0, 0, 1]
def B₁ : Fin 3 → ℝ := ![1, 0, 1]
def C₁ : Fin 3 → ℝ := ![1, 1, 1]
def D₁ : Fin 3 → ℝ := ![0, 1, 1]

-- Define O as the center of the bottom face
def O : Fin 3 → ℝ := ![1/2, 1/2, 0]

-- Define M as the midpoint of A₁D₁
def M : Fin 3 → ℝ := ![0, 1/2, 1]

-- Define N as the midpoint of CC₁
def N : Fin 3 → ℝ := ![1, 1, 1/2]

-- Define the tetrahedron OMNB₁
def TetrahedronOMNB₁ : Set (Fin 3 → ℝ) :=
  {p | ∃ (a b c d : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 1 ∧
    p = fun i => a * O i + b * M i + c * N i + d * B₁ i}

-- State the theorem
theorem volume_of_tetrahedron_OMNB₁ : 
  MeasureTheory.volume TetrahedronOMNB₁ = 7/48 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_tetrahedron_OMNB₁_l46_4652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l46_4631

/-- Given a hyperbola and its properties, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →  -- Given hyperbola equation
  (∃ k : ℝ, ∀ x : ℝ, k * x = Real.sqrt 3 * x) →     -- Asymptote equation
  (∃ c : ℝ, c = 6) →                       -- Focus on directrix of y^2 = 24x
  (b / a = Real.sqrt 3) →                           -- Derived from asymptote
  (∀ x y : ℝ, x^2 / 9 - y^2 / 27 = 1) :=   -- Equation to prove
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l46_4631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probabilities_l46_4600

theorem badminton_match_probabilities :
  let p : ℚ := 1/4
  let prob_first_game : ℚ := 1/2
  let prob_win_after_win : ℚ := (1 + p) / 2
  let prob_win_after_loss : ℚ := (1 - p) / 2
  let prob_first_two_consecutive : ℚ := 5/16

  -- Condition: Probability of winning first two consecutive games
  prob_first_game * prob_win_after_win = prob_first_two_consecutive →

  -- Prove:
  -- 1. The value of p is 1/4
  p = 1/4 ∧
  -- 2. The probability of the match ending after 4 games is 165/512
  (let prob_end_after_four : ℚ :=
    2 * (prob_first_game * prob_win_after_win * prob_win_after_loss * prob_win_after_loss +
         prob_first_game * prob_win_after_loss * prob_win_after_win * prob_win_after_win +
         prob_first_game * prob_win_after_loss * prob_win_after_win * prob_win_after_win)
   prob_end_after_four = 165/512) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_badminton_match_probabilities_l46_4600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l46_4628

-- Define the exponential function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem exponential_function_difference (a : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc 0 2, f a x ≤ max) ∧
                    (∀ x ∈ Set.Icc 0 2, f a x ≥ min) ∧
                    max - min = 3) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_difference_l46_4628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_average_speed_l46_4633

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: Mark's average speed is 7 miles per hour -/
theorem marks_average_speed :
  let total_distance : ℝ := 42
  let total_time : ℝ := 6
  average_speed total_distance total_time = 7 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

#check marks_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_average_speed_l46_4633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l46_4616

-- Define the train's length in meters
noncomputable def train_length : ℝ := 150

-- Define the train's speed in km/hr
noncomputable def train_speed_kmh : ℝ := 195

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Calculate the train's speed in m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Define the time it takes for the train to cross the pole
noncomputable def crossing_time : ℝ := train_length / train_speed_ms

-- Theorem to prove
theorem train_crossing_time :
  ∃ ε > 0, |crossing_time - 2.77| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l46_4616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_23_l46_4607

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | n + 1 => 2 * sequence_a n + 1

theorem a_4_equals_23 : sequence_a 4 = 23 := by
  -- Unfold the definition of sequence_a for n = 4, 3, 2, 1
  have h1 : sequence_a 4 = 2 * sequence_a 3 + 1 := rfl
  have h2 : sequence_a 3 = 2 * sequence_a 2 + 1 := rfl
  have h3 : sequence_a 2 = 2 * sequence_a 1 + 1 := rfl
  have h4 : sequence_a 1 = 2 := rfl

  -- Calculate the value of sequence_a 4
  calc
    sequence_a 4 = 2 * sequence_a 3 + 1 := h1
    _ = 2 * (2 * sequence_a 2 + 1) + 1 := by rw [h2]
    _ = 2 * (2 * (2 * sequence_a 1 + 1) + 1) + 1 := by rw [h3]
    _ = 2 * (2 * (2 * 2 + 1) + 1) + 1 := by rw [h4]
    _ = 2 * (2 * 5 + 1) + 1 := by norm_num
    _ = 2 * 11 + 1 := by norm_num
    _ = 23 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_23_l46_4607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_announcement_l46_4689

/-- Represents a measurement with a value and uncertainty -/
structure Measurement where
  value : Float
  uncertainty : Float

/-- Checks if a given number is the most accurate announcement for a measurement -/
def isAccurateAnnouncement (m : Measurement) (announcement : Float) : Prop :=
  let lowerBound := m.value - m.uncertainty
  let upperBound := m.value + m.uncertainty
  (∀ n : Nat, (Float.round (lowerBound * Float.ofNat (10^n)) / Float.ofNat (10^n)) = 
              (Float.round (upperBound * Float.ofNat (10^n)) / Float.ofNat (10^n))) ∧
  (∃ n : Nat, announcement = (Float.round (m.value * Float.ofNat (10^n)) / Float.ofNat (10^n))) ∧
  (∀ x : Float, x > announcement → 
    ∃ n : Nat, (Float.round (lowerBound * Float.ofNat (10^n)) / Float.ofNat (10^n)) ≠ 
                (Float.round (upperBound * Float.ofNat (10^n)) / Float.ofNat (10^n)))

theorem most_accurate_announcement (D : Measurement) 
  (h1 : D.value = 3.78249)
  (h2 : D.uncertainty = 0.00295) :
  isAccurateAnnouncement D 3.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_announcement_l46_4689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_sum_inverse_squares_l46_4618

theorem eccentricity_sum_inverse_squares (e₁ e₂ : ℝ) 
  (h₁ : e₁ > 0) (h₂ : e₂ > 0) 
  (h₃ : ∃ (a₁ a₂ c : ℝ), a₁ > 0 ∧ a₂ > 0 ∧ c > 0 ∧ 
    e₁^2 = (c^2) / (a₁^2) ∧ 
    e₂^2 = (c^2) / (a₂^2) + 1 ∧
    a₁^2 + a₂^2 = 2 * c^2) : 
  1 / e₁^2 + 1 / e₂^2 = 2 :=
sorry

#check eccentricity_sum_inverse_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_sum_inverse_squares_l46_4618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_problem_l46_4693

/-- Given a function f(x) = b * a^x, where a and b are constants, a > 0, and a ≠ 1,
    prove that if f passes through points (1,8) and (3,32), then a = 2, b = 4,
    and the maximum value of m such that (1/a)^x + (1/b)^x - m ≥ 0 for all x ≤ 1 is 3/4 -/
theorem function_and_inequality_problem 
  (f : ℝ → ℝ)
  (a b : ℝ) 
  (ha_pos : a > 0) 
  (ha_neq_one : a ≠ 1) 
  (hf : ∀ x, f x = b * a^x) 
  (hpoint1 : f 1 = 8) 
  (hpoint2 : f 3 = 32) : 
  a = 2 ∧ b = 4 ∧ 
  (∀ m, (∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) ↔ m ≤ 3/4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_inequality_problem_l46_4693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_quadrants_l46_4658

/-- A function that represents a linear relationship passing through the origin -/
def proportional_function (k : ℝ) : ℝ → ℝ := λ x ↦ k * x

/-- Predicate to check if a point (x, y) is in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Predicate to check if a point (x, y) is in the fourth quadrant -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that if the graph of y = kx passes through the second and fourth quadrants, then k = -2 -/
theorem proportional_function_quadrants (k : ℝ) : 
  (∃ x₁ y₁, in_second_quadrant x₁ y₁ ∧ y₁ = proportional_function k x₁) ∧
  (∃ x₂ y₂, in_fourth_quadrant x₂ y₂ ∧ y₂ = proportional_function k x₂) →
  k = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_quadrants_l46_4658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telescope_range_increase_l46_4605

/-- Given an original visual range and a telescope that increases the range by a certain percentage,
    calculate the new visual range with the telescope. -/
noncomputable def visual_range_with_telescope (original_range : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_range * (1 + increase_percentage / 100)

/-- Theorem stating that for an original visual range of 100 km and a 50% increase,
    the new visual range with the telescope is 150 km. -/
theorem telescope_range_increase :
  visual_range_with_telescope 100 50 = 150 := by
  -- Unfold the definition of visual_range_with_telescope
  unfold visual_range_with_telescope
  -- Simplify the arithmetic
  simp [mul_add, mul_div_cancel']
  -- Check that 100 * (1 + 50 / 100) = 150
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telescope_range_increase_l46_4605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l46_4671

theorem triangle_side_ratio (A B C : Real) (a b c : Real) 
  (h1 : A + B + C = Real.pi)
  (h2 : B = Real.pi/6) 
  (h3 : C = 2*Real.pi/3) 
  (h4 : a/Real.sin A = b/Real.sin B) 
  (h5 : b/Real.sin B = c/Real.sin C) : 
  ∃ (k : Real), k > 0 ∧ a = k ∧ b = k ∧ c = k * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_l46_4671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_increases_fastest_l46_4656

open Real

-- Define the domain
def Domain := {x : ℝ | x > 0}

-- Define the functions
noncomputable def f1 : Domain → ℝ := fun _ ↦ 100
noncomputable def f2 : Domain → ℝ := fun _ ↦ 10
noncomputable def f3 : Domain → ℝ := fun x ↦ log x / log 2  -- log base 2
noncomputable def f4 : Domain → ℝ := fun x ↦ exp x

-- Define what it means for a function to increase faster
def increases_faster (f g : Domain → ℝ) : Prop :=
  ∀ x y : Domain, x < y → (f y - f x) > (g y - g x)

-- State the theorem
theorem exp_increases_fastest :
  ∀ x : Domain, 
    increases_faster f4 f1 ∧
    increases_faster f4 f2 ∧
    increases_faster f4 f3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_increases_fastest_l46_4656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l46_4680

theorem angle_in_third_quadrant (a : Real) 
  (h1 : Real.sin a + Real.cos a < 0) 
  (h2 : Real.tan a > 0) : 
  a ∈ Set.Ioo (π) (3 * π / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l46_4680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_num_acquaintances_exists_meeting_with_sixteen_people_l46_4642

/-- Represents a group of people at a meeting -/
structure Meeting where
  n : Nat
  acquainted : Fin n → Fin n → Bool
  n_gt_one : n > 1
  two_common_acquaintances : 
    ∀ (a b : Fin n), a ≠ b → 
      (Finset.filter (λ c ↦ acquainted a c ∧ acquainted b c) (Finset.univ : Finset (Fin n))).card = 2

/-- The number of acquaintances for each person -/
def num_acquaintances (m : Meeting) (person : Fin m.n) : Nat :=
  (Finset.filter (λ x ↦ m.acquainted person x) (Finset.univ : Finset (Fin m.n))).card

theorem all_same_num_acquaintances (m : Meeting) :
  ∀ (a b : Fin m.n), num_acquaintances m a = num_acquaintances m b := by
  sorry

theorem exists_meeting_with_sixteen_people :
  ∃ (m : Meeting), m.n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_num_acquaintances_exists_meeting_with_sixteen_people_l46_4642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l46_4670

/-- Given two vectors a and b in ℝ², prove that |4a - b| = √78 -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) 
  (ha : ‖a‖ = Real.sqrt 3)
  (hb : ‖b‖ = Real.sqrt 6)
  (hab : a.1 * b.1 + a.2 * b.2 = -(Real.sqrt 3 * Real.sqrt 6) / 2) :
  ‖(4 * a.1 - b.1, 4 * a.2 - b.2)‖ = Real.sqrt 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l46_4670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_rect_eq_line_l_general_eq_line_l1_eq_l46_4613

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 - t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Theorem for the rectangular equation of curve C
theorem curve_C_rect_eq (x y : ℝ) :
  (∃ θ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ↔ x^2 + (y - 1)^2 = 1 :=
sorry

-- Theorem for the general equation of line l
theorem line_l_general_eq (x y : ℝ) :
  (∃ t, (x, y) = line_l t) ↔ x + y = 3 :=
sorry

-- Theorem for the equations of line l₁
theorem line_l1_eq (x y : ℝ) :
  (∃ k, x + y + k = 0 ∧
    (∃ x1 y1 x2 y2, x1^2 + (y1 - 1)^2 = 1 ∧
                    x2^2 + (y2 - 1)^2 = 1 ∧
                    x1 + y1 + k = 0 ∧
                    x2 + y2 + k = 0 ∧
                    (x1 - x2)^2 + (y1 - y2)^2 = 3)) ↔
  (x + y = 1 + Real.sqrt 2 / 2 ∨ x + y = 1 - Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_rect_eq_line_l_general_eq_line_l1_eq_l46_4613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cinematic_academy_members_proof_l46_4660

noncomputable def cinematic_academy_members : ℕ :=
  let minimum_fraction : ℚ := 1 / 4
  let minimum_lists : ℚ := 191.25
  let total_members : ℚ := minimum_lists / minimum_fraction
  (Int.ceil total_members).toNat

theorem cinematic_academy_members_proof :
  cinematic_academy_members = 765 :=
by
  unfold cinematic_academy_members
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cinematic_academy_members_proof_l46_4660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l46_4639

/-- The function f(x) with parameter ω -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * (Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)) - Real.sqrt 3

/-- The minimum positive period of f is π -/
def min_period (ω : ℝ) : Prop := ∃ (k : ℝ), k > 0 ∧ ∀ (x : ℝ), f ω (x + k) = f ω x ∧ ∀ (k' : ℝ), 0 < k' ∧ k' < k → ∃ (x : ℝ), f ω (x + k') ≠ f ω x

theorem f_properties (ω : ℝ) (h1 : ω > 0) (h2 : min_period ω) :
  ω = 1 ∧ Set.Icc (-2) 0 = Set.range (fun x => f ω x) ∩ Set.Icc (-Real.pi/6) (Real.pi/6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l46_4639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_one_seventh_l46_4625

theorem inverse_expression_equals_one_seventh :
  (3 - 4 * (4 - 6)⁻¹ + 2 : ℝ)⁻¹ = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_expression_equals_one_seventh_l46_4625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cosine_tangent_ratio_l46_4690

theorem limit_cosine_tangent_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    |x - Real.pi| < δ → 
    x ≠ Real.pi → 
    |(Real.cos (3*x) - Real.cos x) / (Real.tan (2*x))^2 - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_cosine_tangent_ratio_l46_4690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l46_4644

theorem sum_of_m_values (a b c m : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a + 1) / (2 - b) = m ∧
  (b + 1) / (2 - c) = m ∧
  (c + 1) / (2 - a) = m →
  ∃ m₁ m₂ : ℂ, m₁ + m₂ = 1 ∧ (m = m₁ ∨ m = m₂) := by
  sorry

#check sum_of_m_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_m_values_l46_4644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pharmacy_profit_growth_rate_l46_4694

/-- The average monthly growth rate of a pharmacy's profit -/
noncomputable def average_monthly_growth_rate (initial_profit final_profit : ℝ) (months : ℕ) : ℝ :=
  (final_profit / initial_profit) ^ (1 / months : ℝ) - 1

/-- Theorem stating that the average monthly growth rate is 0.5 given the conditions -/
theorem pharmacy_profit_growth_rate :
  let initial_profit : ℝ := 5000
  let final_profit : ℝ := 11250
  let months : ℕ := 2
  average_monthly_growth_rate initial_profit final_profit months = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pharmacy_profit_growth_rate_l46_4694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l46_4698

/-- A graph representing the acquaintance relationships in the meeting. -/
structure AcquaintanceGraph where
  V : Finset ℕ
  E : Finset (ℕ × ℕ)
  vertex_count : V.card = 30
  degree_bound : ∀ v, v ∈ V → (E.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ 5
  no_complete_five : ∀ S, S ⊆ V → S.card = 5 → ∃ u v, u ∈ S ∧ v ∈ S ∧ u ≠ v ∧ (u, v) ∉ E ∧ (v, u) ∉ E

/-- An independent set in the graph. -/
def IndependentSet (G : AcquaintanceGraph) (S : Finset ℕ) : Prop :=
  S ⊆ G.V ∧ ∀ u v, u ∈ S → v ∈ S → u ≠ v → (u, v) ∉ G.E ∧ (v, u) ∉ G.E

/-- The size of the maximum independent set in the graph is 6. -/
theorem max_independent_set_size (G : AcquaintanceGraph) :
  (∃ S, IndependentSet G S ∧ S.card = 6) ∧
  (∀ S, IndependentSet G S → S.card ≤ 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_independent_set_size_l46_4698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_lines_exist_l46_4661

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the distance from a point to a line
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

-- Define the property of three points being non-collinear
def nonCollinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the property of a line satisfying the distance ratio condition
def satisfiesDistanceRatio (l : Line) (A B C : Point) : Prop :=
  let dA := distancePointToLine A l
  let dB := distancePointToLine B l
  let dC := distancePointToLine C l
  (dA = dB ∧ dC = 2*dA) ∨ (dA = dC ∧ dB = 2*dA) ∨ (dB = dC ∧ dA = 2*dB)

-- Theorem statement
theorem twelve_lines_exist (A B C : Point) (h : nonCollinear A B C) :
  ∃ (S : Finset Line), (∀ l ∈ S, satisfiesDistanceRatio l A B C) ∧ (S.card = 12) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_lines_exist_l46_4661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l46_4638

/-- Calculates the total time for a round trip given the following conditions:
    - The total distance is 30 miles
    - The first 18 miles are traveled at 9 miles per hour
    - The next 12 miles are traveled at 10 miles per hour
    - The return trip is traveled at an average speed of 7.5 miles per hour
-/
theorem round_trip_time : 
  let total_distance : ℝ := 30
  let first_leg_distance : ℝ := 18
  let second_leg_distance : ℝ := 12
  let first_leg_speed : ℝ := 9
  let second_leg_speed : ℝ := 10
  let return_speed : ℝ := 7.5
  let outbound_time := first_leg_distance / first_leg_speed + second_leg_distance / second_leg_speed
  let return_time := total_distance / return_speed
  outbound_time + return_time = 7.2 := by
  -- Proof steps would go here
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l46_4638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_audition_assignments_l46_4699

def num_men : ℕ := 7
def num_women : ℕ := 8
def male_roles : ℕ := 3
def female_roles : ℕ := 3
def neutral_roles : ℕ := 4

def total_roles : ℕ := male_roles + female_roles + neutral_roles

def remaining_people : ℕ := num_men + num_women - male_roles - female_roles

theorem audition_assignments :
  (Nat.factorial num_men / Nat.factorial (num_men - male_roles)) *
  (Nat.factorial num_women / Nat.factorial (num_women - female_roles)) *
  (Nat.factorial remaining_people / Nat.factorial (remaining_people - neutral_roles)) =
  213542400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_audition_assignments_l46_4699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_triangles_l46_4675

/-- Given two triangles PQR and STU with side lengths (6, 8, 10) and (9, 12, 15) respectively,
    prove that the ratio of their areas is 4/9. -/
theorem area_ratio_of_triangles (P Q R S T U : ℝ × ℝ) : 
  let pqr_sides : Fin 3 → ℝ := ![6, 8, 10]
  let stu_sides : Fin 3 → ℝ := ![9, 12, 15]
  let pqr_area := abs ((P.1 * Q.2 - P.2 * Q.1 + Q.1 * R.2 - Q.2 * R.1 + R.1 * P.2 - R.2 * P.1) / 2)
  let stu_area := abs ((S.1 * T.2 - S.2 * T.1 + T.1 * U.2 - T.2 * U.1 + U.1 * S.2 - U.2 * S.1) / 2)
  (∀ (i : Fin 3), ‖(P.1 - Q.1, P.2 - Q.2)‖ = pqr_sides i ∨ 
                  ‖(Q.1 - R.1, Q.2 - R.2)‖ = pqr_sides i ∨ 
                  ‖(R.1 - P.1, R.2 - P.2)‖ = pqr_sides i) →
  (∀ (i : Fin 3), ‖(S.1 - T.1, S.2 - T.2)‖ = stu_sides i ∨ 
                  ‖(T.1 - U.1, T.2 - U.2)‖ = stu_sides i ∨ 
                  ‖(U.1 - S.1, U.2 - S.2)‖ = stu_sides i) →
  pqr_area / stu_area = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_triangles_l46_4675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l46_4691

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 3 * x

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2 = -4

-- Define the distance function
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l x₁ y₁ ∧ curve_C x₁ y₁ ∧
    line_l x₂ y₂ ∧ curve_C x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    distance x₁ y₁ x₂ y₂ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l46_4691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_sculpting_l46_4602

/-- The percentage of marble cut away in the first week of sculpting -/
def first_week_cut_percentage : ℝ := 30

theorem statue_sculpting (original_weight : ℝ) (final_weight : ℝ)
  (h_original : original_weight = 250)
  (h_final : final_weight = 105) :
  (0.75 * (0.8 * ((1 - first_week_cut_percentage / 100) * original_weight))) = final_weight :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_sculpting_l46_4602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_sum_bound_l46_4685

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.log x - x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x + m * x^2

def is_extreme_point (m : ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), g m y ≤ g m x

theorem f_monotonicity_and_g_sum_bound :
  (∀ x ∈ Set.Ioo 0 (1/2), ∀ y ∈ Set.Ioo 0 (1/2), x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioi (1/2), ∀ y ∈ Set.Ioi (1/2), x < y → f x > f y) ∧
  (∀ m ∈ Set.Ioo 0 (1/4), ∀ x₁ x₂ : ℝ, is_extreme_point m x₁ → is_extreme_point m x₂ → x₁ ≠ x₂ →
    g m x₁ + g m x₂ < -3/2) :=
by sorry

#check f_monotonicity_and_g_sum_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_sum_bound_l46_4685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cost_theorem_l46_4627

/-- The minimal cost to mark all numbers from 2 to 30 -/
def minimalCost : ℕ := 5

/-- The set of numbers on the board -/
def boardNumbers : Finset ℕ := Finset.range 29 \ Finset.range 1

/-- A number is freely markable if it's a divisor or multiple of a marked number -/
def freelyMarkable (marked : Finset ℕ) (n : ℕ) : Prop :=
  ∃ m ∈ marked, (n ∣ m ∨ m ∣ n)

/-- The set of prime numbers between 2 and 30 -/
def primes : Finset ℕ := boardNumbers.filter Nat.Prime

/-- The cost of marking a set of numbers -/
def markingCost (S : Finset ℕ) : ℕ := S.card

/-- The main theorem stating that the minimal cost is indeed 5 -/
theorem minimal_cost_theorem :
  ∀ (marked : Finset ℕ),
    (∀ n ∈ boardNumbers, n ∈ marked ∨ freelyMarkable marked n) →
    minimalCost ≤ markingCost marked :=
sorry

#check minimal_cost_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_cost_theorem_l46_4627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_of_point_l46_4610

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- A point on a line with a specific x-coordinate -/
noncomputable def point_on_line (l : Line3D) (x : ℝ) : ℝ × ℝ × ℝ :=
  let t := (x - l.point1.1) / (l.point2.1 - l.point1.1)
  (x, 
   l.point1.2.1 + t * (l.point2.2.1 - l.point1.2.1),
   l.point1.2.2 + t * (l.point2.2.2 - l.point1.2.2))

/-- The theorem to be proved -/
theorem z_coordinate_of_point (l : Line3D) (h : l.point1 = (1, 3, 2) ∧ l.point2 = (4, 2, -1)) :
  (point_on_line l 3).2.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_coordinate_of_point_l46_4610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l46_4663

noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def sphereDiameter (r : ℝ) : ℝ := 2 * r

theorem sphere_diameter_theorem :
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧
    ¬ (∃ (k : ℕ), k > 1 ∧ k ^ 3 ∣ b) ∧
    sphereDiameter (((3 * sphereVolume 7) / ((4 / 3) * Real.pi)) ^ (1 / 3)) = a * b ^ (1 / 3 : ℝ) ∧
    a = 14 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_theorem_l46_4663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_zeros_implies_a_range_l46_4697

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the property of having exactly 7 zeros
def has_exactly_seven_zeros (g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ), 
    (∀ x, g x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ ∨ x = x₅ ∨ x = x₆ ∨ x = x₇) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧ x₁ ≠ x₇ ∧
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧ x₂ ≠ x₇ ∧
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧ x₃ ≠ x₇ ∧
    x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧ x₄ ≠ x₇ ∧
    x₅ ≠ x₆ ∧ x₅ ≠ x₇ ∧
    x₆ ≠ x₇

theorem seven_zeros_implies_a_range (a : ℝ) : 
  a > 0 → has_exactly_seven_zeros (λ x ↦ f (f x - a)) → 2 - Real.sqrt 3 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_zeros_implies_a_range_l46_4697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_tan4_l46_4655

theorem min_sin4_plus_tan4 :
  (∀ x : ℝ, Real.sin x ^ 4 + Real.tan x ^ 4 ≥ 0) ∧
  (∃ x : ℝ, Real.sin x ^ 4 + Real.tan x ^ 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin4_plus_tan4_l46_4655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_properties_l46_4678

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the circle P
def circle_P (x y r : ℝ) : Prop := x^2 + (y - 3)^2 = r^2

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the number of intersection points
def intersection_points (n : ℕ) : Prop := n ≥ 0

theorem hyperbola_circle_properties :
  -- The eccentricity of C is √6/2
  eccentricity (Real.sqrt 2) 1 = Real.sqrt 6 / 2 ∧
  -- When r = √6, C and P have no common points
  (∀ x y : ℝ, hyperbola_C x y → circle_P x y (Real.sqrt 6) → False) ∧
  -- When r = 2√2, C and P have exactly two common points
  (∃! n : ℕ, n = 2 ∧ intersection_points n ∧
    ∀ x y : ℝ, hyperbola_C x y → circle_P x y (2 * Real.sqrt 2) →
      (x = -2 ∧ y = 1) ∨ (x = 2 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_properties_l46_4678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_A_significantly_improved_l46_4668

noncomputable def x : Fin 10 → ℝ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

noncomputable def y : Fin 10 → ℝ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

noncomputable def z (i : Fin 10) : ℝ := x i - y i

noncomputable def z_mean : ℝ := (Finset.sum Finset.univ z) / 10

noncomputable def z_variance : ℝ := (Finset.sum Finset.univ (fun i => (z i - z_mean)^2)) / 10

theorem process_A_significantly_improved : z_mean ≥ 2 * Real.sqrt (z_variance / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_A_significantly_improved_l46_4668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l46_4683

noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x - 1) / (x - 2))

theorem range_of_x (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ (x > 2 ∨ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l46_4683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soybeans_to_oil_amount_l46_4612

/- Define the conversion rates and prices -/
noncomputable def soybeans_to_tofu : ℝ := 3
noncomputable def soybeans_to_oil : ℝ := 1 / 6
noncomputable def price_soybeans : ℝ := 2
noncomputable def price_tofu : ℝ := 3
noncomputable def price_oil : ℝ := 15

/- Define the purchase and sales amounts -/
noncomputable def purchase_amount : ℝ := 920
noncomputable def sales_amount : ℝ := 1800

/- Define the theorem -/
theorem soybeans_to_oil_amount :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x ≤ purchase_amount / price_soybeans ∧
    (x / soybeans_to_oil) * price_oil +
    ((purchase_amount / price_soybeans - x) * soybeans_to_tofu) * price_tofu = sales_amount ∧
    x = 360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soybeans_to_oil_amount_l46_4612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_product_intersection_l46_4669

def A : Set ℕ := {x | x^2 - 2 * x ≤ 0}
def B : Set ℕ := {1, 2, 3}

theorem cartesian_product_intersection :
  (A.prod B) ∩ (B.prod A) = {(1, 1), (1, 2), (2, 1), (2, 2)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_product_intersection_l46_4669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_G_properties_l46_4606

/-- An ellipse with eccentricity √2/2 and minor axis endpoints at (0, 1) and (0, -1) -/
structure EllipseG where
  eccentricity : ℝ
  minor_axis_endpoint : ℝ
  ecc_eq : eccentricity = Real.sqrt 2 / 2
  endpoint_eq : minor_axis_endpoint = 1

/-- A point on the ellipse G -/
structure PointOnEllipseG (G : EllipseG) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / 2 + y^2 = 1

/-- Theorem about the standard equation of ellipse G and a property of points on it -/
theorem ellipse_G_properties (G : EllipseG) :
  (∀ (x y : ℝ), (x^2 / 2 + y^2 = 1) ↔ ∃ (p : PointOnEllipseG G), p.x = x ∧ p.y = y) ∧
  (∀ (C D : PointOnEllipseG G),
    C.x = -D.x →
    C.y = D.y →
    C.x ≠ 0 →
    let M : ℝ × ℝ := (C.x / (C.y + 1), 0)
    let A : ℝ × ℝ := (0, 1)
    ¬ (((A.1 - M.1)^2 + (A.2 - M.2)^2) * ((A.1 - D.x)^2 + (A.2 - D.y)^2) =
       ((M.1 - D.x)^2 + (M.2 - D.y)^2) * ((A.1 - M.1)^2 + (A.2 - M.2)^2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_G_properties_l46_4606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_equals_pi_minus_two_over_four_l46_4657

theorem integral_sqrt_minus_x_equals_pi_minus_two_over_four :
  ∫ x in Set.Icc 0 1, (Real.sqrt (1 - (x - 1)^2) - x) = (π - 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_minus_x_equals_pi_minus_two_over_four_l46_4657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l46_4640

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ h : ℝ, (p.1 - h)^2 + p.2^2 = 10}

-- Define points A and B
def A : ℝ × ℝ := (5, 1)
def B : ℝ × ℝ := (1, 3)

theorem circle_equation :
  (A ∈ C ∧ B ∈ C) →
  (∃ h : ℝ, ∀ p ∈ C, p.2 = 0 → p.1 = h) →
  C = {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 10} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l46_4640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l46_4634

noncomputable def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

noncomputable def triangle_area (x1 y1 : ℝ) : ℝ :=
  (abs x1 * abs y1) / 2

theorem triangle_area_is_18 :
  ∃ (x1 y1 : ℝ),
    line_through_points (-1) 5 (-3) 3 x1 0 ∧
    line_through_points (-1) 5 (-3) 3 0 y1 ∧
    triangle_area x1 y1 = 18 := by
  sorry

#check triangle_area_is_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_18_l46_4634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_in_group_l46_4615

/-- Represents a restaurant group with adults and kids -/
structure RestaurantGroup where
  total : ℕ
  adults : ℕ
  kids : ℕ
  adultMealCost : ℕ
  totalCost : ℕ

/-- The conditions of the restaurant group problem -/
def problemConditions (k : ℕ) : RestaurantGroup where
  total := 12
  adults := 12 - k
  kids := k
  adultMealCost := 3
  totalCost := 15

theorem kids_in_group : ∃ (k : ℕ), 
  (problemConditions k).kids = 7 ∧ 
  (problemConditions k).adults * (problemConditions k).adultMealCost = (problemConditions k).totalCost := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kids_in_group_l46_4615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_pizza_calories_l46_4688

theorem bob_pizza_calories (total_slices eaten_fraction calories_per_slice : ℕ) :
  let eaten_slices := total_slices * eaten_fraction / 2
  eaten_slices * calories_per_slice = 1200 := by
  sorry

#check bob_pizza_calories

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_pizza_calories_l46_4688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l46_4619

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

theorem function_properties (k : ℝ) :
  ((∀ x : ℝ, f k x > 0) ↔ k > -2) ∧
  ((∃ x : ℝ, f k x = -2 ∧ ∀ y : ℝ, f k y ≥ -2) ↔ k = -8) ∧
  ((∀ x₁ x₂ x₃ : ℝ, f k x₁ + f k x₂ > f k x₃) ↔ -1/2 ≤ k ∧ k ≤ 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l46_4619
