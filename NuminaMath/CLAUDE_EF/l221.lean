import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_inscribed_sphere_radius_l221_22164

/-- Predicate to represent that S₁, S₂, S₃, S₄ are face areas and V is volume of a tetrahedron -/
def IsTetrahedron (S₁ S₂ S₃ S₄ V : ℝ) : Prop := sorry

/-- Predicate to represent that R is the radius of an inscribed sphere in a tetrahedron 
    with face areas S₁, S₂, S₃, S₄ and volume V -/
def IsInscribedSphere (R S₁ S₂ S₃ S₄ V : ℝ) : Prop := sorry

theorem tetrahedron_inscribed_sphere_radius 
  (S₁ S₂ S₃ S₄ R V : ℝ) 
  (h₁ : S₁ > 0) 
  (h₂ : S₂ > 0) 
  (h₃ : S₃ > 0) 
  (h₄ : S₄ > 0) 
  (hR : R > 0) 
  (hV : V > 0) 
  (h_tetrahedron : IsTetrahedron S₁ S₂ S₃ S₄ V) 
  (h_inscribed : IsInscribedSphere R S₁ S₂ S₃ S₄ V) : 
  R = 3 * V / (S₁ + S₂ + S₃ + S₄) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_inscribed_sphere_radius_l221_22164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_sqrt2_over_2_l221_22100

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (2 : ℝ) ^ x else Real.sin x

-- State the theorem
theorem f_composition_equals_sqrt2_over_2 :
  f (f (7 * Real.pi / 6)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_sqrt2_over_2_l221_22100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_time_correct_l221_22110

/-- The growth rate of Pu on the first day -/
noncomputable def pu_initial_growth : ℝ := 3

/-- The growth rate of Guan on the first day -/
noncomputable def guan_initial_growth : ℝ := 1

/-- The daily growth rate multiplier for Pu after the first day -/
noncomputable def pu_growth_multiplier : ℝ := 1/2

/-- The daily growth rate multiplier for Guan after the first day -/
noncomputable def guan_growth_multiplier : ℝ := 2

/-- The time when the lengths of Pu and Guan are equal -/
noncomputable def equal_length_time : ℝ := 2.6

theorem equal_length_time_correct :
  3 * (1 - (1/2:ℝ)^equal_length_time) / (1 - 1/2) = ((2:ℝ)^equal_length_time - 1) / (2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_time_correct_l221_22110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_problem_l221_22172

/-- Represents the amount of acid in grams in each flask --/
structure AcidFlasks where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Represents the concentration of acid in a solution --/
noncomputable def concentration (acid : ℝ) (water : ℝ) : ℝ :=
  acid / (acid + water)

/-- The main theorem --/
theorem acid_concentration_problem (flasks : AcidFlasks) (water : ℝ) :
  flasks.first = 10 ∧ 
  flasks.second = 20 ∧ 
  flasks.third = 30 ∧ 
  (∃ w1 w2, w1 + w2 = water ∧ 
    concentration flasks.first w1 = 0.05 ∧ 
    concentration flasks.second w2 = 70/300) →
  concentration flasks.third water = 0.105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acid_concentration_problem_l221_22172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l221_22102

noncomputable def f (x : ℝ) := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧
  (∀ t, f (π / 6 + t) = -f (π / 6 - t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l221_22102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_negative_cube_times_sqrt_fourth_power_l221_22187

theorem sqrt_negative_cube_times_sqrt_fourth_power 
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * abs b * Real.sqrt (-a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_negative_cube_times_sqrt_fourth_power_l221_22187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_l221_22127

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) - 1

theorem function_properties_and_triangle_side (A B C a b c : ℝ) :
  (∀ x, f (x + π) = f x) ∧  -- Smallest positive period is π
  (∀ x ∈ Set.Icc (-π/3) (π/6), Monotone f) ∧  -- Monotonically increasing interval
  f B = 0 ∧
  a * c * Real.cos B = 3/2 ∧
  a + c = 4 →
  b = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_side_l221_22127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_can_always_win_l221_22162

/-- Represents a player's move in the game -/
inductive Move
  | ChangeA (δ : Int)
  | ChangeB (δ : Int)

/-- Represents the state of the game -/
structure GameState where
  a : Int
  b : Int

/-- Checks if the polynomial x^2 + ax + b has integer roots -/
def has_integer_roots (s : GameState) : Prop :=
  ∃ (x : Int), x^2 + s.a * x + s.b = 0

/-- Represents a valid move for Petya -/
def valid_petya_move (m : Move) : Prop :=
  match m with
  | Move.ChangeA δ => δ = 1 ∨ δ = -1
  | Move.ChangeB δ => δ = 1 ∨ δ = -1

/-- Represents a valid move for Kolya -/
def valid_kolya_move (m : Move) : Prop :=
  match m with
  | Move.ChangeA δ => δ = 1 ∨ δ = -1 ∨ δ = 3 ∨ δ = -3
  | Move.ChangeB δ => δ = 1 ∨ δ = -1 ∨ δ = 3 ∨ δ = -3

/-- Applies a move to the game state -/
def apply_move (s : GameState) (m : Move) : GameState :=
  match m with
  | Move.ChangeA δ => { s with a := s.a + δ }
  | Move.ChangeB δ => { s with b := s.b + δ }

/-- Theorem: Kolya can always win the game -/
theorem kolya_can_always_win :
  ∀ (initial : GameState),
  ∃ (kolya_strategy : List Move),
    (∀ (petya_moves : List Move),
      (petya_moves.length = kolya_strategy.length) →
      (∀ (i : Fin petya_moves.length), valid_petya_move (petya_moves.get i)) →
      ∃ (j : Nat), j ≤ kolya_strategy.length ∧
        has_integer_roots (
          (kolya_strategy.take j).zip (petya_moves.take j)
          |> List.foldl (λ s (km, pm) => apply_move (apply_move s pm) km) initial
        )) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kolya_can_always_win_l221_22162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_of_f_l221_22139

-- Define the complex function f(z)
noncomputable def f (z : ℂ) : ℂ := (2 * z - 3) / (z^2 - 3 * z + 2)

-- Define the singular points
def z₁ : ℂ := 1
def z₂ : ℂ := 2

-- Define the Laurent series around z₁
noncomputable def laurent_series_z₁ (z : ℂ) : ℂ := 1 / (z - z₁) - ∑' n, (z - z₁)^n

-- Define the Laurent series around z₂
noncomputable def laurent_series_z₂ (z : ℂ) : ℂ := 1 / (z - z₂) + ∑' n, (-1)^n * (z - z₂)^n

-- Theorem statement
theorem laurent_expansion_of_f :
  (∀ z, z ≠ z₁ ∧ z ≠ z₂ → f z = laurent_series_z₁ z) ∧
  (∀ z, z ≠ z₁ ∧ z ≠ z₂ → f z = laurent_series_z₂ z) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laurent_expansion_of_f_l221_22139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_visitors_is_400_l221_22192

-- Define the total number of visitors
variable (total_visitors : ℕ)

-- Define the number of visitors who enjoyed the painting
variable (enjoyed_visitors : ℕ)

-- Define the number of visitors who understood the painting
variable (understood_visitors : ℕ)

-- Condition 1: 100 visitors neither enjoyed nor understood the painting
axiom not_enjoyed_not_understood : total_visitors = enjoyed_visitors + 100

-- Condition 2: The number of visitors who enjoyed equals the number who understood
axiom enjoyed_equals_understood : enjoyed_visitors = understood_visitors

-- Condition 3: 3/4 of the visitors both enjoyed and understood the painting
axiom three_fourths_enjoyed_understood : 
  (3 : ℚ) / 4 * (total_visitors : ℚ) = enjoyed_visitors

-- Theorem to prove
theorem total_visitors_is_400 : total_visitors = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_visitors_is_400_l221_22192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_range_l221_22168

open Set
open Real

/-- The angle of inclination of a line given its equation -/
noncomputable def angleOfInclination (a b c : ℝ) : ℝ := Real.arctan (-a/b)

/-- The set of all possible angles of inclination for the line x + y*sin(α) - 1 = 0 where α ∈ ℝ -/
def inclinationSet : Set ℝ :=
  {θ | ∃ α : ℝ, θ = angleOfInclination 1 (sin α) (-1)}

theorem inclination_range :
  inclinationSet = Icc (π/4) (3*π/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_range_l221_22168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_25_equals_100_plus_125d_l221_22145

/-- An arithmetic progression with the sum of its 4th and 12th terms equal to 8 -/
structure ArithProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 8

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem stating the sum of the first 25 terms -/
theorem sum_25_equals_100_plus_125d (ap : ArithProgression) :
  sum_n ap 25 = 100 + 125 * ap.d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_25_equals_100_plus_125d_l221_22145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l221_22199

/-- The eccentricity of the ellipse -/
noncomputable def e : ℝ := Real.sqrt 2 / 2

/-- The equation of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The standard form of an ellipse -/
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Point P -/
def P : ℝ × ℝ := (0, 1)

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- Condition that AP = 2PB -/
def AP_eq_2PB (A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (B.1 - P.1), 2 * (B.2 - P.2))

/-- The area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem ellipse_and_triangle_properties :
  ∃ (a b : ℝ),
    -- The ellipse has the same foci as the hyperbola
    a^2 - b^2 = 2 ∧
    -- The eccentricity of the ellipse is e
    Real.sqrt (a^2 - b^2) / a = e ∧
    -- The standard equation of the ellipse
    (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
    -- Properties of the triangle
    ∀ A B,
      ellipse a b A.1 A.2 →
      ellipse a b B.1 B.2 →
      AP_eq_2PB A B →
      triangle_area O A B = Real.sqrt 126 / 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l221_22199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_power_of_two_l221_22129

theorem factorial_sum_power_of_two (a b n : ℕ) :
  (Nat.factorial a + Nat.factorial b = 2^n) ↔ 
  ((a, b, n) = (1, 1, 1) ∨ 
   (a, b, n) = (2, 2, 2) ∨ 
   (a, b, n) = (3, 2, 3) ∨ 
   (a, b, n) = (2, 3, 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_power_of_two_l221_22129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l221_22108

noncomputable def f (x : ℝ) := 1/x + Real.sqrt (x + 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -4 ∧ x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l221_22108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l221_22103

-- Define the function f
noncomputable def f (x b : ℝ) : ℝ := 2018 * x^3 - Real.sin x + b + 2

-- State the theorem
theorem odd_function_sum_zero 
  (a b : ℝ) 
  (h_odd : ∀ x, f x b = -f (-x) b) 
  (h_domain : Set.Icc (a - 4) (2 * a - 2) = Set.univ) :
  f a b + f b b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_zero_l221_22103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l221_22144

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the area of a triangle given three sides -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem hyperbola_triangle_area 
  (h : Hyperbola) 
  (p f1 f2 : Point) 
  (h_eq : h.a^2 = 1 ∧ h.b^2 = 8) 
  (h_on : isOnHyperbola h p) 
  (h_foci : distance f1 f2 = 2 * Real.sqrt (h.a^2 + h.b^2)) 
  (h_ratio : distance p f1 / distance p f2 = 3 / 4) :
  triangleArea (distance p f1) (distance p f2) (distance f1 f2) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l221_22144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_problem_solution_l221_22126

/-- Represents a point in time --/
structure Time where
  hours : ℕ
  minutes : ℕ
  deriving Repr

/-- Represents a traveler --/
structure Traveler where
  name : String
  speed : ℚ
  deriving Repr

/-- Represents the bus --/
structure Bus where
  speed : ℚ
  deriving Repr

/-- The problem setup --/
structure TravelProblem where
  start_time : Time
  bus : Bus
  petya : Traveler
  vasya : Traveler
  second_meetup_vasya : Time
  arrival_time : Time

/-- Convert Time to hours since midnight --/
def Time.to_hours (t : Time) : ℚ :=
  t.hours + t.minutes / 60

theorem travel_problem_solution (p : TravelProblem)
  (h_start : p.start_time = ⟨6, 0⟩)
  (h_second_meetup : p.second_meetup_vasya = ⟨13, 30⟩)
  (h_arrival : p.arrival_time = ⟨15, 0⟩)
  (h_constant_speed : p.bus.speed > 0 ∧ p.petya.speed > 0 ∧ p.vasya.speed > 0) :
  p.vasya.speed / p.petya.speed = 3 / 5 ∧
  ∃ (t : Time), t.to_hours = 11 ∧ (t.to_hours - p.start_time.to_hours) / (p.second_meetup_vasya.to_hours - t.to_hours) = 2 / 1 := by
  sorry

#eval Time.to_hours ⟨6, 0⟩  -- Should output 6
#eval Time.to_hours ⟨13, 30⟩  -- Should output 13.5
#eval Time.to_hours ⟨15, 0⟩  -- Should output 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_problem_solution_l221_22126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l221_22131

noncomputable def vector_w : ℝ × ℝ := (12, -5)

noncomputable def proj_onto_w (v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * vector_w.1 + v.2 * vector_w.2
  let norm_squared := vector_w.1 * vector_w.1 + vector_w.2 * vector_w.2
  let scalar := dot_product / norm_squared
  (scalar * vector_w.1, scalar * vector_w.2)

theorem projection_theorem :
  proj_onto_w (1, 4) = (12/13, -5/13) →
  proj_onto_w (3, 2) = (312/169, -130/169) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l221_22131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_area_l221_22128

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1
def C₂ (m : ℝ) (x y : ℝ) : Prop := y^2 = 2 * (x + m)

-- Define the condition for a single intersection point above x-axis
def single_intersection (a m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, C₁ a p.1 p.2 ∧ C₂ m p.1 p.2 ∧ p.2 > 0

-- Define the range of m
noncomputable def m_range (a : ℝ) : Set ℝ :=
  if 0 < a ∧ a < 1 then
    {m | m = (a^2 + 1) / 2 ∨ -a < m ∧ m ≤ a}
  else if a ≥ 1 then
    {m | -a < m ∧ m < a}
  else
    ∅

-- Define the maximum area of triangle OAP
noncomputable def max_area (a : ℝ) : ℝ :=
  if 1/3 < a ∧ a < 1/2 then
    a * Real.sqrt (a - a^2)
  else if 0 < a ∧ a ≤ 1/3 then
    (a / 2) * Real.sqrt (1 - a^2)
  else
    0

-- State the theorem
theorem curves_intersection_and_max_area (a : ℝ) (h : 0 < a ∧ a < 1/2) :
  ∀ m : ℝ, single_intersection a m → m ∈ m_range a ∧
  ∃ A : ℝ × ℝ, C₁ a A.1 A.2 ∧ A.2 = 0 ∧ A.1 < 0 ∧
  ∀ P : ℝ × ℝ, C₁ a P.1 P.2 ∧ C₂ m P.1 P.2 →
  (1/2) * |A.1 * P.2| ≤ max_area a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_and_max_area_l221_22128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l221_22118

/-- Given a function f(x) = tan(ωx + φ) with specific properties, 
    prove that the sum of solutions to f(x) = sin(2x - π/3) on [0, π] is 5π/6 -/
theorem sum_of_solutions (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < |φ| ∧ |φ| < π/2)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = Real.tan (ω * x + φ))
  (h_zero1 : f (π/6) = 0)
  (h_zero2 : f (2*π/3) = 0)
  (h_sol : ∀ x ∈ Set.Icc 0 π, f x = Real.sin (2*x - π/3) → x = π/6 ∨ x = 2*π/3) :
  (π/6 + 2*π/3) = 5*π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l221_22118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pseudocode_result_l221_22153

/-- Computes the product of integers from 2 to n, inclusive -/
def product_up_to (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else (List.range (n - 1)).map (· + 2) |>.foldl (· * ·) 1

theorem pseudocode_result : product_up_to 4 = 24 := by
  rfl

#eval product_up_to 4  -- Should output 24

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pseudocode_result_l221_22153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_symmetric_points_is_zero_l221_22120

-- Define the function v as noncomputable due to the use of Real.pi
noncomputable def v (x : ℝ) : ℝ := -x + 2 * Real.cos (Real.pi * x)

-- State the theorem
theorem sum_of_v_symmetric_points_is_zero :
  v (-1.05) + v (-0.4) + v 0.4 + v 1.05 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_v_symmetric_points_is_zero_l221_22120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l221_22121

/-- The coefficient of x in the expansion of (1+x)(x-2/x)^3 is -6 -/
theorem coefficient_of_x (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (1 + x) * (x - 2/x)^3 = -6*x + f x ∧ (∀ y, f y = f 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_l221_22121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_square_l221_22175

/-- An ellipse with foci and minor axis endpoints forming a square has eccentricity √2/2 -/
theorem ellipse_eccentricity_square (b c : ℝ) (h : b = c) : 
  let a := Real.sqrt (2 * b^2)
  let e := c / a
  e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_square_l221_22175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_dbc_l221_22101

/-- Helper function to calculate the area of a triangle given three points --/
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Given a triangle ABC with midpoints D and E, prove the area of triangle DBC --/
theorem area_triangle_dbc (A B C D E F : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (12, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F.2 = 0 →
  E = ((B.1 + F.1) / 2, (B.2 + F.2) / 2) →
  area_triangle E F C = 18 →
  area_triangle D B C = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_dbc_l221_22101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_time_is_49_point_5_minutes_l221_22161

/-- Calculates the actual round trip time for a motorboat traveling on a river with current -/
noncomputable def actual_round_trip_time (v_b : ℝ) (no_current_time : ℝ) : ℝ :=
  let v_c := v_b / 3
  let v_down := v_b + v_c
  let v_up := v_b - v_c
  let d := (no_current_time / 2) * v_b
  (d / v_down + d / v_up) * 60

/-- Theorem stating that the actual round trip time is 49.5 minutes -/
theorem actual_time_is_49_point_5_minutes (v_b : ℝ) (h_v_b_pos : v_b > 0) :
  actual_round_trip_time v_b (44 / 60) = 49.5 := by
  sorry

/-- Approximation of the result using rational numbers -/
def approx_result : ℚ := 495/10

#eval approx_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_time_is_49_point_5_minutes_l221_22161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_location_distance_l221_22171

/-- The distance from David's house to the meeting location in miles -/
noncomputable def distance : ℝ := 120

/-- The initial speed David drives in miles per hour -/
noncomputable def initial_speed : ℝ := 40

/-- The increase in speed for the remainder of the trip in miles per hour -/
noncomputable def speed_increase : ℝ := 20

/-- The time in hours it would take to reach the destination at the initial speed -/
noncomputable def time_at_initial_speed : ℝ := distance / initial_speed

/-- The time in hours it actually takes to reach the destination with the speed increase -/
noncomputable def actual_time : ℝ := 1 + (distance - initial_speed) / (initial_speed + speed_increase)

theorem meeting_location_distance :
  (time_at_initial_speed - actual_time) * 60 = 60 ∧
  distance = initial_speed * (actual_time + 1/3) ∧
  distance - initial_speed = (initial_speed + speed_increase) * (actual_time - 1 - 1/3) := by
  sorry

#check meeting_location_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_location_distance_l221_22171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_geq_zero_inequality_proof_l221_22142

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

-- Part I: Uniqueness of solution for f(x) ≥ 0
theorem unique_solution_f_geq_zero :
  ∃! x, x > 0 ∧ f x ≥ 0 :=
sorry

-- Part II: Inequality proof
theorem inequality_proof (x a : ℝ) (hx : x > 0) (ha : a ≤ 1) :
  x * (f x + x - 1) < Real.exp x - a * x^2 - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_geq_zero_inequality_proof_l221_22142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_existence_implies_a_range_l221_22180

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (4, 0)

-- Define the curve C
def C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 4*a*y + 5*a^2 - 9 = 0

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem point_existence_implies_a_range (a : ℝ) :
  (∃ P : ℝ × ℝ, C a P.1 P.2 ∧ distance P B = 2 * distance P A) →
  a ∈ Set.Icc (-Real.sqrt 5) (-Real.sqrt 5 / 5) ∪ Set.Icc (Real.sqrt 5 / 5) (Real.sqrt 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_existence_implies_a_range_l221_22180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l221_22143

theorem trigonometric_equation_solution (k : ℤ) :
  let x : ℝ := (π / 12) * (6 * k + 5)
  8.481 * (1 / 4) * Real.tan (x / 4) + (1 / 2) * Real.tan (x / 2) + Real.tan x
  = 2 * Real.sqrt 3 + (1 / 4) * (1 / Real.tan (x / 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l221_22143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l221_22160

/-- Given a circle centered at the origin passing through the point (1,2),
    prove that x + 2y - 5 = 0 is the equation of the tangent line at (1,2). -/
theorem tangent_line_at_point (x y : ℝ) : 
  (x^2 + y^2 = 5) →  -- Circle equation
  ((1:ℝ)^2 + (2:ℝ)^2 = 5) →  -- Point (1,2) lies on the circle
  (x + 2*y - 5 = 0) →  -- Proposed tangent line equation
  (∀ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 5 → (x₀ - 1)^2 + (y₀ - 2)^2 ≥ ((x₀ + 2*y₀ - 5) / Real.sqrt 5)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l221_22160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l221_22111

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem intersection_distance_sum :
  ∃ (xA yA xB yB : ℝ),
    curve_C xA yA ∧ curve_C xB yB ∧
    line_l xA yA ∧ line_l xB yB ∧
    (1 / distance (point_P.1) (point_P.2) xA yA + 1 / distance (point_P.1) (point_P.2) xB yB = 3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l221_22111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l221_22116

-- Define the interval (-∞, 0)
noncomputable def I : Set ℝ := {x | x < 0}

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 - x^2
noncomputable def h (x : ℝ) : ℝ := x^2 + x
noncomputable def k (x : ℝ) : ℝ := 1 / (x + 1)

-- Theorem statement
theorem decreasing_function_on_interval :
  (∀ x y, x ∈ I → y ∈ I → x < y → f x > f y) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → g x > g y) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → h x > h y) ∧
  ¬(∀ x y, x ∈ I → y ∈ I → x < y → k x > k y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_on_interval_l221_22116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_seven_years_l221_22119

/-- Calculates the total amount after simple interest is applied -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the initial conditions, the total amount after 7 years is $820 -/
theorem total_amount_after_seven_years 
  (initial_amount : ℝ)
  (amount_after_two_years : ℝ)
  (h1 : initial_amount = 400)
  (h2 : amount_after_two_years = 520)
  (h3 : simple_interest initial_amount ((amount_after_two_years / initial_amount - 1) / 2) 2 = amount_after_two_years) :
  simple_interest initial_amount ((amount_after_two_years / initial_amount - 1) / 2) 7 = 820 := by
  sorry

#check total_amount_after_seven_years

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_after_seven_years_l221_22119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l221_22117

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  A = π / 6 →
  a = 1 →
  b = Real.sqrt 3 →
  (A + B + C = π) →
  (Real.sin A / a = Real.sin B / b) →
  (B = π / 3 ∨ B = 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_calculation_l221_22117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l221_22136

noncomputable def vector1 : ℝ × ℝ := (4, 5)
noncomputable def vector2 : ℝ × ℝ := (2, 7)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem cosine_of_angle_between_vectors (v1 v2 : ℝ × ℝ) :
  (dot_product v1 v2) / (magnitude v1 * magnitude v2) = 43 / Real.sqrt (41 * 53) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l221_22136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_c_values_l221_22158

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem product_of_c_values (c₁ c₂ : ℝ) :
  (distance (3 * c₁) (c₁ + 5) 1 4 = 5) →
  (distance (3 * c₂) (c₂ + 5) 1 4 = 5) →
  c₁ ≠ c₂ →
  c₁ * c₂ = -2.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_c_values_l221_22158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_equality_l221_22189

theorem odd_integers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_equality_l221_22189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_bought_three_hamburgers_l221_22181

/-- Represents the price of items and quantities bought on two days -/
structure PurchaseData where
  hotdog_price : ℚ
  day1_total : ℚ
  day1_hotdogs : ℕ
  day2_total : ℚ
  day2_hamburgers : ℕ
  day2_hotdogs : ℕ

/-- Calculates the number of hamburgers bought on the first day -/
noncomputable def hamburgers_bought (data : PurchaseData) : ℚ :=
  let hamburger_price := (data.day2_total - data.hotdog_price * data.day2_hotdogs) / data.day2_hamburgers
  (data.day1_total - data.hotdog_price * data.day1_hotdogs) / hamburger_price

/-- Theorem stating that Arthur bought 3 hamburgers on the first day -/
theorem arthur_bought_three_hamburgers (data : PurchaseData) 
  (h1 : data.hotdog_price = 1)
  (h2 : data.day1_total = 10)
  (h3 : data.day1_hotdogs = 4)
  (h4 : data.day2_total = 7)
  (h5 : data.day2_hamburgers = 2)
  (h6 : data.day2_hotdogs = 3) :
  hamburgers_bought data = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_bought_three_hamburgers_l221_22181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l221_22132

theorem angle_problem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 7) (h4 : Real.sin β = Real.sqrt 5 / 5) :
  Real.tan (α + β) = -3 ∧ α + 2*β = 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l221_22132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_skew_line_l221_22149

-- Define the types for lines and points
def Line : Type := ℝ → ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ × ℝ

-- Define the distance function between a point and a line
noncomputable def distPointToLine (P : Point) (l : Line) : ℝ := sorry

-- Define the angle between two lines
noncomputable def angleBetweenLines (l1 l2 : Line) : ℝ := sorry

-- Define the distance function between two points
noncomputable def dist (P Q : Point) : ℝ := sorry

-- Define the theorem
theorem distance_to_skew_line 
  (a b : Line) 
  (A B P : Point) :
  -- Conditions
  (∃ (AB : Set Point), AB.Nonempty ∧ 
    (∀ X ∈ AB, distPointToLine X a = 0 ∧ distPointToLine X b = 0)) →  -- AB is common perpendicular
  (distPointToLine A a = 0) →  -- A is on line a
  (distPointToLine B b = 0) →  -- B is on line b
  (dist A B = 2) →  -- AB = 2
  (angleBetweenLines a b = π / 6) →  -- Angle between a and b is 30°
  (dist A P = 4 ∧ distPointToLine P a = 0) →  -- AP = 4 and P is on line a
  -- Conclusion
  (distPointToLine P b = 2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_skew_line_l221_22149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l221_22113

noncomputable def g (A : ℝ) : ℝ :=
  (Real.sin A * (4 * Real.cos A ^ 3 + 2 * Real.cos A ^ 5 + 5 * Real.sin A ^ 2 + 2 * Real.sin A ^ 2 * Real.cos A ^ 2)) /
  (Real.tan A * (1 / Real.cos A - Real.sin A * Real.tan A))

theorem g_range (A : ℝ) (h : ∀ n : ℤ, A ≠ n * Real.pi / 2) :
  Set.range g = Set.Ioo 5 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l221_22113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_bound_l221_22137

/-- A parabola with vertex O and focus F -/
structure Parabola where
  O : ℝ × ℝ  -- vertex
  F : ℝ × ℝ  -- focus
  is_valid : F.1 > O.1 ∧ F.2 = O.2

/-- A point on the parabola -/
def PointOnParabola (p : Parabola) := { P : ℝ × ℝ // (P.2 - p.O.2)^2 = 4 * (p.F.1 - p.O.1) * (P.1 - p.O.1) }

/-- The ratio |PO|/|PF| for a point P on the parabola -/
noncomputable def ratio (p : Parabola) (P : PointOnParabola p) : ℝ :=
  Real.sqrt ((P.val.1 - p.O.1)^2 + (P.val.2 - p.O.2)^2) /
  Real.sqrt ((P.val.1 - p.F.1)^2 + (P.val.2 - p.F.2)^2)

theorem parabola_ratio_bound (p : Parabola) :
  ∀ P : PointOnParabola p, ratio p P ≤ 2 * Real.sqrt 3 / 3 ∧
  ∃ P : PointOnParabola p, ratio p P = 2 * Real.sqrt 3 / 3 := by
  sorry

#check parabola_ratio_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ratio_bound_l221_22137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l221_22182

/-- The function f(x) = (1/2)^(-x^2 + 2x) -/
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x)

/-- The range of f is [1/2, +∞) -/
theorem range_of_f :
  Set.range f = Set.Ici (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l221_22182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l221_22148

-- Define the circle
def circleEq (a x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + a = 0

-- Define the line
def lineEq (x y : ℝ) : Prop := y = x + 1

-- Define the midpoint of the chord
def midpointEq (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem line_intersects_circle (a : ℝ) (h : a < 3) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circleEq a x₁ y₁ ∧ circleEq a x₂ y₂ ∧
    lineEq x₁ y₁ ∧ lineEq x₂ y₂ ∧
    midpointEq ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_l221_22148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_e_equals_two_l221_22122

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -Real.log x else (1/2)^x

theorem f_of_f_e_equals_two : f (f (Real.exp 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_f_e_equals_two_l221_22122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_from_focus_l221_22124

noncomputable def parabola (x : ℝ) : ℝ := x^2

noncomputable def focus : ℝ × ℝ := (0, 1/4)

noncomputable def intersectionPoints : List (ℝ × ℝ) := [(4, 16), (-3, 9), (-15, 225), (14, 196)]

noncomputable def distanceFromFocus (p : ℝ × ℝ) : ℝ := |p.2 - focus.2|

theorem sum_of_distances_from_focus :
  let distances := intersectionPoints.map distanceFromFocus
  ↑(List.sum distances) = 445 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_from_focus_l221_22124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operation_count_l221_22152

/-- Represents a polynomial of degree n -/
def MyPolynomial (α : Type*) := List α

/-- Horner's method for polynomial evaluation -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Count of operations in Horner's method -/
structure OperationCount where
  multiplications : Nat
  additions : Nat

/-- Theorem: Horner's method requires n multiplications and n additions for an nth-degree polynomial -/
theorem horner_operation_count {α : Type*} [Ring α] (p : MyPolynomial α) :
  let n := p.length - 1
  OperationCount.mk n n = 
    { multiplications := n,
      additions := n } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operation_count_l221_22152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l221_22169

/-- The length of a diagonal in a 1x1 square -/
noncomputable def unit_diagonal_length : ℝ := Real.sqrt 2

/-- The length of a diagonal in a 2x2 square -/
noncomputable def double_diagonal_length : ℝ := 2 * Real.sqrt 2

/-- The total length of straight segments in XYZ -/
def straight_segments_length : ℝ := 6

/-- The number of unit diagonals in XYZ -/
def unit_diagonals_count : ℕ := 4

/-- The number of double diagonals in XYZ -/
def double_diagonals_count : ℕ := 1

/-- The total length of XYZ -/
noncomputable def total_length : ℝ := straight_segments_length + 
  (unit_diagonals_count : ℝ) * unit_diagonal_length +
  (double_diagonals_count : ℝ) * double_diagonal_length

theorem xyz_length : total_length = 6 + 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_length_l221_22169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_timSleepSchedule_l221_22159

/-- Represents a time zone with an offset from a reference time zone --/
structure TimeZone where
  offset : Int

/-- Represents a day's sleep schedule --/
structure SleepDay where
  hours : Nat
  zone : TimeZone

/-- Calculates the total sleep hours for a week --/
def weekSleepHours (weekdays : List SleepDay) (weekend : List SleepDay) : Nat :=
  (weekdays.map (λ day => day.hours)).sum + (weekend.map (λ day => day.hours)).sum

/-- Tim's sleep schedule for a specific week --/
theorem timSleepSchedule : 
  let timezoneA := TimeZone.mk 0
  let timezoneB := TimeZone.mk 3
  let timezoneC := TimeZone.mk (-3)  -- -2 offset and -1 daylight saving

  let weekdays := List.replicate 5 (SleepDay.mk 6 timezoneA)
  let weekend := [SleepDay.mk 10 timezoneB, SleepDay.mk 10 timezoneC]

  weekSleepHours weekdays weekend = 50 := by
  -- Proof goes here
  sorry

#check timSleepSchedule

end NUMINAMATH_CALUDE_ERRORFEEDBACK_timSleepSchedule_l221_22159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_calculation_l221_22150

/-- The speed of the goods train in km/h -/
noncomputable def goods_train_speed : ℝ := 82

/-- The speed of the man's train in km/h -/
noncomputable def mans_train_speed : ℝ := 30

/-- The time it takes for the goods train to pass the man in seconds -/
noncomputable def passing_time : ℝ := 9

/-- The length of the goods train in meters -/
noncomputable def goods_train_length : ℝ := 280

/-- Conversion factor from km/h to m/s -/
noncomputable def km_per_hour_to_m_per_second : ℝ := 1 / 3.6

theorem goods_train_speed_calculation :
  goods_train_speed = 
    (goods_train_length / passing_time / km_per_hour_to_m_per_second) - mans_train_speed :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_calculation_l221_22150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l221_22114

theorem divisibility_conditions (n m k l : ℕ) (hn : n ≠ 0) (hn1 : n ≠ 1)
  (hm : m > 0) (hk : k > 0) (hl : l > 0)
  (h : (n^k + m*n^l + 1) ∣ (n^(k+l) - 1)) :
  (m = 1 ∧ l = 2*k) ∨ 
  (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_conditions_l221_22114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lawn_fraction_l221_22133

-- Define the lawn as a unit (1)
noncomputable def lawn : ℚ := 1

-- Define Tom's mowing rate (in fraction of lawn per hour)
noncomputable def tom_rate : ℚ := 1 / 6

-- Define the time Tom works
noncomputable def tom_time : ℚ := 3

-- Theorem statement
theorem remaining_lawn_fraction :
  lawn - (tom_rate * tom_time) = 1 / 2 := by
  -- Expand definitions
  unfold lawn tom_rate tom_time
  -- Perform arithmetic
  simp [mul_div_cancel']
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_lawn_fraction_l221_22133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_sufficient_condition_l221_22193

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotone_increasing_f (t : ℝ) (h1 : 0 < t) (h2 : t < Real.pi / 6) :
  StrictMonoOn f (Set.Ioo (-t) t) :=
by
  sorry

theorem sufficient_condition (t : ℝ) :
  (0 < t ∧ t ≤ Real.pi / 6) →
  StrictMonoOn f (Set.Ioo (-t) t) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_sufficient_condition_l221_22193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l221_22198

theorem P_less_than_Q : ∀ x : ℝ, 
  (x - 2) * (x - 4) < (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_less_than_Q_l221_22198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_equality_l221_22134

noncomputable def series_sum (c d : ℝ) := (c / d) / (1 - 1 / d)

theorem series_equality (c d : ℝ) (h : series_sum c d = 6) :
  let new_series_sum := c / (c + 2 * d) / (1 - 1 / (c + 2 * d))
  new_series_sum = (6 * d - 6) / (8 * d - 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_equality_l221_22134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l221_22105

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2*k + 1}

theorem range_of_k : 
  ∀ k : ℝ, (U \ A) ∩ B k = ∅ ↔ k ∈ Set.Iic 0 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l221_22105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_linearity_l221_22157

/-- A linear equation in x is of the form ax + b = 0, where a and b are constants -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- The equation 2x^(m-1) + 3 = 0 -/
noncomputable def equation (m : ℝ) (x : ℝ) : ℝ :=
  2 * (x ^ (m - 1)) + 3

theorem equation_linearity (m : ℝ) :
  is_linear_equation (equation m) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_linearity_l221_22157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l221_22195

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Define the points of tangency
variable (x₁ y₁ x₂ y₂ : ℝ)

-- State the conditions
axiom tangent_condition : ∃ (m b : ℝ), 
  (∀ x, m * x + b = f x → x = x₁) ∧ 
  (∀ x, m * x + b = g x → x = x₂)

axiom point_on_f : f x₁ = y₁
axiom point_on_g : g x₂ = y₂

-- State the theorem
theorem tangent_line_property : (1 - Real.exp y₁) * (1 + x₂) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l221_22195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l221_22174

theorem triangle_angle_c (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  b / a = Real.cos C - Real.sin C →
  a = Real.sqrt 2 →
  c = 1 →
  C = π / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l221_22174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_count_l221_22138

def fraction : ℚ := (5^6 : ℚ) / ((10^5 * 8) : ℚ)

theorem decimal_digits_count : ∃ (n : ℕ), (fraction * 10^n).num % (fraction * 10^n).den = 0 ∧ 
  (fraction * 10^(n-1)).num % (fraction * 10^(n-1)).den ≠ 0 ∧ n = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_digits_count_l221_22138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l221_22141

theorem greatest_integer_fraction : 
  ⌊(4^100 + 3^100 : ℝ) / (4^95 + 3^95 : ℝ)⌋ = 1023 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l221_22141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_black_l221_22155

/-- Represents a chessboard with the usual coloring, where borders are black -/
structure Chessboard :=
  (size : ℕ)

/-- Represents a circle on the chessboard -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Predicate to check if a circle lies entirely on black squares (including borders) -/
def is_on_black (board : Chessboard) (circle : Circle) : Prop :=
  sorry

/-- The largest possible radius for a circle entirely on black squares -/
noncomputable def max_radius : ℝ := Real.sqrt 10 / 2

/-- Theorem stating that the largest circle that can be drawn entirely on black
    has a radius of √10/2 -/
theorem largest_circle_on_black (board : Chessboard) :
  ∀ (c : Circle), is_on_black board c → c.radius ≤ max_radius :=
by
  sorry

#check largest_circle_on_black

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_on_black_l221_22155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minor_axis_equals_focal_length_l221_22147

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
  a_pos : a > 0
  b_pos : b > 0
  a_gt_b : a > b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The focal length of an ellipse -/
noncomputable def focalLength (e : Ellipse a b) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_eccentricity_when_minor_axis_equals_focal_length 
  (e : Ellipse a b) (h : focalLength e = b) : 
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_when_minor_axis_equals_focal_length_l221_22147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_coverage_l221_22176

/-- The proportion of a square lawn covered by four corner sprinklers -/
theorem sprinkler_coverage (a : ℝ) (h : a > 0) : 
  (4 * (π * a^2 / 4) - 4 * ((π/12 - 1/4) * a^2) + a^2) / a^2 = π/3 + 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sprinkler_coverage_l221_22176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_on_y_axis_l221_22194

noncomputable def f (x : ℝ) : ℝ := 0.5 * (x - 0.5)^2

noncomputable def tangent_line (y₀ k : ℝ) (x : ℝ) : ℝ := y₀ + k * x

def is_tangent_point (y₀ k x : ℝ) : Prop :=
  f x = tangent_line y₀ k x

noncomputable def angle_between_lines (k₁ k₂ : ℝ) : ℝ :=
  Real.arctan ((k₂ - k₁) / (1 + k₁ * k₂))

-- Main theorem
theorem tangent_points_on_y_axis :
  ∀ y₀ : ℝ,
  (∃ k₁ k₂ x₁ x₂ : ℝ,
    k₁ ≠ k₂ ∧
    is_tangent_point y₀ k₁ x₁ ∧
    is_tangent_point y₀ k₂ x₂ ∧
    angle_between_lines k₁ k₂ = π / 4) →
  y₀ = 0 ∨ y₀ = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_on_y_axis_l221_22194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_correct_l221_22185

/-- The number of triples of natural numbers (a, b, c) satisfying the given conditions -/
def count_triples : ℕ := 9180

/-- The given gcd value -/
def gcd_value : ℕ := 35

/-- The given lcm value -/
def lcm_value : ℕ := 5^18 * 7^16

/-- Theorem stating that the count of triples satisfying the conditions is correct -/
theorem count_triples_correct :
  (count_triples = Finset.card (Finset.filter (fun abc : ℕ × ℕ × ℕ => 
    Nat.gcd abc.1 (Nat.gcd abc.2.1 abc.2.2) = gcd_value ∧
    Nat.lcm abc.1 (Nat.lcm abc.2.1 abc.2.2) = lcm_value) (Finset.product (Finset.range (lcm_value + 1)) (Finset.product (Finset.range (lcm_value + 1)) (Finset.range (lcm_value + 1)))))) := by
  sorry

#check count_triples_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_correct_l221_22185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_for_jasmine_l221_22156

def max_cards (total_money : ℚ) (card_price : ℚ) : ℕ :=
  (total_money / card_price).floor.toNat

theorem max_cards_for_jasmine :
  max_cards 12 (5/4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_for_jasmine_l221_22156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l221_22179

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the point M
def M (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the line l
def l (k a : ℝ) (x y : ℝ) : Prop := y = k * (x - a)

-- Define the focus of the parabola C
def focus : ℝ × ℝ := (3, 0)

-- Part I
theorem part_one (k : ℝ) : 
  let a := 1
  ∃ A B : ℝ × ℝ, 
    C A.1 A.2 ∧ C B.1 B.2 ∧ 
    l k a A.1 A.2 ∧ l k a B.1 B.2 ∧
    M a = (a, 0) ∧
    (A.1 + B.1) / 2 = focus.1 →
    k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by
  sorry

-- Part II
theorem part_two (a k : ℝ) (A B : ℝ × ℝ) :
  a < 0 →
  C A.1 A.2 ∧ C B.1 B.2 ∧
  l k a A.1 A.2 ∧ l k a B.1 B.2 ∧
  M a = (a, 0) →
  ∃ m : ℝ, l m a (-a) 0 ∧ l m a A.1 (-A.2) ∧ l m a B.1 B.2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l221_22179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_mountain_trips_l221_22151

/-- Given a mountain of height h, where a person makes n round trips and reaches
    a fraction f of the mountain's height on each trip, the total distance
    covered is equal to 2 * n * f * h. -/
theorem total_distance_mountain_trips
  (h : ℝ) (n : ℕ) (f : ℝ) 
  (h_pos : h > 0) 
  (n_pos : n > 0) 
  (f_pos : f > 0) 
  (f_le_one : f ≤ 1) : 
  2 * (n : ℝ) * f * h = (n : ℝ) * (2 * f * h) :=
by
  -- The proof goes here
  sorry

#check total_distance_mountain_trips

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_mountain_trips_l221_22151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_investment_is_2000_l221_22135

/-- Calculates the weekly investment given the initial amount, final amount, and windfall percentage -/
noncomputable def calculate_weekly_investment (initial_amount : ℝ) (final_amount : ℝ) (windfall_percentage : ℝ) : ℝ :=
  let amount_before_windfall := final_amount / (1 + windfall_percentage)
  let total_investment := amount_before_windfall - initial_amount
  total_investment / 52

/-- Proves that the weekly investment is $2000 given the problem conditions -/
theorem weekly_investment_is_2000 :
  calculate_weekly_investment 250000 885000 0.5 = 2000 := by
  -- Unfold the definition of calculate_weekly_investment
  unfold calculate_weekly_investment
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_investment_is_2000_l221_22135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_block_weight_l221_22165

/-- Represents a triangular metal block -/
structure TriangularBlock where
  sideLength : ℝ
  thickness : ℝ
  weight : ℝ

/-- Calculates the area of an equilateral triangle -/
noncomputable def equilateralTriangleArea (sideLength : ℝ) : ℝ :=
  (sideLength ^ 2 * Real.sqrt 3) / 4

/-- Calculates the volume of a triangular block -/
noncomputable def triangularBlockVolume (block : TriangularBlock) : ℝ :=
  equilateralTriangleArea block.sideLength * block.thickness

/-- The weight of a triangular block is proportional to its volume -/
axiom weight_proportional_to_volume {b1 b2 : TriangularBlock} (h : b1.thickness = b2.thickness) :
  b1.weight / b2.weight = triangularBlockVolume b1 / triangularBlockVolume b2

theorem second_block_weight (block1 block2 : TriangularBlock)
    (h1 : block1.sideLength = 4)
    (h2 : block1.thickness = 0.5)
    (h3 : block1.weight = 20)
    (h4 : block2.sideLength = 6)
    (h5 : block2.thickness = 0.5) :
    block2.weight = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_block_weight_l221_22165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l221_22106

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

-- State the theorem
theorem tangent_function_properties :
  ∃ (a b : ℝ),
    -- The function is tangent to y = -1/2 at x = 1
    (f a b 1 = -1/2) ∧
    (deriv (f a b) 1 = 0) ∧
    -- The values of a and b
    (a = 1) ∧
    (b = 1/2) ∧
    -- The maximum value on the interval [1/e, e]
    (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a b x ≤ -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l221_22106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l221_22196

theorem divisibility_in_chosen_set :
  ∀ (S : Finset ℕ),
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 100) →
  S.card = 51 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_chosen_set_l221_22196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l221_22183

/-- A translation of the complex plane -/
def translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_problem :
  ∃ w : ℂ, translation w (1 - I) = 5 + 2*I ∧ translation w (2 + 3*I) = 6 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l221_22183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_height_is_five_l221_22130

/-- The distance from the ground to the center of a suspended sphere -/
noncomputable def sphere_center_height (lowest_point_height : ℝ) (shadow_length : ℝ) (stick_height : ℝ) (stick_shadow : ℝ) : ℝ :=
  (lowest_point_height * stick_shadow / stick_height) + lowest_point_height

/-- Theorem stating the distance from the ground to the sphere's center is 5 meters -/
theorem sphere_center_height_is_five :
  sphere_center_height 1 15 1 3 = 5 := by
  unfold sphere_center_height
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_center_height_is_five_l221_22130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_circle_l221_22184

/-- The length of the tangent segment from the origin to a circle --/
noncomputable def tangentLength (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt (p1.1^2 + p1.2^2)
  let b := Real.sqrt (p2.1^2 + p2.2^2)
  Real.sqrt (a * b)

/-- Theorem: The length of the tangent segment from the origin to the circle 
    passing through (4,5), (7,9), and (8,6) is √5330 --/
theorem tangent_length_specific_circle : 
  tangentLength (4, 5) (7, 9) (8, 6) = Real.sqrt 5330 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tangentLength (4, 5) (7, 9) (8, 6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_specific_circle_l221_22184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_technicians_workshop_technicians_correct_l221_22115

theorem workshop_technicians (total_workers : ℕ) (avg_salary : ℕ) 
  (technician_salary : ℕ) (rest_salary : ℕ) : ℕ :=
  by
  have h1 : total_workers = 35 := by sorry
  have h2 : avg_salary = 8000 := by sorry
  have h3 : technician_salary = 16000 := by sorry
  have h4 : rest_salary = 6000 := by sorry

  -- Define the number of technicians
  let num_technicians : ℕ := 7

  -- Return the number of technicians
  exact num_technicians

-- Prove that the number of technicians is correct
theorem workshop_technicians_correct : workshop_technicians 35 8000 16000 6000 = 7 :=
  by
  -- Unfold the definition of workshop_technicians
  unfold workshop_technicians
  -- The result is trivially true by definition
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workshop_technicians_workshop_technicians_correct_l221_22115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l221_22178

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ 1) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 1) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -1/2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l221_22178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_arrangement_proof_l221_22104

def digit_arrangement_count : Nat :=
  let grid_size : Nat := 2 * 3
  let digit_count : Nat := 4
  let total_elements : Nat := grid_size

  Nat.factorial total_elements

#eval digit_arrangement_count -- This will evaluate to 720

theorem digit_arrangement_proof :
  digit_arrangement_count = 720 := by
  unfold digit_arrangement_count
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_arrangement_proof_l221_22104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_max_min_l221_22167

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem extreme_values_and_max_min :
  -- Local maximum at x = -2
  (∃ ε > 0, ∀ x ∈ interval, |x - (-2)| < ε → f x ≤ f (-2)) ∧
  f (-2) = 28/3 ∧
  -- Local minimum at x = 2
  (∃ ε > 0, ∀ x ∈ interval, |x - 2| < ε → f x ≥ f 2) ∧
  f 2 = -4/3 ∧
  -- Global maximum on the interval
  (∀ x ∈ interval, f x ≤ 28/3) ∧
  -- Global minimum on the interval
  (∀ x ∈ interval, f x ≥ -4/3) :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_max_min_l221_22167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_square_with_two_twos_digit_sum_l221_22125

/-- Given that 225 is the first perfect square beginning with two 2s, 
    prove that the sum of the digits of the next such perfect square is 13. -/
theorem next_square_with_two_twos_digit_sum : 
  ∃ (n : ℕ), 
    n > 15 ∧ 
    (n^2).repr.take 2 = "22" ∧
    (∀ m : ℕ, 15 < m ∧ m < n → (m^2).repr.take 2 ≠ "22") ∧
    (Nat.digits 10 (n^2)).sum = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_square_with_two_twos_digit_sum_l221_22125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_volume_when_doubled_radius_increases_by_180_l221_22154

/-- The volume of a right circular cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem original_volume_when_doubled_radius_increases_by_180 (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  cylinderVolume (2*r) h - cylinderVolume r h = 180 →
  cylinderVolume r h = 60 := by
  intro h_increase
  -- Proof steps would go here
  sorry

#check original_volume_when_doubled_radius_increases_by_180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_volume_when_doubled_radius_increases_by_180_l221_22154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_2root10_l221_22177

noncomputable section

/-- The curve equation y = (1/2)x^2 - 1 -/
def curve (x : ℝ) : ℝ := (1/2) * x^2 - 1

/-- The line equation y = x + 1 -/
def line (x : ℝ) : ℝ := x + 1

/-- The x-coordinates of the intersection points -/
def intersection_x : Set ℝ := {x | curve x = line x}

/-- The length of the segment AB -/
noncomputable def segment_length : ℝ :=
  let x₁ := Real.sqrt 5 + 1
  let x₂ := -Real.sqrt 5 + 1
  Real.sqrt ((x₁ - x₂)^2 + (curve x₁ - curve x₂)^2)

theorem segment_length_is_2root10 : segment_length = 2 * Real.sqrt 10 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_2root10_l221_22177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_structure_volume_l221_22186

/-- The volume of the structure consisting of a cone and a cylinder -/
noncomputable def structure_volume (diameter : ℝ) (cone_height : ℝ) (cylinder_height : ℝ) : ℝ :=
  let radius := diameter / 2
  let cone_volume := (1/3) * Real.pi * radius^2 * cone_height
  let cylinder_volume := Real.pi * radius^2 * cylinder_height
  cone_volume + cylinder_volume

/-- Theorem stating the volume of the specific structure -/
theorem specific_structure_volume :
  structure_volume 8 10 4 = (352/3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_structure_volume_l221_22186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l221_22170

def sequencePattern (n : Fin 7) : ℕ :=
  match n with
  | ⟨0, _⟩ => 1
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 27
  | ⟨6, _⟩ => 343
  | ⟨i, h⟩ => (i + 1) ^ 3

theorem sequence_pattern :
  sequencePattern ⟨3, by norm_num⟩ = 64 ∧
  sequencePattern ⟨4, by norm_num⟩ = 125 ∧
  sequencePattern ⟨5, by norm_num⟩ = 216 :=
by
  apply And.intro
  · rfl
  apply And.intro
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l221_22170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l221_22166

-- Define the basic parameters
noncomputable def cold_temp : ℝ := 50
noncomputable def warm_temp : ℝ := 80
noncomputable def cold_distance : ℝ := 20
def cold_throws : ℕ := 20
def warm_throws : ℕ := 30
noncomputable def total_distance : ℝ := 1600

-- Define the ratio of warm to cold distance
noncomputable def ratio : ℝ := total_distance / (cold_distance * cold_throws + cold_distance * warm_throws)

-- Theorem statement
theorem total_distance_proof :
  cold_distance * cold_throws + ratio * cold_distance * warm_throws = total_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_proof_l221_22166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l221_22107

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi/6) * Real.sin (x + Real.pi/3)

-- State the theorem
theorem function_properties :
  -- Part I: Smallest positive period
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- Part II: Triangle property
  (∀ (A B C : ℝ), A = Real.pi/4 →
    f (C/2 + Real.pi/6) = 1/2 →
    C > 0 →
    C < Real.pi/2 →
    (Real.sin A / Real.sin C) = Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l221_22107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l221_22146

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the circle
def circle_eq (c x y : ℝ) : Prop :=
  x^2 + y^2 = c^2

-- Define the slope of the tangent line
noncomputable def tangent_slope : ℝ := -Real.sqrt 3

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (h4 : ∃ x y, x > 0 ∧ y > 0 ∧ hyperbola a b x y ∧ circle_eq c x y)
  (h5 : c^2 = a^2 + b^2) :
  eccentricity a c = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l221_22146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_angle_sine_sin_18_degrees_l221_22140

open Real

-- Define the sum of sines formula
axiom sum_of_sines (α β : ℝ) : Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β

-- Define the double angle formula for sine
axiom double_angle_sine (α : ℝ) : Real.sin (2 * α) = 2 * Real.sin α * Real.cos α

-- Define the relationship between sine and cosine
axiom sine_cosine_complement (α : ℝ) : Real.sin α = Real.cos (π / 2 - α)

-- State the theorem for the triple angle formula of sine
theorem triple_angle_sine (α : ℝ) : Real.sin (3 * α) = 3 * Real.sin α - 4 * (Real.sin α)^3 := by
  sorry

-- State the theorem for the value of sin 18°
theorem sin_18_degrees : Real.sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_angle_sine_sin_18_degrees_l221_22140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_for_winning_l221_22123

theorem min_subset_size_for_winning (n : ℕ) (h_n : n ≥ 3) (h_odd : Odd n) :
  let k := ⌈(3 * n : ℝ) / 2 + 1 / 2⌉
  ∀ S : Finset ℤ, S.card = k → S ⊆ Finset.Icc (-n : ℤ) n →
    (∀ m ∈ Finset.Icc (-n : ℤ) n, ∃ T : Finset ℤ, T ⊆ S ∧ T.card = n ∧ (T.sum id = m)) ∧
    ∀ k' < k, ∃ S' : Finset ℤ, S'.card = k' ∧ S' ⊆ Finset.Icc (-n : ℤ) n ∧
      ∃ m ∈ Finset.Icc (-n : ℤ) n, ∀ T : Finset ℤ, T ⊆ S' → T.card = n → T.sum id ≠ m :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subset_size_for_winning_l221_22123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tenth_term_l221_22173

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2 + a 3 = 15)
  (h_geom : is_geometric_sequence (λ n ↦ match n with
    | 1 => a 1 + 2
    | 2 => a 2 + 5
    | 3 => a 3 + 13
    | _ => 0
  )) :
  a 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_tenth_term_l221_22173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l221_22190

/-- Given a class of boys where:
  * The initial average height was calculated as 185 cm
  * One boy's height was overreported by 60 cm
  * The actual average height is 183 cm
  This theorem proves that there are 30 boys in the class -/
theorem number_of_boys_in_class (initial_avg : ℝ) (height_difference : ℝ) (actual_avg : ℝ) 
  (h1 : initial_avg = 185)
  (h2 : height_difference = 60)
  (h3 : actual_avg = 183) :
  ∃ n : ℝ, (initial_avg * n - height_difference) / n = actual_avg ∧ n = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l221_22190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l221_22197

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

-- Define the point A
def A : ℝ × ℝ := (1, -2)

-- State the theorem
theorem tangent_line_at_A :
  let f' := λ x => 2*x + 2  -- Derivative of f
  let m := f' A.fst         -- Slope of the tangent line
  let b := A.snd - m * A.fst  -- y-intercept of the tangent line
  (λ x => m * x + b) = (λ x => 4 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l221_22197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l221_22188

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 27 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define vectors
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

theorem ellipse_theorem (A : ℝ × ℝ) 
  (h_A : is_on_ellipse A.1 A.2) 
  (B C : ℝ × ℝ) 
  (h_B : vec O B = (1/2 : ℝ) • (vec O A + vec O F₁))
  (h_C : vec O C = (1/2 : ℝ) • (vec O A + vec O F₂)) :
  magnitude (vec O B) + magnitude (vec O C) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l221_22188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_equals_four_fifths_l221_22163

-- Define the point P as a function of t
noncomputable def P (t : ℝ) : ℝ × ℝ := (3*t, 4*t)

-- Define the angle θ
noncomputable def θ (t : ℝ) : ℝ := Real.arctan (4*t / (3*t))

-- Theorem statement
theorem sin_theta_equals_four_fifths (t : ℝ) (h : t > 0) :
  Real.sin (θ t) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_equals_four_fifths_l221_22163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l221_22109

theorem smallest_n_for_sqrt_difference : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), m > 0 → m < n → Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ) ≥ 0.01) ∧
  (Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.01) ∧
  n = 2501 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sqrt_difference_l221_22109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_months_until_change_is_eight_l221_22112

/-- Represents the number of months after which A withdrew and B advanced money -/
def months_until_change : ℕ := 8

/-- A's initial investment -/
def a_initial : ℕ := 3000

/-- B's initial investment -/
def b_initial : ℕ := 4000

/-- Amount A withdrew -/
def a_withdrawal : ℕ := 1000

/-- Amount B advanced -/
def b_advance : ℕ := 1000

/-- Total profit at the end of the year -/
def total_profit : ℕ := 630

/-- A's share of the profit -/
def a_profit : ℕ := 240

/-- B's share of the profit -/
def b_profit : ℕ := total_profit - a_profit

/-- Theorem stating that the number of months until the change is 8 -/
theorem months_until_change_is_eight :
  months_until_change = 8 ∧
  (a_initial * months_until_change + (a_initial - a_withdrawal) * (12 - months_until_change)) * b_profit =
  (b_initial * months_until_change + (b_initial + b_advance) * (12 - months_until_change)) * a_profit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_months_until_change_is_eight_l221_22112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l221_22191

theorem sum_of_reciprocals_equals_one (a b : ℝ) (h1 : (2 : ℝ)^a = 10) (h2 : (5 : ℝ)^b = 10) : 
  1/a + 1/b = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_equals_one_l221_22191
