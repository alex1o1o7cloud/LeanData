import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l65_6540

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def lineSlope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are parallel -/
def parallel (l₁ l₂ : Line) : Prop := lineSlope l₁ = lineSlope l₂

/-- Two lines are coincident -/
def coincident (l₁ l₂ : Line) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

theorem parallel_lines_a_value (a : ℝ) :
  let l₁ : Line := ⟨a, 2, 6⟩
  let l₂ : Line := ⟨1, a - 1, a^2 - 1⟩
  parallel l₁ l₂ ∧ ¬coincident l₁ l₂ → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l65_6540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l65_6518

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - 4*x < 0}

-- Define set N
def N : Set ℝ := {0, 4}

-- Theorem statement
theorem union_of_M_and_N : M ∪ N = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l65_6518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_monotonic_f_l65_6504

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then
    -1/3 * x^3 + (1-a)/2 * x^2 + a*x - 4/3
  else
    (a-1) * Real.log x + 1/2 * x^2 - a*x

theorem a_range_for_monotonic_f (a : ℝ) :
  a > 0 ∧ 
  (∀ x y, -a < x ∧ x < y ∧ y < 2*a → f a x ≤ f a y) →
  0 < a ∧ a ≤ 10/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_monotonic_f_l65_6504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_both_rational_l65_6584

theorem no_x_both_rational : ¬∃ x : ℝ, (∃ p q : ℚ, Real.sin x + Real.sqrt 2 = ↑p ∧ Real.cos x - Real.sqrt 2 = ↑q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_x_both_rational_l65_6584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_hyperbola_intersection_l65_6549

/-- Represents a hyperbola with equation y^2 - x^2/3 = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : a^2 = 3 * b^2 + 3

/-- Represents a line with equation y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- Angle of inclination of a line -/
noncomputable def angle_of_inclination (l : Line) : ℝ := Real.arctan l.k

/-- Predicate to check if a line passes through a focus of the hyperbola -/
def passes_through_focus (_h : Hyperbola) (l : Line) : Prop :=
  l.m = 2

/-- Predicate to check if a line intersects the hyperbola at two points on the same branch -/
def intersects_same_branch (_h : Hyperbola) (l : Line) : Prop :=
  (3 * l.k^2 - 1) * 9 < 0 ∧ 3 * l.k^2 - 1 ≠ 0

/-- The main theorem about the range of possible angles of inclination -/
theorem angle_range_for_hyperbola_intersection (h : Hyperbola) (l : Line) :
  passes_through_focus h l →
  intersects_same_branch h l →
  let α := angle_of_inclination l
  0 < α ∧ α < π/6 ∨ 5*π/6 < α ∧ α < π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_for_hyperbola_intersection_l65_6549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_same_direction_l65_6591

noncomputable def a : ℝ × ℝ := (5, 4)
noncomputable def b : ℝ × ℝ := (3, 2)

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := vector_sum v (scalar_mult (-1) w)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let m := magnitude v
  (v.1 / m, v.2 / m)

theorem unit_vector_same_direction :
  let v := vector_sub (scalar_mult 2 a) (scalar_mult 3 b)
  unit_vector v = (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_same_direction_l65_6591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenters_collinear_l65_6585

/-- Circle with center and three points on it -/
structure Circle where
  center : ℝ × ℝ
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ
  pointC : ℝ × ℝ

/-- Point outside the circle -/
def ExternalPoint := ℝ × ℝ

/-- Intersection point of two lines -/
def IntersectionPoint := ℝ × ℝ

/-- Center of circumcircle of a triangle -/
def CircumcenterPoint := ℝ × ℝ

/-- Auxiliary definitions (not proven) -/

def is_tangent : ExternalPoint → ℝ × ℝ → Circle → Prop := sorry
def intersection_point : Set (ℝ × ℝ) → Set (ℝ × ℝ) → IntersectionPoint := sorry
def line : ℝ × ℝ → ℝ × ℝ → Set (ℝ × ℝ) := sorry
def circumcenter : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → CircumcenterPoint := sorry
def collinear : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop := sorry

/-- Main theorem statement -/
theorem circumcenters_collinear 
  (Ω : Circle) 
  (O : ExternalPoint) 
  (D E : IntersectionPoint) 
  (h1 : is_tangent O Ω.pointA Ω)
  (h2 : is_tangent O Ω.pointB Ω)
  (h3 : D = intersection_point (line Ω.pointA Ω.pointC) (line O Ω.pointB))
  (h4 : E = intersection_point (line Ω.pointB Ω.pointC) (line O Ω.pointA)) :
  let ACE_center : CircumcenterPoint := circumcenter Ω.pointA Ω.pointC E
  let BCD_center : CircumcenterPoint := circumcenter Ω.pointB Ω.pointC D
  let OCI_center : CircumcenterPoint := circumcenter O Ω.pointC Ω.center
  collinear ACE_center BCD_center OCI_center := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenters_collinear_l65_6585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relationship_l65_6596

-- Define the sets M, P, and S
def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

-- Theorem stating the relationship between M, P, and S
theorem set_relationship : (M = P) ∧ (S ⊂ P) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_relationship_l65_6596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_10_50_l65_6528

/-- Represents a time on a clock --/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  h_hours_valid : hours < 12
  h_minutes_valid : minutes < 60

/-- Calculates the angle of the hour hand from 12 o'clock position --/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position --/
def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Calculates the acute angle between hour and minute hands --/
noncomputable def acuteAngleBetweenHands (t : ClockTime) : ℝ :=
  min (abs (hourHandAngle t - minuteHandAngle t)) (360 - abs (hourHandAngle t - minuteHandAngle t))

theorem angle_at_10_50 :
  let t : ClockTime := ⟨10, 50, by norm_num, by norm_num⟩
  acuteAngleBetweenHands t = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_10_50_l65_6528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l65_6551

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x^2 + a*x + 1)

-- State the theorem
theorem function_increasing_condition (a : ℝ) : 
  (∀ x < 3, Monotone (fun x => f a x)) → a > 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_increasing_condition_l65_6551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l65_6553

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | |x| < 2}
def N : Set ℝ := {y : ℝ | ∃ x, y = 2^x - 1}

-- State the theorem
theorem complement_union_theorem :
  (Set.univ \ M) ∪ (Set.univ \ N) = Set.Iic (-1) ∪ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l65_6553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_sequence_l65_6556

def sequence_a : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (sequence_a n)^3 + 1

theorem divisibility_in_sequence (p : ℕ) (h_prime : Nat.Prime p) 
  (h_form : ∃ l : ℕ, p = 3 * l + 2) :
  ∃ n : ℕ, p ∣ sequence_a n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_sequence_l65_6556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_when_divided_by_two_l65_6529

theorem remainder_when_divided_by_two (n : ℕ) 
  (h1 : n % 7 = 5)
  (h2 : ∀ p : ℕ, p > 0 → (n + p) % 10 = 0 → p ≥ 5)
  (h3 : (n + 5) % 10 = 0) : 
  n % 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_when_divided_by_two_l65_6529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_percentage_in_water_l65_6502

/-- The atomic mass of hydrogen in g/mol -/
noncomputable def hydrogen_mass : ℝ := 1.008

/-- The atomic mass of oxygen in g/mol -/
noncomputable def oxygen_mass : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def hydrogen_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def oxygen_count : ℕ := 1

/-- The mass percentage of oxygen in water -/
noncomputable def oxygen_percentage : ℝ :=
  (oxygen_count * oxygen_mass) / ((hydrogen_count * hydrogen_mass) + (oxygen_count * oxygen_mass)) * 100

theorem oxygen_percentage_in_water :
  abs (oxygen_percentage - 88.81) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oxygen_percentage_in_water_l65_6502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l65_6571

/-- A set of points in the plane where each point is the midpoint of two other points in the set -/
structure MidpointSet where
  S : Set (ℝ × ℝ)
  midpoint_property : ∀ p, p ∈ S → ∃ a b, a ∈ S ∧ b ∈ S ∧ p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- Theorem stating that a MidpointSet contains infinitely many points -/
theorem midpoint_set_infinite (M : MidpointSet) : Set.Infinite M.S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l65_6571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l65_6508

noncomputable def f (A ω φ x : Real) : Real := A * Real.sin (ω * x + φ)

theorem function_properties 
  (A ω φ : Real) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : abs φ < Real.pi / 2) 
  (h4 : f A ω φ (3 * Real.pi / 8) = 0) 
  (h5 : f A ω φ (Real.pi / 8) = 2) 
  (h6 : ∀ x, x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 8) → f A ω φ x ≤ 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f A ω φ x ≥ -Real.sqrt 2) ∧
  (f A ω φ (-Real.pi / 4) = -Real.sqrt 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), f A ω φ x ≤ 2) ∧
  (f A ω φ (Real.pi / 8) = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l65_6508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF₁F₂_l65_6530

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-5/2, 0)
noncomputable def F₂ : ℝ × ℝ := (5/2, 0)

-- Define the equations of the hyperbola and ellipse
def is_on_hyperbola (x y m : ℝ) : Prop := x^2/4 - y^2/m = 1
def is_on_ellipse (x y n : ℝ) : Prop := x^2/9 + y^2/n = 1

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := sorry

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_PF₁F₂ (m n : ℝ) 
  (h1 : is_on_hyperbola P.1 P.2 m)
  (h2 : is_on_ellipse P.1 P.2 n) :
  area_triangle P F₁ F₂ = 3 * Real.sqrt 11 / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PF₁F₂_l65_6530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l65_6595

theorem cos_2beta_value (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos (α - Real.cos (α - β) * Real.sin α) = 3/5) : 
  Real.cos (2 * β) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_value_l65_6595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_set_size_l65_6520

def is_valid_set (M : Finset ℕ) : Prop :=
  1 ∈ M ∧ 
  100 ∈ M ∧
  ∀ x, x ∈ M → 1 ≤ x ∧ x ≤ 100 ∧
  (x ≠ 1 → ∃ y z, y ∈ M ∧ z ∈ M ∧ x = y + z)

theorem smallest_valid_set_size :
  ∃ M : Finset ℕ, is_valid_set M ∧ M.card = 9 ∧
  ∀ N : Finset ℕ, is_valid_set N → M.card ≤ N.card :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_set_size_l65_6520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_constant_sum_l65_6594

/-- The trajectory of a point S in the Cartesian plane -/
def Trajectory (S : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, S (x, y) ↔ x^2 / 2 + y^2 = 1

/-- The distance ratio condition for point S -/
def DistanceRatio (S : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, S (x, y) → 
    Real.sqrt ((x - 1)^2 + y^2) / |x - 2| = Real.sqrt 2 / 2

/-- The constant sum of squared distances property -/
def ConstantSumProperty (S : ℝ × ℝ → Prop) : Prop :=
  ∀ x₀ : ℝ, 
    let l := λ t ↦ (x₀ + Real.sqrt 6 * t / 3, Real.sqrt 3 * t / 3)
    ∃ t₁ t₂, 
      S (l t₁) ∧ S (l t₂) ∧
      (l t₁).1^2 + (l t₁).2^2 + (l t₂).1^2 + (l t₂).2^2 - 2 * x₀^2 = 3

theorem trajectory_and_constant_sum 
  (S : ℝ × ℝ → Prop) 
  (h : DistanceRatio S) :
  Trajectory S ∧ ConstantSumProperty S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_constant_sum_l65_6594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_value_l65_6509

/-- The speed of person A when two people walk towards each other --/
noncomputable def speed_of_A (initial_distance : ℝ) (time : ℝ) (speed_B : ℝ) : ℝ :=
  initial_distance / time - speed_B

theorem speed_of_A_value : speed_of_A 25 1 13 = 12 := by
  -- Unfold the definition of speed_of_A
  unfold speed_of_A
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_A_value_l65_6509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_value_l65_6589

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 1

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f 1 2 = Real.log 2 - 1/2 →
  (deriv (f 1)) 2 = 1/4 →
  x - 4*y + 4*Real.log 2 - 4 = 0 ↔ 
  y - (Real.log 2 - 1/2) = 1/4 * (x - 2) :=
sorry

-- Theorem for the minimum value of f
theorem minimum_value (a : ℝ) :
  (0 < a → a < Real.exp 1 → ∃ m, m = Real.log a ∧ ∀ x ∈ Set.Ioo 0 (Real.exp 1), m ≤ f a x) ∧
  (a ≥ Real.exp 1 → ∃ m, m = a / Real.exp 1 ∧ ∀ x ∈ Set.Ioo 0 (Real.exp 1), m ≤ f a x) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_minimum_value_l65_6589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l65_6563

/-- Given a cube with the sum of edge lengths equal to 96 cm, its surface area is 384 cm^2. -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 96) : 
  6 * (edge_sum / 12)^2 = 384 := by
  sorry

-- Remove the #eval line as it's causing the universe level error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l65_6563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_e_l65_6535

-- Define e as Euler's number
noncomputable def e : ℝ := Real.exp 1

-- Define the properties of e
axiom e_bounds : 2 < e ∧ e < 3

-- Theorem to prove
theorem floor_of_e : ⌊e⌋ = 2 := by
  -- We'll use the property of floor function and the bounds of e
  have h1 : 2 < e := e_bounds.left
  have h2 : e < 3 := e_bounds.right
  
  -- For any real x, if n ≤ x < n+1, then ⌊x⌋ = n
  have floor_prop : ∀ (x : ℝ) (n : ℤ), ↑n ≤ x ∧ x < ↑(n + 1) → ⌊x⌋ = n := by
    sorry  -- We'll leave the proof of this property as an exercise
  
  -- Apply the floor property to e
  apply floor_prop e 2
  constructor
  · exact le_of_lt h1
  · exact h2
  
  -- The proof is complete


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_e_l65_6535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_monomial_properties_l65_6557

/-- Represents a monomial term in a polynomial -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  powers : List ℕ

/-- Calculates the degree of a monomial -/
def degree {α : Type*} [Ring α] (m : Monomial α) : ℕ := m.powers.sum

/-- The monomial -1/3xy² -/
def our_monomial : Monomial ℚ := 
  { coeff := -1/3,
    powers := [1, 2] }

theorem our_monomial_properties : 
  our_monomial.coeff = -1/3 ∧ degree our_monomial = 3 := by
  constructor
  · rfl
  · rfl

#eval our_monomial.coeff
#eval degree our_monomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_monomial_properties_l65_6557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_everlee_family_children_l65_6512

/-- Proves the number of children in Everlee's family --/
theorem everlee_family_children : 
  ∀ (total_cookies : ℕ) (adult_fraction : ℚ) (cookies_per_child : ℕ),
    total_cookies = 120 →
    adult_fraction = 1 / 3 →
    cookies_per_child = 20 →
    ∃ (num_children : ℕ),
      num_children = (total_cookies - (adult_fraction * ↑total_cookies).floor) / cookies_per_child ∧
      num_children = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_everlee_family_children_l65_6512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_percentage_approx_l65_6534

/-- Represents the amount of food a single dog receives -/
noncomputable def dog_food : ℝ := 1

/-- The number of dogs -/
def num_dogs : ℕ := 9

/-- The number of cats -/
def num_cats : ℕ := 6

/-- The number of hamsters -/
def num_hamsters : ℕ := 10

/-- The total amount of food for all cats relative to a single dog's food -/
noncomputable def total_cat_food : ℝ := 1.5 * dog_food

/-- The total amount of food for all hamsters relative to a single dog's food -/
noncomputable def total_hamster_food : ℝ := 0.25 * dog_food

/-- The amount of food a single cat receives -/
noncomputable def cat_food : ℝ := total_cat_food / num_cats

/-- The total amount of food for all pets -/
noncomputable def total_food : ℝ := num_dogs * dog_food + total_cat_food + total_hamster_food

/-- The percentage of food a single cat receives -/
noncomputable def cat_food_percentage : ℝ := (cat_food / total_food) * 100

theorem cat_food_percentage_approx :
  abs (cat_food_percentage - 2.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_food_percentage_approx_l65_6534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_over_q_at_0_l65_6564

-- Define the polynomials p and q
variable (p q : ℝ → ℝ)

-- Define the conditions
axiom quadratic : ∃ a b c d e f : ℝ, ∀ x, p x = a * x^2 + b * x + c ∧ q x = d * x^2 + e * x + f

axiom horizontal_asymptote : ∀ ε > 0, ∃ M : ℝ, ∀ x, abs x > M → abs (p x / q x - 1) < ε

axiom vertical_asymptote_neg3 : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs (x + 3) ∧ abs (x + 3) < δ → abs (p x / q x) > 1 / ε

axiom vertical_asymptote_1 : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs (x - 1) ∧ abs (x - 1) < δ → abs (p x / q x) > 1 / ε

axiom hole_at_4 : p 4 = 0 ∧ q 4 = 0 ∧ (∃ L : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs (x - 4) ∧ abs (x - 4) < δ → abs (p x / q x - L) < ε)

-- The theorem to prove
theorem p_over_q_at_0 : p 0 / q 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_over_q_at_0_l65_6564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_approx_l65_6573

/-- Calculates the percentage reduction in oil price given the conditions -/
noncomputable def oil_price_reduction (additional_kg : ℝ) (total_cost : ℝ) (reduced_price : ℝ) : ℝ := 
  let original_kg : ℝ := total_cost / reduced_price - additional_kg
  let original_price : ℝ := total_cost / original_kg
  (original_price - reduced_price) / original_price * 100

/-- Theorem stating the percentage reduction in oil price -/
theorem oil_price_reduction_approx :
  let additional_kg : ℝ := 9
  let total_cost : ℝ := 900
  let reduced_price : ℝ := 30
  ∃ (x : ℝ), abs (x - oil_price_reduction additional_kg total_cost reduced_price) < 0.005 ∧ x = 30.23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_approx_l65_6573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l65_6590

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x / Real.sqrt (x - 1)

-- State the theorem
theorem domain_of_g :
  (∀ x, f (x + 2) ≠ 0 ↔ -3 < x ∧ x < 4) →
  (∀ x, g x ≠ 0 ↔ 1 < x ∧ x < 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l65_6590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_floor_l65_6516

/-- Calculates the number of tiles needed to cover a floor -/
def tiles_needed (floor_length floor_width tile_length tile_width : ℚ) : ℕ :=
  (floor_length * floor_width / (tile_length * tile_width)).ceil.toNat

/-- Proves that 800 tiles of size 3 inches by 9 inches are needed to cover a floor of size 10 feet by 15 feet -/
theorem tiles_for_floor : tiles_needed 10 15 (1/4) (3/4) = 800 := by
  sorry

#eval tiles_needed 10 15 (1/4) (3/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_floor_l65_6516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l65_6598

/-- A function that returns the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + numDigits (n / 10)

/-- A function that returns the rightmost four digits of a number -/
def rightmostFourDigits (n : ℕ) : ℕ :=
  n % 10000

/-- The main theorem statement -/
theorem count_special_numbers : 
  let validNumbers := {M : ℕ | 
    numDigits M = 5 ∧ 
    rightmostFourDigits M = M / 8 ∧
    M ≥ 10000 ∧ M < 100000
  }
  (Finset.filter (fun M => 
    numDigits M = 5 ∧ 
    rightmostFourDigits M = M / 8 ∧
    M ≥ 10000 ∧ M < 100000) (Finset.range 100000)).card = 1224 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_l65_6598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_one_l65_6513

-- Define the expression
noncomputable def complex_expression (t : ℝ) : ℝ :=
  ((1 - Real.sqrt (2 * t)) / ((1 - (8 * t^3)^(1/4)) / (1 - (2 * t)^(1/4)) - Real.sqrt (2 * t))) *
  (((((1 / (1/2)) + (4 * t^2)^(1/4))^(1/4)) / (1 + (1 / (2 * t))^(1/4)) - Real.sqrt (2 * t))⁻¹)

-- State the theorem
theorem expression_simplifies_to_one (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1/2) :
  complex_expression t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplifies_to_one_l65_6513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_consumption_l65_6539

/-- Represents the rates at which the fish and bird eat the apple -/
structure EatingRates where
  fish : ℝ
  bird : ℝ

/-- Theorem stating the portions of the apple consumed by the fish and bird -/
theorem apple_consumption (rates : EatingRates) 
  (h1 : rates.fish = 120)
  (h2 : rates.bird = 60) : 
  (rates.fish / (rates.fish + rates.bird) = 2/3) ∧ 
  (rates.bird / (rates.fish + rates.bird) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_consumption_l65_6539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_iff_two_two_l65_6579

/-- A sequence of positive integers -/
def RecursiveSequence (a₁ a₂ : ℕ+) : ℕ → ℕ+
  | 0 => a₁
  | 1 => a₂
  | (n + 2) => 
    let prev := RecursiveSequence a₁ a₂ n
    let curr := RecursiveSequence a₁ a₂ (n + 1)
    ⟨(prev + curr) / Nat.gcd prev curr, by 
      sorry -- Proof that the result is positive
    ⟩

/-- A sequence is periodic if there exists a positive integer p such that
    a_{n+p} = a_n for all n ≥ 0 -/
def IsPeriodic (a₁ a₂ : ℕ+) : Prop :=
  ∃ p : ℕ+, ∀ n : ℕ, RecursiveSequence a₁ a₂ (n + p) = RecursiveSequence a₁ a₂ n

/-- The main theorem: The sequence is periodic if and only if (a₁, a₂) = (2, 2) -/
theorem periodic_iff_two_two (a₁ a₂ : ℕ+) :
  IsPeriodic a₁ a₂ ↔ a₁ = 2 ∧ a₂ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_iff_two_two_l65_6579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_painting_l65_6531

-- Define the students and activities
inductive Student : Type
  | A | B | C | D

inductive Activity : Type
  | Basketball | Painting | Dancing | Running

-- Define the function that assigns an activity to each student
variable (activity : Student → Activity)

-- Define the conditions
axiom condition1 : activity Student.A ≠ Activity.Running ∧ activity Student.A ≠ Activity.Basketball
axiom condition2 : activity Student.B ≠ Activity.Dancing ∧ activity Student.B ≠ Activity.Running
axiom condition3 : activity Student.C = Activity.Running → activity Student.A = Activity.Dancing
axiom condition4 : activity Student.D ≠ Activity.Basketball ∧ activity Student.D ≠ Activity.Running
axiom condition5 : activity Student.C ≠ Activity.Dancing ∧ activity Student.C ≠ Activity.Basketball

-- Define the theorem to be proved
theorem d_is_painting : activity Student.D = Activity.Painting :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_painting_l65_6531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l65_6526

-- Define the given conditions
def conditions (a b : ℝ) : Prop := (2 : ℝ)^a = 3 ∧ (3 : ℝ)^b = 2

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

-- Theorem statement
theorem zero_in_interval (a b : ℝ) (h : conditions a b) :
  ∃! x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f a b x = 0 := by
  sorry

#check zero_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l65_6526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_circles_l65_6565

/-- Two circles intersect at points (1, 3) and (m, 1), and their centers lie on the line x - y + c/2 = 0 -/
theorem intersection_of_circles (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)) (center1 center2 : ℝ × ℝ),
    -- The circles intersect at (1, 3) and (m, 1)
    (1, 3) ∈ circle1 ∩ circle2 ∧
    (m, 1) ∈ circle1 ∩ circle2 ∧
    -- The centers lie on the line x - y + c/2 = 0
    center1.1 - center1.2 + c/2 = 0 ∧
    center2.1 - center2.2 + c/2 = 0 ∧
    -- circle1 and circle2 are circles with centers center1 and center2 respectively
    ∃ (r1 r2 : ℝ), circle1 = {p : ℝ × ℝ | (p.1 - center1.1)^2 + (p.2 - center1.2)^2 = r1^2} ∧
                   circle2 = {p : ℝ × ℝ | (p.1 - center2.1)^2 + (p.2 - center2.2)^2 = r2^2}) →
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_circles_l65_6565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l65_6574

/-- The angle θ when a line is tangent to a circle --/
theorem line_tangent_to_circle (θ : Real) (h1 : 0 ≤ θ) (h2 : θ < π) :
  (∃ t α : Real, 
    (∀ x y : Real, x = t * Real.cos θ ∧ y = t * Real.sin θ) ∧
    (∀ x y : Real, x = 4 + 2 * Real.cos α ∧ y = 2 * Real.sin α) ∧
    (∀ x y : Real, (x - 4)^2 + y^2 = 4) ∧
    (∀ x y : Real, y = x * Real.tan θ) ∧
    (|4 * Real.tan θ| / Real.sqrt (Real.tan θ^2 + 1) = 2)) →
  θ = π/6 ∨ θ = 5*π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l65_6574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_999_value_l65_6558

def a : ℕ → ℚ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | (n + 1) => a n + (2 * a n) / n

theorem a_999_value : a 999 = 499500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_999_value_l65_6558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l65_6587

theorem expansion_theorem (m n : ℕ) : 
  (∀ x : ℚ, (x + m) * (x^2 - 3*x + n) = x^3 + (-1)*x^2 + 0*x + m*n) → 
  n^m = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_theorem_l65_6587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l65_6578

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 3 * Real.sqrt 3 = 0

/-- The circle C in the Cartesian plane -/
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The minimum distance from a point on circle C to line l -/
noncomputable def min_distance : ℝ := 2 * Real.sqrt 3 - 1

/-- Theorem stating the minimum distance from circle C to line l -/
theorem min_distance_circle_to_line :
  ∀ (x y : ℝ), circle_C x y →
  ∃ (x' y' : ℝ), line_l x' y' ∧
  ∀ (p q : ℝ), line_l p q →
  Real.sqrt ((x - p)^2 + (y - q)^2) ≥ min_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l65_6578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_line_arrangements_l65_6546

def number_of_animals : ℕ := 6
def fixed_positions : ℕ := 2

theorem animal_line_arrangements :
  Nat.factorial (number_of_animals - fixed_positions) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_animal_line_arrangements_l65_6546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abd_l65_6537

/-- Definition of An as a function of n and a -/
def An (n : ℕ) (a : ℕ) : ℕ := a * (10^n - 1) / 9

/-- Definition of Bn as a function of n and b -/
def Bn (n : ℕ) (b : ℕ) : ℕ := b * (10^n - 1) / 9

/-- Definition of Cn as a function of n and d -/
def Cn (n : ℕ) (d : ℕ) : ℕ := d * (10^(2*n) - 1) / 9

/-- Theorem stating the maximum value of a + b + d -/
theorem max_sum_abd (a b d : ℕ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) (hd : 0 < d ∧ d < 10) :
  (∃ n1 n2 : ℕ, n1 ≠ n2 ∧ n1 > 0 ∧ n2 > 0 ∧
    Cn n1 d - Bn n1 b = 2 * (An n1 a)^2 ∧ 
    Cn n2 d - Bn n2 b = 2 * (An n2 a)^2) →
  a + b + d ≤ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abd_l65_6537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l65_6527

/-- A parabola with equation y = ax² and directrix y = 1 has a = -1/4 --/
theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x, k) ∈ Set.range (λ x ↦ (x, 1))) →  -- Directrix equation
  a = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l65_6527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_set_probability_l65_6538

noncomputable def digit_probability (d : ℕ) : ℝ := Real.log (d + 1) - Real.log d

theorem correct_set_probability :
  let p4 := digit_probability 4
  let p5 := digit_probability 5
  let p6 := digit_probability 6
  let p7 := digit_probability 7
  3 * p4 = p5 + p6 + p7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_set_probability_l65_6538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_ratio_l65_6586

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfTime : ℝ
  secondHalfTime : ℝ

/-- Calculates the ratio of speeds given a RunnerJourney --/
noncomputable def speedRatio (journey : RunnerJourney) : ℝ :=
  (journey.firstHalfTime * journey.secondHalfTime) / 
  ((journey.totalDistance / 2) * (journey.secondHalfTime - journey.firstHalfTime))

/-- Theorem statement for the runner's speed ratio --/
theorem runner_speed_ratio :
  ∀ (journey : RunnerJourney),
    journey.totalDistance = 40 ∧ 
    journey.secondHalfTime = 16 ∧
    journey.secondHalfTime = journey.firstHalfTime + 8 →
    speedRatio journey = 1 / 2 := by
  sorry

#eval "Runner speed ratio theorem defined."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_ratio_l65_6586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l65_6511

/-- Non-isosceles triangle with sides a, b, c and opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  non_isosceles : a ≠ b ∧ b ≠ c ∧ a ≠ c
  angle_sum : A + B + C = Real.pi
  sine_law : Real.sin A / a = Real.sin B / b
  arithmetic_sequence : B - A = C - B

theorem triangle_equality (t : Triangle) : 
  1 / (t.a - t.b) + 1 / (t.c - t.b) = 3 / (t.a - t.b + t.c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equality_l65_6511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l65_6523

noncomputable def data : List ℝ := [1, 3, 3, 6, 7]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem data_variance :
  variance data = 24 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_variance_l65_6523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l65_6559

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_acute (t : Triangle) : Prop := sorry

def angle_A_is_pi_div_3 (t : Triangle) : Prop := sorry

def AM_is_angle_bisector (t : Triangle) (M : ℝ × ℝ) : Prop := sorry

def AM_intersects_BC (t : Triangle) (M : ℝ × ℝ) : Prop := sorry

def AM_length_is_2 (t : Triangle) (M : ℝ × ℝ) : Prop := sorry

def O_is_circumcenter (t : Triangle) (O : ℝ × ℝ) : Prop := sorry

def circumradius_is_1 (t : Triangle) (O : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_properties (t : Triangle) (M O : ℝ × ℝ) :
  is_acute t →
  angle_A_is_pi_div_3 t →
  AM_is_angle_bisector t M →
  AM_intersects_BC t M →
  AM_length_is_2 t M →
  O_is_circumcenter t O →
  circumradius_is_1 t O →
  (∃ (min_value : ℝ), min_value = (8 * Real.sqrt 3) / 3 ∧
    ∀ (x : ℝ), x ≥ min_value → x ≥ (t.B.1 - t.A.1) + 3 * (t.C.1 - t.A.1)) ∧
  (∀ (dot_product : ℝ),
    dot_product = (O.1 - t.A.1) * ((t.B.1 - t.A.1) + (t.C.1 - t.A.1)) +
                  (O.2 - t.A.2) * ((t.B.2 - t.A.2) + (t.C.2 - t.A.2)) →
    -3 ≤ dot_product ∧ dot_product < -5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l65_6559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_three_l65_6533

-- Define g as a function from real numbers to real numbers
def g : ℝ → ℝ := λ x => 5

-- Theorem to prove
theorem g_shift_three (x : ℝ) : g (x - 3) = 5 := by
  -- Unfold the definition of g
  unfold g
  -- Simplify
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_shift_three_l65_6533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_cost_is_4_25_l65_6501

/-- The cost of the compound per pound -/
noncomputable def compound_cost_per_pound (limestone_price shale_price : ℝ) 
  (total_weight limestone_weight : ℝ) : ℝ :=
  let shale_weight := total_weight - limestone_weight
  let total_cost := limestone_price * limestone_weight + shale_price * shale_weight
  total_cost / total_weight

/-- Theorem stating that the cost of the compound per pound is $4.25 -/
theorem compound_cost_is_4_25 :
  compound_cost_per_pound 3 5 100 37.5 = 4.25 := by
  -- Unfold the definition of compound_cost_per_pound
  unfold compound_cost_per_pound
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_cost_is_4_25_l65_6501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piglet_straws_calculation_l65_6575

/-- Given a total number of straws, a fraction fed to adult pigs, and a number of piglets,
    calculate the number of straws each piglet receives. -/
def straws_per_piglet (total_straws : ℕ) (adult_pig_fraction : ℚ) (num_piglets : ℕ) : ℚ :=
  let adult_pig_straws := (adult_pig_fraction * total_straws).floor
  let piglet_straws := total_straws - adult_pig_straws
  (piglet_straws : ℚ) / num_piglets

/-- Theorem stating that given 300 straws, with 3/5 fed to adult pigs and the remainder
    equally distributed among 20 piglets, each piglet receives 6 straws. -/
theorem piglet_straws_calculation :
  straws_per_piglet 300 (3/5) 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piglet_straws_calculation_l65_6575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_l65_6597

/-- The number of mixed groups in a chess club session --/
theorem chess_club_mixed_groups 
  (total_children : ℕ) 
  (num_groups : ℕ) 
  (children_per_group : ℕ) 
  (boy_games : ℕ) 
  (girl_games : ℕ) 
  (num_mixed_groups : ℕ)
  (h1 : total_children = 90)
  (h2 : num_groups = 30)
  (h3 : children_per_group = 3)
  (h4 : boy_games = 30)
  (h5 : girl_games = 14)
  (h6 : num_groups * Nat.choose children_per_group 2 = boy_games + girl_games + num_mixed_groups * 2) :
  num_mixed_groups = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_club_mixed_groups_l65_6597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_value_l65_6519

/-- The nth term of the exponent series -/
def a (n : ℕ) : ℚ := (2^n * 2^n) / (3^n)

/-- The sum of the exponent series -/
noncomputable def S : ℝ := ∑' n, (a n : ℝ)

/-- The infinite product in the original problem -/
noncomputable def infiniteProduct : ℝ := 3^(S : ℝ)

theorem infinite_product_value : infiniteProduct = 27 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_value_l65_6519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l65_6560

noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x - 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l65_6560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_product_polynomials_l65_6568

theorem sum_of_roots_product_polynomials :
  let p₁ := fun x : ℝ => 3 * x^3 - 9 * x^2 + 12 * x - 4
  let p₂ := fun x : ℝ => 4 * x^3 + 2 * x^2 - 3 * x + 1
  let roots := {x : ℝ | p₁ x = 0 ∨ p₂ x = 0}
  (⨆ x ∈ roots, x) = (5 : ℝ) / 2 := by
  sorry

#check sum_of_roots_product_polynomials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_product_polynomials_l65_6568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_5x_div_9_l65_6547

theorem remainder_5x_div_9 (x : ℕ) (h : x % 9 = 5) : (5 * x) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_5x_div_9_l65_6547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_of_differences_l65_6525

/-- A geometric sequence with a_3 = 2 and a_4 * a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) ∧
  a 3 = 2 ∧
  a 4 * a 6 = 16

/-- The ratio of differences in the geometric sequence -/
noncomputable def ratio_of_differences (a : ℕ → ℝ) : ℝ :=
  (a 10 - a 12) / (a 6 - a 8)

theorem geometric_sequence_ratio_of_differences (a : ℕ → ℝ) :
  geometric_sequence a → ratio_of_differences a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_of_differences_l65_6525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l65_6569

/-- A regular tetrahedron with the following properties:
    - M and N are midpoints of opposite edges
    - Its orthogonal projection onto a plane parallel to line MN is a quadrilateral
    - The quadrilateral has area S
    - One of the quadrilateral's angles is 60° -/
structure RegularTetrahedron where
  S : ℝ  -- Area of the projected quadrilateral
  angle_60 : ℝ  -- One of the angles of the projected quadrilateral

/-- The surface area of the tetrahedron -/
noncomputable def surface_area (t : RegularTetrahedron) : ℝ := 3 * t.S * Real.sqrt 2

/-- Theorem stating that the surface area of the tetrahedron is 3S√2 -/
theorem tetrahedron_surface_area (t : RegularTetrahedron) :
  surface_area t = 3 * t.S * Real.sqrt 2 := by
  -- Proof goes here
  sorry

#check tetrahedron_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_area_l65_6569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_equipment_cost_l65_6545

/-- Calculate the total cost of equipment for a football team --/
theorem football_team_equipment_cost :
  let num_players : ℕ := 25
  let jersey_cost : ℚ := 25
  let shorts_cost : ℚ := 15.20
  let socks_cost : ℚ := 6.80
  let cleats_cost : ℚ := 40
  let water_bottle_cost : ℚ := 12
  let discount_rate : ℚ := 0.10
  let discount_threshold : ℚ := 500
  let tax_rate : ℚ := 0.07

  let equipment_cost_per_player : ℚ := jersey_cost + shorts_cost + socks_cost + cleats_cost + water_bottle_cost
  let total_cost : ℚ := equipment_cost_per_player * num_players
  let discount : ℚ := if total_cost > discount_threshold then total_cost * discount_rate else 0
  let discounted_total : ℚ := total_cost - discount
  let tax : ℚ := discounted_total * tax_rate
  let final_cost : ℚ := discounted_total + tax

  (Rat.floor (final_cost * 100 + 1/2) : ℚ) / 100 = 2383.43 := by
  sorry

#eval (let x : ℚ := 2383.425; (Rat.floor (x * 100 + 1/2) : ℚ) / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_team_equipment_cost_l65_6545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_90_percent_l65_6570

/-- Blood oxygen saturation model -/
noncomputable def S (S₀ K t : ℝ) : ℝ := S₀ * Real.exp (K * t)

/-- Theorem: Additional time needed to reach 90% saturation -/
theorem additional_time_to_90_percent (S₀ K : ℝ) 
  (h₁ : S₀ = 0.6)
  (h₂ : S S₀ K 1 = 0.8)
  (h₃ : ∃ t, S S₀ K t = 0.9) :
  ∃ t, t > 1 ∧ S S₀ K t = 0.9 ∧ t - 1 = 0.5 := by
  sorry

#check additional_time_to_90_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_90_percent_l65_6570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillating_bounded_examples_oscillating_unbounded_examples_l65_6510

-- Part (a)
def oscillating_bounded (f : ℝ → ℝ) (p a b : ℝ) : Prop :=
  a < b ∧
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - p| ∧ |x - p| < δ → 
    a ≤ f x ∧ f x ≤ b ∧
    (∃ x₁ x₂ : ℝ, |x₁ - p| < δ ∧ |x₂ - p| < δ ∧ |f x₁ - f x₂| > (b - a) - ε)

theorem oscillating_bounded_examples (p a b : ℝ) (h : a < b) :
  oscillating_bounded (fun x ↦ Real.sin (1 / (x - p))) p (-1) 1 ∧
  oscillating_bounded (fun x ↦ a + (b - a) * (Real.sin (1 / (x - p)))^2) p a b :=
sorry

-- Part (b)
def oscillating_unbounded (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ ε > 0, ∃ x₁ x₂ : ℝ,
    |x₁ - p| < δ ∧ |x₂ - p| < δ ∧ |f x₁ - f x₂| > M

theorem oscillating_unbounded_examples (p : ℝ) :
  oscillating_unbounded (fun x ↦ 1 / (x - p)) p ∧
  oscillating_unbounded (fun x ↦ Real.sin (1 / (x - p))) p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oscillating_bounded_examples_oscillating_unbounded_examples_l65_6510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l65_6503

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the line m
def line_m (x : ℝ) : Prop := x = 4/3

-- Define the circle
def problem_circle (x y : ℝ) : Prop := x^2 + (y - 1/3)^2 = 16/9

-- Define the intersection points A and B
noncomputable def point_A : ℝ × ℝ := (0, -1)
noncomputable def point_B : ℝ × ℝ := (4/3, 1/3)

-- Theorem statement
theorem circle_equation :
  ∀ (x y : ℝ),
  (ellipse x y ∧ line_l x y) →
  ((x = point_A.1 ∧ y = point_A.2) ∨ (x = point_B.1 ∧ y = point_B.2)) →
  problem_circle x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l65_6503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a10_is_248_l65_6592

/-- Represents the sum of all elements in a set of natural numbers -/
def sigma (S : Finset Nat) : Nat := S.sum id

/-- The set A of 11 positive integers -/
def A : Finset Nat := Finset.range 11

/-- The elements of A in ascending order -/
def a : Fin 11 → Nat := sorry

/-- The elements of A are in strictly ascending order -/
axiom a_ascending : ∀ i j, i < j → a i < a j

/-- For every positive integer n ≤ 1500, there exists a subset S of A such that σ(S) = n -/
axiom subset_sum_property : ∀ n : Nat, n ≤ 1500 → ∃ S : Finset Nat, S ⊆ A ∧ sigma S = n

/-- The minimum value of a₁₀ is 248 -/
theorem min_a10_is_248 : a 9 = 248 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a10_is_248_l65_6592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l65_6541

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem f_composition_value : f (f 2) = 394 := by
  -- Calculate f(2)
  have h1 : f 2 = 10 := by
    calc f 2 = 4 * 2^2 - 6 := rfl
         _ = 4 * 4 - 6 := by ring
         _ = 16 - 6 := by ring
         _ = 10 := by ring

  -- Calculate f(f(2))
  calc f (f 2) = f 10 := by rw [h1]
       _ = 4 * 10^2 - 6 := rfl
       _ = 4 * 100 - 6 := by ring
       _ = 400 - 6 := by ring
       _ = 394 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l65_6541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_at_six_oclock_l65_6554

-- Define the clock and disk
noncomputable def clock_radius : ℝ := 30
noncomputable def disk_radius : ℝ := 15

-- Define the starting position
noncomputable def start_position : ℝ := 0 -- 0 radians represents 12 o'clock

-- Define the function to calculate the position of the disk
noncomputable def disk_position (angle : ℝ) : ℝ :=
  (clock_radius + disk_radius) * angle / disk_radius

-- Theorem statement
theorem disk_at_six_oclock :
  disk_position (2 * Real.pi) = Real.pi := by
  -- Proof steps would go here
  sorry

#check disk_at_six_oclock

end NUMINAMATH_CALUDE_ERRORFEEDBACK_disk_at_six_oclock_l65_6554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l65_6543

theorem sin_cos_sum (x : ℝ) 
  (h1 : Real.sin x * Real.cos x = -1/4)
  (h2 : 3*Real.pi/4 < x ∧ x < Real.pi) : 
  Real.sin x + Real.cos x = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l65_6543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_property_l65_6505

noncomputable def quadratic_equation_roots (z₁ z₂ m : ℂ) : ℂ × ℂ := sorry

theorem quadratic_equation_roots_property (z₁ z₂ m : ℂ) 
  (h₁ : z₁^2 - 4*z₂ = 16 + 20*Complex.I) 
  (h₂ : Complex.abs ((quadratic_equation_roots z₁ z₂ m).1 - (quadratic_equation_roots z₁ z₂ m).2) = 2 * Real.sqrt 7) :
  (Set.Icc (7 - Real.sqrt 41 : ℝ) (Real.sqrt 41 + 7)) = { x : ℝ | ∃ (m : ℂ), Complex.abs m = x ∧ 
    quadratic_equation_roots z₁ z₂ m = quadratic_equation_roots z₁ z₂ m } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_property_l65_6505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_is_8_l65_6599

def A : Finset ℕ := Finset.range 17

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ x ∈ A, f x ∈ A

def f_power (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | 1, x => f x
  | k + 1, x => f (f_power f k x)

def condition_1 (f : ℕ → ℕ) (M : ℕ) : Prop :=
  ∀ m < M, ∀ i ∈ A.filter (λ x => x ≤ 16),
    (f_power f m (i + 1) - f_power f m i) % 17 ≠ 1 ∧
    (f_power f m (i + 1) - f_power f m i) % 17 ≠ 16 ∧
    (f_power f m 1 - f_power f m 17) % 17 ≠ 1 ∧
    (f_power f m 1 - f_power f m 17) % 17 ≠ 16

def condition_2 (f : ℕ → ℕ) (M : ℕ) : Prop :=
  ∀ i ∈ A.filter (λ x => x ≤ 16),
    ((f_power f M (i + 1) - f_power f M i) % 17 = 1 ∨
     (f_power f M (i + 1) - f_power f M i) % 17 = 16) ∧
    ((f_power f M 1 - f_power f M 17) % 17 = 1 ∨
     (f_power f M 1 - f_power f M 17) % 17 = 16)

theorem max_M_is_8 :
  ∃ (f : ℕ → ℕ), is_valid_f f ∧ condition_1 f 8 ∧ condition_2 f 8 ∧
  ∀ M > 8, ¬(condition_1 f M ∧ condition_2 f M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_M_is_8_l65_6599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l65_6524

-- Define the triangle and points
variable (A B C E F P : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
axiom triangle_exists : A ≠ B ∧ B ≠ C ∧ C ≠ A

axiom E_on_AC : ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ E = (1 - t) • A + t • C
axiom E_ratio : E = (3/4) • A + (1/4) • C

axiom F_on_AB : ∃ s : ℝ, s ∈ Set.Ioo 0 1 ∧ F = (1 - s) • A + s • B
axiom F_ratio : F = (1/5) • A + (4/5) • B

axiom P_intersection : ∃ u v : ℝ, 
  P = (1 - u) • B + u • E ∧
  P = (1 - v) • C + v • F

-- The theorem to prove
theorem intersection_point_coordinates :
  P = (12/19) • A + (3/19) • B + (4/19) • C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_coordinates_l65_6524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_wins_four_consecutive_prob_fifth_game_prob_C_wins_badminton_game_probabilities_l65_6555

/-- Represents the outcome of a single game -/
inductive GameOutcome
| Win
| Lose

/-- Represents a player in the game -/
inductive Player
| A
| B
| C

/-- The probability of winning a single game -/
def winProbability : ℚ := 1 / 2

/-- The game ends when a player loses two games in a row -/
def isEliminated (games : List GameOutcome) : Prop :=
  games.take 2 = [GameOutcome.Lose, GameOutcome.Lose]

/-- Theorem: Probability of A winning four consecutive games -/
theorem prob_A_wins_four_consecutive : ℚ :=
  winProbability ^ 4

/-- Theorem: Probability of needing a fifth game -/
theorem prob_fifth_game : ℚ :=
  1 - 4 * winProbability ^ 4

/-- Theorem: Probability of C winning in the end -/
theorem prob_C_wins : ℚ :=
  7 / 16

/-- Main theorem combining all results -/
theorem badminton_game_probabilities :
  (prob_A_wins_four_consecutive = 1 / 16) ∧
  (prob_fifth_game = 3 / 4) ∧
  (prob_C_wins = 7 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_wins_four_consecutive_prob_fifth_game_prob_C_wins_badminton_game_probabilities_l65_6555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_outscores_B_value_l65_6532

/-- Represents a soccer tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (win_probability : ℚ)
  (h_num_teams : num_teams = 8)
  (h_win_probability : win_probability = 1/2)

/-- The probability that team A finishes with more total points than team B,
    given that A has already won against B -/
noncomputable def probability_A_outscores_B (t : Tournament) : ℚ :=
  523/1024

/-- Theorem stating that the probability of team A finishing with more total points
    than team B is 523/1024, given the tournament conditions and A's initial win against B -/
theorem probability_A_outscores_B_value (t : Tournament) :
  probability_A_outscores_B t = 523/1024 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_outscores_B_value_l65_6532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l65_6567

noncomputable def f (x : ℝ) : ℝ := ((-x^2 - 3*x + 4).sqrt) / (Real.log (x + 1))

theorem domain_of_f :
  {x : ℝ | -x^2 - 3*x + 4 ≥ 0 ∧ x + 1 > 0 ∧ x ≠ 0} = Set.Ioo (-1) 0 ∪ Set.Ioc 0 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l65_6567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_0_30_l65_6576

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
| Miss
| Hit
deriving Repr, DecidableEq

/-- Converts a digit to a ShotOutcome --/
def digitToOutcome (d : Nat) : ShotOutcome :=
  if d ∈ [1, 2, 3, 4] then ShotOutcome.Miss else ShotOutcome.Hit

/-- Represents a group of three shots --/
structure ThreeShots where
  first : ShotOutcome
  second : ShotOutcome
  third : ShotOutcome
deriving Repr

/-- Counts the number of hits in a ThreeShots --/
def countHits (shots : ThreeShots) : Nat :=
  (if shots.first = ShotOutcome.Hit then 1 else 0) +
  (if shots.second = ShotOutcome.Hit then 1 else 0) +
  (if shots.third = ShotOutcome.Hit then 1 else 0)

/-- Converts a three-digit number to a ThreeShots --/
def numberToThreeShots (n : Nat) : ThreeShots :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  { first := digitToOutcome d1
    second := digitToOutcome d2
    third := digitToOutcome d3 }

/-- The list of randomly generated numbers --/
def randomNumbers : List Nat := [907, 966, 191, 925, 271, 932, 812, 458, 569, 683]

/-- Converts the list of random numbers to ThreeShots --/
def simulationResults : List ThreeShots :=
  randomNumbers.map numberToThreeShots

/-- Counts the number of ThreeShots with exactly two hits --/
def countTwoHits (results : List ThreeShots) : Nat :=
  results.filter (fun shots => countHits shots = 2) |>.length

theorem estimated_probability_is_0_30 :
  (countTwoHits simulationResults : ℚ) / simulationResults.length = 3/10 := by
  sorry

#eval (countTwoHits simulationResults : ℚ) / simulationResults.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_probability_is_0_30_l65_6576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_53_multiples_l65_6515

/-- Represents the triangular array of numbers -/
def TriangularArray : Type := Nat → Nat → Nat

/-- The entry in the nth row and kth position of the array -/
def b (n k : Nat) : Nat := 2^(n-1) * (n + 2*k - 1)

/-- The number of entries in the nth row -/
def rowLength (n : Nat) : Nat := 51 - n

theorem triangular_array_53_multiples 
  (arr : TriangularArray)
  (first_row : ∀ k, k ≤ 50 → arr 1 k = 2*k)
  (subsequent_rows : ∀ n k, n > 1 → k ≤ rowLength n → 
    arr n k = arr (n-1) k + arr (n-1) (k+1))
  (entry_formula : ∀ n k, arr n k = b n k) :
  (Finset.sum (Finset.range 51) (λ n ↦ 
    Finset.sum (Finset.range (rowLength n)) (λ k ↦ 
      if (arr n k) % 53 = 0 then 1 else 0))) = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_53_multiples_l65_6515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_four_l65_6506

-- Define the cost function as noncomputable
noncomputable def cost_function (x : ℝ) : ℝ := 900 * (x + 16 / x) + 5800

-- State the theorem
theorem min_cost_at_four :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 5 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 5 → cost_function x ≤ cost_function y) ∧
  x = 4 ∧ cost_function x = 13000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_four_l65_6506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l65_6536

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from point (1,2) to the line 3x+4y+5=0 is 16/5 -/
theorem distance_point_to_specific_line :
  distance_point_to_line 1 2 3 4 5 = 16/5 := by
  -- Unfold the definition of distance_point_to_line
  unfold distance_point_to_line
  -- Simplify the expression
  simp
  -- The rest of the proof is skipped
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_line_l65_6536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l65_6552

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 3) ^ 2 = a 1 * a 13

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The expression to be minimized -/
noncomputable def f (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (2 * S seq n + 16) / (seq.a n + 3)

theorem min_value_is_four (seq : ArithmeticSequence) :
    ∃ n₀ : ℕ, ∀ n : ℕ, f seq n ≥ 4 ∧ f seq n₀ = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_is_four_l65_6552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l65_6583

/-- Represents a hyperbola with parameters a, b, and c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) (x y : ℝ) : Prop :=
  y = (h.b / h.a) * x ∨ y = -(h.b / h.a) * x

/-- The line that intersects the asymptotes -/
def intersecting_line (h : Hyperbola) (x : ℝ) : Prop :=
  x = h.a^2 / h.c

/-- The angle AFB, where A and B are intersection points and F is the right focus -/
noncomputable def angle_AFB (h : Hyperbola) : ℝ :=
  Real.arctan ((h.b * h.c) / (h.c^2 - h.a^2))

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.a^2 / h.b^2)

/-- Theorem stating the range of eccentricity given the conditions -/
theorem eccentricity_range (h : Hyperbola) 
  (angle_condition : π/3 < angle_AFB h ∧ angle_AFB h < π/2) :
  Real.sqrt 2 < eccentricity h ∧ eccentricity h < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l65_6583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_l65_6500

-- Define a type for geometric figures
inductive GeometricFigure
  | EquilateralTriangle
  | RegularPentagon
  | NonSquareRectangle
  | Circle
  | Square

-- Define a custom type for infinity
inductive Infinity
  | infinity

-- Function to get the number of lines of symmetry for each figure
def linesOfSymmetry (figure : GeometricFigure) : ℕ ⊕ Infinity :=
  match figure with
  | .EquilateralTriangle => Sum.inl 3
  | .RegularPentagon => Sum.inl 5
  | .NonSquareRectangle => Sum.inl 2
  | .Circle => Sum.inr Infinity.infinity
  | .Square => Sum.inl 4

-- Define an ordering on ℕ ⊕ Infinity
def orderSymmetry : ℕ ⊕ Infinity → ℕ ⊕ Infinity → Prop
  | Sum.inl n, Sum.inl m => n > m
  | Sum.inr Infinity.infinity, Sum.inl _ => True
  | Sum.inl _, Sum.inr Infinity.infinity => False
  | Sum.inr Infinity.infinity, Sum.inr Infinity.infinity => False

-- Theorem stating that the circle has the greatest number of lines of symmetry
theorem circle_has_most_symmetry :
  ∀ (figure : GeometricFigure), figure ≠ GeometricFigure.Circle →
    orderSymmetry (linesOfSymmetry GeometricFigure.Circle) (linesOfSymmetry figure) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_has_most_symmetry_l65_6500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_rational_roots_l65_6542

def is_valid_cubic_polynomial (a b c : ℚ) : Prop :=
  (a = 0 ∧ b = 0 ∧ c = 0) ∨
  (a = 1 ∧ b = -2 ∧ c = 0) ∨
  (a = 1 ∧ b = -1 ∧ c = -1)

theorem cubic_polynomials_with_rational_roots :
  ∀ (a b c : ℚ),
    (∃ (x y z : ℚ), x^3 + a*x^2 + b*x + c = (x - x)*(x - y)*(x - z)) →
    is_valid_cubic_polynomial a b c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_with_rational_roots_l65_6542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_proposition_two_true_l65_6582

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Four points are coplanar if there exists a plane containing all of them -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
                   a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
                   a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
                   a * p4.x + b * p4.y + c * p4.z + d = 0

/-- Three points are collinear if they lie on the same line -/
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t1 t2 : ℝ), p3 = Point3D.mk (p1.x + t1 * (p2.x - p1.x))
                                 (p1.y + t1 * (p2.y - p1.y))
                                 (p1.z + t1 * (p2.z - p1.z)) ∨
                 p2 = Point3D.mk (p1.x + t2 * (p3.x - p1.x))
                                 (p1.y + t2 * (p3.y - p1.y))
                                 (p1.z + t2 * (p3.z - p1.z))

/-- A point is on a line if it satisfies the line equation -/
def on_line (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk (l.point.x + t * l.direction.x)
                          (l.point.y + t * l.direction.y)
                          (l.point.z + t * l.direction.z)

/-- Two lines are skew if they do not intersect and are not parallel -/
def skew_lines (l1 l2 : Line3D) : Prop :=
  ¬ ∃ (p : Point3D), on_line p l1 ∧ on_line p l2 ∧
  ¬ (l1.direction.x * l2.direction.y = l1.direction.y * l2.direction.x ∧
     l1.direction.x * l2.direction.z = l1.direction.z * l2.direction.x ∧
     l1.direction.y * l2.direction.z = l1.direction.z * l2.direction.y)

theorem converse_proposition_two_true :
  (∀ l1 l2 : Line3D, skew_lines l1 l2 → ¬ ∃ (p : Point3D), on_line p l1 ∧ on_line p l2) ∧
  ¬ (∀ p1 p2 p3 p4 : Point3D,
     (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p3 p4 ∧ ¬ collinear p2 p3 p4)
     → ¬ coplanar p1 p2 p3 p4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_proposition_two_true_l65_6582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l65_6581

/-- The distance between two points in 3D space -/
noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- The theorem stating that the point C(0, 0, 14/9) is equidistant from A(-4, 1, 7) and B(3, 5, -2) -/
theorem equidistant_point :
  let xA : ℝ := -4
  let yA : ℝ := 1
  let zA : ℝ := 7
  let xB : ℝ := 3
  let yB : ℝ := 5
  let zB : ℝ := -2
  let xC : ℝ := 0
  let yC : ℝ := 0
  let zC : ℝ := 14/9
  distance xA yA zA xC yC zC = distance xB yB zB xC yC zC :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l65_6581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l65_6550

noncomputable def i : ℂ := Complex.I

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def is_in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

theorem complex_problem (z : ℂ) (h1 : is_purely_imaginary z) 
  (h2 : ((z + 2) / (1 - i) + z).im = 0) :
  z = -2/3 * i ∧ 
  ∀ m : ℝ, is_in_first_quadrant (((m : ℂ) - z)^2) ↔ m > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l65_6550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l65_6521

/-- The distance between points A and B in kilometers -/
noncomputable def distance : ℝ := 45

/-- Person A's speed relative to Person B's initial speed -/
noncomputable def speed_ratio : ℝ := 1.2

/-- Distance Person B travels before breakdown in kilometers -/
noncomputable def breakdown_distance : ℝ := 5

/-- Fraction of total journey time that the breakdown delay represents -/
noncomputable def delay_fraction : ℝ := 1/6

/-- Person B's speed increase factor after repair -/
noncomputable def speed_increase : ℝ := 1.6

theorem journey_distance :
  ∃ (time_A time_B initial_speed_B : ℝ),
    time_A > 0 ∧ time_B > 0 ∧ initial_speed_B > 0 ∧
    speed_ratio * initial_speed_B * time_A = distance ∧
    initial_speed_B * (time_B - delay_fraction * time_B) + 
      speed_increase * initial_speed_B * (delay_fraction * time_B) = distance ∧
    time_A = time_B ∧
    breakdown_distance = initial_speed_B * (delay_fraction * time_B) := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l65_6521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_delphi_population_2070_l65_6507

-- Define the initial year and population
def initial_year : ℕ := 2020
def initial_population : ℕ := 350

-- Define the doubling period in years
def doubling_period : ℕ := 30

-- Define the target year
def target_year : ℕ := 2070

-- Define the population growth function
noncomputable def population (year : ℕ) : ℝ :=
  initial_population * (2 : ℝ) ^ ((year - initial_year : ℝ) / doubling_period)

-- Theorem statement
theorem delphi_population_2070 :
  population target_year = 700 * (2 : ℝ) ^ (2/3) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_delphi_population_2070_l65_6507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l65_6588

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The proposed tangent line function -/
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

theorem tangent_line_is_correct :
  let (x₀, y₀) := P
  (f x₀ = y₀) ∧                               -- Point P is on the curve
  (f' x₀ = -4) ∧                              -- Slope at P is -4
  (∀ x y, tangent_line x y ↔ y = -4 * x - 1)  -- Tangent line equation is correct
  →
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε),
    x ≠ x₀ → |f x - (f x₀ + f' x₀ * (x - x₀))| < |f' x₀ * (x - x₀)| :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_is_correct_l65_6588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l65_6580

/-- The golden ratio -/
noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

/-- The optimal selection method popularized by Hua Luogeng -/
def optimal_selection_method : Type := Unit

/-- The mathematical concept used in the optimal selection method -/
noncomputable def concept_used (method : optimal_selection_method) : ℝ := golden_ratio

/-- Theorem stating that the optimal selection method uses the golden ratio -/
theorem optimal_selection_uses_golden_ratio (method : optimal_selection_method) :
  concept_used method = golden_ratio := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_selection_uses_golden_ratio_l65_6580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l65_6517

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (-1, 2) to the line y = x is 3√2/2 -/
theorem distance_point_to_line_example : 
  distance_point_to_line (-1) 2 (-1) 1 0 = 3 * Real.sqrt 2 / 2 := by
  sorry

#check distance_point_to_line_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l65_6517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_for_divisibility_by_nine_l65_6572

theorem smallest_digit_for_divisibility_by_nine (d : Nat) :
  (d < 10) →
  (528000 + d * 100 + 46) % 9 = 0 →
  (∀ k < d, (528000 + k * 100 + 46) % 9 ≠ 0) →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_for_divisibility_by_nine_l65_6572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l65_6577

open Real

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 / 2 + (1 - k) * x - k * log x

-- State the theorem
theorem k_range (k : ℝ) :
  (k > 0) →
  (∃ x₀ : ℝ, f k x₀ < 3/2 - k^2) →
  (0 < k ∧ k < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l65_6577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l65_6514

def sequencePattern : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | 5 => 17
  | 6 => 23
  | _ => 0

theorem sequence_pattern : 
  (∀ n : ℕ, n < 6 → sequencePattern (n + 1) = sequencePattern n + (n + 1)) ∧
  sequencePattern 3 = 8 ∧
  sequencePattern 4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_pattern_l65_6514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle1_correct_triangle2_sol1_correct_triangle2_sol2_correct_l65_6544

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Triangle 1
noncomputable def triangle1 : Triangle := {
  a := 2,
  b := Real.sqrt 3,
  c := 1,
  A := Real.pi / 2,  -- 90°
  B := Real.pi / 3,  -- 60°
  C := Real.pi / 6   -- 30°
}

-- Triangle 2 (Solution 1)
noncomputable def triangle2_sol1 : Triangle := {
  a := 2,
  b := Real.sqrt 3 + 1,
  c := Real.sqrt 6,
  A := Real.pi / 4,    -- 45°
  B := 5 * Real.pi / 12,  -- 75°
  C := Real.pi / 3     -- 60°
}

-- Triangle 2 (Solution 2)
noncomputable def triangle2_sol2 : Triangle := {
  a := 2,
  b := Real.sqrt 3 - 1,
  c := Real.sqrt 6,
  A := Real.pi / 4,    -- 45°
  B := Real.pi / 12,   -- 15°
  C := 2 * Real.pi / 3 -- 120°
}

-- Law of Sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧
  t.b / Real.sin t.B = t.c / Real.sin t.C ∧
  t.c / Real.sin t.C = t.a / Real.sin t.A

-- Sum of angles in a triangle is π (180°)
def angleSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

-- Pythagorean theorem
def pythagorean (t : Triangle) : Prop :=
  t.A = Real.pi / 2 → t.a^2 = t.b^2 + t.c^2

theorem triangle1_correct : 
  lawOfSines triangle1 ∧ 
  angleSum triangle1 ∧ 
  pythagorean triangle1 := by sorry

theorem triangle2_sol1_correct : 
  lawOfSines triangle2_sol1 ∧ 
  angleSum triangle2_sol1 := by sorry

theorem triangle2_sol2_correct : 
  lawOfSines triangle2_sol2 ∧ 
  angleSum triangle2_sol2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle1_correct_triangle2_sol1_correct_triangle2_sol2_correct_l65_6544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l65_6548

/-- Two parallel lines with a specific distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  m_pos : m > 0
  parallel : 1 / 2 = -2 / n
  distance : |m + 3| / Real.sqrt 5 = Real.sqrt 5

/-- The sum of coefficients m and n for parallel lines with given properties -/
theorem parallel_lines_sum (lines : ParallelLines) : lines.m + lines.n = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l65_6548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_line_intersects_circle_l65_6522

-- Define the line L
def line_L (t : ℝ) : ℝ × ℝ := (t, 1 + 2*t)

-- Define the circle C in polar form
noncomputable def circle_C_polar (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin (θ + Real.pi/4)

-- Define the circle C in Cartesian form
def circle_C_cartesian (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

-- Theorem 1: The Cartesian equation of circle C is (x - 1)² + (y - 1)² = 2
theorem circle_C_equation : ∀ x y : ℝ, 
  (∃ θ : ℝ, x^2 + y^2 = (circle_C_polar θ)^2 ∧ 
             x = (circle_C_polar θ) * Real.cos θ ∧ 
             y = (circle_C_polar θ) * Real.sin θ) ↔ 
  circle_C_cartesian x y := by sorry

-- Theorem 2: Line L intersects circle C
theorem line_intersects_circle : ∃ t : ℝ, 
  let (x, y) := line_L t
  circle_C_cartesian x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_equation_line_intersects_circle_l65_6522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_boundary_equation_l65_6593

-- Define the sphere
def sphere_radius : ℝ := 2
def sphere_center : Fin 3 → ℝ := ![0, 0, 2]

-- Define the light source
def light_source : Fin 3 → ℝ := ![0, -2, 4]

-- Define a point on the shadow boundary
def shadow_point (x y : ℝ) : Fin 3 → ℝ := ![x, y, 0]

-- Define the vector from light source to sphere center
def vector_PO : Fin 3 → ℝ := ![0, 2, -2]

-- Define the vector from light source to shadow point
def vector_PX (x y : ℝ) : Fin 3 → ℝ := ![x, y+2, -4]

-- Theorem statement
theorem shadow_boundary_equation :
  ∀ x : ℝ, ∃ y : ℝ,
    (vector_PO • vector_PX x y) = 0 →
    y = -10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shadow_boundary_equation_l65_6593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l65_6561

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * |x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂)|

theorem triangle_area_is_six :
  let line1 : Line := { point := (2, 2), slope := 1/2 }
  let line2 : Line := { point := (2, 2), slope := 2 }
  let line3 := fun (x y : ℝ) => x + y = 10
  let a := (2, 2)
  let b := (4, 6)  -- Intersection of line2 and line3
  let c := (6, 4)  -- Intersection of line1 and line3
  triangleArea a b c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_six_l65_6561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_class_drives_l65_6566

theorem probability_second_class_drives (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ) 
  (h1 : total = 50) 
  (h2 : first_class = 45) 
  (h3 : second_class = 5) 
  (h4 : selected = 3) 
  (h5 : total = first_class + second_class) :
  (1 : ℚ) - (Nat.choose first_class selected : ℚ) / (Nat.choose total selected : ℚ) = 
  (Nat.choose total selected - Nat.choose first_class selected : ℚ) / (Nat.choose total selected : ℚ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_second_class_drives_l65_6566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l65_6562

def U : Set ℕ := {x : ℕ | x > 0 ∧ x ≤ 5}
def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_in_U : (U \ M) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l65_6562
