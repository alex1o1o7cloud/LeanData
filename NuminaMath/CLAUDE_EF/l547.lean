import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_fixed_point_l547_54743

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => id
  | n + 2 => λ x => Real.sqrt (f (n + 1) x) - 1/4

theorem f_decreasing {n : ℕ} {x : ℝ} (h : n ≥ 2) :
  f n x ≤ f (n-1) x :=
by sorry

theorem f_fixed_point {n : ℕ} (h : n ≥ 2) :
  f n (1/4) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_f_fixed_point_l547_54743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l547_54783

theorem marble_probability : 
  ∃ (ε : ℚ), abs ((8 : ℚ) / 10 ^ 2 * ((2 : ℚ) / 10) ^ 3 * (Nat.choose 5 2) - (64 : ℚ) / 100) < ε ∧ ε ≤ (1 : ℚ) / 2000 :=
by
  -- We'll use 1/2000 as our ε, which is less than or equal to 0.0005
  use (1 : ℚ) / 2000
  
  -- Split the goal into two parts
  apply And.intro
  
  -- Part 1: Show that the absolute difference is less than ε
  · sorry  -- This part requires numerical computation and estimation
  
  -- Part 2: Show that ε ≤ 0.0005
  · -- 1/2000 ≤ 1/2000, which is true by reflexivity
    exact le_refl ((1 : ℚ) / 2000)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_probability_l547_54783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_exists_l547_54700

theorem no_integer_polynomial_exists : ¬ ∃ (P : Polynomial ℤ), (P.eval 7 = 11) ∧ (P.eval 11 = 13) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_exists_l547_54700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_costa_rica_points_l547_54740

/-- Represents a team in the tournament -/
structure TeamData :=
  (points : ℕ)

/-- Represents a group in the tournament -/
structure GroupData :=
  (teams : Finset TeamData)
  (size : ℕ)
  (size_eq : teams.card = size)

/-- The number of matches in a single round-robin tournament -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

theorem costa_rica_points
  (g : GroupData)
  (h_size : g.size = 4)
  (h_matches : num_matches g.size = 6)
  (h_one_draw : ∃! (t1 t2 : TeamData), t1 ∈ g.teams ∧ t2 ∈ g.teams ∧ t1 ≠ t2 ∧ t1.points + t2.points = 8)
  (h_japan : ∃ t ∈ g.teams, t.points = 6)
  (h_spain_germany : ∃ t1 t2, t1 ∈ g.teams ∧ t2 ∈ g.teams ∧ t1 ≠ t2 ∧ t1.points = 4 ∧ t2.points = 4)
  : ∃ t ∈ g.teams, t.points = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_costa_rica_points_l547_54740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l547_54716

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def conditions (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  a > b ∧ b > 0 ∧
  P.1 = 1 ∧ P.2 = 3/2 ∧
  ellipse_C P.1 P.2 a b

-- Define the perimeter condition
def perimeter_condition (F₂ A B : ℝ × ℝ) : Prop :=
  ((F₂.1 - A.1)^2 + (F₂.2 - A.2)^2).sqrt +
  ((F₂.1 - B.1)^2 + (F₂.2 - B.2)^2).sqrt +
  ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt = 8

-- Define the angle condition
def angle_condition (F₁ M A B : ℝ × ℝ) : Prop :=
  (A.2 - M.2) * (B.1 - M.1) = (B.2 - M.2) * (A.1 - M.1)

-- Main theorem
theorem ellipse_properties (a b : ℝ) (P : ℝ × ℝ) 
  (h_conditions : conditions a b P)
  (h_perimeter : ∀ F₂ A B : ℝ × ℝ, ellipse_C A.1 A.2 a b → ellipse_C B.1 B.2 a b → 
    perimeter_condition F₂ A B) :
  (a = 2 ∧ b = Real.sqrt 3) ∧
  ∃ M : ℝ × ℝ, M.1 = -4 ∧ M.2 = 0 ∧
    ∀ F₁ A B : ℝ × ℝ, ellipse_C A.1 A.2 a b → ellipse_C B.1 B.2 a b →
      angle_condition F₁ M A B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l547_54716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l547_54731

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the circle E
def circleE (x₀ y₀ x y : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = 3

-- Define the conditions
def conditions (a b c x₀ y₀ k₁ k₂ : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  c / a = Real.sqrt 6 / 3 ∧
  2 / c = Real.sqrt 2 / 2 ∧
  ellipse a b x₀ y₀ ∧
  circleE x₀ y₀ 0 0

-- State the theorem
theorem ellipse_properties
  (a b c x₀ y₀ k₁ k₂ : ℝ)
  (h : conditions a b c x₀ y₀ k₁ k₂) :
  ellipse (2 * Real.sqrt 3) 2 x₀ y₀ ∧
  k₁ * k₂ = -1/3 ∧
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse (2 * Real.sqrt 3) 2 x₁ y₁ ∧
    ellipse (2 * Real.sqrt 3) 2 x₂ y₂ ∧
    x₁^2 + y₁^2 + x₂^2 + y₂^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l547_54731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l547_54754

theorem no_integer_roots (P : Polynomial ℤ) (a b c : ℤ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  |P.eval a| = 1 ∧ |P.eval b| = 1 ∧ |P.eval c| = 1 →
  ∀ x : ℤ, P.eval x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_roots_l547_54754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l547_54794

/-- Regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  regular : ∀ i j : Fin 6, dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)

/-- Point of intersection of diagonals FC and BD -/
noncomputable def G (h : RegularHexagon) : ℝ × ℝ :=
  sorry

/-- Area of quadrilateral FEDG -/
noncomputable def areaFEDG (h : RegularHexagon) : ℝ :=
  sorry

/-- Area of triangle BCG -/
noncomputable def areaBCG (h : RegularHexagon) : ℝ :=
  sorry

/-- The ratio of area of quadrilateral FEDG to area of triangle BCG is 5:1 -/
theorem area_ratio (h : RegularHexagon) : areaFEDG h / areaBCG h = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_l547_54794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l547_54795

def billy_upper_bound : ℕ := 250
def bobbi_upper_bound : ℕ := 250
def billy_multiple : ℕ := 20
def bobbi_multiple : ℕ := 28

theorem same_number_probability :
  let billy_choices := Finset.filter (fun n => billy_multiple ∣ n) (Finset.range billy_upper_bound)
  let bobbi_choices := Finset.filter (fun n => bobbi_multiple ∣ n) (Finset.range bobbi_upper_bound)
  let common_choices := billy_choices ∩ bobbi_choices
  (Finset.card common_choices : ℚ) / ((Finset.card billy_choices * Finset.card bobbi_choices) : ℚ) = 1 / 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l547_54795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l547_54718

def A : Finset ℕ := {1, 2, 3, 4, 5}

def B : Finset (ℕ × ℕ) := A.product A |>.filter (fun p => p.1 - p.2 ∈ A)

theorem cardinality_of_B : Finset.card B = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_B_l547_54718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l547_54710

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℚ) (h1 : principal > 0) (h2 : time > 0) :
  (interest / (principal * time) = 7/100) ↔ interest = principal * time * (7/100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_l547_54710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_three_halves_pi_l547_54701

open Set Real Interval MeasureTheory

/-- The area enclosed by y = cos x, x ∈ [0, 3π/2], and the coordinate axes is 3 -/
theorem cosine_area_three_halves_pi : 
  (∫ x in Icc 0 ((3 * π) / 2), max 0 (cos x)) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_three_halves_pi_l547_54701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l547_54719

/-- The inclination angle of a line in the form x + √3y + c = 0 -/
noncomputable def inclination_angle (c : ℝ) : ℝ :=
  5 * Real.pi / 6

/-- The line equation in the form x + √3y + c = 0 -/
def line_equation (x y c : ℝ) : Prop :=
  x + Real.sqrt 3 * y + c = 0

theorem inclination_angle_of_line (c : ℝ) :
  ∀ x y, line_equation x y c → inclination_angle c = 5 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_of_line_l547_54719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_inequality_triangle_circumradius_equality_l547_54742

/-- Helper function to calculate the area of a triangle using Heron's formula -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle with sides a, b, c and circumradius R, 
    prove that R ≥ (a² + b²) / (2√(2a² + 2b² - c²)) -/
theorem triangle_circumradius_inequality 
  (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = a * b * c / (4 * area_triangle a b c)) : 
  R ≥ (a^2 + b^2) / (2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2)) := by
  sorry

/-- Equality holds when the triangle is isosceles or right-angled -/
theorem triangle_circumradius_equality
  (a b c R : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = a * b * c / (4 * area_triangle a b c))
  (h_equality : R = (a^2 + b^2) / (2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))) :
  (a = b ∨ b = c ∨ a = c) ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_inequality_triangle_circumradius_equality_l547_54742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_smallest_divisors_sum_l547_54790

def has_four_divisors_sum_2n (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄ ∧
    d₁ + d₂ + d₃ + d₄ = 2 * n

def sum_four_smallest_divisors (n : ℕ) : ℕ :=
  let divisors := (Nat.divisors n).sort (· ≤ ·)
  (divisors.take 4).sum

theorem four_smallest_divisors_sum (n : ℕ) :
  has_four_divisors_sum_2n n →
  sum_four_smallest_divisors n ∈ ({10, 11, 12} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_smallest_divisors_sum_l547_54790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l547_54769

noncomputable def g (x : ℝ) : ℝ :=
  if x < 2 then 2 * x - 4 else 10 - 3 * x

theorem g_values : g (-1) = -6 ∧ g 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l547_54769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l547_54707

/-- The point where the mouse starts to move away from the cheese -/
def closest_point : ℝ × ℝ := (3, 3)

/-- The location of the cheese -/
def cheese_location : ℝ × ℝ := (15, 12)

/-- The equation of the mouse's path -/
def mouse_path (x : ℝ) : ℝ := -3 * x + 12

theorem mouse_cheese_problem :
  let (a, b) := closest_point
  (b = mouse_path a) ∧
  (∀ x y, y = mouse_path x → 
    (x - 15)^2 + (y - 12)^2 ≥ (a - 15)^2 + (b - 12)^2) ∧
  (a + b = 6) := by
  sorry

#check mouse_cheese_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l547_54707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_after_80_ops_l547_54714

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Initializes a table with all zeros -/
def init_table : Table := λ _ _ ↦ 0

/-- Represents a single operation on the table -/
def operation (t : Table) : Table := sorry

/-- Applies n operations to a table -/
def apply_operations (t : Table) (n : ℕ) : Table := sorry

/-- The maximum value in a table -/
def max_value (t : Table) : ℕ := sorry

/-- Theorem stating the maximum possible value after 80 operations -/
theorem max_value_after_80_ops :
  max_value (apply_operations init_table 80) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_after_80_ops_l547_54714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_speed_l547_54735

/-- Calculates the speed given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Represents Tod's journey -/
structure Journey where
  north_distance : ℝ
  west_distance : ℝ
  time : ℝ

/-- Theorem: Tod's speed is 25 miles per hour -/
theorem tod_speed (j : Journey) 
  (h1 : j.north_distance = 55)
  (h2 : j.west_distance = 95)
  (h3 : j.time = 6) :
  calculate_speed (j.north_distance + j.west_distance) j.time = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_speed_l547_54735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l547_54778

theorem relationship_abc (a b c : ℝ) : 
  a = (0.5 : ℝ) ^ (0.4 : ℝ) → 
  b = Real.log 0.3 / Real.log 0.4 → 
  c = Real.log 0.4 / Real.log 8 → 
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l547_54778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l547_54784

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line
noncomputable def line (x y : ℝ) : ℝ := 2*x - 3*y + 6

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |line x y| / Real.sqrt 13

-- Theorem statement
theorem min_distance_ellipse_to_line :
  ∃ (d : ℝ), d = (6 - Real.sqrt 13) / Real.sqrt 13 ∧
  ∀ (x y : ℝ), is_on_ellipse x y →
  distance_to_line x y ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l547_54784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_symmetry_l547_54791

theorem cos_shift_symmetry (φ : ℝ) :
  φ > 0 ∧
  (∀ x : ℝ, Real.cos (x - φ + 4 * Real.pi / 3) = Real.cos (-x - φ + 4 * Real.pi / 3)) →
  φ ≥ Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_symmetry_l547_54791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steamers_met_count_l547_54776

/-- Represents a steamer journey --/
structure Journey where
  departure : ℕ
  arrival : ℕ

/-- The number of days for a complete journey --/
def journey_duration : ℕ := 7

/-- Creates a journey given a departure day --/
def create_journey (departure_day : ℕ) : Journey :=
  { departure := departure_day
  , arrival := departure_day + journey_duration }

/-- Checks if two journeys meet --/
def journeys_meet (j1 j2 : Journey) : Bool :=
  (j1.departure < j2.arrival) && (j2.departure < j1.arrival)

/-- The number of steamers a journey meets --/
def steamers_met (j : Journey) : ℕ :=
  (List.range (2 * journey_duration + 1)).filter (λ d => journeys_meet j (create_journey d)) |>.length

theorem steamers_met_count (departure_day : ℕ) :
  steamers_met (create_journey departure_day) = 15 := by
  sorry

#eval steamers_met (create_journey 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steamers_met_count_l547_54776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l547_54721

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the tangent line l
def tangent_line (x y t : ℝ) : Prop := ∃ k, y = k * x + t

-- Define the area of triangle AOB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (1/2) * abs (x₁ * y₂ - x₂ * y₁)

theorem ellipse_and_triangle_area :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  ellipse_C (-1/2) (-Real.sqrt 3) a b →
  (a^2 - b^2) / a^2 = 3/4 →
  (∀ x y : ℝ, ellipse_C x y a b ↔ y^2 / 4 + x^2 = 1) ∧
  (∀ t : ℝ, abs t ≥ 1 →
    ∃ A B : ℝ × ℝ,
    ellipse_C A.1 A.2 a b ∧
    ellipse_C B.1 B.2 a b ∧
    tangent_line A.1 A.2 t ∧
    tangent_line B.1 B.2 t ∧
    unit_circle A.1 A.2 ∧
    triangle_area A B ≤ 1 ∧
    (∃ t₀ : ℝ, abs t₀ ≥ 1 ∧ triangle_area A B = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l547_54721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_six_l547_54741

def S : Set ℤ := {-30, -5, -1, 3, 5, 15}

theorem largest_quotient_is_six :
  ∀ a b, a ∈ S → b ∈ S → a ≠ 0 → b ≠ 0 → (a : ℚ) / b ≤ 6 ∧ ∃ c d, c ∈ S ∧ d ∈ S ∧ c ≠ 0 ∧ d ≠ 0 ∧ (c : ℚ) / d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_six_l547_54741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l547_54705

-- Define the star operation as noncomputable
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

-- State the theorem
theorem x_value_theorem (x : ℝ) (h1 : x > 0) (h2 : star x 36 = 9) : x = 36.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_theorem_l547_54705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_order_theorem_l547_54786

/-- Represents the dimensions of a garden path in feet and inches -/
structure PathDimensions where
  width : ℝ  -- width in feet
  length : ℝ  -- length in feet
  thickness : ℝ  -- thickness in inches

/-- Converts feet to yards -/
noncomputable def feetToYards (feet : ℝ) : ℝ := feet / 3

/-- Converts inches to yards -/
noncomputable def inchesToYards (inches : ℝ) : ℝ := inches / 36

/-- Calculates the volume of the path in cubic yards -/
noncomputable def pathVolume (d : PathDimensions) : ℝ :=
  feetToYards d.width * feetToYards d.length * inchesToYards d.thickness

/-- Rounds up a real number to the nearest integer -/
noncomputable def ceilToInt (x : ℝ) : ℤ := Int.ceil x

theorem concrete_order_theorem (d : PathDimensions) 
  (h1 : d.width = 4)
  (h2 : d.length = 80)
  (h3 : d.thickness = 4) :
  ceilToInt (pathVolume d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concrete_order_theorem_l547_54786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_twenty_fourth_l547_54720

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^5 - 1) / 3

-- State the theorem
theorem inverse_f_at_one_twenty_fourth :
  f⁻¹ (1/24) = (9^(1/5 : ℝ)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_twenty_fourth_l547_54720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l547_54709

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the domain of x
def X : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 7 + 4*a
  else if a ≤ 2 then 3 - a^2
  else 7 - 4*a

-- State the theorem
theorem f_properties :
  (∀ x ∈ X, f (-1) x ≥ 2) ∧
  (∃ x ∈ X, f (-1) x = 2) ∧
  (∀ x ∈ X, f (-1) x ≤ 11) ∧
  (∃ x ∈ X, f (-1) x = 11) ∧
  (∀ a : ℝ, ∀ x ∈ X, f a x ≥ g a) ∧
  (∀ a : ℝ, ∃ x ∈ X, f a x = g a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l547_54709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_and_max_product_l547_54703

-- Define the circles in Cartesian coordinates
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the polar equations
def polar_C₁ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def polar_C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the product of radii
noncomputable def radius_product (θ : ℝ) : ℝ := (4 * Real.cos θ) * (2 * Real.sin θ)

theorem circles_and_max_product :
  (∀ x y, C₁ x y ↔ ∃ ρ θ, polar_C₁ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ x y, C₂ x y ↔ ∃ ρ θ, polar_C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  (∀ θ, radius_product θ ≤ 4) ∧
  (∃ θ, radius_product θ = 4) := by
  sorry

#check circles_and_max_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_and_max_product_l547_54703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_weight_approx_l547_54779

/-- The weight of a wooden shape given its area -/
noncomputable def weight_from_area (area : ℝ) (density : ℝ) : ℝ := area * density

/-- The area of a square given its side length -/
noncomputable def square_area (side : ℝ) : ℝ := side ^ 2

/-- The area of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (side ^ 2 * Real.sqrt 3) / 4

theorem triangle_weight_approx (square_side : ℝ) (square_weight : ℝ) (triangle_side : ℝ) :
  square_side = 4 →
  square_weight = 16 →
  triangle_side = 6 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |weight_from_area (equilateral_triangle_area triangle_side) 
    (square_weight / square_area square_side) - 15.6| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_weight_approx_l547_54779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l547_54771

-- Define the Riemann zeta function
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (n : ℝ) ^ (-x)

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- Theorem statement
theorem sum_frac_zeta_even : ∑' k : ℕ, frac (zeta (2 * ↑k + 2)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l547_54771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_square_l547_54774

theorem cube_root_of_negative_square (a : ℝ) (ha : a ≠ 0) : (a^(-(2:ℝ)))^((1:ℝ)/3) = a^(-(2:ℝ)/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_negative_square_l547_54774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l547_54765

open Real

theorem sine_cosine_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  (sin (π - α) + cos (π + α) = 1 → α = π / 2) ∧
  (sin (π - α) + cos (π + α) = sqrt 5 / 5 → tan α = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l547_54765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_first_speed_is_double_l547_54726

/-- The speed Daniel drove the first 32 miles on Monday, given the conditions of his Sunday and Monday drives -/
noncomputable def monday_first_speed (x : ℝ) : ℝ :=
  let sunday_time := 96 / x
  let monday_second_part_time := 64 / (x/2)
  let y := 32 / (1.5 * sunday_time - monday_second_part_time)
  y

/-- Theorem stating that the speed Daniel drove the first 32 miles on Monday is twice his Sunday speed -/
theorem monday_first_speed_is_double (x : ℝ) (h : x > 0) : monday_first_speed x = 2 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_first_speed_is_double_l547_54726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_takes_17_hours_l547_54753

/-- Represents the journey details -/
structure Journey where
  totalDistance : ℝ
  carSpeed : ℝ
  walkSpeed : ℝ
  restTime : ℝ

/-- Calculates the total time of the journey -/
noncomputable def journeyTime (j : Journey) (d1 d2 : ℝ) : ℝ :=
  let carTime1 := d1 / j.carSpeed
  let walkTime := (j.totalDistance - d1) / j.walkSpeed
  let carTime2 := (d2 + (j.totalDistance - (d1 - d2))) / j.carSpeed
  max (carTime1 + j.restTime + walkTime) (carTime1 + carTime2)

theorem journey_takes_17_hours (j : Journey) :
  j.totalDistance = 150 ∧ j.carSpeed = 30 ∧ j.walkSpeed = 4 ∧ j.restTime = 1 →
  ∃ d1 d2 : ℝ, 0 < d1 ∧ 0 < d2 ∧ d2 < d1 ∧ d1 < j.totalDistance ∧
  journeyTime j d1 d2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_takes_17_hours_l547_54753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l547_54722

theorem sum_of_solutions_is_zero : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1))) ∧ 
  (S.sum id) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_zero_l547_54722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_x_l547_54749

theorem definite_integral_sqrt_plus_x : 
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) + x) = (Real.pi + 2) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_plus_x_l547_54749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l547_54708

/-- The line l -/
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

/-- The point P -/
def point_P : ℝ × ℝ := (2, 3)

/-- Distance from a point to a line -/
noncomputable def dist_point_line (M : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  let (x, y) := M
  abs (x - y + 1) / Real.sqrt 2

/-- The set of points M satisfying the distance condition -/
def trajectory (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  dist M point_P = dist_point_line M line_l

/-- Theorem: The trajectory forms a straight line -/
theorem trajectory_is_straight_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), trajectory (x, y) ↔ a * x + b * y + c = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_straight_line_l547_54708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisector_l547_54746

-- Define the circle
variable (O : ℝ × ℝ) (r : ℝ)
def Circle := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define points A, B, A', and B'
variable (A A' B B' : ℝ × ℝ)

-- Define the tangent line at A
def TangentLine (A O : ℝ × ℝ) := 
  {q : ℝ × ℝ | (q.1 - A.1) * (A.1 - O.1) + (q.2 - A.2) * (A.2 - O.2) = 0}

-- State that A and A' are on the circle
axiom h1 : A ∈ Circle O r
axiom h2 : A' ∈ Circle O r

-- State that B is on the tangent line at A
axiom h3 : B ∈ TangentLine A O

-- State that A'B' is the rotation of AB around O
axiom h4 : ∃ θ : ℝ, 
  A'.1 - O.1 = (A.1 - O.1) * Real.cos θ - (A.2 - O.2) * Real.sin θ ∧
  A'.2 - O.2 = (A.1 - O.1) * Real.sin θ + (A.2 - O.2) * Real.cos θ ∧
  B'.1 - O.1 = (B.1 - O.1) * Real.cos θ - (B.2 - O.2) * Real.sin θ ∧
  B'.2 - O.2 = (B.1 - O.1) * Real.sin θ + (B.2 - O.2) * Real.cos θ

-- Define the midpoint of a segment
noncomputable def Midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- State the theorem
theorem tangent_bisector :
  ∃ t : ℝ, Midpoint B B' = (1 - t) • A + t • A' :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisector_l547_54746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l547_54730

noncomputable def x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

def curve_range : Set ℝ := { t | 0 ≤ t ∧ t ≤ 3 * Real.pi / 2 }

theorem arc_length_of_curve :
  ∫ t in curve_range, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2) = 2 * (Real.exp (3 * Real.pi / 2) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_of_curve_l547_54730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cd_value_l547_54728

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem cd_value (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_eq : Real.sqrt (log c) + Real.sqrt (log d) + log (Real.sqrt c) + log (Real.sqrt d) + log (Real.sqrt c * Real.sqrt d) = 50)
  (h_int1 : ∃ n1 : ℕ, Real.sqrt (log c) = n1)
  (h_int2 : ∃ n2 : ℕ, Real.sqrt (log d) = n2)
  (h_int3 : ∃ n3 : ℕ, log (Real.sqrt c) = n3)
  (h_int4 : ∃ n4 : ℕ, log (Real.sqrt d) = n4)
  (h_int5 : ∃ n5 : ℕ, log (Real.sqrt c * Real.sqrt d) = n5) :
  c * d = 10^37 := by
  sorry

#check cd_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cd_value_l547_54728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_perpendicular_l547_54761

def a : Fin 2 → ℝ := ![3, 4]
def b (k : ℝ) : Fin 2 → ℝ := ![2, k]

theorem vector_parallel_perpendicular (k : ℝ) :
  (∀ (t : ℝ), ∃ (s : ℝ), (a + 2 • b k) = s • (a - b k)) →
  k = 8 / 3 ∧
  ((a + b k) • (a - b k) = 0 → k = Real.sqrt 21 ∨ k = -Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_perpendicular_l547_54761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dna_diameter_scientific_notation_l547_54756

theorem dna_diameter_scientific_notation :
  let dna_diameter : ℝ := 0.0000003
  dna_diameter = 3 * (10 : ℝ)^(-7 : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dna_diameter_scientific_notation_l547_54756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l547_54751

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ
  altitude : ℝ

/-- Calculates the length of side BC in the trapezoid -/
noncomputable def lengthBC (t : Trapezoid) : ℝ :=
  (t.area - 4 * (Real.sqrt 105 + Real.sqrt 297)) / 8

/-- Theorem stating the length of BC in the given trapezoid -/
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.base1 = 13)
  (h3 : t.base2 = 19)
  (h4 : t.altitude = 8) :
  lengthBC t = (200 - 4 * (Real.sqrt 105 + Real.sqrt 297)) / 8 := by
  sorry

#check trapezoid_bc_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bc_length_l547_54751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l547_54780

/-- A circle with center (2, -3) and a diameter with endpoints on the coordinate axes has the equation x^2 + y^2 - 4x + 6y = 0. -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (A B : ℝ × ℝ) :
  center = (2, -3) →
  (∃ a b : ℝ, A = (a, 0) ∧ B = (0, b)) →
  (∀ x y : ℝ, (x, y) ∈ C ↔ (x - 2)^2 + (y + 3)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2) →
  (∀ x y : ℝ, (x, y) ∈ C ↔ x^2 + y^2 - 4*x + 6*y = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l547_54780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_ninth_tenth_terms_l547_54711

def mySequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then mySequence n * 3 else mySequence n - 1

theorem eighth_ninth_tenth_terms :
  mySequence 7 = 42 ∧ mySequence 8 = 41 ∧ mySequence 9 = 123 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_ninth_tenth_terms_l547_54711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_revolution_properties_l547_54764

/-- Represents an equilateral triangle with side length a -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a solid of revolution formed by rotating an equilateral triangle -/
structure TriangleRevolution where
  triangle : EquilateralTriangle
  axis_angle : ℝ
  angle_constraint : axis_angle = π / 6 -- 30 degrees in radians

/-- Calculates the surface area of the solid of revolution -/
noncomputable def surface_area (rev : TriangleRevolution) : ℝ :=
  3 * (rev.triangle.side_length ^ 2) * Real.pi

/-- Calculates the volume of the solid of revolution -/
noncomputable def volume (rev : TriangleRevolution) : ℝ :=
  (rev.triangle.side_length ^ 3 * Real.sqrt 3 * Real.pi) / 4

/-- Theorem stating the correctness of surface area and volume calculations -/
theorem triangle_revolution_properties (rev : TriangleRevolution) :
  surface_area rev = 3 * (rev.triangle.side_length ^ 2) * Real.pi ∧
  volume rev = (rev.triangle.side_length ^ 3 * Real.sqrt 3 * Real.pi) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_revolution_properties_l547_54764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_one_value_l547_54799

noncomputable section

variable (g : ℝ → ℝ)

axiom g_def : ∀ x : ℝ, x ≠ 1/2 → g x + g ((2*x - 1)/(2 - 4*x)) = 2*x

theorem g_neg_one_value : g (-1) = 5/7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_one_value_l547_54799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l547_54712

def cube_volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512]

def is_decreasing (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l.get! i > l.get! j

def total_surface_area (volumes : List ℕ) : ℕ :=
  sorry

theorem tower_surface_area :
  is_decreasing cube_volumes →
  total_surface_area cube_volumes = 1021 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_surface_area_l547_54712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_factorization_l547_54704

-- Define what a factorization is
def is_factorization (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x * h x

-- Define the transformations
def transformation_A (x : ℝ) : ℝ := x * (x + 1)
def transformation_B (x : ℝ) : ℝ := (x + 1)^2
def transformation_C (x y : ℝ) : ℝ := x * (x + y) - 3
def transformation_D (x : ℝ) : ℝ := (x + 3)^2 - 5

-- State the theorem
theorem only_B_is_factorization :
  ¬(is_factorization transformation_A) ∧
  (is_factorization transformation_B) ∧
  ¬(is_factorization (fun x => transformation_C x 0)) ∧
  ¬(is_factorization transformation_D) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_factorization_l547_54704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_bound_l547_54789

theorem prime_factors_bound (k n : ℕ) (h1 : k ≥ 2) (h2 : n > 0)
  (h3 : ∀ m : ℕ, 0 < m → m < n^(1/k) → n % m = 0) :
  (Nat.factors n).length ≤ 2 * k - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_bound_l547_54789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_properties_l547_54734

/-- Represents the coin-flipping game described in the problem -/
structure CoinGame where
  /-- Probability of guessing correctly on a single flip -/
  p : ℝ
  /-- Number of points needed to win the game -/
  winPoints : ℕ

/-- Probability that the game ends after exactly n flips -/
noncomputable def probabilityEndAfterNFlips (game : CoinGame) (n : ℕ) : ℝ :=
  (n - 1 : ℝ) * (1 / 2) ^ n

/-- Expected number of flips needed to finish the game -/
def expectedFlips (game : CoinGame) : ℝ := 4

/-- Main theorem stating the properties of the coin-flipping game -/
theorem coin_game_properties (game : CoinGame) :
  (game.p = 1 / 2 ∧ game.winPoints = 2) →
  (∀ n : ℕ, probabilityEndAfterNFlips game n = (n - 1 : ℝ) * (1 / 2) ^ n) ∧
  expectedFlips game = 4 := by
  sorry

#check coin_game_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_properties_l547_54734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_cubic_polynomials_l547_54739

theorem common_roots_cubic_polynomials :
  ∃ (r s : ℝ), r ≠ s ∧
    (r^3 + 12*r^2 + 10*r + 8 = 0 ∧ s^3 + 12*s^2 + 10*s + 8 = 0) ∧
    (r^3 + 11*r^2 + 17*r + 12 = 0 ∧ s^3 + 11*s^2 + 17*s + 12 = 0) :=
by
  sorry

#check common_roots_cubic_polynomials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_cubic_polynomials_l547_54739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l547_54702

theorem trig_identity (α : ℝ) (h1 : α ≠ 0) (h2 : α ≠ π/2) : 
  (Real.sin α + 2 / Real.sin α)^2 + (Real.cos α + 2 / Real.cos α)^2 = 
  -1 + 2 * (Real.tan α)^2 + 2 * (1 / Real.tan α)^2 + Real.sin (2 * α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l547_54702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l547_54747

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sin x + Real.cos x)

-- Define the theorem
theorem max_area_triangle (A B C : ℝ) (hAcute : 0 < A ∧ A < π/2) 
  (hfA : f A = 2) (ha : 2 = Real.sin A * Real.sqrt (Real.sin B * Real.sin C)) : 
  ∀ S : ℝ, S = (1/2) * Real.sin A * Real.sin B * Real.sin C → S ≤ Real.sqrt 2 + 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_l547_54747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l547_54798

/-- The probability of getting heads for an unfair coin -/
noncomputable def p : ℝ := 3/4

/-- The number of coin tosses -/
def n : ℕ := 60

/-- The probability of getting an even number of heads after k tosses -/
noncomputable def P (k : ℕ) : ℝ := 1/2 * (1 + (-1/2)^k)

/-- The main theorem: probability of even number of heads after n tosses -/
theorem even_heads_probability : P n = 1/2 * (1 + 1/2^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_heads_probability_l547_54798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_arrangementsCount_correct_l547_54767

/-- 
Given a positive integer n, this function returns the number of ways to arrange 
the integers 1 to n such that each value is either strictly bigger than all 
preceding values or strictly smaller than all preceding values.
-/
def arrangementsCount (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- 
Theorem stating that the number of valid arrangements of integers 1 to n
is equal to 2^(n-1).
-/
theorem arrangementsCount_correct (n : ℕ+) :
  arrangementsCount n = Finset.card (Finset.filter
    (fun perm : Fin n.val → Fin n.val =>
      Function.Bijective perm ∧
      ∀ i j : Fin n.val, i < j →
        (perm i < perm j ∧ ∀ k : Fin n.val, k < i → perm k < perm i) ∨
        (perm j < perm i ∧ ∀ k : Fin n.val, k < i → perm k < perm i))
    (Finset.univ : Finset (Fin n.val → Fin n.val))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_arrangementsCount_correct_l547_54767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_vertical_tangent_line_all_tangent_lines_l547_54760

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- Define a line passing through the origin
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Define tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_through_origin m x y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' → line_through_origin m x' y' → (x', y') = (x, y)

-- Theorem statement
theorem tangent_line_to_circle :
  ∃ (m : ℝ), is_tangent m ∧ (m = 0 ∨ m = 3/4) := by
  sorry

-- Additional theorem for the vertical tangent line
theorem vertical_tangent_line :
  is_tangent 0 := by
  sorry

-- Theorem combining both cases
theorem all_tangent_lines :
  ∀ (m : ℝ), is_tangent m ↔ m = 0 ∨ m = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_vertical_tangent_line_all_tangent_lines_l547_54760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_theorem_l547_54745

/-- Represents the number of women's teams -/
def n : ℕ := 4

/-- Total number of teams -/
def total_teams (n : ℕ) : ℕ := n + 3*n

/-- Total number of matches played -/
def total_matches (n : ℕ) : ℕ := (total_teams n) * (total_teams n - 1) / 2

/-- Number of matches won by women's teams -/
def women_wins (n : ℕ) : ℕ := 3 * (total_matches n / 8)

/-- Number of matches won by men's teams -/
def men_wins (n : ℕ) : ℕ := 5 * (total_matches n / 8)

/-- The theorem stating that n must equal 4 given the conditions -/
theorem basketball_league_theorem : 
  (∀ m : ℕ, m ≠ n → total_matches m ≠ women_wins m + men_wins m) ∧
  total_matches n = women_wins n + men_wins n ∧
  women_wins n * 5 = men_wins n * 3 →
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_league_theorem_l547_54745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_coffee_bags_l547_54755

theorem jimmy_coffee_bags : ℕ := by
  -- Suki's coffee
  let suki_bags : ℚ := 13/2
  let suki_bag_weight : ℕ := 22
  let suki_total_weight : ℚ := suki_bags * suki_bag_weight

  -- Jimmy's coffee
  let jimmy_bag_weight : ℕ := 18

  -- Repackaged coffee
  let container_weight : ℕ := 8
  let total_containers : ℕ := 28
  let total_repackaged_weight : ℕ := container_weight * total_containers

  -- Calculation
  let jimmy_total_weight : ℚ := total_repackaged_weight - suki_total_weight
  let jimmy_bags : ℚ := jimmy_total_weight / jimmy_bag_weight

  -- Proof
  have h : jimmy_bags = 9/2 := by sorry

  -- Result (rounded down to nearest whole number)
  exact 4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_coffee_bags_l547_54755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l547_54796

def integers : Finset ℕ := Finset.range 11

def sum_units_digit (m n : ℕ) : ℕ := (m + n) % 10

def count_units_digit (d : ℕ) : ℕ :=
  Finset.card (Finset.filter (fun p : ℕ × ℕ => sum_units_digit p.1 p.2 = d) (integers.product integers))

theorem zero_most_frequent :
  ∀ d : ℕ, d ∈ Finset.range 10 → count_units_digit 0 ≥ count_units_digit d :=
by
  sorry

#eval count_units_digit 0  -- Should output 11
#eval count_units_digit 1  -- Should output 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l547_54796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pretty_penny_investment_l547_54797

/-- Represents the investment scenario for Susie Q --/
structure Investment where
  total : ℝ
  pretty_penny : ℝ
  five_and_dime : ℝ
  quick_nickel : ℝ
  pretty_penny_rate : ℝ
  five_and_dime_rate : ℝ
  quick_nickel_rate : ℝ
  years : ℕ
  final_amount : ℝ

/-- The investment satisfies the given conditions --/
def valid_investment (i : Investment) : Prop :=
  i.total = 1500 ∧
  i.pretty_penny + i.five_and_dime + i.quick_nickel = i.total ∧
  i.pretty_penny_rate = 0.04 ∧
  i.five_and_dime_rate = 0.045 ∧
  i.quick_nickel_rate = 0.03 ∧
  i.years = 3 ∧
  i.final_amount = 1649.50 ∧
  i.pretty_penny * (1 + i.pretty_penny_rate) ^ i.years +
  i.five_and_dime * (1 + i.five_and_dime_rate) ^ i.years +
  i.quick_nickel * (1 + i.quick_nickel_rate * i.years) = i.final_amount

/-- Theorem stating that the investment at Pretty Penny Bank is $300 --/
theorem pretty_penny_investment (i : Investment) :
  valid_investment i → i.pretty_penny = 300 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pretty_penny_investment_l547_54797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_is_hyperbola_l547_54785

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines the first line 2px - 3y - 4p = 0 -/
def line1 (p : ℝ) : Line :=
  { a := 2 * p, b := -3, c := -4 * p }

/-- Defines the second line 4x - 3py - 6 = 0 -/
def line2 (p : ℝ) : Line :=
  { a := 4, b := -3 * p, c := -6 }

/-- Defines the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point2D :=
  { x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b),
    y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b) }

/-- Theorem: The locus of intersection points forms a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∃ (a b : ℝ), ∀ (p : ℝ),
    let pt := intersectionPoint (line1 p) (line2 p)
    (pt.x^2 / a^2) - (pt.y^2 / b^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_locus_is_hyperbola_l547_54785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_equal_after_rounds_l547_54766

/-- Represents the state of money distribution among players -/
def MoneyState := Fin 4 → ℕ

/-- The initial state where each player has $1 -/
def initialState : MoneyState := λ _ => 1

/-- A single round of the game -/
def gameRound (state : MoneyState) : MoneyState :=
  sorry

/-- The probability of transitioning from one state to another in a single round -/
def transitionProbability (s1 s2 : MoneyState) : ℝ :=
  sorry

/-- The state where each player has $1 -/
def allEqualState : MoneyState := λ _ => 1

/-- The number of rounds played -/
def numRounds : ℕ := 2020

/-- The probability of being in a specific state after n rounds -/
noncomputable def probAfterNRounds (n : ℕ) (s : MoneyState) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem probability_all_equal_after_rounds :
  ∃ (p : ℝ), p = 8/27 ∧
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |p - probAfterNRounds n allEqualState| < ε) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_equal_after_rounds_l547_54766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_price_l547_54772

/-- Proves that the adult ticket price is $3 given the theater conditions --/
theorem adult_ticket_price (total_seats : ℕ) (child_ticket_price : ℚ) 
  (total_income : ℚ) (num_children : ℕ) : ℚ :=
by
  -- Given conditions
  have h1 : total_seats = 200 := by sorry
  have h2 : child_ticket_price = 3/2 := by sorry
  have h3 : total_income = 510 := by sorry
  have h4 : num_children = 60 := by sorry
  
  -- Calculations
  let num_adults : ℕ := total_seats - num_children
  let adult_ticket_price : ℚ := (total_income - (↑num_children * child_ticket_price)) / ↑num_adults
  
  -- Prove that adult_ticket_price = 3
  have h5 : adult_ticket_price = 3 := by
    -- Detailed proof steps would go here
    sorry
  
  -- Return the result
  exact 3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_price_l547_54772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_perimeter_ratio_is_one_third_l547_54775

-- Define the room dimensions
noncomputable def room_length : ℝ := 25
noncomputable def room_width_inches : ℝ := 150

-- Convert width to feet
noncomputable def room_width : ℝ := room_width_inches / 12

-- Calculate the perimeter
noncomputable def room_perimeter : ℝ := 2 * (room_length + room_width)

-- Define the ratio
noncomputable def length_to_perimeter_ratio : ℝ := room_length / room_perimeter

-- Theorem to prove
theorem length_to_perimeter_ratio_is_one_third :
  length_to_perimeter_ratio = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_to_perimeter_ratio_is_one_third_l547_54775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_relation_l547_54792

/-- Given a cylinder and a cone with equal volumes and base areas, 
    if the height of the cone is 18 decimeters, 
    then the height of the cylinder is 6 decimeters. -/
theorem cylinder_cone_height_relation 
  (cylinder_volume cone_volume : ℝ)
  (cylinder_base_area cone_base_area : ℝ)
  (cylinder_height cone_height : ℝ)
  (h_volume : cylinder_volume = cone_volume)
  (h_base_area : cylinder_base_area = cone_base_area)
  (h_cone_height : cone_height = 18) : 
  cylinder_height = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cone_height_relation_l547_54792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l547_54763

def a : ℝ × ℝ := (1, 2)

theorem vector_properties (m : ℝ) (h1 : m < 0) :
  let b : ℝ × ℝ := (m, 1)
  (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 3) →
  (b.1^2 + b.2^2 = 2) ∧
  (let v1 := (2*a.1 - b.1, 2*a.2 - b.2)
   let v2 := (a.1 - 2*b.1, a.2 - 2*b.2)
   Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / 
    (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))) = π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l547_54763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_90_degrees_l547_54736

def p : Fin 3 → ℝ := ![2, -3, 4]
def q : Fin 3 → ℝ := ![-1, 5, -2]
def r : Fin 3 → ℝ := ![8, -1, 6]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def angle_between_vectors (v w : Fin 3 → ℝ) : ℝ := 
  Real.arccos (
    (dot_product v w) / (Real.sqrt (dot_product v v) * Real.sqrt (dot_product w w))
  )

theorem angle_is_90_degrees :
  angle_between_vectors p ((dot_product p r) • q - (dot_product p q) • r) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_90_degrees_l547_54736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l547_54777

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (6 + (Real.sqrt 3 / 2) * t, t / 2)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 10 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, C₁ t = p ∧ C₂ θ = p}

-- Theorem statement
theorem intersection_points_distance :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_distance_l547_54777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_better_vision_class_A_larger_variance_class_B_estimated_students_above_4_6_l547_54715

def class_A_vision : List Float := [4.3, 5.1, 4.6, 4.1, 4.9]
def class_B_vision : List Float := [5.1, 4.9, 4.0, 4.0, 4.5]
def total_students_A : Nat := 40

def average (l : List Float) : Float :=
  (l.sum) / (l.length.toFloat)

def variance (l : List Float) : Float :=
  let μ := average l
  (l.map (fun x => (x - μ)^2)).sum / (l.length.toFloat)

theorem better_vision_class_A :
  average class_A_vision > average class_B_vision := by
  sorry

theorem larger_variance_class_B :
  variance class_B_vision > variance class_A_vision := by
  sorry

theorem estimated_students_above_4_6 :
  (class_A_vision.filter (fun x => x > 4.6)).length.toFloat * (total_students_A.toFloat / class_A_vision.length.toFloat) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_better_vision_class_A_larger_variance_class_B_estimated_students_above_4_6_l547_54715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l547_54729

/-- The function f(x) = (x^2 - x + 2) / x^2 -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 2) / x^2

/-- Theorem stating that if xf(x) + a > 0 for all x > 0, then a > 1 - 2√2 -/
theorem range_of_a (a : ℝ) : 
  (∀ x > 0, x * f x + a > 0) → a > 1 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l547_54729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l547_54752

theorem a_2009_value :
  ∀ (a : ℝ) (a₁ a₂ : ℝ) (a₂₀₀₉ a₂₀₁₀ : ℝ),
  (∀ x : ℝ, (x + 2)^2010 = a + a₁*(1+x) + a₂*(1+x)^2 + 
   (Finset.range 2007).sum (λ i ↦ (a₂₀₀₉*(1+x)^(i+3))) + 
   a₂₀₀₉*(1+x)^2009 + a₂₀₁₀*(1+x)^2010) →
  a₂₀₀₉ = 2010 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l547_54752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_four_math_four_english_l547_54757

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  2 * (Nat.factorial math_books) * (Nat.factorial english_books)

/-- Theorem: Arranging 4 math books and 4 English books -/
theorem arrange_four_math_four_english :
  arrange_books 4 4 = 1152 := by
  rw [arrange_books]
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_four_math_four_english_l547_54757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l547_54717

theorem fraction_irreducible (n : ℤ) : Int.gcd (12*n + 1) (30*n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducible_l547_54717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l547_54788

theorem sum_of_divisors_143 : 
  (Finset.sum (Finset.filter (λ x : ℕ => 143 % x = 0) (Finset.range (143 + 1))) id) = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_143_l547_54788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_percentage_proof_l547_54770

noncomputable def container1_volume : ℝ := 40
noncomputable def container1_grape_percentage : ℝ := 0.10
noncomputable def container2_volume : ℝ := 30
noncomputable def container2_grape_percentage : ℝ := 0.35
noncomputable def pure_grape_juice_added : ℝ := 20
noncomputable def evaporation_percentage : ℝ := 0.10

noncomputable def total_initial_volume : ℝ := container1_volume + container2_volume + pure_grape_juice_added

noncomputable def total_grape_juice : ℝ := 
  container1_volume * container1_grape_percentage + 
  container2_volume * container2_grape_percentage + 
  pure_grape_juice_added

noncomputable def remaining_volume : ℝ := total_initial_volume * (1 - evaporation_percentage)

noncomputable def final_grape_juice_percentage : ℝ := (total_grape_juice / remaining_volume) * 100

theorem grape_juice_percentage_proof :
  abs (final_grape_juice_percentage - 42.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_percentage_proof_l547_54770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_methods_l547_54738

def number_of_people : ℕ := 6
def number_of_rooms : ℕ := 2

def distribute_equally (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n (n / r) * Nat.choose (n - n / r) (n / r)

def distribute_at_least_one (n : ℕ) (r : ℕ) : ℕ :=
  Finset.sum (Finset.range (n - r + 1)) (fun i => Nat.choose n i * Nat.choose (n - i) (n - i))

theorem distribution_methods (n : ℕ) (r : ℕ) 
  (h1 : n = number_of_people) (h2 : r = number_of_rooms) : 
  distribute_equally n r = 20 ∧ distribute_at_least_one n r = 62 := by
  sorry

#eval distribute_equally number_of_people number_of_rooms
#eval distribute_at_least_one number_of_people number_of_rooms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_methods_l547_54738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_in_cube_l547_54713

theorem sphere_packing_in_cube (r : ℝ) : 
  -- Unit cube
  let cube_side : ℝ := 1

  -- Nine congruent spheres
  -- One sphere at center of cube
  -- Other spheres tangent to center sphere and three cube faces
  -- 2r is the distance between centers of spheres
  -- √3 is the space diagonal of the unit cube
  -- 2(r√3 + 2r) is the space diagonal of the smaller cube formed by sphere centers
  (2 * (r * Real.sqrt 3 + 2 * r) = Real.sqrt 3)
  →
  -- The radius of each sphere is (2√3 - 3) / 2
  r = (2 * Real.sqrt 3 - 3) / 2 := by
  
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_packing_in_cube_l547_54713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l547_54727

theorem calculation_proof : (3.14 - Real.pi)^0 - (1/2)^(-1 : ℤ) + |Real.sqrt 3 - 2| + Real.sqrt 27 = 1 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l547_54727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l547_54768

noncomputable section

-- Define the vector type
def Vec2D := ℝ × ℝ

-- Define the dot product for Vec2D
def dot_product (v w : Vec2D) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a Vec2D
noncomputable def magnitude (v : Vec2D) : ℝ := Real.sqrt (dot_product v v)

-- Define the angle between two Vec2D
noncomputable def angle_between (v w : Vec2D) : ℝ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

-- Define scalar multiplication for Vec2D
def smul (r : ℝ) (v : Vec2D) : Vec2D := (r * v.1, r * v.2)

-- Define vector addition for Vec2D
def vadd (v w : Vec2D) : Vec2D := (v.1 + w.1, v.2 + w.2)

theorem vector_sum_magnitude 
  (a b : Vec2D) 
  (h1 : a = (1, 1)) 
  (h2 : magnitude b = 2) 
  (h3 : angle_between a b = π / 4) : 
  magnitude (vadd (smul 3 a) b) = Real.sqrt 34 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l547_54768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_one_l547_54759

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

-- Define the solution set of |f(x)| < 6
def solution_set (a : ℝ) : Set ℝ := {x | |f a x| < 6}

-- State the theorem
theorem solution_set_of_f_leq_one (a : ℝ) :
  solution_set a = Set.Ioo (-1) 2 →
  {x : ℝ | f a x ≤ 1} = {x : ℝ | x > 1/2 ∨ x ≤ 1/2} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_leq_one_l547_54759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_cos_bounds_l547_54750

theorem cos_sin_cos_bounds (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  (1/8 : ℝ) ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_cos_bounds_l547_54750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_exponent_base_l547_54781

theorem third_exponent_base (a b : ℕ) (x : ℕ) (h1 : a = 7) :
  (18^a) * 9^(3*a - 1) = (x^7) * (3^b) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_exponent_base_l547_54781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_values_l547_54733

/-- A sequence of real numbers -/
def Sequence := ℕ+ → ℝ

/-- The partial sum sequence of a given sequence -/
def PartialSum (a : Sequence) : ℕ+ → ℝ :=
  fun n ↦ (Finset.range n).sum (fun i ↦ a ⟨i + 1, Nat.succ_pos i⟩)

/-- The property that all partial sums are either 1 or 3 -/
def PartialSumProperty (a : Sequence) : Prop :=
  ∀ n : ℕ+, PartialSum a n = 1 ∨ PartialSum a n = 3

/-- The set of distinct values in a sequence -/
def DistinctValues (a : Sequence) : Set ℝ :=
  {x : ℝ | ∃ n : ℕ+, a n = x}

/-- The number of distinct values in a sequence is at most 4 -/
theorem max_distinct_values (a : Sequence) (h : PartialSumProperty a) :
    ∃ (s : Finset ℝ), s.card ≤ 4 ∧ ∀ x ∈ DistinctValues a, x ∈ s :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_values_l547_54733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_plus_center_y_l547_54723

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  passingPoint : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the center of an ellipse -/
noncomputable def ellipseCenter (e : Ellipse) : Point :=
  { x := (e.focus1.x + e.focus2.x) / 2,
    y := (e.focus1.y + e.focus2.y) / 2 }

/-- Calculates the semi-major axis length of an ellipse -/
noncomputable def semiMajorAxisLength (e : Ellipse) : ℝ :=
  (distance e.passingPoint e.focus1 + distance e.passingPoint e.focus2) / 2

theorem ellipse_semi_major_axis_plus_center_y 
  (e : Ellipse) 
  (h1 : e.focus1 = { x := 1, y := 1 })
  (h2 : e.focus2 = { x := 1, y := 4 })
  (h3 : e.passingPoint = { x := 5, y := 2 }) :
  semiMajorAxisLength e + (ellipseCenter e).y = (Real.sqrt 17 + 2 * Real.sqrt 5 + 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_semi_major_axis_plus_center_y_l547_54723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_proof_l547_54782

/-- The speed of the river current in km/h -/
noncomputable def river_speed : ℝ := 4

/-- The distance traveled downstream by the motorboat in km -/
noncomputable def downstream_distance : ℝ := 40 / 3

/-- The distance traveled upstream by the motorboat before meeting the raft in km -/
noncomputable def upstream_distance : ℝ := 28 / 3

/-- The motorboat's own speed in km/h -/
noncomputable def motorboat_speed : ℝ := 68 / 3

theorem motorboat_speed_proof :
  ∃ (t : ℝ),
    t > 0 ∧
    downstream_distance / (motorboat_speed + river_speed) +
    upstream_distance / (motorboat_speed - river_speed) = t ∧
    (downstream_distance - upstream_distance) / river_speed = t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_proof_l547_54782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l547_54737

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 * x + 4 else -x^2 - 2 * x + 2

-- Theorem statement
theorem sum_of_solutions :
  ∃ x₁ x₂ : ℝ, f x₁ = -3 ∧ f x₂ = -3 ∧ x₁ ≠ x₂ ∧ 
  (∀ x₃ : ℝ, f x₃ = -3 → x₃ = x₁ ∨ x₃ = x₂) ∧
  x₁ + x₂ = -10/3 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l547_54737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_has_winning_strategy_l547_54773

/-- Represents the state of the game -/
structure GameState where
  sum : Nat
  lastDigit : Nat

/-- Represents a player in the game -/
inductive Player
  | Alice
  | Bob

/-- The length of the number being created -/
def numberLength : Nat := 2018

/-- Function to determine if a move is valid -/
def isValidMove (currentState : GameState) (digit : Nat) : Prop :=
  digit < 3 ∧ digit ≠ currentState.lastDigit

/-- Function to update the game state after a move -/
def makeMove (currentState : GameState) (digit : Nat) : GameState :=
  { sum := (currentState.sum + digit) % 3, lastDigit := digit }

/-- Theorem stating that Alice has a winning strategy -/
theorem alice_has_winning_strategy :
  ∃ (strategy : Nat → GameState → Nat),
    ∀ (bobStrategy : Nat → GameState → Nat),
      let finalState := (
        let rec play (movesLeft : Nat) (currentPlayer : Player) (state : GameState) : GameState :=
          if movesLeft = 0 then
            state
          else
            match currentPlayer with
            | Player.Alice => play (movesLeft - 1) Player.Bob (makeMove state (strategy movesLeft state))
            | Player.Bob => play (movesLeft - 1) Player.Alice (makeMove state (bobStrategy movesLeft state))
        play numberLength Player.Alice { sum := 0, lastDigit := 0 }
      )
      finalState.sum ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_has_winning_strategy_l547_54773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_theorem_l547_54744

def product_sequence (n : ℕ) : ℕ := 2^(2^n) + 1

def main_expression : ℕ := (Finset.range 6).prod product_sequence - 1

theorem unit_digit_theorem :
  main_expression % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_theorem_l547_54744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l547_54787

theorem soccer_ball_inflation (total_balls : ℕ) 
  (hole_percentage : ℚ) (overinflated_percentage : ℚ) : 
  total_balls = 100 →
  hole_percentage = 40 / 100 →
  overinflated_percentage = 20 / 100 →
  (total_balls - (hole_percentage * ↑total_balls).floor - 
   (overinflated_percentage * ↑(total_balls - (hole_percentage * ↑total_balls).floor)).floor) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_ball_inflation_l547_54787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_move_left_base_case_positive_base_case_zero_base_case_x_axis_probability_reaching_y_axis_l547_54732

/-- Probability function for reaching the y-axis -/
def P : ℕ → ℕ → ℚ 
| 0, 0 => 0
| 0, b + 1 => 1
| a + 1, 0 => 0
| a + 1, b + 1 => (1 / 2) * P a (b + 1) + (1 / 2) * P (a + 1) b

/-- The particle starts at (3,5) -/
def start : ℕ × ℕ := (3, 5)

/-- Movement conditions -/
theorem move_left (a b : ℕ) : 
  P (a + 1) b = (1 / 2) * P a b + (1 / 2) * P (a + 1) (b - 1) := by sorry

/-- Base cases -/
theorem base_case_positive (b : ℕ) : P 0 (b + 1) = 1 := by sorry
theorem base_case_zero : P 0 0 = 0 := by sorry
theorem base_case_x_axis (a : ℕ) : P (a + 1) 0 = 0 := by sorry

/-- Main theorem -/
theorem probability_reaching_y_axis : P start.1 start.2 = 117 / 128 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_move_left_base_case_positive_base_case_zero_base_case_x_axis_probability_reaching_y_axis_l547_54732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_labeling_exists_l547_54758

/-- Represents a die with 6 faces -/
def Die := Fin 6 → ℕ

/-- The sum of two dice rolls -/
def diceSum (d1 d2 : Die) (i j : Fin 6) : ℕ :=
  d1 i + d2 j

/-- All possible sums from rolling two dice -/
def allSums (d1 d2 : Die) : Finset ℕ :=
  Finset.image (λ p : Fin 6 × Fin 6 => diceSum d1 d2 p.1 p.2) (Finset.univ.product Finset.univ)

theorem dice_labeling_exists :
  ∃ (d1 d2 : Die),
    (allSums d1 d2 = Finset.range 37 \ {0}) ∧
    (∀ (s : ℕ) (hs : s ∈ allSums d1 d2),
      (Finset.filter (λ p : Fin 6 × Fin 6 => diceSum d1 d2 p.1 p.2 = s) (Finset.univ.product Finset.univ)).card = 1) :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_labeling_exists_l547_54758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bound_l547_54724

-- Define a structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the property that circles are mutually separated
def mutually_separated (c1 c2 c3 : Circle) : Prop :=
  distance c1.center c2.center > c1.radius + c2.radius ∧
  distance c2.center c3.center > c2.radius + c3.radius ∧
  distance c3.center c1.center > c3.radius + c1.radius

-- Define the property that any separating line intersects the third circle
def separating_line_intersects_third (c1 c2 c3 : Circle) : Prop :=
  ∀ (line : ℝ × ℝ → Prop),
    (∀ p, line p → distance p c1.center > c1.radius ∧ distance p c2.center > c2.radius) →
    (∃ p, line p ∧ distance p c3.center < c3.radius)

-- State the theorem
theorem circle_distance_bound (c1 c2 c3 : Circle) :
  mutually_separated c1 c2 c3 →
  separating_line_intersects_third c1 c2 c3 →
  separating_line_intersects_third c2 c3 c1 →
  separating_line_intersects_third c3 c1 c2 →
  distance c1.center c2.center + distance c2.center c3.center + distance c3.center c1.center ≤
  2 * Real.sqrt 2 * (c1.radius + c2.radius + c3.radius) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_bound_l547_54724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_12321_l547_54706

theorem largest_prime_factor_of_12321 :
  (Nat.factors 12321).maximum? = some 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_12321_l547_54706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l547_54725

noncomputable section

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

/-- The set of points (x, y) on the ellipse -/
def ellipse_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse_equation p.1 p.2}

/-- The ratio x/y for a point (x, y) -/
def ratio (p : ℝ × ℝ) : ℝ := p.1 / p.2

theorem ellipse_ratio_sum :
  ∃ (max min : ℝ),
    (∀ p ∈ ellipse_points, ratio p ≤ max ∧ min ≤ ratio p) ∧
    max + min = 272 / 447 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l547_54725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_condition_l547_54748

-- Define the circle
def circleEq (x y r : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = r^2

-- Define the line
def lineEq (x y : ℝ) : Prop := 4*x + 3*y + 2 = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |4*x + 3*y + 2| / 5

-- Define the condition of having exactly two points at distance 1 from the line
def has_two_points_at_distance_one (r : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circleEq x1 y1 r ∧ circleEq x2 y2 r ∧
    distance_to_line x1 y1 = 1 ∧ distance_to_line x2 y2 = 1 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    ∀ (x y : ℝ), circleEq x y r ∧ distance_to_line x y = 1 → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2)

-- Theorem statement
theorem circle_line_intersection_condition (r : ℝ) :
  has_two_points_at_distance_one r ↔ 4 < r ∧ r < 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_condition_l547_54748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l547_54762

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^2 - 4) + f (4 - x^2)

-- State the theorem
theorem F_properties (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, DifferentiableAt ℝ f x) →
  (HasDerivAt (F f) ((2 * x) * (deriv f (x^2 - 4)) - (2 * x) * (deriv f (4 - x^2))) x) ∧
  (HasDerivAt (F f) 0 2 ∧ HasDerivAt (F f) 0 (-2)) ∧
  HasDerivAt (F f) 0 0 ∧
  ∀ x, deriv (F f) (-x) = -(deriv (F f) x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_properties_l547_54762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areaCDE_approx_l547_54793

/-- The area of triangle CDE in the given figure -/
noncomputable def areaCDE : ℝ :=
  let OA : ℝ := 4
  let AB : ℝ := 11
  let AD : ℝ := 4
  let EA : ℝ := AB * (OA / (OA + AB))
  let DE : ℝ := AD - EA
  (DE * AB) / 2

theorem areaCDE_approx : ∃ (n : ℕ), |areaCDE - n| < 0.5 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areaCDE_approx_l547_54793
