import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1136_113605

-- Define the function f with domain [-2, 2]
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def f_domain : Set ℝ := Set.Icc (-2) 2

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

-- Theorem stating the domain of g
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range g} = Set.Ioc (-1/2 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1136_113605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cube_in_tetrahedron_l1136_113665

/-- The edge length of the tetrahedron -/
noncomputable def tetrahedron_edge : ℝ := 3 * Real.sqrt 6

/-- The maximum edge length of the inscribed cube -/
noncomputable def max_cube_edge : ℝ := Real.sqrt 3

/-- Theorem stating that the maximum edge length of a cube that can rotate freely
    inside a tetrahedron with edge length 3√6 is √3 -/
theorem max_cube_in_tetrahedron :
  let r := tetrahedron_edge * Real.sqrt 6 / 4
  (4 * (1/3) * r * (Real.sqrt 3 / 4) * tetrahedron_edge^2 =
   (1/3) * (Real.sqrt 3 / 4) * tetrahedron_edge^2 *
   Real.sqrt (tetrahedron_edge^2 - (2/3 * Real.sqrt 3 / 2 * tetrahedron_edge)^2)) →
  (r = 3/2) →
  (3 * max_cube_edge^2 = 9) →
  (max_cube_edge = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cube_in_tetrahedron_l1136_113665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1136_113679

-- Define the sign function
noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1
  else if a < 0 then -1
  else 0

-- Define the system of equations
def satisfies_system (x y z : ℝ) : Prop :=
  x = 2020 - 2021 * sign (y + z) ∧
  y = 2020 - 2021 * sign (x + z) ∧
  z = 2020 - 2021 * sign (x + y)

-- Theorem stating that there are exactly 3 solutions
theorem exactly_three_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔ satisfies_system x y z) ∧
    Finset.card solutions = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_solutions_l1136_113679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_length_is_30_l1136_113619

/-- Represents the length of a deck given its width, construction cost per square foot,
    sealant cost per square foot, and total cost. -/
noncomputable def deck_length (width : ℝ) (construction_cost : ℝ) (sealant_cost : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (width * (construction_cost + sealant_cost))

/-- Theorem stating that under the given conditions, the deck length is 30 feet. -/
theorem deck_length_is_30 :
  deck_length 40 3 1 4800 = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval deck_length 40 3 1 4800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deck_length_is_30_l1136_113619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_triangle_relation_l1136_113617

/-- A triangle that can be divided into two parts of equal area and equal perimeter by a line parallel to one side -/
structure EqualDivisionTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  equal_division : ∃ (lambda : ℝ), 0 < lambda ∧ lambda < 1 ∧
    lambda^2 = 1/2 ∧
    lambda * (a + b) = (1 - lambda) * (a + b) + c

theorem equal_division_triangle_relation (t : EqualDivisionTriangle) :
  t.c = (Real.sqrt 2 - 1) * (t.a + t.b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_division_triangle_relation_l1136_113617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_washes_for_dirt_removal_l1136_113620

/-- Theorem: The minimum number of washes required to ensure that the remaining dirt does not exceed 1% is 4. -/
theorem min_washes_for_dirt_removal : ∀ n : ℕ, (1 - 3/4 : ℚ)^n ≤ 1/100 ↔ n ≥ 4 := by
  -- Define the dirt removal rate
  let removal_rate : ℚ := 3/4

  -- Define the maximum allowable remaining dirt
  let max_remaining_dirt : ℚ := 1/100

  -- Define the function that calculates remaining dirt after n washes
  let remaining_dirt (n : ℕ) : ℚ := (1 - removal_rate) ^ n

  -- Define the predicate for sufficient washing
  let sufficient_washing (n : ℕ) : Prop := remaining_dirt n ≤ max_remaining_dirt

  -- Theorem proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_washes_for_dirt_removal_l1136_113620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l1136_113640

-- Define the function f(t) = t + 2/t
noncomputable def f (t : ℝ) : ℝ := t + 2/t

-- State the theorem
theorem min_value_on_interval :
  ∃ (m : ℝ), m = 3 ∧ ∀ t ∈ Set.Ioc 0 1, f t ≥ m := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_interval_l1136_113640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l1136_113608

-- Define the train parameters
noncomputable def train1_length : ℝ := 100
noncomputable def train2_length : ℝ := 280
noncomputable def train1_speed : ℝ := 42
noncomputable def train2_speed : ℝ := 30

-- Define the relative speed in km/h
noncomputable def relative_speed : ℝ := train1_speed + train2_speed

-- Convert relative speed to m/s
noncomputable def relative_speed_ms : ℝ := relative_speed * (1000 / 3600)

-- Define the total length of both trains
noncomputable def total_length : ℝ := train1_length + train2_length

-- Theorem: The time for the trains to clear each other is 19 seconds
theorem trains_clear_time : 
  (total_length / relative_speed_ms) = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l1136_113608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_runs_at_15kmh_l1136_113610

-- Define the constants
noncomputable def distance : ℝ := 30
noncomputable def ursula_speed : ℝ := 10
noncomputable def time_difference : ℝ := 1

-- Define Jeff's speed as a function of the given parameters
noncomputable def jeff_speed (d u_s t_diff : ℝ) : ℝ :=
  d / (d / u_s - t_diff)

-- Theorem statement
theorem jeff_runs_at_15kmh :
  jeff_speed distance ursula_speed time_difference = 15 := by
  -- Unfold the definition of jeff_speed
  unfold jeff_speed
  -- Substitute the values
  simp [distance, ursula_speed, time_difference]
  -- Perform numerical calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_runs_at_15kmh_l1136_113610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l1136_113657

theorem complex_number_modulus : 
  let z : ℂ := (2 * Complex.I) / (1 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_modulus_l1136_113657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l1136_113639

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then (2*x + 1) / (x^2)
  else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem b_range (a b : ℝ) (h : ∃ a, f a + g b = 0) : b ∈ Set.Icc (-1 : ℝ) 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l1136_113639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1136_113618

noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 3
  let n : ℕ := 8
  geometricSum a r n = 3280 := by
  -- Unfold the definition of geometricSum
  unfold geometricSum
  -- Simplify the expression
  simp [pow_succ]
  -- Perform the numerical computation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1136_113618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1136_113682

/-- Polar coordinate equation of line l -/
def line_l (θ : Real) : Prop := θ = Real.pi/4

/-- Parametric equation of curve C -/
def curve_C (x y θ : Real) : Prop := x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ

/-- Point M with coordinates (x₀, y₀) -/
structure Point_M where
  x₀ : Real
  y₀ : Real

/-- Line parallel to l passing through M -/
noncomputable def line_parallel_l (M : Point_M) (t : Real) : Real × Real :=
  (M.x₀ + Real.sqrt 2 * t / 2, M.y₀ + Real.sqrt 2 * t / 2)

/-- Intersection condition of parallel line and curve C -/
def intersection_condition (M : Point_M) (t : Real) : Prop :=
  let (x, y) := line_parallel_l M t
  curve_C x y (Real.arccos (x / Real.sqrt 2))

/-- Distance product condition -/
def distance_product (M : Point_M) : Prop :=
  ∃ t₁ t₂ : Real, 
    intersection_condition M t₁ ∧ 
    intersection_condition M t₂ ∧ 
    t₁ * t₂ = 16/9

/-- Theorem: Trajectory of point M -/
theorem trajectory_of_M (M : Point_M) : 
  distance_product M → M.x₀^2 + 2*M.y₀^2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l1136_113682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_l1136_113660

noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

def IsOnEllipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def IsOnLineFromOrigin (x y α : ℝ) : Prop := y = Real.tan α * x

def IsOnLineFromFocus (x y β : ℝ) : Prop := y = Real.tan β * (x + 1)

theorem ellipse_intersection_ratio 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (e : ℝ) 
  (he : e = Eccentricity a b) 
  (hb_mean : b^2 = 3 * e * a) 
  (α β : ℝ) 
  (hαβ : α + β = Real.pi) 
  (A B D E : ℝ × ℝ) 
  (hA : IsOnEllipse A.1 A.2 a b ∧ IsOnLineFromOrigin A.1 A.2 α) 
  (hB : IsOnEllipse B.1 B.2 a b ∧ IsOnLineFromOrigin B.1 B.2 α) 
  (hD : IsOnEllipse D.1 D.2 a b ∧ IsOnLineFromFocus D.1 D.2 β) 
  (hE : IsOnEllipse E.1 E.2 a b ∧ IsOnLineFromFocus E.1 E.2 β) 
  : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_ratio_l1136_113660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_two_pm_l1136_113670

/-- Represents a clock with minute and hour hands -/
structure Clock :=
  (minute_angle : ℝ)
  (hour_angle : ℝ)

/-- Calculates the smaller angle between two angles on a circle -/
noncomputable def smaller_angle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

/-- The theorem stating that the smaller angle between minute and hour hands at 2:00 p.m. is 60 degrees -/
theorem clock_angle_at_two_pm :
  let c : Clock := { minute_angle := 0, hour_angle := 2 * 360 / 12 }
  smaller_angle c.minute_angle c.hour_angle = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_two_pm_l1136_113670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1136_113668

noncomputable section

/-- The function f(x) = ln x + 2x + 3 -/
def f (x : ℝ) : ℝ := Real.log x + 2 * x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 1 / x + 2

/-- The line perpendicular to the tangent -/
def perp_line (x y : ℝ) : Prop := x + 3 * y + 1 = 0

/-- The slope of the perpendicular line -/
def perp_slope : ℝ := -1 / 3

/-- The point of tangency -/
def point_of_tangency : ℝ := 1

/-- The y-coordinate of the point of tangency -/
noncomputable def y_coord : ℝ := f point_of_tangency

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 3 * x - y + 2 = 0

theorem tangent_line_equation :
  ∃ (x₀ : ℝ), (f' x₀ = -1 / perp_slope) ∧
              (x₀ = point_of_tangency) ∧
              (∀ (x y : ℝ), tangent_line x y ↔ y - y_coord = f' x₀ * (x - x₀)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1136_113668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_baskets_theorem_l1136_113664

/-- Represents the weight difference of a basket from the standard weight -/
inductive WeightDifference
| Neg3 : WeightDifference
| Neg2 : WeightDifference
| Neg1_5 : WeightDifference
| Zero : WeightDifference
| Pos1 : WeightDifference
| Pos2_5 : WeightDifference

/-- Represents the count of baskets for each weight difference -/
def basketCounts : WeightDifference → Nat
| WeightDifference.Neg3 => 1
| WeightDifference.Neg2 => 4
| WeightDifference.Neg1_5 => 2
| WeightDifference.Zero => 3
| WeightDifference.Pos1 => 2
| WeightDifference.Pos2_5 => 8

/-- Converts WeightDifference to its numerical value in kg -/
def weightDifferenceValue : WeightDifference → Real
| WeightDifference.Neg3 => -3
| WeightDifference.Neg2 => -2
| WeightDifference.Neg1_5 => -1.5
| WeightDifference.Zero => 0
| WeightDifference.Pos1 => 1
| WeightDifference.Pos2_5 => 2.5

def standardWeight : Real := 25

def totalBaskets : Nat := 20

/-- Calculates the total weight difference for all baskets -/
def totalWeightDifference : Real :=
  (weightDifferenceValue WeightDifference.Neg3 * basketCounts WeightDifference.Neg3) +
  (weightDifferenceValue WeightDifference.Neg2 * basketCounts WeightDifference.Neg2) +
  (weightDifferenceValue WeightDifference.Neg1_5 * basketCounts WeightDifference.Neg1_5) +
  (weightDifferenceValue WeightDifference.Zero * basketCounts WeightDifference.Zero) +
  (weightDifferenceValue WeightDifference.Pos1 * basketCounts WeightDifference.Pos1) +
  (weightDifferenceValue WeightDifference.Pos2_5 * basketCounts WeightDifference.Pos2_5)

theorem apple_baskets_theorem :
  (weightDifferenceValue WeightDifference.Pos2_5 - weightDifferenceValue WeightDifference.Neg3 = 5.5) ∧
  (standardWeight * totalBaskets + totalWeightDifference = 508) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_baskets_theorem_l1136_113664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_allocation_l1136_113625

/-- Represents the profit function for product A -/
noncomputable def profit_A (t : ℝ) : ℝ := 3 * Real.sqrt t

/-- Represents the profit function for product B -/
def profit_B (t : ℝ) : ℝ := t

/-- Represents the total profit function -/
noncomputable def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (300 - x)

theorem max_profit_allocation :
  ∃ (x : ℝ), x ∈ Set.Icc 0 300 ∧
    (∀ y, y ∈ Set.Icc 0 300 → total_profit y ≤ total_profit x) ∧
    x = 225 ∧
    total_profit x = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_allocation_l1136_113625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opqrs_configurations_l1136_113658

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for four points forming a parallelogram -/
def parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y) ∧
  (C.x - A.x = D.x - B.x) ∧ (C.y - A.y = D.y - B.y)

/-- Predicate for three points being collinear -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

/-- Predicate for four points forming a trapezoid -/
def trapezoid (A B C D : Point) : Prop :=
  ((B.x - A.x) * (D.y - C.y) = (D.x - C.x) * (B.y - A.y)) ∨
  ((C.x - B.x) * (A.y - D.y) = (A.x - D.x) * (C.y - B.y))

/-- Given distinct points P, Q, R, S, and origin O, prove that OPQRS can form
    a parallelogram, a straight line, and a trapezoid -/
theorem opqrs_configurations
  (P Q : Point)
  (R : Point := ⟨P.x + Q.x, P.y + Q.y⟩)
  (S : Point := ⟨Q.x - P.x, Q.y - P.y⟩)
  (O : Point := ⟨0, 0⟩)
  (h_distinct : P ≠ Q) :
  (∃ (A B C D : Point), parallelogram A B C D) ∧
  (∃ (X Y Z : Point), collinear X Y Z) ∧
  (∃ (E F G H : Point), trapezoid E F G H) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opqrs_configurations_l1136_113658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_another_divisor_l1136_113687

def least_number : ℕ := 857

theorem another_divisor :
  ∃ (n : ℕ), n ∉ ({24, 36, 54} : Set ℕ) ∧ 
  (n ∣ (least_number + 7)) ∧
  (∀ m < least_number, ¬(24 ∣ (m + 7)) ∨ ¬(36 ∣ (m + 7)) ∨ ¬(54 ∣ (m + 7))) ∧
  n = 32 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_another_divisor_l1136_113687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wellcome_arrangements_eq_3600_l1136_113609

/-- The number of distinct arrangements of the letters in "Wellcome" where no two vowels are adjacent -/
def wellcome_arrangements : ℕ :=
  let total_letters : ℕ := 8
  let num_vowels : ℕ := 3
  let num_consonants : ℕ := 5
  let repeated_vowel : ℕ := 1
  let repeated_consonant : ℕ := 1
  let consonant_arrangements : ℕ := (num_consonants.factorial) / (repeated_consonant.factorial)
  let vowel_position_choices : ℕ := Nat.choose (num_consonants + 1) num_vowels
  let vowel_arrangements : ℕ := (num_vowels.factorial) / (repeated_vowel.factorial)
  consonant_arrangements * vowel_position_choices * vowel_arrangements

theorem wellcome_arrangements_eq_3600 : wellcome_arrangements = 3600 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wellcome_arrangements_eq_3600_l1136_113609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_materials_for_bracelets_l1136_113662

/-- 
Given:
- Alice made 52 friendship bracelets
- She gave away 8 bracelets
- She sold the remaining bracelets at $0.25 each
- She made a profit of $8

Prove that the cost of materials for Alice's bracelets was $3.
-/
theorem cost_of_materials_for_bracelets 
  (total_bracelets : ℕ) 
  (given_away : ℕ) 
  (price_per_bracelet : ℚ) 
  (profit : ℚ) 
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : price_per_bracelet = 1/4)
  (h4 : profit = 8) :
  (total_bracelets - given_away : ℚ) * price_per_bracelet - profit = 3 := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_materials_for_bracelets_l1136_113662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_four_l1136_113690

/-- The radius of a sphere whose surface area equals the curved surface area of a cylinder -/
noncomputable def sphere_radius (cylinder_height : ℝ) (cylinder_diameter : ℝ) : ℝ :=
  let cylinder_radius := cylinder_diameter / 2
  let cylinder_surface_area := 2 * Real.pi * cylinder_radius * cylinder_height
  Real.sqrt (cylinder_surface_area / (4 * Real.pi))

/-- Theorem: The radius of the sphere is 4 cm -/
theorem sphere_radius_is_four :
  sphere_radius 8 8 = 4 := by
  -- Unfold the definition of sphere_radius
  unfold sphere_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_is_four_l1136_113690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1136_113696

theorem negation_of_proposition :
  (¬∀ x : ℝ, x^2 > Real.log x) ↔ ∃ x₀ : ℝ, x₀^2 ≤ Real.log x₀ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1136_113696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1136_113633

theorem geometric_sequence_first_term (a r : ℝ) :
  a * r^5 = Nat.factorial 8 ∧ a * r^8 = Nat.factorial 9 → a = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l1136_113633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_problem_l1136_113642

/-- Calculates the length of the second train given the speeds of both trains,
    the length of the first train, and the time they take to clear each other. -/
noncomputable def second_train_length (speed1 speed2 : ℝ) (length1 : ℝ) (clear_time : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18)  -- Convert km/h to m/s
  relative_speed * clear_time - length1

/-- The theorem stating the length of the second train given the problem conditions -/
theorem second_train_length_problem :
  let speed1 : ℝ := 42
  let speed2 : ℝ := 36
  let length1 : ℝ := 120
  let clear_time : ℝ := 18.460061656605934
  abs (second_train_length speed1 speed2 length1 clear_time - 278.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_problem_l1136_113642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sin_cos_inequality_l1136_113600

theorem largest_n_sin_cos_inequality : 
  ∀ n : ℕ, n > 0 → n ≤ 4 ↔ 
    ∀ x : ℝ, (Real.sin x)^n + (Real.cos x)^n ≥ 2 / (n : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sin_cos_inequality_l1136_113600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_salary_proof_l1136_113689

noncomputable def average_salary (s1 s2 s3 s4 : ℝ) : ℝ := (s1 + s2 + s3 + s4) / 4

theorem june_salary_proof 
  (j f m a may_s : ℝ)
  (h1 : average_salary j f m a = 8000)
  (h2 : average_salary f m a may_s = 8450)
  (h3 : may_s = 6500)
  (h4 : ∃ (june_s : ℝ), average_salary m a may_s june_s = 9000 ∧ june_s = 1.2 * may_s) :
  ∃ (june_s : ℝ), june_s = 7800 := by
  sorry

#check june_salary_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_salary_proof_l1136_113689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quality_related_to_production_line_probability_of_selecting_from_line_A_l1136_113656

/-- Contingency table data --/
structure ContingencyTable where
  fairA : ℕ
  goodA : ℕ
  fairB : ℕ
  goodB : ℕ

/-- Chi-square statistic calculation --/
def chiSquare (ct : ContingencyTable) : ℚ :=
  let n : ℚ := (ct.fairA + ct.goodA + ct.fairB + ct.goodB : ℚ)
  let numerator := n * ((ct.fairA * ct.goodB - ct.goodA * ct.fairB : ℤ) ^ 2 : ℚ)
  let denominator := ((ct.fairA + ct.goodA) * (ct.fairB + ct.goodB) * (ct.fairA + ct.fairB) * (ct.goodA + ct.goodB) : ℚ)
  numerator / denominator

/-- Critical value for 90% confidence --/
def criticalValue : ℚ := 2706 / 1000

/-- Theorem statement for product quality relation to production line --/
theorem product_quality_related_to_production_line (ct : ContingencyTable) 
  (h : ct = ⟨40, 80, 80, 100⟩) : 
  chiSquare ct > criticalValue := by sorry

/-- Theorem statement for probability of selecting items from Line A --/
theorem probability_of_selecting_from_line_A : 
  (Nat.choose 6 2 - Nat.choose 4 2 : ℚ) / Nat.choose 6 2 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_quality_related_to_production_line_probability_of_selecting_from_line_A_l1136_113656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x_y_tightness_of_bound_l1136_113649

theorem min_sum_x_y (x y : ℝ) : 
  ((x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) → x + y ≥ 0 :=
by sorry

theorem tightness_of_bound : 
  ∃ x y : ℝ, ((x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) ∧ x + y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_x_y_tightness_of_bound_l1136_113649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_drop_probability_l1136_113628

-- Define the diameter of the circular coin
noncomputable def coin_diameter : ℝ := 3

-- Define the side length of the square hole
noncomputable def hole_side : ℝ := 1

-- Define the probability of the oil drop falling into the hole
noncomputable def probability : ℝ := 4 / (9 * Real.pi)

-- Theorem statement
theorem oil_drop_probability :
  probability = (hole_side^2) / (Real.pi * (coin_diameter / 2)^2) := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_drop_probability_l1136_113628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l1136_113667

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

-- Theorem statement
theorem absolute_difference_of_solution (x y : ℝ) : 
  (floor x : ℝ) + frac y = 3.7 ∧ 
  frac x + (floor y : ℝ) = 4.2 → 
  |x - y| = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l1136_113667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1136_113634

theorem percentage_difference : ∃ (difference : ℝ), difference = 47 := by
  let half_of_hundred : ℝ := 100 * 0.5
  let fifth_of_fifteen : ℝ := 15 * 0.2
  let difference : ℝ := half_of_hundred - fifth_of_fifteen
  have h : difference = 47 := by sorry
  exact ⟨difference, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l1136_113634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_unextended_twice_l1136_113637

/-- The oscillation function for a mass on a spring -/
noncomputable def x (a : ℝ) (t : ℝ) : ℝ := a * Real.exp (-2 * t) + (1 - 2 * a) * Real.exp (-t) + 1

/-- The theorem stating the conditions for the spring to be in an unextended state twice -/
theorem spring_unextended_twice (a : ℝ) : 
  (∃ t₁ t₂ : ℝ, 0 < t₁ ∧ 0 < t₂ ∧ t₁ ≠ t₂ ∧ x a t₁ = 0 ∧ x a t₂ = 0) ↔ 
  (1 + Real.sqrt 3 / 2 < a ∧ a < 2) := by
  sorry

#check spring_unextended_twice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_unextended_twice_l1136_113637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_value_l1136_113643

-- Define the endpoints of the line segment
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (5, 7)

-- Define the perpendicular bisector line
def perpendicular_bisector (b : ℝ) (x y : ℝ) : Prop := x + y = b

-- Theorem statement
theorem perpendicular_bisector_b_value :
  ∃ b : ℝ, perpendicular_bisector b ((point1.1 + point2.1) / 2) ((point1.2 + point2.2) / 2) :=
by
  -- We claim that b = 8 satisfies the condition
  use 8
  -- Expand the definition of perpendicular_bisector
  unfold perpendicular_bisector
  -- Simplify the equation
  simp [point1, point2]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_value_l1136_113643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_from_medians_l1136_113644

-- Define the necessary types and functions
def Point := ℝ × ℝ
def Triangle := Set Point

def is_median (triangle : Triangle) (m : Point → Point) : Prop := sorry
def length (p q : Point) : ℝ := sorry
def perimeter (triangle : Triangle) : ℝ := sorry

theorem triangle_perimeter_from_medians :
  ∀ (triangle : Triangle),
    (∃ (m₁ m₂ m₃ : Point → Point),
      (is_median triangle m₁ ∧ length (m₁ (0, 0)) (1, 0) = 3) ∧
      (is_median triangle m₂ ∧ length (m₂ (0, 0)) (1, 0) = 4) ∧
      (is_median triangle m₃ ∧ length (m₃ (0, 0)) (1, 0) = 6))
    → perimeter triangle = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_from_medians_l1136_113644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_30240_l1136_113673

def n : ℕ := 30240

theorem divisors_of_30240 : 
  (Finset.filter (λ i => n % i = 0) (Finset.range 10)).card = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_30240_l1136_113673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_pentagon_to_hexagon_l1136_113635

/-- A regular hexagon with side length s -/
structure RegularHexagon :=
  (s : ℝ)
  (s_pos : s > 0)

/-- A point on the side of a hexagon, located 1/4 of the side length from a vertex -/
structure PointOnSide (hex : RegularHexagon) :=
  (distance_from_vertex : ℝ)
  (is_quarter : distance_from_vertex = hex.s / 4)

/-- The area of a regular hexagon -/
noncomputable def area_hexagon (hex : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * hex.s^2

/-- The theorem to be proved -/
theorem area_ratio_pentagon_to_hexagon (hex : RegularHexagon) 
  (P Q R S : PointOnSide hex) :
  let pentagon_area := area_hexagon hex * ((48 * Real.sqrt 3 - 3) / (48 * Real.sqrt 3))
  ∃ (area_APQRS : ℝ), area_APQRS = pentagon_area := by
  sorry

#check area_ratio_pentagon_to_hexagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_pentagon_to_hexagon_l1136_113635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_destination_l1136_113616

/-- Calculates the distance to a destination given rowing speed, current velocity, and round trip time -/
noncomputable def calculate_distance (rowing_speed : ℝ) (current_velocity : ℝ) (total_time : ℝ) : ℝ :=
  let upstream_speed := rowing_speed - current_velocity
  let downstream_speed := rowing_speed + current_velocity
  (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))

/-- Theorem stating that under given conditions, the distance to the destination is 48 km -/
theorem distance_to_destination :
  let rowing_speed : ℝ := 10
  let current_velocity : ℝ := 2
  let total_time : ℝ := 10
  calculate_distance rowing_speed current_velocity total_time = 48 := by
  -- Unfold the definition of calculate_distance
  unfold calculate_distance
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_destination_l1136_113616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizing_point_theorem_l1136_113602

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The angle between three points in 3D space -/
noncomputable def angle (p q r : Point3D) : ℝ := sorry

/-- The theorem statement -/
theorem minimizing_point_theorem (A B C D X₀ : Point3D) 
  (h_not_coplanar : sorry)  -- A, B, C, D are not coplanar
  (h_not_collinear : sorry)  -- No three points among A, B, C, D are collinear
  (h_minimizes : ∀ X : Point3D, 
    distance A X + distance B X + distance C X + distance D X ≥ 
    distance A X₀ + distance B X₀ + distance C X₀ + distance D X₀)
  (h_different : X₀ ≠ A ∧ X₀ ≠ B ∧ X₀ ≠ C ∧ X₀ ≠ D) :
  angle A X₀ B = angle C X₀ D := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimizing_point_theorem_l1136_113602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_and_max_value_l1136_113601

noncomputable def f (x : ℝ) := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by
  sorry

theorem f_period_and_max_value :
  (∃ (T : ℝ), T = 6 * Real.pi ∧ T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_and_max_value_l1136_113601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookshelf_theorem_l1136_113684

theorem bookshelf_theorem 
  (A H S M E : ℝ) 
  (hd : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
        H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
        S ≠ M ∧ S ≠ E ∧ 
        M ≠ E) 
  (hp : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ E > 0)
  (h1 : ∃ (d e : ℝ), A * d + H * e = E * d)
  (h2 : ∃ (d e : ℝ), S * d + M * e = E * d)
  (h3 : M ≠ H) : 
  E = (A * M - S * H) / (M - H) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookshelf_theorem_l1136_113684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l1136_113648

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ
deriving Inhabited

/-- Represents an isosceles trapezoid PQRS -/
structure IsoscelesTrapezoid where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Checks if a line has integer slope -/
def hasIntegerSlope (p1 p2 : Point) : Prop :=
  ∃ k : ℤ, k * (p2.x - p1.x) = p2.y - p1.y

/-- Checks if two lines are parallel -/
def areParallel (p1 p2 p3 p4 : Point) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Main theorem statement -/
theorem isosceles_trapezoid_slope_sum (t : IsoscelesTrapezoid) : 
  t.P = Point.mk 30 200 →
  t.S = Point.mk 31 215 →
  (∀ side : List Point, side.length = 2 → (∀ p ∈ side, p ∈ [t.P, t.Q, t.R, t.S]) → 
    hasIntegerSlope side.head! side.tail!.head!) →
  (¬ areParallel t.P t.Q t.Q t.R) →
  (¬ areParallel t.P t.S t.S t.R) →
  areParallel t.P t.Q t.R t.S →
  (∀ side : List Point, side.length = 2 → (∀ p ∈ side, p ∈ [t.P, t.Q, t.R, t.S]) → 
    side ≠ [t.P, t.Q] → side ≠ [t.R, t.S] → ¬ areParallel side.head! side.tail!.head! t.P t.Q) →
  (∃ slopes : List ℚ, 
    (∀ slope ∈ slopes, ∃ x y : ℤ, t.Q = Point.mk x y ∧ slope = (y - 200 : ℚ) / (x - 30 : ℚ)) ∧
    (slopes.map abs).sum = 255 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l1136_113648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1136_113694

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles is π radians (180°)
  t.area = (t.a^2 + t.b^2 - t.c^2) / 2 ∧  -- Given area formula
  t.area = (1/2) * t.a * t.b * Real.sin t.C  -- Area formula with sine

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_properties t) :
  t.C = Real.pi / 3 ∧  -- C = 60° (π/3 radians)
  (∀ A' B' : ℝ, A' + B' = 2 * Real.pi / 3 → 
    Real.sin A' + Real.sin B' ≤ 1 + Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1136_113694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_honey_container_price_l1136_113655

/-- Represents a cylindrical container --/
structure Container where
  diameter : ℝ
  height : ℝ

/-- Calculates the volume of a cylindrical container --/
noncomputable def volume (c : Container) : ℝ :=
  Real.pi * (c.diameter / 2)^2 * c.height

/-- Calculates the price of a container given a base price and container --/
noncomputable def price (base_price : ℝ) (base_container : Container) (target_container : Container) : ℝ :=
  base_price * (volume target_container) / (volume base_container)

/-- Theorem: The price of a new container is proportional to its volume relative to the base container --/
theorem price_proportional_to_volume 
  (base_price : ℝ) (base_container target_container : Container) :
  price base_price base_container target_container = 
    base_price * (target_container.diameter^2 * target_container.height) / 
                 (base_container.diameter^2 * base_container.height) := by
  sorry

/-- Given containers and prices, proves the price of the new container --/
theorem honey_container_price 
  (base_price : ℝ) (base_container target_container : Container) :
  base_price = 1.5 ∧ 
  base_container.diameter = 5 ∧ 
  base_container.height = 6 ∧
  target_container.diameter = 10 ∧ 
  target_container.height = 8 →
  price base_price base_container target_container = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_proportional_to_volume_honey_container_price_l1136_113655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_equidistant_l1136_113653

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := { p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-6, 0)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (-3, 0)
def D : ℝ × ℝ := (3, 0)

def largeCircle := Circle O 6
def smallCircle := Circle O 1

-- Define the broken-line path length
noncomputable def pathLength (Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) + Real.sqrt ((Q.1 - D.1)^2 + (Q.2 - D.2)^2)

-- Theorem statement
theorem longest_path_equidistant (Q : ℝ × ℝ) (hQ : Q ∈ largeCircle) :
  pathLength Q ≤ pathLength (0, 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_path_equidistant_l1136_113653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_PQR_l1136_113651

/-- The radius of the inscribed circle in a triangle with sides a, b, and c --/
noncomputable def inscribedCircleRadius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- Theorem: The radius of the inscribed circle in triangle PQR is √17 --/
theorem inscribed_circle_radius_PQR :
  inscribedCircleRadius 26 10 18 = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_PQR_l1136_113651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_quadratic_sum_of_coefficients_l1136_113607

/-- A quadratic function f(x) = ax² + bx + 1 that is even and defined on [-2a, a² - 3] -/
def EvenQuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

/-- The domain of the function is [-2a, a² - 3] -/
def FunctionDomain (a : ℝ) : Set ℝ := Set.Icc (-2 * a) (a^2 - 3)

theorem even_quadratic_sum_of_coefficients 
  (a b : ℝ) 
  (h_even : ∀ x ∈ FunctionDomain a, EvenQuadraticFunction a b x = EvenQuadraticFunction a b (-x))
  (h_domain : ∀ x ∈ FunctionDomain a, x ≥ -2 * a ∧ x ≤ a^2 - 3) :
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_quadratic_sum_of_coefficients_l1136_113607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1136_113693

def Sequence := Fin 9 → ℚ

def ValidSequence (a : Sequence) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i.succ) / a i) ∈ ({2, 1, -1/2} : Set ℚ)

-- We need to prove that the set of valid sequences is finite
instance : Fintype { a : Sequence | ValidSequence a } := by sorry

theorem count_valid_sequences :
  Fintype.card { a : Sequence | ValidSequence a } = 491 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_sequences_l1136_113693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1136_113697

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 4

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ x₀ ∈ Set.Icc (-Real.pi/4) (Real.pi/4), ∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≤ f x₀ ∧ f x₀ = 1/4) ∧
  (∃ x₁ ∈ Set.Icc (-Real.pi/4) (Real.pi/4), ∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≥ f x₁ ∧ f x₁ = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1136_113697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l1136_113681

theorem ticket_price_possibilities : ∃ (n : ℕ), n = (Finset.filter (λ x => 36 % x = 0 ∧ 60 % x = 0) (Finset.range 61)).card ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ticket_price_possibilities_l1136_113681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l1136_113646

def alternating_polynomial (n : ℕ) (x : ℝ) : ℝ := 
  Finset.sum (Finset.range (n + 1)) (fun k => (-1)^k * x^(n - k))

theorem max_real_roots (n : ℕ) (h : n > 0) :
  (∃ x : ℝ, alternating_polynomial n x = 0) ∧
  (n % 2 = 1 → ∀ y z : ℝ, y ≠ z → (alternating_polynomial n y = 0 ∧ alternating_polynomial n z = 0) → False) ∧
  (n % 2 = 0 → ∃ y z : ℝ, y ≠ z ∧ alternating_polynomial n y = 0 ∧ alternating_polynomial n z = 0) ∧
  (∀ w x y z : ℝ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z →
    ¬(alternating_polynomial n w = 0 ∧ alternating_polynomial n x = 0 ∧ 
      alternating_polynomial n y = 0 ∧ alternating_polynomial n z = 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_l1136_113646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_decrease_l1136_113630

theorem cost_price_decrease (x : ℝ) : 
  let original_cost : ℝ → ℝ := λ m ↦ m
  let original_selling_price : ℝ → ℝ := λ m ↦ m * 1.15
  let new_cost : ℝ → ℝ := λ m ↦ m * (1 - x / 100)
  let new_selling_price : ℝ → ℝ := λ m ↦ m * (1 - x / 100) * 1.25
  (∀ m, original_selling_price m = new_selling_price m) →
  x = 8 := by
  intro h
  -- Proof steps would go here
  sorry

#check cost_price_decrease

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_decrease_l1136_113630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1136_113688

/-- The parabola equation -/
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- The directrix of the parabola -/
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

/-- The length of the line segment cut by the circle from the directrix is 4 -/
def segment_length (p : ℝ) : Prop := ∃ x y : ℝ, directrix p x ∧ circle_eq x y ∧ 4 = 2 * (x + 1)

theorem parabola_circle_intersection (p : ℝ) :
  p > 0 →
  (∀ x y : ℝ, parabola p x y → x ≥ 0) →
  segment_length p →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1136_113688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_spheres_l1136_113683

/-- A triangular pyramid with opposite edges of lengths a, b, and c. -/
structure TriangularPyramid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The radius of the circumscribed sphere of a triangular pyramid. -/
noncomputable def circumscribedRadius (p : TriangularPyramid) : ℝ :=
  (1/4) * Real.sqrt (2 * (p.a^2 + p.b^2 + p.c^2))

/-- The semi-perimeter of the face triangle of a triangular pyramid. -/
noncomputable def semiPerimeter (p : TriangularPyramid) : ℝ :=
  (p.a + p.b + p.c) / 2

/-- The radius of the inscribed sphere of a triangular pyramid. -/
noncomputable def inscribedRadius (p : TriangularPyramid) : ℝ :=
  let ρ := semiPerimeter p
  Real.sqrt ((1/8) * (p.a^2 + p.b^2 + p.c^2) - 
    (p.a^2 * p.b^2 * p.c^2) / (16 * ρ * (ρ - p.a) * (ρ - p.b) * (ρ - p.c)))

/-- Predicate to check if a point is a vertex of the pyramid -/
def IsVertexOf (point : ℝ × ℝ × ℝ) (p : TriangularPyramid) : Prop := sorry

/-- Predicate to check if a set of points forms a face of the pyramid -/
def IsFaceOf (face : Set (ℝ × ℝ × ℝ)) (p : TriangularPyramid) : Prop := sorry

/-- Predicate to check if a set of points forms a sphere -/
def IsSphere (s : Set (ℝ × ℝ × ℝ)) (center : ℝ × ℝ × ℝ) (radius : ℝ) : Prop := sorry

/-- Predicate to check if a sphere is tangent to a plane -/
def Tangent (s : Set (ℝ × ℝ × ℝ)) (face : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem triangular_pyramid_spheres (p : TriangularPyramid) :
  ∃ (center : ℝ × ℝ × ℝ),
    (∃ (sphere : Set (ℝ × ℝ × ℝ)), IsSphere sphere center (circumscribedRadius p) ∧
      (∀ vertex, IsVertexOf vertex p → vertex ∈ sphere)) ∧
    (∃ (sphere : Set (ℝ × ℝ × ℝ)), IsSphere sphere center (inscribedRadius p) ∧
      (∀ face, IsFaceOf face p → Tangent sphere face)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_spheres_l1136_113683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_perimeter_l1136_113624

/-- An ellipse with equation 2x^2 + 3y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + 3 * p.2^2 = 1}

/-- Predicate to check if a point is a focus of the ellipse -/
def IsFocus (e : Set (ℝ × ℝ)) (f : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), ∀ p ∈ e, dist p f + dist p ((-f.1, f.2)) = c

/-- Triangle ABC where B and C are on the ellipse, A is a focus, and F is the other focus on BC -/
structure SpecialTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ
  B_on_ellipse : B ∈ Ellipse
  C_on_ellipse : C ∈ Ellipse
  A_is_focus : IsFocus Ellipse A
  F_is_focus : IsFocus Ellipse F
  F_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t) • B + t • C

/-- The perimeter of a triangle given by its vertices -/
def trianglePerimeter (A B C : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C A

/-- The main theorem stating the perimeter of the special triangle -/
theorem special_triangle_perimeter (T : SpecialTriangle) :
    trianglePerimeter T.A T.B T.C = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_perimeter_l1136_113624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_correct_l1136_113661

/-- The distance from the center of the circle ρ = 2cos θ to the line ρ sin θ + 2ρ cos θ = 1 -/
noncomputable def distance_circle_to_line : ℝ := Real.sqrt 5 / 5

/-- The polar equation of the circle -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The polar equation of the line -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

theorem distance_is_correct : 
  ∀ ρ θ : ℝ, circle_equation ρ θ → line_equation ρ θ → 
  distance_circle_to_line = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_correct_l1136_113661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_15_points_l1136_113691

/-- A type representing a point in a space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a plane in a space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Function to create a plane from three points -/
def planeThroughPoints (p1 p2 p3 : Point) : Plane := sorry

/-- Function to count unique planes given a list of points -/
def countUniquePlanes (points : List Point) : ℕ := sorry

theorem max_planes_from_15_points :
  ∀ (points : List Point),
    points.length = 15 →
    (∀ p1 p2 p3 p4, p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points →
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 → ¬ areCoplanar p1 p2 p3 p4) →
    countUniquePlanes points = 455 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_15_points_l1136_113691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_plane_dihedral_angle_l1136_113695

/-- A regular tetrahedron with side edge 2 and base side length 1 -/
structure RegularTetrahedron where
  sideEdge : ℝ
  baseSide : ℝ
  sideEdge_eq : sideEdge = 2
  baseSide_eq : baseSide = 1

/-- A plane that bisects the volume of the tetrahedron -/
structure BisectingPlane (t : RegularTetrahedron) where
  passesThroughBaseEdge : Prop
  bisectsVolume : Prop

/-- The cosine of the dihedral angle between the bisecting plane and the base -/
noncomputable def dihedralAngleCosine (t : RegularTetrahedron) (p : BisectingPlane t) : ℝ := 
  2 / Real.sqrt 15

theorem bisecting_plane_dihedral_angle 
  (t : RegularTetrahedron) 
  (p : BisectingPlane t) : 
  dihedralAngleCosine t p = 2 / Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_plane_dihedral_angle_l1136_113695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_attack_after_knight_move_l1136_113606

/-- Represents a position on the chess board -/
structure Position where
  x : Nat
  y : Nat
deriving Repr, Inhabited

/-- Represents a knight's move -/
inductive KnightMove where
  | move1 : KnightMove  -- Represents a move of 2 horizontally and 1 vertically
  | move2 : KnightMove  -- Represents a move of 1 horizontally and 2 vertically
deriving Repr, Inhabited

/-- Apply a knight's move to a position -/
def applyKnightMove (p : Position) (m : KnightMove) : Position :=
  match m with
  | KnightMove.move1 => ⟨p.x + 2, p.y + 1⟩
  | KnightMove.move2 => ⟨p.x + 1, p.y + 2⟩

/-- Check if two positions are in the same row or column -/
def attackEachOther (p1 p2 : Position) : Prop :=
  p1.x = p2.x ∨ p1.y = p2.y

theorem rooks_attack_after_knight_move 
  (initial_positions : Fin 15 → Position)
  (h1 : ∀ i j, i ≠ j → ¬(attackEachOther (initial_positions i) (initial_positions j)))
  (h2 : ∀ i, (initial_positions i).x ≤ 15 ∧ (initial_positions i).y ≤ 15)
  (moves : Fin 15 → KnightMove) :
  ∃ i j, i ≠ j ∧ attackEachOther (applyKnightMove (initial_positions i) (moves i)) 
                                 (applyKnightMove (initial_positions j) (moves j)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rooks_attack_after_knight_move_l1136_113606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nestedSqrt_eq_nestedSqrt_value_l1136_113626

/-- The value of the infinite nested square root expression √(3 - √(3 - √(3 - √(3 - ...)))) -/
noncomputable def nestedSqrt : ℝ := (Real.sqrt 13 - 1) / 2

/-- Theorem stating that the nested square root satisfies the equation x = √(3 - x) -/
theorem nestedSqrt_eq : nestedSqrt = Real.sqrt (3 - nestedSqrt) := by
  sorry

/-- Theorem stating that the nested square root equals (√13 - 1) / 2 -/
theorem nestedSqrt_value : nestedSqrt = (Real.sqrt 13 - 1) / 2 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nestedSqrt_eq_nestedSqrt_value_l1136_113626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_is_sqrt_15_l1136_113671

-- Define a function to represent the simplicity of a square root
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x → (∃ (n : ℕ), Real.sqrt x = n * Real.sqrt y) → False

-- State the theorem
theorem simplest_sqrt_is_sqrt_15 :
  is_simplest_sqrt 15 ∧
  ¬is_simplest_sqrt 18 ∧
  ¬is_simplest_sqrt (1/2) ∧
  ¬is_simplest_sqrt 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_is_sqrt_15_l1136_113671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_perimeter_bound_l1136_113614

/-- A quadrilateral inscribed around a circle. -/
structure InscribedQuadrilateral where
  /-- The radius of the circle. -/
  r : ℝ
  /-- The perimeter of the quadrilateral. -/
  P : ℝ
  /-- The length of the first diagonal. -/
  l₁ : ℝ
  /-- The length of the second diagonal. -/
  l₂ : ℝ
  /-- The radius is positive. -/
  r_pos : r > 0
  /-- The perimeter is positive. -/
  P_pos : P > 0
  /-- The diagonals have positive length. -/
  l₁_pos : l₁ > 0
  l₂_pos : l₂ > 0

/-- The perimeter of a quadrilateral inscribed around a circle is at most half the sum of the squares of its diagonals. -/
theorem inscribed_quadrilateral_perimeter_bound (q : InscribedQuadrilateral) : q.P ≤ (q.l₁^2 + q.l₂^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_quadrilateral_perimeter_bound_l1136_113614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elsas_team_wins_l1136_113678

/-- Represents a hockey team's performance --/
structure TeamPerformance where
  wins : ℕ
  ties : ℕ

/-- Calculates the points for a team based on their performance --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  2 * team.wins + team.ties

/-- Theorem: Given the conditions of the hockey league, Elsa's team has 8 wins --/
theorem elsas_team_wins :
  ∃ x : ℕ,
    let firstPlace : TeamPerformance := ⟨12, 4⟩
    let secondPlace : TeamPerformance := ⟨13, 1⟩
    let elsasTeam : TeamPerformance := ⟨x, 10⟩
    let avgPoints : ℕ := 27
    let numTeams : ℕ := 3
    x = 8 ∧ 
    (calculatePoints firstPlace + calculatePoints secondPlace + calculatePoints elsasTeam) / numTeams = avgPoints := by
  sorry

#check elsas_team_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elsas_team_wins_l1136_113678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_eq_two_B_subset_A_iff_m_geq_three_l1136_113645

def A (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 2 * m + 1}

noncomputable def B : Set ℝ := {x | (1 : ℝ) / 9 ≤ Real.exp (x * Real.log 3) ∧ Real.exp (x * Real.log 3) ≤ 81}

theorem intersection_and_union_when_m_eq_two :
  (A 2 ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4}) ∧
  (A 2 ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 5}) := by sorry

theorem B_subset_A_iff_m_geq_three (m : ℝ) :
  B ⊆ A m ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_m_eq_two_B_subset_A_iff_m_geq_three_l1136_113645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1136_113698

/-- A right triangle with perimeter 36 and area 24 has hypotenuse approximately 16.67 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  a + b + c = 36 ∧  -- perimeter
  a * b / 2 = 24 ∧  -- area
  abs (c - 16.67) < 0.01 := by  -- approximate equality
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1136_113698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_formula_l1136_113629

/-- The function g satisfying the given conditions -/
noncomputable def g : ℝ → ℝ := sorry

/-- The condition g(0) = 2 -/
axiom g_zero : g 0 = 2

/-- The functional equation for g -/
axiom g_eq (x y : ℝ) : g (x * y) = g ((x^2 + y^2) / 2) + 3 * (x - y)^2

/-- The main theorem: g(x) = (3/2)x^2 - 6x + 3/2 -/
theorem g_formula (x : ℝ) : g x = (3/2) * x^2 - 6 * x + 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_formula_l1136_113629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l1136_113638

-- Define an odd function that is monotonically increasing on [0,+∞)
def is_odd_and_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

-- Main theorem
theorem odd_increasing_function_inequality 
  (f : ℝ → ℝ) 
  (h_f : is_odd_and_increasing f) 
  (a : ℝ) 
  (h_ineq : f a < f (2*a - 1)) : 
  a ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_increasing_function_inequality_l1136_113638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_minus_one_leq_one_iff_zero_leq_x_leq_four_l1136_113675

theorem abs_abs_minus_one_leq_one_iff_zero_leq_x_leq_four :
  ∀ x : ℝ, |abs (x - 2) - 1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_minus_one_leq_one_iff_zero_leq_x_leq_four_l1136_113675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l1136_113686

-- Define the function f(x) = x/e^x
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- State the theorem
theorem derivative_f (x : ℝ) : 
  deriv f x = (1 - x) / Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l1136_113686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1136_113672

-- Define the coefficients a, b, c as real numbers
variable (a b c : ℝ)

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | -3 < x ∧ x < 4}

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | x < -1/4 ∨ x > 1/3}

-- State the theorem
theorem inequality_solution_sets :
  (∀ x : ℝ, a * x^2 + b * x + c > 0 ↔ x ∈ solution_set_1) →
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ x ∈ solution_set_2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1136_113672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_go_match_probability_l1136_113654

structure GoMatch where
  prob_A_wins_as_black : ℚ
  prob_B_wins_as_black : ℚ
  prob_first_black_A : ℚ

def prob_A_wins_B_wins (m : GoMatch) : ℚ :=
  (m.prob_first_black_A * m.prob_A_wins_as_black * (1 - m.prob_A_wins_as_black)) +
  ((1 - m.prob_first_black_A) * (1 - m.prob_B_wins_as_black) * m.prob_B_wins_as_black)

theorem go_match_probability (m : GoMatch) 
  (h1 : m.prob_A_wins_as_black = 4/5)
  (h2 : m.prob_B_wins_as_black = 2/3)
  (h3 : m.prob_first_black_A = 1/2) :
  prob_A_wins_B_wins m = 17/45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_go_match_probability_l1136_113654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_control_criteria_l1136_113669

def satisfies_epidemic_control (seq : List ℕ) : Prop :=
  seq.length = 7 ∧ ∀ x ∈ seq, x ≤ 5

def average (seq : List ℕ) : ℚ :=
  (seq.sum : ℚ) / seq.length

def range (seq : List ℕ) : ℕ :=
  seq.maximum.getD 0 - seq.minimum.getD 0

def mode (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => if seq.count x > seq.count acc then x else acc) 0

theorem epidemic_control_criteria (seq : List ℕ) :
  (average seq ≤ 3 ∧ range seq ≤ 2) ∨ (mode seq = 1 ∧ range seq ≤ 4) →
  satisfies_epidemic_control seq :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_control_criteria_l1136_113669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1136_113612

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l1136_113612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_sum_bound_l1136_113676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- Represents a hyperbola -/
structure Hyperbola where
  center : Point
  a : ℝ -- real semi-axis
  b : ℝ -- imaginary semi-axis

/-- Definition of eccentricity for an ellipse -/
noncomputable def ellipse_eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Definition of eccentricity for a hyperbola -/
noncomputable def hyperbola_eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Theorem statement -/
theorem eccentricity_sum_bound 
  (F₁ F₂ P : Point) 
  (e : Ellipse) 
  (h : Hyperbola) 
  (common_foci : e.center = h.center ∧ 
    ∃ c : ℝ, (F₁.x - e.center.x)^2 + (F₁.y - e.center.y)^2 = c^2 ∧ 
             (F₂.x - e.center.x)^2 + (F₂.y - e.center.y)^2 = c^2)
  (common_point : ∃ t : ℝ, 
    (P.x - e.center.x)^2 / e.a^2 + (P.y - e.center.y)^2 / e.b^2 = 1 ∧
    (P.x - h.center.x)^2 / h.a^2 - (P.y - h.center.y)^2 / h.b^2 = 1)
  (angle_condition : 
    let v₁ := (F₁.x - P.x, F₁.y - P.y)
    let v₂ := (F₂.x - P.x, F₂.y - P.y)
    (v₁.1 * v₂.1 + v₁.2 * v₂.2) / 
    (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = 1/2) :
  1 / ellipse_eccentricity e + 1 / hyperbola_eccentricity h ≤ 4 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_sum_bound_l1136_113676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_2401_l1136_113603

theorem number_of_divisors_2401 : 
  Nat.card (Nat.divisors 2401) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_2401_l1136_113603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_is_two_mol_l1136_113692

/-- Represents the properties and behavior of an ideal gas during expansion --/
structure IdealGasExpansion where
  -- Pressure is directly proportional to volume
  p_prop_to_v : ∃ (α : ℝ), ∀ (p v : ℝ), p = α * v
  -- Temperature change in Kelvin
  delta_T : ℝ
  -- Work done by the gas in Joules
  A : ℝ
  -- Gas constant in J/(mol·K)
  R : ℝ

/-- Calculates the amount of substance involved in the ideal gas expansion process --/
noncomputable def amount_of_substance (gas : IdealGasExpansion) : ℝ :=
  2 * gas.A / (gas.R * gas.delta_T)

/-- Theorem stating that for the given conditions, the amount of substance is 2 mol --/
theorem amount_is_two_mol (gas : IdealGasExpansion) 
  (h1 : gas.delta_T = 100)
  (h2 : gas.A = 831)
  (h3 : gas.R = 8.31) : 
  amount_of_substance gas = 2 := by
  sorry

#check amount_is_two_mol

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_is_two_mol_l1136_113692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jansi_shopping_ratio_l1136_113632

/-- Represents the shopping trip of Jansi --/
structure ShoppingTrip where
  initial_rupees : ℚ
  initial_coins : ℚ
  spent : ℚ

/-- The conditions of Jansi's shopping trip --/
def jansi_trip : ShoppingTrip where
  initial_rupees := 0  -- This will be defined by the theorem
  initial_coins := 0   -- This will be defined by the theorem
  spent := 9.6

/-- The theorem representing Jansi's shopping trip --/
theorem jansi_shopping_ratio 
  (trip : ShoppingTrip) 
  (h1 : trip.initial_rupees + trip.initial_coins * (1/5) = 15) 
  (h2 : trip.spent = 9.6) :
  (15 - trip.spent) / 15 = 9 / 25 := by
  sorry

#check jansi_shopping_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jansi_shopping_ratio_l1136_113632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_semicircle_rotation_l1136_113621

noncomputable section

-- Define the radius and rotation angle
variable (R : ℝ) (h : R > 0)

-- Define the rotation angle in radians
noncomputable def α : ℝ := Real.pi / 6

-- Define the area of the shaded figure
noncomputable def shaded_area (R : ℝ) : ℝ := Real.pi * R^2 / 3

-- Theorem statement
theorem shaded_area_semicircle_rotation (R : ℝ) (h : R > 0) :
  shaded_area R = Real.pi * R^2 / 3 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_semicircle_rotation_l1136_113621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1136_113627

-- Define the points
def A : ℚ × ℚ := (-1, 0)
def B : ℚ × ℚ := (3, 0)

-- Define the property of point C being on the y-axis
def on_y_axis (C : ℚ × ℚ) : Prop := C.1 = 0

-- Define the area of a triangle given three points
noncomputable def triangle_area (P Q R : ℚ × ℚ) : ℚ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

-- Theorem statement
theorem point_C_coordinates :
  ∃ C : ℚ × ℚ, 
    on_y_axis C ∧ 
    triangle_area A B C = 6 ∧
    (C = (0, 3) ∨ C = (0, -3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1136_113627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_l1136_113641

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  let d_ab := ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2)
  let d_bc := ((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)
  let d_ca := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2)
  d_ab = d_bc ∧ d_bc = d_ca

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem -/
theorem inscribed_triangle_ratio 
  (e : Ellipse)
  (t : Triangle)
  (F1 F2 : Point)
  (h1 : e.a = 2 ∧ e.b = 3)
  (h2 : isEquilateral t)
  (h3 : t.B = ⟨2, 0⟩)
  (h4 : t.A.x = t.C.x)
  (h5 : isOnEllipse t.A e ∧ isOnEllipse t.B e ∧ isOnEllipse t.C e)
  (h6 : F1.x < F2.x ∧ F1.y = 0 ∧ F2.y = 0)
  (h7 : distance F1 F2 = 2 * Real.sqrt 5) :
  distance t.A t.B / distance F1 F2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratio_l1136_113641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_channel_finishes_earlier_l1136_113623

/-- Represents the duration of a film part in minutes -/
def film_part_duration : ℕ := 10

/-- Represents the duration of a commercial break on the first channel in minutes -/
def first_channel_break : ℕ := 12

/-- Represents the duration of a commercial break on the second channel in minutes -/
def second_channel_break : ℕ := 1

/-- Theorem stating that the first channel finishes the film one minute earlier -/
theorem first_channel_finishes_earlier (n : ℕ) :
  (n * film_part_duration + (n - 1) * first_channel_break) + 1 =
  n * film_part_duration + n * second_channel_break :=
by
  sorry

#check first_channel_finishes_earlier

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_channel_finishes_earlier_l1136_113623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l1136_113604

/-- The lateral area of a cone -/
def LateralArea (cone : Real → Real → Real) : Real := 
  sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateralTriangle (side : Real) : Prop :=
  sorry

/-- Given a cone where the axis section is an equilateral triangle with side length 1,
    prove that the lateral area of the cone is π/2. -/
theorem cone_lateral_area (cone : Real → Real → Real) 
  (h1 : ∀ x y, cone x y = 1 → IsEquilateralTriangle (cone x y))
  (h2 : ∃ x y, cone x y = 1) : 
  LateralArea cone = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_area_l1136_113604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_ellipse_l1136_113615

/-- The distance between the foci of the ellipse x²/25 + y²/9 = 12 -/
noncomputable def distance_between_foci (x y : ℝ) : ℝ :=
  16 * Real.sqrt 3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 12

theorem distance_between_foci_of_ellipse :
  ∀ x y : ℝ, ellipse_equation x y →
  distance_between_foci x y = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_foci_of_ellipse_l1136_113615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l1136_113699

/-- Represents the feeding pattern of a whale over 9 hours -/
def whale_feeding (x : ℕ) : ℕ → ℕ
  | 0 => x
  | n + 1 => whale_feeding x n + 3

/-- The total amount of plankton consumed over 9 hours -/
def total_consumption (x : ℕ) : ℕ :=
  (List.range 9).map (whale_feeding x) |>.sum

theorem whale_sixth_hour_consumption :
  ∃ x : ℕ, total_consumption x = 360 ∧ whale_feeding x 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_sixth_hour_consumption_l1136_113699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1136_113613

/-- Represents the speed of the car at the nth kilometer -/
noncomputable def speed (n : ℕ) : ℝ :=
  4 / (3 * ((n : ℝ) - 1)^2)

/-- Represents the time taken to traverse the nth kilometer -/
noncomputable def time (n : ℕ) : ℝ :=
  1 / speed n

theorem car_travel_time (n : ℕ) (h : n ≥ 3) : time n = 3 * ((n : ℝ) - 1)^2 / 4 := by
  sorry

#check car_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_time_l1136_113613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_element_l1136_113677

def S (x : ℤ) : Finset ℤ := {-4, x, 0, 6, 9}

def mean (s : Finset ℤ) : ℚ :=
  (s.sum (fun i => (i : ℚ))) / s.card

def smallest_primes : Finset ℕ := {2, 3}

theorem second_smallest_element :
  ∀ x : ℤ,
  x > -4 →
  x < 0 →
  mean (S x) * 2 ≤ mean {2, 3, 0, 6, 9} →
  x = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_element_l1136_113677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_edge_probability_l1136_113663

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : ℕ
  edges_per_vertex : ℕ
  is_regular : Prop := vertices = 12 ∧ edges_per_vertex = 3

/-- The probability of selecting two vertices that are endpoints of an edge in a dodecahedron -/
def edge_endpoint_probability (d : Dodecahedron) : ℚ :=
  let total_edges := d.vertices * d.edges_per_vertex / 2
  let total_vertex_pairs := Nat.choose d.vertices 2
  total_edges / total_vertex_pairs

/-- Theorem: The probability of selecting two vertices that are endpoints of an edge in a regular dodecahedron is 3/11 -/
theorem dodecahedron_edge_probability :
  ∀ d : Dodecahedron, d.is_regular → edge_endpoint_probability d = 3 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodecahedron_edge_probability_l1136_113663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_ratio_theorem_l1136_113611

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Check if two spheres are touching -/
def areTouching (s1 s2 : Sphere) : Prop :=
  let (x1, y1, z1) := s1.center
  let (x2, y2, z2) := s2.center
  (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2 = (s1.radius + s2.radius)^2

/-- Check if a point lies on a plane -/
def liesOnPlane (point : ℝ × ℝ × ℝ) (plane : Plane) : Prop :=
  let (nx, ny, nz) := plane.normal
  let (px, py, pz) := plane.point
  let (x, y, z) := point
  nx * (x - px) + ny * (y - py) + nz * (z - pz) = 0

/-- Calculate the distance from a point to a plane -/
noncomputable def distanceToPlane (point : ℝ × ℝ × ℝ) (plane : Plane) : ℝ :=
  let (nx, ny, nz) := plane.normal
  let (px, py, pz) := plane.point
  let (x, y, z) := point
  abs (nx * (x - px) + ny * (y - py) + nz * (z - pz)) / Real.sqrt (nx^2 + ny^2 + nz^2)

theorem sphere_ratio_theorem
  (s1 s2 s3 s4 s : Sphere) (π : Plane)
  (h1 : areTouching s1 s2 ∧ areTouching s1 s3 ∧ areTouching s1 s4 ∧
        areTouching s2 s3 ∧ areTouching s2 s4 ∧ areTouching s3 s4)
  (h2 : liesOnPlane s1.center π ∧ liesOnPlane s2.center π ∧
        liesOnPlane s3.center π ∧ liesOnPlane s4.center π)
  (h3 : areTouching s s1 ∧ areTouching s s2 ∧ areTouching s s3 ∧ areTouching s s4) :
  s.radius / distanceToPlane s.center π = 1 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_ratio_theorem_l1136_113611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poutine_price_is_eight_l1136_113650

/-- Represents the daily business operations of Lucius' food stand --/
structure FoodStand where
  daily_ingredient_cost : ℚ
  french_fries_price : ℚ
  weekly_earnings : ℚ
  tax_rate : ℚ
  days_per_week : ℕ

/-- Calculates the price of Poutine given the FoodStand parameters --/
def poutine_price (stand : FoodStand) : ℚ :=
  let weekly_ingredient_cost := stand.daily_ingredient_cost * stand.days_per_week
  let total_weekly_income := (stand.weekly_earnings + weekly_ingredient_cost) / (1 - stand.tax_rate)
  let daily_income := total_weekly_income / stand.days_per_week
  daily_income - stand.french_fries_price

/-- Theorem stating that the price of Poutine is $8 given the specified conditions --/
theorem poutine_price_is_eight :
  let stand : FoodStand := {
    daily_ingredient_cost := 10,
    french_fries_price := 12,
    weekly_earnings := 56,
    tax_rate := 1/10,
    days_per_week := 7
  }
  poutine_price stand = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_poutine_price_is_eight_l1136_113650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_c_lt_b_l1136_113652

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f'(x) + f(x)/x > 0 for x ≠ 0
axiom f_derivative_condition : ∀ x ≠ 0, deriv f x + f x / x > 0

-- Define a, b, and c
def a : ℝ := (1/2) * f (1/2)
def b : ℝ := -2 * f (-2)
def c : ℝ := Real.log (1/2) * f (Real.log (1/2))

-- Theorem to prove
theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lt_c_lt_b_l1136_113652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_classes_result_matches_theorem_l1136_113680

-- Define the given conditions
def sheets_per_class_per_day : ℕ := 200
def total_sheets_per_week : ℕ := 9000
def school_days_per_week : ℕ := 5

-- Define the theorem to prove
theorem number_of_classes :
  total_sheets_per_week / (school_days_per_week * sheets_per_class_per_day) = 9 :=
by
  -- The proof steps would go here
  sorry

-- Define the result as a natural number
def result : ℕ := 9

-- Prove that the result matches the theorem
theorem result_matches_theorem : 
  result = total_sheets_per_week / (school_days_per_week * sheets_per_class_per_day) :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_classes_result_matches_theorem_l1136_113680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_special_case_l1136_113636

/-- A geometric sequence with first term a and common ratio r -/
noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_six_special_case 
  (a : ℕ → ℝ) 
  (h_increasing : ∀ n, a n < a (n + 1)) 
  (h_roots : a 1 * a 3 = 4 ∧ a 1 + a 3 = 5) : 
  geometric_sum (a 1) ((a 3) / (a 1)) 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_six_special_case_l1136_113636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_minimum_value_condition_l1136_113666

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1/2) + 2 / (2 * x + 1)

-- Theorem 1
theorem monotone_increasing_condition (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, x > 0 → StrictMono (fun x => f a x)) → a ≥ 2 := by sorry

-- Theorem 2
theorem minimum_value_condition :
  ∃ a : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ 1) ∧ 
    (∃ x : ℝ, x > 0 ∧ f a x = 1) ∧ 
    a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_condition_minimum_value_condition_l1136_113666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_arrangements_count_l1136_113622

/-- Represents the number of students participating in the tests -/
def num_students : ℕ := 4

/-- Represents the number of projects being tested -/
def num_projects : ℕ := 5

/-- Represents whether a project can be tested in the morning -/
def morning_allowed : Fin num_projects → Bool :=
  sorry

/-- Represents whether a project can be tested in the afternoon -/
def afternoon_allowed : Fin num_projects → Bool :=
  sorry

/-- The number of different test arrangements possible given the constraints -/
def num_arrangements (morning_allowed : Fin num_projects → Bool) 
                     (afternoon_allowed : Fin num_projects → Bool) : ℕ :=
  sorry

theorem test_arrangements_count :
  ∃ (grip_strength stairs : Fin num_projects),
    grip_strength ≠ stairs ∧
    morning_allowed = (λ p => p ≠ grip_strength) ∧
    afternoon_allowed = (λ p => p ≠ stairs) ∧
    num_arrangements morning_allowed afternoon_allowed = 264 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_arrangements_count_l1136_113622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1136_113631

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + 4/5 * t, -1 - 3/5 * t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt 2 * Real.cos (θ + Real.pi/4)

-- Define the ordinary equation of line l
def line_l_ordinary (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

-- Define the rectangular coordinate equation of curve C
def curve_C_rectangular (x y : ℝ) : Prop := x^2 + y^2 - x + y = 0

-- Theorem: The length of the chord cut off by line l on curve C is 7/5
theorem chord_length : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l_ordinary x₁ y₁ ∧ 
    line_l_ordinary x₂ y₂ ∧ 
    curve_C_rectangular x₁ y₁ ∧ 
    curve_C_rectangular x₂ y₂ ∧ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7/5 := by
  sorry

#check chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1136_113631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l1136_113659

theorem triangle_angle_C (A B C : Real) (a b c : Real) : 
  A = π/3 → c = 4 → a = 2 * Real.sqrt 6 → 
  0 < A ∧ A < π → 0 < B ∧ B < π → 0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  C = π/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_C_l1136_113659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1136_113685

noncomputable def v1 : ℝ × ℝ × ℝ := (4, 2, -3)
noncomputable def v2 : ℝ × ℝ × ℝ := (4, -2, 10)

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := v
  Real.sqrt (x^2 + y^2 + z^2)

theorem parallelogram_area : magnitude (cross_product v1 v2) = 56 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1136_113685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1136_113647

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the line passing through the right focus
def Line (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - c}

-- Define the collinearity condition
def IsCollinear (v w : ℝ × ℝ) : Prop :=
  3 * (v.2 + w.2) + (v.1 + w.1) = 0

-- Main theorem
theorem ellipse_eccentricity (a b c : ℝ) (A B : ℝ × ℝ)
  (h_ellipse : A ∈ Ellipse a b ∧ B ∈ Ellipse a b)
  (h_line : A ∈ Line c ∧ B ∈ Line c)
  (h_collinear : IsCollinear A B)
  (h_positive : a > 0 ∧ b > 0)
  : (c / a)^2 = 6 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1136_113647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_sum_l1136_113674

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def digit_sum (n : ℕ) : ℕ :=
  n / 10 + n % 10

theorem two_digit_reverse_sum (n : ℕ) :
  is_two_digit n →
  (n : ℤ) - (reverse_digits n : ℤ) = 7 * (digit_sum n : ℤ) →
  n + reverse_digits n = 99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_sum_l1136_113674
