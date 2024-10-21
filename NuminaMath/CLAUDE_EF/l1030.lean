import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l1030_103096

/-- A lattice point on a 6x6 grid. -/
structure LatticePoint where
  x : Fin 6
  y : Fin 6

/-- A square on the 6x6 lattice grid. -/
structure LatticeSquare where
  vertices : Fin 4 → LatticePoint

/-- Determines if two squares are congruent. -/
def areCongruent (s1 s2 : LatticeSquare) : Prop :=
  sorry

/-- The set of all possible squares on the 6x6 lattice grid. -/
def allSquares : Set LatticeSquare :=
  sorry

/-- The set of non-congruent squares on the 6x6 lattice grid. -/
noncomputable def nonCongruentSquares : Finset LatticeSquare :=
  sorry

/-- The number of non-congruent squares on a 6x6 lattice grid is 141. -/
theorem count_non_congruent_squares :
  Finset.card nonCongruentSquares = 141 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l1030_103096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_distributor_cost_l1030_103021

/-- The cost of an item for a distributor, given specific conditions on commission, profit, and final price. -/
noncomputable def distributor_cost (commission_rate : ℝ) (profit_rate : ℝ) (final_price : ℝ) : ℝ :=
  let selling_price := final_price / (1 - commission_rate)
  let cost := selling_price / (1 + profit_rate)
  cost

/-- The cost of the item for the distributor is $29.6875, given the specified conditions. -/
theorem specific_distributor_cost :
  distributor_cost 0.2 0.2 28.5 = 29.6875 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_distributor_cost_l1030_103021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_l1030_103048

/-- A sequence of numbers obtained by repeatedly replacing two numbers with their average divided by 2 -/
inductive ReplacementSequence : ℕ → List ℝ → Prop
  | base : ReplacementSequence 0 (List.replicate n 1)
  | step {n : ℕ} {l l' : List ℝ} {a b : ℝ} :
      ReplacementSequence n l' →
      l'.length ≥ 2 →
      a ∈ l' →
      b ∈ l' →
      l = (l'.erase a).erase b ++ [(a + b) / 4] →
      ReplacementSequence (n + 1) l

/-- The theorem stating that the final number after n-1 steps is at least 1/n -/
theorem final_number_lower_bound (n : ℕ) (h : n > 0) :
  ∀ (x : ℝ), ReplacementSequence (n-1) [x] → x ≥ 1 / n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_lower_bound_l1030_103048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l1030_103003

/-- The point where the mouse starts getting farther from the cheese -/
def turning_point : ℚ × ℚ := (24/37, 123/37)

/-- The distance traveled by the mouse from the starting point to the turning point -/
def distance_traveled : ℚ := 250/37

/-- Theorem stating the turning point and distance traveled for the mouse problem -/
theorem mouse_cheese_problem (x y : ℚ) :
  let cheese_loc : ℚ × ℚ := (15, 12)
  let start_point : ℚ × ℚ := (3, -3)
  let mouse_line (x : ℚ) : ℚ := -6 * x + 15
  (∀ t : ℚ, mouse_line t = -6 * t + 15) →
  (turning_point.1 = 24/37 ∧ turning_point.2 = 123/37) →
  (turning_point.2 = mouse_line turning_point.1) →
  (∀ p : ℚ × ℚ, p.2 = mouse_line p.1 → 
    (p.1 < turning_point.1 → (p.1 - cheese_loc.1)^2 + (p.2 - cheese_loc.2)^2 < 
      (turning_point.1 - cheese_loc.1)^2 + (turning_point.2 - cheese_loc.2)^2) ∧
    (p.1 > turning_point.1 → (p.1 - cheese_loc.1)^2 + (p.2 - cheese_loc.2)^2 > 
      (turning_point.1 - cheese_loc.1)^2 + (turning_point.2 - cheese_loc.2)^2)) →
  distance_traveled = 
    Real.sqrt ((turning_point.1 - start_point.1)^2 + (turning_point.2 - start_point.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l1030_103003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probability_l1030_103024

/-- Represents a particle's position on a 2D grid -/
structure Position where
  x : ℕ
  y : ℕ

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Right

/-- Calculates the binomial coefficient -/
def binomial (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Simulates the movement of a particle -/
def move (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Calculates the probability of reaching a specific position after n moves -/
def probability_to_reach (start end_ : Position) (n : ℕ) : ℚ :=
  let right_moves := end_.x - start.x
  let up_moves := end_.y - start.y
  if right_moves + up_moves ≠ n then 0
  else (binomial n right_moves : ℚ) * (1 / 2) ^ n

theorem particle_movement_probability :
  let start := ⟨0, 0⟩
  let end_ := ⟨2, 3⟩
  let total_moves := 5
  probability_to_reach start end_ total_moves = (binomial total_moves 2 : ℚ) * (1 / 2) ^ total_moves := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_movement_probability_l1030_103024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_bisecting_chord_equation_l1030_103043

/-- Definition of an ellipse with given parameters -/
noncomputable def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of eccentricity for an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem about the equation of an ellipse with specific parameters -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : eccentricity a b = Real.sqrt 3 / 2) (h4 : 2 * b = 4) :
  Ellipse a b = Ellipse 4 2 := by sorry

/-- Definition of a chord passing through a point -/
def Chord (x₀ y₀ m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - y₀ = m * (p.1 - x₀)}

/-- Theorem about the equation of the bisecting chord -/
theorem bisecting_chord_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : eccentricity a b = Real.sqrt 3 / 2) (h4 : 2 * b = 4) :
  ∃ m : ℝ, Chord 2 1 m ∩ Ellipse a b = Chord 2 1 (-1/2) ∩ Ellipse a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_bisecting_chord_equation_l1030_103043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_students_punctual_l1030_103007

universe u

variable (Student : Type u)
variable (isPunctual : Student → Prop)

theorem negation_of_all_students_punctual :
  (¬ ∀ (s : Student), isPunctual s) ↔ (∃ (s : Student), ¬ isPunctual s) :=
by
  apply Iff.intro
  · intro h
    by_contra hc
    apply h
    intro s
    by_contra hnp
    exact hc ⟨s, hnp⟩
  · intro ⟨s, hnp⟩ h
    exact hnp (h s)

#check negation_of_all_students_punctual

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_students_punctual_l1030_103007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_dne_l1030_103082

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.sin x * Real.cos (5 / x) else 0

-- State the theorem
theorem derivative_f_at_zero_dne :
  ¬ (∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 
    0 < |h| ∧ |h| < δ → |((f (0 + h) - f 0) / h) - L| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_zero_dne_l1030_103082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contingency_table_independence_test_l1030_103008

/-- Represents the chi-square statistic for a 2x2 contingency table -/
def chi_square : ℝ := 13.097

/-- The critical value for the chi-square distribution with 1 degree of freedom at 0.01 significance level -/
def critical_value : ℝ := 6.635

/-- The significance level we want to prove -/
def significance_level : ℝ := 0.01

/-- Represents the probability of making a mistake in assuming the variables are related -/
def mistake_probability : ℝ → ℝ := sorry

theorem contingency_table_independence_test :
  chi_square > critical_value →
  ∃ p : ℝ, p ≤ significance_level ∧ p = mistake_probability chi_square :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contingency_table_independence_test_l1030_103008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_steve_time_ratio_l1030_103091

/-- Prove that the ratio of Danny's time to Steve's time is 1:2 -/
theorem danny_steve_time_ratio :
  let danny_time : ℚ := 31
  let steve_halfway_time : ℚ := danny_time / 2 + 15.5
  let steve_time : ℚ := steve_halfway_time * 2
  danny_time / steve_time = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_steve_time_ratio_l1030_103091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riverside_park_ducks_percentage_l1030_103081

/-- Represents the percentage of each type of bird in Riverside Park -/
structure BirdPercentages where
  geese : ℚ
  pelicans : ℚ
  herons : ℚ
  ducks : ℚ

/-- Calculates the percentage of ducks among non-heron birds -/
def ducks_among_non_herons (bp : BirdPercentages) : ℚ :=
  (bp.ducks / (100 - bp.herons)) * 100

/-- Theorem stating that given the bird percentages in Riverside Park,
    the percentage of ducks among non-heron birds is approximately 30% -/
theorem riverside_park_ducks_percentage :
  let bp : BirdPercentages := {
    geese := 20,
    pelicans := 40,
    herons := 15,
    ducks := 25
  }
  ∃ ε > 0, |ducks_among_non_herons bp - 30| < ε := by
  sorry

#eval ducks_among_non_herons {
  geese := 20,
  pelicans := 40,
  herons := 15,
  ducks := 25
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_riverside_park_ducks_percentage_l1030_103081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_two_implications_l1030_103011

theorem tan_alpha_eq_two_implications (α : ℝ) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧ ((Real.sin (2*α) - Real.cos α^2) / (1 + Real.cos (2*α)) = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_two_implications_l1030_103011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l1030_103088

theorem min_floor_equation (n : ℕ) : 
  (∃ k : ℕ, k^2 + (n / k^2 : ℕ) = 1991) ∧ 
  (∀ k : ℕ, k^2 + (n / k^2 : ℕ) ≥ 1991) ↔ 
  1024 * 967 ≤ n ∧ n < 1024 * 968 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_floor_equation_l1030_103088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1030_103005

/-- The value of r that makes the circle x^2 + y^2 = r^2 tangent to the line x + y = r + 1 -/
theorem circle_tangent_to_line (r : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → x + y = r + 1 → 
   ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = r^2 ∧ p.1 + p.2 = r + 1) → 
  r = Real.sqrt 2 + 1 := by
  sorry

#check circle_tangent_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1030_103005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_power_equation_l1030_103066

theorem sixteen_power_equation (x : ℝ) : (16 : ℝ)^(x + 1) = 256 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixteen_power_equation_l1030_103066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_square_remainders_mod_11_l1030_103041

theorem sum_of_distinct_square_remainders_mod_11 : 
  (Finset.sum (Finset.filter (λ r : ℕ => r ∈ Finset.image 
    (λ n : ℕ => n^2 % 11) (Finset.range 10)) (Finset.range 11)) id) / 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distinct_square_remainders_mod_11_l1030_103041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grace_baked_four_pies_verify_solution_l1030_103052

/-- The number of whole pumpkin pies Grace baked -/
def P : ℕ := 4

/-- The number of whole pies sold or given away -/
def sold_or_given : ℕ := 2

/-- The number of slices each remaining pie was cut into -/
def slices_per_pie : ℕ := 6

/-- The fraction of slices eaten by Grace's family -/
def fraction_eaten : ℚ := 2/3

/-- The number of pie slices left -/
def slices_left : ℕ := 4

/-- Theorem stating that Grace baked 4 whole pumpkin pies -/
theorem grace_baked_four_pies : P = 4 := by
  rfl  -- reflexivity, since P is defined as 4

/-- Theorem verifying the problem solution -/
theorem verify_solution : 
  (P - sold_or_given) * slices_per_pie * fraction_eaten + slices_left = 
  (P - sold_or_given) * slices_per_pie := by
  sorry  -- The actual proof would go here, but we're using sorry as requested

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grace_baked_four_pies_verify_solution_l1030_103052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_value_l1030_103067

-- Define the circle C
def circleC (x y r : ℝ) : Prop := (x - 5)^2 + (y - 4)^2 = r^2

-- Define the condition for point P
def point_condition (x y : ℝ) : Prop := x^2 + y^2 = 2*((x - 1)^2 + y^2)

-- Theorem statement
theorem min_radius_value :
  ∀ r : ℝ, r > 0 →
  (∃ x y : ℝ, circleC x y r ∧ point_condition x y) →
  (∀ s : ℝ, s > 0 → (∃ x y : ℝ, circleC x y s ∧ point_condition x y) → s ≥ r) →
  r = 5 - Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_radius_value_l1030_103067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_days_closest_to_million_seconds_l1030_103029

-- Define constants
noncomputable def seconds_in_minute : ℝ := 60
noncomputable def minutes_in_hour : ℝ := 60
noncomputable def hours_in_day : ℝ := 24
noncomputable def days_in_year : ℝ := 365.25  -- Accounting for leap years

-- Define the number of seconds we're comparing to
noncomputable def million_seconds : ℝ := 1000000

-- Function to convert seconds to days
noncomputable def seconds_to_days (s : ℝ) : ℝ :=
  s / (seconds_in_minute * minutes_in_hour * hours_in_day)

-- Define the options given in the problem
noncomputable def option_a : ℝ := 1        -- 1 day
noncomputable def option_b : ℝ := 10       -- 10 days
noncomputable def option_c : ℝ := 100      -- 100 days
noncomputable def option_d : ℝ := days_in_year  -- 1 year
noncomputable def option_e : ℝ := 10 * days_in_year  -- 10 years

-- Function to calculate the absolute difference between two numbers
noncomputable def abs_diff (x y : ℝ) : ℝ := abs (x - y)

-- Theorem stating that 10 days is the closest to one million seconds
theorem ten_days_closest_to_million_seconds :
  let million_days := seconds_to_days million_seconds
  abs_diff million_days option_b = 
    min (abs_diff million_days option_a)
      (min (abs_diff million_days option_b)
        (min (abs_diff million_days option_c)
          (min (abs_diff million_days option_d)
            (abs_diff million_days option_e)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_days_closest_to_million_seconds_l1030_103029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_perpendicular_ray_l1030_103053

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane (implementation details omitted)
  mk :: -- Add a constructor to avoid errors

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point is in a plane -/
def isInPlane (p : Point3D) (k : Plane3D) : Prop :=
  sorry

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Find the foot of the perpendicular from a point to a plane -/
noncomputable def perpendicularFoot (q : Point3D) (k : Plane3D) : Point3D :=
  sorry

/-- Check if three points form a right angle -/
def isRightAngle (p q r : Point3D) : Prop :=
  sorry

/-- Calculate the ratio (QP + PR) / QR -/
noncomputable def ratio (p q r : Point3D) : ℝ :=
  (distance q p + distance p r) / distance q r

theorem max_ratio_on_perpendicular_ray 
  (k : Plane3D) (p q : Point3D) 
  (h1 : isInPlane p k) 
  (h2 : ¬isInPlane q k) :
  ∃ (r : Point3D), isInPlane r k ∧
    ∀ (r' : Point3D), isInPlane r' k →
      ratio p q r ≥ ratio p q r' :=
by
  let x := perpendicularFoot q k
  let s := Point3D.mk (2 * p.x - q.x) (2 * p.y - q.y) (2 * p.z - q.z)  -- S such that PS = PQ
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_perpendicular_ray_l1030_103053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_value_l1030_103019

noncomputable def original_function (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 4)

noncomputable def left_translated_function (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + (ω - 1) * Real.pi / 4)

noncomputable def right_translated_function (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - (ω + 1) * Real.pi / 4)

theorem smallest_omega_value :
  ∃ (ω : ℝ), ω > 0 ∧
  (∀ (x : ℝ), left_translated_function ω x = right_translated_function ω x ∨
               ∃ (k : ℤ), left_translated_function ω x = right_translated_function ω x + k * Real.pi) ∧
  (∀ (ω' : ℝ), ω' > 0 ∧ ω' < ω →
    ¬(∀ (x : ℝ), left_translated_function ω' x = right_translated_function ω' x ∨
                  ∃ (k : ℤ), left_translated_function ω' x = right_translated_function ω' x + k * Real.pi)) ∧
  ω = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_omega_value_l1030_103019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_change_ratio_l1030_103045

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water in a mixture -/
noncomputable def ratio (m : Mixture) : ℝ := m.alcohol / m.water

/-- Adds water to a mixture -/
def add_water (m : Mixture) (amount : ℝ) : Mixture :=
  { alcohol := m.alcohol, water := m.water + amount }

theorem water_added_to_change_ratio (m : Mixture) (added : ℝ) :
  ratio m = 4 / 3 →
  m.alcohol = 10 →
  ratio (add_water m added) = 4 / 5 →
  added = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_change_ratio_l1030_103045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1030_103049

theorem polynomial_divisibility (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) = 
    (∀ x : ℂ, (x^2 + x + 1) ∣ ((x + 1)^n - x^n - 1)) ∧
  (∃ k : ℤ, n = 6 * k + 2 ∨ n = 6 * k - 2) = 
    (∀ x : ℂ, (x^2 + x + 1) ∣ ((x + 1)^n + x^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1030_103049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_b_value_l1030_103012

/-- A custom polynomial of degree 4 with integer coefficients -/
structure CustomPolynomial :=
  (a : ℤ)
  (b : ℤ)

/-- The roots of the polynomial -/
structure Roots :=
  (r₁ : ℕ+)
  (r₂ : ℕ+)
  (r₃ : ℕ+)
  (r₄ : ℕ+)

/-- The polynomial has the form z^4 - 6z^3 + Az^2 + Bz + 9 -/
def is_valid_polynomial (p : CustomPolynomial) : Prop :=
  ∃ (r : Roots), 
    r.r₁ + r.r₂ + r.r₃ + r.r₄ = 6 ∧
    r.r₁ * r.r₂ * r.r₃ + r.r₁ * r.r₂ * r.r₄ + r.r₁ * r.r₃ * r.r₄ + r.r₂ * r.r₃ * r.r₄ = -p.b

theorem polynomial_b_value (p : CustomPolynomial) :
  is_valid_polynomial p → p.b = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_b_value_l1030_103012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_theorem_l1030_103030

/-- A regular convex heptagon -/
structure RegularConvexHeptagon where
  vertices : Finset (Fin 7)
  sides : Finset (Fin 7 × Fin 7)
  diagonals : Finset (Fin 7 × Fin 7)

/-- Properties of a regular convex heptagon -/
axiom heptagon_properties (H : RegularConvexHeptagon) :
  H.vertices.card = 7 ∧
  H.sides.card = 7 ∧
  H.diagonals.card = 14

/-- Count of intersections -/
def count_intersections (lines : Finset (Fin 7 × Fin 7)) : ℕ := sorry

/-- Count of triangles -/
def count_triangles (lines : Finset (Fin 7 × Fin 7)) : ℕ := sorry

/-- Count of isosceles triangles -/
def count_isosceles_triangles (lines : Finset (Fin 7 × Fin 7)) : ℕ := sorry

/-- Count of regions -/
def count_regions (H : RegularConvexHeptagon) : ℕ := sorry

theorem heptagon_theorem (H : RegularConvexHeptagon) :
  count_intersections H.sides = 21 ∧
  count_intersections H.diagonals = 49 ∧
  count_intersections (H.sides ∪ H.diagonals) = 91 ∧
  count_triangles H.sides = 35 ∧
  count_triangles H.diagonals = 252 ∧
  count_triangles (H.sides ∪ H.diagonals) = 805 ∧
  count_isosceles_triangles H.sides = 21 ∧
  count_isosceles_triangles H.diagonals = 154 ∧
  count_isosceles_triangles (H.sides ∪ H.diagonals) = 483 ∧
  count_regions H = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_theorem_l1030_103030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1030_103014

-- Define the parameters for Route X
noncomputable def route_x_distance : ℝ := 8
noncomputable def route_x_speed : ℝ := 40

-- Define the parameters for Route Y
noncomputable def route_y_total_distance : ℝ := 6
noncomputable def route_y_normal_distance : ℝ := 5
noncomputable def route_y_construction_distance : ℝ := 1
noncomputable def route_y_normal_speed : ℝ := 50
noncomputable def route_y_construction_speed : ℝ := 10

-- Define the function to calculate time in minutes
noncomputable def time_in_minutes (distance : ℝ) (speed : ℝ) : ℝ :=
  (distance / speed) * 60

-- Calculate the time for Route X
noncomputable def route_x_time : ℝ :=
  time_in_minutes route_x_distance route_x_speed

-- Calculate the time for Route Y
noncomputable def route_y_time : ℝ :=
  time_in_minutes route_y_normal_distance route_y_normal_speed +
  time_in_minutes route_y_construction_distance route_y_construction_speed

-- Theorem to prove
theorem route_time_difference :
  route_y_time - route_x_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l1030_103014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_2_eq_0_l1030_103075

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * (1 - x)
  else if 1 < x ∧ x ≤ 2 then x - 1
  else 0  -- We add this else clause to make the function total

-- Define f_n recursively
noncomputable def f_n : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => f (f_n n x)

-- State the theorem
theorem f_2018_2_eq_0 : f_n 2018 2 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_2_eq_0_l1030_103075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l1030_103069

noncomputable def F (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 1 then x^2 
  else if 1 < x ∧ x ≤ 2 then x + 1 
  else 0

theorem work_done_by_force : 
  (∫ x in Set.Icc 0 1, F x) + (∫ x in Set.Ico 1 2, F x) = 17/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_by_force_l1030_103069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purified_area_l1030_103034

noncomputable def path_length (t : ℝ) : ℝ := 140 - |t - 40|

noncomputable def water_width (t a : ℝ) : ℝ := 1 + a^2 / t

noncomputable def purified_area (t a : ℝ) : ℝ := path_length t * water_width t a

theorem min_purified_area (a : ℕ) (h : a ≥ 1) :
  (∃ (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 60), 
    ∀ (s : ℕ) (hs : 1 ≤ s ∧ s ≤ 60), purified_area (t : ℝ) (a : ℝ) ≤ purified_area (s : ℝ) (a : ℝ)) ∧
  (if a = 1 then
    (∃ (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 60), purified_area (t : ℝ) (a : ℝ) = 121)
   else
    (∃ (t : ℕ) (ht : 1 ≤ t ∧ t ≤ 60), purified_area (t : ℝ) (a : ℝ) = 2 * (a^2 : ℝ) + 120)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_purified_area_l1030_103034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jhons_daily_pay_l1030_103094

theorem jhons_daily_pay (total_days : ℕ) (present_days : ℕ) (absent_pay : ℚ) (total_pay : ℚ) :
  total_days = 60 →
  present_days = 35 →
  absent_pay = 3 →
  total_pay = 170 →
  ∃ (present_pay : ℚ),
    present_pay * present_days + absent_pay * (total_days - present_days) = total_pay ∧
    (present_pay * 100).floor / 100 = 271 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jhons_daily_pay_l1030_103094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_ratio_l1030_103078

theorem arithmetic_geometric_mean_ratio : 
  ∃ (a b : ℝ), 
    a > b ∧ 
    b > 0 ∧ 
    (a + b) / 2 = 3 * Real.sqrt (a * b) ∧ 
    Int.floor (a / b + 0.5) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_mean_ratio_l1030_103078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangle_solutions_l1030_103055

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  let d := Real.sqrt
  d ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 3 ∧
  d ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) = 4 ∧
  Real.arccos ((t.A.1 - t.C.1) * (t.B.1 - t.C.1) + (t.A.2 - t.C.2) * (t.B.2 - t.C.2)) /
    (d ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) * d ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)) = Real.pi / 6

-- Theorem statement
theorem two_triangle_solutions :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ 
    satisfiesConditions t1 ∧ 
    satisfiesConditions t2 ∧
    ∀ (t : Triangle), satisfiesConditions t → (t = t1 ∨ t = t2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_triangle_solutions_l1030_103055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cylinder_specific_gravity_l1030_103093

/-- The specific gravity of a cylinder floating in water -/
noncomputable def cylinder_specific_gravity : ℝ := (1/3) * (1 - (3 * Real.sqrt 3) / (4 * Real.pi))

/-- 
  Theorem: The specific gravity of a solid cylinder floating on water 
  with its axis parallel to the water surface and submerged to a depth 
  equal to half of its radius is equal to (1/3) * (1 - (3√3)/(4π)).
-/
theorem floating_cylinder_specific_gravity : 
  ∀ (r : ℝ), r > 0 → 
  cylinder_specific_gravity = (1/3) * (1 - (3 * Real.sqrt 3) / (4 * Real.pi)) :=
by
  sorry

/-- Approximation of cylinder_specific_gravity as a Float -/
def cylinder_specific_gravity_float : Float := 0.1955

#eval cylinder_specific_gravity_float

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floating_cylinder_specific_gravity_l1030_103093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaO_l1030_103086

noncomputable section

/-- The mass percentage of an element in a compound -/
def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

/-- The molar mass of barium (Ba) in g/mol -/
def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of oxygen (O) in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- The molar mass of barium oxide (BaO) in g/mol -/
def molar_mass_BaO : ℝ := molar_mass_Ba + molar_mass_O

theorem mass_percentage_Ba_in_BaO :
  abs (mass_percentage molar_mass_Ba molar_mass_BaO - 89.55) < 0.01 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaO_l1030_103086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_identity_l1030_103028

theorem cosine_sine_identity (x : ℝ) : 
  Real.cos x + 3 * Real.sin x = 2 → 3 * Real.sin x - Real.cos x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_identity_l1030_103028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_conditions_l1030_103071

-- Define the divisors
def divisors : List ℚ := [3/5, 5/7, 7/9, 9/11]

-- Define the expected fractional parts
def fractional_parts : List ℚ := [2/3, 2/5, 2/7, 2/9]

-- Function to check if a number satisfies the conditions
def satisfies_conditions (n : ℕ) : Prop :=
  ∀ (i : Fin 4), 
    (n : ℚ) / divisors[i.val] - ((n : ℚ) / divisors[i.val]).floor = fractional_parts[i.val]

theorem smallest_integer_satisfying_conditions :
  satisfies_conditions 316 ∧
  ∀ (m : ℕ), 1 < m ∧ m < 316 → ¬(satisfies_conditions m) := by
  sorry

#check smallest_integer_satisfying_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_satisfying_conditions_l1030_103071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1030_103073

theorem triangle_cosine_theorem (D E F : ℝ) : 
  D + E + F = Real.pi →
  Real.sin D = 4/5 →
  Real.cos E = 12/13 →
  Real.cos F = -16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_theorem_l1030_103073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_theorem_l1030_103085

-- Define the polynomial
noncomputable def P (x k : ℝ) : ℝ := x^4 + 2*x^3 + (3+k)*x^2 + (2+k)*x + 2*k

-- Define the condition that the equation has real roots
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, P x k = 0

-- Define the product of roots
noncomputable def product_of_roots (k : ℝ) : ℝ := 2 / k

-- Define the sum of squares of roots
noncomputable def sum_of_squares_of_roots (k : ℝ) : ℝ :=
  let x₁ := (-1 + Real.sqrt (1 + 4*(-k))) / 2
  let x₂ := (-1 - Real.sqrt (1 + 4*(-k))) / 2
  x₁^2 + x₂^2

-- Theorem statement
theorem sum_of_squares_theorem (k : ℝ) :
  has_real_roots k ∧ product_of_roots k = -2 →
  sum_of_squares_of_roots k = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_theorem_l1030_103085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l1030_103027

open Set

-- Define the universal set U as the real numbers
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {y | ∃ x, y = 2^(Real.sqrt (2*x - x^2 + 3))}

-- Define set N
def N : Set ℝ := Ioo (-3) 2

-- State the theorem
theorem complement_M_intersect_N :
  (U \ M) ∩ N = Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_l1030_103027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_project_duration_l1030_103051

/-- Represents the construction project parameters -/
structure TunnelProject where
  totalLength : ℕ
  initialWorkforce : ℕ
  daysWorked : ℕ
  lengthCompleted : ℕ
  additionalWorkforce : ℕ

/-- Calculates the initial planned duration for the tunnel construction project -/
def initialPlannedDuration (project : TunnelProject) : ℕ :=
  let initialRate := project.lengthCompleted / project.daysWorked
  let remainingLength := project.totalLength - project.lengthCompleted
  let newWorkforce := project.initialWorkforce + project.additionalWorkforce
  let newRate := initialRate * newWorkforce / project.initialWorkforce
  let remainingDays := remainingLength / newRate
  project.daysWorked + remainingDays

/-- Theorem stating that the initial planned duration for the given project is 220 days -/
theorem tunnel_project_duration (project : TunnelProject) 
  (h1 : project.totalLength = 720)
  (h2 : project.initialWorkforce = 50)
  (h3 : project.daysWorked = 120)
  (h4 : project.lengthCompleted = 240)
  (h5 : project.additionalWorkforce = 70) :
  initialPlannedDuration project = 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_project_duration_l1030_103051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1030_103025

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

/-- Proves that for a geometric sequence with sum S_n, if 8S_6 = 7S_3, then the common ratio is -1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (8 * geometricSum a r 6 = 7 * geometricSum a r 3) →
  r = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1030_103025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_correct_theorem_l1030_103032

/-- The probability that both individuals A and B solve a problem correctly,
    given their individual error probabilities. -/
def probability_both_correct (a b : ℝ) : ℝ := (1 - a) * (1 - b)

theorem probability_both_correct_theorem (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  ∃ (p : ℝ), p = (1 - a) * (1 - b) ∧ 
  p = probability_both_correct a b := by
  use (1 - a) * (1 - b)
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_correct_theorem_l1030_103032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1030_103059

noncomputable def f (x : ℝ) := Real.exp x / Real.exp x

theorem f_properties :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max ∧ f x_max = 1) ∧
  (∃ (t_max : ℝ), ∀ (t : ℝ),
    (Real.exp (1 - t) / Real.exp t - Real.exp (t^2) / Real.exp t) ≤ Real.exp 2) ∧
  (∀ (x : ℝ), f (f x) = x ↔ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1030_103059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1030_103020

/-- An arithmetic progression with a non-zero common difference -/
structure ArithmeticProgression where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a 1 + (n - 1) * ap.d)

theorem arithmetic_geometric_ratio
  (ap : ArithmeticProgression)
  (h_geom : (ap.a 2) ^ 2 = ap.a 1 * ap.a 4) :
  sum_n ap 4 / sum_n ap 2 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1030_103020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_comparison_choose_travel_agency_l1030_103002

/-- Represents the cost calculation for a school trip based on the number of students -/
noncomputable def TripCost (full_price : ℝ) : ℝ → (ℝ → ℝ) × (ℝ → ℝ) :=
  λ num_students =>
    let cost_a := λ x : ℝ => full_price + (full_price / 2) * x
    let cost_b := λ x : ℝ => full_price * 0.6 * (x + 1)
    (cost_a, cost_b)

/-- Theorem stating the properties of the trip cost for different numbers of students -/
theorem trip_cost_comparison (full_price : ℝ) (h_price : full_price = 240) :
  ∃ (x : ℝ),
    let (cost_a, cost_b) := TripCost full_price x
    (x = 4 → cost_a x = cost_b x) ∧
    (x > 4 → cost_a x < cost_b x) ∧
    (x < 4 → cost_a x > cost_b x) :=
by sorry

/-- Corollary stating which travel agency to choose based on the number of students -/
theorem choose_travel_agency (full_price : ℝ) (h_price : full_price = 240) (num_students : ℝ) :
  let (cost_a, cost_b) := TripCost full_price num_students
  (num_students = 4 → cost_a num_students = cost_b num_students) ∧
  (num_students > 4 → cost_a num_students < cost_b num_students) ∧
  (num_students < 4 → cost_a num_students > cost_b num_students) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_comparison_choose_travel_agency_l1030_103002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1030_103063

/-- The speed of the firetruck on roads in miles per hour -/
noncomputable def road_speed : ℝ := 60

/-- The speed of the firetruck in the desert in miles per hour -/
noncomputable def desert_speed : ℝ := 10

/-- The time limit in minutes -/
noncomputable def time_limit : ℝ := 5

/-- The area of reachable points for the firetruck -/
noncomputable def reachable_area : ℝ := (25 * Real.pi) / 9

theorem firetruck_reachable_area :
  let max_road_distance := road_speed * time_limit / 60
  let max_desert_distance := desert_speed * time_limit / 60
  (4 * Real.pi * (max_desert_distance ^ 2)) = reachable_area := by
  sorry

#check firetruck_reachable_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1030_103063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_perimeter_l1030_103046

theorem two_squares_perimeter (p1 p2 : ℝ) (h1 : p1 = 40) (h2 : p2 = 100) : 
  p1 + p2 - 2 * (p1 / 4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_squares_perimeter_l1030_103046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_guards_per_team_l1030_103016

theorem guards_per_team (forwards guards centers num_teams : Nat)
  (h_forwards : forwards = 64)
  (h_guards : guards = 160)
  (h_centers : centers = 48)
  (h_min_centers : ∀ team : Nat, team ≥ 2)
  (h_all_included : num_teams * (forwards / num_teams + guards / num_teams + centers / num_teams) = forwards + guards + centers)
  (h_equal_forwards : num_teams * (forwards / num_teams) = forwards)
  (h_equal_guards : num_teams * (guards / num_teams) = guards)
  (h_max_teams : num_teams = Nat.gcd forwards (Nat.gcd guards (2 * centers))) :
  guards / num_teams = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_guards_per_team_l1030_103016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l1030_103099

/-- Given points A, B, and C in ℝ³, this theorem states that the plane equation
    5x - 3y + z - 18 = 0 represents a plane passing through point A and
    perpendicular to vector BC. -/
theorem plane_equation_through_point_perpendicular_to_vector
  (A B C : ℝ × ℝ × ℝ)
  (h_A : A = (3, -3, -6))
  (h_B : B = (1, 9, -5))
  (h_C : C = (6, 6, -4)) :
  let plane_eq := fun (p : ℝ × ℝ × ℝ) ↦ 5 * p.1 - 3 * p.2.1 + p.2.2 - 18 = 0
  let vector_BC := (C.1 - B.1, C.2.1 - B.2.1, C.2.2 - B.2.2)
  plane_eq A ∧
  (∀ (p : ℝ × ℝ × ℝ), plane_eq p → 
    (p.1 - A.1, p.2.1 - A.2.1, p.2.2 - A.2.2) • vector_BC = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_through_point_perpendicular_to_vector_l1030_103099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_collecting_sticks_l1030_103017

/-- The number of lollipop sticks needed for the fort -/
def total_sticks : ℕ := 1500

/-- The average number of store visits per week -/
def avg_store_visits : ℚ := 3.5

/-- The number of sticks Felicity receives from her aunt every 5 weeks -/
def aunt_sticks : ℕ := 35

/-- The number of weeks between aunt's shipments -/
def aunt_shipment_interval : ℕ := 5

/-- The percentage of the fort that is complete -/
def fort_completion : ℚ := 0.65

/-- The average number of sticks Felicity collects per week -/
noncomputable def avg_sticks_per_week : ℚ := avg_store_visits + (aunt_sticks : ℚ) / (aunt_shipment_interval : ℚ)

/-- The number of sticks Felicity has collected so far -/
noncomputable def collected_sticks : ℕ := Int.toNat ⌊(total_sticks : ℚ) * fort_completion⌋

theorem weeks_collecting_sticks : 
  ∃ (weeks : ℕ), weeks = 93 ∧ 
  (collected_sticks : ℚ) / avg_sticks_per_week ≤ weeks ∧ 
  weeks < (collected_sticks : ℚ) / avg_sticks_per_week + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weeks_collecting_sticks_l1030_103017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_with_even_product_l1030_103037

theorem max_odd_integers_with_even_product (integers : Finset ℕ) : 
  integers.card = 6 → 
  (integers.prod id) % 2 = 0 → 
  (integers.filter (λ x => x % 2 = 1)).card ≤ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_with_even_product_l1030_103037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l1030_103001

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ := by
  let interior_angle_sum : ℝ := 1080
  let num_sides : ℕ := 8
  let interior_angle : ℝ := interior_angle_sum / num_sides
  let exterior_angle : ℝ := 180 - interior_angle
  have h : exterior_angle = 45 := by sorry
  exact 45


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l1030_103001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisible_by_three_l1030_103036

theorem digit_sum_divisible_by_three (N : ℕ) 
  (digit_count : (N.repr).length = 1580)
  (digit_types : ∀ d, d ∈ (N.repr).data → d ∈ ['3', '5', '7'])
  (seven_count : ((N.repr).data.filter (· = '7')).length = 
                 ((N.repr).data.filter (· = '3')).length - 20) : 
  ((N.repr).data.map (λ c => c.toNat - '0'.toNat)).sum % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisible_by_three_l1030_103036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConsecutiveAfterOperation_l1030_103015

/-- Represents the operation of replacing a pair of integers with their difference and sum -/
def pairOperation (x y : ℤ) : (ℤ × ℤ) :=
  (x - y, x + y)

/-- Represents a sequence of 2n consecutive integers -/
def consecutiveIntegers (n : ℕ) (a : ℤ) : List ℤ :=
  List.range (2 * n) |>.map (λ i => a + i)

/-- Applies the pair operation once to a list of integers -/
def applyOperationOnce : List ℤ → List ℤ
  | [] => []
  | [x] => [x]
  | x :: y :: rest =>
      let (diff, sum) := pairOperation x y
      diff :: sum :: applyOperationOnce rest

/-- Applies the pair operation a given number of times to a list of integers -/
def applyOperations : ℕ → List ℤ → List ℤ
  | 0, xs => xs
  | n + 1, xs => applyOperations n (applyOperationOnce xs)

/-- Theorem stating that it's impossible to obtain 2n consecutive integers
    after applying the pair operation any number of times -/
theorem impossibleConsecutiveAfterOperation (n : ℕ) (a : ℤ) :
  ∀ (operations : ℕ), ¬∃ (b : ℤ),
    (consecutiveIntegers n a |>.map (λ x => x + b)) =
    (consecutiveIntegers n a |> applyOperations operations) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleConsecutiveAfterOperation_l1030_103015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_min_speed_to_break_record_l1030_103077

/-- Represents the relay race scenario -/
structure RelayRace where
  total_distance : ℝ
  record_time : ℝ
  first_runner_speed : ℝ

/-- Calculates the minimum speed required for the second runner to break the record -/
noncomputable def min_speed_to_break_record (race : RelayRace) : ℝ :=
  let first_runner_time := race.total_distance / race.first_runner_speed
  let remaining_time := race.record_time - first_runner_time
  race.total_distance / remaining_time

/-- Theorem stating the minimum speed required for Carlos to break the record -/
theorem carlos_min_speed_to_break_record (race : RelayRace) 
  (h1 : race.total_distance = 21)
  (h2 : race.record_time = 2 + 48 / 60)
  (h3 : race.first_runner_speed = 12) :
  min_speed_to_break_record race > 20 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_min_speed_to_break_record_l1030_103077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_APQ_l1030_103004

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given conditions
def is_on_segment (P Q R : Point) : Prop := sorry

def are_parallel (P Q R S : Point) : Prop := sorry

def is_inside_triangle (P A B C : Point) : Prop := sorry

def intersects_at (L₁ L₂ P Q R : Point) : Prop := sorry

def on_circumcircle (P A B C : Point) : Prop := sorry

-- State the theorem
theorem collinear_APQ 
  (ABC : Triangle) 
  (D E F G P Q : Point) :
  is_on_segment D ABC.A ABC.B →
  is_on_segment E ABC.A ABC.C →
  are_parallel D E ABC.B ABC.C →
  is_inside_triangle P ABC.A D E →
  intersects_at D E F ABC.B P →
  intersects_at D E G ABC.C P →
  on_circumcircle Q P D G →
  on_circumcircle Q P F E →
  Q ≠ P →
  (∃ R, on_circumcircle R P D G ∧ on_circumcircle R P F E ∧ R ≠ P ∧ R ≠ Q) →
  ∃ (t : ℝ), Q.x = ABC.A.x + t * (P.x - ABC.A.x) ∧ 
             Q.y = ABC.A.y + t * (P.y - ABC.A.y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_APQ_l1030_103004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_proof_l1030_103092

theorem equilateral_triangle_proof (A B C : Real) (a b c : Real) :
  A = π / 3 →
  Real.sin A ^ 2 = Real.sin B * Real.sin C →
  a = b ∧ b = c ∧ c = a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_proof_l1030_103092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1030_103076

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x * (x + 1)

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := (3 * x + 1) / (2 * Real.sqrt x)

-- Theorem statement
theorem tangent_angle_range (x : ℝ) (hx : x > 0) :
  let θ := Real.arctan (f' x)
  π / 3 ≤ θ ∧ θ < π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1030_103076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_and_log_conjunction_false_l1030_103074

theorem sin_and_log_conjunction_false :
  ¬(∃ (x₀ : ℝ), Real.sin x₀ ≥ 1 ∧ 
    ∀ (a b : ℝ), a > 0 → b > 0 → (Real.log a > Real.log b ↔ Real.sqrt a > Real.sqrt b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_and_log_conjunction_false_l1030_103074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_dig_time_l1030_103010

/-- Represents the digging scenario for Pirate Rick's treasure --/
structure DiggingScenario where
  initial_depth : ℝ  -- Initial depth of sand in feet
  initial_time : ℝ   -- Initial digging time in hours
  storm_factor : ℝ   -- Factor by which storm reduces sand (0.5 for half)
  tsunami_depth : ℝ  -- Depth of sand added by tsunami in feet

/-- Calculates the time required to dig up the treasure after the events --/
noncomputable def dig_time (scenario : DiggingScenario) : ℝ :=
  let digging_rate := scenario.initial_time / scenario.initial_depth
  let final_depth := scenario.initial_depth * scenario.storm_factor + scenario.tsunami_depth
  digging_rate * final_depth

/-- Theorem stating that the digging time for the given scenario is 3 hours --/
theorem pirate_rick_dig_time :
  let scenario := DiggingScenario.mk 8 4 0.5 2
  dig_time scenario = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_rick_dig_time_l1030_103010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1030_103097

theorem coefficient_x_cubed_in_expansion : ∃ (c : ℤ), c = 14 ∧ 
  (∀ (x : ℝ), (x - 1)^4 * (x - 2) = c * x^3 + x^4 * (x - 5) + x^2 * (10 - 2*x) + x * (-10 + x) + 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_cubed_in_expansion_l1030_103097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coughing_duration_l1030_103035

/-- Proves that Georgia and Robert have been coughing for 20 minutes -/
theorem coughing_duration (georgia_coughs_per_minute robert_coughs_ratio total_coughs : ℕ) 
  (h1 : georgia_coughs_per_minute = 5)
  (h2 : robert_coughs_ratio = 2)
  (h3 : total_coughs = 300) :
  20 = (total_coughs : ℚ) / (georgia_coughs_per_minute * (1 + robert_coughs_ratio)) := by
  sorry

#check coughing_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coughing_duration_l1030_103035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_director_circle_intersection_l1030_103095

/-- The ellipse x² + y²/3 = 1 -/
def myEllipse (x y : ℝ) : Prop := x^2 + y^2/3 = 1

/-- The circle (x-4)² + (y-3)² = r² -/
def myCircle (x y r : ℝ) : Prop := (x-4)^2 + (y-3)^2 = r^2

/-- The director circle of the ellipse -/
def directorCircle (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem ellipse_director_circle_intersection (r : ℝ) :
  (r > 0) →
  (∃ x y : ℝ, myEllipse x y ∧ myCircle x y r ∧ directorCircle x y) ↔ 
  (3 ≤ r ∧ r ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_director_circle_intersection_l1030_103095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_vote_percentage_l1030_103033

def votes : List Nat := [4136, 7636, 11628, 8735, 9917]

def total_votes : Nat := votes.sum

noncomputable def winning_votes : Nat :=
  match votes.maximum? with
  | some n => n
  | none => 0

noncomputable def winning_percentage : Float :=
  (winning_votes.toFloat / total_votes.toFloat) * 100

theorem winning_vote_percentage :
  (winning_percentage - 29.03).abs < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_vote_percentage_l1030_103033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_distance_ratio_l1030_103098

/-- Given five distinct points on a plane, the ratio of the maximum distance to the minimum distance
    between these points is greater than or equal to 2 sin 54°. -/
theorem five_points_distance_ratio (P : Fin 5 → ℝ × ℝ) (h : Function.Injective P) :
  let distances := {d | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (P i) (P j)}
  (⨆ d ∈ distances, d) / (⨅ d ∈ distances, d) ≥ 2 * Real.sin (54 * π / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_distance_ratio_l1030_103098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l1030_103013

/-- A function from natural numbers to natural numbers -/
def NatFunction := ℕ → ℕ

/-- The property that a function satisfies the given condition -/
def SatisfiesCondition (f : NatFunction) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → ∃ k : ℕ, a * (f a)^3 + 2 * a * b * (f a) + b * (f b) = k^2

/-- The identity function on natural numbers -/
def identityNat : NatFunction := λ n ↦ n

/-- The main theorem stating that the identity function is the only function satisfying the condition -/
theorem unique_satisfying_function :
  ∀ f : NatFunction, SatisfiesCondition f ↔ f = identityNat := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_satisfying_function_l1030_103013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_midpoint_trapezoid_area_formula_l1030_103061

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -x

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y - 1

-- Define the directrix (x = 1/4)
def directrix (x : ℝ) : Prop := x = 1/4

-- Define the midpoint of a segment
def segment_midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

-- Define the area of a trapezoid
noncomputable def trapezoid_area (a b h : ℝ) : ℝ := (a + b) * h / 2

-- Theorem 1
theorem line_equation_from_midpoint 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : segment_midpoint x1 y1 x2 y2 (-4) 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + (4 * k + 1) ∧ x + 2 * y + 2 = 0 :=
sorry

-- Theorem 2
theorem trapezoid_area_formula 
  (m x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line_l m x1 y1) 
  (h4 : line_l m x2 y2) :
  ∃ (A : ℝ), A = (2 * m^2 + 5) / 4 * Real.sqrt (m^2 + 4) ∧
    A = trapezoid_area (|x1 - 1/4| + |x2 - 1/4|) (|y1 - y2|) 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_midpoint_trapezoid_area_formula_l1030_103061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sequence_properties_l1030_103070

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_A : 0 < A
  pos_B : 0 < B
  pos_C : 0 < C
  sum_angles : A + B + C = π
  sine_rule_a : a / (Real.sin A) = b / (Real.sin B)
  sine_rule_c : c / (Real.sin C) = b / (Real.sin B)

/-- Arithmetic sequence property for side lengths -/
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Geometric sequence property for side lengths -/
def isGeometricSequence (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

theorem triangle_sequence_properties (t : Triangle) :
  (isArithmeticSequence t → Real.sin t.A + Real.sin t.C = 2 * Real.sin (t.A + t.C)) ∧
  (isGeometricSequence t → Real.cos t.B ≥ (1/2 : ℝ) ∧
    (Real.cos t.B = 1/2 ↔ t.a = t.c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sequence_properties_l1030_103070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_is_2_sqrt_3_l1030_103044

open Real

-- Define the parametric equations of the ellipse
noncomputable def x (t : ℝ) : ℝ := 2 * cos t
noncomputable def y (t : ℝ) : ℝ := 4 * sin t

-- Define the point M on the ellipse
noncomputable def M : ℝ × ℝ := (x (π/3), y (π/3))

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem to prove
theorem slope_of_OM_is_2_sqrt_3 :
  let slope := (M.2 - O.2) / (M.1 - O.1)
  slope = 2 * sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_OM_is_2_sqrt_3_l1030_103044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_watching_on_saturday_l1030_103060

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving Repr, Ord

-- Define the activities
inductive Activity : Type
  | Yoga | Painting | Cooking | BirdWatching | Cycling
  deriving Repr, Ord

-- Define a schedule as a function from Day to Activity
def Schedule := Day → Activity

-- Helper function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define the conditions of the problem
def ValidSchedule (s : Schedule) : Prop :=
  -- Different activity each day
  (∀ d1 d2 : Day, d1 ≠ d2 → s d1 ≠ s d2) ∧
  -- Yoga three days a week, not on consecutive days
  (∃ d1 d2 d3 : Day, s d1 = Activity.Yoga ∧ s d2 = Activity.Yoga ∧ s d3 = Activity.Yoga ∧
    d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
    (∀ d : Day, s d = Activity.Yoga → (d = d1 ∨ d = d2 ∨ d = d3)) ∧
    (∀ d : Day, s d = Activity.Yoga → s (nextDay d) ≠ Activity.Yoga)) ∧
  -- Painting on Monday
  (s Day.Monday = Activity.Painting) ∧
  -- Cooking two days after painting
  (s Day.Wednesday = Activity.Cooking) ∧
  -- No cycling after yoga or bird watching
  (∀ d : Day, (s d = Activity.Yoga ∨ s d = Activity.BirdWatching) → s (nextDay d) ≠ Activity.Cycling)

-- Theorem statement
theorem bird_watching_on_saturday (s : Schedule) (h : ValidSchedule s) :
  s Day.Saturday = Activity.BirdWatching :=
by sorry

#check bird_watching_on_saturday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_watching_on_saturday_l1030_103060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1030_103018

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of three spheres with radii 1, 4, and 6 -/
noncomputable def total_volume : ℝ := sphere_volume 1 + sphere_volume 4 + sphere_volume 6

theorem snowman_volume :
  total_volume = (1124 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snowman_volume_l1030_103018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_meet_l1030_103064

def horse_lap_times : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def is_at_start_line (time : Nat) (lap_time : Nat) : Bool :=
  time % lap_time = 0

def count_horses_at_start (time : Nat) (lap_times : List Nat) : Nat :=
  (lap_times.filter (λ lap_time => is_at_start_line time lap_time)).length

theorem least_time_six_horses_meet :
  ∃ (T : Nat), T > 0 ∧
  count_horses_at_start T horse_lap_times ≥ 6 ∧
  ∀ (t : Nat), t > 0 ∧ t < T → count_horses_at_start t horse_lap_times < 6 :=
by sorry

#eval count_horses_at_start 60 horse_lap_times

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_time_six_horses_meet_l1030_103064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_constant_l1030_103089

/-- Represents a circle with center (a-2, √3a) and radius 4 -/
def circle_equation (a x y : ℝ) : Prop :=
  (x - (a - 2))^2 + (y - Real.sqrt 3 * a)^2 = 16

/-- Represents a line passing through (1,0) with slope √3 -/
def line_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 1)

/-- The length of the chord intercepted by the line on the circle -/
noncomputable def chord_length (a : ℝ) : ℝ :=
  2 * Real.sqrt (16 - (3 * Real.sqrt 3 / 2)^2)

theorem chord_length_is_constant (a : ℝ) :
  chord_length a = Real.sqrt 37 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_constant_l1030_103089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_is_60_degrees_l1030_103023

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = 2

-- Define the line that P is on
def line_P (x y : ℝ) : Prop := x + y = 0

-- Define the line of symmetry
def line_symmetry (x y : ℝ) : Prop := y = -x

-- Define a point P
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_P x y

-- Define tangent points A and B
structure TangentPoint where
  x : ℝ
  y : ℝ
  on_circle : my_circle x y

-- Define the angle function (this is just a placeholder)
def angle_APB (P : Point_P) (A B : TangentPoint) : ℝ := sorry

-- Define the theorem
theorem tangent_angle_is_60_degrees 
  (P : Point_P) 
  (A B : TangentPoint) 
  (symmetric : ∃ (x y : ℝ), line_symmetry x y ∧ 
    (x - P.x) * (A.y - P.y) = (y - P.y) * (A.x - P.x) ∧
    (x - P.x) * (B.y - P.y) = (y - P.y) * (B.x - P.x)) :
  angle_APB P A B = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_is_60_degrees_l1030_103023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_properties_l1030_103026

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x - y + 2 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 2 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂ : ℝ, 
  l₁ m x₁ y₁ → l₂ m x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) + (x₁ - x₂) * (y₁ - y₂) = 0

-- Define the intersection point of l₁ and l₂
noncomputable def intersection (m : ℝ) : ℝ × ℝ := 
  ((2 * m - 2) / (m^2 + 1), (2 * m + 2) / (m^2 + 1))

-- Define the distance from a point to the origin
noncomputable def distance_to_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

theorem lines_properties :
  ∀ m : ℝ, 
    (perpendicular m) ∧ 
    (l₁ m 0 2) ∧ 
    (l₂ m (-2) 0) ∧ 
    (∀ m' : ℝ, distance_to_origin (intersection m') ≤ 2 * Real.sqrt 2) ∧
    (∃ m' : ℝ, distance_to_origin (intersection m') = 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_properties_l1030_103026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1030_103062

noncomputable section

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the slope angle
noncomputable def slope_angle : ℝ := 2 * Real.pi / 3

-- Define the circle equation
noncomputable def circle_equation (θ : ℝ) : ℝ := 2 * Real.cos (θ + Real.pi / 3)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (P.1 + t * Real.cos slope_angle, P.2 + t * Real.sin slope_angle)

-- Theorem statement
theorem intersection_product :
  ∃ (M N : ℝ × ℝ),
    (∃ (t₁ t₂ : ℝ), line_l t₁ = M ∧ line_l t₂ = N) ∧
    (∃ (θ₁ θ₂ : ℝ),
      Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) = circle_equation θ₁ ∧
      Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = circle_equation θ₂) →
    Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) *
    Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2) = 6 + 2 * Real.sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1030_103062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lithium_carbonate_price_proof_l1030_103040

/-- Represents a data point with x and y coordinates -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression function -/
def linearRegression (x : ℝ) : ℝ := 0.28 * x + 0.16

/-- Known data points -/
def knownDataPoints : List DataPoint := [
  ⟨1, 0.5⟩,
  ⟨3, 1⟩,
  ⟨4, 1.4⟩,
  ⟨5, 1.5⟩
]

/-- Calculate the average of a list of real numbers -/
noncomputable def average (list : List ℝ) : ℝ :=
  (list.sum) / (list.length : ℝ)

theorem lithium_carbonate_price_proof :
  let xValues := (knownDataPoints.map (λ p => p.x)) ++ [2]
  let yValues := knownDataPoints.map (λ p => p.y)
  let xMean := average xValues
  let yMean := (yValues.sum + 0.6) / 5
  xMean = 3 ∧ yMean = linearRegression xMean ∧
  ∃ a : ℝ, a = 0.6 ∧ linearRegression 2 = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lithium_carbonate_price_proof_l1030_103040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l1030_103083

/-- Represents a dwarf who either always tells the truth or always lies -/
inductive Dwarf
  | truthful
  | liar
deriving DecidableEq

/-- Represents the three types of ice cream -/
inductive IceCream
  | vanilla
  | chocolate
  | fruit
deriving DecidableEq

/-- The main theorem statement -/
theorem dwarf_truth_count 
  (dwarfs : Finset Dwarf) 
  (ice_cream_preference : Dwarf → IceCream)
  (h_count : dwarfs.card = 10)
  (h_unique_preference : ∀ d : Dwarf, d ∈ dwarfs → 
    (ice_cream_preference d = IceCream.vanilla) ∨ 
    (ice_cream_preference d = IceCream.chocolate) ∨ 
    (ice_cream_preference d = IceCream.fruit))
  (h_vanilla_raise : dwarfs.card = (dwarfs.filter (λ d => 
    (d = Dwarf.truthful ∧ ice_cream_preference d = IceCream.vanilla) ∨ 
    (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.vanilla))).card)
  (h_chocolate_raise : (dwarfs.filter (λ d => 
    (d = Dwarf.truthful ∧ ice_cream_preference d = IceCream.chocolate) ∨ 
    (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.chocolate))).card = 5)
  (h_fruit_raise : (dwarfs.filter (λ d => 
    (d = Dwarf.truthful ∧ ice_cream_preference d = IceCream.fruit) ∨ 
    (d = Dwarf.liar ∧ ice_cream_preference d ≠ IceCream.fruit))).card = 1)
  : (dwarfs.filter (λ d => d = Dwarf.truthful)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l1030_103083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_sales_l1030_103087

/-- The price of a spruce tree in Kč -/
def spruce_price : ℕ := 220

/-- The price of a pine tree in Kč -/
def pine_price : ℕ := 250

/-- The price of a fir tree in Kč -/
def fir_price : ℕ := 330

/-- The total earnings in Kč -/
def total_earnings : ℕ := 36000

/-- The number of each type of tree initially -/
def num_each_tree : ℕ := 45

theorem christmas_tree_sales : 
  spruce_price * num_each_tree + pine_price * num_each_tree + fir_price * num_each_tree = total_earnings ∧
  3 * num_each_tree = 135 := by
  sorry

#eval spruce_price * num_each_tree + pine_price * num_each_tree + fir_price * num_each_tree
#eval 3 * num_each_tree

end NUMINAMATH_CALUDE_ERRORFEEDBACK_christmas_tree_sales_l1030_103087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l1030_103009

noncomputable def sum_series : ℝ :=
  1 / (Real.sin (30 * Real.pi / 180) * Real.sin (31 * Real.pi / 180)) +
  1 / (Real.sin (32 * Real.pi / 180) * Real.sin (33 * Real.pi / 180)) +
  1 / (Real.sin (34 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) +
  1 / (Real.sin (36 * Real.pi / 180) * Real.sin (37 * Real.pi / 180)) +
  1 / (Real.sin (38 * Real.pi / 180) * Real.sin (39 * Real.pi / 180)) +
  1 / (Real.sin (40 * Real.pi / 180) * Real.sin (41 * Real.pi / 180)) +
  1 / (Real.sin (42 * Real.pi / 180) * Real.sin (43 * Real.pi / 180)) +
  1 / (Real.sin (44 * Real.pi / 180) * Real.sin (45 * Real.pi / 180)) +
  1 / (Real.sin (46 * Real.pi / 180) * Real.sin (47 * Real.pi / 180)) +
  1 / (Real.sin (48 * Real.pi / 180) * Real.sin (49 * Real.pi / 180)) +
  1 / (Real.sin (50 * Real.pi / 180) * Real.sin (51 * Real.pi / 180)) +
  1 / (Real.sin (52 * Real.pi / 180) * Real.sin (53 * Real.pi / 180)) +
  1 / (Real.sin (54 * Real.pi / 180) * Real.sin (55 * Real.pi / 180)) +
  1 / (Real.sin (56 * Real.pi / 180) * Real.sin (57 * Real.pi / 180)) +
  1 / (Real.sin (58 * Real.pi / 180) * Real.sin (59 * Real.pi / 180)) +
  1 / (Real.sin (60 * Real.pi / 180) * Real.sin (61 * Real.pi / 180)) +
  1 / (Real.sin (62 * Real.pi / 180) * Real.sin (63 * Real.pi / 180)) +
  1 / (Real.sin (64 * Real.pi / 180) * Real.sin (65 * Real.pi / 180)) +
  1 / (Real.sin (66 * Real.pi / 180) * Real.sin (67 * Real.pi / 180)) +
  1 / (Real.sin (68 * Real.pi / 180) * Real.sin (69 * Real.pi / 180)) +
  1 / (Real.sin (70 * Real.pi / 180) * Real.sin (71 * Real.pi / 180)) +
  1 / (Real.sin (72 * Real.pi / 180) * Real.sin (73 * Real.pi / 180)) +
  1 / (Real.sin (74 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) +
  1 / (Real.sin (76 * Real.pi / 180) * Real.sin (77 * Real.pi / 180)) +
  1 / (Real.sin (78 * Real.pi / 180) * Real.sin (79 * Real.pi / 180)) +
  1 / (Real.sin (80 * Real.pi / 180) * Real.sin (81 * Real.pi / 180)) +
  1 / (Real.sin (82 * Real.pi / 180) * Real.sin (83 * Real.pi / 180)) +
  1 / (Real.sin (84 * Real.pi / 180) * Real.sin (85 * Real.pi / 180)) +
  1 / (Real.sin (86 * Real.pi / 180) * Real.sin (87 * Real.pi / 180)) +
  1 / (Real.sin (88 * Real.pi / 180) * Real.sin (89 * Real.pi / 180)) +
  Real.cos (89 * Real.pi / 180)

theorem least_positive_integer_n : ∃! n : ℕ, n > 0 ∧ sum_series = 1 / Real.sin (n * Real.pi / 180) ∧ ∀ m : ℕ, m > 0 ∧ m < n → sum_series ≠ 1 / Real.sin (m * Real.pi / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_n_l1030_103009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_divisible_by_nine_l1030_103084

theorem perfect_squares_divisible_by_nine (a b c : ℤ) :
  (∃ (x y z : ℤ), a = x^2 ∧ b = y^2 ∧ c = z^2) →
  (a + b + c) % 9 = 0 →
  ∃ (p q : ℤ), p ∈ ({a, b, c} : Set ℤ) ∧ q ∈ ({a, b, c} : Set ℤ) ∧ p ≠ q ∧ (p - q) % 9 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_divisible_by_nine_l1030_103084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_sequence_ratio_l1030_103050

noncomputable def AlternatingGeometricSequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ :=
  fun n => a₁ * r^(n-1) * (-1)^(n-1)

noncomputable def CommonRatio (a₁ a₂ : ℝ) : ℝ := a₂ / a₁

theorem alternating_geometric_sequence_ratio :
  let a₁ := (10 : ℝ)
  let a₂ := (-15 : ℝ)
  let seq := AlternatingGeometricSequence a₁ (CommonRatio a₁ a₂)
  (∀ n : ℕ, n > 0 → seq n = a₁ * (CommonRatio a₁ a₂)^(n-1) * (-1)^(n-1)) →
  CommonRatio a₁ a₂ = -1.5 :=
by
  intros
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_geometric_sequence_ratio_l1030_103050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_rate_is_two_l1030_103058

/-- The walking problem setup -/
structure WalkingProblem where
  total_distance : ℚ
  yolanda_rate : ℚ
  bob_distance : ℚ
  time_difference : ℚ

/-- Calculate Bob's walking rate given the problem setup -/
def calculate_bob_rate (p : WalkingProblem) : ℚ :=
  let remaining_distance := p.total_distance - p.yolanda_rate * p.time_difference
  let yolanda_distance := remaining_distance - p.bob_distance
  let meeting_time := yolanda_distance / p.yolanda_rate
  p.bob_distance / meeting_time

/-- Theorem stating that Bob's walking rate is 2 miles per hour -/
theorem bob_rate_is_two (p : WalkingProblem) 
    (h1 : p.total_distance = 31)
    (h2 : p.yolanda_rate = 1)
    (h3 : p.bob_distance = 20)
    (h4 : p.time_difference = 1) : 
  calculate_bob_rate p = 2 := by
  sorry

#eval calculate_bob_rate { total_distance := 31, yolanda_rate := 1, bob_distance := 20, time_difference := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_rate_is_two_l1030_103058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_vectors_l1030_103039

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_vectors (a b c : V) (k : ℕ) (hk : k > 0) :
  ‖a‖ = 1 →
  ‖b‖ = k →
  ‖c‖ = 3 →
  b - a = (2 : ℝ) • (c - b) →
  inner a c / (‖a‖ * ‖c‖) = -1/12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_vectors_l1030_103039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bundle_radical_axis_relation_l1030_103031

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a bundle of circles -/
structure CircleBundle where
  circle1 : Circle
  circle2 : Circle

/-- Represents the possible relationships between a circle and the radical axis -/
inductive RadicalAxisRelation
  | Elliptic   -- Intersects at two fixed points
  | Parabolic  -- Touches at one fixed point
  | Hyperbolic -- Does not intersect

/-- The radical axis of two circles -/
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  sorry

/-- A circle from the bundle, parameterized by l and m -/
def bundleCircle (b : CircleBundle) (l m : ℝ) : Circle :=
  sorry

/-- The relation between a circle and the radical axis -/
def circleRadicalAxisRelation (c : Circle) (ra : Set (ℝ × ℝ)) : RadicalAxisRelation :=
  sorry

/-- Main theorem: Any circle in the bundle has one of three relationships with the radical axis -/
theorem circle_bundle_radical_axis_relation (b : CircleBundle) :
  ∀ (l m : ℝ), ∃ (r : RadicalAxisRelation),
    r = circleRadicalAxisRelation (bundleCircle b l m) (radicalAxis b.circle1 b.circle2) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bundle_radical_axis_relation_l1030_103031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_parallelogram_l1030_103054

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a Vector type
structure Vec where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define vector equality
def vector_eq (v1 v2 : Vec) : Prop :=
  v1.x = v2.x ∧ v1.y = v2.y ∧ v1.z = v2.z

-- Define parallelogram
def is_parallelogram (A B C D : Point) : Prop :=
  vector_eq (Vec.mk (B.x - A.x) (B.y - A.y) (B.z - A.z))
            (Vec.mk (C.x - D.x) (C.y - D.y) (C.z - D.z))

-- Define non-collinearity
def non_collinear (A B C D : Point) : Prop :=
  ¬(∃ (t : ℝ), Vec.mk (C.x - A.x) (C.y - A.y) (C.z - A.z) =
               Vec.mk (t * (B.x - A.x)) (t * (B.y - A.y)) (t * (B.z - A.z)))

-- Theorem statement
theorem vector_equality_implies_parallelogram 
  (A B C D : Point) 
  (h1 : non_collinear A B C D) 
  (h2 : vector_eq (Vec.mk (B.x - A.x) (B.y - A.y) (B.z - A.z))
                  (Vec.mk (C.x - D.x) (C.y - D.y) (C.z - D.z))) : 
  is_parallelogram A B C D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_parallelogram_l1030_103054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_decreases_l1030_103079

open Real

-- Define a minute in radians
noncomputable def minute : ℝ := π / (180 * 60)

-- Define the sine difference function
noncomputable def sineDifference (α : ℝ) : ℝ := sin (α + minute) - sin α

-- Theorem statement
theorem sine_difference_decreases (α β : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
  sineDifference β < sineDifference α := by
  sorry

-- Additional lemma to support the main theorem
lemma cos_decreases_on_first_quadrant (α β : ℝ)
  (h1 : 0 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  cos β < cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_difference_decreases_l1030_103079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1030_103080

theorem product_remainder (a b c : ℕ) : 
  a % 7 = 3 → b % 7 = 5 → c % 7 = 6 → (a * b * c) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1030_103080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_part_height_satisfies_ratio_lower_part_height_correct_l1030_103068

/-- The height of the lower part of a statue satisfying the golden ratio -/
noncomputable def lower_part_height (H : ℝ) : ℝ :=
  Real.sqrt 5 - 1

/-- Theorem stating that the lower part height satisfies the given ratio condition -/
theorem lower_part_height_satisfies_ratio (H : ℝ) (h : H = 2) :
  let L := lower_part_height H
  (H - L) / L = L / H :=
by sorry

/-- Theorem proving that the lower part height is correct for a 2m statue -/
theorem lower_part_height_correct :
  lower_part_height 2 = Real.sqrt 5 - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_part_height_satisfies_ratio_lower_part_height_correct_l1030_103068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1030_103000

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A B : Point) : 
  (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1) →  -- Hyperbola equation
  (distance A F₂)^2 + (distance A F₁)^2 = (distance F₁ F₂)^2 →  -- AF₂ ⟂ AF₁
  distance B F₂ = 2 * distance A F₁ →  -- |BF₂| = 2|AF₁|
  eccentricity h = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1030_103000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l1030_103090

theorem tan_double_angle_third_quadrant (α : ℝ) :
  Real.cos (π - α) = 4/5 →
  π < α ∧ α < 3*π/2 →
  Real.tan (2*α) = 24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_third_quadrant_l1030_103090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_average_speed_l1030_103057

/-- Represents the race conditions -/
structure RaceConditions where
  distance : ℝ
  time : ℝ
  elevationChanges : List ℝ
  windSpeed : ℝ
  maxSpeed : ℝ

/-- Calculates the average speed given race conditions -/
noncomputable def averageSpeed (conditions : RaceConditions) : ℝ :=
  conditions.distance / conditions.time

/-- Theorem stating that the average speed is 5 m/s for the given conditions -/
theorem race_average_speed (conditions : RaceConditions) 
  (h1 : conditions.distance = 200)
  (h2 : conditions.time = 40)
  (h3 : conditions.elevationChanges = [5, 0, -5])
  (h4 : conditions.windSpeed = 2)
  (h5 : conditions.maxSpeed = 6.5) :
  averageSpeed conditions = 5 := by
  unfold averageSpeed
  rw [h1, h2]
  norm_num

#check race_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_average_speed_l1030_103057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l1030_103072

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
axiom perpendicular : Line → Line → Prop
axiom parallel_lines : Line → Line → Prop
axiom parallel_planes : Plane → Plane → Prop
axiom line_parallel_to_plane : Line → Plane → Prop
axiom line_perpendicular_to_plane : Line → Plane → Prop
axiom planes_perpendicular : Plane → Plane → Prop
axiom angle_with_plane : Line → Plane → ℝ

-- Define the theorem
theorem all_statements_false 
  (m n : Line) 
  (α β γ : Plane) 
  (h_different : m ≠ n) 
  (h_non_coincident : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (¬ (perpendicular m n → parallel_planes α β → line_parallel_to_plane m α 
    → line_perpendicular_to_plane n β)) ∧
  (¬ (angle_with_plane m α = angle_with_plane n α → parallel_lines m n)) ∧
  (¬ (line_perpendicular_to_plane m α → perpendicular m n 
    → line_parallel_to_plane n α)) ∧
  (¬ (planes_perpendicular α γ → planes_perpendicular β γ 
    → planes_perpendicular α β)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_statements_false_l1030_103072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_independent_of_alpha_l1030_103038

theorem expression_independent_of_alpha (α : ℝ) 
  (h : ∀ n : ℤ, α ≠ π * n / 2 + π / 12) : 
  (1 - 2 * Real.sin (α - 3 * π / 2) ^ 2 + Real.sqrt 3 * Real.cos (2 * α + 3 * π / 2)) / 
  Real.sin (π / 6 - 2 * α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_independent_of_alpha_l1030_103038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mama_permutations_l1030_103006

def word : String := "MAMA"

def count_m : Nat := 2
def count_a : Nat := 2
def total_letters : Nat := 4

theorem mama_permutations :
  (Nat.factorial total_letters) / ((Nat.factorial count_m) * (Nat.factorial count_a)) = 6 := by
  simp [total_letters, count_m, count_a]
  norm_num
  rfl

#eval (Nat.factorial total_letters) / ((Nat.factorial count_m) * (Nat.factorial count_a))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mama_permutations_l1030_103006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_correct_l1030_103056

/-- The amount Berengere needs to contribute to buy a croissant -/
noncomputable def berengere_contribution (croissant_cost : ℝ) (lucas_money : ℝ) (exchange_rate : ℝ) : ℝ :=
  croissant_cost - lucas_money / exchange_rate

/-- Theorem stating the correct amount Berengere needs to contribute -/
theorem berengere_contribution_correct :
  berengere_contribution 8 10 1.5 = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berengere_contribution_correct_l1030_103056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1030_103047

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - a * Real.exp (-x))

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem tangent_slope_at_one (a : ℝ) :
  is_even (f a) → deriv (f a) 1 = 2 * Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l1030_103047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capricious_function_k_range_l1030_103022

/-- Definition of a capricious function -/
def isCapricious (f g h : ℝ → ℝ) (D : Set ℝ) :=
  ∀ x ∈ D, g x ≤ f x ∧ f x ≤ h x

/-- The interval [1, e] -/
def D : Set ℝ := {x | 1 ≤ x ∧ x ≤ Real.exp 1}

/-- The given functions -/
def f (k : ℝ) : ℝ → ℝ := fun x ↦ k * x
def g : ℝ → ℝ := fun x ↦ x^2 - 2*x
noncomputable def h : ℝ → ℝ := fun x ↦ (x + 1) * (Real.log x + 1)

/-- The theorem to prove -/
theorem capricious_function_k_range :
  {k : ℝ | isCapricious (f k) g h D} = {k | Real.exp 1 - 2 ≤ k ∧ k ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capricious_function_k_range_l1030_103022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_approx_l1030_103065

/-- Represents a segment of the train's journey -/
structure JourneySegment where
  distance : ℝ  -- Distance in km
  time : ℝ      -- Time in hours

/-- Calculates the average speed for the entire journey -/
noncomputable def average_speed (segments : List JourneySegment) : ℝ :=
  let total_distance := segments.foldl (λ acc seg => acc + seg.distance) 0
  let total_time := segments.foldl (λ acc seg => acc + seg.time) 0
  total_distance / total_time

/-- The train's journey segments -/
def train_journey : List JourneySegment := [
  ⟨290, 4.5⟩,
  ⟨400, 5.5⟩,
  ⟨350, 7⟩,
  ⟨480, 6⟩
]

theorem train_average_speed_approx :
  abs (average_speed train_journey - 66.09) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval average_speed train_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_average_speed_approx_l1030_103065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1030_103042

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp x

-- State the theorem
theorem problem_statement :
  ∀ a b c : ℝ,
  (f a b c 0 = 1) →
  (f a b c 1 = 0) →
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a b c x ≥ f a b c y) →
  (0 ≤ a ∧ a ≤ 1) ∧
  (∃ m : ℝ, m = 4 ∧
    ∀ x : ℝ, 2 * (f 0 b c x) + 4 * x * Real.exp x ≥ m * x + 1 ∧
              m * x + 1 ≥ -x^2 + 4 * x + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1030_103042
