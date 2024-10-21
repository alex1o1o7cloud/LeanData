import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l1279_127936

theorem tan_half_sum (p q : ℝ) 
  (h1 : Real.cos p + Real.cos q = 1) 
  (h2 : Real.sin p + Real.sin q = 1/2) : 
  Real.tan ((p + q)/2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l1279_127936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_period_of_f_l1279_127957

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 4)

theorem minimal_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_period_of_f_l1279_127957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_satisfies_conditions_l1279_127934

theorem no_integer_polynomial_satisfies_conditions : 
  ¬ ∃ (P : Polynomial ℤ), (P.eval 7 = 5) ∧ (P.eval 15 = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_satisfies_conditions_l1279_127934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1279_127966

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := 2 * y^2 = 2 * x + 3

/-- The line equation -/
def line (x y : ℝ) : Prop := y - x * Real.sqrt 3 + 3 = 0

/-- Point P -/
noncomputable def P : ℝ × ℝ := (Real.sqrt 3, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_difference :
  ∀ (A B : ℝ × ℝ),
  parabola A.1 A.2 → parabola B.1 B.2 →
  line A.1 A.2 → line B.1 B.2 →
  A ≠ B →
  |distance A P - distance B P| = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l1279_127966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1279_127943

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f2x1 : Set ℝ := Set.Ioo (-2) (1/2)

-- Theorem statement
theorem domain_of_f : 
  (∀ y ∈ domain_f2x1, ∃ x, y = 2*x + 1) → 
  (∀ x, x ∈ Set.Ioo (-3) 2 ↔ f x ≠ 0) :=
by
  sorry

-- Example to show the relationship between the domains
example : Set.image (fun x ↦ 2*x + 1) domain_f2x1 = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1279_127943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bounds_l1279_127981

/-- The eccentricity of a hyperbola with given parameters satisfies the specified bounds -/
theorem hyperbola_eccentricity_bounds (a : ℝ) (h1 : a > 0) : 
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2 / (a^2 + 1) - y^2 = 1
  let A : ℝ × ℝ := (Real.sqrt (a^2 + 1), 0)
  let O : ℝ × ℝ := (0, 0)
  let e := Real.sqrt (1 + 1 / (a^2 + 1))
  (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) < 2) →
  (Real.sqrt 5 / 2 < e ∧ e < Real.sqrt 2)
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_bounds_l1279_127981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_projection_parallelogram_projection_square_projection_rhombus_projection_oblique_projection_properties_l1279_127906

/-- Represents a shape in 2D space -/
inductive Shape
| Triangle
| Parallelogram
| Square
| Rhombus

/-- Represents the result of an oblique projection -/
def ObliqueProjection (s : Shape) : Shape := 
  match s with
  | Shape.Triangle => Shape.Triangle
  | Shape.Parallelogram => Shape.Parallelogram
  | Shape.Square => Shape.Parallelogram
  | Shape.Rhombus => Shape.Parallelogram

/-- The oblique projection of a triangle is a triangle -/
theorem triangle_projection : ObliqueProjection Shape.Triangle = Shape.Triangle := 
  rfl

/-- The oblique projection of a parallelogram is a parallelogram -/
theorem parallelogram_projection : ObliqueProjection Shape.Parallelogram = Shape.Parallelogram := 
  rfl

/-- The oblique projection of a square is not necessarily a square -/
theorem square_projection : ObliqueProjection Shape.Square ≠ Shape.Square := 
  fun h => Shape.noConfusion h

/-- The oblique projection of a rhombus is not necessarily a rhombus -/
theorem rhombus_projection : ObliqueProjection Shape.Rhombus ≠ Shape.Rhombus := 
  fun h => Shape.noConfusion h

theorem oblique_projection_properties :
  (ObliqueProjection Shape.Triangle = Shape.Triangle) ∧
  (ObliqueProjection Shape.Parallelogram = Shape.Parallelogram) ∧
  (ObliqueProjection Shape.Square ≠ Shape.Square) ∧
  (ObliqueProjection Shape.Rhombus ≠ Shape.Rhombus) := by
  constructor
  · exact triangle_projection
  constructor
  · exact parallelogram_projection
  constructor
  · exact square_projection
  · exact rhombus_projection

#check oblique_projection_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_projection_parallelogram_projection_square_projection_rhombus_projection_oblique_projection_properties_l1279_127906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janna_sleep_hours_l1279_127951

/-- Proves that Janna sleeps 7 hours each weekday given the conditions -/
theorem janna_sleep_hours
  (weekday_sleep : ℕ)
  (weekend_sleep : ℕ)
  (weekdays : ℕ)
  (weekend_days : ℕ)
  (total_sleep : ℕ)
  (h1 : weekend_sleep = 8)
  (h2 : weekdays = 5)
  (h3 : weekend_days = 2)
  (h4 : total_sleep = 51)
  (h5 : total_sleep = weekday_sleep * weekdays + weekend_sleep * weekend_days) :
  weekday_sleep = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janna_sleep_hours_l1279_127951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_geq_one_l1279_127908

open Real Set

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a vector in a plane -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- The line e -/
def Line : Set Point := sorry

/-- The plane containing the vectors -/
def Plane : Set Point := sorry

/-- A function to check if a point is on the same side of a line as another point -/
def sameSideOfLine (l : Set Point) (p q : Point) : Prop := sorry

/-- A function to calculate the magnitude of a vector -/
def magnitude (v : PlaneVector) : ℝ := sorry

/-- A function to add vectors -/
def addVectors (vs : List PlaneVector) : PlaneVector := sorry

/-- The main theorem -/
theorem vector_sum_magnitude_geq_one 
  (O : Point) 
  (n : ℕ) 
  (vectors : List PlaneVector) 
  (h_O_on_e : O ∈ Line)
  (h_n_odd : Odd n)
  (h_n_vectors : vectors.length = n)
  (h_unit_vectors : ∀ v ∈ vectors, magnitude v = 1)
  (h_same_plane : ∀ v ∈ vectors, ∃ P ∈ Plane, v = PlaneVector.mk (P.x - O.x) (P.y - O.y))
  (h_same_side : ∀ v₁ v₂, v₁ ∈ vectors → v₂ ∈ vectors → 
    ∀ P₁ P₂, P₁ ∈ Plane → P₂ ∈ Plane → 
    v₁ = PlaneVector.mk (P₁.x - O.x) (P₁.y - O.y) → 
    v₂ = PlaneVector.mk (P₂.x - O.x) (P₂.y - O.y) → 
    sameSideOfLine Line P₁ P₂) :
  magnitude (addVectors vectors) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_geq_one_l1279_127908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_53_l1279_127960

theorem remainder_sum_mod_53 (a b c d : ℕ) 
  (ha : a % 53 = 33)
  (hb : b % 53 = 26)
  (hc : c % 53 = 18)
  (hd : d % 53 = 6) :
  (2 * a + 2 * b + 2 * c + 2 * d) % 53 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_53_l1279_127960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_in_square_with_semicircles_l1279_127964

/-- The radius of the circle tangent to all semicircles in a square configuration -/
noncomputable def inner_circle_radius (square_side : ℝ) (num_semicircles : ℕ) : ℝ :=
  (9 * Real.sqrt 5) / 10

/-- Theorem stating the radius of the inner circle in the given configuration -/
theorem inner_circle_radius_in_square_with_semicircles 
  (square_side : ℝ) (num_semicircles : ℕ) 
  (h1 : square_side = 4) 
  (h2 : num_semicircles = 16) : 
  inner_circle_radius square_side num_semicircles = (9 * Real.sqrt 5) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_circle_radius_in_square_with_semicircles_l1279_127964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1279_127947

/-- Calculates the annual interest rate for compound interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (total_interest : ℝ) (years : ℕ) : ℝ :=
  let total_amount := principal + total_interest
  let nth_root := (total_amount / principal) ^ (1 / (years : ℝ))
  nth_root - 1

/-- The calculated interest rate is approximately 0.0574 (5.74%) -/
theorem interest_rate_calculation (principal : ℝ) (total_interest : ℝ) (years : ℕ) 
  (h1 : principal = 5000)
  (h2 : total_interest = 2850)
  (h3 : years = 8) :
  |calculate_interest_rate principal total_interest years - 0.0574| < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1279_127947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_unique_solution_l1279_127933

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line x + ay = 1 - a -/
noncomputable def slope1 (a : ℝ) : ℝ := -1 / a

/-- The slope of the second line (a - 2)x + 3y + 2 = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := (a - 2) / 3

theorem perpendicular_lines_unique_solution :
  ∃! a : ℝ, a ≠ 0 ∧ are_perpendicular (slope1 a) (slope2 a) ∧ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_unique_solution_l1279_127933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1279_127956

/-- Represents a game state with two piles of matches -/
structure GameState :=
  (pile1 : ℕ)
  (pile2 : ℕ)

/-- Predicate to check if a move is valid -/
def validMove (state : GameState) (move : ℕ × ℕ) : Prop :=
  (move.1 = 1 ∧ move.2 ∣ state.pile2) ∨
  (move.1 = 2 ∧ move.2 ∣ state.pile1)

/-- Predicate to check if a game state is winning for the current player -/
def isWinning : GameState → Prop :=
  sorry -- Definition to be implemented

/-- The initial game state -/
def initialState : GameState :=
  ⟨100, 252⟩

/-- Theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ¬ isWinning initialState :=
by
  sorry -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l1279_127956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1279_127941

/-- The asymptote equations for the hyperbola (x^2 / 6) - (y^2 / 3) = 1 are y = ± (√2 / 2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 6) - (y^2 / 3) = 1 →
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1279_127941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1279_127983

/-- The perimeter of the shaded region formed by four identical circles -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (num_circles : ℕ) 
  (shaded_region_perimeter : ℝ) : 
  circle_circumference = 72 →
  num_circles = 4 →
  shaded_region_perimeter = num_circles * (circle_circumference / 4) →
  shaded_region_perimeter = 72 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_region_perimeter_l1279_127983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1279_127929

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (0, 2)

/-- Distance function between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to the y-axis -/
def distToYAxis (p : ℝ × ℝ) : ℝ := abs p.1

/-- The sum of distances for a point P -/
noncomputable def sumOfDistances (p : ℝ × ℝ) : ℝ :=
  distance p A + distToYAxis p

/-- Theorem: The minimum sum of distances for points on the parabola -/
theorem min_sum_distances :
  ∃ (m : ℝ), m = Real.sqrt 5 - 1 ∧
  ∀ (p : ℝ × ℝ), p ∈ Parabola → sumOfDistances p ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1279_127929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_minimum_value_l1279_127997

-- Define the function A
noncomputable def A (a : ℝ) : ℝ := a + 1 / (a + 2)

-- State the theorem
theorem A_minimum_value (a : ℝ) (h : a > -2) :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > -2, A x ≥ min :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_minimum_value_l1279_127997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_cos_le_exp_l1279_127903

theorem negation_of_exists_cos_le_exp :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ Real.cos x ≤ Real.exp x) ↔ (∀ x : ℝ, x ≥ 0 → Real.cos x > Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_cos_le_exp_l1279_127903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_x_l1279_127905

def f (x : ℤ) : ℕ := Int.natAbs (10 * x^2 - 61 * x + 21)

def is_smallest_prime (x : ℤ) : Prop :=
  (∀ y : ℤ, y < x → ¬ Nat.Prime (f y)) ∧ Nat.Prime (f x)

theorem smallest_prime_x :
  is_smallest_prime 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_x_l1279_127905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1279_127932

-- Define the function f as noncomputable due to its dependence on Real.sqrt
noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 3 * Real.tan x) * Real.cos x

-- State the theorem
theorem f_max_value :
  ∃ (M : ℝ), M = 2 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x < Real.pi / 2 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x < Real.pi / 2 ∧ f x = M) :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1279_127932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_probability_l1279_127962

-- Define the dice roll
def DiceRoll : Type := Fin 6

-- Define the hyperbola
def Hyperbola (a b : ℕ+) : Type := {p : ℝ × ℝ | p.1^2 / a.val^2 - p.2^2 / b.val^2 = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℕ+) : ℝ := Real.sqrt (a.val^2 + b.val^2) / a.val

-- Define the condition for eccentricity > √5
def eccentricityCondition (a b : ℕ+) : Prop := eccentricity a b > Real.sqrt 5

-- Define the probability space
def ProbabilitySpace : Type := DiceRoll × DiceRoll

-- Define the probability measure
noncomputable def prob : Set ProbabilitySpace → ℝ := fun _ => 1 / 36

-- Theorem statement
theorem eccentricity_probability : 
  prob {p : ProbabilitySpace | eccentricityCondition ⟨p.1.val + 1, Nat.succ_pos _⟩ ⟨p.2.val + 1, Nat.succ_pos _⟩} = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_probability_l1279_127962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1279_127976

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2:ℝ)^x < 1/2 ∨ x^2 > x) ↔ (∀ x : ℝ, (2:ℝ)^x ≥ 1/2 ∧ x^2 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1279_127976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_correlation_pairs_l1279_127913

-- Define the types for our variables
structure Car where
  weight : ℝ
  fuelEfficiency : ℝ
  fuelConsumption : ℝ

structure Student where
  studyTime : ℝ
  academicPerformance : ℝ

structure Person where
  smokingHabit : ℝ
  healthCondition : ℝ

structure Square where
  sideLength : ℝ

-- Define correlation function
noncomputable def correlation (x y : ℝ → ℝ) : ℝ := sorry

-- Define negative correlation
def negativelyCorrelated (x y : ℝ → ℝ) : Prop :=
  correlation x y < 0

-- Theorem statement
theorem negative_correlation_pairs :
  ∃ (f₁ f₂ : ℝ → ℝ) 
    (g₁ g₂ : ℝ → ℝ)
    (h₁ h₂ : ℝ → ℝ)
    (i₁ i₂ : ℝ → ℝ)
    (j₁ j₂ : ℝ → ℝ),
  negativelyCorrelated f₁ f₂ ∧
  negativelyCorrelated h₁ h₂ ∧
  ¬negativelyCorrelated g₁ g₂ ∧
  ¬negativelyCorrelated i₁ i₂ ∧
  ¬negativelyCorrelated j₁ j₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_correlation_pairs_l1279_127913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_overtaking_l1279_127901

/-- The speed increase required for car A to safely overtake car B -/
noncomputable def speed_increase : ℝ := 5

/-- The initial speed of car A in mph -/
noncomputable def initial_speed_A : ℝ := 50

/-- The speed of car B in mph -/
noncomputable def speed_B : ℝ := 40

/-- The speed of car C in mph -/
noncomputable def speed_C : ℝ := 50

/-- The distance between car A and car C in feet -/
noncomputable def distance_AC : ℝ := 210

/-- The safe distance to maintain after overtaking in feet -/
noncomputable def safe_distance : ℝ := 30

/-- Conversion factor from feet to miles -/
noncomputable def feet_to_miles : ℝ := 1 / 5280

theorem safe_overtaking :
  ∃ (d : ℝ), 
    d > 0 ∧ 
    d / (initial_speed_A + speed_increase) = (d - safe_distance * feet_to_miles) / speed_B ∧
    d / (initial_speed_A + speed_increase) = (distance_AC * feet_to_miles - d) / speed_C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_overtaking_l1279_127901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_decreasing_l1279_127900

-- Define the four functions
noncomputable def f1 (x : ℝ) : ℝ := 6 * x
noncomputable def f2 (x : ℝ) : ℝ := -6 * x
noncomputable def f3 (x : ℝ) : ℝ := 6 / x
noncomputable def f4 (x : ℝ) : ℝ := -6 / x

-- Define the property of decreasing as x increases
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x2 < f x1

-- Theorem statement
theorem only_f2_decreasing :
  is_decreasing f2 ∧
  ¬is_decreasing f1 ∧
  ¬is_decreasing f3 ∧
  ¬is_decreasing f4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_decreasing_l1279_127900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l1279_127977

-- Define the condition p
noncomputable def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + m > 0

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  -1/3 * x^3 - 2 * x^2 - m*x - 1

-- Define the condition q
noncomputable def condition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f m x > f m y

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, condition_p m → condition_q m) ∧
  (∃ m : ℝ, condition_q m ∧ ¬condition_p m) := by
  sorry

#check p_sufficient_not_necessary_for_q

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_sufficient_not_necessary_for_q_l1279_127977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f3_is_expected_f4_is_expected_l1279_127996

/-- A function f is an "expected function" if there exists a constant k such that 
    |f(x)| ≤ (k/2017)|x| for all real x -/
def is_expected_function (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, |f x| ≤ (k / 2017) * |x|

/-- Function 3: f(x) = x / (x^2 - x + 1) -/
noncomputable def f3 (x : ℝ) : ℝ := x / (x^2 - x + 1)

/-- Function 4: f(x) = x / (e^x + 1) -/
noncomputable def f4 (x : ℝ) : ℝ := x / (Real.exp x + 1)

/-- Theorem: f3 is an expected function -/
theorem f3_is_expected : is_expected_function f3 := by
  sorry

/-- Theorem: f4 is an expected function -/
theorem f4_is_expected : is_expected_function f4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f3_is_expected_f4_is_expected_l1279_127996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l1279_127986

/-- An ellipse with left focus at (-2, 0) and a point E(0, 3/2) on PF₂ -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b
  focus_left : ((-2 : ℝ), (0 : ℝ)) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  point_E : ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ (0, (3/2 : ℝ)) ∈ Set.range (λ t => ((1-t)*x + t*2, (1-t)*y))

/-- The eccentricity, standard equation, and slope property of the special ellipse -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∃ (P Q : ℝ × ℝ), P ∈ {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧ 
                     Q ∈ {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} ∧
                     P.1 = Q.1 ∧ P.1 = -2) →
  (∃ (c : ℝ), c^2 = e.a^2 - e.b^2 ∧ c / e.a = (1/2 : ℝ)) ∧
  (e.a^2 = 16 ∧ e.b^2 = 12) ∧
  (∀ (A B P : ℝ × ℝ), A ∈ {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} →
                     B ∈ {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} →
                     P ∈ {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1} →
                     (A.2 - P.2) / (A.1 - P.1) = -(B.2 - P.2) / (B.1 - P.1) →
                     (B.2 - A.2) / (B.1 - A.1) = -(1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_ellipse_properties_l1279_127986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banker_true_discount_l1279_127953

/-- Calculates the true discount given banker's gain, interest rate, and time period. -/
noncomputable def true_discount (bg : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  (bg * 100) / (r * t)

/-- Theorem: Given a banker's gain of 6.6, an interest rate of 12% per annum, 
    and a time period of 1 year, the true discount is 55. -/
theorem banker_true_discount :
  let bg : ℝ := 6.6
  let r : ℝ := 12
  let t : ℝ := 1
  true_discount bg r t = 55 := by
  -- Unfold the definition of true_discount
  unfold true_discount
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banker_true_discount_l1279_127953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_cells_theorem_l1279_127907

/-- Represents the color of a cell -/
inductive Color
| Blue
| Red

/-- Represents a row of cells -/
def Row := List Color

/-- Function to determine if Arepito can reach the end given a row configuration -/
def canReachEnd (row : Row) : Bool :=
  sorry

/-- The minimum number of blue cells required for Arepito to reach the end -/
def minBlueCells (n : Nat) : Nat :=
  Nat.ceil ((n + 1) / 2)

/-- Theorem stating the minimum number of blue cells required -/
theorem min_blue_cells_theorem (n : Nat) (h : n > 2) :
  minBlueCells n = Nat.ceil ((n + 1) / 2) := by
  rfl

#check min_blue_cells_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_cells_theorem_l1279_127907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l1279_127967

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := Real.exp x - y = 0

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    (y = m * (x - 1) + f 1) ↔ tangent_line x y :=
by
  -- Introduce the slope (m) and y-intercept (b)
  let m := Real.exp 1
  let b := Real.exp 1
  
  -- Show that these values satisfy the theorem
  use m, b
  intro x y
  
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x_1_l1279_127967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_nonnegative_dot_product_l1279_127915

theorem at_least_one_nonnegative_dot_product 
  (a b c d e f g h : ℝ) : 
  ∃ x ∈ ({a*c + b*d, a*e + b*f, a*g + b*h, c*e + d*f, c*g + d*h, e*g + f*h} : Set ℝ), x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_nonnegative_dot_product_l1279_127915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1279_127975

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1279_127975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1279_127926

/-- The equation of the moving line l -/
def line_equation (lambda x y : ℝ) : Prop :=
  (3*lambda + 1)*x + (1 - lambda)*y + 6 - 6*lambda = 0

/-- The fixed point P through which the line always passes -/
def fixed_point : ℝ × ℝ := (0, -6)

/-- The condition for the line to intersect the positive half of the x-axis -/
def intersects_positive_x_axis (lambda : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ line_equation lambda x 0

/-- The theorem stating the properties of the line and lambda -/
theorem line_properties :
  (∀ lambda : ℝ, line_equation lambda fixed_point.1 fixed_point.2) ∧
  (∀ lambda : ℝ, intersects_positive_x_axis lambda ↔ (lambda > 1 ∨ lambda < -1/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1279_127926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_equals_triangle_area_l1279_127952

/-- Triangle with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Area of a triangle using Heron's formula -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Centroid of a triangle -/
def Triangle.centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Rotation of a point around another point -/
def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Area of the union of a triangle and its 180° rotation around its centroid -/
noncomputable def unionArea (t : Triangle) : ℝ := sorry

theorem union_area_equals_triangle_area (t : Triangle) : 
  t.a = 12 ∧ t.b = 13 ∧ t.c = 15 → unionArea t = t.area := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_equals_triangle_area_l1279_127952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1279_127954

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 r2 θ1 θ2 : Real) : Real :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

/-- Theorem: The distance between points A(2, π/6) and B(2, -π/6) in polar coordinates is 2 -/
theorem distance_between_polar_points :
  polar_distance 2 2 (π/6) (-π/6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1279_127954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dad_transport_mode_l1279_127912

/-- Represents different modes of transportation -/
inductive TransportMode
  | Walking
  | Bicycle
  | Car
deriving Repr

/-- Determines the most likely mode of transportation based on average speed -/
def mostLikelyTransportMode (speed : ℚ) : TransportMode :=
  if speed ≤ 5 then TransportMode.Walking
  else if speed ≤ 20 then TransportMode.Bicycle
  else TransportMode.Car

theorem dad_transport_mode (distance : ℚ) (time : ℚ) (h1 : distance = 60) (h2 : time = 3) :
  mostLikelyTransportMode (distance / time) = TransportMode.Car := by
  sorry

#eval mostLikelyTransportMode (60 / 3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dad_transport_mode_l1279_127912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_preference_l1279_127991

theorem cake_preference (total : ℕ) (ice_cream_percent : ℚ) (cake_percent : ℚ)
  (ice_cream_count : ℕ) (h1 : ice_cream_percent = 2/5)
  (h2 : cake_percent = 3/10) (h3 : ice_cream_count = 80)
  (h4 : ↑ice_cream_count = ice_cream_percent * ↑total) :
  (cake_percent * ↑total).num = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_preference_l1279_127991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_81n4_l1279_127902

theorem divisors_of_81n4 (n : ℕ+) (h : (Nat.divisors (110 * n.val ^ 3)).card = 110) :
  (Nat.divisors (81 * n.val ^ 4)).card = 325 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_81n4_l1279_127902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_properties_l1279_127949

-- Define the function f(x) as noncomputable due to Real.log
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

-- State the theorem
theorem function_max_min_properties (a b c : ℝ) (ha : a ≠ 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ 
    (∀ x, x > 0 → f a b c x ≤ f a b c x₁) ∧ 
    (∀ x, x > 0 → f a b c x ≥ f a b c x₂)) :
  a * b > 0 ∧ b^2 + 8*a*c > 0 ∧ a*c < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_properties_l1279_127949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_property_l1279_127994

open Real

-- Define the points in the Euclidean plane
variable (A B C D P : EuclideanSpace ℝ (Fin 2))

-- Define the angles and distances
variable (angle : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → ℝ)
variable (dist : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2) → ℝ)

-- Define the conditions
variable (h1 : angle D A C = π / 2)
variable (h2 : 2 * angle A D B = angle A C B)
variable (h3 : angle D B C + 2 * angle A D C = π)

-- Define that ABCD is a convex quadrilateral
variable (h4 : IsConvexQuadrilateral A B C D)

-- Define that P is the intersection of diagonals
variable (h5 : P ∈ SegmentOpen A C ∩ SegmentOpen B D)

-- State the theorem
theorem diagonal_intersection_property :
  2 * (dist A P) = dist B P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_intersection_property_l1279_127994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l1279_127910

theorem angle_trigonometry (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 4) = Real.sqrt 5 / 5) : 
  Real.tan (α + π / 4) = 2 ∧ 
  Real.sin (2 * α + π / 3) = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l1279_127910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_t_lower_bound_l1279_127988

-- Define the functions f and g
noncomputable def f (t x : ℝ) : ℝ := (x - 1) * Real.exp x - (t / 2) * x^2 - 2 * x
noncomputable def g (t x : ℝ) : ℝ := Real.exp x - 2 / x - t

-- State the theorem
theorem prove_t_lower_bound (t : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∀ x, f t x = 0 → x = x₁ ∨ x = x₂) →
  g t x₁ = 0 →
  f t x₁ + 5 / (2 * Real.exp 1) - 1 < 0 →
  t > 2 + 1 / Real.exp 1 := by
  sorry

#check prove_t_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_t_lower_bound_l1279_127988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_tourists_speed_l1279_127919

/-- Given two car tourists traveling from A to B, prove their average speeds. -/
theorem car_tourists_speed
  (s : ℝ) -- distance between A and B in km
  (n : ℝ) -- time difference for first tourist in hours
  (r : ℝ) -- speed difference between tourists in km/h
  (h₁ : s > 0) -- distance is positive
  (h₂ : n > 0) -- time difference is positive
  (h₃ : r > 0) -- speed difference is positive
  : ∃ (v₁ v₂ : ℝ),
    v₁ > 0 ∧ v₂ > 0 ∧  -- speeds are positive
    v₁ - v₂ = r ∧      -- speed difference
    s / v₁ = s / v₂ - 4 * n ∧  -- time difference equation
    v₁ = (r + Real.sqrt (n * r * (n * r + s))) / (2 * n) ∧
    v₂ = (-r + Real.sqrt (n * r * (n * r + s))) / (2 * n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_tourists_speed_l1279_127919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_negative_implies_cos_two_alpha_negative_l1279_127920

theorem tan_alpha_plus_pi_fourth_negative_implies_cos_two_alpha_negative
  (α : ℝ)
  (h : Real.tan (α + π / 4) < 0) :
  Real.cos (2 * α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_negative_implies_cos_two_alpha_negative_l1279_127920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1279_127985

open Real

noncomputable def f (x : ℝ) : ℝ := 
  1 / (x + 2) + (1 / 2) * log (x^2 + 4) + arctan (x / 2)

noncomputable def g (x : ℝ) : ℝ := 
  (x^3 + 5*x^2 + 12*x + 4) / ((x + 2)^2 * (x^2 + 4))

theorem integral_proof (x : ℝ) : deriv f x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1279_127985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_four_l1279_127917

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the function g
def g : ℝ → ℝ := sorry

-- f(2x+2)-1 is an odd function
axiom f_odd (x : ℝ) : f (2*x + 2) - 1 = -(f (-2*x + 2) - 1)

-- g is symmetric to f about the line x-y=0
axiom g_sym_f (x : ℝ) : g x = f x

-- Theorem to prove
theorem sum_g_equals_four (x₁ x₂ : ℝ) (h : x₁ + x₂ = 2) : g x₁ + g x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_four_l1279_127917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1279_127999

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y = 0

-- Define the centers and radii of the circles
def center_C₁ : ℝ × ℝ := (0, 0)
def radius_C₁ : ℝ := 2
def center_C₂ : ℝ × ℝ := (-3, 2)
noncomputable def radius_C₂ : ℝ := Real.sqrt 13

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

-- Theorem statement
theorem circles_intersect :
  distance_between_centers > abs (radius_C₁ - radius_C₂) ∧
  distance_between_centers < radius_C₁ + radius_C₂ :=
by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1279_127999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_speed_in_chase_l1279_127998

/-- The speed of the thief in a chase scenario -/
noncomputable def thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ) : ℝ :=
  (policeman_speed * thief_distance) / (initial_distance + thief_distance)

/-- Theorem stating the speed of the thief given the problem conditions -/
theorem thief_speed_in_chase :
  let initial_distance : ℝ := 225 / 1000  -- Convert to km
  let policeman_speed : ℝ := 10
  let thief_distance : ℝ := 900 / 1000  -- Convert to km
  thief_speed initial_distance policeman_speed thief_distance = 8 := by
  -- Unfold the definitions
  unfold thief_speed
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_speed_in_chase_l1279_127998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_interval_count_l1279_127965

theorem sqrt_interval_count :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (16.2 : ℝ) < Real.sqrt n ∧ Real.sqrt n < (16.3 : ℝ)) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_interval_count_l1279_127965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1279_127911

theorem product_of_roots (R : ℝ) : 
  ∀ N₁ N₂ : ℝ, N₁ - 5 / N₁ = R → N₂ - 5 / N₂ = R → N₁ * N₂ = -5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l1279_127911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_first_friend_is_120_l1279_127940

/-- The distance between Tony's town and his first friend's town -/
def distance_to_first_friend : ℝ := sorry

/-- The travel time between Tony's town and his first friend's town -/
def time_to_first_friend : ℝ := 3

/-- The distance to the second friend's town -/
def distance_to_second_friend : ℝ := 200

/-- The travel time to the second friend's town -/
def time_to_second_friend : ℝ := 5

/-- Tony's driving speed -/
noncomputable def driving_speed : ℝ := distance_to_second_friend / time_to_second_friend

theorem distance_to_first_friend_is_120 :
  distance_to_first_friend = 120 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_first_friend_is_120_l1279_127940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_diagonal_property_l1279_127961

/-- Represents a cyclic quadrilateral ABCD with diagonal intersection point M -/
structure CyclicQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  M : ℝ × ℝ
  is_cyclic : Bool
  is_diagonal_intersection : Bool

/-- The length of a line segment between two points -/
def length (p q : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_diagonal_property
  (ABCD : CyclicQuadrilateral)
  (h1 : length ABCD.A ABCD.B = 2)
  (h2 : length ABCD.B ABCD.C = 5)
  (h3 : length ABCD.A ABCD.M = 4)
  (h4 : (length ABCD.C ABCD.D) / (length ABCD.C ABCD.M) = 0.6) :
  length ABCD.A ABCD.D = 2 := by
  sorry

#check cyclic_quadrilateral_diagonal_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_diagonal_property_l1279_127961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_with_tens_digit_eight_l1279_127937

theorem three_digit_number_with_tens_digit_eight (m : ℕ) : 
  (100 ≤ m ∧ m < 1000) →  -- m is a three-digit number
  (m / 10 % 10 = 8) →     -- tens digit of m is 8
  (∃ n : ℕ, m - 40 * n = 24) →  -- m - 40n = 24 for some natural n
  (∃! (list : List ℕ), list.length = 2 ∧ ∀ x ∈ list, 
    (100 ≤ x ∧ x < 1000) ∧ 
    (x / 10 % 10 = 8) ∧ 
    (∃ n : ℕ, x - 40 * n = 24)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_with_tens_digit_eight_l1279_127937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l1279_127904

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define the base case for 0
  | 1 => 1
  | (n + 2) => sequence_a (n + 1) / (3 * sequence_a (n + 1) + 1)

theorem a_2009_value : sequence_a 2009 = 1 / 6025 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2009_value_l1279_127904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l1279_127927

/-- A polynomial in z -/
def MyPolynomial (α : Type*) [Semiring α] := α → ℕ → α

/-- The degree of a polynomial -/
noncomputable def degree {α : Type*} [Semiring α] (p : MyPolynomial α) : ℕ := sorry

/-- Evaluation of a polynomial at a point -/
def eval {α : Type*} [Semiring α] (p : MyPolynomial α) (x : α) : α := sorry

/-- Addition of polynomials -/
def add {α : Type*} [Semiring α] (p q : MyPolynomial α) : MyPolynomial α := sorry

/-- Multiplication of polynomials -/
def mul {α : Type*} [Semiring α] (p q : MyPolynomial α) : MyPolynomial α := sorry

/-- Constant polynomial -/
def constant {α : Type*} [Semiring α] (c : α) : MyPolynomial α := sorry

/-- Power of z -/
def power {α : Type*} [Semiring α] (n : ℕ) : MyPolynomial α := sorry

variable {α : Type*} [Field α]

/-- The main theorem -/
theorem polynomial_equation 
  (S T : MyPolynomial α) 
  (h1 : add (power 2022) (constant 1) = 
        add (mul (add (add (power 2) (power 1)) (constant 1)) S) T)
  (h2 : degree T < 2) :
  T = constant 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equation_l1279_127927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_in_family_album_l1279_127973

/-- Represents a photo with three people -/
structure Photo where
  middle : ℕ
  right : ℕ
  left : ℕ

/-- The minimum number of people in the family album -/
def min_people (photos : List Photo) : ℕ :=
  16

theorem min_people_in_family_album 
  (photos : List Photo) 
  (h1 : photos.length = 10)
  (h2 : ∀ p ∈ photos, p.right ≠ p.middle ∧ p.left ≠ p.middle)
  (h3 : ∀ p ∈ photos, ∃ q ∈ photos, p.right = q.middle)
  (h4 : ∀ p ∈ photos, ∃ q ∈ photos, p.left = q.middle)
  (h5 : ∀ p q, p ∈ photos → q ∈ photos → p ≠ q → p.middle ≠ q.middle) :
  min_people photos = 16 := by
  sorry

#check min_people_in_family_album

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_people_in_family_album_l1279_127973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_l1279_127970

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (4 * x + φ)

-- State the theorem
theorem f_pi_third (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π / 2) (h3 : f φ (π / 12) = 3) :
  f φ (π / 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_third_l1279_127970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1279_127923

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the circle
def circleQ (Q : ℝ × ℝ) : Prop := Q.1^2 + (Q.2 - 4)^2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem min_distance_sum (P Q : ℝ × ℝ) (hP : parabola P) (hQ : circleQ Q) :
  ∃ (min : ℝ), ∀ (P' Q' : ℝ × ℝ), parabola P' → circleQ Q' →
    distance P' Q' + distance P' focus ≥ min ∧
    min = Real.sqrt 17 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1279_127923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1279_127942

theorem trig_identity (α : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos (Real.pi - α) = (6 - Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1279_127942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_increase_good_air_quality_days_l1279_127935

theorem min_increase_good_air_quality_days 
  (total_days : ℕ) 
  (last_year_ratio : ℚ) 
  (next_year_ratio : ℚ) 
  (h1 : total_days = 365) 
  (h2 : last_year_ratio = 60 / 100) 
  (h3 : next_year_ratio > 70 / 100) : 
  ∃ (min_increase : ℕ), 
    min_increase ≥ 37 ∧ 
    ∀ (increase : ℕ), 
      (increase + (last_year_ratio * ↑total_days : ℚ)) / ↑total_days > next_year_ratio → 
      increase ≥ min_increase := by
  sorry

#check min_increase_good_air_quality_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_increase_good_air_quality_days_l1279_127935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_bound_l1279_127930

/-- Represents the state of the board after some operations -/
structure BoardState where
  numbers : List ℚ
  deriving Repr

/-- Perform one operation on the board -/
def performOperation (state : BoardState) : BoardState :=
  sorry

/-- Perform n-1 operations on a board with n ones -/
def finalState (n : ℕ) : BoardState :=
  sorry

/-- The theorem to be proved -/
theorem final_number_bound (n : ℕ) (h : n > 0) :
  ∀ x ∈ (finalState n).numbers, x ≥ 1 / n :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_bound_l1279_127930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_amount_is_120_20_l1279_127928

/-- Calculates the net amount received after selling a stock and deducting brokerage -/
def net_amount_after_brokerage (cash_realized : ℚ) (brokerage_rate : ℚ) : ℚ :=
  let brokerage := (brokerage_rate / 100) * cash_realized
  cash_realized - (Rat.floor (brokerage * 100) / 100)

/-- Theorem stating that for the given conditions, the net amount is 120.20 -/
theorem net_amount_is_120_20 :
  net_amount_after_brokerage 120.50 (1/4) = 120.20 := by
  sorry

#eval net_amount_after_brokerage 120.50 (1/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_amount_is_120_20_l1279_127928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_zero_l1279_127918

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_angle_at_zero :
  Real.arctan (Real.exp 0 * (Real.sin 0 + Real.cos 0)) = π / 4 := by
  have h1 : Real.exp 0 = 1 := by simp
  have h2 : Real.sin 0 = 0 := by simp
  have h3 : Real.cos 0 = 1 := by simp
  have h4 : 1 * (0 + 1) = 1 := by simp
  have h5 : Real.arctan 1 = π / 4 := by sorry
  simp [h1, h2, h3, h4, h5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_at_zero_l1279_127918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1279_127995

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)) →
  t.B = 2 * Real.pi / 3 ∧
  (t.B = 2 * Real.pi / 3 ∧ t.a + t.c = 2 ∧ (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 4) →
  t.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1279_127995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1279_127982

/-- The area of a quadrilateral given its diagonal and offsets -/
noncomputable def quadrilateralArea (d h₁ h₂ : ℝ) : ℝ := (1/2) * d * (h₁ + h₂)

/-- Theorem: The area of a quadrilateral with diagonal 24 cm and offsets 9 cm and 6 cm is 180 cm² -/
theorem quadrilateral_area_example : quadrilateralArea 24 9 6 = 180 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_example_l1279_127982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_bottle_ounces_l1279_127948

/-- Calculates the number of ounces in a bottle of sunscreen given the following conditions:
  * Reapplication interval is 2 hours
  * Amount needed per application is 3 ounces
  * Total beach time is 16 hours
  * Cost of sunscreen is $7
  * Cost per bottle is $3.5
-/
theorem sunscreen_bottle_ounces : ℝ := by
  let reapplication_interval : ℝ := 2
  let amount_per_application : ℝ := 3
  let total_beach_time : ℝ := 16
  let total_cost : ℝ := 7
  let cost_per_bottle : ℝ := 3.5

  let num_applications : ℝ := total_beach_time / reapplication_interval
  let total_ounces_needed : ℝ := num_applications * amount_per_application
  let num_bottles : ℝ := total_cost / cost_per_bottle
  let ounces_per_bottle : ℝ := total_ounces_needed / num_bottles

  have h : ounces_per_bottle = 12 := by
    -- Proof steps would go here
    sorry

  exact 12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunscreen_bottle_ounces_l1279_127948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rook_free_square_l1279_127972

/-- A peaceful configuration of rooks on an n × n chessboard. -/
def PeacefulRooks (n : ℕ) := Fin n → Fin n

/-- Checks if a configuration of rooks is peaceful. -/
def isPeaceful (n : ℕ) (config : PeacefulRooks n) : Prop :=
  Function.Injective config ∧ Function.Surjective config

/-- Checks if a k × k square is rook-free in a given configuration. -/
def isRookFreeSquare (n k : ℕ) (config : PeacefulRooks n) (i j : Fin (n - k + 1)) : Prop :=
  ∀ x y : Fin k, config ⟨(x : ℕ) + (i : ℕ), sorry⟩ ≠ ⟨(y : ℕ) + (j : ℕ), sorry⟩

/-- The main theorem stating the largest guaranteed rook-free square size. -/
theorem largest_rook_free_square (n : ℕ) (h : n ≥ 2) :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (config : PeacefulRooks n), isPeaceful n config →
      ∃ (i j : Fin (n - k + 1)), isRookFreeSquare n k config i j) ∧
    (∀ (k' : ℕ), k' > k →
      ∃ (config : PeacefulRooks n), isPeaceful n config ∧
        ∀ (i j : Fin (n - k' + 1)), ¬isRookFreeSquare n k' config i j)) ∧
  (∀ (k : ℕ), 
    (∀ (config : PeacefulRooks n), isPeaceful n config →
      ∃ (i j : Fin (n - k + 1)), isRookFreeSquare n k config i j) ∧
    (∀ (k' : ℕ), k' > k →
      ∃ (config : PeacefulRooks n), isPeaceful n config ∧
        ∀ (i j : Fin (n - k' + 1)), ¬isRookFreeSquare n k' config i j) →
    k = Int.ceil (Real.sqrt (n : ℝ)) - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rook_free_square_l1279_127972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_condition_for_a_l1279_127959

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Statement 1: Tangent line at x=0
theorem tangent_line_at_zero :
  ∃ (m b : ℝ), ∀ x, m * x + b = (deriv f) 0 * x + f 0 ∧ m = 1 ∧ b = 2 := by
  sorry

-- Statement 2: Condition for a
theorem condition_for_a :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f x ≤ a * x + 2) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_condition_for_a_l1279_127959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l1279_127914

theorem triangle_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B)
  (h_given : a * Real.cos B = b * Real.cos A) :
  a = b := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_l1279_127914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_l1279_127924

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ  -- First term
  diff : ℝ   -- Common difference

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (seq.first + (n - 1) / 2 * seq.diff)

/-- Sum of S_k for k from 1 to n -/
noncomputable def T (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (n + 1) / 6 * (3 * seq.first + (n - 1) * seq.diff)

/-- The theorem stating that 3034 is the smallest n for which T_n is uniquely determined -/
theorem unique_determination_of_T (seq : ArithmeticSequence) :
  (∃ (x : ℝ), S seq 2023 = x) →
  (∀ (m : ℕ), m < 3034 → ¬∃! (y : ℝ), T seq m = y) ∧
  (∃! (y : ℝ), T seq 3034 = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_determination_of_T_l1279_127924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l1279_127921

/-- Represents a digit (1-9) -/
def Digit : Type := { n : Nat // 1 ≤ n ∧ n ≤ 9 }

/-- Converts a list of four digits to a natural number -/
def fourDigitToNat (a b c d : Digit) : Nat :=
  1000 * a.val + 100 * b.val + 10 * c.val + d.val

/-- Converts two digits to a natural number -/
def twoDigitToNat (c d : Digit) : Nat :=
  10 * c.val + d.val

/-- The main theorem stating that 2916 is the unique solution to the rebus -/
theorem rebus_solution :
  ∃! (a b c d : Digit),
    a.val ≠ b.val ∧ a.val ≠ c.val ∧ a.val ≠ d.val ∧
    b.val ≠ c.val ∧ b.val ≠ d.val ∧
    c.val ≠ d.val ∧
    fourDigitToNat a b c a = 182 * twoDigitToNat c d ∧
    fourDigitToNat a b c d = 2916 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rebus_solution_l1279_127921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l1279_127925

/-- Represents a circular keystone arch constructed with congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  angle_at_center : ℚ
  larger_interior_angle : ℚ

/-- The properties of the keystone arch in the problem -/
def problem_arch : KeystoneArch where
  num_trapezoids := 12
  angle_at_center := 360 / 12
  larger_interior_angle := 195 / 2

/-- Theorem stating that the larger interior angle of a trapezoid in the given keystone arch is 97.5 degrees -/
theorem keystone_arch_angle (arch : KeystoneArch) 
  (h1 : arch.num_trapezoids = 12)
  (h2 : arch.angle_at_center = 360 / arch.num_trapezoids)
  (h3 : ∀ θ : ℚ, θ = (180 - arch.angle_at_center / 2) / 2 → 
             arch.larger_interior_angle = 180 - θ) :
  arch.larger_interior_angle = 195 / 2 := by
  sorry

#eval problem_arch.larger_interior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_keystone_arch_angle_l1279_127925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_distance_difference_l1279_127950

/-- Represents a parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola -/
noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- The intersection of the directrix with the x-axis -/
noncomputable def directrix_x_intersection (c : Parabola) : Point :=
  { x := -c.p / 2, y := 0 }

/-- Represents a line passing through the focus and intersecting the parabola -/
structure FocusLine (c : Parabola) where
  A : Point
  B : Point
  h_on_parabola : A.y^2 = 2 * c.p * A.x ∧ B.y^2 = 2 * c.p * B.x
  h_through_focus : ∃ t : ℝ, A.x = (focus c).x + t * (B.x - (focus c).x) ∧
                              A.y = (focus c).y + t * (B.y - (focus c).y)

/-- The theorem to be proved -/
theorem parabola_focus_line_distance_difference (c : Parabola) (l : FocusLine c) :
  let F := focus c
  let Q := directrix_x_intersection c
  (l.B.x - Q.x)^2 + (l.B.y - Q.y)^2 = (l.B.x - F.x)^2 + (l.B.y - F.y)^2 →
  (Real.sqrt ((l.A.x - F.x)^2 + (l.A.y - F.y)^2) -
   Real.sqrt ((l.B.x - F.x)^2 + (l.B.y - F.y)^2)) = 2 * c.p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_line_distance_difference_l1279_127950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_c_properties_l1279_127980

/-- Definition of the ellipse E -/
noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of a point on the ellipse -/
noncomputable def point_on_ellipse (x y a b : ℝ) : Prop :=
  ellipse x y a b

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Theorem: Properties of the ellipse and point C -/
theorem ellipse_and_point_c_properties :
  ∀ a b : ℝ,
  ellipse 2 3 a b →
  eccentricity a b = 1/2 →
  ∃ x y : ℝ,
    a^2 = 16 ∧
    b^2 = 12 ∧
    point_on_ellipse x y 4 (Real.sqrt 12) ∧
    x = -16 * Real.sqrt 19 / 19 ∧
    y = 6 * Real.sqrt 19 / 19 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_point_c_properties_l1279_127980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_floor_fraction_l1279_127931

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- Last two digits of an integer -/
def lastTwoDigits (n : ℤ) : ℕ := n.natAbs % 100

theorem last_two_digits_of_floor_fraction : 
  lastTwoDigits (floor (10^93 / (10^31 + 3))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_digits_of_floor_fraction_l1279_127931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_l1279_127939

/-- The number of distinct arrangements of n distinct beads on a bracelet,
    considering rotational and reflectional symmetry -/
def braceletArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem eight_bead_bracelet :
  braceletArrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_bead_bracelet_l1279_127939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1279_127945

def a : ℝ × ℝ := (3, -1)
def b (lambda : ℝ) : ℝ × ℝ := (1, lambda)

theorem vector_equality (lambda : ℝ) : 
  (a.1 - (b lambda).1)^2 + (a.2 - (b lambda).2)^2 = a.1^2 + a.2^2 + (b lambda).1^2 + (b lambda).2^2 → lambda = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l1279_127945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_in_triangle_l1279_127909

theorem sin_cos_relation_in_triangle (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A < Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_in_triangle_l1279_127909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_covered_area_theorem_l1279_127993

/-- The largest area of an equilateral triangle that can be covered by three unit equilateral triangles -/
noncomputable def largest_covered_area : ℝ := 9 * Real.sqrt 3 / 16

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : ℝ × ℝ
  side_length : ℝ

/-- Checks if one triangle covers another -/
def covers_triangle (t1 t2 : EquilateralTriangle) : Prop := sorry

/-- Three unit equilateral triangles can cover an equilateral triangle -/
axiom can_cover (side_length : ℝ) : 
  side_length ≤ 3 / 2 → ∃ (t1 t2 t3 : EquilateralTriangle), 
    t1.side_length = 1 ∧ 
    t2.side_length = 1 ∧ 
    t3.side_length = 1 ∧
    covers_triangle (EquilateralTriangle.mk (0, 0) side_length) 
      (EquilateralTriangle.mk t1.center 1)

/-- The area of an equilateral triangle with side length a -/
noncomputable def equilateral_triangle_area (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2

/-- Theorem: The largest area of an equilateral triangle that can be covered by three equilateral triangles with side length 1 is 9√3/16 -/
theorem largest_covered_area_theorem : 
  ∀ (side_length : ℝ), 
    (∃ (t1 t2 t3 : EquilateralTriangle), 
      t1.side_length = 1 ∧ 
      t2.side_length = 1 ∧ 
      t3.side_length = 1 ∧
      covers_triangle (EquilateralTriangle.mk (0, 0) side_length) 
        (EquilateralTriangle.mk t1.center 1)) →
    equilateral_triangle_area side_length ≤ largest_covered_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_covered_area_theorem_l1279_127993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1279_127987

theorem remainder_problem (x y : ℤ) (r : ℕ) 
  (h1 : x % 9 = r)
  (h2 : (2 * x) % 7 = 1)
  (h3 : 5 * y - x = 3) :
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1279_127987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l1279_127984

noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

noncomputable def a (n : ℕ) : ℝ := φ ^ n

theorem unique_sequence :
  (∀ n, a n > 0) ∧
  a 0 = 1 ∧
  (∀ n, a n - a (n + 1) = a (n + 2)) ∧
  (∀ b : ℕ → ℝ, (∀ n, b n > 0) → b 0 = 1 → (∀ n, b n - b (n + 1) = b (n + 2)) → b = a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_l1279_127984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1279_127958

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := sorry

/-- The focal points of an ellipse -/
noncomputable def focal_points (e : Ellipse) : ℝ × ℝ := sorry

/-- A circle with diameter equal to the distance between focal points -/
def circle_from_focal_points (e : Ellipse) : Set (ℝ × ℝ) := sorry

/-- A point on the ellipse in the first quadrant -/
noncomputable def intersection_point (e : Ellipse) : ℝ × ℝ := sorry

/-- The slope of a line passing through two points -/
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem ellipse_eccentricity (e : Ellipse) :
  let F1F2 := focal_points e
  let P := intersection_point e
  let O : ℝ × ℝ := ((F1F2.1 + F1F2.2) / 2, 0)  -- midpoint of F1 and F2
  line_slope O P = Real.sqrt 3 →
  eccentricity e = Real.sqrt 3 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1279_127958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l1279_127944

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (6 + x - x^2) / Real.log (1/2)

-- Define the domain
def domain : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}

-- State the theorem
theorem monotonic_increasing_interval :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 3 ∧
  (∀ (x : ℝ), x ∈ domain → a < x → x < b → ∀ (y : ℝ), y ∈ domain → x < y → f x < f y) ∧
  (∀ (x y : ℝ), x ∈ domain → y ∈ domain → x < y → f x < f y → a ≤ x ∧ y ≤ b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l1279_127944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_open_one_two_A_union_B_equals_A_l1279_127989

-- Define the function f
noncomputable def f (x : ℝ) := Real.sqrt (2 - (x + 6) / (x + 2))

-- Define the domain A of f
def A : Set ℝ := {x | x < -2 ∨ x ≥ 2}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | x^2 - (m + 3) * x + 3 * m < 0}

-- Theorem 1
theorem complement_A_intersect_B_equals_open_one_two (m : ℝ) :
  (Set.compl A) ∩ (B m) = Set.Ioo 1 2 → m = 1 := by sorry

-- Theorem 2
theorem A_union_B_equals_A (m : ℝ) :
  A ∪ (B m) = A → m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_equals_open_one_two_A_union_B_equals_A_l1279_127989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_perimeter_range_l1279_127978

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  sine_law : a / Real.sin A = b / Real.sin B
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * (Real.cos t.C) * (t.a * (Real.cos t.B) + t.b * (Real.cos t.A)) = t.c

theorem angle_C_value (t : Triangle) (h : given_condition t) : t.C = π/3 := by
  sorry

theorem perimeter_range (t : Triangle) (h : given_condition t) (h_c : t.c = Real.sqrt 3) :
  2 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_perimeter_range_l1279_127978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_inequality_l1279_127963

/-- A triangle with its orthocenter, incenter, and circumcenter -/
structure TriangleWithCenters where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  O : ℝ × ℝ

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- A triangle is equilateral if all its sides have equal length -/
def isEquilateral (t : TriangleWithCenters) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

theorem triangle_centers_inequality (t : TriangleWithCenters) :
  2 * distance t.I t.O ≥ distance t.I t.H ∧
  (2 * distance t.I t.O = distance t.I t.H ↔ isEquilateral t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centers_inequality_l1279_127963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_234_78_base4_l1279_127938

/-- Converts a number from base 4 to base 10 -/
def base4ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def decimalToBase4 (n : ℕ) : ℕ := sorry

/-- The sum of two numbers in base 4 -/
def sumBase4 (a b : ℕ) : ℕ :=
  decimalToBase4 (base4ToDecimal a + base4ToDecimal b)

theorem sum_234_78_base4 :
  sumBase4 234 78 = 13020 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_234_78_base4_l1279_127938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_lower_bound_l1279_127974

/-- The prime-counting function -/
noncomputable def prime_counting (x : ℕ) : ℕ := sorry

/-- The set of odd positive integers less than 30m which are not multiples of 5 -/
def S (m : ℕ) : Set ℕ :=
  {n : ℕ | n < 30 * m ∧ n % 2 = 1 ∧ n % 5 ≠ 0}

/-- The property that one number divides another -/
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_k_lower_bound (m : ℕ) :
  ∃ k : ℕ, ∀ T : Finset ℕ, (↑T : Set ℕ) ⊆ S m → T.card = k →
    (∃ a b : ℕ, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ divides a b) →
  k ≥ prime_counting (30 * m) - prime_counting (6 * m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_lower_bound_l1279_127974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1279_127916

/-- The distance from a point (x, y) to a line ax + by + c = 0 is |ax + by + c| / √(a² + b²) -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
def is_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_tangent_to_line :
  let center_x := 2
  let center_y := -1
  let line_a := 3
  let line_b := -4
  let line_c := 5
  let radius := distance_point_to_line center_x center_y line_a line_b line_c
  ∀ x y : ℝ, is_circle_equation x y center_x center_y radius ↔ 
    (x - center_x)^2 + (y - center_y)^2 = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l1279_127916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1279_127979

/-- Given a hyperbola x^2 + my^2 = m with focal length 4, its asymptotes are y = ± (√3/3)x -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = m) →  -- Hyperbola equation
  (∃ a b c : ℝ, a^2 = -m ∧ b^2 = 1 ∧ c = 2) →  -- Focal length condition
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ x y : ℝ, (y = k*x ∨ y = -k*x) → (x^2 + m*y^2 = m)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l1279_127979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1279_127969

theorem max_product_sum (a b c d e : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (a * b + b * c + c * d + d * e + e * a) ≤ 47 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l1279_127969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_in_square_l1279_127955

/-- Given a square ABCD with side length a and an equilateral triangle ADE inscribed inside it,
    the ratio of the area of triangle ADE to the area of triangle DEC is √3. -/
theorem area_ratio_equilateral_in_square (a : ℝ) (h : a > 0) :
  let square := {A : ℝ × ℝ | 0 ≤ A.1 ∧ A.1 ≤ a ∧ 0 ≤ A.2 ∧ A.2 ≤ a}
  let A := (0, 0)
  let B := (a, 0)
  let C := (a, a)
  let D := (0, a)
  let E := (a, a / 2)
  let triangle_ADE := {P : ℝ × ℝ | P ∈ square ∧ (∃ t u : ℝ, 0 ≤ t ∧ 0 ≤ u ∧ t + u ≤ 1 ∧ P = (t * a, u * a))}
  let triangle_DEC := {P : ℝ × ℝ | P ∈ square ∧ (∃ t u : ℝ, 0 ≤ t ∧ 0 ≤ u ∧ t + u ≤ 1 ∧ P = ((1 - t) * a, (1 - u) * a))}
  let area_ADE := Real.sqrt 3 / 4 * a^2
  let area_DEC := 1 / 4 * a^2
  area_ADE / area_DEC = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_in_square_l1279_127955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_restoration_l1279_127990

theorem wage_restoration (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.7 * original_wage
  let required_raise := (original_wage / reduced_wage - 1) * 100
  ∃ ε > 0, |required_raise - 42.86| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_restoration_l1279_127990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_general_term_formula_l1279_127946

noncomputable def sequence_a (n : ℕ+) : ℝ := 2 * n.val - 1

noncomputable def S (n : ℕ+) : ℝ := (n.val * (2 * n.val - 1)) / 2

theorem sequence_property (n : ℕ+) :
  ∀ k : ℕ+, k ≤ n → sequence_a k > 0 ∧ (sequence_a k)^2 = 4 * S k - 2 * sequence_a k - 1 :=
by sorry

theorem general_term_formula :
  ∀ n : ℕ+, sequence_a n = 2 * n.val - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_general_term_formula_l1279_127946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_minus_x_l1279_127968

theorem definite_integral_sqrt_minus_x (f g : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (1 - (x - 1)^2)) →
  (∀ x, g x = x) →
  (∫ x in Set.Icc 0 1, f x - g x) = π / 4 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_minus_x_l1279_127968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_abs_reciprocal_neg_one_point_five_l1279_127922

theorem opposite_abs_reciprocal_neg_one_point_five :
  -(|(1 : ℚ) / (-3/2)|) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_abs_reciprocal_neg_one_point_five_l1279_127922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1279_127971

-- Define the given parameters
noncomputable def train_length : ℝ := 120
noncomputable def bridge_length : ℝ := 150
noncomputable def train_speed_kmph : ℝ := 36

-- Define the conversion factor from kmph to m/s
noncomputable def kmph_to_ms : ℝ := 1000 / 3600

-- Theorem statement
theorem train_crossing_time :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmph * kmph_to_ms
  let crossing_time := total_distance / train_speed_ms
  crossing_time = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1279_127971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_sqrt3_l1279_127992

open Real

-- Define the expression
noncomputable def trigExpression : ℝ := (2 * cos (10 * π / 180) - cos (70 * π / 180)) / cos (20 * π / 180)

-- State the theorem
theorem trigExpression_equals_sqrt3 : trigExpression = sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_sqrt3_l1279_127992
