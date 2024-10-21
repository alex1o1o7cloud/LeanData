import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_45_seconds_l450_45063

/-- The time taken for a train to completely cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that the train crossing time is approximately 45 seconds -/
theorem train_crossing_approx_45_seconds :
  ∃ ε > 0, |train_crossing_time 250 300 44 - 45| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_45_seconds_l450_45063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_angle_theorem_l450_45072

-- Define the vectors
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define points A and F
def A : ℝ × ℝ := (-1, 0)
def F : ℝ × ℝ := (2, 0)

-- Define vectors a and b
def a (x y : ℝ) : ℝ × ℝ := ((x + 2), y)
def b (x y : ℝ) : ℝ × ℝ := ((x - 2), y)

-- Define the magnitude difference condition
def magnitude_diff (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) - Real.sqrt ((x - 2)^2 + y^2) = 2

-- Define the locus equation
def locus (x y : ℝ) : Prop :=
  x^2 - y^2/3 = 1 ∧ x > 0

-- Define the angle function (this is a placeholder, as Lean doesn't have a built-in angle function)
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

-- Define the angle condition
def angle_condition (P : ℝ × ℝ) : Prop :=
  ∃ (lambda : ℝ), lambda = 2 ∧ 
    angle P F A = lambda * angle P A F

-- Main theorem
theorem locus_and_angle_theorem :
  ∀ (x y : ℝ), magnitude_diff x y → locus x y ∧ angle_condition (x, y) := by
  sorry

#check locus_and_angle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_angle_theorem_l450_45072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l450_45050

-- Define the ellipse E
structure Ellipse where
  foci : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)
  eccentricity : ℝ

-- Define a line
structure Line where
  point : EuclideanSpace ℝ (Fin 2)
  slope : ℝ

-- Define the triangle PF₁F₂
structure RightTriangle where
  vertices : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)

-- Define the problem setup
def ellipse_problem (E : Ellipse) (l : Line) (t : RightTriangle) : Prop :=
  -- The line passes through F₁
  l.point = E.foci.1 ∧
  -- The line has slope 2
  l.slope = 2 ∧
  -- The line intersects E at R and Q (implied by the problem)
  ∃ R Q : EuclideanSpace ℝ (Fin 2), R ≠ Q ∧
  -- Triangle PF₁F₂ is a right triangle
  t.vertices = (E.foci.1, E.foci.2, t.vertices.2.2)

-- The theorem to prove
theorem ellipse_eccentricity (E : Ellipse) (l : Line) (t : RightTriangle) :
  ellipse_problem E l t →
  E.eccentricity = Real.sqrt 5 - 2 ∨ E.eccentricity = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l450_45050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_mundane_primes_l450_45015

/-- A prime number p is mundane if there exist positive integers a and b 
    less than p/2 such that (ab-1)/p is a positive integer -/
def IsMundane (p : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < p/2 ∧ b < p/2 ∧ (∃ k : ℕ, a * b - 1 = k * p)

theorem not_mundane_primes (p : ℕ) : 
  Nat.Prime p → (¬ IsMundane p ↔ p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13) := by
  sorry

#check not_mundane_primes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_mundane_primes_l450_45015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_cannot_reach_125_l450_45039

/-- The height of the balloon after n minutes -/
noncomputable def balloonHeight (n : ℕ) : ℝ :=
  25 * (1 - 0.8^n) / (1 - 0.8)

/-- Theorem stating that the balloon cannot reach 125 meters -/
theorem balloon_cannot_reach_125 :
  ∀ n : ℕ, balloonHeight n < 125 := by
  intro n
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_cannot_reach_125_l450_45039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_theorem_l450_45078

noncomputable def park_fencing_cost (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) 
                      (cost_a : ℝ) (cost_b : ℝ) : ℝ :=
  let length := Real.sqrt (area * ratio_long / ratio_short)
  let width := length * ratio_short / ratio_long
  let diagonal := Real.sqrt (length^2 + width^2)
  let cost_long_sides := 2 * length * cost_a
  let cost_short_sides := 2 * width * cost_b
  let cost_diagonal := diagonal * cost_b
  cost_long_sides + cost_short_sides + cost_diagonal

theorem park_fencing_cost_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |park_fencing_cost 3750 3 2 0.80 1.20 - 348.17| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_fencing_cost_theorem_l450_45078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_zero_range_l450_45045

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log x + 1 / (2 * x)

-- Define the function g
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem f_monotonicity_and_g_zero_range :
  ∀ m : ℝ, 
  (∃! x : ℝ, x ∈ Set.Icc (1/exp 1) 1 ∧ g x m = 0) →
  m ∈ Set.Ioo ((exp 1)/2 - 1) (1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_g_zero_range_l450_45045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeline_project_rate_l450_45062

/-- The originally planned daily construction rate for a pipeline project --/
noncomputable def planned_daily_rate (total_length : ℝ) (actual_rate_multiplier : ℝ) (days_saved : ℝ) : ℝ :=
  total_length / (total_length / (actual_rate_multiplier * (total_length / (total_length / (actual_rate_multiplier * 60) + days_saved))) + days_saved)

/-- Theorem stating that the planned daily rate for the given project conditions is 60 meters per day --/
theorem pipeline_project_rate : 
  planned_daily_rate 720 1.2 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeline_project_rate_l450_45062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_f_maximal_monotone_interval_l450_45018

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x)

-- State the theorem
theorem f_monotone_increasing_interval :
  ∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂ :=
by sorry

-- Define the set representing the interval (2, +∞)
def monotone_interval : Set ℝ := {x | 2 < x}

-- State that this is the maximal interval where f is strictly increasing
theorem f_maximal_monotone_interval :
  ∀ a b, a < b ∧ (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) →
  ∃ c d, c ≤ a ∧ b ≤ d ∧ monotone_interval = {x | c < x ∧ x < d} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_interval_f_maximal_monotone_interval_l450_45018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_with_18_factors_l450_45043

theorem integer_with_18_factors (x : ℕ) 
  (h1 : (Finset.filter (λ y : ℕ => y ∣ x) (Finset.range (x + 1))).card = 18)
  (h2 : 18 ∣ x)
  (h3 : 20 ∣ x) :
  x = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_with_18_factors_l450_45043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_volume_of_specific_pyramid_l450_45068

/-- Represents a pyramid with a square base -/
structure Pyramid where
  base_side : ℝ
  height : ℝ

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * radius^3

/-- Theorem: The volume of the circumscribed sphere of a specific pyramid -/
theorem circumscribed_sphere_volume_of_specific_pyramid :
  let p := Pyramid.mk 2 1
  sphere_volume (3/2) = 9/2 * Real.pi := by
  sorry

#check circumscribed_sphere_volume_of_specific_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_volume_of_specific_pyramid_l450_45068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flaw_in_thiefs_reasoning_l450_45001

-- Define the possible locations of the flaw
inductive FlawLocation
  | MajorPremise
  | MinorPremise
  | Conclusion
  | None

-- Define a type for VCR
structure VCR where
  id : Nat

-- Define predicates for ownership and ability to open
def is_owner : VCR → Prop := sorry
def can_open : VCR → Prop := sorry

-- Define the structure of the argument
structure ThiefArgument where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define the thief's specific argument
def thiefs_argument : ThiefArgument where
  major_premise := ∀ (vcr : VCR), is_owner vcr → can_open vcr
  minor_premise := can_open ⟨0⟩  -- Using an arbitrary VCR with id 0
  conclusion := is_owner ⟨0⟩

-- Define the theorem
theorem flaw_in_thiefs_reasoning :
  ∃ (flaw : FlawLocation), flaw = FlawLocation.MajorPremise := by
  -- The proof is omitted and replaced with sorry
  sorry

#check flaw_in_thiefs_reasoning

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flaw_in_thiefs_reasoning_l450_45001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_francesca_fruit_drink_calories_l450_45049

/-- Represents the recipe and calorie content of Francesca's fruit drink --/
structure FruitDrink where
  lemon_juice : ℚ
  sugar : ℚ
  orange_juice : ℚ
  water : ℚ
  lemon_juice_calories : ℚ
  orange_juice_calories : ℚ
  sugar_calories : ℚ

/-- Calculates the total calories in the fruit drink --/
def total_calories (drink : FruitDrink) : ℚ :=
  drink.lemon_juice * drink.lemon_juice_calories / 100 +
  drink.sugar * drink.sugar_calories / 100 +
  drink.orange_juice * drink.orange_juice_calories / 100

/-- Calculates the total weight of the fruit drink --/
def total_weight (drink : FruitDrink) : ℚ :=
  drink.lemon_juice + drink.sugar + drink.orange_juice + drink.water

/-- Theorem: The number of calories in 300g of Francesca's fruit drink is approximately 235 --/
theorem francesca_fruit_drink_calories :
  let drink : FruitDrink := {
    lemon_juice := 150,
    sugar := 120,
    orange_juice := 50,
    water := 380,
    lemon_juice_calories := 30,
    orange_juice_calories := 45,
    sugar_calories := 400
  }
  let calories_per_gram : ℚ := total_calories drink / total_weight drink
  let calories_in_300g : ℚ := 300 * calories_per_gram
  ∃ ε : ℚ, ε > 0 ∧ |calories_in_300g - 235| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_francesca_fruit_drink_calories_l450_45049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MON_l450_45098

noncomputable section

/-- Line l parameterized by t -/
def line_l (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 - t, 1 + Real.sqrt 3 * t)

/-- Curve C in polar coordinates -/
def curve_C (θ : ℝ) : ℝ := 4 * Real.sin (θ + Real.pi / 3)

/-- Distance from origin to line l -/
def distance_origin_to_line : ℝ := 2

/-- Length of chord MN -/
def chord_length : ℝ := 4

/-- Area of a triangle given three points -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem area_triangle_MON :
  let M : ℝ × ℝ := sorry
  let N : ℝ × ℝ := sorry
  let O : ℝ × ℝ := (0, 0)
  area_triangle O M N = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MON_l450_45098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_passes_fixed_point_l450_45033

noncomputable def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

noncomputable def point_A : ℝ × ℝ := (0, -Real.sqrt 3)

def line_l_condition (P Q : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2 ∧ 
  ellipse_C Q.1 Q.2 ∧ 
  P ≠ point_A ∧ 
  Q ≠ point_A ∧
  (P.2 - point_A.2) / (P.1 - point_A.1) + (Q.2 - point_A.2) / (Q.1 - point_A.1) = 2

noncomputable def fixed_point : ℝ × ℝ := (Real.sqrt 3, Real.sqrt 3)

theorem ellipse_intersection_passes_fixed_point :
  ∀ P Q : ℝ × ℝ, line_l_condition P Q → 
  ∃ k t : ℝ, ∀ x y : ℝ, y = k * x + t → 
  (x = P.1 ∧ y = P.2) ∨ (x = Q.1 ∧ y = Q.2) → 
  fixed_point.2 = k * fixed_point.1 + t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_passes_fixed_point_l450_45033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l450_45009

-- Define the points
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, 9)
def C : ℝ × ℝ := (6, 6)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the perimeter function
noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

-- Theorem statement
theorem triangle_perimeter : perimeter A B C = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l450_45009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_theorem_l450_45090

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given three points -/
noncomputable def triangleArea (A B C : Point) : ℝ := sorry

/-- A set of points is covered by a triangle if all points in the set are inside or on the triangle -/
def isCoveredBy (S : Set Point) (A B C : Point) : Prop := sorry

theorem triangle_covering_theorem (S : Set Point) (h : Set.Finite S) :
  (∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → triangleArea A B C ≤ 1) →
  ∃ A' B' C' : Point, triangleArea A' B' C' = 4 ∧ isCoveredBy S A' B' C' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covering_theorem_l450_45090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_area_ratio_l450_45086

-- Define the square dartboard
def Dartboard : Set (ℝ × ℝ) := {p | let (x, y) := p; -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1}

-- Define the 45° angle
noncomputable def angle45 : ℝ := Real.pi / 4

-- Define a triangular region
def TriangularRegion : Set (ℝ × ℝ) := {p | let (x, y) := p; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ x}

-- Define a quadrilateral region
def QuadrilateralRegion : Set (ℝ × ℝ) := {p | let (x, y) := p; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≤ y}

-- Define the area of a region
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := (MeasureTheory.volume S).toReal

-- State the theorem
theorem dartboard_area_ratio :
  (area QuadrilateralRegion) / (area TriangularRegion) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_area_ratio_l450_45086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l450_45019

/-- Represents the time in hours since the start of work -/
abbrev Time := ℝ

/-- Represents the fraction of work completed -/
abbrev WorkFraction := ℝ

/-- The rate at which person a completes work per hour -/
noncomputable def rate_a : ℝ := 1 / 4

/-- The rate at which person b completes work per hour -/
noncomputable def rate_b : ℝ := 1 / 12

/-- The amount of work completed in a two-hour cycle -/
noncomputable def work_per_cycle : WorkFraction := rate_a + rate_b

/-- The time it takes to complete the entire work -/
noncomputable def completion_time : Time := 6

theorem work_completion_time :
  work_per_cycle * (completion_time / 2) = (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l450_45019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_first_not_greater_than_second_l450_45095

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  card_set.product card_set |>.filter (fun (a, b) => a ≤ b)

theorem probability_first_not_greater_than_second :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card card_set ^ 2 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_first_not_greater_than_second_l450_45095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_evaluation_l450_45038

-- Proposition ①
def proposition1 (x y : ℝ) : Prop :=
  x ≥ 2 ∧ y ≥ 3 → x + y ≥ 5

-- Proposition ②
def proposition2 (x : ℝ) : Prop :=
  (x^2 - 4*x + 3 = 0 → x = 3) →
  (x ≠ 3 → x^2 - 4*x + 3 ≠ 0)

-- Proposition ③
def proposition3 : Prop :=
  (∀ x : ℝ, x > 1 → |x| > 0) ∧ ∃ x : ℝ, |x| > 0 ∧ x ≤ 1

-- Proposition ④
def proposition4 (m : ℝ) : Prop :=
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ m) → m ≤ 4

theorem propositions_evaluation :
  (¬∀ x y : ℝ, proposition1 x y) ∧
  (∀ x : ℝ, proposition2 x) ∧
  proposition3 ∧
  (∀ m : ℝ, proposition4 m) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_evaluation_l450_45038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l450_45066

/-- Helper function to define a line through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))}

/-- Given a triangle ABC with vertices A(-3,0), B(2,1), and C(-2,3),
    prove the equations of line BC and median AD -/
theorem triangle_equations (A B C : ℝ × ℝ) : 
  A = (-3, 0) → B = (2, 1) → C = (-2, 3) →
  (∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ line_through B C ↔ x + 2*y - 4 = 0) ∧
  (∃ (m b : ℝ), ∀ (x y : ℝ), (x, y) ∈ line_through A ((B.1 + C.1)/2, (B.2 + C.2)/2) ↔ 2*x - 3*y + 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equations_l450_45066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_after_walk_l450_45040

-- Define a regular hexagon
def RegularHexagon := {s : ℝ // s > 0}

-- Define the side length of our specific hexagon
def hexagonSide : RegularHexagon := ⟨4, by norm_num⟩

-- Define the distance walked along the perimeter
def distanceWalked : ℝ := 10

-- Define the function to calculate the end point coordinates
noncomputable def endPoint (h : RegularHexagon) (d : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem to be proved
theorem shortest_distance_after_walk (h : RegularHexagon) (d : ℝ) :
  distance (0, 0) (endPoint h d) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_after_walk_l450_45040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l450_45027

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define A' as the intersection of angle bisector of ∠A and perpendicular bisector of AB
noncomputable def A' : ℂ := (A^2 - B*C) * B / (B - C)

-- Define B' as the intersection of angle bisector of ∠B and median of BC
noncomputable def B' : ℂ := (B^2 - C*A) * C / (C - A)

-- Define C' as the intersection of angle bisector of ∠C and perpendicular bisector of CA
noncomputable def C' : ℂ := (C^2 - A*B) * A / (A - B)

-- Theorem statement
theorem angle_relation (hDistinct : A' A B C ≠ B' A B C ∧ B' A B C ≠ C' A B C ∧ C' A B C ≠ A' A B C) :
  Complex.arg ((B' A B C - A' A B C) / (C' A B C - A' A B C)) = π/2 - (1/2) * Complex.arg ((B - A) / (C - A)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l450_45027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l450_45085

-- Define the complex number Z₁
def Z₁ (a : ℝ) : ℂ := 2 + a * Complex.I

-- Theorem statement
theorem complex_number_properties (a : ℝ) (h1 : a > 0) 
  (h2 : (Z₁ a)^2 = Complex.I * Complex.im ((Z₁ a)^2)) : 
  a = 2 ∧ Complex.abs (Z₁ a / (1 - Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l450_45085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l450_45089

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f (x + y)) = f (y * f x) + x^2) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l450_45089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l450_45006

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  strata_sizes : List ℕ
  sample_size : ℕ

/-- Calculates the sample sizes for each stratum in a stratified sampling -/
def calculate_strata_samples (s : StratifiedSample) : List ℕ :=
  s.strata_sizes.map (λ stratum_size => (stratum_size * s.sample_size) / s.total_population)

/-- Theorem stating that the calculated sample sizes are correct for the given scenario -/
theorem correct_stratified_sample :
  let s : StratifiedSample := {
    total_population := 100,
    strata_sizes := [45, 25, 30],
    sample_size := 20
  }
  calculate_strata_samples s = [9, 5, 6] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stratified_sample_l450_45006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l450_45070

theorem problem_statement (x : ℝ) : Real.exp (Real.log 7) = 9*x + 2 → x = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l450_45070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_circles_tangent_internally_alt_l450_45011

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 17 = 0

-- Define the centers and radii of the circles
def center₁ : ℝ × ℝ := (-1, -2)
def center₂ : ℝ × ℝ := (2, -2)
def radius₁ : ℝ := 2
def radius₂ : ℝ := 5

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 9

-- Theorem stating that the circles are tangent internally
theorem circles_tangent_internally :
  distance_between_centers = abs (radius₂ - radius₁) := by
  sorry

-- Theorem stating that the circles are tangent internally (alternative formulation)
theorem circles_tangent_internally_alt :
  distance_between_centers = radius₂ - radius₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_internally_circles_tangent_internally_alt_l450_45011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_always_one_l450_45069

def b (n : ℕ) : ℚ := (7^n - 1) / 6

theorem gcd_b_always_one (n : ℕ) : Nat.gcd (Int.natAbs (Int.floor (b n))) (Int.natAbs (Int.floor (b (n + 1)))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_b_always_one_l450_45069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l450_45044

/-- A function f is defined as f(x) = (x * e^x) / (e^(ax) - 1) where a is a real number. -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x * Real.exp x) / (Real.exp (a * x) - 1)

/-- The theorem states that if f is an even function, then a must equal 2. -/
theorem f_even_implies_a_eq_two (a : ℝ) : 
  (∀ x, f a x = f a (-x)) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l450_45044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_are_squares_l450_45053

theorem rectangles_are_squares (n : ℕ) (h_n : n > 1) : 
  ∀ (width height : ℕ) (squares : Fin n → ℕ),
    (∀ i : Fin n, ∃ (a b : ℕ), width = a * (squares i) ∧ height = b * (squares i)) →
    Nat.Prime (Finset.sum (Finset.univ : Finset (Fin n)) squares) →
    width = height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangles_are_squares_l450_45053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_ratio_l450_45065

/-- An arithmetic sequence with a non-zero common difference where a₁, a₃, and a₄ form a geometric sequence. -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  is_geometric : (a 3) ^ 2 = a 1 * a 4

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a 1 + (n - 1 : ℝ) * seq.d)

/-- The main theorem -/
theorem special_sequence_ratio (seq : SpecialArithmeticSequence) :
  (S seq 3 - S seq 2) / (S seq 5 - S seq 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_ratio_l450_45065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_waterproof_cost_l450_45036

-- Define the dimensions and costs
def sheet_width : ℝ := 10
def sheet_height : ℝ := 15
def sheet_cost : ℝ := 35
def wall_width : ℝ := 9
def wall_height : ℝ := 7
def roof_base : ℝ := 9
def roof_height : ℝ := 6

-- Define the theorem
theorem jack_waterproof_cost : 
  let wall_area := wall_width * wall_height
  let roof_area := roof_base * roof_height / 2
  let total_area := wall_area + 2 * roof_area
  let sheet_area := sheet_width * sheet_height
  let sheets_needed := if total_area ≤ sheet_area then 1 else 2
  sheets_needed * sheet_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_waterproof_cost_l450_45036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_correct_l450_45099

/-- The minimum positive value of ω that satisfies the given conditions -/
noncomputable def min_omega : ℝ := 3/2

/-- The period of the sine function with angular frequency ω -/
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

/-- The horizontal shift of the graph -/
noncomputable def shift : ℝ := 4 * Real.pi / 3

theorem min_omega_is_correct (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (n : ℕ), shift = n * period ω) : 
  ω ≥ min_omega := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_is_correct_l450_45099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_has_96_animals_l450_45091

/-- Represents the farm with its animals and food consumption --/
structure Farm where
  sheep_to_horse_ratio : ℚ
  horse_food_per_day : ℕ
  total_horse_food_per_day : ℕ
  sheep_food_per_day : ℕ
  total_food_per_day : ℕ

/-- Calculates the total number of animals on the farm --/
def total_animals (f : Farm) : ℕ :=
  let horses := f.total_horse_food_per_day / f.horse_food_per_day
  let sheep := (horses * f.sheep_to_horse_ratio.num) / f.sheep_to_horse_ratio.den
  (horses + sheep).toNat

/-- Theorem stating that the farm has 96 animals --/
theorem farm_has_96_animals : ∃ (f : Farm),
  f.sheep_to_horse_ratio = 5 / 7 ∧
  f.horse_food_per_day = 230 ∧
  f.total_horse_food_per_day = 12880 ∧
  f.sheep_food_per_day = 150 ∧
  f.total_food_per_day = 25000 ∧
  total_animals f = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_has_96_animals_l450_45091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_range_l450_45083

-- Define the circle equation
def circle_equation (x y θ : ℝ) : Prop :=
  y^2 - 6*y*Real.sin θ + x^2 - 8*x*Real.cos θ + 7*(Real.cos θ)^2 + 8 = 0

-- Define the theorem
theorem circle_center_trajectory_range :
  ∀ (x y θ : ℝ), 
    0 ≤ θ → θ < 2*Real.pi → 
    circle_equation x y θ → 
    -Real.sqrt 73 ≤ 2*x + y ∧ 2*x + y ≤ Real.sqrt 73 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_trajectory_range_l450_45083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ordering_l450_45088

-- Define the set of line numbers
inductive LineNumber
| one
| two
| three
| four

-- Define a structure for a line
structure Line where
  number : LineNumber
  k : ℝ
  b : ℝ

-- Define the set of lines
def lines : Set Line := sorry

-- Axioms based on the problem conditions
axiom line2_negative_slope : ∀ l, l ∈ lines → l.number = LineNumber.two → l.k < 0
axiom other_lines_positive_slope : ∀ l, l ∈ lines → l.number ≠ LineNumber.two → l.k > 0
axiom slope_order : ∀ l1 l2 l3, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → 
  l1.number = LineNumber.one → l2.number = LineNumber.three → l3.number = LineNumber.four →
  l1.k < l2.k ∧ l2.k < l3.k

axiom intercept_order : ∀ l1 l2 l3 l4, l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → l4 ∈ lines →
  l1.number = LineNumber.four → l2.number = LineNumber.one → 
  l3.number = LineNumber.three → l4.number = LineNumber.two →
  l1.b < l2.b ∧ l2.b < l3.b ∧ l3.b < l4.b

-- Theorem to prove
theorem line_ordering :
  (∃ l1 l2 l3 l4, l1 ∈ lines ∧ l2 ∈ lines ∧ l3 ∈ lines ∧ l4 ∈ lines ∧
    l1.number = LineNumber.two ∧ l2.number = LineNumber.one ∧ 
    l3.number = LineNumber.three ∧ l4.number = LineNumber.four ∧
    l1.k < l2.k ∧ l2.k < l3.k ∧ l3.k < l4.k) ∧
  (∃ l1 l2 l3 l4, l1 ∈ lines ∧ l2 ∈ lines ∧ l3 ∈ lines ∧ l4 ∈ lines ∧
    l1.number = LineNumber.four ∧ l2.number = LineNumber.one ∧ 
    l3.number = LineNumber.three ∧ l4.number = LineNumber.two ∧
    l1.b < l2.b ∧ l2.b < l3.b ∧ l3.b < l4.b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ordering_l450_45088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_properties_l450_45030

theorem cos_alpha_properties (α : ℝ) 
  (h1 : Real.cos α = Real.sqrt 5 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 ∧ Real.tan (α + Real.pi / 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_properties_l450_45030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l450_45004

noncomputable def g (n : ℕ) : ℝ := 
  (2 + Real.sqrt 2) / 4 * (1 + Real.sqrt 2) ^ n + (2 - Real.sqrt 2) / 4 * (1 - Real.sqrt 2) ^ n

theorem g_difference (n : ℕ) : g (n + 1) - g (n - 1) = 2 * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l450_45004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_5_7_not_arithmetic_l450_45013

theorem sqrt_3_5_7_not_arithmetic : ¬ (∃ d : ℝ, (Real.sqrt 5 - Real.sqrt 3 = d) ∧ (Real.sqrt 7 - Real.sqrt 5 = d)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_5_7_not_arithmetic_l450_45013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_need_to_buy_two_dice_l450_45020

/-- The number of dice Mark and James need to buy -/
def dice_to_buy (total_needed : ℕ) (mark_total : ℕ) (mark_percent : ℚ) (james_total : ℕ) (james_percent : ℚ) : ℕ :=
  (total_needed : ℤ) - (mark_total : ℚ) * mark_percent - (james_total : ℚ) * james_percent |>.floor.toNat

/-- Theorem stating that Mark and James need to buy 2 dice -/
theorem need_to_buy_two_dice :
  dice_to_buy 14 10 (60/100) 8 (75/100) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_need_to_buy_two_dice_l450_45020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l450_45096

/-- Represents a parabola with equation y² = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : Point :=
  { x := par.p / 2, y := 0 }

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Condition that a point lies on the parabola -/
def on_parabola (par : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * par.p * p.x

theorem parabola_focus_distance (par : Parabola) :
  ∃ (m : ℝ), 
    let p : Point := { x := 1, y := m }
    on_parabola par p ∧ distance p (focus par) = 3 → par.p = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l450_45096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_l450_45054

/-- IsArithmeticSequence for a list of three real numbers -/
def IsArithmeticSequence (seq : List ℝ) : Prop :=
  seq.length = 3 ∧ seq[1]! - seq[0]! = seq[2]! - seq[1]!

theorem arithmetic_sequence_square_root (x : ℝ) : 
  x > 0 → 
  IsArithmeticSequence [2^2, x^2, 4^2] → 
  x = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_square_root_l450_45054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l450_45079

/-- Represents the number of balls in the bag -/
def totalBalls : ℕ := 5

/-- Represents the number of red balls in the bag -/
def redBalls : ℕ := 2

/-- Represents the number of white balls in the bag -/
def whiteBalls : ℕ := 3

/-- Represents the number of balls drawn -/
def ballsDrawn : ℕ := 3

/-- Random variable X representing the number of red balls drawn -/
noncomputable def X : ℕ → ℝ := sorry

/-- Probability of drawing exactly one red ball -/
noncomputable def probOneRed : ℝ := 3/5

/-- Probability distribution of X -/
noncomputable def probDistX : ℕ → ℝ
  | 0 => 1/10
  | 1 => 3/5
  | 2 => 3/10
  | _ => 0

theorem ball_drawing_probabilities :
  (totalBalls = redBalls + whiteBalls) ∧
  (probOneRed = 3/5) ∧
  (∀ k, probDistX k = X k) ∧
  (probDistX 0 + probDistX 1 + probDistX 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_drawing_probabilities_l450_45079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_of_squares_l450_45042

theorem consecutive_even_numbers_sum_of_squares : 
  ∀ (x : ℤ), 
  (∃ (a b c d : ℤ), 
    (a = x) ∧ (b = x + 2) ∧ (c = x + 4) ∧ (d = x + 6) ∧
    (a + b + c + d = 36) ∧
    (Even a ∧ Even b ∧ Even c ∧ Even d)) →
  (x^2 + (x + 2)^2 + (x + 4)^2 + (x + 6)^2 = 344) :=
by
  intro x h
  -- The proof steps would go here, but we'll use sorry as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_of_squares_l450_45042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_sum_of_coefficients_main_result_l450_45035

theorem sum_of_coefficients_expanded_form (d : ℝ) : 
  let expanded := -2 * (4 - d) * (d + 3 * (4 - d))
  expanded = -4 * d^2 + 40 * d - 96 := by
  sorry

theorem sum_of_coefficients (d : ℝ) :
  let expanded := -4 * d^2 + 40 * d - 96
  (-4) + 40 + (-96) = -60 := by
  sorry

theorem main_result (d : ℝ) :
  let expanded := -2 * (4 - d) * (d + 3 * (4 - d))
  let coefficients_sum := (-4) + 40 + (-96)
  coefficients_sum = -60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_sum_of_coefficients_main_result_l450_45035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_problem_l450_45087

noncomputable section

def Point : Type := ℝ × ℝ

def IsOnUnitCircle (p : Point) : Prop := p.1^2 + p.2^2 = 1

theorem unit_circle_problem (θ : ℝ) (A B : Point) :
  IsOnUnitCircle A ∧
  IsOnUnitCircle B ∧
  A.1 = 1 ∧
  A.2 = 0 ∧
  B.1 < 0 ∧
  B.2 > 0 ∧
  Real.sin θ = 4/5 →
  (B.1 = -3/5 ∧ B.2 = 4/5) ∧
  (Real.sin (π + θ) + 2 * Real.sin (π/2 + θ)) / (2 * Real.cos (π - θ)) = -5/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_problem_l450_45087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l450_45076

/-- Given three points on a horizontal plane at distances 100m, 200m, and 300m from the base of a
    vertical tower, if the sum of the angles of elevation from these points to the top of the tower
    is 90°, then the height of the tower is 100 meters. -/
theorem tower_height (α β γ : ℝ) (h : α + β + γ = π / 2) :
  (∃ x : ℝ, x > 0 ∧ 
    Real.tan α = x / 100 ∧
    Real.tan β = x / 200 ∧
    Real.tan γ = x / 300) →
  (∃ x : ℝ, x = 100 ∧
    Real.tan α = x / 100 ∧
    Real.tan β = x / 200 ∧
    Real.tan γ = x / 300) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l450_45076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_volume_sum_l450_45051

/-- A regular rectangular pyramid with height 3 and angle π/3 between lateral faces and base -/
structure RegularPyramid where
  height : ℝ
  angle : ℝ
  height_eq : height = 3
  angle_eq : angle = Real.pi / 3

/-- A sequence of inscribed spheres in the pyramid -/
noncomputable def InscribedSpheres (pyramid : RegularPyramid) : ℕ → ℝ
  | 0 => 1  -- radius of the first sphere
  | n + 1 => InscribedSpheres pyramid n / 3  -- radius of subsequent spheres

/-- The volume of a sphere given its radius -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The sum of volumes of all inscribed spheres -/
noncomputable def totalVolume (pyramid : RegularPyramid) : ℝ :=
  ∑' n, sphereVolume (InscribedSpheres pyramid n)

/-- The main theorem stating that the sum of volumes of all inscribed spheres is 18π/13 -/
theorem inscribed_spheres_volume_sum (pyramid : RegularPyramid) :
  totalVolume pyramid = 18 * Real.pi / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_spheres_volume_sum_l450_45051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l450_45067

-- Define the function s(x)
noncomputable def s (x : ℝ) : ℝ := 1 / |1 - x|^3

-- State the theorem
theorem s_range :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 1 ∧ s x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l450_45067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_l450_45024

theorem gilda_marbles (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) : 
  (initial_marbles * (1 - 0.3) * (1 - 0.4)) / initial_marbles = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gilda_marbles_l450_45024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_matchings_power_of_two_l450_45057

/-- A bipartite graph representing beads and boxes -/
structure BeadBoxGraph where
  beads : Finset (Fin 20)
  boxes : Finset (Fin 10)
  edges : Finset ((Fin 20) × (Fin 10))
  bead_degree_two : ∀ b, b ∈ beads → (edges.filter (λ e => e.1 = b)).card = 2
  perfect_matching_exists : ∃ m : Finset ((Fin 20) × (Fin 10)), m ⊆ edges ∧ m.card = 10 ∧
    (∀ b, b ∈ beads → ∃! x, x ∈ boxes ∧ (b, x) ∈ m) ∧
    (∀ x, x ∈ boxes → ∃! b, b ∈ beads ∧ (b, x) ∈ m)

/-- The number of perfect matchings in a bead-box graph -/
def numPerfectMatchings (g : BeadBoxGraph) : ℕ := sorry

/-- The main theorem: the number of perfect matchings is a non-zero power of 2 -/
theorem perfect_matchings_power_of_two (g : BeadBoxGraph) :
  ∃ k : ℕ, numPerfectMatchings g = 2^k ∧ k > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_matchings_power_of_two_l450_45057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l450_45034

/-- The maximum value of f(x) = cos²x + 2a*sin(x) - 1 is 2a - 1, given that a > 1 and 0 ≤ x ≤ 2π -/
theorem max_value_of_f (a : ℝ) (h : a > 1) :
  ∃ (max : ℝ), max = 2 * a - 1 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi →
    (Real.cos x) ^ 2 + 2 * a * Real.sin x - 1 ≤ max :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l450_45034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_principal_change_l450_45064

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

theorem total_interest_after_principal_change 
  (P R : ℝ) 
  (h1 : simple_interest P R 10 = 1200) 
  (h2 : P > 0) 
  (h3 : R > 0) : 
  simple_interest P R 5 + simple_interest (3 * P) R 5 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_principal_change_l450_45064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wanderer_probability_l450_45003

theorem wanderer_probability (total_bars : ℕ) (bars_visited : ℕ) (in_bar_prob : ℚ) :
  total_bars = 8 →
  bars_visited = 7 →
  in_bar_prob = 4/5 →
  (in_bar_prob / total_bars) / (1 - (bars_visited * (in_bar_prob / total_bars))) = 1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wanderer_probability_l450_45003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_product_probability_l450_45058

def standard_die := Finset.range 6

theorem dice_product_probability :
  let outcomes := Finset.product standard_die (Finset.product standard_die (Finset.product standard_die standard_die))
  (Finset.filter (fun (a, b, c, d) => (a + 1) * (b + 1) * (c + 1) * (d + 1) ≠ 2) outcomes).card / outcomes.card = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_product_probability_l450_45058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_example_theorem_l450_45008

theorem example_theorem (x y : ℕ) : x + y = y + x := by
  rw [Nat.add_comm]

#check example_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_example_theorem_l450_45008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_circumscribed_cone_l450_45000

/-- A pyramid with specific properties -/
structure Pyramid where
  h : ℝ  -- height of the pyramid
  base_is_right_triangle : Bool
  lateral_edges_equal : Bool
  angle_30 : Bool  -- one lateral face makes 30° with base
  angle_60 : Bool  -- other lateral face makes 60° with base

/-- Volume of a cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Theorem: Volume of cone circumscribed around the specific pyramid -/
theorem volume_of_circumscribed_cone (p : Pyramid) : 
  ∃ (r : ℝ), cone_volume r p.h = (10 * Real.pi * p.h^3) / 9 := by
  sorry

#check volume_of_circumscribed_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_circumscribed_cone_l450_45000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l450_45060

-- Define the set S
def S : Set ℕ := {n : ℕ | n ≥ 2}

-- State the theorem
theorem no_such_function_exists :
  ¬ ∃ (f : S → S), ∀ (a b : S), a ≠ b → (f a : ℕ) * (f b : ℕ) = f ⟨(a : ℕ)^2 * (b : ℕ)^2, sorry⟩ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l450_45060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_120_x_intercept_2_parallel_distance_l450_45075

-- Define the line l₁: ax + y + 2 = 0
def line_l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define the line l₂: 2x - y + 1 = 0
def line_l₂ (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem 1: When the slope angle of l₁ is 120°, a = √3
theorem slope_angle_120 (a : ℝ) :
  (∃ x y, line_l₁ a x y ∧ Real.tan (120 * π / 180) = -a) → a = Real.sqrt 3 := by
  sorry

-- Theorem 2: When the x-intercept of l₁ is 2, a = -1
theorem x_intercept_2 (a : ℝ) :
  (∃ x, line_l₁ a x 0 ∧ x = 2) → a = -1 := by
  sorry

-- Theorem 3: When l₁ is parallel to l₂, the distance between them is (3√5) / 5
theorem parallel_distance (a : ℝ) :
  (∀ x y, line_l₁ a x y ↔ ∃ k, line_l₂ x y ∧ k ≠ 0) →
  (∃ d, d = (3 * Real.sqrt 5) / 5 ∧
    ∀ x₁ y₁ x₂ y₂, line_l₁ a x₁ y₁ → line_l₂ x₂ y₂ →
      d = |2 * x₂ - y₂ + 1| / Real.sqrt (2^2 + (-1)^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_120_x_intercept_2_parallel_distance_l450_45075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_and_segment_area_max_area_sector_l450_45071

noncomputable section

open Real

-- Constants and variables
def R : ℝ := 10
def α : ℝ := π / 3

-- Part 1
theorem arc_length_and_segment_area :
  let l := α * R
  let S_sector := (1 / 2) * α * R^2
  let S_triangle := (1 / 2) * R * (R * sin (π / 6))
  let S_segment := S_sector - S_triangle
  (l = (10 * π) / 3) ∧
  (S_segment = 50 * (π / 3 - Real.sqrt 3 / 2)) := by
  sorry

-- Part 2
theorem max_area_sector :
  ∃ (α : ℝ), 
    (2 * R + α * R = 12) →
    (∀ β, (2 * R + β * R = 12) → 
      ((1/2) * α * R^2 ≥ (1/2) * β * R^2)) ∧
    α = 2 ∧
    (1/2) * α * R^2 = 9 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_and_segment_area_max_area_sector_l450_45071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l450_45047

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (Real.sin (ω * x))^2 + Real.sqrt 3 * Real.sin (ω * x) * Real.sin (ω * x + Real.pi / 2)

theorem f_range (ω : ℝ) (h_ω : ω > 0) (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  Set.range (fun x ↦ f ω x) ∩ Set.Icc 0 (2 * Real.pi / 3) = Set.Icc 0 (3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l450_45047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_calculator_time_l450_45092

/-- The time Matt takes to do a math problem with a calculator -/
def time_with_calculator : ℝ := 2

/-- The time Matt takes to do a math problem without a calculator -/
def time_without_calculator : ℝ := 5

/-- The number of problems in Matt's assignment -/
def number_of_problems : ℕ := 20

/-- The total time saved by using a calculator for all problems -/
def total_time_saved : ℝ := 60

theorem matt_calculator_time : time_with_calculator = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matt_calculator_time_l450_45092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l450_45002

/-- Curve C₁ -/
def C₁ (x y : ℝ) : Prop := x - y - 1 = 0

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_ratio :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    C₁ A.1 A.2 ∧
    C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧
    C₂ B.1 B.2 ∧
    (distance M A * distance M B) / distance A B = Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_ratio_l450_45002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_l450_45037

theorem grid_coloring (n : ℕ) (k : ℕ) (hn : 0 < n) :
  let total_cells := n ^ 2
  let blue_cells := k
  let red_cells := total_cells - blue_cells
  let S_B := 2 * (n * blue_cells - n ^ 3)
  let S_R := 2 * (n * red_cells - n ^ 3)
  (S_B - S_R = 50) → (k = 15 ∨ k = 313) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_l450_45037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l450_45041

/-- The volume of a sphere with radius r -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of the region between two concentric spheres -/
noncomputable def concentricSpheresVolume (r₁ r₂ : ℝ) : ℝ := sphereVolume r₂ - sphereVolume r₁

theorem volume_between_spheres :
  concentricSpheresVolume 4 7 = 372 * Real.pi := by
  -- Expand the definition of concentricSpheresVolume
  unfold concentricSpheresVolume
  -- Expand the definition of sphereVolume
  unfold sphereVolume
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l450_45041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l450_45080

-- Define the square
def square_center : ℝ × ℝ := (4, -6)
def square_area : ℝ := 16

-- Define the dilation
def dilation_center : ℝ × ℝ := (2, -2)
def scale_factor : ℝ := 3

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Function to apply dilation to a point
def dilate (p : ℝ × ℝ) : ℝ × ℝ :=
  (dilation_center.1 + scale_factor * (p.1 - dilation_center.1),
   dilation_center.2 + scale_factor * (p.2 - dilation_center.2))

-- Theorem statement
theorem farthest_vertex_after_dilation :
  ∃ (vertices : List (ℝ × ℝ)),
    (vertices.length = 4) ∧
    (∀ v ∈ vertices, distance v square_center = Real.sqrt square_area / 2) ∧
    (∃ v ∈ vertices, dilate v = (14, -20)) ∧
    (∀ v ∈ vertices, distance (dilate v) dilation_center ≤ distance (14, -20) dilation_center) := by
  sorry

#eval square_center
#eval square_area
#eval dilation_center
#eval scale_factor

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l450_45080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l450_45077

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 1)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ¬(x = 1) ∧ x + 2 ≥ 0

theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ -2 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l450_45077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l450_45005

/-- A tile is blue if its number is congruent to 3 mod 7 -/
def isBlue (n : ℕ) : Prop := n % 7 = 3

/-- The total number of tiles in the box -/
def totalTiles : ℕ := 100

/-- The set of all tile numbers -/
def tileSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ totalTiles}

/-- The set of blue tile numbers -/
def blueTileSet : Set ℕ := {n ∈ tileSet | isBlue n}

/-- The number of blue tiles -/
def numBlueTiles : ℕ := Finset.card (Finset.filter (fun n => n % 7 = 3) (Finset.range totalTiles))

theorem blue_tile_probability : 
  (↑numBlueTiles : ℚ) / (↑totalTiles : ℚ) = 7 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tile_probability_l450_45005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l450_45048

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / ((x - 5)^3)

-- Define the solution set
def solution_set : Set ℝ := Set.Iic (5/3) ∪ Set.Ioi 5

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, x ≠ 5 → (f x ≥ 25 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_equivalence_l450_45048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l450_45094

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

/-- Theorem: If a hyperbola has an asymptotic line y = 2x, its eccentricity is √5 -/
theorem hyperbola_eccentricity_sqrt_5 {a b : ℝ} (h : Hyperbola a b) 
  (asymptote : b = 2*a) : eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l450_45094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beetles_on_line_l450_45061

/-- Represents a beetle's position on a 2D grid -/
structure BeetlePosition where
  x : ℤ
  y : ℤ
deriving Repr, DecidableEq

/-- Calculates the distance between two beetle positions -/
noncomputable def distance (a b : BeetlePosition) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Represents the configuration of beetles after landing -/
def LandedConfiguration (n : ℕ) := Fin (n^2) → BeetlePosition

/-- Checks if a configuration satisfies the distance constraint -/
def satisfiesDistanceConstraint (n : ℕ) (config : LandedConfiguration n) : Prop :=
  ∀ (i j : Fin (n^2)), distance (config i) (config j) ≤ 1 →
    distance ({ x := i.val % n, y := i.val / n } : BeetlePosition)
             ({ x := j.val % n, y := j.val / n } : BeetlePosition) ≤ 1

/-- Checks if a line with slope 1 contains at least n beetles -/
def hasLineWithNBeetles (n : ℕ) (config : LandedConfiguration n) : Prop :=
  ∃ (k : ℤ), (Finset.filter (λ p : BeetlePosition => p.y = p.x + k)
    (Finset.image config Finset.univ)).card ≥ n

/-- The main theorem -/
theorem beetles_on_line (n : ℕ) (config : LandedConfiguration n) :
  satisfiesDistanceConstraint n config → hasLineWithNBeetles n config := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beetles_on_line_l450_45061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_value_at_one_l450_45052

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ := 3 * Real.cos (Real.pi * x + θ)

theorem f_symmetric_value_at_one (θ : ℝ) :
  (∀ x, f x θ = f (2 - x) θ) →
  (f 1 θ = 3 ∨ f 1 θ = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_value_at_one_l450_45052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_time_per_side_l450_45022

-- Define the given conditions
def total_guests : Nat := 30
def grill_capacity : Nat := 5
def total_cooking_time : Nat := 72

-- Define the function to calculate total burgers
def total_burgers (guests : Nat) : Nat :=
  (guests / 2) * 2 + (guests / 2) * 1

-- Define the function to calculate number of batches
def num_batches (total : Nat) (capacity : Nat) : Nat :=
  (total + capacity - 1) / capacity

-- Theorem to prove
theorem cooking_time_per_side (guests : Nat) (capacity : Nat) (total_time : Nat)
  (h1 : guests = total_guests)
  (h2 : capacity = grill_capacity)
  (h3 : total_time = total_cooking_time) :
  total_time / (2 * num_batches (total_burgers guests) capacity) = 4 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_time_per_side_l450_45022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_A_l450_45074

noncomputable def A : ℝ × ℝ := (4, 0)
noncomputable def B : ℝ × ℝ := (8, 10)
noncomputable def C : ℝ × ℝ := (0, 6)

noncomputable def slope_BC : ℝ := (C.2 - B.2) / (C.1 - B.1)

def parallel_line_equation (x y : ℝ) : Prop :=
  x - 2*y - 4 = 0

theorem parallel_line_through_A :
  ∀ x y : ℝ, 
    (y - A.2 = slope_BC * (x - A.1)) ↔ parallel_line_equation x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_through_A_l450_45074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l450_45032

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem derivative_f_at_one : 
  deriv f 1 = 1 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l450_45032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l450_45082

noncomputable def a : ℝ := Real.sqrt 0.6
noncomputable def b : ℝ := Real.sqrt 0.7
noncomputable def c : ℝ := Real.log 0.7 / Real.log 10

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l450_45082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l450_45055

theorem max_value_expression (p q r : ℝ) 
  (non_neg_p : p ≥ 0) (non_neg_q : q ≥ 0) (non_neg_r : r ≥ 0)
  (sum_condition : p + 2*q + 3*r = 1) :
  p + 2 * Real.sqrt (p*q) + 3 * (p*q*r)^(1/3) ≤ 1/2 + 1/Real.sqrt 3 + 1/(4^(1/3)) := by
  sorry

-- Approximate evaluation
#eval (1/2 : Float) + 1/Float.sqrt 3 + 1/(4 : Float)^(1/3)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l450_45055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l450_45017

def alice_numbers : Finset Nat := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ 20 ∣ n) (Finset.range 300)
def alex_numbers : Finset Nat := Finset.filter (λ n => n > 0 ∧ n < 300 ∧ 30 ∣ n) (Finset.range 300)

theorem same_number_probability :
  (Finset.card (alice_numbers ∩ alex_numbers) : Rat) /
  ((Finset.card alice_numbers) * (Finset.card alex_numbers) : Rat) = 1 / 30 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l450_45017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l450_45016

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 7*x₁ = -5*x₁ + 3) ∧ 
  (x₂^2 - 7*x₂ = -5*x₂ + 3) ∧ 
  (∀ x : ℝ, x^2 = 3*x - 8 → False) ∧
  (∀ x : ℝ, 7*x^2 - 14*x + 7 = 0 → ∃ y : ℝ, y ≠ x ∧ 7*y^2 - 14*y + 7 = 0 → False) ∧
  (∀ x : ℝ, x^2 + 5*x = -10 → False) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l450_45016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l450_45093

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem equation_solutions :
  let solutions : Set ℝ := {2, 2 * Real.sqrt 19, 2 * Real.sqrt 22, 10}
  ∀ x : ℝ, x^2 - 12 * (floor x) + 20 = 0 ↔ x ∈ solutions :=
by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l450_45093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l450_45023

/-- Parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Point on the parabola with x-coordinate 3 -/
noncomputable def PointOnParabola : ℝ × ℝ :=
  (3, 2 * Real.sqrt 3)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance :
  PointOnParabola ∈ Parabola →
  distance PointOnParabola Focus = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l450_45023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_180_miles_l450_45014

/-- The distance between two cities given travel times and a hypothetical speed --/
noncomputable def distance_between_cities (time_ab time_ba : ℝ) (hypothetical_speed : ℝ) : ℝ :=
  let total_time := time_ab + time_ba
  let saved_time := 1  -- 30 minutes saved each way, so 1 hour total
  let new_total_time := total_time - saved_time
  hypothetical_speed * new_total_time / 2

theorem distance_is_180_miles :
  distance_between_cities 3 2.5 80 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_180_miles_l450_45014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_lower_bound_l450_45031

/-- A polynomial represented as a list of coefficients -/
def MyPolynomial (α : Type*) := List α

/-- The minimum of a list of real numbers -/
noncomputable def listMin (l : List ℝ) : ℝ := l.foldl min (l.head!)

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : MyPolynomial ℝ) (x : ℝ) : ℝ :=
  p.enum.foldl (fun acc (i, a) => acc + a * x ^ (p.length - 1 - i)) 0

/-- Compute the list of partial sums of coefficients -/
def partialSums (p : MyPolynomial ℝ) : List ℝ :=
  p.scanl (· + ·) 0

theorem polynomial_lower_bound (p : MyPolynomial ℝ) (x : ℝ) (h : x ≥ 1) :
  evalPoly p x ≥ (listMin (partialSums p)) * x ^ (p.length - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_lower_bound_l450_45031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_condition_l450_45059

def satisfies_condition (S : Set ℕ) : Prop :=
  ∀ i j, i ∈ S → j ∈ S → (i + j) / Nat.gcd i j ∈ S

theorem unique_set_satisfying_condition :
  ∀ S : Set ℕ,
    S.Finite →
    S.Nonempty →
    (∀ x, x ∈ S → x > 0) →
    satisfies_condition S →
    S = {2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_set_satisfying_condition_l450_45059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_centered_on_wall_l450_45010

/-- The width of the wall in feet -/
noncomputable def wall_width : ℝ := 25

/-- The width of the picture in feet -/
noncomputable def picture_width : ℝ := 7

/-- The space on one side of the picture -/
noncomputable def side_space : ℝ := (wall_width - picture_width) / 2

theorem picture_centered_on_wall :
  side_space = 9 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picture_centered_on_wall_l450_45010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l450_45046

-- Define the complex number type
variable (z : ℂ)

-- State the theorem
theorem complex_division_simplification :
  (3 - 4 * Complex.I) / Complex.I = -4 - 3 * Complex.I :=
by
  -- Multiply numerator and denominator by -I
  have h1 : (3 - 4 * Complex.I) / Complex.I = (3 - 4 * Complex.I) * (-Complex.I) / (Complex.I * (-Complex.I)) := by sorry
  
  -- Simplify the denominator
  have h2 : Complex.I * (-Complex.I) = 1 := by sorry
  
  -- Distribute -I in the numerator
  have h3 : (3 - 4 * Complex.I) * (-Complex.I) = -3 * Complex.I + 4 * Complex.I * Complex.I := by sorry
  
  -- Simplify I * I to -1
  have h4 : Complex.I * Complex.I = -1 := by sorry
  
  -- Combine all steps
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_simplification_l450_45046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_consecutive_zero_coefficients_l450_45073

open Real
open Function
open Nat

noncomputable def f (x : ℝ) : ℝ := exp (x * (x - 1)^2)

noncomputable def power_series (a : ℕ → ℝ) (x : ℝ) : ℝ := ∑' n, a n * x^n

theorem no_three_consecutive_zero_coefficients 
  (a : ℕ → ℝ) 
  (h : ∀ x, f x = power_series a x) :
  ¬∃ n, a n = 0 ∧ a (n + 1) = 0 ∧ a (n + 2) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_three_consecutive_zero_coefficients_l450_45073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_and_xaxis_l450_45097

/-- Definition of a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_xaxis :
  let circle_A : Point := ⟨4, 4⟩
  let circle_B : Point := ⟨12, 4⟩
  let radius : ℝ := 4
  let rectangle_area : ℝ := (circle_B.x - circle_A.x) * circle_A.y
  let sector_area : ℝ := (1/4) * Real.pi * radius^2
  rectangle_area - 2 * sector_area = 32 - 8 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_circles_and_xaxis_l450_45097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_symmetry_preserves_distance_l450_45028

/-- Central symmetry transformation with respect to a point O -/
noncomputable def centralSymmetry (O A : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  2 • O - A

/-- Theorem: Central symmetry preserves distances -/
theorem central_symmetry_preserves_distance 
  (O A B : EuclideanSpace ℝ (Fin 2)) :
  let A' := centralSymmetry O A
  let B' := centralSymmetry O B
  dist A B = dist A' B' :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_symmetry_preserves_distance_l450_45028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l450_45012

/-- The binomial expansion of (2x - 1/x)^6 -/
noncomputable def binomial_expansion (x : ℝ) : ℝ := (2*x - 1/x)^6

/-- The sum of all binomial coefficients in the expansion of (2x - 1/x)^6 -/
def sum_of_coefficients : ℕ := 64

/-- The coefficient of the term containing x^2 in the expansion of (2x - 1/x)^6 -/
def coefficient_of_x_squared : ℕ := 240

theorem binomial_expansion_properties :
  (∀ x : ℝ, binomial_expansion x = (2*x - 1/x)^6) ∧
  sum_of_coefficients = 64 ∧
  coefficient_of_x_squared = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l450_45012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracket_placement_error_l450_45056

-- Define the expressions
def expr_A (x : ℝ) := -x + 5 = -(x + 5)
def expr_B (m n : ℝ) := -7*m - 2*n = -(7*m + 2*n)
def expr_C (a : ℝ) := a^2 - 3 = (a^2 - 3)  -- Removed the '+' sign
def expr_D (x y : ℝ) := 2*x - y = -(y - 2*x)

-- Theorem stating that A is incorrect while B, C, and D are correct
theorem bracket_placement_error :
  (∃ x : ℝ, ¬(expr_A x)) ∧
  (∀ m n : ℝ, expr_B m n) ∧
  (∀ a : ℝ, expr_C a) ∧
  (∀ x y : ℝ, expr_D x y) :=
by
  sorry  -- Skipping the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracket_placement_error_l450_45056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_sqrt_50_over_sqrt_25_minus_sqrt_5_l450_45029

noncomputable def rationalize_and_minimize (x y z : ℝ) : Prop :=
  let original := Real.sqrt x / (Real.sqrt y - Real.sqrt z)
  let rationalized := Real.sqrt 2 * (5 + Real.sqrt 10) / 4
  (original = rationalized) ∧
  (∀ a b c d : ℤ, 
    (d > 0) →
    (¬ ∃ p : ℕ, Nat.Prime p ∧ (↑p : ℤ)^2 ∣ b) →
    (rationalized = (↑a * Real.sqrt (↑b : ℝ) + ↑c) / ↑d) →
    (a + b + c + d ≥ 1 + 2 + 5 + 4))

theorem rationalize_sqrt_50_over_sqrt_25_minus_sqrt_5 :
  rationalize_and_minimize 50 25 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_sqrt_50_over_sqrt_25_minus_sqrt_5_l450_45029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_one_min_value_one_implies_a_geq_two_l450_45025

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

-- Theorem 1: If f has an extremum at x=1, then a = 1
theorem extremum_implies_a_equals_one (a : ℝ) (h : a > 0) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 1 := by
  sorry

-- Theorem 2: If the minimum value of f is 1, then a ≥ 2
theorem min_value_one_implies_a_geq_two (a : ℝ) (h : a > 0) :
  (∀ (x : ℝ), f a x ≥ 1) →
  a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_implies_a_equals_one_min_value_one_implies_a_geq_two_l450_45025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_two_l450_45021

-- Define the function f(x) = 2 - |x|
def f (x : ℝ) : ℝ := 2 - abs x

-- State the theorem
theorem integral_f_minus_one_to_two :
  ∫ x in (-1 : ℝ)..2, f x = (3.5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_minus_one_to_two_l450_45021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l450_45026

-- Define the fraction
def fraction : ℚ := 4 / 3^5

-- Define a function to get the last digit of a decimal expansion
def last_digit (q : ℚ) : ℕ :=
  (((q * 10^6).floor : ℤ).toNat) % 10

-- State the theorem
theorem last_digit_of_fraction :
  last_digit fraction = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_fraction_l450_45026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_matches_negative_derivative_l450_45084

open Real

noncomputable def f (x : ℝ) : ℝ := cos x - sin x

theorem graph_translation_matches_negative_derivative :
  ∃ m : ℝ, ∀ x : ℝ, f (x - m) = -((-sin x) - cos x) → m = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_matches_negative_derivative_l450_45084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l450_45007

/-- Sequence sum function -/
def S (n : ℕ) (k : ℝ) : ℝ := k * n ^ 2 + n

/-- General term of the sequence -/
def a (n : ℕ) (k : ℝ) : ℝ := 2 * k * n - k + 1

theorem sequence_properties (k : ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n k - S (n - 1) k = a n k) ∧
  (∀ m : ℕ, m ≥ 1 → (a m k) * (a (4 * m) k) = (a (2 * m) k) ^ 2 → k = 0 ∨ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l450_45007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l450_45081

theorem smallest_number_divisible (n : ℕ) : 
  (∀ d ∈ ({25, 50, 75, 100, 150} : Set ℕ), (n - 20) % d = 0) →
  (∀ m < n, ∃ d ∈ ({25, 50, 75, 100, 150} : Set ℕ), (m - 20) % d ≠ 0) →
  n = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_l450_45081
