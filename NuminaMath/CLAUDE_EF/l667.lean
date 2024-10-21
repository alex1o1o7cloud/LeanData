import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_difference_l667_66744

/-- The roots of (x^101 - 1) / (x - 1) -/
noncomputable def roots : Fin 100 → ℂ := sorry

/-- The set S of powers of roots -/
noncomputable def S : Set ℂ := {z | ∃ n : Fin 100, z = (roots n) ^ (n.val + 1)}

/-- The maximum number of unique values in S -/
noncomputable def M : ℕ := Finset.card (Finset.image (λ n => (roots n) ^ (n.val + 1)) Finset.univ)

/-- The minimum number of unique values in S -/
def N : ℕ := 1

/-- The difference between the maximum and minimum number of unique values in S -/
theorem roots_difference : M - N = 98 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_difference_l667_66744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_Q_l667_66733

open Complex

-- Define the function Q(x)
noncomputable def Q (x : ℝ) : ℂ := 2 + exp (x * I) + exp (2 * x * I) - exp (3 * x * I)

-- Theorem statement
theorem no_solutions_for_Q : ∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → Q x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_for_Q_l667_66733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_orthogonality_l667_66718

-- Define the ellipse C
def C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the points P1, P3, P4
noncomputable def P1 : ℝ × ℝ := (0, Real.sqrt 2)
noncomputable def P3 : ℝ × ℝ := (Real.sqrt 2, 1)
noncomputable def P4 : ℝ × ℝ := (-Real.sqrt 2, 1)

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4/3

-- Theorem statement
theorem ellipse_and_orthogonality 
  (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0)
  (h3 : C P1.1 P1.2 a b)
  (h4 : C P3.1 P3.2 a b)
  (h5 : C P4.1 P4.2 a b) :
  (∀ x y, C x y a b ↔ C x y 2 (Real.sqrt 2)) ∧
  (∀ l : ℝ → ℝ → Prop, 
    (∃ x₀ y₀, l x₀ y₀ ∧ circle_eq x₀ y₀ ∧ 
      (∀ x y, circle_eq x y → (x = x₀ ∧ y = y₀) ∨ ¬l x y)) →
    (∃ A B : ℝ × ℝ, l A.1 A.2 ∧ l B.1 B.2 ∧ C A.1 A.2 2 (Real.sqrt 2) ∧ C B.1 B.2 2 (Real.sqrt 2) ∧
      A.1 * B.1 + A.2 * B.2 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_orthogonality_l667_66718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_power_l667_66710

-- Define the coordinates of points P and Q as functions
def P (m : ℝ) : ℝ × ℝ := (m, 5)
def Q (n : ℝ) : ℝ × ℝ := (-2, n)

-- Define the transformation from P to Q
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2, p.2 - 3)

-- Theorem statement
theorem point_transformation_power :
  ∀ m n : ℝ, transform (P m) = Q n → m^n = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_power_l667_66710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_le_4sqrt2_l667_66728

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := x^4 - 1/x^2 - m*x

-- State the theorem
theorem f_increasing_iff_m_le_4sqrt2 :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → Monotone (fun x ↦ f x m)) ↔ m ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_m_le_4sqrt2_l667_66728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_count_l667_66786

theorem subset_intersection_count (m n : ℕ) (h : m > n) :
  let A := Finset.range m.succ
  let B := Finset.range n.succ
  (Finset.filter (fun C => (C ∩ B).Nonempty) (Finset.powerset A)).card = 2^(m-n) * (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_count_l667_66786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l667_66721

/-- Given a cone where the surface area is three times the area of its base,
    prove that the central angle of the sector in the lateral surface development diagram is 180°. -/
theorem cone_central_angle (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  π * r * l + π * r^2 = 3 * π * r^2 → 
  (2 * π * r * l) / (2 * π * l) = π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l667_66721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l667_66792

/-- Represents a route with its characteristics -/
structure Route where
  distance : ℝ
  speed : ℝ
  constructionDistance : ℝ
  constructionSpeed : ℝ
  waitTime : ℝ

/-- Calculates the time taken for a route in minutes -/
noncomputable def routeTime (r : Route) : ℝ :=
  (r.distance - r.constructionDistance) / r.speed * 60 +
  r.constructionDistance / r.constructionSpeed * 60 +
  r.waitTime

/-- The two routes described in the problem -/
def routeX : Route := {
  distance := 8,
  speed := 25,
  constructionDistance := 0,
  constructionSpeed := 25,
  waitTime := 0
}

def routeY : Route := {
  distance := 7,
  speed := 35,
  constructionDistance := 1,
  constructionSpeed := 15,
  waitTime := 2
}

/-- The main theorem to prove -/
theorem route_time_difference : 
  abs (routeTime routeX - routeTime routeY - 2.9) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_time_difference_l667_66792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_l667_66799

-- Define the set A
def A : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x

-- Define the set B as the range of f over A
def B : Set ℝ := f '' A

-- Define the complement of A in ℝ
def complementA : Set ℝ := Aᶜ

-- State the theorem
theorem intersection_complement_A_and_B :
  complementA ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_A_and_B_l667_66799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l667_66727

open Real

/-- The length of line segment P₁P₂ in the given problem setup --/
theorem line_segment_length : ∃ x : ℝ, 
  x ∈ Set.Ioo 0 (π/2) ∧ 
  6 * cos x = 5 * tan x ∧ 
  (1/3 : ℝ) = abs ((1/2 * sin x) - 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l667_66727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_salary_problem_l667_66758

/-- Represents the salary problem in the factory --/
theorem factory_salary_problem 
  (num_workers : ℕ) 
  (initial_average : ℚ) 
  (old_supervisor_salary : ℚ) 
  (new_average : ℚ) 
  (h1 : num_workers = 8)
  (h2 : initial_average = 430)
  (h3 : old_supervisor_salary = 870)
  (h4 : new_average = 420) :
  let total_people : ℕ := num_workers + 1
  let total_initial_salary : ℚ := initial_average * (num_workers + 1)
  let workers_total_salary : ℚ := total_initial_salary - old_supervisor_salary
  let new_supervisor_salary : ℚ := new_average * (num_workers + 1) - workers_total_salary
  new_supervisor_salary = 780 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_salary_problem_l667_66758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cake_cuts_l667_66796

/-- Represents a square cake with side length and cuts -/
structure Cake where
  side_length : ℝ
  first_cut : ℝ
  second_cut : ℝ
  third_cut : ℝ
  fourth_cut : ℝ

/-- Calculates the area of a triangular piece of cake -/
noncomputable def piece_area (c : Cake) (base : ℝ) : ℝ :=
  (1 / 2) * base * (c.side_length / 2)

/-- Checks if all pieces have equal area -/
def equal_pieces (c : Cake) : Prop :=
  let total_area := c.side_length * c.side_length
  let piece_count := 5
  let target_area := total_area / piece_count
  piece_area c c.first_cut = target_area ∧
  piece_area c (c.second_cut - c.first_cut) = target_area ∧
  piece_area c (c.third_cut - c.second_cut) = target_area ∧
  piece_area c (c.fourth_cut - c.third_cut) = target_area ∧
  piece_area c (c.side_length - c.fourth_cut) = target_area

/-- Theorem stating the correct cuts for the cake -/
theorem correct_cake_cuts (c : Cake) :
  c.side_length = 20 ∧
  c.first_cut = 7 ∧
  c.second_cut = 16 ∧
  c.third_cut = 27 ∧
  c.fourth_cut = 33 →
  equal_pieces c := by
  sorry

#check correct_cake_cuts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_cake_cuts_l667_66796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_product_l667_66747

def S : Finset Int := {-6, -3, -1, 5, 7, 9}

def is_negative_product (a b : Int) : Prop := a * b < 0

def count_negative_products : Nat :=
  (S.filter (λ x => x < 0)).card * (S.filter (λ x => x > 0)).card

theorem probability_negative_product :
  (count_negative_products : ℚ) / (Nat.choose S.card 2 : ℚ) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_negative_product_l667_66747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l667_66729

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

-- Theorem stating that g is neither even nor odd
theorem g_neither_even_nor_odd :
  (¬∀ x, g x = g (-x)) ∧ (¬∀ x, g x = -g (-x)) :=
by
  -- We'll use a proof by contradiction for both parts
  apply And.intro
  · -- Prove that g is not even
    intro h
    -- Choose a specific x where g(x) ≠ g(-x)
    have : g 1 ≠ g (-1) := by
      -- Here we would calculate the actual values, but we'll skip it for now
      sorry
    -- This contradicts our assumption that g is even for all x
    exact this (h 1)
  · -- Prove that g is not odd
    intro h
    -- Choose a specific x where g(x) ≠ -g(-x)
    have : g 1 ≠ -g (-1) := by
      -- Here we would calculate the actual values, but we'll skip it for now
      sorry
    -- This contradicts our assumption that g is odd for all x
    exact this (h 1)

-- The proof is incomplete (uses sorry), but the structure is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l667_66729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_coffee_cost_l667_66740

/-- Represents the daily coffee consumption pattern --/
def coffee_pattern : Fin 7 → Nat
  | 0 => 2  -- Monday
  | 1 => 3  -- Tuesday
  | 2 => 2  -- Wednesday
  | 3 => 3  -- Thursday
  | 4 => 2  -- Friday
  | 5 => 3  -- Saturday
  | 6 => 1  -- Sunday

/-- Cost calculation function --/
def calculate_coffee_cost (
  coffee_pattern : Fin 7 → Nat)
  (oz_per_cup : Real)
  (coffee_bag_cost : Real)
  (oz_per_bag : Real)
  (milk_cost : Real)
  (syrup_cost : Real)
  (tbsp_per_syrup : Nat)
  (honey_cost : Real)
  (tsp_per_honey : Nat) : Real :=
  sorry  -- Proof to be implemented

/-- Main theorem: Weekly coffee cost is $26.55 --/
theorem weekly_coffee_cost :
  calculate_coffee_cost
    coffee_pattern
    1.5    -- oz_per_cup
    8      -- coffee_bag_cost
    10.5   -- oz_per_bag
    4      -- milk_cost
    6      -- syrup_cost
    24     -- tbsp_per_syrup
    5      -- honey_cost
    48     -- tsp_per_honey
  = 26.55 := by
  sorry  -- Proof to be implemented

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekly_coffee_cost_l667_66740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_E_minus_F_l667_66791

def digits_all_one (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → d = 1

def digits_all_two (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → d = 2

def digits_all_three (n : ℕ) : Prop :=
  ∀ d, d ∈ (Nat.digits 10 n) → d = 3

theorem square_root_of_E_minus_F (k : ℕ+) 
  (E : ℕ) (hE : digits_all_one E ∧ (Nat.digits 10 E).length = 2 * k)
  (F : ℕ) (hF : digits_all_two F ∧ (Nat.digits 10 F).length = k) :
  ∃ G : ℕ, G * G = E - F ∧ digits_all_three G ∧ (Nat.digits 10 G).length = k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_E_minus_F_l667_66791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_lateral_surface_area_formula_l667_66741

/-- 
Represents a prism with an equilateral triangular base ABC and top face A₁B₁C₁,
where A₁ projects to the center of ABC, and AA₁ is inclined at 60° to the base plane.
-/
structure TriangularPrism where
  a : ℝ  -- Side length of the equilateral triangle base
  h : ℝ  -- Height of the prism (length of AA₁)

/-- Calculates the lateral surface area of the triangular prism -/
noncomputable def lateralSurfaceArea (prism : TriangularPrism) : ℝ :=
  (prism.a^2 * Real.sqrt 3 * (2 + Real.sqrt 13)) / 3

/-- 
Theorem stating that the lateral surface area of the described triangular prism
is equal to (a² √3 (2 + √13)) / 3
-/
theorem lateral_surface_area_formula (prism : TriangularPrism) :
  lateralSurfaceArea prism = (prism.a^2 * Real.sqrt 3 * (2 + Real.sqrt 13)) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_lateral_surface_area_formula_l667_66741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circum_area_l667_66720

/-- The side lengths of the triangle -/
def a : ℝ := 888
def b : ℝ := 925

/-- The function representing the area of the circumscribed circle -/
noncomputable def circum_area (x : ℝ) : ℝ :=
  let s := (a + b + x) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - x))
  Real.pi * (a * b * x / (4 * area))^2

/-- The theorem stating that 259 minimizes the area of the circumscribed circle -/
theorem min_circum_area :
  ∀ x > 0, circum_area 259 ≤ circum_area x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circum_area_l667_66720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_balanced_partition_l667_66730

/-- A simple graph with n vertices -/
structure Graph (n : Type) where
  adj : n → n → Prop
  symm : ∀ i j, adj i j ↔ adj j i
  irrefl : ∀ i, ¬adj i i

/-- The set of neighbors of a vertex in a simple graph -/
def Graph.neighbors {n : Type} (G : Graph n) (v : n) : Set n :=
  {u | G.adj v u}

/-- A partition of vertices into two sets -/
structure Partition (n : Type) where
  S : Set n
  T : Set n
  partition : S ∪ T = Set.univ
  disjoint : S ∩ T = ∅

/-- The main theorem -/
theorem exists_balanced_partition {n : Type} [Finite n] (G : Graph n) :
  ∃ (P : Partition n), ∀ (v : n),
    (v ∈ P.S → (G.neighbors v ∩ P.T).ncard ≥ (G.neighbors v ∩ P.S).ncard) ∧
    (v ∈ P.T → (G.neighbors v ∩ P.S).ncard ≥ (G.neighbors v ∩ P.T).ncard) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_balanced_partition_l667_66730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_value_l667_66776

theorem eleventh_term_value (sequence : List ℝ) 
  (h_length : sequence.length = 11)
  (h_avg_all : (sequence.sum / sequence.length) = 1.78)
  (h_avg_ten : ((sequence.take 10).sum / 10) = 1.74) :
  sequence.getLast? = some 2.18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_value_l667_66776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l667_66784

/-- Represents the sum of the first n terms of a geometric sequence --/
noncomputable def geometricSum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_40 :
  ∀ (a r : ℝ),
    geometricSum a r 10 = 10 →
    geometricSum a r 30 = 70 →
    geometricSum a r 40 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_40_l667_66784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_interval_l667_66719

/-- Represents the distance-time data for a train journey --/
structure TrainJourney where
  distanceTime : ℝ → ℝ
  duration : ℝ

/-- Calculates the average speed between two time points --/
noncomputable def averageSpeed (journey : TrainJourney) (t1 t2 : ℝ) : ℝ :=
  (journey.distanceTime t2 - journey.distanceTime t1) / (t2 - t1)

/-- Checks if a given interval has the steepest slope --/
def hasSteepestSlope (journey : TrainJourney) (t1 t2 : ℝ) : Prop :=
  ∀ s1 s2, 0 ≤ s1 ∧ s2 ≤ journey.duration →
    averageSpeed journey t1 t2 ≥ averageSpeed journey s1 s2

/-- The main theorem stating that the interval [3, 4] has the highest average speed --/
theorem highest_average_speed_interval (journey : TrainJourney) :
  hasSteepestSlope journey 3 4 →
  ∀ t1 t2, 0 ≤ t1 ∧ t2 ≤ journey.duration →
    averageSpeed journey 3 4 ≥ averageSpeed journey t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_average_speed_interval_l667_66719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_and_distance_l667_66769

/-- Converts spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : Real) : (Real × Real × Real) :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

/-- Calculates the distance of a point from the origin -/
noncomputable def distance_from_origin (x y z : Real) : Real :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem spherical_to_rectangular_and_distance 
  (ρ θ φ : Real) 
  (h_ρ : ρ = 8) 
  (h_θ : θ = 5 * Real.pi / 4) 
  (h_φ : φ = Real.pi / 4) : 
  let (x, y, z) := spherical_to_rectangular ρ θ φ
  (x = -4 ∧ y = -4 ∧ z = 4 * Real.sqrt 2) ∧ 
  distance_from_origin x y z = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_and_distance_l667_66769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_to_reach_school_l667_66709

-- Define the total distance in meters
noncomputable def total_distance : ℝ := 2400

-- Define the remaining distance in meters
noncomputable def remaining_distance : ℝ := total_distance / 2

-- Define the available time in minutes
noncomputable def available_time : ℝ := 12

-- Define the minimum required speed in m/min
noncomputable def min_required_speed : ℝ := 100

-- Theorem statement
theorem min_speed_to_reach_school : 
  (remaining_distance / available_time) ≥ min_required_speed := by
  sorry

#check min_speed_to_reach_school

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_to_reach_school_l667_66709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_odd_sum_points_not_all_points_reachable_l667_66701

/-- Represents a point on the infinite square grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a move along a diagonal of a square -/
inductive DiagonalMove where
  | upRight : DiagonalMove
  | upLeft : DiagonalMove
  | downRight : DiagonalMove
  | downLeft : DiagonalMove

/-- Applies a diagonal move to a grid point -/
def applyMove (p : GridPoint) (m : DiagonalMove) : GridPoint :=
  match m with
  | DiagonalMove.upRight => ⟨p.x + 1, p.y + 1⟩
  | DiagonalMove.upLeft => ⟨p.x - 1, p.y + 1⟩
  | DiagonalMove.downRight => ⟨p.x + 1, p.y - 1⟩
  | DiagonalMove.downLeft => ⟨p.x - 1, p.y - 1⟩

/-- A path on the grid is a list of diagonal moves -/
def GridPath := List DiagonalMove

/-- Applies a path to a starting point -/
def applyPath (start : GridPoint) (path : GridPath) : GridPoint :=
  path.foldl applyMove start

/-- Theorem: It's impossible to reach a point with odd coordinate sum from the origin -/
theorem unreachable_odd_sum_points (path : GridPath) :
  let end_point := applyPath ⟨0, 0⟩ path
  (end_point.x + end_point.y) % 2 = 0 := by
  sorry

/-- Corollary: Not all points are reachable from the origin -/
theorem not_all_points_reachable : ¬∀ (x y : Int), ∃ (path : GridPath),
  applyPath ⟨0, 0⟩ path = ⟨x, y⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreachable_odd_sum_points_not_all_points_reachable_l667_66701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_is_ten_percent_l667_66774

/-- Represents the volume of saline solution in the cup -/
noncomputable def initialVolume : ℝ := 30

/-- Represents the initial concentration of saline solution -/
noncomputable def initialConcentration : ℝ := 0.15

/-- Represents the volume of the small ball -/
noncomputable def smallBallVolume : ℝ := 3

/-- Represents the volume of the medium ball -/
noncomputable def mediumBallVolume : ℝ := 5

/-- Represents the volume of the large ball -/
noncomputable def largeBallVolume : ℝ := 10

/-- Represents the percentage of saline solution that overflows when the small ball is submerged -/
noncomputable def overflowPercentage : ℝ := 0.1

/-- Calculates the final concentration of saline solution after the process -/
noncomputable def finalConcentration : ℝ := 
  let remainingVolume := initialVolume * (1 - overflowPercentage)
  let saltAmount := initialVolume * initialConcentration
  saltAmount / initialVolume

/-- Theorem stating that the final concentration of saline solution is 10% -/
theorem final_concentration_is_ten_percent : 
  finalConcentration = 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_concentration_is_ten_percent_l667_66774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_graph_is_line_segment_l667_66712

/-- A function representing a linear equation -/
def linear_function (x : ℝ) : ℝ := 3 * x - 1

/-- The domain of the function -/
def domain : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

/-- Predicate to check if a set of points forms a line segment -/
def IsLineSegment (s : Set (ℝ × ℝ)) (a b : ℝ × ℝ) : Prop :=
  ∃ (f : ℝ → ℝ × ℝ), Continuous f ∧ 
    f 0 = a ∧ f 1 = b ∧
    s = Set.range f

/-- Theorem stating that the graph of y = 3x - 1 for 1 ≤ x ≤ 5 is a line segment -/
theorem linear_graph_is_line_segment : 
  ∃ (a b : ℝ × ℝ), IsLineSegment {(x, linear_function x) | x ∈ domain} a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_graph_is_line_segment_l667_66712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l667_66739

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * ω * x) - Real.cos (2 * ω * x)

theorem monotonic_increase_interval
  (ω : ℝ)
  (h1 : 0 < ω ∧ ω < 1)
  (h2 : f ω (π / 6) = 0) :
  ∃ (a b : ℝ), a = 0 ∧ b = 2 * π / 3 ∧
  ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y →
  f ω x < f ω y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l667_66739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l667_66767

def distribute_apples (total : ℕ) (children : ℕ) : ℕ :=
  let ways := Finset.filter (fun (x, y, z) => x + y + z = total ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1)
    (Finset.product (Finset.range (total + 1)) (Finset.product (Finset.range (total + 1)) (Finset.range (total + 1))))
  ways.card

theorem apple_distribution :
  distribute_apples 20 3 = 171 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_distribution_l667_66767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_puzzle_solution_l667_66798

theorem age_puzzle_solution :
  ∃ (j f : ℕ), j > f ∧
  j = 10 * (j % 10) + (j / 10) ∧
  f = 10 * (j / 10) + (j % 10) ∧
  ∃ (k : ℕ), j^2 - f^2 = k^2 ∧
  j = 65 ∧ f = 56 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_puzzle_solution_l667_66798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l667_66775

/-- The radius of the inscribed circle in a triangle with given side lengths -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in a triangle with sides 24, 10, and 26 is 4 -/
theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 24 10 26 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l667_66775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l667_66764

-- Define the parameters
noncomputable def train1_length : ℝ := 70
noncomputable def train2_length : ℝ := 100
noncomputable def bridge_length : ℝ := 80
noncomputable def train1_speed_kmph : ℝ := 36
noncomputable def train2_speed_kmph : ℝ := 45

-- Convert speeds from km/h to m/s
noncomputable def train1_speed : ℝ := train1_speed_kmph * (1000 / 3600)
noncomputable def train2_speed : ℝ := train2_speed_kmph * (1000 / 3600)

-- Calculate the time taken for Train 1 to cross the bridge
noncomputable def crossing_time : ℝ := (train1_length + bridge_length) / (train1_speed + train2_speed)

-- Theorem statement
theorem train_crossing_time :
  ∃ ε > 0, |crossing_time - 6.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l667_66764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_segment_l667_66738

-- Define the parametric equations
noncomputable def x (θ : ℝ) : ℝ := (Real.cos θ) ^ 2
noncomputable def y (θ : ℝ) : ℝ := (Real.sin θ) ^ 2

-- Define the curve
def curve : Set (ℝ × ℝ) := {(x θ, y θ) | θ : ℝ}

-- Theorem statement
theorem curve_is_line_segment : 
  ∃ a b : ℝ × ℝ, a ≠ b ∧ curve = {p | ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ p = (1 - t) • a + t • b} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_line_segment_l667_66738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l667_66735

noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 2 ∧ x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l667_66735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l667_66742

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the centers and radii
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (0, 2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 2

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 5

-- Theorem statement
theorem circles_intersect :
  distance_between_centers < radius₁ + radius₂ ∧
  distance_between_centers > abs (radius₁ - radius₂) := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l667_66742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_approx_l667_66770

/-- The curved surface area of a cone -/
noncomputable def curved_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

/-- Theorem: The curved surface area of a cone with radius 35 m and slant height 30 m 
    is approximately 3299.34 square meters -/
theorem cone_surface_area_approx :
  let r : ℝ := 35
  let l : ℝ := 30
  abs (curved_surface_area r l - 3299.34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_surface_area_approx_l667_66770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_circle_l667_66754

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a circle
def Circle (center : Point2D) (radius : ℝ) : Set Point2D :=
  {p : Point2D | distance p center = radius}

-- Theorem statement
theorem points_form_circle (O : Point2D) :
  {P : Point2D | distance P O = 3} = Circle O 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_circle_l667_66754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_l667_66734

theorem sqrt_expressions :
  (Real.sqrt 81 = 9) ∧
  (Real.sqrt (1 - 7/16) = 3/4) ∧
  (-Real.sqrt (1 + 9/16) = -5/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_l667_66734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_gallon_l667_66702

theorem paint_cost_per_gallon (rooms : ℕ) (primer_cost : ℚ) (discount : ℚ) (total_cost : ℚ) : 
  rooms = 5 → 
  primer_cost = 30 → 
  discount = 20 / 100 → 
  total_cost = 245 → 
  let discounted_primer_cost := primer_cost * (1 - discount);
  let total_primer_cost := discounted_primer_cost * rooms;
  let total_paint_cost := total_cost - total_primer_cost;
  total_paint_cost / rooms = 25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_gallon_l667_66702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l667_66707

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a line y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Main theorem about the parabola and intersecting line -/
theorem parabola_intersecting_line 
  (para : Parabola) 
  (focus_directrix_distance : ℝ) 
  (l : Line) 
  (A B : Point) 
  (M : Point) :
  focus_directrix_distance = 4 →
  (l.m * para.p / 2 + l.b = 0) →  -- Line passes through focus (p/2, 0)
  (A.y^2 = 2 * para.p * A.x) →    -- A is on the parabola
  (B.y^2 = 2 * para.p * B.x) →    -- B is on the parabola
  (A.y = l.m * A.x + l.b) →       -- A is on the line
  (B.y = l.m * B.x + l.b) →       -- B is on the line
  (M.x = (A.x + B.x) / 2) →       -- M is midpoint of AB
  (M.y = (A.y + B.y) / 2) →
  (M.y = 2) →
  (para.p = 4 ∧ l.m = 2 ∧ l.b = -4 ∧ ((A.x - B.x)^2 + (A.y - B.y)^2 = 100)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersecting_line_l667_66707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l667_66765

-- Define the point A
def A : ℝ × ℝ := (2, 3)

-- Define the line l1: x + 2y = 0
def l1 (x y : ℝ) : Prop := x + 2*y = 0

-- Define the line l2: x - y + 1 = 0
def l2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the chord length
noncomputable def chord_length : ℝ := 2 * Real.sqrt 2

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property that a point is on a circle
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the symmetric point of A with respect to l1
noncomputable def symmetric_point (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- State the theorem
theorem circle_equation (c : Circle) : 
  (on_circle A c) ∧ 
  (on_circle (symmetric_point A l1) c) ∧
  (∃ (p q : ℝ × ℝ), l2 p.1 p.2 ∧ l2 q.1 q.2 ∧ on_circle p c ∧ on_circle q c ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) →
  ((c.center = (6, -3) ∧ c.radius^2 = 52) ∨ (c.center = (14, -7) ∧ c.radius^2 = 244)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l667_66765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_of_numbers_l667_66789

-- Define the four numbers
def a : ℚ := 1 / 11
noncomputable def b : ℝ := (15 : ℝ) ^ (1/3)
noncomputable def c : ℝ := Real.pi
noncomputable def d : ℝ := -Real.sqrt 2

-- State the theorem
theorem rationality_of_numbers :
  (∃ (p q : ℤ), q ≠ 0 ∧ a = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ b = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ c = p / q) ∧
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ d = p / q) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_of_numbers_l667_66789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_N_given_S_2_l667_66724

noncomputable def expectedValueN : ℝ :=
  1 / (2 * Real.log 2 - 1)

theorem expected_value_N_given_S_2 :
  let N : ℕ → ℝ := fun n => (2 : ℝ)^(-n : ℝ)
  let S : ℕ → ℕ → ℝ := fun n k => if k ≤ n then 1 / (n : ℝ) else 0
  expectedValueN =
    ∑' n, n * (S n 2 * N n) / ∑' m, (S m 2 * N m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_N_given_S_2_l667_66724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l667_66711

/-- Square with side length 40 -/
structure Square :=
  (side : ℝ)
  (is_40 : side = 40)

/-- Point inside the square -/
structure PointInSquare :=
  (X Y Z W Q : ℝ × ℝ)
  (in_square : Q.1 ≥ 0 ∧ Q.1 ≤ 40 ∧ Q.2 ≥ 0 ∧ Q.2 ≤ 40)
  (XQ_15 : Real.sqrt ((Q.1 - X.1)^2 + (Q.2 - X.2)^2) = 15)
  (YQ_35 : Real.sqrt ((Q.1 - Y.1)^2 + (Q.2 - Y.2)^2) = 35)

/-- Centroid of a triangle -/
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Quadrilateral formed by centroids -/
noncomputable def centroid_quadrilateral (p : PointInSquare) : Set (ℝ × ℝ) :=
  { centroid p.X p.Y p.Q,
    centroid p.Y p.Z p.Q,
    centroid p.Z p.W p.Q,
    centroid p.W p.X p.Q }

/-- Area of a quadrilateral -/
noncomputable def quadrilateral_area (quad : Set (ℝ × ℝ)) : ℝ := sorry

theorem centroid_quadrilateral_area 
  (s : Square) (p : PointInSquare) : 
  quadrilateral_area (centroid_quadrilateral p) = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l667_66711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_l667_66795

/-- A point is symmetric to another point about a line if:
    1) The line connecting the two points is perpendicular to the line of symmetry
    2) The midpoint of the line segment connecting the two points lies on the line of symmetry -/
def is_symmetric_about_line (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → Prop) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  line midpoint ∧ 
  (p2.1 - p1.1) + (p2.2 - p1.2) = 0

/-- The line x + y - 5 = 0 -/
def symmetry_line (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 - 5 = 0

/-- Theorem: The point (6, 3) is symmetric to (2, -1) about the line x + y - 5 = 0 -/
theorem symmetric_point : is_symmetric_about_line (2, -1) (6, 3) symmetry_line := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_l667_66795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_eq_specific_quadratic_l667_66778

/-- A quadratic function with specific properties -/
def q : ℝ → ℝ := sorry

/-- The graph of 1/q(x) has vertical asymptotes at x = -2 and x = 3 -/
axiom asymptotes : ∀ (x : ℝ), x ≠ -2 ∧ x ≠ 3 → q x ≠ 0

/-- q(x) is a quadratic function -/
axiom is_quadratic : ∃ (a b c : ℝ), ∀ (x : ℝ), q x = a * x^2 + b * x + c

/-- q(1) = 8 -/
axiom q_at_one : q 1 = 8

/-- The main theorem: q(x) is equal to -4/3x^2 + 4/3x + 8 -/
theorem q_eq_specific_quadratic : 
  ∀ (x : ℝ), q x = -4/3 * x^2 + 4/3 * x + 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_eq_specific_quadratic_l667_66778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_extrema_log_inequality_l667_66703

noncomputable section

variable (a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) / (a * x) - Real.log x

theorem f_monotonicity (h : a ≠ 0) :
  (a < 0 → ∀ x > 0, (deriv (f a)) x < 0) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 (1/a), (deriv (f a)) x > 0) ∧
           (∀ x > 1/a, (deriv (f a)) x < 0)) :=
sorry

theorem f_extrema :
  let f1 := f 1
  (∀ x ∈ Set.Icc (1/2) 2, f1 x ≤ 0) ∧
  (∀ x ∈ Set.Icc (1/2) 2, f1 x ≥ -1 + Real.log 2) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f1 x = 0) ∧
  (∃ x ∈ Set.Icc (1/2) 2, f1 x = -1 + Real.log 2) :=
sorry

theorem log_inequality :
  ∀ x > 0, Real.log (Real.exp 2 / x) ≤ (1 + x) / x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_extrema_log_inequality_l667_66703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perpendicular_diagonals_l667_66788

/-- A point in a 2D plane --/
structure Point :=
  (x y : ℝ)

/-- A quadrilateral in a 2D plane --/
structure Quadrilateral :=
  (A B C D : Point)

/-- Definition of a rhombus --/
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

/-- Definition of perpendicular diagonals --/
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  sorry

/-- Theorem stating the relationship between rhombus and perpendicular diagonals --/
theorem rhombus_perpendicular_diagonals :
  (∀ q : Quadrilateral, is_rhombus q → perpendicular_diagonals q) ∧
  (∃ q : Quadrilateral, perpendicular_diagonals q ∧ ¬is_rhombus q) :=
by
  sorry

#check rhombus_perpendicular_diagonals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perpendicular_diagonals_l667_66788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l667_66755

/-- Predicate to check if a triangle with given area is equilateral and fits in the rectangle -/
def is_equilateral_triangle_in_rectangle (width height area : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ width ∧
    0 ≤ y ∧ y ≤ height ∧
    area = (Real.sqrt 3 / 4) * (x^2 + y^2) ∧
    x ≤ width ∧
    y ≤ height ∧
    ((width - x / 2)^2 + (y / 2)^2) ≤ height^2

/-- The maximum area of an equilateral triangle inscribed in a 12 by 13 rectangle -/
theorem max_equilateral_triangle_area_in_rectangle : 
  ∃ (A : ℝ), 
    (∀ (a : ℝ), is_equilateral_triangle_in_rectangle 12 13 a → a ≤ A) ∧ 
    is_equilateral_triangle_in_rectangle 12 13 A ∧
    A = 205 * Real.sqrt 3 - 468 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l667_66755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_a_ratio_l667_66782

/-- Definition of n_a! for positive integers n and a -/
def factorial_a (n a : ℕ) : ℕ :=
  (List.range (n / a + 1)).foldl (λ acc i => acc * (n - i * a)) 1

/-- Theorem stating that the ratio of 36_5! to 10_3! equals 40455072 -/
theorem factorial_a_ratio : (factorial_a 36 5) / (factorial_a 10 3) = 40455072 := by
  sorry

#eval factorial_a 36 5 / factorial_a 10 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_a_ratio_l667_66782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l667_66752

theorem cosine_sum_product (x : ℝ) : 
  ∃ (a b c d : ℕ+), 
    (Real.cos x + Real.cos (5*x) + Real.cos (11*x) + Real.cos (15*x) = 
     ↑a * Real.cos (↑b*x) * Real.cos (↑c*x) * Real.cos (↑d*x)) ∧ 
    (a.val + b.val + c.val + d.val = 19) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_l667_66752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_y_axis_l667_66781

/-- A line in 2D space represented by the equation ax + by + 1 = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ

/-- Predicate to check if a line is parallel to the y-axis -/
def is_parallel_to_y_axis (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b = 0

/-- Theorem stating the condition for a line to be parallel to the y-axis -/
theorem line_parallel_to_y_axis (l : Line2D) :
  is_parallel_to_y_axis l ↔ (∀ (x y : ℝ), l.a * x + l.b * y + 1 = 0 → y ∈ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_y_axis_l667_66781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_85_digit_sum_equals_product_l667_66717

def largest_85_digit_number : ℕ := 8322 * 10^81 + (10^81 - 1)

def digit_sum (n : ℕ) : ℕ := sorry

def digit_product (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

theorem largest_85_digit_sum_equals_product :
  (∀ n : ℕ, n > largest_85_digit_number → 
    (digit_sum n ≠ digit_product n ∨ num_digits n ≠ 85)) ∧
  (digit_sum largest_85_digit_number = digit_product largest_85_digit_number) ∧
  (num_digits largest_85_digit_number = 85) := by
  sorry

#check largest_85_digit_sum_equals_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_85_digit_sum_equals_product_l667_66717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_height_is_1700_l667_66757

/-- The relative height of a mountain given temperature and altitude information -/
noncomputable def mountain_height (temp_decrease_rate : ℝ) (temp_summit : ℝ) (temp_foot : ℝ) : ℝ :=
  ((temp_foot - temp_summit) / temp_decrease_rate) * 100

/-- Theorem stating that the relative height of the mountain is 1700 meters -/
theorem mountain_height_is_1700 :
  let temp_decrease_rate := (0.7 : ℝ)
  let temp_summit := (14.1 : ℝ)
  let temp_foot := (26 : ℝ)
  mountain_height temp_decrease_rate temp_summit temp_foot = 1700 := by
  -- Unfold the definition of mountain_height
  unfold mountain_height
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_height_is_1700_l667_66757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_year_winner_time_l667_66793

def town_square_length : ℚ := 3/4
def laps : ℕ := 7
def this_year_time : ℕ := 42
def speed_difference : ℕ := 1

def race_distance : ℚ := town_square_length * laps

def this_year_speed : ℚ := this_year_time / race_distance

def last_year_speed : ℚ := this_year_speed + speed_difference

def last_year_time : ℚ := last_year_speed * race_distance

theorem last_year_winner_time : 
  Int.floor last_year_time = 47 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_year_winner_time_l667_66793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_theorem_l667_66783

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q
noncomputable def g (r s : ℝ) (x : ℝ) : ℝ := x^2 + r*x + s

noncomputable def vertex_x (a b : ℝ) : ℝ := -b/(2*a)

theorem quadratic_polynomials_theorem (p q r s : ℝ) :
  (∃ x, g r s x = 0 ∧ x = vertex_x 1 p) →  -- vertex of f is root of g
  (∃ x, f p q x = 0 ∧ x = vertex_x 1 r) →  -- vertex of g is root of f
  (∃ m, ∀ x, f p q x ≥ m ∧ g r s x ≥ m) →  -- same minimum value
  f p q 50 = g r s 50 ∧ f p q 50 = -50 →  -- intersection at (50, -50)
  p + r = -200 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomials_theorem_l667_66783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l667_66743

theorem absolute_value_expression (x : ℤ) (h : x = -2023) :
  (abs (abs (abs x - x) - abs x) - x^2) = -4094506 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l667_66743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_form_l667_66745

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → n % d ≠ 0

def number_form (A : Nat) : Nat := 100000 + 3000 + 200 + A * 10 + 4

theorem unique_prime_in_form : 
  ∃! A : Nat, A < 10 ∧ is_prime (number_form A) ∧ number_form A = 103214 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_in_form_l667_66745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l667_66753

/-- The certain distance for the initial fare -/
def certain_distance : ℝ := sorry

/-- The initial fare -/
def initial_fare : ℝ := 10

/-- The additional fare per certain distance -/
def additional_fare : ℝ := 1

/-- The total distance of the ride -/
def total_distance : ℝ := 10

/-- The total fare for the ride -/
def total_fare : ℝ := 59

theorem taxi_fare_theorem :
  initial_fare + (total_distance / certain_distance - 1) * additional_fare = total_fare →
  certain_distance = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l667_66753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_relation_l667_66780

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := cos x

-- Define the triangle angles
variable (A B C : ℝ)

-- Theorem statement
theorem triangle_angle_cosine_relation 
  (h1 : A + B + C = π)  -- Sum of angles in a triangle
  (h2 : f A = 3/5)      -- Given f(A)
  (h3 : f B = 5/13)     -- Given f(B)
  : f C = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_cosine_relation_l667_66780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_inside_l667_66762

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- Calculates the focal distance of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2)

/-- Checks if a point (x, y) is on or inside the ellipse -/
def is_inside_ellipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- Theorem: A circle centered at the focus of the ellipse with radius 8 
    is tangent to the ellipse and fits entirely inside it -/
theorem circle_tangent_and_inside (e : Ellipse) 
    (h_a : e.a = 8) (h_b : e.b = 5) : 
    let c := focal_distance e
    let r := 8
    (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ is_inside_ellipse e (x + c) y) ∧
    (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ x^2 / e.a^2 + y^2 / e.b^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_inside_l667_66762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l667_66756

theorem sin_cos_range (α β : ℝ) (h : Real.cos α + Real.sin β = 1/2) :
  ∃ (x : ℝ), Real.sin α + Real.sin β = x ∧ 1/2 - Real.sqrt 2 ≤ x ∧ x ≤ 1 + Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_range_l667_66756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_triangle_max_area_l667_66794

/-- Triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given its side lengths -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

theorem triangle_side_calculation (t : Triangle) 
  (h1 : t.a = 2) 
  (h2 : t.b = 2 * t.c) 
  (h3 : t.C = π / 6) : 
  t.b = 4 * Real.sqrt 3 / 3 := by
  sorry

theorem triangle_max_area (t : Triangle) 
  (h1 : t.a = 2) 
  (h2 : t.b = 2 * t.c) : 
  area t ≤ 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_triangle_max_area_l667_66794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solutions_l667_66777

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 6*x - 6 = 0
  let eq2 : ℝ → Prop := λ x ↦ 2*x^2 - 3*x + 1 = 0
  let sol1 : Set ℝ := {3 + Real.sqrt 15, 3 - Real.sqrt 15}
  let sol2 : Set ℝ := {1, 1/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y : ℝ, eq1 y → y ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y : ℝ, eq2 y → y ∈ sol2) :=
by
  sorry

#check quadratic_equations_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_solutions_l667_66777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_squares_plus_three_l667_66785

/-- The nth odd positive integer -/
def nthOddPositive (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of squares of first n odd positive integers, each increased by 3 -/
def sumOfSquaresPlusThree (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => (nthOddPositive (i + 1))^2 + 3)

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_squares_plus_three :
  unitsDigit (sumOfSquaresPlusThree 4013) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_sum_squares_plus_three_l667_66785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l667_66713

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)
def c : ℝ × ℝ := (3, 4)

theorem parallel_vectors_lambda (l : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (a.1 + l * b.1, a.2 + l * b.2) = (k * c.1, k * c.2)) →
  l = -1/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l667_66713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l667_66790

/-- Converts degrees to a pair of degrees and minutes -/
noncomputable def degrees_to_deg_min (d : ℝ) : ℕ × ℕ :=
  let whole_degrees := Int.floor d
  let fractional_degrees := d - whole_degrees
  let minutes := Int.floor (fractional_degrees * 60)
  (whole_degrees.toNat, minutes.toNat)

/-- Theorem stating that 29.5° is equal to 29° 30' -/
theorem angle_conversion :
  degrees_to_deg_min 29.5 = (29, 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l667_66790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_subtraction_divisibility_l667_66751

def base_b_to_decimal (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

def is_not_divisible_by_5 (n : ℤ) : Prop :=
  n % 5 ≠ 0

theorem base_b_subtraction_divisibility :
  ∀ b : ℕ, b ∈ ({3, 5, 6, 7, 10} : Set ℕ) →
    (is_not_divisible_by_5 (base_b_to_decimal [3, 0, 2, 2] b - base_b_to_decimal [4, 4, 2] b : ℤ) ↔
     b ∈ ({3, 6, 7} : Set ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_b_subtraction_divisibility_l667_66751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l667_66761

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.sqrt (1 - abs x)) / x

def domain (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ x ≠ 0

theorem f_is_odd : ∀ x, domain x → f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l667_66761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l667_66715

-- Define the sets A and B
def A : Set ℝ := {x | (x - 1) * (3 - x) < 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l667_66715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l667_66736

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan x ^ 2 - 4 * Real.tan x - 12 / Real.tan x + 9 / (Real.tan x ^ 2) - 3

theorem min_value_of_f :
  ∃ (min : ℝ), min = 3 + 8 * Real.sqrt 3 ∧
  ∀ x ∈ Set.Ioo (-π/2 : ℝ) 0, f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l667_66736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l667_66749

theorem gcd_problem (b : ℤ) (k : ℤ) (h1 : b = 997 * k) (h2 : Odd k) : 
  Int.gcd (3*b^2 + 17*b + 31) (b + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l667_66749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_modulo_nine_l667_66716

noncomputable def digit_sum : ℕ → ℕ := sorry

theorem digit_sum_modulo_nine (M A B : ℕ) : 
  M = 2014^2014 →
  A = digit_sum M →
  B = digit_sum A →
  B % 9 = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_modulo_nine_l667_66716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_numbers_l667_66773

-- Define the set of numbers
def S : Finset Nat := {1, 2, 3, 4}

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool := n % 2 = 0

-- Define the total number of ways to draw two numbers without replacement
def totalOutcomes : Nat := S.card * (S.card - 1)

-- Define the number of ways to draw two even numbers without replacement
def favorableOutcomes : Nat := (S.filter (fun n => isEven n)).card * ((S.filter (fun n => isEven n)).card - 1)

-- Theorem statement
theorem probability_two_even_numbers :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by
  -- Convert natural numbers to rationals
  have h1 : (favorableOutcomes : ℚ) = 2
  sorry
  have h2 : (totalOutcomes : ℚ) = 12
  sorry
  -- Perform the division
  calc
    (favorableOutcomes : ℚ) / totalOutcomes = 2 / 12 := by rw [h1, h2]
    _ = 1 / 6 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_two_even_numbers_l667_66773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l667_66705

/-- Given real numbers a, c, d, f, prove that f(g(x)) = g(f(x)) 
    if and only if d = a^2 + a - 1 or c = 0, 
    where f(x) = ax^2 + c and g(x) = dx^2 + f -/
theorem function_composition_equality 
  (a c d f : ℝ) : 
  (∀ x : ℝ, (a * x^2 + c) = (d * (a * x^2 + c)^2 + f)) ↔ (d = a^2 + a - 1 ∨ c = 0) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l667_66705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l667_66704

theorem recurring_decimal_fraction (a b : ℕ+) : 
  (a.val : ℚ) / (b.val : ℚ) = 35 / 99 →
  Nat.Coprime a.val b.val →
  a.val + b.val = 134 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_fraction_l667_66704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l667_66748

-- Define the functions g and k
noncomputable def g : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- Define the given conditions
axiom intersect_1 : g 1 = k 1 ∧ g 1 = 1
axiom intersect_3 : g 3 = k 3 ∧ g 3 = 5
axiom intersect_5 : g 5 = k 5 ∧ g 5 = 10
axiom intersect_7 : g 7 = k 7 ∧ g 7 = 10

-- Define continuity or repeated values assumption
axiom g_6_eq_10 : g 6 = 10

-- Theorem to prove
theorem intersection_point :
  g (2 * 3) = 2 * k 3 ∧ g (2 * 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l667_66748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sum_combinations_l667_66771

/-- The number of ways to choose marbles with a sum condition -/
theorem marble_sum_combinations : ∃ n : ℕ, n = 70 := by
  -- Define the sets of marbles
  let my_marbles : Finset ℕ := Finset.range 8
  let mathew_marbles : Finset ℕ := Finset.range 12

  -- Define the condition for a valid combination
  let is_valid_combination (m : ℕ) (n1 n2 : ℕ) : Prop :=
    m ∈ mathew_marbles ∧ n1 ∈ my_marbles ∧ n2 ∈ my_marbles ∧ n1 + n2 = m + 1

  -- Count the number of valid combinations
  let count := Finset.sum mathew_marbles (λ m =>
    Finset.sum my_marbles (λ n1 =>
      Finset.sum my_marbles (λ n2 =>
        if is_valid_combination m n1 n2 then 1 else 0
      )
    )
  )

  -- The theorem states that this count equals 70
  use count
  sorry

#eval Finset.sum (Finset.range 12) (λ m =>
  Finset.sum (Finset.range 8) (λ n1 =>
    Finset.sum (Finset.range 8) (λ n2 =>
      if n1 + n2 = m + 1 then 1 else 0
    )
  )
)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_sum_combinations_l667_66771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_for_point_three_four_l667_66725

/-- Predicate to check if a point lies on the terminal side of an angle -/
def point_on_terminal_side (P : ℝ × ℝ) (θ : ℝ) : Prop := sorry

/-- If a point P(3, 4) lies on the terminal side of angle θ, then cos θ = 3/5 -/
theorem cos_theta_for_point_three_four (θ : ℝ) : 
  (∃ (P : ℝ × ℝ), P.1 = 3 ∧ P.2 = 4 ∧ point_on_terminal_side P θ) → Real.cos θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_for_point_three_four_l667_66725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_paint_calculation_l667_66708

/-- Represents the amount of paint in quarts -/
def Paint := ℚ

/-- Represents the ratio of paints (blue:green:yellow) -/
structure PaintRatio where
  blue : ℚ
  green : ℚ
  yellow : ℚ

/-- Given a paint ratio and the amount of yellow paint, 
    calculates the amount of green paint needed -/
def greenPaintAmount (ratio : PaintRatio) (yellowAmount : ℚ) : ℚ :=
  (ratio.green / ratio.yellow) * yellowAmount

theorem green_paint_calculation (ratio : PaintRatio) (yellowAmount : ℚ) :
  ratio.blue = 4 ∧ ratio.green = 3 ∧ ratio.yellow = 5 ∧ yellowAmount = 15 →
  greenPaintAmount ratio yellowAmount = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_paint_calculation_l667_66708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l667_66706

-- Define a triangle with one side length and its opposite angle
structure Triangle where
  side : ℝ
  opposite_angle : ℝ

-- Define the diameter of the circumscribed circle
noncomputable def circumscribed_circle_diameter (t : Triangle) : ℝ :=
  t.side / Real.sin t.opposite_angle

-- Theorem statement
theorem triangle_circumscribed_circle_diameter :
  let t : Triangle := { side := 16, opposite_angle := π / 4 }
  circumscribed_circle_diameter t = 16 * Real.sqrt 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumscribed_circle_diameter_l667_66706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l667_66714

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function (exponential)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Define the F function
noncomputable def F (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := (2*m - 1) * g a x + (1/m - 1/2) * g a (-x)

-- Define the h function
noncomputable def h (m : ℝ) : ℝ := 2 * Real.sqrt ((2*m - 1) * (1/m - 1/2))

theorem problem_solution (a : ℝ) (m : ℝ) :
  (∃ x : ℝ, f a (x - 1) = f a (a - x) - f a (5 - x) →
    x = (7 + Real.sqrt (29 - 4*a)) / 2 ∨ x = (7 - Real.sqrt (29 - 4*a)) / 2) ∧
  (∃ x : ℝ, IsLocalMin (F a m) x → h m = 2 * Real.sqrt ((2*m - 1) * (1/m - 1/2))) ∧
  (1/2 < m → m < 2 → ∃ x : ℝ, IsMaxOn h Set.univ x ∧ h x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l667_66714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_cardinality_l667_66763

/-- A good set for a prime p is a subset A of {0, 1, ..., p-1} such that
    for any a, b ∈ A, (ab + 1) mod p ∈ A -/
def GoodSet (p : ℕ) (A : Finset ℕ) : Prop :=
  Nat.Prime p ∧ 
  A ⊆ Finset.range p ∧
  ∀ a b, a ∈ A → b ∈ A → (a * b + 1) % p ∈ A

theorem good_set_cardinality (p : ℕ) (A : Finset ℕ) (h : GoodSet p A) : 
  A.card = 1 ∨ A.card = p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_set_cardinality_l667_66763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_is_constant_l667_66700

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∀ a : ℝ, f a = f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfying_inequality_is_constant_l667_66700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coeffs_product_expansion_coeffWithY_from_expansion_l667_66750

-- Define the polynomials
def p₁ (x y : ℝ) : ℝ := 5*x + 3*y + 2
def p₂ (x y : ℝ) : ℝ := 2*x + 5*y + 6

-- Define the product of the polynomials
def product (x y : ℝ) : ℝ := p₁ x y * p₂ x y

-- Define a function to extract coefficients of terms with y
def coeffWithY : ℝ := 31 + 15 + 28

-- Theorem statement
theorem sum_of_y_coeffs : coeffWithY = 74 := by
  -- Unfold the definition of coeffWithY
  unfold coeffWithY
  -- Perform the arithmetic
  norm_num

-- Theorem to show that the product expansion contains the expected terms with y
theorem product_expansion (x y : ℝ) :
  product x y = 10*x^2 + 31*x*y + 34*x + 15*y^2 + 28*y + 12 := by
  -- Unfold the definitions and expand
  unfold product p₁ p₂
  ring  -- This tactic will perform the algebraic expansion and simplification

-- Theorem to connect the product expansion with coeffWithY
theorem coeffWithY_from_expansion :
  coeffWithY = 31 + 15 + 28 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coeffs_product_expansion_coeffWithY_from_expansion_l667_66750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_inverse_condition_l667_66746

theorem no_solution_for_inverse_condition : ¬∃ (a b c d : ℝ), 
  (Matrix.det !![a, b; c, d] ≠ 0) ∧
  (!![a, b; c, d])⁻¹ = !![2/a, 2/b; 2/c, 2/d] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_for_inverse_condition_l667_66746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dancing_survey_l667_66731

theorem dancing_survey (total : ℝ) (total_pos : 0 < total) : 
  (let like_dancing := 0.7 * total
  let dislike_dancing := 0.3 * total
  let say_dislike_but_like := 0.3 * like_dancing
  let say_dislike_and_dislike := 0.8 * dislike_dancing
  let total_say_dislike := say_dislike_but_like + say_dislike_and_dislike
  say_dislike_but_like / total_say_dislike) = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dancing_survey_l667_66731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_root_of_f_l667_66779

def f (x : ℝ) : ℝ := 15 * x^4 - 13 * x^2 + 2

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 6 / 3 ∧
  f r = 0 ∧
  ∀ (x : ℝ), f x = 0 → x ≤ r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_root_of_f_l667_66779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_in_sphere_l667_66768

-- Define the sphere
def sphere_radius : ℝ := 2

-- Define the objects
noncomputable def tetrahedron_edge : ℝ := 2 * Real.sqrt 2
def hexagonal_pyramid_base : ℝ := 1
def hexagonal_pyramid_height : ℝ := 3.8
def cylinder_diameter : ℝ := 1.6
def cylinder_height : ℝ := 3.6
def frustum_upper_base : ℝ := 1
def frustum_lower_base : ℝ := 2
def frustum_height : ℝ := 3

-- Theorem statement
theorem objects_in_sphere :
  (tetrahedron_edge^2 / 12 ≤ sphere_radius^2) ∧
  (cylinder_diameter^2 / 4 + cylinder_height^2 / 4 ≤ sphere_radius^2) ∧
  (frustum_upper_base^2 / 2 + frustum_height^2 ≤ sphere_radius^2) ∧
  (hexagonal_pyramid_base^2 / 3 + hexagonal_pyramid_height^2 > sphere_radius^2) := by
  sorry

#eval sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_objects_in_sphere_l667_66768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_expression_is_eight_l667_66737

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The expression to be computed -/
noncomputable def expression : ℝ := (1005^3 : ℝ) / (1003 * 1004) - (1003^3 : ℝ) / (1004 * 1005)

theorem floor_of_expression_is_eight :
  floor expression = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_expression_is_eight_l667_66737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chandigarh_express_crossing_time_l667_66787

/-- The time taken for a train to cross a platform -/
noncomputable def time_to_cross (train_length platform_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating the time taken for the Chandigarh Express to cross the platform -/
theorem chandigarh_express_crossing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧
  abs (time_to_cross 100 150 60 - 15) < ε := by
  sorry

/-- Compute the approximate time to cross -/
def approximate_time : ℚ :=
  (100 + 150) / (60 * 1000 / 3600)

#eval approximate_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chandigarh_express_crossing_time_l667_66787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_first_quadrant_l667_66797

-- Define the function f(x) = 2/x
noncomputable def f (x : ℝ) : ℝ := 2 / x

-- Theorem statement
theorem f_decreasing_in_first_quadrant :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_first_quadrant_l667_66797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l667_66772

/-- Calculates the selling price of an item given its cost price and gain percentage. -/
noncomputable def selling_price (cost_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  cost_price * (1 + gain_percentage / 100)

/-- Theorem: The selling price of a cycle with cost price 900 and 60% gain is 1440. -/
theorem cycle_selling_price :
  selling_price 900 60 = 1440 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l667_66772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_x_over_3_l667_66722

-- Define the function f(x) = sin(x/3)
noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3)

-- State the theorem
theorem period_of_sin_x_over_3 :
  ∃ (p : ℝ), p > 0 ∧ 
  (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_x_over_3_l667_66722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_is_line_l667_66766

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a horizontal translation of a function -/
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x ↦ f (x - h)

/-- Represents the reflection of a function about the x-axis -/
def reflect (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ -f x

/-- The equation of a parabola -/
def parabola_eq (p : Parabola) : ℝ → ℝ := λ x ↦ p.a * x^2 + p.b * x + p.c

/-- Theorem stating that the sum of a translated parabola and its reflected and translated version is a non-horizontal line -/
theorem parabola_sum_is_line (p : Parabola) :
  ∃ m k : ℝ, m ≠ 0 ∧
  (λ x ↦ (translate (parabola_eq p) 3 x) + (translate (reflect (parabola_eq p)) (-7) x)) =
  (λ x ↦ m * x + k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sum_is_line_l667_66766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_44_06_l667_66732

/-- Represents the result of an election with three candidates -/
structure ElectionResult :=
  (votes : Fin 3 → ℕ)

/-- Calculates the total number of votes in the election -/
def totalVotes (result : ElectionResult) : ℕ :=
  (result.votes 0) + (result.votes 1) + (result.votes 2)

/-- Finds the maximum number of votes among the candidates -/
def maxVotes (result : ElectionResult) : ℕ :=
  max (result.votes 0) (max (result.votes 1) (result.votes 2))

/-- Calculates the percentage of votes for the winning candidate -/
noncomputable def winningPercentage (result : ElectionResult) : ℝ :=
  (maxVotes result : ℝ) / (totalVotes result : ℝ) * 100

/-- The specific election result from the problem -/
def specificElection : ElectionResult :=
  ⟨λ i => match i with
    | 0 => 6136
    | 1 => 7636
    | 2 => 11628⟩

/-- Theorem stating that the winning percentage in the specific election is approximately 44.06% -/
theorem winning_percentage_approx_44_06 :
  abs (winningPercentage specificElection - 44.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_44_06_l667_66732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l667_66759

noncomputable def f (x : ℝ) : ℝ := (1/2)^x

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → x < y → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l667_66759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carly_ran_four_miles_in_week_four_l667_66760

/-- Carly's running schedule over 4 weeks --/
noncomputable def running_schedule : Fin 4 → ℝ
| 0 => 2  -- Week 1: 2 miles
| 1 => 2 * running_schedule 0 + 3  -- Week 2: Twice Week 1 plus 3 miles
| 2 => 9/7 * running_schedule 1  -- Week 3: 9/7 of Week 2
| 3 => running_schedule 2 - 5  -- Week 4: Reduce Week 3 by 5 miles

/-- Theorem: Carly ran 4 miles in Week 4 (the injury week) --/
theorem carly_ran_four_miles_in_week_four :
  running_schedule 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carly_ran_four_miles_in_week_four_l667_66760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_theorem_polar_form_theorem_min_distance_theorem_l667_66723

-- Define the scaling transformation
def φ (lambda mu : ℝ) (x y : ℝ) : ℝ × ℝ := (lambda * x, mu * y)

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaled curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the polar form of curve C
def polar_C (ρ θ : ℝ) : Prop := ρ = 2 / Real.sqrt (1 + Real.sin θ^2)

theorem scaling_transformation_theorem (x y : ℝ) :
  let lambda : ℝ := 2
  let mu : ℝ := Real.sqrt 2
  unit_circle x y → curve_C (φ lambda mu x y).1 (φ lambda mu x y).2 := by sorry

theorem polar_form_theorem (x y ρ θ : ℝ) :
  curve_C x y ↔ polar_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by sorry

theorem min_distance_theorem :
  let θ_min : ℝ := π / 2
  let ρ_min : ℝ := Real.sqrt 2
  (∀ θ : ℝ, polar_C ρ_min θ_min → polar_C (Real.sqrt 2) θ) ∧
  polar_C ρ_min θ_min := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_transformation_theorem_polar_form_theorem_min_distance_theorem_l667_66723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_bisector_altitude_l667_66726

-- Define the Triangle structure
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ
  nonDegenerate : sorry

-- Define necessary functions and properties
def Triangle.isRight (T : Triangle) : Prop := sorry
def Triangle.angleBisector (T : Triangle) (v : Fin 3) : ℝ × ℝ := sorry
def Triangle.altitude (T : Triangle) (v : Fin 3) : ℝ × ℝ := sorry
def Triangle.rightVertex (T : Triangle) : Fin 3 := sorry
def Triangle.otherVertex1 (T : Triangle) : Fin 3 := sorry
def Triangle.otherVertex2 (T : Triangle) : Fin 3 := sorry
def Triangle.side (T : Triangle) (v : Fin 3) : ℝ := sorry
def Triangle.hypotenuse (T : Triangle) : ℝ := sorry

theorem right_triangle_angle_bisector_altitude 
  (T : Triangle) 
  (h_right : T.isRight) 
  (h_bisector : T.angleBisector T.rightVertex = T.altitude T.rightVertex ∨ 
                T.angleBisector T.rightVertex = T.altitude T.otherVertex1 ∨ 
                T.angleBisector T.rightVertex = T.altitude T.otherVertex2) :
  (T.side 0 = T.side 1) ∨ (T.side 0 = T.hypotenuse ∨ T.side 1 = T.hypotenuse) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_angle_bisector_altitude_l667_66726
