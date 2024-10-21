import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x₄_minus_x₁_l507_50719

-- Define the quadratic functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom g_def : ∀ x, g x = f (60 - x)

-- Define the x-intercepts
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry
noncomputable def x₃ : ℝ := sorry
noncomputable def x₄ : ℝ := sorry

-- Define the ordering of x-intercepts
axiom x_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄

-- Define the relationship between x₂ and x₃
axiom x₂_x₃_diff : x₃ - x₂ = 90

-- Define that g contains the vertex of f
axiom g_contains_f_vertex : ∃ v, f v = g v ∧ ∀ x, f x ≤ f v

-- State the theorem to be proved
theorem x₄_minus_x₁ : x₄ - x₁ = 90 + 90 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x₄_minus_x₁_l507_50719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l507_50722

/-- Represents the walking speeds of two people -/
structure Speeds where
  person_a : ℝ
  person_b : ℝ
  speed_relation : person_b = (3/2) * person_a

/-- Represents the distance between two points -/
def distance_between_points : ℝ → ℝ := sorry

/-- Represents the distance between two meeting points -/
def distance_between_meetings : ℝ → ℝ := sorry

/-- Theorem stating the distance between points A and B -/
theorem distance_between_AB (s : Speeds) 
  (h1 : distance_between_meetings s.person_a = 20) :
  distance_between_points s.person_a = 50 := by
  sorry

#check distance_between_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_AB_l507_50722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l507_50761

/-- The radius of a circle inscribed within three mutually externally tangent circles -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  1 / (1/a + 1/b + 1/c + 2 * Real.sqrt (1/(a*b) + 1/(a*c) + 1/(b*c)))

/-- Theorem: The radius of the inscribed circle for given radii 3, 6, and 18 is 9/8 -/
theorem inscribed_circle_radius_specific : inscribed_circle_radius 3 6 18 = 9/8 := by
  -- Unfold the definition of inscribed_circle_radius
  unfold inscribed_circle_radius
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_l507_50761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_large_flasks_l507_50782

/-- Represents the number of large flasks in the laboratory -/
def N : ℕ := sorry

/-- Represents the number of small flasks in the laboratory -/
def n : ℕ := sorry

/-- The total number of flasks in the laboratory -/
def total_flasks : ℕ := 100

/-- There are at least 2 flasks of each size -/
axiom min_flasks : N ≥ 2 ∧ n ≥ 2

/-- The total number of flasks is the sum of large and small flasks -/
axiom flask_sum : N + n = total_flasks

/-- Probability of selecting two large flasks -/
noncomputable def P_large : ℚ := N * (N - 1) / (total_flasks * (total_flasks - 1))

/-- Probability of selecting two small flasks -/
noncomputable def P_small : ℚ := n * (n - 1) / (total_flasks * (total_flasks - 1))

/-- Probability of the salt concentration being between 50% and 60% -/
noncomputable def P_event : ℚ := P_large + P_small

/-- The main theorem stating the minimum number of large flasks -/
theorem min_large_flasks : 
  (∀ m : ℕ, m < N → P_event > (1 : ℚ) / 2) ∧ 
  P_event < (1 : ℚ) / 2 → 
  N = 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_large_flasks_l507_50782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_l507_50774

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sqrt x - 18 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

-- State the theorem
theorem f_of_g_of_2 : f (g 2) = -6 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_l507_50774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_falling_body_theorem_l507_50756

/-- Air resistance coefficient -/
noncomputable def k : ℝ := 0.083

/-- Acceleration due to gravity -/
noncomputable def g : ℝ := 9.8

/-- Mass of the falling body -/
noncomputable def m : ℝ := 100

/-- Area of the largest section perpendicular to motion -/
noncomputable def S : ℝ → ℝ := sorry

/-- Velocity as a function of time -/
noncomputable def v (t : ℝ) : ℝ := Real.sqrt (m * g / (k * S t)) * Real.tanh (t * Real.sqrt (g * k * S t / m))

/-- Terminal velocity (maximum speed) -/
noncomputable def v_max : ℝ := Real.sqrt (m * g / (k * S 0))

/-- Required area for a given maximum speed -/
noncomputable def required_area (v_max : ℝ) : ℝ := m * g / (k * v_max^2)

theorem falling_body_theorem (t : ℝ) (v_max : ℝ) :
  v t = Real.sqrt (m * g / (k * S t)) * Real.tanh (t * Real.sqrt (g * k * S t / m)) ∧
  v_max = Real.sqrt (m * g / (k * S 0)) ∧
  required_area v_max = m * g / (k * v_max^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_falling_body_theorem_l507_50756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_in_first_third_fourth_quadrants_l507_50765

-- Define the function
noncomputable def f (a m x : ℝ) : ℝ := a^x + m - 1

-- State the theorem
theorem graph_in_first_third_fourth_quadrants 
  (a m : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, y = f a m x → (y > 0 ∧ x > 0) ∨ (y < 0 ∧ x < 0) ∨ (y < 0 ∧ x > 0)) :
  a > 1 ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_in_first_third_fourth_quadrants_l507_50765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_remainder_l507_50771

theorem existence_of_large_remainder : ∃ n : ℕ, (3^n : ℕ) % (2^n : ℕ) > 10^2021 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_remainder_l507_50771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_shifted_sine_l507_50711

/-- Given a sine function f(x) = sin(ωx + φ) where ω > 0,
    if the graph of f(x) shifted left by π/3 overlaps with the original graph,
    then the minimum value of ω is 6. -/
theorem min_omega_for_shifted_sine (ω φ : ℝ) (h_pos : ω > 0) :
  (∀ x, Real.sin (ω * (x + Real.pi / 3) + φ) = Real.sin (ω * x + φ)) →
  ω ≥ 6 ∧ ∃ n : ℕ, ω = 6 * n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_shifted_sine_l507_50711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l507_50752

-- Define the vertices of the triangles
def D : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (0, 20)
def F : ℝ × ℝ := (30, 0)
def D' : ℝ × ℝ := (30, 50)
def E' : ℝ × ℝ := (60, 50)
def F' : ℝ × ℝ := (30, 10)

-- Define the rotation angle and center
noncomputable def n : ℝ := 90
noncomputable def u : ℝ := 40
noncomputable def v : ℝ := 10

-- State the theorem
theorem triangle_rotation (hn : 0 < n ∧ n < 180) :
  (∃ (rotation : ℝ × ℝ → ℝ × ℝ), 
    rotation D = D' ∧ 
    rotation E = E' ∧ 
    rotation F = F' ∧
    (∀ p : ℝ × ℝ, rotation p = 
      ( u + (p.1 - u) * Real.cos (n * π / 180) + (p.2 - v) * Real.sin (n * π / 180),
        v - (p.1 - u) * Real.sin (n * π / 180) + (p.2 - v) * Real.cos (n * π / 180) ))) →
  n + u + v = 140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l507_50752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_second_quadrant_l507_50724

theorem cosine_value_second_quadrant :
  ∀ (α : ℝ) (x y : ℝ),
  x^2 + y^2 = 1 →
  y = 4/5 →
  x < 0 →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_second_quadrant_l507_50724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_open_interval_B_subset_A_iff_m_range_l507_50729

-- Define the set A as the solution set of the inequality
def A : Set ℝ := {x | (4:ℝ)^x - 5 * (2:ℝ)^x + 4 < 0}

-- Define the set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | 3 - 2*m < x ∧ x < m + 1}

-- Statement 1: A is equal to the open interval (0, 2)
theorem A_equals_open_interval : A = Set.Ioo 0 2 := by sorry

-- Statement 2: B is a subset of A if and only if m is in (-∞, 1]
theorem B_subset_A_iff_m_range (m : ℝ) : B m ⊆ A ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_open_interval_B_subset_A_iff_m_range_l507_50729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_72_plus_30sqrt12_l507_50793

theorem sqrt_72_plus_30sqrt12 (a b c : ℤ) : 
  (Real.sqrt (72 + 30 * Real.sqrt 12) = a + b * Real.sqrt c) →
  (∀ (k : ℤ), k > 1 → ¬(∃ (m : ℤ), c = k * k * m)) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_72_plus_30sqrt12_l507_50793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circles_in_unit_square_l507_50709

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles overlap -/
def overlaps (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 < (c1.radius + c2.radius)^2

/-- Checks if a circle is within the unit square -/
def inUnitSquare (c : Circle) : Prop :=
  let (x, y) := c.center
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 
  c.radius ≤ min x (min y (min (1 - x) (1 - y)))

/-- Configuration of 8 circles -/
noncomputable def eightCircles (ε : ℝ) : List Circle :=
  [{ center := (0, 0), radius := 1/2 - ε },
   { center := (0, 1), radius := 1/2 - ε },
   { center := (1, 0), radius := 1/2 - ε },
   { center := (1, 1), radius := 1/2 - ε },
   { center := (1/2, 0), radius := ε },
   { center := (1/2, 1), radius := ε },
   { center := (1, 1/2), radius := ε },
   { center := (1/2, 1/2), radius := 1/Real.sqrt 2 + 1/2 - ε }]

/-- Theorem stating that there exists a configuration of 8 non-overlapping circles in a unit square -/
theorem eight_circles_in_unit_square :
  ∃ ε > 0, 
    (∀ c, c ∈ eightCircles ε → inUnitSquare c) ∧ 
    (∀ c1 c2, c1 ∈ eightCircles ε → c2 ∈ eightCircles ε → c1 ≠ c2 → ¬ overlaps c1 c2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_circles_in_unit_square_l507_50709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_tournament_bounds_l507_50726

/-- Represents the number of wins for each player in a round-robin tournament -/
def WinCounts (n : ℕ) := Fin (2*n+1) → ℕ

/-- The sum of squares of win counts -/
def SumOfSquares (n : ℕ) (w : WinCounts n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin (2*n+1))) (fun i => (w i)^2)

/-- The total number of games played in the tournament -/
def TotalGames (n : ℕ) : ℕ := n * (2*n+1)

theorem round_robin_tournament_bounds (n : ℕ) (w : WinCounts n) 
  (h1 : Finset.sum (Finset.univ : Finset (Fin (2*n+1))) w = TotalGames n) -- Total wins equal total games
  (h2 : ∀ i, w i ≤ 2*n) -- No player can win more than 2n games
  : n^2 * (2*n+1) ≤ SumOfSquares n w ∧ 
    SumOfSquares n w ≤ (n * (2*n+1) * (4*n+1)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_robin_tournament_bounds_l507_50726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l507_50716

/-- The number of Democrats in the Senate subcommittee -/
def num_democrats : ℕ := 6

/-- The number of Republicans in the Senate subcommittee -/
def num_republicans : ℕ := 4

/-- The total number of politicians in the Senate subcommittee -/
def total_politicians : ℕ := num_democrats + num_republicans

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (d : ℕ) (r : ℕ) : ℕ :=
  (d - 1).factorial * (Nat.choose d r) * r.factorial

/-- Theorem stating the number of valid seating arrangements -/
theorem seating_theorem :
  seating_arrangements num_democrats num_republicans = 43200 := by
  sorry

#eval seating_arrangements num_democrats num_republicans

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_theorem_l507_50716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_nonreal_roots_l507_50777

noncomputable def equation (x : ℂ) : Prop :=
  x^4 - 4*x^3 + 6*x^2 - 4*x = 1007

def is_nonreal_root (x : ℂ) : Prop :=
  equation x ∧ x.im ≠ 0

theorem product_of_nonreal_roots :
  ∃ (r₁ r₂ : ℂ), is_nonreal_root r₁ ∧ is_nonreal_root r₂ ∧ r₁ * r₂ = 1 + Real.sqrt 1008 :=
by
  sorry

#check product_of_nonreal_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_nonreal_roots_l507_50777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l507_50703

noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

def isArithmeticSequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem unique_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ 
    isArithmeticSequence (frac ((frac x)^2)) (frac x) ⌊x⌋ x ∧
    x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l507_50703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_removal_l507_50749

theorem red_ball_removal (total : ℕ) (initial_red_percent : ℚ) (final_red_percent : ℚ) 
  (removed : ℕ) (h1 : total = 600) (h2 : initial_red_percent = 70/100) 
  (h3 : final_red_percent = 60/100) (h4 : removed = 150) : 
  (initial_red_percent * (total : ℚ) - removed) / ((total : ℚ) - removed) = final_red_percent :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_removal_l507_50749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_correct_specific_train_crossing_time_l507_50758

/-- The time (in seconds) it takes for two trains of equal length to cross each other
    when moving in opposite directions. -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  (2 * train_length) / (2 * train_speed_mps)

/-- Theorem stating that the train crossing time calculation is correct. -/
theorem train_crossing_time_correct
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length > 0)
  (h2 : train_speed_kmph > 0) :
  train_crossing_time train_length train_speed_kmph =
  (2 * train_length) / (2 * (train_speed_kmph * 1000 / 3600)) :=
by
  sorry

/-- Corollary for the specific case in the problem. -/
theorem specific_train_crossing_time :
  ∃ (ε : ℝ), ε > 0 ∧ |train_crossing_time 180 80 - 8.1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_correct_specific_train_crossing_time_l507_50758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l507_50795

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a parabola -/
structure Parabola where
  k : ℝ

/-- The theorem statement -/
theorem hyperbola_parabola_intersection (h : Hyperbola) (p : Parabola) (F P : Point) :
  (P.x^2 / h.a^2) - (P.y^2 / h.b^2) = 1 →  -- P satisfies hyperbola equation
  P.y^2 = 8 * P.x →                        -- P satisfies parabola equation
  F.x = 2 ∧ F.y = 0 →                      -- Common focus
  (P.x - F.x)^2 + (P.y - F.y)^2 = 5^2 →    -- |PF| = 5
  (P.x = 3 ∧                               -- x-coordinate of P is 3
   ∀ (x y : ℝ), (y = Real.sqrt 3 * x ∨ y = -(Real.sqrt 3) * x) ↔ -- Equation of asymptotes
     x^2 / h.a^2 - y^2 / h.b^2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l507_50795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_largest_monotone_interval_l507_50792

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.sqrt 3 * Real.cos x * Real.cos (x + Real.pi / 2)

-- Theorem statement
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ Real.pi / 3 → f x₁ < f x₂ := by
  sorry

-- The interval where f(x) is monotonically increasing
def monotone_interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.pi / 3 }

-- Theorem stating that monotone_interval is the largest interval in [0, π/2] where f is increasing
theorem largest_monotone_interval :
  ∀ x : ℝ, x ∈ monotone_interval ↔ (0 ≤ x ∧ x ≤ Real.pi / 2 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < x → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_largest_monotone_interval_l507_50792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distances_l507_50717

/-- The circle equation: x^2 - 18x + y^2 + 6y + 145 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 18*x + y^2 + 6*y + 145 = 0

/-- The line equation: y = -3x + 1 -/
def line_equation (x y : ℝ) : Prop :=
  y = -3*x + 1

/-- Helper function to represent the minimum distance from origin to the circle -/
noncomputable def min_distance_origin_to_circle : ℝ :=
  3 * Real.sqrt 10 - Real.sqrt 55

/-- Helper function to represent the minimum distance from the line to the circle -/
noncomputable def min_distance_line_to_circle : ℝ :=
  29 / Real.sqrt 10 - Real.sqrt 55

/-- The theorem stating the shortest distances -/
theorem shortest_distances :
  ∃ (d1 d2 : ℝ), 
    d1 = min_distance_origin_to_circle ∧ 
    d2 = min_distance_line_to_circle ∧
    (d1 = 3 * Real.sqrt 10 - Real.sqrt 55) ∧
    (d2 = 29 / Real.sqrt 10 - Real.sqrt 55) :=
by
  use min_distance_origin_to_circle, min_distance_line_to_circle
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distances_l507_50717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_value_l507_50723

noncomputable def expr1 : ℝ := 15682 + 1 / 3579
noncomputable def expr2 : ℝ := 15682 - 1 / 3579
noncomputable def expr3 : ℝ := 15682 * (1 / 3579)
noncomputable def expr4 : ℝ := 15682 / (1 / 3579)
def expr5 : ℝ := 15682.3579

theorem highest_value :
  expr4 > expr1 ∧ expr4 > expr2 ∧ expr4 > expr3 ∧ expr4 > expr5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_value_l507_50723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_5_l507_50785

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its four vertices -/
noncomputable def trapezoidArea (E F G H : Point) : ℝ :=
  let base1 := |F.y - E.y|
  let base2 := |H.y - G.y|
  let height := |G.x - E.x|
  (base1 + base2) * height / 2

/-- Theorem: The area of the trapezoid EFGH with given vertices is 27.5 square units -/
theorem trapezoid_area_is_27_5 :
  let E : Point := ⟨0, 0⟩
  let F : Point := ⟨0, -3⟩
  let G : Point := ⟨5, 0⟩
  let H : Point := ⟨5, 8⟩
  trapezoidArea E F G H = 27.5 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_5_l507_50785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_deadlift_increase_l507_50772

/-- Represents Bobby's deadlift capabilities over time -/
structure BobbyDeadlift where
  initial_weight : ℕ  -- Initial weight Bobby can deadlift at age 13
  final_weight : ℕ    -- Final weight Bobby can deadlift at age 18
  years : ℕ           -- Number of years between initial and final deadlift

/-- Calculates the yearly increase in Bobby's deadlift capability -/
def yearly_increase (b : BobbyDeadlift) : ℚ :=
  (b.final_weight - b.initial_weight : ℚ) / b.years

/-- Theorem: Bobby's yearly deadlift increase is 110 pounds -/
theorem bobby_deadlift_increase :
  ∀ b : BobbyDeadlift,
    b.initial_weight = 300 ∧
    b.final_weight = 100 + (250 * b.initial_weight) / 100 ∧
    b.years = 18 - 13 →
    yearly_increase b = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_deadlift_increase_l507_50772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l507_50763

/-- Two vectors are parallel if their coordinates are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (a.1 = k * b.1 ∧ a.2 = k * b.2)

theorem parallel_vectors_lambda (l : ℝ) :
  are_parallel (2, -1) (l, -3) → l = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l507_50763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_must_be_included_l507_50701

def OneDigitPositiveInt (n : ℕ) : Prop := 0 < n ∧ n < 10

def SumToTen (a b : ℕ) : Prop := a + b = 10

theorem five_must_be_included (S : Finset ℕ) :
  (∀ n, n ∈ S → OneDigitPositiveInt n) →
  S.card = 5 →
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → ¬SumToTen a b) →
  5 ∈ S := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_must_be_included_l507_50701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l507_50768

def sequenceProperty (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  S 2 = 7 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * S n + 1

theorem fourth_term_value (a : ℕ → ℕ) (S : ℕ → ℕ) :
  sequenceProperty a S → a 4 = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_value_l507_50768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_w_l507_50733

-- Define the polynomials and their roots
def polynomial1 (x : ℝ) := x^3 + 4*x^2 + 5*x - 14
def polynomial2 (x u v w : ℝ) := x^3 + u*x^2 + v*x + w

-- Define the roots and coefficients
variable (p q r u v w : ℝ)

-- State the conditions
axiom root1_p : polynomial1 p = 0
axiom root1_q : polynomial1 q = 0
axiom root1_r : polynomial1 r = 0

axiom root2_pq : polynomial2 (p + q) u v w = 0
axiom root2_qr : polynomial2 (q + r) u v w = 0
axiom root2_rp : polynomial2 (r + p) u v w = 0

axiom sum_squares : p^2 + q^2 + r^2 = 21

-- Theorem to prove
theorem find_w : w = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_w_l507_50733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l507_50731

def sequence_a : ℕ → ℚ
  | 0 => 7  -- Add a case for 0
  | 1 => 7
  | (n + 1) => (7 * sequence_a n) / (sequence_a n + 7)

theorem sequence_a_formula (n : ℕ) (h : n > 0) : sequence_a n = 7 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l507_50731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_sampling_theorem_l507_50778

def total_sample_percent : ℝ := 24.444444444444443
def not_caught_percent : ℝ := 10

theorem candy_sampling_theorem : 
  ∃ ε > 0, abs (((100 - not_caught_percent) / 100 * total_sample_percent) - 22) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_sampling_theorem_l507_50778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_composed_10_l507_50740

noncomputable def f (x : ℝ) : ℝ := 1 + 2 / x

noncomputable def f_composed : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (f_composed n)

theorem fixed_points_of_f_composed_10 :
  {x : ℝ | f_composed 10 x = x} = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_of_f_composed_10_l507_50740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_equals_reciprocal_sum_l507_50746

theorem quadratic_roots_sum_equals_reciprocal_sum
  (p q : ℝ) (hp : p ≠ 0) :
  let roots := λ x ↦ x^2 - p*x + q = 0
  let sum_roots := p
  let sum_reciprocals := p / q
  (∀ r s, roots r ∧ roots s → r + s = sum_roots) →
  sum_roots = sum_reciprocals →
  q = 1 ∧ p ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_sum_equals_reciprocal_sum_l507_50746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_terms_arithmetic_sequence_l507_50736

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (a 1 + a n) * n / 2

theorem sum_of_four_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 1)
  (h_a3 : a 3 = 3) :
  sum_arithmetic_sequence a 4 = 8 := by
  sorry

#check sum_of_four_terms_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_four_terms_arithmetic_sequence_l507_50736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_alpha_beta_l507_50712

theorem tan_sum_alpha_beta (α β : ℝ) (a b : ℝ × ℝ) :
  a = (2 * Real.tan α, Real.tan β) →
  b = (4, -3) →
  a + b = (0, 0) →
  Real.tan (α + β) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_alpha_beta_l507_50712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_treasure_signs_l507_50769

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Represents that only signs under which there is no treasure are truthful -/
def truthful_signs (n : ℕ) : Prop := n ≤ total_trees

/-- The theorem stating that the smallest number of signs under which a treasure can be buried is 15 -/
theorem smallest_number_of_treasure_signs : 
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ m : ℕ, m < n → ¬(truthful_signs m)) ∧ 
  (truthful_signs n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_of_treasure_signs_l507_50769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_m_range_l507_50713

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := if a ≥ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := otimes x (2 - x)

-- Define the function g
def g (m x : ℝ) : ℝ := m^2 * x + 2 + m

theorem f_symmetry (x : ℝ) : f (2 - x) = f x := by sorry

theorem m_range : 
  {m : ℝ | ∀ x : ℝ, x ≥ 0 → f (Real.exp x) ≤ g m x} = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_m_range_l507_50713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l507_50728

/-- A rectangle in the xy-plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability that a randomly chosen point (x,y) from the rectangle satisfies x < 2y --/
noncomputable def probability_x_less_2y (r : Rectangle) : ℝ :=
  let area_satisfying := (min r.x_max (2 * r.y_max) - r.x_min) * r.y_max / 2
  let total_area := (r.x_max - r.x_min) * (r.y_max - r.y_min)
  area_satisfying / total_area

/-- The specific rectangle from the problem --/
def problem_rectangle : Rectangle where
  x_min := 0
  x_max := 4
  y_min := 0
  y_max := 3
  h_x := by norm_num
  h_y := by norm_num

theorem probability_is_one_third :
  probability_x_less_2y problem_rectangle = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_one_third_l507_50728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l507_50779

theorem min_sum_squares (a b c n : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  n > 0 →
  Finset.toSet {a + b, b + c, c + a} = Finset.toSet {n^2, (n+1)^2, (n+2)^2} →
  (∀ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 →
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    Finset.toSet {x + y, y + z, z + x} = Finset.toSet {n^2, (n+1)^2, (n+2)^2} →
    a^2 + b^2 + c^2 ≤ x^2 + y^2 + z^2) →
  a^2 + b^2 + c^2 = 1297 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l507_50779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_coefficients_sum_l507_50705

theorem last_three_coefficients_sum (a : ℝ) (a_nonzero : a ≠ 0) :
  let expansion := (1 - 1/a)^8
  let coefficients := List.range 9 |>.map (fun k => ((-1)^k : ℝ) * (Nat.choose 8 k))
  (coefficients.take 3).sum = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_coefficients_sum_l507_50705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_expression_l507_50702

theorem incorrect_expression :
  (∀ x : ℝ, x > 0 → (Real.sqrt x)^2 = x) ∧  -- Definition of square root
  ((Real.sqrt 2)^2 = 2) ∧                   -- Given condition A
  (Real.sqrt ((-2)^2) = 2) ∧                -- Given condition B
  ((-2 * Real.sqrt 2)^2 = 8) →              -- Correct evaluation of C
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) ≠ -1 := -- Negation of D
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_expression_l507_50702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_x_properties_l507_50738

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ -4 ∧ x ≤ -1 then 2*x + 8
  else if x > -1 ∧ x < 1 then -x^2 + 2
  else if x ≥ 1 ∧ x ≤ 4 then -2*(x - 3)
  else 0  -- default value for x outside the defined range

-- State the theorem
theorem g_negative_x_properties :
  (∀ x ∈ Set.Icc 1 4, g (-x) = -2*x + 8) ∧
  (∀ x ∈ Set.Ioo (-1) 1, g (-x) = -x^2 + 2) ∧
  (∀ x ∈ Set.Icc (-4) (-1), g (-x) = 2*x + 6) := by
  sorry

#check g_negative_x_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_negative_x_properties_l507_50738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twigs_per_branch_is_90_l507_50787

/-- Represents a tree with branches, twigs, and leaves. -/
structure TreeStructure where
  branches : ℕ
  twigs_per_branch : ℕ
  leaves_per_twig_type1 : ℕ
  leaves_per_twig_type2 : ℕ
  proportion_type1 : ℚ
  total_leaves : ℕ

/-- Calculates the total number of twigs on the tree. -/
def total_twigs (t : TreeStructure) : ℕ := t.branches * t.twigs_per_branch

/-- Calculates the total number of leaves on the tree based on the given parameters. -/
def calculated_leaves (t : TreeStructure) : ℚ :=
  (t.proportion_type1 * t.leaves_per_twig_type1 + (1 - t.proportion_type1) * t.leaves_per_twig_type2) *
  (total_twigs t)

/-- Theorem stating that given the specific conditions, the number of twigs per branch is 90. -/
theorem twigs_per_branch_is_90 (t : TreeStructure)
  (h1 : t.branches = 30)
  (h2 : t.proportion_type1 = 3 / 10)
  (h3 : t.leaves_per_twig_type1 = 4)
  (h4 : t.leaves_per_twig_type2 = 5)
  (h5 : t.total_leaves = 12690)
  (h6 : calculated_leaves t = t.total_leaves) :
  t.twigs_per_branch = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_twigs_per_branch_is_90_l507_50787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l507_50755

/-- The area of a trapezium given the lengths of its parallel sides and height -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a height of 25 cm, is equal to 475 square centimeters -/
theorem trapezium_area_example : trapezium_area 20 18 25 = 475 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l507_50755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mairead_exercise_distance_l507_50764

/-- The total distance Mairead covered during her exercises -/
noncomputable def total_distance (run_distance : ℝ) : ℝ :=
  let jog_distance := (3 / 5) * run_distance
  let walk_distance := 5 * jog_distance
  run_distance + jog_distance + walk_distance

/-- Theorem stating that Mairead's total distance is 184 miles -/
theorem mairead_exercise_distance :
  total_distance 40 = 184 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Simplify the arithmetic
  simp [mul_add, add_mul, mul_assoc]
  -- Perform the final calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mairead_exercise_distance_l507_50764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l507_50794

/-- The complex number representing the point in question -/
noncomputable def z : ℂ := (2 + Complex.I) / Complex.I^3

/-- Theorem stating that the point is in the second quadrant -/
theorem point_in_second_quadrant : 
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_second_quadrant_l507_50794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l507_50789

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_point_to_line (ρ : ℝ) (θ : ℝ) (l : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- The line l in polar form --/
def line_l (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ - Real.pi/6) = 1

theorem distance_point_to_line_is_one :
  distance_point_to_line 2 (-Real.pi/6) line_l = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l507_50789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_problem_instance_verify_answer_l507_50745

theorem least_subtraction_for_divisibility (n a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ (k : ℕ), k ≤ n ∧ 
  (∀ (m : ℕ), m < k → ¬((n - m) % a = 0 ∧ (n - m) % b = 0)) ∧
  ((n - k) % a = 0 ∧ (n - k) % b = 0) := by
  sorry

-- The specific problem instance
theorem problem_instance : 
  ∃ (k : ℕ), k ≤ 7538 ∧ 
  (∀ (m : ℕ), m < k → ¬((7538 - m) % 17 = 0 ∧ (7538 - m) % 23 = 0)) ∧
  ((7538 - k) % 17 = 0 ∧ (7538 - k) % 23 = 0) := by
  apply least_subtraction_for_divisibility 7538 17 23
  · exact Nat.zero_lt_succ 16
  · exact Nat.zero_lt_succ 22

-- Verifying that 109 is indeed the answer
theorem verify_answer : 
  (7538 - 109) % 17 = 0 ∧ (7538 - 109) % 23 = 0 := by
  apply And.intro
  · norm_num
  · norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_problem_instance_verify_answer_l507_50745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l507_50715

def a (n : ℕ) (l : ℝ) : ℝ := n * (n + l)

theorem lambda_range (l : ℝ) :
  (∀ n : ℕ, n > 0 → a (n + 1) l > a n l) →
  l > -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l507_50715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_theorem_l507_50759

-- Use the built-in complex number type
open Complex

-- Define the quadratic equation
def quadratic (z : ℂ) : ℂ := z^2 - (12 + 10*I)*z + (5 + 66*I)

theorem complex_roots_theorem (p q : ℝ) :
  quadratic (p + 3*I) = 0 ∧ quadratic (q + 7*I) = 0 → p = 7.5 ∧ q = 4.5 := by
  sorry

#check complex_roots_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_roots_theorem_l507_50759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l507_50741

/-- A function is quadratic in x if it can be expressed as ax^2 + bx + c for some constants a, b, c with a ≠ 0 -/
def IsQuadraticIn (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, (a - 3) * x^(|a - 1|) + x - 1 = 0 → IsQuadraticIn (fun x => (a - 3) * x^(|a - 1|) + x - 1) x) 
  ↔ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l507_50741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_145_l507_50786

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_first_two : a 1 + a 2 = 5
  sum_fourth_fifth : a 4 + a 5 = 23

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- The sum of the first 10 terms of the specific arithmetic sequence is 145 -/
theorem sum_10_is_145 (seq : ArithmeticSequence) : sum_n seq 10 = 145 := by
  sorry

#check sum_10_is_145

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_10_is_145_l507_50786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l507_50790

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Represents a can with its contents and capacity -/
structure Can where
  contents : CanContents
  capacity : ℝ

/-- The ratio of milk to water in a can -/
noncomputable def milkWaterRatio (c : CanContents) : ℝ := c.milk / c.water

theorem can_capacity_theorem (initialCan : Can) (h1 : milkWaterRatio initialCan.contents = 4/3)
    (h2 : initialCan.contents.milk + initialCan.contents.water + 8 = initialCan.capacity)
    (h3 : milkWaterRatio ⟨initialCan.contents.milk + 8, initialCan.contents.water⟩ = 2) :
    initialCan.capacity = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l507_50790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l507_50788

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin α = 3 / 5) : 
  Real.tan (α + π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l507_50788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l507_50781

noncomputable def vector_a : ℝ × ℝ := (2 - (-2), 1 - 4)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ := 
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

theorem vector_properties : 
  magnitude vector_a = 5 ∧ unit_vector vector_a = (4/5, -3/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l507_50781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l507_50796

theorem three_true_propositions (p q : Prop) 
  [Decidable p] [Decidable q]
  (h : ¬(¬p ∨ ¬q)) : 
  ∃! n : Nat, n = (if p ∨ q then 1 else 0) + 
                  (if p ∧ q then 1 else 0) + 
                  (if ¬p ∨ q then 1 else 0) + 
                  (if ¬p ∧ q then 1 else 0) ∧ 
                  n = 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l507_50796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_equation_l507_50798

noncomputable def symmetricPoint (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  let t := 1 / (x^2 + y^2)
  (x * t, y * t)

def onHyperbola (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x^2 - y^2 = 1

theorem symmetric_point_equation (P : ℝ × ℝ) (h : onHyperbola P) :
  let P' := symmetricPoint P
  let (x', y') := P'
  x'^2 - y'^2 = (x'^2 + y'^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_equation_l507_50798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_max_value_min_value_l507_50767

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

-- Theorem for the solution set of f(x) ≤ 1
theorem solution_set : {x : ℝ | f x ≤ 1} = Set.Ici (-1) := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value : ∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ y : ℝ, f y = m ∧ m = 3 := by sorry

-- Theorem for the minimum value of 3/a + a/b
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  3 / a + a / b ≥ 3 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 3 / a₀ + a₀ / b₀ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_max_value_min_value_l507_50767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polygons_l507_50760

/-- Given n points on a circle (n > 3), with one point labeled A, 
    the number of convex polygons containing A minus 
    the number of convex polygons not containing A 
    is equal to (n-1) choose 2 -/
theorem circle_polygons (n : ℕ) (hn : n > 3) :
  let polygons_with_A := Finset.sum (Finset.range (n - 1)) (λ k ↦ Nat.choose (n - 1) (k + 2))
  let polygons_without_A := Finset.sum (Finset.range (n - 2)) (λ k ↦ Nat.choose (n - 1) (k + 3))
  polygons_with_A - polygons_without_A = Nat.choose (n - 1) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polygons_l507_50760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l507_50766

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 4

-- Define the domain of x
def X : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem f_range : Set.Icc 7 12 = { y | ∃ x ∈ X, f x = y } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l507_50766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l507_50748

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l507_50748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l507_50739

theorem trig_inequality (x : ℝ) (h : 0 < x ∧ x < π/2) : Real.sin x < x ∧ x < Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l507_50739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l507_50708

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 2
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y + 3 = 0

-- Define the center and radius of circle O1
def center_O1 : ℝ × ℝ := (0, 0)
noncomputable def radius_O1 : ℝ := Real.sqrt 2

-- Define the center and radius of circle O2
def center_O2 : ℝ × ℝ := (0, -2)
def radius_O2 : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_O1.1 - center_O2.1)^2 + (center_O1.2 - center_O2.2)^2)

-- Theorem: The circles intersect
theorem circles_intersect : distance_between_centers = radius_O1 + radius_O2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l507_50708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l507_50732

theorem hyperbola_asymptote (m : ℝ) : 
  (m > 0) → 
  (∀ x y : ℝ, x^2 - y^2/m = 1 → (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) → 
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l507_50732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_still_water_speed_l507_50799

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  upstream : ℚ
  downstream : ℚ

/-- Calculates the speed in still water given upstream and downstream speeds -/
def stillWaterSpeed (s : RowingSpeed) : ℚ := (s.upstream + s.downstream) / 2

/-- Theorem: Given a man who can row upstream at 15 kmph and downstream at 25 kmph,
    his speed in still water is 20 kmph -/
theorem man_still_water_speed :
  let s : RowingSpeed := { upstream := 15, downstream := 25 }
  stillWaterSpeed s = 20 := by
  -- Unfold the definition of stillWaterSpeed
  unfold stillWaterSpeed
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_still_water_speed_l507_50799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_parallel_lines_integer_distances_l507_50742

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a regular hexagon -/
structure RegularHexagon where
  sideLength : ℝ

/-- Calculates the distance between two parallel lines -/
noncomputable def distanceBetweenParallelLines (l1 l2 : Line) : ℝ :=
  abs (l1.intercept - l2.intercept) / Real.sqrt (1 + l1.slope^2)

/-- Theorem: There exists a configuration of six parallel lines passing through
    the vertices of a regular hexagon such that all pairwise distances
    between these lines are integer values -/
theorem hexagon_parallel_lines_integer_distances (h : RegularHexagon) :
  ∃ (lines : Fin 6 → Line),
    (∀ i j, i ≠ j → Int.floor (distanceBetweenParallelLines (lines i) (lines j)) = distanceBetweenParallelLines (lines i) (lines j)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_parallel_lines_integer_distances_l507_50742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l507_50751

theorem exponent_problem (a b c : ℝ) 
  (ha : (2 : ℝ)^a = 3) 
  (hb : (2 : ℝ)^b = 5) 
  (hc : (2 : ℝ)^c = 75) : 
  ((2 : ℝ)^(c+b-a) = 125) ∧ (a = c - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l507_50751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aux_angle_formula_l507_50721

theorem aux_angle_formula (φ : Real) : 
  (∀ θ : Real, Real.sin θ - Real.sqrt 3 * Real.cos θ = 2 * Real.sin (θ + φ)) → 
  (-Real.pi < φ ∧ φ < Real.pi) → 
  φ = -Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aux_angle_formula_l507_50721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_sequence_1997_to_2000_l507_50780

/-- The standard arrow sequence -/
def standard_sequence : Fin 6 → Char
| 0 => '→'
| 1 => '↓'
| 2 => '↓'
| 3 => '↑'
| 4 => '↑'
| 5 => '→'

/-- The sequence of arrows from one number to another -/
def arrow_sequence (start : Nat) (endNum : Nat) : List Char :=
  List.range (endNum - start) |>.map (fun i => standard_sequence ⟨(start + i) % 6, by sorry⟩)

theorem arrow_sequence_1997_to_2000 :
  arrow_sequence 1997 2000 = ['↑', '→', '↓'] := by
  sorry

#eval arrow_sequence 1997 2000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrow_sequence_1997_to_2000_l507_50780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_servings_l507_50743

/-- The number of servings in a bottle of spirits -/
def num_servings (total_revenue : ℚ) (price_per_serving : ℚ) : ℕ :=
  (total_revenue / price_per_serving).floor.toNat

/-- Proof that the number of servings in a bottle is 12 -/
theorem bottle_servings :
  num_servings 98 8 = 12 := by
  rfl

#eval num_servings 98 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_servings_l507_50743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_chromic_acid_approx_l507_50700

/-- The mass percentage of hydrogen in chromic acid -/
noncomputable def mass_percentage_H_in_chromic_acid : ℝ :=
  let molar_mass_H : ℝ := 1.01
  let molar_mass_Cr : ℝ := 51.99
  let molar_mass_O : ℝ := 16.00
  let molar_mass_H2CrO4 : ℝ := 2 * molar_mass_H + molar_mass_Cr + 4 * molar_mass_O
  let mass_H_in_H2CrO4 : ℝ := 2 * molar_mass_H
  (mass_H_in_H2CrO4 / molar_mass_H2CrO4) * 100

/-- Theorem stating that the mass percentage of hydrogen in chromic acid is approximately 1.712% -/
theorem mass_percentage_H_in_chromic_acid_approx :
  |mass_percentage_H_in_chromic_acid - 1.712| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_H_in_chromic_acid_approx_l507_50700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_half_l507_50775

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Define the base case for 0 to avoid missing case error
  | 1 => 2
  | (n + 2) => 1 / (1 - sequence_a (n + 1))

theorem a_9_equals_half : sequence_a 9 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_half_l507_50775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l507_50710

/-- The function g(x) defined on [0, 10] -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (80 - x)) + Real.sqrt (x * (10 - x))

/-- The domain of g(x) -/
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 10

/-- The maximum value of g(x) -/
def N : ℝ := 22.5

/-- The point where g(x) attains its maximum value -/
def y₀ : ℝ := 33.75

theorem g_max_value :
  ∀ x, domain x → g x ≤ N ∧ g y₀ = N := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_l507_50710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pair_probability_l507_50744

/-- The set of integers from 1 to 30 inclusive -/
def IntSet : Finset ℕ := Finset.range 30

/-- The set of prime numbers from 1 to 30 inclusive -/
def PrimeSet : Finset ℕ := IntSet.filter Nat.Prime

/-- The number of ways to choose 2 elements from a set of size n -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

theorem prime_pair_probability :
  probability (choose PrimeSet.card 2) (choose IntSet.card 2) = 5 / 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_pair_probability_l507_50744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l507_50725

structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  congruentSide : ℝ

def IsoscelesTriangle.area (t : IsoscelesTriangle) : ℝ :=
  0.5 * t.base * t.height

theorem isosceles_triangle_side_length 
  (t : IsoscelesTriangle)
  (h_base : t.base = 30)
  (h_area : t.area = 120) :
  t.congruentSide = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_side_length_l507_50725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_and_distance_range_l507_50770

/-- The ellipse C: (x^2 / a^2) + (y^2 / b^2) = 1 -/
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The circle M: x^2 + (y-2)^2 = 1 -/
def circle_M (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

/-- The line l: x - my + 1 = 0 -/
def line (m x y : ℝ) : Prop := x - m*y + 1 = 0

/-- The distance from a point (x₀, y₀) to the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2)

theorem ellipse_circle_intersection_and_distance_range 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_major_axis : 2*a = 4) 
  (h_point_on_ellipse : ellipse a b (Real.sqrt 2) (Real.sqrt 6 / 2)) :
  (∃ x y : ℝ, ellipse a b x y ∧ circle_M x y) ∧ 
  (∀ m x₁ y₁ x₂ y₂ : ℝ, 
    line m x₁ y₁ ∧ line m x₂ y₂ ∧ 
    ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧ 
    distance_point_to_line 1 0 1 m (-1) > Real.sqrt 2 →
    24/13 < |y₁ - y₂| ∧ |y₁ - y₂| ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_and_distance_range_l507_50770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l507_50730

/-- The munificence of a polynomial on [-1, 1] -/
noncomputable def munificence (p : ℝ → ℝ) : ℝ :=
  ⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), |p x|

/-- A monic cubic polynomial -/
def monicCubic (a b c : ℝ) (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

/-- The smallest munificence of a monic cubic polynomial is 1 -/
theorem smallest_munificence_monic_cubic :
  ∃ (a₀ b₀ c₀ : ℝ), ∀ (a b c : ℝ),
    munificence (monicCubic a b c) ≥ munificence (monicCubic a₀ b₀ c₀) ∧
    munificence (monicCubic a₀ b₀ c₀) = 1 := by
  -- We choose a₀ = b₀ = c₀ = 0, which gives us x^3
  use 0, 0, 0
  intro a b c
  sorry  -- The proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_munificence_monic_cubic_l507_50730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_l507_50714

-- Define the region
def in_region (x y : ℤ) : Prop :=
  (y = x^2 ∨ y = -(x.natAbs) + 5 ∨ (y ≤ x^2 ∧ y ≤ -(x.natAbs) + 5))

-- Define the set of lattice points in the region
def lattice_points_in_region : Set (ℤ × ℤ) :=
  {p | in_region p.1 p.2}

-- State the theorem
theorem count_lattice_points :
  ∃ (s : Finset (ℤ × ℤ)), s.card = 18 ∧ ∀ p, p ∈ s ↔ p ∈ lattice_points_in_region :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_lattice_points_l507_50714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l507_50747

/-- A line in 2D space --/
structure Line where
  slope : Option ℝ
  intercept : ℝ

/-- Distance from a point to a line --/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  match l.slope with
  | none => |x - l.intercept|
  | some m => |m * x - y + l.intercept| / Real.sqrt (m^2 + 1)

/-- The main theorem --/
theorem line_equation_with_equal_distances (l : Line) :
  l.intercept = 1 →
  distancePointToLine (-2) (-1) l = distancePointToLine 4 5 l →
  (l.slope = some 1 ∧ l.intercept = -1) ∨ (l.slope = none ∧ l.intercept = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l507_50747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_inequality_l507_50750

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

def inequality (t : ℕ) (x : ℝ) : Prop :=
  f 1 (x + 1) > (x^2 + (t + 2) * x + t + 2) / (x^2 + 3 * x + 2)

theorem extreme_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → (∀ (h : ℝ), h ≠ 0 → (f a (x + h) - f a x) / h ≥ 0) ∧
                                              (∀ (h : ℝ), h ≠ 0 → (f a (x + h) - f a x) / h ≤ 0)) ∧
  (∃! (t : ℕ), ∀ (x : ℝ), x ≥ 1 → inequality t x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_and_inequality_l507_50750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_can_coincide_by_rotation_l507_50718

-- Define the lines
noncomputable def line_l1 (α : ℝ) (x : ℝ) : ℝ := x * Real.sin α
def line_l2 (c : ℝ) (x : ℝ) : ℝ := 2 * x + c

-- Define the theorem
theorem lines_can_coincide_by_rotation (α : ℝ) (c : ℝ) :
  ∃ (P : ℝ × ℝ), ∃ (θ : ℝ),
    ∀ (x : ℝ),
      let rotated_y := (x - P.fst) * Real.cos θ * Real.sin α -
                       (line_l1 α P.fst - P.snd) * Real.sin θ + P.snd
      rotated_y = line_l2 c x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_can_coincide_by_rotation_l507_50718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l507_50735

noncomputable def printer_speed : ℝ := 20
def maintenance_interval : ℕ := 150
noncomputable def maintenance_duration : ℝ := 5
def total_pages : ℕ := 350

noncomputable def print_time (pages : ℕ) : ℝ :=
  (pages : ℝ) / printer_speed

noncomputable def total_print_time : ℝ :=
  let full_cycles := total_pages / maintenance_interval
  let remaining_pages := total_pages % maintenance_interval
  full_cycles * (print_time maintenance_interval + maintenance_duration) + print_time remaining_pages

theorem print_time_rounded : 
  ⌊total_print_time + 0.5⌋ = 23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_print_time_rounded_l507_50735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l507_50704

theorem max_angle_difference (α β : Real) (h1 : Real.tan α = 3 * Real.tan β) 
  (h2 : 0 ≤ β) (h3 : β ≤ α) (h4 : α < π/2) :
  ∃ (max_diff : Real), max_diff = π/6 ∧ 
  ∀ (γ δ : Real), Real.tan γ = 3 * Real.tan δ → 0 ≤ δ → δ ≤ γ → γ < π/2 → γ - δ ≤ max_diff :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_difference_l507_50704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_median_relation_l507_50706

-- Define a triangle
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define a square
structure Square where
  center : EuclideanSpace ℝ (Fin 2)
  side_length : ℝ

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → EuclideanSpace ℝ (Fin 2)

-- Define the median of a triangle
noncomputable def median (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

-- Function to construct squares on triangle sides
noncomputable def construct_squares (t : Triangle) : Fin 3 → Square :=
  sorry

-- Function to construct hexagon from squares
noncomputable def construct_hexagon (t : Triangle) (squares : Fin 3 → Square) : Hexagon :=
  sorry

-- Function to get the length of a hexagon side
noncomputable def hexagon_side_length (h : Hexagon) (i : Fin 6) : ℝ :=
  sorry

-- Main theorem
theorem hexagon_median_relation (t : Triangle) :
  let squares := construct_squares t
  let hexagon := construct_hexagon t squares
  ∀ (i : Fin 3), 
    hexagon_side_length hexagon (2 * i) = 2 * (median t i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_median_relation_l507_50706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l507_50776

noncomputable def f : ℝ → ℝ := fun x ↦ (1/4) * x^2 + 2 * x + 19/4

theorem function_composition_equality :
  (∀ x : ℝ, f (2 * x - 3) = x^2 + x + 1) →
  (∀ x : ℝ, f x = (1/4) * x^2 + 2 * x + 19/4) :=
by
  intro h
  intro x
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_equality_l507_50776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_and_max_k_l507_50707

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

theorem tangent_slope_and_max_k (a : ℝ) :
  (∃ x, x > 0 ∧ (deriv (f a)) x = 3) →
  (∀ k : ℤ, (∀ x > 1, ↑k < (f a x) / (x - 1)) → k ≤ 3) ∧
  (∃ k : ℤ, k = 3 ∧ ∀ x > 1, ↑k < (f a x) / (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_and_max_k_l507_50707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_zero_l507_50737

theorem cos_2beta_zero (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.sin α = (Real.sqrt 5) / 5)
  (h4 : Real.sin (α - β) = -(Real.sqrt 10) / 10) :
  Real.cos (2 * β) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2beta_zero_l507_50737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l507_50753

noncomputable def f (x : ℝ) := Real.sqrt (4 - 2 * x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l507_50753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_equal_neg_ln_two_l507_50734

open Real

-- Define the functions h and g
noncomputable def h (t : ℝ) : ℝ := exp (t - 1)
noncomputable def g (t : ℝ) : ℝ := log (2 * t - 1) + 2

-- State the theorem
theorem min_difference_equal_neg_ln_two :
  ∃ (t₁ t₂ : ℝ), t₁ > 1/2 ∧ t₂ > 1/2 ∧ h t₁ = g t₂ ∧
  (∀ (s₁ s₂ : ℝ), s₁ > 1/2 → s₂ > 1/2 → h s₁ = g s₂ → t₂ - t₁ ≤ s₂ - s₁) ∧
  t₂ - t₁ = -log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_difference_equal_neg_ln_two_l507_50734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_perfect_square_l507_50791

def sequence_a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => 7 * sequence_a (k + 1) - sequence_a k

theorem sequence_sum_is_perfect_square (n : ℕ) :
  ∃ m : ℤ, sequence_a n + sequence_a (n + 1) + 2 = m * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_is_perfect_square_l507_50791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_fractions_l507_50784

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def is_integer_fraction (n : ℕ) : Prop :=
  ∃ k : ℕ, k * (factorial n)^(n-2) = factorial (n^2-4)

theorem count_integer_fractions :
  ∃ (S : Finset ℕ), S.card = 96 ∧ 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 100 ∧ is_integer_fraction n) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 100 → is_integer_fraction n → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_fractions_l507_50784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_positivity_l507_50797

noncomputable def f (x : ℝ) : ℝ := (3/2) * x^2 - x - x * Real.log x

theorem tangent_line_and_positivity (x : ℝ) :
  (∃ (m b : ℝ), m = 1 ∧ b = -1/2 ∧ ∀ (x y : ℝ), y = m * x + b ↔ 2*x - 2*y - 1 = 0) ∧
  (x > 0 → f x + Real.cos x - 1 > 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_positivity_l507_50797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l507_50783

/-- Definition of a parabola with given focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → ℝ

/-- The parabola equation coefficients -/
structure ParabolaCoefficients where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  a_pos : a > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs) f.natAbs = 1

/-- Theorem stating the equation of the parabola -/
theorem parabola_equation (p : Parabola) (c : ParabolaCoefficients) : 
  p.focus = (2, 5) ∧ 
  p.directrix = (fun x y ↦ 4*x + 5*y - 20) → 
  c = { a := 25, b := -40, c := 16, d := -4, e := -210, f := 51, 
        a_pos := by simp, 
        gcd_one := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l507_50783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_l507_50773

/-- A polynomial of degree 2010 with real coefficients -/
def Polynomial2010 : Type := {p : Polynomial ℝ // p.degree = 2010}

/-- The roots of a polynomial -/
noncomputable def roots (p : Polynomial2010) : Finset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
noncomputable def distinctAbsValues (p : Polynomial2010) : ℕ := 
  (roots p).image Complex.abs |>.card

/-- The number of real roots of a polynomial -/
noncomputable def realRootCount (p : Polynomial2010) : ℕ := sorry

/-- Theorem: If a polynomial of degree 2010 with real coefficients has 
    exactly 1008 distinct absolute values among its roots, 
    then it has at least 6 real roots -/
theorem min_real_roots (p : Polynomial2010) 
    (h : distinctAbsValues p = 1008) : 
    realRootCount p ≥ 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_l507_50773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_equality_l507_50727

/-- 
Given real numbers x and y, and a percentage p, 
if p% of (x - y) equals 20% of (x + y), and y is 42.857142857142854% of x,
then p is approximately equal to 50%.
-/
theorem percentage_equality (x y : ℝ) (p : ℝ) 
  (h1 : p / 100 * (x - y) = 20 / 100 * (x + y))
  (h2 : y = 0.42857142857142854 * x) :
  abs (p - 50) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_equality_l507_50727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l507_50757

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def point_B : ℝ × ℝ := (3, Real.sqrt 3)

theorem slope_angle_of_line (A B : ℝ × ℝ) (θ : ℝ) : 
  A = point_A → B = point_B → 
  0 ≤ θ → θ < π →
  Real.tan θ = (B.2 - A.2) / (B.1 - A.1) →
  θ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l507_50757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l507_50754

-- Define the triangle ABC
theorem triangle_area (A B C : Real × Real) (D : Real × Real) 
  (CD : Real) (angle_BAC : Real) :
  -- Conditions
  CD = 2 * Real.sqrt 2 →
  angle_BAC = 45 * (Real.pi / 180) →  -- Convert to radians
  -- Altitude CD is perpendicular to BC
  (D.1 - B.1) * (C.2 - B.2) = (C.1 - B.1) * (D.2 - B.2) →
  -- D is on BC
  ∃ t : Real, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) →
  -- CD has length 2√2
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = CD^2 →
  -- Angle BAC is 45°
  Real.cos angle_BAC = (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  -- Conclusion: Area of triangle ABC is 8
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l507_50754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l507_50720

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

-- State the theorem
theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  f x ≠ -1/3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l507_50720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_zero_l507_50762

/-- A structure representing a linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- A structure representing an observation point -/
structure ObservationPoint where
  x : ℝ
  y : ℝ

/-- Calculate the residual for a single observation point -/
def calculateResidual (model : LinearRegression) (point : ObservationPoint) : ℝ :=
  point.y - (model.slope * point.x + model.intercept)

/-- Calculate the sum of squared residuals for a set of observation points -/
def sumSquaredResiduals (model : LinearRegression) (data : List ObservationPoint) : ℝ :=
  (data.map (fun point => (calculateResidual model point) ^ 2)).sum

/-- Theorem: The sum of squared residuals is zero when all points lie on the regression line -/
theorem sum_squared_residuals_zero
  (model : LinearRegression)
  (data : List ObservationPoint)
  (h1 : model.slope = 1/3)
  (h2 : model.intercept = 2)
  (h3 : ∀ point ∈ data, point.y = model.slope * point.x + model.intercept) :
  sumSquaredResiduals model data = 0 := by
  sorry

#check sum_squared_residuals_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_squared_residuals_zero_l507_50762
