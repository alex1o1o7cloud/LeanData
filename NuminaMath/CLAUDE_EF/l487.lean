import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l487_48777

/-- Defines a sequence based on the given recurrence relation -/
def recurrenceSequence (x₁ x₂ : ℝ) : ℕ → ℝ
  | 0 => x₁
  | 1 => x₂
  | (n + 2) => |recurrenceSequence x₁ x₂ (n + 1)| - recurrenceSequence x₁ x₂ n

/-- States that the sequence is periodic with period 9 -/
theorem sequence_periodic (x₁ x₂ : ℝ) (n : ℕ) :
  recurrenceSequence x₁ x₂ (n + 9) = recurrenceSequence x₁ x₂ n := by
  sorry

#check sequence_periodic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodic_l487_48777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l487_48746

/-- Given a point P(2,4) on the terminal side of angle α in the first quadrant
    of a rectangular coordinate system, where the initial side of α is the positive x-axis. -/
theorem angle_problem (α : Real) (h : (2 : Real) * Real.tan α = 4) :
  (Real.tan α = 2) ∧
  ((2 * Real.sin (Real.pi - α) + 2 * (Real.cos (α / 2))^2 - 1) / 
   (Real.sqrt 2 * Real.sin (α + Real.pi / 4)) = 5 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_problem_l487_48746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_of_y₁_y₂_l487_48755

-- Define the functions y₁ and y₂
def y₁ : ℝ → ℝ := λ x ↦ x
def y₂ : ℝ → ℝ := λ x ↦ 2 * x

-- Define the inner product of two functions on [0,1]
noncomputable def inner_product (f g : ℝ → ℝ) : ℝ :=
  ∫ x in Set.Icc 0 1, f x * g x

-- State the theorem
theorem linear_dependence_of_y₁_y₂ :
  let gram_det := Matrix.det ![![inner_product y₁ y₁, inner_product y₁ y₂],
                               ![inner_product y₂ y₁, inner_product y₂ y₂]]
  gram_det = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_of_y₁_y₂_l487_48755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_distance_function_l487_48738

-- Define the speed function
noncomputable def speed (t : ℝ) : ℝ :=
  if t ≤ 1 then 60
  else if t ≤ 1.5 then 0
  else 80

-- Define the distance function
noncomputable def distance (t : ℝ) : ℝ :=
  if t ≤ 1 then 60 * t
  else if t ≤ 1.5 then 60
  else 80 * (t - 1.5) + 60

-- Theorem statement
theorem bus_journey_distance_function (t : ℝ) (h : 0 ≤ t ∧ t ≤ 2.5) :
  distance t = if t ≤ 1 then 60 * t
               else if t ≤ 1.5 then 60
               else 80 * (t - 1.5) + 60 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_journey_distance_function_l487_48738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_count_l487_48768

theorem blue_balls_count (total_balls removed_blue : ℕ) (prob_after : ℚ) 
  (h1 : total_balls = 18)
  (h2 : removed_blue = 3)
  (h3 : prob_after = 1 / 5) :
  ∃ init_blue : ℕ, 
    (init_blue : ℚ) - removed_blue = prob_after * ((total_balls : ℚ) - removed_blue) := by
  sorry

#check blue_balls_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_balls_count_l487_48768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_zero_implies_quartic_sum_l487_48796

theorem cosine_sine_sum_zero_implies_quartic_sum (A B C : ℝ) :
  (Real.cos A + Real.cos B + Real.cos C = 0) →
  (Real.sin A + Real.sin B + Real.sin C = 0) →
  (Real.cos A)^4 + (Real.cos B)^4 + (Real.cos C)^4 = 9/8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_sum_zero_implies_quartic_sum_l487_48796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l487_48744

open Real

/-- The function f(x) = 3 + x ln x -/
noncomputable def f (x : ℝ) : ℝ := 3 + x * log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := log x + 1

theorem monotonically_decreasing_interval (x : ℝ) :
  x ∈ Set.Ioo (0 : ℝ) (1 / ℯ) ↔ 
  (∀ y ∈ Set.Ioo (0 : ℝ) x, f_deriv y < 0) ∧ 
  (∀ z ∈ Set.Ioi x, f_deriv z ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l487_48744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l487_48767

/-- Two perpendicular lines passing through specific points -/
structure PerpendicularLines where
  m : ℝ
  line1 : ℝ → ℝ → Prop := λ x y => x + m * y = 0
  line2 : ℝ → ℝ → Prop := λ x y => m * x - y - m + 3 = 0
  perp : m * (-1/m) = -1
  point1 : line1 0 0
  point2 : line2 1 3

/-- The target line -/
def targetLine (x y : ℝ) : Prop := x + y - 8 = 0

/-- The minimum distance from the intersection point to the target line -/
noncomputable def minDistance (pl : PerpendicularLines) : ℝ :=
  2 * Real.sqrt 2 - Real.sqrt 10 / 2

/-- The main theorem -/
theorem min_distance_theorem (pl : PerpendicularLines) :
  ∃ (P : ℝ × ℝ), pl.line1 P.1 P.2 ∧ pl.line2 P.1 P.2 ∧
  (∀ (Q : ℝ × ℝ), pl.line1 Q.1 Q.2 ∧ pl.line2 Q.1 Q.2 →
    ∃ (d : ℝ), d ≥ minDistance pl ∧
    (Q.1 - 0)^2 + (Q.2 - d)^2 = (minDistance pl)^2 ∧ targetLine 0 d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l487_48767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l487_48713

/-- Represents the principal amount in Rupees -/
noncomputable def principal : ℝ := sorry

/-- The annual interest rate as a decimal -/
def annual_rate : ℝ := 0.2

/-- The time period in years -/
def time : ℝ := 2

/-- The difference in final amounts between half-yearly and yearly compounding -/
def difference : ℝ := 1446

/-- Calculates the compound interest for a given number of compounds per year -/
noncomputable def compound_interest (p : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  p * (1 + r / n) ^ (n * t)

theorem principal_approximation :
  abs (compound_interest principal annual_rate time 2 - 
       compound_interest principal annual_rate time 1 - 
       difference) < 1 ∧
  abs (principal - 60000) < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_approximation_l487_48713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l487_48751

-- Define the solution set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

-- Define the domain N
def N : Set ℝ := {x : ℝ | 2 - |x| > 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l487_48751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l487_48756

/-- The function f defined as f(x) = ax^2 + bx - √2 -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.sqrt 2

/-- Theorem stating that there exist multiple pairs of positive real numbers a and b
    that satisfy the equation f(f(√2)) = 1 -/
theorem multiple_solutions_exist :
  ∃ (a₁ b₁ a₂ b₂ : ℝ), a₁ > 0 ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 ∧
    a₁ ≠ a₂ ∧ b₁ ≠ b₂ ∧
    f a₁ b₁ (f a₁ b₁ (Real.sqrt 2)) = 1 ∧
    f a₂ b₂ (f a₂ b₂ (Real.sqrt 2)) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l487_48756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_property_equivalence_l487_48781

/-- A bipartite graph with 2n vertices in each part -/
structure BipartiteGraph (n : ℕ) where
  left : Finset (Fin (2*n))
  right : Finset (Fin (2*n))
  edges : Finset ((Fin (2*n)) × (Fin (2*n)))
  left_size : left.card = 2*n
  right_size : right.card = 2*n
  bipartite : ∀ e ∈ edges, (e.1 ∈ left ∧ e.2 ∈ right) ∨ (e.1 ∈ right ∧ e.2 ∈ left)

/-- The property that for any two vertices in one part, 
    n vertices in the other part are connected to exactly one of them -/
def HasSymmetricProperty (G : BipartiteGraph n) : Prop :=
  (∀ u v, u ∈ G.left → v ∈ G.left → u ≠ v → 
    (G.right.filter (λ w => xor (⟨u, w⟩ ∈ G.edges) (⟨v, w⟩ ∈ G.edges))).card = n) ∧
  (∀ u v, u ∈ G.right → v ∈ G.right → u ≠ v → 
    (G.left.filter (λ w => xor (⟨w, u⟩ ∈ G.edges) (⟨w, v⟩ ∈ G.edges))).card = n)

/-- The theorem to be proved -/
theorem symmetric_property_equivalence (n : ℕ) (G : BipartiteGraph n) :
  (∀ u v, u ∈ G.left → v ∈ G.left → u ≠ v → 
    (G.right.filter (λ w => xor (⟨u, w⟩ ∈ G.edges) (⟨v, w⟩ ∈ G.edges))).card = n) →
  (∀ u v, u ∈ G.right → v ∈ G.right → u ≠ v → 
    (G.left.filter (λ w => xor (⟨w, u⟩ ∈ G.edges) (⟨w, v⟩ ∈ G.edges))).card = n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_property_equivalence_l487_48781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l487_48707

open Real

/-- The limit of (e^(tan(2x)) - e^(-sin(2x))) / (sin(x) - 1) as x approaches π/2 is 0 -/
theorem limit_special_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, 
    |x - π/2| < δ → |x - π/2| > 0 → 
    |((exp (tan (2*x))) - (exp (-sin (2*x)))) / (sin x - 1) - 0| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_special_function_l487_48707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l487_48705

noncomputable def otimes (a b : ℝ) : ℝ := 1 / (a - b^2)

theorem solution_to_equation :
  ∃ (x : ℝ), x ≠ 4 ∧ otimes x (-2) = 2 / (x - 4) - 1 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_equation_l487_48705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindrome_addition_l487_48726

def is_palindrome (n : ℕ) : Prop :=
  (n.digits 10).reverse = n.digits 10

theorem smallest_palindrome_addition : 
  ∀ k : ℕ, k < 87 → ¬(is_palindrome (40317 + k)) ∧ is_palindrome (40317 + 87) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_palindrome_addition_l487_48726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_with_equal_triangle_perimeters_is_rhombus_l487_48785

-- Define the quadrilateral ABCD
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the intersection point O
variable (O : EuclideanSpace ℝ (Fin 2))

-- Define the property that ABCD is convex
def is_convex (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property that O is the intersection of diagonals
def is_diagonal_intersection (A B C D O : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the perimeter of a triangle
noncomputable def perimeter (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define the property of being a rhombus
def is_rhombus (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- The main theorem
theorem quadrilateral_with_equal_triangle_perimeters_is_rhombus
  (h_convex : is_convex A B C D)
  (h_intersection : is_diagonal_intersection A B C D O)
  (h_equal_perimeters : perimeter A B O = perimeter B C O ∧ 
                        perimeter B C O = perimeter C D O ∧ 
                        perimeter C D O = perimeter A D O) :
  is_rhombus A B C D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_with_equal_triangle_perimeters_is_rhombus_l487_48785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_leq_sum_sin_l487_48748

theorem sin_sum_leq_sum_sin (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) :
  Real.sin (α + β) ≤ Real.sin α + Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_leq_sum_sin_l487_48748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_height_ratio_l487_48776

theorem right_triangle_inradius_height_ratio (a b c r h : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) (hh : 0 < h)
  (right_triangle : a^2 + b^2 = c^2)
  (inradius : r * (a + b + c) = a * b)
  (height : c * h = a * b) :
  0.4 < r / h ∧ r / h < 0.5 := by
  sorry

#check right_triangle_inradius_height_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inradius_height_ratio_l487_48776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_chord_l487_48727

-- Define the curve C in polar coordinates
noncomputable def curve_C (ρ θ : ℝ) : Prop := ρ^2 * Real.cos (2 * θ) = 1

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + t, Real.sqrt 3 * t)

-- State the theorem
theorem curve_and_chord :
  (∀ x y : ℝ, (∃ ρ θ : ℝ, curve_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ x^2 - y^2 = 1) ∧
  (∃ t₁ t₂ : ℝ, 
    let (x₁, y₁) := line_l t₁
    let (x₂, y₂) := line_l t₂
    x₁^2 - y₁^2 = 1 ∧
    x₂^2 - y₂^2 = 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_chord_l487_48727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_slice_volume_l487_48716

-- Define the cake parameters
noncomputable def cake_thickness : ℝ := 1/2
noncomputable def cake_diameter : ℝ := 16
def number_of_pieces : ℕ := 16

-- Theorem statement
theorem cake_slice_volume :
  let cake_radius : ℝ := cake_diameter / 2
  let cake_volume : ℝ := π * cake_radius^2 * cake_thickness
  let slice_volume : ℝ := cake_volume / (number_of_pieces : ℝ)
  slice_volume = 2 * π :=
by
  -- Unfold the definitions
  unfold cake_thickness cake_diameter number_of_pieces
  -- Simplify the expressions
  simp [Real.pi]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_slice_volume_l487_48716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_f_odd_l487_48764

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 10

-- State the theorem
theorem f_inequalities :
  f (7/9) > f (Real.log 5 / Real.log 8) ∧
  -f (-2/3) < f (Real.log 5 / Real.log 8) ∧
  Real.log 5 / Real.log 8 < 7/9 := by
  sorry

-- Additional theorem to show f(-x) = -f(x)
theorem f_odd (x : ℝ) : f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequalities_f_odd_l487_48764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_paper_scissors_expectation_l487_48708

/-- The probability of winning a single game of rock-paper-scissors -/
noncomputable def win_prob : ℝ := 1/3

/-- The probability of not winning a single game of rock-paper-scissors -/
noncomputable def lose_prob : ℝ := 1 - win_prob

/-- The expected number of games until player A wins 2 consecutive games in rock-paper-scissors -/
noncomputable def expected_games : ℝ := 12

theorem rock_paper_scissors_expectation :
  expected_games = (lose_prob * (expected_games + 1)) + (win_prob * lose_prob * (expected_games + 2)) + (win_prob * win_prob * 2) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_paper_scissors_expectation_l487_48708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_false_l487_48730

/-- Binary operation ★ defined for positive real numbers -/
noncomputable def star (a b : ℝ) : ℝ := b ^ a

/-- Theorem stating that none of the given properties hold for the star operation -/
theorem star_properties_false :
  ∃ (a b c n : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n > 0 ∧
    (star a b ≠ star b a) ∧
    (star a (star b c) ≠ star (star a b) c) ∧
    (star a (b ^ n) ≠ star (star a n) b) ∧
    ((star a b) ^ n ≠ star a (b * n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_properties_false_l487_48730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_max_points_l487_48772

theorem soccer_league_max_points (n : ℕ) (top_teams : ℕ) : 
  n = 8 →
  top_teams = 3 →
  let total_games := n * (n - 1)
  let max_points_per_game := 3
  let total_points := total_games * max_points_per_game
  ∃ (points_per_top_team : ℕ),
    points_per_top_team ≤ total_points ∧
    points_per_top_team * top_teams ≤ total_points ∧
    (∀ p : ℕ, p > points_per_top_team → 
      p * top_teams > total_points ∨ p > total_points) ∧
    points_per_top_team = 40 :=
by
  intros h_n h_top_teams
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_league_max_points_l487_48772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l487_48717

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a line passing through (-1, -1) with slope k
def my_line (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x + 1)

-- Define the intersection of the line and the circle
def intersects (k : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    my_circle x1 y1 ∧ my_circle x2 y2 ∧ 
    my_line k x1 y1 ∧ my_line k x2 y2

-- Theorem statement
theorem slope_range : 
  ∀ k : ℝ, intersects k ↔ 0 < k ∧ k < 3/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l487_48717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l487_48799

/-- Represents the distance from home to school in kilometers -/
noncomputable def distance : ℝ := sorry

/-- Represents the speed during peak traffic in km/h -/
noncomputable def peak_speed : ℝ := sorry

/-- Time taken during peak traffic in hours -/
noncomputable def peak_time : ℝ := 20 / 60

/-- Time taken without peak traffic in hours -/
noncomputable def normal_time : ℝ := 12 / 60

/-- Speed increase without peak traffic in km/h -/
noncomputable def speed_increase : ℝ := 18

theorem distance_to_school :
  (distance / peak_speed = peak_time) ∧
  (distance / (peak_speed + speed_increase) = normal_time) →
  distance = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_school_l487_48799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_negative_min_subsidy_min_avg_cost_at_40_l487_48761

-- Define the cost function
noncomputable def cost (x : ℝ) : ℝ :=
  if 10 ≤ x ∧ x < 30 then (1/25) * x^3 + 640
  else if 30 ≤ x ∧ x ≤ 50 then x^2 - 40*x + 1600
  else 0

-- Define the profit function
noncomputable def profit (x : ℝ) : ℝ := 20*x - cost x

-- Define the average cost function
noncomputable def avg_cost (x : ℝ) : ℝ := cost x / x

-- Theorem 1: Maximum profit is negative and minimum subsidy is 700
theorem max_profit_negative_min_subsidy (x : ℝ) (h : 30 ≤ x ∧ x ≤ 50) :
  (∀ y, 30 ≤ y ∧ y ≤ 50 → profit y ≤ profit x) → profit x < 0 ∧ -profit x ≥ 700 := by
  sorry

-- Theorem 2: Average cost is minimized at x = 40
theorem min_avg_cost_at_40 :
  ∀ x, 10 ≤ x ∧ x ≤ 50 → avg_cost 40 ≤ avg_cost x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_negative_min_subsidy_min_avg_cost_at_40_l487_48761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l487_48729

theorem equation_solution :
  ∃ x : ℝ, 125 = 5 * (25 : ℝ) ^ (x - 1) ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l487_48729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l487_48735

/-- Given a triangle ABC with angle C ≥ 60°, prove that (a+b)(1/a + 1/b + 1/c) ≥ 4 + 1/sin(C/2) -/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C ≥ π/3) 
  (h7 : a / Real.sin A = b / Real.sin B) (h8 : b / Real.sin B = c / Real.sin C) 
  (h9 : A + B + C = π) : 
  (a + b) * (1/a + 1/b + 1/c) ≥ 4 + 1 / Real.sin (C/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l487_48735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l487_48745

noncomputable def f (m n x : ℝ) : ℝ := m / (x + 1) + n * Real.log x

theorem tangent_line_and_range_of_a (m n p a : ℝ) :
  (∀ x, f m n x = m / (x + 1) + n * Real.log x) →
  (∃ y, x + y - 2 = 0 ∧ HasDerivAt (f m n) y 1) →
  p ∈ Set.Ioo 0 1 →
  f m n p = 2 →
  (∀ x ∈ Set.Ioo p 1, ∀ t ∈ Set.Icc (1/2) 2,
    f m n x ≥ t^3 - t^2 - 2*a*t + 2 ∨ f m n x ≤ t^3 - t^2 - 2*a*t + 2) →
  a ∈ Set.Iic (-1/8) ∪ Set.Ici (5/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_of_a_l487_48745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_check_l487_48759

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_check :
  let given_sticks : ℝ × ℝ := (5, 9)
  let possible_third_sticks : List ℝ := [3, 4, 5, 14]
  ∃! x, x ∈ possible_third_sticks ∧
    can_form_triangle given_sticks.1 given_sticks.2 x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_check_l487_48759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydroxide_requirement_l487_48740

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  nh4no3 : Moles  -- Ammonium nitrate
  naoh : Moles    -- Sodium hydroxide
  h2o : Moles     -- Water

/-- The balanced chemical equation for the reaction -/
def balanced_equation (r : Reaction) : Prop :=
  r.nh4no3 = r.naoh ∧ r.nh4no3 = r.h2o

/-- The theorem stating the required amount of Sodium hydroxide -/
theorem sodium_hydroxide_requirement (r : Reaction) 
  (h1 : r.h2o = (2 : ℝ))           -- 2 moles of Water are formed
  (h2 : r.nh4no3 = (2 : ℝ))        -- 2 moles of Ammonium nitrate are used
  (h3 : balanced_equation r) -- The reaction follows the balanced equation
  : r.naoh = (2 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydroxide_requirement_l487_48740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l487_48703

theorem cosine_sum_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α < Real.pi/2) 
  (h2 : 0 ≤ β ∧ β < Real.pi/2) 
  (h3 : 0 ≤ γ ∧ γ < Real.pi/2) 
  (h4 : Real.tan α + Real.tan β + Real.tan γ ≤ 3) : 
  Real.cos (2*α) + Real.cos (2*β) + Real.cos (2*γ) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_inequality_l487_48703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_satisfying_points_l487_48742

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the set of points P satisfying the conditions
def SatisfyingPoints (center : ℝ × ℝ) (diameter_end1 diameter_end2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ |
    p ∈ Circle center 2 ∧
    (distance p diameter_end1)^2 + (distance p diameter_end2)^2 = 8 ∧
    distance p center = 1}

-- Theorem statement
theorem four_satisfying_points (center : ℝ × ℝ) (diameter_end1 diameter_end2 : ℝ × ℝ) :
  ∃ (points : Finset (ℝ × ℝ)), points.card = 4 ∧ 
  (∀ p ∈ points, p ∈ SatisfyingPoints center diameter_end1 diameter_end2) ∧
  (∀ p, p ∈ SatisfyingPoints center diameter_end1 diameter_end2 → p ∈ points) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_satisfying_points_l487_48742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l487_48747

/-- Parking cost structure -/
structure ParkingCost where
  base_cost : ℚ  -- Cost for the first 2 hours
  base_hours : ℕ  -- Number of hours covered by base cost
  extra_cost : ℚ  -- Cost per hour after base hours

/-- Calculate total parking cost -/
def total_parking_cost (p : ParkingCost) (hours : ℕ) : ℚ :=
  p.base_cost + p.extra_cost * (hours - p.base_hours)

/-- Calculate average cost per hour -/
def average_cost_per_hour (p : ParkingCost) (hours : ℕ) : ℚ :=
  (total_parking_cost p hours) / hours

/-- Theorem: The average cost per hour for 9 hours of parking is approximately $2.36 -/
theorem parking_cost_theorem (p : ParkingCost) :
  p.base_cost = 9 ∧ p.base_hours = 2 ∧ p.extra_cost = 7/4 →
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/200 ∧ |average_cost_per_hour p 9 - 236/100| < ε :=
by
  sorry

#eval average_cost_per_hour ⟨9, 2, 7/4⟩ 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parking_cost_theorem_l487_48747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l487_48724

-- Define the parabola
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  k : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = k * x + b

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_and_line_theorem 
  (C : Parabola)
  (P : Point)
  (focus : Point)
  (l : Line)
  (A B : Point)
  (h1 : C.equation P.x P.y)
  (h2 : P.x = 4)
  (h3 : focus.y = 0)
  (h4 : distance P focus = 6)
  (h5 : C.equation A.x A.y ∧ C.equation B.x B.y)
  (h6 : l.equation A.x A.y ∧ l.equation B.x B.y)
  (h7 : (A.x + B.x) / 2 = 2 ∧ (A.y + B.y) / 2 = 2)
  (h8 : A ≠ B) :
  C.p = 4 ∧ l.k = 2 ∧ l.b = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_line_theorem_l487_48724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_grid_sum_l487_48795

/-- A type representing a cell in the grid -/
structure Cell where
  row : Fin 100
  col : Fin 100

/-- A function representing the value in each cell -/
def GridValue := Cell → ℕ

/-- Predicate to check if two cells are neighbors -/
def are_neighbors (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Predicate to check if a grid value satisfies the neighbor condition -/
def satisfies_neighbor_condition (g : GridValue) : Prop :=
  ∀ c : Cell, (∀ c' : Cell, are_neighbors c c' → g c > g c') ∨
              (∀ c' : Cell, are_neighbors c c' → g c < g c')

/-- The sum of all values in the grid -/
def grid_sum (g : GridValue) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 100)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 100)) (λ j => g ⟨i, j⟩))

/-- The main theorem statement -/
theorem min_grid_sum :
  ∀ g : GridValue, satisfies_neighbor_condition g → grid_sum g ≥ 15000 := by
  sorry

#check min_grid_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_grid_sum_l487_48795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_l487_48754

/-- The number of radars -/
def n : ℕ := 5

/-- The radius of each radar's coverage area in km -/
noncomputable def r : ℝ := 25

/-- The width of the coverage ring in km -/
noncomputable def w : ℝ := 14

/-- The central angle of the regular pentagon in radians -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The distance from the center to a radar -/
noncomputable def d : ℝ := 24 / Real.sin (θ / 2)

/-- The area of the coverage ring -/
noncomputable def A : ℝ := 672 * Real.pi / Real.tan (θ / 2)

theorem radar_coverage :
  d = 24 / Real.sin (Real.pi / 5) ∧
  A = 672 * Real.pi / Real.tan (Real.pi / 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radar_coverage_l487_48754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_correct_l487_48722

-- Define the lines and point
def l₁ (x y : ℝ) : Prop := 2 * x + y - 4 = 0
def l₂ (x y : ℝ) : Prop := x - 2 * y + 3 = 0
def m (x y : ℝ) : Prop := x + y - 3 = 0
def point : ℝ × ℝ := (-3, 0)

-- Define the perpendicularity condition
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, f x₁ y₁ → f x₂ y₂ → g x₃ y₃ → g x₄ y₄ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) *
    ((x₄ - x₃) * (x₄ - x₃) + (y₄ - y₃) * (y₄ - y₃)) =
    ((x₂ - x₁) * (x₄ - x₃) + (y₂ - y₁) * (y₄ - y₃))^2

-- Define the equal intercepts condition
def equal_intercepts (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ f a 0 ∧ f 0 a

-- Theorem statement
theorem line_equations_correct :
  l₂ point.1 point.2 ∧
  perpendicular l₁ l₂ ∧
  (∃ x y : ℝ, l₁ x y ∧ l₂ x y ∧ m x y) ∧
  equal_intercepts m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_correct_l487_48722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l487_48749

open Matrix

theorem matrix_transformation (N : Matrix (Fin 3) (Fin 3) ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 0, 3, 0; 1, 0, 0]
  M * N = !![N 2 0, N 2 1, N 2 2; 
             3 * N 1 0, 3 * N 1 1, 3 * N 1 2; 
             N 0 0, N 0 1, N 0 2] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_transformation_l487_48749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paraboloid_surface_area_enclosed_l487_48783

/-- The surface area of a part of a paraboloid of revolution -/
noncomputable def paraboloid_surface_area (R : ℝ) : ℝ := 
  (2 * Real.pi / 3) * ((1 + R^2)^(3/2) - 1)

/-- The equation of the paraboloid -/
def paraboloid_equation (x y z : ℝ) : Prop := 2 * z = x^2 + y^2

/-- The equation of the enclosing cylinder -/
def cylinder_equation (x y R : ℝ) : Prop := x^2 + y^2 = R^2

/-- Theorem stating the surface area of the part of the paraboloid enclosed by the cylinder -/
theorem paraboloid_surface_area_enclosed (R : ℝ) :
  ∀ x y z, paraboloid_equation x y z → cylinder_equation x y R →
  ∃ S, S = paraboloid_surface_area R := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paraboloid_surface_area_enclosed_l487_48783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_power_of_eight_l487_48788

theorem expression_equals_power_of_eight :
  2 * (8 : ℝ) ^ (1/5 : ℝ) / (3 * (8 : ℝ) ^ (1/2 : ℝ)) = (8 : ℝ) ^ (-3/10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_power_of_eight_l487_48788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l487_48775

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 2 * (Real.sin x) ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 12)

theorem function_properties :
  (∀ k : ℤ, ∃ y : ℝ, f (Real.pi / 12 + k * Real.pi / 2) = y ∧ 
    (∀ x : ℝ, f (Real.pi / 12 + k * Real.pi / 2 + x) = f (Real.pi / 12 + k * Real.pi / 2 - x))) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 5 * Real.pi / 6 → f y < f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → g x ≤ 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g x = 2) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → g x ≥ -Real.sqrt 3 / 2 + 1) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ g x = -Real.sqrt 3 / 2 + 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l487_48775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lihua_combined_purchase_l487_48753

/-- Calculates the discounted price based on the original price --/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 30 then price
  else if price ≤ 50 then price * 0.9
  else 50 * 0.9 + (price - 50) * 0.8

/-- Represents Li Hua's two purchases --/
def purchase1 : ℝ := 23
def purchase2 : ℝ := 36

/-- Theorem stating the amount Li Hua should pay if he buys the same books in one trip --/
theorem lihua_combined_purchase :
  discountedPrice (purchase1 + (purchase2 / 0.9)) = 55.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lihua_combined_purchase_l487_48753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l487_48763

/-- Represents a parabolic arch --/
structure ParabolicArch where
  a : ℝ
  k : ℝ

/-- The height of the parabolic arch at a given x-coordinate --/
def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  arch.a * x^2 + arch.k

/-- Creates a parabolic arch given its maximum height and span --/
noncomputable def createArch (maxHeight : ℝ) (span : ℝ) : ParabolicArch :=
  { a := -4 * maxHeight / span^2,
    k := maxHeight }

theorem parabolic_arch_height_at_10 (maxHeight span : ℝ) 
  (h_max : maxHeight = 20)
  (h_span : span = 60) :
  let arch := createArch maxHeight span
  |archHeight arch 10 - 17.78| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_at_10_l487_48763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OABC_l487_48718

/-- The ellipse C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

/-- The right focus of C₁ and focus of C₂ -/
def focus : ℝ × ℝ := (1, 0)

/-- The intersection point of C₁ and C₂ -/
noncomputable def intersection : ℝ × ℝ := (2/3, 2/3 * Real.sqrt 6)

/-- A line passing through the right focus -/
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

/-- The area of quadrilateral OABC -/
noncomputable def area (m : ℝ) : ℝ := 6 * Real.sqrt ((1 + m^2) / (4 + 3 * m^2))

theorem min_area_OABC :
  ∀ m : ℝ, area m ≥ 3 ∧ ∃ m₀ : ℝ, area m₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OABC_l487_48718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_max_area_l487_48732

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- State the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_on_2_3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x - 1

-- Define the theorem
theorem f_properties_and_max_area (a : ℝ) (h : 2 < a ∧ a < 3) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f x = -x + 3) ∧
  (∃ S : ℝ, S = (a^2 - 2*a + 1) / 4 ∧
    ∀ t, 1 ≤ t ∧ t ≤ 2 →
      S ≥ (1/2) * (2*t - 2) * (a - f (3-t))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_max_area_l487_48732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_parameter_l487_48731

/-- Parametric equation of line l -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 3 / 2) * t, 1 + t / 2)

/-- Equation of parabola -/
def parabola (x : ℝ) : ℝ := x^2

/-- Theorem: The parameter t for the midpoint of AB is 1/3 -/
theorem midpoint_parameter :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ : ℝ),
    line_l t₁ = A ∧
    line_l t₂ = B ∧
    (A.2 = parabola A.1) ∧
    (B.2 = parabola B.1) ∧
    line_l ((t₁ + t₂) / 2) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (t₁ + t₂) / 2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_parameter_l487_48731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l487_48757

/-- A fair eight-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8

/-- The probability of an event occurring when rolling two eight-sided dice -/
def probability (event : ℕ → ℕ → Bool) : ℚ :=
  (Finset.filter (fun p => event p.1 p.2) (EightSidedDie.product EightSidedDie)).card /
    (EightSidedDie.card * EightSidedDie.card)

/-- The event of rolling an even product -/
def evenProduct (x y : ℕ) : Bool := Even (x * y)

theorem probability_even_product :
  probability evenProduct = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l487_48757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l487_48792

theorem sin_beta_value (α β : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : Real.sin (α/2) - Real.cos (α/2) = Real.sqrt 10 / 5)
  (h4 : Real.tan (α - β) = -5/12) : 
  Real.sin β = 16/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l487_48792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l487_48789

/-- The function f(x) = e^(2x+1) - 3x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1) - 3 * x

/-- The derivative of f at x = 0 is 2e - 3 -/
theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_zero_l487_48789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sum_in_chain_l487_48798

theorem segment_sum_in_chain (vertices : Fin 14 → ℕ) 
  (h1 : ∀ i : Fin 14, vertices i ∈ Finset.range 15 \ {0})
  (h2 : Function.Injective vertices)
  (h3 : ∀ i : Fin 7, ∃ a b c d : Fin 14, 
    vertices a + vertices b + vertices c + vertices d = 
    vertices (Fin.ofNat (4*i.val)) + 
    vertices (Fin.ofNat (4*i.val+1)) + 
    vertices (Fin.ofNat (4*i.val+2)) + 
    vertices (Fin.ofNat (4*i.val+3)))
  (h4 : ∀ i j : Fin 7, 
    vertices (Fin.ofNat (4*i.val)) + 
    vertices (Fin.ofNat (4*i.val+1)) + 
    vertices (Fin.ofNat (4*i.val+2)) + 
    vertices (Fin.ofNat (4*i.val+3)) =
    vertices (Fin.ofNat (4*j.val)) + 
    vertices (Fin.ofNat (4*j.val+1)) + 
    vertices (Fin.ofNat (4*j.val+2)) + 
    vertices (Fin.ofNat (4*j.val+3))) :
  ∀ i : Fin 7, 
    vertices (Fin.ofNat (4*i.val)) + 
    vertices (Fin.ofNat (4*i.val+1)) + 
    vertices (Fin.ofNat (4*i.val+2)) + 
    vertices (Fin.ofNat (4*i.val+3)) = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_sum_in_chain_l487_48798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_trip_properties_l487_48733

/-- Represents Mike's trip as a piecewise function --/
noncomputable def mikes_trip (t : ℝ) : ℝ :=
  sorry

/-- The time when Mike reaches the highway --/
def t_highway : ℝ := sorry

/-- The time when Mike reaches the mall --/
def t_mall : ℝ := sorry

/-- The time when Mike leaves the mall --/
def t_leave_mall : ℝ := sorry

/-- The time when Mike returns to city traffic --/
def t_return_city : ℝ := sorry

/-- The total time of Mike's trip --/
def t_final : ℝ := sorry

/-- Theorem stating that mikes_trip satisfies the required properties --/
theorem mikes_trip_properties :
  (mikes_trip 0 = 0) ∧
  (∀ t ∈ Set.Ioo 0 t_highway, DifferentiableAt ℝ mikes_trip t ∧ 0 < (deriv mikes_trip t) ∧ (deriv mikes_trip t) < (deriv mikes_trip t_highway)) ∧
  (∀ t ∈ Set.Ioo t_highway t_mall, DifferentiableAt ℝ mikes_trip t ∧ (deriv mikes_trip t_highway) < (deriv mikes_trip t)) ∧
  (∀ t ∈ Set.Icc t_mall t_leave_mall, mikes_trip t = mikes_trip t_mall) ∧
  (∀ t ∈ Set.Ioo t_leave_mall t_return_city, DifferentiableAt ℝ mikes_trip t ∧ (deriv mikes_trip t) < 0 ∧ (deriv mikes_trip t) < (deriv mikes_trip t_leave_mall)) ∧
  (∀ t ∈ Set.Ioo t_return_city t_final, DifferentiableAt ℝ mikes_trip t ∧ (deriv mikes_trip t) < 0 ∧ (deriv mikes_trip t_return_city) < (deriv mikes_trip t)) ∧
  (mikes_trip t_final = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_trip_properties_l487_48733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_a_and_b_is_three_l487_48737

theorem mean_of_a_and_b_is_three (a b : ℝ) (h : (2:ℝ)^a * (2:ℝ)^b = 64) : (a + b) / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_a_and_b_is_three_l487_48737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_total_distance_l487_48702

/-- Represents Jeff's running schedule for a week -/
structure RunningSchedule where
  monday_pace : ℚ
  tuesday_pace : ℚ
  wednesday_pace : ℚ
  thursday_pace : ℚ
  friday_pace : ℚ
  thursday_time : ℚ
  friday_time : ℚ

/-- Calculates the total distance run in a week based on the given schedule -/
def total_distance (schedule : RunningSchedule) : ℚ :=
  schedule.monday_pace * 1 +
  schedule.tuesday_pace * 1 +
  schedule.wednesday_pace * 1 +
  schedule.thursday_pace * schedule.thursday_time +
  schedule.friday_pace * schedule.friday_time

/-- Jeff's actual running schedule for the week -/
def jeff_schedule : RunningSchedule := {
  monday_pace := 6,
  tuesday_pace := 7,
  wednesday_pace := 8,
  thursday_pace := 15/2,
  friday_pace := 9,
  thursday_time := 2/3,
  friday_time := 7/6
}

theorem jeff_total_distance :
  total_distance jeff_schedule = 73/2 := by
  -- Expand the definition of total_distance
  unfold total_distance
  -- Simplify the arithmetic
  simp [jeff_schedule]
  -- The proof is complete
  rfl

#eval total_distance jeff_schedule

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_total_distance_l487_48702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_wins_both_games_l487_48750

/- Game A: 111 matches, take 1-10 per turn -/
def game_a (total : Nat) (max_take : Nat) : Prop :=
  total = 111 ∧ max_take = 10

/- Game B: 3 piles with 3, 4, and 5 matches -/
def game_b (piles : List Nat) : Prop :=
  piles = [3, 4, 5]

/- Nim-sum calculation -/
def nim_sum (piles : List Nat) : Nat :=
  piles.foldl Nat.xor 0

/- Winning condition for the first player -/
inductive first_player_wins : Prop → Prop
  | strategy {game : Prop} : 
    (∀ (opponent_move : Unit), first_player_wins game) → 
    first_player_wins game

theorem masha_wins_both_games 
  (a : game_a 111 10) 
  (b : game_b [3, 4, 5]) : 
  first_player_wins (game_a 111 10) ∧ first_player_wins (game_b [3, 4, 5]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_masha_wins_both_games_l487_48750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_of_right_triangle_with_centroid_at_origin_l487_48734

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z^3 + qz + r,
    if the sum of their squared magnitudes is 300, they form a right triangle on the complex plane,
    and the triangle's centroid is at the origin, then the square of the length of the hypotenuse is 450. -/
theorem hypotenuse_squared_of_right_triangle_with_centroid_at_origin
  (a b c : ℂ) (q r : ℂ) (P : ℂ → ℂ) (h₁ : P = fun z ↦ z^3 + q*z + r)
  (h₂ : P a = 0 ∧ P b = 0 ∧ P c = 0)
  (h₃ : Complex.abs a^2 + Complex.abs b^2 + Complex.abs c^2 = 300)
  (h₄ : ∃ (x y : ℝ), x^2 + y^2 = Complex.abs (b - c)^2 ∧ x^2 + y^2 = Complex.abs (a - b)^2)
  (h₅ : a + b + c = 0) :
  ∃ h : ℝ, h^2 = 450 ∧ h^2 = Complex.abs (a - b)^2 + Complex.abs (b - c)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_squared_of_right_triangle_with_centroid_at_origin_l487_48734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l487_48765

noncomputable section

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The first focus of the ellipse -/
  F₁ : ℝ × ℝ
  /-- The second focus of the ellipse -/
  F₂ : ℝ × ℝ
  /-- The sum of distances from any point on the ellipse to the foci -/
  dist_sum : ℝ
  /-- The lower vertex of the ellipse -/
  R : ℝ × ℝ

/-- The given ellipse (C) -/
noncomputable def ellipse_C : Ellipse where
  F₁ := (-2 * Real.sqrt 2, 0)
  F₂ := (2 * Real.sqrt 2, 0)
  dist_sum := 4 * Real.sqrt 3
  R := (0, -2)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, (x^2 / 12 + y^2 / 4 = 1) ↔ ((x, y) ∈ {p | ‖p - e.F₁‖ + ‖p - e.F₂‖ = e.dist_sum})

/-- A line passing through (0, 1) with slope k -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 = k * p.1 + 1}

/-- The dot product of vectors RM and RN is always 0 -/
def constant_dot_product (e : Ellipse) : Prop :=
  ∀ k : ℝ, ∀ M N : ℝ × ℝ,
    M ∈ line k ∩ {p | ‖p - e.F₁‖ + ‖p - e.F₂‖ = e.dist_sum} →
    N ∈ line k ∩ {p | ‖p - e.F₁‖ + ‖p - e.F₂‖ = e.dist_sum} →
    M ≠ N →
    ((M.1 - e.R.1) * (N.1 - e.R.1) + (M.2 - e.R.2) * (N.2 - e.R.2) = 0)

theorem ellipse_properties :
  standard_equation ellipse_C ∧ constant_dot_product ellipse_C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l487_48765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charged_calls_exist_free_call_arrangement_exists_l487_48714

/-- Represents a group of students and their free call arrangements -/
structure PhoneNetwork where
  n : ℕ  -- number of students
  k : ℕ  -- number of free calls each student can make
  free_calls : Fin n → Finset (Fin n)
  free_calls_bound : ∀ i, (free_calls i).card ≤ k

/-- Part (a): If n ≥ 2k + 2, then there exist two students who will be charged when speaking -/
theorem charged_calls_exist (G : PhoneNetwork) (h : G.n ≥ 2 * G.k + 2) :
  ∃ i j : Fin G.n, i ≠ j ∧ i ∉ G.free_calls j ∧ j ∉ G.free_calls i := by
  sorry

/-- Part (b): If n = 2k + 1, then there exists a free call arrangement where everyone can speak free -/
theorem free_call_arrangement_exists (n k : ℕ) (h : n = 2 * k + 1) :
  ∃ G : PhoneNetwork, G.n = n ∧ G.k = k ∧
    ∀ i j : Fin G.n, i ≠ j → (i ∈ G.free_calls j ∨ j ∈ G.free_calls i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charged_calls_exist_free_call_arrangement_exists_l487_48714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_line_equation_is_correct_max_chord_line_passes_through_center_l487_48758

/-- The circle with equation (x-1)^2 + (y+2)^2 = 5 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

/-- The center of the circle -/
def center : ℝ × ℝ := (1, -2)

/-- The point that the line must pass through -/
def point : ℝ × ℝ := (2, 1)

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3*x - y - 5 = 0

theorem line_passes_through_points :
  line_equation center.1 center.2 ∧
  line_equation point.1 point.2 := by sorry

theorem line_equation_is_correct :
  ∀ (x y : ℝ), line_equation x y ↔ 
  (y - point.2) / (x - point.1) = (center.2 - point.2) / (center.1 - point.1) := by sorry

theorem max_chord_line_passes_through_center :
  ∀ (x y : ℝ), line_equation x y → circle_equation x y →
  ∃ (t : ℝ), x = center.1 + t * (point.1 - center.1) ∧
             y = center.2 + t * (point.2 - center.2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_points_line_equation_is_correct_max_chord_line_passes_through_center_l487_48758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l487_48773

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + 1

theorem tangent_line_values (a b : ℝ) :
  (∀ x, HasDerivAt (f a) (4 * x - (f a x) + b) 1) →
  a = 6 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_values_l487_48773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l487_48739

-- Define P(n) as the greatest prime factor of n
def greatest_prime_factor (n : ℕ) : ℕ :=
  if n > 1 then
    (Nat.factors n).foldl max 0
  else 1

-- Theorem statement
theorem no_solution_exists : 
  ¬ ∃ (n : ℕ), n > 1 ∧ 
    (greatest_prime_factor n = Nat.sqrt n) ∧ 
    (greatest_prime_factor (n + 63) = Nat.sqrt (n + 63)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l487_48739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_beta_equals_negative_one_l487_48774

theorem tan_alpha_minus_beta_equals_negative_one
  (α β : ℝ)
  (h : Real.sin (α + β) + Real.cos (α + β) = 2 * Real.sqrt 2 * Real.cos (α + π / 4) * Real.sin β) :
  Real.tan (α - β) = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_beta_equals_negative_one_l487_48774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_variance_l487_48710

noncomputable def apple_weights : List ℝ := [125, 124, 122, 123, 126]

noncomputable def sample_mean (weights : List ℝ) : ℝ :=
  (weights.sum) / weights.length

noncomputable def sample_variance (weights : List ℝ) : ℝ :=
  let mean := sample_mean weights
  (weights.map (fun x => (x - mean)^2)).sum / weights.length

theorem apple_weights_variance :
  sample_variance apple_weights = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_weights_variance_l487_48710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_area_between_circles_l487_48762

/-- The area of the region inside a large circle and outside eight congruent circles arranged in a ring -/
noncomputable def area_between_circles (R : ℝ) : ℝ :=
  let r := R / 3
  Real.pi * (R^2 - 8 * r^2)

/-- The theorem stating the floor of the area for the given configuration -/
theorem floor_area_between_circles :
  ⌊area_between_circles 40⌋ = 5268 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_area_between_circles_l487_48762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_problem_l487_48791

/-- Represents a color. -/
def Color : Type := ℕ

/-- The minimum number of colors needed for the circle coloring problem. -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 3 then 1 else (n + 1) / 2

/-- Determines if two circles intersect. -/
def circle_intersect (n : ℕ) (p q r s : Fin n) : Prop :=
  sorry

/-- Assigns a color to a circle defined by two points. -/
def circle_color (n : ℕ) (p q : Fin n) : Fin ((n + 1) / 2) :=
  sorry

/-- The circle coloring problem statement. -/
theorem circle_coloring_problem (n : ℕ) (h : n > 3) :
  ∃ (coloring : Fin ((n + 1) / 2) → Color),
    ∀ (p q r s : Fin n),
      p ≠ q ∧ r ≠ s ∧ 
      circle_intersect n p q r s →
      coloring (circle_color n p q) ≠ coloring (circle_color n r s) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_coloring_problem_l487_48791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_coop_min_cost_l487_48752

/-- Represents the cost function for the chicken coop construction -/
noncomputable def cost_function (x : ℝ) : ℝ := 80 * (x + 36 / x) + 1800

/-- Theorem stating the minimum cost and corresponding side length -/
theorem chicken_coop_min_cost :
  ∃ (x : ℝ), 0 < x ∧ x ≤ 7 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ 7 → cost_function x ≤ cost_function y) ∧
  x = 6 ∧ cost_function x = 2760 := by
  sorry

/-- The area of the chicken coop -/
def coop_area : ℝ := 36

/-- The maximum allowed length of the side -/
def max_side_length : ℝ := 7

/-- The height of the wall -/
def wall_height : ℝ := 2

/-- The cost of constructing the front per square meter -/
def front_cost_per_sqm : ℝ := 40

/-- The cost of constructing the side per square meter -/
def side_cost_per_sqm : ℝ := 20

/-- The fixed cost for ground and other expenses -/
def fixed_cost : ℝ := 1800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_coop_min_cost_l487_48752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l487_48709

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y = -32

/-- The line passing through (4,4) with slope m -/
def problem_line (m x y : ℝ) : Prop := y - 4 = m * (x - 4)

/-- The dot product of OA and OB -/
def problem_dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem dot_product_bounds :
  ∀ m x₁ y₁ x₂ y₂ : ℝ,
  problem_circle x₁ y₁ → problem_circle x₂ y₂ →
  problem_line m x₁ y₁ → problem_line m x₂ y₂ →
  28 - 4 * Real.sqrt 2 ≤ problem_dot_product x₁ y₁ x₂ y₂ ∧
  problem_dot_product x₁ y₁ x₂ y₂ ≤ 28 + 4 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_bounds_l487_48709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_condition_l487_48770

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log a
  else if -4 ≤ x ∧ x < 0 then |x + 3|
  else 0  -- We need to define a value for x outside the given ranges

-- Define the condition of symmetry about y-axis
def symmetric_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ = -x₂ ∧ f x₁ = f x₂ ∧
  ∀ x y : ℝ, x ≠ y ∧ x = -y ∧ f x = f y → (x = x₁ ∧ y = x₂) ∨ (x = x₂ ∧ y = x₁)

-- State the theorem
theorem f_symmetry_condition (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  symmetric_points (f a) ↔ (a > 0 ∧ a < 1) ∨ (a > 1 ∧ a < 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_condition_l487_48770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l487_48794

/-- Represents the speed of a particle at the nth mile -/
noncomputable def speed (n : ℕ) : ℝ :=
  if n ≤ 1 then 0.5 else (1 / (2 * Real.sqrt (n - 1 : ℝ)))

/-- Represents the time taken to traverse the nth mile -/
noncomputable def time (n : ℕ) : ℝ :=
  if n ≤ 1 then 2 else (1 / speed n)

/-- Theorem stating that the time to traverse the nth mile is 2√(n-1) for n > 1 -/
theorem nth_mile_time (n : ℕ) (h : n > 1) : time n = 2 * Real.sqrt ((n - 1 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_mile_time_l487_48794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_linear_independence_l487_48706

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def is_basis (e₁ e₂ : α) : Prop :=
  ∀ v : α, ∃! (a b : ℝ), v = a • e₁ + b • e₂

theorem basis_linear_independence (e₁ e₂ : α) (h : is_basis e₁ e₂) :
  ∀ a b : ℝ, a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basis_linear_independence_l487_48706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l487_48719

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Get the slope of a line if it's not vertical -/
noncomputable def Line.slope (l : Line) : ℚ := -l.a / l.b

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem perpendicular_line_through_point 
  (l1 : Line) 
  (p : Point) 
  (h1 : l1.a = 1 ∧ l1.b = -2 ∧ l1.c = 3) 
  (h2 : p.x = -1 ∧ p.y = 3) :
  ∃ l2 : Line, l2.a = 2 ∧ l2.b = 1 ∧ l2.c = -1 ∧ 
    l2.contains p ∧ l2.perpendicular l1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l487_48719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_l487_48790

-- Define the lines
noncomputable def l₁ (x y : ℝ) : Prop := 2*x + 3*y - 5 = 0
noncomputable def l₂ (x y : ℝ) : Prop := 3*x - 2*y - 3 = 0
noncomputable def l_parallel (x y : ℝ) : Prop := 2*x + y - 3 = 0

-- Define the intersection point
noncomputable def intersection_point : ℝ × ℝ := (19/13, 9/13)

-- Define the resulting line
noncomputable def result_line (x y : ℝ) : Prop := 26*x + 13*y - 47 = 0

-- Theorem statement
theorem line_through_intersection_parallel :
  (∀ x y, l₁ x y ∧ l₂ x y ↔ x = intersection_point.1 ∧ y = intersection_point.2) →
  (∃ k : ℝ, ∀ x y, result_line x y ↔ 2*x + y + k = 0) →
  result_line intersection_point.1 intersection_point.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_l487_48790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l487_48784

def M : ℕ := 2^3 * 3^2 * 5^5 * 7^1 * 11^2

theorem number_of_factors_of_M : (Finset.filter (fun x => x ∣ M) (Finset.range (M + 1))).card = 432 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l487_48784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_105_45_plus_sin_45_105_l487_48725

open Real

/-- The cosine of the sum of two angles is equal to the product of their cosines minus the product of their sines -/
axiom cos_sum (α β : ℝ) : cos (α + β) = cos α * cos β - sin α * sin β

/-- The cosine of the difference of two angles is equal to the product of their cosines plus the product of their sines -/
axiom cos_diff (α β : ℝ) : cos (α - β) = cos α * cos β + sin α * sin β

/-- The cosine of 60 degrees is 1/2 -/
axiom cos_60_deg : cos (π/3) = 1/2

/-- 105 degrees in radians -/
noncomputable def angle_105 : ℝ := 7*π/12

/-- 45 degrees in radians -/
noncomputable def angle_45 : ℝ := π/4

/-- The main theorem: cos(105°)cos(45°) + sin(45°)sin(105°) = 1/2 -/
theorem cos_105_45_plus_sin_45_105 : 
  cos angle_105 * cos angle_45 + sin angle_45 * sin angle_105 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_105_45_plus_sin_45_105_l487_48725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l487_48786

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A > 0 ∧ t.A < Real.pi ∧
  t.B > 0 ∧ t.B < Real.pi ∧
  t.C > 0 ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.sin t.A = Real.sqrt 3 * (1 - Real.cos t.A) ∧
  t.a = 7 ∧
  Real.sin t.B + Real.sin t.C = (13 * Real.sqrt 3) / 14

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l487_48786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_partnership_problem_l487_48760

/-- Investment partnership problem -/
theorem investment_partnership_problem 
  (x : ℝ)  -- A's investment amount
  (annual_gain : ℝ)  -- Total annual gain
  (h_annual_gain : annual_gain = 21000)  -- Annual gain is Rs. 21,000
  : (x * 12) / (x * 12 + 2 * x * 6 + 3 * x * 4) * annual_gain = 7000 := by
  sorry

-- Remove the #eval line as it's not necessary for this theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_partnership_problem_l487_48760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_l487_48712

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  4 * x - 2 * y = 5

-- Define what it means for a point to lie on the perpendicular bisector
def lies_on_perpendicular_bisector (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  perpendicular_bisector x y

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  ∀ p : ℝ × ℝ, lies_on_perpendicular_bisector p ↔ 
  p.1 - 2 = 2 * (p.2 - 3/2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_AB_l487_48712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l487_48711

/-- Two cyclists on a circular track -/
structure CyclistProblem where
  speed1 : ℝ
  speed2 : ℝ
  circumference : ℝ

/-- Time for cyclists to meet at the starting point -/
noncomputable def meetingTime (p : CyclistProblem) : ℝ :=
  p.circumference / (p.speed1 + p.speed2)

/-- Theorem: The cyclists meet at the starting point after 40 seconds -/
theorem cyclists_meeting_time (p : CyclistProblem) 
  (h1 : p.speed1 = 7)
  (h2 : p.speed2 = 8)
  (h3 : p.circumference = 600) :
  meetingTime p = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meeting_time_l487_48711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l487_48720

theorem cube_root_54880000 :
  (54880000 : ℝ) ^ (1/3 : ℝ) = 20 * ((5^2 * 137 : ℝ) ^ (1/3 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_54880000_l487_48720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_identity_l487_48721

noncomputable def a (n : ℕ) : ℝ := (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identity (n : ℕ) (h : n ≥ 1) : 
  (a n + a (n + 2)) / 4 = a (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_identity_l487_48721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triangle_iff_k_range_l487_48771

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

-- Define the triangle inequality condition
def triangle_inequality (k : ℝ) : Prop :=
  ∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 →
    f k x₁ + f k x₂ > f k x₃ ∧
    f k x₂ + f k x₃ > f k x₁ ∧
    f k x₃ + f k x₁ > f k x₂

-- State the theorem
theorem f_triangle_iff_k_range :
  ∀ k : ℝ, triangle_inequality k ↔ -1/2 ≤ k ∧ k ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_triangle_iff_k_range_l487_48771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l487_48728

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}

def M : Set ℕ := {x ∈ U | (4 : ℝ) ^ (x : ℝ) ≤ 16}

theorem complement_of_M_in_U : Mᶜ = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_M_in_U_l487_48728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l487_48700

theorem inscribed_square_area_ratio (r : ℝ) (h : r > 0) :
  (r^2 * 2) / (π * r^2) = 2 / π := by
  sorry

#check inscribed_square_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_ratio_l487_48700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_to_five_coprime_l487_48704

/-- A four-digit number is an integer between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A set of numbers is pairwise coprime if the greatest common divisor
    of any two distinct numbers in the set is 1. -/
def PairwiseCoprime (S : Finset ℕ) : Prop :=
  ∀ a b, a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1

theorem six_to_five_coprime
  (S : Finset ℕ)
  (h_card : S.card = 6)
  (h_four_digit : ∀ n, n ∈ S → FourDigitNumber n)
  (h_coprime : PairwiseCoprime S) :
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 5 ∧ PairwiseCoprime T :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_to_five_coprime_l487_48704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AEF_is_4_over_35_l487_48736

-- Define the triangle ABC and points D, E, F
variable (A B C D E F : ℝ × ℝ)

-- Define the areas of the given triangles
noncomputable def area_BCD : ℝ := 1
noncomputable def area_BDE : ℝ := 1/3
noncomputable def area_CDF : ℝ := 1/5

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_AEF_is_4_over_35 :
  E.1 ≠ A.1 ∨ E.2 ≠ A.2 →  -- E is not equal to A
  F.1 ≠ A.1 ∨ F.2 ≠ A.2 →  -- F is not equal to A
  E.1 ≠ B.1 ∨ E.2 ≠ B.2 →  -- E is not equal to B
  F.1 ≠ C.1 ∨ F.2 ≠ C.2 →  -- F is not equal to C
  triangle_area B C D = area_BCD →
  triangle_area B D E = area_BDE →
  triangle_area C D F = area_CDF →
  triangle_area A E F = 4/35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_AEF_is_4_over_35_l487_48736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_l487_48793

/-- The area of the shaded region in a regular hexagon with side length 4,
    containing six semicircles along its sides and six unit-diameter circles at its vertices. -/
theorem shaded_area_hexagon : ℝ := by
  -- Define the side length of the hexagon
  let s : ℝ := 4

  -- Define the area of the hexagon
  let hexagon_area : ℝ := 3 * Real.sqrt 3 / 2 * s^2

  -- Define the area of one semicircle
  let semicircle_area : ℝ := Real.pi * (s/2)^2 / 2

  -- Define the area of one small circle
  let small_circle_area : ℝ := Real.pi * (1/2)^2

  -- Define the total area of all semicircles
  let total_semicircle_area : ℝ := 6 * semicircle_area

  -- Define the total area of all small circles
  let total_small_circle_area : ℝ := 6 * small_circle_area

  -- Define the shaded area
  let shaded_area : ℝ := hexagon_area - total_semicircle_area - total_small_circle_area

  -- Prove that the shaded area equals 24√3 - 27π/2
  have : shaded_area = 24 * Real.sqrt 3 - 27 * Real.pi / 2 := by sorry

  -- Return the result
  exact shaded_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_l487_48793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_y_l487_48715

-- Define the function
noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

-- State the theorem
theorem third_derivative_of_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_y_l487_48715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_always_negative_inequality_solution_sets_l487_48723

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1: Range of m for which y < 0 for all x
theorem y_always_negative (m : ℝ) :
  (∀ x : ℝ, y m x < 0) ↔ m ∈ Set.Ioc (-4) 0 := by sorry

-- Part 2: Solution sets for the inequality
theorem inequality_solution_sets (m : ℝ) :
  (m = 0 → {x : ℝ | y m x < (1 - m) * x - 1} = {x : ℝ | x > 0}) ∧
  (m > 0 → {x : ℝ | y m x < (1 - m) * x - 1} = {x : ℝ | 0 < x ∧ x < 1 / m}) ∧
  (m < 0 → {x : ℝ | y m x < (1 - m) * x - 1} = {x : ℝ | x < 1 / m ∨ x > 0}) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_always_negative_inequality_solution_sets_l487_48723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l487_48797

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x : ℝ | 4 * x^2 - 3 * x + 1 = x^2 - 5 * x + 7}

/-- The y-coordinates of the intersection points given an x-coordinate -/
def intersection_y (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 1

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 1

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ := x^2 - 5 * x + 7

theorem parabolas_intersection :
  intersection_x = {4/3, -3/2} ∧
  (∀ x ∈ intersection_x, (x, intersection_y x) ∈ ({(4/3, 37/9), (-3/2, 29/2)} : Set (ℝ × ℝ))) ∧
  (∀ x y : ℝ, parabola1 x = parabola2 x → (x, y) ∈ ({(4/3, 37/9), (-3/2, 29/2)} : Set (ℝ × ℝ))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_l487_48797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l487_48701

-- Define the center of the circle
def center : ℝ × ℝ := (-1, 1)

-- Define the line equation
def line (x y : ℝ) : Prop := x - y = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  ∀ (x' y' : ℝ), line x' y' → (x' - center.1)^2 + (y' - center.2)^2 ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l487_48701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anns_age_l487_48766

/-- Ann's current age -/
def a : ℕ := sorry

/-- Barbara's current age -/
def b : ℕ := sorry

/-- The difference between Ann's and Barbara's ages -/
def y : ℤ := a - b

/-- The sum of Ann's and Barbara's current ages is 44 -/
axiom sum_of_ages : a + b = 44

/-- Barbara's current age equals Ann's age when Barbara was as old as Ann had been when Barbara was half as old as Ann is now -/
axiom age_relation : b = a / 2 + 2 * y

theorem anns_age : a = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anns_age_l487_48766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flood_damage_usd_calculation_l487_48741

/-- Converts Australian dollars to American dollars given the exchange rate -/
noncomputable def aud_to_usd (aud : ℝ) (exchange_rate : ℝ) : ℝ :=
  aud / exchange_rate

/-- The flood damage in Australian dollars -/
def flood_damage_aud : ℝ := 45000000

/-- The exchange rate from Australian dollars to American dollars -/
def aud_usd_exchange_rate : ℝ := 2

/-- Theorem stating the equivalent flood damage in American dollars -/
theorem flood_damage_usd_calculation :
  aud_to_usd flood_damage_aud aud_usd_exchange_rate = 22500000 := by
  -- Unfold the definitions
  unfold aud_to_usd flood_damage_aud aud_usd_exchange_rate
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flood_damage_usd_calculation_l487_48741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runs_by_running_percentage_l487_48778

/-- Calculates the percentage of runs made by running between the wickets -/
def percentage_runs_by_running (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let runs_from_boundaries : ℕ := 4 * boundaries
  let runs_from_sixes : ℕ := 6 * sixes
  let runs_by_running : ℕ := total_runs - (runs_from_boundaries + runs_from_sixes)
  (runs_by_running : ℚ) / (total_runs : ℚ) * 100

/-- The percentage of runs made by running between the wickets is approximately 66.67% -/
theorem runs_by_running_percentage :
  ∃ ε > 0, abs ((percentage_runs_by_running 150 5 5 : ℚ) - 200/3) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runs_by_running_percentage_l487_48778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_vote_reversal_l487_48787

/-- Represents the outcome of a vote --/
inductive VoteResult
  | Rejected
  | Passed

/-- Represents a board vote --/
structure BoardVote where
  total_members : ℕ
  votes_for : ℕ
  votes_against : ℕ
  result : VoteResult

/-- Theorem about a board vote reversal --/
theorem board_vote_reversal
  (initial_vote second_vote : BoardVote)
  (h_total : initial_vote.total_members = 300)
  (h_initial_sum : initial_vote.votes_for + initial_vote.votes_against = initial_vote.total_members)
  (h_second_sum : second_vote.votes_for + second_vote.votes_against = second_vote.total_members)
  (h_initial_reject : initial_vote.result = VoteResult.Rejected)
  (h_second_pass : second_vote.result = VoteResult.Passed)
  (h_margin : second_vote.votes_for - second_vote.votes_against = 3 * (initial_vote.votes_against - initial_vote.votes_for))
  (h_ratio : second_vote.votes_for = (7 * initial_vote.votes_against) / 6) :
  second_vote.votes_for - initial_vote.votes_for = 55 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_vote_reversal_l487_48787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_inequality_l487_48782

-- Define the determinant function
noncomputable def det (x : ℝ) : ℝ := x / (x - 1) + 2

-- Define the solution set
def solution_set : Set ℝ := {x | (0 < x ∧ x ≤ 2/3) ∨ x > 1}

-- State the theorem
theorem determinant_inequality (x : ℝ) (hx : x > 0 ∧ x ≠ 1) :
  det x ≥ 0 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_inequality_l487_48782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l487_48769

/-- Proves that a train traveling at 55 km/hr crossing a 520 m platform in 43.196544276457885 seconds has a length of 140 m. -/
theorem train_length_proof (speed platform_length crossing_time : ℝ) :
  speed = 55 →
  platform_length = 520 →
  crossing_time = 43.196544276457885 →
  let speed_mps := speed * 1000 / 3600
  let total_distance := speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 140 :=
by
  intros h_speed h_platform h_crossing
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_proof_l487_48769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_1450_minutes_to_6am_l487_48779

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a natural number to a Time -/
def natToTime (n : Nat) : Time :=
  sorry

theorem add_1450_minutes_to_6am :
  let start : Time := { hours := 6, minutes := 0, h_valid := by sorry, m_valid := by sorry }
  let end_time : Time := addMinutes start 1450
  end_time = { hours := 6, minutes := 10, h_valid := by sorry, m_valid := by sorry } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_1450_minutes_to_6am_l487_48779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l487_48743

theorem divisibility_property (n : ℕ) : ∃ k : ℕ, (2^n : ℤ) ∣ (19^k - 97) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l487_48743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l487_48780

theorem dart_probability (s : ℝ) (hs : s > 0) : 
  (π * (s / 2)^2) / s^2 = π / 4 := by
  -- Simplify the left side of the equation
  have h1 : (π * (s / 2)^2) / s^2 = π * (s^2 / 4) / s^2 := by
    ring
  
  -- Further simplify
  have h2 : π * (s^2 / 4) / s^2 = π / 4 := by
    field_simp
    ring
  
  -- Combine the steps
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_probability_l487_48780
