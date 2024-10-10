import Mathlib

namespace chess_tournament_participants_l1593_159382

/-- The number of games played in a chess tournament where each participant
    plays exactly one game with each other participant. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that a chess tournament with 105 games has 15 participants. -/
theorem chess_tournament_participants :
  ∃ n : ℕ, n > 0 ∧ num_games n = 105 ∧ n = 15 := by
  sorry

#check chess_tournament_participants

end chess_tournament_participants_l1593_159382


namespace laura_running_speed_approx_l1593_159362

/-- Laura's workout parameters --/
structure WorkoutParams where
  totalDuration : ℝ  -- Total workout duration in minutes
  bikingDistance : ℝ  -- Biking distance in miles
  transitionTime : ℝ  -- Transition time in minutes
  runningDistance : ℝ  -- Running distance in miles

/-- Calculate Laura's running speed given workout parameters --/
def calculateRunningSpeed (params : WorkoutParams) (x : ℝ) : ℝ :=
  x^2 - 1

/-- Theorem stating that Laura's running speed is approximately 83.33 mph --/
theorem laura_running_speed_approx (params : WorkoutParams) :
  ∃ x : ℝ,
    params.totalDuration = 150 ∧
    params.bikingDistance = 30 ∧
    params.transitionTime = 10 ∧
    params.runningDistance = 5 ∧
    (params.totalDuration - params.transitionTime) / 60 = params.bikingDistance / (3*x + 2) + params.runningDistance / (x^2 - 1) ∧
    abs (calculateRunningSpeed params x - 83.33) < 0.01 :=
  sorry


end laura_running_speed_approx_l1593_159362


namespace inequality_proof_l1593_159304

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_squares : a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5)) + (b / (b + 5)) + (c / (c + 5)) ≤ 1/2 := by
  sorry

end inequality_proof_l1593_159304


namespace ellipse_parabola_intersection_range_l1593_159306

-- Define the ellipse and parabola equations
def ellipse (x y a : ℝ) : Prop := x^2 + 4*(y-a)^2 = 4
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the theorem
theorem ellipse_parabola_intersection_range (a : ℝ) :
  (∃ x y : ℝ, ellipse x y a ∧ parabola x y) →
  -1 ≤ a ∧ a ≤ 17/8 :=
by sorry

end ellipse_parabola_intersection_range_l1593_159306


namespace max_diff_color_triangles_17gon_l1593_159359

/-- Regular 17-gon with colored edges -/
structure ColoredPolygon where
  n : Nat
  colors : Nat
  no_monochromatic : Bool

/-- The number of edges in a regular 17-gon -/
def num_edges (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1)) / 2

/-- The total number of triangles in a regular 17-gon -/
def total_triangles (p : ColoredPolygon) : Nat :=
  (p.n * (p.n - 1) * (p.n - 2)) / 6

/-- The minimum number of isosceles triangles (triangles with at least two sides of the same color) -/
def min_isosceles_triangles (p : ColoredPolygon) : Nat :=
  p.n * p.colors

/-- The maximum number of triangles with all edges of different colors -/
def max_diff_color_triangles (p : ColoredPolygon) : Nat :=
  total_triangles p - min_isosceles_triangles p

/-- Theorem: The maximum number of triangles with all edges of different colors in a regular 17-gon
    with 8 colors and no monochromatic triangles is 544 -/
theorem max_diff_color_triangles_17gon :
  ∀ p : ColoredPolygon,
    p.n = 17 →
    p.colors = 8 →
    p.no_monochromatic = true →
    num_edges p = 136 →
    max_diff_color_triangles p = 544 := by
  sorry

end max_diff_color_triangles_17gon_l1593_159359


namespace shirt_tie_combination_count_l1593_159361

/-- The number of possible shirt-and-tie combinations given:
  * total_shirts: The total number of shirts
  * total_ties: The total number of ties
  * incompatible_shirts: The number of shirts that are incompatible with some ties
  * incompatible_ties: The number of ties that are incompatible with some shirts
-/
def shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) 
  (incompatible_shirts : ℕ) (incompatible_ties : ℕ) : ℕ :=
  total_shirts * total_ties - incompatible_shirts * incompatible_ties

/-- Theorem stating that with 8 shirts, 7 ties, and 1 shirt incompatible with 2 ties,
    the total number of possible shirt-and-tie combinations is 54. -/
theorem shirt_tie_combination_count :
  shirt_tie_combinations 8 7 1 2 = 54 := by
  sorry

end shirt_tie_combination_count_l1593_159361


namespace external_roads_different_colors_l1593_159321

/-- Represents a city with colored streets and intersections -/
structure ColoredCity where
  /-- Number of intersections in the city -/
  n : ℕ
  /-- Number of colors used for streets (assumed to be 3) -/
  num_colors : ℕ
  /-- Number of streets meeting at each intersection (assumed to be 3) -/
  streets_per_intersection : ℕ
  /-- Number of roads leading out of the city (assumed to be 3) -/
  num_external_roads : ℕ
  /-- Condition: Streets are colored using three colors -/
  h_num_colors : num_colors = 3
  /-- Condition: Exactly three streets meet at each intersection -/
  h_streets_per_intersection : streets_per_intersection = 3
  /-- Condition: Three roads lead out of the city -/
  h_num_external_roads : num_external_roads = 3

/-- Theorem: In a ColoredCity, the three roads leading out of the city have different colors -/
theorem external_roads_different_colors (city : ColoredCity) :
  ∃ (c₁ c₂ c₃ : ℕ), c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ ∧
  c₁ ≤ city.num_colors ∧ c₂ ≤ city.num_colors ∧ c₃ ≤ city.num_colors :=
sorry

end external_roads_different_colors_l1593_159321


namespace sandy_saturday_hours_l1593_159311

/-- Sandy's hourly rate in dollars -/
def hourly_rate : ℚ := 15

/-- Hours Sandy worked on Friday -/
def friday_hours : ℚ := 10

/-- Hours Sandy worked on Sunday -/
def sunday_hours : ℚ := 14

/-- Total earnings for Friday, Saturday, and Sunday in dollars -/
def total_earnings : ℚ := 450

/-- Calculates the number of hours Sandy worked on Saturday -/
def saturday_hours : ℚ :=
  (total_earnings - hourly_rate * (friday_hours + sunday_hours)) / hourly_rate

theorem sandy_saturday_hours :
  saturday_hours = 6 := by sorry

end sandy_saturday_hours_l1593_159311


namespace platform_length_l1593_159345

/-- Given a train that passes a pole and a platform, calculate the platform length -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 100 →
  pole_time = 15 →
  platform_time = 40 →
  ∃ (platform_length : ℝ),
    platform_length = 500 / 3 ∧
    train_length / pole_time = (train_length + platform_length) / platform_time :=
by
  sorry

#check platform_length

end platform_length_l1593_159345


namespace notebook_cost_l1593_159315

theorem notebook_cost (notebook_cost pen_cost : ℝ) 
  (total_cost : notebook_cost + pen_cost = 3.50)
  (cost_difference : notebook_cost = pen_cost + 3) : 
  notebook_cost = 3.25 := by
sorry

end notebook_cost_l1593_159315


namespace tommy_initial_candy_l1593_159343

/-- The amount of candy each person has after sharing equally -/
def shared_amount : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

/-- Hugh's initial amount of candy -/
def hugh_initial : ℕ := 8

/-- Melany's initial amount of candy -/
def melany_initial : ℕ := 7

/-- Tommy's initial amount of candy -/
def tommy_initial : ℕ := shared_amount * num_people - hugh_initial - melany_initial

theorem tommy_initial_candy : tommy_initial = 6 := by
  sorry

end tommy_initial_candy_l1593_159343


namespace fencing_required_l1593_159356

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : area = 680 ∧ uncovered_side = 10 → 
  ∃ (width : ℝ), area = uncovered_side * width ∧ 2 * width + uncovered_side = 146 := by
  sorry

end fencing_required_l1593_159356


namespace train_distance_l1593_159369

/-- Given a train that travels 1 mile every 2 minutes, prove it will travel 45 miles in 90 minutes -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 1 / 2 → time = 90 → distance = speed * time → distance = 45 := by
  sorry

end train_distance_l1593_159369


namespace A_and_D_independent_l1593_159360

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | ω.1 = 0}
def D : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 7}

-- State the theorem
theorem A_and_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end A_and_D_independent_l1593_159360


namespace equation_solution_count_l1593_159388

theorem equation_solution_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (n^2 - 2*n - 2)*n^2 + 47 = (n^2 - 2*n - 2)*16*n - 16) ∧ 
    S.card = 3 := by
  sorry

end equation_solution_count_l1593_159388


namespace water_consumption_theorem_l1593_159302

/-- Calculates the number of glasses of water drunk per day given the bottle capacity,
    number of refills per week, glass size, and days in a week. -/
def glassesPerDay (bottleCapacity : ℕ) (refillsPerWeek : ℕ) (glassSize : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (bottleCapacity * refillsPerWeek) / (glassSize * daysPerWeek)

/-- Theorem stating that given the specified conditions, the number of glasses of water
    drunk per day is equal to 4. -/
theorem water_consumption_theorem :
  let bottleCapacity : ℕ := 35
  let refillsPerWeek : ℕ := 4
  let glassSize : ℕ := 5
  let daysPerWeek : ℕ := 7
  glassesPerDay bottleCapacity refillsPerWeek glassSize daysPerWeek = 4 := by
  sorry

end water_consumption_theorem_l1593_159302


namespace quadratic_equivalence_l1593_159346

theorem quadratic_equivalence (c : ℝ) : 
  ({a : ℝ | ∀ x : ℝ, x^2 + a*x + a/4 + 1/2 > 0} = {x : ℝ | x^2 - x + c < 0}) → 
  c = -2 :=
by sorry

end quadratic_equivalence_l1593_159346


namespace ice_palace_steps_count_l1593_159331

/-- The number of steps in the Ice Palace staircase -/
def ice_palace_steps : ℕ := 30

/-- The time Alice takes to walk 20 steps (in seconds) -/
def time_for_20_steps : ℕ := 120

/-- The time Alice takes to walk all steps (in seconds) -/
def time_for_all_steps : ℕ := 180

/-- Theorem: The number of steps in the Ice Palace staircase is 30 -/
theorem ice_palace_steps_count :
  ice_palace_steps = (time_for_all_steps * 20) / time_for_20_steps :=
sorry

end ice_palace_steps_count_l1593_159331


namespace algebraic_expression_value_l1593_159310

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x = 3) : 2*x^2 - 4*x + 3 = 9 := by
  sorry

end algebraic_expression_value_l1593_159310


namespace complex_modulus_l1593_159397

theorem complex_modulus (z : ℂ) (h : z = (1/2 : ℂ) + (5/2 : ℂ) * Complex.I) : 
  Complex.abs z = Real.sqrt 26 / 2 := by
  sorry

end complex_modulus_l1593_159397


namespace sum_of_squares_of_conjugates_l1593_159355

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_squares_of_conjugates : (1 + i)^2 + (1 - i)^2 = 0 := by
  sorry

end sum_of_squares_of_conjugates_l1593_159355


namespace f_5_solutions_l1593_159365

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself 5 times -/
def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

/-- Theorem: The solutions to f(f(f(f(f(x))))) = 0 are x = -6 ± 6^(1/32) -/
theorem f_5_solutions :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32) :=
by sorry

end f_5_solutions_l1593_159365


namespace elizabeth_climb_time_l1593_159391

-- Define the climbing times
def tom_time : ℕ := 2 * 60  -- Tom's time in minutes
def elizabeth_time : ℕ := tom_time / 4  -- Elizabeth's time in minutes

-- State the theorem
theorem elizabeth_climb_time :
  (tom_time = 4 * elizabeth_time) →  -- Tom takes 4 times as long as Elizabeth
  (tom_time = 2 * 60) →  -- Tom takes 2 hours (120 minutes)
  elizabeth_time = 30 :=  -- Elizabeth takes 30 minutes
by
  sorry

end elizabeth_climb_time_l1593_159391


namespace unique_integer_solution_l1593_159354

theorem unique_integer_solution :
  ∃! x : ℤ, (x - 3 : ℚ) ^ (27 - x^2) = 1 :=
by sorry

end unique_integer_solution_l1593_159354


namespace quadratic_inequality_l1593_159318

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ 1/2 < x ∧ x < 2

-- Theorem statement
theorem quadratic_inequality (a : ℝ) (h : solution_set a) :
  a = -2 ∧
  ∀ x, a * x^2 - 5 * x + a^2 - 1 > 0 ↔ -1/2 < x ∧ x < 3 :=
sorry

end quadratic_inequality_l1593_159318


namespace vector_problem_l1593_159372

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-1, 7)

theorem vector_problem (a b : ℝ × ℝ) (ha : a = (3, 4)) (hb : b = (-1, 7)) :
  (a.1 * b.1 + a.2 * b.2 = 25) ∧ 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4) := by
  sorry

end vector_problem_l1593_159372


namespace no_solution_arccos_arcsin_l1593_159316

theorem no_solution_arccos_arcsin : ¬∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end no_solution_arccos_arcsin_l1593_159316


namespace factorization_problems_l1593_159374

variable (a b : ℝ)

theorem factorization_problems :
  (-25 + a^4 = (a^2 + 5) * (a + 5) * (a - 5)) ∧
  (a^3 * b - 10 * a^2 * b + 25 * a * b = a * b * (a - 5)^2) :=
by sorry

end factorization_problems_l1593_159374


namespace geometric_series_sum_and_comparison_l1593_159312

theorem geometric_series_sum_and_comparison :
  let a : ℝ := 2
  let r : ℝ := 1/4
  let S : ℝ := a / (1 - r)
  S = 8/3 ∧ S ≤ 3 := by sorry

end geometric_series_sum_and_comparison_l1593_159312


namespace square_sum_fourth_powers_l1593_159325

theorem square_sum_fourth_powers (a b c : ℝ) 
  (h1 : a^2 - b^2 = 5)
  (h2 : a * b = 2)
  (h3 : a^2 + b^2 + c^2 = 8) :
  a^4 + b^4 + c^4 = 38 := by
sorry

end square_sum_fourth_powers_l1593_159325


namespace perpendicular_tangents_trajectory_l1593_159378

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a tangent line from P to the unit circle
def is_tangent (P : Point) (A : Point) : Prop :=
  unit_circle A.x A.y ∧ 
  (P.x - A.x) * A.x + (P.y - A.y) * A.y = 0

-- State the theorem
theorem perpendicular_tangents_trajectory :
  ∀ P : Point,
  (∃ A B : Point,
    is_tangent P A ∧
    is_tangent P B ∧
    (P.x - A.x) * (P.x - B.x) + (P.y - A.y) * (P.y - B.y) = 0) →
  P.x^2 + P.y^2 = 2 := by
  sorry

end perpendicular_tangents_trajectory_l1593_159378


namespace min_d_value_l1593_159398

theorem min_d_value (t a b d : ℕ) : 
  (3 * t = 2 * a + 2 * b + 2016) →  -- Triangle perimeter exceeds rectangle perimeter by 2016
  (t = a + d) →                     -- Triangle side exceeds one rectangle side by d
  (t = b + 2 * d) →                 -- Triangle side exceeds other rectangle side by 2d
  (a > 0 ∧ b > 0) →                 -- Rectangle has non-zero perimeter
  (∀ d' : ℕ, d' < d → 
    ¬(∃ t' a' b' : ℕ, 
      (3 * t' = 2 * a' + 2 * b' + 2016) ∧ 
      (t' = a' + d') ∧ 
      (t' = b' + 2 * d') ∧ 
      (a' > 0 ∧ b' > 0))) →
  d = 505 :=
by sorry

end min_d_value_l1593_159398


namespace meeting_probability_4x3_grid_l1593_159314

/-- Represents a grid network --/
structure GridNetwork where
  rows : ℕ
  cols : ℕ

/-- Represents a person moving on the grid --/
structure Person where
  start_row : ℕ
  start_col : ℕ
  end_row : ℕ
  end_col : ℕ

/-- The probability of two persons meeting on a grid network --/
def meeting_probability (grid : GridNetwork) (p1 p2 : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of meeting in a 4x3 grid --/
theorem meeting_probability_4x3_grid :
  let grid : GridNetwork := ⟨4, 3⟩
  let person1 : Person := ⟨0, 0, 3, 4⟩  -- A to B
  let person2 : Person := ⟨3, 4, 0, 0⟩  -- B to A
  meeting_probability grid person1 person2 = 1/5 := by
  sorry

end meeting_probability_4x3_grid_l1593_159314


namespace line_plane_parallelism_l1593_159317

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α β : Plane) :
  contained_in a β → parallel α β → line_parallel_to_plane a α :=
sorry

end line_plane_parallelism_l1593_159317


namespace min_trig_fraction_l1593_159335

theorem min_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ (17/8) * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end min_trig_fraction_l1593_159335


namespace hyperbola_eccentricity_l1593_159330

/-- A hyperbola with the given properties has eccentricity √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (M N Q E P : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (M.1^2 / a^2 - M.2^2 / b^2 = 1) →
  (N.1^2 / a^2 - N.2^2 / b^2 = 1) →
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) →
  N = (-M.1, -M.2) →
  Q = (M.1, -M.2) →
  E = (M.1, -3 * M.2) →
  (P.2 - M.2) * (P.1 - M.1) = -(N.2 - M.2) * (N.1 - M.1) →
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l1593_159330


namespace power_calculation_l1593_159305

theorem power_calculation : (-2 : ℝ)^2023 * (1/2 : ℝ)^2022 = -2 := by sorry

end power_calculation_l1593_159305


namespace no_lcm_arithmetic_progression_l1593_159379

theorem no_lcm_arithmetic_progression (n : ℕ) (h : n > 100) :
  ¬ ∃ (S : Finset ℕ) (d : ℕ) (first : ℕ),
    S.card = n ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y) ∧
    d > 0 ∧
    ∃ (f : Finset ℕ),
      f.card = n * (n - 1) / 2 ∧
      (∀ x ∈ S, ∀ y ∈ S, x < y → Nat.lcm x y ∈ f) ∧
      (∀ i < n * (n - 1) / 2, first + i * d ∈ f) :=
by sorry

end no_lcm_arithmetic_progression_l1593_159379


namespace absolute_value_sum_range_l1593_159368

theorem absolute_value_sum_range (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ∈ Set.Iic 2 := by
  sorry

end absolute_value_sum_range_l1593_159368


namespace triangle_town_intersections_l1593_159367

/-- The number of intersections for n non-parallel lines in a plane where no three lines meet at a single point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Theorem: In a configuration of 10 non-parallel lines in a plane, 
    where no three lines intersect at a single point, 
    the maximum number of intersection points is 45 -/
theorem triangle_town_intersections :
  max_intersections 10 = 45 := by
  sorry

end triangle_town_intersections_l1593_159367


namespace inverse_prop_parallel_lines_interior_angles_l1593_159340

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := sorry

/-- Definition of interior alternate angles -/
def interior_alternate_angles_equal (l1 l2 : Line) : Prop := sorry

/-- The inverse proposition of a statement "if P, then Q" is "if Q, then P" -/
def inverse_proposition (P Q : Prop) : Prop :=
  (Q → P) = (¬P → ¬Q)

/-- Theorem stating the inverse proposition of the given statement -/
theorem inverse_prop_parallel_lines_interior_angles :
  inverse_proposition
    (∀ l1 l2 : Line, parallel l1 l2 → interior_alternate_angles_equal l1 l2)
    (∀ l1 l2 : Line, interior_alternate_angles_equal l1 l2 → parallel l1 l2) :=
by
  sorry

end inverse_prop_parallel_lines_interior_angles_l1593_159340


namespace max_m_plus_2n_l1593_159395

-- Define the sets A and B
def A : Set ℕ := {x | ∃ k : ℕ+, x = 2 * k - 1}
def B : Set ℕ := {x | ∃ k : ℕ+, x = 8 * k - 8}

-- Define a function to calculate the sum of m different elements from A
def sumA (m : ℕ) : ℕ := m^2

-- Define a function to calculate the sum of n different elements from B
def sumB (n : ℕ) : ℕ := 4 * n^2 - 4 * n

-- State the theorem
theorem max_m_plus_2n (m n : ℕ) :
  sumA m + sumB n ≤ 967 → m + 2 * n ≤ 44 :=
sorry

end max_m_plus_2n_l1593_159395


namespace video_game_spend_l1593_159370

/-- Calculates the amount spent on video games given total pocket money and fractions spent on other items --/
def video_game_expenditure (total : ℚ) (books : ℚ) (snacks : ℚ) (toys : ℚ) : ℚ :=
  total - (books * total + snacks * total + toys * total)

/-- Theorem stating that the amount spent on video games is 6 dollars --/
theorem video_game_spend :
  let total : ℚ := 40
  let books : ℚ := 2 / 5
  let snacks : ℚ := 1 / 4
  let toys : ℚ := 1 / 5
  video_game_expenditure total books snacks toys = 6 := by
  sorry

end video_game_spend_l1593_159370


namespace rectangle_area_l1593_159341

/-- Given a rectangle with perimeter 28 cm and length 9 cm, its area is 45 cm² -/
theorem rectangle_area (perimeter length : ℝ) (h1 : perimeter = 28) (h2 : length = 9) :
  let width := (perimeter - 2 * length) / 2
  length * width = 45 :=
by sorry

end rectangle_area_l1593_159341


namespace least_non_lucky_multiple_of_11_l1593_159380

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

def isMultipleOf11 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 11 * k

theorem least_non_lucky_multiple_of_11 :
  (∀ m : ℕ, m < 11 → ¬(isMultipleOf11 m ∧ ¬isLucky m)) ∧
  (isMultipleOf11 11 ∧ ¬isLucky 11) :=
sorry

end least_non_lucky_multiple_of_11_l1593_159380


namespace max_additional_plates_is_24_l1593_159392

def initial_plates : ℕ := 3 * 2 * 4

def scenario1 : ℕ := (3 + 2) * 2 * 4
def scenario2 : ℕ := 3 * 2 * (4 + 2)
def scenario3 : ℕ := (3 + 1) * 2 * (4 + 1)
def scenario4 : ℕ := (3 + 1) * (2 + 1) * 4

def max_additional_plates : ℕ := max scenario1 (max scenario2 (max scenario3 scenario4)) - initial_plates

theorem max_additional_plates_is_24 : max_additional_plates = 24 := by
  sorry

end max_additional_plates_is_24_l1593_159392


namespace sector_arc_length_l1593_159386

/-- Given a circular sector with area 60π cm² and central angle 150°, its arc length is 10π cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 60 * Real.pi ∧ angle = 150 → arc_length = 10 * Real.pi := by
  sorry

end sector_arc_length_l1593_159386


namespace binomial_coefficient_two_l1593_159371

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l1593_159371


namespace complex_equation_real_part_condition_l1593_159399

theorem complex_equation_real_part_condition (z : ℂ) (a b : ℝ) : 
  z * (z + 2*I) * (z + 4*I) = 1001*I → 
  z = a + b*I → 
  a > 0 → 
  b > 0 → 
  a * (a^2 - b^2 - 6*b - 8) = 0 := by
sorry

end complex_equation_real_part_condition_l1593_159399


namespace smallest_solution_l1593_159320

def equation (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6 ∧
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4)

theorem smallest_solution :
  ∀ x : ℝ, equation x → x ≥ 16 ∧ equation 16 := by sorry

end smallest_solution_l1593_159320


namespace boris_neighbors_l1593_159357

-- Define the type for people
inductive Person : Type
  | Arkady | Boris | Vera | Galya | Danya | Egor

-- Define the circle as a function from positions to people
def Circle := Fin 6 → Person

-- Define the conditions
def satisfies_conditions (c : Circle) : Prop :=
  -- Danya stands next to Vera, on her right side
  ∃ i, c i = Person.Vera ∧ c (i + 1) = Person.Danya
  -- Galya stands opposite Egor
  ∧ ∃ j, c j = Person.Egor ∧ c (j + 3) = Person.Galya
  -- Egor stands next to Danya
  ∧ ∃ k, c k = Person.Danya ∧ (c (k + 1) = Person.Egor ∨ c (k - 1) = Person.Egor)
  -- Arkady and Galya do not stand next to each other
  ∧ ∀ l, c l = Person.Arkady → c (l + 1) ≠ Person.Galya ∧ c (l - 1) ≠ Person.Galya

-- Theorem statement
theorem boris_neighbors (c : Circle) (h : satisfies_conditions c) :
  ∃ i, c i = Person.Boris ∧ 
    ((c (i - 1) = Person.Arkady ∧ c (i + 1) = Person.Galya) ∨
     (c (i - 1) = Person.Galya ∧ c (i + 1) = Person.Arkady)) :=
by
  sorry

end boris_neighbors_l1593_159357


namespace no_valid_function_l1593_159393

/-- The set M = {0, 1, 2, ..., 2022} -/
def M : Set Nat := Finset.range 2023

/-- The theorem stating that no function f satisfies both required conditions -/
theorem no_valid_function :
  ¬∃ (f : M → M → M),
    (∀ (a b : M), f a (f b a) = b) ∧
    (∀ (x : M), f x x ≠ x) := by
  sorry

end no_valid_function_l1593_159393


namespace number_difference_l1593_159385

theorem number_difference (L S : ℕ) (h1 : L = 1608) (h2 : L = 6 * S + 15) : L - S = 1343 := by
  sorry

end number_difference_l1593_159385


namespace binomial_coefficient_20_19_l1593_159350

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end binomial_coefficient_20_19_l1593_159350


namespace largest_result_operation_l1593_159396

theorem largest_result_operation : 
  let a := -1
  let b := -(1/2)
  let add_result := a + b
  let sub_result := a - b
  let mul_result := a * b
  let div_result := a / b
  (div_result > add_result) ∧ 
  (div_result > sub_result) ∧ 
  (div_result > mul_result) ∧
  (div_result = 2) := by
sorry

end largest_result_operation_l1593_159396


namespace price_increase_quantity_decrease_l1593_159375

theorem price_increase_quantity_decrease (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let original_cost := P * Q
  let new_price := P * 1.15
  let new_quantity := Q * 0.6
  let new_cost := new_price * new_quantity
  new_cost = original_cost * 0.69 :=
by sorry

end price_increase_quantity_decrease_l1593_159375


namespace remainder_17_power_1999_mod_29_l1593_159322

theorem remainder_17_power_1999_mod_29 : 17^1999 % 29 = 17 := by
  sorry

end remainder_17_power_1999_mod_29_l1593_159322


namespace larger_divided_by_smaller_l1593_159363

theorem larger_divided_by_smaller : 
  let a := 8
  let b := 22
  let larger := max a b
  let smaller := min a b
  larger / smaller = 2.75 := by sorry

end larger_divided_by_smaller_l1593_159363


namespace adult_ticket_cost_l1593_159342

theorem adult_ticket_cost 
  (total_tickets : ℕ) 
  (total_receipts : ℕ) 
  (child_ticket_cost : ℕ) 
  (child_tickets_sold : ℕ) 
  (h1 : total_tickets = 130) 
  (h2 : total_receipts = 840) 
  (h3 : child_ticket_cost = 4) 
  (h4 : child_tickets_sold = 90) : 
  (total_receipts - child_tickets_sold * child_ticket_cost) / (total_tickets - child_tickets_sold) = 12 := by
sorry

end adult_ticket_cost_l1593_159342


namespace geometry_theorem_l1593_159303

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (distinct_lines : m ≠ n) 
  (distinct_planes : α ≠ β) :
  (perpendicular m n → perpendicularLP m α → ¬subset n α → parallel n α) ∧
  (perpendicularLP m β → perpendicularPP α β → (parallel m α ∨ subset m α)) ∧
  (perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β) :=
sorry

end geometry_theorem_l1593_159303


namespace max_value_of_product_sum_l1593_159347

theorem max_value_of_product_sum (a b c : ℝ) (h : a + 3 * b + c = 5) :
  (∀ x y z : ℝ, x + 3 * y + z = 5 → a * b + a * c + b * c ≥ x * y + x * z + y * z) ∧
  a * b + a * c + b * c = 25 / 6 :=
sorry

end max_value_of_product_sum_l1593_159347


namespace bridge_length_calculation_l1593_159309

/-- Given a train crossing a bridge, calculate the length of the bridge. -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 295 →
  train_speed_kmh = 75 →
  crossing_time = 45 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 642.5 := by
  sorry

end bridge_length_calculation_l1593_159309


namespace balls_distribution_theorem_l1593_159384

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem balls_distribution_theorem :
  distribute_balls 5 3 = 150 :=
sorry

end balls_distribution_theorem_l1593_159384


namespace quadratic_equation_conversion_l1593_159387

theorem quadratic_equation_conversion (x : ℝ) : 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n) → 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n ∧ m + n = 5) := by
sorry

end quadratic_equation_conversion_l1593_159387


namespace sams_age_l1593_159338

/-- Given that Sam and Drew have a combined age of 54 and Sam is half of Drew's age,
    prove that Sam is 18 years old. -/
theorem sams_age (total_age : ℕ) (drews_age : ℕ) (sams_age : ℕ) 
    (h1 : total_age = 54)
    (h2 : sams_age + drews_age = total_age)
    (h3 : sams_age = drews_age / 2) : 
  sams_age = 18 := by
  sorry

end sams_age_l1593_159338


namespace maddie_had_15_books_l1593_159381

/-- The number of books Maddie had -/
def maddie_books : ℕ := sorry

/-- The number of books Luisa had -/
def luisa_books : ℕ := 18

/-- The number of books Amy had -/
def amy_books : ℕ := 6

/-- Theorem stating that Maddie had 15 books -/
theorem maddie_had_15_books : maddie_books = 15 := by
  have h1 : amy_books + luisa_books = maddie_books + 9 := sorry
  sorry

end maddie_had_15_books_l1593_159381


namespace glass_volume_l1593_159390

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) 
  (h_pessimist : 0.4 * V = V - 0.6 * V) 
  (h_optimist : 0.6 * V = V - 0.4 * V) 
  (h_difference : 0.6 * V - 0.4 * V = 46) : 
  V = 230 := by
  sorry

end glass_volume_l1593_159390


namespace union_P_Q_l1593_159329

def P : Set ℝ := { x | -1 < x ∧ x < 1 }
def Q : Set ℝ := { x | x^2 - 2*x < 0 }

theorem union_P_Q : P ∪ Q = { x | -1 < x ∧ x < 2 } := by sorry

end union_P_Q_l1593_159329


namespace xoxoxox_probability_l1593_159376

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem xoxoxox_probability :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = (1 : ℚ) / 35 :=
sorry

end xoxoxox_probability_l1593_159376


namespace equation_implication_l1593_159326

theorem equation_implication (x y : ℝ) : 
  x^2 - 3*x*y + 2*y^2 + x - y = 0 → 
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 → 
  x*y - 12*x + 15*y = 0 := by
sorry

end equation_implication_l1593_159326


namespace f_has_two_zeros_l1593_159313

def f (x : ℝ) := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b := by
  sorry

end f_has_two_zeros_l1593_159313


namespace concrete_mixture_percentage_l1593_159324

/-- Proves that mixing 7 tons of 80% cement mixture with 3 tons of 20% cement mixture
    results in a 62% cement mixture when making 10 tons of concrete. -/
theorem concrete_mixture_percentage : 
  let total_concrete : ℝ := 10
  let mixture_80_percent : ℝ := 7
  let mixture_20_percent : ℝ := total_concrete - mixture_80_percent
  let cement_in_80_percent : ℝ := mixture_80_percent * 0.8
  let cement_in_20_percent : ℝ := mixture_20_percent * 0.2
  let total_cement : ℝ := cement_in_80_percent + cement_in_20_percent
  total_cement / total_concrete = 0.62 := by
sorry

end concrete_mixture_percentage_l1593_159324


namespace train_length_proof_l1593_159308

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 52 km/hr and the slower train at 36 km/hr,
    if the faster train passes the slower train in 36 seconds,
    then the length of each train is 80 meters. -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) : 
  faster_speed = 52 →
  slower_speed = 36 →
  passing_time = 36 →
  (faster_speed - slower_speed) * passing_time * (5 / 18) = 2 * train_length →
  train_length = 80 := by
  sorry

#check train_length_proof

end train_length_proof_l1593_159308


namespace slope_angle_range_l1593_159319

/-- Given two lines and their intersection in the first quadrant, 
    prove the range of the slope angle of one line -/
theorem slope_angle_range (k : ℝ) : 
  let l1 : ℝ → ℝ := λ x => k * x - Real.sqrt 3
  let l2 : ℝ → ℝ := λ x => (6 - 2 * x) / 3
  let x_intersect := (3 * Real.sqrt 3 + 6) / (2 + 3 * k)
  let y_intersect := (6 * k - 2 * Real.sqrt 3) / (2 + 3 * k)
  (x_intersect > 0 ∧ y_intersect > 0) →
  let θ := Real.arctan k
  θ > π / 6 ∧ θ < π / 2 := by
sorry

end slope_angle_range_l1593_159319


namespace zero_not_in_range_of_g_l1593_159332

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then
    ⌊(Real.cos x) / (x + 3)⌋
  else
    0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
sorry

end zero_not_in_range_of_g_l1593_159332


namespace base3_20202_equals_182_l1593_159352

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The theorem stating that the base-3 number 20202 is equal to 182 in base 10 -/
theorem base3_20202_equals_182 : base3_to_base10 [2, 0, 2, 0, 2] = 182 := by
  sorry

#eval base3_to_base10 [2, 0, 2, 0, 2]

end base3_20202_equals_182_l1593_159352


namespace pool_drain_time_l1593_159336

/-- Represents the pool draining problem -/
structure PoolDraining where
  capacity : ℝ
  fillTime : ℝ
  drainTime : ℝ
  elapsedTime : ℝ
  remainingWater : ℝ

/-- Theorem stating the solution to the pool draining problem -/
theorem pool_drain_time (p : PoolDraining) 
  (h_capacity : p.capacity = 120)
  (h_fillTime : p.fillTime = 6)
  (h_elapsedTime : p.elapsedTime = 3)
  (h_remainingWater : p.remainingWater = 90) :
  p.drainTime = 4 := by
  sorry


end pool_drain_time_l1593_159336


namespace geometric_sequence_a10_l1593_159348

/-- A geometric sequence with positive common ratio -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 2 = 1 →
  a 10 = 16 := by
sorry

end geometric_sequence_a10_l1593_159348


namespace compound_interest_rate_l1593_159307

theorem compound_interest_rate : 
  ∀ (P : ℝ) (A : ℝ) (I : ℝ) (t : ℕ) (r : ℝ),
  A = 19828.80 →
  I = 2828.80 →
  t = 2 →
  A = P + I →
  A = P * (1 + r) ^ t →
  r = 0.08 :=
by sorry

end compound_interest_rate_l1593_159307


namespace simplify_and_evaluate_algebraic_expression_value_l1593_159383

-- Problem 1
theorem simplify_and_evaluate :
  let x : ℤ := -3
  (x^2 + 4*x - (2*x^2 - x + x^2) - (3*x - 1)) = -23 := by sorry

-- Problem 2
theorem algebraic_expression_value (m n : ℤ) 
  (h1 : m + n = 2) (h2 : m * n = -3) :
  2*(m*n + (-3*m)) - 3*(2*n - m*n) = -27 := by sorry

end simplify_and_evaluate_algebraic_expression_value_l1593_159383


namespace mikes_shopping_l1593_159353

/-- Mike's shopping problem -/
theorem mikes_shopping (food wallet shirt : ℝ) 
  (h1 : shirt = wallet / 3)
  (h2 : wallet = food + 60)
  (h3 : shirt + wallet + food = 150) :
  food = 52.5 := by
  sorry

end mikes_shopping_l1593_159353


namespace square_perimeter_area_l1593_159377

/-- Theorem: A square with a perimeter of 24 inches has an area of 36 square inches. -/
theorem square_perimeter_area : 
  ∀ (side : ℝ), 
  (4 * side = 24) → (side * side = 36) :=
by
  sorry

end square_perimeter_area_l1593_159377


namespace determine_opposite_resident_l1593_159358

/-- Represents a resident on the hexagonal street -/
inductive Resident
| Knight
| Liar

/-- Represents a vertex of the hexagonal street -/
def Vertex := Fin 6

/-- Represents the street layout -/
structure HexagonalStreet where
  residents : Vertex → Resident

/-- Represents a letter asking about neighbor relationships -/
structure Letter where
  sender : Vertex
  recipient : Vertex
  askedAbout : Vertex

/-- Determines if two vertices are neighbors in a regular hexagon -/
def areNeighbors (v1 v2 : Vertex) : Bool :=
  (v1.val + 1) % 6 = v2.val ∨ (v1.val + 5) % 6 = v2.val

/-- The main theorem stating that it's possible to determine the opposite resident with at most 4 letters -/
theorem determine_opposite_resident (street : HexagonalStreet) (start : Vertex) :
  ∃ (letters : List Letter), letters.length ≤ 4 ∧
    ∃ (opposite : Vertex), (start.val + 3) % 6 = opposite.val ∧
      (∀ (response : Letter → Bool), 
        ∃ (deduced_resident : Resident), street.residents opposite = deduced_resident) :=
  sorry

end determine_opposite_resident_l1593_159358


namespace sum_13_impossible_l1593_159337

-- Define the type for dice faces
def DieFace := Fin 6

-- Define the function to calculate the sum of two dice
def diceSum (d1 d2 : DieFace) : Nat := d1.val + d2.val + 2

-- Theorem statement
theorem sum_13_impossible :
  ¬ ∃ (d1 d2 : DieFace), diceSum d1 d2 = 13 := by
  sorry

end sum_13_impossible_l1593_159337


namespace geometric_sequence_common_ratio_l1593_159349

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (h_arithmetic : a 3 + a 4 = a 5) :
  q = (Real.sqrt 5 + 1) / 2 := by
sorry

end geometric_sequence_common_ratio_l1593_159349


namespace stratified_sampling_equality_l1593_159327

/-- Represents the number of people in each age group -/
structure Population where
  elderly : ℕ
  middleAged : ℕ

/-- Represents the number of people selected from each age group -/
structure Selected where
  elderly : ℕ
  middleAged : ℕ

/-- Checks if the selection maintains equal probability across strata -/
def isEqualProbability (pop : Population) (sel : Selected) : Prop :=
  (sel.elderly : ℚ) / pop.elderly = (sel.middleAged : ℚ) / pop.middleAged

theorem stratified_sampling_equality 
  (pop : Population) (sel : Selected) 
  (h1 : pop.elderly = 140) 
  (h2 : pop.middleAged = 210) 
  (h3 : sel.elderly = 4) 
  (h4 : isEqualProbability pop sel) : 
  sel.middleAged = 6 := by
  sorry

#check stratified_sampling_equality

end stratified_sampling_equality_l1593_159327


namespace road_building_equation_l1593_159351

theorem road_building_equation (x : ℝ) 
  (h_positive : x > 0) 
  (h_team_a_length : 9 > 0) 
  (h_team_b_length : 12 > 0) 
  (h_team_b_faster : x + 1 > x) : 
  9 / x - 12 / (x + 1) = 1 / 2 := by
  sorry

end road_building_equation_l1593_159351


namespace lindas_broken_eggs_l1593_159334

/-- The number of eggs Linda broke -/
def broken_eggs (initial_white : ℕ) (initial_brown : ℕ) (total_after : ℕ) : ℕ :=
  initial_white + initial_brown - total_after

theorem lindas_broken_eggs :
  let initial_brown := 5
  let initial_white := 3 * initial_brown
  let total_after := 12
  broken_eggs initial_white initial_brown total_after = 8 := by
  sorry

#eval broken_eggs (3 * 5) 5 12  -- Should output 8

end lindas_broken_eggs_l1593_159334


namespace number_divided_by_005_equals_1500_l1593_159373

theorem number_divided_by_005_equals_1500 (x : ℝ) : x / 0.05 = 1500 → x = 75 := by
  sorry

end number_divided_by_005_equals_1500_l1593_159373


namespace touchdown_points_l1593_159344

theorem touchdown_points : ℕ → Prop :=
  fun p =>
    let team_a_touchdowns : ℕ := 7
    let team_b_touchdowns : ℕ := 9
    let point_difference : ℕ := 14
    (team_b_touchdowns * p = team_a_touchdowns * p + point_difference) →
    p = 7

-- Proof
example : touchdown_points 7 := by
  sorry

end touchdown_points_l1593_159344


namespace triangle_angle_measure_l1593_159366

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 70 → 
  E = 2 * F + 18 → 
  D + E + F = 180 → 
  F = 92 / 3 :=
by sorry

end triangle_angle_measure_l1593_159366


namespace trigonometric_identities_l1593_159389

theorem trigonometric_identities (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  ((3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 8/9) ∧ 
  (Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 1 = 13/10) := by
  sorry

end trigonometric_identities_l1593_159389


namespace log_9_81_equals_2_l1593_159394

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_9_81_equals_2 : log 9 81 = 2 := by
  sorry

end log_9_81_equals_2_l1593_159394


namespace baked_goods_distribution_l1593_159339

/-- Calculates the number of items not placed in containers --/
def itemsNotPlaced (totalItems : Nat) (itemsPerContainer : Nat) : Nat :=
  totalItems % itemsPerContainer

theorem baked_goods_distribution (gingerbreadCookies sugarCookies fruitTarts : Nat) 
  (gingerbreadPerJar sugarPerBox tartsPerBox : Nat) :
  gingerbreadCookies = 47 → 
  sugarCookies = 78 → 
  fruitTarts = 36 → 
  gingerbreadPerJar = 6 → 
  sugarPerBox = 9 → 
  tartsPerBox = 4 → 
  (itemsNotPlaced gingerbreadCookies gingerbreadPerJar = 5 ∧ 
   itemsNotPlaced sugarCookies sugarPerBox = 6 ∧ 
   itemsNotPlaced fruitTarts tartsPerBox = 0) := by
  sorry

#eval itemsNotPlaced 47 6  -- Should output 5
#eval itemsNotPlaced 78 9  -- Should output 6
#eval itemsNotPlaced 36 4  -- Should output 0

end baked_goods_distribution_l1593_159339


namespace associate_professor_pencils_l1593_159323

theorem associate_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 8 →
    P * A + B = 10 →
    A + 2 * B = 14 →
    P = 2 :=
by
  sorry

end associate_professor_pencils_l1593_159323


namespace proposition_falsity_l1593_159364

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 6) : 
  ¬ P 5 := by
sorry

end proposition_falsity_l1593_159364


namespace max_sum_on_circle_l1593_159300

theorem max_sum_on_circle (x y : ℤ) : x^2 + y^2 = 25 → x + y ≤ 7 := by
  sorry

end max_sum_on_circle_l1593_159300


namespace joan_wednesday_spending_l1593_159333

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := 18 - 14

/-- The total amount Joan spent in half-dollars -/
def total_half_dollars : ℕ := 18

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

theorem joan_wednesday_spending :
  wednesday_half_dollars = 4 :=
by sorry

end joan_wednesday_spending_l1593_159333


namespace grass_seed_coverage_l1593_159328

/-- Calculates the area covered by one bag of grass seed given the dimensions of a rectangular lawn
and the total area covered by a known number of bags. -/
theorem grass_seed_coverage 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (extra_area : ℝ) 
  (num_bags : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : extra_area = 208)
  (h4 : num_bags = 4) :
  (lawn_length * lawn_width + extra_area) / num_bags = 250 :=
by sorry

end grass_seed_coverage_l1593_159328


namespace no_solution_implies_m_greater_2023_l1593_159301

theorem no_solution_implies_m_greater_2023 (m : ℝ) :
  (∀ x : ℝ, ¬(x ≥ m ∧ x ≤ 2023)) → m > 2023 := by
  sorry

end no_solution_implies_m_greater_2023_l1593_159301
