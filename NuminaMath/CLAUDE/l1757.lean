import Mathlib

namespace xiaogangSavings_l1757_175743

/-- Represents the correct inequality for Xiaogang's savings plan -/
theorem xiaogangSavings (x : ℕ) (initialSavings : ℕ) (monthlySavings : ℕ) (targetAmount : ℕ) : 
  initialSavings = 50 → monthlySavings = 30 → targetAmount = 280 →
  (monthlySavings * x + initialSavings ≥ targetAmount ↔ 
   x ≥ (targetAmount - initialSavings) / monthlySavings) :=
by sorry

end xiaogangSavings_l1757_175743


namespace perfect_square_trinomial_m_values_l1757_175735

theorem perfect_square_trinomial_m_values (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end perfect_square_trinomial_m_values_l1757_175735


namespace cylindrical_cans_radius_l1757_175719

/-- Proves that for two cylindrical cans with equal volumes, where one can is four times taller than the other,
    if the taller can has a radius of 5 units, then the shorter can has a radius of 10 units. -/
theorem cylindrical_cans_radius (volume : ℝ) (h : ℝ) (r : ℝ) :
  volume = 500 ∧
  volume = π * 5^2 * (4 * h) ∧
  volume = π * r^2 * h →
  r = 10 := by
  sorry

end cylindrical_cans_radius_l1757_175719


namespace profit_share_ratio_l1757_175784

/-- The ratio of profit shares for two investors is proportional to their investments -/
theorem profit_share_ratio (p_investment q_investment : ℕ) 
  (hp : p_investment = 52000)
  (hq : q_investment = 65000) :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    p_investment * 5 = q_investment * 4 * k ∧ 
    q_investment * 4 = p_investment * 4 * k :=
sorry

end profit_share_ratio_l1757_175784


namespace amusement_park_group_composition_l1757_175752

theorem amusement_park_group_composition :
  let total_cost : ℕ := 720
  let adult_price : ℕ := 15
  let child_price : ℕ := 8
  let num_children : ℕ := 15
  let num_adults : ℕ := (total_cost - child_price * num_children) / adult_price
  num_adults - num_children = 25 := by
  sorry

end amusement_park_group_composition_l1757_175752


namespace overlap_area_of_intersecting_rectangles_l1757_175727

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of overlap between two rectangles intersecting at 45 degrees -/
noncomputable def overlapArea (r1 r2 : Rectangle) : ℝ :=
  min r1.width r2.width * min r1.height r2.height

/-- The theorem stating the area of the overlapping region -/
theorem overlap_area_of_intersecting_rectangles :
  let r1 : Rectangle := { width := 3, height := 12 }
  let r2 : Rectangle := { width := 4, height := 10 }
  rectangleArea r1 + rectangleArea r2 - overlapArea r1 r2 = 64 := by
  sorry

end overlap_area_of_intersecting_rectangles_l1757_175727


namespace quadratic_comparison_l1757_175725

/-- Proves that for a quadratic function y = a(x-1)^2 + 3 where a < 0,
    if (-1, y₁) and (2, y₂) are points on the graph, then y₁ < y₂ -/
theorem quadratic_comparison (a : ℝ) (y₁ y₂ : ℝ)
    (h₁ : a < 0)
    (h₂ : y₁ = a * (-1 - 1)^2 + 3)
    (h₃ : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
  sorry

end quadratic_comparison_l1757_175725


namespace distance_between_harper_and_jack_l1757_175729

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (jack_position : ℕ) : ℕ :=
  race_length - jack_position

/-- Theorem: The distance between Harper and Jack when Harper finished the race is 848 meters --/
theorem distance_between_harper_and_jack :
  let race_length_meters : ℕ := 1000  -- 1 km = 1000 meters
  let jack_position : ℕ := 152
  distance_between_runners race_length_meters jack_position = 848 := by
  sorry

end distance_between_harper_and_jack_l1757_175729


namespace linear_function_not_in_first_quadrant_l1757_175799

theorem linear_function_not_in_first_quadrant :
  ∀ x y : ℝ, y = -2 * x - 1 → ¬(x > 0 ∧ y > 0) := by
  sorry

end linear_function_not_in_first_quadrant_l1757_175799


namespace jellybean_problem_l1757_175721

theorem jellybean_problem (initial : ℕ) (first_removal : ℕ) (second_removal : ℕ) (final : ℕ) 
  (h1 : initial = 37)
  (h2 : first_removal = 15)
  (h3 : second_removal = 4)
  (h4 : final = 23) :
  ∃ (added_back : ℕ), initial - first_removal + added_back - second_removal = final ∧ added_back = 5 := by
  sorry

end jellybean_problem_l1757_175721


namespace constant_term_expansion_l1757_175797

/-- Given that the constant term in the expansion of (x + a/√x)^6 is 15, 
    prove that the positive value of a is 1. -/
theorem constant_term_expansion (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), (x + a / Real.sqrt x)^6 = 15 + x * (1 + 1/x)) → a = 1 := by
  sorry

end constant_term_expansion_l1757_175797


namespace trig_identity_l1757_175704

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end trig_identity_l1757_175704


namespace diagonal_passes_at_least_length_squares_l1757_175746

/-- Represents an irregular hexagon composed of unit squares -/
structure IrregularHexagon where
  total_squares : ℕ
  length : ℕ
  width1 : ℕ
  width2 : ℕ

/-- The minimum number of squares a diagonal passes through -/
def diagonal_squares_count (h : IrregularHexagon) : ℕ :=
  h.length

/-- Theorem stating that the diagonal passes through at least as many squares as the length -/
theorem diagonal_passes_at_least_length_squares (h : IrregularHexagon)
  (h_total : h.total_squares = 78)
  (h_length : h.length = 12)
  (h_width1 : h.width1 = 8)
  (h_width2 : h.width2 = 6) :
  diagonal_squares_count h ≥ h.length :=
sorry

end diagonal_passes_at_least_length_squares_l1757_175746


namespace find_other_number_l1757_175714

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 2 * b = 120) (h2 : a = 26 ∨ b = 26) : 
  (a ≠ 26 → a = 21) ∧ (b ≠ 26 → b = 21) :=
sorry

end find_other_number_l1757_175714


namespace one_nonnegative_solution_l1757_175785

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -9*x :=
by sorry

end one_nonnegative_solution_l1757_175785


namespace apple_tree_production_l1757_175733

theorem apple_tree_production (first_season : ℕ) : 
  (first_season : ℝ) + 0.8 * first_season + 1.6 * first_season = 680 →
  first_season = 200 := by
sorry

end apple_tree_production_l1757_175733


namespace sum_of_parts_l1757_175736

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 60) (h2 : y = 45) (h3 : x ≥ 0) (h4 : y ≥ 0) :
  10 * x + 22 * y = 1140 := by
  sorry

end sum_of_parts_l1757_175736


namespace complex_fraction_equality_l1757_175715

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end complex_fraction_equality_l1757_175715


namespace equilateral_cone_central_angle_l1757_175767

/-- Represents a cone with an equilateral triangle as its axial section -/
structure EquilateralCone where
  /-- The radius of the base of the cone -/
  r : ℝ
  /-- The slant height of the cone, which is twice the radius for an equilateral axial section -/
  slant_height : ℝ
  /-- Condition that the slant height is twice the radius -/
  slant_height_eq : slant_height = 2 * r

/-- The central angle of the side surface development of an equilateral cone is π radians (180°) -/
theorem equilateral_cone_central_angle (cone : EquilateralCone) :
  Real.pi = (2 * Real.pi * cone.r) / cone.slant_height :=
by sorry

end equilateral_cone_central_angle_l1757_175767


namespace sum_of_solutions_l1757_175788

theorem sum_of_solutions (S : ℝ) : 
  ∃ (N₁ N₂ : ℝ), N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ 
  (6 * N₁ + 2 / N₁ = S) ∧ 
  (6 * N₂ + 2 / N₂ = S) ∧ 
  (N₁ + N₂ = S / 6) := by
sorry

end sum_of_solutions_l1757_175788


namespace quadratic_roots_property_l1757_175765

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end quadratic_roots_property_l1757_175765


namespace president_secretary_selection_l1757_175787

theorem president_secretary_selection (n : ℕ) (h : n = 6) :
  (n * (n - 1) : ℕ) = 30 := by
  sorry

end president_secretary_selection_l1757_175787


namespace tetrahedron_altitude_volume_inequality_l1757_175764

/-- A tetrahedron with volume and altitudes -/
structure Tetrahedron where
  volume : ℝ
  altitude : Fin 4 → ℝ

/-- Predicate to check if a tetrahedron is right-angled -/
def isRightAngled (t : Tetrahedron) : Prop := sorry

/-- Theorem stating the relationship between altitudes and volume of a tetrahedron -/
theorem tetrahedron_altitude_volume_inequality (t : Tetrahedron) :
  ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    t.altitude i * t.altitude j * t.altitude k ≤ 6 * t.volume ∧
    (t.altitude i * t.altitude j * t.altitude k = 6 * t.volume ↔ isRightAngled t) := by
  sorry

end tetrahedron_altitude_volume_inequality_l1757_175764


namespace pikes_caught_l1757_175705

theorem pikes_caught (total_fishes sturgeons herrings : ℕ) 
  (h1 : total_fishes = 145)
  (h2 : sturgeons = 40)
  (h3 : herrings = 75) :
  total_fishes - (sturgeons + herrings) = 30 := by
  sorry

end pikes_caught_l1757_175705


namespace inequality_proof_l1757_175751

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) ≤ (Real.sqrt 33 + 1) / 4 := by
  sorry

end inequality_proof_l1757_175751


namespace hyperbola_third_point_x_squared_l1757_175750

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  opens_horizontally : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The theorem states that for a specific hyperbola, the square of the x-coordinate of its third point is 361/36. -/
theorem hyperbola_third_point_x_squared (h : Hyperbola) 
  (h_center : h.center = (1, 0))
  (h_orientation : h.opens_horizontally = true)
  (h_point1 : h.point1 = (0, 3))
  (h_point2 : h.point2 = (1, -4))
  (h_point3 : h.point3.2 = -1) :
  (h.point3.1)^2 = 361/36 := by
  sorry

end hyperbola_third_point_x_squared_l1757_175750


namespace no_valid_propositions_l1757_175702

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line) (α β : Plane)
variable (h1 : perpendicular m α)
variable (h2 : contains β n)

-- State the theorem
theorem no_valid_propositions :
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular m α → contains β n → 
    ((parallel_planes α β → perpendicular_lines m n) ∧
     (perpendicular_lines m n → parallel_planes α β) ∧
     (parallel_lines m n → perpendicular_planes α β) ∧
     (perpendicular_planes α β → parallel_lines m n))) :=
sorry

end no_valid_propositions_l1757_175702


namespace barbaras_allowance_l1757_175716

theorem barbaras_allowance 
  (watch_cost : ℕ) 
  (current_savings : ℕ) 
  (weeks_left : ℕ) 
  (h1 : watch_cost = 100)
  (h2 : current_savings = 20)
  (h3 : weeks_left = 16) :
  (watch_cost - current_savings) / weeks_left = 5 :=
by sorry

end barbaras_allowance_l1757_175716


namespace expression_evaluation_l1757_175747

theorem expression_evaluation : 
  let x : ℕ := 3
  x + x^2 * (x^(x^2)) = 177150 := by
sorry

end expression_evaluation_l1757_175747


namespace difference_of_squares_l1757_175742

theorem difference_of_squares (x : ℝ) : (2 + 3*x) * (2 - 3*x) = 4 - 9*x^2 := by
  sorry

end difference_of_squares_l1757_175742


namespace sharon_in_middle_l1757_175748

-- Define the set of people
inductive Person : Type
  | Maren : Person
  | Aaron : Person
  | Sharon : Person
  | Darren : Person
  | Karen : Person

-- Define the seating arrangement as a function from position (1 to 5) to Person
def Seating := Fin 5 → Person

-- Define the constraints
def satisfies_constraints (s : Seating) : Prop :=
  -- Maren sat in the last car
  s 5 = Person.Maren ∧
  -- Aaron sat directly behind Sharon
  (∃ i : Fin 4, s i = Person.Sharon ∧ s (i.succ) = Person.Aaron) ∧
  -- Darren sat directly behind Karen
  (∃ i : Fin 4, s i = Person.Karen ∧ s (i.succ) = Person.Darren) ∧
  -- At least one person sat between Aaron and Maren
  (∃ i j : Fin 5, i < j ∧ j < 5 ∧ s i = Person.Aaron ∧ s j ≠ Person.Maren ∧ s (j+1) = Person.Maren)

-- Theorem: Given the constraints, Sharon must be in the middle car
theorem sharon_in_middle (s : Seating) (h : satisfies_constraints s) : s 3 = Person.Sharon :=
sorry

end sharon_in_middle_l1757_175748


namespace emilia_berry_cobbler_l1757_175731

/-- The number of cartons of berries needed for Emilia's berry cobbler -/
def total_cartons (strawberry_cartons blueberry_cartons additional_cartons : ℕ) : ℕ :=
  strawberry_cartons + blueberry_cartons + additional_cartons

/-- Theorem stating that the total number of cartons is 42 given the specific quantities -/
theorem emilia_berry_cobbler : total_cartons 2 7 33 = 42 := by
  sorry

end emilia_berry_cobbler_l1757_175731


namespace planes_parallel_from_perpendicular_lines_l1757_175779

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_perpendicular_lines 
  (α β : Plane) (m n : Line) :
  perpendicular m α → 
  perpendicular n β → 
  line_parallel m n → 
  parallel α β :=
by sorry

end planes_parallel_from_perpendicular_lines_l1757_175779


namespace trigonometric_identity_l1757_175707

theorem trigonometric_identity : 
  - Real.sin (133 * π / 180) * Real.cos (197 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end trigonometric_identity_l1757_175707


namespace probability_problem_l1757_175737

-- Define the sample space and events
def Ω : Type := Unit
def A₁ : Set Ω := sorry
def A₂ : Set Ω := sorry
def A₃ : Set Ω := sorry
def B : Set Ω := sorry

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Theorem statement
theorem probability_problem :
  -- 1. A₁, A₂, and A₃ are pairwise mutually exclusive
  (A₁ ∩ A₂ = ∅ ∧ A₁ ∩ A₃ = ∅ ∧ A₂ ∩ A₃ = ∅) ∧
  -- 2. P(B|A₁) = 1/3
  P B / P A₁ = 1/3 ∧
  -- 3. P(B) = 19/48
  P B = 19/48 ∧
  -- 4. A₂ and B are not independent events
  P (A₂ ∩ B) ≠ P A₂ * P B :=
by sorry

end probability_problem_l1757_175737


namespace teal_color_survey_l1757_175739

theorem teal_color_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  more_green = 90 →
  both = 40 →
  neither = 25 →
  ∃ (more_blue : ℕ), more_blue = 75 ∧ 
    more_blue + (more_green - both) + neither = total :=
by sorry

end teal_color_survey_l1757_175739


namespace line_symmetry_l1757_175740

/-- Given two lines l₁ and l, prove that l₂ is symmetric to l₁ with respect to l -/
theorem line_symmetry (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 4 = 0
  let l : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 4 * y - 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 6 = 0
  (∀ x y, l₁ x y ↔ l₂ x y) ∧ 
  (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → 
    ∃ x₀ y₀, l x₀ y₀ ∧ 
    (x₀ - x₁)^2 + (y₀ - y₁)^2 = (x₀ - x₂)^2 + (y₀ - y₂)^2) :=
by sorry

end line_symmetry_l1757_175740


namespace blue_has_most_marbles_blue_greater_than_others_l1757_175780

/-- Represents the colors of marbles -/
inductive Color
| Red
| Blue
| Yellow

/-- Represents the marble counting problem -/
structure MarbleCounting where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- The conditions of the marble counting problem -/
def marbleProblem : MarbleCounting where
  total := 24
  red := 24 / 4
  blue := 24 / 4 + 6
  yellow := 24 - (24 / 4 + (24 / 4 + 6))

/-- Function to determine which color has the most marbles -/
def mostMarbles (mc : MarbleCounting) : Color :=
  if mc.blue > mc.red ∧ mc.blue > mc.yellow then Color.Blue
  else if mc.red > mc.blue ∧ mc.red > mc.yellow then Color.Red
  else Color.Yellow

/-- Theorem stating that blue has the most marbles in the given problem -/
theorem blue_has_most_marbles :
  mostMarbles marbleProblem = Color.Blue :=
by
  sorry

/-- Theorem proving that the number of blue marbles is greater than both red and yellow -/
theorem blue_greater_than_others (mc : MarbleCounting) :
  mc.blue > mc.red ∧ mc.blue > mc.yellow →
  mostMarbles mc = Color.Blue :=
by
  sorry

end blue_has_most_marbles_blue_greater_than_others_l1757_175780


namespace ghost_entrance_exit_ways_l1757_175789

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways Georgie can enter and exit the mansion is 56 -/
theorem ghost_entrance_exit_ways : num_ways = 56 := by
  sorry

end ghost_entrance_exit_ways_l1757_175789


namespace inequality_range_l1757_175724

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
sorry

end inequality_range_l1757_175724


namespace curve_area_range_l1757_175774

theorem curve_area_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0 → π * ((x + m)^2 + y^2) ≥ 4 * π) →
  m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6 :=
by sorry

end curve_area_range_l1757_175774


namespace fifteen_to_binary_l1757_175732

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem fifteen_to_binary :
  decimal_to_binary 15 = [1, 1, 1, 1] :=
by sorry

end fifteen_to_binary_l1757_175732


namespace three_teams_of_four_from_twelve_l1757_175795

-- Define the number of participants
def n : ℕ := 12

-- Define the number of teams
def k : ℕ := 3

-- Define the number of players per team
def m : ℕ := 4

-- Theorem statement
theorem three_teams_of_four_from_twelve (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m) / (Nat.factorial k) = 5775 := by
  sorry

end three_teams_of_four_from_twelve_l1757_175795


namespace exists_color_with_all_distances_l1757_175775

-- Define a type for colors
inductive Color
| Yellow
| Red

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a distance function between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem exists_color_with_all_distances :
  ∃ c : Color, ∀ x : ℝ, x > 0 → ∃ p1 p2 : Point,
    coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by sorry

end exists_color_with_all_distances_l1757_175775


namespace oliver_age_l1757_175796

/-- Given the ages of Mark, Nina, and Oliver, prove that Oliver is 22 years old. -/
theorem oliver_age (m n o : ℕ) : 
  (m + n + o) / 3 = 12 →  -- Average age is 12
  o - 5 = 2 * n →  -- Five years ago, Oliver was twice Nina's current age
  m + 2 = (4 * (n + 2)) / 5 →  -- In 2 years, Mark's age will be 4/5 of Nina's
  m + 4 + n + 4 + o + 4 = 60 →  -- In 4 years, total age will be 60
  o = 22 := by
  sorry

end oliver_age_l1757_175796


namespace pen_pencil_ratio_l1757_175756

theorem pen_pencil_ratio : 
  ∀ (num_pencils num_pens : ℕ),
    num_pencils = 42 →
    num_pencils = num_pens + 7 →
    (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end pen_pencil_ratio_l1757_175756


namespace quadratic_one_solution_l1757_175762

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) ↔ (m = 12 * Real.sqrt 3 ∨ m = -12 * Real.sqrt 3) :=
sorry

end quadratic_one_solution_l1757_175762


namespace pen_pricing_gain_percentage_l1757_175771

theorem pen_pricing_gain_percentage 
  (cost_price selling_price : ℝ) 
  (h : 20 * cost_price = 12 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 200 / 3 :=
by sorry

end pen_pricing_gain_percentage_l1757_175771


namespace one_correct_statement_l1757_175769

theorem one_correct_statement (a b : ℤ) : 
  (∃! n : Nat, n < 3 ∧ n > 0 ∧
    ((n = 1 → (Even (a + 5*b) → Even (a - 7*b))) ∧
     (n = 2 → ((a + b) % 3 = 0 → a % 3 = 0 ∧ b % 3 = 0)) ∧
     (n = 3 → (Prime (a + b) → ¬ Prime (a - b))))) := by
  sorry

end one_correct_statement_l1757_175769


namespace grocery_bagging_l1757_175770

/-- The number of ways to distribute n distinct objects into k indistinguishable containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- There are 5 different items and 3 identical bags. -/
theorem grocery_bagging : distribute 5 3 = 41 := by
  sorry

end grocery_bagging_l1757_175770


namespace physics_marks_proof_l1757_175760

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 92
def average_marks : ℚ := 90.4
def total_subjects : ℕ := 5

theorem physics_marks_proof :
  ∃ (physics_marks : ℕ),
    (english_marks + math_marks + chemistry_marks + biology_marks + physics_marks : ℚ) / total_subjects = average_marks ∧
    physics_marks = 82 := by
  sorry

end physics_marks_proof_l1757_175760


namespace scientific_notation_of_small_number_l1757_175793

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.000000032 = ScientificNotation.mk 3.2 (-8) sorry := by
  sorry

end scientific_notation_of_small_number_l1757_175793


namespace sine_cosine_power_sum_l1757_175786

theorem sine_cosine_power_sum (x : ℝ) (h : Real.sin x + Real.cos x = -1) :
  ∀ n : ℕ, (Real.sin x)^n + (Real.cos x)^n = (-1)^n := by
sorry

end sine_cosine_power_sum_l1757_175786


namespace fourth_power_sum_l1757_175758

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 7) :
  a^4 + b^4 + c^4 = 3 := by
  sorry

end fourth_power_sum_l1757_175758


namespace golf_cost_l1757_175792

/-- If 5 rounds of golf cost $400, then one round of golf costs $80 -/
theorem golf_cost (total_cost : ℝ) (num_rounds : ℕ) (cost_per_round : ℝ) 
  (h1 : total_cost = 400)
  (h2 : num_rounds = 5)
  (h3 : total_cost = num_rounds * cost_per_round) : 
  cost_per_round = 80 := by
  sorry

end golf_cost_l1757_175792


namespace cos_max_value_l1757_175768

open Real

theorem cos_max_value (x : ℝ) :
  let f := fun x => 3 - 2 * cos (x + π / 4)
  (∀ x, f x ≤ 5) ∧
  (∃ k : ℤ, f (2 * k * π + 3 * π / 4) = 5) :=
sorry

end cos_max_value_l1757_175768


namespace bicycle_inventory_decrease_is_58_l1757_175745

/-- Calculates the decrease in bicycle inventory from January 1 to October 1 -/
def bicycle_inventory_decrease : ℕ :=
  let initial_inventory : ℕ := 200
  let feb_to_june_decrease : ℕ := 4 + 6 + 8 + 10 + 12
  let july_decrease : ℕ := 14
  let august_decrease : ℕ := 16 + 20  -- Including sales event
  let september_decrease : ℕ := 18
  let new_shipment : ℕ := 50
  (feb_to_june_decrease + july_decrease + august_decrease + september_decrease) - new_shipment

/-- Theorem stating that the bicycle inventory decrease from January 1 to October 1 is 58 -/
theorem bicycle_inventory_decrease_is_58 : bicycle_inventory_decrease = 58 := by
  sorry

#eval bicycle_inventory_decrease

end bicycle_inventory_decrease_is_58_l1757_175745


namespace part_one_part_two_l1757_175757

/-- The quadratic equation -/
def quadratic (k x : ℝ) : ℝ := k * x^2 + 4 * x + 1

/-- Part 1: Prove that if x = -1 is a solution, then k = 3 -/
theorem part_one (k : ℝ) :
  quadratic k (-1) = 0 → k = 3 := by sorry

/-- Part 2: Prove that if the equation has two real roots and k ≠ 0, then k ≤ 4 and k ≠ 0 -/
theorem part_two (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) →
  k ≠ 0 →
  k ≤ 4 ∧ k ≠ 0 := by sorry

end part_one_part_two_l1757_175757


namespace dave_spent_29_dollars_l1757_175761

/-- Represents the cost of rides for a day at the fair -/
structure DayAtFair where
  rides : List ℕ

/-- Calculates the total cost of rides for a day -/
def totalCost (day : DayAtFair) : ℕ :=
  day.rides.sum

/-- Represents Dave's two days at the fair -/
def davesFairDays : List DayAtFair := [
  { rides := [4, 5, 3, 2] },  -- First day
  { rides := [5, 6, 4] }     -- Second day
]

theorem dave_spent_29_dollars : 
  (davesFairDays.map totalCost).sum = 29 := by
  sorry

end dave_spent_29_dollars_l1757_175761


namespace simplify_sqrt_expression_l1757_175710

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l1757_175710


namespace necessary_condition_for_inequality_l1757_175781

theorem necessary_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 2 3 → x^2 - a ≤ 0) → 
  (a ≥ 8 ∧ ∃ b : ℝ, b ≥ 8 ∧ ∃ y : ℝ, y ∈ Set.Icc 2 3 ∧ y^2 - b > 0) :=
by sorry

end necessary_condition_for_inequality_l1757_175781


namespace a_minus_b_plus_c_value_l1757_175794

theorem a_minus_b_plus_c_value (a b c : ℝ) :
  (abs a = 1) → (abs b = 2) → (abs c = 3) → (a > b) → (b > c) →
  ((a - b + c = 0) ∨ (a - b + c = -2)) := by
sorry

end a_minus_b_plus_c_value_l1757_175794


namespace value_of_b_l1757_175712

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 2 = 4) : 
  b = 15 := by
sorry

end value_of_b_l1757_175712


namespace round_trip_time_l1757_175726

/-- The total time for a round trip between two points, given the distance and speeds in each direction -/
theorem round_trip_time (distance : ℝ) (speed_to : ℝ) (speed_from : ℝ) :
  distance = 19.999999999999996 →
  speed_to = 25 →
  speed_from = 4 →
  (distance / speed_to) + (distance / speed_from) = 5.8 := by
  sorry

#check round_trip_time

end round_trip_time_l1757_175726


namespace intersection_condition_implies_a_geq_5_l1757_175753

open Set Real

theorem intersection_condition_implies_a_geq_5 (a : ℝ) :
  let A := {x : ℝ | x ≤ a}
  let B := {x : ℝ | x^2 - 5*x < 0}
  A ∩ B = B → a ≥ 5 := by
  sorry

end intersection_condition_implies_a_geq_5_l1757_175753


namespace canvas_bag_break_even_trips_eq_300_l1757_175773

/-- The number of shopping trips required for a canvas bag to become the lower-carbon solution compared to plastic bags. -/
def canvas_bag_break_even_trips (canvas_bag_co2_pounds : ℕ) (plastic_bag_co2_ounces : ℕ) (bags_per_trip : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  (canvas_bag_co2_pounds * ounces_per_pound) / (plastic_bag_co2_ounces * bags_per_trip)

/-- Theorem stating that 300 shopping trips are required for the canvas bag to become the lower-carbon solution. -/
theorem canvas_bag_break_even_trips_eq_300 :
  canvas_bag_break_even_trips 600 4 8 16 = 300 := by
  sorry

end canvas_bag_break_even_trips_eq_300_l1757_175773


namespace pentagonal_to_triangular_prism_l1757_175713

/-- The number of cans in a pentagonal pyramid with l layers -/
def T (l : ℕ) : ℕ := l * (3 * l^2 - l) / 2

/-- A triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pentagonal_to_triangular_prism (l : ℕ) (h : l ≥ 2) :
  ∃ n : ℕ, T l = l * triangular_number n :=
by
  sorry

end pentagonal_to_triangular_prism_l1757_175713


namespace cube_of_product_l1757_175754

theorem cube_of_product (x y : ℝ) : (-2 * x^2 * y)^3 = -8 * x^6 * y^3 := by
  sorry

end cube_of_product_l1757_175754


namespace complex_determinant_solution_l1757_175791

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_determinant_solution :
  ∀ z : ℂ, det 1 (-1) z (z * i) = 2 → z = 1 - i := by
  sorry

end complex_determinant_solution_l1757_175791


namespace tomatoes_left_l1757_175738

theorem tomatoes_left (initial : ℕ) (picked_day1 : ℕ) (picked_day2 : ℕ) 
  (h1 : initial = 171) 
  (h2 : picked_day1 = 134) 
  (h3 : picked_day2 = 30) : 
  initial - picked_day1 - picked_day2 = 7 := by
  sorry

end tomatoes_left_l1757_175738


namespace min_value_reciprocal_sum_l1757_175706

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 :=
by sorry

end min_value_reciprocal_sum_l1757_175706


namespace function_value_at_negative_five_l1757_175722

/-- Given a function f(x) = ax + b * sin(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem function_value_at_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b * Real.sin x + 1)
  (h2 : f 5 = 7) : 
  f (-5) = -5 := by
sorry

end function_value_at_negative_five_l1757_175722


namespace weight_difference_l1757_175782

/-- Antonio's weight in kilograms -/
def antonio_weight : ℕ := 50

/-- Total weight of Antonio and his sister in kilograms -/
def total_weight : ℕ := 88

/-- Antonio's sister's weight in kilograms -/
def sister_weight : ℕ := total_weight - antonio_weight

theorem weight_difference :
  antonio_weight > sister_weight ∧
  antonio_weight - sister_weight = 12 := by
  sorry

#check weight_difference

end weight_difference_l1757_175782


namespace expression_equivalence_l1757_175730

/-- Prove that the given expression is equivalent to 4xy(x^2 + y^2)/(x^4 + y^4) -/
theorem expression_equivalence (x y : ℝ) :
  let P := x^2 + y^2
  let Q := x*y
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (4*x*y*(x^2 + y^2)) / (x^4 + y^4) := by
sorry

end expression_equivalence_l1757_175730


namespace constant_term_expansion_l1757_175720

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x - x^(1/2))^6) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) ∧ c = 15) :=
sorry

end constant_term_expansion_l1757_175720


namespace polynomial_value_l1757_175755

theorem polynomial_value (p q : ℝ) : 
  (2*p - q + 3)^2 + 6*(2*p - q + 3) + 6 = (p + 4*q)^2 + 6*(p + 4*q) + 6 →
  p - 5*q + 3 ≠ 0 →
  (5*(p + q + 1))^2 + 6*(5*(p + q + 1)) + 6 = 46 := by
  sorry

end polynomial_value_l1757_175755


namespace distribution_within_one_std_dev_l1757_175759

-- Define a symmetric distribution type
structure SymmetricDistribution where
  -- The cumulative distribution function (CDF)
  cdf : ℝ → ℝ
  -- The mean of the distribution
  mean : ℝ
  -- The standard deviation of the distribution
  std_dev : ℝ
  -- Symmetry property
  symmetry : ∀ x, cdf (mean - x) + cdf (mean + x) = 1
  -- Property that 84% of the distribution is less than mean + std_dev
  eighty_four_percent : cdf (mean + std_dev) = 0.84

-- Theorem statement
theorem distribution_within_one_std_dev 
  (d : SymmetricDistribution) : 
  d.cdf (d.mean + d.std_dev) - d.cdf (d.mean - d.std_dev) = 0.68 := by
  sorry

end distribution_within_one_std_dev_l1757_175759


namespace sqrt_product_equality_l1757_175776

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 50 * Real.sqrt 6 = 90 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l1757_175776


namespace average_age_after_leaving_l1757_175708

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 22
def remaining_people : ℕ := initial_people - 1

theorem average_age_after_leaving :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 258 / 7 := by
  sorry

end average_age_after_leaving_l1757_175708


namespace pipe_filling_speed_l1757_175717

/-- Proves that if Pipe A fills a tank in 24 minutes, and both Pipe A and Pipe B together fill the tank in 3 minutes, then Pipe B fills the tank 7 times faster than Pipe A. -/
theorem pipe_filling_speed (fill_time_A : ℝ) (fill_time_both : ℝ) (speed_ratio : ℝ) : 
  fill_time_A = 24 → 
  fill_time_both = 3 → 
  (1 / fill_time_A + speed_ratio / fill_time_A) * fill_time_both = 1 →
  speed_ratio = 7 := by
sorry

end pipe_filling_speed_l1757_175717


namespace monotonically_decreasing_interval_l1757_175772

-- Define the function f(x) = x³ - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem monotonically_decreasing_interval :
  ∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
by sorry

end monotonically_decreasing_interval_l1757_175772


namespace julie_age_l1757_175741

theorem julie_age (julie aaron : ℕ) 
  (h1 : julie = 4 * aaron) 
  (h2 : julie + 10 = 2 * (aaron + 10)) : 
  julie = 20 := by
sorry

end julie_age_l1757_175741


namespace traffic_light_change_probability_l1757_175703

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds before each color change, and there are 3 changes

/-- Theorem: The probability of observing a color change in a 5-second interval
    for the given traffic light cycle is 3/20 -/
theorem traffic_light_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 50) 
  (h2 : cycle.yellow = 5) 
  (h3 : cycle.red = 45) :
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle) = 3 / 20 := by
  sorry


end traffic_light_change_probability_l1757_175703


namespace chandler_total_rolls_l1757_175790

/-- The total number of rolls Chandler needs to sell for the school fundraiser -/
def total_rolls_to_sell : ℕ :=
  let grandmother_rolls := 3
  let uncle_rolls := 4
  let neighbor_rolls := 3
  let additional_rolls := 2
  grandmother_rolls + uncle_rolls + neighbor_rolls + additional_rolls

/-- Theorem stating that Chandler needs to sell 12 rolls in total -/
theorem chandler_total_rolls : total_rolls_to_sell = 12 := by
  sorry

end chandler_total_rolls_l1757_175790


namespace opposite_of_negative_2023_l1757_175778

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l1757_175778


namespace rotten_apples_percentage_l1757_175749

theorem rotten_apples_percentage (total_apples : ℕ) (smelling_ratio : ℚ) (non_smelling_rotten : ℕ) :
  total_apples = 200 →
  smelling_ratio = 7/10 →
  non_smelling_rotten = 24 →
  (non_smelling_rotten : ℚ) / ((1 - smelling_ratio) * total_apples) = 2/5 :=
by
  sorry

end rotten_apples_percentage_l1757_175749


namespace division_problem_l1757_175766

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 172)
  (h2 : quotient = 10)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * divisor + remainder) :
  divisor = 17 := by
sorry

end division_problem_l1757_175766


namespace complex_exponential_sum_l1757_175734

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/5 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/5 : ℂ) - (4/9 : ℂ) * Complex.I :=
by
  sorry

end complex_exponential_sum_l1757_175734


namespace sum_of_squares_l1757_175798

theorem sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (product_eq : x * y * z = 72)
  (sum_reciprocals_eq : 1/x + 1/y + 1/z = 3/4) :
  x^2 + y^2 + z^2 = 61 := by
sorry

end sum_of_squares_l1757_175798


namespace reciprocal_of_negative_2023_l1757_175728

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1/2023 :=
by sorry

end reciprocal_of_negative_2023_l1757_175728


namespace rectangular_prism_volume_l1757_175777

theorem rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 18) 
  (h3 : c * a = 20) 
  (h4 : max a (max b c) = 2 * min a (min b c)) : 
  a * b * c = 30 * Real.sqrt 10 := by
sorry

end rectangular_prism_volume_l1757_175777


namespace hyperbola_b_value_l1757_175711

/-- The value of b for a hyperbola with given equation and asymptote -/
theorem hyperbola_b_value (b : ℝ) (h1 : b > 0) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, 3*x - 2*y = 0 ∧ x^2 / 4 - y^2 / b^2 = 1) →
  b = 3 := by
sorry

end hyperbola_b_value_l1757_175711


namespace division_problem_l1757_175723

theorem division_problem : ∃ (n : ℕ), n ≠ 0 ∧ 45 = 11 * n + 1 ∧ n = 4 := by
  sorry

end division_problem_l1757_175723


namespace geometric_sequence_sum_five_l1757_175783

/-- A geometric sequence with common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℚ
  q : ℚ
  q_ne_one : q ≠ 1
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (g : GeometricSequence) (n : ℕ) : ℚ :=
  (g.a 1) * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_sum_five (g : GeometricSequence) 
  (h1 : g.a 1 * g.a 2 * g.a 3 * g.a 4 * g.a 5 = 1 / 1024)
  (h2 : 2 * g.a 4 = g.a 2 + g.a 3) : 
  sum_n g 5 = 11 / 16 := by
  sorry

end geometric_sequence_sum_five_l1757_175783


namespace females_advanced_count_l1757_175700

/-- A company's employee distribution by gender and education level -/
structure Company where
  total_employees : ℕ
  females : ℕ
  males : ℕ
  advanced_degrees : ℕ
  college_degrees : ℕ
  vocational_training : ℕ
  males_college : ℕ
  females_vocational : ℕ

/-- The number of females with advanced degrees in the company -/
def females_advanced (c : Company) : ℕ :=
  c.advanced_degrees - (c.males - c.males_college - (c.vocational_training - c.females_vocational))

/-- Theorem stating the number of females with advanced degrees -/
theorem females_advanced_count (c : Company) 
  (h1 : c.total_employees = 360)
  (h2 : c.females = 220)
  (h3 : c.males = 140)
  (h4 : c.advanced_degrees = 140)
  (h5 : c.college_degrees = 160)
  (h6 : c.vocational_training = 60)
  (h7 : c.males_college = 55)
  (h8 : c.females_vocational = 25)
  (h9 : c.total_employees = c.females + c.males)
  (h10 : c.total_employees = c.advanced_degrees + c.college_degrees + c.vocational_training) :
  females_advanced c = 90 := by
  sorry

#eval females_advanced {
  total_employees := 360,
  females := 220,
  males := 140,
  advanced_degrees := 140,
  college_degrees := 160,
  vocational_training := 60,
  males_college := 55,
  females_vocational := 25
}

end females_advanced_count_l1757_175700


namespace sin_neg_135_degrees_l1757_175718

theorem sin_neg_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end sin_neg_135_degrees_l1757_175718


namespace no_heptagon_intersection_l1757_175701

/-- A cube in 3D space -/
structure Cube where
  -- Add necessary fields for a cube

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- Represents the intersection of a plane and a cube -/
def Intersection (c : Cube) (p : Plane) : Set Point := sorry

/-- The number of edges of the cube that the plane intersects -/
def numIntersectedEdges (c : Cube) (p : Plane) : ℕ := sorry

/-- Predicate to check if a plane passes through a vertex more than once -/
def passesVertexTwice (c : Cube) (p : Plane) : Prop := sorry

/-- Theorem: A plane intersecting a cube cannot form a heptagon -/
theorem no_heptagon_intersection (c : Cube) (p : Plane) : 
  ¬(numIntersectedEdges c p = 7 ∧ ¬passesVertexTwice c p) := by
  sorry

end no_heptagon_intersection_l1757_175701


namespace opinion_change_difference_is_twenty_percent_l1757_175744

/-- Represents the percentage of students who like science -/
structure ScienceOpinion where
  initial_like : ℚ
  final_like : ℚ

/-- Calculate the difference between maximum and minimum percentage of students who changed their opinion -/
def opinion_change_difference (opinion : ScienceOpinion) : ℚ :=
  let initial_dislike := 1 - opinion.initial_like
  let final_dislike := 1 - opinion.final_like
  let min_change := |opinion.final_like - opinion.initial_like|
  let max_change := min opinion.initial_like final_dislike + min initial_dislike opinion.final_like
  max_change - min_change

/-- Theorem statement for the specific problem -/
theorem opinion_change_difference_is_twenty_percent :
  ∃ (opinion : ScienceOpinion),
    opinion.initial_like = 2/5 ∧
    opinion.final_like = 4/5 ∧
    opinion_change_difference opinion = 1/5 := by
  sorry

end opinion_change_difference_is_twenty_percent_l1757_175744


namespace power_quotient_23_l1757_175763

theorem power_quotient_23 : (23 ^ 11) / (23 ^ 8) = 12167 := by sorry

end power_quotient_23_l1757_175763


namespace factor_polynomial_l1757_175709

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end factor_polynomial_l1757_175709
