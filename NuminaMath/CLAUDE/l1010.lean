import Mathlib

namespace NUMINAMATH_CALUDE_probability_of_shaded_triangle_l1010_101038

/-- Given a diagram with triangles, this theorem proves the probability of selecting a shaded triangle -/
theorem probability_of_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : 
  total_triangles = 5 → shaded_triangles = 3 → (shaded_triangles : ℚ) / total_triangles = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_shaded_triangle_l1010_101038


namespace NUMINAMATH_CALUDE_max_visible_cubes_l1010_101092

/-- The size of the cube's edge -/
def n : ℕ := 12

/-- The number of unit cubes on one face of the large cube -/
def face_cubes : ℕ := n^2

/-- The number of unit cubes along one edge of the large cube -/
def edge_cubes : ℕ := n

/-- The number of visible faces from a corner -/
def visible_faces : ℕ := 3

/-- The number of visible edges from a corner -/
def visible_edges : ℕ := 3

/-- The number of visible corners from a corner -/
def visible_corners : ℕ := 1

theorem max_visible_cubes :
  visible_faces * face_cubes - (visible_edges * edge_cubes - visible_corners) = 398 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_l1010_101092


namespace NUMINAMATH_CALUDE_sin_alpha_minus_pi_4_increases_with_k_l1010_101040

theorem sin_alpha_minus_pi_4_increases_with_k (α : Real) (k : Real)
  (h1 : 0 < α)
  (h2 : α < π / 4)
  (h3 : (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = k) :
  ∀ ε > 0, ∃ δ > 0, ∀ k' > k,
    k' - k < δ → Real.sin (α - π / 4) < Real.sin (α - π / 4) + ε :=
by sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_pi_4_increases_with_k_l1010_101040


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1010_101064

/-- Given two perpendicular lines with direction vectors (4, -5) and (b, 8), prove that b = 10 -/
theorem perpendicular_lines_b_value :
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ (i j : Fin 2), i ≠ j → v1 i * v2 i + v1 j * v2 j = 0) →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1010_101064


namespace NUMINAMATH_CALUDE_main_rectangle_tiled_by_tetraminoes_l1010_101057

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tetramino (2 × 3 rectangle with two opposite corners removed) -/
def Tetramino : Rectangle :=
  { width := 2, height := 3 }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ :=
  r.width * r.height

/-- The area of a tetramino -/
def tetraminoArea : ℕ :=
  area Tetramino - 2

/-- The main rectangle to be tiled -/
def mainRectangle : Rectangle :=
  { width := 2008, height := 2010 }

/-- Theorem: The main rectangle can be tiled using only tetraminoes -/
theorem main_rectangle_tiled_by_tetraminoes :
  ∃ (n : ℕ), n * tetraminoArea = area mainRectangle :=
sorry

end NUMINAMATH_CALUDE_main_rectangle_tiled_by_tetraminoes_l1010_101057


namespace NUMINAMATH_CALUDE_no_solution_to_system_l1010_101083

theorem no_solution_to_system : ¬∃ x : ℝ, (Real.arccos (Real.cos x) = x / 3) ∧ (Real.sin x = Real.cos (x / 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l1010_101083


namespace NUMINAMATH_CALUDE_line_with_y_intercept_two_l1010_101009

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The equation of a line in slope-intercept form. -/
def line_equation (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Theorem: The equation of a line with y-intercept 2 is y = kx + 2 -/
theorem line_with_y_intercept_two (k : ℝ) :
  ∃ (l : Line), l.y_intercept = 2 ∧ ∀ (x y : ℝ), y = line_equation l x ↔ y = k * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_line_with_y_intercept_two_l1010_101009


namespace NUMINAMATH_CALUDE_sneakers_cost_l1010_101080

theorem sneakers_cost (sneakers_cost socks_cost : ℝ) 
  (total_cost : sneakers_cost + socks_cost = 101)
  (cost_difference : sneakers_cost = socks_cost + 100) : 
  sneakers_cost = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_cost_l1010_101080


namespace NUMINAMATH_CALUDE_monkey_climbing_l1010_101025

/-- Monkey's tree climbing problem -/
theorem monkey_climbing (tree_height : ℝ) (hop_distance : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 21)
  (h2 : hop_distance = 3)
  (h3 : total_time = 19) :
  ∃ (slip_distance : ℝ), 
    slip_distance = 2 ∧ 
    (hop_distance - slip_distance) * (total_time - 1 : ℝ) + hop_distance = tree_height :=
by sorry

end NUMINAMATH_CALUDE_monkey_climbing_l1010_101025


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l1010_101010

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem log_sum_equals_zero
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_prod : a 3 * a 5 * a 7 = 1) :
  Real.log (a 1) + Real.log (a 9) = 0 :=
sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l1010_101010


namespace NUMINAMATH_CALUDE_hydrogen_oxygen_reaction_certain_l1010_101070

/-- Represents the certainty of an event --/
inductive EventCertainty
  | Possible
  | Impossible
  | Certain

/-- Represents a chemical reaction --/
structure ChemicalReaction where
  reactants : List String
  products : List String

/-- The chemical reaction of hydrogen burning in oxygen to form water --/
def hydrogenOxygenReaction : ChemicalReaction :=
  { reactants := ["Hydrogen", "Oxygen"],
    products := ["Water"] }

/-- Theorem stating that the hydrogen-oxygen reaction is certain --/
theorem hydrogen_oxygen_reaction_certain :
  (hydrogenOxygenReaction.reactants = ["Hydrogen", "Oxygen"] ∧
   hydrogenOxygenReaction.products = ["Water"]) →
  EventCertainty.Certain = 
    match hydrogenOxygenReaction with
    | { reactants := ["Hydrogen", "Oxygen"], products := ["Water"] } => EventCertainty.Certain
    | _ => EventCertainty.Possible
  := by sorry

end NUMINAMATH_CALUDE_hydrogen_oxygen_reaction_certain_l1010_101070


namespace NUMINAMATH_CALUDE_four_leaf_area_l1010_101042

/-- The area of a four-leaf shape formed by semicircles drawn on each side of a square -/
theorem four_leaf_area (a : ℝ) (h : a > 0) :
  let square_side := a
  let semicircle_radius := a / 2
  let leaf_area := π * semicircle_radius^2 / 2 - square_side^2 / 4
  4 * leaf_area = a^2 / 2 * (π - 2) :=
by sorry

end NUMINAMATH_CALUDE_four_leaf_area_l1010_101042


namespace NUMINAMATH_CALUDE_value_added_to_forty_percent_l1010_101044

theorem value_added_to_forty_percent (N : ℝ) (V : ℝ) : 
  N = 100 → 0.4 * N + V = N → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_forty_percent_l1010_101044


namespace NUMINAMATH_CALUDE_debate_team_max_groups_l1010_101021

/-- Given a debate team with boys and girls, calculate the maximum number of groups
    that can be formed with a minimum number of boys and girls per group. -/
def max_groups (num_boys num_girls min_boys_per_group min_girls_per_group : ℕ) : ℕ :=
  min (num_boys / min_boys_per_group) (num_girls / min_girls_per_group)

/-- Theorem stating that for a debate team with 31 boys and 32 girls,
    where each group must have at least 2 boys and 3 girls,
    the maximum number of groups that can be formed is 10. -/
theorem debate_team_max_groups :
  max_groups 31 32 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_max_groups_l1010_101021


namespace NUMINAMATH_CALUDE_unique_m_value_l1010_101073

def U (m : ℝ) : Set ℝ := {4, m^2 + 2*m - 3, 19}
def A : Set ℝ := {5}

theorem unique_m_value :
  ∃! m : ℝ, (U m \ A = {|4*m - 3|, 4}) ∧ (m^2 + 2*m - 3 = 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1010_101073


namespace NUMINAMATH_CALUDE_exponential_inequality_l1010_101059

theorem exponential_inequality (m n : ℝ) : (0.2 : ℝ) ^ m < (0.2 : ℝ) ^ n → m > n := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1010_101059


namespace NUMINAMATH_CALUDE_right_triangle_area_l1010_101016

theorem right_triangle_area (longer_leg : ℝ) (angle : ℝ) :
  longer_leg = 10 →
  angle = 30 * (π / 180) →
  ∃ (area : ℝ), area = (50 * Real.sqrt 3) / 3 ∧
  area = (1 / 2) * longer_leg * (longer_leg / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1010_101016


namespace NUMINAMATH_CALUDE_house_glass_panels_l1010_101005

/-- The number of glass panels per window -/
def panels_per_window : ℕ := 4

/-- The number of double windows downstairs -/
def double_windows_downstairs : ℕ := 6

/-- The number of single windows upstairs -/
def single_windows_upstairs : ℕ := 8

/-- The total number of glass panels in the house -/
def total_panels : ℕ := panels_per_window * (2 * double_windows_downstairs + single_windows_upstairs)

theorem house_glass_panels :
  total_panels = 80 :=
by sorry

end NUMINAMATH_CALUDE_house_glass_panels_l1010_101005


namespace NUMINAMATH_CALUDE_f_properties_l1010_101065

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x

theorem f_properties :
  (∃ (x_max : ℝ), x_max > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f x_max ∧ f x_max = 0) ∧
  (∀ (a : ℝ), a ≥ 2 → ∀ (x : ℝ), x > 0 → f x < (a/2 - 1) * x^2 + a * x - 1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 → 
    f x₁ + f x₂ + 2 * (x₁^2 + x₂^2) + x₁ * x₂ = 0 → 
    x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1010_101065


namespace NUMINAMATH_CALUDE_cubic_common_roots_l1010_101028

theorem cubic_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 12 = 0 ∧ 
    r^3 + b*r^2 + 23*r + 15 = 0 ∧
    s^3 + a*s^2 + 17*s + 12 = 0 ∧ 
    s^3 + b*s^2 + 23*s + 15 = 0) →
  a = -10 ∧ b = -11 := by
sorry

end NUMINAMATH_CALUDE_cubic_common_roots_l1010_101028


namespace NUMINAMATH_CALUDE_solve_salary_problem_l1010_101001

def salary_problem (a b : ℝ) : Prop :=
  a + b = 4000 ∧
  0.05 * a = 0.15 * b ∧
  a = 3000

theorem solve_salary_problem :
  ∃ (a b : ℝ), salary_problem a b :=
sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l1010_101001


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1010_101006

theorem divisibility_equivalence (a b c : ℕ) (h : c ≥ 1) :
  a ∣ b ↔ a^c ∣ b^c := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1010_101006


namespace NUMINAMATH_CALUDE_library_visitors_average_l1010_101050

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 4
  let totalOtherDays : ℕ := 26
  let totalDays : ℕ := 30
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  (totalVisitors : ℚ) / totalDays

theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitors sundayVisitors otherDayVisitors = 276 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l1010_101050


namespace NUMINAMATH_CALUDE_mo_drink_difference_l1010_101089

theorem mo_drink_difference (n : ℕ) : 
  (n ≥ 0) →  -- n is non-negative
  (2 * n + 5 * 4 = 26) →  -- total cups constraint
  (5 * 4 - 2 * n = 14) :=  -- difference between tea and hot chocolate
by
  sorry

end NUMINAMATH_CALUDE_mo_drink_difference_l1010_101089


namespace NUMINAMATH_CALUDE_marathon_remainder_l1010_101097

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
    (h : Distance.mk m y = 
      { miles := num_marathons * marathon.miles + (num_marathons * marathon.yards) / yards_per_mile,
        yards := (num_marathons * marathon.yards) % yards_per_mile }) :
  y = 495 := by sorry

end NUMINAMATH_CALUDE_marathon_remainder_l1010_101097


namespace NUMINAMATH_CALUDE_simplify_expression_l1010_101049

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1010_101049


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1010_101047

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1010_101047


namespace NUMINAMATH_CALUDE_fishing_problem_l1010_101033

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jason + ryan + jeffery = 100 →
  jeffery = 60 →
  ryan = 30 := by sorry

end NUMINAMATH_CALUDE_fishing_problem_l1010_101033


namespace NUMINAMATH_CALUDE_total_distance_proof_l1010_101088

/-- The total distance across the country in kilometers -/
def total_distance : ℕ := 8205

/-- The distance Amelia drove on Monday in kilometers -/
def monday_distance : ℕ := 907

/-- The distance Amelia drove on Tuesday in kilometers -/
def tuesday_distance : ℕ := 582

/-- The remaining distance Amelia has to drive in kilometers -/
def remaining_distance : ℕ := 6716

/-- Theorem stating that the total distance is the sum of the distances driven on Monday, Tuesday, and the remaining distance -/
theorem total_distance_proof : 
  total_distance = monday_distance + tuesday_distance + remaining_distance := by
  sorry

end NUMINAMATH_CALUDE_total_distance_proof_l1010_101088


namespace NUMINAMATH_CALUDE_square_side_length_l1010_101041

theorem square_side_length (x : ℝ) (h : x > 0) (h_area : x^2 = 4 * 3) : x = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1010_101041


namespace NUMINAMATH_CALUDE_airbnb_rental_cost_l1010_101066

/-- Calculates the Airbnb rental cost for a vacation -/
theorem airbnb_rental_cost 
  (num_people : ℕ) 
  (car_rental_cost : ℕ) 
  (share_per_person : ℕ) 
  (h1 : num_people = 8)
  (h2 : car_rental_cost = 800)
  (h3 : share_per_person = 500) :
  num_people * share_per_person - car_rental_cost = 3200 := by
  sorry

end NUMINAMATH_CALUDE_airbnb_rental_cost_l1010_101066


namespace NUMINAMATH_CALUDE_cosine_sum_zero_implies_angle_difference_l1010_101081

theorem cosine_sum_zero_implies_angle_difference (α β γ : ℝ) 
  (h1 : 0 < α ∧ α < β ∧ β < γ ∧ γ < 2 * π)
  (h2 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
  γ - α = 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_implies_angle_difference_l1010_101081


namespace NUMINAMATH_CALUDE_phone_charges_count_l1010_101062

def daily_mileages : List Nat := [135, 259, 159, 189]
def charge_interval : Nat := 106

theorem phone_charges_count : 
  (daily_mileages.sum / charge_interval : Nat) = 7 := by
  sorry

end NUMINAMATH_CALUDE_phone_charges_count_l1010_101062


namespace NUMINAMATH_CALUDE_game_ends_after_63_rounds_l1010_101034

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- Represents the state of the game -/
structure GameState :=
  (tokens : Player → ℕ)
  (round : ℕ)

/-- Initial state of the game -/
def initialState : GameState :=
  { tokens := λ p => match p with
    | Player.A => 20
    | Player.B => 18
    | Player.C => 16
    | Player.D => 14
  , round := 0 }

/-- Updates the game state for one round -/
def updateState (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Theorem stating that the game ends after 63 rounds -/
theorem game_ends_after_63_rounds :
  ∃ (finalState : GameState),
    finalState.round = 63 ∧
    isGameOver finalState ∧
    (∀ (prevState : GameState),
      prevState.round < 63 →
      ¬isGameOver prevState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_63_rounds_l1010_101034


namespace NUMINAMATH_CALUDE_myrtle_hens_eggs_per_day_l1010_101026

/-- The number of eggs laid by Myrtle's hens -/
theorem myrtle_hens_eggs_per_day :
  ∀ (num_hens : ℕ) (days_gone : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ) (eggs_remaining : ℕ),
  num_hens = 3 →
  days_gone = 7 →
  eggs_taken = 12 →
  eggs_dropped = 5 →
  eggs_remaining = 46 →
  ∃ (eggs_per_hen_per_day : ℕ),
    eggs_per_hen_per_day = 3 ∧
    num_hens * eggs_per_hen_per_day * days_gone - eggs_taken - eggs_dropped = eggs_remaining :=
by
  sorry


end NUMINAMATH_CALUDE_myrtle_hens_eggs_per_day_l1010_101026


namespace NUMINAMATH_CALUDE_division_problem_l1010_101063

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1010_101063


namespace NUMINAMATH_CALUDE_range_of_3x_minus_y_l1010_101086

theorem range_of_3x_minus_y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 1) 
  (h2 : 1 ≤ x - y ∧ x - y ≤ 3) : 
  1 ≤ 3*x - y ∧ 3*x - y ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_range_of_3x_minus_y_l1010_101086


namespace NUMINAMATH_CALUDE_smallest_k_is_2011_l1010_101052

def is_valid_sequence (s : ℕ → ℕ) : Prop :=
  (∀ n, s n < s (n + 1)) ∧
  (∀ n, (1005 ∣ s n) ∨ (1006 ∣ s n)) ∧
  (∀ n, ¬(97 ∣ s n)) ∧
  (∀ n, s (n + 1) - s n ≤ 2011)

theorem smallest_k_is_2011 :
  ∀ k : ℕ, (∃ s : ℕ → ℕ, is_valid_sequence s ∧ ∀ n, s (n + 1) - s n ≤ k) → k ≥ 2011 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_2011_l1010_101052


namespace NUMINAMATH_CALUDE_solve_system_l1010_101078

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 6) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1010_101078


namespace NUMINAMATH_CALUDE_grandmothers_current_age_l1010_101027

/-- The age of Minyoung's grandmother this year, given that Minyoung is 7 years old this year
    and her grandmother turns 65 when Minyoung turns 10. -/
def grandmothers_age (minyoung_age : ℕ) (grandmother_future_age : ℕ) (years_until_future : ℕ) : ℕ :=
  grandmother_future_age - years_until_future

/-- Proof that Minyoung's grandmother is 62 years old this year. -/
theorem grandmothers_current_age :
  grandmothers_age 7 65 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_grandmothers_current_age_l1010_101027


namespace NUMINAMATH_CALUDE_min_sum_squared_l1010_101061

theorem min_sum_squared (x₁ x₂ : ℝ) (h : x₁ * x₂ = 2013) : 
  (x₁ + x₂)^2 ≥ 8052 ∧ ∃ y₁ y₂ : ℝ, y₁ * y₂ = 2013 ∧ (y₁ + y₂)^2 = 8052 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_l1010_101061


namespace NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l1010_101077

/-- Given two pipes that can fill a tank in 10 and 20 hours respectively,
    prove that when both are opened simultaneously, the tank fills in 20/3 hours. -/
theorem simultaneous_pipe_filling_time :
  ∀ (tank_capacity : ℝ) (pipe_a_rate pipe_b_rate : ℝ),
    pipe_a_rate = tank_capacity / 10 →
    pipe_b_rate = tank_capacity / 20 →
    tank_capacity / (pipe_a_rate + pipe_b_rate) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l1010_101077


namespace NUMINAMATH_CALUDE_waiter_problem_l1010_101031

/-- Given an initial number of customers and two groups of customers leaving,
    calculate the final number of customers remaining. -/
def remaining_customers (initial : ℝ) (first_group : ℝ) (second_group : ℝ) : ℝ :=
  initial - first_group - second_group

/-- Theorem stating that for the given problem, the number of remaining customers is 3.0 -/
theorem waiter_problem (initial : ℝ) (first_group : ℝ) (second_group : ℝ)
    (h1 : initial = 36.0)
    (h2 : first_group = 19.0)
    (h3 : second_group = 14.0) :
    remaining_customers initial first_group second_group = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l1010_101031


namespace NUMINAMATH_CALUDE_cube_of_negative_four_equals_negative_cube_of_four_l1010_101087

theorem cube_of_negative_four_equals_negative_cube_of_four : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_four_equals_negative_cube_of_four_l1010_101087


namespace NUMINAMATH_CALUDE_correct_calculation_l1010_101072

theorem correct_calculation (x : ℤ) (h : x - 954 = 468) : x + 954 = 2376 :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l1010_101072


namespace NUMINAMATH_CALUDE_average_expenditure_for_week_l1010_101012

/-- The average expenditure for a week given the average expenditures for two parts of the week -/
theorem average_expenditure_for_week 
  (avg_first_3_days : ℝ) 
  (avg_next_4_days : ℝ) 
  (h1 : avg_first_3_days = 350)
  (h2 : avg_next_4_days = 420) :
  (3 * avg_first_3_days + 4 * avg_next_4_days) / 7 = 390 := by
  sorry

#check average_expenditure_for_week

end NUMINAMATH_CALUDE_average_expenditure_for_week_l1010_101012


namespace NUMINAMATH_CALUDE_fraction_equation_l1010_101079

theorem fraction_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 3) : 
  a / b = 1.2 - 0.4 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_l1010_101079


namespace NUMINAMATH_CALUDE_trapezoid_area_l1010_101002

/-- A trapezoid ABCD with the following properties:
  * BC = 5
  * Distance from A to line BC is 3
  * Distance from D to line BC is 7
-/
structure Trapezoid where
  BC : ℝ
  dist_A_to_BC : ℝ
  dist_D_to_BC : ℝ
  h_BC : BC = 5
  h_dist_A : dist_A_to_BC = 3
  h_dist_D : dist_D_to_BC = 7

/-- The area of the trapezoid ABCD -/
def area (t : Trapezoid) : ℝ := 25

/-- Theorem stating that the area of the trapezoid ABCD is 25 -/
theorem trapezoid_area (t : Trapezoid) : area t = 25 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1010_101002


namespace NUMINAMATH_CALUDE_odometer_reading_l1010_101068

theorem odometer_reading (initial_reading traveled_distance : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : traveled_distance = 159.7) :
  initial_reading + traveled_distance = 372.0 := by
sorry

end NUMINAMATH_CALUDE_odometer_reading_l1010_101068


namespace NUMINAMATH_CALUDE_cafeteria_apple_count_l1010_101084

def initial_apples : ℕ := 65
def used_percentage : ℚ := 20 / 100
def bought_apples : ℕ := 15

theorem cafeteria_apple_count : 
  initial_apples - (initial_apples * used_percentage).floor + bought_apples = 67 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apple_count_l1010_101084


namespace NUMINAMATH_CALUDE_total_pencils_and_crayons_l1010_101007

theorem total_pencils_and_crayons (rows : ℕ) (pencils_per_row : ℕ) (crayons_per_row : ℕ)
  (h_rows : rows = 11)
  (h_pencils : pencils_per_row = 31)
  (h_crayons : crayons_per_row = 27) :
  rows * pencils_per_row + rows * crayons_per_row = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_and_crayons_l1010_101007


namespace NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1010_101046

theorem quadratic_two_roots_condition (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + c = x₁ + 2 ∧ x₂^2 - 3*x₂ + c = x₂ + 2) ↔ c < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_condition_l1010_101046


namespace NUMINAMATH_CALUDE_bulb_replacement_probabilities_l1010_101017

/-- Represents the probability of a bulb lasting more than a given number of years -/
def bulb_survival_prob (years : ℕ) : ℝ :=
  match years with
  | 1 => 0.8
  | 2 => 0.3
  | _ => 0 -- Assuming 0 probability for other years

/-- The number of lamps in the conference room -/
def num_lamps : ℕ := 3

/-- Calculates the probability of not replacing any bulbs during the first replacement -/
def prob_no_replace : ℝ := (bulb_survival_prob 1) ^ num_lamps

/-- Calculates the probability of replacing exactly 2 bulbs during the first replacement -/
def prob_replace_two : ℝ := 
  (Nat.choose num_lamps 2 : ℝ) * (bulb_survival_prob 1) * (1 - bulb_survival_prob 1)^2

/-- Calculates the probability that a bulb needs to be replaced during the second replacement -/
def prob_replace_second : ℝ := 
  (1 - bulb_survival_prob 1)^2 + (bulb_survival_prob 1) * (1 - bulb_survival_prob 2)

theorem bulb_replacement_probabilities :
  (prob_no_replace = 0.512) ∧
  (prob_replace_two = 0.096) ∧
  (prob_replace_second = 0.6) :=
sorry

end NUMINAMATH_CALUDE_bulb_replacement_probabilities_l1010_101017


namespace NUMINAMATH_CALUDE_unique_a_value_l1010_101056

/-- A function to represent the exponent |a-2| --/
def abs_a_minus_2 (a : ℝ) : ℝ := |a - 2|

/-- The coefficient of x in the equation --/
def coeff_x (a : ℝ) : ℝ := a - 3

/-- Predicate to check if the equation is linear --/
def is_linear (a : ℝ) : Prop := abs_a_minus_2 a = 1

/-- Theorem stating that a = 1 is the only value satisfying the conditions --/
theorem unique_a_value : ∃! a : ℝ, is_linear a ∧ coeff_x a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1010_101056


namespace NUMINAMATH_CALUDE_sheepdog_speed_l1010_101022

/-- Proves that a sheepdog running at the specified speed can catch a sheep in the given time --/
theorem sheepdog_speed (sheep_speed : ℝ) (initial_distance : ℝ) (catch_time : ℝ) 
  (h1 : sheep_speed = 12)
  (h2 : initial_distance = 160)
  (h3 : catch_time = 20) :
  (initial_distance + sheep_speed * catch_time) / catch_time = 20 := by
  sorry

#check sheepdog_speed

end NUMINAMATH_CALUDE_sheepdog_speed_l1010_101022


namespace NUMINAMATH_CALUDE_negative_two_cubed_equality_l1010_101094

theorem negative_two_cubed_equality : -2^3 = (-2)^3 := by sorry

end NUMINAMATH_CALUDE_negative_two_cubed_equality_l1010_101094


namespace NUMINAMATH_CALUDE_ryan_learning_days_l1010_101029

/-- Given that Ryan spends 4 hours on Chinese daily and a total of 24 hours on Chinese,
    prove that the number of days he learns is 6. -/
theorem ryan_learning_days :
  ∀ (hours_chinese_per_day : ℕ) (total_hours_chinese : ℕ),
    hours_chinese_per_day = 4 →
    total_hours_chinese = 24 →
    total_hours_chinese / hours_chinese_per_day = 6 :=
by sorry

end NUMINAMATH_CALUDE_ryan_learning_days_l1010_101029


namespace NUMINAMATH_CALUDE_female_puppies_count_l1010_101030

theorem female_puppies_count (total : ℕ) (male : ℕ) (ratio : ℚ) : ℕ :=
  let female := total - male
  have h1 : total = 12 := by sorry
  have h2 : male = 10 := by sorry
  have h3 : ratio = 1/5 := by sorry
  have h4 : (female : ℚ) / male = ratio := by sorry
  2

#check female_puppies_count

end NUMINAMATH_CALUDE_female_puppies_count_l1010_101030


namespace NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l1010_101004

/-- The number of squares on a chessboard. -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the chessboard. -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook. -/
def probNotAttackedByOneRook : ℚ := 49 / 64

/-- The expected number of squares under attack when three rooks are randomly placed on a chessboard. -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttackedByOneRook ^ numberOfRooks)

/-- Theorem stating that the expected number of squares under attack is equal to the calculated value. -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l1010_101004


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1010_101008

theorem unique_integer_solution : ∃! x : ℤ, 3 * (x + 200000) = 10 * x + 2 :=
  by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1010_101008


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l1010_101053

theorem inequality_solution_sets : ∃ (x : ℝ), 
  (x > 15 ∧ ¬(x - 7 < 2*x + 8)) ∨ (x - 7 < 2*x + 8 ∧ ¬(x > 15)) ∧
  (∀ y : ℝ, (5*y > 10 ↔ 3*y > 6)) ∧
  (∀ z : ℝ, (6*z - 9 < 3*z + 6 ↔ z < 5)) ∧
  (∀ w : ℝ, (w < -2 ↔ -14*w > 28)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l1010_101053


namespace NUMINAMATH_CALUDE_max_students_above_average_l1010_101048

theorem max_students_above_average (n : ℕ) (scores : Fin n → ℝ) : 
  n = 80 → (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card) → 
  (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧ k ≤ 79) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_average_l1010_101048


namespace NUMINAMATH_CALUDE_min_value_inequality_l1010_101096

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = Real.sqrt 3) :
  1/a^2 + 2/b^2 ≥ 2 ∧ ∀ x, f x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1010_101096


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1010_101067

def inequality_system (x : ℝ) : Prop :=
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)

def integer_solutions : Set ℤ :=
  {-1, 0, 1}

theorem inequality_system_integer_solutions :
  ∀ (n : ℤ), (n ∈ integer_solutions) ↔ (inequality_system (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1010_101067


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1010_101069

/-- Given a wire cut into two pieces of lengths a and b, where piece a forms a rectangle
    with length twice its width and piece b forms a circle, if the areas of the rectangle
    and circle are equal, then a/b = 3/√(2π). -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∃ x : ℝ, a = 6 * x ∧ 2 * x^2 = π * (b / (2 * π))^2) →
  a / b = 3 / Real.sqrt (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1010_101069


namespace NUMINAMATH_CALUDE_hamburgers_leftover_count_l1010_101051

/-- The number of hamburgers made by the restaurant -/
def hamburgers_made : ℕ := 9

/-- The number of hamburgers served during lunch -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := hamburgers_made - hamburgers_served

theorem hamburgers_leftover_count : hamburgers_leftover = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_leftover_count_l1010_101051


namespace NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l1010_101035

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twelve_factorial_mod_thirteen : 
  factorial 12 % 13 = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_factorial_mod_thirteen_l1010_101035


namespace NUMINAMATH_CALUDE_negative_three_plus_four_equals_one_l1010_101003

theorem negative_three_plus_four_equals_one : -3 + 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_plus_four_equals_one_l1010_101003


namespace NUMINAMATH_CALUDE_expression_equality_l1010_101090

theorem expression_equality (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x - y) * (2 * x + y) = 5 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1010_101090


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l1010_101055

/-- Represents a score interval with its frequency -/
structure ScoreInterval :=
  (lowerBound upperBound : ℕ)
  (frequency : ℕ)

/-- The problem setup -/
def testScores : List ScoreInterval :=
  [ ⟨45, 49, 8⟩
  , ⟨50, 54, 15⟩
  , ⟨55, 59, 20⟩
  , ⟨60, 64, 18⟩
  , ⟨65, 69, 17⟩
  , ⟨70, 74, 12⟩
  , ⟨75, 79, 9⟩
  , ⟨80, 84, 6⟩
  ]

def totalStudents : ℕ := 105

/-- The median interval is the one containing the (n+1)/2 th student -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median -/
def findMedianInterval (scores : List ScoreInterval) (medianPos : ℕ) : Option ScoreInterval :=
  let rec go (acc : ℕ) (remaining : List ScoreInterval) : Option ScoreInterval :=
    match remaining with
    | [] => none
    | interval :: rest =>
      let newAcc := acc + interval.frequency
      if newAcc ≥ medianPos then some interval
      else go newAcc rest
  go 0 scores

/-- Theorem stating that the median score is in the interval 60-64 -/
theorem median_in_60_64_interval :
  findMedianInterval testScores medianPosition = some ⟨60, 64, 18⟩ := by
  sorry


end NUMINAMATH_CALUDE_median_in_60_64_interval_l1010_101055


namespace NUMINAMATH_CALUDE_three_k_values_with_integer_roots_l1010_101074

/-- A quadratic equation with coefficient k has only integer roots -/
def has_only_integer_roots (k : ℝ) : Prop :=
  ∃ r s : ℤ, ∀ x : ℝ, x^2 + k*x + 4*k = 0 ↔ (x = r ∨ x = s)

/-- The set of real numbers k for which the quadratic equation has only integer roots -/
def integer_root_k_values : Set ℝ :=
  {k : ℝ | has_only_integer_roots k}

/-- There are exactly three values of k for which the quadratic equation has only integer roots -/
theorem three_k_values_with_integer_roots :
  ∃ k₁ k₂ k₃ : ℝ, k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₂ ≠ k₃ ∧
  integer_root_k_values = {k₁, k₂, k₃} :=
sorry

end NUMINAMATH_CALUDE_three_k_values_with_integer_roots_l1010_101074


namespace NUMINAMATH_CALUDE_simplify_expression_l1010_101020

theorem simplify_expression (a b : ℝ) : 4 * (a - 2 * b) - 2 * (2 * a + 3 * b) = -14 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1010_101020


namespace NUMINAMATH_CALUDE_product_of_fractions_l1010_101082

theorem product_of_fractions : (1 + 1/3) * (1 + 1/4) = 5/3 := by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1010_101082


namespace NUMINAMATH_CALUDE_not_all_rectangles_similar_l1010_101058

/-- A rectangle is a parallelogram with all interior angles equal to 90 degrees. -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angle_measure : ℝ
  is_parallelogram : True
  right_angles : angle_measure = 90

/-- Similarity in shapes means corresponding angles are equal and ratios of corresponding sides are constant. -/
def are_similar (r1 r2 : Rectangle) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 4, r1.sides i = k * r2.sides i

/-- Theorem: Not all rectangles are similar to each other. -/
theorem not_all_rectangles_similar : ¬ ∀ r1 r2 : Rectangle, are_similar r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_not_all_rectangles_similar_l1010_101058


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1010_101000

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ a b : ℝ, a + b > 0 ∧ a * b ≤ 0) ∧ 
  (∃ a b : ℝ, a * b > 0 ∧ a + b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1010_101000


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l1010_101045

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 25 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l1010_101045


namespace NUMINAMATH_CALUDE_complex_number_absolute_value_squared_l1010_101024

theorem complex_number_absolute_value_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a b
  (z + Complex.abs z = 1 + 12 * Complex.I) → Complex.abs z ^ 2 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_absolute_value_squared_l1010_101024


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1010_101037

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1010_101037


namespace NUMINAMATH_CALUDE_wine_glass_ball_radius_l1010_101085

theorem wine_glass_ball_radius 
  (parabola : ℝ → ℝ → Prop) 
  (h_parabola : ∀ x y, parabola x y ↔ x^2 = 2*y) 
  (h_y_range : ∀ y, parabola x y → 0 ≤ y ∧ y ≤ 20) 
  (ball_touches_bottom : ∃ r, r > 0 ∧ ∀ x y, parabola x y → x^2 + y^2 ≥ r^2) :
  ∃ r, r > 0 ∧ r ≤ 1 ∧ 
    (∀ x y, parabola x y → x^2 + y^2 ≥ r^2) ∧
    (∀ r', r' > 0 ∧ r' ≤ 1 → 
      (∀ x y, parabola x y → x^2 + y^2 ≥ r'^2) → 
      r' ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_wine_glass_ball_radius_l1010_101085


namespace NUMINAMATH_CALUDE_inequality_proof_l1010_101054

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1010_101054


namespace NUMINAMATH_CALUDE_max_above_average_students_l1010_101018

theorem max_above_average_students (n : ℕ) (h : n = 150) :
  ∃ (scores : Fin n → ℚ),
    (∃ (count : ℕ), count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧
                    count = n - 1) ∧
    ∀ (count : ℕ),
      count = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card →
      count ≤ n - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_above_average_students_l1010_101018


namespace NUMINAMATH_CALUDE_problem_statement_l1010_101071

theorem problem_statement (a b c : ℝ) (h1 : a - b = 3) (h2 : a - c = 1) :
  (c - b)^2 - 2*(c - b) + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1010_101071


namespace NUMINAMATH_CALUDE_system_solution_l1010_101091

theorem system_solution : 
  ∃! (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l1010_101091


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1010_101060

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1010_101060


namespace NUMINAMATH_CALUDE_lemonade_percentage_l1010_101098

/-- Proves that the percentage of lemonade in the second solution is 45% -/
theorem lemonade_percentage
  (first_solution_carbonated : ℝ)
  (second_solution_carbonated : ℝ)
  (mixture_ratio : ℝ)
  (mixture_carbonated : ℝ)
  (h1 : first_solution_carbonated = 0.8)
  (h2 : second_solution_carbonated = 0.55)
  (h3 : mixture_ratio = 0.5)
  (h4 : mixture_carbonated = 0.675)
  (h5 : mixture_ratio * first_solution_carbonated + (1 - mixture_ratio) * second_solution_carbonated = mixture_carbonated) :
  1 - second_solution_carbonated = 0.45 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_percentage_l1010_101098


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l1010_101013

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 343 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l1010_101013


namespace NUMINAMATH_CALUDE_philips_banana_groups_l1010_101075

theorem philips_banana_groups (total_bananas : ℕ) (group_size : ℕ) (h1 : total_bananas = 180) (h2 : group_size = 18) :
  total_bananas / group_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l1010_101075


namespace NUMINAMATH_CALUDE_min_shared_side_length_l1010_101023

/-- Given two triangles EFG and HFG sharing side FG, with specified side lengths,
    prove that the smallest possible integral value for FG is 15. -/
theorem min_shared_side_length (EF HG : ℝ) (EG HF : ℝ) : EF = 6 → EG = 15 → HG = 10 → HF = 25 →
  ∃ (FG : ℕ), FG = 15 ∧ ∀ (x : ℕ), (x : ℝ) > EG - EF ∧ (x : ℝ) > HF - HG → x ≥ FG :=
by sorry

end NUMINAMATH_CALUDE_min_shared_side_length_l1010_101023


namespace NUMINAMATH_CALUDE_max_angles_theorem_l1010_101015

theorem max_angles_theorem (k : ℕ) :
  let n := 2 * k
  ∃ (max_angles : ℕ), max_angles = 2 * k - 1 ∧
    ∀ (num_angles : ℕ), num_angles ≤ max_angles :=
by sorry

end NUMINAMATH_CALUDE_max_angles_theorem_l1010_101015


namespace NUMINAMATH_CALUDE_rackets_sold_l1010_101014

/-- Given the total sales and average price of ping pong rackets, 
    prove the number of pairs sold. -/
theorem rackets_sold (total_sales : ℝ) (avg_price : ℝ) 
  (h1 : total_sales = 686)
  (h2 : avg_price = 9.8) : 
  total_sales / avg_price = 70 := by
  sorry

end NUMINAMATH_CALUDE_rackets_sold_l1010_101014


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1010_101039

theorem longest_side_of_triangle (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- Side lengths are positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given conditions
  (Real.tan A = 1/4) →
  (Real.tan B = 3/5) →
  (min a (min b c) = Real.sqrt 2) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Conclusion
  max a (max b c) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1010_101039


namespace NUMINAMATH_CALUDE_system_solution_l1010_101099

theorem system_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1010_101099


namespace NUMINAMATH_CALUDE_sqrt_three_between_fractions_l1010_101043

theorem sqrt_three_between_fractions (n : ℕ+) :
  ((n + 3 : ℝ) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4 : ℝ) / (n + 1)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_between_fractions_l1010_101043


namespace NUMINAMATH_CALUDE_dodecagon_min_rotation_l1010_101093

/-- The minimum rotation angle for a regular dodecagon to coincide with itself -/
def min_rotation_angle_dodecagon : ℝ := 30

/-- Theorem: The minimum rotation angle for a regular dodecagon to coincide with itself is 30° -/
theorem dodecagon_min_rotation :
  min_rotation_angle_dodecagon = 30 := by sorry

end NUMINAMATH_CALUDE_dodecagon_min_rotation_l1010_101093


namespace NUMINAMATH_CALUDE_function_upper_bound_implies_parameter_range_l1010_101019

theorem function_upper_bound_implies_parameter_range 
  (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) →
  (∀ x, f x = Real.sin x ^ 2 + a * Real.cos x + a) →
  a ∈ Set.Iic 0 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_implies_parameter_range_l1010_101019


namespace NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1010_101095

def total_tiles : ℕ := 6
def x_tiles : ℕ := 4
def o_tiles : ℕ := 2

theorem probability_of_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_arrangement_l1010_101095


namespace NUMINAMATH_CALUDE_permutation_inequalities_l1010_101032

/-- Given a set X of n elements and 0 ≤ k ≤ n, a_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at least k components in common -/
def a (n k : ℕ) : ℕ := sorry

/-- Given a set X of n elements and 0 ≤ k ≤ n, b_{n,k} is the maximum number of permutations
    acting on X such that every two of them have at most k components in common -/
def b (n k : ℕ) : ℕ := sorry

theorem permutation_inequalities (n k : ℕ) (h : k ≤ n) :
  a n k * b n (k - 1) ≤ n! ∧ ∀ p : ℕ, Nat.Prime p → a p 2 = p! / 2 := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequalities_l1010_101032


namespace NUMINAMATH_CALUDE_bike_race_distance_difference_l1010_101076

theorem bike_race_distance_difference 
  (race_duration : ℝ) 
  (clara_speed : ℝ) 
  (denise_speed : ℝ) 
  (h1 : race_duration = 5)
  (h2 : clara_speed = 18)
  (h3 : denise_speed = 16) :
  clara_speed * race_duration - denise_speed * race_duration = 10 := by
sorry

end NUMINAMATH_CALUDE_bike_race_distance_difference_l1010_101076


namespace NUMINAMATH_CALUDE_x_range_proof_l1010_101036

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem x_range_proof (f : ℝ → ℝ) (h_odd : is_odd f) (h_decreasing : is_decreasing f)
  (h_domain : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ∈ Set.Icc (-3 : ℝ) 3)
  (h_inequality : ∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f (x^2 - 2*x) + f (x - 2) < 0) :
  ∀ x, x ∈ Set.Ioc (2 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_x_range_proof_l1010_101036


namespace NUMINAMATH_CALUDE_region_D_properties_l1010_101011

def region_D (x y : ℝ) : Prop :=
  2 ≤ x ∧ x ≤ 6 ∧
  1 ≤ y ∧ y ≤ 3 ∧
  x^2 / 9 + y^2 / 4 < 1 ∧
  4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧
  0 < y ∧ y < x

theorem region_D_properties (x y : ℝ) :
  region_D x y →
  (2 ≤ x ∧ x ≤ 6) ∧
  (1 ≤ y ∧ y ≤ 3) ∧
  (x^2 / 9 + y^2 / 4 < 1) ∧
  (4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9) ∧
  (0 < y ∧ y < x) :=
by sorry

end NUMINAMATH_CALUDE_region_D_properties_l1010_101011
