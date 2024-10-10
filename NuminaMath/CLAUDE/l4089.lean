import Mathlib

namespace min_distance_to_origin_l4089_408935

/-- The minimum distance from a point on the line 5x + 12y = 60 to the origin (0, 0) is 60/13 -/
theorem min_distance_to_origin (x y : ℝ) : 
  5 * x + 12 * y = 60 → 
  (∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p.1 * 5 + p.2 * 12 = 60 → 
      d ≤ Real.sqrt (p.1^2 + p.2^2)) := by
  sorry

end min_distance_to_origin_l4089_408935


namespace sum_simplification_l4089_408933

theorem sum_simplification : -2^3 + (-2)^4 + 2^2 - 2^3 = 4 := by
  sorry

end sum_simplification_l4089_408933


namespace carson_carpool_expense_l4089_408963

/-- Represents the carpool scenario with given parameters --/
structure CarpoolScenario where
  num_friends : Nat
  one_way_miles : Nat
  gas_price : Rat
  miles_per_gallon : Nat
  days_per_week : Nat
  weeks_per_month : Nat

/-- Calculates the monthly gas expense per person for a given carpool scenario --/
def monthly_gas_expense_per_person (scenario : CarpoolScenario) : Rat :=
  let total_miles := 2 * scenario.one_way_miles * scenario.days_per_week * scenario.weeks_per_month
  let total_gallons := total_miles / scenario.miles_per_gallon
  let total_cost := total_gallons * scenario.gas_price
  total_cost / scenario.num_friends

/-- The given carpool scenario --/
def carson_carpool : CarpoolScenario :=
  { num_friends := 5
  , one_way_miles := 21
  , gas_price := 5/2
  , miles_per_gallon := 30
  , days_per_week := 5
  , weeks_per_month := 4
  }

/-- Theorem stating that the monthly gas expense per person for Carson's carpool is $14 --/
theorem carson_carpool_expense :
  monthly_gas_expense_per_person carson_carpool = 14 := by
  sorry


end carson_carpool_expense_l4089_408963


namespace complex_fraction_simplification_l4089_408906

theorem complex_fraction_simplification : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_simplification_l4089_408906


namespace y_intercept_of_line_l4089_408990

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4). -/
theorem y_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 0 → y = 4) :=
by sorry

end y_intercept_of_line_l4089_408990


namespace complex_number_problem_l4089_408932

theorem complex_number_problem (a b c : ℂ) 
  (h_a_real : a.im = 0)
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 6)
  (h_prod : a * b * c = 4) :
  a = 2 := by
sorry

end complex_number_problem_l4089_408932


namespace equation_simplification_l4089_408910

theorem equation_simplification (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end equation_simplification_l4089_408910


namespace quadratic_inequality_equivalence_l4089_408934

theorem quadratic_inequality_equivalence (x : ℝ) :
  x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end quadratic_inequality_equivalence_l4089_408934


namespace cube_split_sequence_l4089_408909

theorem cube_split_sequence (n : ℕ) : ∃ (k : ℕ), 
  2019 = n^2 - (n - 1) + 2 * k ∧ 
  0 ≤ k ∧ 
  k < n ∧ 
  n = 45 := by
  sorry

end cube_split_sequence_l4089_408909


namespace min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l4089_408951

/-- Represents the types of fruits on the magical tree -/
inductive Fruit
  | Banana
  | Orange

/-- Represents the state of the magical tree -/
structure TreeState where
  bananas : Nat
  oranges : Nat

/-- Represents the possible picking actions -/
inductive PickAction
  | PickOne (f : Fruit)
  | PickTwo (f1 f2 : Fruit)

/-- Applies a picking action to the tree state -/
def applyAction (state : TreeState) (action : PickAction) : TreeState :=
  match action with
  | PickAction.PickOne Fruit.Banana => state
  | PickAction.PickOne Fruit.Orange => state
  | PickAction.PickTwo Fruit.Banana Fruit.Banana => 
      { bananas := state.bananas - 2, oranges := state.oranges + 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Banana Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Banana => 
      { bananas := state.bananas, oranges := state.oranges - 1 }

/-- Defines the initial state of the tree -/
def initialState : TreeState := { bananas := 15, oranges := 20 }

/-- Theorem: The minimum number of fruits that can remain on the tree is 1 -/
theorem min_remaining_fruits (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas + 
    (List.foldl applyAction initialState actions).oranges ≥ 1 :=
  sorry

/-- Theorem: The last remaining fruit is always a banana -/
theorem last_fruit_is_banana (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 1 ∧
    (List.foldl applyAction initialState actions).oranges = 0 :=
  sorry

/-- Theorem: It's impossible to remove all fruits from the tree -/
theorem cannot_remove_all_fruits (actions : List PickAction) :
  ¬(∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 0 ∧
    (List.foldl applyAction initialState actions).oranges = 0) :=
  sorry

end min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l4089_408951


namespace total_legs_of_three_spiders_l4089_408940

def human_legs : ℕ := 2

def spider1_legs : ℕ := 2 * (2 * human_legs)

def spider2_legs : ℕ := 3 * spider1_legs

def spider3_legs : ℕ := spider2_legs - 5

def total_spider_legs : ℕ := spider1_legs + spider2_legs + spider3_legs

theorem total_legs_of_three_spiders :
  total_spider_legs = 51 := by sorry

end total_legs_of_three_spiders_l4089_408940


namespace cube_root_of_3375_l4089_408920

theorem cube_root_of_3375 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 3375) : x = 15 := by
  sorry

end cube_root_of_3375_l4089_408920


namespace quadratic_roots_property_l4089_408905

theorem quadratic_roots_property (r s : ℝ) : 
  (∃ α β : ℝ, α + β = 10 ∧ α^2 - β^2 = 8 ∧ α^2 + r*α + s = 0 ∧ β^2 + r*β + s = 0) → 
  r = -10 := by
sorry

end quadratic_roots_property_l4089_408905


namespace new_person_weight_l4089_408927

/-- Proves that the weight of a new person is 380 kg given the conditions of the problem -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (average_increase : ℝ) :
  initial_count = 20 →
  replaced_weight = 80 →
  average_increase = 15 →
  (initial_count : ℝ) * average_increase + replaced_weight = 380 :=
by sorry

end new_person_weight_l4089_408927


namespace fraction_simplification_l4089_408925

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + x^2) - (x^2 - b*a^2) / Real.sqrt (a^2 + x^2) + b) / (a^2 + x^2 + b^2) =
  (1 + b) / Real.sqrt ((a^2 + x^2) * (a^2 + x^2 + b^2)) :=
by sorry

end fraction_simplification_l4089_408925


namespace compound_interest_calculation_l4089_408918

/-- Given a principal amount that yields $50 in simple interest over 2 years at 5% per annum,
    the compound interest for the same principal, rate, and time is $51.25. -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 0.05 * 2 = 50) →  -- Simple interest condition
  (P * (1 + 0.05)^2 - P = 51.25) :=  -- Compound interest calculation
by
  sorry


end compound_interest_calculation_l4089_408918


namespace eulers_formula_applications_l4089_408999

open Complex

theorem eulers_formula_applications :
  let e_2pi_3i : ℂ := Complex.exp ((2 * Real.pi / 3) * I)
  let e_pi_2i : ℂ := Complex.exp ((Real.pi / 2) * I)
  let e_pi_i : ℂ := Complex.exp (Real.pi * I)
  (e_2pi_3i.re < 0 ∧ e_2pi_3i.im > 0) ∧
  (e_pi_2i = I) ∧
  (abs (e_pi_i / (Real.sqrt 3 + I)) = 1/2) :=
by sorry

end eulers_formula_applications_l4089_408999


namespace claire_gift_card_balance_l4089_408969

/-- Calculates the remaining balance on a gift card after a week of purchases --/
def remaining_balance (gift_card_amount : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (cookie_count : ℕ) : ℚ :=
  gift_card_amount - (latte_cost + croissant_cost) * days - cookie_cost * cookie_count

/-- Proves that the remaining balance on Claire's gift card is $43 --/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end claire_gift_card_balance_l4089_408969


namespace segment_length_on_number_line_l4089_408991

theorem segment_length_on_number_line : 
  let a : ℝ := -3
  let b : ℝ := 5
  |b - a| = 8 := by sorry

end segment_length_on_number_line_l4089_408991


namespace seven_b_value_l4089_408944

theorem seven_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 7 * b = 168 / 11 := by
  sorry

end seven_b_value_l4089_408944


namespace pyramid_height_l4089_408976

/-- The height of a pyramid with a rectangular base and isosceles triangular faces -/
theorem pyramid_height (ab bc : ℝ) (volume : ℝ) (h_ab : ab = 15 * Real.sqrt 3) 
  (h_bc : bc = 14 * Real.sqrt 3) (h_volume : volume = 750) : ℝ := 
  let base_area := ab * bc
  let height := 3 * volume / base_area
  by
    -- Proof goes here
    sorry

#check pyramid_height

end pyramid_height_l4089_408976


namespace stating_reach_target_probability_approx_l4089_408967

/-- Represents the probability of winning in a single bet -/
def win_probability : ℝ := 0.1

/-- Represents the cost of a single bet -/
def bet_cost : ℝ := 10

/-- Represents the amount won in a single successful bet -/
def win_amount : ℝ := 30

/-- Represents the initial amount of money -/
def initial_amount : ℝ := 20

/-- Represents the target amount to reach -/
def target_amount : ℝ := 45

/-- 
Represents the probability of reaching the target amount 
starting from the initial amount through a series of bets
-/
noncomputable def reach_target_probability : ℝ := sorry

/-- 
Theorem stating that the probability of reaching the target amount 
is approximately 0.033
-/
theorem reach_target_probability_approx : 
  |reach_target_probability - 0.033| < 0.001 := by sorry

end stating_reach_target_probability_approx_l4089_408967


namespace complex_on_imaginary_axis_l4089_408904

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) : ℂ).re = 0 → a = 0 ∨ a = 2 :=
by sorry

end complex_on_imaginary_axis_l4089_408904


namespace triangle_exterior_angle_l4089_408943

theorem triangle_exterior_angle (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : A = B) (h3 : A = 40 ∨ B = 40 ∨ C = 40) : 
  180 - C = 80 ∨ 180 - C = 140 := by
  sorry

end triangle_exterior_angle_l4089_408943


namespace range_of_a_l4089_408917

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a > 1 := by
  sorry

end range_of_a_l4089_408917


namespace distance_equality_l4089_408958

/-- Given four points in 3D space, prove that a specific point P satisfies the distance conditions --/
theorem distance_equality (A B C D P : ℝ × ℝ × ℝ) : 
  A = (10, 0, 0) →
  B = (0, -6, 0) →
  C = (0, 0, 8) →
  D = (1, 1, 1) →
  P = (3, -2, 5) →
  dist A P = dist B P ∧ 
  dist A P = dist C P ∧ 
  dist A P = dist D P - 3 := by
  sorry

where
  dist : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ
  | (x₁, y₁, z₁), (x₂, y₂, z₂) => Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

end distance_equality_l4089_408958


namespace parabola_and_line_properties_l4089_408922

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the intersecting line
def intersecting_line (x y : ℝ) : Prop := ∃ t, x = t*y + 4

-- Define the tangent line
def tangent_line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the circle condition
def circle_condition (x₀ k m r : ℝ) : Prop :=
  ∃ x y, tangent_line k m x y ∧
  (2*m^2 - r)*(x₀ - r) + 2*k*m*x₀ + 2*m^2 = 0

theorem parabola_and_line_properties :
  ∀ p : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    intersecting_line x₁ y₁ ∧ intersecting_line x₂ y₂ ∧
    y₁ * y₂ = -8) →
  p = 1 ∧
  (∀ k m r : ℝ,
    (∃ x y, parabola 1 x y ∧ tangent_line k m x y) →
    (∀ x₀, circle_condition x₀ k m r → x₀ = -1/2)) :=
by sorry

end parabola_and_line_properties_l4089_408922


namespace robertos_salary_proof_l4089_408993

theorem robertos_salary_proof (current_salary : ℝ) : 
  current_salary = 134400 →
  ∃ (starting_salary : ℝ),
    starting_salary = 80000 ∧
    current_salary = 1.2 * (1.4 * starting_salary) :=
by
  sorry

end robertos_salary_proof_l4089_408993


namespace quadratic_equations_solutions_l4089_408964

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 + 5*x - 24 = 0) ∧
  (∃ x : ℝ, 3*x^2 = 2*(2-x)) ∧
  (∀ x : ℝ, x^2 + 5*x - 24 = 0 ↔ (x = -8 ∨ x = 3)) ∧
  (∀ x : ℝ, 3*x^2 = 2*(2-x) ↔ (x = (-1 + Real.sqrt 13) / 3 ∨ x = (-1 - Real.sqrt 13) / 3)) :=
by sorry


end quadratic_equations_solutions_l4089_408964


namespace stewart_farm_sheep_count_l4089_408926

/-- Proves that the number of sheep is 32 given the farm conditions --/
theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  (sheep : ℚ) / (horses : ℚ) = 4 / 7 →
  horses * 230 = 12880 →
  sheep = 32 := by
sorry

end stewart_farm_sheep_count_l4089_408926


namespace D_2021_2022_2023_odd_l4089_408941

def D : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n + 3 => D (n + 2) + D (n + 1)

theorem D_2021_2022_2023_odd :
  Odd (D 2021) ∧ Odd (D 2022) ∧ Odd (D 2023) := by
  sorry

end D_2021_2022_2023_odd_l4089_408941


namespace f_max_value_l4089_408912

/-- The quadratic function f(x) = -3x^2 + 15x + 9 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 15 * x + 9

/-- The maximum value of f(x) is 111/4 -/
theorem f_max_value : ∃ (M : ℝ), M = 111/4 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end f_max_value_l4089_408912


namespace exchange_point_configuration_exists_multiple_configurations_exist_l4089_408946

/-- A planar graph representing a city map -/
structure CityMap where
  -- The number of edges (roads) in the map
  num_edges : ℕ
  -- The initial number of vertices (exchange points)
  initial_vertices : ℕ
  -- The number of faces in the planar graph (city parts)
  num_faces : ℕ
  -- Euler's formula for planar graphs
  euler_formula : num_faces = num_edges - initial_vertices + 2

/-- The configuration of exchange points in the city -/
structure ExchangePointConfig where
  -- The total number of exchange points after adding new ones
  total_points : ℕ
  -- The number of points in each face
  points_per_face : ℕ
  -- Condition that each face has exactly two points
  two_points_per_face : points_per_face = 2
  -- The total number of points is consistent with the number of faces
  total_points_condition : total_points = num_faces * points_per_face

/-- Theorem stating that it's possible to add three exchange points to satisfy the conditions -/
theorem exchange_point_configuration_exists (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config : ExchangePointConfig), config.total_points = m.initial_vertices + 3 :=
sorry

/-- Theorem stating that multiple valid configurations exist -/
theorem multiple_configurations_exist (m : CityMap) (h : m.initial_vertices = 1) :
  ∃ (config1 config2 config3 config4 : ExchangePointConfig),
    config1 ≠ config2 ∧ config1 ≠ config3 ∧ config1 ≠ config4 ∧
    config2 ≠ config3 ∧ config2 ≠ config4 ∧ config3 ≠ config4 ∧
    (∀ c ∈ [config1, config2, config3, config4], c.total_points = m.initial_vertices + 3) :=
sorry

end exchange_point_configuration_exists_multiple_configurations_exist_l4089_408946


namespace translation_left_proof_l4089_408954

def translate_left (x y : ℝ) (d : ℝ) : ℝ × ℝ :=
  (x - d, y)

theorem translation_left_proof :
  let A : ℝ × ℝ := (1, 2)
  let A₁ : ℝ × ℝ := translate_left A.1 A.2 1
  A₁ = (0, 2) := by sorry

end translation_left_proof_l4089_408954


namespace smallest_integer_with_remainders_l4089_408913

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 9 = 5 ∧ 
  b % 11 = 7 ∧
  ∀ c : ℕ, c > 0 ∧ c % 9 = 5 ∧ c % 11 = 7 → b ≤ c :=
by sorry

end smallest_integer_with_remainders_l4089_408913


namespace remaining_math_problems_l4089_408953

theorem remaining_math_problems (total : ℕ) (completed : ℕ) (remaining : ℕ) : 
  total = 9 → completed = 5 → remaining = total - completed → remaining = 4 := by
  sorry

end remaining_math_problems_l4089_408953


namespace mechanic_hourly_rate_l4089_408901

/-- The mechanic's hourly rate calculation -/
theorem mechanic_hourly_rate :
  let hours_per_day : ℕ := 8
  let days_worked : ℕ := 14
  let parts_cost : ℕ := 2500
  let total_cost : ℕ := 9220
  let total_hours : ℕ := hours_per_day * days_worked
  let labor_cost : ℕ := total_cost - parts_cost
  labor_cost / total_hours = 60 := by sorry

end mechanic_hourly_rate_l4089_408901


namespace parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l4089_408948

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Axioms for the relations
axiom parallel_planes_trans {a b c : Plane} : 
  parallel_planes a b → parallel_planes b c → parallel_planes a c

axiom perpendicular_planes_of_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b

axiom parallel_line_plane_of_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a

-- Theorems to prove
theorem parallel_planes_transitive {a b c : Plane} :
  parallel_planes a b → parallel_planes b c → parallel_planes a c :=
sorry

theorem perpendicular_planes_from_perpendicular_lines {a b : Plane} {m n : Line} :
  perpendicular_plane_line a m → perpendicular_plane_line b n → 
  perpendicular_lines m n → perpendicular_planes a b :=
sorry

theorem parallel_line_plane_from_perpendicular_planes {a b : Plane} {m : Line} :
  perpendicular_planes a b → perpendicular_plane_line b m → 
  ¬line_in_plane m a → parallel_line_plane m a :=
sorry

end parallel_planes_transitive_perpendicular_planes_from_perpendicular_lines_parallel_line_plane_from_perpendicular_planes_l4089_408948


namespace aquarium_fish_count_l4089_408914

theorem aquarium_fish_count (total : ℕ) (blue orange green : ℕ) : 
  total = 80 ∧ 
  blue = total / 2 ∧ 
  orange = blue - 15 ∧ 
  total = blue + orange + green → 
  green = 15 := by
sorry

end aquarium_fish_count_l4089_408914


namespace exponential_models_for_rapid_change_l4089_408937

/-- Represents an exponential function model -/
structure ExponentialModel where
  -- Add necessary fields here
  rapidChange : Bool
  largeChangeInShortTime : Bool

/-- Represents a practical problem with rapid changes and large amounts of change in short periods -/
structure RapidChangeProblem where
  -- Add necessary fields here
  hasRapidChange : Bool
  hasLargeChangeInShortTime : Bool

/-- States that exponential models are generally used for rapid change problems -/
theorem exponential_models_for_rapid_change 
  (model : ExponentialModel) 
  (problem : RapidChangeProblem) : 
  model.rapidChange ∧ model.largeChangeInShortTime → 
  problem.hasRapidChange ∧ problem.hasLargeChangeInShortTime →
  (∃ (usage : Bool), usage = true) :=
by
  sorry

#check exponential_models_for_rapid_change

end exponential_models_for_rapid_change_l4089_408937


namespace trapezoid_perimeter_is_46_l4089_408980

/-- A trapezoid within a rectangle -/
structure TrapezoidInRectangle where
  a : ℝ  -- Length of longer parallel side of trapezoid
  b : ℝ  -- Length of shorter parallel side of trapezoid
  h : ℝ  -- Height of trapezoid (equal to non-parallel sides)
  rect_perimeter : ℝ  -- Perimeter of the rectangle

/-- The perimeter of the trapezoid -/
def trapezoid_perimeter (t : TrapezoidInRectangle) : ℝ :=
  t.a + t.b + 2 * t.h

/-- Theorem stating the perimeter of the trapezoid is 46 meters -/
theorem trapezoid_perimeter_is_46 (t : TrapezoidInRectangle)
  (h1 : t.a = 15)
  (h2 : t.b = 9)
  (h3 : t.rect_perimeter = 52)
  (h4 : t.h = (t.rect_perimeter - 2 * t.a) / 2) :
  trapezoid_perimeter t = 46 := by
  sorry

end trapezoid_perimeter_is_46_l4089_408980


namespace max_gold_marbles_is_66_l4089_408915

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange of marbles --/
inductive Exchange
  | RedToGold : Exchange
  | BlueToGold : Exchange

/-- Applies an exchange to a MarbleCount --/
def applyExchange (mc : MarbleCount) (e : Exchange) : MarbleCount :=
  match e with
  | Exchange.RedToGold => 
      if mc.red ≥ 3 then ⟨mc.red - 3, mc.blue + 2, mc.gold + 1⟩ else mc
  | Exchange.BlueToGold => 
      if mc.blue ≥ 4 then ⟨mc.red + 1, mc.blue - 4, mc.gold + 1⟩ else mc

/-- Checks if any exchange is possible --/
def canExchange (mc : MarbleCount) : Prop :=
  mc.red ≥ 3 ∨ mc.blue ≥ 4

/-- The maximum number of gold marbles obtainable --/
def maxGoldMarbles : ℕ := 66

/-- The theorem to be proved --/
theorem max_gold_marbles_is_66 :
  ∀ (exchanges : List Exchange),
    let finalCount := (exchanges.foldl applyExchange ⟨80, 60, 0⟩)
    ¬(canExchange finalCount) →
    finalCount.gold = maxGoldMarbles :=
  sorry

end max_gold_marbles_is_66_l4089_408915


namespace eleven_bonnets_per_orphanage_l4089_408957

/-- The number of bonnets Mrs. Young makes in a week and distributes to orphanages -/
def bonnet_distribution (monday : ℕ) : ℕ → ℕ :=
  fun orphanages =>
    let tuesday_wednesday := 2 * monday
    let thursday := monday + 5
    let friday := thursday - 5
    let total := monday + tuesday_wednesday + thursday + friday
    total / orphanages

/-- Theorem stating that given the conditions in the problem, each orphanage receives 11 bonnets -/
theorem eleven_bonnets_per_orphanage :
  bonnet_distribution 10 5 = 11 := by
  sorry

end eleven_bonnets_per_orphanage_l4089_408957


namespace matches_played_before_increase_l4089_408978

def cricket_matches (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) : Prop :=
  ∃ m : ℕ,
    (current_average * m + next_match_runs) / (m + 1) = new_average ∧
    m > 0

theorem matches_played_before_increase (current_average : ℚ) (next_match_runs : ℕ) (new_average : ℚ) :
  cricket_matches current_average next_match_runs new_average →
  current_average = 51 →
  next_match_runs = 78 →
  new_average = 54 →
  ∃ m : ℕ, m = 8 ∧ cricket_matches current_average next_match_runs new_average :=
by
  sorry

#check matches_played_before_increase

end matches_played_before_increase_l4089_408978


namespace equation_solution_l4089_408961

theorem equation_solution :
  ∃ y : ℚ, 3 * y^(1/4) - 5 * (y / y^(3/4)) = 2 + y^(1/4) ∧ y = 16/81 :=
by
  sorry

end equation_solution_l4089_408961


namespace three_digit_factorial_sum_l4089_408942

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_digit_factorials (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  factorial hundreds + factorial tens + factorial units

theorem three_digit_factorial_sum :
  ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧ n / 100 = 2 ∧ n = sum_of_digit_factorials n :=
by
  sorry

end three_digit_factorial_sum_l4089_408942


namespace negative_division_subtraction_l4089_408908

theorem negative_division_subtraction : (-96) / (-24) - 3 = 1 := by
  sorry

end negative_division_subtraction_l4089_408908


namespace bus_ride_cost_proof_l4089_408968

def bus_ride_cost : ℚ := 1.40
def train_ride_cost : ℚ := bus_ride_cost + 6.85
def combined_cost : ℚ := 9.65
def price_multiple : ℚ := 0.35

theorem bus_ride_cost_proof :
  (train_ride_cost = bus_ride_cost + 6.85) ∧
  (train_ride_cost + bus_ride_cost = combined_cost) ∧
  (∃ n : ℕ, bus_ride_cost = n * price_multiple) ∧
  (∃ m : ℕ, train_ride_cost = m * price_multiple) →
  bus_ride_cost = 1.40 :=
by sorry

end bus_ride_cost_proof_l4089_408968


namespace max_value_expression_l4089_408921

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_3 : x + y + z = 3)
  (x_ge_y : x ≥ y) (y_ge_z : y ≥ z) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 12 := by
  sorry

end max_value_expression_l4089_408921


namespace complex_fraction_equals_negative_two_l4089_408960

theorem complex_fraction_equals_negative_two
  (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + a*b + b^2 = 0) :
  (a^7 + b^7) / (a + b)^7 = -2 := by
  sorry

end complex_fraction_equals_negative_two_l4089_408960


namespace common_solution_conditions_l4089_408965

theorem common_solution_conditions (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end common_solution_conditions_l4089_408965


namespace unique_solution_cube_equation_l4089_408971

theorem unique_solution_cube_equation (x : ℝ) (h : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 ↔ x = 27 := by
  sorry

end unique_solution_cube_equation_l4089_408971


namespace axis_of_symmetry_is_x_equals_one_l4089_408903

/-- Represents a parabola of the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end axis_of_symmetry_is_x_equals_one_l4089_408903


namespace percentage_of_x_l4089_408972

theorem percentage_of_x (x y z : ℚ) : 
  x / y = 4 → 
  x + y = z → 
  y ≠ 0 → 
  z > 0 → 
  (2 * x - y) / x = 175 / 100 := by
  sorry

end percentage_of_x_l4089_408972


namespace retail_price_calculation_l4089_408923

/-- The retail price of a machine given wholesale price, discount rate, and profit rate -/
theorem retail_price_calculation (W D R : ℚ) (h1 : W = 126) (h2 : D = 0.10) (h3 : R = 0.20) :
  ∃ P : ℚ, (1 - D) * P = W + R * W :=
by
  sorry

end retail_price_calculation_l4089_408923


namespace spend_fifty_is_negative_fifty_l4089_408987

-- Define a type for monetary transactions
inductive MonetaryTransaction
| Receive (amount : ℤ)
| Spend (amount : ℤ)

-- Define a function to represent the sign of a transaction
def transactionSign (t : MonetaryTransaction) : ℤ :=
  match t with
  | MonetaryTransaction.Receive _ => 1
  | MonetaryTransaction.Spend _ => -1

-- State the theorem
theorem spend_fifty_is_negative_fifty 
  (h1 : transactionSign (MonetaryTransaction.Receive 80) = 1)
  (h2 : transactionSign (MonetaryTransaction.Spend 50) = -transactionSign (MonetaryTransaction.Receive 50)) :
  transactionSign (MonetaryTransaction.Spend 50) * 50 = -50 := by
  sorry

end spend_fifty_is_negative_fifty_l4089_408987


namespace inequality_proof_l4089_408985

theorem inequality_proof (x b a : ℝ) (h1 : x < b) (h2 : b < a) (h3 : a < 0) :
  x^2 > a*b ∧ a*b > a^2 := by sorry

end inequality_proof_l4089_408985


namespace grandmother_age_is_132_l4089_408930

-- Define the ages as natural numbers
def mason_age : ℕ := 20
def sydney_age : ℕ := 3 * mason_age
def father_age : ℕ := sydney_age + 6
def grandmother_age : ℕ := 2 * father_age

-- Theorem to prove
theorem grandmother_age_is_132 : grandmother_age = 132 := by
  sorry


end grandmother_age_is_132_l4089_408930


namespace wolves_out_hunting_l4089_408998

def wolves_in_pack : ℕ := 16
def meat_per_wolf_per_day : ℕ := 8
def days_between_hunts : ℕ := 5
def meat_per_deer : ℕ := 200
def deer_per_hunting_wolf : ℕ := 1

def total_meat_needed : ℕ := wolves_in_pack * meat_per_wolf_per_day * days_between_hunts

def deer_needed : ℕ := (total_meat_needed + meat_per_deer - 1) / meat_per_deer

theorem wolves_out_hunting (hunting_wolves : ℕ) : 
  hunting_wolves * deer_per_hunting_wolf = deer_needed → hunting_wolves = 4 := by
  sorry

end wolves_out_hunting_l4089_408998


namespace telescope_visual_range_increase_l4089_408938

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 87.5 := by
  sorry

end telescope_visual_range_increase_l4089_408938


namespace function_and_inequality_proof_l4089_408973

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x ↦ sorry

-- Define the function g
noncomputable def g : ℝ → ℝ := fun x ↦ (f x - 2 * x) / x

-- Theorem statement
theorem function_and_inequality_proof :
  (∀ x y : ℝ, f (x + y) - f y = (x + 2 * y - 2) * x) ∧
  (f 1 = 0) ∧
  (∀ x : ℝ, f x = (x - 1)^2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → g (2^x) - k * 2^x ≤ 0) ↔ k ≥ 1) :=
by sorry

end function_and_inequality_proof_l4089_408973


namespace alpha_plus_beta_eq_115_l4089_408931

theorem alpha_plus_beta_eq_115 :
  ∃ (α β : ℝ), (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 116*x + 2783) / (x^2 + 99*x - 4080)) →
  α + β = 115 := by
  sorry

end alpha_plus_beta_eq_115_l4089_408931


namespace total_dolls_l4089_408988

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that the total number of dolls they have together is 48. -/
theorem total_dolls (hannah_multiplier sister_dolls : ℕ) 
  (h1 : hannah_multiplier = 5)
  (h2 : sister_dolls = 8) :
  hannah_multiplier * sister_dolls + sister_dolls = 48 := by
  sorry

end total_dolls_l4089_408988


namespace angle_b_in_axisymmetric_triangle_l4089_408928

-- Define an axisymmetric triangle
structure AxisymmetricTriangle :=
  (A B C : ℝ)
  (axisymmetric : True)  -- This is a placeholder for the axisymmetric property
  (sum_of_angles : A + B + C = 180)

-- Theorem statement
theorem angle_b_in_axisymmetric_triangle 
  (triangle : AxisymmetricTriangle) 
  (angle_a_value : triangle.A = 70) :
  triangle.B = 70 ∨ triangle.B = 55 := by
  sorry

end angle_b_in_axisymmetric_triangle_l4089_408928


namespace orange_face_probability_l4089_408981

def die_sides : ℕ := 12
def orange_faces : ℕ := 4

theorem orange_face_probability :
  (orange_faces : ℚ) / die_sides = 1 / 3 := by sorry

end orange_face_probability_l4089_408981


namespace coronavirus_case_ratio_l4089_408959

/-- Proves that the ratio of new coronavirus cases in the second week to the first week is 1/4 -/
theorem coronavirus_case_ratio :
  let first_week : ℕ := 5000
  let third_week (second_week : ℕ) : ℕ := second_week + 2000
  let total_cases : ℕ := 9500
  ∀ second_week : ℕ,
    first_week + second_week + third_week second_week = total_cases →
    (second_week : ℚ) / first_week = 1 / 4 := by
  sorry

end coronavirus_case_ratio_l4089_408959


namespace hyperbola_k_range_l4089_408955

/-- Given points A(-3,m) and B(-2,n) lying on the hyperbolic function y = (k-1)/x, 
    with m > n, the range of k is k > 1 -/
theorem hyperbola_k_range (k m n : ℝ) : 
  (m = (k - 1) / (-3)) → 
  (n = (k - 1) / (-2)) → 
  (m > n) → 
  (k > 1) := by
  sorry

end hyperbola_k_range_l4089_408955


namespace sum_base_6_100_equals_666_l4089_408919

def base_6_to_10 (n : ℕ) : ℕ := sorry

def sum_base_6 (n : ℕ) : ℕ := sorry

theorem sum_base_6_100_equals_666 :
  sum_base_6 (base_6_to_10 100) = 666 := by sorry

end sum_base_6_100_equals_666_l4089_408919


namespace largest_divisor_five_consecutive_integers_l4089_408929

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 60 ∧ ¬(m ∣ (n * (n+1) * (n+2) * (n+3) * (n+4))) ∧
  ∀ k : ℤ, k ≤ 60 → k ∣ (n * (n+1) * (n+2) * (n+3) * (n+4)) :=
by sorry

end largest_divisor_five_consecutive_integers_l4089_408929


namespace A_eq_B_l4089_408989

/-- A coloring of points in the plane -/
structure Coloring (n : ℕ+) where
  color : ℕ → ℕ → Bool
  valid : ∀ x y x' y', x' ≤ x → y' ≤ y → x + y ≤ n → color x y = false → color x' y' = false

/-- The number of ways to choose n blue points with distinct x-coordinates -/
def A (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The number of ways to choose n blue points with distinct y-coordinates -/
def B (n : ℕ+) (c : Coloring n) : ℕ := sorry

/-- The main theorem: A = B for any valid coloring -/
theorem A_eq_B (n : ℕ+) (c : Coloring n) : A n c = B n c := by sorry

end A_eq_B_l4089_408989


namespace polynomial_functional_equation_l4089_408936

/-- A polynomial satisfies P(x+1) = P(x) + 2x + 1 for all x if and only if it is of the form x^2 + c for some constant c. -/
theorem polynomial_functional_equation (P : ℝ → ℝ) :
  (∀ x, P (x + 1) = P x + 2 * x + 1) ↔
  (∃ c, ∀ x, P x = x^2 + c) :=
sorry

end polynomial_functional_equation_l4089_408936


namespace washer_dryer_cost_l4089_408979

theorem washer_dryer_cost (total_cost : ℝ) (price_difference : ℝ) (dryer_cost : ℝ) : 
  total_cost = 1200 →
  price_difference = 220 →
  total_cost = dryer_cost + (dryer_cost + price_difference) →
  dryer_cost = 490 := by
sorry

end washer_dryer_cost_l4089_408979


namespace batsman_average_17th_inning_l4089_408949

def batsman_average (total_innings : ℕ) (last_inning_score : ℕ) (average_increase : ℚ) : ℚ :=
  (total_innings - 1 : ℚ) * (average_increase + last_inning_score / total_innings) + last_inning_score / total_innings

theorem batsman_average_17th_inning :
  batsman_average 17 92 3 = 44 := by sorry

end batsman_average_17th_inning_l4089_408949


namespace workshop_attendance_l4089_408995

/-- Represents the number of scientists at a workshop with various prize distributions -/
structure WorkshopAttendance where
  total : ℕ
  wolfPrize : ℕ
  nobelPrize : ℕ
  wolfAndNobel : ℕ

/-- Theorem stating the total number of scientists at the workshop -/
theorem workshop_attendance (w : WorkshopAttendance) 
  (h1 : w.wolfPrize = 31)
  (h2 : w.wolfAndNobel = 12)
  (h3 : w.nobelPrize = 23)
  (h4 : w.nobelPrize - w.wolfAndNobel = (w.total - w.wolfPrize - (w.nobelPrize - w.wolfAndNobel)) + 3) :
  w.total = 39 := by
  sorry


end workshop_attendance_l4089_408995


namespace orange_distribution_l4089_408986

theorem orange_distribution (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) 
    (h1 : total_oranges = 108)
    (h2 : bad_oranges = 36)
    (h3 : num_students = 12)
    (h4 : bad_oranges < total_oranges) :
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students) = 3 := by
  sorry

end orange_distribution_l4089_408986


namespace function_inequality_condition_l4089_408970

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 2 :=
by sorry

end function_inequality_condition_l4089_408970


namespace board_game_spaces_l4089_408982

/-- A board game with a certain number of spaces -/
structure BoardGame where
  total_spaces : ℕ

/-- A player's progress in the board game -/
structure PlayerProgress where
  spaces_moved : ℕ
  spaces_to_win : ℕ

/-- Susan's moves in the game -/
def susan_moves : ℕ := 8 + (2 - 5) + 6

/-- The number of spaces Susan needs to move to win -/
def spaces_to_win : ℕ := 37

/-- Theorem stating that the total number of spaces in the game
    is equal to the spaces Susan has moved plus the remaining spaces to win -/
theorem board_game_spaces (game : BoardGame) (susan : PlayerProgress) 
    (h1 : susan.spaces_moved = susan_moves)
    (h2 : susan.spaces_to_win = spaces_to_win - susan_moves) :
  game.total_spaces = susan.spaces_moved + susan.spaces_to_win := by
  sorry

end board_game_spaces_l4089_408982


namespace f_decreasing_iff_a_range_l4089_408975

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else -(x+1)^2 + 2

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem f_decreasing_iff_a_range (a : ℝ) :
  (is_decreasing (f a)) ↔ a ≥ 1/3 :=
sorry

end f_decreasing_iff_a_range_l4089_408975


namespace inscribed_circle_radius_l4089_408939

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 10) (hc : c = 20) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by sorry

end inscribed_circle_radius_l4089_408939


namespace quarter_count_proof_l4089_408994

/-- Represents the types of coins in the collection -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a collection of coins -/
structure CoinCollection where
  coins : List Coin

def CoinCollection.averageValue (c : CoinCollection) : ℚ :=
  sorry

def CoinCollection.addDimes (c : CoinCollection) (n : ℕ) : CoinCollection :=
  sorry

def CoinCollection.countQuarters (c : CoinCollection) : ℕ :=
  sorry

theorem quarter_count_proof (c : CoinCollection) :
  c.averageValue = 15 / 100 →
  (c.addDimes 2).averageValue = 17 / 100 →
  c.countQuarters = 4 :=
sorry

end quarter_count_proof_l4089_408994


namespace min_value_expression_l4089_408984

theorem min_value_expression (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end min_value_expression_l4089_408984


namespace sams_dimes_given_to_dad_l4089_408956

theorem sams_dimes_given_to_dad (initial_dimes : ℕ) (remaining_dimes : ℕ) 
  (h1 : initial_dimes = 9) 
  (h2 : remaining_dimes = 2) : 
  initial_dimes - remaining_dimes = 7 := by
  sorry

end sams_dimes_given_to_dad_l4089_408956


namespace roberto_outfits_l4089_408977

/-- The number of different outfits Roberto can put together -/
def number_of_outfits : ℕ := 180

/-- The number of pairs of trousers Roberto has -/
def number_of_trousers : ℕ := 6

/-- The number of shirts Roberto has -/
def number_of_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def number_of_jackets : ℕ := 4

/-- The number of shirts that cannot be worn with Jacket 1 -/
def number_of_restricted_shirts : ℕ := 2

theorem roberto_outfits :
  number_of_outfits = 
    number_of_trousers * number_of_shirts * number_of_jackets - 
    number_of_trousers * number_of_restricted_shirts := by
  sorry

end roberto_outfits_l4089_408977


namespace meeting_seating_arrangement_l4089_408950

theorem meeting_seating_arrangement (n : ℕ) (h : n = 7) : 
  Nat.choose n 2 = 21 := by
  sorry

end meeting_seating_arrangement_l4089_408950


namespace symmetric_point_proof_l4089_408947

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) in a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- Determines if two points are symmetric with respect to the origin. -/
def isSymmetricToOrigin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- The given point (3, -1). -/
def givenPoint : Point := ⟨3, -1⟩

/-- The point to be proven symmetric to the given point. -/
def symmetricPoint : Point := ⟨-3, 1⟩

/-- Theorem stating that the symmetricPoint is symmetric to the givenPoint with respect to the origin. -/
theorem symmetric_point_proof : isSymmetricToOrigin givenPoint symmetricPoint := by
  sorry

end symmetric_point_proof_l4089_408947


namespace spurs_total_basketballs_l4089_408992

/-- Represents a basketball team -/
structure BasketballTeam where
  num_players : ℕ
  balls_per_player : ℕ

/-- Calculates the total number of basketballs for a team -/
def total_basketballs (team : BasketballTeam) : ℕ :=
  team.num_players * team.balls_per_player

/-- The Spurs basketball team -/
def spurs : BasketballTeam :=
  { num_players := 35
    balls_per_player := 15 }

/-- Theorem: The Spurs basketball team has 525 basketballs in total -/
theorem spurs_total_basketballs :
  total_basketballs spurs = 525 := by
  sorry

end spurs_total_basketballs_l4089_408992


namespace inequality_proof_l4089_408966

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 * (a ^ (1/2)) + 3 * (b ^ (1/3)) ≥ 5 * ((a * b) ^ (1/5)) := by
  sorry

end inequality_proof_l4089_408966


namespace mike_books_equal_sum_l4089_408997

/-- The number of books Bobby has -/
def bobby_books : Nat := 142

/-- The number of books Kristi has -/
def kristi_books : Nat := 78

/-- The number of books Mike needs to have -/
def mike_books : Nat := bobby_books + kristi_books

theorem mike_books_equal_sum :
  mike_books = bobby_books + kristi_books := by
  sorry

end mike_books_equal_sum_l4089_408997


namespace p_range_l4089_408907

-- Define the function p(x)
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem p_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ p x = y) ↔ y ≥ 9 :=
by sorry

end p_range_l4089_408907


namespace sum_first_seven_primes_mod_eighth_prime_l4089_408945

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime : 
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end sum_first_seven_primes_mod_eighth_prime_l4089_408945


namespace least_common_multiple_first_ten_l4089_408983

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end least_common_multiple_first_ten_l4089_408983


namespace monotonic_decreasing_interval_of_f_l4089_408962

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem monotonic_decreasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≥ f y} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end monotonic_decreasing_interval_of_f_l4089_408962


namespace ceiling_light_ratio_l4089_408974

/-- Represents the number of bulbs required for each type of ceiling light -/
structure BulbRequirement where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the counts of different types of ceiling lights -/
structure CeilingLightCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of bulbs required -/
def totalBulbs (req : BulbRequirement) (counts : CeilingLightCounts) : Nat :=
  req.small * counts.small + req.medium * counts.medium + req.large * counts.large

/-- Theorem statement for the ceiling light problem -/
theorem ceiling_light_ratio 
  (req : BulbRequirement)
  (counts : CeilingLightCounts)
  (h1 : req.small = 1 ∧ req.medium = 2 ∧ req.large = 3)
  (h2 : counts.medium = 12)
  (h3 : counts.small = counts.medium + 10)
  (h4 : totalBulbs req counts = 118) :
  counts.large = 2 * counts.medium := by
  sorry

#check ceiling_light_ratio

end ceiling_light_ratio_l4089_408974


namespace class_composition_l4089_408902

theorem class_composition (total_students : ℕ) (girls_ratio boys_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3) :
  ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    girls * boys_ratio = boys * girls_ratio ∧
    girls = 32 ∧ 
    boys = 24 :=
by sorry

end class_composition_l4089_408902


namespace apples_in_market_l4089_408996

theorem apples_in_market (apples oranges : ℕ) : 
  apples = oranges + 27 →
  apples + oranges = 301 →
  apples = 164 := by
sorry

end apples_in_market_l4089_408996


namespace count_numbers_with_2_and_3_is_52_l4089_408952

/-- A function that counts the number of three-digit numbers with at least one 2 and one 3 -/
def count_numbers_with_2_and_3 : ℕ :=
  let hundreds_not_2_or_3 := 7 * 2  -- Case 1
  let hundreds_is_2 := 10 + 9       -- Case 2
  let hundreds_is_3 := 10 + 9       -- Case 3
  hundreds_not_2_or_3 + hundreds_is_2 + hundreds_is_3

/-- Theorem stating that the count of three-digit numbers with at least one 2 and one 3 is 52 -/
theorem count_numbers_with_2_and_3_is_52 : count_numbers_with_2_and_3 = 52 := by
  sorry

end count_numbers_with_2_and_3_is_52_l4089_408952


namespace money_equalization_l4089_408911

theorem money_equalization (xiaoli_money xiaogang_money : ℕ) : 
  xiaoli_money = 18 → xiaogang_money = 24 → 
  (xiaogang_money - (xiaoli_money + xiaogang_money) / 2) = 3 := by
  sorry

end money_equalization_l4089_408911


namespace exponent_division_equality_l4089_408916

theorem exponent_division_equality (a : ℝ) : a^6 / (-a)^2 = a^4 := by
  sorry

end exponent_division_equality_l4089_408916


namespace rectangle_square_overlap_ratio_l4089_408924

/-- Given a rectangle ABCD and a square EFGH, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the rectangle's length
    to its width is 10. -/
theorem rectangle_square_overlap_ratio (AB AD s : ℝ) (h1 : AB > 0) (h2 : AD > 0) (h3 : s > 0) : 
  (0.4 * AB * AD = 0.25 * s^2) → (AD = s / 4) → AB / AD = 10 := by
  sorry

end rectangle_square_overlap_ratio_l4089_408924


namespace quadratic_equation_solution_l4089_408900

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 ↔ x = 0) ∧ 
  (k - 1 ≠ 0) → 
  k = -1 := by
sorry

end quadratic_equation_solution_l4089_408900
