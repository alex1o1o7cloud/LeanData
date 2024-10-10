import Mathlib

namespace table_tennis_tournament_l201_20173

theorem table_tennis_tournament (n : ℕ) : 
  (∃ r : ℕ, r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) → 
  (∃! r : ℕ, r = 1 ∧ r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) :=
by sorry

end table_tennis_tournament_l201_20173


namespace magic_stick_height_difference_l201_20101

-- Define the edge length of the large cube in meters
def large_cube_edge : ℝ := 1

-- Define the edge length of the small cubes in centimeters
def small_cube_edge : ℝ := 1

-- Define the height of Mount Everest in meters
def everest_height : ℝ := 8844

-- Conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem magic_stick_height_difference :
  let large_cube_volume : ℝ := large_cube_edge ^ 3
  let small_cube_volume : ℝ := (small_cube_edge * cm_to_m) ^ 3
  let num_small_cubes : ℝ := large_cube_volume / small_cube_volume
  let magic_stick_height : ℝ := num_small_cubes * small_cube_edge * cm_to_m
  magic_stick_height - everest_height = 1156 := by
  sorry

end magic_stick_height_difference_l201_20101


namespace floor_of_expression_equals_2016_l201_20124

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define the expression
def expression : ℚ :=
  (factorial 2017 + factorial 2014) / (factorial 2016 + factorial 2015)

-- Theorem statement
theorem floor_of_expression_equals_2016 :
  ⌊expression⌋ = 2016 := by sorry

end floor_of_expression_equals_2016_l201_20124


namespace point_of_tangency_parabolas_l201_20158

/-- The point of tangency for two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 10*x + 18
  let g (y : ℝ) := y^2 + 60*y + 910
  ∃! p : ℝ × ℝ, 
    (p.2 = f p.1 ∧ p.1 = g p.2) ∧ 
    (∀ x y, y = f x ∧ x = g y → (x, y) = p) :=
by
  sorry

end point_of_tangency_parabolas_l201_20158


namespace range_of_3a_minus_b_l201_20191

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 2 ≤ a + b ∧ a + b ≤ 5) 
  (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 3*a - b ≤ x → x ≤ 7) ∧ 
  (∀ y, -2 ≤ y → y ≤ 3*a - b) :=
by sorry

end range_of_3a_minus_b_l201_20191


namespace fishermans_red_snappers_l201_20165

/-- The number of Red snappers caught daily -/
def red_snappers : ℕ := sorry

/-- The number of Tunas caught daily -/
def tunas : ℕ := 14

/-- The price of a Red snapper in dollars -/
def red_snapper_price : ℕ := 3

/-- The price of a Tuna in dollars -/
def tuna_price : ℕ := 2

/-- The total daily earnings in dollars -/
def total_earnings : ℕ := 52

theorem fishermans_red_snappers :
  red_snappers * red_snapper_price + tunas * tuna_price = total_earnings ∧
  red_snappers = 8 := by sorry

end fishermans_red_snappers_l201_20165


namespace geometric_sequence_ratio_l201_20110

/-- A geometric sequence with common ratio q satisfying 2a₄ = a₆ - a₅ -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = q * a n) ∧ (2 * a 4 = a 6 - a 5)

/-- The common ratio of a geometric sequence satisfying 2a₄ = a₆ - a₅ is either -1 or 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → q = -1 ∨ q = 2 := by
  sorry

end geometric_sequence_ratio_l201_20110


namespace only_isosceles_trapezoid_axially_not_centrally_symmetric_l201_20186

-- Define the set of geometric figures
inductive GeometricFigure
  | LineSegment
  | Square
  | Circle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define axial symmetry
def is_axially_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => true
  | GeometricFigure.Parallelogram => false

-- Define central symmetry
def is_centrally_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => false
  | GeometricFigure.Parallelogram => true

-- Theorem stating that only the isosceles trapezoid satisfies the condition
theorem only_isosceles_trapezoid_axially_not_centrally_symmetric :
  ∀ (figure : GeometricFigure),
    (is_axially_symmetric figure ∧ ¬is_centrally_symmetric figure) ↔
    (figure = GeometricFigure.IsoscelesTrapezoid) :=
by
  sorry

end only_isosceles_trapezoid_axially_not_centrally_symmetric_l201_20186


namespace solution_exists_unique_solution_l201_20167

theorem solution_exists : ∃ x : ℚ, 60 + x * 12 / (180 / 3) = 61 :=
by
  use 5
  sorry

theorem unique_solution (x : ℚ) : 60 + x * 12 / (180 / 3) = 61 ↔ x = 5 :=
by
  sorry

end solution_exists_unique_solution_l201_20167


namespace large_number_arithmetic_l201_20196

/-- The result of a series of arithmetic operations on large numbers. -/
theorem large_number_arithmetic :
  let start : ℕ := 1500000000000
  let subtract : ℕ := 877888888888
  let add : ℕ := 123456789012
  (start - subtract + add : ℕ) = 745567900124 := by
  sorry

end large_number_arithmetic_l201_20196


namespace castle_provisions_l201_20103

/-- Represents the initial number of people in the castle -/
def initial_people : ℕ := sorry

/-- Represents the number of days the initial provisions last -/
def initial_days : ℕ := 90

/-- Represents the number of days after which people leave -/
def days_before_leaving : ℕ := 30

/-- Represents the number of people who leave the castle -/
def people_leaving : ℕ := 100

/-- Represents the number of days the remaining provisions last -/
def remaining_days : ℕ := 90

theorem castle_provisions :
  initial_people * initial_days = 
  (initial_people * days_before_leaving) + 
  ((initial_people - people_leaving) * remaining_days) ∧
  initial_people = 300 :=
sorry

end castle_provisions_l201_20103


namespace field_trip_students_l201_20143

theorem field_trip_students (van_capacity : Nat) (num_vans : Nat) (num_adults : Nat) :
  van_capacity = 8 →
  num_vans = 3 →
  num_adults = 2 →
  (van_capacity * num_vans) - num_adults = 22 :=
by sorry

end field_trip_students_l201_20143


namespace total_shoes_l201_20129

/-- Given that Ellie has 8 pairs of shoes and Riley has 3 fewer pairs than Ellie,
    prove that they have 13 pairs of shoes in total. -/
theorem total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) :
  ellie_shoes = 8 →
  riley_difference = 3 →
  ellie_shoes + (ellie_shoes - riley_difference) = 13 := by
  sorry

end total_shoes_l201_20129


namespace frame_width_proof_l201_20100

theorem frame_width_proof (photo_width : ℝ) (photo_height : ℝ) (frame_width : ℝ) :
  photo_width = 12 →
  photo_height = 18 →
  (photo_width + 2 * frame_width) * (photo_height + 2 * frame_width) - photo_width * photo_height = photo_width * photo_height →
  frame_width = 3 := by
  sorry

end frame_width_proof_l201_20100


namespace tim_sleep_hours_l201_20198

/-- The number of hours Tim sleeps each day for the first two days -/
def initial_sleep_hours : ℕ := 6

/-- The number of days Tim sleeps for the initial period -/
def initial_days : ℕ := 2

/-- The total number of days Tim sleeps -/
def total_days : ℕ := 4

/-- The total number of hours Tim sleeps over all days -/
def total_sleep_hours : ℕ := 32

/-- Theorem stating that Tim slept 10 hours each for the next 2 days -/
theorem tim_sleep_hours :
  (total_sleep_hours - initial_sleep_hours * initial_days) / (total_days - initial_days) = 10 :=
sorry

end tim_sleep_hours_l201_20198


namespace unique_monic_quadratic_l201_20146

-- Define a monic polynomial of degree 2
def monicQuadratic (b c : ℝ) : ℝ → ℝ := λ x => x^2 + b*x + c

-- State the theorem
theorem unique_monic_quadratic (g : ℝ → ℝ) :
  (∃ b c : ℝ, ∀ x, g x = monicQuadratic b c x) →  -- g is a monic quadratic polynomial
  g 0 = 6 →                                       -- g(0) = 6
  g 1 = 8 →                                       -- g(1) = 8
  ∀ x, g x = x^2 + x + 6 :=                       -- Conclusion: g(x) = x^2 + x + 6
by
  sorry  -- Proof is omitted as per instructions

end unique_monic_quadratic_l201_20146


namespace same_quotient_remainder_divisible_by_seven_l201_20157

theorem same_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ r : ℕ, 1 ≤ r ∧ r ≤ 6 ∧ n = 8 * r} = {8, 16, 24, 32, 40, 48} := by
sorry

end same_quotient_remainder_divisible_by_seven_l201_20157


namespace parallel_transitivity_l201_20132

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (notContained : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity
  (a b : Line) (α : Plane)
  (h1 : parallelLine a b)
  (h2 : parallelLinePlane a α)
  (h3 : notContained b α) :
  parallelLinePlane b α :=
sorry

end parallel_transitivity_l201_20132


namespace popton_bus_toes_l201_20153

/-- Represents a race of beings on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of toes for an individual of a given race -/
def toesPerIndividual (r : Race) : ℕ :=
  (hands r) * (toesPerHand r)

/-- Number of students of each race on the bus -/
def studentsOnBus (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes on the Popton school bus -/
def totalToesOnBus : ℕ :=
  (toesPerIndividual Race.Hoopit) * (studentsOnBus Race.Hoopit) +
  (toesPerIndividual Race.Neglart) * (studentsOnBus Race.Neglart)

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem popton_bus_toes : totalToesOnBus = 164 := by
  sorry

end popton_bus_toes_l201_20153


namespace calculate_expression_l201_20199

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - (2023 : ℝ)^0 = -2 := by
  sorry

end calculate_expression_l201_20199


namespace qinghai_lake_travel_solution_l201_20117

/-- Represents the travel plans and costs for two teams visiting Qinghai Lake. -/
structure TravelPlan where
  distanceA : ℕ  -- Distance for Team A in km
  distanceB : ℕ  -- Distance for Team B in km
  daysA : ℕ      -- Number of days for Team A
  daysB : ℕ      -- Number of days for Team B
  costA : ℕ      -- Daily cost per person for Team A in yuan
  costB : ℕ      -- Daily cost per person for Team B in yuan
  peopleA : ℕ    -- Number of people in Team A
  peopleB : ℕ    -- Number of people in Team B
  m : ℕ          -- Additional people joining Team A

/-- The theorem stating the solution to the Qinghai Lake travel problem. -/
theorem qinghai_lake_travel_solution (plan : TravelPlan) : 
  plan.distanceA = 2700 ∧ 
  plan.distanceB = 1800 ∧
  plan.distanceA / plan.daysA = 2 * (plan.distanceB / plan.daysB) ∧
  plan.daysA + 1 = plan.daysB ∧
  plan.costA = 200 ∧
  plan.costB = 150 ∧
  plan.peopleA = 10 ∧
  plan.peopleB = 8 ∧
  (plan.costA - 30) * (plan.peopleA + plan.m) * plan.daysA + plan.costB * plan.peopleB * plan.daysB = 
    (plan.costA * plan.peopleA * plan.daysA + plan.costB * plan.peopleB * plan.daysB) * 120 / 100 →
  plan.daysA = 3 ∧ plan.daysB = 4 ∧ plan.m = 6 := by
  sorry

end qinghai_lake_travel_solution_l201_20117


namespace curve_represents_two_points_l201_20128

theorem curve_represents_two_points :
  ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
  (∀ (x y : ℝ), ((x - y)^2 + (x*y - 1)^2 = 0) ↔ (x, y) = p1 ∨ (x, y) = p2) :=
sorry

end curve_represents_two_points_l201_20128


namespace probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l201_20175

/-- The probability of getting at least two successes in three independent trials 
    with a 50% success rate for each trial -/
theorem probability_at_least_two_successes : ℝ := by sorry

/-- The probability of selecting exactly one item from a specific pair 
    when selecting 2 out of 4 items -/
theorem probability_one_from_pair : ℝ := by sorry

/-- Proof of the probabilities for the contest scenario -/
theorem contest_probabilities : 
  probability_at_least_two_successes = 1/2 ∧ 
  probability_one_from_pair = 2/3 := by sorry

end probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l201_20175


namespace alyssa_cherries_cost_l201_20145

/-- The amount Alyssa paid for cherries -/
def cherries_cost (total_spent grapes_cost : ℚ) : ℚ :=
  total_spent - grapes_cost

/-- Proof that Alyssa paid $9.85 for cherries -/
theorem alyssa_cherries_cost :
  let total_spent : ℚ := 21.93
  let grapes_cost : ℚ := 12.08
  cherries_cost total_spent grapes_cost = 9.85 := by
  sorry

#eval cherries_cost 21.93 12.08

end alyssa_cherries_cost_l201_20145


namespace same_school_probability_same_school_probability_proof_l201_20166

/-- The probability of selecting two teachers from the same school when randomly choosing
    two teachers out of three from School A and three from School B. -/
theorem same_school_probability : ℚ :=
  let total_teachers : ℕ := 6
  let teachers_per_school : ℕ := 3
  let selected_teachers : ℕ := 2

  2 / 5

/-- Proof that the probability of selecting two teachers from the same school is 2/5. -/
theorem same_school_probability_proof :
  same_school_probability = 2 / 5 := by
  sorry

end same_school_probability_same_school_probability_proof_l201_20166


namespace min_omega_for_coinciding_symmetry_axes_l201_20119

/-- Given a sinusoidal function y = 2sin(ωx + π/3) where ω > 0, 
    if the graph is shifted left and right by π/3 units and 
    the axes of symmetry of the resulting graphs coincide, 
    then the minimum value of ω is 3/2. -/
theorem min_omega_for_coinciding_symmetry_axes (ω : ℝ) : 
  ω > 0 → 
  (∀ x : ℝ, ∃ y : ℝ, y = 2 * Real.sin (ω * x + π / 3)) →
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = 2 * Real.sin (ω * (x + π / 3) + π / 3) ∧
    y₂ = 2 * Real.sin (ω * (x - π / 3) + π / 3)) →
  (∃ k : ℤ, ω * (π / 3) = k * π) →
  ω ≥ 3 / 2 :=
by sorry

end min_omega_for_coinciding_symmetry_axes_l201_20119


namespace quadratic_roots_l201_20122

/-- A quadratic function f(x) = ax² - 12ax + 36a - 5 has roots at x = 4 and x = 8 -/
theorem quadratic_roots (a : ℝ) : 
  (∀ x ∈ Set.Ioo 4 5, a * x^2 - 12 * a * x + 36 * a - 5 < 0) →
  (∀ x ∈ Set.Ioo 8 9, a * x^2 - 12 * a * x + 36 * a - 5 > 0) →
  a = 5/4 := by
sorry


end quadratic_roots_l201_20122


namespace detergent_in_altered_solution_l201_20133

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let new_bleach := original.bleach * 3
  let new_detergent := original.detergent
  let new_water := original.water * 2
  let total_parts := new_bleach + new_detergent + new_water
  (new_detergent / total_parts) * water_amount

/-- Theorem stating the amount of detergent in the altered solution -/
theorem detergent_in_altered_solution 
  (original : CleaningSolution)
  (h1 : original.bleach = 2)
  (h2 : original.detergent = 25)
  (h3 : original.water = 100)
  (water_amount : ℚ)
  (h4 : water_amount = 300) :
  altered_detergent_amount original water_amount = 37.5 := by
  sorry

end detergent_in_altered_solution_l201_20133


namespace number_puzzle_l201_20115

theorem number_puzzle : ∃! x : ℚ, x / 5 + 6 = x / 4 - 6 := by
  sorry

end number_puzzle_l201_20115


namespace dragon_invincible_l201_20121

-- Define the possible head-cutting operations
inductive CutOperation
| cut13 : CutOperation
| cut17 : CutOperation
| cut6 : CutOperation

-- Define the state of the dragon
structure DragonState :=
  (heads : ℕ)

-- Define the rules for head regeneration
def regenerate (s : DragonState) : DragonState :=
  match s.heads with
  | 1 => ⟨8⟩  -- 1 + 7 regenerated
  | 2 => ⟨13⟩ -- 2 + 11 regenerated
  | 3 => ⟨12⟩ -- 3 + 9 regenerated
  | n => s

-- Define a single step of the process (cutting and potential regeneration)
def step (s : DragonState) (op : CutOperation) : DragonState :=
  let s' := match op with
    | CutOperation.cut13 => ⟨s.heads - min s.heads 13⟩
    | CutOperation.cut17 => ⟨s.heads - min s.heads 17⟩
    | CutOperation.cut6 => ⟨s.heads - min s.heads 6⟩
  regenerate s'

-- Define the theorem
theorem dragon_invincible :
  ∀ (ops : List CutOperation),
    let final_state := ops.foldl step ⟨100⟩
    final_state.heads > 0 ∨ final_state.heads ≤ 5 :=
sorry

end dragon_invincible_l201_20121


namespace simplify_expression_l201_20130

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end simplify_expression_l201_20130


namespace product_and_power_constraint_l201_20148

theorem product_and_power_constraint (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h_product : a * b * c = 10)
  (h_power : a^(Real.log a) * b^(Real.log b) * c^(Real.log c) ≥ 10) :
  (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) :=
by sorry

end product_and_power_constraint_l201_20148


namespace cooking_and_yoga_count_l201_20176

/-- Represents the number of people in various curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people studying both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 18)
  (h3 : g.weaving = 10)
  (h4 : g.cookingOnly = 4)
  (h5 : g.allCurriculums = 4)
  (h6 : g.cookingAndWeaving = 5) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 9 :=
by sorry

end cooking_and_yoga_count_l201_20176


namespace luncheon_invitees_l201_20171

theorem luncheon_invitees (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) :
  no_shows = 10 →
  table_capacity = 7 →
  tables_needed = 2 →
  no_shows + (tables_needed * table_capacity) = 24 := by
sorry

end luncheon_invitees_l201_20171


namespace convex_polygon_sides_l201_20180

theorem convex_polygon_sides (n : ℕ) (a₁ : ℝ) (d : ℝ) : 
  n > 2 →
  a₁ = 120 →
  d = 5 →
  (n - 2) * 180 = (2 * a₁ + (n - 1) * d) * n / 2 →
  (∀ k : ℕ, k ≤ n → a₁ + (k - 1) * d < 180) →
  n = 9 := by
sorry

end convex_polygon_sides_l201_20180


namespace root_of_quadratic_l201_20147

theorem root_of_quadratic (a : ℝ) : (2 : ℝ)^2 + a * 2 - 3 * a = 0 → a = 4 := by
  sorry

end root_of_quadratic_l201_20147


namespace lunch_slices_count_l201_20184

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := sorry

/-- The total number of slices of pie served today -/
def total_slices : ℕ := 12

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- Theorem stating that the number of slices served during lunch today is 7 -/
theorem lunch_slices_count : lunch_slices = total_slices - dinner_slices := by sorry

end lunch_slices_count_l201_20184


namespace billy_total_tickets_l201_20111

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def ticket_cost_per_ride : ℕ := 5

/-- Theorem: The total number of tickets Billy used is 50 -/
theorem billy_total_tickets : 
  (ferris_rides + bumper_rides) * ticket_cost_per_ride = 50 := by
  sorry

end billy_total_tickets_l201_20111


namespace cube_of_square_of_third_smallest_prime_l201_20144

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem cube_of_square_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 2 ^ 3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l201_20144


namespace sum_removal_proof_l201_20177

theorem sum_removal_proof : 
  let original_sum := (1 : ℚ) / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18
  let removed_sum := (1 : ℚ) / 12 + 1 / 15
  original_sum - removed_sum = 2 / 3 := by
  sorry

end sum_removal_proof_l201_20177


namespace simplify_expression_l201_20161

theorem simplify_expression : (81^(1/2) - 144^(1/2)) / 3^(1/2) = -Real.sqrt 3 := by
  sorry

end simplify_expression_l201_20161


namespace shoes_selection_ways_l201_20195

/-- The number of pairs of distinct shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes taken out -/
def shoes_taken : ℕ := 4

/-- The number of ways to select 4 shoes from 10 pairs such that
    exactly two form a pair and the other two don't form a pair -/
def ways_to_select : ℕ := 1440

/-- Theorem stating the number of ways to select 4 shoes from 10 pairs
    such that exactly two form a pair and the other two don't form a pair -/
theorem shoes_selection_ways (n : ℕ) (h : n = total_pairs) :
  ways_to_select = Nat.choose n 1 * Nat.choose (n - 1) 2 * 2^2 :=
sorry

end shoes_selection_ways_l201_20195


namespace square_area_with_four_circles_l201_20134

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end square_area_with_four_circles_l201_20134


namespace rectangle_sides_l201_20168

theorem rectangle_sides (x y : ℚ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  x = 7 / 2 ∧ y = 14 / 3 := by
  sorry

end rectangle_sides_l201_20168


namespace pure_imaginary_complex_number_l201_20155

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2)
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 :=
by
  sorry

end pure_imaginary_complex_number_l201_20155


namespace parabola_tangent_line_l201_20156

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + 3 = 2 * x + 1

/-- If the parabola y = ax^2 + 3 is tangent to the line y = 2x + 1, then a = 1/2. -/
theorem parabola_tangent_line (a : ℝ) : is_tangent a → a = 1/2 := by
  sorry

end parabola_tangent_line_l201_20156


namespace captain_birth_year_is_1938_l201_20131

/-- Represents the ages of the crew members -/
structure CrewAges where
  sailor : ℕ
  cook : ℕ
  engineer : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : CrewAges) : Prop :=
  Odd ages.cook ∧
  ¬Odd ages.sailor ∧
  ¬Odd ages.engineer ∧
  ages.engineer = ages.sailor + 4 ∧
  ages.cook = 3 * (ages.sailor / 2) ∧
  ages.sailor = 2 * (ages.sailor / 2)

/-- The captain's birth year is the LCM of the crew's ages -/
def captainBirthYear (ages : CrewAges) : ℕ :=
  Nat.lcm ages.sailor (Nat.lcm ages.cook ages.engineer)

/-- The main theorem stating that the captain's birth year is 1938 -/
theorem captain_birth_year_is_1938 :
  ∃ ages : CrewAges, satisfiesConditions ages ∧ captainBirthYear ages = 1938 :=
sorry

end captain_birth_year_is_1938_l201_20131


namespace jovana_shell_weight_l201_20162

/-- Proves that the total weight of shells in Jovana's bucket is approximately 11.29 pounds -/
theorem jovana_shell_weight (initial_weight : ℝ) (large_shell_weight : ℝ) (additional_weight : ℝ) 
  (conversion_rate : ℝ) (h1 : initial_weight = 5.25) (h2 : large_shell_weight = 700) 
  (h3 : additional_weight = 4.5) (h4 : conversion_rate = 453.592) : 
  ∃ (total_weight : ℝ), abs (total_weight - 11.29) < 0.01 ∧ 
  total_weight = initial_weight + (large_shell_weight / conversion_rate) + additional_weight :=
sorry

end jovana_shell_weight_l201_20162


namespace spinner_probability_l201_20193

def spinner_numbers : List ℕ := [4, 6, 7, 11, 12, 13, 17, 18]

def total_sections : ℕ := 8

def favorable_outcomes : ℕ := (spinner_numbers.filter (λ x => x > 10)).length

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_sections = 5 / 8 := by
  sorry

end spinner_probability_l201_20193


namespace solution_set_quadratic_inequality_l201_20150

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
by sorry

end solution_set_quadratic_inequality_l201_20150


namespace subtraction_result_l201_20188

theorem subtraction_result : (1000000000000 : ℕ) - 777777777777 = 222222222223 := by
  sorry

end subtraction_result_l201_20188


namespace polynomial_coefficient_values_l201_20106

theorem polynomial_coefficient_values (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₄ = 16 ∧ a₅ = 4 := by
sorry

end polynomial_coefficient_values_l201_20106


namespace mixed_number_sum_l201_20197

theorem mixed_number_sum : 
  (2 + 1/10) + (3 + 11/100) + (4 + 111/1000) = 9321/1000 := by sorry

end mixed_number_sum_l201_20197


namespace snow_probability_l201_20181

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end snow_probability_l201_20181


namespace alpha_minus_beta_eq_pi_fourth_l201_20151

theorem alpha_minus_beta_eq_pi_fourth 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (π/4) (π/2)) 
  (h2 : β ∈ Set.Icc (π/4) (π/2)) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 2 * Real.cos β) : 
  α - β = π/4 := by
sorry

end alpha_minus_beta_eq_pi_fourth_l201_20151


namespace max_gcd_2015_l201_20112

theorem max_gcd_2015 (x y : ℤ) (h : Int.gcd x y = 1) :
  (∃ a b : ℤ, Int.gcd (a + 2015 * b) (b + 2015 * a) = 4060224) ∧
  (∀ c d : ℤ, Int.gcd (c + 2015 * d) (d + 2015 * c) ≤ 4060224) := by
sorry

end max_gcd_2015_l201_20112


namespace no_common_integer_solutions_l201_20127

theorem no_common_integer_solutions : ¬∃ y : ℤ, (-3 * y ≥ y + 9) ∧ (2 * y ≥ 14) ∧ (-4 * y ≥ 2 * y + 21) := by
  sorry

end no_common_integer_solutions_l201_20127


namespace parabola_directrix_l201_20105

/-- The directrix of a parabola y = ax^2 where a < 0 -/
def directrix_equation (a : ℝ) : ℝ → Prop :=
  fun y => y = -1 / (4 * a)

/-- The parabola equation y = ax^2 -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y = a * x^2

theorem parabola_directrix (a : ℝ) (h : a < 0) :
  ∃ y, directrix_equation a y ∧
    ∀ x, parabola_equation a x y →
      ∃ p, p > 0 ∧ (x^2 = 4 * p * y) ∧ (y = -p) :=
by sorry

end parabola_directrix_l201_20105


namespace betty_beads_l201_20141

/-- Given a ratio of red to blue beads and a number of red beads, 
    calculate the number of blue beads -/
def blue_beads (red_ratio blue_ratio red_count : ℕ) : ℕ :=
  (blue_ratio * red_count) / red_ratio

/-- Theorem stating that given 3 red beads for every 2 blue beads,
    and 30 red beads in total, there are 20 blue beads -/
theorem betty_beads : blue_beads 3 2 30 = 20 := by
  sorry

end betty_beads_l201_20141


namespace greatest_ba_value_l201_20187

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem greatest_ba_value (a b : ℕ) :
  is_prime a →
  is_prime b →
  a < 10 →
  b < 10 →
  is_divisible_by (110 * 10 + a * 10 + b) 55 →
  (∀ a' b' : ℕ, 
    is_prime a' → 
    is_prime b' → 
    a' < 10 → 
    b' < 10 → 
    is_divisible_by (110 * 10 + a' * 10 + b') 55 → 
    b * a ≥ b' * a') →
  b * a = 15 := by
sorry

end greatest_ba_value_l201_20187


namespace polynomial_real_root_iff_b_in_set_l201_20142

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^5 + b*x^4 - x^3 + b*x^2 - x + b

/-- The set of b values for which the polynomial has at least one real root -/
def valid_b_set : Set ℝ := Set.Iic (-1) ∪ Set.Ici 1

theorem polynomial_real_root_iff_b_in_set (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ∈ valid_b_set := by sorry

end polynomial_real_root_iff_b_in_set_l201_20142


namespace circle_passes_through_points_l201_20140

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation O.1 O.2 ∧
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 := by
  sorry

end circle_passes_through_points_l201_20140


namespace cloth_sold_proof_l201_20178

/-- Represents the profit per meter of cloth in Rs. -/
def profit_per_meter : ℕ := 35

/-- Represents the total profit earned in Rs. -/
def total_profit : ℕ := 1400

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := total_profit / profit_per_meter

theorem cloth_sold_proof : meters_sold = 40 := by
  sorry

end cloth_sold_proof_l201_20178


namespace cages_needed_proof_l201_20192

def initial_gerbils : ℕ := 150
def sold_gerbils : ℕ := 98

theorem cages_needed_proof :
  initial_gerbils - sold_gerbils = 52 :=
by sorry

end cages_needed_proof_l201_20192


namespace remaining_balloons_l201_20109

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown starts with -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- Theorem: The clown is left with 21 balloons after selling to boys and girls -/
theorem remaining_balloons :
  initial_dozens * dozen - (boys + girls) = 21 := by
  sorry

end remaining_balloons_l201_20109


namespace mutually_exclusive_pairs_count_l201_20118

-- Define the total number of volunteers
def total_volunteers : ℕ := 7

-- Define the number of male and female volunteers
def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3

-- Define the number of selected volunteers
def selected_volunteers : ℕ := 2

-- Define the events
def event1 : Prop := False  -- Logically inconsistent event
def event2 : Prop := True   -- At least 1 female and all females
def event3 : Prop := True   -- At least 1 male and at least 1 female
def event4 : Prop := True   -- At least 1 female and all males

-- Define a function to count mutually exclusive pairs
def count_mutually_exclusive_pairs (events : List Prop) : ℕ := 1

-- Theorem statement
theorem mutually_exclusive_pairs_count :
  count_mutually_exclusive_pairs [event1, event2, event3, event4] = 1 := by
  sorry

end mutually_exclusive_pairs_count_l201_20118


namespace range_of_a_l201_20126

-- Define the function f
def f (x : ℝ) : ℝ := -2*x^5 - x^3 - 7*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : f (a^2) + f (a-2) > 4 → a ∈ Set.Ioo (-2) 1 := by
  sorry

end range_of_a_l201_20126


namespace least_2310_divisors_form_l201_20116

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is not divisible by 10 -/
def not_div_by_ten (m : ℕ) : Prop := ¬(10 ∣ m)

/-- The least positive integer with exactly 2310 distinct positive divisors -/
def least_with_2310_divisors : ℕ := sorry

theorem least_2310_divisors_form :
  ∃ (m k : ℕ), 
    least_with_2310_divisors = m * 10^k ∧ 
    not_div_by_ten m ∧ 
    m + k = 10 := by sorry

end least_2310_divisors_form_l201_20116


namespace total_weekly_sleep_is_123_l201_20189

/-- Represents the type of day (odd or even) -/
inductive DayType
| odd
| even

/-- Calculates the sleep time for a cougar based on the day type -/
def cougarSleep (day : DayType) : ℕ :=
  match day with
  | DayType.odd => 6
  | DayType.even => 4

/-- Calculates the sleep time for a zebra based on the cougar's sleep time -/
def zebraSleep (cougarSleepTime : ℕ) : ℕ :=
  cougarSleepTime + 2

/-- Calculates the sleep time for a lion based on the day type and other animals' sleep times -/
def lionSleep (day : DayType) (cougarSleepTime zebraSleepTime : ℕ) : ℕ :=
  match day with
  | DayType.odd => cougarSleepTime + 1
  | DayType.even => zebraSleepTime - 3

/-- Calculates the total weekly sleep time for all three animals -/
def totalWeeklySleep : ℕ :=
  let oddDays := 4
  let evenDays := 3
  let cougarTotal := oddDays * cougarSleep DayType.odd + evenDays * cougarSleep DayType.even
  let zebraTotal := oddDays * zebraSleep (cougarSleep DayType.odd) + evenDays * zebraSleep (cougarSleep DayType.even)
  let lionTotal := oddDays * lionSleep DayType.odd (cougarSleep DayType.odd) (zebraSleep (cougarSleep DayType.odd)) +
                   evenDays * lionSleep DayType.even (cougarSleep DayType.even) (zebraSleep (cougarSleep DayType.even))
  cougarTotal + zebraTotal + lionTotal

theorem total_weekly_sleep_is_123 : totalWeeklySleep = 123 := by
  sorry

end total_weekly_sleep_is_123_l201_20189


namespace max_gcd_13n_plus_4_7n_plus_3_l201_20185

theorem max_gcd_13n_plus_4_7n_plus_3 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (13 * n + 4) (7 * n + 3) ≤ k ∧
  ∃ (m : ℕ+), Nat.gcd (13 * m + 4) (7 * m + 3) = k ∧
  k = 11 :=
sorry

end max_gcd_13n_plus_4_7n_plus_3_l201_20185


namespace sum_binary_digits_365_l201_20107

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the digits in the binary representation of 365 is 6 -/
theorem sum_binary_digits_365 : sumBinaryDigits 365 = 6 := by
  sorry

end sum_binary_digits_365_l201_20107


namespace theater_ticket_sales_l201_20182

theorem theater_ticket_sales (adult_price child_price : ℕ) 
  (total_tickets adult_tickets child_tickets : ℕ) : 
  adult_price = 12 → 
  child_price = 4 → 
  total_tickets = 130 → 
  adult_tickets = 90 → 
  child_tickets = 40 → 
  adult_price * adult_tickets + child_price * child_tickets = 1240 := by
  sorry

end theater_ticket_sales_l201_20182


namespace rectangular_plot_length_l201_20183

/-- Given a rectangular plot with the following properties:
  - The length is 20 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Then the length of the plot is 60 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 20 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 60 := by
  sorry

end rectangular_plot_length_l201_20183


namespace power_function_through_point_value_l201_20160

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Theorem statement
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 16 →
  f (Real.sqrt 3) = 9 := by
sorry

end power_function_through_point_value_l201_20160


namespace multiple_with_all_digits_l201_20170

/-- For any integer n, there exists a multiple m of n whose decimal representation
    contains each digit from 0 to 9 at least once. -/
theorem multiple_with_all_digits (n : ℤ) : ∃ m : ℤ,
  (n ∣ m) ∧ (∀ d : ℕ, d < 10 → ∃ k : ℕ, (m.natAbs / 10^k) % 10 = d) := by
  sorry

end multiple_with_all_digits_l201_20170


namespace geometric_series_sum_l201_20135

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 2
  let r : ℚ := 2/5
  let n : ℕ := 5
  geometric_sum a r n = 10310/3125 := by
sorry

end geometric_series_sum_l201_20135


namespace symmetric_function_inequality_l201_20136

/-- A function that is symmetric about x = 1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The derivative condition for x < 1 -/
def DerivativeCondition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, x < 1 → 2 * f x + (x - 1) * f' x < 0

theorem symmetric_function_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_symmetric : SymmetricAboutOne f)
  (h_derivative : DerivativeCondition f f') :
  {x : ℝ | (x + 1)^2 * f (x + 2) > f 2} = Set.Ioo (-2) 0 := by
  sorry

end symmetric_function_inequality_l201_20136


namespace max_pencils_purchased_l201_20154

/-- Given a pencil price of 25 cents and $50 available, 
    prove that the maximum number of pencils that can be purchased is 200. -/
theorem max_pencils_purchased (pencil_price : ℕ) (available_money : ℕ) :
  pencil_price = 25 →
  available_money = 5000 →
  (∀ n : ℕ, n * pencil_price ≤ available_money → n ≤ 200) ∧
  200 * pencil_price ≤ available_money :=
by
  sorry

#check max_pencils_purchased

end max_pencils_purchased_l201_20154


namespace total_distance_biking_and_jogging_l201_20169

theorem total_distance_biking_and_jogging 
  (total_time : ℝ) 
  (biking_time : ℝ) 
  (biking_rate : ℝ) 
  (jogging_time : ℝ) 
  (jogging_rate : ℝ) 
  (h1 : total_time = 1.75) -- 1 hour and 45 minutes
  (h2 : biking_time = 1) -- 60 minutes in hours
  (h3 : biking_rate = 12)
  (h4 : jogging_time = 0.75) -- 45 minutes in hours
  (h5 : jogging_rate = 6) : 
  biking_rate * biking_time + jogging_rate * jogging_time = 16.5 := by
  sorry

#check total_distance_biking_and_jogging

end total_distance_biking_and_jogging_l201_20169


namespace sum_a_b_equals_eleven_l201_20164

theorem sum_a_b_equals_eleven (a b c d : ℝ) 
  (h1 : b + c = 9)
  (h2 : c + d = 3)
  (h3 : a + d = 5) :
  a + b = 11 := by
sorry

end sum_a_b_equals_eleven_l201_20164


namespace dj_snake_engagement_treats_value_l201_20123

/-- The total value of treats received by DJ Snake on his engagement day -/
def total_value (hotel_nights : ℕ) (hotel_price_per_night : ℕ) (car_value : ℕ) : ℕ :=
  hotel_nights * hotel_price_per_night + car_value + 4 * car_value

/-- Theorem stating the total value of treats received by DJ Snake on his engagement day -/
theorem dj_snake_engagement_treats_value :
  total_value 2 4000 30000 = 158000 := by
  sorry

end dj_snake_engagement_treats_value_l201_20123


namespace marion_has_23_paperclips_l201_20120

-- Define the variables
def x : ℚ := 30
def y : ℚ := 7

-- Define Yun's remaining paperclips
def yun_remaining : ℚ := 2/5 * x

-- Define Marion's paperclips
def marion_paperclips : ℚ := 4/3 * yun_remaining + y

-- Theorem to prove
theorem marion_has_23_paperclips : marion_paperclips = 23 := by
  sorry

end marion_has_23_paperclips_l201_20120


namespace sum_of_reciprocals_of_roots_l201_20125

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 3 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = 3) := by
sorry

end sum_of_reciprocals_of_roots_l201_20125


namespace expression_evaluation_l201_20102

theorem expression_evaluation :
  ((3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2) / (10^1002) = 28 := by
  sorry

end expression_evaluation_l201_20102


namespace trig_simplification_l201_20190

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end trig_simplification_l201_20190


namespace initial_kittens_count_l201_20174

/-- The number of kittens Tim initially had -/
def initial_kittens : ℕ := 18

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating that the initial number of kittens is equal to
    the sum of kittens given away and kittens left -/
theorem initial_kittens_count :
  initial_kittens = kittens_to_jessica + kittens_to_sara + kittens_left :=
by sorry

end initial_kittens_count_l201_20174


namespace arithmetic_sequence_first_term_l201_20138

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_a8 : a 8 = -(a 9)) :
  a 1 = 15 :=
by
  sorry

end arithmetic_sequence_first_term_l201_20138


namespace students_without_A_l201_20114

def class_size : ℕ := 35
def history_A : ℕ := 10
def math_A : ℕ := 15
def both_A : ℕ := 5

theorem students_without_A : 
  class_size - (history_A + math_A - both_A) = 15 := by sorry

end students_without_A_l201_20114


namespace root_of_polynomial_l201_20104

theorem root_of_polynomial (x : ℝ) : x^5 - 2*x^4 - x^2 + 2*x - 3 = 0 ↔ x = 3 := by
  sorry

end root_of_polynomial_l201_20104


namespace coefficient_c_nonzero_l201_20139

def Q (a' b' c' d' x : ℝ) : ℝ := x^4 + a'*x^3 + b'*x^2 + c'*x + d'

theorem coefficient_c_nonzero 
  (a' b' c' d' : ℝ) 
  (h1 : ∃ u v w : ℝ, u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0)
  (h2 : ∀ x : ℝ, Q a' b' c' d' x = x * (x - u) * (x - v) * (x - w))
  (h3 : d' = 0) :
  c' ≠ 0 := by
sorry

end coefficient_c_nonzero_l201_20139


namespace complex_equation_real_l201_20194

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_real (a : ℝ) : 
  (((2 * a : ℂ) / (1 + i) + 1 + i).im = 0) → a = 1 := by
  sorry

end complex_equation_real_l201_20194


namespace fill_container_l201_20163

/-- The capacity of a standard jar in milliliters -/
def standard_jar_capacity : ℕ := 60

/-- The capacity of the big container in milliliters -/
def big_container_capacity : ℕ := 840

/-- The minimum number of standard jars needed to fill the big container -/
def min_jars_needed : ℕ := 14

theorem fill_container :
  min_jars_needed = (big_container_capacity + standard_jar_capacity - 1) / standard_jar_capacity :=
by sorry

end fill_container_l201_20163


namespace consecutive_numbers_problem_l201_20149

/-- Proves that y = 3 given the conditions of the problem -/
theorem consecutive_numbers_problem (x y z : ℤ) 
  (h1 : x = z + 2) 
  (h2 : y = z + 1) 
  (h3 : x > y ∧ y > z) 
  (h4 : 2*x + 3*y + 3*z = 5*y + 8) 
  (h5 : z = 2) : 
  y = 3 := by
sorry

end consecutive_numbers_problem_l201_20149


namespace xy_value_l201_20137

theorem xy_value (x y : ℝ) (h1 : x + y = 2) (h2 : x^2 * y^3 + y^2 * x^3 = 32) : x * y = 2^(5/3) := by
  sorry

end xy_value_l201_20137


namespace min_distance_line_circle_l201_20152

/-- The minimum distance between a point on the line y = m + √3x and the circle (x-√3)² + (y-1)² = 2² is 1 if and only if m = 2 or m = -6. -/
theorem min_distance_line_circle (m : ℝ) : 
  (∃ (x y : ℝ), y = m + Real.sqrt 3 * x ∧ 
   (∀ (x' y' : ℝ), y' = m + Real.sqrt 3 * x' → 
     ((x' - Real.sqrt 3)^2 + (y' - 1)^2 ≥ ((x - Real.sqrt 3)^2 + (y - 1)^2))) ∧
   (x - Real.sqrt 3)^2 + (y - 1)^2 = 5) ↔ 
  (m = 2 ∨ m = -6) :=
sorry

end min_distance_line_circle_l201_20152


namespace jeff_took_six_cans_l201_20172

/-- Represents the number of soda cans in various stages --/
structure SodaCans where
  initial : ℕ
  taken : ℕ
  final : ℕ

/-- Calculates the number of cans Jeff took from Tim --/
def cans_taken (s : SodaCans) : Prop :=
  s.initial - s.taken + (s.initial - s.taken) / 2 = s.final

/-- The main theorem to prove --/
theorem jeff_took_six_cans : ∃ (s : SodaCans), s.initial = 22 ∧ s.final = 24 ∧ s.taken = 6 ∧ cans_taken s := by
  sorry


end jeff_took_six_cans_l201_20172


namespace max_students_per_dentist_l201_20159

theorem max_students_per_dentist (num_dentists num_students min_students_per_dentist : ℕ) 
  (h1 : num_dentists = 12)
  (h2 : num_students = 29)
  (h3 : min_students_per_dentist = 2)
  (h4 : num_dentists * min_students_per_dentist ≤ num_students) :
  ∃ (max_students : ℕ), max_students = 7 ∧ 
  (∀ (d : ℕ), d ≤ num_dentists → ∃ (s : ℕ), s ≤ num_students ∧ s ≤ max_students) ∧
  (∃ (d : ℕ), d ≤ num_dentists ∧ ∃ (s : ℕ), s = max_students) :=
by sorry

end max_students_per_dentist_l201_20159


namespace triangle_angle_measure_l201_20108

theorem triangle_angle_measure (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : Y = 2 * X) : Z = 90 := by
  sorry

end triangle_angle_measure_l201_20108


namespace system_solution_l201_20179

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + x₁*y₁ + y₁ = 1 ∧ y₁^2 + x₁*y₁ + x₁ = 5) ∧
    (x₂^2 + x₂*y₂ + y₂ = 1 ∧ y₂^2 + x₂*y₂ + x₂ = 5) ∧
    x₁ = -1 ∧ y₁ = 3 ∧ x₂ = -1 ∧ y₂ = -2 ∧
    ∀ (x y : ℝ), (x^2 + x*y + y = 1 ∧ y^2 + x*y + x = 5) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end system_solution_l201_20179


namespace constant_difference_function_property_l201_20113

/-- A linear function with constant difference -/
def ConstantDifferenceFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ d : ℝ, f (d + 2) - f d = 6)

theorem constant_difference_function_property (f : ℝ → ℝ) 
  (h : ConstantDifferenceFunction f) : f 1 - f 7 = -18 := by
  sorry

end constant_difference_function_property_l201_20113
