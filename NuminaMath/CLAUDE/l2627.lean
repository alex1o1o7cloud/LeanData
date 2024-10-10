import Mathlib

namespace heroes_on_front_l2627_262793

theorem heroes_on_front (total : ℕ) (on_back : ℕ) (on_front : ℕ) : 
  total = 15 → on_back = 9 → on_front = total - on_back → on_front = 6 := by
  sorry

end heroes_on_front_l2627_262793


namespace no_solution_exists_l2627_262753

theorem no_solution_exists :
  ¬∃ (a b c x y z : ℕ+),
    (a ≥ b) ∧ (b ≥ c) ∧
    (x ≥ y) ∧ (y ≥ z) ∧
    (2 * a + b + 4 * c = 4 * x * y * z) ∧
    (2 * x + y + 4 * z = 4 * a * b * c) :=
by sorry

end no_solution_exists_l2627_262753


namespace submarine_invention_uses_analogy_l2627_262730

/-- Represents the type of reasoning used in an invention process. -/
inductive ReasoningType
  | Analogy
  | Deduction
  | Induction

/-- Represents an invention process. -/
structure Invention where
  name : String
  inspiration : String
  reasoning : ReasoningType

/-- The submarine invention process. -/
def submarineInvention : Invention :=
  { name := "submarine",
    inspiration := "fish shape",
    reasoning := ReasoningType.Analogy }

/-- Theorem stating that the reasoning used in inventing submarines by imitating
    the shape of fish is analogy. -/
theorem submarine_invention_uses_analogy :
  submarineInvention.reasoning = ReasoningType.Analogy := by
  sorry

end submarine_invention_uses_analogy_l2627_262730


namespace jared_popcorn_theorem_l2627_262769

/-- The number of pieces of popcorn in a serving -/
def popcorn_per_serving : ℕ := 30

/-- The number of pieces of popcorn each of Jared's friends can eat -/
def friend_popcorn_consumption : ℕ := 60

/-- The number of Jared's friends -/
def number_of_friends : ℕ := 3

/-- The number of servings Jared should order -/
def servings_ordered : ℕ := 9

/-- The number of pieces of popcorn Jared can eat -/
def jared_popcorn_consumption : ℕ := 
  servings_ordered * popcorn_per_serving - number_of_friends * friend_popcorn_consumption

theorem jared_popcorn_theorem : jared_popcorn_consumption = 90 := by
  sorry

end jared_popcorn_theorem_l2627_262769


namespace coins_missing_l2627_262751

theorem coins_missing (initial : ℚ) : 
  initial > 0 → 
  let lost := (1 : ℚ) / 3 * initial
  let found := (2 : ℚ) / 3 * lost
  let remaining := initial - lost + found
  (initial - remaining) / initial = (1 : ℚ) / 9 := by
  sorry

end coins_missing_l2627_262751


namespace partnership_profit_calculation_l2627_262749

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (investment_a investment_b investment_c c_profit : ℚ) : ℚ :=
  let ratio_sum := (investment_a / 1000) + (investment_b / 1000) + (investment_c / 1000)
  let c_ratio := investment_c / 1000
  (c_profit * ratio_sum) / c_ratio

/-- Theorem stating that given the specified investments and C's profit share, the total profit is approximately 97777.78 -/
theorem partnership_profit_calculation :
  let investment_a : ℚ := 5000
  let investment_b : ℚ := 8000
  let investment_c : ℚ := 9000
  let c_profit : ℚ := 36000
  let total_profit := calculate_total_profit investment_a investment_b investment_c c_profit
  ∃ ε > 0, |total_profit - 97777.78| < ε :=
sorry

end partnership_profit_calculation_l2627_262749


namespace infinite_geometric_series_first_term_l2627_262719

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 27) 
  (h3 : S = a / (1 - r)) : 
  a = 36 := by sorry

end infinite_geometric_series_first_term_l2627_262719


namespace needle_intersection_probability_l2627_262797

/-- Represents the experimental data for needle throwing --/
structure NeedleExperiment where
  trials : ℕ
  intersections : ℕ
  frequency : ℚ

/-- The set of experimental data --/
def experimentalData : List NeedleExperiment := [
  ⟨50, 23, 23/50⟩,
  ⟨100, 48, 12/25⟩,
  ⟨200, 83, 83/200⟩,
  ⟨500, 207, 207/500⟩,
  ⟨1000, 404, 101/250⟩,
  ⟨2000, 802, 401/1000⟩
]

/-- The distance between adjacent lines in cm --/
def lineDistance : ℚ := 5

/-- The length of the needle in cm --/
def needleLength : ℚ := 3

/-- The estimated probability of intersection --/
def estimatedProbability : ℚ := 2/5

/-- Theorem stating that the estimated probability approaches 0.4 as trials increase --/
theorem needle_intersection_probability :
  ∀ ε > 0, ∃ N : ℕ, ∀ e ∈ experimentalData,
    e.trials ≥ N → |e.frequency - estimatedProbability| < ε :=
sorry

end needle_intersection_probability_l2627_262797


namespace james_carrot_sticks_l2627_262715

/-- Given that James originally had 50 carrot sticks, ate 22 before dinner,
    ate 15 after dinner, and gave away 8 during dinner, prove that he has 5 left. -/
theorem james_carrot_sticks (original : ℕ) (eaten_before : ℕ) (eaten_after : ℕ) (given_away : ℕ)
    (h1 : original = 50)
    (h2 : eaten_before = 22)
    (h3 : eaten_after = 15)
    (h4 : given_away = 8) :
    original - eaten_before - eaten_after - given_away = 5 := by
  sorry

end james_carrot_sticks_l2627_262715


namespace complex_roots_imaginary_condition_l2627_262790

theorem complex_roots_imaginary_condition (k : ℝ) (hk : k > 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ z₂ ∧
    12 * z₁^2 - 4 * I * z₁ - k = 0 ∧
    12 * z₂^2 - 4 * I * z₂ - k = 0 ∧
    (z₁.im = 0 ∧ z₂.re = 0) ∨ (z₁.re = 0 ∧ z₂.im = 0)) ↔
  k = (1 : ℝ) / 4 :=
sorry

end complex_roots_imaginary_condition_l2627_262790


namespace polynomial_factorization_l2627_262705

theorem polynomial_factorization (x y : ℝ) : 2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) := by
  sorry

end polynomial_factorization_l2627_262705


namespace negation_equivalence_l2627_262710

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by sorry

end negation_equivalence_l2627_262710


namespace wednesday_to_tuesday_ratio_l2627_262742

/-- The amount of money Max's mom gave him on Tuesday, Wednesday, and Thursday --/
structure MoneyGiven where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def ProblemConditions (m : MoneyGiven) : Prop :=
  m.tuesday = 8 ∧
  ∃ k : ℕ, m.wednesday = k * m.tuesday ∧
  m.thursday = m.wednesday + 9 ∧
  m.thursday = m.tuesday + 41

/-- The theorem to be proved --/
theorem wednesday_to_tuesday_ratio
  (m : MoneyGiven)
  (h : ProblemConditions m) :
  m.wednesday / m.tuesday = 5 := by
sorry

end wednesday_to_tuesday_ratio_l2627_262742


namespace book_profit_rate_l2627_262783

/-- Calculates the overall rate of profit for three books --/
def overall_rate_of_profit (cost_a cost_b cost_c sell_a sell_b sell_c : ℚ) : ℚ :=
  let total_cost := cost_a + cost_b + cost_c
  let total_sell := sell_a + sell_b + sell_c
  (total_sell - total_cost) / total_cost * 100

/-- Theorem: The overall rate of profit for the given book prices is approximately 42.86% --/
theorem book_profit_rate :
  let cost_a : ℚ := 50
  let cost_b : ℚ := 120
  let cost_c : ℚ := 75
  let sell_a : ℚ := 90
  let sell_b : ℚ := 150
  let sell_c : ℚ := 110
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |overall_rate_of_profit cost_a cost_b cost_c sell_a sell_b sell_c - 42.86| < ε :=
by sorry

end book_profit_rate_l2627_262783


namespace problem_solution_l2627_262700

theorem problem_solution (c d : ℝ) 
  (eq1 : 5 + c = 3 - d)
  (eq2 : 3 + d = 8 + c)
  (eq3 : c - d = 2) : 
  5 - c = 5 := by
sorry

end problem_solution_l2627_262700


namespace rectangle_to_tetrahedron_sphere_area_l2627_262718

/-- A rectangle ABCD with sides AB and BC -/
structure Rectangle where
  AB : ℝ
  BC : ℝ

/-- A tetrahedron formed by folding a rectangle along its diagonal -/
structure Tetrahedron where
  base : Rectangle

/-- The surface area of the circumscribed sphere of a tetrahedron -/
def circumscribed_sphere_area (t : Tetrahedron) : ℝ := sorry

theorem rectangle_to_tetrahedron_sphere_area 
  (r : Rectangle) 
  (h1 : r.AB = 8) 
  (h2 : r.BC = 6) : 
  circumscribed_sphere_area (Tetrahedron.mk r) = 100 * Real.pi := by
  sorry

end rectangle_to_tetrahedron_sphere_area_l2627_262718


namespace exactly_three_rainy_days_l2627_262722

/-- The probability of exactly k successes in n independent trials
    with probability p of success on each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

/-- The number of days considered -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

theorem exactly_three_rainy_days :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end exactly_three_rainy_days_l2627_262722


namespace bodhi_cow_count_l2627_262706

/-- Proves that the number of cows is 20 given the conditions of Mr. Bodhi's animal transportation problem -/
theorem bodhi_cow_count :
  let foxes : ℕ := 15
  let zebras : ℕ := 3 * foxes
  let sheep : ℕ := 20
  let total_animals : ℕ := 100
  ∃ cows : ℕ, cows + foxes + zebras + sheep = total_animals ∧ cows = 20 :=
by
  sorry

end bodhi_cow_count_l2627_262706


namespace h_zero_iff_b_eq_seven_fifths_l2627_262701

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- For the function h(x) = 5x - 7, h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by
  sorry

end h_zero_iff_b_eq_seven_fifths_l2627_262701


namespace skew_lines_properties_l2627_262788

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (inPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (planePlaneIntersection : Plane → Plane → Line)

-- Define the theorem
theorem skew_lines_properties
  (α β : Plane)
  (a b c : Line)
  (h1 : inPlane a α)
  (h2 : inPlane b β)
  (h3 : c = planePlaneIntersection α β)
  (h4 : skew a b) :
  (∃ (config : Prop), intersect c a ∧ intersect c b) ∧
  (∃ (lines : ℕ → Line), ∀ (i j : ℕ), i ≠ j → skew (lines i) (lines j)) :=
sorry

end skew_lines_properties_l2627_262788


namespace total_cows_l2627_262785

/-- The number of cows owned by four men given specific conditions -/
theorem total_cows (matthews tyron aaron marovich : ℕ) : 
  matthews = 60 ∧ 
  aaron = 4 * matthews ∧ 
  tyron = matthews - 20 ∧ 
  aaron + matthews + tyron = marovich + 30 → 
  matthews + tyron + aaron + marovich = 650 := by
sorry

end total_cows_l2627_262785


namespace alchemists_less_than_half_l2627_262762

theorem alchemists_less_than_half (k : ℕ) (c a : ℕ) : 
  k > 0 → 
  k = c + a → 
  c > a → 
  a < k / 2 := by
sorry

end alchemists_less_than_half_l2627_262762


namespace min_road_length_on_grid_min_road_length_specific_points_l2627_262792

/-- Represents a point on a grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Calculates the Manhattan distance between two grid points -/
def manhattan_distance (p1 p2 : GridPoint) : ℕ :=
  (Int.natAbs (p1.x - p2.x)) + (Int.natAbs (p1.y - p2.y))

/-- Theorem: Minimum road length on a grid -/
theorem min_road_length_on_grid (square_side_length : ℕ) 
  (A B C : GridPoint) (h : square_side_length = 100) :
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

/-- Main theorem application -/
theorem min_road_length_specific_points :
  let A : GridPoint := ⟨0, 0⟩
  let B : GridPoint := ⟨3, 2⟩
  let C : GridPoint := ⟨4, 3⟩
  let square_side_length : ℕ := 100
  let total_distance := 
    (manhattan_distance A B + manhattan_distance B C + manhattan_distance A C) / 2
  total_distance * square_side_length = 1000 :=
by
  sorry

end min_road_length_on_grid_min_road_length_specific_points_l2627_262792


namespace peg_arrangement_count_l2627_262774

/-- The number of ways to arrange colored pegs on a triangular board. -/
def arrangeColoredPegs (yellow red green blue orange : Nat) : Nat :=
  Nat.factorial yellow * Nat.factorial red * Nat.factorial green * Nat.factorial blue * Nat.factorial orange

/-- Theorem stating the number of arrangements for the given peg counts. -/
theorem peg_arrangement_count :
  arrangeColoredPegs 6 5 4 3 2 = 12441600 := by
  sorry

end peg_arrangement_count_l2627_262774


namespace price_difference_l2627_262744

def original_price : ℝ := 1200

def price_after_increase (p : ℝ) : ℝ := p * 1.1

def price_after_decrease (p : ℝ) : ℝ := p * 0.85

def final_price : ℝ := price_after_decrease (price_after_increase original_price)

theorem price_difference : original_price - final_price = 78 := by
  sorry

end price_difference_l2627_262744


namespace solve_system_for_y_l2627_262727

theorem solve_system_for_y :
  ∃ (x y : ℚ), (3 * x - y = 24 ∧ x + 2 * y = 10) → y = 6/7 := by
  sorry

end solve_system_for_y_l2627_262727


namespace beach_attendance_l2627_262763

theorem beach_attendance (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
  sorry

end beach_attendance_l2627_262763


namespace count_convex_polygons_l2627_262779

/-- A point in the 2D plane with integer coordinates -/
structure Point :=
  (x : ℕ)
  (y : ℕ)

/-- A convex polygon with vertices as a list of points -/
structure ConvexPolygon :=
  (vertices : List Point)
  (is_convex : Bool)

/-- Function to check if a polygon contains the required three consecutive vertices -/
def has_required_vertices (p : ConvexPolygon) : Bool :=
  sorry

/-- Function to count the number of valid convex polygons -/
def count_valid_polygons : ℕ :=
  sorry

/-- The main theorem stating that the count of valid convex polygons is 77 -/
theorem count_convex_polygons :
  count_valid_polygons = 77 :=
sorry

end count_convex_polygons_l2627_262779


namespace birds_in_tree_l2627_262736

theorem birds_in_tree (initial_birds : Real) (birds_flew_away : Real) 
  (h1 : initial_birds = 42.5)
  (h2 : birds_flew_away = 27.3) : 
  initial_birds - birds_flew_away = 15.2 := by
  sorry

end birds_in_tree_l2627_262736


namespace blake_change_l2627_262799

-- Define the quantities and prices
def num_lollipops : ℕ := 4
def num_chocolate_packs : ℕ := 6
def lollipop_price : ℕ := 2
def num_bills : ℕ := 6
def bill_value : ℕ := 10

-- Define the relationship between chocolate and lollipop prices
def chocolate_pack_price : ℕ := 4 * lollipop_price

-- Calculate the total cost
def total_cost : ℕ := num_lollipops * lollipop_price + num_chocolate_packs * chocolate_pack_price

-- Calculate the amount given
def amount_given : ℕ := num_bills * bill_value

-- Theorem to prove
theorem blake_change : amount_given - total_cost = 4 := by
  sorry

end blake_change_l2627_262799


namespace circle_radius_from_intersecting_line_l2627_262755

/-- Given a line intersecting a circle, prove the radius of the circle --/
theorem circle_radius_from_intersecting_line (r : ℝ) :
  let line := {(x, y) : ℝ × ℝ | x - Real.sqrt 3 * y + 8 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  ∃ (A B : ℝ × ℝ), A ∈ line ∧ A ∈ circle ∧ B ∈ line ∧ B ∈ circle ∧
    ‖A - B‖ = 6 →
  r = 5 := by
sorry

end circle_radius_from_intersecting_line_l2627_262755


namespace divisibility_properties_l2627_262703

theorem divisibility_properties (a : ℤ) : 
  (∃ k : ℤ, a^5 - a = 30 * k) ∧
  (∃ l : ℤ, a^17 - a = 510 * l) ∧
  (∃ m : ℤ, a^11 - a = 66 * m) ∧
  (∃ n : ℤ, a^73 - a = (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) * n) :=
by sorry

end divisibility_properties_l2627_262703


namespace sin_alpha_minus_pi_sixth_l2627_262717

theorem sin_alpha_minus_pi_sixth (α : Real) 
  (h : Real.sin (α + π/6) + 2 * Real.sin (α/2)^2 = 1 - Real.sqrt 2 / 2) : 
  Real.sin (α - π/6) = - Real.sqrt 2 / 2 := by
sorry

end sin_alpha_minus_pi_sixth_l2627_262717


namespace symmetric_line_wrt_y_axis_l2627_262787

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the y-axis
    has the equation 3x + 4y - 5 = 0. -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (3 * (-x) - 4 * y + 5 = 0) ↔ (3 * x + 4 * y - 5 = 0) := by
sorry

end symmetric_line_wrt_y_axis_l2627_262787


namespace geese_count_l2627_262766

theorem geese_count (initial : ℕ) (flew_away : ℕ) (joined : ℕ) 
  (h1 : initial = 372) 
  (h2 : flew_away = 178) 
  (h3 : joined = 57) : 
  initial - flew_away + joined = 251 := by
sorry

end geese_count_l2627_262766


namespace solve_equation_l2627_262741

theorem solve_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (3 / x) + (4 / y) = 1) : x = (3 * y) / (y - 4) := by
  sorry

end solve_equation_l2627_262741


namespace tony_fish_count_l2627_262795

def fish_count (initial : ℕ) (years : ℕ) (yearly_addition : ℕ) (yearly_loss : ℕ) : ℕ :=
  initial + years * (yearly_addition - yearly_loss)

theorem tony_fish_count :
  fish_count 2 5 2 1 = 7 := by
  sorry

end tony_fish_count_l2627_262795


namespace walking_distance_l2627_262728

-- Define the total journey time in hours
def total_time : ℚ := 50 / 60

-- Define the speeds in km/h
def bike_speed : ℚ := 20
def walk_speed : ℚ := 4

-- Define the function to calculate the total time given a distance x
def journey_time (x : ℚ) : ℚ := x / (2 * bike_speed) + x / (2 * walk_speed)

-- State the theorem
theorem walking_distance : 
  ∃ (x : ℚ), journey_time x = total_time ∧ 
  (round (10 * (x / 2)) / 10 : ℚ) = 28 / 10 :=
sorry

end walking_distance_l2627_262728


namespace gilled_mushroom_count_l2627_262770

/-- Represents the number of mushrooms on a log -/
structure MushroomCount where
  total : ℕ
  gilled : ℕ
  spotted : ℕ

/-- Conditions for the mushroom problem -/
def mushroom_conditions (m : MushroomCount) : Prop :=
  m.total = 30 ∧
  m.gilled + m.spotted = m.total ∧
  m.spotted = 9 * m.gilled

/-- Theorem stating the number of gilled mushrooms -/
theorem gilled_mushroom_count (m : MushroomCount) :
  mushroom_conditions m → m.gilled = 3 := by
  sorry

end gilled_mushroom_count_l2627_262770


namespace log_simplification_l2627_262794

theorem log_simplification : (2 * (Real.log 3 / Real.log 4) + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 2 := by
  sorry

end log_simplification_l2627_262794


namespace equation_solution_l2627_262748

theorem equation_solution : 
  ∃! x : ℚ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 := by
  sorry

end equation_solution_l2627_262748


namespace northern_shoe_capital_relocation_l2627_262773

structure XionganNewArea where
  green_ecological : Bool
  innovation_driven : Bool
  coordinated_development : Bool
  open_development : Bool

structure AnxinCounty where
  santai_town : Bool
  traditional_shoemaking : Bool
  northern_shoe_capital : Bool
  nationwide_market : Bool
  adequate_transportation : Bool

def industrial_structure_adjustment (county : AnxinCounty) (new_area : XionganNewArea) : Bool :=
  county.traditional_shoemaking ∧ 
  (new_area.green_ecological ∧ new_area.innovation_driven ∧ 
   new_area.coordinated_development ∧ new_area.open_development)

def relocation_reason (county : AnxinCounty) (new_area : XionganNewArea) : String :=
  if industrial_structure_adjustment county new_area then
    "Industrial structure adjustment"
  else
    "Other reasons"

theorem northern_shoe_capital_relocation 
  (anxin : AnxinCounty) 
  (xiong_an : XionganNewArea) 
  (h1 : anxin.santai_town = true)
  (h2 : anxin.traditional_shoemaking = true)
  (h3 : anxin.northern_shoe_capital = true)
  (h4 : anxin.nationwide_market = true)
  (h5 : anxin.adequate_transportation = true)
  (h6 : xiong_an.green_ecological = true)
  (h7 : xiong_an.innovation_driven = true)
  (h8 : xiong_an.coordinated_development = true)
  (h9 : xiong_an.open_development = true) :
  relocation_reason anxin xiong_an = "Industrial structure adjustment" := by
  sorry

#check northern_shoe_capital_relocation

end northern_shoe_capital_relocation_l2627_262773


namespace unique_function_satisfying_equation_l2627_262743

/-- A function from non-negative reals to non-negative reals. -/
def NonNegativeRealFunction := {f : ℝ → ℝ // ∀ x, 0 ≤ x → 0 ≤ f x}

/-- The functional equation f(f(x)) + f(x) = 6x for all x ≥ 0. -/
def FunctionalEquation (f : NonNegativeRealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x → f.val (f.val x) + f.val x = 6 * x

theorem unique_function_satisfying_equation :
  ∀ f : NonNegativeRealFunction, FunctionalEquation f → 
    ∀ x : ℝ, 0 ≤ x → f.val x = 2 * x :=
sorry

end unique_function_satisfying_equation_l2627_262743


namespace shipment_weight_change_l2627_262786

theorem shipment_weight_change (total_boxes : Nat) (initial_avg : ℝ) (light_weight heavy_weight : ℝ) (removed_boxes : Nat) (new_avg : ℝ) : 
  total_boxes = 30 →
  light_weight = 10 →
  heavy_weight = 20 →
  initial_avg = 18 →
  removed_boxes = 18 →
  new_avg = 15 →
  ∃ (light_count heavy_count : Nat),
    light_count + heavy_count = total_boxes ∧
    (light_count * light_weight + heavy_count * heavy_weight) / total_boxes = initial_avg ∧
    ((light_count * light_weight + (heavy_count - removed_boxes) * heavy_weight) / (total_boxes - removed_boxes) = new_avg) :=
by sorry

end shipment_weight_change_l2627_262786


namespace least_positive_linear_combination_l2627_262726

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (x y : ℤ), 24 * x + 16 * y = n ∨ 24 * x + 16 * y < 0 ∨ 24 * x + 16 * y > n) ∧ 
  (∃ (a b : ℤ), 24 * a + 16 * b = n) :=
by sorry

end least_positive_linear_combination_l2627_262726


namespace equation_solution_l2627_262782

theorem equation_solution :
  ∀ x : ℝ, x^3 - 4*x + 80 ≥ 0 →
  ((x / Real.sqrt 2 + 3 * Real.sqrt 2) * Real.sqrt (x^3 - 4*x + 80) = x^2 + 10*x + 24) ↔
  (x = 4 ∨ x = -1 + Real.sqrt 13) :=
by sorry

end equation_solution_l2627_262782


namespace ball_placement_theorem_l2627_262733

/-- Represents the number of balls and boxes -/
def n : ℕ := 5

/-- Calculates the number of ways to place n balls into n boxes with one empty box -/
def ways_one_empty (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with no empty box and not all numbers matching -/
def ways_no_empty_not_all_match (n : ℕ) : ℕ := sorry

/-- Calculates the number of ways to place n balls into n boxes with one ball in each box and at least two balls matching their box numbers -/
def ways_at_least_two_match (n : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of ways for each scenario with 5 balls and 5 boxes -/
theorem ball_placement_theorem :
  ways_one_empty n = 1200 ∧
  ways_no_empty_not_all_match n = 119 ∧
  ways_at_least_two_match n = 31 := by sorry

end ball_placement_theorem_l2627_262733


namespace smallest_multiple_greater_than_30_l2627_262789

theorem smallest_multiple_greater_than_30 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 5 → k > 0 → n % k = 0) ∧ 
  (n > 30) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 5 ∧ k > 0 ∧ m % k ≠ 0) ∨ m ≤ 30) :=
by sorry

end smallest_multiple_greater_than_30_l2627_262789


namespace larger_number_proof_l2627_262767

theorem larger_number_proof (a b : ℕ) (h1 : Nat.gcd a b = 50) (h2 : Nat.lcm a b = 50 * 13 * 23 * 31) :
  max a b = 463450 := by
  sorry

end larger_number_proof_l2627_262767


namespace quadrilateral_offset_l2627_262756

/-- Given a quadrilateral with one diagonal of 20 cm, one offset of 4 cm, and an area of 90 square cm,
    the length of the other offset is 5 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 4 →
  area = 90 →
  area = (diagonal * (offset1 + 5)) / 2 →
  ∃ (offset2 : ℝ), offset2 = 5 :=
by sorry

end quadrilateral_offset_l2627_262756


namespace september_solution_l2627_262739

/-- A function that maps a month number to its corresponding solution in the equations -/
def month_solution : ℕ → ℝ
| 2 => 2  -- February
| 4 => 4  -- April
| 9 => 9  -- September
| _ => 0  -- Other months (not relevant for this problem)

/-- The theorem stating that the solution of 48 = 5x + 3 corresponds to the 9th month -/
theorem september_solution :
  (month_solution 2 - 1 = 1) ∧
  (18 - 2 * month_solution 4 = 10) ∧
  (48 = 5 * month_solution 9 + 3) := by
  sorry

#check september_solution

end september_solution_l2627_262739


namespace triangle_problem_l2627_262702

theorem triangle_problem (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  a / Real.sin A = b / Real.sin B →
  c / Real.sin C = a / Real.sin A →
  A + B + C = π →
  (A = π/3 ∧ Real.sin (2*B + π/6) = -1/7) := by
sorry


end triangle_problem_l2627_262702


namespace line_intercepts_l2627_262746

/-- The equation of the line -/
def line_equation (x y : ℚ) : Prop := 4 * x + 7 * y = 28

/-- Definition of x-intercept -/
def is_x_intercept (x : ℚ) : Prop := line_equation x 0

/-- Definition of y-intercept -/
def is_y_intercept (y : ℚ) : Prop := line_equation 0 y

/-- Theorem: The x-intercept of the line 4x + 7y = 28 is (7, 0), and the y-intercept is (0, 4) -/
theorem line_intercepts : is_x_intercept 7 ∧ is_y_intercept 4 := by sorry

end line_intercepts_l2627_262746


namespace product_sum_theorem_l2627_262775

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  a*b + b*c + a*c = 72 := by
  sorry

end product_sum_theorem_l2627_262775


namespace geometric_sequence_sum_l2627_262796

/-- A geometric sequence with sum of first n terms S_n = 4^n + a has a = -1 -/
theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, S n = 4^n + a) →
  (∃ r : ℝ, ∀ n : ℕ, S (n + 1) - S n = r * (S n - S (n - 1))) →
  a = -1 :=
by sorry

end geometric_sequence_sum_l2627_262796


namespace weight_of_b_l2627_262791

theorem weight_of_b (wa wb wc : ℝ) (ha hb hc : ℝ) : 
  (wa + wb + wc) / 3 = 45 →
  hb = 2 * ha →
  hc = ha + 20 →
  (wa + wb) / 2 = 40 →
  (wb + wc) / 2 = 43 →
  (ha + hc) / 2 = 155 →
  wb = 31 := by
sorry

end weight_of_b_l2627_262791


namespace repeating_decimal_to_fraction_l2627_262759

/-- Given a repeating decimal 3.565656..., prove it equals 353/99 -/
theorem repeating_decimal_to_fraction : 
  ∀ (x : ℚ), (∃ (n : ℕ), x = 3 + (56 : ℚ) / (10^2 - 1) / 10^n) → x = 353 / 99 := by
  sorry

end repeating_decimal_to_fraction_l2627_262759


namespace parabola_property_l2627_262745

/-- Given a parabola y = ax² + bx + c with the following points:
    (-2, 0), (-1, 4), (0, 6), (1, 6)
    Prove that (a - b + c)(4a + 2b + c) > 0 -/
theorem parabola_property (a b c : ℝ) : 
  (4 * a - 2 * b + c = 0) →
  (a - b + c = 4) →
  (c = 6) →
  (a * 1^2 + b * 1 + c = 6) →
  (a - b + c) * (4 * a + 2 * b + c) > 0 := by
sorry

end parabola_property_l2627_262745


namespace hen_count_is_28_l2627_262764

/-- Represents the count of animals on a farm -/
structure FarmCount where
  hens : ℕ
  cows : ℕ

/-- Checks if the farm count satisfies the given conditions -/
def isValidCount (farm : FarmCount) : Prop :=
  farm.hens + farm.cows = 48 ∧
  2 * farm.hens + 4 * farm.cows = 136

theorem hen_count_is_28 :
  ∃ (farm : FarmCount), isValidCount farm ∧ farm.hens = 28 :=
by
  sorry

#check hen_count_is_28

end hen_count_is_28_l2627_262764


namespace perpendicular_condition_l2627_262750

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x1 y1 x2 y2 : ℝ, line1 m x1 y1 → line2 m x2 y2 →
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 →
    (x2 - x1) * (y2 - y1) = 0

-- State the theorem
theorem perpendicular_condition (m : ℝ) :
  (m = -1 → perpendicular m) ∧ ¬(perpendicular m → m = -1) :=
sorry

end perpendicular_condition_l2627_262750


namespace inequality_proof_l2627_262713

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end inequality_proof_l2627_262713


namespace parkway_soccer_players_l2627_262771

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) 
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : girls_not_playing = 135)
  (h4 : (86 : ℚ) / 100 * (total_students - (total_students - boys - girls_not_playing)) = boys - (total_students - boys - girls_not_playing)) :
  total_students - (total_students - boys - girls_not_playing) = 250 := by
  sorry

end parkway_soccer_players_l2627_262771


namespace calculate_expression_l2627_262740

theorem calculate_expression (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^2 = 12 * x^4 * y^3 := by
  sorry

end calculate_expression_l2627_262740


namespace range_of_k_l2627_262754

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8 = 0

-- Define the line
def line (k x : ℝ) : ℝ := 2*k*x - 2

-- Define the condition for a point on the line to be a valid center
def valid_center (k x : ℝ) : Prop :=
  ∃ (y : ℝ), y = line k x ∧ 
  ∃ (x' y' : ℝ), circle_C x' y' ∧ (x' - x)^2 + (y' - y)^2 ≤ 4

-- Theorem statement
theorem range_of_k :
  ∀ k : ℝ, (∃ x : ℝ, valid_center k x) ↔ 0 ≤ k ∧ k ≤ 6/5 :=
by sorry

end range_of_k_l2627_262754


namespace count_valid_assignments_l2627_262735

/-- Represents a student --/
inductive Student : Type
| jia : Student
| other : Fin 4 → Student

/-- Represents a dormitory --/
inductive Dormitory : Type
| A : Dormitory
| B : Dormitory
| C : Dormitory

/-- An assignment of students to dormitories --/
def Assignment := Student → Dormitory

/-- Checks if an assignment is valid --/
def isValidAssignment (a : Assignment) : Prop :=
  (∃ s, a s = Dormitory.A) ∧
  (∃ s, a s = Dormitory.B) ∧
  (∃ s, a s = Dormitory.C) ∧
  (a Student.jia ≠ Dormitory.A)

/-- The number of valid assignments --/
def numValidAssignments : ℕ := sorry

theorem count_valid_assignments :
  numValidAssignments = 40 := by sorry

end count_valid_assignments_l2627_262735


namespace divisibility_in_sequence_l2627_262772

theorem divisibility_in_sequence (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ (n ∣ 2^(k + 1) - 1) :=
by sorry

end divisibility_in_sequence_l2627_262772


namespace local_minimum_condition_l2627_262765

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- Define the derivative of f(x)
def f_prime (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*b

-- Theorem statement
theorem local_minimum_condition (b : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ IsLocalMin (f b) x) →
  (f_prime b 0 < 0 ∧ f_prime b 1 > 0) :=
sorry

end local_minimum_condition_l2627_262765


namespace circle_equation_correct_l2627_262798

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * Real.cos (p.θ - c.center.θ)

theorem circle_equation_correct (c : PolarCircle) :
  c.center.ρ = 1 ∧ c.center.θ = 1 ∧ c.radius = 1 →
  ∀ p : PolarPoint, circleEquation c p ↔
    (p.ρ * Real.cos p.θ - c.center.ρ * Real.cos c.center.θ)^2 +
    (p.ρ * Real.sin p.θ - c.center.ρ * Real.sin c.center.θ)^2 = c.radius^2 :=
sorry

end circle_equation_correct_l2627_262798


namespace intersection_point_x_coord_l2627_262768

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line intersecting the left branch of the hyperbola -/
structure IntersectingLine where
  point : ℝ × ℝ
  intersection1 : ℝ × ℝ
  intersection2 : ℝ × ℝ

/-- Point P is the intersection of lines MA₁ and NA₂ -/
def intersection_point (h : Hyperbola) (l : IntersectingLine) : ℝ × ℝ := sorry

/-- Main theorem: The x-coordinate of point P is always -1 -/
theorem intersection_point_x_coord (h : Hyperbola) (l : IntersectingLine) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  l.point = (-4, 0) →
  l.intersection1.1 < 0 ∧ l.intersection1.2 > 0 →  -- M is in the second quadrant
  (intersection_point h l).1 = -1 := by sorry

end intersection_point_x_coord_l2627_262768


namespace train_speed_l2627_262778

theorem train_speed (t_pole : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_cross = 18 →
  l_stationary = 400 →
  ∃ v l, v = l / t_pole ∧ v = (l + l_stationary) / t_cross ∧ v = 40 :=
by sorry

end train_speed_l2627_262778


namespace expand_product_l2627_262731

theorem expand_product (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5*x - 24 := by
  sorry

end expand_product_l2627_262731


namespace range_of_m_l2627_262707

def p (m : ℝ) : Prop := m < -1

def q (m : ℝ) : Prop := -2 < m ∧ m < 3

theorem range_of_m : 
  {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)} = 
  {m : ℝ | m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)} := by sorry

end range_of_m_l2627_262707


namespace no_valid_digit_c_l2627_262781

theorem no_valid_digit_c : ¬∃ (C : ℕ), C < 10 ∧ (200 + 10 * C + 7) % 2 = 0 ∧ (200 + 10 * C + 7) % 5 = 0 := by
  sorry

end no_valid_digit_c_l2627_262781


namespace solution_x_percent_l2627_262780

/-- Represents a chemical solution with a certain percentage of chemical A -/
structure Solution where
  percentA : ℝ
  percentB : ℝ
  sum_to_one : percentA + percentB = 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution1 : Solution
  solution2 : Solution
  ratio1 : ℝ
  ratio2 : ℝ
  sum_to_one : ratio1 + ratio2 = 1
  percentA : ℝ

/-- The main theorem to be proved -/
theorem solution_x_percent (solution2 : Solution) (mixture : Mixture) :
  solution2.percentA = 0.4 →
  mixture.percentA = 0.32 →
  mixture.ratio1 = 0.8 →
  mixture.ratio2 = 0.2 →
  mixture.solution2 = solution2 →
  mixture.solution1.percentA = 0.3 := by
  sorry

end solution_x_percent_l2627_262780


namespace election_votes_theorem_l2627_262724

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (invalid_votes : ℕ),
    invalid_votes = 100 →
    valid_votes = total_votes - invalid_votes →
    ∃ (loser_votes winner_votes : ℕ),
      loser_votes = (30 * valid_votes) / 100 ∧
      winner_votes = valid_votes - loser_votes ∧
      winner_votes = loser_votes + 5000 →
      total_votes = 12600 :=
by sorry

end election_votes_theorem_l2627_262724


namespace min_value_sum_of_roots_l2627_262712

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), (∀ (x : ℝ), Real.sqrt (x^2 + (1 + x)^2) + Real.sqrt ((1 + x)^2 + (1 - x)^2) ≥ Real.sqrt 5) ∧
  (Real.sqrt (y^2 + (1 + y)^2) + Real.sqrt ((1 + y)^2 + (1 - y)^2) = Real.sqrt 5) :=
by sorry

end min_value_sum_of_roots_l2627_262712


namespace inequality_proof_l2627_262784

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end inequality_proof_l2627_262784


namespace rectangles_in_3x2_grid_l2627_262758

/-- The number of rectangles in a grid -/
def count_rectangles (m n : ℕ) : ℕ :=
  let one_by_one := m * n
  let one_by_two := m * (n - 1)
  let two_by_one := (m - 1) * n
  let two_by_two := (m - 1) * (n - 1)
  one_by_one + one_by_two + two_by_one + two_by_two

/-- Theorem: The number of rectangles in a 3x2 grid is 14 -/
theorem rectangles_in_3x2_grid :
  count_rectangles 3 2 = 14 := by
  sorry

end rectangles_in_3x2_grid_l2627_262758


namespace triangle_properties_max_area_l2627_262721

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √2 sin A = √3 cos A -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 2 * Real.sin t.A = Real.sqrt 3 * Real.cos t.A

/-- The equation a² - c² = b² - mbc -/
def equation (t : Triangle) (m : ℝ) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - m * t.b * t.c

theorem triangle_properties (t : Triangle) (m : ℝ) 
    (h1 : condition t) 
    (h2 : equation t m) : 
    m = 1 := by sorry

theorem max_area (t : Triangle) 
    (h1 : condition t) 
    (h2 : t.a = 2) : 
    (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 := by sorry

end triangle_properties_max_area_l2627_262721


namespace lucas_100th_mod8_l2627_262777

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

def lucas_mod8 (n : ℕ) : ℕ := lucas n % 8

theorem lucas_100th_mod8 : lucas_mod8 99 = 7 := by sorry

end lucas_100th_mod8_l2627_262777


namespace polynomial_perfect_square_condition_l2627_262737

/-- A polynomial ax^2 + by^2 + cz^2 + dxy + exz + fyz is a perfect square of a trinomial
    if and only if d = 2√(ab), e = 2√(ac), and f = 2√(bc) -/
theorem polynomial_perfect_square_condition
  (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + d * x * y + e * x * z + f * y * z = (p * x + q * y + r * z)^2)
  ↔
  (d^2 = 4 * a * b ∧ e^2 = 4 * a * c ∧ f^2 = 4 * b * c) :=
by sorry

end polynomial_perfect_square_condition_l2627_262737


namespace marys_average_speed_l2627_262757

/-- Mary's round trip walking problem -/
theorem marys_average_speed (distance_up distance_down : ℝ) (time_up time_down : ℝ) 
  (h1 : distance_up = 1.5)
  (h2 : distance_down = 1.5)
  (h3 : time_up = 45 / 60)
  (h4 : time_down = 15 / 60) :
  (distance_up + distance_down) / (time_up + time_down) = 3 := by
  sorry

end marys_average_speed_l2627_262757


namespace equation_equivalence_implies_mnp_30_l2627_262760

theorem equation_equivalence_implies_mnp_30 
  (b x z c : ℝ) (m n p : ℤ) 
  (h : ∀ x z c, b^8*x*z - b^7*z - b^6*x = b^5*(c^5 - 1) ↔ (b^m*x-b^n)*(b^p*z-b^3)=b^5*c^5) : 
  m * n * p = 30 := by
sorry

end equation_equivalence_implies_mnp_30_l2627_262760


namespace hexagon_angle_measure_l2627_262729

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, and C are congruent
  A = B ∧ B = C →
  -- Angles D, E, and F are congruent
  D = E ∧ E = F →
  -- Measure of angle A is 20 degrees less than measure of angle D
  A + 20 = D →
  -- Prove that the measure of angle D is 130 degrees
  D = 130 := by
sorry

end hexagon_angle_measure_l2627_262729


namespace square_of_sum_l2627_262711

theorem square_of_sum (x y : ℝ) 
  (h1 : 3 * x * (2 * x + y) = 14) 
  (h2 : y * (2 * x + y) = 35) : 
  (2 * x + y)^2 = 49 := by sorry

end square_of_sum_l2627_262711


namespace average_of_five_numbers_l2627_262714

theorem average_of_five_numbers (x : ℝ) : 
  (3 + 5 + 6 + 8 + x) / 5 = 7 → x = 13 := by
  sorry

end average_of_five_numbers_l2627_262714


namespace two_distinct_real_roots_l2627_262776

variable (a : ℝ)
variable (x : ℝ)

def f (a x : ℝ) : ℝ := (a+1)*(x^2+1)^2-(2*a+3)*(x^2+1)*x+(a+2)*x^2

theorem two_distinct_real_roots :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x₃ : ℝ, f a x₃ = 0 → x₃ = x₁ ∨ x₃ = x₂) ↔ a ≠ -1 := by
  sorry

end two_distinct_real_roots_l2627_262776


namespace square_even_implies_even_l2627_262752

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end square_even_implies_even_l2627_262752


namespace probability_ones_not_adjacent_l2627_262761

def total_arrangements : ℕ := 10

def favorable_arrangements : ℕ := 6

theorem probability_ones_not_adjacent :
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 5 := by
  sorry

end probability_ones_not_adjacent_l2627_262761


namespace zoo_animals_count_l2627_262732

theorem zoo_animals_count (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  monkeys = giraffes + 22 →
  giraffes = 2 :=
by
  sorry

end zoo_animals_count_l2627_262732


namespace pencil_count_l2627_262720

/-- The number of pencils Reeta has -/
def reeta_pencils : ℕ := 30

/-- The number of pencils Anika has -/
def anika_pencils : ℕ := 2 * reeta_pencils + 4

/-- The number of pencils Kamal has -/
def kamal_pencils : ℕ := 3 * reeta_pencils - 2

/-- The total number of pencils all three have together -/
def total_pencils : ℕ := reeta_pencils + anika_pencils + kamal_pencils

theorem pencil_count : total_pencils = 182 := by
  sorry

end pencil_count_l2627_262720


namespace multiplication_result_l2627_262704

theorem multiplication_result : (300000 : ℕ) * 100000 = 30000000000 := by
  sorry

end multiplication_result_l2627_262704


namespace max_value_theorem_l2627_262738

theorem max_value_theorem (x y : ℝ) (h : x^2 + y^2 = 18*x + 8*y + 10) :
  ∀ z : ℝ, 4*x + 3*y ≤ z → z ≤ 63 :=
sorry

end max_value_theorem_l2627_262738


namespace power_division_rule_l2627_262709

theorem power_division_rule (a : ℝ) : a^5 / a^3 = a^2 := by
  sorry

end power_division_rule_l2627_262709


namespace tangent_line_problem_l2627_262716

theorem tangent_line_problem (x y a : ℝ) :
  (∃ m : ℝ, y = 3 * x - 2 ∧ y = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  (a = 0 ∨ a = 2) :=
by sorry

end tangent_line_problem_l2627_262716


namespace intersection_of_A_and_B_l2627_262734

def A : Set Nat := {1, 3, 5, 7}
def B : Set Nat := {4, 5, 6, 7}

theorem intersection_of_A_and_B : A ∩ B = {5, 7} := by
  sorry

end intersection_of_A_and_B_l2627_262734


namespace partnership_profit_l2627_262723

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ     -- A's investment period
  b_period : ℕ     -- B's investment period
  b_profit : ℕ     -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let a_profit := p.b_profit * 6
  a_profit + p.b_profit

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment)
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 4500) : 
  total_profit p = 31500 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 4500 }

end partnership_profit_l2627_262723


namespace unique_root_quadratic_l2627_262747

/-- The quadratic equation x^2 - 6kx + 9k has exactly one real root if and only if k = 1, where k is positive. -/
theorem unique_root_quadratic (k : ℝ) (h : k > 0) :
  (∃! x : ℝ, x^2 - 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end unique_root_quadratic_l2627_262747


namespace batsman_highest_score_l2627_262725

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 140)
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = 
      ((total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score) ∧
    highest_score = 174 := by
  sorry

end batsman_highest_score_l2627_262725


namespace faye_candy_problem_l2627_262708

/-- Calculates the number of candy pieces Faye's sister gave her -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Theorem stating that given the problem conditions, Faye's sister gave her 40 pieces of candy -/
theorem faye_candy_problem (initial eaten final : ℕ) 
  (h_initial : initial = 47)
  (h_eaten : eaten = 25)
  (h_final : final = 62) :
  candy_from_sister initial eaten final = 40 := by
  sorry

end faye_candy_problem_l2627_262708
