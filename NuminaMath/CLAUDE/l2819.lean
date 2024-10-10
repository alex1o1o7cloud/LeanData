import Mathlib

namespace sqrt_11_between_3_and_4_l2819_281936

theorem sqrt_11_between_3_and_4 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := by
  sorry

end sqrt_11_between_3_and_4_l2819_281936


namespace percentage_of_girls_l2819_281947

theorem percentage_of_girls (boys girls : ℕ) (h1 : boys = 300) (h2 : girls = 450) :
  (girls : ℚ) / ((boys : ℚ) + (girls : ℚ)) * 100 = 60 := by
  sorry

end percentage_of_girls_l2819_281947


namespace simplify_and_evaluate_l2819_281928

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  ((2 * x - 2) / x - 1) / ((x^2 - 4*x + 4) / (x^2 - x)) = 3/2 :=
by sorry

end simplify_and_evaluate_l2819_281928


namespace cuboid_circumscribed_sphere_area_l2819_281917

theorem cuboid_circumscribed_sphere_area (x y z : ℝ) : 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  x * y = Real.sqrt 6 ∧ 
  y * z = Real.sqrt 2 ∧ 
  z * x = Real.sqrt 3 → 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi := by
sorry

end cuboid_circumscribed_sphere_area_l2819_281917


namespace percentage_calculation_l2819_281986

theorem percentage_calculation : (0.47 * 1442 - 0.36 * 1412) + 63 = 232.42 := by
  sorry

end percentage_calculation_l2819_281986


namespace min_a_for_quadratic_inequality_l2819_281910

theorem min_a_for_quadratic_inequality :
  (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → ∀ a : ℝ, x^2 + 2*a*x + 1 ≥ 0) →
  (∃ a_min : ℝ, a_min = -5/4 ∧
    (∀ a : ℝ, (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a*x + 1 ≥ 0) → a ≥ a_min) ∧
    (∀ x : ℝ, x > 0 ∧ x ≤ 1/2 → x^2 + 2*a_min*x + 1 ≥ 0)) :=
sorry

end min_a_for_quadratic_inequality_l2819_281910


namespace quadratic_function_with_log_range_l2819_281954

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_log_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : Set.range (fun x ↦ Real.log (f x)) = Set.Ici 0) :
  ∃ a b : ℝ, f = fun x ↦ x^2 + 2*x + 2 :=
sorry

end quadratic_function_with_log_range_l2819_281954


namespace binomial_coefficient_congruence_l2819_281925

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → a ≥ b → b ≥ 0 → 
  (Nat.choose (p * a) (p * b)) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end binomial_coefficient_congruence_l2819_281925


namespace ellipse_fraction_bounds_l2819_281956

theorem ellipse_fraction_bounds (x y : ℝ) (h : (x - 3)^2 + 4*(y - 1)^2 = 4) :
  ∃ (t : ℝ), (x + y - 3) / (x - y + 1) = t ∧ -1 ≤ t ∧ t ≤ 1 ∧
  (∃ (x₁ y₁ : ℝ), (x₁ - 3)^2 + 4*(y₁ - 1)^2 = 4 ∧ (x₁ + y₁ - 3) / (x₁ - y₁ + 1) = -1) ∧
  (∃ (x₂ y₂ : ℝ), (x₂ - 3)^2 + 4*(y₂ - 1)^2 = 4 ∧ (x₂ + y₂ - 3) / (x₂ - y₂ + 1) = 1) :=
by sorry

end ellipse_fraction_bounds_l2819_281956


namespace subgroup_equality_l2819_281937

variable {G : Type*} [Group G]

theorem subgroup_equality (S : Set G) (x s : G) (hs : s ∈ Subgroup.closure S) :
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {x * s}) ∧
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {s * x}) := by
  sorry

end subgroup_equality_l2819_281937


namespace balance_difference_theorem_l2819_281992

def initial_deposit : ℝ := 10000
def jasmine_rate : ℝ := 0.04
def lucas_rate : ℝ := 0.06
def years : ℕ := 20

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

theorem balance_difference_theorem :
  ∃ ε > 0, ε < 1 ∧
  (simple_interest initial_deposit lucas_rate years -
   compound_interest initial_deposit jasmine_rate years) - 89 < ε :=
sorry

end balance_difference_theorem_l2819_281992


namespace ladder_problem_l2819_281941

theorem ladder_problem (c a b : ℝ) : 
  c = 25 → a = 15 → c^2 = a^2 + b^2 → b = 20 := by
  sorry

end ladder_problem_l2819_281941


namespace gcd_12345_6789_l2819_281929

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l2819_281929


namespace complex_number_in_first_quadrant_l2819_281926

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 + I^2023)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_first_quadrant_l2819_281926


namespace power_of_two_floor_l2819_281962

theorem power_of_two_floor (n : ℕ) (h1 : n ≥ 4) 
  (h2 : ∃ k : ℕ, ⌊(2^n : ℝ) / n⌋ = 2^k) : 
  ∃ m : ℕ, n = 2^m :=
sorry

end power_of_two_floor_l2819_281962


namespace prime_cube_plus_one_l2819_281958

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) :
  Prime p ∧ p^(x : ℕ) = y^3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by sorry

end prime_cube_plus_one_l2819_281958


namespace inequality_range_l2819_281933

theorem inequality_range (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) ↔ (x < -1 ∨ x > 3) := by
  sorry

end inequality_range_l2819_281933


namespace fortieth_day_from_tuesday_is_sunday_l2819_281950

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem fortieth_day_from_tuesday_is_sunday :
  advanceDay DayOfWeek.Tuesday 40 = DayOfWeek.Sunday := by
  sorry


end fortieth_day_from_tuesday_is_sunday_l2819_281950


namespace tree_cutting_percentage_l2819_281967

theorem tree_cutting_percentage (initial_trees : ℕ) (final_trees : ℕ) (replant_rate : ℕ) : 
  initial_trees = 400 → 
  final_trees = 720 → 
  replant_rate = 5 → 
  (100 * (final_trees - initial_trees)) / (initial_trees * (replant_rate - 1)) = 20 := by
  sorry

end tree_cutting_percentage_l2819_281967


namespace solar_panel_flat_fee_l2819_281951

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def total_cost : ℕ := 147700

theorem solar_panel_flat_fee :
  total_cost - (land_acres * land_cost_per_acre + house_cost + 
    cow_count * cow_cost_per_unit + chicken_count * chicken_cost_per_unit + 
    solar_installation_hours * solar_installation_cost_per_hour) = 26000 := by
  sorry

end solar_panel_flat_fee_l2819_281951


namespace expand_polynomial_l2819_281952

theorem expand_polynomial (x : ℝ) : 
  (13 * x^2 + 5 * x + 3) * (3 * x^3) = 39 * x^5 + 15 * x^4 + 9 * x^3 := by
  sorry

end expand_polynomial_l2819_281952


namespace min_framing_for_specific_picture_l2819_281903

/-- Calculate the minimum number of linear feet of framing needed for an enlarged picture with border -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ⌈(perimeter_inches : ℚ) / 12⌉₊

/-- Theorem stating the minimum number of linear feet of framing needed for the specific picture -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 7 4 3 = 10 := by sorry

end min_framing_for_specific_picture_l2819_281903


namespace enrollment_increase_l2819_281998

theorem enrollment_increase (E : ℝ) (E_1992 : ℝ) (E_1993 : ℝ)
  (h1 : E_1993 = 1.26 * E)
  (h2 : E_1993 = 1.05 * E_1992) :
  (E_1992 - E) / E * 100 = 20 := by
  sorry

end enrollment_increase_l2819_281998


namespace f_as_difference_of_increasing_functions_l2819_281923

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3

theorem f_as_difference_of_increasing_functions :
  ∃ (g h : ℝ → ℝ), 
    (∀ x y, x < y → g x < g y) ∧ 
    (∀ x y, x < y → h x < h y) ∧ 
    (∀ x, f x = g x - h x) :=
sorry

end f_as_difference_of_increasing_functions_l2819_281923


namespace three_toppings_from_seven_l2819_281964

theorem three_toppings_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end three_toppings_from_seven_l2819_281964


namespace integral_x_exp_x_squared_l2819_281943

theorem integral_x_exp_x_squared (x : ℝ) :
  (deriv (fun x => (1/2) * Real.exp (x^2))) x = x * Real.exp (x^2) := by
  sorry

end integral_x_exp_x_squared_l2819_281943


namespace distance_circle_C_to_line_l_l2819_281935

/-- Circle C with center (1, 0) and radius 1 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}

/-- Line l with equation x + y + 2√2 - 1 = 0 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 2 * Real.sqrt 2 - 1 = 0}

/-- Center of circle C -/
def center_C : ℝ × ℝ := (1, 0)

/-- Distance from a point to a line -/
def point_to_line_distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem distance_circle_C_to_line_l :
  point_to_line_distance center_C line_l = 2 := by
  sorry

end distance_circle_C_to_line_l_l2819_281935


namespace fraction_to_decimal_l2819_281900

theorem fraction_to_decimal : (7 : ℚ) / 50 = 0.14 := by
  sorry

end fraction_to_decimal_l2819_281900


namespace radius_difference_is_zero_l2819_281940

/-- A circle with center C tangent to positive x and y-axes and externally tangent to another circle -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  externally_tangent : (radius - 2)^2 + radius^2 = (radius + 2)^2

/-- The radius difference between the largest and smallest possible radii is 0 -/
theorem radius_difference_is_zero : 
  ∀ (c₁ c₂ : TangentCircle), c₁.radius - c₂.radius = 0 := by
  sorry

end radius_difference_is_zero_l2819_281940


namespace tangent_problem_l2819_281904

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (π + α) = -1/3)
  (h2 : Real.tan (α + β) = (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α)) :
  (Real.tan (α + β) = 5/16) ∧ (Real.tan β = 31/43) := by
  sorry

end tangent_problem_l2819_281904


namespace frequency_distribution_best_for_proportions_l2819_281963

-- Define the possible statistical measures
inductive StatisticalMeasure
  | Average
  | Variance
  | Mode
  | FrequencyDistribution

-- Define a function to determine if a measure can calculate proportions within ranges
def canCalculateProportionsInRange (measure : StatisticalMeasure) : Prop :=
  match measure with
  | StatisticalMeasure.FrequencyDistribution => True
  | _ => False

-- Theorem statement
theorem frequency_distribution_best_for_proportions :
  ∀ (measure : StatisticalMeasure),
    canCalculateProportionsInRange measure →
    measure = StatisticalMeasure.FrequencyDistribution :=
by sorry

end frequency_distribution_best_for_proportions_l2819_281963


namespace amulet_seller_profit_l2819_281982

/-- Calculates the profit for an amulet seller at a Ren Faire --/
theorem amulet_seller_profit
  (days : ℕ)
  (amulets_per_day : ℕ)
  (selling_price : ℕ)
  (cost_price : ℕ)
  (faire_cut_percent : ℕ)
  (h1 : days = 2)
  (h2 : amulets_per_day = 25)
  (h3 : selling_price = 40)
  (h4 : cost_price = 30)
  (h5 : faire_cut_percent = 10)
  : (days * amulets_per_day * selling_price) - 
    (days * amulets_per_day * cost_price) - 
    (days * amulets_per_day * selling_price * faire_cut_percent / 100) = 300 :=
by sorry

end amulet_seller_profit_l2819_281982


namespace marble_distribution_l2819_281966

theorem marble_distribution (a : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ) :
  angela = a ∧ 
  brian = 2 * a ∧ 
  caden = 6 * a ∧ 
  daryl = 42 * a ∧
  angela + brian + caden + daryl = 126 →
  a = 42 / 17 := by
sorry

end marble_distribution_l2819_281966


namespace expression_value_l2819_281991

theorem expression_value (a : ℝ) (h1 : a - 1 ≥ 0) (h2 : 1 - a ≥ 0) :
  a + 2 * Real.sqrt (a - 1) - Real.sqrt (1 - a) + 3 = 4 := by
  sorry

end expression_value_l2819_281991


namespace exam_comparison_l2819_281942

theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 120)
  (h2 : lyssa_incorrect_percent = 25 / 100)
  (h3 : precious_mistakes = 17) :
  (total_items - (lyssa_incorrect_percent * total_items).num) - (total_items - precious_mistakes) = -13 :=
by sorry

end exam_comparison_l2819_281942


namespace chord_length_l2819_281985

theorem chord_length (r : ℝ) (h : r = 15) : 
  let chord_length : ℝ := 2 * (r^2 - (r/3)^2).sqrt
  chord_length = 20 * Real.sqrt 2 := by
  sorry

end chord_length_l2819_281985


namespace ball_max_height_l2819_281974

/-- The height function of the ball's path -/
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 40

/-- Theorem stating that the maximum value of f is equal to max_height -/
theorem ball_max_height : ∀ t : ℝ, f t ≤ max_height := by sorry

end ball_max_height_l2819_281974


namespace tangent_parallel_to_x_axis_l2819_281921

/-- The curve y = x^2 - 3x -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem tangent_parallel_to_x_axis :
  let P : ℝ × ℝ := (3/2, -9/4)
  (f P.1 = P.2) ∧ (f' P.1 = 0) := by sorry

end tangent_parallel_to_x_axis_l2819_281921


namespace lesser_fraction_l2819_281944

theorem lesser_fraction (x y : ℚ) : 
  x + y = 8/9 → x * y = 1/8 → min x y = 7/40 := by
  sorry

end lesser_fraction_l2819_281944


namespace even_sum_converse_true_l2819_281906

theorem even_sum_converse_true (a b : ℤ) : 
  (∀ (a b : ℤ), Even (a + b) → Even a ∧ Even b) → 
  (Even a ∧ Even b → Even (a + b)) := by sorry

end even_sum_converse_true_l2819_281906


namespace fraction_equality_l2819_281953

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (2 * m * r - 5 * n * t) / (5 * n * t - 4 * m * r) = -2 / 9 := by
  sorry

end fraction_equality_l2819_281953


namespace train_length_l2819_281949

/-- The length of a train given its speed and time to pass a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h1 : speed_kmh = 63) (h2 : time_s = 16) :
  speed_kmh * 1000 / 3600 * time_s = 280 := by
  sorry

end train_length_l2819_281949


namespace vanya_can_always_win_l2819_281924

/-- Represents a sequence of signs (+1 for "+", -1 for "-") -/
def SignSequence := List Int

/-- Represents a move that swaps two adjacent signs -/
def Move := Nat

/-- Applies a move to a sign sequence -/
def applyMove (seq : SignSequence) (m : Move) : SignSequence :=
  sorry

/-- Evaluates the expression given a sign sequence -/
def evaluateExpression (seq : SignSequence) : Int :=
  sorry

/-- Checks if a number is divisible by 7 -/
def isDivisibleBy7 (n : Int) : Prop :=
  n % 7 = 0

/-- The main theorem: Vanya can always achieve a sum divisible by 7 -/
theorem vanya_can_always_win (initialSeq : SignSequence) :
  ∃ (moves : List Move), isDivisibleBy7 (evaluateExpression (moves.foldl applyMove initialSeq)) :=
sorry

end vanya_can_always_win_l2819_281924


namespace apple_distribution_l2819_281987

theorem apple_distribution (x : ℕ) (h : x > 0) :
  (1430 / x : ℚ) - (1430 / (x + 45) : ℚ) = 9 → 1430 / x = 22 := by
  sorry

end apple_distribution_l2819_281987


namespace certain_number_proof_l2819_281909

theorem certain_number_proof (x : ℝ) : x + 6 = 8 → x = 2 := by
  sorry

end certain_number_proof_l2819_281909


namespace B_profit_share_l2819_281920

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_difference_AC : ℕ := 560

theorem B_profit_share :
  let total_investment := investment_A + investment_B + investment_C
  let profit_ratio_A := investment_A / total_investment
  let profit_ratio_B := investment_B / total_investment
  let profit_ratio_C := investment_C / total_investment
  let total_profit := profit_difference_AC * total_investment / (profit_ratio_C - profit_ratio_A)
  profit_ratio_B * total_profit = 1400 := by sorry

end B_profit_share_l2819_281920


namespace midpoint_sum_invariant_problem_solution_l2819_281978

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  { vertices := sorry }  -- Implementation details omitted

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  List.sum (List.map Prod.fst p.vertices)

theorem midpoint_sum_invariant (p : Polygon) (h : p.vertices.length = 50) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = sumXCoordinates p := by
  sorry

/-- The main theorem that proves the result for the specific case in the problem -/
theorem problem_solution (p : Polygon) (h1 : p.vertices.length = 50) (h2 : sumXCoordinates p = 1005) :
  let p2 := midpointPolygon p
  let p3 := midpointPolygon p2
  sumXCoordinates p3 = 1005 := by
  sorry

end midpoint_sum_invariant_problem_solution_l2819_281978


namespace frank_pepe_height_difference_l2819_281918

-- Define the players
structure Player where
  name : String
  height : Float

-- Define the team
def team : List Player :=
  [
    { name := "Big Joe", height := 8 },
    { name := "Ben", height := 7 },
    { name := "Larry", height := 6 },
    { name := "Frank", height := 5.5 },
    { name := "Pepe", height := 4.5 }
  ]

-- Define the height difference function
def heightDifference (p1 p2 : Player) : Float :=
  p1.height - p2.height

-- Theorem statement
theorem frank_pepe_height_difference :
  let frank := team.find? (fun p => p.name = "Frank")
  let pepe := team.find? (fun p => p.name = "Pepe")
  ∀ (f p : Player), frank = some f → pepe = some p →
    heightDifference f p = 1 := by
  sorry

end frank_pepe_height_difference_l2819_281918


namespace veranda_width_l2819_281983

/-- Proves that the width of a veranda surrounding a rectangular room is 2 meters -/
theorem veranda_width (room_length room_width veranda_area : ℝ) : 
  room_length = 18 → 
  room_width = 12 → 
  veranda_area = 136 → 
  ∃ w : ℝ, w = 2 ∧ 
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end veranda_width_l2819_281983


namespace age_birth_year_problem_l2819_281901

theorem age_birth_year_problem :
  ∃ (age1 age2 : ℕ) (birth_year1 birth_year2 : ℕ),
    age1 > 11 ∧ age2 > 11 ∧
    birth_year1 ≥ 1900 ∧ birth_year1 < 2010 ∧
    birth_year2 ≥ 1900 ∧ birth_year2 < 2010 ∧
    age1 = (birth_year1 / 1000) + ((birth_year1 % 1000) / 100) + ((birth_year1 % 100) / 10) + (birth_year1 % 10) ∧
    age2 = (birth_year2 / 1000) + ((birth_year2 % 1000) / 100) + ((birth_year2 % 100) / 10) + (birth_year2 % 10) ∧
    2010 - birth_year1 = age1 ∧
    2009 - birth_year2 = age2 ∧
    age1 ≠ age2 ∧
    birth_year1 ≠ birth_year2 :=
by sorry

end age_birth_year_problem_l2819_281901


namespace field_purchase_problem_l2819_281960

theorem field_purchase_problem :
  let good_field_value : ℚ := 300  -- value of 1 acre of good field
  let bad_field_value : ℚ := 500 / 7  -- value of 1 acre of bad field
  let total_area : ℚ := 100  -- total area in acres
  let total_cost : ℚ := 10000  -- total cost in coins
  let good_field_acres : ℚ := 25 / 2  -- solution for good field acres
  let bad_field_acres : ℚ := 175 / 2  -- solution for bad field acres
  (good_field_acres + bad_field_acres = total_area) ∧
  (good_field_value * good_field_acres + bad_field_value * bad_field_acres = total_cost) :=
by sorry


end field_purchase_problem_l2819_281960


namespace boris_clock_theorem_l2819_281932

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_valid_time (h m : ℕ) : Prop :=
  h ≤ 23 ∧ m ≤ 59

def satisfies_clock_conditions (h m : ℕ) : Prop :=
  is_valid_time h m ∧ digit_sum h + digit_sum m = 6 ∧ h + m = 15

def possible_times : Set (ℕ × ℕ) :=
  {(0,15), (1,14), (2,13), (3,12), (4,11), (5,10), (10,5), (11,4), (12,3), (13,2), (14,1), (15,0)}

theorem boris_clock_theorem :
  {(h, m) | satisfies_clock_conditions h m} = possible_times :=
sorry

end boris_clock_theorem_l2819_281932


namespace bobs_walking_rate_l2819_281989

/-- Proves that Bob's walking rate is 3 miles per hour given the conditions of the problem -/
theorem bobs_walking_rate (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) :
  total_distance = 17 →
  yolanda_rate = 3 →
  bob_distance = 8 →
  ∃ (bob_rate : ℝ), bob_rate = 3 ∧ bob_rate * (total_distance / (yolanda_rate + bob_rate) - 1) = bob_distance :=
by sorry

end bobs_walking_rate_l2819_281989


namespace john_total_spend_l2819_281913

-- Define the given quantities
def silver_amount : Real := 1.5
def silver_price_per_ounce : Real := 20
def gold_amount : Real := 2 * silver_amount
def gold_price_per_ounce : Real := 50 * silver_price_per_ounce

-- Define the total cost function
def total_cost : Real :=
  silver_amount * silver_price_per_ounce + gold_amount * gold_price_per_ounce

-- Theorem statement
theorem john_total_spend :
  total_cost = 3030 := by
  sorry

end john_total_spend_l2819_281913


namespace cubic_polynomial_theorem_l2819_281969

/-- Represents a cubic polynomial a₃x³ - x² + a₁x - 7 = 0 -/
structure CubicPolynomial where
  a₃ : ℝ
  a₁ : ℝ

/-- Represents the roots of the cubic polynomial -/
structure Roots where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Checks if the given roots satisfy the condition -/
def satisfiesCondition (r : Roots) : Prop :=
  (225 * r.α^2) / (r.α^2 + 7) = (144 * r.β^2) / (r.β^2 + 7) ∧
  (144 * r.β^2) / (r.β^2 + 7) = (100 * r.γ^2) / (r.γ^2 + 7)

/-- Checks if the given roots are positive -/
def arePositive (r : Roots) : Prop :=
  r.α > 0 ∧ r.β > 0 ∧ r.γ > 0

/-- Checks if the given roots are valid for the cubic polynomial -/
def areValidRoots (p : CubicPolynomial) (r : Roots) : Prop :=
  p.a₃ * r.α^3 - r.α^2 + p.a₁ * r.α - 7 = 0 ∧
  p.a₃ * r.β^3 - r.β^2 + p.a₁ * r.β - 7 = 0 ∧
  p.a₃ * r.γ^3 - r.γ^2 + p.a₁ * r.γ - 7 = 0

theorem cubic_polynomial_theorem (p : CubicPolynomial) (r : Roots) 
  (h1 : satisfiesCondition r)
  (h2 : arePositive r)
  (h3 : areValidRoots p r) :
  abs (p.a₁ - 130.6667) < 0.0001 := by
  sorry

end cubic_polynomial_theorem_l2819_281969


namespace sandra_son_age_l2819_281979

/-- Sandra's current age -/
def sandra_age : ℕ := 36

/-- The ratio of Sandra's age to her son's age 3 years ago -/
def age_ratio : ℕ := 3

/-- Sandra's son's current age -/
def son_age : ℕ := 14

theorem sandra_son_age : 
  sandra_age - 3 = age_ratio * (son_age - 3) :=
sorry

end sandra_son_age_l2819_281979


namespace vkontakte_users_l2819_281916

-- Define the people as propositions (being on VKontakte)
variable (M : Prop) -- Marya Ivanovna
variable (I : Prop) -- Ivan Ilyich
variable (A : Prop) -- Alexandra Varfolomeevna
variable (P : Prop) -- Petr Petrovich

-- Define the conditions
def condition1 : Prop := M → (I ∧ A)
def condition2 : Prop := (A ∧ ¬P) ∨ (¬A ∧ P)
def condition3 : Prop := I ∨ M
def condition4 : Prop := I ↔ P

-- Theorem statement
theorem vkontakte_users 
  (h1 : condition1 M I A)
  (h2 : condition2 A P)
  (h3 : condition3 I M)
  (h4 : condition4 I P) :
  I ∧ P ∧ ¬M ∧ ¬A :=
sorry

end vkontakte_users_l2819_281916


namespace inequality_solution_set_l2819_281976

theorem inequality_solution_set 
  (a b : ℝ) (ha : a < 0) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > -b / a} := by
sorry

end inequality_solution_set_l2819_281976


namespace candy_division_l2819_281990

theorem candy_division (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0)
  (h_order : p < q ∧ q < r) (h_a : 20 = 3 * r - 2 * p) (h_b : 10 = r - p)
  (h_c : 9 = 3 * q - 3 * p) (h_c_sum : 3 * q = 18) :
  p = 3 ∧ q = 6 ∧ r = 13 := by sorry

end candy_division_l2819_281990


namespace probability_zeros_not_adjacent_l2819_281919

-- Define the total number of elements
def total_elements : ℕ := 5

-- Define the number of ones
def num_ones : ℕ := 3

-- Define the number of zeros
def num_zeros : ℕ := 2

-- Define the total number of arrangements
def total_arrangements : ℕ := Nat.factorial total_elements

-- Define the number of arrangements where zeros are adjacent
def adjacent_zero_arrangements : ℕ := 2 * Nat.factorial (total_elements - 1)

-- Statement to prove
theorem probability_zeros_not_adjacent :
  (1 : ℚ) - (adjacent_zero_arrangements : ℚ) / total_arrangements = 0.6 := by
  sorry

end probability_zeros_not_adjacent_l2819_281919


namespace negation_of_existence_power_of_two_bound_negation_l2819_281959

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_bound_negation : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end negation_of_existence_power_of_two_bound_negation_l2819_281959


namespace max_value_expression_l2819_281915

def A : Set Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

theorem max_value_expression (v w x y z : Int) 
  (hv : v ∈ A) (hw : w ∈ A) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A)
  (h_vw : v * w = x) (h_w : w ≠ 0) :
  (∀ v' w' x' y' z' : Int, 
    v' ∈ A → w' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A → 
    v' * w' = x' → w' ≠ 0 →
    v * x - y * z ≥ v' * x' - y' * z') →
  v * x - y * z = 150 :=
sorry

end max_value_expression_l2819_281915


namespace chessboard_square_rectangle_ratio_l2819_281938

/-- The number of rectangles formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_rectangles : ℕ := 2025

/-- The number of squares formed by 10 horizontal and 10 vertical lines on a 9x9 chessboard -/
def num_squares : ℕ := 285

/-- The ratio of squares to rectangles expressed as a fraction with relatively prime positive integers -/
def square_rectangle_ratio : ℚ := 19 / 135

theorem chessboard_square_rectangle_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = square_rectangle_ratio := by
  sorry

end chessboard_square_rectangle_ratio_l2819_281938


namespace infinitely_many_a_for_perfect_cube_l2819_281996

theorem infinitely_many_a_for_perfect_cube (n : ℕ) : 
  ∃ (f : ℕ → ℤ), Function.Injective f ∧ ∀ (k : ℕ), ∃ (m : ℕ), (n^6 + 3 * (f k) : ℤ) = m^3 := by
  sorry

end infinitely_many_a_for_perfect_cube_l2819_281996


namespace algebraic_simplification_l2819_281972

theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  (a^2 + a*b + b^2) / (a + b) - (a^2 - a*b + b^2) / (a - b) + (2*b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := by
  sorry

end algebraic_simplification_l2819_281972


namespace geometric_sequence_first_term_l2819_281968

/-- Given a geometric sequence {a_n} with common ratio q = 2,
    if the arithmetic mean of a_2 and 2a_3 is 5, then a_1 = 1 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h_geom : ∀ n, a (n + 1) = 2 * a n)  -- geometric sequence with ratio 2
  (h_mean : (a 2 + 2 * a 3) / 2 = 5)  -- arithmetic mean condition
  : a 1 = 1 := by
sorry


end geometric_sequence_first_term_l2819_281968


namespace popsicle_sticks_count_l2819_281993

theorem popsicle_sticks_count (gino_sticks : ℕ) (total_sticks : ℕ) (my_sticks : ℕ) : 
  gino_sticks = 63 → total_sticks = 113 → total_sticks = gino_sticks + my_sticks → my_sticks = 50 := by
  sorry

end popsicle_sticks_count_l2819_281993


namespace school_boys_count_l2819_281912

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end school_boys_count_l2819_281912


namespace circle_properties_l2819_281997

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 12*x - 12*y - 88

def line_equation (x y : ℝ) : ℝ := x + 3*y + 16

def point_A : ℝ × ℝ := (-6, 10)
def point_B : ℝ × ℝ := (2, -6)

theorem circle_properties :
  (circle_equation point_A.1 point_A.2 = 0) ∧
  (circle_equation point_B.1 point_B.2 = 0) ∧
  (line_equation point_B.1 point_B.2 = 0) ∧
  (∃ (t : ℝ), t ≠ 0 ∧
    (2 * point_B.1 - 12) * 1 + (2 * point_B.2 - 12) * 3 = t * (1^2 + 3^2)) :=
sorry

end circle_properties_l2819_281997


namespace sum_of_powers_l2819_281905

theorem sum_of_powers (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a^1 + a^2 + a^3 + a^4 + a^5 + a^6 + a^7 + a^8 = 0 :=
by sorry

end sum_of_powers_l2819_281905


namespace janes_farm_chickens_l2819_281980

/-- Represents the farm scenario with chickens and egg production --/
structure Farm where
  chickens : ℕ
  eggs_per_chicken_per_week : ℕ
  price_per_dozen : ℕ
  weeks : ℕ
  total_revenue : ℕ

/-- Calculates the total number of eggs produced by the farm in the given period --/
def total_eggs (f : Farm) : ℕ :=
  f.chickens * f.eggs_per_chicken_per_week * f.weeks

/-- Calculates the revenue generated from selling all eggs --/
def revenue (f : Farm) : ℕ :=
  (total_eggs f / 12) * f.price_per_dozen

/-- Theorem stating that Jane's farm has 10 chickens given the conditions --/
theorem janes_farm_chickens :
  ∃ (f : Farm),
    f.eggs_per_chicken_per_week = 6 ∧
    f.price_per_dozen = 2 ∧
    f.weeks = 2 ∧
    f.total_revenue = 20 ∧
    f.chickens = 10 :=
  sorry

end janes_farm_chickens_l2819_281980


namespace fraction_comparison_l2819_281911

theorem fraction_comparison : 
  (100 : ℚ) / 101 > 199 / 201 ∧ 199 / 201 > 99 / 100 := by
  sorry

end fraction_comparison_l2819_281911


namespace water_added_to_container_l2819_281922

/-- The amount of water added to a container -/
def water_added (capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) : ℝ :=
  capacity * final_fraction - capacity * initial_fraction

/-- Theorem stating the amount of water added to the container -/
theorem water_added_to_container : 
  water_added 80 0.4 0.75 = 28 := by sorry

end water_added_to_container_l2819_281922


namespace diagonal_not_parallel_to_sides_l2819_281927

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) :=
sorry

end diagonal_not_parallel_to_sides_l2819_281927


namespace unique_divisible_number_l2819_281908

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem unique_divisible_number :
  ∃! D : ℕ, D < 10 ∧ 
    is_divisible_by_3 (sum_of_digits (1000 + D * 10 + 4)) ∧ 
    is_divisible_by_4 (last_two_digits (1000 + D * 10 + 4)) :=
sorry

end unique_divisible_number_l2819_281908


namespace mean_problem_l2819_281965

theorem mean_problem (x y : ℝ) : 
  (28 + x + 42 + y + 78 + 104) / 6 = 62 → 
  x + y = 120 ∧ (x + y) / 2 = 60 := by
sorry

end mean_problem_l2819_281965


namespace total_cost_is_four_dollars_l2819_281994

/-- The cost of a single tire in dollars -/
def cost_per_tire : ℝ := 0.50

/-- The number of tires -/
def number_of_tires : ℕ := 8

/-- The total cost of all tires -/
def total_cost : ℝ := cost_per_tire * number_of_tires

theorem total_cost_is_four_dollars : total_cost = 4 := by
  sorry

end total_cost_is_four_dollars_l2819_281994


namespace reinforcement_size_l2819_281971

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - initial_garrison * days_passed
  let new_total_men := initial_garrison + (provisions_left / remaining_duration - initial_garrison)
  new_total_men - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end reinforcement_size_l2819_281971


namespace cat_relocation_proportion_l2819_281914

/-- Calculates the proportion of cats relocated in the second mission -/
def proportion_relocated (initial_cats : ℕ) (first_mission_relocated : ℕ) (final_remaining : ℕ) : ℚ :=
  let remaining_after_first := initial_cats - first_mission_relocated
  let relocated_second := remaining_after_first - final_remaining
  relocated_second / remaining_after_first

theorem cat_relocation_proportion :
  proportion_relocated 1800 600 600 = 1/2 := by
  sorry

#eval proportion_relocated 1800 600 600

end cat_relocation_proportion_l2819_281914


namespace function_bound_l2819_281995

-- Define the properties of functions f and g
def satisfies_functional_equation (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y

def not_identically_zero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x ≠ 0

def bounded_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x, |f x| ≤ 1

-- Theorem statement
theorem function_bound (f g : ℝ → ℝ) 
  (h1 : satisfies_functional_equation f g)
  (h2 : not_identically_zero f)
  (h3 : bounded_by_one f) :
  bounded_by_one g :=
sorry

end function_bound_l2819_281995


namespace propositions_truth_l2819_281977

-- Proposition 1
def proposition1 : Prop := ∃ a b : ℝ, a ≤ b ∧ a^2 > b^2

-- Proposition 2
def proposition2 : Prop := ∀ x y : ℝ, x = -y → x + y = 0

-- Proposition 3
def proposition3 : Prop := ∀ x : ℝ, (x ≤ -2 ∨ x ≥ 2) → x^2 ≥ 4

theorem propositions_truth : ¬proposition1 ∧ proposition2 ∧ proposition3 := by
  sorry

end propositions_truth_l2819_281977


namespace emails_left_theorem_l2819_281981

/-- Given an initial number of emails, calculate the number of emails left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
def emails_left_in_inbox (initial_emails : ℕ) : ℕ :=
  let after_trash := initial_emails / 2
  let to_work_folder := (after_trash * 40) / 100
  after_trash - to_work_folder

/-- Theorem stating that given 400 initial emails, 120 emails are left in the inbox
    after moving half to trash and 40% of the remainder to a work folder. -/
theorem emails_left_theorem : emails_left_in_inbox 400 = 120 := by
  sorry

#eval emails_left_in_inbox 400

end emails_left_theorem_l2819_281981


namespace jane_lemonade_glasses_l2819_281946

/-- The number of glasses of lemonade that can be made -/
def glasses_of_lemonade (total_lemons : ℕ) (lemons_per_glass : ℕ) : ℕ :=
  total_lemons / lemons_per_glass

/-- Theorem: Jane can make 9 glasses of lemonade -/
theorem jane_lemonade_glasses : glasses_of_lemonade 18 2 = 9 := by
  sorry

end jane_lemonade_glasses_l2819_281946


namespace union_equals_reals_l2819_281975

-- Define the sets E and F
def E : Set ℝ := {x | x^2 - 5*x - 6 > 0}
def F (a : ℝ) : Set ℝ := {x | x - 5 < a}

-- State the theorem
theorem union_equals_reals (a : ℝ) (h : (11 : ℝ) ∈ F a) : E ∪ F a = Set.univ := by
  sorry

end union_equals_reals_l2819_281975


namespace remainder_7n_mod_4_l2819_281961

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l2819_281961


namespace book_price_l2819_281902

def original_price : ℝ → Prop :=
  fun price =>
    let first_discount := price * (1 - 1/5)
    let second_discount := first_discount * (1 - 1/5)
    second_discount = 32

theorem book_price : original_price 50 := by
  sorry

end book_price_l2819_281902


namespace salon_customers_count_l2819_281984

/-- Represents the number of customers who made only one visit -/
def single_visit_customers : ℕ := 44

/-- Represents the number of customers who made two visits -/
def double_visit_customers : ℕ := 30

/-- Represents the number of customers who made three visits -/
def triple_visit_customers : ℕ := 10

/-- The cost of the first visit in a calendar month -/
def first_visit_cost : ℕ := 10

/-- The cost of each subsequent visit in the same calendar month -/
def subsequent_visit_cost : ℕ := 8

/-- The total revenue for the calendar month -/
def total_revenue : ℕ := 1240

theorem salon_customers_count :
  single_visit_customers + double_visit_customers + triple_visit_customers = 84 ∧
  first_visit_cost * (single_visit_customers + double_visit_customers + triple_visit_customers) +
  subsequent_visit_cost * (double_visit_customers + 2 * triple_visit_customers) = total_revenue :=
sorry

end salon_customers_count_l2819_281984


namespace speeding_motorists_percentage_l2819_281955

theorem speeding_motorists_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_speeders : ℝ)
  (h2 : ticketed_speeders = 0.2 * total_motorists)
  (h3 : ticketed_speeders = 0.8 * (ticketed_speeders + (0.2 * (ticketed_speeders / 0.8))))
  : (ticketed_speeders / 0.8) / total_motorists = 0.25 := by
sorry

end speeding_motorists_percentage_l2819_281955


namespace subset_implies_a_equals_one_l2819_281930

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, -1, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l2819_281930


namespace sin_has_property_T_l2819_281939

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁) * (deriv f x₂) = -1

-- State the theorem
theorem sin_has_property_T :
  has_property_T Real.sin :=
sorry

end sin_has_property_T_l2819_281939


namespace increasing_quadratic_function_condition_l2819_281999

/-- A function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function_condition (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end increasing_quadratic_function_condition_l2819_281999


namespace expression_simplification_l2819_281907

theorem expression_simplification (x : ℝ) (h : x = (Real.sqrt 3 - 1) / 3) :
  (2 / (x - 1) + 1 / (x + 1)) * (x^2 - 1) = Real.sqrt 3 := by
  sorry

end expression_simplification_l2819_281907


namespace birds_in_pet_shop_l2819_281945

/-- The number of birds in a pet shop -/
def number_of_birds (total animals : ℕ) (kittens hamsters : ℕ) : ℕ :=
  total - kittens - hamsters

/-- Theorem: There are 30 birds in the pet shop -/
theorem birds_in_pet_shop :
  let total := 77
  let kittens := 32
  let hamsters := 15
  number_of_birds total kittens hamsters = 30 := by
sorry

end birds_in_pet_shop_l2819_281945


namespace specific_tangent_distances_l2819_281931

/-- Two externally tangent circles with radii R and r -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_external : R > r

/-- The distances from the point of tangency to the common tangents -/
def tangent_distances (c : TangentCircles) : Set ℝ :=
  {0, (c.R + c.r) * c.r / c.R}

/-- Theorem about the distances for specific radii -/
theorem specific_tangent_distances :
  ∃ c : TangentCircles, c.R = 3 ∧ c.r = 1 ∧ tangent_distances c = {0, 7/3} := by
  sorry

end specific_tangent_distances_l2819_281931


namespace polynomial_factoring_l2819_281988

theorem polynomial_factoring (a x y : ℝ) : a * x^2 - a * y^2 = a * (x + y) * (x - y) := by
  sorry

end polynomial_factoring_l2819_281988


namespace apples_per_pie_l2819_281948

theorem apples_per_pie 
  (total_apples : ℕ) 
  (unripe_apples : ℕ) 
  (num_pies : ℕ) 
  (h1 : total_apples = 34) 
  (h2 : unripe_apples = 6) 
  (h3 : num_pies = 7) 
  (h4 : unripe_apples < total_apples) :
  (total_apples - unripe_apples) / num_pies = 4 := by
  sorry

end apples_per_pie_l2819_281948


namespace quadrilateral_front_view_solids_l2819_281970

-- Define the set of geometric solids
inductive GeometricSolid
  | Cone
  | Cylinder
  | TriangularPyramid
  | QuadrangularPrism

-- Define a predicate for having a quadrilateral front view
def hasQuadrilateralFrontView (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | GeometricSolid.QuadrangularPrism => True
  | _ => False

-- Theorem statement
theorem quadrilateral_front_view_solids :
  ∀ (solid : GeometricSolid),
    hasQuadrilateralFrontView solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.QuadrangularPrism) :=
by sorry

end quadrilateral_front_view_solids_l2819_281970


namespace vertical_angles_are_congruent_l2819_281957

-- Define what it means for two angles to be vertical
def are_vertical_angles (α β : Angle) : Prop := sorry

-- Define what it means for two angles to be congruent
def are_congruent (α β : Angle) : Prop := sorry

-- Theorem statement
theorem vertical_angles_are_congruent (α β : Angle) :
  are_vertical_angles α β → are_congruent α β := by
  sorry

end vertical_angles_are_congruent_l2819_281957


namespace green_tile_probability_l2819_281973

theorem green_tile_probability :
  let tiles := Finset.range 100
  let green_tiles := tiles.filter (fun n => (n + 1) % 7 = 3)
  (green_tiles.card : ℚ) / tiles.card = 7 / 50 := by
  sorry

end green_tile_probability_l2819_281973


namespace log_function_fixed_point_l2819_281934

theorem log_function_fixed_point (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log a + 1
  f 1 = 1 := by
  sorry

end log_function_fixed_point_l2819_281934
