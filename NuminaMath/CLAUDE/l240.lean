import Mathlib

namespace NUMINAMATH_CALUDE_equal_area_line_slope_l240_24070

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The problem setup -/
def circles : List Circle := [
  { center := (10, 80), radius := 4 },
  { center := (13, 60), radius := 4 },
  { center := (15, 70), radius := 4 }
]

/-- A line passing through a given point -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line divides the total area of circles equally -/
def dividesAreaEqually (l : Line) (cs : List Circle) : Prop := sorry

/-- The main theorem -/
theorem equal_area_line_slope :
  ∃ l : Line, l.passesThrough = (13, 60) ∧ 
    dividesAreaEqually l circles ∧ 
    abs l.slope = 5 := by sorry

end NUMINAMATH_CALUDE_equal_area_line_slope_l240_24070


namespace NUMINAMATH_CALUDE_inequality_solution_l240_24067

theorem inequality_solution (x : ℝ) : 
  (3 + 1 / (3 * x - 2) ≥ 5) ∧ (3 * x - 2 ≠ 0) → 
  x ∈ Set.Iio (2 / 3) ∪ Set.Ioc (2 / 3) (5 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l240_24067


namespace NUMINAMATH_CALUDE_player_one_points_l240_24086

/-- Represents the sectors on the rotating table -/
def sectors : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

/-- The number of players -/
def num_players : ℕ := 16

/-- The number of rotations -/
def num_rotations : ℕ := 13

/-- Calculate the points for a player after given number of rotations -/
def player_points (player : ℕ) (rotations : ℕ) : ℕ := sorry

theorem player_one_points :
  player_points 5 num_rotations = 72 →
  player_points 9 num_rotations = 84 →
  player_points 1 num_rotations = 20 := by sorry

end NUMINAMATH_CALUDE_player_one_points_l240_24086


namespace NUMINAMATH_CALUDE_usual_time_calculation_l240_24071

theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) (h2 : usual_speed > 0) : 
  (usual_speed * usual_time = (usual_speed / 2) * (usual_time + 24)) → 
  usual_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l240_24071


namespace NUMINAMATH_CALUDE_count_congruent_integers_l240_24025

theorem count_congruent_integers (n : ℕ) (m : ℕ) (a : ℕ) (b : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < n ∧ x % m = a) (Finset.range n)).card = b + 1 :=
by
  sorry

#check count_congruent_integers 1500 13 7 114

end NUMINAMATH_CALUDE_count_congruent_integers_l240_24025


namespace NUMINAMATH_CALUDE_product_of_numbers_l240_24021

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l240_24021


namespace NUMINAMATH_CALUDE_product_of_tripled_numbers_with_reciprocals_l240_24049

theorem product_of_tripled_numbers_with_reciprocals (x : ℝ) : 
  (x + 1/x = 3*x) → (∃ y : ℝ, (y + 1/y = 3*y) ∧ (x * y = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_tripled_numbers_with_reciprocals_l240_24049


namespace NUMINAMATH_CALUDE_spending_vs_earning_difference_l240_24085

def initial_amount : Int := 153
def part_time_earnings : Int := 65
def atm_collection : Int := 195
def supermarket_spending : Int := 87
def electronics_spending : Int := 134
def clothes_spending : Int := 78

theorem spending_vs_earning_difference :
  (supermarket_spending + electronics_spending + clothes_spending) -
  (part_time_earnings + atm_collection) = -39 :=
by sorry

end NUMINAMATH_CALUDE_spending_vs_earning_difference_l240_24085


namespace NUMINAMATH_CALUDE_negation_of_forall_greater_than_two_l240_24001

theorem negation_of_forall_greater_than_two :
  (¬ (∀ x : ℝ, x > 2)) ↔ (∃ x : ℝ, x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_greater_than_two_l240_24001


namespace NUMINAMATH_CALUDE_john_hats_cost_l240_24032

def weeks : ℕ := 20
def days_per_week : ℕ := 7
def odd_day_price : ℕ := 45
def even_day_price : ℕ := 60
def discount_threshold : ℕ := 50
def discount_rate : ℚ := 1 / 10

def total_hats : ℕ := weeks * days_per_week
def odd_days : ℕ := total_hats / 2
def even_days : ℕ := total_hats / 2

def total_cost : ℕ := odd_days * odd_day_price + even_days * even_day_price
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

theorem john_hats_cost : 
  total_hats ≥ discount_threshold → discounted_cost = 6615 := by
  sorry

end NUMINAMATH_CALUDE_john_hats_cost_l240_24032


namespace NUMINAMATH_CALUDE_min_value_quadratic_l240_24055

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 + 10*x - 8*y + 34 ≥ -7 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 + 10*a - 8*b + 34 = -7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l240_24055


namespace NUMINAMATH_CALUDE_perimeter_after_cuts_l240_24093

/-- The perimeter of a square after cutting out shapes --/
theorem perimeter_after_cuts (initial_side : ℝ) (green_side : ℝ) : 
  initial_side = 10 → green_side = 2 → 
  (4 * initial_side) + (4 * green_side) = 44 := by
  sorry

#check perimeter_after_cuts

end NUMINAMATH_CALUDE_perimeter_after_cuts_l240_24093


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l240_24041

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The difference in production time between Machine A and Machine G -/
def time_difference : ℕ := 10

/-- The production rate of Machine G relative to Machine A -/
def g_to_a_ratio : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 6

theorem machine_a_production_rate :
  ∃ (machine_g_rate : ℚ) (time_g : ℚ),
    machine_g_rate = g_to_a_ratio * machine_a_rate ∧
    time_g * machine_g_rate = total_sprockets ∧
    (time_g + time_difference) * machine_a_rate = total_sprockets :=
by sorry

end NUMINAMATH_CALUDE_machine_a_production_rate_l240_24041


namespace NUMINAMATH_CALUDE_min_value_of_expression_l240_24012

theorem min_value_of_expression (x y : ℝ) : (x * y + 2)^2 + (x - y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l240_24012


namespace NUMINAMATH_CALUDE_expression_between_two_and_three_l240_24018

theorem expression_between_two_and_three (a b : ℝ) (h : 3 * a = 5 * b) :
  2 < |a + b| / b ∧ |a + b| / b < 3 := by sorry

end NUMINAMATH_CALUDE_expression_between_two_and_three_l240_24018


namespace NUMINAMATH_CALUDE_wild_weatherman_answers_l240_24062

/-- Represents the format of the text --/
inductive TextFormat
  | Interview
  | Diary
  | NewsStory
  | Announcement

/-- Represents Sam Champion's childhood career aspiration --/
inductive ChildhoodAspiration
  | SpaceScientist
  | Weatherman
  | NewsReporter
  | Meteorologist

/-- Represents the state of present weather forecasting technology --/
structure WeatherForecastingTechnology where
  moreExact : Bool
  stillImperfect : Bool

/-- Represents the name of the study of weather science --/
inductive WeatherScienceName
  | Meteorology
  | Forecasting
  | Geography
  | EarthScience

/-- The main theorem statement --/
theorem wild_weatherman_answers 
  (text_format : TextFormat)
  (sam_aspiration : ChildhoodAspiration)
  (forecast_tech : WeatherForecastingTechnology)
  (weather_science : WeatherScienceName) :
  text_format = TextFormat.Interview ∧
  sam_aspiration = ChildhoodAspiration.NewsReporter ∧
  forecast_tech.moreExact = true ∧
  forecast_tech.stillImperfect = true ∧
  weather_science = WeatherScienceName.Meteorology :=
by sorry

end NUMINAMATH_CALUDE_wild_weatherman_answers_l240_24062


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l240_24019

theorem imaginary_part_of_complex_division : 
  let z : ℂ := -3 + 4*I
  let w : ℂ := 1 + I
  (z / w).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l240_24019


namespace NUMINAMATH_CALUDE_shortest_path_length_l240_24077

/-- A regular octahedron with edge length 1 -/
structure RegularOctahedron where
  /-- The edge length of the octahedron is 1 -/
  edge_length : ℝ
  edge_length_eq : edge_length = 1

/-- A path on the surface of an octahedron -/
structure SurfacePath (o : RegularOctahedron) where
  /-- The length of the path -/
  length : ℝ
  /-- The path starts at a vertex -/
  starts_at_vertex : Bool
  /-- The path ends at the opposite vertex -/
  ends_at_opposite_vertex : Bool

/-- The theorem stating that the shortest path between opposite vertices has length 2 -/
theorem shortest_path_length (o : RegularOctahedron) : 
  ∃ (p : SurfacePath o), p.length = 2 ∧ 
  ∀ (q : SurfacePath o), q.starts_at_vertex ∧ q.ends_at_opposite_vertex → q.length ≥ p.length :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l240_24077


namespace NUMINAMATH_CALUDE_james_beef_purchase_l240_24092

/-- Proves that James bought 20 pounds of beef given the problem conditions -/
theorem james_beef_purchase :
  ∀ (beef pork : ℝ) (meals : ℕ),
    pork = beef / 2 →
    meals * 1.5 = beef + pork →
    meals * 20 = 400 →
    beef = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_james_beef_purchase_l240_24092


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l240_24031

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℕ, k * a = 48) → 
  (∃ l : ℕ, l * b = 48) → 
  ¬(∃ m : ℕ, m * (a * b) = 48) → 
  (∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → 
    (∃ k : ℕ, k * c = 48) → 
    (∃ l : ℕ, l * d = 48) → 
    ¬(∃ m : ℕ, m * (c * d) = 48) → 
    a * b ≤ c * d) → 
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l240_24031


namespace NUMINAMATH_CALUDE_extremum_implies_a_b_values_l240_24058

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≥ f 1) ∧
  f 1 = 10

/-- If f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1, then a = -4 and b = 11 -/
theorem extremum_implies_a_b_values :
  ∀ a b : ℝ, has_extremum a b → a = -4 ∧ b = 11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_b_values_l240_24058


namespace NUMINAMATH_CALUDE_correct_life_insights_l240_24030

/- Define the types of connections -/
inductive ConnectionType
  | Objective
  | Diverse
  | Inevitable
  | Conditional

/- Define the actions related to connections -/
inductive ConnectionAction
  | CannotAdjust
  | EstablishNew
  | EliminateAccidental
  | GraspConditions

/- Define a proposition that represents an insight about connections -/
structure ConnectionInsight where
  type : ConnectionType
  action : ConnectionAction

/- Define the function that determines if an insight is correct -/
def isCorrectInsight (insight : ConnectionInsight) : Prop :=
  (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
  (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions)

/- The theorem to prove -/
theorem correct_life_insights :
  ∀ (insight : ConnectionInsight),
    isCorrectInsight insight ↔
      (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
      (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions) :=
by sorry


end NUMINAMATH_CALUDE_correct_life_insights_l240_24030


namespace NUMINAMATH_CALUDE_transformed_sine_value_l240_24003

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem transformed_sine_value 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 ≤ φ ∧ φ < π/2) 
  (h_transform : ∀ x, Real.sin x = Real.sin (2 * ω * (x - π/6) + φ)) :
  f ω φ (π/6) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_transformed_sine_value_l240_24003


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l240_24068

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2/(4a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + (-b^2) / (4 * a)
  ∃ x₀, ∀ x, f x₀ ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l240_24068


namespace NUMINAMATH_CALUDE_circle_area_difference_l240_24080

/-- The difference in area between a circle with radius 30 inches and a circle with circumference 60π inches is 0 square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 60 * Real.pi
  let r2 : ℝ := c2 / (2 * Real.pi)
  let area1 : ℝ := Real.pi * r1^2
  let area2 : ℝ := Real.pi * r2^2
  area1 - area2 = 0 := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l240_24080


namespace NUMINAMATH_CALUDE_soccer_team_points_l240_24074

theorem soccer_team_points : ∀ (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ),
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  draws = total_games - wins - losses →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss = 46 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_points_l240_24074


namespace NUMINAMATH_CALUDE_fraction_inequality_l240_24039

theorem fraction_inequality (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (h1 : 1 / a > 1 / b) (h2 : x > y) : 
  x / (x + a) > y / (y + b) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l240_24039


namespace NUMINAMATH_CALUDE_power_division_rule_l240_24037

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end NUMINAMATH_CALUDE_power_division_rule_l240_24037


namespace NUMINAMATH_CALUDE_platform_length_l240_24082

/-- Given a train crossing a platform, calculate the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 225) 
  (h2 : train_speed_kmph = 90) 
  (h3 : crossing_time = 25) : 
  ℝ := by
  
  -- Convert train speed from km/h to m/s
  let train_speed_ms := train_speed_kmph * 1000 / 3600

  -- Calculate total distance covered (train + platform)
  let total_distance := train_speed_ms * crossing_time

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that the platform length is 400 meters
  have : platform_length = 400 := by sorry

  -- Return the platform length
  exact platform_length


end NUMINAMATH_CALUDE_platform_length_l240_24082


namespace NUMINAMATH_CALUDE_elephant_weighing_l240_24099

/-- The weight of a stone block in catties -/
def stone_weight : ℕ := 240

/-- The number of stone blocks initially on the boat -/
def initial_stones : ℕ := 20

/-- The number of workers initially on the boat -/
def initial_workers : ℕ := 3

/-- The number of stone blocks after adjustment -/
def adjusted_stones : ℕ := 21

/-- The number of workers after adjustment -/
def adjusted_workers : ℕ := 1

/-- The weight of the elephant in catties -/
def elephant_weight : ℕ := 5160

theorem elephant_weighing :
  ∃ (worker_weight : ℕ),
    (initial_stones * stone_weight + initial_workers * worker_weight =
     adjusted_stones * stone_weight + adjusted_workers * worker_weight) ∧
    (elephant_weight = initial_stones * stone_weight + initial_workers * worker_weight) :=
by sorry

end NUMINAMATH_CALUDE_elephant_weighing_l240_24099


namespace NUMINAMATH_CALUDE_removal_ways_count_l240_24014

/-- Represents a block in the stack -/
structure Block where
  layer : Nat
  exposed : Bool

/-- Represents the stack of blocks -/
def Stack : Type := List Block

/-- The initial stack configuration -/
def initialStack : Stack := sorry

/-- Function to check if a block can be removed -/
def canRemove (b : Block) (s : Stack) : Bool := sorry

/-- Function to remove a block and update the stack -/
def removeBlock (b : Block) (s : Stack) : Stack := sorry

/-- Function to count the number of ways to remove 5 blocks -/
def countRemovalWays (s : Stack) : Nat := sorry

/-- The main theorem stating the number of ways to remove 5 blocks -/
theorem removal_ways_count : 
  countRemovalWays initialStack = 3384 := by sorry

end NUMINAMATH_CALUDE_removal_ways_count_l240_24014


namespace NUMINAMATH_CALUDE_owls_on_fence_l240_24054

/-- The number of owls on a fence after more owls join is the sum of the initial number and the number that joined. -/
theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  let total_owls := initial_owls + joining_owls
  total_owls = initial_owls + joining_owls :=
by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l240_24054


namespace NUMINAMATH_CALUDE_median_of_special_list_l240_24024

/-- Represents the special list where each number n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def list_length : ℕ := (List.range 100).sum + 100

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem median_of_special_list : median special_list = 71 := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l240_24024


namespace NUMINAMATH_CALUDE_derivative_at_zero_l240_24044

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan ((3 * x / 2) - x^2 * Real.sin (1 / x))
  else 0

-- State the theorem
theorem derivative_at_zero (h : HasDerivAt f (3/2) 0) : 
  deriv f 0 = 3/2 := by sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l240_24044


namespace NUMINAMATH_CALUDE_resistance_value_l240_24076

/-- Given two identical resistors connected in series to a DC voltage source,
    prove that the resistance of each resistor is 2 Ω based on voltmeter and ammeter readings. -/
theorem resistance_value (R U Uv IA : ℝ) : 
  Uv = 10 →  -- Voltmeter reading
  IA = 10 →  -- Ammeter reading
  U = 2 * Uv →  -- Total voltage
  U = R * IA →  -- Ohm's law for the circuit with ammeter
  R = 2 := by
  sorry

end NUMINAMATH_CALUDE_resistance_value_l240_24076


namespace NUMINAMATH_CALUDE_power_sum_l240_24027

theorem power_sum (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 2) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l240_24027


namespace NUMINAMATH_CALUDE_systematic_sample_fifth_seat_l240_24004

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 4 → ℕ
  (class_size_pos : class_size > 0)
  (sample_size_pos : sample_size > 0)
  (sample_size_le_class : sample_size ≤ class_size)
  (known_seats_valid : ∀ i, known_seats i ≤ class_size)
  (known_seats_ordered : ∀ i j, i < j → known_seats i < known_seats j)

/-- The theorem to be proved -/
theorem systematic_sample_fifth_seat
  (s : SystematicSample)
  (h1 : s.class_size = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.known_seats 0 = 3)
  (h4 : s.known_seats 1 = 15)
  (h5 : s.known_seats 2 = 39)
  (h6 : s.known_seats 3 = 51) :
  ∃ (fifth_seat : ℕ), fifth_seat = 27 ∧
    (∀ i j, i ≠ j → s.known_seats i ≠ fifth_seat) ∧
    fifth_seat ≤ s.class_size :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_fifth_seat_l240_24004


namespace NUMINAMATH_CALUDE_victor_initial_books_l240_24097

/-- The number of books Victor had initially -/
def initial_books : ℕ := sorry

/-- The number of books Victor bought during the book fair -/
def bought_books : ℕ := 3

/-- The total number of books Victor had after buying more -/
def total_books : ℕ := 12

/-- Theorem stating that Victor initially had 9 books -/
theorem victor_initial_books : 
  initial_books + bought_books = total_books → initial_books = 9 := by
  sorry

end NUMINAMATH_CALUDE_victor_initial_books_l240_24097


namespace NUMINAMATH_CALUDE_product_xy_l240_24011

theorem product_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_l240_24011


namespace NUMINAMATH_CALUDE_train_length_calculation_l240_24006

theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 95) (h2 : v2 = 85) (h3 : t = 6) :
  let relative_speed := (v1 + v2) * (5/18)
  let total_length := relative_speed * t
  let train_length := total_length / 2
  train_length = 150 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l240_24006


namespace NUMINAMATH_CALUDE_student_mistake_difference_l240_24005

theorem student_mistake_difference : (5/6 : ℚ) * 96 - (5/16 : ℚ) * 96 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_mistake_difference_l240_24005


namespace NUMINAMATH_CALUDE_at_least_one_square_l240_24075

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : Nat
  height : Nat
  width_gt_one : width > 1
  height_gt_one : height > 1

/-- Represents a division of a square into rectangles -/
structure SquareDivision where
  side_length : Nat
  rectangles : List Rectangle
  total_rectangles : rectangles.length = 17
  covers_square : (rectangles.map (λ r => r.width * r.height)).sum = side_length * side_length

theorem at_least_one_square (d : SquareDivision) (h : d.side_length = 10) :
  ∃ (r : Rectangle), r ∈ d.rectangles ∧ r.width = r.height := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_square_l240_24075


namespace NUMINAMATH_CALUDE_shiela_neighbors_l240_24023

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  (h3 : total_drawings % drawings_per_neighbor = 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end NUMINAMATH_CALUDE_shiela_neighbors_l240_24023


namespace NUMINAMATH_CALUDE_inverse_proportional_point_l240_24065

theorem inverse_proportional_point :
  let f : ℝ → ℝ := λ x => 6 / x
  f (-2) = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportional_point_l240_24065


namespace NUMINAMATH_CALUDE_cube_structure_extension_l240_24061

/-- Represents a cube structure with a central cube and attached cubes -/
structure CubeStructure :=
  (central : ℕ)
  (attached : ℕ)

/-- The number of cubes in the initial structure -/
def initial_cubes (s : CubeStructure) : ℕ := s.central + s.attached

/-- The number of exposed faces in the initial structure -/
def exposed_faces (s : CubeStructure) : ℕ := s.attached * 5

/-- The number of extra cubes needed for the extended structure -/
def extra_cubes_needed (s : CubeStructure) : ℕ := 12 + 6

theorem cube_structure_extension (s : CubeStructure) 
  (h1 : s.central = 1) 
  (h2 : s.attached = 6) : 
  extra_cubes_needed s = 18 := by sorry

end NUMINAMATH_CALUDE_cube_structure_extension_l240_24061


namespace NUMINAMATH_CALUDE_prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l240_24015

-- Define the success rates and scores
def triple_jump_success_rate : ℝ := 0.7
def quadruple_jump_success_rate : ℝ := 0.3
def successful_triple_jump_score : ℕ := 8
def failed_triple_jump_score : ℕ := 4
def successful_quadruple_jump_score : ℕ := 15
def failed_quadruple_jump_score : ℕ := 6

-- Define the probability of score exceeding 14 points for triple jump followed by quadruple jump
def prob_score_exceeds_14 : ℝ := 
  triple_jump_success_rate * quadruple_jump_success_rate + 
  (1 - triple_jump_success_rate) * quadruple_jump_success_rate

-- Define the expected value of score for two consecutive triple jumps
def expected_value_two_triple_jumps : ℝ := 
  (1 - triple_jump_success_rate)^2 * (2 * failed_triple_jump_score) +
  2 * triple_jump_success_rate * (1 - triple_jump_success_rate) * (successful_triple_jump_score + failed_triple_jump_score) +
  triple_jump_success_rate^2 * (2 * successful_triple_jump_score)

-- Theorem statements
theorem prob_score_exceeds_14_is_0_3 : 
  prob_score_exceeds_14 = 0.3 := by sorry

theorem expected_value_two_triple_jumps_is_13_6 : 
  expected_value_two_triple_jumps = 13.6 := by sorry

end NUMINAMATH_CALUDE_prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l240_24015


namespace NUMINAMATH_CALUDE_current_babysitter_rate_is_16_l240_24050

/-- Represents the babysitting scenario with given conditions -/
structure BabysittingScenario where
  new_hourly_rate : ℕ
  scream_charge : ℕ
  hours : ℕ
  scream_count : ℕ
  cost_difference : ℕ

/-- Calculates the hourly rate of the current babysitter -/
def current_babysitter_rate (scenario : BabysittingScenario) : ℕ :=
  ((scenario.new_hourly_rate * scenario.hours + scenario.scream_charge * scenario.scream_count) + scenario.cost_difference) / scenario.hours

/-- Theorem stating that given the conditions, the current babysitter's hourly rate is $16 -/
theorem current_babysitter_rate_is_16 (scenario : BabysittingScenario) 
    (h1 : scenario.new_hourly_rate = 12)
    (h2 : scenario.scream_charge = 3)
    (h3 : scenario.hours = 6)
    (h4 : scenario.scream_count = 2)
    (h5 : scenario.cost_difference = 18) :
  current_babysitter_rate scenario = 16 := by
  sorry

#eval current_babysitter_rate { new_hourly_rate := 12, scream_charge := 3, hours := 6, scream_count := 2, cost_difference := 18 }

end NUMINAMATH_CALUDE_current_babysitter_rate_is_16_l240_24050


namespace NUMINAMATH_CALUDE_complex_number_modulus_l240_24060

theorem complex_number_modulus (a : ℝ) (i : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  Complex.abs (a * i / (1 + 2 * i)) = Real.sqrt 5 → 
  a = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l240_24060


namespace NUMINAMATH_CALUDE_specific_prism_volume_l240_24022

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  -- Radius of the sphere
  R : ℝ
  -- Length of AD
  AD : ℝ
  -- Assertion that CD is a diameter
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem specific_prism_volume :
  let p : InscribedPrism := {
    R := 6,
    AD := 4 * Real.sqrt 6,
    is_diameter := true
  }
  prism_volume p = 48 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_volume_l240_24022


namespace NUMINAMATH_CALUDE_class_ratio_proof_l240_24064

theorem class_ratio_proof (eduardo_classes : ℕ) (total_classes : ℕ) 
  (h1 : eduardo_classes = 3)
  (h2 : total_classes = 9) :
  (total_classes - eduardo_classes) / eduardo_classes = 2 := by
sorry

end NUMINAMATH_CALUDE_class_ratio_proof_l240_24064


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l240_24078

/-- The universal set U -/
def U : Set Int := {-2, -1, 0, 1, 2, 3}

/-- Set A -/
def A : Set Int := {-1, 2}

/-- Set B -/
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

/-- The main theorem -/
theorem complement_of_union_equals_set (h : U = {-2, -1, 0, 1, 2, 3}) :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l240_24078


namespace NUMINAMATH_CALUDE_engine_system_theorems_l240_24096

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  x : ℝ  -- length of PQ

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- 1. Sine rule relation
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- 2. Maximum value of sin β
  (∃ (max_sin_β : ℝ), max_sin_β = sys.a / sys.b ∧
    ∀ β', Real.sin β' ≤ max_sin_β) ∧
  -- 3. Relation for x
  sys.x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) :=
by sorry

end NUMINAMATH_CALUDE_engine_system_theorems_l240_24096


namespace NUMINAMATH_CALUDE_raccoon_lock_ratio_l240_24034

/-- Proves that the ratio of time both locks stall raccoons to time second lock alone stalls raccoons is 5 -/
theorem raccoon_lock_ratio : 
  let first_lock_time : ℕ := 5
  let second_lock_time : ℕ := 3 * first_lock_time - 3
  let both_locks_time : ℕ := 60
  both_locks_time / second_lock_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_ratio_l240_24034


namespace NUMINAMATH_CALUDE_exam_score_problem_l240_24095

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l240_24095


namespace NUMINAMATH_CALUDE_driver_net_rate_of_pay_l240_24088

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_rate_of_pay
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_rate = 0.60)
  (h5 : gas_price = 2.50)
  : (pay_rate * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_rate_of_pay_l240_24088


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l240_24089

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {2, 3, 4, 5}
def N : Set Nat := {1, 4, 5, 7}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l240_24089


namespace NUMINAMATH_CALUDE_marble_sculpture_weight_l240_24047

theorem marble_sculpture_weight (W : ℝ) : 
  W > 0 →
  (1 - 0.3) * (1 - 0.2) * (1 - 0.25) * W = 105 →
  W = 250 := by
sorry

end NUMINAMATH_CALUDE_marble_sculpture_weight_l240_24047


namespace NUMINAMATH_CALUDE_deductive_reasoning_is_general_to_specific_l240_24017

/-- Represents a form of reasoning --/
inductive ReasoningForm
  | GeneralToSpecific
  | SpecificToGeneral
  | GeneralToGeneral
  | SpecificToSpecific

/-- Definition of deductive reasoning --/
def deductive_reasoning : ReasoningForm := ReasoningForm.GeneralToSpecific

/-- Theorem stating that deductive reasoning is from general to specific --/
theorem deductive_reasoning_is_general_to_specific :
  deductive_reasoning = ReasoningForm.GeneralToSpecific := by sorry

end NUMINAMATH_CALUDE_deductive_reasoning_is_general_to_specific_l240_24017


namespace NUMINAMATH_CALUDE_product_equality_implies_composite_sums_l240_24056

theorem product_equality_implies_composite_sums (a b c d : ℕ) (h : a * b = c * d) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + b + c + d = x * y) ∧
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a^2 + b^2 + c^2 + d^2 = x * y) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_implies_composite_sums_l240_24056


namespace NUMINAMATH_CALUDE_problem_statement_l240_24069

-- Define the proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x| - a * x

-- Define the proposition q
def q : Prop := ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x = f a (-x)

-- State the theorem
theorem problem_statement : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l240_24069


namespace NUMINAMATH_CALUDE_binomial_expansion_degree_l240_24084

theorem binomial_expansion_degree (n : ℕ) :
  (∀ x, (1 + x)^n = 1 + 6*x + 15*x^2 + 20*x^3 + 15*x^4 + 6*x^5 + x^6) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_degree_l240_24084


namespace NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l240_24033

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_three_distinct_roots_l240_24033


namespace NUMINAMATH_CALUDE_thousandth_digit_is_three_l240_24072

/-- The sequence of digits obtained by concatenating integers from 1 to 499 -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => if n + 1 < 499 then digit_sequence n * 10 + (n + 2) else digit_sequence n

/-- The nth digit in the sequence -/
def nth_digit (n : ℕ) : ℕ :=
  (digit_sequence (n / 9) / (10 ^ (n % 9))) % 10

/-- Theorem stating that the 1000th digit is 3 -/
theorem thousandth_digit_is_three : nth_digit 999 = 3 := by
  sorry

end NUMINAMATH_CALUDE_thousandth_digit_is_three_l240_24072


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l240_24029

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team plays every other team 12 times,
    the total number of games played in the season is 3600. -/
theorem hockey_league_season_games :
  hockey_league_games 25 12 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l240_24029


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_least_people_l240_24079

/-- Cost function for the first caterer -/
def cost1 (n : ℕ) : ℚ := 120 + 18 * n

/-- Cost function for the second caterer -/
def cost2 (n : ℕ) : ℚ := 250 + 15 * n

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 44

theorem second_caterer_cheaper_at_least_people :
  cost2 least_people < cost1 least_people ∧
  cost1 (least_people - 1) ≤ cost2 (least_people - 1) :=
by sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_least_people_l240_24079


namespace NUMINAMATH_CALUDE_sum_of_digits_cube_n_nines_l240_24066

/-- The sum of digits function for natural numbers -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function that returns a number composed of n nines -/
def n_nines (n : ℕ) : ℕ := 10^n - 1

theorem sum_of_digits_cube_n_nines (n : ℕ) :
  sum_of_digits ((n_nines n)^3) = 18 * n := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_cube_n_nines_l240_24066


namespace NUMINAMATH_CALUDE_expression_simplification_l240_24013

theorem expression_simplification (x y : ℝ) 
  (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2*y) - (x + y)^2) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l240_24013


namespace NUMINAMATH_CALUDE_solution_product_l240_24020

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 8) = r^2 - 20 * r + 75 →
  (s - 3) * (3 * s + 8) = s^2 - 20 * s + 75 →
  r ≠ s →
  (r + 4) * (s + 4) = -119/2 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l240_24020


namespace NUMINAMATH_CALUDE_red_mailbox_houses_l240_24098

/-- Proves the number of houses with red mailboxes given the total junk mail,
    total houses, houses with white mailboxes, and junk mail per house. -/
theorem red_mailbox_houses
  (total_junk_mail : ℕ)
  (total_houses : ℕ)
  (white_mailbox_houses : ℕ)
  (junk_mail_per_house : ℕ)
  (h1 : total_junk_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_mailbox_houses = 2)
  (h4 : junk_mail_per_house = 6)
  : total_houses - white_mailbox_houses = 6 := by
  sorry

#check red_mailbox_houses

end NUMINAMATH_CALUDE_red_mailbox_houses_l240_24098


namespace NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l240_24083

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l240_24083


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l240_24046

/-- Given the cost of two varieties of rice and their mixing ratio, 
    calculate the cost of the second variety. -/
theorem rice_mixture_cost 
  (cost_first : ℝ) 
  (cost_mixture : ℝ) 
  (ratio : ℝ) 
  (h1 : cost_first = 5.5)
  (h2 : cost_mixture = 7.50)
  (h3 : ratio = 0.625) :
  ∃ (cost_second : ℝ), 
    cost_second = 10.7 ∧ 
    (cost_first - cost_mixture) / (cost_mixture - cost_second) = ratio / 1 :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l240_24046


namespace NUMINAMATH_CALUDE_area_of_integral_triangle_with_perimeter_12_l240_24016

/-- Represents a triangle with integral sides --/
structure IntegralTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of an integral triangle with perimeter 12 is 2√6 --/
theorem area_of_integral_triangle_with_perimeter_12 (t : IntegralTriangle) : 
  Real.sqrt (6 * (6 - t.a) * (6 - t.b) * (6 - t.c)) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_area_of_integral_triangle_with_perimeter_12_l240_24016


namespace NUMINAMATH_CALUDE_expression_simplification_l240_24042

theorem expression_simplification (b c x : ℝ) (hb : b ≠ 1) (hc : c ≠ 1) (hbc : b ≠ c) :
  (x + 1)^2 / ((1 - b) * (1 - c)) + (x + b)^2 / ((b - 1) * (b - c)) + (x + c)^2 / ((c - 1) * (c - b)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l240_24042


namespace NUMINAMATH_CALUDE_not_square_and_floor_sqrt_cube_divides_square_l240_24000

theorem not_square_and_floor_sqrt_cube_divides_square (n : ℕ) :
  (∀ k : ℕ, n ≠ k^2) →
  (Nat.floor (Real.sqrt n))^3 ∣ n^2 →
  n = 2 ∨ n = 3 ∨ n = 8 ∨ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_not_square_and_floor_sqrt_cube_divides_square_l240_24000


namespace NUMINAMATH_CALUDE_unique_arrangement_l240_24051

-- Define the containers and liquids as enumerated types
inductive Container : Type
| Cup : Container
| Glass : Container
| Jug : Container
| Jar : Container

inductive Liquid : Type
| Milk : Liquid
| Lemonade : Liquid
| Kvass : Liquid
| Water : Liquid

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr Container.Cup ≠ Liquid.Water ∧ arr Container.Cup ≠ Liquid.Milk) ∧
  (∃ c, (c = Container.Jug ∨ c = Container.Jar) ∧
        arr c = Liquid.Kvass ∧
        (arr Container.Cup = Liquid.Lemonade ∨
         arr Container.Glass = Liquid.Lemonade)) ∧
  (arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water) ∧
  ((arr Container.Glass = Liquid.Milk ∧ arr Container.Jug = Liquid.Milk) ∨
   (arr Container.Glass = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk) ∨
   (arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk))

-- Define the correct arrangement
def correctArrangement : Arrangement
| Container.Cup => Liquid.Lemonade
| Container.Glass => Liquid.Water
| Container.Jug => Liquid.Milk
| Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement), satisfiesConditions arr → arr = correctArrangement :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l240_24051


namespace NUMINAMATH_CALUDE_quadratic_factorization_l240_24087

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l240_24087


namespace NUMINAMATH_CALUDE_intersection_point_values_l240_24038

theorem intersection_point_values (m n : ℚ) : 
  (1 / 2 : ℚ) * 1 + n = -2 → -- y = x/2 + n at x = 1
  m * 1 - 1 = -2 →          -- y = mx - 1 at x = 1
  m = -1 ∧ n = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_values_l240_24038


namespace NUMINAMATH_CALUDE_smallest_coverage_l240_24081

/-- Represents a checkerboard configuration -/
structure CheckerBoard :=
  (rows : Nat)
  (cols : Nat)
  (checkers : Nat)
  (at_most_one_per_square : checkers ≤ rows * cols)

/-- Defines the coverage property for a given k -/
def covers (board : CheckerBoard) (k : Nat) : Prop :=
  ∀ (arrangement : Fin board.checkers → Fin board.rows × Fin board.cols),
    ∃ (rows : Fin k → Fin board.rows) (cols : Fin k → Fin board.cols),
      ∀ (c : Fin board.checkers),
        (arrangement c).1 ∈ Set.range rows ∨ (arrangement c).2 ∈ Set.range cols

/-- The main theorem statement -/
theorem smallest_coverage (board : CheckerBoard) 
  (h_rows : board.rows = 2011)
  (h_cols : board.cols = 2011)
  (h_checkers : board.checkers = 3000) :
  (covers board 1006 ∧ ∀ k < 1006, ¬covers board k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_coverage_l240_24081


namespace NUMINAMATH_CALUDE_programmers_remote_work_cycle_l240_24091

def alex_cycle : ℕ := 5
def brooke_cycle : ℕ := 3
def charlie_cycle : ℕ := 8
def dana_cycle : ℕ := 9

theorem programmers_remote_work_cycle : 
  Nat.lcm alex_cycle (Nat.lcm brooke_cycle (Nat.lcm charlie_cycle dana_cycle)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_programmers_remote_work_cycle_l240_24091


namespace NUMINAMATH_CALUDE_total_addresses_l240_24052

/-- The number of commencement addresses given by each governor -/
structure GovernorAddresses where
  sandoval : ℕ
  hawkins : ℕ
  sloan : ℕ
  davenport : ℕ
  adkins : ℕ

/-- The conditions of the problem -/
def problem_conditions (g : GovernorAddresses) : Prop :=
  g.sandoval = 12 ∧
  g.hawkins = g.sandoval / 2 ∧
  g.sloan = g.sandoval + 10 ∧
  g.davenport = (g.sandoval + g.sloan) / 2 - 3 ∧
  g.adkins = g.hawkins + g.davenport + 2

/-- The theorem to be proved -/
theorem total_addresses (g : GovernorAddresses) :
  problem_conditions g →
  g.sandoval + g.hawkins + g.sloan + g.davenport + g.adkins = 70 :=
by sorry

end NUMINAMATH_CALUDE_total_addresses_l240_24052


namespace NUMINAMATH_CALUDE_thirteen_bead_necklace_l240_24057

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def arrangements (n : ℕ) : ℕ :=
  fibonacci (n + 2) - fibonacci (n - 2)

def circular_arrangements (n : ℕ) : ℕ :=
  (arrangements n - 1) / n + 1

theorem thirteen_bead_necklace :
  circular_arrangements 13 = 41 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_bead_necklace_l240_24057


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l240_24009

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 + x = 0 ∧ x ≠ -1) ∧
  (∀ x : ℝ, x = -1 → x^2 + x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l240_24009


namespace NUMINAMATH_CALUDE_max_value_of_sine_plus_one_l240_24002

theorem max_value_of_sine_plus_one :
  ∀ x : ℝ, 1 + Real.sin x ≤ 2 ∧ ∃ x : ℝ, 1 + Real.sin x = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sine_plus_one_l240_24002


namespace NUMINAMATH_CALUDE_equal_roots_when_m_is_negative_half_l240_24040

theorem equal_roots_when_m_is_negative_half :
  let f (x m : ℝ) := (x * (x - 1) - (m^2 + m*x + 1)) / ((x - 1) * (m - 1)) - x / m
  ∀ x₁ x₂ : ℝ, f x₁ (-1/2) = 0 → f x₂ (-1/2) = 0 → x₁ = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_when_m_is_negative_half_l240_24040


namespace NUMINAMATH_CALUDE_intersection_M_N_l240_24043

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

theorem intersection_M_N : M ∩ N = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l240_24043


namespace NUMINAMATH_CALUDE_problem_1_solution_l240_24059

theorem problem_1_solution (x : ℝ) : 
  (2 / (x - 3) = 1 / x) ↔ (x = -3) :=
sorry

end NUMINAMATH_CALUDE_problem_1_solution_l240_24059


namespace NUMINAMATH_CALUDE_pool_capacity_l240_24045

theorem pool_capacity (C : ℝ) (h1 : C > 0) : 
  (0.4 * C + 300 = 0.8 * C) → C = 750 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l240_24045


namespace NUMINAMATH_CALUDE_irrational_pi_among_options_l240_24053

theorem irrational_pi_among_options : 
  (∃ (a b : ℤ), (3.142 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (Real.sqrt 4 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (22 / 7 : ℝ) = a / b) ∧ 
  (¬ ∃ (a b : ℤ), (Real.pi : ℝ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_pi_among_options_l240_24053


namespace NUMINAMATH_CALUDE_sector_central_angle_l240_24073

theorem sector_central_angle (area : Real) (radius : Real) (centralAngle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  centralAngle = area * 2 / (radius ^ 2) →
  centralAngle = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l240_24073


namespace NUMINAMATH_CALUDE_antonov_candy_packs_l240_24008

/-- Given a total number of candies and packs, calculate the number of candies per pack -/
def candies_per_pack (total_candies : ℕ) (total_packs : ℕ) : ℕ :=
  total_candies / total_packs

/-- Theorem: The number of candies per pack is 20 -/
theorem antonov_candy_packs : candies_per_pack 60 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_antonov_candy_packs_l240_24008


namespace NUMINAMATH_CALUDE_elevator_unreachable_l240_24090

def is_valid_floor (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 15

def elevator_move (n : ℤ) : ℤ → ℤ
  | 0 => n  -- base case: no moves
  | 1 => n + 7  -- move up 7 floors
  | -1 => n - 9  -- move down 9 floors
  | _ => n  -- invalid move, stay on the same floor

def can_reach (start finish : ℤ) : Prop :=
  ∃ (moves : List ℤ), 
    (∀ m ∈ moves, m = 1 ∨ m = -1) ∧
    (List.foldl elevator_move start moves = finish) ∧
    (∀ i, is_valid_floor (List.foldl elevator_move start (moves.take i)))

theorem elevator_unreachable :
  ¬(can_reach 3 12) :=
sorry

end NUMINAMATH_CALUDE_elevator_unreachable_l240_24090


namespace NUMINAMATH_CALUDE_no_valid_solution_l240_24048

-- Define the equation
def equation (x : ℝ) : Prop :=
  (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x))

-- Theorem stating that there is no valid solution
theorem no_valid_solution : ¬∃ (x : ℝ), x ≥ 0 ∧ equation x :=
sorry

end NUMINAMATH_CALUDE_no_valid_solution_l240_24048


namespace NUMINAMATH_CALUDE_percentage_of_sum_l240_24028

theorem percentage_of_sum (x y : ℝ) (P : ℝ) :
  (0.6 * (x - y) = (P / 100) * (x + y)) →
  (y = (1 / 3) * x) →
  P = 45 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l240_24028


namespace NUMINAMATH_CALUDE_cookie_sales_revenue_l240_24036

theorem cookie_sales_revenue : 
  let chocolate_cookies : ℕ := 220
  let vanilla_cookies : ℕ := 70
  let chocolate_price : ℚ := 1
  let vanilla_price : ℚ := 2
  let chocolate_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  
  let chocolate_revenue := chocolate_cookies * chocolate_price
  let chocolate_discount_amount := chocolate_revenue * chocolate_discount
  let discounted_chocolate_revenue := chocolate_revenue - chocolate_discount_amount
  let vanilla_revenue := vanilla_cookies * vanilla_price
  let total_revenue_before_tax := discounted_chocolate_revenue + vanilla_revenue
  let sales_tax := total_revenue_before_tax * sales_tax_rate
  let total_revenue_after_tax := total_revenue_before_tax + sales_tax
  
  total_revenue_after_tax = 354.90 := by sorry

end NUMINAMATH_CALUDE_cookie_sales_revenue_l240_24036


namespace NUMINAMATH_CALUDE_profit_equation_l240_24035

/-- Represents the profit equation for a product with given cost and selling prices,
    initial quantity sold, and price reduction effects. -/
theorem profit_equation
  (cost_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_quantity : ℝ)
  (additional_units_per_reduction : ℝ)
  (target_profit : ℝ)
  (h1 : cost_price = 40)
  (h2 : initial_selling_price = 60)
  (h3 : initial_quantity = 200)
  (h4 : additional_units_per_reduction = 8)
  (h5 : target_profit = 8450)
  (x : ℝ) :
  (initial_selling_price - cost_price - x) * (initial_quantity + additional_units_per_reduction * x) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_l240_24035


namespace NUMINAMATH_CALUDE_equation_to_parabola_l240_24007

/-- The equation y^4 - 16x^2 = 2y^2 - 64 can be transformed into a parabolic form -/
theorem equation_to_parabola :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    y^4 - 16*x^2 = 2*y^2 - 64 →
    ∃ (t : ℝ), y^2 = a*x + b*t + c :=
by sorry

end NUMINAMATH_CALUDE_equation_to_parabola_l240_24007


namespace NUMINAMATH_CALUDE_exam_pass_probability_l240_24094

theorem exam_pass_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 2/3) 
  (h_B : p_B = 3/4) 
  (h_C : p_C = 2/5) : 
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l240_24094


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l240_24063

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 6) ↔ ((-3 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 9)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l240_24063


namespace NUMINAMATH_CALUDE_equation_condition_l240_24026

theorem equation_condition (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (20 * a + b) * (20 * a + c) = 400 * a * (a + 1) + 10 * b * c →
  b + c = 20 :=
sorry

end NUMINAMATH_CALUDE_equation_condition_l240_24026


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l240_24010

theorem inscribed_circles_radii_sum (D : ℝ) (r₁ r₂ : ℝ) : 
  D = 23 → r₁ > 0 → r₂ > 0 → r₁ + r₂ = D / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l240_24010
