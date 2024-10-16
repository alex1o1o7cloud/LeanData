import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l688_68827

theorem sum_of_number_and_its_square (x : ℝ) : x = 18 → x + x^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l688_68827


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l688_68831

/-- Given a parabola and a hyperbola with specific properties, prove that the eccentricity of the hyperbola is 1 + √2 -/
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 2*p*x) 
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) 
  (focus_shared : ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
    F.1^2/a^2 + F.2^2/b^2 = (a^2 + b^2)/a^2) 
  (intersection_line : ∃ I₁ I₂ : ℝ × ℝ, 
    parabola I₁.1 I₁.2 ∧ hyperbola I₁.1 I₁.2 ∧ 
    parabola I₂.1 I₂.2 ∧ hyperbola I₂.1 I₂.2 ∧ 
    (I₂.2 - I₁.2) * (p/2 - I₁.1) = (I₂.1 - I₁.1) * (0 - I₁.2)) :
  (a^2 + b^2)/a^2 = (1 + Real.sqrt 2)^2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l688_68831


namespace NUMINAMATH_CALUDE_evening_temp_calculation_l688_68867

/-- Given a noon temperature and a temperature decrease, calculate the evening temperature. -/
def evening_temperature (noon_temp : Int) (decrease : Int) : Int :=
  noon_temp - decrease

/-- Theorem: If the noon temperature is 1℃ and it decreases by 3℃, then the evening temperature is -2℃. -/
theorem evening_temp_calculation :
  evening_temperature 1 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_evening_temp_calculation_l688_68867


namespace NUMINAMATH_CALUDE_positive_A_value_l688_68851

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + 3*B^2

-- Theorem statement
theorem positive_A_value :
  ∃ A : ℝ, A > 0 ∧ hash A 6 = 270 ∧ A = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l688_68851


namespace NUMINAMATH_CALUDE_spaceship_break_time_l688_68844

/-- Represents the travel pattern of a spaceship -/
structure TravelPattern where
  initialTravel : ℕ
  initialBreak : ℕ
  secondTravel : ℕ
  secondBreak : ℕ
  subsequentTravel : ℕ
  subsequentBreak : ℕ

/-- Calculates the total break time for a spaceship journey -/
def calculateBreakTime (pattern : TravelPattern) (totalJourneyTime : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for the given travel pattern and journey time, 
    the total break time is 8 hours -/
theorem spaceship_break_time :
  let pattern : TravelPattern := {
    initialTravel := 10,
    initialBreak := 3,
    secondTravel := 10,
    secondBreak := 1,
    subsequentTravel := 11,
    subsequentBreak := 1
  }
  let totalJourneyTime : ℕ := 72
  calculateBreakTime pattern totalJourneyTime = 8 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_break_time_l688_68844


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l688_68889

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l688_68889


namespace NUMINAMATH_CALUDE_highest_validity_rate_is_91_percent_l688_68816

/-- Represents the voting results for three candidates -/
structure VotingResult where
  total_ballots : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Calculates the highest possible validity rate for a given voting result -/
def highest_validity_rate (result : VotingResult) : ℚ :=
  let total_votes := result.votes_a + result.votes_b + result.votes_c
  let invalid_votes := total_votes - 2 * result.total_ballots
  (result.total_ballots - invalid_votes : ℚ) / result.total_ballots

/-- The main theorem stating the highest possible validity rate -/
theorem highest_validity_rate_is_91_percent (result : VotingResult) :
  result.total_ballots = 100 ∧
  result.votes_a = 88 ∧
  result.votes_b = 75 ∧
  result.votes_c = 46 →
  highest_validity_rate result = 91 / 100 :=
by sorry

#eval highest_validity_rate ⟨100, 88, 75, 46⟩

end NUMINAMATH_CALUDE_highest_validity_rate_is_91_percent_l688_68816


namespace NUMINAMATH_CALUDE_initial_disappearance_percentage_l688_68854

/-- Represents the population changes in a village --/
def village_population (initial_population : ℕ) (final_population : ℕ) (panic_exodus_percent : ℚ) : Prop :=
  ∃ (initial_disappearance_percent : ℚ),
    final_population = initial_population * (1 - initial_disappearance_percent / 100) * (1 - panic_exodus_percent / 100) ∧
    initial_disappearance_percent = 10

/-- Theorem stating the initial disappearance percentage in the village --/
theorem initial_disappearance_percentage :
  village_population 7800 5265 25 := by sorry

end NUMINAMATH_CALUDE_initial_disappearance_percentage_l688_68854


namespace NUMINAMATH_CALUDE_game_cost_l688_68818

/-- Given Frank's lawn mowing earnings, expenses, and game purchasing ability, prove the cost of each game. -/
theorem game_cost (total_earned : ℕ) (spent : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 19)
  (h2 : spent = 11)
  (h3 : num_games = 4)
  (h4 : ∃ (cost : ℕ), (total_earned - spent) = num_games * cost) :
  ∃ (cost : ℕ), cost = 2 ∧ (total_earned - spent) = num_games * cost := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l688_68818


namespace NUMINAMATH_CALUDE_product_equals_243_l688_68824

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l688_68824


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l688_68874

/-- Represents the dimensions of a rectangular garden --/
structure GardenDimensions where
  perpendicular_side : ℝ
  parallel_side : ℝ

/-- Calculates the area of the garden given its dimensions --/
def garden_area (d : GardenDimensions) : ℝ :=
  d.perpendicular_side * d.parallel_side

/-- Represents the constraints of the garden problem --/
structure GardenConstraints where
  wall_length : ℝ
  fence_cost_per_foot : ℝ
  total_fence_cost : ℝ

/-- Theorem stating that the optimal garden dimensions maximize the area --/
theorem optimal_garden_dimensions (c : GardenConstraints)
  (h1 : c.wall_length = 300)
  (h2 : c.fence_cost_per_foot = 10)
  (h3 : c.total_fence_cost = 1500) :
  ∃ (d : GardenDimensions),
    d.parallel_side = 75 ∧
    ∀ (d' : GardenDimensions),
      d'.perpendicular_side + d'.perpendicular_side + d'.parallel_side = c.total_fence_cost / c.fence_cost_per_foot →
      garden_area d ≥ garden_area d' :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l688_68874


namespace NUMINAMATH_CALUDE_miss_at_least_once_probability_l688_68852

/-- The probability of missing a target at least once in three shots -/
def miss_at_least_once (P : ℝ) : ℝ :=
  1 - P^3

theorem miss_at_least_once_probability (P : ℝ) 
  (h1 : 0 ≤ P) (h2 : P ≤ 1) : 
  miss_at_least_once P = 1 - P^3 := by
sorry

end NUMINAMATH_CALUDE_miss_at_least_once_probability_l688_68852


namespace NUMINAMATH_CALUDE_system_solution_fractional_equation_solution_l688_68859

-- System of equations
theorem system_solution :
  ∃ (x y : ℚ), 3 * x - 5 * y = 3 ∧ x / 2 - y / 3 = 1 ∧ x = 8 / 3 ∧ y = 1 := by sorry

-- Fractional equation
theorem fractional_equation_solution :
  ∃ (x : ℚ), x ≠ 1 ∧ x / (x - 1) + 1 = 3 / (2 * x - 2) ∧ x = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_system_solution_fractional_equation_solution_l688_68859


namespace NUMINAMATH_CALUDE_compare_exponentials_l688_68822

theorem compare_exponentials (h1 : 0 < 0.7) (h2 : 0.7 < 0.8) (h3 : 0.8 < 1) :
  0.8^0.7 > 0.7^0.7 ∧ 0.7^0.7 > 0.7^0.8 := by
  sorry

end NUMINAMATH_CALUDE_compare_exponentials_l688_68822


namespace NUMINAMATH_CALUDE_traffic_police_distribution_l688_68808

def officers : ℕ := 5
def specific_officers : ℕ := 2
def intersections : ℕ := 3

theorem traffic_police_distribution :
  (Nat.choose (officers - specific_officers + 1) (intersections - 1)) *
  (Nat.factorial intersections) = 36 := by sorry

end NUMINAMATH_CALUDE_traffic_police_distribution_l688_68808


namespace NUMINAMATH_CALUDE_not_all_projections_same_l688_68877

/-- Represents a 3D shape -/
inductive Shape
  | Cube
  | Sphere
  | Cone

/-- Represents a type of orthographic projection -/
inductive Projection
  | FrontView
  | SideView
  | TopView

/-- Represents the result of an orthographic projection -/
inductive ProjectionResult
  | Square
  | Circle
  | IsoscelesTriangle

/-- Returns the projection result for a given shape and projection type -/
def projectShape (s : Shape) (p : Projection) : ProjectionResult :=
  match s, p with
  | Shape.Cube, _ => ProjectionResult.Square
  | Shape.Sphere, _ => ProjectionResult.Circle
  | Shape.Cone, Projection.TopView => ProjectionResult.Circle
  | Shape.Cone, _ => ProjectionResult.IsoscelesTriangle

/-- Theorem stating that it's not true that all projections are the same for all shapes -/
theorem not_all_projections_same : ¬ (∀ (s1 s2 : Shape) (p1 p2 : Projection), 
  projectShape s1 p1 = projectShape s2 p2) := by
  sorry


end NUMINAMATH_CALUDE_not_all_projections_same_l688_68877


namespace NUMINAMATH_CALUDE_sin_2x_value_l688_68882

theorem sin_2x_value (x : ℝ) (h : Real.cos (x - π/4) = 4/5) : Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l688_68882


namespace NUMINAMATH_CALUDE_susie_rhode_island_reds_l688_68891

/-- The number of Rhode Island Reds that Susie has -/
def susie_reds : ℕ := sorry

/-- The number of Golden Comets that Susie has -/
def susie_comets : ℕ := 6

/-- The number of Rhode Island Reds that Britney has -/
def britney_reds : ℕ := 2 * susie_reds

/-- The number of Golden Comets that Britney has -/
def britney_comets : ℕ := susie_comets / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_reds + susie_comets

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_reds + britney_comets

theorem susie_rhode_island_reds :
  (britney_total = susie_total + 8) → susie_reds = 11 := by
  sorry

end NUMINAMATH_CALUDE_susie_rhode_island_reds_l688_68891


namespace NUMINAMATH_CALUDE_squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l688_68810

-- Define a type for the different problems
inductive Problem
  | oppositeNumber
  | squarePerimeter
  | maxOfThree
  | binaryToDecimal

-- Function to determine if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.oppositeNumber => False
  | Problem.squarePerimeter => False
  | Problem.maxOfThree => True
  | Problem.binaryToDecimal => True

-- Theorem stating that the square perimeter problem doesn't require conditional statements
theorem squarePerimeter_doesnt_require_conditional_statements :
  ¬(requiresConditionalStatements Problem.squarePerimeter) :=
by
  sorry

-- Theorem stating that the square perimeter problem is the only one among the four that doesn't require conditional statements
theorem only_squarePerimeter_doesnt_require_conditional_statements :
  ∀ (p : Problem), ¬(requiresConditionalStatements p) → p = Problem.squarePerimeter :=
by
  sorry

end NUMINAMATH_CALUDE_squarePerimeter_doesnt_require_conditional_statements_only_squarePerimeter_doesnt_require_conditional_statements_l688_68810


namespace NUMINAMATH_CALUDE_rational_set_not_just_positive_and_negative_l688_68809

theorem rational_set_not_just_positive_and_negative : 
  ∃ q : ℚ, q ∉ {x : ℚ | x > 0} ∪ {x : ℚ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_rational_set_not_just_positive_and_negative_l688_68809


namespace NUMINAMATH_CALUDE_alcohol_concentration_after_dilution_l688_68841

/-- Calculates the alcohol concentration in a mixture after adding water -/
theorem alcohol_concentration_after_dilution
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_concentration = 0.42)
  (h3 : added_water = 3)
  : (initial_volume * initial_concentration) / (initial_volume + added_water) = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_after_dilution_l688_68841


namespace NUMINAMATH_CALUDE_base8_subtraction_l688_68886

-- Define a function to convert base 8 numbers to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction :
  natToBase8 ((base8ToNat 256 + base8ToNat 167) - base8ToNat 145) = 370 := by
  sorry

end NUMINAMATH_CALUDE_base8_subtraction_l688_68886


namespace NUMINAMATH_CALUDE_solution_f_greater_than_three_range_of_m_for_f_geq_g_l688_68817

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := m * |x| - 2

-- Theorem for the solution of f(x) > 3
theorem solution_f_greater_than_three :
  ∀ x : ℝ, f x > 3 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for the range of m where f(x) ≥ g(x) for all x
theorem range_of_m_for_f_geq_g :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_f_greater_than_three_range_of_m_for_f_geq_g_l688_68817


namespace NUMINAMATH_CALUDE_cosine_ratio_comparison_l688_68885

theorem cosine_ratio_comparison : 
  (Real.cos (2016 * π / 180)) / (Real.cos (2017 * π / 180)) < 
  (Real.cos (2018 * π / 180)) / (Real.cos (2019 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_comparison_l688_68885


namespace NUMINAMATH_CALUDE_john_can_buy_max_notebooks_l688_68863

/-- The amount of money John has, in cents -/
def johns_money : ℕ := 4575

/-- The cost of each notebook, in cents -/
def notebook_cost : ℕ := 325

/-- The maximum number of notebooks John can buy -/
def max_notebooks : ℕ := 14

theorem john_can_buy_max_notebooks :
  (max_notebooks * notebook_cost ≤ johns_money) ∧
  ∀ n : ℕ, n > max_notebooks → n * notebook_cost > johns_money :=
by sorry

end NUMINAMATH_CALUDE_john_can_buy_max_notebooks_l688_68863


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l688_68826

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := 4

/-- The number of people that can sit in each seat -/
def people_per_seat : ℕ := 4

/-- The total number of people that can ride the Ferris wheel at the same time -/
def total_people : ℕ := num_seats * people_per_seat

theorem ferris_wheel_capacity : total_people = 16 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l688_68826


namespace NUMINAMATH_CALUDE_cubic_function_range_l688_68849

/-- Given a cubic function f(x) = ax³ + bx satisfying certain conditions,
    prove that its range is [-2, 18] -/
theorem cubic_function_range (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^3 + b * x)
    (h_point : f 2 = 2) (h_slope : (fun x ↦ 3 * a * x^2 + b) 2 = 9) :
    Set.range f = Set.Icc (-2) 18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_range_l688_68849


namespace NUMINAMATH_CALUDE_certain_number_proof_l688_68881

theorem certain_number_proof (h : 213 * 16 = 3408) : 
  ∃ x : ℝ, 0.016 * x = 0.03408 ∧ x = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l688_68881


namespace NUMINAMATH_CALUDE_light_bulb_survey_not_appropriate_l688_68811

-- Define the types of surveys
inductive SurveyMethod
| Sampling
| Comprehensive

-- Define the characteristics of a survey subject
structure SurveySubject where
  population_size : Nat
  requires_destruction : Bool

-- Define when a survey method is appropriate
def is_appropriate (method : SurveyMethod) (subject : SurveySubject) : Prop :=
  match method with
  | SurveyMethod.Sampling => subject.population_size > 100 ∨ subject.requires_destruction
  | SurveyMethod.Comprehensive => subject.population_size ≤ 100 ∧ ¬subject.requires_destruction

-- Theorem statement
theorem light_bulb_survey_not_appropriate :
  let light_bulbs : SurveySubject := ⟨1000, true⟩
  ¬(is_appropriate SurveyMethod.Comprehensive light_bulbs) :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_survey_not_appropriate_l688_68811


namespace NUMINAMATH_CALUDE_santinos_garden_fruit_count_l688_68846

/-- Represents the number of trees for each fruit type in Santino's garden -/
structure TreeCounts where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Represents the fruit production rate for each tree type -/
structure FruitProduction where
  papaya : ℕ
  mango : ℕ
  apple : ℕ
  orange : ℕ

/-- Calculates the total number of fruits in Santino's garden -/
def totalFruits (trees : TreeCounts) (production : FruitProduction) : ℕ :=
  trees.papaya * production.papaya +
  trees.mango * production.mango +
  trees.apple * production.apple +
  trees.orange * production.orange

theorem santinos_garden_fruit_count :
  let trees : TreeCounts := ⟨2, 3, 4, 5⟩
  let production : FruitProduction := ⟨10, 20, 15, 25⟩
  totalFruits trees production = 265 := by
  sorry

end NUMINAMATH_CALUDE_santinos_garden_fruit_count_l688_68846


namespace NUMINAMATH_CALUDE_gardening_project_cost_correct_l688_68801

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) (fertilizer_cost : ℕ) 
  (gardener_hours : List ℕ) (gardener_rate : ℕ) (soil_volume : ℕ) (soil_cost : ℕ) : ℕ :=
  let bush_total := rose_bushes * rose_bush_cost
  let fertilizer_total := rose_bushes * fertilizer_cost
  let labor_total := (List.sum gardener_hours) * gardener_rate
  let soil_total := soil_volume * soil_cost
  bush_total + fertilizer_total + labor_total + soil_total

theorem gardening_project_cost_correct : 
  gardening_project_cost 20 150 25 [6, 5, 4, 7] 30 100 5 = 4660 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_cost_correct_l688_68801


namespace NUMINAMATH_CALUDE_sum_zero_sufficient_not_necessary_for_parallel_l688_68823

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem sum_zero_sufficient_not_necessary_for_parallel :
  (∀ (a b : V), a ≠ 0 → b ≠ 0 → a + b = 0 → parallel a b) ∧
  (∃ (a b : V), a ≠ 0 ∧ b ≠ 0 ∧ parallel a b ∧ a + b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_sufficient_not_necessary_for_parallel_l688_68823


namespace NUMINAMATH_CALUDE_country_club_monthly_cost_l688_68819

/-- Calculates the monthly cost per person for a country club membership --/
def monthly_cost_per_person (
  num_people : ℕ
  ) (initial_fee_per_person : ℚ
  ) (john_payment : ℚ
  ) : ℚ :=
  let total_cost := 2 * john_payment
  let total_initial_fee := num_people * initial_fee_per_person
  let total_monthly_cost := total_cost - total_initial_fee
  let yearly_cost_per_person := total_monthly_cost / num_people
  yearly_cost_per_person / 12

theorem country_club_monthly_cost :
  monthly_cost_per_person 4 4000 32000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_country_club_monthly_cost_l688_68819


namespace NUMINAMATH_CALUDE_nine_pointed_star_sum_tip_angles_l688_68812

/-- A 9-pointed star formed by connecting nine evenly spaced points on a circle -/
structure NinePointedStar where
  /-- The measure of the angle at each tip of the star -/
  tip_angle : ℝ
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The points are evenly spaced on the circle -/
  evenly_spaced : num_points = 9
  /-- The measure of the arc between two consecutive points -/
  arc_measure : ℝ
  /-- The arc measure is 360° divided by the number of points -/
  arc_measure_def : arc_measure = 360 / num_points
  /-- Each tip angle subtends an arc that spans 3 consecutive points -/
  tip_angle_subtends_three_arcs : tip_angle = 3 * arc_measure / 2

/-- The sum of the measures of all tip angles in a 9-pointed star is 540° -/
theorem nine_pointed_star_sum_tip_angles (star : NinePointedStar) :
  star.num_points * star.tip_angle = 540 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_sum_tip_angles_l688_68812


namespace NUMINAMATH_CALUDE_min_pool_cost_l688_68887

def pool_volume : ℝ := 18
def pool_depth : ℝ := 2
def bottom_cost_per_sqm : ℝ := 200
def wall_cost_per_sqm : ℝ := 150

theorem min_pool_cost :
  let length : ℝ → ℝ → ℝ := λ x y => x
  let width : ℝ → ℝ → ℝ := λ x y => y
  let volume : ℝ → ℝ → ℝ := λ x y => x * y * pool_depth
  let bottom_area : ℝ → ℝ → ℝ := λ x y => x * y
  let wall_area : ℝ → ℝ → ℝ := λ x y => 2 * (x + y) * pool_depth
  let total_cost : ℝ → ℝ → ℝ := λ x y => 
    bottom_cost_per_sqm * bottom_area x y + wall_cost_per_sqm * wall_area x y
  ∃ x y : ℝ, 
    volume x y = pool_volume ∧ 
    (∀ a b : ℝ, volume a b = pool_volume → total_cost x y ≤ total_cost a b) ∧
    total_cost x y = 5400 :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_l688_68887


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l688_68876

theorem linear_coefficient_of_quadratic (m : ℝ) : 
  m^2 - 2*m - 1 = 2 → 
  m - 3 ≠ 0 → 
  ∃ a b c, (m - 3)*x + 4*m^2 - 2*m - 1 - m*x + 6 = a*x^2 + b*x + c ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l688_68876


namespace NUMINAMATH_CALUDE_cylinder_reciprocal_sum_l688_68836

theorem cylinder_reciprocal_sum (r h : ℝ) (volume_eq : π * r^2 * h = 2) (surface_area_eq : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_reciprocal_sum_l688_68836


namespace NUMINAMATH_CALUDE_david_average_speed_l688_68871

def distance : ℚ := 49 / 3  -- 16 1/3 miles as a fraction

def time : ℚ := 7 / 3  -- 2 hours and 20 minutes as a fraction of hours

def average_speed (d t : ℚ) : ℚ := d / t

theorem david_average_speed :
  average_speed distance time = 7 := by sorry

end NUMINAMATH_CALUDE_david_average_speed_l688_68871


namespace NUMINAMATH_CALUDE_ten_people_two_vip_seats_l688_68862

/-- The number of ways to arrange n people around a round table with k marked VIP seats,
    where arrangements are considered the same if rotations preserve who sits in the VIP seats -/
def roundTableArrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose k) * (n - k).factorial

/-- Theorem stating that for 10 people and 2 VIP seats, there are 1,814,400 arrangements -/
theorem ten_people_two_vip_seats :
  roundTableArrangements 10 2 = 1814400 := by
  sorry

#eval roundTableArrangements 10 2

end NUMINAMATH_CALUDE_ten_people_two_vip_seats_l688_68862


namespace NUMINAMATH_CALUDE_max_book_combination_l688_68866

theorem max_book_combination (total : ℕ) (math_books logic_books : ℕ → ℕ) : 
  total = 20 →
  (∀ k, math_books k + logic_books k = total) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → math_books k = 10 - k ∧ logic_books k = 10 + k) →
  (∀ k, 0 ≤ k ∧ k ≤ 10 → Nat.choose (math_books k) 5 * Nat.choose (logic_books k) 5 ≤ (Nat.choose 10 5)^2) :=
by sorry

end NUMINAMATH_CALUDE_max_book_combination_l688_68866


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l688_68878

theorem quadratic_polynomial_proof (p q : ℝ) : 
  (∃ a b : ℝ, a + b + p + q = 2 ∧ a * b * p * q = 12 ∧ a + b = -p ∧ a * b = q) →
  p = 3 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l688_68878


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l688_68820

/-- A hyperbola with center at the origin, focus on the x-axis, and an asymptote tangent to a specific circle has eccentricity 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*y + 3 = 0 → (b*x - a*y)^2 ≤ (a^2 + b^2) * ((x - 0)^2 + (y - 2)^2)) → 
  (∃ x : ℝ, x ≠ 0 ∧ (0, x) ∈ {(x, y) | (x/a)^2 - (y/b)^2 = 1}) →
  c^2 = a^2 + b^2 →
  c / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l688_68820


namespace NUMINAMATH_CALUDE_largest_a_value_l688_68879

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_triangular (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2

def valid_phone_number (a b c d e f g h i j : ℕ) : Prop :=
  a > b ∧ b > c ∧
  d > e ∧ e > f ∧
  g > h ∧ h > i ∧ i > j ∧
  is_square d ∧ is_square e ∧ is_square f ∧
  is_triangular g ∧ is_triangular h ∧ is_triangular i ∧ is_triangular j ∧
  a + b + c = 10 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem largest_a_value :
  ∀ a b c d e f g h i j : ℕ,
  valid_phone_number a b c d e f g h i j →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_value_l688_68879


namespace NUMINAMATH_CALUDE_inequality_proof_l688_68800

theorem inequality_proof (a b c d e f : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) (h5 : 0 ≤ e) (h6 : 0 ≤ f)
  (h7 : a + b ≤ e) (h8 : c + d ≤ f) : 
  Real.sqrt (a * c) + Real.sqrt (b * d) ≤ Real.sqrt (e * f) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l688_68800


namespace NUMINAMATH_CALUDE_parabola_f_value_l688_68890

/-- A parabola with equation y = dx² + ex + f, vertex at (-1, 3), and passing through (0, 2) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : d * (-1)^2 + e * (-1) + f = 3
  point_condition : d * 0^2 + e * 0 + f = 2

/-- The f-value of the parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_f_value_l688_68890


namespace NUMINAMATH_CALUDE_donation_problem_l688_68838

theorem donation_problem (total_donation_A total_donation_B : ℝ)
  (percent_more : ℝ) (diff_avg_donation : ℝ)
  (h1 : total_donation_A = 1200)
  (h2 : total_donation_B = 1200)
  (h3 : percent_more = 0.2)
  (h4 : diff_avg_donation = 5) :
  ∃ (students_A students_B : ℕ),
    students_A = 48 ∧ 
    students_B = 40 ∧
    students_A = (1 + percent_more) * students_B ∧
    (total_donation_B / students_B) - (total_donation_A / students_A) = diff_avg_donation :=
by
  sorry


end NUMINAMATH_CALUDE_donation_problem_l688_68838


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l688_68869

theorem ceiling_floor_product (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → y ∈ Set.Ioo (-9 : ℝ) (-8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l688_68869


namespace NUMINAMATH_CALUDE_elderly_selected_l688_68806

/-- Given a population with the following properties:
  - Total population of 1500
  - Divided into three equal groups (children, elderly, middle-aged)
  - 60 people are selected using stratified sampling
  This theorem proves that the number of elderly people selected is 20. -/
theorem elderly_selected (total_population : ℕ) (sample_size : ℕ) (num_groups : ℕ) :
  total_population = 1500 →
  sample_size = 60 →
  num_groups = 3 →
  (total_population / num_groups : ℚ) * (sample_size / total_population : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_elderly_selected_l688_68806


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l688_68855

/-- Given a parabola and a moving line with common points, prove the range of t and minimum value of c -/
theorem parabola_line_intersection (t c x₁ x₂ y₁ y₂ : ℝ) : 
  (∀ x, y₁ = x^2 ∧ y₁ = (2*t - 1)*x - c) →  -- Parabola and line equations
  (∀ x, y₂ = x^2 ∧ y₂ = (2*t - 1)*x - c) →  -- Parabola and line equations
  x₁^2 + x₂^2 = t^2 + 2*t - 3 →             -- Given condition
  (2 - Real.sqrt 2 ≤ t ∧ t ≤ 2 + Real.sqrt 2 ∧ t ≠ 1/2) ∧  -- Range of t
  (c ≥ (11 - 6*Real.sqrt 2) / 4) ∧                        -- Minimum value of c
  (c = (11 - 6*Real.sqrt 2) / 4 ↔ t = 2 - Real.sqrt 2)    -- When minimum occurs
  := by sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l688_68855


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l688_68884

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℤ, (x + 5 > 0 ∧ x - m ≤ 1) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  (-3 ≤ m ∧ m < -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l688_68884


namespace NUMINAMATH_CALUDE_fraction_simplification_l688_68843

theorem fraction_simplification : (180 : ℚ) / 16200 = 1 / 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l688_68843


namespace NUMINAMATH_CALUDE_log_expression_equals_negative_one_l688_68840

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_expression_equals_negative_one :
  2 * log2 2 + log2 (5/8) - log2 25 = -1 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_negative_one_l688_68840


namespace NUMINAMATH_CALUDE_investment_rate_problem_l688_68828

/-- Proves that given the specified conditions, the unknown interest rate is 5% -/
theorem investment_rate_problem (total : ℝ) (first_part : ℝ) (first_rate : ℝ) (total_interest : ℝ)
  (h1 : total = 4000)
  (h2 : first_part = 2800)
  (h3 : first_rate = 3)
  (h4 : total_interest = 144)
  (h5 : first_part * (first_rate / 100) + (total - first_part) * (unknown_rate / 100) = total_interest) :
  unknown_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l688_68828


namespace NUMINAMATH_CALUDE_pages_per_day_l688_68842

theorem pages_per_day (total_pages : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (h1 : total_pages = 2100)
  (h2 : weeks = 7)
  (h3 : days_per_week = 3) :
  total_pages / (weeks * days_per_week) = 100 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l688_68842


namespace NUMINAMATH_CALUDE_gcd_1113_1897_l688_68804

theorem gcd_1113_1897 : Nat.gcd 1113 1897 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1113_1897_l688_68804


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_leq_l688_68898

theorem solution_set_reciprocal_leq (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ x ↔ (-1 ≤ x ∧ x < 0) ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_leq_l688_68898


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l688_68864

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l688_68864


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l688_68861

/-- In a right triangle DEF where angle D is 90 degrees and sin E = 3/5, cos F = 3/5 -/
theorem right_triangle_cosine (D E F : ℝ) : 
  D = Real.pi / 2 → 
  Real.sin E = 3 / 5 → 
  Real.cos F = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l688_68861


namespace NUMINAMATH_CALUDE_alex_age_problem_l688_68850

/-- Alex's age problem -/
theorem alex_age_problem (A M : ℝ) : 
  (A - M = 3 * (A - 4 * M)) → A / M = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_age_problem_l688_68850


namespace NUMINAMATH_CALUDE_payment_for_remaining_worker_l688_68883

/-- Given a total payment for a job and the fraction of work done by two workers,
    calculate the payment for the remaining worker. -/
theorem payment_for_remaining_worker
  (total_payment : ℚ)
  (work_fraction_two_workers : ℚ)
  (h1 : total_payment = 529)
  (h2 : work_fraction_two_workers = 19 / 23) :
  (1 - work_fraction_two_workers) * total_payment = 92 := by
sorry

end NUMINAMATH_CALUDE_payment_for_remaining_worker_l688_68883


namespace NUMINAMATH_CALUDE_exactly_two_b_values_l688_68868

theorem exactly_two_b_values : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), 
      (∀ x ∈ t, x^2 + b*x + 6 ≤ 0) ∧ 
      (∀ x ∉ t, x^2 + b*x + 6 > 0) ∧ 
      t.card = 3) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_b_values_l688_68868


namespace NUMINAMATH_CALUDE_first_coaster_speed_is_50_l688_68870

/-- The speed of the first rollercoaster given the speeds of the other four and the average speed -/
def first_coaster_speed (second_speed third_speed fourth_speed fifth_speed average_speed : ℝ) : ℝ :=
  5 * average_speed - (second_speed + third_speed + fourth_speed + fifth_speed)

/-- Theorem stating that the first coaster's speed is 50 mph given the problem conditions -/
theorem first_coaster_speed_is_50 :
  first_coaster_speed 62 73 70 40 59 = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_coaster_speed_is_50_l688_68870


namespace NUMINAMATH_CALUDE_max_initial_pieces_is_285_l688_68888

/-- Represents a Go board with dimensions n x n -/
structure GoBoard (n : ℕ) where
  size : n > 0

/-- Represents a rectangular arrangement of pieces on a Go board -/
structure Rectangle (n : ℕ) where
  width : ℕ
  height : ℕ
  pieces : ℕ
  width_valid : width ≤ n
  height_valid : height ≤ n
  area_eq_pieces : width * height = pieces

/-- The maximum number of pieces in the initial rectangle -/
def max_initial_pieces (board : GoBoard 19) : ℕ := 285

/-- Theorem stating the maximum number of pieces in the initial rectangle -/
theorem max_initial_pieces_is_285 (board : GoBoard 19) :
  ∃ (init final : Rectangle 19),
    init.pieces = max_initial_pieces board ∧
    final.pieces = init.pieces + 45 ∧
    final.width = init.width ∧
    final.height > init.height ∧
    ∀ (other : Rectangle 19),
      (∃ (other_final : Rectangle 19),
        other_final.pieces = other.pieces + 45 ∧
        other_final.width = other.width ∧
        other_final.height > other.height) →
      other.pieces ≤ init.pieces :=
by sorry

end NUMINAMATH_CALUDE_max_initial_pieces_is_285_l688_68888


namespace NUMINAMATH_CALUDE_carrie_phone_trade_in_l688_68892

/-- The trade-in value of Carrie's old phone -/
def trade_in_value : ℕ := sorry

/-- The cost of the new iPhone -/
def iphone_cost : ℕ := 800

/-- Carrie's weekly earnings from babysitting -/
def weekly_earnings : ℕ := 80

/-- The number of weeks Carrie has to work -/
def weeks_to_work : ℕ := 7

/-- The total amount Carrie earns from babysitting -/
def total_earnings : ℕ := weekly_earnings * weeks_to_work

theorem carrie_phone_trade_in :
  trade_in_value = iphone_cost - total_earnings :=
sorry

end NUMINAMATH_CALUDE_carrie_phone_trade_in_l688_68892


namespace NUMINAMATH_CALUDE_investigator_strategy_equivalence_l688_68814

/-- Represents the investigator's questioning strategy -/
structure InvestigatorStrategy where
  num_questions : ℕ
  max_lie : ℕ

/-- Defines the original strategy with all truthful answers -/
def original_strategy : InvestigatorStrategy :=
  { num_questions := 91
  , max_lie := 0 }

/-- Defines the new strategy allowing for one possible lie -/
def new_strategy : InvestigatorStrategy :=
  { num_questions := 105
  , max_lie := 1 }

/-- Represents the information obtained from questioning -/
def Information : Type := Unit

/-- Function to obtain information given a strategy -/
def obtain_information (strategy : InvestigatorStrategy) : Information := sorry

theorem investigator_strategy_equivalence :
  obtain_information original_strategy = obtain_information new_strategy :=
by sorry

end NUMINAMATH_CALUDE_investigator_strategy_equivalence_l688_68814


namespace NUMINAMATH_CALUDE_floor_sum_example_l688_68839

theorem floor_sum_example : ⌊(24.7 : ℝ)⌋ + ⌊(-24.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l688_68839


namespace NUMINAMATH_CALUDE_money_distribution_l688_68853

theorem money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * vasim = 5 * faruk →
  8 * faruk = 3 * ranjith →
  ranjith - faruk = 1500 →
  vasim = 1500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l688_68853


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l688_68833

/-- A line passing through point (2,3) and parallel to 2x+4y-3=0 has equation x + 2y - 8 = 0 -/
theorem line_through_point_parallel_to_line :
  let line1 : ℝ → ℝ → Prop := λ x y => x + 2*y - 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2*x + 4*y - 3 = 0
  (line1 2 3) ∧ 
  (∀ (x y : ℝ), line1 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) ∧
  (∀ (x y : ℝ), line2 x y ↔ ∃ (k : ℝ), y = (-1/2)*x + k) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l688_68833


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l688_68875

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a_1 + a_3 = 10 and a_4 + a_6 = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l688_68875


namespace NUMINAMATH_CALUDE_inequality_proof_l688_68858

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l688_68858


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l688_68895

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi ∧ 
   ∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x + Real.cos y - 1) ↔ 
  (y = 0 ∨ y = Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l688_68895


namespace NUMINAMATH_CALUDE_distinct_collections_count_l688_68834

/-- Represents the count of each letter in "MATHEMATICAL" --/
structure LetterCount where
  a : Nat
  e : Nat
  i : Nat
  t : Nat
  m : Nat
  h : Nat
  l : Nat
  c : Nat

/-- The initial count of letters in "MATHEMATICAL" --/
def initialCount : LetterCount := {
  a := 3, e := 1, i := 1,
  t := 2, m := 2, h := 1, l := 1, c := 2
}

/-- A collection of letters that fell off --/
structure FallenLetters where
  vowels : Finset Char
  consonants : Finset Char

/-- Checks if a collection of fallen letters is valid --/
def isValidCollection (letters : FallenLetters) : Prop :=
  letters.vowels.card = 3 ∧ letters.consonants.card = 3

/-- Counts distinct collections considering indistinguishable letters --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 80 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l688_68834


namespace NUMINAMATH_CALUDE_students_just_passed_l688_68848

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 28 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 54 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l688_68848


namespace NUMINAMATH_CALUDE_charles_picked_50_pears_l688_68803

/-- The number of pears Charles picked -/
def pears_picked : ℕ := sorry

/-- The number of bananas Charles cooked -/
def bananas_cooked : ℕ := sorry

/-- The number of dishes Sandrine washed -/
def dishes_washed : ℕ := 160

theorem charles_picked_50_pears :
  (dishes_washed = bananas_cooked + 10) ∧
  (bananas_cooked = 3 * pears_picked) →
  pears_picked = 50 := by sorry

end NUMINAMATH_CALUDE_charles_picked_50_pears_l688_68803


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l688_68829

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →  -- This condition is necessary to ensure the square root is defined
  (Real.sqrt (8 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 18 / 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l688_68829


namespace NUMINAMATH_CALUDE_distance_before_stop_correct_concert_drive_distance_l688_68837

/-- Calculates the distance driven before stopping for gas --/
def distance_before_stop (total_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is equal to 
    the total distance minus the remaining distance --/
theorem distance_before_stop_correct (total_distance : ℕ) (remaining_distance : ℕ) 
    (h : remaining_distance ≤ total_distance) :
  distance_before_stop total_distance remaining_distance = 
    total_distance - remaining_distance := by
  sorry

/-- Given the total distance and remaining distance, 
    prove that the distance driven before stopping is 32 miles --/
theorem concert_drive_distance :
  distance_before_stop 78 46 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distance_before_stop_correct_concert_drive_distance_l688_68837


namespace NUMINAMATH_CALUDE_project_hours_difference_l688_68807

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l688_68807


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l688_68894

theorem absolute_value_equation_product (x : ℝ) : 
  (|20 / x + 4| = 3) → (∃ y : ℝ, (|20 / y + 4| = 3) ∧ (x * y = 400 / 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l688_68894


namespace NUMINAMATH_CALUDE_nancy_museum_pictures_l688_68896

theorem nancy_museum_pictures :
  ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
    zoo_pics = 49 →
    deleted_pics = 38 →
    remaining_pics = 19 →
    zoo_pics + museum_pics = deleted_pics + remaining_pics →
    museum_pics = 8 :=
by sorry

end NUMINAMATH_CALUDE_nancy_museum_pictures_l688_68896


namespace NUMINAMATH_CALUDE_factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l688_68825

theorem factorial_difference_is_cubic_polynomial (n : ℕ) (h : n ≥ 9) :
  (((n + 3).factorial - (n + 2).factorial) / n.factorial : ℚ) = (n + 2)^2 * (n + 1) :=
by sorry

theorem cubic_polynomial_form (n : ℕ) (h : n ≥ 9) :
  ∃ (a b c d : ℚ), (n + 2)^2 * (n + 1) = a * n^3 + b * n^2 + c * n + d :=
by sorry

end NUMINAMATH_CALUDE_factorial_difference_is_cubic_polynomial_cubic_polynomial_form_l688_68825


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_distance_comparison_l688_68873

theorem quadratic_inequality_and_distance_comparison :
  (∀ (k : ℝ), (∀ (x : ℝ), 2 * k * x^2 + k * x - 3/8 < 0) ↔ (k > -3 ∧ k ≤ 0)) ∧
  (∀ (a b : ℝ), a ≠ b → |(a^2 + b^2)/2 - ((a+b)/2)^2| > |a*b - ((a+b)/2)^2|) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_distance_comparison_l688_68873


namespace NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l688_68802

/-- The number of legs for each animal type -/
def dog_legs : ℕ := 4
def chicken_legs : ℕ := 2
def spider_legs : ℕ := 8
def octopus_legs : ℕ := 8

/-- The number of each animal type Mrs. Hilt saw -/
def dogs_seen : ℕ := 3
def chickens_seen : ℕ := 4
def spiders_seen : ℕ := 2
def octopuses_seen : ℕ := 1

/-- The total number of animal legs Mrs. Hilt saw -/
def total_legs : ℕ := dogs_seen * dog_legs + chickens_seen * chicken_legs + 
                      spiders_seen * spider_legs + octopuses_seen * octopus_legs

theorem mrs_hilt_animal_legs : total_legs = 44 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_animal_legs_l688_68802


namespace NUMINAMATH_CALUDE_star_3_5_l688_68897

def star (c d : ℝ) : ℝ := c^2 - 2*c*d + d^2

theorem star_3_5 : star 3 5 = 4 := by sorry

end NUMINAMATH_CALUDE_star_3_5_l688_68897


namespace NUMINAMATH_CALUDE_base_r_is_10_l688_68856

/-- Converts a number from base r to base 10 -/
def toBase10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_is_10 (r : Nat) : r > 0 → 
  toBase10 [1, 3, 5] r + toBase10 [1, 5, 4] r = toBase10 [0, 0, 1, 1] r → 
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_r_is_10_l688_68856


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_square_l688_68893

theorem integral_sqrt_minus_square : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_square_l688_68893


namespace NUMINAMATH_CALUDE_min_variance_of_sample_l688_68860

theorem min_variance_of_sample (x y : ℝ) : 
  (x + 1 + y + 5) / 4 = 2 → 
  ((x - 2)^2 + (1 - 2)^2 + (y - 2)^2 + (5 - 2)^2) / 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_variance_of_sample_l688_68860


namespace NUMINAMATH_CALUDE_winnie_the_pooh_fall_damage_ratio_l688_68830

/-- The ratio of damages in Winnie-the-Pooh's fall -/
theorem winnie_the_pooh_fall_damage_ratio 
  (g M τ H : ℝ) 
  (n k : ℝ) 
  (h : ℝ := H / n) 
  (V_I : ℝ := Real.sqrt (2 * g * H)) 
  (V_1 : ℝ := Real.sqrt (2 * g * h)) 
  (V_1_prime : ℝ := (1 / k) * Real.sqrt (2 * g * h)) 
  (V_II : ℝ := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))) 
  (I_I : ℝ := M * V_I * τ) 
  (I_II : ℝ := M * τ * ((V_1 - V_1_prime) + V_II)) 
  (hg : g > 0) 
  (hM : M > 0) 
  (hτ : τ > 0) 
  (hH : H > 0) 
  (hn : n > 0) 
  (hk : k > 0) : 
  I_II / I_I = 5 / 4 := by
  sorry


end NUMINAMATH_CALUDE_winnie_the_pooh_fall_damage_ratio_l688_68830


namespace NUMINAMATH_CALUDE_intersection_theorem_union_theorem_l688_68813

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + (a^2-5) = 0}

theorem intersection_theorem : 
  A ∩ B a = {2} → a = -1 ∨ a = -3 := by sorry

theorem union_theorem : 
  A ∪ B a = A → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_union_theorem_l688_68813


namespace NUMINAMATH_CALUDE_constant_intersection_point_range_l688_68880

/-- Given that when m ∈ ℝ, the function f(x) = m(x^2 - 1) + x - a has a constant 
    intersection point with the x-axis, then a ∈ ℝ when m = 0 and a ∈ [-1, 1] when m ≠ 0 -/
theorem constant_intersection_point_range (m a : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, m * (x^2 - 1) + x - a = 0 → x = k) → 
  ((m = 0 → a ∈ Set.univ) ∧ (m ≠ 0 → a ∈ Set.Icc (-1) 1)) :=
sorry

end NUMINAMATH_CALUDE_constant_intersection_point_range_l688_68880


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l688_68845

/-- A sequence is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The property that a_n^2 = a_{n-1} * a_{n+1} for all n -/
def HasSquareProperty (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, IsGeometricSequence a → HasSquareProperty a) ∧
  (∃ a : ℕ → ℝ, HasSquareProperty a ∧ ¬IsGeometricSequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l688_68845


namespace NUMINAMATH_CALUDE_third_week_vegetable_intake_l688_68872

/-- Represents the daily vegetable intake in pounds -/
structure DailyIntake where
  asparagus : ℝ
  broccoli : ℝ
  cauliflower : ℝ
  spinach : ℝ
  kale : ℝ
  zucchini : ℝ

/-- Calculates the total daily intake -/
def totalDailyIntake (intake : DailyIntake) : ℝ :=
  intake.asparagus + intake.broccoli + intake.cauliflower + intake.spinach + intake.kale + intake.zucchini

/-- Initial daily intake -/
def initialIntake : DailyIntake :=
  { asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5, spinach := 0, kale := 0, zucchini := 0 }

/-- Daily intake after second week changes -/
def secondWeekIntake : DailyIntake :=
  { asparagus := initialIntake.asparagus * 2,
    broccoli := initialIntake.broccoli * 3,
    cauliflower := initialIntake.cauliflower * 1.75,
    spinach := 0.5,
    kale := 0,
    zucchini := 0 }

/-- Daily intake in the third week -/
def thirdWeekIntake : DailyIntake :=
  { asparagus := secondWeekIntake.asparagus,
    broccoli := secondWeekIntake.broccoli,
    cauliflower := secondWeekIntake.cauliflower,
    spinach := secondWeekIntake.spinach,
    kale := 0.5,  -- 1 pound every two days
    zucchini := 0.15 }  -- 0.3 pounds every two days

theorem third_week_vegetable_intake :
  totalDailyIntake thirdWeekIntake * 7 = 22.925 := by
  sorry

end NUMINAMATH_CALUDE_third_week_vegetable_intake_l688_68872


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l688_68857

/-- Given positive integers A, B, C, D satisfying the conditions:
    1. A, B, C form an arithmetic sequence
    2. B, C, D form a geometric sequence
    3. C/B = 4/3
    The smallest possible value of A + B + C + D is 43. -/
theorem smallest_sum_of_sequence (A B C D : ℕ+) : 
  (∃ r : ℚ, C = A + r ∧ B = A + 2*r) →  -- A, B, C form an arithmetic sequence
  (∃ q : ℚ, C = B * q ∧ D = C * q) →   -- B, C, D form a geometric sequence
  (C : ℚ) / B = 4 / 3 →                -- The ratio of the geometric sequence
  A + B + C + D ≥ 43 ∧ 
  (∃ A' B' C' D' : ℕ+, A' + B' + C' + D' = 43 ∧ 
    (∃ r : ℚ, C' = A' + r ∧ B' = A' + 2*r) ∧
    (∃ q : ℚ, C' = B' * q ∧ D' = C' * q) ∧
    (C' : ℚ) / B' = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l688_68857


namespace NUMINAMATH_CALUDE_chosen_number_proof_l688_68815

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_proof_l688_68815


namespace NUMINAMATH_CALUDE_tim_score_theorem_l688_68835

/-- Sum of the first n even numbers -/
def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- A number is recognizable if it's 90 (for this specific problem) -/
def isRecognizable (x : ℕ) : Prop := x = 90

/-- A number is a square number if it's the square of some integer -/
def isSquareNumber (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem tim_score_theorem :
  ∃ n : ℕ, isSquareNumber n ∧ isRecognizable (sumFirstNEvenNumbers n) ∧
  ∀ m : ℕ, m < n → ¬(isSquareNumber m ∧ isRecognizable (sumFirstNEvenNumbers m)) :=
by sorry

end NUMINAMATH_CALUDE_tim_score_theorem_l688_68835


namespace NUMINAMATH_CALUDE_triangle_angle_C_l688_68821

/-- Given a triangle ABC with side lengths a and c, and angle A, prove that C is either 60° or 120°. -/
theorem triangle_angle_C (a c : ℝ) (A : Real) (h1 : a = 2) (h2 : c = Real.sqrt 6) (h3 : A = π / 4) :
  let C := Real.arcsin ((c * Real.sin A) / a)
  C = π / 3 ∨ C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l688_68821


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l688_68847

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-35 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-35 : ℚ) / 4⌋⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l688_68847


namespace NUMINAMATH_CALUDE_expression_evaluation_l688_68865

theorem expression_evaluation (x y : ℚ) (hx : x = 1) (hy : y = -3) :
  ((x - 2*y)^2 + (3*x - y)*(3*x + y) - 3*y^2) / (-2*x) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l688_68865


namespace NUMINAMATH_CALUDE_rhombus_area_l688_68805

/-- A rhombus with specific properties. -/
structure Rhombus where
  /-- The side length of the rhombus. -/
  side_length : ℝ
  /-- The length of half of the shorter diagonal. -/
  half_shorter_diagonal : ℝ
  /-- The difference between the diagonals. -/
  diagonal_difference : ℝ
  /-- The side length is √109. -/
  side_length_eq : side_length = Real.sqrt 109
  /-- The diagonal difference is 12. -/
  diagonal_difference_eq : diagonal_difference = 12
  /-- The Pythagorean theorem holds for the right triangle formed by half of each diagonal and the side. -/
  pythagorean_theorem : half_shorter_diagonal ^ 2 + (half_shorter_diagonal + diagonal_difference / 2) ^ 2 = side_length ^ 2

/-- The area of a rhombus with the given properties is 364 square units. -/
theorem rhombus_area (r : Rhombus) : r.half_shorter_diagonal * (r.half_shorter_diagonal + r.diagonal_difference / 2) * 2 = 364 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l688_68805


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l688_68899

/-- Given two parallel vectors a and b in ℝ², prove that if a = (x, 3) and b = (4, 6), then x = 2 -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (x, 3)) 
  (h2 : b = (4, 6)) 
  (h3 : ∃ (k : ℝ), a = k • b) : 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l688_68899


namespace NUMINAMATH_CALUDE_income_ratio_l688_68832

/-- Proof of the ratio of monthly incomes --/
theorem income_ratio (c_income b_income a_annual_income : ℝ) 
  (hb : b_income = c_income * 1.12)
  (hc : c_income = 12000)
  (ha : a_annual_income = 403200.0000000001) :
  (a_annual_income / 12) / b_income = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_l688_68832
