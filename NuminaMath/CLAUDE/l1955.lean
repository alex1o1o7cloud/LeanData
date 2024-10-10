import Mathlib

namespace jacks_total_yen_l1955_195554

/-- Represents the amount of money Jack has in different currencies -/
structure JacksMoney where
  pounds : ℕ
  euros : ℕ
  yen : ℕ

/-- Represents the exchange rates between currencies -/
structure ExchangeRates where
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the total amount of yen Jack has -/
def total_yen (money : JacksMoney) (rates : ExchangeRates) : ℕ :=
  money.yen +
  money.pounds * rates.yen_per_pound +
  money.euros * rates.pounds_per_euro * rates.yen_per_pound

/-- Theorem stating that Jack's total amount in yen is 9400 -/
theorem jacks_total_yen :
  let money := JacksMoney.mk 42 11 3000
  let rates := ExchangeRates.mk 2 100
  total_yen money rates = 9400 := by
  sorry

end jacks_total_yen_l1955_195554


namespace simplify_expression_l1955_195561

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (125 : ℝ) ^ (1/2) = 20 * Real.sqrt 5 := by
  sorry

end simplify_expression_l1955_195561


namespace marcos_dad_strawberries_weight_l1955_195537

theorem marcos_dad_strawberries_weight (marco_weight dad_weight total_weight : ℕ) :
  marco_weight = 8 →
  total_weight = 40 →
  total_weight = marco_weight + dad_weight →
  dad_weight = 32 := by
sorry

end marcos_dad_strawberries_weight_l1955_195537


namespace inequality_proof_l1955_195578

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l1955_195578


namespace abby_and_damon_weight_l1955_195511

theorem abby_and_damon_weight (a b c d : ℝ) 
  (h1 : a + b = 260)
  (h2 : b + c = 245)
  (h3 : c + d = 270)
  (h4 : a + c = 220) :
  a + d = 285 := by
  sorry

end abby_and_damon_weight_l1955_195511


namespace largest_integer_less_than_100_remainder_3_mod_8_l1955_195597

theorem largest_integer_less_than_100_remainder_3_mod_8 :
  ∃ n : ℕ, n < 100 ∧ n % 8 = 3 ∧ ∀ m : ℕ, m < 100 ∧ m % 8 = 3 → m ≤ n :=
by sorry

end largest_integer_less_than_100_remainder_3_mod_8_l1955_195597


namespace quadratic_transformation_l1955_195528

/-- Given a quadratic equation x² + px + q = 0 with roots x₁ and x₂,
    this theorem proves the form of the quadratic equation whose roots are
    y₁ = (x₁ + x₁²) / (1 - x₂) and y₂ = (x₂ + x₂²) / (1 - x₁) -/
theorem quadratic_transformation (p q : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + p*x₁ + q = 0 →
  x₂^2 + p*x₂ + q = 0 →
  x₁ ≠ x₂ →
  x₁ ≠ 1 →
  x₂ ≠ 1 →
  let y₁ := (x₁ + x₁^2) / (1 - x₂)
  let y₂ := (x₂ + x₂^2) / (1 - x₁)
  ∃ (y : ℝ), y^2 + (p*(1 + 3*q - p^2) / (1 + p + q))*y + (q*(1 - p + q) / (1 + p + q)) = 0 ↔
             (y = y₁ ∨ y = y₂) :=
by sorry

end quadratic_transformation_l1955_195528


namespace polygon_sides_l1955_195525

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1800 → n = 10 := by
sorry

end polygon_sides_l1955_195525


namespace quadratic_roots_difference_specific_quadratic_roots_difference_l1955_195548

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * (x^2) + b * x + c = 0 → |r₁ - r₂| = Real.sqrt ((b^2 - 4*a*c) / (a^2)) :=
by sorry

theorem specific_quadratic_roots_difference :
  let r₁ := (-(-7) + Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  let r₂ := (-(-7) - Real.sqrt ((-7)^2 - 4*1*12)) / (2*1)
  |r₁ - r₂| = 1 :=
by sorry

end quadratic_roots_difference_specific_quadratic_roots_difference_l1955_195548


namespace bird_count_problem_l1955_195553

/-- Represents the number of birds in a group -/
structure BirdGroup where
  adults : ℕ
  offspring_per_adult : ℕ

/-- Calculates the total number of birds in a group -/
def total_birds (group : BirdGroup) : ℕ :=
  group.adults * (group.offspring_per_adult + 1)

/-- The problem statement -/
theorem bird_count_problem (duck_group1 duck_group2 duck_group3 geese_group swan_group : BirdGroup)
  (h1 : duck_group1 = { adults := 2, offspring_per_adult := 5 })
  (h2 : duck_group2 = { adults := 6, offspring_per_adult := 3 })
  (h3 : duck_group3 = { adults := 9, offspring_per_adult := 6 })
  (h4 : geese_group = { adults := 4, offspring_per_adult := 7 })
  (h5 : swan_group = { adults := 3, offspring_per_adult := 4 }) :
  (total_birds duck_group1 + total_birds duck_group2 + total_birds duck_group3 +
   total_birds geese_group + total_birds swan_group) * 3 = 438 := by
  sorry

end bird_count_problem_l1955_195553


namespace boat_against_stream_distance_l1955_195577

/-- The distance a boat travels against the stream in one hour -/
def distance_against_stream (downstream_distance : ℝ) (still_water_speed : ℝ) : ℝ :=
  still_water_speed - (downstream_distance - still_water_speed)

/-- Theorem: Given a boat that travels 13 km downstream in one hour with a still water speed of 9 km/hr,
    the distance it travels against the stream in one hour is 5 km. -/
theorem boat_against_stream_distance :
  distance_against_stream 13 9 = 5 := by
  sorry

end boat_against_stream_distance_l1955_195577


namespace triangles_are_similar_l1955_195529

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def are_similar (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ b = k * e ∧ c = k * f ∧ a = k * d

/-- Triangle ABC has sides of length 1, √2, and √5 -/
def triangle_ABC (a b c : ℝ) : Prop :=
  a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 5

/-- Triangle DEF has sides of length √3, √6, and √15 -/
def triangle_DEF (d e f : ℝ) : Prop :=
  d = Real.sqrt 3 ∧ e = Real.sqrt 6 ∧ f = Real.sqrt 15

theorem triangles_are_similar :
  ∀ (a b c d e f : ℝ),
    triangle_ABC a b c →
    triangle_DEF d e f →
    are_similar a b c d e f :=
by sorry

end triangles_are_similar_l1955_195529


namespace hyperbola_y_foci_coeff_signs_l1955_195584

/-- A curve represented by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_foci (c : Curve) : Prop :=
  ∃ (p q : ℝ), p > 0 ∧ q > 0 ∧ ∀ (x y : ℝ), c.a * x^2 + c.b * y^2 = 1 ↔ x^2/p - y^2/q = 1

/-- Theorem stating that if a curve is a hyperbola with foci on the y-axis,
    then its 'a' coefficient is negative and 'b' coefficient is positive -/
theorem hyperbola_y_foci_coeff_signs (c : Curve) :
  is_hyperbola_y_foci c → c.a < 0 ∧ c.b > 0 :=
by sorry

end hyperbola_y_foci_coeff_signs_l1955_195584


namespace columns_in_first_arrangement_l1955_195556

/-- Given a group of people, prove the number of columns formed when 30 people stand in each column. -/
theorem columns_in_first_arrangement 
  (total_people : ℕ) 
  (people_per_column_second : ℕ) 
  (columns_second : ℕ) 
  (people_per_column_first : ℕ) 
  (h1 : people_per_column_second = 32) 
  (h2 : columns_second = 15) 
  (h3 : people_per_column_first = 30) 
  (h4 : total_people = people_per_column_second * columns_second) :
  total_people / people_per_column_first = 16 :=
by sorry

end columns_in_first_arrangement_l1955_195556


namespace regression_lines_intersect_l1955_195589

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on a regression line -/
def RegressionLine.evaluate (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- Represents a dataset used for linear regression -/
structure Dataset where
  size : ℕ
  avg_x : ℝ
  avg_y : ℝ

/-- Theorem: Two regression lines from datasets with the same average x and y intersect at (avg_x, avg_y) -/
theorem regression_lines_intersect (data1 data2 : Dataset) (line1 line2 : RegressionLine)
    (h1 : data1.avg_x = data2.avg_x)
    (h2 : data1.avg_y = data2.avg_y)
    (h3 : line1.evaluate data1.avg_x = data1.avg_y)
    (h4 : line2.evaluate data2.avg_x = data2.avg_y) :
    line1.evaluate data1.avg_x = line2.evaluate data2.avg_x := by
  sorry


end regression_lines_intersect_l1955_195589


namespace correct_match_l1955_195559

/-- Represents a philosophical statement --/
structure PhilosophicalStatement :=
  (text : String)
  (interpretation : String)

/-- Checks if a statement represents seizing opportunity for qualitative change --/
def representsQualitativeChange (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "Decisively seize the opportunity to promote qualitative change"

/-- Checks if a statement represents forward development --/
def representsForwardDevelopment (statement : PhilosophicalStatement) : Prop :=
  statement.interpretation = "The future is bright"

/-- The four given statements --/
def statement1 : PhilosophicalStatement :=
  { text := "As cold comes and heat goes, the four seasons change"
  , interpretation := "Things are developing" }

def statement2 : PhilosophicalStatement :=
  { text := "Thousands of flowers arranged, just waiting for the first thunder"
  , interpretation := "Decisively seize the opportunity to promote qualitative change" }

def statement3 : PhilosophicalStatement :=
  { text := "Despite the intention to plant flowers, they don't bloom; unintentionally planting willows, they grow into shade"
  , interpretation := "The unity of contradictions" }

def statement4 : PhilosophicalStatement :=
  { text := "There will be times when the strong winds break the waves, and we will sail across the sea with clouds"
  , interpretation := "The future is bright" }

/-- Theorem stating that statements 2 and 4 correctly match the required interpretations --/
theorem correct_match :
  representsQualitativeChange statement2 ∧
  representsForwardDevelopment statement4 :=
by sorry

end correct_match_l1955_195559


namespace dans_car_mpg_l1955_195522

/-- Given the cost of gas and the distance a car can travel on a certain amount of gas,
    calculate the miles per gallon of the car. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (gas_expense : ℝ) (mpg : ℝ) : 
  gas_cost = 4 →
  miles = 464 →
  gas_expense = 58 →
  mpg = miles / (gas_expense / gas_cost) →
  mpg = 32 := by
sorry

end dans_car_mpg_l1955_195522


namespace smallest_whole_number_above_sum_l1955_195501

theorem smallest_whole_number_above_sum : 
  ⌈(3 + 1/7 : ℚ) + (4 + 1/8 : ℚ) + (5 + 1/9 : ℚ) + (6 + 1/10 : ℚ)⌉ = 19 := by
  sorry

end smallest_whole_number_above_sum_l1955_195501


namespace beth_and_jan_total_money_l1955_195514

def beth_money : ℕ := 70
def jan_money : ℕ := 80

theorem beth_and_jan_total_money :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) →
  beth_money + jan_money = 150 := by
  sorry

end beth_and_jan_total_money_l1955_195514


namespace floor_sum_example_l1955_195580

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1955_195580


namespace increase_by_percentage_l1955_195566

/-- Theorem: Increasing 350 by 50% results in 525. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 350 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 525 := by
  sorry

end increase_by_percentage_l1955_195566


namespace parallel_planes_theorem_l1955_195562

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (subset : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (α β : Plane) (a b : Line) (A : Point) :
  subset a α →
  subset b α →
  intersect a b A →
  parallel_line_plane a β →
  parallel_line_plane b β →
  parallel_plane α β :=
sorry

end parallel_planes_theorem_l1955_195562


namespace reciprocal_abs_eq_neg_self_l1955_195569

theorem reciprocal_abs_eq_neg_self :
  ∃! (a : ℝ), |1 / a| = -a :=
by
  -- The proof goes here
  sorry

end reciprocal_abs_eq_neg_self_l1955_195569


namespace product_draw_probabilities_l1955_195558

/-- Represents the probability space for drawing products -/
structure ProductDraw where
  total : Nat
  defective : Nat
  nonDefective : Nat
  hTotal : total = defective + nonDefective

/-- The probability of drawing a defective product on the first draw -/
def probFirstDefective (pd : ProductDraw) : Rat :=
  pd.defective / pd.total

/-- The probability of drawing defective products on both draws -/
def probBothDefective (pd : ProductDraw) : Rat :=
  (pd.defective / pd.total) * ((pd.defective - 1) / (pd.total - 1))

/-- The probability of drawing a defective product on the second draw, given the first was defective -/
def probSecondDefectiveGivenFirst (pd : ProductDraw) : Rat :=
  (pd.defective - 1) / (pd.total - 1)

theorem product_draw_probabilities (pd : ProductDraw) 
  (h1 : pd.total = 20) 
  (h2 : pd.defective = 5) 
  (h3 : pd.nonDefective = 15) : 
  probFirstDefective pd = 1/4 ∧ 
  probBothDefective pd = 1/19 ∧ 
  probSecondDefectiveGivenFirst pd = 4/19 := by
  sorry

end product_draw_probabilities_l1955_195558


namespace sarah_reading_speed_l1955_195542

/-- Calculates Sarah's reading speed in words per minute -/
def sarahReadingSpeed (wordsPerPage : ℕ) (pagesPerBook : ℕ) (readingHours : ℕ) (numberOfBooks : ℕ) : ℕ :=
  let totalWords := wordsPerPage * pagesPerBook * numberOfBooks
  let totalMinutes := readingHours * 60
  totalWords / totalMinutes

/-- Theorem stating Sarah's reading speed under given conditions -/
theorem sarah_reading_speed :
  sarahReadingSpeed 100 80 20 6 = 40 := by
  sorry

end sarah_reading_speed_l1955_195542


namespace professor_chair_choices_l1955_195544

/-- The number of chairs in a row -/
def num_chairs : ℕ := 13

/-- The number of professors -/
def num_professors : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 9

/-- A function that returns true if a chair position is valid for a professor -/
def is_valid_chair (pos : ℕ) : Prop :=
  1 < pos ∧ pos < num_chairs

/-- A function that returns true if two chair positions are valid for two professors -/
def are_valid_chairs (pos1 pos2 : ℕ) : Prop :=
  is_valid_chair pos1 ∧ is_valid_chair pos2 ∧ pos1 + 1 < pos2

/-- The total number of ways professors can choose their chairs -/
def num_ways : ℕ := 45

/-- Theorem stating that the number of ways professors can choose their chairs is 45 -/
theorem professor_chair_choices :
  (Finset.sum (Finset.range (num_chairs - 3))
    (λ k => num_chairs - (k + 3))) = num_ways :=
sorry

end professor_chair_choices_l1955_195544


namespace sin_monotone_increasing_interval_l1955_195555

/-- The function f(x) = sin(2π/3 - 2x) is monotonically increasing on the interval [7π/12, 13π/12] -/
theorem sin_monotone_increasing_interval :
  let f : ℝ → ℝ := λ x => Real.sin (2 * Real.pi / 3 - 2 * x)
  ∀ x y, 7 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ 13 * Real.pi / 12 → f x < f y :=
by sorry

end sin_monotone_increasing_interval_l1955_195555


namespace number_of_kids_l1955_195510

theorem number_of_kids (total_money : ℕ) (apple_cost : ℕ) (apples_per_kid : ℕ) : 
  total_money = 360 → 
  apple_cost = 4 → 
  apples_per_kid = 5 → 
  (total_money / apple_cost) / apples_per_kid = 18 := by
  sorry

end number_of_kids_l1955_195510


namespace solve_exponential_equation_l1955_195526

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = (125 : ℝ)^(1/3) ∧ x = 1/3 := by
  sorry

end solve_exponential_equation_l1955_195526


namespace logarithm_sum_simplification_l1955_195519

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 20 + 1) +
  1 / (Real.log 4 / Real.log 15 + 1) +
  1 / (Real.log 7 / Real.log 12 + 1) = 2 := by
  sorry

end logarithm_sum_simplification_l1955_195519


namespace oil_cylinder_capacity_l1955_195504

theorem oil_cylinder_capacity : ∀ (C : ℚ),
  (4 / 5 : ℚ) * C - (3 / 4 : ℚ) * C = 4 →
  C = 80 := by
sorry

end oil_cylinder_capacity_l1955_195504


namespace amy_homework_rate_l1955_195500

/-- Given a total number of problems and the time taken to complete them,
    calculate the number of problems completed per hour. -/
def problems_per_hour (total_problems : ℕ) (total_hours : ℕ) : ℚ :=
  total_problems / total_hours

/-- Theorem stating that with 24 problems completed in 6 hours,
    the number of problems completed per hour is 4. -/
theorem amy_homework_rate :
  problems_per_hour 24 6 = 4 := by
  sorry

end amy_homework_rate_l1955_195500


namespace one_nonnegative_solution_l1955_195557

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end one_nonnegative_solution_l1955_195557


namespace ratio_and_mean_determine_a_l1955_195508

theorem ratio_and_mean_determine_a (a b c : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  (a : ℚ) / c = 2 / 4 →
  (b : ℚ) / c = 3 / 4 →
  (a + b + c : ℚ) / 3 = 42 →
  a = 28 := by
sorry

end ratio_and_mean_determine_a_l1955_195508


namespace sum_of_integers_l1955_195586

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 181) (h2 : x.val * y.val = 90) :
  x.val + y.val = 19 := by
sorry

end sum_of_integers_l1955_195586


namespace range_of_expression_l1955_195587

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  ∃ (min max : ℝ), min = Real.sqrt 5 ∧ max = Real.sqrt 53 ∧
  min ≤ Real.sqrt (2*x^2 + y^2 - 4*x + 5) ∧
  Real.sqrt (2*x^2 + y^2 - 4*x + 5) ≤ max :=
sorry

end range_of_expression_l1955_195587


namespace mike_video_game_days_l1955_195513

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of hours Mike watches TV per day -/
def tv_hours_per_day : ℕ := 4

/-- The total hours Mike spends on TV and video games in a week -/
def total_hours_per_week : ℕ := 34

/-- The number of days Mike plays video games in a week -/
def video_game_days : ℕ := 3

theorem mike_video_game_days :
  ∃ (video_game_hours_per_day : ℕ),
    video_game_hours_per_day = tv_hours_per_day / 2 ∧
    video_game_days * video_game_hours_per_day =
      total_hours_per_week - (days_in_week * tv_hours_per_day) :=
by
  sorry

end mike_video_game_days_l1955_195513


namespace line_intersection_range_l1955_195593

theorem line_intersection_range (m : ℝ) : 
  (∀ x y : ℝ, y = (m + 1) * x + m - 1 → (x = 0 → y ≤ 0)) → m ≤ 1 := by
  sorry

end line_intersection_range_l1955_195593


namespace multiple_exists_l1955_195571

theorem multiple_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h : ∀ i, 0 < a i ∧ a i ≤ 2*n) : 
  ∃ i j, i ≠ j ∧ (a i ∣ a j ∨ a j ∣ a i) := by
  sorry

end multiple_exists_l1955_195571


namespace root_implies_a_value_l1955_195533

theorem root_implies_a_value (a : ℝ) : (1 : ℝ)^2 - 2*(1 : ℝ) + a = 0 → a = 1 := by
  sorry

end root_implies_a_value_l1955_195533


namespace shrinking_cities_proportion_comparison_l1955_195541

/-- Represents a circle in Hubei province -/
structure Circle where
  total_cities : ℕ
  shrinking_cities : ℕ
  shrinking_cities_le_total : shrinking_cities ≤ total_cities

/-- Calculates the proportion of shrinking cities in a circle -/
def shrinking_proportion (c : Circle) : ℚ :=
  c.shrinking_cities / c.total_cities

theorem shrinking_cities_proportion_comparison 
  (west : Circle)
  (middle : Circle)
  (east : Circle)
  (hw : west.total_cities = 5 ∧ west.shrinking_cities = 5)
  (hm : middle.total_cities = 13 ∧ middle.shrinking_cities = 9)
  (he : east.total_cities = 18 ∧ east.shrinking_cities = 13) :
  shrinking_proportion middle < shrinking_proportion west ∧ 
  shrinking_proportion middle < shrinking_proportion east :=
sorry

end shrinking_cities_proportion_comparison_l1955_195541


namespace perpendicular_to_countless_lines_iff_perpendicular_to_plane_l1955_195505

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is perpendicular to a plane -/
def Line.perpendicular_to_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- Defines when a line is perpendicular to countless lines within a plane -/
def Line.perpendicular_to_countless_lines_in_plane (l : Line) (a : Plane) : Prop :=
  sorry

/-- 
  The statement that a line being perpendicular to countless lines within a plane
  is a necessary and sufficient condition for the line being perpendicular to the plane
-/
theorem perpendicular_to_countless_lines_iff_perpendicular_to_plane
  (l : Line) (a : Plane) :
  Line.perpendicular_to_countless_lines_in_plane l a ↔ Line.perpendicular_to_plane l a :=
sorry

end perpendicular_to_countless_lines_iff_perpendicular_to_plane_l1955_195505


namespace second_number_proof_l1955_195592

theorem second_number_proof (x : ℤ) (h1 : x + (x + 4) = 56) : x + 4 = 30 := by
  sorry

end second_number_proof_l1955_195592


namespace find_principal_l1955_195596

/-- Given a sum of money P (principal) and an interest rate R,
    calculate the amount after T years with simple interest. -/
def simpleInterest (P R T : ℚ) : ℚ :=
  P + (P * R * T) / 100

theorem find_principal (R : ℚ) :
  ∃ P : ℚ,
    simpleInterest P R 1 = 1717 ∧
    simpleInterest P R 2 = 1734 ∧
    P = 1700 := by
  sorry

end find_principal_l1955_195596


namespace sum_of_coefficients_l1955_195507

/-- A quadratic function passing through (-3,0) and (5,0) with maximum value 76 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  passes_through_five : a * 5^2 + b * 5 + c = 0
  max_value : ∃ x, a * x^2 + b * x + c = 76
  is_max : ∀ x, a * x^2 + b * x + c ≤ 76

/-- The sum of coefficients of the quadratic function is 76 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 76 := by
  sorry


end sum_of_coefficients_l1955_195507


namespace set_relations_theorem_l1955_195506

universe u

theorem set_relations_theorem (U : Type u) (A B : Set U) : 
  (A ∩ B = ∅ → (Set.compl A ∪ Set.compl B) = Set.univ) ∧
  (A ∪ B = Set.univ → (Set.compl A ∩ Set.compl B) = ∅) ∧
  (A ∪ B = ∅ → A = ∅ ∧ B = ∅) := by
  sorry

end set_relations_theorem_l1955_195506


namespace inequality_theorem_l1955_195565

theorem inequality_theorem (p q r s t u : ℝ) 
  (h1 : p^2 < s^2) (h2 : q^2 < t^2) (h3 : r^2 < u^2) :
  p^2 * q^2 + q^2 * r^2 + r^2 * p^2 < s^2 * t^2 + t^2 * u^2 + u^2 * s^2 := by
  sorry

end inequality_theorem_l1955_195565


namespace product_equality_l1955_195585

theorem product_equality (a b c d e f : ℝ) 
  (sum_zero : a + b + c + d + e + f = 0)
  (sum_cubes_zero : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) :
  (a+c)*(a+d)*(a+e)*(a+f) = (b+c)*(b+d)*(b+e)*(b+f) := by
  sorry

end product_equality_l1955_195585


namespace prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l1955_195516

-- Define the contents of the bags
def bag_A : ℕ × ℕ := (2, 2)  -- (red balls, white balls)
def bag_B (n : ℕ) : ℕ × ℕ := (2, n)  -- (red balls, white balls)

-- Define the probability of drawing all red balls
def prob_all_red (n : ℕ) : ℚ :=
  (Nat.choose 2 2 * Nat.choose 2 2) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

-- Define the probability of drawing at least 2 red balls
def prob_at_least_2_red (n : ℕ) : ℚ :=
  1 - (Nat.choose 2 2 * Nat.choose n 2 + Nat.choose 2 1 * Nat.choose 2 1 * Nat.choose n 2 + Nat.choose 2 2 * Nat.choose 2 1 * Nat.choose n 1) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

theorem prob_all_red_when_n_3 :
  prob_all_red 3 = 1 / 60 := by sorry

theorem n_value_when_prob_at_least_2_red_is_3_4 :
  ∃ n : ℕ, prob_at_least_2_red n = 3 / 4 ∧ n = 2 := by sorry

end prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l1955_195516


namespace triangle_side_length_l1955_195573

theorem triangle_side_length (a b c : ℝ) (C : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : C = π/3) :
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) → c = Real.sqrt 3 :=
by
  sorry

end triangle_side_length_l1955_195573


namespace expression_equality_l1955_195599

theorem expression_equality :
  ((-3)^2 ≠ -3^2) ∧
  ((-3)^2 = 3^2) ∧
  ((-2)^3 = -2^3) ∧
  (|-2|^3 = |-2^3|) :=
by sorry

end expression_equality_l1955_195599


namespace complex_equation_solution_l1955_195509

theorem complex_equation_solution (z : ℂ) (h : (2 - Complex.I) * z = 5) : z = 2 + Complex.I := by
  sorry

end complex_equation_solution_l1955_195509


namespace estimate_fish_population_l1955_195568

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_recaught : ℕ)
  (h1 : initial_catch = 100)
  (h2 : second_catch = 200)
  (h3 : marked_recaught = 5) :
  (initial_catch * second_catch) / marked_recaught = 4000 :=
sorry

end estimate_fish_population_l1955_195568


namespace problem_solution_l1955_195539

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h_a : a > 0) (h_a_neq_1 : a ≠ 1) :
  -- Part 1
  (∀ x, f a 2 x = -f a 2 (-x)) →
  -- Part 2
  f a 2 1 < 0 →
  (∀ x t, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0 ↔ -3 < t ∧ t < 5) ∧
  (∀ x y, x < y → f a 2 y < f a 2 x) →
  -- Part 3
  f a 2 1 = 3/2 →
  (∃ m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) →
  (∃! m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2 ∧
               (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2)) →
  ∃ m, m = 2 ∧
    (∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) ∧
    (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2) := by
  sorry


end problem_solution_l1955_195539


namespace percentage_of_boys_l1955_195518

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 := by
  sorry

end percentage_of_boys_l1955_195518


namespace butter_calculation_l1955_195563

/-- Calculates the required amount of butter given a change in sugar amount -/
def required_butter (original_butter original_sugar new_sugar : ℚ) : ℚ :=
  (new_sugar / original_sugar) * original_butter

theorem butter_calculation (original_butter original_sugar new_sugar : ℚ) 
  (h1 : original_butter = 25)
  (h2 : original_sugar = 125)
  (h3 : new_sugar = 1000) :
  required_butter original_butter original_sugar new_sugar = 200 := by
  sorry

#eval required_butter 25 125 1000

end butter_calculation_l1955_195563


namespace hawk_percentage_l1955_195523

theorem hawk_percentage (total : ℝ) (hawk paddyfield kingfisher other : ℝ) : 
  total > 0 ∧
  hawk ≥ 0 ∧ paddyfield ≥ 0 ∧ kingfisher ≥ 0 ∧ other ≥ 0 ∧
  hawk + paddyfield + kingfisher + other = total ∧
  paddyfield = 0.4 * (total - hawk) ∧
  kingfisher = 0.25 * paddyfield ∧
  other = 0.35 * total →
  hawk = 0.3 * total :=
by sorry

end hawk_percentage_l1955_195523


namespace triangle_inequality_l1955_195515

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = π)
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)) :
  A * a + B * b + C * c ≥ (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) := by
  sorry

end triangle_inequality_l1955_195515


namespace nicky_running_time_l1955_195517

/-- Proves that Nicky runs for 60 seconds before Cristina catches up to him in a race with given conditions -/
theorem nicky_running_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 1500)
  (h2 : head_start = 25)
  (h3 : cristina_speed = 6)
  (h4 : nicky_speed = 3.5) :
  ∃ (t : ℝ), t = 60 ∧ cristina_speed * (t - head_start) = nicky_speed * t :=
by sorry

end nicky_running_time_l1955_195517


namespace budget_allocation_l1955_195512

theorem budget_allocation (research_dev : ℝ) (utilities : ℝ) (equipment : ℝ) (supplies : ℝ) 
  (transportation_degrees : ℝ) (total_degrees : ℝ) :
  research_dev = 9 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  transportation_degrees = 72 →
  total_degrees = 360 →
  let transportation := (transportation_degrees / total_degrees) * 100
  let other_categories := research_dev + utilities + equipment + supplies + transportation
  let salaries := 100 - other_categories
  salaries = 60 := by
sorry

end budget_allocation_l1955_195512


namespace vector_magnitude_cosine_sine_l1955_195547

theorem vector_magnitude_cosine_sine (α : Real) : 
  let a : Fin 2 → Real := ![Real.cos α, Real.sin α]
  ‖a‖ = 1 := by
  sorry

end vector_magnitude_cosine_sine_l1955_195547


namespace abs_one_minus_sqrt_three_l1955_195560

theorem abs_one_minus_sqrt_three (h : Real.sqrt 3 > 1) :
  |1 - Real.sqrt 3| = Real.sqrt 3 - 1 := by
  sorry

end abs_one_minus_sqrt_three_l1955_195560


namespace fifth_island_not_maya_l1955_195570

-- Define the types of residents
inductive Resident
| Knight
| Liar

-- Define the possible island names
inductive IslandName
| Maya
| NotMaya

-- Define the statements made by A and B
def statement_A (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Liar ∧ resident_B = Resident.Liar) ∧ island = IslandName.Maya

def statement_B (resident_A resident_B : Resident) (island : IslandName) : Prop :=
  (resident_A = Resident.Knight ∨ resident_B = Resident.Knight) ∧ island = IslandName.NotMaya

-- Define the truthfulness of statements based on the resident type
def is_truthful (r : Resident) (s : Prop) : Prop :=
  (r = Resident.Knight ∧ s) ∨ (r = Resident.Liar ∧ ¬s)

-- Theorem statement
theorem fifth_island_not_maya :
  ∀ (resident_A resident_B : Resident) (island : IslandName),
    is_truthful resident_A (statement_A resident_A resident_B island) →
    is_truthful resident_B (statement_B resident_A resident_B island) →
    island = IslandName.NotMaya :=
sorry

end fifth_island_not_maya_l1955_195570


namespace part_to_whole_ratio_l1955_195530

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : P + 6 = N / 4 - 6) : 
  (P + 6) / N = 9 / 40 := by
  sorry

end part_to_whole_ratio_l1955_195530


namespace jerry_pills_clock_time_l1955_195540

theorem jerry_pills_clock_time (total_pills : ℕ) (interval : ℕ) (start_time : ℕ) : 
  total_pills = 150 →
  interval = 5 →
  start_time = 12 →
  (start_time + (total_pills - 1) * interval) % 12 = 1 :=
by sorry

end jerry_pills_clock_time_l1955_195540


namespace range_of_a_l1955_195551

theorem range_of_a (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (x : ℤ), (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  (15 < a ∧ a ≤ 18) := by
sorry

end range_of_a_l1955_195551


namespace scientific_notation_of_7413000000_l1955_195502

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_7413000000 :
  toScientificNotation 7413000000 = ScientificNotation.mk 7.413 9 := by
  sorry

end scientific_notation_of_7413000000_l1955_195502


namespace convex_polygon_25_sides_l1955_195531

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- Number of diagonals in a polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Sum of interior angles in a polygon with n sides (in degrees) -/
def sumInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

theorem convex_polygon_25_sides :
  let p : ConvexPolygon 25 := ⟨by norm_num⟩
  numDiagonals 25 = 275 ∧ sumInteriorAngles 25 = 4140 := by sorry

end convex_polygon_25_sides_l1955_195531


namespace questionnaires_from_unit_D_l1955_195594

/-- Represents the number of questionnaires drawn from each unit -/
structure SampleDistribution where
  unitA : ℕ
  unitB : ℕ
  unitC : ℕ
  unitD : ℕ

/-- The sample distribution forms an arithmetic sequence -/
def is_arithmetic_sequence (s : SampleDistribution) : Prop :=
  s.unitB - s.unitA = s.unitC - s.unitB ∧ s.unitC - s.unitB = s.unitD - s.unitC

/-- The total sample size is 150 -/
def total_sample_size (s : SampleDistribution) : ℕ :=
  s.unitA + s.unitB + s.unitC + s.unitD

theorem questionnaires_from_unit_D 
  (s : SampleDistribution)
  (h1 : is_arithmetic_sequence s)
  (h2 : total_sample_size s = 150)
  (h3 : s.unitB = 30) :
  s.unitD = 60 := by
  sorry

end questionnaires_from_unit_D_l1955_195594


namespace parallel_vectors_m_value_l1955_195598

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = 3 := by
  sorry

#check parallel_vectors_m_value

end parallel_vectors_m_value_l1955_195598


namespace solution_set_implies_a_range_l1955_195532

theorem solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x > 1) ↔ x > 1) → a ≤ 1 := by
  sorry

end solution_set_implies_a_range_l1955_195532


namespace ratio_equivalence_l1955_195579

theorem ratio_equivalence (a b : ℚ) (h : 5 * a = 6 * b) :
  (a / b = 6 / 5) ∧ (b / a = 5 / 6) := by
  sorry

end ratio_equivalence_l1955_195579


namespace stanley_distance_difference_l1955_195588

-- Define the constants
def running_distance : ℝ := 4.8
def walking_distance_meters : ℝ := 950

-- Define the conversion factor
def meters_per_kilometer : ℝ := 1000

-- Define the theorem
theorem stanley_distance_difference :
  running_distance - (walking_distance_meters / meters_per_kilometer) = 3.85 := by
  sorry

end stanley_distance_difference_l1955_195588


namespace complex_power_magnitude_l1955_195546

theorem complex_power_magnitude : Complex.abs ((1 + Complex.I) ^ 8) = 16 := by
  sorry

end complex_power_magnitude_l1955_195546


namespace prob_sum_odd_is_13_27_l1955_195545

/-- Represents an unfair die where even numbers are twice as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability of rolling a sum of three rolls being odd -/
def prob_sum_odd (d : UnfairDie) : ℝ :=
  3 * d.odd_prob * d.even_prob^2 + d.odd_prob^3

/-- Theorem stating the probability of rolling a sum of three rolls being odd is 13/27 -/
theorem prob_sum_odd_is_13_27 (d : UnfairDie) : prob_sum_odd d = 13/27 := by
  sorry

end prob_sum_odd_is_13_27_l1955_195545


namespace unique_solution_l1955_195582

def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 = 1 ∧ p.1 + 4 * p.2 = 5}

theorem unique_solution : solution_set = {(1, 1)} := by
  sorry

end unique_solution_l1955_195582


namespace z_in_first_quadrant_l1955_195503

theorem z_in_first_quadrant :
  ∀ z : ℂ, (1 + Complex.I)^2 * z = -1 + Complex.I →
  (z.re > 0 ∧ z.im > 0) :=
by sorry

end z_in_first_quadrant_l1955_195503


namespace inequality_proof_l1955_195576

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) := by
  sorry

end inequality_proof_l1955_195576


namespace inverse_variation_problem_l1955_195583

/-- Given that x and y are positive real numbers, x² and y vary inversely,
    and y = 25 when x = 3, prove that x = √3/4 when y = 1200. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ k : ℝ, ∀ x y, x * x * y = k)
  (h_initial : 3 * 3 * 25 = 9 * 25) :
  y = 1200 → x = Real.sqrt 3 / 4 := by
sorry

end inverse_variation_problem_l1955_195583


namespace quadratic_inequality_l1955_195595

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4*a*c := by
  sorry

end quadratic_inequality_l1955_195595


namespace max_points_32_l1955_195538

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates for three-point and two-point shots
def three_point_rate : ℚ := 1/4
def two_point_rate : ℚ := 2/5

-- Define the function that calculates the total points based on the number of three-point attempts
def total_points (three_point_attempts : ℕ) : ℚ :=
  3 * three_point_rate * three_point_attempts + 
  2 * two_point_rate * (total_shots - three_point_attempts)

-- Theorem: The maximum number of points Jamal could score is 32
theorem max_points_32 : 
  ∀ x : ℕ, x ≤ total_shots → total_points x ≤ 32 :=
sorry

end max_points_32_l1955_195538


namespace geometric_sequence_property_l1955_195549

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a)
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 := by
sorry

end geometric_sequence_property_l1955_195549


namespace percentage_relation_l1955_195527

theorem percentage_relation (x y c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 := by
  sorry

end percentage_relation_l1955_195527


namespace tan_4530_degrees_l1955_195575

theorem tan_4530_degrees : Real.tan (4530 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_4530_degrees_l1955_195575


namespace no_prime_perfect_square_l1955_195535

theorem no_prime_perfect_square : ¬∃ (p : ℕ), Prime p ∧ ∃ (a : ℕ), 7 * p + 3^p - 4 = a^2 := by
  sorry

end no_prime_perfect_square_l1955_195535


namespace library_books_l1955_195564

theorem library_books (original_books : ℕ) : 
  (original_books + 140 = (27 : ℚ) / 25 * original_books) → 
  original_books = 1750 := by
  sorry

end library_books_l1955_195564


namespace root_product_theorem_l1955_195591

theorem root_product_theorem (x₁ x₂ x₃ : ℝ) : 
  x₃ < x₂ ∧ x₂ < x₁ →
  (Real.sqrt 120 * x₁^3 - 480 * x₁^2 + 8 * x₁ + 1 = 0) →
  (Real.sqrt 120 * x₂^3 - 480 * x₂^2 + 8 * x₂ + 1 = 0) →
  (Real.sqrt 120 * x₃^3 - 480 * x₃^2 + 8 * x₃ + 1 = 0) →
  x₂ * (x₁ + x₃) = -1/120 := by
sorry

end root_product_theorem_l1955_195591


namespace quadratic_roots_unique_l1955_195534

theorem quadratic_roots_unique (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end quadratic_roots_unique_l1955_195534


namespace solution_set_part1_range_of_a_part2_l1955_195581

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end solution_set_part1_range_of_a_part2_l1955_195581


namespace vasyas_numbers_l1955_195567

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end vasyas_numbers_l1955_195567


namespace expression_evaluation_l1955_195552

theorem expression_evaluation :
  let w : ℤ := 3
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  w^2 * x^2 * y * z - w * x^2 * y * z^2 + w * y^3 * z^2 - w * y^2 * x * z^4 = 1536 := by
  sorry

end expression_evaluation_l1955_195552


namespace gumball_probability_l1955_195572

/-- Represents a jar of gumballs -/
structure GumballJar where
  blue : ℕ
  pink : ℕ

/-- The probability of drawing a blue gumball -/
def prob_blue (jar : GumballJar) : ℚ :=
  jar.blue / (jar.blue + jar.pink)

/-- The probability of drawing a pink gumball -/
def prob_pink (jar : GumballJar) : ℚ :=
  jar.pink / (jar.blue + jar.pink)

theorem gumball_probability (jar : GumballJar) :
  (prob_blue jar) ^ 2 = 36 / 49 →
  prob_pink jar = 1 / 7 := by
  sorry

end gumball_probability_l1955_195572


namespace dice_sides_for_given_probability_l1955_195524

theorem dice_sides_for_given_probability (n : ℕ+) : 
  (((6 : ℝ) / (n : ℝ)^2)^2 = 0.027777777777777776) → n = 6 := by
  sorry

end dice_sides_for_given_probability_l1955_195524


namespace perpendicular_lines_b_value_l1955_195574

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + y + c = 0 is -a -/
axiom line_slope (a c : ℝ) : ∃ (m : ℝ), m = -a ∧ ∀ (x y : ℝ), a * x + y + c = 0 → y = m * x - c

theorem perpendicular_lines_b_value :
  ∀ (b : ℝ), 
  (∀ (x y : ℝ), 3 * x + y - 5 = 0 → bx + y + 2 = 0 → 
    ∃ (m₁ m₂ : ℝ), (m₁ * m₂ = -1 ∧ 
      (∀ (x₁ y₁ : ℝ), 3 * x₁ + y₁ - 5 = 0 → y₁ = m₁ * x₁ + 5) ∧
      (∀ (x₂ y₂ : ℝ), b * x₂ + y₂ + 2 = 0 → y₂ = m₂ * x₂ - 2))) →
  b = -1/3 := by
sorry

end perpendicular_lines_b_value_l1955_195574


namespace complex_number_in_first_quadrant_l1955_195543

/-- The complex number z = i / (1 + i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l1955_195543


namespace factorial_sum_equality_l1955_195521

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 3 = 35904 := by
  sorry

end factorial_sum_equality_l1955_195521


namespace complex_fraction_simplification_l1955_195590

theorem complex_fraction_simplification (a : ℝ) (h : a ≠ 2) :
  let x := Real.rpow (Real.sqrt 5 - Real.sqrt 3) (1/3) * Real.rpow (8 + 2 * Real.sqrt 15) (1/6) - Real.rpow a (1/3)
  let y := Real.rpow (Real.sqrt 20 + Real.sqrt 12) (1/3) * Real.rpow (8 - 2 * Real.sqrt 15) (1/6) - 2 * Real.rpow (2*a) (1/3) + Real.rpow (a^2) (1/3)
  x / y = 1 / (Real.rpow 2 (1/3) - Real.rpow a (1/3)) :=
by sorry

end complex_fraction_simplification_l1955_195590


namespace routes_on_grid_l1955_195520

/-- The number of routes on a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_on_grid : 
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end routes_on_grid_l1955_195520


namespace line_properties_l1955_195536

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given the equation of a line y + 7 = -x - 3, prove it passes through (-3, -7) with slope -1 -/
theorem line_properties :
  let l : Line := { slope := -1, yIntercept := -10 }
  let p : Point := { x := -3, y := -7 }
  (p.y + 7 = -p.x - 3) ∧ 
  (l.slope = -1) ∧
  (p.y = l.slope * p.x + l.yIntercept) := by
  sorry


end line_properties_l1955_195536


namespace cookies_difference_l1955_195550

theorem cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end cookies_difference_l1955_195550
