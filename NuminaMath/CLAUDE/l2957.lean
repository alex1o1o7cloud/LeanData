import Mathlib

namespace NUMINAMATH_CALUDE_marble_probability_l2957_295755

theorem marble_probability (red blue green : ℕ) 
  (h_red : red = 4) 
  (h_blue : blue = 3) 
  (h_green : green = 6) : 
  (red + blue : ℚ) / (red + blue + green) = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l2957_295755


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2957_295762

theorem trigonometric_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + Real.cos (156 * π / 180) * Real.cos (106 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2957_295762


namespace NUMINAMATH_CALUDE_complex_roots_of_equation_l2957_295718

theorem complex_roots_of_equation : ∃ (z₁ z₂ : ℂ),
  z₁ = -1 + 2 * Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₂ = -1 - 2 * Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * Complex.I ∧
  z₁^2 + 2*z₁ = 16 + 8*Complex.I ∧
  z₂^2 + 2*z₂ = 16 + 8*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_of_equation_l2957_295718


namespace NUMINAMATH_CALUDE_pencils_at_meeting_pencils_at_meeting_proof_l2957_295773

/-- The number of pencils brought to a committee meeting -/
theorem pencils_at_meeting : ℕ :=
  let associate_prof : ℕ → ℕ := λ x ↦ x  -- Number of associate professors
  let assistant_prof : ℕ → ℕ := λ x ↦ x  -- Number of assistant professors
  let total_people : ℕ := 7  -- Total number of people at the meeting
  let total_charts : ℕ := 11  -- Total number of charts brought to the meeting
  let pencils_per_associate : ℕ := 2  -- Pencils brought by each associate professor
  let pencils_per_assistant : ℕ := 1  -- Pencils brought by each assistant professor
  let charts_per_associate : ℕ := 1  -- Charts brought by each associate professor
  let charts_per_assistant : ℕ := 2  -- Charts brought by each assistant professor

  10  -- The theorem states that the number of pencils is 10

theorem pencils_at_meeting_proof :
  ∀ (x y : ℕ),
  x + y = total_people →
  charts_per_associate * x + charts_per_assistant * y = total_charts →
  pencils_per_associate * x + pencils_per_assistant * y = pencils_at_meeting :=
by
  sorry

#check pencils_at_meeting
#check pencils_at_meeting_proof

end NUMINAMATH_CALUDE_pencils_at_meeting_pencils_at_meeting_proof_l2957_295773


namespace NUMINAMATH_CALUDE_base_85_problem_l2957_295791

/-- Represents a number in base 85 --/
def BaseEightyFive : Type := List Nat

/-- Converts a number in base 85 to its decimal representation --/
def to_decimal (n : BaseEightyFive) : Nat :=
  sorry

/-- The specific number 3568432 in base 85 --/
def number : BaseEightyFive :=
  [3, 5, 6, 8, 4, 3, 2]

theorem base_85_problem (b : Int) 
  (h1 : 0 ≤ b) (h2 : b ≤ 19) 
  (h3 : (to_decimal number - b) % 17 = 0) : 
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_85_problem_l2957_295791


namespace NUMINAMATH_CALUDE_simplify_expression_l2957_295711

theorem simplify_expression (x y : ℝ) (h : x ≠ y) :
  (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2957_295711


namespace NUMINAMATH_CALUDE_range_of_m_l2957_295775

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2957_295775


namespace NUMINAMATH_CALUDE_gold_value_calculation_l2957_295757

/-- The total value of gold for Legacy and Aleena -/
def total_gold_value (legacy_bars : ℕ) (aleena_difference : ℕ) (bar_value : ℕ) : ℕ :=
  (legacy_bars + (legacy_bars - aleena_difference)) * bar_value

/-- Theorem stating the total value of gold for Legacy and Aleena -/
theorem gold_value_calculation :
  total_gold_value 12 4 3500 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_gold_value_calculation_l2957_295757


namespace NUMINAMATH_CALUDE_cakes_baked_lunch_is_eight_l2957_295797

/-- The number of cakes baked during lunch today -/
def cakes_baked_lunch : ℕ := sorry

/-- The number of cakes sold during dinner -/
def cakes_sold_dinner : ℕ := 6

/-- The number of cakes baked yesterday -/
def cakes_baked_yesterday : ℕ := 3

/-- The number of cakes left -/
def cakes_left : ℕ := 2

/-- Theorem stating that the number of cakes baked during lunch today is 8 -/
theorem cakes_baked_lunch_is_eight :
  cakes_baked_lunch = 8 :=
by sorry

end NUMINAMATH_CALUDE_cakes_baked_lunch_is_eight_l2957_295797


namespace NUMINAMATH_CALUDE_sandbox_area_l2957_295750

theorem sandbox_area (length width : ℕ) (h1 : length = 312) (h2 : width = 146) :
  length * width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l2957_295750


namespace NUMINAMATH_CALUDE_crosswalk_wait_probability_l2957_295756

/-- Represents the duration of the red light in seconds -/
def red_light_duration : ℝ := 40

/-- Represents the minimum waiting time in seconds -/
def min_wait_time : ℝ := 15

/-- Theorem: The probability of waiting at least 15 seconds for a green light
    when encountering a red light that lasts 40 seconds is 5/8 -/
theorem crosswalk_wait_probability :
  (red_light_duration - min_wait_time) / red_light_duration = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_wait_probability_l2957_295756


namespace NUMINAMATH_CALUDE_problem_statement_l2957_295793

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (9/5 ≤ a^2 + b^2 ∧ a^2 + b^2 < 9) ∧ a^3*b + 4*a*b^3 ≤ 81/16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2957_295793


namespace NUMINAMATH_CALUDE_mans_age_ratio_l2957_295743

theorem mans_age_ratio (mans_age father_age : ℕ) : 
  father_age = 60 →
  mans_age + 12 = (father_age + 12) / 2 →
  mans_age * 5 = father_age * 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_ratio_l2957_295743


namespace NUMINAMATH_CALUDE_ratio_of_sequences_l2957_295760

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def numerator_sequence : ℚ := arithmetic_sum 2 2 17
def denominator_sequence : ℚ := arithmetic_sum 3 3 17

theorem ratio_of_sequences : numerator_sequence / denominator_sequence = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sequences_l2957_295760


namespace NUMINAMATH_CALUDE_a_value_satisfies_condition_l2957_295746

-- Define the property that needs to be satisfied
def satisfies_condition (a : ℕ) : Prop :=
  ∀ K : ℤ, K ≠ 27 → (27 - K) ∣ (a - K^1964)

-- State the theorem
theorem a_value_satisfies_condition :
  satisfies_condition (3^5892) :=
sorry

end NUMINAMATH_CALUDE_a_value_satisfies_condition_l2957_295746


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l2957_295777

theorem line_slope_intercept_product (m b : ℚ) : 
  m = 3/5 → b = -3/2 → -1 < m * b ∧ m * b < 0 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l2957_295777


namespace NUMINAMATH_CALUDE_base_number_irrelevant_l2957_295759

def decimal_places (n : ℝ) : ℕ := sorry

theorem base_number_irrelevant (x : ℤ) :
  decimal_places ((x^4 * 3.456789)^14) = decimal_places (3.456789^14) := by sorry

end NUMINAMATH_CALUDE_base_number_irrelevant_l2957_295759


namespace NUMINAMATH_CALUDE_track_length_l2957_295798

/-- The length of a circular track given specific running conditions -/
theorem track_length (first_lap : ℝ) (other_laps : ℝ) (avg_speed : ℝ) : 
  first_lap = 70 →
  other_laps = 85 →
  avg_speed = 5 →
  (3 : ℝ) * (first_lap + 2 * other_laps) * avg_speed / 3 = 400 := by
  sorry

end NUMINAMATH_CALUDE_track_length_l2957_295798


namespace NUMINAMATH_CALUDE_button_probability_l2957_295741

/-- Represents a jar containing buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / (j.red + j.blue)

theorem button_probability (initialJarA initialJarB finalJarA finalJarB : Jar) :
  initialJarA.red = 5 →
  initialJarA.blue = 10 →
  initialJarB.red = 0 →
  initialJarB.blue = 0 →
  finalJarA.red + finalJarB.red = initialJarA.red →
  finalJarA.blue + finalJarB.blue = initialJarA.blue →
  finalJarB.red = finalJarB.blue / 2 →
  finalJarA.red + finalJarA.blue = (3 * (initialJarA.red + initialJarA.blue)) / 5 →
  redProbability finalJarA = 1/3 ∧ redProbability finalJarB = 1/3 ∧
  redProbability finalJarA * redProbability finalJarB = 1/9 := by
  sorry


end NUMINAMATH_CALUDE_button_probability_l2957_295741


namespace NUMINAMATH_CALUDE_prime_difference_theorem_l2957_295724

theorem prime_difference_theorem (x y : ℝ) 
  (h1 : Prime (⌊x - y⌋ : ℤ))
  (h2 : Prime (⌊x^2 - y^2⌋ : ℤ))
  (h3 : Prime (⌊x^3 - y^3⌋ : ℤ)) :
  x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_difference_theorem_l2957_295724


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l2957_295742

theorem right_triangle_increase_sides_acute (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 →
  c^2 = a^2 + b^2 →  -- right-angled triangle condition
  (a + x)^2 + (b + x)^2 > (c + x)^2  -- acute triangle condition
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l2957_295742


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l2957_295738

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectangleDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a rectangle given its dimensions and the size of small rectangles -/
def calculatePerimeter (dim : RectangleDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  calculatePerimeter ⟨6, 1⟩ x y = 56 →
  calculatePerimeter ⟨4, 3⟩ x y = 56 →
  calculatePerimeter ⟨2, 3⟩ x y = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l2957_295738


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l2957_295787

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 316

/-- The cost price per bowl in rupees -/
def cost_price : ℚ := 12

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 15

/-- The number of bowls sold -/
def bowls_sold : ℕ := 102

/-- The percentage gain -/
def percentage_gain : ℚ := 8050847457627118 / 1000000000000000

theorem glass_bowls_problem :
  initial_bowls = 316 ∧
  (bowls_sold : ℚ) * (selling_price - cost_price) / (initial_bowls * cost_price) = percentage_gain / 100 := by
  sorry


end NUMINAMATH_CALUDE_glass_bowls_problem_l2957_295787


namespace NUMINAMATH_CALUDE_corner_sum_equals_164_l2957_295754

-- Define the size of the checkerboard
def boardSize : Nat := 9

-- Define a function to get the number at a specific position
def getNumber (row : Nat) (col : Nat) : Nat :=
  if row % 2 = 1 then
    (row - 1) * boardSize + col
  else
    row * boardSize - col + 1

-- Theorem statement
theorem corner_sum_equals_164 :
  getNumber 1 1 + getNumber 1 boardSize + getNumber boardSize 1 + getNumber boardSize boardSize = 164 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_equals_164_l2957_295754


namespace NUMINAMATH_CALUDE_kite_area_in_regular_hexagon_l2957_295735

/-- The area of a kite-shaped region in a regular hexagon -/
theorem kite_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 8) :
  let radius := side_length
  let angle := 120 * π / 180
  let kite_area := (1 / 2) * radius * radius * Real.sin angle
  kite_area = 16 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_kite_area_in_regular_hexagon_l2957_295735


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l2957_295767

def apples : ℝ := 280
def oranges : ℝ := 150
def bananas : ℝ := 100

def apples_high_profit_ratio : ℝ := 0.4
def oranges_high_profit_ratio : ℝ := 0.45
def bananas_high_profit_ratio : ℝ := 0.5

def apples_high_profit_percentage : ℝ := 0.2
def oranges_high_profit_percentage : ℝ := 0.25
def bananas_high_profit_percentage : ℝ := 0.3

def low_profit_percentage : ℝ := 0.15

def total_fruits : ℝ := apples + oranges + bananas

theorem overall_profit_percentage (ε : ℝ) (h : ε > 0) :
  ∃ (profit_percentage : ℝ),
    abs (profit_percentage - 0.1875) < ε ∧
    profit_percentage = 
      (apples_high_profit_ratio * apples * apples_high_profit_percentage +
       oranges_high_profit_ratio * oranges * oranges_high_profit_percentage +
       bananas_high_profit_ratio * bananas * bananas_high_profit_percentage +
       (1 - apples_high_profit_ratio) * apples * low_profit_percentage +
       (1 - oranges_high_profit_ratio) * oranges * low_profit_percentage +
       (1 - bananas_high_profit_ratio) * bananas * low_profit_percentage) /
      total_fruits :=
by sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l2957_295767


namespace NUMINAMATH_CALUDE_mean_of_class_scores_l2957_295765

def class_scores : List ℕ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_of_class_scores : 
  (List.sum class_scores) / (List.length class_scores) = 48 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_class_scores_l2957_295765


namespace NUMINAMATH_CALUDE_february_first_is_friday_l2957_295702

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in February -/
structure FebruaryDay where
  date : Nat
  weekday : Weekday

/-- Represents the condition of the student groups visiting Teacher Li -/
structure StudentVisit where
  day : FebruaryDay
  groupSize : Nat

/-- The main theorem -/
theorem february_first_is_friday 
  (visit : StudentVisit)
  (h1 : visit.day.weekday = Weekday.Sunday)
  (h2 : visit.day.date = 3 * visit.groupSize * visit.groupSize)
  (h3 : visit.groupSize > 1)
  : (⟨1, Weekday.Friday⟩ : FebruaryDay) = 
    {date := 1, weekday := Weekday.Friday} :=
by sorry

end NUMINAMATH_CALUDE_february_first_is_friday_l2957_295702


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2957_295789

theorem inequality_system_solution_range (m : ℝ) : 
  (∀ x : ℝ, ((x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - m ≥ x) ↔ x ≥ m) → 
  m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2957_295789


namespace NUMINAMATH_CALUDE_water_equals_sugar_in_new_recipe_l2957_295730

/-- Represents a recipe with ratios of flour, water, and sugar -/
structure Recipe :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Creates a new recipe by doubling the flour to water ratio and halving the flour to sugar ratio -/
def newRecipe (r : Recipe) : Recipe :=
  { flour := r.flour * 2,
    water := r.water,
    sugar := r.sugar / 2 }

/-- Calculates the amount of water needed given the amount of sugar and the recipe ratios -/
def waterNeeded (r : Recipe) (sugarAmount : ℚ) : ℚ :=
  (r.water / r.sugar) * sugarAmount

theorem water_equals_sugar_in_new_recipe (originalRecipe : Recipe) (sugarAmount : ℚ) :
  let newRecipe := newRecipe originalRecipe
  waterNeeded newRecipe sugarAmount = sugarAmount :=
by sorry

#check water_equals_sugar_in_new_recipe

end NUMINAMATH_CALUDE_water_equals_sugar_in_new_recipe_l2957_295730


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2957_295708

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 13) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2957_295708


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l2957_295731

/-- 
Calculates the total revenue from movie ticket sales given the prices and quantities sold.
-/
theorem movie_theater_revenue 
  (matinee_price : ℕ) 
  (evening_price : ℕ) 
  (three_d_price : ℕ)
  (matinee_quantity : ℕ)
  (evening_quantity : ℕ)
  (three_d_quantity : ℕ)
  (h1 : matinee_price = 5)
  (h2 : evening_price = 12)
  (h3 : three_d_price = 20)
  (h4 : matinee_quantity = 200)
  (h5 : evening_quantity = 300)
  (h6 : three_d_quantity = 100) :
  matinee_price * matinee_quantity + 
  evening_price * evening_quantity + 
  three_d_price * three_d_quantity = 6600 :=
by
  sorry

#check movie_theater_revenue

end NUMINAMATH_CALUDE_movie_theater_revenue_l2957_295731


namespace NUMINAMATH_CALUDE_number_and_square_average_l2957_295714

theorem number_and_square_average (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5*x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_average_l2957_295714


namespace NUMINAMATH_CALUDE_det_B_equals_four_l2957_295779

theorem det_B_equals_four (b c : ℝ) (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B = ![![b, 2], ![-3, c]] →
  B + 2 * B⁻¹ = 0 →
  Matrix.det B = 4 := by
sorry

end NUMINAMATH_CALUDE_det_B_equals_four_l2957_295779


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2957_295709

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 8 * 2 / 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2957_295709


namespace NUMINAMATH_CALUDE_cos_double_angle_specific_l2957_295799

theorem cos_double_angle_specific (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cos_double_angle_specific_l2957_295799


namespace NUMINAMATH_CALUDE_problem_statement_l2957_295795

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m : ℝ, (∀ x₁ ∈ Set.Icc 0 (Real.pi / 2), ∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₁ + g x₂ ≥ m) → m ≤ -1 - Real.sqrt 2) ∧
  (∀ x > -1, f x - g x > 0) :=
by sorry

end

end NUMINAMATH_CALUDE_problem_statement_l2957_295795


namespace NUMINAMATH_CALUDE_usual_walking_time_l2957_295785

/-- Given a constant distance and the fact that walking at 40% of usual speed takes 24 minutes more, 
    the usual time to cover the distance is 16 minutes. -/
theorem usual_walking_time (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0)
  (h2 : usual_time > 0)
  (h3 : distance = usual_speed * usual_time)
  (h4 : distance = (0.4 * usual_speed) * (usual_time + 24)) :
  usual_time = 16 := by
sorry

end NUMINAMATH_CALUDE_usual_walking_time_l2957_295785


namespace NUMINAMATH_CALUDE_third_offense_sentence_extension_l2957_295758

theorem third_offense_sentence_extension (original_sentence total_time : ℕ) 
  (h1 : original_sentence = 27)
  (h2 : total_time = 36) :
  (total_time - original_sentence) / original_sentence = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_third_offense_sentence_extension_l2957_295758


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2957_295706

theorem binomial_expansion_coefficients 
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b) ^ n = 243)
  (h2 : (1 + |a|) ^ n = 32) : 
  a = 1 ∧ b = 2 ∧ n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_l2957_295706


namespace NUMINAMATH_CALUDE_calculation_proof_l2957_295734

theorem calculation_proof : (1/4) * 6.16^2 - 4 * 1.04^2 = 5.16 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2957_295734


namespace NUMINAMATH_CALUDE_expected_interval_is_three_minutes_l2957_295700

/-- Represents the train system with given conditions -/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  travel_time_difference : ℝ

/-- The expected interval between trains in one direction -/
def expected_interval (ts : TrainSystem) : ℝ :=
  3

/-- Theorem stating that the expected interval is 3 minutes -/
theorem expected_interval_is_three_minutes (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.travel_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three_minutes

end NUMINAMATH_CALUDE_expected_interval_is_three_minutes_l2957_295700


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2957_295781

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = 4 * a n) 
  (h_sum : a 1 + a 2 + a 3 = 21) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2957_295781


namespace NUMINAMATH_CALUDE_difference_of_trailing_zeros_l2957_295786

def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem difference_of_trailing_zeros : trailingZeros 300 - trailingZeros 280 = 5 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_trailing_zeros_l2957_295786


namespace NUMINAMATH_CALUDE_extremum_and_monotonicity_l2957_295728

/-- The function f(x) defined as e^x - ln(x + m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_and_monotonicity (m : ℝ) :
  (f_deriv m 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_deriv m x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_deriv m x < 0) := by
  sorry

#check extremum_and_monotonicity

end NUMINAMATH_CALUDE_extremum_and_monotonicity_l2957_295728


namespace NUMINAMATH_CALUDE_bob_water_percentage_l2957_295721

-- Define the water requirements for each crop
def water_corn : ℕ := 20
def water_cotton : ℕ := 80
def water_beans : ℕ := 2 * water_corn

-- Define the acreage for each farmer
def bob_corn : ℕ := 3
def bob_cotton : ℕ := 9
def bob_beans : ℕ := 12

def brenda_corn : ℕ := 6
def brenda_cotton : ℕ := 7
def brenda_beans : ℕ := 14

def bernie_corn : ℕ := 2
def bernie_cotton : ℕ := 12

-- Calculate water usage for each farmer
def bob_water : ℕ := bob_corn * water_corn + bob_cotton * water_cotton + bob_beans * water_beans
def brenda_water : ℕ := brenda_corn * water_corn + brenda_cotton * water_cotton + brenda_beans * water_beans
def bernie_water : ℕ := bernie_corn * water_corn + bernie_cotton * water_cotton

-- Calculate total water usage
def total_water : ℕ := bob_water + brenda_water + bernie_water

-- Define the theorem
theorem bob_water_percentage :
  (bob_water : ℚ) / total_water * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_bob_water_percentage_l2957_295721


namespace NUMINAMATH_CALUDE_largest_valid_marking_l2957_295739

/-- A marking function that assigns a boolean value to each cell in an n × n grid. -/
def Marking (n : ℕ) := Fin n → Fin n → Bool

/-- Predicate that checks if a rectangle contains a marked cell. -/
def ContainsMarkedCell (m : Marking n) (x y w h : Fin n) : Prop :=
  ∃ i j, i < w ∧ j < h ∧ m (x + i) (y + j) = true

/-- Predicate that checks if a marking satisfies the condition for all rectangles. -/
def ValidMarking (n : ℕ) (m : Marking n) : Prop :=
  ∀ x y w h : Fin n, w * h ≥ n → ContainsMarkedCell m x y w h

/-- The main theorem stating that 7 is the largest n for which a valid marking exists. -/
theorem largest_valid_marking :
  (∃ (m : Marking 7), ValidMarking 7 m) ∧
  (∀ n > 7, ¬∃ (m : Marking n), ValidMarking n m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_marking_l2957_295739


namespace NUMINAMATH_CALUDE_probability_sum_six_three_dice_l2957_295732

/-- A function that returns the number of ways to roll a sum of 6 with three dice -/
def waysToRollSixWithThreeDice : ℕ :=
  -- We don't implement the function, just declare it
  sorry

/-- The total number of possible outcomes when rolling three six-sided dice -/
def totalOutcomes : ℕ := 6^3

/-- The probability of rolling a sum of 6 with three fair six-sided dice -/
theorem probability_sum_six_three_dice :
  (waysToRollSixWithThreeDice : ℚ) / totalOutcomes = 5 / 108 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_three_dice_l2957_295732


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2957_295761

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 28*x + 24

theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-3, -3) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2957_295761


namespace NUMINAMATH_CALUDE_min_value_inequality_sum_squared_ratio_inequality_l2957_295764

-- Part 1
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 := by sorry

-- Part 2
theorem sum_squared_ratio_inequality (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = m) :
  a^2 / b + b^2 / c + c^2 / a ≥ m := by sorry

end NUMINAMATH_CALUDE_min_value_inequality_sum_squared_ratio_inequality_l2957_295764


namespace NUMINAMATH_CALUDE_root_sum_power_property_l2957_295783

theorem root_sum_power_property (x₁ x₂ : ℂ) (n : ℤ) : 
  x₁^2 - 6*x₁ + 1 = 0 → 
  x₂^2 - 6*x₂ + 1 = 0 → 
  (∃ m : ℤ, x₁^n + x₂^n = m) ∧ 
  ¬(∃ k : ℤ, x₁^n + x₂^n = 5*k) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_power_property_l2957_295783


namespace NUMINAMATH_CALUDE_two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l2957_295729

-- Definition of symbiotic rational number pair
def is_symbiotic_pair (a b : ℚ) : Prop := a - b = a * b + 1

-- Theorem 1: (2, 1/3) is a symbiotic rational number pair
theorem two_and_one_third_symbiotic : is_symbiotic_pair 2 (1/3) := by sorry

-- Theorem 2: (-1/3, -2) is a symbiotic rational number pair
theorem neg_one_third_and_neg_two_symbiotic : is_symbiotic_pair (-1/3) (-2) := by sorry

-- Theorem 3: If (m, n) is a symbiotic rational number pair, then (-n, -m) is also a symbiotic rational number pair
theorem symbiotic_pair_negation (m n : ℚ) : 
  is_symbiotic_pair m n → is_symbiotic_pair (-n) (-m) := by sorry

end NUMINAMATH_CALUDE_two_and_one_third_symbiotic_neg_one_third_and_neg_two_symbiotic_symbiotic_pair_negation_l2957_295729


namespace NUMINAMATH_CALUDE_elle_piano_practice_l2957_295766

/-- The number of minutes Elle practices piano on a weekday -/
def weekday_practice : ℕ := 30

/-- The number of minutes Elle practices piano on Saturday -/
def saturday_practice : ℕ := 3 * weekday_practice

/-- The total number of minutes Elle practices piano in a week -/
def total_practice : ℕ := 5 * weekday_practice + saturday_practice

theorem elle_piano_practice :
  weekday_practice = 30 ∧
  saturday_practice = 3 * weekday_practice ∧
  total_practice = 5 * weekday_practice + saturday_practice ∧
  total_practice = 4 * 60 := by
  sorry

end NUMINAMATH_CALUDE_elle_piano_practice_l2957_295766


namespace NUMINAMATH_CALUDE_tan_cot_15_sum_even_l2957_295712

theorem tan_cot_15_sum_even (n : ℕ+) : 
  ∃ k : ℤ, (2 - Real.sqrt 3) ^ n.val + (2 + Real.sqrt 3) ^ n.val = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_tan_cot_15_sum_even_l2957_295712


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l2957_295790

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisa_marbles_problem :
  min_additional_marbles 12 50 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l2957_295790


namespace NUMINAMATH_CALUDE_second_grade_students_selected_l2957_295736

/-- Given a school with 3300 students and a ratio of 12:10:11 for first, second, and third grades,
    prove that when 66 students are randomly selected using stratified sampling,
    the number of second-grade students selected is 20. -/
theorem second_grade_students_selected
  (total_students : ℕ)
  (first_grade_ratio second_grade_ratio third_grade_ratio : ℕ)
  (selected_students : ℕ)
  (h1 : total_students = 3300)
  (h2 : first_grade_ratio = 12)
  (h3 : second_grade_ratio = 10)
  (h4 : third_grade_ratio = 11)
  (h5 : selected_students = 66) :
  (second_grade_ratio : ℚ) / (first_grade_ratio + second_grade_ratio + third_grade_ratio) * selected_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_students_selected_l2957_295736


namespace NUMINAMATH_CALUDE_f_min_value_f_inequality_condition_l2957_295726

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 7/4) :=
sorry

-- Theorem for the inequality condition
theorem f_inequality_condition (a b c : ℝ) :
  (∀ (x : ℝ), f x ≥ a^2 + 2*b^2 + 3*c^2) → a*c + 2*b*c ≤ 7/8 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_f_inequality_condition_l2957_295726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2957_295794

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_first : a 0 = 23)
  (h_last : a 4 = 53) :
  a 2 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2957_295794


namespace NUMINAMATH_CALUDE_subtract_fractions_l2957_295748

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l2957_295748


namespace NUMINAMATH_CALUDE_work_completion_time_l2957_295745

/-- Given a work that can be completed by two workers A and B, this theorem proves
    the number of days B needs to complete the work alone, given certain conditions. -/
theorem work_completion_time (W : ℝ) (h_pos : W > 0) : 
  (∃ (work_A work_B : ℝ),
    -- A can finish the work in 21 days
    21 * work_A = W ∧
    -- B worked for 10 days
    10 * work_B + 
    -- A finished the remaining work in 7 days
    7 * work_A = W) →
  -- B can finish the work in 15 days
  15 * work_B = W :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l2957_295745


namespace NUMINAMATH_CALUDE_time_to_run_square_field_l2957_295768

/-- The time taken for a boy to run around a square field -/
theorem time_to_run_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 35 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 56 := by
  sorry

end NUMINAMATH_CALUDE_time_to_run_square_field_l2957_295768


namespace NUMINAMATH_CALUDE_inequality_of_squares_and_sum_l2957_295780

theorem inequality_of_squares_and_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_squares_and_sum_l2957_295780


namespace NUMINAMATH_CALUDE_total_pumpkins_l2957_295727

theorem total_pumpkins (sandy_pumpkins mike_pumpkins : ℕ) 
  (h1 : sandy_pumpkins = 51) 
  (h2 : mike_pumpkins = 23) : 
  sandy_pumpkins + mike_pumpkins = 74 := by
  sorry

end NUMINAMATH_CALUDE_total_pumpkins_l2957_295727


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2957_295715

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) (n_bread : Nat) :
  n_meat = 12 →
  n_cheese = 11 →
  n_bread = 5 →
  (n_meat.choose 2) * (n_cheese.choose 2) * (n_bread.choose 1) = 18150 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2957_295715


namespace NUMINAMATH_CALUDE_evening_milk_is_380_l2957_295788

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ
  evening : ℕ
  leftover : ℕ
  sold : ℕ
  remaining : ℕ

/-- Calculates the evening milk production given the other parameters --/
def calculate_evening_milk (mp : MilkProduction) : ℕ :=
  mp.remaining + mp.sold - mp.morning - mp.leftover

/-- Theorem stating that the evening milk production is 380 gallons --/
theorem evening_milk_is_380 (mp : MilkProduction) 
  (h1 : mp.morning = 365)
  (h2 : mp.leftover = 15)
  (h3 : mp.sold = 612)
  (h4 : mp.remaining = 148) :
  calculate_evening_milk mp = 380 := by
  sorry

#eval calculate_evening_milk { morning := 365, evening := 0, leftover := 15, sold := 612, remaining := 148 }

end NUMINAMATH_CALUDE_evening_milk_is_380_l2957_295788


namespace NUMINAMATH_CALUDE_even_decreasing_function_inequality_l2957_295705

noncomputable def e : ℝ := Real.exp 1

theorem even_decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x ≥ 0, ∀ y ≥ x, f x ≥ f y) →
  (∀ x ∈ Set.Icc 1 3, f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / e) ((2 + Real.log 3) / 3) :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_function_inequality_l2957_295705


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2957_295796

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2957_295796


namespace NUMINAMATH_CALUDE_max_kids_on_bus_l2957_295716

/-- Represents the school bus configuration -/
structure SchoolBus where
  lowerDeckRows : Nat
  upperDeckRows : Nat
  lowerDeckCapacity : Nat
  upperDeckCapacity : Nat
  staffMembers : Nat
  reservedSeats : Nat

/-- Calculates the maximum number of kids that can ride the school bus -/
def maxKids (bus : SchoolBus) : Nat :=
  (bus.lowerDeckRows * bus.lowerDeckCapacity + bus.upperDeckRows * bus.upperDeckCapacity)
  - bus.staffMembers - bus.reservedSeats

/-- The theorem stating the maximum number of kids that can ride the school bus -/
theorem max_kids_on_bus :
  let bus : SchoolBus := {
    lowerDeckRows := 15,
    upperDeckRows := 10,
    lowerDeckCapacity := 5,
    upperDeckCapacity := 3,
    staffMembers := 4,
    reservedSeats := 10
  }
  maxKids bus = 91 := by
  sorry

#eval maxKids {
  lowerDeckRows := 15,
  upperDeckRows := 10,
  lowerDeckCapacity := 5,
  upperDeckCapacity := 3,
  staffMembers := 4,
  reservedSeats := 10
}

end NUMINAMATH_CALUDE_max_kids_on_bus_l2957_295716


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l2957_295722

theorem gross_profit_percentage 
  (selling_price : ℝ) 
  (wholesale_cost : ℝ) 
  (h1 : selling_price = 28) 
  (h2 : wholesale_cost = 25) : 
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_gross_profit_percentage_l2957_295722


namespace NUMINAMATH_CALUDE_properties_of_negative_2010_l2957_295720

theorem properties_of_negative_2010 :
  let n : ℤ := -2010
  (1 / n = 1 / -2010) ∧
  (-n = 2010) ∧
  (abs n = 2010) ∧
  (-(1 / n) = 1 / 2010) := by
sorry

end NUMINAMATH_CALUDE_properties_of_negative_2010_l2957_295720


namespace NUMINAMATH_CALUDE_coin_stack_solution_l2957_295792

/-- Represents the different types of coins --/
inductive CoinType
  | A
  | B
  | C
  | D

/-- Returns the thickness of a given coin type in millimeters --/
def coinThickness (t : CoinType) : ℚ :=
  match t with
  | CoinType.A => 21/10
  | CoinType.B => 18/10
  | CoinType.C => 12/10
  | CoinType.D => 2

/-- Represents a stack of coins --/
structure CoinStack :=
  (a b c d : ℕ)

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (s : CoinStack) : ℚ :=
  s.a * coinThickness CoinType.A +
  s.b * coinThickness CoinType.B +
  s.c * coinThickness CoinType.C +
  s.d * coinThickness CoinType.D

/-- The target height of the stack in millimeters --/
def targetHeight : ℚ := 18

theorem coin_stack_solution :
  ∃ (s : CoinStack), stackHeight s = targetHeight ∧
  s.a = 0 ∧ s.b = 0 ∧ s.c = 0 ∧ s.d = 9 :=
sorry

end NUMINAMATH_CALUDE_coin_stack_solution_l2957_295792


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2957_295774

theorem arithmetic_calculations :
  (-4 - 4 = -8) ∧ ((-32) / 4 = -8) ∧ (-(-2)^3 = 8) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2957_295774


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2957_295778

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2957_295778


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2957_295772

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l2957_295772


namespace NUMINAMATH_CALUDE_triangle_properties_l2957_295733

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_equation (t : Triangle) : Prop :=
  (Real.sqrt 3 * t.c) / (t.b * Real.cos t.A) = Real.tan t.A + Real.tan t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_acute : is_acute_triangle t)
  (h_eq : satisfies_equation t) :
  t.B = Real.pi/3 ∧ 
  (t.c = 4 → 2 * Real.sqrt 3 < (1/2 * t.a * t.c * Real.sin t.B) ∧ 
                (1/2 * t.a * t.c * Real.sin t.B) < 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2957_295733


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2957_295749

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > p.1 + 1 ∧ p.2 > 3 - 2*p.1}

-- Theorem stating that all points in S are in Quadrants I or II
theorem points_in_quadrants_I_and_II : 
  ∀ p ∈ S, (p.1 > 0 ∧ p.2 > 0) ∨ (p.1 < 0 ∧ p.2 > 0) := by
  sorry

-- Helper lemma: All points in S have positive y-coordinate
lemma points_have_positive_y : 
  ∀ p ∈ S, p.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2957_295749


namespace NUMINAMATH_CALUDE_problem_statement_l2957_295704

theorem problem_statement (d : ℕ) (h : d = 4) :
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2957_295704


namespace NUMINAMATH_CALUDE_range_of_a_l2957_295701

def f (x a : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 1|

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 2*a^2 - 13) → a ∈ Set.Icc (-Real.sqrt 7) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2957_295701


namespace NUMINAMATH_CALUDE_largest_triangular_square_under_50_l2957_295707

def isTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem largest_triangular_square_under_50 :
  ∃ n : ℕ, n ≤ 50 ∧ isTriangular n ∧ isPerfectSquare n ∧
  ∀ m : ℕ, m ≤ 50 → isTriangular m → isPerfectSquare m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_triangular_square_under_50_l2957_295707


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2957_295753

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2957_295753


namespace NUMINAMATH_CALUDE_probability_of_decagon_side_l2957_295725

/-- A regular decagon -/
def RegularDecagon : Type := Unit

/-- A triangle formed by three vertices of a regular decagon -/
def DecagonTriangle : Type := Fin 3 → Fin 10

/-- Predicate to check if a DecagonTriangle has at least one side that is also a side of the decagon -/
def HasDecagonSide (t : DecagonTriangle) : Prop := sorry

/-- The set of all possible DecagonTriangles -/
def AllDecagonTriangles : Finset DecagonTriangle := sorry

/-- The set of DecagonTriangles that have at least one side that is also a side of the decagon -/
def TrianglesWithDecagonSide : Finset DecagonTriangle := sorry

/-- The probability of selecting a DecagonTriangle that has at least one side that is also a side of the decagon -/
def ProbabilityOfDecagonSide : ℚ := Finset.card TrianglesWithDecagonSide / Finset.card AllDecagonTriangles

theorem probability_of_decagon_side :
  ProbabilityOfDecagonSide = 7 / 12 :=
sorry

end NUMINAMATH_CALUDE_probability_of_decagon_side_l2957_295725


namespace NUMINAMATH_CALUDE_range_of_m_l2957_295751

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) 
  (h_ineq : ∀ m : ℝ, x + y > m^2 + 8*m) : 
  -9 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2957_295751


namespace NUMINAMATH_CALUDE_square_construction_l2957_295782

noncomputable section

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the line
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the square
structure Square (P Q V U : ℝ × ℝ) : Prop where
  side_equal : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (Q.1 - V.1)^2 + (Q.2 - V.2)^2
             ∧ (Q.1 - V.1)^2 + (Q.2 - V.2)^2 = (V.1 - U.1)^2 + (V.2 - U.2)^2
             ∧ (V.1 - U.1)^2 + (V.2 - U.2)^2 = (U.1 - P.1)^2 + (U.2 - P.2)^2
  right_angles : (P.1 - Q.1) * (Q.1 - V.1) + (P.2 - Q.2) * (Q.2 - V.2) = 0
                ∧ (Q.1 - V.1) * (V.1 - U.1) + (Q.2 - V.2) * (V.2 - U.2) = 0
                ∧ (V.1 - U.1) * (U.1 - P.1) + (V.2 - U.2) * (U.2 - P.2) = 0
                ∧ (U.1 - P.1) * (P.1 - Q.1) + (U.2 - P.2) * (P.2 - Q.2) = 0

theorem square_construction (O : ℝ × ℝ) (r : ℝ) (a b c : ℝ) 
  (h : ∀ p ∈ Line a b c, p ∉ Circle O r) :
  ∃ P Q V U : ℝ × ℝ, Square P Q V U ∧ 
    P ∈ Line a b c ∧ Q ∈ Line a b c ∧
    V ∈ Circle O r ∧ U ∈ Circle O r :=
sorry

end NUMINAMATH_CALUDE_square_construction_l2957_295782


namespace NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l2957_295740

theorem bicycle_wheel_revolutions 
  (front_radius : ℝ) 
  (back_radius : ℝ) 
  (front_revolutions : ℝ) 
  (h1 : front_radius = 3) 
  (h2 : back_radius = 6 / 12) 
  (h3 : front_revolutions = 150) :
  (2 * Real.pi * front_radius * front_revolutions) / (2 * Real.pi * back_radius) = 900 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheel_revolutions_l2957_295740


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2957_295769

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2957_295769


namespace NUMINAMATH_CALUDE_square_perimeter_l2957_295719

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 675 → perimeter = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2957_295719


namespace NUMINAMATH_CALUDE_fraction_multiplication_identity_l2957_295717

theorem fraction_multiplication_identity : (5 : ℚ) / 7 * 7 / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_identity_l2957_295717


namespace NUMINAMATH_CALUDE_martha_cards_l2957_295723

/-- The number of cards Martha ends up with after receiving more cards -/
def final_cards (start : ℕ) (received : ℕ) : ℕ :=
  start + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : final_cards 3 76 = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2957_295723


namespace NUMINAMATH_CALUDE_probability_three_players_complete_theorem_l2957_295713

/-- The probability that after N matches in a 5-player round-robin tournament,
    there are at least 3 players who have played all their matches against each other. -/
def probability_three_players_complete (N : ℕ) : ℚ :=
  if N < 3 then 0
  else if N = 3 then 1/12
  else if N = 4 then 1/3
  else if N = 5 then 5/7
  else if N = 6 then 20/21
  else 1

/-- The theorem stating the probability of having at least 3 players who have played
    all their matches against each other after N matches in a 5-player round-robin tournament. -/
theorem probability_three_players_complete_theorem (N : ℕ) (h : 3 ≤ N ∧ N ≤ 10) :
  probability_three_players_complete N =
    if N < 3 then 0
    else if N = 3 then 1/12
    else if N = 4 then 1/3
    else if N = 5 then 5/7
    else if N = 6 then 20/21
    else 1 := by sorry

end NUMINAMATH_CALUDE_probability_three_players_complete_theorem_l2957_295713


namespace NUMINAMATH_CALUDE_function_equality_l2957_295770

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2957_295770


namespace NUMINAMATH_CALUDE_second_white_given_first_red_l2957_295784

structure Bag where
  white : ℕ
  red : ℕ

def bagA : Bag := ⟨3, 2⟩
def bagB : Bag := ⟨2, 4⟩

def probChooseBag : ℚ := 1/2

def probRedFirst (b : Bag) : ℚ := b.red / (b.white + b.red)

def probWhiteSecondGivenRedFirst (b : Bag) : ℚ := b.white / (b.white + b.red - 1)

def probRedFirstThenWhite (b : Bag) : ℚ := probRedFirst b * probWhiteSecondGivenRedFirst b

theorem second_white_given_first_red :
  (probChooseBag * probRedFirstThenWhite bagA + probChooseBag * probRedFirstThenWhite bagB) /
  (probChooseBag * probRedFirst bagA + probChooseBag * probRedFirst bagB) = 17/32 := by
  sorry

end NUMINAMATH_CALUDE_second_white_given_first_red_l2957_295784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2957_295744

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a d)
  (h3 : a 1 = 4 * d)
  (h4 : a k ^ 2 = a 1 * a (2 * k)) :
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2957_295744


namespace NUMINAMATH_CALUDE_widget_purchase_theorem_l2957_295703

/-- Given a person can buy exactly 6 widgets at price p, and 8 widgets at price (p - 1.15),
    prove that the total amount of money they have is 27.60 -/
theorem widget_purchase_theorem (p : ℝ) (h1 : 6 * p = 8 * (p - 1.15)) : 6 * p = 27.60 := by
  sorry

end NUMINAMATH_CALUDE_widget_purchase_theorem_l2957_295703


namespace NUMINAMATH_CALUDE_seating_arrangements_l2957_295771

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def totalArrangements (n : ℕ) : ℕ := factorial n

def adjacentArrangements (n : ℕ) : ℕ := factorial (n - 1) * factorial 2

theorem seating_arrangements (n : ℕ) (h : n = 8) :
  totalArrangements n - adjacentArrangements n = 30240 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2957_295771


namespace NUMINAMATH_CALUDE_vertex_coordinates_l2957_295737

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := -(x - 1)^2 - 2

-- State the theorem
theorem vertex_coordinates :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ ∀ (t : ℝ), parabola t ≤ parabola x :=
sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l2957_295737


namespace NUMINAMATH_CALUDE_odometer_sum_of_squares_l2957_295710

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10

def circular_shift (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def original_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem odometer_sum_of_squares (a b c : ℕ) :
  is_valid_number a b c →
  (circular_shift a b c - original_number a b c) % 60 = 0 →
  a^2 + b^2 + c^2 = 54 := by
sorry

end NUMINAMATH_CALUDE_odometer_sum_of_squares_l2957_295710


namespace NUMINAMATH_CALUDE_integral_reciprocal_x_xplus1_l2957_295776

theorem integral_reciprocal_x_xplus1 : 
  ∫ x in (1 : ℝ)..2, 1 / (x * (x + 1)) = Real.log (4 / 3) := by sorry

end NUMINAMATH_CALUDE_integral_reciprocal_x_xplus1_l2957_295776


namespace NUMINAMATH_CALUDE_cubic_sum_l2957_295763

theorem cubic_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 7)
  (sum_prod_eq : a * b + a * c + b * c = 11)
  (prod_eq : a * b * c = -6) :
  a^3 + b^3 + c^3 = 223 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l2957_295763


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l2957_295752

theorem quadratic_roots_max_value (s p r₁ : ℝ) (h1 : r₁ ≠ 0) : 
  (r₁ + (-r₁) = 0) → 
  (r₁ * (-r₁) = p) → 
  (∀ (n : ℕ), n ≤ 2005 → r₁^(2*n) + (-r₁)^(2*n) = 2 * r₁^(2*n)) →
  (∃ (x : ℝ), x^2 - s*x + p = 0) →
  (∀ (y : ℝ), (1 / r₁^2006) + (1 / (-r₁)^2006) ≤ y) →
  y = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l2957_295752


namespace NUMINAMATH_CALUDE_sum_f_negative_l2957_295747

noncomputable def f (x : ℝ) : ℝ := x + x^3 + x^5

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) 
  (h₂ : x₂ + x₃ < 0) 
  (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l2957_295747
