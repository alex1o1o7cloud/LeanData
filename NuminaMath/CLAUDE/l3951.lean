import Mathlib

namespace NUMINAMATH_CALUDE_system_solvability_l3951_395171

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x = 6 / a - |y - a| ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ := {a | a ≤ -2/3 ∨ a > 0}

-- Theorem statement
theorem system_solvability (a : ℝ) :
  (∃ b x y, system a b x y) ↔ a ∈ valid_a_set :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l3951_395171


namespace NUMINAMATH_CALUDE_sequence_next_term_l3951_395157

theorem sequence_next_term (a₁ a₂ a₃ a₄ a₅ x : ℕ) : 
  a₁ = 2 ∧ a₂ = 5 ∧ a₃ = 11 ∧ a₄ = 20 ∧ a₅ = 32 ∧
  (a₂ - a₁) = 3 ∧ (a₃ - a₂) = 6 ∧ (a₄ - a₃) = 9 ∧ (a₅ - a₄) = 12 ∧
  (x - a₅) = (a₅ - a₄) + 3 →
  x = 47 := by
sorry

end NUMINAMATH_CALUDE_sequence_next_term_l3951_395157


namespace NUMINAMATH_CALUDE_largest_decimal_l3951_395197

theorem largest_decimal : ∀ (a b c d e : ℚ),
  a = 97/100 ∧ b = 979/1000 ∧ c = 9709/10000 ∧ d = 907/1000 ∧ e = 9089/10000 →
  b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e :=
by sorry

end NUMINAMATH_CALUDE_largest_decimal_l3951_395197


namespace NUMINAMATH_CALUDE_annika_return_time_l3951_395188

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  rate : ℝ  -- Hiking rate in minutes per kilometer
  initialDistance : ℝ  -- Initial distance hiked east in kilometers
  totalDistance : ℝ  -- Total distance to hike east in kilometers

/-- Calculates the time needed to return to the start of the trail -/
def timeToReturn (scenario : HikingScenario) : ℝ :=
  let remainingDistance := scenario.totalDistance - scenario.initialDistance
  let timeToCompleteEast := remainingDistance * scenario.rate
  let timeToReturnWest := scenario.totalDistance * scenario.rate
  timeToCompleteEast + timeToReturnWest

/-- Theorem stating that Annika needs 35 minutes to return to the start -/
theorem annika_return_time :
  let scenario : HikingScenario := {
    rate := 10,
    initialDistance := 2.5,
    totalDistance := 3
  }
  timeToReturn scenario = 35 := by sorry

end NUMINAMATH_CALUDE_annika_return_time_l3951_395188


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l3951_395116

/-- Proves that if 90% of a farmer's land is cleared, and 20% of the cleared land
    is planted with tomatoes covering 360 acres, then the total land owned by the
    farmer is 2000 acres. -/
theorem farmer_land_calculation (total_land : ℝ) (cleared_land : ℝ) (tomato_land : ℝ) :
  cleared_land = 0.9 * total_land →
  tomato_land = 0.2 * cleared_land →
  tomato_land = 360 →
  total_land = 2000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l3951_395116


namespace NUMINAMATH_CALUDE_evaluate_g_l3951_395164

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l3951_395164


namespace NUMINAMATH_CALUDE_linen_tablecloth_cost_is_25_l3951_395191

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecorations where
  num_tables : ℕ
  place_settings_per_table : ℕ
  place_setting_cost : ℕ
  roses_per_centerpiece : ℕ
  lilies_per_centerpiece : ℕ
  rose_cost : ℕ
  lily_cost : ℕ
  total_decoration_cost : ℕ

/-- Calculates the cost of a single linen tablecloth --/
def linen_tablecloth_cost (d : WeddingDecorations) : ℕ :=
  let place_settings_cost := d.num_tables * d.place_settings_per_table * d.place_setting_cost
  let centerpiece_cost := d.num_tables * (d.roses_per_centerpiece * d.rose_cost + d.lilies_per_centerpiece * d.lily_cost)
  let tablecloth_total_cost := d.total_decoration_cost - (place_settings_cost + centerpiece_cost)
  tablecloth_total_cost / d.num_tables

/-- Theorem stating that the cost of a single linen tablecloth is $25 --/
theorem linen_tablecloth_cost_is_25 (d : WeddingDecorations)
  (h1 : d.num_tables = 20)
  (h2 : d.place_settings_per_table = 4)
  (h3 : d.place_setting_cost = 10)
  (h4 : d.roses_per_centerpiece = 10)
  (h5 : d.lilies_per_centerpiece = 15)
  (h6 : d.rose_cost = 5)
  (h7 : d.lily_cost = 4)
  (h8 : d.total_decoration_cost = 3500) :
  linen_tablecloth_cost d = 25 := by
  sorry

end NUMINAMATH_CALUDE_linen_tablecloth_cost_is_25_l3951_395191


namespace NUMINAMATH_CALUDE_function_two_zeros_implies_a_range_l3951_395130

/-- If the function y = x + a/x + 1 has two zeros, then a ∈ (-∞, 1/4) -/
theorem function_two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + a / x₁ + 1 = 0 ∧ x₂ + a / x₂ + 1 = 0) →
  a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_function_two_zeros_implies_a_range_l3951_395130


namespace NUMINAMATH_CALUDE_speaker_arrangement_count_l3951_395178

-- Define the number of speakers
def n : ℕ := 6

-- Theorem statement
theorem speaker_arrangement_count :
  (n.factorial / 2 : ℕ) = (n.factorial / 2 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_speaker_arrangement_count_l3951_395178


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3951_395148

-- Define a right triangle with side lengths a, b, and c
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : c^2 = a^2 + b^2

-- Define the theorem
theorem hypotenuse_length (t : RightTriangle) 
  (h : Real.sqrt ((t.a - 3)^2 + (t.b - 2)^2) = 0) :
  t.c = 3 ∨ t.c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3951_395148


namespace NUMINAMATH_CALUDE_article_count_l3951_395159

theorem article_count (cost_price selling_price : ℝ) (gain_percentage : ℝ) : 
  gain_percentage = 42.857142857142854 →
  50 * cost_price = 35 * selling_price →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  35 = 50 * (100 / (100 + gain_percentage)) :=
by sorry

end NUMINAMATH_CALUDE_article_count_l3951_395159


namespace NUMINAMATH_CALUDE_squad_selection_ways_l3951_395181

/-- The number of ways to choose a squad of 8 players (including one dedicated setter) from a team of 12 members -/
def choose_squad (team_size : ℕ) (squad_size : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (squad_size - 1))

/-- Theorem stating that choosing a squad of 8 players (including one dedicated setter) from a team of 12 members can be done in 3960 ways -/
theorem squad_selection_ways :
  choose_squad 12 8 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_squad_selection_ways_l3951_395181


namespace NUMINAMATH_CALUDE_power_product_equals_sixteen_l3951_395152

theorem power_product_equals_sixteen (m n : ℤ) (h : 2*m + 3*n - 4 = 0) : 
  (4:ℝ)^m * (8:ℝ)^n = 16 := by
sorry

end NUMINAMATH_CALUDE_power_product_equals_sixteen_l3951_395152


namespace NUMINAMATH_CALUDE_panthers_score_l3951_395112

theorem panthers_score (total_score cougars_margin : ℕ) 
  (h1 : total_score = 48)
  (h2 : cougars_margin = 20) :
  ∃ (panthers_score cougars_score : ℕ),
    panthers_score + cougars_score = total_score ∧
    cougars_score = panthers_score + cougars_margin ∧
    panthers_score = 14 :=
by sorry

end NUMINAMATH_CALUDE_panthers_score_l3951_395112


namespace NUMINAMATH_CALUDE_faulty_key_is_seven_or_nine_l3951_395132

/-- Represents a digit key on a keypad -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents whether a key press was registered or not -/
inductive KeyPress
| registered
| notRegistered

/-- Represents a sequence of ten attempted key presses -/
def AttemptedSequence := Vector Digit 10

/-- Represents the actual registered sequence after pressing keys -/
def RegisteredSequence := Vector Digit 7

/-- Checks if a digit appears at least five times in a sequence -/
def appearsAtLeastFiveTimes (d : Digit) (s : AttemptedSequence) : Prop := sorry

/-- Checks if the registration pattern of a digit matches the faulty key pattern -/
def matchesFaultyPattern (d : Digit) (s : AttemptedSequence) (r : RegisteredSequence) : Prop := sorry

/-- The main theorem stating that the faulty key must be either 7 or 9 -/
theorem faulty_key_is_seven_or_nine
  (attempted : AttemptedSequence)
  (registered : RegisteredSequence)
  (h1 : ∃ (d : Digit), appearsAtLeastFiveTimes d attempted)
  (h2 : ∀ (d : Digit), appearsAtLeastFiveTimes d attempted → matchesFaultyPattern d attempted registered) :
  ∃ (d : Digit), d = Digit.seven ∨ d = Digit.nine :=
sorry

end NUMINAMATH_CALUDE_faulty_key_is_seven_or_nine_l3951_395132


namespace NUMINAMATH_CALUDE_exists_self_power_congruence_l3951_395128

theorem exists_self_power_congruence : ∃ N : ℕ, 
  (10^2000 ≤ N) ∧ (N < 10^2001) ∧ (N ≡ N^2001 [ZMOD 10^2001]) := by
  sorry

end NUMINAMATH_CALUDE_exists_self_power_congruence_l3951_395128


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l3951_395147

/-- Represents the taxi fare structure -/
structure TaxiFare where
  startingFee : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.startingFee + tf.ratePerMile * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.startingFee = 20)
  (h2 : totalFare tf 80 = 160)
  : totalFare tf 120 = 230 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l3951_395147


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3951_395124

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3951_395124


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3951_395100

theorem complex_equation_solution (x y : ℝ) : 
  (2*x - 1 : ℂ) + (y + 1 : ℂ) * I = (x - y : ℂ) - (x + y : ℂ) * I → x = 3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3951_395100


namespace NUMINAMATH_CALUDE_area_original_figure_l3951_395120

/-- Given an isosceles trapezoid representing the isometric drawing of a horizontally placed figure,
    with a bottom angle of 60°, legs and top base of length 1,
    the area of the original plane figure is 3√6/2. -/
theorem area_original_figure (bottom_angle : ℝ) (leg_length : ℝ) (top_base : ℝ) : 
  bottom_angle = π / 3 →
  leg_length = 1 →
  top_base = 1 →
  ∃ (area : ℝ), area = (3 * Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_original_figure_l3951_395120


namespace NUMINAMATH_CALUDE_cubic_sequence_problem_l3951_395140

theorem cubic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 8*y₂ + 27*y₃ + 64*y₄ + 125*y₅ = 7)
  (eq2 : 8*y₁ + 27*y₂ + 64*y₃ + 125*y₄ + 216*y₅ = 100)
  (eq3 : 27*y₁ + 64*y₂ + 125*y₃ + 216*y₄ + 343*y₅ = 1000) :
  64*y₁ + 125*y₂ + 216*y₃ + 343*y₄ + 512*y₅ = -5999 := by
sorry

end NUMINAMATH_CALUDE_cubic_sequence_problem_l3951_395140


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3951_395144

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3951_395144


namespace NUMINAMATH_CALUDE_jason_after_school_rate_l3951_395195

/-- Calculates Jason's hourly rate for after-school work --/
def after_school_rate (total_earnings weekly_hours saturday_hours saturday_rate : ℚ) : ℚ :=
  let saturday_earnings := saturday_hours * saturday_rate
  let after_school_earnings := total_earnings - saturday_earnings
  let after_school_hours := weekly_hours - saturday_hours
  after_school_earnings / after_school_hours

/-- Theorem stating Jason's after-school hourly rate --/
theorem jason_after_school_rate :
  after_school_rate 88 18 8 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_after_school_rate_l3951_395195


namespace NUMINAMATH_CALUDE_fraction_simplification_l3951_395111

theorem fraction_simplification : (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3951_395111


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l3951_395129

theorem sin_x_squared_not_periodic : ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l3951_395129


namespace NUMINAMATH_CALUDE_base6_addition_subtraction_l3951_395158

-- Define a function to convert from base 6 to decimal
def base6ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

-- Define a function to convert from decimal to base 6
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition_subtraction :
  let a := [2, 4, 5, 3, 1]  -- 13542₆ in reverse order
  let b := [5, 3, 4, 3, 2]  -- 23435₆ in reverse order
  let c := [2, 1, 3, 4]     -- 4312₆ in reverse order
  let result := [5, 0, 4, 1, 3]  -- 31405₆ in reverse order
  decimalToBase6 ((base6ToDecimal a + base6ToDecimal b) - base6ToDecimal c) = result := by
  sorry


end NUMINAMATH_CALUDE_base6_addition_subtraction_l3951_395158


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3951_395131

theorem bowling_ball_weight (canoe_weight : ℝ) (bowling_ball_weight : ℝ) : 
  canoe_weight = 36 →
  6 * bowling_ball_weight = 4 * canoe_weight →
  bowling_ball_weight = 24 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3951_395131


namespace NUMINAMATH_CALUDE_rectangle_length_equality_l3951_395185

/-- Given a figure composed of rectangles with right angles, prove that the unknown length Y is 1 cm --/
theorem rectangle_length_equality (Y : ℝ) : Y = 1 := by
  -- Define the sum of top segment lengths
  let top_sum := 3 + 2 + 3 + 4 + Y
  -- Define the sum of bottom segment lengths
  let bottom_sum := 7 + 4 + 2
  -- Assert that the sums are equal (property of rectangles)
  have sum_equality : top_sum = bottom_sum := by sorry
  -- Solve for Y
  sorry


end NUMINAMATH_CALUDE_rectangle_length_equality_l3951_395185


namespace NUMINAMATH_CALUDE_worker_count_proof_l3951_395141

theorem worker_count_proof : ∃ (x y : ℕ), 
  y = (15 * x) / 19 ∧ 
  (4 * y) / 7 < 1000 ∧ 
  (3 * x) / 5 > 1000 ∧ 
  x = 1995 ∧ 
  y = 1575 := by
sorry

end NUMINAMATH_CALUDE_worker_count_proof_l3951_395141


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_exists_l3951_395125

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 → a^2 * b^3 * c ≤ x^2 * y^3 * z :=
by sorry

theorem min_value_exists (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  a^2 * b^3 * c = 1/108 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_exists_l3951_395125


namespace NUMINAMATH_CALUDE_landscape_ratio_l3951_395187

/-- Given a rectangular landscape with the following properties:
  - length is 120 meters
  - contains a playground of 1200 square meters
  - playground occupies 1/3 of the total landscape area
  Prove that the ratio of length to breadth is 4:1 -/
theorem landscape_ratio (length : ℝ) (playground_area : ℝ) (breadth : ℝ) : 
  length = 120 →
  playground_area = 1200 →
  playground_area = (1/3) * (length * breadth) →
  length / breadth = 4 := by
sorry


end NUMINAMATH_CALUDE_landscape_ratio_l3951_395187


namespace NUMINAMATH_CALUDE_work_together_proof_l3951_395184

/-- The number of days after which Alice, Bob, Carol, and Dave work together again -/
def days_until_work_together_again : ℕ := 360

/-- Alice's work schedule (every 5th day) -/
def alice_schedule : ℕ := 5

/-- Bob's work schedule (every 6th day) -/
def bob_schedule : ℕ := 6

/-- Carol's work schedule (every 8th day) -/
def carol_schedule : ℕ := 8

/-- Dave's work schedule (every 9th day) -/
def dave_schedule : ℕ := 9

theorem work_together_proof :
  days_until_work_together_again = Nat.lcm alice_schedule (Nat.lcm bob_schedule (Nat.lcm carol_schedule dave_schedule)) :=
by
  sorry

#eval days_until_work_together_again

end NUMINAMATH_CALUDE_work_together_proof_l3951_395184


namespace NUMINAMATH_CALUDE_y_axis_symmetry_of_P_l3951_395110

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The y-axis symmetry operation on a point -/
def yAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the y-axis symmetry of P(0, -2, 3) is (0, -2, -3) -/
theorem y_axis_symmetry_of_P :
  let P : Point3D := { x := 0, y := -2, z := 3 }
  yAxisSymmetry P = { x := 0, y := -2, z := -3 } := by
  sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_of_P_l3951_395110


namespace NUMINAMATH_CALUDE_tomato_plant_ratio_l3951_395177

/-- Proves that the ratio of dead tomato plants to initial tomato plants is 1/2 --/
theorem tomato_plant_ratio (total_vegetables : ℕ) (vegetables_per_plant : ℕ) 
  (initial_tomato : ℕ) (initial_eggplant : ℕ) (initial_pepper : ℕ) 
  (dead_pepper : ℕ) : 
  total_vegetables = 56 →
  vegetables_per_plant = 7 →
  initial_tomato = 6 →
  initial_eggplant = 2 →
  initial_pepper = 4 →
  dead_pepper = 1 →
  (initial_tomato - (total_vegetables / vegetables_per_plant - initial_eggplant - (initial_pepper - dead_pepper))) / initial_tomato = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plant_ratio_l3951_395177


namespace NUMINAMATH_CALUDE_pens_cost_after_discount_and_tax_l3951_395151

/-- The cost of one pen in terms of the cost of one pencil -/
def pen_cost (pencil_cost : ℚ) : ℚ := 5 * pencil_cost

/-- The total cost of pens and pencils -/
def total_cost (pencil_cost : ℚ) : ℚ := 3 * pen_cost pencil_cost + 5 * pencil_cost

/-- The cost of one dozen pens -/
def dozen_pens_cost (pencil_cost : ℚ) : ℚ := 12 * pen_cost pencil_cost

/-- The discount rate applied to one dozen pens -/
def discount_rate : ℚ := 1 / 10

/-- The tax rate applied after the discount -/
def tax_rate : ℚ := 18 / 100

/-- The final cost of one dozen pens after discount and tax -/
def final_cost (pencil_cost : ℚ) : ℚ :=
  let discounted_cost := dozen_pens_cost pencil_cost * (1 - discount_rate)
  discounted_cost * (1 + tax_rate)

theorem pens_cost_after_discount_and_tax :
  ∃ (pencil_cost : ℚ),
    total_cost pencil_cost = 260 ∧
    final_cost pencil_cost = 828.36 := by
  sorry


end NUMINAMATH_CALUDE_pens_cost_after_discount_and_tax_l3951_395151


namespace NUMINAMATH_CALUDE_opposite_and_reciprocal_sum_l3951_395183

theorem opposite_and_reciprocal_sum (a b x y : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : x * y = 1)  -- x and y are reciprocals
  : 2 * (a + b) + (7 / 4) * x * y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_reciprocal_sum_l3951_395183


namespace NUMINAMATH_CALUDE_books_given_correct_l3951_395169

/-- The number of books Melissa gives to Jordan --/
def books_given : ℝ := 10.5

/-- Initial number of books Melissa had --/
def melissa_initial : ℕ := 123

/-- Initial number of books Jordan had --/
def jordan_initial : ℕ := 27

theorem books_given_correct :
  let melissa_final := melissa_initial - books_given
  let jordan_final := jordan_initial + books_given
  (melissa_initial + jordan_initial : ℝ) = melissa_final + jordan_final ∧
  melissa_final = 3 * jordan_final := by
  sorry

end NUMINAMATH_CALUDE_books_given_correct_l3951_395169


namespace NUMINAMATH_CALUDE_smallest_n_factor_smallest_n_is_75_l3951_395134

theorem smallest_n_factor (n : ℕ+) : 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) →
  n ≥ 75 :=
by sorry

theorem smallest_n_is_75 : 
  ∃ (n : ℕ+), n = 75 ∧ 
  (5^2 ∣ n * (2^5) * (6^2) * (7^3)) ∧ 
  (3^3 ∣ n * (2^5) * (6^2) * (7^3)) ∧
  ∀ (m : ℕ+), m < 75 → 
    ¬((5^2 ∣ m * (2^5) * (6^2) * (7^3)) ∧ 
      (3^3 ∣ m * (2^5) * (6^2) * (7^3))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_factor_smallest_n_is_75_l3951_395134


namespace NUMINAMATH_CALUDE_range_of_x_plus_cos_y_l3951_395194

theorem range_of_x_plus_cos_y (x y : ℝ) (h : 2 * x + Real.cos (2 * y) = 1) :
  ∃ (z : ℝ), z = x + Real.cos y ∧ -1 ≤ z ∧ z ≤ 5/4 ∧
  (∃ (x' y' : ℝ), 2 * x' + Real.cos (2 * y') = 1 ∧ x' + Real.cos y' = -1) ∧
  (∃ (x'' y'' : ℝ), 2 * x'' + Real.cos (2 * y'') = 1 ∧ x'' + Real.cos y'' = 5/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_cos_y_l3951_395194


namespace NUMINAMATH_CALUDE_mango_rate_is_55_l3951_395156

/-- The rate per kg of mangoes given Bruce's purchase --/
def mango_rate (grape_kg : ℕ) (grape_rate : ℕ) (mango_kg : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - grape_kg * grape_rate) / mango_kg

/-- Theorem stating that the rate per kg of mangoes is 55 --/
theorem mango_rate_is_55 :
  mango_rate 8 70 10 1110 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_is_55_l3951_395156


namespace NUMINAMATH_CALUDE_intersection_value_l3951_395161

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}

theorem intersection_value (m : ℕ) : A ∩ B m = {1, m} → m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l3951_395161


namespace NUMINAMATH_CALUDE_system_solution_existence_l3951_395138

theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) ↔ 
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3951_395138


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l3951_395126

theorem max_value_trig_sum (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l3951_395126


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3951_395105

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3951_395105


namespace NUMINAMATH_CALUDE_triangle_angles_l3951_395154

/-- Triangle angles theorem -/
theorem triangle_angles (ω φ θ : ℝ) : 
  ω + φ + θ = 180 → 
  2 * ω + θ = 180 → 
  φ = 2 * θ → 
  θ = 36 ∧ φ = 72 ∧ ω = 72 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l3951_395154


namespace NUMINAMATH_CALUDE_book_cost_l3951_395199

/-- If three identical books cost $45, then seven of these books cost $105. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  7 * (cost_of_three / 3) = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l3951_395199


namespace NUMINAMATH_CALUDE_lcm_inequality_l3951_395162

theorem lcm_inequality (k m n : ℕ) : 
  (Nat.lcm k m) * (Nat.lcm m n) * (Nat.lcm n k) ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
sorry

end NUMINAMATH_CALUDE_lcm_inequality_l3951_395162


namespace NUMINAMATH_CALUDE_english_homework_time_l3951_395145

def total_time : ℕ := 180
def math_time : ℕ := 45
def science_time : ℕ := 50
def history_time : ℕ := 25
def project_time : ℕ := 30

theorem english_homework_time :
  total_time - (math_time + science_time + history_time + project_time) = 30 := by
sorry

end NUMINAMATH_CALUDE_english_homework_time_l3951_395145


namespace NUMINAMATH_CALUDE_reading_time_difference_example_l3951_395146

/-- The difference in reading time (in minutes) between two readers for a given book -/
def reading_time_difference (xavier_speed maya_speed : ℕ) (book_pages : ℕ) : ℕ :=
  ((book_pages / maya_speed - book_pages / xavier_speed) * 60)

/-- Theorem: Given Xavier's and Maya's reading speeds and the book length, 
    the difference in reading time is 180 minutes -/
theorem reading_time_difference_example : 
  reading_time_difference 120 60 360 = 180 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_example_l3951_395146


namespace NUMINAMATH_CALUDE_sequence_fourth_term_l3951_395139

theorem sequence_fourth_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^3) : a 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fourth_term_l3951_395139


namespace NUMINAMATH_CALUDE_simplify_expression_l3951_395122

theorem simplify_expression : 
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3951_395122


namespace NUMINAMATH_CALUDE_dot_product_of_intersection_vectors_l3951_395167

/-- Parabola type representing y² = 4x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 4*x

/-- Line type representing y = 2/3(x+2) -/
structure Line where
  x : ℝ
  y : ℝ
  eq : y = 2/3*(x+2)

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Intersection points of the parabola and the line -/
def intersection_points (p : Parabola) (l : Line) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- Vector from focus to a point -/
def vector_from_focus (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - focus.1, p.2 - focus.2)

/-- Dot product of two vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Main theorem -/
theorem dot_product_of_intersection_vectors (p : Parabola) (l : Line) :
  let (m, n) := intersection_points p l
  dot_product (vector_from_focus m) (vector_from_focus n) = 8 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_intersection_vectors_l3951_395167


namespace NUMINAMATH_CALUDE_problem_1_l3951_395142

theorem problem_1 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3951_395142


namespace NUMINAMATH_CALUDE_unique_pair_solution_l3951_395113

theorem unique_pair_solution : 
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧  -- Positive integers
    y ≥ x ∧          -- y ≥ x
    x + y ≤ 20 ∧     -- Sum constraint
    ¬(Nat.Prime (x * y)) ∧  -- Product is composite
    (∀ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ≥ a ∧ a + b ≤ 20 ∧ a * b = x * y → a + b = x + y) ∧  -- Unique sum given product and constraints
    x = 2 ∧ y = 11 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_solution_l3951_395113


namespace NUMINAMATH_CALUDE_water_price_this_year_l3951_395107

-- Define the price of water last year
def price_last_year : ℝ := 1.6

-- Define the price increase rate
def price_increase_rate : ℝ := 0.2

-- Define Xiao Li's water bill in December last year
def december_bill : ℝ := 17

-- Define Xiao Li's water bill in January this year
def january_bill : ℝ := 30

-- Define the difference in water consumption between January and December
def consumption_difference : ℝ := 5

-- Theorem: The price of residential water this year is 1.92 yuan per cubic meter
theorem water_price_this_year :
  let price_this_year := price_last_year * (1 + price_increase_rate)
  price_this_year = 1.92 ∧
  january_bill / price_this_year - december_bill / price_last_year = consumption_difference :=
by sorry

end NUMINAMATH_CALUDE_water_price_this_year_l3951_395107


namespace NUMINAMATH_CALUDE_black_area_proof_l3951_395106

theorem black_area_proof (white_area black_area : ℝ) : 
  white_area + black_area = 9^2 + 5^2 →
  white_area + 2 * black_area = 11^2 + 7^2 →
  black_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_black_area_proof_l3951_395106


namespace NUMINAMATH_CALUDE_egg_usage_ratio_l3951_395170

/-- Proves that the ratio of eggs used to total eggs bought is 1:2 --/
theorem egg_usage_ratio (total_dozen : ℕ) (broken : ℕ) (left : ℕ) : 
  total_dozen = 6 → broken = 15 → left = 21 → 
  (total_dozen * 12 - (left + broken)) * 2 = total_dozen * 12 := by
  sorry

end NUMINAMATH_CALUDE_egg_usage_ratio_l3951_395170


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3951_395174

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  ratio : ℝ
  ratio_pos : ratio > 0
  geom : ∀ n, a (n + 1) = a n * ratio

/-- Theorem: In a geometric sequence with positive terms, if a₁a₃ = 4 and a₂ + a₄ = 10, then the common ratio is 2 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 2 + seq.a 4 = 10) :
  seq.ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3951_395174


namespace NUMINAMATH_CALUDE_n_cube_minus_n_l3951_395143

theorem n_cube_minus_n (n : ℕ) (h : ∃ k : ℕ, 33 * 20 * n = k) : n^3 - n = 388944 := by
  sorry

end NUMINAMATH_CALUDE_n_cube_minus_n_l3951_395143


namespace NUMINAMATH_CALUDE_count_prime_pairs_sum_80_l3951_395166

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The sum of the pair is 80 -/
def sumIs80 (p q : ℕ) : Prop := p + q = 80

/-- The statement to be proved -/
theorem count_prime_pairs_sum_80 :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 4 ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs → isPrime p ∧ isPrime q ∧ sumIs80 p q) ∧
    (∀ (p q : ℕ), isPrime p → isPrime q → sumIs80 p q → (p, q) ∈ pairs ∨ (q, p) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_count_prime_pairs_sum_80_l3951_395166


namespace NUMINAMATH_CALUDE_find_number_l3951_395189

theorem find_number : ∃ x : ℝ, 8 * x = 0.4 * 900 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3951_395189


namespace NUMINAMATH_CALUDE_wario_expected_wide_right_misses_l3951_395102

/-- Represents a football kicker's field goal statistics -/
structure KickerStats where
  totalAttempts : ℕ
  missRate : ℚ
  missTypes : Fin 4 → ℚ
  underFortyYardsSuccessRate : ℚ

/-- Represents the conditions for a specific game -/
structure GameConditions where
  attempts : ℕ
  attemptsUnderForty : ℕ
  windSpeed : ℚ

/-- Calculates the expected number of wide right misses for a kicker in a game -/
def expectedWideRightMisses (stats : KickerStats) (game : GameConditions) : ℚ :=
  (game.attempts : ℚ) * stats.missRate * (stats.missTypes 3)

/-- Theorem stating that Wario's expected wide right misses in the next game is 1 -/
theorem wario_expected_wide_right_misses :
  let warioStats : KickerStats := {
    totalAttempts := 80,
    missRate := 1/3,
    missTypes := λ _ => 1/4,
    underFortyYardsSuccessRate := 7/10
  }
  let gameConditions : GameConditions := {
    attempts := 12,
    attemptsUnderForty := 9,
    windSpeed := 18
  }
  expectedWideRightMisses warioStats gameConditions = 1 := by sorry

end NUMINAMATH_CALUDE_wario_expected_wide_right_misses_l3951_395102


namespace NUMINAMATH_CALUDE_tomato_plants_count_l3951_395115

/-- Represents the number of vegetables harvested from each surviving plant. -/
def vegetables_per_plant : ℕ := 7

/-- Represents the total number of vegetables harvested. -/
def total_vegetables : ℕ := 56

/-- Represents the number of eggplant plants. -/
def eggplant_plants : ℕ := 2

/-- Represents the initial number of pepper plants. -/
def initial_pepper_plants : ℕ := 4

/-- Represents the number of pepper plants that died. -/
def dead_pepper_plants : ℕ := 1

theorem tomato_plants_count (T : ℕ) : 
  (T / 2 + eggplant_plants + (initial_pepper_plants - dead_pepper_plants)) * vegetables_per_plant = total_vegetables → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_tomato_plants_count_l3951_395115


namespace NUMINAMATH_CALUDE_paul_reading_time_l3951_395137

/-- The number of hours Paul spent reading after nine weeks -/
def reading_hours (books_per_week : ℕ) (pages_per_book : ℕ) (pages_per_hour : ℕ) (weeks : ℕ) : ℕ :=
  books_per_week * pages_per_book * weeks / pages_per_hour

/-- Theorem stating that Paul spent 540 hours reading after nine weeks -/
theorem paul_reading_time : reading_hours 10 300 50 9 = 540 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_time_l3951_395137


namespace NUMINAMATH_CALUDE_translation_theorem_l3951_395155

/-- The original function -/
def f (x : ℝ) : ℝ := x^2 + x

/-- The translated function -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The translation amount -/
def a : ℝ := 2

theorem translation_theorem (h : a > 0) : 
  ∀ x, g x = f (x - a) :=
by sorry

end NUMINAMATH_CALUDE_translation_theorem_l3951_395155


namespace NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l3951_395163

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders :
  let initial_radius : ℝ := 5
  let initial_height : ℝ := 10
  let radius_increase : ℝ := 4
  let volume1 := π * (initial_radius + radius_increase)^2 * initial_height
  let volume2 (x : ℝ) := π * initial_radius^2 * (initial_height + x)
  ∀ x : ℝ, volume1 = volume2 x ↔ x = 112 / 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_volumes_of_modified_cylinders_l3951_395163


namespace NUMINAMATH_CALUDE_instantaneous_velocity_zero_l3951_395117

/-- The motion law of an object -/
def S (t : ℝ) : ℝ := t^3 - 6*t^2 + 5

/-- The instantaneous velocity of the object -/
def V (t : ℝ) : ℝ := 3*t^2 - 12*t

theorem instantaneous_velocity_zero (t : ℝ) (h : t > 0) :
  V t = 0 → t = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_zero_l3951_395117


namespace NUMINAMATH_CALUDE_peters_candies_l3951_395149

theorem peters_candies : ∃ (initial : ℚ), 
  initial > 0 ∧ 
  (1/4 * initial - 13/2 : ℚ) = 6 ∧
  initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_peters_candies_l3951_395149


namespace NUMINAMATH_CALUDE_koi_fish_count_l3951_395119

/-- Calculates the number of koi fish after 3 weeks given the initial conditions --/
def koi_fish_after_three_weeks (initial_total : ℕ) (koi_added_per_day : ℕ) (goldfish_added_per_day : ℕ) (days : ℕ) (final_goldfish : ℕ) : ℕ :=
  let total_added := (koi_added_per_day + goldfish_added_per_day) * days
  let final_total := initial_total + total_added
  final_total - final_goldfish

/-- Theorem stating that the number of koi fish after 3 weeks is 227 --/
theorem koi_fish_count : koi_fish_after_three_weeks 280 2 5 21 200 = 227 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l3951_395119


namespace NUMINAMATH_CALUDE_chord_segment_lengths_l3951_395193

theorem chord_segment_lengths (R : ℝ) (OM : ℝ) (AB : ℝ) (h1 : R = 15) (h2 : OM = 13) (h3 : AB = 18) :
  ∃ (AM MB : ℝ), AM + MB = AB ∧ AM = 14 ∧ MB = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_segment_lengths_l3951_395193


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3951_395135

theorem largest_digit_divisible_by_six : 
  ∃ (M : ℕ), M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ 
  ∀ (N : ℕ), N ≤ 9 ∧ (45670 + N) % 6 = 0 → N ≤ M :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3951_395135


namespace NUMINAMATH_CALUDE_fish_count_theorem_l3951_395198

def is_valid_fish_count (t : ℕ) : Prop :=
  (t > 10 ∧ t > 15 ∧ t ≤ 18) ∨
  (t > 10 ∧ t ≤ 15 ∧ t > 18) ∨
  (t ≤ 10 ∧ t > 15 ∧ t > 18)

theorem fish_count_theorem :
  ∀ t : ℕ, is_valid_fish_count t ↔ (t = 16 ∨ t = 17 ∨ t = 18) :=
by sorry

end NUMINAMATH_CALUDE_fish_count_theorem_l3951_395198


namespace NUMINAMATH_CALUDE_solution_count_is_49_l3951_395182

/-- The number of positive integer pairs (x, y) satisfying xy / (x + y) = 1000 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x * y / (x + y) = 1000
  ) (Finset.product (Finset.range 2001) (Finset.range 2001))).card

theorem solution_count_is_49 : solution_count = 49 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_is_49_l3951_395182


namespace NUMINAMATH_CALUDE_vectors_in_same_plane_l3951_395172

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b c : V)

def is_basis (a b c : V) : Prop :=
  LinearIndependent ℝ ![a, b, c] ∧ Submodule.span ℝ {a, b, c} = ⊤

def coplanar (u v w : V) : Prop :=
  ∃ (x y z : ℝ), x • u + y • v + z • w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

theorem vectors_in_same_plane (h : is_basis V a b c) :
  coplanar V (2 • a + b) (a + b + c) (7 • a + 5 • b + 3 • c) :=
sorry

end NUMINAMATH_CALUDE_vectors_in_same_plane_l3951_395172


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l3951_395176

/-- Given that 8 oranges weigh the same as 6 apples, 
    prove that 48 oranges weigh the same as 36 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℕ → ℝ),
  (∀ n : ℕ, orange_weight n > 0 ∧ apple_weight n > 0) →
  (orange_weight 8 = apple_weight 6) →
  (orange_weight 48 = apple_weight 36) :=
by sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l3951_395176


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3951_395165

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3951_395165


namespace NUMINAMATH_CALUDE_number_divided_by_ratio_l3951_395114

theorem number_divided_by_ratio (x : ℝ) : 
  0.55 * x = 4.235 → x / 0.55 = 14 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_ratio_l3951_395114


namespace NUMINAMATH_CALUDE_tangent_slope_circle_tangent_slope_specific_circle_l3951_395133

theorem tangent_slope_circle (center : ℝ × ℝ) (tangent_point : ℝ × ℝ) : ℝ :=
  let center_x : ℝ := center.1
  let center_y : ℝ := center.2
  let tangent_x : ℝ := tangent_point.1
  let tangent_y : ℝ := tangent_point.2
  let radius_slope : ℝ := (tangent_y - center_y) / (tangent_x - center_x)
  let tangent_slope : ℝ := -1 / radius_slope
  tangent_slope

theorem tangent_slope_specific_circle : 
  tangent_slope_circle (2, 3) (7, 8) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_tangent_slope_specific_circle_l3951_395133


namespace NUMINAMATH_CALUDE_factory_working_days_l3951_395175

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 6500

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1300

/-- The number of working days per week -/
def working_days : ℕ := toys_per_week / toys_per_day

theorem factory_working_days :
  working_days = 5 :=
sorry

end NUMINAMATH_CALUDE_factory_working_days_l3951_395175


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l3951_395109

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola x^2 = 4y -/
def focus : Point := ⟨0, 1⟩

/-- The directrix of the parabola x^2 = 4y -/
def directrix : ℝ := -1

theorem parabola_triangle_area 
  (P : Point) 
  (h_P : P ∈ Parabola) 
  (M : Point) 
  (h_M : M.y = directrix) 
  (h_perp : (P.x - M.x) * (P.y - M.y) + (P.y - M.y) * (M.y - directrix) = 0) 
  (h_dist : (P.x - M.x)^2 + (P.y - M.y)^2 = 25) : 
  (1/2) * |P.x - M.x| * |P.y - focus.y| = 10 :=
sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l3951_395109


namespace NUMINAMATH_CALUDE_darius_drove_679_miles_l3951_395108

/-- The number of miles Julia drove -/
def julia_miles : ℕ := 998

/-- The total number of miles Darius and Julia drove -/
def total_miles : ℕ := 1677

/-- The number of miles Darius drove -/
def darius_miles : ℕ := total_miles - julia_miles

theorem darius_drove_679_miles : darius_miles = 679 := by sorry

end NUMINAMATH_CALUDE_darius_drove_679_miles_l3951_395108


namespace NUMINAMATH_CALUDE_total_tulips_count_l3951_395186

def tulips_per_eye : ℕ := 8
def number_of_eyes : ℕ := 2
def tulips_for_smile : ℕ := 18
def background_multiplier : ℕ := 9

def total_tulips : ℕ := 
  (tulips_per_eye * number_of_eyes + tulips_for_smile) + 
  (background_multiplier * tulips_for_smile)

theorem total_tulips_count : total_tulips = 196 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_count_l3951_395186


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3951_395101

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - x + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3951_395101


namespace NUMINAMATH_CALUDE_yearly_savings_multiple_l3951_395118

theorem yearly_savings_multiple (monthly_salary : ℝ) (h : monthly_salary > 0) :
  let monthly_spending := 0.75 * monthly_salary
  let monthly_savings := monthly_salary - monthly_spending
  let yearly_savings := 12 * monthly_savings
  yearly_savings = 4 * monthly_spending :=
by sorry

end NUMINAMATH_CALUDE_yearly_savings_multiple_l3951_395118


namespace NUMINAMATH_CALUDE_two_real_roots_l3951_395136

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem two_real_roots (a b c : ℝ) 
  (h : ∀ x ∈ Set.Icc (-1) 1, |f a b c x| < 1) :
  ∃ x y : ℝ, x ≠ y ∧ f a b c x = 2 * x^2 - 1 ∧ f a b c y = 2 * y^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_two_real_roots_l3951_395136


namespace NUMINAMATH_CALUDE_puzzle_completion_time_l3951_395168

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_set : ℕ) (minutes_per_set : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let num_sets := total_pieces / pieces_per_set
  num_sets * minutes_per_set

/-- Theorem stating that completing 2 puzzles of 2000 pieces each, at a rate of 100 pieces per 10 minutes, takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

#eval time_to_complete_puzzles 2 2000 100 10

end NUMINAMATH_CALUDE_puzzle_completion_time_l3951_395168


namespace NUMINAMATH_CALUDE_bowTie_solution_l3951_395180

noncomputable def bowTie (a b : ℝ) : ℝ := a^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

theorem bowTie_solution (y : ℝ) : bowTie 3 y = 18 → y = 6 * Real.sqrt 2 ∨ y = -6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_solution_l3951_395180


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3951_395179

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 35)
  (h2 : technicians = 7)
  (h3 : avg_salary_technicians = 16000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3951_395179


namespace NUMINAMATH_CALUDE_product_of_mixed_numbers_l3951_395127

theorem product_of_mixed_numbers :
  let a : Rat := 2 + 1/6
  let b : Rat := 3 + 2/9
  a * b = 377/54 := by
  sorry

end NUMINAMATH_CALUDE_product_of_mixed_numbers_l3951_395127


namespace NUMINAMATH_CALUDE_max_triangle_area_l3951_395123

/-- The parabola function y = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The area of triangle ABC given p -/
def triangle_area (p : ℝ) : ℝ := 2 * |((p - 1) * (p - 3))|

theorem max_triangle_area :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 4 ∧
  f 0 = 3 ∧ f 4 = 3 ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → triangle_area x ≤ 2) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ triangle_area x = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3951_395123


namespace NUMINAMATH_CALUDE_percentage_female_on_duty_l3951_395121

def total_on_duty : ℕ := 200
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 1000

theorem percentage_female_on_duty :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_on_duty_l3951_395121


namespace NUMINAMATH_CALUDE_choir_singing_problem_l3951_395173

theorem choir_singing_problem (total_singers : ℕ) 
  (h1 : total_singers = 30)
  (first_verse : ℕ) 
  (h2 : first_verse = total_singers / 2)
  (second_verse : ℕ)
  (h3 : second_verse = (total_singers - first_verse) / 3)
  (final_verse : ℕ)
  (h4 : final_verse = total_singers - first_verse - second_verse) :
  final_verse = 10 := by
  sorry

end NUMINAMATH_CALUDE_choir_singing_problem_l3951_395173


namespace NUMINAMATH_CALUDE_museum_visit_permutations_l3951_395150

theorem museum_visit_permutations : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_museum_visit_permutations_l3951_395150


namespace NUMINAMATH_CALUDE_four_digit_number_count_l3951_395196

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns true if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := sorry

/-- A function that returns true if all digits in a natural number are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- A function that returns the leftmost digit of a natural number -/
def leftmostDigit (n : ℕ) : ℕ := sorry

/-- A function that returns the rightmost digit of a natural number -/
def rightmostDigit (n : ℕ) : ℕ := sorry

theorem four_digit_number_count :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1000 ≤ n ∧ n < 10000) ∧ 
    (∀ n ∈ S, isPrime (leftmostDigit n)) ∧
    (∀ n ∈ S, isPerfectSquare (rightmostDigit n)) ∧
    (∀ n ∈ S, allDigitsDifferent n) ∧
    Finset.card S ≥ 288 := by sorry

end NUMINAMATH_CALUDE_four_digit_number_count_l3951_395196


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l3951_395103

theorem factorial_ratio_equals_sixty_sevenths : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l3951_395103


namespace NUMINAMATH_CALUDE_calculation_comparison_l3951_395153

theorem calculation_comparison : 
  (3.04 / 0.25 > 1) ∧ (1.01 * 0.99 < 1) ∧ (0.15 / 0.25 < 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_comparison_l3951_395153


namespace NUMINAMATH_CALUDE_twoPointThreeFive_equals_fraction_l3951_395192

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 2.35̄ -/
def twoPointThreeFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 35 }

theorem twoPointThreeFive_equals_fraction :
  toRational twoPointThreeFive = 233 / 99 := by
  sorry

end NUMINAMATH_CALUDE_twoPointThreeFive_equals_fraction_l3951_395192


namespace NUMINAMATH_CALUDE_existence_of_fractions_l3951_395104

theorem existence_of_fractions (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  ∃ (p q r s : ℕ+), 
    (a < (p : ℝ) / q ∧ (p : ℝ) / q < (r : ℝ) / s ∧ (r : ℝ) / s < b) ∧
    (p : ℝ)^2 + (q : ℝ)^2 = (r : ℝ)^2 + (s : ℝ)^2 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_fractions_l3951_395104


namespace NUMINAMATH_CALUDE_range_of_b_over_a_l3951_395160

def quadratic_equation (a b x : ℝ) : ℝ := x^2 + (a+1)*x + a + b + 1

theorem range_of_b_over_a (a b : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, quadratic_equation a b x = 0) →
  (x₁ ≠ x₂) →
  (quadratic_equation a b x₁ = 0) →
  (quadratic_equation a b x₂ = 0) →
  (0 < x₁ ∧ x₁ < 1) →
  (x₂ > 1) →
  (-2 < b/a ∧ b/a < -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_over_a_l3951_395160


namespace NUMINAMATH_CALUDE_fraction_transformation_l3951_395190

theorem fraction_transformation (x : ℝ) (h : x ≠ 2) : 2 / (2 - x) = -(2 / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3951_395190
