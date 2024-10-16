import Mathlib

namespace NUMINAMATH_CALUDE_equations_represent_problem_l2702_270299

/-- Represents the money held by person A -/
def money_A : ℝ := sorry

/-- Represents the money held by person B -/
def money_B : ℝ := sorry

/-- The system of equations representing the problem -/
def problem_equations (x y : ℝ) : Prop :=
  (x + (1/2) * y = 50) ∧ (y + (2/3) * x = 50)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem equations_represent_problem :
  problem_equations money_A money_B ↔
  ((money_A + (1/2) * money_B = 50) ∧
   (money_B + (2/3) * money_A = 50)) :=
sorry

end NUMINAMATH_CALUDE_equations_represent_problem_l2702_270299


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l2702_270286

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 53 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l2702_270286


namespace NUMINAMATH_CALUDE_problem_statement_l2702_270233

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1 / a^2 + 1 / b^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2702_270233


namespace NUMINAMATH_CALUDE_ginger_water_usage_l2702_270297

/-- Calculates the total water used by Ginger for drinking and watering plants -/
def total_water_used (work_hours : ℕ) (bottle_capacity : ℚ) 
  (first_hour_drink : ℚ) (second_hour_drink : ℚ) (third_hour_drink : ℚ) 
  (hourly_increase : ℚ) (plant_type1_water : ℚ) (plant_type2_water : ℚ) 
  (plant_type3_water : ℚ) (plant_type1_count : ℕ) (plant_type2_count : ℕ) 
  (plant_type3_count : ℕ) : ℚ :=
  sorry

theorem ginger_water_usage :
  total_water_used 8 2 1 (3/2) 2 (1/2) 3 4 5 2 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ginger_water_usage_l2702_270297


namespace NUMINAMATH_CALUDE_kate_balloons_kate_balloons_proof_l2702_270282

theorem kate_balloons (initial_blue : ℕ) (inflated_red inflated_blue : ℕ) 
  (probability_red : ℚ) (initial_red : ℕ) : Prop :=
  initial_blue = 4 →
  inflated_red = 2 →
  inflated_blue = 2 →
  probability_red = 2/5 →
  (initial_red + inflated_red : ℚ) / 
    ((initial_red + inflated_red + initial_blue + inflated_blue) : ℚ) = probability_red →
  initial_red = 2

theorem kate_balloons_proof : 
  ∃ (initial_blue inflated_red inflated_blue : ℕ) (probability_red : ℚ) (initial_red : ℕ),
    kate_balloons initial_blue inflated_red inflated_blue probability_red initial_red :=
sorry

end NUMINAMATH_CALUDE_kate_balloons_kate_balloons_proof_l2702_270282


namespace NUMINAMATH_CALUDE_expected_blue_correct_without_replacement_more_reliable_l2702_270237

-- Define the population
def total_population : ℕ := 200
def blue_population : ℕ := 120
def pink_population : ℕ := 80

-- Define the sample sizes
def sample_size_small : ℕ := 2
def sample_size_large : ℕ := 10

-- Define the true proportion of blue items
def true_proportion : ℚ := blue_population / total_population

-- Part 1: Expected number of blue items in small sample
def expected_blue_small_sample : ℚ := 6/5

-- Part 2: Probabilities for large sample
def prob_within_error_with_replacement : ℚ := 66647/100000
def prob_within_error_without_replacement : ℚ := 67908/100000

-- Theorem statements
theorem expected_blue_correct : 
  ∀ (sampling_method : String), 
  (sampling_method = "with_replacement" ∨ sampling_method = "without_replacement") → 
  expected_blue_small_sample = sample_size_small * true_proportion :=
sorry

theorem without_replacement_more_reliable :
  prob_within_error_without_replacement > prob_within_error_with_replacement :=
sorry

end NUMINAMATH_CALUDE_expected_blue_correct_without_replacement_more_reliable_l2702_270237


namespace NUMINAMATH_CALUDE_adam_change_l2702_270260

-- Define the given amounts
def adam_money : ℚ := 5.00
def airplane_cost : ℚ := 4.28

-- Define the change function
def change (money cost : ℚ) : ℚ := money - cost

-- Theorem statement
theorem adam_change :
  change adam_money airplane_cost = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_adam_change_l2702_270260


namespace NUMINAMATH_CALUDE_polynomial_coefficient_difference_l2702_270257

theorem polynomial_coefficient_difference (a b : ℝ) : 
  (∀ x, (1 + x) + (1 + x)^4 = 2 + 5*x + a*x^2 + b*x^3 + x^4) → 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_difference_l2702_270257


namespace NUMINAMATH_CALUDE_cubic_identity_l2702_270229

theorem cubic_identity (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l2702_270229


namespace NUMINAMATH_CALUDE_total_balls_bought_l2702_270280

/-- Represents the amount of money Mr. Li has -/
def total_money : ℚ := 1

/-- The cost of one plastic ball -/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of one glass ball -/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of one wooden ball -/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li buys -/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li buys -/
def glass_balls_bought : ℕ := 10

/-- Theorem stating the total number of balls Mr. Li buys -/
theorem total_balls_bought : 
  ∃ (wooden_balls : ℕ), 
    (plastic_balls_bought * plastic_ball_cost + 
     glass_balls_bought * glass_ball_cost + 
     wooden_balls * wooden_ball_cost = total_money) ∧
    (plastic_balls_bought + glass_balls_bought + wooden_balls = 45) :=
by sorry

end NUMINAMATH_CALUDE_total_balls_bought_l2702_270280


namespace NUMINAMATH_CALUDE_chloe_second_level_treasures_l2702_270223

/-- Represents the game scenario for Chloe's treasure hunt --/
structure GameScenario where
  points_per_treasure : ℕ
  treasures_first_level : ℕ
  total_score : ℕ

/-- Calculates the number of treasures found on the second level --/
def treasures_second_level (game : GameScenario) : ℕ :=
  (game.total_score - game.points_per_treasure * game.treasures_first_level) / game.points_per_treasure

/-- Theorem stating that Chloe found 3 treasures on the second level --/
theorem chloe_second_level_treasures :
  ∃ (game : GameScenario),
    game.points_per_treasure = 9 ∧
    game.treasures_first_level = 6 ∧
    game.total_score = 81 ∧
    treasures_second_level game = 3 := by
  sorry

end NUMINAMATH_CALUDE_chloe_second_level_treasures_l2702_270223


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l2702_270288

/-- The price of apples per kg -/
def apple_price : ℕ := 70

/-- The amount of mangoes Tom bought in kg -/
def mango_amount : ℕ := 9

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The total amount Tom paid -/
def total_paid : ℕ := 1055

/-- Theorem stating that Tom purchased 8 kg of apples -/
theorem tom_apple_purchase :
  ∃ (x : ℕ), x * apple_price + mango_amount * mango_price = total_paid ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l2702_270288


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2702_270225

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2702_270225


namespace NUMINAMATH_CALUDE_division_simplification_l2702_270215

theorem division_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  6 * a^2 * b / (2 * a * b) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2702_270215


namespace NUMINAMATH_CALUDE_tan_beta_plus_pi_third_l2702_270296

theorem tan_beta_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - π/3) = 1/3) : 
  Real.tan (β + π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_plus_pi_third_l2702_270296


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2702_270291

theorem sum_geq_sqrt_three (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l2702_270291


namespace NUMINAMATH_CALUDE_base5_arithmetic_l2702_270252

/-- Converts a base 5 number to base 10 --/
def base5_to_base10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

/-- Converts a base 10 number to base 5 --/
noncomputable def base10_to_base5 (n : ℕ) : ℕ × ℕ × ℕ :=
  let d₂ := n / 25
  let r₂ := n % 25
  let d₁ := r₂ / 5
  let d₀ := r₂ % 5
  (d₂, d₁, d₀)

/-- Theorem stating that 142₅ + 324₅ - 213₅ = 303₅ --/
theorem base5_arithmetic : 
  let x := base5_to_base10 1 4 2
  let y := base5_to_base10 3 2 4
  let z := base5_to_base10 2 1 3
  base10_to_base5 (x + y - z) = (3, 0, 3) := by sorry

end NUMINAMATH_CALUDE_base5_arithmetic_l2702_270252


namespace NUMINAMATH_CALUDE_infinite_greater_than_index_l2702_270231

theorem infinite_greater_than_index :
  ∀ (a : ℕ → ℕ), (∀ n, a n ≠ 1) →
  ¬ (∃ N, ∀ n > N, a n ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_infinite_greater_than_index_l2702_270231


namespace NUMINAMATH_CALUDE_min_roots_count_l2702_270214

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ∧ (∀ x : ℝ, f (7 - x) = f (7 + x))

/-- The theorem stating the minimum number of roots -/
theorem min_roots_count
  (f : ℝ → ℝ)
  (h_symmetric : SymmetricFunction f)
  (h_root_zero : f 0 = 0) :
  (∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    (∀ roots' : Finset ℝ, (∀ x ∈ roots', f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) → 
      roots'.card ≤ roots.card) ∧
    roots.card = 401) :=
  sorry

end NUMINAMATH_CALUDE_min_roots_count_l2702_270214


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2702_270263

/-- Given a quadratic equation 3x^2 = 5x - 1, prove that its standard form coefficients are a = 3 and b = -5 --/
theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x ↦ 3 * x^2 = 5 * x - 1
  let standard_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x ↦ a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ standard_form a b c x) ∧ a = 3 ∧ b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2702_270263


namespace NUMINAMATH_CALUDE_percentage_problem_l2702_270272

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 99) → x = 4400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2702_270272


namespace NUMINAMATH_CALUDE_tape_winding_turns_l2702_270261

/-- Represents the parameters of the tape winding problem -/
structure TapeWindingParams where
  tape_length : ℝ  -- in mm
  tape_thickness : ℝ  -- in mm
  reel_diameter : ℝ  -- in mm

/-- Calculates the minimum number of turns needed to wind a tape onto a reel -/
def min_turns (params : TapeWindingParams) : ℕ :=
  sorry

/-- Theorem stating that for the given parameters, the minimum number of turns is 791 -/
theorem tape_winding_turns :
  let params : TapeWindingParams := {
    tape_length := 90000,  -- 90 m converted to mm
    tape_thickness := 0.018,
    reel_diameter := 22
  }
  min_turns params = 791 := by
  sorry

end NUMINAMATH_CALUDE_tape_winding_turns_l2702_270261


namespace NUMINAMATH_CALUDE_length_AD_is_zero_l2702_270243

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ca := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  ab = 9 ∧ bc = 40 ∧ ca = 41

-- Define right angle at C
def RightAngleC (A B C : ℝ × ℝ) : Prop :=
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0

-- Define the circumscribed circle ω
def CircumscribedCircle (ω : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  ∀ P : ℝ × ℝ, P ∈ ω ↔ (P.1 - A.1)^2 + (P.2 - A.2)^2 = 
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 = 
                      (P.1 - C.1)^2 + (P.2 - C.2)^2

-- Define point D
def PointD (D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) (A C : ℝ × ℝ) : Prop :=
  D ∈ ω ∧ 
  (D.1 - (A.1 + C.1)/2) * (C.2 - A.2) = (D.2 - (A.2 + C.2)/2) * (C.1 - A.1) ∧
  (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) < 0

theorem length_AD_is_zero 
  (A B C D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  Triangle A B C → 
  RightAngleC A B C → 
  CircumscribedCircle ω A B C → 
  PointD D ω A C → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 0 := by
    sorry

end NUMINAMATH_CALUDE_length_AD_is_zero_l2702_270243


namespace NUMINAMATH_CALUDE_ellipse_properties_l2702_270293

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The sum of distances from a point to the foci of the ellipse -/
def Ellipse.foci_distance_sum (e : Ellipse) (x y : ℝ) : ℝ :=
  2 * e.a

theorem ellipse_properties (e : Ellipse) 
    (h_point : e.equation 0 (Real.sqrt 3))
    (h_sum : e.foci_distance_sum 0 (Real.sqrt 3) = 4) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ 
  (∀ x y, e.equation x y ↔ x^2/4 + y^2/3 = 1) ∧
  e.b * 2 = 2 * Real.sqrt 3 ∧
  2 * Real.sqrt (e.a^2 - e.b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2702_270293


namespace NUMINAMATH_CALUDE_quadratic_properties_l2702_270221

-- Define the quadratic function
def quadratic (x b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the points
def point_A : ℝ × ℝ := (2, 0)
def point_B (n y1 : ℝ) : ℝ × ℝ := (3*n - 4, y1)
def point_C (n y2 : ℝ) : ℝ × ℝ := (5*n + 6, y2)

theorem quadratic_properties (b c n y1 y2 : ℝ) 
  (h1 : quadratic 2 b c = 0)  -- A(2,0) is on the curve
  (h2 : quadratic (3*n - 4) b c = y1)  -- B is on the curve
  (h3 : quadratic (5*n + 6) b c = y2)  -- C is on the curve
  (h4 : ∀ x, quadratic x b c ≥ quadratic 2 b c)  -- A is the vertex
  (h5 : n < -5) :  -- Given condition
  -- 1) The function can be expressed as y = x^2 - 4x + 4
  (∀ x, quadratic x b c = x^2 - 4*x + 4) ∧
  -- 2) If y1 = y2, then b+c < -38
  (y1 = y2 → b + c < -38) ∧
  -- 3) If c > 0, then y1 < y2
  (c > 0 → y1 < y2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2702_270221


namespace NUMINAMATH_CALUDE_abs_less_sufficient_not_necessary_for_decreasing_l2702_270283

def is_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) ≤ a n

theorem abs_less_sufficient_not_necessary_for_decreasing :
  (∃ a : ℕ → ℝ, (∀ n, |a (n + 1)| < a n) → is_decreasing a) ∧
  (∃ a : ℕ → ℝ, is_decreasing a ∧ ¬(∀ n, |a (n + 1)| < a n)) :=
sorry

end NUMINAMATH_CALUDE_abs_less_sufficient_not_necessary_for_decreasing_l2702_270283


namespace NUMINAMATH_CALUDE_function_bounds_l2702_270224

/-- Given a function f(θ) = 1 - a cos θ - b sin θ - A cos 2θ - B sin 2θ that is non-negative for all real θ,
    prove that a² + b² ≤ 2 and A² + B² ≤ 1 -/
theorem function_bounds (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l2702_270224


namespace NUMINAMATH_CALUDE_compare_roots_l2702_270295

theorem compare_roots : 
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧ 
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧ 
  (16 : ℝ) ^ (1/16) > (27 : ℝ) ^ (1/27) := by
  sorry

#check compare_roots

end NUMINAMATH_CALUDE_compare_roots_l2702_270295


namespace NUMINAMATH_CALUDE_marley_has_31_fruits_l2702_270244

-- Define the number of fruits for Louis and Samantha
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define Marley's fruits in terms of Louis and Samantha
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define the total number of Marley's fruits
def marley_total_fruits : ℕ := marley_oranges + marley_apples

-- Theorem statement
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end NUMINAMATH_CALUDE_marley_has_31_fruits_l2702_270244


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2702_270210

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) * 2

/-- Theorem: In a chess tournament with 16 players, where each player plays twice with every other player, the total number of games is 480 -/
theorem chess_tournament_games :
  tournament_games 16 = 480 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2702_270210


namespace NUMINAMATH_CALUDE_cube_distance_theorem_l2702_270276

/-- Represents a cube with a specific configuration above a plane -/
structure CubeAbovePlane where
  side_length : ℝ
  adjacent_heights : Fin 3 → ℝ
  distance_numerator : ℕ
  distance_denominator : ℕ

/-- The specific cube configuration given in the problem -/
def problem_cube : CubeAbovePlane :=
  { side_length := 12
  , adjacent_heights := ![13, 14, 16]
  , distance_numerator := 9
  , distance_denominator := 1 }

/-- Theorem stating the properties of the cube's distance from the plane -/
theorem cube_distance_theorem (cube : CubeAbovePlane) 
  (h_side : cube.side_length = 12)
  (h_heights : cube.adjacent_heights = ![13, 14, 16])
  (h_distance : ∃ (p q u : ℕ), p + q + u < 1200 ∧ 
    (cube.distance_numerator : ℝ) / cube.distance_denominator = p - Real.sqrt q) :
  cube.distance_numerator = 9 ∧ cube.distance_denominator = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_distance_theorem_l2702_270276


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l2702_270232

/-- The line equation 4x + 3y = 12 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

/-- A point (x, y) is on the line if it satisfies the line equation -/
def point_on_line (x y : ℝ) : Prop := line_equation x y

/-- The point (x, y) is in the first quadrant -/
def in_first_quadrant (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

/-- The point (x, y) is equidistant from coordinate axes -/
def equidistant_from_axes (x y : ℝ) : Prop := x = y

/-- The theorem stating that (12/7, 12/7) is the unique point satisfying all conditions -/
theorem unique_equidistant_point :
  ∃! (x y : ℝ), point_on_line x y ∧ in_first_quadrant x y ∧ equidistant_from_axes x y ∧ x = 12/7 ∧ y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l2702_270232


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2702_270287

/-- A complex number ω such that ω^2 + ω + 1 = 0 -/
def ω : ℂ := sorry

/-- The polynomial x^11 + Ax^2 + B -/
def P (A B : ℝ) (x : ℂ) : ℂ := x^11 + A * x^2 + B

/-- The polynomial x^2 + x + 1 -/
def Q (x : ℂ) : ℂ := x^2 + x + 1

theorem polynomial_divisibility (A B : ℝ) :
  (∀ x, Q x = 0 → P A B x = 0) → A = -1 ∧ B = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2702_270287


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2702_270265

theorem absolute_value_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : |a| > -b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2702_270265


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2702_270290

/-- An ellipse is represented by the equation x²/(25 - m) + y²/(m + 9) = 1 with foci on the y-axis -/
def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b > a ∧ 
  (∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

/-- The range of m for the given ellipse -/
theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_with_y_foci m ↔ 8 < m ∧ m < 25 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2702_270290


namespace NUMINAMATH_CALUDE_house_transaction_result_l2702_270250

theorem house_transaction_result (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  gain_percent = 0.20 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = -240 := by
sorry

end NUMINAMATH_CALUDE_house_transaction_result_l2702_270250


namespace NUMINAMATH_CALUDE_sum_is_composite_l2702_270259

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ m + n = a * b := by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2702_270259


namespace NUMINAMATH_CALUDE_line_slope_l2702_270228

theorem line_slope (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2702_270228


namespace NUMINAMATH_CALUDE_work_days_of_a_l2702_270209

/-- Represents the number of days worked by each person -/
structure WorkDays where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the daily wages of each person -/
structure DailyWages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total earnings given work days and daily wages -/
def totalEarnings (days : WorkDays) (wages : DailyWages) : ℕ :=
  days.a * wages.a + days.b * wages.b + days.c * wages.c

/-- The main theorem to prove -/
theorem work_days_of_a (days : WorkDays) (wages : DailyWages) : 
  days.b = 9 ∧ 
  days.c = 4 ∧ 
  wages.a * 4 = wages.b * 3 ∧ 
  wages.b * 5 = wages.c * 4 ∧ 
  wages.c = 125 ∧ 
  totalEarnings days wages = 1850 → 
  days.a = 6 := by
  sorry


end NUMINAMATH_CALUDE_work_days_of_a_l2702_270209


namespace NUMINAMATH_CALUDE_union_equality_implies_range_l2702_270239

def A : Set ℝ := {x | |x| > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem union_equality_implies_range (a : ℝ) : A ∪ B a = A → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_range_l2702_270239


namespace NUMINAMATH_CALUDE_x_value_l2702_270251

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2702_270251


namespace NUMINAMATH_CALUDE_smallest_divisor_perfect_cube_l2702_270275

theorem smallest_divisor_perfect_cube : ∃! n : ℕ, 
  n > 0 ∧ 
  n ∣ 34560 ∧ 
  (∃ m : ℕ, 34560 / n = m^3) ∧
  (∀ k : ℕ, k > 0 → k ∣ 34560 → (∃ l : ℕ, 34560 / k = l^3) → k ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_perfect_cube_l2702_270275


namespace NUMINAMATH_CALUDE_exponential_function_property_l2702_270217

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∀ x y : ℝ, f a (x + y) = f a x * f a y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l2702_270217


namespace NUMINAMATH_CALUDE_prove_n_equals_two_l2702_270254

def a (n k : ℕ) : ℕ := (n * k + 1) ^ k

theorem prove_n_equals_two (n : ℕ) : a n (a n (a n 0)) = 343 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_prove_n_equals_two_l2702_270254


namespace NUMINAMATH_CALUDE_intersection_implies_a_in_range_l2702_270273

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + a}

theorem intersection_implies_a_in_range (a : ℝ) :
  (∃! p, p ∈ set_A a ∩ set_B a) → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_a_in_range_l2702_270273


namespace NUMINAMATH_CALUDE_original_number_l2702_270211

theorem original_number : ∃ x : ℝ, 10 * x = x + 81 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_original_number_l2702_270211


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2702_270284

theorem doctors_lawyers_ratio (d l : ℕ) (h_total : d + l > 0) :
  (38 * d + 55 * l) / (d + l) = 45 →
  d / l = 10 / 7 := by
sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l2702_270284


namespace NUMINAMATH_CALUDE_arrangements_count_l2702_270281

/-- The number of people in the row -/
def total_people : ℕ := 7

/-- The number of people with adjacency restrictions -/
def restricted_people : ℕ := 3

/-- The number of unrestricted people -/
def unrestricted_people : ℕ := total_people - restricted_people

/-- The number of gaps available for placing restricted people -/
def available_gaps : ℕ := unrestricted_people + 1

/-- Calculate the number of arrangements given the conditions -/
def arrangements : ℕ := (Nat.factorial unrestricted_people) * (Nat.factorial available_gaps / Nat.factorial (available_gaps - restricted_people))

theorem arrangements_count : arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l2702_270281


namespace NUMINAMATH_CALUDE_repeating_decimal_multiplication_l2702_270218

/-- Given a real number x where x = 0.000272727... (27 repeats indefinitely),
    prove that (10^5 - 10^3) * x = 27 -/
theorem repeating_decimal_multiplication (x : ℝ) : 
  (∃ (n : ℕ), x * 10^(n+5) - x * 10^5 = 27 * (10^n - 1) / 99) → 
  (10^5 - 10^3) * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_multiplication_l2702_270218


namespace NUMINAMATH_CALUDE_set_relations_l2702_270201

def A : Set ℝ := {x | x > 1}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_relations (a : ℝ) :
  (B a ⊆ A ↔ a ∈ Set.Ici 1) ∧
  (Set.Nonempty (A ∩ B a) ↔ a ∈ Set.Ioi 0) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_l2702_270201


namespace NUMINAMATH_CALUDE_m_range_l2702_270230

/-- A circle in a 2D Cartesian coordinate system --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point A on circle C --/
def A : ℝ × ℝ := (3, 2)

/-- Circle C --/
def C : Circle := { center := (3, 4), radius := 2 }

/-- First fold line equation --/
def foldLine1 (x y : ℝ) : Prop := x - y + 1 = 0

/-- Second fold line equation --/
def foldLine2 (x y : ℝ) : Prop := x + y - 7 = 0

/-- Point M on x-axis --/
def M (m : ℝ) : ℝ × ℝ := (-m, 0)

/-- Point N on x-axis --/
def N (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Theorem stating the range of m --/
theorem m_range : 
  ∀ m : ℝ, 
  (∃ P : ℝ × ℝ, 
    (P.1 - C.center.1)^2 + (P.2 - C.center.2)^2 = C.radius^2 ∧ 
    (P.1 - (M m).1)^2 + (P.2 - (M m).2)^2 = (P.1 - (N m).1)^2 + (P.2 - (N m).2)^2
  ) ↔ 3 ≤ m ∧ m ≤ 7 := by sorry

end NUMINAMATH_CALUDE_m_range_l2702_270230


namespace NUMINAMATH_CALUDE_equilateral_pyramid_volume_l2702_270249

/-- A pyramid with an equilateral triangle base -/
structure EquilateralPyramid where
  -- The side length of the base triangle
  base_side : ℝ
  -- The angle between two edges from the apex to the base
  apex_angle : ℝ

/-- The volume of an equilateral pyramid -/
noncomputable def volume (p : EquilateralPyramid) : ℝ :=
  (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2))

/-- Theorem: The volume of a specific equilateral pyramid -/
theorem equilateral_pyramid_volume :
    ∀ (p : EquilateralPyramid),
      p.base_side = 2 →
      volume p = (Real.sqrt 3 / 9) * (2 / 3 * Real.sqrt 3 + 1 / Real.tan (p.apex_angle / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_pyramid_volume_l2702_270249


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2702_270258

theorem triangle_angle_proof (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Side-angle relationship
  c * Real.sin A = a * Real.cos C →
  -- Conclusion
  C = π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2702_270258


namespace NUMINAMATH_CALUDE_mean_of_playground_counts_l2702_270294

def playground_counts : List ℕ := [6, 12, 1, 12, 7, 3, 8]

theorem mean_of_playground_counts :
  (playground_counts.sum : ℚ) / playground_counts.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_playground_counts_l2702_270294


namespace NUMINAMATH_CALUDE_two_integer_solutions_l2702_270227

theorem two_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, |3*x - 4| + |3*x + 2| = 6) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_integer_solutions_l2702_270227


namespace NUMINAMATH_CALUDE_sqrt_calculation_and_exponent_simplification_l2702_270285

theorem sqrt_calculation_and_exponent_simplification :
  (∃ x : ℝ, x^2 = 18) ∧ (∃ y : ℝ, y^2 = 32) ∧ (∃ z : ℝ, z^2 = 2) →
  (∃ a : ℝ, a^2 = 3) →
  (∀ x y z : ℝ, x^2 = 18 ∧ y^2 = 32 ∧ z^2 = 2 → x - y + z = 0) ∧
  (∀ a : ℝ, a^2 = 3 → (a + 2)^2022 * (a - 2)^2021 * (a - 3) = 3 + a) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculation_and_exponent_simplification_l2702_270285


namespace NUMINAMATH_CALUDE_product_and_sum_inequality_l2702_270204

theorem product_and_sum_inequality (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  x * y ≥ 64 ∧ x + y ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_inequality_l2702_270204


namespace NUMINAMATH_CALUDE_painted_stripe_area_l2702_270216

/-- The area of a painted stripe on a cylindrical tank -/
theorem painted_stripe_area (d h w1 w2 r1 r2 : ℝ) (hd : d = 40) (hh : h = 100) 
  (hw1 : w1 = 5) (hw2 : w2 = 7) (hr1 : r1 = 3) (hr2 : r2 = 3) : 
  w1 * (π * d * r1) + w2 * (π * d * r2) = 1440 * π := by
  sorry

end NUMINAMATH_CALUDE_painted_stripe_area_l2702_270216


namespace NUMINAMATH_CALUDE_smallest_student_count_l2702_270245

theorem smallest_student_count (sophomore freshman junior : ℕ) : 
  sophomore * 4 = freshman * 7 →
  junior * 7 = sophomore * 6 →
  sophomore > 0 →
  freshman > 0 →
  junior > 0 →
  ∀ (s f j : ℕ), 
    s * 4 = f * 7 →
    j * 7 = s * 6 →
    s > 0 → f > 0 → j > 0 →
    s + f + j ≥ sophomore + freshman + junior :=
by
  sorry

#eval 7 + 4 + 6 -- Expected output: 17

end NUMINAMATH_CALUDE_smallest_student_count_l2702_270245


namespace NUMINAMATH_CALUDE_exponent_division_l2702_270220

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^5 = 1 / x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2702_270220


namespace NUMINAMATH_CALUDE_continuous_diff_function_properties_l2702_270235

/-- A function with a continuous derivative on ℝ -/
structure ContinuousDiffFunction where
  f : ℝ → ℝ
  f_continuous : Continuous f
  f_deriv : ℝ → ℝ
  f_deriv_continuous : Continuous f_deriv
  f_has_deriv : ∀ x, HasDerivAt f (f_deriv x) x

/-- The theorem statement -/
theorem continuous_diff_function_properties
  (f : ContinuousDiffFunction) (a b : ℝ) (hab : a < b)
  (h_deriv_a : f.f_deriv a > 0) (h_deriv_b : f.f_deriv b < 0) :
  (∃ x ∈ Set.Icc a b, f.f x > f.f b) ∧
  (∃ x ∈ Set.Icc a b, f.f a - f.f b > f.f_deriv x * (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_continuous_diff_function_properties_l2702_270235


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l2702_270206

/-- The curve xy = 2 -/
def curve (x y : ℝ) : Prop := x * y = 2

/-- A circle in the coordinate plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The four intersection points of the curve and the circle -/
def intersection_points (c : Circle) : Prop :=
  ∃ (p1 p2 p3 p4 : ℝ × ℝ),
    curve p1.1 p1.2 ∧ on_circle p1 c ∧
    curve p2.1 p2.2 ∧ on_circle p2 c ∧
    curve p3.1 p3.2 ∧ on_circle p3 c ∧
    curve p4.1 p4.2 ∧ on_circle p4 c ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

theorem fourth_intersection_point (c : Circle) :
  intersection_points c →
  curve 4 (1/2) →
  curve (-2) (-1) →
  curve (1/4) 8 →
  ∃ (p : ℝ × ℝ), p = (-2, -1) ∧ curve p.1 p.2 ∧ on_circle p c :=
by sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l2702_270206


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l2702_270205

theorem constant_ratio_problem (x₀ y₀ x y : ℝ) (h : x₀ = 3 ∧ y₀ = 4) :
  (∃ k : ℝ, (5 * x₀ - 3) / (y₀ + 10) = k ∧ (5 * x - 3) / (y + 10) = k) →
  (y = 19 → x = 39 / 7) := by
sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l2702_270205


namespace NUMINAMATH_CALUDE_tiller_swath_width_l2702_270236

/-- Calculates the swath width of a tiller given plot dimensions, tilling rate, and total tilling time -/
theorem tiller_swath_width
  (plot_width : ℝ)
  (plot_length : ℝ)
  (tilling_rate : ℝ)
  (total_time : ℝ)
  (h1 : plot_width = 110)
  (h2 : plot_length = 120)
  (h3 : tilling_rate = 2)  -- 2 seconds per foot
  (h4 : total_time = 220 * 60)  -- 220 minutes in seconds
  : (plot_width * plot_length) / (total_time / tilling_rate) = 2 := by
  sorry

#check tiller_swath_width

end NUMINAMATH_CALUDE_tiller_swath_width_l2702_270236


namespace NUMINAMATH_CALUDE_revenue_difference_l2702_270222

/-- The revenue generated by a single jersey -/
def jersey_revenue : ℕ := 210

/-- The revenue generated by a single t-shirt -/
def tshirt_revenue : ℕ := 240

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 177

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 23

/-- The difference in revenue between t-shirts and jerseys -/
theorem revenue_difference : 
  tshirt_revenue * tshirts_sold - jersey_revenue * jerseys_sold = 37650 := by
  sorry

end NUMINAMATH_CALUDE_revenue_difference_l2702_270222


namespace NUMINAMATH_CALUDE_range_of_expression_l2702_270279

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) :
  -π/6 < 2*α - β/2 ∧ 2*α - β/2 < π :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l2702_270279


namespace NUMINAMATH_CALUDE_initial_salary_calculation_l2702_270268

/-- Represents the yearly salary increase rate -/
def salary_increase_rate : ℝ := 1.4

/-- Calculates the salary after n years given the initial salary -/
def salary_after_n_years (initial_salary : ℝ) (n : ℕ) : ℝ :=
  initial_salary * salary_increase_rate ^ n

/-- Theorem: If the salary after 3 years is 8232, then the initial salary is 3000 -/
theorem initial_salary_calculation (initial_salary : ℝ) :
  salary_after_n_years initial_salary 3 = 8232 → initial_salary = 3000 := by
  sorry

end NUMINAMATH_CALUDE_initial_salary_calculation_l2702_270268


namespace NUMINAMATH_CALUDE_parabola_directrix_l2702_270278

/-- The directrix of the parabola y = -3x^2 + 6x - 5 is y = -23/12 -/
theorem parabola_directrix : ∀ x y : ℝ, 
  y = -3 * x^2 + 6 * x - 5 → 
  ∃ (k : ℝ), k = -23/12 ∧ (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 2)^2 / 12) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2702_270278


namespace NUMINAMATH_CALUDE_probability_of_same_color_l2702_270234

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (total_sides : ℕ)
  (side_sum : red + green + blue + yellow = total_sides)

/-- Calculates the probability of two identical dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.yellow^2) / d.total_sides^2

/-- The specific 12-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 3
    green := 4
    blue := 2
    yellow := 3
    total_sides := 12
    side_sum := by rfl }

theorem probability_of_same_color :
  same_color_probability problem_die = 19 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_same_color_l2702_270234


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l2702_270289

theorem greatest_multiple_of_four (y : ℕ) : 
  y > 0 ∧ 
  ∃ k : ℕ, y = 4 * k ∧ 
  y^3 < 4096 →
  y ≤ 12 ∧ 
  ∀ z : ℕ, z > 0 ∧ (∃ m : ℕ, z = 4 * m) ∧ z^3 < 4096 → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l2702_270289


namespace NUMINAMATH_CALUDE_different_chord_length_l2702_270202

-- Define the ellipse
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / m + y^2 / 4 = 1

-- Define the chord length for a line y = ax + b on the ellipse
noncomputable def chordLength (m a b : ℝ) : ℝ :=
  let A := 4 + m * a^2
  let B := 2 * m * a
  let C := m * (b^2 - 1)
  Real.sqrt ((B^2 - 4*A*C) / A^2)

-- Theorem statement
theorem different_chord_length (k m : ℝ) (hm : m > 0) :
  chordLength m k 1 ≠ chordLength m (-k) 2 :=
sorry

end NUMINAMATH_CALUDE_different_chord_length_l2702_270202


namespace NUMINAMATH_CALUDE_division_simplification_l2702_270240

theorem division_simplification (a b c d e f : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) / (e * f) = (a * d) / (b * c * e * f) := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l2702_270240


namespace NUMINAMATH_CALUDE_product_of_roots_l2702_270248

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 18 → ∃ y : ℝ, (x + 3) * (x - 5) = 18 ∧ (y + 3) * (y - 5) = 18 ∧ x * y = -33 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2702_270248


namespace NUMINAMATH_CALUDE_puppy_count_l2702_270264

theorem puppy_count (total_ears : ℕ) (ears_per_puppy : ℕ) (h1 : total_ears = 210) (h2 : ears_per_puppy = 2) :
  total_ears / ears_per_puppy = 105 :=
by sorry

end NUMINAMATH_CALUDE_puppy_count_l2702_270264


namespace NUMINAMATH_CALUDE_salary_problem_l2702_270298

/-- Proves that given the conditions of the problem, A's salary is Rs. 3000 --/
theorem salary_problem (total : ℝ) (a_salary : ℝ) (b_salary : ℝ) 
  (h1 : total = 4000)
  (h2 : a_salary + b_salary = total)
  (h3 : 0.05 * a_salary = 0.15 * b_salary) :
  a_salary = 3000 := by
  sorry

#check salary_problem

end NUMINAMATH_CALUDE_salary_problem_l2702_270298


namespace NUMINAMATH_CALUDE_average_age_calculation_l2702_270256

theorem average_age_calculation (fifth_graders : ℕ) (fifth_graders_avg : ℝ)
  (parents : ℕ) (parents_avg : ℝ) (teachers : ℕ) (teachers_avg : ℝ) :
  fifth_graders = 40 ∧ fifth_graders_avg = 10 ∧
  parents = 60 ∧ parents_avg = 35 ∧
  teachers = 10 ∧ teachers_avg = 45 →
  let total_age := fifth_graders * fifth_graders_avg + parents * parents_avg + teachers * teachers_avg
  let total_people := fifth_graders + parents + teachers
  abs ((total_age / total_people) - 26.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_age_calculation_l2702_270256


namespace NUMINAMATH_CALUDE_existence_of_valid_numbers_l2702_270269

def is_valid_move (a b c : ℕ) : Prop :=
  a ≤ 40 ∧ b ≤ 40 ∧ c ≤ 40

def can_transform (a b c : ℕ) (a' b' c' : ℕ) : Prop :=
  (a' = a + a * b / 100 ∧ b' = b ∧ c' = c) ∨
  (a' = a + a * c / 100 ∧ b' = b ∧ c' = c) ∨
  (a' = a ∧ b' = b + b * a / 100 ∧ c' = c) ∨
  (a' = a ∧ b' = b + b * c / 100 ∧ c' = c) ∨
  (a' = a ∧ b' = b ∧ c' = c + c * a / 100) ∨
  (a' = a ∧ b' = b ∧ c' = c + c * b / 100)

def can_reach_target (a b c : ℕ) : Prop :=
  ∃ (n : ℕ) (f : ℕ → ℕ × ℕ × ℕ),
    f 0 = (a, b, c) ∧
    (∀ i : ℕ, i < n → can_transform (f i).1 (f i).2.1 (f i).2.2 (f (i+1)).1 (f (i+1)).2.1 (f (i+1)).2.2) ∧
    ((f n).1 > 2011 ∨ (f n).2.1 > 2011 ∨ (f n).2.2 > 2011)

theorem existence_of_valid_numbers :
  ∃ a b c : ℕ, is_valid_move a b c ∧ can_reach_target a b c :=
sorry

end NUMINAMATH_CALUDE_existence_of_valid_numbers_l2702_270269


namespace NUMINAMATH_CALUDE_problem_solution_l2702_270271

theorem problem_solution :
  (∃ n : ℕ, 20 = 4 * n) ∧
  (∃ m : ℕ, 180 = 9 * m) ∧
  (∃ k : ℕ, 209 = 19 * k) ∧
  (∃ l : ℕ, 57 = 19 * l) ∧
  (∃ p : ℕ, 90 = 30 * p) ∧
  (∃ q : ℕ, 34 = 17 * q) ∧
  (∃ r : ℕ, 51 = 17 * r) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2702_270271


namespace NUMINAMATH_CALUDE_johns_water_usage_l2702_270219

/-- Calculates the total water usage for John's showers over 4 weeks -/
def total_water_usage (weeks : ℕ) (shower_frequency : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let num_showers := days / shower_frequency
  let water_per_shower := shower_duration * water_per_minute
  num_showers * water_per_shower

/-- Proves that John's total water usage over 4 weeks is 280 gallons -/
theorem johns_water_usage : total_water_usage 4 2 10 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_johns_water_usage_l2702_270219


namespace NUMINAMATH_CALUDE_k_value_l2702_270270

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 2*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 4)) →
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_k_value_l2702_270270


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_terms_l2702_270212

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric sequence -/
def a : ℚ := 1/5

/-- The common ratio of the geometric sequence -/
def r : ℚ := 1/3

/-- The number of terms to sum -/
def n : ℕ := 7

theorem geometric_sum_first_seven_terms :
  geometric_sum a r n = 2186/3645 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_seven_terms_l2702_270212


namespace NUMINAMATH_CALUDE_probability_odd_limit_l2702_270255

/-- Represents the probability of getting an odd number after n button presses -/
def probability_odd (n : ℕ) : ℝ := sorry

/-- The recurrence relation for the probability of getting an odd number -/
axiom probability_recurrence (n : ℕ) : 
  probability_odd (n + 1) = probability_odd n - (1/2) * (probability_odd n)^2

/-- The initial probability (after one button press) is not exactly 1/3 -/
axiom initial_probability : probability_odd 1 ≠ 1/3

/-- Theorem: The probability of getting an odd number converges to 1/3 -/
theorem probability_odd_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |probability_odd n - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_odd_limit_l2702_270255


namespace NUMINAMATH_CALUDE_skylar_age_l2702_270277

/-- Represents the age when Skylar started donating -/
def starting_age : ℕ := 17

/-- Represents the annual donation amount in thousands -/
def annual_donation : ℕ := 8

/-- Represents the total amount donated in thousands -/
def total_donated : ℕ := 440

/-- Calculates Skylar's current age -/
def current_age : ℕ := starting_age + (total_donated / annual_donation)

/-- Proves that Skylar's current age is 72 years -/
theorem skylar_age : current_age = 72 := by
  sorry

end NUMINAMATH_CALUDE_skylar_age_l2702_270277


namespace NUMINAMATH_CALUDE_smallest_regular_polygon_sides_l2702_270274

theorem smallest_regular_polygon_sides (n : ℕ) : n > 0 → (∃ k : ℕ, k > 0 ∧ 360 * k / (2 * n) = 28) → n ≥ 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_regular_polygon_sides_l2702_270274


namespace NUMINAMATH_CALUDE_actual_speed_proof_l2702_270242

theorem actual_speed_proof (time_reduction : Real) (speed_increase : Real) 
  (h1 : time_reduction = Real.pi / 4)
  (h2 : speed_increase = Real.sqrt 15) : 
  ∃ (actual_speed : Real), actual_speed = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_proof_l2702_270242


namespace NUMINAMATH_CALUDE_amy_ticket_cost_l2702_270253

/-- The total cost of tickets purchased by Amy at the fair -/
theorem amy_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) (price_per_ticket : ℚ) :
  initial_tickets = 33 →
  additional_tickets = 21 →
  price_per_ticket = 3/2 →
  (initial_tickets + additional_tickets : ℚ) * price_per_ticket = 81 := by
sorry

end NUMINAMATH_CALUDE_amy_ticket_cost_l2702_270253


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2702_270208

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) →
  1/a + 9/b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2702_270208


namespace NUMINAMATH_CALUDE_fraction_simplification_l2702_270226

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2702_270226


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l2702_270207

theorem sqrt_difference_equals_seven : 
  Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l2702_270207


namespace NUMINAMATH_CALUDE_league_teams_l2702_270200

theorem league_teams (n : ℕ) : n * (n - 1) / 2 = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_league_teams_l2702_270200


namespace NUMINAMATH_CALUDE_m_equals_five_l2702_270266

theorem m_equals_five (m : ℝ) (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_m_equals_five_l2702_270266


namespace NUMINAMATH_CALUDE_mars_visibility_time_l2702_270203

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  if totalMinutes2 ≥ totalMinutes1 then
    totalMinutes2 - totalMinutes1
  else
    (24 * 60) - (totalMinutes1 - totalMinutes2)

/-- Subtracts a given number of minutes from a time -/
def subtractMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes
  let newTotalMinutes := (totalMinutes + 24 * 60 - m) % (24 * 60)
  { hours := newTotalMinutes / 60,
    minutes := newTotalMinutes % 60,
    h_valid := by sorry }

theorem mars_visibility_time 
  (jupiter_after_mars : ℕ) 
  (uranus_after_jupiter : ℕ)
  (uranus_appearance : Time)
  (h1 : jupiter_after_mars = 2 * 60 + 41)
  (h2 : uranus_after_jupiter = 3 * 60 + 16)
  (h3 : uranus_appearance = { hours := 6, minutes := 7, h_valid := by sorry }) :
  let jupiter_time := subtractMinutes uranus_appearance uranus_after_jupiter
  let mars_time := subtractMinutes jupiter_time jupiter_after_mars
  mars_time = { hours := 0, minutes := 10, h_valid := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_mars_visibility_time_l2702_270203


namespace NUMINAMATH_CALUDE_employee_pay_l2702_270267

/-- Given two employees x and y are paid a total of 330 rupees per week,
    and x is paid 120% of y, prove that y is paid 150 rupees per week. -/
theorem employee_pay (x y : ℝ) 
  (total_pay : x + y = 330)
  (x_pay_ratio : x = 1.2 * y) : 
  y = 150 := by sorry

end NUMINAMATH_CALUDE_employee_pay_l2702_270267


namespace NUMINAMATH_CALUDE_sqrt_x_squared_nonnegative_l2702_270241

theorem sqrt_x_squared_nonnegative (x : ℝ) : 0 ≤ Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_nonnegative_l2702_270241


namespace NUMINAMATH_CALUDE_negative_number_identification_l2702_270247

theorem negative_number_identification :
  let numbers : List ℚ := [1, 0, 1/2, -2]
  ∀ x ∈ numbers, x < 0 ↔ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l2702_270247


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2702_270238

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_twelve_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l2702_270238


namespace NUMINAMATH_CALUDE_a_range_l2702_270262

/-- The piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

/-- f(x) is a decreasing function -/
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The theorem statement -/
theorem a_range (a : ℝ) :
  is_decreasing (f a) → a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2702_270262


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2702_270292

/-- The limiting sum of the geometric series 4 - 8/3 + 16/9 - ... equals 2.4 -/
theorem geometric_series_sum : 
  let a : ℝ := 4
  let r : ℝ := -2/3
  let s : ℝ := a / (1 - r)
  s = 2.4 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2702_270292


namespace NUMINAMATH_CALUDE_jasmine_percentage_l2702_270213

/-- Calculates the percentage of jasmine in a solution after adding jasmine and water -/
theorem jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let total_jasmine := initial_jasmine + added_jasmine
  let total_volume := initial_volume + added_jasmine + added_water
  let final_percentage := (total_jasmine / total_volume) * 100
  final_percentage = 12.5 := by
sorry


end NUMINAMATH_CALUDE_jasmine_percentage_l2702_270213


namespace NUMINAMATH_CALUDE_initial_condition_recursive_relation_diamonds_in_tenth_figure_l2702_270246

/-- The number of diamonds in the n-th figure of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

/-- The sequence starts with 4 diamonds for n = 1 -/
theorem initial_condition : num_diamonds 1 = 4 := by sorry

/-- The recursive relation for n ≥ 2 -/
theorem recursive_relation (n : ℕ) (h : n ≥ 2) :
  num_diamonds n = num_diamonds (n-1) + 12 * (n-1) := by sorry

/-- The main theorem: The number of diamonds in the 10th figure is 112 -/
theorem diamonds_in_tenth_figure : num_diamonds 10 = 112 := by sorry

end NUMINAMATH_CALUDE_initial_condition_recursive_relation_diamonds_in_tenth_figure_l2702_270246
