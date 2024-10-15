import Mathlib

namespace NUMINAMATH_CALUDE_trig_identity_l1074_107492

theorem trig_identity (θ : ℝ) (h : Real.sin (π + θ) = 1/4) :
  (Real.cos (π + θ)) / (Real.cos θ * (Real.cos (π + θ) - 1)) +
  (Real.sin (π/2 - θ)) / (Real.cos (θ + 2*π) * Real.cos (π + θ) + Real.cos (-θ)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1074_107492


namespace NUMINAMATH_CALUDE_jerry_original_butterflies_l1074_107448

/-- The number of butterflies Jerry let go -/
def butterflies_released : ℕ := 11

/-- The number of butterflies Jerry still has -/
def butterflies_remaining : ℕ := 82

/-- The original number of butterflies Jerry had -/
def original_butterflies : ℕ := butterflies_released + butterflies_remaining

theorem jerry_original_butterflies : original_butterflies = 93 := by
  sorry

end NUMINAMATH_CALUDE_jerry_original_butterflies_l1074_107448


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l1074_107469

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 12 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l1074_107469


namespace NUMINAMATH_CALUDE_prob_two_red_in_three_draws_l1074_107425

def total_balls : ℕ := 8
def red_balls : ℕ := 3
def white_balls : ℕ := 5

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem prob_two_red_in_three_draws :
  (prob_white * prob_red * prob_red) + (prob_red * prob_white * prob_red) = 45 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_in_three_draws_l1074_107425


namespace NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l1074_107421

def total_on_duty : ℕ := 240
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 300

theorem female_officers_on_duty_percentage :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_on_duty_percentage_l1074_107421


namespace NUMINAMATH_CALUDE_not_all_bisecting_diameters_perpendicular_l1074_107426

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A diameter of a circle -/
structure Diameter (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- Predicate to check if a diameter bisects a chord -/
def bisects (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Predicate to check if a diameter is perpendicular to a chord -/
def perpendicular (d : Diameter c) (ch : Chord c) : Prop :=
  sorry

/-- Theorem stating that it's not always true that a diameter bisecting a chord is perpendicular to it -/
theorem not_all_bisecting_diameters_perpendicular (c : Circle) :
  ∃ (d : Diameter c) (ch : Chord c), bisects d ch ∧ ¬perpendicular d ch :=
sorry

end NUMINAMATH_CALUDE_not_all_bisecting_diameters_perpendicular_l1074_107426


namespace NUMINAMATH_CALUDE_male_to_female_ratio_l1074_107496

/-- Represents the Math club with its member composition -/
structure MathClub where
  total_members : ℕ
  female_members : ℕ
  male_members : ℕ
  total_is_sum : total_members = female_members + male_members

/-- The specific Math club instance from the problem -/
def problem_club : MathClub :=
  { total_members := 18
    female_members := 6
    male_members := 12
    total_is_sum := by rfl }

/-- The ratio of male to female members is 2:1 -/
theorem male_to_female_ratio (club : MathClub) 
  (h1 : club.total_members = 18) 
  (h2 : club.female_members = 6) : 
  club.male_members / club.female_members = 2 := by
  sorry

#check male_to_female_ratio problem_club rfl rfl

end NUMINAMATH_CALUDE_male_to_female_ratio_l1074_107496


namespace NUMINAMATH_CALUDE_total_tickets_proof_l1074_107446

/-- The number of tickets Tom spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth : ℕ := 28

/-- The number of rides Tom went on -/
def number_of_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 4

/-- The total number of tickets Tom bought at the state fair -/
def total_tickets : ℕ := tickets_spent_at_booth + number_of_rides * cost_per_ride

theorem total_tickets_proof : total_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_proof_l1074_107446


namespace NUMINAMATH_CALUDE_fuel_efficiency_savings_l1074_107440

theorem fuel_efficiency_savings
  (old_efficiency : ℝ)
  (new_efficiency_improvement : ℝ)
  (fuel_cost_increase : ℝ)
  (h1 : new_efficiency_improvement = 0.3)
  (h2 : fuel_cost_increase = 0.25)
  : ∃ (savings : ℝ), abs (savings - 0.0385) < 0.0001 :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_efficiency_savings_l1074_107440


namespace NUMINAMATH_CALUDE_mall_profit_l1074_107401

-- Define the cost prices of type A and B
def cost_A : ℝ := 120
def cost_B : ℝ := 100

-- Define the number of units of each type
def units_A : ℝ := 50
def units_B : ℝ := 30

-- Define the conditions
axiom cost_difference : cost_A = cost_B + 20
axiom cost_equality : 5 * cost_A = 6 * cost_B
axiom total_cost : cost_A * units_A + cost_B * units_B = 9000
axiom total_units : units_A + units_B = 80

-- Define the selling prices
def sell_A : ℝ := cost_A * 1.5 * 0.8
def sell_B : ℝ := cost_B + 30

-- Define the total profit
def total_profit : ℝ := (sell_A - cost_A) * units_A + (sell_B - cost_B) * units_B

-- Theorem to prove
theorem mall_profit : 
  cost_A = 120 ∧ cost_B = 100 ∧ total_profit = 2100 :=
sorry

end NUMINAMATH_CALUDE_mall_profit_l1074_107401


namespace NUMINAMATH_CALUDE_g_2_4_neg1_eq_neg7_div_3_l1074_107403

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (a + b - c) / (a - b + c)

/-- Theorem stating that g(2, 4, -1) = -7/3 -/
theorem g_2_4_neg1_eq_neg7_div_3 : g 2 4 (-1) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_g_2_4_neg1_eq_neg7_div_3_l1074_107403


namespace NUMINAMATH_CALUDE_pen_sales_problem_l1074_107433

theorem pen_sales_problem (d : ℕ) : 
  (96 + 44 * d) / (d + 1) = 48 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_problem_l1074_107433


namespace NUMINAMATH_CALUDE_odot_composition_l1074_107420

/-- Custom operation ⊙ -/
def odot (x y : ℝ) : ℝ := x^2 + x*y - y^2

/-- Theorem stating that h ⊙ (h ⊙ h) = -4 when h = 2 -/
theorem odot_composition (h : ℝ) (h_eq : h = 2) : odot h (odot h h) = -4 := by
  sorry

end NUMINAMATH_CALUDE_odot_composition_l1074_107420


namespace NUMINAMATH_CALUDE_cube_inequality_l1074_107436

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x > a^y) : x^3 < y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1074_107436


namespace NUMINAMATH_CALUDE_system_solutions_l1074_107435

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * (x^2 - 3*y^2) = 16
def equation2 (x y : ℝ) : Prop := y * (3*x^2 - y^2) = 88

-- Define the approximate equality for real numbers
def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

-- Theorem statement
theorem system_solutions :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    -- Exact solution
    equation1 x₁ y₁ ∧ equation2 x₁ y₁ ∧ x₁ = 4 ∧ y₁ = 2 ∧
    -- Approximate solutions
    equation1 x₂ y₂ ∧ equation2 x₂ y₂ ∧ 
    approx_equal x₂ (-3.7) 0.1 ∧ approx_equal y₂ 2.5 0.1 ∧
    equation1 x₃ y₃ ∧ equation2 x₃ y₃ ∧ 
    approx_equal x₃ (-0.3) 0.1 ∧ approx_equal y₃ (-4.5) 0.1 :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l1074_107435


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_range_theorem_l1074_107424

noncomputable section

def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (A + B + C = Real.pi) ∧ 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem triangle_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c) :
  a / b = 2 := 
sorry

theorem triangle_range_theorem 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_equation : (Real.cos B - 2 * Real.cos A) / (2 * a - b) = Real.cos C / c)
  (h_obtuse : A > Real.pi / 2)
  (h_c : c = 3) :
  Real.sqrt 3 < b ∧ b < 3 := 
sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_range_theorem_l1074_107424


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1074_107485

def team_size : ℕ := 12
def lineup_size : ℕ := 5
def non_captain_size : ℕ := lineup_size - 1

theorem starting_lineup_count : 
  team_size * (Nat.choose (team_size - 1) non_captain_size) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1074_107485


namespace NUMINAMATH_CALUDE_survival_probability_estimate_l1074_107470

/-- Represents the survival data for a sample of seedlings -/
structure SeedlingSample where
  transplanted : ℕ
  survived : ℕ
  survivalRate : ℚ

/-- The data set of seedling survival samples -/
def seedlingData : List SeedlingSample := [
  ⟨20, 15, 75/100⟩,
  ⟨40, 33, 33/40⟩,
  ⟨100, 78, 39/50⟩,
  ⟨200, 158, 79/100⟩,
  ⟨400, 321, 801/1000⟩,
  ⟨1000, 801, 801/1000⟩
]

/-- Estimates the overall probability of seedling survival -/
def estimateSurvivalProbability (data : List SeedlingSample) : ℚ :=
  sorry

/-- Theorem stating that the estimated survival probability is approximately 0.80 -/
theorem survival_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateSurvivalProbability seedlingData - 4/5| < ε :=
sorry

end NUMINAMATH_CALUDE_survival_probability_estimate_l1074_107470


namespace NUMINAMATH_CALUDE_composition_f_equals_inverse_e_l1074_107495

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem composition_f_equals_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_composition_f_equals_inverse_e_l1074_107495


namespace NUMINAMATH_CALUDE_max_log_sum_and_min_reciprocal_sum_l1074_107417

open Real

theorem max_log_sum_and_min_reciprocal_sum (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + 5 * y = 20) :
  (∃ (u : ℝ), u = log x + log y ∧ ∀ (v : ℝ), v = log x + log y → v ≤ u) ∧
  u = 1 ∧
  (∃ (w : ℝ), w = 1/x + 1/y ∧ ∀ (z : ℝ), z = 1/x + 1/y → w ≤ z) ∧
  w = (7 + 2 * sqrt 10) / 20 := by
  sorry

end NUMINAMATH_CALUDE_max_log_sum_and_min_reciprocal_sum_l1074_107417


namespace NUMINAMATH_CALUDE_max_distance_between_cubic_and_quadratic_roots_l1074_107458

open Complex Set

theorem max_distance_between_cubic_and_quadratic_roots : ∃ (max_dist : ℝ),
  max_dist = 3 * Real.sqrt 7 ∧
  ∀ (a b : ℂ),
    (a^3 - 27 = 0) →
    (b^2 - 6*b + 9 = 0) →
    abs (a - b) ≤ max_dist ∧
    ∃ (a₀ b₀ : ℂ),
      (a₀^3 - 27 = 0) ∧
      (b₀^2 - 6*b₀ + 9 = 0) ∧
      abs (a₀ - b₀) = max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_cubic_and_quadratic_roots_l1074_107458


namespace NUMINAMATH_CALUDE_three_top_numbers_count_l1074_107478

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if three consecutive numbers satisfy the "Three Top Numbers" conditions -/
def isThreeTopNumbers (n : ℕ) : Prop :=
  isTwoDigit n ∧ isTwoDigit (n + 1) ∧ isTwoDigit (n + 2) ∧
  isTwoDigit (n + (n + 1) + (n + 2)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit n) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 1)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 2))

/-- The theorem stating that there are exactly 5 sets of "Three Top Numbers" -/
theorem three_top_numbers_count :
  ∃! (s : Finset ℕ), (∀ n ∈ s, isThreeTopNumbers n) ∧ s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_three_top_numbers_count_l1074_107478


namespace NUMINAMATH_CALUDE_symmetry_x_axis_of_point_A_l1074_107456

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_x_axis_of_point_A :
  let A : Point3D := { x := 1, y := 2, z := 1 }
  symmetry_x_axis A = { x := 1, y := 2, z := -1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_of_point_A_l1074_107456


namespace NUMINAMATH_CALUDE_sum_and_product_problem_l1074_107439

theorem sum_and_product_problem (x y : ℝ) 
  (sum_eq : x + y = 15) 
  (product_eq : x * y = 36) : 
  (1 / x + 1 / y = 5 / 12) ∧ (x^2 + y^2 = 153) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_problem_l1074_107439


namespace NUMINAMATH_CALUDE_money_needed_proof_l1074_107498

def car_wash_count : ℕ := 5
def car_wash_price : ℚ := 8.5
def dog_walk_count : ℕ := 4
def dog_walk_price : ℚ := 6.75
def lawn_mow_count : ℕ := 3
def lawn_mow_price : ℚ := 12.25
def bicycle_price : ℚ := 150.25
def helmet_price : ℚ := 35.75
def lock_price : ℚ := 24.5

def total_money_made : ℚ := 
  car_wash_count * car_wash_price + 
  dog_walk_count * dog_walk_price + 
  lawn_mow_count * lawn_mow_price

def total_cost : ℚ := 
  bicycle_price + helmet_price + lock_price

theorem money_needed_proof : 
  total_cost - total_money_made = 104.25 := by sorry

end NUMINAMATH_CALUDE_money_needed_proof_l1074_107498


namespace NUMINAMATH_CALUDE_sally_balloons_count_l1074_107423

/-- The number of blue balloons Joan initially has -/
def initial_balloons : ℕ := 9

/-- The number of blue balloons Joan gives to Jessica -/
def balloons_given_away : ℕ := 2

/-- The number of blue balloons Joan has after all transactions -/
def final_balloons : ℕ := 12

/-- The number of blue balloons Sally gives to Joan -/
def balloons_from_sally : ℕ := 5

theorem sally_balloons_count : 
  initial_balloons + balloons_from_sally - balloons_given_away = final_balloons :=
by sorry

end NUMINAMATH_CALUDE_sally_balloons_count_l1074_107423


namespace NUMINAMATH_CALUDE_florist_roses_l1074_107408

theorem florist_roses (initial : Float) (first_pick : Float) (second_pick : Float) 
  (h1 : initial = 37.0) 
  (h2 : first_pick = 16.0) 
  (h3 : second_pick = 19.0) : 
  initial + first_pick + second_pick = 72.0 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l1074_107408


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1074_107480

theorem largest_perfect_square_factor_of_1800 : 
  (∃ (n : ℕ), n^2 = 900 ∧ n^2 ∣ 1800 ∧ ∀ (m : ℕ), m^2 ∣ 1800 → m^2 ≤ 900) := by
  sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l1074_107480


namespace NUMINAMATH_CALUDE_five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l1074_107490

/-- The maximum number of intersection points for n lines in a plane,
    where no three lines intersect at one point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_triple_intersections : Bool

/-- Theorem stating the impossibility of 5 lines with 11 intersections -/
theorem five_lines_eleven_intersections_impossible :
  ∀ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true →
    config.num_intersections ≠ 11 :=
sorry

/-- Theorem stating the possibility of 5 lines with 9 intersections -/
theorem five_lines_nine_intersections_possible :
  ∃ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true ∧
    config.num_intersections = 9 :=
sorry

end NUMINAMATH_CALUDE_five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l1074_107490


namespace NUMINAMATH_CALUDE_street_length_l1074_107491

theorem street_length (forest_area : ℝ) (street_area : ℝ) (trees_per_sqm : ℝ) (total_trees : ℝ) :
  forest_area = 3 * street_area →
  trees_per_sqm = 4 →
  total_trees = 120000 →
  street_area = (100 : ℝ) ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_street_length_l1074_107491


namespace NUMINAMATH_CALUDE_gis_main_functions_l1074_107489

/-- Represents the main functions of geographic information technology -/
inductive GISFunction
  | input : GISFunction
  | manage : GISFunction
  | analyze : GISFunction
  | express : GISFunction

/-- Represents the type of data handled by geographic information technology -/
def GeospatialData : Type := Unit

/-- The set of main functions of geographic information technology -/
def mainFunctions : Set GISFunction :=
  {GISFunction.input, GISFunction.manage, GISFunction.analyze, GISFunction.express}

/-- States that the main functions of geographic information technology
    are to input, manage, analyze, and express geospatial data -/
theorem gis_main_functions :
  ∀ f : GISFunction, f ∈ mainFunctions →
  ∃ (d : GeospatialData), (f = GISFunction.input ∨ f = GISFunction.manage ∨
                           f = GISFunction.analyze ∨ f = GISFunction.express) :=
sorry

end NUMINAMATH_CALUDE_gis_main_functions_l1074_107489


namespace NUMINAMATH_CALUDE_rational_set_not_just_positive_and_negative_l1074_107473

theorem rational_set_not_just_positive_and_negative : 
  ∃ q : ℚ, q ∉ {x : ℚ | x > 0} ∪ {x : ℚ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_rational_set_not_just_positive_and_negative_l1074_107473


namespace NUMINAMATH_CALUDE_triangular_array_coins_l1074_107413

theorem triangular_array_coins (N : ℕ) : 
  (N * (N + 1)) / 2 = 2485 → N = 70 ∧ (N / 10 * (N % 10)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coins_l1074_107413


namespace NUMINAMATH_CALUDE_problem_solution_l1074_107434

theorem problem_solution :
  (∃ n : ℕ, n = 4 * 7 + 5 ∧ n = 33) ∧
  (∃ m : ℕ, m * 6 = 300 ∧ m = 50) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1074_107434


namespace NUMINAMATH_CALUDE_fault_line_movement_l1074_107443

/-- The total movement of a fault line over two years, given its movement in each year. -/
theorem fault_line_movement (movement_past_year : ℝ) (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_fault_line_movement_l1074_107443


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt10_l1074_107429

theorem sqrt_sum_equals_2sqrt10 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt10_l1074_107429


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_l1074_107404

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 6*x - 14

-- Theorem statement
theorem sum_of_abs_roots :
  ∃ (r₁ r₂ r₃ r₄ : ℝ),
    (∀ x : ℝ, p x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 3 + Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_l1074_107404


namespace NUMINAMATH_CALUDE_factorial_ratio_l1074_107452

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1074_107452


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l1074_107493

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * (x + y) = 5 * x + y) :
  ∀ z : ℝ, 2 * x + y ≥ 9 ∧ (∃ x₀ y₀ : ℝ, 2 * x₀ + y₀ = 9 ∧ x₀ > 0 ∧ y₀ > 0 ∧ x₀ * (x₀ + y₀) = 5 * x₀ + y₀) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l1074_107493


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1074_107428

/-- Restaurant bill calculation -/
theorem restaurant_bill_calculation
  (appetizer_cost : ℝ)
  (num_entrees : ℕ)
  (entree_cost : ℝ)
  (tip_percentage : ℝ)
  (h1 : appetizer_cost = 10)
  (h2 : num_entrees = 4)
  (h3 : entree_cost = 20)
  (h4 : tip_percentage = 0.20) :
  appetizer_cost + num_entrees * entree_cost + (appetizer_cost + num_entrees * entree_cost) * tip_percentage = 108 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1074_107428


namespace NUMINAMATH_CALUDE_rachel_apples_remaining_l1074_107427

/-- The number of apples remaining on a tree after some are picked. -/
def applesRemaining (initial : ℕ) (picked : ℕ) : ℕ :=
  initial - picked

/-- Theorem: There are 3 apples remaining on Rachel's tree. -/
theorem rachel_apples_remaining :
  applesRemaining 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_remaining_l1074_107427


namespace NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l1074_107476

/-- Given a sequence of seven consecutive even integers with a sum of 700,
    the smallest number in the sequence is 94. -/
theorem smallest_of_seven_consecutive_evens (seq : List ℤ) : 
  seq.length = 7 ∧ 
  (∀ i ∈ seq, ∃ k : ℤ, i = 2 * k) ∧ 
  (∀ i j, i ∈ seq → j ∈ seq → i ≠ j → (i - j).natAbs = 2) ∧
  seq.sum = 700 →
  seq.minimum? = some 94 := by
sorry

end NUMINAMATH_CALUDE_smallest_of_seven_consecutive_evens_l1074_107476


namespace NUMINAMATH_CALUDE_hannah_strawberries_l1074_107477

/-- The number of strawberries Hannah has at the end of April -/
def strawberries_at_end_of_april (daily_harvest : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) : ℕ :=
  daily_harvest * days_in_april - (given_away + stolen)

theorem hannah_strawberries :
  strawberries_at_end_of_april 5 30 20 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_hannah_strawberries_l1074_107477


namespace NUMINAMATH_CALUDE_simplify_expression_l1074_107414

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1074_107414


namespace NUMINAMATH_CALUDE_jasmine_total_weight_l1074_107460

/-- Calculates the total weight in pounds that Jasmine has to carry given the weight of chips and cookies, and the quantities purchased. -/
theorem jasmine_total_weight (chip_weight : ℕ) (cookie_weight : ℕ) (chip_quantity : ℕ) (cookie_multiplier : ℕ) : 
  chip_weight = 20 →
  cookie_weight = 9 →
  chip_quantity = 6 →
  cookie_multiplier = 4 →
  (chip_weight * chip_quantity + cookie_weight * (cookie_multiplier * chip_quantity)) / 16 = 21 := by
  sorry

#check jasmine_total_weight

end NUMINAMATH_CALUDE_jasmine_total_weight_l1074_107460


namespace NUMINAMATH_CALUDE_centroid_distance_relation_l1074_107451

/-- Given a triangle ABC with centroid G and any point P in the plane, 
    prove that the sum of squared distances from P to the vertices of the triangle 
    is equal to the sum of squared distances from G to the vertices 
    plus three times the squared distance from G to P. -/
theorem centroid_distance_relation (A B C G P : ℝ × ℝ) : 
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + 
  (B.1 - P.1)^2 + (B.2 - P.2)^2 + 
  (C.1 - P.1)^2 + (C.2 - P.2)^2 = 
  (A.1 - G.1)^2 + (A.2 - G.2)^2 + 
  (B.1 - G.1)^2 + (B.2 - G.2)^2 + 
  (C.1 - G.1)^2 + (C.2 - G.2)^2 + 
  3 * ((G.1 - P.1)^2 + (G.2 - P.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_centroid_distance_relation_l1074_107451


namespace NUMINAMATH_CALUDE_chris_video_game_cost_l1074_107418

def video_game_cost (hourly_rate : ℕ) (hours_worked : ℕ) (candy_cost : ℕ) (leftover : ℕ) : ℕ :=
  hourly_rate * hours_worked - candy_cost - leftover

theorem chris_video_game_cost :
  video_game_cost 8 9 5 7 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chris_video_game_cost_l1074_107418


namespace NUMINAMATH_CALUDE_route_down_length_l1074_107449

/-- Proves that the length of the route down the mountain is 15 miles -/
theorem route_down_length (time_up time_down : ℝ) (rate_up rate_down : ℝ) 
  (h1 : time_up = time_down)
  (h2 : rate_down = 1.5 * rate_up)
  (h3 : rate_up = 5)
  (h4 : time_up = 2) : 
  rate_down * time_down = 15 := by
  sorry

end NUMINAMATH_CALUDE_route_down_length_l1074_107449


namespace NUMINAMATH_CALUDE_table_height_l1074_107405

/-- Given three rectangular boxes with heights b, r, and g, and a table with height h,
    prove that h = 91 when the following conditions are met:
    1. h + b - g = 111
    2. h + r - b = 80
    3. h + g - r = 82 -/
theorem table_height (h b r g : ℝ) 
    (eq1 : h + b - g = 111)
    (eq2 : h + r - b = 80)
    (eq3 : h + g - r = 82) : h = 91 := by
  sorry

end NUMINAMATH_CALUDE_table_height_l1074_107405


namespace NUMINAMATH_CALUDE_book_price_increase_l1074_107457

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) :
  new_price = 420 ∧ 
  increase_percentage = 40 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l1074_107457


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1074_107407

/-- Given a cone with base radius 1 and lateral surface that unfolds to a 
    sector with a 90° central angle, its lateral surface area is 4π. -/
theorem cone_lateral_surface_area (r : Real) (θ : Real) : 
  r = 1 → θ = 90 → ∃ (l : Real), l * θ / 360 * (2 * Real.pi) = 2 * Real.pi ∧ 
    r * l * Real.pi = 4 * Real.pi := by
  sorry

#check cone_lateral_surface_area

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1074_107407


namespace NUMINAMATH_CALUDE_ambiguous_decomposition_l1074_107402

def M : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k - 3}

def isSimple (n : ℕ) : Prop :=
  n ∈ M ∧ ∀ a b : ℕ, a ∈ M → b ∈ M → a * b = n → (a = 1 ∨ b = 1)

theorem ambiguous_decomposition : ∃ n : ℕ,
  n ∈ M ∧ (∃ a b c d : ℕ,
    isSimple a ∧ isSimple b ∧ isSimple c ∧ isSimple d ∧
    a * b = n ∧ c * d = n ∧ (a ≠ c ∨ b ≠ d)) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_decomposition_l1074_107402


namespace NUMINAMATH_CALUDE_yolanda_total_points_l1074_107450

/-- Calculate the total points scored by a basketball player over a season. -/
def total_points_scored (games : ℕ) (free_throws two_pointers three_pointers : ℕ) : ℕ :=
  games * (free_throws * 1 + two_pointers * 2 + three_pointers * 3)

/-- Theorem: Yolanda's total points scored over the entire season is 345. -/
theorem yolanda_total_points : 
  total_points_scored 15 4 5 3 = 345 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_total_points_l1074_107450


namespace NUMINAMATH_CALUDE_min_values_l1074_107406

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the condition from the problem
def condition (x y : ℝ) : Prop := lg (3 * x) + lg y = lg (x + y + 1)

-- Theorem statement
theorem min_values {x y : ℝ} (h : condition x y) :
  (∀ a b : ℝ, condition a b → x * y ≤ a * b) ∧
  (∀ c d : ℝ, condition c d → x + y ≤ c + d) :=
by sorry

end NUMINAMATH_CALUDE_min_values_l1074_107406


namespace NUMINAMATH_CALUDE_park_trees_l1074_107494

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: The park will have 77 walnut trees after planting -/
theorem park_trees : total_trees 33 44 = 77 := by
  sorry

end NUMINAMATH_CALUDE_park_trees_l1074_107494


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1074_107463

theorem polygon_interior_angles (n : ℕ) (d : ℝ) (largest_angle : ℝ) : 
  n ≥ 3 →
  d = 3 →
  largest_angle = 150 →
  (n : ℝ) * (2 * largest_angle - d * (n - 1)) / 2 = 180 * (n - 2) →
  n = 24 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1074_107463


namespace NUMINAMATH_CALUDE_philip_paintings_l1074_107486

/-- The number of paintings a painter will have after a certain number of days -/
def total_paintings (initial_paintings : ℕ) (paintings_per_day : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings : total_paintings 20 2 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_l1074_107486


namespace NUMINAMATH_CALUDE_city_population_ratio_l1074_107432

theorem city_population_ratio (x y z : ℕ) (hxy : ∃ k : ℕ, x = k * y) (hyz : y = 2 * z) (hxz : x = 14 * z) :
  x / y = 7 :=
by sorry

end NUMINAMATH_CALUDE_city_population_ratio_l1074_107432


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1074_107474

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (t s : ℝ), 
    t > 0 ∧ s > 0 ∧  -- positive side lengths
    3 * t - 4 * s = 4020 ∧  -- perimeter difference
    t = |s - 12| + d ∧  -- side length relationship
    4 * s > 0)  -- square perimeter > 0
  ↔ d > 1352 :=
sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_difference_l1074_107474


namespace NUMINAMATH_CALUDE_parallelogram_adjacent_side_l1074_107484

/-- The length of the other adjacent side of a parallelogram with perimeter 16 and one side length 5 is 3. -/
theorem parallelogram_adjacent_side (perimeter : ℝ) (side_a : ℝ) (side_b : ℝ) 
  (h1 : perimeter = 16) 
  (h2 : side_a = 5) 
  (h3 : perimeter = 2 * (side_a + side_b)) : 
  side_b = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_adjacent_side_l1074_107484


namespace NUMINAMATH_CALUDE_vladimir_digits_puzzle_l1074_107455

/-- Represents a three-digit number formed by digits a, b, c in that order -/
def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem vladimir_digits_puzzle :
  ∀ a b c : ℕ,
  a > b → b > c → c > 0 →
  form_number a b c = form_number c b a + form_number c a b →
  a = 9 ∧ b = 5 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_vladimir_digits_puzzle_l1074_107455


namespace NUMINAMATH_CALUDE_orange_crates_count_l1074_107499

theorem orange_crates_count :
  ∀ (num_crates : ℕ),
    (∀ (crate : ℕ), crate ≤ num_crates → 150 * num_crates + 16 * 30 = 2280) →
    num_crates = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_crates_count_l1074_107499


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_ratio_l1074_107409

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ

/-- Sum of the first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithmeticGeometricSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

/-- Theorem: For an arithmetic-geometric sequence, if s_6 : s_3 = 1 : 2, then s_9 : s_3 = 3/4 -/
theorem arithmetic_geometric_sum_ratio 
  (seq : ArithmeticGeometricSequence) 
  (h : sum_n seq 6 / sum_n seq 3 = 1 / 2) : 
  sum_n seq 9 / sum_n seq 3 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_ratio_l1074_107409


namespace NUMINAMATH_CALUDE_train_length_proof_l1074_107464

/-- Calculates the length of a train given bridge length, crossing time, and train speed. -/
def train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - bridge_length

/-- Proves that given the specific conditions, the train length is 844 meters. -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ)
  (h1 : bridge_length = 200)
  (h2 : crossing_time = 36)
  (h3 : train_speed = 29) :
  train_length bridge_length crossing_time train_speed = 844 := by
  sorry

#eval train_length 200 36 29

end NUMINAMATH_CALUDE_train_length_proof_l1074_107464


namespace NUMINAMATH_CALUDE_race_time_B_b_finish_time_l1074_107475

/-- Calculates the time taken by runner B to finish a race given the conditions --/
theorem race_time_B (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) : ℝ :=
  let distance_B_in_time_A := race_distance - beat_distance
  let speed_B := distance_B_in_time_A / time_A
  race_distance / speed_B

/-- Proves that B finishes the race in 25 seconds given the specified conditions --/
theorem b_finish_time (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) :
  race_time_B race_distance time_A beat_distance = 25 :=
by
  -- Assuming race_distance = 110, time_A = 20, and beat_distance = 22
  have h1 : race_distance = 110 := by sorry
  have h2 : time_A = 20 := by sorry
  have h3 : beat_distance = 22 := by sorry
  
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_race_time_B_b_finish_time_l1074_107475


namespace NUMINAMATH_CALUDE_problem_solution_l1074_107415

theorem problem_solution : 
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + (8 : Real)^(1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1074_107415


namespace NUMINAMATH_CALUDE_chocolate_chip_recipe_l1074_107488

theorem chocolate_chip_recipe (total_recipes : ℕ) (total_cups : ℕ) 
  (h1 : total_recipes = 23) (h2 : total_cups = 46) :
  total_cups / total_recipes = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_recipe_l1074_107488


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_points_l1074_107471

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of AB
  c : ℝ  -- length of CD
  h : ℝ  -- perpendicular distance from A to CD
  a_positive : 0 < a
  c_positive : 0 < c
  h_positive : 0 < h
  c_le_a : c ≤ a  -- As CD is parallel to and shorter than AB

/-- The point X on the axis of symmetry -/
def X (t : IsoscelesTrapezoid) := {x : ℝ // 0 ≤ x ∧ x ≤ t.h}

/-- The theorem stating the conditions for X to exist and its distance from AB -/
theorem isosceles_trapezoid_right_angle_points (t : IsoscelesTrapezoid) :
  ∃ (x : X t), (x.val = t.h / 2 - Real.sqrt (t.h^2 - t.a * t.c) / 2 ∨
                x.val = t.h / 2 + Real.sqrt (t.h^2 - t.a * t.c) / 2) ↔
  t.h^2 ≥ t.a * t.c :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_points_l1074_107471


namespace NUMINAMATH_CALUDE_parabola_focus_x_coord_l1074_107445

/-- A parabola defined by parametric equations -/
structure ParametricParabola where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The focus of a parabola -/
structure ParabolaFocus where
  x : ℝ
  y : ℝ

/-- Theorem: The x-coordinate of the focus of the parabola given by x = 4t² and y = 4t is 1 -/
theorem parabola_focus_x_coord (p : ParametricParabola) 
  (h1 : p.x = fun t => 4 * t^2)
  (h2 : p.y = fun t => 4 * t) : 
  ∃ f : ParabolaFocus, f.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_x_coord_l1074_107445


namespace NUMINAMATH_CALUDE_bmw_length_l1074_107466

theorem bmw_length : 
  let straight_segments : ℕ := 7
  let straight_length : ℝ := 2
  let diagonal_segments : ℕ := 2
  let diagonal_length : ℝ := Real.sqrt 2
  straight_segments * straight_length + diagonal_segments * diagonal_length = 14 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bmw_length_l1074_107466


namespace NUMINAMATH_CALUDE_bad_carrots_count_l1074_107465

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  haley_carrots = 39 → mom_carrots = 38 → good_carrots = 64 → 
  haley_carrots + mom_carrots - good_carrots = 13 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l1074_107465


namespace NUMINAMATH_CALUDE_no_eighteen_consecutive_good_l1074_107467

/-- A natural number is "good" if it has exactly two prime divisors -/
def isGood (n : ℕ) : Prop :=
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n.divisors = {1, p, q, n})

/-- Theorem: There do not exist 18 consecutive natural numbers that are all "good" -/
theorem no_eighteen_consecutive_good :
  ¬ ∃ k : ℕ, ∀ i : ℕ, i < 18 → isGood (k + i) := by
  sorry

end NUMINAMATH_CALUDE_no_eighteen_consecutive_good_l1074_107467


namespace NUMINAMATH_CALUDE_wall_width_proof_l1074_107431

/-- Given a rectangular wall and a square mirror, if the mirror's area is half the wall's area,
    prove that the wall's width is 68 inches. -/
theorem wall_width_proof (wall_length wall_width mirror_side : ℝ) : 
  wall_length = 85.76470588235294 →
  mirror_side = 54 →
  (mirror_side * mirror_side) = (wall_length * wall_width) / 2 →
  wall_width = 68 := by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l1074_107431


namespace NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l1074_107462

theorem sum_of_squares_with_given_means (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_with_given_means_l1074_107462


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1074_107461

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampledCount where
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Checks if the sampling is stratified correctly -/
def is_stratified_sampling (ec : EmployeeCount) (sample_size : ℕ) (sc : SampledCount) : Prop :=
  sc.senior * ec.total = sample_size * ec.senior ∧
  sc.middle * ec.total = sample_size * ec.middle ∧
  sc.general * ec.total = sample_size * ec.general

theorem stratified_sampling_theorem (ec : EmployeeCount) (sample_size : ℕ) :
  ec.total = ec.senior + ec.middle + ec.general →
  ∃ (sc : SampledCount), 
    sc.senior + sc.middle + sc.general = sample_size ∧
    is_stratified_sampling ec sample_size sc := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1074_107461


namespace NUMINAMATH_CALUDE_parents_in_auditorium_l1074_107459

/-- Given a school play with girls and boys, and both parents of each kid attending,
    calculate the total number of parents in the auditorium. -/
theorem parents_in_auditorium (girls boys : ℕ) (h1 : girls = 6) (h2 : boys = 8) :
  2 * (girls + boys) = 28 := by
  sorry

end NUMINAMATH_CALUDE_parents_in_auditorium_l1074_107459


namespace NUMINAMATH_CALUDE_xiaolin_mean_calculation_l1074_107497

theorem xiaolin_mean_calculation 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (x : ℝ) 
  (hx : x = (a + b) / 2) 
  (y : ℝ) 
  (hy : y = (x + c) / 2) : 
  y < (a + b + c) / 3 := by
sorry

end NUMINAMATH_CALUDE_xiaolin_mean_calculation_l1074_107497


namespace NUMINAMATH_CALUDE_beatrice_has_highest_answer_l1074_107483

def albert_calculation (x : ℕ) : ℕ := 2 * ((3 * x + 5) - 3)

def beatrice_calculation (x : ℕ) : ℕ := 2 * ((x * x + 3) - 7)

def carlos_calculation (x : ℕ) : ℚ := ((5 * x - 4 + 6) : ℚ) / 2

theorem beatrice_has_highest_answer :
  let start := 15
  beatrice_calculation start > albert_calculation start ∧
  (beatrice_calculation start : ℚ) > carlos_calculation start := by
sorry

#eval albert_calculation 15
#eval beatrice_calculation 15
#eval carlos_calculation 15

end NUMINAMATH_CALUDE_beatrice_has_highest_answer_l1074_107483


namespace NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l1074_107447

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.cos (α + Real.pi/4)^2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_plus_pi_fourth_l1074_107447


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1074_107481

theorem quadratic_factorization (m n : ℤ) : 
  (∀ x, x^2 - 7*x + n = (x - 3) * (x + m)) → m - n = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1074_107481


namespace NUMINAMATH_CALUDE_min_value_of_m_plus_2n_l1074_107400

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem min_value_of_m_plus_2n (a : ℝ) (m n : ℝ) :
  (∀ x, f a x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3) →
  (m > 0) →
  (n > 0) →
  (1 / m + 1 / (2 * n) = a) →
  (∀ p q, p > 0 → q > 0 → 1 / p + 1 / (2 * q) = a → p + 2 * q ≥ m + 2 * n) →
  m + 2 * n = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_m_plus_2n_l1074_107400


namespace NUMINAMATH_CALUDE_race_head_start_l1074_107438

/-- Given two runners A and B, where A's speed is 20/16 times B's speed,
    this theorem proves that A should give B a head start of 1/5 of the
    total race length for the race to end in a dead heat. -/
theorem race_head_start (v_B : ℝ) (L : ℝ) (h_pos_v : v_B > 0) (h_pos_L : L > 0) :
  let v_A := (20 / 16) * v_B
  let x := 1 / 5
  L / v_A = (L - x * L) / v_B := by
  sorry

#check race_head_start

end NUMINAMATH_CALUDE_race_head_start_l1074_107438


namespace NUMINAMATH_CALUDE_oplus_three_equals_fifteen_implies_a_equals_eleven_l1074_107437

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 3*a - 2*b^2

-- Theorem statement
theorem oplus_three_equals_fifteen_implies_a_equals_eleven :
  ∀ a : ℝ, oplus a 3 = 15 → a = 11 := by
sorry

end NUMINAMATH_CALUDE_oplus_three_equals_fifteen_implies_a_equals_eleven_l1074_107437


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_88200_l1074_107487

def sum_of_prime_factors (n : ℕ) : ℕ := (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_88200 :
  sum_of_prime_factors 88200 = 17 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_88200_l1074_107487


namespace NUMINAMATH_CALUDE_number_relationship_l1074_107412

theorem number_relationship (x m : ℚ) : 
  x = 25 / 3 → 
  (3 * x + 15 = m * x - 10) → 
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l1074_107412


namespace NUMINAMATH_CALUDE_equation_solution_l1074_107411

theorem equation_solution :
  ∃ y : ℝ, (6 * y / (y + 2) - 4 / (y + 2) = 2 / (y + 2)) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1074_107411


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1074_107430

theorem chess_tournament_players (total_games : ℕ) (h_total_games : total_games = 42) : 
  ∃ n : ℕ, n > 0 ∧ total_games = n * (n - 1) ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1074_107430


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1074_107419

theorem sum_of_fractions_equals_one 
  (a b c : ℝ) 
  (h : a * b * c = 1) : 
  (a / (a * b + a + 1)) + (b / (b * c + b + 1)) + (c / (c * a + c + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1074_107419


namespace NUMINAMATH_CALUDE_square_value_l1074_107453

theorem square_value : ∃ (square : ℝ), (6400000 : ℝ) / 400 = 1.6 * square ∧ square = 10000 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l1074_107453


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1074_107472

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1074_107472


namespace NUMINAMATH_CALUDE_rahul_savings_l1074_107442

/-- Rahul's savings problem -/
theorem rahul_savings (NSC PPF : ℕ) (h1 : 3 * (NSC / 3) = 2 * (PPF / 2)) (h2 : PPF = 72000) :
  NSC + PPF = 180000 := by
  sorry

end NUMINAMATH_CALUDE_rahul_savings_l1074_107442


namespace NUMINAMATH_CALUDE_omm_moo_not_synonyms_l1074_107422

/-- Represents a word in the Ancient Tribe language --/
inductive Word
| empty : Word
| cons : Char → Word → Word

/-- Counts the number of occurrences of a given character in a word --/
def count_char (c : Char) : Word → Nat
| Word.empty => 0
| Word.cons x rest => (if x = c then 1 else 0) + count_char c rest

/-- Calculates the difference between the count of 'M's and 'O's in a word --/
def m_o_difference (w : Word) : Int :=
  (count_char 'M' w : Int) - (count_char 'O' w : Int)

/-- Defines when two words are synonyms --/
def are_synonyms (w1 w2 : Word) : Prop :=
  m_o_difference w1 = m_o_difference w2

/-- Represents the word OMM --/
def omm : Word := Word.cons 'O' (Word.cons 'M' (Word.cons 'M' Word.empty))

/-- Represents the word MOO --/
def moo : Word := Word.cons 'M' (Word.cons 'O' (Word.cons 'O' Word.empty))

/-- Theorem stating that OMM and MOO are not synonyms --/
theorem omm_moo_not_synonyms : ¬(are_synonyms omm moo) := by
  sorry

end NUMINAMATH_CALUDE_omm_moo_not_synonyms_l1074_107422


namespace NUMINAMATH_CALUDE_pythagorean_theorem_geometric_dissection_l1074_107468

/-- Pythagorean theorem using geometric dissection -/
theorem pythagorean_theorem_geometric_dissection 
  (a b c : ℝ) 
  (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_hypotenuse : c = max a b)
  (h_inner_square : ∃ (s : ℝ), s = |b - a| ∧ s^2 = (b - a)^2)
  (h_area_equality : c^2 = 2*a*b + (b - a)^2) : 
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_geometric_dissection_l1074_107468


namespace NUMINAMATH_CALUDE_circle_through_three_points_l1074_107444

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The standard equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_three_points :
  let A : Point := ⟨-4, 0⟩
  let B : Point := ⟨0, 2⟩
  let O : Point := ⟨0, 0⟩
  let center : Point := ⟨-2, 1⟩
  let radius : ℝ := Real.sqrt 5
  (CircleEquation center radius) ∧
  (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
  (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
  (center.x - O.x)^2 + (center.y - O.y)^2 = radius^2 :=
by
  sorry

#check circle_through_three_points

end NUMINAMATH_CALUDE_circle_through_three_points_l1074_107444


namespace NUMINAMATH_CALUDE_probability_of_exact_score_l1074_107482

def num_questions : ℕ := 20
def num_choices : ℕ := 4
def correct_answers : ℕ := 10

def probability_correct : ℚ := 1 / num_choices
def probability_incorrect : ℚ := 1 - probability_correct

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_exact_score :
  (binomial_coefficient num_questions correct_answers : ℚ) *
  (probability_correct ^ correct_answers) *
  (probability_incorrect ^ (num_questions - correct_answers)) =
  93350805 / 1073741824 := by sorry

end NUMINAMATH_CALUDE_probability_of_exact_score_l1074_107482


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l1074_107416

theorem points_in_quadrant_I (x y : ℝ) : 
  y > -x + 6 ∧ y > 3*x - 2 → x > 0 ∧ y > 0 := by
sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l1074_107416


namespace NUMINAMATH_CALUDE_clothing_selection_probability_l1074_107410

/-- The probability of selecting exactly one shirt, one pair of shorts, one pair of socks, and one hat
    when randomly choosing 4 articles of clothing from a drawer containing 6 shirts, 7 pairs of shorts,
    8 pairs of socks, and 3 hats. -/
theorem clothing_selection_probability :
  let num_shirts : ℕ := 6
  let num_shorts : ℕ := 7
  let num_socks : ℕ := 8
  let num_hats : ℕ := 3
  let total_items : ℕ := num_shirts + num_shorts + num_socks + num_hats
  let favorable_outcomes : ℕ := num_shirts * num_shorts * num_socks * num_hats
  let total_outcomes : ℕ := Nat.choose total_items 4
  (favorable_outcomes : ℚ) / total_outcomes = 144 / 1815 := by
  sorry


end NUMINAMATH_CALUDE_clothing_selection_probability_l1074_107410


namespace NUMINAMATH_CALUDE_root_product_equality_l1074_107454

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_equality_l1074_107454


namespace NUMINAMATH_CALUDE_smallest_congruent_n_l1074_107441

theorem smallest_congruent_n (a b : ℤ) (h1 : a ≡ 23 [ZMOD 60]) (h2 : b ≡ 95 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 191 ∧ a - b ≡ n [ZMOD 60] ∧
  ∀ m : ℤ, 150 ≤ m ∧ m < n → ¬(a - b ≡ m [ZMOD 60]) ∧ n = 168 := by
  sorry

end NUMINAMATH_CALUDE_smallest_congruent_n_l1074_107441


namespace NUMINAMATH_CALUDE_parabola_intersection_l1074_107479

/-- The function f(x) = x² --/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: Given two points on the parabola y = x², with the first point at x = 1
    and the second at x = 4, if we trisect the line segment between these points
    and draw a horizontal line through the first trisection point (closer to the first point),
    then this line intersects the parabola at x = -2. --/
theorem parabola_intersection (x₁ x₂ x₃ : ℝ) (y₁ y₂ y₃ : ℝ) :
  x₁ = 1 →
  x₂ = 4 →
  y₁ = f x₁ →
  y₂ = f x₂ →
  let xc := (2 * x₁ + x₂) / 3
  let yc := f xc
  y₃ = yc →
  y₃ = f x₃ →
  x₃ ≠ xc →
  x₃ = -2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1074_107479
