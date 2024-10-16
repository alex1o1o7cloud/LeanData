import Mathlib

namespace NUMINAMATH_CALUDE_min_sticks_for_13_triangles_l2664_266404

/-- The minimum number of sticks needed to form n equilateral triangles -/
def min_sticks (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 3
  else if n = 2 then 5
  else if n = 3 then 7
  else 2 * n + 1

/-- Theorem: The minimum number of sticks required to form 13 equilateral triangles is 27 -/
theorem min_sticks_for_13_triangles :
  min_sticks 13 = 27 := by
  sorry

/-- Lemma: The minimum number of sticks for n triangles (n > 3) follows the pattern 2n + 1 -/
lemma min_sticks_pattern (n : ℕ) (h : n > 3) :
  min_sticks n = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_sticks_for_13_triangles_l2664_266404


namespace NUMINAMATH_CALUDE_square_root_problem_l2664_266468

theorem square_root_problem (x : ℝ) (a : ℝ) 
  (h1 : x > 0)
  (h2 : Real.sqrt x = 3 * a - 4)
  (h3 : Real.sqrt x = 1 - 6 * a) :
  a = -1 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2664_266468


namespace NUMINAMATH_CALUDE_trig_equation_solution_l2664_266480

theorem trig_equation_solution (x : ℝ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 →
  ∃ k : ℤ, x = π * (2 * k + 1) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l2664_266480


namespace NUMINAMATH_CALUDE_double_age_in_two_years_l2664_266410

/-- The number of years until a man's age is twice his son's age -/
def years_until_double_age (son_age : ℕ) (age_difference : ℕ) : ℕ :=
  let man_age := son_age + age_difference
  (man_age - 2 * son_age) / (2 - 1)

/-- Theorem: Given the son's age is 22 and the age difference is 24,
    the number of years until the man's age is twice his son's age is 2 -/
theorem double_age_in_two_years :
  years_until_double_age 22 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_two_years_l2664_266410


namespace NUMINAMATH_CALUDE_sin_theta_value_l2664_266417

theorem sin_theta_value (θ : Real) (h1 : θ ∈ Set.Icc (π/4) (π/2)) (h2 : Real.sin (2*θ) = 3*Real.sqrt 7/8) : 
  Real.sin θ = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l2664_266417


namespace NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2664_266459

/-- Given a train that travels 40 km in one hour (including stoppages) and stops for 20 minutes each hour, 
    its speed excluding stoppages is 60 kmph. -/
theorem train_speed_excluding_stoppages : 
  ∀ (speed_with_stops : ℝ) (stop_time : ℝ) (total_time : ℝ),
  speed_with_stops = 40 →
  stop_time = 20 →
  total_time = 60 →
  (total_time - stop_time) / total_time * speed_with_stops = 60 := by
sorry

end NUMINAMATH_CALUDE_train_speed_excluding_stoppages_l2664_266459


namespace NUMINAMATH_CALUDE_network_connections_l2664_266490

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  (n * k) / 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_network_connections_l2664_266490


namespace NUMINAMATH_CALUDE_b_squared_neq_ac_sufficient_not_necessary_l2664_266411

-- Define what it means for three numbers to form a geometric sequence
def is_geometric_sequence (a b c : ℝ) : Prop :=
  (b / a = c / b) ∨ (a = 0 ∧ b = 0) ∨ (b = 0 ∧ c = 0)

-- State the theorem
theorem b_squared_neq_ac_sufficient_not_necessary :
  (∀ a b c : ℝ, b^2 ≠ a*c → ¬is_geometric_sequence a b c) ∧
  (∃ a b c : ℝ, b^2 = a*c ∧ ¬is_geometric_sequence a b c) := by sorry

end NUMINAMATH_CALUDE_b_squared_neq_ac_sufficient_not_necessary_l2664_266411


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2664_266400

-- Define the given line
def given_line : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + 5 = 0}

-- Define the point that line l passes through
def point : ℝ × ℝ := (1, -4)

-- Define parallel line
def parallel_line (m : ℝ) : Set (ℝ × ℝ) := {(x, y) | 2 * x + 3 * y + m = 0}

-- Define perpendicular line
def perpendicular_line (n : ℝ) : Set (ℝ × ℝ) := {(x, y) | 3 * x - 2 * y - n = 0}

-- Theorem for parallel case
theorem parallel_line_theorem :
  ∃ m : ℝ, point ∈ parallel_line m ∧ m = 10 :=
sorry

-- Theorem for perpendicular case
theorem perpendicular_line_theorem :
  ∃ n : ℝ, point ∈ perpendicular_line n ∧ n = 11 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l2664_266400


namespace NUMINAMATH_CALUDE_worker_count_proof_l2664_266495

theorem worker_count_proof : ∃ (x y : ℕ), 
  y = (15 * x) / 19 ∧ 
  (4 * y) / 7 < 1000 ∧ 
  (3 * x) / 5 > 1000 ∧ 
  x = 1995 ∧ 
  y = 1575 := by
sorry

end NUMINAMATH_CALUDE_worker_count_proof_l2664_266495


namespace NUMINAMATH_CALUDE_no_real_roots_for_ff_l2664_266433

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The property that f(x) = x has no real roots -/
def NoRealRootsForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

theorem no_real_roots_for_ff (a b c : ℝ) :
  let f := QuadraticPolynomial a b c
  NoRealRootsForFX f → NoRealRootsForFX (f ∘ f) := by
  sorry

#check no_real_roots_for_ff

end NUMINAMATH_CALUDE_no_real_roots_for_ff_l2664_266433


namespace NUMINAMATH_CALUDE_machine_purchase_price_l2664_266447

/-- Given a machine with specified costs and selling price, calculates the original purchase price. -/
theorem machine_purchase_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percentage : ℚ) (selling_price : ℕ) : 
  repair_cost = 5000 →
  transport_cost = 1000 →
  profit_percentage = 50 / 100 →
  selling_price = 25500 →
  ∃ (purchase_price : ℕ), 
    (purchase_price : ℚ) + repair_cost + transport_cost = 
      selling_price / (1 + profit_percentage) ∧
    purchase_price = 11000 :=
by sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l2664_266447


namespace NUMINAMATH_CALUDE_inequality_proof_l2664_266421

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2664_266421


namespace NUMINAMATH_CALUDE_quadratic_properties_l2664_266479

/-- A quadratic function passing through (3, -1) -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 4

theorem quadratic_properties (a b : ℝ) (h_a : a ≠ 0) :
  let f := quadratic_function a b
  -- The function passes through (3, -1)
  (f 3 = -1) →
  -- (2, 2-2a) does not lie on the graph
  (f 2 ≠ 2 - 2*a) ∧
  -- When the graph intersects the x-axis at only one point
  (∃ x : ℝ, (f x = 0 ∧ ∀ y : ℝ, f y = 0 → y = x)) →
  -- The function is either y = -x^2 + 4x - 4 or y = -1/9x^2 + 4/3x - 4
  ((a = -1 ∧ b = 4) ∨ (a = -1/9 ∧ b = 4/3)) ∧
  -- When the graph passes through points (x₁, y₁) and (x₂, y₂) with x₁ < x₂ ≤ 2/3 and y₁ > y₂
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → x₂ ≤ 2/3 → f x₁ = y₁ → f x₂ = y₂ → y₁ > y₂ → a ≥ 3/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2664_266479


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2664_266405

/-- The function R(x) representing the sum of fractions --/
noncomputable def R (a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ) : ℝ :=
  a₁ / (x^2 + 1) + a₂ / (x^2 + 2) + a₃ / (x^2 + 3) + a₄ / (x^2 + 4) + a₅ / (x^2 + 5)

/-- The theorem statement --/
theorem sum_of_fractions (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : R a₁ a₂ a₃ a₄ a₅ 1 = 1)
  (h₂ : R a₁ a₂ a₃ a₄ a₅ 2 = 1/4)
  (h₃ : R a₁ a₂ a₃ a₄ a₅ 3 = 1/9)
  (h₄ : R a₁ a₂ a₃ a₄ a₅ 4 = 1/16)
  (h₅ : R a₁ a₂ a₃ a₄ a₅ 5 = 1/25) :
  a₁/37 + a₂/38 + a₃/39 + a₄/40 + a₅/41 = 187465/6744582 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2664_266405


namespace NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l2664_266460

theorem quadratic_equation_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_solutions_l2664_266460


namespace NUMINAMATH_CALUDE_matrix_power_4_l2664_266448

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_4 :
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l2664_266448


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2664_266452

theorem vector_addition_scalar_multiplication (a b : ℝ × ℝ) :
  a = (2, 1) → b = (1, 5) → (2 • a + b) = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2664_266452


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l2664_266415

/-- Proves that if 90% of a farmer's land is cleared, and 20% of the cleared land
    is planted with tomatoes covering 360 acres, then the total land owned by the
    farmer is 2000 acres. -/
theorem farmer_land_calculation (total_land : ℝ) (cleared_land : ℝ) (tomato_land : ℝ) :
  cleared_land = 0.9 * total_land →
  tomato_land = 0.2 * cleared_land →
  tomato_land = 360 →
  total_land = 2000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l2664_266415


namespace NUMINAMATH_CALUDE_function_zero_l2664_266425

theorem function_zero (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f x) 
  (h2 : ∀ x, f (-x) = f x) : 
  ∀ x, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_function_zero_l2664_266425


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2664_266474

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 4^5 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2664_266474


namespace NUMINAMATH_CALUDE_books_from_second_shop_l2664_266441

theorem books_from_second_shop 
  (first_shop_books : ℕ) 
  (first_shop_cost : ℕ) 
  (second_shop_cost : ℕ) 
  (average_price : ℚ) : ℕ :=
  by
    have h1 : first_shop_books = 40 := by sorry
    have h2 : first_shop_cost = 600 := by sorry
    have h3 : second_shop_cost = 240 := by sorry
    have h4 : average_price = 14 := by sorry
    
    -- The number of books from the second shop
    let second_shop_books : ℕ := 20
    
    -- Prove that this satisfies the conditions
    sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l2664_266441


namespace NUMINAMATH_CALUDE_scooter_final_price_l2664_266477

/-- The final sale price of a scooter after two consecutive discounts -/
theorem scooter_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 150 ∧ discount1 = 0.4 ∧ discount2 = 0.35 →
  initial_price * (1 - discount1) * (1 - discount2) = 58.50 := by
sorry

end NUMINAMATH_CALUDE_scooter_final_price_l2664_266477


namespace NUMINAMATH_CALUDE_color_drawing_percentage_increase_l2664_266482

def black_and_white_cost : ℝ := 160
def color_cost : ℝ := 240

theorem color_drawing_percentage_increase : 
  (color_cost - black_and_white_cost) / black_and_white_cost * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_color_drawing_percentage_increase_l2664_266482


namespace NUMINAMATH_CALUDE_bus_probability_l2664_266440

theorem bus_probability (p3 p6 : ℝ) (h1 : p3 = 0.20) (h2 : p6 = 0.60) :
  p3 + p6 = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_bus_probability_l2664_266440


namespace NUMINAMATH_CALUDE_quartic_equation_minimum_l2664_266431

theorem quartic_equation_minimum (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  a^2 + b^2 ≥ 4/5 := by
sorry

end NUMINAMATH_CALUDE_quartic_equation_minimum_l2664_266431


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2664_266418

theorem complex_equation_solution (x y : ℝ) : 
  (2*x - 1 : ℂ) + (y + 1 : ℂ) * I = (x - y : ℂ) - (x + y : ℂ) * I → x = 3 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2664_266418


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_is_eight_fifteenths_l2664_266435

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

def probability_one_white_one_black : ℚ := (white_balls * black_balls) / (total_balls.choose 2)

theorem probability_one_white_one_black_is_eight_fifteenths :
  probability_one_white_one_black = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_is_eight_fifteenths_l2664_266435


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2664_266419

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 - x + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2664_266419


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2664_266498

theorem simplify_sqrt_expression (x : ℝ) (h : x < 1) :
  (x - 1) * Real.sqrt (-1 / (x - 1)) = -Real.sqrt (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2664_266498


namespace NUMINAMATH_CALUDE_pie_selection_theorem_l2664_266492

/-- Represents the types of pie packets -/
inductive PiePacket
  | CabbageCabbage
  | CherryCherry
  | CabbageCherry

/-- Represents the possible fillings of a pie -/
inductive PieFilling
  | Cabbage
  | Cherry

/-- Represents the state of a pie -/
inductive PieState
  | Whole
  | Broken

/-- Represents a strategy for selecting a pie -/
def Strategy := PiePacket → PieFilling → PieState

/-- The probability of giving a whole cherry pie given a strategy -/
def probability_whole_cherry (s : Strategy) : ℚ := sorry

/-- The simple strategy described in part (a) -/
def simple_strategy : Strategy := sorry

/-- The improved strategy described in part (b) -/
def improved_strategy : Strategy := sorry

theorem pie_selection_theorem :
  (probability_whole_cherry simple_strategy = 2/3) ∧
  (probability_whole_cherry improved_strategy > 2/3) := by
  sorry


end NUMINAMATH_CALUDE_pie_selection_theorem_l2664_266492


namespace NUMINAMATH_CALUDE_total_workers_count_l2664_266486

def num_other_workers : ℕ := 5

def probability_jack_and_jill : ℚ := 1 / 21

theorem total_workers_count (num_selected : ℕ) (h1 : num_selected = 2) :
  ∃ (total_workers : ℕ),
    total_workers = num_other_workers + 2 ∧
    probability_jack_and_jill = 1 / (total_workers.choose num_selected) :=
by sorry

end NUMINAMATH_CALUDE_total_workers_count_l2664_266486


namespace NUMINAMATH_CALUDE_tank_capacity_l2664_266414

theorem tank_capacity (fill_time_A fill_time_B drain_rate_C combined_fill_time : ℝ) 
  (h1 : fill_time_A = 12)
  (h2 : fill_time_B = 20)
  (h3 : drain_rate_C = 45)
  (h4 : combined_fill_time = 15) :
  ∃ V : ℝ, V = 675 ∧ 
    (V / fill_time_A + V / fill_time_B - drain_rate_C = V / combined_fill_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2664_266414


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2664_266463

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (|x| - 3) / (x + 3) = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2664_266463


namespace NUMINAMATH_CALUDE_combined_swim_time_l2664_266499

def freestyle_time : ℕ := 48

def backstroke_time : ℕ := freestyle_time + 4

def butterfly_time : ℕ := backstroke_time + 3

def breaststroke_time : ℕ := butterfly_time + 2

def total_time : ℕ := freestyle_time + backstroke_time + butterfly_time + breaststroke_time

theorem combined_swim_time : total_time = 212 := by
  sorry

end NUMINAMATH_CALUDE_combined_swim_time_l2664_266499


namespace NUMINAMATH_CALUDE_problem_solution_l2664_266427

theorem problem_solution : 
  (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1/2) = 2 * Real.sqrt 2) ∧
  (Real.sqrt 12 - 9 * Real.sqrt (1/3) + |2 - Real.sqrt 3| = 2 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2664_266427


namespace NUMINAMATH_CALUDE_wario_expected_wide_right_misses_l2664_266420

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

end NUMINAMATH_CALUDE_wario_expected_wide_right_misses_l2664_266420


namespace NUMINAMATH_CALUDE_zero_point_existence_l2664_266449

theorem zero_point_existence (a : ℝ) :
  a < -2 → 
  (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
  (¬ ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2) :=
sorry

end NUMINAMATH_CALUDE_zero_point_existence_l2664_266449


namespace NUMINAMATH_CALUDE_john_memory_card_cost_l2664_266454

/-- The total cost of memory cards for storing John's pictures -/
theorem john_memory_card_cost : 
  let pictures_per_day : ℕ := 25
  let years : ℕ := 6
  let images_per_card : ℕ := 40
  let cost_per_card : ℕ := 75
  let days_per_year : ℕ := 365

  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := (total_pictures + images_per_card - 1) / images_per_card
  let total_cost : ℕ := cards_needed * cost_per_card

  total_cost = 102675 := by
  sorry


end NUMINAMATH_CALUDE_john_memory_card_cost_l2664_266454


namespace NUMINAMATH_CALUDE_emily_trivia_score_l2664_266444

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l2664_266444


namespace NUMINAMATH_CALUDE_point_division_theorem_l2664_266422

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points A, B, and P
variable (A B P : V)

-- Define the condition that P is on the line segment AB with the given ratio
def on_segment_with_ratio (A B P : V) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B ∧ t = 5 / 8

-- Theorem statement
theorem point_division_theorem (h : on_segment_with_ratio A B P) :
  P = (3 / 8) • A + (5 / 8) • B := by sorry

end NUMINAMATH_CALUDE_point_division_theorem_l2664_266422


namespace NUMINAMATH_CALUDE_cousins_arrangement_l2664_266488

/-- The number of ways to arrange n indistinguishable objects into k distinct boxes -/
def arrange (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to arrange -/
def num_cousins : ℕ := 5

/-- The number of arrangements of 5 cousins in 4 rooms is 76 -/
theorem cousins_arrangement : arrange num_cousins num_rooms = 76 := by sorry

end NUMINAMATH_CALUDE_cousins_arrangement_l2664_266488


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2664_266429

/-- Given a point P and a line L, this function returns the point symmetric to P with respect to L -/
def symmetricPoint (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x + y = 1 -/
def lineXPlusYEq1 (p : ℝ × ℝ) : Prop := p.1 + p.2 = 1

theorem symmetry_of_point :
  symmetricPoint (2, 5) lineXPlusYEq1 = (-4, -1) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2664_266429


namespace NUMINAMATH_CALUDE_darius_drove_679_miles_l2664_266475

/-- The number of miles Julia drove -/
def julia_miles : ℕ := 998

/-- The total number of miles Darius and Julia drove -/
def total_miles : ℕ := 1677

/-- The number of miles Darius drove -/
def darius_miles : ℕ := total_miles - julia_miles

theorem darius_drove_679_miles : darius_miles = 679 := by sorry

end NUMINAMATH_CALUDE_darius_drove_679_miles_l2664_266475


namespace NUMINAMATH_CALUDE_discount_equation_l2664_266443

/-- Represents the discount scenario for a clothing item -/
structure DiscountScenario where
  original_price : ℝ
  final_price : ℝ
  discount_percentage : ℝ

/-- Theorem stating the relationship between original price, discount, and final price -/
theorem discount_equation (scenario : DiscountScenario) 
  (h1 : scenario.original_price = 280)
  (h2 : scenario.final_price = 177)
  (h3 : scenario.discount_percentage ≥ 0)
  (h4 : scenario.discount_percentage < 1) :
  scenario.original_price * (1 - scenario.discount_percentage)^2 = scenario.final_price := by
  sorry

#check discount_equation

end NUMINAMATH_CALUDE_discount_equation_l2664_266443


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2664_266497

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < (1/2) → S = Set.Ioo a (1 - a)) ∧
  (a = (1/2) → S = ∅) ∧
  ((1/2) < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2664_266497


namespace NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l2664_266469

theorem reciprocal_of_sqrt_two : Real.sqrt 2 * (Real.sqrt 2 / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sqrt_two_l2664_266469


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l2664_266438

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 → 3 * x - y - 2 = 0 → 
    (a * x + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0) → 
    ((-a/2) * 3 = -1)) → 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l2664_266438


namespace NUMINAMATH_CALUDE_thirty_six_in_binary_l2664_266472

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The binary representation of 36 -/
def binary_36 : List Bool := [false, false, true, false, false, true]

/-- Theorem stating that the binary representation of 36 is 100100₂ -/
theorem thirty_six_in_binary :
  to_binary 36 = binary_36 := by sorry

end NUMINAMATH_CALUDE_thirty_six_in_binary_l2664_266472


namespace NUMINAMATH_CALUDE_average_mark_proof_l2664_266426

/-- Given an examination with 50 candidates and a total of 2000 marks,
    prove that the average mark obtained by each candidate is 40. -/
theorem average_mark_proof (candidates : ℕ) (total_marks : ℕ) :
  candidates = 50 →
  total_marks = 2000 →
  (total_marks : ℚ) / (candidates : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_proof_l2664_266426


namespace NUMINAMATH_CALUDE_overlapping_segments_length_l2664_266401

/-- Given a set of overlapping segments with known total length and span, 
    this theorem proves the length of each overlapping part. -/
theorem overlapping_segments_length 
  (total_length : ℝ) 
  (edge_to_edge : ℝ) 
  (num_overlaps : ℕ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge = 83) 
  (h3 : num_overlaps = 6) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

#check overlapping_segments_length

end NUMINAMATH_CALUDE_overlapping_segments_length_l2664_266401


namespace NUMINAMATH_CALUDE_last_four_average_l2664_266412

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 :=
by sorry

end NUMINAMATH_CALUDE_last_four_average_l2664_266412


namespace NUMINAMATH_CALUDE_k_of_five_eq_eight_point_five_l2664_266467

noncomputable def h (x : ℝ) : ℝ := 5 / (3 - x)

noncomputable def h_inverse (x : ℝ) : ℝ := 3 - 5 / x

noncomputable def k (x : ℝ) : ℝ := 1 / (h_inverse x) + 8

theorem k_of_five_eq_eight_point_five : k 5 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_k_of_five_eq_eight_point_five_l2664_266467


namespace NUMINAMATH_CALUDE_blue_whale_tongue_weight_l2664_266445

/-- The weight of an adult blue whale's tongue in pounds -/
def tongue_weight : ℕ := 6000

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2000

/-- The weight of an adult blue whale's tongue in tons -/
def tongue_weight_in_tons : ℚ := tongue_weight / pounds_per_ton

theorem blue_whale_tongue_weight : tongue_weight_in_tons = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_whale_tongue_weight_l2664_266445


namespace NUMINAMATH_CALUDE_burger_cost_proof_l2664_266439

/-- The cost of a burger at McDonald's -/
def burger_cost : ℝ := 5

/-- The cost of one pack of fries -/
def fries_cost : ℝ := 2

/-- The cost of a salad -/
def salad_cost : ℝ := 3 * fries_cost

/-- The total cost of the meal -/
def total_cost : ℝ := 15

theorem burger_cost_proof :
  burger_cost + 2 * fries_cost + salad_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l2664_266439


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l2664_266455

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) :
  (∀ x, p x = 1.8 * x^2 - 5.4 * x - 32.4) →
  p (-3) = 0 ∧ p 6 = 0 ∧ p 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l2664_266455


namespace NUMINAMATH_CALUDE_joint_account_final_amount_l2664_266407

/-- Calculates the final amount in a joint account after one year with changing interest rates and tax --/
theorem joint_account_final_amount 
  (deposit_lopez : ℝ) 
  (deposit_johnson : ℝ) 
  (initial_rate : ℝ) 
  (changed_rate : ℝ) 
  (tax_rate : ℝ) 
  (h1 : deposit_lopez = 100)
  (h2 : deposit_johnson = 150)
  (h3 : initial_rate = 0.20)
  (h4 : changed_rate = 0.18)
  (h5 : tax_rate = 0.05) : 
  ∃ (final_amount : ℝ), abs (final_amount - 272.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_joint_account_final_amount_l2664_266407


namespace NUMINAMATH_CALUDE_divide_powers_of_nineteen_l2664_266430

theorem divide_powers_of_nineteen : 19^12 / 19^8 = 130321 := by sorry

end NUMINAMATH_CALUDE_divide_powers_of_nineteen_l2664_266430


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2664_266462

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 7 →
    length = 3 * width →
    width = 2 * r →
    length * width = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2664_266462


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2664_266466

theorem consecutive_numbers_theorem (a b c d e : ℕ) : 
  (a > b) ∧ (b > c) ∧ (c > d) ∧ (d > e) ∧  -- Descending order
  (a - b = 1) ∧ (b - c = 1) ∧ (c - d = 1) ∧ (d - e = 1) ∧  -- Consecutive numbers
  ((a + b + c) / 3 = 45) ∧  -- Average of first three
  ((c + d + e) / 3 = 43) →  -- Average of last three
  c = 44 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l2664_266466


namespace NUMINAMATH_CALUDE_pennies_indeterminate_l2664_266451

/-- Represents the number of coins Sandy has -/
structure SandyCoins where
  pennies : ℕ
  nickels : ℕ

/-- Represents the state of Sandy's coins before and after her dad's borrowing -/
structure SandyState where
  initial : SandyCoins
  borrowed_nickels : ℕ
  remaining : SandyCoins

/-- Defines the conditions of the problem -/
def valid_state (s : SandyState) : Prop :=
  s.initial.nickels = 31 ∧
  s.borrowed_nickels = 20 ∧
  s.remaining.nickels = 11 ∧
  s.initial.nickels = s.remaining.nickels + s.borrowed_nickels

/-- Theorem stating that the initial number of pennies cannot be determined -/
theorem pennies_indeterminate (s1 s2 : SandyState) :
  valid_state s1 → valid_state s2 → s1.initial.pennies ≠ s2.initial.pennies → True := by
  sorry

end NUMINAMATH_CALUDE_pennies_indeterminate_l2664_266451


namespace NUMINAMATH_CALUDE_insurance_agents_count_l2664_266434

/-- The number of claims Jan can handle -/
def jan_claims : ℕ := 20

/-- The number of claims John can handle -/
def john_claims : ℕ := jan_claims + jan_claims * 30 / 100

/-- The number of claims Missy can handle -/
def missy_claims : ℕ := john_claims + 15

/-- The total number of agents -/
def num_agents : ℕ := 3

theorem insurance_agents_count :
  missy_claims = 41 → num_agents = 3 := by
  sorry

end NUMINAMATH_CALUDE_insurance_agents_count_l2664_266434


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2664_266483

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2664_266483


namespace NUMINAMATH_CALUDE_frustum_height_calc_l2664_266413

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  height : ℝ              -- Total height of the pyramid
  cut_height : ℝ          -- Height of the cut part
  area_ratio : ℝ          -- Ratio of upper to lower base areas

/-- The height of the frustum in a cut pyramid -/
def frustum_height (p : CutPyramid) : ℝ := p.height - p.cut_height

/-- Theorem stating the height of the frustum given specific conditions -/
theorem frustum_height_calc (p : CutPyramid) 
  (h1 : p.area_ratio = 1 / 4)
  (h2 : p.cut_height = 3) :
  frustum_height p = 3 := by
  sorry

#check frustum_height_calc

end NUMINAMATH_CALUDE_frustum_height_calc_l2664_266413


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l2664_266476

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

end NUMINAMATH_CALUDE_parabola_triangle_area_l2664_266476


namespace NUMINAMATH_CALUDE_smallest_wonder_number_l2664_266436

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for a "wonder number" -/
def is_wonder_number (n : ℕ) : Prop :=
  (digit_sum n = digit_sum (3 * n)) ∧ 
  (digit_sum n ≠ digit_sum (2 * n))

/-- Theorem stating that 144 is the smallest wonder number -/
theorem smallest_wonder_number : 
  is_wonder_number 144 ∧ ∀ n < 144, ¬is_wonder_number n := by sorry

end NUMINAMATH_CALUDE_smallest_wonder_number_l2664_266436


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2664_266496

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2664_266496


namespace NUMINAMATH_CALUDE_smallest_angle_solution_l2664_266471

theorem smallest_angle_solution : 
  ∃ x : ℝ, x > 0 ∧ 
    6 * Real.sin x * (Real.cos x)^3 - 6 * (Real.sin x)^3 * Real.cos x = 3 * Real.sqrt 3 / 2 ∧
    ∀ y : ℝ, y > 0 → 
      6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 3 / 2 → 
      x ≤ y ∧
    x = π / 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_solution_l2664_266471


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_l2664_266437

theorem mashed_potatoes_count (tomatoes bacon : ℕ) 
  (h1 : tomatoes = 79)
  (h2 : bacon = 467) : 
  ∃ mashed_potatoes : ℕ, mashed_potatoes = tomatoes + 65 ∧ mashed_potatoes = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_l2664_266437


namespace NUMINAMATH_CALUDE_tire_circumference_l2664_266408

/-- Calculates the circumference of a car tire given the car's speed and tire rotation rate. -/
theorem tire_circumference (speed : ℝ) (rotations : ℝ) : 
  speed = 168 → rotations = 400 → (speed * 1000 / 60) / rotations = 7 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l2664_266408


namespace NUMINAMATH_CALUDE_calculation_comparison_l2664_266457

theorem calculation_comparison : 
  (3.04 / 0.25 > 1) ∧ (1.01 * 0.99 < 1) ∧ (0.15 / 0.25 < 1) := by
  sorry

end NUMINAMATH_CALUDE_calculation_comparison_l2664_266457


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l2664_266409

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x + 12)^2 + y^2 = 225

-- Define the shortest tangent line segment
def shortest_tangent (R S : ℝ × ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    R = (x₁, y₁) ∧ S = (x₂, y₂) ∧
    C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
    ∀ (T U : ℝ × ℝ),
      C₁ T.1 T.2 → C₂ U.1 U.2 →
      Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) ≥ 
      Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∀ (R S : ℝ × ℝ),
    shortest_tangent R S →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 
    Real.sqrt (16 - (60/19)^2) + Real.sqrt (225 - (225/19)^2) := by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l2664_266409


namespace NUMINAMATH_CALUDE_cubic_sequence_problem_l2664_266494

theorem cubic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 8*y₂ + 27*y₃ + 64*y₄ + 125*y₅ = 7)
  (eq2 : 8*y₁ + 27*y₂ + 64*y₃ + 125*y₄ + 216*y₅ = 100)
  (eq3 : 27*y₁ + 64*y₂ + 125*y₃ + 216*y₄ + 343*y₅ = 1000) :
  64*y₁ + 125*y₂ + 216*y₃ + 343*y₄ + 512*y₅ = -5999 := by
sorry

end NUMINAMATH_CALUDE_cubic_sequence_problem_l2664_266494


namespace NUMINAMATH_CALUDE_part_one_part_two_l2664_266432

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one :
  let a : ℝ := 1
  let b : ℝ := 1
  {x : ℝ | f x a b > 8} = {x : ℝ | x < -1 ∨ x > 1.5} := by sorry

-- Part II
theorem part_two :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x : ℝ, f x a b ≥ 5) →
  (∃ x : ℝ, f x a b = 5) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 →
    (∀ x : ℝ, f x a' b' ≥ 5) →
    (∃ x : ℝ, f x a' b' = 5) →
    1/a + 1/b ≤ 1/a' + 1/b') →
  1/a + 1/b = (3 + 2 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2664_266432


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l2664_266428

theorem quadratic_radical_equality (x y : ℚ) : 
  (x - y*x + y - 1 = 2 ∧ x + y - 1 = 3*x + 2*y - 4) → x*y = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l2664_266428


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2664_266461

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2664_266461


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2664_266487

noncomputable def f (t : ℝ) : ℝ := Real.exp t + 1

noncomputable def g (t : ℝ) : ℝ := 2 * t - 1

theorem min_distance_between_curves :
  ∃ (t_min : ℝ), ∀ (t : ℝ), |f t - g t| ≥ |f t_min - g t_min| ∧ 
  |f t_min - g t_min| = 4 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2664_266487


namespace NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l2664_266484

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily + baby_daily) * 7

/-- Theorem: The total amount of bamboo eaten by an adult panda and a baby panda in a week -/
theorem pandas_weekly_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end NUMINAMATH_CALUDE_pandas_weekly_bamboo_consumption_l2664_266484


namespace NUMINAMATH_CALUDE_parabola_vertex_l2664_266406

/-- The vertex of the parabola y = 5(x-2)^2 + 6 is at the point (2,6) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 5*(x-2)^2 + 6 → (2, 6) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2664_266406


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l2664_266478

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
def p : ℂ := sorry
def q : ℂ := sorry
def r : ℂ := sorry
def s : ℂ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f p = 0 ∧ f q = 0 ∧ f r = 0 ∧ f s = 0 →
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l2664_266478


namespace NUMINAMATH_CALUDE_range_of_f_on_large_interval_l2664_266402

/-- A function with period 1 --/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = g x

/-- The function f defined as f(x) = x + g(x) --/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := x + g x

/-- The range of a function on an interval --/
def range_on (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_f_on_large_interval
    (g : ℝ → ℝ)
    (h_periodic : periodic_function g)
    (h_range : range_on (f g) 3 4 = Set.Icc (-2) 5) :
    range_on (f g) (-10) 10 = Set.Icc (-15) 11 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_on_large_interval_l2664_266402


namespace NUMINAMATH_CALUDE_probability_sphere_in_cube_l2664_266465

/-- The probability of a point (x, y, z) satisfying x^2 + y^2 + z^2 ≤ 4,
    given that -2 ≤ x ≤ 2, -2 ≤ y ≤ 2, and -2 ≤ z ≤ 2 -/
theorem probability_sphere_in_cube : 
  let cube_volume := (2 - (-2))^3
  let sphere_volume := (4/3) * Real.pi * 2^3
  sphere_volume / cube_volume = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_probability_sphere_in_cube_l2664_266465


namespace NUMINAMATH_CALUDE_no_inscribed_sphere_l2664_266403

structure Polyhedron where
  faces : ℕ
  paintedFaces : ℕ
  convex : Bool
  noAdjacentPainted : Bool

def canInscribeSphere (p : Polyhedron) : Prop :=
  sorry

theorem no_inscribed_sphere (p : Polyhedron) 
  (h_convex : p.convex = true)
  (h_painted : p.paintedFaces > p.faces / 2)
  (h_noAdjacent : p.noAdjacentPainted = true) :
  ¬(canInscribeSphere p) := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_sphere_l2664_266403


namespace NUMINAMATH_CALUDE_shares_owned_problem_solution_l2664_266493

/-- A function that calculates the dividend per share based on actual earnings --/
def dividend_per_share (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let base_dividend := expected_earnings / 2
  let additional_earnings := max (actual_earnings - expected_earnings) 0
  let additional_dividend := (additional_earnings / (1/10)) * (4/100)
  base_dividend + additional_dividend

theorem shares_owned (expected_earnings actual_earnings total_dividend : ℚ) : ℚ :=
  total_dividend / (dividend_per_share expected_earnings actual_earnings)

/-- Proves the number of shares owned given the problem conditions --/
theorem problem_solution :
  let expected_earnings : ℚ := 80/100
  let actual_earnings : ℚ := 110/100
  let total_dividend : ℚ := 260
  shares_owned expected_earnings actual_earnings total_dividend = 500 := by
  sorry

end NUMINAMATH_CALUDE_shares_owned_problem_solution_l2664_266493


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2664_266473

theorem quadratic_root_condition (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (∀ x : ℝ, x^2 + p*x + p - 1 = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁^2 + x₁^3 = -(x₂^2 + x₂^3)) ↔ 
  p = 1 ∨ p = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2664_266473


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2664_266485

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 400) (h_goat : goat_value = 250) :
  ∃ (D : ℕ), D > 0 ∧ 
  (∃ (p g : ℤ), D = pig_value * p + goat_value * g) ∧
  (∀ (D' : ℕ), D' > 0 → 
    (∃ (p' g' : ℤ), D' = pig_value * p' + goat_value * g') → 
    D ≤ D') :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2664_266485


namespace NUMINAMATH_CALUDE_combination_distinctness_and_divisor_count_l2664_266481

theorem combination_distinctness_and_divisor_count (n : ℕ) (hn : n > 3) :
  -- Part (a)
  (∀ x y z : ℕ, x > n / 2 → y > n / 2 → z > n / 2 → x < y → y < z → z ≤ n →
    (let exprs := [x + y + z, x + y * z, x * y + z, y + z * x, (x + y) * z, (z + x) * y, (y + z) * x, x * y * z]
     exprs.Pairwise (·≠·))) ∧
  -- Part (b)
  (∀ p : ℕ, Nat.Prime p → p ≤ Real.sqrt n →
    (Finset.filter (fun i => i > 1 ∧ (p - 1) % i = 0) (Finset.range (p - 1))).card =
    (Finset.filter (fun pair : ℕ × ℕ =>
      let (y, z) := pair
      p < y ∧ y < z ∧ z ≤ n ∧
      ¬(let exprs := [p + y + z, p + y * z, p * y + z, y + z * p, (p + y) * z, (z + p) * y, (y + z) * p, p * y * z]
        exprs.Pairwise (·≠·)))
     (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card) :=
by sorry

end NUMINAMATH_CALUDE_combination_distinctness_and_divisor_count_l2664_266481


namespace NUMINAMATH_CALUDE_camila_weeks_to_match_steven_l2664_266489

-- Define the initial number of hikes for Camila
def camila_initial_hikes : ℕ := 7

-- Define Amanda's hikes in terms of Camila's
def amanda_hikes : ℕ := 8 * camila_initial_hikes

-- Define Steven's hikes in terms of Amanda's
def steven_hikes : ℕ := amanda_hikes + 15

-- Define Camila's planned hikes per week
def camila_weekly_hikes : ℕ := 4

-- Theorem to prove
theorem camila_weeks_to_match_steven :
  (steven_hikes - camila_initial_hikes) / camila_weekly_hikes = 16 := by
  sorry

end NUMINAMATH_CALUDE_camila_weeks_to_match_steven_l2664_266489


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_left_l2664_266470

/-- Given a series with a total number of books and a number of books read,
    calculate the number of books left to read. -/
def booksLeftToRead (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series with 19 books, if 4 books have been read,
    then the number of books left to read is 15. -/
theorem crazy_silly_school_books_left :
  booksLeftToRead 19 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_left_l2664_266470


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2664_266453

theorem adult_tickets_sold (total_tickets : ℕ) (student_ratio : ℕ) : 
  total_tickets = 600 →
  student_ratio = 3 →
  (student_ratio * (total_tickets / (student_ratio + 1))) + (total_tickets / (student_ratio + 1)) = total_tickets →
  (total_tickets / (student_ratio + 1)) = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2664_266453


namespace NUMINAMATH_CALUDE_existence_of_relatively_prime_divisible_combination_l2664_266423

theorem existence_of_relatively_prime_divisible_combination (a b p : ℤ) :
  ∃ k l : ℤ, Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_relatively_prime_divisible_combination_l2664_266423


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l2664_266424

theorem weekly_allowance_calculation (weekly_allowance : ℚ) : 
  (4 * weekly_allowance / 2 * 3 / 4 = 15) → weekly_allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l2664_266424


namespace NUMINAMATH_CALUDE_multiply_1307_by_1307_l2664_266450

theorem multiply_1307_by_1307 : 1307 * 1307 = 1709249 := by
  sorry

end NUMINAMATH_CALUDE_multiply_1307_by_1307_l2664_266450


namespace NUMINAMATH_CALUDE_instantaneous_velocity_zero_l2664_266442

/-- The motion law of an object -/
def S (t : ℝ) : ℝ := t^3 - 6*t^2 + 5

/-- The instantaneous velocity of the object -/
def V (t : ℝ) : ℝ := 3*t^2 - 12*t

theorem instantaneous_velocity_zero (t : ℝ) (h : t > 0) :
  V t = 0 → t = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_zero_l2664_266442


namespace NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2664_266416

-- Define a die as having 6 faces
def die : Finset ℕ := Finset.range 6

-- Define the total number of outcomes when rolling three dice
def total_outcomes : ℕ := die.card ^ 3

-- Define the favorable outcomes (sum of 10)
def favorable_outcomes : ℕ := 27

-- Theorem statement
theorem probability_sum_10_three_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_three_dice_l2664_266416


namespace NUMINAMATH_CALUDE_power_product_equals_sixteen_l2664_266456

theorem power_product_equals_sixteen (m n : ℤ) (h : 2*m + 3*n - 4 = 0) : 
  (4:ℝ)^m * (8:ℝ)^n = 16 := by
sorry

end NUMINAMATH_CALUDE_power_product_equals_sixteen_l2664_266456


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2664_266464

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2^(1/4))
    (h₂ : a₂ = 2^(1/6)) (h₃ : a₃ = 2^(1/12)) :
  let r := a₂ / a₁
  a₃ * r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2664_266464


namespace NUMINAMATH_CALUDE_min_lines_theorem_l2664_266491

/-- A plane -/
structure Plane where

/-- A point in a plane -/
structure Point (α : Plane) where

/-- A line in a plane -/
structure Line (α : Plane) where

/-- A ray in a plane -/
structure Ray (α : Plane) where

/-- Predicate for a line not passing through a point -/
def LineNotThroughPoint (α : Plane) (l : Line α) (P : Point α) : Prop :=
  sorry

/-- Predicate for a ray intersecting a line -/
def RayIntersectsLine (α : Plane) (r : Ray α) (l : Line α) : Prop :=
  sorry

/-- The minimum number of lines theorem -/
theorem min_lines_theorem (α : Plane) (P : Point α) (k : ℕ) (h : k > 0) :
  ∃ (n : ℕ),
    (∀ (m : ℕ),
      (∃ (lines : Fin m → Line α),
        (∀ i, LineNotThroughPoint α (lines i) P) ∧
        (∀ r : Ray α, ∃ (S : Finset (Fin m)), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i)))
      → m ≥ n) ∧
    (∃ (lines : Fin (2 * k + 1) → Line α),
      (∀ i, LineNotThroughPoint α (lines i) P) ∧
      (∀ r : Ray α, ∃ (S : Finset (Fin (2 * k + 1))), S.card ≥ k ∧ ∀ i ∈ S, RayIntersectsLine α r (lines i))) :=
  sorry

end NUMINAMATH_CALUDE_min_lines_theorem_l2664_266491


namespace NUMINAMATH_CALUDE_inequality_proof_l2664_266458

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1)) + (1 / (b + c + 1)) + (1 / (a + c + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2664_266458


namespace NUMINAMATH_CALUDE_initial_juice_percentage_l2664_266446

/-- Proves that the initial percentage of pure fruit juice in a 2-liter mixture is 10% -/
theorem initial_juice_percentage :
  let initial_volume : ℝ := 2
  let added_juice : ℝ := 0.4
  let final_percentage : ℝ := 25
  let final_volume : ℝ := initial_volume + added_juice
  ∀ initial_percentage : ℝ,
    (initial_percentage / 100 * initial_volume + added_juice) / final_volume * 100 = final_percentage →
    initial_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_juice_percentage_l2664_266446
