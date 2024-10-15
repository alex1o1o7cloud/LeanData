import Mathlib

namespace NUMINAMATH_CALUDE_halloween_candy_problem_l3293_329391

/-- Given the initial candy counts for Katie and her sister, and the number of pieces eaten,
    calculate the remaining candy count. -/
def remaining_candy (katie_candy : ℕ) (sister_candy : ℕ) (eaten : ℕ) : ℕ :=
  katie_candy + sister_candy - eaten

/-- Theorem stating that for the given problem, the remaining candy count is 7. -/
theorem halloween_candy_problem :
  remaining_candy 10 6 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l3293_329391


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_quadratic_roots_l3293_329306

theorem triangle_perimeter_from_quadratic_roots :
  ∀ a b c : ℝ,
  (a^2 - 7*a + 10 = 0) →
  (b^2 - 7*b + 10 = 0) →
  (c^2 - 7*c + 10 = 0) →
  (a + b > c) → (b + c > a) → (c + a > b) →
  (a + b + c = 12 ∨ a + b + c = 6 ∨ a + b + c = 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_quadratic_roots_l3293_329306


namespace NUMINAMATH_CALUDE_no_real_roots_l3293_329387

theorem no_real_roots : ∀ x : ℝ, x^2 - 2*x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3293_329387


namespace NUMINAMATH_CALUDE_average_race_time_l3293_329308

/-- Calculates the average time in seconds for two racers to complete a block,
    given the time for one racer to complete the full block and the time for
    the other racer to complete half the block. -/
theorem average_race_time (carlos_time : ℝ) (diego_half_time : ℝ) : 
  carlos_time = 3 →
  diego_half_time = 2.5 →
  (carlos_time + 2 * diego_half_time) / 2 * 60 = 240 := by
  sorry

#check average_race_time

end NUMINAMATH_CALUDE_average_race_time_l3293_329308


namespace NUMINAMATH_CALUDE_sqrt_xy_eq_three_halves_l3293_329358

theorem sqrt_xy_eq_three_halves (x y : ℝ) (h : |2*x + 1| + Real.sqrt (9 + 2*y) = 0) :
  Real.sqrt (x * y) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_xy_eq_three_halves_l3293_329358


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3293_329314

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if 2a_7 - a_8 = 5, then S_11 = 55 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (∀ n, S n = n * (a 1 + a n) / 2) →    -- sum formula
  (2 * a 7 - a 8 = 5) →                 -- given condition
  S 11 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3293_329314


namespace NUMINAMATH_CALUDE_position_vector_coefficients_l3293_329321

/-- Given points A and B, and points P and Q on line segment AB,
    prove that their position vectors have the specified coefficients. -/
theorem position_vector_coefficients
  (A B P Q : ℝ × ℝ) -- A, B, P, Q are points in 2D space
  (h_P : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) -- P is on AB
  (h_Q : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • A + s • B) -- Q is on AB
  (h_P_ratio : (dist A P) / (dist P B) = 3 / 5) -- AP:PB = 3:5
  (h_Q_ratio : (dist A Q) / (dist Q B) = 4 / 3) -- AQ:QB = 4:3
  : (∃ t₁ u₁ : ℝ, P = t₁ • A + u₁ • B ∧ t₁ = 5/8 ∧ u₁ = 3/8) ∧
    (∃ t₂ u₂ : ℝ, Q = t₂ • A + u₂ • B ∧ t₂ = 3/7 ∧ u₂ = 4/7) :=
by sorry

end NUMINAMATH_CALUDE_position_vector_coefficients_l3293_329321


namespace NUMINAMATH_CALUDE_women_at_dance_event_l3293_329352

/-- Represents a dance event with men and women -/
structure DanceEvent where
  men : ℕ
  women : ℕ
  men_dances : ℕ
  women_dances : ℕ

/-- Calculates the total number of dance pairs in the event -/
def total_dance_pairs (event : DanceEvent) : ℕ :=
  event.men * event.men_dances

/-- Theorem: Given the conditions of the dance event, prove that 24 women attended -/
theorem women_at_dance_event (event : DanceEvent) 
  (h1 : event.men_dances = 4)
  (h2 : event.women_dances = 3)
  (h3 : event.men = 18) :
  event.women = 24 := by
  sorry

#check women_at_dance_event

end NUMINAMATH_CALUDE_women_at_dance_event_l3293_329352


namespace NUMINAMATH_CALUDE_triangle_area_from_inradius_and_perimeter_l3293_329317

/-- Given a triangle with angles A and B, perimeter p, and inradius r, 
    proves that the area of the triangle is equal to r * (p / 2) -/
theorem triangle_area_from_inradius_and_perimeter 
  (A B : Real) (p r : Real) (h1 : A = 40) (h2 : B = 60) (h3 : p = 40) (h4 : r = 2.5) : 
  r * (p / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_inradius_and_perimeter_l3293_329317


namespace NUMINAMATH_CALUDE_remainder_sum_modulo_l3293_329359

theorem remainder_sum_modulo (x y : ℤ) 
  (hx : x % 126 = 37) 
  (hy : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_modulo_l3293_329359


namespace NUMINAMATH_CALUDE_power_eleven_mod_120_l3293_329302

theorem power_eleven_mod_120 : 11^2023 % 120 = 11 := by sorry

end NUMINAMATH_CALUDE_power_eleven_mod_120_l3293_329302


namespace NUMINAMATH_CALUDE_tribal_leadership_theorem_l3293_329384

def tribal_leadership_arrangements (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2

theorem tribal_leadership_theorem :
  tribal_leadership_arrangements 13 = 18604800 := by
  sorry

end NUMINAMATH_CALUDE_tribal_leadership_theorem_l3293_329384


namespace NUMINAMATH_CALUDE_max_gcd_sum_1025_l3293_329369

theorem max_gcd_sum_1025 : 
  ∃ (max : ℕ), max > 0 ∧ 
  (∀ a b : ℕ, a > 0 → b > 0 → a + b = 1025 → Nat.gcd a b ≤ max) ∧
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a + b = 1025 ∧ Nat.gcd a b = max) ∧
  max = 205 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_sum_1025_l3293_329369


namespace NUMINAMATH_CALUDE_equivalent_operation_l3293_329367

theorem equivalent_operation (x : ℚ) : x * (4/5) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l3293_329367


namespace NUMINAMATH_CALUDE_total_cotton_needed_l3293_329307

/-- The amount of cotton needed for one tee-shirt in feet -/
def cotton_per_shirt : ℝ := 4

/-- The number of tee-shirts to be made -/
def num_shirts : ℕ := 15

/-- Theorem stating the total amount of cotton needed -/
theorem total_cotton_needed : 
  cotton_per_shirt * (num_shirts : ℝ) = 60 := by sorry

end NUMINAMATH_CALUDE_total_cotton_needed_l3293_329307


namespace NUMINAMATH_CALUDE_mason_car_nuts_l3293_329395

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nuts_in_car (busy_squirrels : ℕ) (busy_nuts_per_day : ℕ) (sleepy_squirrels : ℕ) (sleepy_nuts_per_day : ℕ) (days : ℕ) : ℕ :=
  (busy_squirrels * busy_nuts_per_day + sleepy_squirrels * sleepy_nuts_per_day) * days

/-- Theorem stating the number of nuts in Mason's car -/
theorem mason_car_nuts :
  nuts_in_car 2 30 1 20 40 = 3200 :=
by sorry

end NUMINAMATH_CALUDE_mason_car_nuts_l3293_329395


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l3293_329342

theorem fraction_cube_equality : 
  (81000 : ℝ)^3 / (27000 : ℝ)^3 = 27 :=
by
  have h : (81000 : ℝ) = 3 * 27000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l3293_329342


namespace NUMINAMATH_CALUDE_tickets_to_buy_l3293_329346

/-- The number of additional tickets Zach needs to buy for three rides -/
theorem tickets_to_buy (ferris_wheel_cost roller_coaster_cost log_ride_cost current_tickets : ℕ) 
  (h1 : ferris_wheel_cost = 2)
  (h2 : roller_coaster_cost = 7)
  (h3 : log_ride_cost = 1)
  (h4 : current_tickets = 1) :
  ferris_wheel_cost + roller_coaster_cost + log_ride_cost - current_tickets = 9 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l3293_329346


namespace NUMINAMATH_CALUDE_john_got_36_rolls_l3293_329366

/-- The number of rolls John got given the price and amount spent -/
def rolls_bought (price_per_dozen : ℚ) (amount_spent : ℚ) : ℚ :=
  (amount_spent / price_per_dozen) * 12

/-- Theorem: John got 36 rolls -/
theorem john_got_36_rolls :
  rolls_bought 5 15 = 36 := by
  sorry

end NUMINAMATH_CALUDE_john_got_36_rolls_l3293_329366


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l3293_329381

/-- Calculates the total number of wheels in a parking lot with cars and motorcycles -/
theorem total_wheels_in_parking_lot 
  (num_cars : ℕ) 
  (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) 
  (wheels_per_motorcycle : ℕ) 
  (h1 : num_cars = 19) 
  (h2 : num_motorcycles = 11) 
  (h3 : wheels_per_car = 5) 
  (h4 : wheels_per_motorcycle = 2) : 
  num_cars * wheels_per_car + num_motorcycles * wheels_per_motorcycle = 117 := by
sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l3293_329381


namespace NUMINAMATH_CALUDE_no_8002_integers_divisibility_property_l3293_329348

theorem no_8002_integers_divisibility_property (P : ℕ → ℕ) 
  (h_P : ∀ x : ℕ, P x = x^2000 - x^1000 + 1) : 
  ¬ ∃ (a : Fin 8002 → ℕ), 
    (∀ i j : Fin 8002, i ≠ j → a i ≠ a j) ∧ 
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → 
      (a i * a j * a k) ∣ (P (a i) * P (a j) * P (a k))) :=
sorry

end NUMINAMATH_CALUDE_no_8002_integers_divisibility_property_l3293_329348


namespace NUMINAMATH_CALUDE_undefined_value_expression_undefined_l3293_329330

theorem undefined_value (x : ℝ) : 
  (x^2 - 24*x + 144 = 0) ↔ (x = 12) := by sorry

theorem expression_undefined (x : ℝ) : 
  ¬(∃y : ℝ, y = (3*x^3 + 5) / (x^2 - 24*x + 144)) ↔ (x = 12) := by sorry

end NUMINAMATH_CALUDE_undefined_value_expression_undefined_l3293_329330


namespace NUMINAMATH_CALUDE_algebraic_expressions_simplification_l3293_329360

theorem algebraic_expressions_simplification (x y m a b c : ℝ) :
  (4 * y * (-2 * x * y^2) = -8 * x * y^3) ∧
  ((-5/2 * x^2) * (-4 * x) = 10 * x^3) ∧
  ((3 * m^2) * (-2 * m^3)^2 = 12 * m^8) ∧
  ((-a * b^2 * c^3)^2 * (-a^2 * b)^3 = -a^8 * b^7 * c^6) := by
sorry


end NUMINAMATH_CALUDE_algebraic_expressions_simplification_l3293_329360


namespace NUMINAMATH_CALUDE_income_expenditure_ratio_l3293_329326

/-- Given a person's income and savings, calculate the ratio of income to expenditure -/
theorem income_expenditure_ratio (income savings : ℕ) (h1 : income = 18000) (h2 : savings = 3600) :
  (income : ℚ) / (income - savings) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_income_expenditure_ratio_l3293_329326


namespace NUMINAMATH_CALUDE_frame_area_percentage_l3293_329303

theorem frame_area_percentage (square_side : ℝ) (frame_width : ℝ) : 
  square_side = 80 → frame_width = 4 → 
  (square_side^2 - (square_side - 2 * frame_width)^2) / square_side^2 * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_frame_area_percentage_l3293_329303


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3293_329325

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ 1983 = 1982 * 31 * 5 - 1981 * (31 * 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3293_329325


namespace NUMINAMATH_CALUDE_staircase_step_difference_l3293_329396

/-- Theorem: Difference in steps between second and third staircases --/
theorem staircase_step_difference :
  ∀ (steps1 steps2 steps3 : ℕ) (step_height : ℚ) (total_height : ℚ),
    steps1 = 20 →
    steps2 = 2 * steps1 →
    step_height = 1/2 →
    total_height = 45 →
    (steps1 + steps2 + steps3 : ℚ) * step_height = total_height →
    steps2 - steps3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_staircase_step_difference_l3293_329396


namespace NUMINAMATH_CALUDE_melanie_picked_zero_pears_l3293_329350

/-- The number of pears Melanie picked -/
def melanie_pears : ℕ := 0

/-- The number of plums Alyssa picked -/
def alyssa_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

theorem melanie_picked_zero_pears :
  alyssa_plums + jason_plums = total_plums → melanie_pears = 0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_picked_zero_pears_l3293_329350


namespace NUMINAMATH_CALUDE_merchant_profit_problem_l3293_329337

theorem merchant_profit_problem (X : ℕ) (C S : ℝ) : 
  X * C = 25 * S → -- Cost price of X articles equals selling price of 25 articles
  S = 1.6 * C →    -- 60% profit, selling price is 160% of cost price
  X = 40           -- Number of articles bought at cost price is 40
  := by sorry

end NUMINAMATH_CALUDE_merchant_profit_problem_l3293_329337


namespace NUMINAMATH_CALUDE_simplify_complex_square_l3293_329333

theorem simplify_complex_square : 
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_square_l3293_329333


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l3293_329368

/-- Given three vectors in R³, prove that they are not coplanar. -/
theorem vectors_not_coplanar : 
  let a : Fin 3 → ℝ := ![3, 3, 1]
  let b : Fin 3 → ℝ := ![1, -2, 1]
  let c : Fin 3 → ℝ := ![1, 1, 1]
  ¬ (∃ (x y z : ℝ), x • a + y • b + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l3293_329368


namespace NUMINAMATH_CALUDE_henrys_shells_l3293_329372

theorem henrys_shells (perfect_shells : ℕ) (non_spiral_perfect : ℕ) (broken_spiral_diff : ℕ) :
  perfect_shells = 17 →
  non_spiral_perfect = 12 →
  broken_spiral_diff = 21 →
  (perfect_shells - non_spiral_perfect + broken_spiral_diff) * 2 = 52 := by
sorry

end NUMINAMATH_CALUDE_henrys_shells_l3293_329372


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l3293_329392

theorem perfect_square_function_characterization 
  (g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l3293_329392


namespace NUMINAMATH_CALUDE_circle_equation_l3293_329336

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A and B
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    line_l center ∧
    center = (-3, -2) ∧
    radius = 5 :=
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3293_329336


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_l3293_329364

def last_two_digits (n : ℕ) : ℕ := n % 100

def periodic_sequence (a : ℕ → ℕ) (period : ℕ) :=
  ∀ n : ℕ, a (n + period) = a n

theorem last_two_digits_of_7_power (n : ℕ) (h : n ≥ 2) :
  periodic_sequence (λ k => last_two_digits (7^k)) 4 →
  last_two_digits (7^2017) = last_two_digits (7^5) :=
by
  sorry

#eval last_two_digits (7^5)  -- Should output 7

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_l3293_329364


namespace NUMINAMATH_CALUDE_weight_after_first_week_l3293_329313

/-- Given Jessie's initial weight and weight loss in the first week, 
    calculate her weight after the first week of jogging. -/
theorem weight_after_first_week 
  (initial_weight : ℕ) 
  (weight_loss_first_week : ℕ) 
  (h1 : initial_weight = 92) 
  (h2 : weight_loss_first_week = 56) : 
  initial_weight - weight_loss_first_week = 36 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_first_week_l3293_329313


namespace NUMINAMATH_CALUDE_apple_delivery_problem_l3293_329344

theorem apple_delivery_problem (first_grade_value second_grade_value : ℝ)
  (price_difference : ℝ) (quantity_difference : ℝ) :
  first_grade_value = 228 →
  second_grade_value = 180 →
  price_difference = 0.9 →
  quantity_difference = 5 →
  ∃ x : ℝ,
    x > 0 ∧
    x + quantity_difference > 0 ∧
    (first_grade_value / x - price_difference) * (2 * x + quantity_difference) =
      first_grade_value + second_grade_value ∧
    2 * x + quantity_difference = 85 :=
by sorry

end NUMINAMATH_CALUDE_apple_delivery_problem_l3293_329344


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3293_329312

theorem regular_polygon_sides (internal_angle : ℝ) (h : internal_angle = 150) :
  (360 : ℝ) / (180 - internal_angle) = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3293_329312


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3293_329319

-- Define set A
def A : Set ℝ := {x | x * (x - 1) < 0}

-- Define set B
def B : Set ℝ := {x | Real.exp x > 1}

-- Define the closed interval [1,+∞)
def closed_interval : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = closed_interval := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3293_329319


namespace NUMINAMATH_CALUDE_alternating_series_sum_l3293_329373

def arithmetic_series (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 = 0 then x else -x)) 0

theorem alternating_series_sum :
  let series := arithmetic_series 3 4 18
  alternating_sum series = -36 := by sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l3293_329373


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3293_329347

theorem lcm_from_product_and_hcf (a b : ℕ+) : 
  a * b = 18000 → Nat.gcd a b = 30 → Nat.lcm a b = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l3293_329347


namespace NUMINAMATH_CALUDE_circle_squares_inequality_l3293_329315

theorem circle_squares_inequality (x y : ℝ) : 
  abs x + abs y ≤ Real.sqrt (2 * (x^2 + y^2)) ∧ 
  Real.sqrt (2 * (x^2 + y^2)) ≤ 2 * max (abs x) (abs y) := by
  sorry

end NUMINAMATH_CALUDE_circle_squares_inequality_l3293_329315


namespace NUMINAMATH_CALUDE_kara_water_consumption_l3293_329332

/-- The amount of water Kara drinks with each dose of medication -/
def water_per_dose : ℕ := 4

/-- The number of doses Kara takes per day -/
def doses_per_day : ℕ := 3

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Kara followed her medication routine -/
def total_weeks : ℕ := 2

/-- The number of doses Kara forgot in the second week -/
def forgotten_doses : ℕ := 2

/-- Calculates the total amount of water Kara drank with her medication over two weeks -/
def total_water_consumption : ℕ :=
  water_per_dose * doses_per_day * days_per_week * total_weeks - water_per_dose * forgotten_doses

theorem kara_water_consumption :
  total_water_consumption = 160 := by
  sorry

end NUMINAMATH_CALUDE_kara_water_consumption_l3293_329332


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_point_specific_circle_equation_l3293_329389

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of a circle given its center and a point on the circle -/
theorem circle_equation_from_center_and_point 
  (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  ∃ (c : Circle), 
    c.center = center ∧ 
    lies_on_circle c point ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ (x - center.1)^2 + (y - center.2)^2 = c.radius^2 := by
  sorry

/-- The specific circle equation for the given problem -/
theorem specific_circle_equation : 
  ∃ (c : Circle), 
    c.center = (0, 4) ∧ 
    lies_on_circle c (3, 0) ∧
    ∀ (x y : ℝ), lies_on_circle c (x, y) ↔ x^2 + (y - 4)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_point_specific_circle_equation_l3293_329389


namespace NUMINAMATH_CALUDE_election_winner_votes_l3293_329398

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  winner_percentage = 65 / 100 →
  vote_difference = 300 →
  winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference →
  winner_percentage * total_votes = 650 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3293_329398


namespace NUMINAMATH_CALUDE_min_value_theorem_l3293_329363

/-- Given real numbers a, b, c, d satisfying the equation,
    the minimum value of the expression is 8 -/
theorem min_value_theorem (a b c d : ℝ) 
  (h : (b - 2*a^2 + 3*Real.log a)^2 + (c - d - 3)^2 = 0) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y z w : ℝ), 
    (y - 2*x^2 + 3*Real.log x)^2 + (z - w - 3)^2 = 0 → 
    (x - z)^2 + (y - w)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3293_329363


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3293_329375

/-- The perimeter of a rectangle with longer sides 28cm and shorter sides 22cm is 100cm -/
theorem rectangle_perimeter : ℕ → ℕ → ℕ
  | 28, 22 => 100
  | _, _ => 0

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l3293_329375


namespace NUMINAMATH_CALUDE_count_distinct_prime_factors_30_factorial_l3293_329310

/-- The number of distinct prime factors of 30! -/
def distinct_prime_factors_30_factorial : ℕ := sorry

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem count_distinct_prime_factors_30_factorial :
  distinct_prime_factors_30_factorial = 10 := by sorry

end NUMINAMATH_CALUDE_count_distinct_prime_factors_30_factorial_l3293_329310


namespace NUMINAMATH_CALUDE_intersecting_planes_parallel_line_l3293_329340

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection relation for planes
variable (intersect : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Theorem statement
theorem intersecting_planes_parallel_line 
  (α β : Plane) 
  (h_intersect : intersect α β) :
  ∃ l : Line, parallel l α ∧ parallel l β := by
  sorry

end NUMINAMATH_CALUDE_intersecting_planes_parallel_line_l3293_329340


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3293_329385

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3293_329385


namespace NUMINAMATH_CALUDE_perimeter_ratio_l3293_329354

/-- Triangle PQR with sides of length 6, 8, and 10 units -/
def PQR : Fin 3 → ℝ := ![6, 8, 10]

/-- Triangle STU with sides of length 9, 12, and 15 units -/
def STU : Fin 3 → ℝ := ![9, 12, 15]

/-- Perimeter of a triangle given its side lengths -/
def perimeter (triangle : Fin 3 → ℝ) : ℝ :=
  triangle 0 + triangle 1 + triangle 2

/-- The ratio of the perimeter of triangle PQR to the perimeter of triangle STU is 2/3 -/
theorem perimeter_ratio :
  perimeter PQR / perimeter STU = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_l3293_329354


namespace NUMINAMATH_CALUDE_plane_through_line_and_point_l3293_329318

/-- A line in 3D space defined by symmetric equations -/
structure Line3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space defined by its equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  (p.x - l.a) / l.b = (p.y - l.c) / l.d ∧ (p.y - l.c) / l.d = p.z / l.f

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

theorem plane_through_line_and_point 
  (l : Line3D) 
  (p : Point3D) : 
  ∃ (pl : Plane), 
    (∀ (q : Point3D), pointOnLine q l → pointOnPlane q pl) ∧ 
    pointOnPlane p pl ∧ 
    pl.a = 5 ∧ pl.b = -2 ∧ pl.c = 2 ∧ pl.d = 1 := by
  sorry

#check plane_through_line_and_point

end NUMINAMATH_CALUDE_plane_through_line_and_point_l3293_329318


namespace NUMINAMATH_CALUDE_particle_max_elevation_l3293_329322

noncomputable def s (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

theorem particle_max_elevation :
  ∃ (max_height : ℝ), 
    (∀ t : ℝ, t ≥ 0 → s t ≤ max_height) ∧ 
    (∃ t : ℝ, t ≥ 0 ∧ s t = max_height) ∧
    (abs (max_height - 368.1) < 0.1) := by
  sorry

end NUMINAMATH_CALUDE_particle_max_elevation_l3293_329322


namespace NUMINAMATH_CALUDE_virginia_april_rainfall_l3293_329380

/-- Calculates the rainfall in April given the rainfall in other months and the average -/
def april_rainfall (march may june july average : ℝ) : ℝ :=
  5 * average - (march + may + june + july)

/-- Theorem stating that given the specified rainfall amounts and average, April's rainfall was 4.5 inches -/
theorem virginia_april_rainfall :
  let march : ℝ := 3.79
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  april_rainfall march may june july average = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_virginia_april_rainfall_l3293_329380


namespace NUMINAMATH_CALUDE_proposition_truth_l3293_329399

theorem proposition_truth (p q : Prop) (hp : ¬p) (hq : ¬q) :
  (p ∨ ¬q) ∧ ¬(p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l3293_329399


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3293_329355

/-- Given a hyperbola and a parabola with specific properties, prove that the eccentricity of the hyperbola is √2 + 1 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c = Real.sqrt (a^2 + b^2)) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4 * c * x
  let A := (c, 2 * c)
  let B := (-c, -2 * c)
  (hyperbola A.1 A.2 ∧ parabola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ parabola B.1 B.2) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * c)^2 →
  let e := c / a
  e = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3293_329355


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3293_329316

/-- The quadratic function f(x) = 2(x - 4)² + 6 has a minimum value of 6 -/
theorem quadratic_minimum (x : ℝ) : ∀ y : ℝ, 2 * (x - 4)^2 + 6 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3293_329316


namespace NUMINAMATH_CALUDE_probability_4H_before_3T_value_l3293_329382

/-- The probability of getting 4 heads before 3 tails in repeated fair coin flips -/
def probability_4H_before_3T : ℚ :=
  13 / 17

/-- Theorem stating that the probability of getting 4 heads before 3 tails
    in repeated fair coin flips is 13/17 -/
theorem probability_4H_before_3T_value :
  probability_4H_before_3T = 13 / 17 := by
  sorry

#eval Nat.gcd 13 17  -- To verify that 13 and 17 are coprime

end NUMINAMATH_CALUDE_probability_4H_before_3T_value_l3293_329382


namespace NUMINAMATH_CALUDE_abc_product_one_l3293_329323

theorem abc_product_one (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b^2 = b^2 + 1/c^2 ∧ b^2 + 1/c^2 = c^2 + 1/a^2) :
  |a*b*c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_one_l3293_329323


namespace NUMINAMATH_CALUDE_smallest_denominator_for_repeating_2015_l3293_329353

/-- Given positive integers a and b where a/b is a repeating decimal with the sequence 2015,
    the smallest possible value of b is 129. -/
theorem smallest_denominator_for_repeating_2015 (a b : ℕ+) :
  (∃ k : ℕ, (a : ℚ) / b = 2015 / (10000 ^ k - 1)) →
  (∀ c : ℕ+, c < b → ¬∃ d : ℕ+, (d : ℚ) / c = 2015 / 9999) →
  b = 129 := by
  sorry


end NUMINAMATH_CALUDE_smallest_denominator_for_repeating_2015_l3293_329353


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_inequality_l3293_329356

theorem sum_reciprocal_squares_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 3) :
  1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 ≤ 1/(a^2*b^2*c^2*d^2) := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_inequality_l3293_329356


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3293_329335

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 3)^2 ≠ 0 → False) ∧
  (∀ x : ℝ, |2*x| + 4 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (3*x) - 1 ≠ 0 → False) ∧
  (∀ x : ℝ, x ≤ 0 → Real.sqrt (-3*x) - 3 ≠ 0 → False) ∧
  (∀ x : ℝ, |5*x| - 6 ≠ 0 → False) := by
  sorry

#check no_solution_absolute_value_equation

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l3293_329335


namespace NUMINAMATH_CALUDE_color_paint_can_size_is_one_gallon_l3293_329345

/-- Represents the paint job for a house --/
structure PaintJob where
  bedrooms : Nat
  otherRooms : Nat
  gallonsPerRoom : Nat
  whitePaintCanSize : Nat
  totalCans : Nat

/-- Calculates the size of each can of color paint --/
def colorPaintCanSize (job : PaintJob) : Rat :=
  let totalRooms := job.bedrooms + job.otherRooms
  let totalPaint := totalRooms * job.gallonsPerRoom
  let whitePaint := job.otherRooms * job.gallonsPerRoom
  let whiteCans := whitePaint / job.whitePaintCanSize
  let colorCans := job.totalCans - whiteCans
  let colorPaint := job.bedrooms * job.gallonsPerRoom
  colorPaint / colorCans

/-- Theorem stating that the size of each can of color paint is 1 gallon --/
theorem color_paint_can_size_is_one_gallon (job : PaintJob)
  (h1 : job.bedrooms = 3)
  (h2 : job.otherRooms = 2 * job.bedrooms)
  (h3 : job.gallonsPerRoom = 2)
  (h4 : job.whitePaintCanSize = 3)
  (h5 : job.totalCans = 10) :
  colorPaintCanSize job = 1 := by
  sorry

#eval colorPaintCanSize { bedrooms := 3, otherRooms := 6, gallonsPerRoom := 2, whitePaintCanSize := 3, totalCans := 10 }

end NUMINAMATH_CALUDE_color_paint_can_size_is_one_gallon_l3293_329345


namespace NUMINAMATH_CALUDE_journey_time_proof_l3293_329327

theorem journey_time_proof (s : ℝ) (h1 : s > 0) (h2 : s - 1/2 > 0) : 
  (45 / (s - 1/2) - 45 / s = 3/4) → (45 / s = 45 / s) :=
by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l3293_329327


namespace NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l3293_329351

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a^2 = k * l) :
  ¬ ∃ (n : ℕ), (k + l : ℚ) / (2 * a) = n := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_not_integer_l3293_329351


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3293_329349

theorem partial_fraction_decomposition :
  let P : ℚ := 17/7
  let Q : ℚ := 4/7
  ∀ x : ℚ, x ≠ 10 → x ≠ -4 →
    (3*x + 4) / (x^2 - 6*x - 40) = P / (x - 10) + Q / (x + 4) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3293_329349


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l3293_329331

/-- Given a geometric sequence with common ratio q < 0, prove that a₉S₈ > a₈S₉ -/
theorem geometric_sequence_inequality (a₁ : ℝ) (q : ℝ) (hq : q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  let S : ℕ → ℝ := λ n => a₁ * (1 - q^n) / (1 - q)
  (a 9) * (S 8) > (a 8) * (S 9) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l3293_329331


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l3293_329338

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 27

-- State the theorem
theorem f_extrema_on_interval :
  (∀ x ∈ Set.Icc 0 3, f x ≤ 54) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 54) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ 27/4) ∧
  (∃ x ∈ Set.Icc 0 3, f x = 27/4) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l3293_329338


namespace NUMINAMATH_CALUDE_det2022_2023_2021_2022_solve_det_eq_16_l3293_329388

-- Definition of second-order determinant
def det2 (a b c d : ℤ) : ℤ := a * d - b * c

-- Theorem 1
theorem det2022_2023_2021_2022 : det2 2022 2023 2021 2022 = 1 := by sorry

-- Theorem 2
theorem solve_det_eq_16 (m : ℤ) : det2 (m + 2) (m - 2) (m - 2) (m + 2) = 16 → m = 2 := by sorry

end NUMINAMATH_CALUDE_det2022_2023_2021_2022_solve_det_eq_16_l3293_329388


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l3293_329386

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45° -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let rotation_angle : ℝ := 45 * π / 180
  let shaded_area := (2 * R)^2 * rotation_angle / 2
  shaded_area = π * R^2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_rotated_semicircle_area_l3293_329386


namespace NUMINAMATH_CALUDE_chord_length_squared_l3293_329329

/-- Two circles with radii 8 and 6, centers 12 units apart, intersecting at P.
    Q and R are points on the circles such that QP = PR. -/
structure CircleConfiguration where
  circle1_radius : ℝ
  circle2_radius : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h1 : circle1_radius = 8
  h2 : circle2_radius = 6
  h3 : center_distance = 12
  h4 : chord_length > 0

/-- The square of the chord length in the given circle configuration is 130. -/
theorem chord_length_squared (config : CircleConfiguration) : 
  config.chord_length ^ 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3293_329329


namespace NUMINAMATH_CALUDE_line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l3293_329324

/-- A line passing through point P(1,3) with equal x and y intercepts -/
def line_with_equal_intercepts : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem line_passes_through_point :
  (1, 3) ∈ line_with_equal_intercepts := by sorry

theorem line_has_equal_intercepts :
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ line_with_equal_intercepts ∧ (0, a) ∈ line_with_equal_intercepts := by sorry

theorem line_equation_is_correct :
  line_with_equal_intercepts = {p : ℝ × ℝ | p.1 + p.2 = 4} := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_line_has_equal_intercepts_line_equation_is_correct_l3293_329324


namespace NUMINAMATH_CALUDE_greatest_fraction_l3293_329320

theorem greatest_fraction (a b : ℕ) (h1 : a + b = 101) (h2 : (a : ℚ) / b ≤ 1/3) :
  (a : ℚ) / b ≤ 25/76 ∧ ∃ (a' b' : ℕ), a' + b' = 101 ∧ (a' : ℚ) / b' = 25/76 ∧ (a' : ℚ) / b' ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_fraction_l3293_329320


namespace NUMINAMATH_CALUDE_quadratic_system_sum_l3293_329377

theorem quadratic_system_sum (x y r₁ s₁ r₂ s₂ : ℝ) : 
  (9 * x^2 - 27 * x - 54 = 0) →
  (4 * y^2 + 28 * y + 49 = 0) →
  ((x - r₁)^2 = s₁) →
  ((y - r₂)^2 = s₂) →
  (r₁ + s₁ + r₂ + s₂ = -11/4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_system_sum_l3293_329377


namespace NUMINAMATH_CALUDE_f_composite_negative_two_l3293_329379

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composite_negative_two :
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composite_negative_two_l3293_329379


namespace NUMINAMATH_CALUDE_fruit_store_discount_l3293_329304

/-- 
Given a fruit store scenario with:
- Total weight of fruit: 1000kg
- Cost price: 7 yuan per kg
- Original selling price: 10 yuan per kg
- Half of the fruit is sold at original price
- Total profit must not be less than 2000 yuan

This theorem states that the minimum discount factor x for the remaining half of the fruit
satisfies: x ≤ 7/11
-/
theorem fruit_store_discount (total_weight : ℝ) (cost_price selling_price : ℝ) 
  (min_profit : ℝ) (x : ℝ) :
  total_weight = 1000 →
  cost_price = 7 →
  selling_price = 10 →
  min_profit = 2000 →
  (total_weight / 2 * (selling_price - cost_price) + 
   total_weight / 2 * (selling_price * (1 - x) - cost_price) ≥ min_profit) →
  x ≤ 7 / 11 := by
  sorry


end NUMINAMATH_CALUDE_fruit_store_discount_l3293_329304


namespace NUMINAMATH_CALUDE_find_other_number_l3293_329305

theorem find_other_number (n m : ℕ+) 
  (h_lcm : Nat.lcm n m = 52)
  (h_gcd : Nat.gcd n m = 8)
  (h_n : n = 26) : 
  m = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3293_329305


namespace NUMINAMATH_CALUDE_max_product_is_18000_l3293_329376

def numbers : List ℕ := [10, 15, 20, 30, 40, 60]

def is_valid_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 6 ∧ 
  arrangement.toFinset = numbers.toFinset ∧
  ∃ (product : ℕ), 
    (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
    (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
    (arrangement[2]! * arrangement[4]! * arrangement[5]! = product)

theorem max_product_is_18000 :
  ∀ (arrangement : List ℕ), is_valid_arrangement arrangement →
    ∃ (product : ℕ), 
      (arrangement[0]! * arrangement[1]! * arrangement[2]! = product) ∧
      (arrangement[1]! * arrangement[3]! * arrangement[4]! = product) ∧
      (arrangement[2]! * arrangement[4]! * arrangement[5]! = product) ∧
      product ≤ 18000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_is_18000_l3293_329376


namespace NUMINAMATH_CALUDE_custom_mult_equation_solution_l3293_329341

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := a * b + a + b

/-- Theorem stating that if 3 * (3x - 1) = 27 under the custom multiplication, then x = 7/3 -/
theorem custom_mult_equation_solution :
  ∀ x : ℝ, custom_mult 3 (3 * x - 1) = 27 → x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_equation_solution_l3293_329341


namespace NUMINAMATH_CALUDE_video_game_lives_l3293_329309

theorem video_game_lives (initial_lives lives_lost lives_gained : ℕ) :
  initial_lives - lives_lost + lives_gained = initial_lives + lives_gained - lives_lost :=
by sorry

#check video_game_lives 43 14 27

end NUMINAMATH_CALUDE_video_game_lives_l3293_329309


namespace NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l3293_329394

/-- A regular dodecagon -/
structure RegularDodecagon where
  -- Add any necessary properties here

/-- Represents a diagonal in the dodecagon -/
structure Diagonal where
  -- Add any necessary properties here

/-- The probability that two randomly chosen diagonals intersect inside a regular dodecagon -/
def intersection_probability (d : RegularDodecagon) : ℚ :=
  495 / 1431

/-- Theorem stating that the probability of two randomly chosen diagonals 
    intersecting inside a regular dodecagon is 495/1431 -/
theorem dodecagon_diagonal_intersection_probability (d : RegularDodecagon) :
  intersection_probability d = 495 / 1431 := by
  sorry


end NUMINAMATH_CALUDE_dodecagon_diagonal_intersection_probability_l3293_329394


namespace NUMINAMATH_CALUDE_equation_solution_l3293_329383

theorem equation_solution : ∃ x : ℝ, (2*x - 1)^2 - (1 - 3*x)^2 = 5*(1 - x)*(x + 1) ∧ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3293_329383


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l3293_329328

/-- A hexagon in 2D space -/
structure Hexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon from the problem -/
def specificHexagon : Hexagon :=
  { v1 := (0, 0)
    v2 := (1, 4)
    v3 := (3, 4)
    v4 := (4, 0)
    v5 := (3, -4)
    v6 := (1, -4) }

/-- Theorem stating that the area of the specific hexagon is 24 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 24 := by sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l3293_329328


namespace NUMINAMATH_CALUDE_card_exchange_probability_l3293_329397

def number_of_people : ℕ := 4

def probability_B_drew_A_given_A_drew_B : ℚ :=
  1 / 3

theorem card_exchange_probability :
  ∀ (A B : Fin number_of_people),
  A ≠ B →
  (probability_B_drew_A_given_A_drew_B : ℚ) =
    (1 : ℚ) / (number_of_people - 1 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_card_exchange_probability_l3293_329397


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3293_329390

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3293_329390


namespace NUMINAMATH_CALUDE_speedster_convertible_fraction_l3293_329362

theorem speedster_convertible_fraction (T S : ℕ) (h1 : S = 3 * T / 4) (h2 : T - S = 30) : 
  54 / S = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertible_fraction_l3293_329362


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3293_329361

theorem least_number_for_divisibility : ∃! x : ℕ, x < 25 ∧ (1056 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1056 + y) % 25 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3293_329361


namespace NUMINAMATH_CALUDE_product_of_roots_l3293_329374

theorem product_of_roots : (32 : ℝ) ^ (1/5 : ℝ) * (128 : ℝ) ^ (1/7 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3293_329374


namespace NUMINAMATH_CALUDE_base_conversion_l3293_329371

/-- Given a base r where 175 in base r equals 125 in base 10, 
    prove that 76 in base r equals 62 in base 10 -/
theorem base_conversion (r : ℕ) (hr : r > 1) : 
  (1 * r^2 + 7 * r + 5 = 125) → (7 * r + 6 = 62) :=
by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3293_329371


namespace NUMINAMATH_CALUDE_two_digit_number_relationship_l3293_329393

theorem two_digit_number_relationship :
  ∀ (tens units : ℕ),
    tens * 10 + units = 16 →
    tens + units = 7 →
    ∃ (k : ℕ), units = k * tens →
    units = 6 * tens :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_relationship_l3293_329393


namespace NUMINAMATH_CALUDE_cake_sharing_l3293_329370

theorem cake_sharing (n : ℕ) : 
  (∃ (shares : Fin n → ℚ), 
    (∀ i, 0 < shares i) ∧ 
    (∃ j, shares j = 1/11) ∧
    (∃ k, shares k = 1/14) ∧
    (∀ i, 1/14 ≤ shares i ∧ shares i ≤ 1/11) ∧
    (Finset.sum Finset.univ shares = 1)) ↔ 
  (n = 12 ∨ n = 13) :=
by sorry


end NUMINAMATH_CALUDE_cake_sharing_l3293_329370


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3293_329357

def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3293_329357


namespace NUMINAMATH_CALUDE_ratio_proof_l3293_329365

theorem ratio_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (hx : x = 1.25 * a) (hm : m = 0.2 * b) (hm_x : m / x = 0.2) : 
  a / b = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l3293_329365


namespace NUMINAMATH_CALUDE_right_triangle_with_condition_l3293_329301

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def satisfies_condition (a b c : ℕ) : Prop :=
  a + b = c + 6

theorem right_triangle_with_condition :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ≤ b →
    is_right_triangle a b c →
    satisfies_condition a b c →
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
by
  sorry

#check right_triangle_with_condition

end NUMINAMATH_CALUDE_right_triangle_with_condition_l3293_329301


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l3293_329343

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l3293_329343


namespace NUMINAMATH_CALUDE_only_component_life_uses_experiments_l3293_329300

/-- Represents a method of data collection --/
inductive DataCollectionMethod
  | Observation
  | Experiment
  | Investigation

/-- Represents the different scenarios --/
inductive Scenario
  | TemperatureMeasurement
  | ComponentLifeDetermination
  | TVRatings
  | CounterfeitDetection

/-- Maps each scenario to its typical data collection method --/
def typicalMethod (s : Scenario) : DataCollectionMethod :=
  match s with
  | Scenario.TemperatureMeasurement => DataCollectionMethod.Observation
  | Scenario.ComponentLifeDetermination => DataCollectionMethod.Experiment
  | Scenario.TVRatings => DataCollectionMethod.Investigation
  | Scenario.CounterfeitDetection => DataCollectionMethod.Investigation

theorem only_component_life_uses_experiments :
  ∀ s : Scenario, typicalMethod s = DataCollectionMethod.Experiment ↔ s = Scenario.ComponentLifeDetermination :=
by sorry


end NUMINAMATH_CALUDE_only_component_life_uses_experiments_l3293_329300


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3293_329339

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), (y = k * x + 1 ∧ x^2 + y^2 = 2) ∧ 
  ¬(∃ (x y : ℝ), y = k * x + 1 ∧ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3293_329339


namespace NUMINAMATH_CALUDE_class_average_calculation_l3293_329378

/-- Proves that the overall average of a class is 63.4 marks given specific score distributions -/
theorem class_average_calculation (total_students : ℕ) 
  (high_scorers : ℕ) (high_score : ℝ)
  (zero_scorers : ℕ) 
  (mid_scorers : ℕ) (mid_score : ℝ)
  (remaining_scorers : ℕ) (remaining_score : ℝ) :
  total_students = 50 ∧ 
  high_scorers = 6 ∧ 
  high_score = 95 ∧
  zero_scorers = 4 ∧
  mid_scorers = 10 ∧
  mid_score = 80 ∧
  remaining_scorers = total_students - (high_scorers + zero_scorers + mid_scorers) ∧
  remaining_score = 60 →
  (high_scorers * high_score + zero_scorers * 0 + mid_scorers * mid_score + remaining_scorers * remaining_score) / total_students = 63.4 := by
  sorry

#eval (6 * 95 + 4 * 0 + 10 * 80 + 30 * 60) / 50

end NUMINAMATH_CALUDE_class_average_calculation_l3293_329378


namespace NUMINAMATH_CALUDE_E_parity_2021_2022_2023_l3293_329311

def E : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => E (n + 2) + E (n + 1) + E n

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem E_parity_2021_2022_2023 :
  is_even (E 2021) ∧ ¬is_even (E 2022) ∧ ¬is_even (E 2023) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_2021_2022_2023_l3293_329311


namespace NUMINAMATH_CALUDE_power_of_64_l3293_329334

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 :=
by
  have h : 64 = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l3293_329334
