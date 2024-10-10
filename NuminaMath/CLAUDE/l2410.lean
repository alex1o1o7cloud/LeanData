import Mathlib

namespace trapezoid_y_property_l2410_241006

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  -- Length of the shorter base
  c : ℝ
  -- Height of the trapezoid
  k : ℝ
  -- The segment joining midpoints divides the trapezoid into regions with area ratio 3:4
  midpoint_ratio : (c + 75) / (c + 150) = 3 / 4
  -- Length of the segment that divides the trapezoid into two equal areas
  y : ℝ
  -- The segment y divides the trapezoid into two equal areas
  equal_areas : y^2 = 65250

/-- The main theorem stating the property of y -/
theorem trapezoid_y_property (t : Trapezoid) : ⌊t.y^2 / 150⌋ = 435 := by
  sorry

#check trapezoid_y_property

end trapezoid_y_property_l2410_241006


namespace intersection_points_sum_greater_than_two_l2410_241020

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem intersection_points_sum_greater_than_two (a t x₁ x₂ : ℝ) 
  (ha : a ≤ 0) (ht : -1 < t ∧ t < 0) (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf₁ : f a x₁ = t) (hf₂ : f a x₂ = t) : 
  x₁ + x₂ > 2 := by sorry

end intersection_points_sum_greater_than_two_l2410_241020


namespace merchant_pricing_strategy_l2410_241005

theorem merchant_pricing_strategy 
  (list_price : ℝ) 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (selling_price : ℝ) 
  (h1 : cost_price = 0.7 * list_price)  -- 30% discount on purchase
  (h2 : selling_price = 0.8 * marked_price)  -- 20% discount on sale
  (h3 : cost_price = 0.7 * selling_price)  -- 30% profit on selling price
  : marked_price = 1.25 * list_price :=
by sorry

end merchant_pricing_strategy_l2410_241005


namespace ratio_x_to_2y_l2410_241064

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : 
  x / (2 * y) = 3 / 2 := by
sorry

end ratio_x_to_2y_l2410_241064


namespace max_guaranteed_pastries_l2410_241095

/-- Represents a game with circular arrangement of plates and pastries. -/
structure PastryGame where
  num_plates : Nat
  max_move : Nat

/-- Represents the result of the game. -/
inductive GameResult
  | CanGuarantee
  | CannotGuarantee

/-- Determines if a certain number of pastries can be guaranteed on a single plate. -/
def can_guarantee (game : PastryGame) (k : Nat) : GameResult :=
  sorry

/-- The main theorem stating the maximum number of pastries that can be guaranteed. -/
theorem max_guaranteed_pastries (game : PastryGame) : 
  game.num_plates = 2019 → game.max_move = 16 → can_guarantee game 32 = GameResult.CanGuarantee ∧ 
  can_guarantee game 33 = GameResult.CannotGuarantee :=
  sorry

end max_guaranteed_pastries_l2410_241095


namespace prime_odd_sum_product_l2410_241029

theorem prime_odd_sum_product (p q : ℕ) : 
  Prime p → 
  Odd q → 
  q > 0 → 
  p^2 + q = 125 → 
  p * q = 242 := by
sorry

end prime_odd_sum_product_l2410_241029


namespace function_inequality_l2410_241003

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
    |f a x₁ - f a x₂| ≤ a - 1) →
  a ≥ Real.exp 1 := by
  sorry

end function_inequality_l2410_241003


namespace ellipse_equation_with_shared_focus_l2410_241098

/-- Given a parabola and an ellipse with shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_with_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x) →  -- Parabola exists
  (∃ (x y : ℝ), x^2/a^2 + y^2 = 1) →  -- Ellipse exists
  (2 : ℝ) = a * (1 - 1/a^2).sqrt →  -- Focus of parabola is right focus of ellipse
  (∃ (x y : ℝ), x^2/5 + y^2 = 1) :=  -- Resulting ellipse equation
by sorry

end ellipse_equation_with_shared_focus_l2410_241098


namespace imaginary_part_of_z_l2410_241032

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.im z = -1 := by sorry

end imaginary_part_of_z_l2410_241032


namespace multiply_decimals_l2410_241045

theorem multiply_decimals : (0.25 : ℝ) * 0.08 = 0.02 := by
  sorry

end multiply_decimals_l2410_241045


namespace average_books_borrowed_l2410_241011

theorem average_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ)
  (h1 : total_students = 40)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 12)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (total_books : ℚ) / total_students = 39/20 := by
sorry

end average_books_borrowed_l2410_241011


namespace greatest_multiple_of_nine_with_unique_digits_mod_1000_l2410_241049

/-- A function that checks if a natural number has all unique digits -/
def hasUniqueDigits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_1000 :
  M % 1000 = 981 := by sorry

end greatest_multiple_of_nine_with_unique_digits_mod_1000_l2410_241049


namespace puppy_count_l2410_241093

theorem puppy_count (total_ears : ℕ) (ears_per_puppy : ℕ) (h1 : total_ears = 210) (h2 : ears_per_puppy = 2) :
  total_ears / ears_per_puppy = 105 :=
by sorry

end puppy_count_l2410_241093


namespace solve_linear_equation_l2410_241030

theorem solve_linear_equation : ∃ x : ℝ, 4 * x - 5 = 3 ∧ x = 2 := by
  sorry

end solve_linear_equation_l2410_241030


namespace infinite_solutions_iff_c_eq_five_halves_l2410_241034

theorem infinite_solutions_iff_c_eq_five_halves (c : ℚ) :
  (∀ y : ℚ, 3 * (5 + 2 * c * y) = 15 * y + 15) ↔ c = 5 / 2 := by
  sorry

end infinite_solutions_iff_c_eq_five_halves_l2410_241034


namespace sum_exterior_angles_regular_decagon_l2410_241033

/-- A regular decagon is a polygon with 10 sides -/
def RegularDecagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def SumExteriorAngles (p : Type) : ℝ := sorry

/-- Theorem: The sum of exterior angles of a regular decagon is 360° -/
theorem sum_exterior_angles_regular_decagon :
  SumExteriorAngles RegularDecagon = 360 := by sorry

end sum_exterior_angles_regular_decagon_l2410_241033


namespace coefficient_x_squared_in_binomial_expansion_l2410_241092

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (x + 2/x)^4
  ∃ (a b c d e : ℝ), binomial = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧ c = 8 :=
by sorry

end coefficient_x_squared_in_binomial_expansion_l2410_241092


namespace min_value_implies_a_l2410_241023

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |2*x + a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : (∀ x : ℝ, f a x ≥ 3) ∧ (∃ x : ℝ, f a x = 3) → a = -4 ∨ a = 8 := by
  sorry

end min_value_implies_a_l2410_241023


namespace additive_inverses_solution_l2410_241051

theorem additive_inverses_solution (x : ℝ) : (6 * x - 12) + (4 + 2 * x) = 0 → x = 1 := by
  sorry

end additive_inverses_solution_l2410_241051


namespace rainy_days_probability_l2410_241017

/-- The probability of rain on any given day -/
def p : ℚ := 1/5

/-- The number of days considered -/
def n : ℕ := 10

/-- The number of rainy days we're interested in -/
def k : ℕ := 3

/-- The probability of exactly k rainy days out of n days -/
def prob_k_rainy_days (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rainy_days_probability : 
  prob_k_rainy_days p n k = 1966080/9765625 := by sorry

end rainy_days_probability_l2410_241017


namespace volume_sin_squared_rotation_l2410_241002

theorem volume_sin_squared_rotation (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x ^ 2) :
  ∫ x in (0)..(Real.pi / 2), π * (f x)^2 = (3 * Real.pi^2) / 16 := by
  sorry

end volume_sin_squared_rotation_l2410_241002


namespace workers_completion_time_l2410_241074

/-- Given two workers who can each complete a task in 32 days, 
    prove they can complete the task together in 16 days -/
theorem workers_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 32 →
  work_rate_B = 1 / 32 →
  1 / (work_rate_A + work_rate_B) = 16 := by
sorry

end workers_completion_time_l2410_241074


namespace no_common_root_for_quadratics_l2410_241047

/-- Two quadratic polynomials with coefficients satisfying certain inequalities cannot have a common root -/
theorem no_common_root_for_quadratics (k m n l : ℝ) 
  (h1 : k > m) (h2 : m > n) (h3 : n > l) (h4 : l > 0) :
  ¬∃ x : ℝ, x^2 + m*x + n = 0 ∧ x^2 + k*x + l = 0 := by
  sorry


end no_common_root_for_quadratics_l2410_241047


namespace fraction_sum_l2410_241066

theorem fraction_sum : (2 : ℚ) / 3 + 5 / 18 - 1 / 6 = 7 / 9 := by
  sorry

end fraction_sum_l2410_241066


namespace sequence_correct_l2410_241081

def sequence_term (n : ℕ) : ℤ := (-1)^n * (2^n - 1)

theorem sequence_correct : 
  sequence_term 1 = -1 ∧ 
  sequence_term 2 = 3 ∧ 
  sequence_term 3 = -7 ∧ 
  sequence_term 4 = 15 := by
sorry

end sequence_correct_l2410_241081


namespace a₃_value_l2410_241008

/-- The function f(x) = x^6 -/
def f (x : ℝ) : ℝ := x^6

/-- The expansion of f(x) in terms of (1+x) -/
def f_expansion (x a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ := 
  a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5 + a₆*(1+x)^6

/-- Theorem: If f(x) = x^6 can be expressed as the expansion, then a₃ = -20 -/
theorem a₃_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, f x = f_expansion x a₀ a₁ a₂ a₃ a₄ a₅ a₆) → a₃ = -20 := by
  sorry

end a₃_value_l2410_241008


namespace infinite_solutions_imply_a_equals_five_l2410_241097

/-- If the equation 3(5 + ay) = 15y + 15 has infinitely many solutions for y, then a = 5 -/
theorem infinite_solutions_imply_a_equals_five (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 15) → a = 5 := by
  sorry

end infinite_solutions_imply_a_equals_five_l2410_241097


namespace vector_dot_product_equation_l2410_241048

/-- Given vectors a, b, c, and a dot product equation, prove that x = 1 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) :
  a = (1, 1) →
  b = (-1, 3) →
  c = (2, x) →
  (3 • a + b) • c = 10 →
  x = 1 := by sorry

end vector_dot_product_equation_l2410_241048


namespace lagrange_interpolation_identities_l2410_241031

theorem lagrange_interpolation_identities 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hca : c ≠ a) : 
  (1 / ((a - b) * (a - c)) + 1 / ((b - c) * (b - a)) + 1 / ((c - a) * (c - b)) = 0) ∧
  (a / ((a - b) * (a - c)) + b / ((b - c) * (b - a)) + c / ((c - a) * (c - b)) = 0) ∧
  (a^2 / ((a - b) * (a - c)) + b^2 / ((b - c) * (b - a)) + c^2 / ((c - a) * (c - b)) = 1) :=
by sorry

end lagrange_interpolation_identities_l2410_241031


namespace apples_bought_is_three_l2410_241056

/-- Calculates the number of apples bought given the total cost, number of oranges,
    price difference between oranges and apples, and the cost of each fruit. -/
def apples_bought (total_cost orange_count price_diff fruit_cost : ℚ) : ℚ :=
  (total_cost - orange_count * (fruit_cost + price_diff)) / fruit_cost

/-- Theorem stating that under the given conditions, the number of apples bought is 3. -/
theorem apples_bought_is_three :
  let total_cost : ℚ := 456/100
  let orange_count : ℚ := 7
  let price_diff : ℚ := 28/100
  let fruit_cost : ℚ := 26/100
  apples_bought total_cost orange_count price_diff fruit_cost = 3 := by
  sorry

#eval apples_bought (456/100) 7 (28/100) (26/100)

end apples_bought_is_three_l2410_241056


namespace jacks_remaining_money_l2410_241041

/-- Calculates the remaining money after currency conversion, fees, and spending --/
def calculate_remaining_money (
  initial_dollars : ℝ)
  (initial_euros : ℝ)
  (initial_yen : ℝ)
  (initial_rubles : ℝ)
  (euro_to_dollar : ℝ)
  (yen_to_dollar : ℝ)
  (ruble_to_dollar : ℝ)
  (transaction_fee : ℝ)
  (spending_percentage : ℝ) : ℝ :=
  let converted_euros := initial_euros * euro_to_dollar
  let converted_yen := initial_yen * yen_to_dollar
  let converted_rubles := initial_rubles * ruble_to_dollar
  let total_before_fees := initial_dollars + converted_euros + converted_yen + converted_rubles
  let fees := (converted_euros + converted_yen + converted_rubles) * transaction_fee
  let total_after_fees := total_before_fees - fees
  let amount_spent := total_after_fees * spending_percentage
  total_after_fees - amount_spent

/-- Theorem stating that Jack's remaining money is approximately $132.85 --/
theorem jacks_remaining_money :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |calculate_remaining_money 45 36 1350 1500 2 0.009 0.013 0.01 0.1 - 132.85| < ε :=
sorry

end jacks_remaining_money_l2410_241041


namespace rectangle_area_l2410_241059

/-- The area of a rectangular region bounded by y = a, y = a-2b, x = -2c, and x = d -/
theorem rectangle_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a - (a - 2 * b)) * (d - (-2 * c)) = 2 * b * d + 4 * b * c := by
  sorry

end rectangle_area_l2410_241059


namespace intersection_M_N_l2410_241071

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l2410_241071


namespace three_cubes_sum_equals_three_to_fourth_l2410_241024

theorem three_cubes_sum_equals_three_to_fourth : 3^3 + 3^3 + 3^3 = 3^4 := by
  sorry

end three_cubes_sum_equals_three_to_fourth_l2410_241024


namespace sam_initial_watermelons_l2410_241090

/-- The number of watermelons Sam grew initially -/
def initial_watermelons : ℕ := sorry

/-- The number of additional watermelons Sam grew -/
def additional_watermelons : ℕ := 3

/-- The total number of watermelons Sam has now -/
def total_watermelons : ℕ := 7

/-- Theorem stating that Sam grew 4 watermelons initially -/
theorem sam_initial_watermelons : 
  initial_watermelons + additional_watermelons = total_watermelons → initial_watermelons = 4 := by
  sorry

end sam_initial_watermelons_l2410_241090


namespace vector_operation_proof_l2410_241083

theorem vector_operation_proof (a b : ℝ × ℝ) :
  a = (2, 1) → b = (2, -2) → 2 • a - b = (2, 4) := by
  sorry

end vector_operation_proof_l2410_241083


namespace exactly_two_statements_true_l2410_241073

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

-- Define symmetry about x-axis
def symmetricAboutXAxis (a b : Point2D) : Prop :=
  a.x = b.x ∧ a.y = -b.y

-- Define the four statements
def statement1 (a b : Point2D) : Prop :=
  symmetricAboutYAxis a b → a.y = b.y

def statement2 (a b : Point2D) : Prop :=
  a.y = b.y → symmetricAboutYAxis a b

def statement3 (a b : Point2D) : Prop :=
  a.x = b.x → symmetricAboutXAxis a b

def statement4 (a b : Point2D) : Prop :=
  symmetricAboutXAxis a b → a.x = b.x

-- Theorem stating that exactly two of the statements are true
theorem exactly_two_statements_true :
  ∃ (a b : Point2D),
    (statement1 a b ∧ ¬statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) :=
  sorry

end exactly_two_statements_true_l2410_241073


namespace expected_rain_total_l2410_241035

/-- The number of days in the weather forecast. -/
def num_days : ℕ := 5

/-- The probability of a sunny day with no rain. -/
def prob_sun : ℝ := 0.4

/-- The probability of a day with 4 inches of rain. -/
def prob_rain_4 : ℝ := 0.25

/-- The probability of a day with 10 inches of rain. -/
def prob_rain_10 : ℝ := 0.35

/-- The amount of rain on a sunny day. -/
def rain_sun : ℝ := 0

/-- The amount of rain on a day with 4 inches of rain. -/
def rain_4 : ℝ := 4

/-- The amount of rain on a day with 10 inches of rain. -/
def rain_10 : ℝ := 10

/-- The expected value of rain for a single day. -/
def expected_rain_day : ℝ :=
  prob_sun * rain_sun + prob_rain_4 * rain_4 + prob_rain_10 * rain_10

/-- Theorem: The expected value of the total number of inches of rain for 5 days is 22.5 inches. -/
theorem expected_rain_total : num_days * expected_rain_day = 22.5 := by
  sorry

end expected_rain_total_l2410_241035


namespace algebraic_expression_value_l2410_241044

theorem algebraic_expression_value (x : ℝ) :
  x^2 + x + 5 = 8 → 2*x^2 + 2*x - 4 = 2 := by
  sorry

#check algebraic_expression_value

end algebraic_expression_value_l2410_241044


namespace power_zero_eq_one_l2410_241069

theorem power_zero_eq_one (x : ℝ) : x ^ (0 : ℕ) = 1 := by
  sorry

end power_zero_eq_one_l2410_241069


namespace negation_equivalence_l2410_241022

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 1 > 0) := by
sorry

end negation_equivalence_l2410_241022


namespace product_and_sum_inequality_l2410_241084

theorem product_and_sum_inequality (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) : 
  x * y ≥ 64 ∧ x + y ≥ 18 := by
sorry

end product_and_sum_inequality_l2410_241084


namespace certain_number_is_eleven_l2410_241052

theorem certain_number_is_eleven (n : ℕ) : 
  (0 < n) → (n < 11) → (n = 1) → (∃ k : ℕ, 18888 - n = 11 * k) → 
  ∀ m : ℕ, (∃ j : ℕ, 18888 - n = m * j) → m = 11 :=
by sorry

end certain_number_is_eleven_l2410_241052


namespace geometric_sequence_common_ratio_l2410_241037

theorem geometric_sequence_common_ratio
  (a : ℝ)
  (seq : ℕ → ℝ)
  (h_seq : ∀ n : ℕ, seq n = a + Real.log 3 / Real.log (2^(2^n)))
  : (∃ q : ℝ, ∀ n : ℕ, seq (n + 1) = q * seq n) ∧
    (∀ q : ℝ, (∀ n : ℕ, seq (n + 1) = q * seq n) → q = 1/3) :=
by sorry

end geometric_sequence_common_ratio_l2410_241037


namespace cable_car_travel_time_l2410_241085

/-- Represents the time in minutes to travel half the circular route -/
def travel_time : ℝ := 22.5

/-- Represents the number of cable cars on the circular route -/
def num_cars : ℕ := 80

/-- Represents the time interval in seconds between encounters with opposing cars -/
def encounter_interval : ℝ := 15

/-- Theorem stating that given the conditions, the travel time from A to B is 22.5 minutes -/
theorem cable_car_travel_time :
  ∀ (cars : ℕ) (interval : ℝ),
  cars = num_cars →
  interval = encounter_interval →
  travel_time = (cars : ℝ) * interval / (2 * 60) :=
by sorry

end cable_car_travel_time_l2410_241085


namespace lotus_flower_problem_l2410_241021

theorem lotus_flower_problem (x : ℚ) : 
  (x / 3 + x / 5 + x / 6 + x / 4 + 6 = x) → x = 120 := by
  sorry

end lotus_flower_problem_l2410_241021


namespace tan_half_product_l2410_241016

theorem tan_half_product (a b : Real) : 
  3 * (Real.cos a + Real.sin b) + 7 * (Real.cos a * Real.cos b + 1) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 3 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -3 := by
sorry

end tan_half_product_l2410_241016


namespace f_symmetric_about_origin_l2410_241050

/-- The function f(x) = 2sin(x)cos(x) is symmetric about the origin -/
theorem f_symmetric_about_origin :
  ∀ x : ℝ, (2 * Real.sin x * Real.cos x) = -(2 * Real.sin (-x) * Real.cos (-x)) := by
  sorry

end f_symmetric_about_origin_l2410_241050


namespace gcd_1248_1001_l2410_241076

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end gcd_1248_1001_l2410_241076


namespace problem_statement_l2410_241099

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  (x - y) * ((x + y)^2) = -5400 := by
sorry

end problem_statement_l2410_241099


namespace cat_arrangements_eq_six_l2410_241001

/-- The number of distinct arrangements of the letters in the word "CAT" -/
def cat_arrangements : ℕ :=
  Nat.factorial 3

theorem cat_arrangements_eq_six :
  cat_arrangements = 6 := by
  sorry

end cat_arrangements_eq_six_l2410_241001


namespace activities_alignment_period_l2410_241088

def activity_frequencies : List Nat := [6, 4, 16, 12, 8, 13, 17]

theorem activities_alignment_period :
  Nat.lcm (List.foldl Nat.lcm 1 activity_frequencies) = 10608 := by
  sorry

end activities_alignment_period_l2410_241088


namespace range_of_g_l2410_241082

theorem range_of_g (x : ℝ) : -1 ≤ Real.sin x ^ 3 + Real.cos x ^ 2 ∧ Real.sin x ^ 3 + Real.cos x ^ 2 ≤ 1 := by
  sorry

end range_of_g_l2410_241082


namespace vector_relationships_l2410_241067

/-- Given two vectors OA and OB in R², this theorem states the value of m in OB
    when OA is perpendicular to OB and when OA is parallel to OB. -/
theorem vector_relationships (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → OB = (3, m) →
  ((OA.1 * OB.1 + OA.2 * OB.2 = 0 → m = 3/2) ∧
   (∃ k : ℝ, OB = (k * OA.1, k * OA.2) → m = -6)) := by
  sorry

end vector_relationships_l2410_241067


namespace egg_groups_l2410_241096

/-- Given 16 eggs split into groups of 2, prove that the number of groups is 8 -/
theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (num_groups : ℕ) : 
  total_eggs = 16 → eggs_per_group = 2 → num_groups = total_eggs / eggs_per_group → num_groups = 8 := by
  sorry

end egg_groups_l2410_241096


namespace smallest_positive_period_monotonically_increasing_intervals_l2410_241004

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x * Real.cos x - 5 * Real.sqrt 3 * (Real.cos x)^2 + 5 * Real.sqrt 3 / 2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (- Real.pi / 12 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi)) :=
sorry

end smallest_positive_period_monotonically_increasing_intervals_l2410_241004


namespace floor_sum_equality_implies_integer_difference_l2410_241018

theorem floor_sum_equality_implies_integer_difference (a b c d : ℝ) 
  (h : ∀ (n : ℕ+), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) : 
  (∃ (z : ℤ), a + b = z) ∨ (∃ (z : ℤ), a - c = z) ∨ (∃ (z : ℤ), a - d = z) := by
  sorry

end floor_sum_equality_implies_integer_difference_l2410_241018


namespace equation_solution_range_l2410_241079

-- Define the equation
def equation (x a : ℝ) : Prop := Real.sqrt (x^2 - 1) = a*x - 2

-- Define the condition of having exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x, equation x a

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a ∈ Set.Icc (-Real.sqrt 5) (-1) ∪ Set.Ioc 1 (Real.sqrt 5))

-- Theorem statement
theorem equation_solution_range :
  ∀ a : ℝ, has_unique_solution a ↔ a_range a :=
sorry

end equation_solution_range_l2410_241079


namespace sum_of_roots_l2410_241054

theorem sum_of_roots (a β : ℝ) (ha : a^2 - 2*a = 1) (hβ : β^2 - 2*β - 1 = 0) (hneq : a ≠ β) :
  a + β = 2 := by
  sorry

end sum_of_roots_l2410_241054


namespace probability_at_least_three_same_l2410_241019

def num_dice : ℕ := 5
def num_sides : ℕ := 8

def total_outcomes : ℕ := num_sides ^ num_dice

def favorable_outcomes : ℕ :=
  -- Exactly 3 dice showing the same number
  (num_sides * (num_dice.choose 3) * (num_sides - 1)^2) +
  -- Exactly 4 dice showing the same number
  (num_sides * (num_dice.choose 4) * (num_sides - 1)) +
  -- All 5 dice showing the same number
  num_sides

theorem probability_at_least_three_same (h : favorable_outcomes = 4208) :
  (favorable_outcomes : ℚ) / total_outcomes = 1052 / 8192 := by
  sorry

end probability_at_least_three_same_l2410_241019


namespace simplify_expression_l2410_241010

theorem simplify_expression (a b : ℝ) : a - 4*(2*a - b) - 2*(a + 2*b) = -9*a := by
  sorry

end simplify_expression_l2410_241010


namespace sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l2410_241091

theorem sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths :
  ∀ x : ℚ, (Real.sqrt (8 * x) / Real.sqrt (4 * (x - 2)) = 3) → x = 18 / 7 := by
  sorry

end sqrt_ratio_eq_three_implies_x_eq_eighteen_sevenths_l2410_241091


namespace pythagorean_triple_3_4_5_l2410_241089

theorem pythagorean_triple_3_4_5 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 := by
  sorry

end pythagorean_triple_3_4_5_l2410_241089


namespace jerrys_average_increase_l2410_241025

theorem jerrys_average_increase :
  ∀ (initial_average : ℝ) (fourth_test_score : ℝ),
    initial_average = 90 →
    fourth_test_score = 98 →
    (3 * initial_average + fourth_test_score) / 4 = initial_average + 2 :=
by sorry

end jerrys_average_increase_l2410_241025


namespace remainder_equality_l2410_241027

theorem remainder_equality (P P' D R R' r r' : ℕ) 
  (h1 : P > P') 
  (h2 : R = P % D) 
  (h3 : R' = P' % D) 
  (h4 : r = (P * P') % D) 
  (h5 : r' = (R * R') % D) : 
  r = r' := by
sorry

end remainder_equality_l2410_241027


namespace fraction_value_l2410_241061

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end fraction_value_l2410_241061


namespace simons_flower_purchase_l2410_241013

def flower_purchase (pansy_price petunia_price hydrangea_price : ℝ)
                    (pansy_count petunia_count : ℕ)
                    (discount_rate : ℝ)
                    (change_received : ℝ) : Prop :=
  let total_before_discount := pansy_price * (pansy_count : ℝ) +
                               petunia_price * (petunia_count : ℝ) +
                               hydrangea_price
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let amount_paid := total_after_discount + change_received
  amount_paid = 50

theorem simons_flower_purchase :
  flower_purchase 2.5 1 12.5 5 5 0.1 23 := by
  sorry

end simons_flower_purchase_l2410_241013


namespace line_characteristics_l2410_241038

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The line y = -x - 3 -/
def line : Line := { slope := -1, y_intercept := -3 }

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on the line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- Check if the line passes through a quadrant -/
def Line.passes_through_quadrant (l : Line) (q : ℕ) : Prop :=
  ∃ (p : Point), l.contains p ∧
    match q with
    | 1 => p.x > 0 ∧ p.y > 0
    | 2 => p.x < 0 ∧ p.y > 0
    | 3 => p.x < 0 ∧ p.y < 0
    | 4 => p.x > 0 ∧ p.y < 0
    | _ => False

theorem line_characteristics :
  (line.passes_through_quadrant 2 ∧
   line.passes_through_quadrant 3 ∧
   line.passes_through_quadrant 4) ∧
  line.slope < 0 ∧
  line.contains { x := 0, y := -3 } ∧
  ¬ line.contains { x := 3, y := 0 } := by sorry

end line_characteristics_l2410_241038


namespace yearly_pet_feeding_cost_l2410_241040

-- Define the number of each type of pet
def num_geckos : ℕ := 3
def num_iguanas : ℕ := 2
def num_snakes : ℕ := 4

-- Define the monthly feeding cost for each type of pet
def gecko_cost : ℕ := 15
def iguana_cost : ℕ := 5
def snake_cost : ℕ := 10

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem yearly_pet_feeding_cost :
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) * months_per_year = 1140 :=
by sorry

end yearly_pet_feeding_cost_l2410_241040


namespace olivia_spent_89_dollars_l2410_241060

/-- Calculates the amount spent at a supermarket given initial amount, amount collected, and amount left --/
def amount_spent (initial : ℕ) (collected : ℕ) (left : ℕ) : ℕ :=
  initial + collected - left

theorem olivia_spent_89_dollars : amount_spent 100 148 159 = 89 := by
  sorry

end olivia_spent_89_dollars_l2410_241060


namespace contrapositive_equivalence_l2410_241009

theorem contrapositive_equivalence (a : ℝ) :
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end contrapositive_equivalence_l2410_241009


namespace local_call_cost_is_five_cents_l2410_241063

/-- Represents the cost structure and duration of Freddy's phone calls -/
structure CallData where
  local_duration : ℕ
  international_duration : ℕ
  international_cost_per_minute : ℕ
  total_cost_cents : ℕ

/-- Calculates the cost of a local call per minute -/
def local_call_cost_per_minute (data : CallData) : ℚ :=
  (data.total_cost_cents - data.international_duration * data.international_cost_per_minute) / data.local_duration

/-- Theorem stating that the local call cost per minute is 5 cents -/
theorem local_call_cost_is_five_cents (data : CallData) 
    (h1 : data.local_duration = 45)
    (h2 : data.international_duration = 31)
    (h3 : data.international_cost_per_minute = 25)
    (h4 : data.total_cost_cents = 1000) :
    local_call_cost_per_minute data = 5 := by
  sorry

#eval local_call_cost_per_minute {
  local_duration := 45,
  international_duration := 31,
  international_cost_per_minute := 25,
  total_cost_cents := 1000
}

end local_call_cost_is_five_cents_l2410_241063


namespace gcd_of_three_numbers_l2410_241014

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end gcd_of_three_numbers_l2410_241014


namespace expression_evaluation_l2410_241078

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3 * x + y / 3 ≠ 0) :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ := by
  sorry

end expression_evaluation_l2410_241078


namespace probability_three_one_is_five_ninths_l2410_241070

def total_balls : ℕ := 18
def blue_balls : ℕ := 10
def red_balls : ℕ := 8
def drawn_balls : ℕ := 4

def probability_three_one : ℚ :=
  let favorable_outcomes := Nat.choose blue_balls 3 * Nat.choose red_balls 1 +
                            Nat.choose blue_balls 1 * Nat.choose red_balls 3
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_three_one_is_five_ninths :
  probability_three_one = 5 / 9 := by
  sorry

end probability_three_one_is_five_ninths_l2410_241070


namespace intersection_product_sum_l2410_241086

/-- Given a line and a circle in R², prove that the sum of the products of the x-coordinate of one
    intersection point and the y-coordinate of the other equals 16. -/
theorem intersection_product_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ + y₁ = 5) →
  (x₂ + y₂ = 5) →
  (x₁^2 + y₁^2 = 16) →
  (x₂^2 + y₂^2 = 16) →
  x₁ * y₂ + x₂ * y₁ = 16 := by
  sorry

end intersection_product_sum_l2410_241086


namespace planted_field_fraction_l2410_241039

theorem planted_field_fraction (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_leg1 : a = 5) (h_leg2 : b = 12) (s : ℝ) (h_distance : 3 / 5 = s / (s + 3)) :
  (a * b / 2 - s^2) / (a * b / 2) = 13 / 40 := by
  sorry

end planted_field_fraction_l2410_241039


namespace ellipse_intersection_length_l2410_241077

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the conditions for the rhombus formed by the vertices
def rhombus_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * a * b = 2 * Real.sqrt 2 ∧ a^2 + b^2 = 3

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 2

-- Define the slope product condition
def slope_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / x₁) * (y₂ / x₂) = -1

-- Main theorem
theorem ellipse_intersection_length :
  ∀ (a b : ℝ),
  rhombus_condition a b →
  (∀ (x y : ℝ), ellipse_C a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (k x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C a b x₁ y₁ ∧ 
    ellipse_C a b x₂ y₂ ∧
    line_l k x₁ y₁ ∧ 
    line_l k x₂ y₂ ∧
    slope_product_condition x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 21 / 11) :=
sorry

end ellipse_intersection_length_l2410_241077


namespace help_desk_services_percentage_l2410_241028

theorem help_desk_services_percentage (total_hours software_hours help_user_hours : ℝ) 
  (h1 : total_hours = 68.33333333333333)
  (h2 : software_hours = 24)
  (h3 : help_user_hours = 17) :
  (total_hours - software_hours - help_user_hours) / total_hours * 100 = 40 := by
  sorry

end help_desk_services_percentage_l2410_241028


namespace fixed_point_exponential_l2410_241046

/-- The function f(x) = a^(x+1) - 2 passes through the point (-1, -1) for all a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) - 2
  f (-1) = -1 := by sorry

end fixed_point_exponential_l2410_241046


namespace tangent_circles_sum_l2410_241068

-- Define the circles w1 and w2
def w1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def w2 (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 115 = 0

-- Define the condition for a circle to be externally tangent to w1
def externally_tangent_w1 (cx cy r : ℝ) : Prop :=
  (cx + 4)^2 + (cy - 10)^2 = (r + 11)^2

-- Define the condition for a circle to be internally tangent to w2
def internally_tangent_w2 (cx cy r : ℝ) : Prop :=
  (cx - 6)^2 + (cy - 10)^2 = (7 - r)^2

-- Define the theorem
theorem tangent_circles_sum (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  (∃ (m : ℝ), m > 0 ∧ m^2 = p / q ∧
    (∃ (cx cy r : ℝ), cy = m * cx ∧
      externally_tangent_w1 cx cy r ∧
      internally_tangent_w2 cx cy r) ∧
    (∀ (a : ℝ), a > 0 → a < m →
      ¬∃ (cx cy r : ℝ), cy = a * cx ∧
        externally_tangent_w1 cx cy r ∧
        internally_tangent_w2 cx cy r)) →
  p + q = 181 := by sorry

end tangent_circles_sum_l2410_241068


namespace count_flippy_divisible_by_25_is_24_l2410_241087

/-- A flippy number alternates between two distinct digits. -/
def is_flippy (n : ℕ) : Prop := sorry

/-- Checks if a number is six digits long. -/
def is_six_digit (n : ℕ) : Prop := sorry

/-- Counts the number of six-digit flippy numbers divisible by 25. -/
def count_flippy_divisible_by_25 : ℕ := sorry

/-- Theorem stating that the count of six-digit flippy numbers divisible by 25 is 24. -/
theorem count_flippy_divisible_by_25_is_24 : count_flippy_divisible_by_25 = 24 := by sorry

end count_flippy_divisible_by_25_is_24_l2410_241087


namespace smallest_number_of_cubes_for_given_box_l2410_241062

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let sideLengthOfCube := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / sideLengthOfCube) * (box.width / sideLengthOfCube) * (box.depth / sideLengthOfCube)

/-- The theorem stating that for a box with given dimensions, 
    the smallest number of identical cubes that can fill it completely is 84 -/
theorem smallest_number_of_cubes_for_given_box : 
  smallestNumberOfCubes ⟨49, 42, 14⟩ = 84 := by
  sorry

end smallest_number_of_cubes_for_given_box_l2410_241062


namespace sum_of_coefficients_l2410_241053

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem sum_of_coefficients :
  (∀ x, f (x + 2) = 2 * x^3 + 5 * x^2 + 3 * x + 6) →
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (∃ a b c d : ℝ, (∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧ a + b + c + d = 6) :=
by sorry

end sum_of_coefficients_l2410_241053


namespace cross_product_example_l2410_241042

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example : 
  cross_product (3, 2, -1) (-2, 4, 6) = (16, -16, 16) := by
  sorry

end cross_product_example_l2410_241042


namespace triangle_side_difference_l2410_241075

theorem triangle_side_difference (x : ℤ) : 
  (∀ y : ℤ, 3 ≤ y ∧ y ≤ 17 → (y + 8 > 10 ∧ y + 10 > 8 ∧ 8 + 10 > y)) →
  (∀ z : ℤ, z < 3 ∨ z > 17 → ¬(z + 8 > 10 ∧ z + 10 > 8 ∧ 8 + 10 > z)) →
  (17 - 3 : ℤ) = 14 := by
sorry

end triangle_side_difference_l2410_241075


namespace no_solution_factorial_equation_l2410_241036

theorem no_solution_factorial_equation :
  ∀ (k m : ℕ+), k.val.factorial + 48 ≠ 48 * (k.val + 1) ^ m.val := by
  sorry

end no_solution_factorial_equation_l2410_241036


namespace same_remainder_for_282_l2410_241043

theorem same_remainder_for_282 : ∃ r : ℕ, r < 9 ∧ r < 31 ∧ 282 % 31 = r ∧ 282 % 9 = r ∧ r = 3 := by
  sorry

end same_remainder_for_282_l2410_241043


namespace problem_solution_l2410_241065

theorem problem_solution (x y : ℝ) 
  (eq1 : |x| + x + y = 15)
  (eq2 : x + |y| - y = 9)
  (eq3 : y = 3*x - 7) : 
  x + y = 53/5 := by
sorry

end problem_solution_l2410_241065


namespace b_speed_is_20_l2410_241057

/-- The speed of person A in km/h -/
def speed_a : ℝ := 10

/-- The head start time of person A in hours -/
def head_start : ℝ := 5

/-- The total distance traveled when B catches up with A in km -/
def total_distance : ℝ := 100

/-- The speed of person B in km/h -/
def speed_b : ℝ := 20

theorem b_speed_is_20 :
  speed_b = (total_distance - speed_a * head_start) / (total_distance / speed_a - head_start) :=
by sorry

end b_speed_is_20_l2410_241057


namespace card_difference_l2410_241000

theorem card_difference (janet brenda mara : ℕ) : 
  janet > brenda →
  mara = 2 * janet →
  janet + brenda + mara = 211 →
  mara = 150 - 40 →
  janet - brenda = 9 := by
sorry

end card_difference_l2410_241000


namespace terminal_side_in_second_quadrant_l2410_241055

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom cos_α : Real.cos α = -1/5
axiom sin_α : Real.sin α = 2 * Real.sqrt 6 / 5

-- Define the second quadrant
def second_quadrant (θ : Real) : Prop :=
  Real.cos θ < 0 ∧ Real.sin θ > 0

-- Theorem to prove
theorem terminal_side_in_second_quadrant : second_quadrant α := by
  sorry

end terminal_side_in_second_quadrant_l2410_241055


namespace product_of_numbers_l2410_241072

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end product_of_numbers_l2410_241072


namespace third_game_difference_l2410_241058

/-- The number of people who watched the second game -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The total number of people who watched the games last week -/
def last_week_total : ℕ := 200

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := last_week_total + 35

/-- The number of people who watched the third game -/
def third_game_viewers : ℕ := this_week_total - (first_game_viewers + second_game_viewers)

theorem third_game_difference : 
  third_game_viewers - second_game_viewers = 15 := by sorry

end third_game_difference_l2410_241058


namespace boys_share_is_14_l2410_241080

/-- The amount of money each boy makes from selling shrimp -/
def boys_share (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (per_shrimp : ℕ) : ℚ :=
  let austin_shrimp := victor_shrimp - austin_diff
  let victor_austin_total := victor_shrimp + austin_shrimp
  let brian_shrimp := victor_austin_total / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_money := (total_shrimp / per_shrimp) * price
  total_money / 3

/-- Theorem stating that each boy's share is $14 given the problem conditions -/
theorem boys_share_is_14 :
  boys_share 26 8 7 11 = 14 := by
  sorry

end boys_share_is_14_l2410_241080


namespace units_digit_of_7_62_l2410_241026

theorem units_digit_of_7_62 : ∃ n : ℕ, 7^62 ≡ 9 [MOD 10] :=
by
  -- We'll use n = 9 to prove the existence
  use 9
  -- The proof goes here
  sorry

end units_digit_of_7_62_l2410_241026


namespace percent_calculation_l2410_241012

theorem percent_calculation (x y : ℝ) (h : x = 120.5 ∧ y = 80.75) :
  (x / y) * 100 = 149.26 := by
  sorry

end percent_calculation_l2410_241012


namespace revenue_change_l2410_241007

theorem revenue_change 
  (P : ℝ) 
  (N : ℝ) 
  (price_decrease : ℝ) 
  (sales_increase : ℝ) 
  (h1 : price_decrease = 0.2) 
  (h2 : sales_increase = 0.6) 
  : (1 - price_decrease) * (1 + sales_increase) * (P * N) = 1.28 * (P * N) := by
sorry

end revenue_change_l2410_241007


namespace factorization_1_factorization_2_factorization_3_l2410_241094

-- Define variables
variable (a b m x y : ℝ)

-- Theorem 1
theorem factorization_1 : 3*m - 3*y + a*m - a*y = (m - y) * (3 + a) := by sorry

-- Theorem 2
theorem factorization_2 : a^2*x + a^2*y + b^2*x + b^2*y = (x + y) * (a^2 + b^2) := by sorry

-- Theorem 3
theorem factorization_3 : a^2 + 2*a*b + b^2 - 1 = (a + b + 1) * (a + b - 1) := by sorry

end factorization_1_factorization_2_factorization_3_l2410_241094


namespace min_value_of_sum_squares_l2410_241015

theorem min_value_of_sum_squares (a b c m : ℝ) 
  (sum_eq_one : a + b + c = 1) 
  (m_def : m = a^2 + b^2 + c^2) : 
  m ≥ 1/3 ∧ ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 + b^2 + c^2 = 1/3 :=
sorry

end min_value_of_sum_squares_l2410_241015
