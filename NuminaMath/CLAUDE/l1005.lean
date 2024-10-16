import Mathlib

namespace NUMINAMATH_CALUDE_lollipops_for_class_l1005_100585

/-- Calculates the number of lollipops given away based on the number of people and the lollipop distribution rate. -/
def lollipops_given (total_people : ℕ) (people_per_lollipop : ℕ) : ℕ :=
  total_people / people_per_lollipop

/-- Proves that given 60 people and 1 lollipop per 5 people, the teacher gives away 12 lollipops. -/
theorem lollipops_for_class (total_people : ℕ) (people_per_lollipop : ℕ) 
    (h1 : total_people = 60) 
    (h2 : people_per_lollipop = 5) : 
  lollipops_given total_people people_per_lollipop = 12 := by
  sorry

#eval lollipops_given 60 5  -- Expected output: 12

end NUMINAMATH_CALUDE_lollipops_for_class_l1005_100585


namespace NUMINAMATH_CALUDE_wheel_probabilities_l1005_100551

theorem wheel_probabilities :
  ∀ (p_C p_D : ℚ),
    (1 : ℚ)/3 + (1 : ℚ)/4 + p_C + p_D = 1 →
    p_C = 2 * p_D →
    p_C = (5 : ℚ)/18 ∧ p_D = (5 : ℚ)/36 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_probabilities_l1005_100551


namespace NUMINAMATH_CALUDE_no_simultaneous_extrema_l1005_100514

/-- A partition of rational numbers -/
structure RationalPartition where
  M : Set ℚ
  N : Set ℚ
  M_nonempty : M.Nonempty
  N_nonempty : N.Nonempty
  union_eq_rat : M ∪ N = Set.univ
  intersection_empty : M ∩ N = ∅
  M_lt_N : ∀ m ∈ M, ∀ n ∈ N, m < n

/-- Theorem stating that in a partition of rationals, M cannot have a maximum and N cannot have a minimum simultaneously -/
theorem no_simultaneous_extrema (p : RationalPartition) :
  ¬(∃ (max_M : ℚ), max_M ∈ p.M ∧ ∀ m ∈ p.M, m ≤ max_M) ∨
  ¬(∃ (min_N : ℚ), min_N ∈ p.N ∧ ∀ n ∈ p.N, min_N ≤ n) :=
sorry

end NUMINAMATH_CALUDE_no_simultaneous_extrema_l1005_100514


namespace NUMINAMATH_CALUDE_empire_state_building_height_l1005_100552

/-- The height of the Empire State Building to the top floor, in feet -/
def height_to_top_floor : ℕ := 1250

/-- The height of the antenna spire, in feet -/
def antenna_height : ℕ := 204

/-- The total height of the Empire State Building, in feet -/
def total_height : ℕ := height_to_top_floor + antenna_height

/-- Theorem stating that the total height of the Empire State Building is 1454 feet -/
theorem empire_state_building_height : total_height = 1454 := by
  sorry

end NUMINAMATH_CALUDE_empire_state_building_height_l1005_100552


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1005_100526

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5 / (a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1005_100526


namespace NUMINAMATH_CALUDE_original_prices_calculation_l1005_100520

/-- Given the price increases and final prices of three items, prove their original prices. -/
theorem original_prices_calculation (computer_increase : ℝ) (tv_increase : ℝ) (fridge_increase : ℝ)
  (computer_final : ℝ) (tv_final : ℝ) (fridge_final : ℝ)
  (h1 : computer_increase = 0.30)
  (h2 : tv_increase = 0.20)
  (h3 : fridge_increase = 0.15)
  (h4 : computer_final = 377)
  (h5 : tv_final = 720)
  (h6 : fridge_final = 1150) :
  ∃ (computer_original tv_original fridge_original : ℝ),
    computer_original = 290 ∧
    tv_original = 600 ∧
    fridge_original = 1000 ∧
    computer_final = computer_original * (1 + computer_increase) ∧
    tv_final = tv_original * (1 + tv_increase) ∧
    fridge_final = fridge_original * (1 + fridge_increase) :=
by sorry

end NUMINAMATH_CALUDE_original_prices_calculation_l1005_100520


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1005_100544

/-- The number of intersection points between a line and a hyperbola -/
theorem line_hyperbola_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃! p : ℝ × ℝ, 
    (p.2 = (b / a) * p.1 + 3) ∧ 
    ((p.1^2 / a^2) - (p.2^2 / b^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1005_100544


namespace NUMINAMATH_CALUDE_runner_problem_l1005_100565

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 8) → (8 - 20 / v = 4) := by sorry

end NUMINAMATH_CALUDE_runner_problem_l1005_100565


namespace NUMINAMATH_CALUDE_absolute_value_of_squared_negative_l1005_100554

theorem absolute_value_of_squared_negative : |(-2)^2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_squared_negative_l1005_100554


namespace NUMINAMATH_CALUDE_prob_two_odd_chips_l1005_100561

-- Define the set of numbers on the chips
def ChipNumbers : Set ℕ := {1, 2, 3, 4}

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop := n % 2 = 1

-- Define the probability of drawing an odd-numbered chip from one box
def probOddFromOneBox : ℚ := (2 : ℚ) / 4

-- Theorem statement
theorem prob_two_odd_chips :
  (probOddFromOneBox * probOddFromOneBox) = (1 : ℚ) / 4 :=
sorry

end NUMINAMATH_CALUDE_prob_two_odd_chips_l1005_100561


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1005_100556

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1005_100556


namespace NUMINAMATH_CALUDE_regression_prediction_at_2_l1005_100537

/-- Represents a linear regression model -/
structure LinearRegression where
  b : ℝ
  c : ℝ := 0.2

/-- Calculates the y value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.c

/-- Theorem: Given the conditions, the predicted y value when x = 2 is 2.6 -/
theorem regression_prediction_at_2 
  (model : LinearRegression)
  (h₁ : predict model 4 = 5) -- condition: ȳ = 5 when x̄ = 4
  (h₂ : model.c = 0.2) -- condition: intercept is 0.2
  : predict model 2 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_regression_prediction_at_2_l1005_100537


namespace NUMINAMATH_CALUDE_product_and_sum_of_consecutive_integers_l1005_100503

theorem product_and_sum_of_consecutive_integers : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (a > 0) ∧ 
    (a * b = 198) ∧ 
    (c * d * e = 198) ∧ 
    (a + b + c + d + e = 39) := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_consecutive_integers_l1005_100503


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l1005_100510

/-- Represents the number of socks of each color --/
structure SockCount where
  blue : ℕ
  black : ℕ
  red : ℕ
  white : ℕ

/-- Conditions for Joseph's sock collection --/
def josephSocks : SockCount → Prop :=
  fun s => s.blue = s.black + 6 ∧
           s.red + 2 = s.white ∧
           s.red = 6 ∧
           s.blue + s.black + s.red + s.white = 28

/-- The ratio of blue to red socks is 7:3 --/
theorem blue_to_red_ratio (s : SockCount) (h : josephSocks s) :
  s.blue * 3 = s.red * 7 :=
by sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l1005_100510


namespace NUMINAMATH_CALUDE_exercise_book_price_l1005_100584

/-- The price of an exercise book in yuan -/
def price_per_book : ℚ := 0.55

/-- The number of books Xiaoming took -/
def xiaoming_books : ℕ := 8

/-- The number of books Xiaohong took -/
def xiaohong_books : ℕ := 12

/-- The amount Xiaohong gave to Xiaoming in yuan -/
def amount_given : ℚ := 1.1

theorem exercise_book_price :
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaoming_books : ℚ) * price_per_book + amount_given / 2 ∧
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaohong_books : ℚ) * price_per_book - amount_given / 2 :=
sorry

end NUMINAMATH_CALUDE_exercise_book_price_l1005_100584


namespace NUMINAMATH_CALUDE_factor_sum_l1005_100577

theorem factor_sum (P Q : ℚ) : 
  (∃ b c : ℚ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^3 + Q*X^2 + 45*X - 14) →
  P + Q = 260/7 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1005_100577


namespace NUMINAMATH_CALUDE_expression_simplification_l1005_100588

theorem expression_simplification :
  -2 * Real.sqrt 2 + 2^(-(1/2 : ℝ)) + 1 / (Real.sqrt 2 + 1) + (Real.sqrt 2 - 1)^(0 : ℝ) = -(Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1005_100588


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1005_100522

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

theorem complement_A_intersect_B : (Aᶜ ∩ B) = {7, 9} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1005_100522


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1005_100502

theorem min_value_quadratic (x y : ℝ) : 
  3 * x^2 + 2 * x * y + y^2 - 6 * x + 2 * y + 8 ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1005_100502


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l1005_100578

def S : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (S \ M) ∩ (S \ N) = ∅ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l1005_100578


namespace NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_one_l1005_100579

-- Define the sets P and Q
def P (k : ℝ) : Set ℝ := {y | y = k}
def Q (a : ℝ) : Set ℝ := {y | ∃ x : ℝ, y = a^x + 1}

-- State the theorem
theorem intersection_empty_implies_k_leq_one
  (k : ℝ) (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  (P k ∩ Q a = ∅) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_k_leq_one_l1005_100579


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1005_100543

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 5) :
  (1 / x + 4 / y + 9 / z) ≥ 36 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1005_100543


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1005_100591

theorem complex_fraction_equality : Complex.I / (1 + Complex.I) = (1 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1005_100591


namespace NUMINAMATH_CALUDE_quadratic_above_x_axis_condition_l1005_100504

/-- A quadratic function f(x) = ax^2 + bx + c is always above the x-axis -/
def AlwaysAboveXAxis (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c > 0

/-- The discriminant of a quadratic function f(x) = ax^2 + bx + c is negative -/
def NegativeDiscriminant (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem quadratic_above_x_axis_condition (a b c : ℝ) :
  ¬(NegativeDiscriminant a b c ↔ AlwaysAboveXAxis a b c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_above_x_axis_condition_l1005_100504


namespace NUMINAMATH_CALUDE_rook_placements_l1005_100599

def chessboard_size : ℕ := 8
def num_rooks : ℕ := 3

theorem rook_placements : 
  (chessboard_size.choose num_rooks) * (chessboard_size * (chessboard_size - 1) * (chessboard_size - 2)) = 18816 :=
by sorry

end NUMINAMATH_CALUDE_rook_placements_l1005_100599


namespace NUMINAMATH_CALUDE_three_liters_to_pints_l1005_100574

-- Define the conversion rate from liters to pints
def liters_to_pints (liters : ℝ) : ℝ := 2.16 * liters

-- Theorem statement
theorem three_liters_to_pints : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, 
  |x - 3| < δ → |liters_to_pints x - 6.5| < ε :=
sorry

end NUMINAMATH_CALUDE_three_liters_to_pints_l1005_100574


namespace NUMINAMATH_CALUDE_max_d_value_l1005_100593

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ e % 2 = 0

def number_value (d e : ℕ) : ℕ :=
  505220 + d * 1000 + e

theorem max_d_value :
  ∃ (d : ℕ), d = 8 ∧
  ∀ (d' e : ℕ), is_valid_number d' e →
  number_value d' e % 22 = 0 →
  d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1005_100593


namespace NUMINAMATH_CALUDE_apple_count_theorem_l1005_100559

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_count_theorem : 
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l1005_100559


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1005_100598

theorem arithmetic_sequence_product (b : ℕ → ℕ) : 
  (∀ n, b n < b (n + 1)) →  -- increasing sequence
  (∃ d : ℕ, ∀ n, b (n + 1) = b n + d) →  -- arithmetic sequence
  b 3 * b 4 = 72 → 
  b 2 * b 5 = 70 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1005_100598


namespace NUMINAMATH_CALUDE_no_prime_10101_base_n_l1005_100513

theorem no_prime_10101_base_n : ¬ ∃ (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_10101_base_n_l1005_100513


namespace NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l1005_100564

theorem min_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) : 
  x^3 * y^2 * z ≥ 1/108 := by
sorry

theorem min_value_achieved (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 1/x₀ + 1/y₀ + 1/z₀ = 6 ∧ x₀^3 * y₀^2 * z₀ = 1/108 := by
sorry

end NUMINAMATH_CALUDE_min_value_product_min_value_achieved_l1005_100564


namespace NUMINAMATH_CALUDE_stevens_weight_l1005_100531

/-- Given that Danny weighs 40 kg and Steven weighs 20% more than Danny, 
    prove that Steven's weight is 48 kg. -/
theorem stevens_weight (danny_weight : ℝ) (steven_weight : ℝ) 
    (h1 : danny_weight = 40)
    (h2 : steven_weight = danny_weight * 1.2) : 
  steven_weight = 48 := by
  sorry

end NUMINAMATH_CALUDE_stevens_weight_l1005_100531


namespace NUMINAMATH_CALUDE_jeanne_ticket_purchase_l1005_100538

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets (ferris_wheel_cost roller_coaster_cost bumper_cars_cost current_tickets : ℕ) : ℕ :=
  ferris_wheel_cost + roller_coaster_cost + bumper_cars_cost - current_tickets

theorem jeanne_ticket_purchase : additional_tickets 5 4 4 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_ticket_purchase_l1005_100538


namespace NUMINAMATH_CALUDE_binary_101_is_5_l1005_100569

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₂ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₂ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_is_5_l1005_100569


namespace NUMINAMATH_CALUDE_xyz_value_l1005_100589

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) : 
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1005_100589


namespace NUMINAMATH_CALUDE_unique_solution_floor_product_l1005_100518

theorem unique_solution_floor_product : 
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 45 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_product_l1005_100518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1005_100521

/-- Given an arithmetic sequence {a_n} where a_1 = 2, a_2 = 4, and a_3 = 6, 
    prove that the fourth term a_4 = 8. -/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) 
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l1005_100521


namespace NUMINAMATH_CALUDE_roys_pen_ratio_l1005_100568

/-- Proves that the ratio of black pens to blue pens is 2:1 given the conditions of Roy's pen collection --/
theorem roys_pen_ratio :
  ∀ (blue black red : ℕ),
    blue = 2 →
    red = 2 * black - 2 →
    blue + black + red = 12 →
    black / blue = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_roys_pen_ratio_l1005_100568


namespace NUMINAMATH_CALUDE_time_to_fill_cistern_l1005_100542

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 7)
  (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 77 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_cistern_l1005_100542


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l1005_100500

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_9 = 17 -/
theorem ninth_term_of_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : a 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l1005_100500


namespace NUMINAMATH_CALUDE_fifth_term_value_l1005_100567

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n / 2 * (a 1 + a n)

/-- Theorem: If S₆ = 12 and a₂ = 5 in an arithmetic sequence, then a₅ = -1 -/
theorem fifth_term_value (seq : ArithmeticSequence) 
  (sum_6 : seq.S 6 = 12) 
  (second_term : seq.a 2 = 5) : 
  seq.a 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1005_100567


namespace NUMINAMATH_CALUDE_bob_gardening_project_cost_l1005_100592

/-- Calculates the total cost of a gardening project -/
def gardening_project_cost (num_rose_bushes : ℕ) (cost_per_rose_bush : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (num_days : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * hours_per_day * num_days +
  soil_volume * soil_cost_per_unit

/-- The total cost of Bob's gardening project is $4100 -/
theorem bob_gardening_project_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_bob_gardening_project_cost_l1005_100592


namespace NUMINAMATH_CALUDE_lucas_payment_l1005_100515

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_3_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_earned := total_windows * payment_per_window
  let deductions := (days_taken / 3) * deduction_per_3_days
  total_earned - deductions

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 6 = 16 := by
  sorry

#eval calculate_payment 3 3 2 1 6

end NUMINAMATH_CALUDE_lucas_payment_l1005_100515


namespace NUMINAMATH_CALUDE_home_appliances_promotion_l1005_100530

theorem home_appliances_promotion (salespersons technicians : ℕ) 
  (h1 : salespersons = 5)
  (h2 : technicians = 4)
  (h3 : salespersons + technicians = 9) :
  (Nat.choose 9 3) - (Nat.choose 5 3) - (Nat.choose 4 3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_home_appliances_promotion_l1005_100530


namespace NUMINAMATH_CALUDE_range_of_a_l1005_100590

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set B defined by the distance from a point a -/
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (h : A ∪ B a = Set.univ) : a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1005_100590


namespace NUMINAMATH_CALUDE_prob_win_is_four_sevenths_l1005_100562

/-- The probability of Lola losing a match -/
def prob_lose : ℚ := 3/7

/-- The theorem stating that the probability of Lola winning a match is 4/7 -/
theorem prob_win_is_four_sevenths :
  let prob_win := 1 - prob_lose
  prob_win = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_win_is_four_sevenths_l1005_100562


namespace NUMINAMATH_CALUDE_import_tax_problem_l1005_100532

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount. -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the specific values in the problem. -/
theorem import_tax_problem :
  let total_value : ℚ := 2560
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 109.2
  sorry


end NUMINAMATH_CALUDE_import_tax_problem_l1005_100532


namespace NUMINAMATH_CALUDE_ending_number_proof_l1005_100516

/-- The ending number of the range [100, n] where the average of integers
    in [100, n] is 100 greater than the average of integers in [50, 250]. -/
def ending_number : ℕ :=
  400

theorem ending_number_proof :
  ∃ (n : ℕ),
    n ≥ 100 ∧
    (n + 100) / 2 = (250 + 50) / 2 + 100 ∧
    n = ending_number :=
by sorry

end NUMINAMATH_CALUDE_ending_number_proof_l1005_100516


namespace NUMINAMATH_CALUDE_circle_intersection_trajectory_l1005_100524

/-- Given two circles with centers at (a₁, 0) and (a₂, 0), both passing through (1, 0),
    intersecting the positive y-axis at (0, y₁) and (0, y₂) respectively,
    prove that the trajectory of (1/a₁, 1/a₂) is a straight line when ln y₁ + ln y₂ = 0 -/
theorem circle_intersection_trajectory (a₁ a₂ y₁ y₂ : ℝ) 
    (h1 : (1 - a₁)^2 = a₁^2 + y₁^2)
    (h2 : (1 - a₂)^2 = a₂^2 + y₂^2)
    (h3 : Real.log y₁ + Real.log y₂ = 0) :
    ∃ (m b : ℝ), ∀ (x y : ℝ), (x = 1/a₁ ∧ y = 1/a₂) → y = m*x + b :=
  sorry


end NUMINAMATH_CALUDE_circle_intersection_trajectory_l1005_100524


namespace NUMINAMATH_CALUDE_b_age_is_twelve_l1005_100586

/-- Given three people a, b, and c, where a is two years older than b, b is twice as old as c, 
    and the sum of their ages is 32, prove that b is 12 years old. -/
theorem b_age_is_twelve (a b c : ℕ) 
    (h1 : a = b + 2) 
    (h2 : b = 2 * c) 
    (h3 : a + b + c = 32) : 
  b = 12 := by sorry

end NUMINAMATH_CALUDE_b_age_is_twelve_l1005_100586


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l1005_100570

theorem abs_eq_sqrt_square (x : ℝ) : |x - 1| = Real.sqrt ((x - 1)^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l1005_100570


namespace NUMINAMATH_CALUDE_line_parallel_to_intersection_l1005_100548

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersect_planes : Plane → Plane → Line)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_intersection
  (m n : Line) (α β : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : subset_line_plane m β)
  (h3 : intersect_planes α β = n) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_intersection_l1005_100548


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1005_100582

theorem x_range_for_inequality (x : ℝ) : 
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2 * x - 1 > m * (x^2 - 1)) →
  ((-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1005_100582


namespace NUMINAMATH_CALUDE_evaluate_expression_l1005_100580

theorem evaluate_expression (b : ℚ) (h : b = 4/3) :
  (6*b^2 - 17*b + 8) * (3*b - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1005_100580


namespace NUMINAMATH_CALUDE_equation_root_greater_than_zero_l1005_100509

theorem equation_root_greater_than_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (3*x - 1)/(x - 3) = a/(3 - x) - 1) → a = -8 := by
sorry

end NUMINAMATH_CALUDE_equation_root_greater_than_zero_l1005_100509


namespace NUMINAMATH_CALUDE_max_value_fraction_l1005_100563

theorem max_value_fraction (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x = b/(a-3) → x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1005_100563


namespace NUMINAMATH_CALUDE_S_max_at_n_max_l1005_100519

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n.val^2 + 8*n.val

/-- n_max is the value of n at which S_n attains its maximum value -/
def n_max : ℕ+ := 4

theorem S_max_at_n_max :
  ∀ n : ℕ+, S n ≤ S n_max :=
sorry

end NUMINAMATH_CALUDE_S_max_at_n_max_l1005_100519


namespace NUMINAMATH_CALUDE_direct_sort_5_rounds_l1005_100560

def initial_sequence : List Nat := [49, 38, 65, 97, 76, 13, 27]

def direct_sort_step (l : List Nat) : List Nat :=
  match l with
  | [] => []
  | _ => let max := l.maximum? |>.getD 0
         max :: (l.filter (· ≠ max))

def direct_sort (l : List Nat) (n : Nat) : List Nat :=
  match n with
  | 0 => l
  | n + 1 => direct_sort (direct_sort_step l) n

theorem direct_sort_5_rounds :
  direct_sort initial_sequence 5 = [97, 76, 65, 49, 38, 13, 27] := by
  sorry

end NUMINAMATH_CALUDE_direct_sort_5_rounds_l1005_100560


namespace NUMINAMATH_CALUDE_binomial_n_value_l1005_100527

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_n_value (ξ : BinomialRV) 
  (h_exp : expectation ξ = 6)
  (h_var : variance ξ = 3) : 
  ξ.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_value_l1005_100527


namespace NUMINAMATH_CALUDE_height_inequality_l1005_100587

theorem height_inequality (a b : ℕ) (m : ℝ) (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : m = (a * b : ℝ) / Real.sqrt ((a^2 + b^2 : ℕ) : ℝ)) :
  m ≤ Real.sqrt (((a^a * b^b : ℕ) : ℝ)^(1 / (a + b : ℝ))) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_height_inequality_l1005_100587


namespace NUMINAMATH_CALUDE_a_7_equals_neg_3_l1005_100534

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem stating that a₇ = -3 in the given geometric sequence -/
theorem a_7_equals_neg_3 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 ^ 2 + 2016 * a 5 + 9 = 0 →
  a 9 ^ 2 + 2016 * a 9 + 9 = 0 →
  a 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_7_equals_neg_3_l1005_100534


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l1005_100597

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (hf_cont : Continuous f)
variable (hf_diff : Differentiable ℝ f)
variable (hf_pos : ∀ x, x * (deriv f x) + f x > 0)

-- Define the function g
def g (x : ℝ) := x * f x + 1

-- State the theorem
theorem no_zeros_of_g :
  ∀ x > 0, g f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l1005_100597


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_range_for_positive_f_l1005_100517

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} := by sorry

-- Part 2
theorem a_range_for_positive_f :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x > 0) → -5/2 < a ∧ a < -2 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_a_range_for_positive_f_l1005_100517


namespace NUMINAMATH_CALUDE_problem_statement_l1005_100550

theorem problem_statement :
  (∀ x : ℝ, x^4 - x^3 - x + 1 ≥ 0) ∧
  (1 + 1 + 1 = 3 ∧ 1^3 + 1^3 + 1^3 = 1^4 + 1^4 + 1^4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1005_100550


namespace NUMINAMATH_CALUDE_initial_pencils_l1005_100535

/-- Given that a person:
  - starts with an initial number of pencils
  - gives away 18 pencils
  - buys 22 more pencils
  - ends up with 43 pencils
  This theorem proves that the initial number of pencils was 39. -/
theorem initial_pencils (initial : ℕ) : 
  initial - 18 + 22 = 43 → initial = 39 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_l1005_100535


namespace NUMINAMATH_CALUDE_sector_area_of_ring_l1005_100546

/-- The area of a 60° sector of the ring between two concentric circles with radii 12 and 8 -/
theorem sector_area_of_ring (π : ℝ) : 
  let r₁ : ℝ := 12  -- radius of larger circle
  let r₂ : ℝ := 8   -- radius of smaller circle
  let ring_area : ℝ := π * (r₁^2 - r₂^2)
  let sector_angle : ℝ := 60
  let full_angle : ℝ := 360
  let sector_area : ℝ := (sector_angle / full_angle) * ring_area
  sector_area = (40 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_of_ring_l1005_100546


namespace NUMINAMATH_CALUDE_number_equation_and_interval_l1005_100507

theorem number_equation_and_interval : ∃ (x : ℝ), 
  x = (1 / x) * x^2 + 3 ∧ x = 4 ∧ 3 < x ∧ x ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_and_interval_l1005_100507


namespace NUMINAMATH_CALUDE_proposition_and_converse_l1005_100541

theorem proposition_and_converse (a b : ℝ) :
  (a + b ≥ 2 → max a b ≥ 1) ∧
  ¬(max a b ≥ 1 → a + b ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l1005_100541


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l1005_100596

theorem sqrt_expression_simplification :
  Real.sqrt 5 - Real.sqrt 20 + Real.sqrt 90 / Real.sqrt 2 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l1005_100596


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l1005_100583

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 2 * x^2 - 4 * x + 4 = -x^2 - 2 * x + 4 → x = a ∨ x = c) ∧
  c ≥ a ∧
  c - a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l1005_100583


namespace NUMINAMATH_CALUDE_f_zero_implies_a_bound_l1005_100508

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - (Real.log x) / x + a

theorem f_zero_implies_a_bound (a : ℝ) :
  (∃ x > 0, f x a = 0) →
  a ≤ Real.exp 2 + 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_implies_a_bound_l1005_100508


namespace NUMINAMATH_CALUDE_maria_car_rental_cost_l1005_100557

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, number of days, and miles driven. -/
def carRentalCost (dailyRate perMileRate : ℚ) (days miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Theorem stating that Maria's car rental cost is $275 given the specified conditions. -/
theorem maria_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_maria_car_rental_cost_l1005_100557


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1005_100575

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 15) 
  (h2 : x*y + y*z + z*x = 34) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 1845 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1005_100575


namespace NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l1005_100540

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem trailing_zeros_50_factorial :
  trailingZeros 50 = 12 := by
  sorry


end NUMINAMATH_CALUDE_trailing_zeros_50_factorial_l1005_100540


namespace NUMINAMATH_CALUDE_number_puzzle_l1005_100511

theorem number_puzzle (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1005_100511


namespace NUMINAMATH_CALUDE_perimeter_of_circular_sector_problem_perimeter_l1005_100553

/-- The perimeter of a region formed by two radii and an arc of a circle -/
theorem perimeter_of_circular_sector (r : ℝ) (arc_fraction : ℝ) : 
  r > 0 → 
  0 < arc_fraction → 
  arc_fraction ≤ 1 → 
  2 * r + arc_fraction * (2 * π * r) = 2 * r + 2 * arc_fraction * π * r :=
by sorry

/-- The perimeter of the specific region in the problem -/
theorem problem_perimeter : 
  let r : ℝ := 8
  let arc_fraction : ℝ := 5/6
  2 * r + arc_fraction * (2 * π * r) = 16 + (40/3) * π :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_circular_sector_problem_perimeter_l1005_100553


namespace NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1005_100595

/-- The maximum number of identical pieces a cake can be divided into with a given number of cuts. -/
def max_cake_pieces (cuts : ℕ) : ℕ := 2^cuts

/-- Theorem: The maximum number of identical pieces a cake can be divided into with 3 cuts is 8. -/
theorem max_pieces_with_three_cuts :
  max_cake_pieces 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_with_three_cuts_l1005_100595


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1005_100512

theorem triangle_angle_problem (A B C : Real) (BC AC : Real) :
  BC = Real.sqrt 3 →
  AC = Real.sqrt 2 →
  A = π / 3 →
  B = π / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1005_100512


namespace NUMINAMATH_CALUDE_child_admission_price_l1005_100539

-- Define the given conditions
def total_people : ℕ := 610
def adult_price : ℚ := 2
def total_receipts : ℚ := 960
def num_adults : ℕ := 350

-- Define the admission price for children
def child_price : ℚ := 1

-- Theorem to prove
theorem child_admission_price :
  child_price * (total_people - num_adults) + adult_price * num_adults = total_receipts :=
sorry

end NUMINAMATH_CALUDE_child_admission_price_l1005_100539


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1005_100528

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) (h_a : a > 0)
  (h_solution : ∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 4) :
  QuadraticFunction a b c 2 < QuadraticFunction a b c (-1) ∧
  QuadraticFunction a b c (-1) < QuadraticFunction a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1005_100528


namespace NUMINAMATH_CALUDE_max_a_for_zero_points_l1005_100525

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / x^2 - x - a/x + 2*Real.exp 1

theorem max_a_for_zero_points :
  (∃ a : ℝ, ∃ x : ℝ, x > 0 ∧ f a x = 0) →
  (∀ a : ℝ, (∃ x : ℝ, x > 0 ∧ f a x = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1) ∧
  (∃ x : ℝ, x > 0 ∧ f (Real.exp 2 + 1 / Real.exp 1) x = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_zero_points_l1005_100525


namespace NUMINAMATH_CALUDE_remainder_a52_mod_52_l1005_100533

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of a_n as the concatenation of integers from 1 to n
  sorry

theorem remainder_a52_mod_52 : concatenate_integers 52 % 52 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_a52_mod_52_l1005_100533


namespace NUMINAMATH_CALUDE_marbles_in_bag_l1005_100536

theorem marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) : 
  red_marbles = 12 →
  ((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = 9 / 16 →
  total_marbles = 48 := by
sorry

end NUMINAMATH_CALUDE_marbles_in_bag_l1005_100536


namespace NUMINAMATH_CALUDE_iphone_average_cost_l1005_100506

/-- Proves that the average cost of an iPhone is $1000 given the sales data --/
theorem iphone_average_cost (iphone_count : Nat) (ipad_count : Nat) (appletv_count : Nat)
  (ipad_cost : ℝ) (appletv_cost : ℝ) (total_average : ℝ)
  (h1 : iphone_count = 100)
  (h2 : ipad_count = 20)
  (h3 : appletv_count = 80)
  (h4 : ipad_cost = 900)
  (h5 : appletv_cost = 200)
  (h6 : total_average = 670) :
  (iphone_count * 1000 + ipad_count * ipad_cost + appletv_count * appletv_cost) /
    (iphone_count + ipad_count + appletv_count : ℝ) = total_average :=
by sorry

end NUMINAMATH_CALUDE_iphone_average_cost_l1005_100506


namespace NUMINAMATH_CALUDE_division_evaluation_l1005_100581

theorem division_evaluation : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by sorry

end NUMINAMATH_CALUDE_division_evaluation_l1005_100581


namespace NUMINAMATH_CALUDE_sum_abc_equals_two_l1005_100571

theorem sum_abc_equals_two (a b c : ℝ) 
  (h : (a - 1)^2 + |b + 1| + Real.sqrt (b + c - a) = 0) : 
  a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_two_l1005_100571


namespace NUMINAMATH_CALUDE_cubic_function_c_value_l1005_100573

/-- A function f: ℝ → ℝ has exactly two roots if there exist exactly two distinct real numbers x₁ and x₂ such that f(x₁) = f(x₂) = 0 -/
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂

/-- The main theorem stating that if y = x³ - 3x + c has exactly two roots, then c = -2 or c = 2 -/
theorem cubic_function_c_value (c : ℝ) :
  has_exactly_two_roots (λ x : ℝ => x^3 - 3*x + c) → c = -2 ∨ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_c_value_l1005_100573


namespace NUMINAMATH_CALUDE_schedule_count_is_576_l1005_100576

/-- Represents a table tennis match between two schools -/
structure TableTennisMatch where
  /-- Number of players in each school -/
  players_per_school : Nat
  /-- Number of opponents each player faces from the other school -/
  opponents_per_player : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- The specific match configuration from the problem -/
def match_config : TableTennisMatch :=
  { players_per_school := 4
  , opponents_per_player := 2
  , total_rounds := 6
  , games_per_round := 4
  }

/-- Calculate the number of ways to schedule the match -/
def schedule_count (m : TableTennisMatch) : Nat :=
  (Nat.factorial m.total_rounds) * (Nat.factorial m.games_per_round)

/-- Theorem stating that the number of ways to schedule the match is 576 -/
theorem schedule_count_is_576 : schedule_count match_config = 576 := by
  sorry


end NUMINAMATH_CALUDE_schedule_count_is_576_l1005_100576


namespace NUMINAMATH_CALUDE_problem_solution_l1005_100555

theorem problem_solution (n : ℕ+) 
  (x : ℝ) (hx : x = (Real.sqrt (n + 2) - Real.sqrt n) / (Real.sqrt (n + 2) + Real.sqrt n))
  (y : ℝ) (hy : y = (Real.sqrt (n + 2) + Real.sqrt n) / (Real.sqrt (n + 2) - Real.sqrt n))
  (h : 14 * x^2 + 26 * x * y + 14 * y^2 = 2014) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1005_100555


namespace NUMINAMATH_CALUDE_smallest_representable_66_88_l1005_100566

/-- Represents a base-b digit --/
def IsDigitBase (d : ℕ) (b : ℕ) : Prop := d < b

/-- Converts a two-digit number in base b to base 10 --/
def BaseToDecimal (d₁ d₂ : ℕ) (b : ℕ) : ℕ := d₁ * b + d₂

/-- States that a number n can be represented as CC₆ and DD₈ --/
def RepresentableAs66And88 (n : ℕ) : Prop :=
  ∃ (c d : ℕ), IsDigitBase c 6 ∧ IsDigitBase d 8 ∧
    n = BaseToDecimal c c 6 ∧ n = BaseToDecimal d d 8

theorem smallest_representable_66_88 :
  (∀ m, RepresentableAs66And88 m → m ≥ 63) ∧ RepresentableAs66And88 63 := by sorry

end NUMINAMATH_CALUDE_smallest_representable_66_88_l1005_100566


namespace NUMINAMATH_CALUDE_houses_before_boom_correct_l1005_100501

/-- The number of houses in Lawrence County before the housing boom. -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County. -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom. -/
def houses_built_during_boom : ℕ := 574

/-- Theorem stating that the number of houses before the boom
    plus the number of houses built during the boom
    equals the current number of houses. -/
theorem houses_before_boom_correct :
  houses_before_boom + houses_built_during_boom = current_houses :=
by sorry

end NUMINAMATH_CALUDE_houses_before_boom_correct_l1005_100501


namespace NUMINAMATH_CALUDE_bruce_triple_age_l1005_100547

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Bruce's son's current age -/
def son_age : ℕ := 8

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple : ℕ := 6

/-- Theorem stating that in 6 years, Bruce will be three times as old as his son -/
theorem bruce_triple_age :
  bruce_age + years_until_triple = 3 * (son_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_bruce_triple_age_l1005_100547


namespace NUMINAMATH_CALUDE_opposites_sum_l1005_100572

theorem opposites_sum (x y : ℝ) : (x + 5)^2 + |y - 2| = 0 → x + 2*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l1005_100572


namespace NUMINAMATH_CALUDE_max_available_is_two_l1005_100523

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the colleagues
inductive Colleague
| Alice
| Bob
| Charlie
| Diana

-- Define a function that represents the availability of a colleague on a given day
def isAvailable (c : Colleague) (d : Day) : Bool :=
  match c, d with
  | Colleague.Alice, Day.Monday => false
  | Colleague.Alice, Day.Tuesday => true
  | Colleague.Alice, Day.Wednesday => false
  | Colleague.Alice, Day.Thursday => true
  | Colleague.Alice, Day.Friday => false
  | Colleague.Bob, Day.Monday => true
  | Colleague.Bob, Day.Tuesday => false
  | Colleague.Bob, Day.Wednesday => true
  | Colleague.Bob, Day.Thursday => false
  | Colleague.Bob, Day.Friday => true
  | Colleague.Charlie, Day.Monday => false
  | Colleague.Charlie, Day.Tuesday => false
  | Colleague.Charlie, Day.Wednesday => true
  | Colleague.Charlie, Day.Thursday => true
  | Colleague.Charlie, Day.Friday => false
  | Colleague.Diana, Day.Monday => true
  | Colleague.Diana, Day.Tuesday => true
  | Colleague.Diana, Day.Wednesday => false
  | Colleague.Diana, Day.Thursday => false
  | Colleague.Diana, Day.Friday => true

-- Define a function that counts the number of available colleagues on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun c => isAvailable c d) [Colleague.Alice, Colleague.Bob, Colleague.Charlie, Colleague.Diana]).length

-- Theorem: The maximum number of available colleagues on any day is 2
theorem max_available_is_two :
  (List.map countAvailable [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]).maximum? = some 2 := by
  sorry


end NUMINAMATH_CALUDE_max_available_is_two_l1005_100523


namespace NUMINAMATH_CALUDE_inequality_proof_l1005_100505

theorem inequality_proof (a b c : ℝ) (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  (a + b + c ≥ 3) ∧ (a * b * c ≤ 1) ∧
  ((a + b + c = 3 ∧ a * b * c = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1005_100505


namespace NUMINAMATH_CALUDE_inequality_proof_l1005_100545

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1005_100545


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1005_100594

theorem complex_fraction_equality (a b : ℝ) :
  (1 + Complex.I) / (1 - Complex.I) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1005_100594


namespace NUMINAMATH_CALUDE_mary_shopping_total_l1005_100549

def store1_total : ℚ := 13.04 + 12.27
def store2_total : ℚ := 44.15 + 25.50
def store3_total : ℚ := 2 * 9.99 * (1 - 0.1)
def store4_total : ℚ := 30.93 + 7.42
def store5_total : ℚ := 20.75 * (1 + 0.05)

def total_spent : ℚ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_shopping_total :
  total_spent = 173.08 := by sorry

end NUMINAMATH_CALUDE_mary_shopping_total_l1005_100549


namespace NUMINAMATH_CALUDE_eight_solutions_l1005_100529

/-- The function f(x) = x^2 - 2 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The theorem stating that f(f(f(x))) = x has exactly eight distinct real solutions -/
theorem eight_solutions :
  ∃! (s : Finset ℝ), s.card = 8 ∧ (∀ x ∈ s, f (f (f x)) = x) ∧
    (∀ y : ℝ, f (f (f y)) = y → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_eight_solutions_l1005_100529


namespace NUMINAMATH_CALUDE_power_sum_theorem_l1005_100558

theorem power_sum_theorem (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d ∧ a^3 + b^3 = c^3 + d^3 ∧ a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l1005_100558
