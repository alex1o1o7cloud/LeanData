import Mathlib

namespace NUMINAMATH_CALUDE_rose_difference_l25_2519

theorem rose_difference (santiago_roses garrett_roses : ℕ) 
  (h1 : santiago_roses = 58) 
  (h2 : garrett_roses = 24) : 
  santiago_roses - garrett_roses = 34 := by
sorry

end NUMINAMATH_CALUDE_rose_difference_l25_2519


namespace NUMINAMATH_CALUDE_expression_evaluation_l25_2588

theorem expression_evaluation (x : ℚ) (h : x = 1/2) : 
  (1 + x) * (1 - x) + x * (x + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l25_2588


namespace NUMINAMATH_CALUDE_system_solution_l25_2556

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  (3 * y - Real.sqrt (y / x) - 6 * Real.sqrt (x * y) + 2 = 0) ∧
  (x^2 + 81 * x^2 * y^4 = 2 * y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ (x = Real.rpow 31 (1/4) / 12 ∧ y = Real.rpow 31 (1/4) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l25_2556


namespace NUMINAMATH_CALUDE_right_triangle_area_l25_2535

theorem right_triangle_area (a c : ℝ) (h1 : a = 15) (h2 : c = 17) : ∃ b : ℝ, 
  a^2 + b^2 = c^2 ∧ (1/2) * a * b = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l25_2535


namespace NUMINAMATH_CALUDE_square_friendly_unique_l25_2566

def square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, ∃ n : ℤ, m^2 + 18*m + c = n^2

theorem square_friendly_unique : 
  (square_friendly 81) ∧ (∀ c : ℤ, square_friendly c → c = 81) :=
sorry

end NUMINAMATH_CALUDE_square_friendly_unique_l25_2566


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l25_2521

theorem quadratic_equation_solution : 
  ∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ x = 2 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l25_2521


namespace NUMINAMATH_CALUDE_sqrt3_minus1_power0_plus_2_power_neg1_l25_2575

theorem sqrt3_minus1_power0_plus_2_power_neg1 : (Real.sqrt 3 - 1) ^ 0 + 2 ^ (-1 : ℤ) = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_minus1_power0_plus_2_power_neg1_l25_2575


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l25_2540

theorem quadratic_two_distinct_roots (c : ℝ) (h : c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + c = 0 ∧ x₂^2 + 2*x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l25_2540


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_inverses_squared_l25_2544

theorem quadratic_roots_sum_of_inverses_squared (p q : ℝ) : 
  (3 * p^2 - 5 * p + 2 = 0) → 
  (3 * q^2 - 5 * q + 2 = 0) → 
  (1 / p^2 + 1 / q^2 = 13 / 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_inverses_squared_l25_2544


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l25_2552

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * x^2 + m * x) / Real.exp x

theorem extreme_value_and_monotonicity (m : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x| ∧ |x| < ε → f 0 0 ≤ f 0 x) ∧
  (∀ x, x ≥ 3 → ∀ y, y > x → f m y ≤ f m x) ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l25_2552


namespace NUMINAMATH_CALUDE_ashton_pencils_l25_2502

theorem ashton_pencils (initial_pencils_per_box : ℕ) : 
  (2 * initial_pencils_per_box) - 6 = 22 → initial_pencils_per_box = 14 :=
by sorry

end NUMINAMATH_CALUDE_ashton_pencils_l25_2502


namespace NUMINAMATH_CALUDE_money_left_after_transactions_l25_2507

def initial_money : ℕ := 50 * 10 + 24 * 25 + 40 * 5 + 75

def candy_cost : ℕ := 6 * 85
def lollipop_cost : ℕ := 3 * 50
def chips_cost : ℕ := 4 * 95
def soda_cost : ℕ := 2 * 125

def total_cost : ℕ := candy_cost + lollipop_cost + chips_cost + soda_cost

theorem money_left_after_transactions : 
  initial_money - total_cost = 85 := by
sorry

end NUMINAMATH_CALUDE_money_left_after_transactions_l25_2507


namespace NUMINAMATH_CALUDE_dumbbell_weight_l25_2531

theorem dumbbell_weight (total_dumbbells : ℕ) (total_weight : ℕ) 
  (h1 : total_dumbbells = 6)
  (h2 : total_weight = 120) :
  total_weight / total_dumbbells = 20 := by
sorry

end NUMINAMATH_CALUDE_dumbbell_weight_l25_2531


namespace NUMINAMATH_CALUDE_courtyard_length_proof_l25_2581

/-- Proves that the length of a rectangular courtyard is 15 m given specific conditions -/
theorem courtyard_length_proof (width : ℝ) (stone_length : ℝ) (stone_width : ℝ) (total_stones : ℕ) :
  width = 6 →
  stone_length = 3 →
  stone_width = 2 →
  total_stones = 15 →
  (width * (width * total_stones * stone_length * stone_width / width / stone_length / stone_width)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_proof_l25_2581


namespace NUMINAMATH_CALUDE_sons_age_l25_2567

/-- Proves that given the conditions, the son's present age is 33 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 35 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 33 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l25_2567


namespace NUMINAMATH_CALUDE_power_product_equals_4410000_l25_2518

theorem power_product_equals_4410000 : 2^4 * 3^2 * 5^4 * 7^2 = 4410000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_4410000_l25_2518


namespace NUMINAMATH_CALUDE_unique_solution_mod_30_l25_2522

theorem unique_solution_mod_30 : 
  ∃! x : ℕ, x < 30 ∧ (x^4 + 2*x^3 + 3*x^2 - x + 1) % 30 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_mod_30_l25_2522


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l25_2593

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = 3 * p.1 + 4}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 ^ 2}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {(-1, 1), (4, 16)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l25_2593


namespace NUMINAMATH_CALUDE_canoe_rental_cost_l25_2528

/-- Represents the daily rental cost and count of canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℝ
  kayak_cost : ℝ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Calculates the total revenue from canoe and kayak rentals --/
def total_revenue (info : RentalInfo) : ℝ :=
  info.canoe_cost * info.canoe_count + info.kayak_cost * info.kayak_count

/-- Theorem stating that the daily rental cost of a canoe is $15 --/
theorem canoe_rental_cost (info : RentalInfo) :
  info.kayak_cost = 18 ∧
  info.canoe_count = (3 * info.kayak_count) / 2 ∧
  total_revenue info = 405 ∧
  info.canoe_count = info.kayak_count + 5 →
  info.canoe_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_canoe_rental_cost_l25_2528


namespace NUMINAMATH_CALUDE_eighth_grade_girls_l25_2558

theorem eighth_grade_girls (total_students : ℕ) (boys girls : ℕ) : 
  total_students = 68 →
  boys = 2 * girls - 16 →
  total_students = boys + girls →
  girls = 28 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_girls_l25_2558


namespace NUMINAMATH_CALUDE_parking_duration_for_5_5_yuan_l25_2524

/-- Calculates the parking duration given the total fee paid -/
def parking_duration (total_fee : ℚ) : ℚ :=
  (total_fee - 0.5) / (0.5 + 0.5) + 1

/-- Theorem stating that given the specific fee paid, the parking duration is 6 hours -/
theorem parking_duration_for_5_5_yuan :
  parking_duration 5.5 = 6 := by sorry

end NUMINAMATH_CALUDE_parking_duration_for_5_5_yuan_l25_2524


namespace NUMINAMATH_CALUDE_root_sum_zero_l25_2559

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_zero_l25_2559


namespace NUMINAMATH_CALUDE_wedding_decoration_cost_l25_2533

/-- Calculates the total cost of decorations for a wedding reception --/
def total_decoration_cost (num_tables : ℕ) (tablecloth_cost : ℕ) (place_settings_per_table : ℕ) 
  (place_setting_cost : ℕ) (roses_per_centerpiece : ℕ) (rose_cost : ℕ) (lilies_per_centerpiece : ℕ) 
  (lily_cost : ℕ) : ℕ :=
  num_tables * (tablecloth_cost + place_settings_per_table * place_setting_cost + 
  roses_per_centerpiece * rose_cost + lilies_per_centerpiece * lily_cost)

/-- Theorem stating that the total decoration cost for the given parameters is 3500 --/
theorem wedding_decoration_cost : 
  total_decoration_cost 20 25 4 10 10 5 15 4 = 3500 := by
  sorry

#eval total_decoration_cost 20 25 4 10 10 5 15 4

end NUMINAMATH_CALUDE_wedding_decoration_cost_l25_2533


namespace NUMINAMATH_CALUDE_tire_cost_l25_2584

theorem tire_cost (window_cost tire_count total_cost : ℕ) 
  (h1 : window_cost = 700)
  (h2 : tire_count = 3)
  (h3 : total_cost = 1450)
  (h4 : tire_count * (total_cost - window_cost) / tire_count = 250) :
  ∃ (single_tire_cost : ℕ), 
    single_tire_cost * tire_count + window_cost = total_cost ∧ 
    single_tire_cost = 250 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_l25_2584


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_c_is_2root3_l25_2546

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0)
  (h4 : a * sin B = b * sin A)
  (h5 : b * sin C = c * sin B)
  (h6 : c * sin A = a * sin C)

variable (t : Triangle)

-- Given condition
axiom given_condition : t.a * cos t.B + t.b * cos t.A = 2 * t.c * cos t.C

-- Theorem 1: Prove that C = π/3
theorem angle_C_is_pi_third : t.C = π/3 :=
sorry

-- Theorem 2: Prove that if c = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_when_c_is_2root3 (h : t.c = 2 * Real.sqrt 3) :
  (∀ s : Triangle, s.c = t.c → t.a * t.b * sin t.C / 2 ≥ s.a * s.b * sin s.C / 2) ∧
  t.a * t.b * sin t.C / 2 = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_c_is_2root3_l25_2546


namespace NUMINAMATH_CALUDE_bus_car_speed_equation_l25_2514

theorem bus_car_speed_equation (x : ℝ) (h1 : x > 0) : 
  (20 / x - 20 / (1.5 * x) = 1 / 6) ↔ 
  (20 / x = 20 / (1.5 * x) + 1 / 6) := by sorry

end NUMINAMATH_CALUDE_bus_car_speed_equation_l25_2514


namespace NUMINAMATH_CALUDE_variance_binomial_4_half_l25_2525

/-- The variance of a binomial distribution with n trials and probability p -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem variance_binomial_4_half :
  binomialVariance 4 (1/2 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_variance_binomial_4_half_l25_2525


namespace NUMINAMATH_CALUDE_area_PQRSTU_l25_2512

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (P Q R S T U : Point)

/-- The given hexagonal polygon PQRSTU -/
def PQRSTU : Hexagon := sorry

/-- Point V, the intersection of extended lines QT and PU -/
def V : Point := sorry

/-- Length of side PQ -/
def PQ_length : ℝ := 8

/-- Length of side QR -/
def QR_length : ℝ := 10

/-- Length of side UT -/
def UT_length : ℝ := 7

/-- Length of side TU -/
def TU_length : ℝ := 3

/-- Predicate stating that PQRV is a rectangle -/
def is_rectangle_PQRV (h : Hexagon) (v : Point) : Prop := sorry

/-- Predicate stating that VUT is a rectangle -/
def is_rectangle_VUT (h : Hexagon) (v : Point) : Prop := sorry

/-- Function to calculate the area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- Theorem stating that the area of PQRSTU is 65 square units -/
theorem area_PQRSTU :
  is_rectangle_PQRV PQRSTU V →
  is_rectangle_VUT PQRSTU V →
  area PQRSTU = 65 := by sorry

end NUMINAMATH_CALUDE_area_PQRSTU_l25_2512


namespace NUMINAMATH_CALUDE_adam_apples_l25_2503

theorem adam_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10) 
  (h2 : jackie_apples = adam_apples + 1) : 
  adam_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_l25_2503


namespace NUMINAMATH_CALUDE_jenny_jill_game_percentage_l25_2573

theorem jenny_jill_game_percentage :
  -- Define the number of games Jenny played against Mark
  ∀ (games_with_mark : ℕ),
  -- Define Mark's wins
  ∀ (mark_wins : ℕ),
  -- Define Jenny's total wins
  ∀ (jenny_total_wins : ℕ),
  -- Conditions
  games_with_mark = 10 →
  mark_wins = 1 →
  jenny_total_wins = 14 →
  -- Conclusion: Jill's win percentage is 75%
  (((2 * games_with_mark) - (jenny_total_wins - (games_with_mark - mark_wins))) / (2 * games_with_mark) : ℚ) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_jenny_jill_game_percentage_l25_2573


namespace NUMINAMATH_CALUDE_friday_sales_l25_2578

/-- Kim's cupcake sales pattern --/
def cupcake_sales (tuesday_before_discount : ℕ) : ℕ :=
  let tuesday := tuesday_before_discount + (tuesday_before_discount * 5 / 100)
  let monday := tuesday + (tuesday * 50 / 100)
  let wednesday := tuesday * 3 / 2
  let thursday := wednesday - (wednesday * 20 / 100)
  let friday := thursday * 13 / 10
  friday

/-- Theorem: Kim sold 1310 boxes on Friday --/
theorem friday_sales : cupcake_sales 800 = 1310 := by
  sorry

end NUMINAMATH_CALUDE_friday_sales_l25_2578


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l25_2545

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  ((n - 2) * 180 : ℝ) / n = 160 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l25_2545


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l25_2591

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (true for 1, false for 0) -/
def binary_10110 : List Bool := [true, false, true, true, false]
def binary_1101 : List Bool := [true, true, false, true]
def binary_110 : List Bool := [true, true, false]
def binary_101 : List Bool := [true, false, true]
def binary_1010 : List Bool := [true, false, true, false]

/-- The main theorem to prove -/
theorem binary_arithmetic_equality :
  binary_to_decimal binary_10110 - binary_to_decimal binary_1101 +
  binary_to_decimal binary_110 - binary_to_decimal binary_101 =
  binary_to_decimal binary_1010 := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l25_2591


namespace NUMINAMATH_CALUDE_kerosene_cost_friday_l25_2511

/-- The cost of a liter of kerosene on Friday given the market conditions --/
theorem kerosene_cost_friday (rice_cost_monday : ℝ) 
  (h1 : rice_cost_monday = 0.36)
  (h2 : ∀ x, x > 0 → x * 12 * rice_cost_monday = x * 8 * (0.5 * rice_cost_monday))
  (h3 : ∀ x, x > 0 → 1.2 * x * rice_cost_monday = x * 1.2 * rice_cost_monday) :
  ∃ (kerosene_cost_friday : ℝ), kerosene_cost_friday = 0.576 :=
by sorry

end NUMINAMATH_CALUDE_kerosene_cost_friday_l25_2511


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequality_l25_2532

-- Define a quadrilateral structure
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  c_positive : 0 < c
  d_positive : 0 < d
  area_positive : 0 < area

-- State the theorem
theorem quadrilateral_area_inequality (q : Quadrilateral) : 2 * q.area ≤ q.a * q.c + q.b * q.d := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequality_l25_2532


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l25_2596

theorem intersection_point_k_value (k : ℝ) : 
  (∃ y : ℝ, -3 * (-9.6) + 2 * y = k ∧ 0.25 * (-9.6) + y = 16) → k = 65.6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l25_2596


namespace NUMINAMATH_CALUDE_homework_probability_l25_2539

theorem homework_probability (p : ℚ) (h : p = 5 / 9) :
  1 - p = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_homework_probability_l25_2539


namespace NUMINAMATH_CALUDE_avery_donation_l25_2529

/-- The number of clothes Avery is donating -/
def total_clothes (shirts pants shorts : ℕ) : ℕ := shirts + pants + shorts

/-- Theorem stating the total number of clothes Avery is donating -/
theorem avery_donation :
  ∀ (shirts pants shorts : ℕ),
    shirts = 4 →
    pants = 2 * shirts →
    shorts = pants / 2 →
    total_clothes shirts pants shorts = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l25_2529


namespace NUMINAMATH_CALUDE_unit_digit_of_expression_l25_2543

-- Define the expression
def expression : ℕ := (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1

-- Theorem statement
theorem unit_digit_of_expression : expression % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_expression_l25_2543


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l25_2550

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l25_2550


namespace NUMINAMATH_CALUDE_alices_spending_l25_2541

theorem alices_spending (B : ℝ) : 
  ∃ (book magazine : ℝ),
    book = 0.25 * (B - magazine) ∧
    magazine = 0.1 * (B - book) ∧
    book + magazine = (4/13) * B :=
by sorry

end NUMINAMATH_CALUDE_alices_spending_l25_2541


namespace NUMINAMATH_CALUDE_expression_value_l25_2599

theorem expression_value (a b : ℚ) (ha : a = -1) (hb : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l25_2599


namespace NUMINAMATH_CALUDE_function_properties_l25_2504

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_deriv : ∀ x ∈ Set.Ioo 0 (π/2), f' x * Real.sin x - f x * Real.cos x > 0) :
  f (π/4) > -Real.sqrt 2 * f (-π/6) ∧ f (π/3) > Real.sqrt 3 * f (π/6) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l25_2504


namespace NUMINAMATH_CALUDE_coefficient_m5n3_in_expansion_l25_2568

theorem coefficient_m5n3_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 5 : ℕ) = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_m5n3_in_expansion_l25_2568


namespace NUMINAMATH_CALUDE_number_puzzle_l25_2555

theorem number_puzzle : ∃ x : ℝ, (x / 5) + 10 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l25_2555


namespace NUMINAMATH_CALUDE_asterisk_replacement_l25_2577

theorem asterisk_replacement : (54 / 18) * (54 / 162) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l25_2577


namespace NUMINAMATH_CALUDE_solution_set_equality_l25_2561

theorem solution_set_equality : Set ℝ := by
  have h : Set ℝ := {x | (x - 1)^2 < 1}
  have g : Set ℝ := Set.Ioo 0 2
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l25_2561


namespace NUMINAMATH_CALUDE_pens_pencils_cost_l25_2564

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def num_notebooks : ℕ := 5

def cost_pens_pencils : ℝ := total_spent - (backpack_cost + notebook_cost * num_notebooks)

theorem pens_pencils_cost (h : cost_pens_pencils = 2) : 
  cost_pens_pencils / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_pens_pencils_cost_l25_2564


namespace NUMINAMATH_CALUDE_sequence_general_term_l25_2586

theorem sequence_general_term (a : ℕ → ℚ) :
  a 1 = -1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = a (n + 1) - a n) →
  ∀ n : ℕ, n ≥ 1 → a n = -1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l25_2586


namespace NUMINAMATH_CALUDE_length_AB_on_parabola_l25_2554

/-- Parabola type -/
structure Parabola where
  a : ℝ
  C : ℝ × ℝ → Prop
  focus : ℝ × ℝ

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.C point

/-- Tangent line to a parabola at a point -/
def tangent_line (p : Parabola) (pt : PointOnParabola p) : ℝ × ℝ → Prop := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of AB on parabola y² = 6x -/
theorem length_AB_on_parabola (p : Parabola) 
  (h_eq : p.C = fun (x, y) ↦ y^2 = 6*x) 
  (A B : PointOnParabola p) 
  (F : ℝ × ℝ) 
  (h_focus : F = p.focus)
  (h_collinear : ∃ (m : ℝ), A.point.1 = m * A.point.2 + F.1 ∧ 
                             B.point.1 = m * B.point.2 + F.1)
  (P : ℝ × ℝ)
  (h_tangent_intersect : (tangent_line p A) P ∧ (tangent_line p B) P)
  (h_PF_distance : distance P F = 2 * Real.sqrt 3) :
  distance A.point B.point = 8 := by sorry

end NUMINAMATH_CALUDE_length_AB_on_parabola_l25_2554


namespace NUMINAMATH_CALUDE_smallest_bdf_l25_2516

theorem smallest_bdf (a b c d e f : ℕ+) : 
  (∃ A : ℚ, A = (a / b) * (c / d) * (e / f) ∧ 
   ((a + 1) / b) * (c / d) * (e / f) = A + 3 ∧
   (a / b) * ((c + 1) / d) * (e / f) = A + 4 ∧
   (a / b) * (c / d) * ((e + 1) / f) = A + 5) →
  (∃ m : ℕ+, b * d * f = m ∧ ∀ n : ℕ+, b * d * f ≤ n) →
  b * d * f = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bdf_l25_2516


namespace NUMINAMATH_CALUDE_magnitude_2a_plus_b_l25_2589

variable (a b : ℝ × ℝ)

theorem magnitude_2a_plus_b (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 2) 
  (h3 : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
  ‖2 • a + b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_2a_plus_b_l25_2589


namespace NUMINAMATH_CALUDE_factor_expression_l25_2585

theorem factor_expression : ∀ x : ℝ, 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l25_2585


namespace NUMINAMATH_CALUDE_specific_grid_area_l25_2563

/-- A rectangular grid formed by perpendicular lines -/
structure RectangularGrid where
  num_boundary_lines : ℕ
  perimeter : ℝ
  is_rectangular : Bool
  has_perpendicular_lines : Bool

/-- The area of a rectangular grid -/
def grid_area (grid : RectangularGrid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific rectangular grid -/
theorem specific_grid_area :
  ∀ (grid : RectangularGrid),
    grid.num_boundary_lines = 36 ∧
    grid.perimeter = 72 ∧
    grid.is_rectangular = true ∧
    grid.has_perpendicular_lines = true →
    grid_area grid = 84 :=
  sorry

end NUMINAMATH_CALUDE_specific_grid_area_l25_2563


namespace NUMINAMATH_CALUDE_maddie_spent_95_l25_2523

/-- Calculates the total amount spent on T-shirts with a bulk discount -/
def total_spent (white_packs blue_packs : ℕ) 
                (white_per_pack blue_per_pack : ℕ) 
                (white_price blue_price : ℚ) 
                (discount_percent : ℚ) : ℚ :=
  let white_total := white_packs * white_per_pack * white_price
  let blue_total := blue_packs * blue_per_pack * blue_price
  let subtotal := white_total + blue_total
  let discount := subtotal * (discount_percent / 100)
  subtotal - discount

/-- Proves that Maddie spent $95 on T-shirts -/
theorem maddie_spent_95 : 
  total_spent 2 4 5 3 4 5 5 = 95 := by
  sorry

end NUMINAMATH_CALUDE_maddie_spent_95_l25_2523


namespace NUMINAMATH_CALUDE_circle_properties_l25_2548

/-- For a circle with area 4π, prove its diameter is 4 and circumference is 4π -/
theorem circle_properties (r : ℝ) (h : r^2 * π = 4 * π) : 
  2 * r = 4 ∧ 2 * π * r = 4 * π :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l25_2548


namespace NUMINAMATH_CALUDE_take_home_pay_calculation_l25_2505

/-- Calculate take-home pay after deductions -/
theorem take_home_pay_calculation (total_pay : ℝ) 
  (tax_rate insurance_rate pension_rate union_rate : ℝ) :
  total_pay = 500 →
  tax_rate = 0.10 →
  insurance_rate = 0.05 →
  pension_rate = 0.03 →
  union_rate = 0.02 →
  total_pay * (1 - (tax_rate + insurance_rate + pension_rate + union_rate)) = 400 := by
  sorry

end NUMINAMATH_CALUDE_take_home_pay_calculation_l25_2505


namespace NUMINAMATH_CALUDE_quadratic_term_elimination_l25_2590

/-- The polynomial in question -/
def polynomial (x m : ℝ) : ℝ := 3*x^2 - 10 - 2*x - 4*x^2 + m*x^2

/-- The coefficient of x^2 in the polynomial -/
def x_squared_coefficient (m : ℝ) : ℝ := 3 - 4 + m

theorem quadratic_term_elimination :
  ∃ (m : ℝ), x_squared_coefficient m = 0 ∧ m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_term_elimination_l25_2590


namespace NUMINAMATH_CALUDE_arccos_cos_three_l25_2527

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l25_2527


namespace NUMINAMATH_CALUDE_football_team_yardage_l25_2565

/-- A football team's yardage problem -/
theorem football_team_yardage (L : ℤ) : 
  ((-L : ℤ) + 13 = 8) → L = 5 := by
  sorry

end NUMINAMATH_CALUDE_football_team_yardage_l25_2565


namespace NUMINAMATH_CALUDE_sum_of_variables_l25_2513

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 18 - 4*x)
  (eq2 : x + z = 22 - 4*y)
  (eq3 : x + y = 15 - 4*z) :
  3*x + 3*y + 3*z = 55/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l25_2513


namespace NUMINAMATH_CALUDE_volume_of_parallelepiped_l25_2515

/-- A rectangular parallelepiped with given diagonal and side face diagonals -/
structure RectParallelepiped where
  diag : ℝ
  side_diag1 : ℝ
  side_diag2 : ℝ
  volume : ℝ

/-- The volume of a rectangular parallelepiped with the given dimensions -/
def volume_calc (p : RectParallelepiped) : Prop :=
  p.diag = 13 ∧ p.side_diag1 = 4 * Real.sqrt 10 ∧ p.side_diag2 = 3 * Real.sqrt 17 → p.volume = 144

theorem volume_of_parallelepiped :
  ∀ p : RectParallelepiped, volume_calc p :=
sorry

end NUMINAMATH_CALUDE_volume_of_parallelepiped_l25_2515


namespace NUMINAMATH_CALUDE_complex_sum_equals_seven_plus_three_i_l25_2526

theorem complex_sum_equals_seven_plus_three_i :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -3
  let R : ℂ := -2*I
  let T : ℂ := 1 + 3*I
  B - Q + R + T = 7 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_seven_plus_three_i_l25_2526


namespace NUMINAMATH_CALUDE_power_product_equality_l25_2557

theorem power_product_equality : (-4 : ℝ)^2013 * (-0.25 : ℝ)^2014 = -0.25 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l25_2557


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l25_2508

theorem trigonometric_expression_equality : 
  (Real.sqrt 3 * Real.sin (-20/3 * Real.pi)) / Real.tan (11/3 * Real.pi) - 
  Real.cos (13/4 * Real.pi) * Real.tan (-35/4 * Real.pi) = 
  (Real.sqrt 2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l25_2508


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_quadratic_roots_bound_l25_2517

theorem cubic_sum_inequality (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

theorem quadratic_roots_bound (a b : ℝ) (h : |a| + |b| < 1) :
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_quadratic_roots_bound_l25_2517


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l25_2572

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ c d : ℕ+, c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l25_2572


namespace NUMINAMATH_CALUDE_M_mod_55_l25_2560

def M : ℕ := sorry

theorem M_mod_55 : M % 55 = 50 := by sorry

end NUMINAMATH_CALUDE_M_mod_55_l25_2560


namespace NUMINAMATH_CALUDE_complex_equation_solution_l25_2500

theorem complex_equation_solution (a b : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l25_2500


namespace NUMINAMATH_CALUDE_curve_symmetry_l25_2534

/-- A curve in the xy-plane -/
class Curve (f : ℝ → ℝ → Prop) : Prop

/-- Symmetry of a curve with respect to a line -/
def symmetricTo (f : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y ↔ f (y + 3) (x - 3)

/-- The line x - y - 3 = 0 -/
def symmetryLine (x y : ℝ) : ℝ := x - y - 3

/-- Theorem: If a curve f is symmetric with respect to the line x - y - 3 = 0,
    then its equation is f(y + 3, x - 3) = 0 -/
theorem curve_symmetry (f : ℝ → ℝ → Prop) [Curve f] 
    (h : symmetricTo f symmetryLine) :
  ∀ x y, f x y ↔ f (y + 3) (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l25_2534


namespace NUMINAMATH_CALUDE_difference_of_squares_l25_2506

theorem difference_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x < y → 
  Real.sqrt x + Real.sqrt y = 1 → 
  Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3 → 
  y - x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l25_2506


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_l25_2536

/-- The function f(x) defined as ax³ + ax² + 7x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + 7 * x

/-- A function has no extreme points if its derivative is always non-negative or always non-positive -/
def has_no_extreme_points (g : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, (deriv g) x ≥ 0) ∨ (∀ x : ℝ, (deriv g) x ≤ 0)

/-- The main theorem: f(x) has no extreme points if and only if 0 ≤ a ≤ 21 -/
theorem no_extreme_points_iff (a : ℝ) :
  has_no_extreme_points (f a) ↔ 0 ≤ a ∧ a ≤ 21 := by sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_l25_2536


namespace NUMINAMATH_CALUDE_locus_of_points_l25_2510

/-- Given two parallel lines e₁ and e₂ in the plane, separated by a distance 2g,
    and a perpendicular line f intersecting them at O₁ and O₂ respectively,
    this theorem characterizes the locus of points P(x, y) such that a line through P
    intersects e₁ at P₁ and e₂ at P₂ with O₁P₁ · O₂P₂ = k. -/
theorem locus_of_points (g : ℝ) (k : ℝ) (x y : ℝ) :
  k = 1 → (y^2 / g^2 ≥ 1 - x^2) ∧
  k = -1 → (y^2 / g^2 ≤ 1 + x^2) := by
  sorry


end NUMINAMATH_CALUDE_locus_of_points_l25_2510


namespace NUMINAMATH_CALUDE_part_one_part_two_l25_2549

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 ≤ 4}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Part I
theorem part_one (m : ℝ) : A m ∩ B = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) : B ⊆ (Set.univ \ A m) → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l25_2549


namespace NUMINAMATH_CALUDE_trip_duration_l25_2592

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_hours : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (average_speed : ℝ) : Prop :=
  ∃ (additional_hours : ℝ),
    let total_hours := initial_hours + additional_hours
    let total_distance := initial_hours * initial_speed + additional_hours * additional_speed
    (total_distance / total_hours = average_speed) ∧
    (total_hours = 15)

/-- The main theorem stating that under given conditions, the trip duration is 15 hours -/
theorem trip_duration :
  car_trip 5 30 42 38 := by
  sorry

end NUMINAMATH_CALUDE_trip_duration_l25_2592


namespace NUMINAMATH_CALUDE_set_identities_l25_2594

variable {α : Type*}
variable (A B C : Set α)

theorem set_identities :
  (A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end NUMINAMATH_CALUDE_set_identities_l25_2594


namespace NUMINAMATH_CALUDE_number_of_tangent_lines_l25_2597

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The hyperbola 4x^2-9y^2=36 -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - 9 * y^2 = 36

/-- A line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- A line has only one intersection point with the hyperbola -/
def has_one_intersection (l : Line) : Prop :=
  ∃! x y, passes_through l x y ∧ hyperbola x y

/-- The theorem to be proved -/
theorem number_of_tangent_lines : 
  ∃! (l₁ l₂ l₃ : Line), 
    (passes_through l₁ 3 0 ∧ has_one_intersection l₁) ∧
    (passes_through l₂ 3 0 ∧ has_one_intersection l₂) ∧
    (passes_through l₃ 3 0 ∧ has_one_intersection l₃) ∧
    (∀ l, passes_through l 3 0 ∧ has_one_intersection l → l = l₁ ∨ l = l₂ ∨ l = l₃) :=
sorry

end NUMINAMATH_CALUDE_number_of_tangent_lines_l25_2597


namespace NUMINAMATH_CALUDE_ron_height_is_13_l25_2598

/-- The height of Ron in feet -/
def ron_height : ℝ := 13

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height + 4

/-- The depth of the water in feet -/
def water_depth : ℝ := 255

theorem ron_height_is_13 :
  (water_depth = 15 * dean_height) →
  (dean_height = ron_height + 4) →
  (water_depth = 255) →
  ron_height = 13 := by
sorry

end NUMINAMATH_CALUDE_ron_height_is_13_l25_2598


namespace NUMINAMATH_CALUDE_intersection_M_N_l25_2583

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l25_2583


namespace NUMINAMATH_CALUDE_symmetric_f_max_value_l25_2595

/-- A function f(x) that is symmetric about x = -2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (1 - x^2) * (x^2 + a*x + b)

/-- The symmetry condition for f(x) about x = -2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x, f a b (-2 - x) = f a b (-2 + x)

/-- The theorem stating that if f(x) is symmetric about x = -2, its maximum value is 16 -/
theorem symmetric_f_max_value (a b : ℝ) (h : is_symmetric a b) :
  ∃ x, f a b x = 16 ∧ ∀ y, f a b y ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_symmetric_f_max_value_l25_2595


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l25_2542

theorem min_value_a_plus_b (a b : ℝ) (h : a^2 + 2*b^2 = 6) :
  ∃ (m : ℝ), (∀ (x y : ℝ), x^2 + 2*y^2 = 6 → m ≤ x + y) ∧ (m = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l25_2542


namespace NUMINAMATH_CALUDE_system_solution_proof_l25_2582

theorem system_solution_proof (x y : ℝ) : 
  (2 * x + y = 2 ∧ x - y = 1) → (x = 1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l25_2582


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l25_2538

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179)
  (h2 : a*b + b*c + a*c = 131) :
  a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l25_2538


namespace NUMINAMATH_CALUDE_circle_equation_l25_2537

/-- 
Given a circle with radius 2, center on the positive x-axis, and tangent to the y-axis,
prove that its equation is x^2 + y^2 - 4x = 0.
-/
theorem circle_equation (x y : ℝ) : 
  ∃ (h : ℝ), h > 0 ∧ 
  (∀ (a b : ℝ), (a - h)^2 + b^2 = 4 → a ≥ 0) ∧
  (∃ (c : ℝ), c^2 = 4 ∧ (h - 0)^2 + c^2 = 4) →
  x^2 + y^2 - 4*x = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l25_2537


namespace NUMINAMATH_CALUDE_seating_theorem_l25_2569

/-- The number of desks in a row -/
def num_desks : ℕ := 6

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- The minimum number of empty desks required between students -/
def min_gap : ℕ := 1

/-- The number of ways to seat students in desks with the given constraints -/
def seating_arrangements (n_desks n_students min_gap : ℕ) : ℕ :=
  sorry

theorem seating_theorem :
  seating_arrangements num_desks num_students min_gap = 9 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l25_2569


namespace NUMINAMATH_CALUDE_inequality_solution_set_l25_2579

theorem inequality_solution_set (x : ℝ) : 
  ((x + 2) * (1 - x) > 0) ↔ (-2 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l25_2579


namespace NUMINAMATH_CALUDE_sparklers_burn_time_l25_2501

/-- The number of sparklers -/
def num_sparklers : ℕ := 10

/-- The time it takes for one sparkler to burn down completely (in minutes) -/
def burn_time : ℚ := 2

/-- The fraction of time left when the next sparkler is lit -/
def fraction_left : ℚ := 1/10

/-- The time each sparkler burns before the next one is lit -/
def individual_burn_time : ℚ := burn_time * (1 - fraction_left)

/-- The total time for all sparklers to burn down (in minutes) -/
def total_burn_time : ℚ := (num_sparklers - 1) * individual_burn_time + burn_time

/-- Conversion function from minutes to minutes and seconds -/
def to_minutes_and_seconds (time : ℚ) : ℕ × ℕ :=
  let minutes := time.floor
  let seconds := ((time - minutes) * 60).floor
  (minutes.toNat, seconds.toNat)

theorem sparklers_burn_time :
  to_minutes_and_seconds total_burn_time = (18, 12) :=
sorry

end NUMINAMATH_CALUDE_sparklers_burn_time_l25_2501


namespace NUMINAMATH_CALUDE_school_population_theorem_l25_2547

/-- Represents the school population statistics -/
structure SchoolPopulation where
  y : ℕ  -- Total number of students
  x : ℚ  -- Percentage of boys that 162 students represent
  z : ℚ  -- Percentage of girls in the school

/-- The conditions given in the problem -/
def school_conditions (pop : SchoolPopulation) : Prop :=
  (162 : ℚ) = pop.x / 100 * (1/2 : ℚ) * pop.y ∧ 
  pop.z = 100 - 50

/-- The theorem to be proved -/
theorem school_population_theorem (pop : SchoolPopulation) 
  (h : school_conditions pop) : 
  pop.z = 50 ∧ pop.x = 32400 / pop.y := by
  sorry


end NUMINAMATH_CALUDE_school_population_theorem_l25_2547


namespace NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_proof_l25_2553

/-- Given a triangle ABC with each side divided into n equal parts and parallel lines drawn through
    the division points, the number of parallelograms formed is 3 * (n choose 2). -/
theorem parallelogram_count (n : ℕ) : ℕ :=
  3 * (n.choose 2)

#check parallelogram_count

/-- Proof of the parallelogram count theorem -/
theorem parallelogram_count_proof (n : ℕ) :
  parallelogram_count n = 3 * (n.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_parallelogram_count_proof_l25_2553


namespace NUMINAMATH_CALUDE_parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l25_2576

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l: y = k(x+1) passing through M(-1, 0)
def line (k x y : ℝ) : Prop := y = k*(x+1)

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point M on the x-axis where the directrix intersects
def M : ℝ × ℝ := (-1, 0)

-- Define the relationship between AM and AF
def AM_AF_relation (A : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  (x₁ + 1)^2 + (y₁)^2 = (25/16) * ((x₁ - 1)^2 + y₁^2)

-- Define the perpendicularity condition for QA and QB
def perpendicular_condition (Q A B : ℝ × ℝ) : Prop :=
  let (x₀, y₀) := Q
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₀ - y₁) * (y₀ - y₂) = -(x₀ - x₁) * (x₀ - x₂)

theorem parabola_intersection_and_perpendicularity (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line k A.1 A.2 ∧ 
    line k B.1 B.2 ∧ 
    AM_AF_relation A) →
  k = 3/4 ∨ k = -3/4 :=
sorry

theorem perpendicular_intersection_range (k : ℝ) :
  (∃ Q A B : ℝ × ℝ,
    parabola Q.1 Q.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    perpendicular_condition Q A B) ↔
  (k > 0 ∧ k < Real.sqrt 5 / 5) ∨ (k < 0 ∧ k > -Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_and_perpendicularity_perpendicular_intersection_range_l25_2576


namespace NUMINAMATH_CALUDE_tin_weight_in_water_l25_2587

theorem tin_weight_in_water (total_weight : ℝ) (weight_lost : ℝ) (tin_silver_ratio : ℝ) 
  (tin_loss : ℝ) (silver_weight : ℝ) (silver_loss : ℝ) :
  total_weight = 60 →
  weight_lost = 6 →
  tin_silver_ratio = 2/3 →
  tin_loss = 1.375 →
  silver_weight = 5 →
  silver_loss = 0.375 →
  ∃ (tin_weight : ℝ), tin_weight * (weight_lost / total_weight) = tin_loss ∧ 
    tin_weight = 13.75 := by
  sorry

end NUMINAMATH_CALUDE_tin_weight_in_water_l25_2587


namespace NUMINAMATH_CALUDE_johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l25_2580

/-- The size of John's donation to the charity fund -/
def johns_donation (initial_average : ℝ) (num_initial_contributions : ℕ) : ℝ :=
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_initial_amount := initial_average * num_initial_contributions
  new_average * num_total_contributions - total_initial_amount

/-- Proof that John's donation is $260 given the conditions -/
theorem johns_donation_is_260 :
  let initial_average := 50
  let num_initial_contributions := 5
  johns_donation initial_average num_initial_contributions = 260 :=
by sorry

/-- The average contribution size increases by 70% after John's donation -/
theorem average_increase_70_percent (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  new_average = initial_average * 1.7 :=
by sorry

/-- The new average contribution size is $85 per person -/
theorem new_average_is_85 (initial_average : ℝ) (num_initial_contributions : ℕ) :
  let new_average := 85
  let num_total_contributions := num_initial_contributions + 1
  let total_amount := initial_average * num_initial_contributions + johns_donation initial_average num_initial_contributions
  total_amount / num_total_contributions = new_average :=
by sorry

/-- There were 5 other contributions made before John's -/
theorem five_initial_contributions :
  let num_initial_contributions := 5
  num_initial_contributions = 5 :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_is_260_average_increase_70_percent_new_average_is_85_five_initial_contributions_l25_2580


namespace NUMINAMATH_CALUDE_pineapple_problem_l25_2509

/-- Calculates the number of rotten pineapples given the initial count, sold count, and remaining fresh count. -/
def rottenPineapples (initial sold fresh : ℕ) : ℕ :=
  initial - sold - fresh

/-- Theorem stating that given the specific conditions from the problem, 
    the number of rotten pineapples thrown away is 9. -/
theorem pineapple_problem : rottenPineapples 86 48 29 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_problem_l25_2509


namespace NUMINAMATH_CALUDE_trigonometric_identity_l25_2574

theorem trigonometric_identity : 
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l25_2574


namespace NUMINAMATH_CALUDE_mikes_total_payment_l25_2520

/-- Calculates the amount Mike needs to pay after insurance coverage for his medical tests. -/
def mikes_payment (xray_cost : ℚ) (blood_test_cost : ℚ) : ℚ :=
  let mri_cost := 3 * xray_cost
  let ct_scan_cost := 2 * mri_cost
  let xray_payment := xray_cost * (1 - 0.8)
  let mri_payment := mri_cost * (1 - 0.8)
  let ct_scan_payment := ct_scan_cost * (1 - 0.7)
  let blood_test_payment := blood_test_cost * (1 - 0.5)
  xray_payment + mri_payment + ct_scan_payment + blood_test_payment

/-- Theorem stating that Mike's payment after insurance coverage is $750. -/
theorem mikes_total_payment :
  mikes_payment 250 200 = 750 := by
  sorry

end NUMINAMATH_CALUDE_mikes_total_payment_l25_2520


namespace NUMINAMATH_CALUDE_four_more_laps_needed_l25_2570

/-- Calculates the number of additional laps needed to reach a total distance -/
def additional_laps_needed (total_distance : ℕ) (track_length : ℕ) (laps_run_per_person : ℕ) (num_people : ℕ) : ℕ :=
  let total_laps_run := laps_run_per_person * num_people
  let distance_covered := total_laps_run * track_length
  let remaining_distance := total_distance - distance_covered
  remaining_distance / track_length

/-- Theorem: Given the problem conditions, 4 additional laps are needed -/
theorem four_more_laps_needed :
  additional_laps_needed 2400 150 6 2 = 4 := by
  sorry

#eval additional_laps_needed 2400 150 6 2

end NUMINAMATH_CALUDE_four_more_laps_needed_l25_2570


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l25_2562

theorem unique_three_digit_divisible_by_11 : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧          -- units digit is 2
  n / 100 = 7 ∧         -- hundreds digit is 7
  n % 11 = 0 ∧          -- divisible by 11
  n = 792 := by          -- the number is 792
sorry


end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l25_2562


namespace NUMINAMATH_CALUDE_sum_of_consecutive_primes_has_three_prime_factors_l25_2530

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p < q ∧ ∀ n, p < n → n < q → ¬(is_prime n)

theorem sum_of_consecutive_primes_has_three_prime_factors (p q : ℕ) :
  p > 2 → q > 2 → consecutive_primes p q →
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ p + q = a * b * c :=
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_primes_has_three_prime_factors_l25_2530


namespace NUMINAMATH_CALUDE_tickets_spent_on_beanie_l25_2571

/-- Proves the number of tickets spent on a beanie given initial tickets, additional tickets won, and final ticket count. -/
theorem tickets_spent_on_beanie 
  (initial_tickets : ℕ) 
  (additional_tickets : ℕ) 
  (final_tickets : ℕ) 
  (h1 : initial_tickets = 49)
  (h2 : additional_tickets = 6)
  (h3 : final_tickets = 30)
  : initial_tickets - (initial_tickets - final_tickets + additional_tickets) = 25 := by
  sorry

end NUMINAMATH_CALUDE_tickets_spent_on_beanie_l25_2571


namespace NUMINAMATH_CALUDE_beginner_course_fraction_l25_2551

theorem beginner_course_fraction :
  ∀ (total_students : ℕ) (calculus_students : ℕ) (trigonometry_students : ℕ) 
    (beginner_calculus : ℕ) (beginner_trigonometry : ℕ),
  total_students > 0 →
  calculus_students + trigonometry_students = total_students →
  trigonometry_students = (3 * calculus_students) / 2 →
  beginner_calculus = (4 * calculus_students) / 5 →
  (beginner_trigonometry : ℚ) / total_students = 48 / 100 →
  (beginner_calculus + beginner_trigonometry : ℚ) / total_students = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_beginner_course_fraction_l25_2551
