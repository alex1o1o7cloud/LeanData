import Mathlib

namespace NUMINAMATH_CALUDE_garden_width_l239_23918

/-- A rectangular garden with given length and area has a specific width. -/
theorem garden_width (length area : ℝ) (h1 : length = 12) (h2 : area = 60) :
  area / length = 5 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_l239_23918


namespace NUMINAMATH_CALUDE_one_and_one_third_problem_l239_23999

theorem one_and_one_third_problem :
  ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_one_and_one_third_problem_l239_23999


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l239_23962

theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 2 * d + g) * (4 * d^2 + h * d - 6) = 
             20 * d^4 - 18 * d^3 + 7 * d^2 + 10 * d - 18) →
  g + h = 7/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l239_23962


namespace NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l239_23913

theorem cubic_and_quadratic_equations :
  (∃ x : ℝ, 8 * x^3 = 27 ∧ x = 3/2) ∧
  (∃ x y : ℝ, (x - 2)^2 = 3 ∧ (y - 2)^2 = 3 ∧ 
   x = Real.sqrt 3 + 2 ∧ y = -Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_and_quadratic_equations_l239_23913


namespace NUMINAMATH_CALUDE_billy_cherries_l239_23977

theorem billy_cherries (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 74 → remaining = 2 → eaten = initial - remaining → eaten = 72 := by
  sorry

end NUMINAMATH_CALUDE_billy_cherries_l239_23977


namespace NUMINAMATH_CALUDE_subtract_negative_l239_23944

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l239_23944


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l239_23968

theorem sufficient_not_necessary_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l239_23968


namespace NUMINAMATH_CALUDE_extreme_points_condition_l239_23922

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a*x + Real.log x

theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 
   (∀ x > 0, f a x ≥ f a x₁ ∨ f a x ≥ f a x₂) ∧
   |f a x₁ - f a x₂| ≥ 3/4 - Real.log 2) →
  a ≥ 3 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_extreme_points_condition_l239_23922


namespace NUMINAMATH_CALUDE_inserted_square_side_length_l239_23994

/-- An isosceles triangle with a square inserted -/
structure TriangleWithSquare where
  /-- Length of the lateral sides of the isosceles triangle -/
  lateral_side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Side length of the inserted square -/
  square_side : ℝ

/-- Theorem: In an isosceles triangle with lateral sides of 10 and base of 12, 
    the side length of an inserted square is 24/5 -/
theorem inserted_square_side_length 
  (t : TriangleWithSquare) 
  (h1 : t.lateral_side = 10) 
  (h2 : t.base = 12) : 
  t.square_side = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_inserted_square_side_length_l239_23994


namespace NUMINAMATH_CALUDE_cube_sum_inequality_equality_iff_condition_l239_23990

/-- For any pairwise distinct natural numbers a, b, and c, 
    (a³ + b³ + c³) / 3 ≥ abc + a + b + c holds. -/
theorem cube_sum_inequality (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

/-- Characterization of when equality holds in the cube sum inequality. -/
def equality_condition (a b c : ℕ) : Prop :=
  (a = b + 1 ∧ b = c + 1) ∨ 
  (b = a + 1 ∧ a = c + 1) ∨ 
  (c = a + 1 ∧ a = b + 1)

/-- The equality condition is necessary and sufficient for the cube sum inequality to be an equality. -/
theorem equality_iff_condition (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ equality_condition a b c :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_equality_iff_condition_l239_23990


namespace NUMINAMATH_CALUDE_overbridge_length_l239_23943

/-- Calculates the length of an overbridge given train parameters --/
theorem overbridge_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 600 →
  train_speed_kmh = 36 →
  crossing_time = 70 →
  (train_length + (train_speed_kmh * 1000 / 3600 * crossing_time)) - train_length = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_overbridge_length_l239_23943


namespace NUMINAMATH_CALUDE_inequality_proof_l239_23929

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l239_23929


namespace NUMINAMATH_CALUDE_expression_equals_zero_l239_23915

theorem expression_equals_zero (x : ℚ) (h : x = 1/3) :
  (2*x + 1) * (2*x - 1) + x * (3 - 4*x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l239_23915


namespace NUMINAMATH_CALUDE_and_or_sufficient_not_necessary_l239_23987

theorem and_or_sufficient_not_necessary :
  (∃ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_or_sufficient_not_necessary_l239_23987


namespace NUMINAMATH_CALUDE_curve_equation_l239_23902

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation (t : ℝ) :
  let a : ℝ := 1/9
  let b : ℝ := -4/15
  let c : ℝ := 19/375
  a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_l239_23902


namespace NUMINAMATH_CALUDE_triangle_trig_inequality_triangle_trig_equality_l239_23919

/-- For any triangle ABC, sin A + sin B sin C + cos B cos C ≤ 2 -/
theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B * Real.sin C + Real.cos B * Real.cos C ≤ 2 :=
sorry

/-- The equality holds when A = π/2 and B = C = π/4 -/
theorem triangle_trig_equality :
  Real.sin (Real.pi/2) + Real.sin (Real.pi/4) * Real.sin (Real.pi/4) + 
  Real.cos (Real.pi/4) * Real.cos (Real.pi/4) = 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_trig_inequality_triangle_trig_equality_l239_23919


namespace NUMINAMATH_CALUDE_product_seventeen_reciprocal_squares_sum_l239_23954

theorem product_seventeen_reciprocal_squares_sum (x y : ℕ) :
  x * y = 17 → (1 : ℚ) / x^2 + 1 / y^2 = 290 / 289 := by
  sorry

end NUMINAMATH_CALUDE_product_seventeen_reciprocal_squares_sum_l239_23954


namespace NUMINAMATH_CALUDE_possible_theta_value_l239_23937

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem possible_theta_value :
  ∃ θ : ℝ,
    (∀ x : ℝ, (2015 : ℝ) ^ (f θ (-x)) = 1 / ((2015 : ℝ) ^ (f θ x))) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π/4 → f θ y < f θ x) ∧
    θ = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_possible_theta_value_l239_23937


namespace NUMINAMATH_CALUDE_remainder_of_factorial_sum_l239_23960

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_of_factorial_sum (n : ℕ) (h : n ≥ 100) :
  (sum_factorials n) % 30 = (sum_factorials 4) % 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_factorial_sum_l239_23960


namespace NUMINAMATH_CALUDE_cos_equality_problem_l239_23982

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 270 → 
  Real.cos (n * π / 180) = Real.cos (962 * π / 180) →
  n = 118 := by sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l239_23982


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l239_23934

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in "BANANA" -/
def total_letters : ℕ := 6

/-- The number of 'A's in "BANANA" -/
def num_a : ℕ := 3

/-- The number of 'N's in "BANANA" -/
def num_n : ℕ := 2

/-- The number of 'B's in "BANANA" -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l239_23934


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l239_23953

/-- A parallelogram with an area of 200 sq m and an altitude that is twice the corresponding base has a base length of 10 meters. -/
theorem parallelogram_base_length (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 200 →
  altitude = 2 * base →
  area = base * altitude →
  base = 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l239_23953


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l239_23921

-- Define the right triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = 180

-- Define the right angle at C
def RightAngleAtC (C : ℝ) : Prop := C = 90

-- Define the sine of angle A
def SineA (sinA : ℝ) : Prop := sinA = Real.sqrt 5 / 3

-- Define the length of side BC
def LengthBC (BC : ℝ) : Prop := BC = 2 * Real.sqrt 5

-- Theorem statement
theorem right_triangle_side_length 
  (A B C AC BC : ℝ) 
  (h_triangle : Triangle A B C) 
  (h_right_angle : RightAngleAtC C) 
  (h_sine_A : SineA (Real.sin (A * π / 180))) 
  (h_BC : LengthBC BC) : 
  AC = 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l239_23921


namespace NUMINAMATH_CALUDE_radio_station_survey_l239_23956

theorem radio_station_survey (total_listeners total_non_listeners male_non_listeners female_listeners : ℕ) 
  (h1 : total_listeners = 180)
  (h2 : total_non_listeners = 160)
  (h3 : male_non_listeners = 85)
  (h4 : female_listeners = 75) :
  total_listeners - female_listeners = 105 := by
  sorry

end NUMINAMATH_CALUDE_radio_station_survey_l239_23956


namespace NUMINAMATH_CALUDE_inequality_proof_l239_23985

theorem inequality_proof (a b : ℝ) (h : |a + b| ≤ 2) :
  |a^2 + 2*a - b^2 + 2*b| ≤ 4*(|a| + 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l239_23985


namespace NUMINAMATH_CALUDE_cube_order_preserving_l239_23900

theorem cube_order_preserving (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_order_preserving_l239_23900


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_280_l239_23931

/-- Calculates Scout's earnings for the weekend given the specified conditions --/
def scout_weekend_earnings (base_pay : ℕ) (sat_hours : ℕ) (sat_deliveries : ℕ) (sat_tip : ℕ)
  (sun_hours : ℕ) (sun_deliveries : ℕ) (sun_low_tip : ℕ) (sun_high_tip : ℕ)
  (transport_cost : ℕ) : ℕ :=
  let sat_earnings := base_pay * sat_hours + sat_deliveries * sat_tip - sat_deliveries * transport_cost
  let sun_earnings := 2 * base_pay * sun_hours + (sun_deliveries / 2) * (sun_low_tip + sun_high_tip) - sun_deliveries * transport_cost
  sat_earnings + sun_earnings

/-- Theorem stating that Scout's weekend earnings are $280.00 --/
theorem scout_weekend_earnings_280 :
  scout_weekend_earnings 10 6 5 5 8 10 3 7 1 = 280 := by
  sorry

#eval scout_weekend_earnings 10 6 5 5 8 10 3 7 1

end NUMINAMATH_CALUDE_scout_weekend_earnings_280_l239_23931


namespace NUMINAMATH_CALUDE_opposite_sides_range_l239_23933

def line_equation (x y a : ℝ) : ℝ := 3 * x - 2 * y + a

theorem opposite_sides_range (a : ℝ) : 
  (line_equation 3 1 a) * (line_equation (-4) 6 a) < 0 ↔ -7 < a ∧ a < 24 := by sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l239_23933


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_1320_l239_23998

theorem sum_of_distinct_prime_factors_1320 :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (1320 + 1)))
    (fun p => if p ∣ 1320 then p else 0)) = 21 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_1320_l239_23998


namespace NUMINAMATH_CALUDE_T_equals_one_l239_23988

theorem T_equals_one (S : ℝ) : 
  let T := Real.sin (50 * π / 180) * (S + Real.sqrt 3 * Real.tan (10 * π / 180))
  T = 1 :=
by sorry

end NUMINAMATH_CALUDE_T_equals_one_l239_23988


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l239_23949

def north_students : ℕ := 1800
def south_students : ℕ := 2700
def north_tennis_percent : ℚ := 25 / 100
def south_tennis_percent : ℚ := 35 / 100

theorem combined_tennis_percentage :
  let total_students := north_students + south_students
  let north_tennis := (north_students : ℚ) * north_tennis_percent
  let south_tennis := (south_students : ℚ) * south_tennis_percent
  let total_tennis := north_tennis + south_tennis
  (total_tennis / total_students) * 100 = 31 := by
sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l239_23949


namespace NUMINAMATH_CALUDE_A_power_150_is_identity_l239_23959

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_150_is_identity :
  A ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end NUMINAMATH_CALUDE_A_power_150_is_identity_l239_23959


namespace NUMINAMATH_CALUDE_triangle_problem_l239_23946

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The cosine of an angle in a triangle -/
def cosine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The sine of an angle in a triangle -/
def sine (t : Triangle) (angle : ℕ) : ℝ :=
  sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - (3/4) * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (k : ℝ), sine t 1 = k - (sine t 2) ∧ sine t 3 = k + (sine t 2)) :
  cosine t 2 = 5/8 ∧ area t = (3 * Real.sqrt 39) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l239_23946


namespace NUMINAMATH_CALUDE_at_least_one_divisible_by_three_l239_23927

theorem at_least_one_divisible_by_three (a b : ℤ) : 
  (3 ∣ a) ∨ (3 ∣ b) ∨ (3 ∣ (a + b)) ∨ (3 ∣ (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_divisible_by_three_l239_23927


namespace NUMINAMATH_CALUDE_white_ball_count_l239_23905

/-- Given a bag with red and white balls, if the probability of drawing a red ball
    is 1/4 and there are 5 red balls, prove that there are 15 white balls. -/
theorem white_ball_count (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
    (h1 : red_balls = 5)
    (h2 : total_balls = red_balls + white_balls)
    (h3 : (red_balls : ℚ) / total_balls = 1 / 4) :
  white_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l239_23905


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l239_23991

theorem number_of_divisors_of_60 : ∃ (s : Finset Nat), ∀ d : Nat, d ∈ s ↔ d ∣ 60 ∧ d > 0 ∧ Finset.card s = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l239_23991


namespace NUMINAMATH_CALUDE_power_zero_simplify_expression_l239_23907

-- Theorem 1: For any real number x ≠ 0, x^0 = 1
theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

-- Theorem 2: For any real numbers a and b, (-2a^2)^2 * 3ab^2 = 12a^5b^2
theorem simplify_expression (a b : ℝ) : (-2*a^2)^2 * 3*a*b^2 = 12*a^5*b^2 := by sorry

end NUMINAMATH_CALUDE_power_zero_simplify_expression_l239_23907


namespace NUMINAMATH_CALUDE_train_speed_l239_23916

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 6) :
  length / time = 140 / 6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l239_23916


namespace NUMINAMATH_CALUDE_expected_waiting_time_for_last_suitcase_l239_23992

theorem expected_waiting_time_for_last_suitcase 
  (total_suitcases : ℕ) 
  (business_suitcases : ℕ) 
  (placement_interval : ℕ) 
  (h1 : total_suitcases = 200) 
  (h2 : business_suitcases = 10) 
  (h3 : placement_interval = 2) :
  (((total_suitcases + 1) * placement_interval * business_suitcases) / (business_suitcases + 1) : ℚ) = 4020 / 11 := by
  sorry

#check expected_waiting_time_for_last_suitcase

end NUMINAMATH_CALUDE_expected_waiting_time_for_last_suitcase_l239_23992


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l239_23906

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (x - 2)^6 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + a₅*(x+1)^5 + a₆*(x+1)^6) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l239_23906


namespace NUMINAMATH_CALUDE_people_owning_cats_and_dogs_l239_23983

theorem people_owning_cats_and_dogs (
  total_pet_owners : ℕ)
  (only_dog_owners : ℕ)
  (only_cat_owners : ℕ)
  (cat_dog_snake_owners : ℕ)
  (h1 : total_pet_owners = 69)
  (h2 : only_dog_owners = 15)
  (h3 : only_cat_owners = 10)
  (h4 : cat_dog_snake_owners = 3) :
  total_pet_owners = only_dog_owners + only_cat_owners + 41 + cat_dog_snake_owners :=
by
  sorry

end NUMINAMATH_CALUDE_people_owning_cats_and_dogs_l239_23983


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l239_23930

/-- Given the initial number of small and big bottles, the percentage of big bottles sold,
    and the total number of bottles remaining, prove that 15% of small bottles were sold. -/
theorem small_bottles_sold_percentage
  (initial_small : ℕ)
  (initial_big : ℕ)
  (big_bottles_sold_percent : ℚ)
  (total_remaining : ℕ)
  (h1 : initial_small = 5000)
  (h2 : initial_big = 12000)
  (h3 : big_bottles_sold_percent = 18/100)
  (h4 : total_remaining = 14090)
  (h5 : total_remaining = initial_small + initial_big -
        (initial_small * small_bottles_sold_percent / 100 +
         initial_big * big_bottles_sold_percent).floor) :
  small_bottles_sold_percent = 15/100 :=
sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l239_23930


namespace NUMINAMATH_CALUDE_sequence_properties_l239_23996

-- Define the sequence a_n and its partial sum S_n
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n - 2^n

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n, S n a = 2 * a n - 2^n) →
  (∃ r : ℝ, ∀ n, a (n + 1) - 2 * a n = r * (a n - 2 * a (n - 1))) ∧
  (∀ n, a n = (n + 1) * 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l239_23996


namespace NUMINAMATH_CALUDE_michelle_initial_crayons_l239_23932

theorem michelle_initial_crayons :
  ∀ (michelle_initial janet : ℕ),
    janet = 2 →
    michelle_initial + janet = 4 →
    michelle_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_michelle_initial_crayons_l239_23932


namespace NUMINAMATH_CALUDE_midpoint_locus_l239_23925

/-- The locus of midpoints of line segments from P(4, -2) to points on x^2 + y^2 = 4 -/
theorem midpoint_locus (x y u v : ℝ) : 
  (u^2 + v^2 = 4) →  -- Point (u, v) is on the circle
  (x = (u + 4) / 2 ∧ y = (v - 2) / 2) →  -- (x, y) is the midpoint
  (x - 2)^2 + (y + 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_locus_l239_23925


namespace NUMINAMATH_CALUDE_trader_markup_percentage_l239_23980

theorem trader_markup_percentage (discount : ℝ) (loss : ℝ) : 
  discount = 7.857142857142857 / 100 →
  loss = 1 / 100 →
  ∃ (markup : ℝ), 
    (1 + markup) * (1 - discount) = 1 - loss ∧ 
    abs (markup - 7.4285714285714 / 100) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_trader_markup_percentage_l239_23980


namespace NUMINAMATH_CALUDE_multiple_of_nine_is_multiple_of_three_l239_23939

theorem multiple_of_nine_is_multiple_of_three (n : ℤ) : 
  (∃ k : ℤ, n = 9 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_nine_is_multiple_of_three_l239_23939


namespace NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l239_23978

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10
  let b : ℚ := 5/1000
  (a + b) / 2 = 2025/10000 := by
  sorry

end NUMINAMATH_CALUDE_average_of_tenths_and_thousandths_l239_23978


namespace NUMINAMATH_CALUDE_identical_projections_implies_sphere_or_cube_l239_23986

/-- A solid is a three-dimensional object. -/
structure Solid :=
  (shape : Type)

/-- Orthographic projection is a method of representing a 3D object in 2D. -/
def orthographic_projection (s : Solid) : Type := sorry

/-- A solid has identical orthographic projections if all three standard views are the same. -/
def has_identical_projections (s : Solid) : Prop :=
  ∃ (view : orthographic_projection s), ∀ (p : orthographic_projection s), p = view

/-- The theorem states that if a solid has identical orthographic projections,
    it can be either a sphere or a cube. -/
theorem identical_projections_implies_sphere_or_cube (s : Solid) :
  has_identical_projections s → (s.shape = Sphere ∨ s.shape = Cube) :=
sorry

end NUMINAMATH_CALUDE_identical_projections_implies_sphere_or_cube_l239_23986


namespace NUMINAMATH_CALUDE_yard_area_l239_23984

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_side : ℝ) 
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : cut_side = 3) : 
  length * width - cut_side^2 = 171 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l239_23984


namespace NUMINAMATH_CALUDE_randy_blocks_total_l239_23910

theorem randy_blocks_total (house_blocks tower_blocks : ℕ) 
  (house_tower_diff : ℕ) (total_blocks : ℕ) : 
  house_blocks = 20 →
  tower_blocks = 50 →
  tower_blocks = house_blocks + house_tower_diff →
  house_tower_diff = 30 →
  total_blocks = house_blocks + tower_blocks →
  total_blocks = 70 := by
sorry

end NUMINAMATH_CALUDE_randy_blocks_total_l239_23910


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l239_23989

theorem sum_seven_consecutive_integers (m : ℤ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l239_23989


namespace NUMINAMATH_CALUDE_average_of_combined_data_points_l239_23923

theorem average_of_combined_data_points (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  let total_points := n1 + n2
  let combined_avg := (n1 * avg1 + n2 * avg2) / total_points
  combined_avg = (n1 * avg1 + n2 * avg2) / (n1 + n2) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_data_points_l239_23923


namespace NUMINAMATH_CALUDE_p_recurrence_l239_23914

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n, k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
sorry

end NUMINAMATH_CALUDE_p_recurrence_l239_23914


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l239_23938

def polynomial (x : ℝ) : ℝ := 5*(x^2 - 2*x^3) + 3*(2*x - 3*x^2 + x^4) - (6*x^3 - 2*x^2)

theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + b*x^3 + (-2)*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l239_23938


namespace NUMINAMATH_CALUDE_problem_statement_l239_23995

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x
noncomputable def g (x a : ℝ) : ℝ := Real.log x - a * x + 1

theorem problem_statement :
  (∀ x : ℝ, x > 0 → deriv f x = Real.log x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g x a ≤ 0) → a ≥ 1) ∧
  (∀ m x n : ℝ, 0 < m → m < x → x < n →
    (f x - f m) / (x - m) < (f x - f n) / (x - n)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l239_23995


namespace NUMINAMATH_CALUDE_fraction_value_l239_23967

theorem fraction_value : (150 + (150 / 10)) / (15 - 5) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l239_23967


namespace NUMINAMATH_CALUDE_original_number_reciprocal_l239_23920

theorem original_number_reciprocal (x : ℝ) : 1 / x - 3 = 5 / 2 → x = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_number_reciprocal_l239_23920


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l239_23957

theorem triangle_radii_inequality (r R α β γ : Real) : 
  r > 0 → R > 0 → 
  0 < α ∧ α < π → 0 < β ∧ β < π → 0 < γ ∧ γ < π →
  α + β + γ = π →
  r / R ≤ 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l239_23957


namespace NUMINAMATH_CALUDE_kConnectedSubgraph_l239_23961

/-- A graph G is a pair (V, E) where V is a finite set of vertices and E is a set of edges. -/
structure Graph (α : Type*) where
  V : Finset α
  E : Finset (α × α)

/-- The minimum degree of a graph G. -/
def minDegree {α : Type*} (G : Graph α) : ℕ :=
  sorry

/-- A graph G is k-connected if it remains connected after removing any k-1 vertices. -/
def isKConnected {α : Type*} (G : Graph α) (k : ℕ) : Prop :=
  sorry

/-- A subgraph H of G is a graph whose vertices and edges are subsets of G's vertices and edges. -/
def isSubgraph {α : Type*} (H G : Graph α) : Prop :=
  sorry

/-- The main theorem stating that if δ(G) ≥ 8k and |G| ≤ 16k, then G contains a k-connected subgraph. -/
theorem kConnectedSubgraph {α : Type*} (G : Graph α) (k : ℕ) :
  minDegree G ≥ 8 * k →
  G.V.card ≤ 16 * k →
  ∃ H : Graph α, isSubgraph H G ∧ isKConnected H k :=
sorry

end NUMINAMATH_CALUDE_kConnectedSubgraph_l239_23961


namespace NUMINAMATH_CALUDE_shooting_performance_and_probability_l239_23979

def shooter_A_scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]
def shooter_B_scores : List ℕ := [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m)^2)).sum / scores.length

def is_excellent (score : ℕ) : Bool :=
  score ≥ 8

def excellent_probability (scores : List ℕ) : ℚ :=
  (scores.filter is_excellent).length / scores.length

theorem shooting_performance_and_probability :
  (variance shooter_B_scores < variance shooter_A_scores) ∧
  (excellent_probability shooter_A_scores + excellent_probability shooter_B_scores = 19/25) := by
  sorry

end NUMINAMATH_CALUDE_shooting_performance_and_probability_l239_23979


namespace NUMINAMATH_CALUDE_remi_and_father_seedlings_l239_23976

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings (day1 : ℕ) (total : ℕ) : ℕ :=
  total - (day1 + 2 * day1)

theorem remi_and_father_seedlings :
  fathers_seedlings 200 1200 = 600 := by
  sorry

end NUMINAMATH_CALUDE_remi_and_father_seedlings_l239_23976


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l239_23909

theorem complex_number_in_third_quadrant (z : ℂ) (h : (z + Complex.I) * Complex.I = 1 + z) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l239_23909


namespace NUMINAMATH_CALUDE_common_chord_circles_l239_23947

/-- Given two circles with equations x^2 + (y - 3/2)^2 = 25/4 and x^2 + y^2 = m,
    if they have a common chord passing through the point (0, 3/2), then m = 17/2. -/
theorem common_chord_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y - 3/2)^2 = 25/4 ∧ x^2 + y^2 = m) ∧ 
  (∃ (x : ℝ), x^2 + (3/2 - 3/2)^2 = 25/4 ∧ x^2 + (3/2)^2 = m) →
  m = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_circles_l239_23947


namespace NUMINAMATH_CALUDE_family_birth_years_l239_23903

def current_year : ℕ := 1967

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def satisfies_condition (birth_year : ℕ) (multiplier : ℕ) : Prop :=
  current_year - birth_year = multiplier * sum_of_digits birth_year

theorem family_birth_years :
  ∃ (grandpa eldest_son father pali brother mother grandfather grandmother : ℕ),
    satisfies_condition grandpa 3 ∧
    satisfies_condition eldest_son 3 ∧
    satisfies_condition father 3 = false ∧
    satisfies_condition (father - 1) 3 ∧
    satisfies_condition grandfather 3 = false ∧
    satisfies_condition (grandfather - 1) 3 ∧
    satisfies_condition grandmother 3 = false ∧
    satisfies_condition (grandmother + 1) 3 ∧
    satisfies_condition mother 2 = false ∧
    satisfies_condition (mother - 1) 2 ∧
    satisfies_condition pali 1 ∧
    satisfies_condition brother 1 = false ∧
    satisfies_condition (brother - 1) 1 ∧
    grandpa = 1889 ∧
    eldest_son = 1916 ∧
    father = 1928 ∧
    pali = 1951 ∧
    brother = 1947 ∧
    mother = 1934 ∧
    grandfather = 1896 ∧
    grandmother = 1909 :=
by
  sorry

end NUMINAMATH_CALUDE_family_birth_years_l239_23903


namespace NUMINAMATH_CALUDE_sequence_increasing_l239_23975

theorem sequence_increasing (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_rel : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a (n + 1) > a n :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_l239_23975


namespace NUMINAMATH_CALUDE_license_combinations_l239_23997

/-- Represents the number of choices for the letter in a license -/
def letter_choices : ℕ := 3

/-- Represents the number of choices for each digit in a license -/
def digit_choices : ℕ := 10

/-- Represents the number of digits in a license -/
def num_digits : ℕ := 4

/-- Calculates the total number of possible license combinations -/
def total_combinations : ℕ := letter_choices * digit_choices ^ num_digits

/-- Proves that the number of unique license combinations is 30000 -/
theorem license_combinations : total_combinations = 30000 := by
  sorry

end NUMINAMATH_CALUDE_license_combinations_l239_23997


namespace NUMINAMATH_CALUDE_range_of_a_l239_23901

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | f (x^2 + 1) > f (a * x)}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → x ∈ A a) →
  a ∈ Set.Ioo (-5/2) (5/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l239_23901


namespace NUMINAMATH_CALUDE_area_triangle_abc_l239_23969

/-- Given a point A(x, y) where x ≠ 0 and y ≠ 0, with B symmetric to A with respect to the x-axis,
    C symmetric to A with respect to the y-axis, and the area of triangle AOB equal to 4,
    prove that the area of triangle ABC is equal to 8. -/
theorem area_triangle_abc (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  let A : ℝ × ℝ := (x, y)
  let B : ℝ × ℝ := (x, -y)
  let C : ℝ × ℝ := (-x, y)
  let O : ℝ × ℝ := (0, 0)
  let area_AOB := abs (x * y)
  area_AOB = 4 → abs (2 * x * y) = 8 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_abc_l239_23969


namespace NUMINAMATH_CALUDE_inequality_implication_l239_23970

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l239_23970


namespace NUMINAMATH_CALUDE_max_value_of_5x_minus_25x_l239_23948

theorem max_value_of_5x_minus_25x : 
  ∃ (max : ℝ), max = 1/4 ∧ ∀ x : ℝ, 5^x - 25^x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_5x_minus_25x_l239_23948


namespace NUMINAMATH_CALUDE_totalDays_is_25_l239_23952

/-- Calculates the total number of days in a work period given the following conditions:
  * A woman is paid $20 for each day she works
  * She forfeits $5 for each day she is idle
  * She nets $450
  * She worked for 23 days
-/
def totalDaysInPeriod (dailyPay : ℕ) (dailyForfeit : ℕ) (netEarnings : ℕ) (daysWorked : ℕ) : ℕ :=
  sorry

/-- Proves that the total number of days in the period is 25 -/
theorem totalDays_is_25 :
  totalDaysInPeriod 20 5 450 23 = 25 := by
  sorry

end NUMINAMATH_CALUDE_totalDays_is_25_l239_23952


namespace NUMINAMATH_CALUDE_range_of_f_l239_23963

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (-x^2 + 2*x)

theorem range_of_f :
  Set.range f = Set.Ioo (1/2) (Real.pi) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l239_23963


namespace NUMINAMATH_CALUDE_geometric_sum_of_powers_of_five_l239_23955

theorem geometric_sum_of_powers_of_five : 
  (Finset.range 6).sum (fun i => 5^(i+1)) = 19530 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_of_powers_of_five_l239_23955


namespace NUMINAMATH_CALUDE_root_equation_problem_l239_23966

theorem root_equation_problem (c d : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ 
    (∀ x : ℝ, (x + c) * (x + d) * (x + 10) / (x + 2)^2 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3)) ∧
  (∃! (r : ℝ), ∀ x : ℝ, (x + 2*c) * (x + 4) * (x + 8) / ((x + d) * (x + 10)) = 0 ↔ x = r) →
  200 * c + d = 392 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_problem_l239_23966


namespace NUMINAMATH_CALUDE_triangle_problem_l239_23945

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
    (h1 : Real.cos t.B * (Real.sqrt 3 * t.a - t.b * Real.sin t.C) - t.b * Real.sin t.B * Real.cos t.C = 0)
    (h2 : t.c = 2 * t.a)
    (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
    t.B = π / 3 ∧ t.a + t.b + t.c = 3 * Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l239_23945


namespace NUMINAMATH_CALUDE_parallelogram_formation_condition_l239_23971

/-- Represents a point in a one-dimensional space -/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points -/
def LineSegment (P Q : Point) : ℝ :=
  |Q.x - P.x|

/-- Condition for forming a parallelogram when rotating line segments -/
def ParallelogramCondition (P Q R S T : Point) (a b c : ℝ) : Prop :=
  P.x < Q.x ∧ Q.x < R.x ∧ R.x < S.x ∧ S.x < T.x ∧
  LineSegment P Q = a ∧
  LineSegment P R = b ∧
  LineSegment P T = c ∧
  b = c - a

theorem parallelogram_formation_condition 
  (P Q R S T : Point) (a b c : ℝ) :
  ParallelogramCondition P Q R S T a b c →
  ∃ (P' T' : Point),
    LineSegment Q P' = a ∧
    LineSegment R T' = c - b ∧
    LineSegment P' T' = b - a ∧
    LineSegment S P' = LineSegment S T' :=
sorry

end NUMINAMATH_CALUDE_parallelogram_formation_condition_l239_23971


namespace NUMINAMATH_CALUDE_initial_disappearance_percentage_l239_23942

/-- Proof of the initial percentage of inhabitants that disappeared from a village --/
theorem initial_disappearance_percentage 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (initial_population_eq : initial_population = 7600)
  (final_population_eq : final_population = 5130) :
  ∃ (p : ℝ), 
    p = 10 ∧ 
    (initial_population : ℝ) * (1 - p / 100) * 0.75 = final_population := by
  sorry

end NUMINAMATH_CALUDE_initial_disappearance_percentage_l239_23942


namespace NUMINAMATH_CALUDE_beach_trip_time_l239_23974

theorem beach_trip_time :
  let drive_time_one_way : ℝ := 2
  let total_drive_time : ℝ := 2 * drive_time_one_way
  let beach_time : ℝ := 2.5 * total_drive_time
  let total_trip_time : ℝ := total_drive_time + beach_time
  total_trip_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_time_l239_23974


namespace NUMINAMATH_CALUDE_walking_time_equals_early_arrival_l239_23908

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  D : ℝ  -- Total distance from station to home
  Vw : ℝ  -- Wife's driving speed
  Vm : ℝ  -- Man's walking speed
  T : ℝ  -- Usual time for wife to drive from station to home
  t : ℝ  -- Time man spent walking before being picked up
  early_arrival : ℝ  -- Time they arrived home earlier than usual

/-- The time the man spent walking is equal to the time they arrived home earlier --/
theorem walking_time_equals_early_arrival (scenario : WalkingScenario) 
  (h1 : scenario.D = scenario.Vw * scenario.T)
  (h2 : scenario.D - scenario.Vm * scenario.t = scenario.Vw * (scenario.T - scenario.t))
  (h3 : scenario.early_arrival = scenario.t) :
  scenario.t = scenario.early_arrival :=
by
  sorry

#check walking_time_equals_early_arrival

end NUMINAMATH_CALUDE_walking_time_equals_early_arrival_l239_23908


namespace NUMINAMATH_CALUDE_third_number_proof_l239_23950

theorem third_number_proof (A B C : ℕ+) : 
  A = 24 → B = 36 → Nat.gcd A (Nat.gcd B C) = 32 → Nat.lcm A (Nat.lcm B C) = 1248 → C = 32 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l239_23950


namespace NUMINAMATH_CALUDE_kevin_bought_two_watermelons_l239_23936

-- Define the weights of the watermelons and the total weight
def weight1 : ℝ := 9.91
def weight2 : ℝ := 4.11
def totalWeight : ℝ := 14.02

-- Define the number of watermelons Kevin bought
def numberOfWatermelons : ℕ := 2

-- Theorem to prove
theorem kevin_bought_two_watermelons :
  weight1 + weight2 = totalWeight ∧ numberOfWatermelons = 2 :=
by sorry

end NUMINAMATH_CALUDE_kevin_bought_two_watermelons_l239_23936


namespace NUMINAMATH_CALUDE_present_value_log_formula_l239_23904

theorem present_value_log_formula (c s P k n : ℝ) (h_pos : 0 < 1 + k) :
  P = c * s / (1 + k) ^ n →
  n = (Real.log (c * s / P)) / (Real.log (1 + k)) :=
by sorry

end NUMINAMATH_CALUDE_present_value_log_formula_l239_23904


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l239_23973

theorem rectangle_dimensions (area perimeter : ℝ) (h1 : area = 12) (h2 : perimeter = 26) :
  ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    ((length = 1 ∧ width = 12) ∨ (length = 12 ∧ width = 1)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l239_23973


namespace NUMINAMATH_CALUDE_square_diff_sqrt_l239_23951

theorem square_diff_sqrt : (Real.sqrt 81 - Real.sqrt 144)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sqrt_l239_23951


namespace NUMINAMATH_CALUDE_ab_equals_one_l239_23965

theorem ab_equals_one (a b : ℝ) (ha : a = Real.sqrt 3 / 3) (hb : b = Real.sqrt 3) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_one_l239_23965


namespace NUMINAMATH_CALUDE_textbook_order_cost_l239_23940

/-- Calculate the total cost of textbooks --/
def total_cost (english_count : ℕ) (english_price : ℚ)
                (geography_count : ℕ) (geography_price : ℚ)
                (math_count : ℕ) (math_price : ℚ)
                (science_count : ℕ) (science_price : ℚ) : ℚ :=
  english_count * english_price +
  geography_count * geography_price +
  math_count * math_price +
  science_count * science_price

/-- The total cost of the textbook order is $1155.00 --/
theorem textbook_order_cost :
  total_cost 35 (7.5) 35 (10.5) 20 12 30 (9.5) = 1155 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l239_23940


namespace NUMINAMATH_CALUDE_second_next_perfect_square_l239_23981

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end NUMINAMATH_CALUDE_second_next_perfect_square_l239_23981


namespace NUMINAMATH_CALUDE_g_over_log16_2_eq_4n_l239_23912

/-- Sum of squares of elements in nth row of Pascal's triangle -/
def pascal_row_sum_squares (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Base-16 logarithm function -/
noncomputable def log16 (x : ℝ) : ℝ := Real.log x / Real.log 16

/-- Function g(n) as defined in the problem -/
noncomputable def g (n : ℕ) : ℝ := log16 (pascal_row_sum_squares n)

/-- Theorem stating the relationship between g(n) and n -/
theorem g_over_log16_2_eq_4n (n : ℕ) : g n / log16 2 = 4 * n := by sorry

end NUMINAMATH_CALUDE_g_over_log16_2_eq_4n_l239_23912


namespace NUMINAMATH_CALUDE_garden_perimeter_l239_23917

/-- The perimeter of a rectangular garden with the same area as a given playground -/
theorem garden_perimeter (garden_width playground_length playground_width : ℝ) :
  garden_width = 4 ∧
  playground_length = 16 ∧
  playground_width = 12 →
  (garden_width * (playground_length * playground_width / garden_width) + garden_width) * 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l239_23917


namespace NUMINAMATH_CALUDE_quadratic_function_property_l239_23924

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties, 
    prove that a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  (f 0 = f 4) ∧ (f 0 > f 1) → a > 0 ∧ 4 * a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l239_23924


namespace NUMINAMATH_CALUDE_jesse_blocks_left_l239_23935

/-- The number of building blocks Jesse has left after constructing various structures --/
def blocks_left (initial : ℕ) (building : ℕ) (farmhouse : ℕ) (fence : ℕ) : ℕ :=
  initial - (building + farmhouse + fence)

/-- Theorem stating that Jesse has 84 blocks left --/
theorem jesse_blocks_left :
  blocks_left 344 80 123 57 = 84 := by
  sorry

end NUMINAMATH_CALUDE_jesse_blocks_left_l239_23935


namespace NUMINAMATH_CALUDE_min_value_of_P_l239_23911

/-- The polynomial P as a function of a real number a -/
def P (a : ℝ) : ℝ := a^2 + 4*a + 2014

/-- Theorem stating that the minimum value of P is 2010 -/
theorem min_value_of_P :
  ∃ (min : ℝ), min = 2010 ∧ ∀ (a : ℝ), P a ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_P_l239_23911


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_3n_l239_23958

/-- An arithmetic progression with partial sums S_n -/
structure ArithmeticProgression where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- Given S_n = a and S_{2n} = b, prove S_{3n} = 3b - 2a -/
theorem arithmetic_progression_sum_3n 
  (ap : ArithmeticProgression) (n : ℕ) (a b : ℝ) 
  (h1 : ap.S n = a) 
  (h2 : ap.S (2 * n) = b) : 
  ap.S (3 * n) = 3 * b - 2 * a := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_3n_l239_23958


namespace NUMINAMATH_CALUDE_abcdef_hex_bit_length_l239_23972

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- This case should not occur for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Calculates the number of bits needed to represent a natural number -/
def bit_length (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem abcdef_hex_bit_length :
  bit_length (hex_to_decimal "ABCDEF") = 24 := by
  sorry

#eval bit_length (hex_to_decimal "ABCDEF")

end NUMINAMATH_CALUDE_abcdef_hex_bit_length_l239_23972


namespace NUMINAMATH_CALUDE_second_discount_percentage_l239_23993

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_price : ℝ)
  (h1 : original_price = 175)
  (h2 : first_discount_percent = 20)
  (h3 : final_price = 133)
  : ∃ (second_discount_percent : ℝ),
    final_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) ∧
    second_discount_percent = 5 :=
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l239_23993


namespace NUMINAMATH_CALUDE_modified_fibonacci_sum_l239_23941

def G : ℕ → ℚ
  | 0 => 2
  | 1 => 1
  | (n + 2) => G (n + 1) + G n

theorem modified_fibonacci_sum :
  (∑' n, G n / 5^n) = 280 / 99 := by
  sorry

end NUMINAMATH_CALUDE_modified_fibonacci_sum_l239_23941


namespace NUMINAMATH_CALUDE_prop_2_prop_3_prop_4_l239_23964

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the existence of two distinct lines and two distinct planes
variable (a b : Line)
variable (α β : Plane)
variable (h_distinct_lines : a ≠ b)
variable (h_distinct_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (perpendicular a α ∧ perpendicular a β) → parallel_planes α β :=
sorry

-- Proposition ③
theorem prop_3 :
  perpendicular_planes α β → 
  ∃ γ : Plane, perpendicular_planes γ α ∧ perpendicular_planes γ β :=
sorry

-- Proposition ④
theorem prop_4 :
  perpendicular_planes α β → 
  ∃ l : Line, perpendicular l α ∧ parallel l β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_prop_4_l239_23964


namespace NUMINAMATH_CALUDE_bicycle_price_adjustment_l239_23928

theorem bicycle_price_adjustment (original_price : ℝ) 
  (wednesday_discount : ℝ) (friday_increase : ℝ) (saturday_discount : ℝ) : 
  original_price = 200 →
  wednesday_discount = 0.40 →
  friday_increase = 0.20 →
  saturday_discount = 0.25 →
  original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount) = 108 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_adjustment_l239_23928


namespace NUMINAMATH_CALUDE_remainder_s_15_plus_1_l239_23926

theorem remainder_s_15_plus_1 (s : ℤ) : (s^15 + 1) % (s - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_s_15_plus_1_l239_23926
