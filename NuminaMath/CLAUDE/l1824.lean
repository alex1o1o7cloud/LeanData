import Mathlib

namespace NUMINAMATH_CALUDE_repeating_base_k_representation_l1824_182471

theorem repeating_base_k_representation (k : ℕ) (h1 : k > 0) : 
  (4 * k + 5) / (k^2 - 1) = 11 / 143 → k = 52 :=
by sorry

end NUMINAMATH_CALUDE_repeating_base_k_representation_l1824_182471


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1824_182464

theorem inequality_and_equality_condition (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2) ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1824_182464


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l1824_182430

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + w)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) ≤ 2 ∧
  ∀ M : ℝ, M < 2 → ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    Real.sqrt (a / (b + c + d)) + Real.sqrt (b / (a + c + d)) +
    Real.sqrt (c / (a + b + d)) + Real.sqrt (d / (a + b + c)) > M :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l1824_182430


namespace NUMINAMATH_CALUDE_isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l1824_182498

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem isosceles_when_neg_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.b)^2 = 4 * (t.a + t.c) * (t.a - t.c) → t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) (h : t.a = t.b ∧ t.b = t.c) :
  ∃ x y, x = 0 ∧ y = -1 ∧ quadratic t x = 0 ∧ quadratic t y = 0 :=
sorry

end NUMINAMATH_CALUDE_isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l1824_182498


namespace NUMINAMATH_CALUDE_fair_draw_l1824_182437

/-- Represents the number of players in the game -/
def num_players : ℕ := 10

/-- Represents the number of red balls in the hat -/
def red_balls : ℕ := 1

/-- Represents the number of white balls in the hat -/
def white_balls (h : ℕ) : ℕ := 10 * h - 1

/-- The probability of the host drawing a red ball -/
def host_probability (k n : ℕ) : ℚ := k / (k + n)

/-- The probability of the next player drawing a red ball -/
def next_player_probability (k n : ℕ) : ℚ := (n / (k + n)) * (k / (k + n - 1))

/-- Theorem stating the condition for a fair draw -/
theorem fair_draw (h : ℕ) :
  host_probability red_balls (white_balls h) = next_player_probability red_balls (white_balls h) :=
sorry

end NUMINAMATH_CALUDE_fair_draw_l1824_182437


namespace NUMINAMATH_CALUDE_remainder_1999_div_7_l1824_182413

theorem remainder_1999_div_7 : 1999 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_div_7_l1824_182413


namespace NUMINAMATH_CALUDE_max_piles_is_30_l1824_182446

/-- Represents a configuration of stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : (piles.sum = 660)
  size_constraint : ∀ i j, i < piles.length → j < piles.length → 2 * piles[i]! > piles[j]!

/-- Represents a valid split operation on stone piles -/
def split (sp : StonePiles) (index : Nat) (amount : Nat) : Option StonePiles :=
  sorry

/-- The maximum number of piles that can be formed -/
def max_piles : Nat := 30

/-- Theorem stating that the maximum number of piles is 30 -/
theorem max_piles_is_30 :
  ∀ sp : StonePiles,
  (∀ index amount, split sp index amount = none) →
  sp.piles.length ≤ max_piles :=
sorry

end NUMINAMATH_CALUDE_max_piles_is_30_l1824_182446


namespace NUMINAMATH_CALUDE_smallest_bdf_value_l1824_182439

theorem smallest_bdf_value (a b c d e f : ℕ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) →
  (((a + 1) / b * c / d * e / f) - (a / b * c / d * e / f) = 3) →
  ((a / b * (c + 1) / d * e / f) - (a / b * c / d * e / f) = 4) →
  ((a / b * c / d * (e + 1) / f) - (a / b * c / d * e / f) = 5) →
  60 ≤ b * d * f ∧ ∃ (b' d' f' : ℕ), b' * d' * f' = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bdf_value_l1824_182439


namespace NUMINAMATH_CALUDE_jill_makes_30_trips_l1824_182457

/-- Represents the water-carrying problem with Jack and Jill --/
structure WaterProblem where
  tank_capacity : ℕ
  bucket_capacity : ℕ
  jack_buckets_per_trip : ℕ
  jill_buckets_per_trip : ℕ
  jack_trips_ratio : ℕ
  jill_trips_ratio : ℕ

/-- Calculates the number of trips Jill makes to fill the tank --/
def jill_trips (wp : WaterProblem) : ℕ :=
  let jack_water_per_trip := wp.jack_buckets_per_trip * wp.bucket_capacity
  let jill_water_per_trip := wp.jill_buckets_per_trip * wp.bucket_capacity
  let water_per_cycle := jack_water_per_trip * wp.jack_trips_ratio + jill_water_per_trip * wp.jill_trips_ratio
  let cycles := wp.tank_capacity / water_per_cycle
  cycles * wp.jill_trips_ratio

/-- Theorem stating that Jill makes 30 trips to fill the tank under the given conditions --/
theorem jill_makes_30_trips :
  let wp : WaterProblem := {
    tank_capacity := 600,
    bucket_capacity := 5,
    jack_buckets_per_trip := 2,
    jill_buckets_per_trip := 1,
    jack_trips_ratio := 3,
    jill_trips_ratio := 2
  }
  jill_trips wp = 30 := by
  sorry


end NUMINAMATH_CALUDE_jill_makes_30_trips_l1824_182457


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1824_182456

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 432)
  (h3 : ∀ rate time, interest_paid = principal * rate * time / 100 → rate = time) :
  ∃ rate : ℝ, rate = 6 ∧ interest_paid = principal * rate * rate / 100 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1824_182456


namespace NUMINAMATH_CALUDE_express_u_in_terms_of_f_and_g_l1824_182492

/-- Given functions u, f, and g satisfying certain conditions, 
    prove that u can be expressed in terms of f and g. -/
theorem express_u_in_terms_of_f_and_g 
  (u f g : ℝ → ℝ)
  (h1 : ∀ x, u (x + 1) + u (x - 1) = 2 * f x)
  (h2 : ∀ x, u (x + 4) + u (x - 4) = 2 * g x) :
  ∀ x, u x = g (x + 4) - f (x + 7) + f (x + 5) - f (x + 3) + f (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_express_u_in_terms_of_f_and_g_l1824_182492


namespace NUMINAMATH_CALUDE_bonus_calculation_l1824_182416

/-- Represents the wages of a worker for three months -/
structure Wages where
  october : ℝ
  november : ℝ
  december : ℝ

/-- Calculates the bonus based on the given wages -/
def calculate_bonus (w : Wages) : ℝ :=
  0.2 * (w.october + w.november + w.december)

theorem bonus_calculation (w : Wages) 
  (h1 : w.october / w.november = 3/2 / (4/3))
  (h2 : w.november / w.december = 2 / (8/3))
  (h3 : w.december = w.october + 450) :
  calculate_bonus w = 1494 := by
  sorry

#eval calculate_bonus { october := 2430, november := 2160, december := 2880 }

end NUMINAMATH_CALUDE_bonus_calculation_l1824_182416


namespace NUMINAMATH_CALUDE_cube_structure_ratio_l1824_182465

/-- A structure formed by joining unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  central_cube : Bool
  shared_faces : ℕ

/-- Calculate the volume of the cube structure -/
def volume (s : CubeStructure) : ℕ :=
  s.num_cubes

/-- Calculate the surface area of the cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  (s.num_cubes - 1) * 5

/-- The ratio of volume to surface area -/
def volume_to_surface_ratio (s : CubeStructure) : ℚ :=
  (volume s : ℚ) / (surface_area s : ℚ)

/-- Theorem stating the ratio of volume to surface area for the specific cube structure -/
theorem cube_structure_ratio :
  ∃ (s : CubeStructure),
    s.num_cubes = 8 ∧
    s.central_cube = true ∧
    s.shared_faces = 6 ∧
    volume_to_surface_ratio s = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_structure_ratio_l1824_182465


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1824_182480

theorem quadratic_factorization (A B : ℤ) :
  (∀ y : ℝ, 10 * y^2 - 51 * y + 21 = (A * y - 7) * (B * y - 3)) →
  A * B + B = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1824_182480


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l1824_182474

/-- Given a row of 20 cars, with a black car 16th from the right and a white car 11th from the left,
    the number of cars between the black and white cars is 5. -/
theorem cars_between_black_and_white :
  ∀ (total_cars : ℕ) (black_from_right : ℕ) (white_from_left : ℕ),
    total_cars = 20 →
    black_from_right = 16 →
    white_from_left = 11 →
    white_from_left - (total_cars - black_from_right + 1) - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l1824_182474


namespace NUMINAMATH_CALUDE_first_year_interest_rate_l1824_182408

/-- Proves that the first-year interest rate is 4% given the problem conditions --/
theorem first_year_interest_rate (initial_amount : ℝ) (final_amount : ℝ) 
  (second_year_rate : ℝ) (h1 : initial_amount = 5000) 
  (h2 : final_amount = 5460) (h3 : second_year_rate = 0.05) : 
  ∃ (R : ℝ), R = 0.04 ∧ 
  initial_amount * (1 + R) * (1 + second_year_rate) = final_amount :=
sorry

end NUMINAMATH_CALUDE_first_year_interest_rate_l1824_182408


namespace NUMINAMATH_CALUDE_cubic_roots_from_conditions_l1824_182403

theorem cubic_roots_from_conditions (p q r : ℂ) :
  p + q + r = 0 →
  p * q + p * r + q * r = -1 →
  p * q * r = -1 →
  {p, q, r} = {x : ℂ | x^3 - x - 1 = 0} := by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_conditions_l1824_182403


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l1824_182461

/-- Given that Alice takes 25 minutes to clean her room and Bob takes 2/5 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) 
  (h1 : alice_time = 25)
  (h2 : bob_fraction = 2 / 5) :
  bob_fraction * alice_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l1824_182461


namespace NUMINAMATH_CALUDE_wait_time_difference_l1824_182410

/-- Proves that the difference in wait times between swings and slide is 270 seconds -/
theorem wait_time_difference : 
  let kids_on_swings : ℕ := 3
  let kids_on_slide : ℕ := 2 * kids_on_swings
  let swing_wait_time : ℕ := 2 * 60  -- 2 minutes in seconds
  let slide_wait_time : ℕ := 15      -- 15 seconds
  let total_swing_wait : ℕ := kids_on_swings * swing_wait_time
  let total_slide_wait : ℕ := kids_on_slide * slide_wait_time
  total_swing_wait - total_slide_wait = 270 := by
sorry


end NUMINAMATH_CALUDE_wait_time_difference_l1824_182410


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1824_182479

/-- A regular polygon with interior angle sum of 540° has an exterior angle of 72° --/
theorem regular_polygon_exterior_angle (n : ℕ) : 
  (n - 2) * 180 = 540 → 360 / n = 72 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1824_182479


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1824_182427

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 2 * x + (1/2 : ℝ) < 0) ↔
  (∃ x₀ : ℝ, 2 * x₀^2 + 2 * x₀ + (1/2 : ℝ) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1824_182427


namespace NUMINAMATH_CALUDE_min_value_theorem_l1824_182431

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 27) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 96 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1824_182431


namespace NUMINAMATH_CALUDE_city_rental_rate_proof_l1824_182484

/-- The cost per mile for City Rentals -/
def city_rental_rate : ℝ := 0.31

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def safety_rental_rate : ℝ := 0.29

/-- The number of miles at which the costs are equal -/
def equal_miles : ℝ := 150.0

theorem city_rental_rate_proof :
  city_base_cost + equal_miles * city_rental_rate =
  safety_base_cost + equal_miles * safety_rental_rate :=
by sorry

end NUMINAMATH_CALUDE_city_rental_rate_proof_l1824_182484


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1824_182469

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^502 * k = 15^504 - 6^502) ∧ 
  (∀ m : ℕ, m > 502 → ¬(∃ k : ℕ, 2^m * k = 15^504 - 6^502)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1824_182469


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1824_182406

/-- Given a hyperbola C with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0, 
    and eccentricity √10, its asymptotes are y = ±3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 10  -- eccentricity
  (∀ p ∈ C, (p.1^2 / a^2 - p.2^2 / b^2 = 1)) →
  (e^2 = (a^2 + b^2) / a^2) →
  (∃ k : ℝ, k = b / a ∧ 
    (∀ (x y : ℝ), (x, y) ∈ C → (y = k * x ∨ y = -k * x)) ∧
    k = 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1824_182406


namespace NUMINAMATH_CALUDE_fundraiser_proof_l1824_182422

/-- The number of students asked to bring brownies -/
def num_brownie_students : ℕ := 30

/-- The number of brownies each student brings -/
def brownies_per_student : ℕ := 12

/-- The number of students asked to bring cookies -/
def num_cookie_students : ℕ := 20

/-- The number of cookies each student brings -/
def cookies_per_student : ℕ := 24

/-- The number of students asked to bring donuts -/
def num_donut_students : ℕ := 15

/-- The number of donuts each student brings -/
def donuts_per_student : ℕ := 12

/-- The price of each item in dollars -/
def price_per_item : ℕ := 2

/-- The total amount raised in dollars -/
def total_amount_raised : ℕ := 2040

theorem fundraiser_proof :
  num_brownie_students * brownies_per_student * price_per_item +
  num_cookie_students * cookies_per_student * price_per_item +
  num_donut_students * donuts_per_student * price_per_item =
  total_amount_raised :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_proof_l1824_182422


namespace NUMINAMATH_CALUDE_average_half_median_l1824_182418

theorem average_half_median (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = b / 2 → c / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_average_half_median_l1824_182418


namespace NUMINAMATH_CALUDE_difference_of_squares_l1824_182425

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1824_182425


namespace NUMINAMATH_CALUDE_monomial_coefficient_degree_product_l1824_182462

/-- 
Given a monomial of the form $-\frac{3}{4}{x^2}{y^2}$, 
this theorem proves that the product of its coefficient and degree is -3.
-/
theorem monomial_coefficient_degree_product : 
  ∃ (m n : ℚ), (m = -3/4) ∧ (n = 4) ∧ (m * n = -3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_coefficient_degree_product_l1824_182462


namespace NUMINAMATH_CALUDE_largest_integer_l1824_182472

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 210)
  (sum2 : a + b + d = 230)
  (sum3 : a + c + d = 245)
  (sum4 : b + c + d = 260) :
  max a (max b (max c d)) = 105 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_l1824_182472


namespace NUMINAMATH_CALUDE_multiples_of_4_between_80_and_300_l1824_182452

theorem multiples_of_4_between_80_and_300 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n > 80 ∧ n < 300) (Finset.range 300)).card = 54 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_4_between_80_and_300_l1824_182452


namespace NUMINAMATH_CALUDE_cross_product_result_l1824_182412

def a : ℝ × ℝ × ℝ := (4, 3, -7)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result : cross_product a b = (5, -30, -10) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l1824_182412


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l1824_182489

theorem arithmetic_progression_problem (a d : ℝ) : 
  2 * (a - d) * a * (a + d + 7) = 1000 ∧ 
  a^2 = 2 * (a - d) * (a + d + 7) →
  d = 8 ∨ d = -8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l1824_182489


namespace NUMINAMATH_CALUDE_sum_of_rational_coefficients_in_binomial_expansion_l1824_182401

theorem sum_of_rational_coefficients_in_binomial_expansion :
  let n : ℕ := 6
  let expansion := fun (x : ℝ) => (1 + Real.sqrt x) ^ n
  let rational_term_coefficient := fun (r : ℕ) =>
    if r % 2 = 0 ∧ r ≤ n then Nat.choose n r else 0
  (Finset.sum (Finset.range (n + 1)) rational_term_coefficient) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rational_coefficients_in_binomial_expansion_l1824_182401


namespace NUMINAMATH_CALUDE_incorrect_height_correction_l1824_182415

theorem incorrect_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) :
  n = 35 →
  initial_avg = 180 →
  wrong_height = 166 →
  actual_avg = 178 →
  (n * initial_avg - wrong_height + (n * actual_avg - n * initial_avg + wrong_height)) / n = 236 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_height_correction_l1824_182415


namespace NUMINAMATH_CALUDE_multiply_after_subtract_l1824_182441

theorem multiply_after_subtract (n : ℝ) (x : ℝ) : n = 12 → 4 * n - 3 = (n - 7) * x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_after_subtract_l1824_182441


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1824_182433

/-- Given a quadratic function with vertex (5, 12) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 12 - a * (x - 5)^2) →  -- vertex form
  a * 1^2 + b * 1 + c = 0 →                          -- x-intercept at (1, 0)
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9 :=    -- other x-intercept at 9
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l1824_182433


namespace NUMINAMATH_CALUDE_jennifer_fruits_left_l1824_182468

/-- Calculates the number of fruits Jennifer has left after giving some to her sister. -/
def fruits_left (initial_pears initial_oranges : ℕ) (apples_multiplier : ℕ) (given_away : ℕ) : ℕ :=
  let initial_apples := initial_pears * apples_multiplier
  let remaining_pears := initial_pears - given_away
  let remaining_oranges := initial_oranges - given_away
  let remaining_apples := initial_apples - given_away
  remaining_pears + remaining_oranges + remaining_apples

/-- Theorem stating that Jennifer has 44 fruits left after giving some to her sister. -/
theorem jennifer_fruits_left : 
  fruits_left 10 20 2 2 = 44 := by sorry

end NUMINAMATH_CALUDE_jennifer_fruits_left_l1824_182468


namespace NUMINAMATH_CALUDE_cone_radius_l1824_182497

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of the base is 3 cm. -/
theorem cone_radius (l : ℝ) (A : ℝ) (r : ℝ) : 
  l = 5 → A = 15 * Real.pi → A = Real.pi * r * l → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_radius_l1824_182497


namespace NUMINAMATH_CALUDE_system_solution_l1824_182440

theorem system_solution : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1824_182440


namespace NUMINAMATH_CALUDE_work_completion_time_l1824_182400

/-- The number of days B needs to complete the entire work alone -/
def B_total_days : ℝ := 14.999999999999996

/-- The number of days A works before leaving -/
def A_partial_days : ℝ := 5

/-- The number of days B needs to complete the remaining work after A leaves -/
def B_remaining_days : ℝ := 10

/-- The number of days A needs to complete the entire work alone -/
def A_total_days : ℝ := 15

theorem work_completion_time :
  B_total_days = 14.999999999999996 →
  A_partial_days = 5 →
  B_remaining_days = 10 →
  A_total_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1824_182400


namespace NUMINAMATH_CALUDE_girls_combined_score_is_87_l1824_182481

-- Define the schools
structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

-- Define the problem parameters
def cedar : School := { boys_score := 68, girls_score := 80, combined_score := 73 }
def drake : School := { boys_score := 75, girls_score := 88, combined_score := 83 }
def combined_boys_score : ℝ := 74

-- Theorem statement
theorem girls_combined_score_is_87 :
  ∃ (cedar_boys cedar_girls drake_boys drake_girls : ℕ),
    (cedar_boys : ℝ) * cedar.boys_score + (cedar_girls : ℝ) * cedar.girls_score = 
      (cedar_boys + cedar_girls : ℝ) * cedar.combined_score ∧
    (drake_boys : ℝ) * drake.boys_score + (drake_girls : ℝ) * drake.girls_score = 
      (drake_boys + drake_girls : ℝ) * drake.combined_score ∧
    ((cedar_boys : ℝ) * cedar.boys_score + (drake_boys : ℝ) * drake.boys_score) / 
      (cedar_boys + drake_boys : ℝ) = combined_boys_score ∧
    ((cedar_girls : ℝ) * cedar.girls_score + (drake_girls : ℝ) * drake.girls_score) / 
      (cedar_girls + drake_girls : ℝ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_girls_combined_score_is_87_l1824_182481


namespace NUMINAMATH_CALUDE_imaginary_number_properties_l1824_182421

/-- An imaginary number is a complex number with a non-zero imaginary part -/
def IsImaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_number_properties (x y : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk x y) (h2 : IsImaginary z) : 
  x ∈ Set.univ ∧ y ≠ 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_number_properties_l1824_182421


namespace NUMINAMATH_CALUDE_larger_number_problem_l1824_182476

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 1311)
  (h2 : L = 11 * S + 11) : 
  L = 1441 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1824_182476


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1824_182463

/-- A linear function y = (2k-1)x + k does not pass through the third quadrant
    if and only if 0 ≤ k < 1/2 -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 1)*x + k → ¬(x < 0 ∧ y < 0)) ↔ (0 ≤ k ∧ k < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1824_182463


namespace NUMINAMATH_CALUDE_melanies_turnips_l1824_182494

/-- The number of turnips Benny grew -/
def bennys_turnips : ℕ := 113

/-- The total number of turnips grown by Melanie and Benny -/
def total_turnips : ℕ := 252

/-- Melanie's turnips are equal to the total minus Benny's -/
theorem melanies_turnips : ℕ := total_turnips - bennys_turnips

#check melanies_turnips

end NUMINAMATH_CALUDE_melanies_turnips_l1824_182494


namespace NUMINAMATH_CALUDE_corporation_full_time_employees_l1824_182411

/-- Given a corporation with part-time and full-time employees, 
    we calculate the number of full-time employees. -/
theorem corporation_full_time_employees 
  (total_employees : ℕ) 
  (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : part_time_employees = 2041) : 
  total_employees - part_time_employees = 63093 := by
  sorry

end NUMINAMATH_CALUDE_corporation_full_time_employees_l1824_182411


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1824_182442

/-- The rate of interest given specific investment conditions -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (simple_interest : ℝ) (compound_interest : ℝ) 
  (h1 : principal = 4000)
  (h2 : time = 2)
  (h3 : simple_interest = 400)
  (h4 : compound_interest = 410) :
  ∃ (rate : ℝ), 
    rate = 5 ∧
    simple_interest = (principal * rate * time) / 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ time - 1) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1824_182442


namespace NUMINAMATH_CALUDE_largest_angle_is_E_l1824_182495

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  /-- Angle A is 100 degrees -/
  angle_A : ℝ
  angle_A_eq : angle_A = 100

  /-- Angle B is 120 degrees -/
  angle_B : ℝ
  angle_B_eq : angle_B = 120

  /-- Angles C and D are equal -/
  angle_C : ℝ
  angle_D : ℝ
  angle_C_eq_D : angle_C = angle_D

  /-- Angle E is 30 degrees more than the average of angles C, D, and F -/
  angle_E : ℝ
  angle_F : ℝ
  angle_E_eq : angle_E = (angle_C + angle_D + angle_F) / 3 + 30

  /-- The sum of all angles in a hexagon is 720 degrees -/
  sum_of_angles : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F = 720

/-- Theorem: The largest angle in the hexagon is 147.5 degrees -/
theorem largest_angle_is_E (h : Hexagon) : h.angle_E = 147.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_is_E_l1824_182495


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l1824_182451

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2) ^ 6 - n| ≤ |((3 : ℝ) + Real.sqrt 2) ^ 6 - m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l1824_182451


namespace NUMINAMATH_CALUDE_tan_alpha_third_quadrant_l1824_182417

theorem tan_alpha_third_quadrant (α : Real) 
  (h1 : Real.sin (Real.pi + α) = 3/5)
  (h2 : π < α ∧ α < 3*π/2) : 
  Real.tan α = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_third_quadrant_l1824_182417


namespace NUMINAMATH_CALUDE_complement_M_inter_N_eq_one_two_l1824_182482

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {1, 2, 5}

theorem complement_M_inter_N_eq_one_two :
  (U \ M) ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_eq_one_two_l1824_182482


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_point_symmetric_line_wrt_line_l1824_182454

-- Define the original line l
def l (x y : ℝ) : Prop := y = 2 * x + 1

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line to be reflected
def line_to_reflect (x y : ℝ) : Prop := x - y - 2 = 0

-- Statement for the first part of the problem
theorem symmetric_line_wrt_point :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), l x' y' → 
      (x + x') / 2 = M.1 ∧ (y + y') / 2 = M.2) →
    y = a * x + b ↔ y = 2 * x - 9 :=
sorry

-- Statement for the second part of the problem
theorem symmetric_line_wrt_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    (∀ (x' y' : ℝ), line_to_reflect x' y' → 
      ∃ (x'' y'' : ℝ), l ((x + x'') / 2) ((y + y'') / 2) ∧
      (y'' - y) / (x'' - x) = -1 / (2 : ℝ)) →
    a * x + b * y + c = 0 ↔ 7 * x - y + 16 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_point_symmetric_line_wrt_line_l1824_182454


namespace NUMINAMATH_CALUDE_min_value_theorem_l1824_182409

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  (2/x + 1/y) ≥ 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 4 ∧ 2/x + 1/y = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1824_182409


namespace NUMINAMATH_CALUDE_slope_of_line_with_60_degree_inclination_l1824_182438

theorem slope_of_line_with_60_degree_inclination :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_with_60_degree_inclination_l1824_182438


namespace NUMINAMATH_CALUDE_pizza_toppings_l1824_182458

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l1824_182458


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1824_182448

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1824_182448


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_5_l1824_182443

theorem smallest_five_digit_mod_9_5 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit integer
  n % 9 = 5 ∧                 -- equivalent to 5 mod 9
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m) ∧ 
  n = 10004 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_5_l1824_182443


namespace NUMINAMATH_CALUDE_abs_x_minus_one_plus_three_minimum_l1824_182475

theorem abs_x_minus_one_plus_three_minimum (x : ℝ) :
  |x - 1| + 3 ≥ 3 ∧ (|x - 1| + 3 = 3 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_plus_three_minimum_l1824_182475


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1824_182432

/-- Two hyperbolas have the same asymptotes if and only if M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) ↔ M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l1824_182432


namespace NUMINAMATH_CALUDE_cubic_point_tangent_l1824_182459

theorem cubic_point_tangent (a : ℝ) (h : a^3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_point_tangent_l1824_182459


namespace NUMINAMATH_CALUDE_opposite_of_abs_negative_2023_l1824_182436

theorem opposite_of_abs_negative_2023 : -(|-2023|) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_negative_2023_l1824_182436


namespace NUMINAMATH_CALUDE_output_increase_percentage_l1824_182402

/-- Represents the increase in output per hour when production increases by 80% and working hours decrease by 10% --/
theorem output_increase_percentage (B : ℝ) (H : ℝ) (B_pos : B > 0) (H_pos : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  (new_output - original_output) / original_output = 1 := by
sorry

end NUMINAMATH_CALUDE_output_increase_percentage_l1824_182402


namespace NUMINAMATH_CALUDE_ellipse_condition_l1824_182466

/-- If the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2 ↔
      (x^2 / a^2) + ((y + 1)^2 / b^2) = 1) →
  m > 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1824_182466


namespace NUMINAMATH_CALUDE_multiply_algebraic_expression_l1824_182450

theorem multiply_algebraic_expression (a b : ℝ) : -3 * a * b * (2 * a) = -6 * a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_algebraic_expression_l1824_182450


namespace NUMINAMATH_CALUDE_permutations_count_l1824_182449

def word_length : ℕ := 12
def repeated_letter_count : ℕ := 2

theorem permutations_count :
  (word_length.factorial / repeated_letter_count.factorial) = 239500800 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l1824_182449


namespace NUMINAMATH_CALUDE_amy_money_calculation_l1824_182493

theorem amy_money_calculation (initial : ℕ) (chores : ℕ) (birthday : ℕ) : 
  initial = 2 → chores = 13 → birthday = 3 → initial + chores + birthday = 18 := by
  sorry

end NUMINAMATH_CALUDE_amy_money_calculation_l1824_182493


namespace NUMINAMATH_CALUDE_cosine_power_expansion_sum_of_squares_l1824_182414

open Real

theorem cosine_power_expansion_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (cos θ)^7 = b₁ * cos θ + b₂ * cos (2*θ) + b₃ * cos (3*θ) + 
                          b₄ * cos (4*θ) + b₅ * cos (5*θ) + b₆ * cos (6*θ) + 
                          b₇ * cos (7*θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_expansion_sum_of_squares_l1824_182414


namespace NUMINAMATH_CALUDE_hcl_moles_in_reaction_l1824_182483

-- Define the reaction components
structure ReactionComponent where
  name : String
  moles : ℚ

-- Define the reaction
def reaction (hcl koh kcl h2o : ReactionComponent) : Prop :=
  hcl.name = "HCl" ∧ koh.name = "KOH" ∧ kcl.name = "KCl" ∧ h2o.name = "H2O" ∧
  hcl.moles = koh.moles ∧ hcl.moles = kcl.moles ∧ hcl.moles = h2o.moles

-- Theorem statement
theorem hcl_moles_in_reaction 
  (hcl koh kcl h2o : ReactionComponent)
  (h1 : reaction hcl koh kcl h2o)
  (h2 : koh.moles = 1)
  (h3 : kcl.moles = 1) :
  hcl.moles = 1 := by
  sorry


end NUMINAMATH_CALUDE_hcl_moles_in_reaction_l1824_182483


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l1824_182488

theorem consecutive_odd_squares_difference (x : ℤ) : 
  Odd x → Odd (x + 2) → (x + 2)^2 - x^2 = 2000 → (x = 499 ∨ x = -501) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l1824_182488


namespace NUMINAMATH_CALUDE_categorical_variables_l1824_182445

-- Define the variables
def Smoking : Type := String
def Gender : Type := String
def ReligiousBelief : Type := String
def Nationality : Type := String

-- Define what it means for a variable to be categorical
def IsCategorical (α : Type) : Prop := ∃ (categories : Set α), Finite categories ∧ (∀ x : α, x ∈ categories)

-- State the theorem
theorem categorical_variables :
  IsCategorical Gender ∧ IsCategorical ReligiousBelief ∧ IsCategorical Nationality :=
sorry

end NUMINAMATH_CALUDE_categorical_variables_l1824_182445


namespace NUMINAMATH_CALUDE_exists_phi_and_x0_for_sin_product_equals_one_l1824_182499

theorem exists_phi_and_x0_for_sin_product_equals_one : 
  ∃ (φ : ℝ) (x₀ : ℝ), Real.sin x₀ * Real.sin (x₀ + φ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_phi_and_x0_for_sin_product_equals_one_l1824_182499


namespace NUMINAMATH_CALUDE_line_through_interior_point_no_intersection_l1824_182426

/-- Theorem: A line through a point inside a parabola has no intersection with the parabola --/
theorem line_through_interior_point_no_intersection 
  (x y_o : ℝ) (h : y_o^2 < 4*x) : 
  ∀ y : ℝ, (y^2 = 4*((y*y_o)/(2) - x)) → False :=
by sorry

end NUMINAMATH_CALUDE_line_through_interior_point_no_intersection_l1824_182426


namespace NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l1824_182460

theorem remainder_3_1000_mod_7 : 3^1000 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_1000_mod_7_l1824_182460


namespace NUMINAMATH_CALUDE_parallel_vector_proof_l1824_182477

/-- Given a planar vector b parallel to a = (2, 1) with magnitude 2√5, prove b is either (4, 2) or (-4, -2) -/
theorem parallel_vector_proof (b : ℝ × ℝ) : 
  (∃ k : ℝ, b = (2*k, k)) → -- b is parallel to (2, 1)
  (b.1^2 + b.2^2 = 20) →    -- |b| = 2√5
  (b = (4, 2) ∨ b = (-4, -2)) := by
sorry

end NUMINAMATH_CALUDE_parallel_vector_proof_l1824_182477


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1824_182455

theorem sum_of_reciprocals_positive 
  (a b c d : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hd : |d| > 1) 
  (h_sum : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) : 
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_positive_l1824_182455


namespace NUMINAMATH_CALUDE_valid_parameterization_l1824_182473

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a vector parameterization of a line -/
structure VectorParam where
  point : Vector2D
  direction : Vector2D

def isOnLine (v : Vector2D) : Prop :=
  v.y = 3 * v.x + 5

def isParallel (v : Vector2D) : Prop :=
  ∃ k : ℝ, v.x = k * 1 ∧ v.y = k * 3

theorem valid_parameterization (param : VectorParam) :
  (isOnLine param.point ∧ isParallel param.direction) ↔
  ∀ t : ℝ, isOnLine (Vector2D.mk
    (param.point.x + t * param.direction.x)
    (param.point.y + t * param.direction.y)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1824_182473


namespace NUMINAMATH_CALUDE_complex_modulus_one_l1824_182424

theorem complex_modulus_one (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a ∧ a < (n + 1) / (n - 1 : ℝ)) 
  (h3 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l1824_182424


namespace NUMINAMATH_CALUDE_sum_of_squares_l1824_182490

theorem sum_of_squares (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ = 135) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ = 832 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1824_182490


namespace NUMINAMATH_CALUDE_conference_center_distance_l1824_182428

/-- Represents the problem of calculating the distance to the conference center --/
theorem conference_center_distance :
  -- Initial speed
  ∀ (initial_speed : ℝ),
  -- Speed increase
  ∀ (speed_increase : ℝ),
  -- Distance covered in first hour
  ∀ (first_hour_distance : ℝ),
  -- Late arrival time if continued at initial speed
  ∀ (late_arrival_time : ℝ),
  -- Early arrival time with increased speed
  ∀ (early_arrival_time : ℝ),
  -- Conditions from the problem
  initial_speed = 40 →
  speed_increase = 20 →
  first_hour_distance = 40 →
  late_arrival_time = 1.5 →
  early_arrival_time = 1 →
  -- Conclusion: The distance to the conference center is 100 miles
  ∃ (distance : ℝ), distance = 100 := by
  sorry


end NUMINAMATH_CALUDE_conference_center_distance_l1824_182428


namespace NUMINAMATH_CALUDE_building_shadow_length_l1824_182444

/-- Given a flagpole and a building under similar lighting conditions, 
    this theorem proves the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18) 
  (h2 : flagpole_shadow = 45) 
  (h3 : building_height = 28) : 
  ∃ (building_shadow : ℝ), building_shadow = 70 ∧ 
  flagpole_height / flagpole_shadow = building_height / building_shadow :=
sorry

end NUMINAMATH_CALUDE_building_shadow_length_l1824_182444


namespace NUMINAMATH_CALUDE_sum_digits_888_base_8_l1824_182485

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base 8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base 8 representation -/
def sumDigits (repr : BaseEightRepresentation) : Nat :=
  sorry

theorem sum_digits_888_base_8 :
  sumDigits (toBaseEight 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_888_base_8_l1824_182485


namespace NUMINAMATH_CALUDE_triangle_segment_proof_l1824_182423

theorem triangle_segment_proof (a b c h x : ℝ) : 
  a = 40 ∧ b = 75 ∧ c = 100 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 70.125 := by
sorry

end NUMINAMATH_CALUDE_triangle_segment_proof_l1824_182423


namespace NUMINAMATH_CALUDE_smallest_n_equality_l1824_182487

def C (n : ℕ) : ℚ := 512 * (1 - (1/4)^n) / (1 - 1/4)

def D (n : ℕ) : ℚ := 3072 * (1 - (1/(-3))^n) / (1 + 1/3)

theorem smallest_n_equality :
  ∃ (n : ℕ), n ≥ 1 ∧ C n = D n ∧ ∀ (m : ℕ), m ≥ 1 ∧ m < n → C m ≠ D m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_equality_l1824_182487


namespace NUMINAMATH_CALUDE_factorization_equality_l1824_182434

theorem factorization_equality (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1824_182434


namespace NUMINAMATH_CALUDE_absolute_difference_of_xy_l1824_182404

theorem absolute_difference_of_xy (x y : ℝ) 
  (h1 : x * y = 6) 
  (h2 : x + y = 7) : 
  |x - y| = 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_xy_l1824_182404


namespace NUMINAMATH_CALUDE_closest_to_100_l1824_182405

def expression : ℝ := (2.1 * (50.2 + 0.08)) - 5

def options : List ℝ := [95, 100, 101, 105]

theorem closest_to_100 : 
  ∀ x ∈ options, |expression - 100| ≤ |expression - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_100_l1824_182405


namespace NUMINAMATH_CALUDE_four_engine_safer_than_two_engine_l1824_182470

-- Define the success rate of an engine
variable (P : ℝ) 

-- Define the probability of successful flight for a 2-engine airplane
def prob_success_2engine (P : ℝ) : ℝ := P^2 + 2*P*(1-P)

-- Define the probability of successful flight for a 4-engine airplane
def prob_success_4engine (P : ℝ) : ℝ := P^4 + 4*P^3*(1-P) + 6*P^2*(1-P)^2

-- Theorem statement
theorem four_engine_safer_than_two_engine :
  ∀ P, 2/3 < P ∧ P < 1 → prob_success_4engine P > prob_success_2engine P :=
sorry

end NUMINAMATH_CALUDE_four_engine_safer_than_two_engine_l1824_182470


namespace NUMINAMATH_CALUDE_matrix_product_zero_l1824_182407

variable {R : Type*} [CommRing R]

def A (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![0, d, -e],
    ![-d, 0, d],
    ![e, -d, 0]]

def B (d e : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![d * d, d * e, d * d],
    ![d * e, e * e, e * d],
    ![d * d, e * d, d * d]]

theorem matrix_product_zero (d e : R) (h1 : d = e) :
  A d e * B d e = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_zero_l1824_182407


namespace NUMINAMATH_CALUDE_probability_two_fives_l1824_182478

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_value : ℕ := 5
def target_count : ℕ := 2

theorem probability_two_fives (num_dice : ℕ) (num_sides : ℕ) (target_value : ℕ) (target_count : ℕ) :
  num_dice = 12 →
  num_sides = 6 →
  target_value = 5 →
  target_count = 2 →
  (Nat.choose num_dice target_count : ℚ) * (1 / num_sides : ℚ)^target_count * ((num_sides - 1) / num_sides : ℚ)^(num_dice - target_count) =
  (66 * 5^10 : ℚ) / 6^12 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_fives_l1824_182478


namespace NUMINAMATH_CALUDE_pencils_per_box_l1824_182453

theorem pencils_per_box (num_boxes : ℕ) (pencils_given : ℕ) (pencils_left : ℕ) :
  num_boxes = 2 ∧ pencils_given = 15 ∧ pencils_left = 9 →
  (pencils_given + pencils_left) / num_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_box_l1824_182453


namespace NUMINAMATH_CALUDE_fruit_purchase_price_l1824_182447

/-- The price of an orange in cents -/
def orange_price : ℕ := 3000

/-- The price of a pear in cents -/
def pear_price : ℕ := 9000

/-- The price of a banana in cents -/
def banana_price : ℕ := pear_price - orange_price

/-- The total cost of an orange and a pear in cents -/
def orange_pear_total : ℕ := orange_price + pear_price

/-- The total cost of 50 oranges and 25 bananas in cents -/
def fifty_orange_twentyfive_banana : ℕ := 50 * orange_price + 25 * banana_price

/-- The number of items purchased -/
def total_items : ℕ := 200 + 400

/-- The discount rate as a rational number -/
def discount_rate : ℚ := 1 / 10

theorem fruit_purchase_price :
  orange_pear_total = 12000 ∧
  fifty_orange_twentyfive_banana % 700 = 0 ∧
  total_items > 300 →
  (200 * banana_price + 400 * orange_price) * (1 - discount_rate) = 2160000 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_price_l1824_182447


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l1824_182420

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 4 is the only positive integer solution to n^2 - 17n + 56 = a(n) -/
theorem unique_solution_is_four :
  ∃! (n : ℕ+), n^2 - 17*n + 56 = digit_product n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l1824_182420


namespace NUMINAMATH_CALUDE_orange_harvest_total_l1824_182429

/-- The number of days the orange harvest lasts -/
def harvest_days : ℕ := 4

/-- The number of sacks harvested per day -/
def sacks_per_day : ℕ := 14

/-- The total number of sacks harvested -/
def total_sacks : ℕ := harvest_days * sacks_per_day

theorem orange_harvest_total :
  total_sacks = 56 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_total_l1824_182429


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1824_182419

theorem arithmetic_mean_of_fractions :
  let a := 8 / 12
  let b := 5 / 6
  let c := 9 / 12
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1824_182419


namespace NUMINAMATH_CALUDE_parallelogram_secant_minimum_sum_l1824_182486

/-- Given a parallelogram ABCD with side lengths a and b, and a secant through
    vertex B intersecting extensions of sides DA and DC at points P and Q
    respectively, the sum of segments PA and CQ is minimized when PA = CQ = √(ab). -/
theorem parallelogram_secant_minimum_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  let f : ℝ → ℝ := λ x => x + (a * b) / x
  ∃ (x : ℝ), x > 0 ∧ f x = Real.sqrt (a * b) + Real.sqrt (a * b) ∧
    ∀ (y : ℝ), y > 0 → f y ≥ f x :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_secant_minimum_sum_l1824_182486


namespace NUMINAMATH_CALUDE_lawrence_county_kids_l1824_182491

theorem lawrence_county_kids (home_percentage : Real) (kids_at_home : ℕ) : 
  home_percentage = 0.607 →
  kids_at_home = 907611 →
  ∃ total_kids : ℕ, total_kids = (kids_at_home : Real) / home_percentage := by
    sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_l1824_182491


namespace NUMINAMATH_CALUDE_star_difference_l1824_182496

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by sorry

end NUMINAMATH_CALUDE_star_difference_l1824_182496


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_point_l1824_182435

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_passes_through_point (a : ℝ) :
  (f_derivative a 1) * (2 - 1) + (f a 1) = 7 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_point_l1824_182435


namespace NUMINAMATH_CALUDE_new_average_weight_l1824_182467

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_avg_weight : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 29 → 
  initial_avg_weight = 28 → 
  new_student_weight = 10 → 
  let total_weight := initial_students * initial_avg_weight + new_student_weight
  let new_total_students := initial_students + 1
  (total_weight / new_total_students : ℝ) = 27.4 := by
sorry

end NUMINAMATH_CALUDE_new_average_weight_l1824_182467
