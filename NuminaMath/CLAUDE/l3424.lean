import Mathlib

namespace NUMINAMATH_CALUDE_power_function_through_point_l3424_342412

/-- A power function is a function of the form f(x) = x^a for some real number a. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_through_point (f : ℝ → ℝ) :
  is_power_function f → f 2 = 16 → f = fun x ↦ x^4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3424_342412


namespace NUMINAMATH_CALUDE_fathers_age_l3424_342422

theorem fathers_age (son_age father_age : ℕ) : 
  father_age = 3 * son_age →
  father_age + 15 = 2 * (son_age + 15) →
  father_age = 45 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l3424_342422


namespace NUMINAMATH_CALUDE_digit_2023_of_7_18_l3424_342461

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/18 is 3 -/
theorem digit_2023_of_7_18 : ∃ (d : ℕ), d = 3 ∧ 
  (∃ (a b : ℕ+) (s : Finset ℕ), 
    (7 : ℚ) / 18 = (a : ℚ) / b ∧ 
    s.card = 2023 ∧ 
    (∀ n ∈ s, (10 ^ n * ((7 : ℚ) / 18) % 1).floor % 10 = d) ∧
    (∀ m < 2023, m ∉ s)) :=
by sorry

end NUMINAMATH_CALUDE_digit_2023_of_7_18_l3424_342461


namespace NUMINAMATH_CALUDE_max_rope_piece_length_l3424_342474

theorem max_rope_piece_length : Nat.gcd 60 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_rope_piece_length_l3424_342474


namespace NUMINAMATH_CALUDE_jessica_journey_length_l3424_342494

/-- Represents Jessica's journey in miles -/
def journey_distance : ℝ → Prop :=
  λ total_distance =>
    ∃ (rough_trail tunnel bridge : ℝ),
      -- The journey consists of three parts
      total_distance = rough_trail + tunnel + bridge ∧
      -- The rough trail is one-quarter of the total distance
      rough_trail = (1/4) * total_distance ∧
      -- The tunnel is 25 miles long
      tunnel = 25 ∧
      -- The bridge is one-fourth of the total distance
      bridge = (1/4) * total_distance

/-- Theorem stating that Jessica's journey is 50 miles long -/
theorem jessica_journey_length :
  journey_distance 50 := by
  sorry

end NUMINAMATH_CALUDE_jessica_journey_length_l3424_342494


namespace NUMINAMATH_CALUDE_soda_price_proof_l3424_342480

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.8

/-- The total price of 72 cans purchased in 24-can cases -/
def total_price : ℝ := 34.56

theorem soda_price_proof : 
  (discounted_price * 72 = total_price) → regular_price = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_proof_l3424_342480


namespace NUMINAMATH_CALUDE_class_average_problem_l3424_342413

theorem class_average_problem (x : ℝ) : 
  let total_students : ℕ := 20
  let group1_students : ℕ := 10
  let group2_students : ℕ := 10
  let group2_average : ℝ := 60
  let class_average : ℝ := 70
  (group1_students : ℝ) * x + (group2_students : ℝ) * group2_average = 
    (total_students : ℝ) * class_average → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l3424_342413


namespace NUMINAMATH_CALUDE_sunlight_is_ray_telephone_line_is_segment_l3424_342414

-- Define the different types of lines
inductive LineType
  | Ray
  | LineSegment
  | StraightLine

-- Define the number of endpoints for each line type
def numberOfEndpoints (lt : LineType) : Nat :=
  match lt with
  | .Ray => 1
  | .LineSegment => 2
  | .StraightLine => 0

-- Define the light emitted by the sun
def sunlight : LineType := LineType.Ray

-- Define the line between telephone poles
def telephoneLine : LineType := LineType.LineSegment

-- Theorem stating that the light emitted by the sun is a ray
theorem sunlight_is_ray : sunlight = LineType.Ray := by sorry

-- Theorem stating that the line between telephone poles is a line segment
theorem telephone_line_is_segment : telephoneLine = LineType.LineSegment := by sorry

end NUMINAMATH_CALUDE_sunlight_is_ray_telephone_line_is_segment_l3424_342414


namespace NUMINAMATH_CALUDE_trapezoid_area_is_787_5_l3424_342429

/-- Represents a trapezoid ABCD with given measurements -/
structure Trapezoid where
  ab : ℝ
  bc : ℝ
  ad : ℝ
  altitude : ℝ
  slant_height : ℝ

/-- Calculates the area of the trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the given trapezoid is 787.5 -/
theorem trapezoid_area_is_787_5 (t : Trapezoid) 
  (h_ab : t.ab = 40)
  (h_bc : t.bc = 30)
  (h_ad : t.ad = 17)
  (h_altitude : t.altitude = 15)
  (h_slant_height : t.slant_height = 34) :
  trapezoid_area t = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_787_5_l3424_342429


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3424_342403

theorem polynomial_factorization (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3424_342403


namespace NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l3424_342488

/-- A natural number is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- Four consecutive natural numbers are all composite. -/
def FourConsecutiveComposites (n : ℕ) : Prop :=
  IsComposite n ∧ IsComposite (n + 1) ∧ IsComposite (n + 2) ∧ IsComposite (n + 3)

/-- The sum of four consecutive natural numbers starting from n. -/
def SumFourConsecutive (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3)

theorem smallest_sum_four_consecutive_composites :
  (∃ n : ℕ, FourConsecutiveComposites n) ∧
  (∀ m : ℕ, FourConsecutiveComposites m → SumFourConsecutive m ≥ 102) ∧
  (∃ k : ℕ, FourConsecutiveComposites k ∧ SumFourConsecutive k = 102) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_four_consecutive_composites_l3424_342488


namespace NUMINAMATH_CALUDE_fruit_merchant_problem_l3424_342423

/-- Fruit merchant problem -/
theorem fruit_merchant_problem 
  (total_cost : ℝ) 
  (quantity : ℝ) 
  (cost_difference : ℝ) 
  (large_selling_price : ℝ) 
  (small_selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (earnings_percentage : ℝ) 
  (h1 : total_cost = 8000) 
  (h2 : quantity = 200) 
  (h3 : cost_difference = 20) 
  (h4 : large_selling_price = 40) 
  (h5 : small_selling_price = 16) 
  (h6 : loss_percentage = 0.2) 
  (h7 : earnings_percentage = 0.9) :
  ∃ (small_cost large_cost earnings min_large_price : ℝ),
    small_cost = 10 ∧ 
    large_cost = 30 ∧ 
    earnings = 3200 ∧ 
    min_large_price = 41.6 ∧
    quantity * small_cost + quantity * large_cost = total_cost ∧
    large_cost = small_cost + cost_difference ∧
    earnings = quantity * (large_selling_price - large_cost) + quantity * (small_selling_price - small_cost) ∧
    quantity * min_large_price + small_selling_price * quantity * (1 - loss_percentage) - total_cost ≥ earnings * earnings_percentage :=
by sorry

end NUMINAMATH_CALUDE_fruit_merchant_problem_l3424_342423


namespace NUMINAMATH_CALUDE_train_speed_l3424_342438

/-- The speed of a train given the time to cross an electric pole and a platform -/
theorem train_speed (pole_time : ℝ) (platform_length : ℝ) (platform_time : ℝ) :
  pole_time = 12 →
  platform_length = 320 →
  platform_time = 44 →
  ∃ (train_length : ℝ) (speed_mps : ℝ),
    train_length = speed_mps * pole_time ∧
    train_length + platform_length = speed_mps * platform_time ∧
    speed_mps * 3.6 = 36 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3424_342438


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3424_342419

def hop_distance (n : ℕ) (remaining : ℚ) : ℚ :=
  if n % 2 = 1 then remaining / 2 else remaining / 4

def total_distance (hops : ℕ) : ℚ :=
  let rec aux (n : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if n = 0 then acc
    else
      let dist := hop_distance n remaining
      aux (n - 1) (remaining - dist) (acc + dist)
  aux hops 2 0

theorem kevin_kangaroo_hops :
  total_distance 6 = 485 / 256 := by
  sorry

#eval total_distance 6

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l3424_342419


namespace NUMINAMATH_CALUDE_tangent_condition_l3424_342405

/-- A line with equation kx - y - 3√2 = 0 is tangent to the circle x² + y² = 9 -/
def is_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), k*x - y - 3*Real.sqrt 2 = 0 ∧ x^2 + y^2 = 9 ∧
  ∀ (x' y' : ℝ), k*x' - y' - 3*Real.sqrt 2 = 0 → x'^2 + y'^2 ≥ 9

/-- k = 1 is a sufficient but not necessary condition for the line to be tangent -/
theorem tangent_condition : 
  (is_tangent 1) ∧ (∃ (k : ℝ), k ≠ 1 ∧ is_tangent k) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l3424_342405


namespace NUMINAMATH_CALUDE_a_spends_95_percent_l3424_342498

/-- Represents the salaries and spending percentages of two individuals A and B -/
structure SalaryData where
  total_salary : ℝ
  a_salary : ℝ
  b_spend_percent : ℝ
  a_spend_percent : ℝ

/-- Calculates the savings of an individual given their salary and spending percentage -/
def savings (salary : ℝ) (spend_percent : ℝ) : ℝ :=
  salary * (1 - spend_percent)

/-- Theorem stating that under given conditions, A spends 95% of their salary -/
theorem a_spends_95_percent (data : SalaryData) 
  (h1 : data.total_salary = 3000)
  (h2 : data.a_salary = 2250)
  (h3 : data.b_spend_percent = 0.85)
  (h4 : savings data.a_salary data.a_spend_percent = 
        savings (data.total_salary - data.a_salary) data.b_spend_percent) :
  data.a_spend_percent = 0.95 := by
  sorry


end NUMINAMATH_CALUDE_a_spends_95_percent_l3424_342498


namespace NUMINAMATH_CALUDE_square_equals_eight_times_reciprocal_l3424_342499

theorem square_equals_eight_times_reciprocal (x : ℝ) : 
  x > 0 → x^2 = 8 * (1/x) → x = 2 := by sorry

end NUMINAMATH_CALUDE_square_equals_eight_times_reciprocal_l3424_342499


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l3424_342410

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  (z = 3) →
  (5 * y = 20) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l3424_342410


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3424_342479

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ P : ℝ × ℝ, P.1 = -c ∧ (P.2 = b^2 / a ∨ P.2 = -b^2 / a)) →
  (Real.arctan ((2 * c) / (b^2 / a)) = π / 3) →
  c / a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3424_342479


namespace NUMINAMATH_CALUDE_fourth_power_sum_l3424_342416

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l3424_342416


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3424_342432

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3424_342432


namespace NUMINAMATH_CALUDE_georgia_carnation_problem_l3424_342401

/-- The number of teachers Georgia sent a dozen carnations to -/
def num_teachers : ℕ := 4

/-- The cost of a single carnation in cents -/
def single_carnation_cost : ℕ := 50

/-- The cost of a dozen carnations in cents -/
def dozen_carnation_cost : ℕ := 400

/-- The number of Georgia's friends -/
def num_friends : ℕ := 14

/-- The total amount Georgia spent in cents -/
def total_spent : ℕ := 2500

theorem georgia_carnation_problem :
  num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost ≤ total_spent ∧
  (num_teachers + 1) * dozen_carnation_cost + num_friends * single_carnation_cost > total_spent :=
by sorry

end NUMINAMATH_CALUDE_georgia_carnation_problem_l3424_342401


namespace NUMINAMATH_CALUDE_some_number_value_l3424_342449

theorem some_number_value (n : ℕ) (some_number : ℝ) :
  n = 35 →
  (1/5)^35 * (1/4)^18 = 1/(2*(some_number)^35) →
  some_number = 10 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l3424_342449


namespace NUMINAMATH_CALUDE_mixture_price_calculation_l3424_342435

/-- Calculates the price of a mixture given the prices of two components and their ratio -/
def mixturePricePerKg (pricePeas : ℚ) (priceSoybean : ℚ) (ratioPeas : ℕ) (ratioSoybean : ℕ) : ℚ :=
  let totalParts := ratioPeas + ratioSoybean
  let totalPrice := pricePeas * ratioPeas + priceSoybean * ratioSoybean
  totalPrice / totalParts

theorem mixture_price_calculation (pricePeas priceSoybean : ℚ) (ratioPeas ratioSoybean : ℕ) :
  pricePeas = 16 →
  priceSoybean = 25 →
  ratioPeas = 2 →
  ratioSoybean = 1 →
  mixturePricePerKg pricePeas priceSoybean ratioPeas ratioSoybean = 19 := by
  sorry

end NUMINAMATH_CALUDE_mixture_price_calculation_l3424_342435


namespace NUMINAMATH_CALUDE_inequalities_hold_l3424_342441

theorem inequalities_hold (x y z a b c : ℕ+) 
  (hx : x ≤ a) (hy : y ≤ b) (hz : z ≤ c) : 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2) ∧ 
  (x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3) ∧ 
  (x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b) :=
by sorry

#check inequalities_hold

end NUMINAMATH_CALUDE_inequalities_hold_l3424_342441


namespace NUMINAMATH_CALUDE_hypotenuse_square_l3424_342469

-- Define a right triangle with integer legs
def RightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ b = a + 1

-- Theorem statement
theorem hypotenuse_square (a : ℕ) :
  ∀ b c : ℕ, RightTriangle a b c → c^2 = 2*a^2 + 2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_square_l3424_342469


namespace NUMINAMATH_CALUDE_units_digit_of_product_is_8_l3424_342495

def first_four_composites : List Nat := [4, 6, 8, 9]

theorem units_digit_of_product_is_8 :
  (first_four_composites.prod % 10 = 8) := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_is_8_l3424_342495


namespace NUMINAMATH_CALUDE_reflection_result_l3424_342454

/-- Reflect a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflect a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point C -/
def C : ℝ × ℝ := (-1, 4)

theorem reflection_result :
  (reflect_x ∘ reflect_y) C = (1, -4) := by sorry

end NUMINAMATH_CALUDE_reflection_result_l3424_342454


namespace NUMINAMATH_CALUDE_pond_width_proof_l3424_342404

/-- 
Given a rectangular pond with length 20 meters, depth 8 meters, and volume 1600 cubic meters,
prove that its width is 10 meters.
-/
theorem pond_width_proof (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 20)
    (h2 : depth = 8)
    (h3 : volume = 1600)
    (h4 : volume = length * width * depth) : width = 10 := by
  sorry

end NUMINAMATH_CALUDE_pond_width_proof_l3424_342404


namespace NUMINAMATH_CALUDE_negative_125_to_four_thirds_l3424_342425

theorem negative_125_to_four_thirds : (-125 : ℝ) ^ (4/3) = 625 := by sorry

end NUMINAMATH_CALUDE_negative_125_to_four_thirds_l3424_342425


namespace NUMINAMATH_CALUDE_initial_girls_count_l3424_342451

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 12) = b) →
  (4 * (b - 36) = g - 12) →
  g = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3424_342451


namespace NUMINAMATH_CALUDE_smallest_towel_sets_l3424_342487

def hand_towels_per_set : ℕ := 23
def bath_towels_per_set : ℕ := 29

def total_towels (sets : ℕ) : ℕ :=
  sets * hand_towels_per_set + sets * bath_towels_per_set

theorem smallest_towel_sets :
  ∃ (sets : ℕ),
    (500 ≤ total_towels sets) ∧
    (total_towels sets ≤ 700) ∧
    (∀ (other_sets : ℕ),
      (500 ≤ total_towels other_sets) ∧
      (total_towels other_sets ≤ 700) →
      sets ≤ other_sets) ∧
    sets * hand_towels_per_set = 230 ∧
    sets * bath_towels_per_set = 290 :=
by sorry

end NUMINAMATH_CALUDE_smallest_towel_sets_l3424_342487


namespace NUMINAMATH_CALUDE_sum_of_powers_mod_17_l3424_342471

theorem sum_of_powers_mod_17 : ∃ (a b c d : ℕ), 
  (3 * a) % 17 = 1 ∧ 
  (3 * b) % 17 = 3 ∧ 
  (3 * c) % 17 = 9 ∧ 
  (3 * d) % 17 = 10 ∧ 
  (a + b + c + d) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_mod_17_l3424_342471


namespace NUMINAMATH_CALUDE_binomial_15_12_l3424_342406

theorem binomial_15_12 : Nat.choose 15 12 = 2730 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l3424_342406


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_t_eq_neg_one_l3424_342409

/-- Given a sequence {a_n} with sum of first n terms S_n = 2^n + t,
    prove it's a geometric sequence iff t = -1 -/
theorem geometric_sequence_iff_t_eq_neg_one
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h_S : ∀ n, S n = 2^n + t)
  (h_a : ∀ n, a n = S n - S (n-1)) :
  (∃ r : ℝ, ∀ n > 1, a (n+1) = r * a n) ↔ t = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_t_eq_neg_one_l3424_342409


namespace NUMINAMATH_CALUDE_integer_solution_problem_l3424_342470

theorem integer_solution_problem :
  let S : Set (ℤ × ℤ × ℤ) := {(a, b, c) | a + b + c = 15 ∧ (a - 3)^3 + (b - 5)^3 + (c - 7)^3 = 540}
  S = {(12, 0, 3), (-2, 14, 3), (-1, 0, 16), (-2, 1, 16)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_problem_l3424_342470


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3424_342497

theorem product_sum_theorem (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3424_342497


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3424_342431

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) = Real.sqrt 210 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3424_342431


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l3424_342415

theorem least_sum_of_bases (c d : ℕ+) : 
  (6 * c.val + 5 = 5 * d.val + 6) →
  (∀ c' d' : ℕ+, (6 * c'.val + 5 = 5 * d'.val + 6) → c'.val + d'.val ≥ c.val + d.val) →
  c.val + d.val = 13 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l3424_342415


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3424_342463

theorem cubic_equation_root (a b : ℚ) : 
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 20 = 0 → 
  b = -79 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3424_342463


namespace NUMINAMATH_CALUDE_units_digit_of_M_M8_l3424_342496

-- Define the Lucas-like sequence M_n
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | n + 2 => 2 * M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M8 : unitsDigit (M (M 8)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_M_M8_l3424_342496


namespace NUMINAMATH_CALUDE_joan_remaining_oranges_l3424_342424

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara sold -/
def sara_sold : ℕ := 10

/-- The number of oranges Joan is left with -/
def joan_remaining : ℕ := joan_oranges - sara_sold

theorem joan_remaining_oranges : joan_remaining = 27 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_oranges_l3424_342424


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3424_342483

/-- For a fraction 3x/(5-x) to be meaningful, x must not equal 5 -/
theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 * x / (5 - x)) ↔ x ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3424_342483


namespace NUMINAMATH_CALUDE_inequality_solution_l3424_342440

theorem inequality_solution (x : ℝ) (h1 : x > 0) 
  (h2 : x * Real.sqrt (20 - x) + Real.sqrt (20 * x - x^3) ≥ 20) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3424_342440


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_is_seven_tenths_l3424_342481

/-- The probability of drawing at least one white ball when randomly selecting two balls from a bag containing 3 black balls and 2 white balls. -/
def prob_at_least_one_white : ℚ := 7/10

/-- The total number of balls in the bag. -/
def total_balls : ℕ := 5

/-- The number of black balls in the bag. -/
def black_balls : ℕ := 3

/-- The number of white balls in the bag. -/
def white_balls : ℕ := 2

/-- The theorem stating that the probability of drawing at least one white ball
    when randomly selecting two balls from a bag containing 3 black balls and
    2 white balls is equal to 7/10. -/
theorem prob_at_least_one_white_is_seven_tenths :
  prob_at_least_one_white = 7/10 ∧
  total_balls = black_balls + white_balls ∧
  black_balls = 3 ∧
  white_balls = 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_is_seven_tenths_l3424_342481


namespace NUMINAMATH_CALUDE_johnnys_travel_time_l3424_342458

/-- Proves that given the specified conditions, Johnny's total travel time is 1.6 hours -/
theorem johnnys_travel_time 
  (distance_to_school : ℝ)
  (jogging_speed : ℝ)
  (bus_speed : ℝ)
  (h1 : distance_to_school = 6.461538461538462)
  (h2 : jogging_speed = 5)
  (h3 : bus_speed = 21) :
  distance_to_school / jogging_speed + distance_to_school / bus_speed = 1.6 :=
by sorry


end NUMINAMATH_CALUDE_johnnys_travel_time_l3424_342458


namespace NUMINAMATH_CALUDE_max_profit_is_2180_l3424_342456

/-- Represents the production and profit constraints for products A and B --/
structure ProductionConstraints where
  steel_A : ℝ
  nonferrous_A : ℝ
  steel_B : ℝ
  nonferrous_B : ℝ
  profit_A : ℝ
  profit_B : ℝ
  steel_reserve : ℝ
  nonferrous_reserve : ℝ

/-- Represents the production quantities of products A and B --/
structure ProductionQuantities where
  qty_A : ℝ
  qty_B : ℝ

/-- Calculates the total profit given production quantities and constraints --/
def calculateProfit (q : ProductionQuantities) (c : ProductionConstraints) : ℝ :=
  q.qty_A * c.profit_A + q.qty_B * c.profit_B

/-- Checks if the production quantities satisfy the resource constraints --/
def isValidProduction (q : ProductionQuantities) (c : ProductionConstraints) : Prop :=
  q.qty_A * c.steel_A + q.qty_B * c.steel_B ≤ c.steel_reserve ∧
  q.qty_A * c.nonferrous_A + q.qty_B * c.nonferrous_B ≤ c.nonferrous_reserve ∧
  q.qty_A ≥ 0 ∧ q.qty_B ≥ 0

/-- Theorem stating that the maximum profit under given constraints is 2180 --/
theorem max_profit_is_2180 (c : ProductionConstraints)
    (h1 : c.steel_A = 10 ∧ c.nonferrous_A = 23)
    (h2 : c.steel_B = 70 ∧ c.nonferrous_B = 40)
    (h3 : c.profit_A = 80 ∧ c.profit_B = 100)
    (h4 : c.steel_reserve = 700 ∧ c.nonferrous_reserve = 642) :
    ∃ (q : ProductionQuantities),
      isValidProduction q c ∧
      calculateProfit q c = 2180 ∧
      ∀ (q' : ProductionQuantities),
        isValidProduction q' c → calculateProfit q' c ≤ 2180 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_is_2180_l3424_342456


namespace NUMINAMATH_CALUDE_middle_number_is_nine_l3424_342486

theorem middle_number_is_nine (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 15) (h4 : x + z = 23) (h5 : y + z = 26) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_nine_l3424_342486


namespace NUMINAMATH_CALUDE_multiples_properties_l3424_342402

theorem multiples_properties (x y : ℤ) 
  (hx : ∃ k : ℤ, x = 5 * k) 
  (hy : ∃ m : ℤ, y = 10 * m) : 
  (∃ n : ℤ, y = 5 * n) ∧ 
  (∃ p : ℤ, x - y = 5 * p) ∧ 
  (∃ q : ℤ, y - x = 5 * q) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l3424_342402


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l3424_342417

/-- Given two intersecting circles O and M in a Cartesian coordinate system,
    where O has center (0, 0) and radius r (r > 0),
    and M has center (3, -4) and radius 2,
    the range of possible values for r is 3 < r < 7. -/
theorem circle_intersection_radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x - 3)^2 + (y + 4)^2 = 4) →
  3 < r ∧ r < 7 := by
  sorry

#check circle_intersection_radius_range

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l3424_342417


namespace NUMINAMATH_CALUDE_sales_minimum_value_l3424_342444

/-- A quadratic function f(x) representing monthly sales -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the minimum value of the sales function -/
theorem sales_minimum_value (p q : ℝ) 
  (h1 : f p q 1 = 10) 
  (h2 : f p q 3 = 2) : 
  ∃ x, ∀ y, f p q x ≤ f p q y ∧ f p q x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_sales_minimum_value_l3424_342444


namespace NUMINAMATH_CALUDE_prob_female_math_correct_expected_xi_correct_l3424_342450

-- Define the number of students in each group
def math_male : Nat := 5
def math_female : Nat := 3
def eng_male : Nat := 1
def eng_female : Nat := 3

-- Define the total number of students
def total_students : Nat := math_male + math_female + eng_male + eng_female

-- Define the number of students selected
def selected_students : Nat := 3

-- Define the number of students selected from math group
def math_selected : Nat := 2

-- Define the number of students selected from English group
def eng_selected : Nat := 1

-- Define the probability of selecting at least 1 female from math group
def prob_female_math : ℚ := 9/14

-- Define the expected value of ξ (number of male students selected)
def expected_xi : ℚ := 3/2

-- Theorem for the probability of selecting at least 1 female from math group
theorem prob_female_math_correct :
  (Nat.choose math_female 1 * Nat.choose math_male 1 + Nat.choose math_female 2) / 
  Nat.choose (math_male + math_female) 2 = prob_female_math := by sorry

-- Theorem for the expected value of ξ
theorem expected_xi_correct :
  (0 * (Nat.choose eng_female 1 * Nat.choose math_female 2) / (Nat.choose total_students 3) +
   1 * (Nat.choose eng_female 1 * Nat.choose math_female 1 * Nat.choose math_male 1 + 
        Nat.choose eng_male 1 * Nat.choose math_female 2) / (Nat.choose total_students 3) +
   2 * (Nat.choose eng_male 1 * Nat.choose math_female 1 * Nat.choose math_male 1 + 
        Nat.choose eng_female 1 * Nat.choose math_male 2) / (Nat.choose total_students 3) +
   3 * (Nat.choose eng_male 1 * Nat.choose math_male 2) / (Nat.choose total_students 3)) = expected_xi := by sorry

end NUMINAMATH_CALUDE_prob_female_math_correct_expected_xi_correct_l3424_342450


namespace NUMINAMATH_CALUDE_total_cargo_after_loading_l3424_342490

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem total_cargo_after_loading : initial_cargo + loaded_cargo = 14696 := by
  sorry

end NUMINAMATH_CALUDE_total_cargo_after_loading_l3424_342490


namespace NUMINAMATH_CALUDE_toy_count_l3424_342459

/-- The position of the yellow toy from the left -/
def position_from_left : ℕ := 10

/-- The position of the yellow toy from the right -/
def position_from_right : ℕ := 7

/-- The total number of toys in the row -/
def total_toys : ℕ := position_from_left + position_from_right - 1

theorem toy_count : total_toys = 16 := by
  sorry

end NUMINAMATH_CALUDE_toy_count_l3424_342459


namespace NUMINAMATH_CALUDE_smallest_2016_div_2017_correct_l3424_342464

/-- The smallest natural number that starts with 2016 and is divisible by 2017 -/
def smallest_2016_div_2017 : ℕ := 20162001

/-- A number starts with 2016 if it's greater than or equal to 2016 * 10^4 and less than 2017 * 10^4 -/
def starts_with_2016 (n : ℕ) : Prop :=
  2016 * 10^4 ≤ n ∧ n < 2017 * 10^4

theorem smallest_2016_div_2017_correct :
  starts_with_2016 smallest_2016_div_2017 ∧
  smallest_2016_div_2017 % 2017 = 0 ∧
  ∀ n : ℕ, n < smallest_2016_div_2017 →
    ¬(starts_with_2016 n ∧ n % 2017 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_2016_div_2017_correct_l3424_342464


namespace NUMINAMATH_CALUDE_set_relationship_l3424_342411

theorem set_relationship (A B C : Set α) 
  (h1 : A ∩ B = C) 
  (h2 : B ∩ C = A) : 
  A = C ∧ A ⊆ B := by
sorry

end NUMINAMATH_CALUDE_set_relationship_l3424_342411


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3424_342447

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3424_342447


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3424_342430

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + a/b) ≥ 1 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3424_342430


namespace NUMINAMATH_CALUDE_parking_lot_buses_l3424_342466

/-- Given a parking lot with buses and cars, prove the number of buses -/
theorem parking_lot_buses (total_vehicles : ℕ) (total_wheels : ℕ) : 
  total_vehicles = 40 →
  total_wheels = 210 →
  ∃ (buses cars : ℕ),
    buses + cars = total_vehicles ∧
    6 * buses + 4 * cars = total_wheels ∧
    buses = 25 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_buses_l3424_342466


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficients_sum_l3424_342475

theorem binomial_expansion_coefficients_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 + 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = 48 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficients_sum_l3424_342475


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3424_342428

theorem quadratic_factorization :
  ∀ x : ℝ, 2 * x^2 - 10 * x - 12 = 2 * (x - 6) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3424_342428


namespace NUMINAMATH_CALUDE_divisibility_condition_l3424_342433

theorem divisibility_condition (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3424_342433


namespace NUMINAMATH_CALUDE_solution_difference_l3424_342493

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) 
  (hr : equation r) 
  (hs : equation s) 
  (hdistinct : r ≠ s) 
  (horder : r > s) : 
  r - s = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l3424_342493


namespace NUMINAMATH_CALUDE_value_equivalence_l3424_342472

theorem value_equivalence : 3000 * (3000^3000 + 3000^2999) = 3001 * 3000^3000 := by
  sorry

end NUMINAMATH_CALUDE_value_equivalence_l3424_342472


namespace NUMINAMATH_CALUDE_f_properties_l3424_342468

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- State the theorem
theorem f_properties :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, m = 1 → (f x m ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1)) ∧
  (∀ x : ℝ, x ∈ Set.Icc m (2*m^2) → (1/2 * f x m ≤ |x + 1|)) ↔
  (1/2 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3424_342468


namespace NUMINAMATH_CALUDE_sequence_general_term_l3424_342489

/-- Given a sequence {a_n} with sum of first n terms S_n = (3(3^n + 1)) / 2,
    prove that a_n = 3^n for n ≥ 2 -/
theorem sequence_general_term (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_sum : ∀ k, S k = (3 * (3^k + 1)) / 2) 
    (h_def : ∀ k, k ≥ 2 → a k = S k - S (k-1)) :
  ∀ m, m ≥ 2 → a m = 3^m :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3424_342489


namespace NUMINAMATH_CALUDE_employee_pay_l3424_342421

theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = 880 ∧ x = 1.2 * y → y = 400 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l3424_342421


namespace NUMINAMATH_CALUDE_parts_production_proportion_l3424_342442

/-- The relationship between parts produced per minute and total parts is direct proportion -/
theorem parts_production_proportion (parts_per_minute parts_total : ℝ → ℝ) (t : ℝ) :
  (∀ t, parts_total t = (parts_per_minute t) * t) →
  ∃ k : ℝ, ∀ t, parts_total t = k * (parts_per_minute t) := by
  sorry

end NUMINAMATH_CALUDE_parts_production_proportion_l3424_342442


namespace NUMINAMATH_CALUDE_crosswalk_distance_l3424_342460

/-- Given a parallelogram with one side of length 25 feet, the perpendicular distance
    between this side and its opposite side being 60 feet, and another side of length 70 feet,
    the perpendicular distance between this side and its opposite side is 150/7 feet. -/
theorem crosswalk_distance (side1 side2 height1 height2 : ℝ) : 
  side1 = 25 →
  side2 = 70 →
  height1 = 60 →
  side1 * height1 = side2 * height2 →
  height2 = 150 / 7 := by sorry

end NUMINAMATH_CALUDE_crosswalk_distance_l3424_342460


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3424_342426

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 4}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A ∩ B a = {3} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3424_342426


namespace NUMINAMATH_CALUDE_hyperbola_parameter_value_l3424_342452

/-- Represents a hyperbola with parameter a -/
structure Hyperbola (a : ℝ) :=
  (equation : ∀ (x y : ℝ), x^2 / (a - 3) + y^2 / (1 - a) = 1)

/-- Condition that the foci lie on the x-axis -/
def foci_on_x_axis (h : Hyperbola a) : Prop :=
  a > 1 ∧ a > 3

/-- Condition that the focal distance is 4 -/
def focal_distance_is_4 (h : Hyperbola a) : Prop :=
  ∃ (c : ℝ), c^2 = (a - 3) - (1 - a) ∧ 2 * c = 4

/-- Theorem stating that for a hyperbola with the given conditions, a = 4 -/
theorem hyperbola_parameter_value
  (a : ℝ)
  (h : Hyperbola a)
  (h_foci : foci_on_x_axis h)
  (h_focal : focal_distance_is_4 h) :
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parameter_value_l3424_342452


namespace NUMINAMATH_CALUDE_norris_september_savings_l3424_342485

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := sorry

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris spent on an online game -/
def game_cost : ℕ := 75

/-- The amount of money Norris has left -/
def money_left : ℕ := 10

/-- Theorem stating that Norris saved $29 in September -/
theorem norris_september_savings :
  september_savings = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_norris_september_savings_l3424_342485


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3424_342407

/-- The maximum area of a rectangle inscribed in a circular segment -/
theorem max_area_inscribed_rectangle (r : ℝ) (α : ℝ) (h : 0 < α ∧ α ≤ π / 2) :
  ∃ (T_max : ℝ), T_max = (r^2 / 8) * (-3 * Real.cos α + Real.sqrt (8 + Real.cos α ^ 2)) *
    Real.sqrt (8 - 2 * Real.cos α ^ 2 - 2 * Real.cos α * Real.sqrt (8 + Real.cos α ^ 2)) ∧
  ∀ (T : ℝ), T ≤ T_max := by
  sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l3424_342407


namespace NUMINAMATH_CALUDE_expand_binomials_l3424_342462

theorem expand_binomials (x : ℝ) : (2*x - 3) * (x + 2) = 2*x^2 + x - 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3424_342462


namespace NUMINAMATH_CALUDE_yellow_shirts_count_l3424_342436

theorem yellow_shirts_count (total : ℕ) (blue green red : ℕ) (h1 : total = 36) (h2 : blue = 8) (h3 : green = 11) (h4 : red = 6) :
  total - (blue + green + red) = 11 := by
sorry

end NUMINAMATH_CALUDE_yellow_shirts_count_l3424_342436


namespace NUMINAMATH_CALUDE_equality_abs_condition_l3424_342443

theorem equality_abs_condition (x y : ℝ) : 
  (x = y → abs x = abs y) ∧ 
  ∃ a b : ℝ, abs a = abs b ∧ a ≠ b := by
sorry

end NUMINAMATH_CALUDE_equality_abs_condition_l3424_342443


namespace NUMINAMATH_CALUDE_intersection_point_m_value_l3424_342473

theorem intersection_point_m_value (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = x + 1 ∧ y = -x) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_m_value_l3424_342473


namespace NUMINAMATH_CALUDE_rectangle_perimeter_equal_triangle_area_l3424_342439

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter_equal_triangle_area (a b c w : ℝ) : 
  a = 9 → b = 12 → c = 15 → w = 6 → 
  (1/2) * a * b = w * ((1/2) * a * b / w) → 
  2 * (w + ((1/2) * a * b / w)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_equal_triangle_area_l3424_342439


namespace NUMINAMATH_CALUDE_evaluate_expression_l3424_342467

theorem evaluate_expression (a b : ℝ) (h1 : a = 5) (h2 : b = 6) :
  3 / (2 * a + b) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3424_342467


namespace NUMINAMATH_CALUDE_johns_allowance_spending_l3424_342453

theorem johns_allowance_spending (allowance : ℚ) 
  (h1 : allowance = 9/4)  -- $2.25 as a fraction
  (arcade_fraction : ℚ) (h2 : arcade_fraction = 3/5)
  (toy_fraction : ℚ) (h3 : toy_fraction = 1/3) :
  allowance - arcade_fraction * allowance - toy_fraction * (allowance - arcade_fraction * allowance) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_spending_l3424_342453


namespace NUMINAMATH_CALUDE_floor_abs_sum_l3424_342400

theorem floor_abs_sum (x : ℝ) (h : x = -5.7) : 
  ⌊|x|⌋ + |⌊x⌋| = 11 := by
sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l3424_342400


namespace NUMINAMATH_CALUDE_section_b_average_weight_l3424_342477

/-- Proves that the average weight of section B is 30 kg given the conditions of the problem -/
theorem section_b_average_weight 
  (students_a : ℕ) 
  (students_b : ℕ) 
  (total_students : ℕ) 
  (avg_weight_a : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_a = 36)
  (h2 : students_b = 24)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 30)
  (h5 : avg_weight_total = 30) :
  (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 30 :=
by sorry

end NUMINAMATH_CALUDE_section_b_average_weight_l3424_342477


namespace NUMINAMATH_CALUDE_reflected_arcs_area_l3424_342420

/-- The area of the region bounded by 8 reflected arcs in a circle with an inscribed regular octagon -/
theorem reflected_arcs_area (s : ℝ) (h : s = 1) : 
  let r : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2)
  let arc_area : ℝ := π * (2 + Real.sqrt 2) / 2 - 2 * Real.sqrt 3
  octagon_area - arc_area = 2 * (1 + Real.sqrt 2) - π * (2 + Real.sqrt 2) / 2 + 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_reflected_arcs_area_l3424_342420


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3424_342434

theorem quadratic_form_ratio (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 784*x + 500
  ∃ b c : ℝ, (∀ x, f x = (x + b)^2 + c) ∧ c / b = -391 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3424_342434


namespace NUMINAMATH_CALUDE_regions_on_sphere_l3424_342408

/-- 
Given n great circles on a sphere where no three circles intersect at the same point,
a_n represents the number of regions formed by these circles.
-/
def a_n (n : ℕ) : ℕ := n^2 - n + 2

/-- 
Theorem: The number of regions formed by n great circles on a sphere,
where no three circles intersect at the same point, is equal to n^2 - n + 2.
-/
theorem regions_on_sphere (n : ℕ) : 
  a_n n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_regions_on_sphere_l3424_342408


namespace NUMINAMATH_CALUDE_triangle_circle_perimeter_triangle_circle_perimeter_proof_l3424_342478

/-- The total perimeter of a right triangle with legs 3 and 4, and its inscribed circle -/
theorem triangle_circle_perimeter : ℝ → Prop :=
  fun total_perimeter =>
    ∃ (hypotenuse radius : ℝ),
      -- Triangle properties
      hypotenuse^2 = 3^2 + 4^2 ∧
      -- Circle properties
      radius > 0 ∧
      -- Area of triangle equals semiperimeter times radius
      (3 * 4 / 2 : ℝ) = ((3 + 4 + hypotenuse) / 2) * radius ∧
      -- Total perimeter calculation
      total_perimeter = (3 + 4 + hypotenuse) + 2 * Real.pi * radius ∧
      total_perimeter = 12 + 2 * Real.pi

/-- Proof of the theorem -/
theorem triangle_circle_perimeter_proof : triangle_circle_perimeter (12 + 2 * Real.pi) := by
  sorry

#check triangle_circle_perimeter_proof

end NUMINAMATH_CALUDE_triangle_circle_perimeter_triangle_circle_perimeter_proof_l3424_342478


namespace NUMINAMATH_CALUDE_min_value_of_cubic_l3424_342455

/-- The function f(x) = 2x³ + 3x² - 12x has a minimum value of -7. -/
theorem min_value_of_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 2 * x^3 + 3 * x^2 - 12 * x
  ∃ (min_x : ℝ), f min_x = -7 ∧ ∀ y, f y ≥ f min_x :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_cubic_l3424_342455


namespace NUMINAMATH_CALUDE_jar_capacity_ratio_l3424_342445

theorem jar_capacity_ratio (capacity_x capacity_y : ℝ) : 
  capacity_x > 0 → 
  capacity_y > 0 → 
  (1/2 : ℝ) * capacity_x + (1/2 : ℝ) * capacity_y = (3/4 : ℝ) * capacity_x → 
  capacity_y / capacity_x = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_jar_capacity_ratio_l3424_342445


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l3424_342437

theorem alcohol_mixture_proof (x y z : Real) : 
  x = 112.5 ∧ 
  y = 112.5 ∧ 
  z = 225 ∧
  x + y + z = 450 ∧
  0.10 * x + 0.30 * y + 0.50 * z = 0.35 * 450 :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l3424_342437


namespace NUMINAMATH_CALUDE_rightward_translation_of_point_l3424_342457

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation to the right by a given distance -/
def translateRight (p : Point2D) (distance : ℝ) : Point2D :=
  { x := p.x + distance, y := p.y }

theorem rightward_translation_of_point :
  let initial_point : Point2D := { x := 4, y := -3 }
  let translated_point := translateRight initial_point 1
  translated_point = { x := 5, y := -3 } := by sorry

end NUMINAMATH_CALUDE_rightward_translation_of_point_l3424_342457


namespace NUMINAMATH_CALUDE_inverse_mod_two_million_l3424_342476

/-- The multiplicative inverse of (222222 * 142857) modulo 2,000,000 is 126. -/
theorem inverse_mod_two_million : ∃ N : ℕ, 
  N < 1000000 ∧ (N * (222222 * 142857)) % 2000000 = 1 :=
by
  use 126
  sorry

end NUMINAMATH_CALUDE_inverse_mod_two_million_l3424_342476


namespace NUMINAMATH_CALUDE_equation_solution_l3424_342427

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 →
  (x^3 - 3*x^2)/(x^2 - 4) + 2*x = -16 ↔ 
  x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3424_342427


namespace NUMINAMATH_CALUDE_tangent_curves_l3424_342418

theorem tangent_curves (m : ℝ) : 
  (∃ x y : ℝ, y = x^3 + 2 ∧ y^2 - m*x = 1 ∧ 
   ∀ x' : ℝ, x' ≠ x → (x'^3 + 2)^2 - m*x' ≠ 1) ↔ 
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_curves_l3424_342418


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3424_342448

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3424_342448


namespace NUMINAMATH_CALUDE_hyperbola_C_eccentricity_l3424_342484

/-- Hyperbola C with foci F₁ and F₂, and points P and Q satisfying given conditions -/
structure HyperbolaC where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_C : P.1^2 / a^2 - P.2^2 / b^2 = 1
  h_Q_on_asymptote : Q.2 / Q.1 = b / a
  h_first_quadrant : P.1 > 0 ∧ P.2 > 0 ∧ Q.1 > 0 ∧ Q.2 > 0
  h_QP_eq_PF₂ : (Q.1 - P.1, Q.2 - P.2) = (P.1 - F₂.1, P.2 - F₂.2)
  h_QF₁_perp_QF₂ : (Q.1 - F₁.1) * (Q.1 - F₂.1) + (Q.2 - F₁.2) * (Q.2 - F₂.2) = 0

/-- The eccentricity of hyperbola C is √5 - 1 -/
theorem hyperbola_C_eccentricity (hC : HyperbolaC) : 
  ∃ e : ℝ, e = Real.sqrt 5 - 1 ∧ e^2 = (hC.a^2 + hC.b^2) / hC.a^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_eccentricity_l3424_342484


namespace NUMINAMATH_CALUDE_sequence_shortening_l3424_342446

/-- A sequence of digits where each digit is independently chosen from {0, 9} -/
def DigitSequence := Fin 2015 → Fin 10

/-- The probability of a digit being 0 or 9 -/
def p : ℝ := 0.1

/-- The number of digits in the original sequence -/
def n : ℕ := 2015

/-- The number of digits that can potentially be removed -/
def k : ℕ := 2014

theorem sequence_shortening (seq : DigitSequence) :
  /- The probability of the sequence shortening by exactly one digit -/
  (Nat.choose k 1 : ℝ) * p^1 * (1 - p)^(k - 1) = 
    (2014 : ℝ) * 0.1 * 0.9^2013 ∧
  /- The expected length of the new sequence -/
  (n : ℝ) - (k : ℝ) * p = 1813.6 := by
  sorry


end NUMINAMATH_CALUDE_sequence_shortening_l3424_342446


namespace NUMINAMATH_CALUDE_not_all_observed_values_yield_significant_regression_l3424_342491

/-- A set of observed values -/
structure ObservedValues where
  values : Set (ℝ × ℝ)

/-- A regression line equation -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Definition of representative significance for a regression line -/
def has_representative_significance (ov : ObservedValues) (rl : RegressionLine) : Prop :=
  sorry

/-- The theorem stating that not all sets of observed values yield a regression line with representative significance -/
theorem not_all_observed_values_yield_significant_regression :
  ¬ ∀ (ov : ObservedValues), ∃ (rl : RegressionLine), has_representative_significance ov rl :=
sorry

end NUMINAMATH_CALUDE_not_all_observed_values_yield_significant_regression_l3424_342491


namespace NUMINAMATH_CALUDE_f_zero_is_zero_l3424_342482

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem f_zero_is_zero (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 4) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_is_zero_l3424_342482


namespace NUMINAMATH_CALUDE_quadratic_expansion_sum_l3424_342465

theorem quadratic_expansion_sum (d : ℝ) (h : d ≠ 0) : 
  ∃ (a b c : ℤ), (15 * d^2 + 15 + 7 * d) + (3 * d + 9)^2 = a * d^2 + b * d + c ∧ a + b + c = 181 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expansion_sum_l3424_342465


namespace NUMINAMATH_CALUDE_exponent_simplification_l3424_342492

theorem exponent_simplification :
  3^6 * 6^6 * 3^12 * 6^12 = 18^18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l3424_342492
