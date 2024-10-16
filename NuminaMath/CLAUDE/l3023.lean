import Mathlib

namespace NUMINAMATH_CALUDE_complex_exponential_identity_l3023_302332

theorem complex_exponential_identity (n : ℕ) (hn : n > 0 ∧ n ≤ 500) (t : ℝ) :
  (Complex.exp (Complex.I * t))^n = Complex.exp (Complex.I * (n * t)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_identity_l3023_302332


namespace NUMINAMATH_CALUDE_impossibility_of_all_prime_combinations_l3023_302345

/-- A digit is a natural number from 0 to 9 -/
def Digit := {n : ℕ // n < 10}

/-- A two-digit number formed from two digits -/
def TwoDigitNumber (d1 d2 : Digit) : ℕ := d1.val * 10 + d2.val

/-- Predicate to check if a natural number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem impossibility_of_all_prime_combinations :
  ∀ (d1 d2 d3 d4 : Digit),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 →
    ∃ (i j : Fin 4),
      i ≠ j ∧
      ¬(IsPrime (TwoDigitNumber (([d1, d2, d3, d4].get i) : Digit) (([d1, d2, d3, d4].get j) : Digit))) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_all_prime_combinations_l3023_302345


namespace NUMINAMATH_CALUDE_clothing_discount_l3023_302300

theorem clothing_discount (original_price : ℝ) (first_sale_price second_sale_price : ℝ) :
  first_sale_price = (4 / 5) * original_price →
  second_sale_price = (1 - 0.4) * first_sale_price →
  second_sale_price = (12 / 25) * original_price :=
by sorry

end NUMINAMATH_CALUDE_clothing_discount_l3023_302300


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3023_302338

def point_A : ℝ × ℝ := (-1, -5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  let vector_AB : ℝ × ℝ := (3 * vector_a.1, 3 * vector_a.2)
  let point_B : ℝ × ℝ := (point_A.1 + vector_AB.1, point_A.2 + vector_AB.2)
  point_B = (5, 4) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l3023_302338


namespace NUMINAMATH_CALUDE_max_area_CDFE_l3023_302389

/-- The area of quadrilateral CDFE in a square ABCD with side length 2,
    where E and F are on sides AB and AD respectively, and AE = AF = 2k. -/
def area_CDFE (k : ℝ) : ℝ := 2 * (1 - k)^2

/-- The theorem stating that the area of CDFE is maximized when k = 1/2,
    and the maximum area is 1/2. -/
theorem max_area_CDFE :
  ∀ k : ℝ, 0 < k → k < 1 →
  area_CDFE k ≤ area_CDFE (1/2) ∧ area_CDFE (1/2) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_area_CDFE_l3023_302389


namespace NUMINAMATH_CALUDE_unique_score_theorem_l3023_302339

/-- Represents a score in the mathematics competition. -/
structure Score where
  value : ℕ
  correct : ℕ
  wrong : ℕ
  total_questions : ℕ
  h1 : value = 5 * correct - 2 * wrong
  h2 : correct + wrong ≤ total_questions

/-- The unique score over 70 that allows determination of correct answers. -/
def unique_determinable_score : ℕ := 71

theorem unique_score_theorem (s : Score) (h_total : s.total_questions = 25) 
    (h_over_70 : s.value > 70) : 
  (∃! c w, s.correct = c ∧ s.wrong = w) ↔ s.value = unique_determinable_score :=
sorry

end NUMINAMATH_CALUDE_unique_score_theorem_l3023_302339


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3023_302375

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x > 0 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3023_302375


namespace NUMINAMATH_CALUDE_circular_track_circumference_l3023_302357

/-- Represents a circular track with two moving points. -/
structure CircularTrack where
  /-- The circumference of the track in yards. -/
  circumference : ℝ
  /-- The constant speed of both points (assumed to be the same). -/
  speed : ℝ
  /-- The distance B travels before the first meeting. -/
  first_meeting_distance : ℝ
  /-- The remaining distance A needs to travel after the second meeting to complete a lap. -/
  second_meeting_remaining : ℝ

/-- The theorem stating the conditions and the result to be proved. -/
theorem circular_track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting_distance = 100)
  (h2 : track.second_meeting_remaining = 60)
  (h3 : track.speed > 0) :
  track.circumference = 480 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l3023_302357


namespace NUMINAMATH_CALUDE_pencil_store_theorem_l3023_302359

/-- Represents the store's pencil purchases and sales -/
structure PencilStore where
  first_purchase_cost : ℝ
  first_purchase_quantity : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity : ℝ
  selling_price : ℝ

/-- The conditions of the pencil store problem -/
def pencil_store_conditions (s : PencilStore) : Prop :=
  s.first_purchase_cost * s.first_purchase_quantity = 600 ∧
  s.second_purchase_cost * s.second_purchase_quantity = 600 ∧
  s.second_purchase_cost = (5/4) * s.first_purchase_cost ∧
  s.second_purchase_quantity = s.first_purchase_quantity - 30

/-- The profit calculation for the pencil store -/
def profit (s : PencilStore) : ℝ :=
  s.selling_price * (s.first_purchase_quantity + s.second_purchase_quantity) -
  (s.first_purchase_cost * s.first_purchase_quantity + s.second_purchase_cost * s.second_purchase_quantity)

/-- The main theorem about the pencil store problem -/
theorem pencil_store_theorem (s : PencilStore) :
  pencil_store_conditions s →
  s.first_purchase_cost = 4 ∧
  (∀ p, profit { s with selling_price := p } ≥ 420 → p ≥ 6) :=
by sorry


end NUMINAMATH_CALUDE_pencil_store_theorem_l3023_302359


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3023_302386

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^4 + 8 * X^3 - 35 * X^2 - 45 * X + 52 = 
  (X^2 + 5 * X - 3) * q + (-21 * X + 79) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3023_302386


namespace NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l3023_302318

theorem complex_fraction_equality : Complex → Prop :=
  fun i => (3 : ℂ) / (1 - i)^2 = (3 / 2 : ℂ) * i

-- The proof is omitted
theorem complex_fraction_equality_proof : complex_fraction_equality Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_complex_fraction_equality_proof_l3023_302318


namespace NUMINAMATH_CALUDE_smallest_proportional_part_l3023_302307

theorem smallest_proportional_part (total : ℕ) (parts : List ℕ) : 
  total = 360 → 
  parts = [5, 7, 4, 8] → 
  List.length parts = 4 → 
  (List.sum parts) ∣ total → 
  (List.minimum parts).isSome → 
  (total / (List.sum parts)) * (List.minimum parts).get! = 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_proportional_part_l3023_302307


namespace NUMINAMATH_CALUDE_vector_operation_l3023_302377

/-- Given two vectors a and b in ℝ², prove that 3b - a equals the specified result. -/
theorem vector_operation (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
  (3 : ℝ) • b - a = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3023_302377


namespace NUMINAMATH_CALUDE_digit_sum_theorem_l3023_302305

theorem digit_sum_theorem (f o g : ℕ) : 
  f < 10 → o < 10 → g < 10 →
  4 * (100 * f + 10 * o + g) = 1464 →
  f + o + g = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_theorem_l3023_302305


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3023_302398

theorem base_conversion_problem : ∃! (n : ℕ), ∃ (A C : ℕ), 
  (A < 8 ∧ C < 8) ∧
  (A < 6 ∧ C < 6) ∧
  (n = 8 * A + C) ∧
  (n = 6 * C + A) ∧
  (n = 47) := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3023_302398


namespace NUMINAMATH_CALUDE_min_value_theorem_l3023_302320

/-- The minimum value of a specific function given certain conditions -/
theorem min_value_theorem (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h1 : c * y + b * z = a) 
  (h2 : a * z + c * x = b) 
  (h3 : b * x + a * y = c) : 
  (∃ (m : ℝ), m = (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)) ∧ 
   ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
   x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ m) ∧ 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3023_302320


namespace NUMINAMATH_CALUDE_carters_dog_height_l3023_302394

-- Define heights in inches
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches
def carter_height : ℕ := betty_height + 12
def dog_height : ℕ := carter_height / 2

-- Theorem statement
theorem carters_dog_height : dog_height = 24 := by
  sorry

end NUMINAMATH_CALUDE_carters_dog_height_l3023_302394


namespace NUMINAMATH_CALUDE_product_calculation_l3023_302306

theorem product_calculation : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l3023_302306


namespace NUMINAMATH_CALUDE_gcf_of_30_90_75_l3023_302313

theorem gcf_of_30_90_75 : Nat.gcd 30 (Nat.gcd 90 75) = 15 := by sorry

end NUMINAMATH_CALUDE_gcf_of_30_90_75_l3023_302313


namespace NUMINAMATH_CALUDE_consecutive_digits_difference_l3023_302366

theorem consecutive_digits_difference (a : ℤ) : 
  (100 * (a + 1) + 10 * a + (a - 1)) - (100 * (a - 1) + 10 * a + (a + 1)) = 198 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_digits_difference_l3023_302366


namespace NUMINAMATH_CALUDE_square_area_l3023_302325

/-- Given a square ABCD composed of two identical rectangles and two squares with side lengths 2 cm and 4 cm respectively, prove that the area of ABCD is 36 cm². -/
theorem square_area (s : ℝ) (h1 : s > 0) : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  a + 2 = 4 ∧
  b + 4 = s ∧
  s = 6 ∧
  s^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_l3023_302325


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3023_302378

/-- Given a rectangular field with one side of 30 feet and three sides fenced using 
    a total of 78 feet of fencing, prove that the area of the field is 720 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →  -- Length of uncovered side
  2 * W + L = 78 →  -- Total fencing equation
  L * W = 720 :=  -- Area of the field
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3023_302378


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3023_302365

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 20.7 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l3023_302365


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3023_302326

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  -3*x*y - 3*x^2 + 4*x*y + 2*x^2 = x*y - x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  3*(a^2 - 2*a*b) - 5*(a^2 + 4*a*b) = -2*a^2 - 26*a*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3023_302326


namespace NUMINAMATH_CALUDE_extra_red_pencil_packs_l3023_302349

theorem extra_red_pencil_packs (total_packs : ℕ) (normal_red_per_pack : ℕ) (total_red_pencils : ℕ) :
  total_packs = 15 →
  normal_red_per_pack = 1 →
  total_red_pencils = 21 →
  ∃ (extra_packs : ℕ),
    extra_packs * 2 + total_packs * normal_red_per_pack = total_red_pencils ∧
    extra_packs = 3 :=
by sorry

end NUMINAMATH_CALUDE_extra_red_pencil_packs_l3023_302349


namespace NUMINAMATH_CALUDE_distance_to_focus_l3023_302379

/-- Given a parabola y² = 8x and a point P(4, y) on it, 
    the distance from P to the focus of the parabola is 6. -/
theorem distance_to_focus (y : ℝ) : 
  y^2 = 32 →  -- Point P(4, y) is on the parabola y² = 8x
  let F := (2, 0)  -- Focus of the parabola
  Real.sqrt ((4 - 2)^2 + y^2) = 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3023_302379


namespace NUMINAMATH_CALUDE_total_prime_factors_l3023_302310

def expression (a b c : ℕ) := (4^a) * (7^b) * (11^c)

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem total_prime_factors (a b c : ℕ) :
  a = 11 → b = 7 → c = 2 → is_prime 7 → is_prime 11 →
  (∃ n : ℕ, expression a b c = 2^(2*a) * 7^b * 11^c ∧ 
   n = (2*a) + b + c ∧ n = 31) :=
sorry

end NUMINAMATH_CALUDE_total_prime_factors_l3023_302310


namespace NUMINAMATH_CALUDE_inequality_proofs_l3023_302354

theorem inequality_proofs 
  (a b c d : ℝ) 
  (hab : a > b) 
  (hcd : c > d) 
  (hac2bc2 : a * c^2 < b * c^2) 
  (hab_pos : a > b ∧ b > 0) 
  (hc_pos : c > 0) : 
  (a + c > b + d) ∧ 
  (a < b) ∧ 
  ((b + c) / (a + c) > b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3023_302354


namespace NUMINAMATH_CALUDE_reflect_P_x_axis_l3023_302358

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The original point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- Theorem: Reflecting point P(-2, 4) across the x-axis results in (-2, -4) -/
theorem reflect_P_x_axis :
  reflect_x P = { x := -2, y := -4 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_x_axis_l3023_302358


namespace NUMINAMATH_CALUDE_smallest_positive_number_l3023_302374

theorem smallest_positive_number : 
  let a := 8 - 2 * Real.sqrt 17
  let b := 2 * Real.sqrt 17 - 8
  let c := 25 - 7 * Real.sqrt 5
  let d := 40 - 9 * Real.sqrt 2
  let e := 9 * Real.sqrt 2 - 40
  (0 < b) ∧ 
  (a ≤ b ∨ a ≤ 0) ∧ 
  (b ≤ c ∨ c ≤ 0) ∧ 
  (b ≤ d ∨ d ≤ 0) ∧ 
  (b ≤ e ∨ e ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l3023_302374


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3023_302373

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3023_302373


namespace NUMINAMATH_CALUDE_product_of_roots_l3023_302319

theorem product_of_roots (k m x₁ x₂ : ℝ) (h_distinct : x₁ ≠ x₂)
  (h₁ : 4 * x₁^2 - k * x₁ - m = 0) (h₂ : 4 * x₂^2 - k * x₂ - m = 0) :
  x₁ * x₂ = -m / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3023_302319


namespace NUMINAMATH_CALUDE_cube_edge_from_volume_l3023_302324

theorem cube_edge_from_volume (volume : ℝ) (edge : ℝ) :
  volume = 3375 ∧ volume = edge ^ 3 → edge = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_from_volume_l3023_302324


namespace NUMINAMATH_CALUDE_negative_sum_l3023_302329

theorem negative_sum (a b c : ℝ) 
  (ha : -2 < a ∧ a < -1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 0) : 
  b + c < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3023_302329


namespace NUMINAMATH_CALUDE_field_trip_cost_l3023_302308

/-- Calculate the total cost of renting buses and paying tolls for a field trip -/
theorem field_trip_cost (total_people : ℕ) (seats_per_bus : ℕ) 
  (rental_cost_per_bus : ℕ) (toll_per_bus : ℕ) : 
  total_people = 260 → 
  seats_per_bus = 41 → 
  rental_cost_per_bus = 300000 → 
  toll_per_bus = 7500 → 
  (((total_people + seats_per_bus - 1) / seats_per_bus) * 
   (rental_cost_per_bus + toll_per_bus)) = 2152500 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_l3023_302308


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l3023_302350

/-- Systematic sampling function -/
def systematic_sampling (population : ℕ) (sample_size : ℕ) : 
  (ℕ × ℕ × ℕ) :=
  let remaining := population % sample_size
  let eliminated := remaining
  let segment_size := (population - eliminated) / sample_size
  (eliminated, sample_size, segment_size)

/-- Theorem for the given systematic sampling problem -/
theorem systematic_sampling_problem :
  systematic_sampling 1650 35 = (5, 35, 47) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l3023_302350


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3023_302363

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 30

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 3.75

/-- Theorem stating that the percentage of water in fresh grapes is 90% -/
theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 :=
by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3023_302363


namespace NUMINAMATH_CALUDE_circle_ratio_l3023_302382

/-- For a circle with diameter 100 cm and circumference 314 cm, 
    the ratio of circumference to diameter is 3.14 -/
theorem circle_ratio : 
  ∀ (diameter circumference : ℝ), 
    diameter = 100 → 
    circumference = 314 → 
    circumference / diameter = 3.14 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l3023_302382


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l3023_302335

/-- The smallest possible area of a right triangle with sides 6 and 8 is 24 square units -/
theorem smallest_area_right_triangle (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l3023_302335


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3023_302364

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3023_302364


namespace NUMINAMATH_CALUDE_power_equality_l3023_302381

theorem power_equality (n b : ℝ) : n = 2 ^ (1/4) → n ^ b = 8 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3023_302381


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3023_302304

theorem unique_four_digit_number :
  ∃! (a b c d : ℕ), 
    0 < a ∧ a < 10 ∧
    0 ≤ b ∧ b < 10 ∧
    0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 10 * a + b ∧
    a * b * c * d = 10 * c + d :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3023_302304


namespace NUMINAMATH_CALUDE_max_bar_weight_example_l3023_302328

/-- Calculates the maximum weight that can be put on a weight bench bar given the bench's maximum support weight, safety margin percentage, and the weights of two people using the bench. -/
def maxBarWeight (benchMax : ℝ) (safetyMargin : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  benchMax * (1 - safetyMargin) - (weight1 + weight2)

/-- Theorem stating that for a 1000-pound bench with 20% safety margin and two people weighing 250 and 180 pounds, the maximum weight on the bar is 370 pounds. -/
theorem max_bar_weight_example :
  maxBarWeight 1000 0.2 250 180 = 370 := by
  sorry

end NUMINAMATH_CALUDE_max_bar_weight_example_l3023_302328


namespace NUMINAMATH_CALUDE_unique_modular_equivalent_in_range_l3023_302331

theorem unique_modular_equivalent_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalent_in_range_l3023_302331


namespace NUMINAMATH_CALUDE_square_of_integer_proof_l3023_302387

theorem square_of_integer_proof (n : ℕ+) (h : ∃ (k : ℤ), k^2 = 1 + 12 * (n : ℤ)^2) :
  ∃ (m : ℤ), (2 : ℤ) + 2 * Int.sqrt (1 + 12 * (n : ℤ)^2) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_proof_l3023_302387


namespace NUMINAMATH_CALUDE_original_number_proof_l3023_302361

theorem original_number_proof : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(25 ∣ (y + 19))) ∧ 
  (25 ∣ (x + 19)) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3023_302361


namespace NUMINAMATH_CALUDE_power_inequality_l3023_302360

theorem power_inequality (a x y : ℝ) (ha : 0 < a) (ha1 : a < 1) (h : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3023_302360


namespace NUMINAMATH_CALUDE_bouquets_calculation_l3023_302316

/-- Given the initial number of flowers, flowers per bouquet, and wilted flowers,
    calculates the number of bouquets that can be made. -/
def calculate_bouquets (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Proves that given 88 initial flowers, 5 flowers per bouquet, and 48 wilted flowers,
    the number of bouquets that can be made is equal to 8. -/
theorem bouquets_calculation :
  calculate_bouquets 88 5 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bouquets_calculation_l3023_302316


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l3023_302336

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, 
    then the x-coordinate of c is -15. -/
theorem parallel_vectors_x_coordinate 
  (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (2, -1)) 
  (hc : c.2 = 3) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2*b.1, a.2 + 2*b.2) = (k * c.1, k * c.2)) : 
  c.1 = -15 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l3023_302336


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3023_302346

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3023_302346


namespace NUMINAMATH_CALUDE_M_equals_P_l3023_302356

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def P : Set ℝ := {a | ∃ b, a = b^2 - 1}

-- Theorem statement
theorem M_equals_P : M = P := by
  sorry

end NUMINAMATH_CALUDE_M_equals_P_l3023_302356


namespace NUMINAMATH_CALUDE_initial_average_marks_l3023_302311

/-- 
Given a class of students with an incorrect average mark, prove that the initial average 
before correcting an error in one student's mark is equal to a specific value.
-/
theorem initial_average_marks 
  (n : ℕ) -- number of students
  (wrong_mark correct_mark : ℕ) -- the wrong and correct marks for one student
  (final_average : ℚ) -- the correct average after fixing the error
  (h1 : n = 25) -- there are 25 students
  (h2 : wrong_mark = 60) -- the wrong mark was 60
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : final_average = 98) -- the final correct average is 98
  : ∃ (initial_average : ℚ), initial_average = 100 ∧ 
    n * initial_average - (wrong_mark - correct_mark) = n * final_average :=
by sorry

end NUMINAMATH_CALUDE_initial_average_marks_l3023_302311


namespace NUMINAMATH_CALUDE_bread_distribution_l3023_302362

theorem bread_distribution (a : ℚ) (d : ℚ) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 100) →
  ((a + 3*d) + (a + 4*d) + (a + 2*d))/7 = a + (a + d) →
  a = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_l3023_302362


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3023_302340

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 180) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3023_302340


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3023_302309

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3023_302309


namespace NUMINAMATH_CALUDE_bigger_number_problem_l3023_302392

theorem bigger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 77)
  (ratio_eq : 5 * x = 6 * y)
  (x_geq_y : x ≥ y) : 
  x = 42 := by
  sorry

end NUMINAMATH_CALUDE_bigger_number_problem_l3023_302392


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l3023_302334

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 16 ↔ (3^k : ℕ) ∣ p := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l3023_302334


namespace NUMINAMATH_CALUDE_savings_calculation_l3023_302352

theorem savings_calculation (income expenditure : ℕ) 
  (h1 : income = 36000)
  (h2 : income * 8 = expenditure * 9) : 
  income - expenditure = 4000 :=
sorry

end NUMINAMATH_CALUDE_savings_calculation_l3023_302352


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3023_302343

theorem quadratic_inequality_equivalence (x : ℝ) :
  -x^2 - 2*x + 3 ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l3023_302343


namespace NUMINAMATH_CALUDE_min_sum_is_negative_442_l3023_302315

/-- An arithmetic progression with sum S_n for the first n terms. -/
structure ArithmeticProgression where
  S : ℕ → ℤ
  sum_3 : S 3 = -141
  sum_35 : S 35 = 35

/-- The minimum value of S_n for an arithmetic progression satisfying the given conditions. -/
def min_sum (ap : ArithmeticProgression) : ℤ :=
  sorry

/-- Theorem stating that the minimum value of S_n is -442. -/
theorem min_sum_is_negative_442 (ap : ArithmeticProgression) :
  min_sum ap = -442 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_is_negative_442_l3023_302315


namespace NUMINAMATH_CALUDE_enclosed_area_is_four_thirds_l3023_302342

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2

-- Define the line function
def line : ℝ → ℝ := Function.const ℝ 1

-- Define the difference between the line and the parabola
def difference (x : ℝ) : ℝ := line x - parabola x

-- Theorem statement
theorem enclosed_area_is_four_thirds :
  (∫ x in (-1)..(1), difference x) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_is_four_thirds_l3023_302342


namespace NUMINAMATH_CALUDE_tax_calculation_l3023_302314

/-- Given gross pay and net pay, calculates the tax amount -/
def calculate_tax (gross_pay : ℝ) (net_pay : ℝ) : ℝ :=
  gross_pay - net_pay

theorem tax_calculation :
  let gross_pay : ℝ := 450
  let net_pay : ℝ := 315
  calculate_tax gross_pay net_pay = 135 := by
sorry

end NUMINAMATH_CALUDE_tax_calculation_l3023_302314


namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l3023_302372

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ c ≥ a ∧ c ≥ b

theorem pythagorean_triple_6_8_10 : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l3023_302372


namespace NUMINAMATH_CALUDE_second_pipe_filling_time_l3023_302312

/-- Given a pool that can be filled by one pipe in 10 hours and by both pipes in 3.75 hours,
    prove that the second pipe alone takes 6 hours to fill the pool. -/
theorem second_pipe_filling_time
  (time_pipe1 : ℝ) (time_both : ℝ) (time_pipe2 : ℝ)
  (h1 : time_pipe1 = 10)
  (h2 : time_both = 3.75)
  (h3 : 1 / time_pipe1 + 1 / time_pipe2 = 1 / time_both) :
  time_pipe2 = 6 :=
sorry

end NUMINAMATH_CALUDE_second_pipe_filling_time_l3023_302312


namespace NUMINAMATH_CALUDE_power_of_power_l3023_302327

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3023_302327


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3023_302397

theorem polynomial_real_root (a : ℝ) : ∃ x : ℝ, x^4 - a*x^2 + a*x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3023_302397


namespace NUMINAMATH_CALUDE_no_equal_factorial_and_even_factorial_l3023_302368

theorem no_equal_factorial_and_even_factorial :
  ¬ ∃ (n m : ℕ), n.factorial = 2^m * m.factorial ∧ m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_equal_factorial_and_even_factorial_l3023_302368


namespace NUMINAMATH_CALUDE_failing_marks_difference_l3023_302301

/-- The number of marks needed to pass the exam -/
def passing_marks : ℝ := 199.99999999999997

/-- The percentage of marks obtained by the failing candidate -/
def failing_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the passing candidate -/
def passing_percentage : ℝ := 0.45

/-- The number of marks the passing candidate gets above the passing mark -/
def marks_above_passing : ℝ := 25

/-- Theorem stating the number of marks by which the failing candidate fails -/
theorem failing_marks_difference : 
  let total_marks := (passing_marks + marks_above_passing) / passing_percentage
  passing_marks - (failing_percentage * total_marks) = 50 := by
sorry

end NUMINAMATH_CALUDE_failing_marks_difference_l3023_302301


namespace NUMINAMATH_CALUDE_enrique_shredder_y_feeds_l3023_302303

/-- Calculates the number of times a shredder needs to be fed to shred all pages of a contract type -/
def shredder_feeds (num_contracts : ℕ) (pages_per_contract : ℕ) (pages_per_shred : ℕ) : ℕ :=
  ((num_contracts * pages_per_contract + pages_per_shred - 1) / pages_per_shred : ℕ)

theorem enrique_shredder_y_feeds : 
  let type_b_contracts : ℕ := 350
  let pages_per_type_b : ℕ := 10
  let shredder_y_capacity : ℕ := 8
  shredder_feeds type_b_contracts pages_per_type_b shredder_y_capacity = 438 := by
sorry

end NUMINAMATH_CALUDE_enrique_shredder_y_feeds_l3023_302303


namespace NUMINAMATH_CALUDE_least_amount_to_add_l3023_302380

def savings : ℕ := 642986
def children : ℕ := 9

theorem least_amount_to_add : 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0) → 
  (∃ (x : ℕ), (savings + x) % children = 0 ∧ 
  ∀ (y : ℕ), y < x → (savings + y) % children ≠ 0 ∧ x = 1) :=
sorry

end NUMINAMATH_CALUDE_least_amount_to_add_l3023_302380


namespace NUMINAMATH_CALUDE_total_birds_count_l3023_302317

/-- The number of geese in the marsh -/
def num_geese : ℕ := 58

/-- The number of ducks in the marsh -/
def num_ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := num_geese + num_ducks

/-- Theorem: The total number of birds in the marsh is 95 -/
theorem total_birds_count : total_birds = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l3023_302317


namespace NUMINAMATH_CALUDE_neg_three_at_neg_two_l3023_302395

-- Define the "@" operation
def at_op (x y : ℤ) : ℤ := x * y - y

-- Theorem statement
theorem neg_three_at_neg_two : at_op (-3) (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_neg_three_at_neg_two_l3023_302395


namespace NUMINAMATH_CALUDE_P_no_negative_roots_l3023_302393

/-- The polynomial P(x) = x^4 - 5x^3 + 3x^2 - 7x + 1 -/
def P (x : ℝ) : ℝ := x^4 - 5*x^3 + 3*x^2 - 7*x + 1

/-- Theorem: The polynomial P(x) has no negative roots -/
theorem P_no_negative_roots : ∀ x : ℝ, x < 0 → P x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_P_no_negative_roots_l3023_302393


namespace NUMINAMATH_CALUDE_teacher_budget_theorem_l3023_302337

/-- Calculates the remaining budget for a teacher after purchasing school supplies. -/
def remaining_budget (last_year_budget : ℕ) (this_year_budget : ℕ) (supply1_cost : ℕ) (supply2_cost : ℕ) : ℕ :=
  (last_year_budget + this_year_budget) - (supply1_cost + supply2_cost)

/-- Proves that the remaining budget is 19 given the specific conditions. -/
theorem teacher_budget_theorem :
  remaining_budget 6 50 13 24 = 19 := by
  sorry

end NUMINAMATH_CALUDE_teacher_budget_theorem_l3023_302337


namespace NUMINAMATH_CALUDE_raghu_investment_l3023_302341

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  vishal = 1.1 * trishul →
  trishul = 0.9 * raghu →
  raghu + trishul + vishal = 6358 →
  raghu = 2200 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l3023_302341


namespace NUMINAMATH_CALUDE_lillian_candy_distribution_l3023_302370

theorem lillian_candy_distribution (initial_candies : ℕ) 
  (father_multiplier : ℕ) (num_friends : ℕ) : 
  initial_candies = 205 → 
  father_multiplier = 2 → 
  num_friends = 7 → 
  (initial_candies + father_multiplier * initial_candies) / num_friends = 87 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_distribution_l3023_302370


namespace NUMINAMATH_CALUDE_f_inequality_l3023_302321

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 1) - 1 / (1 + x^2)
  else Real.log (-x + 1) - 1 / (1 + x^2)

theorem f_inequality (a : ℝ) :
  f (a - 2) < f (4 - a^2) ↔ a > 2 ∨ a < -3 ∨ (-1 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l3023_302321


namespace NUMINAMATH_CALUDE_greatest_XPM_l3023_302344

/-- A function that checks if a number is a two-digit number with equal digits -/
def is_two_digit_equal (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

/-- A function that checks if a number is a one-digit prime -/
def is_one_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

/-- The main theorem -/
theorem greatest_XPM :
  ∀ M N XPM : ℕ,
  is_two_digit_equal M →
  is_one_digit_prime N →
  N ≠ M / 11 →
  M * N = XPM →
  100 ≤ XPM ∧ XPM ≤ 999 →
  XPM ≤ 462 :=
sorry

end NUMINAMATH_CALUDE_greatest_XPM_l3023_302344


namespace NUMINAMATH_CALUDE_riverton_soccer_team_l3023_302376

theorem riverton_soccer_team (total_players : ℕ) (math_players : ℕ) (both_players : ℕ) :
  total_players = 15 →
  math_players = 9 →
  both_players = 3 →
  math_players + (total_players - math_players) ≥ total_players →
  total_players - math_players + both_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_riverton_soccer_team_l3023_302376


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l3023_302347

theorem cubic_root_theorem (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 7) →
  ((-6 : ℝ)^3 + b*(-6) + c = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l3023_302347


namespace NUMINAMATH_CALUDE_rectangle_area_l3023_302390

/-- Proves that a rectangular field with sides in ratio 3:4 and perimeter costing 8750 paise
    at 25 paise per metre has an area of 7500 square meters. -/
theorem rectangle_area (length width : ℝ) (h1 : length / width = 3 / 4)
    (h2 : 2 * (length + width) * 25 = 8750) : length * width = 7500 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3023_302390


namespace NUMINAMATH_CALUDE_markup_markdown_l3023_302396

theorem markup_markdown (original_price : ℝ) (markup1 markup2 markup3 markdown : ℝ) : 
  markup1 = 0.1 →
  markup2 = 0.1 →
  markup3 = 0.05 →
  original_price > 0 →
  original_price * (1 + markup1) * (1 + markup2) * (1 + markup3) * (1 - markdown) = original_price →
  ∀ x : ℕ, x < 22 → (1 - (x : ℝ) / 100) > 1 - markdown :=
by sorry

end NUMINAMATH_CALUDE_markup_markdown_l3023_302396


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l3023_302302

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log a / Real.log 9) = (Real.log b / Real.log 12) ∧ 
       (Real.log a / Real.log 9) = (Real.log (2 * (a + b)) / Real.log 16)) : 
  b / a = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l3023_302302


namespace NUMINAMATH_CALUDE_star_perimeter_sum_l3023_302385

theorem star_perimeter_sum (X Y Z : ℕ) : 
  Prime X → Prime Y → Prime Z →
  X < Z → Z < Y → X + Y < 2 * Z →
  X + Y + Z ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_star_perimeter_sum_l3023_302385


namespace NUMINAMATH_CALUDE_range_of_sum_l3023_302371

theorem range_of_sum (a b : ℝ) (ha : -2 < a ∧ a < -1) (hb : -1 < b ∧ b < 0) :
  -3 < a + b ∧ a + b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3023_302371


namespace NUMINAMATH_CALUDE_a_range_l3023_302333

theorem a_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) →
  (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0) →
  ¬((∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) ∧ 
    (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0)) →
  a ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3023_302333


namespace NUMINAMATH_CALUDE_trapezoid_height_theorem_l3023_302322

/-- Represents a trapezoid with given diagonal lengths and midsegment length -/
structure Trapezoid where
  diag1 : ℝ
  diag2 : ℝ
  midsegment : ℝ

/-- Calculates the height of a trapezoid given its properties -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 6 and 8 and midsegment 5 has height 4.8 -/
theorem trapezoid_height_theorem (t : Trapezoid) 
  (h1 : t.diag1 = 6) 
  (h2 : t.diag2 = 8) 
  (h3 : t.midsegment = 5) : 
  trapezoidHeight t = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_height_theorem_l3023_302322


namespace NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_max_distance_l3023_302384

/-- A line passing through point M(1, 2) and intersecting the x-axis and y-axis -/
structure Line where
  m : ℝ × ℝ := (1, 2)
  intersects_x_axis : ℝ → Prop
  intersects_y_axis : ℝ → Prop

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : LineEquation) : ℝ := sorry

theorem line_equation_midpoint (l : Line) : 
  (∃ p q : ℝ × ℝ, l.intersects_x_axis p.1 ∧ l.intersects_y_axis q.2 ∧ 
   l.m = ((p.1 + q.1) / 2, (p.2 + q.2) / 2)) → 
  ∃ eq : LineEquation, eq.a = 2 ∧ eq.b = 1 ∧ eq.c = -4 :=
sorry

theorem line_equation_max_distance (l : Line) :
  (∀ eq : LineEquation, distance_point_to_line (0, 0) eq ≤ 
   distance_point_to_line (0, 0) ⟨1, 2, -5⟩) →
  ∃ eq : LineEquation, eq.a = 1 ∧ eq.b = 2 ∧ eq.c = -5 :=
sorry

end NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_max_distance_l3023_302384


namespace NUMINAMATH_CALUDE_total_coins_always_odd_never_equal_coins_l3023_302388

/-- Represents the state of Laura's coins -/
structure CoinState where
  red : Nat
  green : Nat

/-- Represents the slot machine operation -/
def slotMachine (state : CoinState) (insertRed : Bool) : CoinState :=
  if insertRed then
    { red := state.red - 1, green := state.green + 5 }
  else
    { red := state.red + 5, green := state.green - 1 }

/-- The initial state of Laura's coins -/
def initialState : CoinState := { red := 0, green := 1 }

/-- Theorem stating that the total number of coins is always odd -/
theorem total_coins_always_odd (state : CoinState) (n : Nat) :
  (state.red + state.green) % 2 = 1 := by
  sorry

/-- Theorem stating that Laura can never have an equal number of red and green coins -/
theorem never_equal_coins (state : CoinState) :
  state.red ≠ state.green := by
  sorry

end NUMINAMATH_CALUDE_total_coins_always_odd_never_equal_coins_l3023_302388


namespace NUMINAMATH_CALUDE_equation_roots_l3023_302399

/-- The equation a²(x-2) + a(39-20x) + 20 = 0 has at least two distinct roots if and only if a = 20 -/
theorem equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a^2 * (x - 2) + a * (39 - 20*x) + 20 = 0 ∧ a^2 * (y - 2) + a * (39 - 20*y) + 20 = 0) ↔ 
  a = 20 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l3023_302399


namespace NUMINAMATH_CALUDE_cube_difference_eq_108_l3023_302330

/-- Given two real numbers x and y, if x - y = 3 and x^2 + y^2 = 27, then x^3 - y^3 = 108 -/
theorem cube_difference_eq_108 (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) :
  x^3 - y^3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_eq_108_l3023_302330


namespace NUMINAMATH_CALUDE_impossible_transformation_l3023_302383

/-- Represents a binary sequence -/
inductive BinarySeq
| empty : BinarySeq
| cons : Bool → BinarySeq → BinarySeq

/-- Represents the color of a digit in the sequence -/
inductive Color
| Red
| Green
| Blue

/-- Assigns colors to a binary sequence -/
def colorSequence : BinarySeq → List Color
| BinarySeq.empty => []
| BinarySeq.cons _ rest => [Color.Red, Color.Green, Color.Blue] ++ colorSequence rest

/-- Counts the number of red 1s in a colored binary sequence -/
def countRed1s : BinarySeq → Nat
| BinarySeq.empty => 0
| BinarySeq.cons true (BinarySeq.cons _ (BinarySeq.cons _ rest)) => 1 + countRed1s rest
| BinarySeq.cons false (BinarySeq.cons _ (BinarySeq.cons _ rest)) => countRed1s rest
| _ => 0

/-- Represents an operation on the binary sequence -/
inductive Operation
| Insert : BinarySeq → Operation
| Delete : BinarySeq → Operation

/-- Applies an operation to a binary sequence -/
def applyOperation : BinarySeq → Operation → BinarySeq := sorry

/-- Theorem: It's impossible to transform "10" into "01" using the allowed operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation),
    let initial := BinarySeq.cons true (BinarySeq.cons false BinarySeq.empty)
    let final := BinarySeq.cons false (BinarySeq.cons true BinarySeq.empty)
    let result := ops.foldl applyOperation initial
    result ≠ final :=
sorry

end NUMINAMATH_CALUDE_impossible_transformation_l3023_302383


namespace NUMINAMATH_CALUDE_modulus_of_z_l3023_302369

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3023_302369


namespace NUMINAMATH_CALUDE_rose_friends_count_l3023_302348

def total_apples : ℕ := 9
def apples_per_friend : ℕ := 3

theorem rose_friends_count : 
  total_apples / apples_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_rose_friends_count_l3023_302348


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3023_302355

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sum_of_digits (10^95 - 95 - 2) = 840 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3023_302355


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_c_eq_2a_cos_B_l3023_302323

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Side lengths

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- State the theorem
theorem triangle_isosceles_if_c_eq_2a_cos_B (t : Triangle) 
  (h : t.c = 2 * t.a * Real.cos t.B) : isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_c_eq_2a_cos_B_l3023_302323


namespace NUMINAMATH_CALUDE_max_sum_constrained_l3023_302351

theorem max_sum_constrained (x y : ℝ) 
  (h1 : 5 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y = 15) : 
  x + y ≤ 47 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l3023_302351


namespace NUMINAMATH_CALUDE_max_value_x_y4_z5_l3023_302353

theorem max_value_x_y4_z5 (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  ∃ (max : ℝ), max = 243 ∧ x + y^4 + z^5 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_y4_z5_l3023_302353


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l3023_302391

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  blue_white : ℕ

/-- Given the initial triangle counts and the number of coinciding pairs of various types,
    calculates the number of white-white pairs that coincide when the figure is folded -/
def calculate_white_pairs (counts : TriangleCounts) (pairs : CoincidingPairs) : ℕ :=
  sorry

/-- Theorem stating that under the given conditions, 5 white pairs coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) 
  (h1 : counts.red = 5)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : pairs.red_red = 3)
  (h5 : pairs.blue_blue = 2)
  (h6 : pairs.red_white = 3)
  (h7 : pairs.blue_white = 1) :
  calculate_white_pairs counts pairs = 5 :=
by sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l3023_302391


namespace NUMINAMATH_CALUDE_polynomial_equality_l3023_302367

theorem polynomial_equality (x : ℝ) (h : 2 * x^2 - x = 1) :
  4 * x^4 - 4 * x^3 + 3 * x^2 - x - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3023_302367
