import Mathlib

namespace NUMINAMATH_CALUDE_student_d_not_top_student_l84_8464

/-- Represents a student's rankings in three consecutive exams -/
structure StudentRankings :=
  (r1 r2 r3 : ℕ)

/-- Calculates the mode of three numbers -/
def mode (a b c : ℕ) : ℕ := sorry

/-- Calculates the variance of three numbers -/
def variance (a b c : ℕ) : ℚ := sorry

/-- Determines if a student is a top student based on their rankings -/
def is_top_student (s : StudentRankings) : Prop :=
  s.r1 ≤ 3 ∧ s.r2 ≤ 3 ∧ s.r3 ≤ 3

theorem student_d_not_top_student (s : StudentRankings) :
  mode s.r1 s.r2 s.r3 = 2 ∧ variance s.r1 s.r2 s.r3 > 1 →
  ¬(is_top_student s) := by sorry

end NUMINAMATH_CALUDE_student_d_not_top_student_l84_8464


namespace NUMINAMATH_CALUDE_point_symmetric_second_quadrant_l84_8457

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is symmetric about the y-axis -/
def isSymmetricAboutYAxis (p : Point) : Prop :=
  p.x < 0

/-- The main theorem -/
theorem point_symmetric_second_quadrant (a : ℝ) :
  let A : Point := ⟨a - 1, 2 * a - 4⟩
  isInSecondQuadrant A ∧ isSymmetricAboutYAxis A → a > 2 :=
by sorry

end NUMINAMATH_CALUDE_point_symmetric_second_quadrant_l84_8457


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l84_8444

theorem unique_remainder_mod_10 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 10] := by sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l84_8444


namespace NUMINAMATH_CALUDE_union_M_N_l84_8433

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_M_N_l84_8433


namespace NUMINAMATH_CALUDE_value_of_expression_l84_8479

theorem value_of_expression (x y : ℤ) (hx : x = 12) (hy : y = 18) :
  3 * (x - y) * (x + y) = -540 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l84_8479


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l84_8475

/-- Calculates the profit percentage for a retailer given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific conditions, the profit percentage is 20%. -/
theorem retailer_profit_percentage :
  let wholesale_price : ℚ := 81
  let retail_price : ℚ := 108
  let discount_percent : ℚ := 10
  profit_percentage wholesale_price retail_price discount_percent = 20 := by
sorry

#eval profit_percentage 81 108 10

end NUMINAMATH_CALUDE_retailer_profit_percentage_l84_8475


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l84_8495

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ := 180 - t.A

-- Theorem statement
theorem isosceles_triangle_angles (t : Triangle) :
  exteriorAngleA t = 110 →
  isIsosceles t →
  t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angles_l84_8495


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l84_8493

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f: ℝ → ℝ, f(x)f(-x) ≤ 0 for all x ∈ ℝ -/
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : IsOdd f) :
  ∀ x, f x * f (-x) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l84_8493


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l84_8401

theorem sqrt_expression_equality : 
  (Real.sqrt 48 - Real.sqrt 27) / Real.sqrt 3 + Real.sqrt 6 * 2 * Real.sqrt (1/3) = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l84_8401


namespace NUMINAMATH_CALUDE_annie_cookies_count_l84_8424

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 2 / 5)

/-- The total number of cookies Annie ate during the three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

theorem annie_cookies_count : total_cookies = 29 := by
  sorry

end NUMINAMATH_CALUDE_annie_cookies_count_l84_8424


namespace NUMINAMATH_CALUDE_pump_fills_in_four_hours_l84_8480

/-- Represents the time (in hours) it takes to fill or empty the tank -/
structure TankTime where
  hours : ℝ
  hours_pos : hours > 0

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : TankTime := sorry

/-- The time it takes for the leak to empty the full tank -/
def leak_time : TankTime :=
  { hours := 5
    hours_pos := by norm_num }

/-- The time it takes for the pump and leak combined to fill the tank -/
def combined_time : TankTime :=
  { hours := 20
    hours_pos := by norm_num }

theorem pump_fills_in_four_hours :
  pump_time.hours = 4 :=
by
  have h1 : (1 / pump_time.hours) - (1 / leak_time.hours) = 1 / combined_time.hours :=
    sorry
  sorry

end NUMINAMATH_CALUDE_pump_fills_in_four_hours_l84_8480


namespace NUMINAMATH_CALUDE_inequality_proof_l84_8488

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / (a * b) ≥ 1 / 4) ∧ (a^2 + b^2 ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l84_8488


namespace NUMINAMATH_CALUDE_square_and_reciprocal_square_l84_8486

theorem square_and_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_and_reciprocal_square_l84_8486


namespace NUMINAMATH_CALUDE_parabola_directrix_l84_8434

/-- Given a parabola y^2 = 2px where p > 0 that passes through the point (1, 1/2),
    its directrix has the equation x = -1/16 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x) →
  ((1 : ℝ)^2 = 2*p*(1/2 : ℝ)^2) →
  (∃ k : ℝ, ∀ x : ℝ, x = k ↔ x = -1/16) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l84_8434


namespace NUMINAMATH_CALUDE_article_cost_l84_8456

theorem article_cost (decreased_price : ℝ) (decrease_percentage : ℝ) (h1 : decreased_price = 760) (h2 : decrease_percentage = 24) : 
  ∃ (original_price : ℝ), original_price * (1 - decrease_percentage / 100) = decreased_price ∧ original_price = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_article_cost_l84_8456


namespace NUMINAMATH_CALUDE_polynomial_without_xy_term_l84_8468

theorem polynomial_without_xy_term (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 - 3*y^2 - 8) ↔ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_without_xy_term_l84_8468


namespace NUMINAMATH_CALUDE_vartan_recreation_spending_l84_8432

theorem vartan_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
  last_week_percent > 0 →
  let this_week_wages := 0.9 * last_week_wages
  let last_week_spending := (last_week_percent / 100) * last_week_wages
  let this_week_spending := 0.3 * this_week_wages
  this_week_spending = 1.8 * last_week_spending →
  last_week_percent = 15 := by
sorry

end NUMINAMATH_CALUDE_vartan_recreation_spending_l84_8432


namespace NUMINAMATH_CALUDE_sum_five_consecutive_integers_l84_8425

/-- Given a sequence of five consecutive integers with middle number m,
    prove that their sum is equal to 5m. -/
theorem sum_five_consecutive_integers (m : ℤ) : 
  (m - 2) + (m - 1) + m + (m + 1) + (m + 2) = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_integers_l84_8425


namespace NUMINAMATH_CALUDE_solve_for_y_l84_8494

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 7) : y = 55/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l84_8494


namespace NUMINAMATH_CALUDE_complex_quadratic_roots_l84_8404

theorem complex_quadratic_roots (z : ℂ) : 
  z^2 = -63 + 16*I ∧ (7 + 4*I)^2 = -63 + 16*I → 
  z = 7 + 4*I ∨ z = -7 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_roots_l84_8404


namespace NUMINAMATH_CALUDE_jane_daffodil_bulbs_l84_8472

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

theorem jane_daffodil_bulbs :
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let crocus_bulbs : ℕ := 3 * daffodil_bulbs
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let earnings_per_bulb : ℚ := 1/2
  let total_earnings : ℚ := 75
  total_earnings = earnings_per_bulb * total_bulbs :=
by sorry


end NUMINAMATH_CALUDE_jane_daffodil_bulbs_l84_8472


namespace NUMINAMATH_CALUDE_paperClips_in_two_cases_l84_8447

/-- The number of paper clips in 2 cases -/
def paperClipsIn2Cases (c b : ℕ) : ℕ := 2 * c * b * 300

/-- Theorem: The number of paper clips in 2 cases is 2c * b * 300 -/
theorem paperClips_in_two_cases (c b : ℕ) : 
  paperClipsIn2Cases c b = 2 * c * b * 300 := by
  sorry

end NUMINAMATH_CALUDE_paperClips_in_two_cases_l84_8447


namespace NUMINAMATH_CALUDE_inequality_equivalence_l84_8489

theorem inequality_equivalence (x : ℝ) : (x - 5) / 2 + 1 > x - 3 ↔ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l84_8489


namespace NUMINAMATH_CALUDE_even_sum_probability_l84_8463

/-- Represents a wheel with sections --/
structure Wheel where
  totalSections : Nat
  evenSections : Nat
  oddSections : Nat
  zeroSections : Nat

/-- First wheel configuration --/
def wheel1 : Wheel := {
  totalSections := 6
  evenSections := 2
  oddSections := 3
  zeroSections := 1
}

/-- Second wheel configuration --/
def wheel2 : Wheel := {
  totalSections := 4
  evenSections := 2
  oddSections := 2
  zeroSections := 0
}

/-- Calculate the probability of getting an even sum when spinning two wheels --/
def probabilityEvenSum (w1 w2 : Wheel) : Real :=
  sorry

/-- Theorem: The probability of getting an even sum when spinning the two given wheels is 1/2 --/
theorem even_sum_probability :
  probabilityEvenSum wheel1 wheel2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_probability_l84_8463


namespace NUMINAMATH_CALUDE_geese_flew_away_l84_8441

/-- Proves that the number of geese that flew away is equal to the difference
    between the initial number of geese and the number of geese left in the field. -/
theorem geese_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ)
    (h1 : initial = 51)
    (h2 : left = 23)
    (h3 : initial ≥ left) :
  flew_away = initial - left :=
by sorry

end NUMINAMATH_CALUDE_geese_flew_away_l84_8441


namespace NUMINAMATH_CALUDE_bathtub_fill_time_with_open_drain_l84_8427

/-- Represents the time it takes to fill a bathtub with the drain open. -/
def fill_time_with_open_drain (fill_time drain_time : ℚ) : ℚ :=
  (fill_time * drain_time) / (drain_time - fill_time)

/-- Theorem stating that a bathtub taking 10 minutes to fill and 12 minutes to drain
    will take 60 minutes to fill with the drain open. -/
theorem bathtub_fill_time_with_open_drain :
  fill_time_with_open_drain 10 12 = 60 := by
  sorry

#eval fill_time_with_open_drain 10 12

end NUMINAMATH_CALUDE_bathtub_fill_time_with_open_drain_l84_8427


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l84_8414

theorem sum_of_cubes_difference (d e f : ℕ+) :
  (d + e + f : ℕ)^3 - d^3 - e^3 - f^3 = 300 → d + e + f = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l84_8414


namespace NUMINAMATH_CALUDE_expression_simplification_l84_8465

theorem expression_simplification (b x : ℝ) 
  (hb : b ≠ 0) (hx : x ≠ 0) (hxb : x ≠ b/2) (hxb2 : x ≠ -2/b) :
  ((b*x + 4 + 4/(b*x)) / (2*b + (b^2 - 4)*x - 2*b*x^2) + 
   ((4*x^2 - b^2) / b) / ((b + 2*x)^2 - 8*b*x)) * (b*x/2) = 
  (x^2 - 1) / (2*x - b) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l84_8465


namespace NUMINAMATH_CALUDE_tan_half_angle_problem_l84_8481

theorem tan_half_angle_problem (α : Real) (h : Real.tan (α / 2) = 2) :
  (Real.tan (α + Real.pi / 4) = -1 / 7) ∧
  ((6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_problem_l84_8481


namespace NUMINAMATH_CALUDE_find_a_l84_8408

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

-- Define set A
def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

-- Define the complement of A with respect to U
def complementA (a : ℝ) : Set ℝ := (U a) \ (A a)

-- Theorem statement
theorem find_a : ∃ a : ℝ, (U a = {1, 2, a^2 + 2*a - 3}) ∧ 
                          (A a = {|a - 2|, 2}) ∧ 
                          (complementA a = {0}) ∧ 
                          (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l84_8408


namespace NUMINAMATH_CALUDE_factors_of_product_l84_8413

/-- A function that returns the number of factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that returns the number of factors of n^k for a natural number n and exponent k -/
def num_factors_power (n k : ℕ) : ℕ := sorry

theorem factors_of_product (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  num_factors a = 3 →
  num_factors b = 3 →
  num_factors c = 4 →
  num_factors_power a 3 * num_factors_power b 4 * num_factors_power c 5 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_product_l84_8413


namespace NUMINAMATH_CALUDE_distance_sum_bounds_l84_8477

/-- Given points A, B, C in a 2D plane and a point P satisfying x^2 + y^2 ≤ 4,
    the sum of squared distances from P to A, B, and C is between 72 and 88. -/
theorem distance_sum_bounds (x y : ℝ) :
  x^2 + y^2 ≤ 4 →
  72 ≤ ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ∧
  ((x + 2)^2 + (y - 2)^2) + ((x + 2)^2 + (y - 6)^2) + ((x - 4)^2 + (y + 2)^2) ≤ 88 :=
by sorry

end NUMINAMATH_CALUDE_distance_sum_bounds_l84_8477


namespace NUMINAMATH_CALUDE_largest_integer_solution_l84_8487

theorem largest_integer_solution (m : ℤ) : (2 * m + 7 ≤ 3) → m ≤ -2 ∧ ∀ k : ℤ, (2 * k + 7 ≤ 3) → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l84_8487


namespace NUMINAMATH_CALUDE_total_tires_is_101_l84_8400

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of tires on a bicycle -/
def bicycle_tires : ℕ := 2

/-- The number of tires on a pickup truck -/
def pickup_truck_tires : ℕ := 4

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of cars Juan saw -/
def cars_seen : ℕ := 15

/-- The number of bicycles Juan saw -/
def bicycles_seen : ℕ := 3

/-- The number of pickup trucks Juan saw -/
def pickup_trucks_seen : ℕ := 8

/-- The number of tricycles Juan saw -/
def tricycles_seen : ℕ := 1

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 
  cars_seen * car_tires + 
  bicycles_seen * bicycle_tires + 
  pickup_trucks_seen * pickup_truck_tires + 
  tricycles_seen * tricycle_tires

theorem total_tires_is_101 : total_tires = 101 := by
  sorry

end NUMINAMATH_CALUDE_total_tires_is_101_l84_8400


namespace NUMINAMATH_CALUDE_cos_450_degrees_l84_8429

theorem cos_450_degrees : Real.cos (450 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_450_degrees_l84_8429


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l84_8430

theorem cos_alpha_minus_beta (α β : Real) 
  (h1 : α > -π/4 ∧ α < π/4) 
  (h2 : β > -π/4 ∧ β < π/4) 
  (h3 : Real.cos (2*α + 2*β) = -7/9) 
  (h4 : Real.sin α * Real.sin β = 1/4) : 
  Real.cos (α - β) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l84_8430


namespace NUMINAMATH_CALUDE_range_of_a_l84_8499

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → 
    (x + y)^2 - a * (x + y) + 1 ≥ 0) ↔ 
  a ≤ 37 / 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l84_8499


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l84_8483

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = a^2 - 1 ∧ (a - 1 ≠ 0) → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l84_8483


namespace NUMINAMATH_CALUDE_gift_contribution_l84_8442

theorem gift_contribution (a b c d e : ℝ) : 
  a + b + c + d + e = 120 →
  a = 2 * b →
  b = (1/3) * (c + d) →
  c = 2 * e →
  e = 12 := by sorry

end NUMINAMATH_CALUDE_gift_contribution_l84_8442


namespace NUMINAMATH_CALUDE_gift_wrap_sales_l84_8452

theorem gift_wrap_sales (total_goal : ℕ) (grandmother_sales uncle_sales neighbor_sales : ℕ) : 
  total_goal = 45 ∧ 
  grandmother_sales = 1 ∧ 
  uncle_sales = 10 ∧ 
  neighbor_sales = 6 → 
  total_goal - (grandmother_sales + uncle_sales + neighbor_sales) = 28 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrap_sales_l84_8452


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l84_8470

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 8 * y + 44 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l84_8470


namespace NUMINAMATH_CALUDE_xoz_symmetry_of_M_l84_8415

/-- Defines a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the xoz plane symmetry operation -/
def xozPlaneSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point of M(5, 1, -2) with respect to the xoz plane is (5, -1, -2) -/
theorem xoz_symmetry_of_M :
  let M : Point3D := { x := 5, y := 1, z := -2 }
  xozPlaneSymmetry M = { x := 5, y := -1, z := -2 } := by
  sorry

end NUMINAMATH_CALUDE_xoz_symmetry_of_M_l84_8415


namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l84_8491

/-- Calculate Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (price_per_pan : ℝ) (num_pans : ℕ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

/-- Theorem: Jenny's profit is $300 given the problem conditions -/
theorem jenny_lasagna_profit :
  jennys_profit 10 25 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l84_8491


namespace NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l84_8446

/-- The surface area of a cube with space diagonal length 6 is 72 -/
theorem cube_surface_area_from_diagonal (d : ℝ) (h : d = 6) : 
  6 * (d / Real.sqrt 3) ^ 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_diagonal_l84_8446


namespace NUMINAMATH_CALUDE_reciprocal_comparison_l84_8461

theorem reciprocal_comparison : ∃ (S : Set ℝ), 
  S = {-3, -1/2, 0.5, 1, 3} ∧ 
  (∀ x ∈ S, x < 1 / x ↔ (x = -3 ∨ x = 0.5)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_l84_8461


namespace NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l84_8428

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l84_8428


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l84_8438

theorem twenty_five_percent_less_than_80 (x : ℝ) : 
  (60 : ℝ) = 80 * (3/4) → x + x/4 = 60 → x = 48 := by sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l84_8438


namespace NUMINAMATH_CALUDE_seniority_ordering_l84_8448

-- Define the colleagues
inductive Colleague
| Tom
| Jerry
| Sam

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Jerry Colleague.Sam
def statement_II : Prop := more_senior Colleague.Sam Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Jerry
def statement_III : Prop := more_senior Colleague.Jerry Colleague.Tom ∨ more_senior Colleague.Sam Colleague.Tom

-- Theorem statement
theorem seniority_ordering :
  -- Exactly one statement is true
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ statement_II ∧ ¬statement_III) ∨
  (¬statement_I ∧ ¬statement_II ∧ statement_III) →
  -- Seniority relation is transitive
  (∀ a b c : Colleague, more_senior a b → more_senior b c → more_senior a c) →
  -- Seniority relation is asymmetric
  (∀ a b : Colleague, more_senior a b → ¬more_senior b a) →
  -- All colleagues have different seniorities
  (∀ a b : Colleague, a ≠ b → more_senior a b ∨ more_senior b a) →
  -- The correct seniority ordering
  more_senior Colleague.Jerry Colleague.Tom ∧ more_senior Colleague.Tom Colleague.Sam :=
sorry

end NUMINAMATH_CALUDE_seniority_ordering_l84_8448


namespace NUMINAMATH_CALUDE_convex_polygon_angle_theorem_l84_8478

theorem convex_polygon_angle_theorem (n : ℕ) (x : ℝ) :
  n ≥ 3 →
  x > 0 →
  x < 180 →
  (n : ℝ) * 180 - 3 * x = 3330 + 180 * 2 →
  x = 54 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_angle_theorem_l84_8478


namespace NUMINAMATH_CALUDE_circle_intersection_range_l84_8462

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + m + 6 = 0

-- Define the condition for intersection with y-axis
def intersects_y_axis (m : ℝ) : Prop :=
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ circle_equation 0 y1 m ∧ circle_equation 0 y2 m

-- Define the condition for points being on the same side of the origin
def same_side_of_origin (y1 y2 : ℝ) : Prop :=
  (y1 > 0 ∧ y2 > 0) ∨ (y1 < 0 ∧ y2 < 0)

-- Main theorem
theorem circle_intersection_range (m : ℝ) :
  (intersects_y_axis m ∧ 
   ∀ y1 y2 : ℝ, circle_equation 0 y1 m → circle_equation 0 y2 m → same_side_of_origin y1 y2) →
  -6 < m ∧ m < -5 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l84_8462


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l84_8497

theorem unique_solution_sqrt_equation :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l84_8497


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l84_8426

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l84_8426


namespace NUMINAMATH_CALUDE_nickel_dime_difference_l84_8420

/-- The value of one dollar in cents -/
def dollar : ℕ := 100

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The number of coins needed to make one dollar using only coins of a given value -/
def coinsNeeded (coinValue : ℕ) : ℕ := dollar / coinValue

theorem nickel_dime_difference :
  coinsNeeded nickel - coinsNeeded dime = 10 := by sorry

end NUMINAMATH_CALUDE_nickel_dime_difference_l84_8420


namespace NUMINAMATH_CALUDE_local_maximum_value_l84_8416

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x - 1

theorem local_maximum_value (x : ℝ) :
  ∃ (a : ℝ), (∀ (y : ℝ), ∃ (ε : ℝ), ε > 0 ∧ ∀ (z : ℝ), |z - a| < ε → f z ≤ f a) ∧
  f a = -23/27 :=
sorry

end NUMINAMATH_CALUDE_local_maximum_value_l84_8416


namespace NUMINAMATH_CALUDE_half_vector_MN_l84_8492

/-- Given two vectors OM and ON in ℝ², prove that half of vector MN equals (-4, 1/2) -/
theorem half_vector_MN (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  (1 / 2 : ℝ) • (ON - OM) = (-4, 1/2) := by sorry

end NUMINAMATH_CALUDE_half_vector_MN_l84_8492


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l84_8439

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l84_8439


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l84_8409

theorem imaginary_part_of_complex_division : 
  Complex.im ((3 + 4 * Complex.I) / Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l84_8409


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l84_8454

theorem triangle_third_side_length (a b : ℝ) (ha : a = 6.31) (hb : b = 0.82) :
  ∃! c : ℕ, (c : ℝ) > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l84_8454


namespace NUMINAMATH_CALUDE_continuous_function_composition_eq_power_l84_8436

/-- A continuous function satisfying f(f(x)) = kx^9 exists if and only if k ≥ 0 -/
theorem continuous_function_composition_eq_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) ↔ k ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_continuous_function_composition_eq_power_l84_8436


namespace NUMINAMATH_CALUDE_number_equation_l84_8421

theorem number_equation (x : ℝ) : (9 * x) / 3 = 27 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l84_8421


namespace NUMINAMATH_CALUDE_octopus_ink_conversion_l84_8455

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (units : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + units * 8^0

/-- The octopus ink problem -/
theorem octopus_ink_conversion :
  base8ToBase10 2 7 6 = 190 := by
  sorry

end NUMINAMATH_CALUDE_octopus_ink_conversion_l84_8455


namespace NUMINAMATH_CALUDE_max_value_abc_expression_l84_8431

theorem max_value_abc_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b * c^2) (hbc : b ≥ c * a^2) (hca : c ≥ a * b^2) :
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2) ≤ 0 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    a₀ ≥ b₀ * c₀^2 ∧ b₀ ≥ c₀ * a₀^2 ∧ c₀ ≥ a₀ * b₀^2 ∧
    a₀ * b₀ * c₀ * (a₀ - b₀ * c₀^2) * (b₀ - c₀ * a₀^2) * (c₀ - a₀ * b₀^2) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_expression_l84_8431


namespace NUMINAMATH_CALUDE_sprint_team_distance_l84_8471

/-- Given a sprint team with a certain number of people, where each person runs a fixed distance,
    calculate the total distance run by the team. -/
def total_distance (team_size : ℝ) (distance_per_person : ℝ) : ℝ :=
  team_size * distance_per_person

/-- Theorem: A sprint team of 150.0 people, where each person runs 5.0 miles,
    will run a total of 750.0 miles. -/
theorem sprint_team_distance :
  total_distance 150.0 5.0 = 750.0 := by
  sorry

end NUMINAMATH_CALUDE_sprint_team_distance_l84_8471


namespace NUMINAMATH_CALUDE_class_size_calculation_l84_8412

theorem class_size_calculation (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 72 →
  excluded_average = 40 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ (total_count : ℕ),
    (total_count : ℝ) * total_average = 
      (total_count - excluded_count : ℝ) * remaining_average + (excluded_count : ℝ) * excluded_average ∧
    total_count = 13 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l84_8412


namespace NUMINAMATH_CALUDE_unique_k_l84_8496

theorem unique_k : ∃! (k : ℕ), k > 0 ∧ (k + 2).factorial + (k + 3).factorial = k.factorial * 1344 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_l84_8496


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l84_8485

theorem fifteenth_student_age
  (total_students : ℕ)
  (total_average_age : ℝ)
  (group1_students : ℕ)
  (group1_average_age : ℝ)
  (group2_students : ℕ)
  (group2_average_age : ℝ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 4)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 10)
  (h6 : group2_average_age = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l84_8485


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l84_8469

/-- A geometric sequence of 5 terms -/
def GeometricSequence (a : Fin 5 → ℝ) : Prop :=
  ∀ i j k, i < j → j < k → a i * a k = a j ^ 2

theorem geometric_sequence_product (a : Fin 5 → ℝ) 
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 1/2)
  (h_last : a 4 = 8) :
  a 1 * a 2 * a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l84_8469


namespace NUMINAMATH_CALUDE_jordana_age_proof_l84_8443

/-- Jennifer's current age -/
def jennifer_current_age : ℕ := 30 - 10

/-- Jennifer's age in 10 years -/
def jennifer_future_age : ℕ := 30

/-- Jordana's age in 10 years -/
def jordana_future_age : ℕ := 3 * jennifer_future_age

/-- Jordana's current age -/
def jordana_current_age : ℕ := jordana_future_age - 10

theorem jordana_age_proof : jordana_current_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_jordana_age_proof_l84_8443


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l84_8482

def a : Fin 2 → ℝ := ![3, 2]
def b (n : ℝ) : Fin 2 → ℝ := ![2, n]

theorem perpendicular_vectors (n : ℝ) : 
  (∀ i : Fin 2, (a i) * (b n i) = 0) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l84_8482


namespace NUMINAMATH_CALUDE_x_value_proof_l84_8407

theorem x_value_proof (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x - y^2 = 3) (h4 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l84_8407


namespace NUMINAMATH_CALUDE_prepend_append_divisible_by_72_l84_8411

theorem prepend_append_divisible_by_72 : ∃ (a b : Nat), 
  a < 10 ∧ b < 10 ∧ 
  (1000 * a + 100 + b = 4104) ∧ 
  (4104 % 72 = 0) := by
  sorry

end NUMINAMATH_CALUDE_prepend_append_divisible_by_72_l84_8411


namespace NUMINAMATH_CALUDE_lines_non_intersecting_l84_8419

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being parallel to a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the property of a line being contained within a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the property of two lines being non-intersecting
variable (non_intersecting : Line → Line → Prop)

-- State the theorem
theorem lines_non_intersecting
  (l a : Line) (α : Plane)
  (h1 : parallel_to_plane l α)
  (h2 : contained_in_plane a α) :
  non_intersecting l a :=
sorry

end NUMINAMATH_CALUDE_lines_non_intersecting_l84_8419


namespace NUMINAMATH_CALUDE_square_measurement_error_l84_8498

theorem square_measurement_error (S : ℝ) (S' : ℝ) (h : S > 0) :
  S'^2 = S^2 * (1 + 0.0404) → (S' - S) / S * 100 = 2 := by sorry

end NUMINAMATH_CALUDE_square_measurement_error_l84_8498


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l84_8435

theorem rational_sum_theorem (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l84_8435


namespace NUMINAMATH_CALUDE_committee_count_l84_8417

theorem committee_count (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (m - 2)) = 840 := by
  sorry

end NUMINAMATH_CALUDE_committee_count_l84_8417


namespace NUMINAMATH_CALUDE_qqLive_higher_score_l84_8451

structure SoftwareRating where
  name : String
  studentRatings : List Nat
  studentAverage : Float
  teacherAverage : Float

def comprehensiveScore (rating : SoftwareRating) : Float :=
  rating.studentAverage * 0.4 + rating.teacherAverage * 0.6

def dingtalk : SoftwareRating := {
  name := "DingTalk",
  studentRatings := [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
  studentAverage := 3.4,
  teacherAverage := 3.9
}

def qqLive : SoftwareRating := {
  name := "QQ Live",
  studentRatings := [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5],
  studentAverage := 3.35,
  teacherAverage := 4.0
}

theorem qqLive_higher_score : comprehensiveScore qqLive > comprehensiveScore dingtalk := by
  sorry

end NUMINAMATH_CALUDE_qqLive_higher_score_l84_8451


namespace NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l84_8405

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

-- Theorem for part I
theorem unique_solution_condition (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 := by sorry

-- Theorem for part II
theorem inequality_condition (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l84_8405


namespace NUMINAMATH_CALUDE_ann_age_is_36_l84_8403

/-- Represents the ages of Ann and Barbara -/
structure Ages where
  ann : ℕ
  barbara : ℕ

/-- The condition that the sum of their ages is 72 -/
def sum_of_ages (ages : Ages) : Prop :=
  ages.ann + ages.barbara = 72

/-- The complex relationship between their ages as described in the problem -/
def age_relationship (ages : Ages) : Prop :=
  ages.barbara = ages.ann - (ages.barbara - (ages.ann - (ages.ann / 3)))

/-- The theorem stating that given the conditions, Ann's age is 36 -/
theorem ann_age_is_36 (ages : Ages) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relationship ages) : 
  ages.ann = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_age_is_36_l84_8403


namespace NUMINAMATH_CALUDE_complex_root_modulus_l84_8445

theorem complex_root_modulus (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_l84_8445


namespace NUMINAMATH_CALUDE_function_passes_through_point_l84_8453

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 3 passes through the point (1, 4) -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l84_8453


namespace NUMINAMATH_CALUDE_prob_three_games_correct_constant_term_is_three_f_one_half_l84_8402

/-- Represents the probability of player A winning a single game -/
def p : ℝ := sorry

/-- Assumption that p is between 0 and 1 -/
axiom p_range : 0 ≤ p ∧ p ≤ 1

/-- The probability of the match ending in three games -/
def prob_three_games : ℝ := p^3 + (1-p)^3

/-- The expected number of games played in the match -/
def f (p : ℝ) : ℝ := 6*p^4 - 12*p^3 + 3*p^2 + 3*p + 3

/-- Theorem: The probability of the match ending in three games is p³ + (1-p)³ -/
theorem prob_three_games_correct : prob_three_games = p^3 + (1-p)^3 := by sorry

/-- Theorem: The constant term of f(p) is 3 -/
theorem constant_term_is_three : f 0 = 3 := by sorry

/-- Theorem: f(1/2) = 33/8 -/
theorem f_one_half : f (1/2) = 33/8 := by sorry

end NUMINAMATH_CALUDE_prob_three_games_correct_constant_term_is_three_f_one_half_l84_8402


namespace NUMINAMATH_CALUDE_limit_of_sequence_l84_8422

def a (n : ℕ) : ℚ := (2 * n - 5 : ℚ) / (3 * n + 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l84_8422


namespace NUMINAMATH_CALUDE_certain_number_proof_l84_8450

theorem certain_number_proof (x : ℝ) : x / 14.5 = 177 → x = 2566.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l84_8450


namespace NUMINAMATH_CALUDE_not_divisible_and_only_prime_l84_8467

theorem not_divisible_and_only_prime (n : ℕ) : 
  (n > 1 → ¬(n ∣ (2^n - 1))) ∧ 
  (n.Prime ∧ n^2 ∣ (2^n + 1) ↔ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_and_only_prime_l84_8467


namespace NUMINAMATH_CALUDE_percentage_problem_l84_8484

theorem percentage_problem (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 99 → x = 4400 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l84_8484


namespace NUMINAMATH_CALUDE_trig_equation_solution_l84_8449

theorem trig_equation_solution (a b c α β : ℝ) 
  (h1 : a * Real.cos α + b * Real.sin α = c)
  (h2 : a * Real.cos β + b * Real.sin β = c)
  (h3 : a^2 + b^2 ≠ 0)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  (Real.cos ((α - β) / 2))^2 = c^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l84_8449


namespace NUMINAMATH_CALUDE_dr_strange_food_choices_l84_8440

/-- Represents the number of food items and days --/
def n : ℕ := 12

/-- Represents the ways to choose food items each day --/
def choices : ℕ → ℕ
  | 0 => 2  -- First day has 2 choices
  | i => 2  -- Each subsequent day has 2 choices

/-- The total number of ways to choose food items over n days --/
def totalWays : ℕ := 2^n

theorem dr_strange_food_choices :
  totalWays = 2048 := by sorry

end NUMINAMATH_CALUDE_dr_strange_food_choices_l84_8440


namespace NUMINAMATH_CALUDE_function_property_l84_8459

/-- Given a function g : ℝ → ℝ satisfying g(x)g(y) - g(xy) = x - y for all real x and y,
    prove that g(3) = -2 -/
theorem function_property (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g x * g y - g (x * y) = x - y) : 
    g 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l84_8459


namespace NUMINAMATH_CALUDE_cycle_loss_percentage_l84_8418

/-- Calculates the percentage of loss given the cost price and selling price -/
def percentage_loss (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The percentage of loss for a cycle with cost price 1200 and selling price 1020 is 15% -/
theorem cycle_loss_percentage :
  let cost_price : ℚ := 1200
  let selling_price : ℚ := 1020
  percentage_loss cost_price selling_price = 15 := by
  sorry

#eval percentage_loss 1200 1020

end NUMINAMATH_CALUDE_cycle_loss_percentage_l84_8418


namespace NUMINAMATH_CALUDE_last_digit_379_base_4_l84_8473

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_379_base_4 :
  last_digit_base_4 379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_379_base_4_l84_8473


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l84_8406

theorem consecutive_numbers_sum (n : ℕ) : 
  (∃ a : ℕ, (∀ k : ℕ, k < n → ∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6))) ∧
  (∀ k : ℕ, k ≥ 385 → ¬∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    ∃ a : ℕ, a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6)) →
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l84_8406


namespace NUMINAMATH_CALUDE_greatest_fleet_number_l84_8410

/-- A ship is a set of connected unit squares on a grid. -/
def Ship := Set (Nat × Nat)

/-- A fleet is a set of vertex-disjoint ships. -/
def Fleet := Set Ship

/-- The grid size. -/
def gridSize : Nat := 10

/-- Checks if two ships are vertex-disjoint. -/
def vertexDisjoint (s1 s2 : Ship) : Prop := sorry

/-- Checks if a fleet is valid (all ships are vertex-disjoint). -/
def validFleet (f : Fleet) : Prop := sorry

/-- Checks if a ship is within the grid bounds. -/
def inGridBounds (s : Ship) : Prop := sorry

/-- Checks if a fleet configuration is valid for a given partition. -/
def validFleetForPartition (n : Nat) (partition : List Nat) (f : Fleet) : Prop := sorry

/-- The main theorem stating that 25 is the greatest number satisfying the fleet condition. -/
theorem greatest_fleet_number : 
  (∀ (partition : List Nat), partition.sum = 25 → 
    ∃ (f : Fleet), validFleet f ∧ validFleetForPartition 25 partition f) ∧
  (∀ (n : Nat), n > 25 → 
    ∃ (partition : List Nat), partition.sum = n ∧
      ¬∃ (f : Fleet), validFleet f ∧ validFleetForPartition n partition f) :=
sorry

end NUMINAMATH_CALUDE_greatest_fleet_number_l84_8410


namespace NUMINAMATH_CALUDE_product_of_fractions_l84_8490

theorem product_of_fractions (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l84_8490


namespace NUMINAMATH_CALUDE_movie_ticket_price_difference_l84_8437

theorem movie_ticket_price_difference (regular_price children_price : ℕ) : 
  regular_price = 9 →
  2 * regular_price + 3 * children_price = 39 →
  regular_price - children_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_difference_l84_8437


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l84_8466

def euler_family_ages : List ℕ := [8, 8, 12, 12, 10, 14]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l84_8466


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l84_8476

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [2, 4, 3]
def den1 : List Nat := [1, 3]
def num2 : List Nat := [2, 0, 4]
def den2 : List Nat := [2, 3]

-- Convert the numbers to base 10
def num1_base10 : Nat := to_base_10 num1 8
def den1_base10 : Nat := to_base_10 den1 4
def num2_base10 : Nat := to_base_10 num2 7
def den2_base10 : Nat := to_base_10 den2 5

-- Define the theorem
theorem base_conversion_theorem :
  (num1_base10 : ℚ) / den1_base10 + (num2_base10 : ℚ) / den2_base10 = 31 + 51 / 91 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l84_8476


namespace NUMINAMATH_CALUDE_restaurant_bill_l84_8460

theorem restaurant_bill (num_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  num_friends = 8 →
  extra_payment = 5/2 →
  (num_friends - 1) * (total_bill / num_friends + extra_payment) = total_bill →
  total_bill = 140 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_l84_8460


namespace NUMINAMATH_CALUDE_inequality_solution_set_l84_8474

theorem inequality_solution_set : 
  ∀ x : ℝ, abs (x - 4) + abs (3 - x) < 2 ↔ 2.5 < x ∧ x < 4.5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l84_8474


namespace NUMINAMATH_CALUDE_expression_simplification_l84_8423

theorem expression_simplification (m : ℝ) : 
  ((7*m + 3) - 3*m*2)*4 + (5 - 2/4)*(8*m - 12) = 40*m - 42 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l84_8423


namespace NUMINAMATH_CALUDE_subtract_preserves_inequality_l84_8458

theorem subtract_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_preserves_inequality_l84_8458
