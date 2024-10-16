import Mathlib

namespace NUMINAMATH_CALUDE_expression_equality_l2442_244250

theorem expression_equality : 
  4 * (Real.sin (π / 3)) + (1 / 2)⁻¹ - Real.sqrt 12 + |(-3)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2442_244250


namespace NUMINAMATH_CALUDE_equal_roots_equation_l2442_244299

theorem equal_roots_equation : ∃ x : ℝ, (x - 1) * (x - 1) = 0 ∧ 
  (∀ y : ℝ, (y - 1) * (y - 1) = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_equation_l2442_244299


namespace NUMINAMATH_CALUDE_circles_intersect_l2442_244253

theorem circles_intersect : 
  let c1 : ℝ × ℝ := (-2, 0)
  let r1 : ℝ := 2
  let c2 : ℝ × ℝ := (2, 1)
  let r2 : ℝ := 3
  let d := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  (abs (r1 - r2) < d) ∧ (d < r1 + r2) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_l2442_244253


namespace NUMINAMATH_CALUDE_shaded_area_l2442_244272

/-- The area of the shaded region in a grid with given properties -/
theorem shaded_area (total_area : ℝ) (triangle_base : ℝ) (triangle_height : ℝ)
  (h1 : total_area = 38)
  (h2 : triangle_base = 12)
  (h3 : triangle_height = 4) :
  total_area - (1/2 * triangle_base * triangle_height) = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l2442_244272


namespace NUMINAMATH_CALUDE_min_value_abc_l2442_244258

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (a₀ + 3 * b₀) * (b₀ + 3 * c₀) * (a₀ * c₀ + 2) = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l2442_244258


namespace NUMINAMATH_CALUDE_intersection_sum_l2442_244233

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 1 = (y - 2)^2

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄)

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ,
  (parabola1 x₁ y₁ ∧ parabola2 x₁ y₁) ∧
  (parabola1 x₂ y₂ ∧ parabola2 x₂ y₂) ∧
  (parabola1 x₃ y₃ ∧ parabola2 x₃ y₃) ∧
  (parabola1 x₄ y₄ ∧ parabola2 x₄ y₄) ∧
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_sum_l2442_244233


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2442_244209

/-- Calculates the profit percentage for a shopkeeper who sold 30 articles at the cost price of 35 articles -/
theorem shopkeeper_profit_percentage :
  let articles_sold : ℕ := 30
  let cost_price_articles : ℕ := 35
  let profit_ratio : ℚ := (cost_price_articles - articles_sold) / articles_sold
  profit_ratio * 100 = 5 / 30 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l2442_244209


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l2442_244266

/-- Given a rectangular field with one side of 15 meters and an area of 120 square meters,
    the length of its diagonal is 17 meters. -/
theorem rectangular_field_diagonal (l w d : ℝ) : 
  l = 15 → 
  l * w = 120 → 
  d^2 = l^2 + w^2 → 
  d = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l2442_244266


namespace NUMINAMATH_CALUDE_percent_of_l_equal_to_75_percent_of_m_l2442_244246

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Define the theorem
theorem percent_of_l_equal_to_75_percent_of_m 
  (h1 : condition1 j k)
  (h2 : condition2 k l)
  (h3 : condition3 j m) :
  ∃ x : ℝ, x / 100 * l = 0.75 * m ∧ x = 175 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_l_equal_to_75_percent_of_m_l2442_244246


namespace NUMINAMATH_CALUDE_sheila_hourly_rate_l2442_244232

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculates the hourly rate given a work schedule --/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
  , tuesday_hours := 6
  , wednesday_hours := 8
  , thursday_hours := 6
  , friday_hours := 8
  , weekly_earnings := 432 }

theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 12 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_rate_l2442_244232


namespace NUMINAMATH_CALUDE_rectangleABCD_area_is_196_l2442_244210

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of rectangle ABCD formed by three identical rectangles -/
def rectangleABCD_area (small_rect : Rectangle) : ℝ :=
  (2 * small_rect.width) * small_rect.length

theorem rectangleABCD_area_is_196 (small_rect : Rectangle) 
  (h1 : small_rect.width = 7)
  (h2 : small_rect.length = 2 * small_rect.width) :
  rectangleABCD_area small_rect = 196 := by
  sorry

#eval rectangleABCD_area { width := 7, length := 14 }

end NUMINAMATH_CALUDE_rectangleABCD_area_is_196_l2442_244210


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l2442_244269

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 1

/-- The travel time of ferry P in hours -/
def time_P : ℝ := 3

/-- The travel time of ferry Q in hours -/
def time_Q : ℝ := time_P + 5

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := speed_Q * time_Q

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 1 ∧
  time_P = 3 ∧
  time_Q = time_P + 5 ∧
  distance_Q = 3 * distance_P :=
by sorry

end NUMINAMATH_CALUDE_ferry_speed_proof_l2442_244269


namespace NUMINAMATH_CALUDE_box_width_is_ten_inches_l2442_244263

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_width_is_ten_inches (box : Dimensions) (block : Dimensions) :
  box.height = 8 →
  box.length = 12 →
  block.height = 3 →
  block.width = 2 →
  block.length = 4 →
  volume box = 40 * volume block →
  box.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_ten_inches_l2442_244263


namespace NUMINAMATH_CALUDE_power_24_in_terms_of_a_and_t_l2442_244234

theorem power_24_in_terms_of_a_and_t (x a t : ℝ) 
  (h1 : 2^x = a) (h2 : 3^x = t) : 24^x = a^3 * t := by
  sorry

end NUMINAMATH_CALUDE_power_24_in_terms_of_a_and_t_l2442_244234


namespace NUMINAMATH_CALUDE_truck_driver_earnings_l2442_244222

/-- Calculates the net earnings of a truck driver given specific conditions --/
theorem truck_driver_earnings
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (driving_speed : ℝ)
  (payment_rate : ℝ)
  (driving_duration : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : driving_speed = 30)
  (h4 : payment_rate = 0.5)
  (h5 : driving_duration = 10)
  : ∃ (net_earnings : ℝ), net_earnings = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_truck_driver_earnings_l2442_244222


namespace NUMINAMATH_CALUDE_sum_and_powers_equality_l2442_244215

theorem sum_and_powers_equality : (3 + 7)^3 + (3^2 + 7^2 + 3^3 + 7^3) = 1428 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_powers_equality_l2442_244215


namespace NUMINAMATH_CALUDE_three_circles_equal_angle_points_l2442_244270

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate to check if two circles do not intersect and neither is contained within the other -/
def are_separate (c1 c2 : Circle) : Prop := sorry

/-- The locus of points from which two circles are seen at the same angle -/
def equal_angle_locus (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- The angle at which a circle is seen from a point -/
def viewing_angle (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

theorem three_circles_equal_angle_points 
  (k1 k2 k3 : Circle)
  (h12 : are_separate k1 k2)
  (h23 : are_separate k2 k3)
  (h13 : are_separate k1 k3) :
  ∃ p : ℝ × ℝ, 
    viewing_angle k1 p = viewing_angle k2 p ∧ 
    viewing_angle k2 p = viewing_angle k3 p ∧
    p ∈ (equal_angle_locus k1 k2) ∩ (equal_angle_locus k2 k3) := by
  sorry

end NUMINAMATH_CALUDE_three_circles_equal_angle_points_l2442_244270


namespace NUMINAMATH_CALUDE_square_equality_l2442_244221

theorem square_equality (n : ℕ) : (n + 3)^2 = 3*(n + 2)^2 - 3*(n + 1)^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l2442_244221


namespace NUMINAMATH_CALUDE_exam_average_l2442_244265

theorem exam_average (students_group1 : ℕ) (average_group1 : ℚ) 
                      (students_group2 : ℕ) (average_group2 : ℚ) : 
  students_group1 = 15 →
  average_group1 = 73 / 100 →
  students_group2 = 10 →
  average_group2 = 88 / 100 →
  let total_students := students_group1 + students_group2
  let total_score := students_group1 * average_group1 + students_group2 * average_group2
  let overall_average := total_score / total_students
  overall_average = 79 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l2442_244265


namespace NUMINAMATH_CALUDE_unit_digit_of_3_to_2022_l2442_244277

def unit_digit (n : ℕ) : ℕ := n % 10

def power_of_3_unit_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | 0 => 1
  | _ => 0  -- This case should never occur

theorem unit_digit_of_3_to_2022 :
  unit_digit (3^2022) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_to_2022_l2442_244277


namespace NUMINAMATH_CALUDE_number_of_students_l2442_244257

theorem number_of_students (x : ℕ) 
  (h1 : 3600 = (3600 / x) * x)  -- Retail price for x tools
  (h2 : 3600 = (3600 / (x + 60)) * (x + 60))  -- Wholesale price for x + 60 tools
  (h3 : (3600 / x) * 50 = (3600 / (x + 60)) * 60)  -- Cost equality condition
  : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l2442_244257


namespace NUMINAMATH_CALUDE_composite_sum_of_product_equal_l2442_244294

theorem composite_sum_of_product_equal (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^1984 + b^1984 + c^1984 + d^1984 = m * n :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_product_equal_l2442_244294


namespace NUMINAMATH_CALUDE_congruence_solution_l2442_244261

theorem congruence_solution (x : ℤ) : 
  (∃ (a m : ℤ), m ≥ 2 ∧ 0 ≤ a ∧ a < m ∧ x ≡ a [ZMOD m]) →
  ((10 * x + 3) ≡ 6 [ZMOD 15] ↔ x ≡ 0 [ZMOD 3]) := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2442_244261


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_l2442_244220

/-- A quadratic function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem quadratic_monotone_increasing (a : ℝ) (h : a > 1) :
  ∀ x > 1, Monotone (fun x => f a x) := by sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_l2442_244220


namespace NUMINAMATH_CALUDE_trailingZeros_100_factorial_l2442_244228

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailingZeros_100_factorial_l2442_244228


namespace NUMINAMATH_CALUDE_elisa_math_books_l2442_244240

theorem elisa_math_books :
  ∀ (total math lit : ℕ),
  total < 100 →
  total = 24 + math + lit →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 := by
sorry

end NUMINAMATH_CALUDE_elisa_math_books_l2442_244240


namespace NUMINAMATH_CALUDE_derivative_at_zero_implies_k_value_l2442_244297

def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 + k^3 * x

theorem derivative_at_zero_implies_k_value (k : ℝ) :
  (deriv (f k)) 0 = 27 → k = 3 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_zero_implies_k_value_l2442_244297


namespace NUMINAMATH_CALUDE_pension_program_participation_rate_l2442_244287

structure Shift where
  members : ℕ
  participation_rate : ℚ

def company_x : List Shift := [
  { members := 60, participation_rate := 1/5 },
  { members := 50, participation_rate := 2/5 },
  { members := 40, participation_rate := 1/10 }
]

theorem pension_program_participation_rate :
  let total_workers := (company_x.map (λ s => s.members)).sum
  let participating_workers := (company_x.map (λ s => (s.members : ℚ) * s.participation_rate)).sum
  participating_workers / total_workers = 6/25 := by
sorry

end NUMINAMATH_CALUDE_pension_program_participation_rate_l2442_244287


namespace NUMINAMATH_CALUDE_range_of_a_in_second_quadrant_l2442_244231

/-- A complex number z = x + yi is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem range_of_a_in_second_quadrant (a : ℝ) :
  is_in_second_quadrant ((a - 2) + (a + 1) * I) ↔ -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_in_second_quadrant_l2442_244231


namespace NUMINAMATH_CALUDE_conic_sections_properties_l2442_244283

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1

-- Define a parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

theorem conic_sections_properties :
  -- Proposition 2
  (∃ e₁ e₂ : ℝ, quadratic_equation e₁ ∧ quadratic_equation e₂ ∧ 
   0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1) ∧
  -- Proposition 3
  (∃ c : ℝ, c^2 = 34 ∧ 
   (∀ x y : ℝ, hyperbola x y ↔ (x - c)^2 / 25 + y^2 / 9 = 1) ∧
   (∀ x y : ℝ, ellipse x y ↔ (x - c)^2 / 35 + y^2 = 1)) ∧
  -- Proposition 4
  (∀ p a b : ℝ, parabola a b p →
   ∃ r : ℝ, r > 0 ∧
   (∀ x y : ℝ, (x - (a + r))^2 + (y - b)^2 = r^2 →
    x = p ∨ (x = p ∧ y = b))) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_properties_l2442_244283


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2442_244225

theorem solution_set_inequality (x : ℝ) :
  (x^2 - |x| > 0) ↔ (x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2442_244225


namespace NUMINAMATH_CALUDE_stellas_antique_shop_profit_l2442_244229

/-- Calculates the profit for Stella's antique shop given the inventory and prices --/
theorem stellas_antique_shop_profit :
  let dolls : ℕ := 6
  let clocks : ℕ := 4
  let glasses : ℕ := 8
  let vases : ℕ := 3
  let postcards : ℕ := 10
  let doll_price : ℕ := 8
  let clock_price : ℕ := 25
  let glass_price : ℕ := 6
  let vase_price : ℕ := 12
  let postcard_price : ℕ := 3
  let purchase_cost : ℕ := 250
  let revenue := dolls * doll_price + clocks * clock_price + glasses * glass_price + 
                 vases * vase_price + postcards * postcard_price
  let profit := revenue - purchase_cost
  profit = 12 := by sorry

end NUMINAMATH_CALUDE_stellas_antique_shop_profit_l2442_244229


namespace NUMINAMATH_CALUDE_altitude_equals_harmonic_mean_of_excircle_radii_l2442_244213

/-- For a triangle ABC with altitude h_a from vertex A, area t, semiperimeter s,
    and excircle radii r_b and r_c, the altitude h_a is equal to 2t/a. -/
theorem altitude_equals_harmonic_mean_of_excircle_radii 
  (a b c : ℝ) 
  (h_a : ℝ) 
  (t : ℝ) 
  (s : ℝ) 
  (r_b r_c : ℝ) 
  (h_s : s = (a + b + c) / 2) 
  (h_r_b : r_b = t / (s - b)) 
  (h_r_c : r_c = t / (s - c)) 
  (h_positive : a > 0 ∧ t > 0) : 
  h_a = 2 * t / a := by
  sorry

end NUMINAMATH_CALUDE_altitude_equals_harmonic_mean_of_excircle_radii_l2442_244213


namespace NUMINAMATH_CALUDE_female_officers_count_l2442_244298

theorem female_officers_count (total_on_duty : ℕ) (female_percentage : ℚ) : 
  total_on_duty = 152 →
  female_percentage = 19 / 100 →
  (total_on_duty / 2 : ℚ) = female_percentage * 400 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2442_244298


namespace NUMINAMATH_CALUDE_unique_solution_l2442_244204

def machine_step (N : ℕ) : ℕ :=
  if N % 2 = 1 then 5 * N + 3
  else if N % 3 = 0 then N / 3
  else N + 1

def machine_process (N : ℕ) : ℕ :=
  (machine_step ∘ machine_step ∘ machine_step ∘ machine_step ∘ machine_step) N

theorem unique_solution :
  ∀ N : ℕ, N > 0 → (machine_process N = 1 ↔ N = 6) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2442_244204


namespace NUMINAMATH_CALUDE_equalizeTable_l2442_244235

-- Define the table as a matrix
def Table (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

-- Initial configuration of the table
def initialTable (n : ℕ) : Table n :=
  Matrix.diagonal (λ _ => 1)

-- Define a rook path as a list of positions
def RookPath (n : ℕ) := List (Fin n × Fin n)

-- Predicate to check if a path is valid (closed and non-self-intersecting)
def isValidPath (n : ℕ) (path : RookPath n) : Prop := sorry

-- Function to apply a rook transformation
def applyRookTransformation (t : Table n) (path : RookPath n) : Table n := sorry

-- Predicate to check if all numbers in the table are equal
def allEqual (t : Table n) : Prop := sorry

-- The main theorem
theorem equalizeTable (n : ℕ) :
  (∃ (transformations : List (RookPath n)), 
    allEqual (transformations.foldl applyRookTransformation (initialTable n))) ↔ 
  Odd n := by sorry

end NUMINAMATH_CALUDE_equalizeTable_l2442_244235


namespace NUMINAMATH_CALUDE_largest_of_three_negatives_l2442_244278

theorem largest_of_three_negatives (a b c : ℝ) 
  (neg_a : a < 0) (neg_b : b < 0) (neg_c : c < 0)
  (h : c / (a + b) < a / (b + c) ∧ a / (b + c) < b / (c + a)) :
  c > a ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_largest_of_three_negatives_l2442_244278


namespace NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l2442_244288

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there exists one whose sum of digits is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_exists_sum_of_digits_div_11_l2442_244288


namespace NUMINAMATH_CALUDE_total_fireworks_is_1188_l2442_244219

/-- Calculates the total number of fireworks used in the New Year's Eve display -/
def total_fireworks : ℕ :=
  let fireworks_per_number : ℕ := 6
  let fireworks_per_regular_letter : ℕ := 5
  let fireworks_for_H : ℕ := 8
  let fireworks_for_E : ℕ := 7
  let fireworks_for_L : ℕ := 6
  let fireworks_for_O : ℕ := 9
  let fireworks_for_square : ℕ := 4
  let fireworks_for_triangle : ℕ := 3
  let fireworks_for_circle : ℕ := 12
  let additional_boxes : ℕ := 100
  let fireworks_per_box : ℕ := 10

  let years_fireworks := fireworks_per_number * 4 * 3
  let happy_new_year_fireworks := fireworks_per_regular_letter * 11 + fireworks_per_number
  let geometric_shapes_fireworks := fireworks_for_square + fireworks_for_triangle + fireworks_for_circle
  let hello_fireworks := fireworks_for_H + fireworks_for_E + fireworks_for_L * 2 + fireworks_for_O
  let additional_fireworks := additional_boxes * fireworks_per_box

  years_fireworks + happy_new_year_fireworks + geometric_shapes_fireworks + hello_fireworks + additional_fireworks

theorem total_fireworks_is_1188 : total_fireworks = 1188 := by
  sorry

end NUMINAMATH_CALUDE_total_fireworks_is_1188_l2442_244219


namespace NUMINAMATH_CALUDE_library_visitors_l2442_244255

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) 
  (h1 : sunday_avg = 510)
  (h2 : total_days = 30)
  (h3 : month_avg = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let other_days : ℕ := total_days - sundays
  let other_days_avg : ℕ := (month_avg * total_days - sunday_avg * sundays) / other_days
  other_days_avg = 240 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l2442_244255


namespace NUMINAMATH_CALUDE_stating_one_empty_neighborhood_probability_l2442_244289

/-- The number of neighborhoods --/
def num_neighborhoods : ℕ := 3

/-- The number of staff members --/
def num_staff : ℕ := 4

/-- The probability of exactly one neighborhood not being assigned any staff members --/
def probability_one_empty : ℚ := 14/27

/-- 
Theorem stating that the probability of exactly one neighborhood out of three 
not being assigned any staff members, when four staff members are independently 
assigned to the neighborhoods, is 14/27.
--/
theorem one_empty_neighborhood_probability : 
  (num_neighborhoods = 3 ∧ num_staff = 4) → 
  probability_one_empty = 14/27 := by
  sorry

end NUMINAMATH_CALUDE_stating_one_empty_neighborhood_probability_l2442_244289


namespace NUMINAMATH_CALUDE_simplify_powers_l2442_244271

theorem simplify_powers (x : ℝ) : x^5 * x^3 * 2 = 2 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_l2442_244271


namespace NUMINAMATH_CALUDE_divisible_by_101_l2442_244245

def repeat_two_digit (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

theorem divisible_by_101 (n : ℕ) (h : n < 100) :
  101 ∣ repeat_two_digit n :=
sorry

end NUMINAMATH_CALUDE_divisible_by_101_l2442_244245


namespace NUMINAMATH_CALUDE_cylinder_side_diagonal_l2442_244251

theorem cylinder_side_diagonal (h l d : ℝ) (h_height : h = 16) (h_length : l = 12) : 
  d = 20 → d^2 = h^2 + l^2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_side_diagonal_l2442_244251


namespace NUMINAMATH_CALUDE_negation_at_most_four_l2442_244226

-- Define "at most four" for natural numbers
def at_most_four (n : ℕ) : Prop := n ≤ 4

-- Define "at least five" for natural numbers
def at_least_five (n : ℕ) : Prop := n ≥ 5

-- Theorem stating that the negation of "at most four" is equivalent to "at least five"
theorem negation_at_most_four (n : ℕ) : ¬(at_most_four n) ↔ at_least_five n := by
  sorry

end NUMINAMATH_CALUDE_negation_at_most_four_l2442_244226


namespace NUMINAMATH_CALUDE_original_square_side_length_l2442_244236

def is_valid_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ (n + k)^2 - n^2 = 47

theorem original_square_side_length :
  ∃! (n : ℕ), is_valid_square n ∧ n > 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_original_square_side_length_l2442_244236


namespace NUMINAMATH_CALUDE_d_properties_l2442_244279

/-- Given a nonnegative integer c, define sequences a_n and d_n -/
def a (c n : ℕ) : ℕ := n^2 + c

def d (c n : ℕ) : ℕ := Nat.gcd (a c n) (a c (n + 1))

/-- Theorem stating the properties of d_n for different values of c -/
theorem d_properties (c : ℕ) :
  (∀ n : ℕ, n ≥ 1 → c = 0 → d c n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → c = 1 → d c n = 1 ∨ d c n = 5) ∧
  (∀ n : ℕ, n ≥ 1 → d c n ≤ 4 * c + 1) :=
sorry

end NUMINAMATH_CALUDE_d_properties_l2442_244279


namespace NUMINAMATH_CALUDE_computer_sales_ratio_l2442_244212

theorem computer_sales_ratio : 
  ∀ (total netbooks desktops laptops : ℕ),
  total = 72 →
  netbooks = total / 3 →
  desktops = 12 →
  laptops = total - netbooks - desktops →
  (laptops : ℚ) / total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_computer_sales_ratio_l2442_244212


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l2442_244264

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 3 * n = 2007 := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l2442_244264


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l2442_244259

theorem reciprocal_sum_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l2442_244259


namespace NUMINAMATH_CALUDE_divisibility_reversal_implies_factor_of_99_l2442_244238

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_reversal_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  k ∣ 99 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_reversal_implies_factor_of_99_l2442_244238


namespace NUMINAMATH_CALUDE_max_third_term_is_16_l2442_244214

/-- An arithmetic sequence of four positive integers with sum 50 -/
structure ArithmeticSequence where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference
  sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value of the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 ∧ ∃ seq : ArithmeticSequence, third_term seq = 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_is_16_l2442_244214


namespace NUMINAMATH_CALUDE_one_acute_triangle_in_1997_gon_l2442_244216

/-- A convex regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A decomposition of a polygon into triangles using non-intersecting diagonals -/
structure TriangularDecomposition (n : ℕ) where
  (polygon : RegularPolygon n)

/-- An acute triangle -/
structure AcuteTriangle

/-- The number of acute triangles in a triangular decomposition of a regular polygon -/
def num_acute_triangles (n : ℕ) (decomp : TriangularDecomposition n) : ℕ :=
  sorry

/-- The main theorem: In a regular 1997-gon, there is exactly one acute triangle
    in its triangular decomposition -/
theorem one_acute_triangle_in_1997_gon :
  ∀ (decomp : TriangularDecomposition 1997),
    num_acute_triangles 1997 decomp = 1 :=
  sorry

end NUMINAMATH_CALUDE_one_acute_triangle_in_1997_gon_l2442_244216


namespace NUMINAMATH_CALUDE_triangle_properties_l2442_244296

noncomputable section

-- Define the triangle
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Theorem statement
theorem triangle_properties {A B C a b c : ℝ} (h : triangle A B C a b c) :
  a = 2 * Real.sqrt 3 ∧ 
  Real.cos (2 * A + π / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2442_244296


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2442_244237

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2442_244237


namespace NUMINAMATH_CALUDE_convention_handshakes_specific_l2442_244284

/-- Calculates the number of handshakes in a convention with representatives from multiple companies. -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Proves that in a convention with 5 representatives from each of 5 companies, 
    where representatives only shake hands with people from other companies, 
    the total number of handshakes is 250. -/
theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_specific_l2442_244284


namespace NUMINAMATH_CALUDE_equation_solutions_l2442_244276

-- Define the equation
def equation (x : ℝ) : Prop := (x ^ (1/4 : ℝ)) = 16 / (9 - (x ^ (1/4 : ℝ)))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 4096) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2442_244276


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2442_244243

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < a)) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2442_244243


namespace NUMINAMATH_CALUDE_fraction_sum_between_extremes_l2442_244206

theorem fraction_sum_between_extremes 
  (a b c d n p x y : ℚ) 
  (h_pos : b > 0 ∧ d > 0 ∧ p > 0 ∧ y > 0)
  (h_order : a/b > c/d ∧ c/d > n/p ∧ n/p > x/y) : 
  x/y < (a + c + n + x) / (b + d + p + y) ∧ 
  (a + c + n + x) / (b + d + p + y) < a/b := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_between_extremes_l2442_244206


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2442_244275

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (((a > 2 ∧ b > 2) → (a + b > 4)) ∧ 
   (∃ x y : ℝ, x + y > 4 ∧ ¬(x > 2 ∧ y > 2))) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2442_244275


namespace NUMINAMATH_CALUDE_number_problem_l2442_244201

theorem number_problem (x : ℝ) : (1 / 5 * x - 5 = 5) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2442_244201


namespace NUMINAMATH_CALUDE_inequality_integer_solutions_l2442_244268

theorem inequality_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, (x - 1 : ℚ) / 3 < 5 / 7 ∧ 5 / 7 < (x + 4 : ℚ) / 5) ∧
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_integer_solutions_l2442_244268


namespace NUMINAMATH_CALUDE_car_speed_problem_l2442_244205

/-- Proves that given the conditions, car R's average speed is 50 miles per hour -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 800 →
  time_diff = 2 →
  speed_diff = 10 →
  ∃ (speed_R : ℝ),
    distance / speed_R - time_diff = distance / (speed_R + speed_diff) ∧
    speed_R = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2442_244205


namespace NUMINAMATH_CALUDE_difference_h_f_l2442_244224

theorem difference_h_f (e f g h : ℕ+) 
  (he : e^5 = f^4)
  (hg : g^3 = h^2)
  (hge : g - e = 31) : 
  h - f = 971 := by sorry

end NUMINAMATH_CALUDE_difference_h_f_l2442_244224


namespace NUMINAMATH_CALUDE_toms_brick_cost_l2442_244242

/-- The total cost of bricks for Tom's shed -/
def total_cost (total_bricks : ℕ) (full_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let half_bricks := total_bricks / 2
  let discounted_price := full_price * (1 - discount_percent)
  (half_bricks : ℚ) * discounted_price + (half_bricks : ℚ) * full_price

/-- Theorem stating the total cost for Tom's bricks -/
theorem toms_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end NUMINAMATH_CALUDE_toms_brick_cost_l2442_244242


namespace NUMINAMATH_CALUDE_lorenzo_thumbtacks_l2442_244218

/-- The number of cans of thumbtacks Lorenzo had -/
def number_of_cans : ℕ := sorry

/-- The number of boards Lorenzo tested -/
def boards_tested : ℕ := 120

/-- The number of tacks remaining in each can at the end of the day -/
def tacks_remaining : ℕ := 30

/-- The total combined number of thumbtacks from the full cans -/
def total_thumbtacks : ℕ := 450

theorem lorenzo_thumbtacks :
  (number_of_cans * (boards_tested + tacks_remaining) = total_thumbtacks) →
  number_of_cans = 3 := by sorry

end NUMINAMATH_CALUDE_lorenzo_thumbtacks_l2442_244218


namespace NUMINAMATH_CALUDE_no_2016_subsequence_l2442_244280

-- Define the sequence
def seq : ℕ → ℕ
  | 0 => 2
  | 1 => 0
  | 2 => 1
  | 3 => 7
  | 4 => 0
  | n + 5 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

-- Define a function to check if a subsequence appears at a given position
def subsequenceAt (start : ℕ) (subseq : List ℕ) : Prop :=
  ∀ i, i < subseq.length → seq (start + i) = subseq.get ⟨i, by sorry⟩

-- Theorem statement
theorem no_2016_subsequence :
  ¬ ∃ start : ℕ, start ≥ 4 ∧ subsequenceAt start [2, 0, 1, 6] :=
by sorry

end NUMINAMATH_CALUDE_no_2016_subsequence_l2442_244280


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l2442_244290

theorem geometric_arithmetic_progression_sum : ∃ x y : ℝ, 
  (5 < x ∧ x < y ∧ y < 12) ∧ 
  (∃ r : ℝ, r > 1 ∧ x = 5 * r ∧ y = 5 * r^2) ∧
  (∃ d : ℝ, d > 0 ∧ y = x + d ∧ 12 = y + d) ∧
  (abs (x + y - 16.2788) < 0.0001) := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l2442_244290


namespace NUMINAMATH_CALUDE_first_month_bill_l2442_244203

/-- Represents the telephone bill structure -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  totalCharge_eq : totalCharge = callCharge + internetCharge

/-- The telephone bill problem -/
theorem first_month_bill (
  firstMonth secondMonth : TelephoneBill
) (h1 : firstMonth.totalCharge = 46)
  (h2 : secondMonth.totalCharge = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.totalCharge = 46 := by
sorry

end NUMINAMATH_CALUDE_first_month_bill_l2442_244203


namespace NUMINAMATH_CALUDE_largest_visible_sum_l2442_244286

/-- Represents a standard die with opposite faces summing to 7 -/
structure Die where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents a 3x3x3 cube assembled from 27 dice -/
structure Cube where
  dice : Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible values on the 6 faces of the cube -/
def visibleSum (c : Cube) : Nat :=
  sorry

/-- States that the largest possible sum of visible values is 288 -/
theorem largest_visible_sum (c : Cube) : 
  visibleSum c ≤ 288 ∧ ∃ c' : Cube, visibleSum c' = 288 :=
sorry

end NUMINAMATH_CALUDE_largest_visible_sum_l2442_244286


namespace NUMINAMATH_CALUDE_red_on_third_prob_l2442_244293

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a specific outcome on the RedDie -/
def roll_prob (d : RedDie) (is_red : Bool) : ℚ :=
  if is_red then d.red_sides / d.sides else (d.sides - d.red_sides) / d.sides

/-- The probability of the die landing with a red side up for the first time on the third roll -/
def red_on_third (d : RedDie) : ℚ :=
  (roll_prob d false) * (roll_prob d false) * (roll_prob d true)

theorem red_on_third_prob (d : RedDie) : 
  red_on_third d = 147 / 1000 := by sorry

end NUMINAMATH_CALUDE_red_on_third_prob_l2442_244293


namespace NUMINAMATH_CALUDE_ten_people_seating_arrangement_l2442_244241

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange 3 people in a block where one person is fixed between the other two -/
def fixedBlockArrangements : ℕ := 2

theorem ten_people_seating_arrangement :
  roundTableArrangements 9 * fixedBlockArrangements = 80640 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_seating_arrangement_l2442_244241


namespace NUMINAMATH_CALUDE_pen_cost_is_four_l2442_244252

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 2 * pencil_cost

/-- The total cost of a pen and pencil in dollars -/
def total_cost : ℝ := 6

theorem pen_cost_is_four :
  pen_cost = 4 ∧ pencil_cost + pen_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pen_cost_is_four_l2442_244252


namespace NUMINAMATH_CALUDE_sum_transformed_sequence_formula_l2442_244202

/-- Given a sequence {aₙ} where the sum of its first n terms Sₙ satisfies 3Sₙ = 4^(n+1) - 4,
    this function computes the sum of the first n terms of the sequence {(3n-2)aₙ}. -/
def sumTransformedSequence (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) : ℝ :=
  4 + (n - 1 : ℝ) * 4^(n+1)

/-- Theorem stating that the sum of the first n terms of {(3n-2)aₙ} is 4 + (n-1) * 4^(n+1),
    given that the sum of the first n terms of {aₙ} satisfies 3Sₙ = 4^(n+1) - 4. -/
theorem sum_transformed_sequence_formula (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) :
  sumTransformedSequence n S h = 4 + (n - 1 : ℝ) * 4^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_sum_transformed_sequence_formula_l2442_244202


namespace NUMINAMATH_CALUDE_parametric_equations_represent_line_l2442_244281

/-- Proves that the given parametric equations represent the straight line 2x - y + 1 = 0 -/
theorem parametric_equations_represent_line :
  ∀ (t : ℝ), 2 * (1 - t) - (3 - 2*t) + 1 = 0 := by
  sorry

#check parametric_equations_represent_line

end NUMINAMATH_CALUDE_parametric_equations_represent_line_l2442_244281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2442_244285

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_sum : a 7 + a 8 = 28) :
  a 4 = 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2442_244285


namespace NUMINAMATH_CALUDE_ln_third_derivative_value_l2442_244211

open Real

theorem ln_third_derivative_value (x₀ : ℝ) (h : x₀ > 0) : 
  let f : ℝ → ℝ := λ x => log x
  (deriv^[3] f) x₀ = 1 / x₀^2 → x₀ = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ln_third_derivative_value_l2442_244211


namespace NUMINAMATH_CALUDE_one_fourth_more_than_32_5_l2442_244249

theorem one_fourth_more_than_32_5 : (1 / 4 : ℚ) + 32.5 = 32.75 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_more_than_32_5_l2442_244249


namespace NUMINAMATH_CALUDE_equipment_production_calculation_l2442_244247

/-- Given a total production and a sample with known quantities from two equipment sets,
    calculate the total production of the second equipment set. -/
theorem equipment_production_calculation
  (total_production : ℕ)
  (sample_size : ℕ)
  (sample_A : ℕ)
  (h_total : total_production = 4800)
  (h_sample : sample_size = 80)
  (h_sample_A : sample_A = 50)
  : (total_production * (sample_size - sample_A)) / sample_size = 1800 :=
by sorry

end NUMINAMATH_CALUDE_equipment_production_calculation_l2442_244247


namespace NUMINAMATH_CALUDE_polynomial_square_l2442_244217

theorem polynomial_square (a b : ℚ) : 
  (∃ q₀ q₁ : ℚ, ∀ x, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + q₁*x + q₀)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l2442_244217


namespace NUMINAMATH_CALUDE_direct_proportion_exponent_l2442_244262

theorem direct_proportion_exponent (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, -2 * x^(m-2) = k * x) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_exponent_l2442_244262


namespace NUMINAMATH_CALUDE_inequality_proofs_l2442_244267

theorem inequality_proofs (a b : ℝ) (h : a + b > 0) : 
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l2442_244267


namespace NUMINAMATH_CALUDE_range_of_a_l2442_244248

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on [1,5]
def IsIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 5 → y ∈ Set.Icc 1 5 → x < y → f x < f y

-- Define the theorem
theorem range_of_a (h1 : IsIncreasingOn f) 
  (h2 : ∀ a, f (a + 1) < f (2 * a - 1)) :
  ∃ a, a ∈ Set.Ioo 2 3 ∧ 
    (∀ x, x ∈ Set.Ioo 2 3 → 
      (f (x + 1) < f (2 * x - 1) ∧ 
       x + 1 ∈ Set.Icc 1 5 ∧ 
       2 * x - 1 ∈ Set.Icc 1 5)) :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l2442_244248


namespace NUMINAMATH_CALUDE_sin_negative_225_degrees_l2442_244244

theorem sin_negative_225_degrees :
  Real.sin (-(225 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_225_degrees_l2442_244244


namespace NUMINAMATH_CALUDE_special_function_value_l2442_244208

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  f 0 = 1008 ∧
  (∀ x : ℝ, f (x + 4) - f x ≤ 2 * (x + 1)) ∧
  (∀ x : ℝ, f (x + 12) - f x ≥ 6 * (x + 5))

/-- The main theorem -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 2016 / 2016 = 504 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2442_244208


namespace NUMINAMATH_CALUDE_equation_solution_l2442_244273

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : (7*x - 4) / (x - 2) = 5 / (x - 2)) : x = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2442_244273


namespace NUMINAMATH_CALUDE_inequality_holds_iff_x_leq_3_l2442_244282

theorem inequality_holds_iff_x_leq_3 (x : ℕ+) :
  (x + 1 : ℚ) / 3 - (2 * x - 1 : ℚ) / 4 ≥ (x - 3 : ℚ) / 6 ↔ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_x_leq_3_l2442_244282


namespace NUMINAMATH_CALUDE_isabel_paper_count_l2442_244256

/-- The number of pieces of paper Isabel used -/
def used : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def left : ℕ := 744

/-- The initial number of pieces of paper Isabel bought -/
def initial : ℕ := used + left

theorem isabel_paper_count : initial = 900 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_count_l2442_244256


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2442_244223

def A : Set ℝ := {x | x + 1 ≤ 0 ∨ x - 4 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

theorem intersection_nonempty (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a ≤ -1/2 ∨ a = 2 := by sorry

theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2442_244223


namespace NUMINAMATH_CALUDE_car_trade_profit_l2442_244274

theorem car_trade_profit (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let buying_price := original_price * (1 - 0.05)
  let selling_price := buying_price * (1 + 0.60)
  let profit := selling_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 52 := by sorry

end NUMINAMATH_CALUDE_car_trade_profit_l2442_244274


namespace NUMINAMATH_CALUDE_roller_coaster_problem_l2442_244260

/-- The number of times a roller coaster must run to accommodate all people in line -/
def roller_coaster_runs (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

/-- Theorem stating that for 84 people in line, 7 cars, and 2 people per car, 6 runs are needed -/
theorem roller_coaster_problem : roller_coaster_runs 84 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_problem_l2442_244260


namespace NUMINAMATH_CALUDE_machines_count_l2442_244239

/-- The number of machines that complete a job lot in 6 hours -/
def N : ℕ := 8

/-- The time taken by N machines to complete the job lot -/
def time_N : ℕ := 6

/-- The number of machines in the second scenario -/
def machines_2 : ℕ := 4

/-- The time taken by machines_2 to complete the job lot -/
def time_2 : ℕ := 12

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / 48

theorem machines_count :
  N * work_rate * time_N = 1 ∧
  machines_2 * work_rate * time_2 = 1 :=
sorry

#check machines_count

end NUMINAMATH_CALUDE_machines_count_l2442_244239


namespace NUMINAMATH_CALUDE_first_number_proof_l2442_244254

theorem first_number_proof : ∃ x : ℝ, x + 2.017 + 0.217 + 2.0017 = 221.2357 ∧ x = 217 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l2442_244254


namespace NUMINAMATH_CALUDE_daily_production_l2442_244207

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 5

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 12000

/-- The total number of bottles produced per day -/
def total_bottles : ℕ := bottles_per_case * cases_per_day

theorem daily_production :
  total_bottles = 60000 :=
by sorry

end NUMINAMATH_CALUDE_daily_production_l2442_244207


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2442_244230

theorem triangle_side_sum (side_length : ℚ) (h : side_length = 14/8) : 
  3 * side_length = 21/4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2442_244230


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2442_244292

theorem cos_2alpha_value (α : Real) 
  (h1 : 2 * Real.cos (2 * α) = Real.sin (α - π/4))
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.cos (2 * α) = Real.sqrt 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2442_244292


namespace NUMINAMATH_CALUDE_family_income_problem_l2442_244291

theorem family_income_problem (initial_members : ℕ) (deceased_income new_average : ℚ) 
  (h1 : initial_members = 4)
  (h2 : deceased_income = 1170)
  (h3 : new_average = 590) :
  let initial_average := (initial_members * new_average + deceased_income) / initial_members
  initial_average = 735 := by
sorry

end NUMINAMATH_CALUDE_family_income_problem_l2442_244291


namespace NUMINAMATH_CALUDE_compound_weight_proof_l2442_244227

/-- Molar mass of Nitrogen in g/mol -/
def N_mass : ℝ := 14.01

/-- Molar mass of Hydrogen in g/mol -/
def H_mass : ℝ := 1.01

/-- Molar mass of Iodine in g/mol -/
def I_mass : ℝ := 126.90

/-- Molar mass of Oxygen in g/mol -/
def O_mass : ℝ := 16.00

/-- Molar mass of NH4I in g/mol -/
def NH4I_mass : ℝ := N_mass + 4 * H_mass + I_mass

/-- Molar mass of H2O in g/mol -/
def H2O_mass : ℝ := 2 * H_mass + O_mass

/-- Number of moles of NH4I -/
def NH4I_moles : ℝ := 15

/-- Number of moles of H2O -/
def H2O_moles : ℝ := 7

/-- Total weight of the compound (NH4I·H2O) in grams -/
def total_weight : ℝ := NH4I_moles * NH4I_mass + H2O_moles * H2O_mass

theorem compound_weight_proof : total_weight = 2300.39 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l2442_244227


namespace NUMINAMATH_CALUDE_division_reciprocal_l2442_244295

theorem division_reciprocal (a b c d e : ℝ) (ha : a ≠ 0) (hbcde : b - c + d - e ≠ 0) :
  a / (b - c + d - e) = 1 / ((b - c + d - e) / a) := by
  sorry

end NUMINAMATH_CALUDE_division_reciprocal_l2442_244295


namespace NUMINAMATH_CALUDE_f_extrema_l2442_244200

/-- A cubic function f(x) = x³ - px² - qx that is tangent to the x-axis at (1,0) -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^3 - p*x^2 - q*x

/-- The condition that f(x) is tangent to the x-axis at (1,0) -/
def is_tangent (p q : ℝ) : Prop :=
  f p q 1 = 0 ∧ (p + q = 1) ∧ (p^2 + 4*q = 0)

theorem f_extrema (p q : ℝ) (h : is_tangent p q) :
  (∃ x, f p q x = 4/27) ∧ (∀ x, f p q x ≥ 0) ∧ (∃ x, f p q x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_l2442_244200
