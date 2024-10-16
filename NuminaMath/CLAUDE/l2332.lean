import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_l2332_233298

theorem square_difference (n : ℝ) : 
  let m : ℝ := 4 * n + 3
  m^2 - 8 * m * n + 16 * n^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2332_233298


namespace NUMINAMATH_CALUDE_kids_in_movie_l2332_233206

theorem kids_in_movie (riverside_total : ℕ) (westside_total : ℕ) (mountaintop_total : ℕ)
  (riverside_denied_percent : ℚ) (westside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : westside_total = 90)
  (h3 : mountaintop_total = 50)
  (h4 : riverside_denied_percent = 20/100)
  (h5 : westside_denied_percent = 70/100)
  (h6 : mountaintop_denied_percent = 1/2) :
  ↑riverside_total - ↑riverside_total * riverside_denied_percent +
  ↑westside_total - ↑westside_total * westside_denied_percent +
  ↑mountaintop_total - ↑mountaintop_total * mountaintop_denied_percent = 148 := by
  sorry

end NUMINAMATH_CALUDE_kids_in_movie_l2332_233206


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2332_233205

/-- An isosceles triangle with side lengths satisfying x^2 - 5x + 6 = 0 has perimeter 7 or 8 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  (a + b > c ∧ a + c > b ∧ b + c > a) →  -- triangle inequality
  (a + b + c = 7 ∨ a + b + c = 8) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2332_233205


namespace NUMINAMATH_CALUDE_no_solution_exists_l2332_233280

theorem no_solution_exists : ¬∃ (a b c x : ℝ),
  (2 : ℝ)^(x * 0.15) = 5^(a * Real.sin c) ∧
  ((2 : ℝ)^(x * 0.15))^b = 32 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2332_233280


namespace NUMINAMATH_CALUDE_progression_ratio_l2332_233272

/-- Given an arithmetic progression and a geometric progression with shared elements,
    prove that the ratio of the difference of middle terms of the arithmetic progression
    to the middle term of the geometric progression is either 1/2 or -1/2. -/
theorem progression_ratio (a₁ a₂ b : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂ ∧ a₂ - (-8) = a₁ - a₂) →  -- arithmetic progression condition
  (b^2 = (-2) * (-8)) →                              -- geometric progression condition
  (a₂ - a₁) / b = 1/2 ∨ (a₂ - a₁) / b = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_progression_ratio_l2332_233272


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2332_233287

theorem min_value_x_plus_y (x y : ℝ) (h1 : 4/x + 9/y = 1) (h2 : x > 0) (h3 : y > 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2332_233287


namespace NUMINAMATH_CALUDE_prob_drawing_10_red_in_12_draws_l2332_233278

-- Define the number of white and red balls
def white_balls : ℕ := 5
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a red ball
def prob_red : ℚ := red_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of draws
def total_draws : ℕ := 12

-- Define the number of red balls needed to stop
def red_balls_to_stop : ℕ := 10

-- Define the probability of the event
def prob_event : ℚ := (Nat.choose (total_draws - 1) (red_balls_to_stop - 1)) * 
                      (prob_red ^ red_balls_to_stop) * 
                      (prob_white ^ (total_draws - red_balls_to_stop))

-- Theorem statement
theorem prob_drawing_10_red_in_12_draws : 
  prob_event = (Nat.choose 11 9) * ((3 / 8) ^ 10) * ((5 / 8) ^ 2) :=
sorry

end NUMINAMATH_CALUDE_prob_drawing_10_red_in_12_draws_l2332_233278


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2332_233227

theorem binomial_coefficient_problem (n : ℕ) (a b : ℝ) :
  (2 * n.choose 1 = 8) →
  n.choose 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2332_233227


namespace NUMINAMATH_CALUDE_a_value_l2332_233237

def round_down_tens (n : ℕ) : ℕ :=
  (n / 10) * 10

theorem a_value (A : ℕ) : 
  A < 10 → 
  round_down_tens (A * 1000 + 567) = 2560 → 
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_a_value_l2332_233237


namespace NUMINAMATH_CALUDE_license_plate_combinations_l2332_233208

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- 
Theorem: The number of license plate combinations with three letters 
(one repeated exactly once) followed by a dash and three distinct digits 
in increasing order is equal to 23,400.
-/
theorem license_plate_combinations : 
  (num_letters * (num_letters - 1) * Nat.choose letter_positions 2 * 
   Nat.choose num_digits digit_positions) = 23400 := by
  sorry


end NUMINAMATH_CALUDE_license_plate_combinations_l2332_233208


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2332_233223

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallel_plane m α) 
  (h5 : perpendicular n β) : 
  perpendicular_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2332_233223


namespace NUMINAMATH_CALUDE_special_trapezoid_area_l2332_233209

/-- Represents a trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of the longer base -/
  longer_base : ℝ
  /-- The length of one non-parallel side -/
  side : ℝ
  /-- Condition that one diagonal is perpendicular to the non-parallel side -/
  diagonal_perpendicular : Prop
  /-- Condition that the other diagonal bisects the angle between the side and base -/
  diagonal_bisects : Prop

/-- The area of the special trapezoid is 12 -/
theorem special_trapezoid_area (t : SpecialTrapezoid) 
  (h1 : t.longer_base = 5)
  (h2 : t.side = 3) : ℝ :=
by
  sorry

#check special_trapezoid_area

end NUMINAMATH_CALUDE_special_trapezoid_area_l2332_233209


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2332_233210

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  ∀ x y : ℝ, equation x y →
  (∃ x1 y1 x2 y2 : ℝ, 
    parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
    (x1, y1) = vertex1 ∧ (x2, y2) = vertex2 ∧
    abs (y1 - y2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2332_233210


namespace NUMINAMATH_CALUDE_range_of_abc_l2332_233211

theorem range_of_abc (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_l2332_233211


namespace NUMINAMATH_CALUDE_power_function_increasing_iff_m_eq_two_l2332_233238

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing_iff_m_eq_two (m : ℝ) :
  (∀ x > 0, StrictMono (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_iff_m_eq_two_l2332_233238


namespace NUMINAMATH_CALUDE_expression_result_l2332_233202

theorem expression_result : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l2332_233202


namespace NUMINAMATH_CALUDE_girls_average_score_l2332_233268

theorem girls_average_score (num_boys num_girls : ℕ) (boys_avg class_avg girls_avg : ℚ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  boys_avg = 84 → 
  class_avg = 86 → 
  (num_boys * boys_avg + num_girls * girls_avg) / (num_boys + num_girls) = class_avg → 
  girls_avg = 92 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_score_l2332_233268


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l2332_233231

theorem consecutive_numbers_sum (x : ℕ) : 
  x * (x + 1) = 12650 → x + (x + 1) = 225 := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l2332_233231


namespace NUMINAMATH_CALUDE_sum_c_d_equals_3_l2332_233282

/-- Represents a repeating decimal in the form 0.cdc... -/
structure RepeatingDecimal where
  c : ℕ
  d : ℕ

/-- The fraction 4/13 as a repeating decimal -/
def fraction_4_13 : RepeatingDecimal := sorry

theorem sum_c_d_equals_3 : fraction_4_13.c + fraction_4_13.d = 3 := by sorry

end NUMINAMATH_CALUDE_sum_c_d_equals_3_l2332_233282


namespace NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l2332_233285

/-- Calculates the number of logs left in a cookfire after a given number of hours. -/
def logs_left (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (hours : ℕ) : ℕ :=
  initial_logs + hours * add_rate - hours * burn_rate

/-- Proves that after 3 hours, the cookfire will have 3 logs left. -/
theorem cookfire_logs_after_three_hours :
  logs_left 6 3 2 3 = 3 := by
  sorry

#eval logs_left 6 3 2 3

end NUMINAMATH_CALUDE_cookfire_logs_after_three_hours_l2332_233285


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l2332_233216

theorem product_from_lcm_and_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 48) (h_gcd : Nat.gcd a b = 8) : a * b = 384 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l2332_233216


namespace NUMINAMATH_CALUDE_monochromatic_square_exists_l2332_233220

/-- A color type with two possible values -/
inductive Color
  | Red
  | Blue

/-- A point in the 2D grid -/
structure Point where
  x : Nat
  y : Nat
  h_x : x ≥ 1 ∧ x ≤ 5
  h_y : y ≥ 1 ∧ y ≤ 5

/-- A coloring of the 5x5 grid -/
def Coloring := Point → Color

/-- Check if four points form a square with sides parallel to the axes -/
def isSquare (p1 p2 p3 p4 : Point) : Prop :=
  ∃ k : Nat, k > 0 ∧
    ((p1.x + k = p2.x ∧ p1.y = p2.y ∧
      p2.x = p3.x ∧ p2.y + k = p3.y ∧
      p3.x - k = p4.x ∧ p3.y = p4.y ∧
      p4.x = p1.x ∧ p4.y + k = p1.y) ∨
     (p1.y + k = p2.y ∧ p1.x = p2.x ∧
      p2.y = p3.y ∧ p2.x + k = p3.x ∧
      p3.y - k = p4.y ∧ p3.x = p4.x ∧
      p4.y = p1.y ∧ p4.x + k = p1.x))

/-- The main theorem -/
theorem monochromatic_square_exists (c : Coloring) :
  ∃ p1 p2 p3 p4 : Point,
    isSquare p1 p2 p3 p4 ∧
    (c p1 = c p2 ∧ c p2 = c p3 ∨
     c p1 = c p2 ∧ c p2 = c p4 ∨
     c p1 = c p3 ∧ c p3 = c p4 ∨
     c p2 = c p3 ∧ c p3 = c p4) := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_square_exists_l2332_233220


namespace NUMINAMATH_CALUDE_building_units_l2332_233232

-- Define the cost of 1-bedroom and 2-bedroom units
def cost_1bed : ℕ := 360
def cost_2bed : ℕ := 450

-- Define the total cost when all units are full
def total_cost : ℕ := 4950

-- Define the number of 2-bedroom units
def num_2bed : ℕ := 7

-- Define the function to calculate the total number of units
def total_units (num_1bed : ℕ) : ℕ := num_1bed + num_2bed

-- Theorem statement
theorem building_units : 
  ∃ (num_1bed : ℕ), 
    num_1bed * cost_1bed + num_2bed * cost_2bed = total_cost ∧ 
    total_units num_1bed = 12 :=
sorry

end NUMINAMATH_CALUDE_building_units_l2332_233232


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2332_233270

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2332_233270


namespace NUMINAMATH_CALUDE_other_root_is_one_l2332_233289

-- Define the quadratic equation
def quadratic (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

-- Theorem statement
theorem other_root_is_one (b : ℝ) :
  (∃ x : ℝ, quadratic b x = 0 ∧ x = 3) →
  (∃ y : ℝ, y ≠ 3 ∧ quadratic b y = 0 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_other_root_is_one_l2332_233289


namespace NUMINAMATH_CALUDE_matrix_self_inverse_l2332_233250

/-- A 2x2 matrix is its own inverse if and only if p = 15/2 and q = -4 -/
theorem matrix_self_inverse (p q : ℚ) : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![4, p; -2, q]
  (A * A = 1) ↔ (p = 15/2 ∧ q = -4) := by
sorry

end NUMINAMATH_CALUDE_matrix_self_inverse_l2332_233250


namespace NUMINAMATH_CALUDE_complex_division_result_l2332_233246

theorem complex_division_result : (4 - 2*I) / (1 + I) = 1 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2332_233246


namespace NUMINAMATH_CALUDE_inequality_maximum_value_l2332_233283

theorem inequality_maximum_value (x y : ℝ) (hx : x > 1/2) (hy : y > 1) :
  (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ 8 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 1/2 ∧ y₀ > 1 ∧ (4 * x₀^2) / (y₀ - 1) + y₀^2 / (2 * x₀ - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_maximum_value_l2332_233283


namespace NUMINAMATH_CALUDE_mike_average_weekly_time_l2332_233276

/-- Represents Mike's weekly TV and video game schedule --/
structure MikeSchedule where
  mon_wed_fri_tv : ℕ -- Hours of TV on Monday, Wednesday, Friday
  tue_thu_tv : ℕ -- Hours of TV on Tuesday, Thursday
  weekend_tv : ℕ -- Hours of TV on weekends
  vg_days : ℕ -- Number of days Mike plays video games

/-- Calculates the average weekly time Mike spends on TV and video games over 4 weeks --/
def average_weekly_time (s : MikeSchedule) : ℚ :=
  let weekly_tv := s.mon_wed_fri_tv * 3 + s.tue_thu_tv * 2 + s.weekend_tv * 2
  let daily_vg := (weekly_tv / 7 : ℚ) / 2
  let weekly_vg := daily_vg * s.vg_days
  (weekly_tv + weekly_vg) / 7

/-- Theorem stating that Mike's average weekly time spent on TV and video games is 34 hours --/
theorem mike_average_weekly_time :
  let s : MikeSchedule := { mon_wed_fri_tv := 4, tue_thu_tv := 3, weekend_tv := 5, vg_days := 3 }
  average_weekly_time s = 34 := by sorry

end NUMINAMATH_CALUDE_mike_average_weekly_time_l2332_233276


namespace NUMINAMATH_CALUDE_power_2m_equals_half_l2332_233234

theorem power_2m_equals_half (a m n : ℝ) 
  (h1 : a^(m+n) = 1/4)
  (h2 : a^(m-n) = 2)
  (h3 : a > 0)
  (h4 : a ≠ 1) :
  a^(2*m) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_2m_equals_half_l2332_233234


namespace NUMINAMATH_CALUDE_sampling_methods_appropriateness_l2332_233259

/-- Represents a sampling scenario with a population size and sample size -/
structure SamplingScenario where
  populationSize : Nat
  sampleSize : Nat

/-- Determines if simple random sampling is appropriate for a given scenario -/
def isSimpleRandomSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize ≤ scenario.populationSize ∧ scenario.sampleSize ≤ 10

/-- Determines if systematic sampling is appropriate for a given scenario -/
def isSystematicSamplingAppropriate (scenario : SamplingScenario) : Prop :=
  scenario.sampleSize > 10 ∧ scenario.populationSize ≥ 100

theorem sampling_methods_appropriateness :
  let scenario1 : SamplingScenario := ⟨10, 2⟩
  let scenario2 : SamplingScenario := ⟨1000, 50⟩
  isSimpleRandomSamplingAppropriate scenario1 ∧
  isSystematicSamplingAppropriate scenario2 :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_appropriateness_l2332_233259


namespace NUMINAMATH_CALUDE_max_min_values_l2332_233207

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  |5*x + y| + |5*x - y| = 20

-- Define the objective function
def objective (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

-- Theorem statement
theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ objective x y = 124) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = 84) ∧
  (∀ x y : ℝ, constraint x y → 84 ≤ objective x y ∧ objective x y ≤ 124) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l2332_233207


namespace NUMINAMATH_CALUDE_factorization_cubic_l2332_233288

theorem factorization_cubic (a : ℝ) : a^3 - 6*a^2 + 9*a = a*(a-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_l2332_233288


namespace NUMINAMATH_CALUDE_inequality_proof_l2332_233212

theorem inequality_proof (A B C : ℝ) (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C) 
  (h_ineq : A^4 + B^4 + C^4 ≤ 2*(A^2*B^2 + B^2*C^2 + C^2*A^2)) :
  A^2 + B^2 + C^2 ≤ 2*(A*B + B*C + C*A) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2332_233212


namespace NUMINAMATH_CALUDE_pencil_costs_two_l2332_233260

def pencil_cost : ℝ → Prop := λ x =>
  ∃ (pen_cost : ℝ),
    pen_cost = x + 9 ∧
    x + pen_cost = 13

theorem pencil_costs_two : pencil_cost 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_costs_two_l2332_233260


namespace NUMINAMATH_CALUDE_dan_picked_more_l2332_233290

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- Theorem: Dan picked 7 more apples than Benny -/
theorem dan_picked_more : dan_apples - benny_apples = 7 := by sorry

end NUMINAMATH_CALUDE_dan_picked_more_l2332_233290


namespace NUMINAMATH_CALUDE_base8_subtraction_l2332_233257

-- Define a function to convert base 8 to decimal
def base8ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert decimal to base 8
def decimalToBase8 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_subtraction : base8ToDecimal 52 - base8ToDecimal 27 = base8ToDecimal 23 := by
  sorry

end NUMINAMATH_CALUDE_base8_subtraction_l2332_233257


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2332_233230

/-- Given a square of side length 4, with E and F as midpoints of opposite sides,
    and AG perpendicular to BF, prove that when dissected into four pieces and
    reassembled into a rectangle, the ratio of height to base is 4/5 -/
theorem square_to_rectangle_ratio (square_side : ℝ) (E F G : ℝ × ℝ) 
  (h1 : square_side = 4)
  (h2 : E.1 = 2 ∧ E.2 = 0)
  (h3 : F.1 = 0 ∧ F.2 = 2)
  (h4 : (G.1 - 4) * (F.2 - 0) = (G.2 - 0) * (F.1 - 4)) -- AG ⟂ BF
  : ∃ (rect_height rect_base : ℝ),
    rect_height * rect_base = square_side^2 ∧
    rect_height / rect_base = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l2332_233230


namespace NUMINAMATH_CALUDE_survivor_quitter_probability_survivor_quitter_probability_proof_l2332_233255

/-- The probability that both quitters are from the same tribe in a Survivor game -/
theorem survivor_quitter_probability : ℚ :=
  let total_participants : ℕ := 32
  let tribe_size : ℕ := 16
  let num_quitters : ℕ := 2

  -- The probability that both quitters are from the same tribe
  15 / 31

/-- Proof of the survivor_quitter_probability theorem -/
theorem survivor_quitter_probability_proof :
  survivor_quitter_probability = 15 / 31 := by
  sorry

end NUMINAMATH_CALUDE_survivor_quitter_probability_survivor_quitter_probability_proof_l2332_233255


namespace NUMINAMATH_CALUDE_opposite_of_eight_l2332_233256

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (x : ℤ) : ℤ := -x

/-- Theorem: The opposite of 8 is -8. -/
theorem opposite_of_eight : opposite 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_eight_l2332_233256


namespace NUMINAMATH_CALUDE_mikas_height_l2332_233291

/-- Proves that Mika's current height is 66 inches given the problem conditions --/
theorem mikas_height (initial_height : ℝ) : 
  initial_height > 0 →
  initial_height * 1.25 = 75 →
  initial_height * 1.1 = 66 :=
by
  sorry

#check mikas_height

end NUMINAMATH_CALUDE_mikas_height_l2332_233291


namespace NUMINAMATH_CALUDE_expand_product_l2332_233249

theorem expand_product (x y : ℝ) : 4 * (x + 3) * (x + 2 + y) = 4 * x^2 + 4 * x * y + 20 * x + 12 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2332_233249


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2332_233293

theorem sufficient_not_necessary_condition :
  (∃ a : ℝ, a > 1 ∧ 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2332_233293


namespace NUMINAMATH_CALUDE_first_car_speed_l2332_233243

/-- Proves that the speed of the first car is 40 miles per hour given the conditions of the problem -/
theorem first_car_speed (black_car_speed : ℝ) (initial_distance : ℝ) (overtake_time : ℝ) : 
  black_car_speed = 50 →
  initial_distance = 10 →
  overtake_time = 1 →
  (black_car_speed * overtake_time - initial_distance) / overtake_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_first_car_speed_l2332_233243


namespace NUMINAMATH_CALUDE_a_range_l2332_233219

-- Define the propositions and variables
variable (p q : Prop)
variable (x a : ℝ)

-- Define the conditions
axiom x_range : 1/2 ≤ x ∧ x ≤ 1
axiom q_def : q ↔ (x - a) * (x - a - 1) ≤ 0
axiom not_p_necessary : ¬q → ¬p
axiom not_p_not_sufficient : ¬(¬p → ¬q)

-- State the theorem
theorem a_range : 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_a_range_l2332_233219


namespace NUMINAMATH_CALUDE_ratio_equality_l2332_233279

theorem ratio_equality (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2332_233279


namespace NUMINAMATH_CALUDE_estate_division_l2332_233245

theorem estate_division (estate : ℝ) 
  (wife_share son_share daughter_share cook_share : ℝ) : 
  (daughter_share + son_share = estate / 2) →
  (daughter_share = 4 * son_share / 3) →
  (wife_share = 2 * son_share) →
  (cook_share = 500) →
  (estate = wife_share + son_share + daughter_share + cook_share) →
  estate = 7000 := by
  sorry

#check estate_division

end NUMINAMATH_CALUDE_estate_division_l2332_233245


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l2332_233264

/-- Given a rectangular metallic sheet with one dimension of 52 meters,
    if squares of 8 meters are cut from each corner to form an open box
    with a volume of 5760 cubic meters, then the length of the second
    dimension of the metallic sheet is 36 meters. -/
theorem metallic_sheet_dimension (w : ℝ) :
  w > 0 →
  (w - 2 * 8) * (52 - 2 * 8) * 8 = 5760 →
  w = 36 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l2332_233264


namespace NUMINAMATH_CALUDE_trisection_distances_l2332_233213

/-- An isosceles triangle with given distances to trisection points -/
structure IsoTriangle where
  -- Side lengths
  ab : ℝ
  ac : ℝ
  bc : ℝ
  -- Distances from C to trisection points of AB
  d1 : ℝ
  d2 : ℝ
  -- Triangle is isosceles
  isIsosceles : ab = ac
  -- d1 and d2 are the given distances
  distancesGiven : (d1 = 17 ∧ d2 = 20) ∨ (d1 = 20 ∧ d2 = 17)

/-- The theorem to be proved -/
theorem trisection_distances (t : IsoTriangle) :
  let x := Real.sqrt ((8 * t.d2^2 + 5 * t.d1^2) / 3)
  x = Real.sqrt 585 ∨ x = Real.sqrt 104 := by
  sorry

end NUMINAMATH_CALUDE_trisection_distances_l2332_233213


namespace NUMINAMATH_CALUDE_no_solution_condition_l2332_233240

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ -4 → 1 / (x - 4) + m / (x + 4) ≠ (m + 3) / (x^2 - 16)) ↔ 
  (m = -1 ∨ m = 5 ∨ m = -1/3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l2332_233240


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l2332_233215

-- Define the type for 2D points
structure Point where
  x : ℝ
  y : ℝ

-- Define the set of points
def points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩,
  ⟨0.5, 2⟩, ⟨1.5, 2⟩, ⟨2.5, 2⟩
]

-- Function to check if two triangles are congruent
def are_congruent (t1 t2 : List Point) : Prop := sorry

-- Function to count non-congruent triangles
def count_non_congruent_triangles (pts : List Point) : ℕ := sorry

-- Theorem stating the number of non-congruent triangles
theorem non_congruent_triangles_count :
  count_non_congruent_triangles points = 18 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l2332_233215


namespace NUMINAMATH_CALUDE_range_of_a_l2332_233224

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2332_233224


namespace NUMINAMATH_CALUDE_sixth_power_sum_l2332_233258

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l2332_233258


namespace NUMINAMATH_CALUDE_garden_perimeter_l2332_233236

/-- A rectangular garden with given diagonal and area has a specific perimeter. -/
theorem garden_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  a^2 + b^2 = 15^2 → -- Diagonal condition
  a * b = 54 → -- Area condition
  2 * (a + b) = 2 * Real.sqrt 333 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l2332_233236


namespace NUMINAMATH_CALUDE_garden_ratio_l2332_233261

theorem garden_ratio (area width length : ℝ) : 
  area = 507 →
  width = 13 →
  area = length * width →
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_garden_ratio_l2332_233261


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_solution_set_l2332_233262

-- Part 1
theorem quadratic_inequality_range (a : ℝ) : 
  (a > 0 ∧ ∃ x, a * x^2 - 3 * x + 2 < 0) ↔ (0 < a ∧ a < 9/8) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a < 0 then {x | 3/a < x ∧ x < 1}
  else if 0 < a ∧ a < 3 then {x | x < 3/a ∨ x > 1}
  else if a = 3 then {x | x ≠ 1}
  else {x | x < 1 ∨ x > 3/a}

theorem quadratic_inequality_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - 3 * x + 2 > a * x - 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_quadratic_inequality_solution_set_l2332_233262


namespace NUMINAMATH_CALUDE_point_outside_circle_l2332_233201

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle ∧ ∀ (x y : ℝ), (x, y) ∈ circle → (m^2 - x)^2 + (5 - y)^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2332_233201


namespace NUMINAMATH_CALUDE_cyclic_sum_square_inequality_l2332_233235

theorem cyclic_sum_square_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_square_inequality_l2332_233235


namespace NUMINAMATH_CALUDE_chessboard_problem_l2332_233297

/-- Distance function on the infinite chessboard -/
def distance (p q : ℤ × ℤ) : ℕ :=
  (Int.natAbs (p.1 - q.1)).max (Int.natAbs (p.2 - q.2))

/-- The problem statement -/
theorem chessboard_problem (A B C : ℤ × ℤ) 
  (hAB : distance A B = 100)
  (hBC : distance B C = 100)
  (hAC : distance A C = 100) :
  ∃! X : ℤ × ℤ, distance X A = 50 ∧ distance X B = 50 ∧ distance X C = 50 :=
sorry

end NUMINAMATH_CALUDE_chessboard_problem_l2332_233297


namespace NUMINAMATH_CALUDE_smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l2332_233248

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0

theorem smallest_divisible_by_all :
  ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ 168 :=
by sorry

theorem is_divisible_by_all_168 : is_divisible_by_all 168 :=
by sorry

theorem smallest_number_of_books : 
  ∃! n : ℕ, n > 0 ∧ is_divisible_by_all n ∧ ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l2332_233248


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l2332_233222

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_not_two : n ≠ 2) (h_not_three : n ≠ 3) :
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l2332_233222


namespace NUMINAMATH_CALUDE_product_evaluation_l2332_233244

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2332_233244


namespace NUMINAMATH_CALUDE_scarlet_savings_l2332_233274

/-- The amount of money Scarlet saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of the earrings Scarlet bought -/
def earrings_cost : ℕ := 23

/-- The cost of the necklace Scarlet bought -/
def necklace_cost : ℕ := 48

/-- The amount of money Scarlet has left -/
def money_left : ℕ := 9

/-- Theorem stating that Scarlet's initial savings equals the sum of her purchases and remaining money -/
theorem scarlet_savings : initial_savings = earrings_cost + necklace_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_scarlet_savings_l2332_233274


namespace NUMINAMATH_CALUDE_hall_dimension_difference_l2332_233292

/-- For a rectangular hall with width equal to half its length and area 450 sq. m,
    the difference between length and width is 15 meters. -/
theorem hall_dimension_difference (length width : ℝ) : 
  width = length / 2 →
  length * width = 450 →
  length - width = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_dimension_difference_l2332_233292


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2332_233200

theorem divisibility_theorem (a b c d e f : ℤ) 
  (h : (13 : ℤ) ∣ (a^12 + b^12 + c^12 + d^12 + e^12 + f^12)) : 
  (13^6 : ℤ) ∣ (a * b * c * d * e * f) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2332_233200


namespace NUMINAMATH_CALUDE_max_y_coordinate_difference_l2332_233247

-- Define the two functions
def f (x : ℝ) : ℝ := 3 - x^2 + x^3
def g (x : ℝ) : ℝ := 1 + x^2 + x^3

-- Define the set of intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (a b : ℝ), a ∈ intersection_points ∧ b ∈ intersection_points ∧
  ∀ (x y : ℝ), x ∈ intersection_points → y ∈ intersection_points →
  |f x - f y| ≤ |f a - f b| ∧ |f a - f b| = 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_difference_l2332_233247


namespace NUMINAMATH_CALUDE_convex_polygon_with_arithmetic_angles_l2332_233299

/-- A convex polygon with interior angles forming an arithmetic sequence,
    where the smallest angle is 100° and the largest angle is 140°, has exactly 6 sides. -/
theorem convex_polygon_with_arithmetic_angles (n : ℕ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  (∃ (a d : ℝ), 
    a = 100 ∧ -- smallest angle
    a + (n - 1) * d = 140 ∧ -- largest angle
    ∀ i : ℕ, i < n → a + i * d ≥ 0 ∧ a + i * d ≤ 180) → -- all angles are between 0° and 180°
  (n : ℝ) * (100 + 140) / 2 = 180 * (n - 2) →
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_with_arithmetic_angles_l2332_233299


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l2332_233204

/-- Checks if a number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n ≥ 17 := by sorry

theorem seventeen_is_dual_palindrome : 
  isPalindrome 17 2 ∧ isPalindrome 17 4 := by sorry

theorem smallest_dual_palindrome_is_17 : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n = 17 := by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l2332_233204


namespace NUMINAMATH_CALUDE_friends_marbles_theorem_l2332_233241

/-- Calculates the number of marbles Reggie's friend arrived with -/
def friends_initial_marbles (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ) : ℕ :=
  let games_won := total_games - games_lost
  let marbles_gained := games_won * marbles_per_game
  let reggies_initial_marbles := reggies_final_marbles - marbles_gained
  reggies_initial_marbles + marbles_per_game

theorem friends_marbles_theorem (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ)
  (h1 : total_games = 9)
  (h2 : marbles_per_game = 10)
  (h3 : reggies_final_marbles = 90)
  (h4 : games_lost = 1) :
  friends_initial_marbles total_games marbles_per_game reggies_final_marbles games_lost = 20 := by
  sorry

#eval friends_initial_marbles 9 10 90 1

end NUMINAMATH_CALUDE_friends_marbles_theorem_l2332_233241


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2332_233251

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem arithmetic_geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : geometric_sequence (a 1) (a 3) (a 4)) :
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2332_233251


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2332_233265

/-- The constant term in the expansion of (x^2 - 2/x)^6 is 240 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (x^2 - 2/x)^6
  ∃ c : ℝ, (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 240 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2332_233265


namespace NUMINAMATH_CALUDE_age_height_not_function_l2332_233266

-- Define the types for our variables
def Age := ℕ
def Height := ℝ
def Radius := ℝ
def Circumference := ℝ
def Angle := ℝ
def SineValue := ℝ
def NumSides := ℕ
def SumInteriorAngles := ℝ

-- Define the relationships as functions
def radiusToCircumference : Radius → Circumference := sorry
def angleToSine : Angle → SineValue := sorry
def sidesToInteriorAnglesSum : NumSides → SumInteriorAngles := sorry

-- Define the relationship between age and height
def ageHeightRelation : Age → Set Height := sorry

-- Theorem to prove
theorem age_height_not_function :
  ¬(∃ (f : Age → Height), ∀ a : Age, ∃! h : Height, h ∈ ageHeightRelation a) :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l2332_233266


namespace NUMINAMATH_CALUDE_stock_value_return_l2332_233277

theorem stock_value_return (initial_value : ℝ) (h : initial_value > 0) :
  let first_year_value := initial_value * 1.4
  let second_year_decrease := 2 / 7
  first_year_value * (1 - second_year_decrease) = initial_value :=
by sorry

end NUMINAMATH_CALUDE_stock_value_return_l2332_233277


namespace NUMINAMATH_CALUDE_total_coughs_equation_georgia_coughs_five_times_l2332_233267

/-- Georgia's coughs per minute -/
def G : ℕ := sorry

/-- The total number of coughs after 20 minutes -/
def total_coughs : ℕ := 300

/-- Robert coughs twice as much as Georgia -/
def roberts_coughs_per_minute : ℕ := 2 * G

/-- The total coughs after 20 minutes equals 300 -/
theorem total_coughs_equation : 20 * (G + roberts_coughs_per_minute) = total_coughs := by sorry

/-- Georgia coughs 5 times per minute -/
theorem georgia_coughs_five_times : G = 5 := by sorry

end NUMINAMATH_CALUDE_total_coughs_equation_georgia_coughs_five_times_l2332_233267


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l2332_233281

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def total_weight : ℕ := bag_weights.sum

theorem turnip_bag_weights (turnip_weight : ℕ) 
  (h_turnip : turnip_weight ∈ bag_weights) :
  (∃ (onion_weights carrot_weights : List ℕ),
    onion_weights ++ carrot_weights ++ [turnip_weight] = bag_weights ∧
    onion_weights.sum * 2 = carrot_weights.sum ∧
    onion_weights.sum + carrot_weights.sum + turnip_weight = total_weight) ↔
  (turnip_weight = 13 ∨ turnip_weight = 16) :=
sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l2332_233281


namespace NUMINAMATH_CALUDE_min_of_quadratic_l2332_233271

/-- The quadratic function f(x) = x^2 - 2px + 4q -/
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + 4*q

/-- Theorem stating that the minimum of f occurs at x = p -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = p :=
sorry

end NUMINAMATH_CALUDE_min_of_quadratic_l2332_233271


namespace NUMINAMATH_CALUDE_dividing_line_halves_area_l2332_233269

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the L-shaped region -/
def LShapedRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- Calculates the area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The line y = (5/7)x -/
def dividingLine (p : Point) : Prop := p.y = (5/7) * p.x

/-- Regions above and below the dividing line -/
def upperRegion : Set Point := {p ∈ LShapedRegion | p.y ≥ (5/7) * p.x}
def lowerRegion : Set Point := {p ∈ LShapedRegion | p.y ≤ (5/7) * p.x}

theorem dividing_line_halves_area : 
  area upperRegion = area lowerRegion := by sorry

end NUMINAMATH_CALUDE_dividing_line_halves_area_l2332_233269


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l2332_233233

theorem triangle_cosine_inequality (α β γ : Real) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = π) : 
  (Real.cos α / Real.cos (β - γ)) + 
  (Real.cos β / Real.cos (γ - α)) + 
  (Real.cos γ / Real.cos (α - β)) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l2332_233233


namespace NUMINAMATH_CALUDE_calculation_correctness_l2332_233296

theorem calculation_correctness : 
  (4 + (-2) = 2) ∧ 
  (-2 - (-1.5) = -0.5) ∧ 
  (-(-4) + 4 = 8) ∧ 
  (|-6| + |2| ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_calculation_correctness_l2332_233296


namespace NUMINAMATH_CALUDE_claire_balloons_l2332_233275

theorem claire_balloons (initial : ℕ) : 
  initial - 12 - 9 + 11 = 39 → initial = 49 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_l2332_233275


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2332_233294

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := total / (a + b + c)
  let part1 := a * x
  let part2 := b * x
  let part3 := c * x
  (total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7) →
  min part1 (min part2 part3) = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2332_233294


namespace NUMINAMATH_CALUDE_chocolate_bar_problem_l2332_233217

/-- Represents the problem of calculating unsold chocolate bars -/
theorem chocolate_bar_problem (cost_per_bar : ℕ) (total_bars : ℕ) (revenue : ℕ) : 
  cost_per_bar = 3 → 
  total_bars = 9 → 
  revenue = 18 → 
  total_bars - (revenue / cost_per_bar) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_problem_l2332_233217


namespace NUMINAMATH_CALUDE_after_two_right_turns_l2332_233295

/-- Represents a position in the square formation -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- The size of the square formation -/
def formationSize : Nat := 9

/-- Counts the number of people in front of a given position -/
def peopleInFront (pos : Position) : Nat :=
  formationSize - pos.row - 1

/-- Performs a right turn on a position -/
def rightTurn (pos : Position) : Position :=
  ⟨formationSize - pos.col + 1, pos.row⟩

/-- The main theorem to prove -/
theorem after_two_right_turns 
  (initialPos : Position)
  (h1 : peopleInFront initialPos = 2)
  (h2 : peopleInFront (rightTurn initialPos) = 4) :
  peopleInFront (rightTurn (rightTurn initialPos)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_after_two_right_turns_l2332_233295


namespace NUMINAMATH_CALUDE_equal_angle_slope_value_l2332_233252

/-- The slope of a line that forms equal angles with y = x and y = 2x --/
def equal_angle_slope : ℝ → Prop := λ k =>
  let l₁ : ℝ → ℝ := λ x => x
  let l₂ : ℝ → ℝ := λ x => 2 * x
  let angle (m₁ m₂ : ℝ) : ℝ := |((m₂ - m₁) / (1 + m₁ * m₂))|
  (angle k 1 = angle 2 k) ∧ (3 * k^2 - 2 * k - 3 = 0)

/-- The slope of a line that forms equal angles with y = x and y = 2x
    is (1 ± √10) / 3 --/
theorem equal_angle_slope_value :
  ∃ k : ℝ, equal_angle_slope k ∧ (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
sorry

end NUMINAMATH_CALUDE_equal_angle_slope_value_l2332_233252


namespace NUMINAMATH_CALUDE_sequence_formula_T_formula_l2332_233229

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : n > 0 → S n = 2 * sequence_a n - 2

theorem sequence_formula (n : ℕ) (h : n > 0) : sequence_a n = 2^n := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_formula (n : ℕ) (h : n > 0) : T n = 2^(n+2) - 4 - 2*n := by sorry

end NUMINAMATH_CALUDE_sequence_formula_T_formula_l2332_233229


namespace NUMINAMATH_CALUDE_coefficient_x3_in_product_l2332_233253

theorem coefficient_x3_in_product : 
  let p1 : Polynomial ℤ := 2 * X^4 + 3 * X^3 - 4 * X^2 + 2
  let p2 : Polynomial ℤ := X^3 - 8 * X + 3
  (p1 * p2).coeff 3 = 41 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_product_l2332_233253


namespace NUMINAMATH_CALUDE_xuzhou_metro_scientific_notation_l2332_233214

theorem xuzhou_metro_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 31900 = a * (10 : ℝ) ^ n ∧ a = 3.19 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_xuzhou_metro_scientific_notation_l2332_233214


namespace NUMINAMATH_CALUDE_charles_stroll_distance_l2332_233254

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Charles strolled 6 miles -/
theorem charles_stroll_distance :
  let speed : ℝ := 3
  let time : ℝ := 2
  distance speed time = 6 := by sorry

end NUMINAMATH_CALUDE_charles_stroll_distance_l2332_233254


namespace NUMINAMATH_CALUDE_roots_imply_p_zero_q_negative_l2332_233221

theorem roots_imply_p_zero_q_negative (α β p q : ℝ) : 
  α ≠ β →  -- α and β are distinct
  α^2 + p*α + q = 0 →  -- α is a root of the equation
  β^2 + p*β + q = 0 →  -- β is a root of the equation
  α^3 - α^2*β - α*β^2 + β^3 = 0 →  -- given condition
  p = 0 ∧ q < 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_p_zero_q_negative_l2332_233221


namespace NUMINAMATH_CALUDE_total_distance_in_feet_l2332_233286

/-- Conversion factor from miles to feet -/
def miles_to_feet : ℝ := 5280

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Conversion factor from kilometers to feet -/
def km_to_feet : ℝ := 3280.84

/-- Conversion factor from meters to feet -/
def meters_to_feet : ℝ := 3.28084

/-- Distance walked by Lionel in miles -/
def lionel_distance : ℝ := 4

/-- Distance walked by Esther in yards -/
def esther_distance : ℝ := 975

/-- Distance walked by Niklaus in feet -/
def niklaus_distance : ℝ := 1287

/-- Distance biked by Isabella in kilometers -/
def isabella_distance : ℝ := 18

/-- Distance swam by Sebastian in meters -/
def sebastian_distance : ℝ := 2400

/-- Theorem stating the total combined distance traveled by the friends in feet -/
theorem total_distance_in_feet :
  lionel_distance * miles_to_feet +
  esther_distance * yards_to_feet +
  niklaus_distance +
  isabella_distance * km_to_feet +
  sebastian_distance * meters_to_feet = 89261.136 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_in_feet_l2332_233286


namespace NUMINAMATH_CALUDE_square_of_sum_l2332_233203

theorem square_of_sum (x y k m : ℝ) (h1 : x * y = k) (h2 : x^2 + y^2 = m) :
  (x + y)^2 = m + 2*k := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2332_233203


namespace NUMINAMATH_CALUDE_crackers_distribution_l2332_233273

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 45 → num_friends = 15 → crackers_per_friend = total_crackers / num_friends → 
  crackers_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l2332_233273


namespace NUMINAMATH_CALUDE_spencer_jumps_per_minute_l2332_233263

-- Define the parameters
def minutes_per_session : ℕ := 10
def sessions_per_day : ℕ := 2
def total_jumps : ℕ := 400
def total_days : ℕ := 5

-- Theorem to prove
theorem spencer_jumps_per_minute :
  (total_jumps : ℚ) / ((minutes_per_session * sessions_per_day * total_days) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jumps_per_minute_l2332_233263


namespace NUMINAMATH_CALUDE_smallest_integer_l2332_233226

theorem smallest_integer (a b : ℕ+) (ha : a = 60) (h : Nat.lcm a b / Nat.gcd a b = 44) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), (Nat.lcm a n / Nat.gcd a n = 44) → m ≤ n ∧ m = 165 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l2332_233226


namespace NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l2332_233228

/-- Represents the price increase in yuan -/
def price_increase : ℝ := 5

/-- Initial profit per kilogram in yuan -/
def initial_profit_per_kg : ℝ := 10

/-- Initial daily sales volume in kilograms -/
def initial_sales_volume : ℝ := 500

/-- Decrease in sales volume per yuan of price increase -/
def sales_volume_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_daily_profit : ℝ := 6000

/-- Theorem stating that the given price increase achieves the target daily profit -/
theorem price_increase_achieves_target_profit :
  (initial_sales_volume - sales_volume_decrease_rate * price_increase) *
  (initial_profit_per_kg + price_increase) = target_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l2332_233228


namespace NUMINAMATH_CALUDE_cricket_bat_selling_price_l2332_233218

-- Define the profit
def profit : ℝ := 150

-- Define the profit percentage
def profitPercentage : ℝ := 20

-- Define the selling price
def sellingPrice : ℝ := 900

-- Theorem to prove
theorem cricket_bat_selling_price :
  let costPrice := profit / (profitPercentage / 100)
  sellingPrice = costPrice + profit := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_selling_price_l2332_233218


namespace NUMINAMATH_CALUDE_jennys_number_l2332_233284

theorem jennys_number (y : ℝ) : 10 * (y / 2 - 6) = 70 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_jennys_number_l2332_233284


namespace NUMINAMATH_CALUDE_roots_of_cubic_equation_l2332_233239

variable (a b c d α β : ℝ)

def original_quadratic (x : ℝ) : ℝ := x^2 - (a + d)*x + (a*d - b*c)

def new_quadratic (x : ℝ) : ℝ := x^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*x + (a*d - b*c)^3

theorem roots_of_cubic_equation 
  (h1 : original_quadratic α = 0)
  (h2 : original_quadratic β = 0) :
  new_quadratic (α^3) = 0 ∧ new_quadratic (β^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_cubic_equation_l2332_233239


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l2332_233225

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 1106 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l2332_233225


namespace NUMINAMATH_CALUDE_girls_distance_calculation_l2332_233242

/-- The number of laps run by boys -/
def boys_laps : ℕ := 124

/-- The additional laps run by girls compared to boys -/
def extra_girls_laps : ℕ := 48

/-- The fraction of a mile per lap -/
def mile_per_lap : ℚ := 5 / 13

/-- The distance run by girls in miles -/
def girls_distance : ℚ := (boys_laps + extra_girls_laps) * mile_per_lap

theorem girls_distance_calculation :
  girls_distance = (124 + 48) * (5 / 13) := by sorry

end NUMINAMATH_CALUDE_girls_distance_calculation_l2332_233242
