import Mathlib

namespace NUMINAMATH_CALUDE_meet_once_l2092_209294

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def numberOfMeetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michaelSpeed = 4)
  (h2 : scenario.truckSpeed = 8)
  (h3 : scenario.pailDistance = 300)
  (h4 : scenario.truckStopTime = 45)
  (h5 : scenario.initialDistance = 300) : 
  numberOfMeetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l2092_209294


namespace NUMINAMATH_CALUDE_inequality_properties_l2092_209244

theorem inequality_properties (a b c : ℝ) (h : a < b) :
  (a + c < b + c) ∧
  (a - 2 < b - 2) ∧
  (2 * a < 2 * b) ∧
  (-3 * a > -3 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2092_209244


namespace NUMINAMATH_CALUDE_correct_page_difference_l2092_209298

/-- Calculates the difference in pages read between yesterday and today -/
def pagesDifference (totalPages yesterday tomorrow : ℕ) : ℕ :=
  yesterday - (totalPages - yesterday - tomorrow)

theorem correct_page_difference :
  pagesDifference 100 35 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_page_difference_l2092_209298


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2092_209249

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 80 + 10 * x

/-- Beta T-Shirt Company's pricing function -/
def beta_price (x : ℕ) : ℕ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_acme_cheaper : ℕ := 13

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < beta_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ beta_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2092_209249


namespace NUMINAMATH_CALUDE_min_value_w_l2092_209266

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45 ≥ 28 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 5 * b^2 + 12 * a - 10 * b + 45 = 28 := by
sorry

end NUMINAMATH_CALUDE_min_value_w_l2092_209266


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2092_209216

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h : Real.sqrt (a^2 - 6*a + 9) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2092_209216


namespace NUMINAMATH_CALUDE_symmetric_difference_properties_l2092_209254

open Set

variable {α : Type*} [MeasurableSpace α]

def symmetricDifference (A B : Set α) : Set α := (A \ B) ∪ (B \ A)

theorem symmetric_difference_properties 
  (A B : ℕ → Set α) : 
  (symmetricDifference (A 1) (B 1) = symmetricDifference (Aᶜ 1) (Bᶜ 1)) ∧ 
  (symmetricDifference (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) ∧
  (symmetricDifference (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_difference_properties_l2092_209254


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l2092_209209

/-- Given two points A and B in 2D space, this theorem proves that if the coordinates of A are (-2, -1) and the vector from A to B is (3, 4), then the coordinates of B are (1, 3). -/
theorem coordinates_of_point_B (A B : ℝ × ℝ) : 
  A = (-2, -1) → (B.1 - A.1, B.2 - A.2) = (3, 4) → B = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l2092_209209


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2092_209221

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 11) :
  x^2 + y^2 + z^2 ≥ 121/29 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2092_209221


namespace NUMINAMATH_CALUDE_water_bottles_stolen_solve_water_bottle_theft_l2092_209264

theorem water_bottles_stolen (initial_bottles : ℕ) (lost_bottles : ℕ) (stickers_per_bottle : ℕ) (total_stickers : ℕ) : ℕ :=
  let remaining_after_loss := initial_bottles - lost_bottles
  let remaining_after_theft := total_stickers / stickers_per_bottle
  remaining_after_loss - remaining_after_theft

theorem solve_water_bottle_theft : water_bottles_stolen 10 2 3 21 = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_stolen_solve_water_bottle_theft_l2092_209264


namespace NUMINAMATH_CALUDE_possible_c_value_l2092_209205

/-- An obtuse-angled triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  obtuse : c^2 > a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that 2√5 is a possible value for c in the given obtuse triangle -/
theorem possible_c_value (t : ObtuseTriangle) 
  (ha : t.a = Real.sqrt 2)
  (hb : t.b = 2 * Real.sqrt 2)
  (hc : t.c > t.b) :
  ∃ (t' : ObtuseTriangle), t'.a = t.a ∧ t'.b = t.b ∧ t'.c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_possible_c_value_l2092_209205


namespace NUMINAMATH_CALUDE_samantha_birth_year_l2092_209265

def first_amc_year : ℕ := 1985

def samantha_age_at_seventh_amc : ℕ := 12

theorem samantha_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = first_amc_year + 6 - samantha_age_at_seventh_amc ∧
    birth_year = 1979 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l2092_209265


namespace NUMINAMATH_CALUDE_hyperbolic_amplitude_properties_l2092_209286

/-- Hyperbolic cosine -/
noncomputable def ch (x : ℝ) : ℝ := sorry

/-- Hyperbolic sine -/
noncomputable def sh (x : ℝ) : ℝ := sorry

/-- Hyperbolic tangent -/
noncomputable def th (x : ℝ) : ℝ := sorry

/-- Tangent -/
noncomputable def tg (α : ℝ) : ℝ := sorry

theorem hyperbolic_amplitude_properties (x α : ℝ) 
  (h1 : ch x ^ 2 - sh x ^ 2 = 1)
  (h2 : tg α = sh x / ch x) : 
  (ch x = 1 / Real.cos α) ∧ (th (x / 2) = tg (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_amplitude_properties_l2092_209286


namespace NUMINAMATH_CALUDE_min_club_members_l2092_209220

theorem min_club_members : ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ 2/5 < m/n ∧ m/n < 1/2) ∧ 
  (∀ (k : ℕ), 0 < k ∧ k < n → ¬∃ (j : ℕ), j > 0 ∧ 2/5 < j/k ∧ j/k < 1/2) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_club_members_l2092_209220


namespace NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2092_209284

theorem gcd_of_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_4557_1953_5115_l2092_209284


namespace NUMINAMATH_CALUDE_point_c_coordinates_l2092_209247

/-- Given points A, B, and C on a line, where C divides AB in the ratio 2:1,
    prove that C has the specified coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-3, -2) →
  B = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C is on line segment AB
  dist A C = 2 * dist C B →                             -- AC = 2CB
  C = (11/3, 8) := by
  sorry


end NUMINAMATH_CALUDE_point_c_coordinates_l2092_209247


namespace NUMINAMATH_CALUDE_power_four_squared_cubed_minus_four_l2092_209217

theorem power_four_squared_cubed_minus_four : (4^2)^3 - 4 = 4092 := by
  sorry

end NUMINAMATH_CALUDE_power_four_squared_cubed_minus_four_l2092_209217


namespace NUMINAMATH_CALUDE_minutes_before_noon_l2092_209246

theorem minutes_before_noon (x : ℕ) : 
  (180 - (x + 40) = 3 * x) →  -- Condition 1 and 3
  x = 35                      -- The result we want to prove
  := by sorry

end NUMINAMATH_CALUDE_minutes_before_noon_l2092_209246


namespace NUMINAMATH_CALUDE_tim_initial_amount_l2092_209243

/-- Tim's initial amount of money in cents -/
def initial_amount : ℕ := sorry

/-- Amount Tim paid for the candy bar in cents -/
def candy_bar_cost : ℕ := 45

/-- Amount Tim received as change in cents -/
def change_received : ℕ := 5

/-- Theorem stating that Tim's initial amount equals 50 cents -/
theorem tim_initial_amount : initial_amount = candy_bar_cost + change_received := by sorry

end NUMINAMATH_CALUDE_tim_initial_amount_l2092_209243


namespace NUMINAMATH_CALUDE_spoiled_cross_to_square_l2092_209223

/-- Represents a symmetrical Greek cross -/
structure GreekCross where
  arm_length : ℝ
  arm_width : ℝ
  symmetrical : arm_length > 0 ∧ arm_width > 0

/-- Represents a square -/
structure Square where
  side_length : ℝ
  is_positive : side_length > 0

/-- Represents a Greek cross with a square cut out -/
structure SpoiledGreekCross where
  cross : GreekCross
  cut_out : Square
  fits_end : cut_out.side_length = cross.arm_width

/-- Represents a piece obtained from cutting the spoiled Greek cross -/
structure Piece where
  area : ℝ
  is_positive : area > 0

/-- Theorem stating that a spoiled Greek cross can be cut into four pieces
    that can be reassembled into a square -/
theorem spoiled_cross_to_square (sc : SpoiledGreekCross) :
  ∃ (p1 p2 p3 p4 : Piece) (result : Square),
    p1.area + p2.area + p3.area + p4.area = result.side_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_spoiled_cross_to_square_l2092_209223


namespace NUMINAMATH_CALUDE_g_difference_l2092_209274

/-- The function g defined as g(n) = n^3 + 3n^2 + 3n + 1 -/
def g (n : ℝ) : ℝ := n^3 + 3*n^2 + 3*n + 1

/-- Theorem stating that g(s) - g(s-2) = 6s^2 + 2 for any real number s -/
theorem g_difference (s : ℝ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2092_209274


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l2092_209291

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of measures is 4:1
  abs (a - b) = 54 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l2092_209291


namespace NUMINAMATH_CALUDE_max_value_parabola_l2092_209292

theorem max_value_parabola :
  ∀ x : ℝ, 0 < x → x < 6 → x * (6 - x) ≤ 9 ∧ ∃ y : ℝ, 0 < y ∧ y < 6 ∧ y * (6 - y) = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_parabola_l2092_209292


namespace NUMINAMATH_CALUDE_inequality_proof_l2092_209240

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2092_209240


namespace NUMINAMATH_CALUDE_hexadecagon_area_theorem_l2092_209225

/-- A hexadecagon inscribed in a square with specific properties -/
structure InscribedHexadecagon where
  /-- The perimeter of the square in which the hexadecagon is inscribed -/
  square_perimeter : ℝ
  /-- The property that every side of the square is trisected twice equally -/
  trisected_twice : Prop

/-- The area of the inscribed hexadecagon -/
def hexadecagon_area (h : InscribedHexadecagon) : ℝ := sorry

/-- Theorem stating the area of the inscribed hexadecagon with given properties -/
theorem hexadecagon_area_theorem (h : InscribedHexadecagon) 
  (h_perimeter : h.square_perimeter = 160) : hexadecagon_area h = 1344 := by sorry

end NUMINAMATH_CALUDE_hexadecagon_area_theorem_l2092_209225


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_quadratic_function_coefficient_range_l2092_209255

-- Part 1
def is_symmetric_about_negative_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-1 - x) = f (-1 + x)

theorem quadratic_function_uniqueness
  (f : ℝ → ℝ)
  (h1 : ∃ a b c, ∀ x, f x = a * x^2 + b * x + c)
  (h2 : is_symmetric_about_negative_one f)
  (h3 : f 0 = 1)
  (h4 : ∃ x_min, ∀ x, f x ≥ f x_min ∧ f x_min = 0) :
  ∀ x, f x = (x + 1)^2 := by sorry

-- Part 2
theorem quadratic_function_coefficient_range
  (b : ℝ)
  (h : ∀ x ∈ Set.Ioo 0 1, |x^2 + b*x| ≤ 1) :
  b ∈ Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_quadratic_function_coefficient_range_l2092_209255


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l2092_209269

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

/-- The x-coordinate of the point of interest -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of interest -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at x₀ -/
def m : ℝ := 2*x₀ - 2

theorem tangent_and_normal_equations :
  (∀ x y, 2*x - y + 1 = 0 ↔ y = m*(x - x₀) + y₀) ∧
  (∀ x y, x + 2*y - 12 = 0 ↔ y = -1/(2*m)*(x - x₀) + y₀) := by
  sorry


end NUMINAMATH_CALUDE_tangent_and_normal_equations_l2092_209269


namespace NUMINAMATH_CALUDE_probability_two_colored_is_four_ninths_l2092_209233

/-- Represents a cube divided into smaller cubes -/
structure DividedCube where
  total_small_cubes : ℕ
  two_colored_faces : ℕ

/-- The probability of selecting a cube with exactly 2 colored faces -/
def probability_two_colored (cube : DividedCube) : ℚ :=
  cube.two_colored_faces / cube.total_small_cubes

/-- Theorem stating the probability of selecting a cube with exactly 2 colored faces -/
theorem probability_two_colored_is_four_ninths (cube : DividedCube) 
    (h1 : cube.total_small_cubes = 27)
    (h2 : cube.two_colored_faces = 12) : 
  probability_two_colored cube = 4/9 := by
  sorry

#check probability_two_colored_is_four_ninths

end NUMINAMATH_CALUDE_probability_two_colored_is_four_ninths_l2092_209233


namespace NUMINAMATH_CALUDE_grapes_purchased_l2092_209271

/-- Represents the price of grapes per kilogram -/
def grape_price : ℕ := 70

/-- Represents the price of mangoes per kilogram -/
def mango_price : ℕ := 55

/-- Represents the amount of mangoes purchased in kilograms -/
def mango_amount : ℕ := 11

/-- Represents the total amount paid -/
def total_paid : ℕ := 1165

/-- Theorem stating that the amount of grapes purchased is 8 kg -/
theorem grapes_purchased : ∃ (g : ℕ), g * grape_price + mango_amount * mango_price = total_paid ∧ g = 8 := by
  sorry

end NUMINAMATH_CALUDE_grapes_purchased_l2092_209271


namespace NUMINAMATH_CALUDE_rain_probability_theorem_l2092_209289

/-- The probability of rain on each day -/
def rain_prob : ℚ := 2/3

/-- The number of consecutive days -/
def num_days : ℕ := 5

/-- The probability of no rain on a single day -/
def no_rain_prob : ℚ := 1 - rain_prob

/-- The probability of two consecutive dry days -/
def two_dry_days_prob : ℚ := no_rain_prob ^ 2

/-- The number of pairs of consecutive days in the given period -/
def num_pairs : ℕ := num_days - 1

theorem rain_probability_theorem :
  (no_rain_prob ^ num_days = 1/243) ∧
  (two_dry_days_prob * num_pairs = 4/9) := by
  sorry


end NUMINAMATH_CALUDE_rain_probability_theorem_l2092_209289


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2092_209257

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m : ℂ) + (m^2 - 5*m + 6 : ℂ)*Complex.I = Complex.I * ((m^2 - 5*m + 6 : ℂ)) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2092_209257


namespace NUMINAMATH_CALUDE_complex_magnitude_l2092_209237

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2092_209237


namespace NUMINAMATH_CALUDE_right_triangle_to_square_l2092_209267

theorem right_triangle_to_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  b = 10 →           -- longer leg is 10
  b = 2*a →          -- condition for forming a square
  a = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_to_square_l2092_209267


namespace NUMINAMATH_CALUDE_emily_savings_l2092_209230

def shoe_price : ℕ := 50
def promotion_b_discount : ℕ := 20

def cost_promotion_a (price : ℕ) : ℕ := price + price / 2

def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ := price + (price - discount)

theorem emily_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
sorry

end NUMINAMATH_CALUDE_emily_savings_l2092_209230


namespace NUMINAMATH_CALUDE_slope_of_line_l2092_209224

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2092_209224


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l2092_209297

/-- Given a line l passing through M(-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at P1 and P2, and P being the midpoint of P1P2,
    if k2 is the slope of OP, then k1k2 = -1/2 -/
theorem ellipse_intersection_slope_product 
  (k1 : ℝ) (P1 P2 P : ℝ × ℝ) (k2 : ℝ)
  (h1 : k1 ≠ 0)
  (h2 : P1.1^2 + 2*P1.2^2 = 4)
  (h3 : P2.1^2 + 2*P2.2^2 = 4)
  (h4 : P1.2 = k1 * (P1.1 + 2))
  (h5 : P2.2 = k1 * (P2.1 + 2))
  (h6 : P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2))
  (h7 : k2 = P.2 / P.1) :
  k1 * k2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l2092_209297


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2092_209235

theorem parallelogram_perimeter (a b : ℝ) (ha : a = Real.sqrt 20) (hb : b = Real.sqrt 125) :
  2 * (a + b) = 14 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2092_209235


namespace NUMINAMATH_CALUDE_expression_always_defined_l2092_209208

theorem expression_always_defined (x : ℝ) (h : x > 12) : x^2 - 24*x + 144 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_defined_l2092_209208


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l2092_209214

/-- The y-intercept of the common external tangent line with positive slope to two circles -/
theorem external_tangent_y_intercept 
  (center1 : ℝ × ℝ) (radius1 : ℝ) (center2 : ℝ × ℝ) (radius2 : ℝ) 
  (h1 : center1 = (1, 5)) 
  (h2 : radius1 = 3) 
  (h3 : center2 = (15, 10)) 
  (h4 : radius2 = 10) : 
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨ 
       (x - center2.1)^2 + (y - center2.2)^2 = radius2^2)) ∧ 
    b = 7416 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l2092_209214


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2092_209270

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2092_209270


namespace NUMINAMATH_CALUDE_line_mb_equals_two_l2092_209268

/-- Given a line with equation y = mx + b passing through points (0, 1) and (1, 3), prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (1 = m * 0 + b) →  -- The line passes through (0, 1)
  (3 = m * 1 + b) →  -- The line passes through (1, 3)
  m * b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_two_l2092_209268


namespace NUMINAMATH_CALUDE_min_value_S_n_l2092_209258

/-- The sum of the first n terms of the sequence -/
def S_n (n : ℕ+) : ℤ := n^2 - 12*n

/-- The minimum value of S_n for positive integers n -/
def min_S_n : ℤ := -36

theorem min_value_S_n :
  ∀ n : ℕ+, S_n n ≥ min_S_n ∧ ∃ m : ℕ+, S_n m = min_S_n :=
sorry

end NUMINAMATH_CALUDE_min_value_S_n_l2092_209258


namespace NUMINAMATH_CALUDE_sine_amplitude_l2092_209202

/-- Given a sine function y = a * sin(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and -3, then a = 4 -/
theorem sine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by sorry

end NUMINAMATH_CALUDE_sine_amplitude_l2092_209202


namespace NUMINAMATH_CALUDE_johnson_family_seating_l2092_209212

/-- The number of ways to seat 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

theorem johnson_family_seating :
  seating_arrangements 5 4 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_seating_l2092_209212


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2092_209273

theorem parallelepiped_volume (base_area : ℝ) (angle : ℝ) (lateral_area1 : ℝ) (lateral_area2 : ℝ) :
  base_area = 4 →
  angle = 30 * π / 180 →
  lateral_area1 = 6 →
  lateral_area2 = 12 →
  ∃ (a b c : ℝ),
    a * b * Real.sin angle = base_area ∧
    a * c = lateral_area1 ∧
    b * c = lateral_area2 ∧
    a * b * c = 12 := by
  sorry

#check parallelepiped_volume

end NUMINAMATH_CALUDE_parallelepiped_volume_l2092_209273


namespace NUMINAMATH_CALUDE_rectangle_tiling_existence_l2092_209288

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a tiling of rectangles -/
def Tiling := List Rectangle

/-- Checks if a list of rectangles can tile a larger rectangle -/
def canTile (tiles : Tiling) (target : Rectangle) : Prop := sorry

/-- The specific tiles we're allowed to use -/
def allowedTiles : Tiling := [Rectangle.mk 4 6, Rectangle.mk 5 7]

/-- The theorem stating the existence of N and that 840 is a valid value -/
theorem rectangle_tiling_existence :
  ∃ (N : ℕ), ∀ (m n : ℕ), m > N → n > N →
    canTile allowedTiles (Rectangle.mk m n) ∧ canTile allowedTiles (Rectangle.mk 841 841) := by
  sorry

#check rectangle_tiling_existence

end NUMINAMATH_CALUDE_rectangle_tiling_existence_l2092_209288


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2092_209245

theorem complex_fraction_equals_i : (Complex.I + 3) / (1 - 3 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2092_209245


namespace NUMINAMATH_CALUDE_tangent_and_normal_lines_l2092_209234

-- Define the curve
def x (t : ℝ) : ℝ := t - t^4
def y (t : ℝ) : ℝ := t^2 - t^3

-- Define the parameter value
def t₀ : ℝ := 1

-- State the theorem
theorem tangent_and_normal_lines :
  let x₀ := x t₀
  let y₀ := y t₀
  let dx := deriv x t₀
  let dy := deriv y t₀
  let m_tangent := dy / dx
  let m_normal := -1 / m_tangent
  (∀ t : ℝ, y t - y₀ = m_tangent * (x t - x₀)) ∧
  (∀ t : ℝ, y t - y₀ = m_normal * (x t - x₀)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_normal_lines_l2092_209234


namespace NUMINAMATH_CALUDE_infinite_x₀_finite_values_l2092_209299

/-- The function f(x) = 3x - x^2 -/
def f (x : ℝ) : ℝ := 3 * x - x^2

/-- The sequence x_n defined by x_n = f(x_{n-1}) -/
def seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (seq x₀ n)

/-- A set is finite if it is empty or there exists a bijection with a finite segment of ℕ -/
def IsFiniteSet (S : Set ℝ) : Prop :=
  S = ∅ ∨ ∃ n : ℕ, ∃ h : Fin n → S, Function.Bijective h

/-- The set of values in the sequence starting from x₀ -/
def seqValues (x₀ : ℝ) : Set ℝ :=
  { x | ∃ n : ℕ, seq x₀ n = x }

/-- The theorem stating that infinitely many x₀ in [0, 3] lead to finite value sets -/
theorem infinite_x₀_finite_values :
  ∃ S : Set ℝ, S ⊆ Set.Icc 0 3 ∧ Set.Infinite S ∧ ∀ x₀ ∈ S, IsFiniteSet (seqValues x₀) := by
  sorry


end NUMINAMATH_CALUDE_infinite_x₀_finite_values_l2092_209299


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2092_209236

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2092_209236


namespace NUMINAMATH_CALUDE_solution_greater_than_two_l2092_209226

theorem solution_greater_than_two (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_greater_than_two_l2092_209226


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2092_209295

theorem circle_area_from_circumference (r : ℝ) (k : ℝ) : 
  (2 * π * r = 36 * π) → (π * r^2 = k * π) → k = 324 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2092_209295


namespace NUMINAMATH_CALUDE_linear_relationship_values_l2092_209276

/-- Given a linear relationship between x and y, prove the values of y for specific x values -/
theorem linear_relationship_values (x y : ℝ) :
  (y = 3 * x - 1) →
  (x = 1 → y = 2) ∧ (x = 5 → y = 14) := by
  sorry

end NUMINAMATH_CALUDE_linear_relationship_values_l2092_209276


namespace NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l2092_209277

theorem cube_root_neg_eight_plus_sqrt_nine_equals_one :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (9 : ℝ).sqrt = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l2092_209277


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2092_209279

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- Define the derivative of f(x)
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 8 * x - m

theorem quadratic_function_value : 
  ∀ (m : ℝ), 
  (∀ x : ℝ, x ≥ -2 → (f_derivative m) x ≥ 0) →  -- f(x) is increasing on [−2, +∞)
  (∀ x : ℝ, x < -2 → (f_derivative m) x < 0) →  -- f(x) is decreasing on (-∞, −2)
  f m 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2092_209279


namespace NUMINAMATH_CALUDE_triangle_side_product_l2092_209293

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if (a+b)^2 - c^2 = 4 and C = 60°, then ab = 4/3 -/
theorem triangle_side_product (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (π/3) = 1/2) :
  a * b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_l2092_209293


namespace NUMINAMATH_CALUDE_sandra_leftover_money_l2092_209207

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 1/2
def jelly_bean_cost : ℚ := 1/5
def candy_count : ℕ := 14
def jelly_bean_count : ℕ := 20

theorem sandra_leftover_money :
  (sandra_savings + mother_gift + father_gift : ℚ) - 
  (candy_count * candy_cost + jelly_bean_count * jelly_bean_cost) = 11 :=
by sorry

end NUMINAMATH_CALUDE_sandra_leftover_money_l2092_209207


namespace NUMINAMATH_CALUDE_num_clips_property_l2092_209296

/-- The number of clips on a curtain rod after k halving steps -/
def num_clips (k : ℕ) : ℕ :=
  2^(k-1) + 1

/-- The property that each interval has a middle clip -/
def has_middle_clip (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ j > i ∧ j - i = (n - i) / 2

/-- The theorem stating that num_clips satisfies the middle clip property for all steps -/
theorem num_clips_property (k : ℕ) : 
  k > 0 → has_middle_clip (num_clips k) :=
sorry

end NUMINAMATH_CALUDE_num_clips_property_l2092_209296


namespace NUMINAMATH_CALUDE_correct_arrangements_l2092_209248

def num_seats : ℕ := 5
def num_teachers : ℕ := 4

/-- The number of arrangements where Teacher A is to the left of Teacher B -/
def arrangements_a_left_of_b : ℕ := 60

theorem correct_arrangements :
  arrangements_a_left_of_b = (num_seats.factorial / (num_seats - num_teachers).factorial) / 2 :=
sorry

end NUMINAMATH_CALUDE_correct_arrangements_l2092_209248


namespace NUMINAMATH_CALUDE_duck_park_problem_l2092_209285

theorem duck_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (geese_leave : ℕ) : 
  initial_ducks = 25 →
  geese_arrive = 4 →
  geese_leave = 10 →
  ((2 * initial_ducks) - 10) - geese_leave - (initial_ducks + geese_arrive) = 1 := by
  sorry

end NUMINAMATH_CALUDE_duck_park_problem_l2092_209285


namespace NUMINAMATH_CALUDE_recruit_count_l2092_209241

theorem recruit_count (peter nikolai denis total : ℕ) : 
  peter = 50 →
  nikolai = 100 →
  denis = 170 →
  (total - peter - 1 = 4 * (total - denis - 1) ∨
   total - nikolai - 1 = 4 * (total - denis - 1) ∨
   total - peter - 1 = 4 * (total - nikolai - 1)) →
  total = 213 :=
by sorry

end NUMINAMATH_CALUDE_recruit_count_l2092_209241


namespace NUMINAMATH_CALUDE_one_absent_two_present_probability_l2092_209287

def absent_probability : ℚ := 1 / 20

def present_probability : ℚ := 1 - absent_probability

def probability_one_absent_two_present (p q : ℚ) : ℚ := 3 * p * q * q

theorem one_absent_two_present_probability : 
  probability_one_absent_two_present absent_probability present_probability = 1083 / 8000 := by
  sorry

end NUMINAMATH_CALUDE_one_absent_two_present_probability_l2092_209287


namespace NUMINAMATH_CALUDE_donut_purchase_proof_l2092_209211

/-- Represents the number of items purchased over the week -/
def total_items : ℕ := 4

/-- Price of a croissant in cents -/
def croissant_price : ℕ := 60

/-- Price of a donut in cents -/
def donut_price : ℕ := 90

/-- Represents the number of donuts purchased -/
def num_donuts : ℕ := sorry

/-- Represents the number of croissants purchased -/
def num_croissants : ℕ := total_items - num_donuts

/-- Total cost in cents -/
def total_cost : ℕ := num_donuts * donut_price + num_croissants * croissant_price

theorem donut_purchase_proof : 
  (num_donuts + num_croissants = total_items) ∧ 
  (total_cost % 100 = 0) ∧ 
  (num_donuts = 2) := by sorry

end NUMINAMATH_CALUDE_donut_purchase_proof_l2092_209211


namespace NUMINAMATH_CALUDE_waiter_initial_customers_l2092_209201

def initial_customers (customers_left : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  remaining_tables * people_per_table + customers_left

theorem waiter_initial_customers : 
  initial_customers 12 4 8 = 44 := by
  sorry

end NUMINAMATH_CALUDE_waiter_initial_customers_l2092_209201


namespace NUMINAMATH_CALUDE_sean_sunday_spending_l2092_209282

/-- Represents Sean's Sunday purchases and their costs --/
structure SundayPurchases where
  almond_croissant_price : ℝ
  salami_cheese_croissant_price : ℝ
  plain_croissant_price : ℝ
  focaccia_price : ℝ
  latte_price : ℝ
  almond_croissant_quantity : ℕ
  salami_cheese_croissant_quantity : ℕ
  plain_croissant_quantity : ℕ
  focaccia_quantity : ℕ
  latte_quantity : ℕ

/-- Calculates the total cost of Sean's Sunday purchases --/
def total_cost (purchases : SundayPurchases) : ℝ :=
  purchases.almond_croissant_price * purchases.almond_croissant_quantity +
  purchases.salami_cheese_croissant_price * purchases.salami_cheese_croissant_quantity +
  purchases.plain_croissant_price * purchases.plain_croissant_quantity +
  purchases.focaccia_price * purchases.focaccia_quantity +
  purchases.latte_price * purchases.latte_quantity

/-- Theorem stating that Sean's total spending on Sunday is $21.00 --/
theorem sean_sunday_spending (purchases : SundayPurchases)
  (h1 : purchases.almond_croissant_price = 4.5)
  (h2 : purchases.salami_cheese_croissant_price = 4.5)
  (h3 : purchases.plain_croissant_price = 3)
  (h4 : purchases.focaccia_price = 4)
  (h5 : purchases.latte_price = 2.5)
  (h6 : purchases.almond_croissant_quantity = 1)
  (h7 : purchases.salami_cheese_croissant_quantity = 1)
  (h8 : purchases.plain_croissant_quantity = 1)
  (h9 : purchases.focaccia_quantity = 1)
  (h10 : purchases.latte_quantity = 2)
  : total_cost purchases = 21 := by
  sorry

end NUMINAMATH_CALUDE_sean_sunday_spending_l2092_209282


namespace NUMINAMATH_CALUDE_cyclist_distance_l2092_209218

/-- Represents a cyclist's journey -/
structure CyclistJourney where
  v : ℝ  -- speed in mph
  t : ℝ  -- time in hours
  d : ℝ  -- distance in miles

/-- Conditions for the cyclist's journey -/
def journeyConditions (j : CyclistJourney) : Prop :=
  j.d = j.v * j.t ∧
  j.d = (j.v + 1) * (3/4 * j.t) ∧
  j.d = (j.v - 1) * (j.t + 3)

theorem cyclist_distance (j : CyclistJourney) 
  (h : journeyConditions j) : j.d = 36 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_distance_l2092_209218


namespace NUMINAMATH_CALUDE_herd_size_l2092_209280

theorem herd_size (herd : ℕ) : 
  (herd / 3 + herd / 6 + herd / 8 + herd / 24 + 15 = herd) → 
  herd = 45 := by
  sorry

end NUMINAMATH_CALUDE_herd_size_l2092_209280


namespace NUMINAMATH_CALUDE_investment_average_rate_l2092_209219

/-- Proves that for a $6000 investment split between 3% and 5.5% interest rates
    with equal annual returns, the average interest rate is 3.88% -/
theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.055 →
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < total ∧
    rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.0388 :=
by sorry


end NUMINAMATH_CALUDE_investment_average_rate_l2092_209219


namespace NUMINAMATH_CALUDE_max_daily_sales_revenue_l2092_209252

def P (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def dailySalesRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales_revenue :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ dailySalesRevenue t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → dailySalesRevenue t ≤ 1125) ∧
  (dailySalesRevenue 25 = 1125) :=
sorry

end NUMINAMATH_CALUDE_max_daily_sales_revenue_l2092_209252


namespace NUMINAMATH_CALUDE_p_squared_plus_98_composite_l2092_209203

theorem p_squared_plus_98_composite (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 98) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_98_composite_l2092_209203


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2092_209272

theorem arithmetic_calculation : 5 * 12 + 6 * 11 + 13 * 5 + 7 * 9 = 254 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2092_209272


namespace NUMINAMATH_CALUDE_code_number_correspondence_exists_l2092_209261

-- Define the set of codes
def Codes : Type := Fin 5 → Fin 3 → Char

-- Define the set of numbers
def Numbers : Type := Fin 5 → Nat

-- Define the given codes
def given_codes : Codes := λ i j ↦ 
  match i, j with
  | 0, 0 => 'R' | 0, 1 => 'W' | 0, 2 => 'Q'
  | 1, 0 => 'S' | 1, 1 => 'X' | 1, 2 => 'W'
  | 2, 0 => 'P' | 2, 1 => 'S' | 2, 2 => 'T'
  | 3, 0 => 'X' | 3, 1 => 'N' | 3, 2 => 'Y'
  | 4, 0 => 'N' | 4, 1 => 'X' | 4, 2 => 'Y'
  | _, _ => 'A' -- Default case, should never be reached

-- Define the given and solution numbers
def given_and_solution_numbers : Numbers := λ i ↦
  match i with
  | 0 => 286
  | 1 => 540
  | 2 => 793
  | 3 => 948
  | 4 => 450

-- Define a bijection type between Codes and Numbers
def CodeNumberBijection := {f : Codes → Numbers // Function.Bijective f}

theorem code_number_correspondence_exists : ∃ (f : CodeNumberBijection), 
  ∀ (i : Fin 5), f.val given_codes i = given_and_solution_numbers i :=
sorry

end NUMINAMATH_CALUDE_code_number_correspondence_exists_l2092_209261


namespace NUMINAMATH_CALUDE_book_sale_percentage_gain_l2092_209251

/-- Calculates the percentage gain for a book sale given the number of books purchased,
    the number of books whose selling price equals the total cost price,
    and the total number of books purchased. -/
def calculatePercentageGain (booksPurchased : ℕ) (booksSoldForCost : ℕ) : ℚ :=
  ((booksPurchased : ℚ) / booksSoldForCost - 1) * 100

/-- Theorem stating that the percentage gain for the given book sale scenario is (3/7) * 100. -/
theorem book_sale_percentage_gain :
  calculatePercentageGain 50 35 = (3/7) * 100 := by
  sorry

#eval calculatePercentageGain 50 35

end NUMINAMATH_CALUDE_book_sale_percentage_gain_l2092_209251


namespace NUMINAMATH_CALUDE_girls_average_score_l2092_209242

theorem girls_average_score (num_boys num_girls : ℕ) (boys_avg class_avg girls_avg : ℚ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  boys_avg = 84 → 
  class_avg = 86 → 
  (num_boys * boys_avg + num_girls * girls_avg) / (num_boys + num_girls) = class_avg → 
  girls_avg = 92 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_score_l2092_209242


namespace NUMINAMATH_CALUDE_rotate_parabola_180_l2092_209259

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola -/
def originalParabola : Parabola := ⟨1, -5, 9⟩

/-- Theorem stating that rotating the original parabola 180 degrees results in the new parabola -/
theorem rotate_parabola_180 :
  let (x, y) := rotate180 x y
  y = -(originalParabola.a * x^2 + originalParabola.b * x + originalParabola.c) :=
by sorry

end NUMINAMATH_CALUDE_rotate_parabola_180_l2092_209259


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l2092_209239

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def N (a : ℝ) : Set ℝ := {x : ℝ | 1 - 3*a < x ∧ x ≤ 2*a}

theorem intersection_implies_a_range (a : ℝ) : M ∩ N a = M → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l2092_209239


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2092_209250

/-- Given a, b, c form a geometric sequence, the quadratic function f(x) = ax^2 + bx + c has no real roots -/
theorem quadratic_no_real_roots (a b c : ℝ) (h_geo : b^2 = a*c) (h_pos : a*c > 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2092_209250


namespace NUMINAMATH_CALUDE_younger_brother_age_after_30_years_l2092_209232

/-- Given two brothers with an age difference of 10 years, where the elder is 40 years old now,
    prove that the younger brother will be 60 years old after 30 years. -/
theorem younger_brother_age_after_30_years
  (age_difference : ℕ)
  (elder_brother_current_age : ℕ)
  (years_from_now : ℕ)
  (h1 : age_difference = 10)
  (h2 : elder_brother_current_age = 40)
  (h3 : years_from_now = 30) :
  elder_brother_current_age - age_difference + years_from_now = 60 :=
by sorry

end NUMINAMATH_CALUDE_younger_brother_age_after_30_years_l2092_209232


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l2092_209238

/-- A quadratic function f(x) = x^2 + ax + b satisfying f(f(x) + x) / f(x) = x^2 + 2023x + 1776 
    has coefficients a = 2021 and b = -246. -/
theorem quadratic_function_coefficients (a b : ℝ) : 
  (∀ x, (((x^2 + a*x + b)^2 + a*(x^2 + a*x + b) + b) / (x^2 + a*x + b) = x^2 + 2023*x + 1776)) → 
  (a = 2021 ∧ b = -246) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l2092_209238


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l2092_209210

/-- A triangle with side lengths 3, 4, and 5 has an inscribed circle with radius 1. -/
theorem inscribed_circle_radius_345_triangle :
  ∀ (a b c r : ℝ),
  a = 3 ∧ b = 4 ∧ c = 5 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_345_triangle_l2092_209210


namespace NUMINAMATH_CALUDE_total_squares_is_83_l2092_209263

/-- Represents the count of squares of a specific size in the figure -/
structure SquareCount where
  size : Nat
  count : Nat

/-- Represents the figure composed of squares and isosceles right triangles -/
structure Figure where
  squareCounts : List SquareCount

/-- Calculates the total number of squares in the figure -/
def totalSquares (f : Figure) : Nat :=
  f.squareCounts.foldl (fun acc sc => acc + sc.count) 0

/-- The specific figure described in the problem -/
def problemFigure : Figure :=
  { squareCounts := [
      { size := 1, count := 40 },
      { size := 2, count := 25 },
      { size := 3, count := 12 },
      { size := 4, count := 5 },
      { size := 5, count := 1 }
    ] }

theorem total_squares_is_83 : totalSquares problemFigure = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_is_83_l2092_209263


namespace NUMINAMATH_CALUDE_inequality_proof_l2092_209200

theorem inequality_proof (a b c A α : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hα : α > 0)
  (hsum : a + b + c = A) (hA : A ≤ 1) :
  (1/a - a)^α + (1/b - b)^α + (1/c - c)^α ≥ 3 * (3/A - A/3)^α := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2092_209200


namespace NUMINAMATH_CALUDE_first_number_value_l2092_209281

theorem first_number_value (x : ℝ) : x + 2 * (8 - 3) = 24.16 → x = 14.16 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l2092_209281


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l2092_209206

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (a n)^2 + 1

def is_prime_divisor_of_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, p.Prime ∧ p ∣ a n

theorem infinitely_many_prime_divisors :
  ∀ m : ℕ, ∃ p : ℕ, p > m ∧ is_prime_divisor_of_sequence p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l2092_209206


namespace NUMINAMATH_CALUDE_grid_adjacent_difference_l2092_209231

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 18
  col : Fin 18

/-- The type of the grid -/
def Grid := Fin 18 → Fin 18 → ℕ+

/-- Two cells are adjacent if they share an edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- The main theorem -/
theorem grid_adjacent_difference (g : Grid) 
  (h : ∀ (c1 c2 : Cell), c1 ≠ c2 → g c1.row c1.col ≠ g c2.row c2.col) :
  ∃ (c1 c2 c3 c4 : Cell), 
    adjacent c1 c2 ∧ adjacent c3 c4 ∧ 
    (c1, c2) ≠ (c3, c4) ∧
    (g c1.row c1.col).val + 10 ≤ (g c2.row c2.col).val ∧
    (g c3.row c3.col).val + 10 ≤ (g c4.row c4.col).val :=
sorry

end NUMINAMATH_CALUDE_grid_adjacent_difference_l2092_209231


namespace NUMINAMATH_CALUDE_min_treasures_is_15_l2092_209262

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried." -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried." -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried." -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried." -/
def signs_3 : ℕ := 3

/-- Predicate to check if a sign is truthful given the number of treasures -/
def is_truthful (sign_value : ℕ) (num_treasures : ℕ) : Prop :=
  sign_value ≠ num_treasures

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ),
    n = 15 ∧
    (∀ m : ℕ, m < n →
      ¬(is_truthful signs_15 m ∧
        is_truthful signs_8 m ∧
        is_truthful signs_4 m ∧
        is_truthful signs_3 m)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_treasures_is_15_l2092_209262


namespace NUMINAMATH_CALUDE_tea_sale_prices_l2092_209290

/-- Calculates the sale price per kg for a given tea type -/
def salePricePerKg (quantity : ℕ) (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  (quantity * costPrice + quantity * costPrice * profitPercentage) / quantity

theorem tea_sale_prices :
  let teaA := salePricePerKg 80 15 (25/100)
  let teaB := salePricePerKg 20 20 (30/100)
  let teaC := salePricePerKg 50 25 (20/100)
  let teaD := salePricePerKg 30 30 (15/100)
  teaA = 75/4 ∧ teaB = 26 ∧ teaC = 30 ∧ teaD = 69/2 :=
by sorry

end NUMINAMATH_CALUDE_tea_sale_prices_l2092_209290


namespace NUMINAMATH_CALUDE_pool_dimensions_l2092_209283

/-- Represents the dimensions and costs of a rectangular open-top swimming pool. -/
structure Pool where
  shortSide : ℝ  -- Length of the shorter side of the rectangular bottom
  depth : ℝ      -- Depth of the pool
  bottomCost : ℝ -- Cost per square meter for constructing the bottom
  wallCost : ℝ   -- Cost per square meter for constructing the walls
  totalCost : ℝ  -- Total construction cost

/-- Calculates the total cost of constructing the pool. -/
def calculateCost (p : Pool) : ℝ :=
  p.bottomCost * p.shortSide * (2 * p.shortSide) + 
  p.wallCost * (p.shortSide + 2 * p.shortSide) * 2 * p.depth

/-- Theorem stating that the pool with given specifications has sides of 3m and 6m. -/
theorem pool_dimensions (p : Pool) 
  (h1 : p.depth = 2)
  (h2 : p.bottomCost = 200)
  (h3 : p.wallCost = 100)
  (h4 : p.totalCost = 7200)
  (h5 : calculateCost p = p.totalCost) :
  p.shortSide = 3 ∧ 2 * p.shortSide = 6 := by
  sorry

#check pool_dimensions

end NUMINAMATH_CALUDE_pool_dimensions_l2092_209283


namespace NUMINAMATH_CALUDE_decimal_expansion_222nd_digit_l2092_209253

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeat_length : ℕ := 6

/-- The position we're interested in -/
def position : ℕ := 222

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_expansion_222nd_digit :
  nth_digit position = 5 := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_222nd_digit_l2092_209253


namespace NUMINAMATH_CALUDE_group_size_l2092_209228

theorem group_size (N : ℝ) 
  (h1 : N / 5 = N * (1 / 5))  -- 1/5 of the group plays at least one instrument
  (h2 : N * (1 / 5) - 128 = N * 0.04)  -- Probability of playing exactly one instrument is 0.04
  : N = 800 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2092_209228


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2092_209256

/-- The range of m for which the quadratic equation (m-3)x^2 + 4x + 1 = 0 has two real roots -/
theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m - 3) * x₁^2 + 4 * x₁ + 1 = 0 ∧ (m - 3) * x₂^2 + 4 * x₂ + 1 = 0) ↔ 
  (m ≤ 7 ∧ m ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2092_209256


namespace NUMINAMATH_CALUDE_pseudoprime_construction_infinite_pseudoprimes_l2092_209278

/-- A number n is a pseudoprime to base a if it's composite and a^(n-1) ≡ 1 (mod n) -/
def IsPseudoprime (n : ℕ) (a : ℕ) : Prop :=
  ¬ Nat.Prime n ∧ a^(n-1) % n = 1

/-- Given a pseudoprime m, 2^m - 1 is also a pseudoprime -/
theorem pseudoprime_construction (m : ℕ) (a : ℕ) (h : IsPseudoprime m a) :
  ∃ b : ℕ, IsPseudoprime (2^m - 1) b :=
sorry

/-- There are infinitely many pseudoprimes -/
theorem infinite_pseudoprimes : ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ a : ℕ, IsPseudoprime m a :=
sorry

end NUMINAMATH_CALUDE_pseudoprime_construction_infinite_pseudoprimes_l2092_209278


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2092_209222

theorem quadratic_roots_difference_squared :
  ∀ α β : ℝ, 
  (α^2 - 3*α + 1 = 0) → 
  (β^2 - 3*β + 1 = 0) → 
  (α ≠ β) →
  (α - β)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l2092_209222


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l2092_209260

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1 ∨ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l2092_209260


namespace NUMINAMATH_CALUDE_discounted_notebooks_cost_l2092_209229

/-- The total cost of purchasing discounted notebooks -/
theorem discounted_notebooks_cost 
  (x : ℝ) -- original price of a notebook in yuan
  (y : ℝ) -- discount amount in yuan
  : 5 * (x - y) = 5 * x - 5 * y := by
  sorry

end NUMINAMATH_CALUDE_discounted_notebooks_cost_l2092_209229


namespace NUMINAMATH_CALUDE_problem_solution_l2092_209204

def row1 (n : ℕ) : ℤ := (-2) ^ n

def row2 (n : ℕ) : ℤ := row1 n + 2

def row3 (n : ℕ) : ℤ := (-2) ^ (n - 1)

theorem problem_solution :
  (row1 4 = 16) ∧
  (∀ n : ℕ, row2 n = row1 n + 2) ∧
  (∃ k : ℕ, row3 k + row3 (k + 1) + row3 (k + 2) = -192 ∧
            row3 k = -64 ∧ row3 (k + 1) = 128 ∧ row3 (k + 2) = -256) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2092_209204


namespace NUMINAMATH_CALUDE_specific_factory_production_l2092_209227

/-- A factory that produces toys -/
structure ToyFactory where
  workingDaysPerWeek : ℕ
  dailyProduction : ℕ
  constantProduction : Prop

/-- Calculate the weekly production of a toy factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.workingDaysPerWeek * factory.dailyProduction

/-- Theorem stating the weekly production of a specific factory -/
theorem specific_factory_production :
  ∀ (factory : ToyFactory),
    factory.workingDaysPerWeek = 4 →
    factory.dailyProduction = 1375 →
    factory.constantProduction →
    weeklyProduction factory = 5500 := by
  sorry

end NUMINAMATH_CALUDE_specific_factory_production_l2092_209227


namespace NUMINAMATH_CALUDE_pencil_cartons_theorem_l2092_209213

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_cartons : ℕ
  marker_boxes_per_carton : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of pencil cartons bought -/
def pencil_cartons_bought (s : SchoolSupplies) : ℕ :=
  (s.total_spent - s.marker_cartons * s.marker_carton_cost) / (s.pencil_boxes_per_carton * s.pencil_box_cost)

/-- Theorem stating the number of pencil cartons bought -/
theorem pencil_cartons_theorem (s : SchoolSupplies) 
  (h1 : s.pencil_boxes_per_carton = 10)
  (h2 : s.pencil_box_cost = 2)
  (h3 : s.marker_cartons = 10)
  (h4 : s.marker_boxes_per_carton = 5)
  (h5 : s.marker_carton_cost = 4)
  (h6 : s.total_spent = 600) :
  pencil_cartons_bought s = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cartons_theorem_l2092_209213


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2092_209275

theorem square_root_equation_solution (A C : ℝ) (hA : A ≥ 0) (hC : C ≥ 0) :
  ∃ x : ℝ, x > 0 ∧
    Real.sqrt (2 + A * C + 2 * C * x) + Real.sqrt (A * C - 2 + 2 * A * x) =
    Real.sqrt (2 * (A + C) * x + 2 * A * C) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2092_209275


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2092_209215

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k a₂ a₃ b₂ b₃ : ℝ)
  (hk : k ≠ 0)
  (ha : ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2)
  (hb : ∃ r : ℝ, r ≠ 1 ∧ b₂ = k * r ∧ b₃ = k * r^2)
  (hdiff : ∀ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) → p ≠ r)
  (hcond : a₃ - b₃ = 3 * (a₂ - b₂)) :
  ∃ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) ∧ p + r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l2092_209215
