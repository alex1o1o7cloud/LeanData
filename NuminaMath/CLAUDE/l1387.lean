import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_real_iff_m_eq_neg_one_l1387_138772

/-- The complex number (m^2 + i) / (1 - mi) is real if and only if m = -1 -/
theorem complex_fraction_real_iff_m_eq_neg_one (m : ℝ) :
  (((m^2 : ℂ) + Complex.I) / (1 - m * Complex.I)).im = 0 ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_iff_m_eq_neg_one_l1387_138772


namespace NUMINAMATH_CALUDE_train_speed_difference_l1387_138730

/-- Given two trains traveling towards each other, this theorem proves that
    the difference in their speeds is 30 km/hr under specific conditions. -/
theorem train_speed_difference 
  (distance : ℝ) 
  (meeting_time : ℝ) 
  (express_speed : ℝ) 
  (h1 : distance = 390) 
  (h2 : meeting_time = 3) 
  (h3 : express_speed = 80) : 
  express_speed - (distance / meeting_time - express_speed) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_difference_l1387_138730


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l1387_138712

/-- Given a cistern with two taps, prove the emptying time of the second tap -/
theorem cistern_emptying_time (fill_time : ℝ) (combined_time : ℝ) (empty_time : ℝ) : 
  fill_time = 4 → combined_time = 44 / 7 → empty_time = 11 → 
  1 / fill_time - 1 / empty_time = 1 / combined_time := by
sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l1387_138712


namespace NUMINAMATH_CALUDE_unique_point_on_line_l1387_138767

-- Define the line passing through (4, 11) and (16, 1)
def line_equation (x y : ℤ) : Prop :=
  5 * x + 6 * y = 43

-- Define the condition for positive integers
def positive_integer (n : ℤ) : Prop :=
  0 < n

theorem unique_point_on_line :
  ∃! p : ℤ × ℤ, line_equation p.1 p.2 ∧ positive_integer p.1 ∧ positive_integer p.2 ∧ p = (5, 3) :=
by
  sorry

#check unique_point_on_line

end NUMINAMATH_CALUDE_unique_point_on_line_l1387_138767


namespace NUMINAMATH_CALUDE_power_function_implies_m_eq_neg_one_l1387_138728

/-- A function f is a power function if it has the form f(x) = ax^n where a and n are constants and a ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- Given f(x) = (2m+3)x^(m^2-3) is a power function, prove that m = -1 -/
theorem power_function_implies_m_eq_neg_one (m : ℝ) 
    (h : IsPowerFunction (fun x ↦ (2*m+3) * x^(m^2-3))) : 
    m = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_implies_m_eq_neg_one_l1387_138728


namespace NUMINAMATH_CALUDE_commission_allocation_l1387_138750

theorem commission_allocation (commission_rate : ℚ) (total_sales : ℚ) (amount_saved : ℚ)
  (h1 : commission_rate = 12 / 100)
  (h2 : total_sales = 24000)
  (h3 : amount_saved = 1152) :
  (total_sales * commission_rate - amount_saved) / (total_sales * commission_rate) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_commission_allocation_l1387_138750


namespace NUMINAMATH_CALUDE_mod_twelve_six_nine_l1387_138790

theorem mod_twelve_six_nine (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_six_nine_l1387_138790


namespace NUMINAMATH_CALUDE_debate_team_boys_l1387_138708

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (total : ℕ) (boys : ℕ) : 
  girls = 32 →
  groups = 7 →
  group_size = 9 →
  total = groups * group_size →
  boys = total - girls →
  boys = 31 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l1387_138708


namespace NUMINAMATH_CALUDE_sheep_raising_profit_range_l1387_138756

/-- Represents the profit calculation for sheep raising with and without technical guidance. -/
theorem sheep_raising_profit_range (x : ℝ) : 
  x > 0 →
  (0.15 * (1 + 0.25*x) * (100000 - x) ≥ 0.15 * 100000) ↔ 
  (0 < x ∧ x ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_sheep_raising_profit_range_l1387_138756


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l1387_138733

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 12*a*b) :
  |((a+b)/(a-b))| = Real.sqrt (7/5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l1387_138733


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_solutions_l1387_138729

theorem sum_of_squares_quadratic_solutions : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 16*x₁ + 15 = 0 → 
  x₂^2 - 16*x₂ + 15 = 0 → 
  x₁ ≠ x₂ → 
  x₁^2 + x₂^2 = 226 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_solutions_l1387_138729


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1387_138771

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, x^4 = (x^3 + 3*x^2 + 2*x + 1) * q + (-x^2 - x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1387_138771


namespace NUMINAMATH_CALUDE_exists_m_with_x_squared_leq_eight_l1387_138746

theorem exists_m_with_x_squared_leq_eight : ∃ m : ℝ, m ≤ 2 ∧ ∃ x > m, x^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_with_x_squared_leq_eight_l1387_138746


namespace NUMINAMATH_CALUDE_water_level_rise_l1387_138786

/-- The rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 15)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l1387_138786


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1387_138774

theorem complex_fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 15) = 75 / 8 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1387_138774


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1387_138709

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 10 < 0 ↔ -2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1387_138709


namespace NUMINAMATH_CALUDE_unique_valid_number_l1387_138791

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n < 10000000) ∧
  (∀ d : ℕ, d < 7 → (∃! i : ℕ, i < 7 ∧ (n / 10^i) % 10 = d)) ∧
  (n % 100 % 2 = 0 ∧ (n / 100000) % 100 % 2 = 0) ∧
  (n % 1000 % 3 = 0 ∧ (n / 10000) % 1000 % 3 = 0) ∧
  (n % 10000 % 4 = 0 ∧ (n / 1000) % 10000 % 4 = 0) ∧
  (n % 100000 % 5 = 0 ∧ (n / 100) % 100000 % 5 = 0) ∧
  (n % 1000000 % 6 = 0 ∧ (n / 10) % 1000000 % 6 = 0)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n ∧ n = 3216540 := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1387_138791


namespace NUMINAMATH_CALUDE_two_cubed_and_three_squared_are_like_terms_l1387_138758

-- Define what it means for two expressions to be like terms
def are_like_terms (a b : ℕ) : Prop :=
  (∃ (x y : ℕ), a = x ∧ b = y) ∨ (∀ (x y : ℕ), a ≠ x ∧ b ≠ y)

-- Theorem statement
theorem two_cubed_and_three_squared_are_like_terms :
  are_like_terms (2^3) (3^2) :=
sorry

end NUMINAMATH_CALUDE_two_cubed_and_three_squared_are_like_terms_l1387_138758


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1387_138773

theorem simplify_fraction_product : 8 * (18 / 5) * (-40 / 27) = -128 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1387_138773


namespace NUMINAMATH_CALUDE_simplify_fraction_l1387_138783

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 50 + Real.sqrt 32 + 3 * Real.sqrt 18) = (5 * Real.sqrt 2) / 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1387_138783


namespace NUMINAMATH_CALUDE_min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l1387_138725

/-- Represents a hexagon formed by unit equilateral triangles -/
structure TriangleHexagon where
  /-- The number of unit equilateral triangles used to form the hexagon -/
  num_triangles : ℕ
  /-- The perimeter of the hexagon -/
  perimeter : ℕ
  /-- Assertion that the hexagon is formed without gaps or overlaps -/
  no_gaps_or_overlaps : Prop
  /-- Assertion that all internal angles of the hexagon are not greater than 180 degrees -/
  angles_not_exceeding_180 : Prop

/-- Theorem stating the minimum perimeter of a hexagon formed by 57 unit equilateral triangles -/
theorem min_perimeter_57_triangle_hexagon :
  ∀ h : TriangleHexagon,
    h.num_triangles = 57 →
    h.no_gaps_or_overlaps →
    h.angles_not_exceeding_180 →
    h.perimeter ≥ 19 := by
  sorry

/-- Existence of a hexagon with perimeter 19 formed by 57 unit equilateral triangles -/
theorem exists_57_triangle_hexagon_with_perimeter_19 :
  ∃ h : TriangleHexagon,
    h.num_triangles = 57 ∧
    h.perimeter = 19 ∧
    h.no_gaps_or_overlaps ∧
    h.angles_not_exceeding_180 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l1387_138725


namespace NUMINAMATH_CALUDE_original_water_amount_l1387_138795

/-- Proves that the original amount of water in a glass is 15 ounces, given the daily evaporation rate,
    evaporation period, and the percentage of water evaporated. -/
theorem original_water_amount
  (daily_evaporation : ℝ)
  (evaporation_period : ℕ)
  (evaporation_percentage : ℝ)
  (h1 : daily_evaporation = 0.05)
  (h2 : evaporation_period = 15)
  (h3 : evaporation_percentage = 0.05)
  (h4 : daily_evaporation * ↑evaporation_period = evaporation_percentage * original_amount) :
  original_amount = 15 :=
by
  sorry

#check original_water_amount

end NUMINAMATH_CALUDE_original_water_amount_l1387_138795


namespace NUMINAMATH_CALUDE_root_multiplicity_two_l1387_138707

variable (n : ℕ)

def f (A B : ℝ) (x : ℝ) : ℝ := A * x^(n+1) + B * x^n + 1

theorem root_multiplicity_two (A B : ℝ) :
  (f n A B 1 = 0 ∧ (deriv (f n A B)) 1 = 0) ↔ (A = n ∧ B = -(n+1)) := by sorry

end NUMINAMATH_CALUDE_root_multiplicity_two_l1387_138707


namespace NUMINAMATH_CALUDE_reciprocals_from_product_l1387_138732

theorem reciprocals_from_product (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_reciprocals_from_product_l1387_138732


namespace NUMINAMATH_CALUDE_hypotenuse_of_45_45_90_triangle_l1387_138745

-- Define a right triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  angle_opposite_leg1 : ℝ
  is_right_triangle : leg1^2 + leg2^2 = hypotenuse^2

-- Theorem statement
theorem hypotenuse_of_45_45_90_triangle 
  (triangle : RightTriangle) 
  (h1 : triangle.leg1 = 12)
  (h2 : triangle.angle_opposite_leg1 = 45) :
  triangle.hypotenuse = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_of_45_45_90_triangle_l1387_138745


namespace NUMINAMATH_CALUDE_vincents_earnings_l1387_138766

def fantasy_book_price : ℚ := 4
def literature_book_price : ℚ := fantasy_book_price / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def days : ℕ := 5

theorem vincents_earnings :
  (fantasy_book_price * fantasy_books_sold_per_day +
   literature_book_price * literature_books_sold_per_day) * days = 180 := by
  sorry

end NUMINAMATH_CALUDE_vincents_earnings_l1387_138766


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1387_138727

theorem triangle_side_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Sides are positive
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Angles are positive
  A + B + C = Real.pi →  -- Sum of angles in a triangle
  a / (Real.sin A) = b / (Real.sin B) →  -- Sine law
  a / (Real.sin A) = c / (Real.sin C) →  -- Sine law
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →  -- Cosine law
  (Real.sin B)^2 = 2 * (Real.sin A) * (Real.sin C) →
  a > c →
  Real.cos B = 1/4 →
  a / c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1387_138727


namespace NUMINAMATH_CALUDE_range_of_t_l1387_138798

def A (t : ℝ) : Set ℝ := {1, t}

theorem range_of_t (t : ℝ) : t ∈ {x : ℝ | x ≠ 1} ↔ t ∈ A t := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l1387_138798


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l1387_138781

theorem simplified_fourth_root (c d : ℕ+) :
  (2^5 * 5^3 : ℝ)^(1/4) = c * d^(1/4) → c + d = 252 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l1387_138781


namespace NUMINAMATH_CALUDE_forest_foxes_l1387_138742

theorem forest_foxes (total : ℕ) (deer_fraction : ℚ) (fox_fraction : ℚ) : 
  total = 160 →
  deer_fraction = 7 / 8 →
  fox_fraction = 1 - deer_fraction →
  (fox_fraction * total : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_forest_foxes_l1387_138742


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1387_138722

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1387_138722


namespace NUMINAMATH_CALUDE_original_ratio_proof_l1387_138755

/-- Represents the number of students in each category -/
structure StudentCount where
  boarders : ℕ
  dayStudents : ℕ

/-- The ratio of boarders to day students -/
def ratio (sc : StudentCount) : ℚ :=
  sc.boarders / sc.dayStudents

theorem original_ratio_proof (initial : StudentCount) (final : StudentCount) :
  initial.boarders = 330 →
  final.boarders = initial.boarders + 66 →
  ratio final = 1 / 2 →
  ratio initial = 5 / 12 := by
  sorry

#check original_ratio_proof

end NUMINAMATH_CALUDE_original_ratio_proof_l1387_138755


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l1387_138782

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 4^2 - 2^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l1387_138782


namespace NUMINAMATH_CALUDE_other_factor_of_60n_l1387_138748

theorem other_factor_of_60n (n : ℕ) (k : ℕ) : 
  (4 ∣ 60 * n) → 
  (k ∣ 60 * n) → 
  n ≥ 8 → 
  k = 120 := by
sorry

end NUMINAMATH_CALUDE_other_factor_of_60n_l1387_138748


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l1387_138716

theorem simplified_fraction_ratio (m : ℝ) : 
  let expression := (6 * m + 12) / 3
  ∃ (c d : ℤ), expression = c * m + d ∧ (c : ℚ) / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l1387_138716


namespace NUMINAMATH_CALUDE_y_coordinate_product_l1387_138740

/-- The product of y-coordinates for points on x = -2 that are 12 units from (6, 3) -/
theorem y_coordinate_product : ∃ y₁ y₂ : ℝ,
  ((-2 - 6)^2 + (y₁ - 3)^2 = 12^2) ∧
  ((-2 - 6)^2 + (y₂ - 3)^2 = 12^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -71 := by
sorry

end NUMINAMATH_CALUDE_y_coordinate_product_l1387_138740


namespace NUMINAMATH_CALUDE_residue_1237_mod_17_l1387_138752

theorem residue_1237_mod_17 : 1237 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_1237_mod_17_l1387_138752


namespace NUMINAMATH_CALUDE_unpainted_area_specific_case_l1387_138741

/-- Represents the configuration of two crossed boards -/
structure CrossedBoards where
  width1 : ℝ
  width2 : ℝ
  angle : ℝ

/-- Calculates the area of the unpainted region on the first board -/
def unpainted_area (boards : CrossedBoards) : ℝ :=
  boards.width1 * boards.width2

/-- Theorem stating the area of the unpainted region for specific board widths and angle -/
theorem unpainted_area_specific_case :
  let boards : CrossedBoards := ⟨5, 7, 45⟩
  unpainted_area boards = 35 := by sorry

end NUMINAMATH_CALUDE_unpainted_area_specific_case_l1387_138741


namespace NUMINAMATH_CALUDE_muirhead_inequality_l1387_138734

open Real

/-- Muirhead's Inequality -/
theorem muirhead_inequality (a₁ a₂ a₃ b₁ b₂ b₃ x y z : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ ≥ 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ ≥ 0)
  (hab : a₁ ≥ b₁ ∧ a₁ + a₂ ≥ b₁ + b₂ ∧ a₁ + a₂ + a₃ ≥ b₁ + b₂ + b₃)
  (hxyz : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^a₁ * y^a₂ * z^a₃ + x^a₁ * y^a₃ * z^a₂ + x^a₂ * y^a₁ * z^a₃ + 
  x^a₂ * y^a₃ * z^a₁ + x^a₃ * y^a₁ * z^a₂ + x^a₃ * y^a₂ * z^a₁ ≥ 
  x^b₁ * y^b₂ * z^b₃ + x^b₁ * y^b₃ * z^b₂ + x^b₂ * y^b₁ * z^b₃ + 
  x^b₂ * y^b₃ * z^b₁ + x^b₃ * y^b₁ * z^b₂ + x^b₃ * y^b₂ * z^b₁ :=
sorry

end NUMINAMATH_CALUDE_muirhead_inequality_l1387_138734


namespace NUMINAMATH_CALUDE_range_of_m_l1387_138704

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 2) :
  (∃ m : ℝ, x + y/4 < m^2 - m) ↔ ∃ m : ℝ, m < -1 ∨ m > 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1387_138704


namespace NUMINAMATH_CALUDE_triangle_max_sin_sum_l1387_138738

theorem triangle_max_sin_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C →
  (∃ (max : ℝ), max = Real.sqrt 3 ∧ 
    ∀ A' B' : ℝ, 0 < A' ∧ 0 < B' ∧ A' + B' = 2*π/3 →
      Real.sin A' + Real.sin B' ≤ max) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_sin_sum_l1387_138738


namespace NUMINAMATH_CALUDE_ninety_degrees_to_radians_l1387_138764

theorem ninety_degrees_to_radians :
  (90 : ℝ) * π / 180 = π / 2 := by sorry

end NUMINAMATH_CALUDE_ninety_degrees_to_radians_l1387_138764


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1387_138759

/-- 
For a regular polygon where each exterior angle measures 30 degrees, 
the sum of the measures of the interior angles is 1800 degrees.
-/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 2 → 
  exterior_angle = 30 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1387_138759


namespace NUMINAMATH_CALUDE_multiple_is_two_l1387_138717

-- Define the variables
def mother_age : ℕ := 40
def daughter_age : ℕ := 30 -- This is derived, not given directly
def multiple : ℚ := 2 -- This is what we want to prove

-- Define the conditions
def condition1 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  m + x * d = 70

def condition2 (m : ℕ) (d : ℕ) (x : ℚ) : Prop :=
  d + x * m = 95

-- Theorem statement
theorem multiple_is_two :
  condition1 mother_age daughter_age multiple ∧
  condition2 mother_age daughter_age multiple ∧
  multiple = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_is_two_l1387_138717


namespace NUMINAMATH_CALUDE_tangent_points_focus_slope_l1387_138715

/-- The slope of the line connecting the tangent points and the focus of a parabola -/
theorem tangent_points_focus_slope (x₀ y₀ : ℝ) : 
  x₀ = -1 → y₀ = 2 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- Tangent points satisfy the parabola equation
    y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ ∧
    -- Tangent lines pass through (x₀, y₀)
    (∃ k₁ k₂ : ℝ, y₁ - y₀ = k₁*(x₁ - x₀) ∧ y₂ - y₀ = k₂*(x₂ - x₀)) →
    -- Slope of the line connecting tangent points and focus
    (y₁ - 1/4) / (x₁ - 1/4) = 1 ∧ (y₂ - 1/4) / (x₂ - 1/4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_points_focus_slope_l1387_138715


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1387_138731

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1387_138731


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l1387_138751

theorem quadratic_linear_intersection
  (a d : ℝ) (x₁ x₂ : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x₁ ≠ x₂)
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (y : ℝ → ℝ)
  (h_y₁ : ∀ x, y₁ x = a * (x - x₁) * (x - x₂))
  (h_y₂ : ∃ e, ∀ x, y₂ x = d * x + e)
  (h_intersect : y₂ x₁ = 0)
  (h_single_root : ∃! x, y x = 0) :
  x₂ - x₁ = d / a := by sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l1387_138751


namespace NUMINAMATH_CALUDE_distinct_angles_in_twelve_sided_polygon_l1387_138793

/-- A circle with an inscribed regular pentagon and heptagon -/
structure InscribedPolygons where
  circle : Set ℝ × ℝ  -- Representing a circle in 2D plane
  pentagon : Set (ℝ × ℝ)  -- Vertices of the pentagon
  heptagon : Set (ℝ × ℝ)  -- Vertices of the heptagon

/-- The resulting 12-sided polygon -/
def twelveSidedPolygon (ip : InscribedPolygons) : Set (ℝ × ℝ) :=
  ip.pentagon ∪ ip.heptagon

/-- Predicate to check if two polygons have no common vertices -/
def noCommonVertices (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  p1 ∩ p2 = ∅

/-- Predicate to check if two polygons have no common axes of symmetry -/
def noCommonAxesOfSymmetry (p1 p2 : Set (ℝ × ℝ)) : Prop :=
  sorry  -- Definition omitted for brevity

/-- Function to count distinct angle values in a polygon -/
def countDistinctAngles (p : Set (ℝ × ℝ)) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The main theorem -/
theorem distinct_angles_in_twelve_sided_polygon
  (ip : InscribedPolygons)
  (h1 : noCommonVertices ip.pentagon ip.heptagon)
  (h2 : noCommonAxesOfSymmetry ip.pentagon ip.heptagon)
  : countDistinctAngles (twelveSidedPolygon ip) = 6 :=
sorry

end NUMINAMATH_CALUDE_distinct_angles_in_twelve_sided_polygon_l1387_138793


namespace NUMINAMATH_CALUDE_computer_multiplications_l1387_138743

/-- Represents the number of multiplications a computer can perform per minute -/
def multiplications_per_minute : ℕ := 25000

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 3

/-- Theorem stating that the computer will perform 4,500,000 multiplications in three hours -/
theorem computer_multiplications :
  multiplications_per_minute * minutes_per_hour * hours = 4500000 :=
by sorry

end NUMINAMATH_CALUDE_computer_multiplications_l1387_138743


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1387_138739

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 → (11 + 3*i) / (1 - 2*i) = 1 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1387_138739


namespace NUMINAMATH_CALUDE_polynomial_equality_l1387_138749

theorem polynomial_equality (q : Polynomial ℝ) :
  (q + (2 * X^4 - 5 * X^2 + 8 * X + 3) = 10 * X^3 - 7 * X^2 + 15 * X + 6) →
  q = -2 * X^4 + 10 * X^3 - 2 * X^2 + 7 * X + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1387_138749


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1387_138744

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (2 + I) / (1 - 2*I) → 
  z = Complex.mk a b → 
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1387_138744


namespace NUMINAMATH_CALUDE_distance_to_focus_l1387_138705

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola with x-coordinate 4
def point_on_parabola : {P : ℝ × ℝ // parabola P.1 P.2 ∧ P.1 = 4} :=
  sorry

-- Theorem statement
theorem distance_to_focus :
  let P := point_on_parabola.val
  (P.1 - 0)^2 = 4^2 →  -- Distance from P to y-axis is 4
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 6^2  -- Distance from P to focus is 6
:= by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1387_138705


namespace NUMINAMATH_CALUDE_positive_t_value_l1387_138796

theorem positive_t_value (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_positive_t_value_l1387_138796


namespace NUMINAMATH_CALUDE_committee_selection_ways_l1387_138719

def club_size : ℕ := 30
def committee_size : ℕ := 5

theorem committee_selection_ways :
  Nat.choose club_size committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l1387_138719


namespace NUMINAMATH_CALUDE_rajs_house_bathrooms_l1387_138737

/-- Represents the floor plan of Raj's house -/
structure HouseFloorPlan where
  total_area : ℕ
  bedroom_count : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bathrooms in Raj's house -/
def calculate_bathrooms (house : HouseFloorPlan) : ℕ :=
  let bedroom_area := house.bedroom_count * house.bedroom_side * house.bedroom_side
  let living_area := house.kitchen_area
  let remaining_area := house.total_area - (bedroom_area + house.kitchen_area + living_area)
  let bathroom_area := house.bathroom_length * house.bathroom_width
  remaining_area / bathroom_area

/-- Theorem stating that Raj's house has exactly 2 bathrooms -/
theorem rajs_house_bathrooms :
  let house : HouseFloorPlan := {
    total_area := 1110,
    bedroom_count := 4,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    kitchen_area := 265
  }
  calculate_bathrooms house = 2 := by
  sorry


end NUMINAMATH_CALUDE_rajs_house_bathrooms_l1387_138737


namespace NUMINAMATH_CALUDE_range_of_f_l1387_138799

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3,
  ∃ y ∈ Set.Ico 0 9,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Ico 0 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1387_138799


namespace NUMINAMATH_CALUDE_max_classes_less_than_1968_l1387_138775

/-- Relation between two natural numbers where they belong to the same class if one can be obtained from the other by deleting two adjacent digits or identical groups of digits -/
def SameClass (m n : ℕ) : Prop := sorry

/-- The maximum number of equivalence classes under the SameClass relation -/
def MaxClasses : ℕ := sorry

theorem max_classes_less_than_1968 : MaxClasses < 1968 := by sorry

end NUMINAMATH_CALUDE_max_classes_less_than_1968_l1387_138775


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1387_138770

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) :
  n = 15 →
  ∃ k : ℕ, (1 : ℚ) / 3^n = k / 10 + (0 : ℚ) / 10 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1387_138770


namespace NUMINAMATH_CALUDE_birthday_crayons_l1387_138700

theorem birthday_crayons (given_away lost remaining : ℕ) 
  (h1 : given_away = 111)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  given_away + lost + remaining = 440 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l1387_138700


namespace NUMINAMATH_CALUDE_teachers_class_size_l1387_138776

/-- The number of students in Teacher Yang's class -/
def num_students : ℕ := 28

theorem teachers_class_size :
  (num_students / 2 : ℕ) +     -- Half in math competition
  (num_students / 4 : ℕ) +     -- Quarter in music group
  (num_students / 7 : ℕ) +     -- One-seventh in reading room
  3 =                          -- Remaining three watching TV
  num_students :=              -- Equals total number of students
by sorry

end NUMINAMATH_CALUDE_teachers_class_size_l1387_138776


namespace NUMINAMATH_CALUDE_price_reduction_proof_optimal_price_increase_proof_l1387_138702

/-- Initial price in yuan per kilogram -/
def initial_price : ℝ := 50

/-- Final price after two reductions in yuan per kilogram -/
def final_price : ℝ := 32

/-- Initial profit in yuan per kilogram -/
def initial_profit : ℝ := 10

/-- Initial daily sales in kilograms -/
def initial_sales : ℝ := 500

/-- Maximum allowed price increase in yuan per kilogram -/
def max_price_increase : ℝ := 8

/-- Sales decrease per yuan of price increase in kilograms -/
def sales_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_profit : ℝ := 6000

/-- Percentage reduction after each price cut -/
def reduction_percentage : ℝ := 0.2

/-- Price increase to achieve target profit -/
def optimal_price_increase : ℝ := 5

theorem price_reduction_proof :
  initial_price * (1 - reduction_percentage)^2 = final_price :=
sorry

theorem optimal_price_increase_proof :
  (initial_profit + optimal_price_increase) * 
  (initial_sales - sales_decrease_rate * optimal_price_increase) = target_profit ∧
  0 < optimal_price_increase ∧
  optimal_price_increase ≤ max_price_increase :=
sorry

end NUMINAMATH_CALUDE_price_reduction_proof_optimal_price_increase_proof_l1387_138702


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_x_squared_l1387_138763

theorem max_value_x_sqrt_1_minus_x_squared :
  (∀ x : ℝ, x * Real.sqrt (1 - x^2) ≤ 1/2) ∧
  (∃ x : ℝ, x * Real.sqrt (1 - x^2) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_x_squared_l1387_138763


namespace NUMINAMATH_CALUDE_trapezoid_area_l1387_138711

/-- The area of a trapezoid with given vertices in a standard rectangular coordinate system -/
theorem trapezoid_area (E F G H : ℝ × ℝ) : 
  E = (2, -3) → 
  F = (2, 2) → 
  G = (7, 8) → 
  H = (7, 3) → 
  (1/2 : ℝ) * ((F.2 - E.2) + (G.2 - H.2)) * (G.1 - E.1) = 25 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l1387_138711


namespace NUMINAMATH_CALUDE_multiple_power_divisibility_l1387_138780

theorem multiple_power_divisibility (a n m : ℕ) (ha : a > 0) : 
  m % (a^n) = 0 → (a + 1)^m - 1 % (a^(n+1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_power_divisibility_l1387_138780


namespace NUMINAMATH_CALUDE_factorial_ratio_l1387_138794

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio : 
  factorial 10 / (factorial 5 * factorial 2) = 15120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1387_138794


namespace NUMINAMATH_CALUDE_corresponding_angles_relationships_l1387_138769

/-- Two angles are corresponding if they occupy the same relative position at each intersection where a straight line crosses two others. -/
def corresponding_angles (α β : Real) : Prop := sorry

/-- The statement that all relationships (equal, greater than, less than) are possible for corresponding angles. -/
theorem corresponding_angles_relationships (α β : Real) (h : corresponding_angles α β) :
  (∃ (α₁ β₁ : Real), corresponding_angles α₁ β₁ ∧ α₁ = β₁) ∧
  (∃ (α₂ β₂ : Real), corresponding_angles α₂ β₂ ∧ α₂ > β₂) ∧
  (∃ (α₃ β₃ : Real), corresponding_angles α₃ β₃ ∧ α₃ < β₃) :=
sorry

end NUMINAMATH_CALUDE_corresponding_angles_relationships_l1387_138769


namespace NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l1387_138777

/-- Lagrange interpolation polynomial for the given points -/
def P₂ (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 8

/-- The x-coordinates of the interpolation points -/
def x₀ : ℝ := -3
def x₁ : ℝ := -1
def x₂ : ℝ := 2

/-- The y-coordinates of the interpolation points -/
def y₀ : ℝ := -5
def y₁ : ℝ := -11
def y₂ : ℝ := 10

/-- Theorem stating that P₂ is the Lagrange interpolation polynomial for the given points -/
theorem lagrange_interpolation_polynomial :
  P₂ x₀ = y₀ ∧ P₂ x₁ = y₁ ∧ P₂ x₂ = y₂ := by
  sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l1387_138777


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1387_138785

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1387_138785


namespace NUMINAMATH_CALUDE_share_distribution_l1387_138721

theorem share_distribution (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 6600) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  (total * ratio1) / (ratio1 + ratio2 + ratio3) = 1100 := by
sorry

end NUMINAMATH_CALUDE_share_distribution_l1387_138721


namespace NUMINAMATH_CALUDE_expression_evaluation_l1387_138747

theorem expression_evaluation :
  101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1387_138747


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1387_138784

theorem min_sum_with_reciprocal_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a + b ≥ 3/2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l1387_138784


namespace NUMINAMATH_CALUDE_jack_stair_step_height_l1387_138714

/-- Given Jack's stair climbing scenario, prove the height of each step. -/
theorem jack_stair_step_height :
  -- Net flights descended
  ∀ (net_flights : ℕ),
  -- Steps per flight
  ∀ (steps_per_flight : ℕ),
  -- Total descent in inches
  ∀ (total_descent : ℕ),
  -- Given conditions
  net_flights = 3 →
  steps_per_flight = 12 →
  total_descent = 288 →
  -- Prove that the height of each step is 8 inches
  (total_descent : ℚ) / (net_flights * steps_per_flight : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_stair_step_height_l1387_138714


namespace NUMINAMATH_CALUDE_computer_price_problem_l1387_138706

theorem computer_price_problem (sticker_price : ℝ) : 
  (0.85 * sticker_price - 50 = 0.7 * sticker_price - 20) → 
  sticker_price = 200 := by
sorry

end NUMINAMATH_CALUDE_computer_price_problem_l1387_138706


namespace NUMINAMATH_CALUDE_negation_equivalence_l1387_138788

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∀ x : ℝ, x ≤ 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1387_138788


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l1387_138787

theorem farmer_tomatoes (initial : ℕ) (picked : ℕ) (difference : ℕ) 
  (h1 : picked = 9)
  (h2 : difference = 8)
  (h3 : initial - picked = difference) :
  initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l1387_138787


namespace NUMINAMATH_CALUDE_min_red_vertices_l1387_138726

/-- Given a square partitioned into n^2 unit squares, each divided into two triangles,
    the minimum number of red vertices needed to ensure each triangle has a red vertex is ⌈n^2/2⌉ -/
theorem min_red_vertices (n : ℕ) (h : n > 0) :
  ∃ (red_vertices : Finset (ℕ × ℕ)),
    (∀ i j : ℕ, i < n → j < n →
      (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ red_vertices)) ∧
    red_vertices.card = ⌈(n^2 : ℝ) / 2⌉ ∧
    (∀ rv : Finset (ℕ × ℕ), 
      (∀ i j : ℕ, i < n → j < n →
        (∃ k l : ℕ, (k = i ∨ k = i + 1) ∧ (l = j ∨ l = j + 1) ∧ (k, l) ∈ rv)) →
      rv.card ≥ ⌈(n^2 : ℝ) / 2⌉) := by
  sorry


end NUMINAMATH_CALUDE_min_red_vertices_l1387_138726


namespace NUMINAMATH_CALUDE_max_value_problem_l1387_138720

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * z * Real.sqrt 2 + 9 * y * z ≤ Real.sqrt 27 := by
sorry

end NUMINAMATH_CALUDE_max_value_problem_l1387_138720


namespace NUMINAMATH_CALUDE_root_product_l1387_138792

theorem root_product (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ∃ s : ℝ, ((c + 1/d)^2 - r*(c + 1/d) + s = 0) ∧ 
           ((d + 1/c)^2 - r*(d + 1/c) + s = 0) ∧ 
           (s = 16/3) := by
  sorry

end NUMINAMATH_CALUDE_root_product_l1387_138792


namespace NUMINAMATH_CALUDE_count_valid_configurations_l1387_138762

/-- Represents a configuration of 8's with + signs inserted -/
structure Configuration where
  singles : ℕ  -- number of individual 8's
  doubles : ℕ  -- number of 88's
  triples : ℕ  -- number of 888's

/-- The total number of 8's used in a configuration -/
def Configuration.total_eights (c : Configuration) : ℕ :=
  c.singles + 2 * c.doubles + 3 * c.triples

/-- The sum of a configuration -/
def Configuration.sum (c : Configuration) : ℕ :=
  8 * c.singles + 88 * c.doubles + 888 * c.triples

/-- A configuration is valid if its sum is 8880 -/
def Configuration.is_valid (c : Configuration) : Prop :=
  c.sum = 8880

theorem count_valid_configurations :
  (∃ (s : Finset ℕ), s.card = 119 ∧
    (∀ n, n ∈ s ↔ ∃ c : Configuration, c.is_valid ∧ c.total_eights = n)) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_configurations_l1387_138762


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1387_138765

/-- A quadrilateral in 2D space -/
structure Quadrilateral (V : Type*) [AddCommGroup V] :=
  (P Q R S : V)

/-- Extended points of a quadrilateral -/
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] extends Quadrilateral V :=
  (P' Q' R' S' : V)

/-- Condition that P, Q, R, S are midpoints of PP', QQ', RR', SS' respectively -/
def is_midpoint_quadrilateral {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) : Prop :=
  quad.P = (1/2 : ℚ) • (quad.P + ext_quad.P') ∧
  quad.Q = (1/2 : ℚ) • (quad.Q + ext_quad.Q') ∧
  quad.R = (1/2 : ℚ) • (quad.R + ext_quad.R') ∧
  quad.S = (1/2 : ℚ) • (quad.S + ext_quad.S')

/-- Main theorem -/
theorem quadrilateral_reconstruction {V : Type*} [AddCommGroup V] [Module ℚ V] 
  (quad : Quadrilateral V) (ext_quad : ExtendedQuadrilateral V) 
  (h : is_midpoint_quadrilateral quad ext_quad) :
  quad.P = (1/15 : ℚ) • ext_quad.P' + (2/15 : ℚ) • ext_quad.Q' + 
           (4/15 : ℚ) • ext_quad.R' + (8/15 : ℚ) • ext_quad.S' := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1387_138765


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1387_138757

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (length width : ℝ),
    r = 6 →
    length = 3 * width →
    width = 2 * r →
    length * width = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1387_138757


namespace NUMINAMATH_CALUDE_bat_wings_area_is_3_25_l1387_138723

/-- Rectangle DEFA with given dimensions and points -/
structure Rectangle where
  width : ℝ
  height : ℝ
  dc : ℝ
  cb : ℝ
  ba : ℝ

/-- Calculate the area of the "bat wings" in the given rectangle -/
def batWingsArea (rect : Rectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3.25 -/
theorem bat_wings_area_is_3_25 (rect : Rectangle) 
  (h1 : rect.width = 5)
  (h2 : rect.height = 3)
  (h3 : rect.dc = 2)
  (h4 : rect.cb = 1.5)
  (h5 : rect.ba = 1.5) :
  batWingsArea rect = 3.25 := by sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_3_25_l1387_138723


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_intersection_when_a_is_4_l1387_138713

/-- The set A depending on the parameter a -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}

/-- The set B depending on the parameter a -/
def B (a : ℝ) : Set ℝ := {x | x > 2*a ∧ x < a^2 + 2}

/-- The theorem stating the range of a -/
theorem range_of_a_for_subset (a : ℝ) : 
  (a > -3/2) → (B a ⊆ A a) → (1 ≤ a ∧ a ≤ 3) :=
sorry

/-- The theorem for the specific case when a = 4 -/
theorem intersection_when_a_is_4 : 
  A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_intersection_when_a_is_4_l1387_138713


namespace NUMINAMATH_CALUDE_semicircle_area_comparison_l1387_138789

theorem semicircle_area_comparison : 
  let rectangle_width : ℝ := 8
  let rectangle_length : ℝ := 12
  let small_semicircle_radius : ℝ := rectangle_width / 2
  let large_semicircle_radius : ℝ := rectangle_length / 2
  let small_semicircle_area : ℝ := π * small_semicircle_radius^2 / 2
  let large_semicircle_area : ℝ := π * large_semicircle_radius^2 / 2
  (large_semicircle_area / small_semicircle_area - 1) * 100 = 125 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_comparison_l1387_138789


namespace NUMINAMATH_CALUDE_matrix_N_property_l1387_138718

open Matrix

theorem matrix_N_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec ![3, -2] = ![4, 1])
  (h2 : N.mulVec ![-4, 6] = ![-2, 0]) :
  N.mulVec ![7, 0] = ![6, 2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_property_l1387_138718


namespace NUMINAMATH_CALUDE_alpine_ridge_length_l1387_138724

/-- Represents the Alpine Ridge Trail hike --/
structure AlpineRidgeTrail where
  /-- Distance hiked on each of the five days --/
  day : Fin 5 → ℝ
  /-- First three days total 30 miles --/
  first_three_days : day 0 + day 1 + day 2 = 30
  /-- Second and fourth days average 15 miles --/
  second_fourth_avg : (day 1 + day 3) / 2 = 15
  /-- Last two days total 28 miles --/
  last_two_days : day 3 + day 4 = 28
  /-- First and fourth days total 34 miles --/
  first_fourth_days : day 0 + day 3 = 34

/-- The total length of the Alpine Ridge Trail is 58 miles --/
theorem alpine_ridge_length (trail : AlpineRidgeTrail) : 
  trail.day 0 + trail.day 1 + trail.day 2 + trail.day 3 + trail.day 4 = 58 := by
  sorry


end NUMINAMATH_CALUDE_alpine_ridge_length_l1387_138724


namespace NUMINAMATH_CALUDE_f_minimum_value_l1387_138736

def f (x y : ℝ) : ℝ := (1 - y)^2 + (x + y - 3)^2 + (2*x + y - 6)^2

theorem f_minimum_value :
  ∀ x y : ℝ, f x y ≥ 1/6 ∧
  (f x y = 1/6 ↔ x = 17/4 ∧ y = 1/4) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1387_138736


namespace NUMINAMATH_CALUDE_clown_balloons_l1387_138768

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial final : ℕ) : ℕ := final - initial

/-- Theorem: Given the initial and final number of balloons, prove that the clown blew up 13 more balloons -/
theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1387_138768


namespace NUMINAMATH_CALUDE_seating_arrangements_seven_people_l1387_138703

/-- The number of ways to arrange n people around a circular table, considering rotations as identical -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n blocks around a circular table, with one block containing 3 fixed people -/
def arrangementsWithFixedBlock (n : ℕ) : ℕ := 
  circularArrangements (n - 2) * 2

theorem seating_arrangements_seven_people : 
  arrangementsWithFixedBlock 7 = 240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_seven_people_l1387_138703


namespace NUMINAMATH_CALUDE_sample_size_proof_l1387_138761

theorem sample_size_proof (n : ℕ) : 
  (∃ x : ℚ, 
    x > 0 ∧ 
    2*x + 3*x + 4*x + 6*x + 4*x + x = 1 ∧ 
    (2*x + 3*x + 4*x) * n = 27) → 
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l1387_138761


namespace NUMINAMATH_CALUDE_diana_charge_account_debt_l1387_138710

/-- Calculate the total amount owed after applying simple interest --/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem: Given the specified conditions, the total amount owed is $63.60 --/
theorem diana_charge_account_debt : 
  let principal : ℝ := 60
  let rate : ℝ := 0.06
  let time : ℝ := 1
  total_amount_owed principal rate time = 63.60 := by
  sorry

end NUMINAMATH_CALUDE_diana_charge_account_debt_l1387_138710


namespace NUMINAMATH_CALUDE_distance_proof_l1387_138701

/-- The distance between points A and B in kilometers -/
def distance : ℝ := 180

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- The velocity of the stream in km/h -/
def stream_velocity : ℝ := 4

/-- The speed of the boat in still water in km/h -/
def boat_speed : ℝ := 14

/-- The downstream speed of the boat in km/h -/
def downstream_speed : ℝ := boat_speed + stream_velocity

/-- The upstream speed of the boat in km/h -/
def upstream_speed : ℝ := boat_speed - stream_velocity

theorem distance_proof :
  distance / downstream_speed + (distance / 2) / upstream_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_distance_proof_l1387_138701


namespace NUMINAMATH_CALUDE_developed_countries_completed_transformation_l1387_138797

-- Define the different stages of population growth patterns
inductive PopulationGrowthStage
| Traditional
| Transitional
| Modern

-- Define the types of countries
inductive CountryType
| Developed
| Developing

-- Define the world population distribution
def worldPopulationDistribution : CountryType → Bool
| CountryType.Developing => true
| CountryType.Developed => false

-- Define the population growth stage for each country type
def populationGrowthStage : CountryType → PopulationGrowthStage
| CountryType.Developing => PopulationGrowthStage.Traditional
| CountryType.Developed => PopulationGrowthStage.Modern

-- Define the overall global population growth stage
def globalPopulationGrowthStage : PopulationGrowthStage :=
  PopulationGrowthStage.Transitional

-- Theorem statement
theorem developed_countries_completed_transformation :
  (∀ c : CountryType, worldPopulationDistribution c → populationGrowthStage c = PopulationGrowthStage.Traditional) →
  globalPopulationGrowthStage = PopulationGrowthStage.Transitional →
  populationGrowthStage CountryType.Developed = PopulationGrowthStage.Modern :=
by
  sorry

end NUMINAMATH_CALUDE_developed_countries_completed_transformation_l1387_138797


namespace NUMINAMATH_CALUDE_largest_base6_5digit_value_l1387_138754

def largest_base6_5digit : ℕ := 5 * 6^4 + 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

theorem largest_base6_5digit_value : largest_base6_5digit = 7775 := by
  sorry

end NUMINAMATH_CALUDE_largest_base6_5digit_value_l1387_138754


namespace NUMINAMATH_CALUDE_problem_1_l1387_138735

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1387_138735


namespace NUMINAMATH_CALUDE_marbles_left_theorem_l1387_138778

def initial_marbles : ℕ := 87
def marbles_given_away : ℕ := 8

theorem marbles_left_theorem : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_theorem_l1387_138778


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1387_138760

theorem arctan_sum_equals_pi_over_four (n : ℕ) :
  (n > 0) →
  (Real.arctan (1/2) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/n : ℝ) = π/4) →
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1387_138760


namespace NUMINAMATH_CALUDE_function_value_at_sine_l1387_138753

/-- Given a function f(x) = 4x² + 2x, prove that f(sin(7π/6)) = 0 -/
theorem function_value_at_sine (f : ℝ → ℝ) : 
  (∀ x, f x = 4 * x^2 + 2 * x) → f (Real.sin (7 * π / 6)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_sine_l1387_138753


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1387_138779

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 3 / 2)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 5) :
  w / y = 45 / 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1387_138779
