import Mathlib

namespace NUMINAMATH_CALUDE_three_lines_determine_plane_l92_9231

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if two lines intersect
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

-- Function to check if three lines intersect at the same point
def threeLinesSameIntersection (l1 l2 l3 : Line3D) : Prop := sorry

-- Function to determine if three lines define a unique plane
def defineUniquePlane (l1 l2 l3 : Line3D) : Prop := sorry

-- Theorem stating that three lines intersecting pairwise but not at the same point determine a unique plane
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  linesIntersect l1 l2 ∧ linesIntersect l2 l3 ∧ linesIntersect l3 l1 ∧
  ¬threeLinesSameIntersection l1 l2 l3 →
  defineUniquePlane l1 l2 l3 := by sorry

end NUMINAMATH_CALUDE_three_lines_determine_plane_l92_9231


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l92_9244

/-- A farmer picks tomatoes from his garden. -/
theorem farmer_tomatoes (initial : ℕ) (remaining : ℕ) (picked : ℕ)
    (h1 : initial = 97)
    (h2 : remaining = 14)
    (h3 : picked = initial - remaining) :
  picked = 83 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l92_9244


namespace NUMINAMATH_CALUDE_max_min_product_l92_9202

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 9 → 
  a * b + b * c + c * a = 27 → 
  ∀ m : ℝ, m = min (a * b) (min (b * c) (c * a)) → 
  m ≤ 6.75 ∧ ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 9 ∧ 
    a' * b' + b' * c' + c' * a' = 27 ∧ 
    min (a' * b') (min (b' * c') (c' * a')) = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_max_min_product_l92_9202


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l92_9225

theorem combined_mean_of_two_sets (set1_mean set2_mean : ℚ) :
  set1_mean = 18 →
  set2_mean = 16 →
  (7 * set1_mean + 8 * set2_mean) / 15 = 254 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l92_9225


namespace NUMINAMATH_CALUDE_remaining_donuts_l92_9270

theorem remaining_donuts (initial_donuts : ℕ) (missing_percentage : ℚ) 
  (h1 : initial_donuts = 30)
  (h2 : missing_percentage = 70/100) :
  ↑initial_donuts * (1 - missing_percentage) = 9 :=
by sorry

end NUMINAMATH_CALUDE_remaining_donuts_l92_9270


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l92_9233

theorem interest_rate_calculation (P r : ℝ) 
  (h1 : P * (1 + 4 * r) = 400)
  (h2 : P * (1 + 6 * r) = 500) :
  r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l92_9233


namespace NUMINAMATH_CALUDE_carpenter_tables_total_l92_9286

/-- The number of tables made this month -/
def tables_this_month : ℕ := 10

/-- The difference in tables made between this month and last month -/
def difference : ℕ := 3

/-- The number of tables made last month -/
def tables_last_month : ℕ := tables_this_month - difference

/-- The total number of tables made over two months -/
def total_tables : ℕ := tables_this_month + tables_last_month

theorem carpenter_tables_total :
  total_tables = 17 := by sorry

end NUMINAMATH_CALUDE_carpenter_tables_total_l92_9286


namespace NUMINAMATH_CALUDE_anatoliy_handshakes_l92_9289

theorem anatoliy_handshakes (n : ℕ) (total_handshakes : ℕ) : 
  total_handshakes = 197 →
  (n * (n - 1)) / 2 + 7 = total_handshakes →
  ∃ (k : ℕ), k = 7 ∧ k ≤ n ∧ (n * (n - 1)) / 2 + k = total_handshakes :=
by sorry

end NUMINAMATH_CALUDE_anatoliy_handshakes_l92_9289


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l92_9283

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^4 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l92_9283


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l92_9272

theorem tripled_base_and_exponent (a b : ℝ) (x : ℝ) (hx : x > 0) :
  (3*a)^(3*b) = a^b * x^b → x = 27 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l92_9272


namespace NUMINAMATH_CALUDE_virus_radius_scientific_notation_l92_9205

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The radius of the virus in meters -/
def virus_radius : ℝ := 0.00000000495

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_radius_scientific_notation :
  to_scientific_notation virus_radius = ScientificNotation.mk 4.95 (-9) (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_virus_radius_scientific_notation_l92_9205


namespace NUMINAMATH_CALUDE_max_value_theorem_l92_9217

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 1 ∧ 2 * a * b * Real.sqrt 3 + 2 * b * c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l92_9217


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l92_9277

def M : Set ℤ := {1, 2, 3, 4}
def N : Set ℤ := {-2, 2}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l92_9277


namespace NUMINAMATH_CALUDE_integer_expression_l92_9250

theorem integer_expression (m : ℤ) : ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l92_9250


namespace NUMINAMATH_CALUDE_paths_AC_count_l92_9236

/-- The number of paths from A to B -/
def paths_AB : Nat := 2

/-- The number of paths from B to C -/
def paths_BC : Nat := 2

/-- The number of direct paths from A to C -/
def direct_paths_AC : Nat := 1

/-- The total number of paths from A to C -/
def total_paths_AC : Nat := paths_AB * paths_BC + direct_paths_AC

theorem paths_AC_count : total_paths_AC = 5 := by
  sorry

end NUMINAMATH_CALUDE_paths_AC_count_l92_9236


namespace NUMINAMATH_CALUDE_y_to_x_equals_one_l92_9235

theorem y_to_x_equals_one (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_to_x_equals_one_l92_9235


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l92_9201

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x + 3)^2

-- Theorem statement
theorem axis_of_symmetry (x : ℝ) :
  (∀ h : ℝ, f (x + h) = f (x - h)) ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l92_9201


namespace NUMINAMATH_CALUDE_age_difference_proof_l92_9206

/-- Proves that the age difference between a man and his son is 34 years. -/
theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 32 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 34 := by
  sorry

#check age_difference_proof

end NUMINAMATH_CALUDE_age_difference_proof_l92_9206


namespace NUMINAMATH_CALUDE_complex_equation_sum_l92_9282

theorem complex_equation_sum (x y : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : x - 3 * i = (8 * x - y) * i) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l92_9282


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l92_9271

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ (a^1984 + b^1984 + c^1984 + d^1984 = x * y) := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l92_9271


namespace NUMINAMATH_CALUDE_crayon_selection_l92_9213

theorem crayon_selection (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 5) :
  (Nat.choose (n - 1) (k - 1)) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_l92_9213


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l92_9290

/-- A quadratic function that satisfies the given conditions -/
def f (x : ℝ) : ℝ := x^2

/-- The vertex of f is on the x-axis -/
theorem vertex_on_x_axis : ∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h :=
sorry

/-- f is rising on the right side of the y-axis -/
theorem rising_right_side : ∀ x > 0, ∀ y > x, f y > f x :=
sorry

/-- f is in its simplest form -/
theorem simplest_form : ∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0 :=
sorry

/-- f satisfies all the required conditions -/
theorem f_satisfies_conditions : 
  (∃ h : ℝ, f h = 0 ∧ ∀ x : ℝ, f x ≥ f h) ∧ 
  (∀ x > 0, ∀ y > x, f y > f x) ∧
  (∀ a b c : ℝ, (∀ x : ℝ, a * x^2 + b * x + c = f x) → a = 1 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_rising_right_side_simplest_form_f_satisfies_conditions_l92_9290


namespace NUMINAMATH_CALUDE_fraction_problem_l92_9280

theorem fraction_problem (f : ℝ) : 
  (f * 8.0 = 0.25 * 8.0 + 2) → f = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l92_9280


namespace NUMINAMATH_CALUDE_point_inside_circle_m_range_l92_9288

/-- A point (x, y) is inside a circle with center (a, b) and radius r if the square of the distance
    from the point to the center is less than r^2 -/
def IsInsideCircle (x y a b : ℝ) (r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 < r^2

theorem point_inside_circle_m_range :
  ∀ m : ℝ, IsInsideCircle 1 (-3) 2 (-1) (m^(1/2)) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_m_range_l92_9288


namespace NUMINAMATH_CALUDE_distance_walked_l92_9262

/-- Given a walking speed and a total walking time, calculate the distance walked. -/
theorem distance_walked (speed : ℝ) (time : ℝ) (h1 : speed = 1 / 15) (h2 : time = 45) :
  speed * time = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l92_9262


namespace NUMINAMATH_CALUDE_loss_percent_example_l92_9253

/-- Calculate the loss percent given the cost price and selling price -/
def loss_percent (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percent is 100/3% when an article is bought for 1200 and sold for 800 -/
theorem loss_percent_example : loss_percent 1200 800 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_loss_percent_example_l92_9253


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l92_9269

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, ¬(x ≥ a + 2 ∧ x < 3*a - 2)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l92_9269


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l92_9267

theorem imaginary_part_of_complex_product : Complex.im ((3 * Complex.I - 1) * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l92_9267


namespace NUMINAMATH_CALUDE_square_EFGH_side_length_l92_9238

/-- Square ABCD with side length 10 cm -/
def square_ABCD : Real := 10

/-- Distance of line p from side AB -/
def line_p_distance : Real := 6.5

/-- Area difference between the two parts divided by line p -/
def area_difference : Real := 13.8

/-- Side length of square EFGH -/
def square_EFGH_side : Real := 5.4

theorem square_EFGH_side_length :
  ∃ (square_EFGH : Real),
    square_EFGH = square_EFGH_side ∧
    square_EFGH > 0 ∧
    square_EFGH < square_ABCD ∧
    (square_ABCD - square_EFGH) * line_p_distance = area_difference / 2 ∧
    (square_ABCD - square_EFGH) * (square_ABCD - line_p_distance) = area_difference / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_EFGH_side_length_l92_9238


namespace NUMINAMATH_CALUDE_second_fund_interest_rate_l92_9228

/-- Proves that the interest rate of the second fund is 8.5% given the problem conditions --/
theorem second_fund_interest_rate : 
  ∀ (total_investment : ℝ) 
    (fund1_rate : ℝ) 
    (annual_interest : ℝ) 
    (fund1_investment : ℝ),
  total_investment = 50000 →
  fund1_rate = 8 →
  annual_interest = 4120 →
  fund1_investment = 26000 →
  ∃ (fund2_rate : ℝ),
    fund2_rate = 8.5 ∧
    annual_interest = (fund1_investment * fund1_rate / 100) + 
                      ((total_investment - fund1_investment) * fund2_rate / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_second_fund_interest_rate_l92_9228


namespace NUMINAMATH_CALUDE_milkman_A_grazing_period_l92_9251

/-- Represents the rental arrangement for a pasture shared by four milkmen. -/
structure PastureRental where
  /-- Number of cows grazed by milkman A -/
  cows_A : ℕ
  /-- Number of months milkman A grazed his cows (to be determined) -/
  months_A : ℕ
  /-- Number of cows grazed by milkman B -/
  cows_B : ℕ
  /-- Number of months milkman B grazed his cows -/
  months_B : ℕ
  /-- Number of cows grazed by milkman C -/
  cows_C : ℕ
  /-- Number of months milkman C grazed his cows -/
  months_C : ℕ
  /-- Number of cows grazed by milkman D -/
  cows_D : ℕ
  /-- Number of months milkman D grazed his cows -/
  months_D : ℕ
  /-- A's share of the rent in Rupees -/
  share_A : ℕ
  /-- Total rent of the field in Rupees -/
  total_rent : ℕ

/-- Theorem stating that given the conditions of the pasture rental,
    milkman A grazed his cows for 3 months. -/
theorem milkman_A_grazing_period (r : PastureRental)
  (h1 : r.cows_A = 24)
  (h2 : r.cows_B = 10)
  (h3 : r.months_B = 5)
  (h4 : r.cows_C = 35)
  (h5 : r.months_C = 4)
  (h6 : r.cows_D = 21)
  (h7 : r.months_D = 3)
  (h8 : r.share_A = 1440)
  (h9 : r.total_rent = 6500) :
  r.months_A = 3 := by
  sorry

end NUMINAMATH_CALUDE_milkman_A_grazing_period_l92_9251


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l92_9241

theorem absolute_value_equation_product (x : ℝ) : 
  (|15 / x + 4| = 3) → (∃ y : ℝ, (|15 / y + 4| = 3) ∧ (x * y = 225 / 7)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l92_9241


namespace NUMINAMATH_CALUDE_square_of_negative_half_a_squared_b_l92_9299

theorem square_of_negative_half_a_squared_b (a b : ℝ) :
  (- (1/2 : ℝ) * a^2 * b)^2 = (1/4 : ℝ) * a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_half_a_squared_b_l92_9299


namespace NUMINAMATH_CALUDE_kevins_age_exists_and_unique_l92_9296

theorem kevins_age_exists_and_unique :
  ∃! x : ℕ, 
    0 < x ∧ 
    x ≤ 120 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 := by
  sorry

end NUMINAMATH_CALUDE_kevins_age_exists_and_unique_l92_9296


namespace NUMINAMATH_CALUDE_fraction_equality_implies_division_l92_9287

theorem fraction_equality_implies_division (A B C : ℕ) : 
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C →
  1 - 1 / (6 + 1 / (6 + 1 / 6)) = 1 / (A + 1 / (B + 1 / C)) →
  (A + B) / C = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_division_l92_9287


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l92_9239

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌈x⌉ - x = 1 - (x - ⌊x⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l92_9239


namespace NUMINAMATH_CALUDE_expected_games_specific_l92_9243

/-- Represents a game with given win probabilities -/
structure Game where
  p_frank : ℝ  -- Probability of Frank winning a game
  p_joe : ℝ    -- Probability of Joe winning a game
  games_to_win : ℕ  -- Number of games needed to win the match

/-- Expected number of games in a match -/
def expected_games (g : Game) : ℝ := sorry

/-- Theorem stating the expected number of games in the specific scenario -/
theorem expected_games_specific :
  let g : Game := {
    p_frank := 0.3,
    p_joe := 0.7,
    games_to_win := 21
  }
  expected_games g = 30 := by sorry

end NUMINAMATH_CALUDE_expected_games_specific_l92_9243


namespace NUMINAMATH_CALUDE_interior_angles_sum_increase_l92_9208

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_sum_increase {n : ℕ} (h : sum_interior_angles n = 1800) :
  sum_interior_angles (n + 2) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_increase_l92_9208


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l92_9256

theorem max_value_of_trigonometric_expression :
  let y : ℝ → ℝ := λ x => Real.tan (x + 5 * Real.pi / 6) - Real.tan (x + Real.pi / 3) + Real.sin (x + Real.pi / 3)
  let max_value := (4 + Real.sqrt 3) / (2 * Real.sqrt 3)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x ≤ max_value ∧
  ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6), y x₀ = max_value := by
sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l92_9256


namespace NUMINAMATH_CALUDE_square_rectangle_area_ratio_l92_9204

/-- Represents a square with side length s -/
structure Square (s : ℝ) where
  area : ℝ := s^2

/-- Represents a rectangle with width w and height h -/
structure Rectangle (w h : ℝ) where
  area : ℝ := w * h

/-- The theorem statement -/
theorem square_rectangle_area_ratio 
  (s : ℝ) 
  (w h : ℝ) 
  (square : Square s) 
  (rect : Rectangle w h) 
  (h1 : rect.area = 0.25 * square.area) 
  (h2 : w = 8 * h) : 
  square.area / rect.area = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_ratio_l92_9204


namespace NUMINAMATH_CALUDE_min_sum_squares_l92_9219

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2 * a + 3 * b + 5 * c = 100) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 2 * x + 3 * y + 5 * z = 100 → 
  a^2 + b^2 + c^2 ≤ x^2 + y^2 + z^2 ∧ 
  a^2 + b^2 + c^2 = 5000 / 19 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l92_9219


namespace NUMINAMATH_CALUDE_existence_of_s_l92_9215

theorem existence_of_s (a : ℕ → ℕ) (k r : ℕ) (h1 : ∀ n m : ℕ, n ≤ m → a n ≤ a m) 
  (h2 : k > 0) (h3 : r > 0) (h4 : r = a r * (k + 1)) :
  ∃ s : ℕ, s > 0 ∧ s = a s * k := by
  sorry

end NUMINAMATH_CALUDE_existence_of_s_l92_9215


namespace NUMINAMATH_CALUDE_maggie_picked_40_apples_l92_9279

/-- The number of apples Kelsey picked -/
def kelsey_apples : ℕ := 28

/-- The number of apples Layla picked -/
def layla_apples : ℕ := 22

/-- The average number of apples picked by the three -/
def average_apples : ℕ := 30

/-- The number of people who picked apples -/
def num_people : ℕ := 3

/-- The number of apples Maggie picked -/
def maggie_apples : ℕ := 40

theorem maggie_picked_40_apples :
  kelsey_apples + layla_apples + maggie_apples = average_apples * num_people :=
by sorry

end NUMINAMATH_CALUDE_maggie_picked_40_apples_l92_9279


namespace NUMINAMATH_CALUDE_sheilas_weekly_earnings_l92_9276

/-- Sheila's weekly earnings calculation -/
theorem sheilas_weekly_earnings :
  let hourly_rate : ℕ := 12
  let hours_mon_wed_fri : ℕ := 8
  let hours_tue_thu : ℕ := 6
  let days_8_hours : ℕ := 3
  let days_6_hours : ℕ := 2
  let earnings_8_hour_days : ℕ := hourly_rate * hours_mon_wed_fri * days_8_hours
  let earnings_6_hour_days : ℕ := hourly_rate * hours_tue_thu * days_6_hours
  let total_earnings : ℕ := earnings_8_hour_days + earnings_6_hour_days
  total_earnings = 432 :=
by sorry

end NUMINAMATH_CALUDE_sheilas_weekly_earnings_l92_9276


namespace NUMINAMATH_CALUDE_three_card_sequence_l92_9257

-- Define the ranks and suits
inductive Rank
| Ace | King

inductive Suit
| Heart | Diamond

-- Define a card as a pair of rank and suit
structure Card :=
  (rank : Rank)
  (suit : Suit)

def is_king (c : Card) : Prop := c.rank = Rank.King
def is_ace (c : Card) : Prop := c.rank = Rank.Ace
def is_heart (c : Card) : Prop := c.suit = Suit.Heart
def is_diamond (c : Card) : Prop := c.suit = Suit.Diamond

-- Define the theorem
theorem three_card_sequence (c1 c2 c3 : Card) : 
  -- Condition 1
  (is_king c2 ∨ is_king c3) ∧ is_ace c1 →
  -- Condition 2
  (is_king c1 ∨ is_king c2) ∧ is_king c3 →
  -- Condition 3
  (is_heart c1 ∨ is_heart c2) ∧ is_diamond c3 →
  -- Condition 4
  is_heart c1 ∧ (is_heart c2 ∨ is_heart c3) →
  -- Conclusion
  is_heart c1 ∧ is_ace c1 ∧ 
  is_heart c2 ∧ is_king c2 ∧
  is_diamond c3 ∧ is_king c3 := by
  sorry


end NUMINAMATH_CALUDE_three_card_sequence_l92_9257


namespace NUMINAMATH_CALUDE_non_student_ticket_price_l92_9237

/-- Proves that the price of a non-student ticket was $8 -/
theorem non_student_ticket_price :
  let total_tickets : ℕ := 150
  let student_ticket_price : ℕ := 5
  let total_revenue : ℕ := 930
  let student_tickets_sold : ℕ := 90
  let non_student_tickets_sold : ℕ := 60
  let non_student_ticket_price : ℕ := (total_revenue - student_ticket_price * student_tickets_sold) / non_student_tickets_sold
  non_student_ticket_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_non_student_ticket_price_l92_9237


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_989_l92_9254

theorem largest_prime_factor_of_989 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 989 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 989 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_989_l92_9254


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_6_l92_9222

def is_divisible_by_6 (n : Nat) : Prop :=
  ∃ k : Nat, 7123 * 10 + n = 6 * k

theorem five_digit_divisible_by_6 :
  ∀ n : Nat, n < 10 →
    (is_divisible_by_6 n ↔ (n = 2 ∨ n = 8)) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_6_l92_9222


namespace NUMINAMATH_CALUDE_parametric_line_position_vector_l92_9200

/-- A line in a plane parameterized by t -/
structure ParametricLine where
  a : ℝ × ℝ  -- Point on the line
  d : ℝ × ℝ  -- Direction vector

/-- The position vector on a parametric line at a given t -/
def position_vector (line : ParametricLine) (t : ℝ) : ℝ × ℝ :=
  (line.a.1 + t * line.d.1, line.a.2 + t * line.d.2)

theorem parametric_line_position_vector :
  ∀ (line : ParametricLine),
    position_vector line 5 = (4, -1) →
    position_vector line (-1) = (-2, 13) →
    position_vector line 8 = (7, -8/3) := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_position_vector_l92_9200


namespace NUMINAMATH_CALUDE_ratio_equality_l92_9263

theorem ratio_equality : (2^2001 * 3^2003) / 6^2002 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l92_9263


namespace NUMINAMATH_CALUDE_skipping_rope_price_solution_l92_9268

def skipping_rope_prices (price_A price_B : ℚ) : Prop :=
  (price_B = price_A + 10) ∧
  (3150 / price_A = 3900 / price_B) ∧
  (price_A = 42) ∧
  (price_B = 52)

theorem skipping_rope_price_solution :
  ∃ (price_A price_B : ℚ), skipping_rope_prices price_A price_B :=
sorry

end NUMINAMATH_CALUDE_skipping_rope_price_solution_l92_9268


namespace NUMINAMATH_CALUDE_pipe_fill_time_l92_9209

/-- Given pipes P, Q, and R that can fill a tank, this theorem proves the time it takes for pipe P to fill the tank. -/
theorem pipe_fill_time (fill_rate_Q : ℝ) (fill_rate_R : ℝ) (fill_rate_all : ℝ) 
  (hQ : fill_rate_Q = 1 / 9)
  (hR : fill_rate_R = 1 / 18)
  (hAll : fill_rate_all = 1 / 2)
  (h_sum : ∃ (fill_rate_P : ℝ), fill_rate_P + fill_rate_Q + fill_rate_R = fill_rate_all) :
  ∃ (fill_time_P : ℝ), fill_time_P = 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l92_9209


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l92_9298

def original_price : ℝ := 760
def discount_percentage : ℝ := 75

theorem discounted_price_theorem :
  original_price * (1 - discount_percentage / 100) = 570 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l92_9298


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l92_9240

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+), 
    (∀ x : ℝ, x > 0 → x^2 + 10*x = 34 ↔ x = Real.sqrt a - b) ∧
    a + b = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l92_9240


namespace NUMINAMATH_CALUDE_students_behind_hoseok_l92_9252

/-- Given a line of students with the following properties:
  * There are 20 students in total
  * 11 students are in front of Yoongi
  * Hoseok is directly behind Yoongi
  Prove that there are 7 students behind Hoseok -/
theorem students_behind_hoseok (total : ℕ) (front_yoongi : ℕ) (hoseok_pos : ℕ) : 
  total = 20 → front_yoongi = 11 → hoseok_pos = front_yoongi + 2 → 
  total - hoseok_pos = 7 := by sorry

end NUMINAMATH_CALUDE_students_behind_hoseok_l92_9252


namespace NUMINAMATH_CALUDE_clock_hand_position_l92_9232

/-- Represents the angle of the hour hand in degrees -/
def hour_angle (hours : ℕ) : ℝ := (hours * 30 : ℝ)

/-- Theorem: For a clock with radius 15 units, when the hour hand points to 7 hours,
    the cosine of the angle is -√3/2 and the horizontal displacement of the tip
    of the hour hand from the center is -15√3/2 units. -/
theorem clock_hand_position (radius : ℝ) (hours : ℕ) 
    (h1 : radius = 15)
    (h2 : hours = 7) :
    let angle := hour_angle hours
    (Real.cos angle = -Real.sqrt 3 / 2) ∧ 
    (radius * Real.cos angle = -15 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_clock_hand_position_l92_9232


namespace NUMINAMATH_CALUDE_runner_b_lap_time_l92_9247

/-- A runner on a circular track -/
structure Runner where
  lap_time : ℝ
  speed : ℝ

/-- The circular track -/
structure Track where
  circumference : ℝ

/-- The scenario of two runners on a circular track -/
structure RunningScenario where
  track : Track
  runner_a : Runner
  runner_b : Runner
  meeting_time : ℝ
  b_time_to_start : ℝ

/-- The theorem stating that under given conditions, runner B takes 12 minutes to complete a lap -/
theorem runner_b_lap_time (scenario : RunningScenario) :
  scenario.runner_a.lap_time = 6 ∧
  scenario.b_time_to_start = 8 ∧
  scenario.runner_a.speed = scenario.track.circumference / scenario.runner_a.lap_time ∧
  scenario.runner_b.speed = scenario.track.circumference / scenario.runner_b.lap_time ∧
  scenario.meeting_time = scenario.track.circumference / (scenario.runner_a.speed + scenario.runner_b.speed) ∧
  scenario.runner_b.lap_time = scenario.meeting_time + scenario.b_time_to_start
  →
  scenario.runner_b.lap_time = 12 := by
sorry

end NUMINAMATH_CALUDE_runner_b_lap_time_l92_9247


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l92_9203

theorem divisibility_by_eight (a : ℤ) (h : Even a) :
  (∃ k : ℤ, a * (a^2 + 20) = 8 * k) ∧
  (∃ l : ℤ, a * (a^2 - 20) = 8 * l) ∧
  (∃ m : ℤ, a * (a^2 - 4) = 8 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l92_9203


namespace NUMINAMATH_CALUDE_min_value_abc_l92_9212

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1 / 2) :
  ∃ (min : ℝ), min = 18 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x * y * z = 1 / 2 →
    x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l92_9212


namespace NUMINAMATH_CALUDE_daniel_goats_count_l92_9221

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

theorem daniel_goats_count :
  num_goats * legs_per_animal + 
  num_horses * legs_per_animal + 
  num_dogs * legs_per_animal + 
  num_cats * legs_per_animal + 
  num_turtles * legs_per_animal = total_legs :=
by sorry

end NUMINAMATH_CALUDE_daniel_goats_count_l92_9221


namespace NUMINAMATH_CALUDE_hill_distance_l92_9210

theorem hill_distance (speed_up speed_down : ℝ) (total_time : ℝ) 
  (h1 : speed_up = 1.5)
  (h2 : speed_down = 4.5)
  (h3 : total_time = 6) :
  ∃ d : ℝ, d = 6.75 ∧ d / speed_up + d / speed_down = total_time :=
sorry

end NUMINAMATH_CALUDE_hill_distance_l92_9210


namespace NUMINAMATH_CALUDE_no_prime_pairs_with_integer_ratios_l92_9255

theorem no_prime_pairs_with_integer_ratios : 
  ¬ ∃ (x y : ℕ), Prime x ∧ Prime y ∧ y < x ∧ x ≤ 200 ∧ 
  (x / y : ℚ).isInt ∧ ((x + 1) / (y + 1) : ℚ).isInt := by
  sorry

end NUMINAMATH_CALUDE_no_prime_pairs_with_integer_ratios_l92_9255


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l92_9234

def polynomial (x : ℝ) : ℝ := x^4 + 6*x^3 + 11*x^2 + 7*x + 5

theorem root_sum_reciprocal_products (p q r s : ℝ) :
  polynomial p = 0 → polynomial q = 0 → polynomial r = 0 → polynomial s = 0 →
  (1 / (p * q)) + (1 / (p * r)) + (1 / (p * s)) + (1 / (q * r)) + (1 / (q * s)) + (1 / (r * s)) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l92_9234


namespace NUMINAMATH_CALUDE_salad_total_calories_l92_9275

/-- Represents the total calories in a salad. -/
def saladCalories (lettuce_cal : ℕ) (cucumber_cal : ℕ) (crouton_count : ℕ) (crouton_cal : ℕ) : ℕ :=
  lettuce_cal + cucumber_cal + crouton_count * crouton_cal

/-- Proves that the total calories in the salad is 350. -/
theorem salad_total_calories :
  saladCalories 30 80 12 20 = 350 := by
  sorry

end NUMINAMATH_CALUDE_salad_total_calories_l92_9275


namespace NUMINAMATH_CALUDE_blocks_between_39_and_40_l92_9218

/-- Represents the number of blocks in the original tower -/
def original_tower_size : ℕ := 90

/-- Represents the number of blocks taken at a time to build the new tower -/
def blocks_per_group : ℕ := 3

/-- Calculates the group number for a given block number in the original tower -/
def group_number (block : ℕ) : ℕ :=
  (original_tower_size - block) / blocks_per_group + 1

/-- Calculates the position of a block within its group in the new tower -/
def position_in_group (block : ℕ) : ℕ :=
  (original_tower_size - block) % blocks_per_group + 1

/-- Theorem stating that there are 4 blocks between blocks 39 and 40 in the new tower -/
theorem blocks_between_39_and_40 :
  ∃ (a b c d : ℕ),
    group_number 39 = group_number a ∧
    group_number 39 = group_number b ∧
    group_number 40 = group_number c ∧
    group_number 40 = group_number d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧
    position_in_group 39 < position_in_group a ∧
    position_in_group a < position_in_group b ∧
    position_in_group b < position_in_group c ∧
    position_in_group c < position_in_group d ∧
    position_in_group d < position_in_group 40 :=
by
  sorry

end NUMINAMATH_CALUDE_blocks_between_39_and_40_l92_9218


namespace NUMINAMATH_CALUDE_circle_common_chord_l92_9226

theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) → 
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_circle_common_chord_l92_9226


namespace NUMINAMATH_CALUDE_square_area_five_equal_rectangles_l92_9261

/-- A square divided into five rectangles of equal area, where one rectangle has a width of 5, has a total area of 400. -/
theorem square_area_five_equal_rectangles (s : ℝ) (w : ℝ) : 
  s > 0 ∧ w > 0 ∧ w = 5 ∧ 
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  a = b ∧ b = c ∧ c = d ∧ d = e ∧
  s * s = a + b + c + d + e ∧
  w * (s - w) = a →
  s * s = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_five_equal_rectangles_l92_9261


namespace NUMINAMATH_CALUDE_non_red_percentage_is_27_percent_l92_9274

/-- Represents the car population data for a city --/
structure CarPopulation where
  total : ℕ
  honda : ℕ
  toyota : ℕ
  nissan : ℕ
  honda_red_ratio : ℚ
  toyota_red_ratio : ℚ
  nissan_red_ratio : ℚ

/-- Calculate the percentage of non-red cars in the given car population --/
def non_red_percentage (pop : CarPopulation) : ℚ :=
  let total_red := pop.honda * pop.honda_red_ratio +
                   pop.toyota * pop.toyota_red_ratio +
                   pop.nissan * pop.nissan_red_ratio
  let total_non_red := pop.total - total_red
  (total_non_red / pop.total) * 100

/-- The theorem stating that the percentage of non-red cars is 27% --/
theorem non_red_percentage_is_27_percent (pop : CarPopulation)
  (h1 : pop.total = 30000)
  (h2 : pop.honda = 12000)
  (h3 : pop.toyota = 10000)
  (h4 : pop.nissan = 8000)
  (h5 : pop.honda_red_ratio = 80 / 100)
  (h6 : pop.toyota_red_ratio = 75 / 100)
  (h7 : pop.nissan_red_ratio = 60 / 100) :
  non_red_percentage pop = 27 := by
  sorry

end NUMINAMATH_CALUDE_non_red_percentage_is_27_percent_l92_9274


namespace NUMINAMATH_CALUDE_total_distance_is_75_miles_l92_9292

/-- Calculates the total distance traveled given initial speed and time, where the second part of the journey is twice as long at twice the speed. -/
def totalDistance (initialSpeed : ℝ) (initialTime : ℝ) : ℝ :=
  let distance1 := initialSpeed * initialTime
  let distance2 := (2 * initialSpeed) * (2 * initialTime)
  distance1 + distance2

/-- Proves that given an initial speed of 30 mph and an initial time of 0.5 hours, the total distance traveled is 75 miles. -/
theorem total_distance_is_75_miles :
  totalDistance 30 0.5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_75_miles_l92_9292


namespace NUMINAMATH_CALUDE_cosine_product_equality_l92_9260

theorem cosine_product_equality : 
  3.416 * Real.cos (π/33) * Real.cos (2*π/33) * Real.cos (4*π/33) * Real.cos (8*π/33) * Real.cos (16*π/33) = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equality_l92_9260


namespace NUMINAMATH_CALUDE_four_propositions_correct_l92_9242

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define symmetry about a point
def SymmetricAboutPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x) + 2 * b

-- Define symmetry about a line
def SymmetricAboutLine (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Define periodicity
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem four_propositions_correct (f : ℝ → ℝ) :
  (IsOdd f → SymmetricAboutPoint (fun x => f (x - 1)) 1 0) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 → IsEven f) ∧
  ((∀ x, f (x - 1) = -f x) → HasPeriod f 2) ∧
  (SymmetricAboutLine (fun x => f (x - 1)) 1 ∧ SymmetricAboutLine (fun x => f (1 - x)) 1) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_correct_l92_9242


namespace NUMINAMATH_CALUDE_range_of_x_plus_3y_l92_9211

theorem range_of_x_plus_3y (x y : ℝ) 
  (h1 : -1 ≤ x + y ∧ x + y ≤ 4) 
  (h2 : 2 ≤ x - y ∧ x - y ≤ 3) : 
  -5 ≤ x + 3*y ∧ x + 3*y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_3y_l92_9211


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l92_9258

theorem geometric_series_first_term 
  (r : ℚ) (S : ℚ) (h1 : r = -3/7) (h2 : S = 20) :
  S = a / (1 - r) → a = 200/7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l92_9258


namespace NUMINAMATH_CALUDE_largest_prime_factor_l92_9265

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (17^4 + 2*17^3 + 17^2 - 16^4) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (17^4 + 2*17^3 + 17^2 - 16^4) → q ≤ p :=
by
  use 17
  sorry

#check largest_prime_factor

end NUMINAMATH_CALUDE_largest_prime_factor_l92_9265


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l92_9284

/-- A parabola is defined by the equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope (p : Parabola) (x : ℝ) : ℝ := 2 * p.a * x + p.b

theorem parabola_unique_coefficients : 
  ∀ p : Parabola, 
    p.y 1 = 1 → 
    p.y 2 = -1 → 
    p.slope 2 = 1 → 
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l92_9284


namespace NUMINAMATH_CALUDE_max_cookies_andy_l92_9264

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : Nat
  alexa : Nat
  john : Nat

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.andy + d.alexa + d.john = 36 ∧
  d.andy % d.alexa = 0 ∧
  d.andy % d.john = 0 ∧
  d.alexa > 0 ∧
  d.john > 0

/-- Theorem stating the maximum number of cookies Andy could have eaten -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l92_9264


namespace NUMINAMATH_CALUDE_this_is_2345_l92_9297

def letter_to_digit : Char → Nat
| 'M' => 0
| 'A' => 1
| 'T' => 2
| 'H' => 3
| 'I' => 4
| 'S' => 5
| 'F' => 6
| 'U' => 7
| 'N' => 8
| _ => 9  -- Default case for completeness

def code_to_number (code : List Char) : Nat :=
  code.foldl (fun acc d => acc * 10 + letter_to_digit d) 0

theorem this_is_2345 :
  code_to_number ['T', 'H', 'I', 'S'] = 2345 := by
  sorry

end NUMINAMATH_CALUDE_this_is_2345_l92_9297


namespace NUMINAMATH_CALUDE_rakesh_cash_calculation_l92_9246

/-- Calculates the cash in hand after fixed deposit and grocery expenses --/
def cash_in_hand (salary : ℚ) (fixed_deposit_rate : ℚ) (grocery_rate : ℚ) : ℚ :=
  let fixed_deposit := salary * fixed_deposit_rate
  let remaining := salary - fixed_deposit
  let groceries := remaining * grocery_rate
  remaining - groceries

/-- Proves that given the conditions, the cash in hand is 2380 --/
theorem rakesh_cash_calculation :
  cash_in_hand 4000 (15/100) (30/100) = 2380 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_cash_calculation_l92_9246


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l92_9285

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l92_9285


namespace NUMINAMATH_CALUDE_jonathan_typing_time_l92_9229

/-- Represents the time it takes for Jonathan to type the document alone -/
def jonathan_time : ℝ := 40

/-- Represents the time it takes for Susan to type the document alone -/
def susan_time : ℝ := 30

/-- Represents the time it takes for Jack to type the document alone -/
def jack_time : ℝ := 24

/-- Represents the time it takes for all three to type the document together -/
def combined_time : ℝ := 10

/-- Theorem stating that Jonathan's individual typing time satisfies the given conditions -/
theorem jonathan_typing_time :
  1 / jonathan_time + 1 / susan_time + 1 / jack_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_jonathan_typing_time_l92_9229


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_angle_and_k_permissible_k_values_l92_9223

-- Define the cone and sphere
structure ConeWithInscribedSphere where
  R : ℝ  -- radius of the cone's base
  α : ℝ  -- angle between slant height and base plane
  k : ℝ  -- ratio of cone volume to sphere volume

-- Define the theorem
theorem cone_sphere_ratio_angle_and_k (c : ConeWithInscribedSphere) :
  c.k ≥ 2 →
  c.α = 2 * Real.arctan (Real.sqrt ((c.k + Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) ∨
  c.α = 2 * Real.arctan (Real.sqrt ((c.k - Real.sqrt (c.k^2 - 2*c.k)) / (2*c.k))) :=
by sorry

-- Define the permissible values of k
theorem permissible_k_values (c : ConeWithInscribedSphere) :
  c.k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_angle_and_k_permissible_k_values_l92_9223


namespace NUMINAMATH_CALUDE_pipe_b_fill_time_l92_9295

/-- Given a tank and three pipes A, B, and C, prove that pipe B fills the tank in 4 hours. -/
theorem pipe_b_fill_time (fill_time_A fill_time_B empty_time_C all_pipes_time : ℝ) 
  (h1 : fill_time_A = 3)
  (h2 : empty_time_C = 4)
  (h3 : all_pipes_time = 3.000000000000001)
  (h4 : 1 / fill_time_A + 1 / fill_time_B - 1 / empty_time_C = 1 / all_pipes_time) :
  fill_time_B = 4 := by
sorry

end NUMINAMATH_CALUDE_pipe_b_fill_time_l92_9295


namespace NUMINAMATH_CALUDE_root_difference_l92_9220

-- Define the equation
def equation (r : ℝ) : Prop :=
  (r^2 - 5*r - 20) / (r - 2) = 2*r + 7

-- Define the roots of the equation
def roots : Set ℝ :=
  {r : ℝ | equation r}

-- Theorem statement
theorem root_difference : ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 4 :=
sorry

end NUMINAMATH_CALUDE_root_difference_l92_9220


namespace NUMINAMATH_CALUDE_circles_separate_l92_9278

theorem circles_separate (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂) :
  (∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧
   ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) →
  R₁ + R₂ = d ∧ d > R₁ + R₂ := by
sorry

end NUMINAMATH_CALUDE_circles_separate_l92_9278


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_beta_l92_9227

theorem sin_2alpha_plus_beta (p α β : ℝ) : 
  (∀ x, x^2 - 4*p*x - 2 = 1 → x = Real.tan α ∨ x = Real.tan β) →
  Real.sin (2 * (α + β)) = (2 * p) / (p^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_beta_l92_9227


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l92_9293

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 20) 
  (h2 : x + 4 * y = 16) : 
  17 * x^2 + 20 * x * y + 17 * y^2 = 656 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l92_9293


namespace NUMINAMATH_CALUDE_certain_number_proof_l92_9216

theorem certain_number_proof : ∃ x : ℝ, 0.80 * x = (4/5 * 20) + 16 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l92_9216


namespace NUMINAMATH_CALUDE_inverse_proposition_is_false_l92_9248

theorem inverse_proposition_is_false : 
  ¬(∀ a : ℝ, |a| = |6| → a = 6) := by
sorry

end NUMINAMATH_CALUDE_inverse_proposition_is_false_l92_9248


namespace NUMINAMATH_CALUDE_school_journey_problem_l92_9249

/-- Represents the time taken for John's journey to and from school -/
structure SchoolJourney where
  road_one_way : ℕ        -- Time taken to walk one way by road
  shortcut_one_way : ℕ    -- Time taken to walk one way by shortcut

/-- The theorem representing John's school journey problem -/
theorem school_journey_problem (j : SchoolJourney) 
  (h1 : j.road_one_way + j.shortcut_one_way = 50)  -- Road + Shortcut = 50 minutes
  (h2 : 2 * j.shortcut_one_way = 30)               -- Shortcut both ways = 30 minutes
  : 2 * j.road_one_way = 70 := by                  -- Road both ways = 70 minutes
  sorry

#check school_journey_problem

end NUMINAMATH_CALUDE_school_journey_problem_l92_9249


namespace NUMINAMATH_CALUDE_car_travel_distance_l92_9214

/-- Proves that Car X travels 98 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start_time : ℝ) : 
  speed_x = 35 →
  speed_y = 50 →
  head_start_time = 72 / 60 →
  ∃ (travel_time : ℝ), 
    travel_time > 0 ∧
    speed_x * (head_start_time + travel_time) = speed_y * travel_time ∧
    speed_x * travel_time = 98 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l92_9214


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l92_9224

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (|x| < 1 → x < a) ∧ ¬(x < a → |x| < 1)) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l92_9224


namespace NUMINAMATH_CALUDE_trailing_zeros_remainder_l92_9230

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the product of factorials from 1 to 120
def productOfFactorials : ℕ := (List.range 120).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define the function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

-- Theorem statement
theorem trailing_zeros_remainder :
  (trailingZeros productOfFactorials) % 1000 = 224 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_remainder_l92_9230


namespace NUMINAMATH_CALUDE_age_difference_l92_9259

/-- The difference in ages between two people given a ratio and one person's age -/
theorem age_difference (sachin_age rahul_age : ℝ) : 
  sachin_age = 24.5 → 
  sachin_age / rahul_age = 7 / 9 → 
  rahul_age - sachin_age = 7 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l92_9259


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l92_9245

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l92_9245


namespace NUMINAMATH_CALUDE_f_intersects_negative_axes_l92_9294

def f (x : ℝ) : ℝ := -x - 1

theorem f_intersects_negative_axes :
  (∃ x, x < 0 ∧ f x = 0) ∧ (∃ y, y < 0 ∧ f 0 = y) := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_negative_axes_l92_9294


namespace NUMINAMATH_CALUDE_draw_three_cards_not_same_color_l92_9266

/-- Given a set of 16 cards with 4 of each color (red, yellow, blue, green),
    this theorem states that the number of ways to draw 3 cards such that
    they are not all the same color is equal to C(16,3) - 4 * C(4,3). -/
theorem draw_three_cards_not_same_color (total_cards : ℕ) (cards_per_color : ℕ) 
  (num_colors : ℕ) (draw : ℕ) (h1 : total_cards = 16) (h2 : cards_per_color = 4) 
  (h3 : num_colors = 4) (h4 : draw = 3) :
  (Nat.choose total_cards draw) - (num_colors * Nat.choose cards_per_color draw) = 544 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_cards_not_same_color_l92_9266


namespace NUMINAMATH_CALUDE_min_value_expression_l92_9281

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 2/3) : 
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l92_9281


namespace NUMINAMATH_CALUDE_system_solution_condition_l92_9291

theorem system_solution_condition (a : ℝ) :
  (∃ (x y b : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) →
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l92_9291


namespace NUMINAMATH_CALUDE_xian_temp_difference_l92_9273

/-- Given the highest and lowest temperatures on a day, calculate the maximum temperature difference. -/
def max_temp_difference (highest lowest : ℝ) : ℝ :=
  highest - lowest

/-- Theorem: The maximum temperature difference on January 1, 2008 in Xi'an was 6°C. -/
theorem xian_temp_difference :
  let highest : ℝ := 3
  let lowest : ℝ := -3
  max_temp_difference highest lowest = 6 := by
  sorry

end NUMINAMATH_CALUDE_xian_temp_difference_l92_9273


namespace NUMINAMATH_CALUDE_function_divisibility_property_l92_9207

theorem function_divisibility_property (f : ℕ → ℕ) :
  (∀ x y : ℕ, (f x + f y) ∣ (x^2 - y^2)) →
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_property_l92_9207
