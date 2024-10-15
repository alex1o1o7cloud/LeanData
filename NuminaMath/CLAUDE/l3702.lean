import Mathlib

namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l3702_370284

theorem product_base_8_units_digit : 
  (123 * 58) % 8 = 6 := by sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l3702_370284


namespace NUMINAMATH_CALUDE_no_good_filling_for_1399_l3702_370232

theorem no_good_filling_for_1399 :
  ¬ ∃ (f : Fin 1399 → Fin 2798), 
    (∀ i : Fin 1399, f i ≠ f (i + 1)) ∧ 
    (∀ i j : Fin 1399, i ≠ j → f i ≠ f j) ∧
    (∀ i : Fin 1399, (f i.succ - f i) % 2798 = i.val + 1) :=
by
  sorry

#check no_good_filling_for_1399

end NUMINAMATH_CALUDE_no_good_filling_for_1399_l3702_370232


namespace NUMINAMATH_CALUDE_tangent_slope_at_1_l3702_370293

/-- The function f(x) = (x-2)(x^2+c) has an extremum at x=2 -/
def has_extremum_at_2 (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2 ∨ f x ≥ f 2

/-- The main theorem -/
theorem tangent_slope_at_1 (c : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - 2) * (x^2 + c)
  has_extremum_at_2 f c → (deriv f) 1 = -5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_1_l3702_370293


namespace NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3702_370288

open Real

theorem intersection_perpendicular_tangents (a : ℝ) :
  ∃ x ∈ Set.Ioo 0 (π / 2),
    2 * sin x = a * cos x ∧
    (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_tangents_l3702_370288


namespace NUMINAMATH_CALUDE_no_solution_for_socks_l3702_370271

theorem no_solution_for_socks : ¬∃ (n m : ℕ), n + m = 2009 ∧ (n^2 - 2*n*m + m^2) = (n + m) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_socks_l3702_370271


namespace NUMINAMATH_CALUDE_blanch_dinner_slices_l3702_370265

/-- Calculates the number of pizza slices eaten for dinner given the initial number of slices and consumption throughout the day. -/
def pizza_slices_for_dinner (initial_slices breakfast_slices lunch_slices snack_slices remaining_slices : ℕ) : ℕ :=
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + remaining_slices)

/-- Proves that Blanch ate 5 slices of pizza for dinner given the conditions of the problem. -/
theorem blanch_dinner_slices :
  pizza_slices_for_dinner 15 4 2 2 2 = 5 := by
  sorry

#eval pizza_slices_for_dinner 15 4 2 2 2

end NUMINAMATH_CALUDE_blanch_dinner_slices_l3702_370265


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l3702_370225

/-- Given a wire of length 70 cm cut into two pieces, where the shorter piece is 27.999999999999993 cm long,
    prove that the ratio of the shorter piece to the longer piece is 2:3. -/
theorem wire_ratio_proof (total_length : ℝ) (shorter_piece : ℝ) (longer_piece : ℝ) :
  total_length = 70 →
  shorter_piece = 27.999999999999993 →
  longer_piece = total_length - shorter_piece →
  (shorter_piece / longer_piece) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l3702_370225


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3702_370220

theorem chess_tournament_games (n : ℕ) (h : n = 50) : 
  (n * (n - 1)) / 2 = 1225 :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3702_370220


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3702_370281

/-- Given a hyperbola with equation x²/(2m) - y²/m = 1, if one of its asymptotes
    has the equation y = 1, then m = -3 -/
theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2/(2*m) - y^2/m = 1) →
  (∃ y : ℝ → ℝ, y = λ _ => 1) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3702_370281


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3702_370266

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6 →
  a 1 * a 15 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3702_370266


namespace NUMINAMATH_CALUDE_cafe_round_trip_time_l3702_370272

/-- Represents a walking journey with constant pace -/
structure Walk where
  time : ℝ  -- Time in minutes
  distance : ℝ  -- Distance in miles
  pace : ℝ  -- Pace in minutes per mile

/-- Represents a location of a cafe relative to a full journey -/
structure CafeLocation where
  fraction : ℝ  -- Fraction of the full journey where the cafe is located

theorem cafe_round_trip_time 
  (full_walk : Walk) 
  (cafe : CafeLocation) 
  (h1 : full_walk.time = 30) 
  (h2 : full_walk.distance = 3) 
  (h3 : full_walk.pace = full_walk.time / full_walk.distance) 
  (h4 : cafe.fraction = 1/2) : 
  2 * (cafe.fraction * full_walk.distance * full_walk.pace) = 30 := by
sorry

end NUMINAMATH_CALUDE_cafe_round_trip_time_l3702_370272


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3702_370212

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∀ (a b : ℝ), 4 < a ∧ a < 6 → 6 < b ∧ b < 10 → ⌊b - a⌋ ≤ 4) ∧ 
  (∃ (a b : ℝ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ ⌊b - a⌋ = 4) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3702_370212


namespace NUMINAMATH_CALUDE_count_multiples_of_seven_perfect_squares_l3702_370257

theorem count_multiples_of_seven_perfect_squares : 
  let lower_bound := 10^6
  let upper_bound := 10^9
  (Finset.range (Nat.floor (Real.sqrt (upper_bound / 49)) + 1) \ 
   Finset.range (Nat.floor (Real.sqrt (lower_bound / 49)))).card = 4376 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_seven_perfect_squares_l3702_370257


namespace NUMINAMATH_CALUDE_two_digit_cube_diff_reverse_l3702_370210

/-- A function that reverses a two-digit number -/
def reverse (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- A predicate that checks if a number is a positive perfect cube -/
def is_positive_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ k^3 = n

/-- The main theorem -/
theorem two_digit_cube_diff_reverse :
  ∀ M : ℕ,
    10 ≤ M ∧ M < 100 ∧  -- M is a two-digit number
    (M % 10 ≠ 0) ∧      -- M's unit digit is non-zero
    is_positive_perfect_cube (M - reverse M) →
    M = 81 ∨ M = 92 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_cube_diff_reverse_l3702_370210


namespace NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l3702_370290

-- Define the angle measure in degrees
def angle_measure : ℝ → Prop :=
  λ x => x > 0 ∧ x < 180

-- Define the supplement of an angle
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem angle_measure_with_special_supplement_complement :
  ∀ x : ℝ, angle_measure x → supplement x = 4 * complement x → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_with_special_supplement_complement_l3702_370290


namespace NUMINAMATH_CALUDE_budget_calculation_l3702_370228

/-- The original budget in Euros -/
def original_budget : ℝ := sorry

/-- The amount left after spending -/
def amount_left : ℝ := 13500

/-- The fraction of budget spent on clothes -/
def clothes_fraction : ℝ := 0.25

/-- The discount on clothes -/
def clothes_discount : ℝ := 0.1

/-- The fraction of budget spent on groceries -/
def groceries_fraction : ℝ := 0.15

/-- The sales tax on groceries -/
def groceries_tax : ℝ := 0.05

/-- The fraction of budget spent on electronics -/
def electronics_fraction : ℝ := 0.1

/-- The exchange rate for electronics (EUR to USD) -/
def exchange_rate : ℝ := 1.2

/-- The fraction of budget spent on dining -/
def dining_fraction : ℝ := 0.05

/-- The service charge on dining -/
def dining_service_charge : ℝ := 0.12

theorem budget_calculation :
  amount_left = original_budget * (1 - (
    clothes_fraction * (1 - clothes_discount) +
    groceries_fraction * (1 + groceries_tax) +
    electronics_fraction * exchange_rate +
    dining_fraction * (1 + dining_service_charge)
  )) := by sorry

end NUMINAMATH_CALUDE_budget_calculation_l3702_370228


namespace NUMINAMATH_CALUDE_m_range_l3702_370230

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 - 2*m*x + 7*m - 10 ≠ 0

def q (m : ℝ) : Prop := ∀ x > 0, x^2 - m*x + 4 ≥ 0

-- State the theorem
theorem m_range (m : ℝ) 
  (h1 : p m ∨ q m) 
  (h2 : p m ∧ q m) : 
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end NUMINAMATH_CALUDE_m_range_l3702_370230


namespace NUMINAMATH_CALUDE_line_connecting_circle_centers_l3702_370289

/-- The equation of the line connecting the centers of two circles -/
theorem line_connecting_circle_centers 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 - 4*x + 6*y = 0)
  (circle2 : ∀ x y : ℝ, x^2 + y^2 - 6*x = 0) :
  ∃ (x y : ℝ), 3*x - y - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_connecting_circle_centers_l3702_370289


namespace NUMINAMATH_CALUDE_f_range_theorem_l3702_370238

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

-- State the theorem
theorem f_range_theorem :
  (∀ x : ℝ, f (6 - x) = f x) →  -- Symmetry condition
  (∃ y : ℝ, ∀ x : ℝ, f x ≥ y) ∧ -- Lower bound exists
  (∀ y : ℝ, y ≥ -36 → ∃ x : ℝ, f x = y) -- All values ≥ -36 are in the range
  :=
by sorry

end NUMINAMATH_CALUDE_f_range_theorem_l3702_370238


namespace NUMINAMATH_CALUDE_triangle_side_length_l3702_370226

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 60 * π / 180 →  -- Convert 60° to radians
  a = Real.sqrt 31 →
  b = 6 →
  (c = 1 ∨ c = 5) →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3702_370226


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l3702_370299

theorem rectangle_width_proof (length width : ℝ) : 
  length = 24 →
  2 * length + 2 * width = 80 →
  length / width = 6 / 5 →
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l3702_370299


namespace NUMINAMATH_CALUDE_acute_angle_tan_value_l3702_370296

theorem acute_angle_tan_value (α : Real) (h : α > 0 ∧ α < Real.pi / 2) 
  (h_eq : Real.sqrt (369 - 360 * Real.cos α) + Real.sqrt (544 - 480 * Real.sin α) - 25 = 0) : 
  40 * Real.tan α = 30 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_tan_value_l3702_370296


namespace NUMINAMATH_CALUDE_four_dice_probability_l3702_370255

/-- The probability of a single standard six-sided die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed simultaneously -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same specific number -/
def all_dice_prob : ℚ := (single_die_prob ^ num_dice)

/-- Theorem stating that the probability of all four standard six-sided dice 
    showing the number 3 when tossed simultaneously is 1/1296 -/
theorem four_dice_probability : all_dice_prob = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_probability_l3702_370255


namespace NUMINAMATH_CALUDE_remainder_845307_div_6_l3702_370259

theorem remainder_845307_div_6 : Nat.mod 845307 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_845307_div_6_l3702_370259


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l3702_370285

theorem square_area_equal_perimeter (triangle_area : ℝ) : 
  triangle_area = 16 * Real.sqrt 3 → 
  ∃ (triangle_side square_side : ℝ), 
    triangle_side > 0 ∧ 
    square_side > 0 ∧ 
    (triangle_side^2 * Real.sqrt 3) / 4 = triangle_area ∧ 
    3 * triangle_side = 4 * square_side ∧ 
    square_side^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l3702_370285


namespace NUMINAMATH_CALUDE_ratio_limit_is_one_l3702_370253

/-- The ratio of the largest element (2^20) to the sum of other elements in the set {1, 2, 2^2, ..., 2^20} -/
def ratio (n : ℕ) : ℚ :=
  2^n / (2^n - 1)

/-- The limit of the ratio as n approaches infinity is 1 -/
theorem ratio_limit_is_one : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_limit_is_one_l3702_370253


namespace NUMINAMATH_CALUDE_altitude_intersection_angle_l3702_370223

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the altitude intersection point H
def H (t : Triangle) : Point := sorry

-- Define the angles of the triangle
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry

-- Define the angle AHB
def angle_AHB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_intersection_angle (t : Triangle) 
  (h1 : angle_BAC t = 40)
  (h2 : angle_ABC t = 65) :
  angle_AHB t = 105 := by sorry

end NUMINAMATH_CALUDE_altitude_intersection_angle_l3702_370223


namespace NUMINAMATH_CALUDE_tank_filling_time_tank_filling_time_proof_l3702_370235

theorem tank_filling_time : ℝ → Prop :=
  fun T : ℝ =>
    let fill_rate_A : ℝ := 1 / 60
    let fill_rate_B : ℝ := 1 / 40
    let first_half : ℝ := T / 2 * fill_rate_B
    let second_half : ℝ := T / 2 * (fill_rate_A + fill_rate_B)
    (first_half + second_half = 1) → (T = 48)

-- The proof goes here
theorem tank_filling_time_proof : tank_filling_time 48 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_tank_filling_time_proof_l3702_370235


namespace NUMINAMATH_CALUDE_total_pieces_four_row_triangle_l3702_370298

/-- Calculates the sum of the first n multiples of 3 -/
def sum_multiples_of_three (n : ℕ) : ℕ := 
  3 * n * (n + 1) / 2

/-- Calculates the sum of the first n even numbers -/
def sum_even_numbers (n : ℕ) : ℕ := 
  n * (n + 1)

/-- Represents the number of rows in the triangle configuration -/
def num_rows : ℕ := 4

/-- Theorem: The total number of pieces in a four-row triangle configuration is 60 -/
theorem total_pieces_four_row_triangle : 
  sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1) = 60 := by
  sorry

#eval sum_multiples_of_three num_rows + sum_even_numbers (num_rows + 1)

end NUMINAMATH_CALUDE_total_pieces_four_row_triangle_l3702_370298


namespace NUMINAMATH_CALUDE_ceiling_distance_l3702_370245

/-- A point in a right-angled corner formed by two walls and a ceiling -/
structure CornerPoint where
  x : ℝ  -- distance from one wall
  y : ℝ  -- distance from the other wall
  z : ℝ  -- distance from the ceiling
  corner_distance : ℝ  -- distance from the corner point

/-- The theorem stating the distance from the ceiling for a specific point -/
theorem ceiling_distance (p : CornerPoint) 
  (h1 : p.x = 3)  -- 3 meters from one wall
  (h2 : p.y = 7)  -- 7 meters from the other wall
  (h3 : p.corner_distance = 10)  -- 10 meters from the corner point
  : p.z = Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_distance_l3702_370245


namespace NUMINAMATH_CALUDE_chords_intersection_concyclic_l3702_370267

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
structure PointOnEllipse (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse a b x y

-- Define the theorem
theorem chords_intersection_concyclic 
  (a b : ℝ) 
  (A B C D S : PointOnEllipse a b) 
  (h1 : S.x ≠ A.x ∨ S.y ≠ A.y) 
  (h2 : S.x ≠ B.x ∨ S.y ≠ B.y)
  (h3 : S.x ≠ C.x ∨ S.y ≠ C.y)
  (h4 : S.x ≠ D.x ∨ S.y ≠ D.y)
  (h5 : (A.y - S.y) * (C.x - S.x) = (A.x - S.x) * (C.y - S.y)) -- AB and CD intersect at S
  (h6 : (B.y - S.y) * (D.x - S.x) = (B.x - S.x) * (D.y - S.y)) -- AB and CD intersect at S
  (h7 : (A.y - S.y) * (C.x - S.x) = (C.y - S.y) * (D.x - S.x)) -- ∠ASC = ∠BSD
  : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.x - center.1)^2 + (A.y - center.2)^2 = radius^2 ∧
    (B.x - center.1)^2 + (B.y - center.2)^2 = radius^2 ∧
    (C.x - center.1)^2 + (C.y - center.2)^2 = radius^2 ∧
    (D.x - center.1)^2 + (D.y - center.2)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_chords_intersection_concyclic_l3702_370267


namespace NUMINAMATH_CALUDE_bag_cost_is_eight_l3702_370229

/-- Represents the coffee consumption and cost for Maddie's mom --/
structure CoffeeConsumption where
  cups_per_day : ℕ
  ounces_per_cup : ℚ
  ounces_per_bag : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  total_weekly_cost : ℚ

/-- Calculates the cost of a bag of coffee based on the given consumption data --/
def bag_cost (c : CoffeeConsumption) : ℚ :=
  let ounces_per_week := c.cups_per_day * c.ounces_per_cup * 7
  let bags_per_week := ounces_per_week / c.ounces_per_bag
  let milk_cost_per_week := c.milk_gallons_per_week * c.milk_cost_per_gallon
  let coffee_cost_per_week := c.total_weekly_cost - milk_cost_per_week
  coffee_cost_per_week / bags_per_week

/-- Theorem stating that the cost of a bag of coffee is $8 --/
theorem bag_cost_is_eight (c : CoffeeConsumption) 
  (h1 : c.cups_per_day = 2)
  (h2 : c.ounces_per_cup = 3/2)
  (h3 : c.ounces_per_bag = 21/2)
  (h4 : c.milk_gallons_per_week = 1/2)
  (h5 : c.milk_cost_per_gallon = 4)
  (h6 : c.total_weekly_cost = 18) :
  bag_cost c = 8 := by
  sorry

#eval bag_cost {
  cups_per_day := 2,
  ounces_per_cup := 3/2,
  ounces_per_bag := 21/2,
  milk_gallons_per_week := 1/2,
  milk_cost_per_gallon := 4,
  total_weekly_cost := 18
}

end NUMINAMATH_CALUDE_bag_cost_is_eight_l3702_370229


namespace NUMINAMATH_CALUDE_suma_work_time_l3702_370241

/-- Proves the time taken by Suma to complete the work alone -/
theorem suma_work_time (renu_time suma_renu_time : ℝ) 
  (h1 : renu_time = 8)
  (h2 : suma_renu_time = 3)
  (h3 : renu_time > 0)
  (h4 : suma_renu_time > 0) :
  ∃ (suma_time : ℝ), 
    suma_time > 0 ∧ 
    1 / renu_time + 1 / suma_time = 1 / suma_renu_time ∧ 
    suma_time = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_suma_work_time_l3702_370241


namespace NUMINAMATH_CALUDE_inverse_of_sixteen_point_six_periodic_l3702_370277

/-- Given that 1 divided by a number is equal to 16.666666666666668,
    prove that the number is equal to 1/60. -/
theorem inverse_of_sixteen_point_six_periodic : ∃ x : ℚ, (1 : ℚ) / x = 16666666666666668 / 1000000000000000 ∧ x = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_sixteen_point_six_periodic_l3702_370277


namespace NUMINAMATH_CALUDE_angle_relation_in_triangle_l3702_370268

theorem angle_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) (h_sin : Real.sin A > Real.sin B) : A > B := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_in_triangle_l3702_370268


namespace NUMINAMATH_CALUDE_composition_is_rotation_l3702_370218

-- Define a rotation
def Rotation (center : Point) (angle : ℝ) : Point → Point :=
  sorry

-- Define the composition of two rotations
def ComposeRotations (A B : Point) (α β : ℝ) : Point → Point :=
  Rotation B β ∘ Rotation A α

-- Theorem statement
theorem composition_is_rotation (A B : Point) (α β : ℝ) 
  (h1 : A ≠ B) 
  (h2 : ¬ (∃ k : ℤ, α + β = 2 * π * k)) :
  ∃ (O : Point) (γ : ℝ), ComposeRotations A B α β = Rotation O γ ∧ γ = α + β :=
sorry

end NUMINAMATH_CALUDE_composition_is_rotation_l3702_370218


namespace NUMINAMATH_CALUDE_tournament_games_l3702_370209

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- The number of teams in the tournament -/
def numTeams : ℕ := 20

theorem tournament_games :
  gamesInTournament numTeams = 19 := by sorry

end NUMINAMATH_CALUDE_tournament_games_l3702_370209


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3702_370239

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -1 < x < 2},
    prove that the solution set of a(x^2 + 1) + b(x - 1) + c > 2ax is {x | 0 < x < 3} -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x, a*(x^2 + 1) + b*(x - 1) + c > 2*a*x ↔ 0 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3702_370239


namespace NUMINAMATH_CALUDE_dog_park_ratio_l3702_370204

theorem dog_park_ratio (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_ear_dogs : ℕ) :
  spotted_dogs = total_dogs / 2 →
  spotted_dogs = 15 →
  pointy_ear_dogs = 6 →
  (pointy_ear_dogs : ℚ) / total_dogs = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_dog_park_ratio_l3702_370204


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3702_370260

theorem expression_equals_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) :
  (x + 1/x) * (y - 1/y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3702_370260


namespace NUMINAMATH_CALUDE_bert_stamp_ratio_l3702_370249

def stamps_before (total_after purchase : ℕ) : ℕ := total_after - purchase

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem bert_stamp_ratio : 
  let purchase := 300
  let total_after := 450
  let before := stamps_before total_after purchase
  ratio_simplify before purchase = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_bert_stamp_ratio_l3702_370249


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3702_370221

/-- Given the equation (1+x)+(1+x)^2+...+(1+x)^5 = a₀+a₁(1-x)+a₂(1-x)^2+...+a₅(1-x)^5,
    prove that a₁+a₂+a₃+a₄+a₅ = -57 -/
theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  ((1+x) + (1+x)^2 + (1+x)^3 + (1+x)^4 + (1+x)^5 = 
   a₀ + a₁*(1-x) + a₂*(1-x)^2 + a₃*(1-x)^3 + a₄*(1-x)^4 + a₅*(1-x)^5) → 
  (a₁ + a₂ + a₃ + a₄ + a₅ = -57) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3702_370221


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3702_370217

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3702_370217


namespace NUMINAMATH_CALUDE_A_has_min_l3702_370263

/-- The function f_{a,b} from R^2 to R^2 -/
def f (a b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (a - b * y - x^2, x)

/-- The n-th iteration of f_{a,b} -/
def f_iter (a b : ℝ) : ℕ → (ℝ × ℝ → ℝ × ℝ)
  | 0 => id
  | n + 1 => f a b ∘ f_iter a b n

/-- The set of periodic points of f_{a,b} -/
def per (a b : ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ n : ℕ+, f_iter a b n P = P}

/-- The set A_b -/
def A (b : ℝ) : Set ℝ :=
  {a | per a b ≠ ∅}

/-- The theorem stating that A_b has a minimum equal to -(b+1)^2/4 -/
theorem A_has_min (b : ℝ) : 
  ∃ min : ℝ, IsGLB (A b) min ∧ min = -(b + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_A_has_min_l3702_370263


namespace NUMINAMATH_CALUDE_cos_equality_integer_l3702_370240

theorem cos_equality_integer (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (↑n * π / 180) = Real.cos (430 * π / 180) →
  n = 70 ∨ n = -70 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_integer_l3702_370240


namespace NUMINAMATH_CALUDE_total_distance_is_12_17_l3702_370200

def walking_time : ℚ := 30 / 60
def walking_rate : ℚ := 3
def running_time : ℚ := 20 / 60
def running_rate : ℚ := 8
def cycling_time : ℚ := 40 / 60
def cycling_rate : ℚ := 12

def total_distance : ℚ :=
  walking_time * walking_rate +
  running_time * running_rate +
  cycling_time * cycling_rate

theorem total_distance_is_12_17 :
  total_distance = 12.17 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_12_17_l3702_370200


namespace NUMINAMATH_CALUDE_checkerboard_corner_sum_l3702_370206

/-- The size of the checkerboard -/
def boardSize : Nat := 9

/-- The total number of squares on the board -/
def totalSquares : Nat := boardSize * boardSize

/-- The number in the top-left corner -/
def topLeft : Nat := 1

/-- The number in the top-right corner -/
def topRight : Nat := boardSize

/-- The number in the bottom-left corner -/
def bottomLeft : Nat := totalSquares - boardSize + 1

/-- The number in the bottom-right corner -/
def bottomRight : Nat := totalSquares

/-- The sum of the numbers in the four corners of the checkerboard -/
def cornerSum : Nat := topLeft + topRight + bottomLeft + bottomRight

theorem checkerboard_corner_sum : cornerSum = 164 := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_corner_sum_l3702_370206


namespace NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3702_370201

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x > 0}

-- State the theorem
theorem intersection_complement_equals_interval :
  A ∩ (Set.univ \ B) = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_interval_l3702_370201


namespace NUMINAMATH_CALUDE_total_stickers_is_36_l3702_370256

/-- The number of stickers Elizabeth uses on her water bottles -/
def total_stickers : ℕ :=
  let initial_bottles : ℕ := 20
  let lost_school : ℕ := 5
  let found_park : ℕ := 3
  let stolen_dance : ℕ := 4
  let misplaced_library : ℕ := 2
  let acquired_friend : ℕ := 6
  let stickers_school : ℕ := 4
  let stickers_dance : ℕ := 3
  let stickers_library : ℕ := 2

  let school_stickers := lost_school * stickers_school
  let dance_stickers := stolen_dance * stickers_dance
  let library_stickers := misplaced_library * stickers_library

  school_stickers + dance_stickers + library_stickers

theorem total_stickers_is_36 : total_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_is_36_l3702_370256


namespace NUMINAMATH_CALUDE_properties_of_f_l3702_370282

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem properties_of_f :
  let f' := fun x => Real.exp x - 1 / (x + 3)
  let f'' := fun x => Real.exp x + 1 / ((x + 3)^2)
  (∀ x > -3, f'' x > 0) ∧
  (∃! x₀ : ℝ, -1 < x₀ ∧ x₀ < 0 ∧ f' x₀ = 0) ∧
  (∃ x_min : ℝ, ∀ x > -3, f x ≥ f x_min) ∧
  (∀ x > -3, f x > -1/2) :=
by sorry

end NUMINAMATH_CALUDE_properties_of_f_l3702_370282


namespace NUMINAMATH_CALUDE_number_of_sweaters_l3702_370291

def washing_machine_capacity : ℕ := 7
def number_of_shirts : ℕ := 2
def number_of_loads : ℕ := 5

theorem number_of_sweaters : 
  (washing_machine_capacity * number_of_loads) - number_of_shirts = 33 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sweaters_l3702_370291


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3702_370243

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * (a^2 + b*c))/(b + c) + (b * (b^2 + a*c))/(a + c) + (c * (c^2 + a*b))/(a + b) ≥
  a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3702_370243


namespace NUMINAMATH_CALUDE_smallest_number_l3702_370231

theorem smallest_number (a b c : ℝ) : 
  c = 2 * a →
  b = 4 * a →
  (a + b + c) / 3 = 77 →
  a = 33 ∧ a ≤ b ∧ a ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3702_370231


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l3702_370270

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 1225) 
  (h2 : rectangle_area = 140) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_of_circle_radius_l3702_370270


namespace NUMINAMATH_CALUDE_exponential_inequality_l3702_370273

theorem exponential_inequality (x y a : ℝ) 
  (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : 
  a^x < a^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3702_370273


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3702_370227

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 7

/-- Theorem: In a chess tournament with 7 players, where each player plays twice
    with every other player, the total number of games played is 84. --/
theorem chess_tournament_games :
  2 * tournament_games num_players = 84 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l3702_370227


namespace NUMINAMATH_CALUDE_hexagon_implies_face_fits_l3702_370275

/-- A rectangular parallelepiped with dimensions a, b, and c. -/
structure RectangularParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c

/-- A rectangle with dimensions d₁ and d₂. -/
structure Rectangle where
  d₁ : ℝ
  d₂ : ℝ
  hd₁ : 0 < d₁
  hd₂ : 0 < d₂

/-- A hexagonal cross-section of a rectangular parallelepiped. -/
structure HexagonalCrossSection (rp : RectangularParallelepiped) where

/-- The proposition that a hexagonal cross-section fits in a rectangle. -/
def fits_in (h : HexagonalCrossSection rp) (r : Rectangle) : Prop :=
  sorry

/-- The proposition that a face of a rectangular parallelepiped fits in a rectangle. -/
def face_fits_in (rp : RectangularParallelepiped) (r : Rectangle) : Prop :=
  (rp.a ≤ r.d₁ ∧ rp.b ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.b ≤ r.d₁) ∨
  (rp.b ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.b ≤ r.d₂ ∧ rp.c ≤ r.d₁) ∨
  (rp.a ≤ r.d₁ ∧ rp.c ≤ r.d₂) ∨ (rp.a ≤ r.d₂ ∧ rp.c ≤ r.d₁)

/-- The main theorem to be proved. -/
theorem hexagon_implies_face_fits 
  (rp : RectangularParallelepiped) 
  (r : Rectangle) 
  (h : HexagonalCrossSection rp) 
  (h_fits : fits_in h r) : 
  face_fits_in rp r :=
sorry

end NUMINAMATH_CALUDE_hexagon_implies_face_fits_l3702_370275


namespace NUMINAMATH_CALUDE_consistent_number_theorem_l3702_370211

def is_consistent_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ), m = 1000 * a + 100 * b + 10 * c + d ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a + b = c + d

def F (m : ℕ) : ℚ :=
  let m' := (m / 10) % 10 * 1000 + m % 10 * 100 + m / 1000 * 10 + (m / 100) % 10
  (m + m') / 101

def G (N : ℕ) : ℕ := N / 10 + N % 10

theorem consistent_number_theorem :
  ∀ (m : ℕ), is_consistent_number m →
    let a := m / 1000
    let b := (m / 100) % 10
    let c := (m / 10) % 10
    let d := m % 10
    let N := 10 * a + 2 * b
    a ≤ 8 →
    d = 1 →
    Even (G N) →
    ∃ (k : ℤ), F m - G N - 4 * a = k^2 + 3 →
    (k = 6 ∨ k = -6) ∧ m = 2231 := by
  sorry

end NUMINAMATH_CALUDE_consistent_number_theorem_l3702_370211


namespace NUMINAMATH_CALUDE_negation_equivalence_l3702_370244

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x < x) ↔ (∀ x : ℝ, Real.exp x ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3702_370244


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3702_370295

/-- Represents the repair cost calculation for Ramu's car sale --/
theorem repair_cost_calculation (initial_cost selling_price profit_percent : ℝ) (R : ℝ) : 
  initial_cost = 34000 →
  selling_price = 65000 →
  profit_percent = 41.30434782608695 →
  profit_percent = ((selling_price - (initial_cost + R)) / (initial_cost + R)) * 100 :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l3702_370295


namespace NUMINAMATH_CALUDE_probability_of_two_tails_in_three_flips_l3702_370254

def probability_of_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem probability_of_two_tails_in_three_flips :
  probability_of_k_successes 3 2 (1/2) = 0.375 := by
sorry

end NUMINAMATH_CALUDE_probability_of_two_tails_in_three_flips_l3702_370254


namespace NUMINAMATH_CALUDE_sam_drew_age_problem_l3702_370213

/-- The combined age of Sam and Drew given Sam's age and the relation between their ages -/
def combinedAge (samAge : ℕ) (drewAge : ℕ) : ℕ := samAge + drewAge

theorem sam_drew_age_problem :
  let samAge : ℕ := 18
  let drewAge : ℕ := 2 * samAge
  combinedAge samAge drewAge = 54 := by
  sorry

end NUMINAMATH_CALUDE_sam_drew_age_problem_l3702_370213


namespace NUMINAMATH_CALUDE_walkway_area_calculation_l3702_370276

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the layout of the garden -/
structure GardenLayout where
  bed : FlowerBed
  rows : ℕ
  columns : ℕ
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (layout : GardenLayout) : ℝ :=
  let totalWidth := layout.columns * layout.bed.length + (layout.columns + 1) * layout.walkwayWidth
  let totalHeight := layout.rows * layout.bed.width + (layout.rows + 1) * layout.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := layout.rows * layout.columns * layout.bed.length * layout.bed.width
  totalArea - bedArea

theorem walkway_area_calculation (layout : GardenLayout) : 
  layout.bed.length = 6 ∧ 
  layout.bed.width = 2 ∧ 
  layout.rows = 3 ∧ 
  layout.columns = 2 ∧ 
  layout.walkwayWidth = 1 → 
  walkwayArea layout = 78 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_calculation_l3702_370276


namespace NUMINAMATH_CALUDE_factorization_a5_minus_a3b2_l3702_370242

theorem factorization_a5_minus_a3b2 (a b : ℝ) : 
  a^5 - a^3 * b^2 = a^3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_a5_minus_a3b2_l3702_370242


namespace NUMINAMATH_CALUDE_sheela_deposit_l3702_370215

/-- Sheela's monthly income in Rupees -/
def monthly_income : ℕ := 25000

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 20 / 100

/-- Calculate the deposit amount based on monthly income and deposit percentage -/
def deposit_amount (income : ℕ) (percentage : ℚ) : ℚ :=
  percentage * income

/-- Theorem stating that Sheela's deposit amount is 5000 Rupees -/
theorem sheela_deposit :
  deposit_amount monthly_income deposit_percentage = 5000 := by
  sorry

end NUMINAMATH_CALUDE_sheela_deposit_l3702_370215


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_and_thirty_l3702_370292

theorem greatest_integer_with_gcf_five_and_thirty : ∃ n : ℕ, 
  n < 200 ∧ 
  n > 185 ∧
  Nat.gcd n 30 = 5 → False ∧ 
  Nat.gcd 185 30 = 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_and_thirty_l3702_370292


namespace NUMINAMATH_CALUDE_function_property_l3702_370222

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x : ℝ, f (x + 94) ≥ f x + 94) : 
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3702_370222


namespace NUMINAMATH_CALUDE_jerry_added_ten_books_l3702_370258

/-- The number of books Jerry added to his shelf -/
def books_added (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that Jerry added 10 books to his shelf -/
theorem jerry_added_ten_books (initial_books final_books : ℕ) 
  (h1 : initial_books = 9)
  (h2 : final_books = 19) : 
  books_added initial_books final_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_ten_books_l3702_370258


namespace NUMINAMATH_CALUDE_equation_transformation_l3702_370297

theorem equation_transformation (x : ℝ) :
  (x + 2) / 4 = (2 * x + 3) / 6 →
  12 * ((x + 2) / 4) = 12 * ((2 * x + 3) / 6) →
  3 * (x + 2) = 2 * (2 * x + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3702_370297


namespace NUMINAMATH_CALUDE_system_three_solutions_l3702_370234

/-- The system of equations has exactly three solutions if and only if a = 49 or a = 40 - 4√51 -/
theorem system_three_solutions (a : ℝ) : 
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 - 10) + abs (x.1 + 3) - 2) * (x.1^2 + x.2^2 - 6) = 0 ∧
     (x.1 + 3)^2 + (x.2 - 5)^2 = a) ∧
    ((abs (y.2 - 10) + abs (y.1 + 3) - 2) * (y.1^2 + y.2^2 - 6) = 0 ∧
     (y.1 + 3)^2 + (y.2 - 5)^2 = a) ∧
    ((abs (z.2 - 10) + abs (z.1 + 3) - 2) * (z.1^2 + z.2^2 - 6) = 0 ∧
     (z.1 + 3)^2 + (z.2 - 5)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
by sorry

end NUMINAMATH_CALUDE_system_three_solutions_l3702_370234


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3702_370236

theorem line_tangent_to_circle (t θ : ℝ) (α : ℝ) : 
  (∃ t, ∀ θ, (t * Real.cos α - (4 + 2 * Real.cos θ))^2 + (t * Real.sin α - 2 * Real.sin θ)^2 = 4) →
  α = π / 6 ∨ α = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3702_370236


namespace NUMINAMATH_CALUDE_distance_between_points_l3702_370274

/-- The distance between points (5, 5) and (0, 0) is 5√2 -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (5, 5)
  let p2 : ℝ × ℝ := (0, 0)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3702_370274


namespace NUMINAMATH_CALUDE_exists_same_answer_question_l3702_370247

/-- Represents a person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- Represents a question that can be asked to a person -/
def Question := Type

/-- Represents an answer to a question -/
def Answer := Type

/-- The response function that determines how a person answers a question -/
def respond (p : Person) (q : Question) : Answer :=
  sorry

/-- Theorem stating that there exists a question that elicits the same answer from both a truth-teller and a liar -/
theorem exists_same_answer_question :
  ∃ (q : Question), ∀ (p1 p2 : Person), p1 ≠ p2 → respond p1 q = respond p2 q :=
sorry

end NUMINAMATH_CALUDE_exists_same_answer_question_l3702_370247


namespace NUMINAMATH_CALUDE_soft_drinks_bought_l3702_370279

theorem soft_drinks_bought (soft_drink_cost : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (total_spent : ℕ) : 
  soft_drink_cost = 4 →
  candy_bars = 5 →
  candy_bar_cost = 4 →
  total_spent = 28 →
  ∃ (num_soft_drinks : ℕ), num_soft_drinks * soft_drink_cost + candy_bars * candy_bar_cost = total_spent ∧ num_soft_drinks = 2 :=
by sorry

end NUMINAMATH_CALUDE_soft_drinks_bought_l3702_370279


namespace NUMINAMATH_CALUDE_quiz_probability_l3702_370202

/-- The probability of answering a multiple-choice question with 5 options correctly -/
def prob_multiple_choice : ℚ := 1 / 5

/-- The probability of answering a true/false question correctly -/
def prob_true_false : ℚ := 1 / 2

/-- The number of true/false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The probability of answering all questions in the quiz correctly -/
def prob_all_correct : ℚ := prob_multiple_choice * prob_true_false ^ num_true_false

theorem quiz_probability :
  prob_all_correct = 1 / 80 := by sorry

end NUMINAMATH_CALUDE_quiz_probability_l3702_370202


namespace NUMINAMATH_CALUDE_eighth_minus_seventh_difference_l3702_370224

/-- The number of tiles in the nth square of the sequence -/
def tiles (n : ℕ) : ℕ := n^2 + 2*n

/-- The difference in tiles between the 8th and 7th squares -/
def tile_difference : ℕ := tiles 8 - tiles 7

theorem eighth_minus_seventh_difference :
  tile_difference = 17 := by sorry

end NUMINAMATH_CALUDE_eighth_minus_seventh_difference_l3702_370224


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3702_370205

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3702_370205


namespace NUMINAMATH_CALUDE_g_16_48_l3702_370261

/-- A function on ordered pairs of positive integers satisfying specific properties -/
def g : ℕ+ → ℕ+ → ℕ+ :=
  sorry

/-- The first property: g(x,x) = 2x -/
axiom g_diag (x : ℕ+) : g x x = 2 * x

/-- The second property: g(x,y) = g(y,x) -/
axiom g_comm (x y : ℕ+) : g x y = g y x

/-- The third property: (x + y) g(x,y) = x g(x, x + y) -/
axiom g_prop (x y : ℕ+) : (x + y) * g x y = x * g x (x + y)

/-- The main theorem: g(16, 48) = 96 -/
theorem g_16_48 : g 16 48 = 96 :=
  sorry

end NUMINAMATH_CALUDE_g_16_48_l3702_370261


namespace NUMINAMATH_CALUDE_tromino_tiling_l3702_370248

/-- An L-shaped tromino covers exactly 3 unit squares. -/
def Tromino : ℕ := 3

/-- Represents whether an m×n grid can be tiled with L-shaped trominoes. -/
def can_tile (m n : ℕ) : Prop := 6 ∣ (m * n)

/-- 
Theorem: An m×n grid can be tiled with L-shaped trominoes if and only if 6 divides mn.
-/
theorem tromino_tiling (m n : ℕ) : can_tile m n ↔ 6 ∣ (m * n) := by sorry

end NUMINAMATH_CALUDE_tromino_tiling_l3702_370248


namespace NUMINAMATH_CALUDE_equal_probability_l3702_370264

/-- The number of black gloves in the pocket -/
def black_gloves : ℕ := 15

/-- The number of white gloves in the pocket -/
def white_gloves : ℕ := 10

/-- The total number of gloves in the pocket -/
def total_gloves : ℕ := black_gloves + white_gloves

/-- The number of ways to choose 2 gloves from n gloves -/
def choose (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The probability of drawing two gloves of the same color -/
def prob_same_color : ℚ :=
  (choose black_gloves + choose white_gloves) / choose total_gloves

/-- The probability of drawing two gloves of different colors -/
def prob_diff_color : ℚ :=
  (black_gloves * white_gloves) / choose total_gloves

theorem equal_probability : prob_same_color = prob_diff_color := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_l3702_370264


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l3702_370237

theorem maintenance_check_increase (original_time : ℝ) (percentage_increase : ℝ) 
  (h1 : original_time = 45)
  (h2 : percentage_increase = 33.33333333333333) : 
  original_time * (1 + percentage_increase / 100) = 60 := by
sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l3702_370237


namespace NUMINAMATH_CALUDE_f_is_odd_iff_l3702_370216

/-- A function f is odd if f(-x) = -f(x) for all x in the domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) : ℝ → ℝ := fun x ↦ x * |x + a| + b

/-- Theorem: f is an odd function if and only if a = 0 and b = 0 -/
theorem f_is_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_iff_l3702_370216


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l3702_370286

/-- A quadratic function with a symmetry axis at x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  symmetry_axis : a ≠ 0 ∧ -|b| / (2 * a) = -1

/-- Three points on the parabola -/
structure ParabolaPoints where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  on_parabola : ∀ (f : QuadraticFunction),
    f.a * (-14/3)^2 + |f.b| * (-14/3) + f.c = y₁ ∧
    f.a * (5/2)^2 + |f.b| * (5/2) + f.c = y₂ ∧
    f.a * 3^2 + |f.b| * 3 + f.c = y₃

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem parabola_point_relationship (f : QuadraticFunction) (p : ParabolaPoints) :
  p.y₂ < p.y₁ ∧ p.y₁ < p.y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l3702_370286


namespace NUMINAMATH_CALUDE_equation_represents_ellipse_l3702_370208

/-- The equation x^2 + 2y^2 - 6x - 8y + 9 = 0 represents an ellipse -/
theorem equation_represents_ellipse :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), x^2 + 2*y^2 - 6*x - 8*y + 9 = 0 ↔
      ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_ellipse_l3702_370208


namespace NUMINAMATH_CALUDE_unique_number_power_ten_sum_l3702_370294

theorem unique_number_power_ten_sum : ∃! (N : ℕ), 
  N > 0 ∧ 
  (∃ (k : ℕ), N + (Nat.factors N).foldl Nat.lcm 1 = 10^k) ∧
  N = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_power_ten_sum_l3702_370294


namespace NUMINAMATH_CALUDE_chess_match_results_l3702_370207

/-- Chess match between players A and B -/
structure ChessMatch where
  prob_draw : ℝ
  prob_a_win : ℝ
  prob_b_win : ℝ

/-- Conditions for the chess match -/
def match_conditions : ChessMatch where
  prob_draw := 0.5
  prob_a_win := 0.3
  prob_b_win := 0.2

/-- Expected number of games in the match -/
def expected_games (m : ChessMatch) : ℝ := sorry

/-- Probability that player B wins the match -/
def prob_b_wins (m : ChessMatch) : ℝ := sorry

/-- Theorem stating the expected number of games and probability of B winning -/
theorem chess_match_results (m : ChessMatch) 
  (h1 : m = match_conditions) : 
  expected_games m = 3.175 ∧ prob_b_wins m = 0.315 := by sorry

end NUMINAMATH_CALUDE_chess_match_results_l3702_370207


namespace NUMINAMATH_CALUDE_distance_BC_l3702_370251

/-- Represents a point on the route --/
structure Point :=
  (position : ℝ)

/-- Represents the route with points A, B, and C --/
structure Route :=
  (A B C : Point)
  (speed : ℝ)
  (time : ℝ)
  (AC_distance : ℝ)

/-- The theorem statement --/
theorem distance_BC (route : Route) : 
  route.A.position = 0 ∧ 
  route.speed = 50 ∧ 
  route.time = 20 ∧ 
  route.AC_distance = 600 →
  route.C.position - route.B.position = 400 :=
sorry

end NUMINAMATH_CALUDE_distance_BC_l3702_370251


namespace NUMINAMATH_CALUDE_ampersand_composition_l3702_370280

-- Define the operations
def ampersand_right (x : ℝ) : ℝ := 9 - x
def ampersand_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem ampersand_composition : ampersand_left (ampersand_right 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_ampersand_composition_l3702_370280


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l3702_370214

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareConfig where
  s : ℝ  -- Side length of the inner square
  x : ℝ  -- Shorter side of the rectangle
  y : ℝ  -- Longer side of the rectangle

/-- Theorem: If four congruent rectangles are placed around a central square such that 
    the area of the outer square is 9 times the area of the inner square, 
    then the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_square_ratio 
  (config : RectangleSquareConfig) 
  (h1 : config.s > 0)  -- Inner square has positive side length
  (h2 : config.x > 0)  -- Rectangle has positive width
  (h3 : config.y > 0)  -- Rectangle has positive height
  (h4 : config.s + 2 * config.x = 3 * config.s)  -- Outer square side length relation
  (h5 : config.y + config.x = 3 * config.s)  -- Outer square side length relation (alternative)
  : config.y / config.x = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_ratio_l3702_370214


namespace NUMINAMATH_CALUDE_special_cylinder_lateral_area_l3702_370287

/-- A cylinder with base area S and lateral surface that unfolds into a square -/
structure SpecialCylinder where
  S : ℝ
  baseArea : S > 0
  lateralSurfaceIsSquare : True

/-- The lateral surface area of a SpecialCylinder is 4πS -/
theorem special_cylinder_lateral_area (c : SpecialCylinder) :
  ∃ (lateralArea : ℝ), lateralArea = 4 * Real.pi * c.S := by
  sorry

end NUMINAMATH_CALUDE_special_cylinder_lateral_area_l3702_370287


namespace NUMINAMATH_CALUDE_power_two_mod_nine_l3702_370252

theorem power_two_mod_nine : 2 ^ 46655 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_nine_l3702_370252


namespace NUMINAMATH_CALUDE_inequality_chain_l3702_370246

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l3702_370246


namespace NUMINAMATH_CALUDE_base5_123_to_base10_l3702_370250

/-- Converts a base-5 number represented as a list of digits to its base-10 equivalent -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Theorem: The base-10 representation of the base-5 number 123 is 38 -/
theorem base5_123_to_base10 :
  base5ToBase10 [3, 2, 1] = 38 := by
  sorry

#eval base5ToBase10 [3, 2, 1]

end NUMINAMATH_CALUDE_base5_123_to_base10_l3702_370250


namespace NUMINAMATH_CALUDE_circle_C_properties_l3702_370203

-- Define the circles and points
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2
def point_P : ℝ × ℝ := (1, 1)
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the vector dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the theorem
theorem circle_C_properties
  (r : ℝ)
  (h_r : r > 0)
  (h_symmetry : ∀ x y, circle_C x y ↔ 
    ∃ x' y', circle_M r x' y' ∧ symmetry_line ((x + x')/2) ((y + y')/2))
  (h_P_on_C : circle_C point_P.1 point_P.2)
  (h_complementary_slopes : ∀ A B : ℝ × ℝ, 
    circle_C A.1 A.2 → circle_C B.1 B.2 → 
    (A.2 - point_P.2) * (B.2 - point_P.2) = -(A.1 - point_P.1) * (B.1 - point_P.1)) :
  (∀ x y, circle_C x y ↔ x^2 + y^2 = 2) ∧
  (∀ Q : ℝ × ℝ, point_Q Q.1 Q.2 → 
    dot_product (Q.1 - point_P.1, Q.2 - point_P.2) (Q.1 + 2, Q.2 + 2) ≥ -4) ∧
  (∀ A B : ℝ × ℝ, circle_C A.1 A.2 → circle_C B.1 B.2 → A ≠ B →
    (A.2 - B.2) * point_P.1 = (A.1 - B.1) * point_P.2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l3702_370203


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3702_370278

/-- Represents a pentagon with angles P, Q, R, S, and T -/
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

/-- The sum of angles in a pentagon is 540° -/
axiom pentagon_angle_sum (p : Pentagon) : p.P + p.Q + p.R + p.S + p.T = 540

/-- Theorem: In a pentagon PQRST where P = 70°, Q = 110°, R = S, and T = 3R + 20°,
    the measure of the largest angle is 224° -/
theorem largest_angle_in_pentagon (p : Pentagon)
  (h1 : p.P = 70)
  (h2 : p.Q = 110)
  (h3 : p.R = p.S)
  (h4 : p.T = 3 * p.R + 20) :
  max p.P (max p.Q (max p.R (max p.S p.T))) = 224 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3702_370278


namespace NUMINAMATH_CALUDE_circle_assignment_exists_l3702_370233

structure Circle where
  value : ℕ

structure Graph where
  A : Circle
  B : Circle
  C : Circle
  D : Circle

def connected (x y : Circle) (g : Graph) : Prop :=
  (x = g.A ∧ y = g.B) ∨ (x = g.B ∧ y = g.A) ∨
  (x = g.A ∧ y = g.D) ∨ (x = g.D ∧ y = g.A) ∨
  (x = g.B ∧ y = g.C) ∨ (x = g.C ∧ y = g.B)

def ratio (x y : Circle) : ℚ :=
  (x.value : ℚ) / (y.value : ℚ)

theorem circle_assignment_exists : ∃ g : Graph,
  (∀ x y : Circle, connected x y g → (ratio x y = 3 ∨ ratio x y = 9)) ∧
  (∀ x y : Circle, ¬connected x y g → (ratio x y ≠ 3 ∧ ratio x y ≠ 9)) :=
sorry

end NUMINAMATH_CALUDE_circle_assignment_exists_l3702_370233


namespace NUMINAMATH_CALUDE_find_M_l3702_370269

theorem find_M : ∃ M : ℕ+, (18^2 * 45^2 : ℕ) = 30^2 * M^2 ∧ M = 81 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3702_370269


namespace NUMINAMATH_CALUDE_special_rhombus_perimeter_l3702_370219

/-- A rhombus with integer side lengths where the area equals the perimeter -/
structure SpecialRhombus where
  side_length : ℕ
  area_eq_perimeter : (side_length ^ 2 * Real.sin (π / 6)) = (4 * side_length)

/-- The perimeter of a SpecialRhombus is 32 -/
theorem special_rhombus_perimeter (r : SpecialRhombus) : 4 * r.side_length = 32 := by
  sorry

#check special_rhombus_perimeter

end NUMINAMATH_CALUDE_special_rhombus_perimeter_l3702_370219


namespace NUMINAMATH_CALUDE_work_completion_days_l3702_370283

/-- Proves that the original number of days planned to complete the work is 15,
    given the conditions of the problem. -/
theorem work_completion_days : ∀ (total_men : ℕ) (absent_men : ℕ) (actual_days : ℕ),
  total_men = 48 →
  absent_men = 8 →
  actual_days = 18 →
  (total_men - absent_men) * actual_days = total_men * 15 :=
by
  sorry

#check work_completion_days

end NUMINAMATH_CALUDE_work_completion_days_l3702_370283


namespace NUMINAMATH_CALUDE_celine_change_l3702_370262

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine buys -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine buys -/
def smartphones_bought : ℕ := 4

/-- The amount of money Celine has in dollars -/
def money_available : ℕ := 3000

/-- The change Celine receives after her purchase -/
def change : ℕ := money_available - (laptop_price * laptops_bought + smartphone_price * smartphones_bought)

theorem celine_change : change = 200 := by
  sorry

end NUMINAMATH_CALUDE_celine_change_l3702_370262
