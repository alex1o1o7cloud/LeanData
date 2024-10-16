import Mathlib

namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2697_269709

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 2 →
  Complex.abs b = Real.sqrt 26 →
  a * b = t - 2 * Complex.I →
  t > 0 →
  t = 10 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2697_269709


namespace NUMINAMATH_CALUDE_eighth_term_is_15_l2697_269727

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem eighth_term_is_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_15_l2697_269727


namespace NUMINAMATH_CALUDE_cubic_planes_quintic_planes_l2697_269788

/-- The type of points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The condition for the cubic equation -/
def cubicCondition (p : Point3D) : Prop :=
  p.x^3 + p.y^3 + p.z^3 = (p.x + p.y + p.z)^3

/-- The condition for the quintic equation -/
def quinticCondition (p : Point3D) : Prop :=
  p.x^5 + p.y^5 + p.z^5 = (p.x + p.y + p.z)^5

/-- The theorem for the cubic case -/
theorem cubic_planes (p : Point3D) : 
  cubicCondition p → (p.x + p.y) * (p.y + p.z) * (p.z + p.x) = 0 := by
  sorry

/-- The theorem for the quintic case -/
theorem quintic_planes (p : Point3D) : 
  quinticCondition p → (p.x + p.y) * (p.y + p.z) * (p.z + p.x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_planes_quintic_planes_l2697_269788


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2697_269797

/-- The minimum distance from the origin to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 → Real.sqrt (x^2 + y^2) ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2697_269797


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l2697_269719

/-- Calculates the total cost of hiking equipment for Celina --/
theorem hiking_equipment_cost :
  let hoodie_cost : ℚ := 80
  let flashlight_cost : ℚ := 0.2 * hoodie_cost
  let boots_original : ℚ := 110
  let boots_discount : ℚ := 0.1
  let water_filter_original : ℚ := 65
  let water_filter_discount : ℚ := 0.25
  let camping_mat_original : ℚ := 45
  let camping_mat_discount : ℚ := 0.15
  let total_cost : ℚ := 
    hoodie_cost + 
    flashlight_cost + 
    (boots_original * (1 - boots_discount)) + 
    (water_filter_original * (1 - water_filter_discount)) + 
    (camping_mat_original * (1 - camping_mat_discount))
  total_cost = 282 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l2697_269719


namespace NUMINAMATH_CALUDE_sugar_measurement_l2697_269787

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 3 + 3 / 4 ∧ cup_capacity = 1 / 3 → 
  ↑(Int.ceil ((sugar_needed / cup_capacity) : ℚ)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_sugar_measurement_l2697_269787


namespace NUMINAMATH_CALUDE_total_tree_count_l2697_269717

def douglas_fir_count : ℕ := 350
def douglas_fir_cost : ℕ := 300
def ponderosa_pine_cost : ℕ := 225
def total_cost : ℕ := 217500

theorem total_tree_count : 
  ∃ (ponderosa_pine_count : ℕ),
    douglas_fir_count * douglas_fir_cost + 
    ponderosa_pine_count * ponderosa_pine_cost = total_cost ∧
    douglas_fir_count + ponderosa_pine_count = 850 :=
by sorry

end NUMINAMATH_CALUDE_total_tree_count_l2697_269717


namespace NUMINAMATH_CALUDE_unrolled_value_is_four_fifty_l2697_269735

/-- The number of quarters -/
def total_quarters : ℕ := 100

/-- The number of dimes -/
def total_dimes : ℕ := 185

/-- The capacity of a roll of quarters -/
def quarters_per_roll : ℕ := 45

/-- The capacity of a roll of dimes -/
def dimes_per_roll : ℕ := 55

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1/10

/-- The total dollar value of coins that cannot be rolled -/
def unrolled_value : ℚ :=
  (total_quarters % quarters_per_roll) * quarter_value +
  (total_dimes % dimes_per_roll) * dime_value

theorem unrolled_value_is_four_fifty :
  unrolled_value = 9/2 := by sorry

end NUMINAMATH_CALUDE_unrolled_value_is_four_fifty_l2697_269735


namespace NUMINAMATH_CALUDE_parabola_line_slope_l2697_269773

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - focus.1)

-- Define the condition for a point to be on both the line and the parabola
def intersection_point (k : ℝ) (x y : ℝ) : Prop :=
  parabola x y ∧ line_through_focus k x y

-- Define the ratio condition
def ratio_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = 16 * ((B.1 - focus.1)^2 + (B.2 - focus.2)^2)

theorem parabola_line_slope (k : ℝ) (A B : ℝ × ℝ) :
  intersection_point k A.1 A.2 →
  intersection_point k B.1 B.2 →
  A ≠ B →
  ratio_condition A B →
  k = 4/3 ∨ k = -4/3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l2697_269773


namespace NUMINAMATH_CALUDE_inequality_solution_l2697_269725

theorem inequality_solution (x : ℝ) : 1 - 1 / (3 * x + 4) < 3 ↔ x < -5/3 ∨ x > -4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2697_269725


namespace NUMINAMATH_CALUDE_total_packs_bought_l2697_269771

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 2

/-- The number of classes John has -/
def num_classes : ℕ := 6

/-- The number of students in each of John's classes -/
def students_per_class : ℕ := 30

/-- Theorem: John buys 360 packs of index cards in total -/
theorem total_packs_bought : packs_per_student * num_classes * students_per_class = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_bought_l2697_269771


namespace NUMINAMATH_CALUDE_park_benches_l2697_269748

theorem park_benches (bench_capacity : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) : 
  bench_capacity = 4 →
  people_sitting = 80 →
  spaces_available = 120 →
  (people_sitting + spaces_available) / bench_capacity = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_benches_l2697_269748


namespace NUMINAMATH_CALUDE_fifteenth_term_equals_44_l2697_269766

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 15th term of the specific arithmetic progression -/
def fifteenthTerm : ℝ :=
  arithmeticProgressionTerm 2 3 15

theorem fifteenth_term_equals_44 : fifteenthTerm = 44 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_equals_44_l2697_269766


namespace NUMINAMATH_CALUDE_sum_of_roots_l2697_269761

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2697_269761


namespace NUMINAMATH_CALUDE_simplify_fraction_l2697_269772

theorem simplify_fraction (a b : ℝ) (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2*a + 1) / (1 - b / (2*b - 1)) = (2*a + 1) * (2*b - 1) / (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2697_269772


namespace NUMINAMATH_CALUDE_cone_volume_l2697_269767

-- Define the right triangle
structure RightTriangle where
  area : ℝ
  centroidCircumference : ℝ

-- Define the cone formed by rotating the right triangle
structure Cone where
  triangle : RightTriangle

-- Define the volume of the cone
def volume (c : Cone) : ℝ := c.triangle.area * c.triangle.centroidCircumference

-- Theorem statement
theorem cone_volume (c : Cone) : volume c = c.triangle.area * c.triangle.centroidCircumference := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2697_269767


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2697_269729

theorem quadratic_equation_solutions (a b : ℝ) :
  (∀ x : ℝ, x = -1 ∨ x = 2 → -a * x^2 + b * x = -2) →
  (-a * (-1)^2 + b * (-1) + 2 = 0) ∧ (-a * 2^2 + b * 2 + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2697_269729


namespace NUMINAMATH_CALUDE_win_sector_area_l2697_269777

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the WIN sector is 75π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  win_area = win_prob * π * radius^2 →
  win_area = 75 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l2697_269777


namespace NUMINAMATH_CALUDE_cans_distribution_l2697_269744

theorem cans_distribution (father_weight son_weight : ℕ) 
  (h1 : father_weight = 6500)
  (h2 : son_weight = 2600) : 
  ∃ (can_weight : ℕ), 
    300 ≤ can_weight ∧ 
    can_weight ≤ 400 ∧
    father_weight % can_weight = 0 ∧
    son_weight % can_weight = 0 ∧
    father_weight / can_weight = 20 ∧
    son_weight / can_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_cans_distribution_l2697_269744


namespace NUMINAMATH_CALUDE_sequence_properties_l2697_269720

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sum_S n = 2 * sequence_a n - sequence_a 1) ∧
  (sequence_a 1 + sequence_a 3 = 2 * (sequence_a 2 + 1)) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n) ∧
  (∀ n : ℕ, n > 0 → sequence_T n = 2 - (n + 2) / 2^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2697_269720


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l2697_269793

theorem odd_prime_square_root (p k : ℕ) : 
  Prime p → 
  Odd p → 
  k > 0 → 
  ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k → 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l2697_269793


namespace NUMINAMATH_CALUDE_all_reals_satisfy_property_l2697_269736

theorem all_reals_satisfy_property :
  ∀ (α : ℝ), ∀ (n : ℕ), n > 0 → ∃ (m : ℤ), |α - (m : ℝ) / n| < 1 / (3 * n) :=
by sorry

end NUMINAMATH_CALUDE_all_reals_satisfy_property_l2697_269736


namespace NUMINAMATH_CALUDE_binomial_cube_plus_one_l2697_269739

theorem binomial_cube_plus_one : 7^3 + 3*(7^2) + 3*7 + 2 = 513 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_plus_one_l2697_269739


namespace NUMINAMATH_CALUDE_fraction_addition_and_simplification_l2697_269726

theorem fraction_addition_and_simplification :
  ∃ (n d : ℤ), (8 : ℚ) / 19 + (5 : ℚ) / 57 = n / d ∧ 
  n / d = (29 : ℚ) / 57 ∧
  (∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_addition_and_simplification_l2697_269726


namespace NUMINAMATH_CALUDE_line_equation_l2697_269755

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

/-- The main theorem -/
theorem line_equation (l : Line) (m a b : Point) : 
  pointOnLine m l → 
  m.x = -1 ∧ m.y = 2 →
  a.y = 0 →
  b.x = 0 →
  isMidpoint m a b →
  l.a = 2 ∧ l.b = -1 ∧ l.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2697_269755


namespace NUMINAMATH_CALUDE_swimming_scenario_l2697_269749

/-- The number of weeks in the swimming scenario -/
def weeks : ℕ := 4

/-- Camden's total number of swims -/
def camden_total : ℕ := 16

/-- Susannah's total number of swims -/
def susannah_total : ℕ := 24

/-- Camden's swims per week -/
def camden_per_week : ℚ := camden_total / weeks

/-- Susannah's swims per week -/
def susannah_per_week : ℚ := susannah_total / weeks

theorem swimming_scenario :
  (susannah_per_week = camden_per_week + 2) ∧
  (camden_per_week * weeks = camden_total) ∧
  (susannah_per_week * weeks = susannah_total) :=
by sorry

end NUMINAMATH_CALUDE_swimming_scenario_l2697_269749


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2697_269723

theorem simultaneous_equations_solution (n : ℕ+) (u v : ℝ) :
  (∃ (a b c : ℕ+),
    (a^2 + b^2 + c^2 : ℝ) = 169 * (n : ℝ)^2 ∧
    (a^2 * (u * a^2 + v * b^2) + b^2 * (u * b^2 + v * c^2) + c^2 * (u * c^2 + v * a^2) : ℝ) = 
      ((2 * u + v) * (13 * (n : ℝ))^4) / 4) ↔
  v = 2 * u :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2697_269723


namespace NUMINAMATH_CALUDE_calculation_proof_factorization_proof_l2697_269792

-- Problem 1
theorem calculation_proof : 
  |Real.sqrt 3 - 1| - 4 * Real.sin (π / 6) + (1 / 2)⁻¹ + (4 - π)^0 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem factorization_proof (a : ℝ) : 
  2 * a^3 - 12 * a^2 + 18 * a = 2 * a * (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_factorization_proof_l2697_269792


namespace NUMINAMATH_CALUDE_expression_value_l2697_269703

theorem expression_value : 2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2697_269703


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l2697_269774

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ k : ℝ, (a - I) / (1 + I) = k * I) → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l2697_269774


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2697_269786

theorem second_polygon_sides (n : ℕ) (s : ℝ) : 
  s > 0 → 
  50 * (3 * s) = n * s → 
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2697_269786


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l2697_269769

def original_mean : ℝ := 36
def num_observations : ℕ := 50
def error_1 : (ℝ × ℝ) := (46, 23)
def error_2 : (ℝ × ℝ) := (55, 40)
def error_3 : (ℝ × ℝ) := (28, 15)

theorem corrected_mean_calculation :
  let total_sum := original_mean * num_observations
  let error_sum := error_1.1 + error_2.1 + error_3.1 - (error_1.2 + error_2.2 + error_3.2)
  let corrected_sum := total_sum + error_sum
  corrected_sum / num_observations = 37.02 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l2697_269769


namespace NUMINAMATH_CALUDE_bridge_length_l2697_269715

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2697_269715


namespace NUMINAMATH_CALUDE_flagpole_length_correct_flagpole_length_is_60_l2697_269779

/-- The length of the flagpole in feet. -/
def flagpole_length : ℝ := 60

/-- The total distance the flag moves up and down the pole in feet. -/
def total_flag_movement : ℝ := 180

/-- Theorem stating that the flagpole length is correct given the total flag movement. -/
theorem flagpole_length_correct :
  flagpole_length * 3 = total_flag_movement :=
by sorry

/-- Theorem proving that the flagpole length is 60 feet. -/
theorem flagpole_length_is_60 :
  flagpole_length = 60 :=
by sorry

end NUMINAMATH_CALUDE_flagpole_length_correct_flagpole_length_is_60_l2697_269779


namespace NUMINAMATH_CALUDE_ellipse_midpoint_y_coordinate_l2697_269701

/-- The y-coordinate of the midpoint M of PF, where F is a focus of the ellipse
    x^2/12 + y^2/3 = 1 and P is a point on the ellipse such that M lies on the y-axis. -/
theorem ellipse_midpoint_y_coordinate (x_P y_P : ℝ) (x_F y_F : ℝ) :
  x_P^2 / 12 + y_P^2 / 3 = 1 →  -- P is on the ellipse
  x_F^2 = 9 ∧ y_F = 0 →  -- F is a focus
  (x_P + x_F) / 2 = 0 →  -- M is on the y-axis
  ∃ (y_M : ℝ), y_M = (y_P + y_F) / 2 ∧ y_M^2 = 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_midpoint_y_coordinate_l2697_269701


namespace NUMINAMATH_CALUDE_triangle_area_from_circle_and_chord_data_l2697_269732

/-- Given a circle and a triangle circumscribed around it, this theorem proves
    the area of the triangle based on given measurements. -/
theorem triangle_area_from_circle_and_chord_data (R : ℝ) (chord_length : ℝ) (center_to_chord : ℝ) (perimeter : ℝ)
  (h1 : chord_length = 16)
  (h2 : center_to_chord = 15)
  (h3 : perimeter = 200)
  (h4 : R^2 = center_to_chord^2 + (chord_length/2)^2) :
  R * (perimeter / 2) = 1700 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_circle_and_chord_data_l2697_269732


namespace NUMINAMATH_CALUDE_orange_seller_loss_percentage_l2697_269759

theorem orange_seller_loss_percentage :
  ∀ (cost_price selling_price_10 selling_price_6 : ℚ),
    cost_price > 0 →
    selling_price_10 = 1 / 10 →
    selling_price_6 = 1 / 6 →
    selling_price_6 = 3/2 * cost_price →
    (cost_price - selling_price_10) / cost_price * 100 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_loss_percentage_l2697_269759


namespace NUMINAMATH_CALUDE_grade_calculation_l2697_269728

/-- Represents the weighted average calculation for a student's grades --/
def weighted_average (math history science geography : ℝ) : ℝ :=
  0.3 * math + 0.3 * history + 0.2 * science + 0.2 * geography

/-- Theorem stating the conditions and the result to be proven --/
theorem grade_calculation (math history science geography : ℝ) :
  math = 74 →
  history = 81 →
  science = geography + 5 →
  science ≥ 75 →
  science = 86.25 →
  geography = 81.25 →
  weighted_average math history science geography = 80 :=
by
  sorry

#eval weighted_average 74 81 86.25 81.25

end NUMINAMATH_CALUDE_grade_calculation_l2697_269728


namespace NUMINAMATH_CALUDE_probability_diamond_spade_heart_l2697_269785

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- Calculates the probability of drawing a specific sequence of cards -/
def probability_specific_sequence (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.spades : ℚ) / (d.cards - 1) *
  (d.hearts : ℚ) / (d.cards - 2)

/-- A standard deck of 52 cards with 13 cards of each suit -/
def standard_deck : Deck :=
  { cards := 52,
    diamonds := 13,
    spades := 13,
    hearts := 13 }

theorem probability_diamond_spade_heart :
  probability_specific_sequence standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_spade_heart_l2697_269785


namespace NUMINAMATH_CALUDE_exp_addition_property_l2697_269753

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by sorry

end NUMINAMATH_CALUDE_exp_addition_property_l2697_269753


namespace NUMINAMATH_CALUDE_grape_bowls_problem_l2697_269764

theorem grape_bowls_problem (n : ℕ) : 
  (8 * 12 = 6 * n) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_grape_bowls_problem_l2697_269764


namespace NUMINAMATH_CALUDE_smallest_sum_of_pairwise_sums_l2697_269722

theorem smallest_sum_of_pairwise_sums (a b c d : ℝ) (y : ℝ) : 
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  ({170, 305, 270, 255, 320, y} : Set ℝ) = sums →
  (320 ∈ sums) →
  (∀ z ∈ sums, 320 + y ≤ z + y) →
  320 + y = 255 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_pairwise_sums_l2697_269722


namespace NUMINAMATH_CALUDE_simplify_expression_l2697_269789

theorem simplify_expression (b : ℚ) (h : b = 2) :
  (15 * b^4 - 45 * b^3) / (75 * b^2) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2697_269789


namespace NUMINAMATH_CALUDE_school_population_l2697_269796

theorem school_population (total boys girls : ℕ) : 
  (total = boys + girls) →
  (boys = 50 → girls = total / 2) →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_school_population_l2697_269796


namespace NUMINAMATH_CALUDE_two_out_of_three_correct_probability_l2697_269763

def probability_correct_forecast : ℝ := 0.8

def probability_two_out_of_three_correct : ℝ :=
  3 * probability_correct_forecast^2 * (1 - probability_correct_forecast)

theorem two_out_of_three_correct_probability :
  probability_two_out_of_three_correct = 0.384 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_correct_probability_l2697_269763


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2697_269751

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (Q a ⊂ P ∧ Q a ≠ P) ↔ a ∈ ({0, 1/3, -1/2} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2697_269751


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l2697_269791

/-- The cost of a single photocopy --/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 photocopies --/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs --/
def copies_per_person : ℕ := 80

/-- The amount saved per person when ordering together --/
def savings_per_person : ℝ := 0.40

/-- Theorem stating the cost of a single photocopy --/
theorem photocopy_cost_calculation : photocopy_cost = 0.02 := by
  have h1 : 2 * copies_per_person * photocopy_cost - 
    (2 * copies_per_person * photocopy_cost * (1 - discount_rate)) = 
    2 * savings_per_person := by sorry
  
  -- The rest of the proof steps would go here
  sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l2697_269791


namespace NUMINAMATH_CALUDE_min_alpha_gamma_sum_l2697_269738

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (α γ : ℂ) (z : ℂ) : ℂ := (5 + 2*i)*z^3 + (4 + i)*z^2 + α*z + γ

-- State the theorem
theorem min_alpha_gamma_sum (α γ : ℂ) : 
  (f α γ 1).im = 0 → (f α γ i).im = 0 → (f α γ (-1)).im = 0 → 
  Complex.abs α + Complex.abs γ ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_alpha_gamma_sum_l2697_269738


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l2697_269795

theorem max_sum_of_roots (c b : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - c*x + b = 0 ∧ y^2 - c*y + b = 0 ∧ x - y = 1) →
  c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l2697_269795


namespace NUMINAMATH_CALUDE_range_of_a_for_zero_in_interval_l2697_269711

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem range_of_a_for_zero_in_interval :
  ∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ [-1, 1] ∧ f a x₀ = 0) → a ∈ [-1/2, -1/3] :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_zero_in_interval_l2697_269711


namespace NUMINAMATH_CALUDE_sine_inequality_unique_solution_l2697_269731

theorem sine_inequality_unique_solution :
  ∀ y ∈ Set.Icc 0 (Real.pi / 2),
    (∀ x ∈ Set.Icc 0 Real.pi, Real.sin (x + y) < Real.sin x + Real.sin y) ↔
    y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_unique_solution_l2697_269731


namespace NUMINAMATH_CALUDE_circle_portion_area_l2697_269716

/-- The area of the portion of the circle x^2 - 16x + y^2 = 51 that lies above the x-axis 
    and to the left of the line y = 10 - x is equal to 8π. -/
theorem circle_portion_area : 
  ∃ (A : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 = 51 → y ≥ 0 → y ≤ 10 - x → 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 - 16*p.1 + p.2^2 = 51 ∧ p.2 ≥ 0 ∧ p.2 ≤ 10 - p.1}) ∧
    A = Real.pi * 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_portion_area_l2697_269716


namespace NUMINAMATH_CALUDE_preimage_of_four_negative_two_l2697_269747

/-- Given a function f : ℝ × ℝ → ℝ × ℝ defined as f(x, y) = (x+y, x-y),
    prove that the pre-image of (4, -2) under f is (1, 3) -/
theorem preimage_of_four_negative_two (f : ℝ × ℝ → ℝ × ℝ) 
    (h : ∀ x y : ℝ, f (x, y) = (x + y, x - y)) :
  ∃ a b : ℝ, f (a, b) = (4, -2) ∧ (a, b) = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_negative_two_l2697_269747


namespace NUMINAMATH_CALUDE_equation_solutions_l2697_269758

theorem equation_solutions : 
  (∃ (s : Set ℝ), s = {0, 3} ∧ ∀ x ∈ s, 4 * x^2 = 12 * x) ∧
  (∃ (t : Set ℝ), t = {-3, -1} ∧ ∀ x ∈ t, x^2 + 4 * x + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2697_269758


namespace NUMINAMATH_CALUDE_bmw_sales_l2697_269718

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (nissan_percent : ℚ) (chevrolet_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_chevrolet : chevrolet_percent = 10 / 100)
  (h_sum : ford_percent + nissan_percent + chevrolet_percent < 1) :
  ↑total * (1 - (ford_percent + nissan_percent + chevrolet_percent)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_l2697_269718


namespace NUMINAMATH_CALUDE_commercial_length_l2697_269721

theorem commercial_length 
  (total_time : ℕ) 
  (long_commercial_count : ℕ) 
  (long_commercial_length : ℕ) 
  (short_commercial_count : ℕ) : 
  total_time = 37 ∧ 
  long_commercial_count = 3 ∧ 
  long_commercial_length = 5 ∧ 
  short_commercial_count = 11 → 
  (total_time - long_commercial_count * long_commercial_length) / short_commercial_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_commercial_length_l2697_269721


namespace NUMINAMATH_CALUDE_sibling_product_l2697_269737

/-- Represents a family with a specific structure -/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : ℕ × ℕ :=
  (f.sisters - 1, f.brothers)

/-- The main theorem stating the product of sisters and brothers for a sibling -/
theorem sibling_product (f : Family) (h1 : f.sisters = 6) (h2 : f.brothers = 3) :
  let (s, b) := sibling_count f
  s * b = 15 := by
  sorry

#check sibling_product

end NUMINAMATH_CALUDE_sibling_product_l2697_269737


namespace NUMINAMATH_CALUDE_height_difference_l2697_269710

theorem height_difference (height_B : ℝ) (height_A : ℝ) 
  (h : height_A = height_B * 1.25) : 
  (height_B - height_A) / height_A * 100 = -20 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l2697_269710


namespace NUMINAMATH_CALUDE_runner_problem_l2697_269757

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 20 / v + 11) → (40 / (v / 2) = 22) :=
by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l2697_269757


namespace NUMINAMATH_CALUDE_regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l2697_269724

theorem regular_pentagon_angle_excess : ℝ → Prop :=
  λ total_excess : ℝ =>
    -- Define a regular pentagon
    ∃ (interior_angle : ℝ),
      -- The sum of interior angles of a pentagon is (5-2)*180 = 540 degrees
      5 * interior_angle = 540 ∧
      -- The total excess over 90 degrees for all angles
      5 * (interior_angle - 90) = total_excess ∧
      -- The theorem to prove
      total_excess = 90

-- The proof of the theorem
theorem prove_regular_pentagon_angle_excess :
  ∃ total_excess : ℝ, regular_pentagon_angle_excess total_excess :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_angle_excess_prove_regular_pentagon_angle_excess_l2697_269724


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2697_269705

/-- Proves that given a cycle sold at a loss of 18% with a selling price of Rs. 1558, the original price of the cycle is Rs. 1900. -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1558)
  (h2 : loss_percentage = 18) : 
  ∃ (original_price : ℝ), 
    original_price = 1900 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2697_269705


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l2697_269704

theorem trig_product_equals_one :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let cos60 : ℝ := 1/2
  (1 - 1/sin30) * (1 + 1/cos60) * (1 - 1/cos30) * (1 + 1/sin60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l2697_269704


namespace NUMINAMATH_CALUDE_average_of_xyz_l2697_269733

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_l2697_269733


namespace NUMINAMATH_CALUDE_sum_of_digits_of_2012_power_l2697_269750

def A : ℕ := 2012^2012

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def B : ℕ := sum_of_digits A
def C : ℕ := sum_of_digits B
def D : ℕ := sum_of_digits C

theorem sum_of_digits_of_2012_power : D = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_2012_power_l2697_269750


namespace NUMINAMATH_CALUDE_inequality_proof_l2697_269756

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2697_269756


namespace NUMINAMATH_CALUDE_discount_equation_l2697_269700

theorem discount_equation (original_price final_price x : ℝ) 
  (h_original : original_price = 200)
  (h_final : final_price = 162)
  (h_positive : 0 < x ∧ x < 1) :
  original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_discount_equation_l2697_269700


namespace NUMINAMATH_CALUDE_max_pieces_on_chessboard_l2697_269730

/-- Represents a chessboard with red and blue pieces -/
structure Chessboard :=
  (size : Nat)
  (red_pieces : Finset (Nat × Nat))
  (blue_pieces : Finset (Nat × Nat))

/-- Counts the number of pieces of the opposite color that a piece can see -/
def count_opposite_color (board : Chessboard) (pos : Nat × Nat) (is_red : Bool) : Nat :=
  sorry

/-- Checks if the chessboard configuration is valid -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 200 ∧
  (∀ pos ∈ board.red_pieces, count_opposite_color board pos true = 5) ∧
  (∀ pos ∈ board.blue_pieces, count_opposite_color board pos false = 5)

/-- The main theorem stating the maximum number of pieces on the chessboard -/
theorem max_pieces_on_chessboard (board : Chessboard) :
  is_valid_configuration board →
  Finset.card board.red_pieces + Finset.card board.blue_pieces ≤ 3800 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_on_chessboard_l2697_269730


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2697_269713

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x^2 + 5*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2697_269713


namespace NUMINAMATH_CALUDE_false_statement_l2697_269734

-- Define the types for planes and lines
variable {α β : Plane} {m n : Line}

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem false_statement :
  ¬(∀ (α β : Plane) (m n : Line),
    (¬parallel_line_plane m α ∧ parallel α β ∧ contained_in n β) →
    parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_false_statement_l2697_269734


namespace NUMINAMATH_CALUDE_car_journey_speed_l2697_269790

theorem car_journey_speed (s : ℝ) (h : s > 0) : 
  let first_part := 0.4 * s
  let second_part := 0.6 * s
  let first_speed := 40
  let average_speed := 100
  let first_time := first_part / first_speed
  let total_time := s / average_speed
  ∃ d : ℝ, d > 0 ∧ second_part / d = total_time - first_time ∧ d = 120 := by
sorry

end NUMINAMATH_CALUDE_car_journey_speed_l2697_269790


namespace NUMINAMATH_CALUDE_linear_function_proof_l2697_269754

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the theorem
theorem linear_function_proof :
  ∃ (k b : ℝ),
    (linear_function k b 1 = 5) ∧
    (linear_function k b (-1) = -1) ∧
    (∀ (x : ℝ), linear_function k b x = 3 * x + 2) ∧
    (linear_function k b 2 = 8) := by
  sorry


end NUMINAMATH_CALUDE_linear_function_proof_l2697_269754


namespace NUMINAMATH_CALUDE_sine_inequality_l2697_269742

theorem sine_inequality (n : ℕ) (x : ℝ) : 
  Real.sin x * (n * Real.sin x - Real.sin (n * x)) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2697_269742


namespace NUMINAMATH_CALUDE_picture_frame_interior_edges_sum_l2697_269765

theorem picture_frame_interior_edges_sum 
  (frame_width : ℝ) 
  (frame_area : ℝ) 
  (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 68 →
  outer_edge = 15 →
  ∃ (inner_width inner_height : ℝ),
    inner_width = outer_edge - 2 * frame_width ∧
    frame_area = outer_edge * (inner_height + 2 * frame_width) - inner_width * inner_height ∧
    2 * (inner_width + inner_height) = 26 :=
by sorry

end NUMINAMATH_CALUDE_picture_frame_interior_edges_sum_l2697_269765


namespace NUMINAMATH_CALUDE_cube_difference_l2697_269784

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) :
  a^3 - b^3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l2697_269784


namespace NUMINAMATH_CALUDE_trig_expression_value_l2697_269714

theorem trig_expression_value : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l2697_269714


namespace NUMINAMATH_CALUDE_lefty_points_lefty_scored_20_points_l2697_269768

theorem lefty_points : ℝ → Prop :=
  fun L : ℝ =>
    let righty : ℝ := L / 2
    let third_teammate : ℝ := 3 * L
    let total_points : ℝ := L + righty + third_teammate
    let average_points : ℝ := total_points / 3
    average_points = 30 → L = 20

-- Proof
theorem lefty_scored_20_points : ∃ L : ℝ, lefty_points L :=
  sorry

end NUMINAMATH_CALUDE_lefty_points_lefty_scored_20_points_l2697_269768


namespace NUMINAMATH_CALUDE_prism_volume_in_cubic_yards_l2697_269712

/-- Conversion factor from cubic feet to cubic yards -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- Volume of the rectangular prism in cubic feet -/
def prism_volume_cubic_feet : ℝ := 216

/-- Theorem stating that the volume of the prism in cubic yards is 8 -/
theorem prism_volume_in_cubic_yards :
  prism_volume_cubic_feet / cubic_feet_per_cubic_yard = 8 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_in_cubic_yards_l2697_269712


namespace NUMINAMATH_CALUDE_sand_pile_volume_l2697_269798

/-- The volume of a conical sand pile -/
theorem sand_pile_volume (d h r : ℝ) : 
  d = 10 →  -- diameter is 10 feet
  h = 0.6 * d →  -- height is 60% of diameter
  r = d / 2 →  -- radius is half of diameter
  (1 / 3) * π * r^2 * h = 50 * π := by
  sorry

end NUMINAMATH_CALUDE_sand_pile_volume_l2697_269798


namespace NUMINAMATH_CALUDE_daves_total_expense_l2697_269743

/-- The amount Dave spent on books -/
def daves_book_expense (animal_books outer_space_books train_books book_price : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * book_price

/-- Theorem stating the total amount Dave spent on books -/
theorem daves_total_expense : 
  daves_book_expense 8 6 3 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_daves_total_expense_l2697_269743


namespace NUMINAMATH_CALUDE_binomial_distribution_params_l2697_269706

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (bd : BinomialDistribution) : ℝ := bd.n * bd.p

/-- The variance of a binomial distribution -/
def variance (bd : BinomialDistribution) : ℝ := bd.n * bd.p * (1 - bd.p)

/-- Theorem: For a binomial distribution with expectation 8 and variance 1.6,
    the parameters are n = 10 and p = 0.8 -/
theorem binomial_distribution_params :
  ∀ (bd : BinomialDistribution),
    expectation bd = 8 →
    variance bd = 1.6 →
    bd.n = 10 ∧ bd.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_params_l2697_269706


namespace NUMINAMATH_CALUDE_pyramid_properties_l2697_269775

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  base_area : ℝ
  cross_section_area1 : ℝ
  cross_section_area2 : ℝ
  cross_section_distance : ℝ

/-- Calculates the distance of the larger cross section from the apex -/
def larger_cross_section_distance (p : RightOctagonalPyramid) : ℝ := sorry

/-- Calculates the total height of the pyramid -/
def total_height (p : RightOctagonalPyramid) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid -/
theorem pyramid_properties (p : RightOctagonalPyramid) 
  (h1 : p.base_area = 1200)
  (h2 : p.cross_section_area1 = 300 * Real.sqrt 2)
  (h3 : p.cross_section_area2 = 675 * Real.sqrt 2)
  (h4 : p.cross_section_distance = 10) :
  larger_cross_section_distance p = 30 ∧ total_height p = 40 := by sorry

end NUMINAMATH_CALUDE_pyramid_properties_l2697_269775


namespace NUMINAMATH_CALUDE_toms_incorrect_calculation_correct_calculation_l2697_269740

/-- The original number Tom was working with -/
def y : ℤ := 114

/-- Tom's incorrect calculation -/
theorem toms_incorrect_calculation : (y - 14) / 2 = 50 := by sorry

/-- The correct calculation -/
theorem correct_calculation : ((y - 5) / 7 : ℚ).floor = 15 := by sorry

end NUMINAMATH_CALUDE_toms_incorrect_calculation_correct_calculation_l2697_269740


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_8000_l2697_269760

theorem last_three_digits_of_3_to_8000 (h : 3^400 ≡ 1 [ZMOD 800]) :
  3^8000 ≡ 1 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_8000_l2697_269760


namespace NUMINAMATH_CALUDE_factorization_equality_l2697_269783

theorem factorization_equality (x y : ℝ) :
  x^2 * (y^2 - 1) + 2 * x * (y^2 - 1) + (y^2 - 1) = (y + 1) * (y - 1) * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2697_269783


namespace NUMINAMATH_CALUDE_tonys_laundry_problem_l2697_269708

/-- The problem of determining the weight of shirts in Tony's laundry. -/
theorem tonys_laundry_problem (
  wash_limit : ℕ)
  (sock_weight pants_weight shorts_weight underwear_weight : ℕ)
  (num_socks num_underwear : ℕ)
  (total_weight : ℕ → ℕ → ℕ → ℕ → ℕ) :
  wash_limit = 50 →
  sock_weight = 2 →
  pants_weight = 10 →
  shorts_weight = 8 →
  underwear_weight = 4 →
  num_socks = 3 →
  num_underwear = 4 →
  total_weight sock_weight pants_weight shorts_weight underwear_weight =
    sock_weight * num_socks + pants_weight + shorts_weight + underwear_weight * num_underwear →
  wash_limit - total_weight sock_weight pants_weight shorts_weight underwear_weight = 10 :=
by sorry

end NUMINAMATH_CALUDE_tonys_laundry_problem_l2697_269708


namespace NUMINAMATH_CALUDE_product_of_fractions_l2697_269799

theorem product_of_fractions : (1 / 3 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * (3 / 7 : ℚ) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2697_269799


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l2697_269746

theorem sum_of_roots_equals_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l2697_269746


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2697_269707

/-- The number of ways five people can sit in a row of six chairs -/
def seating_arrangements : ℕ :=
  let total_chairs : ℕ := 6
  let total_people : ℕ := 5
  let odd_numbered_chairs : ℕ := 3  -- chairs 1, 3, and 5
  odd_numbered_chairs * (total_chairs - 1) * (total_chairs - 2) * (total_chairs - 3) * (total_chairs - 4)

/-- Theorem stating that the number of seating arrangements is 360 -/
theorem seating_arrangements_count : seating_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2697_269707


namespace NUMINAMATH_CALUDE_log_equation_solution_l2697_269702

theorem log_equation_solution (x : ℝ) :
  x > 0 → (Real.log 4 / Real.log x = Real.log 3 / Real.log 27) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2697_269702


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2697_269762

/-- The sum of all terms in the geometric sequence {(2/3)^n, n ∈ ℕ*} is 2. -/
theorem geometric_sequence_sum : 
  let a : ℕ → ℝ := fun n => (2/3)^n
  ∑' n, a n = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2697_269762


namespace NUMINAMATH_CALUDE_no_real_roots_l2697_269794

theorem no_real_roots (m : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 4 ∧ ∀ x ∈ s, (x : ℝ) - m < 0 ∧ 7 - 2*(x : ℝ) ≤ 1) →
  ∀ x : ℝ, 8*x^2 - 8*x + m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l2697_269794


namespace NUMINAMATH_CALUDE_candy_bar_to_caramel_ratio_l2697_269780

/-- The price of caramel in dollars -/
def caramel_price : ℚ := 3

/-- The price of a candy bar as a multiple of the caramel price -/
def candy_bar_price (k : ℚ) : ℚ := k * caramel_price

/-- The price of cotton candy -/
def cotton_candy_price (k : ℚ) : ℚ := 2 * candy_bar_price k

/-- The total cost of 6 candy bars, 3 caramel, and 1 cotton candy -/
def total_cost (k : ℚ) : ℚ := 6 * candy_bar_price k + 3 * caramel_price + cotton_candy_price k

theorem candy_bar_to_caramel_ratio :
  ∃ k : ℚ, total_cost k = 57 ∧ candy_bar_price k / caramel_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_to_caramel_ratio_l2697_269780


namespace NUMINAMATH_CALUDE_circle_k_bound_l2697_269776

/-- A circle in the Cartesian plane --/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- The equation x^2 + y^2 - 2x + y + k = 0 represents a circle --/
def isCircle (k : ℝ) : Prop :=
  ∃ (c : Circle), ∀ (x y : ℝ), c.equation x y k ↔ x^2 + y^2 - 2*x + y + k = 0

/-- If x^2 + y^2 - 2x + y + k = 0 is the equation of a circle, then k < 5/4 --/
theorem circle_k_bound (k : ℝ) : isCircle k → k < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_k_bound_l2697_269776


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_midpoint_l2697_269778

theorem absolute_value_equality_implies_midpoint (x : ℚ) :
  |x - 2| = |x - 5| → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_midpoint_l2697_269778


namespace NUMINAMATH_CALUDE_distinct_results_count_l2697_269782

/-- Represents the possible operators that can replace * in the expression -/
inductive Operator
| Add
| Sub
| Mul
| Div

/-- Represents the expression as a list of operators -/
def Expression := List Operator

/-- Evaluates an expression according to the given rules -/
def evaluate (expr : Expression) : ℚ :=
  sorry

/-- Generates all possible expressions -/
def allExpressions : List Expression :=
  sorry

/-- Counts the number of distinct results -/
def countDistinctResults (exprs : List Expression) : ℕ :=
  sorry

/-- The main theorem stating that the number of distinct results is 15 -/
theorem distinct_results_count :
  countDistinctResults allExpressions = 15 := by
  sorry

end NUMINAMATH_CALUDE_distinct_results_count_l2697_269782


namespace NUMINAMATH_CALUDE_floor_sum_rationality_l2697_269745

theorem floor_sum_rationality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ⌊p * n⌋ + ⌊q * n⌋ + ⌊r * n⌋ = n) :
  (∃ a b c : ℤ, p = a / b ∧ q = a / c) ∧ p + q + r = 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_rationality_l2697_269745


namespace NUMINAMATH_CALUDE_cliff_rock_ratio_l2697_269770

/-- Represents Cliff's rock collection -/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- The properties of Cliff's rock collection -/
def cliffCollection : RockCollection where
  igneous := 90
  sedimentary := 180
  shinyIgneous := 30
  shinySedimentary := 36

theorem cliff_rock_ratio :
  let c := cliffCollection
  c.igneous + c.sedimentary = 270 ∧
  c.shinyIgneous = 30 ∧
  c.shinyIgneous = c.igneous / 3 ∧
  c.shinySedimentary = c.sedimentary / 5 →
  c.igneous / c.sedimentary = 1 / 2 := by
  sorry

#check cliff_rock_ratio

end NUMINAMATH_CALUDE_cliff_rock_ratio_l2697_269770


namespace NUMINAMATH_CALUDE_problem_statements_l2697_269781

theorem problem_statements :
  (∀ x : ℝ, x ≥ 0 → x + 1 + 1 / (x + 1) ≥ 2) ∧
  (∀ x : ℝ, x > 0 → (x + 1) / Real.sqrt x ≥ 2) ∧
  (∃ x : ℝ, x + 1 / x < 2) ∧
  (∀ x : ℝ, Real.sqrt (x^2 + 2) + 1 / Real.sqrt (x^2 + 2) > 2) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statements_l2697_269781


namespace NUMINAMATH_CALUDE_apples_in_baskets_l2697_269741

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (apples_removed : ℕ) : 
  total_apples = 64 → num_baskets = 4 → apples_removed = 3 →
  (total_apples / num_baskets) - apples_removed = 13 :=
by
  sorry

#check apples_in_baskets

end NUMINAMATH_CALUDE_apples_in_baskets_l2697_269741


namespace NUMINAMATH_CALUDE_event_probability_l2697_269752

noncomputable def probability_event (a b : Real) : Real :=
  (min b (3/2) - max a 0) / (b - a)

theorem event_probability : probability_event 0 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l2697_269752
