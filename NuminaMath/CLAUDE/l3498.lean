import Mathlib

namespace NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l3498_349803

theorem floor_sqrt_sum_eq_floor_sqrt_4n_plus_2 (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_sum_eq_floor_sqrt_4n_plus_2_l3498_349803


namespace NUMINAMATH_CALUDE_fraction_equality_l3498_349879

theorem fraction_equality (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : (x + y) / y = 2.2) :
  w / y = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3498_349879


namespace NUMINAMATH_CALUDE_arccos_cos_three_l3498_349805

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l3498_349805


namespace NUMINAMATH_CALUDE_factor_polynomial_l3498_349856

theorem factor_polynomial (x : ℝ) : 
  36 * x^6 - 189 * x^12 + 81 * x^9 = 9 * x^6 * (4 + 9 * x^3 - 21 * x^6) := by sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3498_349856


namespace NUMINAMATH_CALUDE_boom_boom_language_size_l3498_349842

/-- The number of letters in the Boom-Boom tribe's alphabet -/
def alphabet_size : ℕ := 6

/-- The length of a word in the Boom-Boom tribe's language -/
def word_length : ℕ := 6

/-- The number of words in the Boom-Boom tribe's language -/
def num_words : ℕ := alphabet_size ^ word_length - Nat.factorial alphabet_size

theorem boom_boom_language_size :
  num_words = 45936 :=
sorry

end NUMINAMATH_CALUDE_boom_boom_language_size_l3498_349842


namespace NUMINAMATH_CALUDE_length_units_ordering_l3498_349874

-- Define an enumeration for length units
inductive LengthUnit
  | Kilometer
  | Meter
  | Centimeter
  | Millimeter

-- Define a function to compare two length units
def isLargerThan (a b : LengthUnit) : Prop :=
  match a, b with
  | LengthUnit.Kilometer, _ => a ≠ b
  | LengthUnit.Meter, LengthUnit.Centimeter => True
  | LengthUnit.Meter, LengthUnit.Millimeter => True
  | LengthUnit.Centimeter, LengthUnit.Millimeter => True
  | _, _ => False

-- Theorem to prove the correct ordering of length units
theorem length_units_ordering :
  isLargerThan LengthUnit.Kilometer LengthUnit.Meter ∧
  isLargerThan LengthUnit.Meter LengthUnit.Centimeter ∧
  isLargerThan LengthUnit.Centimeter LengthUnit.Millimeter :=
by sorry


end NUMINAMATH_CALUDE_length_units_ordering_l3498_349874


namespace NUMINAMATH_CALUDE_emilys_cards_l3498_349824

theorem emilys_cards (initial_cards : ℕ) (cards_per_apple : ℕ) (bruce_apples : ℕ) :
  initial_cards + cards_per_apple * bruce_apples = 
  initial_cards + cards_per_apple * bruce_apples := by
  sorry

#check emilys_cards 63 7 13

end NUMINAMATH_CALUDE_emilys_cards_l3498_349824


namespace NUMINAMATH_CALUDE_soap_box_width_l3498_349884

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    prove that the width of each soap box is 7 inches -/
theorem soap_box_width
  (carton : BoxDimensions)
  (soap_box : BoxDimensions)
  (max_soap_boxes : ℕ)
  (h1 : carton.length = 25)
  (h2 : carton.width = 42)
  (h3 : carton.height = 60)
  (h4 : soap_box.length = 6)
  (h5 : soap_box.height = 6)
  (h6 : max_soap_boxes = 250)
  (h7 : max_soap_boxes * boxVolume soap_box = boxVolume carton) :
  soap_box.width = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_width_l3498_349884


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3498_349861

/-- The diagonal length of a rectangular prism with given surface area and total edge length -/
theorem rectangular_prism_diagonal
  (x y z : ℝ)  -- lengths of sides
  (h1 : 2*x*y + 2*x*z + 2*y*z = 22)  -- surface area condition
  (h2 : 4*x + 4*y + 4*z = 24)  -- total edge length condition
  : ∃ d : ℝ, d^2 = x^2 + y^2 + z^2 ∧ d^2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3498_349861


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3498_349829

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_theorem (a b c d : ℝ) :
  P a b c d 1 = 7 →
  P a b c d 2 = 52 →
  P a b c d 3 = 97 →
  (P a b c d 9 + P a b c d (-5)) / 4 = 1202 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3498_349829


namespace NUMINAMATH_CALUDE_x_squared_plus_3x_plus_4_range_l3498_349834

theorem x_squared_plus_3x_plus_4_range :
  ∀ x : ℝ, x^2 - 8*x + 15 < 0 → 22 < x^2 + 3*x + 4 ∧ x^2 + 3*x + 4 < 44 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_3x_plus_4_range_l3498_349834


namespace NUMINAMATH_CALUDE_shifted_function_is_linear_l3498_349870

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Represents a horizontal shift transformation on a function -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - shift)

/-- The original direct proportion function y = -2x -/
def originalFunction : ℝ → ℝ :=
  λ x => -2 * x

/-- The result of shifting the original function 3 units to the right -/
def shiftedFunction : ℝ → ℝ :=
  horizontalShift originalFunction 3

theorem shifted_function_is_linear :
  ∃ (k b : ℝ), k ≠ 0 ∧ (∀ x, shiftedFunction x = k * x + b) ∧ k = -2 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_shifted_function_is_linear_l3498_349870


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_largest_l3498_349885

theorem triangle_angle_ratio_largest (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = 2 * a →              -- second angle is twice the first
  c = 3 * a →              -- third angle is thrice the first
  max a (max b c) = 90     -- the largest angle is 90°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_largest_l3498_349885


namespace NUMINAMATH_CALUDE_derivative_of_f_l3498_349836

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x + 9

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3498_349836


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3498_349828

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes with no empty boxes -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes with no empty boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l3498_349828


namespace NUMINAMATH_CALUDE_m_value_l3498_349869

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_m_value_l3498_349869


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3498_349897

/-- Given 4 persons, if replacing one person with a new person weighing 129 kg
    increases the average weight by 8.5 kg, then the weight of the replaced person was 95 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 4)
  (h2 : weight_increase = 8.5)
  (h3 : new_person_weight = 129) :
  new_person_weight - (initial_count : ℝ) * weight_increase = 95 := by
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3498_349897


namespace NUMINAMATH_CALUDE_kim_total_points_l3498_349882

/-- Calculates the total points in a math contest with three rounds -/
def totalPoints (easyPoints averagePoints hardPoints : ℕ) 
                (easyCorrect averageCorrect hardCorrect : ℕ) : ℕ :=
  easyPoints * easyCorrect + averagePoints * averageCorrect + hardPoints * hardCorrect

/-- Theorem: Kim's total points in the contest -/
theorem kim_total_points :
  totalPoints 2 3 5 6 2 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_kim_total_points_l3498_349882


namespace NUMINAMATH_CALUDE_division_simplification_l3498_349898

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (12 * x^2 * y) / (-6 * x * y) = -2 * x :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l3498_349898


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3498_349844

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + Complex.I) * Complex.I^3 / (1 - Complex.I) = 1 - Complex.I →
  z.im = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3498_349844


namespace NUMINAMATH_CALUDE_pepperjack_cheese_probability_l3498_349866

theorem pepperjack_cheese_probability :
  let cheddar : ℕ := 15
  let mozzarella : ℕ := 30
  let pepperjack : ℕ := 45
  let total : ℕ := cheddar + mozzarella + pepperjack
  (pepperjack : ℚ) / (total : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pepperjack_cheese_probability_l3498_349866


namespace NUMINAMATH_CALUDE_exam_mean_score_l3498_349862

/-- Given an exam where a score of 58 is 2 standard deviations below the mean
    and a score of 98 is 3 standard deviations above the mean,
    prove that the mean score is 74. -/
theorem exam_mean_score (mean std_dev : ℝ) 
    (h1 : 58 = mean - 2 * std_dev)
    (h2 : 98 = mean + 3 * std_dev) : 
  mean = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3498_349862


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l3498_349847

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∃ x y z : ℝ, x = 1 ∧ y = (1/2 : ℝ) ∧ z = 4 ∧
    ∀ t : ℝ, a * t^3 + b * t^2 + c * t + d = 0 ↔ t = x ∨ t = y ∨ t = z) →
  c / d = -(13/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l3498_349847


namespace NUMINAMATH_CALUDE_teammates_average_points_l3498_349855

/-- Proves that the teammates' average points per game is 40, given Wade's average and team total -/
theorem teammates_average_points (wade_avg : ℝ) (team_total : ℝ) (num_games : ℕ) : 
  wade_avg = 20 →
  team_total = 300 →
  num_games = 5 →
  (team_total - wade_avg * num_games) / num_games = 40 := by
  sorry

end NUMINAMATH_CALUDE_teammates_average_points_l3498_349855


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3498_349801

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3498_349801


namespace NUMINAMATH_CALUDE_max_xy_value_l3498_349883

theorem max_xy_value (x y : ℕ) (h : 69 * x + 54 * y ≤ 2008) : x * y ≤ 270 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l3498_349883


namespace NUMINAMATH_CALUDE_triangles_containing_center_201_l3498_349850

/-- Given a regular 201-sided polygon inscribed in a circle with center C,
    this function computes the number of triangles formed by connecting
    any three vertices of the polygon such that C lies inside the triangle. -/
def triangles_containing_center (n : ℕ) : ℕ :=
  if n = 201 then
    let vertex_count := n
    let half_vertex_count := (vertex_count - 1) / 2
    let triangles_per_vertex := half_vertex_count * (half_vertex_count + 1) / 2
    vertex_count * triangles_per_vertex / 3
  else
    0

/-- Theorem stating that the number of triangles containing the center
    for a regular 201-sided polygon is 338350. -/
theorem triangles_containing_center_201 :
  triangles_containing_center 201 = 338350 := by
  sorry

end NUMINAMATH_CALUDE_triangles_containing_center_201_l3498_349850


namespace NUMINAMATH_CALUDE_overlap_length_l3498_349807

/-- Given red line segments with equal lengths and overlaps, prove the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (edge_to_edge : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : edge_to_edge = 83)
  (h3 : num_overlaps = 6)
  (h4 : total_length - edge_to_edge = num_overlaps * (total_length - edge_to_edge) / num_overlaps) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l3498_349807


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l3498_349871

theorem cube_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 3) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l3498_349871


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3498_349857

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4, 5}

theorem complement_intersection_equals_set : 
  (Aᶜ : Set ℕ) ∩ B = {1, 4, 5, 6, 7, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3498_349857


namespace NUMINAMATH_CALUDE_star_theorem_l3498_349853

/-- The star operation for real numbers -/
def star (a b : ℝ) : ℝ := (a - b) ^ 3

/-- Theorem: For real numbers x and y, (x-y)^3 ⋆ (y-x)^3 = 8(x-y)^9 -/
theorem star_theorem (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := by
  sorry

end NUMINAMATH_CALUDE_star_theorem_l3498_349853


namespace NUMINAMATH_CALUDE_vector_c_value_l3498_349831

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ,
  (4 • a) + (3 • b - 2 • a) + c = (0, 0) →
  c = (4, -6) :=
by sorry

end NUMINAMATH_CALUDE_vector_c_value_l3498_349831


namespace NUMINAMATH_CALUDE_only_pairC_not_opposite_l3498_349864

-- Define a type for quantities
inductive Quantity
| WinGames (n : ℕ)
| LoseGames (n : ℕ)
| RotateCounterclockwise (n : ℕ)
| RotateClockwise (n : ℕ)
| ReceiveMoney (amount : ℕ)
| IncreaseMoney (amount : ℕ)
| TemperatureRise (degrees : ℕ)
| TemperatureDecrease (degrees : ℕ)

-- Define a function to check if two quantities have opposite meanings
def haveOppositeMeanings (q1 q2 : Quantity) : Prop :=
  match q1, q2 with
  | Quantity.WinGames n, Quantity.LoseGames m => true
  | Quantity.RotateCounterclockwise n, Quantity.RotateClockwise m => true
  | Quantity.ReceiveMoney n, Quantity.IncreaseMoney m => false
  | Quantity.TemperatureRise n, Quantity.TemperatureDecrease m => true
  | _, _ => false

-- Define the pairs of quantities
def pairA := (Quantity.WinGames 3, Quantity.LoseGames 3)
def pairB := (Quantity.RotateCounterclockwise 3, Quantity.RotateClockwise 5)
def pairC := (Quantity.ReceiveMoney 3000, Quantity.IncreaseMoney 3000)
def pairD := (Quantity.TemperatureRise 4, Quantity.TemperatureDecrease 10)

-- Theorem statement
theorem only_pairC_not_opposite : 
  (haveOppositeMeanings pairA.1 pairA.2) ∧
  (haveOppositeMeanings pairB.1 pairB.2) ∧
  ¬(haveOppositeMeanings pairC.1 pairC.2) ∧
  (haveOppositeMeanings pairD.1 pairD.2) :=
by sorry

end NUMINAMATH_CALUDE_only_pairC_not_opposite_l3498_349864


namespace NUMINAMATH_CALUDE_janet_roller_coaster_rides_l3498_349840

/-- The number of tickets required for one roller coaster ride -/
def roller_coaster_tickets : ℕ := 5

/-- The number of tickets required for one giant slide ride -/
def giant_slide_tickets : ℕ := 3

/-- The number of times Janet wants to ride the giant slide -/
def giant_slide_rides : ℕ := 4

/-- The total number of tickets Janet needs -/
def total_tickets : ℕ := 47

/-- The number of times Janet wants to ride the roller coaster -/
def roller_coaster_rides : ℕ := 7

theorem janet_roller_coaster_rides : 
  roller_coaster_tickets * roller_coaster_rides + 
  giant_slide_tickets * giant_slide_rides = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_janet_roller_coaster_rides_l3498_349840


namespace NUMINAMATH_CALUDE_sticker_distribution_l3498_349852

theorem sticker_distribution (n m : ℕ) (hn : n = 5) (hm : m = 5) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = 126 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3498_349852


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l3498_349888

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std_dev : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std_dev = 84) : 
  ∃ (p : ℝ), p = 68 ∧ 
  p = dist.percent_less_than_mean_plus_std_dev - (100 - dist.percent_less_than_mean_plus_std_dev) := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l3498_349888


namespace NUMINAMATH_CALUDE_tangent_slope_minimum_value_l3498_349859

theorem tangent_slope_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (8 * a + b) / (a * b) ≥ 9 ∧
  ((8 * a + b) / (a * b) = 9 ↔ a = 1/3 ∧ b = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_minimum_value_l3498_349859


namespace NUMINAMATH_CALUDE_jerry_average_additional_hours_l3498_349838

def tom_total_hours : ℝ := 10
def jerry_daily_differences : List ℝ := [-2, 1, -2, 2, 2, 1]

theorem jerry_average_additional_hours :
  let jerry_total_hours := tom_total_hours + jerry_daily_differences.sum
  let total_difference := jerry_total_hours - tom_total_hours
  let num_days := jerry_daily_differences.length
  total_difference / num_days = 1/3 := by sorry

end NUMINAMATH_CALUDE_jerry_average_additional_hours_l3498_349838


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3498_349894

theorem diophantine_equation_solutions :
  ∀ m n k : ℕ, 2 * m + 3 * n = k^2 ↔
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3498_349894


namespace NUMINAMATH_CALUDE_hana_stamp_collection_l3498_349841

/-- Represents the fraction of Hana's stamp collection that was sold -/
def fraction_sold : ℚ := 28 / 49

/-- The amount Hana received for the part of the collection she sold -/
def amount_received : ℕ := 28

/-- The total value of Hana's entire stamp collection -/
def total_value : ℕ := 49

theorem hana_stamp_collection :
  fraction_sold = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_hana_stamp_collection_l3498_349841


namespace NUMINAMATH_CALUDE_triangle_side_length_l3498_349810

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  b = Real.sqrt 7 → 
  a = 3 → 
  Real.tan C = Real.sqrt 3 / 2 → 
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) → 
  c = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3498_349810


namespace NUMINAMATH_CALUDE_not_necessarily_monotonic_increasing_l3498_349854

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def strictly_increasing_by_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x < f (x + 1)

-- Define monotonic increasing
def monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≤ f y

-- Theorem statement
theorem not_necessarily_monotonic_increasing 
  (h : strictly_increasing_by_one f) : 
  ¬ (monotonic_increasing f) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_monotonic_increasing_l3498_349854


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3498_349826

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), a < 65 ∧ b < 65 ∧ (4 * a) % 65 = 1 ∧ (13 * b) % 65 = 1 ∧
  (3 * a + 7 * b) % 65 = 47 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3498_349826


namespace NUMINAMATH_CALUDE_cosine_of_angle_through_point_l3498_349845

/-- If the terminal side of angle α passes through point P (-1, -√2), then cos α = -√3/3 -/
theorem cosine_of_angle_through_point :
  ∀ α : Real,
  let P : Real × Real := (-1, -Real.sqrt 2)
  (∃ t : Real, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_of_angle_through_point_l3498_349845


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3498_349891

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 
  1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3498_349891


namespace NUMINAMATH_CALUDE_x_value_proof_l3498_349812

theorem x_value_proof (x y z a b c : ℝ) 
  (ha : xy / (x - y) = a)
  (hb : xz / (x - z) = b)
  (hc : yz / (y - z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  x = 2 * a * b * c / (a * b + b * c + a * c) :=
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3498_349812


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l3498_349818

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) := x^2 - 2*x

-- State the theorem
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l3498_349818


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3498_349843

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, prove that if a₃ + a₅ = 12 - a₇, then a₁ + a₉ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 3 + a 5 = 12 - a 7) : a 1 + a 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3498_349843


namespace NUMINAMATH_CALUDE_remainder_sum_l3498_349846

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 9) = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l3498_349846


namespace NUMINAMATH_CALUDE_compute_expression_l3498_349858

theorem compute_expression : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 := by
sorry

end NUMINAMATH_CALUDE_compute_expression_l3498_349858


namespace NUMINAMATH_CALUDE_max_value_relationship_l3498_349802

theorem max_value_relationship (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (a + b)^2 ≤ 2005 - (x + y)^2) → x = -y := by
  sorry

end NUMINAMATH_CALUDE_max_value_relationship_l3498_349802


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l3498_349820

theorem geometric_series_second_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 25) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : 
  second_term = -100/9 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l3498_349820


namespace NUMINAMATH_CALUDE_min_calls_for_gossip_l3498_349808

theorem min_calls_for_gossip (n : ℕ) (h : n > 0) : ℕ :=
  2 * (n - 1)

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_min_calls_for_gossip_l3498_349808


namespace NUMINAMATH_CALUDE_locus_is_conic_locus_degeneration_l3498_349809

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  sideLength : ℝ
  vertex1 : Point
  vertex2 : Point

/-- The locus equation of point P on square S -/
def locusEquation (a b c x y : ℝ) : Prop :=
  (b^2 + c^2) * x^2 - 4 * a * c * x * y + (4 * a^2 + b^2 + c^2 - 4 * a * b) * y^2 = (b^2 + c^2 - 2 * a * b)^2

/-- The condition for locus degeneration -/
def degenerationCondition (a b c : ℝ) : Prop :=
  (a - b)^2 + c^2 = a^2

/-- Theorem stating that the locus of point P on square S is part of a conic -/
theorem locus_is_conic (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  locusEquation a b c P.x P.y :=
by sorry

/-- Theorem stating the condition for locus degeneration -/
theorem locus_degeneration (S : Square) (P : Point) :
  S.sideLength = 2 * a →
  S.vertex1.x ≥ 0 →
  S.vertex1.y = 0 →
  S.vertex2.x = 0 →
  S.vertex2.y ≥ 0 →
  P.x = b →
  P.y = c →
  degenerationCondition a b c →
  ∃ (m k : ℝ), P.y = m * P.x + k :=
by sorry

end NUMINAMATH_CALUDE_locus_is_conic_locus_degeneration_l3498_349809


namespace NUMINAMATH_CALUDE_rotate_vector_90_degrees_l3498_349875

/-- Given points O and P in a 2D Cartesian coordinate system, and Q obtained by rotating OP counterclockwise by π/2, prove that Q has coordinates (-2, 1) -/
theorem rotate_vector_90_degrees (O P Q : ℝ × ℝ) : 
  O = (0, 0) → 
  P = (1, 2) → 
  (Q.1 - O.1, Q.2 - O.2) = (-(P.2 - O.2), P.1 - O.1) → 
  Q = (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_rotate_vector_90_degrees_l3498_349875


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3498_349860

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose n k

/-- The number of triangles with exactly one side coinciding with a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides coinciding with sides of the decagon 
    (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side 
    coinciding with a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of a randomly chosen triangle having at least one side 
    that is also a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3498_349860


namespace NUMINAMATH_CALUDE_school_trip_distances_l3498_349814

/-- Represents the problem of finding the distances in the school trip scenario. -/
theorem school_trip_distances 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (bus_speed : ℝ) 
  (rest_time : ℝ) : 
  total_distance = 21 ∧ 
  walking_speed = 4 ∧ 
  bus_speed = 60 ∧ 
  rest_time = 1/6 →
  ∃ (distance_to_A : ℝ) (distance_walked : ℝ),
    distance_to_A = 19 ∧
    distance_walked = 2 ∧
    distance_to_A + distance_walked = total_distance ∧
    distance_to_A / bus_speed + total_distance / bus_speed = 
      rest_time + distance_walked / walking_speed :=
by sorry

end NUMINAMATH_CALUDE_school_trip_distances_l3498_349814


namespace NUMINAMATH_CALUDE_total_potatoes_l3498_349821

theorem total_potatoes (nancy_potatoes sandy_potatoes : ℕ) 
  (h1 : nancy_potatoes = 6) 
  (h2 : sandy_potatoes = 7) : 
  nancy_potatoes + sandy_potatoes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_potatoes_l3498_349821


namespace NUMINAMATH_CALUDE_production_cost_correct_l3498_349878

/-- The production cost per performance for Steve's circus investment -/
def production_cost_per_performance : ℝ := 7000

/-- The overhead cost for the circus production -/
def overhead_cost : ℝ := 81000

/-- The income from a single sold-out performance -/
def sold_out_income : ℝ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the production cost per performance is correct -/
theorem production_cost_correct :
  production_cost_per_performance * break_even_performances + overhead_cost =
  sold_out_income * break_even_performances :=
by sorry

end NUMINAMATH_CALUDE_production_cost_correct_l3498_349878


namespace NUMINAMATH_CALUDE_ratio_of_vectors_l3498_349872

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that if OC = 2/3 * OA + 1/3 * OB, then |AC| / |AB| = 1/3 -/
theorem ratio_of_vectors (O A B C : ℝ × ℝ × ℝ) (h : C = (2/3 : ℝ) • A + (1/3 : ℝ) • B) :
  ‖C - A‖ / ‖B - A‖ = 1/3 := by sorry

end NUMINAMATH_CALUDE_ratio_of_vectors_l3498_349872


namespace NUMINAMATH_CALUDE_cassini_identity_l3498_349830

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Cassini's identity for Fibonacci numbers -/
theorem cassini_identity (n : ℕ) (h : n > 0) : 
  (fib (n + 1) * fib (n - 1) - fib n ^ 2 : ℤ) = (-1) ^ n := by
  sorry

end NUMINAMATH_CALUDE_cassini_identity_l3498_349830


namespace NUMINAMATH_CALUDE_hundredth_figure_squares_l3498_349848

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem hundredth_figure_squares :
  f 100 = 30301 := by sorry

end NUMINAMATH_CALUDE_hundredth_figure_squares_l3498_349848


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l3498_349827

/-- The number of siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 19

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 45

/-- The number of cats sold during the sale -/
def cats_sold : ℕ := 56

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

theorem pet_store_siamese_cats :
  initial_siamese_cats = 19 :=
by
  have h1 : initial_siamese_cats + initial_house_cats = initial_siamese_cats + 45 := by rfl
  have h2 : initial_siamese_cats + initial_house_cats - cats_sold = cats_remaining :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l3498_349827


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3498_349833

theorem rectangle_area (square_area : Real) (rectangle_length_factor : Real) : Real :=
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_factor * rectangle_width
  rectangle_width * rectangle_length

theorem rectangle_area_proof :
  rectangle_area 36 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l3498_349833


namespace NUMINAMATH_CALUDE_physics_group_size_l3498_349823

theorem physics_group_size (total : ℕ) (math_ratio physics_ratio chem_ratio : ℕ) : 
  total = 135 → 
  math_ratio = 6 →
  physics_ratio = 5 →
  chem_ratio = 4 →
  (physics_ratio : ℚ) / (math_ratio + physics_ratio + chem_ratio : ℚ) * total = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_physics_group_size_l3498_349823


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3498_349817

theorem quadratic_equation_m_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 4*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + 3*x₂ = 5 →
  m = 7/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3498_349817


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l3498_349813

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  z₁ = 2 + I → 
  z₂.re = -z₁.re ∧ z₂.im = z₁.im → 
  z₁ * z₂ = -5 := by sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l3498_349813


namespace NUMINAMATH_CALUDE_second_frog_hops_l3498_349880

theorem second_frog_hops :
  ∀ (h1 h2 h3 : ℕ),
  h1 = 4 * h2 →
  h2 = 2 * h3 →
  h1 + h2 + h3 = 99 →
  h2 = 18 := by
sorry

end NUMINAMATH_CALUDE_second_frog_hops_l3498_349880


namespace NUMINAMATH_CALUDE_box_width_is_48_l3498_349816

/-- Represents the dimensions of a box and the number of cubes that fill it -/
structure BoxWithCubes where
  length : ℕ
  width : ℕ
  depth : ℕ
  num_cubes : ℕ

/-- The box is completely filled by the cubes -/
def is_filled (box : BoxWithCubes) : Prop :=
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    box.length % cube_side = 0 ∧
    box.width % cube_side = 0 ∧
    box.depth % cube_side = 0 ∧
    box.length * box.width * box.depth = box.num_cubes * (cube_side ^ 3)

/-- The main theorem: if a box with given dimensions is filled with 80 cubes, its width is 48 inches -/
theorem box_width_is_48 (box : BoxWithCubes) : 
  box.length = 30 → box.depth = 12 → box.num_cubes = 80 → is_filled box → box.width = 48 := by
  sorry

end NUMINAMATH_CALUDE_box_width_is_48_l3498_349816


namespace NUMINAMATH_CALUDE_cubic_minus_x_factorization_l3498_349877

theorem cubic_minus_x_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_x_factorization_l3498_349877


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l3498_349865

theorem normal_distribution_std_dev (μ σ : ℝ) : 
  μ = 55 → μ - 3 * σ > 48 → σ < 7/3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l3498_349865


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l3498_349815

/-- A collection of stamps with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- The number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.old + sc.foreignAndOld - sc.neitherForeignNorOld

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection)
    (h1 : sc.total = 200)
    (h2 : sc.old = 50)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 90) :
    foreignStamps sc = 80 := by
  sorry

end NUMINAMATH_CALUDE_foreign_stamps_count_l3498_349815


namespace NUMINAMATH_CALUDE_square_minus_product_l3498_349851

theorem square_minus_product (a : ℝ) : (a - 1)^2 - a*(a - 1) = -a + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_l3498_349851


namespace NUMINAMATH_CALUDE_fraction_problem_l3498_349825

theorem fraction_problem : 
  let x : ℚ := 2/3
  (3/4 : ℚ) * (4/5 : ℚ) * x = (2/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3498_349825


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2015_l3498_349887

theorem last_four_digits_of_5_pow_2015 : ∃ n : ℕ, 5^2015 ≡ 8125 [MOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2015_l3498_349887


namespace NUMINAMATH_CALUDE_imaginary_unit_cubed_l3498_349896

theorem imaginary_unit_cubed (i : ℂ) (h : i^2 = -1) : i^3 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cubed_l3498_349896


namespace NUMINAMATH_CALUDE_symmetric_derivative_implies_cosine_possible_l3498_349876

/-- A function whose derivative's graph is symmetric about the origin -/
class SymmetricDerivative (f : ℝ → ℝ) : Prop :=
  (symmetric : ∀ x : ℝ, (deriv f) x = -(deriv f) (-x))

/-- The theorem stating that if f'(x) is symmetric about the origin, 
    then f(x) = 3cos(x) is a possible expression for f(x) -/
theorem symmetric_derivative_implies_cosine_possible 
  (f : ℝ → ℝ) [SymmetricDerivative f] : 
  ∃ g : ℝ → ℝ, (∀ x, f x = 3 * Real.cos x) ∧ SymmetricDerivative g :=
sorry

end NUMINAMATH_CALUDE_symmetric_derivative_implies_cosine_possible_l3498_349876


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l3498_349899

theorem cosine_sine_identity : 
  Real.cos (35 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (145 * π / 180) * Real.cos (65 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l3498_349899


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3498_349804

/-- A rectangular prism with given surface area and total edge length has a specific interior diagonal length -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + a * c + b * c) = 54)
  (h_edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3498_349804


namespace NUMINAMATH_CALUDE_gecko_eggs_calcification_fraction_l3498_349873

def total_eggs : ℕ := 30
def infertile_percentage : ℚ := 1/5
def hatched_eggs : ℕ := 16

theorem gecko_eggs_calcification_fraction :
  (total_eggs * (1 - infertile_percentage) - hatched_eggs) / (total_eggs * (1 - infertile_percentage)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_gecko_eggs_calcification_fraction_l3498_349873


namespace NUMINAMATH_CALUDE_drug_price_reduction_l3498_349811

theorem drug_price_reduction (x : ℝ) : 
  (36 : ℝ) * (1 - x)^2 = 25 ↔ 
  (∃ (price1 price2 : ℝ), 
    36 * (1 - x) = price1 ∧ 
    price1 * (1 - x) = price2 ∧ 
    price2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l3498_349811


namespace NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l3498_349822

theorem power_of_seven_mod_hundred : 7^700 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_hundred_l3498_349822


namespace NUMINAMATH_CALUDE_simplify_fraction_l3498_349867

theorem simplify_fraction (b : ℚ) (h : b = 4) : 18 * b^4 / (27 * b^3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3498_349867


namespace NUMINAMATH_CALUDE_scientists_from_usa_l3498_349886

theorem scientists_from_usa (total : ℕ) (europe : ℕ) (canada : ℕ) (usa : ℕ)
  (h1 : total = 70)
  (h2 : europe = total / 2)
  (h3 : canada = total / 5)
  (h4 : usa = total - (europe + canada)) :
  usa = 21 := by
  sorry

end NUMINAMATH_CALUDE_scientists_from_usa_l3498_349886


namespace NUMINAMATH_CALUDE_range_of_m_l3498_349832

/-- An odd function f: ℝ → ℝ with domain [-2,2] -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 2)

/-- f is monotonically decreasing on [0,2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f y < f x

/-- The main theorem -/
theorem range_of_m (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : MonoDecreasing f) :
  {m : ℝ | f (1 + m) + f m < 0} = Set.Ioo (-1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3498_349832


namespace NUMINAMATH_CALUDE_hose_fill_time_proof_l3498_349819

/-- Represents the time (in hours) it takes for the hose to fill the pool -/
def hose_fill_time (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ) : ℝ :=
  3

/-- Proves that the hose fill time is correct given the problem conditions -/
theorem hose_fill_time_proof (pool_capacity : ℝ) (drain_time : ℝ) (time_elapsed : ℝ) (remaining_water : ℝ)
  (h1 : pool_capacity = 120)
  (h2 : drain_time = 4)
  (h3 : time_elapsed = 3)
  (h4 : remaining_water = 90) :
  hose_fill_time pool_capacity drain_time time_elapsed remaining_water = 3 := by
  sorry

#eval hose_fill_time 120 4 3 90

end NUMINAMATH_CALUDE_hose_fill_time_proof_l3498_349819


namespace NUMINAMATH_CALUDE_product_of_largest_and_smallest_l3498_349849

/-- The set of digits allowed to form the numbers -/
def allowed_digits : Finset Nat := {0, 2, 4, 6}

/-- Predicate to check if a number is a valid three-digit number using allowed digits -/
def is_valid_number (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c, n = 100 * a + 10 * b + c ∧ 
            a ∈ allowed_digits ∧ b ∈ allowed_digits ∧ c ∈ allowed_digits ∧
            a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- The largest valid number -/
def largest_number : Nat := 642

/-- The smallest valid number -/
def smallest_number : Nat := 204

theorem product_of_largest_and_smallest :
  is_valid_number largest_number ∧
  is_valid_number smallest_number ∧
  (∀ n : Nat, is_valid_number n → n ≤ largest_number) ∧
  (∀ n : Nat, is_valid_number n → n ≥ smallest_number) ∧
  largest_number * smallest_number = 130968 := by
  sorry

#check product_of_largest_and_smallest

end NUMINAMATH_CALUDE_product_of_largest_and_smallest_l3498_349849


namespace NUMINAMATH_CALUDE_triangle_calculation_l3498_349890

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a * b + 2 * a

-- State the theorem
theorem triangle_calculation : triangle (-3) (triangle (-4) (1/2)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_calculation_l3498_349890


namespace NUMINAMATH_CALUDE_total_interest_is_350_l3498_349889

/-- Calculates the total interest for two loans over a given time period. -/
def totalInterest (principal1 : ℝ) (rate1 : ℝ) (principal2 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal1 * rate1 * time + principal2 * rate2 * time

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 800 0.03 1400 0.05 3.723404255319149 = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_is_350_l3498_349889


namespace NUMINAMATH_CALUDE_daily_toy_production_l3498_349800

/-- Given a factory that produces toys, this theorem proves the daily production
    when the weekly production and number of working days are known. -/
theorem daily_toy_production
  (weekly_production : ℕ)
  (working_days : ℕ)
  (h_weekly : weekly_production = 6500)
  (h_days : working_days = 5)
  (h_equal_daily : weekly_production % working_days = 0) :
  weekly_production / working_days = 1300 := by
  sorry

#check daily_toy_production

end NUMINAMATH_CALUDE_daily_toy_production_l3498_349800


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3498_349863

theorem simplify_sqrt_expression : 
  (Real.sqrt 192 / Real.sqrt 27) - (Real.sqrt 500 / Real.sqrt 125) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3498_349863


namespace NUMINAMATH_CALUDE_perpendicular_length_between_l3498_349837

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relations and functions
variable (on_line : Point → Line → Prop)
variable (between : Point → Point → Point → Prop)
variable (perpendicular : Point → Point → Line → Prop)
variable (length : Point → Point → ℝ)

-- State the theorem
theorem perpendicular_length_between
  (a b : Line)
  (A₁ A₂ A₃ B₁ B₂ B₃ : Point)
  (h1 : on_line A₁ a)
  (h2 : on_line A₂ a)
  (h3 : on_line A₃ a)
  (h4 : between A₁ A₂ A₃)
  (h5 : perpendicular A₁ B₁ b)
  (h6 : perpendicular A₂ B₂ b)
  (h7 : perpendicular A₃ B₃ b) :
  (length A₁ B₁ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₃ B₃) ∨
  (length A₃ B₃ ≤ length A₂ B₂ ∧ length A₂ B₂ ≤ length A₁ B₁) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_length_between_l3498_349837


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l3498_349892

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 50 → 
  (sarah_score + greg_score) / 2 = 110 → 
  sarah_score = 135 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l3498_349892


namespace NUMINAMATH_CALUDE_pentagon_rods_l3498_349839

theorem pentagon_rods (rods : Finset ℕ) : 
  rods = {4, 9, 18, 25} →
  (∀ e ∈ Finset.range 41 \ rods, 
    (e ≠ 0 ∧ 
     e < 4 + 9 + 18 + 25 ∧
     4 + 9 + 18 + e > 25 ∧
     4 + 9 + 25 + e > 18 ∧
     4 + 18 + 25 + e > 9 ∧
     9 + 18 + 25 + e > 4)) →
  (Finset.range 41 \ rods).card = 51 :=
sorry

end NUMINAMATH_CALUDE_pentagon_rods_l3498_349839


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3498_349893

theorem smallest_x_absolute_value_equation :
  (∀ x : ℝ, |3 * x + 7| = 26 → x ≥ -11) ∧
  |3 * (-11) + 7| = 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3498_349893


namespace NUMINAMATH_CALUDE_biff_voting_percentage_l3498_349868

theorem biff_voting_percentage (total_polled : ℕ) (marty_votes : ℕ) (undecided_percent : ℚ) :
  total_polled = 200 →
  marty_votes = 94 →
  undecided_percent = 8 / 100 →
  (↑(total_polled - marty_votes - (undecided_percent * ↑total_polled).num) / ↑total_polled : ℚ) = 45 / 100 := by
  sorry

end NUMINAMATH_CALUDE_biff_voting_percentage_l3498_349868


namespace NUMINAMATH_CALUDE_similar_cuts_possible_equilateral_cuts_impossible_l3498_349835

-- Define a triangular prism
structure TriangularPrism :=
  (base : Set (ℝ × ℝ × ℝ))
  (height : ℝ)

-- Define a cut on the prism
structure Cut :=
  (shape : Set (ℝ × ℝ × ℝ))
  (is_triangular : Bool)

-- Define similarity between two cuts
def are_similar (c1 c2 : Cut) : Prop := sorry

-- Define equality between two cuts
def are_equal (c1 c2 : Cut) : Prop := sorry

-- Define if a cut touches the base
def touches_base (c : Cut) (p : TriangularPrism) : Prop := sorry

-- Define if two cuts touch each other
def cuts_touch (c1 c2 : Cut) : Prop := sorry

-- Theorem for part (a)
theorem similar_cuts_possible (p : TriangularPrism) :
  ∃ (c1 c2 : Cut),
    c1.is_triangular ∧
    c2.is_triangular ∧
    are_similar c1 c2 ∧
    ¬are_equal c1 c2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

-- Define an equilateral triangular cut
def is_equilateral_triangle (c : Cut) (side_length : ℝ) : Prop := sorry

-- Theorem for part (b)
theorem equilateral_cuts_impossible (p : TriangularPrism) :
  ¬∃ (c1 c2 : Cut),
    is_equilateral_triangle c1 1 ∧
    is_equilateral_triangle c2 2 ∧
    ¬touches_base c1 p ∧
    ¬touches_base c2 p ∧
    ¬cuts_touch c1 c2 := by sorry

end NUMINAMATH_CALUDE_similar_cuts_possible_equilateral_cuts_impossible_l3498_349835


namespace NUMINAMATH_CALUDE_circular_sum_equivalence_l3498_349806

/-- 
Given integers n > m > 1 arranged in a circle, s_i is the sum of m integers 
starting at the i-th position moving clockwise, and t_i is the sum of the 
remaining n-m integers. f(a, b) is the number of elements i in {1, 2, ..., n} 
such that s_i ≡ a (mod 4) and t_i ≡ b (mod 4).
-/
def f (n m : ℕ) (a b : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem circular_sum_equivalence (n m : ℕ) (h1 : n > m) (h2 : m > 1) :
  f n m 1 3 ≡ f n m 3 1 [MOD 4] ↔ Even (f n m 2 2) := by sorry

end NUMINAMATH_CALUDE_circular_sum_equivalence_l3498_349806


namespace NUMINAMATH_CALUDE_paint_contribution_is_360_l3498_349895

/-- Calculates the contribution of each person for the paint cost --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ)
  (jason_wall_area : ℚ) (jason_coats : ℕ) (jeremy_wall_area : ℚ) (jeremy_coats : ℕ) : ℚ :=
  let total_area := jason_wall_area * jason_coats + jeremy_wall_area * jeremy_coats
  let gallons_needed := (total_area / coverage_per_gallon).ceil
  let total_cost := gallons_needed * paint_cost_per_gallon
  total_cost / 2

/-- Theorem stating that each person's contribution is $360 --/
theorem paint_contribution_is_360 :
  calculate_contribution 45 400 1025 3 1575 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_contribution_is_360_l3498_349895


namespace NUMINAMATH_CALUDE_club_membership_increase_l3498_349881

theorem club_membership_increase (current_members : ℕ) (h : current_members = 10) : 
  (2 * current_members + 5) - current_members = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_increase_l3498_349881
