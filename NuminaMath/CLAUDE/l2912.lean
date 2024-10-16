import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2912_291239

theorem inequality_proof (x : ℝ) : x > -4/3 → 3 - 1/(3*x + 4) < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2912_291239


namespace NUMINAMATH_CALUDE_smaller_solid_volume_is_one_sixth_l2912_291232

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edgeLength : ℝ
  vertex : Point3D

/-- Calculates the volume of the smaller solid created by a plane cutting a cube -/
def smallerSolidVolume (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The volume of the smaller solid in a cube with edge length 2,
    cut by a plane passing through vertex D and midpoints of AB and CG, is 1/6 -/
theorem smaller_solid_volume_is_one_sixth :
  let cube := Cube.mk 2 (Point3D.mk 0 0 0)
  let plane := Plane.mk 2 (-4) (-8) 0
  smallerSolidVolume cube plane = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solid_volume_is_one_sixth_l2912_291232


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l2912_291228

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, (m - 3) * x^2 - 3 * x + m^2 = 9) → m = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l2912_291228


namespace NUMINAMATH_CALUDE_remaining_backpack_price_l2912_291272

-- Define the problem parameters
def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_sold : ℕ := 10
def dept_store_price : ℕ := 25
def total_profit : ℕ := 442

-- Define the theorem
theorem remaining_backpack_price :
  let remaining_backpacks := total_backpacks - (swap_meet_sold + dept_store_sold)
  let swap_meet_revenue := swap_meet_sold * swap_meet_price
  let dept_store_revenue := dept_store_sold * dept_store_price
  let total_revenue := total_cost + total_profit
  let remaining_revenue := total_revenue - (swap_meet_revenue + dept_store_revenue)
  remaining_revenue / remaining_backpacks = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_backpack_price_l2912_291272


namespace NUMINAMATH_CALUDE_consecutive_numbers_percentage_l2912_291238

theorem consecutive_numbers_percentage (a b c d e f g : ℤ) : 
  (a + b + c + d + e + f + g = 7 * 9) →
  (b = a + 1) →
  (c = b + 1) →
  (d = c + 1) →
  (e = d + 1) →
  (f = e + 1) →
  (g = f + 1) →
  (a : ℚ) / g * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_percentage_l2912_291238


namespace NUMINAMATH_CALUDE_number_of_women_at_tables_l2912_291277

/-- Proves that the number of women at the tables is 7.0 -/
theorem number_of_women_at_tables 
  (num_tables : Float) 
  (num_men : Float) 
  (avg_customers_per_table : Float) 
  (h1 : num_tables = 9.0)
  (h2 : num_men = 3.0)
  (h3 : avg_customers_per_table = 1.111111111) : 
  Float.round ((num_tables * avg_customers_per_table) - num_men) = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_women_at_tables_l2912_291277


namespace NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2912_291248

theorem negative_abs_of_negative_one : -|-1| = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_of_negative_one_l2912_291248


namespace NUMINAMATH_CALUDE_newspaper_distribution_l2912_291217

theorem newspaper_distribution (F : ℚ) : 
  200 * F + 0.6 * (200 - 200 * F) = 200 - 48 → F = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_distribution_l2912_291217


namespace NUMINAMATH_CALUDE_polyline_distance_bound_l2912_291214

/-- Polyline distance between two points -/
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + |y₁ - y₂|

/-- Theorem: For any point C(x, y) with polyline distance 1 from O(0, 0), √(x² + y²) ≥ √2/2 -/
theorem polyline_distance_bound (x y : ℝ) 
  (h : polyline_distance 0 0 x y = 1) : 
  Real.sqrt (x^2 + y^2) ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polyline_distance_bound_l2912_291214


namespace NUMINAMATH_CALUDE_min_children_for_all_colors_l2912_291270

/-- Represents the distribution of pencils among children -/
structure PencilDistribution where
  total_pencils : ℕ
  num_colors : ℕ
  pencils_per_color : ℕ
  num_children : ℕ
  pencils_per_child : ℕ

/-- Theorem stating the minimum number of children to select to guarantee all colors -/
theorem min_children_for_all_colors (d : PencilDistribution) 
  (h1 : d.total_pencils = 24)
  (h2 : d.num_colors = 4)
  (h3 : d.pencils_per_color = 6)
  (h4 : d.num_children = 6)
  (h5 : d.pencils_per_child = 4)
  (h6 : d.total_pencils = d.num_colors * d.pencils_per_color)
  (h7 : d.total_pencils = d.num_children * d.pencils_per_child) :
  ∃ (n : ℕ), n = 5 ∧ 
  (∀ (m : ℕ), m < n → ¬(∀ (selection : Finset (Fin d.num_children)), 
    selection.card = m → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val))) ∧
  (∀ (selection : Finset (Fin d.num_children)), 
    selection.card = n → 
    (∃ (colors : Finset (Fin d.num_colors)), colors.card = d.num_colors ∧
      ∀ (c : Fin d.num_colors), c ∈ colors → 
        ∃ (child : Fin d.num_children), child ∈ selection ∧ 
          ∃ (pencil : Fin d.pencils_per_child), pencil.val < d.pencils_per_child ∧
            (child.val * d.pencils_per_child + pencil.val) % d.num_colors = c.val)) :=
by sorry

end NUMINAMATH_CALUDE_min_children_for_all_colors_l2912_291270


namespace NUMINAMATH_CALUDE_min_value_problem_l2912_291213

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 4) = 1 / 2) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 3) + 1 / (b + 4) = 1 / 2 →
  2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 1 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2912_291213


namespace NUMINAMATH_CALUDE_correct_second_number_l2912_291284

/-- Proves that the correct value of the second wrongly copied number is 27 --/
theorem correct_second_number (n : ℕ) (original_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (h1 : n = 10) (h2 : original_avg = 40.2) 
  (h3 : correct_avg = 40) (h4 : first_error = 16) (h5 : second_error = 13) : 
  ∃ (x : ℚ), n * correct_avg = n * original_avg - first_error - second_error + x ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_correct_second_number_l2912_291284


namespace NUMINAMATH_CALUDE_tire_price_proof_l2912_291289

/-- The regular price of a tire -/
def regular_price : ℝ := 126

/-- The promotional price for three tires -/
def promotional_price : ℝ := 315

/-- The promotion discount on the third tire -/
def discount : ℝ := 0.5

theorem tire_price_proof :
  2 * regular_price + discount * regular_price = promotional_price :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l2912_291289


namespace NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l2912_291242

theorem complex_sum_and_reciprocal (z : ℂ) : z = 1 + I → z + 2 / z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_and_reciprocal_l2912_291242


namespace NUMINAMATH_CALUDE_set_B_elements_l2912_291224

def A : Set Int := {-2, 0, 1, 3}

def B : Set Int := {x | -x ∈ A ∧ (1 - x) ∉ A}

theorem set_B_elements : B = {-3, -1, 2} := by sorry

end NUMINAMATH_CALUDE_set_B_elements_l2912_291224


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prisms_l2912_291220

-- Define the dimensions of a single prism
def prism_length : ℝ := 10
def prism_width : ℝ := 3
def prism_height : ℝ := 30

-- Define the number of prisms
def num_prisms : ℕ := 2

-- Theorem statement
theorem cube_surface_area_from_prisms :
  let prism_volume := prism_length * prism_width * prism_height
  let total_volume := num_prisms * prism_volume
  let cube_edge := total_volume ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 600 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prisms_l2912_291220


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_increase_percentage_l2912_291227

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_increase_percentage (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  ((1.3 * l) * (1.2 * w) - l * w) / (l * w) = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_increase_percentage_l2912_291227


namespace NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l2912_291205

theorem spring_festival_gala_arrangements :
  let original_programs : ℕ := 10
  let new_programs : ℕ := 3
  let available_spaces : ℕ := original_programs + 1 - 2  -- excluding first and last positions
  
  (available_spaces.choose new_programs) * (new_programs.factorial) = 990 :=
by
  sorry

end NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l2912_291205


namespace NUMINAMATH_CALUDE_sin_theta_value_l2912_291200

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 4 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l2912_291200


namespace NUMINAMATH_CALUDE_probability_point_between_C_and_E_l2912_291215

/-- Given a line segment AB with points C, D, and E, where AB = 4AD = 8BC and E divides CD into two equal parts,
    the probability of a randomly selected point on AB falling between C and E is 5/16. -/
theorem probability_point_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  E - C = D - E →                  -- E divides CD into two equal parts
  (E - C) / (B - A) = 5 / 16 :=     -- Probability is 5/16
by sorry

end NUMINAMATH_CALUDE_probability_point_between_C_and_E_l2912_291215


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2912_291255

/-- The number M as defined in the problem -/
def M : ℕ := 25 * 48 * 49 * 81

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the ratio of sum of odd divisors to sum of even divisors of M is 1:30 -/
theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_M_l2912_291255


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2912_291268

theorem least_positive_integer_with_remainders : ∃! n : ℕ,
  (n > 0) ∧
  (n % 11 = 10) ∧
  (n % 12 = 11) ∧
  (n % 13 = 12) ∧
  (n % 14 = 13) ∧
  (n % 15 = 14) ∧
  (n % 16 = 15) ∧
  (∀ m : ℕ, m > 0 ∧ 
    (m % 11 = 10) ∧
    (m % 12 = 11) ∧
    (m % 13 = 12) ∧
    (m % 14 = 13) ∧
    (m % 15 = 14) ∧
    (m % 16 = 15) → m ≥ n) ∧
  n = 720719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2912_291268


namespace NUMINAMATH_CALUDE_angle_properties_l2912_291294

def angle_set (α : Real) : Set Real :=
  {x | ∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3}

theorem angle_properties (α : Real) 
  (h : ∃ x y : Real, x = 1 ∧ y = Real.sqrt 3 ∧ x * Real.cos α = x ∧ y * Real.sin α = y) :
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α) = (Real.sqrt 3 - 1) / 2) ∧
  (angle_set α = {α}) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l2912_291294


namespace NUMINAMATH_CALUDE_max_min_x2_minus_y2_l2912_291206

theorem max_min_x2_minus_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - x*y + y^2 = 1) :
  ∃ (min max : ℝ), (∀ z, z = x^2 - y^2 → min ≤ z ∧ z ≤ max) ∧
                   min = -2*Real.sqrt 3/3 ∧
                   max = 2*Real.sqrt 3/3 := by
  sorry

end NUMINAMATH_CALUDE_max_min_x2_minus_y2_l2912_291206


namespace NUMINAMATH_CALUDE_relationship_depends_on_b_relationship_only_b_l2912_291258

theorem relationship_depends_on_b (a b : ℝ) : 
  (a + b) - (a - b) = 2 * b :=
sorry

theorem relationship_only_b (a b : ℝ) : 
  (a + b > a - b ↔ b > 0) ∧
  (a + b < a - b ↔ b < 0) ∧
  (a + b = a - b ↔ b = 0) :=
sorry

end NUMINAMATH_CALUDE_relationship_depends_on_b_relationship_only_b_l2912_291258


namespace NUMINAMATH_CALUDE_ac_length_l2912_291286

/-- Given a quadrilateral ABCD with specified side lengths, prove the length of AC --/
theorem ac_length (AB DC AD : ℝ) (h1 : AB = 13) (h2 : DC = 15) (h3 : AD = 12) :
  ∃ (AC : ℝ), abs (AC - Real.sqrt (369 + 240 * Real.sqrt 2)) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l2912_291286


namespace NUMINAMATH_CALUDE_complex_number_location_l2912_291222

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = 1 + Complex.I) :
  is_in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l2912_291222


namespace NUMINAMATH_CALUDE_optimal_zongzi_purchase_l2912_291259

/-- Represents the unit price and quantity of zongzi --/
structure Zongzi where
  unit_price : ℝ
  quantity : ℕ

/-- Represents the shopping mall's zongzi purchase plan --/
structure ZongziPurchasePlan where
  zongzi_a : Zongzi
  zongzi_b : Zongzi

/-- Defines the conditions of the zongzi purchase problem --/
def zongzi_problem (plan : ZongziPurchasePlan) : Prop :=
  let a := plan.zongzi_a
  let b := plan.zongzi_b
  (3000 / a.unit_price - 3360 / b.unit_price = 40) ∧
  (b.unit_price = 1.2 * a.unit_price) ∧
  (a.quantity + b.quantity = 2200) ∧
  (a.unit_price * a.quantity ≤ b.unit_price * b.quantity)

/-- Theorem stating the optimal solution to the zongzi purchase problem --/
theorem optimal_zongzi_purchase :
  ∃ (plan : ZongziPurchasePlan),
    zongzi_problem plan ∧
    plan.zongzi_a.unit_price = 5 ∧
    plan.zongzi_b.unit_price = 6 ∧
    plan.zongzi_a.quantity = 1200 ∧
    plan.zongzi_b.quantity = 1000 ∧
    plan.zongzi_a.unit_price * plan.zongzi_a.quantity +
    plan.zongzi_b.unit_price * plan.zongzi_b.quantity = 12000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_zongzi_purchase_l2912_291259


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l2912_291210

theorem quadratic_roots_max_value (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ (n : ℕ), 1 ≤ n → n ≤ 2010 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^2 - t*a₁ + q = 0 →
  a₂^2 - t*a₂ + q = 0 →
  (∀ (x : ℝ), x^2 - t*x + q ≠ 0 ∨ x = a₁ ∨ x = a₂) →
  (1 / a₁^2011 + 1 / a₂^2011) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l2912_291210


namespace NUMINAMATH_CALUDE_largest_n_divisible_equality_l2912_291257

def divisibleCount (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def divisibleBy5or7 (n : ℕ) : ℕ :=
  divisibleCount n 5 + divisibleCount n 7 - divisibleCount n 35

theorem largest_n_divisible_equality : ∀ n : ℕ, n > 65 →
  (divisibleCount n 3 ≠ divisibleBy5or7 n) ∧
  (divisibleCount 65 3 = divisibleBy5or7 65) := by
  sorry

#eval divisibleCount 65 3  -- Expected: 21
#eval divisibleBy5or7 65   -- Expected: 21

end NUMINAMATH_CALUDE_largest_n_divisible_equality_l2912_291257


namespace NUMINAMATH_CALUDE_N_eq_P_l2912_291282

def N : Set ℚ := {x | ∃ n : ℤ, x = n / 2 - 1 / 3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p / 2 + 1 / 6}

theorem N_eq_P : N = P := by sorry

end NUMINAMATH_CALUDE_N_eq_P_l2912_291282


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l2912_291208

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) :
  c / d = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l2912_291208


namespace NUMINAMATH_CALUDE_small_boxes_in_big_box_l2912_291263

theorem small_boxes_in_big_box 
  (total_big_boxes : ℕ) 
  (candles_per_small_box : ℕ) 
  (total_candles : ℕ) 
  (h1 : total_big_boxes = 50)
  (h2 : candles_per_small_box = 40)
  (h3 : total_candles = 8000) :
  (total_candles / candles_per_small_box) / total_big_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_small_boxes_in_big_box_l2912_291263


namespace NUMINAMATH_CALUDE_sum_of_penultimate_terms_l2912_291297

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_penultimate_terms 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 3) 
  (h_last : ∃ n : ℕ, a n = 33 ∧ a (n - 1) + a (n - 2) = x + y) : 
  x + y = 51 := by
sorry

end NUMINAMATH_CALUDE_sum_of_penultimate_terms_l2912_291297


namespace NUMINAMATH_CALUDE_max_green_cards_achievable_green_cards_l2912_291267

/-- Represents the number of cards of each color in the box -/
structure CardCount where
  green : ℕ
  yellow : ℕ

/-- The probability of selecting three cards of the same color -/
def prob_same_color (cc : CardCount) : ℚ :=
  let total := cc.green + cc.yellow
  (cc.green.choose 3 + cc.yellow.choose 3) / total.choose 3

/-- The main theorem stating the maximum number of green cards possible -/
theorem max_green_cards (cc : CardCount) : 
  cc.green + cc.yellow ≤ 2209 →
  prob_same_color cc = 1/3 →
  cc.green ≤ 1092 := by
  sorry

/-- The theorem stating that 1092 green cards is achievable -/
theorem achievable_green_cards : 
  ∃ (cc : CardCount), cc.green + cc.yellow ≤ 2209 ∧ 
  prob_same_color cc = 1/3 ∧ 
  cc.green = 1092 := by
  sorry

end NUMINAMATH_CALUDE_max_green_cards_achievable_green_cards_l2912_291267


namespace NUMINAMATH_CALUDE_not_periodic_x_plus_cos_x_l2912_291251

theorem not_periodic_x_plus_cos_x : ¬∃ (T : ℝ), T ≠ 0 ∧ ∀ (x : ℝ), x + T + Real.cos (x + T) = x + Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_not_periodic_x_plus_cos_x_l2912_291251


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l2912_291276

theorem sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two :
  Real.sqrt ((-4)^2) + ((-8 : ℝ) ^ (1/3 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l2912_291276


namespace NUMINAMATH_CALUDE_complex_division_equivalence_l2912_291212

theorem complex_division_equivalence : Complex.I * (4 - 3 * Complex.I) = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equivalence_l2912_291212


namespace NUMINAMATH_CALUDE_min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed_l2912_291246

theorem min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed 
  (y : ℝ) (h : y > 0) : y^2 + 9*y + 81/y^3 ≥ 39 ∧ 
  (y^2 + 9*y + 81/y^3 = 39 ↔ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_y_squared_plus_nine_y_plus_eightyone_over_y_cubed_l2912_291246


namespace NUMINAMATH_CALUDE_serving_size_is_six_ounces_l2912_291223

-- Define the given constants
def concentrate_cans : ℕ := 12
def water_cans_per_concentrate : ℕ := 4
def ounces_per_can : ℕ := 12
def total_servings : ℕ := 120

-- Define the theorem
theorem serving_size_is_six_ounces :
  let total_cans := concentrate_cans * (water_cans_per_concentrate + 1)
  let total_ounces := total_cans * ounces_per_can
  let serving_size := total_ounces / total_servings
  serving_size = 6 := by sorry

end NUMINAMATH_CALUDE_serving_size_is_six_ounces_l2912_291223


namespace NUMINAMATH_CALUDE_laurence_to_missy_relation_keith_receives_32_messages_l2912_291249

/-- Messages sent from Juan to Laurence -/
def messages_juan_to_laurence : ℕ := sorry

/-- Messages sent from Juan to Keith -/
def messages_juan_to_keith : ℕ := 8 * messages_juan_to_laurence

/-- Messages sent from Laurence to Missy -/
def messages_laurence_to_missy : ℕ := 18

/-- Relation between messages from Laurence to Missy and from Juan to Laurence -/
theorem laurence_to_missy_relation : 
  messages_laurence_to_missy = (4.5 : ℚ) * messages_juan_to_laurence := sorry

theorem keith_receives_32_messages : messages_juan_to_keith = 32 := by sorry

end NUMINAMATH_CALUDE_laurence_to_missy_relation_keith_receives_32_messages_l2912_291249


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2912_291230

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 3 * a 7 = 8 →
  a 4 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2912_291230


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2912_291269

/-- The quadratic equation x^2 + ax + a = 0 has no real roots -/
def has_no_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a*x + a ≠ 0

/-- The condition 0 ≤ a ≤ 4 is necessary but not sufficient for x^2 + ax + a = 0 to have no real roots -/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, has_no_real_roots a → 0 ≤ a ∧ a ≤ 4) ∧
  (∃ a : ℝ, 0 ≤ a ∧ a ≤ 4 ∧ ¬(has_no_real_roots a)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2912_291269


namespace NUMINAMATH_CALUDE_pomelo_price_at_6kg_l2912_291295

-- Define the relationship between weight and price
def price_function (x : ℝ) : ℝ := 1.4 * x

-- Theorem statement
theorem pomelo_price_at_6kg : price_function 6 = 8.4 := by
  sorry

end NUMINAMATH_CALUDE_pomelo_price_at_6kg_l2912_291295


namespace NUMINAMATH_CALUDE_april_earnings_l2912_291262

def rose_price : ℕ := 7
def initial_roses : ℕ := 9
def remaining_roses : ℕ := 4

theorem april_earnings : (initial_roses - remaining_roses) * rose_price = 35 := by
  sorry

end NUMINAMATH_CALUDE_april_earnings_l2912_291262


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2912_291283

theorem max_value_of_fraction (x : ℝ) : (4*x^2 + 8*x + 19) / (4*x^2 + 8*x + 5) ≤ 15 := by
  sorry

#check max_value_of_fraction

end NUMINAMATH_CALUDE_max_value_of_fraction_l2912_291283


namespace NUMINAMATH_CALUDE_school_dinosaur_cost_l2912_291271

def dinosaur_model_cost : ℕ := 100

def kindergarten_models : ℕ := 2
def elementary_models : ℕ := 2 * kindergarten_models
def high_school_models : ℕ := 3 * kindergarten_models

def total_models : ℕ := kindergarten_models + elementary_models + high_school_models

def discount_rate : ℚ :=
  if total_models > 10 then 1/10
  else if total_models > 5 then 1/20
  else 0

def discounted_price : ℚ := dinosaur_model_cost * (1 - discount_rate)

def total_cost : ℚ := total_models * discounted_price

theorem school_dinosaur_cost : total_cost = 1080 := by
  sorry

end NUMINAMATH_CALUDE_school_dinosaur_cost_l2912_291271


namespace NUMINAMATH_CALUDE_function_max_abs_bound_l2912_291265

theorem function_max_abs_bound (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f (x : ℝ) := a * x^3 + (1 - 4*a) * x^2 + (5*a - 1) * x - 5*a + 3
  let g (x : ℝ) := (1 - a) * x^3 - x^2 + (2 - a) * x - 3*a - 1
  ∀ x : ℝ, max (|f x|) (|g x|) ≥ a + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_max_abs_bound_l2912_291265


namespace NUMINAMATH_CALUDE_determine_d_value_l2912_291274

theorem determine_d_value : ∃ d : ℝ, 
  (∀ x : ℝ, x * (2 * x + 3) < d ↔ -5/2 < x ∧ x < 3) → d = 15 := by
  sorry

end NUMINAMATH_CALUDE_determine_d_value_l2912_291274


namespace NUMINAMATH_CALUDE_expression_simplification_l2912_291221

theorem expression_simplification (a x : ℝ) 
  (h1 : x ≠ a / 3) (h2 : x ≠ -a / 3) (h3 : x ≠ -a) : 
  (3 * a^2 + 2 * a * x - x^2) / ((3 * x + a) * (a + x)) - 2 + 
  10 * (a * x - 3 * x^2) / (a^2 - 9 * x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2912_291221


namespace NUMINAMATH_CALUDE_sad_girls_count_l2912_291264

theorem sad_girls_count (total_children happy_children sad_children neutral_children
                         boys girls happy_boys neutral_boys : ℕ)
                        (h1 : total_children = 60)
                        (h2 : happy_children = 30)
                        (h3 : sad_children = 10)
                        (h4 : neutral_children = 20)
                        (h5 : boys = 16)
                        (h6 : girls = 44)
                        (h7 : happy_boys = 6)
                        (h8 : neutral_boys = 4)
                        (h9 : total_children = happy_children + sad_children + neutral_children)
                        (h10 : total_children = boys + girls) :
  girls - (happy_children - happy_boys) - (neutral_children - neutral_boys) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sad_girls_count_l2912_291264


namespace NUMINAMATH_CALUDE_equation_solution_l2912_291253

theorem equation_solution (x a : ℝ) : 
  3 * x + 6 = 12 ∧ 6 * x + 3 * a = 24 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2912_291253


namespace NUMINAMATH_CALUDE_kite_smallest_angle_l2912_291231

/-- Represents the angles of a kite in degrees -/
structure KiteAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for a valid kite with angles in arithmetic sequence -/
def is_valid_kite (k : KiteAngles) : Prop :=
  k.a > 0 ∧ 
  k.a + k.d > 0 ∧ 
  k.a + 2*k.d > 0 ∧ 
  k.a + 3*k.d > 0 ∧
  k.a + (k.a + 3*k.d) = 180 ∧  -- opposite angles are supplementary
  k.a + 3*k.d = 150  -- largest angle is 150°

theorem kite_smallest_angle (k : KiteAngles) (h : is_valid_kite k) : k.a = 15 := by
  sorry

end NUMINAMATH_CALUDE_kite_smallest_angle_l2912_291231


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainder_one_l2912_291245

theorem smallest_positive_integer_with_remainder_one : ∃ m : ℕ,
  m > 1 ∧
  m % 3 = 1 ∧
  m % 5 = 1 ∧
  m % 7 = 1 ∧
  (∀ n : ℕ, n > 1 → n % 3 = 1 → n % 5 = 1 → n % 7 = 1 → m ≤ n) ∧
  m = 106 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainder_one_l2912_291245


namespace NUMINAMATH_CALUDE_smallest_factor_for_cube_l2912_291219

theorem smallest_factor_for_cube (a : ℕ) : a = 1575 ↔ 
  (a > 0 ∧ 
   ∃ n : ℕ, 5880 * a = n^3 ∧ 
   ∀ b : ℕ, b > 0 → b < a → ¬∃ m : ℕ, 5880 * b = m^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_cube_l2912_291219


namespace NUMINAMATH_CALUDE_shopping_mall_probabilities_l2912_291218

/-- Probability of a customer buying product A -/
def prob_A : ℝ := 0.5

/-- Probability of a customer buying product B -/
def prob_B : ℝ := 0.6

/-- Probability of a customer buying neither product A nor B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- Probability of a customer buying at least one product -/
def prob_at_least_one : ℝ := 1 - prob_neither

theorem shopping_mall_probabilities :
  (1 - (prob_A * prob_B) - prob_neither = 0.5) ∧
  (1 - (prob_at_least_one^3 + 3 * prob_at_least_one^2 * prob_neither) = 0.104) :=
sorry

end NUMINAMATH_CALUDE_shopping_mall_probabilities_l2912_291218


namespace NUMINAMATH_CALUDE_min_point_sum_l2912_291236

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem min_point_sum :
  ∃ (a b : ℝ), (∀ x, f x ≥ f a) ∧ (f a = b) ∧ (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_min_point_sum_l2912_291236


namespace NUMINAMATH_CALUDE_hexagon_percentage_is_25_percent_l2912_291250

/-- Represents a tiling of a plane with squares and hexagons -/
structure PlaneTiling where
  /-- The fraction of each large square unit covered by squares -/
  square_fraction : ℚ
  /-- The fraction of each large square unit covered by hexagons -/
  hexagon_fraction : ℚ
  /-- The sum of square_fraction and hexagon_fraction equals 1 -/
  fraction_sum_one : square_fraction + hexagon_fraction = 1

/-- The percentage of the plane enclosed by hexagons -/
def hexagon_percentage (tiling : PlaneTiling) : ℚ :=
  tiling.hexagon_fraction * 100

theorem hexagon_percentage_is_25_percent (tiling : PlaneTiling) 
  (h : tiling.square_fraction = 3/4) : hexagon_percentage tiling = 25 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_percentage_is_25_percent_l2912_291250


namespace NUMINAMATH_CALUDE_min_bags_l2912_291240

theorem min_bags (total_objects : ℕ) (red_boxes blue_boxes : ℕ) 
  (objects_per_red : ℕ) (objects_per_blue : ℕ) :
  total_objects = 731 ∧ 
  red_boxes = 17 ∧ 
  blue_boxes = 43 ∧ 
  objects_per_red = 43 ∧ 
  objects_per_blue = 17 →
  ∃ (n : ℕ), n > 0 ∧ 
    (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
      objects_per_red * a + objects_per_blue * b = total_objects) ∧
    (∀ (m : ℕ), m > 0 ∧ 
      (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
        objects_per_red * a + objects_per_blue * b = total_objects) → 
      n ≤ m) ∧
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_bags_l2912_291240


namespace NUMINAMATH_CALUDE_larger_circle_radius_l2912_291293

theorem larger_circle_radius (r : ℝ) (h1 : r = 2) : ∃ R : ℝ,
  (∀ i j : Fin 4, i ≠ j → (∃ c₁ c₂ : ℝ × ℝ, 
    dist c₁ c₂ = 2 * r ∧ 
    (∀ x : ℝ × ℝ, dist x c₁ ≤ r ∨ dist x c₂ ≤ r))) →
  (∃ C : ℝ × ℝ, ∀ i : Fin 4, ∃ c : ℝ × ℝ, 
    dist C c = R - r ∧ 
    (∀ x : ℝ × ℝ, dist x c ≤ r → dist x C ≤ R)) →
  R = 4 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l2912_291293


namespace NUMINAMATH_CALUDE_reciprocal_sum_diff_l2912_291229

theorem reciprocal_sum_diff : (1 / (1/4 + 1/6 - 1/12) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_diff_l2912_291229


namespace NUMINAMATH_CALUDE_family_gallery_photos_l2912_291290

/-- Proves that the initial number of photos in the family gallery was 400 --/
theorem family_gallery_photos : 
  ∀ (P : ℕ), 
  (P + (P / 2) + (P / 2 + 120) = 920) → 
  P = 400 := by
sorry

end NUMINAMATH_CALUDE_family_gallery_photos_l2912_291290


namespace NUMINAMATH_CALUDE_course_selection_schemes_l2912_291275

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def min_courses : ℕ := 2
def max_courses : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

def two_course_selections : ℕ := choose physical_education_courses 1 * choose art_courses 1

def three_course_selections : ℕ := 
  choose physical_education_courses 2 * choose art_courses 1 +
  choose physical_education_courses 1 * choose art_courses 2

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_schemes : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l2912_291275


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_ratio_l2912_291216

/-- Given a quadratic equation 3ax^2 + 6bx + 2c = 0 with zero discriminant,
    prove that b^2 = (2/3)ac -/
theorem zero_discriminant_implies_ratio (a b c : ℝ) :
  (6 * b)^2 - 4 * (3 * a) * (2 * c) = 0 →
  b^2 = (2/3) * a * c := by
  sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_ratio_l2912_291216


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l2912_291211

theorem fourth_root_equation_solution :
  ∀ x : ℝ, (x > 0 ∧ x^(1/4) = 16 / (8 - x^(1/4))) ↔ x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l2912_291211


namespace NUMINAMATH_CALUDE_faster_train_speed_l2912_291202

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (train_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  crossing_time = 8 →
  let relative_speed := 2 * train_length / crossing_time
  let slower_speed := relative_speed / 3
  let faster_speed := 2 * slower_speed
  faster_speed = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l2912_291202


namespace NUMINAMATH_CALUDE_flower_theorem_l2912_291201

def flower_problem (alissa_flowers melissa_flowers flowers_left : ℕ) : Prop :=
  alissa_flowers + melissa_flowers - flowers_left = 18

theorem flower_theorem :
  flower_problem 16 16 14 := by
  sorry

end NUMINAMATH_CALUDE_flower_theorem_l2912_291201


namespace NUMINAMATH_CALUDE_complex_multiplication_l2912_291296

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  i * (2 - i) = 1 + 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2912_291296


namespace NUMINAMATH_CALUDE_batsman_average_runs_l2912_291203

def average_runs (total_runs : ℕ) (num_matches : ℕ) : ℚ :=
  (total_runs : ℚ) / (num_matches : ℚ)

theorem batsman_average_runs :
  let first_20_matches := 20
  let next_10_matches := 10
  let total_matches := first_20_matches + next_10_matches
  let avg_first_20 := 40
  let avg_next_10 := 13
  let total_runs_first_20 := first_20_matches * avg_first_20
  let total_runs_next_10 := next_10_matches * avg_next_10
  let total_runs := total_runs_first_20 + total_runs_next_10
  average_runs total_runs total_matches = 31 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_runs_l2912_291203


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l2912_291243

/-- The area of a square wrapping paper used to wrap a rectangular box with a square base. -/
theorem wrapping_paper_area (w h x : ℝ) (hw : w > 0) (hh : h > 0) (hx : x ≥ 0) :
  let s := Real.sqrt ((h + x)^2 + (w/2)^2)
  s^2 = (h + x)^2 + w^2/4 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l2912_291243


namespace NUMINAMATH_CALUDE_larger_cuboid_height_l2912_291281

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem larger_cuboid_height :
  let smallerCuboid : CuboidDimensions := ⟨5, 6, 3⟩
  let largerCuboidBase : CuboidDimensions := ⟨18, 15, 2⟩
  let numSmallerCuboids : ℕ := 6
  cuboidVolume largerCuboidBase = numSmallerCuboids * cuboidVolume smallerCuboid := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_height_l2912_291281


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_fraction_l2912_291247

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + 4 * x + b

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a 2 x ≥ 0) → a ≥ -5/2 :=
sorry

-- Theorem 2
theorem min_value_of_fraction (a b : ℝ) :
  a > b →
  (∀ x : ℝ, f a b x ≥ 0) →
  (∃ x₀ : ℝ, f a b x₀ = 0) →
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_fraction_l2912_291247


namespace NUMINAMATH_CALUDE_inequality_condition_l2912_291244

theorem inequality_condition (a b : ℝ) :
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2912_291244


namespace NUMINAMATH_CALUDE_matchsticks_left_l2912_291288

def totalMatchsticks : ℕ := 50
def elvisSquareMatchsticks : ℕ := 4
def ralphSquareMatchsticks : ℕ := 8
def zoeyTriangleMatchsticks : ℕ := 6
def elvisMaxMatchsticks : ℕ := 20
def ralphMaxMatchsticks : ℕ := 20
def zoeyMaxMatchsticks : ℕ := 15
def maxTotalShapes : ℕ := 9

theorem matchsticks_left : 
  ∃ (elvisShapes ralphShapes zoeyShapes : ℕ),
    elvisShapes * elvisSquareMatchsticks ≤ elvisMaxMatchsticks ∧
    ralphShapes * ralphSquareMatchsticks ≤ ralphMaxMatchsticks ∧
    zoeyShapes * zoeyTriangleMatchsticks ≤ zoeyMaxMatchsticks ∧
    elvisShapes + ralphShapes + zoeyShapes = maxTotalShapes ∧
    totalMatchsticks - (elvisShapes * elvisSquareMatchsticks + 
                        ralphShapes * ralphSquareMatchsticks + 
                        zoeyShapes * zoeyTriangleMatchsticks) = 2 :=
by sorry

end NUMINAMATH_CALUDE_matchsticks_left_l2912_291288


namespace NUMINAMATH_CALUDE_rebus_solution_l2912_291256

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (A < 10 ∧ B < 10 ∧ C < 10) ∧
  (100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C) ∧
  (100 * A + 10 * C + C = 1416) := by
sorry

end NUMINAMATH_CALUDE_rebus_solution_l2912_291256


namespace NUMINAMATH_CALUDE_class_size_calculation_l2912_291260

theorem class_size_calculation (baseball_football : ℕ) (only_baseball : ℕ) (only_football : ℕ) 
  (basketball_as_well : ℕ) (basketball_football_not_baseball : ℕ) (all_three : ℕ) (none : ℕ) :
  baseball_football = 7 →
  only_baseball = 3 →
  only_football = 4 →
  basketball_as_well = 2 →
  basketball_football_not_baseball = 1 →
  all_three = 2 →
  none = 5 →
  baseball_football + only_baseball + only_football + basketball_as_well + 
  basketball_football_not_baseball + none = 17 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l2912_291260


namespace NUMINAMATH_CALUDE_power_function_decreasing_interval_l2912_291291

/-- A power function passing through (2, 4) is monotonically decreasing on (-∞, 0) -/
theorem power_function_decreasing_interval 
  (f : ℝ → ℝ) 
  (α : ℝ) 
  (h1 : ∀ x : ℝ, f x = x^α) 
  (h2 : f 2 = 4) :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_power_function_decreasing_interval_l2912_291291


namespace NUMINAMATH_CALUDE_polyhedron_volume_l2912_291285

theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h1 : prism_volume = Real.sqrt 2 - 1)
  (h2 : pyramid_volume = 1/6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2/3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l2912_291285


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l2912_291287

/-- Represents the dimensions and properties of a metallic sheet and the resulting box --/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSquareSide : ℝ
  boxVolume : ℝ

/-- Theorem stating the width of the metallic sheet given the conditions --/
theorem metallic_sheet_width (sheet : MetallicSheet) 
  (h1 : sheet.length = 48)
  (h2 : sheet.cutSquareSide = 7)
  (h3 : sheet.boxVolume = 5236)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSquareSide) * 
                          (sheet.width - 2 * sheet.cutSquareSide) * 
                          sheet.cutSquareSide) : 
  sheet.width = 36 := by
  sorry

#check metallic_sheet_width

end NUMINAMATH_CALUDE_metallic_sheet_width_l2912_291287


namespace NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l2912_291235

/-- A quadratic function f(x) = x^2 + 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- The discriminant of the quadratic function f -/
def discriminant (a : ℝ) : ℝ := 4 - 4*a

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0) → a > 1 := by
  sorry

#check no_real_roots_implies_a_greater_than_one

end NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l2912_291235


namespace NUMINAMATH_CALUDE_kim_cherry_saplings_l2912_291241

/-- Given that Kim plants 80 cherry pits, 25% of them sprout, and she sells 6 saplings,
    prove that she has 14 cherry saplings left. -/
theorem kim_cherry_saplings (total_pits : ℕ) (sprout_rate : ℚ) (sold_saplings : ℕ) :
  total_pits = 80 →
  sprout_rate = 1/4 →
  sold_saplings = 6 →
  (total_pits : ℚ) * sprout_rate - sold_saplings = 14 := by
  sorry

end NUMINAMATH_CALUDE_kim_cherry_saplings_l2912_291241


namespace NUMINAMATH_CALUDE_tavern_keeper_pays_for_beer_l2912_291233

/-- Represents the currency of a country -/
structure Currency where
  name : String
  value : ℚ

/-- Represents a country with its currency and exchange rate -/
structure Country where
  name : String
  currency : Currency
  exchangeRate : ℚ

/-- Represents a transaction in a country -/
structure Transaction where
  country : Country
  amountPaid : ℚ
  itemCost : ℚ
  changeReceived : ℚ

/-- The beer lover's transactions -/
def beerLoverTransactions (anchuria gvaiasuela : Country) : List Transaction := sorry

/-- The tavern keeper's profit or loss -/
def tavernKeeperProfit (transactions : List Transaction) : ℚ := sorry

/-- Theorem stating that the tavern keeper pays for the beer -/
theorem tavern_keeper_pays_for_beer (anchuria gvaiasuela : Country) 
  (h1 : anchuria.currency.value = gvaiasuela.currency.value)
  (h2 : anchuria.exchangeRate = 90 / 100)
  (h3 : gvaiasuela.exchangeRate = 90 / 100)
  (h4 : ∀ t ∈ beerLoverTransactions anchuria gvaiasuela, t.itemCost = 10 / 100) :
  tavernKeeperProfit (beerLoverTransactions anchuria gvaiasuela) < 0 := by
  sorry

#check tavern_keeper_pays_for_beer

end NUMINAMATH_CALUDE_tavern_keeper_pays_for_beer_l2912_291233


namespace NUMINAMATH_CALUDE_decimal_to_binary_23_l2912_291209

theorem decimal_to_binary_23 : 
  (23 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_23_l2912_291209


namespace NUMINAMATH_CALUDE_wax_sculpture_problem_l2912_291204

theorem wax_sculpture_problem (large_animal_wax : ℕ) (small_animal_wax : ℕ) 
  (small_animal_total_wax : ℕ) (total_wax : ℕ) :
  large_animal_wax = 4 →
  small_animal_wax = 2 →
  small_animal_total_wax = 12 →
  total_wax = 20 →
  total_wax = small_animal_total_wax + (total_wax - small_animal_total_wax) :=
by sorry

end NUMINAMATH_CALUDE_wax_sculpture_problem_l2912_291204


namespace NUMINAMATH_CALUDE_power_function_through_point_l2912_291254

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2912_291254


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l2912_291280

def regular_ticket_cost : ℝ := 10

theorem museum_ticket_cost :
  let discounted_ticket := 0.7 * regular_ticket_cost
  let full_price_ticket := regular_ticket_cost
  let total_spent := 44
  2 * discounted_ticket + 3 * full_price_ticket = total_spent :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l2912_291280


namespace NUMINAMATH_CALUDE_min_c_value_l2912_291298

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : ∃! (x y : ℝ), 2*x + y = 2029 ∧ y = |x - a| + |x - b| + |x - c|) : 
  c ≥ 1015 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2912_291298


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2912_291226

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 120 0.80 0.30 = 151.20 := by
  sorry


end NUMINAMATH_CALUDE_stock_price_after_two_years_l2912_291226


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2912_291278

/-- Given a function f(x) = x³ + ax² + bx + 1 with f'(1) = 2a and f'(2) = -b,
    where a and b are real constants, the equation of the tangent line
    to y = f(x) at (1, f(1)) is 6x + 2y - 1 = 0. -/
theorem tangent_line_equation (a b : ℝ) :
  let f (x : ℝ) := x^3 + a*x^2 + b*x + 1
  let f' (x : ℝ) := 3*x^2 + 2*a*x + b
  f' 1 = 2*a →
  f' 2 = -b →
  ∃ (m c : ℝ), m = -3 ∧ c = -1/2 ∧ ∀ x y, y - f 1 = m * (x - 1) ↔ 6*x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2912_291278


namespace NUMINAMATH_CALUDE_triple_root_values_l2912_291237

/-- A polynomial with integer coefficients of the form x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 -/
def IntPolynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 24

/-- r is a triple root of the polynomial if (x - r)^3 divides the polynomial -/
def IsTripleRoot (r : ℤ) (b₄ b₃ b₂ b₁ : ℤ) : Prop :=
  ∃ (q : ℤ → ℤ), ∀ x, IntPolynomial b₄ b₃ b₂ b₁ x = (x - r)^3 * q x

theorem triple_root_values (r : ℤ) :
  (∃ b₄ b₃ b₂ b₁ : ℤ, IsTripleRoot r b₄ b₃ b₂ b₁) ↔ r ∈ ({-2, -1, 1, 2} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_triple_root_values_l2912_291237


namespace NUMINAMATH_CALUDE_casino_money_theorem_l2912_291252

/-- The amount of money on table A -/
def table_a : ℕ := 40

/-- The amount of money on table C -/
def table_c : ℕ := table_a + 20

/-- The amount of money on table B -/
def table_b : ℕ := 2 * table_c

/-- The total amount of money on all tables -/
def total_money : ℕ := table_a + table_b + table_c

theorem casino_money_theorem : total_money = 220 := by
  sorry

end NUMINAMATH_CALUDE_casino_money_theorem_l2912_291252


namespace NUMINAMATH_CALUDE_mod_equivalence_problem_l2912_291292

theorem mod_equivalence_problem : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_problem_l2912_291292


namespace NUMINAMATH_CALUDE_sum_of_valid_inputs_is_1020_l2912_291261

def machine_function (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 4 * n + 2

def apply_n_times (f : ℕ → ℕ) (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_n_times f n x)

def valid_inputs : List ℕ :=
  (List.range 1000).filter (λ n => apply_n_times machine_function 8 n = 2)

theorem sum_of_valid_inputs_is_1020 : valid_inputs.sum = 1020 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_valid_inputs_is_1020_l2912_291261


namespace NUMINAMATH_CALUDE_great_8_teams_l2912_291266

-- Define the number of teams
def n : ℕ := sorry

-- Define the total number of games
def total_games : ℕ := 36

-- Theorem stating the conditions and the result to be proven
theorem great_8_teams :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 9 := by sorry

end NUMINAMATH_CALUDE_great_8_teams_l2912_291266


namespace NUMINAMATH_CALUDE_smallest_p_value_l2912_291234

theorem smallest_p_value (p q : ℕ+) 
  (h1 : (5 : ℚ) / 8 < p / q)
  (h2 : p / q < (7 : ℚ) / 8)
  (h3 : p + q = 2005) : 
  p.val ≥ 772 ∧ (∀ m : ℕ+, m < p → ¬((5 : ℚ) / 8 < m / (2005 - m) ∧ m / (2005 - m) < (7 : ℚ) / 8)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_p_value_l2912_291234


namespace NUMINAMATH_CALUDE_problem_solution_l2912_291299

theorem problem_solution (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2912_291299


namespace NUMINAMATH_CALUDE_non_equilateral_combinations_l2912_291225

/-- The number of dots evenly spaced on the circle's circumference -/
def n : ℕ := 6

/-- The number of dots to be selected in each combination -/
def k : ℕ := 3

/-- The total number of combinations of k dots from n dots -/
def total_combinations : ℕ := Nat.choose n k

/-- The number of equilateral triangles that can be formed -/
def equilateral_triangles : ℕ := 2

/-- Theorem: The number of combinations of 3 dots that do not form an equilateral triangle
    is equal to the total number of 3-dot combinations minus the number of equilateral triangles -/
theorem non_equilateral_combinations :
  total_combinations - equilateral_triangles = 18 := by sorry

end NUMINAMATH_CALUDE_non_equilateral_combinations_l2912_291225


namespace NUMINAMATH_CALUDE_complement_of_union_l2912_291273

open Set

theorem complement_of_union (U M N : Set ℕ) : 
  U = {1, 2, 3, 4} →
  M = {1, 2} →
  N = {2, 3} →
  (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2912_291273


namespace NUMINAMATH_CALUDE_baseball_league_games_l2912_291207

theorem baseball_league_games (N M : ℕ) : 
  N > M →
  M > 5 →
  4 * N + 5 * M = 90 →
  4 * N = 60 := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l2912_291207


namespace NUMINAMATH_CALUDE_rotated_point_coordinates_l2912_291279

def triangle_OAB (A : ℝ × ℝ) : Prop :=
  A.1 ≥ 0 ∧ A.2 > 0

def angle_ABO_is_right (A : ℝ × ℝ) : Prop :=
  (A.2 / A.1) * 10 = A.2

def angle_AOB_is_30 (A : ℝ × ℝ) : Prop :=
  A.2 / A.1 = Real.sqrt 3 / 3

def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

theorem rotated_point_coordinates (A : ℝ × ℝ) :
  triangle_OAB A →
  angle_ABO_is_right A →
  angle_AOB_is_30 A →
  rotate_90_ccw A = (-10 * Real.sqrt 3 / 3, 10) := by
  sorry

end NUMINAMATH_CALUDE_rotated_point_coordinates_l2912_291279
