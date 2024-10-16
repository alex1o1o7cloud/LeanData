import Mathlib

namespace NUMINAMATH_CALUDE_unique_positive_number_l2052_205297

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l2052_205297


namespace NUMINAMATH_CALUDE_unique_krakozyabr_count_l2052_205264

def Krakozyabr : Type := Unit

structure KrakozyabrPopulation where
  total : ℕ
  horns : ℕ
  wings : ℕ
  both : ℕ
  all_have_horns_or_wings : total = horns + wings - both
  horns_with_wings_ratio : both = horns / 5
  wings_with_horns_ratio : both = wings / 4
  total_range : 25 < total ∧ total < 35

theorem unique_krakozyabr_count : 
  ∀ (pop : KrakozyabrPopulation), pop.total = 32 := by
  sorry

end NUMINAMATH_CALUDE_unique_krakozyabr_count_l2052_205264


namespace NUMINAMATH_CALUDE_fourth_root_64_times_cube_root_27_times_sqrt_9_l2052_205238

theorem fourth_root_64_times_cube_root_27_times_sqrt_9 :
  (64 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 18 * (2 : ℝ) ^ (1/2) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_64_times_cube_root_27_times_sqrt_9_l2052_205238


namespace NUMINAMATH_CALUDE_triangle_trigonometry_l2052_205257

theorem triangle_trigonometry (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  Real.cos C = 5 / 13 →
  Real.sin A = Real.sqrt 3 / 2 ∧
  Real.cos B = (12 * Real.sqrt 3 - 5) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_trigonometry_l2052_205257


namespace NUMINAMATH_CALUDE_like_terms_proof_l2052_205200

/-- Given that -3x^(m-1)y^3 and 4xy^(m+n) are like terms, prove that m = 2 and n = 1 -/
theorem like_terms_proof (m n : ℤ) : 
  (∀ x y : ℝ, ∃ k : ℝ, -3 * x^(m-1) * y^3 = k * 4 * x * y^(m+n)) → 
  m = 2 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_like_terms_proof_l2052_205200


namespace NUMINAMATH_CALUDE_freds_allowance_l2052_205228

/-- Fred's weekly allowance problem -/
theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 11 = 20) → allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_freds_allowance_l2052_205228


namespace NUMINAMATH_CALUDE_symmetry_x_axis_coordinates_l2052_205218

/-- Two points are symmetric with respect to the x-axis if they have the same x-coordinate
    and opposite y-coordinates -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given that P(-2, 3) is symmetric to Q(a, b) with respect to the x-axis,
    prove that a = -2 and b = -3 -/
theorem symmetry_x_axis_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let Q : ℝ × ℝ := (a, b)
  symmetric_x_axis P Q → a = -2 ∧ b = -3 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_x_axis_coordinates_l2052_205218


namespace NUMINAMATH_CALUDE_projection_problem_l2052_205251

/-- Given a projection that takes (2, -3) to (1, -3/2), 
    prove that the projection of (3, -2) is (24/13, -36/13) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (2, -3) = (1, -3/2)) :
  proj (3, -2) = (24/13, -36/13) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l2052_205251


namespace NUMINAMATH_CALUDE_roberts_ride_time_l2052_205244

/-- The time taken for Robert to ride along a semi-circular path on a highway segment -/
theorem roberts_ride_time 
  (highway_length : ℝ) 
  (highway_width : ℝ) 
  (speed : ℝ) 
  (miles_to_feet : ℝ) 
  (h1 : highway_length = 1) 
  (h2 : highway_width = 40) 
  (h3 : speed = 5) 
  (h4 : miles_to_feet = 5280) : 
  ∃ (time : ℝ), time = π / 10 := by
sorry

end NUMINAMATH_CALUDE_roberts_ride_time_l2052_205244


namespace NUMINAMATH_CALUDE_fraction_simplification_l2052_205223

theorem fraction_simplification : 
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2052_205223


namespace NUMINAMATH_CALUDE_a_squared_plus_b_squared_eq_25_l2052_205221

/-- Definition of the sequence S_n -/
def S (n : ℕ) : ℕ := sorry

/-- The guessed formula for S_{2n-1} -/
def S_odd (n a b : ℕ) : ℕ := (4*n - 3) * (a*n + b)

/-- Theorem stating the relation between a, b and the sequence S -/
theorem a_squared_plus_b_squared_eq_25 (a b : ℕ) :
  S 1 = 1 ∧ S 3 = 25 ∧ (∀ n, S (2*n - 1) = S_odd n a b) →
  a^2 + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_plus_b_squared_eq_25_l2052_205221


namespace NUMINAMATH_CALUDE_unique_integer_solution_inequality_proof_l2052_205259

-- Part 1
theorem unique_integer_solution (m : ℤ) 
  (h : ∃! (x : ℤ), |2*x - m| < 1 ∧ x = 2) : m = 4 := by
  sorry

-- Part 2
theorem inequality_proof (a b : ℝ) 
  (h1 : a * b = 4)
  (h2 : a > b)
  (h3 : b > 0) : 
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_inequality_proof_l2052_205259


namespace NUMINAMATH_CALUDE_calculation_proof_l2052_205266

theorem calculation_proof :
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-99 - 11/12) * 24 = -2398) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2052_205266


namespace NUMINAMATH_CALUDE_sequence_last_term_l2052_205230

theorem sequence_last_term (n : ℕ) : 2^n - 1 = 127 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sequence_last_term_l2052_205230


namespace NUMINAMATH_CALUDE_fib_gcd_property_fib_1960_1988_gcd_l2052_205250

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fib_gcd_property (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) := by
  sorry

theorem fib_1960_1988_gcd : Nat.gcd (fibonacci 1960) (fibonacci 1988) = fibonacci 28 := by
  sorry

end NUMINAMATH_CALUDE_fib_gcd_property_fib_1960_1988_gcd_l2052_205250


namespace NUMINAMATH_CALUDE_axis_of_symmetry_at_1_5_l2052_205211

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if g is symmetric about x = 1.5 -/
theorem axis_of_symmetry_at_1_5 (g : ℝ → ℝ) (h : IsSymmetricAbout1_5 g) :
  ∀ x y, g x = y → g (3 - x) = y :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_at_1_5_l2052_205211


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2052_205265

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₃ + a₅ + a₇ = 8256 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2052_205265


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l2052_205201

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem preimage_of_3_1 : f⁻¹' {(3, 1)} = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l2052_205201


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l2052_205273

/-- Given a rectangle with initial dimensions 5 × 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (initial_width initial_length : ℝ)
  (h_initial_width : initial_width = 5)
  (h_initial_length : initial_length = 7)
  (h_reduced_area : (initial_width - 2) * initial_length = 21) :
  initial_width * (initial_length - 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l2052_205273


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l2052_205295

def sum_of_divisors (n : ℕ) : ℕ := sorry

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l2052_205295


namespace NUMINAMATH_CALUDE_corn_farmer_profit_percentage_l2052_205256

/-- Calculates the profit percentage for a farmer's corn harvest. -/
theorem corn_farmer_profit_percentage 
  (seed_cost fertilizer_cost labor_cost : ℕ)
  (bags_harvested price_per_bag : ℕ)
  (h_seed : seed_cost = 50)
  (h_fertilizer : fertilizer_cost = 35)
  (h_labor : labor_cost = 15)
  (h_bags : bags_harvested = 10)
  (h_price : price_per_bag = 11) :
  let total_cost := seed_cost + fertilizer_cost + labor_cost
  let total_revenue := bags_harvested * price_per_bag
  let profit := total_revenue - total_cost
  (profit / total_cost : ℚ) = 1/10 := by
sorry

end NUMINAMATH_CALUDE_corn_farmer_profit_percentage_l2052_205256


namespace NUMINAMATH_CALUDE_chris_newspaper_collection_l2052_205240

theorem chris_newspaper_collection (chris_newspapers lily_newspapers : ℕ) : 
  lily_newspapers = chris_newspapers + 23 →
  chris_newspapers + lily_newspapers = 65 →
  chris_newspapers = 21 := by
sorry

end NUMINAMATH_CALUDE_chris_newspaper_collection_l2052_205240


namespace NUMINAMATH_CALUDE_linked_rings_distance_l2052_205288

/-- Calculates the total distance of a series of linked rings -/
def total_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let num_rings := (top_diameter - bottom_diameter) / 2 + 1
  let sum_inside_diameters := num_rings * (top_diameter - thickness + bottom_diameter - thickness) / 2
  sum_inside_diameters + 2 * thickness

/-- Theorem stating the total distance of the linked rings -/
theorem linked_rings_distance :
  total_distance 30 4 2 = 214 := by
  sorry

end NUMINAMATH_CALUDE_linked_rings_distance_l2052_205288


namespace NUMINAMATH_CALUDE_min_value_theorem_l2052_205231

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 1) :
  2/a + 3/b ≥ 25 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2052_205231


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2052_205260

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2052_205260


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l2052_205296

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem yuna_has_biggest_number :
  yuna_number = max yoongi_number (max jungkook_number yuna_number) :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l2052_205296


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l2052_205216

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*x + 3 - 2*m = 0

-- Define the condition for distinct real roots
def has_distinct_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m

-- Define the additional condition
def additional_condition (x₁ x₂ m : ℝ) : Prop :=
  x₁ * x₂ + x₁ + x₂ - m^2 = 4

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) :
  has_distinct_real_roots m ↔ m ≥ -1/2 :=
sorry

-- Theorem 2: Value of m
theorem value_of_m (m : ℝ) :
  has_distinct_real_roots m ∧
  (∃ (x₁ x₂ : ℝ), quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ additional_condition x₁ x₂ m) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l2052_205216


namespace NUMINAMATH_CALUDE_divisibility_condition_l2052_205263

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2052_205263


namespace NUMINAMATH_CALUDE_uncle_pill_duration_l2052_205267

/-- Represents the duration in days that a bottle of pills lasts -/
def bottle_duration (pills_per_bottle : ℕ) (dose : ℚ) (days_between_doses : ℕ) : ℚ :=
  (pills_per_bottle : ℚ) * (days_between_doses : ℚ) / dose

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem uncle_pill_duration :
  let pills_per_bottle : ℕ := 60
  let dose : ℚ := 3/4
  let days_between_doses : ℕ := 3
  days_to_months (bottle_duration pills_per_bottle dose days_between_doses) = 8 := by
  sorry

#eval days_to_months (bottle_duration 60 (3/4) 3)

end NUMINAMATH_CALUDE_uncle_pill_duration_l2052_205267


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l2052_205203

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of the polygon formed by overlapping rotated squares -/
def overlappingArea (sheets : List Sheet) : ℝ :=
  sorry

theorem overlap_area_theorem : 
  let sheets : List Sheet := [
    { side := 8, rotation := 0 },
    { side := 8, rotation := 15 },
    { side := 8, rotation := 45 },
    { side := 8, rotation := 75 }
  ]
  overlappingArea sheets = 512 := by sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l2052_205203


namespace NUMINAMATH_CALUDE_inequality_pattern_l2052_205212

theorem inequality_pattern (x : ℝ) (a : ℝ) 
  (h_x : x > 0)
  (h1 : x + 1/x ≥ 2)
  (h2 : x + 4/x^2 ≥ 3)
  (h3 : x + 27/x^3 ≥ 4)
  (h4 : x + a/x^4 ≥ 5) :
  a = 4^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_pattern_l2052_205212


namespace NUMINAMATH_CALUDE_jack_birth_year_l2052_205291

def first_amc8_year : ℕ := 1990

def jack_age_at_ninth_amc8 : ℕ := 15

def ninth_amc8_year : ℕ := first_amc8_year + 8

theorem jack_birth_year :
  first_amc8_year = 1990 →
  jack_age_at_ninth_amc8 = 15 →
  ninth_amc8_year - jack_age_at_ninth_amc8 = 1983 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_birth_year_l2052_205291


namespace NUMINAMATH_CALUDE_lunch_cost_calculation_l2052_205208

/-- Calculates the total cost of lunches for a field trip --/
theorem lunch_cost_calculation (total_lunches : ℕ) 
  (vegetarian_lunches : ℕ) (gluten_free_lunches : ℕ) (both_veg_gf : ℕ)
  (regular_cost : ℕ) (special_cost : ℕ) (both_cost : ℕ) : 
  total_lunches = 44 ∧ 
  vegetarian_lunches = 10 ∧ 
  gluten_free_lunches = 5 ∧ 
  both_veg_gf = 2 ∧
  regular_cost = 7 ∧ 
  special_cost = 8 ∧ 
  both_cost = 9 → 
  (both_veg_gf * both_cost + 
   (vegetarian_lunches - both_veg_gf) * special_cost + 
   (gluten_free_lunches - both_veg_gf) * special_cost + 
   (total_lunches - vegetarian_lunches - gluten_free_lunches + both_veg_gf) * regular_cost) = 323 := by
  sorry


end NUMINAMATH_CALUDE_lunch_cost_calculation_l2052_205208


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2052_205207

theorem repeating_decimal_47 : ∃ (x : ℚ), x = 47 / 99 ∧ 
  (∀ (n : ℕ), (100 * x - ⌊100 * x⌋) * 10^n = 47 / 100) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2052_205207


namespace NUMINAMATH_CALUDE_circle_transformation_l2052_205280

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point vertically by a given amount -/
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ := (p.1, p.2 + dy)

/-- The initial coordinates of the center of circle S -/
def initial_point : ℝ × ℝ := (3, -4)

/-- The vertical translation amount -/
def translation_amount : ℝ := 5

theorem circle_transformation :
  translate_y (reflect_x initial_point) translation_amount = (3, 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_transformation_l2052_205280


namespace NUMINAMATH_CALUDE_f_f_has_four_roots_l2052_205286

def f (x : ℝ) := x^2 - 3*x + 2

theorem f_f_has_four_roots :
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, f (f x) = 0) ∧ (∀ y, f (f y) = 0 → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_f_f_has_four_roots_l2052_205286


namespace NUMINAMATH_CALUDE_pascals_triangle_50th_number_l2052_205289

theorem pascals_triangle_50th_number (n : ℕ) (h : n + 1 = 52) : 
  Nat.choose n 49 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_50th_number_l2052_205289


namespace NUMINAMATH_CALUDE_average_temperature_l2052_205247

/-- The average temperature for Monday, Tuesday, Wednesday, and Thursday given the conditions -/
theorem average_temperature (t w th : ℝ) : 
  (t + w + th + 33) / 4 = 46 →
  (41 + t + w + th) / 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_l2052_205247


namespace NUMINAMATH_CALUDE_profit_is_eight_percent_l2052_205220

/-- Given a markup percentage and a discount percentage, calculate the profit percentage. -/
def profit_percentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100

/-- Theorem stating that given a 30% markup and 16.92307692307692% discount, the profit is 8%. -/
theorem profit_is_eight_percent :
  profit_percentage 0.3 0.1692307692307692 = 8 := by
  sorry

end NUMINAMATH_CALUDE_profit_is_eight_percent_l2052_205220


namespace NUMINAMATH_CALUDE_seventh_alignment_time_l2052_205283

/-- Represents a standard clock with 12 divisions -/
structure Clock :=
  (divisions : Nat)
  (minute_hand_speed : Nat)
  (hour_hand_speed : Nat)

/-- Represents a time in hours and minutes -/
structure Time :=
  (hours : Nat)
  (minutes : Nat)

/-- Calculates the time until the nth alignment of clock hands -/
def time_until_nth_alignment (c : Clock) (start : Time) (n : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem seventh_alignment_time (c : Clock) (start : Time) :
  c.divisions = 12 →
  c.minute_hand_speed = 12 →
  c.hour_hand_speed = 1 →
  start.hours = 16 →
  start.minutes = 45 →
  time_until_nth_alignment c start 7 = 435 :=
sorry

end NUMINAMATH_CALUDE_seventh_alignment_time_l2052_205283


namespace NUMINAMATH_CALUDE_clea_escalator_time_l2052_205272

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed (units per second) -/
  walking_speed : ℝ
  /-- Total distance of the escalator (units) -/
  escalator_distance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalator_speed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def time_stationary (scenario : EscalatorScenario) : ℝ :=
  80

/-- Time taken for Clea to walk down the moving escalator -/
def time_moving (scenario : EscalatorScenario) : ℝ :=
  32

/-- Theorem stating the time taken for the given scenario -/
theorem clea_escalator_time (scenario : EscalatorScenario) :
  scenario.escalator_speed = 1.5 * scenario.walking_speed →
  (scenario.escalator_distance / scenario.walking_speed / 2) +
  (scenario.escalator_distance / (2 * scenario.escalator_speed)) = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clea_escalator_time_l2052_205272


namespace NUMINAMATH_CALUDE_div_three_sevenths_by_four_l2052_205278

theorem div_three_sevenths_by_four :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_div_three_sevenths_by_four_l2052_205278


namespace NUMINAMATH_CALUDE_sam_coin_problem_l2052_205248

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total number of coins -/
def totalCoins (coins : CoinCount) : ℕ :=
  coins.dimes + coins.nickels + coins.pennies

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Represents the transactions Sam made -/
def samTransactions (initial : CoinCount) : CoinCount :=
  let afterDad := CoinCount.mk (initial.dimes + 7) (initial.nickels - 3) initial.pennies
  CoinCount.mk (afterDad.dimes + 2) afterDad.nickels 2

theorem sam_coin_problem (initial : CoinCount) 
  (h_initial : initial = CoinCount.mk 9 5 12) : 
  let final := samTransactions initial
  totalCoins final = 22 ∧ totalValue final = 192 := by
  sorry

end NUMINAMATH_CALUDE_sam_coin_problem_l2052_205248


namespace NUMINAMATH_CALUDE_min_sum_squares_l2052_205222

def S : Finset Int := {-9, -6, -3, 0, 1, 3, 6, 10}

theorem min_sum_squares (a b c d e f g h : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) 
  (he : e ∈ S) (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
               f ≠ g ∧ f ≠ h ∧
               g ≠ h) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 2 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧
    a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
    b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
    c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
    d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
    e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
    f' ≠ g' ∧ f' ≠ h' ∧
    g' ≠ h' ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2052_205222


namespace NUMINAMATH_CALUDE_passengers_boarding_other_stops_eq_five_l2052_205246

/-- Calculates the number of passengers who got on the bus at other stops -/
def passengers_boarding_other_stops (initial : ℕ) (first_stop : ℕ) (getting_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial + first_stop - getting_off)

/-- Theorem: Given the initial, first stop, getting off, and final passenger counts, 
    prove that 5 passengers got on at other stops -/
theorem passengers_boarding_other_stops_eq_five :
  passengers_boarding_other_stops 50 16 22 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_passengers_boarding_other_stops_eq_five_l2052_205246


namespace NUMINAMATH_CALUDE_f_1998_is_zero_l2052_205284

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_1998_is_zero
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 1998 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_1998_is_zero_l2052_205284


namespace NUMINAMATH_CALUDE_birth_outcome_probabilities_l2052_205243

def num_children : ℕ := 5
def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

theorem birth_outcome_probabilities :
  let prob_all_boys : ℚ := prob_boy ^ num_children
  let prob_all_girls : ℚ := prob_girl ^ num_children
  let prob_three_girls_two_boys : ℚ := (Nat.choose num_children 3) * (prob_girl ^ 3) * (prob_boy ^ 2)
  let prob_four_one : ℚ := 2 * (Nat.choose num_children 1) * (prob_girl ^ 4) * prob_boy
  prob_three_girls_two_boys = prob_four_one ∧
  prob_three_girls_two_boys > prob_all_boys ∧
  prob_three_girls_two_boys > prob_all_girls :=
by
  sorry

#check birth_outcome_probabilities

end NUMINAMATH_CALUDE_birth_outcome_probabilities_l2052_205243


namespace NUMINAMATH_CALUDE_certain_number_is_seven_l2052_205252

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_is_seven (n : ℕ) (h : factorial 9 / factorial n = 72) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_seven_l2052_205252


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2052_205287

theorem units_digit_of_7_cubed (n : ℕ) : n = 7^3 → n % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l2052_205287


namespace NUMINAMATH_CALUDE_kyle_age_l2052_205237

/-- Given the relationships between Kyle, Julian, Frederick, and Tyson's ages, prove Kyle's age. -/
theorem kyle_age (tyson_age : ℕ) (kyle_julian : ℕ) (julian_frederick : ℕ) (frederick_tyson : ℕ) :
  tyson_age = 20 →
  kyle_julian = 5 →
  julian_frederick = 20 →
  frederick_tyson = 2 →
  tyson_age * frederick_tyson - julian_frederick + kyle_julian = 25 :=
by sorry

end NUMINAMATH_CALUDE_kyle_age_l2052_205237


namespace NUMINAMATH_CALUDE_max_planes_in_hangar_l2052_205205

def hangar_length : ℕ := 300
def plane_length : ℕ := 40

theorem max_planes_in_hangar :
  (hangar_length / plane_length : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_planes_in_hangar_l2052_205205


namespace NUMINAMATH_CALUDE_factorial_calculation_l2052_205239

theorem factorial_calculation : (Nat.factorial 9 * Nat.factorial 5 * Nat.factorial 2) / (Nat.factorial 8 * Nat.factorial 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l2052_205239


namespace NUMINAMATH_CALUDE_circplus_not_commutative_l2052_205227

/-- Definition of the ⊕ operation -/
def circplus (a b : ℚ) : ℚ := a * b + 2 * a

/-- Theorem stating that ⊕ is not commutative -/
theorem circplus_not_commutative : ¬ (∀ a b : ℚ, circplus a b = circplus b a) := by
  sorry

end NUMINAMATH_CALUDE_circplus_not_commutative_l2052_205227


namespace NUMINAMATH_CALUDE_matrix_power_negative_identity_l2052_205255

open Matrix

theorem matrix_power_negative_identity
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (h : ∃ (n : ℕ), n ≠ 0 ∧ A ^ n = -1 • 1) :
  A ^ 2 = -1 • 1 ∨ A ^ 3 = -1 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_negative_identity_l2052_205255


namespace NUMINAMATH_CALUDE_sarah_hardback_count_l2052_205270

/-- The number of hardback books Sarah bought -/
def sarah_hardback : ℕ := sorry

/-- The number of paperback books Sarah bought -/
def sarah_paperback : ℕ := 6

/-- The number of paperback books Sarah's brother bought -/
def brother_paperback : ℕ := sarah_paperback / 3

/-- The number of hardback books Sarah's brother bought -/
def brother_hardback : ℕ := 2 * sarah_hardback

/-- The total number of books Sarah's brother bought -/
def brother_total : ℕ := 10

theorem sarah_hardback_count : sarah_hardback = 4 := by
  sorry

end NUMINAMATH_CALUDE_sarah_hardback_count_l2052_205270


namespace NUMINAMATH_CALUDE_equation_solution_l2052_205249

theorem equation_solution : ∃ x : ℤ, (158 - x = 59) ∧ (x = 99) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2052_205249


namespace NUMINAMATH_CALUDE_unique_right_triangle_l2052_205219

/-- Check if three numbers form a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The given sets of line segments -/
def segment_sets : List (ℕ × ℕ × ℕ) :=
  [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]

/-- Theorem: Only one set in segment_sets satisfies the Pythagorean theorem -/
theorem unique_right_triangle : 
  ∃! (a b c : ℕ), (a, b, c) ∈ segment_sets ∧ is_pythagorean_triple a b c :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_l2052_205219


namespace NUMINAMATH_CALUDE_expansion_distinct_terms_l2052_205206

/-- The number of distinct terms in the expansion of (a+b)(a+c+d+e+f) -/
def num_distinct_terms : ℕ := 9

/-- The first polynomial -/
def first_poly (a b : ℝ) : ℝ := a + b

/-- The second polynomial -/
def second_poly (a c d e f : ℝ) : ℝ := a + c + d + e + f

/-- Theorem stating that the number of distinct terms in the expansion is 9 -/
theorem expansion_distinct_terms 
  (a b c d e f : ℝ) : 
  num_distinct_terms = 9 := by sorry

end NUMINAMATH_CALUDE_expansion_distinct_terms_l2052_205206


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l2052_205275

theorem consecutive_even_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2 ∧ d = n + 4) →
  (a + c = 92) →
  d = 50 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l2052_205275


namespace NUMINAMATH_CALUDE_loan_sum_calculation_l2052_205241

/-- Proves that a sum P lent at 6% simple interest per annum for 8 years, 
    where the interest is $572 less than P, equals $1100. -/
theorem loan_sum_calculation (P : ℝ) : 
  (P * 0.06 * 8 = P - 572) → P = 1100 := by
  sorry

end NUMINAMATH_CALUDE_loan_sum_calculation_l2052_205241


namespace NUMINAMATH_CALUDE_curtis_farm_egg_laying_hens_l2052_205202

/-- The number of egg-laying hens on Mr. Curtis's farm -/
def egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) : ℕ :=
  total_chickens - roosters - non_laying_hens

/-- Theorem stating the number of egg-laying hens on Mr. Curtis's farm -/
theorem curtis_farm_egg_laying_hens :
  egg_laying_hens 325 28 20 = 277 := by
  sorry

end NUMINAMATH_CALUDE_curtis_farm_egg_laying_hens_l2052_205202


namespace NUMINAMATH_CALUDE_abc_sum_problem_l2052_205213

theorem abc_sum_problem (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  100 * A + 10 * B + C + 10 * A + B + A = C →
  C = 1 := by sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l2052_205213


namespace NUMINAMATH_CALUDE_waiting_by_stump_is_random_waiting_by_stump_unique_random_l2052_205236

-- Define the type for idioms
inductive Idiom
  | FishingForMoon
  | CastlesInAir
  | WaitingByStump
  | CatchingTurtle

-- Define a property for idioms
def describesRandomEvent (i : Idiom) : Prop :=
  match i with
  | Idiom.FishingForMoon => False
  | Idiom.CastlesInAir => False
  | Idiom.WaitingByStump => True
  | Idiom.CatchingTurtle => False

-- Theorem stating that "Waiting by a stump for a hare" describes a random event
theorem waiting_by_stump_is_random :
  describesRandomEvent Idiom.WaitingByStump :=
by sorry

-- Theorem stating that "Waiting by a stump for a hare" is the only idiom
-- among the given options that describes a random event
theorem waiting_by_stump_unique_random :
  ∀ (i : Idiom), describesRandomEvent i ↔ i = Idiom.WaitingByStump :=
by sorry

end NUMINAMATH_CALUDE_waiting_by_stump_is_random_waiting_by_stump_unique_random_l2052_205236


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_angle_l2052_205253

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - φ)

theorem function_properties_and_triangle_angle 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < Real.pi / 2)
  (h_point : f ω φ (Real.pi / 4) = Real.sqrt 3 / 2)
  (h_symmetry : ∃ (k : ℤ), Real.pi / 2 = 2 * Real.pi / ω - 2 * Real.pi * k / ω)
  (h_triangle : ∃ (A : ℝ), 0 < A ∧ A < Real.pi ∧ f ω φ (A / 2) + Real.cos A = 1 / 2) :
  ω = 2 ∧ φ = Real.pi / 6 ∧ ∃ (A : ℝ), A = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_angle_l2052_205253


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l2052_205292

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b :=
by sorry

theorem twelve_divided_by_one_sixth : 
  12 / (1 / 6) = 72 :=
by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l2052_205292


namespace NUMINAMATH_CALUDE_cases_in_2005_l2052_205217

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def caseCount (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearsSinceInitial)

/-- Theorem stating that the number of cases in 2005 is 134,000 --/
theorem cases_in_2005 :
  caseCount 1980 800000 2010 800 2005 = 134000 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_2005_l2052_205217


namespace NUMINAMATH_CALUDE_evaluate_expression_l2052_205224

theorem evaluate_expression : 4^3 - 4 * 4^2 + 6 * 4 - 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2052_205224


namespace NUMINAMATH_CALUDE_fold_square_crease_length_l2052_205271

-- Define the square ABCD
def square_side : ℝ := 8

-- Define point E on AD
def AE : ℝ := 2
def ED : ℝ := 6

-- Define FD as x
def FD : ℝ → ℝ := λ x => x

-- Define CF and EF
def CF (x : ℝ) : ℝ := square_side - x
def EF (x : ℝ) : ℝ := square_side - x

-- State the theorem
theorem fold_square_crease_length :
  ∃ x : ℝ, FD x = 7/4 ∧ CF x = EF x ∧ CF x^2 = FD x^2 + ED^2 := by
  sorry

end NUMINAMATH_CALUDE_fold_square_crease_length_l2052_205271


namespace NUMINAMATH_CALUDE_crabapple_sequences_l2052_205299

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 11

/-- The number of times Mrs. Crabapple teaches per week -/
def classes_per_week : ℕ := 5

/-- The number of different sequences of crabapple recipients in one week -/
def num_sequences : ℕ := num_students ^ classes_per_week

theorem crabapple_sequences :
  num_sequences = 161051 :=
by sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l2052_205299


namespace NUMINAMATH_CALUDE_sqrt_xy_plus_3_l2052_205279

theorem sqrt_xy_plus_3 (x y : ℝ) (h : y = Real.sqrt (1 - 4*x) + Real.sqrt (4*x - 1) + 4) :
  Real.sqrt (x*y + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_xy_plus_3_l2052_205279


namespace NUMINAMATH_CALUDE_at_least_one_geq_one_l2052_205293

theorem at_least_one_geq_one (x y : ℝ) (h : x + y ≥ 2) : max x y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_one_l2052_205293


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2052_205225

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone (r₁ r₂ : ℝ) (hr₁ : r₁ = 25) (hr₂ : r₂ = 7) :
  let h := Real.sqrt ((r₁ + r₂)^2 - (r₁ - r₂)^2)
  (h / 2 : ℝ) = 5 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2052_205225


namespace NUMINAMATH_CALUDE_smallest_class_size_l2052_205214

theorem smallest_class_size (n : ℕ) : n = 274 ↔ 
  n > 0 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 2 ∧ 
  n % 10 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 1 → m % 8 = 2 → m % 10 = 4 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2052_205214


namespace NUMINAMATH_CALUDE_inequality_range_l2052_205274

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ 
  (m > -10 ∧ m ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2052_205274


namespace NUMINAMATH_CALUDE_cheryl_material_calculation_l2052_205242

theorem cheryl_material_calculation (material_used total_bought second_type leftover : ℝ) :
  material_used = 0.21052631578947367 →
  second_type = 2 / 13 →
  leftover = 4 / 26 →
  total_bought = material_used + leftover →
  total_bought = second_type + (0.21052631578947367 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_calculation_l2052_205242


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2052_205209

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 4 < x ∧ x < 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2052_205209


namespace NUMINAMATH_CALUDE_calculation_proof_l2052_205261

theorem calculation_proof : 19 * 0.125 + 281 * (1/8) - 12.5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2052_205261


namespace NUMINAMATH_CALUDE_cupcake_frosting_l2052_205276

theorem cupcake_frosting (cagney_rate lacey_rate lacey_rest total_time : ℕ) :
  cagney_rate = 15 →
  lacey_rate = 25 →
  lacey_rest = 10 →
  total_time = 480 →
  (total_time : ℚ) / ((1 : ℚ) / cagney_rate + (1 : ℚ) / (lacey_rate + lacey_rest)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_frosting_l2052_205276


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_parallel_planes_l2052_205290

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  perpendicular n β → 
  perpendicular_planes α β → 
  perpendicular_lines m n :=
sorry

-- Theorem 2
theorem perpendicular_lines_from_parallel_planes 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel n β → 
  parallel_planes α β → 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_parallel_planes_l2052_205290


namespace NUMINAMATH_CALUDE_sequence_not_convergent_l2052_205298

theorem sequence_not_convergent (x : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∀ k, ∃ n > k, |x n - a| ≥ ε) →
  ¬ (∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - a| < ε) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_convergent_l2052_205298


namespace NUMINAMATH_CALUDE_cab_journey_time_l2052_205269

/-- Given a cab traveling at 5/6 of its usual speed is 8 minutes late, 
    prove that its usual time to cover the journey is 48 minutes. -/
theorem cab_journey_time (usual_time : ℝ) : 
  (5 / 6 : ℝ) * usual_time + 8 = usual_time → usual_time = 48 :=
by sorry

end NUMINAMATH_CALUDE_cab_journey_time_l2052_205269


namespace NUMINAMATH_CALUDE_unique_factor_solution_l2052_205215

theorem unique_factor_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A + B = C - D →
  A = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_factor_solution_l2052_205215


namespace NUMINAMATH_CALUDE_inequality_proof_l2052_205294

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2052_205294


namespace NUMINAMATH_CALUDE_matrix_determinant_l2052_205229

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 0, 3, -1; 5, -1, 2]
  Matrix.det A = 20 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2052_205229


namespace NUMINAMATH_CALUDE_certain_number_calculation_l2052_205277

theorem certain_number_calculation : ∀ (x y : ℕ),
  x + y = 36 →
  x = 19 →
  8 * x + 3 * y = 203 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l2052_205277


namespace NUMINAMATH_CALUDE_lee_lawn_mowing_l2052_205226

/-- The number of lawns Lee mowed last week -/
def num_lawns : ℕ := 16

/-- The price Lee charges for mowing one lawn -/
def price_per_lawn : ℕ := 33

/-- The number of customers who gave Lee a tip -/
def num_tips : ℕ := 3

/-- The amount of each tip -/
def tip_amount : ℕ := 10

/-- Lee's total earnings last week -/
def total_earnings : ℕ := 558

theorem lee_lawn_mowing :
  num_lawns * price_per_lawn + num_tips * tip_amount = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_lee_lawn_mowing_l2052_205226


namespace NUMINAMATH_CALUDE_poll_total_count_l2052_205204

theorem poll_total_count : ∀ (total : ℕ),
  (45 : ℚ) / 100 * total + (8 : ℚ) / 100 * total + (94 : ℕ) = total →
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_poll_total_count_l2052_205204


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2052_205234

/-- A function f is an H-function if for any two distinct real numbers x₁ and x₂,
    x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁ -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function f is strictly increasing if for any two real numbers x₁ and x₂,
    x₁ < x₂ implies f x₁ < f x₂ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2052_205234


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2052_205254

theorem sqrt_expression_equality : (Real.sqrt 5 + 3) * (3 - Real.sqrt 5) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2052_205254


namespace NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l2052_205262

-- Define a quadratic polynomial
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_of_special_quadratic (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c (x^3 + 2*x) ≥ QuadraticPolynomial a b c (x^2 + 3)) →
  (b / a = -4 / 5) :=
by sorry

-- The sum of roots is -b/a, so if b/a = -4/5, then the sum of roots is 4/5

end NUMINAMATH_CALUDE_sum_of_roots_of_special_quadratic_l2052_205262


namespace NUMINAMATH_CALUDE_product_expansion_l2052_205210

theorem product_expansion (x : ℝ) : 
  (7 * x^2 + 3) * (5 * x^3 + 2 * x + 1) = 35 * x^5 + 29 * x^3 + 7 * x^2 + 6 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2052_205210


namespace NUMINAMATH_CALUDE_pairwise_products_sum_l2052_205258

theorem pairwise_products_sum (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a + b + c = 24) : 
  a*b + b*c + c*a = 8 := by
sorry

end NUMINAMATH_CALUDE_pairwise_products_sum_l2052_205258


namespace NUMINAMATH_CALUDE_curve_self_intersection_l2052_205268

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2 - 4, t^3 - 6*t + 3)

-- Define the self-intersection point
def intersection_point : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = intersection_point :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l2052_205268


namespace NUMINAMATH_CALUDE_triangle_problem_l2052_205235

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  -- Conclusions
  B = π / 3 ∧
  a = Real.sqrt 3 ∧
  c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2052_205235


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2052_205232

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2052_205232


namespace NUMINAMATH_CALUDE_consistent_coloring_pattern_l2052_205281

-- Define a hexagonal board
structure HexBoard where
  size : ℕ
  traversal : List ℕ

-- Define the coloring function
def color (n : ℕ) : String :=
  if n % 3 = 0 then "Black"
  else if n % 3 = 1 then "Red"
  else "White"

-- Define a property that checks if two boards have the same coloring pattern
def sameColoringPattern (board1 board2 : HexBoard) : Prop :=
  board1.traversal.map color = board2.traversal.map color

-- Theorem statement
theorem consistent_coloring_pattern 
  (board1 board2 : HexBoard) 
  (h : board1.traversal.length = board2.traversal.length) : 
  sameColoringPattern board1 board2 := by
  sorry

end NUMINAMATH_CALUDE_consistent_coloring_pattern_l2052_205281


namespace NUMINAMATH_CALUDE_number_of_hens_l2052_205233

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 44)
  (h2 : total_feet = 140)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l2052_205233


namespace NUMINAMATH_CALUDE_yoga_studio_total_people_l2052_205245

theorem yoga_studio_total_people :
  let num_men : ℕ := 8
  let num_women : ℕ := 6
  let avg_weight_men : ℝ := 190
  let avg_weight_women : ℝ := 120
  let avg_weight_all : ℝ := 160
  num_men + num_women = 14 := by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_total_people_l2052_205245


namespace NUMINAMATH_CALUDE_number_reciprocal_problem_l2052_205285

theorem number_reciprocal_problem : ∃ x : ℚ, (1 + 1 / x = 5 / 2) ∧ (x = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_number_reciprocal_problem_l2052_205285


namespace NUMINAMATH_CALUDE_constant_term_of_polynomial_product_l2052_205282

theorem constant_term_of_polynomial_product :
  let p : Polynomial ℤ := X^3 + 2*X + 7
  let q : Polynomial ℤ := 2*X^4 + 3*X^2 + 10
  (p * q).coeff 0 = 70 := by sorry

end NUMINAMATH_CALUDE_constant_term_of_polynomial_product_l2052_205282
