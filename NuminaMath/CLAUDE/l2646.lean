import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2646_264610

-- Define the repeating decimals
def repeating_6 : ℚ := 2/3
def repeating_45 : ℚ := 5/11

-- State the theorem
theorem sum_of_repeating_decimals :
  repeating_6 + repeating_45 = 37/33 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2646_264610


namespace NUMINAMATH_CALUDE_average_difference_l2646_264602

theorem average_difference : 
  let set1 := [20, 40, 60]
  let set2 := [10, 70, 16]
  let avg1 := (set1.sum) / (set1.length : ℝ)
  let avg2 := (set2.sum) / (set2.length : ℝ)
  avg1 - avg2 = 8 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2646_264602


namespace NUMINAMATH_CALUDE_red_paint_calculation_l2646_264626

/-- Given a mixture with a ratio of red paint to white paint and a total number of cans,
    calculate the number of cans of red paint required. -/
def red_paint_cans (red_ratio white_ratio total_cans : ℕ) : ℕ :=
  (red_ratio * total_cans) / (red_ratio + white_ratio)

/-- Theorem stating that for a 3:2 ratio of red to white paint and 30 total cans,
    18 cans of red paint are required. -/
theorem red_paint_calculation :
  red_paint_cans 3 2 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_paint_calculation_l2646_264626


namespace NUMINAMATH_CALUDE_backyard_area_l2646_264639

/-- Represents a rectangular backyard with specific walking properties. -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  length_covers_total : length * length_walks = total_distance
  perimeter_covers_total : (2 * length + 2 * width) * perimeter_walks = total_distance

/-- The theorem stating the area of the backyard with given properties. -/
theorem backyard_area (b : Backyard) (h1 : b.total_distance = 2000)
    (h2 : b.length_walks = 50) (h3 : b.perimeter_walks = 20) :
    b.length * b.width = 400 := by
  sorry


end NUMINAMATH_CALUDE_backyard_area_l2646_264639


namespace NUMINAMATH_CALUDE_power_of_product_l2646_264699

theorem power_of_product (x y : ℝ) : (-2 * x^2 * y)^2 = 4 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2646_264699


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2646_264696

/-- Given a cosine function y = a * cos(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 5 and the minimum value is -3, then a = 4. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, a * Real.cos (b * x + c) + d ≤ 5) ∧
  (∀ x, a * Real.cos (b * x + c) + d ≥ -3) ∧
  (∃ x, a * Real.cos (b * x + c) + d = 5) ∧
  (∃ x, a * Real.cos (b * x + c) + d = -3) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2646_264696


namespace NUMINAMATH_CALUDE_even_function_property_l2646_264617

-- Define an even function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x * (x + 1)) :
  ∀ x > 0, f x = x * (x - 1) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l2646_264617


namespace NUMINAMATH_CALUDE_ab_range_l2646_264642

theorem ab_range (a b : ℝ) : 
  a > 0 → b > 0 → a * b = a + b + 3 → a * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l2646_264642


namespace NUMINAMATH_CALUDE_train_cars_problem_l2646_264615

theorem train_cars_problem (total_cars engine_and_caboose passenger_cars cargo_cars : ℕ) :
  total_cars = 71 →
  engine_and_caboose = 2 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + engine_and_caboose →
  passenger_cars = 44 := by
sorry

end NUMINAMATH_CALUDE_train_cars_problem_l2646_264615


namespace NUMINAMATH_CALUDE_sales_difference_l2646_264662

/-- Represents a company selling bottled milk -/
structure Company where
  big_bottle_price : ℝ
  small_bottle_price : ℝ
  big_bottle_discount : ℝ
  small_bottle_discount : ℝ
  big_bottles_sold : ℕ
  small_bottles_sold : ℕ

def tax_rate : ℝ := 0.07

def company_A : Company := {
  big_bottle_price := 4
  small_bottle_price := 2
  big_bottle_discount := 0.1
  small_bottle_discount := 0
  big_bottles_sold := 300
  small_bottles_sold := 400
}

def company_B : Company := {
  big_bottle_price := 3.5
  small_bottle_price := 1.75
  big_bottle_discount := 0
  small_bottle_discount := 0.05
  big_bottles_sold := 350
  small_bottles_sold := 600
}

def calculate_total_sales (c : Company) : ℝ :=
  let big_bottle_revenue := c.big_bottle_price * c.big_bottles_sold
  let small_bottle_revenue := c.small_bottle_price * c.small_bottles_sold
  let total_before_discount := big_bottle_revenue + small_bottle_revenue
  let big_bottle_discount := if c.big_bottles_sold ≥ 10 then c.big_bottle_discount * big_bottle_revenue else 0
  let small_bottle_discount := if c.small_bottles_sold > 20 then c.small_bottle_discount * small_bottle_revenue else 0
  let total_after_discount := total_before_discount - big_bottle_discount - small_bottle_discount
  let total_after_tax := total_after_discount * (1 + tax_rate)
  total_after_tax

theorem sales_difference : 
  calculate_total_sales company_B - calculate_total_sales company_A = 366.475 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_l2646_264662


namespace NUMINAMATH_CALUDE_sphere_radius_from_perpendicular_chords_l2646_264613

/-- Given a sphere with three mutually perpendicular chords APB, CPD, and EPF passing through
    a common point P, where AP = 2a, BP = 2b, CP = 2c, DP = 2d, EP = 2e, and FP = 2f,
    the radius R of the sphere is √(a² + b² + c² + d² + e² + f² - 2ab - 2cd - 2ef). -/
theorem sphere_radius_from_perpendicular_chords
  (a b c d e f : ℝ) : ∃ (R : ℝ),
  R = Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 - 2*a*b - 2*c*d - 2*e*f) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_perpendicular_chords_l2646_264613


namespace NUMINAMATH_CALUDE_parabola_distance_l2646_264647

/-- The parabola y^2 = 8x with focus F and a point M satisfying |MO|^2 = 3|MF| -/
structure Parabola :=
  (F : ℝ × ℝ)
  (M : ℝ × ℝ)
  (h1 : F = (2, 0))
  (h2 : (M.2)^2 = 8 * M.1)
  (h3 : M.1^2 + M.2^2 = 3 * (dist M F))

/-- The distance between M and F is 3 -/
theorem parabola_distance (p : Parabola) : dist p.M p.F = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_distance_l2646_264647


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l2646_264661

theorem daily_wage_of_c (a b c : ℕ) (total_earning : ℕ) : 
  a * 6 + b * 9 + c * 4 = total_earning →
  4 * a = 3 * b →
  5 * a = 3 * c →
  total_earning = 1554 →
  c = 105 := by
  sorry

end NUMINAMATH_CALUDE_daily_wage_of_c_l2646_264661


namespace NUMINAMATH_CALUDE_cost_of_items_l2646_264698

/-- Given the costs of combinations of pencils and pens, prove the cost of one of each item -/
theorem cost_of_items (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 4.10)
  (h2 : 2 * pencil + 3 * pen = 3.70)
  (eraser : ℝ := 0.85) : 
  pencil + pen + eraser = 2.41 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_items_l2646_264698


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_l2646_264685

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_l2646_264685


namespace NUMINAMATH_CALUDE_bear_mass_before_hibernation_l2646_264669

/-- The mass of a bear after hibernation, given as a fraction of its original mass -/
def mass_after_hibernation_fraction : ℚ := 80 / 100

/-- The mass of the bear after hibernation in kilograms -/
def mass_after_hibernation : ℚ := 220

/-- Theorem: If a bear loses 20% of its original mass during hibernation and 
    its mass after hibernation is 220 kg, then its mass before hibernation was 275 kg -/
theorem bear_mass_before_hibernation :
  mass_after_hibernation = mass_after_hibernation_fraction * (275 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_bear_mass_before_hibernation_l2646_264669


namespace NUMINAMATH_CALUDE_delegates_without_badges_l2646_264655

theorem delegates_without_badges (total : ℕ) (preprinted : ℕ) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l2646_264655


namespace NUMINAMATH_CALUDE_gravel_pile_volume_l2646_264692

/-- The volume of a hemispherical pile of gravel -/
theorem gravel_pile_volume (d : ℝ) (h : ℝ) (v : ℝ) : 
  d = 10 → -- diameter is 10 feet
  h = d / 2 → -- height is half the diameter
  v = (250 * Real.pi) / 3 → -- volume is (250π)/3 cubic feet
  v = (2 / 3) * Real.pi * (d / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_gravel_pile_volume_l2646_264692


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2646_264653

theorem add_preserves_inequality (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2646_264653


namespace NUMINAMATH_CALUDE_cone_volume_from_sector_l2646_264641

/-- The volume of a right circular cone formed by rolling up a two-third sector of a circle -/
theorem cone_volume_from_sector (r : ℝ) (h : r = 6) :
  let sector_angle : ℝ := 2 * π * (2/3)
  let base_circumference : ℝ := sector_angle * r / (2 * π)
  let base_radius : ℝ := base_circumference / (2 * π)
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = (32/3) * π * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_sector_l2646_264641


namespace NUMINAMATH_CALUDE_cos_inequality_l2646_264601

theorem cos_inequality (ε x y : Real) : 
  ε > 0 → 
  x ∈ Set.Ioo (-π/4) (π/4) → 
  y ∈ Set.Ioo (-π/4) (π/4) → 
  Real.exp (x + ε) * Real.sin y = Real.exp y * Real.sin x → 
  Real.cos x ≤ Real.cos y := by
  sorry

end NUMINAMATH_CALUDE_cos_inequality_l2646_264601


namespace NUMINAMATH_CALUDE_division_problem_l2646_264695

theorem division_problem (x y : ℕ+) (h1 : x = 10 * y + 3) (h2 : 2 * x = 21 * y + 1) : 
  11 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2646_264695


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l2646_264646

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  mesh_AB : A.teeth * A.speed = B.teeth * B.speed
  mesh_BC : B.teeth * B.speed = C.teeth * C.speed
  mesh_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth)) ∧
    (sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth)) :=
by
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l2646_264646


namespace NUMINAMATH_CALUDE_space_division_by_five_spheres_l2646_264623

/-- Maximum number of regions into which a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => a (n + 2) + 2 * (n + 2)

/-- Maximum number of regions into which space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + a (n + 2)

theorem space_division_by_five_spheres :
  b 5 = 30 := by sorry

end NUMINAMATH_CALUDE_space_division_by_five_spheres_l2646_264623


namespace NUMINAMATH_CALUDE_milk_problem_l2646_264666

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l2646_264666


namespace NUMINAMATH_CALUDE_g_monotone_and_range_l2646_264630

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (2^x + b) / (2^x - b)

theorem g_monotone_and_range (b : ℝ) :
  (b < 0 → ∀ x y : ℝ, x < y → g b x < g b y) ∧
  (b = -1 → ∀ a : ℝ, (∀ x : ℝ, g (-1) (x^2 + 1) + g (-1) (3 - a*x) > 0) ↔ -4 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_g_monotone_and_range_l2646_264630


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l2646_264614

/-- Represents a seating arrangement for two families in two cars -/
structure SeatingArrangement where
  audi : Finset (Fin 6)
  jetta : Finset (Fin 6)

/-- The set of all valid seating arrangements -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of adults in the group -/
def numAdults : Nat := 4

/-- The number of children in the group -/
def numChildren : Nat := 2

/-- The maximum capacity of each car -/
def maxCapacity : Nat := 4

/-- Theorem stating the number of valid seating arrangements -/
theorem count_valid_arrangements :
  Finset.card validArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l2646_264614


namespace NUMINAMATH_CALUDE_max_product_xy_l2646_264691

theorem max_product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (eq1 : x + 1/y = 3) (eq2 : y + 2/x = 3) :
  ∃ (C : ℝ), C = x*y ∧ C ≤ 3 + Real.sqrt 7 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 1/y' = 3 ∧ y' + 2/x' = 3 ∧ x'*y' = 3 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_product_xy_l2646_264691


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l2646_264621

-- Define the arithmetic sequences
def arithmetic_seq (a₁ d : ℕ+) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_divisibility 
  (a₁ d_a b₁ d_b : ℕ+) 
  (h : ∃ (S : Set (ℕ × ℕ)), S.Infinite ∧ 
    ∀ (i j : ℕ), (i, j) ∈ S → 
      i ≤ j ∧ j ≤ i + 2021 ∧ 
      (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j)) :
  ∀ i : ℕ, ∃ j : ℕ, (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l2646_264621


namespace NUMINAMATH_CALUDE_gunther_dusting_time_l2646_264636

/-- Represents the time in minutes for Gunther's cleaning tasks -/
structure CleaningTime where
  vacuuming : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  total_free_time : ℕ
  remaining_free_time : ℕ

/-- Calculates the time spent dusting furniture -/
def dusting_time (ct : CleaningTime) : ℕ :=
  ct.total_free_time - ct.remaining_free_time - 
  (ct.vacuuming + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther spends 60 minutes dusting furniture -/
theorem gunther_dusting_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    total_free_time := 3 * 60,
    remaining_free_time := 30
  }
  dusting_time ct = 60 := by
  sorry

end NUMINAMATH_CALUDE_gunther_dusting_time_l2646_264636


namespace NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l2646_264682

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ 
   a * x^2 - (a + 3) * x + 2 = 0 ∧
   a * y^2 - (a + 3) * y + 2 = 0) ↔ 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_opposite_signs_l2646_264682


namespace NUMINAMATH_CALUDE_intersection_A_B_l2646_264651

-- Define set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2646_264651


namespace NUMINAMATH_CALUDE_squirrel_count_l2646_264645

theorem squirrel_count (total_acorns : ℕ) (needed_acorns : ℕ) (shortage : ℕ) : 
  total_acorns = 575 →
  needed_acorns = 130 →
  shortage = 15 →
  (total_acorns / (needed_acorns - shortage) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_l2646_264645


namespace NUMINAMATH_CALUDE_expression_evaluation_l2646_264628

theorem expression_evaluation : -20 + 12 * (8 / 4) * 3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2646_264628


namespace NUMINAMATH_CALUDE_expression_value_l2646_264665

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2646_264665


namespace NUMINAMATH_CALUDE_tin_in_mixed_alloy_tin_amount_in_new_alloy_l2646_264664

/-- Amount of tin in a mixture of two alloys -/
theorem tin_in_mixed_alloy (mass_A mass_B : ℝ) 
  (lead_tin_ratio_A : ℝ) (tin_copper_ratio_B : ℝ) : ℝ :=
  let tin_fraction_A := lead_tin_ratio_A / (1 + lead_tin_ratio_A)
  let tin_fraction_B := tin_copper_ratio_B / (1 + tin_copper_ratio_B)
  tin_fraction_A * mass_A + tin_fraction_B * mass_B

/-- The amount of tin in the new alloy is 221.25 kg -/
theorem tin_amount_in_new_alloy : 
  tin_in_mixed_alloy 170 250 (1/3) (3/5) = 221.25 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_mixed_alloy_tin_amount_in_new_alloy_l2646_264664


namespace NUMINAMATH_CALUDE_pool_water_calculation_l2646_264635

/-- Calculates the amount of water in a pool after five hours of filling and a leak -/
def water_in_pool (rate1 : ℕ) (rate2 : ℕ) (rate3 : ℕ) (leak : ℕ) : ℕ :=
  rate1 + 2 * rate2 + rate3 - leak

theorem pool_water_calculation :
  water_in_pool 8 10 14 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_calculation_l2646_264635


namespace NUMINAMATH_CALUDE_symmetric_line_l2646_264675

/-- Given a line with equation 2x - y + 3 = 0 and a fixed point M(-1, 2),
    the equation of the line symmetric to the given line with respect to M is 2x - y + 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∀ x y, 2*x - y + 3 = 0 → 2*x - y + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_l2646_264675


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l2646_264657

theorem no_multiple_of_five_2c4 :
  ∀ C : ℕ, C < 10 → ¬(∃ k : ℕ, 200 + 10 * C + 4 = 5 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_2c4_l2646_264657


namespace NUMINAMATH_CALUDE_Q_roots_l2646_264694

def Q (x : ℝ) : ℝ := x^6 - 5*x^5 - 12*x^3 - x + 16

theorem Q_roots :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x = 0) := by
sorry

end NUMINAMATH_CALUDE_Q_roots_l2646_264694


namespace NUMINAMATH_CALUDE_sandy_change_l2646_264608

def pants_cost : Float := 13.58
def shirt_cost : Float := 10.29
def sweater_cost : Float := 24.97
def shoes_cost : Float := 39.99
def paid_amount : Float := 100.00

def total_cost : Float := pants_cost + shirt_cost + sweater_cost + shoes_cost

theorem sandy_change : paid_amount - total_cost = 11.17 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l2646_264608


namespace NUMINAMATH_CALUDE_log_gt_x_squared_over_one_plus_x_l2646_264634

theorem log_gt_x_squared_over_one_plus_x :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, 0 < x → x < a → Real.log (1 + x) > x^2 / (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_gt_x_squared_over_one_plus_x_l2646_264634


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l2646_264672

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l2646_264672


namespace NUMINAMATH_CALUDE_number_added_after_doubling_l2646_264638

theorem number_added_after_doubling (x y : ℝ) : x = 4 → 3 * (2 * x + y) = 51 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_added_after_doubling_l2646_264638


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l2646_264681

-- Define the slopes and y-intercept
def m₁ : ℝ := 8
def m₂ : ℝ := 4
def c : ℝ := 0  -- y-intercept, defined as non-zero in the theorem

-- Define the x-intercepts
def u : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem
def v : ℝ := 0  -- actual value doesn't matter, will be constrained in the theorem

-- Theorem statement
theorem x_intercept_ratio (h₁ : c ≠ 0) 
                          (h₂ : m₁ * u + c = 0) 
                          (h₃ : m₂ * v + c = 0) : 
  u / v = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l2646_264681


namespace NUMINAMATH_CALUDE_a_17_value_l2646_264625

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem a_17_value (a : ℕ → ℝ) (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) (h_a2a8 : a 2 * a 8 = 16) : a 17 = 256 := by
  sorry

end NUMINAMATH_CALUDE_a_17_value_l2646_264625


namespace NUMINAMATH_CALUDE_mad_hatter_winning_condition_l2646_264605

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoterFractions where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the additional fraction of undecided voters each candidate receives -/
structure UndecidedAllocation where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

/-- The minimum fraction of undecided voters the Mad Hatter needs to secure -/
def minimum_fraction_for_mad_hatter (v : VoterFractions) : ℝ :=
  0.7

theorem mad_hatter_winning_condition 
  (v : VoterFractions)
  (h1 : v.mad_hatter = 0.2)
  (h2 : v.march_hare = 0.25)
  (h3 : v.dormouse = 0.3)
  (h4 : v.undecided = 1 - (v.mad_hatter + v.march_hare + v.dormouse))
  (h5 : v.mad_hatter + v.march_hare + v.dormouse + v.undecided = 1) :
  ∀ (u : UndecidedAllocation),
    (u.mad_hatter + u.march_hare + u.dormouse = 1) →
    (u.mad_hatter ≥ minimum_fraction_for_mad_hatter v) →
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.march_hare + v.undecided * u.march_hare) ∧
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.dormouse + v.undecided * u.dormouse) :=
sorry

end NUMINAMATH_CALUDE_mad_hatter_winning_condition_l2646_264605


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2646_264663

theorem cosine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (∃ x1, a * Real.cos (b * x1 + c) + d = 5) →
  (∃ x2, a * Real.cos (b * x2 + c) + d = -3) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2646_264663


namespace NUMINAMATH_CALUDE_encyclopedia_pages_l2646_264667

/-- The Encyclopedia of Life and Everything Else --/
structure Encyclopedia where
  chapters : Nat
  pages_per_chapter : Nat

/-- Calculate the total number of pages in the encyclopedia --/
def total_pages (e : Encyclopedia) : Nat :=
  e.chapters * e.pages_per_chapter

/-- Theorem: The encyclopedia has 9384 pages in total --/
theorem encyclopedia_pages :
  ∃ (e : Encyclopedia), e.chapters = 12 ∧ e.pages_per_chapter = 782 ∧ total_pages e = 9384 := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_pages_l2646_264667


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2646_264629

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = (3 * Real.sqrt 10) / 10 ∧ 
  Real.cos (2 * β) = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2646_264629


namespace NUMINAMATH_CALUDE_green_hat_cost_l2646_264624

/-- Proves that the cost of each green hat is $1 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 600 →
  green_hats = 90 →
  ∃ (green_hat_cost : ℕ), green_hat_cost = 1 ∧
    total_price = blue_hat_cost * (total_hats - green_hats) + green_hat_cost * green_hats :=
by
  sorry


end NUMINAMATH_CALUDE_green_hat_cost_l2646_264624


namespace NUMINAMATH_CALUDE_expression_simplification_l2646_264654

theorem expression_simplification (x y : ℝ) (hx : x = 4) (hy : y = -2) :
  1 - (x - y) / (x + 2*y) / ((x^2 - y^2) / (x^2 + 4*x*y + 4*y^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2646_264654


namespace NUMINAMATH_CALUDE_tablet_value_proof_compensation_for_m_days_l2646_264627

-- Define the total days of internship
def total_days : ℕ := 30

-- Define the cash compensation for full internship
def full_cash_compensation : ℕ := 1500

-- Define the number of days Xiaomin worked
def worked_days : ℕ := 20

-- Define the cash compensation Xiaomin received
def received_cash_compensation : ℕ := 300

-- Define the value of the M type tablet
def tablet_value : ℕ := 2100

-- Define the daily compensation rate
def daily_rate : ℚ := 120

-- Theorem for the value of the M type tablet
theorem tablet_value_proof :
  (worked_days : ℚ) / total_days * (tablet_value + full_cash_compensation) =
  tablet_value + received_cash_compensation :=
sorry

-- Theorem for the compensation for m days of work
theorem compensation_for_m_days (m : ℕ) :
  (m : ℚ) * daily_rate = (m : ℚ) * ((tablet_value + full_cash_compensation) / total_days) :=
sorry

end NUMINAMATH_CALUDE_tablet_value_proof_compensation_for_m_days_l2646_264627


namespace NUMINAMATH_CALUDE_total_materials_ordered_l2646_264648

-- Define the amounts of materials ordered
def concrete : Real := 0.17
def bricks : Real := 0.237
def sand : Real := 0.646
def stone : Real := 0.5
def steel : Real := 1.73
def wood : Real := 0.894

-- Theorem statement
theorem total_materials_ordered :
  concrete + bricks + sand + stone + steel + wood = 4.177 := by
  sorry

end NUMINAMATH_CALUDE_total_materials_ordered_l2646_264648


namespace NUMINAMATH_CALUDE_taxi_fare_theorem_l2646_264670

/-- Taxi fare function for distances greater than 5 kilometers -/
def taxi_fare (x : ℝ) : ℝ :=
  10 + 2 * 1.3 + 2.4 * (x - 5)

/-- Theorem stating the taxi fare function and its value for 6 kilometers -/
theorem taxi_fare_theorem (x : ℝ) (h : x > 5) :
  taxi_fare x = 2.4 * x + 0.6 ∧ taxi_fare 6 = 15 := by
  sorry

#check taxi_fare_theorem

end NUMINAMATH_CALUDE_taxi_fare_theorem_l2646_264670


namespace NUMINAMATH_CALUDE_product_mod_25_l2646_264607

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (123 * 156 * 198) % 25 = m ∧ m = 24 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2646_264607


namespace NUMINAMATH_CALUDE_cost_of_groceries_l2646_264659

/-- The cost of groceries problem -/
theorem cost_of_groceries
  (mango_cost : ℝ → ℝ)  -- Cost function for mangos (kg → $)
  (rice_cost : ℝ → ℝ)   -- Cost function for rice (kg → $)
  (flour_cost : ℝ → ℝ)  -- Cost function for flour (kg → $)
  (h1 : mango_cost 10 = rice_cost 10)  -- 10 kg mangos cost same as 10 kg rice
  (h2 : flour_cost 6 = rice_cost 2)    -- 6 kg flour costs same as 2 kg rice
  (h3 : ∀ x, flour_cost x = 21 * x)    -- Flour costs $21 per kg
  : mango_cost 4 + rice_cost 3 + flour_cost 5 = 546 := by
  sorry

#check cost_of_groceries

end NUMINAMATH_CALUDE_cost_of_groceries_l2646_264659


namespace NUMINAMATH_CALUDE_only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l2646_264616

/-- Represents a survey option -/
inductive SurveyOption
  | ClassmateExercise
  | CarCrashResistance
  | GalaViewership
  | ShoeSoleBending

/-- Defines the characteristics of a comprehensive investigation -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.ClassmateExercise => true
  | _ => false

/-- Theorem stating that only the classmate exercise survey is comprehensive -/
theorem only_classmate_exercise_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.ClassmateExercise :=
by sorry

/-- Main theorem proving which survey is suitable for a comprehensive investigation -/
theorem comprehensive_investigation_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end NUMINAMATH_CALUDE_only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l2646_264616


namespace NUMINAMATH_CALUDE_age_condition_amount_per_year_is_five_l2646_264656

/-- Mikail's age on his birthday -/
def age : ℕ := 9

/-- The total amount Mikail receives on his birthday -/
def total_amount : ℕ := 45

/-- The condition that Mikail's age is 3 times as old as he was when he was three -/
theorem age_condition : age = 3 * 3 := by sorry

/-- The amount Mikail receives per year of his age -/
def amount_per_year : ℚ := total_amount / age

/-- Proof that the amount Mikail receives per year is $5 -/
theorem amount_per_year_is_five : amount_per_year = 5 := by sorry

end NUMINAMATH_CALUDE_age_condition_amount_per_year_is_five_l2646_264656


namespace NUMINAMATH_CALUDE_unique_modular_inverse_l2646_264640

theorem unique_modular_inverse (p : Nat) (a : Nat) (h_p : p.Prime) (h_p_odd : p % 2 = 1)
  (h_a_range : 2 ≤ a ∧ a ≤ p - 2) :
  ∃! i : Nat, 2 ≤ i ∧ i ≤ p - 2 ∧ i ≠ a ∧ (i * a) % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_inverse_l2646_264640


namespace NUMINAMATH_CALUDE_line_conditions_vector_at_zero_l2646_264686

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
theorem line_conditions :
  line_param 1 = (2, 5) ∧ line_param 4 = (11, -7) := sorry

-- Theorem to prove
theorem vector_at_zero :
  line_param 0 = (-1, 9) := by sorry

end NUMINAMATH_CALUDE_line_conditions_vector_at_zero_l2646_264686


namespace NUMINAMATH_CALUDE_smallest_fraction_l2646_264678

theorem smallest_fraction (x : ℝ) (h : x = 7) : 
  6 / (x + 1) < 6 / x ∧ 
  6 / (x + 1) < 6 / (x - 1) ∧ 
  6 / (x + 1) < x / 6 ∧ 
  6 / (x + 1) < (x + 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_l2646_264678


namespace NUMINAMATH_CALUDE_prob_more_ones_than_eights_l2646_264609

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The total number of possible outcomes when rolling numDice dice with numSides sides -/
def totalOutcomes : ℕ := numSides ^ numDice

/-- The number of ways to roll an equal number of 1's and 8's -/
def equalOnesAndEights : ℕ := 12276

/-- The probability of rolling more 1's than 8's when rolling numDice fair dice with numSides sides -/
def probMoreOnesThanEights : ℚ := 10246 / 32768

theorem prob_more_ones_than_eights :
  probMoreOnesThanEights = 1/2 * (1 - equalOnesAndEights / totalOutcomes) :=
sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_eights_l2646_264609


namespace NUMINAMATH_CALUDE_triangle_properties_l2646_264674

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2646_264674


namespace NUMINAMATH_CALUDE_ln_b_over_a_range_l2646_264644

theorem ln_b_over_a_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (1 : ℝ) / Real.exp 1 ≤ c / a) (h2 : c / a ≤ 2)
  (h3 : c * Real.log b = a + c * Real.log c) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1 - 1) ∧ Real.log (b / a) = x :=
sorry

end NUMINAMATH_CALUDE_ln_b_over_a_range_l2646_264644


namespace NUMINAMATH_CALUDE_fish_cost_is_80_l2646_264633

/-- The cost of fish in pesos per kilogram -/
def fish_cost : ℕ := 80

/-- The cost of pork in pesos per kilogram -/
def pork_cost : ℕ := 105

/-- Theorem stating that the cost of fish is 80 pesos per kilogram -/
theorem fish_cost_is_80 :
  (530 = 4 * fish_cost + 2 * pork_cost) →
  (875 = 7 * fish_cost + 3 * pork_cost) →
  fish_cost = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_cost_is_80_l2646_264633


namespace NUMINAMATH_CALUDE_mr_johnson_class_size_l2646_264676

def mrs_finley_class : ℕ := 24

def mr_johnson_class : ℕ := (mrs_finley_class / 2) + 10

theorem mr_johnson_class_size : mr_johnson_class = 22 := by
  sorry

end NUMINAMATH_CALUDE_mr_johnson_class_size_l2646_264676


namespace NUMINAMATH_CALUDE_project_completion_time_l2646_264649

/-- The number of days it takes for person A to complete the project alone -/
def days_A : ℝ := 20

/-- The number of days it takes for person B to complete the project alone -/
def days_B : ℝ := 40

/-- The total duration of the project when A and B work together, and A quits 10 days before completion -/
def total_days : ℝ := 20

/-- The number of days A works before quitting -/
def days_A_works : ℝ := total_days - 10

theorem project_completion_time :
  (days_A_works * (1 / days_A + 1 / days_B)) + (10 * (1 / days_B)) = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l2646_264649


namespace NUMINAMATH_CALUDE_rectangle_area_l2646_264622

/-- Given a rectangle with diagonal length x and length three times its width, 
    the area of the rectangle is 3x^2/10 -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w : ℝ), w > 0 ∧ x^2 = (3*w)^2 + w^2 → 3*w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2646_264622


namespace NUMINAMATH_CALUDE_pie_remainder_l2646_264684

theorem pie_remainder (whole_pie : ℝ) (carlos_share : ℝ) (maria_share : ℝ) : 
  carlos_share = 0.8 * whole_pie →
  maria_share = 0.25 * (whole_pie - carlos_share) →
  whole_pie - carlos_share - maria_share = 0.15 * whole_pie :=
by sorry

end NUMINAMATH_CALUDE_pie_remainder_l2646_264684


namespace NUMINAMATH_CALUDE_product_xy_value_l2646_264603

/-- A parallelogram EFGH with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram -/
def product_xy (p : Parallelogram) (x y : ℝ) : ℝ := x * y

/-- Theorem: The product of x and y in the given parallelogram is 18 * ∛4 -/
theorem product_xy_value (p : Parallelogram) 
  (h1 : p.EF = 110)
  (h2 : p.FG = fun y => 16 * y^3)
  (h3 : p.GH = fun x => 6 * x + 2)
  (h4 : p.HE = 64)
  : ∃ x y, product_xy p x y = 18 * (4 ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_product_xy_value_l2646_264603


namespace NUMINAMATH_CALUDE_fraction_inequality_l2646_264637

theorem fraction_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : 
  a / b + b / a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2646_264637


namespace NUMINAMATH_CALUDE_pen_calculation_l2646_264618

theorem pen_calculation (x y z : ℕ) (hx : x = 5) (hy : y = 20) (hz : z = 19) :
  2 * (x + y) - z = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_calculation_l2646_264618


namespace NUMINAMATH_CALUDE_equality_of_fractions_l2646_264679

theorem equality_of_fractions (x y z k : ℝ) : 
  (5 / (x + y) = k / (x + z)) ∧ (k / (x + z) = 9 / (z - y)) → k = 14 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l2646_264679


namespace NUMINAMATH_CALUDE_max_cross_pattern_sum_l2646_264650

/-- Represents the cross-shaped pattern -/
structure CrossPattern where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 6, 9, 11, 14}

/-- Checks if the pattern satisfies the sum conditions -/
def isValidPattern (p : CrossPattern) : Prop :=
  p.a + p.b + p.e = p.a + p.c + p.e ∧
  p.a + p.c + p.e = p.b + p.d + p.e ∧
  p.a + p.d = p.b + p.c

/-- Checks if the pattern uses all available numbers exactly once -/
def usesAllNumbers (p : CrossPattern) : Prop :=
  {p.a, p.b, p.c, p.d, p.e} = availableNumbers

/-- The sum of any row, column, or diagonal in a valid pattern -/
def patternSum (p : CrossPattern) : ℕ := p.a + p.b + p.e

/-- Theorem: The maximum sum in a valid cross pattern is 31 -/
theorem max_cross_pattern_sum :
  ∀ p : CrossPattern,
    isValidPattern p →
    usesAllNumbers p →
    patternSum p ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_cross_pattern_sum_l2646_264650


namespace NUMINAMATH_CALUDE_divisors_of_m_squared_l2646_264604

def m : ℕ := 2^42 * 3^26 * 5^12

theorem divisors_of_m_squared (d : ℕ) : 
  (d ∣ m^2) ∧ (d < m) ∧ ¬(d ∣ m) → 
  (Finset.filter (λ x => (x ∣ m^2) ∧ (x < m) ∧ ¬(x ∣ m)) (Finset.range (m + 1))).card = 38818 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_m_squared_l2646_264604


namespace NUMINAMATH_CALUDE_remainder_problem_l2646_264673

theorem remainder_problem : 123456789012 % 112 = 76 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2646_264673


namespace NUMINAMATH_CALUDE_fraction_equality_l2646_264668

theorem fraction_equality : ∃ x : ℚ, x * (7/8 * 1/3) = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2646_264668


namespace NUMINAMATH_CALUDE_sqrt_8_simplification_l2646_264620

theorem sqrt_8_simplification :
  Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_8_simplification_l2646_264620


namespace NUMINAMATH_CALUDE_marys_remaining_money_l2646_264619

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that Mary's remaining money is 50 - 12p -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 12 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_l2646_264619


namespace NUMINAMATH_CALUDE_return_trip_time_l2646_264693

/-- The time taken for a return trip given the conditions of the original journey -/
theorem return_trip_time 
  (total_distance : ℝ) 
  (uphill_speed downhill_speed : ℝ)
  (forward_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : uphill_speed = 4)
  (h3 : downhill_speed = 6)
  (h4 : forward_time = 4.25)
  (h5 : ∃ (uphill_distance downhill_distance : ℝ), 
    uphill_distance + downhill_distance = total_distance ∧
    uphill_distance / uphill_speed + downhill_distance / downhill_speed = forward_time) :
  ∃ (return_time : ℝ), return_time = 4.5 := by
sorry

end NUMINAMATH_CALUDE_return_trip_time_l2646_264693


namespace NUMINAMATH_CALUDE_wednesday_occurs_five_times_l2646_264683

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Properties of December in year M -/
structure DecemberProperties :=
  (sundays : List Date)
  (hasFiveSundays : sundays.length = 5)
  (has31Days : Nat)

/-- Properties of January in year M+1 -/
structure JanuaryProperties :=
  (firstDay : DayOfWeek)
  (has31Days : Nat)

/-- Function to determine the number of occurrences of a day in January -/
def countOccurrencesInJanuary (day : DayOfWeek) (january : JanuaryProperties) : Nat :=
  sorry

/-- Main theorem -/
theorem wednesday_occurs_five_times
  (december : DecemberProperties)
  (january : JanuaryProperties)
  : countOccurrencesInJanuary DayOfWeek.Wednesday january = 5 :=
sorry

end NUMINAMATH_CALUDE_wednesday_occurs_five_times_l2646_264683


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l2646_264689

/-- The motion equation of a ball rolling down an inclined plane -/
def motion_equation (t : ℝ) : ℝ := t^2

/-- The velocity function derived from the motion equation -/
def velocity (t : ℝ) : ℝ := 2 * t

theorem instantaneous_velocity_at_5 : velocity 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l2646_264689


namespace NUMINAMATH_CALUDE_right_triangle_solution_l2646_264631

theorem right_triangle_solution :
  ∃ (x : ℝ), x > 0 ∧
  (4 * x + 2) > 0 ∧
  ((x - 3)^2) > 0 ∧
  (5 * x + 1) > 0 ∧
  (4 * x + 2)^2 + (x - 3)^4 = (5 * x + 1)^2 ∧
  x = Real.sqrt (3/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l2646_264631


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2646_264677

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x < 4) ↔ (x - a < 1 ∧ x + b > 2)) →
  b - a = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2646_264677


namespace NUMINAMATH_CALUDE_vicente_spent_25_dollars_l2646_264658

-- Define the quantities and prices
def rice_kg : ℕ := 5
def meat_lb : ℕ := 3
def rice_price_per_kg : ℕ := 2
def meat_price_per_lb : ℕ := 5

-- Define the total cost function
def total_cost (rice_kg meat_lb rice_price_per_kg meat_price_per_lb : ℕ) : ℕ :=
  rice_kg * rice_price_per_kg + meat_lb * meat_price_per_lb

-- Theorem statement
theorem vicente_spent_25_dollars :
  total_cost rice_kg meat_lb rice_price_per_kg meat_price_per_lb = 25 := by
  sorry

end NUMINAMATH_CALUDE_vicente_spent_25_dollars_l2646_264658


namespace NUMINAMATH_CALUDE_last_digit_is_square_of_second_l2646_264652

/-- Represents a 4-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : d1 ≠ 0 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10

/-- The specific 4-digit number 1349 -/
def number : FourDigitNumber where
  d1 := 1
  d2 := 3
  d3 := 4
  d4 := 9
  is_four_digit := by sorry

theorem last_digit_is_square_of_second :
  (number.d1 = number.d2 / 3) →
  (number.d3 = number.d1 + number.d2) →
  (number.d4 = number.d2 * number.d2) := by sorry

end NUMINAMATH_CALUDE_last_digit_is_square_of_second_l2646_264652


namespace NUMINAMATH_CALUDE_max_value_operation_l2646_264611

theorem max_value_operation : 
  ∃ (max : ℕ), max = 600 ∧ 
  (∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) ≤ max) ∧
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 3 * (300 - n) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_operation_l2646_264611


namespace NUMINAMATH_CALUDE_spice_jar_cost_is_six_l2646_264671

/-- Represents the cost and point structure for Martha's grocery shopping -/
structure GroceryShopping where
  pointsPerTenDollars : ℕ
  bonusThreshold : ℕ
  bonusPoints : ℕ
  beefPounds : ℕ
  beefPricePerPound : ℕ
  fruitVegPounds : ℕ
  fruitVegPricePerPound : ℕ
  spiceJars : ℕ
  otherGroceriesCost : ℕ
  totalPoints : ℕ

/-- Calculates the cost of each jar of spices based on the given shopping information -/
def calculateSpiceJarCost (shopping : GroceryShopping) : ℕ :=
  sorry

/-- Theorem stating that the cost of each jar of spices is $6 -/
theorem spice_jar_cost_is_six (shopping : GroceryShopping) 
  (h1 : shopping.pointsPerTenDollars = 50)
  (h2 : shopping.bonusThreshold = 100)
  (h3 : shopping.bonusPoints = 250)
  (h4 : shopping.beefPounds = 3)
  (h5 : shopping.beefPricePerPound = 11)
  (h6 : shopping.fruitVegPounds = 8)
  (h7 : shopping.fruitVegPricePerPound = 4)
  (h8 : shopping.spiceJars = 3)
  (h9 : shopping.otherGroceriesCost = 37)
  (h10 : shopping.totalPoints = 850) :
  calculateSpiceJarCost shopping = 6 :=
  sorry


end NUMINAMATH_CALUDE_spice_jar_cost_is_six_l2646_264671


namespace NUMINAMATH_CALUDE_find_N_l2646_264660

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.6 * 2500) ∧ (N = 7500) := by
  sorry

end NUMINAMATH_CALUDE_find_N_l2646_264660


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_inequality_l2646_264606

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_inequality_l2646_264606


namespace NUMINAMATH_CALUDE_final_temp_is_50_l2646_264600

/-- Represents the thermal equilibrium problem with two metal bars and water. -/
structure ThermalEquilibrium where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ

/-- Calculates the final equilibrium temperature after adding the second metal bar. -/
def finalEquilibriumTemp (te : ThermalEquilibrium) : ℝ :=
  sorry

/-- Theorem stating that the final equilibrium temperature is 50°C. -/
theorem final_temp_is_50 (te : ThermalEquilibrium)
    (h1 : te.initialWaterTemp = 80)
    (h2 : te.initialBarTemp = 20)
    (h3 : te.firstEquilibriumTemp = 60) :
  finalEquilibriumTemp te = 50 :=
by sorry

end NUMINAMATH_CALUDE_final_temp_is_50_l2646_264600


namespace NUMINAMATH_CALUDE_scorpion_millipedes_l2646_264687

/-- Calculates the number of millipedes needed to reach a daily segment goal -/
def millipedes_needed (daily_requirement : ℕ) (eaten_segments : ℕ) (remaining_millipede_segments : ℕ) : ℕ :=
  (daily_requirement - eaten_segments) / remaining_millipede_segments

theorem scorpion_millipedes :
  let daily_requirement : ℕ := 800
  let first_millipede_segments : ℕ := 60
  let long_millipede_segments : ℕ := 2 * first_millipede_segments
  let eaten_segments : ℕ := first_millipede_segments + 2 * long_millipede_segments
  let remaining_millipede_segments : ℕ := 50
  millipedes_needed daily_requirement eaten_segments remaining_millipede_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_millipedes_l2646_264687


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2646_264690

theorem age_ratio_proof (b_age : ℕ) (a_age : ℕ) : 
  b_age = 39 →
  a_age = b_age + 9 →
  (a_age + 10) / (b_age - 10) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2646_264690


namespace NUMINAMATH_CALUDE_triple_solution_l2646_264688

theorem triple_solution (k : ℕ) (hk : k > 0) :
  ∀ a b c : ℕ, 
    a > 0 → b > 0 → c > 0 →
    a + b + c = 3 * k + 1 →
    a * b + b * c + c * a = 3 * k^2 + 2 * k →
    (a = k + 1 ∧ b = k ∧ c = k) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l2646_264688


namespace NUMINAMATH_CALUDE_cakes_sold_minus_bought_l2646_264632

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 274. -/
theorem cakes_sold_minus_bought (initial : ℕ) (sold : ℕ) (bought : ℕ) 
    (h1 : initial = 648) 
    (h2 : sold = 467) 
    (h3 : bought = 193) : 
    sold - bought = 274 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_minus_bought_l2646_264632


namespace NUMINAMATH_CALUDE_zachs_babysitting_pay_rate_l2646_264680

/-- The problem of calculating Zach's babysitting pay rate -/
theorem zachs_babysitting_pay_rate 
  (bike_cost : ℚ)
  (weekly_allowance : ℚ)
  (lawn_mowing_pay : ℚ)
  (current_savings : ℚ)
  (additional_needed : ℚ)
  (babysitting_hours : ℚ)
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : lawn_mowing_pay = 10)
  (h4 : current_savings = 65)
  (h5 : additional_needed = 6)
  (h6 : babysitting_hours = 2)
  : ∃ (babysitting_rate : ℚ), 
    babysitting_rate = (current_savings + weekly_allowance + lawn_mowing_pay + additional_needed - bike_cost) / babysitting_hours ∧ 
    babysitting_rate = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_zachs_babysitting_pay_rate_l2646_264680


namespace NUMINAMATH_CALUDE_min_values_a_b_l2646_264697

theorem min_values_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 2 * a + b + 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a * b ≤ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 6 + 4 * Real.sqrt 2 ∧
  a + 2 * b = 4 * Real.sqrt 2 + 5 :=
by sorry

end NUMINAMATH_CALUDE_min_values_a_b_l2646_264697


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2646_264643

theorem sum_of_fractions : (3 / 100 : ℚ) + (5 / 1000 : ℚ) + (7 / 10000 : ℚ) = (357 / 10000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2646_264643


namespace NUMINAMATH_CALUDE_characterize_N_l2646_264612

def StrictlyIncreasing (s : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → s i < s j

def IsPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ ∀ i : ℕ, a (i + m) = a i

def SatisfiesConditions (s : ℕ → ℕ) (N : ℕ) : Prop :=
  StrictlyIncreasing s ∧
  IsPeriodic (fun i => s (i + 1) - s i) ∧
  ∀ n : ℕ, n > 0 → s (s n) - s (s (n - 1)) ≤ N ∧ N < s (1 + s n) - s (s (n - 1))

theorem characterize_N :
  ∀ N : ℕ, (∃ s : ℕ → ℕ, SatisfiesConditions s N) ↔
    (∃ k : ℕ, k > 0 ∧ k^2 ≤ N ∧ N < k^2 + k) :=
by sorry

end NUMINAMATH_CALUDE_characterize_N_l2646_264612
