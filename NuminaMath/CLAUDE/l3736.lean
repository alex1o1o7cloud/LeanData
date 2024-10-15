import Mathlib

namespace NUMINAMATH_CALUDE_jerrys_average_score_l3736_373626

theorem jerrys_average_score (A : ℝ) : 
  (∀ (new_average : ℝ), new_average = A + 2 → 
    3 * A + 102 = 4 * new_average) → 
  A = 94 := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_score_l3736_373626


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3736_373627

/-- The area of a triangle with vertices at (0, 0), (2, 2), and (4, 0) is 4 -/
theorem triangle_area : ℝ → Prop :=
  fun a =>
    let X : ℝ × ℝ := (0, 0)
    let Y : ℝ × ℝ := (2, 2)
    let Z : ℝ × ℝ := (4, 0)
    let base : ℝ := 4
    let height : ℝ := 2
    a = (1 / 2) * base * height ∧ a = 4

/-- The proof of the theorem -/
theorem triangle_area_proof : triangle_area 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l3736_373627


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l3736_373677

/-- Proves that the cost price of a computer table is 6672 when the selling price is 8340 and the markup is 25% -/
theorem computer_table_cost_price (selling_price : ℝ) (markup_percentage : ℝ) 
  (h1 : selling_price = 8340)
  (h2 : markup_percentage = 25) : 
  selling_price / (1 + markup_percentage / 100) = 6672 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l3736_373677


namespace NUMINAMATH_CALUDE_units_digit_a_2017_l3736_373681

/-- Represents the units digit of an integer -/
def M (x : ℤ) : ℕ :=
  x.natAbs % 10

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sorry  -- Definition to be filled based on the recurrence relation

/-- The main theorem to be proved -/
theorem units_digit_a_2017 :
  M (Int.floor (a 2016)) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_a_2017_l3736_373681


namespace NUMINAMATH_CALUDE_square_configuration_counts_l3736_373600

/-- A configuration of points and line segments in a square -/
structure SquareConfiguration where
  /-- The number of interior points in the square -/
  interior_points : Nat
  /-- The total number of line segments -/
  line_segments : Nat
  /-- The total number of triangles formed -/
  triangles : Nat
  /-- No three points (including square vertices) are collinear -/
  no_collinear_triple : Prop
  /-- No two segments (except at endpoints) share common points -/
  no_intersecting_segments : Prop

/-- Theorem about the number of line segments and triangles in a specific square configuration -/
theorem square_configuration_counts (config : SquareConfiguration) :
  config.interior_points = 1000 →
  config.no_collinear_triple →
  config.no_intersecting_segments →
  config.line_segments = 3001 ∧ config.triangles = 2002 := by
  sorry


end NUMINAMATH_CALUDE_square_configuration_counts_l3736_373600


namespace NUMINAMATH_CALUDE_find_x_l3736_373650

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(2*b) = a^b * x^b → x = a^3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3736_373650


namespace NUMINAMATH_CALUDE_farmer_land_calculation_l3736_373606

/-- The total land owned by a farmer, given the following conditions:
  * 90% of the total land is cleared for planting
  * 10% of the cleared land is planted with grapes
  * 80% of the cleared land is planted with potatoes
  * The remaining cleared land (450 acres) is planted with tomatoes
-/
def farmer_land : ℝ := 1666.67

/-- The proportion of total land that is cleared for planting -/
def cleared_proportion : ℝ := 0.90

/-- The proportion of cleared land planted with grapes -/
def grape_proportion : ℝ := 0.10

/-- The proportion of cleared land planted with potatoes -/
def potato_proportion : ℝ := 0.80

/-- The amount of cleared land planted with tomatoes (in acres) -/
def tomato_acres : ℝ := 450

/-- Theorem stating that the farmer's land calculation is correct -/
theorem farmer_land_calculation (ε : ℝ) (hε : ε > 0) :
  ∃ (land : ℝ), abs (land - farmer_land) < ε ∧
  cleared_proportion * land * (1 - grape_proportion - potato_proportion) = tomato_acres :=
sorry

end NUMINAMATH_CALUDE_farmer_land_calculation_l3736_373606


namespace NUMINAMATH_CALUDE_fraction_of_number_l3736_373620

theorem fraction_of_number (N : ℝ) (h : N = 180) : 
  6 + (1/2) * (1/3) * (1/5) * N = (1/25) * N := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_number_l3736_373620


namespace NUMINAMATH_CALUDE_min_delivery_time_75_minutes_l3736_373632

/-- Represents the train's cargo and delivery constraints -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  wood_cars : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  travel_time : ℕ

/-- Calculates the minimum number of stops required to deliver all cars -/
def min_stops (td : TrainDelivery) : ℕ :=
  max (td.coal_cars / td.max_coal_deposit)
      (max (td.iron_cars / td.max_iron_deposit)
           (td.wood_cars / td.max_wood_deposit))

/-- Calculates the total delivery time based on the number of stops -/
def total_delivery_time (td : TrainDelivery) : ℕ :=
  (min_stops td - 1) * td.travel_time

/-- The main theorem stating the minimum time required for delivery -/
theorem min_delivery_time_75_minutes (td : TrainDelivery) 
  (h1 : td.coal_cars = 6)
  (h2 : td.iron_cars = 12)
  (h3 : td.wood_cars = 2)
  (h4 : td.max_coal_deposit = 2)
  (h5 : td.max_iron_deposit = 3)
  (h6 : td.max_wood_deposit = 1)
  (h7 : td.travel_time = 25) :
  total_delivery_time td = 75 := by
  sorry

#eval total_delivery_time {
  coal_cars := 6,
  iron_cars := 12,
  wood_cars := 2,
  max_coal_deposit := 2,
  max_iron_deposit := 3,
  max_wood_deposit := 1,
  travel_time := 25
}

end NUMINAMATH_CALUDE_min_delivery_time_75_minutes_l3736_373632


namespace NUMINAMATH_CALUDE_product_of_roots_l3736_373699

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y : ℝ, (x + 3) * (x - 5) = 20 ∧ (y + 3) * (y - 5) = 20 ∧ x * y = -35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3736_373699


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3736_373637

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3736_373637


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3736_373611

theorem solution_set_inequality (x : ℝ) : 
  (1 / x > 1 / (x - 1)) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3736_373611


namespace NUMINAMATH_CALUDE_white_ball_mutually_exclusive_l3736_373685

-- Define the set of balls
inductive Ball : Type
  | Red : Ball
  | Black : Ball
  | White : Ball

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "person receives the white ball"
def receives_white_ball (p : Person) (d : Distribution) : Prop :=
  d p = Ball.White

-- State the theorem
theorem white_ball_mutually_exclusive :
  ∀ (d : Distribution),
    (∀ (p1 p2 : Person), p1 ≠ p2 → d p1 ≠ d p2) →
    ¬(receives_white_ball Person.A d ∧ receives_white_ball Person.B d) :=
by sorry

end NUMINAMATH_CALUDE_white_ball_mutually_exclusive_l3736_373685


namespace NUMINAMATH_CALUDE_tuesday_wednesday_thursday_avg_l3736_373661

def tuesday_temp : ℝ := 38
def friday_temp : ℝ := 44
def wed_thur_fri_avg : ℝ := 34

theorem tuesday_wednesday_thursday_avg :
  let wed_thur_sum := 3 * wed_thur_fri_avg - friday_temp
  (tuesday_temp + wed_thur_sum) / 3 = 32 :=
by sorry

end NUMINAMATH_CALUDE_tuesday_wednesday_thursday_avg_l3736_373661


namespace NUMINAMATH_CALUDE_cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3736_373612

theorem cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero (α k : ℝ) : 
  (∃ x y : ℝ, x = Real.tan α ∧ y = (Real.tan α)⁻¹ ∧ 
   x^2 - k*x + k^2 - 3 = 0 ∧ y^2 - k*y + k^2 - 3 = 0) →
  3*Real.pi < α ∧ α < (7/2)*Real.pi →
  Real.cos (3*Real.pi + α) - Real.sin (Real.pi + α) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_3pi_plus_alpha_minus_sin_pi_plus_alpha_is_zero_l3736_373612


namespace NUMINAMATH_CALUDE_gnome_ratio_is_half_l3736_373623

/-- Represents the ratio of gnomes with big noses to total gnomes -/
def gnome_ratio (total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat : ℕ) : ℚ := 
  let big_nose_red_hat := red_hat_gnomes - small_nose_red_hat
  let total_big_nose := big_nose_blue_hat + big_nose_red_hat
  (total_big_nose : ℚ) / total_gnomes

theorem gnome_ratio_is_half :
  let total_gnomes : ℕ := 28
  let red_hat_gnomes : ℕ := (3 * total_gnomes) / 4
  let blue_hat_gnomes : ℕ := total_gnomes - red_hat_gnomes
  let big_nose_blue_hat : ℕ := 6
  let small_nose_red_hat : ℕ := 13
  gnome_ratio total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gnome_ratio_is_half_l3736_373623


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l3736_373682

theorem quadratic_roots_imply_c_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 8 ∨ x = (-8 - Real.sqrt 16) / 8) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l3736_373682


namespace NUMINAMATH_CALUDE_outfits_count_l3736_373668

/-- The number of different outfits that can be created -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items -/
theorem outfits_count :
  number_of_outfits 8 4 3 2 = 360 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l3736_373668


namespace NUMINAMATH_CALUDE_football_count_white_patch_count_l3736_373659

/- Define the number of students -/
def num_students : ℕ := 36

/- Define the number of footballs -/
def num_footballs : ℕ := 27

/- Define the number of black patches -/
def num_black_patches : ℕ := 12

/- Define the number of white patches -/
def num_white_patches : ℕ := 20

/- Theorem for the number of footballs -/
theorem football_count : 
  (num_students - 9 = num_footballs) ∧ 
  (num_students / 2 + 9 = num_footballs) := by
  sorry

/- Theorem for the number of white patches -/
theorem white_patch_count :
  2 * num_black_patches * 5 = 6 * num_white_patches := by
  sorry

end NUMINAMATH_CALUDE_football_count_white_patch_count_l3736_373659


namespace NUMINAMATH_CALUDE_smallest_rectangle_area_l3736_373610

theorem smallest_rectangle_area (r : ℝ) (h : r = 5) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w * l = 200 ∧
  ∀ (w' l' : ℝ), w' > 0 → l' > 0 → w' * l' ≥ 200 →
  (∀ (x y : ℝ), x^2 + y^2 ≤ r^2 → 0 ≤ x ∧ x ≤ w' ∧ 0 ≤ y ∧ y ≤ l') :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_area_l3736_373610


namespace NUMINAMATH_CALUDE_conductor_loop_properties_l3736_373638

/-- Parameters for the conductor and loop setup -/
structure ConductorLoopSetup where
  k : Real  -- Current rate of change (A/s)
  r : Real  -- Side length of square loop (m)
  R : Real  -- Resistance of loop (Ω)
  l : Real  -- Distance from straight conductor to loop (m)

/-- Calculate the induced voltage in the loop -/
noncomputable def inducedVoltage (setup : ConductorLoopSetup) : Real :=
  (setup.k * setup.r * Real.log (1 + setup.r / setup.l)) / (2 * Real.pi)

/-- Calculate the time when magnetic induction at the center is zero -/
noncomputable def zeroInductionTime (setup : ConductorLoopSetup) : Real :=
  (4 * Real.sqrt 2 * (setup.l + setup.r / 2) * (inducedVoltage setup / setup.R)) / (setup.k * setup.r)

/-- Theorem stating the properties of the conductor-loop system -/
theorem conductor_loop_properties (setup : ConductorLoopSetup) 
  (h_k : setup.k = 1000)
  (h_r : setup.r = 0.2)
  (h_R : setup.R = 0.01)
  (h_l : setup.l = 0.05) :
  abs (inducedVoltage setup - 6.44e-5) < 1e-7 ∧ 
  abs (zeroInductionTime setup - 2.73e-4) < 1e-6 := by
  sorry


end NUMINAMATH_CALUDE_conductor_loop_properties_l3736_373638


namespace NUMINAMATH_CALUDE_girls_count_l3736_373625

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The properties of the college in the problem -/
def ProblemCollege : Prop :=
  ∃ (c : College),
    (c.boys : ℚ) / c.girls = 8 / 5 ∧
    c.boys + c.girls = 780

/-- The theorem to be proved -/
theorem girls_count (h : ProblemCollege) : ∃ (c : College), c.girls = 300 ∧ ProblemCollege := by
  sorry

end NUMINAMATH_CALUDE_girls_count_l3736_373625


namespace NUMINAMATH_CALUDE_probability_triangle_or_square_l3736_373671

theorem probability_triangle_or_square (total_figures : ℕ) 
  (triangle_count : ℕ) (square_count : ℕ) :
  total_figures = 10 →
  triangle_count = 3 →
  square_count = 4 →
  (triangle_count + square_count : ℚ) / total_figures = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_square_l3736_373671


namespace NUMINAMATH_CALUDE_gift_box_wrapping_l3736_373648

theorem gift_box_wrapping (total_ribbon : ℝ) (ribbon_per_box : ℝ) :
  total_ribbon = 25 →
  ribbon_per_box = 1.6 →
  ⌊total_ribbon / ribbon_per_box⌋ = 15 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_wrapping_l3736_373648


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3736_373674

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 5 * x^3 - 10 * x^2 + x - 24
  ∃ (r₁ r₂ r₃ : ℝ), p r₁ = 0 ∧ p r₂ = 0 ∧ p r₃ = 0 ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3736_373674


namespace NUMINAMATH_CALUDE_bottle_cap_cost_l3736_373619

/-- Given that 5 bottle caps cost $25, prove that each bottle cap costs $5. -/
theorem bottle_cap_cost : 
  ∀ (cost_per_cap : ℚ), 
  (5 : ℚ) * cost_per_cap = 25 → cost_per_cap = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_cost_l3736_373619


namespace NUMINAMATH_CALUDE_seventh_term_equals_33_l3736_373614

/-- An arithmetic sequence with 15 terms, first term 3, and last term 72 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  3 + (n - 1) * ((72 - 3) / 14)

/-- The 7th term of the arithmetic sequence -/
def seventh_term : ℚ := arithmetic_sequence 7

theorem seventh_term_equals_33 : ⌊seventh_term⌋ = 33 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_equals_33_l3736_373614


namespace NUMINAMATH_CALUDE_patio_rearrangement_l3736_373633

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (column_reduction : ℕ) :
  total_tiles = 96 →
  initial_rows = 8 →
  column_reduction = 2 →
  let initial_columns := total_tiles / initial_rows
  let new_columns := initial_columns - column_reduction
  let new_rows := total_tiles / new_columns
  new_rows - initial_rows = 4 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l3736_373633


namespace NUMINAMATH_CALUDE_terminal_side_angles_l3736_373667

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem terminal_side_angles :
  (∃ k : ℤ, angle_set k = 240) ∧
  (∃ k : ℤ, angle_set k = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) → α ≥ 240 ∨ α ≤ -120) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_angles_l3736_373667


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3736_373616

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_product (n : ℕ) :
  (3 * sum_factorials n) % 10 = 9 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3736_373616


namespace NUMINAMATH_CALUDE_max_daily_sales_l3736_373690

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_sales (t : ℕ) : ℝ := price t * sales_volume t

theorem max_daily_sales :
  ∃ (max_sales : ℝ) (max_day : ℕ),
    max_sales = 1125 ∧
    max_day = 25 ∧
    ∀ t : ℕ, 0 < t ∧ t ≤ 30 → daily_sales t ≤ max_sales ∧
    daily_sales max_day = max_sales :=
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_l3736_373690


namespace NUMINAMATH_CALUDE_platform_length_l3736_373613

/-- Given a train with the following properties:
  * Length: 300 meters
  * Starting from rest
  * Constant acceleration
  * Crosses a signal pole in 24 seconds
  * Crosses a platform in 39 seconds
  Prove that the length of the platform is approximately 492.19 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_time = 24)
  (h3 : platform_time = 39) :
  ∃ (platform_length : ℝ), 
    (abs (platform_length - 492.19) < 0.01) ∧ 
    (∃ (a : ℝ), 
      (train_length = (1/2) * a * pole_time^2) ∧
      (train_length + platform_length = (1/2) * a * platform_time^2)) :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3736_373613


namespace NUMINAMATH_CALUDE_number_in_mind_l3736_373618

theorem number_in_mind (x : ℝ) : (x - 6) / 13 = 2 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_in_mind_l3736_373618


namespace NUMINAMATH_CALUDE_square_side_length_l3736_373604

/-- Given a rectangle composed of rectangles R1 and R2, and squares S1, S2, and S3,
    this theorem proves that the side length of S2 is 875 units. -/
theorem square_side_length (total_width total_height : ℕ) (s2 s3 : ℕ) :
  total_width = 4020 →
  total_height = 2160 →
  s3 = s2 + 110 →
  ∃ (r : ℕ), 
    2 * r + s2 = total_height ∧
    2 * r + 3 * s2 + 110 = total_width →
  s2 = 875 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3736_373604


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3736_373679

theorem exponent_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3736_373679


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l3736_373664

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_ten_million (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10000000

theorem smallest_divisible_term : 
  (∀ n < 8, ¬ is_divisible_by_ten_million (geometric_sequence (5/6) 25 n)) ∧ 
  is_divisible_by_ten_million (geometric_sequence (5/6) 25 8) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l3736_373664


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l3736_373628

theorem smallest_of_five_consecutive_even_numbers (a b c d e : ℕ) : 
  (∀ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6 ∧ e = 2*n + 8) → 
  a + b + c + d + e = 320 → 
  a = 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_numbers_l3736_373628


namespace NUMINAMATH_CALUDE_expression_simplification_l3736_373653

theorem expression_simplification (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  (((x / (x + b)) + (b / (x - b))) / ((b / (x + b)) - (x / (x - b)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3736_373653


namespace NUMINAMATH_CALUDE_product_equals_eight_l3736_373631

theorem product_equals_eight (x : ℝ) (hx : x ≠ 0) : 
  ∃ y : ℝ, x * y = 8 ∧ y = 8 / x := by sorry

end NUMINAMATH_CALUDE_product_equals_eight_l3736_373631


namespace NUMINAMATH_CALUDE_inequality_proof_l3736_373639

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3736_373639


namespace NUMINAMATH_CALUDE_functional_equation_identity_l3736_373660

theorem functional_equation_identity (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_identity_l3736_373660


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3736_373666

/-- The number of ways to place 5 numbered balls into 5 numbered boxes, leaving one box empty -/
def ball_placement_count : ℕ := 1200

/-- The number of balls -/
def num_balls : ℕ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 5

theorem ball_placement_theorem : 
  ball_placement_count = 1200 ∧ 
  num_balls = 5 ∧ 
  num_boxes = 5 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3736_373666


namespace NUMINAMATH_CALUDE_line_parallel_plane_condition_l3736_373622

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the "lies in" relation for a line in a plane
variable (liesIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_condition 
  (a : Line) (α β : Plane) :
  parallelPlane α β → liesIn a β → parallelLinePlane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_condition_l3736_373622


namespace NUMINAMATH_CALUDE_alex_makes_100_dresses_l3736_373673

/-- Given the initial amount of silk, silk given to friends, and silk required per dress,
    calculate the number of dresses Alex can make. -/
def dresses_alex_can_make (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (silk_per_dress : ℕ) : ℕ :=
  (initial_silk - friends * silk_per_friend) / silk_per_dress

/-- Prove that Alex can make 100 dresses given the conditions. -/
theorem alex_makes_100_dresses :
  dresses_alex_can_make 600 5 20 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_alex_makes_100_dresses_l3736_373673


namespace NUMINAMATH_CALUDE_pyramid_width_height_difference_l3736_373615

/-- The Great Pyramid of Giza's dimensions --/
structure PyramidDimensions where
  height : ℝ
  width : ℝ
  height_is_520 : height = 520
  width_greater_than_height : width > height
  sum_of_dimensions : height + width = 1274

/-- The difference between the width and height of the pyramid is 234 feet --/
theorem pyramid_width_height_difference (p : PyramidDimensions) : 
  p.width - p.height = 234 := by
sorry

end NUMINAMATH_CALUDE_pyramid_width_height_difference_l3736_373615


namespace NUMINAMATH_CALUDE_equiv_class_characterization_l3736_373644

/-- Given a positive integer m and an integer a, this theorem states that 
    an integer b is in the equivalence class of a modulo m if and only if 
    there exists an integer t such that b = m * t + a. -/
theorem equiv_class_characterization (m : ℕ+) (a b : ℤ) : 
  b ≡ a [ZMOD m] ↔ ∃ t : ℤ, b = m * t + a := by sorry

end NUMINAMATH_CALUDE_equiv_class_characterization_l3736_373644


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3736_373636

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≥ 2}
def N : Set ℝ := {x | x^2 - 4*x ≥ 0}

-- State the theorem
theorem set_intersection_equality : 
  M ∩ N = {x | x ≤ -1 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3736_373636


namespace NUMINAMATH_CALUDE_factorization_problems_l3736_373651

theorem factorization_problems :
  (∀ a : ℝ, a^3 - 4*a = a*(a+2)*(a-2)) ∧
  (∀ m x y : ℝ, 3*m*x^2 - 6*m*x*y + 3*m*y^2 = 3*m*(x-y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3736_373651


namespace NUMINAMATH_CALUDE_field_trip_attendance_l3736_373665

theorem field_trip_attendance (vans buses : ℕ) (people_per_van people_per_bus : ℕ) 
  (h1 : vans = 9)
  (h2 : buses = 10)
  (h3 : people_per_van = 8)
  (h4 : people_per_bus = 27) :
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l3736_373665


namespace NUMINAMATH_CALUDE_gcd_14658_11241_l3736_373687

theorem gcd_14658_11241 : Nat.gcd 14658 11241 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14658_11241_l3736_373687


namespace NUMINAMATH_CALUDE_combined_completion_time_l3736_373640

/-- Given the time taken by X, Y, and Z to complete a task individually,
    calculate the time taken when they work together. -/
theorem combined_completion_time
  (x_time y_time z_time : ℝ)
  (hx : x_time = 15)
  (hy : y_time = 30)
  (hz : z_time = 20)
  : (1 : ℝ) / ((1 / x_time) + (1 / y_time) + (1 / z_time)) = 20 / 3 := by
  sorry

#check combined_completion_time

end NUMINAMATH_CALUDE_combined_completion_time_l3736_373640


namespace NUMINAMATH_CALUDE_y_equation_implies_expression_equals_two_l3736_373662

theorem y_equation_implies_expression_equals_two (y : ℝ) (h : y + 2/y = 2) :
  y^6 + 3*y^4 - 4*y^2 + 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_equation_implies_expression_equals_two_l3736_373662


namespace NUMINAMATH_CALUDE_monkeys_for_three_bananas_l3736_373692

/-- The number of monkeys needed to eat a given number of bananas in 8 minutes -/
def monkeys_needed (bananas : ℕ) : ℕ :=
  bananas

theorem monkeys_for_three_bananas :
  monkeys_needed 3 = 3 :=
by
  sorry

/-- Given condition: 8 monkeys take 8 minutes to eat 8 bananas -/
axiom eight_monkeys_eight_bananas : monkeys_needed 8 = 8

end NUMINAMATH_CALUDE_monkeys_for_three_bananas_l3736_373692


namespace NUMINAMATH_CALUDE_trajectory_theorem_l3736_373645

def trajectory_problem (R h : ℝ) (θ : ℝ) : Prop :=
  let r₁ := R * Real.cos θ
  let r₂ := (R + h) * Real.cos θ
  let s := 2 * Real.pi * r₂ - 2 * Real.pi * r₁
  s = h ∧ θ = Real.arccos (1 / (2 * Real.pi))

theorem trajectory_theorem :
  ∀ (R h : ℝ), R > 0 → h > 0 → ∃ θ : ℝ, trajectory_problem R h θ :=
sorry

end NUMINAMATH_CALUDE_trajectory_theorem_l3736_373645


namespace NUMINAMATH_CALUDE_new_person_weight_l3736_373605

/-- The weight of a new person joining a group, given the initial group size,
    average weight increase, and weight of the replaced person. -/
theorem new_person_weight
  (initial_group_size : ℕ)
  (average_weight_increase : ℝ)
  (replaced_person_weight : ℝ)
  (h1 : initial_group_size = 8)
  (h2 : average_weight_increase = 6)
  (h3 : replaced_person_weight = 40) :
  replaced_person_weight + initial_group_size * average_weight_increase = 88 :=
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3736_373605


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l3736_373601

/-- If 46 cows eat 46 bags of husk in 46 days, then one cow will eat one bag of husk in 46 days. -/
theorem dairy_farm_husk_consumption 
  (cows : ℕ) (bags : ℕ) (days : ℕ) (one_cow_days : ℕ) :
  cows = 46 → bags = 46 → days = 46 → 
  (cows * bags = cows * days) →
  one_cow_days = 46 :=
by sorry

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l3736_373601


namespace NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l3736_373609

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a plane -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersection points between a circle and a line -/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersection points between a parabola and a line -/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersection points between a circle and a parabola -/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersection points among a circle, a line, and a parabola on a plane is 8 -/
theorem max_intersections_circle_line_parabola :
  ∀ (c : Circle) (l : Line) (p : Parabola),
  max_intersections_circle_line +
  max_intersections_parabola_line +
  max_intersections_circle_parabola = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_circle_line_parabola_l3736_373609


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3736_373607

theorem unique_quadratic_solution (c : ℝ) : 
  (c = -1 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) ↔ 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b ≠ 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3736_373607


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l3736_373678

theorem sum_of_squares_theorem (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l3736_373678


namespace NUMINAMATH_CALUDE_yanna_afternoon_butter_cookies_l3736_373624

/-- Represents the number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := sorry

/-- Represents the number of butter cookies Yanna baked in the morning -/
def morning_butter_cookies : ℕ := 20

/-- Represents the number of biscuits Yanna baked in the morning -/
def morning_biscuits : ℕ := 40

/-- Represents the number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := 20

theorem yanna_afternoon_butter_cookies :
  afternoon_butter_cookies = 20 ∧
  morning_butter_cookies + afternoon_butter_cookies + 30 =
  morning_biscuits + afternoon_biscuits :=
sorry

end NUMINAMATH_CALUDE_yanna_afternoon_butter_cookies_l3736_373624


namespace NUMINAMATH_CALUDE_commuting_days_businessman_commute_l3736_373691

/-- Represents the commuting options for a businessman over a period of days. -/
structure CommutingData where
  /-- Total number of days -/
  total_days : ℕ
  /-- Number of times taking bus to work in the morning -/
  morning_bus : ℕ
  /-- Number of times coming home by bus in the afternoon -/
  afternoon_bus : ℕ
  /-- Number of train commuting segments (either morning or afternoon) -/
  train_segments : ℕ

/-- Theorem stating that given the commuting conditions, the total number of days is 32 -/
theorem commuting_days (data : CommutingData) : 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 →
  data.total_days = 32 := by
  sorry

/-- Main theorem proving the specific case -/
theorem businessman_commute : ∃ (data : CommutingData), 
  data.morning_bus = 12 ∧ 
  data.afternoon_bus = 20 ∧ 
  data.train_segments = 15 ∧
  data.total_days = 32 := by
  sorry

end NUMINAMATH_CALUDE_commuting_days_businessman_commute_l3736_373691


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3736_373695

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 ∧ 
  ∃ (a' b' c' d' : ℝ), a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ d' ≥ 0 ∧ 
    a' + b' + c' + d' = 200 ∧
    a' * b' + b' * c' + c' * d' + d' * a' = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3736_373695


namespace NUMINAMATH_CALUDE_questionnaires_from_unit_d_l3736_373617

/-- Represents the number of questionnaires drawn from each unit -/
structure SampledQuestionnaires :=
  (a b c d : ℕ)

/-- Represents the total number of questionnaires collected from each unit -/
structure TotalQuestionnaires :=
  (a b c d : ℕ)

/-- The properties of the survey and sampling -/
class SurveyProperties (sampled : SampledQuestionnaires) (total : TotalQuestionnaires) :=
  (total_sum : total.a + total.b + total.c + total.d = 1000)
  (sample_sum : sampled.a + sampled.b + sampled.c + sampled.d = 150)
  (total_arithmetic : ∃ (r : ℕ), total.b = total.a + r ∧ total.c = total.b + r ∧ total.d = total.c + r)
  (sample_arithmetic : ∃ (d : ℤ), sampled.b = sampled.a + d ∧ sampled.c = sampled.b + d ∧ sampled.d = sampled.c + d)
  (unit_b_sample : sampled.b = 30)
  (stratified_sampling : ∀ (x y : Fin 4), 
    (sampled.a * total.b = sampled.b * total.a) ∧
    (sampled.b * total.c = sampled.c * total.b) ∧
    (sampled.c * total.d = sampled.d * total.c))

/-- The main theorem to prove -/
theorem questionnaires_from_unit_d 
  (sampled : SampledQuestionnaires) 
  (total : TotalQuestionnaires) 
  [SurveyProperties sampled total] : 
  sampled.d = 60 := by
  sorry

end NUMINAMATH_CALUDE_questionnaires_from_unit_d_l3736_373617


namespace NUMINAMATH_CALUDE_prob_box1_given_defective_l3736_373642

-- Define the number of components and defective components in each box
def total_box1 : ℕ := 10
def defective_box1 : ℕ := 3
def total_box2 : ℕ := 20
def defective_box2 : ℕ := 2

-- Define the probability of selecting each box
def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

-- Define the probability of selecting a defective component from each box
def prob_defective_given_box1 : ℚ := defective_box1 / total_box1
def prob_defective_given_box2 : ℚ := defective_box2 / total_box2

-- Define the overall probability of selecting a defective component
def prob_defective : ℚ := 
  prob_select_box1 * prob_defective_given_box1 + 
  prob_select_box2 * prob_defective_given_box2

-- State the theorem
theorem prob_box1_given_defective : 
  (prob_select_box1 * prob_defective_given_box1) / prob_defective = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_box1_given_defective_l3736_373642


namespace NUMINAMATH_CALUDE_average_equation_solution_l3736_373629

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 4) + (5*x + 3) + (3*x + 8)) / 3 = 3*x - 5 → x = -30 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l3736_373629


namespace NUMINAMATH_CALUDE_max_value_a_plus_2b_l3736_373657

theorem max_value_a_plus_2b (a b : ℝ) (h : a^2 + 2*b^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 ∧ ∀ (x y : ℝ), x^2 + 2*y^2 = 1 → x + 2*y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_a_plus_2b_l3736_373657


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3736_373683

theorem unique_solution_for_equation : 
  ∃! (n m : ℕ), (n - 1) * 2^(n - 1) + 5 = m^2 + 4*m ∧ n = 6 ∧ m = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3736_373683


namespace NUMINAMATH_CALUDE_xias_initial_sticker_count_l3736_373696

/-- Theorem: Xia's initial sticker count
Given that Xia shared 100 stickers with her friends, had 5 sheets of stickers left,
and each sheet contains 10 stickers, prove that she initially had 150 stickers. -/
theorem xias_initial_sticker_count
  (shared_stickers : ℕ)
  (remaining_sheets : ℕ)
  (stickers_per_sheet : ℕ)
  (h1 : shared_stickers = 100)
  (h2 : remaining_sheets = 5)
  (h3 : stickers_per_sheet = 10) :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 :=
by sorry

end NUMINAMATH_CALUDE_xias_initial_sticker_count_l3736_373696


namespace NUMINAMATH_CALUDE_altitude_sum_less_than_perimeter_l3736_373647

/-- For any triangle, the sum of its altitudes is less than its perimeter -/
theorem altitude_sum_less_than_perimeter (a b c h_a h_b h_c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)
  (altitude_a : h_a ≤ b ∧ h_a ≤ c)
  (altitude_b : h_b ≤ a ∧ h_b ≤ c)
  (altitude_c : h_c ≤ a ∧ h_c ≤ b)
  (non_degenerate : h_a < b ∨ h_a < c ∨ h_b < a ∨ h_b < c ∨ h_c < a ∨ h_c < b) :
  h_a + h_b + h_c < a + b + c := by
  sorry

end NUMINAMATH_CALUDE_altitude_sum_less_than_perimeter_l3736_373647


namespace NUMINAMATH_CALUDE_ab_power_2022_l3736_373669

theorem ab_power_2022 (a b : ℝ) (h : (a - 1/2)^2 + |b + 2| = 0) : (a * b)^2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2022_l3736_373669


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3736_373630

theorem solve_linear_equation (x : ℝ) :
  3 * x - 5 * x + 7 * x = 140 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3736_373630


namespace NUMINAMATH_CALUDE_blue_notes_scattered_l3736_373646

def red_rows : ℕ := 5
def red_notes_per_row : ℕ := 6
def blue_notes_under_each_red : ℕ := 2
def total_notes : ℕ := 100

theorem blue_notes_scattered (red_rows : ℕ) (red_notes_per_row : ℕ) (blue_notes_under_each_red : ℕ) (total_notes : ℕ) :
  red_rows = 5 →
  red_notes_per_row = 6 →
  blue_notes_under_each_red = 2 →
  total_notes = 100 →
  total_notes - (red_rows * red_notes_per_row + red_rows * red_notes_per_row * blue_notes_under_each_red) = 10 :=
by sorry

end NUMINAMATH_CALUDE_blue_notes_scattered_l3736_373646


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l3736_373670

/-- The number of ways to distribute n distinct objects into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    where the first box is not empty -/
def distributeWithFirstBoxNonEmpty (n k : ℕ) : ℕ :=
  distribute n k - distribute n (k - 1)

/-- The problem statement -/
theorem ball_distribution_problem :
  distributeWithFirstBoxNonEmpty 3 4 = 37 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l3736_373670


namespace NUMINAMATH_CALUDE_expression_evaluation_l3736_373641

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3736_373641


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3736_373621

theorem max_value_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  x * (1 - 2*x) ≤ 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3736_373621


namespace NUMINAMATH_CALUDE_analogical_conclusions_correctness_l3736_373652

theorem analogical_conclusions_correctness :
  (∃! i : Fin 3, 
    (i = 0 → ∀ (a b : ℝ) (n : ℕ), (a + b)^n = a^n + b^n) ∨
    (i = 1 → ∀ (α β : ℝ), Real.sin (α + β) = Real.sin α * Real.sin β) ∨
    (i = 2 → ∀ (a b : ℝ), (a + b)^2 = a^2 + 2*a*b + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_analogical_conclusions_correctness_l3736_373652


namespace NUMINAMATH_CALUDE_rod_length_proof_l3736_373634

/-- Given that a 12-meter long rod weighs 14 kg, prove that a rod weighing 7 kg is 6 meters long -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14 / 12) : 
  7 / weight_per_meter = 6 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_proof_l3736_373634


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3736_373688

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 3/7; c, 4/7]
  is_projection_matrix P ↔ a = 1 ∧ c = 3/7 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3736_373688


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3736_373658

def A : Set ℝ := {y | ∃ x, y = 2^x ∧ 0 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3736_373658


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l3736_373602

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def digit_factorial_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.map factorial).sum

def contains_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d ∈ n.digits 10

theorem three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
    contains_digits n [7, 2, 1] ∧
    n = digit_factorial_sum n :=
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l3736_373602


namespace NUMINAMATH_CALUDE_first_restaurant_meals_first_restaurant_meals_proof_l3736_373697

theorem first_restaurant_meals (total_restaurants : Nat) 
  (second_restaurant_meals : Nat) (third_restaurant_meals : Nat) 
  (total_weekly_meals : Nat) (days_per_week : Nat) : Nat :=
  let first_restaurant_daily_meals := 
    (total_weekly_meals - (second_restaurant_meals + third_restaurant_meals) * days_per_week) / days_per_week
  first_restaurant_daily_meals

#check @first_restaurant_meals

theorem first_restaurant_meals_proof 
  (h1 : total_restaurants = 3)
  (h2 : second_restaurant_meals = 40)
  (h3 : third_restaurant_meals = 50)
  (h4 : total_weekly_meals = 770)
  (h5 : days_per_week = 7) :
  first_restaurant_meals total_restaurants second_restaurant_meals third_restaurant_meals total_weekly_meals days_per_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_restaurant_meals_first_restaurant_meals_proof_l3736_373697


namespace NUMINAMATH_CALUDE_triangle_tan_A_l3736_373654

theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / b = (b + Real.sqrt 3 * c) / a →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  Real.tan A = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l3736_373654


namespace NUMINAMATH_CALUDE_circle_symmetry_axis_l3736_373656

theorem circle_symmetry_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 2*y₀ + 1 = 0 ∧
      m*x₀ + y₀ - 1 = 0 ∧
      ∀ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 →
        (m*x' + y' - 1 = 0 ↔ m*(2*x₀ - x') + (2*y₀ - y') - 1 = 0))) →
  m = 1 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_axis_l3736_373656


namespace NUMINAMATH_CALUDE_machine_work_time_l3736_373603

theorem machine_work_time (x : ℝ) 
  (h1 : (1 / (x + 6) + 1 / (x + 1) + 1 / (2 * x)) = 1 / x) 
  (h2 : x > 0) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l3736_373603


namespace NUMINAMATH_CALUDE_lizzy_money_after_loan_l3736_373686

def calculate_final_amount (initial_amount : ℝ) (loan_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

theorem lizzy_money_after_loan (initial_amount loan_amount interest_rate : ℝ) 
  (h1 : initial_amount = 30)
  (h2 : loan_amount = 15)
  (h3 : interest_rate = 0.2) :
  calculate_final_amount initial_amount loan_amount interest_rate = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_after_loan_l3736_373686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3736_373698

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_third : a 3 = 7) :
  a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3736_373698


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3736_373675

-- Define the propositions
variable (f : ℝ → ℝ)
def p (f : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, deriv f x = c
def q (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ f : ℝ → ℝ, q f → p f) ∧ (∃ f : ℝ → ℝ, p f ∧ ¬q f) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3736_373675


namespace NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l3736_373635

/-- Calculates the mass of a man based on the displacement of a boat -/
theorem man_mass_from_boat_displacement (boat_length boat_breadth additional_depth water_density : ℝ) 
  (h1 : boat_length = 4)
  (h2 : boat_breadth = 2)
  (h3 : additional_depth = 0.01)
  (h4 : water_density = 1000) : 
  boat_length * boat_breadth * additional_depth * water_density = 80 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_from_boat_displacement_l3736_373635


namespace NUMINAMATH_CALUDE_correct_pies_left_l3736_373693

/-- Calculates the number of pies left after baking and dropping some -/
def pies_left (oven_capacity : ℕ) (num_batches : ℕ) (dropped_pies : ℕ) : ℕ :=
  oven_capacity * num_batches - dropped_pies

theorem correct_pies_left :
  let oven_capacity : ℕ := 5
  let num_batches : ℕ := 7
  let dropped_pies : ℕ := 8
  pies_left oven_capacity num_batches dropped_pies = 27 := by
  sorry

end NUMINAMATH_CALUDE_correct_pies_left_l3736_373693


namespace NUMINAMATH_CALUDE_quadratic_trinomial_characterization_l3736_373643

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: Replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_one_root_when_replaced (qt : QuadraticTrinomial) : Prop :=
  (qt.b^2 - 4*qt.c = 0) ∧ 
  (1 - 4*qt.a*qt.c = 0) ∧ 
  (qt.b^2 - 4*qt.a = 0)

/-- Theorem: Characterization of quadratic trinomials satisfying the condition -/
theorem quadratic_trinomial_characterization (qt : QuadraticTrinomial) :
  has_one_root_when_replaced qt →
  (qt.a = 1/2 ∧ qt.c = 1/2 ∧ (qt.b = Real.sqrt 2 ∨ qt.b = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_characterization_l3736_373643


namespace NUMINAMATH_CALUDE_unique_solution_l3736_373694

/-- Represents the number of correct answers for each friend -/
structure ExamResults :=
  (A B C D : Nat)

/-- Checks if the given results satisfy the conditions of the problem -/
def satisfiesConditions (results : ExamResults) : Prop :=
  -- Total correct answers is 6
  results.A + results.B + results.C + results.D = 6 ∧
  -- Each result is between 0 and 3
  results.A ≤ 3 ∧ results.B ≤ 3 ∧ results.C ≤ 3 ∧ results.D ≤ 3 ∧
  -- Number of true statements matches correct answers
  (results.A = 1 ∨ results.A = 2) ∧
  (results.B = 0 ∨ results.B = 2) ∧
  (results.C = 0 ∨ results.C = 1) ∧
  (results.D = 0 ∨ results.D = 3) ∧
  -- Relative performance statements
  (results.A > results.B → results.A = 2) ∧
  (results.C < results.D → results.A = 2) ∧
  (results.C = 0 → results.B = 3) ∧
  (results.A < results.D → results.B = 3) ∧
  (results.D = 2 → results.C = 1) ∧
  (results.B < results.A → results.C = 1) ∧
  (results.C < results.D → results.D = 3) ∧
  (results.A < results.B → results.D = 3)

theorem unique_solution :
  ∃! results : ExamResults, satisfiesConditions results ∧ 
    results.A = 1 ∧ results.B = 2 ∧ results.C = 0 ∧ results.D = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3736_373694


namespace NUMINAMATH_CALUDE_g_composite_three_roots_l3736_373649

/-- The function g(x) defined as x^2 - 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

/-- The composite function g(g(x)) -/
def g_composite (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ 
    (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧
    (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 3 -/
theorem g_composite_three_roots (d : ℝ) :
  has_exactly_three_distinct_real_roots (g_composite d) ↔ d = 3 :=
sorry

end NUMINAMATH_CALUDE_g_composite_three_roots_l3736_373649


namespace NUMINAMATH_CALUDE_problem_classification_l3736_373663

-- Define a type for problems
inductive Problem
  | EquilateralTrianglePerimeter
  | ArithmeticMean
  | SmallerOfTwo
  | PiecewiseFunction

-- Define a function to determine if a problem requires a conditional statement
def requiresConditionalStatement (p : Problem) : Prop :=
  match p with
  | Problem.EquilateralTrianglePerimeter => False
  | Problem.ArithmeticMean => False
  | Problem.SmallerOfTwo => True
  | Problem.PiecewiseFunction => True

-- Theorem statement
theorem problem_classification :
  (¬ requiresConditionalStatement Problem.EquilateralTrianglePerimeter) ∧
  (¬ requiresConditionalStatement Problem.ArithmeticMean) ∧
  (requiresConditionalStatement Problem.SmallerOfTwo) ∧
  (requiresConditionalStatement Problem.PiecewiseFunction) :=
by sorry

end NUMINAMATH_CALUDE_problem_classification_l3736_373663


namespace NUMINAMATH_CALUDE_total_ballpoint_pens_l3736_373689

theorem total_ballpoint_pens (red_pens blue_pens : ℕ) 
  (h1 : red_pens = 37) 
  (h2 : blue_pens = 17) : 
  red_pens + blue_pens = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_ballpoint_pens_l3736_373689


namespace NUMINAMATH_CALUDE_total_upload_hours_l3736_373684

def upload_hours (days : ℕ) (videos_per_day : ℕ) (hours_per_video : ℚ) : ℚ :=
  (days : ℚ) * (videos_per_day : ℚ) * hours_per_video

def june_upload_hours : ℚ :=
  upload_hours 10 5 2 +  -- June 1st to June 10th
  upload_hours 10 10 1 + -- June 11th to June 20th
  upload_hours 5 7 3 +   -- June 21st to June 25th
  upload_hours 5 15 (1/2) -- June 26th to June 30th

theorem total_upload_hours : june_upload_hours = 342.5 := by
  sorry

end NUMINAMATH_CALUDE_total_upload_hours_l3736_373684


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1013_odd_integers_l3736_373676

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1013_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1013)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1013_odd_integers_l3736_373676


namespace NUMINAMATH_CALUDE_percentage_to_original_number_l3736_373680

theorem percentage_to_original_number :
  let percentage : Float := 501.99999999999994
  let original_number : Float := percentage / 100
  original_number = 5.0199999999999994 := by
sorry

end NUMINAMATH_CALUDE_percentage_to_original_number_l3736_373680


namespace NUMINAMATH_CALUDE_hugo_win_given_six_l3736_373608

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The probability of winning the game -/
def prob_win : ℚ := 1 / num_players

/-- The probability of rolling a 6 -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability that no other player rolls a 6 -/
def prob_no_other_six : ℚ := (1 - 1 / die_sides) ^ (num_players - 1)

theorem hugo_win_given_six (
  hugo_first_roll : ℕ
) : 
  hugo_first_roll = die_sides →
  (prob_roll_six * prob_no_other_six) / prob_win = 3125 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_hugo_win_given_six_l3736_373608


namespace NUMINAMATH_CALUDE_log_2_base_10_upper_bound_l3736_373655

theorem log_2_base_10_upper_bound (h1 : 10^3 = 1000) (h2 : 10^4 = 10000)
  (h3 : 2^9 = 512) (h4 : 2^11 = 2048) : Real.log 2 / Real.log 10 < 4/11 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_upper_bound_l3736_373655


namespace NUMINAMATH_CALUDE_seashells_given_l3736_373672

theorem seashells_given (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 62) 
  (h2 : remaining_seashells = 13) : 
  initial_seashells - remaining_seashells = 49 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_l3736_373672
