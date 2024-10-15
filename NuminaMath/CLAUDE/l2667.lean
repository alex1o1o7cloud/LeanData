import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l2667_266777

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2*x*y) / (x^2 + z^2 - y^2 + 2*x*z) = (x + y - z) / (x + z - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2667_266777


namespace NUMINAMATH_CALUDE_solution_to_equation_l2667_266745

theorem solution_to_equation : 
  {x : ℝ | Real.sqrt ((3 + Real.sqrt 8) ^ x) + Real.sqrt ((3 - Real.sqrt 8) ^ x) = 6} = {2, -2} := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2667_266745


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2667_266719

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2667_266719


namespace NUMINAMATH_CALUDE_mask_package_duration_l2667_266736

/-- Calculates the number of days a package of masks will last for a family -/
def mask_duration (total_masks : ℕ) (family_size : ℕ) (days_per_mask : ℕ) : ℕ :=
  (total_masks / family_size) * days_per_mask

/-- Theorem: A package of 100 masks lasts 80 days for a family of 5, changing masks every 4 days -/
theorem mask_package_duration :
  mask_duration 100 5 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mask_package_duration_l2667_266736


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2667_266762

theorem max_value_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2667_266762


namespace NUMINAMATH_CALUDE_parabola_shift_through_origin_l2667_266767

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x + 3)^2 - 1

-- Define the shifted parabola function
def shifted_parabola (h : ℝ) (x : ℝ) : ℝ := parabola (x - h)

-- Theorem statement
theorem parabola_shift_through_origin :
  ∀ h : ℝ, shifted_parabola h 0 = 0 ↔ h = 2 ∨ h = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_through_origin_l2667_266767


namespace NUMINAMATH_CALUDE_range_of_t_l2667_266701

theorem range_of_t (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) 
  (ht : ∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) : 
  t ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l2667_266701


namespace NUMINAMATH_CALUDE_five_double_prime_l2667_266764

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem five_double_prime : prime (prime 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_five_double_prime_l2667_266764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2667_266726

/-- An arithmetic sequence with given first and last terms -/
structure ArithmeticSequence where
  a₁ : ℚ  -- First term
  a₃₀ : ℚ  -- 30th term
  is_arithmetic : a₃₀ = a₁ + 29 * ((a₃₀ - a₁) / 29)  -- Condition for arithmetic sequence

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h₁ : seq.a₁ = 5)
    (h₂ : seq.a₃₀ = 100) : 
  let d := (seq.a₃₀ - seq.a₁) / 29
  let a₈ := seq.a₁ + 7 * d
  let a₁₅ := seq.a₁ + 14 * d
  let S₁₅ := 15 / 2 * (seq.a₁ + a₁₅)
  (a₈ = 25 + 1 / 29) ∧ (S₁₅ = 393 + 2 / 29) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2667_266726


namespace NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_subset_l2667_266782

-- Define sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - 2*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem for part 1
theorem intersection_when_m_3 :
  A ∩ B 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem range_of_m_when_subset (m : ℝ) :
  m > 0 → A ⊆ B m → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_subset_l2667_266782


namespace NUMINAMATH_CALUDE_ellipse_min_major_axis_l2667_266758

/-- Given an ellipse where the maximum area of a triangle formed by a point on the ellipse and its two foci is 1, 
    the minimum length of the major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (b * c = 1) →  -- maximum triangle area condition
  (a^2 = b^2 + c^2) →  -- ellipse equation
  (2 * a ≥ 2 * Real.sqrt 2) ∧ 
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ b₀ * c₀ = 1 ∧ a₀^2 = b₀^2 + c₀^2 ∧ 2 * a₀ = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_major_axis_l2667_266758


namespace NUMINAMATH_CALUDE_expression_value_l2667_266763

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2667_266763


namespace NUMINAMATH_CALUDE_laundry_detergent_cost_l2667_266708

def budget : ℕ := 60
def shower_gel_cost : ℕ := 4
def shower_gel_quantity : ℕ := 4
def toothpaste_cost : ℕ := 3
def remaining_budget : ℕ := 30

theorem laundry_detergent_cost :
  budget - remaining_budget - (shower_gel_cost * shower_gel_quantity + toothpaste_cost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_laundry_detergent_cost_l2667_266708


namespace NUMINAMATH_CALUDE_regression_y_intercept_l2667_266784

/-- Empirical regression equation for height prediction -/
def height_prediction (x : ℝ) (a : ℝ) : ℝ := 3 * x + a

/-- Average height of the 50 classmates -/
def average_height : ℝ := 170

/-- Average shoe size of the 50 classmates -/
def average_shoe_size : ℝ := 40

/-- Theorem stating that the y-intercept (a) of the regression line is 50 -/
theorem regression_y_intercept :
  ∃ (a : ℝ), height_prediction average_shoe_size a = average_height ∧ a = 50 := by
  sorry

end NUMINAMATH_CALUDE_regression_y_intercept_l2667_266784


namespace NUMINAMATH_CALUDE_length_of_segment_AB_l2667_266703

/-- Given two perpendicular lines and a point P, prove the length of AB --/
theorem length_of_segment_AB (a : ℝ) : 
  ∃ (A B : ℝ × ℝ),
    (2 * A.1 - A.2 = 0) ∧ 
    (B.1 + a * B.2 = 0) ∧
    ((0 : ℝ) = (A.1 + B.1) / 2) ∧
    ((10 / a) = (A.2 + B.2) / 2) ∧
    (2 * a = -1) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_l2667_266703


namespace NUMINAMATH_CALUDE_bucket_capacity_l2667_266781

theorem bucket_capacity (tank_capacity : ℕ) : ∃ (x : ℕ),
  (18 * x = tank_capacity) ∧
  (216 * 5 = tank_capacity) ∧
  (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l2667_266781


namespace NUMINAMATH_CALUDE_veggie_servings_per_week_l2667_266793

/-- The number of veggie servings eaten in one week -/
def veggieServingsPerWeek (dailyServings : ℕ) (daysInWeek : ℕ) : ℕ :=
  dailyServings * daysInWeek

/-- Theorem: Given 3 servings daily and 7 days in a week, the total veggie servings per week is 21 -/
theorem veggie_servings_per_week :
  veggieServingsPerWeek 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_veggie_servings_per_week_l2667_266793


namespace NUMINAMATH_CALUDE_als_original_portion_l2667_266779

theorem als_original_portion (total_initial : ℝ) (total_final : ℝ) (al_loss : ℝ) 
  (h1 : total_initial = 1200)
  (h2 : total_final = 1800)
  (h3 : al_loss = 200) :
  ∃ (al betty clare : ℝ),
    al + betty + clare = total_initial ∧
    al - al_loss + 3 * betty + 3 * clare = total_final ∧
    al = 800 := by
  sorry

end NUMINAMATH_CALUDE_als_original_portion_l2667_266779


namespace NUMINAMATH_CALUDE_tamias_dinner_problem_l2667_266714

/-- The number of smaller pieces each large slice is cut into, given the total number of bell peppers,
    the number of large slices per bell pepper, and the total number of slices and pieces. -/
def smaller_pieces_per_slice (total_peppers : ℕ) (large_slices_per_pepper : ℕ) (total_slices_and_pieces : ℕ) : ℕ :=
  let total_large_slices := total_peppers * large_slices_per_pepper
  let large_slices_to_cut := total_large_slices / 2
  let smaller_pieces_needed := total_slices_and_pieces - total_large_slices
  smaller_pieces_needed / large_slices_to_cut

theorem tamias_dinner_problem :
  smaller_pieces_per_slice 5 20 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tamias_dinner_problem_l2667_266714


namespace NUMINAMATH_CALUDE_total_cost_is_30_l2667_266716

-- Define the cost of silverware
def silverware_cost : ℝ := 20

-- Define the cost of dinner plates as 50% of silverware cost
def dinner_plates_cost : ℝ := silverware_cost * 0.5

-- Theorem: The total cost is $30
theorem total_cost_is_30 : silverware_cost + dinner_plates_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_30_l2667_266716


namespace NUMINAMATH_CALUDE_percentage_problem_l2667_266707

theorem percentage_problem (number : ℝ) (excess : ℝ) (base_percentage : ℝ) (base_number : ℝ) (percentage : ℝ) : 
  number = 6400 →
  excess = 190 →
  base_percentage = 20 →
  base_number = 650 →
  percentage = 5 →
  percentage / 100 * number = base_percentage / 100 * base_number + excess :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l2667_266707


namespace NUMINAMATH_CALUDE_complex_real_condition_l2667_266794

/-- If z = (2+mi)/(1+i) is a real number and m is a real number, then m = 2 -/
theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2667_266794


namespace NUMINAMATH_CALUDE_cube_diagonals_count_l2667_266731

structure Cube where
  vertices : Nat
  edges : Nat

def face_diagonals (c : Cube) : Nat := 12

def space_diagonals (c : Cube) : Nat := 4

def total_diagonals (c : Cube) : Nat := face_diagonals c + space_diagonals c

theorem cube_diagonals_count (c : Cube) (h1 : c.vertices = 8) (h2 : c.edges = 12) :
  total_diagonals c = 16 ∧ face_diagonals c = 12 ∧ space_diagonals c = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonals_count_l2667_266731


namespace NUMINAMATH_CALUDE_series_sum_equals_three_l2667_266796

theorem series_sum_equals_three (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (5 * n - 1) / k^n = 13/4) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_l2667_266796


namespace NUMINAMATH_CALUDE_expression_simplification_l2667_266722

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (a + 1) / (a^2 - 2*a + 1) / (1 + 2 / (a - 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2667_266722


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l2667_266751

theorem smallest_n_for_unique_k : ∃ (k : ℤ), (9:ℚ)/16 < (1:ℚ)/(1+k) ∧ (1:ℚ)/(1+k) < 7/12 ∧
  ∀ (n : ℕ), n > 0 → n < 1 →
    ¬(∃! (k : ℤ), (9:ℚ)/16 < (n:ℚ)/(n+k) ∧ (n:ℚ)/(n+k) < 7/12) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l2667_266751


namespace NUMINAMATH_CALUDE_simplify_fraction_l2667_266744

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (1 - 1 / (x - 1)) / ((x^2 - 2*x) / (x^2 - 1)) = (x + 1) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2667_266744


namespace NUMINAMATH_CALUDE_second_divisor_problem_l2667_266715

theorem second_divisor_problem (x : ℕ) : 
  (210 % 13 = 3) → (210 % x = 7) → (x = 203) :=
by sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l2667_266715


namespace NUMINAMATH_CALUDE_limit_of_exponential_sine_l2667_266798

theorem limit_of_exponential_sine (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ →
    |(2 - x / 3)^(Real.sin (π * x)) - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_exponential_sine_l2667_266798


namespace NUMINAMATH_CALUDE_white_square_area_l2667_266757

theorem white_square_area (cube_edge : ℝ) (blue_paint_area : ℝ) : 
  cube_edge = 12 → 
  blue_paint_area = 432 → 
  (cube_edge ^ 2 * 6 - blue_paint_area) / 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l2667_266757


namespace NUMINAMATH_CALUDE_wishing_pond_problem_l2667_266702

/-- The number of coins each person throws into the pond -/
structure CoinCounts where
  cindy_dimes : ℕ
  eric_quarters : ℕ
  garrick_nickels : ℕ
  ivy_pennies : ℕ

/-- The value of each coin type in cents -/
def coin_values : CoinCounts → ℕ
  | ⟨cd, eq, gn, ip⟩ => cd * 10 + eq * 25 + gn * 5 + ip * 1

/-- The problem statement -/
theorem wishing_pond_problem (coins : CoinCounts) : 
  coins.eric_quarters = 3 →
  coins.garrick_nickels = 8 →
  coins.ivy_pennies = 60 →
  coin_values coins = 200 →
  coins.cindy_dimes = 2 := by
  sorry

#eval coin_values ⟨2, 3, 8, 60⟩

end NUMINAMATH_CALUDE_wishing_pond_problem_l2667_266702


namespace NUMINAMATH_CALUDE_hidden_dots_on_three_dice_l2667_266753

def total_dots_on_die : ℕ := 21

def total_dots_on_three_dice : ℕ := 3 * total_dots_on_die

def visible_faces : List ℕ := [1, 2, 2, 3, 5, 4, 5, 6]

def sum_visible_faces : ℕ := visible_faces.sum

theorem hidden_dots_on_three_dice : 
  total_dots_on_three_dice - sum_visible_faces = 35 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_on_three_dice_l2667_266753


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l2667_266729

/-- Given a configuration of circles where four congruent smaller circles
    are arranged inside a larger circle such that their diameters align
    with the diameter of the larger circle, this theorem states that
    the radius of each smaller circle is one-fourth of the radius of the larger circle. -/
theorem smaller_circle_radius (R : ℝ) (r : ℝ) 
    (h1 : R = 8) -- The radius of the larger circle is 8 meters
    (h2 : 4 * r = R) -- Four smaller circle diameters align with the larger circle diameter
    : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l2667_266729


namespace NUMINAMATH_CALUDE_euros_to_rubles_conversion_l2667_266734

/-- Exchange rate from euros to US dollars -/
def euro_to_usd_rate : ℚ := 12 / 10

/-- Exchange rate from US dollars to rubles -/
def usd_to_ruble_rate : ℚ := 60

/-- Cost of the travel package in euros -/
def travel_package_cost : ℚ := 600

/-- Theorem stating the equivalence of 600 euros to 43200 rubles given the exchange rates -/
theorem euros_to_rubles_conversion :
  (travel_package_cost * euro_to_usd_rate * usd_to_ruble_rate : ℚ) = 43200 := by
  sorry


end NUMINAMATH_CALUDE_euros_to_rubles_conversion_l2667_266734


namespace NUMINAMATH_CALUDE_binomial_sum_condition_l2667_266791

theorem binomial_sum_condition (n : ℕ) (hn : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n →
    (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_condition_l2667_266791


namespace NUMINAMATH_CALUDE_monitor_pixels_l2667_266709

/-- Calculates the total number of pixels on a monitor given its dimensions and DPI. -/
theorem monitor_pixels 
  (width : ℕ) 
  (height : ℕ) 
  (dpi : ℕ) 
  (h1 : width = 21) 
  (h2 : height = 12) 
  (h3 : dpi = 100) : 
  width * dpi * (height * dpi) = 2520000 := by
  sorry

#check monitor_pixels

end NUMINAMATH_CALUDE_monitor_pixels_l2667_266709


namespace NUMINAMATH_CALUDE_fraction_calculation_l2667_266723

theorem fraction_calculation : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2667_266723


namespace NUMINAMATH_CALUDE_min_height_is_six_l2667_266717

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base_side : ℝ
  height : ℝ

/-- The surface area of a rectangular box with square bases -/
def surface_area (d : BoxDimensions) : ℝ :=
  2 * d.base_side^2 + 4 * d.base_side * d.height

/-- The constraint that the height is 3 units greater than the base side -/
def height_constraint (d : BoxDimensions) : Prop :=
  d.height = d.base_side + 3

/-- The constraint that the surface area is at least 90 square units -/
def area_constraint (d : BoxDimensions) : Prop :=
  surface_area d ≥ 90

theorem min_height_is_six :
  ∃ (d : BoxDimensions),
    height_constraint d ∧
    area_constraint d ∧
    d.height = 6 ∧
    ∀ (d' : BoxDimensions),
      height_constraint d' → area_constraint d' → d'.height ≥ d.height :=
by sorry

end NUMINAMATH_CALUDE_min_height_is_six_l2667_266717


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2667_266776

variable (x y a : ℝ)

theorem simplify_expression_1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * x * y := by sorry

theorem simplify_expression_2 (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4) :
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2667_266776


namespace NUMINAMATH_CALUDE_margo_distance_l2667_266741

/-- Represents the total distance Margo traveled in miles -/
def total_distance : ℝ := 2.5

/-- Represents the time Margo took to walk to her friend's house in minutes -/
def walk_time : ℝ := 15

/-- Represents the time Margo took to jog back home in minutes -/
def jog_time : ℝ := 10

/-- Represents Margo's average speed for the entire trip in miles per hour -/
def average_speed : ℝ := 6

/-- Theorem stating that the total distance Margo traveled is 2.5 miles -/
theorem margo_distance :
  total_distance = average_speed * (walk_time + jog_time) / 60 :=
by sorry

end NUMINAMATH_CALUDE_margo_distance_l2667_266741


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2667_266711

/-- The area of a square with adjacent vertices at (1,2) and (5,6) is 32 -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (5, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l2667_266711


namespace NUMINAMATH_CALUDE_luna_kibble_remaining_l2667_266728

/-- The amount of kibble remaining in the bag after feeding Luna for a day -/
def remaining_kibble (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) 
  (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating the remaining amount of kibble in Luna's bag -/
theorem luna_kibble_remaining : 
  remaining_kibble 12 1 1 1 = 7 := by sorry

end NUMINAMATH_CALUDE_luna_kibble_remaining_l2667_266728


namespace NUMINAMATH_CALUDE_union_equals_reals_l2667_266706

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x | x > -1}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l2667_266706


namespace NUMINAMATH_CALUDE_ratio_problem_l2667_266725

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (x + y) = 4 / 5) : 
  x / y = 14 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2667_266725


namespace NUMINAMATH_CALUDE_painting_club_teams_l2667_266773

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem painting_club_teams (n : ℕ) (h : n = 7) : 
  choose n 4 * choose (n - 4) 2 = 105 :=
by sorry

end NUMINAMATH_CALUDE_painting_club_teams_l2667_266773


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2667_266789

theorem quadratic_minimum (x : ℝ) : ∃ m : ℝ, m = 1337 ∧ ∀ x, 5*x^2 - 20*x + 1357 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2667_266789


namespace NUMINAMATH_CALUDE_parallelogram_area_is_288_l2667_266749

/-- Represents a parallelogram ABCD -/
structure Parallelogram where
  AB : ℝ
  BC : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.AB * p.height

theorem parallelogram_area_is_288 (p : Parallelogram) 
  (h1 : p.AB = 24)
  (h2 : p.BC = 30)
  (h3 : p.height = 12) : 
  area p = 288 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_288_l2667_266749


namespace NUMINAMATH_CALUDE_savings_percentage_is_10_percent_l2667_266721

def basic_salary : ℝ := 240
def sales : ℝ := 2500
def commission_rate : ℝ := 0.02
def savings : ℝ := 29

def commission : ℝ := sales * commission_rate
def total_earnings : ℝ := basic_salary + commission

theorem savings_percentage_is_10_percent :
  (savings / total_earnings) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_savings_percentage_is_10_percent_l2667_266721


namespace NUMINAMATH_CALUDE_ab_power_2013_l2667_266720

theorem ab_power_2013 (a b : ℚ) (h : |a - 2| + (2*b + 1)^2 = 0) : (a*b)^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2013_l2667_266720


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2667_266780

theorem trigonometric_identities :
  (2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2) ∧
  (Real.cos (45 * π / 180) * Real.cos (15 * π / 180) - Real.sin (45 * π / 180) * Real.sin (15 * π / 180) = 1 / 2) ∧
  ((Real.tan (77 * π / 180) - Real.tan (32 * π / 180)) / (2 * (1 + Real.tan (77 * π / 180) * Real.tan (32 * π / 180))) = 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_identities_l2667_266780


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2667_266756

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = -4 :=
by
  use -207
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2667_266756


namespace NUMINAMATH_CALUDE_parametric_line_unique_constants_l2667_266792

/-- A line passing through two points with given parametric equations -/
structure ParametricLine where
  a : ℝ
  b : ℝ
  passes_through_P : 0 = 0 + a ∧ 2 = (b/2) * 0 + 1
  passes_through_Q : 1 = 1 + a ∧ 3 = (b/2) * 1 + 1

/-- Theorem stating the unique values of a and b for the given line -/
theorem parametric_line_unique_constants (l : ParametricLine) : l.a = -1 ∧ l.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_unique_constants_l2667_266792


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l2667_266795

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l2667_266795


namespace NUMINAMATH_CALUDE_feeding_sequence_count_l2667_266737

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 5

/-- Calculates the number of possible feeding sequences -/
def feeding_sequences : ℕ := 
  (num_pairs)  -- choices for first female
  * (num_pairs - 1)  -- choices for second male
  * (num_pairs - 1)  -- choices for second female
  * (num_pairs - 2)  -- choices for third male
  * (num_pairs - 2)  -- choices for third female
  * (num_pairs - 3)  -- choices for fourth male
  * (num_pairs - 3)  -- choices for fourth female
  * (num_pairs - 4)  -- choices for fifth male
  * (num_pairs - 4)  -- choices for fifth female

theorem feeding_sequence_count : feeding_sequences = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequence_count_l2667_266737


namespace NUMINAMATH_CALUDE_composition_result_l2667_266769

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

-- State the theorem
theorem composition_result : f (g (f 3)) = 79 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l2667_266769


namespace NUMINAMATH_CALUDE_sum_x_y_equals_three_l2667_266768

def A (x : ℝ) : Set ℝ := {2, x}
def B (x y : ℝ) : Set ℝ := {x*y, 1}

theorem sum_x_y_equals_three (x y : ℝ) : A x = B x y → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_three_l2667_266768


namespace NUMINAMATH_CALUDE_find_other_divisor_l2667_266732

theorem find_other_divisor : ∃ (x : ℕ), x > 1 ∧ 261 % 7 = 2 ∧ 261 % x = 2 ∧ ∀ (y : ℕ), y > 1 → 261 % y = 2 → y = 7 ∨ y = x := by
  sorry

end NUMINAMATH_CALUDE_find_other_divisor_l2667_266732


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_larger_base_l2667_266704

/-- An isosceles trapezoid with given measurements -/
structure IsoscelesTrapezoid where
  leg : ℝ
  smallerBase : ℝ
  diagonal : ℝ
  largerBase : ℝ

/-- The isosceles trapezoid satisfies the given conditions -/
def satisfiesConditions (t : IsoscelesTrapezoid) : Prop :=
  t.leg = 10 ∧ t.smallerBase = 6 ∧ t.diagonal = 14

/-- Theorem: The larger base of the isosceles trapezoid is 16 -/
theorem isosceles_trapezoid_larger_base
  (t : IsoscelesTrapezoid)
  (h : satisfiesConditions t) :
  t.largerBase = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_larger_base_l2667_266704


namespace NUMINAMATH_CALUDE_zhang_hua_cards_l2667_266778

-- Define the variables
variable (x y z : ℕ)

-- State the theorem
theorem zhang_hua_cards :
  (Nat.lcm (Nat.lcm x y) z = 60) →
  (Nat.gcd x y = 4) →
  (Nat.gcd y z = 3) →
  (x = 4 ∨ x = 20) :=
by
  sorry

end NUMINAMATH_CALUDE_zhang_hua_cards_l2667_266778


namespace NUMINAMATH_CALUDE_man_walking_time_l2667_266759

/-- The man's usual time to cover the distance -/
def usual_time : ℝ := 72

/-- The man's usual speed -/
def usual_speed : ℝ := 1

/-- The factor by which the man's speed is reduced -/
def speed_reduction_factor : ℝ := 0.75

/-- The additional time taken when walking at reduced speed -/
def additional_time : ℝ := 24

theorem man_walking_time :
  (usual_speed * usual_time = speed_reduction_factor * usual_speed * (usual_time + additional_time)) →
  usual_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_time_l2667_266759


namespace NUMINAMATH_CALUDE_minimum_order_amount_correct_l2667_266761

/-- The minimum order amount to get a discount at Silvia's bakery -/
def minimum_order_amount : ℝ := 60

/-- The discount percentage offered by the bakery -/
def discount_percentage : ℝ := 0.10

/-- The total cost of Silvia's order before discount -/
def order_cost : ℝ := 2 * 15 + 6 * 3 + 6 * 2

/-- The total cost of Silvia's order after discount -/
def discounted_cost : ℝ := 54

/-- Theorem stating that the minimum order amount to get the discount is correct -/
theorem minimum_order_amount_correct :
  minimum_order_amount = order_cost ∧
  discounted_cost = order_cost * (1 - discount_percentage) :=
sorry

end NUMINAMATH_CALUDE_minimum_order_amount_correct_l2667_266761


namespace NUMINAMATH_CALUDE_difference_of_squares_l2667_266742

theorem difference_of_squares : 525^2 - 475^2 = 50000 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2667_266742


namespace NUMINAMATH_CALUDE_triangle_properties_l2667_266799

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 5 ∧
  t.b^2 + t.c^2 - Real.sqrt 2 * t.b * t.c = 25 ∧
  Real.cos t.B = 3/5

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi/4 ∧ t.c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2667_266799


namespace NUMINAMATH_CALUDE_lenas_muffins_l2667_266738

/-- Represents the cost of a single item -/
structure ItemCost where
  cake : ℚ
  muffin : ℚ
  bagel : ℚ

/-- Represents a purchase of items -/
structure Purchase where
  cakes : ℕ
  muffins : ℕ
  bagels : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (cost : ItemCost) (purchase : Purchase) : ℚ :=
  cost.cake * purchase.cakes + cost.muffin * purchase.muffins + cost.bagel * purchase.bagels

/-- The main theorem to prove -/
theorem lenas_muffins (cost : ItemCost) : 
  let petya := Purchase.mk 1 2 3
  let anya := Purchase.mk 3 0 1
  let kolya := Purchase.mk 0 6 0
  let lena := Purchase.mk 2 0 2
  totalCost cost petya = totalCost cost anya ∧ 
  totalCost cost anya = totalCost cost kolya →
  ∃ n : ℕ, totalCost cost lena = totalCost cost (Purchase.mk 0 n 0) ∧ n = 5 := by
  sorry


end NUMINAMATH_CALUDE_lenas_muffins_l2667_266738


namespace NUMINAMATH_CALUDE_boys_count_l2667_266790

theorem boys_count (total_students : ℕ) (girls_ratio boys_ratio : ℕ) (h1 : total_students = 30) 
  (h2 : girls_ratio = 1) (h3 : boys_ratio = 2) : 
  (total_students * boys_ratio) / (girls_ratio + boys_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_l2667_266790


namespace NUMINAMATH_CALUDE_triangle_side_value_l2667_266786

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧
  t.a + t.c = 4 ∧
  (Real.cos t.B) / (Real.cos t.C) = -t.b / (2 * t.a + t.c)

-- State the theorem
theorem triangle_side_value (t : Triangle) (h : satisfiesConditions t) :
  t.a = 1 ∨ t.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2667_266786


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2667_266770

/-- A cylinder with a square axial section of area 4 has a surface area of 6π -/
theorem cylinder_surface_area (r h : Real) : 
  r * h = 2 →  -- axial section is a square
  r * r = 1 →  -- area of square is 4
  2 * Real.pi * r * r + 2 * Real.pi * r * h = 6 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_cylinder_surface_area_l2667_266770


namespace NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l2667_266788

/-- Given a rectangle and an ellipse with specific properties, prove that the rectangle's perimeter is 8√1003 -/
theorem rectangle_ellipse_perimeter :
  ∀ (x y a b : ℝ),
    -- Rectangle properties
    x > 0 ∧ y > 0 ∧
    x * y = 2006 ∧
    -- Ellipse properties
    a > 0 ∧ b > 0 ∧
    x + y = 2 * a ∧
    x^2 + y^2 = 4 * (a^2 - b^2) ∧
    π * a * b = 2006 * π →
    -- Conclusion: Perimeter of the rectangle
    2 * (x + y) = 8 * Real.sqrt 1003 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_perimeter_l2667_266788


namespace NUMINAMATH_CALUDE_base_representation_of_500_l2667_266746

theorem base_representation_of_500 :
  ∃! b : ℕ, 
    b > 1 ∧ 
    (∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), 
      a₁ < b ∧ a₂ < b ∧ a₃ < b ∧ a₄ < b ∧ a₅ < b ∧
      500 = a₁ * b^4 + a₂ * b^3 + a₃ * b^2 + a₄ * b + a₅) ∧
    b^4 ≤ 500 ∧ 
    500 < b^5 :=
by sorry

end NUMINAMATH_CALUDE_base_representation_of_500_l2667_266746


namespace NUMINAMATH_CALUDE_a_eq_one_necessary_not_sufficient_l2667_266710

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define what it means for two lines to be parallel
def parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, l₁ a x y ↔ l₂ a (k * x) (k * y)

-- State the theorem
theorem a_eq_one_necessary_not_sufficient :
  (∀ a : ℝ, parallel a → a = 1) ∧ ¬(∀ a : ℝ, a = 1 → parallel a) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_necessary_not_sufficient_l2667_266710


namespace NUMINAMATH_CALUDE_sum_of_divisors_231_eq_384_l2667_266733

/-- The sum of the positive whole number divisors of 231 -/
def sum_of_divisors_231 : ℕ := sorry

/-- Theorem stating that the sum of the positive whole number divisors of 231 is 384 -/
theorem sum_of_divisors_231_eq_384 : sum_of_divisors_231 = 384 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_231_eq_384_l2667_266733


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2667_266787

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ i : ℕ, i > 0 → k ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  m = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2667_266787


namespace NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l2667_266754

/-- The number of balloons Sam gave to Fred -/
def balloons_given_to_fred (sam_initial : ℝ) (dan : ℝ) (total_after : ℝ) : ℝ :=
  sam_initial - (total_after - dan)

theorem correct_balloons_given_to_fred :
  balloons_given_to_fred 46.0 16.0 52 = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_given_to_fred_l2667_266754


namespace NUMINAMATH_CALUDE_money_division_l2667_266712

/-- Proves that the total amount of money divided amongst a, b, and c is $400 --/
theorem money_division (a b c : ℝ) : 
  a = (2/3) * (b + c) →   -- a gets 2/3 as much as b and c together
  b = (6/9) * (a + c) →   -- b gets 6/9 as much as a and c together
  a = 160 →               -- The share of a is $160
  a + b + c = 400 := by   -- The total amount is $400
sorry


end NUMINAMATH_CALUDE_money_division_l2667_266712


namespace NUMINAMATH_CALUDE_vector_BC_l2667_266755

/-- Given points A(0,1), B(3,2), and vector AC(-4,-3), prove that vector BC is (-7,-4) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (0, 1) → 
  B = (3, 2) → 
  C.1 - A.1 = -4 → 
  C.2 - A.2 = -3 → 
  (C.1 - B.1, C.2 - B.2) = (-7, -4) := by
sorry


end NUMINAMATH_CALUDE_vector_BC_l2667_266755


namespace NUMINAMATH_CALUDE_horner_method_f_at_3_l2667_266774

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 + 2x^3 + 3x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 2, 3, 1, 1]

theorem horner_method_f_at_3 :
  horner f_coeffs 3 = 36 := by
  sorry

#eval horner f_coeffs 3  -- This should output 36
#eval f 3  -- This should also output 36

end NUMINAMATH_CALUDE_horner_method_f_at_3_l2667_266774


namespace NUMINAMATH_CALUDE_line_y_intercept_l2667_266797

/-- A straight line in the xy-plane with slope 2 passing through (239, 480) has y-intercept 2 -/
theorem line_y_intercept (m : ℝ) (x₀ y₀ b : ℝ) : 
  m = 2 → 
  x₀ = 239 → 
  y₀ = 480 → 
  y₀ = m * x₀ + b → 
  b = 2 := by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2667_266797


namespace NUMINAMATH_CALUDE_ian_painted_48_faces_l2667_266739

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

/-- Theorem stating that the total number of faces painted by Ian is 48 -/
theorem ian_painted_48_faces : total_faces_painted = 48 := by
  sorry

end NUMINAMATH_CALUDE_ian_painted_48_faces_l2667_266739


namespace NUMINAMATH_CALUDE_average_age_combined_l2667_266700

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℝ) (avg_age_parents : ℝ) :
  num_students = 33 →
  num_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  ((num_students : ℝ) * avg_age_students + (num_parents : ℝ) * avg_age_parents) / 
   ((num_students : ℝ) + (num_parents : ℝ)) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l2667_266700


namespace NUMINAMATH_CALUDE_marble_theorem_l2667_266748

def marble_problem (adam mary greg john sarah peter emily : ℚ) : Prop :=
  adam = 29 ∧
  mary = adam - 11 ∧
  greg = adam + 14 ∧
  john = 2 * mary ∧
  sarah = greg - 7 ∧
  peter = 3 * adam ∧
  emily = (mary + greg) / 2 ∧
  peter + john + sarah - (adam + mary + greg + emily) = 38.5

theorem marble_theorem :
  ∀ adam mary greg john sarah peter emily : ℚ,
  marble_problem adam mary greg john sarah peter emily :=
by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l2667_266748


namespace NUMINAMATH_CALUDE_cindy_marbles_problem_l2667_266760

def friends_given_marbles (initial_marbles : ℕ) (marbles_per_friend : ℕ) (remaining_marbles_multiplier : ℕ) (final_multiplied_marbles : ℕ) : ℕ :=
  (initial_marbles - final_multiplied_marbles / remaining_marbles_multiplier) / marbles_per_friend

theorem cindy_marbles_problem :
  friends_given_marbles 500 80 4 720 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_problem_l2667_266760


namespace NUMINAMATH_CALUDE_yuna_has_most_apples_l2667_266743

def jungkook_apples : ℚ := 6 / 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yuna_has_most_apples : 
  (jungkook_apples : ℝ) < yuna_apples ∧ yoongi_apples < yuna_apples :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_most_apples_l2667_266743


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2667_266718

theorem purely_imaginary_complex_number (a : ℝ) : 
  (((a^2 - 3*a + 2) : ℂ) + (a - 1)*I = (0 : ℂ) + ((a - 1)*I)) → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2667_266718


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2667_266765

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 8*a*x + 21 < 0 ↔ -7 < x ∧ x < -1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2667_266765


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l2667_266705

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3 / 8) * x + 15 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l2667_266705


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2667_266771

theorem geometric_sequence_problem (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, 140 * r = a ∧ a * r = 45 / 28) →
  a = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2667_266771


namespace NUMINAMATH_CALUDE_largest_divisor_of_cube_divisible_by_127_l2667_266766

theorem largest_divisor_of_cube_divisible_by_127 (n : ℕ+) 
  (h : 127 ∣ n^3) : 
  ∀ m : ℕ+, m ∣ n → m ≤ 127 := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_cube_divisible_by_127_l2667_266766


namespace NUMINAMATH_CALUDE_circle_intersection_m_range_l2667_266747

theorem circle_intersection_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 10*y + 1 = 0 ∧ x^2 + y^2 - 2*x + 2*y - m = 0) →
  -1 < m ∧ m < 79 := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_m_range_l2667_266747


namespace NUMINAMATH_CALUDE_typing_time_proportional_l2667_266775

/-- Given that 450 characters can be typed in 9 minutes, 
    prove that 1800 characters can be typed in 36 minutes. -/
theorem typing_time_proportional 
  (chars_per_9min : ℕ) 
  (h_chars : chars_per_9min = 450) :
  (1800 : ℝ) / (36 : ℝ) = (chars_per_9min : ℝ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_typing_time_proportional_l2667_266775


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2667_266783

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) →
  C * D + C = 25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2667_266783


namespace NUMINAMATH_CALUDE_tall_students_not_well_defined_other_options_well_defined_l2667_266713

-- Define a type for potential sets
inductive PotentialSet
  | NaturalNumbers1to20
  | AllRectangles
  | NaturalNumbersLessThan10
  | TallStudents

-- Define a predicate for well-defined sets
def isWellDefinedSet (s : PotentialSet) : Prop :=
  match s with
  | PotentialSet.NaturalNumbers1to20 => true
  | PotentialSet.AllRectangles => true
  | PotentialSet.NaturalNumbersLessThan10 => true
  | PotentialSet.TallStudents => false

-- Theorem stating that "Tall students" is not a well-defined set
theorem tall_students_not_well_defined :
  ¬(isWellDefinedSet PotentialSet.TallStudents) :=
by sorry

-- Theorem stating that other options are well-defined sets
theorem other_options_well_defined :
  (isWellDefinedSet PotentialSet.NaturalNumbers1to20) ∧
  (isWellDefinedSet PotentialSet.AllRectangles) ∧
  (isWellDefinedSet PotentialSet.NaturalNumbersLessThan10) :=
by sorry

end NUMINAMATH_CALUDE_tall_students_not_well_defined_other_options_well_defined_l2667_266713


namespace NUMINAMATH_CALUDE_division_problem_l2667_266750

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  dividend = divisor * quotient + remainder →
  dividend = 86 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2667_266750


namespace NUMINAMATH_CALUDE_blacksmith_horseshoe_solution_l2667_266730

/-- Represents the blacksmith's horseshoe problem --/
def horseshoe_problem (total_iron kg_per_horseshoe : ℕ)
  (num_farms horses_per_farm : ℕ)
  (num_stables horses_per_stable : ℕ)
  (horseshoes_per_horse : ℕ) : ℕ :=
  let total_horseshoes := total_iron / kg_per_horseshoe
  let farm_horses := num_farms * horses_per_farm
  let stable_horses := num_stables * horses_per_stable
  let total_order_horses := farm_horses + stable_horses
  let horseshoes_for_orders := total_order_horses * horseshoes_per_horse
  let remaining_horseshoes := total_horseshoes - horseshoes_for_orders
  remaining_horseshoes / horseshoes_per_horse

/-- Theorem stating the solution to the blacksmith's horseshoe problem --/
theorem blacksmith_horseshoe_solution :
  horseshoe_problem 400 2 2 2 2 5 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_blacksmith_horseshoe_solution_l2667_266730


namespace NUMINAMATH_CALUDE_complex_square_eq_143_minus_48i_l2667_266727

theorem complex_square_eq_143_minus_48i :
  ∀ z : ℂ, z^2 = -143 - 48*I ↔ z = 2 - 12*I ∨ z = -2 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_143_minus_48i_l2667_266727


namespace NUMINAMATH_CALUDE_sum_five_consecutive_odds_mod_12_l2667_266724

theorem sum_five_consecutive_odds_mod_12 (n : ℕ) : 
  (((2*n + 1) + (2*n + 3) + (2*n + 5) + (2*n + 7) + (2*n + 9)) % 12) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_odds_mod_12_l2667_266724


namespace NUMINAMATH_CALUDE_solve_xy_l2667_266752

def A : Nat := 89252525 -- ... (200-digit number)

def B (x y : Nat) : Nat := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

def digit_from_right (n : Nat) (pos : Nat) : Nat :=
  (n / (10 ^ (pos - 1))) % 10

theorem solve_xy :
  ∀ x y : Nat,
    x < 10 → y < 10 →
    digit_from_right (A * B x y) 53 = 1 →
    digit_from_right (A * B x y) 54 = 0 →
    x = 4 ∧ y = 6 :=
by sorry

end NUMINAMATH_CALUDE_solve_xy_l2667_266752


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l2667_266740

/-- Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
    the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3*x^2 + 9*x + 20 = a*(x + 3/2)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l2667_266740


namespace NUMINAMATH_CALUDE_pencil_count_l2667_266772

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 8 →
  pencils = 48 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l2667_266772


namespace NUMINAMATH_CALUDE_expression_evaluation_l2667_266785

theorem expression_evaluation : (24 * 2 - 6) / ((6 - 2) * 2) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2667_266785


namespace NUMINAMATH_CALUDE_geometric_sequences_with_specific_differences_l2667_266735

/-- Two geometric sequences with the same first term and specific differences between their terms -/
theorem geometric_sequences_with_specific_differences :
  ∃ (a p q : ℚ),
    (a ≠ 0) ∧
    (p ≠ 0) ∧
    (q ≠ 0) ∧
    (a * p - a * q = 5) ∧
    (a * p^2 - a * q^2 = -5/4) ∧
    (a * p^3 - a * q^3 = 35/16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequences_with_specific_differences_l2667_266735
