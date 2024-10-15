import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1196_119678

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1196_119678


namespace NUMINAMATH_CALUDE_gcd_15_2015_l1196_119616

theorem gcd_15_2015 : Nat.gcd 15 2015 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15_2015_l1196_119616


namespace NUMINAMATH_CALUDE_parallelogram_area_l1196_119644

def v : Fin 2 → ℝ := ![6, -4]
def w : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; 2 * w 0, 2 * w 1]) = 52 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1196_119644


namespace NUMINAMATH_CALUDE_previous_weekend_earnings_l1196_119651

-- Define the given amounts
def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def pogo_stick_cost : ℕ := 60
def additional_needed : ℕ := 13

-- Define the total earnings for this weekend
def this_weekend_earnings : ℕ := saturday_earnings + sunday_earnings

-- Define the theorem
theorem previous_weekend_earnings :
  pogo_stick_cost - additional_needed - this_weekend_earnings = 20 := by
  sorry

end NUMINAMATH_CALUDE_previous_weekend_earnings_l1196_119651


namespace NUMINAMATH_CALUDE_cube_volume_percentage_l1196_119697

def box_length : ℕ := 8
def box_width : ℕ := 6
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

def box_volume : ℕ := box_length * box_width * box_height

theorem cube_volume_percentage :
  (total_cube_volume : ℚ) / (box_volume : ℚ) * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_percentage_l1196_119697


namespace NUMINAMATH_CALUDE_range_of_m_l1196_119672

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : (B m ∩ A = B m) ↔ m ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1196_119672


namespace NUMINAMATH_CALUDE_min_value_theorem_l1196_119627

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  1/(a-1) + 4/(b-1) ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1196_119627


namespace NUMINAMATH_CALUDE_hotel_beds_count_l1196_119628

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
  (h1 : total_rooms = 13)
  (h2 : two_bed_rooms = 8)
  (h3 : beds_in_two_bed_room = 2)
  (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 :=
by sorry

end NUMINAMATH_CALUDE_hotel_beds_count_l1196_119628


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1196_119689

def original_expression (y : ℝ) : ℝ := 3 * (y^3 - 2*y^2 + 3) - 5 * (y^2 - 4*y + 2)

def simplified_expression (y : ℝ) : ℝ := 3*y^3 - 11*y^2 + 20*y - 1

theorem sum_of_squared_coefficients :
  (3^2 + (-11)^2 + 20^2 + (-1)^2 = 531) ∧
  (∀ y : ℝ, original_expression y = simplified_expression y) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1196_119689


namespace NUMINAMATH_CALUDE_quarterly_to_annual_compound_interest_l1196_119649

/-- Given an annual interest rate of 8% compounded quarterly, 
    prove that it's equivalent to an 8.24% annual rate compounded annually. -/
theorem quarterly_to_annual_compound_interest : 
  let quarterly_rate : ℝ := 0.08 / 4
  let effective_annual_rate : ℝ := (1 + quarterly_rate)^4 - 1
  ∀ ε > 0, |effective_annual_rate - 0.0824| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_quarterly_to_annual_compound_interest_l1196_119649


namespace NUMINAMATH_CALUDE_mary_found_four_eggs_l1196_119688

/-- The number of eggs Mary started with -/
def initial_eggs : ℕ := 27

/-- The number of eggs Mary ended up with -/
def final_eggs : ℕ := 31

/-- The number of eggs Mary found -/
def found_eggs : ℕ := final_eggs - initial_eggs

theorem mary_found_four_eggs : found_eggs = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_found_four_eggs_l1196_119688


namespace NUMINAMATH_CALUDE_negation_of_forall_abs_plus_square_nonnegative_l1196_119662

theorem negation_of_forall_abs_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_abs_plus_square_nonnegative_l1196_119662


namespace NUMINAMATH_CALUDE_a_range_l1196_119641

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem a_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_range_l1196_119641


namespace NUMINAMATH_CALUDE_perspective_triangle_area_l1196_119663

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The perspective plane triangle of an equilateral triangle -/
structure PerspectiveTriangle (et : EquilateralTriangle) where

/-- The area of a triangle -/
def area (t : Type) : ℝ := sorry

/-- The theorem stating the area of the perspective plane triangle -/
theorem perspective_triangle_area (et : EquilateralTriangle) 
  (pt : PerspectiveTriangle et) : 
  area (PerspectiveTriangle et) = Real.sqrt 6 / 16 := by sorry

end NUMINAMATH_CALUDE_perspective_triangle_area_l1196_119663


namespace NUMINAMATH_CALUDE_discount_percentage_l1196_119668

theorem discount_percentage (tshirt_cost pants_cost shoes_cost : ℝ)
  (tshirt_qty pants_qty shoes_qty : ℕ)
  (total_paid : ℝ)
  (h1 : tshirt_cost = 20)
  (h2 : pants_cost = 80)
  (h3 : shoes_cost = 150)
  (h4 : tshirt_qty = 4)
  (h5 : pants_qty = 3)
  (h6 : shoes_qty = 2)
  (h7 : total_paid = 558) :
  (1 - total_paid / (tshirt_cost * tshirt_qty + pants_cost * pants_qty + shoes_cost * shoes_qty)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l1196_119668


namespace NUMINAMATH_CALUDE_problem_solution_l1196_119676

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) :
  a^2 - b^2 + 2*a*b = 64 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1196_119676


namespace NUMINAMATH_CALUDE_long_division_problem_l1196_119603

theorem long_division_problem (quotient remainder divisor dividend : ℕ) : 
  quotient = 2015 → 
  remainder = 0 → 
  divisor = 105 → 
  dividend = quotient * divisor + remainder → 
  dividend = 20685 := by
sorry

end NUMINAMATH_CALUDE_long_division_problem_l1196_119603


namespace NUMINAMATH_CALUDE_unsold_bars_l1196_119670

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_sold : ℕ) :
  total_bars = 13 →
  price_per_bar = 6 →
  total_sold = 42 →
  total_bars - (total_sold / price_per_bar) = 6 := by
sorry

end NUMINAMATH_CALUDE_unsold_bars_l1196_119670


namespace NUMINAMATH_CALUDE_new_drive_free_space_calculation_l1196_119694

/-- Calculates the free space on a new external drive after file operations -/
def new_drive_free_space (initial_free : ℝ) (initial_used : ℝ) (deleted1 : ℝ) (deleted2 : ℝ) (added1 : ℝ) (added2 : ℝ) (new_drive_size : ℝ) : ℝ :=
  new_drive_size - (initial_used - (deleted1 + deleted2) + (added1 + added2))

/-- Theorem stating that the free space on the new drive is 313.5 GB -/
theorem new_drive_free_space_calculation :
  new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500 = 313.5 := by
  sorry

#eval new_drive_free_space 75.8 210.3 34.5 29.7 13 27.4 500

end NUMINAMATH_CALUDE_new_drive_free_space_calculation_l1196_119694


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1196_119699

theorem point_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, (x = m - 1 ∧ 0 = 2 * m + 3)) → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1196_119699


namespace NUMINAMATH_CALUDE_wife_selection_probability_l1196_119619

theorem wife_selection_probability 
  (p_husband : ℝ) 
  (p_only_one : ℝ) 
  (h1 : p_husband = 1/7)
  (h2 : p_only_one = 0.28571428571428575) : 
  ∃ p_wife : ℝ, p_wife = 1/5 ∧ 
  p_only_one = p_husband * (1 - p_wife) + p_wife * (1 - p_husband) :=
sorry

end NUMINAMATH_CALUDE_wife_selection_probability_l1196_119619


namespace NUMINAMATH_CALUDE_function_properties_l1196_119674

noncomputable section

def f (k a x : ℝ) : ℝ := 
  if x ≥ 0 then k * x + k * (1 - a^2) else x^2 + (a^2 - 4*a) * x + (3 - a)^2

theorem function_properties (a : ℝ) 
  (h1 : ∀ (x₁ : ℝ), x₁ ≠ 0 → ∃! (x₂ : ℝ), x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁) :
  ∃ k : ℝ, k = (3 - a)^2 / (1 - a^2) ∧ 0 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1196_119674


namespace NUMINAMATH_CALUDE_combined_distance_is_twelve_l1196_119658

/-- The combined distance walked by two ladies in Central Park -/
def combined_distance (distance_second : ℝ) : ℝ :=
  distance_second + 2 * distance_second

/-- Theorem: The combined distance walked by two ladies is 12 miles -/
theorem combined_distance_is_twelve : combined_distance 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_twelve_l1196_119658


namespace NUMINAMATH_CALUDE_symmetric_line_x_axis_l1196_119643

/-- The equation of a line symmetric to another line with respect to the x-axis -/
theorem symmetric_line_x_axis (a b c : ℝ) :
  (∀ x y, a * x + b * y + c = 0 ↔ a * x - b * y - c = 0) →
  (∀ x y, 3 * x + 4 * y - 5 = 0 ↔ 3 * x - 4 * y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_x_axis_l1196_119643


namespace NUMINAMATH_CALUDE_sheet_width_l1196_119607

theorem sheet_width (w : ℝ) 
  (h1 : w > 0)
  (h2 : (w - 4) * 24 / (w * 30) = 64 / 100) : 
  w = 20 := by sorry

end NUMINAMATH_CALUDE_sheet_width_l1196_119607


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l1196_119667

/-- The coefficient of x^n in the expansion of (x + 1/x)^m -/
def binomialCoeff (m n : ℕ) : ℚ := sorry

/-- The coefficient of x^n in the expansion of (x^2 - a)(x + 1/x)^m -/
def expandedCoeff (m n : ℕ) (a : ℚ) : ℚ := 
  binomialCoeff m (m - n + 2) - a * binomialCoeff m (m - n)

theorem expansion_coefficient_implies_a_value : 
  expandedCoeff 10 6 a = 30 → a = 2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_a_value_l1196_119667


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1196_119626

open Real

theorem function_inequality_implies_parameter_bound 
  (f g : ℝ → ℝ) 
  (h : ∀ x > 0, f x = 2 * x * log x ∧ g x = -x^2 + a * x - 3) 
  (h2 : ∀ x > 0, f x > g x) : 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l1196_119626


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1196_119634

/-- Given vectors a, b, and c in ℝ², prove that if 3a + b is collinear with c, then x = 4 -/
theorem collinear_vectors_x_value (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (-2, 0)) 
  (hb : b = (2, 1)) 
  (hc : c = (x, -1)) 
  (hcollinear : ∃ (k : ℝ), k ≠ 0 ∧ 3 • a + b = k • c) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1196_119634


namespace NUMINAMATH_CALUDE_initial_water_is_11_l1196_119642

/-- Represents the hiking scenario with given conditions -/
structure HikeScenario where
  hikeLength : ℝ
  hikeDuration : ℝ
  leakRate : ℝ
  lastMileConsumption : ℝ
  regularConsumption : ℝ
  remainingWater : ℝ

/-- Calculates the initial amount of water in the canteen -/
def initialWater (scenario : HikeScenario) : ℝ :=
  scenario.regularConsumption * (scenario.hikeLength - 1) +
  scenario.lastMileConsumption +
  scenario.leakRate * scenario.hikeDuration +
  scenario.remainingWater

/-- Theorem stating that the initial amount of water is 11 cups -/
theorem initial_water_is_11 (scenario : HikeScenario) 
  (hLength : scenario.hikeLength = 7)
  (hDuration : scenario.hikeDuration = 3)
  (hLeak : scenario.leakRate = 1)
  (hLastMile : scenario.lastMileConsumption = 3)
  (hRegular : scenario.regularConsumption = 0.5)
  (hRemaining : scenario.remainingWater = 2) :
  initialWater scenario = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_is_11_l1196_119642


namespace NUMINAMATH_CALUDE_ariane_victory_condition_l1196_119666

/-- The game between Ariane and Bérénice -/
def game (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 30 ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    (∀ d : ℕ, d ≥ 2 →
      (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ d ∣ a ∧ d ∣ b) ∨
      (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)))

/-- Ariane's winning condition -/
def ariane_wins (n : ℕ) : Prop :=
  game n ∧
  ∃ (S : Finset ℕ),
    S.card = n ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ 30) ∧
    ∀ d : ℕ, d ≥ 2 →
      ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ¬(d ∣ a ∧ d ∣ b)

/-- The main theorem: Ariane can ensure victory if and only if 1 ≤ n ≤ 11 -/
theorem ariane_victory_condition :
  ∀ n : ℕ, ariane_wins n ↔ (1 ≤ n ∧ n ≤ 11) :=
sorry

end NUMINAMATH_CALUDE_ariane_victory_condition_l1196_119666


namespace NUMINAMATH_CALUDE_square_configuration_angle_l1196_119630

/-- Theorem: In a configuration of three squares attached by their vertices to each other and to two vertical rods,
    where the sum of the white angles equals the sum of the gray angles, and given angles of 30°, 126°, 75°,
    and three 90° angles, the measure of the remaining angle x is 39°. -/
theorem square_configuration_angle (white_angles gray_angles : List ℝ)
  (h1 : white_angles.sum = gray_angles.sum)
  (h2 : white_angles.length = 4)
  (h3 : gray_angles.length = 3)
  (h4 : white_angles.take 3 = [30, 126, 75])
  (h5 : gray_angles = [90, 90, 90]) :
  white_angles[3] = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_configuration_angle_l1196_119630


namespace NUMINAMATH_CALUDE_number_1991_in_32nd_group_l1196_119633

/-- The function that gives the number of elements in the nth group of odd numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The function that gives the sum of elements in the first n groups -/
def sum_of_first_n_groups (n : ℕ) : ℕ := n^2

/-- The theorem stating that 1991 appears in the 32nd group -/
theorem number_1991_in_32nd_group :
  (∀ k < 32, sum_of_first_n_groups k < 1991) ∧
  sum_of_first_n_groups 32 ≥ 1991 := by
  sorry

end NUMINAMATH_CALUDE_number_1991_in_32nd_group_l1196_119633


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1196_119660

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 →
  a + c = 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_sum_l1196_119660


namespace NUMINAMATH_CALUDE_minimum_phrases_to_study_l1196_119609

/-- 
Given a total of 800 French phrases and a required quiz score of 90%,
prove that the minimum number of phrases to study is 720.
-/
theorem minimum_phrases_to_study (total_phrases : ℕ) (required_score : ℚ) : 
  total_phrases = 800 → required_score = 90 / 100 → 
  ⌈(required_score * total_phrases : ℚ)⌉ = 720 := by
sorry

end NUMINAMATH_CALUDE_minimum_phrases_to_study_l1196_119609


namespace NUMINAMATH_CALUDE_cardboard_pins_l1196_119661

/-- Calculates the total number of pins used on a rectangular cardboard -/
def total_pins (length width pins_per_side : ℕ) : ℕ :=
  2 * pins_per_side * (length + width)

/-- Theorem: For a 34 * 14 cardboard with 35 pins per side, the total pins used is 140 -/
theorem cardboard_pins :
  total_pins 34 14 35 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cardboard_pins_l1196_119661


namespace NUMINAMATH_CALUDE_grasshopper_frog_jump_contest_l1196_119664

theorem grasshopper_frog_jump_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 31) 
  (h2 : frog_jump = 35) : 
  grasshopper_jump + frog_jump = 66 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_frog_jump_contest_l1196_119664


namespace NUMINAMATH_CALUDE_fort_blocks_theorem_l1196_119665

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - wallThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  outerVolume - innerVolume

/-- Theorem stating that a fort with given dimensions requires 480 blocks -/
theorem fort_blocks_theorem :
  let fortDimensions : FortDimensions := ⟨15, 8, 6⟩
  let wallThickness : ℕ := 3/2
  blocksNeeded fortDimensions wallThickness = 480 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_theorem_l1196_119665


namespace NUMINAMATH_CALUDE_distance_between_lines_l1196_119650

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The lengths of the three chords created by the parallel lines -/
  chord1 : ℝ
  chord2 : ℝ
  chord3 : ℝ
  /-- The chords are positive -/
  chord1_pos : chord1 > 0
  chord2_pos : chord2 > 0
  chord3_pos : chord3 > 0
  /-- The radius is positive -/
  r_pos : r > 0
  /-- The distance between lines is positive -/
  d_pos : d > 0
  /-- The chords satisfy Stewart's theorem -/
  stewart_theorem1 : (chord1 / 2) ^ 2 * chord1 + (d / 2) ^ 2 * chord1 = (chord1 / 2) * r ^ 2 + (chord1 / 2) * r ^ 2
  stewart_theorem2 : (chord3 / 2) ^ 2 * chord3 + ((3 * d) / 2) ^ 2 * chord3 = (chord3 / 2) * r ^ 2 + (chord3 / 2) * r ^ 2

/-- The main theorem stating that for the given chord lengths, the distance between lines is 6 -/
theorem distance_between_lines (c : CircleWithParallelLines) 
    (h1 : c.chord1 = 40) (h2 : c.chord2 = 40) (h3 : c.chord3 = 36) : c.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_lines_l1196_119650


namespace NUMINAMATH_CALUDE_sequence_existence_and_bound_l1196_119695

theorem sequence_existence_and_bound (a : ℝ) (n : ℕ) :
  ∃! x : ℕ → ℝ, 
    (x 1 - x (n - 1) = 0) ∧ 
    (∀ i ∈ Finset.range n, (x (i - 1) + x i) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i ∈ Finset.range (n + 2), |x i| ≤ |a|) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_and_bound_l1196_119695


namespace NUMINAMATH_CALUDE_plate_price_l1196_119692

/-- Given the conditions of Chenny's purchase, prove that each plate costs $2 -/
theorem plate_price (num_plates : ℕ) (spoon_price : ℚ) (num_spoons : ℕ) (total_paid : ℚ) :
  num_plates = 9 →
  spoon_price = 3/2 →
  num_spoons = 4 →
  total_paid = 24 →
  ∃ (plate_price : ℚ), plate_price * num_plates + spoon_price * num_spoons = total_paid ∧ plate_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_plate_price_l1196_119692


namespace NUMINAMATH_CALUDE_unit_price_ratio_l1196_119647

theorem unit_price_ratio (quantity_B price_B : ℝ) (quantity_B_pos : quantity_B > 0) (price_B_pos : price_B > 0) :
  let quantity_A := 1.3 * quantity_B
  let price_A := 0.85 * price_B
  (price_A / quantity_A) / (price_B / quantity_B) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_unit_price_ratio_l1196_119647


namespace NUMINAMATH_CALUDE_quadratic_roots_of_nine_l1196_119612

theorem quadratic_roots_of_nine (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_of_nine_l1196_119612


namespace NUMINAMATH_CALUDE_jeans_to_janes_money_ratio_l1196_119631

theorem jeans_to_janes_money_ratio (total : ℕ) (jeans_money : ℕ) :
  total = 76 →
  jeans_money = 57 →
  (jeans_money : ℚ) / (total - jeans_money : ℚ) = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jeans_to_janes_money_ratio_l1196_119631


namespace NUMINAMATH_CALUDE_trent_onions_per_pot_l1196_119646

/-- The number of pots of soup Trent is making -/
def num_pots : ℕ := 6

/-- The total number of tears Trent cries -/
def total_tears : ℕ := 16

/-- The ratio of tears to onions -/
def tear_to_onion_ratio : ℚ := 2 / 3

/-- The number of onions Trent needs to chop per pot of soup -/
def onions_per_pot : ℕ := 4

theorem trent_onions_per_pot :
  onions_per_pot * num_pots * tear_to_onion_ratio = total_tears :=
sorry

end NUMINAMATH_CALUDE_trent_onions_per_pot_l1196_119646


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1196_119654

theorem arithmetic_geometric_sequence (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- distinct real numbers
  2 * b = a + c →  -- arithmetic sequence
  (a * b) ^ 2 = a * c * b * c →  -- geometric sequence
  a + b + c = 6 →  -- sum condition
  a = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1196_119654


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_lines_l1196_119629

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (non_overlapping_planes : Plane → Plane → Prop)
variable (non_overlapping_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_lines
  (α β : Plane) (l m : Line)
  (h1 : non_overlapping_planes α β)
  (h2 : non_overlapping_lines l m)
  (h3 : perpendicular l α)
  (h4 : perpendicular m β)
  (h5 : line_parallel l m) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_lines_l1196_119629


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1196_119657

theorem arithmetic_computation : -7 * 3 - (-5 * -4) + (-9 * -6) = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1196_119657


namespace NUMINAMATH_CALUDE_average_of_first_five_multiples_of_five_l1196_119635

theorem average_of_first_five_multiples_of_five :
  let multiples : List ℕ := [5, 10, 15, 20, 25]
  (multiples.sum / multiples.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_five_multiples_of_five_l1196_119635


namespace NUMINAMATH_CALUDE_trig_identity_l1196_119680

theorem trig_identity (α : Real) (h : α ∈ Set.Ioo (-π) (-π/2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  2 / Real.tan α := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1196_119680


namespace NUMINAMATH_CALUDE_cube_sum_product_l1196_119624

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_product_l1196_119624


namespace NUMINAMATH_CALUDE_seventeen_students_earlier_l1196_119687

/-- The number of students who came earlier than Hoseok, given the total number of students and the number of students who came later. -/
def students_earlier (total : ℕ) (later : ℕ) : ℕ :=
  total - later - 1

/-- Theorem stating that 17 students came earlier than Hoseok. -/
theorem seventeen_students_earlier :
  students_earlier 30 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_students_earlier_l1196_119687


namespace NUMINAMATH_CALUDE_only_negative_one_point_one_less_than_negative_one_l1196_119636

theorem only_negative_one_point_one_less_than_negative_one :
  let numbers : List ℝ := [0, 1, -0.9, -1.1]
  ∀ x ∈ numbers, x < -1 ↔ x = -1.1 :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_point_one_less_than_negative_one_l1196_119636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2012_l1196_119677

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -2012
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * seq.a 1 + (n * (n - 1) : ℤ) * (seq.a 2 - seq.a 1) / 2

theorem arithmetic_sequence_sum_2012 (seq : ArithmeticSequence) 
    (h : (sum_n seq 12 / 12 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2) :
    sum_n seq 2012 = -2012 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2012_l1196_119677


namespace NUMINAMATH_CALUDE_parabola_vertex_on_line_l1196_119659

/-- The parabola function -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 10*x + c

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 5

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (c : ℝ) : ℝ := f c vertex_x

/-- The theorem stating that the value of c for which the vertex of the parabola
    y = x^2 - 10x + c lies on the line y = 3 is 28 -/
theorem parabola_vertex_on_line : ∃ c : ℝ, vertex_y c = 3 ∧ c = 28 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_line_l1196_119659


namespace NUMINAMATH_CALUDE_smallest_square_area_l1196_119673

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 3 ∧ r1_height = 5)
  (h2 : r2_width = 4 ∧ r2_height = 6) :
  (max (r1_width + r2_height) (r1_height + r2_width))^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1196_119673


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1196_119653

theorem simplify_fraction_product (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15) =
  ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1196_119653


namespace NUMINAMATH_CALUDE_square_side_length_l1196_119671

theorem square_side_length (circle_area : ℝ) (h1 : circle_area = 100) :
  ∃ (square_side : ℝ), square_side * 4 = circle_area ∧ square_side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1196_119671


namespace NUMINAMATH_CALUDE_S_infinite_l1196_119615

/-- The set of positive integers n for which the number of positive divisors of 2^n - 1 is greater than n -/
def S : Set Nat :=
  {n : Nat | n > 0 ∧ (Nat.divisors (2^n - 1)).card > n}

/-- Theorem stating that the set S is infinite -/
theorem S_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_infinite_l1196_119615


namespace NUMINAMATH_CALUDE_sum_a_b_range_m_solve_equation_l1196_119613

-- Define the system of equations
def system (a b m : ℝ) : Prop :=
  a + 2*b = 4 ∧ 2*a + b = 3 - m

-- Theorem 1: Express a + b in terms of m
theorem sum_a_b (a b m : ℝ) :
  system a b m → a + b = (7 - m) / 3 := by sorry

-- Theorem 2: Find the range of values for m
theorem range_m (a b m : ℝ) :
  system a b m → a - b > -4 → m < 3 := by sorry

-- Theorem 3: Solve the equation for positive integer m
theorem solve_equation (m : ℕ) (x : ℝ) :
  m < 3 → (m * x - (1 - x) / 2 = 5 ↔ x = 11/3 ∨ x = 2.2) := by sorry

end NUMINAMATH_CALUDE_sum_a_b_range_m_solve_equation_l1196_119613


namespace NUMINAMATH_CALUDE_carries_shopping_money_l1196_119605

theorem carries_shopping_money (initial_amount sweater_cost tshirt_cost shoes_cost : ℕ) 
  (h1 : initial_amount = 91)
  (h2 : sweater_cost = 24)
  (h3 : tshirt_cost = 6)
  (h4 : shoes_cost = 11) :
  initial_amount - (sweater_cost + tshirt_cost + shoes_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_carries_shopping_money_l1196_119605


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1196_119608

def total_candies : ℕ := 50
def chewing_gums : ℕ := 15
def assorted_candies : ℕ := 15

theorem chocolate_bars_count :
  total_candies - chewing_gums - assorted_candies = 20 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1196_119608


namespace NUMINAMATH_CALUDE_not_outperformed_iff_ge_five_l1196_119679

/-- A directed graph representing a table tennis tournament. -/
structure TournamentGraph (n : ℕ) where
  (edges : Fin n → Fin n → Prop)
  (complete : ∀ i j : Fin n, i ≠ j → edges i j ∨ edges j i)

/-- Player i is not out-performed by player j. -/
def not_outperformed {n : ℕ} (G : TournamentGraph n) (i j : Fin n) : Prop :=
  ∃ k : Fin n, G.edges i k ∧ ¬G.edges j k

/-- The tournament satisfies the not out-performed condition for all players. -/
def all_not_outperformed (n : ℕ) : Prop :=
  ∃ G : TournamentGraph n, ∀ i j : Fin n, i ≠ j → not_outperformed G i j

/-- The main theorem: the not out-performed condition holds if and only if n ≥ 5. -/
theorem not_outperformed_iff_ge_five :
  ∀ n : ℕ, n ≥ 3 → (all_not_outperformed n ↔ n ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_not_outperformed_iff_ge_five_l1196_119679


namespace NUMINAMATH_CALUDE_stellas_album_has_50_pages_l1196_119648

/-- Calculates the number of pages in Stella's stamp album --/
def stellas_album_pages (stamps_per_first_page : ℕ) (stamps_per_other_page : ℕ) (total_stamps : ℕ) : ℕ :=
  let first_pages := 10
  let stamps_in_first_pages := first_pages * stamps_per_first_page
  let remaining_stamps := total_stamps - stamps_in_first_pages
  let other_pages := remaining_stamps / stamps_per_other_page
  first_pages + other_pages

/-- Theorem stating that Stella's album has 50 pages --/
theorem stellas_album_has_50_pages :
  stellas_album_pages (5 * 30) 50 3500 = 50 := by
  sorry

#eval stellas_album_pages (5 * 30) 50 3500

end NUMINAMATH_CALUDE_stellas_album_has_50_pages_l1196_119648


namespace NUMINAMATH_CALUDE_point_C_range_l1196_119602

def parabola (x y : ℝ) : Prop := y^2 = x + 4

def perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (y3 - y2) = -(x3 - x2) * (x2 - x1)

theorem point_C_range :
  ∀ y : ℝ,
  (∃ y1 : ℝ,
    parabola (y1^2 - 4) y1 ∧
    parabola (y^2 - 4) y ∧
    perpendicular 0 2 (y1^2 - 4) y1 (y^2 - 4) y) →
  y ≤ 0 ∨ y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_point_C_range_l1196_119602


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1196_119622

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1196_119622


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1196_119690

theorem second_polygon_sides (p1 p2 : ℕ) (s : ℝ) :
  p1 = 50 →                          -- First polygon has 50 sides
  p1 * (3 * s) = p2 * s →            -- Same perimeter
  3 * s > 0 →                        -- Positive side length
  p2 = 150 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1196_119690


namespace NUMINAMATH_CALUDE_rectangle_fold_theorem_l1196_119698

/-- Given a rectangle ABCD with AB = 4 and BC = 10, folded along a line through A
    such that A meets CD at point G where DG = 3, and C touches the extension of AB at point E,
    prove that the length of segment CE is 1. -/
theorem rectangle_fold_theorem (A B C D G E : ℝ × ℝ) : 
  let AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DG : ℝ := Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2)
  let CE : ℝ := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AB = 4 →
  BC = 10 →
  DG = 3 →
  (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0 → -- rectangle condition
  (A.1 = G.1 ∧ A.2 = G.2) → -- A meets CD at G
  (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) ≥ 0 → -- C touches extension of AB
  CE = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_theorem_l1196_119698


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1196_119614

/-- Given:
A lends B Rs. 3500
B lends C Rs. 3500 at 11.5% per annum
B's gain over 3 years is Rs. 157.5
Prove: The interest rate at which A lent to B is 10% per annum
-/
theorem interest_rate_calculation (principal : ℝ) (rate_b_to_c : ℝ) (time : ℝ) (gain : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_b_to_c = 11.5)
  (h3 : time = 3)
  (h4 : gain = 157.5)
  (h5 : gain = principal * rate_b_to_c / 100 * time - principal * rate_a_to_b / 100 * time) :
  rate_a_to_b = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1196_119614


namespace NUMINAMATH_CALUDE_inequality_range_l1196_119645

theorem inequality_range (a x : ℝ) : 
  (∀ a, |a| ≤ 1 → x^2 + (a - 6) * x + (9 - 3 * a) > 0) ↔ 
  (x < 2 ∨ x > 4) := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1196_119645


namespace NUMINAMATH_CALUDE_largest_ball_on_torus_l1196_119675

/-- The radius of the largest spherical ball that can be placed on top of a torus -/
def largest_ball_radius (inner_radius outer_radius : ℝ) : ℝ :=
  outer_radius - inner_radius

/-- The torus is formed by revolving a circle with radius 1 centered at (4,0,1) -/
def torus_center_radius : ℝ := 4

/-- The height of the torus center above the table -/
def torus_center_height : ℝ := 1

/-- Theorem: The radius of the largest spherical ball on a torus with inner radius 3 and outer radius 5 is 4 -/
theorem largest_ball_on_torus :
  largest_ball_radius 3 5 = 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_ball_on_torus_l1196_119675


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l1196_119638

theorem sqrt_sum_equals_eight : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eight_l1196_119638


namespace NUMINAMATH_CALUDE_ray_return_characterization_l1196_119617

/-- Represents a point in the triangular grid --/
structure GridPoint where
  a : ℕ
  b : ℕ

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Checks if a GridPoint is on the triangular grid --/
def isOnGrid (p : GridPoint) : Prop :=
  p.a ≡ p.b [MOD 3]

/-- Checks if a line from origin to GridPoint doesn't pass through other grid points --/
def isDirectPath (p : GridPoint) : Prop :=
  Nat.gcd p.a p.b = 1

/-- Calculates the number of bounces for a ray to reach a GridPoint --/
def numberOfBounces (p : GridPoint) : ℕ :=
  2 * (p.a + p.b) - 3

/-- Theorem: Characterization of valid number of bounces for ray to return to A --/
theorem ray_return_characterization (n : ℕ) :
  (∃ (t : EquilateralTriangle) (p : GridPoint), 
    isOnGrid p ∧ isDirectPath p ∧ numberOfBounces p = n) ↔ 
  (n ≡ 1 [MOD 6] ∨ n ≡ 5 [MOD 6]) ∧ n ≠ 5 ∧ n ≠ 17 :=
sorry

end NUMINAMATH_CALUDE_ray_return_characterization_l1196_119617


namespace NUMINAMATH_CALUDE_binomial_coefficient_28_7_l1196_119600

theorem binomial_coefficient_28_7 
  (h1 : Nat.choose 26 3 = 2600)
  (h2 : Nat.choose 26 4 = 14950)
  (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_28_7_l1196_119600


namespace NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l1196_119685

theorem bookstore_new_releases_fraction 
  (total_books : ℕ) 
  (historical_fiction_ratio : ℚ) 
  (historical_fiction_new_release_ratio : ℚ) 
  (other_new_release_ratio : ℚ) 
  (h1 : historical_fiction_ratio = 30 / 100)
  (h2 : historical_fiction_new_release_ratio = 40 / 100)
  (h3 : other_new_release_ratio = 50 / 100)
  (h4 : total_books > 0) :
  let historical_fiction_books := total_books * historical_fiction_ratio
  let historical_fiction_new_releases := historical_fiction_books * historical_fiction_new_release_ratio
  let other_books := total_books - historical_fiction_books
  let other_new_releases := other_books * other_new_release_ratio
  let total_new_releases := historical_fiction_new_releases + other_new_releases
  (historical_fiction_new_releases / total_new_releases : ℚ) = 12 / 47 := by
sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l1196_119685


namespace NUMINAMATH_CALUDE_ancient_chinese_gcd_is_successive_differences_l1196_119691

/-- The algorithm used by ancient Chinese mathematicians to find the GCD of two positive integers -/
def ancient_chinese_gcd_algorithm : Type := sorry

/-- The method of successive differences -/
def successive_differences : Type := sorry

/-- Assertion that the ancient Chinese GCD algorithm is the method of successive differences -/
theorem ancient_chinese_gcd_is_successive_differences : 
  ancient_chinese_gcd_algorithm = successive_differences := by sorry

end NUMINAMATH_CALUDE_ancient_chinese_gcd_is_successive_differences_l1196_119691


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_139_l1196_119611

/-- The first nonzero digit to the right of the decimal point in the decimal representation of 1/139 is 1. -/
theorem first_nonzero_digit_of_one_over_139 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 139 = (n : ℚ) / 10^(d + 1) + (1 : ℚ) / (10 * 10^(d + 1)) + (r : ℚ) / (100 * 10^(d + 1)) ∧ 
  0 ≤ r ∧ r < 10 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_139_l1196_119611


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l1196_119601

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_squared_l1196_119601


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1196_119684

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 29 = 17 % 29 ∧ 
  ∀ (y : ℕ), y > 0 → (5 * y) % 29 = 17 % 29 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1196_119684


namespace NUMINAMATH_CALUDE_opposite_of_sqrt7_minus_3_l1196_119682

theorem opposite_of_sqrt7_minus_3 : 
  -(Real.sqrt 7 - 3) = 3 - Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt7_minus_3_l1196_119682


namespace NUMINAMATH_CALUDE_max_m_value_l1196_119655

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) ∧
  (∀ m' : ℝ, m' > m → 
    ¬((∀ x : ℝ, x < m' → x^2 - 2*x - 8 > 0) ∧ 
      (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m'))) →
  m = 4 := by sorry

end NUMINAMATH_CALUDE_max_m_value_l1196_119655


namespace NUMINAMATH_CALUDE_locus_is_hyperbola_branch_l1196_119620

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define a circle tangent to both circles
def tangentCircle (cx cy r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle1 x y → (x - cx)^2 + (y - cy)^2 = r^2 ∨
                circle2 x y → (x - cx)^2 + (y - cy)^2 = r^2

-- Define the locus of centers
def locusOfCenters (x y : ℝ) : Prop :=
  ∃ (r : ℝ), tangentCircle x y r

-- Theorem statement
theorem locus_is_hyperbola_branch :
  ∃ (a b : ℝ), ∀ (x y : ℝ), locusOfCenters x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_hyperbola_branch_l1196_119620


namespace NUMINAMATH_CALUDE_license_plate_difference_l1196_119640

def california_plates := 26^3 * 10^4
def texas_plates := 26^3 * 10^3

theorem license_plate_difference :
  california_plates - texas_plates = 4553200000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1196_119640


namespace NUMINAMATH_CALUDE_cabbage_distribution_l1196_119625

/-- Given a cabbage patch with 12 rows and 180 total heads of cabbage,
    prove that there are 15 heads of cabbage in each row. -/
theorem cabbage_distribution (rows : ℕ) (total_heads : ℕ) (heads_per_row : ℕ) : 
  rows = 12 → total_heads = 180 → heads_per_row * rows = total_heads → heads_per_row = 15 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_distribution_l1196_119625


namespace NUMINAMATH_CALUDE_x_range_given_sqrt_equality_l1196_119681

theorem x_range_given_sqrt_equality (x : ℝ) :
  Real.sqrt ((5 - x)^2) = x - 5 → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_given_sqrt_equality_l1196_119681


namespace NUMINAMATH_CALUDE_remainder_of_power_minus_ninety_l1196_119669

theorem remainder_of_power_minus_ninety (n : ℕ) : (1 - 90) ^ 10 ≡ 1 [ZMOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_minus_ninety_l1196_119669


namespace NUMINAMATH_CALUDE_division_problem_l1196_119632

theorem division_problem : 
  (1 / 24) / ((1 / 12) - (5 / 16) + (7 / 24) - (2 / 3)) = -(2 / 29) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1196_119632


namespace NUMINAMATH_CALUDE_library_books_not_all_checked_out_l1196_119656

-- Define a type for books
variable (Book : Type)

-- Define a predicate for a book being in the library
variable (in_library : Book → Prop)

-- Define a predicate for a book being checked out
variable (checked_out : Book → Prop)

-- Theorem statement
theorem library_books_not_all_checked_out 
  (h : ¬∀ b : Book, in_library b → checked_out b) :
  (∃ b : Book, in_library b ∧ ¬checked_out b) ∧
  (¬∀ b : Book, in_library b → checked_out b) := by
  sorry

end NUMINAMATH_CALUDE_library_books_not_all_checked_out_l1196_119656


namespace NUMINAMATH_CALUDE_largest_square_tile_l1196_119604

theorem largest_square_tile (board_length board_width tile_size : ℕ) : 
  board_length = 16 →
  board_width = 24 →
  tile_size = Nat.gcd board_length board_width →
  tile_size = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_square_tile_l1196_119604


namespace NUMINAMATH_CALUDE_last_digit_of_product_l1196_119693

def last_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem last_digit_of_product (B : ℤ) : 
  B ≥ 0 ∧ B ≤ 9 →
  (last_digit (287 * 287 + B * B - 2 * 287 * B) = 4 ↔ B = 5 ∨ B = 9) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l1196_119693


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l1196_119618

/-- The number of books in the 'Crazy Silly School' series -/
def total_books : ℕ := 13

/-- The number of books already read -/
def books_read : ℕ := 9

/-- The number of books left to read -/
def books_left : ℕ := 4

/-- Theorem stating that the total number of books is equal to the sum of books read and books left -/
theorem crazy_silly_school_books : 
  total_books = books_read + books_left := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l1196_119618


namespace NUMINAMATH_CALUDE_digit_57_of_21_over_22_l1196_119696

def decimal_representation (n d : ℕ) : ℕ → ℕ
  | 0 => (n * 10 / d) % 10
  | i + 1 => decimal_representation n d i

theorem digit_57_of_21_over_22 :
  decimal_representation 21 22 56 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_57_of_21_over_22_l1196_119696


namespace NUMINAMATH_CALUDE_multiples_count_multiples_of_4_or_5_not_20_l1196_119621

theorem multiples_count : Nat → Nat :=
  fun n => (n / 4 + n / 5 - n / 20)

theorem multiples_of_4_or_5_not_20 (upper_bound : Nat) 
  (h : upper_bound = 1500) : 
  multiples_count upper_bound = 600 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_multiples_of_4_or_5_not_20_l1196_119621


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l1196_119606

theorem min_value_sum_of_squares (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_9 : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l1196_119606


namespace NUMINAMATH_CALUDE_gas_pressure_final_l1196_119652

/-- Given a gas with pressure inversely proportional to volume, prove the final pressure -/
theorem gas_pressure_final (p₀ v₀ v₁ v₂ : ℝ) (h₀ : p₀ > 0) (h₁ : v₀ > 0) (h₂ : v₁ > 0) (h₃ : v₂ > 0)
  (h_initial : p₀ * v₀ = 6 * 3.6)
  (h_v₁ : v₁ = 7.2)
  (h_v₂ : v₂ = 3.6)
  (h_half : v₂ = v₀) :
  ∃ (p₂ : ℝ), p₂ * v₂ = p₀ * v₀ ∧ p₂ = 6 := by
  sorry

#check gas_pressure_final

end NUMINAMATH_CALUDE_gas_pressure_final_l1196_119652


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l1196_119623

theorem lcm_gcf_ratio : 
  (Nat.lcm 252 630) / (Nat.gcd 252 630) = 10 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l1196_119623


namespace NUMINAMATH_CALUDE_inequality_proof_l1196_119686

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : c + d ≤ a) (h6 : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1196_119686


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1196_119683

theorem complex_fraction_simplification : 
  let numerator := (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400)
  let denominator := (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400)
  numerator / denominator = 799 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1196_119683


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1196_119639

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1196_119639


namespace NUMINAMATH_CALUDE_volume_of_extended_box_l1196_119637

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volumeWithinOneUnit (b : Box) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def isRelativelyPrime (a b : ℕ) : Prop := sorry

theorem volume_of_extended_box (m n p : ℕ) :
  (∃ b : Box, b.length = 2 ∧ b.width = 3 ∧ b.height = 6) →
  (∃ v : ℝ, v = volumeWithinOneUnit b) →
  v = (m + n * Real.pi) / p →
  m > 0 ∧ n > 0 ∧ p > 0 →
  isRelativelyPrime n p →
  m + n + p = 364 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_extended_box_l1196_119637


namespace NUMINAMATH_CALUDE_penguin_fish_theorem_l1196_119610

theorem penguin_fish_theorem (fish_counts : List ℕ) : 
  fish_counts.length = 10 ∧ 
  fish_counts.sum = 50 ∧ 
  (∀ x ∈ fish_counts, x > 0) →
  ∃ i j, i ≠ j ∧ i < fish_counts.length ∧ j < fish_counts.length ∧ fish_counts[i]! = fish_counts[j]! := by
  sorry

end NUMINAMATH_CALUDE_penguin_fish_theorem_l1196_119610
