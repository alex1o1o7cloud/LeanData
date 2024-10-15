import Mathlib

namespace NUMINAMATH_CALUDE_quantities_total_l1174_117407

theorem quantities_total (total_avg : ℝ) (subset1_avg : ℝ) (subset2_avg : ℝ) 
  (h1 : total_avg = 8)
  (h2 : subset1_avg = 4)
  (h3 : subset2_avg = 14)
  (h4 : 3 * subset1_avg + 2 * subset2_avg = 5 * total_avg) : 
  5 = (3 * subset1_avg + 2 * subset2_avg) / total_avg :=
by sorry

end NUMINAMATH_CALUDE_quantities_total_l1174_117407


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1174_117453

noncomputable def f (x : ℝ) : ℝ := x - (Real.exp 1 - 1) * Real.log x

theorem solution_set_of_inequality (x : ℝ) :
  (f (Real.exp x) < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1174_117453


namespace NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l1174_117479

theorem spinsters_to_cats_ratio : 
  ∀ (S C : ℕ),
  S = 22 →
  C = S + 55 →
  (S : ℚ) / C = 2 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_spinsters_to_cats_ratio_l1174_117479


namespace NUMINAMATH_CALUDE_modified_triangle_sum_l1174_117496

/-- Represents the sum of numbers in the nth row of the modified triangular array -/
def f : ℕ → ℕ
  | 0 => 0  -- We define f(0) as 0 to make the function total
  | 1 => 0  -- First row starts with 0
  | (n + 2) => 2 * f (n + 1) + (n + 2) * (n + 2)

theorem modified_triangle_sum : f 100 = 2^100 - 10000 := by
  sorry

end NUMINAMATH_CALUDE_modified_triangle_sum_l1174_117496


namespace NUMINAMATH_CALUDE_problems_per_page_l1174_117486

theorem problems_per_page 
  (math_pages : ℕ) 
  (reading_pages : ℕ) 
  (total_problems : ℕ) 
  (h1 : math_pages = 6) 
  (h2 : reading_pages = 4) 
  (h3 : total_problems = 30) : 
  total_problems / (math_pages + reading_pages) = 3 := by
sorry

end NUMINAMATH_CALUDE_problems_per_page_l1174_117486


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l1174_117433

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of ball colors
inductive BallColor : Type
| Red : BallColor
| Black : BallColor
| White : BallColor

-- Define a distribution of balls to people
def Distribution := Person → BallColor

-- Define the event "A receives the white ball"
def event_A_white (d : Distribution) : Prop :=
  d Person.A = BallColor.White

-- Define the event "B receives the white ball"
def event_B_white (d : Distribution) : Prop :=
  d Person.B = BallColor.White

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (d : Distribution), ¬(event_A_white d ∧ event_B_white d) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l1174_117433


namespace NUMINAMATH_CALUDE_mixture_ratio_change_l1174_117442

/-- Given an initial mixture of milk and water, prove the new ratio after adding water -/
theorem mixture_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) 
  (added_water : ℚ) (new_milk_ratio : ℚ) (new_water_ratio : ℚ) : 
  initial_volume = 60 ∧ 
  initial_milk_ratio = 2 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 60 ∧ 
  new_milk_ratio = 1 ∧ 
  new_water_ratio = 2 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  new_milk_ratio / new_water_ratio = initial_milk / new_water :=
by
  sorry


end NUMINAMATH_CALUDE_mixture_ratio_change_l1174_117442


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1174_117420

/-- Given a circle with area A and circumference C, if A/C = 15, then the radius is 30 -/
theorem circle_radius_from_area_circumference_ratio (A C : ℝ) (h : A / C = 15) :
  ∃ (r : ℝ), A = π * r^2 ∧ C = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l1174_117420


namespace NUMINAMATH_CALUDE_intersection_M_N_l1174_117417

def M : Set ℝ := {x | x^2 - x - 2 = 0}
def N : Set ℝ := {-1, 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1174_117417


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_l1174_117474

theorem negation_of_existence_power_of_two (p : Prop) : 
  (p ↔ ∃ n : ℕ, 2^n > 1000) → 
  (¬p ↔ ∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_l1174_117474


namespace NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l1174_117489

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f(x) = ax^2 + bx + c is always above the x-axis -/
def always_above_x_axis (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > 0

/-- The discriminant condition b^2 - 4ac < 0 -/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

/-- Theorem stating that the discriminant condition is neither sufficient nor necessary 
    for the quadratic function to always be above the x-axis -/
theorem discriminant_neither_sufficient_nor_necessary :
  ¬(∀ a b c : ℝ, discriminant_condition a b c → always_above_x_axis a b c) ∧
  ¬(∀ a b c : ℝ, always_above_x_axis a b c → discriminant_condition a b c) :=
sorry

end NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l1174_117489


namespace NUMINAMATH_CALUDE_z_120_20_bounds_l1174_117471

/-- Z_{2k}^s is the s-th member from the center in the 2k-th row -/
def Z (k : ℕ) (s : ℕ) : ℝ := sorry

/-- w_{2k} is a function of k -/
def w (k : ℕ) : ℝ := sorry

/-- Main theorem: Z_{120}^{20} is bounded between 0.012 and 0.016 -/
theorem z_120_20_bounds :
  0.012 < Z 60 10 ∧ Z 60 10 < 0.016 :=
sorry

end NUMINAMATH_CALUDE_z_120_20_bounds_l1174_117471


namespace NUMINAMATH_CALUDE_triangle_side_length_l1174_117444

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C)) →
  (A = π / 3) →
  (a = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1174_117444


namespace NUMINAMATH_CALUDE_largest_four_digit_binary_is_15_l1174_117419

/-- A binary digit is either 0 or 1 -/
def BinaryDigit : Type := {n : Nat // n = 0 ∨ n = 1}

/-- A four-digit binary number -/
def FourDigitBinary : Type := BinaryDigit × BinaryDigit × BinaryDigit × BinaryDigit

/-- Convert a four-digit binary number to its decimal representation -/
def binaryToDecimal (b : FourDigitBinary) : Nat :=
  b.1.val * 8 + b.2.1.val * 4 + b.2.2.1.val * 2 + b.2.2.2.val

/-- The largest four-digit binary number -/
def largestFourDigitBinary : FourDigitBinary :=
  (⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩)

theorem largest_four_digit_binary_is_15 :
  binaryToDecimal largestFourDigitBinary = 15 := by
  sorry

#eval binaryToDecimal largestFourDigitBinary

end NUMINAMATH_CALUDE_largest_four_digit_binary_is_15_l1174_117419


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l1174_117459

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^17 + i^203 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l1174_117459


namespace NUMINAMATH_CALUDE_a_16_value_l1174_117490

def sequence_a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_16_value : sequence_a 16 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_a_16_value_l1174_117490


namespace NUMINAMATH_CALUDE_earth_central_angle_special_case_l1174_117493

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- The Earth, assumed to be a perfect sphere -/
structure Earth where
  center : Point
  radius : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (earth : Earth) (p1 p2 : EarthPoint) : Real :=
  sorry

theorem earth_central_angle_special_case (earth : Earth) :
  let a : EarthPoint := { latitude := 0, longitude := 100 }
  let b : EarthPoint := { latitude := 30, longitude := -90 }
  centralAngle earth a b = 180 := by sorry

end NUMINAMATH_CALUDE_earth_central_angle_special_case_l1174_117493


namespace NUMINAMATH_CALUDE_negative_765_degrees_conversion_l1174_117446

theorem negative_765_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    (-765 : ℝ) * π / 180 = 2 * k * π + α ∧ 
    0 ≤ α ∧ 
    α < 2 * π ∧ 
    k = -3 ∧ 
    α = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_765_degrees_conversion_l1174_117446


namespace NUMINAMATH_CALUDE_inequality_proof_l1174_117469

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1174_117469


namespace NUMINAMATH_CALUDE_right_triangle_cos_z_l1174_117426

theorem right_triangle_cos_z (X Y Z : Real) (h1 : X + Y + Z = π) (h2 : X = π/2) (h3 : Real.sin Y = 3/5) :
  Real.cos Z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_z_l1174_117426


namespace NUMINAMATH_CALUDE_mrs_hilt_pencils_l1174_117462

/-- The number of pencils Mrs. Hilt can buy -/
def pencils_bought (total_money : ℕ) (cost_per_pencil : ℕ) : ℕ :=
  total_money / cost_per_pencil

/-- Proof that Mrs. Hilt can buy 10 pencils -/
theorem mrs_hilt_pencils :
  pencils_bought 50 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pencils_l1174_117462


namespace NUMINAMATH_CALUDE_root_in_interval_l1174_117401

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 1.5 > 0) → ∃ x, x ∈ Set.Ioo 1 1.5 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1174_117401


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1174_117480

/-- Given an initial solution of 5 liters containing 40% alcohol,
    after adding 2 liters of water and 1 liter of pure alcohol,
    the resulting mixture contains 37.5% alcohol. -/
theorem alcohol_mixture_percentage :
  let initial_volume : ℝ := 5
  let initial_alcohol_percentage : ℝ := 40 / 100
  let water_added : ℝ := 2
  let pure_alcohol_added : ℝ := 1
  let final_volume : ℝ := initial_volume + water_added + pure_alcohol_added
  let final_alcohol_volume : ℝ := initial_volume * initial_alcohol_percentage + pure_alcohol_added
  final_alcohol_volume / final_volume = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1174_117480


namespace NUMINAMATH_CALUDE_other_communities_count_l1174_117410

theorem other_communities_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 46 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ∃ (other_boys : ℕ), other_boys = 136 ∧ 
    (↑other_boys : ℚ) / total_boys = 1 - (muslim_percent + hindu_percent + sikh_percent) :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l1174_117410


namespace NUMINAMATH_CALUDE_sin_cube_identity_l1174_117495

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l1174_117495


namespace NUMINAMATH_CALUDE_hockey_team_ties_l1174_117406

theorem hockey_team_ties (wins ties : ℕ) : 
  wins = ties + 12 →
  2 * wins + ties = 60 →
  ties = 12 := by
sorry

end NUMINAMATH_CALUDE_hockey_team_ties_l1174_117406


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1174_117449

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, (1 / 5 : ℚ) + (n : ℚ) / 8 < 9 / 5 ↔ n ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1174_117449


namespace NUMINAMATH_CALUDE_power_product_equals_one_third_l1174_117418

theorem power_product_equals_one_third :
  (-3 : ℚ)^2022 * (1/3 : ℚ)^2023 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_power_product_equals_one_third_l1174_117418


namespace NUMINAMATH_CALUDE_initial_onions_l1174_117454

theorem initial_onions (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 33) :
  sold + left = 98 := by
  sorry

end NUMINAMATH_CALUDE_initial_onions_l1174_117454


namespace NUMINAMATH_CALUDE_red_balls_count_l1174_117438

theorem red_balls_count (white_balls : ℕ) (ratio_white : ℕ) (ratio_red : ℕ) : 
  white_balls = 16 → ratio_white = 4 → ratio_red = 3 → 
  (white_balls * ratio_red) / ratio_white = 12 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1174_117438


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1174_117411

/-- Proves that for an angle α in the second quadrant, 
    if 2sin(2α) = cos(2α) - 1, then cos(α) = -√5/5 -/
theorem cos_alpha_value (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : 2 * Real.sin (2 * α) = Real.cos (2 * α) - 1) -- given equation
  : Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1174_117411


namespace NUMINAMATH_CALUDE_f_extrema_on_interval_l1174_117431

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 5*x^3 + 1

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem f_extrema_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 153) ∧
  (∃ x ∈ interval, f x = -4) := by
sorry

end NUMINAMATH_CALUDE_f_extrema_on_interval_l1174_117431


namespace NUMINAMATH_CALUDE_apartments_greater_than_scales_l1174_117448

theorem apartments_greater_than_scales (houses : ℕ) (K A P C : ℕ) :
  houses > 0 ∧ K > 0 ∧ A > 0 ∧ P > 0 ∧ C > 0 →  -- All quantities are positive
  K * A * P > A * P * C →                      -- Fish in house > scales in apartment
  K > C                                        -- Apartments in house > scales on fish
  := by sorry

end NUMINAMATH_CALUDE_apartments_greater_than_scales_l1174_117448


namespace NUMINAMATH_CALUDE_lady_eagles_score_l1174_117404

theorem lady_eagles_score (total_points : ℕ) (games : ℕ) (jessie_points : ℕ)
  (h1 : total_points = 311)
  (h2 : games = 5)
  (h3 : jessie_points = 41) :
  total_points - 3 * jessie_points = 188 := by
  sorry

end NUMINAMATH_CALUDE_lady_eagles_score_l1174_117404


namespace NUMINAMATH_CALUDE_quadratic_function_range_quadratic_function_range_restricted_l1174_117478

theorem quadratic_function_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + 2 ≥ a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

theorem quadratic_function_range_restricted (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_quadratic_function_range_restricted_l1174_117478


namespace NUMINAMATH_CALUDE_can_form_triangle_l1174_117402

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 4, 5) can form a triangle. -/
theorem can_form_triangle : triangle_inequality 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l1174_117402


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1174_117465

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1) / 2 = 153) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1174_117465


namespace NUMINAMATH_CALUDE_van_tire_mileage_l1174_117423

/-- Calculates the miles each tire is used given the total miles traveled,
    number of tires, and number of tires used at a time. -/
def miles_per_tire (total_miles : ℕ) (num_tires : ℕ) (tires_in_use : ℕ) : ℚ :=
  (total_miles * tires_in_use : ℚ) / num_tires

/-- Proves that for a van with 7 tires, where 6 are used at a time,
    and the van travels 42,000 miles with all tires equally worn,
    each tire is used for 36,000 miles. -/
theorem van_tire_mileage :
  miles_per_tire 42000 7 6 = 36000 := by sorry

end NUMINAMATH_CALUDE_van_tire_mileage_l1174_117423


namespace NUMINAMATH_CALUDE_total_pages_called_l1174_117482

def pages_last_week : ℝ := 10.2
def pages_this_week : ℝ := 8.6

theorem total_pages_called :
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_called_l1174_117482


namespace NUMINAMATH_CALUDE_clock_angles_l1174_117405

/-- Represents the angle between the hour and minute hands on a clock face -/
def clockAngle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

/-- Definition of a straight angle -/
def isStraightAngle (angle : ℝ) : Prop :=
  angle = 180

/-- Definition of a right angle -/
def isRightAngle (angle : ℝ) : Prop :=
  angle = 90

/-- Definition of an obtuse angle -/
def isObtuseAngle (angle : ℝ) : Prop :=
  90 < angle ∧ angle < 180

theorem clock_angles :
  (isStraightAngle (clockAngle 6 0)) ∧
  (isRightAngle (clockAngle 9 0)) ∧
  (isObtuseAngle (clockAngle 4 0)) :=
by sorry

end NUMINAMATH_CALUDE_clock_angles_l1174_117405


namespace NUMINAMATH_CALUDE_chord_length_is_7_exists_unique_P_l1174_117484

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Define the property that G is the midpoint of GT
def is_midpoint_GT (G T : ℝ × ℝ) : Prop :=
  G.1 = (G.1 + T.1) / 2 ∧ G.2 = (G.2 + T.2) / 2

-- Theorem 1: The length of the chord cut by FG on C₁ is 7
theorem chord_length_is_7 (G : ℝ × ℝ) (T : ℝ × ℝ) :
  C₁ G.1 G.2 →
  line_x_eq_neg_4 T.1 T.2 →
  is_midpoint_GT G T →
  ∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 49 :=
sorry

-- Theorem 2: There exists a unique point P(4, 0) such that |GP| = 2|GF| for all G on C₁
theorem exists_unique_P (P : ℝ × ℝ) :
  P = (4, 0) ↔
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
    (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_is_7_exists_unique_P_l1174_117484


namespace NUMINAMATH_CALUDE_solve_snake_problem_l1174_117460

def snake_problem (total_length : ℝ) (head_ratio : ℝ) : Prop :=
  let head_length := head_ratio * total_length
  let body_length := total_length - head_length
  (head_ratio = 1 / 10) ∧ (total_length = 10) → body_length = 9

theorem solve_snake_problem :
  snake_problem 10 (1 / 10) :=
by sorry

end NUMINAMATH_CALUDE_solve_snake_problem_l1174_117460


namespace NUMINAMATH_CALUDE_binomial_distribution_parameters_l1174_117458

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_parameters :
  ∃ (ξ : BinomialDistribution), expectedValue ξ = 12 ∧ variance ξ = 2.4 ∧ ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_parameters_l1174_117458


namespace NUMINAMATH_CALUDE_dorothy_income_l1174_117467

theorem dorothy_income (annual_income : ℝ) : 
  annual_income * (1 - 0.18) = 49200 → annual_income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_income_l1174_117467


namespace NUMINAMATH_CALUDE_school_trip_seats_l1174_117491

/-- Given a total number of students and buses, calculate the number of seats per bus -/
def seatsPerBus (students : ℕ) (buses : ℕ) : ℚ :=
  (students : ℚ) / (buses : ℚ)

/-- Theorem: Given 14 students and 7 buses, the number of seats on each bus is 2 -/
theorem school_trip_seats :
  seatsPerBus 14 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_seats_l1174_117491


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1174_117473

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 7 ∧ 
    ∃ k : ℕ, min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ subset : Finset (Fin total_crates), subset.card = n ∧
        ∀ i ∈ subset, ∃ f : Fin total_crates → ℕ, 
          (∀ j, min_oranges ≤ f j ∧ f j ≤ max_oranges) ∧ f i = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1174_117473


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l1174_117408

/-- A natural number is composite if it has more than two distinct positive divisors -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A sequence of n consecutive composite numbers starting from k -/
def ConsecutiveComposites (k n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → IsComposite (k + i)

theorem consecutive_composites_exist :
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 9) ∧
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l1174_117408


namespace NUMINAMATH_CALUDE_student_tickets_sold_l1174_117497

theorem student_tickets_sold (total_tickets : ℕ) (total_money : ℕ) 
  (student_price : ℕ) (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ), 
    student_tickets + (total_tickets - student_tickets) = total_tickets ∧
    student_price * student_tickets + nonstudent_price * (total_tickets - student_tickets) = total_money ∧
    student_tickets = 530 :=
by
  sorry

#check student_tickets_sold

end NUMINAMATH_CALUDE_student_tickets_sold_l1174_117497


namespace NUMINAMATH_CALUDE_starting_team_combinations_l1174_117463

/-- The number of members in the water polo team -/
def team_size : ℕ := 18

/-- The number of players in the starting team -/
def starting_team_size : ℕ := 7

/-- The number of interchangeable positions -/
def interchangeable_positions : ℕ := 5

/-- The number of ways to choose the starting team -/
def choose_starting_team : ℕ := team_size * (team_size - 1) * (Nat.choose (team_size - 2) interchangeable_positions)

theorem starting_team_combinations :
  choose_starting_team = 1338176 := by
  sorry

end NUMINAMATH_CALUDE_starting_team_combinations_l1174_117463


namespace NUMINAMATH_CALUDE_completing_square_result_l1174_117425

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_completing_square_result_l1174_117425


namespace NUMINAMATH_CALUDE_log_meaningful_iff_in_range_l1174_117409

def meaningful_log (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_meaningful_iff_in_range (a : ℝ) :
  meaningful_log a ↔ (a > 2 ∧ a < 3) ∨ (a > 3 ∧ a < 5) :=
sorry

end NUMINAMATH_CALUDE_log_meaningful_iff_in_range_l1174_117409


namespace NUMINAMATH_CALUDE_train_stop_time_l1174_117457

/-- Proves that a train with given speeds including and excluding stoppages
    stops for 20 minutes per hour. -/
theorem train_stop_time
  (speed_without_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : speed_with_stops = 40)
  : (1 - speed_with_stops / speed_without_stops) * 60 = 20 :=
by sorry

end NUMINAMATH_CALUDE_train_stop_time_l1174_117457


namespace NUMINAMATH_CALUDE_not_q_is_true_l1174_117461

theorem not_q_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_true_l1174_117461


namespace NUMINAMATH_CALUDE_hulk_jumps_theorem_l1174_117466

def jump_sequence (n : ℕ) : ℝ := 4 * (3 : ℝ) ^ (n - 1)

def total_distance (n : ℕ) : ℝ := 2 * ((3 : ℝ) ^ n - 1)

theorem hulk_jumps_theorem :
  (∀ k < 8, total_distance k ≤ 5000) ∧ total_distance 8 > 5000 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jumps_theorem_l1174_117466


namespace NUMINAMATH_CALUDE_computer_purchase_cost_effectiveness_l1174_117456

def store_A_cost (x : ℕ) : ℝ := 4500 * x + 1500
def store_B_cost (x : ℕ) : ℝ := 4800 * x

theorem computer_purchase_cost_effectiveness (x : ℕ) :
  (x < 5 → store_B_cost x < store_A_cost x) ∧
  (x > 5 → store_A_cost x < store_B_cost x) ∧
  (x = 5 → store_A_cost x = store_B_cost x) := by
  sorry

end NUMINAMATH_CALUDE_computer_purchase_cost_effectiveness_l1174_117456


namespace NUMINAMATH_CALUDE_odd_function_property_l1174_117476

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (a b : ℝ) :
  let f := fun (x : ℝ) ↦ (a * x + b) / (x^2 + 1)
  IsOdd f ∧ f (1/2) = 2/5 → f 2 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1174_117476


namespace NUMINAMATH_CALUDE_power_of_product_l1174_117470

theorem power_of_product (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1174_117470


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l1174_117485

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits displayed on the watch -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits on a 24-hour format digital watch -/
def maxSumOfDigits : Nat := 23

theorem largest_sum_of_digits :
  ∀ t : Time24, totalSumOfDigits t ≤ maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l1174_117485


namespace NUMINAMATH_CALUDE_no_equal_conversions_l1174_117439

def fahrenheit_to_celsius (f : ℤ) : ℤ :=
  ⌊(5 : ℚ) / 9 * (f - 32)⌋

def celsius_to_fahrenheit (c : ℤ) : ℤ :=
  ⌊(9 : ℚ) / 5 * c + 33⌋

theorem no_equal_conversions :
  ∀ f : ℤ, 34 ≤ f ∧ f ≤ 1024 →
    f ≠ celsius_to_fahrenheit (fahrenheit_to_celsius f) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_conversions_l1174_117439


namespace NUMINAMATH_CALUDE_divisibility_by_81_invariant_under_reversal_l1174_117427

/-- A sequence of digits represented as a list of natural numbers. -/
def DigitSequence := List Nat

/-- Check if a number represented by a digit sequence is divisible by 81. -/
def isDivisibleBy81 (digits : DigitSequence) : Prop :=
  digits.foldl (fun acc d => (10 * acc + d) % 81) 0 = 0

/-- Reverse a digit sequence. -/
def reverseDigits (digits : DigitSequence) : DigitSequence :=
  digits.reverse

theorem divisibility_by_81_invariant_under_reversal
  (digits : DigitSequence)
  (h : digits.length = 2016)
  (h_divisible : isDivisibleBy81 digits) :
  isDivisibleBy81 (reverseDigits digits) := by
  sorry

#check divisibility_by_81_invariant_under_reversal

end NUMINAMATH_CALUDE_divisibility_by_81_invariant_under_reversal_l1174_117427


namespace NUMINAMATH_CALUDE_top_of_second_column_is_20_l1174_117488

/-- Represents a 7x6 grid of numbers -/
def Grid := Fin 7 → Fin 6 → ℤ

/-- The given grid satisfies the problem conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  -- Row is an arithmetic sequence with first element 15 and common difference 0
  (∀ i : Fin 7, g i 0 = 15) ∧
  -- Third column is an arithmetic sequence containing 10 and 5
  (g 2 1 = 10 ∧ g 2 2 = 5) ∧
  -- Second column's bottom element is -10
  (g 1 5 = -10) ∧
  -- Each column is an arithmetic sequence
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 1 (i + 1) = g 1 i + d) ∧
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 2 (i + 1) = g 2 i + d)

/-- The theorem to be proved -/
theorem top_of_second_column_is_20 (g : Grid) (h : satisfies_conditions g) : g 1 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_top_of_second_column_is_20_l1174_117488


namespace NUMINAMATH_CALUDE_solve_for_a_l1174_117451

theorem solve_for_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 20 - 3*a) : a = 20/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1174_117451


namespace NUMINAMATH_CALUDE_solve_equation_l1174_117455

theorem solve_equation : ∃ y : ℚ, 2*y + 3*y = 500 - (4*y + 6*y) → y = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1174_117455


namespace NUMINAMATH_CALUDE_work_completion_time_l1174_117424

/-- Given workers A and B, where A can finish a job in 4 days and B in 14 days,
    prove that after working together for 2 days and A leaving,
    B will take 5 more days to finish the job. -/
theorem work_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (days_together : ℝ) 
  (h1 : days_A = 4) 
  (h2 : days_B = 14) 
  (h3 : days_together = 2) : 
  (days_B - (1 - (days_together * (1 / days_A + 1 / days_B))) / (1 / days_B)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1174_117424


namespace NUMINAMATH_CALUDE_female_democrats_count_l1174_117494

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 750 →
  female + male = total →
  female / 2 + male / 4 = total / 3 →
  female / 2 = 125 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l1174_117494


namespace NUMINAMATH_CALUDE_sand_delivery_theorem_l1174_117483

/-- The amount of sand remaining after a truck's journey -/
def sand_remaining (initial : Real) (loss : Real) : Real :=
  initial - loss

/-- The total amount of sand from all trucks -/
def total_sand (truck1 : Real) (truck2 : Real) (truck3 : Real) : Real :=
  truck1 + truck2 + truck3

theorem sand_delivery_theorem :
  let truck1_initial : Real := 4.1
  let truck1_loss : Real := 2.4
  let truck2_initial : Real := 5.7
  let truck2_loss : Real := 3.6
  let truck3_initial : Real := 8.2
  let truck3_loss : Real := 1.9
  total_sand
    (sand_remaining truck1_initial truck1_loss)
    (sand_remaining truck2_initial truck2_loss)
    (sand_remaining truck3_initial truck3_loss) = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_sand_delivery_theorem_l1174_117483


namespace NUMINAMATH_CALUDE_megan_books_count_l1174_117413

theorem megan_books_count :
  ∀ (m k g : ℕ),
  k = m / 4 →
  g = 2 * k + 9 →
  m + k + g = 65 →
  m = 32 :=
by sorry

end NUMINAMATH_CALUDE_megan_books_count_l1174_117413


namespace NUMINAMATH_CALUDE_kaleb_chocolate_pieces_l1174_117436

theorem kaleb_chocolate_pieces (initial_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given = 5 → pieces_per_box = 6 →
  (initial_boxes - boxes_given) * pieces_per_box = 54 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_chocolate_pieces_l1174_117436


namespace NUMINAMATH_CALUDE_minimize_sum_of_squares_l1174_117487

/-- The quadratic equation in x has only integer roots -/
def has_integer_roots (k : ℚ) : Prop :=
  ∃ x₁ x₂ : ℤ, k * x₁^2 + (3 - 3*k) * x₁ + (2*k - 6) = 0 ∧
              k * x₂^2 + (3 - 3*k) * x₂ + (2*k - 6) = 0

/-- The quadratic equation in y has two positive integer roots -/
def has_positive_integer_roots (k t : ℚ) : Prop :=
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁ ≠ y₂

theorem minimize_sum_of_squares (k t : ℚ) :
  has_integer_roots k →
  has_positive_integer_roots k t →
  (k = 3/4 ∧ t = 15) →
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁^2 + y₂^2 = 8 ∧
              ∀ y₁' y₂' : ℕ, (k + 3) * y₁'^2 - 15 * y₁' + t = 0 →
                             (k + 3) * y₂'^2 - 15 * y₂' + t = 0 →
                             y₁'^2 + y₂'^2 ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_of_squares_l1174_117487


namespace NUMINAMATH_CALUDE_goods_train_length_l1174_117481

/-- The length of a goods train passing a man in an opposite moving train -/
theorem goods_train_length (man_train_speed goods_train_speed : ℝ) (passing_time : ℝ) : 
  man_train_speed = 20 →
  goods_train_speed = 92 →
  passing_time = 9 →
  ∃ (length : ℝ), abs (length - 279.99) < 0.01 ∧ 
    length = (man_train_speed + goods_train_speed) * (5/18) * passing_time :=
by
  sorry

#check goods_train_length

end NUMINAMATH_CALUDE_goods_train_length_l1174_117481


namespace NUMINAMATH_CALUDE_triangle_inequality_l1174_117464

/-- For any triangle ABC with sides a, b, and c, the sum of squares of the sides
    is greater than or equal to 4√3 times the area of the triangle. -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let S := Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1174_117464


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1174_117429

/-- Given that x varies as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1174_117429


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1174_117437

theorem solve_linear_equation (x : ℝ) :
  3 * x - 4 * x + 5 * x = 140 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1174_117437


namespace NUMINAMATH_CALUDE_maria_apple_sales_l1174_117472

/-- Calculate the average revenue per hour for Maria's apple sales -/
theorem maria_apple_sales (a1 a2 b1 b2 pa1 pa2 pb : ℕ) 
  (h1 : a1 = 10) -- kg of type A apples sold in first hour
  (h2 : a2 = 2)  -- kg of type A apples sold in second hour
  (h3 : b1 = 5)  -- kg of type B apples sold in first hour
  (h4 : b2 = 3)  -- kg of type B apples sold in second hour
  (h5 : pa1 = 3) -- price of type A apples in first hour
  (h6 : pa2 = 4) -- price of type A apples in second hour
  (h7 : pb = 2)  -- price of type B apples (constant)
  : (a1 * pa1 + b1 * pb + a2 * pa2 + b2 * pb) / 2 = 27 := by
  sorry

#check maria_apple_sales

end NUMINAMATH_CALUDE_maria_apple_sales_l1174_117472


namespace NUMINAMATH_CALUDE_road_repair_equivalence_l1174_117435

/-- The number of persons in the first group -/
def first_group : ℕ := 36

/-- The number of days to complete the work -/
def days : ℕ := 12

/-- The number of hours worked per day by the first group -/
def hours_first : ℕ := 5

/-- The number of hours worked per day by the second group -/
def hours_second : ℕ := 6

/-- The number of persons in the second group -/
def second_group : ℕ := 30

theorem road_repair_equivalence :
  first_group * days * hours_first = second_group * days * hours_second :=
sorry

end NUMINAMATH_CALUDE_road_repair_equivalence_l1174_117435


namespace NUMINAMATH_CALUDE_remaining_jelly_beans_l1174_117443

/-- Represents the distribution of jelly beans based on ID endings -/
structure JellyBeanDistribution :=
  (group1 : Nat) (group2 : Nat) (group3 : Nat)
  (group4 : Nat) (group5 : Nat) (group6 : Nat)

/-- Calculates the total number of jelly beans drawn -/
def totalJellyBeansDrawn (dist : JellyBeanDistribution) : Nat :=
  dist.group1 * 2 + dist.group2 * 4 + dist.group3 * 6 +
  dist.group4 * 8 + dist.group5 * 10 + dist.group6 * 12

/-- Theorem stating the number of remaining jelly beans -/
theorem remaining_jelly_beans
  (initial_jelly_beans : Nat)
  (total_children : Nat)
  (allowed_percentage : Rat)
  (dist : JellyBeanDistribution) :
  initial_jelly_beans = 2000 →
  total_children = 100 →
  allowed_percentage = 70 / 100 →
  dist.group1 = 9 →
  dist.group2 = 25 →
  dist.group3 = 20 →
  dist.group4 = 15 →
  dist.group5 = 15 →
  dist.group6 = 14 →
  initial_jelly_beans - totalJellyBeansDrawn dist = 1324 := by
  sorry

end NUMINAMATH_CALUDE_remaining_jelly_beans_l1174_117443


namespace NUMINAMATH_CALUDE_baker_cakes_l1174_117450

theorem baker_cakes (initial : ℕ) (sold : ℕ) (bought : ℕ) :
  initial ≥ sold →
  initial - sold + bought = initial + bought - sold :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_l1174_117450


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1174_117445

def a (n : ℕ) : ℚ := (2 * n - 1) / (2 - 3 * n)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-2/3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1174_117445


namespace NUMINAMATH_CALUDE_soda_price_ratio_l1174_117403

/-- Represents the volume and price of a soda brand relative to Brand Y -/
structure SodaBrand where
  volume : ℚ  -- Relative volume compared to Brand Y
  price : ℚ   -- Relative price compared to Brand Y

/-- Calculates the unit price of a soda brand -/
def unitPrice (brand : SodaBrand) : ℚ :=
  brand.price / brand.volume

theorem soda_price_ratio :
  let brand_x : SodaBrand := { volume := 13/10, price := 17/20 }
  let brand_z : SodaBrand := { volume := 14/10, price := 11/10 }
  (unitPrice brand_z) / (unitPrice brand_x) = 13/11 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l1174_117403


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1174_117400

/-- The set of real numbers a for which the system of equations has at least one solution -/
def SolutionSet : Set ℝ :=
  {a | ∃ x y, x - 1 = a * (y^3 - 1) ∧
               2 * x / (|y^3| + y^3) = Real.sqrt x ∧
               y > 0 ∧
               x ≥ 0}

/-- Theorem stating that the SolutionSet is equal to the union of three intervals -/
theorem solution_set_characterization :
  SolutionSet = {a | a < 0} ∪ {a | 0 ≤ a ∧ a ≤ 1} ∪ {a | a > 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1174_117400


namespace NUMINAMATH_CALUDE_number_of_hydroxide_groups_l1174_117434

/-- The atomic weight of aluminum -/
def atomic_weight_Al : ℝ := 27

/-- The molecular weight of a hydroxide group -/
def molecular_weight_OH : ℝ := 17

/-- The molecular weight of the compound Al(OH)n -/
def molecular_weight_compound : ℝ := 78

/-- The number of hydroxide groups in the compound -/
def n : ℕ := sorry

/-- Theorem stating that the number of hydroxide groups in Al(OH)n is 3 -/
theorem number_of_hydroxide_groups :
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_number_of_hydroxide_groups_l1174_117434


namespace NUMINAMATH_CALUDE_chessboard_coverage_impossible_l1174_117468

/-- Represents the type of L-shaped block -/
inductive LBlockType
  | Type1  -- Covers 3 white squares and 1 black square
  | Type2  -- Covers 3 black squares and 1 white square

/-- Represents the chessboard coverage problem -/
def ChessboardCoverage (n m : ℕ) (square_blocks : ℕ) (l_blocks : ℕ) : Prop :=
  ∃ (x : ℕ),
    -- Total number of white squares covered
    square_blocks * 2 + 3 * x + 1 * (l_blocks - x) = n * m / 2 ∧
    -- Total number of black squares covered
    square_blocks * 2 + 1 * x + 3 * (l_blocks - x) = n * m / 2 ∧
    -- x is the number of Type1 L-blocks, and should not exceed total L-blocks
    x ≤ l_blocks

/-- Theorem stating the impossibility of covering the 18x8 chessboard -/
theorem chessboard_coverage_impossible :
  ¬ ChessboardCoverage 18 8 9 7 :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_impossible_l1174_117468


namespace NUMINAMATH_CALUDE_domain_of_f_squared_l1174_117430

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x - 2)
def domain_f_shifted : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem domain_of_f_squared (h : ∀ x ∈ domain_f_shifted, f (x - 2) = f (x - 2)) :
  {x : ℝ | ∃ y, f (y^2) = x} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_squared_l1174_117430


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1174_117452

/-- The line equation x*cos(θ) + y*sin(θ) + a = 0 intersects the circle x^2 + y^2 = a^2 at exactly one point -/
theorem line_circle_intersection (θ a : ℝ) :
  ∃! p : ℝ × ℝ, 
    (p.1 * Real.cos θ + p.2 * Real.sin θ + a = 0) ∧ 
    (p.1^2 + p.2^2 = a^2) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1174_117452


namespace NUMINAMATH_CALUDE_cylinder_radius_and_volume_l1174_117440

/-- Properties of a cylinder with given height and surface area -/
def Cylinder (h : ℝ) (s : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ h > 0 ∧ s = 2 * Real.pi * r * h + 2 * Real.pi * r^2

theorem cylinder_radius_and_volume 
  (h : ℝ) (s : ℝ) 
  (hh : h = 8) (hs : s = 130 * Real.pi) : 
  ∃ (r v : ℝ), Cylinder h s ∧ r = 5 ∧ v = 200 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_and_volume_l1174_117440


namespace NUMINAMATH_CALUDE_parametric_equations_form_circle_parametric_equations_part_of_circle_l1174_117412

noncomputable def parametricCircle (θ : Real) : Real × Real :=
  (4 - Real.cos θ, 1 - Real.sin θ)

theorem parametric_equations_form_circle (θ : Real) 
  (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) : 
  let (x, y) := parametricCircle θ
  (x - 4)^2 + (y - 1)^2 = 1 := by
sorry

theorem parametric_equations_part_of_circle :
  ∃ (a b r : Real), 
    (∀ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
      let (x, y) := parametricCircle θ
      (x - a)^2 + (y - b)^2 = r^2) ∧
    (∃ θ₁ θ₂, 0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ Real.pi / 2 ∧ 
      parametricCircle θ₁ ≠ parametricCircle θ₂) := by
sorry

end NUMINAMATH_CALUDE_parametric_equations_form_circle_parametric_equations_part_of_circle_l1174_117412


namespace NUMINAMATH_CALUDE_fiftieth_term_divisible_by_five_l1174_117499

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem fiftieth_term_divisible_by_five : 
  5 ∣ modifiedLucas 49 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_divisible_by_five_l1174_117499


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_144_squared_l1174_117441

theorem largest_prime_divisor_of_17_squared_plus_144_squared :
  (Nat.factors (17^2 + 144^2)).maximum? = some 29 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_17_squared_plus_144_squared_l1174_117441


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1174_117421

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a3 : a 3 = 7) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1174_117421


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1174_117498

theorem unique_solution_3x_4y_5z : 
  ∀ x y z : ℕ+, 3^(x:ℕ) + 4^(y:ℕ) = 5^(z:ℕ) → x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l1174_117498


namespace NUMINAMATH_CALUDE_expansion_simplification_l1174_117415

theorem expansion_simplification (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x^2 + 5 * x^3 - 2 / 3) = 3 / x^2 + 15 * x^3 / 4 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l1174_117415


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l1174_117492

theorem not_divisible_by_121 (n : ℤ) : ¬(121 ∣ (n^2 + 3*n + 5)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l1174_117492


namespace NUMINAMATH_CALUDE_spatial_relationship_l1174_117414

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (para_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem spatial_relationship 
  (m l : Line) 
  (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : perp m α) 
  (h4 : para l β) 
  (h5 : para_planes α β) : 
  perp_lines m l :=
sorry

end NUMINAMATH_CALUDE_spatial_relationship_l1174_117414


namespace NUMINAMATH_CALUDE_james_chores_time_l1174_117422

/-- Proves that James spends 12 hours on his chores given the conditions -/
theorem james_chores_time (vacuum_time : ℝ) (other_chores_factor : ℝ) : 
  vacuum_time = 3 →
  other_chores_factor = 3 →
  vacuum_time + (other_chores_factor * vacuum_time) = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_chores_time_l1174_117422


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l1174_117447

theorem max_value_of_linear_combination (x y : ℝ) : 
  x^2 + y^2 = 16*x + 8*y + 8 → (∀ a b : ℝ, 4*x + 3*y ≤ 64) ∧ (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 8 ∧ 4*x₀ + 3*y₀ = 64) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l1174_117447


namespace NUMINAMATH_CALUDE_correct_ages_are_valid_correct_ages_are_unique_l1174_117432

/-- Represents the ages of a family in 1978 -/
structure FamilyAges where
  son : Nat
  daughter : Nat
  mother : Nat
  father : Nat

/-- Checks if the given ages satisfy the problem conditions -/
def validAges (ages : FamilyAges) : Prop :=
  ages.son < 21 ∧
  ages.daughter < 21 ∧
  ages.son ≠ ages.daughter ∧
  ages.father = ages.mother + 8 ∧
  ages.son^3 + ages.daughter^2 > 1900 ∧
  ages.son^3 + ages.daughter^2 < 1978 ∧
  ages.son^3 + ages.daughter^2 + ages.father = 1978

/-- The correct ages of the family members -/
def correctAges : FamilyAges :=
  { son := 12
  , daughter := 14
  , mother := 46
  , father := 54 }

/-- Theorem stating that the correct ages satisfy the problem conditions -/
theorem correct_ages_are_valid : validAges correctAges := by
  sorry

/-- Theorem stating that the correct ages are the only solution -/
theorem correct_ages_are_unique : ∀ ages : FamilyAges, validAges ages → ages = correctAges := by
  sorry

end NUMINAMATH_CALUDE_correct_ages_are_valid_correct_ages_are_unique_l1174_117432


namespace NUMINAMATH_CALUDE_rs_length_l1174_117428

/-- Triangle PQR with point S on PR -/
structure TrianglePQR where
  /-- Length of PQ -/
  PQ : ℝ
  /-- Length of QR -/
  QR : ℝ
  /-- Length of PS -/
  PS : ℝ
  /-- Length of QS -/
  QS : ℝ
  /-- PQ equals QR -/
  PQ_eq_QR : PQ = QR
  /-- PQ equals 8 -/
  PQ_eq_8 : PQ = 8
  /-- PS equals 10 -/
  PS_eq_10 : PS = 10
  /-- QS equals 5 -/
  QS_eq_5 : QS = 5

/-- The length of RS in the given triangle configuration is 3.5 -/
theorem rs_length (t : TrianglePQR) : ∃ RS : ℝ, RS = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_rs_length_l1174_117428


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l1174_117477

theorem function_inequality_implies_upper_bound (a : ℝ) :
  (∀ x1 ∈ Set.Icc (1/2 : ℝ) 3, ∃ x2 ∈ Set.Icc 2 3, x1 + 4/x1 ≥ 2^x2 + a) →
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l1174_117477


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1174_117416

/-- Hyperbola with given properties -/
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧
  (∀ x : ℝ, ∃ y : ℝ, y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) ∧
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |Real.sqrt 3 * x - 3 * y| / Real.sqrt 12 = 1)

/-- The equation of the hyperbola with the given properties -/
theorem hyperbola_equation :
  ∀ a b : ℝ, Hyperbola a b → (∀ x y : ℝ, x^2 / 4 - 3 * y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1174_117416


namespace NUMINAMATH_CALUDE_lost_ship_depth_l1174_117475

/-- The depth of a lost ship given the diver's descent rate and time taken --/
theorem lost_ship_depth (descent_rate : ℝ) (time_taken : ℝ) (h1 : descent_rate = 35) (h2 : time_taken = 100) :
  descent_rate * time_taken = 3500 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l1174_117475
