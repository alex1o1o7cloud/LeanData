import Mathlib

namespace NUMINAMATH_CALUDE_lottery_winning_probability_l2460_246028

/-- The number of options for the MagicBall -/
def magicBallOptions : ℕ := 25

/-- The number of options for each TrophyBall -/
def trophyBallOptions : ℕ := 48

/-- The number of TrophyBalls to be selected -/
def trophyBallsToSelect : ℕ := 5

/-- The probability of winning the lottery -/
def winningProbability : ℚ := 1 / 63180547200

theorem lottery_winning_probability :
  1 / (magicBallOptions * (trophyBallOptions.factorial / (trophyBallOptions - trophyBallsToSelect).factorial)) = winningProbability :=
sorry

end NUMINAMATH_CALUDE_lottery_winning_probability_l2460_246028


namespace NUMINAMATH_CALUDE_dry_mixed_fruits_weight_l2460_246007

/-- Calculates the weight of dry mixed fruits after dehydration -/
def weight_dry_mixed_fruits (fresh_grapes fresh_apples : ℝ) 
  (fresh_grapes_water fresh_apples_water : ℝ) : ℝ :=
  (1 - fresh_grapes_water) * fresh_grapes + (1 - fresh_apples_water) * fresh_apples

/-- Theorem: The weight of dry mixed fruits is 188 kg -/
theorem dry_mixed_fruits_weight :
  weight_dry_mixed_fruits 400 300 0.65 0.84 = 188 := by
  sorry

#eval weight_dry_mixed_fruits 400 300 0.65 0.84

end NUMINAMATH_CALUDE_dry_mixed_fruits_weight_l2460_246007


namespace NUMINAMATH_CALUDE_conic_properties_l2460_246076

-- Define the conic section
def conic (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/n = 1}

-- Define the foci for the conic
def foci (n : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the line with slope √3 passing through the left focus
def line_through_focus (n : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the intersection points A and B
def intersection_points (n : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the perimeter of triangle ABF₂
def perimeter_ABF2 (n : ℝ) : ℝ := sorry

-- Define the product PF₁ · PF₂
def focus_product (n : ℝ) (P : ℝ × ℝ) : ℝ := sorry

theorem conic_properties :
  (perimeter_ABF2 (-1) = 12) ∧
  (∀ P ∈ conic 4, focus_product 4 P ≤ 4) ∧
  (∃ P ∈ conic 4, focus_product 4 P = 4) ∧
  (∀ P ∈ conic 4, focus_product 4 P ≥ 1) ∧
  (∃ P ∈ conic 4, focus_product 4 P = 1) := by sorry

end NUMINAMATH_CALUDE_conic_properties_l2460_246076


namespace NUMINAMATH_CALUDE_f_one_intersection_l2460_246079

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (3*a - 2)*x + a - 1

/-- Theorem stating the condition for f(x) to have exactly one intersection with x-axis in (-1,3) -/
theorem f_one_intersection (a : ℝ) : 
  (∃! x : ℝ, x > -1 ∧ x < 3 ∧ f a x = 0) ↔ (a ≤ -1/5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_f_one_intersection_l2460_246079


namespace NUMINAMATH_CALUDE_danny_shorts_count_l2460_246066

/-- Represents the number of clothes washed by Cally and Danny -/
structure ClothesWashed where
  cally_white_shirts : Nat
  cally_colored_shirts : Nat
  cally_shorts : Nat
  cally_pants : Nat
  danny_white_shirts : Nat
  danny_colored_shirts : Nat
  danny_pants : Nat
  total_clothes : Nat

/-- Theorem stating that Danny washed 10 pairs of shorts -/
theorem danny_shorts_count (cw : ClothesWashed)
    (h1 : cw.cally_white_shirts = 10)
    (h2 : cw.cally_colored_shirts = 5)
    (h3 : cw.cally_shorts = 7)
    (h4 : cw.cally_pants = 6)
    (h5 : cw.danny_white_shirts = 6)
    (h6 : cw.danny_colored_shirts = 8)
    (h7 : cw.danny_pants = 6)
    (h8 : cw.total_clothes = 58) :
    ∃ (danny_shorts : Nat), danny_shorts = 10 ∧
    cw.total_clothes = cw.cally_white_shirts + cw.cally_colored_shirts + cw.cally_shorts + cw.cally_pants +
                       cw.danny_white_shirts + cw.danny_colored_shirts + danny_shorts + cw.danny_pants :=
  by sorry


end NUMINAMATH_CALUDE_danny_shorts_count_l2460_246066


namespace NUMINAMATH_CALUDE_domain_transformation_l2460_246016

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -2 < x ∧ x < -1}

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -1 < x ∧ x < -1/2}

-- Theorem statement
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ∈ Set.univ) →
  (∀ x, x ∈ domain_f_2x_plus_1 f ↔ f (2*x + 1) ∈ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l2460_246016


namespace NUMINAMATH_CALUDE_line_reflection_l2460_246040

-- Define the slope of the original line
def k : ℝ := sorry

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x + y = 1

-- Define the original line
def original_line (x y : ℝ) : Prop := y = k * x

-- Define the resulting line after reflection
def reflected_line (x y : ℝ) : Prop := y = (1 / k) * x + (k - 1) / k

-- State the theorem
theorem line_reflection (h1 : k ≠ 0) (h2 : k ≠ -1) :
  ∀ x y : ℝ, reflected_line x y ↔ 
  ∃ x' y' : ℝ, original_line x' y' ∧ 
  reflection_line ((x + x') / 2) ((y + y') / 2) :=
sorry

end NUMINAMATH_CALUDE_line_reflection_l2460_246040


namespace NUMINAMATH_CALUDE_slope_condition_implies_y_value_l2460_246029

/-- Given two points P and Q in a coordinate plane, where P has coordinates (-3, 5) and Q has coordinates (5, y), prove that if the slope of the line through P and Q is -4/3, then y = -17/3. -/
theorem slope_condition_implies_y_value :
  let P : ℝ × ℝ := (-3, 5)
  let Q : ℝ × ℝ := (5, y)
  let slope := (Q.2 - P.2) / (Q.1 - P.1)
  slope = -4/3 → y = -17/3 :=
by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_y_value_l2460_246029


namespace NUMINAMATH_CALUDE_base7_addition_theorem_l2460_246047

/-- Addition of numbers in base 7 -/
def base7_add (a b c : ℕ) : ℕ :=
  (a + b + c) % 7^4

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 7^3) * 7^3 + ((n / 7^2) % 7) * 7^2 + ((n / 7) % 7) * 7 + (n % 7)

theorem base7_addition_theorem :
  base7_add (base7_to_decimal 256) (base7_to_decimal 463) (base7_to_decimal 132) =
  base7_to_decimal 1214 :=
sorry

end NUMINAMATH_CALUDE_base7_addition_theorem_l2460_246047


namespace NUMINAMATH_CALUDE_flyer_multiple_l2460_246027

theorem flyer_multiple (maisie_flyers donna_flyers : ℕ) (h1 : maisie_flyers = 33) (h2 : donna_flyers = 71) :
  ∃ x : ℕ, donna_flyers = 5 + x * maisie_flyers ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_flyer_multiple_l2460_246027


namespace NUMINAMATH_CALUDE_liam_commute_speed_l2460_246089

theorem liam_commute_speed (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) 
  (h1 : distance = 40)
  (h2 : actual_speed = 60)
  (h3 : early_time = 4/60) :
  let ideal_speed := actual_speed - 5
  let actual_time := distance / actual_speed
  let ideal_time := distance / ideal_speed
  ideal_time - actual_time = early_time := by sorry

end NUMINAMATH_CALUDE_liam_commute_speed_l2460_246089


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2460_246003

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) (h3 : x ≤ y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2460_246003


namespace NUMINAMATH_CALUDE_blue_eyed_brunettes_l2460_246031

theorem blue_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) :
  total = 60 →
  blue_eyed_blondes = 20 →
  brunettes = 36 →
  brown_eyed = 23 →
  ∃ (blue_eyed_brunettes : ℕ),
    blue_eyed_brunettes = 17 ∧
    blue_eyed_brunettes + blue_eyed_blondes = total - brown_eyed ∧
    blue_eyed_brunettes + (brunettes - blue_eyed_brunettes) = brown_eyed :=
by sorry

end NUMINAMATH_CALUDE_blue_eyed_brunettes_l2460_246031


namespace NUMINAMATH_CALUDE_distance_between_points_on_parabola_l2460_246081

/-- The distance between two points on a parabola -/
theorem distance_between_points_on_parabola
  (a b c x₁ x₂ : ℝ) :
  let y₁ := a * x₁^2 + b * x₁ + c
  let y₂ := a * x₂^2 + b * x₂ + c
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = |x₂ - x₁| * Real.sqrt (1 + (a * (x₂ + x₁) + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_parabola_l2460_246081


namespace NUMINAMATH_CALUDE_fraction_equality_l2460_246060

theorem fraction_equality (x y : ℚ) (hx : x = 3/5) (hy : y = 7/9) :
  (5*x + 9*y) / (45*x*y) = 10/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2460_246060


namespace NUMINAMATH_CALUDE_train_length_l2460_246083

theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
  (h1 : platform_crossing_time = 39)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 700) :
  ∃ (train_length : ℝ),
    train_length = 600 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2460_246083


namespace NUMINAMATH_CALUDE_max_value_of_f_l2460_246058

def f (x : ℝ) := x^4 - 4*x + 3

theorem max_value_of_f : 
  ∃ (m : ℝ), m = 72 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2460_246058


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2460_246008

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

/-- The theorem statement -/
theorem symmetry_of_point :
  let A : Point := ⟨3, -2⟩
  let A' : Point := ⟨-3, 2⟩
  symmetricToOrigin A A' := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2460_246008


namespace NUMINAMATH_CALUDE_sin_300_degrees_l2460_246097

theorem sin_300_degrees : 
  Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l2460_246097


namespace NUMINAMATH_CALUDE_linear_function_midpoint_property_quadratic_function_midpoint_property_l2460_246098

/-- Linear function property -/
theorem linear_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let f := fun x => a * x + b
  f ((x₁ + x₂) / 2) = (f x₁ + f x₂) / 2 := by sorry

/-- Quadratic function property -/
theorem quadratic_function_midpoint_property (a b x₁ x₂ : ℝ) :
  let g := fun x => x^2 + a * x + b
  g ((x₁ + x₂) / 2) ≤ (g x₁ + g x₂) / 2 := by sorry

end NUMINAMATH_CALUDE_linear_function_midpoint_property_quadratic_function_midpoint_property_l2460_246098


namespace NUMINAMATH_CALUDE_us_flag_stars_l2460_246036

theorem us_flag_stars (stripes : ℕ) (total_shapes : ℕ) : 
  stripes = 13 → 
  total_shapes = 54 → 
  ∃ (stars : ℕ), 
    (stars / 2 - 3 + 2 * stripes + 6 = total_shapes) ∧ 
    (stars = 50) := by
  sorry

end NUMINAMATH_CALUDE_us_flag_stars_l2460_246036


namespace NUMINAMATH_CALUDE_typing_time_together_l2460_246099

/-- Given Meso's and Tyler's typing speeds, calculate the time it takes them to type 40 pages together -/
theorem typing_time_together 
  (meso_pages : ℕ) (meso_time : ℕ) (tyler_pages : ℕ) (tyler_time : ℕ) (total_pages : ℕ) :
  meso_pages = 15 →
  meso_time = 5 →
  tyler_pages = 15 →
  tyler_time = 3 →
  total_pages = 40 →
  (total_pages : ℚ) / ((meso_pages : ℚ) / (meso_time : ℚ) + (tyler_pages : ℚ) / (tyler_time : ℚ)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_together_l2460_246099


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2460_246088

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x - y + b₁ = 0 ↔ m₂ * x + y + b₂ = 0) ↔ m₁ = -m₂

/-- The value of a when two lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 2 * x - y + 1 = 0 ↔ x + a * y + 2 = 0) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2460_246088


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2460_246026

theorem complex_number_quadrant : 
  let z : ℂ := (3 + Complex.I) * (1 - Complex.I)
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2460_246026


namespace NUMINAMATH_CALUDE_factorization_proof_l2460_246018

theorem factorization_proof (a : ℝ) : 
  45 * a^2 + 135 * a + 90 * a^3 = 45 * a * (90 * a^2 + a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2460_246018


namespace NUMINAMATH_CALUDE_special_sale_savings_l2460_246056

/-- Given a special sale where 25 tickets can be purchased for the price of 21.5 tickets,
    prove that buying 50 tickets at this rate results in a 14% savings compared to the original price. -/
theorem special_sale_savings : ∀ (P : ℝ), P > 0 →
  let sale_price : ℝ := 21.5 * P / 25
  let original_price_50 : ℝ := 50 * P
  let sale_price_50 : ℝ := 50 * sale_price
  let savings : ℝ := original_price_50 - sale_price_50
  let savings_percentage : ℝ := savings / original_price_50 * 100
  savings_percentage = 14 := by
  sorry

end NUMINAMATH_CALUDE_special_sale_savings_l2460_246056


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2460_246055

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c)
  (h4 : a^2 + b^2 = c^2) :
  (1/a) + (1/b) + (1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2460_246055


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l2460_246041

theorem angle_sum_around_point (y : ℝ) : 
  6 * y + 7 * y + 3 * y + 2 * y = 360 → y = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l2460_246041


namespace NUMINAMATH_CALUDE_min_value_z_l2460_246059

theorem min_value_z (x y : ℝ) : 3 * x^2 + 5 * y^2 + 8 * x - 10 * y + 4 * x * y + 35 ≥ 251 / 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l2460_246059


namespace NUMINAMATH_CALUDE_incenter_is_circumcenter_of_A₁B₁C₁_l2460_246071

-- Define a structure for a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry
def is_non_equilateral (t : Triangle) : Prop := sorry

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the heights of the triangle
def height_A (t : Triangle) : ℝ × ℝ := sorry
def height_B (t : Triangle) : ℝ × ℝ := sorry
def height_C (t : Triangle) : ℝ × ℝ := sorry

-- Define points A₁, B₁, C₁ on the heights
def A₁ (t : Triangle) : ℝ × ℝ := sorry
def B₁ (t : Triangle) : ℝ × ℝ := sorry
def C₁ (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- The main theorem
theorem incenter_is_circumcenter_of_A₁B₁C₁ (t : Triangle) 
  (h_acute : is_acute_angled t) 
  (h_non_equilateral : is_non_equilateral t) 
  (h_A₁ : A₁ t = height_A t + (0, circumradius t))
  (h_B₁ : B₁ t = height_B t + (0, circumradius t))
  (h_C₁ : C₁ t = height_C t + (0, circumradius t)) :
  incenter t = circumcenter { A := A₁ t, B := B₁ t, C := C₁ t } := by
  sorry

end NUMINAMATH_CALUDE_incenter_is_circumcenter_of_A₁B₁C₁_l2460_246071


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2460_246021

theorem smallest_number_divisible (n : ℕ) : n = 44398 ↔ 
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 12 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 30 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 48 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 74 * k)) ∧
  (∀ m, m < n → ¬(∃ k : ℕ, (m + 2) = 100 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, (n + 2) = 12 * k₁ ∧ 
                         (n + 2) = 30 * k₂ ∧ 
                         (n + 2) = 48 * k₃ ∧ 
                         (n + 2) = 74 * k₄ ∧ 
                         (n + 2) = 100 * k₅) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2460_246021


namespace NUMINAMATH_CALUDE_S_intersect_T_equals_S_l2460_246004

-- Define the sets S and T
def S : Set ℝ := {y | ∃ x, y = 3^x}
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- State the theorem
theorem S_intersect_T_equals_S : S ∩ T = S := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_equals_S_l2460_246004


namespace NUMINAMATH_CALUDE_walker_round_trip_l2460_246011

/-- Ms. Walker's round trip driving problem -/
theorem walker_round_trip (speed_to_work : ℝ) (speed_from_work : ℝ) (total_time : ℝ) 
  (h1 : speed_to_work = 60)
  (h2 : speed_from_work = 40)
  (h3 : total_time = 1) :
  ∃ (distance : ℝ), distance / speed_to_work + distance / speed_from_work = total_time ∧ distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_walker_round_trip_l2460_246011


namespace NUMINAMATH_CALUDE_bird_families_count_l2460_246084

theorem bird_families_count (africa asia left : ℕ) (h1 : africa = 23) (h2 : asia = 37) (h3 : left = 25) :
  africa + asia + left = 85 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_count_l2460_246084


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2460_246072

/-- For a complex number z = 2x + 3iy, |z|^2 = 4x^2 + 9y^2 -/
theorem complex_magnitude_squared (x y : ℝ) : 
  let z : ℂ := 2*x + 3*y*Complex.I
  Complex.normSq z = 4*x^2 + 9*y^2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2460_246072


namespace NUMINAMATH_CALUDE_count_nines_in_subtraction_l2460_246051

/-- The number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The result of subtracting 101011 from 10000000000 -/
def subtraction_result : ℕ := 10000000000 - 101011

/-- Theorem stating that the number of 9's in the subtraction result is 8 -/
theorem count_nines_in_subtraction : countDigit subtraction_result 9 = 8 := by sorry

end NUMINAMATH_CALUDE_count_nines_in_subtraction_l2460_246051


namespace NUMINAMATH_CALUDE_square_perimeter_from_circle_l2460_246012

theorem square_perimeter_from_circle (circle_perimeter : ℝ) : 
  circle_perimeter = 52.5 → 
  ∃ (square_perimeter : ℝ), square_perimeter = 210 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_circle_l2460_246012


namespace NUMINAMATH_CALUDE_path_length_for_73_segment_l2460_246035

/-- Represents a segment divided into smaller parts with squares constructed on each part --/
structure SegmentWithSquares where
  length : ℝ
  num_parts : ℕ

/-- Calculates the length of the path along the arrows for a given segment with squares --/
def path_length (s : SegmentWithSquares) : ℝ := 3 * s.length

theorem path_length_for_73_segment : 
  let s : SegmentWithSquares := { length := 73, num_parts := 2 }
  path_length s = 219 := by sorry

end NUMINAMATH_CALUDE_path_length_for_73_segment_l2460_246035


namespace NUMINAMATH_CALUDE_product_remainder_l2460_246002

/-- The number of times 23 is repeated in the product -/
def n : ℕ := 23

/-- The divisor -/
def m : ℕ := 32

/-- Function to calculate the remainder of the product of n 23's when divided by m -/
def f (n m : ℕ) : ℕ := (23^n) % m

theorem product_remainder : f n m = 19 := by sorry

end NUMINAMATH_CALUDE_product_remainder_l2460_246002


namespace NUMINAMATH_CALUDE_giant_spider_leg_pressure_l2460_246077

/-- Calculates the pressure on each leg of a giant spider -/
theorem giant_spider_leg_pressure (previous_weight : ℝ) (weight_multiplier : ℝ) (leg_area : ℝ) (num_legs : ℕ) : 
  previous_weight = 6.4 →
  weight_multiplier = 2.5 →
  leg_area = 0.5 →
  num_legs = 8 →
  (previous_weight * weight_multiplier) / (num_legs * leg_area) = 4 := by
sorry

end NUMINAMATH_CALUDE_giant_spider_leg_pressure_l2460_246077


namespace NUMINAMATH_CALUDE_quadrant_restriction_l2460_246073

theorem quadrant_restriction (θ : Real) :
  1 + Real.sin θ * Real.sqrt (Real.sin θ * Real.sin θ) + 
  Real.cos θ * Real.sqrt (Real.cos θ * Real.cos θ) = 0 →
  (Real.sin θ > 0 ∧ Real.cos θ > 0) ∨ 
  (Real.sin θ > 0 ∧ Real.cos θ < 0) ∨ 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) → False := by
  sorry

end NUMINAMATH_CALUDE_quadrant_restriction_l2460_246073


namespace NUMINAMATH_CALUDE_min_value_theorem_l2460_246010

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 ∧ 
  ∃ y > 0, y + 1 / (2 * y) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2460_246010


namespace NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2460_246085

/-- Sum of an arithmetic series with given parameters -/
def arithmeticSeriesSum (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  let n : ℤ := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series (-42) + (-40) + ⋯ + 0 is -462 -/
theorem specific_arithmetic_series_sum :
  arithmeticSeriesSum (-42) 0 2 = -462 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_series_sum_l2460_246085


namespace NUMINAMATH_CALUDE_domain_sqrt_one_minus_x_squared_l2460_246050

theorem domain_sqrt_one_minus_x_squared (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 ↔ 1 - x^2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_domain_sqrt_one_minus_x_squared_l2460_246050


namespace NUMINAMATH_CALUDE_cracked_seashells_l2460_246009

/-- The number of cracked seashells given the conditions from the problem -/
theorem cracked_seashells (mary_shells keith_shells total_shells : ℕ) 
  (h1 : mary_shells = 2)
  (h2 : keith_shells = 5)
  (h3 : total_shells = 7) :
  mary_shells + keith_shells - total_shells = 0 := by
  sorry

end NUMINAMATH_CALUDE_cracked_seashells_l2460_246009


namespace NUMINAMATH_CALUDE_pool_capacity_l2460_246015

theorem pool_capacity (C : ℝ) 
  (h1 : 0.4 * C + 300 = 0.7 * C)  -- Adding 300 gallons fills to 70%
  (h2 : 300 = 0.3 * (0.4 * C))    -- 300 gallons is a 30% increase
  : C = 1000 := by
sorry

end NUMINAMATH_CALUDE_pool_capacity_l2460_246015


namespace NUMINAMATH_CALUDE_complement_union_equals_d_l2460_246094

universe u

def U : Set (Fin 4) := {0, 1, 2, 3}
def A : Set (Fin 4) := {0, 1}
def B : Set (Fin 4) := {2}

theorem complement_union_equals_d : 
  (U \ (A ∪ B)) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_d_l2460_246094


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l2460_246087

theorem intersection_point_k_value (k : ℝ) : 
  (∃ x y : ℝ, x - 2*y - 2*k = 0 ∧ 2*x - 3*y - k = 0 ∧ 3*x - y = 0) → k = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l2460_246087


namespace NUMINAMATH_CALUDE_min_value_expression_l2460_246091

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y < 27) :
  (Real.sqrt x + Real.sqrt y) / Real.sqrt (x * y) + 1 / Real.sqrt (27 - x - y) ≥ 1 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b < 27 ∧
    (Real.sqrt a + Real.sqrt b) / Real.sqrt (a * b) + 1 / Real.sqrt (27 - a - b) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2460_246091


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l2460_246080

/-- The probability that exactly one person answers correctly in a competition -/
theorem exactly_one_correct_probability 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/5) 
  (h2 : prob_B = 1/4) : 
  prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l2460_246080


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l2460_246062

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 7^n % 5 = n^4 % 5) → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 
  7^4 % 5 = 4^4 % 5 :=
by sorry

theorem four_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 4 → 7^m % 5 ≠ m^4 % 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_l2460_246062


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2460_246054

/-- Given a line in vector form, prove it's equivalent to the slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (-3 : ℝ) * (x - 3) + (-7 : ℝ) * (y - 14) = 0 ↔ 
  y = (-3/7 : ℝ) * x + 107/7 := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l2460_246054


namespace NUMINAMATH_CALUDE_abc_def_ratio_l2460_246044

theorem abc_def_ratio (a b c d e f : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l2460_246044


namespace NUMINAMATH_CALUDE_power_of_one_sixth_l2460_246045

def is_greatest_power_of_2_dividing_180 (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_dividing_180 (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_sixth (x y : ℕ) 
  (h1 : is_greatest_power_of_2_dividing_180 x) 
  (h2 : is_greatest_power_of_3_dividing_180 y) : 
  (1/6 : ℚ)^(y - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_sixth_l2460_246045


namespace NUMINAMATH_CALUDE_comparison_sqrt_l2460_246000

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l2460_246000


namespace NUMINAMATH_CALUDE_clips_sold_in_april_l2460_246033

theorem clips_sold_in_april (april_clips : ℕ) (may_clips : ℕ) : 
  may_clips = april_clips / 2 →
  april_clips + may_clips = 72 →
  april_clips = 48 := by
sorry

end NUMINAMATH_CALUDE_clips_sold_in_april_l2460_246033


namespace NUMINAMATH_CALUDE_triangle_problem_l2460_246057

open Real

theorem triangle_problem (A B C : ℝ) (h : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A + B = 3 * C ∧
  2 * sin (A - C) = sin B →
  sin A = 3 * sqrt 10 / 10 ∧
  (∀ AB : ℝ, AB = 5 → h = 6 ∧ h * AB / 2 = sin C * AB * sin A / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2460_246057


namespace NUMINAMATH_CALUDE_prove_b_value_l2460_246037

theorem prove_b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 315 * b) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_value_l2460_246037


namespace NUMINAMATH_CALUDE_area_ratio_small_large_triangles_l2460_246023

/-- The ratio of areas between four small equilateral triangles and one large equilateral triangle -/
theorem area_ratio_small_large_triangles : 
  let small_side : ℝ := 10
  let small_perimeter : ℝ := 3 * small_side
  let total_perimeter : ℝ := 4 * small_perimeter
  let large_side : ℝ := total_perimeter / 3
  let small_area : ℝ := (Real.sqrt 3 / 4) * small_side ^ 2
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side ^ 2
  (4 * small_area) / large_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_small_large_triangles_l2460_246023


namespace NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l2460_246049

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_345_ratio_triangle_l2460_246049


namespace NUMINAMATH_CALUDE_distance_between_points_l2460_246074

/-- The Euclidean distance between two points (7, 0) and (-2, 12) is 15 -/
theorem distance_between_points : Real.sqrt ((7 - (-2))^2 + (0 - 12)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2460_246074


namespace NUMINAMATH_CALUDE_field_goal_percentage_l2460_246065

theorem field_goal_percentage (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right : ℕ) : 
  total_attempts = 60 →
  miss_ratio = 1/4 →
  wide_right = 3 →
  (wide_right : ℚ) / (miss_ratio * total_attempts) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_field_goal_percentage_l2460_246065


namespace NUMINAMATH_CALUDE_pentagonal_prism_volume_l2460_246019

/-- The volume of a pentagonal prism with specific dimensions -/
theorem pentagonal_prism_volume : 
  let square_side : ℝ := 2
  let prism_height : ℝ := 2
  let triangle_leg : ℝ := 1
  let base_area : ℝ := square_side ^ 2 - (1 / 2 * triangle_leg * triangle_leg)
  let volume : ℝ := base_area * prism_height
  volume = 7 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_volume_l2460_246019


namespace NUMINAMATH_CALUDE_total_jumps_l2460_246024

theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) : 
  ronald_jumps = 157 → rupert_extra_jumps = 86 → 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_l2460_246024


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2460_246070

noncomputable def quadrilateral_area (P Q A B : ℝ × ℝ) : ℝ :=
  sorry

theorem quadrilateral_area_theorem (P Q A B : ℝ × ℝ) :
  let d := 3 -- distance between P and Q
  let r1 := Real.sqrt 3 -- radius of circle centered at P
  let r2 := 3 -- radius of circle centered at Q
  dist P Q = d ∧
  dist P A = r1 ∧
  dist Q A = r2 ∧
  dist P B = r1 ∧
  dist Q B = r2
  →
  quadrilateral_area P Q A B = (3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2460_246070


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2460_246022

/-- In a Cartesian coordinate system, if a point P has coordinates (3, -5),
    then its coordinates with respect to the origin are also (3, -5). -/
theorem point_coordinates_wrt_origin :
  ∀ (P : ℝ × ℝ), P = (3, -5) → P = (3, -5) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2460_246022


namespace NUMINAMATH_CALUDE_carmen_paint_area_l2460_246068

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in Carmen's house -/
def total_paint_area (num_rooms : ℕ) (room_dims : RoomDimensions) (unpainted_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area := wall_area - unpainted_area
  (num_rooms : ℝ) * paintable_area

/-- Theorem stating that the total area to be painted in Carmen's house is 1408 square feet -/
theorem carmen_paint_area :
  let room_dims : RoomDimensions := ⟨15, 12, 8⟩
  let num_rooms : ℕ := 4
  let unpainted_area : ℝ := 80
  total_paint_area num_rooms room_dims unpainted_area = 1408 := by
  sorry

end NUMINAMATH_CALUDE_carmen_paint_area_l2460_246068


namespace NUMINAMATH_CALUDE_archery_competition_scores_l2460_246078

/-- Represents an archer's score distribution --/
structure ArcherScore where
  bullseye : Nat
  ring39 : Nat
  ring24 : Nat
  ring23 : Nat
  ring17 : Nat
  ring16 : Nat

/-- Calculates the total score for an archer --/
def totalScore (score : ArcherScore) : Nat :=
  40 * score.bullseye + 39 * score.ring39 + 24 * score.ring24 +
  23 * score.ring23 + 17 * score.ring17 + 16 * score.ring16

/-- Calculates the total number of arrows used --/
def totalArrows (score : ArcherScore) : Nat :=
  score.bullseye + score.ring39 + score.ring24 + score.ring23 + score.ring17 + score.ring16

theorem archery_competition_scores :
  ∃ (dora reggie finch : ArcherScore),
    totalScore dora = 120 ∧
    totalScore reggie = 110 ∧
    totalScore finch = 100 ∧
    totalArrows dora = 6 ∧
    totalArrows reggie = 6 ∧
    totalArrows finch = 6 ∧
    dora.bullseye + reggie.bullseye + finch.bullseye = 1 ∧
    dora = { bullseye := 1, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 0, ring16 := 5 } ∧
    reggie = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 2, ring17 := 0, ring16 := 4 } ∧
    finch = { bullseye := 0, ring39 := 0, ring24 := 0, ring23 := 0, ring17 := 4, ring16 := 2 } :=
by
  sorry


end NUMINAMATH_CALUDE_archery_competition_scores_l2460_246078


namespace NUMINAMATH_CALUDE_students_liking_both_subjects_l2460_246067

theorem students_liking_both_subjects (total : ℕ) (math : ℕ) (english : ℕ) (neither : ℕ) 
  (h1 : total = 48)
  (h2 : math = 38)
  (h3 : english = 36)
  (h4 : neither = 4) :
  math + english - (total - neither) = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_subjects_l2460_246067


namespace NUMINAMATH_CALUDE_valid_choices_count_l2460_246048

/-- The number of objects placed along a circle -/
def n : ℕ := 32

/-- The number of objects to be chosen -/
def k : ℕ := 3

/-- The number of ways to choose k objects from n objects -/
def total_ways : ℕ := n.choose k

/-- The number of pairs of adjacent objects -/
def adjacent_pairs : ℕ := n

/-- The number of pairs of diametrically opposite objects -/
def opposite_pairs : ℕ := n / 2

/-- The number of remaining objects after choosing two adjacent or opposite objects -/
def remaining_objects : ℕ := n - 4

/-- The theorem stating the number of valid ways to choose objects -/
theorem valid_choices_count : 
  total_ways - adjacent_pairs * remaining_objects - opposite_pairs * remaining_objects + n = 3648 := by
  sorry

end NUMINAMATH_CALUDE_valid_choices_count_l2460_246048


namespace NUMINAMATH_CALUDE_integral_x_power_five_minus_one_to_one_equals_zero_l2460_246014

theorem integral_x_power_five_minus_one_to_one_equals_zero :
  ∫ x in (-1)..1, x^5 = 0 := by sorry

end NUMINAMATH_CALUDE_integral_x_power_five_minus_one_to_one_equals_zero_l2460_246014


namespace NUMINAMATH_CALUDE_perimeter_ABCDE_l2460_246030

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom AB_eq_BC : dist A B = dist B C
axiom AB_eq_4 : dist A B = 4
axiom AE_eq_8 : dist A E = 8
axiom ED_eq_9 : dist E D = 9
axiom right_angle_EAB : (A.1 - E.1) * (B.1 - A.1) + (A.2 - E.2) * (B.2 - A.2) = 0
axiom right_angle_ABC : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
axiom right_angle_AED : (E.1 - A.1) * (D.1 - E.1) + (E.2 - A.2) * (D.2 - E.2) = 0

-- Define the perimeter function
def perimeter (A B C D E : ℝ × ℝ) : ℝ :=
  dist A B + dist B C + dist C D + dist D E + dist E A

-- State the theorem
theorem perimeter_ABCDE :
  perimeter A B C D E = 25 + Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDE_l2460_246030


namespace NUMINAMATH_CALUDE_total_fish_count_l2460_246096

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1 : num_fishbowls = 261) 
  (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := by
sorry

end NUMINAMATH_CALUDE_total_fish_count_l2460_246096


namespace NUMINAMATH_CALUDE_sqrt_two_sum_l2460_246053

theorem sqrt_two_sum : 2 * Real.sqrt 2 + 3 * Real.sqrt 2 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sum_l2460_246053


namespace NUMINAMATH_CALUDE_sine_transformation_l2460_246001

theorem sine_transformation (ω A a φ : Real) 
  (h_ω : ω > 0) (h_A : A > 0) (h_a : a > 0) (h_φ : 0 < φ ∧ φ < π) :
  (∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - π / 6) + 1) →
  A + a + ω + φ = 16 / 3 + 11 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_sine_transformation_l2460_246001


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2460_246017

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, 58 * z + 14 ≡ 4 [ZMOD 36] → y ≤ z) ∧ 
    (58 * y + 14 ≡ 4 [ZMOD 36]) ∧
    y = 26 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2460_246017


namespace NUMINAMATH_CALUDE_xiwangbei_puzzle_l2460_246032

theorem xiwangbei_puzzle :
  ∃! N : ℕ,
    1000 ≤ N ∧ N < 1000000 ∧
    N % 2 = 0 ∧
    ∃ X Y : ℕ,
      100 ≤ X ∧ X < 1000 ∧
      100 ≤ Y ∧ Y < 1000 ∧
      N = 1000 * X + Y ∧
      8 * N = 5 * (1000 * Y + X) ∧
      N = 256410 :=
by sorry

end NUMINAMATH_CALUDE_xiwangbei_puzzle_l2460_246032


namespace NUMINAMATH_CALUDE_cases_in_2007_l2460_246095

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def diseaseCases (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (annualDecrease * targetYearsSinceInitial)

/-- The number of disease cases in 2007, given the conditions --/
theorem cases_in_2007 :
  diseaseCases 1980 300000 2016 1000 2007 = 75738 := by
  sorry

#eval diseaseCases 1980 300000 2016 1000 2007

end NUMINAMATH_CALUDE_cases_in_2007_l2460_246095


namespace NUMINAMATH_CALUDE_coloring_four_cells_six_colors_l2460_246006

def ColoringMethods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  let twoColorMethods := (Nat.choose n 2) * 2
  let threeColorMethods := (Nat.choose n 3) * (3 * 2^3 - Nat.choose 3 2 * 2)
  twoColorMethods + threeColorMethods

theorem coloring_four_cells_six_colors :
  ColoringMethods 6 4 3 = 390 :=
sorry

end NUMINAMATH_CALUDE_coloring_four_cells_six_colors_l2460_246006


namespace NUMINAMATH_CALUDE_basketball_not_table_tennis_l2460_246092

theorem basketball_not_table_tennis 
  (total : ℕ) 
  (basketball : ℕ) 
  (table_tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 30) 
  (h2 : basketball = 15) 
  (h3 : table_tennis = 10) 
  (h4 : neither = 8) :
  ∃ (both : ℕ), 
    basketball - both = 12 ∧ 
    total = (basketball - both) + (table_tennis - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_table_tennis_l2460_246092


namespace NUMINAMATH_CALUDE_probability_good_not_less_than_defective_expected_value_defective_l2460_246043

/-- The total number of items -/
def total_items : ℕ := 7

/-- The number of good items -/
def good_items : ℕ := 4

/-- The number of defective items -/
def defective_items : ℕ := 3

/-- The number of items selected in the first scenario -/
def selected_items_1 : ℕ := 3

/-- The number of items selected in the second scenario -/
def selected_items_2 : ℕ := 5

/-- Probability of selecting at least as many good items as defective items -/
theorem probability_good_not_less_than_defective :
  (Nat.choose good_items 2 * Nat.choose defective_items 1 + Nat.choose good_items 3) / 
  Nat.choose total_items selected_items_1 = 22 / 35 := by sorry

/-- Expected value of defective items when selecting 5 out of 7 -/
theorem expected_value_defective :
  (1 * Nat.choose good_items 4 * Nat.choose defective_items 1 +
   2 * Nat.choose good_items 3 * Nat.choose defective_items 2 +
   3 * Nat.choose good_items 2 * Nat.choose defective_items 3) /
  Nat.choose total_items selected_items_2 = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_good_not_less_than_defective_expected_value_defective_l2460_246043


namespace NUMINAMATH_CALUDE_cube_volume_derivative_half_surface_area_l2460_246086

-- Define a cube with edge length x
def cube_volume (x : ℝ) : ℝ := x^3
def cube_surface_area (x : ℝ) : ℝ := 6 * x^2

-- State the theorem
theorem cube_volume_derivative_half_surface_area :
  ∀ x : ℝ, (deriv cube_volume) x = (1/2) * cube_surface_area x :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_derivative_half_surface_area_l2460_246086


namespace NUMINAMATH_CALUDE_range_of_a_l2460_246064

/-- Proposition p: There exists x ∈ ℝ such that x^2 - 2x + a^2 = 0 -/
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

/-- Proposition q: For all x ∈ ℝ, ax^2 - ax + 1 > 0 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

/-- The range of a given p ∧ (¬q) is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ ¬(q a)) : -1 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2460_246064


namespace NUMINAMATH_CALUDE_middle_number_problem_l2460_246082

theorem middle_number_problem :
  ∃! n : ℕ, 
    (n - 1)^2 + n^2 + (n + 1)^2 = 2030 ∧
    7 ∣ (n^3 - n^2) ∧
    n = 26 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_problem_l2460_246082


namespace NUMINAMATH_CALUDE_gcd_1037_425_l2460_246075

theorem gcd_1037_425 : Nat.gcd 1037 425 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1037_425_l2460_246075


namespace NUMINAMATH_CALUDE_cookies_left_to_take_home_l2460_246013

def initial_cookies : ℕ := 120
def dozen : ℕ := 12
def morning_sales : ℕ := 3 * dozen
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16

theorem cookies_left_to_take_home : 
  initial_cookies - morning_sales - lunch_sales - afternoon_sales = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_to_take_home_l2460_246013


namespace NUMINAMATH_CALUDE_f_properties_l2460_246025

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- Define the interval [-3, 1]
def interval : Set ℝ := Set.Icc (-3) 1

theorem f_properties :
  -- f(x) is decreasing on [-3, -2] and increasing on [-2, 1]
  (∀ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), x < y → f x > f y) ∧
  (∀ x ∈ Set.Ioo (-2) 1, ∀ y ∈ Set.Ioo (-2) 1, x < y → f x < f y) ∧
  -- The maximum value is 11
  (∃ x ∈ interval, f x = 11 ∧ ∀ y ∈ interval, f y ≤ 11) ∧
  -- The minimum value is -16
  (∃ x ∈ interval, f x = -16 ∧ ∀ y ∈ interval, f y ≥ -16) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l2460_246025


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_zero_l2460_246093

theorem sum_of_solutions_eq_zero : 
  ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 13*x^2 + 36 = 0) ∧ 
                    (∀ x : ℤ, x^4 - 13*x^2 + 36 = 0 → x ∈ S) ∧ 
                    (S.sum id = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_zero_l2460_246093


namespace NUMINAMATH_CALUDE_basketball_scoring_l2460_246020

/-- Basketball game scoring problem -/
theorem basketball_scoring
  (alex_points : ℕ)
  (sam_points : ℕ)
  (jon_points : ℕ)
  (jack_points : ℕ)
  (tom_points : ℕ)
  (h1 : jon_points = 2 * sam_points + 3)
  (h2 : sam_points = alex_points / 2)
  (h3 : alex_points = jack_points - 7)
  (h4 : jack_points = jon_points + 5)
  (h5 : tom_points = jon_points + jack_points - 4)
  (h6 : alex_points = 18) :
  jon_points + jack_points + tom_points + sam_points + alex_points = 115 := by
sorry

end NUMINAMATH_CALUDE_basketball_scoring_l2460_246020


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_progression_l2460_246052

theorem smallest_b_in_arithmetic_progression (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  (∃ d : ℝ, a = b - d ∧ c = b + d) →
  a * b * c = 125 →
  b ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_progression_l2460_246052


namespace NUMINAMATH_CALUDE_P_on_y_axis_P_in_first_quadrant_with_distance_condition_l2460_246005

-- Define point P
def P (m : ℝ) : ℝ × ℝ := (8 - 2*m, m + 1)

-- Part 1: P lies on y-axis
theorem P_on_y_axis (m : ℝ) : 
  P m = (0, m + 1) → m = 4 := by sorry

-- Part 2: P in first quadrant with distance condition
theorem P_in_first_quadrant_with_distance_condition (m : ℝ) :
  (8 - 2*m > 0 ∧ m + 1 > 0) ∧ (m + 1 = 2*(8 - 2*m)) → P m = (2, 4) := by sorry

end NUMINAMATH_CALUDE_P_on_y_axis_P_in_first_quadrant_with_distance_condition_l2460_246005


namespace NUMINAMATH_CALUDE_percentage_goldfish_special_food_l2460_246069

-- Define the parameters
def total_goldfish : ℕ := 50
def food_per_goldfish : ℚ := 3/2
def special_food_cost : ℚ := 3
def total_special_food_cost : ℚ := 45

-- Define the theorem
theorem percentage_goldfish_special_food :
  (((total_special_food_cost / special_food_cost) / food_per_goldfish) / total_goldfish) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_goldfish_special_food_l2460_246069


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2460_246090

/-- The segment interval for systematic sampling given a population and sample size -/
def segment_interval (population : ℕ) (sample_size : ℕ) : ℕ :=
  population / sample_size

/-- Theorem: The segment interval for systematic sampling of 100 students from a population of 2400 is 24 -/
theorem systematic_sampling_interval :
  segment_interval 2400 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2460_246090


namespace NUMINAMATH_CALUDE_congruence_problem_l2460_246061

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 16 = 4 → (4 * x + 5) % 16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2460_246061


namespace NUMINAMATH_CALUDE_work_completion_time_l2460_246046

theorem work_completion_time 
  (ratio_a : ℚ) 
  (ratio_b : ℚ) 
  (combined_time : ℚ) 
  (h1 : ratio_a / ratio_b = 3 / 2) 
  (h2 : combined_time = 18) : 
  ratio_a / (ratio_a + ratio_b) * combined_time = 30 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2460_246046


namespace NUMINAMATH_CALUDE_milford_future_age_l2460_246042

/-- Proves that Milford's age in 3 years will be 21, given the conditions about Eustace's age. -/
theorem milford_future_age :
  ∀ (eustace_age milford_age : ℕ),
  eustace_age = 2 * milford_age →
  eustace_age + 3 = 39 →
  milford_age + 3 = 21 := by
sorry

end NUMINAMATH_CALUDE_milford_future_age_l2460_246042


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2460_246034

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2460_246034


namespace NUMINAMATH_CALUDE_select_books_result_l2460_246063

/-- The number of ways to select one book from each of two bags of science books -/
def select_books (bag1_count : ℕ) (bag2_count : ℕ) : ℕ :=
  bag1_count * bag2_count

/-- Theorem: The number of ways to select one book from each of two bags,
    where one bag contains 4 different books and the other contains 5 different books,
    is equal to 20. -/
theorem select_books_result : select_books 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_select_books_result_l2460_246063


namespace NUMINAMATH_CALUDE_video_watching_time_l2460_246039

theorem video_watching_time (video_length : ℕ) (num_videos : ℕ) : 
  video_length = 100 → num_videos = 6 → 
  (num_videos * video_length / 2 + num_videos * video_length) = 900 := by
  sorry

end NUMINAMATH_CALUDE_video_watching_time_l2460_246039


namespace NUMINAMATH_CALUDE_sandwich_problem_l2460_246038

theorem sandwich_problem (total : ℕ) (bologna : ℕ) (x : ℕ) :
  total = 80 →
  bologna = 35 →
  bologna = 7 * (total / (1 + 7 + x)) →
  x * (total / (1 + 7 + x)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_problem_l2460_246038
