import Mathlib

namespace NUMINAMATH_CALUDE_inverse_f_at_2_l3887_388717

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_2 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≥ 0, f_inv (f x) = x) ∧
    (∀ y ≥ -1, f (f_inv y) = y) ∧
    f_inv 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_2_l3887_388717


namespace NUMINAMATH_CALUDE_triangle_properties_l3887_388776

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧ t.c = 4

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π / 3 ∧
  (∀ s : ℝ, s = 1/2 * t.a * t.b * Real.sin t.C → s ≤ 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3887_388776


namespace NUMINAMATH_CALUDE_rice_weight_qualification_l3887_388729

def weight_range (x : ℝ) : Prop := 9.9 ≤ x ∧ x ≤ 10.1

theorem rice_weight_qualification :
  ¬(weight_range 9.09) ∧
  weight_range 9.99 ∧
  weight_range 10.01 ∧
  weight_range 10.09 :=
by sorry

end NUMINAMATH_CALUDE_rice_weight_qualification_l3887_388729


namespace NUMINAMATH_CALUDE_prob_three_heads_before_two_tails_l3887_388767

/-- The probability of getting a specific outcome when flipping a fair coin -/
def fair_coin_prob : ℚ := 1/2

/-- The state space for the coin flipping process -/
inductive CoinState
| H0  -- No heads or tails flipped yet
| H1  -- 1 consecutive head flipped
| H2  -- 2 consecutive heads flipped
| T1  -- 1 tail flipped
| HHH -- 3 consecutive heads (win state)
| TT  -- 2 consecutive tails (lose state)

/-- The probability of reaching the HHH state from a given state -/
noncomputable def prob_reach_HHH : CoinState → ℚ
| CoinState.H0 => sorry
| CoinState.H1 => sorry
| CoinState.H2 => sorry
| CoinState.T1 => sorry
| CoinState.HHH => 1
| CoinState.TT => 0

/-- The main theorem: probability of reaching HHH from the initial state is 3/8 -/
theorem prob_three_heads_before_two_tails : prob_reach_HHH CoinState.H0 = 3/8 := by sorry

end NUMINAMATH_CALUDE_prob_three_heads_before_two_tails_l3887_388767


namespace NUMINAMATH_CALUDE_jury_duty_duration_l3887_388794

/-- Calculates the total number of days spent on jury duty -/
def total_jury_duty_days (jury_selection_days : ℕ) (trial_duration_factor : ℕ) 
  (deliberation_full_days : ℕ) (daily_deliberation_hours : ℕ) : ℕ :=
  let trial_days := jury_selection_days * trial_duration_factor
  let deliberation_hours := deliberation_full_days * 24
  let deliberation_days := deliberation_hours / daily_deliberation_hours
  jury_selection_days + trial_days + deliberation_days

/-- Theorem stating that the total number of days spent on jury duty is 19 -/
theorem jury_duty_duration : 
  total_jury_duty_days 2 4 6 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jury_duty_duration_l3887_388794


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3887_388799

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 - Real.cos (x * Real.sin (1 / x))
  else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3887_388799


namespace NUMINAMATH_CALUDE_leftover_value_calculation_l3887_388707

/-- Calculates the value of leftover coins after making complete rolls --/
def leftover_value (quarters_per_roll dimes_per_roll toledo_quarters toledo_dimes brian_quarters brian_dimes : ℕ) : ℚ :=
  let total_quarters := toledo_quarters + brian_quarters
  let total_dimes := toledo_dimes + brian_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_calculation :
  leftover_value 30 50 95 172 137 290 = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_calculation_l3887_388707


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l3887_388710

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l3887_388710


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l3887_388792

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  bottom_face_on_base : Bool
  top_face_edges_on_lateral_faces : Bool

/-- The volume of an inscribed cube -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side = 2 ∧
    cube.pyramid.lateral_face_is_equilateral = true ∧
    cube.bottom_face_on_base = true ∧
    cube.top_face_edges_on_lateral_faces = true →
    inscribed_cube_volume cube = 2 * Real.sqrt 6 / 9 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l3887_388792


namespace NUMINAMATH_CALUDE_percentage_problem_l3887_388758

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3887_388758


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3887_388718

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (π * r / 4)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l3887_388718


namespace NUMINAMATH_CALUDE_multiplication_problem_l3887_388706

theorem multiplication_problem (x : ℝ) : 4 * x = 60 → 8 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l3887_388706


namespace NUMINAMATH_CALUDE_candy_distribution_l3887_388773

theorem candy_distribution (A B : ℕ) 
  (h1 : 7 * A = B + 12) 
  (h2 : 3 * A = B - 20) : 
  A + B = 52 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3887_388773


namespace NUMINAMATH_CALUDE_min_k_plus_l_l3887_388727

theorem min_k_plus_l (k l : ℕ+) (h : 120 * k = l ^ 3) : 
  ∀ (k' l' : ℕ+), 120 * k' = l' ^ 3 → k + l ≤ k' + l' :=
by sorry

end NUMINAMATH_CALUDE_min_k_plus_l_l3887_388727


namespace NUMINAMATH_CALUDE_airplane_passengers_l3887_388779

theorem airplane_passengers (total : ℕ) (men : ℕ) : 
  total = 170 → men = 90 → 2 * (total - men - (men / 2)) = men → total - men - (men / 2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_airplane_passengers_l3887_388779


namespace NUMINAMATH_CALUDE_percentage_problem_l3887_388790

theorem percentage_problem (p : ℝ) : (p / 100) * 40 = 140 → p = 350 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3887_388790


namespace NUMINAMATH_CALUDE_problem_solution_l3887_388743

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3) (h2 : y^2 / z = 4) (h3 : z^2 / x = 5) :
  x = (36 * Real.sqrt 5) ^ (4/11) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3887_388743


namespace NUMINAMATH_CALUDE_tub_drain_time_l3887_388783

/-- Represents the time it takes to drain a tub -/
def drainTime (initialFraction : ℚ) (drainedFraction : ℚ) (initialTime : ℚ) : ℚ :=
  (drainedFraction * initialTime) / initialFraction

theorem tub_drain_time :
  let initialFraction : ℚ := 5 / 7
  let remainingFraction : ℚ := 1 - initialFraction
  let initialTime : ℚ := 4
  drainTime initialFraction remainingFraction initialTime = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tub_drain_time_l3887_388783


namespace NUMINAMATH_CALUDE_wage_decrease_hours_increase_l3887_388734

theorem wage_decrease_hours_increase (W H : ℝ) (W_new H_new : ℝ) :
  W > 0 → H > 0 →
  W_new = 0.8 * W →
  W * H = W_new * H_new →
  (H_new - H) / H = 0.25 :=
sorry

end NUMINAMATH_CALUDE_wage_decrease_hours_increase_l3887_388734


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3887_388704

-- Define the sets U and A
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- State the theorem
theorem complement_A_in_U : (U \ A) = {1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3887_388704


namespace NUMINAMATH_CALUDE_fruit_sales_revenue_l3887_388730

theorem fruit_sales_revenue : 
  let original_lemon_price : ℝ := 8
  let original_grape_price : ℝ := 7
  let lemon_price_increase : ℝ := 4
  let grape_price_increase : ℝ := lemon_price_increase / 2
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let new_lemon_price : ℝ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℝ := original_grape_price + grape_price_increase
  let total_revenue : ℝ := (↑num_lemons * new_lemon_price) + (↑num_grapes * new_grape_price)
  total_revenue = 2220 := by
sorry

end NUMINAMATH_CALUDE_fruit_sales_revenue_l3887_388730


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3887_388754

/-- A plane equation is represented by a tuple of integers (A, B, C, D) corresponding to the equation Ax + By + Cz + D = 0 --/
def PlaneEquation := (ℤ × ℤ × ℤ × ℤ)

/-- The given plane equation 3x - 2y + 4z = 10 --/
def given_plane : PlaneEquation := (3, -2, 4, -10)

/-- The point through which the new plane must pass --/
def point : (ℤ × ℤ × ℤ) := (2, -3, 5)

/-- Check if a plane equation passes through a given point --/
def passes_through (plane : PlaneEquation) (p : ℤ × ℤ × ℤ) : Prop :=
  let (A, B, C, D) := plane
  let (x, y, z) := p
  A * x + B * y + C * z + D = 0

/-- Check if two plane equations are parallel --/
def is_parallel (plane1 plane2 : PlaneEquation) : Prop :=
  let (A1, B1, C1, _) := plane1
  let (A2, B2, C2, _) := plane2
  ∃ (k : ℚ), k ≠ 0 ∧ A1 = k * A2 ∧ B1 = k * B2 ∧ C1 = k * C2

/-- Check if the first coefficient of a plane equation is positive --/
def first_coeff_positive (plane : PlaneEquation) : Prop :=
  let (A, _, _, _) := plane
  A > 0

/-- Calculate the greatest common divisor of the absolute values of all coefficients --/
def gcd_of_coeffs (plane : PlaneEquation) : ℕ :=
  let (A, B, C, D) := plane
  Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D)))

theorem plane_equation_proof (solution : PlaneEquation) : 
  passes_through solution point ∧ 
  is_parallel solution given_plane ∧ 
  first_coeff_positive solution ∧ 
  gcd_of_coeffs solution = 1 ∧ 
  solution = (3, -2, 4, -32) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3887_388754


namespace NUMINAMATH_CALUDE_area_at_stage_8_l3887_388740

/-- The side length of each square -/
def squareSide : ℕ := 4

/-- The area of each square -/
def squareArea : ℕ := squareSide * squareSide

/-- The number of squares at a given stage -/
def numSquaresAtStage (stage : ℕ) : ℕ := stage

/-- The total area of the rectangle at a given stage -/
def totalAreaAtStage (stage : ℕ) : ℕ := numSquaresAtStage stage * squareArea

/-- The theorem stating that the area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : totalAreaAtStage 8 = 128 := by sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l3887_388740


namespace NUMINAMATH_CALUDE_language_course_enrollment_l3887_388769

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 120 →
  french = 52 →
  german = 35 →
  spanish = 48 →
  french_german = 15 →
  french_spanish = 20 →
  german_spanish = 12 →
  all_three = 6 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 32 := by
sorry

end NUMINAMATH_CALUDE_language_course_enrollment_l3887_388769


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l3887_388770

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem f_monotone_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun y ↦ f y) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l3887_388770


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3887_388775

theorem polynomial_factorization (a b : ℤ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x+1)*(x-3)) → (a = -2 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3887_388775


namespace NUMINAMATH_CALUDE_fibonacci_problem_l3887_388756

theorem fibonacci_problem (x : ℕ) (h : x > 0) :
  (10 : ℝ) / x = 40 / (x + 6) →
  ∃ (y : ℕ), y > 0 ∧
    (10 : ℝ) / x = 10 / y ∧
    40 / (x + 6) = 40 / (y + 6) ∧
    (10 : ℝ) / y = 40 / (y + 6) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_problem_l3887_388756


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3887_388768

/-- Proves that the width of a rectangular field is 1400/29 meters given specific conditions -/
theorem rectangular_field_width (w : ℝ) : 
  w > 0 → -- width is positive
  (2*w + 2*(7/5*w) + w = 280) → -- combined perimeter equation
  w = 1400/29 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3887_388768


namespace NUMINAMATH_CALUDE_final_price_percentage_l3887_388700

/-- Given a suggested retail price, store discount, and additional discount,
    calculates the percentage of the original price paid. -/
def percentage_paid (suggested_retail_price : ℝ) (store_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - store_discount) * (1 - additional_discount) * 100

/-- Theorem stating that with a 20% store discount and 10% additional discount,
    the final price paid is 72% of the suggested retail price. -/
theorem final_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0)
  (h2 : store_discount = 0.2)
  (h3 : additional_discount = 0.1) :
  percentage_paid suggested_retail_price store_discount additional_discount = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l3887_388700


namespace NUMINAMATH_CALUDE_william_riding_time_l3887_388748

theorem william_riding_time :
  let max_daily_time : ℝ := 6
  let total_days : ℕ := 6
  let max_time_days : ℕ := 2
  let min_time_days : ℕ := 2
  let half_time_days : ℕ := 2
  let min_daily_time : ℝ := 1.5

  max_time_days * max_daily_time +
  min_time_days * min_daily_time +
  half_time_days * (max_daily_time / 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_william_riding_time_l3887_388748


namespace NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l3887_388747

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_empty_iff (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -4 ∨ a ≥ 5 := by
  sorry

theorem union_equals_B_iff (a : ℝ) : A a ∪ B = B ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l3887_388747


namespace NUMINAMATH_CALUDE_tire_sale_price_l3887_388716

/-- Calculates the sale price of a tire given the number of tires, total savings, and original price. -/
def sale_price (num_tires : ℕ) (total_savings : ℚ) (original_price : ℚ) : ℚ :=
  original_price - (total_savings / num_tires)

/-- Theorem stating that the sale price of each tire is $75 given the problem conditions. -/
theorem tire_sale_price :
  let num_tires : ℕ := 4
  let total_savings : ℚ := 36
  let original_price : ℚ := 84
  sale_price num_tires total_savings original_price = 75 := by
sorry

end NUMINAMATH_CALUDE_tire_sale_price_l3887_388716


namespace NUMINAMATH_CALUDE_sector_arc_length_l3887_388764

/-- Given a sector with a central angle of 60° and a radius of 3,
    the length of the arc is equal to π. -/
theorem sector_arc_length (θ : Real) (r : Real) : 
  θ = 60 * π / 180 → r = 3 → θ * r = π := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3887_388764


namespace NUMINAMATH_CALUDE_uncovered_area_of_squares_l3887_388759

theorem uncovered_area_of_squares (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_of_squares_l3887_388759


namespace NUMINAMATH_CALUDE_light_2011_is_green_l3887_388797

def light_pattern : ℕ → String
  | 0 => "green"
  | 1 => "yellow"
  | 2 => "yellow"
  | 3 => "red"
  | 4 => "red"
  | 5 => "red"
  | n + 6 => light_pattern n

theorem light_2011_is_green : light_pattern 2010 = "green" := by
  sorry

end NUMINAMATH_CALUDE_light_2011_is_green_l3887_388797


namespace NUMINAMATH_CALUDE_vector_magnitude_l3887_388712

theorem vector_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 50 →
  b.1^2 + b.2^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3887_388712


namespace NUMINAMATH_CALUDE_circle_point_x_value_l3887_388709

theorem circle_point_x_value (x : ℝ) :
  let center : ℝ × ℝ := ((21 - (-3)) / 2 + (-3), 0)
  let radius : ℝ := (21 - (-3)) / 2
  (x - center.1) ^ 2 + (12 - center.2) ^ 2 = radius ^ 2 →
  x = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l3887_388709


namespace NUMINAMATH_CALUDE_vectors_opposite_direction_l3887_388738

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction :
  ∃ k : ℝ, k < 0 ∧ a = (k • b) := by sorry

end NUMINAMATH_CALUDE_vectors_opposite_direction_l3887_388738


namespace NUMINAMATH_CALUDE_trouser_sale_price_l3887_388702

theorem trouser_sale_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 70) : 
  original_price * (1 - discount_percentage / 100) = 30 := by
  sorry

end NUMINAMATH_CALUDE_trouser_sale_price_l3887_388702


namespace NUMINAMATH_CALUDE_smallest_group_size_exists_smallest_group_size_l3887_388760

theorem smallest_group_size (n : ℕ) : n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 → n ≥ 65 := by
  sorry

theorem exists_smallest_group_size : ∃ n : ℕ, n > 0 ∧ n % 5 = 0 ∧ n % 13 = 0 ∧ n = 65 := by
  sorry

end NUMINAMATH_CALUDE_smallest_group_size_exists_smallest_group_size_l3887_388760


namespace NUMINAMATH_CALUDE_library_visitors_average_l3887_388715

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (days_in_month : ℕ) (sundays_in_month : ℕ) (h1 : sunday_visitors = 140) 
  (h2 : other_day_visitors = 80) (h3 : days_in_month = 30) (h4 : sundays_in_month = 4) :
  (sunday_visitors * sundays_in_month + other_day_visitors * (days_in_month - sundays_in_month)) / 
  days_in_month = 88 := by
  sorry

#check library_visitors_average

end NUMINAMATH_CALUDE_library_visitors_average_l3887_388715


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3887_388765

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3887_388765


namespace NUMINAMATH_CALUDE_f_sum_theorem_l3887_388726

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_sum_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = -1) : 
  f 5 + f 13 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l3887_388726


namespace NUMINAMATH_CALUDE_flowers_per_set_l3887_388752

theorem flowers_per_set (total_flowers : ℕ) (num_sets : ℕ) (h1 : total_flowers = 270) (h2 : num_sets = 3) :
  total_flowers / num_sets = 90 := by
  sorry

end NUMINAMATH_CALUDE_flowers_per_set_l3887_388752


namespace NUMINAMATH_CALUDE_prime_sequence_recurrence_relation_l3887_388753

theorem prime_sequence_recurrence_relation 
  (p : ℕ → ℕ) 
  (k : ℤ) 
  (h_prime : ∀ n, Nat.Prime (p n)) 
  (h_recurrence : ∀ n, p (n + 2) = p (n + 1) + p n + k) : 
  (∃ (prime : ℕ) (h_prime : Nat.Prime prime), 
    (∀ n, p n = prime) ∧ k = -prime) := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_recurrence_relation_l3887_388753


namespace NUMINAMATH_CALUDE_vector_linear_combination_l3887_388785

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  ∃ (k l : ℝ), c = k • a + l • b ∧ k = (1/2 : ℝ) ∧ l = (-3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l3887_388785


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l3887_388741

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l3887_388741


namespace NUMINAMATH_CALUDE_total_onions_l3887_388771

theorem total_onions (sara sally fred jack : ℕ) 
  (h1 : sara = 4) 
  (h2 : sally = 5) 
  (h3 : fred = 9) 
  (h4 : jack = 7) : 
  sara + sally + fred + jack = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_onions_l3887_388771


namespace NUMINAMATH_CALUDE_fifth_term_smallest_l3887_388780

/-- The sequence term for a given n -/
def sequence_term (n : ℕ) : ℤ := 3 * n^2 - 28 * n

/-- The 5th term is the smallest in the sequence -/
theorem fifth_term_smallest : ∀ k : ℕ, sequence_term 5 ≤ sequence_term k := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_smallest_l3887_388780


namespace NUMINAMATH_CALUDE_hot_dog_consumption_l3887_388757

theorem hot_dog_consumption (x : ℕ) : 
  x + (x + 2) + (x + 4) = 36 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_consumption_l3887_388757


namespace NUMINAMATH_CALUDE_division_of_fractions_l3887_388735

theorem division_of_fractions : (4 - 1/4) / (2 - 1/2) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3887_388735


namespace NUMINAMATH_CALUDE_focus_directrix_distance_l3887_388722

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix (x : ℝ) : Prop := x = -1

/-- The distance from the focus to the directrix of the parabola y^2 = 4x is 2 -/
theorem focus_directrix_distance : 
  ∃ (d : ℝ), d = 2 ∧ d = |focus.1 - (-1)| :=
sorry

end NUMINAMATH_CALUDE_focus_directrix_distance_l3887_388722


namespace NUMINAMATH_CALUDE_set_membership_implies_a_values_l3887_388714

def A (a : ℝ) : Set ℝ := {2, 1-a, a^2-a+2}

theorem set_membership_implies_a_values (a : ℝ) :
  4 ∈ A a → a = -3 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_a_values_l3887_388714


namespace NUMINAMATH_CALUDE_mustard_at_second_table_l3887_388772

/-- The amount of mustard found at each table and the total amount --/
def MustardProblem (total first second third : ℚ) : Prop :=
  total = first + second + third

theorem mustard_at_second_table :
  ∃ (second : ℚ), MustardProblem 0.88 0.25 second 0.38 ∧ second = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_mustard_at_second_table_l3887_388772


namespace NUMINAMATH_CALUDE_max_odd_digits_in_sum_l3887_388746

/-- A function that counts the number of odd digits in a natural number -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 digits -/
def has_ten_digits (n : ℕ) : Prop := sorry

theorem max_odd_digits_in_sum (a b c : ℕ) 
  (ha : has_ten_digits a) 
  (hb : has_ten_digits b) 
  (hc : has_ten_digits c) 
  (sum_eq : a + b = c) : 
  count_odd_digits a + count_odd_digits b + count_odd_digits c ≤ 29 :=
sorry

end NUMINAMATH_CALUDE_max_odd_digits_in_sum_l3887_388746


namespace NUMINAMATH_CALUDE_four_numbers_sum_product_l3887_388762

/-- Given four real numbers x₁, x₂, x₃, x₄, if the sum of any one number and the product 
    of the other three is equal to 2, then the only possible solutions are 
    (1, 1, 1, 1) and (-1, -1, -1, 3) and its permutations. -/
theorem four_numbers_sum_product (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2) ∧ 
  (x₂ + x₃ * x₄ * x₁ = 2) ∧ 
  (x₃ + x₄ * x₁ * x₂ = 2) ∧ 
  (x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_sum_product_l3887_388762


namespace NUMINAMATH_CALUDE_hyperbola_focus_l3887_388751

/-- Given a hyperbola with equation (x-1)^2/7^2 - (y+8)^2/3^2 = 1,
    the coordinates of the focus with the smaller x-coordinate are (1 - √58, -8) -/
theorem hyperbola_focus (x y : ℝ) :
  (x - 1)^2 / 7^2 - (y + 8)^2 / 3^2 = 1 →
  ∃ (focus_x focus_y : ℝ),
    focus_x = 1 - Real.sqrt 58 ∧
    focus_y = -8 ∧
    ∀ (other_focus_x : ℝ),
      ((other_focus_x - 1)^2 / 7^2 - (focus_y + 8)^2 / 3^2 = 1 →
       other_focus_x ≥ focus_x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l3887_388751


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3887_388782

/-- The ratio of area to perimeter for an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 6
  let area : ℝ := s^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * s
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3887_388782


namespace NUMINAMATH_CALUDE_fraction_addition_l3887_388744

theorem fraction_addition : (3/4) / (5/8) + 1/2 = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3887_388744


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l3887_388742

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l3887_388742


namespace NUMINAMATH_CALUDE_union_condition_implies_a_range_l3887_388750

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 5}

theorem union_condition_implies_a_range (a : ℝ) :
  A a ∪ B = B → a < -4 ∨ a > 5 := by
  sorry

end NUMINAMATH_CALUDE_union_condition_implies_a_range_l3887_388750


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3887_388766

theorem system_solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 10 * y - 15 * x = d) : c / d = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3887_388766


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l3887_388793

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l3887_388793


namespace NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l3887_388705

theorem more_freshmen_than_sophomores :
  ∀ (total juniors not_sophomores seniors freshmen sophomores : ℕ),
  total = 800 →
  juniors = (22 * total) / 100 →
  not_sophomores = (75 * total) / 100 →
  seniors = 160 →
  freshmen + sophomores + juniors + seniors = total →
  sophomores = total - not_sophomores →
  freshmen - sophomores = 64 :=
by sorry

end NUMINAMATH_CALUDE_more_freshmen_than_sophomores_l3887_388705


namespace NUMINAMATH_CALUDE_charlie_ate_fifteen_cookies_l3887_388749

/-- The number of cookies eaten by Charlie's family -/
def total_cookies : ℕ := 30

/-- The number of cookies eaten by Charlie's father -/
def father_cookies : ℕ := 10

/-- The number of cookies eaten by Charlie's mother -/
def mother_cookies : ℕ := 5

/-- Charlie's cookies -/
def charlie_cookies : ℕ := total_cookies - (father_cookies + mother_cookies)

theorem charlie_ate_fifteen_cookies : charlie_cookies = 15 := by
  sorry

end NUMINAMATH_CALUDE_charlie_ate_fifteen_cookies_l3887_388749


namespace NUMINAMATH_CALUDE_budgets_equal_in_1996_l3887_388774

/-- Represents the year when the budgets of two projects become equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n : ℕ := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets become equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_budgets_equal_in_1996_l3887_388774


namespace NUMINAMATH_CALUDE_judy_pencil_cost_l3887_388720

-- Define the given conditions
def pencils_per_week : ℕ := 10
def days_per_week : ℕ := 5
def pencils_per_pack : ℕ := 30
def cost_per_pack : ℕ := 4
def total_days : ℕ := 45

-- Define the theorem
theorem judy_pencil_cost :
  let pencils_per_day : ℚ := pencils_per_week / days_per_week
  let total_pencils : ℚ := pencils_per_day * total_days
  let packs_needed : ℚ := total_pencils / pencils_per_pack
  let total_cost : ℚ := packs_needed * cost_per_pack
  total_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_judy_pencil_cost_l3887_388720


namespace NUMINAMATH_CALUDE_peach_difference_l3887_388796

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there are 8 more green peaches than yellow peaches. -/
theorem peach_difference (red yellow green : ℕ) 
    (h_red : red = 2)
    (h_yellow : yellow = 6)
    (h_green : green = 14) :
  green - yellow = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3887_388796


namespace NUMINAMATH_CALUDE_abs_neg_five_plus_three_l3887_388755

theorem abs_neg_five_plus_three : |(-5 + 3)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_plus_three_l3887_388755


namespace NUMINAMATH_CALUDE_carpet_cost_calculation_carpet_cost_result_l3887_388733

/-- Calculate the cost of a carpet with increased dimensions -/
theorem carpet_cost_calculation (breadth_1 : Real) (length_ratio : Real) 
  (length_increase : Real) (breadth_increase : Real) (rate : Real) : Real :=
  let length_1 := breadth_1 * length_ratio
  let breadth_2 := breadth_1 * (1 + breadth_increase)
  let length_2 := length_1 * (1 + length_increase)
  let area_2 := breadth_2 * length_2
  area_2 * rate

/-- The cost of the carpet with specified dimensions and rate -/
theorem carpet_cost_result : 
  carpet_cost_calculation 6 1.44 0.4 0.25 45 = 4082.4 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_calculation_carpet_cost_result_l3887_388733


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3887_388763

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a * Real.sin B - Real.sqrt 5 * b * Real.cos A = 0 →
  -- Theorem 1: cos A = 2/3
  Real.cos A = 2/3 ∧
  -- Theorem 2: If a = √5 and b = 2, area = √5
  (a = Real.sqrt 5 ∧ b = 2 → 
    (1/2) * a * b * Real.sin C = Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3887_388763


namespace NUMINAMATH_CALUDE_range_of_k_l3887_388787

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the non-collinearity of e₁ and e₂
variable (h_non_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)

-- Define vectors a and b
def a : V := 2 • e₁ + e₂
def b (k : ℝ) : V := k • e₁ + 3 • e₂

-- Define the condition that a and b form a basis
variable (h_basis : ∀ (k : ℝ), k ≠ 6 → LinearIndependent ℝ ![a, b k])

-- Theorem statement
theorem range_of_k : 
  {k : ℝ | k ≠ 6} = {k : ℝ | LinearIndependent ℝ ![a, b k]} :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3887_388787


namespace NUMINAMATH_CALUDE_fraction_power_multiply_l3887_388719

theorem fraction_power_multiply (a b c : ℚ) : 
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_multiply_l3887_388719


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3887_388703

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3887_388703


namespace NUMINAMATH_CALUDE_eliana_steps_l3887_388777

/-- The total number of steps Eliana walked during three days -/
def total_steps (first_day_initial : ℕ) (first_day_additional : ℕ) (third_day_additional : ℕ) : ℕ :=
  let first_day := first_day_initial + first_day_additional
  let second_day := 2 * first_day
  let third_day := second_day + third_day_additional
  first_day + second_day + third_day

/-- Theorem stating the total number of steps Eliana walked during three days -/
theorem eliana_steps : 
  total_steps 200 300 100 = 2600 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l3887_388777


namespace NUMINAMATH_CALUDE_social_logistics_turnover_scientific_notation_l3887_388721

/-- Given that one trillion is 10^12, prove that 347.6 trillion yuan is equal to 3.476 × 10^14 yuan -/
theorem social_logistics_turnover_scientific_notation :
  let trillion : ℝ := 10^12
  347.6 * trillion = 3.476 * 10^14 := by
  sorry

end NUMINAMATH_CALUDE_social_logistics_turnover_scientific_notation_l3887_388721


namespace NUMINAMATH_CALUDE_triangle_area_specific_l3887_388728

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of a triangle with vertices at (-1,-1), (2,3), and (-4,0) is 8.5 square units -/
theorem triangle_area_specific : triangleArea (-1) (-1) 2 3 (-4) 0 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l3887_388728


namespace NUMINAMATH_CALUDE_circus_ticket_price_l3887_388739

theorem circus_ticket_price :
  let total_tickets : ℕ := 522
  let child_ticket_price : ℚ := 8
  let total_receipts : ℚ := 5086
  let adult_tickets_sold : ℕ := 130
  let child_tickets_sold : ℕ := total_tickets - adult_tickets_sold
  let adult_ticket_price : ℚ := (total_receipts - child_ticket_price * child_tickets_sold) / adult_tickets_sold
  adult_ticket_price = 15 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l3887_388739


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l3887_388736

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 → 
  red = 14 → 
  blue = 3 * red → 
  yellow = total - (red + blue) → 
  yellow = 29 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l3887_388736


namespace NUMINAMATH_CALUDE_long_sleeved_jersey_cost_l3887_388781

/-- Represents the cost of jerseys and proves the cost of long-sleeved jerseys --/
theorem long_sleeved_jersey_cost 
  (long_sleeved_count : ℕ) 
  (striped_count : ℕ) 
  (striped_cost : ℕ) 
  (total_spent : ℕ) 
  (h1 : long_sleeved_count = 4)
  (h2 : striped_count = 2)
  (h3 : striped_cost = 10)
  (h4 : total_spent = 80) :
  ∃ (long_sleeved_cost : ℕ), 
    long_sleeved_count * long_sleeved_cost + striped_count * striped_cost = total_spent ∧ 
    long_sleeved_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_long_sleeved_jersey_cost_l3887_388781


namespace NUMINAMATH_CALUDE_cos_two_x_value_l3887_388723

theorem cos_two_x_value (x : ℝ) (h : Real.sin (-x) = Real.sqrt 3 / 2) : 
  Real.cos (2 * x) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_two_x_value_l3887_388723


namespace NUMINAMATH_CALUDE_sequence_congruence_l3887_388778

/-- Sequence a_n defined recursively -/
def a : ℕ → ℤ → ℤ
  | 0, _ => 0
  | 1, _ => 1
  | (n + 2), k => 2 * k * a (n + 1) k - (k^2 + 1) * a n k

/-- Main theorem -/
theorem sequence_congruence (k : ℤ) (p : ℕ) (hp : Nat.Prime p) (hp_mod : p % 4 = 3) :
  (∀ n : ℕ, a (n + p^2 - 1) k ≡ a n k [ZMOD p]) ∧
  (∀ n : ℕ, a (n + p^3 - p) k ≡ a n k [ZMOD p^2]) := by
  sorry

#check sequence_congruence

end NUMINAMATH_CALUDE_sequence_congruence_l3887_388778


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3887_388713

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3887_388713


namespace NUMINAMATH_CALUDE_sample_is_reading_time_data_l3887_388737

/-- Represents a resident in the study -/
structure Resident where
  id : Nat
  readingTime : ℝ

/-- Represents the statistical study -/
structure ReadingStudy where
  population : Finset Resident
  sampleSize : Nat
  sample : Finset Resident

/-- Definition of a valid sample in the reading study -/
def validSample (study : ReadingStudy) : Prop :=
  study.sample.card = study.sampleSize ∧
  study.sample ⊆ study.population

/-- The main theorem about the sample definition -/
theorem sample_is_reading_time_data (study : ReadingStudy)
    (h_pop_size : study.population.card = 5000)
    (h_sample_size : study.sampleSize = 200)
    (h_valid_sample : validSample study) :
    ∃ (sample_data : Finset ℝ),
      sample_data = study.sample.image Resident.readingTime ∧
      sample_data.card = study.sampleSize :=
  sorry


end NUMINAMATH_CALUDE_sample_is_reading_time_data_l3887_388737


namespace NUMINAMATH_CALUDE_largest_quotient_l3887_388711

def digits : List Nat := [4, 2, 8, 1, 9]

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  (∃ (d1 d2 d3 d4 d5 : Nat),
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
    a = 100 * d1 + 10 * d2 + d3 ∧ b = 10 * d4 + d5)

theorem largest_quotient :
  ∀ (a b : Nat), is_valid_pair a b →
  ∃ (q : Nat), a / b = q ∧ q ≤ 82 ∧
  (∀ (c d : Nat), is_valid_pair c d → c / d ≤ q) :=
sorry

end NUMINAMATH_CALUDE_largest_quotient_l3887_388711


namespace NUMINAMATH_CALUDE_range_of_m_l3887_388798

/-- Given propositions p and q, where ¬q is a sufficient but not necessary condition for ¬p,
    prove that the range of values for m is m ≥ 1. -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, (x^2 + x - 2 > 0 ↔ x > 1 ∨ x < -2)) →
  (∀ x, (x ≤ m → x^2 + x - 2 ≤ 0) ∧ 
        ∃ y, (y^2 + y - 2 ≤ 0 ∧ y > m)) →
  m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3887_388798


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3887_388725

theorem largest_angle_in_triangle (α β γ : Real) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = (7/5) * 90 →  -- Two angles sum to 7/5 of a right angle
  β = α + 45 →  -- One angle is 45° larger than the other
  max α (max β γ) = 85.5 :=  -- The largest angle is 85.5°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3887_388725


namespace NUMINAMATH_CALUDE_ab_equals_twelve_l3887_388761

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

-- Define the complement of A
def complement_A : Set ℝ := {x | x < 3 ∨ x > 4}

-- Theorem statement
theorem ab_equals_twelve (a b : ℝ) : 
  A a b ∪ complement_A = Set.univ → a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_twelve_l3887_388761


namespace NUMINAMATH_CALUDE_double_quarter_four_percent_l3887_388788

theorem double_quarter_four_percent : (2 * (1/4 * (4/100))) = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_double_quarter_four_percent_l3887_388788


namespace NUMINAMATH_CALUDE_remaining_balance_l3887_388731

def house_price : ℝ := 100000

def down_payment_percentage : ℝ := 0.20

def parents_contribution_percentage : ℝ := 0.30

theorem remaining_balance (hp : ℝ) (dp : ℝ) (pc : ℝ) : 
  hp * (1 - dp) * (1 - pc) = 56000 :=
by
  sorry

#check remaining_balance house_price down_payment_percentage parents_contribution_percentage

end NUMINAMATH_CALUDE_remaining_balance_l3887_388731


namespace NUMINAMATH_CALUDE_singer_work_hours_l3887_388745

-- Define the number of songs
def num_songs : ℕ := 3

-- Define the number of days per song
def days_per_song : ℕ := 10

-- Define the total number of hours worked
def total_hours : ℕ := 300

-- Define the function to calculate hours per day
def hours_per_day (n s d t : ℕ) : ℚ :=
  t / (n * d)

-- Theorem statement
theorem singer_work_hours :
  hours_per_day num_songs days_per_song total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_singer_work_hours_l3887_388745


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3887_388795

def f (x : ℝ) := x^3 + x - 2

theorem tangent_parallel_points :
  {P : ℝ × ℝ | P.1 ^ 3 + P.1 - 2 = P.2 ∧ (3 * P.1 ^ 2 + 1 = 4)} =
  {(-1, -4), (1, 0)} := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3887_388795


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l3887_388784

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_sequence_l3887_388784


namespace NUMINAMATH_CALUDE_abc_value_l3887_388724

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (eq1 : a * (b + c) = 156)
  (eq2 : b * (c + a) = 168)
  (eq3 : c * (a + b) = 176) :
  a * b * c = 754 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3887_388724


namespace NUMINAMATH_CALUDE_colored_spanning_tree_existence_l3887_388708

/-- A colored edge in a graph -/
inductive ColoredEdge
  | Red
  | Green
  | Blue

/-- A graph with colored edges -/
structure ColoredGraph (V : Type) where
  edges : V → V → Option ColoredEdge

/-- A spanning tree of a graph -/
def SpanningTree (V : Type) := V → V → Prop

/-- Count the number of edges of a specific color in a spanning tree -/
def CountEdges (V : Type) (t : SpanningTree V) (c : ColoredEdge) : ℕ := sorry

/-- The main theorem -/
theorem colored_spanning_tree_existence
  (V : Type)
  (G : ColoredGraph V)
  (n : ℕ)
  (r v b : ℕ)
  (h_connected : sorry)  -- G is connected
  (h_vertex_count : sorry)  -- G has n+1 vertices
  (h_sum : r + v + b = n)
  (h_red_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Red = r)
  (h_green_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Green = v)
  (h_blue_tree : ∃ t : SpanningTree V, CountEdges V t ColoredEdge.Blue = b) :
  ∃ t : SpanningTree V,
    CountEdges V t ColoredEdge.Red = r ∧
    CountEdges V t ColoredEdge.Green = v ∧
    CountEdges V t ColoredEdge.Blue = b :=
  sorry

end NUMINAMATH_CALUDE_colored_spanning_tree_existence_l3887_388708


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l3887_388789

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l3887_388789


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3887_388791

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * B) (h3 : B = 15) : A = 120 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3887_388791


namespace NUMINAMATH_CALUDE_triangle_perimeter_strict_l3887_388732

theorem triangle_perimeter_strict (a b x : ℝ) : 
  a = 12 → b = 25 → a > 0 → b > 0 → x > 0 → 
  a + b > x → a + x > b → b + x > a → 
  a + b + x > 50 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_strict_l3887_388732


namespace NUMINAMATH_CALUDE_guide_is_native_l3887_388701

-- Define the two tribes
inductive Tribe
| Native
| Alien

-- Define a function to represent whether a statement is true or false
def isTruthful (t : Tribe) (s : Prop) : Prop :=
  match t with
  | Tribe.Native => s
  | Tribe.Alien => ¬s

-- Define the guide's statement
def guideStatement (encounteredTribe : Tribe) : Prop :=
  isTruthful encounteredTribe (encounteredTribe = Tribe.Native)

-- Theorem: The guide must be a native
theorem guide_is_native :
  ∀ (guideTribe : Tribe),
    (∀ (encounteredTribe : Tribe),
      isTruthful guideTribe (guideStatement encounteredTribe)) →
    guideTribe = Tribe.Native :=
by sorry


end NUMINAMATH_CALUDE_guide_is_native_l3887_388701


namespace NUMINAMATH_CALUDE_largest_fraction_l3887_388786

theorem largest_fraction :
  (202 : ℚ) / 403 > 5 / 11 ∧
  (202 : ℚ) / 403 > 7 / 16 ∧
  (202 : ℚ) / 403 > 23 / 50 ∧
  (202 : ℚ) / 403 > 99 / 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3887_388786
