import Mathlib

namespace NUMINAMATH_CALUDE_vector_dot_product_l3602_360293

/-- Given two vectors a and b in ℝ², prove that their dot product is 25 -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, 2) → a - (1/5 : ℝ) • b = (-2, 1) → a.1 * b.1 + a.2 * b.2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3602_360293


namespace NUMINAMATH_CALUDE_robin_water_consumption_l3602_360280

theorem robin_water_consumption 
  (morning : ℝ) 
  (afternoon : ℝ) 
  (evening : ℝ) 
  (night : ℝ) 
  (m : ℝ) 
  (e : ℝ) 
  (t : ℝ) 
  (h1 : morning = 7.5) 
  (h2 : afternoon = 9.25) 
  (h3 : evening = 5.75) 
  (h4 : night = 3.5) 
  (h5 : m = morning + afternoon) 
  (h6 : e = evening + night) 
  (h7 : t = m + e) : 
  t = 16.75 + 9.25 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_consumption_l3602_360280


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3602_360284

theorem magnitude_of_complex_power (z : ℂ) : 
  z = 4 + 2 * Complex.I * Real.sqrt 5 → Complex.abs (z^4) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3602_360284


namespace NUMINAMATH_CALUDE_symmetry_about_yOz_plane_l3602_360202

/-- The symmetry of a point about the yOz plane in a rectangular coordinate system -/
theorem symmetry_about_yOz_plane (x y z : ℝ) : 
  let original_point := (x, y, z)
  let symmetric_point := (-x, y, z)
  symmetric_point = (- (x : ℝ), y, z) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_yOz_plane_l3602_360202


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3602_360214

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {1, 2, 3, 4}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3602_360214


namespace NUMINAMATH_CALUDE_portias_school_size_l3602_360297

-- Define variables for the number of students in each school
variable (L : ℕ) -- Lara's high school
variable (P : ℕ) -- Portia's high school
variable (M : ℕ) -- Mia's high school

-- Define the conditions
axiom portia_students : P = 4 * L
axiom mia_students : M = 2 * L
axiom total_students : P + L + M = 4200

-- Theorem to prove
theorem portias_school_size : P = 2400 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_size_l3602_360297


namespace NUMINAMATH_CALUDE_correct_answers_count_l3602_360273

/-- Represents a mathematics contest with scoring rules and results. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  answered_problems : ℕ
  total_score : ℤ

/-- Theorem stating that given the contest conditions, 11 correct answers result in a score of 54. -/
theorem correct_answers_count (contest : MathContest) 
  (h1 : contest.total_problems = 15)
  (h2 : contest.correct_points = 6)
  (h3 : contest.incorrect_points = -3)
  (h4 : contest.answered_problems = contest.total_problems)
  (h5 : contest.total_score = 54) :
  ∃ (correct : ℕ), correct = 11 ∧ 
    contest.correct_points * correct + contest.incorrect_points * (contest.total_problems - correct) = contest.total_score :=
by sorry

end NUMINAMATH_CALUDE_correct_answers_count_l3602_360273


namespace NUMINAMATH_CALUDE_rectangle_to_cylinder_volume_l3602_360245

/-- The volume of a cylinder formed by rolling a rectangle with length 6 and width 3 -/
theorem rectangle_to_cylinder_volume :
  ∃ (V : ℝ), (V = 27 / π ∨ V = 27 / (4 * π)) ∧
  ∃ (R h : ℝ), (R * h = 18 ∨ R * h = 9) ∧ V = π * R^2 * h := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_cylinder_volume_l3602_360245


namespace NUMINAMATH_CALUDE_some_number_value_l3602_360263

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : (27 / 4) * x - some_number = 3 * x + 27) 
  (h2 : x = 12) : 
  some_number = 18 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3602_360263


namespace NUMINAMATH_CALUDE_concentric_circles_radii_l3602_360251

theorem concentric_circles_radii 
  (chord_length : ℝ) 
  (ring_width : ℝ) 
  (h_chord : chord_length = 32) 
  (h_width : ring_width = 8) :
  ∃ (r R : ℝ), 
    r > 0 ∧ 
    R > r ∧
    R = r + ring_width ∧
    (r + ring_width)^2 = r^2 + (chord_length/2)^2 ∧
    r = 12 ∧ 
    R = 20 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_l3602_360251


namespace NUMINAMATH_CALUDE_last_date_theorem_l3602_360275

/-- Represents a date in DD.MM.YYYY format -/
structure Date :=
  (day : Nat)
  (month : Nat)
  (year : Nat)

/-- Check if a date is valid -/
def is_valid_date (d : Date) : Bool :=
  d.day ≥ 1 && d.day ≤ 31 && d.month ≥ 1 && d.month ≤ 12 && d.year ≥ 1

/-- Get the set of digits used in a date -/
def date_digits (d : Date) : Finset Nat :=
  sorry

/-- Check if a date is before another date -/
def is_before (d1 d2 : Date) : Bool :=
  sorry

/-- Find the last date before a given date with the same set of digits -/
def last_date_with_same_digits (d : Date) : Date :=
  sorry

theorem last_date_theorem (current_date : Date) :
  let target_date := Date.mk 15 12 2012
  current_date = Date.mk 22 11 2015 →
  is_valid_date target_date ∧
  is_before target_date current_date ∧
  date_digits target_date = date_digits current_date ∧
  (∀ d : Date, is_valid_date d ∧ is_before d current_date ∧ date_digits d = date_digits current_date →
    is_before d target_date ∨ d = target_date) :=
by sorry

end NUMINAMATH_CALUDE_last_date_theorem_l3602_360275


namespace NUMINAMATH_CALUDE_num_valid_colorings_is_7776_l3602_360267

/-- A graph representing the extended figure described in the problem -/
def ExtendedFigureGraph : Type := Unit

/-- The number of vertices in the extended figure graph -/
def num_vertices : Nat := 12

/-- The number of available colors -/
def num_colors : Nat := 4

/-- A function that determines if two vertices are adjacent in the extended figure graph -/
def are_adjacent (v1 v2 : Fin num_vertices) : Bool := sorry

/-- A coloring of the graph is a function from vertices to colors -/
def Coloring := Fin num_vertices → Fin num_colors

/-- A predicate that determines if a coloring is valid (no adjacent vertices have the same color) -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v1 v2 : Fin num_vertices, are_adjacent v1 v2 → c v1 ≠ c v2

/-- The number of valid colorings for the extended figure graph -/
def num_valid_colorings : Nat := sorry

/-- The main theorem stating that the number of valid colorings is 7776 -/
theorem num_valid_colorings_is_7776 : num_valid_colorings = 7776 := by sorry

end NUMINAMATH_CALUDE_num_valid_colorings_is_7776_l3602_360267


namespace NUMINAMATH_CALUDE_marshas_pay_per_mile_l3602_360288

theorem marshas_pay_per_mile :
  let first_package_miles : ℝ := 10
  let second_package_miles : ℝ := 28
  let third_package_miles : ℝ := second_package_miles / 2
  let total_miles : ℝ := first_package_miles + second_package_miles + third_package_miles
  let total_pay : ℝ := 104
  total_pay / total_miles = 2 := by
sorry

end NUMINAMATH_CALUDE_marshas_pay_per_mile_l3602_360288


namespace NUMINAMATH_CALUDE_koala_weight_in_grams_l3602_360248

-- Define the conversion rate from kg to g
def kg_to_g : ℕ → ℕ := (· * 1000)

-- Define the weight of the baby koala
def koala_weight_kg : ℕ := 2
def koala_weight_extra_g : ℕ := 460

-- Theorem: The total weight of the baby koala in grams is 2460
theorem koala_weight_in_grams : 
  kg_to_g koala_weight_kg + koala_weight_extra_g = 2460 := by
  sorry

end NUMINAMATH_CALUDE_koala_weight_in_grams_l3602_360248


namespace NUMINAMATH_CALUDE_percent_calculation_l3602_360277

theorem percent_calculation (x : ℝ) (h : 0.20 * x = 200) : 1.20 * x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l3602_360277


namespace NUMINAMATH_CALUDE_lolita_weekday_milk_l3602_360239

/-- The number of milk boxes Lolita drinks on a single weekday -/
def weekday_milk : ℕ := 3

/-- The number of milk boxes Lolita drinks on Saturday -/
def saturday_milk : ℕ := 2 * weekday_milk

/-- The number of milk boxes Lolita drinks on Sunday -/
def sunday_milk : ℕ := 3 * weekday_milk

/-- The total number of milk boxes Lolita drinks in a week -/
def total_weekly_milk : ℕ := 30

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

theorem lolita_weekday_milk :
  weekdays * weekday_milk = 15 ∧
  weekdays * weekday_milk + saturday_milk + sunday_milk = total_weekly_milk :=
sorry

end NUMINAMATH_CALUDE_lolita_weekday_milk_l3602_360239


namespace NUMINAMATH_CALUDE_negation_equivalence_l3602_360243

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3602_360243


namespace NUMINAMATH_CALUDE_angle_B_in_triangle_l3602_360298

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_B_in_triangle (t : Triangle) :
  t.a = 4 →
  t.b = 2 * Real.sqrt 2 →
  t.A = π / 4 →
  t.B = π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_B_in_triangle_l3602_360298


namespace NUMINAMATH_CALUDE_shorter_leg_length_in_30_60_90_triangle_l3602_360292

theorem shorter_leg_length_in_30_60_90_triangle (median_length : ℝ) :
  median_length = 5 * Real.sqrt 3 →
  ∃ (shorter_leg hypotenuse : ℝ),
    shorter_leg = 5 ∧
    hypotenuse = 2 * shorter_leg ∧
    median_length = hypotenuse / 2 :=
by sorry

end NUMINAMATH_CALUDE_shorter_leg_length_in_30_60_90_triangle_l3602_360292


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3602_360253

/-- 
Given an arithmetic sequence where:
- a₇ is the 7th term
- d is the common difference
- a₁ is the first term
- a₂ is the second term

This theorem states that if a₇ = 17 and d = 2, then a₁ * a₂ = 35.
-/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (a 7 = 17) → (∀ n, a (n + 1) - a n = d) → (d = 2) → (a 1 * a 2 = 35) := by
  sorry

#check arithmetic_sequence_product

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3602_360253


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3602_360238

theorem inequality_system_solutions :
  let S : Set ℤ := {x | x ≥ 0 ∧ 2*x + 5 ≤ 3*(x + 2) ∧ 2*x - (1 + 3*x)/2 < 1}
  S = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3602_360238


namespace NUMINAMATH_CALUDE_greatest_four_digit_number_l3602_360216

theorem greatest_four_digit_number : ∃ (n : ℕ), 
  (n = 9997) ∧ 
  (n < 10000) ∧ 
  (∃ (k : ℕ), n = 7 * k + 1) ∧ 
  (∃ (j : ℕ), n = 8 * j + 5) ∧ 
  (∀ (m : ℕ), m < 10000 → (∃ (k : ℕ), m = 7 * k + 1) → (∃ (j : ℕ), m = 8 * j + 5) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_number_l3602_360216


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l3602_360257

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d * |x^2 - y^2| + x + y ≥ x^2 + y^2) ∧
  (∀ (d' : ℝ), d' > 0 → 
    (∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → Real.sqrt (x^2 * y^2) + d' * |x^2 - y^2| + x + y ≥ x^2 + y^2) → 
    d ≤ d') ∧
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l3602_360257


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3602_360287

theorem expression_equals_negative_one (a y : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 2*y) (h3 : a ≠ -2*y) :
  (a / (a + 2*y) + y / (a - 2*y)) / (y / (a + 2*y) - a / (a - 2*y)) = -1 ↔ y = -a/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3602_360287


namespace NUMINAMATH_CALUDE_three_subject_average_l3602_360236

/-- Given that the average score of Korean and mathematics is 86 points,
    and the English score is 98 points, prove that the average score
    of all three subjects is 90 points. -/
theorem three_subject_average (korean math english : ℝ) : 
  (korean + math) / 2 = 86 → 
  english = 98 → 
  (korean + math + english) / 3 = 90 := by
sorry

end NUMINAMATH_CALUDE_three_subject_average_l3602_360236


namespace NUMINAMATH_CALUDE_jane_started_at_18_l3602_360264

/-- Represents Jane's babysitting career --/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  oldest_babysat_current_age : ℕ
  start_age : ℕ

/-- Checks if the babysitting career satisfies all conditions --/
def is_valid_career (career : BabysittingCareer) : Prop :=
  career.current_age = 34 ∧
  career.years_since_stopped = 12 ∧
  career.oldest_babysat_current_age = 25 ∧
  career.start_age ≤ career.current_age - career.years_since_stopped ∧
  ∀ (child_age : ℕ), child_age ≤ career.oldest_babysat_current_age →
    2 * (child_age - career.years_since_stopped) ≤ career.current_age - career.years_since_stopped

theorem jane_started_at_18 :
  ∃ (career : BabysittingCareer), is_valid_career career ∧ career.start_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_jane_started_at_18_l3602_360264


namespace NUMINAMATH_CALUDE_science_club_board_selection_l3602_360270

theorem science_club_board_selection (total_members : Nat) (prev_served : Nat) (board_size : Nat)
  (h1 : total_members = 20)
  (h2 : prev_served = 9)
  (h3 : board_size = 6) :
  (Nat.choose total_members board_size) - (Nat.choose (total_members - prev_served) board_size) = 38298 := by
  sorry

end NUMINAMATH_CALUDE_science_club_board_selection_l3602_360270


namespace NUMINAMATH_CALUDE_function_value_inequality_l3602_360206

theorem function_value_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f (-x))
  (h2 : ∀ x, 1 < x ∧ x < 2 → f x > 0) :
  f (-1.5) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_inequality_l3602_360206


namespace NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_l3602_360222

theorem simplify_sqrt_x_squared_y (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_x_squared_y_l3602_360222


namespace NUMINAMATH_CALUDE_locus_and_fixed_point_l3602_360262

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.1^2/4 - p.2^2 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

-- Define the line x = 1
def line_x_eq_1 : Set (ℝ × ℝ) := {p | p.1 = 1}

-- Define the property of point M
def is_valid_M (M : ℝ × ℝ) : Prop :=
  let slope_AM := (M.2 - A.2) / (M.1 - A.1)
  let slope_BM := (M.2 - B.2) / (M.1 - B.1)
  slope_AM * slope_BM = 1/4

-- Main theorem
theorem locus_and_fixed_point :
  (∀ M, is_valid_M M → M ∈ C) ∧
  (∀ T ∈ line_x_eq_1, 
    ∃ P Q, P ∈ C ∧ Q ∈ C ∧ 
    (P.2 - A.2) / (P.1 - A.1) = (T.2 - A.2) / (T.1 - A.1) ∧
    (Q.2 - B.2) / (Q.1 - B.1) = (T.2 - B.2) / (T.1 - B.1) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = (0 - P.2) / (4 - P.1)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_point_l3602_360262


namespace NUMINAMATH_CALUDE_donation_conversion_l3602_360279

theorem donation_conversion (usd_donation : ℝ) (exchange_rate : ℝ) (cny_donation : ℝ) : 
  usd_donation = 1.2 →
  exchange_rate = 6.25 →
  cny_donation = usd_donation * exchange_rate →
  cny_donation = 7.5 :=
by sorry

end NUMINAMATH_CALUDE_donation_conversion_l3602_360279


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3602_360221

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 + 12*x = 64) : 
  ∃ y : ℝ, y^2 + 12*y = 64 ∧ x + y = -12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3602_360221


namespace NUMINAMATH_CALUDE_maddies_mom_coffee_cost_l3602_360225

/-- Represents the weekly coffee consumption and cost for Maddie's mom -/
structure CoffeeConsumption where
  cups_per_day : ℕ
  beans_per_cup : ℚ
  beans_per_bag : ℚ
  cost_per_bag : ℚ
  milk_per_week : ℚ
  cost_per_gallon_milk : ℚ

/-- Calculates the weekly cost of coffee -/
def weekly_coffee_cost (c : CoffeeConsumption) : ℚ :=
  let beans_per_week := c.cups_per_day * c.beans_per_cup * 7
  let bags_per_week := beans_per_week / c.beans_per_bag
  let coffee_cost := bags_per_week * c.cost_per_bag
  let milk_cost := c.milk_per_week * c.cost_per_gallon_milk
  coffee_cost + milk_cost

/-- Theorem stating that Maddie's mom's weekly coffee cost is $18 -/
theorem maddies_mom_coffee_cost :
  let c : CoffeeConsumption := {
    cups_per_day := 2,
    beans_per_cup := 3/2,
    beans_per_bag := 21/2,
    cost_per_bag := 8,
    milk_per_week := 1/2,
    cost_per_gallon_milk := 4
  }
  weekly_coffee_cost c = 18 := by
  sorry

end NUMINAMATH_CALUDE_maddies_mom_coffee_cost_l3602_360225


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_six_l3602_360207

theorem fourth_root_over_sixth_root_of_six (x : ℝ) (h : x = 6) :
  (x^(1/4)) / (x^(1/6)) = x^(1/12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_six_l3602_360207


namespace NUMINAMATH_CALUDE_fresh_fruit_weight_l3602_360235

theorem fresh_fruit_weight (total_fruit : ℕ) (fresh_ratio frozen_ratio : ℕ) 
  (h_total : total_fruit = 15000)
  (h_ratio : fresh_ratio = 7 ∧ frozen_ratio = 3) :
  (fresh_ratio * total_fruit) / (fresh_ratio + frozen_ratio) = 10500 :=
by sorry

end NUMINAMATH_CALUDE_fresh_fruit_weight_l3602_360235


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3602_360265

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 2 ∧ 
                x2 = (-1 - Real.sqrt 17) / 2 ∧ 
                x1^2 + x1 - 4 = 0 ∧ 
                x2^2 + x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 1 ∧ 
                x2 = 2 ∧ 
                (2*x1 + 1)^2 + 15 = 8*(2*x1 + 1) ∧ 
                (2*x2 + 1)^2 + 15 = 8*(2*x2 + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3602_360265


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3602_360205

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) :
  x + y ≤ 49 ∧ ∃ (a b : ℕ+), a * b = 48 ∧ a + b = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3602_360205


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3602_360289

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^2 - 4 * t + 3) * (-4 * t^2 + 2 * t - 6) = 
  -12 * t^4 + 22 * t^3 - 38 * t^2 + 30 * t - 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3602_360289


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3602_360209

theorem arithmetic_square_root_of_four :
  Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l3602_360209


namespace NUMINAMATH_CALUDE_brenda_skittles_l3602_360203

theorem brenda_skittles (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 7 → bought = 8 → final = initial + bought → final = 15 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l3602_360203


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l3602_360286

/-- Represents a gear in the system -/
structure Gear where
  teeth : ℕ
  angular_speed : ℝ

/-- Represents the system of gears -/
structure GearSystem where
  P : Gear
  Q : Gear
  R : Gear
  efficiency : ℝ

/-- The theorem stating the correct proportion of angular speeds -/
theorem gear_speed_proportion (sys : GearSystem) 
  (h1 : sys.efficiency = 0.9)
  (h2 : sys.P.teeth * sys.P.angular_speed = sys.Q.teeth * sys.Q.angular_speed)
  (h3 : sys.R.angular_speed = sys.efficiency * sys.Q.angular_speed) :
  ∃ (k : ℝ), k > 0 ∧ 
    sys.P.angular_speed = k * sys.Q.teeth ∧
    sys.Q.angular_speed = k * sys.P.teeth ∧
    sys.R.angular_speed = k * sys.efficiency * sys.P.teeth :=
sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l3602_360286


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3602_360272

theorem triangle_angle_calculation (x : ℝ) : 
  x > 0 ∧ 
  40 + 3 * x + x = 180 →
  x = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3602_360272


namespace NUMINAMATH_CALUDE_water_carriers_capacity_l3602_360211

/-- Represents the water-carrying capacity and trip ratio of two people --/
structure WaterCarriers where
  bucket_capacity : ℕ
  jack_buckets_per_trip : ℕ
  jill_buckets_per_trip : ℕ
  jack_trips_ratio : ℕ
  jill_trips_ratio : ℕ
  jill_total_trips : ℕ

/-- Calculates the total capacity of water carried by both people --/
def total_capacity (w : WaterCarriers) : ℕ :=
  let jack_trips := w.jack_trips_ratio * w.jill_total_trips / w.jill_trips_ratio
  let jack_capacity := w.bucket_capacity * w.jack_buckets_per_trip * jack_trips
  let jill_capacity := w.bucket_capacity * w.jill_buckets_per_trip * w.jill_total_trips
  jack_capacity + jill_capacity

/-- The theorem states that given the specified conditions, the total capacity is 600 gallons --/
theorem water_carriers_capacity : 
  ∀ (w : WaterCarriers), 
  w.bucket_capacity = 5 ∧ 
  w.jack_buckets_per_trip = 2 ∧ 
  w.jill_buckets_per_trip = 1 ∧ 
  w.jack_trips_ratio = 3 ∧ 
  w.jill_trips_ratio = 2 ∧ 
  w.jill_total_trips = 30 → 
  total_capacity w = 600 := by
  sorry

end NUMINAMATH_CALUDE_water_carriers_capacity_l3602_360211


namespace NUMINAMATH_CALUDE_jellybean_ratio_l3602_360269

/-- Proves the ratio of jellybeans Lorelai has eaten to the total jellybeans Rory and Gigi have -/
theorem jellybean_ratio (gigi_jellybeans : ℕ) (rory_extra_jellybeans : ℕ) (lorelai_jellybeans : ℕ) :
  gigi_jellybeans = 15 →
  rory_extra_jellybeans = 30 →
  lorelai_jellybeans = 180 →
  ∃ (m : ℕ), m * (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans)) = lorelai_jellybeans →
  (lorelai_jellybeans : ℚ) / (gigi_jellybeans + (gigi_jellybeans + rory_extra_jellybeans) : ℚ) = 3 := by
  sorry

#check jellybean_ratio

end NUMINAMATH_CALUDE_jellybean_ratio_l3602_360269


namespace NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l3602_360261

theorem x_minus_q_equals_three_minus_two_q (x q : ℝ) 
  (h1 : |x - 3| = q) 
  (h2 : x < 3) : 
  x - q = 3 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_x_minus_q_equals_three_minus_two_q_l3602_360261


namespace NUMINAMATH_CALUDE_irrational_equation_root_l3602_360208

theorem irrational_equation_root (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ Real.sqrt (2 * x + m) = x) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_irrational_equation_root_l3602_360208


namespace NUMINAMATH_CALUDE_power_congruence_l3602_360254

theorem power_congruence (h : 5^200 ≡ 1 [ZMOD 1000]) :
  5^6000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l3602_360254


namespace NUMINAMATH_CALUDE_f_properties_l3602_360227

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1/a) * Real.exp (a * x)

theorem f_properties (h : a ≠ 0) :
  -- Part I
  (a = 1/2 → (f a x = 0 ↔ x = -1 ∨ x = 2)) ∧
  -- Part II
  (∀ x, f a x = 0 → x = 1 ∨ x = -2/a) ∧
  -- Part III
  (a > 0 → (∀ x, f a x + 2/a ≥ 0) ↔ 0 < a ∧ a ≤ Real.log 2) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3602_360227


namespace NUMINAMATH_CALUDE_distribute_unique_items_l3602_360229

theorem distribute_unique_items 
  (num_items : ℕ) 
  (num_recipients : ℕ) 
  (h1 : num_items = 6) 
  (h2 : num_recipients = 8) :
  (num_recipients ^ num_items : ℕ) = 262144 := by
  sorry

end NUMINAMATH_CALUDE_distribute_unique_items_l3602_360229


namespace NUMINAMATH_CALUDE_rachel_homework_pages_l3602_360220

theorem rachel_homework_pages (math_pages reading_pages biology_pages : ℕ) 
  (h1 : math_pages = 2)
  (h2 : reading_pages = 3)
  (h3 : biology_pages = 10) :
  math_pages + reading_pages + biology_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_pages_l3602_360220


namespace NUMINAMATH_CALUDE_eagle_speed_proof_l3602_360296

/-- The speed of the eagle in miles per hour -/
def eagle_speed : ℝ := 15

/-- The speed of the falcon in miles per hour -/
def falcon_speed : ℝ := 46

/-- The speed of the pelican in miles per hour -/
def pelican_speed : ℝ := 33

/-- The speed of the hummingbird in miles per hour -/
def hummingbird_speed : ℝ := 30

/-- The time all birds flew in hours -/
def flight_time : ℝ := 2

/-- The total distance covered by all birds in miles -/
def total_distance : ℝ := 248

theorem eagle_speed_proof :
  eagle_speed * flight_time +
  falcon_speed * flight_time +
  pelican_speed * flight_time +
  hummingbird_speed * flight_time =
  total_distance :=
sorry

end NUMINAMATH_CALUDE_eagle_speed_proof_l3602_360296


namespace NUMINAMATH_CALUDE_sqrt_pattern_l3602_360213

theorem sqrt_pattern (n : ℕ+) : 
  Real.sqrt (1 + 1 / (n : ℝ)^2 + 1 / ((n + 1) : ℝ)^2) = 1 + 1 / ((n : ℝ) * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pattern_l3602_360213


namespace NUMINAMATH_CALUDE_rooks_placement_formula_l3602_360204

/-- The number of ways to place k non-attacking rooks on an n × n chessboard -/
def rooks_placement (n k : ℕ) : ℕ :=
  Nat.choose n k * Nat.descFactorial n k

/-- An n × n chessboard -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

theorem rooks_placement_formula {n k : ℕ} (C : Chessboard n) (h : k ≤ n) :
  rooks_placement n k = Nat.choose n k * Nat.descFactorial n k := by
  sorry

end NUMINAMATH_CALUDE_rooks_placement_formula_l3602_360204


namespace NUMINAMATH_CALUDE_jasons_stove_repair_cost_l3602_360256

theorem jasons_stove_repair_cost (stove_cost : ℝ) (wall_repair_ratio : ℝ) : 
  stove_cost = 1200 →
  wall_repair_ratio = 1 / 6 →
  stove_cost + (wall_repair_ratio * stove_cost) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_jasons_stove_repair_cost_l3602_360256


namespace NUMINAMATH_CALUDE_four_digit_sum_plus_2001_l3602_360223

theorem four_digit_sum_plus_2001 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧
  n = (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10) + 2001 ∧
  n = 1977 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_plus_2001_l3602_360223


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3602_360218

def A : Set ℝ := {0, 1, 2, 3, 4}

def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3602_360218


namespace NUMINAMATH_CALUDE_cos_pi_twelfth_l3602_360278

theorem cos_pi_twelfth : Real.cos (π / 12) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_twelfth_l3602_360278


namespace NUMINAMATH_CALUDE_art_collection_cost_l3602_360276

/-- The total cost of John's art collection --/
def total_cost (first_three_cost : ℝ) (fourth_piece_cost : ℝ) : ℝ :=
  first_three_cost + fourth_piece_cost

/-- The cost of the fourth piece of art --/
def fourth_piece_cost (single_piece_cost : ℝ) : ℝ :=
  single_piece_cost * 1.5

theorem art_collection_cost :
  ∀ (single_piece_cost : ℝ),
    single_piece_cost > 0 →
    single_piece_cost * 3 = 45000 →
    total_cost (single_piece_cost * 3) (fourth_piece_cost single_piece_cost) = 67500 := by
  sorry

end NUMINAMATH_CALUDE_art_collection_cost_l3602_360276


namespace NUMINAMATH_CALUDE_speed_limit_exceeders_l3602_360260

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := 50

/-- The percentage of all motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 40

/-- The percentage of speed limit exceeders who do not receive tickets -/
def no_ticket_percent : ℝ := 20

theorem speed_limit_exceeders :
  exceed_limit_percent = 50 :=
by
  sorry

#check speed_limit_exceeders

end NUMINAMATH_CALUDE_speed_limit_exceeders_l3602_360260


namespace NUMINAMATH_CALUDE_vector_operation_l3602_360228

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 4) → b = (-1, 1) → 2 • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3602_360228


namespace NUMINAMATH_CALUDE_boys_in_second_grade_is_20_l3602_360295

/-- The number of boys in the second grade -/
def boys_in_second_grade : ℕ := sorry

/-- The number of girls in the second grade -/
def girls_in_second_grade : ℕ := 11

/-- The total number of students in the second grade -/
def students_in_second_grade : ℕ := boys_in_second_grade + girls_in_second_grade

/-- The number of students in the third grade -/
def students_in_third_grade : ℕ := 2 * students_in_second_grade

/-- The total number of students in grades 2 and 3 -/
def total_students : ℕ := 93

theorem boys_in_second_grade_is_20 :
  boys_in_second_grade = 20 ∧
  students_in_second_grade + students_in_third_grade = total_students :=
sorry

end NUMINAMATH_CALUDE_boys_in_second_grade_is_20_l3602_360295


namespace NUMINAMATH_CALUDE_water_usage_difference_l3602_360266

theorem water_usage_difference (total_water plants_water : ℕ) : 
  total_water = 65 →
  plants_water < 14 →
  24 * 2 = 65 - 14 - plants_water →
  7 - plants_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_difference_l3602_360266


namespace NUMINAMATH_CALUDE_bracket_computation_l3602_360299

-- Define the operation [x, y, z]
def bracket (x y z : ℚ) : ℚ := (x + y) / z

-- Theorem statement
theorem bracket_computation :
  bracket (bracket 120 60 180) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_bracket_computation_l3602_360299


namespace NUMINAMATH_CALUDE_max_log_sum_l3602_360255

theorem max_log_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 6) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ 3 * Real.log 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 6 ∧ Real.log x + 2 * Real.log y = 3 * Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l3602_360255


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l3602_360210

theorem arithmetic_geometric_sequences :
  -- Arithmetic sequence
  ∃ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (a 8 = 6 ∧ a 10 = 0) →
    (∀ n, a n = 30 - 3 * n) ∧
    (∀ n, S n = -3/2 * n^2 + 57/2 * n) ∧
    (∀ n, n ≠ 9 ∧ n ≠ 10 → S n < S 9) ∧
  -- Geometric sequence
  ∃ (b : ℕ → ℝ) (T : ℕ → ℝ),
    (b 1 = 1/2 ∧ b 4 = 4) →
    (∀ n, b n = 2^(n-2)) ∧
    (∀ n, T n = 2^(n-1) - 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequences_l3602_360210


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l3602_360294

/-- A truncated cube is a polyhedron obtained by truncating each vertex of a cube
    such that a small square face replaces each vertex, and no cutting planes
    intersect each other inside the cube. -/
structure TruncatedCube where
  -- We don't need to define the internal structure, just the concept

/-- The number of edges in a truncated cube -/
def num_edges (tc : TruncatedCube) : ℕ := 16

/-- Theorem stating that the number of edges in a truncated cube is 16 -/
theorem truncated_cube_edges (tc : TruncatedCube) :
  num_edges tc = 16 := by sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l3602_360294


namespace NUMINAMATH_CALUDE_linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l3602_360234

/-- A function is a bounded multiple function if there exists a constant M > 0 
    such that |f(x)| ≤ M|x| for all real x. -/
def BoundedMultipleFunction (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≤ M * |x|

/-- The function f(x) = 2x is a bounded multiple function. -/
theorem linear_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ 2 * x) := by
  sorry

/-- The function f(x) = x/(x^2 - x + 3) is a bounded multiple function. -/
theorem rational_is_bounded_multiple : BoundedMultipleFunction (fun x ↦ x / (x^2 - x + 3)) := by
  sorry

/-- An odd function f(x) defined on ℝ that satisfies |f(x₁) - f(x₂)| ≤ 2|x₁ - x₂| 
    for all x₁, x₂ ∈ ℝ is a bounded multiple function. -/
theorem odd_lipschitz_is_bounded_multiple 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_lipschitz : ∀ x₁ x₂, |f x₁ - f x₂| ≤ 2 * |x₁ - x₂|) : 
  BoundedMultipleFunction f := by
  sorry

end NUMINAMATH_CALUDE_linear_is_bounded_multiple_rational_is_bounded_multiple_odd_lipschitz_is_bounded_multiple_l3602_360234


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l3602_360271

theorem sugar_solution_percentage (x : ℝ) :
  (3/4 * x + 1/4 * 26 = 14) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l3602_360271


namespace NUMINAMATH_CALUDE_ball_probability_l3602_360291

/-- Given a bag of 100 balls with specified colors, prove the probability of choosing a ball that is neither red nor purple -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h1 : total = 100)
  (h2 : white = 50)
  (h3 : green = 30)
  (h4 : yellow = 8)
  (h5 : red = 9)
  (h6 : purple = 3)
  (h7 : total = white + green + yellow + red + purple) :
  (white + green + yellow : ℚ) / total = 88 / 100 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l3602_360291


namespace NUMINAMATH_CALUDE_sequence_problem_l3602_360237

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a : a 1000 + a 1018 = 2 * Real.pi)
  (h_b : b 6 * b 2012 = 2) :
  Real.tan ((a 2 + a 2016) / (1 + b 3 * b 2015)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3602_360237


namespace NUMINAMATH_CALUDE_jean_side_spots_l3602_360232

/-- Represents the number of spots on different parts of Jean the jaguar's body. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the conditions. -/
theorem jean_side_spots (j : JeanSpots) 
    (h1 : j.upperTorso = j.total / 2)
    (h2 : j.backHindquarters = j.total / 3)
    (h3 : j.upperTorso = 30)
    (h4 : j.total = j.upperTorso + j.backHindquarters + j.sides) :
    j.sides = 10 := by
  sorry

end NUMINAMATH_CALUDE_jean_side_spots_l3602_360232


namespace NUMINAMATH_CALUDE_star_calculation_l3602_360233

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 3 ⋆ (5 ⋆ 6) = -112 -/
theorem star_calculation : star 3 (star 5 6) = -112 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3602_360233


namespace NUMINAMATH_CALUDE_base_c_sum_equals_47_l3602_360212

/-- Represents a number in base c -/
structure BaseC (c : ℕ) where
  digits : List ℕ
  valid : ∀ d ∈ digits, d < c

/-- Converts a base c number to its decimal (base 10) representation -/
def to_decimal (c : ℕ) (n : BaseC c) : ℕ := sorry

/-- Converts a decimal (base 10) number to its base c representation -/
def from_decimal (c : ℕ) (n : ℕ) : BaseC c := sorry

/-- Multiplies two numbers in base c -/
def mul_base_c (c : ℕ) (a b : BaseC c) : BaseC c := sorry

/-- Adds two numbers in base c -/
def add_base_c (c : ℕ) (a b : BaseC c) : BaseC c := sorry

theorem base_c_sum_equals_47 (c : ℕ) 
  (h_prod : mul_base_c c (mul_base_c c (from_decimal c 13) (from_decimal c 18)) (from_decimal c 17) = from_decimal c 4357) :
  let s := add_base_c c (add_base_c c (from_decimal c 13) (from_decimal c 18)) (from_decimal c 17)
  s = from_decimal c 47 := by
  sorry

end NUMINAMATH_CALUDE_base_c_sum_equals_47_l3602_360212


namespace NUMINAMATH_CALUDE_tiles_cut_to_square_and_rectangle_l3602_360290

/-- Represents a rectangular tile with width and height -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents a rectangle formed by tiles -/
structure Rectangle where
  width : ℝ
  height : ℝ
  tiles : List Tile

/-- Theorem stating that tiles can be cut to form a square and a rectangle -/
theorem tiles_cut_to_square_and_rectangle 
  (n : ℕ) 
  (original : Rectangle) 
  (h_unequal_sides : original.width ≠ original.height) 
  (h_tile_count : original.tiles.length = n) :
  ∃ (square : Rectangle) (remaining : Rectangle),
    square.width = square.height ∧
    square.tiles.length = n ∧
    remaining.tiles.length = n ∧
    (∀ t ∈ original.tiles, ∃ t1 t2, t1 ∈ square.tiles ∧ t2 ∈ remaining.tiles) :=
sorry

end NUMINAMATH_CALUDE_tiles_cut_to_square_and_rectangle_l3602_360290


namespace NUMINAMATH_CALUDE_opposite_of_difference_l3602_360281

theorem opposite_of_difference (a b : ℝ) : -(a - b) = b - a := by sorry

end NUMINAMATH_CALUDE_opposite_of_difference_l3602_360281


namespace NUMINAMATH_CALUDE_minor_premise_identification_l3602_360215

-- Define the types
def Shape : Type := String

-- Define the properties
def IsRectangle (s : Shape) : Prop := s = "rectangle"
def IsParallelogram (s : Shape) : Prop := s = "parallelogram"
def IsTriangle (s : Shape) : Prop := s = "triangle"

-- Define the syllogism statements
def MajorPremise : Prop := ∀ s : Shape, IsRectangle s → IsParallelogram s
def MinorPremise : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s
def Conclusion : Prop := ∃ s : Shape, IsTriangle s ∧ ¬IsRectangle s

-- Theorem to prove
theorem minor_premise_identification :
  MinorPremise = (∃ s : Shape, IsTriangle s ∧ ¬IsParallelogram s) :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l3602_360215


namespace NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l3602_360250

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 2 * red
  let new_total := blue + new_red
  new_red / new_total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l3602_360250


namespace NUMINAMATH_CALUDE_max_min_on_interval_l3602_360201

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

theorem max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = min) ∧
    max = 5 ∧ min = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l3602_360201


namespace NUMINAMATH_CALUDE_dads_toothpaste_usage_l3602_360240

/-- Represents the amount of toothpaste used by Anne's dad at each brushing -/
def dads_toothpaste_use : ℝ := 3

/-- Theorem stating that Anne's dad uses 3 grams of toothpaste at each brushing -/
theorem dads_toothpaste_usage 
  (total_toothpaste : ℝ) 
  (moms_usage : ℝ)
  (kids_usage : ℝ)
  (brushings_per_day : ℕ)
  (days_to_empty : ℕ)
  (h1 : total_toothpaste = 105)
  (h2 : moms_usage = 2)
  (h3 : kids_usage = 1)
  (h4 : brushings_per_day = 3)
  (h5 : days_to_empty = 5)
  : dads_toothpaste_use = 3 := by
  sorry

#check dads_toothpaste_usage

end NUMINAMATH_CALUDE_dads_toothpaste_usage_l3602_360240


namespace NUMINAMATH_CALUDE_investment_problem_l3602_360244

/-- The investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (z_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (total_time : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : z_investment = 48000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032)
  (h6 : total_time = 12) :
  ∃ y_investment : ℕ, 
    y_investment * total_time * (total_profit - z_profit_share) = 
    x_investment * total_time * z_profit_share - 
    z_investment * (total_time - z_join_time) * (total_profit - z_profit_share) ∧
    y_investment = 25000 := by
  sorry


end NUMINAMATH_CALUDE_investment_problem_l3602_360244


namespace NUMINAMATH_CALUDE_triangle_properties_l3602_360230

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a * t.b = 6) :
  t.C = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 37 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3602_360230


namespace NUMINAMATH_CALUDE_nandan_earning_is_2000_l3602_360226

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  total_gain : ℝ

/-- Calculates Nandan's earning based on the given business investment scenario -/
def nandan_earning (b : BusinessInvestment) : ℝ :=
  b.nandan_investment * b.nandan_time

/-- Theorem stating that Nandan's earning is 2000 given the specified conditions -/
theorem nandan_earning_is_2000 (b : BusinessInvestment) 
  (h1 : b.total_gain = 26000)
  (h2 : b.nandan_investment * b.nandan_time + 
        (4 * b.nandan_investment) * (3 * b.nandan_time) = b.total_gain) :
  nandan_earning b = 2000 := by
  sorry

#check nandan_earning_is_2000

end NUMINAMATH_CALUDE_nandan_earning_is_2000_l3602_360226


namespace NUMINAMATH_CALUDE_lattice_points_count_l3602_360217

/-- The number of lattice points on a line segment --/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (4,19) to (39,239) is 6 --/
theorem lattice_points_count :
  countLatticePoints 4 19 39 239 = 6 := by sorry

end NUMINAMATH_CALUDE_lattice_points_count_l3602_360217


namespace NUMINAMATH_CALUDE_binomial_12_choose_2_l3602_360246

theorem binomial_12_choose_2 : Nat.choose 12 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_2_l3602_360246


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3602_360224

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ + 4 = 5*x₁ + 6) ∧ 
  (x₂^2 + 2*x₂ + 4 = 5*x₂ + 6) → 
  x₁*x₂ + x₁ + x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3602_360224


namespace NUMINAMATH_CALUDE_power_multiplication_l3602_360283

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3602_360283


namespace NUMINAMATH_CALUDE_cans_per_row_is_twelve_l3602_360231

/-- The number of rows on one shelf -/
def rows_per_shelf : ℕ := 4

/-- The number of shelves in one closet -/
def shelves_per_closet : ℕ := 10

/-- The total number of cans Jack can store in one closet -/
def cans_per_closet : ℕ := 480

/-- The number of cans Jack can fit in one row -/
def cans_per_row : ℕ := cans_per_closet / (shelves_per_closet * rows_per_shelf)

theorem cans_per_row_is_twelve : cans_per_row = 12 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_row_is_twelve_l3602_360231


namespace NUMINAMATH_CALUDE_marys_story_characters_l3602_360268

theorem marys_story_characters (total : ℕ) (a c d e : ℕ) : 
  total = 60 →
  a = total / 2 →
  c = a / 2 →
  d + e = total - a - c →
  d = 2 * e →
  d = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_story_characters_l3602_360268


namespace NUMINAMATH_CALUDE_exam_pass_count_l3602_360252

theorem exam_pass_count (total_students : ℕ) (total_average : ℚ) 
  (pass_average : ℚ) (fail_average : ℚ) (weight_ratio : ℚ × ℚ) :
  total_students = 150 ∧ 
  total_average = 40 ∧ 
  pass_average = 45 ∧ 
  fail_average = 20 ∧ 
  weight_ratio = (3, 1) →
  ∃ (pass_count : ℕ), pass_count = 85 ∧ 
    (pass_count : ℚ) * weight_ratio.1 * pass_average + 
    (total_students - pass_count : ℚ) * weight_ratio.2 * fail_average = 
    total_average * (pass_count * weight_ratio.1 + (total_students - pass_count) * weight_ratio.2) :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l3602_360252


namespace NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_3_mod_29_l3602_360258

theorem largest_four_digit_negative_congruent_to_3_mod_29 :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 3 [ZMOD 29] → n ≤ -1012 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_negative_congruent_to_3_mod_29_l3602_360258


namespace NUMINAMATH_CALUDE_ivan_speed_ratio_l3602_360249

/-- Represents the speed of a person or group -/
structure Speed :=
  (value : ℝ)

/-- Represents time in hours -/
def Time : Type := ℝ

theorem ivan_speed_ratio (group_speed : Speed) (ivan_speed : Speed) : 
  -- Ivan left 15 minutes (0.25 hours) after the group started
  -- Ivan took 2.5 hours to catch up with the group after retrieving the flashlight
  -- Speeds of the group and Ivan (when not with the group) are constant
  (0.25 : ℝ) * group_speed.value + 2.5 * group_speed.value = 
    2.5 * ivan_speed.value + 2 * (0.25 * group_speed.value) →
  -- The ratio of Ivan's speed to the group's speed is 1.2
  ivan_speed.value / group_speed.value = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ivan_speed_ratio_l3602_360249


namespace NUMINAMATH_CALUDE_gcd_9011_4403_l3602_360259

theorem gcd_9011_4403 : Nat.gcd 9011 4403 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9011_4403_l3602_360259


namespace NUMINAMATH_CALUDE_positive_A_value_l3602_360241

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 4 = 65) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l3602_360241


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3602_360242

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3602_360242


namespace NUMINAMATH_CALUDE_sara_apples_l3602_360282

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples + ali_ratio * sara_apples →
  sara_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_apples_l3602_360282


namespace NUMINAMATH_CALUDE_intersection_M_N_l3602_360274

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3602_360274


namespace NUMINAMATH_CALUDE_arithmetic_progression_coprime_terms_l3602_360247

theorem arithmetic_progression_coprime_terms :
  ∃ (a r : ℕ), 
    (∀ i j, 0 ≤ i ∧ i < j ∧ j < 100 → 
      (a + i * r).gcd (a + j * r) = 1) ∧
    (∀ i, 0 ≤ i ∧ i < 99 → a + i * r < a + (i + 1) * r) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_coprime_terms_l3602_360247


namespace NUMINAMATH_CALUDE_square_difference_equality_l3602_360200

theorem square_difference_equality : (2 + 3)^2 - (2^2 + 3^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3602_360200


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3602_360219

theorem sum_of_cubes_of_roots (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 5 * x₁ - 2 = 0) → 
  (3 * x₂^2 - 5 * x₂ - 2 = 0) → 
  x₁^3 + x₂^3 = 215 / 27 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3602_360219


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3602_360285

-- Define the number of sides of the polygon
variable (n : ℕ)

-- Define the sum of interior angles
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Define the sum of exterior angles (always 360°)
def sum_exterior_angles : ℝ := 360

-- State the theorem
theorem polygon_sides_count :
  sum_interior_angles n = sum_exterior_angles + 720 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3602_360285
