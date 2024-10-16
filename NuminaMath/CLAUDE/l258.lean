import Mathlib

namespace NUMINAMATH_CALUDE_temperature_at_3pm_l258_25883

-- Define the temperature function
def T (t : ℝ) : ℝ := t^3 - 3*t + 60

-- State the theorem
theorem temperature_at_3pm : T 3 = 78 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_3pm_l258_25883


namespace NUMINAMATH_CALUDE_students_with_B_in_donovans_class_l258_25818

theorem students_with_B_in_donovans_class 
  (christopher_total : ℕ) 
  (christopher_B : ℕ) 
  (donovan_total : ℕ) 
  (h1 : christopher_total = 20) 
  (h2 : christopher_B = 12) 
  (h3 : donovan_total = 30) 
  (h4 : (christopher_B : ℚ) / christopher_total = (donovan_B : ℚ) / donovan_total) :
  donovan_B = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_B_in_donovans_class_l258_25818


namespace NUMINAMATH_CALUDE_quotient_invariance_problem_solution_l258_25877

theorem quotient_invariance (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / b = (c * a) / (c * b) :=
by sorry

theorem problem_solution : (0.75 : ℝ) / 25 = 7.5 / 250 := by
  have h1 : (0.75 : ℝ) / 25 = (10 * 0.75) / (10 * 25) := by
    apply quotient_invariance 0.75 25 10
    norm_num
    norm_num
  have h2 : (10 * 0.75 : ℝ) = 7.5 := by norm_num
  have h3 : (10 * 25 : ℝ) = 250 := by norm_num
  rw [h1, h2, h3]

end NUMINAMATH_CALUDE_quotient_invariance_problem_solution_l258_25877


namespace NUMINAMATH_CALUDE_perpendicular_lines_l258_25809

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (a : ℝ) :
  perpendicular a (1/a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l258_25809


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l258_25885

/-- The number of distinct arrangements of letters in "balloon" -/
def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- The word "balloon" has 7 letters -/
axiom balloon_length : balloon_arrangements = Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of distinct arrangements of letters in "balloon" is 1260 -/
theorem balloon_arrangements_count : balloon_arrangements = 1260 := by
  sorry


end NUMINAMATH_CALUDE_balloon_arrangements_count_l258_25885


namespace NUMINAMATH_CALUDE_solution_sets_equality_l258_25895

-- Define the parameter a
def a : ℝ := 1

-- Define the solution set of ax - 1 > 0
def solution_set_1 : Set ℝ := {x | x > 1}

-- Define the solution set of (ax-1)(x+2) ≥ 0
def solution_set_2 : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

-- State the theorem
theorem solution_sets_equality (h : solution_set_1 = {x | x > 1}) : 
  solution_set_2 = {x | x ≤ -2 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l258_25895


namespace NUMINAMATH_CALUDE_ladder_construction_possible_l258_25897

/-- Represents the ladder construction problem --/
def ladder_problem (total_wood rung_length rung_spacing side_support_length climbing_height : ℝ) : Prop :=
  let num_rungs : ℝ := climbing_height / rung_spacing + 1
  let wood_for_rungs : ℝ := num_rungs * rung_length
  let wood_for_supports : ℝ := 2 * side_support_length
  let total_wood_needed : ℝ := wood_for_rungs + wood_for_supports
  let leftover_wood : ℝ := total_wood - total_wood_needed
  total_wood_needed ≤ total_wood ∧ leftover_wood = 36.5

/-- Theorem stating that the ladder can be built with the given conditions --/
theorem ladder_construction_possible : 
  ladder_problem 300 1.5 0.5 56 50 := by
  sorry

#check ladder_construction_possible

end NUMINAMATH_CALUDE_ladder_construction_possible_l258_25897


namespace NUMINAMATH_CALUDE_plane_parallel_transitivity_l258_25863

-- Define the concept of planes
variable (Plane : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem plane_parallel_transitivity (α β γ : Plane) :
  (∃ γ, parallel γ α ∧ parallel γ β) → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_plane_parallel_transitivity_l258_25863


namespace NUMINAMATH_CALUDE_cakes_sold_l258_25875

theorem cakes_sold (total : ℕ) (left : ℕ) (sold : ℕ) 
  (h1 : total = 54)
  (h2 : left = 13)
  (h3 : sold = total - left) : sold = 41 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_l258_25875


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l258_25819

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 2) :
  1/x + 1/y + 1/z ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l258_25819


namespace NUMINAMATH_CALUDE_simplify_expression_l258_25823

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : -2 * a^3 / a = -2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l258_25823


namespace NUMINAMATH_CALUDE_unicorn_flower_bloom_l258_25808

theorem unicorn_flower_bloom :
  let num_unicorns : ℕ := 12
  let journey_length : ℕ := 15000  -- in meters
  let step_length : ℕ := 3  -- in meters
  let flowers_per_step : ℕ := 7
  
  (journey_length / step_length) * num_unicorns * flowers_per_step = 420000 :=
by
  sorry

end NUMINAMATH_CALUDE_unicorn_flower_bloom_l258_25808


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l258_25817

/-- Given two rectangles of equal area, where one has dimensions 8 by 45 and the other has width 24,
    prove that the length of the second rectangle is 15. -/
theorem equal_area_rectangles (area : ℝ) (length₁ width₁ width₂ : ℝ) 
  (h₁ : area = length₁ * width₁)
  (h₂ : length₁ = 8)
  (h₃ : width₁ = 45)
  (h₄ : width₂ = 24) :
  area / width₂ = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l258_25817


namespace NUMINAMATH_CALUDE_earnings_difference_l258_25845

/-- Calculates the difference in earnings between two sets of tasks with different pay rates -/
theorem earnings_difference (low_tasks : ℕ) (low_rate : ℚ) (high_tasks : ℕ) (high_rate : ℚ) :
  low_tasks = 400 →
  low_rate = 1/4 →
  high_tasks = 5 →
  high_rate = 2 →
  (low_tasks : ℚ) * low_rate - (high_tasks : ℚ) * high_rate = 90 :=
by sorry

end NUMINAMATH_CALUDE_earnings_difference_l258_25845


namespace NUMINAMATH_CALUDE_units_digit_150_factorial_is_zero_l258_25835

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_150_factorial_is_zero :
  unitsDigit (factorial 150) = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_150_factorial_is_zero_l258_25835


namespace NUMINAMATH_CALUDE_no_integer_solution_l258_25861

theorem no_integer_solution : ¬ ∃ (a b : ℤ), a^2 + b^2 = 10^100 + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l258_25861


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_l258_25802

/-- The locus of the midpoint of chord MN on a parabola -/
theorem parabola_midpoint_locus (p : ℝ) (x y : ℝ) :
  let parabola := fun (x y : ℝ) => y^2 - 2*p*x = 0
  let normal_intersection := fun (x y m : ℝ) => y - m*x + p*(m + m^3/2) = 0
  let conjugate_diameter := fun (y m : ℝ) => m*y - p = 0
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    normal_intersection x₂ y₂ m ∧
    conjugate_diameter y₁ m ∧
    x = (x₁ + x₂) / 2 ∧
    y = (y₁ + y₂) / 2
  →
  y^4 - (p*x)*y^2 + p^4/2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_l258_25802


namespace NUMINAMATH_CALUDE_central_angles_sum_l258_25872

theorem central_angles_sum (y : ℝ) : 
  (6 * y + 7 * y + 3 * y + y) * (π / 180) = 2 * π → y = 360 / 17 := by
sorry

end NUMINAMATH_CALUDE_central_angles_sum_l258_25872


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l258_25812

theorem geometric_sequence_first_term (a r : ℝ) : 
  a * r = 5 → a * r^3 = 45 → a = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l258_25812


namespace NUMINAMATH_CALUDE_value_to_decrease_l258_25870

theorem value_to_decrease (x : ℝ) (h1 : x > 0) (h2 : x = 7) : 
  ∃ v : ℝ, x - v = 21 * (1/x) ∧ v = 4 := by
sorry

end NUMINAMATH_CALUDE_value_to_decrease_l258_25870


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l258_25892

theorem vector_sum_magnitude : 
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (0, 1)
  ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l258_25892


namespace NUMINAMATH_CALUDE_suzanna_ride_l258_25839

/-- Calculates the distance traveled given a constant rate and time -/
def distanceTraveled (rate : ℚ) (time : ℚ) : ℚ :=
  rate * time

theorem suzanna_ride : 
  let rate : ℚ := 1.5 / 4  -- miles per minute
  let time : ℚ := 40       -- minutes
  distanceTraveled rate time = 15 := by
sorry

#eval (1.5 / 4) * 40  -- To verify the result

end NUMINAMATH_CALUDE_suzanna_ride_l258_25839


namespace NUMINAMATH_CALUDE_curve_symmetry_l258_25853

theorem curve_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a^2*x + (1-a^2)*y - 4 = 0 ↔ 
              y^2 + x^2 + a^2*y + (1-a^2)*x - 4 = 0) →
  a = Real.sqrt 2 / 2 ∨ a = -Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l258_25853


namespace NUMINAMATH_CALUDE_intersection_point_polar_radius_l258_25841

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 - y^2 = 4 ∧ y ≠ 0

-- Define the line l₃ in polar form
def l₃ (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) - Real.sqrt 2 = 0

-- Define the intersection point M
def M (x y : ℝ) : Prop := C x y ∧ x + y = Real.sqrt 2

-- Theorem statement
theorem intersection_point_polar_radius :
  ∀ x y : ℝ, M x y → x^2 + y^2 = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_polar_radius_l258_25841


namespace NUMINAMATH_CALUDE_min_trig_fraction_l258_25825

theorem min_trig_fraction :
  (∀ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 ≥ 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) ∧
  (∃ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + 1 = 5/6 * ((Real.sin x)^4 + (Real.cos x)^4 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_trig_fraction_l258_25825


namespace NUMINAMATH_CALUDE_hash_of_hash_of_hash_4_l258_25820

def hash (N : ℝ) : ℝ := 0.5 * N^2 + 1

theorem hash_of_hash_of_hash_4 : hash (hash (hash 4)) = 862.125 := by
  sorry

end NUMINAMATH_CALUDE_hash_of_hash_of_hash_4_l258_25820


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l258_25886

theorem mean_proportional_problem (x : ℝ) :
  (x * 9409 : ℝ).sqrt = 8665 → x = 7981 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l258_25886


namespace NUMINAMATH_CALUDE_no_real_solutions_l258_25871

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 2*y^2 - 6*x - 8*y + 21 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l258_25871


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l258_25821

/-- Given a quadratic equation 3x^2 - 4 = -2x, prove that when rearranged 
    into the standard form ax^2 + bx + c = 0, the coefficients are a = 3, b = 2, and c = -4 -/
theorem quadratic_coefficients : 
  ∀ (x : ℝ), 3 * x^2 - 4 = -2 * x → 
  ∃ (a b c : ℝ), a * x^2 + b * x + c = 0 ∧ a = 3 ∧ b = 2 ∧ c = -4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l258_25821


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l258_25846

/-- A quadratic function of the form y = -(x-2)² + h -/
def f (h : ℝ) (x : ℝ) : ℝ := -(x - 2)^2 + h

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship (h : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hA : f h (-1/2) = y₁)
  (hB : f h 1 = y₂)
  (hC : f h 2 = y₃) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l258_25846


namespace NUMINAMATH_CALUDE_sine_power_five_decomposition_l258_25860

theorem sine_power_five_decomposition (b₁ b₂ b₃ b₄ b₅ : ℝ) : 
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 101 / 256 := by
sorry

end NUMINAMATH_CALUDE_sine_power_five_decomposition_l258_25860


namespace NUMINAMATH_CALUDE_remaining_calories_l258_25880

def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

theorem remaining_calories : 
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calories_l258_25880


namespace NUMINAMATH_CALUDE_tangent_line_through_point_l258_25862

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the tangent line equation
def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := 3 * x₀^2 * (x - x₀) + x₀^3

-- State the theorem
theorem tangent_line_through_point :
  ∃ (x₀ : ℝ), (tangent_line x₀ 1 = 1) ∧
  ((tangent_line x₀ x = 3*x - 2) ∨ (tangent_line x₀ x = 3/4*x + 1/4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_through_point_l258_25862


namespace NUMINAMATH_CALUDE_tangent_squares_roots_l258_25856

theorem tangent_squares_roots : ∃ (a b c : ℝ),
  a + b + c = 33 ∧
  a * b * c = 33 ∧
  a * b + b * c + c * a = 27 ∧
  ∀ (x : ℝ), x^3 - 33*x^2 + 27*x - 33 = 0 ↔ (x = a ∨ x = b ∨ x = c) := by
  sorry

end NUMINAMATH_CALUDE_tangent_squares_roots_l258_25856


namespace NUMINAMATH_CALUDE_correct_payments_l258_25804

/-- Represents the payment information for the gardeners' plot plowing problem. -/
structure PlowingPayment where
  totalPayment : ℕ
  rectangularPlotArea : ℕ
  rectangularPlotSide : ℕ
  squarePlot1Side : ℕ
  squarePlot2Side : ℕ

/-- Calculates the payments for each gardener based on the given information. -/
def calculatePayments (info : PlowingPayment) : (ℕ × ℕ × ℕ) :=
  let rectangularPlotWidth := info.rectangularPlotArea / info.rectangularPlotSide
  let squarePlot1Area := info.squarePlot1Side * info.squarePlot1Side
  let squarePlot2Area := info.squarePlot2Side * info.squarePlot2Side
  let totalArea := info.rectangularPlotArea + squarePlot1Area + squarePlot2Area
  let pricePerArea := info.totalPayment / totalArea
  let payment1 := info.rectangularPlotArea * pricePerArea
  let payment2 := squarePlot1Area * pricePerArea
  let payment3 := squarePlot2Area * pricePerArea
  (payment1, payment2, payment3)

/-- Theorem stating that the calculated payments match the expected values. -/
theorem correct_payments (info : PlowingPayment) 
  (h1 : info.totalPayment = 570)
  (h2 : info.rectangularPlotArea = 600)
  (h3 : info.rectangularPlotSide = 20)
  (h4 : info.squarePlot1Side = info.rectangularPlotSide)
  (h5 : info.squarePlot2Side = info.rectangularPlotArea / info.rectangularPlotSide) :
  calculatePayments info = (180, 120, 270) := by
  sorry

end NUMINAMATH_CALUDE_correct_payments_l258_25804


namespace NUMINAMATH_CALUDE_birthday_money_calculation_l258_25866

def money_spent : ℕ := 34
def money_left : ℕ := 33

theorem birthday_money_calculation :
  money_spent + money_left = 67 := by sorry

end NUMINAMATH_CALUDE_birthday_money_calculation_l258_25866


namespace NUMINAMATH_CALUDE_water_cooler_capacity_l258_25852

/-- Represents the capacity of the water cooler in ounces -/
def cooler_capacity : ℕ := 126

/-- Number of linemen on the team -/
def num_linemen : ℕ := 12

/-- Number of skill position players on the team -/
def num_skill_players : ℕ := 10

/-- Amount of water each lineman drinks in ounces -/
def lineman_water : ℕ := 8

/-- Amount of water each skill position player drinks in ounces -/
def skill_player_water : ℕ := 6

/-- Number of skill position players who can drink before refill -/
def skill_players_before_refill : ℕ := 5

theorem water_cooler_capacity : 
  cooler_capacity = num_linemen * lineman_water + skill_players_before_refill * skill_player_water :=
by sorry

end NUMINAMATH_CALUDE_water_cooler_capacity_l258_25852


namespace NUMINAMATH_CALUDE_temperature_height_relationship_l258_25858

/-- The temperature-height relationship function -/
def t (h : ℝ) : ℝ := 20 - 6 * h

/-- The set of given data points -/
def data_points : List (ℝ × ℝ) := [(0, 20), (1, 14), (2, 8), (3, 2), (4, -4)]

/-- Theorem stating that the function t accurately describes the temperature-height relationship -/
theorem temperature_height_relationship :
  ∀ (point : ℝ × ℝ), point ∈ data_points → t point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_height_relationship_l258_25858


namespace NUMINAMATH_CALUDE_complement_of_union_theorem_l258_25829

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | x ≤ 0}

-- State the theorem
theorem complement_of_union_theorem :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_theorem_l258_25829


namespace NUMINAMATH_CALUDE_marcus_earnings_l258_25815

/-- Represents Marcus's work and earnings over two weeks -/
structure MarcusWork where
  hourly_wage : ℝ
  hours_week1 : ℝ
  hours_week2 : ℝ
  earnings_difference : ℝ

/-- Calculates the total earnings for two weeks given Marcus's work data -/
def total_earnings (w : MarcusWork) : ℝ :=
  w.hourly_wage * (w.hours_week1 + w.hours_week2)

theorem marcus_earnings :
  ∀ w : MarcusWork,
  w.hours_week1 = 12 ∧
  w.hours_week2 = 18 ∧
  w.earnings_difference = 36 ∧
  w.hourly_wage * (w.hours_week2 - w.hours_week1) = w.earnings_difference →
  total_earnings w = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_marcus_earnings_l258_25815


namespace NUMINAMATH_CALUDE_complex_square_l258_25849

theorem complex_square (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l258_25849


namespace NUMINAMATH_CALUDE_simplify_expression_l258_25887

theorem simplify_expression (a : ℝ) : 3*a - 5*a + a = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l258_25887


namespace NUMINAMATH_CALUDE_sticker_distribution_l258_25867

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of stickers to be distributed -/
def num_stickers : ℕ := 11

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 1365 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l258_25867


namespace NUMINAMATH_CALUDE_solution_in_interval_implies_a_range_l258_25864

theorem solution_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc 1 5, x^2 + a*x - 2 = 0) →
  a ∈ Set.Icc (-23/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_interval_implies_a_range_l258_25864


namespace NUMINAMATH_CALUDE_ratio_problem_l258_25851

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + b) / (b + c) = 4/15 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l258_25851


namespace NUMINAMATH_CALUDE_locus_is_circle_l258_25811

/-- The locus of points satisfying the given condition is a circle -/
theorem locus_is_circle (k : ℝ) (h : k > 0) :
  ∀ (x y : ℝ), (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x / a + y / b = 1 ∧ 1 / a^2 + 1 / b^2 = 1 / k^2) 
  ↔ x^2 + y^2 = k^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l258_25811


namespace NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l258_25843

/-- Two concentric circles with center O -/
structure ConcentricCircles where
  O : Point
  r₁ : ℝ  -- radius of smaller circle
  r₂ : ℝ  -- radius of larger circle
  h : 0 < r₁ ∧ r₁ < r₂

/-- The length of an arc on a circle -/
def arcLength (r : ℝ) (θ : ℝ) : ℝ := r * θ

theorem area_ratio_of_concentric_circles (C : ConcentricCircles) :
  arcLength C.r₁ (π/3) = arcLength C.r₂ (π/4) →
  (C.r₁^2 / C.r₂^2 : ℝ) = 9/16 := by
  sorry

#check area_ratio_of_concentric_circles

end NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l258_25843


namespace NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l258_25855

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem only_eleven_not_sum_of_two_primes :
  is_sum_of_two_primes 5 ∧
  is_sum_of_two_primes 7 ∧
  is_sum_of_two_primes 9 ∧
  ¬(is_sum_of_two_primes 11) ∧
  is_sum_of_two_primes 13 :=
by sorry

end NUMINAMATH_CALUDE_only_eleven_not_sum_of_two_primes_l258_25855


namespace NUMINAMATH_CALUDE_samia_walk_distance_l258_25813

def average_bike_speed : ℝ := 20
def bike_distance : ℝ := 2
def walk_speed : ℝ := 4
def total_time_minutes : ℝ := 78

theorem samia_walk_distance :
  let total_time_hours : ℝ := total_time_minutes / 60
  let bike_time : ℝ := bike_distance / average_bike_speed
  let walk_time : ℝ := total_time_hours - bike_time
  let walk_distance : ℝ := walk_time * walk_speed
  walk_distance = 4.8 := by sorry

end NUMINAMATH_CALUDE_samia_walk_distance_l258_25813


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l258_25830

/-- Given a sum of money that doubles itself in 5 years at simple interest,
    prove that the rate percent per annum is 20%. -/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 5) / 100 = 2 * P ∧ R = 20 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l258_25830


namespace NUMINAMATH_CALUDE_lawrence_county_summer_break_l258_25806

/-- The number of kids who stay home during summer break in Lawrence county -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 907,611 kids stay home during summer break in Lawrence county -/
theorem lawrence_county_summer_break :
  kids_stay_home 1363293 455682 = 907611 := by
sorry

end NUMINAMATH_CALUDE_lawrence_county_summer_break_l258_25806


namespace NUMINAMATH_CALUDE_cubic_local_minimum_l258_25869

/-- A function f: ℝ → ℝ has a local minimum if there exists a point c such that
    f(c) ≤ f(x) for all x in some open interval containing c. -/
def has_local_minimum (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∃ ε > 0, ∀ x, |x - c| < ε → f c ≤ f x

/-- The cubic function f(x) = x^3 - 3bx + 3b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

theorem cubic_local_minimum (b : ℝ) :
  has_local_minimum (f b) ↔ b > 0 := by sorry

end NUMINAMATH_CALUDE_cubic_local_minimum_l258_25869


namespace NUMINAMATH_CALUDE_insufficient_album_capacity_l258_25894

/-- Represents the capacity and quantity of each album type -/
structure AlbumType where
  capacity : ℕ
  quantity : ℕ

/-- Proves that the total capacity of all available albums is less than the total number of pictures -/
theorem insufficient_album_capacity 
  (type_a : AlbumType)
  (type_b : AlbumType)
  (type_c : AlbumType)
  (total_pictures : ℕ)
  (h1 : type_a.capacity = 12)
  (h2 : type_a.quantity = 6)
  (h3 : type_b.capacity = 18)
  (h4 : type_b.quantity = 4)
  (h5 : type_c.capacity = 24)
  (h6 : type_c.quantity = 3)
  (h7 : total_pictures = 480) :
  type_a.capacity * type_a.quantity + 
  type_b.capacity * type_b.quantity + 
  type_c.capacity * type_c.quantity < total_pictures :=
by sorry

end NUMINAMATH_CALUDE_insufficient_album_capacity_l258_25894


namespace NUMINAMATH_CALUDE_hyperbola_equation_l258_25889

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x^2 + y^2 = 4 ∧ x = -2) →
  (b / a = Real.sqrt 3) →
  (∀ (x y : ℝ), x^2 - y^2 / 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l258_25889


namespace NUMINAMATH_CALUDE_division_problem_l258_25800

theorem division_problem (L S Q : ℕ) : 
  L - S = 1355 → 
  L = 1608 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l258_25800


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l258_25833

def number_of_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).map (· + 1) |>.prod

def is_smallest_with_2020_divisors (n : ℕ) : Prop :=
  number_of_divisors n = 2020 ∧
  ∀ m < n, number_of_divisors m ≠ 2020

theorem smallest_number_with_2020_divisors :
  is_smallest_with_2020_divisors (2^100 * 3^4 * 5 * 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l258_25833


namespace NUMINAMATH_CALUDE_det_matrix_l258_25898

def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y + 2, 2*y, 2*y;
     2*y, y + 2, 2*y;
     2*y, 2*y, y + 2]

theorem det_matrix (y : ℝ) :
  Matrix.det (matrix y) = 5*y^3 - 10*y^2 + 12*y + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_matrix_l258_25898


namespace NUMINAMATH_CALUDE_vector_subtraction_l258_25844

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (-7, 0, 1)
def b : ℝ × ℝ × ℝ := (6, 2, -1)

-- State the theorem
theorem vector_subtraction :
  (a.1 - 5 * b.1, a.2.1 - 5 * b.2.1, a.2.2 - 5 * b.2.2) = (-37, -10, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l258_25844


namespace NUMINAMATH_CALUDE_find_z_l258_25834

theorem find_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (z ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : z = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_z_l258_25834


namespace NUMINAMATH_CALUDE_volvox_face_difference_l258_25874

/-- A spherical polyhedron where each face has 5, 6, or 7 sides, and exactly three faces meet at each vertex. -/
structure VolvoxPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  f₅ : ℕ  -- number of pentagonal faces
  f₆ : ℕ  -- number of hexagonal faces
  f₇ : ℕ  -- number of heptagonal faces
  euler : V - E + F = 2
  face_sum : F = f₅ + f₆ + f₇
  edge_sum : 2 * E = 5 * f₅ + 6 * f₆ + 7 * f₇
  vertex_sum : 3 * V = 5 * f₅ + 6 * f₆ + 7 * f₇

/-- The number of pentagonal faces is always 12 more than the number of heptagonal faces. -/
theorem volvox_face_difference (p : VolvoxPolyhedron) : p.f₅ = p.f₇ + 12 := by
  sorry

end NUMINAMATH_CALUDE_volvox_face_difference_l258_25874


namespace NUMINAMATH_CALUDE_evaluate_expression_l258_25891

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l258_25891


namespace NUMINAMATH_CALUDE_gcd_problem_l258_25890

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1187 * k) :
  Nat.gcd (Int.natAbs (2 * b^2 + 31 * b + 67)) (Int.natAbs (b + 15)) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l258_25890


namespace NUMINAMATH_CALUDE_complex_root_magnitude_one_iff_divisible_by_six_l258_25847

theorem complex_root_magnitude_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_one_iff_divisible_by_six_l258_25847


namespace NUMINAMATH_CALUDE_certain_number_proof_l258_25840

theorem certain_number_proof (x : ℝ) : (1.68 * x) / 6 = 354.2 ↔ x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l258_25840


namespace NUMINAMATH_CALUDE_triangle_angle_C_l258_25899

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 6) →
  (b = 2) →
  (B = π / 4) → -- 45° in radians
  (Real.tan A * Real.tan C > 1) →
  -- Conclusion
  C = 5 * π / 12 -- 75° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l258_25899


namespace NUMINAMATH_CALUDE_frank_allowance_proof_l258_25824

def frank_allowance (initial_amount spent_amount final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - spent_amount)

theorem frank_allowance_proof :
  frank_allowance 11 3 22 = 14 := by
  sorry

end NUMINAMATH_CALUDE_frank_allowance_proof_l258_25824


namespace NUMINAMATH_CALUDE_oscar_elmer_difference_l258_25884

-- Define the given constants
def elmer_strides_per_gap : ℕ := 44
def oscar_leaps_per_gap : ℕ := 12
def total_poles : ℕ := 41
def total_distance : ℕ := 5280

-- Define the theorem
theorem oscar_elmer_difference : 
  let gaps := total_poles - 1
  let elmer_total_strides := elmer_strides_per_gap * gaps
  let oscar_total_leaps := oscar_leaps_per_gap * gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_oscar_elmer_difference_l258_25884


namespace NUMINAMATH_CALUDE_prob_ace_king_is_4_663_l258_25831

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ := 52)
  (num_aces : ℕ := 4)
  (num_kings : ℕ := 4)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * d.num_kings / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace first and a King second from a standard deck. -/
theorem prob_ace_king_is_4_663 (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_is_4_663_l258_25831


namespace NUMINAMATH_CALUDE_annual_music_cost_l258_25827

/-- Calculates the annual cost of music for John given his monthly music consumption, average song length, and price per song. -/
theorem annual_music_cost 
  (monthly_hours : ℕ) 
  (song_length_minutes : ℕ) 
  (price_per_song : ℚ) : 
  monthly_hours = 20 → 
  song_length_minutes = 3 → 
  price_per_song = 1/2 → 
  (monthly_hours * 60 / song_length_minutes) * price_per_song * 12 = 2400 := by
sorry

end NUMINAMATH_CALUDE_annual_music_cost_l258_25827


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l258_25850

theorem restaurant_menu_fraction (total_vegan : ℕ) (total_menu : ℕ) (vegan_with_nuts : ℕ) :
  total_vegan = 6 →
  total_vegan = total_menu / 3 →
  vegan_with_nuts = 1 →
  (total_vegan - vegan_with_nuts : ℚ) / total_menu = 5 / 18 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l258_25850


namespace NUMINAMATH_CALUDE_bucket_water_volume_l258_25832

theorem bucket_water_volume (initial_volume : ℝ) (additional_volume : ℝ) : 
  initial_volume = 2 → 
  additional_volume = 460 → 
  (initial_volume * 1000 + additional_volume : ℝ) = 2460 := by
sorry

end NUMINAMATH_CALUDE_bucket_water_volume_l258_25832


namespace NUMINAMATH_CALUDE_red_marbles_in_bag_l258_25854

theorem red_marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + 3)
  (h2 : (red_marbles : ℝ) / total_marbles * ((red_marbles - 1) : ℝ) / (total_marbles - 1) = 0.1) :
  red_marbles = 2 := by
sorry

end NUMINAMATH_CALUDE_red_marbles_in_bag_l258_25854


namespace NUMINAMATH_CALUDE_two_number_problem_l258_25801

theorem two_number_problem (x y : ℝ) 
  (h1 : 0.35 * x = 0.50 * x - 24)
  (h2 : 0.30 * y = 0.55 * x - 36) : 
  x = 160 ∧ y = 520/3 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l258_25801


namespace NUMINAMATH_CALUDE_range_of_m_l258_25859

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2/x + 1/y = 1) (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) :
  m ∈ Set.Ioo (-4 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l258_25859


namespace NUMINAMATH_CALUDE_circle_area_l258_25888

theorem circle_area (r : ℝ) (h : r = 11) : π * r^2 = π * 11^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l258_25888


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l258_25893

/-- Calculates the savings when using a discount clinic compared to a normal doctor visit -/
theorem discount_clinic_savings
  (normal_cost : ℝ)
  (discount_percentage : ℝ)
  (discount_visits : ℕ)
  (h1 : normal_cost = 200)
  (h2 : discount_percentage = 0.7)
  (h3 : discount_visits = 2) :
  normal_cost - discount_visits * (normal_cost * (1 - discount_percentage)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l258_25893


namespace NUMINAMATH_CALUDE_mary_tim_income_difference_l258_25805

theorem mary_tim_income_difference (juan tim mary : ℝ) 
  (h1 : tim = 0.5 * juan)
  (h2 : mary = 0.8 * juan) :
  (mary - tim) / tim * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_mary_tim_income_difference_l258_25805


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l258_25848

/-- The slope of the tangent line to y = x^3 - 4x at (1, -1) is -1 -/
theorem tangent_slope_at_point : 
  let f (x : ℝ) := x^3 - 4*x
  let x₀ : ℝ := 1
  let y₀ : ℝ := -1
  (deriv f) x₀ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l258_25848


namespace NUMINAMATH_CALUDE_probability_standard_deck_l258_25822

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (diamond_cards : Nat)
  (spade_cards : Nat)

/-- A standard deck has 52 cards, 13 diamonds, and 13 spades -/
def standard_deck : Deck :=
  ⟨52, 13, 13⟩

/-- Calculates the probability of drawing a diamond first, then two spades -/
def probability_diamond_then_two_spades (d : Deck) : Rat :=
  (d.diamond_cards : Rat) / d.total_cards *
  (d.spade_cards : Rat) / (d.total_cards - 1) *
  ((d.spade_cards - 1) : Rat) / (d.total_cards - 2)

/-- Theorem: The probability of drawing a diamond first, then two spades from a standard deck is 13/850 -/
theorem probability_standard_deck :
  probability_diamond_then_two_spades standard_deck = 13 / 850 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_deck_l258_25822


namespace NUMINAMATH_CALUDE_marks_ratio_l258_25838

theorem marks_ratio (P S W : ℚ) 
  (h1 : P / S = 4 / 5) 
  (h2 : S / W = 5 / 2) : 
  P / W = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_marks_ratio_l258_25838


namespace NUMINAMATH_CALUDE_exists_geometric_subsequence_l258_25865

/-- A strictly increasing sequence of positive integers in arithmetic progression -/
def ArithmeticSequence : ℕ → ℕ := λ n => sorry

/-- The first term of the arithmetic progression -/
def a : ℕ := sorry

/-- The common difference of the arithmetic progression -/
def d : ℕ := sorry

/-- Condition: ArithmeticSequence is strictly increasing -/
axiom strictly_increasing : ∀ n : ℕ, ArithmeticSequence n < ArithmeticSequence (n + 1)

/-- Condition: ArithmeticSequence is an arithmetic progression -/
axiom is_arithmetic_progression : ∀ n : ℕ, ArithmeticSequence n = a + (n - 1) * d

/-- The existence of an infinite geometric sub-sequence -/
theorem exists_geometric_subsequence :
  ∃ (SubSeq : ℕ → ℕ) (r : ℚ),
    (∀ n : ℕ, ∃ k : ℕ, ArithmeticSequence k = SubSeq n) ∧
    (∀ n : ℕ, SubSeq (n + 1) = r * SubSeq n) :=
sorry

end NUMINAMATH_CALUDE_exists_geometric_subsequence_l258_25865


namespace NUMINAMATH_CALUDE_paulines_garden_capacity_l258_25882

/-- Represents Pauline's garden -/
structure Garden where
  tomato_kinds : ℕ
  tomatoes_per_kind : ℕ
  cucumber_kinds : ℕ
  cucumbers_per_kind : ℕ
  potatoes : ℕ
  rows : ℕ
  spaces_per_row : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - 
  (g.tomato_kinds * g.tomatoes_per_kind + 
   g.cucumber_kinds * g.cucumbers_per_kind + 
   g.potatoes)

/-- Theorem stating that in Pauline's specific garden, 85 more vegetables can be planted -/
theorem paulines_garden_capacity :
  ∃ (g : Garden), 
    g.tomato_kinds = 3 ∧ 
    g.tomatoes_per_kind = 5 ∧ 
    g.cucumber_kinds = 5 ∧ 
    g.cucumbers_per_kind = 4 ∧ 
    g.potatoes = 30 ∧ 
    g.rows = 10 ∧ 
    g.spaces_per_row = 15 ∧ 
    additional_vegetables g = 85 := by
  sorry

end NUMINAMATH_CALUDE_paulines_garden_capacity_l258_25882


namespace NUMINAMATH_CALUDE_parallelogram_area_l258_25873

/-- The area of a parallelogram with diagonals intersecting at a 60° angle
    and two sides of lengths 6 and 8 is equal to 14√3. -/
theorem parallelogram_area (a b : ℝ) : 
  (a^2 + b^2 - a*b = 36) →  -- From the side of length 6
  (a^2 + b^2 + a*b = 64) →  -- From the side of length 8
  2 * a * b * (Real.sqrt 3 / 2) = 14 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l258_25873


namespace NUMINAMATH_CALUDE_scale_division_l258_25814

/-- Converts feet and inches to total inches -/
def feetInchesToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet and inches -/
def inchesToFeetInches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division (totalFeet : ℕ) (totalInches : ℕ) (parts : ℕ) 
    (h1 : totalFeet = 7) 
    (h2 : totalInches = 6) 
    (h3 : parts = 5) : 
  inchesToFeetInches (feetInchesToInches totalFeet totalInches / parts) = (1, 6) := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l258_25814


namespace NUMINAMATH_CALUDE_simple_interest_proof_l258_25876

/-- Given a principal amount where the compound interest for 2 years at 5% per annum is 41,
    prove that the simple interest for the same principal, rate, and time is 40. -/
theorem simple_interest_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P = 41 → P * 0.05 * 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_proof_l258_25876


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_ten_l258_25816

/-- Represents the number of classes. -/
def num_classes : ℕ := 3

/-- Represents the total number of spots to be allocated. -/
def total_spots : ℕ := 6

/-- Represents the minimum number of spots each class must receive. -/
def min_spots_per_class : ℕ := 1

/-- A function that calculates the number of ways to allocate spots among classes. -/
def allocation_schemes (n c m : ℕ) : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 10. -/
theorem allocation_schemes_eq_ten : 
  allocation_schemes total_spots num_classes min_spots_per_class = 10 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_ten_l258_25816


namespace NUMINAMATH_CALUDE_equation_solution_exists_l258_25810

theorem equation_solution_exists : ∃ (d e f g : ℕ+), 
  (3 : ℝ) * ((7 : ℝ)^(1/3) + (6 : ℝ)^(1/3))^(1/2) = d^(1/3) - e^(1/3) + f^(1/3) + g^(1/3) ∧
  d + e + f + g = 96 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l258_25810


namespace NUMINAMATH_CALUDE_color_change_probability_l258_25803

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationDuration (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℕ :=
  3 * observationInterval  -- 3 color changes per cycle

/-- Theorem: The probability of observing a color change is 12/85 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 35)
  (observationInterval : ℕ)
  (h4 : observationInterval = 4) :
  (changeObservationDuration cycle observationInterval : ℚ) / 
  (cycleDuration cycle : ℚ) = 12 / 85 := by
  sorry

end NUMINAMATH_CALUDE_color_change_probability_l258_25803


namespace NUMINAMATH_CALUDE_cube_root_unity_product_l258_25836

theorem cube_root_unity_product (w : ℂ) : w^3 = 1 → (1 - w + w^2) * (1 + w - w^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_product_l258_25836


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l258_25896

-- Define the set of x values that make the expression meaningful
def meaningful_x : Set ℝ := {x | x ≥ -5 ∧ x ≠ 0}

-- Theorem statement
theorem meaningful_expression_range : 
  {x : ℝ | (∃ y : ℝ, y = Real.sqrt (x + 5) / x) ∧ x ≠ 0} = meaningful_x := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l258_25896


namespace NUMINAMATH_CALUDE_highest_percentage_increase_survey_d_l258_25828

structure Survey where
  customers : ℕ
  responses : ℕ

def response_rate (s : Survey) : ℚ :=
  s.responses / s.customers

def percentage_change (a b : ℚ) : ℚ :=
  (b - a) / a * 100

theorem highest_percentage_increase_survey_d (survey_a survey_b survey_c survey_d : Survey)
  (ha : survey_a = { customers := 100, responses := 15 })
  (hb : survey_b = { customers := 120, responses := 27 })
  (hc : survey_c = { customers := 140, responses := 39 })
  (hd : survey_d = { customers := 160, responses := 56 }) :
  let change_ab := percentage_change (response_rate survey_a) (response_rate survey_b)
  let change_ac := percentage_change (response_rate survey_a) (response_rate survey_c)
  let change_ad := percentage_change (response_rate survey_a) (response_rate survey_d)
  change_ad > change_ab ∧ change_ad > change_ac := by
  sorry

end NUMINAMATH_CALUDE_highest_percentage_increase_survey_d_l258_25828


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l258_25807

/-- A geometric sequence {a_n} satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 5 * a 8 = 6
  condition2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l258_25807


namespace NUMINAMATH_CALUDE_unique_number_in_intersection_l258_25857

theorem unique_number_in_intersection : ∃! x : ℝ, 3 < x ∧ x < 8 ∧ 6 < x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_in_intersection_l258_25857


namespace NUMINAMATH_CALUDE_remaining_crops_l258_25881

/-- Calculates the total number of remaining crops for a farmer after pest damage --/
theorem remaining_crops (corn_per_row potato_per_row wheat_per_row : ℕ)
  (corn_destroyed potato_destroyed wheat_destroyed : ℚ)
  (corn_rows potato_rows wheat_rows : ℕ)
  (h_corn : corn_per_row = 12)
  (h_potato : potato_per_row = 40)
  (h_wheat : wheat_per_row = 60)
  (h_corn_dest : corn_destroyed = 30 / 100)
  (h_potato_dest : potato_destroyed = 40 / 100)
  (h_wheat_dest : wheat_destroyed = 25 / 100)
  (h_corn_rows : corn_rows = 25)
  (h_potato_rows : potato_rows = 15)
  (h_wheat_rows : wheat_rows = 20) :
  (corn_rows * corn_per_row * (1 - corn_destroyed)).floor +
  (potato_rows * potato_per_row * (1 - potato_destroyed)).floor +
  (wheat_rows * wheat_per_row * (1 - wheat_destroyed)).floor = 1470 := by
  sorry

end NUMINAMATH_CALUDE_remaining_crops_l258_25881


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l258_25879

/-- Proves the quantity of sugar sold at 18% profit given the conditions -/
theorem sugar_profit_problem (total_sugar : ℝ) (profit_rate_1 profit_rate_2 overall_profit : ℝ) 
  (h1 : total_sugar = 1000)
  (h2 : profit_rate_1 = 0.08)
  (h3 : profit_rate_2 = 0.18)
  (h4 : overall_profit = 0.14)
  (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
    profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :
  ∃ y : ℝ, y = 600 ∧ y = total_sugar - 
    Classical.choose (h5 : ∃ x : ℝ, x ≥ 0 ∧ x ≤ total_sugar ∧ 
      profit_rate_1 * x + profit_rate_2 * (total_sugar - x) = overall_profit * total_sugar) :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l258_25879


namespace NUMINAMATH_CALUDE_complement_union_equal_l258_25868

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

theorem complement_union_equal : (U \ A) ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equal_l258_25868


namespace NUMINAMATH_CALUDE_line_through_point_with_specific_intercept_ratio_l258_25878

/-- A line passing through the point (-5,2) with an x-intercept twice its y-intercept 
    has the equation 2x + 5y = 0 or x + 2y + 1 = 0 -/
theorem line_through_point_with_specific_intercept_ratio :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∨ b ≠ 0) →
    (a * (-5) + b * 2 + c = 0) →
    (∃ t : ℝ, a * (-2*t) + c = 0 ∧ b * t + c = 0) →
    ((∃ k : ℝ, a = 2*k ∧ b = 5*k ∧ c = 0) ∨ (∃ k : ℝ, a = k ∧ b = 2*k ∧ c = -k)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_specific_intercept_ratio_l258_25878


namespace NUMINAMATH_CALUDE_work_completion_time_l258_25837

/-- The number of days it takes for person B to complete the work alone -/
def B_days : ℝ := 60

/-- The fraction of work completed by A and B together in 6 days -/
def work_completed : ℝ := 0.25

/-- The number of days A and B work together -/
def days_together : ℝ := 6

/-- The number of days it takes for person A to complete the work alone -/
def A_days : ℝ := 40

theorem work_completion_time :
  (1 / A_days + 1 / B_days) * days_together = work_completed :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l258_25837


namespace NUMINAMATH_CALUDE_pond_freezes_on_seventh_day_l258_25826

/-- Represents a rectangular pond with given dimensions and freezing properties -/
structure Pond where
  length : ℝ
  width : ℝ
  daily_freeze_distance : ℝ
  first_day_freeze_percent : ℝ
  second_day_freeze_percent : ℝ

/-- Calculates the day when the pond is completely frozen -/
def freezing_day (p : Pond) : ℕ :=
  sorry

/-- Theorem stating that the pond will be completely frozen on the 7th day -/
theorem pond_freezes_on_seventh_day (p : Pond) 
  (h1 : p.length * p.width = 5000)
  (h2 : p.length + p.width = 70.5)
  (h3 : p.daily_freeze_distance = 10)
  (h4 : p.first_day_freeze_percent = 0.202)
  (h5 : p.second_day_freeze_percent = 0.186) : 
  freezing_day p = 7 :=
sorry

end NUMINAMATH_CALUDE_pond_freezes_on_seventh_day_l258_25826


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l258_25842

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (a - x)}
def B : Set ℝ := {x | x^2 - x - 6 = 0}

-- Part 1
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {-2} := by sorry

-- Part 2
theorem complement_A_intersect_B_empty_iff_a_gt_three (a : ℝ) :
  (Set.univ \ A a) ∩ B = ∅ ↔ a > 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_complement_A_intersect_B_empty_iff_a_gt_three_l258_25842
