import Mathlib

namespace NUMINAMATH_CALUDE_pharmacist_weights_exist_l11_1161

theorem pharmacist_weights_exist : ∃ (a b c : ℝ), 
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b = 100 ∧
  a + c = 101 ∧
  b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_pharmacist_weights_exist_l11_1161


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l11_1158

def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

theorem one_positive_real_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l11_1158


namespace NUMINAMATH_CALUDE_paint_on_third_day_l11_1117

/-- The amount of paint available on the third day of a room refresh project -/
theorem paint_on_third_day (initial_paint : ℝ) (added_paint : ℝ) : 
  initial_paint = 80 → 
  added_paint = 20 → 
  (initial_paint / 2 + added_paint) / 2 = 30 := by
sorry

end NUMINAMATH_CALUDE_paint_on_third_day_l11_1117


namespace NUMINAMATH_CALUDE_faye_coloring_books_l11_1139

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := sorry

/-- The initial number of coloring books Faye had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := 48

/-- The final number of coloring books Faye has -/
def final_books : ℕ := 79

theorem faye_coloring_books : 
  initial_books - books_given_away + books_bought = final_books ∧ 
  books_given_away = 3 := by sorry

end NUMINAMATH_CALUDE_faye_coloring_books_l11_1139


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l11_1125

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l11_1125


namespace NUMINAMATH_CALUDE_cos_equality_proof_l11_1147

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 → (Real.cos (n * π / 180) = Real.cos (123 * π / 180) ↔ n = 123 ∨ n = 237) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l11_1147


namespace NUMINAMATH_CALUDE_f_properties_l11_1177

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Theorem statement
theorem f_properties : (∀ x : ℝ, f (-x) = -f x) ∧
                       (∀ x y : ℝ, x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l11_1177


namespace NUMINAMATH_CALUDE_two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l11_1124

-- Define the conversion factor
def liter_to_milliliter : ℚ := 1000

-- Theorem for the first conversion
theorem two_liters_to_milliliters :
  2 * liter_to_milliliter = 2000 := by sorry

-- Theorem for the second conversion
theorem nine_thousand_milliliters_to_liters :
  9000 / liter_to_milliliter = 9 := by sorry

-- Theorem for the third conversion
theorem eight_liters_to_milliliters :
  8 * liter_to_milliliter = 8000 := by sorry

end NUMINAMATH_CALUDE_two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l11_1124


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l11_1131

theorem oak_grove_library_books (total_books : ℕ) (public_library_books : ℕ) 
  (h1 : total_books = 7092) (h2 : public_library_books = 1986) :
  total_books - public_library_books = 5106 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l11_1131


namespace NUMINAMATH_CALUDE_f_of_g_of_negative_three_l11_1165

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1) ^ 2

-- State the theorem
theorem f_of_g_of_negative_three : f (g (-3)) = 67 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_negative_three_l11_1165


namespace NUMINAMATH_CALUDE_acute_angle_range_l11_1198

theorem acute_angle_range (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  (∃ x : Real, 3 * x^2 * Real.sin α - 4 * x * Real.cos α + 2 = 0) →
  0 < α ∧ α ≤ Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l11_1198


namespace NUMINAMATH_CALUDE_numerator_proof_l11_1115

theorem numerator_proof (x y : ℝ) (h1 : x / y = 7 / 3) :
  ∃ (k N : ℝ), x = 7 * k ∧ y = 3 * k ∧ N / (x - y) = 2.5 ∧ N = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_numerator_proof_l11_1115


namespace NUMINAMATH_CALUDE_license_plate_count_l11_1163

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of even digits -/
def even_digit_count : ℕ := 5

/-- The number of odd digits -/
def odd_digit_count : ℕ := 5

/-- The number of letters in the license plate -/
def letter_count : ℕ := 3

/-- The number of digits in the license plate -/
def plate_digit_count : ℕ := 3

/-- The number of ways to arrange the odd, even, and any digit -/
def digit_arrangements : ℕ := 3

theorem license_plate_count :
  (alphabet_size ^ letter_count) *
  (even_digit_count * odd_digit_count * digit_count) *
  digit_arrangements = 13182000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l11_1163


namespace NUMINAMATH_CALUDE_particle_movement_l11_1133

def num_ways_to_point (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem particle_movement :
  (num_ways_to_point 5 4 = 5) ∧ (num_ways_to_point 20 18 = 190) := by
  sorry

end NUMINAMATH_CALUDE_particle_movement_l11_1133


namespace NUMINAMATH_CALUDE_ball_ratio_l11_1146

theorem ball_ratio (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 → 
  red = 4 → 
  yellow = 2 * red → 
  blue + red + green + yellow = 36 → 
  green / blue = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_l11_1146


namespace NUMINAMATH_CALUDE_four_composition_odd_l11_1126

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem four_composition_odd (f : RealFunction) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end NUMINAMATH_CALUDE_four_composition_odd_l11_1126


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l11_1181

def cost_price : ℝ := 800
def selling_price : ℝ := 680

theorem percentage_loss_calculation : 
  (cost_price - selling_price) / cost_price * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l11_1181


namespace NUMINAMATH_CALUDE_bottle_cap_probability_l11_1178

theorem bottle_cap_probability (p_convex : ℝ) (h1 : p_convex = 0.44) :
  1 - p_convex = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_probability_l11_1178


namespace NUMINAMATH_CALUDE_unknown_number_is_three_l11_1118

theorem unknown_number_is_three (x n : ℝ) (h1 : (3/2) * x - n = 15) (h2 : x = 12) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_three_l11_1118


namespace NUMINAMATH_CALUDE_parabola_properties_l11_1153

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (-1, -8)

-- Define the shift
def m : ℝ := 3

-- Theorem statement
theorem parabola_properties :
  (∀ x, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 ∧
  parabola (m - 0) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l11_1153


namespace NUMINAMATH_CALUDE_inequality_implies_theta_range_l11_1114

open Real

theorem inequality_implies_theta_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  3 * (sin θ ^ 5 + cos (2 * θ) ^ 5) > 5 * (sin θ ^ 3 + cos (2 * θ) ^ 3) →
  θ ∈ Set.Ioo (7 * π / 6) (11 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_theta_range_l11_1114


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l11_1122

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 5 = -3/4)
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) :
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l11_1122


namespace NUMINAMATH_CALUDE_prism_height_l11_1145

/-- Regular prism with base ABC and top A₁B₁C₁ -/
structure RegularPrism where
  a : ℝ  -- side length of the base
  h : ℝ  -- height of the prism
  M : ℝ × ℝ × ℝ  -- midpoint of AC
  N : ℝ × ℝ × ℝ  -- midpoint of A₁B₁

/-- The projection of MN onto BA₁ is a/(2√6) -/
def projection_condition (prism : RegularPrism) : Prop :=
  ∃ (proj : ℝ), proj = prism.a / (2 * Real.sqrt 6)

/-- The theorem stating the possible heights of the prism -/
theorem prism_height (prism : RegularPrism) 
  (h_proj : projection_condition prism) :
  prism.h = prism.a / Real.sqrt 2 ∨ 
  prism.h = prism.a / (2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_prism_height_l11_1145


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l11_1110

/-- The amount of peanut butter in the jar, in tablespoons -/
def jar_amount : ℚ := 37 + 4/5

/-- The amount of peanut butter in one serving, in tablespoons -/
def serving_size : ℚ := 1 + 1/2

/-- The number of servings in the jar -/
def number_of_servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : number_of_servings = 25 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l11_1110


namespace NUMINAMATH_CALUDE_train_crossing_time_l11_1107

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 240 → 
  train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l11_1107


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_correct_l11_1132

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricPointXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem symmetric_point_x_axis_correct :
  let P : Point2D := { x := 5, y := -3 }
  let symmetricP : Point2D := { x := 5, y := 3 }
  symmetricPointXAxis P = symmetricP := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_correct_l11_1132


namespace NUMINAMATH_CALUDE_school_sections_l11_1128

/-- Given a school with 408 boys and 216 girls, prove that when divided into equal sections
    of either boys or girls alone, the total number of sections formed is 26. -/
theorem school_sections (num_boys num_girls : ℕ) 
    (h_boys : num_boys = 408) 
    (h_girls : num_girls = 216) : 
    (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l11_1128


namespace NUMINAMATH_CALUDE_total_money_together_l11_1171

def henry_initial_money : ℕ := 5
def henry_earned_money : ℕ := 2
def friend_money : ℕ := 13

theorem total_money_together : 
  henry_initial_money + henry_earned_money + friend_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_money_together_l11_1171


namespace NUMINAMATH_CALUDE_bakers_cakes_l11_1121

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes : ℕ) : 
  (initial_cakes - 91 + 154 = initial_cakes + 63) →
  initial_cakes = 182 := by
  sorry

#check bakers_cakes

end NUMINAMATH_CALUDE_bakers_cakes_l11_1121


namespace NUMINAMATH_CALUDE_two_digit_triple_sum_product_l11_1101

def digit_sum (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def all_digits_different (p q r : ℕ) : Prop :=
  let digits := [p / 10, p % 10, q / 10, q % 10, r / 10, r % 10]
  ∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry

theorem two_digit_triple_sum_product (p q r : ℕ) : 
  is_two_digit p ∧ is_two_digit q ∧ is_two_digit r ∧ 
  all_digits_different p q r ∧
  p * q * digit_sum r = p * digit_sum q * r ∧
  p * digit_sum q * r = digit_sum p * q * r →
  ((p = 12 ∧ q = 36 ∧ r = 48) ∨ (p = 21 ∧ q = 63 ∧ r = 84)) :=
sorry

end NUMINAMATH_CALUDE_two_digit_triple_sum_product_l11_1101


namespace NUMINAMATH_CALUDE_power_of_negative_power_l11_1127

theorem power_of_negative_power (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_power_l11_1127


namespace NUMINAMATH_CALUDE_triangle_ratio_max_l11_1154

/-- In a triangle ABC, given that the height from BC is a/2, 
    the maximum value of c/b + b/c is 2√2 -/
theorem triangle_ratio_max (a b c : ℝ) (h : ℝ) :
  h = a / 2 →
  (∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    a * h / 2 = b * c * Real.sin A / 2) →
  (∃ (x : ℝ), x = c / b + b / c ∧ 
    ∀ (y : ℝ), y = c / b + b / c → y ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_max_l11_1154


namespace NUMINAMATH_CALUDE_benjamin_steps_to_times_square_l11_1160

/-- The number of steps Benjamin took to reach Rockefeller Center -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square -/
def steps_rockefeller_to_times_square : ℕ := 228

/-- The total number of steps Benjamin took before reaching Times Square -/
def total_steps : ℕ := steps_to_rockefeller + steps_rockefeller_to_times_square

theorem benjamin_steps_to_times_square : total_steps = 582 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_steps_to_times_square_l11_1160


namespace NUMINAMATH_CALUDE_emily_small_gardens_l11_1100

/-- Calculates the number of small gardens Emily has based on her seed distribution --/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_type : ℕ) (vegetable_types : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / (seeds_per_type * vegetable_types)

/-- Theorem stating that Emily has 4 small gardens --/
theorem emily_small_gardens :
  number_of_small_gardens 125 45 4 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l11_1100


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l11_1168

theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (x = -7.5) →
  (-3 * x + y = k) →
  (0.3 * x + y = 12) →
  (k = 36.75) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l11_1168


namespace NUMINAMATH_CALUDE_equation_solution_l11_1191

theorem equation_solution : ∃! x : ℝ, (x - 1) + 2 * Real.sqrt (x + 3) = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l11_1191


namespace NUMINAMATH_CALUDE_double_inequality_solution_l11_1144

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 > (x - 1)^2 ∧ (x - 1)^2 > 3 * x + 6) ↔ 
  (x > 3 + 2 * Real.sqrt 10 ∧ x < (5 + 3 * Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l11_1144


namespace NUMINAMATH_CALUDE_expression_simplification_l11_1105

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l11_1105


namespace NUMINAMATH_CALUDE_square_perimeter_after_scaling_l11_1134

theorem square_perimeter_after_scaling (a : ℝ) (h : a > 0) : 
  let s := Real.sqrt a
  let new_s := 3 * s
  a = 4 → 4 * new_s = 24 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_after_scaling_l11_1134


namespace NUMINAMATH_CALUDE_scaling_property_l11_1170

theorem scaling_property (x y z : ℝ) (h : 2994 * x * 14.5 = 173) : 29.94 * x * 1.45 = 1.73 := by
  sorry

end NUMINAMATH_CALUDE_scaling_property_l11_1170


namespace NUMINAMATH_CALUDE_max_extra_time_matches_2016_teams_l11_1157

/-- Represents a hockey tournament -/
structure HockeyTournament where
  num_teams : Nat
  regular_win_points : Nat
  regular_loss_points : Nat
  extra_time_win_points : Nat
  extra_time_loss_points : Nat

/-- The maximum number of matches that could have ended in extra time -/
def max_extra_time_matches (tournament : HockeyTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of extra time matches for the given tournament -/
theorem max_extra_time_matches_2016_teams 
  (tournament : HockeyTournament)
  (h1 : tournament.num_teams = 2016)
  (h2 : tournament.regular_win_points = 3)
  (h3 : tournament.regular_loss_points = 0)
  (h4 : tournament.extra_time_win_points = 2)
  (h5 : tournament.extra_time_loss_points = 1) :
  max_extra_time_matches tournament = 1512 :=
sorry

end NUMINAMATH_CALUDE_max_extra_time_matches_2016_teams_l11_1157


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l11_1102

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Define the theorem
theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 ∧
  2 * b = 2 ∧
  (Real.sqrt 6) / 3 = Real.sqrt (a^2 - b^2) / a →
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ (k : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse a b x₁ y₁ ∧
      ellipse a b x₂ y₂ ∧
      line k x₁ y₁ ∧
      line k x₂ y₂ ∧
      x₁ ≠ x₂ ∧
      x₁ * x₂ + y₁ * y₂ > 0) ↔
    (k > 1 ∧ k < Real.sqrt 13 / Real.sqrt 3) ∨
    (k < -1 ∧ k > -Real.sqrt 13 / Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l11_1102


namespace NUMINAMATH_CALUDE_sin_product_equality_l11_1103

theorem sin_product_equality : 
  Real.sin (8 * π / 180) * Real.sin (40 * π / 180) * Real.sin (70 * π / 180) * Real.sin (82 * π / 180) = 3 * Real.sqrt 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l11_1103


namespace NUMINAMATH_CALUDE_garden_area_proof_l11_1175

theorem garden_area_proof (x : ℝ) : 
  (x + 2) * (x + 3) = 182 → x^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_proof_l11_1175


namespace NUMINAMATH_CALUDE_last_week_tv_hours_l11_1193

/-- The number of hours of television watched last week -/
def last_week_hours : ℝ := sorry

/-- The average number of hours watched over three weeks -/
def average_hours : ℝ := 10

/-- The number of hours watched the week before last -/
def week_before_hours : ℝ := 8

/-- The number of hours to be watched next week -/
def next_week_hours : ℝ := 12

theorem last_week_tv_hours : last_week_hours = 10 :=
  by
    have h1 : (week_before_hours + last_week_hours + next_week_hours) / 3 = average_hours := by sorry
    sorry


end NUMINAMATH_CALUDE_last_week_tv_hours_l11_1193


namespace NUMINAMATH_CALUDE_all_statements_false_l11_1149

theorem all_statements_false : ∀ (a b c d : ℝ),
  (¬((a ≠ b ∧ c ≠ d) → (a + c ≠ b + d))) ∧
  (¬((a + c ≠ b + d) → (a ≠ b ∧ c ≠ d))) ∧
  (¬(a = b ∧ c = d ∧ a + c ≠ b + d)) ∧
  (¬((a + c = b + d) → (a = b ∨ c = d))) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l11_1149


namespace NUMINAMATH_CALUDE_work_completion_time_l11_1150

/-- Given that A can do a work in 6 days and A and B together can finish the work in 4 days,
    prove that B can do the work alone in 12 days. -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 6)  -- A can do the work in 6 days
  (hab : 1 / a + 1 / b = 1 / 4)  -- A and B together can finish the work in 4 days
  : b = 12 := by  -- B can do the work alone in 12 days
sorry

end NUMINAMATH_CALUDE_work_completion_time_l11_1150


namespace NUMINAMATH_CALUDE_essay_competition_probability_l11_1137

/-- The number of topics in the essay competition -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from a pool of 6 topics is 5/6 -/
theorem essay_competition_probability :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics :=
sorry

end NUMINAMATH_CALUDE_essay_competition_probability_l11_1137


namespace NUMINAMATH_CALUDE_solve_for_y_l11_1182

/-- Custom operation € defined for real numbers -/
def custom_op (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that under given conditions, y must equal 5 -/
theorem solve_for_y (y : ℝ) :
  (custom_op 7 (custom_op 4 y) = 560) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l11_1182


namespace NUMINAMATH_CALUDE_expression_evaluation_l11_1183

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 2
  ((x + 2*y)^2 + (3*x + y)*(3*x - y) - 3*y*(y - x)) / (2*x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l11_1183


namespace NUMINAMATH_CALUDE_solve_channels_problem_l11_1135

def channels_problem (initial_channels : ℕ) 
                     (removed_channels : ℕ) 
                     (replaced_channels : ℕ) 
                     (sports_package_channels : ℕ) 
                     (supreme_sports_package_channels : ℕ) 
                     (final_channels : ℕ) : Prop :=
  let after_company_changes := initial_channels - removed_channels + replaced_channels
  let sports_packages_total := sports_package_channels + supreme_sports_package_channels
  let before_sports_packages := final_channels - sports_packages_total
  after_company_changes - before_sports_packages = 10

theorem solve_channels_problem : 
  channels_problem 150 20 12 8 7 147 := by
  sorry

end NUMINAMATH_CALUDE_solve_channels_problem_l11_1135


namespace NUMINAMATH_CALUDE_ap_terms_count_l11_1156

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Odd n → 
  (n + 1) / 2 * (2 * a + ((n + 1) / 2 - 1) * d) = 30 →
  (n - 1) / 2 * (2 * (a + d) + ((n - 1) / 2 - 1) * d) = 36 →
  n / 2 * (2 * a + (n - 1) * d) = 66 →
  a + (n - 1) * d - a = 12 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_ap_terms_count_l11_1156


namespace NUMINAMATH_CALUDE_cats_remaining_l11_1188

theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 13 → house = 5 → sold = 10 → siamese + house - sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_l11_1188


namespace NUMINAMATH_CALUDE_point_on_line_for_all_k_l11_1187

/-- The point (-2, -1) lies on the line kx + y + 2k + 1 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_for_all_k_l11_1187


namespace NUMINAMATH_CALUDE_correct_sample_sizes_l11_1196

def model1_production : ℕ := 1600
def model2_production : ℕ := 6000
def model3_production : ℕ := 2000
def total_sample_size : ℕ := 48

theorem correct_sample_sizes :
  let total_production := model1_production + model2_production + model3_production
  let sample1 := (model1_production * total_sample_size) / total_production
  let sample2 := (model2_production * total_sample_size) / total_production
  let sample3 := (model3_production * total_sample_size) / total_production
  sample1 = 8 ∧ sample2 = 30 ∧ sample3 = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_sample_sizes_l11_1196


namespace NUMINAMATH_CALUDE_second_smallest_is_four_probability_l11_1106

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 4

def favorable_outcomes : ℕ := 924
def total_outcomes : ℕ := 6435

theorem second_smallest_is_four_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_is_four_probability_l11_1106


namespace NUMINAMATH_CALUDE_remainder_of_2468135792_div_101_l11_1142

theorem remainder_of_2468135792_div_101 :
  (2468135792 : ℕ) % 101 = 52 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2468135792_div_101_l11_1142


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l11_1104

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = √3, prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l11_1104


namespace NUMINAMATH_CALUDE_rectangle_rhombus_perimeter_ratio_l11_1169

/-- The ratio of the perimeter of a 3 by 2 rectangle to the perimeter of a rhombus
    formed by rearranging four congruent right-angled triangles that the rectangle
    is split into is 1:1. -/
theorem rectangle_rhombus_perimeter_ratio :
  let rectangle_length : ℝ := 3
  let rectangle_width : ℝ := 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let triangle_leg1 := rectangle_length / 2
  let triangle_leg2 := rectangle_width
  let triangle_hypotenuse := Real.sqrt (triangle_leg1^2 + triangle_leg2^2)
  let rhombus_side := triangle_hypotenuse
  let rhombus_perimeter := 4 * rhombus_side
  rectangle_perimeter / rhombus_perimeter = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_rhombus_perimeter_ratio_l11_1169


namespace NUMINAMATH_CALUDE_road_travel_cost_l11_1184

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width cost_per_sqm : ℕ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 3 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 3900 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l11_1184


namespace NUMINAMATH_CALUDE_number_of_days_function_l11_1148

/-- The "number of days function" for given day points and a point on its graph -/
theorem number_of_days_function (k : ℝ) :
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ = k * x + 4 ∧ y₂ = 2 * x) →
  (∃ y : ℝ → ℝ, y 2 = 3 ∧ ∀ x : ℝ, y x = (k * x + 4) - (2 * x)) →
  (∃ y : ℝ → ℝ, ∀ x : ℝ, y x = -1/2 * x + 4) :=
by sorry

end NUMINAMATH_CALUDE_number_of_days_function_l11_1148


namespace NUMINAMATH_CALUDE_polynomial_factorization_l11_1195

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b)^2 + (b + c)^2 + (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l11_1195


namespace NUMINAMATH_CALUDE_lcm_12_18_l11_1109

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l11_1109


namespace NUMINAMATH_CALUDE_f_properties_imply_m_range_l11_1176

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the set of valid m values
def valid_m : Set ℝ := {m | m ≤ -2 ∨ m ≥ 2 ∨ m = 0}

theorem f_properties_imply_m_range :
  (∀ x, x ∈ [-1, 1] → f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ a b, a ∈ [-1, 1] → b ∈ [-1, 1] → a + b ≠ 0 → (f a + f b) / (a + b) > 0) →  -- given inequality
  (∀ m, (∀ x a, x ∈ [-1, 1] → a ∈ [-1, 1] → f x ≤ m^2 - 2*a*m + 1) ↔ m ∈ valid_m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_imply_m_range_l11_1176


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l11_1143

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = -4, prove that the common ratio q is -2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) 
  : q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l11_1143


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l11_1173

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3),
    prove that a - b = -1 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l11_1173


namespace NUMINAMATH_CALUDE_max_intersections_square_decagon_l11_1185

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  vertices : Finset (ℝ × ℝ)
  edges : Finset ((ℝ × ℝ) × (ℝ × ℝ))
  convex : Bool
  planar : Bool

/-- Represents the number of sides in a polygon -/
def numSides (p : ConvexPolygon) : ℕ := p.edges.card

/-- Determines if one polygon is inscribed in another -/
def isInscribed (p₁ p₂ : ConvexPolygon) : Prop := sorry

/-- Counts the number of shared vertices between two polygons -/
def sharedVertices (p₁ p₂ : ConvexPolygon) : ℕ := sorry

/-- Counts the number of intersections between edges of two polygons -/
def countIntersections (p₁ p₂ : ConvexPolygon) : ℕ := sorry

theorem max_intersections_square_decagon (p₁ p₂ : ConvexPolygon) : 
  numSides p₁ = 4 →
  numSides p₂ = 10 →
  p₁.convex →
  p₂.convex →
  p₁.planar →
  p₂.planar →
  isInscribed p₁ p₂ →
  sharedVertices p₁ p₂ = 4 →
  countIntersections p₁ p₂ ≤ 8 ∧ 
  ∃ (q₁ q₂ : ConvexPolygon), 
    numSides q₁ = 4 ∧
    numSides q₂ = 10 ∧
    q₁.convex ∧
    q₂.convex ∧
    q₁.planar ∧
    q₂.planar ∧
    isInscribed q₁ q₂ ∧
    sharedVertices q₁ q₂ = 4 ∧
    countIntersections q₁ q₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_square_decagon_l11_1185


namespace NUMINAMATH_CALUDE_john_savings_l11_1174

/-- Calculates the yearly savings when splitting an apartment --/
def yearly_savings (old_rent : ℕ) (price_increase_percent : ℕ) (num_people : ℕ) : ℕ :=
  let new_rent := old_rent + old_rent * price_increase_percent / 100
  let individual_share := new_rent / num_people
  let monthly_savings := old_rent - individual_share
  monthly_savings * 12

/-- Theorem: John saves $7680 per year by splitting the new apartment --/
theorem john_savings : yearly_savings 1200 40 3 = 7680 := by
  sorry

end NUMINAMATH_CALUDE_john_savings_l11_1174


namespace NUMINAMATH_CALUDE_sum_remainder_three_l11_1197

theorem sum_remainder_three (n : ℤ) : (5 - n + (n + 4)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_three_l11_1197


namespace NUMINAMATH_CALUDE_sqrt_greater_than_sum_l11_1116

theorem sqrt_greater_than_sum (a b x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) 
  (hab : a^2 + b^2 < 1) : 
  Real.sqrt (x^2 + y^2) > a*x + b*y := by
sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_sum_l11_1116


namespace NUMINAMATH_CALUDE_discount_ticket_price_l11_1123

theorem discount_ticket_price (discount_rate : ℝ) (discounted_price : ℝ) (original_price : ℝ) :
  discount_rate = 0.3 →
  discounted_price = 1400 →
  discounted_price = (1 - discount_rate) * original_price →
  original_price = 2000 := by
  sorry

end NUMINAMATH_CALUDE_discount_ticket_price_l11_1123


namespace NUMINAMATH_CALUDE_can_cross_all_rivers_l11_1180

def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def existing_bridge : ℕ := 295
def additional_material : ℕ := 1020

def extra_needed (river_width : ℕ) : ℕ :=
  if river_width > existing_bridge then river_width - existing_bridge else 0

theorem can_cross_all_rivers :
  extra_needed river1_width + extra_needed river2_width + extra_needed river3_width ≤ additional_material :=
by sorry

end NUMINAMATH_CALUDE_can_cross_all_rivers_l11_1180


namespace NUMINAMATH_CALUDE_club_officers_count_l11_1108

/-- Represents the number of ways to choose officers from a club with boys and girls --/
def chooseOfficers (boys girls : ℕ) : ℕ :=
  boys * girls * (boys - 1) + girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officers_count :
  chooseOfficers 18 12 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_club_officers_count_l11_1108


namespace NUMINAMATH_CALUDE_hamster_ratio_l11_1130

/-- Proves that the ratio of male hamsters to total hamsters is 1:3 given the specified conditions --/
theorem hamster_ratio (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) 
  (h1 : total_pets = 92)
  (h2 : total_gerbils = 68)
  (h3 : total_males = 25)
  (h4 : total_gerbils * 1/4 = total_gerbils / 4) -- One-quarter of gerbils are male
  (h5 : total_pets = total_gerbils + (total_pets - total_gerbils)) -- Total pets consist of gerbils and hamsters
  : (total_males - total_gerbils / 4) / (total_pets - total_gerbils) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_hamster_ratio_l11_1130


namespace NUMINAMATH_CALUDE_a_property_l11_1111

def gcd_notation (a b : ℕ) : ℕ := Nat.gcd a b

theorem a_property (a : ℕ) : 
  gcd_notation (gcd_notation a 16) (gcd_notation 18 24) = 2 → 
  Even a ∧ ¬(4 ∣ a) := by
  sorry

end NUMINAMATH_CALUDE_a_property_l11_1111


namespace NUMINAMATH_CALUDE_square_root_of_four_l11_1113

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l11_1113


namespace NUMINAMATH_CALUDE_apple_cost_per_kg_main_apple_cost_theorem_l11_1192

/-- Represents the cost structure of apples -/
structure AppleCost where
  p : ℝ  -- Cost per kg for first 30 kgs
  q : ℝ  -- Cost per kg for additional kgs

/-- Theorem stating the cost per kg for first 30 kgs of apples -/
theorem apple_cost_per_kg (cost : AppleCost) : cost.p = 10 :=
  by
  have h1 : 30 * cost.p + 3 * cost.q = 360 := by sorry
  have h2 : 30 * cost.p + 6 * cost.q = 420 := by sorry
  have h3 : 25 * cost.p = 250 := by sorry
  sorry

/-- Main theorem proving the cost per kg for first 30 kgs of apples -/
theorem main_apple_cost_theorem : ∃ (cost : AppleCost), cost.p = 10 :=
  by
  sorry

end NUMINAMATH_CALUDE_apple_cost_per_kg_main_apple_cost_theorem_l11_1192


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l11_1140

def f (x : ℝ) : ℝ := |x - 4| - 2 + 5

theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 4 ∧ f 4 = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l11_1140


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l11_1155

def is_representable (m : ℕ) : Prop :=
  ∃ (n p : ℕ), p.Prime ∧ m = n^2 + p

theorem infinitely_many_non_representable :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ ¬ is_representable m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l11_1155


namespace NUMINAMATH_CALUDE_optimal_promotional_expense_l11_1166

noncomputable section

-- Define the sales volume function
def P (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def profit (x : ℝ) : ℝ := 16 - (4 / (x + 1) + x)

-- Define the theorem
theorem optimal_promotional_expense (a : ℝ) (h : a > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ a → profit x ≤ profit (min 1 a)) ∧
  (a ≥ 1 → profit 1 = (profit ∘ min 1) a) ∧
  (a < 1 → profit a = (profit ∘ min 1) a) := by
  sorry

end

end NUMINAMATH_CALUDE_optimal_promotional_expense_l11_1166


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l11_1167

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 7/8
  | 1 => -14/27
  | 2 => 56/81
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, r = -2/3 ∧
    ∀ n : ℕ, n > 0 → geometric_series n = geometric_series (n-1) * r :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l11_1167


namespace NUMINAMATH_CALUDE_second_player_wins_l11_1164

/-- A game on a circle with 2n + 1 equally spaced points -/
structure CircleGame where
  n : ℕ
  h : n ≥ 2

/-- A strategy for the second player -/
def SecondPlayerStrategy (game : CircleGame) : Type :=
  ℕ → ℕ

/-- Predicate to check if a triangle is obtuse -/
def IsObtuse (p1 p2 p3 : ℕ) : Prop :=
  sorry

/-- Predicate to check if all remaining triangles are obtuse -/
def AllTrianglesObtuse (remaining_points : List ℕ) : Prop :=
  sorry

/-- Predicate to check if a strategy is winning for the second player -/
def IsWinningStrategy (game : CircleGame) (strategy : SecondPlayerStrategy game) : Prop :=
  ∀ (first_player_moves : List ℕ),
    AllTrianglesObtuse (sorry) -- remaining points after applying the strategy

theorem second_player_wins (game : CircleGame) :
  ∃ (strategy : SecondPlayerStrategy game), IsWinningStrategy game strategy :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l11_1164


namespace NUMINAMATH_CALUDE_twice_not_equal_squared_l11_1194

theorem twice_not_equal_squared (m : ℝ) : 2 * m ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_twice_not_equal_squared_l11_1194


namespace NUMINAMATH_CALUDE_f_negative_five_l11_1159

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + 1

theorem f_negative_five (a b : ℝ) 
  (h : f a b 5 = 7) : 
  f a b (-5) = -5 := by
sorry

end NUMINAMATH_CALUDE_f_negative_five_l11_1159


namespace NUMINAMATH_CALUDE_ball_probability_l11_1186

theorem ball_probability (total : Nat) (white green yellow red purple blue black : Nat)
  (h_total : total = 200)
  (h_white : white = 50)
  (h_green : green = 40)
  (h_yellow : yellow = 20)
  (h_red : red = 30)
  (h_purple : purple = 30)
  (h_blue : blue = 10)
  (h_black : black = 20)
  (h_sum : total = white + green + yellow + red + purple + blue + black) :
  (white + green + yellow + blue : ℚ) / total = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l11_1186


namespace NUMINAMATH_CALUDE_unique_b_l11_1151

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of b
def b_properties (b : ℝ) : Prop :=
  b > 1 ∧ 
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ x, x ∈ Set.Icc 1 b → f x ≤ b)

-- Theorem statement
theorem unique_b : ∃! b, b_properties b ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_b_l11_1151


namespace NUMINAMATH_CALUDE_x_plus_y_value_l11_1129

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l11_1129


namespace NUMINAMATH_CALUDE_grandfathers_age_l11_1189

theorem grandfathers_age (grandfather_age : ℕ) (xiaoming_age : ℕ) : 
  grandfather_age > 7 * xiaoming_age →
  grandfather_age < 70 →
  ∃ (k : ℕ), grandfather_age - xiaoming_age = 60 * k →
  grandfather_age = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l11_1189


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l11_1162

theorem unique_solution_sqrt_equation (m n : ℤ) :
  (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n ↔ m = 0 ∧ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l11_1162


namespace NUMINAMATH_CALUDE_problem_solution_l11_1190

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (A = {x : ℝ | -3 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 5}) ∧
  (∀ m : ℝ, A ∪ B m = A ↔ m ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l11_1190


namespace NUMINAMATH_CALUDE_digital_root_of_8_pow_1989_l11_1152

def digital_root (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

theorem digital_root_of_8_pow_1989 :
  digital_root (8^1989) = 8 :=
sorry

end NUMINAMATH_CALUDE_digital_root_of_8_pow_1989_l11_1152


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l11_1141

theorem no_solutions_for_equation (x : ℝ) : 
  x > 6 → 
  ¬(Real.sqrt (x + 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l11_1141


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l11_1120

theorem halfway_between_fractions :
  (1 / 12 + 1 / 20) / 2 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l11_1120


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l11_1199

/-- Represents a trapezoid EFGH with point X dividing the longer base EH -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EX : ℝ
  XH : ℝ

/-- The perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.FG + t.GH + (t.EX + t.XH)

/-- Theorem stating that the perimeter of the given trapezoid is 165 units -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, 
    t.EF = 45 ∧ 
    t.FG = 40 ∧ 
    t.GH = 35 ∧ 
    t.EX = 30 ∧ 
    t.XH = 15 → 
    perimeter t = 165 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l11_1199


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l11_1136

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 10|

-- Define the domain of x
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), (∀ x, domain x → g x ≤ max) ∧ 
                    (∀ x, domain x → min ≤ g x) ∧ 
                    (max + min = 13) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l11_1136


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l11_1138

/-- The number of ways to choose 3 points from n points to form a triangle -/
def triangles_from_points (n : ℕ) : ℕ := n.choose 3

/-- Given 10 points on a circle, the number of inscribed triangles is 360 -/
theorem ten_point_circle_triangles :
  triangles_from_points 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l11_1138


namespace NUMINAMATH_CALUDE_square_division_l11_1179

theorem square_division (original_side : ℝ) (n : ℕ) (smaller_side : ℝ) : 
  original_side = 12 →
  n = 4 →
  smaller_side^2 * n = original_side^2 →
  smaller_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_division_l11_1179


namespace NUMINAMATH_CALUDE_grade_assignment_count_l11_1119

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 12 → num_grades = 4 →
  (num_grades : ℕ) ^ num_students = 16777216 :=
by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l11_1119


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l11_1112

theorem arithmetic_calculation : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l11_1112


namespace NUMINAMATH_CALUDE_train_passing_time_l11_1172

/-- The length of train A in meters -/
def train_a_length : ℝ := 150

/-- The length of train B in meters -/
def train_b_length : ℝ := 200

/-- The time (in seconds) it takes for a passenger on train A to see train B pass by -/
def time_a_sees_b : ℝ := 10

/-- The time (in seconds) it takes for a passenger on train B to see train A pass by -/
def time_b_sees_a : ℝ := 7.5

theorem train_passing_time :
  (train_b_length / time_a_sees_b) = (train_a_length / time_b_sees_a) :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l11_1172
