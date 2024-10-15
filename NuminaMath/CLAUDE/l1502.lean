import Mathlib

namespace NUMINAMATH_CALUDE_annieka_free_throws_l1502_150227

def free_throws_problem (deshawn kayla annieka : ℕ) : Prop :=
  deshawn = 12 ∧
  kayla = deshawn + (deshawn / 2) ∧
  annieka = kayla - 4

theorem annieka_free_throws :
  ∀ deshawn kayla annieka : ℕ,
    free_throws_problem deshawn kayla annieka →
    annieka = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_annieka_free_throws_l1502_150227


namespace NUMINAMATH_CALUDE_sandy_nickels_borrowed_l1502_150268

/-- Given the initial number of nickels and the remaining number of nickels,
    calculate the number of nickels borrowed. -/
def nickels_borrowed (initial : Nat) (remaining : Nat) : Nat :=
  initial - remaining

theorem sandy_nickels_borrowed :
  let initial_nickels : Nat := 31
  let remaining_nickels : Nat := 11
  nickels_borrowed initial_nickels remaining_nickels = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_nickels_borrowed_l1502_150268


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l1502_150259

-- Define a right triangle ABC with angle B = 90°
def RightTriangle (A B C : Real) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧ B = Real.pi / 2

-- State the theorem
theorem right_triangle_sin_value
  (A B C : Real)
  (h_right_triangle : RightTriangle A B C)
  (h_sin_cos_relation : 4 * Real.sin A = 5 * Real.cos A) :
  Real.sin A = 5 * Real.sqrt 41 / 41 := by
    sorry


end NUMINAMATH_CALUDE_right_triangle_sin_value_l1502_150259


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1502_150290

/-- The circumference of the base of a cone formed from a 180° sector of a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = π → 2 * π * r * (θ / (2 * π)) = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1502_150290


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1502_150228

/-- Given a quadratic equation ax² + bx + 3 = 0 with roots -2 and 3,
    prove that the equation a(x+2)² + b(x+2) + 3 = 0 has roots -4 and 1 -/
theorem quadratic_root_transformation (a b : ℝ) :
  (∃ x, a * x^2 + b * x + 3 = 0) →
  ((-2 : ℝ) * (-2 : ℝ) * a + (-2 : ℝ) * b + 3 = 0) →
  ((3 : ℝ) * (3 : ℝ) * a + (3 : ℝ) * b + 3 = 0) →
  (a * ((-4 : ℝ) + 2)^2 + b * ((-4 : ℝ) + 2) + 3 = 0) ∧
  (a * ((1 : ℝ) + 2)^2 + b * ((1 : ℝ) + 2) + 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_transformation_l1502_150228


namespace NUMINAMATH_CALUDE_wills_initial_money_l1502_150266

/-- Will's initial amount of money -/
def initial_money : ℕ := 57

/-- Cost of the game Will bought -/
def game_cost : ℕ := 27

/-- Number of toys Will can buy with the remaining money -/
def num_toys : ℕ := 5

/-- Cost of each toy -/
def toy_cost : ℕ := 6

/-- Theorem stating that Will's initial money is correct given the conditions -/
theorem wills_initial_money :
  initial_money = game_cost + num_toys * toy_cost :=
by sorry

end NUMINAMATH_CALUDE_wills_initial_money_l1502_150266


namespace NUMINAMATH_CALUDE_evaluate_expression_l1502_150212

theorem evaluate_expression : (18 ^ 36) / (54 ^ 18) = 6 ^ 18 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1502_150212


namespace NUMINAMATH_CALUDE_choose_materials_eq_120_l1502_150230

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common -/
def choose_materials : ℕ :=
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  Nat.choose total_materials common_materials *
  (total_materials - common_materials) * (total_materials - common_materials - 1)

theorem choose_materials_eq_120 : choose_materials = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_materials_eq_120_l1502_150230


namespace NUMINAMATH_CALUDE_linda_savings_fraction_l1502_150201

theorem linda_savings_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 880)
  (h2 : tv_cost = 220) :
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_fraction_l1502_150201


namespace NUMINAMATH_CALUDE_cat_catches_rat_l1502_150280

/-- The time (in hours) it takes for the cat to catch the rat after it starts chasing -/
def catchTime : ℝ := 4

/-- The average speed of the cat in km/h -/
def catSpeed : ℝ := 90

/-- The average speed of the rat in km/h -/
def ratSpeed : ℝ := 36

/-- The time (in hours) the cat waits before chasing the rat -/
def waitTime : ℝ := 6

theorem cat_catches_rat : 
  catchTime * catSpeed = (catchTime + waitTime) * ratSpeed :=
by sorry

end NUMINAMATH_CALUDE_cat_catches_rat_l1502_150280


namespace NUMINAMATH_CALUDE_new_average_weight_with_D_l1502_150251

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight
    of the group when D joins is 82 kg. -/
theorem new_average_weight_with_D (w_A w_B w_C w_D : ℝ) : 
  w_A = 95 →
  (w_A + w_B + w_C) / 3 = 80 →
  ∃ w_E : ℝ, w_E = w_D + 3 ∧ (w_B + w_C + w_D + w_E) / 4 = 81 →
  (w_A + w_B + w_C + w_D) / 4 = 82 := by
  sorry


end NUMINAMATH_CALUDE_new_average_weight_with_D_l1502_150251


namespace NUMINAMATH_CALUDE_power_of_power_three_l1502_150272

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_three_l1502_150272


namespace NUMINAMATH_CALUDE_reading_ratio_l1502_150276

/-- Given the reading speeds of Carter, Oliver, and Lucy, prove that the ratio of pages
    Carter can read to pages Lucy can read in 1 hour is 1/2. -/
theorem reading_ratio (carter_pages oliver_pages lucy_extra : ℕ) 
  (h1 : carter_pages = 30)
  (h2 : oliver_pages = 40)
  (h3 : lucy_extra = 20) :
  (carter_pages : ℚ) / ((oliver_pages : ℚ) + lucy_extra) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reading_ratio_l1502_150276


namespace NUMINAMATH_CALUDE_factor_expression_l1502_150269

theorem factor_expression (a m : ℝ) : a * m^2 - a = a * (m - 1) * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1502_150269


namespace NUMINAMATH_CALUDE_compound_interest_duration_l1502_150252

theorem compound_interest_duration (P A r : ℝ) (h_P : P = 979.0209790209791) (h_A : A = 1120) (h_r : r = 0.06) :
  ∃ t : ℝ, A = P * (1 + r) ^ t := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_duration_l1502_150252


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l1502_150296

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : a + b + c > 0) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≥ 2 := by
  sorry

#check min_value_sqrt_sum

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l1502_150296


namespace NUMINAMATH_CALUDE_diana_candies_l1502_150281

/-- The number of candies Diana took out of a box -/
def candies_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem diana_candies :
  let initial_candies : ℕ := 88
  let remaining_candies : ℕ := 82
  candies_taken initial_candies remaining_candies = 6 := by
sorry

end NUMINAMATH_CALUDE_diana_candies_l1502_150281


namespace NUMINAMATH_CALUDE_circles_externally_separate_l1502_150247

theorem circles_externally_separate (m n : ℝ) : 
  2 > 0 ∧ m > 0 ∧ 
  (2 : ℝ)^2 - 10*2 + n = 0 ∧ 
  m^2 - 10*m + n = 0 → 
  n > 2 + m :=
by sorry

end NUMINAMATH_CALUDE_circles_externally_separate_l1502_150247


namespace NUMINAMATH_CALUDE_system_solution_l1502_150274

theorem system_solution (x y z : ℝ) : 
  (x * y = 1 ∧ y * z = 2 ∧ z * x = 8) ↔ 
  ((x = 2 ∧ y = (1/2) ∧ z = 4) ∨ (x = -2 ∧ y = -(1/2) ∧ z = -4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1502_150274


namespace NUMINAMATH_CALUDE_product_mod_seven_l1502_150224

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1502_150224


namespace NUMINAMATH_CALUDE_tangent_perpendicular_line_l1502_150288

-- Define the curve
def C : ℝ → ℝ := fun x ↦ x^2

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at P
def tangent_slope : ℝ := 2

-- Define the perpendicular line
def perpendicular_line (a : ℝ) : ℝ → ℝ := fun x ↦ -a * x - 1

-- State the theorem
theorem tangent_perpendicular_line :
  ∀ a : ℝ, (C P.1 = P.2) →
  (tangent_slope * (-1/a) = -1) →
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_line_l1502_150288


namespace NUMINAMATH_CALUDE_max_gangsters_is_35_l1502_150225

/-- Represents a gang in Chicago -/
structure Gang :=
  (id : Nat)

/-- Represents a gangster in Chicago -/
structure Gangster :=
  (id : Nat)

/-- The total number of gangs in Chicago -/
def totalGangs : Nat := 36

/-- Represents the conflict relation between gangs -/
def inConflict : Gang → Gang → Prop := sorry

/-- Represents the membership of a gangster in a gang -/
def isMember : Gangster → Gang → Prop := sorry

/-- All gangsters belong to multiple gangs -/
axiom multiple_membership (g : Gangster) : ∃ (g1 g2 : Gang), g1 ≠ g2 ∧ isMember g g1 ∧ isMember g g2

/-- Any two gangsters belong to different sets of gangs -/
axiom different_memberships (g1 g2 : Gangster) : g1 ≠ g2 → ∃ (gang : Gang), (isMember g1 gang ∧ ¬isMember g2 gang) ∨ (isMember g2 gang ∧ ¬isMember g1 gang)

/-- No gangster belongs to two gangs that are in conflict -/
axiom no_conflict_membership (g : Gangster) (gang1 gang2 : Gang) : isMember g gang1 → isMember g gang2 → ¬inConflict gang1 gang2

/-- Each gang not including a gangster is in conflict with some gang including that gangster -/
axiom conflict_with_member_gang (g : Gangster) (gang1 : Gang) : ¬isMember g gang1 → ∃ (gang2 : Gang), isMember g gang2 ∧ inConflict gang1 gang2

/-- The maximum number of gangsters in Chicago -/
def maxGangsters : Nat := 35

/-- Theorem: The maximum number of gangsters in Chicago is 35 -/
theorem max_gangsters_is_35 : ∀ (gangsters : Finset Gangster), gangsters.card ≤ maxGangsters :=
  sorry

end NUMINAMATH_CALUDE_max_gangsters_is_35_l1502_150225


namespace NUMINAMATH_CALUDE_bus_ride_cost_l1502_150293

theorem bus_ride_cost (bus_cost train_cost : ℝ) : 
  train_cost = bus_cost + 6.85 →
  bus_cost + train_cost = 9.65 →
  bus_cost = 1.40 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l1502_150293


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1502_150220

/-- A point P with coordinates (2m-6, m-1) lies on the x-axis if and only if its coordinates are (-4, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (2*m - 6, m - 1) ∧ P.2 = 0) ↔ (∃ P : ℝ × ℝ, P = (-4, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1502_150220


namespace NUMINAMATH_CALUDE_coin_coverage_probability_l1502_150261

/-- The probability of a coin covering part of the black region on a square -/
theorem coin_coverage_probability (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) : 
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  (78 + 5 * Real.pi + 12 * Real.sqrt 2) / 64 = 
    (4 * (triangle_leg^2 / 2 + Real.pi + 2 * triangle_leg) + 
     2 * diamond_side^2 + Real.pi + 4 * diamond_side) / 
    ((square_side - coin_diameter)^2) := by
  sorry

end NUMINAMATH_CALUDE_coin_coverage_probability_l1502_150261


namespace NUMINAMATH_CALUDE_range_of_x_l1502_150255

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x, f x a ≥ 3) (h3 : ∃ x, f x a = 3) :
  ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1502_150255


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1502_150221

def fill_cistern (problem : ℝ → Prop) : Prop :=
  ∃ t : ℝ,
    -- Tap A fills 1/12 of the cistern per minute
    let rate_A := 1 / 12
    -- Tap B fills 1/t of the cistern per minute
    let rate_B := 1 / t
    -- Both taps run for 4 minutes
    let combined_fill := 4 * (rate_A + rate_B)
    -- Tap B runs for 8 more minutes
    let remaining_fill := 8 * rate_B
    -- The total fill is 1 (complete cistern)
    combined_fill + remaining_fill = 1 ∧
    -- The solution satisfies the original problem
    problem t

theorem cistern_fill_time :
  fill_cistern (λ t ↦ t = 18) :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1502_150221


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1502_150241

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 25 ∧ x - y = 7 → x * y = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1502_150241


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l1502_150260

def n : ℕ := 5
def k : ℕ := 2
def p : ℚ := 2/5

theorem magic_8_ball_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 216/625 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l1502_150260


namespace NUMINAMATH_CALUDE_remainder_of_product_mod_17_l1502_150265

theorem remainder_of_product_mod_17 : (157^3 * 193^4) % 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_mod_17_l1502_150265


namespace NUMINAMATH_CALUDE_shadow_length_l1502_150217

/-- Given a flagpole and a building under similar conditions, this theorem calculates
    the length of the shadow cast by the building. -/
theorem shadow_length
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_height : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 24)
  : (building_height * flagpole_shadow) / flagpole_height = 60 := by
  sorry


end NUMINAMATH_CALUDE_shadow_length_l1502_150217


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1502_150204

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0) ↔ m ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1502_150204


namespace NUMINAMATH_CALUDE_system_solution_l1502_150208

theorem system_solution :
  ∃ (x y z : ℚ),
    (x + (1/3)*y + (1/3)*z = 14) ∧
    (y + (1/4)*x + (1/4)*z = 8) ∧
    (z + (1/5)*x + (1/5)*y = 8) ∧
    (x = 11) ∧ (y = 4) ∧ (z = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1502_150208


namespace NUMINAMATH_CALUDE_f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l1502_150236

/-- The function f(x) = x³ + ax² + bx + 3 -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 3

/-- A point x₀ is a fixed point of f if f(x₀) = x₀ -/
def is_fixed_point (a b x₀ : ℝ) : Prop := f a b x₀ = x₀

/-- A point x₀ is an extremum point of f if f'(x₀) = 0 -/
def is_extremum_point (a b x₀ : ℝ) : Prop :=
  3*x₀^2 + 2*a*x₀ + b = 0

theorem f_monotonic_intervals (b : ℝ) :
  (b ≥ 0 → StrictMono (f 0 b)) ∧
  (b < 0 → StrictMonoOn (f 0 b) {x | x < -Real.sqrt (-b/3) ∨ x > Real.sqrt (-b/3)}) :=
sorry

theorem fixed_and_extremum_point_condition :
  ∃ x₀ : ℝ, is_fixed_point 0 (-3) x₀ ∧ is_extremum_point 0 (-3) x₀ :=
sorry

theorem no_two_distinct_extrema_fixed_points :
  ¬∃ a b x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    is_fixed_point a b x₁ ∧ is_extremum_point a b x₁ ∧
    is_fixed_point a b x₂ ∧ is_extremum_point a b x₂ :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_intervals_fixed_and_extremum_point_condition_no_two_distinct_extrema_fixed_points_l1502_150236


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l1502_150267

theorem absolute_value_sum_zero (a b : ℝ) (h : |a - 3| + |b + 5| = 0) : 
  (a + b = -2) ∧ (|a| + |b| = 8) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l1502_150267


namespace NUMINAMATH_CALUDE_matrix_determinant_l1502_150262

def matrix : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, 0; 8, 5, -2; 3, -1, 6]

theorem matrix_determinant :
  Matrix.det matrix = 138 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l1502_150262


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1502_150226

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a = 1 → (a - 1) * (a - 2) = 0) ∧ 
  (∃ a : ℝ, (a - 1) * (a - 2) = 0 ∧ a ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1502_150226


namespace NUMINAMATH_CALUDE_campground_distance_l1502_150209

/-- Calculates the total distance traveled given multiple segments of driving at different speeds. -/
def total_distance (segments : List (ℝ × ℝ)) : ℝ :=
  segments.map (fun (speed, time) => speed * time) |>.sum

/-- The driving segments for Sue's family vacation. -/
def vacation_segments : List (ℝ × ℝ) :=
  [(50, 3), (60, 2), (55, 1), (65, 2)]

/-- Theorem stating that the total distance to the campground is 455 miles. -/
theorem campground_distance :
  total_distance vacation_segments = 455 := by
  sorry

#eval total_distance vacation_segments

end NUMINAMATH_CALUDE_campground_distance_l1502_150209


namespace NUMINAMATH_CALUDE_one_third_minus_zero_point_three_three_three_l1502_150248

theorem one_third_minus_zero_point_three_three_three :
  (1 : ℚ) / 3 - (333 : ℚ) / 1000 = 1 / (3 * 1000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_zero_point_three_three_three_l1502_150248


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l1502_150295

theorem complex_number_coordinates : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + 2 * i^3) / (2 + i)
  Complex.re z = 0 ∧ Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l1502_150295


namespace NUMINAMATH_CALUDE_michaels_fish_count_l1502_150239

theorem michaels_fish_count 
  (initial_fish : ℝ) 
  (fish_from_ben : ℝ) 
  (fish_from_maria : ℝ) 
  (h1 : initial_fish = 49.5)
  (h2 : fish_from_ben = 18.25)
  (h3 : fish_from_maria = 23.75) :
  initial_fish + fish_from_ben + fish_from_maria = 91.5 := by
  sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l1502_150239


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1502_150202

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3/2 - 1/(3*x - 1) = 5/(6*x - 2)) ↔ (x = 10/9) :=
sorry

-- Equation 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℚ), (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1502_150202


namespace NUMINAMATH_CALUDE_mod_difference_equals_negative_four_l1502_150205

-- Define the % operation
def mod (x y : ℤ) : ℤ := x * y - 3 * x - y

-- State the theorem
theorem mod_difference_equals_negative_four : 
  (mod 6 4) - (mod 4 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_mod_difference_equals_negative_four_l1502_150205


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1502_150282

theorem max_value_of_expression (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ (∃ x' y' z' : ℕ, (10 ≤ x' ∧ x' ≤ 99) ∧ (10 ≤ y' ∧ y' ≤ 99) ∧ (10 ≤ z' ∧ z' ≤ 99) ∧ ((x' + y' + z') / 3 = 60) ∧ ((x' + y') / z' = 17)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1502_150282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1502_150243

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term : a 1 = 3
  arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Main theorem -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : S seq 8 = seq.a 8) : seq.a 19 = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1502_150243


namespace NUMINAMATH_CALUDE_exists_identical_triangles_l1502_150264

-- Define a triangle type
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)

-- Define a function to represent a cut operation
def cut (t : Triangle) : (Triangle × Triangle) := sorry

-- Define a function to check if two triangles are identical
def are_identical (t1 t2 : Triangle) : Prop := sorry

-- Define the initial set of triangles
def initial_triangles : Finset Triangle := sorry

-- Define the set of triangles after n cuts
def triangles_after_cuts (n : ℕ) : Finset Triangle := sorry

-- The main theorem
theorem exists_identical_triangles (n : ℕ) :
  ∃ t1 t2 : Triangle, t1 ∈ triangles_after_cuts n ∧ t2 ∈ triangles_after_cuts n ∧ t1 ≠ t2 ∧ are_identical t1 t2 :=
sorry

end NUMINAMATH_CALUDE_exists_identical_triangles_l1502_150264


namespace NUMINAMATH_CALUDE_product_digit_sum_l1502_150235

/-- The number of 9's in the factor that, when multiplied by 9, 
    produces a number whose digits sum to 1111 -/
def k : ℕ := 124

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The factor consisting of k 9's -/
def factor (k : ℕ) : ℕ :=
  10^k - 1

theorem product_digit_sum :
  sum_of_digits (9 * factor k) = 1111 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1502_150235


namespace NUMINAMATH_CALUDE_part_I_part_II_l1502_150263

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC -/
def cosineLaw (t : Triangle) : Prop :=
  2 * t.b - t.c = 2 * t.a * Real.cos t.C

/-- Additional condition for part II -/
def additionalCondition (t : Triangle) : Prop :=
  4 * (t.b + t.c) = 3 * t.b * t.c

/-- Theorem for part I -/
theorem part_I (t : Triangle) (h : cosineLaw t) : t.A = 2 * Real.pi / 3 := by sorry

/-- Theorem for part II -/
theorem part_II (t : Triangle) (h1 : cosineLaw t) (h2 : additionalCondition t) (h3 : t.a = 2 * Real.sqrt 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1502_150263


namespace NUMINAMATH_CALUDE_eccentricity_properties_l1502_150294

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the line y = x + 4
def line (x : ℝ) : ℝ := x + 4

-- Define the eccentricity function
noncomputable def eccentricity (x₀ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (x₀, line x₀)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let a := (PA + PB) / 2
  let c := 2  -- half the distance between foci
  c / a

-- Theorem statement
theorem eccentricity_properties :
  (∀ ε > 0, ∃ x₀ : ℝ, eccentricity x₀ < ε) ∧
  (∃ M : ℝ, ∀ x₀ : ℝ, eccentricity x₀ ≤ M) :=
sorry

end NUMINAMATH_CALUDE_eccentricity_properties_l1502_150294


namespace NUMINAMATH_CALUDE_point_D_coordinates_l1502_150271

def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-1, 5)

theorem point_D_coordinates :
  let AD : ℝ × ℝ := (3 * (B.1 - A.1), 3 * (B.2 - A.2))
  let D : ℝ × ℝ := (A.1 + AD.1, A.2 + AD.2)
  D = (-7, 9) := by sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l1502_150271


namespace NUMINAMATH_CALUDE_dolphins_score_l1502_150237

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 72 →
  sharks_points - dolphins_points = 20 →
  sharks_points ≥ 2 * dolphins_points →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 26 := by
sorry

end NUMINAMATH_CALUDE_dolphins_score_l1502_150237


namespace NUMINAMATH_CALUDE_probability_theorem_l1502_150210

def harmonic_number (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (λ i => 1 / (i + 1 : ℚ))

def probability_all_own_hats (n : ℕ) : ℚ :=
  (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ)

theorem probability_theorem (n : ℕ) :
  probability_all_own_hats n =
    (Finset.prod (Finset.range n) (λ i => harmonic_number (i + 1))) / (n.factorial : ℚ) :=
by sorry

#eval probability_all_own_hats 10

end NUMINAMATH_CALUDE_probability_theorem_l1502_150210


namespace NUMINAMATH_CALUDE_cubic_factorization_l1502_150284

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1502_150284


namespace NUMINAMATH_CALUDE_smallest_positive_e_l1502_150214

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure IntPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : IntPolynomial) (x : ℚ) : Prop :=
  p.a * x^4 + p.b * x^3 + p.c * x^2 + p.d * x + p.e = 0

/-- The main theorem stating the smallest possible value of e -/
theorem smallest_positive_e (p : IntPolynomial) : 
  p.e > 0 → 
  isRoot p (-2) → 
  isRoot p 5 → 
  isRoot p 9 → 
  isRoot p (-1/3) → 
  p.e ≥ 90 ∧ ∃ q : IntPolynomial, q.e = 90 ∧ 
    isRoot q (-2) ∧ isRoot q 5 ∧ isRoot q 9 ∧ isRoot q (-1/3) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_e_l1502_150214


namespace NUMINAMATH_CALUDE_football_club_player_selling_price_l1502_150289

/-- Calculates the selling price of each player given the financial transactions of a football club. -/
theorem football_club_player_selling_price 
  (initial_balance : ℝ) 
  (players_sold : ℕ) 
  (players_bought : ℕ) 
  (buying_price : ℝ) 
  (final_balance : ℝ) : 
  initial_balance + players_sold * ((initial_balance - final_balance + players_bought * buying_price) / players_sold) - players_bought * buying_price = final_balance → 
  (initial_balance - final_balance + players_bought * buying_price) / players_sold = 10 :=
by sorry

end NUMINAMATH_CALUDE_football_club_player_selling_price_l1502_150289


namespace NUMINAMATH_CALUDE_least_period_is_36_l1502_150231

-- Define the property that f must satisfy
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define what it means for a function to have a period
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- Define the least positive period
def is_least_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ has_period f p ∧ ∀ q : ℝ, 0 < q ∧ q < p → ¬(has_period f q)

-- The main theorem
theorem least_period_is_36 (f : ℝ → ℝ) (h : satisfies_condition f) :
  is_least_positive_period f 36 := by
  sorry

end NUMINAMATH_CALUDE_least_period_is_36_l1502_150231


namespace NUMINAMATH_CALUDE_m_range_l1502_150211

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem m_range (m : ℝ) (h1 : ¬(p m)) (h2 : p m ∨ q m) : 1 < m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1502_150211


namespace NUMINAMATH_CALUDE_debate_school_ratio_l1502_150279

/-- The number of students in the third school -/
def third_school : ℕ := 200

/-- The number of students in the second school -/
def second_school : ℕ := third_school + 40

/-- The total number of students who shook the mayor's hand -/
def total_students : ℕ := 920

/-- The number of students in the first school -/
def first_school : ℕ := total_students - second_school - third_school

/-- The ratio of students in the first school to students in the second school -/
def school_ratio : ℚ := first_school / second_school

theorem debate_school_ratio : school_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_debate_school_ratio_l1502_150279


namespace NUMINAMATH_CALUDE_shelby_gold_stars_l1502_150254

def gold_stars_problem (yesterday : ℕ) (total : ℕ) : Prop :=
  ∃ today : ℕ, yesterday + today = total

theorem shelby_gold_stars :
  gold_stars_problem 4 7 → ∃ today : ℕ, today = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shelby_gold_stars_l1502_150254


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2939_l1502_150229

theorem smallest_prime_factor_of_2939 :
  (Nat.minFac 2939 = 13) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2939_l1502_150229


namespace NUMINAMATH_CALUDE_limit_x_cubed_minus_eight_over_x_minus_two_l1502_150287

theorem limit_x_cubed_minus_eight_over_x_minus_two : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |((x^3 - 8) / (x - 2)) - 12| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_x_cubed_minus_eight_over_x_minus_two_l1502_150287


namespace NUMINAMATH_CALUDE_line_mb_value_l1502_150256

/-- A line in the 2D plane -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of a point on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

theorem line_mb_value (l : Line) :
  l.contains 0 (-1) → l.contains 1 1 → l.m * l.b = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_mb_value_l1502_150256


namespace NUMINAMATH_CALUDE_count_squares_specific_grid_l1502_150249

/-- Represents a grid with a diagonal line --/
structure DiagonalGrid :=
  (width : Nat)
  (height : Nat)
  (diagonalLength : Nat)

/-- Counts the number of squares in a diagonal grid --/
def countSquares (g : DiagonalGrid) : Nat :=
  sorry

/-- The specific 6x5 grid with a diagonal in the top-left 3x3 square --/
def specificGrid : DiagonalGrid :=
  { width := 6, height := 5, diagonalLength := 3 }

/-- Theorem stating that the number of squares in the specific grid is 64 --/
theorem count_squares_specific_grid :
  countSquares specificGrid = 64 := by sorry

end NUMINAMATH_CALUDE_count_squares_specific_grid_l1502_150249


namespace NUMINAMATH_CALUDE_coefficient_implies_a_value_l1502_150285

theorem coefficient_implies_a_value (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = (1 + a * x)^5 * (1 - 2*x)^4) ∧
   (∃ c : ℝ → ℝ, (∀ x, f x = c 0 + c 1 * x + c 2 * x^2 + c 3 * x^3 + c 4 * x^4 + c 5 * x^5 + c 6 * x^6 + c 7 * x^7 + c 8 * x^8 + c 9 * x^9) ∧
    c 2 = -16)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_coefficient_implies_a_value_l1502_150285


namespace NUMINAMATH_CALUDE_elizabeth_position_l1502_150206

theorem elizabeth_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) : 
  total_distance = 24 → 
  total_steps = 6 → 
  steps_taken = 4 → 
  (total_distance / total_steps) * steps_taken = 16 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_position_l1502_150206


namespace NUMINAMATH_CALUDE_license_plate_difference_l1502_150246

theorem license_plate_difference : 
  let georgia_plates := 26^5 * 10^2
  let texas_plates := 26^4 * 10^3
  georgia_plates - texas_plates = 731161600 := by
sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1502_150246


namespace NUMINAMATH_CALUDE_candy_eaten_count_l1502_150299

/-- Represents the number of candy pieces collected and eaten by Travis and his brother -/
structure CandyCount where
  initial : ℕ
  remaining : ℕ
  eaten : ℕ

/-- Theorem stating that the difference between initial and remaining candy count equals the eaten count -/
theorem candy_eaten_count (c : CandyCount) (h1 : c.initial = 68) (h2 : c.remaining = 60) :
  c.eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_eaten_count_l1502_150299


namespace NUMINAMATH_CALUDE_actual_average_height_l1502_150213

/-- Proves that the actual average height of students is 174.62 cm given the initial conditions --/
theorem actual_average_height (n : ℕ) (initial_avg : ℝ) 
  (h1_recorded h1_actual h2_recorded h2_actual h3_recorded h3_actual : ℝ) :
  n = 50 ∧ 
  initial_avg = 175 ∧
  h1_recorded = 151 ∧ h1_actual = 136 ∧
  h2_recorded = 162 ∧ h2_actual = 174 ∧
  h3_recorded = 185 ∧ h3_actual = 169 →
  (n : ℝ) * initial_avg - (h1_recorded - h1_actual + h2_recorded - h2_actual + h3_recorded - h3_actual) = n * 174.62 :=
by sorry

end NUMINAMATH_CALUDE_actual_average_height_l1502_150213


namespace NUMINAMATH_CALUDE_jill_jack_time_ratio_l1502_150218

/-- The ratio of Jill's time to Jack's time for a given route -/
theorem jill_jack_time_ratio (d : ℝ) (x y : ℝ) : 
  (x = d / (2 * 6) + d / (2 * 12)) →
  (y = d / (3 * 5) + 2 * d / (3 * 15)) →
  x / y = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_jill_jack_time_ratio_l1502_150218


namespace NUMINAMATH_CALUDE_zero_in_interval_l1502_150222

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →               -- f is continuous
  f 1 > 0 →                    -- f(1) > 0
  f 2 < 0 →                    -- f(2) < 0
  ∃ z, z ∈ Set.Ioo 1 2 ∧ f z = 0 := by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l1502_150222


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1502_150240

theorem circle_area_ratio (r : ℝ) (h : r > 0) : 
  (π * r^2) / (π * (3*r)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1502_150240


namespace NUMINAMATH_CALUDE_decagon_perimeter_l1502_150233

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The length of each side of the regular decagon -/
def side_length : ℝ := 3

/-- The perimeter of a regular polygon -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a regular decagon with side length 3 units is 30 units -/
theorem decagon_perimeter : 
  perimeter decagon_sides side_length = 30 := by sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l1502_150233


namespace NUMINAMATH_CALUDE_hermia_election_probability_l1502_150203

theorem hermia_election_probability (n : ℕ) (hodd : Odd n) (hpos : 0 < n) :
  let p := (2^n - 1) / (n * 2^(n-1) : ℝ)
  ∃ (probability_hermia_elected : ℝ),
    probability_hermia_elected = p ∧
    0 ≤ probability_hermia_elected ∧
    probability_hermia_elected ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_hermia_election_probability_l1502_150203


namespace NUMINAMATH_CALUDE_car_speed_conversion_l1502_150200

/-- Converts speed from m/s to km/h -/
def speed_ms_to_kmh (speed_ms : ℝ) : ℝ := speed_ms * 3.6

/-- Given a car's speed of 10 m/s, its speed in km/h is 36 km/h -/
theorem car_speed_conversion :
  let speed_ms : ℝ := 10
  speed_ms_to_kmh speed_ms = 36 := by sorry

end NUMINAMATH_CALUDE_car_speed_conversion_l1502_150200


namespace NUMINAMATH_CALUDE_product_of_numbers_l1502_150286

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 194) : x * y = -25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1502_150286


namespace NUMINAMATH_CALUDE_cookie_radius_cookie_is_circle_l1502_150275

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) ↔ ((x - 3)^2 + (y - 5)^2 = 17) :=
by sorry

theorem cookie_is_circle (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) → ∃ (center_x center_y radius : ℝ),
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ (radius = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_cookie_is_circle_l1502_150275


namespace NUMINAMATH_CALUDE_value_of_a_l1502_150298

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 9 * x^2 + 6 * x - 7

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 18 * x + 6

-- Theorem statement
theorem value_of_a (a : ℝ) : f' a (-1) = 4 → a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1502_150298


namespace NUMINAMATH_CALUDE_unique_linear_function_l1502_150216

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem unique_linear_function :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x ∈ [0, 1] → y ∈ [0, 1] → |f a b x + f a b y - x * y| ≤ 1/4) →
  f a b = f (1/2) (-1/8) := by
sorry

end NUMINAMATH_CALUDE_unique_linear_function_l1502_150216


namespace NUMINAMATH_CALUDE_vector_BC_coordinates_l1502_150297

theorem vector_BC_coordinates :
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (3, 2)
  let AC : ℝ × ℝ := (4, 3)
  let BC : ℝ × ℝ := (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2)
  BC = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_BC_coordinates_l1502_150297


namespace NUMINAMATH_CALUDE_sams_remaining_money_l1502_150223

/-- Given an initial amount of money, the cost per book, and the number of books bought,
    calculate the remaining money after the purchase. -/
def remaining_money (initial_amount cost_per_book num_books : ℕ) : ℕ :=
  initial_amount - cost_per_book * num_books

/-- Theorem stating that given the specific conditions of Sam's book purchase,
    the remaining money is 16 dollars. -/
theorem sams_remaining_money :
  remaining_money 79 7 9 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_money_l1502_150223


namespace NUMINAMATH_CALUDE_smallest_norm_v_l1502_150234

theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_v_l1502_150234


namespace NUMINAMATH_CALUDE_twopirsquared_is_standard_l1502_150273

/-- Represents a mathematical expression -/
inductive MathExpression
  | Constant (c : ℝ)
  | Variable (v : String)
  | Multiplication (e1 e2 : MathExpression)
  | Exponentiation (base : MathExpression) (exponent : ℕ)

/-- Checks if an expression follows standard mathematical notation -/
def isStandardNotation : MathExpression → Bool
  | MathExpression.Constant _ => true
  | MathExpression.Variable _ => true
  | MathExpression.Multiplication e1 e2 => 
      match e1, e2 with
      | MathExpression.Constant _, _ => isStandardNotation e2
      | _, _ => false
  | MathExpression.Exponentiation base _ => isStandardNotation base

/-- Represents the expression 2πr² -/
def twopirsquared : MathExpression :=
  MathExpression.Multiplication
    (MathExpression.Constant 2)
    (MathExpression.Multiplication
      (MathExpression.Variable "π")
      (MathExpression.Exponentiation (MathExpression.Variable "r") 2))

/-- Theorem stating that 2πr² follows standard mathematical notation -/
theorem twopirsquared_is_standard : isStandardNotation twopirsquared = true := by
  sorry

end NUMINAMATH_CALUDE_twopirsquared_is_standard_l1502_150273


namespace NUMINAMATH_CALUDE_james_pizza_slices_l1502_150270

theorem james_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_fraction : ℚ) : 
  num_pizzas = 2 → 
  slices_per_pizza = 6 → 
  james_fraction = 2/3 →
  (↑num_pizzas * ↑slices_per_pizza : ℚ) * james_fraction = 8 := by
  sorry

end NUMINAMATH_CALUDE_james_pizza_slices_l1502_150270


namespace NUMINAMATH_CALUDE_existence_of_unfactorable_number_l1502_150277

theorem existence_of_unfactorable_number (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∃ y : ℕ, y < p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unfactorable_number_l1502_150277


namespace NUMINAMATH_CALUDE_addition_of_decimals_l1502_150250

theorem addition_of_decimals : (0.3 : ℝ) + 0.03 = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_decimals_l1502_150250


namespace NUMINAMATH_CALUDE_total_amount_is_156_l1502_150245

-- Define the ratio of shares
def x_share : ℚ := 1
def y_share : ℚ := 45 / 100
def z_share : ℚ := 50 / 100

-- Define y's actual share
def y_actual_share : ℚ := 36

-- Theorem to prove
theorem total_amount_is_156 :
  let x_actual_share := y_actual_share / y_share
  let total_amount := x_actual_share * (x_share + y_share + z_share)
  total_amount = 156 := by
sorry


end NUMINAMATH_CALUDE_total_amount_is_156_l1502_150245


namespace NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l1502_150258

/-- Given that James ate 22 carrot sticks before dinner and 37 carrot sticks in total,
    prove that he ate 15 carrot sticks after dinner. -/
theorem carrot_sticks_after_dinner
  (before_dinner : ℕ)
  (total : ℕ)
  (h1 : before_dinner = 22)
  (h2 : total = 37) :
  total - before_dinner = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_after_dinner_l1502_150258


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1502_150232

-- Define set A
def A : Set ℝ := {x | 2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1502_150232


namespace NUMINAMATH_CALUDE_equal_sums_iff_odd_l1502_150244

def is_valid_seating (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∀ (boy : ℕ) (girl1 : ℕ) (girl2 : ℕ),
    boy ≤ n ∧ n < girl1 ∧ girl1 ≤ 2*n ∧ n < girl2 ∧ girl2 ≤ 2*n →
    boy + girl1 + girl2 = 4*n + (3*n + 3)/2

theorem equal_sums_iff_odd (n : ℕ) :
  is_valid_seating n ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_equal_sums_iff_odd_l1502_150244


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1502_150238

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1502_150238


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l1502_150257

/-- The ellipse with equation x²/49 + y²/24 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 49) + (p.2^2 / 24) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  (distance P F₁) / (distance P F₂) = 4 / 3 →
  triangleArea P F₁ F₂ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l1502_150257


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_2022_starting_with_2023_l1502_150292

def starts_with (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = m * 10^k + (n % 10^k) ∧ m * 10^k > n / 10

theorem smallest_number_divisible_by_2022_starting_with_2023 :
  ∀ n : ℕ, (n % 2022 = 0 ∧ starts_with n 2023) → n ≥ 20230110 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_2022_starting_with_2023_l1502_150292


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1502_150291

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1502_150291


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1502_150253

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  0.15 * x + 0.35 * 200 = 0.23 * (x + 200) → 
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1502_150253


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1502_150215

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  f 0 = 1 ∧
  ∀ x, f (x + 1) - f x = 2 * x

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∀ x, f x = x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1502_150215


namespace NUMINAMATH_CALUDE_multiplication_result_l1502_150283

theorem multiplication_result : 9995 * 82519 = 824777405 := by sorry

end NUMINAMATH_CALUDE_multiplication_result_l1502_150283


namespace NUMINAMATH_CALUDE_equal_probability_sums_l1502_150207

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The minimum face value on each die -/
def min_face : ℕ := 1

/-- The maximum face value on each die -/
def max_face : ℕ := 6

/-- The sum we're comparing against -/
def sum1 : ℕ := 12

/-- The sum that should have the same probability as sum1 -/
def sum2 : ℕ := 44

/-- The probability of obtaining a specific sum when rolling num_dice dice -/
noncomputable def prob_sum (s : ℕ) : ℝ := sorry

theorem equal_probability_sums : prob_sum sum1 = prob_sum sum2 := by sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l1502_150207


namespace NUMINAMATH_CALUDE_log_inequality_l1502_150278

theorem log_inequality (x : ℝ) : 
  (Real.log (1 + 8 * x^5) / Real.log (1 + x^2) + 
   Real.log (1 + x^2) / Real.log (1 - 3 * x^2 + 16 * x^4) ≤ 
   1 + Real.log (1 + 8 * x^5) / Real.log (1 - 3 * x^2 + 16 * x^4)) ↔ 
  (x ∈ Set.Ioc (-((1/8)^(1/5))) (-1/2) ∪ 
       Set.Ioo (-Real.sqrt 3 / 4) 0 ∪ 
       Set.Ioo 0 (Real.sqrt 3 / 4) ∪ 
       {1/2}) := by sorry

end NUMINAMATH_CALUDE_log_inequality_l1502_150278


namespace NUMINAMATH_CALUDE_pipe_filling_time_l1502_150219

theorem pipe_filling_time (rate_A rate_B : ℝ) (time_B : ℝ) : 
  rate_A = 1 / 12 →
  rate_B = 1 / 36 →
  time_B = 12 →
  ∃ time_A : ℝ, time_A * rate_A + time_B * rate_B = 1 ∧ time_A = 8 :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l1502_150219


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1502_150242

/-- A parallelogram with opposite vertices at (2, -3) and (10, 9) has its diagonals intersecting at (6, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (10, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (6, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1502_150242
