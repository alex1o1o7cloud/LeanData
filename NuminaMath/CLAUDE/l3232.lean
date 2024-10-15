import Mathlib

namespace NUMINAMATH_CALUDE_class_size_calculation_l3232_323233

theorem class_size_calculation (incorrect_mark : ℕ) (correct_mark : ℕ) (average_increase : ℚ) : 
  incorrect_mark = 67 → 
  correct_mark = 45 → 
  average_increase = 1/2 →
  (incorrect_mark - correct_mark : ℚ) / (2 * average_increase) = 44 :=
by sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3232_323233


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l3232_323261

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun d => ∃ x₁ x₂ : ℝ,
    (|x₁ - 3| = 15) ∧
    (|x₂ - 3| = 15) ∧
    (x₁ ≠ x₂) ∧
    (d = |x₁ - x₂|) ∧
    (d = 30)

-- The proof is omitted
theorem absolute_value_equation_solution_difference_is_30 :
  absolute_value_equation_solution_difference 30 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_30_l3232_323261


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3232_323231

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a ^ 3 + 14 * a ^ 2 + 2047 * a + 3024 = 0) →
  (9 * b ^ 3 + 14 * b ^ 2 + 2047 * b + 3024 = 0) →
  (9 * c ^ 3 + 14 * c ^ 2 + 2047 * c + 3024 = 0) →
  (a + b) ^ 3 + (b + c) ^ 3 + (c + a) ^ 3 = -58198 / 729 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3232_323231


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3232_323200

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x+3) = 0 -/
def f (x : ℝ) : ℝ := x * (x + 3)

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3232_323200


namespace NUMINAMATH_CALUDE_calculation_proof_l3232_323241

theorem calculation_proof : (1 / Real.sqrt 3) - (1 / 4)⁻¹ + 4 * Real.sin (60 * π / 180) + |1 - Real.sqrt 3| = (10 / 3) * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3232_323241


namespace NUMINAMATH_CALUDE_final_ethanol_percentage_l3232_323243

/-- Calculates the final ethanol percentage in a fuel mixture after adding pure ethanol -/
theorem final_ethanol_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 27)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : added_ethanol = 1.5)
  : (initial_volume * initial_ethanol_percentage + added_ethanol) / (initial_volume + added_ethanol) = 0.1 := by
  sorry

#check final_ethanol_percentage

end NUMINAMATH_CALUDE_final_ethanol_percentage_l3232_323243


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3232_323294

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_modulus_problem (z : ℂ) (h : (1 + i) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3232_323294


namespace NUMINAMATH_CALUDE_travelers_checks_theorem_l3232_323296

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for the travelers checks -/
def travelersProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The result of spending some $50 checks -/
def spendFiftyChecks (tc : TravelersChecks) (spent : ℕ) : TravelersChecks :=
  { fifty := tc.fifty - spent, hundred := tc.hundred }

/-- Calculate the average value of the remaining checks -/
def averageValue (tc : TravelersChecks) : ℚ :=
  (50 * tc.fifty + 100 * tc.hundred) / (tc.fifty + tc.hundred)

/-- The main theorem to prove -/
theorem travelers_checks_theorem (tc : TravelersChecks) :
  travelersProblem tc →
  averageValue (spendFiftyChecks tc 15) = 70 := by
  sorry


end NUMINAMATH_CALUDE_travelers_checks_theorem_l3232_323296


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3232_323226

def large_number : ℕ := 3 * 10^500 - 2022 * 10^497 - 2022

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_large_number : sum_of_digits large_number = 4491 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3232_323226


namespace NUMINAMATH_CALUDE_matts_future_age_l3232_323257

theorem matts_future_age (bush_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) :
  bush_age = 12 →
  age_difference = 3 →
  years_from_now = 10 →
  bush_age + age_difference + years_from_now = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_matts_future_age_l3232_323257


namespace NUMINAMATH_CALUDE_well_digging_time_l3232_323217

theorem well_digging_time 
  (combined_time : ℝ) 
  (paul_time : ℝ) 
  (hari_time : ℝ) 
  (h1 : combined_time = 8)
  (h2 : paul_time = 24)
  (h3 : hari_time = 48) : 
  ∃ jake_time : ℝ, 
    jake_time = 16 ∧ 
    1 / combined_time = 1 / jake_time + 1 / paul_time + 1 / hari_time :=
by sorry

end NUMINAMATH_CALUDE_well_digging_time_l3232_323217


namespace NUMINAMATH_CALUDE_congruence_problem_l3232_323274

theorem congruence_problem (x : ℤ) : (5 * x + 8) % 19 = 3 → (5 * x + 9) % 19 = 4 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3232_323274


namespace NUMINAMATH_CALUDE_average_female_students_l3232_323265

theorem average_female_students (class_8A class_8B class_8C class_8D class_8E : ℕ) 
  (h1 : class_8A = 10)
  (h2 : class_8B = 14)
  (h3 : class_8C = 7)
  (h4 : class_8D = 9)
  (h5 : class_8E = 13) : 
  (class_8A + class_8B + class_8C + class_8D + class_8E : ℚ) / 5 = 10.6 := by
  sorry

end NUMINAMATH_CALUDE_average_female_students_l3232_323265


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3232_323250

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 10 = 9) ∧ 
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 = 2 → m % 10 = 9 → m ≥ n) ∧
  (n = 59) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3232_323250


namespace NUMINAMATH_CALUDE_acute_angle_alpha_l3232_323275

theorem acute_angle_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
  α = π / 3.6 := by sorry

end NUMINAMATH_CALUDE_acute_angle_alpha_l3232_323275


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3232_323246

/-- The constant term in the expansion of (x^2 - 2/x)^6 is 240 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (x^2 - 2/x)^6
  ∃ c : ℝ, (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 240 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3232_323246


namespace NUMINAMATH_CALUDE_problem_solution_l3232_323242

def A : Set ℝ := {1, 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a = 0}

def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 3 > 0}

theorem problem_solution :
  (∀ a : ℝ, (∀ x ∈ B a, x ∈ A) ↔ a ∈ Set.Ioo 0 1) ∧
  (∀ m : ℝ, (A ⊆ C m) ↔ m ∈ Set.Iic (7/2)) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3232_323242


namespace NUMINAMATH_CALUDE_well_digging_cost_l3232_323252

/-- The cost of digging a cylindrical well -/
theorem well_digging_cost (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) :
  depth = 14 ∧ diameter = 3 ∧ cost_per_cubic_meter = 18 →
  ∃ (total_cost : ℝ), (abs (total_cost - 1782) < 1) ∧
  total_cost = (Real.pi * (diameter / 2)^2 * depth) * cost_per_cubic_meter :=
by sorry

end NUMINAMATH_CALUDE_well_digging_cost_l3232_323252


namespace NUMINAMATH_CALUDE_cross_spectral_density_symmetry_l3232_323295

/-- Cross-spectral density of two random functions -/
noncomputable def cross_spectral_density (X Y : ℝ → ℂ) (ω : ℝ) : ℂ := sorry

/-- Stationarity property for a random function -/
def stationary (X : ℝ → ℂ) : Prop := sorry

/-- Joint stationarity property for two random functions -/
def jointly_stationary (X Y : ℝ → ℂ) : Prop := sorry

/-- Theorem: For stationary and jointly stationary random functions, 
    the cross-spectral densities satisfy s_xy(-ω) = s_yx(ω) -/
theorem cross_spectral_density_symmetry 
  (X Y : ℝ → ℂ) (ω : ℝ) 
  (h1 : stationary X) (h2 : stationary Y) (h3 : jointly_stationary X Y) : 
  cross_spectral_density X Y (-ω) = cross_spectral_density Y X ω := by
  sorry

end NUMINAMATH_CALUDE_cross_spectral_density_symmetry_l3232_323295


namespace NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3232_323253

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_sqrt_seven_l3232_323253


namespace NUMINAMATH_CALUDE_prob_same_color_diff_foot_value_l3232_323210

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_diff_foot : ℚ :=
  (black_pairs * 2 * black_pairs) / (total_shoes * (total_shoes - 1)) +
  (brown_pairs * 2 * brown_pairs) / (total_shoes * (total_shoes - 1)) +
  (gray_pairs * 2 * gray_pairs) / (total_shoes * (total_shoes - 1))

theorem prob_same_color_diff_foot_value :
  prob_same_color_diff_foot = 89 / 435 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_diff_foot_value_l3232_323210


namespace NUMINAMATH_CALUDE_production_time_calculation_l3232_323262

/-- Given that 5 machines can produce 20 units in 10 hours, 
    prove that 10 machines will take 25 hours to produce 100 units. -/
theorem production_time_calculation 
  (machines_initial : ℕ) 
  (units_initial : ℕ) 
  (hours_initial : ℕ) 
  (machines_final : ℕ) 
  (units_final : ℕ) 
  (h1 : machines_initial = 5) 
  (h2 : units_initial = 20) 
  (h3 : hours_initial = 10) 
  (h4 : machines_final = 10) 
  (h5 : units_final = 100) : 
  (units_final : ℚ) * machines_initial * hours_initial / 
  (units_initial * machines_final) = 25 := by
  sorry


end NUMINAMATH_CALUDE_production_time_calculation_l3232_323262


namespace NUMINAMATH_CALUDE_chess_tournament_score_difference_l3232_323232

-- Define the number of players
def num_players : ℕ := 12

-- Define the scoring system
def win_points : ℚ := 1
def draw_points : ℚ := 1/2
def loss_points : ℚ := 0

-- Define the total number of games
def total_games : ℕ := num_players * (num_players - 1) / 2

-- Define Vasya's score (minimum possible given the conditions)
def vasya_score : ℚ := loss_points + (num_players - 2) * draw_points

-- Define the minimum score for other players to be higher than Vasya
def min_other_score : ℚ := vasya_score + 1/2

-- Define Petya's score (maximum possible)
def petya_score : ℚ := (num_players - 1) * win_points

-- Theorem statement
theorem chess_tournament_score_difference :
  petya_score - vasya_score = 1 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_difference_l3232_323232


namespace NUMINAMATH_CALUDE_julia_born_1979_l3232_323203

def wayne_age_2021 : ℕ := 37
def peter_age_diff : ℕ := 3
def julia_age_diff : ℕ := 2

def julia_birth_year : ℕ := 2021 - wayne_age_2021 - peter_age_diff - julia_age_diff

theorem julia_born_1979 : julia_birth_year = 1979 := by
  sorry

end NUMINAMATH_CALUDE_julia_born_1979_l3232_323203


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l3232_323260

theorem square_circle_area_ratio (r : ℝ) (h : r > 0) :
  let s := 2 * r
  (s^2) / (π * r^2) = 4 / π := by
sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l3232_323260


namespace NUMINAMATH_CALUDE_perimeter_area_ratio_not_always_equal_l3232_323279

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  leg : ℝ
  perimeter : ℝ
  area : ℝ

/-- The theorem states that the ratio of perimeters is not always equal to the ratio of areas for two different isosceles triangles -/
theorem perimeter_area_ratio_not_always_equal
  (triangle1 triangle2 : IsoscelesTriangle)
  (h_base_neq : triangle1.base ≠ triangle2.base)
  (h_leg_neq : triangle1.leg ≠ triangle2.leg) :
  ¬ ∀ (triangle1 triangle2 : IsoscelesTriangle),
    triangle1.perimeter / triangle2.perimeter = triangle1.area / triangle2.area :=
by sorry

end NUMINAMATH_CALUDE_perimeter_area_ratio_not_always_equal_l3232_323279


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3232_323277

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, (3 + 5 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3232_323277


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l3232_323291

theorem quadratic_roots_sum_squares_minimum (m : ℝ) :
  let a : ℝ := 6
  let b : ℝ := -8
  let c : ℝ := m
  let discriminant := b^2 - 4*a*c
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2*product_of_roots
  discriminant > 0 →
  (∀ m' : ℝ, discriminant > 0 → sum_of_squares ≤ ((-b/a)^2 - 2*(m'/a))) →
  m = 8/3 ∧ sum_of_squares = 8/9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_minimum_l3232_323291


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_3_dividing_27_factorial_l3232_323263

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + (x + 1) / 3) 0

def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_3_dividing_27_factorial :
  onesDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_3_dividing_27_factorial_l3232_323263


namespace NUMINAMATH_CALUDE_unique_three_digit_number_with_digit_property_l3232_323269

/-- Calculate the total number of digits used to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that a number, when doubled, equals the total digits required to write all numbers up to itself -/
def hasDigitProperty (n : ℕ) : Prop :=
  2 * n = totalDigits n

theorem unique_three_digit_number_with_digit_property :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ hasDigitProperty n ∧ n = 108 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_with_digit_property_l3232_323269


namespace NUMINAMATH_CALUDE_abc_inequality_l3232_323271

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eq_a : a = 2022 * Real.exp (a - 2022))
  (eq_b : b = 2023 * Real.exp (b - 2023))
  (eq_c : c = 2024 * Real.exp (c - 2024)) :
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3232_323271


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l3232_323221

theorem hamburgers_left_over (hamburgers_made : ℕ) (hamburgers_served : ℕ) : 
  hamburgers_made = 15 → hamburgers_served = 8 → hamburgers_made - hamburgers_served = 7 := by
  sorry

#check hamburgers_left_over

end NUMINAMATH_CALUDE_hamburgers_left_over_l3232_323221


namespace NUMINAMATH_CALUDE_inequality_range_l3232_323212

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - m * x - 2 < 0) ↔ -8 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l3232_323212


namespace NUMINAMATH_CALUDE_ariana_flowers_l3232_323283

theorem ariana_flowers (total : ℕ) 
  (h1 : 2 * total = 5 * (total - 10 - 14)) -- 2/5 of flowers were roses
  (h2 : 10 ≤ total) -- 10 flowers were tulips
  (h3 : 14 ≤ total - 10) -- 14 flowers were carnations
  : total = 40 := by sorry

end NUMINAMATH_CALUDE_ariana_flowers_l3232_323283


namespace NUMINAMATH_CALUDE_f_neg_two_equals_nineteen_l3232_323225

/-- Given a function f(x) = 2x^2 - 4x + 3, prove that f(-2) = 19 -/
theorem f_neg_two_equals_nineteen : 
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 3
  f (-2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_nineteen_l3232_323225


namespace NUMINAMATH_CALUDE_altitude_equation_median_equation_l3232_323251

/-- Triangle ABC with vertices A(-2,-1), B(2,1), and C(1,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle given in the problem -/
def given_triangle : Triangle :=
  { A := (-2, -1),
    B := (2, 1),
    C := (1, 3) }

/-- Equation of a line in point-slope form -/
structure PointSlopeLine :=
  (m : ℝ)  -- slope
  (x₀ : ℝ) -- x-coordinate of point
  (y₀ : ℝ) -- y-coordinate of point

/-- Equation of a line in general form -/
structure GeneralLine :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The altitude from side AB of the triangle -/
def altitude (t : Triangle) : PointSlopeLine :=
  { m := -2,
    x₀ := 1,
    y₀ := 3 }

/-- The median from side AB of the triangle -/
def median (t : Triangle) : GeneralLine :=
  { a := 3,
    b := -1,
    c := 0 }

theorem altitude_equation (t : Triangle) :
  t = given_triangle →
  altitude t = { m := -2, x₀ := 1, y₀ := 3 } :=
by sorry

theorem median_equation (t : Triangle) :
  t = given_triangle →
  median t = { a := 3, b := -1, c := 0 } :=
by sorry

end NUMINAMATH_CALUDE_altitude_equation_median_equation_l3232_323251


namespace NUMINAMATH_CALUDE_return_trip_speed_l3232_323219

/-- Given a round trip between two cities, prove the speed of the return trip -/
theorem return_trip_speed 
  (distance : ℝ) 
  (outbound_speed : ℝ) 
  (average_speed : ℝ) :
  distance = 150 →
  outbound_speed = 75 →
  average_speed = 50 →
  (2 * distance) / (distance / outbound_speed + distance / ((2 * distance) / (2 * average_speed) - distance / outbound_speed)) = average_speed →
  (2 * distance) / (2 * average_speed) - distance / outbound_speed = distance / 37.5 :=
by sorry

end NUMINAMATH_CALUDE_return_trip_speed_l3232_323219


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3232_323204

theorem sum_with_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3232_323204


namespace NUMINAMATH_CALUDE_largest_solution_l3232_323290

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The equation from the problem -/
def equation (x : ℝ) : Prop := floor x = 6 + 50 * frac x

/-- The theorem stating the largest solution -/
theorem largest_solution :
  ∃ (x : ℝ), equation x ∧ ∀ (y : ℝ), equation y → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_solution_l3232_323290


namespace NUMINAMATH_CALUDE_line_up_permutations_l3232_323229

def number_of_people : ℕ := 5

theorem line_up_permutations :
  let youngest_not_first := number_of_people - 1
  let eldest_not_last := number_of_people - 1
  let remaining_positions := number_of_people - 2
  youngest_not_first * eldest_not_last * (remaining_positions.factorial) = 96 :=
by sorry

end NUMINAMATH_CALUDE_line_up_permutations_l3232_323229


namespace NUMINAMATH_CALUDE_system_solution_l3232_323238

theorem system_solution (a b c x y z T : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  x = Real.sqrt (y^2 - a^2) + Real.sqrt (z^2 - a^2) →
  y = Real.sqrt (z^2 - b^2) + Real.sqrt (x^2 - b^2) →
  z = Real.sqrt (x^2 - c^2) + Real.sqrt (y^2 - c^2) →
  1 / T^2 = 2 / (a^2 * b^2) + 2 / (b^2 * c^2) + 2 / (c^2 * a^2) - 1 / a^4 - 1 / b^4 - 1 / c^4 →
  1 / T^2 > 0 →
  x = 2 * T / a ∧ y = 2 * T / b ∧ z = 2 * T / c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3232_323238


namespace NUMINAMATH_CALUDE_g_solution_set_a_range_l3232_323201

-- Define the functions f and g
def f (a x : ℝ) := 3 * abs (x - a) + abs (3 * x + 1)
def g (x : ℝ) := abs (4 * x - 1) - abs (x + 2)

-- Theorem for the solution set of g(x) < 6
theorem g_solution_set :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} := by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → a ∈ Set.Icc (-13/12) (5/12) := by sorry

end NUMINAMATH_CALUDE_g_solution_set_a_range_l3232_323201


namespace NUMINAMATH_CALUDE_first_machine_copies_per_minute_l3232_323227

/-- Given two copy machines working together, prove that the first machine makes 25 copies per minute. -/
theorem first_machine_copies_per_minute :
  ∀ (x : ℝ),
  (∃ (rate₁ : ℝ), rate₁ = x) →  -- First machine works at a constant rate x
  (∃ (rate₂ : ℝ), rate₂ = 55) →  -- Second machine works at 55 copies per minute
  (x + 55) * 30 = 2400 →  -- Together they make 2400 copies in 30 minutes
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_first_machine_copies_per_minute_l3232_323227


namespace NUMINAMATH_CALUDE_original_cube_volume_l3232_323287

theorem original_cube_volume (s : ℝ) (h : (2 * s) ^ 3 = 1728) : s ^ 3 = 216 := by
  sorry

#check original_cube_volume

end NUMINAMATH_CALUDE_original_cube_volume_l3232_323287


namespace NUMINAMATH_CALUDE_product_divisible_by_60_l3232_323286

theorem product_divisible_by_60 (a : ℤ) : 
  60 ∣ (a^2 - 1) * a^2 * (a^2 + 1) := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_60_l3232_323286


namespace NUMINAMATH_CALUDE_area_of_composite_rectangle_l3232_323214

/-- The area of a rectangle formed by four identical smaller rectangles --/
theorem area_of_composite_rectangle (short_side : ℝ) : 
  short_side = 7 →
  (2 * short_side) * (2 * short_side) = 392 := by
  sorry

end NUMINAMATH_CALUDE_area_of_composite_rectangle_l3232_323214


namespace NUMINAMATH_CALUDE_fibFactorial_characterization_l3232_323209

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Set of positive integers n for which n! is the product of two Fibonacci numbers -/
def fibFactorialSet : Set ℕ :=
  {n : ℕ | n > 0 ∧ ∃ k m : ℕ, n.factorial = fib k * fib m}

/-- Theorem stating that fibFactorialSet contains exactly 1, 2, 3, 4, and 6 -/
theorem fibFactorial_characterization :
    fibFactorialSet = {1, 2, 3, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_fibFactorial_characterization_l3232_323209


namespace NUMINAMATH_CALUDE_trail_mix_pouches_per_pack_l3232_323259

theorem trail_mix_pouches_per_pack 
  (team_members : ℕ) 
  (coaches : ℕ) 
  (helpers : ℕ) 
  (total_packs : ℕ) 
  (h1 : team_members = 13)
  (h2 : coaches = 3)
  (h3 : helpers = 2)
  (h4 : total_packs = 3)
  : (team_members + coaches + helpers) / total_packs = 6 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_pouches_per_pack_l3232_323259


namespace NUMINAMATH_CALUDE_platform_length_l3232_323224

-- Define the train's properties
variable (l : ℝ) -- length of the train
variable (t : ℝ) -- time to pass a pole
variable (v : ℝ) -- velocity of the train

-- Define the platform
variable (p : ℝ) -- length of the platform

-- State the theorem
theorem platform_length 
  (h1 : v = l / t) -- velocity when passing the pole
  (h2 : v = (l + p) / (5 * t)) -- velocity when passing the platform
  : p = 4 * l := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l3232_323224


namespace NUMINAMATH_CALUDE_chocolates_remaining_theorem_l3232_323247

/-- Number of chocolates remaining after 4 days -/
def chocolates_remaining (total : ℕ) (day1 : ℕ) : ℕ :=
  let day2 := 2 * day1 - 3
  let day3 := day1 - 2
  let day4 := day3 - 1
  total - (day1 + day2 + day3 + day4)

/-- Theorem stating that 12 chocolates remain uneaten after 4 days -/
theorem chocolates_remaining_theorem :
  chocolates_remaining 24 4 = 12 := by
  sorry

#eval chocolates_remaining 24 4

end NUMINAMATH_CALUDE_chocolates_remaining_theorem_l3232_323247


namespace NUMINAMATH_CALUDE_guaranteed_win_for_given_odds_l3232_323285

/-- Represents the odds for a team as a pair of natural numbers -/
def Odds := Nat × Nat

/-- Calculates the return multiplier for given odds -/
def returnMultiplier (odds : Odds) : Rat :=
  1 + odds.2 / odds.1

/-- Represents the odds for all teams in the tournament -/
structure TournamentOdds where
  team1 : Odds
  team2 : Odds
  team3 : Odds
  team4 : Odds

/-- Checks if a betting strategy exists that guarantees a win -/
def guaranteedWinExists (odds : TournamentOdds) : Prop :=
  ∃ (bet1 bet2 bet3 bet4 : Rat),
    bet1 > 0 ∧ bet2 > 0 ∧ bet3 > 0 ∧ bet4 > 0 ∧
    bet1 + bet2 + bet3 + bet4 = 1 ∧
    bet1 * returnMultiplier odds.team1 > 1 ∧
    bet2 * returnMultiplier odds.team2 > 1 ∧
    bet3 * returnMultiplier odds.team3 > 1 ∧
    bet4 * returnMultiplier odds.team4 > 1

/-- The main theorem stating that a guaranteed win exists for the given odds -/
theorem guaranteed_win_for_given_odds :
  let odds : TournamentOdds := {
    team1 := (1, 5)
    team2 := (1, 1)
    team3 := (1, 8)
    team4 := (1, 7)
  }
  guaranteedWinExists odds := by sorry

end NUMINAMATH_CALUDE_guaranteed_win_for_given_odds_l3232_323285


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_21_ending_in_3_l3232_323240

theorem three_digit_divisible_by_21_ending_in_3 :
  ∃! (s : Finset Nat), 
    s.card = 3 ∧
    (∀ n ∈ s, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0) ∧
    (∀ n, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0 → n ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_21_ending_in_3_l3232_323240


namespace NUMINAMATH_CALUDE_triangle_side_length_l3232_323255

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b - c = 2 →
  Real.cos B = -1/2 →
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3232_323255


namespace NUMINAMATH_CALUDE_circle_range_theorem_l3232_323215

/-- The range of 'a' for a circle (x-a)^2 + (y-a)^2 = 8 with a point at distance √2 from origin -/
theorem circle_range_theorem (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a ∈ Set.Icc (-3) (-1) ∪ Set.Icc 1 3) :=
sorry

end NUMINAMATH_CALUDE_circle_range_theorem_l3232_323215


namespace NUMINAMATH_CALUDE_sum_and_fraction_relation_l3232_323288

theorem sum_and_fraction_relation (a b : ℝ) 
  (sum_eq : a + b = 507)
  (frac_eq : (a - b) / b = 1 / 7) : 
  b - a = -34.428571 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_fraction_relation_l3232_323288


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3232_323205

theorem at_least_one_greater_than_one (a b c : ℝ) : 
  (a - 1) * (b - 1) * (c - 1) > 0 → (a > 1 ∨ b > 1 ∨ c > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3232_323205


namespace NUMINAMATH_CALUDE_expression_evaluation_l3232_323230

theorem expression_evaluation : 2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3232_323230


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3232_323216

/-- The number of cards --/
def n : ℕ := 7

/-- The special card that must be at the beginning or end --/
def special_card : ℕ := 7

/-- The number of cards that will remain after removal --/
def remaining_cards : ℕ := 5

/-- The number of possible positions for the special card --/
def special_card_positions : ℕ := 2

/-- The number of ways to choose a card to remove from the non-special cards --/
def removal_choices : ℕ := n - 1

/-- The number of permutations of the remaining cards --/
def remaining_permutations : ℕ := remaining_cards.factorial

/-- The number of possible orderings (ascending or descending) --/
def possible_orderings : ℕ := 2

/-- The total number of valid arrangements --/
def valid_arrangements : ℕ := 
  special_card_positions * removal_choices * remaining_permutations * possible_orderings

theorem valid_arrangements_count : valid_arrangements = 2880 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3232_323216


namespace NUMINAMATH_CALUDE_total_students_in_high_school_l3232_323293

-- Define the number of students in each grade
def freshman_students : ℕ := sorry
def sophomore_students : ℕ := sorry
def senior_students : ℕ := 1200

-- Define the sample sizes
def freshman_sample : ℕ := 75
def sophomore_sample : ℕ := 60
def senior_sample : ℕ := 50

-- Define the total sample size
def total_sample : ℕ := 185

-- Theorem statement
theorem total_students_in_high_school :
  freshman_students + sophomore_students + senior_students = 4440 :=
by
  -- Assuming the stratified sampling method ensures equal ratios
  have h1 : (freshman_sample : ℚ) / freshman_students = (senior_sample : ℚ) / senior_students := sorry
  have h2 : (sophomore_sample : ℚ) / sophomore_students = (senior_sample : ℚ) / senior_students := sorry
  
  -- The total sample size is the sum of individual sample sizes
  have h3 : freshman_sample + sophomore_sample + senior_sample = total_sample := sorry

  sorry -- Complete the proof

end NUMINAMATH_CALUDE_total_students_in_high_school_l3232_323293


namespace NUMINAMATH_CALUDE_isabellas_haircut_l3232_323206

/-- Isabella's haircut problem -/
theorem isabellas_haircut (original_length cut_length : ℕ) (h1 : original_length = 18) (h2 : cut_length = 9) :
  original_length - cut_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircut_l3232_323206


namespace NUMINAMATH_CALUDE_triangle_perimeter_increase_l3232_323213

/-- Given an initial equilateral triangle and four subsequent triangles with increasing side lengths,
    calculate the percent increase in perimeter from the first to the fifth triangle. -/
theorem triangle_perimeter_increase (initial_side : ℝ) (scale_factor : ℝ) (num_triangles : ℕ) :
  initial_side = 3 →
  scale_factor = 2 →
  num_triangles = 5 →
  let first_perimeter := 3 * initial_side
  let last_side := initial_side * scale_factor ^ (num_triangles - 1)
  let last_perimeter := 3 * last_side
  (last_perimeter - first_perimeter) / first_perimeter * 100 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_increase_l3232_323213


namespace NUMINAMATH_CALUDE_negative_rational_identification_l3232_323266

theorem negative_rational_identification :
  let a := -(-2010)
  let b := -|-2010|
  let c := (-2011)^2010
  let d := -2010 / -2011
  (¬ (a < 0 ∧ ∃ (p q : ℤ), a = p / q ∧ q ≠ 0)) ∧
  (b < 0 ∧ ∃ (p q : ℤ), b = p / q ∧ q ≠ 0) ∧
  (¬ (c < 0 ∧ ∃ (p q : ℤ), c = p / q ∧ q ≠ 0)) ∧
  (¬ (d < 0 ∧ ∃ (p q : ℤ), d = p / q ∧ q ≠ 0)) :=
by sorry


end NUMINAMATH_CALUDE_negative_rational_identification_l3232_323266


namespace NUMINAMATH_CALUDE_yolanda_departure_time_yolanda_left_30_minutes_before_l3232_323207

/-- Prove that Yolanda left 30 minutes before her husband caught up to her. -/
theorem yolanda_departure_time 
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)  -- Convert 15 minutes to hours
  (h4 : catch_up_time = 15 / 60)  -- Convert 15 minutes to hours
  : yolanda_speed * (husband_delay + catch_up_time) = husband_speed * catch_up_time :=
by sorry

/-- Yolanda's departure time before being caught -/
def yolanda_departure_before_catch (yolanda_speed : ℝ) (husband_speed : ℝ) (husband_delay : ℝ) (catch_up_time : ℝ) : ℝ :=
  husband_delay + catch_up_time

/-- Prove that Yolanda left 30 minutes (0.5 hours) before her husband caught up to her -/
theorem yolanda_left_30_minutes_before
  (yolanda_speed : ℝ) 
  (husband_speed : ℝ) 
  (husband_delay : ℝ) 
  (catch_up_time : ℝ) 
  (h1 : yolanda_speed = 20)
  (h2 : husband_speed = 40)
  (h3 : husband_delay = 15 / 60)
  (h4 : catch_up_time = 15 / 60)
  : yolanda_departure_before_catch yolanda_speed husband_speed husband_delay catch_up_time = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_yolanda_departure_time_yolanda_left_30_minutes_before_l3232_323207


namespace NUMINAMATH_CALUDE_progression_ratio_l3232_323281

/-- Given an arithmetic progression and a geometric progression with shared elements,
    prove that the ratio of the difference of middle terms of the arithmetic progression
    to the middle term of the geometric progression is either 1/2 or -1/2. -/
theorem progression_ratio (a₁ a₂ b : ℝ) : 
  ((-2 : ℝ) - a₁ = a₁ - a₂ ∧ a₂ - (-8) = a₁ - a₂) →  -- arithmetic progression condition
  (b^2 = (-2) * (-8)) →                              -- geometric progression condition
  (a₂ - a₁) / b = 1/2 ∨ (a₂ - a₁) / b = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_progression_ratio_l3232_323281


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3232_323258

-- Define the displacement function
def s (t : ℝ) : ℝ := 100 * t - 5 * t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 100 - 10 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 :
  ∀ t : ℝ, 0 < t → t < 20 → v 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l3232_323258


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3232_323284

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Proves that the total selling price of 45 meters of cloth is 4500 rupees,
    given a profit of 12 rupees per meter and a cost price of 88 rupees per meter. -/
theorem cloth_selling_price :
  total_selling_price 45 12 88 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3232_323284


namespace NUMINAMATH_CALUDE_tan_product_pi_eighths_l3232_323268

theorem tan_product_pi_eighths : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_eighths_l3232_323268


namespace NUMINAMATH_CALUDE_metallic_sheet_dimension_l3232_323245

/-- Given a rectangular metallic sheet with one dimension of 52 meters,
    if squares of 8 meters are cut from each corner to form an open box
    with a volume of 5760 cubic meters, then the length of the second
    dimension of the metallic sheet is 36 meters. -/
theorem metallic_sheet_dimension (w : ℝ) :
  w > 0 →
  (w - 2 * 8) * (52 - 2 * 8) * 8 = 5760 →
  w = 36 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_dimension_l3232_323245


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3232_323211

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let parabola (x y : ℝ) := y^2 = 20*x
  let hyperbola (x y : ℝ) := x^2/a^2 - y^2/b^2 = 1
  let focus_parabola : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) := b*x + a*y = 0
  let distance_focus_asymptote := 4
  let eccentricity := (Real.sqrt (a^2 + b^2)) / a
  (∀ x y, parabola x y → hyperbola x y) →
  (distance_focus_asymptote = 4) →
  eccentricity = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3232_323211


namespace NUMINAMATH_CALUDE_anna_age_when_married_l3232_323276

/-- Represents the ages and marriage duration of Josh and Anna -/
structure Couple where
  josh_age_at_marriage : ℕ
  years_married : ℕ
  combined_age_factor : ℕ

/-- Calculates Anna's age when they got married -/
def anna_age_at_marriage (c : Couple) : ℕ :=
  c.combined_age_factor * c.josh_age_at_marriage - (c.josh_age_at_marriage + c.years_married)

/-- Theorem stating Anna's age when they got married -/
theorem anna_age_when_married (c : Couple) 
    (h1 : c.josh_age_at_marriage = 22)
    (h2 : c.years_married = 30)
    (h3 : c.combined_age_factor = 5) :
  anna_age_at_marriage c = 28 := by
  sorry

#eval anna_age_at_marriage ⟨22, 30, 5⟩

end NUMINAMATH_CALUDE_anna_age_when_married_l3232_323276


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_algebraic_expression_equality_l3232_323249

-- Part 1
theorem logarithm_expression_equality : 
  Real.log 5 * Real.log 20 - Real.log 2 * Real.log 50 - Real.log 25 = -1 := by sorry

-- Part 2
theorem algebraic_expression_equality (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_algebraic_expression_equality_l3232_323249


namespace NUMINAMATH_CALUDE_seashell_sale_theorem_l3232_323218

/-- Calculates the total money earned from selling items collected over two days -/
def total_money (day1_items : ℕ) (price_per_item : ℚ) : ℚ :=
  let day2_items := day1_items / 2
  let total_items := day1_items + day2_items
  total_items * price_per_item

/-- Proves that collecting 30 items on day 1, half as many on day 2, 
    and selling each for $1.20 results in $54 total -/
theorem seashell_sale_theorem :
  total_money 30 (6/5) = 54 := by
  sorry

end NUMINAMATH_CALUDE_seashell_sale_theorem_l3232_323218


namespace NUMINAMATH_CALUDE_parabola_properties_l3232_323208

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the point M on the parabola
def point_M (p x y : ℝ) : Prop := parabola p x y ∧ x + 2 = x + p/2

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define a line passing through the focus
def line_through_focus (p m : ℝ) (x y : ℝ) : Prop := x = m*y + p/2

-- Define the intersection points of the line and the parabola
def intersection_points (p m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_through_focus p m x₁ y₁ ∧ parabola p x₁ y₁ ∧
  line_through_focus p m x₂ y₂ ∧ parabola p x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

theorem parabola_properties (p : ℝ) :
  (∃ x y : ℝ, point_M p x y) →
  (p = 4 ∧
   ∀ m x₁ y₁ x₂ y₂ : ℝ, intersection_points p m x₁ y₁ x₂ y₂ → y₁ * y₂ = -16) :=
by sorry

end

end NUMINAMATH_CALUDE_parabola_properties_l3232_323208


namespace NUMINAMATH_CALUDE_linda_savings_l3232_323220

theorem linda_savings (savings : ℝ) : 
  savings > 0 →
  savings * (3/4) + savings * (1/8) + 250 = savings * (7/8) →
  250 = (savings * (1/8)) * 0.9 →
  savings = 2222.24 := by
sorry

end NUMINAMATH_CALUDE_linda_savings_l3232_323220


namespace NUMINAMATH_CALUDE_factor_expression_l3232_323254

theorem factor_expression (a : ℝ) : 189 * a^2 + 27 * a - 54 = 9 * (7 * a - 3) * (3 * a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3232_323254


namespace NUMINAMATH_CALUDE_church_female_adults_l3232_323236

/-- Calculates the number of female adults in a church given the total number of people,
    number of children, and number of male adults. -/
def female_adults (total : ℕ) (children : ℕ) (male_adults : ℕ) : ℕ :=
  total - (children + male_adults)

/-- Theorem stating that the number of female adults in the church is 60. -/
theorem church_female_adults :
  female_adults 200 80 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_church_female_adults_l3232_323236


namespace NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l3232_323244

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of elements in T -/
def n : ℕ := Finset.card T

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diag : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diag : ℕ := 3

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diag * (num_short_diag - 1) + num_long_diag * (num_long_diag - 1)) /
  (n * (n - 1))

theorem prob_same_length_regular_hexagon :
  prob_same_length = 22 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_regular_hexagon_l3232_323244


namespace NUMINAMATH_CALUDE_married_women_fraction_l3232_323256

theorem married_women_fraction (total_men : ℕ) (total_women : ℕ) (single_men : ℕ) :
  (single_men : ℚ) / total_men = 3 / 7 →
  total_women = total_men - single_men →
  (total_women : ℚ) / (total_men + total_women) = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_married_women_fraction_l3232_323256


namespace NUMINAMATH_CALUDE_greatest_possible_award_l3232_323223

theorem greatest_possible_award (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ) 
  (h1 : total_prize = 800)
  (h2 : num_winners = 20)
  (h3 : min_award = 20)
  (h4 : (2 : ℚ) / 5 * total_prize = (3 : ℚ) / 5 * num_winners * min_award) :
  ∃ (max_award : ℕ), max_award = 420 ∧ 
    (∀ (award : ℕ), award > max_award → 
      ¬(∃ (awards : List ℕ), awards.length = num_winners ∧ 
        awards.sum = total_prize ∧ 
        (∀ x ∈ awards, x ≥ min_award) ∧
        award ∈ awards)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_award_l3232_323223


namespace NUMINAMATH_CALUDE_home_to_school_distance_proof_l3232_323267

/-- The distance from Xiao Hong's home to her school -/
def home_to_school_distance : ℝ := 12000

/-- The distance the father drives Xiao Hong -/
def father_driving_distance : ℝ := 1000

/-- The time it takes Xiao Hong to get from home to school by car and walking -/
def car_and_walking_time : ℝ := 22.5

/-- The time it takes Xiao Hong to ride her bike from home to school -/
def bike_riding_time : ℝ := 40

/-- Xiao Hong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between father's driving speed and Xiao Hong's bike speed -/
def speed_difference : ℝ := 800

theorem home_to_school_distance_proof :
  home_to_school_distance = 12000 :=
sorry

end NUMINAMATH_CALUDE_home_to_school_distance_proof_l3232_323267


namespace NUMINAMATH_CALUDE_crackers_distribution_l3232_323282

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 45 → num_friends = 15 → crackers_per_friend = total_crackers / num_friends → 
  crackers_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3232_323282


namespace NUMINAMATH_CALUDE_shellys_total_money_l3232_323234

/-- Calculates the total amount of money Shelly has given her bill and coin counts. -/
def shellys_money (ten_dollar_bills : ℕ) : ℕ :=
  let five_dollar_bills := ten_dollar_bills - 12
  let twenty_dollar_bills := ten_dollar_bills / 2
  let one_dollar_coins := five_dollar_bills * 2
  10 * ten_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bills + one_dollar_coins

/-- Proves that Shelly has $726 given the conditions in the problem. -/
theorem shellys_total_money : shellys_money 30 = 726 := by
  sorry

end NUMINAMATH_CALUDE_shellys_total_money_l3232_323234


namespace NUMINAMATH_CALUDE_orange_calories_l3232_323298

theorem orange_calories
  (num_oranges : ℕ)
  (pieces_per_orange : ℕ)
  (num_people : ℕ)
  (calories_per_person : ℕ)
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_person = 100)
  : calories_per_person = num_oranges * calories_per_person / num_oranges :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calories_l3232_323298


namespace NUMINAMATH_CALUDE_luke_paint_area_l3232_323239

/-- Calculates the area to be painted on a wall with a bookshelf -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Proves that Luke needs to paint 135 square feet -/
theorem luke_paint_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_luke_paint_area_l3232_323239


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3232_323222

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem parabola_line_intersection :
  ∀ (x y : ℝ),
  parabola x y →
  line x y →
  first_quadrant x y →
  Real.sqrt ((x - 1)^2 + y^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3232_323222


namespace NUMINAMATH_CALUDE_unique_digit_factorial_sum_l3232_323202

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  factorial d1 + factorial d2 + factorial d3

def has_zero_digit (n : ℕ) : Prop :=
  n % 10 = 0 ∨ (n / 10) % 10 = 0 ∨ n / 100 = 0

theorem unique_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = digit_factorial_sum n ∧ has_zero_digit n :=
sorry

end NUMINAMATH_CALUDE_unique_digit_factorial_sum_l3232_323202


namespace NUMINAMATH_CALUDE_sandy_marbles_multiple_l3232_323299

def melanie_marbles : ℕ := 84
def sandy_dozens : ℕ := 56

def marbles_in_dozen : ℕ := 12

theorem sandy_marbles_multiple : 
  (sandy_dozens * marbles_in_dozen) / melanie_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_multiple_l3232_323299


namespace NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3232_323278

theorem max_n_for_consecutive_product : ∃ (n_max : ℕ), ∀ (n : ℕ), 
  (∃ (k : ℕ), 9*n^2 + 5*n + 26 = k * (k+1)) → n ≤ n_max :=
sorry

end NUMINAMATH_CALUDE_max_n_for_consecutive_product_l3232_323278


namespace NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l3232_323264

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the die -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die :
  expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l3232_323264


namespace NUMINAMATH_CALUDE_sin_plus_cos_from_double_angle_l3232_323289

theorem sin_plus_cos_from_double_angle (A : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.sin (2 * A) = 2 / 3) :
  Real.sin A + Real.cos A = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_from_double_angle_l3232_323289


namespace NUMINAMATH_CALUDE_equation_solution_l3232_323273

theorem equation_solution :
  ∃ x : ℚ, x - 1 ≠ 0 ∧ 1 - 1 / (x - 1) = 2 * x / (1 - x) ∧ x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3232_323273


namespace NUMINAMATH_CALUDE_polynomial_differential_equation_l3232_323280

/-- A polynomial of the form a(x + b)^n satisfies (p'(x))^2 = c * p(x) * p''(x) for some constant c -/
theorem polynomial_differential_equation (a b : ℝ) (n : ℕ) (hn : n > 1) (ha : a ≠ 0) :
  ∃ c : ℝ, ∀ x : ℝ,
    let p := fun x => a * (x + b) ^ n
    let p' := fun x => n * a * (x + b) ^ (n - 1)
    let p'' := fun x => n * (n - 1) * a * (x + b) ^ (n - 2)
    (p' x) ^ 2 = c * (p x) * (p'' x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_differential_equation_l3232_323280


namespace NUMINAMATH_CALUDE_regression_properties_l3232_323297

/-- A dataset of two variables -/
structure Dataset where
  x : List ℝ
  y : List ℝ

/-- Properties of a linear regression model -/
structure RegressionModel (d : Dataset) where
  x_mean : ℝ
  y_mean : ℝ
  r : ℝ
  b_hat : ℝ
  a_hat : ℝ

/-- The regression line passes through the mean point -/
def passes_through_mean (m : RegressionModel d) : Prop :=
  m.y_mean = m.b_hat * m.x_mean + m.a_hat

/-- Strong correlation between variables -/
def strong_correlation (m : RegressionModel d) : Prop :=
  abs m.r > 0.75

/-- Negative slope of the regression line -/
def negative_slope (m : RegressionModel d) : Prop :=
  m.b_hat < 0

/-- Main theorem -/
theorem regression_properties (d : Dataset) (m : RegressionModel d)
  (h1 : m.r = -0.8) :
  passes_through_mean m ∧ strong_correlation m ∧ negative_slope m := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l3232_323297


namespace NUMINAMATH_CALUDE_weight_sum_proof_l3232_323270

/-- Given the weights of four people and their pairwise sums, 
    prove that the sum of the weights of the first and last person is 295 pounds. -/
theorem weight_sum_proof (a b c d : ℝ) 
  (h1 : a + b = 270)
  (h2 : b + c = 255)
  (h3 : c + d = 280)
  (h4 : a + b + c + d = 480) :
  a + d = 295 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_proof_l3232_323270


namespace NUMINAMATH_CALUDE_sixth_quiz_score_achieves_target_mean_l3232_323235

def quiz_scores : List ℕ := [75, 80, 85, 90, 100]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6
def sixth_score : ℕ := 140

theorem sixth_quiz_score_achieves_target_mean :
  (List.sum quiz_scores + sixth_score) / num_quizzes = target_mean := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_achieves_target_mean_l3232_323235


namespace NUMINAMATH_CALUDE_g_seven_value_l3232_323228

theorem g_seven_value (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) 
  (h2 : g 6 = 7) : 
  g 7 = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_g_seven_value_l3232_323228


namespace NUMINAMATH_CALUDE_min_max_quadratic_form_l3232_323292

theorem min_max_quadratic_form (x y : ℝ) (h : 2 * x^2 + 3 * x * y + y^2 = 2) :
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≥ 4) ∧
  (∀ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 → 4 * a^2 + 4 * a * b + 3 * b^2 ≤ 6) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 4) ∧
  (∃ a b : ℝ, 2 * a^2 + 3 * a * b + b^2 = 2 ∧ 4 * a^2 + 4 * a * b + 3 * b^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_max_quadratic_form_l3232_323292


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l3232_323248

-- Define the function that returns the units digit of 7^n
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

theorem units_digit_of_7_pow_2050 :
  units_digit_of_7_pow 2050 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_2050_l3232_323248


namespace NUMINAMATH_CALUDE_simplification_proof_equation_solution_proof_l3232_323237

-- Problem 1: Simplification
theorem simplification_proof (a : ℝ) (ha : a ≠ 0 ∧ a ≠ 1) :
  (a - 1/a) / ((a^2 - 2*a + 1) / a) = (a + 1) / (a - 1) := by sorry

-- Problem 2: Equation Solving
theorem equation_solution_proof :
  ∀ x : ℝ, x = -1 ↔ 2*x/(x-2) = 1 - 1/(2-x) := by sorry

end NUMINAMATH_CALUDE_simplification_proof_equation_solution_proof_l3232_323237


namespace NUMINAMATH_CALUDE_max_intersection_area_l3232_323272

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

theorem max_intersection_area :
  ∀ (r1 r2 : Rectangle),
    r1.height < r1.width →
    r2.height > r2.width →
    r1.area = 2015 →
    r2.area = 2016 →
    (∀ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height →
      r.area ≤ 1302) ∧
    (∃ r : Rectangle,
      r.width ≤ min r1.width r2.width ∧
      r.height ≤ min r1.height r2.height ∧
      r.area = 1302) := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_l3232_323272
