import Mathlib

namespace NUMINAMATH_CALUDE_towel_folding_theorem_l3451_345155

-- Define the folding rates for each person
def jane_rate : ℚ := 5 / 5
def kyla_rate : ℚ := 9 / 10
def anthony_rate : ℚ := 14 / 20
def david_rate : ℚ := 6 / 15

-- Define the total number of towels folded in one hour
def total_towels : ℕ := 180

-- Theorem statement
theorem towel_folding_theorem :
  (jane_rate + kyla_rate + anthony_rate + david_rate) * 60 = total_towels := by
  sorry

end NUMINAMATH_CALUDE_towel_folding_theorem_l3451_345155


namespace NUMINAMATH_CALUDE_expand_expression_l3451_345163

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * 3 * x^3 = 33 * x^5 + 15 * x^4 - 9 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3451_345163


namespace NUMINAMATH_CALUDE_expression_factorization_l3451_345148

theorem expression_factorization (x : ℝ) : 
  (12 * x^3 + 95 * x - 6) - (-3 * x^3 + 5 * x - 6) = 15 * x * (x^2 + 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3451_345148


namespace NUMINAMATH_CALUDE_std_dev_of_scaled_data_l3451_345161

-- Define the type for our data set
def DataSet := Fin 100 → ℝ

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define the standard deviation of a data set
noncomputable def std_dev (data : DataSet) : ℝ := Real.sqrt (variance data)

-- Define a function that multiplies each element of a data set by 3
def scale_by_3 (data : DataSet) : DataSet := λ i => 3 * data i

-- Our theorem
theorem std_dev_of_scaled_data (original_data : DataSet) 
  (h : variance original_data = 2) : 
  std_dev (scale_by_3 original_data) = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_std_dev_of_scaled_data_l3451_345161


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonempty_solution_l3451_345143

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 3|

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_when_a_is_one :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ x, f 1 x ≥ min_val :=
sorry

-- Theorem 2: Range of a when the solution set of f(x) ≤ 3 is non-empty
theorem range_of_a_for_nonempty_solution :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ 3) ↔ (0 ≤ a ∧ a ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_range_of_a_for_nonempty_solution_l3451_345143


namespace NUMINAMATH_CALUDE_arccos_minus_one_equals_pi_l3451_345120

theorem arccos_minus_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_minus_one_equals_pi_l3451_345120


namespace NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l3451_345118

/-- Represents the number of days it takes for a bacteria colony to reach its habitat limit -/
def habitat_limit_days : ℕ := 22

/-- Represents the daily growth factor of the bacteria colony -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that two bacteria colonies starting simultaneously will reach the habitat limit in the same number of days as a single colony -/
theorem two_colonies_reach_limit_same_time (initial_population : ℕ) :
  (initial_population * daily_growth_factor ^ habitat_limit_days) =
  (2 * initial_population * daily_growth_factor ^ habitat_limit_days) / 2 :=
by
  sorry

#check two_colonies_reach_limit_same_time

end NUMINAMATH_CALUDE_two_colonies_reach_limit_same_time_l3451_345118


namespace NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_l3451_345104

/-- The perimeter of a semicircular cubicle with radius 14 is approximately 72 units. -/
theorem semicircular_cubicle_perimeter : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |14 * Real.pi + 28 - 72| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircular_cubicle_perimeter_l3451_345104


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3451_345142

theorem perfect_square_binomial : ∃ a : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3451_345142


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l3451_345114

theorem fraction_inequality_implies_inequality (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 < b / c^2 → a < b :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l3451_345114


namespace NUMINAMATH_CALUDE_horner_v3_value_l3451_345119

/-- Horner's Method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁶ - 5x⁵ + 6x⁴ + x² + 0.3x + 2 -/
def f (x : ℝ) : ℝ :=
  x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, -5, 6, 0, 1, 0.3, 2]

/-- Theorem: v₃ = -40 when evaluating f(-2) using Horner's Method -/
theorem horner_v3_value :
  let x := -2
  let v₀ := 1
  let v₁ := v₀ * x + f_coeffs[1]!
  let v₂ := v₁ * x + f_coeffs[2]!
  let v₃ := v₂ * x + f_coeffs[3]!
  v₃ = -40 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l3451_345119


namespace NUMINAMATH_CALUDE_donation_proof_l3451_345189

/-- The amount donated to Animal Preservation Park -/
def animal_park_donation : ℝ := sorry

/-- The amount donated to Treetown National Park and The Forest Reserve combined -/
def combined_donation : ℝ := animal_park_donation + 140

/-- The total donation to all three parks -/
def total_donation : ℝ := 1000

theorem donation_proof : combined_donation = 570 := by
  sorry

end NUMINAMATH_CALUDE_donation_proof_l3451_345189


namespace NUMINAMATH_CALUDE_f_negative_2011_l3451_345112

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

theorem f_negative_2011 (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2011_l3451_345112


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l3451_345136

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15732 + i^15733 + i^15734 + i^15735 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l3451_345136


namespace NUMINAMATH_CALUDE_triangle_problem_l3451_345141

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A) →
  (c = 7) →
  (b * Real.sin A = Real.sqrt 3) →
  (B = π / 6 ∧ b = Real.sqrt 19) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l3451_345141


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3451_345145

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 1) :
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3451_345145


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l3451_345147

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -3

/-- Theorem: The sum of the x-intercept and y-intercept of the line 3x - 2y - 6 = 0 is -1 -/
theorem sum_of_intercepts :
  line_equation x_intercept 0 ∧ 
  line_equation 0 y_intercept ∧ 
  x_intercept + y_intercept = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l3451_345147


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3451_345181

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 4
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := (total_republicans.choose subcommittee_republicans)
  let ways_to_choose_democrats : ℕ := (total_democrats.choose subcommittee_democrats)
  ways_to_choose_republicans * ways_to_choose_democrats = 1260 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3451_345181


namespace NUMINAMATH_CALUDE_muffin_price_theorem_l3451_345190

/-- Promotional sale: Buy three muffins at regular price, get fourth muffin free -/
def promotional_sale (regular_price : ℝ) : ℝ := 3 * regular_price

/-- The total amount John paid for four muffins -/
def total_paid : ℝ := 15

theorem muffin_price_theorem :
  ∃ (regular_price : ℝ), promotional_sale regular_price = total_paid ∧ regular_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_theorem_l3451_345190


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3451_345175

theorem quadratic_inequality_solution_set (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3451_345175


namespace NUMINAMATH_CALUDE_table_count_l3451_345170

theorem table_count (num_stools : ℕ → ℕ) (num_tables : ℕ) 
  (h1 : num_stools num_tables = 6 * num_tables)
  (h2 : 3 * num_stools num_tables + 4 * num_tables = 484) : 
  num_tables = 22 := by
  sorry

end NUMINAMATH_CALUDE_table_count_l3451_345170


namespace NUMINAMATH_CALUDE_sachins_age_l3451_345169

theorem sachins_age (sachin rahul : ℕ) 
  (age_difference : rahul = sachin + 4)
  (age_ratio : sachin * 9 = rahul * 7) : 
  sachin = 14 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l3451_345169


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l3451_345156

/-- Rainfall problem -/
theorem tuesday_rainfall (total_rainfall average_rainfall : ℝ) 
  (h1 : total_rainfall = 7 * average_rainfall)
  (h2 : average_rainfall = 3)
  (h3 : ∃ tuesday_rainfall : ℝ, 
    tuesday_rainfall = total_rainfall - tuesday_rainfall) :
  ∃ tuesday_rainfall : ℝ, tuesday_rainfall = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l3451_345156


namespace NUMINAMATH_CALUDE_course_size_l3451_345125

theorem course_size (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_course_size_l3451_345125


namespace NUMINAMATH_CALUDE_f_neg_two_value_l3451_345100

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7, if f(2) = -9, then f(-2) = -5 -/
theorem f_neg_two_value (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -a * x^5 - x^3 + b * x - 7
  f 2 = -9 → f (-2) = -5 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l3451_345100


namespace NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l3451_345128

/-- The number of burrs on Andrew's dog -/
def num_burrs : ℕ := 12

/-- The ratio of ticks to burrs on Andrew's dog -/
def tick_to_burr_ratio : ℕ := 6

/-- The total number of foreign objects (burrs and ticks) on Andrew's dog -/
def total_foreign_objects : ℕ := num_burrs + num_burrs * tick_to_burr_ratio

theorem andrews_dog_foreign_objects :
  total_foreign_objects = 84 :=
by sorry

end NUMINAMATH_CALUDE_andrews_dog_foreign_objects_l3451_345128


namespace NUMINAMATH_CALUDE_number_divisible_by_5_power_1000_without_zero_digit_l3451_345157

theorem number_divisible_by_5_power_1000_without_zero_digit :
  ∃ n : ℕ, (5^1000 ∣ n) ∧ (∀ d : ℕ, d < 10 → (n.digits 10).all (λ x => x ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_number_divisible_by_5_power_1000_without_zero_digit_l3451_345157


namespace NUMINAMATH_CALUDE_projectile_motion_time_l3451_345183

/-- The equation of motion for a projectile launched from the ground -/
def equation_of_motion (v : ℝ) (t : ℝ) : ℝ := -16 * t^2 + v * t

/-- The initial velocity of the projectile in feet per second -/
def initial_velocity : ℝ := 80

/-- The height reached by the projectile in feet -/
def height_reached : ℝ := 100

/-- The time taken to reach the specified height -/
def time_to_reach_height : ℝ := 2.5

theorem projectile_motion_time :
  equation_of_motion initial_velocity time_to_reach_height = height_reached :=
by sorry

end NUMINAMATH_CALUDE_projectile_motion_time_l3451_345183


namespace NUMINAMATH_CALUDE_browns_utility_bill_l3451_345123

/-- The total amount of Mrs. Brown's utility bills -/
def utility_bill_total (fifty_count : ℕ) (ten_count : ℕ) : ℕ :=
  50 * fifty_count + 10 * ten_count

/-- Theorem stating that Mrs. Brown's utility bills total $170 -/
theorem browns_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_browns_utility_bill_l3451_345123


namespace NUMINAMATH_CALUDE_hen_count_l3451_345158

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48) 
  (h2 : total_feet = 136) 
  (h3 : hen_feet = 2) 
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧ 
    hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hen_count_l3451_345158


namespace NUMINAMATH_CALUDE_second_company_base_rate_l3451_345199

/-- Represents the base rate and per-minute charge for a telephone company -/
structure TelephoneRate where
  baseRate : ℝ
  perMinuteCharge : ℝ

/-- Calculates the total charge for a given number of minutes -/
def totalCharge (rate : TelephoneRate) (minutes : ℝ) : ℝ :=
  rate.baseRate + rate.perMinuteCharge * minutes

theorem second_company_base_rate :
  let unitedRate : TelephoneRate := { baseRate := 11, perMinuteCharge := 0.25 }
  let otherRate : TelephoneRate := { baseRate := x, perMinuteCharge := 0.20 }
  let minutes : ℝ := 20
  totalCharge unitedRate minutes = totalCharge otherRate minutes →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l3451_345199


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3451_345122

-- Define the equation y^2 = 4
def equation (y : ℝ) : Prop := y^2 = 4

-- Define the statement y = 2
def statement (y : ℝ) : Prop := y = 2

-- Theorem: y = 2 is a sufficient but not necessary condition for y^2 = 4
theorem sufficient_but_not_necessary :
  (∀ y : ℝ, statement y → equation y) ∧
  ¬(∀ y : ℝ, equation y → statement y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3451_345122


namespace NUMINAMATH_CALUDE_smallest_flock_size_l3451_345146

theorem smallest_flock_size (total_sparrows : ℕ) (parrot_flock_size : ℕ) : 
  total_sparrows = 182 →
  parrot_flock_size = 14 →
  ∃ (P : ℕ), total_sparrows = parrot_flock_size * P →
  (∀ (S : ℕ), S > 0 ∧ S ∣ total_sparrows ∧ (∃ (Q : ℕ), S ∣ (parrot_flock_size * Q)) → S ≥ 14) ∧
  14 ∣ total_sparrows ∧ (∃ (R : ℕ), 14 ∣ (parrot_flock_size * R)) :=
by sorry

#check smallest_flock_size

end NUMINAMATH_CALUDE_smallest_flock_size_l3451_345146


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3451_345130

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to specify all properties of an isosceles triangle,
  -- just that it has a vertex angle
  vertexAngle : ℝ

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (has_40_degree_angle : ∃ (angle : ℝ), angle = 40 ∧ 
    (angle = triangle.vertexAngle ∨ 
     2 * angle + triangle.vertexAngle = 180)) :
  triangle.vertexAngle = 40 ∨ triangle.vertexAngle = 100 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3451_345130


namespace NUMINAMATH_CALUDE_min_value_theorem_l3451_345167

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y > z) (hyz : y + z > x) (hzx : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3451_345167


namespace NUMINAMATH_CALUDE_ratio_transitivity_l3451_345196

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 4 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitivity_l3451_345196


namespace NUMINAMATH_CALUDE_factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l3451_345121

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem: 101! · 102! is a perfect square -/
theorem factorial_101_102_is_perfect_square :
  is_perfect_square (factorial 101 * factorial 102) := by sorry

/-- Theorem: 100! · 101! is not a perfect square -/
theorem factorial_100_101_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 101) := by sorry

/-- Theorem: 100! · 102! is not a perfect square -/
theorem factorial_100_102_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 102) := by sorry

/-- Theorem: 101! · 103! is not a perfect square -/
theorem factorial_101_103_not_perfect_square :
  ¬ is_perfect_square (factorial 101 * factorial 103) := by sorry

/-- Theorem: 102! · 103! is not a perfect square -/
theorem factorial_102_103_not_perfect_square :
  ¬ is_perfect_square (factorial 102 * factorial 103) := by sorry

end NUMINAMATH_CALUDE_factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l3451_345121


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l3451_345138

theorem evaluate_polynomial : 2001^3 - 1998 * 2001^2 - 1998^2 * 2001 + 1998^3 = 35991 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l3451_345138


namespace NUMINAMATH_CALUDE_kids_difference_l3451_345160

theorem kids_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 202958) 
  (h2 : home_kids = 777622) : 
  home_kids - camp_kids = 574664 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l3451_345160


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3451_345139

def isIsoscelesRightTriangle (z₁ z₂ : ℂ) : Prop :=
  z₂ = Complex.exp (Real.pi * Complex.I / 4) * z₁

theorem isosceles_right_triangle_roots (a b : ℂ) (z₁ z₂ : ℂ) :
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  isIsoscelesRightTriangle z₁ z₂ →
  a^2 / b = 4 + 2*Complex.I*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3451_345139


namespace NUMINAMATH_CALUDE_game_points_total_l3451_345184

/-- Game points calculation -/
theorem game_points_total (eric mark samanta daisy jake : ℕ) : 
  eric = 6 ∧ 
  mark = eric + eric / 2 ∧ 
  samanta = mark + 8 ∧ 
  daisy = (samanta + mark + eric) - (samanta + mark + eric) / 4 ∧
  jake = max samanta (max mark (max eric daisy)) - min samanta (min mark (min eric daisy)) →
  samanta + mark + eric + daisy + jake = 67 := by
  sorry


end NUMINAMATH_CALUDE_game_points_total_l3451_345184


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3451_345154

theorem tan_alpha_value (α : Real) (h1 : π < α ∧ α < 3*π/2) (h2 : Real.sin (α/2) = Real.sqrt 5 / 3) :
  Real.tan α = -4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3451_345154


namespace NUMINAMATH_CALUDE_no_solutions_cubic_equation_l3451_345195

theorem no_solutions_cubic_equation :
  (∀ x y : ℕ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) ∧
  (∀ x y : ℤ, x ≠ y → x^3 + 5*y ≠ y^3 + 5*x) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_cubic_equation_l3451_345195


namespace NUMINAMATH_CALUDE_smallest_abs_value_rational_l3451_345149

theorem smallest_abs_value_rational (q : ℚ) : |0| ≤ |q| := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_rational_l3451_345149


namespace NUMINAMATH_CALUDE_rock_ratio_l3451_345162

/-- Represents the rock collecting contest between Sydney and Conner --/
structure RockContest where
  sydney_initial : ℕ
  conner_initial : ℕ
  sydney_day1 : ℕ
  conner_day1_multiplier : ℕ
  conner_day2 : ℕ
  conner_day3 : ℕ

/-- Calculates the number of rocks Sydney collected on day 3 --/
def sydney_day3 (contest : RockContest) : ℕ :=
  contest.sydney_initial + contest.sydney_day1 + 
  (contest.conner_initial + contest.sydney_day1 * contest.conner_day1_multiplier + 
   contest.conner_day2 + contest.conner_day3) - 
  (contest.sydney_initial + contest.sydney_day1)

/-- The main theorem stating the ratio of rocks collected --/
theorem rock_ratio (contest : RockContest) 
  (h1 : contest.sydney_initial = 837)
  (h2 : contest.conner_initial = 723)
  (h3 : contest.sydney_day1 = 4)
  (h4 : contest.conner_day1_multiplier = 8)
  (h5 : contest.conner_day2 = 123)
  (h6 : contest.conner_day3 = 27) :
  sydney_day3 contest = 2 * (contest.sydney_day1 * contest.conner_day1_multiplier) := by
  sorry

end NUMINAMATH_CALUDE_rock_ratio_l3451_345162


namespace NUMINAMATH_CALUDE_min_elements_special_relation_l3451_345165

/-- A relation on a set X satisfying the given properties -/
structure SpecialRelation (X : Type) where
  rel : X → X → Prop
  irreflexive : ∀ x, ¬(rel x x)
  trichotomous : ∀ x y, x ≠ y → (rel x y ∨ rel y x) ∧ ¬(rel x y ∧ rel y x)
  transitive_element : ∀ x y, rel x y → ∃ z, rel x z ∧ rel z y

/-- The minimum number of elements in a set with a SpecialRelation is 7 -/
theorem min_elements_special_relation :
  ∀ (X : Type) [Fintype X] (r : SpecialRelation X),
  Fintype.card X ≥ 7 ∧ (∀ (Y : Type) [Fintype Y], SpecialRelation Y → Fintype.card Y < 7 → False) :=
sorry

end NUMINAMATH_CALUDE_min_elements_special_relation_l3451_345165


namespace NUMINAMATH_CALUDE_percent_of_whole_l3451_345107

theorem percent_of_whole (part whole : ℝ) (h : whole ≠ 0) : 
  (part / whole) * 100 = 50 → part = 80 ∧ whole = 160 :=
by sorry

end NUMINAMATH_CALUDE_percent_of_whole_l3451_345107


namespace NUMINAMATH_CALUDE_airport_passenger_ratio_l3451_345171

/-- Proves that the ratio of passengers using Miami Airport to those using Logan Airport is 4:1 -/
theorem airport_passenger_ratio :
  let total_passengers : ℝ := 38.3 * 1000000
  let kennedy_passengers : ℝ := total_passengers / 3
  let miami_passengers : ℝ := kennedy_passengers / 2
  let logan_passengers : ℝ := 1.5958333333333332 * 1000000
  miami_passengers / logan_passengers = 4 := by
  sorry

end NUMINAMATH_CALUDE_airport_passenger_ratio_l3451_345171


namespace NUMINAMATH_CALUDE_addition_verification_l3451_345186

theorem addition_verification (a b s : ℝ) (h : s = a + b) :
  s - a = b ∧ s - b = a := by
  sorry

end NUMINAMATH_CALUDE_addition_verification_l3451_345186


namespace NUMINAMATH_CALUDE_percent_democrat_voters_l3451_345177

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.75 * D + 0.30 * R = 57 →
  D = 60 := by
sorry

end NUMINAMATH_CALUDE_percent_democrat_voters_l3451_345177


namespace NUMINAMATH_CALUDE_room_length_from_carpet_cost_room_length_is_208_l3451_345151

/-- The length of a room given carpet and cost information -/
theorem room_length_from_carpet_cost (room_width : ℝ) (carpet_width : ℝ) 
  (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) : ℝ :=
  let total_area := total_cost / carpet_cost_per_sqm
  let carpet_width_m := carpet_width / 100
  total_area / carpet_width_m

/-- Proof that the room length is 208 meters given specific conditions -/
theorem room_length_is_208 :
  room_length_from_carpet_cost 9 75 12 1872 = 208 := by
  sorry

end NUMINAMATH_CALUDE_room_length_from_carpet_cost_room_length_is_208_l3451_345151


namespace NUMINAMATH_CALUDE_carla_karen_age_difference_l3451_345103

-- Define the current ages
def karen_age : ℕ := 2
def frank_future_age : ℕ := 36
def years_until_frank_future : ℕ := 5

-- Define relationships between ages
def frank_age : ℕ := frank_future_age - years_until_frank_future
def ty_age : ℕ := frank_future_age / 3
def carla_age : ℕ := (ty_age - 4) / 2

-- Theorem to prove
theorem carla_karen_age_difference : carla_age - karen_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_carla_karen_age_difference_l3451_345103


namespace NUMINAMATH_CALUDE_x_over_3_is_directly_proportional_l3451_345180

/-- A function f : ℝ → ℝ is directly proportional if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectlyProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function f(x) = x/3 is directly proportional -/
theorem x_over_3_is_directly_proportional :
  IsDirectlyProportional (fun x => x / 3) := by
  sorry

end NUMINAMATH_CALUDE_x_over_3_is_directly_proportional_l3451_345180


namespace NUMINAMATH_CALUDE_max_value_inequality_l3451_345131

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3451_345131


namespace NUMINAMATH_CALUDE_parabola_properties_l3451_345185

/-- Given a parabola y = ax² - 5x - 3 passing through (-1, 4), prove its properties -/
theorem parabola_properties (a : ℝ) : 
  (a * (-1)^2 - 5 * (-1) - 3 = 4) → -- The parabola passes through (-1, 4)
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 - 5 * x₁ - 3 = 0 ∧ a * x₂^2 - 5 * x₂ - 3 = 0) ∧ -- Intersects x-axis at two points
  (- (-5) / (2 * a) = 5/4) -- Axis of symmetry is x = 5/4
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3451_345185


namespace NUMINAMATH_CALUDE_car_production_is_four_l3451_345159

/-- Represents the factory's production and profit data -/
structure FactoryData where
  car_material_cost : ℕ
  car_selling_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_count : ℕ
  motorcycle_selling_price : ℕ
  profit_difference : ℕ

/-- Calculates the number of cars that could be produced per month -/
def calculate_car_production (data : FactoryData) : ℕ :=
  let motorcycle_profit := data.motorcycle_count * data.motorcycle_selling_price - data.motorcycle_material_cost
  let car_profit := fun c => c * data.car_selling_price - data.car_material_cost
  (motorcycle_profit - data.profit_difference + data.car_material_cost) / data.car_selling_price

theorem car_production_is_four (data : FactoryData) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.car_selling_price = 50)
  (h3 : data.motorcycle_material_cost = 250)
  (h4 : data.motorcycle_count = 8)
  (h5 : data.motorcycle_selling_price = 50)
  (h6 : data.profit_difference = 50) :
  calculate_car_production data = 4 := by
  sorry

end NUMINAMATH_CALUDE_car_production_is_four_l3451_345159


namespace NUMINAMATH_CALUDE_remaining_balance_l3451_345193

def house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_contribution_percentage : ℝ := 0.30

theorem remaining_balance (price : ℝ) (down_percent : ℝ) (parents_percent : ℝ) :
  price = house_price →
  down_percent = down_payment_percentage →
  parents_percent = parents_contribution_percentage →
  price * (1 - down_percent) * (1 - parents_percent) = 56000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balance_l3451_345193


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3451_345140

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  1^2 + (3/2)^2 = r^2 → 
  π * r^2 / 2 = 9 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3451_345140


namespace NUMINAMATH_CALUDE_half_shading_sufficient_l3451_345111

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (total_cells : ℕ)

/-- Represents the minimum number of cells to be shaded --/
def min_shaded_cells (g : Grid) : ℕ := g.total_cells / 2

/-- Theorem stating that shading half the cells is sufficient --/
theorem half_shading_sufficient (g : Grid) (h : g.size = 12) (h' : g.total_cells = 144) :
  ∃ (shaded : ℕ), shaded = min_shaded_cells g ∧ 
  shaded ≤ g.total_cells ∧
  shaded ≥ g.total_cells / 2 :=
sorry

#check half_shading_sufficient

end NUMINAMATH_CALUDE_half_shading_sufficient_l3451_345111


namespace NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_log_x_gt_1_l3451_345134

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement of the theorem
theorem x_gt_1_necessary_not_sufficient_for_log_x_gt_1 :
  (∀ x : ℝ, log10 x > 1 → x > 1) ∧
  ¬(∀ x : ℝ, x > 1 → log10 x > 1) :=
sorry

end NUMINAMATH_CALUDE_x_gt_1_necessary_not_sufficient_for_log_x_gt_1_l3451_345134


namespace NUMINAMATH_CALUDE_area_between_circles_l3451_345133

theorem area_between_circles (R : ℝ) (r : ℝ) (d : ℝ) (chord_length : ℝ) :
  R = 12 →
  d = 2 →
  chord_length = 20 →
  r = Real.sqrt (R^2 - d^2 - (chord_length/2)^2) →
  π * (R^2 - r^2) = 100 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_circles_l3451_345133


namespace NUMINAMATH_CALUDE_remainder_sum_l3451_345109

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 58) 
  (hb : b % 90 = 84) : 
  (a + b) % 30 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3451_345109


namespace NUMINAMATH_CALUDE_pizza_order_cost_l3451_345164

/- Define the problem parameters -/
def num_pizzas : Nat := 3
def price_per_pizza : Nat := 10
def num_toppings : Nat := 4
def price_per_topping : Nat := 1
def tip : Nat := 5

/- Define the total cost calculation -/
def total_cost : Nat :=
  num_pizzas * price_per_pizza +
  num_toppings * price_per_topping +
  tip

/- Theorem statement -/
theorem pizza_order_cost : total_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_cost_l3451_345164


namespace NUMINAMATH_CALUDE_semicircle_inscriptions_l3451_345168

theorem semicircle_inscriptions (D : ℝ) (N : ℕ) (h : N > 0) : 
  let r := D / (2 * N)
  let R := N * r
  let A := N * (π * r^2 / 2)
  let B := π * R^2 / 2 - A
  A / B = 2 / 25 → N = 14 := by
sorry

end NUMINAMATH_CALUDE_semicircle_inscriptions_l3451_345168


namespace NUMINAMATH_CALUDE_complex_equation_implies_ab_eight_l3451_345174

theorem complex_equation_implies_ab_eight (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * (3 + i) = 10 + 10 * i →
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_ab_eight_l3451_345174


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3451_345124

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    and a point P on its right branch,
    a line through P intersects the asymptotes at A and B,
    where A is in the first quadrant and B is in the fourth quadrant,
    O is the origin, AP = (1/2)PB, and the area of triangle AOB is 2b,
    prove that the length of the real axis of C is 32/9. -/
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0)
  (P A B : ℝ × ℝ)
  (hC : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (hP : P.1 > 0)
  (hA : A.1 > 0 ∧ A.2 > 0)
  (hB : B.1 > 0 ∧ B.2 < 0)
  (hAP : A - P = (1/2) • (P - B))
  (hAOB : abs ((A.1 * B.2 - A.2 * B.1) / 2) = 2 * b) :
  2 * a = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3451_345124


namespace NUMINAMATH_CALUDE_greatest_n_value_greatest_n_is_10_l3451_345116

theorem greatest_n_value (n : ℤ) (h : 303 * n^3 ≤ 380000) : n ≤ 10 := by
  sorry

theorem greatest_n_is_10 : ∃ n : ℤ, 303 * n^3 ≤ 380000 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_greatest_n_is_10_l3451_345116


namespace NUMINAMATH_CALUDE_paul_initial_stock_l3451_345150

/-- The number of pencils Paul makes in a day -/
def daily_production : ℕ := 100

/-- The number of days Paul works in a week -/
def working_days : ℕ := 5

/-- The number of pencils Paul sold during the week -/
def pencils_sold : ℕ := 350

/-- The number of pencils in stock at the end of the week -/
def end_stock : ℕ := 230

/-- The number of pencils Paul had at the beginning of the week -/
def initial_stock : ℕ := daily_production * working_days + end_stock - pencils_sold

theorem paul_initial_stock :
  initial_stock = 380 :=
sorry

end NUMINAMATH_CALUDE_paul_initial_stock_l3451_345150


namespace NUMINAMATH_CALUDE_petya_candies_when_masha_gets_101_l3451_345178

def candy_game (n : ℕ) : ℕ × ℕ := 
  let masha_sum := n^2
  let petya_sum := n * (n + 1)
  (masha_sum, petya_sum)

theorem petya_candies_when_masha_gets_101 : 
  ∃ n : ℕ, (candy_game n).1 ≥ 101 ∧ (candy_game (n-1)).1 < 101 → (candy_game (n-1)).2 = 110 :=
by sorry

end NUMINAMATH_CALUDE_petya_candies_when_masha_gets_101_l3451_345178


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3451_345182

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3451_345182


namespace NUMINAMATH_CALUDE_ratio_of_segments_l3451_345129

/-- Given points A, B, C, and D on a line in that order, with AB : AC = 1 : 5 and BC : CD = 2 : 1, prove AB : CD = 1 : 2 -/
theorem ratio_of_segments (A B C D : ℝ) (h_order : A < B ∧ B < C ∧ C < D) 
  (h_ratio1 : (B - A) / (C - A) = 1 / 5)
  (h_ratio2 : (C - B) / (D - C) = 2 / 1) :
  (B - A) / (D - C) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l3451_345129


namespace NUMINAMATH_CALUDE_definite_integral_problem_l3451_345115

open Real MeasureTheory Interval

theorem definite_integral_problem :
  ∫ x in (-1 : ℝ)..1, x * cos x + (x^2)^(1/3) = 6/5 := by sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l3451_345115


namespace NUMINAMATH_CALUDE_calculate_X_l3451_345179

theorem calculate_X : ∀ M N X : ℚ,
  M = 3009 / 3 →
  N = M / 4 →
  X = M + 2 * N →
  X = 1504.5 := by
sorry

end NUMINAMATH_CALUDE_calculate_X_l3451_345179


namespace NUMINAMATH_CALUDE_toothpick_burning_time_l3451_345132

/-- Represents a rectangular structure of toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  toothpicks : Nat

/-- Represents the burning process of the toothpick structure -/
def BurningProcess (r : ToothpickRectangle) (burn_time : Nat) : Prop :=
  r.rows = 3 ∧
  r.cols = 5 ∧
  r.toothpicks = 38 ∧
  burn_time = 10 ∧
  ∃ (total_time : Nat), total_time = 65 ∧
    (∀ (t : Nat), t ≤ total_time →
      ∃ (burned : Nat), burned ≤ r.toothpicks ∧
        burned = min r.toothpicks (2 * (t / burn_time + 1)))

/-- Theorem stating that the entire structure burns in 65 seconds -/
theorem toothpick_burning_time (r : ToothpickRectangle) (burn_time : Nat) :
  BurningProcess r burn_time →
  ∃ (total_time : Nat), total_time = 65 ∧
    (∀ (t : Nat), t > total_time →
      ∀ (burned : Nat), burned = r.toothpicks) :=
by
  sorry

end NUMINAMATH_CALUDE_toothpick_burning_time_l3451_345132


namespace NUMINAMATH_CALUDE_probability_three_girls_l3451_345105

theorem probability_three_girls (total : ℕ) (girls : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 9 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 12 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_l3451_345105


namespace NUMINAMATH_CALUDE_both_are_liars_l3451_345197

-- Define the possible types of islanders
inductive IslanderType
  | Knight
  | Liar

-- Define the islanders
def A : IslanderType := sorry
def B : IslanderType := sorry

-- Define A's statement
def A_statement : Prop := (A = IslanderType.Liar) ∧ (B ≠ IslanderType.Liar)

-- Define the truth-telling property of knights and liars
def tells_truth (i : IslanderType) (p : Prop) : Prop :=
  (i = IslanderType.Knight ∧ p) ∨ (i = IslanderType.Liar ∧ ¬p)

-- Theorem to prove
theorem both_are_liars :
  tells_truth A A_statement →
  A = IslanderType.Liar ∧ B = IslanderType.Liar :=
by sorry

end NUMINAMATH_CALUDE_both_are_liars_l3451_345197


namespace NUMINAMATH_CALUDE_value_of_x_l3451_345194

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3451_345194


namespace NUMINAMATH_CALUDE_union_equality_implies_t_value_l3451_345127

def M (t : ℝ) : Set ℝ := {1, 3, t}
def N (t : ℝ) : Set ℝ := {t^2 - t + 1}

theorem union_equality_implies_t_value (t : ℝ) :
  M t ∪ N t = M t → t = 0 ∨ t = 2 ∨ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_t_value_l3451_345127


namespace NUMINAMATH_CALUDE_reggies_money_l3451_345144

/-- The amount of money Reggie's father gave him -/
def money_given : ℕ := sorry

/-- The number of books Reggie bought -/
def books_bought : ℕ := 5

/-- The cost of each book in dollars -/
def book_cost : ℕ := 2

/-- The amount of money Reggie has left after buying the books -/
def money_left : ℕ := 38

/-- Theorem stating that the money given by Reggie's father is $48 -/
theorem reggies_money : money_given = books_bought * book_cost + money_left := by sorry

end NUMINAMATH_CALUDE_reggies_money_l3451_345144


namespace NUMINAMATH_CALUDE_total_flowers_l3451_345188

def roses : ℕ := 5
def lilies : ℕ := 2

theorem total_flowers : roses + lilies = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l3451_345188


namespace NUMINAMATH_CALUDE_john_works_five_days_l3451_345102

/-- Represents the number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the total number of widgets John makes per week -/
def widgets_per_week : ℕ := 800

/-- Calculates the number of days John works per week -/
def days_worked_per_week : ℕ :=
  widgets_per_week /(widgets_per_hour * hours_per_day)

/-- Theorem stating that John works 5 days per week -/
theorem john_works_five_days :
  days_worked_per_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_works_five_days_l3451_345102


namespace NUMINAMATH_CALUDE_opposites_sum_to_zero_l3451_345173

theorem opposites_sum_to_zero (a b : ℝ) (h : a = -b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_to_zero_l3451_345173


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l3451_345126

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_numbers_greater_than_threshold :
  (numbers.filter (λ x => x > threshold)).sum = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l3451_345126


namespace NUMINAMATH_CALUDE_fox_initial_money_l3451_345172

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 4

/-- The toll paid after each crossing -/
def toll : ℕ := 50

/-- The initial toll paid before the first crossing -/
def initial_toll : ℕ := 10

/-- The function that calculates Fox's money after each crossing -/
def money_after_crossing (initial_money : ℕ) (crossing : ℕ) : ℤ :=
  (2^crossing) * (initial_money - initial_toll) - 
  (2^crossing - 1) * toll - 
  initial_toll

/-- The theorem stating that Fox started with 56 coins -/
theorem fox_initial_money : 
  ∃ (initial_money : ℕ), 
    initial_money = 56 ∧ 
    money_after_crossing initial_money num_crossings = 0 :=
  sorry

end NUMINAMATH_CALUDE_fox_initial_money_l3451_345172


namespace NUMINAMATH_CALUDE_one_root_in_interval_l3451_345101

theorem one_root_in_interval : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ 2 * x^3 - 6 * x^2 + 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_root_in_interval_l3451_345101


namespace NUMINAMATH_CALUDE_pomelo_sales_theorem_l3451_345198

/-- Represents the sales data for a week -/
structure WeeklySales where
  planned_daily : ℕ
  deviations : List ℤ
  selling_price : ℕ
  shipping_cost : ℕ

/-- Calculates the difference between highest and lowest sales days -/
def sales_difference (sales : WeeklySales) : ℕ :=
  let max_dev := sales.deviations.maximum?
  let min_dev := sales.deviations.minimum?
  match max_dev, min_dev with
  | some max, some min => (max - min).natAbs
  | _, _ => 0

/-- Calculates the total sales for the week -/
def total_sales (sales : WeeklySales) : ℕ :=
  sales.planned_daily * 7 + sales.deviations.sum.natAbs

/-- Calculates the total profit for the week -/
def total_profit (sales : WeeklySales) : ℕ :=
  (sales.selling_price - sales.shipping_cost) * (total_sales sales)

/-- Main theorem to prove -/
theorem pomelo_sales_theorem (sales : WeeklySales)
  (h1 : sales.planned_daily = 100)
  (h2 : sales.deviations = [3, -5, -2, 11, -7, 13, 5])
  (h3 : sales.selling_price = 8)
  (h4 : sales.shipping_cost = 3) :
  sales_difference sales = 20 ∧
  total_sales sales = 718 ∧
  total_profit sales = 3590 := by
  sorry


end NUMINAMATH_CALUDE_pomelo_sales_theorem_l3451_345198


namespace NUMINAMATH_CALUDE_quadcycle_count_l3451_345191

theorem quadcycle_count (b t q : ℕ) : 
  b + t + q = 10 →
  2*b + 3*t + 4*q = 29 →
  q = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadcycle_count_l3451_345191


namespace NUMINAMATH_CALUDE_flag_height_l3451_345187

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- The three fabric squares Bobby has -/
def fabric1 : Rectangle := ⟨8, 5⟩
def fabric2 : Rectangle := ⟨10, 7⟩
def fabric3 : Rectangle := ⟨5, 5⟩

/-- The desired length of the flag -/
def flagLength : ℝ := 15

/-- Theorem stating that the height of the flag will be 9 feet -/
theorem flag_height :
  (area fabric1 + area fabric2 + area fabric3) / flagLength = 9 := by
  sorry

end NUMINAMATH_CALUDE_flag_height_l3451_345187


namespace NUMINAMATH_CALUDE_max_profit_at_grade_5_l3451_345137

def profit (x : ℕ) : ℝ :=
  (4 * (x - 1) + 8) * (60 - 6 * (x - 1))

theorem max_profit_at_grade_5 :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 10 → profit x ≤ profit 5 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_grade_5_l3451_345137


namespace NUMINAMATH_CALUDE_sum_of_periodic_function_l3451_345117

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

def translate_right (f : ℝ → ℝ) (n : ℝ) : ℝ → ℝ :=
  fun x ↦ f (x - n)

theorem sum_of_periodic_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_odd_translated : is_odd_function (translate_right f 1))
  (h_f2 : f 2 = -1) :
  (Finset.range 2011).sum (fun i ↦ f (i + 1 : ℝ)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_periodic_function_l3451_345117


namespace NUMINAMATH_CALUDE_solution_set_of_even_decreasing_quadratic_l3451_345153

def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2*a) * x - 2*b

theorem solution_set_of_even_decreasing_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_decreasing : ∀ x y, 0 < x → x < y → f a b y < f a b x) :
  {x : ℝ | f a b x > 0} = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_even_decreasing_quadratic_l3451_345153


namespace NUMINAMATH_CALUDE_polyhedron_distance_equation_l3451_345106

/-- A convex polyhedron with 12 regular triangular faces -/
structure Polyhedron :=
  (e : ℝ)  -- Common edge length
  (t : ℝ)  -- Additional length in the distance between non-adjacent five-edged vertices

/-- The distance between two non-adjacent five-edged vertices is (e+t) -/
def distance (p : Polyhedron) : ℝ := p.e + p.t

/-- Theorem: For the given polyhedron, t³ - 7et² + 2e³ = 0 -/
theorem polyhedron_distance_equation (p : Polyhedron) : 
  p.t^3 - 7 * p.e * p.t^2 + 2 * p.e^3 = 0 :=
sorry

end NUMINAMATH_CALUDE_polyhedron_distance_equation_l3451_345106


namespace NUMINAMATH_CALUDE_direct_variation_problem_l3451_345113

/-- A function representing direct variation --/
def direct_variation (k : ℝ) (x : ℝ) : ℝ := k * x

theorem direct_variation_problem (k : ℝ) :
  (direct_variation k 2.5 = 10) →
  (direct_variation k (-5) = -20) := by
  sorry

#check direct_variation_problem

end NUMINAMATH_CALUDE_direct_variation_problem_l3451_345113


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3451_345108

-- Define a function to get the unit's place digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_expression : unitsDigit ((3^34 * 7^21) + 5^17) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3451_345108


namespace NUMINAMATH_CALUDE_margies_driving_distance_l3451_345192

/-- Proves that Margie can drive 400 miles with $50 worth of gas -/
theorem margies_driving_distance 
  (car_efficiency : ℝ) 
  (gas_price : ℝ) 
  (gas_budget : ℝ) 
  (h1 : car_efficiency = 40) 
  (h2 : gas_price = 5) 
  (h3 : gas_budget = 50) : 
  (gas_budget / gas_price) * car_efficiency = 400 := by
sorry

end NUMINAMATH_CALUDE_margies_driving_distance_l3451_345192


namespace NUMINAMATH_CALUDE_ball_count_proof_l3451_345166

/-- 
Given a bag with m balls, including 6 red balls, 
if the probability of picking a red ball is 0.3, then m = 20.
-/
theorem ball_count_proof (m : ℕ) (h1 : m > 0) (h2 : 6 ≤ m) : 
  (6 : ℝ) / m = 0.3 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_proof_l3451_345166


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3451_345152

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ+) : ℕ := (Finset.range n.val.succ).prod (fun i => i + 1)

/-- Theorem stating that the greatest common divisor of 8! and 10! is equal to 8! -/
theorem gcd_8_factorial_10_factorial :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3451_345152


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_parallel_l3451_345176

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem planes_parallel_from_skew_lines_parallel 
  (h_distinct : α ≠ β)
  (h_different : a ≠ b)
  (h_skew : skew_lines a b)
  (h_a_alpha : parallel_line_plane a α)
  (h_b_alpha : parallel_line_plane b α)
  (h_a_beta : parallel_line_plane a β)
  (h_b_beta : parallel_line_plane b β) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_parallel_l3451_345176


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l3451_345110

theorem quadratic_integer_roots_count : 
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁^2 - m*x₁ + 36 = 0 ∧ x₂^2 - m*x₂ + 36 = 0 ∧ x₁ ≠ x₂)
  (∃! (s : Finset ℤ), (∀ m ∈ s, f m) ∧ s.card = 10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l3451_345110


namespace NUMINAMATH_CALUDE_circle_through_line_intersections_l3451_345135

/-- Given a line that intersects the coordinate axes, prove that the circle passing through
    the origin and the intersection points has a specific equation. -/
theorem circle_through_line_intersections (x y : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 / 2 - A.2 / 4 = 1) ∧ 
    (B.1 / 2 - B.2 / 4 = 1) ∧ 
    (A.2 = 0) ∧ 
    (B.1 = 0) ∧
    ((x - 1)^2 + (y + 2)^2 = 5) ↔ 
    (x^2 + y^2 = A.1^2 + A.2^2 ∧ 
     x^2 + y^2 = B.1^2 + B.2^2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_through_line_intersections_l3451_345135
