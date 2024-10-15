import Mathlib

namespace NUMINAMATH_CALUDE_herman_feeding_months_l2687_268792

/-- The number of months Herman feeds the birds -/
def feeding_months (cups_per_day : ℚ) (total_cups : ℚ) (days_per_month : ℚ) : ℚ :=
  (total_cups / cups_per_day) / days_per_month

theorem herman_feeding_months :
  feeding_months 1 90 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_herman_feeding_months_l2687_268792


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_nine_xy_squared_l2687_268754

theorem factorization_cubic_minus_nine_xy_squared (x y : ℝ) :
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_nine_xy_squared_l2687_268754


namespace NUMINAMATH_CALUDE_principal_is_8925_l2687_268749

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal amount is 8925 -/
theorem principal_is_8925 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
sorry

end NUMINAMATH_CALUDE_principal_is_8925_l2687_268749


namespace NUMINAMATH_CALUDE_next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l2687_268789

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isValidYear (year : Nat) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

theorem next_year_with_sum_4 :
  ∀ year, year > 2020 → sumOfDigits year = 4 → year ≥ 2101 :=
by sorry

theorem year_2101_is_valid :
  isValidYear 2101 :=
by sorry

theorem year_2101_is_smallest :
  ∀ year, isValidYear year → year ≥ 2101 :=
by sorry

end NUMINAMATH_CALUDE_next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l2687_268789


namespace NUMINAMATH_CALUDE_pure_imaginary_square_l2687_268723

theorem pure_imaginary_square (a : ℝ) (z : ℂ) : 
  z = a + (1 + a) * Complex.I → 
  (∃ b : ℝ, z = b * Complex.I) → 
  z^2 = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_l2687_268723


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2687_268738

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 113 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (77 * n) % 385 = 231 % 385 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n → (77 * m) % 385 ≠ 231 % 385 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l2687_268738


namespace NUMINAMATH_CALUDE_common_chord_equation_l2687_268782

/-- The equation of the line containing the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4 = 0) →
  (x^2 + y^2 - 4*x + 4*y - 12 = 0) →
  (x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2687_268782


namespace NUMINAMATH_CALUDE_stability_comparison_l2687_268786

/-- Represents a student's scores in the competition -/
structure StudentScores where
  scores : List ℝ
  mean : ℝ
  variance : ℝ

/-- The competition has 5 rounds -/
def num_rounds : ℕ := 5

/-- Stability comparison of two students' scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : StudentScores) 
  (h1 : a.scores.length = num_rounds)
  (h2 : b.scores.length = num_rounds)
  (h3 : a.mean = 90)
  (h4 : b.mean = 90)
  (h5 : a.variance = 15)
  (h6 : b.variance = 3) :
  more_stable b a :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l2687_268786


namespace NUMINAMATH_CALUDE_box_volume_formula_l2687_268741

/-- The volume of a box formed by cutting rectangles from a sheet and folding up the flaps -/
def box_volume (x y : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*y) * y

/-- The original sheet dimensions -/
def sheet_length : ℝ := 16
def sheet_width : ℝ := 12

theorem box_volume_formula (x y : ℝ) :
  box_volume x y = 4*x*y^2 - 24*x*y + 192*y - 32*y^2 :=
by sorry

end NUMINAMATH_CALUDE_box_volume_formula_l2687_268741


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2687_268730

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2687_268730


namespace NUMINAMATH_CALUDE_movie_start_time_l2687_268779

-- Define the movie duration in minutes
def movie_duration : ℕ := 3 * 60

-- Define the remaining time in minutes
def remaining_time : ℕ := 36

-- Define the end time (5:44 pm) in minutes since midnight
def end_time : ℕ := 17 * 60 + 44

-- Define the start time (to be proven) in minutes since midnight
def start_time : ℕ := 15 * 60 + 20

-- Theorem statement
theorem movie_start_time :
  movie_duration - remaining_time = end_time - start_time :=
by sorry

end NUMINAMATH_CALUDE_movie_start_time_l2687_268779


namespace NUMINAMATH_CALUDE_movie_ticket_price_is_30_l2687_268761

/-- The price of a movie ticket -/
def movie_ticket_price : ℝ := sorry

/-- The price of a football game ticket -/
def football_ticket_price : ℝ := sorry

/-- Eight movie tickets cost 2 times as much as one football game ticket -/
axiom ticket_price_relation : 8 * movie_ticket_price = 2 * football_ticket_price

/-- The total amount paid for 8 movie tickets and 5 football game tickets is $840 -/
axiom total_cost : 8 * movie_ticket_price + 5 * football_ticket_price = 840

theorem movie_ticket_price_is_30 : movie_ticket_price = 30 := by sorry

end NUMINAMATH_CALUDE_movie_ticket_price_is_30_l2687_268761


namespace NUMINAMATH_CALUDE_soda_quarters_l2687_268771

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the total amount paid in dollars -/
def total_paid : ℕ := 4

/-- Represents the number of quarters paid for chips -/
def quarters_for_chips : ℕ := 4

/-- Calculates the number of quarters paid for soda -/
def quarters_for_soda : ℕ := (total_paid - quarters_for_chips / quarters_per_dollar) * quarters_per_dollar

theorem soda_quarters : quarters_for_soda = 12 := by
  sorry

end NUMINAMATH_CALUDE_soda_quarters_l2687_268771


namespace NUMINAMATH_CALUDE_same_terminal_side_l2687_268700

/-- Proves that given an angle of -3π/10 radians, 306° has the same terminal side when converted to degrees -/
theorem same_terminal_side : ∃ (β : ℝ), β = 306 ∧ ∃ (k : ℤ), β = (-3/10 * π) * (180/π) + 360 * k :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_l2687_268700


namespace NUMINAMATH_CALUDE_final_S_value_l2687_268704

def S : ℕ → ℕ
  | 0 => 1
  | n + 1 => S n + 2

theorem final_S_value : S 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_S_value_l2687_268704


namespace NUMINAMATH_CALUDE_nancy_grew_six_potatoes_l2687_268751

/-- The number of potatoes Sandy grew -/
def sandy_potatoes : ℕ := 7

/-- The total number of potatoes Nancy and Sandy grew together -/
def total_potatoes : ℕ := 13

/-- The number of potatoes Nancy grew -/
def nancy_potatoes : ℕ := total_potatoes - sandy_potatoes

theorem nancy_grew_six_potatoes : nancy_potatoes = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_grew_six_potatoes_l2687_268751


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2687_268719

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 + 4 * z - 5) * (4 * z^3 - 3 * z^2 + 2) =
  12 * z^5 + 7 * z^4 - 26 * z^3 + 21 * z^2 + 8 * z - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2687_268719


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l2687_268778

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ z w : ℝ, Real.sin z + Real.sin w = 1/3 → 
    Real.sin y - Real.cos x ^ 2 ≤ Real.sin w - Real.cos z ^ 2) →
  Real.sin y - Real.cos x ^ 2 = 4/9 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l2687_268778


namespace NUMINAMATH_CALUDE_mr_resty_total_units_l2687_268725

/-- Represents the number of apartment units on each floor of a building -/
def BuildingUnits := List Nat

/-- Building A's unit distribution -/
def building_a : BuildingUnits := [2, 4, 6, 8, 10, 12]

/-- Building B's unit distribution (identical to A) -/
def building_b : BuildingUnits := building_a

/-- Building C's unit distribution -/
def building_c : BuildingUnits := [3, 5, 7, 9]

/-- Calculate the total number of units in a building -/
def total_units (building : BuildingUnits) : Nat :=
  building.sum

/-- The main theorem stating the total number of apartment units Mr. Resty has -/
theorem mr_resty_total_units : 
  total_units building_a + total_units building_b + total_units building_c = 108 := by
  sorry

end NUMINAMATH_CALUDE_mr_resty_total_units_l2687_268725


namespace NUMINAMATH_CALUDE_power_sum_of_three_l2687_268715

theorem power_sum_of_three : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_three_l2687_268715


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2687_268717

/-- 
For a quadratic equation (k-1)x^2 + 4x + 2 = 0 to have real roots,
k must satisfy the condition k ≤ 3 and k ≠ 1.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 4 * x + 2 = 0) ↔ (k ≤ 3 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2687_268717


namespace NUMINAMATH_CALUDE_janet_waiting_time_l2687_268747

/-- Proves that the waiting time for Janet is 3 hours given the conditions of the problem -/
theorem janet_waiting_time (lake_width : ℝ) (speedboat_speed : ℝ) (sailboat_speed : ℝ)
  (h1 : lake_width = 60)
  (h2 : speedboat_speed = 30)
  (h3 : sailboat_speed = 12) :
  sailboat_speed * (lake_width / speedboat_speed) - lake_width = 3 * speedboat_speed := by
  sorry


end NUMINAMATH_CALUDE_janet_waiting_time_l2687_268747


namespace NUMINAMATH_CALUDE_simplify_expression_l2687_268774

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 1) - 2*b^2 = 9*b^3 - 8*b^2 + 3*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2687_268774


namespace NUMINAMATH_CALUDE_student_hostel_cost_theorem_l2687_268745

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem student_hostel_cost_theorem (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) 
  (h1 : first_week_daily_rate = 18)
  (h2 : total_days = 23)
  (h3 : total_cost = 334) :
  additional_week_cost first_week_daily_rate total_days total_cost = 13 := by
  sorry

end NUMINAMATH_CALUDE_student_hostel_cost_theorem_l2687_268745


namespace NUMINAMATH_CALUDE_expression_value_at_three_l2687_268756

theorem expression_value_at_three :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x + 2
  f 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l2687_268756


namespace NUMINAMATH_CALUDE_roots_equation_problem_l2687_268714

theorem roots_equation_problem (x₁ x₂ m : ℝ) :
  (2 * x₁^2 - 3 * x₁ + m = 0) →
  (2 * x₂^2 - 3 * x₂ + m = 0) →
  (8 * x₁ - 2 * x₂ = 7) →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_problem_l2687_268714


namespace NUMINAMATH_CALUDE_tan_one_condition_l2687_268743

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ x : Real, (∃ k : Int, x = (k * Real.pi) / 4) ∧ Real.tan x ≠ 1) ∧
  (∀ x : Real, Real.tan x = 1 → ∃ k : Int, x = ((4 * k + 1) * Real.pi) / 4) :=
by sorry

end NUMINAMATH_CALUDE_tan_one_condition_l2687_268743


namespace NUMINAMATH_CALUDE_bobs_hair_length_l2687_268773

/-- Calculates the final hair length after a given time period. -/
def final_hair_length (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * 12 * time

/-- Proves that Bob's hair length after 5 years is 36 inches. -/
theorem bobs_hair_length :
  let initial_length : ℝ := 6
  let growth_rate : ℝ := 0.5
  let time : ℝ := 5
  final_hair_length initial_length growth_rate time = 36 := by
  sorry

end NUMINAMATH_CALUDE_bobs_hair_length_l2687_268773


namespace NUMINAMATH_CALUDE_square_property_iff_4_or_100_l2687_268712

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The condition for n to satisfy the square property -/
def is_square_property (n : ℕ+) : Prop :=
  ∃ k : ℕ, (n ^ (d n + 1) * (n + 21) ^ (d n) : ℕ) = k ^ 2

/-- The main theorem -/
theorem square_property_iff_4_or_100 :
  ∀ n : ℕ+, is_square_property n ↔ n = 4 ∨ n = 100 := by sorry

end NUMINAMATH_CALUDE_square_property_iff_4_or_100_l2687_268712


namespace NUMINAMATH_CALUDE_mascs_age_l2687_268798

/-- Given that Masc is 7 years older than Sam and the sum of their ages is 27,
    prove that Masc's age is 17 years old. -/
theorem mascs_age (sam : ℕ) (masc : ℕ) 
    (h1 : masc = sam + 7)
    (h2 : sam + masc = 27) : 
  masc = 17 := by
  sorry

end NUMINAMATH_CALUDE_mascs_age_l2687_268798


namespace NUMINAMATH_CALUDE_prime_solution_equation_l2687_268739

theorem prime_solution_equation : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 → 
    (p = 17 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l2687_268739


namespace NUMINAMATH_CALUDE_geometric_series_equality_l2687_268770

/-- Given real numbers p, q, and r, if the infinite geometric series
    (p/q) + (p/q^2) + (p/q^3) + ... equals 9, then the infinite geometric series
    (p/(p+r)) + (p/(p+r)^2) + (p/(p+r)^3) + ... equals 9(q-1) / (9q + r - 10) -/
theorem geometric_series_equality (p q r : ℝ) 
  (h : ∑' n, p / q^n = 9) :
  ∑' n, p / (p + r)^n = 9 * (q - 1) / (9 * q + r - 10) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l2687_268770


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2687_268781

theorem fourth_grade_students (initial_students : ℕ) : 
  initial_students + 11 - 5 = 37 → initial_students = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2687_268781


namespace NUMINAMATH_CALUDE_sum_of_composite_function_evaluations_l2687_268733

def p (x : ℝ) : ℝ := 2 * |x| - 4

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function_evaluations :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_evaluations_l2687_268733


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2687_268767

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (1/x + 1/(2*y) ≥ 4) ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 2*y' = 1 ∧ 1/x' + 1/(2*y') = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2687_268767


namespace NUMINAMATH_CALUDE_cube_division_equality_l2687_268735

def cube_edge_lengths : List ℕ := List.range 16

def group1 : List ℕ := [1, 4, 6, 7, 10, 11, 13, 16]
def group2 : List ℕ := [2, 3, 5, 8, 9, 12, 14, 15]

def volume (a : ℕ) : ℕ := a^3
def lateral_surface_area (a : ℕ) : ℕ := 4 * a^2
def edge_length (a : ℕ) : ℕ := 12 * a

theorem cube_division_equality :
  (group1.length = group2.length) ∧
  (group1.sum = group2.sum) ∧
  ((group1.map lateral_surface_area).sum = (group2.map lateral_surface_area).sum) ∧
  ((group1.map volume).sum = (group2.map volume).sum) ∧
  ((group1.map edge_length).sum = (group2.map edge_length).sum) :=
by sorry

end NUMINAMATH_CALUDE_cube_division_equality_l2687_268735


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_exists_l2687_268722

theorem egyptian_fraction_sum_exists : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ), 
  (b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧
   b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧
   b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧
   b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧
   b₆ ≠ b₇) ∧
  (11 : ℚ) / 13 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 7 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 8 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 9 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 10 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
by sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_exists_l2687_268722


namespace NUMINAMATH_CALUDE_rectangle_perimeter_120_l2687_268753

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculate the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.length

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with area 864 and width 12 less than length has perimeter 120 -/
theorem rectangle_perimeter_120 (r : Rectangle) 
  (h_area : r.area = 864)
  (h_width : r.width + 12 = r.length) :
  r.perimeter = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_120_l2687_268753


namespace NUMINAMATH_CALUDE_ln_sqrt2_lt_ln3_div3_lt_inv_e_l2687_268716

theorem ln_sqrt2_lt_ln3_div3_lt_inv_e : 
  Real.log (Real.sqrt 2) < Real.log 3 / 3 ∧ Real.log 3 / 3 < 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_sqrt2_lt_ln3_div3_lt_inv_e_l2687_268716


namespace NUMINAMATH_CALUDE_linear_combination_proof_l2687_268790

theorem linear_combination_proof (A B : Matrix (Fin 3) (Fin 3) ℤ) :
  A = ![![2, -4, 0], ![-1, 5, 1], ![0, 3, -7]] →
  B = ![![4, -1, -2], ![0, -3, 5], ![2, 0, -4]] →
  3 • A - 2 • B = ![![-2, -10, 4], ![-3, 21, -7], ![-4, 9, -13]] := by
  sorry

end NUMINAMATH_CALUDE_linear_combination_proof_l2687_268790


namespace NUMINAMATH_CALUDE_bobby_candy_theorem_l2687_268765

def candy_problem (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - first_eaten - second_eaten

theorem bobby_candy_theorem :
  candy_problem 21 5 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_theorem_l2687_268765


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2687_268727

theorem retail_price_calculation (W S P R : ℚ) : 
  W = 99 → 
  S = 0.9 * P → 
  R = 0.2 * W → 
  S = W + R → 
  P = 132 := by
sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l2687_268727


namespace NUMINAMATH_CALUDE_caroline_lassi_production_caroline_lassi_production_proof_l2687_268794

/-- Given that Caroline can make 7 lassis from 3 mangoes, 
    prove that she can make 35 lassis from 15 mangoes. -/
theorem caroline_lassi_production : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mangoes_small lassis_small mangoes_large lassis_large =>
    mangoes_small = 3 ∧ 
    lassis_small = 7 ∧ 
    mangoes_large = 15 ∧
    lassis_large = 35 ∧
    (mangoes_large * lassis_small = mangoes_small * lassis_large) →
    lassis_large = (mangoes_large * lassis_small) / mangoes_small

theorem caroline_lassi_production_proof : 
  caroline_lassi_production 3 7 15 35 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassi_production_caroline_lassi_production_proof_l2687_268794


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l2687_268762

/-- Proves that 470,000,000 is equal to 4.7 × 10^8 in scientific notation -/
theorem scientific_notation_proof :
  (470000000 : ℝ) = 4.7 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l2687_268762


namespace NUMINAMATH_CALUDE_real_roots_condition_l2687_268718

theorem real_roots_condition (p q : ℝ) : 
  (∃ x : ℝ, x^4 + p*x^2 + q = 0) → 65*p^2 ≥ 4*q ∧ 
  ¬(∀ p q : ℝ, 65*p^2 ≥ 4*q → ∃ x : ℝ, x^4 + p*x^2 + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_condition_l2687_268718


namespace NUMINAMATH_CALUDE_speed_from_x_to_y_l2687_268799

/-- Proves that given two towns and specific travel conditions, the speed from x to y is 60 km/hr -/
theorem speed_from_x_to_y (D : ℝ) (V : ℝ) (h : D > 0) : 
  (2 * D) / (D / V + D / 36) = 45 → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_from_x_to_y_l2687_268799


namespace NUMINAMATH_CALUDE_smallest_c_value_l2687_268787

theorem smallest_c_value : ∃ c : ℚ, (∀ x : ℚ, (3 * x + 4) * (x - 2) = 9 * x → c ≤ x) ∧ (3 * c + 4) * (c - 2) = 9 * c ∧ c = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2687_268787


namespace NUMINAMATH_CALUDE_second_month_bill_l2687_268766

/-- Represents Elvin's monthly telephone bill -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill -/
def TelephoneBill.total (bill : TelephoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem second_month_bill
  (firstMonth secondMonth : TelephoneBill)
  (h1 : firstMonth.total = 46)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  secondMonth.total = 76 := by
  sorry

#check second_month_bill

end NUMINAMATH_CALUDE_second_month_bill_l2687_268766


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2687_268726

theorem toms_age_ratio (T N : ℝ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- The sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about ages N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2687_268726


namespace NUMINAMATH_CALUDE_expand_product_l2687_268702

theorem expand_product (x : ℝ) : (x + 2) * (x^2 - 4*x + 1) = x^3 - 2*x^2 - 7*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2687_268702


namespace NUMINAMATH_CALUDE_investor_initial_investment_l2687_268750

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investor's initial investment --/
theorem investor_initial_investment :
  let principal : ℝ := 7000
  let rate : ℝ := 0.10
  let time : ℕ := 2
  let final_amount : ℝ := 8470
  compound_interest principal rate time = final_amount := by
  sorry

end NUMINAMATH_CALUDE_investor_initial_investment_l2687_268750


namespace NUMINAMATH_CALUDE_unicorn_stitches_unicorn_stitches_proof_l2687_268791

/-- Proves that the number of stitches required to embroider a unicorn is 180 --/
theorem unicorn_stitches : ℕ → Prop :=
  fun (unicorn_stitches : ℕ) =>
    let stitches_per_minute : ℕ := 4
    let flower_stitches : ℕ := 60
    let godzilla_stitches : ℕ := 800
    let total_flowers : ℕ := 50
    let total_unicorns : ℕ := 3
    let total_minutes : ℕ := 1085
    let total_stitches : ℕ := total_minutes * stitches_per_minute
    let flower_and_godzilla_stitches : ℕ := total_flowers * flower_stitches + godzilla_stitches
    let remaining_stitches : ℕ := total_stitches - flower_and_godzilla_stitches
    remaining_stitches = total_unicorns * unicorn_stitches → unicorn_stitches = 180

/-- Proof of the theorem --/
theorem unicorn_stitches_proof : unicorn_stitches 180 :=
  by sorry

end NUMINAMATH_CALUDE_unicorn_stitches_unicorn_stitches_proof_l2687_268791


namespace NUMINAMATH_CALUDE_parallel_vectors_theorem_l2687_268713

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def Parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem parallel_vectors_theorem (e₁ e₂ a b : V) (m : ℝ) 
  (h_non_collinear : ¬ Parallel e₁ e₂)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : b = m • e₁ + 3 • e₂)
  (h_parallel : Parallel a b) :
  m = -6 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_theorem_l2687_268713


namespace NUMINAMATH_CALUDE_probability_six_consecutive_heads_l2687_268724

def coin_flips : ℕ := 8

def favorable_outcomes : ℕ := 17

def total_outcomes : ℕ := 2^coin_flips

theorem probability_six_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 17 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_six_consecutive_heads_l2687_268724


namespace NUMINAMATH_CALUDE_twelve_eat_both_l2687_268772

/-- Represents the eating habits in a family -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (habits : FamilyEatingHabits) : ℕ :=
  habits.total_veg - habits.only_veg

/-- Theorem: In the given family, 12 people eat both veg and non-veg -/
theorem twelve_eat_both (habits : FamilyEatingHabits) 
    (h1 : habits.only_veg = 19)
    (h2 : habits.only_non_veg = 9)
    (h3 : habits.total_veg = 31) :
    both_veg_and_non_veg habits = 12 := by
  sorry

#eval both_veg_and_non_veg ⟨19, 9, 31⟩

end NUMINAMATH_CALUDE_twelve_eat_both_l2687_268772


namespace NUMINAMATH_CALUDE_square_field_perimeter_l2687_268706

/-- Given a square field enclosed by posts, calculate the outer perimeter of the fence. -/
theorem square_field_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℝ)
  (gap_between_posts_feet : ℝ)
  (h_num_posts : num_posts = 36)
  (h_post_width : post_width_inches = 6)
  (h_gap_between : gap_between_posts_feet = 6) :
  let post_width_feet : ℝ := post_width_inches / 12
  let side_length : ℝ := (num_posts / 4 - 1) * gap_between_posts_feet + num_posts / 4 * post_width_feet
  let perimeter : ℝ := 4 * side_length
  perimeter = 236 := by
  sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l2687_268706


namespace NUMINAMATH_CALUDE_expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l2687_268764

-- 1. Prove that (a+1)(a+2) = a^2 + 3a + 2
theorem expand_a_plus_one_a_plus_two (a : ℝ) : 
  (a + 1) * (a + 2) = a^2 + 3*a + 2 := by sorry

-- 2. Prove that (3a+b)(3a-b) = 9a^2 - b^2
theorem expand_three_a_plus_b_three_a_minus_b (a b : ℝ) : 
  (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by sorry

-- 3. Prove that 101^2 = 10201
theorem square_of_101 : 
  (101 : ℕ)^2 = 10201 := by sorry

-- 4. Prove that (y+2)(y-2)-(y-1)(y+5) = -4y + 1
theorem expand_and_simplify (y : ℝ) : 
  (y + 2) * (y - 2) - (y - 1) * (y + 5) = -4*y + 1 := by sorry

end NUMINAMATH_CALUDE_expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l2687_268764


namespace NUMINAMATH_CALUDE_exists_unique_polynomial_l2687_268758

/-- Definition of the polynomial p(x, y) -/
def p (x y : ℕ) : ℕ := (x + y)^2 + 3*x + y

/-- Statement of the theorem -/
theorem exists_unique_polynomial :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
by sorry

end NUMINAMATH_CALUDE_exists_unique_polynomial_l2687_268758


namespace NUMINAMATH_CALUDE_final_pen_count_l2687_268737

def pen_collection (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count : pen_collection 20 22 19 = 65 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l2687_268737


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_inclination_l2687_268732

/-- Given a hyperbola mx^2 - y^2 = m where m > 0, if one of its asymptotes has an angle of inclination
    that is twice the angle of inclination of the line x - √3y = 0, then m = 3. -/
theorem hyperbola_asymptote_inclination (m : ℝ) (h1 : m > 0) : 
  (∃ θ : ℝ, θ = 2 * Real.arctan (1 / Real.sqrt 3) ∧ 
             Real.tan θ = Real.sqrt m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_inclination_l2687_268732


namespace NUMINAMATH_CALUDE_smallest_m_for_cube_sum_inequality_l2687_268760

theorem smallest_m_for_cube_sum_inequality :
  ∃ (m : ℝ), m = 27 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧
  (∀ (m' : ℝ), m' < m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_cube_sum_inequality_l2687_268760


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2687_268720

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 105 [MOD 24] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 105 [MOD 24] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2687_268720


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2687_268795

theorem complex_number_magnitude_squared (z : ℂ) (h : z^2 + Complex.abs z ^ 2 = 8 - 2*I) : 
  Complex.abs z ^ 2 = 17/4 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l2687_268795


namespace NUMINAMATH_CALUDE_raise_calculation_l2687_268703

-- Define the original weekly earnings
def original_earnings : ℚ := 60

-- Define the percentage increase
def percentage_increase : ℚ := 33.33 / 100

-- Define the new weekly earnings
def new_earnings : ℚ := original_earnings * (1 + percentage_increase)

-- Theorem to prove
theorem raise_calculation :
  new_earnings = 80 := by sorry

end NUMINAMATH_CALUDE_raise_calculation_l2687_268703


namespace NUMINAMATH_CALUDE_equation_solution_l2687_268740

theorem equation_solution (p q : ℝ) (h : p^2*q = p*q + p^2) : 
  p = 0 ∨ (q ≠ 1 ∧ p = q / (q - 1)) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2687_268740


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2687_268728

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 3*I
  z₁ / z₂ = 9/13 - (19/13)*I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2687_268728


namespace NUMINAMATH_CALUDE_normal_equation_for_given_conditions_l2687_268759

def normal_equation (p : ℝ) (α : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x * Real.cos α + y * Real.sin α - p = 0

theorem normal_equation_for_given_conditions :
  let p : ℝ := 3
  let α₁ : ℝ := π / 4  -- 45°
  let α₂ : ℝ := 7 * π / 4  -- 315°
  (∀ x y, normal_equation p α₁ x y ↔ Real.sqrt 2 / 2 * x + Real.sqrt 2 / 2 * y - 3 = 0) ∧
  (∀ x y, normal_equation p α₂ x y ↔ Real.sqrt 2 / 2 * x - Real.sqrt 2 / 2 * y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_normal_equation_for_given_conditions_l2687_268759


namespace NUMINAMATH_CALUDE_simplify_expression_l2687_268780

theorem simplify_expression : (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2687_268780


namespace NUMINAMATH_CALUDE_cards_given_away_l2687_268710

theorem cards_given_away (original_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : original_cards = 350) 
  (h2 : remaining_cards = 248) : 
  original_cards - remaining_cards = 102 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_l2687_268710


namespace NUMINAMATH_CALUDE_line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l2687_268721

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem 1
theorem line_perp_plane_if_perp_two_intersecting_lines 
  (l : Line) (α : Plane) (m n : Line) :
  contained_in m α → contained_in n α → 
  intersect m n → 
  perpendicular l m → perpendicular l n → 
  perpendicular_line_plane l α :=
sorry

-- Theorem 2
theorem planes_perp_if_line_in_one_perp_other 
  (l : Line) (α β : Plane) :
  contained_in l β → perpendicular_line_plane l α → 
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l2687_268721


namespace NUMINAMATH_CALUDE_angle_is_rational_multiple_of_360_degrees_l2687_268777

/-- A point moving on two intersecting lines -/
structure JumpingPoint where
  angle : ℝ  -- The angle between the lines in radians
  position : ℕ × Bool  -- The position as (jump number, which line)

/-- The condition that the point returns to its starting position -/
def returnsToStart (jp : JumpingPoint) (n : ℕ) : Prop :=
  ∃ k : ℕ, n * jp.angle = k * (2 * Real.pi)

/-- The main theorem -/
theorem angle_is_rational_multiple_of_360_degrees 
  (jp : JumpingPoint) 
  (returns : ∃ n : ℕ, returnsToStart jp n) 
  (h_angle : 0 < jp.angle ∧ jp.angle < 2 * Real.pi) :
  ∃ q : ℚ, jp.angle = q * (2 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_is_rational_multiple_of_360_degrees_l2687_268777


namespace NUMINAMATH_CALUDE_abc_inequality_l2687_268736

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2687_268736


namespace NUMINAMATH_CALUDE_rational_function_value_l2687_268752

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  q_roots : q (-4) = 0 ∧ q 1 = 0
  point_zero : p 0 = 0 ∧ q 0 ≠ 0
  point_neg_one : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 2 / f.q 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l2687_268752


namespace NUMINAMATH_CALUDE_sheela_deposit_l2687_268746

/-- Calculates the deposit amount given a monthly income and deposit percentage -/
def deposit_amount (monthly_income : ℕ) (deposit_percentage : ℚ) : ℚ :=
  (deposit_percentage * monthly_income : ℚ)

theorem sheela_deposit :
  deposit_amount 10000 (25 / 100) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sheela_deposit_l2687_268746


namespace NUMINAMATH_CALUDE_belinda_pages_per_day_l2687_268784

/-- Given that Janet reads 80 pages a day and 2100 more pages than Belinda in 6 weeks,
    prove that Belinda reads 30 pages a day. -/
theorem belinda_pages_per_day :
  let janet_pages_per_day : ℕ := 80
  let weeks : ℕ := 6
  let days_in_week : ℕ := 7
  let extra_pages : ℕ := 2100
  let belinda_pages_per_day : ℕ := 30
  janet_pages_per_day * (weeks * days_in_week) = 
    belinda_pages_per_day * (weeks * days_in_week) + extra_pages :=
by
  sorry

#check belinda_pages_per_day

end NUMINAMATH_CALUDE_belinda_pages_per_day_l2687_268784


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2687_268705

-- Define the lines
def line1 (x y c : ℝ) : Prop := 3 * x - 4 * y = c
def line2 (x y c d : ℝ) : Prop := 8 * x + d * y = -c

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem intersection_of_perpendicular_lines (c d : ℝ) :
  -- Lines are perpendicular
  perpendicular (3/4) (-8/d) →
  -- Lines intersect at (2, -3)
  line1 2 (-3) c →
  line2 2 (-3) c d →
  -- Then c = 18
  c = 18 := by sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2687_268705


namespace NUMINAMATH_CALUDE_special_divisibility_property_l2687_268788

theorem special_divisibility_property (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) :
  (∀ n : ℕ, n > 0 → a^n - n^a ≠ 0 → (a^n - n^a) ∣ (b^n - n^b)) ↔
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = b ∧ a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_special_divisibility_property_l2687_268788


namespace NUMINAMATH_CALUDE_fraction_of_percentages_l2687_268785

theorem fraction_of_percentages (P R M N : ℝ) 
  (hM : M = 0.4 * R)
  (hR : R = 0.25 * P)
  (hN : N = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_percentages_l2687_268785


namespace NUMINAMATH_CALUDE_sum_of_digits_five_pow_eq_two_pow_l2687_268711

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The only natural number n for which the sum of digits of 5^n equals 2^n is 3 -/
theorem sum_of_digits_five_pow_eq_two_pow :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_five_pow_eq_two_pow_l2687_268711


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2687_268755

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  c = Real.sqrt 50 →
  a = 4 →
  b^2 = c^2 - a^2 →
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2687_268755


namespace NUMINAMATH_CALUDE_area_covered_is_56_l2687_268757

/-- The total area covered by five rectangular strips arranged in a specific pattern. -/
def total_area_covered (strip_length : ℝ) (strip_width : ℝ) (center_overlap : ℝ) : ℝ :=
  let single_strip_area := strip_length * strip_width
  let total_area_without_overlap := 5 * single_strip_area
  let center_overlap_area := 4 * (center_overlap * center_overlap)
  let fifth_strip_overlap_area := 2 * (center_overlap * center_overlap)
  total_area_without_overlap - (center_overlap_area + fifth_strip_overlap_area)

/-- Theorem stating that the total area covered by the strips is 56. -/
theorem area_covered_is_56 :
  total_area_covered 8 2 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_area_covered_is_56_l2687_268757


namespace NUMINAMATH_CALUDE_dvd_book_total_capacity_l2687_268776

def dvd_book_capacity (current_dvds : ℕ) (additional_dvds : ℕ) : ℕ :=
  current_dvds + additional_dvds

theorem dvd_book_total_capacity :
  dvd_book_capacity 81 45 = 126 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_total_capacity_l2687_268776


namespace NUMINAMATH_CALUDE_product_increase_l2687_268797

theorem product_increase : ∃ n : ℤ, 
  53 * n = 1585 ∧ 
  53 * n - 35 * n = 535 :=
by sorry

end NUMINAMATH_CALUDE_product_increase_l2687_268797


namespace NUMINAMATH_CALUDE_right_triangle_shortest_leg_l2687_268796

theorem right_triangle_shortest_leg : ∃ (a b : ℕ),
  a < b ∧ a^2 + b^2 = 65^2 ∧ ∀ (x y : ℕ), x < y ∧ x^2 + y^2 = 65^2 → a ≤ x :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_shortest_leg_l2687_268796


namespace NUMINAMATH_CALUDE_diesel_fuel_usage_l2687_268775

/-- Given weekly spending on diesel fuel and cost per gallon, calculates the amount of diesel fuel used in two weeks -/
theorem diesel_fuel_usage
  (weekly_spending : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_spending = 36)
  (h2 : cost_per_gallon = 3)
  : weekly_spending / cost_per_gallon * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_diesel_fuel_usage_l2687_268775


namespace NUMINAMATH_CALUDE_satisfactory_grades_fraction_l2687_268793

theorem satisfactory_grades_fraction :
  let grades := [3, 7, 4, 2, 4]  -- A, B, C, D, E+F
  let satisfactory := 4  -- Number of satisfactory grade categories (A, B, C, D)
  let total_students := grades.sum
  let satisfactory_students := (grades.take satisfactory).sum
  (satisfactory_students : ℚ) / total_students = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_grades_fraction_l2687_268793


namespace NUMINAMATH_CALUDE_custom_op_one_neg_three_l2687_268744

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := 2 * a * b - b^2

-- Theorem statement
theorem custom_op_one_neg_three : custom_op 1 (-3) = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_one_neg_three_l2687_268744


namespace NUMINAMATH_CALUDE_max_participants_is_seven_l2687_268768

/-- Represents a table tennis tournament -/
structure TableTennisTournament where
  participants : ℕ
  scores : Fin participants → Fin participants → Bool
  no_self_play : ∀ i, scores i i = false
  symmetric : ∀ i j, scores i j = !scores j i

/-- The property that for any four participants, two have the same score -/
def has_equal_scores_property (t : TableTennisTournament) : Prop :=
  ∀ (a b c d : Fin t.participants),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ∃ (i j : Fin t.participants), i ≠ j ∧
      (t.scores a i + t.scores b i + t.scores c i + t.scores d i =
       t.scores a j + t.scores b j + t.scores c j + t.scores d j)

/-- The main theorem: maximum number of participants is 7 -/
theorem max_participants_is_seven :
  ∀ t : TableTennisTournament, has_equal_scores_property t → t.participants ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_participants_is_seven_l2687_268768


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2687_268742

theorem completing_square_equivalence (x : ℝ) : x^2 + 4*x - 3 = 0 ↔ (x + 2)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2687_268742


namespace NUMINAMATH_CALUDE_exists_good_not_next_good_l2687_268763

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The function f(n) = n - S(n) where S(n) is the digit sum of n -/
def f (n : ℕ) : ℕ := n - digitSum n

/-- f^k is f applied k times iteratively -/
def fIterate (k : ℕ) : ℕ → ℕ :=
  match k with
  | 0 => id
  | k+1 => f ∘ fIterate k

/-- A number x is k-good if there exists a y such that f^k(y) = x -/
def isGood (k : ℕ) (x : ℕ) : Prop :=
  ∃ y, fIterate k y = x

/-- The main theorem: for all n, there exists an x that is n-good but not (n+1)-good -/
theorem exists_good_not_next_good :
  ∀ n : ℕ, ∃ x : ℕ, isGood n x ∧ ¬isGood (n + 1) x := sorry

end NUMINAMATH_CALUDE_exists_good_not_next_good_l2687_268763


namespace NUMINAMATH_CALUDE_problem_ratio_is_three_to_one_l2687_268729

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems Frank composes for each type -/
def frank_problems_per_type : ℕ := 30

/-- The total number of problems Frank composes -/
def frank_problems : ℕ := frank_problems_per_type * problem_types

/-- The ratio of problems Frank composes to problems Ryan composes -/
def problem_ratio : ℚ := frank_problems / ryan_problems

theorem problem_ratio_is_three_to_one : problem_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_ratio_is_three_to_one_l2687_268729


namespace NUMINAMATH_CALUDE_perpendicular_vectors_dot_product_l2687_268783

/-- Given two vectors m and n in ℝ², where m = (2, 5) and n = (-5, t),
    if m is perpendicular to n, then (m + n) · (m - 2n) = -29 -/
theorem perpendicular_vectors_dot_product (t : ℝ) :
  let m : Fin 2 → ℝ := ![2, 5]
  let n : Fin 2 → ℝ := ![-5, t]
  (m • n = 0) →  -- m is perpendicular to n
  (m + n) • (m - 2 • n) = -29 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_dot_product_l2687_268783


namespace NUMINAMATH_CALUDE_betty_bracelets_l2687_268731

/-- The number of bracelets that can be made given a total number of stones and stones per bracelet -/
def num_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) : ℕ :=
  total_stones / stones_per_bracelet

/-- Theorem: Given 140 stones and 14 stones per bracelet, the number of bracelets is 10 -/
theorem betty_bracelets :
  num_bracelets 140 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_betty_bracelets_l2687_268731


namespace NUMINAMATH_CALUDE_floor_negative_seven_halves_l2687_268769

theorem floor_negative_seven_halves : ⌊(-7 : ℚ) / 2⌋ = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_halves_l2687_268769


namespace NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l2687_268708

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def ray_intersects_sphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (s1 s2 s3 s4 : Sphere) (light_source : Point3D),
    ∀ (r : Ray),
      r.origin = light_source →
      ray_intersects_sphere r s1 ∨
      ray_intersects_sphere r s2 ∨
      ray_intersects_sphere r s3 ∨
      ray_intersects_sphere r s4 :=
sorry

end NUMINAMATH_CALUDE_four_spheres_cover_all_rays_l2687_268708


namespace NUMINAMATH_CALUDE_mike_five_dollar_bills_l2687_268701

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) 
  (h1 : total_amount = 45)
  (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
sorry

end NUMINAMATH_CALUDE_mike_five_dollar_bills_l2687_268701


namespace NUMINAMATH_CALUDE_complex_sum_zero_l2687_268734

theorem complex_sum_zero (b a : ℝ) : 
  let z₁ : ℂ := 2 + b * Complex.I
  let z₂ : ℂ := a + Complex.I
  z₁ + z₂ = 0 → a + b * Complex.I = -2 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l2687_268734


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2687_268707

theorem inequality_system_solution :
  let S := {x : ℝ | 2 * x - 2 > 0 ∧ 3 * (x - 1) - 7 < -2 * x}
  S = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2687_268707


namespace NUMINAMATH_CALUDE_solution_exists_l2687_268748

theorem solution_exists (x y b : ℝ) : 
  (4 * x + 2 * y = b) →
  (3 * x + 7 * y = 3 * b) →
  (x = -1) →
  b = -22 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l2687_268748


namespace NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l2687_268709

/-- The function g(x) defined as x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem stating that -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a ∈ Set.Ioo (-Real.sqrt 24) (Real.sqrt 24) :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_a_in_interval_l2687_268709
