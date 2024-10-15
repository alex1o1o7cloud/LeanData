import Mathlib

namespace NUMINAMATH_CALUDE_function_value_at_pi_sixth_l3720_372025

/-- Given a function f(x) = 3sin(ωx + φ) that satisfies f(π/3 + x) = f(-x) for any x,
    prove that f(π/6) = -3 or f(π/6) = 3 -/
theorem function_value_at_pi_sixth (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -3 ∨ f (π / 6) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_pi_sixth_l3720_372025


namespace NUMINAMATH_CALUDE_extreme_value_condition_decreasing_function_condition_l3720_372039

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 - b*x

-- Theorem for part (1)
theorem extreme_value_condition (a b : ℝ) :
  (∃ x : ℝ, f a b x = 2 ∧ ∀ y : ℝ, f a b y ≤ f a b x) ∧ f a b 1 = 2 →
  a = 1 ∧ b = 3 :=
sorry

-- Theorem for part (2)
theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 2 → f a (9*a) x > f a (9*a) y) →
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_decreasing_function_condition_l3720_372039


namespace NUMINAMATH_CALUDE_prob_two_co_captains_all_teams_l3720_372033

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  students : Nat
  coCaptains : Nat
  h : coCaptains ≤ students

/-- Calculates the probability of choosing two co-captains from a given team -/
def probTwoCoCaptains (team : MathTeam) : Rat :=
  (Nat.choose team.coCaptains 2 : Rat) / (Nat.choose team.students 2 : Rat)

/-- The list of math teams in the area -/
def mathTeams : List MathTeam := [
  ⟨6, 3, by norm_num⟩,
  ⟨9, 2, by norm_num⟩,
  ⟨10, 4, by norm_num⟩
]

theorem prob_two_co_captains_all_teams : 
  (List.sum (mathTeams.map probTwoCoCaptains) / (mathTeams.length : Rat)) = 65 / 540 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_co_captains_all_teams_l3720_372033


namespace NUMINAMATH_CALUDE_solve_system_l3720_372076

theorem solve_system (u v : ℚ) 
  (eq1 : 4 * u - 5 * v = 23)
  (eq2 : 2 * u + 4 * v = -8) :
  u + v = -1 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3720_372076


namespace NUMINAMATH_CALUDE_correct_number_of_selections_l3720_372005

/-- The number of volunteers who only speak Russian -/
def russian_only : ℕ := 3

/-- The number of volunteers who speak both Russian and English -/
def bilingual : ℕ := 4

/-- The total number of volunteers -/
def total_volunteers : ℕ := russian_only + bilingual

/-- The number of English translators to be selected -/
def english_translators : ℕ := 2

/-- The number of Russian translators to be selected -/
def russian_translators : ℕ := 2

/-- The total number of translators to be selected -/
def total_translators : ℕ := english_translators + russian_translators

/-- The function to calculate the number of ways to select translators -/
def num_ways_to_select_translators : ℕ := sorry

/-- Theorem stating that the number of ways to select translators is 60 -/
theorem correct_number_of_selections :
  num_ways_to_select_translators = 60 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_selections_l3720_372005


namespace NUMINAMATH_CALUDE_shifted_quadratic_function_l3720_372021

/-- The original quadratic function -/
def original_function (x : ℝ) : ℝ := x^2

/-- The shifted function -/
def shifted_function (x : ℝ) : ℝ := (x - 3)^2 - 2

/-- Theorem stating that the shifted function is equivalent to shifting the original function -/
theorem shifted_quadratic_function (x : ℝ) : 
  shifted_function x = original_function (x - 3) - 2 := by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_function_l3720_372021


namespace NUMINAMATH_CALUDE_square_area_proof_l3720_372070

theorem square_area_proof (x : ℝ) : 
  (5 * x - 21 = 36 - 4 * x) → 
  (5 * x - 21)^2 = 113.4225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l3720_372070


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2041_l3720_372073

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem units_digit_of_17_pow_2041 : unitsDigit (17^2041) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2041_l3720_372073


namespace NUMINAMATH_CALUDE_rectangle_area_l3720_372082

/-- The area of a rectangle with perimeter 60 and length-to-width ratio 3:2 is 216 -/
theorem rectangle_area (l w : ℝ) : 
  (2 * l + 2 * w = 60) →  -- Perimeter condition
  (l = (3/2) * w) →       -- Length-to-width ratio condition
  (l * w = 216) :=        -- Area calculation
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3720_372082


namespace NUMINAMATH_CALUDE_max_y_value_l3720_372026

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) :
  y ≤ 27 ∧ ∃ (x₀ : ℤ), x₀ * 27 + 7 * x₀ + 6 * 27 = -8 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3720_372026


namespace NUMINAMATH_CALUDE_power_of_two_special_case_l3720_372061

theorem power_of_two_special_case :
  let n : ℝ := 2^(0.15 : ℝ)
  let b : ℝ := 33.333333333333314
  n^b = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_special_case_l3720_372061


namespace NUMINAMATH_CALUDE_figure_can_form_square_l3720_372042

/-- Represents a figure drawn on squared paper -/
structure Figure where
  -- Add necessary fields to represent the figure

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to cut a figure into triangles -/
def cut_into_triangles (f : Figure) : List Triangle :=
  sorry

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut into 5 triangles that form a square -/
theorem figure_can_form_square (f : Figure) :
  ∃ (triangles : List Triangle), 
    cut_into_triangles f = triangles ∧ 
    triangles.length = 5 ∧ 
    can_form_square triangles = true :=
  sorry

end NUMINAMATH_CALUDE_figure_can_form_square_l3720_372042


namespace NUMINAMATH_CALUDE_product_of_differences_l3720_372044

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2023) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2022)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2023) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2022)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2023) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2022)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2023 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l3720_372044


namespace NUMINAMATH_CALUDE_curve_is_ellipse_iff_l3720_372079

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  3 * x^2 + 7 * y^2 - 12 * x + 14 * y = k

/-- The condition for the curve to be a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -19

/-- Theorem stating that the curve is a non-degenerate ellipse iff k > -19 -/
theorem curve_is_ellipse_iff (x y k : ℝ) :
  (∀ x y, curve_equation x y k) ↔ is_non_degenerate_ellipse k :=
sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_iff_l3720_372079


namespace NUMINAMATH_CALUDE_walking_time_is_half_time_saved_l3720_372041

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure HomeCommuteScenario where
  usual_arrival_time : ℕ  -- Time in minutes when they usually arrive home
  early_station_arrival : ℕ  -- Time in minutes the man arrives early at the station
  actual_arrival_time : ℕ  -- Time in minutes when they actually arrive home
  walking_time : ℕ  -- Time in minutes the man spends walking

/-- Theorem stating that the walking time is half of the time saved --/
theorem walking_time_is_half_time_saved (scenario : HomeCommuteScenario) 
  (h1 : scenario.early_station_arrival = 60)
  (h2 : scenario.usual_arrival_time - scenario.actual_arrival_time = 30) :
  scenario.walking_time = (scenario.usual_arrival_time - scenario.actual_arrival_time) / 2 := by
  sorry

#check walking_time_is_half_time_saved

end NUMINAMATH_CALUDE_walking_time_is_half_time_saved_l3720_372041


namespace NUMINAMATH_CALUDE_razor_blade_profit_equation_l3720_372051

theorem razor_blade_profit_equation (x : ℝ) :
  (x ≥ 0) →                          -- number of razors sold is non-negative
  (30 : ℝ) * x +                     -- profit from razors
  (-0.5 : ℝ) * (2 * x) =             -- loss from blades (twice the number of razors)
  (5800 : ℝ)                         -- total profit
  := by sorry

end NUMINAMATH_CALUDE_razor_blade_profit_equation_l3720_372051


namespace NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3720_372034

theorem units_digit_of_7_cubed : (7^3) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_cubed_l3720_372034


namespace NUMINAMATH_CALUDE_triangle_problem_l3720_372068

def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h_triangle : triangle_ABC a b c) 
  (h_angle : Real.cos (π/3) = (b^2 + c^2 - a^2) / (2*b*c))
  (h_sides : a^2 - c^2 = (2/3) * b^2) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = Real.sqrt 3 / 5 ∧
  (1/2 * b * c * Real.sin (π/3) = 3 * Real.sqrt 3 / 4 → a = Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3720_372068


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l3720_372030

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average of marks --/
def average (m : Marks) : ℚ :=
  (m.english + m.mathematics + m.physics + m.biology + m.chemistry : ℚ) / 5

theorem chemistry_marks_proof (m : Marks) 
  (h1 : m.english = 73)
  (h2 : m.mathematics = 69)
  (h3 : m.physics = 92)
  (h4 : m.biology = 82)
  (h5 : average m = 76) :
  m.chemistry = 64 := by
sorry


end NUMINAMATH_CALUDE_chemistry_marks_proof_l3720_372030


namespace NUMINAMATH_CALUDE_ines_shopping_result_l3720_372029

/-- Represents the shopping scenario for Ines at the farmers' market -/
def shopping_scenario (initial_amount : ℚ) (peach_price peach_qty cherry_price cherry_qty
                       baguette_price baguette_qty strawberry_price strawberry_qty
                       salad_price salad_qty : ℚ) : ℚ :=
  let total_cost := peach_price * peach_qty + cherry_price * cherry_qty +
                    baguette_price * baguette_qty + strawberry_price * strawberry_qty +
                    salad_price * salad_qty
  let discount_rate := if total_cost > 10 then 0.1 else 0 +
                       if peach_qty > 0 && cherry_qty > 0 && baguette_qty > 0 &&
                          strawberry_qty > 0 && salad_qty > 0
                       then 0.05 else 0
  let discounted_total := total_cost * (1 - discount_rate)
  let with_tax := discounted_total * 1.05
  let final_total := with_tax * 1.02
  initial_amount - final_total

/-- Theorem stating that Ines will be short by $4.58 after her shopping trip -/
theorem ines_shopping_result :
  shopping_scenario 20 2 3 3.5 2 1.25 4 4 1 2.5 2 = -4.58 := by
  sorry

end NUMINAMATH_CALUDE_ines_shopping_result_l3720_372029


namespace NUMINAMATH_CALUDE_gcd_40304_30203_l3720_372011

theorem gcd_40304_30203 : Nat.gcd 40304 30203 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_40304_30203_l3720_372011


namespace NUMINAMATH_CALUDE_smallest_natural_power_l3720_372081

theorem smallest_natural_power (n : ℕ) : n^(Nat.zero) = 1 := by
  sorry

#check smallest_natural_power 2009

end NUMINAMATH_CALUDE_smallest_natural_power_l3720_372081


namespace NUMINAMATH_CALUDE_min_sum_given_product_min_sum_value_l3720_372032

theorem min_sum_given_product (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∀ a b : ℝ, a * b = 4 ∧ a > 0 ∧ b > 0 → x + y ≤ a + b :=
by
  sorry

theorem min_sum_value (x y : ℝ) (h1 : x * y = 4) (h2 : x > 0) (h3 : y > 0) :
  ∃ M : ℝ, M = 4 ∧ x + y ≥ M :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_min_sum_value_l3720_372032


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_p_l3720_372009

/-- A polynomial of the form x^4 + px^2 + qx - 144 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  p : ℝ
  q : ℝ
  roots : Fin 4 → ℝ
  distinct_roots : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + p * (roots i)^2 + q * (roots i) - 144 = 0

/-- The value of p in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_p (poly : ArithmeticProgressionPolynomial) : poly.p = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_p_l3720_372009


namespace NUMINAMATH_CALUDE_third_month_sale_l3720_372031

def sales_data : List ℕ := [8435, 8927, 9230, 8562, 6991]
def average_sale : ℕ := 8500
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ x : ℕ, 
    (List.sum sales_data + x) / num_months = average_sale ∧
    x = 8855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l3720_372031


namespace NUMINAMATH_CALUDE_minute_hand_angle_half_hour_l3720_372052

/-- The angle traversed by the minute hand of a clock in a given time period -/
def minute_hand_angle (time_period : ℚ) : ℚ :=
  360 * time_period

theorem minute_hand_angle_half_hour :
  minute_hand_angle (1/2) = 180 := by
  sorry

#check minute_hand_angle_half_hour

end NUMINAMATH_CALUDE_minute_hand_angle_half_hour_l3720_372052


namespace NUMINAMATH_CALUDE_problem_statement_l3720_372022

theorem problem_statement (P Q : ℝ) 
  (h1 : P^2 - P*Q = 1) 
  (h2 : 4*P*Q - 3*Q^2 = 2) : 
  P^2 + 3*P*Q - 3*Q^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3720_372022


namespace NUMINAMATH_CALUDE_miriam_initial_marbles_l3720_372048

/-- The number of marbles Miriam initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Miriam gave to her brother -/
def marbles_to_brother : ℕ := 60

/-- The number of marbles Miriam gave to her sister -/
def marbles_to_sister : ℕ := 2 * marbles_to_brother

/-- The number of marbles Miriam currently has -/
def current_marbles : ℕ := 300

/-- The number of marbles Miriam gave to her friend Savanna -/
def marbles_to_savanna : ℕ := 3 * current_marbles

theorem miriam_initial_marbles :
  initial_marbles = marbles_to_brother + marbles_to_sister + marbles_to_savanna + current_marbles :=
by sorry

end NUMINAMATH_CALUDE_miriam_initial_marbles_l3720_372048


namespace NUMINAMATH_CALUDE_logarithm_properties_l3720_372002

variable (a x y : ℝ)

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem logarithm_properties (h : a > 1) :
  (log a 1 = 0) ∧
  (log a a = 1) ∧
  (∀ x > 0, x < 1 → log a x < 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → log a x < -1/ε) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_properties_l3720_372002


namespace NUMINAMATH_CALUDE_squares_below_line_l3720_372069

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The line 8x + 245y = 1960 --/
def problemLine : Line := { a := 8, b := 245, c := 1960 }

theorem squares_below_line :
  countPointsBelowLine problemLine = 853 :=
sorry

end NUMINAMATH_CALUDE_squares_below_line_l3720_372069


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3720_372045

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3720_372045


namespace NUMINAMATH_CALUDE_victor_finished_last_l3720_372098

-- Define the set of runners
inductive Runner : Type
| Lotar : Runner
| Manfred : Runner
| Victor : Runner
| Jan : Runner
| Eddy : Runner

-- Define the relation "finished before"
def finished_before : Runner → Runner → Prop := sorry

-- State the conditions
axiom lotar_before_manfred : finished_before Runner.Lotar Runner.Manfred
axiom victor_after_jan : finished_before Runner.Jan Runner.Victor
axiom manfred_before_jan : finished_before Runner.Manfred Runner.Jan
axiom eddy_before_victor : finished_before Runner.Eddy Runner.Victor

-- Define what it means to finish last
def finished_last (r : Runner) : Prop :=
  ∀ other : Runner, other ≠ r → finished_before other r

-- State the theorem
theorem victor_finished_last :
  finished_last Runner.Victor :=
sorry

end NUMINAMATH_CALUDE_victor_finished_last_l3720_372098


namespace NUMINAMATH_CALUDE_frog_hops_l3720_372099

theorem frog_hops (frog1 frog2 frog3 : ℕ) : 
  frog1 = 4 * frog2 →
  frog2 = 2 * frog3 →
  frog2 = 18 →
  frog1 + frog2 + frog3 = 99 := by
sorry

end NUMINAMATH_CALUDE_frog_hops_l3720_372099


namespace NUMINAMATH_CALUDE_polynomial_interpolation_l3720_372053

def p (x : ℝ) : ℝ := x^4 + x^3 - 3*x^2 - 2*x + 2

theorem polynomial_interpolation :
  (p (-2) = 2) ∧
  (p (-1) = 1) ∧
  (p 0 = 2) ∧
  (p 1 = -1) ∧
  (p 2 = 10) ∧
  (∀ q : ℝ → ℝ, (q (-2) = 2) ∧ (q (-1) = 1) ∧ (q 0 = 2) ∧ (q 1 = -1) ∧ (q 2 = 10) →
    (∃ a b c d e : ℝ, ∀ x, q x = a*x^4 + b*x^3 + c*x^2 + d*x + e) →
    (∀ x, q x = p x)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_interpolation_l3720_372053


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3720_372020

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

def r₁ : ℝ := linear_correlation_coefficient X Y
def r₂ : ℝ := linear_correlation_coefficient U V

theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3720_372020


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l3720_372089

open Real

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * log x = log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l3720_372089


namespace NUMINAMATH_CALUDE_sandras_sock_purchase_l3720_372038

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≤ 6 ∧ p.three_dollar ≤ 6 ∧ p.five_dollar ≤ 6

/-- Theorem stating that the only valid purchase has 11 pairs of $2 socks --/
theorem sandras_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandras_sock_purchase_l3720_372038


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3720_372064

theorem absolute_value_inequality (x : ℝ) :
  (4 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (2 ≤ x ∧ x ≤ 6)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3720_372064


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3720_372065

/-- The number of diagonals in a regular polygon with n sides -/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : numDiagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3720_372065


namespace NUMINAMATH_CALUDE_stone_123_l3720_372095

/-- The number of stones in the sequence -/
def num_stones : ℕ := 15

/-- The number of counts in a complete cycle -/
def cycle_length : ℕ := 29

/-- The count we're interested in -/
def target_count : ℕ := 123

/-- The function that maps a count to its corresponding initial stone number -/
def count_to_stone (count : ℕ) : ℕ :=
  (count % cycle_length) % num_stones + 1

theorem stone_123 : count_to_stone target_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_stone_123_l3720_372095


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_maximum_marks_is_750_l3720_372092

theorem maximum_marks_calculation (passing_percentage : ℝ) (student_score : ℕ) (shortfall : ℕ) : ℝ :=
  let passing_score : ℕ := student_score + shortfall
  let maximum_marks : ℝ := passing_score / passing_percentage
  maximum_marks

-- Proof that the maximum marks is 750 given the conditions
theorem maximum_marks_is_750 :
  maximum_marks_calculation 0.3 212 13 = 750 :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_maximum_marks_is_750_l3720_372092


namespace NUMINAMATH_CALUDE_min_area_of_B_l3720_372000

-- Define set A
def A : Set (ℝ × ℝ) := {p | |p.1 - 2| + |p.2 - 3| ≤ 1}

-- Define set B
def B (D E F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0}

-- State the theorem
theorem min_area_of_B (D E F : ℝ) (h1 : D^2 + E^2 - 4*F > 0) (h2 : A ⊆ B D E F) :
  ∃ (S : ℝ), S = 2 * Real.pi ∧ ∀ (S' : ℝ), (∃ (D' E' F' : ℝ), D'^2 + E'^2 - 4*F' > 0 ∧ A ⊆ B D' E' F' ∧ S' = Real.pi * ((D'^2 + E'^2) / 4 - F')) → S ≤ S' :=
sorry

end NUMINAMATH_CALUDE_min_area_of_B_l3720_372000


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3720_372091

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 6 < 0 ↔ -3 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3720_372091


namespace NUMINAMATH_CALUDE_combined_girls_average_l3720_372023

/-- Represents a high school with given average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two high schools -/
structure CombinedSchools where
  cedar : School
  delta : School
  boys_combined_avg : ℝ

/-- Theorem stating that the combined girls' average is 86 -/
theorem combined_girls_average (schools : CombinedSchools) 
  (h1 : schools.cedar.boys_avg = 85)
  (h2 : schools.cedar.girls_avg = 80)
  (h3 : schools.cedar.combined_avg = 83)
  (h4 : schools.delta.boys_avg = 76)
  (h5 : schools.delta.girls_avg = 95)
  (h6 : schools.delta.combined_avg = 87)
  (h7 : schools.boys_combined_avg = 73) :
  ∃ (cedar_boys cedar_girls delta_boys delta_girls : ℝ),
    cedar_boys > 0 ∧ cedar_girls > 0 ∧ delta_boys > 0 ∧ delta_girls > 0 ∧
    (cedar_boys * 85 + cedar_girls * 80) / (cedar_boys + cedar_girls) = 83 ∧
    (delta_boys * 76 + delta_girls * 95) / (delta_boys + delta_girls) = 87 ∧
    (cedar_boys * 85 + delta_boys * 76) / (cedar_boys + delta_boys) = 73 ∧
    (cedar_girls * 80 + delta_girls * 95) / (cedar_girls + delta_girls) = 86 :=
by sorry


end NUMINAMATH_CALUDE_combined_girls_average_l3720_372023


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3720_372067

/-- Theorem: In an election with 3 candidates, where one candidate received 71.42857142857143% 
    of the total votes, and the other two candidates received 3000 and 5000 votes respectively, 
    the winning candidate received 20,000 votes. -/
theorem election_votes_theorem : 
  let total_votes : ℝ := (20000 + 3000 + 5000 : ℝ)
  let winning_percentage : ℝ := 71.42857142857143
  let other_votes_1 : ℝ := 3000
  let other_votes_2 : ℝ := 5000
  let winning_votes : ℝ := 20000
  (winning_votes / total_votes) * 100 = winning_percentage ∧
  winning_votes + other_votes_1 + other_votes_2 = total_votes :=
by
  sorry

#check election_votes_theorem

end NUMINAMATH_CALUDE_election_votes_theorem_l3720_372067


namespace NUMINAMATH_CALUDE_no_solution_in_naturals_l3720_372024

theorem no_solution_in_naturals :
  ∀ (x y z t : ℕ), (15^x + 29^y + 43^z) % 7 ≠ (t^2) % 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_naturals_l3720_372024


namespace NUMINAMATH_CALUDE_min_participants_l3720_372028

/-- Represents a single-round robin tournament --/
structure Tournament where
  participants : ℕ
  matches_per_player : ℕ
  winner_wins : ℕ

/-- Conditions for the tournament --/
def valid_tournament (t : Tournament) : Prop :=
  t.participants > 1 ∧
  t.matches_per_player = t.participants - 1 ∧
  (t.winner_wins : ℝ) / t.matches_per_player > 0.68 ∧
  (t.winner_wins : ℝ) / t.matches_per_player < 0.69

/-- The theorem to be proved --/
theorem min_participants (t : Tournament) (h : valid_tournament t) :
  t.participants ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_min_participants_l3720_372028


namespace NUMINAMATH_CALUDE_gcd_45045_30030_l3720_372087

theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45045_30030_l3720_372087


namespace NUMINAMATH_CALUDE_foci_of_hyperbola_l3720_372074

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 7 - y^2 / 3 = 1

/-- The foci coordinates -/
def foci_coordinates : Set (ℝ × ℝ) := {(Real.sqrt 10, 0), (-Real.sqrt 10, 0)}

/-- Theorem: The given coordinates are the foci of the hyperbola -/
theorem foci_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci_coordinates ↔ 
    ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
      (x - x')^2 + (y - y')^2 = ((Real.sqrt 10 + x')^2 + y'^2).sqrt * 
                                ((Real.sqrt 10 - x')^2 + y'^2).sqrt :=
sorry

end NUMINAMATH_CALUDE_foci_of_hyperbola_l3720_372074


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l3720_372027

/-- Calculates the average speed of a cyclist given two trips with different distances and speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 12) (h3 : v1 = 12) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  ∃ ε > 0, |average_speed - 10.1| < ε :=
by sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l3720_372027


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l3720_372057

-- Define the coupon savings functions
def couponA (price : ℝ) : ℝ := 0.18 * price
def couponB : ℝ := 35
def couponC (price : ℝ) : ℝ := 0.20 * (price - 120)

-- Define the conditions for Coupon A to be at least as good as B and C
def couponABestCondition (price : ℝ) : Prop :=
  couponA price ≥ couponB ∧ couponA price ≥ couponC price

-- Define the range of prices where Coupon A is the best
def priceRange : Set ℝ := {price | price > 120 ∧ couponABestCondition price}

-- Theorem statement
theorem coupon_savings_difference :
  ∃ (x y : ℝ), x ∈ priceRange ∧ y ∈ priceRange ∧
  (∀ p ∈ priceRange, x ≤ p ∧ p ≤ y) ∧
  y - x = 1005.56 :=
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l3720_372057


namespace NUMINAMATH_CALUDE_square_table_capacity_square_table_capacity_proof_l3720_372056

theorem square_table_capacity (rectangular_tables : ℕ) (rectangular_capacity : ℕ) 
  (square_tables : ℕ) (total_pupils : ℕ) : ℕ :=
  let remaining_pupils := total_pupils - rectangular_tables * rectangular_capacity
  remaining_pupils / square_tables

#check square_table_capacity 7 10 5 90 = 4

theorem square_table_capacity_proof 
  (h1 : rectangular_tables = 7)
  (h2 : rectangular_capacity = 10)
  (h3 : square_tables = 5)
  (h4 : total_pupils = 90) :
  square_table_capacity rectangular_tables rectangular_capacity square_tables total_pupils = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_table_capacity_square_table_capacity_proof_l3720_372056


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3720_372063

-- Define propositions p and q
def p (x : ℝ) : Prop := 5 * x - 6 ≥ x^2
def q (x : ℝ) : Prop := |x + 1| > 2

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l3720_372063


namespace NUMINAMATH_CALUDE_other_metal_price_l3720_372043

/-- Given the price of Metal A, the ratio of Metal A to another metal, and the cost of their alloy,
    this theorem proves the price of the other metal. -/
theorem other_metal_price
  (price_a : ℝ)
  (ratio : ℝ)
  (alloy_cost : ℝ)
  (h1 : price_a = 68)
  (h2 : ratio = 3)
  (h3 : alloy_cost = 75) :
  (4 * alloy_cost - 3 * price_a) = 96 := by
  sorry

end NUMINAMATH_CALUDE_other_metal_price_l3720_372043


namespace NUMINAMATH_CALUDE_triangle_area_l3720_372054

/-- Given a triangle with one side of length 14 units, the angle opposite to this side
    being 60 degrees, and the ratio of the other two sides being 8:5,
    prove that the area of the triangle is 40√3 square units. -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) :
  a = 14 →
  θ = 60 * π / 180 →
  b / c = 8 / 5 →
  (1 / 2) * b * c * Real.sin θ = 40 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3720_372054


namespace NUMINAMATH_CALUDE_fullPriceRevenue_is_600_l3720_372096

/-- Represents the fundraiser event ticket sales -/
structure FundraiserEvent where
  totalTickets : ℕ
  totalRevenue : ℕ
  fullPrice : ℕ
  halfPrice : ℕ
  fullPriceRevenue : ℕ

/-- The fundraiser event satisfies the given conditions -/
def validFundraiserEvent (e : FundraiserEvent) : Prop :=
  e.totalTickets = 200 ∧
  e.totalRevenue = 2700 ∧
  e.fullPrice > 0 ∧
  e.halfPrice = e.fullPrice / 2 ∧
  e.totalTickets = (e.totalRevenue - e.fullPriceRevenue) / e.halfPrice + e.fullPriceRevenue / e.fullPrice

/-- The theorem stating that the full-price ticket revenue is $600 -/
theorem fullPriceRevenue_is_600 (e : FundraiserEvent) (h : validFundraiserEvent e) : 
  e.fullPriceRevenue = 600 := by
  sorry

end NUMINAMATH_CALUDE_fullPriceRevenue_is_600_l3720_372096


namespace NUMINAMATH_CALUDE_binary_sum_equals_result_l3720_372049

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 11011₂ -/
def num1 : List Bool := [true, true, false, true, true]

/-- Represents the binary number 1010₂ -/
def num2 : List Bool := [true, false, true, false]

/-- Represents the binary number 11100₂ -/
def num3 : List Bool := [true, true, true, false, false]

/-- Represents the binary number 1001₂ -/
def num4 : List Bool := [true, false, false, true]

/-- Represents the binary number 100010₂ (the expected result) -/
def result : List Bool := [true, false, false, false, true, false]

/-- The main theorem stating that the sum of the binary numbers equals the expected result -/
theorem binary_sum_equals_result :
  binaryToNat num1 + binaryToNat num2 - binaryToNat num3 + binaryToNat num4 = binaryToNat result :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_result_l3720_372049


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l3720_372017

theorem inequality_system_solution_range (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 4 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - m < 0 ∧ 7 - 2*x ≤ 1))) →
  (6 < m ∧ m ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l3720_372017


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3720_372014

theorem halfway_between_fractions : 
  let a := (1 : ℚ) / 6
  let b := (1 : ℚ) / 12
  let midpoint := (a + b) / 2
  midpoint = (1 : ℚ) / 8 := by sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3720_372014


namespace NUMINAMATH_CALUDE_max_values_for_constrained_expressions_l3720_372066

theorem max_values_for_constrained_expressions (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1) :
  (∃ (max_ab : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → x * y ≤ max_ab) ∧
  (∃ (max_sqrt : ℝ), ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 → Real.sqrt x + Real.sqrt y ≤ max_sqrt) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x^2 + y^2 > M) ∧
  (∀ (M : ℝ), ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + 4/y > M) :=
by sorry

end NUMINAMATH_CALUDE_max_values_for_constrained_expressions_l3720_372066


namespace NUMINAMATH_CALUDE_expression_value_for_x_2_l3720_372047

theorem expression_value_for_x_2 : 
  let x : ℝ := 2
  (x + x * (x * x)) = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_value_for_x_2_l3720_372047


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3720_372071

/-- The distance between the foci of an ellipse defined by 25x^2 - 100x + 4y^2 + 8y + 36 = 0 -/
theorem ellipse_foci_distance : 
  let ellipse_eq := fun (x y : ℝ) => 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36
  ∃ (h k a b : ℝ), 
    (∀ x y, ellipse_eq x y = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) ∧
    2 * Real.sqrt (a^2 - b^2) = 2 * Real.sqrt 14.28 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3720_372071


namespace NUMINAMATH_CALUDE_imaginary_number_product_l3720_372036

theorem imaginary_number_product (z : ℂ) (a : ℝ) : 
  (z.im ≠ 0 ∧ z.re = 0) → (Complex.I * z.im = z) → ((3 - Complex.I) * z = a + Complex.I) → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_number_product_l3720_372036


namespace NUMINAMATH_CALUDE_m_range_l3720_372080

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the set of x values satisfying p
def A : Set ℝ := {x | p x}

-- Define the set of x values satisfying q
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range :
  (∀ m : ℝ, m > 0 → (A ⊂ B m) ∧ (A ≠ B m)) →
  {m : ℝ | 0 < m ∧ m ≤ 2} = {m : ℝ | ∃ x, q x m ∧ ¬p x} :=
sorry

end NUMINAMATH_CALUDE_m_range_l3720_372080


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l3720_372077

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def num_cubes_passed (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that the number of unit cubes an internal diagonal passes through
    in a 200 × 300 × 450 rectangular solid is 700 -/
theorem internal_diagonal_cubes :
  num_cubes_passed 200 300 450 = 700 := by sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l3720_372077


namespace NUMINAMATH_CALUDE_reflection_of_circle_center_l3720_372040

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_diagonal (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The original center of the circle -/
def original_center : ℝ × ℝ := (8, -3)

/-- The expected center of the reflected circle -/
def expected_reflected_center : ℝ × ℝ := (3, -8)

theorem reflection_of_circle_center :
  reflect_about_diagonal original_center = expected_reflected_center := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_circle_center_l3720_372040


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3720_372078

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1021 + x) % 25 = 0 ∧ 
  ∀ (y : ℕ), (1021 + y) % 25 = 0 → x ≤ y :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3720_372078


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3720_372037

theorem sqrt_calculation : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3720_372037


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3720_372019

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3720_372019


namespace NUMINAMATH_CALUDE_volume_of_specific_parallelepiped_l3720_372007

/-- A rectangular parallelepiped with vertices A B C D A₁ B₁ C₁ D₁ -/
structure RectangularParallelepiped where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- A plane passing through vertices A, C, and D₁ of the parallelepiped -/
structure DiagonalPlane where
  parallelepiped : RectangularParallelepiped
  dihedral_angle : ℝ

/-- The volume of a rectangular parallelepiped -/
def volume (p : RectangularParallelepiped) : ℝ :=
  p.base_length * p.base_width * p.height

/-- Theorem: Volume of the specific parallelepiped -/
theorem volume_of_specific_parallelepiped (p : RectangularParallelepiped) 
  (d : DiagonalPlane) (h1 : p.base_length = 4) (h2 : p.base_width = 3) 
  (h3 : d.parallelepiped = p) (h4 : d.dihedral_angle = π / 3) :
  volume p = (144 * Real.sqrt 3) / 5 := by
  sorry


end NUMINAMATH_CALUDE_volume_of_specific_parallelepiped_l3720_372007


namespace NUMINAMATH_CALUDE_image_of_square_l3720_372093

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^3 - y^3, x^2 * y^2)

/-- Square OABC in xy-plane -/
def square_vertices : List (ℝ × ℝ) :=
  [(0, 0), (2, 0), (2, 2), (0, 2)]

/-- Theorem: Image of square OABC in uv-plane -/
theorem image_of_square :
  (square_vertices.map (λ (x, y) => transform x y)) =
  [(0, 0), (8, 0), (0, 16), (-8, 0)] := by
  sorry


end NUMINAMATH_CALUDE_image_of_square_l3720_372093


namespace NUMINAMATH_CALUDE_exists_equivalent_expr_l3720_372018

/-- Represents the two possible binary operations in our system -/
inductive Op
| add
| sub

/-- Represents an expression in our system -/
inductive Expr
| var : String → Expr
| op : Op → Expr → Expr → Expr

/-- Evaluates an expression given an assignment of values to variables and a mapping of symbols to operations -/
def evaluate (e : Expr) (vars : String → ℝ) (sym_to_op : Op → Op) : ℝ :=
  match e with
  | Expr.var v => vars v
  | Expr.op o e1 e2 =>
    let v1 := evaluate e1 vars sym_to_op
    let v2 := evaluate e2 vars sym_to_op
    match sym_to_op o with
    | Op.add => v1 + v2
    | Op.sub => v1 - v2

/-- The theorem to be proved -/
theorem exists_equivalent_expr :
  ∃ (e : Expr),
    ∀ (vars : String → ℝ) (sym_to_op : Op → Op),
      evaluate e vars sym_to_op = 20 * vars "a" - 18 * vars "b" :=
sorry

end NUMINAMATH_CALUDE_exists_equivalent_expr_l3720_372018


namespace NUMINAMATH_CALUDE_cone_base_radius_l3720_372094

/-- The base radius of a cone with height 4 and volume 4π is √3 -/
theorem cone_base_radius (h : ℝ) (V : ℝ) (r : ℝ) :
  h = 4 → V = 4 * Real.pi → V = (1/3) * Real.pi * r^2 * h → r = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3720_372094


namespace NUMINAMATH_CALUDE_abc_equality_l3720_372050

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (1 - b) = 1/4) (h2 : b * (1 - c) = 1/4) (h3 : c * (1 - a) = 1/4) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_abc_equality_l3720_372050


namespace NUMINAMATH_CALUDE_five_wednesdays_theorem_l3720_372046

/-- The year of the Gregorian calendar reform -/
def gregorian_reform_year : ℕ := 1752

/-- The cycle length for years with 5 Wednesdays in February -/
def cycle_length : ℕ := 28

/-- The reference year with 5 Wednesdays in February -/
def reference_year : ℕ := 1928

/-- Predicate to check if a year has 5 Wednesdays in February -/
def has_five_wednesdays (year : ℕ) : Prop :=
  (year ≥ gregorian_reform_year) ∧ 
  (year = reference_year ∨ (year - reference_year) % cycle_length = 0)

/-- The nearest year before the reference year with 5 Wednesdays in February -/
def nearest_before : ℕ := 1888

/-- The nearest year after the reference year with 5 Wednesdays in February -/
def nearest_after : ℕ := 1956

theorem five_wednesdays_theorem :
  (has_five_wednesdays nearest_before) ∧
  (has_five_wednesdays nearest_after) ∧
  (∀ y : ℕ, nearest_before < y ∧ y < reference_year → ¬(has_five_wednesdays y)) ∧
  (∀ y : ℕ, reference_year < y ∧ y < nearest_after → ¬(has_five_wednesdays y)) :=
sorry

end NUMINAMATH_CALUDE_five_wednesdays_theorem_l3720_372046


namespace NUMINAMATH_CALUDE_x_value_proof_l3720_372075

theorem x_value_proof (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3720_372075


namespace NUMINAMATH_CALUDE_two_digit_multiple_of_eight_l3720_372012

theorem two_digit_multiple_of_eight (A : Nat) : 
  (30 ≤ 10 * 3 + A) ∧ (10 * 3 + A < 40) ∧ (10 * 3 + A) % 8 = 0 → A = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiple_of_eight_l3720_372012


namespace NUMINAMATH_CALUDE_rectangular_field_ratio_l3720_372013

theorem rectangular_field_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a < b) : 
  (a + b = 3 * a) → 
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → 
  a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_ratio_l3720_372013


namespace NUMINAMATH_CALUDE_x_value_proof_l3720_372003

theorem x_value_proof (x : ℚ) 
  (h1 : 6 * x^2 + 5 * x - 1 = 0) 
  (h2 : 18 * x^2 + 17 * x - 1 = 0) : 
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l3720_372003


namespace NUMINAMATH_CALUDE_sample_in_range_l3720_372055

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (total / sampleSize) * n

/-- Theorem: The sample in the range [37, 54] is 42 -/
theorem sample_in_range (total : ℕ) (sampleSize : ℕ) (start : ℕ) :
  total = 900 →
  sampleSize = 50 →
  start = 6 →
  ∃ n : ℕ, 
    37 ≤ systematicSample total sampleSize start n ∧ 
    systematicSample total sampleSize start n ≤ 54 ∧
    systematicSample total sampleSize start n = 42 :=
by
  sorry


end NUMINAMATH_CALUDE_sample_in_range_l3720_372055


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3720_372015

theorem ceiling_fraction_evaluation : 
  (⌈⌈(23:ℝ)/9 - ⌈(35:ℝ)/21⌉⌉⌉ : ℝ) / ⌈⌈(36:ℝ)/9 + ⌈(9:ℝ)*23/36⌉⌉⌉ = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l3720_372015


namespace NUMINAMATH_CALUDE_unique_prime_plus_10_14_prime_l3720_372010

theorem unique_prime_plus_10_14_prime :
  ∃! p : ℕ, Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_plus_10_14_prime_l3720_372010


namespace NUMINAMATH_CALUDE_power_function_with_specific_point_is_odd_l3720_372004

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_with_specific_point_is_odd
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_point : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  isOddFunction f :=
sorry

end NUMINAMATH_CALUDE_power_function_with_specific_point_is_odd_l3720_372004


namespace NUMINAMATH_CALUDE_discount_problem_l3720_372008

theorem discount_problem (x y : ℝ) : 
  (100 - x / 100 * 100) * (1 - y / 100) = 55 →
  (100 - 55) / 100 * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l3720_372008


namespace NUMINAMATH_CALUDE_set_operations_l3720_372090

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3720_372090


namespace NUMINAMATH_CALUDE_y_value_proof_l3720_372097

theorem y_value_proof : ∀ y : ℚ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3720_372097


namespace NUMINAMATH_CALUDE_tan_theta_value_l3720_372059

theorem tan_theta_value (θ : Real) 
  (h : (1 + Real.sin (2 * θ)) / (Real.cos θ ^ 2 - Real.sin θ ^ 2) = -3) : 
  Real.tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3720_372059


namespace NUMINAMATH_CALUDE_cost_of_45_roses_l3720_372016

/-- The cost of a bouquet of roses with discount applied -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_cost := 30 * (n / 15 : ℚ)
  if n > 30 then base_cost * (1 - 1/10) else base_cost

/-- Theorem stating the cost of a bouquet with 45 roses -/
theorem cost_of_45_roses : bouquet_cost 45 = 81 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_45_roses_l3720_372016


namespace NUMINAMATH_CALUDE_morning_campers_count_l3720_372088

/-- Given a total number of campers and a ratio for morning:afternoon:evening,
    calculate the number of campers who went rowing in the morning. -/
def campers_in_morning (total : ℕ) (morning_ratio afternoon_ratio evening_ratio : ℕ) : ℕ :=
  let total_ratio := morning_ratio + afternoon_ratio + evening_ratio
  let part_size := total / total_ratio
  morning_ratio * part_size

/-- Theorem stating that given 60 total campers and a ratio of 3:2:4,
    the number of campers who went rowing in the morning is 18. -/
theorem morning_campers_count :
  campers_in_morning 60 3 2 4 = 18 := by
  sorry

#eval campers_in_morning 60 3 2 4

end NUMINAMATH_CALUDE_morning_campers_count_l3720_372088


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3720_372084

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : x * y = 1) : 
  (5 * x + 3) - (2 * x * y - 5 * y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3720_372084


namespace NUMINAMATH_CALUDE_distribute_seven_balls_to_three_people_l3720_372062

/-- The number of ways to distribute n identical balls to k people, 
    with each person getting at least 1 ball -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 15 ways to distribute 7 identical balls to 3 people, 
    with each person getting at least 1 ball -/
theorem distribute_seven_balls_to_three_people : 
  distribute_balls 7 3 = 15 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_to_three_people_l3720_372062


namespace NUMINAMATH_CALUDE_max_pyramid_volume_is_four_l3720_372035

/-- A triangular prism with given ratios on its lateral edges -/
structure TriangularPrism where
  volume : ℝ
  AM_ratio : ℝ
  BN_ratio : ℝ
  CK_ratio : ℝ

/-- The maximum volume of a pyramid formed inside the prism -/
def max_pyramid_volume (prism : TriangularPrism) : ℝ := sorry

/-- Theorem stating the maximum volume of the pyramid MNKP -/
theorem max_pyramid_volume_is_four (prism : TriangularPrism) 
  (h1 : prism.volume = 16)
  (h2 : prism.AM_ratio = 1/2)
  (h3 : prism.BN_ratio = 1/3)
  (h4 : prism.CK_ratio = 1/4) :
  max_pyramid_volume prism = 4 := by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_is_four_l3720_372035


namespace NUMINAMATH_CALUDE_limit_of_rational_function_l3720_372001

theorem limit_of_rational_function (f : ℝ → ℝ) (h : ∀ x ≠ 1, f x = (x^4 - 1) / (2*x^4 - x^2 - 1)) :
  ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |f x - 2/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_rational_function_l3720_372001


namespace NUMINAMATH_CALUDE_jennifers_spending_l3720_372058

theorem jennifers_spending (initial_amount : ℚ) : 
  initial_amount / 5 + initial_amount / 6 + initial_amount / 2 + 20 = initial_amount →
  initial_amount = 150 := by
  sorry

end NUMINAMATH_CALUDE_jennifers_spending_l3720_372058


namespace NUMINAMATH_CALUDE_eleventh_term_value_l3720_372086

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 15 terms is 56.25
  sum_15_terms : (15 / 2 : ℝ) * (2 * a + 14 * d) = 56.25
  -- 7th term is 3.25
  term_7 : a + 6 * d = 3.25

/-- Theorem: The 11th term of the specified arithmetic progression is 5.25 -/
theorem eleventh_term_value (ap : ArithmeticProgression) : ap.a + 10 * ap.d = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_value_l3720_372086


namespace NUMINAMATH_CALUDE_star_calculation_l3720_372083

/-- The star operation on rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b + 1

/-- Theorem stating that 1 ☆ [3 ☆ (-2)] = -6 -/
theorem star_calculation : star 1 (star 3 (-2)) = -6 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l3720_372083


namespace NUMINAMATH_CALUDE_circle_division_evenness_l3720_372085

theorem circle_division_evenness (N : ℕ) : 
  (∃ (chords : Fin N → Fin (2 * N) × Fin (2 * N)),
    (∀ i : Fin N, (chords i).1 ≠ (chords i).2) ∧ 
    (∀ i j : Fin N, i ≠ j → (chords i).1 ≠ (chords j).1 ∧ (chords i).1 ≠ (chords j).2 ∧
                            (chords i).2 ≠ (chords j).1 ∧ (chords i).2 ≠ (chords j).2) ∧
    (∀ i : Fin N, ∃ k l : ℕ, 
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * k ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * k) ∧
      (((chords i).2 - (chords i).1 : ℤ) % (2 * N : ℤ) = 2 * l ∨
       ((chords i).1 - (chords i).2 : ℤ) % (2 * N : ℤ) = 2 * l) ∧
      k + l = N)) →
  Even N :=
by sorry

end NUMINAMATH_CALUDE_circle_division_evenness_l3720_372085


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_five_l3720_372060

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem tangent_slope_at_negative_five
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv_one : deriv f 1 = 1)
  (hf_periodic : ∀ x, f (x + 2) = f (x - 2)) :
  deriv f (-5) = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_five_l3720_372060


namespace NUMINAMATH_CALUDE_income_calculation_l3720_372072

theorem income_calculation (a b c d e : ℝ) : 
  (a + b) / 2 = 4050 →
  (b + c) / 2 = 5250 →
  (a + c) / 2 = 4200 →
  (a + b + d) / 3 = 4800 →
  (c + d + e) / 3 = 6000 →
  (b + a + e) / 3 = 4500 →
  a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400 :=
by sorry

end NUMINAMATH_CALUDE_income_calculation_l3720_372072


namespace NUMINAMATH_CALUDE_probability_real_roots_l3720_372006

-- Define the interval [0,5]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 5}

-- Define the condition for real roots
def has_real_roots (p : ℝ) : Prop := p^2 ≥ 4

-- Define the measure of the interval where the equation has real roots
def measure_real_roots : ℝ := 3

-- Define the total measure of the interval
def total_measure : ℝ := 5

-- State the theorem
theorem probability_real_roots : 
  (measure_real_roots / total_measure : ℝ) = 0.6 := by sorry

end NUMINAMATH_CALUDE_probability_real_roots_l3720_372006
