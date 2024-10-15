import Mathlib

namespace NUMINAMATH_CALUDE_steve_total_cost_is_23_56_l4080_408023

/-- Calculates the total cost of Steve's DVD purchase --/
def steveTotalCost (mikeDVDPrice baseShippingRate salesTaxRate discountRate : ℚ) : ℚ :=
  let steveDVDPrice := 2 * mikeDVDPrice
  let otherDVDPrice := 7
  let subtotalBeforePromo := otherDVDPrice + otherDVDPrice
  let shippingCost := baseShippingRate * subtotalBeforePromo
  let subtotalWithShipping := subtotalBeforePromo + shippingCost
  let salesTax := salesTaxRate * subtotalWithShipping
  let subtotalWithTax := subtotalWithShipping + salesTax
  let discount := discountRate * subtotalWithTax
  subtotalWithTax - discount

/-- Theorem stating that Steve's total cost is $23.56 --/
theorem steve_total_cost_is_23_56 :
  steveTotalCost 5 0.8 0.1 0.15 = 23.56 := by
  sorry

end NUMINAMATH_CALUDE_steve_total_cost_is_23_56_l4080_408023


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l4080_408061

/-- Triangle with vertices P, Q, R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- Angle bisector equation coefficients -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Theorem: For the given triangle, the angle bisector equation of ∠P has a + c = 89 -/
theorem angle_bisector_sum (t : Triangle) (eq : AngleBisectorEq) : 
  t.P = (-8, 5) → t.Q = (-15, -19) → t.R = (1, -7) → 
  (∃ (x y : ℝ), eq.a * x + 2 * y + eq.c = 0) →
  eq.a + eq.c = 89 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l4080_408061


namespace NUMINAMATH_CALUDE_sum_of_digits_l4080_408007

/-- Given three-digit numbers of the form 4a5 and 9b2, where a and b are single digits,
    if 4a5 + 457 = 9b2 and 9b2 is divisible by 11, then a + b = 4 -/
theorem sum_of_digits (a b : ℕ) : 
  (a < 10) →
  (b < 10) →
  (400 + 10 * a + 5 + 457 = 900 + 10 * b + 2) →
  (900 + 10 * b + 2) % 11 = 0 →
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l4080_408007


namespace NUMINAMATH_CALUDE_function_symmetry_l4080_408002

/-- The function f(x) = (1-x)/(1+x) is symmetric about the line y = x -/
theorem function_symmetry (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (1 - x) / (1 + x)
  f (f x) = x :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_l4080_408002


namespace NUMINAMATH_CALUDE_percentage_of_a_l4080_408001

theorem percentage_of_a (a b c : ℝ) (P : ℝ) : 
  (P / 100) * a = 8 →
  0.08 * b = 2 →
  c = b / a →
  P = 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_a_l4080_408001


namespace NUMINAMATH_CALUDE_sin_120_degrees_l4080_408086

theorem sin_120_degrees (π : Real) :
  Real.sin (2 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l4080_408086


namespace NUMINAMATH_CALUDE_solve_system_l4080_408004

theorem solve_system (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10) 
  (eq2 : 5 * p + 2 * q = 20) : 
  q = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l4080_408004


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_three_and_six_l4080_408021

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (d : Nat) : ℚ := d / 9

/-- The sum of the repeating decimals 0.3333... and 0.6666... is equal to 1 -/
theorem sum_of_repeating_decimals_three_and_six :
  RepeatingDecimal 3 + RepeatingDecimal 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_three_and_six_l4080_408021


namespace NUMINAMATH_CALUDE_cloth_selling_price_l4080_408010

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem cloth_selling_price 
  (quantity : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) :
  quantity = 85 →
  profit_per_meter = 35 →
  cost_price_per_meter = 70 →
  quantity * (profit_per_meter + cost_price_per_meter) = 8925 := by
  sorry

#check cloth_selling_price

end NUMINAMATH_CALUDE_cloth_selling_price_l4080_408010


namespace NUMINAMATH_CALUDE_father_age_twice_marika_correct_target_year_l4080_408089

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age in the reference year -/
def father_age_reference : ℕ := 5 * (reference_year - marika_birth_year)

/-- The year when Marika's father's age will be twice her age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔
  (year - marika_birth_year) * 2 = (year - reference_year) + father_age_reference :=
by sorry

theorem correct_target_year : 
  (target_year - marika_birth_year) * 2 = (target_year - reference_year) + father_age_reference :=
by sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_correct_target_year_l4080_408089


namespace NUMINAMATH_CALUDE_wall_ratio_l4080_408052

/-- Proves that for a rectangular wall with given dimensions, the ratio of length to height is 7:1 -/
theorem wall_ratio (w h l : ℝ) (volume : ℝ) : 
  h = 6 * w →
  volume = l * w * h →
  w = 4 →
  volume = 16128 →
  l / h = 7 := by
  sorry

end NUMINAMATH_CALUDE_wall_ratio_l4080_408052


namespace NUMINAMATH_CALUDE_augmented_matrix_problem_l4080_408081

/-- Given a system of linear equations with augmented matrix
    ⎛ 3 2 1 ⎞
    ⎝ 1 1 m ⎠
    where Dx = 5, prove that m = -2 -/
theorem augmented_matrix_problem (m : ℝ) : 
  let A : Matrix (Fin 2) (Fin 3) ℝ := ![![3, 2, 1], ![1, 1, m]]
  let Dx := (A 0 2 * A 1 1 - A 0 1 * A 1 2) / (A 0 0 * A 1 1 - A 0 1 * A 1 0)
  Dx = 5 → m = -2 := by
  sorry


end NUMINAMATH_CALUDE_augmented_matrix_problem_l4080_408081


namespace NUMINAMATH_CALUDE_hotel_price_difference_l4080_408045

-- Define the charges for single rooms at hotels P, R, and G
def single_room_P (r g : ℝ) : ℝ := 0.45 * r
def single_room_P' (r g : ℝ) : ℝ := 0.90 * g

-- Define the charges for double rooms at hotels P, R, and G
def double_room_P (r g : ℝ) : ℝ := 0.70 * r
def double_room_P' (r g : ℝ) : ℝ := 0.80 * g

-- Define the charges for suites at hotels P, R, and G
def suite_P (r g : ℝ) : ℝ := 0.60 * r
def suite_P' (r g : ℝ) : ℝ := 0.85 * g

theorem hotel_price_difference (r_single g_single r_double g_double : ℝ) :
  single_room_P r_single g_single = single_room_P' r_single g_single ∧
  double_room_P r_double g_double = double_room_P' r_double g_double →
  (r_single / g_single - 1) * 100 - (r_double / g_double - 1) * 100 = 85.71 :=
by sorry

end NUMINAMATH_CALUDE_hotel_price_difference_l4080_408045


namespace NUMINAMATH_CALUDE_P_plus_Q_equals_46_l4080_408029

theorem P_plus_Q_equals_46 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3)) →
  P + Q = 46 := by
sorry

end NUMINAMATH_CALUDE_P_plus_Q_equals_46_l4080_408029


namespace NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l4080_408024

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < -2) :
  Real.sqrt (x / (1 + (x + 1) / (x + 2))) = -x := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l4080_408024


namespace NUMINAMATH_CALUDE_investment_problem_l4080_408031

/-- Investment problem -/
theorem investment_problem 
  (x_investment : ℕ) 
  (y_investment : ℕ) 
  (z_join_time : ℕ) 
  (total_profit : ℕ) 
  (z_profit_share : ℕ) 
  (h1 : x_investment = 36000)
  (h2 : y_investment = 42000)
  (h3 : z_join_time = 4)
  (h4 : total_profit = 13860)
  (h5 : z_profit_share = 4032) :
  ∃ z_investment : ℕ, z_investment = 52000 ∧ 
    (x_investment * 12 + y_investment * 12) * z_profit_share = 
    z_investment * (12 - z_join_time) * (total_profit - z_profit_share) :=
sorry

end NUMINAMATH_CALUDE_investment_problem_l4080_408031


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l4080_408028

/-- The sum of an infinite geometric series with first term 5/3 and common ratio -9/20 is 100/87 -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -9/20
  let S := a / (1 - r)
  S = 100/87 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l4080_408028


namespace NUMINAMATH_CALUDE_direct_proportion_implies_m_eq_two_l4080_408000

/-- A function y of x is a direct proportion if it can be written as y = kx where k is a non-zero constant -/
def is_direct_proportion (y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, y x = k * x

/-- Given y = (m^2 + 2m)x^(m^2 - 3), if y is a direct proportion function of x, then m = 2 -/
theorem direct_proportion_implies_m_eq_two (m : ℝ) :
  is_direct_proportion (fun x => (m^2 + 2*m) * x^(m^2 - 3)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_implies_m_eq_two_l4080_408000


namespace NUMINAMATH_CALUDE_third_median_length_l4080_408075

/-- A triangle with two known medians and area -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  area : ℝ

/-- The length of the third median in a triangle -/
def third_median (t : Triangle) : ℝ := sorry

theorem third_median_length (t : Triangle) 
  (h1 : t.median1 = 4)
  (h2 : t.median2 = 5)
  (h3 : t.area = 10 * Real.sqrt 3) :
  third_median t = 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_third_median_length_l4080_408075


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l4080_408094

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by n 20 ∧ is_perfect_square (n^2) ∧ is_perfect_fifth_power (n^3)

theorem smallest_n_satisfying_conditions :
  ∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000 :=
by sorry

theorem three_million_two_hundred_thousand_satisfies_conditions :
  satisfies_conditions 3200000 :=
by sorry

theorem smallest_n_is_three_million_two_hundred_thousand :
  (∀ m : ℕ, m > 0 → satisfies_conditions m → m ≥ 3200000) ∧
  satisfies_conditions 3200000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_three_million_two_hundred_thousand_satisfies_conditions_smallest_n_is_three_million_two_hundred_thousand_l4080_408094


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l4080_408042

/-- The number of sections created by n line segments in a rectangle,
    where each new line intersects all previous lines -/
def num_sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- The property that each new line intersects all previous lines -/
def intersects_all_previous (n : ℕ) : Prop :=
  ∀ k, k < n → num_sections k < num_sections (k + 1)

theorem max_sections_five_lines :
  intersects_all_previous 5 →
  num_sections 5 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l4080_408042


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l4080_408078

theorem sock_drawing_probability : 
  ∀ (total_socks : ℕ) (colors : ℕ) (socks_per_color : ℕ) (drawn_socks : ℕ),
    total_socks = colors * socks_per_color →
    total_socks = 10 →
    colors = 5 →
    socks_per_color = 2 →
    drawn_socks = 5 →
    (Nat.choose total_socks drawn_socks : ℚ) ≠ 0 →
    (Nat.choose colors 4 * Nat.choose 4 1 * (socks_per_color ^ 3) : ℚ) / 
    (Nat.choose total_socks drawn_socks : ℚ) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l4080_408078


namespace NUMINAMATH_CALUDE_special_function_at_five_l4080_408096

/-- A function satisfying f(x - y) = f(x) + f(y) for all real x and y, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x - y) = f x + f y) ∧ (f 0 = 2)

/-- Theorem: For any function satisfying the special_function property, f(5) = 1 -/
theorem special_function_at_five (f : ℝ → ℝ) (h : special_function f) : f 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_five_l4080_408096


namespace NUMINAMATH_CALUDE_students_remaining_after_four_stops_l4080_408057

theorem students_remaining_after_four_stops :
  let initial_students : ℕ := 60
  let stops : ℕ := 4
  let fraction_remaining : ℚ := 2 / 3
  let final_students := initial_students * fraction_remaining ^ stops
  final_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_after_four_stops_l4080_408057


namespace NUMINAMATH_CALUDE_books_on_shelf_l4080_408017

def initial_books : ℕ := 38
def marta_removes : ℕ := 10
def tom_removes : ℕ := 5
def tom_adds : ℕ := 12

theorem books_on_shelf : 
  initial_books - marta_removes - tom_removes + tom_adds = 35 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l4080_408017


namespace NUMINAMATH_CALUDE_escalator_problem_l4080_408099

/-- The number of steps Petya counts while ascending the escalator -/
def steps_ascending : ℕ := 75

/-- The number of steps Petya counts while descending the escalator -/
def steps_descending : ℕ := 150

/-- The ratio of Petya's descending speed to ascending speed -/
def speed_ratio : ℚ := 3

/-- The speed of the escalator in steps per unit time -/
def escalator_speed : ℚ := 3/5

/-- The number of steps on the stopped escalator -/
def escalator_length : ℕ := 120

theorem escalator_problem :
  steps_ascending * (1 + escalator_speed) = 
  (steps_descending / speed_ratio) * (speed_ratio - escalator_speed) ∧
  escalator_length = steps_ascending * (1 + escalator_speed) := by
  sorry

end NUMINAMATH_CALUDE_escalator_problem_l4080_408099


namespace NUMINAMATH_CALUDE_basketball_league_games_l4080_408022

/-- The number of games played in a league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 5 games with every other team, 
    the total number of games played is 225. --/
theorem basketball_league_games : total_games 10 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l4080_408022


namespace NUMINAMATH_CALUDE_interval_length_for_inequality_l4080_408050

theorem interval_length_for_inequality : ∃ (a b : ℚ),
  (∀ x : ℝ, |5 * x^2 - 2/5| ≤ |x - 8| ↔ a ≤ x ∧ x ≤ b) ∧
  b - a = 13/5 :=
by sorry

end NUMINAMATH_CALUDE_interval_length_for_inequality_l4080_408050


namespace NUMINAMATH_CALUDE_cost_39_roses_l4080_408072

/-- Represents the cost of a bouquet of roses -/
def bouquet_cost (roses : ℕ) : ℚ :=
  sorry

/-- The price of a bouquet is directly proportional to the number of roses -/
axiom price_proportional (r₁ r₂ : ℕ) : 
  bouquet_cost r₁ / bouquet_cost r₂ = r₁ / r₂

/-- A bouquet of 12 roses costs $20 -/
axiom dozen_cost : bouquet_cost 12 = 20

theorem cost_39_roses : bouquet_cost 39 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cost_39_roses_l4080_408072


namespace NUMINAMATH_CALUDE_evaluate_expression_l4080_408097

theorem evaluate_expression (a b : ℚ) (h1 : a = 5) (h2 : b = -3) : 3 / (a + b) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4080_408097


namespace NUMINAMATH_CALUDE_same_color_probability_l4080_408044

def total_plates : ℕ := 11
def red_plates : ℕ := 6
def green_plates : ℕ := 5
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose green_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l4080_408044


namespace NUMINAMATH_CALUDE_triangle_area_l4080_408095

theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) :
  (1/2) * a * b = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4080_408095


namespace NUMINAMATH_CALUDE_correct_equation_l4080_408009

theorem correct_equation (x : ℝ) : 
  (550 + x) + (460 + x) + (359 + x) + (340 + x) = 2012 + x ↔ x = 75.75 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l4080_408009


namespace NUMINAMATH_CALUDE_jenny_toy_spending_l4080_408020

/-- Proves that Jenny spent $200 on toys for the cat in the first year. -/
theorem jenny_toy_spending (adoption_fee vet_cost monthly_food_cost total_months jenny_total_spent : ℕ) 
  (h1 : adoption_fee = 50)
  (h2 : vet_cost = 500)
  (h3 : monthly_food_cost = 25)
  (h4 : total_months = 12)
  (h5 : jenny_total_spent = 625) : 
  jenny_total_spent - (adoption_fee + vet_cost + monthly_food_cost * total_months) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_jenny_toy_spending_l4080_408020


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l4080_408090

theorem abs_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l4080_408090


namespace NUMINAMATH_CALUDE_arnel_kept_pencils_l4080_408064

/-- Calculates the number of pencils Arnel kept given the problem conditions -/
def pencils_kept (num_boxes : ℕ) (num_friends : ℕ) (pencils_per_friend : ℕ) (pencils_left_per_box : ℕ) : ℕ :=
  let total_pencils := num_boxes * (pencils_per_friend * num_friends / num_boxes + pencils_left_per_box)
  total_pencils - pencils_per_friend * num_friends

/-- Theorem stating that Arnel kept 50 pencils under the given conditions -/
theorem arnel_kept_pencils :
  pencils_kept 10 5 8 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arnel_kept_pencils_l4080_408064


namespace NUMINAMATH_CALUDE_amount_with_r_l4080_408018

/-- Given a total amount shared among three parties where one party has
    two-thirds of the combined amount of the other two, this function
    calculates the amount held by the third party. -/
def calculate_third_party_amount (total : ℚ) : ℚ :=
  (2 / 3) * (3 / 5) * total

/-- Theorem stating that given the problem conditions, 
    the amount held by r is 3200. -/
theorem amount_with_r (total : ℚ) (h_total : total = 8000) :
  calculate_third_party_amount total = 3200 := by
  sorry

#eval calculate_third_party_amount 8000

end NUMINAMATH_CALUDE_amount_with_r_l4080_408018


namespace NUMINAMATH_CALUDE_calculation_proof_l4080_408026

theorem calculation_proof : (Real.sqrt 5 - 1)^0 + 3⁻¹ - |-(1/3)| = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4080_408026


namespace NUMINAMATH_CALUDE_harmonic_sum_denominator_not_div_by_five_l4080_408059

/-- The sum of reciprocals from 1 to n -/
def harmonic_sum (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range n) (λ m => 1 / (m + 1 : ℚ))

/-- The set of positive integers n for which 5 does not divide the denominator
    of the harmonic sum when expressed in lowest terms -/
def D : Set ℕ+ :=
  {n | ¬ (5 ∣ (harmonic_sum n).den)}

/-- The theorem stating that D is exactly the given set -/
theorem harmonic_sum_denominator_not_div_by_five :
  D = {1, 2, 3, 4, 20, 21, 22, 23, 24, 100, 101, 102, 103, 104, 120, 121, 122, 123, 124} := by
  sorry


end NUMINAMATH_CALUDE_harmonic_sum_denominator_not_div_by_five_l4080_408059


namespace NUMINAMATH_CALUDE_vector_form_equiv_line_equation_l4080_408036

/-- The line equation y = 2x + 5 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 5

/-- The vector form of the line -/
def vector_form (r k t x y : ℝ) : Prop :=
  x = r + 3 * t ∧ y = -3 + k * t

/-- Theorem stating that the vector form represents the line y = 2x + 5 
    if and only if r = -4 and k = 6 -/
theorem vector_form_equiv_line_equation :
  ∀ r k : ℝ, (∀ t x y : ℝ, vector_form r k t x y → line_equation x y) ∧
             (∀ x y : ℝ, line_equation x y → ∃ t : ℝ, vector_form r k t x y) ↔
  r = -4 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_form_equiv_line_equation_l4080_408036


namespace NUMINAMATH_CALUDE_lcm_36_100_l4080_408068

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l4080_408068


namespace NUMINAMATH_CALUDE_probability_three_blue_marbles_specific_l4080_408098

/-- Represents the probability of drawing 3 blue marbles from a jar --/
def probability_three_blue_marbles (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))

/-- Theorem stating the probability of drawing 3 blue marbles from a specific jar configuration --/
theorem probability_three_blue_marbles_specific :
  probability_three_blue_marbles 3 4 13 = 1 / 285 := by
  sorry

#eval probability_three_blue_marbles 3 4 13

end NUMINAMATH_CALUDE_probability_three_blue_marbles_specific_l4080_408098


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l4080_408091

-- Define the polynomial Q(x)
def Q (x d e f : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

-- Define the conditions
theorem cubic_polynomial_property (d e f : ℝ) :
  -- The y-intercept is 4
  Q 0 d e f = 4 →
  -- The mean of zeros, product of zeros, and sum of coefficients are equal
  -(d/3) = -f ∧ -(d/3) = 1 + d + e + f →
  -- The value of e is 11
  e = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l4080_408091


namespace NUMINAMATH_CALUDE_bus_capacity_l4080_408039

theorem bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) :
  rows = 13 →
  sections_per_row = 2 →
  students_per_section = 2 →
  rows * sections_per_row * students_per_section = 52 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l4080_408039


namespace NUMINAMATH_CALUDE_tourists_scientific_notation_l4080_408069

-- Define the number of tourists
def tourists : ℝ := 4.55e9

-- Theorem statement
theorem tourists_scientific_notation :
  tourists = 4.55 * (10 : ℝ) ^ 9 :=
by sorry

end NUMINAMATH_CALUDE_tourists_scientific_notation_l4080_408069


namespace NUMINAMATH_CALUDE_virgo_island_trip_duration_l4080_408074

/-- The duration of Tom's trip to "Virgo" island -/
theorem virgo_island_trip_duration :
  ∀ (boat_duration : ℝ) (plane_duration : ℝ),
    boat_duration = 2 →
    plane_duration = 4 * boat_duration →
    boat_duration + plane_duration = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_virgo_island_trip_duration_l4080_408074


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l4080_408014

theorem constant_term_of_expansion (x : ℝ) (x_pos : x > 0) :
  ∃ (c : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |((y^(1/2) - 2/y)^3 - (x^(1/2) - 2/x)^3) - c| < ε) ∧ c = -6 :=
sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l4080_408014


namespace NUMINAMATH_CALUDE_zero_lt_m_lt_one_necessary_not_sufficient_l4080_408038

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x^2 + x + m^2 - 1 = 0

-- Define the condition for two real roots with different signs
def has_two_real_roots_diff_signs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ quadratic_eq m x₁ ∧ quadratic_eq m x₂

-- Theorem stating that 0 < m < 1 is a necessary but not sufficient condition
theorem zero_lt_m_lt_one_necessary_not_sufficient :
  (∀ m : ℝ, has_two_real_roots_diff_signs m → 0 < m ∧ m < 1) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ¬has_two_real_roots_diff_signs m) :=
sorry

end NUMINAMATH_CALUDE_zero_lt_m_lt_one_necessary_not_sufficient_l4080_408038


namespace NUMINAMATH_CALUDE_elois_banana_bread_l4080_408006

/-- Represents the number of loaves of banana bread Elois made on Monday -/
def monday_loaves : ℕ := sorry

/-- Represents the number of loaves of banana bread Elois made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- Represents the total number of bananas used for both days -/
def total_bananas : ℕ := 36

/-- Represents the number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

theorem elois_banana_bread :
  monday_loaves = 3 ∧
  tuesday_loaves = 2 * monday_loaves ∧
  total_bananas = bananas_per_loaf * (monday_loaves + tuesday_loaves) :=
sorry

end NUMINAMATH_CALUDE_elois_banana_bread_l4080_408006


namespace NUMINAMATH_CALUDE_ellipse_k_range_l4080_408016

/-- Represents an ellipse equation with parameter k -/
def is_ellipse (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- Foci are on the y-axis if the equation is in the form x^2/b^2 + y^2/a^2 = 1 with a > b -/
def foci_on_y_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (b^2) + y^2 / (a^2) = 1 ↔ x^2 + k * y^2 = 2

/-- The main theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse k ∧ foci_on_y_axis k → 0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l4080_408016


namespace NUMINAMATH_CALUDE_unique_valid_number_l4080_408012

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (∀ i : ℕ, i < 3 → (n / 10^i % 10 + n / 10^(i+1) % 10) ≤ 2) ∧
  (∀ i : ℕ, i < 2 → (n / 10^i % 10 + n / 10^(i+1) % 10 + n / 10^(i+2) % 10) ≥ 3)

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l4080_408012


namespace NUMINAMATH_CALUDE_complex_power_sum_l4080_408087

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i ^ 2 = -1

-- Define the cyclic nature of powers of i
axiom i_cyclic (n : ℕ) : i ^ (n + 4) = i ^ n

-- State the theorem
theorem complex_power_sum : i^20 + i^33 - i^56 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l4080_408087


namespace NUMINAMATH_CALUDE_three_solutions_l4080_408049

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The number of positive integers n satisfying n + S(n) + S(S(n)) = 2500 -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 3 solutions -/
theorem three_solutions : count_solutions = 3 := by sorry

end NUMINAMATH_CALUDE_three_solutions_l4080_408049


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l4080_408051

/-- The distance from the center of a sphere to the plane of a tangent triangle -/
theorem sphere_triangle_distance (r : ℝ) (a b c : ℝ) (h_sphere : r = 10) 
  (h_triangle : a = 18 ∧ b = 18 ∧ c = 30) (h_tangent : True) : 
  ∃ d : ℝ, d = (10 * Real.sqrt 37) / 33 ∧ 
  d^2 + ((a + b + c) / 2 * (2 * a * b) / (a + b + c))^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l4080_408051


namespace NUMINAMATH_CALUDE_total_toys_count_l4080_408084

-- Define the number of toys for each child
def jaxon_toys : ℝ := 15
def gabriel_toys : ℝ := 2.5 * jaxon_toys
def jerry_toys : ℝ := gabriel_toys + 8.5
def sarah_toys : ℝ := jerry_toys - 5.5
def emily_toys : ℝ := 1.5 * gabriel_toys

-- Define the total number of toys
def total_toys : ℝ := jerry_toys + gabriel_toys + jaxon_toys + sarah_toys + emily_toys

-- Theorem to prove
theorem total_toys_count : total_toys = 195.25 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_count_l4080_408084


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l4080_408043

theorem cube_sum_implies_sum_bound (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l4080_408043


namespace NUMINAMATH_CALUDE_room_tiles_count_l4080_408083

/-- Calculates the total number of tiles needed for a room with given dimensions and tile specifications. -/
def total_tiles (room_length room_width border_width border_tile_size inner_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (2 * (room_length - 2 * border_width) + 2 * (room_width - 2 * border_width)) - 8 * border_width
  let inner_area := (room_length - 2 * border_width) * (room_width - 2 * border_width)
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  border_tiles + inner_tiles

/-- Theorem stating that for a 15-foot by 20-foot room with a double border of 1-foot tiles
    and the rest filled with 2-foot tiles, the total number of tiles used is 144. -/
theorem room_tiles_count :
  total_tiles 20 15 2 1 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l4080_408083


namespace NUMINAMATH_CALUDE_prob_B_wins_match_value_l4080_408063

/-- The probability of player B winning a single game -/
def p_B : ℝ := 0.4

/-- The probability of player A winning a single game -/
def p_A : ℝ := 1 - p_B

/-- The probability of player B winning a best-of-three billiards match -/
def prob_B_wins_match : ℝ := p_B^2 + 2 * p_B^2 * p_A

theorem prob_B_wins_match_value :
  prob_B_wins_match = 0.352 := by sorry

end NUMINAMATH_CALUDE_prob_B_wins_match_value_l4080_408063


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4080_408093

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 10*x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4080_408093


namespace NUMINAMATH_CALUDE_tank_capacity_ratio_l4080_408047

theorem tank_capacity_ratio : 
  let h_a : ℝ := 8
  let c_a : ℝ := 8
  let h_b : ℝ := 8
  let c_b : ℝ := 10
  let r_a : ℝ := c_a / (2 * Real.pi)
  let r_b : ℝ := c_b / (2 * Real.pi)
  let v_a : ℝ := Real.pi * r_a^2 * h_a
  let v_b : ℝ := Real.pi * r_b^2 * h_b
  v_a / v_b = 0.64
  := by sorry

end NUMINAMATH_CALUDE_tank_capacity_ratio_l4080_408047


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_000815_l4080_408054

def scientific_notation (n : ℝ) (coefficient : ℝ) (exponent : ℤ) : Prop :=
  1 ≤ coefficient ∧ coefficient < 10 ∧ n = coefficient * (10 : ℝ) ^ exponent

theorem scientific_notation_of_0_000815 :
  scientific_notation 0.000815 8.15 (-4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_000815_l4080_408054


namespace NUMINAMATH_CALUDE_triangle_gp_ratio_lt_two_l4080_408085

/-- Given a triangle with side lengths forming a geometric progression,
    prove that the common ratio of the progression is less than 2. -/
theorem triangle_gp_ratio_lt_two (b q : ℝ) (hb : b > 0) (hq : q > 0) :
  (b + b*q > b*q^2) ∧ (b + b*q^2 > b*q) ∧ (b*q + b*q^2 > b) →
  q < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_gp_ratio_lt_two_l4080_408085


namespace NUMINAMATH_CALUDE_solution_implication_l4080_408048

theorem solution_implication (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → 
  Real.sqrt (2 * m - n) = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_implication_l4080_408048


namespace NUMINAMATH_CALUDE_mass_of_man_is_72_l4080_408008

/-- The density of water in kg/m³ -/
def water_density : ℝ := 1000

/-- Calculates the mass of a man based on boat dimensions and sinking depth -/
def mass_of_man (boat_length boat_breadth sinking_depth : ℝ) : ℝ :=
  water_density * boat_length * boat_breadth * sinking_depth

/-- Theorem stating that the mass of the man is 72 kg given the boat's dimensions and sinking depth -/
theorem mass_of_man_is_72 :
  mass_of_man 3 2 0.012 = 72 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_72_l4080_408008


namespace NUMINAMATH_CALUDE_rook_placement_impossibility_l4080_408067

theorem rook_placement_impossibility :
  ∀ (r b g : ℕ),
  r + b + g = 50 →
  2 * r ≤ b →
  2 * b ≤ g →
  2 * g ≤ r →
  False :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_impossibility_l4080_408067


namespace NUMINAMATH_CALUDE_range_of_x_l4080_408073

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) 
  (h3 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) : 
  x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l4080_408073


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l4080_408058

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 - k*a + 25 = (a - b)^2) → k = 10 ∨ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l4080_408058


namespace NUMINAMATH_CALUDE_soccer_team_selection_l4080_408076

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def max_quadruplets : ℕ := 2

theorem soccer_team_selection :
  (Nat.choose total_players players_to_select) -
  ((Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (players_to_select - 3)) +
   (Nat.choose quadruplets 4) * (Nat.choose (total_players - quadruplets) (players_to_select - 4))) = 9240 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l4080_408076


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l4080_408079

/-- Parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Line passing through the focus of a parabola with slope angle 60° -/
structure FocusLine (c : Parabola) where
  slope : ℝ
  h_slope : slope = Real.sqrt 3
  focus_x : ℝ
  h_focus_x : focus_x = c.p / 2

/-- Theorem stating the ratio of AB to AP is 7/12 -/
theorem parabola_intersection_ratio (c : Parabola) (l : FocusLine c) 
  (A B : ParabolaPoint c) (P : ℝ × ℝ) :
  A.x > 0 → A.y > 0 →  -- A in first quadrant
  B.x > 0 → B.y < 0 →  -- B in fourth quadrant
  P.1 = 0 →  -- P on y-axis
  (A.y - l.focus_x) = l.slope * (A.x - l.focus_x) →  -- A on line l
  (B.y - l.focus_x) = l.slope * (B.x - l.focus_x) →  -- B on line l
  (P.2 - l.focus_x) = l.slope * (P.1 - l.focus_x) →  -- P on line l
  abs (A.x - B.x) / abs (A.x - P.1) = 7 / 12 := by
    sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l4080_408079


namespace NUMINAMATH_CALUDE_travel_time_calculation_l4080_408077

/-- Given a speed of 25 km/hr and a distance of 125 km, the time taken is 5 hours. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (time : ℝ) : 
  speed = 25 → distance = 125 → time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l4080_408077


namespace NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l4080_408040

theorem cos_arcsin_three_fifths : 
  Real.cos (Real.arcsin (3/5)) = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_three_fifths_l4080_408040


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4080_408025

theorem trigonometric_identity : 
  Real.sin (40 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (40 * π / 180) * Real.sin (80 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4080_408025


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l4080_408088

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), k > 15 ∧ ∀ (m : ℕ), Odd m → k ∣ (m * (m + 2) * (m + 4) * (m + 6) * (m + 8))) → False :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l4080_408088


namespace NUMINAMATH_CALUDE_root_implies_a_value_l4080_408080

theorem root_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x = 0 ∧ x = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l4080_408080


namespace NUMINAMATH_CALUDE_quadratic_function_m_value_l4080_408027

/-- A quadratic function of the form y = 3x^2 + 2(m-1)x + n -/
def quadratic_function (m n : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (m - 1) * x + n

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (m - 1)

theorem quadratic_function_m_value (m n : ℝ) :
  (∀ x < 1, quadratic_derivative m x < 0) →
  (∀ x ≥ 1, quadratic_derivative m x ≥ 0) →
  m = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_value_l4080_408027


namespace NUMINAMATH_CALUDE_arc_length_for_given_circle_l4080_408060

/-- Given a circle with radius 2 and a central angle of 2 radians, 
    the corresponding arc length is 4. -/
theorem arc_length_for_given_circle : 
  ∀ (r θ l : ℝ), r = 2 → θ = 2 → l = r * θ → l = 4 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_given_circle_l4080_408060


namespace NUMINAMATH_CALUDE_point_coordinates_in_second_quadrant_l4080_408030

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the second quadrant
def second_quadrant (p : Point) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Define distance to x-axis
def distance_to_x_axis (p : Point) : ℝ :=
  |p.2|

-- Define distance to y-axis
def distance_to_y_axis (p : Point) : ℝ :=
  |p.1|

-- Theorem statement
theorem point_coordinates_in_second_quadrant (M : Point) 
  (h1 : second_quadrant M)
  (h2 : distance_to_x_axis M = 1)
  (h3 : distance_to_y_axis M = 2) :
  M = (-2, 1) :=
sorry

end NUMINAMATH_CALUDE_point_coordinates_in_second_quadrant_l4080_408030


namespace NUMINAMATH_CALUDE_fraction_subtraction_property_l4080_408066

theorem fraction_subtraction_property (a b n : ℕ) (h1 : b > a) (h2 : a > 0) 
  (h3 : ∀ k : ℕ, k > 0 → (1 : ℚ) / k ≤ a / b → k ≥ n) : a * n - b < a := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_property_l4080_408066


namespace NUMINAMATH_CALUDE_partner_p_investment_time_l4080_408032

/-- The investment and profit scenario for two partners -/
structure InvestmentScenario where
  /-- The ratio of investments for partners p and q -/
  investment_ratio : Rat × Rat
  /-- The ratio of profits for partners p and q -/
  profit_ratio : Rat × Rat
  /-- The number of months partner q invested -/
  q_months : ℕ

/-- The theorem stating the investment time for partner p -/
theorem partner_p_investment_time (scenario : InvestmentScenario) 
  (h1 : scenario.investment_ratio = (7, 5))
  (h2 : scenario.profit_ratio = (7, 13))
  (h3 : scenario.q_months = 13) :
  ∃ (p_months : ℕ), p_months = 7 ∧ 
  (scenario.investment_ratio.1 * p_months) / (scenario.investment_ratio.2 * scenario.q_months) = 
  scenario.profit_ratio.1 / scenario.profit_ratio.2 :=
sorry

end NUMINAMATH_CALUDE_partner_p_investment_time_l4080_408032


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l4080_408070

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define set A
def A : Set ℕ := {1, 5, 9}

-- Define set B
def B : Set ℕ := {3, 7, 9}

-- Theorem statement
theorem complement_A_intersect_B : (Aᶜ ∩ B) = {3, 7} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l4080_408070


namespace NUMINAMATH_CALUDE_basketball_probabilities_l4080_408053

/-- Probability of A making a shot -/
def prob_A_makes : ℝ := 0.8

/-- Probability of B missing a shot -/
def prob_B_misses : ℝ := 0.1

/-- Probability of B making a shot -/
def prob_B_makes : ℝ := 1 - prob_B_misses

theorem basketball_probabilities :
  (prob_A_makes * prob_B_makes = 0.72) ∧
  (prob_A_makes * (1 - prob_B_makes) + (1 - prob_A_makes) * prob_B_makes = 0.26) := by
  sorry

end NUMINAMATH_CALUDE_basketball_probabilities_l4080_408053


namespace NUMINAMATH_CALUDE_expression_value_l4080_408071

theorem expression_value : 3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4080_408071


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l4080_408056

theorem circle_equation_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 100) →
  k = 59 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l4080_408056


namespace NUMINAMATH_CALUDE_N_subset_M_l4080_408019

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l4080_408019


namespace NUMINAMATH_CALUDE_wendy_total_profit_l4080_408033

/-- Represents a fruit sale --/
structure FruitSale where
  price : Float
  quantity : Nat
  profit_margin : Float
  discount : Float

/-- Represents a day's sales --/
structure DaySales where
  morning_apples : FruitSale
  morning_oranges : FruitSale
  morning_bananas : FruitSale
  afternoon_apples : FruitSale
  afternoon_oranges : FruitSale
  afternoon_bananas : FruitSale

/-- Represents unsold fruits --/
structure UnsoldFruits where
  banana_quantity : Nat
  banana_price : Float
  banana_discount : Float
  banana_profit_margin : Float
  orange_quantity : Nat
  orange_price : Float
  orange_discount : Float
  orange_profit_margin : Float

/-- Calculate profit for a single fruit sale --/
def calculate_profit (sale : FruitSale) : Float :=
  sale.price * sale.quantity.toFloat * (1 - sale.discount) * sale.profit_margin

/-- Calculate total profit for a day --/
def calculate_day_profit (day : DaySales) : Float :=
  calculate_profit day.morning_apples +
  calculate_profit day.morning_oranges +
  calculate_profit day.morning_bananas +
  calculate_profit day.afternoon_apples +
  calculate_profit day.afternoon_oranges +
  calculate_profit day.afternoon_bananas

/-- Calculate profit from unsold fruits --/
def calculate_unsold_profit (unsold : UnsoldFruits) : Float :=
  unsold.banana_quantity.toFloat * unsold.banana_price * (1 - unsold.banana_discount) * unsold.banana_profit_margin +
  unsold.orange_quantity.toFloat * unsold.orange_price * (1 - unsold.orange_discount) * unsold.orange_profit_margin

/-- Main theorem: Wendy's total profit for the week --/
theorem wendy_total_profit (day1 day2 : DaySales) (unsold : UnsoldFruits) :
  calculate_day_profit day1 + calculate_day_profit day2 + calculate_unsold_profit unsold = 84.07 := by
  sorry

end NUMINAMATH_CALUDE_wendy_total_profit_l4080_408033


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l4080_408041

/-- Calculates the total cost of purchasing a puppy and related items. -/
def total_cost (puppy_cost : ℚ) (food_consumption_per_day : ℚ) (food_duration_weeks : ℕ) 
  (food_cost_per_bag : ℚ) (food_amount_per_bag : ℚ) (leash_cost : ℚ) (collar_cost : ℚ) 
  (dog_bed_cost : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let food_total_consumption := food_consumption_per_day * (food_duration_weeks * 7)
  let food_bags_needed := (food_total_consumption / food_amount_per_bag).ceil
  let food_cost := food_bags_needed * food_cost_per_bag
  let collar_discounted := collar_cost * (1 - 0.1)
  let taxable_items_cost := leash_cost + collar_discounted + dog_bed_cost
  let tax_amount := taxable_items_cost * sales_tax_rate
  puppy_cost + food_cost + taxable_items_cost + tax_amount

/-- Theorem stating that the total cost is $211.85 given the specified conditions. -/
theorem total_cost_is_correct : 
  total_cost 150 (1/3) 6 2 (7/2) 15 12 25 (6/100) = 21185/100 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_is_correct_l4080_408041


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l4080_408013

/-- Proves that for a downward-opening parabola passing through (-1, y₁) and (4, y₂), y₁ > y₂ -/
theorem parabola_point_comparison (a c y₁ y₂ : ℝ) 
  (h_a : a < 0)
  (h_y₁ : y₁ = a * (-1 - 1)^2 + c)
  (h_y₂ : y₂ = a * (4 - 1)^2 + c) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l4080_408013


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l4080_408046

theorem fraction_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : 1/a + 1/b + 1/c = 1/(a + b + c)) 
  (h2 : Odd n) 
  (h3 : n > 0) : 
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l4080_408046


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l4080_408015

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + (2 : ℚ) / (n + 2) + (n + 1) / (n + 2) = 3 ∧ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l4080_408015


namespace NUMINAMATH_CALUDE_apples_in_basket_l4080_408011

def apples_remaining (initial : ℕ) (ricki_removes : ℕ) : ℕ :=
  initial - (ricki_removes + 2 * ricki_removes)

theorem apples_in_basket (initial : ℕ) (ricki_removes : ℕ) 
  (h1 : initial = 74) (h2 : ricki_removes = 14) : 
  apples_remaining initial ricki_removes = 32 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l4080_408011


namespace NUMINAMATH_CALUDE_digit_count_l4080_408055

theorem digit_count (n : ℕ) 
  (h1 : (n : ℚ) * 18 = n * 18) 
  (h2 : 4 * 8 = 32) 
  (h3 : 5 * 26 = 130) 
  (h4 : n * 18 = 32 + 130) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_l4080_408055


namespace NUMINAMATH_CALUDE_two_solutions_exist_l4080_408035

def A (x : ℝ) : Set ℝ := {0, 1, 2, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem two_solutions_exist :
  ∃! (s : Set ℝ), (∃ (x₁ x₂ : ℝ), s = {x₁, x₂} ∧ 
    ∀ (x : ℝ), (A x ∪ B x = A x) ↔ (x ∈ s)) ∧ 
    (∀ (x : ℝ), x ∈ s → x^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_exist_l4080_408035


namespace NUMINAMATH_CALUDE_jane_crayons_jane_crayons_proof_l4080_408082

/-- Proves that Jane ends up with 80 crayons after starting with 87 and losing 7 to a hippopotamus. -/
theorem jane_crayons : ℕ → ℕ → ℕ → Prop :=
  fun initial_crayons eaten_crayons final_crayons =>
    initial_crayons = 87 ∧ 
    eaten_crayons = 7 ∧ 
    final_crayons = initial_crayons - eaten_crayons →
    final_crayons = 80

/-- The proof of the theorem. -/
theorem jane_crayons_proof : jane_crayons 87 7 80 := by
  sorry

end NUMINAMATH_CALUDE_jane_crayons_jane_crayons_proof_l4080_408082


namespace NUMINAMATH_CALUDE_compound_interest_problem_l4080_408062

-- Define the compound interest function
def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem compound_interest_problem :
  ∃ (P r : ℝ), 
    compound_interest P r 2 = 8800 ∧
    compound_interest P r 3 = 9261 ∧
    abs (P - 7945.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l4080_408062


namespace NUMINAMATH_CALUDE_exists_uncovered_vertices_l4080_408005

/-- A regular polygon with 2n vertices -/
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin (2*n) → ℝ × ℝ)

/-- A pattern is a subset of n vertices of a 2n-gon -/
def Pattern (n : ℕ) := Finset (Fin (2*n))

/-- Rotation of a pattern by k positions -/
def rotate (n : ℕ) (p : Pattern n) (k : ℕ) : Pattern n :=
  sorry

/-- The set of vertices covered by 100 rotations of a pattern -/
def coveredVertices (n : ℕ) (p : Pattern n) : Finset (Fin (2*n)) :=
  sorry

/-- Theorem stating that there exists a 2n-gon and a pattern such that
    100 rotations do not cover all vertices -/
theorem exists_uncovered_vertices :
  ∃ (n : ℕ) (p : Pattern n), (coveredVertices n p).card < 2*n :=
sorry

end NUMINAMATH_CALUDE_exists_uncovered_vertices_l4080_408005


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l4080_408092

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l4080_408092


namespace NUMINAMATH_CALUDE_most_suitable_student_l4080_408065

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the average score and variances
def average_score : ℝ := 180

def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 65
  | Student.B => 56.5
  | Student.C => 53
  | Student.D => 50.5

-- Define the suitability criterion
def more_suitable (s1 s2 : Student) : Prop :=
  variance s1 < variance s2

-- Theorem statement
theorem most_suitable_student :
  ∀ s : Student, s ≠ Student.D → more_suitable Student.D s :=
sorry

end NUMINAMATH_CALUDE_most_suitable_student_l4080_408065


namespace NUMINAMATH_CALUDE_unique_common_difference_l4080_408034

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  n : ℕ  -- number of terms
  third_term_is_7 : a + 2 * d = 7
  last_term_is_37 : a + (n - 1) * d = 37
  sum_is_198 : n * (2 * a + (n - 1) * d) / 2 = 198

/-- Theorem stating the existence and uniqueness of the common difference -/
theorem unique_common_difference (seq : ArithmeticSequence) : 
  ∃! d : ℝ, seq.d = d := by sorry

end NUMINAMATH_CALUDE_unique_common_difference_l4080_408034


namespace NUMINAMATH_CALUDE_squat_rack_cost_squat_rack_cost_proof_l4080_408037

/-- The cost of a squat rack, given that the barbell costs 1/10 as much and the total is $2750 -/
theorem squat_rack_cost : ℝ → ℝ → Prop :=
  fun (squat_rack_cost barbell_cost : ℝ) =>
    barbell_cost = squat_rack_cost / 10 ∧
    squat_rack_cost + barbell_cost = 2750 →
    squat_rack_cost = 2500

/-- Proof of the squat rack cost theorem -/
theorem squat_rack_cost_proof : squat_rack_cost 2500 250 := by
  sorry

end NUMINAMATH_CALUDE_squat_rack_cost_squat_rack_cost_proof_l4080_408037


namespace NUMINAMATH_CALUDE_fourth_root_equality_l4080_408003

theorem fourth_root_equality (x : ℝ) (hx : x > 0) : 
  (x * (x^3)^(1/4))^(1/4) = x^(7/16) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equality_l4080_408003
