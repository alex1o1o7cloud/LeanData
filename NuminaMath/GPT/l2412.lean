import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l2412_241204
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are available

-- Define the main problem as a theorem
theorem simplify_expression (t : ℝ) : 
  (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2412_241204


namespace NUMINAMATH_GPT_smallest_abundant_number_not_multiple_of_10_l2412_241262

-- Definition of proper divisors of a number n
def properDivisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

-- Definition of an abundant number
def isAbundant (n : ℕ) : Prop := 
  (properDivisors n).sum > n

-- Definition of not being a multiple of 10
def notMultipleOf10 (n : ℕ) : Prop := 
  n % 10 ≠ 0

-- Statement to prove
theorem smallest_abundant_number_not_multiple_of_10 :
  ∃ n, isAbundant n ∧ notMultipleOf10 n ∧ ∀ m, (isAbundant m ∧ notMultipleOf10 m) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_abundant_number_not_multiple_of_10_l2412_241262


namespace NUMINAMATH_GPT_number_total_11_l2412_241298

theorem number_total_11 (N : ℕ) (S : ℝ)
  (h1 : S = 10.7 * N)
  (h2 : (6 : ℝ) * 10.5 = 63)
  (h3 : (6 : ℝ) * 11.4 = 68.4)
  (h4 : 13.7 = 13.700000000000017)
  (h5 : S = 63 + 68.4 - 13.7) : 
  N = 11 := 
sorry

end NUMINAMATH_GPT_number_total_11_l2412_241298


namespace NUMINAMATH_GPT_k_league_teams_l2412_241270

theorem k_league_teams (n : ℕ) (h : n*(n-1)/2 = 91) : n = 14 := sorry

end NUMINAMATH_GPT_k_league_teams_l2412_241270


namespace NUMINAMATH_GPT_witch_votes_is_seven_l2412_241266

-- Definitions
def votes_for_witch (W : ℕ) : ℕ := W
def votes_for_unicorn (W : ℕ) : ℕ := 3 * W
def votes_for_dragon (W : ℕ) : ℕ := W + 25
def total_votes (W : ℕ) : ℕ := votes_for_witch W + votes_for_unicorn W + votes_for_dragon W

-- Proof Statement
theorem witch_votes_is_seven (W : ℕ) (h1 : total_votes W = 60) : W = 7 :=
by
  sorry

end NUMINAMATH_GPT_witch_votes_is_seven_l2412_241266


namespace NUMINAMATH_GPT_tom_gas_spending_l2412_241237

-- Defining the conditions given in the problem
def miles_per_gallon := 50
def miles_per_day := 75
def gas_price := 3
def number_of_days := 10

-- Defining the main theorem to be proven
theorem tom_gas_spending : 
  (miles_per_day * number_of_days) / miles_per_gallon * gas_price = 45 := 
by 
  sorry

end NUMINAMATH_GPT_tom_gas_spending_l2412_241237


namespace NUMINAMATH_GPT_minimum_value_f_l2412_241280

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end NUMINAMATH_GPT_minimum_value_f_l2412_241280


namespace NUMINAMATH_GPT_probability_not_monday_l2412_241259

theorem probability_not_monday (P_monday : ℚ) (h : P_monday = 1/7) : P_monday ≠ 1 → ∃ P_not_monday : ℚ, P_not_monday = 6/7 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_monday_l2412_241259


namespace NUMINAMATH_GPT_find_value_of_fraction_l2412_241269

theorem find_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > x) (h : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_fraction_l2412_241269


namespace NUMINAMATH_GPT_product_of_5_consecutive_integers_divisible_by_60_l2412_241226

theorem product_of_5_consecutive_integers_divisible_by_60 :
  ∀a : ℤ, 60 ∣ (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_5_consecutive_integers_divisible_by_60_l2412_241226


namespace NUMINAMATH_GPT_melted_mixture_weight_l2412_241202

variable (zinc copper total_weight : ℝ)
variable (ratio_zinc ratio_copper : ℝ := 9 / 11)
variable (weight_zinc : ℝ := 31.5)

theorem melted_mixture_weight :
  (zinc / copper = ratio_zinc / ratio_copper) ∧ (zinc = weight_zinc) →
  (total_weight = zinc + copper) →
  total_weight = 70 := 
sorry

end NUMINAMATH_GPT_melted_mixture_weight_l2412_241202


namespace NUMINAMATH_GPT_boat_speed_still_water_l2412_241272

theorem boat_speed_still_water (b s : ℝ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := 
by 
  -- Solve the system of equations
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l2412_241272


namespace NUMINAMATH_GPT_light_bulbs_circle_l2412_241290

theorem light_bulbs_circle : ∀ (f : ℕ → ℕ),
  (f 0 = 1) ∧
  (f 1 = 2) ∧
  (f 2 = 4) ∧
  (f 3 = 8) ∧
  (∀ n, f n = f (n - 1) + f (n - 2) + f (n - 3) + f (n - 4)) →
  (f 9 - 3 * f 3 - 2 * f 2 - f 1 = 367) :=
by
  sorry

end NUMINAMATH_GPT_light_bulbs_circle_l2412_241290


namespace NUMINAMATH_GPT_units_digit_of_27_mul_36_l2412_241217

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_27_mul_36_l2412_241217


namespace NUMINAMATH_GPT_stacked_lego_volume_l2412_241209

theorem stacked_lego_volume 
  (lego_volume : ℝ)
  (rows columns layers : ℕ)
  (h1 : lego_volume = 1)
  (h2 : rows = 7)
  (h3 : columns = 5)
  (h4 : layers = 3) :
  rows * columns * layers * lego_volume = 105 :=
by
  sorry

end NUMINAMATH_GPT_stacked_lego_volume_l2412_241209


namespace NUMINAMATH_GPT_theater_ticket_sales_l2412_241274

theorem theater_ticket_sales
  (A C : ℕ)
  (h₁ : 8 * A + 5 * C = 236)
  (h₂ : A + C = 34) : A = 22 :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_sales_l2412_241274


namespace NUMINAMATH_GPT_julia_gold_watch_percentage_l2412_241227

def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def total_watches_before_gold : ℕ := silver_watches + bronze_watches
def total_watches_after_gold : ℕ := 88
def gold_watches : ℕ := total_watches_after_gold - total_watches_before_gold
def percentage_gold_watches : ℚ := (gold_watches : ℚ) / (total_watches_after_gold : ℚ) * 100

theorem julia_gold_watch_percentage :
  percentage_gold_watches = 9.09 := by
  sorry

end NUMINAMATH_GPT_julia_gold_watch_percentage_l2412_241227


namespace NUMINAMATH_GPT_bottle_caps_per_box_l2412_241219

theorem bottle_caps_per_box (total_bottle_caps boxes : ℕ) (hb : total_bottle_caps = 316) (bn : boxes = 79) :
  total_bottle_caps / boxes = 4 :=
by
  sorry

end NUMINAMATH_GPT_bottle_caps_per_box_l2412_241219


namespace NUMINAMATH_GPT_koi_fish_after_three_weeks_l2412_241233

theorem koi_fish_after_three_weeks
  (f_0 : ℕ := 280) -- initial total number of fish
  (days : ℕ := 21) -- days in 3 weeks
  (koi_added_per_day : ℕ := 2)
  (goldfish_added_per_day : ℕ := 5)
  (goldfish_after_3_weeks : ℕ := 200) :
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  koi_after_3_weeks = 227 :=
by
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  sorry

end NUMINAMATH_GPT_koi_fish_after_three_weeks_l2412_241233


namespace NUMINAMATH_GPT_sum_of_first_100_terms_AP_l2412_241296

theorem sum_of_first_100_terms_AP (a d : ℕ) :
  (15 / 2) * (2 * a + 14 * d) = 45 →
  (85 / 2) * (2 * a + 84 * d) = 255 →
  (100 / 2) * (2 * a + 99 * d) = 300 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_100_terms_AP_l2412_241296


namespace NUMINAMATH_GPT_kids_stay_home_correct_l2412_241251

def total_number_of_kids : ℕ := 1363293
def kids_who_go_to_camp : ℕ := 455682
def kids_staying_home : ℕ := total_number_of_kids - kids_who_go_to_camp

theorem kids_stay_home_correct :
  kids_staying_home = 907611 := by 
  sorry

end NUMINAMATH_GPT_kids_stay_home_correct_l2412_241251


namespace NUMINAMATH_GPT_abscissa_of_tangent_point_is_2_l2412_241271

noncomputable def f (x : ℝ) : ℝ := (x^2) / 4 - 3 * Real.log x

noncomputable def f' (x : ℝ) : ℝ := (1/2) * x - 3 / x

theorem abscissa_of_tangent_point_is_2 : 
  ∃ x0 : ℝ, f' x0 = -1/2 ∧ x0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_abscissa_of_tangent_point_is_2_l2412_241271


namespace NUMINAMATH_GPT_sequence_general_term_l2412_241210

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ k ≥ 1, a (k + 1) = 2 * a k) : a n = 2 ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l2412_241210


namespace NUMINAMATH_GPT_find_m_l2412_241281

-- Define the condition that the equation has a positive root
def hasPositiveRoot (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (2 / (x - 2) = 1 - (m / (x - 2)))

-- State the theorem
theorem find_m : ∀ m : ℝ, hasPositiveRoot m → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2412_241281


namespace NUMINAMATH_GPT_units_digit_of_G_1000_l2412_241276

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_of_G_1000 : (G 1000) % 10 = 2 := 
  sorry

end NUMINAMATH_GPT_units_digit_of_G_1000_l2412_241276


namespace NUMINAMATH_GPT_find_rate_of_current_l2412_241277

-- Define the conditions
def speed_in_still_water (speed : ℝ) : Prop := speed = 15
def distance_downstream (distance : ℝ) : Prop := distance = 7.2
def time_in_hours (time : ℝ) : Prop := time = 0.4

-- Define the effective speed downstream
def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ := boat_speed + current_speed

-- Define rate of current
def rate_of_current (current_speed : ℝ) : Prop :=
  ∃ (c : ℝ), effective_speed_downstream 15 c * 0.4 = 7.2 ∧ c = current_speed

-- The theorem stating the proof problem
theorem find_rate_of_current : rate_of_current 3 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_of_current_l2412_241277


namespace NUMINAMATH_GPT_arith_seq_a12_value_l2412_241224

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ (a₄ : ℝ), a 4 = 1 ∧ a 7 = a 4 + 3 * d ∧ a 9 = a 4 + 5 * d

theorem arith_seq_a12_value
  (h₁ : arithmetic_sequence a (13 / 8))
  (h₂ : a 7 + a 9 = 15)
  (h₃ : a 4 = 1) :
  a 12 = 14 :=
sorry

end NUMINAMATH_GPT_arith_seq_a12_value_l2412_241224


namespace NUMINAMATH_GPT_intersection_P_Q_l2412_241291

def P (x : ℝ) : Prop := x^2 - x - 2 ≥ 0

def Q (y : ℝ) : Prop := ∃ x, P x ∧ y = (1/2) * x^2 - 1

theorem intersection_P_Q :
  {m | ∃ (x : ℝ), P x ∧ m = (1/2) * x^2 - 1} = {m | m ≥ 2} := sorry

end NUMINAMATH_GPT_intersection_P_Q_l2412_241291


namespace NUMINAMATH_GPT_longest_boat_length_l2412_241218

variable (saved money : ℕ) (license_fee docking_multiplier boat_cost : ℕ)

theorem longest_boat_length (h1 : saved = 20000) 
                           (h2 : license_fee = 500) 
                           (h3 : docking_multiplier = 3)
                           (h4 : boat_cost = 1500) : 
                           (saved - license_fee - docking_multiplier * license_fee) / boat_cost = 12 := 
by 
  sorry

end NUMINAMATH_GPT_longest_boat_length_l2412_241218


namespace NUMINAMATH_GPT_strings_completely_pass_each_other_l2412_241275

-- Define the problem parameters
def d : ℝ := 30    -- distance between A and B in cm
def l1 : ℝ := 151  -- length of string A in cm
def l2 : ℝ := 187  -- length of string B in cm
def v1 : ℝ := 2    -- speed of string A in cm/s
def v2 : ℝ := 3    -- speed of string B in cm/s
def r1 : ℝ := 1    -- burn rate of string A in cm/s
def r2 : ℝ := 2    -- burn rate of string B in cm/s

-- The proof problem statement
theorem strings_completely_pass_each_other : ∀ (T : ℝ), T = 40 :=
by
  sorry

end NUMINAMATH_GPT_strings_completely_pass_each_other_l2412_241275


namespace NUMINAMATH_GPT_evaluate_expression_l2412_241216

theorem evaluate_expression : (-1 : ℤ)^(3^3) + (1 : ℤ)^(3^3) = 0 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2412_241216


namespace NUMINAMATH_GPT_rationalize_denominator_l2412_241200

theorem rationalize_denominator :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2412_241200


namespace NUMINAMATH_GPT_hyperbola_asymptote_perpendicular_to_line_l2412_241235

variable {a : ℝ}

theorem hyperbola_asymptote_perpendicular_to_line (h : a > 0)
  (C : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1)
  (l : ∀ x y : ℝ, 2 * x - y + 1 = 0) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_perpendicular_to_line_l2412_241235


namespace NUMINAMATH_GPT_greatest_prime_factor_391_l2412_241252

theorem greatest_prime_factor_391 : ∃ p, Prime p ∧ p ∣ 391 ∧ ∀ q, Prime q ∧ q ∣ 391 → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_391_l2412_241252


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2412_241282

variable {R : Type*} [LinearOrderedField R]
variable (f : R × R → R)
variable (x₀ y₀ : R)

theorem necessary_and_sufficient_condition :
  (f (x₀, y₀) = 0) ↔ ((x₀, y₀) ∈ {p : R × R | f p = 0}) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2412_241282


namespace NUMINAMATH_GPT_average_speed_of_car_l2412_241265

theorem average_speed_of_car (time : ℝ) (distance : ℝ) (h_time : time = 4.5) (h_distance : distance = 360) : 
  distance / time = 80 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l2412_241265


namespace NUMINAMATH_GPT_simple_interest_rate_l2412_241278

-- Definitions based on conditions
def principal : ℝ := 750
def amount : ℝ := 900
def time : ℕ := 10

-- Statement to prove the rate of simple interest
theorem simple_interest_rate : 
  ∃ (R : ℝ), principal * R * time / 100 = amount - principal ∧ R = 2 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2412_241278


namespace NUMINAMATH_GPT_interior_angle_of_regular_pentagon_is_108_l2412_241289

-- Define the sum of angles in a triangle
def sum_of_triangle_angles : ℕ := 180

-- Define the number of triangles in a convex pentagon
def num_of_triangles_in_pentagon : ℕ := 3

-- Define the total number of interior angles in a pentagon
def num_of_angles_in_pentagon : ℕ := 5

-- Define the total sum of the interior angles of a pentagon
def sum_of_pentagon_interior_angles : ℕ := num_of_triangles_in_pentagon * sum_of_triangle_angles

-- Define the degree measure of an interior angle of a regular pentagon
def interior_angle_of_regular_pentagon : ℕ := sum_of_pentagon_interior_angles / num_of_angles_in_pentagon

theorem interior_angle_of_regular_pentagon_is_108 :
  interior_angle_of_regular_pentagon = 108 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_interior_angle_of_regular_pentagon_is_108_l2412_241289


namespace NUMINAMATH_GPT_area_of_rectangle_given_conditions_l2412_241214

-- Defining the conditions given in the problem
variables (s d r a : ℝ)

-- Given conditions for the problem
def is_square_inscribed_in_circle (s d : ℝ) := 
  d = s * Real.sqrt 2 ∧ 
  d = 4

def is_circle_inscribed_in_rectangle (r : ℝ) :=
  r = 2

def rectangle_dimensions (length width : ℝ) :=
  length = 2 * width ∧ 
  width = 2

-- The theorem we want to prove
theorem area_of_rectangle_given_conditions :
  ∀ (s d r length width : ℝ),
  is_square_inscribed_in_circle s d →
  is_circle_inscribed_in_rectangle r →
  rectangle_dimensions length width →
  a = length * width →
  a = 8 :=
by
  intros s d r length width h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_area_of_rectangle_given_conditions_l2412_241214


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2412_241223

theorem equation_one_solution (x : ℕ) : 8 * (x + 1)^3 = 64 ↔ x = 1 := by 
  sorry

theorem equation_two_solution (x : ℤ) : (x + 1)^2 = 100 ↔ x = 9 ∨ x = -11 := by 
  sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l2412_241223


namespace NUMINAMATH_GPT_roots_of_quadratic_l2412_241299

theorem roots_of_quadratic (x : ℝ) : (x - 3) ^ 2 = 25 ↔ (x = 8 ∨ x = -2) :=
by sorry

end NUMINAMATH_GPT_roots_of_quadratic_l2412_241299


namespace NUMINAMATH_GPT_heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l2412_241234

namespace PolygonColoring

/-- Define a regular n-gon and its coloring -/
def regular_ngon (n : ℕ) : Type := sorry

def isosceles_triangle {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

def same_color {n : ℕ} (p : regular_ngon n) (v1 v2 v3 : ℕ) : Prop := sorry

/-- Part (a) statement -/
theorem heptagon_isosceles_triangle_same_color : 
  ∀ (p : regular_ngon 7), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (b) statement -/
theorem octagon_no_isosceles_triangle_same_color :
  ∃ (p : regular_ngon 8), ¬∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

/-- Part (c) statement -/
theorem general_ngon_isosceles_triangle_same_color :
  ∀ (n : ℕ), (n = 5 ∨ n = 7 ∨ n ≥ 9) → 
  ∀ (p : regular_ngon n), ∃ (v1 v2 v3 : ℕ), isosceles_triangle p v1 v2 v3 ∧ same_color p v1 v2 v3 := 
by
  sorry

end PolygonColoring

end NUMINAMATH_GPT_heptagon_isosceles_triangle_same_color_octagon_no_isosceles_triangle_same_color_general_ngon_isosceles_triangle_same_color_l2412_241234


namespace NUMINAMATH_GPT_planes_perpendicular_of_line_conditions_l2412_241264

variables (a b l : Line) (M N : Plane)

-- Definitions of lines and planes and their relations
def parallel_to_plane (a : Line) (M : Plane) : Prop := sorry
def perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry
def subset_of_plane (a : Line) (M : Plane) : Prop := sorry

-- Statement of the main theorem to be proved
theorem planes_perpendicular_of_line_conditions (a b l : Line) (M N : Plane) :
  (perpendicular_to_plane a M) → (parallel_to_plane a N) → (perpendicular_to_plane N M) :=
  by
  sorry

end NUMINAMATH_GPT_planes_perpendicular_of_line_conditions_l2412_241264


namespace NUMINAMATH_GPT_find_value_l2412_241207

theorem find_value 
    (x y : ℝ) 
    (hx : x = 1 / (Real.sqrt 2 + 1)) 
    (hy : y = 1 / (Real.sqrt 2 - 1)) : 
    x^2 - 3 * x * y + y^2 = 3 := 
by 
    sorry

end NUMINAMATH_GPT_find_value_l2412_241207


namespace NUMINAMATH_GPT_domain_of_expression_l2412_241287

theorem domain_of_expression (x : ℝ) :
  (1 ≤ x ∧ x < 6) ↔ (∃ y : ℝ, y = (x-1) ∧ y = (6-x) ∧ 0 ≤ y) :=
sorry

end NUMINAMATH_GPT_domain_of_expression_l2412_241287


namespace NUMINAMATH_GPT_no_real_solutions_l2412_241285

theorem no_real_solutions :
  ∀ x y z : ℝ, ¬ (x + y + 2 + 4*x*y = 0 ∧ y + z + 2 + 4*y*z = 0 ∧ z + x + 2 + 4*z*x = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l2412_241285


namespace NUMINAMATH_GPT_olivia_savings_l2412_241238

noncomputable def compound_amount 
  (P : ℝ) -- Initial principal
  (r : ℝ) -- Annual interest rate
  (n : ℕ) -- Number of times interest is compounded per year
  (t : ℕ) -- Number of years
  : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem olivia_savings :
  compound_amount 2500 0.045 2 21 = 5077.14 :=
by
  sorry

end NUMINAMATH_GPT_olivia_savings_l2412_241238


namespace NUMINAMATH_GPT_count_right_triangles_with_conditions_l2412_241260

theorem count_right_triangles_with_conditions :
  ∃ n : ℕ, n = 10 ∧
    (∀ (a b : ℕ),
      (a ^ 2 + b ^ 2 = (b + 2) ^ 2) →
      (b < 100) →
      (∃ k : ℕ, a = 2 * k ∧ k ^ 2 = b + 1) →
      n = 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_count_right_triangles_with_conditions_l2412_241260


namespace NUMINAMATH_GPT_average_is_five_plus_D_over_two_l2412_241213

variable (A B C D : ℝ)

def condition1 := 1001 * C - 2004 * A = 4008
def condition2 := 1001 * B + 3005 * A - 1001 * D = 6010

theorem average_is_five_plus_D_over_two (h1 : condition1 A C) (h2 : condition2 A B D) : 
  (A + B + C + D) / 4 = (5 + D) / 2 := 
by
  sorry

end NUMINAMATH_GPT_average_is_five_plus_D_over_two_l2412_241213


namespace NUMINAMATH_GPT_least_k_9_l2412_241288

open Nat

noncomputable def u : ℕ → ℝ
| 0     => 1 / 3
| (n+1) => 3 * u n - 3 * (u n) * (u n)

def M : ℝ := 0.5

def acceptable_error (n : ℕ): Prop := abs (u n - M) ≤ 1 / 2 ^ 500

theorem least_k_9 : ∃ k, 0 ≤ k ∧ acceptable_error k ∧ ∀ j, (0 ≤ j ∧ j < k) → ¬acceptable_error j ∧ k = 9 := by
  sorry

end NUMINAMATH_GPT_least_k_9_l2412_241288


namespace NUMINAMATH_GPT_solve_math_problem_l2412_241241

theorem solve_math_problem (x : ℕ) (h1 : x > 0) (h2 : x % 3 = 0) (h3 : x % x = 9) : x = 30 := by
  sorry

end NUMINAMATH_GPT_solve_math_problem_l2412_241241


namespace NUMINAMATH_GPT_average_output_l2412_241211

theorem average_output (time1 time2 rate1 rate2 cogs1 cogs2 total_cogs total_time: ℝ) :
  rate1 = 20 → cogs1 = 60 → time1 = cogs1 / rate1 →
  rate2 = 60 → cogs2 = 60 → time2 = cogs2 / rate2 →
  total_cogs = cogs1 + cogs2 → total_time = time1 + time2 →
  (total_cogs / total_time = 30) :=
by
  intros hrate1 hcogs1 htime1 hrate2 hcogs2 htime2 htotalcogs htotaltime
  sorry

end NUMINAMATH_GPT_average_output_l2412_241211


namespace NUMINAMATH_GPT_graph_not_pass_first_quadrant_l2412_241257

theorem graph_not_pass_first_quadrant (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ¬ (∃ x y : ℝ, y = a^x + b ∧ x > 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_graph_not_pass_first_quadrant_l2412_241257


namespace NUMINAMATH_GPT_find_k_l2412_241247

def vector (α : Type) := (α × α)
def a : vector ℝ := (1, 3)
def b (k : ℝ) : vector ℝ := (-2, k)
def add (v1 v2 : vector ℝ) : vector ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def smul (c : ℝ) (v : vector ℝ) : vector ℝ := (c * v.1, c * v.2)
def cross_product (v1 v2 : vector ℝ) : ℝ := v1.1 * v2.2 - v1.2 * v2.1

theorem find_k (k : ℝ) (h : cross_product (add a (smul 2 (b k)))
                                          (add (smul 3 a) (smul (-1) (b k))) = 0) : k = -6 :=
sorry

end NUMINAMATH_GPT_find_k_l2412_241247


namespace NUMINAMATH_GPT_abs_value_expression_l2412_241293

theorem abs_value_expression (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 :=
sorry

end NUMINAMATH_GPT_abs_value_expression_l2412_241293


namespace NUMINAMATH_GPT_max_sum_42_l2412_241294

noncomputable def max_horizontal_vertical_sum (numbers : List ℕ) : ℕ :=
  let a := 14
  let b := 11
  let e := 17
  a + b + e

theorem max_sum_42 : 
  max_horizontal_vertical_sum [2, 5, 8, 11, 14, 17] = 42 := by
  sorry

end NUMINAMATH_GPT_max_sum_42_l2412_241294


namespace NUMINAMATH_GPT_monotonic_f_iff_l2412_241248

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a * x + 5 else 1 + 1 / x

theorem monotonic_f_iff {a : ℝ} :  
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_f_iff_l2412_241248


namespace NUMINAMATH_GPT_erik_orange_juice_count_l2412_241244

theorem erik_orange_juice_count (initial_money bread_loaves bread_cost orange_juice_cost remaining_money : ℤ)
  (h₁ : initial_money = 86)
  (h₂ : bread_loaves = 3)
  (h₃ : bread_cost = 3)
  (h₄ : orange_juice_cost = 6)
  (h₅ : remaining_money = 59) :
  (initial_money - remaining_money - (bread_loaves * bread_cost)) / orange_juice_cost = 3 :=
by
  sorry

end NUMINAMATH_GPT_erik_orange_juice_count_l2412_241244


namespace NUMINAMATH_GPT_Jackson_to_Williams_Ratio_l2412_241212

-- Define the amounts of money Jackson and Williams have, given the conditions.
def JacksonMoney : ℤ := 125
def TotalMoney : ℤ := 150
-- Define Williams' money based on the given conditions.
def WilliamsMoney : ℤ := TotalMoney - JacksonMoney

-- State the theorem that the ratio of Jackson's money to Williams' money is 5:1
theorem Jackson_to_Williams_Ratio : JacksonMoney / WilliamsMoney = 5 := 
by
  -- Proof steps are omitted as per the instruction.
  sorry

end NUMINAMATH_GPT_Jackson_to_Williams_Ratio_l2412_241212


namespace NUMINAMATH_GPT_adam_completes_work_in_10_days_l2412_241205

theorem adam_completes_work_in_10_days (W : ℝ) (A : ℝ)
  (h1 : (W / 25) + A = W / 20) :
  W / 10 = (W / 100) * 10 :=
by
  sorry

end NUMINAMATH_GPT_adam_completes_work_in_10_days_l2412_241205


namespace NUMINAMATH_GPT_age_difference_is_20_l2412_241246

-- Definitions for the ages of the two persons
def elder_age := 35
def younger_age := 15

-- Condition: Difference in ages
def age_difference := elder_age - younger_age

-- Theorem to prove the difference in ages is 20 years
theorem age_difference_is_20 : age_difference = 20 := by
  sorry

end NUMINAMATH_GPT_age_difference_is_20_l2412_241246


namespace NUMINAMATH_GPT_denominator_is_five_l2412_241256

-- Define the conditions
variables (n d : ℕ)
axiom h1 : d = n - 4
axiom h2 : n + 6 = 3 * d

-- The theorem that needs to be proven
theorem denominator_is_five : d = 5 :=
by
  sorry

end NUMINAMATH_GPT_denominator_is_five_l2412_241256


namespace NUMINAMATH_GPT_julieta_total_spent_l2412_241297

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end NUMINAMATH_GPT_julieta_total_spent_l2412_241297


namespace NUMINAMATH_GPT_sum_of_ages_today_l2412_241286

variable (RizaWas25WhenSonBorn : ℕ) (SonCurrentAge : ℕ) (SumOfAgesToday : ℕ)

theorem sum_of_ages_today (h1 : RizaWas25WhenSonBorn = 25) (h2 : SonCurrentAge = 40) : SumOfAgesToday = 105 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_today_l2412_241286


namespace NUMINAMATH_GPT_marching_band_formations_l2412_241273

/-- A marching band of 240 musicians can be arranged in p different rectangular formations 
with s rows and t musicians per row where 8 ≤ t ≤ 30. 
This theorem asserts that there are 8 such different rectangular formations. -/
theorem marching_band_formations (s t : ℕ) (h : s * t = 240) (h_t_bounds : 8 ≤ t ∧ t ≤ 30) : 
  ∃ p : ℕ, p = 8 := 
sorry

end NUMINAMATH_GPT_marching_band_formations_l2412_241273


namespace NUMINAMATH_GPT_erased_digit_is_4_l2412_241267

def sum_of_digits (n : ℕ) : ℕ := 
  sorry -- definition of sum of digits

def D (N : ℕ) : ℕ := N - sum_of_digits N

theorem erased_digit_is_4 (N : ℕ) (x : ℕ) 
  (hD : D N % 9 = 0) 
  (h_sum : sum_of_digits (D N) - x = 131) 
  : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_erased_digit_is_4_l2412_241267


namespace NUMINAMATH_GPT_honey_last_nights_l2412_241215

def servings_per_cup : Nat := 1
def cups_per_night : Nat := 2
def container_ounces : Nat := 16
def servings_per_ounce : Nat := 6

theorem honey_last_nights :
  (container_ounces * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 :=
by
  sorry  -- Proof not provided as per requirements

end NUMINAMATH_GPT_honey_last_nights_l2412_241215


namespace NUMINAMATH_GPT_intersection_A_B_l2412_241201

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | ∃ y : ℝ, y = x^2 + 2}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ ∃ y : ℝ, y = x^2 + 2} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end NUMINAMATH_GPT_intersection_A_B_l2412_241201


namespace NUMINAMATH_GPT_number_one_fourth_more_than_it_is_30_percent_less_than_80_l2412_241292

theorem number_one_fourth_more_than_it_is_30_percent_less_than_80 :
    ∃ (n : ℝ), (5 / 4) * n = 56 ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_one_fourth_more_than_it_is_30_percent_less_than_80_l2412_241292


namespace NUMINAMATH_GPT_remainder_when_divided_by_s_minus_2_l2412_241203

noncomputable def f (s : ℤ) : ℤ := s^15 + s^2 + 3

theorem remainder_when_divided_by_s_minus_2 : f 2 = 32775 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_s_minus_2_l2412_241203


namespace NUMINAMATH_GPT_original_sales_tax_percentage_l2412_241242

theorem original_sales_tax_percentage
  (current_sales_tax : ℝ := 10 / 3) -- 3 1/3% in decimal
  (difference : ℝ := 10.999999999999991) -- Rs. 10.999999999999991
  (market_price : ℝ := 6600) -- Rs. 6600
  (original_sales_tax : ℝ := 3.5) -- Expected original tax
  :  ((original_sales_tax / 100) * market_price = (current_sales_tax / 100) * market_price + difference) 
  := sorry

end NUMINAMATH_GPT_original_sales_tax_percentage_l2412_241242


namespace NUMINAMATH_GPT_ceiling_lights_l2412_241249

variable (S M L : ℕ)

theorem ceiling_lights (hM : M = 12) (hL : L = 2 * M)
    (hBulbs : S + 2 * M + 3 * L = 118) : S - M = 10 :=
by
  sorry

end NUMINAMATH_GPT_ceiling_lights_l2412_241249


namespace NUMINAMATH_GPT_flower_bed_area_l2412_241268

theorem flower_bed_area (total_posts : ℕ) (corner_posts : ℕ) (spacing : ℕ) (long_side_multiplier : ℕ)
  (h1 : total_posts = 24)
  (h2 : corner_posts = 4)
  (h3 : spacing = 3)
  (h4 : long_side_multiplier = 3) :
  ∃ (area : ℕ), area = 144 := 
sorry

end NUMINAMATH_GPT_flower_bed_area_l2412_241268


namespace NUMINAMATH_GPT_trajectory_of_center_l2412_241222

-- Define the given conditions
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

def tangent_y_axis (x : ℝ) : Prop := x = 0

-- Define the theorem with the given conditions and the desired conclusion
theorem trajectory_of_center (x y : ℝ) (h1 : tangent_circle x y) (h2 : tangent_y_axis x) :
  (y^2 = 8 * x) ∨ (y = 0 ∧ x ≤ 0) :=
sorry

end NUMINAMATH_GPT_trajectory_of_center_l2412_241222


namespace NUMINAMATH_GPT_domain_of_function_l2412_241255

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 3 * x + 1 > 0
def condition2 (x : ℝ) : Prop := 2 - x ≠ 0

-- Define the domain of the function
def domain (x : ℝ) : Prop := x > -1 / 3 ∧ x ≠ 2

theorem domain_of_function : 
  ∀ x : ℝ, (condition1 x ∧ condition2 x) ↔ domain x := 
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2412_241255


namespace NUMINAMATH_GPT_q_at_14_l2412_241283

noncomputable def q (x : ℝ) : ℝ := - (1 / 2) * x^2 + x + 2

theorem q_at_14 : q 14 = -82 := by
  sorry

end NUMINAMATH_GPT_q_at_14_l2412_241283


namespace NUMINAMATH_GPT_positive_difference_between_numbers_l2412_241295

theorem positive_difference_between_numbers:
  ∃ x y : ℤ, x + y = 40 ∧ 3 * y - 4 * x = 7 ∧ |y - x| = 6 := by
  sorry

end NUMINAMATH_GPT_positive_difference_between_numbers_l2412_241295


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l2412_241279

theorem equilateral_triangle_perimeter (s : ℕ) (h1 : 2 * s + 10 = 50) : 3 * s = 60 :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l2412_241279


namespace NUMINAMATH_GPT_convert_speed_l2412_241228

theorem convert_speed (v_m_s : ℚ) (conversion_factor : ℚ) :
  v_m_s = 12 / 43 → conversion_factor = 3.6 → v_m_s * conversion_factor = 1.0046511624 := by
  intros h1 h2
  have h3 : v_m_s = 12 / 43 := h1
  have h4 : conversion_factor = 3.6 := h2
  rw [h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_convert_speed_l2412_241228


namespace NUMINAMATH_GPT_find_ratio_of_square_to_circle_radius_l2412_241236

def sector_circle_ratio (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) : Prop :=
  (R = (5 * a * sqrt2) / 2) →
  (r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) →
  (a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2))

theorem find_ratio_of_square_to_circle_radius
  (a R : ℝ) (r : ℝ) (sqrt5 sqrt2 : ℝ) (h1 : R = (5 * a * sqrt2) / 2)
  (h2 : r = (a * (sqrt5 + sqrt2) * (3 + sqrt5)) / (6 * sqrt2)) :
  a / R = (sqrt5 + sqrt2) * (3 + sqrt5) / (6 * sqrt2) :=
  sorry

end NUMINAMATH_GPT_find_ratio_of_square_to_circle_radius_l2412_241236


namespace NUMINAMATH_GPT_city_miles_count_l2412_241258

-- Defining the variables used in the conditions
def miles_per_gallon_city : ℝ := 30
def miles_per_gallon_highway : ℝ := 40
def highway_miles : ℝ := 200
def cost_per_gallon : ℝ := 3
def total_cost : ℝ := 42

-- Required statement for the proof, statement to prove: count of city miles is 270
theorem city_miles_count : ∃ (C : ℝ), C = 270 ∧
  (total_cost / cost_per_gallon) = ((C / miles_per_gallon_city) + (highway_miles / miles_per_gallon_highway)) :=
by
  sorry

end NUMINAMATH_GPT_city_miles_count_l2412_241258


namespace NUMINAMATH_GPT_interval_of_a_l2412_241243

theorem interval_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_monotone : ∀ x y, x < y → f y ≤ f x)
  (h_condition : f (2 * a^2 + a + 1) < f (3 * a^2 - 4 * a + 1)) : 
  a ∈ Set.Ioo 0 (1/3) ∪ Set.Ioo 1 5 :=
by
  sorry

end NUMINAMATH_GPT_interval_of_a_l2412_241243


namespace NUMINAMATH_GPT_quadratic_solution_l2412_241253

-- Definition of the quadratic function satisfying the given conditions
def quadraticFunc (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (f (-1) = 12 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → f x ≤ 12)

-- The proof goal: proving the function f(x) is 2x^2 - 10x
theorem quadratic_solution (f : ℝ → ℝ) (h : quadraticFunc f) : ∀ x, f x = 2 * x^2 - 10 * x :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l2412_241253


namespace NUMINAMATH_GPT_find_ordered_triplets_l2412_241254

theorem find_ordered_triplets (x y z : ℝ) :
  x^3 = z / y - 2 * y / z ∧
  y^3 = x / z - 2 * z / x ∧
  z^3 = y / x - 2 * x / y →
  (x = 1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) :=
sorry

end NUMINAMATH_GPT_find_ordered_triplets_l2412_241254


namespace NUMINAMATH_GPT_remainder_of_poly_div_l2412_241240

theorem remainder_of_poly_div (n : ℕ) (h : n > 2) : (n^3 + 3) % (n + 1) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_poly_div_l2412_241240


namespace NUMINAMATH_GPT_bus_distance_covered_l2412_241245

theorem bus_distance_covered (speedTrain speedCar speedBus distanceBus : ℝ) (h1 : speedTrain / speedCar = 16 / 15)
                            (h2 : speedBus = (3 / 4) * speedTrain) (h3 : 450 / 6 = speedCar) (h4 : distanceBus = 8 * speedBus) :
                            distanceBus = 480 :=
by
  sorry

end NUMINAMATH_GPT_bus_distance_covered_l2412_241245


namespace NUMINAMATH_GPT_find_value_of_a_l2412_241221

-- Given conditions
def equation1 (x y : ℝ) : Prop := 4 * y + x + 5 = 0
def equation2 (x y : ℝ) (a : ℝ) : Prop := 3 * y + a * x + 4 = 0

-- The proof problem statement
theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, equation1 x y ∧ equation2 x y a → a = -12) :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l2412_241221


namespace NUMINAMATH_GPT_correct_propositions_identification_l2412_241220

theorem correct_propositions_identification (x y : ℝ) (h1 : x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)
    (h2 : ¬(x * y ≥ 0 → x ≥ 0 ∧ y ≥ 0))
    (h3 : ¬(¬(x ≥ 0 ∧ y ≥ 0 → x * y ≥ 0)))
    (h4 : (¬(x * y ≥ 0) → ¬(x ≥ 0) ∨ ¬(y ≥ 0))) :
  true :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_correct_propositions_identification_l2412_241220


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l2412_241261

noncomputable def sellingPrice : ℝ := 850
noncomputable def profit : ℝ := 230
noncomputable def costPrice : ℝ := sellingPrice - profit

noncomputable def profitPercentage : ℝ :=
  (profit / costPrice) * 100

theorem profit_percentage_is_correct :
  profitPercentage = 37.10 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l2412_241261


namespace NUMINAMATH_GPT_player_A_wins_4_points_game_game_ends_after_5_points_l2412_241239

def prob_A_winning_when_serving : ℚ := 2 / 3
def prob_A_winning_when_B_serving : ℚ := 1 / 4
def prob_A_winning_in_4_points : ℚ := 1 / 12
def prob_game_ending_after_5_points : ℚ := 19 / 216

theorem player_A_wins_4_points_game :
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) = prob_A_winning_in_4_points := 
  sorry

theorem game_ends_after_5_points : 
  ((1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (1 - prob_A_winning_when_serving) * (prob_A_winning_when_B_serving) * 
  (prob_A_winning_when_serving)) + 
  ((prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (prob_A_winning_when_serving) * ((1 - prob_A_winning_when_B_serving)) * 
  (1 - prob_A_winning_when_serving)) = 
  prob_game_ending_after_5_points :=
  sorry

end NUMINAMATH_GPT_player_A_wins_4_points_game_game_ends_after_5_points_l2412_241239


namespace NUMINAMATH_GPT_distance_to_focus_parabola_l2412_241284

theorem distance_to_focus_parabola (F P : ℝ × ℝ) (hF : F = (0, -1/2))
  (hP : P = (1, 2)) (C : ℝ × ℝ → Prop)
  (hC : ∀ x, C (x, 2 * x^2)) : dist P F = 17 / 8 := by
sorry

end NUMINAMATH_GPT_distance_to_focus_parabola_l2412_241284


namespace NUMINAMATH_GPT_painters_needed_days_l2412_241263

-- Let P be the total work required in painter-work-days
def total_painter_work_days : ℕ := 5

-- Let E be the effective number of workers with advanced tools
def effective_workers : ℕ := 4

-- Define the number of days, we need to prove this equals 1.25
def days_to_complete_work (P E : ℕ) : ℚ := P / E

-- The main theorem to prove: for total_painter_work_days and effective_workers, the days to complete the work is 1.25
theorem painters_needed_days :
  days_to_complete_work total_painter_work_days effective_workers = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_painters_needed_days_l2412_241263


namespace NUMINAMATH_GPT_quadratic_coefficient_nonzero_l2412_241250

theorem quadratic_coefficient_nonzero (a : ℝ) (x : ℝ) :
  (a - 3) * x^2 - 3 * x - 4 = 0 → a ≠ 3 :=
sorry

end NUMINAMATH_GPT_quadratic_coefficient_nonzero_l2412_241250


namespace NUMINAMATH_GPT_family_vacation_days_l2412_241231

theorem family_vacation_days
  (rained_days : ℕ)
  (total_days : ℕ)
  (clear_mornings : ℕ)
  (H1 : rained_days = 13)
  (H2 : total_days = 18)
  (H3 : clear_mornings = 11) :
  total_days = 18 :=
by
  -- proof to be filled in here
  sorry

end NUMINAMATH_GPT_family_vacation_days_l2412_241231


namespace NUMINAMATH_GPT_vector_parallel_l2412_241232

theorem vector_parallel (x y : ℝ) (a b : ℝ × ℝ × ℝ) (h_parallel : a = (2, 4, x) ∧ b = (2, y, 2) ∧ ∃ k : ℝ, a = k • b) : x + y = 6 :=
by sorry

end NUMINAMATH_GPT_vector_parallel_l2412_241232


namespace NUMINAMATH_GPT_initial_length_proof_l2412_241230

variables (L : ℕ)

-- Conditions from the problem statement
def condition1 (L : ℕ) : Prop := L - 25 > 118
def condition2 : Prop := 125 - 7 = 118
def initial_length : Prop := L = 143

-- Proof statement
theorem initial_length_proof (L : ℕ) (h1 : condition1 L) (h2 : condition2) : initial_length L :=
sorry

end NUMINAMATH_GPT_initial_length_proof_l2412_241230


namespace NUMINAMATH_GPT_inequality_solution_l2412_241208

noncomputable def solution_set : Set ℝ :=
  {x : ℝ | x < -2} ∪
  {x : ℝ | -2 < x ∧ x ≤ -1} ∪
  {x : ℝ | 1 ≤ x}

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x + 2)^2 ≥ 0} = solution_set := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2412_241208


namespace NUMINAMATH_GPT_certain_number_eq_0_08_l2412_241206

theorem certain_number_eq_0_08 (x : ℝ) (h : 1 / x = 12.5) : x = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_eq_0_08_l2412_241206


namespace NUMINAMATH_GPT_Roja_speed_is_8_l2412_241225

def Pooja_speed : ℝ := 3
def time_in_hours : ℝ := 4
def distance_between_them : ℝ := 44

theorem Roja_speed_is_8 :
  ∃ R : ℝ, R + Pooja_speed = (distance_between_them / time_in_hours) ∧ R = 8 :=
by
  sorry

end NUMINAMATH_GPT_Roja_speed_is_8_l2412_241225


namespace NUMINAMATH_GPT_assembly_time_constants_l2412_241229

theorem assembly_time_constants (a b : ℕ) (f : ℕ → ℝ)
  (h1 : ∀ x, f x = if x < b then a / (Real.sqrt x) else a / (Real.sqrt b))
  (h2 : f 4 = 15)
  (h3 : f b = 10) :
  a = 30 ∧ b = 9 :=
by
  sorry

end NUMINAMATH_GPT_assembly_time_constants_l2412_241229
