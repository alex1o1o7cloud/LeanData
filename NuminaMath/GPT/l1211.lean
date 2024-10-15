import Mathlib

namespace NUMINAMATH_GPT_no_such_function_exists_l1211_121169

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) ≤ -1

theorem no_such_function_exists : (∃ f : ℤ → ℤ, satisfies_condition f) = false :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l1211_121169


namespace NUMINAMATH_GPT_remainder_23_2057_mod_25_l1211_121131

theorem remainder_23_2057_mod_25 : (23^2057) % 25 = 16 := 
by
  sorry

end NUMINAMATH_GPT_remainder_23_2057_mod_25_l1211_121131


namespace NUMINAMATH_GPT_factorization_of_z6_minus_64_l1211_121164

theorem factorization_of_z6_minus_64 :
  ∀ (z : ℝ), (z^6 - 64) = (z - 2) * (z^2 + 2*z + 4) * (z + 2) * (z^2 - 2*z + 4) := 
by
  intros z
  sorry

end NUMINAMATH_GPT_factorization_of_z6_minus_64_l1211_121164


namespace NUMINAMATH_GPT_factor_x6_plus_8_l1211_121144

theorem factor_x6_plus_8 : (x^2 + 2) ∣ (x^6 + 8) :=
by
  sorry

end NUMINAMATH_GPT_factor_x6_plus_8_l1211_121144


namespace NUMINAMATH_GPT_sequence_a4_eq_5_over_3_l1211_121127

theorem sequence_a4_eq_5_over_3 :
  ∀ (a : ℕ → ℚ), a 1 = 1 → (∀ n > 1, a n = 1 / a (n - 1) + 1) → a 4 = 5 / 3 :=
by
  intro a ha1 H
  sorry

end NUMINAMATH_GPT_sequence_a4_eq_5_over_3_l1211_121127


namespace NUMINAMATH_GPT_math_and_science_students_l1211_121118

theorem math_and_science_students (x y : ℕ) 
  (h1 : x + y + 2 = 30)
  (h2 : y = 3 * x + 4) :
  y - 2 = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_and_science_students_l1211_121118


namespace NUMINAMATH_GPT_ratio_of_visible_spots_l1211_121153

theorem ratio_of_visible_spots (S S1 : ℝ) (h1 : ∀ (fold_type : ℕ), 
  (fold_type = 1 ∨ fold_type = 2 ∨ fold_type = 3) → 
  (if fold_type = 1 ∨ fold_type = 2 then S1 else S) = S1) : S1 / S = 2 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_visible_spots_l1211_121153


namespace NUMINAMATH_GPT_sufficient_not_necessary_range_l1211_121101

theorem sufficient_not_necessary_range (a : ℝ) (h : ∀ x : ℝ, x > 2 → x^2 > a ∧ ¬(x^2 > a → x > 2)) : a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_range_l1211_121101


namespace NUMINAMATH_GPT_quadrilateral_with_three_right_angles_is_rectangle_l1211_121167

-- Define a quadrilateral with angles
structure Quadrilateral :=
  (a1 a2 a3 a4 : ℝ)
  (sum_angles : a1 + a2 + a3 + a4 = 360)

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  is_right_angle q.a1 ∧ is_right_angle q.a2 ∧ is_right_angle q.a3 ∧ is_right_angle q.a4

-- The main theorem: if a quadrilateral has three right angles, it is a rectangle
theorem quadrilateral_with_three_right_angles_is_rectangle 
  (q : Quadrilateral) 
  (h1 : is_right_angle q.a1) 
  (h2 : is_right_angle q.a2) 
  (h3 : is_right_angle q.a3) 
  : is_rectangle q :=
sorry

end NUMINAMATH_GPT_quadrilateral_with_three_right_angles_is_rectangle_l1211_121167


namespace NUMINAMATH_GPT_determine_a_value_l1211_121150

-- Define the initial equation and conditions
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 1) - 3 / x = 1

-- Define the existence of a positive root
def has_positive_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ fractional_equation x a

-- The main theorem stating the correct value of 'a' for the given condition
theorem determine_a_value (x : ℝ) : has_positive_root 1 :=
sorry

end NUMINAMATH_GPT_determine_a_value_l1211_121150


namespace NUMINAMATH_GPT_weight_of_5_moles_BaO_molar_concentration_BaO_l1211_121113

-- Definitions based on conditions
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_O : ℝ := 16.00
def molar_mass_BaO : ℝ := atomic_mass_Ba + atomic_mass_O
def moles_BaO : ℝ := 5
def volume_solution : ℝ := 3

-- Theorem statements
theorem weight_of_5_moles_BaO : moles_BaO * molar_mass_BaO = 766.65 := by
  sorry

theorem molar_concentration_BaO : moles_BaO / volume_solution = 1.67 := by
  sorry

end NUMINAMATH_GPT_weight_of_5_moles_BaO_molar_concentration_BaO_l1211_121113


namespace NUMINAMATH_GPT_binom_np_n_mod_p2_l1211_121179

   theorem binom_np_n_mod_p2 (p n : ℕ) (hp : Nat.Prime p) : (Nat.choose (n * p) n) % (p ^ 2) = n % (p ^ 2) :=
   by
     sorry
   
end NUMINAMATH_GPT_binom_np_n_mod_p2_l1211_121179


namespace NUMINAMATH_GPT_set_intersection_l1211_121194

noncomputable def A : Set ℝ := { x | x / (x - 1) < 0 }
noncomputable def B : Set ℝ := { x | 0 < x ∧ x < 3 }
noncomputable def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem set_intersection (x : ℝ) : (x ∈ A ∧ x ∈ B) ↔ x ∈ expected_intersection :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1211_121194


namespace NUMINAMATH_GPT_visitors_answered_questionnaire_l1211_121104

theorem visitors_answered_questionnaire (V : ℕ) (h : (3 / 4 : ℝ) * V = (V : ℝ) - 110) : V = 440 :=
sorry

end NUMINAMATH_GPT_visitors_answered_questionnaire_l1211_121104


namespace NUMINAMATH_GPT_find_group_2018_l1211_121152

-- Definition of the conditions
def group_size (n : Nat) : Nat := 3 * n - 2

def total_numbers (n : Nat) : Nat := 
  (3 * n * n - n) / 2

theorem find_group_2018 : ∃ n : Nat, total_numbers (n - 1) < 1009 ∧ total_numbers n ≥ 1009 ∧ n = 27 :=
  by
  -- This forms the structure for the proof
  sorry

end NUMINAMATH_GPT_find_group_2018_l1211_121152


namespace NUMINAMATH_GPT_rain_at_house_l1211_121170

/-- Define the amounts of rain on the three days Greg was camping. -/
def rain_day1 : ℕ := 3
def rain_day2 : ℕ := 6
def rain_day3 : ℕ := 5

/-- Define the total rain experienced by Greg while camping. -/
def total_rain_camping := rain_day1 + rain_day2 + rain_day3

/-- Define the difference in the rain experienced by Greg while camping and at his house. -/
def rain_difference : ℕ := 12

/-- Define the total amount of rain at Greg's house. -/
def total_rain_house := total_rain_camping + rain_difference

/-- Prove that the total rain at Greg's house is 26 mm. -/
theorem rain_at_house : total_rain_house = 26 := by
  /- We know that total_rain_camping = 14 mm and rain_difference = 12 mm -/
  /- Therefore, total_rain_house = 14 mm + 12 mm = 26 mm -/
  sorry

end NUMINAMATH_GPT_rain_at_house_l1211_121170


namespace NUMINAMATH_GPT_loss_percent_l1211_121129

theorem loss_percent (CP SP Loss : ℝ) (h1 : CP = 600) (h2 : SP = 450) (h3 : Loss = CP - SP) : (Loss / CP) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_loss_percent_l1211_121129


namespace NUMINAMATH_GPT_power_of_expression_l1211_121137

theorem power_of_expression (a b c d e : ℝ)
  (h1 : a - b - c + d = 18)
  (h2 : a + b - c - d = 6)
  (h3 : c + d - e = 5) :
  (2 * b - d + e) ^ 3 = 13824 :=
by
  sorry

end NUMINAMATH_GPT_power_of_expression_l1211_121137


namespace NUMINAMATH_GPT_problem_statement_l1211_121112

variable (n : ℕ)
variable (op : ℕ → ℕ → ℕ)
variable (h1 : op 1 1 = 1)
variable (h2 : ∀ n, op (n+1) 1 = 3 * op n 1)

theorem problem_statement : op 5 1 - op 2 1 = 78 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1211_121112


namespace NUMINAMATH_GPT_regular_octagon_interior_angle_l1211_121145

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end NUMINAMATH_GPT_regular_octagon_interior_angle_l1211_121145


namespace NUMINAMATH_GPT_smallest_number_h_divisible_8_11_24_l1211_121192

theorem smallest_number_h_divisible_8_11_24 : 
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_h_divisible_8_11_24_l1211_121192


namespace NUMINAMATH_GPT_wendy_dentist_bill_l1211_121108

theorem wendy_dentist_bill : 
  let cost_cleaning := 70
  let cost_filling := 120
  let num_fillings := 3
  let cost_root_canal := 400
  let cost_dental_crown := 600
  let total_bill := 9 * cost_root_canal
  let known_costs := cost_cleaning + (num_fillings * cost_filling) + cost_root_canal + cost_dental_crown
  let cost_tooth_extraction := total_bill - known_costs
  cost_tooth_extraction = 2170 := by
  sorry

end NUMINAMATH_GPT_wendy_dentist_bill_l1211_121108


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1211_121134

-- Definitions of sets A, B, and C as per given conditions
def set_A (a : ℝ) : Set ℝ :=
  {x | x^2 - a * x + a^2 - 19 = 0}

def set_B : Set ℝ :=
  {x | x^2 - 5 * x + 6 = 0}

def set_C : Set ℝ :=
  {x | x^2 + 2 * x - 8 = 0}

-- Questions reformulated as proof problems
theorem problem1 (a : ℝ) (h : set_A a = set_B) : a = 5 :=
sorry

theorem problem2 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : ∀ x, x ∈ set_A a → x ∉ set_C) : a = -2 :=
sorry

theorem problem3 (a : ℝ) (h1 : ∃ x, x ∈ set_A a ∧ x ∈ set_B) (h2 : set_A a ∩ set_B = set_A a ∩ set_C) : a = -3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1211_121134


namespace NUMINAMATH_GPT_smallest_number_among_l1211_121114

theorem smallest_number_among
  (π : ℝ) (Hπ_pos : π > 0) :
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -1) → 
    (c = -1.5) → 
    (d = π) → 
    (∀ (x y : ℝ), (x > 0) → (y > 0) → (x > y) ↔ x - y > 0) → 
    (∀ (x : ℝ), x < 0 → x < 0) → 
    (∀ (x y : ℝ), (x > 0) → (y < 0) → x > y) → 
    (∀ (x y : ℝ), (x < 0) → (y < 0) → (|x| > |y|) → x < y) → 
  c = -1.5 := 
by
  intros a b c d Ha Hb Hc Hd Hpos Hneg HposNeg Habs
  sorry

end NUMINAMATH_GPT_smallest_number_among_l1211_121114


namespace NUMINAMATH_GPT_intersection_M_N_l1211_121155

def M := { x : ℝ | -1 < x ∧ x < 2 }
def N := { x : ℝ | x ≤ 1 }
def expectedIntersection := { x : ℝ | -1 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = expectedIntersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1211_121155


namespace NUMINAMATH_GPT_probability_same_gate_l1211_121160

open Finset

-- Definitions based on the conditions
def num_gates : ℕ := 3
def total_combinations : ℕ := num_gates * num_gates -- total number of combinations for both persons
def favorable_combinations : ℕ := num_gates         -- favorable combinations (both choose same gate)

-- Problem statement
theorem probability_same_gate : 
  ∃ (p : ℚ), p = (favorable_combinations : ℚ) / (total_combinations : ℚ) ∧ p = (1 / 3 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_probability_same_gate_l1211_121160


namespace NUMINAMATH_GPT_h_at_neg_eight_l1211_121103

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + x + 1

noncomputable def h (x : ℝ) (a b c : ℝ) : ℝ := (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_neg_eight (a b c : ℝ) (hf : f a = 0) (hf_b : f b = 0) (hf_c : f c = 0) :
  h (-8) a b c = -115 :=
  sorry

end NUMINAMATH_GPT_h_at_neg_eight_l1211_121103


namespace NUMINAMATH_GPT_line_through_A_with_zero_sum_of_intercepts_l1211_121184

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end NUMINAMATH_GPT_line_through_A_with_zero_sum_of_intercepts_l1211_121184


namespace NUMINAMATH_GPT_cells_count_at_day_8_l1211_121109

theorem cells_count_at_day_8 :
  let initial_cells := 3
  let common_ratio := 2
  let days := 8
  let interval := 2
  ∃ days_intervals, days_intervals = days / interval ∧ initial_cells * common_ratio ^ days_intervals = 48 :=
by
  sorry

end NUMINAMATH_GPT_cells_count_at_day_8_l1211_121109


namespace NUMINAMATH_GPT_total_cherry_tomatoes_l1211_121199

-- Definitions based on the conditions
def cherryTomatoesPerJar : Nat := 8
def numberOfJars : Nat := 7

-- The statement we want to prove
theorem total_cherry_tomatoes : cherryTomatoesPerJar * numberOfJars = 56 := by
  sorry

end NUMINAMATH_GPT_total_cherry_tomatoes_l1211_121199


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1211_121124

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition
    (h1 : a ≠ 0)
    (h2 : b ≠ 0) :
    (a^2 + b^2 ≥ 2 * a * b) → 
    (¬(a^2 + b^2 ≥ 2 * a * b) → ¬(a / b + b / a ≥ 2)) ∧ 
    ((a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2 * a * b)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1211_121124


namespace NUMINAMATH_GPT_John_ASMC_score_l1211_121158

def ASMC_score (c w : ℕ) : ℕ := 25 + 5 * c - 2 * w

theorem John_ASMC_score (c w : ℕ) (h1 : ASMC_score c w = 100) (h2 : c + w ≤ 25) :
  c = 19 ∧ w = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_John_ASMC_score_l1211_121158


namespace NUMINAMATH_GPT_g_2023_eq_0_l1211_121180

noncomputable def g (x : ℕ) : ℝ := sorry

axiom g_defined (x : ℕ) : ∃ y : ℝ, g x = y

axiom g_initial : g 1 = 1

axiom g_functional (a b : ℕ) : g (a + b) = g a + g b - 2 * g (a * b + 1)

theorem g_2023_eq_0 : g 2023 = 0 :=
sorry

end NUMINAMATH_GPT_g_2023_eq_0_l1211_121180


namespace NUMINAMATH_GPT_least_value_q_minus_p_l1211_121130

def p : ℝ := 2
def q : ℝ := 5

theorem least_value_q_minus_p (y : ℝ) (h : p < y ∧ y < q) : q - p = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_value_q_minus_p_l1211_121130


namespace NUMINAMATH_GPT_train_cross_pole_in_5_seconds_l1211_121120

/-- A train 100 meters long traveling at 72 kilometers per hour 
    will cross an electric pole in 5 seconds. -/
theorem train_cross_pole_in_5_seconds (L : ℝ) (v : ℝ) (t : ℝ) : 
  L = 100 → v = 72 * (1000 / 3600) → t = L / v → t = 5 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_pole_in_5_seconds_l1211_121120


namespace NUMINAMATH_GPT_monthly_growth_rate_optimal_selling_price_l1211_121141

-- Conditions
def april_sales : ℕ := 150
def june_sales : ℕ := 216
def cost_price_per_unit : ℕ := 30
def initial_selling_price : ℕ := 40
def initial_sales_vol : ℕ := 300
def sales_decrease_rate : ℕ := 10
def desired_profit : ℕ := 3960

-- Questions (Proof statements)
theorem monthly_growth_rate :
  ∃ (x : ℝ), (1 + x) ^ 2 = (june_sales:ℝ) / (april_sales:ℝ) ∧ x = 0.2 := by
  sorry

theorem optimal_selling_price :
  ∃ (y : ℝ), (y - cost_price_per_unit) * (initial_sales_vol - sales_decrease_rate * (y - initial_selling_price)) = desired_profit ∧ y = 48 := by
  sorry

end NUMINAMATH_GPT_monthly_growth_rate_optimal_selling_price_l1211_121141


namespace NUMINAMATH_GPT_range_of_a_l1211_121187

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1211_121187


namespace NUMINAMATH_GPT_find_k_and_a_l1211_121171

noncomputable def polynomial_P : Polynomial ℝ := Polynomial.C 5 + Polynomial.X * (Polynomial.C (-18) + Polynomial.X * (Polynomial.C 13 + Polynomial.X * (Polynomial.C (-4) + Polynomial.X)))
noncomputable def polynomial_D (k : ℝ) : Polynomial ℝ := Polynomial.C k + Polynomial.X * (Polynomial.C (-1) + Polynomial.X)
noncomputable def polynomial_R (a : ℝ) : Polynomial ℝ := Polynomial.C a + (Polynomial.C 2 * Polynomial.X)

theorem find_k_and_a : 
  ∃ k a : ℝ, polynomial_P = polynomial_D k * Polynomial.C 1 + polynomial_R a ∧ k = 10 ∧ a = 5 :=
sorry

end NUMINAMATH_GPT_find_k_and_a_l1211_121171


namespace NUMINAMATH_GPT_point_in_first_quadrant_l1211_121102

theorem point_in_first_quadrant (m : ℝ) (h : m < 0) : 
  (-m > 0) ∧ (-m + 1 > 0) :=
by 
  sorry

end NUMINAMATH_GPT_point_in_first_quadrant_l1211_121102


namespace NUMINAMATH_GPT_find_central_angle_l1211_121149

theorem find_central_angle
  (θ r : ℝ)
  (h1 : r * θ = 2 * π)
  (h2 : (1 / 2) * r^2 * θ = 3 * π) :
  θ = 2 * π / 3 := 
sorry

end NUMINAMATH_GPT_find_central_angle_l1211_121149


namespace NUMINAMATH_GPT_find_n_l1211_121177

theorem find_n (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1211_121177


namespace NUMINAMATH_GPT_expression_evaluation_l1211_121163

theorem expression_evaluation : (2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1) := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1211_121163


namespace NUMINAMATH_GPT_min_value_circles_tangents_l1211_121178

theorem min_value_circles_tangents (a b : ℝ) (h1 : (∃ x y : ℝ, x^2 + y^2 + 2 * a * x + a^2 - 4 = 0) ∧ 
  (∃ x y : ℝ, x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0))
  (h2 : ∃ k : ℕ, k = 3) (h3 : a ≠ 0) (h4 : b ≠ 0) : 
  (∃ m : ℝ, m = 1 ∧  ∀ x : ℝ, (x = (1 / a^2) + (1 / b^2)) → x ≥ m) :=
  sorry

end NUMINAMATH_GPT_min_value_circles_tangents_l1211_121178


namespace NUMINAMATH_GPT_tolu_pencils_l1211_121198

theorem tolu_pencils (price_per_pencil : ℝ) (robert_pencils : ℕ) (melissa_pencils : ℕ) (total_money_spent : ℝ) (tolu_pencils : ℕ) :
  price_per_pencil = 0.20 →
  robert_pencils = 5 →
  melissa_pencils = 2 →
  total_money_spent = 2.00 →
  tolu_pencils * price_per_pencil = 2.00 - (5 * 0.20 + 2 * 0.20) →
  tolu_pencils = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_tolu_pencils_l1211_121198


namespace NUMINAMATH_GPT_plane_equation_l1211_121162

-- Define the point and the normal vector
def point : ℝ × ℝ × ℝ := (8, -2, 2)
def normal_vector : ℝ × ℝ × ℝ := (8, -2, 2)

-- Define integers A, B, C, D such that the plane equation satisfies the conditions
def A : ℤ := 4
def B : ℤ := -1
def C : ℤ := 1
def D : ℤ := -18

-- Prove the equation of the plane
theorem plane_equation (x y z : ℝ) :
  A * x + B * y + C * z + D = 0 ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l1211_121162


namespace NUMINAMATH_GPT_amoeba_population_after_ten_days_l1211_121181

-- Definitions based on the conditions
def initial_population : ℕ := 3
def amoeba_growth (n : ℕ) : ℕ := initial_population * 2^n

-- Lean statement for the proof problem
theorem amoeba_population_after_ten_days : amoeba_growth 10 = 3072 :=
by 
  sorry

end NUMINAMATH_GPT_amoeba_population_after_ten_days_l1211_121181


namespace NUMINAMATH_GPT_box_filling_possibilities_l1211_121157

def possible_numbers : List ℕ := [2015, 2016, 2017, 2018, 2019]

def fill_the_boxes (D O G C W : ℕ) : Prop :=
  D + O + G = C + O + W

theorem box_filling_possibilities :
  (∃ D O G C W : ℕ, 
    D ∈ possible_numbers ∧
    O ∈ possible_numbers ∧
    G ∈ possible_numbers ∧
    C ∈ possible_numbers ∧
    W ∈ possible_numbers ∧
    D ≠ O ∧ D ≠ G ∧ D ≠ C ∧ D ≠ W ∧
    O ≠ G ∧ O ≠ C ∧ O ≠ W ∧
    G ≠ C ∧ G ≠ W ∧
    C ≠ W ∧
    fill_the_boxes D O G C W) → 
    ∃ ways : ℕ, ways = 24 :=
  sorry

end NUMINAMATH_GPT_box_filling_possibilities_l1211_121157


namespace NUMINAMATH_GPT_simplify_expression_l1211_121176

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = -1) :
  (2 * (x - 2 * y) * (2 * x + y) - (x + 2 * y)^2 + x * (8 * y - 3 * x)) / (6 * y) = 2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1211_121176


namespace NUMINAMATH_GPT_shifted_line_does_not_pass_through_third_quadrant_l1211_121196

-- The condition: The original line is y = -2x - 1
def original_line (x : ℝ) : ℝ := -2 * x - 1

-- The condition: The line is shifted 3 units to the right
def shifted_line (x : ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_through_third_quadrant :
  ¬(∃ (x y : ℝ), y = shifted_line x ∧ x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_GPT_shifted_line_does_not_pass_through_third_quadrant_l1211_121196


namespace NUMINAMATH_GPT_range_of_m_for_hyperbola_l1211_121147

theorem range_of_m_for_hyperbola (m : ℝ) :
  (∃ u v : ℝ, (∀ x y : ℝ, x^2/(m+2) + y^2/(m+1) = 1) → (m > -2) ∧ (m < -1)) := by
  sorry

end NUMINAMATH_GPT_range_of_m_for_hyperbola_l1211_121147


namespace NUMINAMATH_GPT_pow_two_sub_one_not_square_l1211_121135

theorem pow_two_sub_one_not_square (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end NUMINAMATH_GPT_pow_two_sub_one_not_square_l1211_121135


namespace NUMINAMATH_GPT_tangent_line_eq_max_min_values_l1211_121117

noncomputable def f (x : ℝ) : ℝ := (1 / (3:ℝ)) * x^3 - 4 * x + 4

theorem tangent_line_eq (x y : ℝ) : 
    y = f 1 → 
    y = -3 * (x - 1) + f 1 → 
    3 * x + y - 10 / 3 = 0 := 
sorry

theorem max_min_values (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3) : 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≤ 4) ∧ 
    (∀ x, (0 ≤ x ∧ x ≤ 3) → f x ≥ -4 / 3) := 
sorry

end NUMINAMATH_GPT_tangent_line_eq_max_min_values_l1211_121117


namespace NUMINAMATH_GPT_max_area_quadrilateral_sum_opposite_angles_l1211_121193

theorem max_area_quadrilateral (a b c d : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) :
  ∃ (area : ℝ), area = 12 :=
by {
  sorry
}

theorem sum_opposite_angles (a b c d : ℝ) (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 4) (h₄ : d = 4) 
  (h_area : ∃ (area : ℝ), area = 12) 
  (h_opposite1 : θ₁ + θ₃ = 180) (h_opposite2 : θ₂ + θ₄ = 180) :
  ∃ θ, θ = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_area_quadrilateral_sum_opposite_angles_l1211_121193


namespace NUMINAMATH_GPT_required_moles_h2so4_l1211_121119

-- Defining chemical equation conditions
def balanced_reaction (nacl h2so4 hcl nahso4 : ℕ) : Prop :=
  nacl = h2so4 ∧ hcl = nacl ∧ nahso4 = nacl

-- Theorem statement
theorem required_moles_h2so4 (nacl_needed moles_h2so4 : ℕ) (hcl_produced nahso4_produced : ℕ)
  (h : nacl_needed = 2 ∧ balanced_reaction nacl_needed moles_h2so4 hcl_produced nahso4_produced) :
  moles_h2so4 = 2 :=
  sorry

end NUMINAMATH_GPT_required_moles_h2so4_l1211_121119


namespace NUMINAMATH_GPT_count_solid_circles_among_first_2006_l1211_121107

-- Definition of the sequence sum for location calculation
def sequence_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2 - 1

-- Main theorem
theorem count_solid_circles_among_first_2006 : 
  ∃ n : ℕ, sequence_sum (n - 1) < 2006 ∧ 2006 ≤ sequence_sum n ∧ n = 62 :=
by {
  sorry
}

end NUMINAMATH_GPT_count_solid_circles_among_first_2006_l1211_121107


namespace NUMINAMATH_GPT_chimes_in_a_day_l1211_121165

-- Definitions for the conditions
def strikes_in_12_hours : ℕ :=
  (1 + 12) * 12 / 2

def strikes_in_24_hours : ℕ :=
  2 * strikes_in_12_hours

def half_hour_strikes : ℕ :=
  24 * 2

def total_chimes_in_a_day : ℕ :=
  strikes_in_24_hours + half_hour_strikes

-- Statement to prove
theorem chimes_in_a_day : total_chimes_in_a_day = 204 :=
by 
  -- The proof would be placed here
  sorry

end NUMINAMATH_GPT_chimes_in_a_day_l1211_121165


namespace NUMINAMATH_GPT_inequality_proof_l1211_121195

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (a + c)) * (1 + 4 * c / (a + b)) > 25 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1211_121195


namespace NUMINAMATH_GPT_average_weight_of_children_l1211_121138

theorem average_weight_of_children 
  (average_weight_boys : ℝ)
  (number_of_boys : ℕ)
  (average_weight_girls : ℝ)
  (number_of_girls : ℕ)
  (total_children : ℕ)
  (average_weight_children : ℝ) :
  average_weight_boys = 160 →
  number_of_boys = 8 →
  average_weight_girls = 130 →
  number_of_girls = 6 →
  total_children = number_of_boys + number_of_girls →
  average_weight_children = 
    (number_of_boys * average_weight_boys + number_of_girls * average_weight_girls) / total_children →
  average_weight_children = 147 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_average_weight_of_children_l1211_121138


namespace NUMINAMATH_GPT_students_with_average_age_of_16_l1211_121168

theorem students_with_average_age_of_16
  (N : ℕ) (A : ℕ) (N14 : ℕ) (A15 : ℕ) (N16 : ℕ)
  (h1 : N = 15) (h2 : A = 15) (h3 : N14 = 5) (h4 : A15 = 11) :
  N16 = 9 :=
sorry

end NUMINAMATH_GPT_students_with_average_age_of_16_l1211_121168


namespace NUMINAMATH_GPT_proportional_distribution_ratio_l1211_121185

theorem proportional_distribution_ratio (B : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : B = 80) 
  (h2 : S = 164)
  (h3 : S = (B / (1 - r)) + (B * (1 - r))) : 
  r = 0.2 := 
sorry

end NUMINAMATH_GPT_proportional_distribution_ratio_l1211_121185


namespace NUMINAMATH_GPT_gym_membership_cost_l1211_121161

theorem gym_membership_cost 
    (cheap_monthly_fee : ℕ := 10)
    (cheap_signup_fee : ℕ := 50)
    (expensive_monthly_multiplier : ℕ := 3)
    (months_in_year : ℕ := 12)
    (expensive_signup_multiplier : ℕ := 4) :
    let cheap_gym_cost := cheap_monthly_fee * months_in_year + cheap_signup_fee
    let expensive_monthly_fee := cheap_monthly_fee * expensive_monthly_multiplier
    let expensive_gym_cost := expensive_monthly_fee * months_in_year + expensive_monthly_fee * expensive_signup_multiplier
    let total_cost := cheap_gym_cost + expensive_gym_cost
    total_cost = 650 :=
by
  sorry -- Proof is omitted because the focus is on the statement equivalency.

end NUMINAMATH_GPT_gym_membership_cost_l1211_121161


namespace NUMINAMATH_GPT_find_a_l1211_121142

-- Define the constants b and the asymptote equation
def asymptote_eq (x y : ℝ) := 3 * x + 2 * y = 0

-- Define the hyperbola equation and the condition
def hyperbola_eq (x y a : ℝ) := x^2 / a^2 - y^2 / 9 = 1
def hyperbola_condition (a : ℝ) := a > 0

-- Theorem stating the value of a given the conditions
theorem find_a (a : ℝ) (hcond : hyperbola_condition a) 
  (h_asymp : ∀ x y : ℝ, asymptote_eq x y → y = -(3/2) * x) :
  a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_l1211_121142


namespace NUMINAMATH_GPT_fuel_tank_ethanol_l1211_121183

theorem fuel_tank_ethanol (x : ℝ) (H : 0.12 * x + 0.16 * (208 - x) = 30) : x = 82 := 
by
  sorry

end NUMINAMATH_GPT_fuel_tank_ethanol_l1211_121183


namespace NUMINAMATH_GPT_average_expression_l1211_121197

-- Define a theorem to verify the given problem
theorem average_expression (E a : ℤ) (h1 : a = 34) (h2 : (E + (3 * a - 8)) / 2 = 89) : E = 84 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_expression_l1211_121197


namespace NUMINAMATH_GPT_sequence_sum_l1211_121115

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l1211_121115


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_special_regular_polygon_l1211_121125

theorem sum_of_interior_angles_of_special_regular_polygon (n : ℕ) (h1 : n = 4 ∨ n = 5) :
  ((n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540) :=
by sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_special_regular_polygon_l1211_121125


namespace NUMINAMATH_GPT_value_of_expression_l1211_121132

theorem value_of_expression (x y z : ℕ) (h1 : x = 3) (h2 : y = 2) (h3 : z = 1) : 
  3 * x - 2 * y + 4 * z = 9 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1211_121132


namespace NUMINAMATH_GPT_arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l1211_121159

open Nat

axiom students : Fin 7 → Type -- Define students indexed by their position in the line.

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem arrangements_A_and_B_together :
  (2 * fact 6) = 1440 := 
by 
  sorry

theorem arrangements_A_not_head_B_not_tail :
  (fact 7 - 2 * fact 6 + fact 5) = 3720 := 
by 
  sorry

theorem arrangements_A_and_B_not_next :
  (3600) = 3600 := 
by 
  sorry

theorem arrangements_one_person_between_A_and_B :
  (fact 5 * 2) = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l1211_121159


namespace NUMINAMATH_GPT_factor_expression_l1211_121189

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1211_121189


namespace NUMINAMATH_GPT_max_value_of_f_l1211_121146

noncomputable def f (x : ℝ) : ℝ := x * (4 - x)

theorem max_value_of_f : ∃ y, ∀ x ∈ Set.Ioo 0 4, f x ≤ y ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1211_121146


namespace NUMINAMATH_GPT_sum_of_digits_eleven_l1211_121133

-- Definitions for the problem conditions
def distinct_digits (p q r : Nat) : Prop :=
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p > 0 ∧ q > 0 ∧ r > 0 ∧ p < 10 ∧ q < 10 ∧ r < 10

def is_two_digit_prime (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n.Prime

def concat_digits (x y : Nat) : Nat :=
  10 * x + y

def problem_conditions (p q r : Nat) : Prop :=
  distinct_digits p q r ∧
  is_two_digit_prime (concat_digits p q) ∧
  is_two_digit_prime (concat_digits p r) ∧
  is_two_digit_prime (concat_digits q r) ∧
  (concat_digits p q) * (concat_digits p r) = 221

-- Lean 4 statement to prove the sum of p, q, r is 11
theorem sum_of_digits_eleven (p q r : Nat) (h : problem_conditions p q r) : p + q + r = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_eleven_l1211_121133


namespace NUMINAMATH_GPT_andy_correct_answer_l1211_121154

-- Let y be the number Andy is using
def y : ℕ := 13  -- Derived from the conditions

-- Given condition based on Andy's incorrect operation
def condition : Prop := 4 * y + 5 = 57

-- Statement of the proof problem
theorem andy_correct_answer : condition → ((y + 5) * 4 = 72) := by
  intros h
  sorry

end NUMINAMATH_GPT_andy_correct_answer_l1211_121154


namespace NUMINAMATH_GPT_average_age_increase_l1211_121121

variable (A : ℝ) -- Original average age of 8 men
variable (age1 age2 : ℝ) -- The ages of the two men being replaced
variable (avg_women : ℝ) -- The average age of the two women

-- Conditions as hypotheses
def conditions : Prop :=
  8 * A - age1 - age2 + avg_women * 2 = 8 * (A + 2)

-- The theorem that needs to be proved
theorem average_age_increase (h1 : age1 = 20) (h2 : age2 = 28) (h3 : avg_women = 32) (h4 : conditions A age1 age2 avg_women) : (8 * A + 16) / 8 - A = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_age_increase_l1211_121121


namespace NUMINAMATH_GPT_minimum_value_sum_l1211_121105

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a / (2 * b)) + (b / (4 * c)) + (c / (8 * a))

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c >= 3/4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_sum_l1211_121105


namespace NUMINAMATH_GPT_simplify_expr_l1211_121173

noncomputable def expr1 : ℝ := 3 * Real.sqrt 8 / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7)
noncomputable def expr2 : ℝ := -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7)

theorem simplify_expr : expr1 = expr2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1211_121173


namespace NUMINAMATH_GPT_four_clique_exists_in_tournament_l1211_121122

open Finset

/-- Given a graph G with 9 vertices and 28 edges, prove that G contains a 4-clique. -/
theorem four_clique_exists_in_tournament 
  (V : Finset ℕ) (E : Finset (ℕ × ℕ)) 
  (hV : V.card = 9) 
  (hE : E.card = 28) :
  ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ (v₁ v₂ : ℕ), v₁ ∈ S → v₂ ∈ S → v₁ ≠ v₂ → (v₁, v₂) ∈ E ∨ (v₂, v₁) ∈ E :=
sorry

end NUMINAMATH_GPT_four_clique_exists_in_tournament_l1211_121122


namespace NUMINAMATH_GPT_Ben_hits_7_l1211_121174

def regions : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
def Alice_score : ℕ := 18
def Ben_score : ℕ := 13
def Cindy_score : ℕ := 19
def Dave_score : ℕ := 16
def Ellen_score : ℕ := 20
def Frank_score : ℕ := 5

def hit_score (name : String) (region1 region2 : ℕ) (score : ℕ) : Prop :=
  region1 ∈ regions ∧ region2 ∈ regions ∧ region1 ≠ region2 ∧ region1 + region2 = score

theorem Ben_hits_7 :
  ∃ r1 r2, hit_score "Ben" r1 r2 Ben_score ∧ (r1 = 7 ∨ r2 = 7) :=
sorry

end NUMINAMATH_GPT_Ben_hits_7_l1211_121174


namespace NUMINAMATH_GPT_angle_equivalence_l1211_121172

theorem angle_equivalence : (2023 % 360 = -137 % 360) := 
by 
  sorry

end NUMINAMATH_GPT_angle_equivalence_l1211_121172


namespace NUMINAMATH_GPT_three_x_plus_four_l1211_121139

theorem three_x_plus_four (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  sorry

end NUMINAMATH_GPT_three_x_plus_four_l1211_121139


namespace NUMINAMATH_GPT_ratio_d_e_l1211_121143

theorem ratio_d_e (a b c d e f : ℝ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  d / e = 1 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_d_e_l1211_121143


namespace NUMINAMATH_GPT_percentage_failing_both_l1211_121182

-- Define the conditions as constants
def percentage_failing_hindi : ℝ := 0.25
def percentage_failing_english : ℝ := 0.48
def percentage_passing_both : ℝ := 0.54

-- Define the percentage of students who failed in at least one subject
def percentage_failing_at_least_one : ℝ := 1 - percentage_passing_both

-- The main theorem statement we want to prove
theorem percentage_failing_both :
  percentage_failing_at_least_one = percentage_failing_hindi + percentage_failing_english - 0.27 := by
sorry

end NUMINAMATH_GPT_percentage_failing_both_l1211_121182


namespace NUMINAMATH_GPT_solve_for_x_l1211_121123

theorem solve_for_x (x : ℝ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -(2 / 11) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1211_121123


namespace NUMINAMATH_GPT_evaluate_expression_l1211_121191

lemma pow_mod_four_cycle (n : ℕ) : (n % 4) = 1 → (i : ℂ)^n = i :=
by sorry

lemma pow_mod_four_cycle2 (n : ℕ) : (n % 4) = 2 → (i : ℂ)^n = -1 :=
by sorry

lemma pow_mod_four_cycle3 (n : ℕ) : (n % 4) = 3 → (i : ℂ)^n = -i :=
by sorry

lemma pow_mod_four_cycle4 (n : ℕ) : (n % 4) = 0 → (i : ℂ)^n = 1 :=
by sorry

theorem evaluate_expression : 
  (i : ℂ)^(2021) + (i : ℂ)^(2022) + (i : ℂ)^(2023) + (i : ℂ)^(2024) = 0 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l1211_121191


namespace NUMINAMATH_GPT_division_of_repeating_decimals_l1211_121126

noncomputable def repeating_to_fraction (r : ℚ) : ℚ := 
  if r == 0.36 then 4 / 11 
  else if r == 0.12 then 4 / 33 
  else 0

theorem division_of_repeating_decimals :
  (repeating_to_fraction 0.36) / (repeating_to_fraction 0.12) = 3 :=
by
  sorry

end NUMINAMATH_GPT_division_of_repeating_decimals_l1211_121126


namespace NUMINAMATH_GPT_encyclopedia_total_pages_l1211_121188

noncomputable def totalPages : ℕ :=
450 + 3 * 90 +
650 + 5 * 68 +
712 + 4 * 75 +
820 + 6 * 120 +
530 + 2 * 110 +
900 + 7 * 95 +
680 + 4 * 80 +
555 + 3 * 180 +
990 + 5 * 53 +
825 + 6 * 150 +
410 + 2 * 200 +
1014 + 7 * 69

theorem encyclopedia_total_pages : totalPages = 13659 := by
  sorry

end NUMINAMATH_GPT_encyclopedia_total_pages_l1211_121188


namespace NUMINAMATH_GPT_solve_quadratic_l1211_121166

theorem solve_quadratic (x : ℝ) : x^2 = x ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1211_121166


namespace NUMINAMATH_GPT_find_base_solve_inequality_case1_solve_inequality_case2_l1211_121136

noncomputable def log_function (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : log_function a 8 = 3 → a = 2 :=
by sorry

theorem solve_inequality_case1 (a : ℝ) (h₁ : 1 < a) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 0 < x ∧ x ≤ 1 / 2 :=
by sorry

theorem solve_inequality_case2 (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) :
  ∀ x : ℝ, log_function a x ≤ log_function a (2 - 3 * x) → 1 / 2 ≤ x ∧ x < 2 / 3 :=
by sorry

end NUMINAMATH_GPT_find_base_solve_inequality_case1_solve_inequality_case2_l1211_121136


namespace NUMINAMATH_GPT_num_bad_oranges_l1211_121151

theorem num_bad_oranges (G B : ℕ) (hG : G = 24) (ratio : G / B = 3) : B = 8 :=
by
  sorry

end NUMINAMATH_GPT_num_bad_oranges_l1211_121151


namespace NUMINAMATH_GPT_angle_complement_30_l1211_121186

def complement_angle (x : ℝ) : ℝ := 90 - x

theorem angle_complement_30 (x : ℝ) (h : x = complement_angle x - 30) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_complement_30_l1211_121186


namespace NUMINAMATH_GPT_train_passes_pole_in_10_seconds_l1211_121175

theorem train_passes_pole_in_10_seconds :
  let L := 150 -- length of the train in meters
  let S_kmhr := 54 -- speed in kilometers per hour
  let S_ms := S_kmhr * 1000 / 3600 -- speed in meters per second
  (L / S_ms = 10) := 
by
  sorry

end NUMINAMATH_GPT_train_passes_pole_in_10_seconds_l1211_121175


namespace NUMINAMATH_GPT_translate_graph_cos_l1211_121111

/-- Let f(x) = cos(2x). 
    Translate f(x) to the left by π/6 units to get g(x), 
    then translate g(x) upwards by 1 unit to get h(x). 
    Prove that h(x) = cos(2x + π/3) + 1. -/
theorem translate_graph_cos :
  let f (x : ℝ) := Real.cos (2 * x)
  let g (x : ℝ) := f (x + Real.pi / 6)
  let h (x : ℝ) := g x + 1
  ∀ (x : ℝ), h x = Real.cos (2 * x + Real.pi / 3) + 1 :=
by
  sorry

end NUMINAMATH_GPT_translate_graph_cos_l1211_121111


namespace NUMINAMATH_GPT_number_of_girls_sampled_in_third_grade_l1211_121190

-- Number of total students in the high school
def total_students : ℕ := 3000

-- Number of students in each grade
def first_grade_students : ℕ := 800
def second_grade_students : ℕ := 1000
def third_grade_students : ℕ := 1200

-- Number of boys and girls in each grade
def first_grade_boys : ℕ := 500
def first_grade_girls : ℕ := 300

def second_grade_boys : ℕ := 600
def second_grade_girls : ℕ := 400

def third_grade_boys : ℕ := 800
def third_grade_girls : ℕ := 400

-- Total number of students sampled
def total_sampled_students : ℕ := 150

-- Hypothesis: stratified sampling method according to grade proportions
theorem number_of_girls_sampled_in_third_grade :
  third_grade_girls * (total_sampled_students / total_students) = 20 :=
by
  -- We will add the proof here
  sorry

end NUMINAMATH_GPT_number_of_girls_sampled_in_third_grade_l1211_121190


namespace NUMINAMATH_GPT_three_digit_number_addition_l1211_121106

theorem three_digit_number_addition (a b : ℕ) (ha : a < 10) (hb : b < 10) (h1 : 307 + 294 = 6 * 100 + b * 10 + 1)
  (h2 : (6 * 100 + b * 10 + 1) % 7 = 0) : a + b = 8 :=
by {
  sorry  -- Proof steps not needed
}

end NUMINAMATH_GPT_three_digit_number_addition_l1211_121106


namespace NUMINAMATH_GPT_tan_alpha_value_l1211_121100

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l1211_121100


namespace NUMINAMATH_GPT_intersection_correct_l1211_121156

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := { y | ∃ x ∈ A, y = 2 * x - 1 }

def intersection : Set ℕ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_correct : intersection = {1, 3} := by
  sorry

end NUMINAMATH_GPT_intersection_correct_l1211_121156


namespace NUMINAMATH_GPT_center_determines_position_l1211_121148

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for the Circle's position being determined by its center.
theorem center_determines_position (c : Circle) : c.center = c.center :=
by
  sorry

end NUMINAMATH_GPT_center_determines_position_l1211_121148


namespace NUMINAMATH_GPT_find_side_length_of_largest_square_l1211_121128

theorem find_side_length_of_largest_square (A : ℝ) (hA : A = 810) :
  ∃ a : ℝ, (5 / 8) * a ^ 2 = A ∧ a = 36 := by
  sorry

end NUMINAMATH_GPT_find_side_length_of_largest_square_l1211_121128


namespace NUMINAMATH_GPT_total_marbles_l1211_121140

-- Definitions based on the given conditions
def jars : ℕ := 16
def pots : ℕ := jars / 2
def marbles_in_jar : ℕ := 5
def marbles_in_pot : ℕ := 3 * marbles_in_jar

-- Main statement to be proved
theorem total_marbles : 
  5 * jars + marbles_in_pot * pots = 200 := 
by
  sorry

end NUMINAMATH_GPT_total_marbles_l1211_121140


namespace NUMINAMATH_GPT_fibers_below_20_count_l1211_121110

variable (fibers : List ℕ)

-- Conditions
def total_fibers := fibers.length = 100
def length_interval (f : ℕ) := 5 ≤ f ∧ f ≤ 40
def fibers_within_interval := ∀ f ∈ fibers, length_interval f

-- Question
def fibers_less_than_20 (fibers : List ℕ) : Nat :=
  (fibers.filter (λ f => f < 20)).length

theorem fibers_below_20_count (h_total : total_fibers fibers)
  (h_interval : fibers_within_interval fibers)
  (histogram_data : fibers_less_than_20 fibers = 30) :
  fibers_less_than_20 fibers = 30 :=
by
  sorry

end NUMINAMATH_GPT_fibers_below_20_count_l1211_121110


namespace NUMINAMATH_GPT_range_of_a_l1211_121116

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ {x | x^2 ≤ 1} ∪ {a} ↔ x ∈ {x | x^2 ≤ 1}) → (-1 ≤ a ∧ a ≤ 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1211_121116
