import Mathlib

namespace NUMINAMATH_GPT_find_number_l121_12197

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l121_12197


namespace NUMINAMATH_GPT_geometric_progression_product_l121_12171

variables {n : ℕ} {b q S S' P : ℝ} 

theorem geometric_progression_product (hb : b ≠ 0) (hq : q ≠ 1)
  (hP : P = b^n * q^(n*(n-1)/2))
  (hS : S = b * (1 - q^n) / (1 - q))
  (hS' : S' = (q^n - 1) / (b * (q - 1)))
  : P = (S * S')^(n/2) := 
sorry

end NUMINAMATH_GPT_geometric_progression_product_l121_12171


namespace NUMINAMATH_GPT_compare_logs_l121_12130

noncomputable def a := Real.log 3
noncomputable def b := Real.log 3 / Real.log 2 / 2
noncomputable def c := Real.log 2 / Real.log 3 / 2

theorem compare_logs : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_logs_l121_12130


namespace NUMINAMATH_GPT_slices_per_birthday_l121_12159

-- Define the conditions: 
-- k is the age, the number of candles, starting from 3.
variable (k : ℕ) (h : k ≥ 3)

-- Define the function for the number of triangular slices
def number_of_slices (k : ℕ) : ℕ := 2 * k - 5

-- State the theorem to prove that the number of slices is 2k - 5
theorem slices_per_birthday (k : ℕ) (h : k ≥ 3) : 
    number_of_slices k = 2 * k - 5 := 
by
  sorry

end NUMINAMATH_GPT_slices_per_birthday_l121_12159


namespace NUMINAMATH_GPT_find_a5_and_sum_l121_12169

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) > a n

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
is_geometric_sequence a q ∧ is_increasing_sequence a ∧ a 2 = 3 ∧ a 4 - a 3 = 18

-- Theorem to prove
theorem find_a5_and_sum {a : ℕ → ℝ} {q : ℝ} (h : given_conditions a q) :
  a 5 = 81 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_a5_and_sum_l121_12169


namespace NUMINAMATH_GPT_profit_calculation_l121_12173

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end NUMINAMATH_GPT_profit_calculation_l121_12173


namespace NUMINAMATH_GPT_unique_subset_empty_set_l121_12175

def discriminant (a : ℝ) : ℝ := 4 - 4 * a^2

theorem unique_subset_empty_set (a : ℝ) :
  (∀ (x : ℝ), ¬(a * x^2 + 2 * x + a = 0)) ↔ (a > 1 ∨ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_unique_subset_empty_set_l121_12175


namespace NUMINAMATH_GPT_find_t_l121_12138

-- Define the utility on both days
def utility_monday (t : ℝ) := t * (10 - t)
def utility_tuesday (t : ℝ) := (4 - t) * (t + 5)

-- Define the total hours spent on activities condition for both days
def total_hours_monday (t : ℝ) := t + (10 - t)
def total_hours_tuesday (t : ℝ) := (4 - t) + (t + 5)

theorem find_t : ∃ t : ℝ, t * (10 - t) = (4 - t) * (t + 5) ∧ 
                            total_hours_monday t ≥ 8 ∧ 
                            total_hours_tuesday t ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l121_12138


namespace NUMINAMATH_GPT_solution_set_of_inequality_l121_12141

theorem solution_set_of_inequality : 
  { x : ℝ | x^2 - 3*x - 4 < 0 } = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l121_12141


namespace NUMINAMATH_GPT_work_earnings_t_l121_12115

theorem work_earnings_t (t : ℤ) (h1 : (t + 2) * (4 * t - 4) = (4 * t - 7) * (t + 3) + 3) : t = 10 := 
by
  sorry

end NUMINAMATH_GPT_work_earnings_t_l121_12115


namespace NUMINAMATH_GPT_jonah_raisins_l121_12182

variable (y : ℝ)

theorem jonah_raisins :
  (y + 0.4 = 0.7) → (y = 0.3) :=
  by
  intro h
  sorry

end NUMINAMATH_GPT_jonah_raisins_l121_12182


namespace NUMINAMATH_GPT_women_decreased_by_3_l121_12142

noncomputable def initial_men := 12
noncomputable def initial_women := 27

theorem women_decreased_by_3 
  (ratio_men_women : 4 / 5 = initial_men / initial_women)
  (men_after_enter : initial_men + 2 = 14)
  (women_after_leave : initial_women - 3 = 24) :
  (24 - 27 = -3) :=
by
  sorry

end NUMINAMATH_GPT_women_decreased_by_3_l121_12142


namespace NUMINAMATH_GPT_Alan_age_is_29_l121_12188

/-- Alan and Chris ages problem -/
theorem Alan_age_is_29
    (A C : ℕ)
    (h1 : A + C = 52)
    (h2 : C = A / 3 + 2 * (A - C)) :
    A = 29 :=
by
  sorry

end NUMINAMATH_GPT_Alan_age_is_29_l121_12188


namespace NUMINAMATH_GPT_solve_x_l121_12156

def δ (x : ℝ) : ℝ := 4 * x + 6
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_x : ∃ x: ℝ, δ (φ x) = 3 → x = -19 / 20 := by
  sorry

end NUMINAMATH_GPT_solve_x_l121_12156


namespace NUMINAMATH_GPT_no_prime_satisfies_condition_l121_12176

theorem no_prime_satisfies_condition :
  ¬ ∃ p : ℕ, p > 1 ∧ 10 * (p : ℝ) = (p : ℝ) + 5.4 := by {
  sorry
}

end NUMINAMATH_GPT_no_prime_satisfies_condition_l121_12176


namespace NUMINAMATH_GPT_simplify_expr_l121_12126

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end NUMINAMATH_GPT_simplify_expr_l121_12126


namespace NUMINAMATH_GPT_inequality_proof_l121_12168

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l121_12168


namespace NUMINAMATH_GPT_find_x_value_l121_12174

noncomputable def log (a b: ℝ): ℝ := Real.log a / Real.log b

theorem find_x_value (a n : ℝ) (t y: ℝ):
  1 < a →
  1 < t →
  y = 8 →
  log n (a^t) - 3 * log a (a^t) * log y 8 = 3 →
  x = a^t →
  x = a^2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l121_12174


namespace NUMINAMATH_GPT_find_z_given_conditions_l121_12163

variable (x y z : ℤ)

theorem find_z_given_conditions :
  (x + y) / 2 = 4 →
  x + y + z = 0 →
  z = -8 := by
  sorry

end NUMINAMATH_GPT_find_z_given_conditions_l121_12163


namespace NUMINAMATH_GPT_ab2c_value_l121_12189

theorem ab2c_value (a b c : ℚ) (h₁ : |a + 1| + (b - 2)^2 = 0) (h₂ : |c| = 3) :
  a + b + 2 * c = 7 ∨ a + b + 2 * c = -5 := sorry

end NUMINAMATH_GPT_ab2c_value_l121_12189


namespace NUMINAMATH_GPT_geometric_series_sum_l121_12121

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l121_12121


namespace NUMINAMATH_GPT_avg_of_9_numbers_l121_12143

theorem avg_of_9_numbers (a b c d e f g h i : ℕ)
  (h1 : (a + b + c + d + e) / 5 = 99)
  (h2 : (e + f + g + h + i) / 5 = 100)
  (h3 : e = 59) : 
  (a + b + c + d + e + f + g + h + i) / 9 = 104 := 
sorry

end NUMINAMATH_GPT_avg_of_9_numbers_l121_12143


namespace NUMINAMATH_GPT_rebecca_groups_eq_l121_12129

-- Definitions
def total_eggs : ℕ := 15
def eggs_per_group : ℕ := 5
def expected_groups : ℕ := 3

-- Theorem to prove
theorem rebecca_groups_eq :
  total_eggs / eggs_per_group = expected_groups :=
by
  sorry

end NUMINAMATH_GPT_rebecca_groups_eq_l121_12129


namespace NUMINAMATH_GPT_factorization_correct_l121_12136

-- Define the given expression
def expression (a b : ℝ) : ℝ := 9 * a^2 * b - b

-- Define the factorized form
def factorized_form (a b : ℝ) : ℝ := b * (3 * a + 1) * (3 * a - 1)

-- Theorem stating that the factorization is correct
theorem factorization_correct (a b : ℝ) : expression a b = factorized_form a b := by
  sorry

end NUMINAMATH_GPT_factorization_correct_l121_12136


namespace NUMINAMATH_GPT_least_possible_sum_l121_12101

theorem least_possible_sum (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 21 * (q + 1)) : p + q = 5 :=
sorry

end NUMINAMATH_GPT_least_possible_sum_l121_12101


namespace NUMINAMATH_GPT_farthest_vertex_coordinates_l121_12140

noncomputable def image_vertex_coordinates_farthest_from_origin 
    (center_EFGH : ℝ × ℝ) (area_EFGH : ℝ) (dilation_center : ℝ × ℝ) 
    (scale_factor : ℝ) : ℝ × ℝ := sorry

theorem farthest_vertex_coordinates 
    (center_EFGH : ℝ × ℝ := (10, -6)) (area_EFGH : ℝ := 16) 
    (dilation_center : ℝ × ℝ := (2, 2)) (scale_factor : ℝ := 3) : 
    image_vertex_coordinates_farthest_from_origin center_EFGH area_EFGH dilation_center scale_factor = (32, -28) := 
sorry

end NUMINAMATH_GPT_farthest_vertex_coordinates_l121_12140


namespace NUMINAMATH_GPT_algebraic_identity_l121_12172

theorem algebraic_identity (x y : ℝ) (h₁ : x * y = 4) (h₂ : x - y = 5) : 
  x^2 + 5 * x * y + y^2 = 53 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_identity_l121_12172


namespace NUMINAMATH_GPT_polar_bear_daily_food_l121_12185

-- Definitions based on the conditions
def bucketOfTroutDaily : ℝ := 0.2
def bucketOfSalmonDaily : ℝ := 0.4

-- The proof statement
theorem polar_bear_daily_food : bucketOfTroutDaily + bucketOfSalmonDaily = 0.6 := by
  sorry

end NUMINAMATH_GPT_polar_bear_daily_food_l121_12185


namespace NUMINAMATH_GPT_mrs_jane_total_coins_l121_12184

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end NUMINAMATH_GPT_mrs_jane_total_coins_l121_12184


namespace NUMINAMATH_GPT_find_x_l121_12113

variables {K J : ℝ} {A B C A_star B_star C_star : Type*}

-- Define the triangles and areas
def triangle_area (K : ℝ) : Prop := K > 0

-- We know the fractions of segments in triangle
def segment_ratios (x : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧
  ∀ (AA_star AB BB_star BC CC_star CA : ℝ),
    AA_star / AB = x ∧ BB_star / BC = x ∧ CC_star / CA = x

-- Area of the smaller inner triangle
def inner_triangle_area (x : ℝ) (K : ℝ) (J : ℝ) : Prop :=
  J = x * K

-- The theorem combining all to show x = 1/3
theorem find_x (x : ℝ) (K J : ℝ) (triangleAreaK : triangle_area K)
    (ratios : segment_ratios x)
    (innerArea : inner_triangle_area x K J) :
  x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l121_12113


namespace NUMINAMATH_GPT_sequence_contains_at_most_one_square_l121_12180

theorem sequence_contains_at_most_one_square 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∀ m n, (m ≠ n) → ¬ (∃ k, a m = k^2 ∧ a n = k^2) :=
sorry

end NUMINAMATH_GPT_sequence_contains_at_most_one_square_l121_12180


namespace NUMINAMATH_GPT_functional_expression_selling_price_for_profit_l121_12112

-- Define the initial conditions
def cost_price : ℚ := 8
def initial_selling_price : ℚ := 10
def initial_sales_volume : ℚ := 200
def sales_decrement_per_yuan_increase : ℚ := 20

-- Functional expression between y (items) and x (yuan)
theorem functional_expression (x : ℚ) : 
  (200 - 20 * (x - 10) = -20 * x + 400) :=
sorry

-- Determine the selling price to achieve a daily profit of 640 yuan
theorem selling_price_for_profit (x : ℚ) (h1 : 8 ≤ x) (h2 : x ≤ 15) : 
  ((x - 8) * (400 - 20 * x) = 640) → (x = 12) :=
sorry

end NUMINAMATH_GPT_functional_expression_selling_price_for_profit_l121_12112


namespace NUMINAMATH_GPT_cost_per_mile_l121_12181

variable (x : ℝ)
variable (monday_miles : ℝ) (thursday_miles : ℝ) (base_cost : ℝ) (total_spent : ℝ)

-- Given conditions
def car_rental_conditions : Prop :=
  monday_miles = 620 ∧
  thursday_miles = 744 ∧
  base_cost = 150 ∧
  total_spent = 832 ∧
  total_spent = base_cost + (monday_miles + thursday_miles) * x

-- Theorem to prove the cost per mile
theorem cost_per_mile (h : car_rental_conditions x 620 744 150 832) : x = 0.50 :=
  by
    sorry

end NUMINAMATH_GPT_cost_per_mile_l121_12181


namespace NUMINAMATH_GPT_polynomial_expansion_l121_12199

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l121_12199


namespace NUMINAMATH_GPT_jimmy_points_l121_12152

theorem jimmy_points (eng_pts init_eng_pts : ℕ) (math_pts init_math_pts : ℕ) 
  (sci_pts init_sci_pts : ℕ) (hist_pts init_hist_pts : ℕ) 
  (phy_pts init_phy_pts : ℕ) (eng_penalty math_penalty sci_penalty hist_penalty phy_penalty : ℕ)
  (passing_points : ℕ) (total_points_required : ℕ):
  init_eng_pts = 60 →
  init_math_pts = 55 →
  init_sci_pts = 40 →
  init_hist_pts = 70 →
  init_phy_pts = 50 →
  eng_penalty = 5 →
  math_penalty = 3 →
  sci_penalty = 8 →
  hist_penalty = 2 →
  phy_penalty = 6 →
  passing_points = 250 →
  total_points_required = (init_eng_pts - eng_penalty) + (init_math_pts - math_penalty) + 
                         (init_sci_pts - sci_penalty) + (init_hist_pts - hist_penalty) + 
                         (init_phy_pts - phy_penalty) →
  ∀ extra_loss, (total_points_required - extra_loss ≥ passing_points) → extra_loss ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_jimmy_points_l121_12152


namespace NUMINAMATH_GPT_line_through_M_intersects_lines_l121_12148

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def line1 (t : ℝ) : Point3D :=
  {x := 2 - t, y := 3, z := -2 + t}

def plane1 (p : Point3D) : Prop :=
  2 * p.x - 2 * p.y - p.z - 4 = 0

def plane2 (p : Point3D) : Prop :=
  p.x + 3 * p.y + 2 * p.z + 1 = 0

def param_eq (t : ℝ) : Point3D :=
  {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t}

theorem line_through_M_intersects_lines : 
  ∀ (t : ℝ), plane1 (param_eq t) ∧ plane2 (param_eq t) -> 
  ∃ t, param_eq t = {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t} :=
by
  intros t h
  sorry

end NUMINAMATH_GPT_line_through_M_intersects_lines_l121_12148


namespace NUMINAMATH_GPT_most_stable_machine_l121_12124

noncomputable def var_A : ℝ := 10.3
noncomputable def var_B : ℝ := 6.9
noncomputable def var_C : ℝ := 3.5

theorem most_stable_machine :
  (var_C < var_B) ∧ (var_C < var_A) :=
by
  sorry

end NUMINAMATH_GPT_most_stable_machine_l121_12124


namespace NUMINAMATH_GPT_skill_position_players_waiting_l121_12164

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end NUMINAMATH_GPT_skill_position_players_waiting_l121_12164


namespace NUMINAMATH_GPT_triangle_circle_fill_l121_12117

theorem triangle_circle_fill (A B C D : ℕ) : 
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9) →
  (B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9) →
  (C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9) →
  (D = 6 ∨ D = 7 ∨ D = 8 ∨ D = 9) →
  (A + B + 1 + 8 =  A + 4 + 3 + 7) →  (D + 4 + 2 + 5 = 5 + 1 + 8 + B) →
  (5 + 1 + 8 + 6 = 5 + C + 7 + 4 ) →
  (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) := by
  sorry

end NUMINAMATH_GPT_triangle_circle_fill_l121_12117


namespace NUMINAMATH_GPT_total_floor_area_covered_l121_12104

theorem total_floor_area_covered (combined_area : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) : 
  combined_area = 200 → 
  area_two_layers = 22 → 
  area_three_layers = 19 → 
  (combined_area - (area_two_layers + 2 * area_three_layers)) = 140 := 
by
  sorry

end NUMINAMATH_GPT_total_floor_area_covered_l121_12104


namespace NUMINAMATH_GPT_Anne_wander_time_l121_12119

theorem Anne_wander_time (distance speed : ℝ) (h1 : distance = 3.0) (h2 : speed = 2.0) : distance / speed = 1.5 := by
  -- Given conditions
  sorry

end NUMINAMATH_GPT_Anne_wander_time_l121_12119


namespace NUMINAMATH_GPT_angle_of_inclination_l121_12146

theorem angle_of_inclination (t : ℝ) (x y : ℝ) :
  (x = 1 + t * (Real.sin (Real.pi / 6))) ∧ 
  (y = 2 + t * (Real.cos (Real.pi / 6))) →
  ∃ α : ℝ, α = Real.arctan (Real.sqrt 3) ∧ (0 ≤ α ∧ α < Real.pi) := 
by 
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l121_12146


namespace NUMINAMATH_GPT_total_overtime_hours_worked_l121_12145

def gary_wage : ℕ := 12
def mary_wage : ℕ := 14
def john_wage : ℕ := 16
def alice_wage : ℕ := 18
def michael_wage : ℕ := 20

def regular_hours : ℕ := 40
def overtime_rate : ℚ := 1.5

def total_paycheck : ℚ := 3646

theorem total_overtime_hours_worked :
  let gary_overtime := gary_wage * overtime_rate
  let mary_overtime := mary_wage * overtime_rate
  let john_overtime := john_wage * overtime_rate
  let alice_overtime := alice_wage * overtime_rate
  let michael_overtime := michael_wage * overtime_rate
  let regular_total := (gary_wage + mary_wage + john_wage + alice_wage + michael_wage) * regular_hours
  let total_overtime_pay := total_paycheck - regular_total
  let total_overtime_rate := gary_overtime + mary_overtime + john_overtime + alice_overtime + michael_overtime
  let overtime_hours := total_overtime_pay / total_overtime_rate
  overtime_hours.floor = 3 := 
by
  sorry

end NUMINAMATH_GPT_total_overtime_hours_worked_l121_12145


namespace NUMINAMATH_GPT_increment_in_displacement_l121_12109

variable (d : ℝ)

def equation_of_motion (t : ℝ) : ℝ := 2 * t^2

theorem increment_in_displacement:
  let t1 := 2
  let t2 := 2 + d
  let s1 := equation_of_motion t1
  let s2 := equation_of_motion t2
  s2 - s1 = 8 * d + 2 * d^2 := by
  sorry

end NUMINAMATH_GPT_increment_in_displacement_l121_12109


namespace NUMINAMATH_GPT_problem_statement_l121_12166

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, f x = 4 * x + 3) (h₁ : a > 0) (h₂ : b > 0) :
  (∀ x, |f x + 5| < a ↔ |x + 3| < b) ↔ b ≤ a / 4 :=
sorry

end NUMINAMATH_GPT_problem_statement_l121_12166


namespace NUMINAMATH_GPT_find_x_if_delta_phi_eq_3_l121_12195

variable (x : ℚ)

def delta (x : ℚ) := 4 * x + 9
def phi (x : ℚ) := 9 * x + 6

theorem find_x_if_delta_phi_eq_3 : 
  delta (phi x) = 3 → x = -5 / 6 := by 
  sorry

end NUMINAMATH_GPT_find_x_if_delta_phi_eq_3_l121_12195


namespace NUMINAMATH_GPT_find_number_l121_12131

theorem find_number (x : ℕ) (h : 15 * x = x + 196) : 15 * x = 210 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l121_12131


namespace NUMINAMATH_GPT_evaluate_f_at_neg_three_l121_12149

def f (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_f_at_neg_three : f (-3) = -14 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_neg_three_l121_12149


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l121_12186

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (h_sum : A + B + C = 180) :
  false :=
by
  -- Here "A > 60, B > 60, C > 60 and A + B + C = 180" leads to a contradiction
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l121_12186


namespace NUMINAMATH_GPT_aaronFoundCards_l121_12179

-- Given conditions
def initialCardsAaron : ℕ := 5
def finalCardsAaron : ℕ := 67

-- Theorem statement
theorem aaronFoundCards : finalCardsAaron - initialCardsAaron = 62 :=
by
  sorry

end NUMINAMATH_GPT_aaronFoundCards_l121_12179


namespace NUMINAMATH_GPT_eggs_not_eaten_is_6_l121_12178

noncomputable def eggs_not_eaten_each_week 
  (trays_purchased : ℕ) 
  (eggs_per_tray : ℕ) 
  (eggs_morning : ℕ) 
  (days_in_week : ℕ) 
  (eggs_night : ℕ) : ℕ :=
  let total_eggs := trays_purchased * eggs_per_tray
  let eggs_eaten_son_daughter := eggs_morning * days_in_week
  let eggs_eaten_rhea_husband := eggs_night * days_in_week
  let eggs_eaten_total := eggs_eaten_son_daughter + eggs_eaten_rhea_husband
  total_eggs - eggs_eaten_total

theorem eggs_not_eaten_is_6 
  (trays_purchased : ℕ := 2) 
  (eggs_per_tray : ℕ := 24) 
  (eggs_morning : ℕ := 2) 
  (days_in_week : ℕ := 7) 
  (eggs_night : ℕ := 4) : 
  eggs_not_eaten_each_week trays_purchased eggs_per_tray eggs_morning days_in_week eggs_night = 6 :=
by
  -- Here should be proof steps, but we use sorry to skip it as per instruction
  sorry

end NUMINAMATH_GPT_eggs_not_eaten_is_6_l121_12178


namespace NUMINAMATH_GPT_dot_but_not_straight_line_l121_12160

theorem dot_but_not_straight_line :
  let total := 80
  let D_n_S := 28
  let S_n_D := 47
  ∃ (D : ℕ), D - D_n_S = 5 ∧ D + S_n_D = total :=
by
  sorry

end NUMINAMATH_GPT_dot_but_not_straight_line_l121_12160


namespace NUMINAMATH_GPT_boris_neighbors_l121_12151

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end NUMINAMATH_GPT_boris_neighbors_l121_12151


namespace NUMINAMATH_GPT_math_problem_l121_12144

theorem math_problem (d r : ℕ) (hd : d > 1)
  (h1 : 1259 % d = r) 
  (h2 : 1567 % d = r) 
  (h3 : 2257 % d = r) : d - r = 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l121_12144


namespace NUMINAMATH_GPT_fraction_mistake_l121_12139

theorem fraction_mistake (n : ℕ) (h : n = 288) (student_answer : ℕ) 
(h_student : student_answer = 240) : student_answer / n = 5 / 6 := 
by 
  -- Given that n = 288 and the student's answer = 240;
  -- we need to prove that 240/288 = 5/6
  sorry

end NUMINAMATH_GPT_fraction_mistake_l121_12139


namespace NUMINAMATH_GPT_materials_total_order_l121_12102

theorem materials_total_order :
  let concrete := 0.16666666666666666
  let bricks := 0.16666666666666666
  let stone := 0.5
  concrete + bricks + stone = 0.8333333333333332 :=
by
  sorry

end NUMINAMATH_GPT_materials_total_order_l121_12102


namespace NUMINAMATH_GPT_valid_number_of_m_values_l121_12110

theorem valid_number_of_m_values : 
  (∃ m : ℕ, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m)) ∧ ∀ m, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m) → m > 1  → 
  ∃ n : ℕ, n = 22 :=
by
  sorry

end NUMINAMATH_GPT_valid_number_of_m_values_l121_12110


namespace NUMINAMATH_GPT_avg_time_stopped_per_hour_l121_12193

-- Definitions and conditions
def avgSpeedInMotion : ℝ := 75
def overallAvgSpeed : ℝ := 40

-- Statement to prove
theorem avg_time_stopped_per_hour :
  (1 - overallAvgSpeed / avgSpeedInMotion) * 60 = 28 := 
by
  sorry

end NUMINAMATH_GPT_avg_time_stopped_per_hour_l121_12193


namespace NUMINAMATH_GPT_largest_n_multiple_of_7_l121_12150

theorem largest_n_multiple_of_7 (n : ℕ) (h1 : n < 50000) (h2 : (5*(n-3)^5 - 3*n^2 + 20*n - 35) % 7 = 0) : n = 49999 :=
sorry

end NUMINAMATH_GPT_largest_n_multiple_of_7_l121_12150


namespace NUMINAMATH_GPT_total_number_of_glasses_l121_12157

open scoped Nat

theorem total_number_of_glasses (x y : ℕ) (h1 : y = x + 16) (h2 : (12 * x + 16 * y) / (x + y) = 15) : 12 * x + 16 * y = 480 := by
  sorry

end NUMINAMATH_GPT_total_number_of_glasses_l121_12157


namespace NUMINAMATH_GPT_joe_bought_books_l121_12165

theorem joe_bought_books (money_given : ℕ) (notebook_cost : ℕ) (num_notebooks : ℕ) (book_cost : ℕ) (leftover_money : ℕ) (total_spent := money_given - leftover_money) (spent_on_notebooks := num_notebooks * notebook_cost) (spent_on_books := total_spent - spent_on_notebooks) (num_books := spent_on_books / book_cost) : money_given = 56 → notebook_cost = 4 → num_notebooks = 7 → book_cost = 7 → leftover_money = 14 → num_books = 2 := by
  intros
  sorry

end NUMINAMATH_GPT_joe_bought_books_l121_12165


namespace NUMINAMATH_GPT_pell_solution_unique_l121_12116

theorem pell_solution_unique 
  (x_0 y_0 x y : ℤ) 
  (h_fundamental : x_0^2 - 2003 * y_0^2 = 1)
  (h_pos_x : 0 < x) 
  (h_pos_y : 0 < y)
  (h_prime_div : ∀ p, Prime p → p ∣ x → p ∣ x_0) :
  x^2 - 2003 * y^2 = 1 → (x, y) = (x_0, y_0) :=
sorry

end NUMINAMATH_GPT_pell_solution_unique_l121_12116


namespace NUMINAMATH_GPT_arrange_books_l121_12192

-- Given conditions
def math_books_count := 4
def history_books_count := 6

-- Question: How many ways can the books be arranged given the conditions?
theorem arrange_books (math_books_count history_books_count : ℕ) :
  math_books_count = 4 → 
  history_books_count = 6 →
  ∃ ways : ℕ, ways = 51840 :=
by
  sorry

end NUMINAMATH_GPT_arrange_books_l121_12192


namespace NUMINAMATH_GPT_remainder_of_division_l121_12100

theorem remainder_of_division :
  Nat.mod 4536 32 = 24 :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l121_12100


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l121_12191

noncomputable def S (n : ℕ) (a_1 : ℚ) (d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_ratio (a_1 d : ℚ) (h : d ≠ 0) (h_ratio : (a_1 + 5 * d) / (a_1 + 2 * d) = 2) :
  S 6 a_1 d / S 3 a_1 d = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_ratio_l121_12191


namespace NUMINAMATH_GPT_minimum_value_l121_12162

variable (m n x y : ℝ)

theorem minimum_value (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (my_nx : ℝ), my_nx = my + nx → my_nx ≥ min_val :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l121_12162


namespace NUMINAMATH_GPT_pure_gold_to_add_eq_46_67_l121_12123

-- Define the given conditions
variable (initial_alloy_weight : ℝ) (initial_gold_percentage : ℝ) (final_gold_percentage : ℝ)
variable (added_pure_gold : ℝ)

-- State the proof problem
theorem pure_gold_to_add_eq_46_67 :
  initial_alloy_weight = 20 ∧
  initial_gold_percentage = 0.50 ∧
  final_gold_percentage = 0.85 ∧
  (10 + added_pure_gold) / (20 + added_pure_gold) = 0.85 →
  added_pure_gold = 46.67 :=
by
  sorry

end NUMINAMATH_GPT_pure_gold_to_add_eq_46_67_l121_12123


namespace NUMINAMATH_GPT_find_k_l121_12120

-- Given: The polynomial x^2 - 3k * x * y - 3y^2 + 6 * x * y - 8
-- We want to prove the value of k such that the polynomial does not contain the term "xy".

theorem find_k (k : ℝ) : 
  (∀ x y : ℝ, (x^2 - 3 * k * x * y - 3 * y^2 + 6 * x * y - 8) = x^2 - 3 * y^2 - 8) → 
  k = 2 := 
by
  intro h
  have h_coeff := h 1 1
  -- We should observe that the polynomial should not contain the xy term
  sorry

end NUMINAMATH_GPT_find_k_l121_12120


namespace NUMINAMATH_GPT_train_length_is_200_l121_12196

noncomputable def train_length 
  (speed_kmh : ℕ) 
  (time_s: ℕ) : ℕ := 
  ((speed_kmh * 1000) / 3600) * time_s

theorem train_length_is_200
  (h_speed : 40 = 40)
  (h_time : 18 = 18) :
  train_length 40 18 = 200 :=
sorry

end NUMINAMATH_GPT_train_length_is_200_l121_12196


namespace NUMINAMATH_GPT_exists_func_satisfies_condition_l121_12170

theorem exists_func_satisfies_condition :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = abs (x + 1) :=
sorry

end NUMINAMATH_GPT_exists_func_satisfies_condition_l121_12170


namespace NUMINAMATH_GPT_inequality_must_hold_l121_12194

section
variables {a b c : ℝ}

theorem inequality_must_hold (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry
end

end NUMINAMATH_GPT_inequality_must_hold_l121_12194


namespace NUMINAMATH_GPT_Buffy_whiskers_l121_12153

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end NUMINAMATH_GPT_Buffy_whiskers_l121_12153


namespace NUMINAMATH_GPT_time_to_coffee_shop_is_18_l121_12105

variable (cycle_constant_pace : Prop)
variable (time_cycle_library : ℕ)
variable (distance_cycle_library : ℕ)
variable (distance_to_coffee_shop : ℕ)

theorem time_to_coffee_shop_is_18
  (h_const_pace : cycle_constant_pace)
  (h_time_library : time_cycle_library = 30)
  (h_distance_library : distance_cycle_library = 5)
  (h_distance_coffee : distance_to_coffee_shop = 3)
  : (30 / 5) * 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_time_to_coffee_shop_is_18_l121_12105


namespace NUMINAMATH_GPT_vasya_max_points_l121_12134

theorem vasya_max_points (cards : Finset (Fin 36)) 
  (petya_hand vasya_hand : Finset (Fin 36)) 
  (h_disjoint : Disjoint petya_hand vasya_hand)
  (h_union : petya_hand ∪ vasya_hand = cards)
  (h_card : cards.card = 36)
  (h_half : petya_hand.card = 18 ∧ vasya_hand.card = 18) : 
  ∃ max_points : ℕ, max_points = 15 := 
sorry

end NUMINAMATH_GPT_vasya_max_points_l121_12134


namespace NUMINAMATH_GPT_average_weight_increase_l121_12183

variable (A N X : ℝ)

theorem average_weight_increase (hN : N = 135.5) (h_avg : A + X = (9 * A - 86 + N) / 9) : 
  X = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l121_12183


namespace NUMINAMATH_GPT_range_of_m_l121_12147

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, f' x < x)
variable (h_ineq : ∀ m, f (4 - m) - f m ≥ 8 - 4 * m)

theorem range_of_m (m : ℝ) : m ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l121_12147


namespace NUMINAMATH_GPT_unique_k_for_prime_roots_of_quadratic_l121_12132

/-- Function to check primality of a natural number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Theorem statement with the given conditions -/
theorem unique_k_for_prime_roots_of_quadratic :
  ∃! k : ℕ, ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p * q = k :=
sorry

end NUMINAMATH_GPT_unique_k_for_prime_roots_of_quadratic_l121_12132


namespace NUMINAMATH_GPT_min_disks_needed_l121_12177

/-- 
  Sandhya must save 35 files onto disks, each with 1.44 MB space. 
  5 of the files take up 0.6 MB, 18 of the files take up 0.5 MB, 
  and the rest take up 0.3 MB. Files cannot be split across disks.
  Prove that the smallest number of disks needed to store all 35 files is 12.
--/
theorem min_disks_needed 
  (total_files : ℕ)
  (disk_capacity : ℝ)
  (file_sizes : ℕ → ℝ)
  (files_0_6_MB : ℕ)
  (files_0_5_MB : ℕ)
  (files_0_3_MB : ℕ)
  (remaining_files : ℕ)
  (storage_per_disk : ℝ)
  (smallest_disks_needed : ℕ) 
  (h1 : total_files = 35)
  (h2 : disk_capacity = 1.44)
  (h3 : file_sizes 0 = 0.6)
  (h4 : file_sizes 1 = 0.5)
  (h5 : file_sizes 2 = 0.3)
  (h6 : files_0_6_MB = 5)
  (h7 : files_0_5_MB = 18)
  (h8 : remaining_files = total_files - files_0_6_MB - files_0_5_MB)
  (h9 : remaining_files = 12)
  (h10 : storage_per_disk = file_sizes 0 * 2 + file_sizes 1 + file_sizes 2)
  (h11 : smallest_disks_needed = 12) :
  total_files = 35 ∧ disk_capacity = 1.44 ∧ storage_per_disk <= 1.44 ∧ smallest_disks_needed = 12 :=
by
  sorry

end NUMINAMATH_GPT_min_disks_needed_l121_12177


namespace NUMINAMATH_GPT_Diamond_evaluation_l121_12106

-- Redefine the operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^3 - b^2 + 1

-- Statement of the proof
theorem Diamond_evaluation : (Diamond 3 2) = 21 := by
  sorry

end NUMINAMATH_GPT_Diamond_evaluation_l121_12106


namespace NUMINAMATH_GPT_last_three_digits_of_8_pow_105_l121_12125

theorem last_three_digits_of_8_pow_105 : (8 ^ 105) % 1000 = 992 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_8_pow_105_l121_12125


namespace NUMINAMATH_GPT_range_of_m_l121_12103

theorem range_of_m (x m : ℝ) (h1 : (m - 1) / (x + 1) = 1) (h2 : x < 0) : m < 2 ∧ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l121_12103


namespace NUMINAMATH_GPT_mean_score_seniors_138_l121_12118

def total_students : ℕ := 200
def mean_score_all : ℕ := 120

variable (s n : ℕ) -- number of seniors and non-seniors
variable (ms mn : ℚ) -- mean score of seniors and non-seniors

def non_seniors_twice_seniors := n = 2 * s
def mean_score_non_seniors := mn = 0.8 * ms
def total_students_eq := s + n = total_students

def total_score := (s : ℚ) * ms + (n : ℚ) * mn = (total_students : ℚ) * mean_score_all

theorem mean_score_seniors_138 :
  ∃ s n ms mn,
    non_seniors_twice_seniors s n ∧
    mean_score_non_seniors ms mn ∧
    total_students_eq s n ∧
    total_score s n ms mn → 
    ms = 138 :=
sorry

end NUMINAMATH_GPT_mean_score_seniors_138_l121_12118


namespace NUMINAMATH_GPT_tangent_intersect_x_axis_l121_12133

-- Defining the conditions based on the given problem
def radius1 : ℝ := 3
def center1 : ℝ × ℝ := (0, 0)

def radius2 : ℝ := 5
def center2 : ℝ × ℝ := (12, 0)

-- Stating what needs to be proved
theorem tangent_intersect_x_axis : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (∀ (x1 x2 : ℝ), 
    (x1 = x) ∧ 
    (x2 = 12 - x) ∧ 
    (radius1 / (center2.1 - x) = radius2 / x2) → 
    (x = 9 / 2)) := 
sorry

end NUMINAMATH_GPT_tangent_intersect_x_axis_l121_12133


namespace NUMINAMATH_GPT_vector_projection_unique_l121_12114

theorem vector_projection_unique (a : ℝ) (c d : ℝ) (h : c + 3 * d = 0) :
    ∃ p : ℝ × ℝ, (∀ a : ℝ, ∀ (v : ℝ × ℝ) (w : ℝ × ℝ), 
      v = (a, 3 * a - 2) → 
      w = (c, d) → 
      ∃ p : ℝ × ℝ, p = (3 / 5, -1 / 5)) :=
sorry

end NUMINAMATH_GPT_vector_projection_unique_l121_12114


namespace NUMINAMATH_GPT_number_of_bedrooms_l121_12107

-- Conditions
def battery_life : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def num_initial_rooms : ℕ := 2 -- kitchen and living room
def num_charges : ℕ := 2

-- Computation of total vacuuming time
def total_vacuuming_time : ℕ := battery_life * (num_charges + 1)

-- Computation of remaining time for bedrooms
def time_for_bedrooms : ℕ := total_vacuuming_time - (vacuum_time_per_room * num_initial_rooms)

-- Proof problem: Prove number of bedrooms
theorem number_of_bedrooms (B : ℕ) (h : B = time_for_bedrooms / vacuum_time_per_room) : B = 5 := by 
  sorry

end NUMINAMATH_GPT_number_of_bedrooms_l121_12107


namespace NUMINAMATH_GPT_add_fractions_l121_12135

theorem add_fractions : (1 / 4 : ℚ) + (3 / 5) = 17 / 20 := 
by
  sorry

end NUMINAMATH_GPT_add_fractions_l121_12135


namespace NUMINAMATH_GPT_bailing_rate_bailing_problem_l121_12167

theorem bailing_rate (distance : ℝ) (rate_in : ℝ) (sink_limit : ℝ) (speed : ℝ) : ℝ :=
  let time_to_shore := distance / speed * 60 -- convert hours to minutes
  let total_intake := rate_in * time_to_shore
  let excess_water := total_intake - sink_limit
  excess_water / time_to_shore

theorem bailing_problem : bailing_rate 2 12 40 3 = 11 := by
  sorry

end NUMINAMATH_GPT_bailing_rate_bailing_problem_l121_12167


namespace NUMINAMATH_GPT_intersection_point_on_y_axis_l121_12187

theorem intersection_point_on_y_axis (k : ℝ) :
  ∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0 ↔ k = 6 ∨ k = -6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_on_y_axis_l121_12187


namespace NUMINAMATH_GPT_value_of_b_add_c_l121_12111

variables {a b c d : ℝ}

theorem value_of_b_add_c (h1 : a + b = 5) (h2 : c + d = 3) (h3 : a + d = 2) : b + c = 6 :=
sorry

end NUMINAMATH_GPT_value_of_b_add_c_l121_12111


namespace NUMINAMATH_GPT_slope_of_line_through_A_B_l121_12155

theorem slope_of_line_through_A_B :
  let A := (2, 1)
  let B := (-1, 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -2/3 :=
by
  have A_x : Int := 2
  have A_y : Int := 1
  have B_x : Int := -1
  have B_y : Int := 3
  sorry

end NUMINAMATH_GPT_slope_of_line_through_A_B_l121_12155


namespace NUMINAMATH_GPT_circle_radius_of_tangent_parabolas_l121_12158

theorem circle_radius_of_tangent_parabolas :
  ∃ r : ℝ, 
  (∀ (x : ℝ), (x^2 + r = x)) →
  r = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_of_tangent_parabolas_l121_12158


namespace NUMINAMATH_GPT_grasshopper_jumps_rational_angle_l121_12128

noncomputable def alpha_is_rational (α : ℝ) (jump : ℕ → ℝ × ℝ) : Prop :=
  ∃ k n : ℕ, (n ≠ 0) ∧ (jump n = (0, 0)) ∧ (α = (k : ℝ) / (n : ℝ) * 360)

theorem grasshopper_jumps_rational_angle :
  ∀ (α : ℝ) (jump : ℕ → ℝ × ℝ),
    (∀ n : ℕ, dist (jump (n + 1)) (jump n) = 1) →
    (jump 0 = (0, 0)) →
    (∃ n : ℕ, n ≠ 0 ∧ jump n = (0, 0)) →
    alpha_is_rational α jump :=
by
  intros α jump jumps_eq_1 start_exists returns_to_start
  sorry

end NUMINAMATH_GPT_grasshopper_jumps_rational_angle_l121_12128


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l121_12108

theorem roots_of_quadratic_eq : ∀ x : ℝ, (x^2 = 9) → (x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l121_12108


namespace NUMINAMATH_GPT_faye_country_albums_l121_12127

theorem faye_country_albums (C : ℕ) (h1 : 6 * C + 18 = 30) : C = 2 :=
by
  -- This is the theorem statement with the necessary conditions and question
  sorry

end NUMINAMATH_GPT_faye_country_albums_l121_12127


namespace NUMINAMATH_GPT_average_of_rest_l121_12161

theorem average_of_rest (A : ℝ) (total_students scoring_95 scoring_0 : ℕ) (total_avg : ℝ)
  (h_total_students : total_students = 25)
  (h_scoring_95 : scoring_95 = 3)
  (h_scoring_0 : scoring_0 = 3)
  (h_total_avg : total_avg = 45.6)
  (h_sum_eq : total_students * total_avg = 3 * 95 + 3 * 0 + (total_students - scoring_95 - scoring_0) * A) :
  A = 45 := sorry

end NUMINAMATH_GPT_average_of_rest_l121_12161


namespace NUMINAMATH_GPT_inequality_holds_for_real_numbers_l121_12198

theorem inequality_holds_for_real_numbers (a1 a2 a3 a4 : ℝ) (h1 : 1 < a1) 
  (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) : 
  8 * (a1 * a2 * a3 * a4 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) :=
by sorry

end NUMINAMATH_GPT_inequality_holds_for_real_numbers_l121_12198


namespace NUMINAMATH_GPT_total_tv_show_cost_correct_l121_12122

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end NUMINAMATH_GPT_total_tv_show_cost_correct_l121_12122


namespace NUMINAMATH_GPT_findCorrectAnswer_l121_12137

-- Definitions
variable (x : ℕ)
def mistakenCalculation : Prop := 3 * x = 90
def correctAnswer : ℕ := x - 30

-- Theorem statement
theorem findCorrectAnswer (h : mistakenCalculation x) : correctAnswer x = 0 :=
sorry

end NUMINAMATH_GPT_findCorrectAnswer_l121_12137


namespace NUMINAMATH_GPT_randy_quiz_goal_l121_12154

def randy_scores : List ℕ := [90, 98, 92, 94]
def randy_next_score : ℕ := 96
def randy_goal_average : ℕ := 94

theorem randy_quiz_goal :
  let total_score := randy_scores.sum
  let required_total_score := 470
  total_score + randy_next_score = required_total_score →
  required_total_score / randy_goal_average = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_randy_quiz_goal_l121_12154


namespace NUMINAMATH_GPT_min_value_of_expression_l121_12190

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 27) 
  : x + 3 * y + 9 * z ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l121_12190
