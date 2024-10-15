import Mathlib

namespace NUMINAMATH_GPT_quadratic_solution_property_l392_39267

theorem quadratic_solution_property :
  (∃ p q : ℝ, 3 * p^2 + 7 * p - 6 = 0 ∧ 3 * q^2 + 7 * q - 6 = 0 ∧ (p - 2) * (q - 2) = 6) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_property_l392_39267


namespace NUMINAMATH_GPT_trees_variance_l392_39252

theorem trees_variance :
  let groups := [3, 4, 3]
  let trees := [5, 6, 7]
  let n := 10
  let mean := (5 * 3 + 6 * 4 + 7 * 3) / n
  let variance := (3 * (5 - mean)^2 + 4 * (6 - mean)^2 + 3 * (7 - mean)^2) / n
  variance = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_trees_variance_l392_39252


namespace NUMINAMATH_GPT_base7_to_base10_l392_39262

theorem base7_to_base10 (a b : ℕ) (h : 235 = 49 * 2 + 7 * 3 + 5) (h_ab : 100 + 10 * a + b = 124) : 
  (a + b) / 7 = 6 / 7 :=
by
  sorry

end NUMINAMATH_GPT_base7_to_base10_l392_39262


namespace NUMINAMATH_GPT_Tyler_needs_more_eggs_l392_39208

noncomputable def recipe_eggs : ℕ := 2
noncomputable def recipe_milk : ℕ := 4
noncomputable def num_people : ℕ := 8
noncomputable def eggs_in_fridge : ℕ := 3

theorem Tyler_needs_more_eggs (recipe_eggs recipe_milk num_people eggs_in_fridge : ℕ)
  (h1 : recipe_eggs = 2)
  (h2 : recipe_milk = 4)
  (h3 : num_people = 8)
  (h4 : eggs_in_fridge = 3) :
  (num_people / 4) * recipe_eggs - eggs_in_fridge = 1 :=
by
  sorry

end NUMINAMATH_GPT_Tyler_needs_more_eggs_l392_39208


namespace NUMINAMATH_GPT_inequality_proof_l392_39245

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l392_39245


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l392_39241

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l392_39241


namespace NUMINAMATH_GPT_Lavinia_daughter_age_difference_l392_39247

-- Define the ages of the individuals involved
variables (Ld Ls Kd : ℕ)

-- Conditions given in the problem
variables (H1 : Kd = 12)
variables (H2 : Ls = 2 * Kd)
variables (H3 : Ls = Ld + 22)

-- Statement we need to prove
theorem Lavinia_daughter_age_difference(Ld Ls Kd : ℕ) (H1 : Kd = 12) (H2 : Ls = 2 * Kd) (H3 : Ls = Ld + 22) : 
  Kd - Ld = 10 :=
sorry

end NUMINAMATH_GPT_Lavinia_daughter_age_difference_l392_39247


namespace NUMINAMATH_GPT_balloon_highest_elevation_l392_39223

theorem balloon_highest_elevation
  (time_rise1 time_rise2 time_descent : ℕ)
  (rate_rise rate_descent : ℕ)
  (t1 : time_rise1 = 15)
  (t2 : time_rise2 = 15)
  (t3 : time_descent = 10)
  (rr : rate_rise = 50)
  (rd : rate_descent = 10)
  : (time_rise1 * rate_rise - time_descent * rate_descent + time_rise2 * rate_rise) = 1400 := 
by
  sorry

end NUMINAMATH_GPT_balloon_highest_elevation_l392_39223


namespace NUMINAMATH_GPT_remainder_for_second_number_l392_39239

theorem remainder_for_second_number (G R1 : ℕ) (first_number second_number : ℕ)
  (hG : G = 144) (hR1 : R1 = 23) (hFirst : first_number = 6215) (hSecond : second_number = 7373) :
  ∃ q2 R2, second_number = G * q2 + R2 ∧ R2 = 29 := 
by {
  -- Ensure definitions are in scope
  exact sorry
}

end NUMINAMATH_GPT_remainder_for_second_number_l392_39239


namespace NUMINAMATH_GPT_exists_k_for_inequality_l392_39259

noncomputable def C : ℕ := sorry -- C is a positive integer > 0
def a : ℕ → ℝ := sorry -- a sequence of positive real numbers

axiom C_pos : 0 < C
axiom a_pos : ∀ n : ℕ, 0 < a n
axiom recurrence_relation : ∀ n : ℕ, a (n + 1) = n / a n + C

theorem exists_k_for_inequality :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a (n + 2) > a n :=
  sorry

end NUMINAMATH_GPT_exists_k_for_inequality_l392_39259


namespace NUMINAMATH_GPT_average_customers_per_table_l392_39246

-- Definitions for conditions
def tables : ℝ := 9.0
def women : ℝ := 7.0
def men : ℝ := 3.0

-- Proof problem statement
theorem average_customers_per_table : (women + men) / tables = 10.0 / 9.0 :=
by
  sorry

end NUMINAMATH_GPT_average_customers_per_table_l392_39246


namespace NUMINAMATH_GPT_kitchen_upgrade_cost_l392_39219

def total_kitchen_upgrade_cost (num_knobs : ℕ) (cost_per_knob : ℝ) (num_pulls : ℕ) (cost_per_pull : ℝ) : ℝ :=
  (num_knobs * cost_per_knob) + (num_pulls * cost_per_pull)

theorem kitchen_upgrade_cost : total_kitchen_upgrade_cost 18 2.50 8 4.00 = 77.00 :=
  by
    sorry

end NUMINAMATH_GPT_kitchen_upgrade_cost_l392_39219


namespace NUMINAMATH_GPT_james_writing_hours_per_week_l392_39260

variables (pages_per_hour : ℕ) (pages_per_day_per_person : ℕ) (people : ℕ) (days_per_week : ℕ)

theorem james_writing_hours_per_week
  (h1 : pages_per_hour = 10)
  (h2 : pages_per_day_per_person = 5)
  (h3 : people = 2)
  (h4 : days_per_week = 7) :
  (pages_per_day_per_person * people * days_per_week) / pages_per_hour = 7 :=
by
  sorry

end NUMINAMATH_GPT_james_writing_hours_per_week_l392_39260


namespace NUMINAMATH_GPT_ice_cream_stacks_l392_39287

theorem ice_cream_stacks :
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  ways_to_stack = 120 :=
by
  let ice_cream_flavors := ["vanilla", "chocolate", "strawberry", "cherry", "banana"]
  let ways_to_stack := Nat.factorial ice_cream_flavors.length
  show (ways_to_stack = 120)
  sorry

end NUMINAMATH_GPT_ice_cream_stacks_l392_39287


namespace NUMINAMATH_GPT_fraction_sum_eq_neg_one_l392_39217

theorem fraction_sum_eq_neg_one (p q : ℝ) (hpq : (1 / p) + (1 / q) = (1 / (p + q))) :
  (p / q) + (q / p) = -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_eq_neg_one_l392_39217


namespace NUMINAMATH_GPT_least_positive_integer_l392_39270

open Nat

theorem least_positive_integer (n : ℕ) (h1 : n ≡ 2 [MOD 5]) (h2 : n ≡ 2 [MOD 4]) (h3 : n ≡ 0 [MOD 3]) : n = 42 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l392_39270


namespace NUMINAMATH_GPT_product_of_sum_and_reciprocal_nonneg_l392_39215

theorem product_of_sum_and_reciprocal_nonneg (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_product_of_sum_and_reciprocal_nonneg_l392_39215


namespace NUMINAMATH_GPT_ab_c_sum_geq_expr_ab_c_sum_eq_iff_l392_39248

theorem ab_c_sum_geq_expr (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) :=
sorry

theorem ab_c_sum_eq_iff (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_ab_c_sum_geq_expr_ab_c_sum_eq_iff_l392_39248


namespace NUMINAMATH_GPT_sin_alpha_is_neg_5_over_13_l392_39255

-- Definition of the problem conditions
variables (α : Real) (h1 : 0 < α) (h2 : α < 2 * Real.pi)
variable (quad4 : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)
variable (h3 : Real.tan α = -5 / 12)

-- Proof statement
theorem sin_alpha_is_neg_5_over_13:
  Real.sin α = -5 / 13 :=
sorry

end NUMINAMATH_GPT_sin_alpha_is_neg_5_over_13_l392_39255


namespace NUMINAMATH_GPT_product_of_a_values_l392_39244

/--
Let a be a real number and consider the points P = (3 * a, a - 5) and Q = (5, -2).
Given that the distance between P and Q is 3 * sqrt 10, prove that the product
of all possible values of a is -28 / 5.
-/
theorem product_of_a_values :
  ∀ (a : ℝ),
  (dist (3 * a, a - 5) (5, -2) = 3 * Real.sqrt 10) →
  ∃ (a₁ a₂ : ℝ), (5 * a₁ * a₁ - 18 * a₁ - 28 = 0) ∧ 
                 (5 * a₂ * a₂ - 18 * a₂ - 28 = 0) ∧ 
                 (a₁ * a₂ = -28 / 5) := 
by
  sorry

end NUMINAMATH_GPT_product_of_a_values_l392_39244


namespace NUMINAMATH_GPT_a_41_eq_6585451_l392_39234

noncomputable def a : ℕ → ℕ
| 0     => 0 /- Not used practically since n >= 1 -/
| 1     => 1
| 2     => 1
| 3     => 2
| (n+4) => a n + a (n+2) + 1

theorem a_41_eq_6585451 : a 41 = 6585451 := by
  sorry

end NUMINAMATH_GPT_a_41_eq_6585451_l392_39234


namespace NUMINAMATH_GPT_vegetation_coverage_relationship_l392_39229

noncomputable def conditions :=
  let n := 20
  let sum_x := 60
  let sum_y := 1200
  let sum_xx := 80
  let sum_xy := 640
  (n, sum_x, sum_y, sum_xx, sum_xy)

theorem vegetation_coverage_relationship
  (n sum_x sum_y sum_xx sum_xy : ℕ)
  (h1 : n = 20)
  (h2 : sum_x = 60)
  (h3 : sum_y = 1200)
  (h4 : sum_xx = 80)
  (h5 : sum_xy = 640) :
  let b1 := sum_xy / sum_xx
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  (b1 = 8) ∧ (b1 * (sum_xx / sum_xy) ≤ 1) ∧ ((3, 60) = (mean_x, mean_y)) :=
by
  sorry

end NUMINAMATH_GPT_vegetation_coverage_relationship_l392_39229


namespace NUMINAMATH_GPT_age_of_youngest_child_l392_39202

/-- Given that the sum of ages of 5 children born at 3-year intervals is 70, prove the age of the youngest child is 8. -/
theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 := 
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l392_39202


namespace NUMINAMATH_GPT_abs_condition_implies_l392_39250

theorem abs_condition_implies (x : ℝ) 
  (h : |x - 1| < 2) : x < 3 := by
  sorry

end NUMINAMATH_GPT_abs_condition_implies_l392_39250


namespace NUMINAMATH_GPT_sum_of_first_n_primes_eq_41_l392_39214

theorem sum_of_first_n_primes_eq_41 : 
  ∃ (n : ℕ) (primes : List ℕ), 
    primes = [2, 3, 5, 7, 11, 13] ∧ primes.sum = 41 ∧ primes.length = n := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_n_primes_eq_41_l392_39214


namespace NUMINAMATH_GPT_percentage_first_less_third_l392_39243

variable (A B C : ℝ)

theorem percentage_first_less_third :
  B = 0.58 * C → B = 0.8923076923076923 * A → (100 - (A / C * 100)) = 35 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_percentage_first_less_third_l392_39243


namespace NUMINAMATH_GPT_find_m_l392_39233

theorem find_m 
  (h : ( (1 ^ m) / (5 ^ m) ) * ( (1 ^ 16) / (4 ^ 16) ) = 1 / (2 * 10 ^ 31)) :
  m = 31 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l392_39233


namespace NUMINAMATH_GPT_line_points_k_l392_39293

noncomputable def k : ℝ := 8

theorem line_points_k (k : ℝ) : 
  (∀ k : ℝ, ∃ b : ℝ, b = (10 - k) / (5 - 5) ∧
  ∀ b, b = (-k) / (20 - 5) → k = 8) :=
  by
  sorry

end NUMINAMATH_GPT_line_points_k_l392_39293


namespace NUMINAMATH_GPT_leap_year_hours_l392_39269

theorem leap_year_hours (days_in_regular_year : ℕ) (hours_in_day : ℕ) (is_leap_year : Bool) : 
  is_leap_year = true ∧ days_in_regular_year = 365 ∧ hours_in_day = 24 → 
  366 * hours_in_day = 8784 :=
by
  intros
  sorry

end NUMINAMATH_GPT_leap_year_hours_l392_39269


namespace NUMINAMATH_GPT_unique_shape_determination_l392_39236

theorem unique_shape_determination (ratio_sides_median : Prop) (ratios_three_sides : Prop) 
                                   (ratio_circumradius_side : Prop) (ratio_two_angles : Prop) 
                                   (length_one_side_heights : Prop) :
  ¬(ratio_circumradius_side → (ratio_sides_median ∧ ratios_three_sides ∧ ratio_two_angles ∧ length_one_side_heights)) := 
sorry

end NUMINAMATH_GPT_unique_shape_determination_l392_39236


namespace NUMINAMATH_GPT_find_expression_for_f_x_neg_l392_39280

theorem find_expression_for_f_x_neg (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_pos : ∀ x, 0 < x → f x = x - Real.log (abs x)) :
  ∀ x, x < 0 → f x = x + Real.log (abs x) :=
by
  sorry

end NUMINAMATH_GPT_find_expression_for_f_x_neg_l392_39280


namespace NUMINAMATH_GPT_geom_seq_div_a5_a7_l392_39274

variable {a : ℕ → ℝ}

-- Given sequence is geometric and positive
def is_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Positive geometric sequence with decreasing terms
def is_positive_decreasing_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  is_geom_sequence a r ∧ ∀ n, a (n + 1) < a n ∧ a n > 0

-- Conditions
variables (r : ℝ) (hp : is_positive_decreasing_geom_sequence a r)
           (h2 : a 2 * a 8 = 6) (h3 : a 4 + a 6 = 5)

-- Goal
theorem geom_seq_div_a5_a7 : a 5 / a 7 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_div_a5_a7_l392_39274


namespace NUMINAMATH_GPT_number_is_7625_l392_39216

-- We define x as a real number
variable (x : ℝ)

-- The condition given in the problem
def condition : Prop := x^2 + 95 = (x - 20)^2

-- The theorem we need to prove
theorem number_is_7625 (h : condition x) : x = 7.625 :=
by
  sorry

end NUMINAMATH_GPT_number_is_7625_l392_39216


namespace NUMINAMATH_GPT_solution_set_of_inequality_l392_39224

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h1 : ∀ x : ℝ, deriv f x = 2 * f x)
                                    (h2 : f 0 = 1) :
  { x : ℝ | f (Real.log (x^2 - x)) < 4 } = { x | -1 < x ∧ x < 0 ∨ 1 < x ∧ x < 2 } :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_of_inequality_l392_39224


namespace NUMINAMATH_GPT_inequality_proof_l392_39227

open Real

theorem inequality_proof (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l392_39227


namespace NUMINAMATH_GPT_total_nuggets_ordered_l392_39290

noncomputable def Alyssa_nuggets : ℕ := 20
noncomputable def Keely_nuggets : ℕ := 2 * Alyssa_nuggets
noncomputable def Kendall_nuggets : ℕ := 2 * Alyssa_nuggets

theorem total_nuggets_ordered : Alyssa_nuggets + Keely_nuggets + Kendall_nuggets = 100 := by
  sorry -- Proof is intentionally omitted

end NUMINAMATH_GPT_total_nuggets_ordered_l392_39290


namespace NUMINAMATH_GPT_siding_cost_l392_39213

noncomputable def front_wall_width : ℝ := 10
noncomputable def front_wall_height : ℝ := 8
noncomputable def triangle_base : ℝ := 10
noncomputable def triangle_height : ℝ := 4
noncomputable def panel_area : ℝ := 100
noncomputable def panel_cost : ℝ := 30

theorem siding_cost :
  let front_wall_area := front_wall_width * front_wall_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_area := front_wall_area + triangle_area
  let panels_needed := total_area / panel_area
  let total_cost := panels_needed * panel_cost
  total_cost = 30 := sorry

end NUMINAMATH_GPT_siding_cost_l392_39213


namespace NUMINAMATH_GPT_prob_A_eq_prob_B_l392_39288

-- Define the number of students and the number of tickets
def num_students : ℕ := 56
def num_tickets : ℕ := 56
def prize_tickets : ℕ := 1

-- Define the probability of winning the prize for a given student (A for first student, B for last student)
def prob_A := prize_tickets / num_tickets
def prob_B := prize_tickets / num_tickets

-- Statement to prove
theorem prob_A_eq_prob_B : prob_A = prob_B :=
by 
  -- We provide the statement to prove without the proof steps
  sorry

end NUMINAMATH_GPT_prob_A_eq_prob_B_l392_39288


namespace NUMINAMATH_GPT_count_even_digits_in_512_base_7_l392_39240

def base7_representation (n : ℕ) : ℕ := 
  sorry  -- Assuming this function correctly computes the base-7 representation of a natural number

def even_digits_count (n : ℕ) : ℕ :=
  sorry  -- Assuming this function correctly counts the even digits in the base-7 representation

theorem count_even_digits_in_512_base_7 : 
  even_digits_count (base7_representation 512) = 0 :=
by
  sorry

end NUMINAMATH_GPT_count_even_digits_in_512_base_7_l392_39240


namespace NUMINAMATH_GPT_power_function_m_eq_4_l392_39258

theorem power_function_m_eq_4 (m : ℝ) :
  (m^2 - 3*m - 3 = 1) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_power_function_m_eq_4_l392_39258


namespace NUMINAMATH_GPT_team_winning_percentage_l392_39230

theorem team_winning_percentage :
  let first_games := 100
  let remaining_games := 125 - first_games
  let won_first_games := 75
  let percentage_won := 50
  let won_remaining_games := Nat.ceil ((percentage_won : ℝ) / 100 * remaining_games)
  let total_won_games := won_first_games + won_remaining_games
  let total_games := 125
  let winning_percentage := (total_won_games : ℝ) / total_games * 100
  winning_percentage = 70.4 :=
by sorry

end NUMINAMATH_GPT_team_winning_percentage_l392_39230


namespace NUMINAMATH_GPT_x_intercept_of_line_l392_39298

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) :
  ∃ x0 : ℝ, (∀ y : ℝ, y = 0 → (∃ m : ℝ, y = m * (x0 - x1) + y1)) ∧ x0 = 4 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l392_39298


namespace NUMINAMATH_GPT_all_statements_imply_negation_l392_39271

theorem all_statements_imply_negation :
  let s1 := (true ∧ true ∧ false)
  let s2 := (false ∧ true ∧ true)
  let s3 := (true ∧ false ∧ true)
  let s4 := (false ∧ false ∧ true)
  (s1 → ¬(true ∧ true ∧ true)) ∧
  (s2 → ¬(true ∧ true ∧ true)) ∧
  (s3 → ¬(true ∧ true ∧ true)) ∧
  (s4 → ¬(true ∧ true ∧ true)) :=
by sorry

end NUMINAMATH_GPT_all_statements_imply_negation_l392_39271


namespace NUMINAMATH_GPT_total_wall_area_l392_39222

variable (L W : ℝ) -- Length and width of the regular tile
variable (R : ℕ) -- Number of regular tiles

-- Conditions:
-- 1. The area covered by regular tiles is 70 square feet.
axiom regular_tiles_cover_area : R * (L * W) = 70

-- 2. Jumbo tiles make up 1/3 of the total tiles, and each jumbo tile has an area three times that of a regular tile.
axiom length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105

theorem total_wall_area (L W : ℝ) (R : ℕ) 
  (regular_tiles_cover_area : R * (L * W) = 70) 
  (length_ratio : ∀ jumbo_tiles, 3 * (jumbo_tiles * (L * W)) = 105) : 
  (R * (L * W)) + (3 * (R / 2) * (L * W)) = 175 :=
by
  sorry

end NUMINAMATH_GPT_total_wall_area_l392_39222


namespace NUMINAMATH_GPT_deposit_increases_l392_39225

theorem deposit_increases (X r s : ℝ) (hX : 0 < X) (hr : 0 ≤ r) (hs : s < 20) : 
  r > 100 * s / (100 - s) :=
by sorry

end NUMINAMATH_GPT_deposit_increases_l392_39225


namespace NUMINAMATH_GPT_find_q_l392_39268

-- Define the roots of the polynomial 2x^2 - 6x + 1 = 0
def roots_of_first_poly (a b : ℝ) : Prop :=
    2 * a^2 - 6 * a + 1 = 0 ∧ 2 * b^2 - 6 * b + 1 = 0

-- Conditions from Vieta's formulas for the first polynomial
def sum_of_roots (a b : ℝ) : Prop := a + b = 3
def product_of_roots (a b : ℝ) : Prop := a * b = 0.5

-- Define the roots of the second polynomial x^2 + px + q = 0
def roots_of_second_poly (a b : ℝ) (p q : ℝ) : Prop :=
    (λ x => x^2 + p * x + q) (3 * a - 1) = 0 ∧ 
    (λ x => x^2 + p * x + q) (3 * b - 1) = 0

-- Proof that q = -0.5 given the conditions
theorem find_q (a b p q : ℝ) (h1 : roots_of_first_poly a b) (h2 : sum_of_roots a b)
    (h3 : product_of_roots a b) (h4 : roots_of_second_poly a b p q) : q = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l392_39268


namespace NUMINAMATH_GPT_village_population_percentage_l392_39207

theorem village_population_percentage (P0 P2 P1 : ℝ) (x : ℝ)
  (hP0 : P0 = 7800)
  (hP2 : P2 = 5265)
  (hP1 : P1 = P0 * (1 - x / 100))
  (hP2_eq : P2 = P1 * 0.75) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_village_population_percentage_l392_39207


namespace NUMINAMATH_GPT_similar_triangles_ratios_l392_39226

-- Define the context
variables {a b c a' b' c' : ℂ}

-- Define the statement of the problem
theorem similar_triangles_ratios (h_sim : ∃ z : ℂ, z ≠ 0 ∧ b - a = z * (b' - a') ∧ c - a = z * (c' - a')) :
  (b - a) / (c - a) = (b' - a') / (c' - a') :=
sorry

end NUMINAMATH_GPT_similar_triangles_ratios_l392_39226


namespace NUMINAMATH_GPT_james_total_room_area_l392_39272

theorem james_total_room_area :
  let original_length := 13
  let original_width := 18
  let increase := 2
  let new_length := original_length + increase
  let new_width := original_width + increase
  let area_of_one_room := new_length * new_width
  let number_of_rooms := 4
  let area_of_four_rooms := area_of_one_room * number_of_rooms
  let area_of_larger_room := area_of_one_room * 2
  let total_area := area_of_four_rooms + area_of_larger_room
  total_area = 1800
  := sorry

end NUMINAMATH_GPT_james_total_room_area_l392_39272


namespace NUMINAMATH_GPT_part_one_part_two_l392_39238

-- 1. Prove that 1 + 2x^4 >= 2x^3 + x^2 for all real numbers x
theorem part_one (x : ℝ) : 1 + 2 * x^4 ≥ 2 * x^3 + x^2 := sorry

-- 2. Given x + 2y + 3z = 6, prove that x^2 + y^2 + z^2 ≥ 18 / 7
theorem part_two (x y z : ℝ) (h : x + 2 * y + 3 * z = 6) : x^2 + y^2 + z^2 ≥ 18 / 7 := sorry

end NUMINAMATH_GPT_part_one_part_two_l392_39238


namespace NUMINAMATH_GPT_simplify_expression_l392_39237

theorem simplify_expression (x y : ℝ) :
  ((x + y)^2 - y * (2 * x + y) - 6 * x) / (2 * x) = (1 / 2) * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l392_39237


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l392_39278

theorem angle_in_third_quadrant (α : ℝ) (h : α = 2023) : 180 < α % 360 ∧ α % 360 < 270 := by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l392_39278


namespace NUMINAMATH_GPT_deposit_percentage_correct_l392_39212

-- Define the conditions
def deposit_amount : ℕ := 50
def remaining_amount : ℕ := 950
def total_cost : ℕ := deposit_amount + remaining_amount

-- Define the proof problem statement
theorem deposit_percentage_correct :
  (deposit_amount / total_cost : ℚ) * 100 = 5 := 
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_deposit_percentage_correct_l392_39212


namespace NUMINAMATH_GPT_rocket_altitude_time_l392_39221

theorem rocket_altitude_time (a₁ d : ℕ) (n : ℕ) (h₁ : a₁ = 2) (h₂ : d = 2)
  (h₃ : n * a₁ + (n * (n - 1) * d) / 2 = 240) : n = 15 :=
by
  -- The proof is ignored as per instruction.
  sorry

end NUMINAMATH_GPT_rocket_altitude_time_l392_39221


namespace NUMINAMATH_GPT_total_rent_of_field_is_correct_l392_39220

namespace PastureRental

def cowMonths (cows : ℕ) (months : ℕ) : ℕ := cows * months

def aCowMonths : ℕ := cowMonths 24 3
def bCowMonths : ℕ := cowMonths 10 5
def cCowMonths : ℕ := cowMonths 35 4
def dCowMonths : ℕ := cowMonths 21 3

def totalCowMonths : ℕ := aCowMonths + bCowMonths + cCowMonths + dCowMonths

def rentPerCowMonth : ℕ := 1440 / aCowMonths

def totalRent : ℕ := rentPerCowMonth * totalCowMonths

theorem total_rent_of_field_is_correct :
  totalRent = 6500 :=
by
  sorry

end PastureRental

end NUMINAMATH_GPT_total_rent_of_field_is_correct_l392_39220


namespace NUMINAMATH_GPT_tan_15_eq_sqrt3_l392_39297

theorem tan_15_eq_sqrt3 :
  (1 + Real.tan (Real.pi / 12)) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_15_eq_sqrt3_l392_39297


namespace NUMINAMATH_GPT_grazing_months_of_A_l392_39276

-- Definitions of conditions
def oxen_months_A (x : ℕ) := 10 * x
def oxen_months_B := 12 * 5
def oxen_months_C := 15 * 3
def total_rent := 140
def rent_C := 36

-- Assuming a is the number of months a put his oxen for grazing, we need to prove that a = 7
theorem grazing_months_of_A (a : ℕ) :
  (45 * 140 = 36 * (10 * a + 60 + 45)) → a = 7 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_grazing_months_of_A_l392_39276


namespace NUMINAMATH_GPT_percent_employed_in_town_l392_39203

theorem percent_employed_in_town (E : ℝ) : 
  (0.14 * E) + 55 = E → E = 64 :=
by
  intro h
  have h1: 0.14 * E + 55 = E := h
  -- Proof step here, but we put sorry to skip the proof
  sorry

end NUMINAMATH_GPT_percent_employed_in_town_l392_39203


namespace NUMINAMATH_GPT_right_triangle_max_value_l392_39261

theorem right_triangle_max_value (a b c : ℝ) (h : a^2 + b^2 = c^2) :
    (a + b) / (ab / c) ≤ 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_right_triangle_max_value_l392_39261


namespace NUMINAMATH_GPT_minimum_ladder_rungs_l392_39295

theorem minimum_ladder_rungs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b): ∃ n, n = a + b - 1 :=
by
    sorry

end NUMINAMATH_GPT_minimum_ladder_rungs_l392_39295


namespace NUMINAMATH_GPT_mod_equiv_example_l392_39265

theorem mod_equiv_example : (185 * 944) % 60 = 40 := by
  sorry

end NUMINAMATH_GPT_mod_equiv_example_l392_39265


namespace NUMINAMATH_GPT_problem_proof_l392_39281

theorem problem_proof :
  1.25 * 67.875 + 125 * 6.7875 + 1250 * 0.053375 = 1000 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l392_39281


namespace NUMINAMATH_GPT_acme_profit_l392_39257

-- Define the given problem conditions
def initial_outlay : ℝ := 12450
def cost_per_set : ℝ := 20.75
def selling_price_per_set : ℝ := 50
def num_sets : ℝ := 950

-- Define the total revenue and total manufacturing costs
def total_revenue : ℝ := num_sets * selling_price_per_set
def total_cost : ℝ := initial_outlay + (cost_per_set * num_sets)

-- State the profit calculation and the expected result
def profit : ℝ := total_revenue - total_cost

theorem acme_profit : profit = 15337.50 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_acme_profit_l392_39257


namespace NUMINAMATH_GPT_Juliska_correct_l392_39277

-- Definitions according to the conditions in a)
def has_three_rum_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "rum" ∈ selected_triplet

def has_three_coffee_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "coffee" ∈ selected_triplet

-- Proof problem statement
theorem Juliska_correct 
  (candies : List String) 
  (h_rum : has_three_rum_candy candies)
  (h_coffee : has_three_coffee_candy candies) : 
  (∀ (selected_triplet : List String), selected_triplet.length = 3 → "walnut" ∈ selected_triplet) :=
sorry

end NUMINAMATH_GPT_Juliska_correct_l392_39277


namespace NUMINAMATH_GPT_num_rooms_l392_39210

theorem num_rooms (r1 r2 w1 w2 p w_paint : ℕ) (h_r1 : r1 = 5) (h_r2 : r2 = 4) (h_w1 : w1 = 4) (h_w2 : w2 = 5)
    (h_p : p = 5) (h_w_paint : w_paint = 8) (h_total_walls_family : p * w_paint = (r1 * w1 + r2 * w2)) :
    (r1 + r2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_num_rooms_l392_39210


namespace NUMINAMATH_GPT_mark_percentage_increase_l392_39256

-- Given a game with the following conditions:
-- Condition 1: Samanta has 8 more points than Mark
-- Condition 2: Eric has 6 points
-- Condition 3: The total points of Samanta, Mark, and Eric is 32

theorem mark_percentage_increase (S M : ℕ) (h1 : S = M + 8) (h2 : 6 + S + M = 32) : 
  (M - 6) / 6 * 100 = 50 :=
sorry

end NUMINAMATH_GPT_mark_percentage_increase_l392_39256


namespace NUMINAMATH_GPT_area_of_rectangle_ABCD_l392_39284

-- Definitions based on conditions
def side_length_smaller_square := 2
def area_smaller_square := side_length_smaller_square ^ 2
def side_length_larger_square := 3 * side_length_smaller_square
def area_larger_square := side_length_larger_square ^ 2
def area_rect_ABCD := 2 * area_smaller_square + area_larger_square

-- Lean theorem statement for the proof problem
theorem area_of_rectangle_ABCD : area_rect_ABCD = 44 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_ABCD_l392_39284


namespace NUMINAMATH_GPT_abs_neg_2023_l392_39294

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l392_39294


namespace NUMINAMATH_GPT_fraction_zero_iff_x_one_l392_39251

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_zero_iff_x_one_l392_39251


namespace NUMINAMATH_GPT_average_speed_of_trip_l392_39209

theorem average_speed_of_trip :
  let speed1 := 30
  let time1 := 5
  let speed2 := 42
  let time2 := 10
  let total_time := 15
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let average_speed := total_distance / total_time
  average_speed = 38 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_of_trip_l392_39209


namespace NUMINAMATH_GPT_millet_more_than_half_l392_39228

def daily_millet (n : ℕ) : ℝ :=
  1 - (0.7)^n

theorem millet_more_than_half (n : ℕ) : daily_millet 2 > 0.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_millet_more_than_half_l392_39228


namespace NUMINAMATH_GPT_total_kids_in_Lawrence_l392_39205

theorem total_kids_in_Lawrence (stay_home kids_camp total_kids : ℕ) (h1 : stay_home = 907611) (h2 : kids_camp = 455682) (h3 : total_kids = stay_home + kids_camp) : total_kids = 1363293 :=
by
  sorry

end NUMINAMATH_GPT_total_kids_in_Lawrence_l392_39205


namespace NUMINAMATH_GPT_sum_coefficients_l392_39200

theorem sum_coefficients (a1 a2 a3 a4 a5 : ℤ) (h : ∀ x : ℕ, a1 * (x - 1) ^ 4 + a2 * (x - 1) ^ 3 + a3 * (x - 1) ^ 2 + a4 * (x - 1) + a5 = x ^ 4) :
  a2 + a3 + a4 = 14 :=
  sorry

end NUMINAMATH_GPT_sum_coefficients_l392_39200


namespace NUMINAMATH_GPT_product_of_three_3_digits_has_four_zeros_l392_39275

noncomputable def has_four_zeros_product : Prop :=
  ∃ (a b c: ℕ),
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (100 ≤ c ∧ c < 1000) ∧
    (∃ (da db dc: Finset ℕ), (da ∪ db ∪ dc = Finset.range 10) ∧
    (∀ x ∈ da, x = a / 10^(x%10) % 10) ∧
    (∀ x ∈ db, x = b / 10^(x%10) % 10) ∧
    (∀ x ∈ dc, x = c / 10^(x%10) % 10)) ∧
    (a * b * c % 10000 = 0)

theorem product_of_three_3_digits_has_four_zeros : has_four_zeros_product := sorry

end NUMINAMATH_GPT_product_of_three_3_digits_has_four_zeros_l392_39275


namespace NUMINAMATH_GPT_total_money_l392_39242

def Billy_money (S : ℕ) := 3 * S - 150
def Lila_money (B S : ℕ) := B - S

theorem total_money (S B L : ℕ) (h1 : B = Billy_money S) (h2 : S = 200) (h3 : L = Lila_money B S) : 
  S + B + L = 900 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_total_money_l392_39242


namespace NUMINAMATH_GPT_problem_statement_l392_39283

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l392_39283


namespace NUMINAMATH_GPT_movie_of_the_year_condition_l392_39231

noncomputable def smallest_needed_lists : Nat :=
  let total_lists := 765
  let required_fraction := 1 / 4
  Nat.ceil (total_lists * required_fraction)

theorem movie_of_the_year_condition :
  smallest_needed_lists = 192 := by
  sorry

end NUMINAMATH_GPT_movie_of_the_year_condition_l392_39231


namespace NUMINAMATH_GPT_squares_characterization_l392_39204

theorem squares_characterization (n : ℕ) (a b : ℤ) (h_cond : n + 1 = a^2 + (a + 1)^2 ∧ n + 1 = b^2 + 2 * (b + 1)^2) :
  ∃ k l : ℤ, 2 * n + 1 = k^2 ∧ 3 * n + 1 = l^2 :=
sorry

end NUMINAMATH_GPT_squares_characterization_l392_39204


namespace NUMINAMATH_GPT_remaining_half_speed_l392_39266

-- Define the given conditions
def total_time : ℕ := 11
def first_half_distance : ℕ := 150
def first_half_speed : ℕ := 30
def total_distance : ℕ := 300

-- Prove the speed for the remaining half of the distance
theorem remaining_half_speed :
  ∃ v : ℕ, v = 25 ∧
  (total_distance = 2 * first_half_distance) ∧
  (first_half_distance / first_half_speed = 5) ∧
  (total_time = 5 + (first_half_distance / v)) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_remaining_half_speed_l392_39266


namespace NUMINAMATH_GPT_not_equivalent_to_0_0000375_l392_39218

theorem not_equivalent_to_0_0000375 : 
    ¬ (3 / 8000000 = 3.75 * 10 ^ (-5)) :=
by sorry

end NUMINAMATH_GPT_not_equivalent_to_0_0000375_l392_39218


namespace NUMINAMATH_GPT_clownfish_display_tank_l392_39292

theorem clownfish_display_tank
  (C B : ℕ)
  (h1 : C = B)
  (h2 : C + B = 100)
  (h3 : ∀ dC dB : ℕ, dC = dB → C - dC = 24)
  (h4 : ∀ b : ℕ, b = (1 / 3) * 24): 
  C - (1 / 3 * 24) = 16 := sorry

end NUMINAMATH_GPT_clownfish_display_tank_l392_39292


namespace NUMINAMATH_GPT_time_after_9999_seconds_l392_39211

theorem time_after_9999_seconds:
  let initial_hours := 5
  let initial_minutes := 45
  let initial_seconds := 0
  let added_seconds := 9999
  let total_seconds := initial_seconds + added_seconds
  let total_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let final_hours := (initial_hours + total_hours + (initial_minutes + remaining_minutes) / 60) % 24
  let final_minutes := (initial_minutes + remaining_minutes) % 60
  initial_hours = 5 →
  initial_minutes = 45 →
  initial_seconds = 0 →
  added_seconds = 9999 →
  final_hours = 8 ∧ final_minutes = 31 ∧ remaining_seconds = 39 :=
by
  intros
  sorry

end NUMINAMATH_GPT_time_after_9999_seconds_l392_39211


namespace NUMINAMATH_GPT_translate_parabola_l392_39286

theorem translate_parabola :
  ∀ (x y : ℝ), y = -5*x^2 + 1 → y = -5*(x + 1)^2 - 1 := by
  sorry

end NUMINAMATH_GPT_translate_parabola_l392_39286


namespace NUMINAMATH_GPT_max_correct_questions_l392_39296

theorem max_correct_questions (a b c : ℕ) (h1 : a + b + c = 60) (h2 : 3 * a - 2 * c = 126) : a ≤ 49 :=
sorry

end NUMINAMATH_GPT_max_correct_questions_l392_39296


namespace NUMINAMATH_GPT_find_k_l392_39273

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n - 2

theorem find_k :
  ∃ k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 35 ∧ k = 29 := 
sorry

end NUMINAMATH_GPT_find_k_l392_39273


namespace NUMINAMATH_GPT_total_movies_shown_l392_39235

theorem total_movies_shown (screen1_movies : ℕ) (screen2_movies : ℕ) (screen3_movies : ℕ)
                          (screen4_movies : ℕ) (screen5_movies : ℕ) (screen6_movies : ℕ)
                          (h1 : screen1_movies = 3) (h2 : screen2_movies = 4) 
                          (h3 : screen3_movies = 2) (h4 : screen4_movies = 3) 
                          (h5 : screen5_movies = 5) (h6 : screen6_movies = 2) :
  screen1_movies + screen2_movies + screen3_movies + screen4_movies + screen5_movies + screen6_movies = 19 := 
by
  sorry

end NUMINAMATH_GPT_total_movies_shown_l392_39235


namespace NUMINAMATH_GPT_compute_p2_q2_compute_p3_q3_l392_39299

variables (p q : ℝ)

theorem compute_p2_q2 (h1 : p * q = 15) (h2 : p + q = 8) : p^2 + q^2 = 34 :=
sorry

theorem compute_p3_q3 (h1 : p * q = 15) (h2 : p + q = 8) : p^3 + q^3 = 152 :=
sorry

end NUMINAMATH_GPT_compute_p2_q2_compute_p3_q3_l392_39299


namespace NUMINAMATH_GPT_museum_discount_l392_39201

theorem museum_discount
  (Dorothy_age : ℕ)
  (total_family_members : ℕ)
  (regular_ticket_cost : ℕ)
  (discountapplies_age : ℕ)
  (before_trip : ℕ)
  (after_trip : ℕ)
  (spend : ℕ := before_trip - after_trip)
  (adults_tickets : ℕ := total_family_members - 2)
  (youth_tickets : ℕ := 2)
  (total_cost := adults_tickets * regular_ticket_cost + youth_tickets * (regular_ticket_cost - regular_ticket_cost * discount))
  (discount : ℚ)
  (expected_spend : ℕ := 44) :
  total_cost = spend :=
by
  sorry

end NUMINAMATH_GPT_museum_discount_l392_39201


namespace NUMINAMATH_GPT_product_of_two_numbers_is_21_l392_39282

noncomputable def product_of_two_numbers (x y : ℝ) : ℝ :=
  x * y

theorem product_of_two_numbers_is_21 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^2 + y^2 = 58) :
  product_of_two_numbers x y = 21 :=
by sorry

end NUMINAMATH_GPT_product_of_two_numbers_is_21_l392_39282


namespace NUMINAMATH_GPT_santana_brothers_l392_39285

theorem santana_brothers (b : ℕ) (x : ℕ) (h1 : x + b = 7) (h2 : 3 + 8 = x + 1 + 2 + 7) : x = 1 :=
by
  -- Providing the necessary definitions and conditions
  let brothers := 7 -- Santana has 7 brothers
  let march_birthday := 3 -- 3 brothers have birthdays in March
  let november_birthday := 1 -- 1 brother has a birthday in November
  let december_birthday := 2 -- 2 brothers have birthdays in December
  let total_presents_first_half := 3 -- Total presents in the first half of the year is 3 (March)
  let x := x -- Number of brothers with birthdays in October to be proved
  let total_presents_second_half := x + 1 + 2 + 7 -- Total presents in the second half of the year
  have h3 : total_presents_first_half + 8 = total_presents_second_half := h2 -- Condition equation
  
  -- Start solving the proof
  sorry

end NUMINAMATH_GPT_santana_brothers_l392_39285


namespace NUMINAMATH_GPT_overall_gain_percentage_l392_39206

theorem overall_gain_percentage :
  let SP1 := 100
  let SP2 := 150
  let SP3 := 200
  let CP1 := SP1 / (1 + 0.20)
  let CP2 := SP2 / (1 + 0.15)
  let CP3 := SP3 / (1 - 0.05)
  let TCP := CP1 + CP2 + CP3
  let TSP := SP1 + SP2 + SP3
  let G := TSP - TCP
  let GP := (G / TCP) * 100
  GP = 6.06 := 
by {
  sorry
}

end NUMINAMATH_GPT_overall_gain_percentage_l392_39206


namespace NUMINAMATH_GPT_measure_of_angle_F_l392_39263

-- Definitions for the angles in triangle DEF
variables (D E F : ℝ)

-- Given conditions
def is_right_triangle (D : ℝ) : Prop := D = 90
def angle_relation (E F : ℝ) : Prop := E = 4 * F - 10
def angle_sum (D E F : ℝ) : Prop := D + E + F = 180

-- The proof problem statement
theorem measure_of_angle_F (h1 : is_right_triangle D) (h2 : angle_relation E F) (h3 : angle_sum D E F) : F = 20 :=
sorry

end NUMINAMATH_GPT_measure_of_angle_F_l392_39263


namespace NUMINAMATH_GPT_number_of_solutions_l392_39232

theorem number_of_solutions :
  ∃ S : Finset (ℤ × ℤ), 
  (∀ (m n : ℤ), (m, n) ∈ S ↔ m^4 + 8 * n^2 + 425 = n^4 + 42 * m^2) ∧ 
  S.card = 16 :=
by { sorry }

end NUMINAMATH_GPT_number_of_solutions_l392_39232


namespace NUMINAMATH_GPT_leak_time_to_empty_tank_l392_39254

-- Define variables for the rates
variable (A L : ℝ)

-- Given conditions
def rate_pipe_A : Prop := A = 1 / 4
def combined_rate : Prop := A - L = 1 / 6

-- Theorem statement: The time it takes for the leak to empty the tank
theorem leak_time_to_empty_tank (A L : ℝ) (h1 : rate_pipe_A A) (h2 : combined_rate A L) : 1 / L = 12 :=
by 
  sorry

end NUMINAMATH_GPT_leak_time_to_empty_tank_l392_39254


namespace NUMINAMATH_GPT_correct_divisor_l392_39264

theorem correct_divisor (D : ℕ) (X : ℕ) (H1 : X = 70 * (D + 12)) (H2 : X = 40 * D) : D = 28 := 
by 
  sorry

end NUMINAMATH_GPT_correct_divisor_l392_39264


namespace NUMINAMATH_GPT_find_f_2018_l392_39289

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom f_zero : f 0 = -1
axiom functional_equation (x : ℝ) : f x = -f (2 - x)

theorem find_f_2018 : f 2018 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_2018_l392_39289


namespace NUMINAMATH_GPT_pencils_purchased_l392_39291

theorem pencils_purchased (n : ℕ) (h1: n ≤ 10) 
  (h2: 2 ≤ 10) 
  (h3: (10 - 2) / 10 * (10 - 2 - 1) / (10 - 1) * (10 - 2 - 2) / (10 - 2) = 0.4666666666666667) :
  n = 3 :=
sorry

end NUMINAMATH_GPT_pencils_purchased_l392_39291


namespace NUMINAMATH_GPT_distance_DE_l392_39249

noncomputable def point := (ℝ × ℝ)

variables (A B C P D E : point)
variables (AB BC AC PC : ℝ)
variables (on_line : point → point → point → Prop)
variables (is_parallel : point → point → point → point → Prop)

axiom AB_length : AB = 13
axiom BC_length : BC = 14
axiom AC_length : AC = 15
axiom PC_length : PC = 10

axiom P_on_AC : on_line A C P
axiom D_on_BP : on_line B P D
axiom E_on_BP : on_line B P E

axiom AD_parallel_BC : is_parallel A D B C
axiom AB_parallel_CE : is_parallel A B C E

theorem distance_DE : ∀ (D E : point), 
  on_line B P D → on_line B P E → 
  is_parallel A D B C → is_parallel A B C E → 
  ∃ dist : ℝ, dist = 12 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_DE_l392_39249


namespace NUMINAMATH_GPT_parabola_vertex_correct_l392_39279

noncomputable def parabola_vertex (p q : ℝ) : ℝ × ℝ :=
  let a := -1
  let b := p
  let c := q
  let x_vertex := -b / (2 * a)
  let y_vertex := a * x_vertex^2 + b * x_vertex + c
  (x_vertex, y_vertex)

theorem parabola_vertex_correct (p q : ℝ) :
  (parabola_vertex 2 24 = (1, 25)) :=
  sorry

end NUMINAMATH_GPT_parabola_vertex_correct_l392_39279


namespace NUMINAMATH_GPT_right_triangle_least_side_l392_39253

theorem right_triangle_least_side (a b : ℕ) (h₁ : a = 8) (h₂ : b = 15) :
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 = c^2 + b^2 ∨ b^2 = c^2 + a^2) ∧ c = Real.sqrt 161 := 
sorry

end NUMINAMATH_GPT_right_triangle_least_side_l392_39253
