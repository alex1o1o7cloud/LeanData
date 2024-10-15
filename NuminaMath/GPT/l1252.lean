import Mathlib

namespace NUMINAMATH_GPT_min_k_spherical_cap_cylinder_l1252_125261

/-- Given a spherical cap and a cylinder sharing a common inscribed sphere with volumes V1 and V2 respectively,
we show that the minimum value of k such that V1 = k * V2 is 4/3. -/
theorem min_k_spherical_cap_cylinder (R : ℝ) (V1 V2 : ℝ) (h1 : V1 = (4/3) * π * R^3) 
(h2 : V2 = 2 * π * R^3) : 
∃ k : ℝ, V1 = k * V2 ∧ k = 4/3 := 
by 
  use (4/3)
  constructor
  . sorry
  . sorry

end NUMINAMATH_GPT_min_k_spherical_cap_cylinder_l1252_125261


namespace NUMINAMATH_GPT_cost_per_mile_l1252_125293

theorem cost_per_mile 
    (round_trip_distance : ℝ)
    (num_days : ℕ)
    (total_cost : ℝ) 
    (h1 : round_trip_distance = 200 * 2)
    (h2 : num_days = 7)
    (h3 : total_cost = 7000) 
  : (total_cost / (round_trip_distance * num_days) = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_cost_per_mile_l1252_125293


namespace NUMINAMATH_GPT_proof_problem_l1252_125263

theorem proof_problem (x : ℝ) (h : x < 1) : -2 * x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1252_125263


namespace NUMINAMATH_GPT_find_a_and_b_maximize_profit_l1252_125222

variable (a b x : ℝ)

-- The given conditions
def condition1 : Prop := 2 * a + b = 120
def condition2 : Prop := 4 * a + 3 * b = 270
def constraint : Prop := 75 ≤ 300 - x

-- The questions translated into a proof problem
theorem find_a_and_b :
  condition1 a b ∧ condition2 a b → a = 45 ∧ b = 30 :=
by
  intros h
  sorry

theorem maximize_profit (a : ℝ) (b : ℝ) (x : ℝ) :
  condition1 a b → condition2 a b → constraint x →
  x = 75 → (300 - x) = 225 → 
  (10 * x + 20 * (300 - x) = 5250) :=
by
  intros h1 h2 hc hx hx1
  sorry

end NUMINAMATH_GPT_find_a_and_b_maximize_profit_l1252_125222


namespace NUMINAMATH_GPT_value_of_k_l1252_125245

theorem value_of_k (k : ℝ) : 
  (∃ x y : ℝ, x = 1/3 ∧ y = -8 ∧ -3/4 - 3 * k * x = 7 * y) → k = 55.25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_k_l1252_125245


namespace NUMINAMATH_GPT_percentage_of_3rd_graders_l1252_125240

theorem percentage_of_3rd_graders (students_jackson students_madison : ℕ)
  (percent_3rd_grade_jackson percent_3rd_grade_madison : ℝ) :
  students_jackson = 200 → percent_3rd_grade_jackson = 25 →
  students_madison = 300 → percent_3rd_grade_madison = 35 →
  ((percent_3rd_grade_jackson / 100 * students_jackson +
    percent_3rd_grade_madison / 100 * students_madison) /
   (students_jackson + students_madison) * 100) = 31 :=
by 
  intros hjackson_percent hmpercent 
    hpercent_jack_percent hpercent_mad_percent
  -- Proof Placeholder
  sorry

end NUMINAMATH_GPT_percentage_of_3rd_graders_l1252_125240


namespace NUMINAMATH_GPT_sqrt_expression_l1252_125290

theorem sqrt_expression : 
  (Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_expression_l1252_125290


namespace NUMINAMATH_GPT_maximum_value_parabola_l1252_125239

theorem maximum_value_parabola (x : ℝ) : 
  ∃ y : ℝ, y = -3 * x^2 + 6 ∧ ∀ z : ℝ, (∃ a : ℝ, z = -3 * a^2 + 6) → z ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_parabola_l1252_125239


namespace NUMINAMATH_GPT_tangent_product_equals_2_pow_23_l1252_125271

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180)) *
  (1 + Real.tan (2 * Real.pi / 180)) *
  (1 + Real.tan (3 * Real.pi / 180)) *
  (1 + Real.tan (4 * Real.pi / 180)) *
  (1 + Real.tan (5 * Real.pi / 180)) *
  (1 + Real.tan (6 * Real.pi / 180)) *
  (1 + Real.tan (7 * Real.pi / 180)) *
  (1 + Real.tan (8 * Real.pi / 180)) *
  (1 + Real.tan (9 * Real.pi / 180)) *
  (1 + Real.tan (10 * Real.pi / 180)) *
  (1 + Real.tan (11 * Real.pi / 180)) *
  (1 + Real.tan (12 * Real.pi / 180)) *
  (1 + Real.tan (13 * Real.pi / 180)) *
  (1 + Real.tan (14 * Real.pi / 180)) *
  (1 + Real.tan (15 * Real.pi / 180)) *
  (1 + Real.tan (16 * Real.pi / 180)) *
  (1 + Real.tan (17 * Real.pi / 180)) *
  (1 + Real.tan (18 * Real.pi / 180)) *
  (1 + Real.tan (19 * Real.pi / 180)) *
  (1 + Real.tan (20 * Real.pi / 180)) *
  (1 + Real.tan (21 * Real.pi / 180)) *
  (1 + Real.tan (22 * Real.pi / 180)) *
  (1 + Real.tan (23 * Real.pi / 180)) *
  (1 + Real.tan (24 * Real.pi / 180)) *
  (1 + Real.tan (25 * Real.pi / 180)) *
  (1 + Real.tan (26 * Real.pi / 180)) *
  (1 + Real.tan (27 * Real.pi / 180)) *
  (1 + Real.tan (28 * Real.pi / 180)) *
  (1 + Real.tan (29 * Real.pi / 180)) *
  (1 + Real.tan (30 * Real.pi / 180)) *
  (1 + Real.tan (31 * Real.pi / 180)) *
  (1 + Real.tan (32 * Real.pi / 180)) *
  (1 + Real.tan (33 * Real.pi / 180)) *
  (1 + Real.tan (34 * Real.pi / 180)) *
  (1 + Real.tan (35 * Real.pi / 180)) *
  (1 + Real.tan (36 * Real.pi / 180)) *
  (1 + Real.tan (37 * Real.pi / 180)) *
  (1 + Real.tan (38 * Real.pi / 180)) *
  (1 + Real.tan (39 * Real.pi / 180)) *
  (1 + Real.tan (40 * Real.pi / 180)) *
  (1 + Real.tan (41 * Real.pi / 180)) *
  (1 + Real.tan (42 * Real.pi / 180)) *
  (1 + Real.tan (43 * Real.pi / 180)) *
  (1 + Real.tan (44 * Real.pi / 180)) *
  (1 + Real.tan (45 * Real.pi / 180))

theorem tangent_product_equals_2_pow_23 : tangent_product = 2 ^ 23 :=
  sorry

end NUMINAMATH_GPT_tangent_product_equals_2_pow_23_l1252_125271


namespace NUMINAMATH_GPT_davids_profit_l1252_125278

-- Definitions of conditions
def weight_of_rice : ℝ := 50
def cost_of_rice : ℝ := 50
def selling_price_per_kg : ℝ := 1.20

-- Theorem stating the expected profit
theorem davids_profit : 
  (selling_price_per_kg * weight_of_rice) - cost_of_rice = 10 := 
by 
  -- Proofs are omitted.
  sorry

end NUMINAMATH_GPT_davids_profit_l1252_125278


namespace NUMINAMATH_GPT_roots_quadratic_eq_sum_prod_l1252_125266

theorem roots_quadratic_eq_sum_prod (r s p q : ℝ) (hr : r + s = p) (hq : r * s = q) : r^2 + s^2 = p^2 - 2 * q :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_eq_sum_prod_l1252_125266


namespace NUMINAMATH_GPT_truncated_cone_sphere_radius_l1252_125243

theorem truncated_cone_sphere_radius :
  ∀ (r1 r2 h : ℝ), 
  r1 = 24 → 
  r2 = 6 → 
  h = 20 → 
  ∃ r, 
  r = 17 * Real.sqrt 2 / 2 := by
  intros r1 r2 h hr1 hr2 hh
  sorry

end NUMINAMATH_GPT_truncated_cone_sphere_radius_l1252_125243


namespace NUMINAMATH_GPT_maximum_xy_l1252_125230

theorem maximum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : xy ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_xy_l1252_125230


namespace NUMINAMATH_GPT_simplify_expression_l1252_125202

def expression1 (x : ℝ) : ℝ :=
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6)

def expression2 (x : ℝ) : ℝ :=
  2 * x^3 + 7 * x^2 - 3 * x + 14

theorem simplify_expression (x : ℝ) : expression1 x = expression2 x :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1252_125202


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1252_125207

variable (m : ℝ)

def P : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0
def Q : Prop := ∀ x : ℝ, 3*x^2 + 4*x + m ≥ 0

theorem sufficient_but_not_necessary : (P m → Q m) ∧ ¬(Q m → P m) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1252_125207


namespace NUMINAMATH_GPT_bill_before_tax_l1252_125219

theorem bill_before_tax (T E : ℝ) (h1 : E = 2) (h2 : 3 * T + 5 * E = 12.70) : 2 * T + 3 * E = 7.80 :=
by
  sorry

end NUMINAMATH_GPT_bill_before_tax_l1252_125219


namespace NUMINAMATH_GPT_original_people_in_room_l1252_125252

theorem original_people_in_room (x : ℕ) (h1 : 18 = (2 * x / 3) - (x / 6)) : x = 36 :=
by sorry

end NUMINAMATH_GPT_original_people_in_room_l1252_125252


namespace NUMINAMATH_GPT_volume_of_inscribed_cube_l1252_125299

theorem volume_of_inscribed_cube (S : ℝ) (π : ℝ) (V : ℝ) (r : ℝ) (s : ℝ) :
    S = 12 * π → 4 * π * r^2 = 12 * π → s = 2 * r → V = s^3 → V = 8 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_inscribed_cube_l1252_125299


namespace NUMINAMATH_GPT_batsman_new_average_l1252_125247

def batsman_average_after_16_innings (A : ℕ) (new_avg : ℕ) (runs_16th : ℕ) : Prop :=
  15 * A + runs_16th = 16 * new_avg

theorem batsman_new_average (A : ℕ) (runs_16th : ℕ) (h1 : batsman_average_after_16_innings A (A + 3) runs_16th) : A + 3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_batsman_new_average_l1252_125247


namespace NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1252_125241

theorem distinct_real_roots_of_quadratic (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x : ℝ, x^2 - 4*x + 2*m = 0 ↔ x = x₁ ∨ x = x₂)) ↔ m < 2 := by
sorry

end NUMINAMATH_GPT_distinct_real_roots_of_quadratic_l1252_125241


namespace NUMINAMATH_GPT_number_of_articles_sold_at_cost_price_l1252_125250

-- Let C be the cost price of one article.
-- Let S be the selling price of one article.
-- Let X be the number of articles sold at cost price.

variables (C S : ℝ) (X : ℕ)

-- Condition 1: The cost price of X articles is equal to the selling price of 32 articles.
axiom condition1 : (X : ℝ) * C = 32 * S

-- Condition 2: The profit is 25%, so the selling price S is 1.25 times the cost price C.
axiom condition2 : S = 1.25 * C

-- The theorem we need to prove
theorem number_of_articles_sold_at_cost_price : X = 40 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_number_of_articles_sold_at_cost_price_l1252_125250


namespace NUMINAMATH_GPT_find_ordered_pairs_l1252_125286

theorem find_ordered_pairs :
  {p : ℝ × ℝ | p.1 > p.2 ∧ (p.1 - p.2 = 2 * p.1 / p.2 ∨ p.1 - p.2 = 2 * p.2 / p.1)} = 
  {(8, 4), (9, 3), (2, 1)} :=
sorry

end NUMINAMATH_GPT_find_ordered_pairs_l1252_125286


namespace NUMINAMATH_GPT_work_completed_in_30_days_l1252_125237

theorem work_completed_in_30_days (ravi_days : ℕ) (prakash_days : ℕ)
  (h1 : ravi_days = 50) (h2 : prakash_days = 75) : 
  let ravi_rate := (1 / 50 : ℚ)
  let prakash_rate := (1 / 75 : ℚ)
  let combined_rate := ravi_rate + prakash_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 30 := by
  sorry

end NUMINAMATH_GPT_work_completed_in_30_days_l1252_125237


namespace NUMINAMATH_GPT_hourly_wage_calculation_l1252_125208

variable (H : ℝ)
variable (hours_per_week : ℝ := 40)
variable (wage_per_widget : ℝ := 0.16)
variable (widgets_per_week : ℝ := 500)
variable (total_earnings : ℝ := 580)

theorem hourly_wage_calculation :
  (hours_per_week * H + widgets_per_week * wage_per_widget = total_earnings) →
  H = 12.5 :=
by
  intro h_equation
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_hourly_wage_calculation_l1252_125208


namespace NUMINAMATH_GPT_diana_hourly_wage_l1252_125246

theorem diana_hourly_wage :
  (∃ (hours_monday : ℕ) (hours_tuesday : ℕ) (hours_wednesday : ℕ) (hours_thursday : ℕ) (hours_friday : ℕ) (weekly_earnings : ℝ),
    hours_monday = 10 ∧
    hours_tuesday = 15 ∧
    hours_wednesday = 10 ∧
    hours_thursday = 15 ∧
    hours_friday = 10 ∧
    weekly_earnings = 1800 ∧
    (weekly_earnings / (hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday) = 30)) :=
sorry

end NUMINAMATH_GPT_diana_hourly_wage_l1252_125246


namespace NUMINAMATH_GPT_price_of_orange_is_60_l1252_125212

-- Given: 
-- 1. The price of each apple is 40 cents.
-- 2. Mary selects 10 pieces of fruit in total.
-- 3. The average price of these 10 pieces is 56 cents.
-- 4. Mary must put back 6 oranges so that the remaining average price is 50 cents.
-- Prove: The price of each orange is 60 cents.

theorem price_of_orange_is_60 (a o : ℕ) (x : ℕ) 
  (h1 : a + o = 10)
  (h2 : 40 * a + x * o = 560)
  (h3 : 40 * a + x * (o - 6) = 200) : 
  x = 60 :=
by
  have eq1 : 40 * a + x * o = 560 := h2
  have eq2 : 40 * a + x * (o - 6) = 200 := h3
  sorry

end NUMINAMATH_GPT_price_of_orange_is_60_l1252_125212


namespace NUMINAMATH_GPT_max_choir_members_l1252_125274

theorem max_choir_members (n : ℕ) (x y : ℕ) : 
  n = x^2 + 11 ∧ n = y * (y + 3) → n = 54 :=
by
  sorry

end NUMINAMATH_GPT_max_choir_members_l1252_125274


namespace NUMINAMATH_GPT_relatively_prime_27x_plus_4_18x_plus_3_l1252_125229

theorem relatively_prime_27x_plus_4_18x_plus_3 (x : ℕ) :
  Nat.gcd (27 * x + 4) (18 * x + 3) = 1 :=
sorry

end NUMINAMATH_GPT_relatively_prime_27x_plus_4_18x_plus_3_l1252_125229


namespace NUMINAMATH_GPT_number_of_schools_in_pythagoras_city_l1252_125218

theorem number_of_schools_in_pythagoras_city (n : ℕ) (h1 : true) 
    (h2 : true) (h3 : ∃ m, m = (3 * n + 1) / 2)
    (h4 : true) (h5 : true) : n = 24 :=
by 
  have h6 : 69 < 3 * n := sorry
  have h7 : 3 * n < 79 := sorry
  sorry

end NUMINAMATH_GPT_number_of_schools_in_pythagoras_city_l1252_125218


namespace NUMINAMATH_GPT_Jerry_wants_to_raise_average_l1252_125225

theorem Jerry_wants_to_raise_average 
  (first_three_tests_avg : ℕ) (fourth_test_score : ℕ) (desired_increase : ℕ) 
  (h1 : first_three_tests_avg = 90) (h2 : fourth_test_score = 98) 
  : desired_increase = 2 := 
by
  sorry

end NUMINAMATH_GPT_Jerry_wants_to_raise_average_l1252_125225


namespace NUMINAMATH_GPT_angle_between_strips_l1252_125292

theorem angle_between_strips (w : ℝ) (a : ℝ) (angle : ℝ) (h_w : w = 1) (h_area : a = 2) :
  ∃ θ : ℝ, θ = 30 ∧ angle = θ :=
by
  sorry

end NUMINAMATH_GPT_angle_between_strips_l1252_125292


namespace NUMINAMATH_GPT_product_of_three_numbers_l1252_125221

theorem product_of_three_numbers :
  ∃ (x y z : ℚ), 
    (x + y + z = 30) ∧ 
    (x = 3 * (y + z)) ∧ 
    (y = 8 * z) ∧ 
    (x * y * z = 125) := 
by
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l1252_125221


namespace NUMINAMATH_GPT_factorize_expression_l1252_125256

variable (x y : ℝ)

theorem factorize_expression : (x - y) ^ 2 + 2 * y * (x - y) = (x - y) * (x + y) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1252_125256


namespace NUMINAMATH_GPT_greatest_possible_length_l1252_125264

-- Define the lengths of the ropes
def rope_lengths : List ℕ := [72, 48, 120, 96]

-- Define the gcd function to find the greatest common divisor of a list of numbers
def list_gcd (l : List ℕ) : ℕ :=
  l.foldr Nat.gcd 0

-- Define the target problem statement
theorem greatest_possible_length 
  (h : list_gcd rope_lengths = 24) : 
  ∀ length ∈ rope_lengths, length % 24 = 0 :=
by
  intros length h_length
  sorry

end NUMINAMATH_GPT_greatest_possible_length_l1252_125264


namespace NUMINAMATH_GPT_third_part_of_156_division_proof_l1252_125217

theorem third_part_of_156_division_proof :
  ∃ (x : ℚ), (2 * x + (1 / 2) * x + (1 / 4) * x + (1 / 8) * x = 156) ∧ ((1 / 4) * x = 13 + 15 / 23) :=
by
  sorry

end NUMINAMATH_GPT_third_part_of_156_division_proof_l1252_125217


namespace NUMINAMATH_GPT_functional_relationship_maximum_profit_desired_profit_l1252_125282

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end NUMINAMATH_GPT_functional_relationship_maximum_profit_desired_profit_l1252_125282


namespace NUMINAMATH_GPT_equivalent_spherical_coords_l1252_125258

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_spherical_coords_l1252_125258


namespace NUMINAMATH_GPT_sequence_ninth_term_l1252_125283

theorem sequence_ninth_term (a b : ℚ) :
  ∀ n : ℕ, n = 9 → (-1 : ℚ) ^ n * (n * b ^ n) / ((n + 1) * a ^ (n + 2)) = -9 * b^9 / (10 * a^11) :=
by
  sorry

end NUMINAMATH_GPT_sequence_ninth_term_l1252_125283


namespace NUMINAMATH_GPT_sequence_polynomial_degree_l1252_125284

theorem sequence_polynomial_degree
  (k : ℕ)
  (v : ℕ → ℤ)
  (u : ℕ → ℤ)
  (h_diff_poly : ∃ p : Polynomial ℤ, ∀ n, v n = Polynomial.eval (n : ℤ) p)
  (h_diff_seq : ∀ n, v n = (u (n + 1) - u n)) :
  ∃ q : Polynomial ℤ, ∀ n, u n = Polynomial.eval (n : ℤ) q := 
sorry

end NUMINAMATH_GPT_sequence_polynomial_degree_l1252_125284


namespace NUMINAMATH_GPT_locus_of_P_coordinates_of_P_l1252_125259

-- Define the points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)

-- Define the line l : 4x + 3y - 2 = 0
def l (x y: ℝ) := 4 * x + 3 * y - 2 = 0

-- Problem (1): Equation of the locus of point P such that |PA| = |PB|
theorem locus_of_P (P : ℝ × ℝ) :
  (∃ P, dist P A = dist P B) ↔ (∀ x y : ℝ, P = (x, y) → x - y - 5 = 0) :=
sorry

-- Problem (2): Coordinates of P such that |PA| = |PB| and the distance from P to line l is 2
theorem coordinates_of_P (a b : ℝ):
  (dist (a, b) A = dist (a, b) B ∧ abs (4 * a + 3 * b - 2) / 5 = 2) ↔
  ((a = 1 ∧ b = -4) ∨ (a = 27 / 7 ∧ b = -8 / 7)) :=
sorry

end NUMINAMATH_GPT_locus_of_P_coordinates_of_P_l1252_125259


namespace NUMINAMATH_GPT_population_in_2060_l1252_125276

noncomputable def population (year : ℕ) : ℕ :=
  if h : (year - 2000) % 20 = 0 then
    250 * 2 ^ ((year - 2000) / 20)
  else
    0 -- This handles non-multiples of 20 cases, which are irrelevant here

theorem population_in_2060 : population 2060 = 2000 := by
  sorry

end NUMINAMATH_GPT_population_in_2060_l1252_125276


namespace NUMINAMATH_GPT_pizza_slices_with_all_three_toppings_l1252_125273

theorem pizza_slices_with_all_three_toppings : 
  ∀ (a b c d e f g : ℕ), 
  a + b + c + d + e + f + g = 24 ∧ 
  a + d + e + g = 12 ∧ 
  b + d + f + g = 15 ∧ 
  c + e + f + g = 10 → 
  g = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_pizza_slices_with_all_three_toppings_l1252_125273


namespace NUMINAMATH_GPT_ab_plus_cd_eq_neg_346_over_9_l1252_125232

theorem ab_plus_cd_eq_neg_346_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := 
sorry

end NUMINAMATH_GPT_ab_plus_cd_eq_neg_346_over_9_l1252_125232


namespace NUMINAMATH_GPT_max_value_expression_l1252_125226

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l1252_125226


namespace NUMINAMATH_GPT_spring_mass_relationship_l1252_125211

theorem spring_mass_relationship (x y : ℕ) (h1 : y = 18 + 2 * x) : 
  y = 32 → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_spring_mass_relationship_l1252_125211


namespace NUMINAMATH_GPT_sidney_thursday_jacks_l1252_125277

open Nat

-- Define the number of jumping jacks Sidney did on each day
def monday_jacks := 20
def tuesday_jacks := 36
def wednesday_jacks := 40

-- Define the total number of jumping jacks done by Sidney
-- on Monday, Tuesday, and Wednesday
def sidney_mon_wed_jacks := monday_jacks + tuesday_jacks + wednesday_jacks

-- Define the total number of jumping jacks done by Brooke
def brooke_jacks := 438

-- Define the relationship between Brooke's and Sidney's total jumping jacks
def sidney_total_jacks := brooke_jacks / 3

-- Prove the number of jumping jacks Sidney did on Thursday
theorem sidney_thursday_jacks :
  sidney_total_jacks - sidney_mon_wed_jacks = 50 :=
by
  sorry

end NUMINAMATH_GPT_sidney_thursday_jacks_l1252_125277


namespace NUMINAMATH_GPT_age_of_b_l1252_125242

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 72) : b = 28 :=
by
  sorry

end NUMINAMATH_GPT_age_of_b_l1252_125242


namespace NUMINAMATH_GPT_salty_cookies_initial_at_least_34_l1252_125268

variable {S : ℕ}  -- S will represent the initial number of salty cookies

-- Conditions from the problem
def sweet_cookies_initial := 8
def sweet_cookies_ate := 20
def salty_cookies_ate := 34
def more_salty_than_sweet := 14

theorem salty_cookies_initial_at_least_34 :
  8 = sweet_cookies_initial ∧
  20 = sweet_cookies_ate ∧
  34 = salty_cookies_ate ∧
  salty_cookies_ate = sweet_cookies_ate + more_salty_than_sweet
  → S ≥ 34 :=
by sorry

end NUMINAMATH_GPT_salty_cookies_initial_at_least_34_l1252_125268


namespace NUMINAMATH_GPT_geometric_sequence_solution_l1252_125228

-- Define the geometric sequence a_n with a common ratio q and first term a_1
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 * q^n

-- Given conditions in the problem
variables {a : ℕ → ℝ} {q a1 : ℝ}

-- Common ratio is greater than 1
axiom ratio_gt_one : q > 1

-- Given conditions a_3a_7 = 72 and a_2 + a_8 = 27
axiom condition1 : a 3 * a 7 = 72
axiom condition2 : a 2 + a 8 = 27

-- Defining the property that we are looking to prove a_12 = 96
theorem geometric_sequence_solution :
  geometric_sequence a a1 q →
  a 12 = 96 :=
by
  -- This part of the proof would be filled in
  -- Show the conditions and relations leading to the solution a_12 = 96
  sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l1252_125228


namespace NUMINAMATH_GPT_gcd_a_b_eq_one_l1252_125248

def a : ℕ := 47^5 + 1
def b : ℕ := 47^5 + 47^3 + 1

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_a_b_eq_one_l1252_125248


namespace NUMINAMATH_GPT_smaller_angle_at_9_am_l1252_125254

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end NUMINAMATH_GPT_smaller_angle_at_9_am_l1252_125254


namespace NUMINAMATH_GPT_smallest_sum_B_d_l1252_125206

theorem smallest_sum_B_d :
  ∃ B d : ℕ, (B < 5) ∧ (d > 6) ∧ (125 * B + 25 * B + B = 4 * d + 4) ∧ (B + d = 77) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_B_d_l1252_125206


namespace NUMINAMATH_GPT_jasmine_money_left_l1252_125231

theorem jasmine_money_left 
  (initial_amount : ℝ)
  (apple_cost : ℝ) (num_apples : ℕ)
  (orange_cost : ℝ) (num_oranges : ℕ)
  (pear_cost : ℝ) (num_pears : ℕ)
  (h_initial : initial_amount = 100.00)
  (h_apple_cost : apple_cost = 1.50)
  (h_num_apples : num_apples = 5)
  (h_orange_cost : orange_cost = 2.00)
  (h_num_oranges : num_oranges = 10)
  (h_pear_cost : pear_cost = 2.25)
  (h_num_pears : num_pears = 4) : 
  initial_amount - (num_apples * apple_cost + num_oranges * orange_cost + num_pears * pear_cost) = 63.50 := 
by 
  sorry

end NUMINAMATH_GPT_jasmine_money_left_l1252_125231


namespace NUMINAMATH_GPT_sample_size_calculation_l1252_125296

theorem sample_size_calculation (n : ℕ) (ratio_A_B_C q_A q_B q_C : ℕ) 
  (ratio_condition : ratio_A_B_C = 2 ∧ ratio_A_B_C * q_A = 2 ∧ ratio_A_B_C * q_B = 3 ∧ ratio_A_B_C * q_C = 5)
  (sample_A_units : q_A = 16) : n = 80 :=
sorry

end NUMINAMATH_GPT_sample_size_calculation_l1252_125296


namespace NUMINAMATH_GPT_polygon_sides_l1252_125216

theorem polygon_sides (side_length perimeter : ℕ) (h1 : side_length = 4) (h2 : perimeter = 24) : 
  perimeter / side_length = 6 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l1252_125216


namespace NUMINAMATH_GPT_variable_swap_l1252_125270

theorem variable_swap (x y t : Nat) (h1 : x = 5) (h2 : y = 6) (h3 : t = x) (h4 : x = y) (h5 : y = t) : 
  x = 6 ∧ y = 5 := 
by
  sorry

end NUMINAMATH_GPT_variable_swap_l1252_125270


namespace NUMINAMATH_GPT_set_intersection_l1252_125288

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def complement (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem set_intersection (hU : U = univ)
                         (hA : A = {x : ℝ | x > 0})
                         (hB : B = {x : ℝ | x > 1}) :
  A ∩ complement B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l1252_125288


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1252_125298

noncomputable def angle_sum : ℝ := 120 -- $\frac{4}{3}$ of 90 degrees
noncomputable def angle_difference : ℝ := 20

theorem largest_angle_in_triangle :
  ∃ (a b c : ℝ), a + b + c = 180 ∧ a + b = angle_sum ∧ b = a + angle_difference ∧
  max a (max b c) = 70 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1252_125298


namespace NUMINAMATH_GPT_triangle_ratio_and_angle_l1252_125289

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinA sinB sinC : ℝ)

theorem triangle_ratio_and_angle
  (h_triangle : a / sinA = b / sinB ∧ b / sinB = c / sinC)
  (h_sin_ratio : sinA / sinB = 5 / 7 ∧ sinB / sinC = 7 / 8) :
  (a / b = 5 / 7 ∧ b / c = 7 / 8) ∧ B = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ratio_and_angle_l1252_125289


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1252_125203

theorem factorize_difference_of_squares (x : ℝ) : 9 - 4*x^2 = (3 - 2*x) * (3 + 2*x) :=
by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1252_125203


namespace NUMINAMATH_GPT_average_speed_of_bike_l1252_125209

theorem average_speed_of_bike (distance : ℕ) (time : ℕ) (h1 : distance = 21) (h2 : time = 7) : distance / time = 3 := by
  sorry

end NUMINAMATH_GPT_average_speed_of_bike_l1252_125209


namespace NUMINAMATH_GPT_pages_called_this_week_l1252_125260

-- Definitions as per conditions
def pages_called_last_week := 10.2
def total_pages_called := 18.8

-- Theorem to prove the solution
theorem pages_called_this_week :
  total_pages_called - pages_called_last_week = 8.6 :=
by
  sorry

end NUMINAMATH_GPT_pages_called_this_week_l1252_125260


namespace NUMINAMATH_GPT_general_term_formula_l1252_125233

def seq (n : ℕ) : ℤ :=
match n with
| 0       => 1
| 1       => -3
| 2       => 5
| 3       => -7
| 4       => 9
| (n + 1) => (-1)^(n+1) * (2*n + 1) -- extends indefinitely for general natural number

theorem general_term_formula (n : ℕ) : 
  seq n = (-1)^(n+1) * (2*n-1) :=
sorry

end NUMINAMATH_GPT_general_term_formula_l1252_125233


namespace NUMINAMATH_GPT_tank_fill_time_l1252_125200

noncomputable def fill_time (T rA rB rC : ℝ) : ℝ :=
  let cycle_fill := rA + rB + rC
  let cycles := T / cycle_fill
  let cycle_time := 3
  cycles * cycle_time

theorem tank_fill_time
  (T : ℝ) (rA rB rC : ℝ) (hT : T = 800) (hrA : rA = 40) (hrB : rB = 30) (hrC : rC = -20) :
  fill_time T rA rB rC = 48 :=
by
  sorry

end NUMINAMATH_GPT_tank_fill_time_l1252_125200


namespace NUMINAMATH_GPT_part1_part2_l1252_125227

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def calculate_P (x y : ℤ) : ℤ := 
  (x - y) / 9

def y_from_x (x : ℤ) : ℤ :=
  let first_three := x / 10
  let last_digit := x % 10
  last_digit * 1000 + first_three

def calculate_s (a b : ℕ) : ℤ :=
  1100 + 20 * a + b

def calculate_t (a b : ℕ) : ℤ :=
  b * 1000 + a * 100 + 23

theorem part1 : calculate_P 5324 (y_from_x 5324) = 88 := by
  sorry

theorem part2 :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 9 ∧
  let s := calculate_s a b
  let t := calculate_t a b
  let P_s := calculate_P s (y_from_x s)
  let P_t := calculate_P t (y_from_x t)
  let difference := P_t - P_s - a - b
  is_perfect_square difference ∧ P_t = -161 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1252_125227


namespace NUMINAMATH_GPT_connor_sleep_duration_l1252_125223

variables {Connor_sleep Luke_sleep Puppy_sleep : ℕ}

def sleeps_two_hours_longer (Luke_sleep Connor_sleep : ℕ) : Prop :=
  Luke_sleep = Connor_sleep + 2

def sleeps_twice_as_long (Puppy_sleep Luke_sleep : ℕ) : Prop :=
  Puppy_sleep = 2 * Luke_sleep

def sleeps_sixteen_hours (Puppy_sleep : ℕ) : Prop :=
  Puppy_sleep = 16

theorem connor_sleep_duration 
  (h1 : sleeps_two_hours_longer Luke_sleep Connor_sleep)
  (h2 : sleeps_twice_as_long Puppy_sleep Luke_sleep)
  (h3 : sleeps_sixteen_hours Puppy_sleep) :
  Connor_sleep = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_connor_sleep_duration_l1252_125223


namespace NUMINAMATH_GPT_diane_head_start_l1252_125294

theorem diane_head_start (x : ℝ) :
  (100 - 11.91) / (88.09 + x) = 99.25 / 100 ->
  abs (x - 12.68) < 0.01 := 
by
  sorry

end NUMINAMATH_GPT_diane_head_start_l1252_125294


namespace NUMINAMATH_GPT_M_infinite_l1252_125267

open Nat

-- Define the set M
def M : Set ℕ := {k | ∃ n : ℕ, 3 ^ n % n = k % n}

-- Statement of the problem
theorem M_infinite : Set.Infinite M :=
sorry

end NUMINAMATH_GPT_M_infinite_l1252_125267


namespace NUMINAMATH_GPT_range_of_x_l1252_125272

open Set

noncomputable def M (x : ℝ) : Set ℝ := {x^2, 1}

theorem range_of_x (x : ℝ) (hx : M x) : x ≠ 1 ∧ x ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1252_125272


namespace NUMINAMATH_GPT_starters_choice_l1252_125269

/-- There are 18 players including a set of quadruplets: Bob, Bill, Ben, and Bert. -/
def total_players : ℕ := 18

/-- The set of quadruplets: Bob, Bill, Ben, and Bert. -/
def quadruplets : Finset (String) := {"Bob", "Bill", "Ben", "Bert"}

/-- We need to choose 7 starters, exactly 3 of which are from the set of quadruplets. -/
def ways_to_choose_starters : ℕ :=
  let quadruplet_combinations := Nat.choose 4 3
  let remaining_spots := 4
  let remaining_players := total_players - 4
  quadruplet_combinations * Nat.choose remaining_players remaining_spots

theorem starters_choice (h1 : total_players = 18)
                        (h2 : quadruplets.card = 4) :
  ways_to_choose_starters = 4004 :=
by 
  -- conditional setups here
  sorry

end NUMINAMATH_GPT_starters_choice_l1252_125269


namespace NUMINAMATH_GPT_arrange_numbers_l1252_125213

namespace MathProofs

theorem arrange_numbers (a b : ℚ) (h1 : a > 0) (h2 : b < 0) (h3 : a + b < 0) :
  b < -a ∧ -a < a ∧ a < -b :=
by
  -- Proof to be completed
  sorry

end MathProofs

end NUMINAMATH_GPT_arrange_numbers_l1252_125213


namespace NUMINAMATH_GPT_find_marks_of_a_l1252_125205

theorem find_marks_of_a (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : (A + B + C + D) / 4 = 47)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 48) : 
  A = 43 :=
by
  sorry

end NUMINAMATH_GPT_find_marks_of_a_l1252_125205


namespace NUMINAMATH_GPT_gift_certificate_value_is_correct_l1252_125236

-- Define the conditions
def total_race_time_minutes : ℕ := 12
def one_lap_meters : ℕ := 100
def total_laps : ℕ := 24
def earning_rate_per_minute : ℕ := 7

-- The total distance run in meters
def total_distance_meters : ℕ := total_laps * one_lap_meters

-- The total earnings in dollars
def total_earnings_dollars : ℕ := earning_rate_per_minute * total_race_time_minutes

-- The worth of the gift certificate per 100 meters (to be proven as 3.50 dollars)
def gift_certificate_value : ℚ := total_earnings_dollars / (total_distance_meters / one_lap_meters)

-- Prove that the gift certificate value is $3.50
theorem gift_certificate_value_is_correct : 
    gift_certificate_value = 3.5 := by
  sorry

end NUMINAMATH_GPT_gift_certificate_value_is_correct_l1252_125236


namespace NUMINAMATH_GPT_minimum_tickets_needed_l1252_125220

noncomputable def min_tickets {α : Type*} (winning_permutation : Fin 50 → α) (tickets : List (Fin 50 → α)) : ℕ :=
  List.length tickets

theorem minimum_tickets_needed
  (winning_permutation : Fin 50 → ℕ)
  (tickets : List (Fin 50 → ℕ))
  (h_tickets_valid : ∀ t ∈ tickets, Function.Surjective t)
  (h_at_least_one_match : ∀ winning_permutation : Fin 50 → ℕ,
      ∃ t ∈ tickets, ∃ i : Fin 50, t i = winning_permutation i) : 
  min_tickets winning_permutation tickets ≥ 26 :=
sorry

end NUMINAMATH_GPT_minimum_tickets_needed_l1252_125220


namespace NUMINAMATH_GPT_value_of_M_l1252_125253

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 1200) : M = 1680 := 
sorry

end NUMINAMATH_GPT_value_of_M_l1252_125253


namespace NUMINAMATH_GPT_race_probability_l1252_125204

theorem race_probability (Px : ℝ) (Py : ℝ) (Pz : ℝ) 
  (h1 : Px = 1 / 6) 
  (h2 : Pz = 1 / 8) 
  (h3 : Px + Py + Pz = 0.39166666666666666) : Py = 0.1 := 
sorry

end NUMINAMATH_GPT_race_probability_l1252_125204


namespace NUMINAMATH_GPT_candy_necklaces_l1252_125214

theorem candy_necklaces (friends : ℕ) (candies_per_necklace : ℕ) (candies_per_block : ℕ)(blocks_needed : ℕ):
  friends = 8 →
  candies_per_necklace = 10 →
  candies_per_block = 30 →
  80 / 30 > 2.67 →
  blocks_needed = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_candy_necklaces_l1252_125214


namespace NUMINAMATH_GPT_smallest_x_mod_equation_l1252_125244

theorem smallest_x_mod_equation : ∃ x : ℕ, 42 * x + 10 ≡ 5 [MOD 15] ∧ ∀ y : ℕ, 42 * y + 10 ≡ 5 [MOD 15] → x ≤ y :=
by
sorry

end NUMINAMATH_GPT_smallest_x_mod_equation_l1252_125244


namespace NUMINAMATH_GPT_triangle_angle_bisector_proportion_l1252_125201

theorem triangle_angle_bisector_proportion
  (a b c x y : ℝ)
  (h : x / c = y / a)
  (h2 : x + y = b) :
  x / c = b / (a + c) :=
sorry

end NUMINAMATH_GPT_triangle_angle_bisector_proportion_l1252_125201


namespace NUMINAMATH_GPT_vector_addition_l1252_125265

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

-- State the theorem to prove 2a + b = (7, -3, 2)
theorem vector_addition : (2 • a + b) = (7, -3, 2) := by
  sorry

end NUMINAMATH_GPT_vector_addition_l1252_125265


namespace NUMINAMATH_GPT_find_x_given_y_l1252_125249

noncomputable def constantRatio : Prop :=
  ∃ k : ℚ, ∀ x y : ℚ, (5 * x - 6) / (2 * y + 10) = k

theorem find_x_given_y :
  (constantRatio ∧ (3, 2) ∈ {(x, y) | (5 * x - 6) / (2 * y + 10) = 9 / 14}) →
  ∃ x : ℚ, ((5 * x - 6) / 20 = 9 / 14 ∧ x = 53 / 14) :=
by
  sorry

end NUMINAMATH_GPT_find_x_given_y_l1252_125249


namespace NUMINAMATH_GPT_cards_exchanged_l1252_125279

theorem cards_exchanged (x : ℕ) (h : x * (x - 1) = 1980) : x * (x - 1) = 1980 :=
by sorry

end NUMINAMATH_GPT_cards_exchanged_l1252_125279


namespace NUMINAMATH_GPT_jelly_bean_probability_l1252_125251

variable (P_red P_orange P_yellow P_green : ℝ)

theorem jelly_bean_probability :
  P_red = 0.15 ∧ P_orange = 0.35 ∧ (P_red + P_orange + P_yellow + P_green = 1) →
  (P_yellow + P_green = 0.5) :=
by
  intro h
  obtain ⟨h_red, h_orange, h_total⟩ := h
  sorry

end NUMINAMATH_GPT_jelly_bean_probability_l1252_125251


namespace NUMINAMATH_GPT_division_quotient_remainder_l1252_125275

theorem division_quotient_remainder :
  ∃ (q r : ℝ), 76.6 = 1.8 * q + r ∧ 0 ≤ r ∧ r < 1.8 ∧ q = 42 ∧ r = 1 := by
  sorry

end NUMINAMATH_GPT_division_quotient_remainder_l1252_125275


namespace NUMINAMATH_GPT_eleven_y_minus_x_l1252_125281

theorem eleven_y_minus_x (x y : ℤ) (h1 : x = 7 * y + 3) (h2 : 2 * x = 18 * y + 2) : 11 * y - x = 1 := by
  sorry

end NUMINAMATH_GPT_eleven_y_minus_x_l1252_125281


namespace NUMINAMATH_GPT_unique_solution_l1252_125280

theorem unique_solution (a b x: ℝ) : 
  (4 * x - 7 + a = (b - 1) * x + 2) ↔ (b ≠ 5) := 
by
  sorry -- proof is omitted as per instructions

end NUMINAMATH_GPT_unique_solution_l1252_125280


namespace NUMINAMATH_GPT_ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l1252_125285

theorem ones_digit_largest_power_of_2_divides_32_factorial : 
  (2^31 % 10) = 8 := 
by
  sorry

theorem ones_digit_largest_power_of_3_divides_32_factorial : 
  (3^14 % 10) = 9 := 
by
  sorry

end NUMINAMATH_GPT_ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l1252_125285


namespace NUMINAMATH_GPT_find_smaller_number_l1252_125210

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : x = 21 := 
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1252_125210


namespace NUMINAMATH_GPT_range_of_a_l1252_125291

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) < 0) ∧
  (∀ x : ℝ, x > 6 → (1/3 * x^3 - (a/2) * x^2 + (a-1) * x + 1) > 0)
  ↔ (5 < a ∧ a < 7) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1252_125291


namespace NUMINAMATH_GPT_omar_remaining_coffee_l1252_125297

noncomputable def remaining_coffee : ℝ := 
  let initial_coffee := 12
  let after_first_drink := initial_coffee - (initial_coffee * 1/4)
  let after_office_drink := after_first_drink - (after_first_drink * 1/3)
  let espresso_in_ounces := 75 / 29.57
  let after_espresso := after_office_drink + espresso_in_ounces
  let after_lunch_drink := after_espresso - (after_espresso * 0.75)
  let iced_tea_addition := 4 * 1/2
  let after_iced_tea := after_lunch_drink + iced_tea_addition
  let after_cold_drink := after_iced_tea - (after_iced_tea * 0.6)
  after_cold_drink

theorem omar_remaining_coffee : remaining_coffee = 1.654 :=
by 
  sorry

end NUMINAMATH_GPT_omar_remaining_coffee_l1252_125297


namespace NUMINAMATH_GPT_white_line_longer_l1252_125224

theorem white_line_longer :
  let white_line := 7.67
  let blue_line := 3.33
  white_line - blue_line = 4.34 := by
  sorry

end NUMINAMATH_GPT_white_line_longer_l1252_125224


namespace NUMINAMATH_GPT_started_with_l1252_125295

-- Define the conditions
def total_eggs : ℕ := 70
def bought_eggs : ℕ := 62

-- Define the statement to prove
theorem started_with (initial_eggs : ℕ) : initial_eggs = total_eggs - bought_eggs → initial_eggs = 8 := by
  intro h
  sorry

end NUMINAMATH_GPT_started_with_l1252_125295


namespace NUMINAMATH_GPT_how_many_trucks_l1252_125257

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end NUMINAMATH_GPT_how_many_trucks_l1252_125257


namespace NUMINAMATH_GPT_find_rate_of_interest_l1252_125215

-- Definitions based on conditions
def Principal : ℝ := 7200
def SimpleInterest : ℝ := 3150
def Time : ℝ := 2.5
def RatePerAnnum (R : ℝ) : Prop := SimpleInterest = (Principal * R * Time) / 100

-- Theorem statement
theorem find_rate_of_interest (R : ℝ) (h : RatePerAnnum R) : R = 17.5 :=
by { sorry }

end NUMINAMATH_GPT_find_rate_of_interest_l1252_125215


namespace NUMINAMATH_GPT_trig_expression_value_l1252_125234

theorem trig_expression_value (α : Real) (h : Real.tan (3 * Real.pi + α) = 3) :
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) /
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l1252_125234


namespace NUMINAMATH_GPT_heaviest_and_lightest_in_13_weighings_l1252_125238

/-- Given ten coins of different weights and a balance scale.
    Prove that it is possible to identify the heaviest and the lightest coin
    within 13 weighings. -/
theorem heaviest_and_lightest_in_13_weighings
  (coins : Fin 10 → ℝ)
  (h_different: ∀ i j : Fin 10, i ≠ j → coins i ≠ coins j)
  : ∃ (heaviest lightest : Fin 10),
      (heaviest ≠ lightest) ∧
      (∀ i : Fin 10, coins i ≤ coins heaviest) ∧
      (∀ i : Fin 10, coins lightest ≤ coins i) :=
sorry

end NUMINAMATH_GPT_heaviest_and_lightest_in_13_weighings_l1252_125238


namespace NUMINAMATH_GPT_glasses_in_smaller_box_l1252_125287

variable (x : ℕ)

theorem glasses_in_smaller_box (h : (x + 16) / 2 = 15) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_glasses_in_smaller_box_l1252_125287


namespace NUMINAMATH_GPT_minimum_filtration_process_l1252_125235

noncomputable def filtration_process (n : ℕ) : Prop :=
  (0.8 : ℝ) ^ n < 0.05

theorem minimum_filtration_process : ∃ n : ℕ, filtration_process n ∧ n ≥ 14 := 
  sorry

end NUMINAMATH_GPT_minimum_filtration_process_l1252_125235


namespace NUMINAMATH_GPT_small_seat_capacity_l1252_125255

-- Definitions for the conditions
def smallSeats : Nat := 2
def largeSeats : Nat := 23
def capacityLargeSeat : Nat := 54
def totalPeopleSmallSeats : Nat := 28

-- Theorem statement
theorem small_seat_capacity : totalPeopleSmallSeats / smallSeats = 14 := by
  sorry

end NUMINAMATH_GPT_small_seat_capacity_l1252_125255


namespace NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l1252_125262

theorem subcommittees_with_at_least_one_teacher :
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  total_subcommittees - non_teacher_subcommittees = 460 :=
by
  -- Definitions and conditions based on the problem statement
  let n := 12
  let t := 5
  let k := 4
  let total_subcommittees := Nat.choose n k
  let non_teacher_subcommittees := Nat.choose (n - t) k
  sorry -- Proof goes here

end NUMINAMATH_GPT_subcommittees_with_at_least_one_teacher_l1252_125262
