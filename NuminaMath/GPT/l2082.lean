import Mathlib

namespace alice_age_proof_l2082_208217

-- Definitions derived from the conditions
def alice_pens : ℕ := 60
def clara_pens : ℕ := (2 * alice_pens) / 5
def clara_age_in_5_years : ℕ := 61
def clara_current_age : ℕ := clara_age_in_5_years - 5
def age_difference : ℕ := alice_pens - clara_pens

-- Proof statement to be proved
theorem alice_age_proof : (clara_current_age - age_difference = 20) :=
sorry

end alice_age_proof_l2082_208217


namespace product_of_x_y_l2082_208220

variable (x y : ℝ)

-- Condition: EF = GH
def EF_eq_GH := (x^2 + 2 * x - 8 = 45)

-- Condition: FG = EH
def FG_eq_EH := (y^2 + 8 * y + 16 = 36)

-- Condition: y > 0
def y_pos := (y > 0)

theorem product_of_x_y : EF_eq_GH x ∧ FG_eq_EH y ∧ y_pos y → 
  x * y = -2 + 6 * Real.sqrt 6 :=
sorry

end product_of_x_y_l2082_208220


namespace possible_value_of_n_l2082_208245

theorem possible_value_of_n :
  ∃ (n : ℕ), (345564 - n) % (13 * 17 * 19) = 0 ∧ 0 < n ∧ n < 1000 ∧ n = 98 :=
sorry

end possible_value_of_n_l2082_208245


namespace solution_set_empty_iff_l2082_208230

def quadratic_no_solution (a b c : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)

theorem solution_set_empty_iff (a b c : ℝ) (h : quadratic_no_solution a b c) : a > 0 ∧ (b^2 - 4 * a * c ≤ 0) :=
sorry

end solution_set_empty_iff_l2082_208230


namespace min_sales_required_l2082_208282

-- Definitions from conditions
def old_salary : ℝ := 75000
def new_base_salary : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750

-- Statement to be proven
theorem min_sales_required (n : ℕ) :
  n ≥ ⌈(old_salary - new_base_salary) / (commission_rate * sale_amount)⌉₊ :=
sorry

end min_sales_required_l2082_208282


namespace old_edition_pages_l2082_208254

theorem old_edition_pages (x : ℕ) 
  (h₁ : 2 * x - 230 = 450) : x = 340 := 
by sorry

end old_edition_pages_l2082_208254


namespace maximal_sector_angle_l2082_208221

theorem maximal_sector_angle (a : ℝ) (r : ℝ) (l : ℝ) (α : ℝ)
  (h1 : l + 2 * r = a)
  (h2 : 0 < r ∧ r < a / 2)
  (h3 : α = l / r)
  (eval_area : ∀ (l r : ℝ), S = 1 / 2 * l * r)
  (S : ℝ) :
  α = 2 := sorry

end maximal_sector_angle_l2082_208221


namespace sum_of_remainders_mod_53_l2082_208204

theorem sum_of_remainders_mod_53 (d e f : ℕ) (hd : d % 53 = 19) (he : e % 53 = 33) (hf : f % 53 = 14) : 
  (d + e + f) % 53 = 13 :=
by
  sorry

end sum_of_remainders_mod_53_l2082_208204


namespace units_digit_p_plus_one_l2082_208246

theorem units_digit_p_plus_one (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 ≠ 0)
  (h3 : (p ^ 3) % 10 = (p ^ 2) % 10) : (p + 1) % 10 = 7 :=
  sorry

end units_digit_p_plus_one_l2082_208246


namespace binom_eight_five_l2082_208275

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l2082_208275


namespace find_expression_l2082_208272

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l2082_208272


namespace fraction_sum_l2082_208248

theorem fraction_sum : (1/4 : ℚ) + (3/9 : ℚ) = (7/12 : ℚ) := 
  by 
  sorry

end fraction_sum_l2082_208248


namespace algebraic_expression_value_l2082_208226

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 1) :
  (2 * x + 4 * y) / (x^2 + 4 * x * y + 4 * y^2) = 2 :=
by
  sorry

end algebraic_expression_value_l2082_208226


namespace cost_per_toy_initially_l2082_208267

-- defining conditions
def num_toys : ℕ := 200
def percent_sold : ℝ := 0.8
def price_per_toy : ℝ := 30
def profit : ℝ := 800

-- defining the problem
theorem cost_per_toy_initially :
  ((num_toys * percent_sold) * price_per_toy - profit) / (num_toys * percent_sold) = 25 :=
by
  sorry

end cost_per_toy_initially_l2082_208267


namespace largest_sum_is_sum3_l2082_208252

-- Definitions of the individual sums given in the conditions
def sum1 : ℚ := (1/4 : ℚ) + (1/5 : ℚ) * (1/2 : ℚ)
def sum2 : ℚ := (1/4 : ℚ) - (1/6 : ℚ)
def sum3 : ℚ := (1/4 : ℚ) + (1/3 : ℚ) * (1/2 : ℚ)
def sum4 : ℚ := (1/4 : ℚ) - (1/8 : ℚ)
def sum5 : ℚ := (1/4 : ℚ) + (1/7 : ℚ) * (1/2 : ℚ)

-- Theorem to prove that sum3 is the largest
theorem largest_sum_is_sum3 : sum3 = (5/12 : ℚ) ∧ sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := 
by 
  -- The proof would go here
  sorry

end largest_sum_is_sum3_l2082_208252


namespace value_of_a_l2082_208210

noncomputable def coefficient_of_x2_term (a : ℝ) : ℝ :=
  a^4 * Nat.choose 8 4

theorem value_of_a (a : ℝ) (h : coefficient_of_x2_term a = 70) : a = 1 ∨ a = -1 := by
  sorry

end value_of_a_l2082_208210


namespace factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l2082_208287

-- Definitions from conditions
theorem factorization_option_a (a b : ℝ) : a^4 * b - 6 * a^3 * b + 9 * a^2 * b = a^2 * b * (a^2 - 6 * a + 9) ↔ a^2 * b * (a - 3)^2 ≠ a^2 * b * (a^2 - 6 * a - 9) := sorry

theorem factorization_option_b (x : ℝ) : (x^2 - x + 1/4) = (x - 1/2)^2 := sorry

theorem factorization_option_c (x : ℝ) : x^2 - 2 * x + 4 = (x - 2)^2 ↔ x^2 - 2 * x + 4 ≠ x^2 - 4 * x + 4 := sorry

theorem factorization_option_d (x y : ℝ) : 4 * x^2 - y^2 = (2 * x + y) * (2 * x - y) ↔ (4 * x + y) * (4 * x - y) ≠ (2 * x + y) * (2 * x - y) := sorry

-- Main theorem that states option B's factorization is correct
theorem correct_factorization_b (x : ℝ) (h1 : x^2 - x + 1/4 = (x - 1/2)^2)
                                (h2 : ∀ (a b : ℝ), a^4 * b - 6 * a^3 * b + 9 * a^2 * b ≠ a^2 * b * (a^2 - 6 * a - 9))
                                (h3 : ∀ (x : ℝ), x^2 - 2 * x + 4 ≠ (x - 2)^2)
                                (h4 : ∀ (x y : ℝ), 4 * x^2 - y^2 ≠ (4 * x + y) * (4 * x - y)) : 
                                (x^2 - x + 1/4 = (x - 1/2)^2) := 
                                by 
                                sorry

end factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l2082_208287


namespace gym_cost_l2082_208238

theorem gym_cost (x : ℕ) (hx : x > 0) (h1 : 50 + 12 * x + 48 * x = 650) : x = 10 :=
by
  sorry

end gym_cost_l2082_208238


namespace part1_l2082_208253

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x, 0 ≤ x → f x = Real.sqrt x)
variable (h1 : 0 ≤ x1)
variable (h2 : 0 ≤ x2)
variable (h3 : x1 ≠ x2)

theorem part1 : (1/2) * (f x1 + f x2) < f ((x1 + x2) / 2) :=
  sorry

end part1_l2082_208253


namespace amanda_more_than_average_l2082_208205

-- Conditions
def jill_peaches : ℕ := 12
def steven_peaches : ℕ := jill_peaches + 15
def jake_peaches : ℕ := steven_peaches - 16
def amanda_peaches : ℕ := jill_peaches * 2
def total_peaches : ℕ := jake_peaches + steven_peaches + jill_peaches
def average_peaches : ℚ := total_peaches / 3

-- Question: Prove that Amanda has 7.33 more peaches than the average peaches Jake, Steven, and Jill have
theorem amanda_more_than_average : amanda_peaches - average_peaches = 22 / 3 := by
  sorry

end amanda_more_than_average_l2082_208205


namespace interval_of_decrease_l2082_208298

theorem interval_of_decrease (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x) :
  ∀ x0 : ℝ, ∀ x1 : ℝ, x0 ≥ 3 → x0 ≤ x1 → f (x1 - 3) ≤ f (x0 - 3) := sorry

end interval_of_decrease_l2082_208298


namespace crayons_per_color_in_each_box_l2082_208277

def crayons_in_each_box : ℕ := 2

theorem crayons_per_color_in_each_box
  (colors : ℕ)
  (boxes_per_hour : ℕ)
  (crayons_in_4_hours : ℕ)
  (hours : ℕ)
  (total_boxes : ℕ := boxes_per_hour * hours)
  (crayons_per_box : ℕ := crayons_in_4_hours / total_boxes)
  (crayons_per_color : ℕ := crayons_per_box / colors)
  (colors_eq : colors = 4)
  (boxes_per_hour_eq : boxes_per_hour = 5)
  (crayons_in_4_hours_eq : crayons_in_4_hours = 160)
  (hours_eq : hours = 4) : crayons_per_color = crayons_in_each_box :=
by {
  sorry
}

end crayons_per_color_in_each_box_l2082_208277


namespace min_value_expr_l2082_208266

noncomputable def expr (x : ℝ) : ℝ := (Real.sin x)^8 + (Real.cos x)^8 + 3 / (Real.sin x)^6 + (Real.cos x)^6 + 3

theorem min_value_expr : ∃ x : ℝ, expr x = 14 / 31 := 
by
  sorry

end min_value_expr_l2082_208266


namespace range_of_k_l2082_208200

def f (x : ℝ) : ℝ := x^3 - 12*x

def not_monotonic_on_I (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), k - 1 < x₁ ∧ x₁ < k + 1 ∧ k - 1 < x₂ ∧ x₂ < k + 1 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) * (x₁ - x₂) < 0

theorem range_of_k (k : ℝ) : not_monotonic_on_I k ↔ (k > -3 ∧ k < -1) ∨ (k > 1 ∧ k < 3) :=
sorry

end range_of_k_l2082_208200


namespace total_fish_l2082_208214

theorem total_fish {lilly_fish rosy_fish : ℕ} (h1 : lilly_fish = 10) (h2 : rosy_fish = 11) : 
lilly_fish + rosy_fish = 21 :=
by 
  sorry

end total_fish_l2082_208214


namespace sequence_increasing_l2082_208233

theorem sequence_increasing (a : ℕ → ℝ) (a0 : ℝ) (h0 : a 0 = 1 / 5)
  (H : ∀ n : ℕ, a (n + 1) = 2^n - 3 * a n) :
  ∀ n : ℕ, a (n + 1) > a n :=
sorry

end sequence_increasing_l2082_208233


namespace intersection_M_N_l2082_208273

-- Given set M defined by the inequality
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Given set N defined by the interval
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The intersection M ∩ N should be equal to the interval [1, 2)
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l2082_208273


namespace unique_solution_l2082_208206

theorem unique_solution (x : ℝ) : 
  ∃! x, 2003^x + 2004^x = 2005^x := 
sorry

end unique_solution_l2082_208206


namespace round_robin_games_count_l2082_208242

theorem round_robin_games_count (n : ℕ) (h : n = 6) : 
  (n * (n - 1)) / 2 = 15 := by
  sorry

end round_robin_games_count_l2082_208242


namespace triangle_is_right_l2082_208209

-- Define the side lengths of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- Define a predicate to check if a triangle is right using Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- The proof problem statement
theorem triangle_is_right : is_right_triangle a b c :=
sorry

end triangle_is_right_l2082_208209


namespace rate_of_discount_l2082_208203

theorem rate_of_discount (Marked_Price Selling_Price : ℝ) (h_marked : Marked_Price = 80) (h_selling : Selling_Price = 68) : 
  ((Marked_Price - Selling_Price) / Marked_Price) * 100 = 15 :=
by
  -- Definitions from conditions
  rw [h_marked, h_selling]
  -- Substitute the values and simplify
  sorry

end rate_of_discount_l2082_208203


namespace spider_total_distance_l2082_208237

theorem spider_total_distance
    (radius : ℝ)
    (diameter : ℝ)
    (half_diameter : ℝ)
    (final_leg : ℝ)
    (total_distance : ℝ) :
    radius = 75 →
    diameter = 2 * radius →
    half_diameter = diameter / 2 →
    final_leg = 90 →
    (half_diameter ^ 2 + final_leg ^ 2 = diameter ^ 2) →
    total_distance = diameter + half_diameter + final_leg →
    total_distance = 315 :=
by
  intros
  sorry

end spider_total_distance_l2082_208237


namespace radii_inequality_l2082_208256

variable {R1 R2 R3 r : ℝ}

/-- Given that R1, R2, and R3 are the radii of three circles passing through a vertex of a triangle 
and touching the opposite side, and r is the radius of the incircle of this triangle,
prove that 1 / R1 + 1 / R2 + 1 / R3 ≤ 1 / r. -/
theorem radii_inequality (h_ge : ∀ i : Fin 3, 0 < [R1, R2, R3][i]) (h_incircle : 0 < r) :
  (1 / R1) + (1 / R2) + (1 / R3) ≤ 1 / r :=
  sorry

end radii_inequality_l2082_208256


namespace complex_multiplication_result_l2082_208202

-- Define the complex numbers used in the problem
def a : ℂ := 4 - 3 * Complex.I
def b : ℂ := 4 + 3 * Complex.I

-- State the theorem we want to prove
theorem complex_multiplication_result : a * b = 25 := 
by
  -- Proof is omitted
  sorry

end complex_multiplication_result_l2082_208202


namespace incorrect_option_l2082_208239

theorem incorrect_option (a : ℝ) (h : a ≠ 0) : (a + 2) ^ 0 ≠ 1 ↔ a = -2 :=
by {
  sorry
}

end incorrect_option_l2082_208239


namespace angle_ABC_is_45_l2082_208231

theorem angle_ABC_is_45
  (x : ℝ)
  (h1 : ∀ (ABC : ℝ), x = 180 - ABC → x = 45) :
  2 * (x / 2) = (180 - x) / 6 → x = 45 :=
by
  sorry

end angle_ABC_is_45_l2082_208231


namespace route_B_is_quicker_l2082_208213

theorem route_B_is_quicker : 
    let distance_A := 6 -- miles
    let speed_A := 30 -- mph
    let distance_B_total := 5 -- miles
    let distance_B_non_school := 4.5 -- miles
    let speed_B_non_school := 40 -- mph
    let distance_B_school := 0.5 -- miles
    let speed_B_school := 20 -- mph
    let time_A := (distance_A / speed_A) * 60 -- minutes
    let time_B_non_school := (distance_B_non_school / speed_B_non_school) * 60 -- minutes
    let time_B_school := (distance_B_school / speed_B_school) * 60 -- minutes
    let time_B := time_B_non_school + time_B_school -- minutes
    let time_difference := time_A - time_B -- minutes
    time_difference = 3.75 :=
sorry

end route_B_is_quicker_l2082_208213


namespace volume_ratio_cones_l2082_208265

theorem volume_ratio_cones :
  let rC := 16.5
  let hC := 33
  let rD := 33
  let hD := 16.5
  let VC := (1 / 3) * Real.pi * rC^2 * hC
  let VD := (1 / 3) * Real.pi * rD^2 * hD
  (VC / VD) = (1 / 2) :=
by
  sorry

end volume_ratio_cones_l2082_208265


namespace range_of_a_l2082_208228

variable (a : ℝ)

theorem range_of_a (h : ¬ ∃ x : ℝ, x^2 + 2 * x + a ≤ 0) : 1 < a :=
by {
  -- Proof will go here.
  sorry
}

end range_of_a_l2082_208228


namespace contrapositive_of_neg_and_inverse_l2082_208257

theorem contrapositive_of_neg_and_inverse (p r s : Prop) (h1 : r = ¬p) (h2 : s = ¬r) : s = (¬p → false) :=
by
  -- We have that r = ¬p
  have hr : r = ¬p := h1
  -- And we have that s = ¬r
  have hs : s = ¬r := h2
  -- Now we need to show that s is the contrapositive of p, which is ¬p → false
  sorry

end contrapositive_of_neg_and_inverse_l2082_208257


namespace sample_size_proof_l2082_208244

-- Define the quantities produced by each workshop
def units_A : ℕ := 120
def units_B : ℕ := 80
def units_C : ℕ := 60

-- Define the number of units sampled from Workshop C
def samples_C : ℕ := 3

-- Calculate the total sample size n
def total_sample_size : ℕ :=
  let sampling_fraction := samples_C / units_C
  let samples_A := sampling_fraction * units_A
  let samples_B := sampling_fraction * units_B
  samples_A + samples_B + samples_C

-- The theorem we want to prove
theorem sample_size_proof : total_sample_size = 13 :=
by sorry

end sample_size_proof_l2082_208244


namespace problem1_l2082_208243

variables (m n : ℝ)

axiom cond1 : 4 * m + n = 90
axiom cond2 : 2 * m - 3 * n = 10

theorem problem1 : (m + 2 * n) ^ 2 - (3 * m - n) ^ 2 = -900 := sorry

end problem1_l2082_208243


namespace total_number_of_balls_in_fish_tank_l2082_208285

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end total_number_of_balls_in_fish_tank_l2082_208285


namespace daily_earning_r_l2082_208219

theorem daily_earning_r :
  exists P Q R : ℝ, 
    (P + Q + R = 220) ∧
    (P + R = 120) ∧
    (Q + R = 130) ∧
    (R = 30) := 
by
  sorry

end daily_earning_r_l2082_208219


namespace correct_ordering_l2082_208294

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom monotonicity (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : 0 ≤ x2) (h3 : x1 ≠ x2) : (x1 - x2) * (f x1 - f x2) > 0

theorem correct_ordering : f 1 < f (-2) ∧ f (-2) < f 3 :=
by sorry

end correct_ordering_l2082_208294


namespace linemen_ounces_per_drink_l2082_208291

-- Definitions corresponding to the conditions.
def linemen := 12
def skill_position_drink := 6
def skill_position_before_refill := 5
def cooler_capacity := 126

-- The theorem that requires proof.
theorem linemen_ounces_per_drink (L : ℕ) (h : 12 * L + 5 * skill_position_drink = cooler_capacity) : L = 8 :=
by
  sorry

end linemen_ounces_per_drink_l2082_208291


namespace sheet_length_l2082_208222

theorem sheet_length (L : ℝ) : 
  (20 * L > 0) → 
  ((16 * (L - 6)) / (20 * L) = 0.64) → 
  L = 30 :=
by
  intro h1 h2
  sorry

end sheet_length_l2082_208222


namespace find_m_l2082_208261

theorem find_m (m : ℝ) (h₁: 0 < m) (h₂: ∀ p q : ℝ × ℝ, p = (m, 4) → q = (2, m) → ∃ s : ℝ, s = m^2 ∧ ((q.2 - p.2) / (q.1 - p.1)) = s) : m = 2 :=
by
  sorry

end find_m_l2082_208261


namespace correct_formulas_l2082_208240

theorem correct_formulas (n : ℕ) :
  ((2 * n - 1)^2 - 4 * (n * (n - 1)) / 2) = (2 * n^2 - 2 * n + 1) ∧ 
  (1 + ((n - 1) * n) / 2 * 4) = (2 * n^2 - 2 * n + 1) ∧ 
  ((n - 1)^2 + n^2) = (2 * n^2 - 2 * n + 1) := by
  sorry

end correct_formulas_l2082_208240


namespace laura_change_l2082_208216

theorem laura_change : 
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250
  (amount_given - total_cost) = 10 :=
by
  -- definitions from conditions
  let pants_cost := 2 * 54
  let shirts_cost := 4 * 33
  let total_cost := pants_cost + shirts_cost
  let amount_given := 250

  -- the statement we are proving
  show (amount_given - total_cost) = 10
  sorry

end laura_change_l2082_208216


namespace negation_of_proposition_l2082_208211

theorem negation_of_proposition (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) :=
by sorry

end negation_of_proposition_l2082_208211


namespace equation_solution_l2082_208290

theorem equation_solution (x : ℝ) (h₁ : 2 * x - 5 ≠ 0) (h₂ : 5 - 2 * x ≠ 0) :
  (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ↔ (x = 0) :=
by
  sorry

end equation_solution_l2082_208290


namespace total_money_found_l2082_208263

-- Define the conditions
def donna_share := 0.40
def friendA_share := 0.35
def friendB_share := 0.25
def donna_amount := 39.0

-- Define the problem statement/proof
theorem total_money_found (donna_share friendA_share friendB_share donna_amount : ℝ) 
  (h1 : donna_share = 0.40) 
  (h2 : friendA_share = 0.35) 
  (h3 : friendB_share = 0.25) 
  (h4 : donna_amount = 39.0) :
  ∃ total_money : ℝ, total_money = 97.50 := 
by
  -- The calculations and actual proof will go here
  sorry

end total_money_found_l2082_208263


namespace tomato_plants_per_row_l2082_208292

-- Definitions based on given conditions.
variables (T C P : ℕ)

-- Condition 1: For each row of tomato plants, she is planting 2 rows of cucumbers
def cucumber_rows (T : ℕ) := 2 * T

-- Condition 2: She has enough room for 15 rows of plants in total
def total_rows (T : ℕ) (C : ℕ) := T + C = 15

-- Condition 3: If each plant produces 3 tomatoes, she will have 120 tomatoes in total
def total_tomatoes (P : ℕ) := 5 * P * 3 = 120

-- The task is to prove that P = 8
theorem tomato_plants_per_row : 
  ∀ T C P : ℕ, cucumber_rows T = C → total_rows T C → total_tomatoes P → P = 8 :=
by
  -- The actual proof will go here.
  sorry

end tomato_plants_per_row_l2082_208292


namespace a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l2082_208278

variable {a b c : ℝ}

theorem a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2 :
  ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) :=
sorry

theorem a_gt_b_necessary_not_sufficient_ac2_gt_bc2 :
  ¬((a > b) → (a * c^2 > b * c^2)) ∧ ((a * c^2 > b * c^2) → (a > b)) :=
sorry

end a_gt_b_neither_sufficient_nor_necessary_a2_gt_b2_a_gt_b_necessary_not_sufficient_ac2_gt_bc2_l2082_208278


namespace value_of_f_at_5_l2082_208212

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_f_at_5 : f 5 = 15 := 
by {
  sorry
}

end value_of_f_at_5_l2082_208212


namespace price_increase_and_decrease_l2082_208264

theorem price_increase_and_decrease (P : ℝ) (x : ℝ) 
  (h1 : 0 < P) 
  (h2 : (P * (1 - (x / 100) ^ 2)) = 0.81 * P) : 
  abs (x - 44) < 1 :=
by
  sorry

end price_increase_and_decrease_l2082_208264


namespace income_expenditure_ratio_l2082_208262

theorem income_expenditure_ratio
  (I E : ℕ)
  (h1 : I = 18000)
  (S : ℕ)
  (h2 : S = 2000)
  (h3 : S = I - E) :
  I.gcd E = 2000 ∧ I / I.gcd E = 9 ∧ E / I.gcd E = 8 :=
by sorry

end income_expenditure_ratio_l2082_208262


namespace cos_square_minus_sin_square_15_l2082_208234

theorem cos_square_minus_sin_square_15 (cos_30 : Real.cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2) : 
  Real.cos (15 * Real.pi / 180) ^ 2 - Real.sin (15 * Real.pi / 180) ^ 2 = (Real.sqrt 3) / 2 := 
by 
  sorry

end cos_square_minus_sin_square_15_l2082_208234


namespace last_two_nonzero_digits_70_factorial_l2082_208247

theorem last_two_nonzero_digits_70_factorial : 
  let N := 70
  (∀ N : ℕ, 0 < N → N % 2 ≠ 0 → N % 5 ≠ 0 → ∃ x : ℕ, x % 100 = N % (N + (N! / (2 ^ 16)))) →
  (N! / 10 ^ 16) % 100 = 68 :=
by
sorry

end last_two_nonzero_digits_70_factorial_l2082_208247


namespace fraction_age_28_to_32_l2082_208289

theorem fraction_age_28_to_32 (F : ℝ) (total_participants : ℝ) 
  (next_year_fraction_increase : ℝ) (next_year_fraction : ℝ) 
  (h1 : total_participants = 500)
  (h2 : next_year_fraction_increase = (1 / 8 : ℝ))
  (h3 : next_year_fraction = 0.5625) 
  (h4 : F + next_year_fraction_increase * F = next_year_fraction) :
  F = 0.5 :=
by
  sorry

end fraction_age_28_to_32_l2082_208289


namespace should_agree_to_buy_discount_card_l2082_208270

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end should_agree_to_buy_discount_card_l2082_208270


namespace average_of_remaining_two_numbers_l2082_208271

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℚ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) : 
  ((e + f) / 2 = 6.9) :=
by
  sorry

end average_of_remaining_two_numbers_l2082_208271


namespace cylinder_height_l2082_208295

noncomputable def height_of_cylinder_inscribed_in_sphere : ℝ := 4 * Real.sqrt 10

theorem cylinder_height :
  ∀ (R_cylinder R_sphere : ℝ), R_cylinder = 3 → R_sphere = 7 →
  (height_of_cylinder_inscribed_in_sphere = 4 * Real.sqrt 10) := by
  intros R_cylinder R_sphere h1 h2
  sorry

end cylinder_height_l2082_208295


namespace math_problem_l2082_208260

variable (a : ℝ)

theorem math_problem (h : a^2 + 3 * a - 2 = 0) :
  ((a^2 - 4) / (a^2 - 4 * a + 4) - 1 / (2 - a)) / (2 / (a^2 - 2 * a)) = 1 := 
sorry

end math_problem_l2082_208260


namespace geometric_sequence_sum_squared_l2082_208250

theorem geometric_sequence_sum_squared (a : ℕ → ℕ) (n : ℕ) (q : ℕ) 
    (h_geometric: ∀ n, a (n + 1) = a n * q)
    (h_a1 : a 1 = 2)
    (h_a3 : a 3 = 4) :
    (a 1)^2 + (a 2)^2 + (a 3)^2 + (a 4)^2 + (a 5)^2 + (a 6)^2 + (a 7)^2 + (a 8)^2 = 1020 :=
by
  sorry

end geometric_sequence_sum_squared_l2082_208250


namespace profit_difference_l2082_208249

variable (a_capital b_capital c_capital b_profit : ℕ)

theorem profit_difference (h₁ : a_capital = 8000) (h₂ : b_capital = 10000) 
                          (h₃ : c_capital = 12000) (h₄ : b_profit = 2000) : 
  c_capital * (b_profit / b_capital) - a_capital * (b_profit / b_capital) = 800 := 
sorry

end profit_difference_l2082_208249


namespace range_of_m_l2082_208258

theorem range_of_m (m : ℝ) (P : Prop) (Q : Prop) : 
  (P ∨ Q) ∧ ¬(P ∧ Q) →
  (P ↔ (m^2 - 4 > 0)) →
  (Q ↔ (16 * (m - 2)^2 - 16 < 0)) →
  (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l2082_208258


namespace find_A_and_evaluate_A_minus_B_l2082_208276

-- Given definitions
def B (x y : ℝ) : ℝ := 4 * x ^ 2 - 3 * y - 1
def result (x y : ℝ) : ℝ := 6 * x ^ 2 - y

-- Defining the polynomial A based on the first condition
def A (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * y + 1

-- The main theorem to be proven
theorem find_A_and_evaluate_A_minus_B :
  (∀ x y : ℝ, B x y + A x y = result x y) →
  (∀ x y : ℝ, |x - 1| * (y + 1) ^ 2 = 0 → A x y - B x y = -5) :=
by
  intro h1 h2
  sorry

end find_A_and_evaluate_A_minus_B_l2082_208276


namespace apples_weight_l2082_208235

theorem apples_weight (x : ℝ) (price1 : ℝ) (price2 : ℝ) (new_price_diff : ℝ) (total_revenue : ℝ)
  (h1 : price1 * x = 228)
  (h2 : price2 * (x + 5) = 180)
  (h3 : ∀ kg: ℝ, kg * (price1 - new_price_diff) = total_revenue)
  (h4 : new_price_diff = 0.9)
  (h5 : total_revenue = 408) :
  2 * x + 5 = 85 :=
by
  sorry

end apples_weight_l2082_208235


namespace slant_asymptote_sum_l2082_208281

theorem slant_asymptote_sum (m b : ℝ) 
  (h : ∀ x : ℝ, y = 3*x^2 + 4*x - 8 / (x - 4) → y = m*x + b) :
  m + b = 19 :=
sorry

end slant_asymptote_sum_l2082_208281


namespace conversion_rate_false_l2082_208293

-- Definition of conversion rates between units
def conversion_rate_hour_minute : ℕ := 60
def conversion_rate_minute_second : ℕ := 60

-- Theorem stating that the rate being 100 is false under the given conditions
theorem conversion_rate_false (h1 : conversion_rate_hour_minute = 60) 
  (h2 : conversion_rate_minute_second = 60) : 
  ¬ (conversion_rate_hour_minute = 100 ∧ conversion_rate_minute_second = 100) :=
by {
  sorry
}

end conversion_rate_false_l2082_208293


namespace green_fraction_is_three_fifths_l2082_208286

noncomputable def fraction_green_after_tripling (total_balloons : ℕ) : ℚ :=
  let green_balloons := total_balloons / 3
  let new_green_balloons := green_balloons * 3
  let new_total_balloons := total_balloons * (5 / 3)
  new_green_balloons / new_total_balloons

theorem green_fraction_is_three_fifths (total_balloons : ℕ) (h : total_balloons > 0) : 
  fraction_green_after_tripling total_balloons = 3 / 5 := 
by 
  sorry

end green_fraction_is_three_fifths_l2082_208286


namespace greater_number_is_84_l2082_208207

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) : x = 84 :=
sorry

end greater_number_is_84_l2082_208207


namespace denominator_or_divisor_cannot_be_zero_l2082_208232

theorem denominator_or_divisor_cannot_be_zero (a b c : ℝ) : b ≠ 0 ∧ c ≠ 0 → (a / b ≠ a ∨ a / c ≠ a) :=
by
  intro h
  sorry

end denominator_or_divisor_cannot_be_zero_l2082_208232


namespace zhang_hua_new_year_cards_l2082_208280

theorem zhang_hua_new_year_cards (x y z : ℕ) 
  (h1 : Nat.lcm (Nat.lcm x y) z = 60)
  (h2 : Nat.gcd x y = 4)
  (h3 : Nat.gcd y z = 3) : 
  x = 4 ∨ x = 20 :=
by
  sorry

end zhang_hua_new_year_cards_l2082_208280


namespace smallest_b_value_l2082_208229

noncomputable def smallest_possible_value_of_b : ℝ :=
  (3 + Real.sqrt 5) / 2

theorem smallest_b_value
  (a b : ℝ)
  (h1 : 1 < a)
  (h2 : a < b)
  (h3 : b ≥ a + 1)
  (h4 : (1/b) + (1/a) ≤ 1) :
  b = smallest_possible_value_of_b :=
sorry

end smallest_b_value_l2082_208229


namespace distinct_square_sum_100_l2082_208283

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l2082_208283


namespace factor_quadratic_l2082_208296

theorem factor_quadratic (m p : ℝ) (h : (m - 8) ∣ (m^2 - p * m - 24)) : p = 5 :=
sorry

end factor_quadratic_l2082_208296


namespace digit_8_appears_300_times_l2082_208201

-- Define a function that counts the occurrences of a specific digit in a list of numbers
def count_digit_occurrences (digit : Nat) (range : List Nat) : Nat :=
  range.foldl (λ acc n => acc + (Nat.digits 10 n).count digit) 0

-- Theorem statement: The digit 8 appears 300 times in the list of integers from 1 to 1000
theorem digit_8_appears_300_times : count_digit_occurrences 8 (List.range' 0 1000) = 300 :=
by
  sorry

end digit_8_appears_300_times_l2082_208201


namespace books_ratio_3_to_1_l2082_208288

-- Definitions based on the conditions
def initial_books : ℕ := 220
def books_rebecca_received : ℕ := 40
def remaining_books : ℕ := 60
def total_books_given_away := initial_books - remaining_books
def books_mara_received := total_books_given_away - books_rebecca_received

-- The proof that the ratio of the number of books Mara received to the number of books Rebecca received is 3:1
theorem books_ratio_3_to_1 : (books_mara_received : ℚ) / books_rebecca_received = 3 := by
  sorry

end books_ratio_3_to_1_l2082_208288


namespace line_equation_l2082_208299

theorem line_equation (x y : ℝ) (m : ℝ) (h1 : (1, 2) = (x, y)) (h2 : m = 3) :
  y = 3 * x - 1 :=
by
  sorry

end line_equation_l2082_208299


namespace probability_at_least_one_heart_l2082_208268

theorem probability_at_least_one_heart (total_cards hearts : ℕ) 
  (top_card_positions : Π n : ℕ, n = 3) 
  (non_hearts_cards : Π n : ℕ, n = total_cards - hearts) 
  (h_total_cards : total_cards = 52) (h_hearts : hearts = 13) 
  : (1 - ((39 * 38 * 37 : ℚ) / (52 * 51 * 50))) = (325 / 425) := 
by {
  sorry
}

end probability_at_least_one_heart_l2082_208268


namespace parabola_equation_line_equation_chord_l2082_208236

section
variables (p : ℝ) (x_A y_A : ℝ) (M_x M_y : ℝ)
variable (h_p_pos : p > 0)
variable (h_A : y_A^2 = 8 * x_A)
variable (h_directrix_A : x_A + p / 2 = 5)
variable (h_M : (M_x, M_y) = (3, 2))

theorem parabola_equation (h_x_A : x_A = 3) : y_A^2 = 8 * x_A :=
sorry

theorem line_equation_chord
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : y1^2 = 8 * x1 ∧ y2^2 = 8 * x2)
  (h_chord_M : (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = 2) :
  y_M - 2 * x_M + 4 = 0 :=
sorry
end

end parabola_equation_line_equation_chord_l2082_208236


namespace solve_system_l2082_208227

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) :=
  x ≠ y ∧
  a ≠ 0 ∧
  c ≠ 0 ∧
  (x + z) * a = x - y ∧
  (x + z) * b = x^2 - y^2 ∧
  (x + z)^2 * (b^2 / (a^2 * c)) = (x^3 + x^2 * y - x * y^2 - y^3)

-- Proof goal: establish the values of x, y, and z
theorem solve_system (a b c x y z : ℝ) (h : system_of_equations a b c x y z):
  x = (a^3 * c + b) / (2 * a) ∧
  y = (b - a^3 * c) / (2 * a) ∧
  z = (2 * a^2 * c - a^3 * c - b) / (2 * a) :=
by
  sorry

end solve_system_l2082_208227


namespace rachel_found_boxes_l2082_208208

theorem rachel_found_boxes (pieces_per_box total_pieces B : ℕ) 
  (h1 : pieces_per_box = 7) 
  (h2 : total_pieces = 49) 
  (h3 : B = total_pieces / pieces_per_box) : B = 7 := 
by 
  sorry

end rachel_found_boxes_l2082_208208


namespace uncolored_vertex_not_original_hexagon_vertex_l2082_208284

theorem uncolored_vertex_not_original_hexagon_vertex
    (point_index : ℕ)
    (orig_hex_vertices : Finset ℕ) -- Assuming the vertices of the original hexagon are represented as a finite set of indices.
    (num_parts : ℕ := 1000) -- Each hexagon side is divided into 1000 parts
    (label : ℕ → Fin 3) -- A function labeling each point with 0, 1, or 2.
    (is_valid_labeling : ∀ (i j k : ℕ), label i ≠ label j ∧ label j ≠ label k ∧ label k ≠ label i) -- No duplicate labeling within a triangle.
    (is_single_uncolored : ∀ (p : ℕ), (p ∈ orig_hex_vertices ∨ ∃ (v : ℕ), v ∈ orig_hex_vertices ∧ p = v) → p ≠ point_index) -- Only one uncolored point
    : point_index ∉ orig_hex_vertices :=
by sorry

end uncolored_vertex_not_original_hexagon_vertex_l2082_208284


namespace Merrill_marbles_Vivian_marbles_l2082_208223

variable (M E S V : ℕ)

-- Conditions
axiom Merrill_twice_Elliot : M = 2 * E
axiom Merrill_Elliot_five_fewer_Selma : M + E = S - 5
axiom Selma_fifty_marbles : S = 50
axiom Vivian_35_percent_more_Elliot : V = (135 * E) / 100 -- since Lean works better with integers, use 135/100 instead of 1.35
axiom Vivian_Elliot_difference_greater_five : V - E > 5

-- Questions
theorem Merrill_marbles (M E S : ℕ) (h1: M = 2 * E) (h2: M + E = S - 5) (h3: S = 50) : M = 30 := by
  sorry

theorem Vivian_marbles (V E : ℕ) (h1: V = (135 * E) / 100) (h2: V - E > 5) (h3: E = 15) : V = 21 := by
  sorry

end Merrill_marbles_Vivian_marbles_l2082_208223


namespace Jaco_budget_for_parents_gifts_l2082_208274

theorem Jaco_budget_for_parents_gifts :
  ∃ (m n : ℕ), (m = 14 ∧ n = 14) ∧ 
  (∀ (friends gifts_friends budget : ℕ), 
   friends = 8 → gifts_friends = 9 → budget = 100 → 
   (budget - (friends * gifts_friends)) / 2 = m ∧ 
   (budget - (friends * gifts_friends)) / 2 = n) := 
sorry

end Jaco_budget_for_parents_gifts_l2082_208274


namespace backpacks_weight_l2082_208255

variables (w_y w_g : ℝ)

theorem backpacks_weight :
  (2 * w_y + 3 * w_g = 44) ∧
  (w_y + w_g + w_g / 2 = w_g + w_y / 2) →
  (w_g = 4) ∧ (w_y = 12) :=
by
  intros h
  sorry

end backpacks_weight_l2082_208255


namespace total_number_of_coins_l2082_208251

theorem total_number_of_coins (x n : Nat) (h1 : 15 * 5 = 75) (h2 : 125 - 75 = 50)
  (h3 : x = 50 / 2) (h4 : n = x + 15) : n = 40 := by
  sorry

end total_number_of_coins_l2082_208251


namespace sequence_term_37_l2082_208269

theorem sequence_term_37 (n : ℕ) (h_pos : 0 < n) (h_eq : 3 * n + 1 = 37) : n = 12 :=
by
  sorry

end sequence_term_37_l2082_208269


namespace num_workers_l2082_208218

-- Define the number of workers (n) and the initial contribution per worker (x)
variable (n x : ℕ)

-- Condition 1: The total contribution is Rs. 3 lacs
axiom h1 : n * x = 300000

-- Condition 2: If each worker contributed Rs. 50 more, the total would be Rs. 3.75 lacs
axiom h2 : n * (x + 50) = 375000

-- Proof Problem: Prove that the number of workers (n) is 1500
theorem num_workers : n = 1500 :=
by
  -- The proof will go here
  sorry

end num_workers_l2082_208218


namespace inequality_equivalence_l2082_208241

theorem inequality_equivalence (a : ℝ) : a < -1 ↔ a + 1 < 0 :=
by
  sorry

end inequality_equivalence_l2082_208241


namespace line_intersection_equation_of_l4_find_a_l2082_208225

theorem line_intersection (P : ℝ × ℝ)
    (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) :
  P = (-2, 2) :=
sorry

theorem equation_of_l4 (l4 : ℝ → ℝ → Prop)
    (P : ℝ × ℝ) (h1: 3 * P.1 + 4 * P.2 - 2 = 0)
    (h2: 2 * P.1 + P.2 + 2 = 0) 
    (h_parallel: ∀ x y, l4 x y ↔ y = 1/2 * x + 3)
    (x y : ℝ) :
  l4 x y ↔ y = 1/2 * x + 3 :=
sorry

theorem find_a (a : ℝ) :
    (∀ x y, 2 * x + y + 2 = 0 → y = -2 * x - 2) →
    (∀ x y, a * x - 2 * y + 1 = 0 → y = 1/2 * x - 1/2) →
    a = 1 :=
sorry

end line_intersection_equation_of_l4_find_a_l2082_208225


namespace pond_to_field_ratio_l2082_208224

theorem pond_to_field_ratio 
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l = 28)
  (side_pond : ℝ := 7) 
  (A_pond : ℝ := side_pond ^ 2) 
  (A_field : ℝ := l * w):
  (A_pond / A_field) = 1 / 8 :=
by
  sorry

end pond_to_field_ratio_l2082_208224


namespace rank_classmates_l2082_208259

-- Definitions of the conditions
def emma_tallest (emma david fiona : ℕ) : Prop := emma > david ∧ emma > fiona
def fiona_not_shortest (david emma fiona : ℕ) : Prop := david > fiona ∧ emma > fiona
def david_not_tallest (david emma fiona : ℕ) : Prop := emma > david ∧ fiona > david

def exactly_one_true (david emma fiona : ℕ) : Prop :=
  (emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ fiona_not_shortest david emma fiona ∧ ¬david_not_tallest david emma fiona) ∨
  (¬emma_tallest emma david fiona ∧ ¬fiona_not_shortest david emma fiona ∧ david_not_tallest david emma fiona)

-- The final proof statement
theorem rank_classmates (david emma fiona : ℕ) (h : exactly_one_true david emma fiona) : david > fiona ∧ fiona > emma :=
  sorry

end rank_classmates_l2082_208259


namespace system_of_equations_solution_l2082_208215

theorem system_of_equations_solution (x y z : ℝ) :
  (x * y + x * z = 8 - x^2) →
  (x * y + y * z = 12 - y^2) →
  (y * z + z * x = -4 - z^2) →
  (x = 2 ∧ y = 3 ∧ z = -1) ∨ (x = -2 ∧ y = -3 ∧ z = 1) :=
by
  sorry

end system_of_equations_solution_l2082_208215


namespace prime_number_solution_l2082_208297

theorem prime_number_solution (X Y : ℤ) (h_prime : Prime (X^4 + 4 * Y^4)) :
  (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) :=
sorry

end prime_number_solution_l2082_208297


namespace age_of_person_l2082_208279

/-- Given that Noah's age is twice someone's age and Noah will be 22 years old after 10 years, 
    this theorem states that the age of the person whose age is half of Noah's age is 6 years old. -/
theorem age_of_person (N : ℕ) (P : ℕ) (h1 : P = N / 2) (h2 : N + 10 = 22) : P = 6 := by
  sorry

end age_of_person_l2082_208279
