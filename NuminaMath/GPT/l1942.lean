import Mathlib

namespace output_y_for_x_eq_5_l1942_194298

def compute_y (x : Int) : Int :=
  if x > 0 then 3 * x + 1 else -2 * x + 3

theorem output_y_for_x_eq_5 : compute_y 5 = 16 := by
  sorry

end output_y_for_x_eq_5_l1942_194298


namespace max_possible_N_l1942_194269

theorem max_possible_N (cities roads N : ℕ) (h1 : cities = 1000) (h2 : roads = 2017) (h3 : N = roads - (cities - 1 + 7 - 1)) :
  N = 1009 :=
by {
  sorry
}

end max_possible_N_l1942_194269


namespace quadratic_eq_roots_are_coeffs_l1942_194224

theorem quadratic_eq_roots_are_coeffs :
  ∃ (a b : ℝ), (a = r_1) → (b = r_2) →
  (r_1 + r_2 = -a) → (r_1 * r_2 = b) →
  r_1 = 1 ∧ r_2 = -2 ∧ (x^2 + x - 2 = 0):=
by
  sorry

end quadratic_eq_roots_are_coeffs_l1942_194224


namespace construction_company_order_l1942_194285

def concrete_weight : ℝ := 0.17
def bricks_weight : ℝ := 0.17
def stone_weight : ℝ := 0.5
def total_weight : ℝ := 0.84

theorem construction_company_order :
  concrete_weight + bricks_weight + stone_weight = total_weight :=
by
  -- The proof would go here but is omitted per instructions.
  sorry

end construction_company_order_l1942_194285


namespace vertical_distance_from_top_to_bottom_l1942_194219

-- Conditions
def ring_thickness : ℕ := 2
def largest_ring_diameter : ℕ := 18
def smallest_ring_diameter : ℕ := 4

-- Additional definitions based on the problem context
def count_rings : ℕ := (largest_ring_diameter - smallest_ring_diameter) / ring_thickness + 1
def inner_diameters_sum : ℕ := count_rings * (largest_ring_diameter - ring_thickness + smallest_ring_diameter) / 2
def vertical_distance : ℕ := inner_diameters_sum + 2 * ring_thickness

-- The problem statement to prove
theorem vertical_distance_from_top_to_bottom :
  vertical_distance = 76 := by
  sorry

end vertical_distance_from_top_to_bottom_l1942_194219


namespace number_exceeds_fraction_l1942_194268

theorem number_exceeds_fraction (x : ℝ) (h : x = (3/8) * x + 15) : x = 24 :=
sorry

end number_exceeds_fraction_l1942_194268


namespace chimpanzee_count_l1942_194249

def total_chimpanzees (moving_chimps : ℕ) (staying_chimps : ℕ) : ℕ :=
  moving_chimps + staying_chimps

theorem chimpanzee_count : total_chimpanzees 18 27 = 45 :=
by
  sorry

end chimpanzee_count_l1942_194249


namespace project_presentation_period_length_l1942_194237

theorem project_presentation_period_length
  (students : ℕ)
  (presentation_time_per_student : ℕ)
  (number_of_periods : ℕ)
  (total_students : students = 32)
  (time_per_student : presentation_time_per_student = 5)
  (periods_needed : number_of_periods = 4) :
  (32 * 5) / 4 = 40 := 
by {
  sorry
}

end project_presentation_period_length_l1942_194237


namespace bananas_bought_l1942_194214

def cost_per_banana : ℝ := 5.00
def total_cost : ℝ := 20.00

theorem bananas_bought : total_cost / cost_per_banana = 4 :=
by {
   sorry
}

end bananas_bought_l1942_194214


namespace roses_ordered_l1942_194275

theorem roses_ordered (tulips carnations roses : ℕ) (cost_per_flower total_expenses : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : cost_per_flower = 2)
  (h4 : total_expenses = 1890)
  (h5 : total_expenses = (tulips + carnations + roses) * cost_per_flower) :
  roses = 320 :=
by 
  -- Using the mathematical equivalence and conditions provided
  sorry

end roses_ordered_l1942_194275


namespace rotate_parabola_180_l1942_194292

theorem rotate_parabola_180 (x y : ℝ) : 
  (y = 2 * (x - 1)^2 + 2) → 
  (∃ x' y', x' = -x ∧ y' = -y ∧ y' = -2 * (x' + 1)^2 - 2) := 
sorry

end rotate_parabola_180_l1942_194292


namespace max_volume_of_cuboid_l1942_194239

theorem max_volume_of_cuboid (x y z : ℝ) (h1 : 4 * (x + y + z) = 60) : 
  x * y * z ≤ 125 :=
by
  sorry

end max_volume_of_cuboid_l1942_194239


namespace range_of_reciprocals_l1942_194242

theorem range_of_reciprocals (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) :
  ∃ c ∈ Set.Ici (9 : ℝ), (c = (1/a + 4/b)) :=
by
  sorry

end range_of_reciprocals_l1942_194242


namespace family_b_initial_members_l1942_194266

variable (x : ℕ)

theorem family_b_initial_members (h : 6 + (x - 1) + 9 + 12 + 5 + 9 = 48) : x = 8 :=
by
  sorry

end family_b_initial_members_l1942_194266


namespace prime_p_geq_7_div_240_l1942_194262

theorem prime_p_geq_7_div_240 (p : ℕ) (hp : Nat.Prime p) (hge7 : p ≥ 7) : 240 ∣ p^4 - 1 := 
sorry

end prime_p_geq_7_div_240_l1942_194262


namespace rationalize_denominator_theorem_l1942_194288

noncomputable def rationalize_denominator : Prop :=
  let num := 5
  let den := 2 + Real.sqrt 5
  let conj := 2 - Real.sqrt 5
  let expr := (num * conj) / (den * conj)
  expr = -10 + 5 * Real.sqrt 5

theorem rationalize_denominator_theorem : rationalize_denominator :=
  sorry

end rationalize_denominator_theorem_l1942_194288


namespace jake_weight_loss_l1942_194246

variable {J K L : Nat}

theorem jake_weight_loss
  (h1 : J + K = 290)
  (h2 : J = 196)
  (h3 : J - L = 2 * K) : L = 8 :=
by
  sorry

end jake_weight_loss_l1942_194246


namespace son_age_l1942_194227

theorem son_age (M S : ℕ) (h1 : M = S + 24) (h2 : M + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end son_age_l1942_194227


namespace each_friend_pays_l1942_194231

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end each_friend_pays_l1942_194231


namespace min_abs_sum_half_l1942_194259

theorem min_abs_sum_half :
  ∀ (f g : ℝ → ℝ),
  (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  (∀ x, g x = Real.sin (2 * x + Real.pi / 3)) →
  (∀ x1 x2 : ℝ, g x1 * g x2 = -1 ∧ x1 ≠ x2 → abs ((x1 + x2) / 2) = Real.pi / 6) := by
-- Definitions and conditions are set, now we can state the theorem.
  sorry

end min_abs_sum_half_l1942_194259


namespace xavier_yvonne_not_zelda_prob_l1942_194250

def Px : ℚ := 1 / 4
def Py : ℚ := 2 / 3
def Pz : ℚ := 5 / 8

theorem xavier_yvonne_not_zelda_prob : 
  (Px * Py * (1 - Pz) = 1 / 16) :=
by 
  sorry

end xavier_yvonne_not_zelda_prob_l1942_194250


namespace at_least_1991_red_points_l1942_194205

theorem at_least_1991_red_points (P : Fin 997 → ℝ × ℝ) :
  ∃ (R : Finset (ℝ × ℝ)), 1991 ≤ R.card ∧ (∀ (i j : Fin 997), i ≠ j → ((P i + P j) / 2) ∈ R) :=
sorry

end at_least_1991_red_points_l1942_194205


namespace find_a2_l1942_194260

variable (x : ℝ)
variable (a₀ a₁ a₂ a₃ : ℝ)
axiom condition : ∀ x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

theorem find_a2 : a₂ = 6 :=
by
  -- The proof that involves verifying the Taylor series expansion will come here
  sorry

end find_a2_l1942_194260


namespace find_N_l1942_194207

-- Definitions based on conditions from the problem
def remainder := 6
def dividend := 86
def divisor (Q : ℕ) := 5 * Q
def number_added_to_thrice_remainder (N : ℕ) := 3 * remainder + N
def quotient (Q : ℕ) := Q

-- The condition that relates dividend, divisor, quotient, and remainder
noncomputable def division_equation (Q : ℕ) := dividend = divisor Q * Q + remainder

-- Now, prove the condition
theorem find_N : ∃ N Q : ℕ, division_equation Q ∧ divisor Q = number_added_to_thrice_remainder N ∧ N = 2 :=
by
  sorry

end find_N_l1942_194207


namespace sufficient_not_necessary_condition_for_positive_quadratic_l1942_194228

variables {a b c : ℝ}

theorem sufficient_not_necessary_condition_for_positive_quadratic 
  (ha : a > 0)
  (hb : b^2 - 4 * a * c < 0) :
  (∀ x : ℝ, a * x ^ 2 + b * x + c > 0) 
  ∧ ¬ (∀ x : ℝ, ∃ a b c : ℝ, a > 0 ∧ b^2 - 4 * a * c ≥ 0 ∧ (a * x ^ 2 + b * x + c > 0)) :=
by
  sorry

end sufficient_not_necessary_condition_for_positive_quadratic_l1942_194228


namespace limes_remaining_l1942_194226

-- Definitions based on conditions
def initial_limes : ℕ := 9
def limes_given_to_Sara : ℕ := 4

-- Theorem to prove
theorem limes_remaining : initial_limes - limes_given_to_Sara = 5 :=
by
  -- Sorry keyword to skip the actual proof
  sorry

end limes_remaining_l1942_194226


namespace suff_but_not_nec_l1942_194200

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end suff_but_not_nec_l1942_194200


namespace knights_in_exchange_l1942_194218

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end knights_in_exchange_l1942_194218


namespace number_of_people_in_team_l1942_194229

variable (x : ℕ) -- Number of people in the team

-- Conditions as definitions
def average_age_all (x : ℕ) : ℝ := 25
def leader_age : ℝ := 45
def average_age_without_leader (x : ℕ) : ℝ := 23

-- Proof problem statement
theorem number_of_people_in_team (h1 : (x : ℝ) * average_age_all x = x * (average_age_without_leader x - 1) + leader_age) : x = 11 := by
  sorry

end number_of_people_in_team_l1942_194229


namespace max_A_min_A_l1942_194273

-- Define the problem and its conditions and question

def A_max (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

def A_min (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

theorem max_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_max B h1 h2 h3 = 999999998 := sorry

theorem min_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_min B h1 h2 h3 = 122222224 := sorry

end max_A_min_A_l1942_194273


namespace cost_price_per_meter_l1942_194243

-- Definitions
def selling_price : ℝ := 9890
def meters_sold : ℕ := 92
def profit_per_meter : ℝ := 24

-- Theorem
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 83.5 :=
by
  sorry

end cost_price_per_meter_l1942_194243


namespace foil_covered_prism_width_l1942_194272

theorem foil_covered_prism_width 
    (l w h : ℕ) 
    (h_w_eq_2l : w = 2 * l)
    (h_w_eq_2h : w = 2 * h)
    (h_volume : l * w * h = 128) 
    (h_foiled_width : q = w + 2) :
  q = 10 := 
sorry

end foil_covered_prism_width_l1942_194272


namespace Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l1942_194286

-- Define P and Q as propositions where P indicates submission of all required essays and Q indicates failing the course.
variable (P Q : Prop)

-- Ms. Thompson's statement translated to logical form.
theorem Ms_Thompsons_statement : ¬P → Q := sorry

-- The goal is to prove that if a student did not fail the course, then they submitted all the required essays.
theorem contrapositive_of_Ms_Thompsons_statement (h : ¬Q) : P := 
by {
  -- Proof will go here
  sorry 
}

end Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l1942_194286


namespace second_caterer_cheaper_l1942_194265

theorem second_caterer_cheaper (x : ℕ) :
  (∀ n : ℕ, n < x → 150 + 18 * n ≤ 250 + 15 * n) ∧ (150 + 18 * x > 250 + 15 * x) ↔ x = 34 :=
by sorry

end second_caterer_cheaper_l1942_194265


namespace tangent_line_equation_l1942_194261

-- Define the function
def f (x : ℝ) : ℝ := x^2

-- Define the point of tangency
def x0 : ℝ := 2

-- Define the value of function at the point of tangency
def y0 : ℝ := f x0

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem tangent_line_equation : ∃ (m b : ℝ), m = f' x0 ∧ b = y0 - m * x0 ∧ ∀ x, (y = m * x + b) ↔ (x = 2 → y = f x - f' x0 * (x - 2)) :=
by
  sorry

end tangent_line_equation_l1942_194261


namespace smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l1942_194271

def is_not_prime (n : ℕ) := ¬ Prime n
def is_not_square (n : ℕ) := ∀ m : ℕ, m * m ≠ n
def no_prime_factors_less_than (n k : ℕ) := ∀ p : ℕ, Prime p → p < k → ¬ p ∣ n
def smallest_integer_prop (n : ℕ) := is_not_prime n ∧ is_not_square n ∧ no_prime_factors_less_than n 60

theorem smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60 : ∃ n : ℕ, smallest_integer_prop n ∧ n = 4087 :=
by
  sorry

end smallest_integer_neither_prime_nor_square_with_prime_factors_ge_60_l1942_194271


namespace sammy_mistakes_l1942_194244

def bryan_score : ℕ := 20
def jen_score : ℕ := bryan_score + 10
def sammy_score : ℕ := jen_score - 2
def total_points : ℕ := 35
def mistakes : ℕ := total_points - sammy_score

theorem sammy_mistakes : mistakes = 7 := by
  sorry

end sammy_mistakes_l1942_194244


namespace johns_total_earnings_l1942_194270

noncomputable def total_earnings_per_week (baskets_monday : ℕ) (baskets_thursday : ℕ) (small_crabs_per_basket : ℕ) (large_crabs_per_basket : ℕ) (price_small_crab : ℕ) (price_large_crab : ℕ) : ℕ :=
  let small_crabs := baskets_monday * small_crabs_per_basket
  let large_crabs := baskets_thursday * large_crabs_per_basket
  (small_crabs * price_small_crab) + (large_crabs * price_large_crab)

theorem johns_total_earnings :
  total_earnings_per_week 3 4 4 5 3 5 = 136 :=
by
  sorry

end johns_total_earnings_l1942_194270


namespace complex_number_calculation_l1942_194251

theorem complex_number_calculation (z : ℂ) (hz : z = 1 - I) : (z^2 / (z - 1)) = 2 := by
  sorry

end complex_number_calculation_l1942_194251


namespace f_strictly_increasing_intervals_l1942_194257

noncomputable def f (x : Real) : Real :=
  x * Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real :=
  x * Real.cos x

theorem f_strictly_increasing_intervals :
  ∀ (x : Real), (-π < x ∧ x < -π / 2 ∨ 0 < x ∧ x < π / 2) → f' x > 0 :=
by
  intros x h
  sorry

end f_strictly_increasing_intervals_l1942_194257


namespace probability_interval_contains_q_l1942_194248

theorem probability_interval_contains_q (P_C P_D : ℝ) (q : ℝ)
    (hC : P_C = 5 / 7) (hD : P_D = 3 / 4) :
    (5 / 28 ≤ q ∧ q ≤ 5 / 7) ↔ (max (P_C + P_D - 1) 0 ≤ q ∧ q ≤ min P_C P_D) :=
by
  sorry

end probability_interval_contains_q_l1942_194248


namespace games_draw_fraction_l1942_194290

-- Definitions from the conditions in the problems
def ben_win_fraction : ℚ := 4 / 9
def tom_win_fraction : ℚ := 1 / 3

-- The theorem we want to prove
theorem games_draw_fraction : 1 - (ben_win_fraction + (1 / 3)) = 2 / 9 := by
  sorry

end games_draw_fraction_l1942_194290


namespace sum_le_six_l1942_194225

theorem sum_le_six (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
    (h3 : ∃ (r s : ℤ), r * s = a + b ∧ r + s = ab) : a + b ≤ 6 :=
sorry

end sum_le_six_l1942_194225


namespace honey_barrel_problem_l1942_194216

theorem honey_barrel_problem
  (x y : ℝ)
  (h1 : x + y = 56)
  (h2 : x / 2 + y = 34) :
  x = 44 ∧ y = 12 :=
by
  sorry

end honey_barrel_problem_l1942_194216


namespace find_value_l1942_194264

theorem find_value (m n : ℤ) (h : 2 * m + n - 2 = 0) : 2 * m + n + 1 = 3 :=
by { sorry }

end find_value_l1942_194264


namespace max_product_min_quotient_l1942_194263

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end max_product_min_quotient_l1942_194263


namespace automobile_travel_distance_l1942_194299

theorem automobile_travel_distance 
  (a r : ℝ) 
  (travel_rate : ℝ) (h1 : travel_rate = a / 6)
  (time_in_seconds : ℝ) (h2 : time_in_seconds = 180):
  (3 * time_in_seconds * travel_rate) * (1 / r) * (1 / 3) = 10 * a / r :=
by
  sorry

end automobile_travel_distance_l1942_194299


namespace unique_solution_zmod_11_l1942_194230

theorem unique_solution_zmod_11 : 
  ∀ (n : ℕ), 
  (2 ≤ n → 
  (∀ x : ZMod n, (x^2 - 3 * x + 5 = 0) → (∃! x : ZMod n, x^2 - (3 : ZMod n) * x + (5 : ZMod n) = 0)) → 
  n = 11) := 
by
  sorry

end unique_solution_zmod_11_l1942_194230


namespace polygon_area_is_nine_l1942_194204

-- Definitions of vertices and coordinates.
def vertexA := (0, 0)
def vertexD := (3, 0)
def vertexP := (3, 3)
def vertexM := (0, 3)

-- Area of the polygon formed by the vertices A, D, P, M.
def polygonArea (A D P M : ℕ × ℕ) : ℕ :=
  (D.1 - A.1) * (P.2 - A.2)

-- Statement of the theorem.
theorem polygon_area_is_nine : polygonArea vertexA vertexD vertexP vertexM = 9 := by
  sorry

end polygon_area_is_nine_l1942_194204


namespace Mikey_leaves_l1942_194233

theorem Mikey_leaves (initial_leaves : ℕ) (leaves_blew_away : ℕ) 
  (h1 : initial_leaves = 356) 
  (h2 : leaves_blew_away = 244) : 
  initial_leaves - leaves_blew_away = 112 :=
by
  -- proof steps would go here
  sorry

end Mikey_leaves_l1942_194233


namespace smallest_constant_obtuse_triangle_l1942_194222

theorem smallest_constant_obtuse_triangle (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^2 > b^2 + c^2) → (b^2 + c^2) / (a^2) ≥ 1 / 2 :=
by 
  sorry

end smallest_constant_obtuse_triangle_l1942_194222


namespace area_of_triangle_PF1F2_l1942_194201

noncomputable def ellipse := {P : ℝ × ℝ // (4 * P.1^2) / 49 + (P.2^2) / 6 = 1}

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) :=
  1 / 2 * abs ((F1.1 - P.1) * (F2.2 - P.2) - (F1.2 - P.2) * (F2.1 - P.1))

theorem area_of_triangle_PF1F2 :
  ∀ (F1 F2 : ℝ × ℝ) (P : ellipse), 
    (dist P.1 F1 = 4) →
    (dist P.1 F2 = 3) →
    (dist F1 F2 = 5) →
    area_triangle P.1 F1 F2 = 6 :=
by sorry

end area_of_triangle_PF1F2_l1942_194201


namespace minimum_balls_l1942_194253

/-- Given that tennis balls are stored in big boxes containing 25 balls each 
    and small boxes containing 20 balls each, and the least number of balls 
    that can be left unboxed is 5, prove that the least number of 
    freshly manufactured balls is 105.
-/
theorem minimum_balls (B S : ℕ) : 
  ∃ (n : ℕ), 25 * B + 20 * S = n ∧ n % 25 = 5 ∧ n % 20 = 5 ∧ n = 105 := 
sorry

end minimum_balls_l1942_194253


namespace solution_for_g0_l1942_194215

variable (g : ℝ → ℝ)

def functional_eq_condition := ∀ x y : ℝ, g (x + y) = g x + g y - 1

theorem solution_for_g0 (h : functional_eq_condition g) : g 0 = 1 :=
by {
  sorry
}

end solution_for_g0_l1942_194215


namespace min_value_of_a_l1942_194252

theorem min_value_of_a (a : ℝ) : 
  (∀ x > 1, x + a / (x - 1) ≥ 5) → a ≥ 4 :=
sorry

end min_value_of_a_l1942_194252


namespace inverse_proposition_of_parallel_lines_l1942_194297

theorem inverse_proposition_of_parallel_lines 
  (P : Prop) (Q : Prop) 
  (h : P ↔ Q) : 
  (Q ↔ P) :=
by 
  sorry

end inverse_proposition_of_parallel_lines_l1942_194297


namespace largest_5_digit_congruent_15_mod_24_l1942_194220

theorem largest_5_digit_congruent_15_mod_24 : ∃ x, 10000 ≤ x ∧ x < 100000 ∧ x % 24 = 15 ∧ x = 99999 := by
  sorry

end largest_5_digit_congruent_15_mod_24_l1942_194220


namespace robin_initial_gum_is_18_l1942_194221

-- Defining the conditions as given in the problem
def given_gum : ℝ := 44
def total_gum : ℝ := 62

-- Statement to prove that the initial number of pieces of gum Robin had is 18
theorem robin_initial_gum_is_18 : total_gum - given_gum = 18 := by
  -- Proof goes here
  sorry

end robin_initial_gum_is_18_l1942_194221


namespace largest_N_exists_l1942_194256

noncomputable def parabola_properties (a T : ℤ) :=
    (∀ (x y : ℤ), y = a * x * (x - 2 * T) → (x = 0 ∨ x = 2 * T) → y = 0) ∧ 
    (∀ (v : ℤ × ℤ), v = (2 * T + 1, 28) → 28 = a * (2 * T + 1))

theorem largest_N_exists : 
    ∃ (a T : ℤ), T ≠ 0 ∧ (∀ (P : ℤ × ℤ), P = (0, 0) ∨ P = (2 * T, 0) ∨ P = (2 * T + 1, 28)) 
    ∧ (s = T - a * T^2) ∧ s = 60 :=
sorry

end largest_N_exists_l1942_194256


namespace find_n_value_l1942_194281

theorem find_n_value (n : ℕ) (h : ∃ k : ℤ, n^2 + 5 * n + 13 = k^2) : n = 4 :=
by
  sorry

end find_n_value_l1942_194281


namespace blueberries_count_l1942_194240

theorem blueberries_count (total_berries raspberries blackberries blueberries : ℕ)
  (h1 : total_berries = 42)
  (h2 : raspberries = total_berries / 2)
  (h3 : blackberries = total_berries / 3)
  (h4 : blueberries = total_berries - raspberries - blackberries) :
  blueberries = 7 :=
sorry

end blueberries_count_l1942_194240


namespace inequality_relationship_l1942_194293

noncomputable def even_function_periodic_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x + 2) = f x) ∧
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 1 → f x1 > f x2)

theorem inequality_relationship (f : ℝ → ℝ) (h : even_function_periodic_decreasing f) : 
  f (-1) < f (2.5) ∧ f (2.5) < f 0 :=
by 
  sorry

end inequality_relationship_l1942_194293


namespace free_throws_count_l1942_194234

-- Given conditions:
variables (a b x : ℕ) -- α is an abbreviation for natural numbers

-- Condition: number of points from all shots
axiom points_condition : 2 * a + 3 * b + x = 79
-- Condition: three-point shots are twice the points of two-point shots
axiom three_point_condition : 3 * b = 4 * a
-- Condition: number of free throws is one more than the number of two-point shots
axiom free_throw_condition : x = a + 1

-- Prove that the number of free throws is 12
theorem free_throws_count : x = 12 :=
by {
  sorry
}

end free_throws_count_l1942_194234


namespace oleg_can_find_adjacent_cells_divisible_by_4_l1942_194254

theorem oleg_can_find_adjacent_cells_divisible_by_4 :
  ∀ (grid : Fin 22 → Fin 22 → ℕ),
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 22 * 22) →
  ∃ i j k l, ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ ((i = k + 1 ∨ i = k - 1) ∧ j = l)) ∧ ((grid i j + grid k l) % 4 = 0) :=
by
  sorry

end oleg_can_find_adjacent_cells_divisible_by_4_l1942_194254


namespace company_p_employees_december_l1942_194277

theorem company_p_employees_december :
  let january_employees := 434.7826086956522
  let percent_more := 0.15
  let december_employees := january_employees + (percent_more * january_employees)
  december_employees = 500 :=
by
  sorry

end company_p_employees_december_l1942_194277


namespace A_share_is_9000_l1942_194210

noncomputable def A_share_in_gain (x : ℝ) : ℝ :=
  let total_gain := 27000
  let A_investment_time := 12 * x
  let B_investment_time := 6 * 2 * x
  let C_investment_time := 4 * 3 * x
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  total_gain * A_investment_time / total_investment_time

theorem A_share_is_9000 (x : ℝ) : A_share_in_gain x = 27000 / 3 :=
by
  sorry

end A_share_is_9000_l1942_194210


namespace x_solves_quadratic_and_sum_is_75_l1942_194291

theorem x_solves_quadratic_and_sum_is_75
  (x a b : ℕ) (h : x^2 + 10 * x = 45) (hx_pos : 0 < x) (hx_form : x = Nat.sqrt a - b) 
  (ha_pos : 0 < a) (hb_pos : 0 < b)
  : a + b = 75 := 
sorry

end x_solves_quadratic_and_sum_is_75_l1942_194291


namespace length_of_train_l1942_194255

noncomputable def speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def total_distance (speed_m_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_s * time_s

noncomputable def train_length (total_distance : ℝ) (bridge_length : ℝ) : ℝ :=
  total_distance - bridge_length

theorem length_of_train
  (speed_kmh : ℝ)
  (time_s : ℝ)
  (bridge_length : ℝ)
  (speed_in_kmh : speed_kmh = 45)
  (time_in_seconds : time_s = 30)
  (length_of_bridge : bridge_length = 220.03) :
  train_length (total_distance (speed_kmh_to_ms speed_kmh) time_s) bridge_length = 154.97 :=
by
  sorry

end length_of_train_l1942_194255


namespace sum_first_23_natural_numbers_l1942_194241

theorem sum_first_23_natural_numbers :
  (23 * (23 + 1)) / 2 = 276 := 
by
  sorry

end sum_first_23_natural_numbers_l1942_194241


namespace initial_tickets_count_l1942_194283

def spent_tickets : ℕ := 5
def additional_tickets : ℕ := 10
def current_tickets : ℕ := 16

theorem initial_tickets_count (initial_tickets : ℕ) :
  initial_tickets - spent_tickets + additional_tickets = current_tickets ↔ initial_tickets = 11 :=
by
  sorry

end initial_tickets_count_l1942_194283


namespace customers_not_wanting_change_l1942_194236

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l1942_194236


namespace survey_min_people_l1942_194295

theorem survey_min_people (p : ℕ) : 
  (∃ p, ∀ k ∈ [18, 10, 5, 9], k ∣ p) → p = 90 :=
by sorry

end survey_min_people_l1942_194295


namespace total_cost_of_items_l1942_194223

variable (M R F : ℝ)
variable (h1 : 10 * M = 24 * R)
variable (h2 : F = 2 * R)
variable (h3 : F = 21)

theorem total_cost_of_items : 4 * M + 3 * R + 5 * F = 237.3 :=
by
  sorry

end total_cost_of_items_l1942_194223


namespace paving_stones_needed_l1942_194284

-- Definition for the dimensions of the paving stone and the courtyard
def paving_stone_length : ℝ := 2.5
def paving_stone_width : ℝ := 2
def courtyard_length : ℝ := 30
def courtyard_width : ℝ := 16.5

-- Compute areas
def paving_stone_area : ℝ := paving_stone_length * paving_stone_width
def courtyard_area : ℝ := courtyard_length * courtyard_width

-- The theorem to prove that the number of paving stones needed is 99
theorem paving_stones_needed :
  (courtyard_area / paving_stone_area) = 99 :=
by
  sorry

end paving_stones_needed_l1942_194284


namespace missing_number_is_eight_l1942_194235

theorem missing_number_is_eight (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  intro h
  sorry

end missing_number_is_eight_l1942_194235


namespace cylinder_height_relation_l1942_194258

theorem cylinder_height_relation (r1 r2 h1 h2 V1 V2 : ℝ) 
  (h_volumes_equal : V1 = V2)
  (h_r2_gt_r1 : r2 = 1.1 * r1)
  (h_volume_first : V1 = π * r1^2 * h1)
  (h_volume_second : V2 = π * r2^2 * h2) : 
  h1 = 1.21 * h2 :=
by 
  sorry

end cylinder_height_relation_l1942_194258


namespace complement_of_P_subset_Q_l1942_194247

-- Definitions based on conditions
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}

-- Theorem statement to prove the correct option C
theorem complement_of_P_subset_Q : {x | ¬ (x < 1)} ⊆ {x | x > -1} :=
by {
  sorry
}

end complement_of_P_subset_Q_l1942_194247


namespace gold_coins_l1942_194208

theorem gold_coins (n c : Nat) : 
  n = 9 * (c - 2) → n = 6 * c + 3 → n = 45 :=
by 
  intros h1 h2 
  sorry

end gold_coins_l1942_194208


namespace regular_polygon_perimeter_l1942_194202

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end regular_polygon_perimeter_l1942_194202


namespace remainder_sum_l1942_194245

theorem remainder_sum (x y z : ℕ) (h1 : x % 15 = 11) (h2 : y % 15 = 13) (h3 : z % 15 = 9) :
  ((2 * (x % 15) + (y % 15) + (z % 15)) % 15) = 14 :=
by
  sorry

end remainder_sum_l1942_194245


namespace marble_boxes_l1942_194279

theorem marble_boxes (m : ℕ) : 
  (720 % m = 0) ∧ (m > 1) ∧ (720 / m > 1) ↔ m = 28 := 
sorry

end marble_boxes_l1942_194279


namespace similar_triangle_angles_l1942_194278

theorem similar_triangle_angles (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : α + β/2 + γ/2 = Real.pi):
  ∃ (k : ℝ), α = k ∧ β = 2 * k ∧ γ = 4 * k ∧ k = Real.pi / 7 := 
sorry

end similar_triangle_angles_l1942_194278


namespace simplify_expression_l1942_194289

theorem simplify_expression :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := 
by
  sorry

end simplify_expression_l1942_194289


namespace common_root_values_l1942_194287

def has_common_root (p x : ℝ) : Prop :=
  (x^2 - (p+1)*x + (p+1) = 0) ∧ (2*x^2 + (p-2)*x - p - 7 = 0)

theorem common_root_values :
  (has_common_root 3 2) ∧ (has_common_root (-3/2) (-1)) :=
by {
  sorry
}

end common_root_values_l1942_194287


namespace range_of_x_l1942_194282

noncomputable def f (x : ℝ) : ℝ := x * (2^x - 1 / 2^x)

theorem range_of_x (x : ℝ) (h : f (x - 1) > f x) : x < 1 / 2 :=
by sorry

end range_of_x_l1942_194282


namespace probability_of_diamond_ace_joker_l1942_194294

noncomputable def probability_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  event_cards / total_cards

noncomputable def probability_not_event (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_event total_cards event_cards

noncomputable def probability_none_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  (probability_not_event total_cards event_cards) * (probability_not_event total_cards event_cards)

noncomputable def probability_at_least_one_event_two_trials (total_cards : ℕ) (event_cards : ℕ) : ℚ :=
  1 - probability_none_event_two_trials total_cards event_cards

theorem probability_of_diamond_ace_joker 
  (total_cards : ℕ := 54) (event_cards : ℕ := 18) :
  probability_at_least_one_event_two_trials total_cards event_cards = 5 / 9 :=
by
  sorry

end probability_of_diamond_ace_joker_l1942_194294


namespace probability_complement_l1942_194217

theorem probability_complement (P_A : ℝ) (h : P_A = 0.992) : 1 - P_A = 0.008 := by
  sorry

end probability_complement_l1942_194217


namespace no_solution_in_positive_integers_l1942_194267

theorem no_solution_in_positive_integers
    (x y : ℕ)
    (h : x > 0 ∧ y > 0) :
    x^2006 - 4 * y^2006 - 2006 ≠ 4 * y^2007 + 2007 * y :=
by
  sorry

end no_solution_in_positive_integers_l1942_194267


namespace smallest_number_with_2020_divisors_l1942_194276

theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, 
  (∀ n : ℕ, (∃ (p : ℕ) (α : ℕ), n = p^α) → 
  ∃ (p1 p2 p3 p4 : ℕ) (α1 α2 α3 α4 : ℕ), 
  n = p1^α1 * p2^α2 * p3^α3 * p4^α4 ∧ 
  (α1 + 1) * (α2 + 1) * (α3 + 1) * (α4 + 1) = 2020) → 
  n = 2^100 * 3^4 * 5 * 7 :=
sorry

end smallest_number_with_2020_divisors_l1942_194276


namespace students_class_division_l1942_194280

theorem students_class_division (n : ℕ) (h1 : n % 15 = 0) (h2 : n % 24 = 0) : n = 120 :=
sorry

end students_class_division_l1942_194280


namespace solve_quadratic_equation_l1942_194274

theorem solve_quadratic_equation :
  ∃ x : ℝ, 2 * x^2 = 4 * x - 1 ∧ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
by
  sorry

end solve_quadratic_equation_l1942_194274


namespace unique_solution_implies_d_999_l1942_194206

variable (a b c d x y : ℤ)

theorem unique_solution_implies_d_999
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : 3 * x + y = 3005)
  (h5 : y = |x-a| + |x-b| + |x-c| + |x-d|)
  (h6 : ∃! x, 3 * x + |x-a| + |x-b| + |x-c| + |x-d| = 3005) :
  d = 999 :=
sorry

end unique_solution_implies_d_999_l1942_194206


namespace correct_input_statement_l1942_194209

-- Definitions based on the conditions
def input_format_A : Prop := sorry
def input_format_B : Prop := sorry
def input_format_C : Prop := sorry
def output_format_D : Prop := sorry

-- The main statement we need to prove
theorem correct_input_statement : input_format_A ∧ ¬ input_format_B ∧ ¬ input_format_C ∧ ¬ output_format_D := 
by sorry

end correct_input_statement_l1942_194209


namespace ferris_wheel_seats_l1942_194212

theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) (h1 : total_people = 16) (h2 : people_per_seat = 4) : (total_people / people_per_seat) = 4 := by
  sorry

end ferris_wheel_seats_l1942_194212


namespace average_first_three_numbers_l1942_194238

theorem average_first_three_numbers (A B C D : ℝ) 
  (hA : A = 33) 
  (hD : D = 18)
  (hBCD : (B + C + D) / 3 = 15) : 
  (A + B + C) / 3 = 20 := 
by 
  sorry

end average_first_three_numbers_l1942_194238


namespace pioneers_club_attendance_l1942_194203

theorem pioneers_club_attendance :
  ∃ (A B : (Fin 11)), A ≠ B ∧
  (∃ (clubs_A clubs_B : Finset (Fin 5)), clubs_A = clubs_B) :=
by
  sorry

end pioneers_club_attendance_l1942_194203


namespace find_B_l1942_194211

noncomputable def g (A B C D x : ℝ) : ℝ :=
  A * x^3 + B * x^2 + C * x + D

theorem find_B (A C D : ℝ) (h1 : ∀ x, g A (-2) C D x = A * (x + 2) * (x - 1) * (x - 2)) 
  (h2 : g A (-2) C D 0 = -8) : 
  (-2 : ℝ) = -2 := 
by
  simp [g] at h2
  sorry

end find_B_l1942_194211


namespace find_y_l1942_194232

-- Hypotheses
variable (x y : ℤ)

-- Given conditions
def condition1 : Prop := x = 4
def condition2 : Prop := x + y = 0

-- The goal is to prove y = -4 given the conditions
theorem find_y (h1 : condition1 x) (h2 : condition2 x y) : y = -4 := by
  sorry

end find_y_l1942_194232


namespace solve_quadratic_l1942_194213

theorem solve_quadratic (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by
  sorry

end solve_quadratic_l1942_194213


namespace parallelogram_base_length_l1942_194296

theorem parallelogram_base_length (Area Height : ℝ) (h1 : Area = 216) (h2 : Height = 18) : 
  Area / Height = 12 := 
by 
  sorry

end parallelogram_base_length_l1942_194296
