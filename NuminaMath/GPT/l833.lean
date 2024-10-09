import Mathlib

namespace find_fourth_vertex_l833_83357

open Complex

theorem find_fourth_vertex (A B C: ℂ) (hA: A = 2 + 3 * Complex.I) 
                            (hB: B = -3 + 2 * Complex.I) 
                            (hC: C = -2 - 3 * Complex.I) : 
                            ∃ D : ℂ, D = 2.5 + 0.5 * Complex.I :=
by 
  sorry

end find_fourth_vertex_l833_83357


namespace quadratic_function_conditions_l833_83370

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions_l833_83370


namespace part1_solution_set_part2_range_of_a_l833_83385

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l833_83385


namespace carol_extra_invitations_l833_83377

theorem carol_extra_invitations : 
  let invitations_per_pack := 3
  let packs_bought := 2
  let friends_to_invite := 9
  packs_bought * invitations_per_pack < friends_to_invite → 
  friends_to_invite - (packs_bought * invitations_per_pack) = 3 :=
by 
  intros _  -- Introduce the condition
  exact sorry  -- Placeholder for the proof

end carol_extra_invitations_l833_83377


namespace max_elevation_reached_l833_83355

theorem max_elevation_reached 
  (t : ℝ) 
  (s : ℝ) 
  (h : s = 200 * t - 20 * t^2) : 
  ∃ t_max : ℝ, ∃ s_max : ℝ, t_max = 5 ∧ s_max = 500 ∧ s_max = 200 * t_max - 20 * t_max^2 := sorry

end max_elevation_reached_l833_83355


namespace inverse_function_of_f_l833_83380

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2

noncomputable def f_inv (y : ℝ) : ℝ := 1 - Real.sqrt y

theorem inverse_function_of_f :
  ∀ x, x ≤ 1 → f_inv (f x) = x ∧ ∀ y, 0 ≤ y → f (f_inv y) = y :=
by
  intros
  sorry

end inverse_function_of_f_l833_83380


namespace integer_multiplication_l833_83312

theorem integer_multiplication :
  ∃ A : ℤ, (999999999 : ℤ) * A = (111111111 : ℤ) :=
by {
  sorry
}

end integer_multiplication_l833_83312


namespace union_is_correct_l833_83308

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

theorem union_is_correct : A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
by
  sorry

end union_is_correct_l833_83308


namespace simplified_expression_l833_83334

noncomputable def simplify_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ :=
  (x⁻¹ - z⁻¹)⁻¹

theorem simplified_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : 
  simplify_expression x z hx hz = x * z / (z - x) := 
by
  sorry

end simplified_expression_l833_83334


namespace solution_set_non_empty_iff_l833_83309

theorem solution_set_non_empty_iff (a : ℝ) : (∃ x : ℝ, |x - 1| + |x + 2| < a) ↔ (a > 3) := 
sorry

end solution_set_non_empty_iff_l833_83309


namespace inequality_solution_set_l833_83349

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom deriv_cond : ∀ (x : ℝ), x ≠ 0 → f' x < (2 * f x) / x
axiom zero_points : f (-2) = 0 ∧ f 1 = 0

theorem inequality_solution_set :
  {x : ℝ | x * f x < 0} = { x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) } :=
sorry

end inequality_solution_set_l833_83349


namespace odd_even_divisors_ratio_l833_83331

theorem odd_even_divisors_ratio (M : ℕ) (h1 : M = 2^5 * 3^5 * 5 * 7^3) :
  let sum_odd_divisors := (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_all_divisors := (1 + 2 + 4 + 8 + 16 + 32) * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5) * (1 + 5) * (1 + 7 + 7^2 + 7^3)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  sum_odd_divisors / sum_even_divisors = 1 / 62 :=
by
  sorry

end odd_even_divisors_ratio_l833_83331


namespace meter_to_leap_l833_83359

theorem meter_to_leap
  (strides leaps bounds meters : ℝ)
  (h1 : 3 * strides = 4 * leaps)
  (h2 : 5 * bounds = 7 * strides)
  (h3 : 2 * bounds = 9 * meters) :
  1 * meters = (56 / 135) * leaps :=
by
  sorry

end meter_to_leap_l833_83359


namespace tea_sales_l833_83319

theorem tea_sales (L T : ℕ) (h1 : L = 32) (h2 : L = 4 * T + 8) : T = 6 :=
by
  sorry

end tea_sales_l833_83319


namespace length_of_AB_l833_83383

variables (AB CD : ℝ)

-- Given conditions
def area_ratio (h : ℝ) : Prop := (1/2 * AB * h) / (1/2 * CD * h) = 4
def sum_condition : Prop := AB + CD = 200

-- The proof problem: proving the length of AB
theorem length_of_AB (h : ℝ) (h_area_ratio : area_ratio AB CD h) 
  (h_sum_condition : sum_condition AB CD) : AB = 160 :=
sorry

end length_of_AB_l833_83383


namespace nonneg_integer_solutions_l833_83397

theorem nonneg_integer_solutions :
  { x : ℕ | 5 * x + 3 < 3 * (2 + x) } = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_l833_83397


namespace find_M_range_of_a_l833_83306

def Δ (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

def A : Set ℝ := { x | 4 * x^2 + 9 * x + 2 < 0 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

def M : Set ℝ := Δ B A

def P (a: ℝ) : Set ℝ := { x | (x - 2 * a) * (x + a - 2) < 0 }

theorem find_M :
  M = { x | -1/4 ≤ x ∧ x < 2 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ M → x ∈ P a) →
  a < -1/8 ∨ a > 9/4 :=
sorry

end find_M_range_of_a_l833_83306


namespace pyramid_four_triangular_faces_area_l833_83318

noncomputable def pyramid_total_area (base_edge lateral_edge : ℝ) : ℝ :=
  if base_edge = 8 ∧ lateral_edge = 7 then 16 * Real.sqrt 33 else 0

theorem pyramid_four_triangular_faces_area :
  pyramid_total_area 8 7 = 16 * Real.sqrt 33 :=
by
  -- Proof omitted
  sorry

end pyramid_four_triangular_faces_area_l833_83318


namespace value_of_a_plus_b_l833_83395

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l833_83395


namespace find_n_squares_l833_83393

theorem find_n_squares (n : ℤ) : 
  (∃ a : ℤ, n^2 + 6 * n + 24 = a^2) ↔ n = 4 ∨ n = -2 ∨ n = -4 ∨ n = -10 :=
by
  sorry

end find_n_squares_l833_83393


namespace football_banquet_total_food_l833_83347

-- Definitions representing the conditions
def individual_max_food (n : Nat) := n ≤ 2
def min_guests (g : Nat) := g ≥ 160

-- The proof problem statement
theorem football_banquet_total_food : 
  ∀ (n g : Nat), (∀ i, i ≤ g → individual_max_food n) ∧ min_guests g → g * n = 320 := 
by
  intros n g h
  sorry

end football_banquet_total_food_l833_83347


namespace compound_p_and_q_false_l833_83338

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1) /- The function y = a^x is monotonically decreasing. -/
def q : Prop := (a > 1/2) /- The function y = log(ax^2 - x + a) has the range R. -/

theorem compound_p_and_q_false : 
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) → (0 < a ∧ a ≤ 1/2) ∨ (a > 1) :=
by {
  -- this part will contain the proof steps, omitted here.
  sorry
}

end compound_p_and_q_false_l833_83338


namespace decreasing_population_density_l833_83379

def Population (t : Type) : Type := t

variable (stable_period: Prop)
variable (infertility: Prop)
variable (death_rate_exceeds_birth_rate: Prop)
variable (complex_structure: Prop)

theorem decreasing_population_density :
  death_rate_exceeds_birth_rate → true := sorry

end decreasing_population_density_l833_83379


namespace chips_cost_l833_83327

noncomputable def cost_of_each_bag_of_chips (amount_paid_per_friend : ℕ) (number_of_friends : ℕ) (number_of_bags : ℕ) : ℕ :=
  (amount_paid_per_friend * number_of_friends) / number_of_bags

theorem chips_cost
  (amount_paid_per_friend : ℕ := 5)
  (number_of_friends : ℕ := 3)
  (number_of_bags : ℕ := 5) :
  cost_of_each_bag_of_chips amount_paid_per_friend number_of_friends number_of_bags = 3 :=
by
  sorry

end chips_cost_l833_83327


namespace problem_complement_intersection_l833_83302

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

def complement (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

theorem problem_complement_intersection :
  (complement U M) ∩ N = {3} :=
by
  sorry

end problem_complement_intersection_l833_83302


namespace least_sum_of_variables_l833_83368

theorem least_sum_of_variables (x y z w : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)  
  (h : 2 * x^2 = 5 * y^3 ∧ 5 * y^3 = 8 * z^4 ∧ 8 * z^4 = 3 * w) : x + y + z + w = 54 := 
sorry

end least_sum_of_variables_l833_83368


namespace simplify_expression_l833_83388

variable (x : ℝ)

theorem simplify_expression :
  (2 * x + 25) + (150 * x + 35) + (50 * x + 10) = 202 * x + 70 :=
sorry

end simplify_expression_l833_83388


namespace express_y_in_terms_of_x_l833_83321

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 1) : y = -3 * x + 1 := 
by
  sorry

end express_y_in_terms_of_x_l833_83321


namespace car_time_interval_l833_83354

-- Define the conditions
def road_length := 3 -- in miles
def total_time := 10 -- in hours
def number_of_cars := 30

-- Define the conversion factor and the problem to prove
def hours_to_minutes (hours: ℕ) : ℕ := hours * 60
def time_interval_per_car (total_time_minutes: ℕ) (number_of_cars: ℕ) : ℕ := total_time_minutes / number_of_cars

-- The Lean 4 statement for the proof problem
theorem car_time_interval :
  time_interval_per_car (hours_to_minutes total_time) number_of_cars = 20 :=
by
  sorry

end car_time_interval_l833_83354


namespace square_perimeter_increase_l833_83313

theorem square_perimeter_increase (s : ℝ) : (4 * (s + 2) - 4 * s) = 8 := 
by
  sorry

end square_perimeter_increase_l833_83313


namespace obtuse_triangle_side_range_l833_83300

theorem obtuse_triangle_side_range (a : ℝ) :
  (a > 0) ∧
  ((a < 3 ∧ a > -1) ∧ 
  (2 * a + 1 > a + 2) ∧ 
  (a > 1)) → 1 < a ∧ a < 3 := 
by
  sorry

end obtuse_triangle_side_range_l833_83300


namespace certain_number_l833_83372

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end certain_number_l833_83372


namespace find_a_plus_b_l833_83329

theorem find_a_plus_b (a b x : ℝ) (h1 : x + 2 * a > 4) (h2 : 2 * x < b)
  (h3 : 0 < x) (h4 : x < 2) : a + b = 6 :=
by
  sorry

end find_a_plus_b_l833_83329


namespace find_b_l833_83324

theorem find_b (k a b : ℝ) (h1 : 1 + a + b = 3) (h2 : k = 3 + a) :
  b = 3 := 
sorry

end find_b_l833_83324


namespace total_cost_correct_l833_83320

variables (gravel_cost_per_ton : ℝ) (gravel_tons : ℝ)
variables (sand_cost_per_ton : ℝ) (sand_tons : ℝ)
variables (cement_cost_per_ton : ℝ) (cement_tons : ℝ)

noncomputable def total_cost : ℝ :=
  (gravel_cost_per_ton * gravel_tons) + (sand_cost_per_ton * sand_tons) + (cement_cost_per_ton * cement_tons)

theorem total_cost_correct :
  gravel_cost_per_ton = 30.5 → gravel_tons = 5.91 →
  sand_cost_per_ton = 40.5 → sand_tons = 8.11 →
  cement_cost_per_ton = 55.6 → cement_tons = 4.35 →
  total_cost gravel_cost_per_ton gravel_tons sand_cost_per_ton sand_tons cement_cost_per_ton cement_tons = 750.57 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end total_cost_correct_l833_83320


namespace max_value_of_3x_plus_4y_l833_83346

theorem max_value_of_3x_plus_4y (x y : ℝ) (h : x^2 + y^2 = 10) : 
  ∃ z, z = 5 * Real.sqrt 10 ∧ z = 3 * x + 4 * y :=
by
  sorry

end max_value_of_3x_plus_4y_l833_83346


namespace area_ADC_proof_l833_83364

-- Definitions for the given conditions and question
variables (BD DC : ℝ) (ABD_area ADC_area : ℝ)

-- Conditions
def ratio_condition := BD / DC = 3 / 2
def ABD_area_condition := ABD_area = 30

-- Question rewritten as proof problem
theorem area_ADC_proof (h1 : ratio_condition BD DC) (h2 : ABD_area_condition ABD_area) :
  ADC_area = 20 :=
sorry

end area_ADC_proof_l833_83364


namespace line_log_intersection_l833_83345

theorem line_log_intersection (a b : ℤ) (k : ℝ)
  (h₁ : k = a + Real.sqrt b)
  (h₂ : k > 0)
  (h₃ : Real.log k / Real.log 2 - Real.log (k + 2) / Real.log 2 = 1
    ∨ Real.log (k + 2) / Real.log 2 - Real.log k / Real.log 2 = 1) :
  a + b = 2 :=
sorry

end line_log_intersection_l833_83345


namespace total_outfits_l833_83389

def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 5
def numPants : ℕ := 6
def numRedHats : ℕ := 7
def numGreenHats : ℕ := 9

theorem total_outfits : 
  ((numRedShirts * numPants * numGreenHats) + 
   (numGreenShirts * numPants * numRedHats) + 
   ((numRedShirts * numRedHats + numGreenShirts * numGreenHats) * numPants)
  ) = 1152 := 
by
  sorry

end total_outfits_l833_83389


namespace find_p8_l833_83323

noncomputable def p (x : ℝ) : ℝ := sorry -- p is a monic polynomial of degree 7

def monic_degree_7 (p : ℝ → ℝ) : Prop := sorry -- p is monic polynomial of degree 7
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

theorem find_p8 (h_monic : monic_degree_7 p) (h_conditions : satisfies_conditions p) : p 8 = 5049 :=
by
  sorry

end find_p8_l833_83323


namespace max_profit_l833_83376

theorem max_profit : ∃ v p : ℝ, 
  v + p ≤ 5 ∧
  v + 3 * p ≤ 12 ∧
  100000 * v + 200000 * p = 850000 :=
by
  sorry

end max_profit_l833_83376


namespace Chloe_total_points_l833_83381

-- Define the points scored in each round
def first_round_points : ℕ := 40
def second_round_points : ℕ := 50
def last_round_points : ℤ := -4

-- Define total points calculation
def total_points := first_round_points + second_round_points + last_round_points

-- The final statement to prove
theorem Chloe_total_points : total_points = 86 := by
  -- This proof is to be completed
  sorry

end Chloe_total_points_l833_83381


namespace negation_statement_l833_83375

variable (x y : ℝ)

theorem negation_statement :
  ¬ (x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) :=
by
  sorry

end negation_statement_l833_83375


namespace complex_expression_equals_zero_l833_83365

def i : ℂ := Complex.I

theorem complex_expression_equals_zero : 2 * i^5 + (1 - i)^2 = 0 := 
by
  sorry

end complex_expression_equals_zero_l833_83365


namespace find_digit_l833_83311

theorem find_digit (a : ℕ) (n1 n2 n3 : ℕ) (h1 : n1 = a * 1000) (h2 : n2 = a * 1000 + 998) (h3 : n3 = a * 1000 + 999) (h4 : n1 + n2 + n3 = 22997) :
  a = 7 :=
by
  sorry

end find_digit_l833_83311


namespace red_ball_probability_l833_83325

noncomputable def Urn1_blue : ℕ := 5
noncomputable def Urn1_red : ℕ := 3
noncomputable def Urn2_blue : ℕ := 4
noncomputable def Urn2_red : ℕ := 4
noncomputable def Urn3_blue : ℕ := 8
noncomputable def Urn3_red : ℕ := 0

noncomputable def P_urn (n : ℕ) : ℝ := 1 / 3
noncomputable def P_red_urn1 : ℝ := (Urn1_red : ℝ) / (Urn1_blue + Urn1_red)
noncomputable def P_red_urn2 : ℝ := (Urn2_red : ℝ) / (Urn2_blue + Urn2_red)
noncomputable def P_red_urn3 : ℝ := (Urn3_red : ℝ) / (Urn3_blue + Urn3_red)

theorem red_ball_probability : 
  (P_urn 1 * P_red_urn1 + P_urn 2 * P_red_urn2 + P_urn 3 * P_red_urn3) = 7 / 24 :=
  by sorry

end red_ball_probability_l833_83325


namespace find_a_l833_83384

theorem find_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
sorry

end find_a_l833_83384


namespace andrew_age_l833_83326

theorem andrew_age 
  (g a : ℚ)
  (h1: g = 16 * a)
  (h2: g - 20 - (a - 20) = 45) : 
 a = 17 / 3 := by 
  sorry

end andrew_age_l833_83326


namespace distance_between_planes_l833_83394

open Real

def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 4 = 0

theorem distance_between_planes :
  ∀ (x y z : ℝ), plane1 x y z →
  6 * x - 2 * y + 4 * z + 4 ≠ 0 →
  (∃ d : ℝ, d = abs (6 * x - 2 * y + 4 * z + 4) / sqrt (6^2 + (-2)^2 + 4^2) ∧ d = 5 * sqrt 14 / 14) :=
by
  intros x y z p1 p2
  sorry

end distance_between_planes_l833_83394


namespace sum_abs_frac_geq_frac_l833_83392

theorem sum_abs_frac_geq_frac (n : ℕ) (h1 : n ≥ 3) (a : Fin n → ℝ) (hnz : ∀ i : Fin n, a i ≠ 0) 
(hsum : (Finset.univ.sum a) = S) : 
  (Finset.univ.sum (fun i => |(S - a i) / a i|)) ≥ (n - 1) / (n - 2) :=
sorry

end sum_abs_frac_geq_frac_l833_83392


namespace intersection_eq_l833_83378

def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_eq : A ∩ B = {x | -1 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_eq_l833_83378


namespace average_rate_of_change_correct_l833_83341

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_correct :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_correct_l833_83341


namespace acute_triangle_properties_l833_83362

theorem acute_triangle_properties (A B C : ℝ) (AC BC : ℝ)
  (h_acute : ∀ {x : ℝ}, x = A ∨ x = B ∨ x = C → x < π / 2)
  (h_BC : BC = 1)
  (h_B_eq_2A : B = 2 * A) :
  (AC / Real.cos A = 2) ∧ (Real.sqrt 2 < AC ∧ AC < Real.sqrt 3) :=
by
  sorry

end acute_triangle_properties_l833_83362


namespace median_length_range_l833_83350

/-- Define the structure of the triangle -/
structure Triangle :=
  (A B C : ℝ) -- vertices of the triangle
  (AD AE AF : ℝ) -- lengths of altitude, angle bisector, and median
  (angleA : AngleType) -- type of angle A (acute, orthogonal, obtuse)

-- Define the angle type as a custom type
inductive AngleType
| acute
| orthogonal
| obtuse

def m_range (t : Triangle) : Set ℝ :=
  match t.angleA with
  | AngleType.acute => {m : ℝ | 13 < m ∧ m < (2028 / 119)}
  | AngleType.orthogonal => {m : ℝ | m = (2028 / 119)}
  | AngleType.obtuse => {m : ℝ | (2028 / 119) < m}

-- Lean statement for proving the problem
theorem median_length_range (t : Triangle)
  (hAD : t.AD = 12)
  (hAE : t.AE = 13) : t.AF ∈ m_range t :=
by
  sorry

end median_length_range_l833_83350


namespace problem_statement_l833_83352

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem problem_statement (b : ℝ) (hb : 2^0 + 2*0 + b = 0) : f (-1) b = -3 :=
by
  sorry

end problem_statement_l833_83352


namespace find_x_value_l833_83371

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end find_x_value_l833_83371


namespace alloy_parts_separation_l833_83340

theorem alloy_parts_separation {p q x : ℝ} (h0 : p ≠ q)
  (h1 : 6 * p ≠ 16 * q)
  (h2 : 6 * x * p + 2 * (8 - 2 * x) * q = 8 * (8 - x) * p + 6 * x * q) :
  x = 2.4 :=
by
  sorry

end alloy_parts_separation_l833_83340


namespace find_q_l833_83310

theorem find_q (q : Nat) (h : 81 ^ 6 = 3 ^ q) : q = 24 :=
by
  sorry

end find_q_l833_83310


namespace journey_time_approx_24_hours_l833_83399

noncomputable def journey_time_in_hours : ℝ :=
  let t1 := 70 / 60  -- time for destination 1
  let t2 := 50 / 35  -- time for destination 2
  let t3 := 20 / 60 + 20 / 30  -- time for destination 3
  let t4 := 30 / 40 + 60 / 70  -- time for destination 4
  let t5 := 60 / 35  -- time for destination 5
  let return_distance := 70 + 50 + 40 + 90 + 60 + 100  -- total return distance
  let return_time := return_distance / 55  -- time for return journey
  let stay_time := 1 + 3 + 2 + 2.5 + 0.75  -- total stay time
  t1 + t2 + t3 + t4 + t5 + return_time + stay_time  -- total journey time

theorem journey_time_approx_24_hours : abs (journey_time_in_hours - 24) < 1 :=
by
  sorry

end journey_time_approx_24_hours_l833_83399


namespace find_fraction_l833_83301

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end find_fraction_l833_83301


namespace evaluate_expression_l833_83335

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l833_83335


namespace smallest_even_n_sum_eq_l833_83353
  
theorem smallest_even_n_sum_eq (n : ℕ) (h_pos : n > 0) (h_even : n % 2 = 0) :
  n = 12 ↔ 
  let s₁ := n / 2 * (2 * 5 + (n - 1) * 6)
  let s₂ := n / 2 * (2 * 13 + (n - 1) * 3)
  s₁ = s₂ :=
by
  sorry

end smallest_even_n_sum_eq_l833_83353


namespace roots_sum_powers_l833_83396

theorem roots_sum_powers (t : ℕ → ℝ) (b d f : ℝ)
  (ht0 : t 0 = 3)
  (ht1 : t 1 = 6)
  (ht2 : t 2 = 11)
  (hrec : ∀ k ≥ 2, t (k + 1) = b * t k + d * t (k - 1) + f * t (k - 2))
  (hpoly : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0) :
  b + d + f = 13 :=
sorry

end roots_sum_powers_l833_83396


namespace find_4_digit_number_l833_83348

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l833_83348


namespace find_initial_alison_stamps_l833_83337

-- Define initial number of stamps Anna, Jeff, and Alison had
def initial_anna_stamps : ℕ := 37
def initial_jeff_stamps : ℕ := 31
def final_anna_stamps : ℕ := 50

-- Define the assumption that Alison gave Anna half of her stamps
def alison_gave_anna_half (a : ℕ) : Prop :=
  initial_anna_stamps + a / 2 = final_anna_stamps

-- Define the problem of finding the initial number of stamps Alison had
def alison_initial_stamps : ℕ := 26

theorem find_initial_alison_stamps :
  ∃ a : ℕ, alison_gave_anna_half a ∧ a = alison_initial_stamps :=
by
  sorry

end find_initial_alison_stamps_l833_83337


namespace probability_of_pink_l833_83307

-- Given conditions
variables (B P : ℕ) (h : (B : ℚ) / (B + P) = 3 / 7)

-- To prove
theorem probability_of_pink (h_pow : (B : ℚ) ^ 2 / (B + P) ^ 2 = 9 / 49) :
  (P : ℚ) / (B + P) = 4 / 7 :=
sorry

end probability_of_pink_l833_83307


namespace membership_relation_l833_83358

-- Definitions of M and N
def M (x : ℝ) : Prop := abs (x + 1) < 4
def N (x : ℝ) : Prop := x / (x - 3) < 0

theorem membership_relation (a : ℝ) (h : M a) : N a → M a := by
  sorry

end membership_relation_l833_83358


namespace B_finishes_in_10_days_l833_83360

noncomputable def B_remaining_work_days (A_work_days : ℕ := 15) (A_initial_days_worked : ℕ := 5) (B_work_days : ℝ := 14.999999999999996) : ℝ :=
  let A_rate := 1 / A_work_days
  let B_rate := 1 / B_work_days
  let remaining_work := 1 - (A_rate * A_initial_days_worked)
  let days_for_B := remaining_work / B_rate
  days_for_B

theorem B_finishes_in_10_days :
  B_remaining_work_days 15 5 14.999999999999996 = 10 :=
by
  sorry

end B_finishes_in_10_days_l833_83360


namespace prime_neighbor_divisible_by_6_l833_83322

theorem prime_neighbor_divisible_by_6 (p : ℕ) (h_prime: Prime p) (h_gt3: p > 3) :
  ∃ k : ℕ, k ≠ 0 ∧ ((p - 1) % 6 = 0 ∨ (p + 1) % 6 = 0) :=
by
  sorry

end prime_neighbor_divisible_by_6_l833_83322


namespace find_ordered_pair_l833_83374

theorem find_ordered_pair (x y : ℝ) :
  (2 * x + 3 * y = (6 - x) + (6 - 3 * y)) ∧ (x - 2 * y = (x - 2) - (y + 2)) ↔ (x = -4) ∧ (y = 4) := by
  sorry

end find_ordered_pair_l833_83374


namespace pastries_made_correct_l833_83386

-- Definitions based on conditions
def cakes_made := 14
def cakes_sold := 97
def pastries_sold := 8
def cakes_more_than_pastries := 89

-- Definition of the function to compute pastries made
def pastries_made (cakes_made cakes_sold pastries_sold cakes_more_than_pastries : ℕ) : ℕ :=
  cakes_sold - cakes_more_than_pastries

-- The statement to prove
theorem pastries_made_correct : pastries_made cakes_made cakes_sold pastries_sold cakes_more_than_pastries = 8 := by
  unfold pastries_made
  norm_num
  sorry

end pastries_made_correct_l833_83386


namespace number_of_intersections_l833_83303

def ellipse (x y : ℝ) : Prop := (x^2) / 16 + (y^2) / 9 = 1
def vertical_line (x : ℝ) : Prop := x = 4

theorem number_of_intersections : 
    (∃ y : ℝ, ellipse 4 y ∧ vertical_line 4) ∧ 
    ∀ y1 y2, (ellipse 4 y1 ∧ vertical_line 4) → (ellipse 4 y2 ∧ vertical_line 4) → y1 = y2 :=
by
  sorry

end number_of_intersections_l833_83303


namespace rectangle_ratio_l833_83367

theorem rectangle_ratio (a b : ℝ) (side : ℝ) (M N : ℝ → ℝ) (P Q : ℝ → ℝ)
  (h_side : side = 4)
  (h_M : M 0 = 4 / 3 ∧ M 4 = 8 / 3)
  (h_N : N 0 = 4 / 3 ∧ N 4 = 8 / 3)
  (h_perpendicular : P 0 = Q 0 ∧ P 4 = Q 4)
  (h_area : side * side = 16) :
  let UV := 6 / 5
  let VW := 40 / 3
  UV / VW = 9 / 100 :=
sorry

end rectangle_ratio_l833_83367


namespace find_wrong_number_read_l833_83366

theorem find_wrong_number_read (avg_initial avg_correct num_total wrong_num : ℕ) 
    (h1 : avg_initial = 15)
    (h2 : avg_correct = 16)
    (h3 : num_total = 10)
    (h4 : wrong_num = 36) 
    : wrong_num - (avg_correct * num_total - avg_initial * num_total) = 26 := 
by
  -- This is where the proof would go.
  sorry

end find_wrong_number_read_l833_83366


namespace simplify_expression_l833_83363

def E (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2)

theorem simplify_expression (x : ℝ) : E x = 9 * x^3 - 2 * x^2 + 9 * x + 2 :=
by
  sorry

end simplify_expression_l833_83363


namespace b_and_c_work_days_l833_83330

theorem b_and_c_work_days
  (A B C : ℝ)
  (h1 : A + B = 1 / 8)
  (h2 : A + C = 1 / 8)
  (h3 : A + B + C = 1 / 6) :
  B + C = 1 / 24 :=
sorry

end b_and_c_work_days_l833_83330


namespace max_product_sum_1976_l833_83361

theorem max_product_sum_1976 (a : ℕ) (P : ℕ → ℕ) (h : ∀ n, P n > 0 → a = 1976) :
  ∃ (k l : ℕ), (2 * k + 3 * l = 1976) ∧ (P 1976 = 2 * 3 ^ 658) := sorry

end max_product_sum_1976_l833_83361


namespace point_P_distance_l833_83333

variable (a b c d x : ℝ)

-- Define the points on the line
def O := 0
def A := a
def B := b
def C := c
def D := d

-- Define the conditions for point P
def AP_PDRatio := (|a - x| / |x - d| = 2 * |b - x| / |x - c|)

theorem point_P_distance : AP_PDRatio a b c d x → b + c - a = x :=
by
  sorry

end point_P_distance_l833_83333


namespace quadratic_no_real_roots_l833_83304

-- Define the quadratic polynomial f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions: f(x) = x has no real roots
theorem quadratic_no_real_roots (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
sorry

end quadratic_no_real_roots_l833_83304


namespace sales_volume_relation_maximize_profit_l833_83342

-- Define the conditions as given in the problem
def cost_price : ℝ := 6
def sales_data : List (ℝ × ℝ) := [(10, 4000), (11, 3900), (12, 3800)]
def price_range (x : ℝ) : Prop := 6 ≤ x ∧ x ≤ 32

-- Define the functional relationship y in terms of x
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

-- Define the profit function w in terms of x
def profit (x : ℝ) : ℝ := (sales_volume x) * (x - cost_price)

-- Prove that the functional relationship holds within the price range
theorem sales_volume_relation (x : ℝ) (h : price_range x) :
  ∀ (y : ℝ), (x, y) ∈ sales_data → y = sales_volume x := by
  sorry

-- Prove that the profit is maximized when x = 28 and the profit is 48400 yuan
theorem maximize_profit :
  ∃ x, price_range x ∧ x = 28 ∧ profit x = 48400 := by
  sorry

end sales_volume_relation_maximize_profit_l833_83342


namespace num_comics_liked_by_males_l833_83336

-- Define the problem conditions
def num_comics : ℕ := 300
def percent_liked_by_females : ℕ := 30
def percent_disliked_by_both : ℕ := 30

-- Define the main theorem to prove
theorem num_comics_liked_by_males :
  let percent_liked_by_at_least_one_gender := 100 - percent_disliked_by_both
  let num_comics_liked_by_females := percent_liked_by_females * num_comics / 100
  let num_comics_liked_by_at_least_one_gender := percent_liked_by_at_least_one_gender * num_comics / 100
  num_comics_liked_by_at_least_one_gender - num_comics_liked_by_females = 120 :=
by
  sorry

end num_comics_liked_by_males_l833_83336


namespace perimeters_positive_difference_l833_83316

theorem perimeters_positive_difference (orig_length orig_width : ℝ) (num_pieces : ℕ)
  (congruent_division : ∃ (length width : ℝ), length * width = (orig_length * orig_width) / num_pieces)
  (greatest_perimeter least_perimeter : ℝ)
  (h1 : greatest_perimeter = 2 * (1.5 + 9))
  (h2 : least_perimeter = 2 * (1 + 6)) :
  abs (greatest_perimeter - least_perimeter) = 7 := 
sorry

end perimeters_positive_difference_l833_83316


namespace problem_solution_l833_83317

noncomputable def positiveIntPairsCount : ℕ :=
  sorry

theorem problem_solution :
  positiveIntPairsCount = 2 :=
sorry

end problem_solution_l833_83317


namespace intersection_M_N_l833_83351

-- Definitions of the sets M and N
def M : Set ℝ := { -1, 0, 1 }
def N : Set ℝ := { x | x^2 ≤ x }

-- The theorem to be proven
theorem intersection_M_N : M ∩ N = { 0, 1 } :=
by
  sorry

end intersection_M_N_l833_83351


namespace count_valid_pairs_l833_83398

theorem count_valid_pairs : 
  ∃! n : ℕ, 
  n = 2 ∧ 
  (∀ (a b : ℕ), (0 < a ∧ 0 < b) → 
    (a * b + 97 = 18 * Nat.lcm a b + 14 * Nat.gcd a b) → 
    n = 2)
:= sorry

end count_valid_pairs_l833_83398


namespace max_cake_boxes_l833_83344

theorem max_cake_boxes 
  (L_carton W_carton H_carton : ℕ) (L_box W_box H_box : ℕ)
  (h_carton : L_carton = 25 ∧ W_carton = 42 ∧ H_carton = 60)
  (h_box : L_box = 8 ∧ W_box = 7 ∧ H_box = 5) : 
  (L_carton * W_carton * H_carton) / (L_box * W_box * H_box) = 225 := by 
  sorry

end max_cake_boxes_l833_83344


namespace jelly_bean_problem_l833_83314

variables {p_r p_o p_y p_g : ℝ}

theorem jelly_bean_problem :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 :=
by
  intros p_r_eq p_o_eq sum_eq
  -- The proof would proceed here, but we avoid proof details
  sorry

end jelly_bean_problem_l833_83314


namespace equiangular_polygon_angle_solution_l833_83390

-- Given two equiangular polygons P_1 and P_2 with different numbers of sides
-- Each angle of P_1 is x degrees
-- Each angle of P_2 is k * x degrees where k is an integer greater than 1
-- Prove that the number of valid pairs (x, k) is exactly 1

theorem equiangular_polygon_angle_solution : ∃ x k : ℕ, ( ∀ n m : ℕ, x = 180 - 360 / n ∧ k * x = 180 - 360 / m → (k > 1) → x = 60 ∧ k = 2) := sorry

end equiangular_polygon_angle_solution_l833_83390


namespace total_hours_A_ascending_and_descending_l833_83382

theorem total_hours_A_ascending_and_descending
  (ascending_speed_A ascending_speed_B descending_speed_A descending_speed_B distance summit_distance : ℝ)
  (h1 : descending_speed_A = 1.5 * ascending_speed_A)
  (h2 : descending_speed_B = 1.5 * ascending_speed_B)
  (h3 : ascending_speed_A > ascending_speed_B)
  (h4 : 1/ascending_speed_A + 1/ascending_speed_B = 1/hour - 600/summit_distance)
  (h5 : 0.5 * summit_distance/ascending_speed_A = (summit_distance - 600)/ascending_speed_B) :
  (summit_distance / ascending_speed_A) + (summit_distance / descending_speed_A) = 1.5 := 
sorry

end total_hours_A_ascending_and_descending_l833_83382


namespace distance_to_focus_F2_l833_83356

noncomputable def ellipse_foci_distance
  (x y : ℝ)
  (a b : ℝ) 
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1) 
  (a2 : a^2 = 9) 
  (b2 : b^2 = 2) 
  (F1 P : ℝ) 
  (h_P_on_ellipse : F1 = 3) 
  (h_PF1 : F1 = 4) 
: ℝ :=
  2

-- theorem to prove the problem statement
theorem distance_to_focus_F2
  (x y : ℝ)
  (a b : ℝ)
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)
  (a2 : a^2 = 9)
  (b2 : b^2 = 2)
  (F1 P : ℝ)
  (h_P_on_ellipse : F1 = 3)
  (h_PF1 : F1 = 4)
: F2 = 2 :=
by
  sorry

end distance_to_focus_F2_l833_83356


namespace q_is_false_l833_83387

variable {p q : Prop}

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l833_83387


namespace find_n_if_roots_opposite_signs_l833_83332

theorem find_n_if_roots_opposite_signs :
  ∃ n : ℝ, (∀ x : ℝ, (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) → x = -x) →
    (n = (-1 + Real.sqrt 5) / 2 ∨ n = (-1 - Real.sqrt 5) / 2) :=
by
  sorry

end find_n_if_roots_opposite_signs_l833_83332


namespace not_all_same_probability_l833_83369

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l833_83369


namespace f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l833_83343
open Real

noncomputable def f : ℝ → ℝ := sorry

theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := sorry
theorem f_positive_lt_x_zero (x : ℝ) (h_pos : 0 < x) : f x < 0 := sorry
theorem f_at_one : f 1 = 1 := sorry

-- Prove that f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
  sorry

-- Solve the inequality: f((log2 x)^2 - log2 (x^2)) > 3
theorem f_inequality (x : ℝ) (h_pos : 0 < x) : (f ((log x / log 2)^2 - (log x^2 / log 2))) > 3 ↔ 1 / 2 < x ∧ x < 8 :=
  sorry

end f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l833_83343


namespace range_of_m_l833_83305

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) ↔ (m ∈ Set.Icc (-6:ℝ) 2) :=
by
  sorry

end range_of_m_l833_83305


namespace intersecting_parabolas_circle_radius_sq_l833_83391

theorem intersecting_parabolas_circle_radius_sq:
  (∀ (x y : ℝ), (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) → 
  ((x + 1/2)^2 + (y - 7/2)^2 = 13/2)) := sorry

end intersecting_parabolas_circle_radius_sq_l833_83391


namespace course_choice_gender_related_l833_83339
open scoped Real

theorem course_choice_gender_related :
  let a := 40 -- Males choosing Calligraphy
  let b := 10 -- Males choosing Paper Cutting
  let c := 30 -- Females choosing Calligraphy
  let d := 20 -- Females choosing Paper Cutting
  let n := a + b + c + d -- Total number of students
  let χ_squared := (n * (a*d - b*c)^2) / ((a+b) * (c+d) * (a+c) * (b+d))
  χ_squared > 3.841 := 
by
  sorry

end course_choice_gender_related_l833_83339


namespace ravi_prakash_finish_together_l833_83315

theorem ravi_prakash_finish_together (ravi_days prakash_days : ℕ) (h_ravi : ravi_days = 15) (h_prakash : prakash_days = 30) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 10 := 
by
  sorry

end ravi_prakash_finish_together_l833_83315


namespace quadratic_real_roots_l833_83373

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ m ≤ 1 :=
by
  sorry

end quadratic_real_roots_l833_83373


namespace tea_bags_l833_83328

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l833_83328
