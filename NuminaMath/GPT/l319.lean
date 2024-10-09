import Mathlib

namespace root_expression_value_l319_31906

theorem root_expression_value 
  (p q r s : ℝ)
  (h1 : p + q + r + s = 15)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = 35)
  (h3 : p*q*r + p*q*s + q*r*s + p*r*s = 27)
  (h4 : p*q*r*s = 9)
  (h5 : ∀ x : ℝ, x^4 - 15*x^3 + 35*x^2 - 27*x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) :
  (p / (1 / p + q*r) + q / (1 / q + r*s) + r / (1 / r + s*p) + s / (1 / s + p*q) = 155 / 123) := 
sorry

end root_expression_value_l319_31906


namespace gcd_266_209_l319_31905

-- Definitions based on conditions
def a : ℕ := 266
def b : ℕ := 209

-- Theorem stating the GCD of a and b
theorem gcd_266_209 : Nat.gcd a b = 19 :=
by {
  -- Declare the specific integers as conditions
  let a := 266
  let b := 209
  -- Use the Euclidean algorithm (steps within the proof are not required)
  -- State that the conclusion is the GCD of a and b 
  sorry
}

end gcd_266_209_l319_31905


namespace fractional_equation_solution_l319_31911

theorem fractional_equation_solution (x : ℝ) (h₁ : x ≠ 0) : (1 / x = 2 / (x + 3)) → x = 3 := by
  sorry

end fractional_equation_solution_l319_31911


namespace percentage_republicans_vote_X_l319_31972

theorem percentage_republicans_vote_X (R : ℝ) (P_R : ℝ) :
  (3 * R * P_R + 2 * R * 0.15) - (3 * R * (1 - P_R) + 2 * R * 0.85) = 0.019999999999999927 * (3 * R + 2 * R) →
  P_R = 4.1 / 6 :=
by
  intro h
  sorry

end percentage_republicans_vote_X_l319_31972


namespace brick_height_l319_31941

def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem brick_height 
  (l : ℝ) (w : ℝ) (SA : ℝ) (h : ℝ) 
  (surface_area_eq : surface_area l w h = SA)
  (length_eq : l = 10)
  (width_eq : w = 4)
  (surface_area_given : SA = 164) :
  h = 3 :=
by
  sorry

end brick_height_l319_31941


namespace sara_letters_ratio_l319_31916

variable (L_J : ℕ) (L_F : ℕ) (L_T : ℕ)

theorem sara_letters_ratio (hLJ : L_J = 6) (hLF : L_F = 9) (hLT : L_T = 33) : 
  (L_T - (L_J + L_F)) / L_J = 3 := by
  sorry

end sara_letters_ratio_l319_31916


namespace minimum_value_of_quadratic_expression_l319_31908

def quadratic_expr (x y : ℝ) : ℝ := x^2 - x * y + y^2

def constraint (x y : ℝ) : Prop := x + y = 5

theorem minimum_value_of_quadratic_expression :
  ∃ m, ∀ x y, constraint x y → quadratic_expr x y ≥ m ∧ (∃ x y, constraint x y ∧ quadratic_expr x y = m) :=
sorry

end minimum_value_of_quadratic_expression_l319_31908


namespace quadratic_inequality_l319_31998

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 16 :=
by sorry

end quadratic_inequality_l319_31998


namespace different_colors_probability_l319_31973

-- Definitions of the chips in the bag
def purple_chips := 7
def green_chips := 6
def orange_chips := 5
def total_chips := purple_chips + green_chips + orange_chips

-- Calculating probabilities for drawing chips of different colors and ensuring the final probability of different colors is correct
def probability_different_colors : ℚ :=
  let P := purple_chips
  let G := green_chips
  let O := orange_chips
  let T := total_chips
  (P / T) * ((G + O) / T) + (G / T) * ((P + O) / T) + (O / T) * ((P + G) / T)

theorem different_colors_probability : probability_different_colors = (107 / 162) := by
  sorry

end different_colors_probability_l319_31973


namespace square_binomial_unique_a_l319_31917

theorem square_binomial_unique_a (a : ℝ) : 
  (∃ r s : ℝ, (ax^2 - 8*x + 16) = (r*x + s)^2) ↔ a = 1 :=
by
  sorry

end square_binomial_unique_a_l319_31917


namespace total_seeds_correct_l319_31929

def seeds_per_bed : ℕ := 6
def flower_beds : ℕ := 9
def total_seeds : ℕ := seeds_per_bed * flower_beds

theorem total_seeds_correct : total_seeds = 54 := by
  sorry

end total_seeds_correct_l319_31929


namespace product_of_d_l319_31940

theorem product_of_d (d1 d2 : ℕ) (h1 : ∃ k1 : ℤ, 49 - 12 * d1 = k1^2)
  (h2 : ∃ k2 : ℤ, 49 - 12 * d2 = k2^2) (h3 : 0 < d1) (h4 : 0 < d2)
  (h5 : d1 ≠ d2) : d1 * d2 = 8 := 
sorry

end product_of_d_l319_31940


namespace age_difference_between_Mandy_and_sister_l319_31981

variable (Mandy_age Brother_age Sister_age : ℕ)

-- Given conditions
def Mandy_is_3_years_old : Mandy_age = 3 := by sorry
def Brother_is_4_times_older : Brother_age = 4 * Mandy_age := by sorry
def Sister_is_5_years_younger_than_brother : Sister_age = Brother_age - 5 := by sorry

-- Prove the question
theorem age_difference_between_Mandy_and_sister :
  Mandy_age = 3 ∧ Brother_age = 4 * Mandy_age ∧ Sister_age = Brother_age - 5 → Sister_age - Mandy_age = 4 := 
by 
  sorry

end age_difference_between_Mandy_and_sister_l319_31981


namespace brother_birth_year_1990_l319_31991

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end brother_birth_year_1990_l319_31991


namespace area_change_factor_l319_31949

theorem area_change_factor (k b : ℝ) (hk : 0 < k) (hb : 0 < b) :
  let S1 := (b * b) / (2 * k)
  let S2 := (b * b) / (16 * k)
  S1 / S2 = 8 :=
by
  sorry

end area_change_factor_l319_31949


namespace speed_of_stream_l319_31955

theorem speed_of_stream (v : ℝ) (canoe_speed : ℝ) 
  (upstream_speed_condition : canoe_speed - v = 3) 
  (downstream_speed_condition : canoe_speed + v = 12) :
  v = 4.5 := 
by 
  sorry

end speed_of_stream_l319_31955


namespace total_people_on_bus_l319_31953

def students_left := 42
def students_right := 38
def students_back := 5
def students_aisle := 15
def teachers := 2
def bus_driver := 1

theorem total_people_on_bus : students_left + students_right + students_back + students_aisle + teachers + bus_driver = 103 :=
by
  sorry

end total_people_on_bus_l319_31953


namespace max_lcm_15_2_3_5_6_9_10_l319_31912

theorem max_lcm_15_2_3_5_6_9_10 : 
  max (max (max (max (max (Nat.lcm 15 2) (Nat.lcm 15 3)) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10) = 45 :=
by
  sorry

end max_lcm_15_2_3_5_6_9_10_l319_31912


namespace pen_price_first_day_l319_31964

theorem pen_price_first_day (x y : ℕ) 
  (h1 : x * y = (x - 1) * (y + 100)) 
  (h2 : x * y = (x + 2) * (y - 100)) : x = 4 :=
by
  sorry

end pen_price_first_day_l319_31964


namespace band_total_earnings_l319_31918

variables (earnings_per_gig_per_member : ℕ)
variables (number_of_members : ℕ)
variables (number_of_gigs : ℕ)

theorem band_total_earnings :
  earnings_per_gig_per_member = 20 →
  number_of_members = 4 →
  number_of_gigs = 5 →
  earnings_per_gig_per_member * number_of_members * number_of_gigs = 400 :=
by
  intros
  sorry

end band_total_earnings_l319_31918


namespace rahul_share_is_100_l319_31962

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100_l319_31962


namespace contrapositive_proof_l319_31976

variable {p q : Prop}

theorem contrapositive_proof : (p → q) ↔ (¬q → ¬p) :=
  by sorry

end contrapositive_proof_l319_31976


namespace no_sol_for_frac_eq_l319_31926

theorem no_sol_for_frac_eq (x y : ℕ) (h : x > 1) : ¬ (y^5 + 1 = (x^7 - 1) / (x - 1)) :=
sorry

end no_sol_for_frac_eq_l319_31926


namespace negation_of_existence_l319_31990

theorem negation_of_existence :
  ¬(∃ x : ℝ, x^2 + 2 * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l319_31990


namespace cafeteria_students_count_l319_31984

def total_students : ℕ := 90

def initial_in_cafeteria : ℕ := total_students * 2 / 3

def initial_outside : ℕ := total_students / 3

def ran_inside : ℕ := initial_outside / 3

def ran_outside : ℕ := 3

def net_change_in_cafeteria : ℕ := ran_inside - ran_outside

def final_in_cafeteria : ℕ := initial_in_cafeteria + net_change_in_cafeteria

theorem cafeteria_students_count : final_in_cafeteria = 67 := 
by
  sorry

end cafeteria_students_count_l319_31984


namespace largest_value_l319_31995

-- Definition: Given the condition of a quadratic equation
def equation (a : ℚ) : Prop :=
  8 * a^2 + 6 * a + 2 = 0

-- Theorem: Prove the largest value of 3a + 2 is 5/4 given the condition
theorem largest_value (a : ℚ) (h : equation a) : 
  ∃ m, ∀ b, equation b → (3 * b + 2 ≤ m) ∧ (m = 5 / 4) :=
by
  sorry

end largest_value_l319_31995


namespace find_fourth_digit_l319_31930

theorem find_fourth_digit (a b c d : ℕ) (h : 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8)
  (h_eq : 511 * a + 54 * b - 92 * c - 999 * d = 0) : d = 6 :=
by
  sorry

end find_fourth_digit_l319_31930


namespace find_n_l319_31956

-- Defining the conditions given in the problem
def condition_eq (n : ℝ) : Prop :=
  10 * 1.8 - (n * 1.5 / 0.3) = 50

-- Stating the goal: Prove that the number n is -6.4
theorem find_n : condition_eq (-6.4) :=
by
  -- Proof is omitted
  sorry

end find_n_l319_31956


namespace total_cost_with_discount_and_tax_l319_31924

theorem total_cost_with_discount_and_tax
  (sandwich_cost : ℝ := 2.44)
  (soda_cost : ℝ := 0.87)
  (num_sandwiches : ℕ := 2)
  (num_sodas : ℕ := 4)
  (discount : ℝ := 0.15)
  (tax_rate : ℝ := 0.09) : 
  (num_sandwiches * sandwich_cost * (1 - discount) + num_sodas * soda_cost) * (1 + tax_rate) = 8.32 :=
by
  sorry

end total_cost_with_discount_and_tax_l319_31924


namespace initial_number_of_men_l319_31959

theorem initial_number_of_men (n : ℕ) (A : ℕ)
  (h1 : 2 * n = 16)
  (h2 : 60 - 44 = 16)
  (h3 : 60 = 2 * 30)
  (h4 : 44 = 21 + 23) :
  n = 8 :=
by
  sorry

end initial_number_of_men_l319_31959


namespace percent_equality_l319_31944

theorem percent_equality :
  (1 / 4 : ℝ) * 100 = (10 / 100 : ℝ) * 250 :=
by
  sorry

end percent_equality_l319_31944


namespace length_of_other_train_l319_31963

def speed1 := 90 -- speed in km/hr
def speed2 := 90 -- speed in km/hr
def length_train1 := 1.10 -- length in km
def crossing_time := 40 -- time in seconds

theorem length_of_other_train : 
  ∀ s1 s2 l1 t l2 : ℝ,
  s1 = 90 → s2 = 90 → l1 = 1.10 → t = 40 → 
  ((s1 + s2) / 3600 * t - l1 = l2) → 
  l2 = 0.90 :=
by
  intros s1 s2 l1 t l2 hs1 hs2 hl1 ht hdist
  sorry

end length_of_other_train_l319_31963


namespace find_n_correct_l319_31938

noncomputable def find_n : Prop :=
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * (Real.pi / 180)) = Real.cos (317 * (Real.pi / 180)) → n = 43

theorem find_n_correct : find_n :=
  sorry

end find_n_correct_l319_31938


namespace intersection_A_notB_l319_31943

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A according to the given condition
def A : Set ℝ := { x | |x - 1| > 1 }

-- Define set B according to the given condition
def B : Set ℝ := { x | (x - 1) * (x - 4) > 0 }

-- Define the complement of set B in U
def notB : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Lean statement to prove A ∩ notB = { x | 2 < x ∧ x ≤ 4 }
theorem intersection_A_notB :
  A ∩ notB = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end intersection_A_notB_l319_31943


namespace upstream_speed_l319_31980

-- Speed of the man in still water
def V_m : ℕ := 32

-- Speed of the man rowing downstream
def V_down : ℕ := 42

-- Speed of the stream
def V_s : ℕ := V_down - V_m

-- Speed of the man rowing upstream
def V_up : ℕ := V_m - V_s

theorem upstream_speed (V_m : ℕ) (V_down : ℕ) (V_s : ℕ) (V_up : ℕ) : 
  V_m = 32 → 
  V_down = 42 → 
  V_s = V_down - V_m → 
  V_up = V_m - V_s → 
  V_up = 22 := 
by intros; 
   repeat {sorry}

end upstream_speed_l319_31980


namespace girls_ran_9_miles_l319_31900

def boys_laps : ℕ := 34
def additional_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6

def girls_laps : ℕ := boys_laps + additional_laps
def girls_miles : ℚ := girls_laps * lap_distance

theorem girls_ran_9_miles : girls_miles = 9 := by
  sorry

end girls_ran_9_miles_l319_31900


namespace total_bill_first_month_l319_31928

theorem total_bill_first_month (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) 
  (h3 : 2 * C = 2 * C) : 
  F + C = 50 := by
  sorry

end total_bill_first_month_l319_31928


namespace kaleb_initial_cherries_l319_31993

/-- Kaleb's initial number of cherries -/
def initial_cherries : ℕ := 67

/-- Cherries that Kaleb ate -/
def eaten_cherries : ℕ := 25

/-- Cherries left after eating -/
def left_cherries : ℕ := 42

/-- Prove that the initial number of cherries is 67 given the conditions. -/
theorem kaleb_initial_cherries :
  eaten_cherries + left_cherries = initial_cherries :=
by
  sorry

end kaleb_initial_cherries_l319_31993


namespace fixed_point_of_family_of_lines_l319_31931

theorem fixed_point_of_family_of_lines :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2 * x - m * y + 1 - 3 * m = 0) ∧ (x = -1 / 2) ∧ (y = -3) :=
by
  intro m
  use -1 / 2, -3
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end fixed_point_of_family_of_lines_l319_31931


namespace intersection_is_correct_l319_31942

def A : Set ℝ := { x | x * (x - 2) < 0 }
def B : Set ℝ := { x | Real.log x > 0 }

theorem intersection_is_correct : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_is_correct_l319_31942


namespace inequality_proof_l319_31986

theorem inequality_proof
  (a b c d : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_cond : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3) :=
by {
  sorry
}

end inequality_proof_l319_31986


namespace fraction_oil_is_correct_l319_31910

noncomputable def fraction_oil_third_bottle (C : ℚ) (oil1 : ℚ) (oil2 : ℚ) (water1 : ℚ) (water2 : ℚ) := 
  (oil1 + oil2) / (oil1 + oil2 + water1 + water2)

theorem fraction_oil_is_correct (C : ℚ) (hC : C > 0) :
  let oil1 := C / 2
  let oil2 := C / 2
  let water1 := C / 2
  let water2 := 3 * C / 4
  fraction_oil_third_bottle C oil1 oil2 water1 water2 = 4 / 9 := by
  sorry

end fraction_oil_is_correct_l319_31910


namespace x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l319_31957

theorem x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3 :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ (∃ x : ℝ, x ≥ 3 ∧ ¬ (x > 3)) :=
by
  sorry

end x_ge_3_is_necessary_but_not_sufficient_for_x_gt_3_l319_31957


namespace amount_of_bill_l319_31903

noncomputable def TD : ℝ := 360
noncomputable def BD : ℝ := 418.9090909090909
noncomputable def FV (TD BD : ℝ) : ℝ := TD * BD / (BD - TD)

theorem amount_of_bill :
  FV TD BD = 2568 :=
by
  sorry

end amount_of_bill_l319_31903


namespace max_min_values_l319_31988

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), 
    max_val = 7 ∧ min_val = -20 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max_val) ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min_val ≤ f x) := 
by
  sorry

end max_min_values_l319_31988


namespace polynomial_inequality_holds_l319_31967

theorem polynomial_inequality_holds (a : ℝ) : (∀ x : ℝ, x^4 + (a-2)*x^2 + a ≥ 0) ↔ a ≥ 4 - 2 * Real.sqrt 3 := 
by
  sorry

end polynomial_inequality_holds_l319_31967


namespace max_sum_of_circle_eq_eight_l319_31969

noncomputable def max_sum_of_integer_solutions (r : ℕ) : ℕ :=
  if r = 6 then 8 else 0

theorem max_sum_of_circle_eq_eight 
  (h1 : ∃ (x y : ℤ), (x - 1)^2 + (y - 1)^2 = 36 ∧ (r : ℕ) = 6) :
  max_sum_of_integer_solutions r = 8 := 
by
  sorry

end max_sum_of_circle_eq_eight_l319_31969


namespace s_plus_t_l319_31901

def g (x : ℝ) : ℝ := 3 * x ^ 4 + 9 * x ^ 3 - 7 * x ^ 2 + 2 * x + 4
def h (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

noncomputable def s (x : ℝ) : ℝ := 3 * x ^ 2 + 3
noncomputable def t (x : ℝ) : ℝ := 3 * x + 6

theorem s_plus_t : s 1 + t (-1) = 9 := by
  sorry

end s_plus_t_l319_31901


namespace correct_calculation_given_conditions_l319_31994

variable (number : ℤ)

theorem correct_calculation_given_conditions 
  (h : number + 16 = 64) : number - 16 = 32 := by
  sorry

end correct_calculation_given_conditions_l319_31994


namespace isabel_games_problem_l319_31958

noncomputable def prime_sum : ℕ := 83 + 89 + 97

theorem isabel_games_problem (initial_games : ℕ) (X : ℕ) (H1 : initial_games = 90) (H2 : X = prime_sum) : X > initial_games :=
by 
  sorry

end isabel_games_problem_l319_31958


namespace simplify_333_div_9999_mul_99_l319_31954

theorem simplify_333_div_9999_mul_99 :
  (333 / 9999) * 99 = 37 / 101 :=
by
  -- Sorry for skipping proof
  sorry

end simplify_333_div_9999_mul_99_l319_31954


namespace cost_of_rice_l319_31919

-- Define the cost variables
variables (E R K : ℝ)

-- State the conditions as assumptions
def conditions (E R K : ℝ) : Prop :=
  (E = R) ∧
  (K = (2 / 3) * E) ∧
  (2 * K = 48)

-- State the theorem to be proven
theorem cost_of_rice (E R K : ℝ) (h : conditions E R K) : R = 36 :=
by
  sorry

end cost_of_rice_l319_31919


namespace diameter_of_circumscribed_circle_l319_31922

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem diameter_of_circumscribed_circle :
  circumscribed_circle_diameter 15 (Real.pi / 4) = 15 * Real.sqrt 2 :=
by
  sorry

end diameter_of_circumscribed_circle_l319_31922


namespace circle_condition_k_l319_31933

theorem circle_condition_k (k : ℝ) : 
  (∃ (h : ℝ), (x^2 + y^2 - 2*x + 6*y + k = 0)) → k < 10 :=
by
  sorry

end circle_condition_k_l319_31933


namespace union_sets_l319_31948

theorem union_sets :
  let A := { x : ℝ | x^2 - x - 2 < 0 }
  let B := { x : ℝ | x > -2 ∧ x < 0 }
  A ∪ B = { x : ℝ | x > -2 ∧ x < 2 } :=
by
  sorry

end union_sets_l319_31948


namespace radius_of_smaller_circle_l319_31947

theorem radius_of_smaller_circle (A1 : ℝ) (r1 r2 : ℝ) (h1 : π * r2^2 = 4 * A1)
    (h2 : r2 = 4) : r1 = 2 :=
by
  sorry

end radius_of_smaller_circle_l319_31947


namespace symmetry_P_over_xOz_l319_31975

-- Definition for the point P and the plane xOz
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def P : Point3D := { x := 2, y := 3, z := 4 }

def symmetry_over_xOz_plane (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_P_over_xOz : symmetry_over_xOz_plane P = { x := 2, y := -3, z := 4 } :=
by
  -- The proof is omitted.
  sorry

end symmetry_P_over_xOz_l319_31975


namespace solve_for_t_l319_31960

theorem solve_for_t (t : ℝ) (h1 : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220) 
  (h2 : 0 ≤ t) : 60 * t + 80 * ((10 : ℝ)/3 - t) = 220 :=
by
  sorry

end solve_for_t_l319_31960


namespace turns_in_two_hours_l319_31996

theorem turns_in_two_hours (turns_per_30_sec : ℕ) (minutes_in_hour : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → 
  minutes_in_hour = 60 → 
  hours = 2 → 
  (12 * (minutes_in_hour * hours)) = 1440 := 
by
  sorry

end turns_in_two_hours_l319_31996


namespace range_of_a_l319_31978

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = a*x^3 + Real.log x) :
  (∃ x : ℝ, x > 0 ∧ (deriv f x = 0)) → a < 0 :=
by
  sorry

end range_of_a_l319_31978


namespace translate_vertex_to_increase_l319_31992

def quadratic_function (x : ℝ) : ℝ := -x^2 + 1

theorem translate_vertex_to_increase (x : ℝ) :
  ∃ v, v = (2, quadratic_function 2) ∧
    (∀ x < 2, quadratic_function (x + 2) = quadratic_function x + 1 ∧
    ∀ x < 2, quadratic_function x < quadratic_function (x + 1)) :=
sorry

end translate_vertex_to_increase_l319_31992


namespace min_abs_val_sum_l319_31913

theorem min_abs_val_sum : ∃ x : ℝ, (∀ y : ℝ, |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ |x - 1| + |x - 2| + |x - 3| = 1 :=
sorry

end min_abs_val_sum_l319_31913


namespace how_many_necklaces_given_away_l319_31961

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def bought_necklaces := 5
def final_necklaces := 37

-- Define the question proof statement
theorem how_many_necklaces_given_away : 
  (initial_necklaces - broken_necklaces + bought_necklaces - final_necklaces) = 15 :=
by sorry

end how_many_necklaces_given_away_l319_31961


namespace coordinates_of_P_with_respect_to_origin_l319_31999

def point (x y : ℝ) : Prop := True

theorem coordinates_of_P_with_respect_to_origin :
  point 2 (-3) ↔ point 2 (-3) := by
  sorry

end coordinates_of_P_with_respect_to_origin_l319_31999


namespace solution_set_of_inequality_l319_31985

def f : Int → Int
| -1 => -1
| 0 => -1
| 1 => 1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

def g : Int → Int
| -1 => 1
| 0 => 1
| 1 => -1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

theorem solution_set_of_inequality :
  {x | f (g x) > 0} = { -1, 0 } :=
by
  sorry

end solution_set_of_inequality_l319_31985


namespace g_432_l319_31945

theorem g_432 (g : ℕ → ℤ)
  (h_mul : ∀ x y : ℕ, 0 < x → 0 < y → g (x * y) = g x + g y)
  (h8 : g 8 = 21)
  (h18 : g 18 = 26) :
  g 432 = 47 :=
  sorry

end g_432_l319_31945


namespace field_length_l319_31909

theorem field_length (w l: ℕ) (hw1: l = 2 * w) (hw2: 8 * 8 = 64) (hw3: 64 = l * w / 2) : l = 16 := 
by
  sorry

end field_length_l319_31909


namespace place_numbers_in_table_l319_31932

theorem place_numbers_in_table (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (table : Fin 10 → Fin 10 → ℝ),
    (∀ i j, table i j = nums ⟨10 * i + j, sorry⟩) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      |table i j - table k l| ≠ 1) := sorry  -- Proof omitted

end place_numbers_in_table_l319_31932


namespace sum_of_numbers_l319_31914

theorem sum_of_numbers (avg : ℝ) (num : ℕ) (h1 : avg = 5.2) (h2 : num = 8) : 
  (avg * num = 41.6) :=
by
  sorry

end sum_of_numbers_l319_31914


namespace john_age_proof_l319_31965

theorem john_age_proof (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end john_age_proof_l319_31965


namespace find_a_l319_31971

theorem find_a (f : ℝ → ℝ)
  (h : ∀ x : ℝ, x < 2 → a - 3 * x > 0) :
  a = 6 :=
by sorry

end find_a_l319_31971


namespace train_speed_computed_l319_31902

noncomputable def train_speed_in_kmh (train_length : ℝ) (platform_length : ℝ) (time_in_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time_in_seconds
  speed_mps * 3.6

theorem train_speed_computed :
  train_speed_in_kmh 250 50.024 15 = 72.006 := by
  sorry

end train_speed_computed_l319_31902


namespace parallelogram_area_l319_31925

theorem parallelogram_area (d : ℝ) (h : ℝ) (α : ℝ) (h_d : d = 30) (h_h : h = 20) : 
  ∃ A : ℝ, A = d * h ∧ A = 600 :=
by
  sorry

end parallelogram_area_l319_31925


namespace flour_price_increase_l319_31935

theorem flour_price_increase (x : ℝ) (hx : x > 0) :
  (9600 / (1.5 * x) - 6000 / x = 0.4) :=
by 
  sorry

end flour_price_increase_l319_31935


namespace simplify_expression_l319_31934

theorem simplify_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by
  sorry

end simplify_expression_l319_31934


namespace system_solution_l319_31977

theorem system_solution (x y : ℤ) (h1 : x + y = 1) (h2 : 2*x + y = 5) : x = 4 ∧ y = -3 :=
by {
  sorry
}

end system_solution_l319_31977


namespace planting_equation_l319_31966

def condition1 (x : ℕ) : ℕ := 5 * x + 3
def condition2 (x : ℕ) : ℕ := 6 * x - 4

theorem planting_equation (x : ℕ) : condition1 x = condition2 x := by
  sorry

end planting_equation_l319_31966


namespace min_points_in_symmetric_set_l319_31921

theorem min_points_in_symmetric_set (T : Set (ℝ × ℝ)) (h1 : ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T)
                                      (h2 : ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T)
                                      (h3 : ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T)
                                      (h4 : (1, 4) ∈ T) : 
    ∃ (S : Finset (ℝ × ℝ)), 
          (∀ p ∈ S, p ∈ T) ∧
          (∀ q ∈ T, ∃ p ∈ S, q = (p.1, p.2) ∨ q = (p.1, -p.2) ∨ q = (-p.1, p.2) ∨ q = (-p.1, -p.2) ∨ q = (-p.2, -p.1) ∨ q = (-p.2, p.1) ∨ q = (p.2, p.1) ∨ q = (p.2, -p.1)) ∧
          S.card = 8 := sorry

end min_points_in_symmetric_set_l319_31921


namespace water_evaporation_correct_l319_31939

noncomputable def water_evaporation_each_day (initial_water: ℝ) (percentage_evaporated: ℝ) (days: ℕ) : ℝ :=
  let total_evaporated := (percentage_evaporated / 100) * initial_water
  total_evaporated / days

theorem water_evaporation_correct :
  water_evaporation_each_day 10 6 30 = 0.02 := by
  sorry

end water_evaporation_correct_l319_31939


namespace only_statement_4_is_correct_l319_31950

-- Defining conditions for input/output statement correctness
def INPUT_statement_is_correct (s : String) : Prop :=
  s = "INPUT x=, 2"

def PRINT_statement_is_correct (s : String) : Prop :=
  s = "PRINT 20, 4"

-- List of statements
def statement_1 := "INPUT a; b; c"
def statement_2 := "PRINT a=1"
def statement_3 := "INPUT x=2"
def statement_4 := "PRINT 20, 4"

-- Predicate for correctness of statements
def statement_is_correct (s : String) : Prop :=
  (s = statement_4) ∧
  ¬(s = statement_1 ∨ s = statement_2 ∨ s = statement_3)

-- Theorem to prove that only statement 4 is correct
theorem only_statement_4_is_correct :
  ∀ s : String, (statement_is_correct s) ↔ (s = statement_4) :=
by
  intros s
  sorry

end only_statement_4_is_correct_l319_31950


namespace problem1_problem2_l319_31927

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |3 * x - 1|

-- Part (1) statement
theorem problem1 (x : ℝ) : f x (-1) ≤ 1 ↔ (1/4 ≤ x ∧ x ≤ 1/2) :=
by
    sorry

-- Part (2) statement
theorem problem2 (x a : ℝ) (h : 1/4 ≤ x ∧ x ≤ 1) : f x a ≤ |3 * x + 1| ↔ -7/3 ≤ a ∧ a ≤ 1 :=
by
    sorry

end problem1_problem2_l319_31927


namespace number_of_family_members_l319_31923

-- Define the number of legs for each type of animal.
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def cat_legs : ℕ := 4

-- Define the number of animals.
def birds : ℕ := 4
def dogs : ℕ := 3
def cats : ℕ := 18

-- Define the total number of legs of all animals.
def total_animal_feet : ℕ := birds * bird_legs + dogs * dog_legs + cats * cat_legs

-- Define the total number of heads of all animals.
def total_animal_heads : ℕ := birds + dogs + cats

-- Main theorem: If the total number of feet in the house is 74 more than the total number of heads, find the number of family members.
theorem number_of_family_members (F : ℕ) (h : total_animal_feet + 2 * F = total_animal_heads + F + 74) : F = 7 :=
by
  sorry

end number_of_family_members_l319_31923


namespace possible_box_dimensions_l319_31997

-- Define the initial conditions
def edge_length_original_box := 4
def edge_length_dice := 1
def total_cubes := (edge_length_original_box * edge_length_original_box * edge_length_original_box)

-- Prove that these are the possible dimensions of boxes with square bases that fit all the dice
theorem possible_box_dimensions :
  ∃ (len1 len2 len3 : ℕ), 
  total_cubes = (len1 * len2 * len3) ∧ 
  (len1 = len2) ∧ 
  ((len1, len2, len3) = (1, 1, 64) ∨ (len1, len2, len3) = (2, 2, 16) ∨ (len1, len2, len3) = (4, 4, 4) ∨ (len1, len2, len3) = (8, 8, 1)) :=
by {
  sorry -- The proof would be placed here
}

end possible_box_dimensions_l319_31997


namespace smallest_number_satisfying_conditions_l319_31983

theorem smallest_number_satisfying_conditions :
  ∃ b : ℕ, b ≡ 3 [MOD 5] ∧ b ≡ 2 [MOD 4] ∧ b ≡ 2 [MOD 6] ∧ b = 38 := 
by
  sorry

end smallest_number_satisfying_conditions_l319_31983


namespace elvis_ralph_matchsticks_l319_31946

/-- 
   Elvis and Ralph are making square shapes with matchsticks from a box containing 
   50 matchsticks. Elvis makes 4-matchstick squares and Ralph makes 8-matchstick 
   squares. If Elvis makes 5 squares and Ralph makes 3, prove the number of matchsticks 
   left in the box is 6. 
-/
def matchsticks_left_in_box
  (initial_matchsticks : ℕ)
  (elvis_squares : ℕ)
  (elvis_matchsticks : ℕ)
  (ralph_squares : ℕ)
  (ralph_matchsticks : ℕ)
  (elvis_squares_count : ℕ)
  (ralph_squares_count : ℕ) : ℕ :=
  initial_matchsticks - (elvis_squares_count * elvis_matchsticks + ralph_squares_count * ralph_matchsticks)

theorem elvis_ralph_matchsticks : matchsticks_left_in_box 50 4 5 8 3 = 6 := 
  sorry

end elvis_ralph_matchsticks_l319_31946


namespace num_zeros_in_product_l319_31987

theorem num_zeros_in_product : ∀ (a b : ℕ), (a = 125) → (b = 960) → (∃ n, a * b = n * 10^4) :=
by
  sorry

end num_zeros_in_product_l319_31987


namespace probability_product_divisible_by_four_l319_31937

open Finset

theorem probability_product_divisible_by_four :
  (∃ (favorable_pairs total_pairs : ℕ), favorable_pairs = 70 ∧ total_pairs = 190 ∧ favorable_pairs / total_pairs = 7 / 19) := 
sorry

end probability_product_divisible_by_four_l319_31937


namespace final_value_l319_31952

noncomputable def f : ℕ → ℝ := sorry

axiom f_mul_add (a b : ℕ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem final_value : 
  (f 1)^2 + f 2 / f 1 + (f 2)^2 + f 4 / f 3 + (f 3)^2 + f 6 / f 5 + (f 4)^2 + f 8 / f 7 = 16 := 
sorry

end final_value_l319_31952


namespace hot_drinks_prediction_at_2_deg_l319_31907

-- Definition of the regression equation as a function
def regression_equation (x : ℝ) : ℝ :=
  -2.35 * x + 147.77

-- The statement to be proved
theorem hot_drinks_prediction_at_2_deg :
  abs (regression_equation 2 - 143) < 1 :=
sorry

end hot_drinks_prediction_at_2_deg_l319_31907


namespace first_player_always_wins_l319_31920

theorem first_player_always_wins (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) : A + B + 1998 = 0 → 
  (∃ (a b c : ℤ), (a = A ∨ a = B ∨ a = 1998) ∧ (b = A ∨ b = B ∨ b = 1998) ∧ (c = A ∨ c = B ∨ c = 1998) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (∃ (r1 r2 : ℚ), r1 ≠ r2 ∧ r1 * r1 * a + r1 * b + c = 0 ∧ r2 * r2 * a + r2 * b + c = 0)) :=
sorry

end first_player_always_wins_l319_31920


namespace train_crossing_time_l319_31936

theorem train_crossing_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (conversion_factor : ℚ := 1000/3600) 
  (train_speed_mps : ℚ := train_speed_kmph * conversion_factor) :
  train_length = 100 →
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  train_length / train_speed_mps = 5 :=
by
  intros
  sorry

end train_crossing_time_l319_31936


namespace unique_infinite_sequence_l319_31904

-- Defining conditions for the infinite sequence of negative integers
variable (a : ℕ → ℤ)
  
-- Condition 1: Elements in sequence are negative integers
def sequence_negative : Prop :=
  ∀ n, a n < 0 

-- Condition 2: For every positive integer n, the first n elements taken modulo n have n distinct remainders
def distinct_mod_remainders (n : ℕ) : Prop :=
  ∀ i j, i < n → j < n → i ≠ j → (a i % n ≠ a j % n) 

-- The main theorem statement
theorem unique_infinite_sequence (a : ℕ → ℤ) 
  (h1 : sequence_negative a) 
  (h2 : ∀ n, distinct_mod_remainders a n) :
  ∀ k : ℤ, ∃! n, a n = k :=
sorry

end unique_infinite_sequence_l319_31904


namespace complex_sum_series_l319_31970

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l319_31970


namespace find_x_squared_plus_y_squared_l319_31989

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y + 7 = (x - 3)^2) (h2 : x + 7 = (y - 3)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 17 :=
by
  sorry  -- Proof to be provided

end find_x_squared_plus_y_squared_l319_31989


namespace least_y_solution_l319_31979

theorem least_y_solution :
  (∃ y : ℝ, 3 * y^2 + 5 * y + 2 = 4 ∧ ∀ z : ℝ, 3 * z^2 + 5 * z + 2 = 4 → y ≤ z) →
  ∃ y : ℝ, y = -2 :=
by
  sorry

end least_y_solution_l319_31979


namespace term_largest_binomial_coeff_constant_term_in_expansion_l319_31968

theorem term_largest_binomial_coeff {n : ℕ} (h : n = 8) :
  ∃ (k : ℕ) (coeff : ℤ), coeff * x ^ k = 1120 * x^4 :=
by
  sorry

theorem constant_term_in_expansion :
  ∃ (const : ℤ), const = 1280 :=
by
  sorry

end term_largest_binomial_coeff_constant_term_in_expansion_l319_31968


namespace parabola_focus_coordinates_l319_31951

theorem parabola_focus_coordinates :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 18) ∧ 
    ∃ (p : ℝ), y = 9 * x^2 → x^2 = 4 * p * y ∧ p = 1 / 18 :=
by
  sorry

end parabola_focus_coordinates_l319_31951


namespace equilateral_triangle_l319_31915

theorem equilateral_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : Real.sin A = 2 * Real.sin B * Real.cos C) 
  (h3 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) 
  (h4 : b = c) : 
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c := 
sorry

end equilateral_triangle_l319_31915


namespace average_salary_excluding_manager_l319_31982

theorem average_salary_excluding_manager
    (A : ℝ)
    (manager_salary : ℝ)
    (total_employees : ℕ)
    (salary_increase : ℝ)
    (h1 : total_employees = 24)
    (h2 : manager_salary = 4900)
    (h3 : salary_increase = 100)
    (h4 : 24 * A + manager_salary = 25 * (A + salary_increase)) :
    A = 2400 := by
  sorry

end average_salary_excluding_manager_l319_31982


namespace maximal_value_6tuple_l319_31974

theorem maximal_value_6tuple :
  ∀ (a b c d e f : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ 
  a + b + c + d + e + f = 6 → 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ((a, b, c, d, e, f) = (0, 0, t, 2, 2, 2 - t) ∨ 
   (a, b, c, d, e, f) = (0, t, 2, 2 - t, 0, 0) ∨ 
   (a, b, c, d, e, f) = (t, 2, 2 - t, 0, 0, 0) ∨ 
   (a, b, c, d, e, f) = (2, 2 - t, 0, 0, 0, t) ∨
   (a, b, c, d, e, f) = (2 - t, 0, 0, 0, t, 2) ∨
   (a, b, c, d, e, f) = (0, 0, 0, t, 2, 2 - t))) := 
sorry

end maximal_value_6tuple_l319_31974
