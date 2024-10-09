import Mathlib

namespace jessica_cut_21_roses_l509_50908

def initial_roses : ℕ := 2
def thrown_roses : ℕ := 4
def final_roses : ℕ := 23

theorem jessica_cut_21_roses : (final_roses - initial_roses) = 21 :=
by
  sorry

end jessica_cut_21_roses_l509_50908


namespace yogurt_combinations_l509_50951

theorem yogurt_combinations (flavors toppings : ℕ) (hflavors : flavors = 5) (htoppings : toppings = 8) :
  (flavors * Nat.choose toppings 3 = 280) :=
by
  rw [hflavors, htoppings]
  sorry

end yogurt_combinations_l509_50951


namespace smallest_whole_number_larger_than_sum_l509_50918

theorem smallest_whole_number_larger_than_sum :
    let sum := 2 + 1 / 2 + 3 + 1 / 3 + 4 + 1 / 4 + 5 + 1 / 5 
    let smallest_whole := 16
    (sum < smallest_whole ∧ smallest_whole - 1 < sum) := 
by
    sorry

end smallest_whole_number_larger_than_sum_l509_50918


namespace alpha_minus_beta_eq_pi_div_4_l509_50914

open Real

theorem alpha_minus_beta_eq_pi_div_4 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 4) 
(h : tan α = (cos β + sin β) / (cos β - sin β)) : α - β = π / 4 :=
sorry

end alpha_minus_beta_eq_pi_div_4_l509_50914


namespace tony_slices_remaining_l509_50983

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l509_50983


namespace accurate_mass_l509_50901

variable (m1 m2 a b x : Real) -- Declare the variables

theorem accurate_mass (h1 : a * x = b * m1) (h2 : b * x = a * m2) : x = Real.sqrt (m1 * m2) := by
  -- We will prove the statement later
  sorry

end accurate_mass_l509_50901


namespace mass_of_man_l509_50954

-- Definitions based on problem conditions
def boat_length : ℝ := 8
def boat_breadth : ℝ := 3
def sinking_height : ℝ := 0.01
def water_density : ℝ := 1000

-- Mass of the man to be proven
theorem mass_of_man : boat_density * (boat_length * boat_breadth * sinking_height) = 240 :=
by
  sorry

end mass_of_man_l509_50954


namespace constant_term_expansion_l509_50985

-- Define the binomial coefficient
noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term in the binomial expansion
noncomputable def general_term (r n : ℕ) (x : ℝ) : ℝ := 
  (2:ℝ)^r * binomial_coeff n r * x^((n-5*r)/2)

-- Given problem conditions
def n := 10
def largest_binomial_term_index := 5  -- Represents the sixth term (r = 5)

-- Statement to prove the constant term equals 180
theorem constant_term_expansion {x : ℝ} : 
  general_term 2 n 1 = 180 :=
by {
  sorry
}

end constant_term_expansion_l509_50985


namespace ratio_a_c_l509_50993

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
by
  sorry

end ratio_a_c_l509_50993


namespace number_problem_l509_50973

theorem number_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 34) / 10 = 2 := by
  sorry

end number_problem_l509_50973


namespace compare_negatives_l509_50902

theorem compare_negatives : -1 < - (2 / 3) := by
  sorry

end compare_negatives_l509_50902


namespace relationship_among_abc_l509_50944

theorem relationship_among_abc (x : ℝ) (e : ℝ) (ln : ℝ → ℝ) (half_pow : ℝ → ℝ) (exp : ℝ → ℝ) 
  (x_in_e_e2 : x > e ∧ x < exp 2) 
  (def_a : ln x = ln x)
  (def_b : half_pow (ln x) = ((1/2)^(ln x)))
  (def_c : exp (ln x) = x):
  (exp (ln x)) > (ln x) ∧ (ln x) > ((1/2)^(ln x)) :=
by 
  sorry

end relationship_among_abc_l509_50944


namespace days_to_complete_work_l509_50949

-- Conditions
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def combined_work_rate := work_rate_A + work_rate_B

-- Statement to prove
theorem days_to_complete_work : 1 / combined_work_rate = 16 / 3 := by
  sorry

end days_to_complete_work_l509_50949


namespace johns_total_animals_l509_50932

variable (Snakes Monkeys Lions Pandas Dogs : ℕ)

theorem johns_total_animals :
  Snakes = 15 →
  Monkeys = 2 * Snakes →
  Lions = Monkeys - 5 →
  Pandas = Lions + 8 →
  Dogs = Pandas / 3 →
  Snakes + Monkeys + Lions + Pandas + Dogs = 114 :=
by
  intros hSnakes hMonkeys hLions hPandas hDogs
  rw [hSnakes] at hMonkeys
  rw [hMonkeys] at hLions
  rw [hLions] at hPandas
  rw [hPandas] at hDogs
  sorry

end johns_total_animals_l509_50932


namespace total_amount_spent_l509_50940

-- Definitions for the conditions
def cost_magazine : ℝ := 0.85
def cost_pencil : ℝ := 0.50
def coupon_discount : ℝ := 0.35

-- The main theorem to prove
theorem total_amount_spent : cost_magazine + cost_pencil - coupon_discount = 1.00 := by
  sorry

end total_amount_spent_l509_50940


namespace coefficient_of_friction_l509_50931

/-- Assume m, Pi and ΔL are positive real numbers, and g is the acceleration due to gravity. 
We need to prove that the coefficient of friction μ is given by Pi / (m * g * ΔL). --/
theorem coefficient_of_friction (m Pi ΔL g : ℝ) (h_m : 0 < m) (h_Pi : 0 < Pi) (h_ΔL : 0 < ΔL) (h_g : 0 < g) :
  ∃ μ : ℝ, μ = Pi / (m * g * ΔL) :=
sorry

end coefficient_of_friction_l509_50931


namespace sum_of_greatest_values_l509_50927

theorem sum_of_greatest_values (b : ℝ) (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 → 2.5 + 2 = 4.5 :=
by sorry

end sum_of_greatest_values_l509_50927


namespace principal_amount_l509_50976

theorem principal_amount (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (h1 : R = 4) 
  (h2 : T = 5) 
  (h3 : SI = P - 1920) 
  (h4 : SI = (P * R * T) / 100) : 
  P = 2400 := 
by 
  sorry

end principal_amount_l509_50976


namespace min_fraction_ineq_l509_50970

theorem min_fraction_ineq (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  ∃ z, (z = x * y / (x^2 + 2 * y^2)) ∧ z = 1 / 3 := sorry

end min_fraction_ineq_l509_50970


namespace range_of_k_l509_50967

noncomputable def quadratic_inequality (k x : ℝ) : ℝ :=
  k * x^2 + 2 * k * x - (k + 2)

theorem range_of_k :
  (∀ x : ℝ, quadratic_inequality k x < 0) ↔ -1 < k ∧ k < 0 :=
by
  sorry

end range_of_k_l509_50967


namespace inequality_AM_GM_l509_50956

variable {a b c d : ℝ}
variable (h₁ : 0 < a)
variable (h₂ : 0 < b)
variable (h₃ : 0 < c)
variable (h₄ : 0 < d)

theorem inequality_AM_GM :
  (c / a * (8 * b + c) + d / b * (8 * c + d) + a / c * (8 * d + a) + b / d * (8 * a + b)) ≥ 9 * (a + b + c + d) :=
sorry

end inequality_AM_GM_l509_50956


namespace trip_time_l509_50962

open Real

variables (d T : Real)

theorem trip_time :
  (T = d / 30 + (150 - d) / 6) ∧
  (T = 2 * (d / 30) + 1 + (150 - d) / 30) ∧
  (T - 1 = d / 6 + (150 - d) / 30) →
  T = 20 :=
by
  sorry

end trip_time_l509_50962


namespace axis_of_symmetry_shift_l509_50906

-- Define that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the problem statement in Lean
theorem axis_of_symmetry_shift (f : ℝ → ℝ) 
  (h_even : is_even_function f) :
  ∃ x, ∀ y, f (x + y) = f ((x - 1) + y) ∧ x = -1 :=
sorry

end axis_of_symmetry_shift_l509_50906


namespace sequence_formula_l509_50977

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 33) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 :=
by
  sorry

end sequence_formula_l509_50977


namespace carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l509_50990

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9
def tom_weight : ℕ := 20

theorem carol_tom_combined_weight :
  carol_weight + tom_weight = 29 := by
  sorry

theorem mildred_heavier_than_carol_tom_combined :
  mildred_weight - (carol_weight + tom_weight) = 30 := by
  sorry

end carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l509_50990


namespace parallel_lines_l509_50988

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, x + a * y - (2 * a + 2) = 0 ∧ a * x + y - (a + 1) = 0 → (∀ x y : ℝ, (1 / a = a / 1) ∧ (1 / a ≠ (2 * -a - 2) / (1 * -a - 1)))) → a = 1 := by
sorry

end parallel_lines_l509_50988


namespace margie_change_is_6_25_l509_50910

-- The conditions are given as definitions in Lean
def numberOfApples : Nat := 5
def costPerApple : ℝ := 0.75
def amountPaid : ℝ := 10.00

-- The statement to be proved
theorem margie_change_is_6_25 :
  (amountPaid - (numberOfApples * costPerApple)) = 6.25 := 
  sorry

end margie_change_is_6_25_l509_50910


namespace sin_cos_expression_l509_50978

noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def cos_15 := Real.cos (Real.pi / 12)
noncomputable def cos_225 := Real.cos (5 * Real.pi / 4)
noncomputable def sin_15 := Real.sin (Real.pi / 12)

theorem sin_cos_expression :
  sin_45 * cos_15 + cos_225 * sin_15 = 1 / 2 :=
by
  sorry

end sin_cos_expression_l509_50978


namespace solve_for_y_l509_50963

def G (a y c d : ℕ) := 3 ^ y + 6 * d

theorem solve_for_y (a c d : ℕ) (h1 : G a 2 c d = 735) : 2 = 2 := 
by
  sorry

end solve_for_y_l509_50963


namespace difference_of_numbers_l509_50929

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 20460) (h2 : b % 12 = 0) (h3 : b / 10 = a) : b - a = 17314 :=
by
  sorry

end difference_of_numbers_l509_50929


namespace A_share_correct_l509_50959

noncomputable def investment_shares (x : ℝ) (annual_gain : ℝ) := 
  let A_share := x * 12
  let B_share := (2 * x) * 6
  let C_share := (3 * x) * 4
  let total_share := A_share + B_share + C_share
  let total_ratio := 1 + 1 + 1
  annual_gain / total_ratio

theorem A_share_correct (x : ℝ) (annual_gain : ℝ) (h_gain : annual_gain = 18000) : 
  investment_shares x annual_gain / 3 = 6000 := by
  sorry

end A_share_correct_l509_50959


namespace factorize_1_factorize_2_factorize_3_l509_50965

theorem factorize_1 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

theorem factorize_2 (x y : ℝ) : 25*x^2*y + 20*x*y^2 + 4*y^3 = y * (5*x + 2*y)^2 :=
sorry

theorem factorize_3 (x y a : ℝ) : x^2 * (a - 1) + y^2 * (1 - a) = (a - 1) * (x + y) * (x - y) :=
sorry

end factorize_1_factorize_2_factorize_3_l509_50965


namespace fraction_white_surface_area_l509_50972

theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let black_faces_corners := 6
  let black_faces_centers := 6
  let black_faces_total := 12
  let white_faces_total := total_surface_area - black_faces_total
  white_faces_total / total_surface_area = 7 / 8 :=
by
  sorry

end fraction_white_surface_area_l509_50972


namespace optionD_is_equation_l509_50950

-- Definitions for options
def optionA (x : ℕ) := 2 * x - 3
def optionB := 2 + 4 = 6
def optionC (x : ℕ) := x > 2
def optionD (x : ℕ) := 2 * x - 1 = 3

-- Goal: prove that option D is an equation.
theorem optionD_is_equation (x : ℕ) : (optionD x) = True :=
sorry

end optionD_is_equation_l509_50950


namespace inequality_abc_l509_50920

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l509_50920


namespace inequality_proof_l509_50953

theorem inequality_proof (a b : ℝ) (h : a + b > 0) : 
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := 
sorry

end inequality_proof_l509_50953


namespace num_factors_36_l509_50928

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end num_factors_36_l509_50928


namespace sequence_relation_l509_50975

theorem sequence_relation
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h1 : ∀ n, b (n + 1) * a n + b n * a (n + 1) = (-2)^n + 1)
  (h2 : ∀ n, b n = (3 + (-1 : ℚ)^(n-1)) / 2)
  (h3 : a 1 = 2) :
  ∀ n, a (2 * n) = (1 - 4^n) / 2 :=
by
  intro n
  sorry

end sequence_relation_l509_50975


namespace exactly_one_pair_probability_l509_50955

def four_dice_probability : ℚ :=
  sorry  -- Here we skip the actual computation and proof

theorem exactly_one_pair_probability : four_dice_probability = 5/9 := by {
  -- Placeholder for proof, explanation, and calculation
  sorry
}

end exactly_one_pair_probability_l509_50955


namespace correct_calculation_l509_50925

theorem correct_calculation (a : ℝ) : (3 * a^3)^2 = 9 * a^6 :=
by sorry

end correct_calculation_l509_50925


namespace mark_owes_820_l509_50982

-- Definitions of the problem conditions
def base_fine : ℕ := 50
def over_speed_fine (mph_over : ℕ) : ℕ := mph_over * 2
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def lawyer_cost_per_hour : ℕ := 80
def lawyer_hours : ℕ := 3

-- Calculation of the total fine
def total_fine (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let mph_over := actual_speed - speed_limit
  let additional_fine := over_speed_fine mph_over
  let fine_before_multipliers := base_fine + additional_fine
  let fine_after_multipliers := fine_before_multipliers * school_zone_multiplier
  fine_after_multipliers

-- Calculation of the total costs
def total_costs (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let fine := total_fine speed_limit actual_speed
  fine + court_costs + (lawyer_cost_per_hour * lawyer_hours)

theorem mark_owes_820 : total_costs 30 75 = 820 := 
by
  sorry

end mark_owes_820_l509_50982


namespace find_value_divided_by_4_l509_50911

theorem find_value_divided_by_4 (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 :=
by
  sorry

end find_value_divided_by_4_l509_50911


namespace solve_quadratic_completing_square_l509_50999

theorem solve_quadratic_completing_square :
  ∃ (a b c : ℤ), a > 0 ∧ 25 * a * a + 30 * b - 45 = (a * x + b)^2 - c ∧
                 a + b + c = 62 :=
by
  sorry

end solve_quadratic_completing_square_l509_50999


namespace sum_of_squares_of_sines_l509_50926

theorem sum_of_squares_of_sines (α : ℝ) : 
  (Real.sin α)^2 + (Real.sin (α + 60 * Real.pi / 180))^2 + (Real.sin (α + 120 * Real.pi / 180))^2 = 3 / 2 := 
by
  sorry

end sum_of_squares_of_sines_l509_50926


namespace steve_speed_l509_50998

theorem steve_speed (v : ℝ) : 
  (John_initial_distance_behind_Steve = 15) ∧ 
  (John_final_distance_ahead_of_Steve = 2) ∧ 
  (John_speed = 4.2) ∧ 
  (final_push_duration = 34) → 
  v * final_push_duration = (John_speed * final_push_duration) - (John_initial_distance_behind_Steve + John_final_distance_ahead_of_Steve) →
  v = 3.7 := 
by
  intros hconds heq
  exact sorry

end steve_speed_l509_50998


namespace hawks_points_l509_50921

theorem hawks_points (x y : ℕ) (h1 : x + y = 50) (h2 : x + 4 - y = 12) : y = 21 :=
by
  sorry

end hawks_points_l509_50921


namespace part1_part2_l509_50952

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 1) / Real.exp x
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := (-a * x^2 + (2 * a - b) * x + b - 1) / Real.exp x

theorem part1 (a b : ℝ) (h : f a b (-1) + f' a b (-1) = 0) : b = 2 * a :=
sorry

theorem part2 (a : ℝ) (h : a ≤ 1 / 2) (x : ℝ) : f a (2 * a) (abs x) ≤ 1 :=
sorry

end part1_part2_l509_50952


namespace sin_alpha_l509_50930

variable (α : Real)
variable (hcos : Real.cos α = 3 / 5)
variable (htan : Real.tan α < 0)

theorem sin_alpha (α : Real) (hcos : Real.cos α = 3 / 5) (htan : Real.tan α < 0) :
  Real.sin α = -4 / 5 :=
sorry

end sin_alpha_l509_50930


namespace quadratic_inequality_l509_50987

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - 3*x - m = 0 ∧ (∃ y : ℝ, y^2 - 3*y - m = 0 ∧ x ≠ y)) ↔ m > - 9 / 4 := 
by
  sorry

end quadratic_inequality_l509_50987


namespace WangLi_final_score_l509_50947

def weightedFinalScore (writtenScore : ℕ) (demoScore : ℕ) (interviewScore : ℕ)
    (writtenWeight : ℕ) (demoWeight : ℕ) (interviewWeight : ℕ) : ℕ :=
  (writtenScore * writtenWeight + demoScore * demoWeight + interviewScore * interviewWeight) /
  (writtenWeight + demoWeight + interviewWeight)

theorem WangLi_final_score :
  weightedFinalScore 96 90 95 5 3 2 = 94 :=
  by
  -- proof goes here
  sorry

end WangLi_final_score_l509_50947


namespace find_possible_first_term_l509_50960

noncomputable def geometric_sequence_first_term (a r : ℝ) : Prop :=
  (a * r^2 = 3) ∧ (a * r^4 = 27)

theorem find_possible_first_term (a r : ℝ) (h : geometric_sequence_first_term a r) :
    a = 1 / 3 :=
by
  sorry

end find_possible_first_term_l509_50960


namespace inverse_function_correct_l509_50939

theorem inverse_function_correct :
  ( ∀ x : ℝ, (x > 1) → (∃ y : ℝ, y = 1 + Real.log (x - 1)) ↔ (∀ y : ℝ, y > 0 → (∃ x : ℝ, x = e^(y + 1) - 1))) :=
by
  sorry

end inverse_function_correct_l509_50939


namespace total_amount_shared_l509_50981

-- Given John (J), Jose (Jo), and Binoy (B) and their proportion of money
variables (J Jo B : ℕ)
-- John received 1440 Rs.
variable (John_received : J = 1440)

-- The ratio of their shares is 2:4:6
axiom ratio_condition : J * 2 = Jo * 4 ∧ J * 2 = B * 6

-- The target statement to prove
theorem total_amount_shared : J + Jo + B = 8640 :=
by {
  sorry
}

end total_amount_shared_l509_50981


namespace largest_three_digit_number_l509_50995

theorem largest_three_digit_number :
  ∃ n k m : ℤ, 100 ≤ n ∧ n < 1000 ∧ n = 7 * k + 2 ∧ n = 4 * m + 1 ∧ n = 989 :=
by
  sorry

end largest_three_digit_number_l509_50995


namespace tennis_ball_price_l509_50997

theorem tennis_ball_price (x y : ℝ) 
  (h₁ : 2 * x + 7 * y = 220)
  (h₂ : x = y + 83) : 
  y = 6 := 
by 
  sorry

end tennis_ball_price_l509_50997


namespace triangle_type_l509_50989

theorem triangle_type (a b c : ℝ) (A B C : ℝ) (h1 : A = 30) (h2 : a = 2 * b ∨ b = 2 * c ∨ c = 2 * a) :
  (C > 90 ∨ B > 90) ∨ C = 90 :=
sorry

end triangle_type_l509_50989


namespace absolute_value_simplify_l509_50903

variable (a : ℝ)

theorem absolute_value_simplify
  (h : a < 3) : |a - 3| = 3 - a := sorry

end absolute_value_simplify_l509_50903


namespace units_digit_calculation_l509_50904

theorem units_digit_calculation : 
  ((33 * (83 ^ 1001) * (7 ^ 1002) * (13 ^ 1003)) % 10) = 9 :=
by
  sorry

end units_digit_calculation_l509_50904


namespace solve_for_a_l509_50913

theorem solve_for_a (x y a : ℝ) (h1 : 2 * x + y = 2 * a + 1) 
                    (h2 : x + 2 * y = a - 1) 
                    (h3 : x - y = 4) : a = 2 :=
by
  sorry

end solve_for_a_l509_50913


namespace find_m_such_that_no_linear_term_in_expansion_l509_50968

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l509_50968


namespace leo_weight_l509_50907

-- Definitions from the conditions
variable (L K J M : ℝ)

-- Conditions 
def condition1 : Prop := L + 15 = 1.60 * K
def condition2 : Prop := L + 15 = 0.40 * J
def condition3 : Prop := J = K + 25
def condition4 : Prop := M = K - 20
def condition5 : Prop := L + K + J + M = 350

-- Final statement to prove based on the conditions
theorem leo_weight (h1 : condition1 L K) (h2 : condition2 L J) (h3 : condition3 J K) 
                   (h4 : condition4 M K) (h5 : condition5 L K J M) : L = 110.22 :=
by 
  sorry

end leo_weight_l509_50907


namespace vector_line_equation_l509_50946

open Real

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let numer := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1 * v.1 + v.2 * v.2)
  (numer * v.1 / denom, numer * v.2 / denom)

theorem vector_line_equation (x y : ℝ) :
  vector_projection (x, y) (3, 4) = (-3, -4) → 
  y = -3 / 4 * x - 25 / 4 :=
  sorry

end vector_line_equation_l509_50946


namespace find_ratio_of_three_numbers_l509_50986

noncomputable def ratio_of_three_numbers (A B C : ℝ) : Prop :=
  (A + B + C) / (A + B - C) = 4 / 3 ∧
  (A + B) / (B + C) = 7 / 6

theorem find_ratio_of_three_numbers (A B C : ℝ) (h₁ : ratio_of_three_numbers A B C) :
  A / C = 2 ∧ B / C = 5 :=
by
  sorry

end find_ratio_of_three_numbers_l509_50986


namespace lucas_should_give_fraction_l509_50958

-- Conditions as Lean definitions
variables (n : ℕ) -- Number of shells Noah has
def Noah_shells := n
def Emma_shells := 2 * n -- Emma has twice as many shells as Noah
def Lucas_shells := 8 * n -- Lucas has four times as many shells as Emma

-- Desired distribution
def Total_shells := Noah_shells n + Emma_shells n + Lucas_shells n
def Each_person_shells := Total_shells n / 3

-- Fraction calculation
def Shells_needed_by_Emma := Each_person_shells n - Emma_shells n
def Fraction_of_Lucas_shells_given_to_Emma := Shells_needed_by_Emma n / Lucas_shells n 

theorem lucas_should_give_fraction :
  Fraction_of_Lucas_shells_given_to_Emma n = 5 / 24 := 
by
  sorry

end lucas_should_give_fraction_l509_50958


namespace trigonometric_identity_l509_50909

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (π / 2 + α) * Real.cos (π + α) = -1 / 5 :=
by
  -- The proof will be skipped but the statement should be correct.
  sorry

end trigonometric_identity_l509_50909


namespace Hamilton_marching_band_members_l509_50971

theorem Hamilton_marching_band_members (m : ℤ) (k : ℤ) :
  30 * m ≡ 5 [ZMOD 31] ∧ m = 26 + 31 * k ∧ 30 * m < 1500 → 30 * m = 780 :=
by
  intro h
  have hmod : 30 * m ≡ 5 [ZMOD 31] := h.left
  have m_eq : m = 26 + 31 * k := h.right.left
  have hlt : 30 * m < 1500 := h.right.right
  sorry

end Hamilton_marching_band_members_l509_50971


namespace area_of_shaded_region_l509_50945

-- Definitions of conditions
def center (O : Type) := O
def radius_large_circle (R : ℝ) := R
def radius_small_circle (r : ℝ) := r
def length_chord_CD (CD : ℝ) := CD = 60
def chord_tangent_to_smaller_circle (r : ℝ) (R : ℝ) := r^2 = R^2 - 900

-- Theorem for the area of the shaded region
theorem area_of_shaded_region 
(O : Type) 
(R r : ℝ) 
(CD : ℝ)
(h1 : length_chord_CD CD)
(h2 : chord_tangent_to_smaller_circle r R) : 
  π * (R^2 - r^2) = 900 * π := by
  sorry

end area_of_shaded_region_l509_50945


namespace sum_of_two_pos_implies_one_pos_l509_50933

theorem sum_of_two_pos_implies_one_pos (x y : ℝ) (h : x + y > 0) : x > 0 ∨ y > 0 :=
  sorry

end sum_of_two_pos_implies_one_pos_l509_50933


namespace prob1_prob2_prob3_prob4_l509_50979

theorem prob1 : (-20) + (-14) - (-18) - 13 = -29 := sorry

theorem prob2 : (-24) * (-1/2 + 3/4 - 1/3) = 2 := sorry

theorem prob3 : (- (49 + 24/25)) * 10 = -499.6 := sorry

theorem prob4 :
  -3^2 + ((-1/3) * (-3) - 8/5 / 2^2) = -8 - 2/5 := sorry

end prob1_prob2_prob3_prob4_l509_50979


namespace parabola_y_axis_symmetry_l509_50957

theorem parabola_y_axis_symmetry (a b c d : ℝ) (r : ℝ) :
  (2019^2 + 2019 * a + b = 0) ∧ (2019^2 + 2019 * c + d = 0) ∧
  (a = -(2019 + r)) ∧ (c = -(2019 - r)) →
  b = -d :=
by
  sorry

end parabola_y_axis_symmetry_l509_50957


namespace tangent_line_eq_l509_50915

theorem tangent_line_eq
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (P : ℝ × ℝ)
  (hP : P = (1, Real.sqrt 3))
  : x - Real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_eq_l509_50915


namespace doubled_dimensions_new_volume_l509_50924

-- Define the original volume condition
def original_volume_condition (π r h : ℝ) : Prop := π * r^2 * h = 5

-- Define the new volume function after dimensions are doubled
def new_volume (π r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The Lean statement for the proof problem 
theorem doubled_dimensions_new_volume (π r h : ℝ) (h_orig : original_volume_condition π r h) : 
  new_volume π r h = 40 :=
by 
  sorry

end doubled_dimensions_new_volume_l509_50924


namespace seq_composite_l509_50942

-- Define the sequence recurrence relation
def seq (a : ℕ → ℕ) : Prop :=
  ∀ (k : ℕ), k ≥ 1 → a (k+2) = a (k+1) * a k + 1

-- Prove that for k ≥ 9, a_k - 22 is composite
theorem seq_composite (a : ℕ → ℕ) (h_seq : seq a) :
  ∀ (k : ℕ), k ≥ 9 → ∃ d, d > 1 ∧ d < a k ∧ d ∣ (a k - 22) :=
by
  sorry

end seq_composite_l509_50942


namespace phone_answer_prob_within_four_rings_l509_50943

def prob_first_ring : ℚ := 1/10
def prob_second_ring : ℚ := 1/5
def prob_third_ring : ℚ := 3/10
def prob_fourth_ring : ℚ := 1/10

theorem phone_answer_prob_within_four_rings :
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring = 7/10 :=
by
  sorry

end phone_answer_prob_within_four_rings_l509_50943


namespace required_lemons_for_20_gallons_l509_50900

-- Conditions
def lemons_for_50_gallons : ℕ := 40
def gallons_for_lemons : ℕ := 50
def additional_lemons_per_10_gallons : ℕ := 1
def number_of_gallons : ℕ := 20
def base_lemons (g: ℕ) : ℕ := (lemons_for_50_gallons * g) / gallons_for_lemons
def additional_lemons (g: ℕ) : ℕ := (g / 10) * additional_lemons_per_10_gallons
def total_lemons (g: ℕ) : ℕ := base_lemons g + additional_lemons g

-- Proof statement
theorem required_lemons_for_20_gallons : total_lemons number_of_gallons = 18 :=
by
  sorry

end required_lemons_for_20_gallons_l509_50900


namespace power_multiplication_l509_50916

theorem power_multiplication :
  (- (4 / 5 : ℚ)) ^ 2022 * (5 / 4 : ℚ) ^ 2023 = 5 / 4 := 
by {
  sorry
}

end power_multiplication_l509_50916


namespace abs_ineq_solution_l509_50980

theorem abs_ineq_solution (x : ℝ) : (2 ≤ |x - 5| ∧ |x - 5| ≤ 4) ↔ (1 ≤ x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_ineq_solution_l509_50980


namespace intersection_points_l509_50984

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem intersection_points (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono_inc : monotonically_increasing f)
  (h_sign_change : f 1 * f 2 < 0) :
  ∃! x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
sorry

end intersection_points_l509_50984


namespace minimum_value_f_l509_50923

noncomputable def f (x : ℝ) (f1 f2 : ℝ) : ℝ :=
  f1 * x + f2 / x - 2

theorem minimum_value_f (f1 f2 : ℝ) (h1 : f2 = 2) (h2 : f1 = 3 / 2) :
  ∃ x > 0, f x f1 f2 = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end minimum_value_f_l509_50923


namespace meaning_of_poverty_l509_50934

theorem meaning_of_poverty (s : String) : s = "poverty" ↔ s = "poverty" := sorry

end meaning_of_poverty_l509_50934


namespace walter_exceptional_days_l509_50996

variable (b w : Nat)

-- Definitions of the conditions
def total_days (b w : Nat) : Prop := b + w = 10
def total_earnings (b w : Nat) : Prop := 3 * b + 6 * w = 42

-- The theorem states that given the conditions, the number of days Walter did his chores exceptionally well is 4
theorem walter_exceptional_days : total_days b w → total_earnings b w → w = 4 := 
  by
    sorry

end walter_exceptional_days_l509_50996


namespace arithmetic_sequence_ratio_l509_50922

noncomputable def A_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def B_n (b e : ℤ) (n : ℕ) : ℤ :=
  n * (2 * b + (n - 1) * e) / 2

theorem arithmetic_sequence_ratio (a d b e : ℤ) :
  (∀ n : ℕ, n ≠ 0 → A_n a d n / B_n b e n = (5 * n - 3) / (n + 9)) →
  (a + 5 * d) / (b + 2 * e) = 26 / 7 :=
by
  sorry

end arithmetic_sequence_ratio_l509_50922


namespace no_distinct_integers_cycle_l509_50938

theorem no_distinct_integers_cycle (p : ℤ → ℤ) 
  (x : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (n : ℕ) (h_n_ge_3 : n ≥ 3)
  (hx_cycle : ∀ i, i < n → p (x i) = x (i + 1) % n) :
  false :=
sorry

end no_distinct_integers_cycle_l509_50938


namespace common_difference_of_arithmetic_seq_l509_50935

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (m - n = 1) → (a (m + 1) - a m) = (a (n + 1) - a n)

/-- The common difference of an arithmetic sequence given certain conditions. -/
theorem common_difference_of_arithmetic_seq (a: ℕ → ℤ) (d : ℤ):
    a 1 + a 2 = 4 → 
    a 3 + a 4 = 16 →
    arithmetic_sequence a →
    (a 2 - a 1) = d → d = 3 :=
by
  intros h1 h2 h3 h4
  -- Proof to be filled in here
  sorry

end common_difference_of_arithmetic_seq_l509_50935


namespace diorama_factor_l509_50974

theorem diorama_factor (P B factor : ℕ) (h1 : P + B = 67) (h2 : B = P * factor - 5) (h3 : B = 49) : factor = 3 :=
by
  sorry

end diorama_factor_l509_50974


namespace swimming_time_l509_50917

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time_l509_50917


namespace exists_a_log_eq_l509_50936

theorem exists_a_log_eq (a : ℝ) (h : a = 10 ^ ((Real.log 2 * Real.log 3) / (Real.log 2 + Real.log 3))) :
  ∀ x > 0, Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a :=
by
  sorry

end exists_a_log_eq_l509_50936


namespace green_peaches_in_each_basket_l509_50966

theorem green_peaches_in_each_basket (G : ℕ) 
  (h1 : ∀ B : ℕ, B = 15) 
  (h2 : ∀ R : ℕ, R = 19) 
  (h3 : ∀ P : ℕ, P = 345) 
  (h_eq : 345 = 15 * (19 + G)) : 
  G = 4 := by
  sorry

end green_peaches_in_each_basket_l509_50966


namespace population_ratios_l509_50941

variable (P_X P_Y P_Z : Nat)

theorem population_ratios
  (h1 : P_Y = 2 * P_Z)
  (h2 : P_X = 10 * P_Z) : P_X / P_Y = 5 := by
  sorry

end population_ratios_l509_50941


namespace number_of_girls_l509_50948

/-- In a school with 632 students, the average age of the boys is 12 years
and that of the girls is 11 years. The average age of the school is 11.75 years.
How many girls are there in the school? Prove that the number of girls is 108. -/
theorem number_of_girls (B G : ℕ) (h1 : B + G = 632) (h2 : 12 * B + 11 * G = 7428) :
  G = 108 :=
sorry

end number_of_girls_l509_50948


namespace distance_between_foci_l509_50961

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36 = 0

-- Define the distance between the foci of the ellipse
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 2 * Real.sqrt 14.28 = 2 * Real.sqrt 14.28 :=
by sorry

end distance_between_foci_l509_50961


namespace train_cross_time_l509_50992

def length_of_train : ℕ := 120 -- the train is 120 m long
def speed_of_train_km_hr : ℕ := 45 -- the train's speed in km/hr
def length_of_bridge : ℕ := 255 -- the bridge is 255 m long

def train_speed_m_s : ℕ := speed_of_train_km_hr * (1000 / 3600)

def total_distance : ℕ := length_of_train + length_of_bridge

def time_to_cross_bridge (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_cross_time :
  time_to_cross_bridge total_distance train_speed_m_s = 30 :=
by
  sorry

end train_cross_time_l509_50992


namespace find_angle_A_l509_50964

theorem find_angle_A
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) : A = 50 :=
by 
  sorry

end find_angle_A_l509_50964


namespace minimize_perimeter_of_sector_l509_50912

theorem minimize_perimeter_of_sector (r θ: ℝ) (h₁: (1 / 2) * θ * r^2 = 16) (h₂: 2 * r + θ * r = 2 * r + 32 / r): θ = 2 :=
by
  sorry

end minimize_perimeter_of_sector_l509_50912


namespace find_a_range_l509_50994

-- Definitions as per conditions
def prop_P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0
def prop_Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Given conditions
def P_true (a : ℝ) (h : prop_P a) : Prop :=
  ∀ (a : ℝ), a^2 - 16 < 0

def Q_false (a : ℝ) (h : ¬prop_Q a) : Prop :=
  ∀ (a : ℝ), a > 1

-- Main theorem
theorem find_a_range (a : ℝ) (hP : prop_P a) (hQ : ¬prop_Q a) : 1 < a ∧ a < 4 :=
sorry

end find_a_range_l509_50994


namespace pyramid_edges_l509_50937

-- Define the conditions
def isPyramid (n : ℕ) : Prop :=
  (n + 1) + (n + 1) = 16

-- Statement to be proved
theorem pyramid_edges : ∃ (n : ℕ), isPyramid n ∧ 2 * n = 14 :=
by {
  sorry
}

end pyramid_edges_l509_50937


namespace sum_of_ages_l509_50905

-- Define ages of Kiana and her twin brothers
variables (kiana_age : ℕ) (twin_age : ℕ)

-- Define conditions
def age_product_condition : Prop := twin_age * twin_age * kiana_age = 162
def age_less_than_condition : Prop := kiana_age < 10
def twins_older_condition : Prop := twin_age > kiana_age

-- The main problem statement
theorem sum_of_ages (h1 : age_product_condition twin_age kiana_age) (h2 : age_less_than_condition kiana_age) (h3 : twins_older_condition twin_age kiana_age) :
  twin_age * 2 + kiana_age = 20 :=
sorry

end sum_of_ages_l509_50905


namespace trigonometric_identity_proof_l509_50969

theorem trigonometric_identity_proof (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = - (Real.sqrt 3 + 2) / 3 :=
by
  sorry

end trigonometric_identity_proof_l509_50969


namespace brock_peanuts_ratio_l509_50991

theorem brock_peanuts_ratio (initial : ℕ) (bonita : ℕ) (remaining : ℕ) (brock : ℕ)
  (h1 : initial = 148) (h2 : bonita = 29) (h3 : remaining = 82) (h4 : brock = 37)
  (h5 : initial - remaining = bonita + brock) :
  (brock : ℚ) / initial = 1 / 4 :=
by {
  sorry
}

end brock_peanuts_ratio_l509_50991


namespace min_value_x_l509_50919

theorem min_value_x (a : ℝ) (h : ∀ a > 0, x^2 ≤ 1 + a) : ∃ x, ∀ a > 0, -1 ≤ x ∧ x ≤ 1 := 
sorry

end min_value_x_l509_50919
