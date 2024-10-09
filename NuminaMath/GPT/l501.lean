import Mathlib

namespace volleyballs_remaining_l501_50167

def initial_volleyballs := 9
def lent_volleyballs := 5

theorem volleyballs_remaining : initial_volleyballs - lent_volleyballs = 4 := 
by
  sorry

end volleyballs_remaining_l501_50167


namespace remaining_amount_correct_l501_50129

-- Definitions for the given conditions
def deposit_percentage : ℝ := 0.05
def deposit_amount : ℝ := 50

-- The correct answer we need to prove
def remaining_amount_to_be_paid : ℝ := 950

-- Stating the theorem (proof not required)
theorem remaining_amount_correct (total_price : ℝ) 
    (H1 : deposit_amount = total_price * deposit_percentage) : 
    total_price - deposit_amount = remaining_amount_to_be_paid :=
by
  sorry

end remaining_amount_correct_l501_50129


namespace vector_properties_l501_50133

-- Definitions of vectors
def vec_a : ℝ × ℝ := (3, 11)
def vec_b : ℝ × ℝ := (-1, -4)
def vec_c : ℝ × ℝ := (1, 3)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Linear combination of vector scaling and addition
def vec_sub_scal (u v : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (u.1 - k * v.1, u.2 - k * v.2)

-- Check if two vectors are parallel
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2

-- Lean statement for the proof problem
theorem vector_properties :
  dot_product vec_a vec_b = -47 ∧
  vec_sub_scal vec_a vec_b 2 = (5, 19) ∧
  dot_product (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) vec_c ≠ 0 ∧
  parallel (vec_sub_scal vec_a vec_c 1) vec_b :=
by sorry

end vector_properties_l501_50133


namespace sum_of_integers_l501_50197

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 255) (h2 : a < 30) (h3 : b < 30) (h4 : a % 2 = 1) :
  a + b = 30 := 
sorry

end sum_of_integers_l501_50197


namespace tangency_condition_l501_50191

def functions_parallel (a b c : ℝ) (f g: ℝ → ℝ)
       (parallel: ∀ x, f x = a * x + b ∧ g x = a * x + c) := 
  ∀ x, f x = a * x + b ∧ g x = a * x + c

theorem tangency_condition (a b c A : ℝ)
    (h_parallel : a ≠ 0)
    (h_tangency : (∀ x, (a * x + b)^2 = 7 * (a * x + c))) :
  A = 0 ∨ A = -7 :=
sorry

end tangency_condition_l501_50191


namespace relationship_of_points_on_inverse_proportion_l501_50161

theorem relationship_of_points_on_inverse_proportion :
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  sorry

end relationship_of_points_on_inverse_proportion_l501_50161


namespace lost_weights_l501_50124

-- Define the weights
def weights : List ℕ := [43, 70, 57]

-- Total remaining weight after loss
def remaining_weight : ℕ := 20172

-- Number of weights lost
def weights_lost : ℕ := 4

-- Whether a given number of weights and types of weights match the remaining weight
def valid_loss (initial_count : ℕ) (lost_weight_count : ℕ) : Prop :=
  let total_initial_weight := initial_count * (weights.sum)
  let lost_weight := lost_weight_count * 57
  total_initial_weight - lost_weight = remaining_weight

-- Proposition we need to prove
theorem lost_weights (initial_count : ℕ) (h : valid_loss initial_count weights_lost) : ∀ w ∈ weights, w = 57 :=
by {
  sorry
}

end lost_weights_l501_50124


namespace sum_of_reciprocals_factors_12_l501_50106

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l501_50106


namespace first_group_men_8_l501_50152

variable (x : ℕ)

theorem first_group_men_8 (h1 : x * 80 = 20 * 32) : x = 8 := by
  -- provide the proof here
  sorry

end first_group_men_8_l501_50152


namespace miles_driven_on_Monday_l501_50125

def miles_Tuesday : ℕ := 18
def miles_Wednesday : ℕ := 21
def avg_miles_per_day : ℕ := 17

theorem miles_driven_on_Monday (miles_Monday : ℕ) :
  (miles_Monday + miles_Tuesday + miles_Wednesday) / 3 = avg_miles_per_day →
  miles_Monday = 12 :=
by
  intro h
  sorry

end miles_driven_on_Monday_l501_50125


namespace decrypted_plaintext_l501_50145

theorem decrypted_plaintext (a b c d : ℕ) : 
  (a + 2 * b = 14) → (2 * b + c = 9) → (2 * c + 3 * d = 23) → (4 * d = 28) → 
  (a = 6 ∧ b = 4 ∧ c = 1 ∧ d = 7) :=
by 
  intros h1 h2 h3 h4
  -- Proof steps go here
  sorry

end decrypted_plaintext_l501_50145


namespace no_real_solution_equation_l501_50163

theorem no_real_solution_equation (x : ℝ) (h : x ≠ -9) : 
  ¬ ∃ x, (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 :=
by
  sorry

end no_real_solution_equation_l501_50163


namespace inequality_proof_l501_50190

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end inequality_proof_l501_50190


namespace monotonicity_and_extrema_l501_50101

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → f x < f (x + 0.0001)) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → f x > f (x + 0.0001)) ∧
  (∀ x, -1 / 2 < x ∧ x < (Real.exp 2 - 3) / 2 → f x < f (x + 0.0001)) ∧
  ∀ x, x ∈ Set.Icc (-1 : ℝ) ((Real.exp 2 - 3) / 2) →
     (f (x) ≥ Real.log 2 + 1 / 4 → x = -1 / 2) ∧
     (f (x) ≤ 2 + (Real.exp 2 - 3)^2 / 4 → x = (Real.exp 2 - 3) / 2) :=
sorry

end monotonicity_and_extrema_l501_50101


namespace number_of_ways_to_enter_and_exit_l501_50170

theorem number_of_ways_to_enter_and_exit (n : ℕ) (h : n = 4) : (n * n) = 16 := by
  sorry

end number_of_ways_to_enter_and_exit_l501_50170


namespace carrie_spent_l501_50127

-- Definitions derived from the problem conditions
def cost_of_one_tshirt : ℝ := 9.65
def number_of_tshirts : ℕ := 12

-- The statement to prove
theorem carrie_spent :
  cost_of_one_tshirt * number_of_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l501_50127


namespace local_maximum_at_1_2_l501_50149

noncomputable def f (x1 x2 : ℝ) : ℝ := x2^2 - x1^2
def constraint (x1 x2 : ℝ) : Prop := x1 - 2 * x2 + 3 = 0
def is_local_maximum (f : ℝ → ℝ → ℝ) (x1 x2 : ℝ) : Prop := 
∃ ε > 0, ∀ (y1 y2 : ℝ), (constraint y1 y2 ∧ (y1 - x1)^2 + (y2 - x2)^2 < ε^2) → f y1 y2 ≤ f x1 x2

theorem local_maximum_at_1_2 : is_local_maximum f 1 2 :=
sorry

end local_maximum_at_1_2_l501_50149


namespace total_license_groups_l501_50156

-- Defining the given conditions
def letter_choices : Nat := 3
def digit_choices_per_slot : Nat := 10
def number_of_digit_slots : Nat := 5

-- Statement to prove that the total number of different license groups is 300000
theorem total_license_groups : letter_choices * (digit_choices_per_slot ^ number_of_digit_slots) = 300000 := by
  sorry

end total_license_groups_l501_50156


namespace value_of_b_l501_50181

-- Definitions
def A := 45  -- in degrees
def B := 60  -- in degrees
def a := 10  -- length of side a

-- Assertion
theorem value_of_b : (b : ℝ) = 5 * Real.sqrt 6 :=
by
  -- Definitions used in previous problem conditions
  let sin_A := Real.sin (Real.pi * A / 180)
  let sin_B := Real.sin (Real.pi * B / 180)
  -- Applying the Law of Sines
  have law_of_sines := (a / sin_A) = (b / sin_B)
  -- Simplified calculation of b (not provided here; proof required later)
  sorry

end value_of_b_l501_50181


namespace no_integer_solution_mx2_minus_sy2_eq_3_l501_50141

theorem no_integer_solution_mx2_minus_sy2_eq_3 (m s : ℤ) (x y : ℤ) (h : m * s = 2000 ^ 2001) :
  ¬ (m * x ^ 2 - s * y ^ 2 = 3) :=
sorry

end no_integer_solution_mx2_minus_sy2_eq_3_l501_50141


namespace max_value_of_a_l501_50143

theorem max_value_of_a (a b c d : ℤ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : d < 100) : a ≤ 2367 := by 
  sorry

end max_value_of_a_l501_50143


namespace unique_function_satisfies_sum_zero_l501_50114

theorem unique_function_satisfies_sum_zero 
  (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x^3) = (f x)^3)
  (h2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2) : 
  f 0 + f 1 + f (-1) = 0 :=
sorry

end unique_function_satisfies_sum_zero_l501_50114


namespace solve_inequality_l501_50177

theorem solve_inequality (a : ℝ) : 
    (∀ x : ℝ, x^2 + (a + 2)*x + 2*a < 0 ↔ 
        (if a < 2 then -2 < x ∧ x < -a
         else if a = 2 then false
         else -a < x ∧ x < -2)) :=
by
  sorry

end solve_inequality_l501_50177


namespace log_domain_inequality_l501_50131

theorem log_domain_inequality {a : ℝ} : 
  (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ a > 1 :=
sorry

end log_domain_inequality_l501_50131


namespace probability_odd_even_draw_correct_l501_50151

noncomputable def probability_odd_even_draw : ℚ := sorry

theorem probability_odd_even_draw_correct :
  probability_odd_even_draw = 17 / 45 := 
sorry

end probability_odd_even_draw_correct_l501_50151


namespace expand_expression_l501_50135

theorem expand_expression (x : ℝ) : 16 * (2 * x + 5) = 32 * x + 80 :=
by
  sorry

end expand_expression_l501_50135


namespace new_trailer_homes_added_l501_50132

theorem new_trailer_homes_added
  (n : ℕ) (avg_age_3_years_ago avg_age_today age_increase new_home_age : ℕ) (k : ℕ) :
  n = 30 → avg_age_3_years_ago = 15 → avg_age_today = 12 → age_increase = 3 → new_home_age = 3 →
  (n * (avg_age_3_years_ago + age_increase) + k * new_home_age) / (n + k) = avg_age_today →
  k = 20 :=
by
  intros h_n h_avg_age_3y h_avg_age_today h_age_increase h_new_home_age h_eq
  sorry

end new_trailer_homes_added_l501_50132


namespace g_at_5_eq_9_l501_50198

-- Define the polynomial function g as given in the conditions
def g (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- Define the hypothesis that g(-5) = -3
axiom g_neg5 (a b c : ℝ) : g a b c (-5) = -3

-- State the theorem to prove that g(5) = 9 given the conditions
theorem g_at_5_eq_9 (a b c : ℝ) : g a b c 5 = 9 := 
by sorry

end g_at_5_eq_9_l501_50198


namespace reduced_fraction_numerator_l501_50168

theorem reduced_fraction_numerator :
  let numerator := 4128 
  let denominator := 4386 
  let gcd := Nat.gcd numerator denominator
  let reduced_numerator := numerator / gcd 
  let reduced_denominator := denominator / gcd 
  (reduced_numerator : ℚ) / (reduced_denominator : ℚ) = 16 / 17 → reduced_numerator = 16 :=
by
  intros
  sorry

end reduced_fraction_numerator_l501_50168


namespace angle_ratio_l501_50122

theorem angle_ratio (x y α β : ℝ)
  (h1 : y = x + β)
  (h2 : 2 * y = 2 * x + α) :
  α / β = 2 :=
by
  sorry

end angle_ratio_l501_50122


namespace factory_ill_days_l501_50110

theorem factory_ill_days
  (average_first_25_days : ℝ)
  (total_days : ℝ)
  (overall_average : ℝ)
  (ill_days_average : ℝ)
  (production_first_25_days_total : ℝ)
  (production_ill_days_total : ℝ)
  (x : ℝ) :
  average_first_25_days = 50 →
  total_days = 25 + x →
  overall_average = 48 →
  ill_days_average = 38 →
  production_first_25_days_total = 25 * 50 →
  production_ill_days_total = x * 38 →
  (25 * 50 + x * 38 = (25 + x) * 48) →
  x = 5 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end factory_ill_days_l501_50110


namespace point_A_coordinates_l501_50146

variable (a x y : ℝ)

def f (a x : ℝ) : ℝ := (a^2 - 1) * (x^2 - 1) + (a - 1) * (x - 1)

theorem point_A_coordinates (h1 : ∃ t : ℝ, ∀ x : ℝ, f a x = t * x + t) (h2 : x = 0) : (0, 2) = (0, f a 0) :=
by
  sorry

end point_A_coordinates_l501_50146


namespace sugar_ratio_l501_50192

theorem sugar_ratio (r : ℝ) (H1 : 24 * r^3 = 3) : (24 * r / 24 = 1 / 2) :=
by
  sorry

end sugar_ratio_l501_50192


namespace find_x_l501_50150

-- Definition of the problem conditions
def angle_ABC : ℝ := 85
def angle_BAC : ℝ := 55
def sum_angles_triangle (a b c : ℝ) : Prop := a + b + c = 180
def corresponding_angle (a b : ℝ) : Prop := a = b
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90

-- The theorem to prove
theorem find_x :
  ∀ (x BCA : ℝ), sum_angles_triangle angle_ABC angle_BAC BCA ∧ corresponding_angle BCA 40 ∧ right_triangle_sum BCA x → x = 50 :=
by
  intros x BCA h
  sorry

end find_x_l501_50150


namespace empty_rooms_le_1000_l501_50128

/--
In a 50x50 grid where each cell can contain at most one tree, 
with the following rules: 
1. A pomegranate tree has at least one apple neighbor
2. A peach tree has at least one apple neighbor and one pomegranate neighbor
3. An empty room has at least one apple neighbor, one pomegranate neighbor, and one peach neighbor
Show that the number of empty rooms is not greater than 1000.
-/
theorem empty_rooms_le_1000 (apple pomegranate peach : ℕ) (empty : ℕ)
  (h1 : apple + pomegranate + peach + empty = 2500)
  (h2 : ∀ p, pomegranate ≥ p → apple ≥ 1)
  (h3 : ∀ p, peach ≥ p → apple ≥ 1 ∧ pomegranate ≥ 1)
  (h4 : ∀ e, empty ≥ e → apple ≥ 1 ∧ pomegranate ≥ 1 ∧ peach ≥ 1) :
  empty ≤ 1000 :=
sorry

end empty_rooms_le_1000_l501_50128


namespace total_money_shared_l501_50158

/-- Assume there are four people Amanda, Ben, Carlos, and David, sharing an amount of money.
    Their portions are in the ratio 1:2:7:3.
    Amanda's portion is $20.
    Prove that the total amount of money shared by them is $260. -/
theorem total_money_shared (A B C D : ℕ) (h_ratio : A = 20 ∧ B = 2 * A ∧ C = 7 * A ∧ D = 3 * A) :
  A + B + C + D = 260 := by 
  sorry

end total_money_shared_l501_50158


namespace factorize_l501_50159

theorem factorize (x : ℝ) : 72 * x ^ 11 + 162 * x ^ 22 = 18 * x ^ 11 * (4 + 9 * x ^ 11) :=
by
  sorry

end factorize_l501_50159


namespace certain_number_is_11_l501_50176

theorem certain_number_is_11 (x : ℝ) (h : 15 * x = 165) : x = 11 :=
by {
  sorry
}

end certain_number_is_11_l501_50176


namespace matvey_healthy_diet_l501_50123

theorem matvey_healthy_diet (n b_1 p_1 : ℕ) (h1 : n * b_1 - (n * (n - 1)) / 2 = 264) (h2 : n * p_1 + (n * (n - 1)) / 2 = 187) :
  n = 11 :=
by
  let buns_diff_pears := b_1 - p_1 - (n - 1)
  have buns_def : 264 = n * buns_diff_pears + n * (n - 1) / 2 := sorry
  have pears_def : 187 = n * buns_diff_pears - n * (n - 1) / 2 := sorry
  have diff : 77 = n * buns_diff_pears := sorry
  sorry

end matvey_healthy_diet_l501_50123


namespace second_lock_less_than_three_times_first_l501_50189

variable (first_lock_time : ℕ := 5)
variable (second_lock_time : ℕ)
variable (combined_lock_time : ℕ := 60)

-- Assuming the second lock time is a fraction of the combined lock time
axiom h1 : 5 * second_lock_time = combined_lock_time

theorem second_lock_less_than_three_times_first : (3 * first_lock_time - second_lock_time) = 3 :=
by
  -- prove that the theorem is true based on given conditions.
  sorry

end second_lock_less_than_three_times_first_l501_50189


namespace scientific_notation_of_192M_l501_50154

theorem scientific_notation_of_192M : 192000000 = 1.92 * 10^8 :=
by 
  sorry

end scientific_notation_of_192M_l501_50154


namespace area_acpq_eq_sum_areas_aekl_cdmn_l501_50111

variables (A B C D E P Q M N K L : Point)

def is_acute_angled_triangle (A B C : Point) : Prop := sorry
def is_altitude (A B C D : Point) : Prop := sorry
def is_square (A P Q C : Point) : Prop := sorry
def is_rectangle (A E K L : Point) : Prop := sorry
def is_rectangle' (C D M N : Point) : Prop := sorry
def length (P Q : Point) : Real := sorry
def area (P Q R S : Point) : Real := sorry

-- Conditions
axiom abc_acute : is_acute_angled_triangle A B C
axiom ad_altitude : is_altitude A B C D
axiom ce_altitude : is_altitude C A B E
axiom acpq_square : is_square A P Q C
axiom aekl_rectangle : is_rectangle A E K L
axiom cdmn_rectangle : is_rectangle' C D M N
axiom al_eq_ab : length A L = length A B
axiom cn_eq_cb : length C N = length C B

-- Question proof statement
theorem area_acpq_eq_sum_areas_aekl_cdmn :
  area A C P Q = area A E K L + area C D M N :=
sorry

end area_acpq_eq_sum_areas_aekl_cdmn_l501_50111


namespace determine_disco_ball_price_l501_50144

variable (x y z : ℝ)

-- Given conditions
def budget_constraint : Prop := 4 * x + 10 * y + 20 * z = 600
def food_cost : Prop := y = 0.85 * x
def decoration_cost : Prop := z = x / 2 - 10

-- Goal
theorem determine_disco_ball_price (h1 : budget_constraint x y z) (h2 : food_cost x y) (h3 : decoration_cost x z) :
  x = 35.56 :=
sorry 

end determine_disco_ball_price_l501_50144


namespace total_percentage_increase_l501_50112

def initial_salary : Float := 60
def first_raise (s : Float) : Float := s + 0.10 * s
def second_raise (s : Float) : Float := s + 0.15 * s
def deduction (s : Float) : Float := s - 0.05 * s
def promotion_raise (s : Float) : Float := s + 0.20 * s
def final_salary (s : Float) : Float := promotion_raise (deduction (second_raise (first_raise s)))

theorem total_percentage_increase :
  final_salary initial_salary = initial_salary * 1.4421 :=
by
  sorry

end total_percentage_increase_l501_50112


namespace nap_time_is_correct_l501_50109

-- Define the total trip time and the hours spent on each activity
def total_trip_time : ℝ := 15
def reading_time : ℝ := 2
def eating_time : ℝ := 1
def movies_time : ℝ := 3
def chatting_time : ℝ := 1
def browsing_time : ℝ := 0.75
def waiting_time : ℝ := 0.5
def working_time : ℝ := 2

-- Define the total activity time
def total_activity_time : ℝ := reading_time + eating_time + movies_time + chatting_time + browsing_time + waiting_time + working_time

-- Define the nap time as the difference between total trip time and total activity time
def nap_time : ℝ := total_trip_time - total_activity_time

-- Prove that the nap time is 4.75 hours
theorem nap_time_is_correct : nap_time = 4.75 :=
by
  -- Calculation hint, can be ignored
  -- nap_time = 15 - (2 + 1 + 3 + 1 + 0.75 + 0.5 + 2) = 15 - 10.25 = 4.75
  sorry

end nap_time_is_correct_l501_50109


namespace math_problem_l501_50164

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end math_problem_l501_50164


namespace contrapositive_eq_inverse_l501_50172

variable (p q : Prop)

theorem contrapositive_eq_inverse (h1 : p → q) :
  (¬ p → ¬ q) ↔ (q → p) := by
  sorry

end contrapositive_eq_inverse_l501_50172


namespace probability_girl_selection_l501_50105

-- Define the conditions
def total_candidates : ℕ := 3 + 1
def girl_candidates : ℕ := 1

-- Define the question in terms of probability
def probability_of_selecting_girl (total: ℕ) (girl: ℕ) : ℚ :=
  girl / total

-- Lean statement to prove
theorem probability_girl_selection : probability_of_selecting_girl total_candidates girl_candidates = 1 / 4 :=
by
  sorry

end probability_girl_selection_l501_50105


namespace amount_of_pizza_needed_l501_50120

theorem amount_of_pizza_needed :
  (1 / 2 + 1 / 3 + 1 / 6) = 1 := by
  sorry

end amount_of_pizza_needed_l501_50120


namespace total_annual_cost_l501_50118

def daily_pills : ℕ := 2
def pill_cost : ℕ := 5
def medication_cost (daily_pills : ℕ) (pill_cost : ℕ) : ℕ := daily_pills * pill_cost
def insurance_coverage : ℚ := 0.80
def visit_cost : ℕ := 400
def visits_per_year : ℕ := 2
def annual_medication_cost (medication_cost : ℕ) (insurance_coverage : ℚ) : ℚ :=
  medication_cost * 365 * (1 - insurance_coverage)
def annual_visit_cost (visit_cost : ℕ) (visits_per_year : ℕ) : ℕ :=
  visit_cost * visits_per_year

theorem total_annual_cost : annual_medication_cost (medication_cost daily_pills pill_cost) insurance_coverage
  + annual_visit_cost visit_cost visits_per_year = 1530 := by
  sorry

end total_annual_cost_l501_50118


namespace eval_expr_l501_50138

theorem eval_expr : (900 ^ 2) / (262 ^ 2 - 258 ^ 2) = 389.4 := 
by
  sorry

end eval_expr_l501_50138


namespace factorization_problem1_factorization_problem2_l501_50137

-- Define the first problem: Factorization of 3x^2 - 27
theorem factorization_problem1 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) :=
by
  sorry 

-- Define the second problem: Factorization of (a + 1)(a - 5) + 9
theorem factorization_problem2 (a : ℝ) : (a + 1) * (a - 5) + 9 = (a - 2) ^ 2 :=
by
  sorry

end factorization_problem1_factorization_problem2_l501_50137


namespace part_one_equation_of_line_part_two_equation_of_line_l501_50175

-- Definition of line passing through a given point
def line_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1 / a + P.2 / b = 1

-- Condition: the sum of intercepts is 12
def sum_of_intercepts (a b : ℝ) : Prop := a + b = 12

-- Condition: area of triangle is 12
def area_of_triangle (a b : ℝ) : Prop := (1/2) * (abs (a * b)) = 12

-- First part: equation of the line when the sum of intercepts is 12
theorem part_one_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (sum_of_intercepts a b) →
  (∃ x, (x = 2 ∧ (2*x)+x - 8 = 0) ∨ (x = 3 ∧ x + 3*x - 9 = 0)) :=
by
  sorry

-- Second part: equation of the line when the area of the triangle is 12
theorem part_two_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (area_of_triangle a b) →
  ∃ x, x = 2 ∧ (2*x + 3*x - 12 = 0) :=
by
  sorry

end part_one_equation_of_line_part_two_equation_of_line_l501_50175


namespace trigonometric_identity_l501_50121

theorem trigonometric_identity (x : ℝ) (h : Real.tan (x + Real.pi / 2) = 5) : 
  1 / (Real.sin x * Real.cos x) = -26 / 5 :=
by
  sorry

end trigonometric_identity_l501_50121


namespace initial_population_l501_50140

theorem initial_population (P : ℝ) 
    (h1 : 1.25 * P * 0.70 = 363650) : 
    P = 415600 :=
sorry

end initial_population_l501_50140


namespace avg_price_pen_is_correct_l501_50155

-- Definitions for the total numbers and expenses:
def number_of_pens : ℕ := 30
def number_of_pencils : ℕ := 75
def total_cost : ℕ := 630
def avg_price_pencil : ℝ := 2.00

-- Calculation of total cost for pencils and pens
def total_cost_pencils : ℝ := number_of_pencils * avg_price_pencil
def total_cost_pens : ℝ := total_cost - total_cost_pencils

-- Statement to prove:
theorem avg_price_pen_is_correct :
  total_cost_pens / number_of_pens = 16 :=
by
  sorry

end avg_price_pen_is_correct_l501_50155


namespace problem1_problem2_problem3_l501_50196

noncomputable 
def f (x : ℝ) : ℝ := Real.exp x

theorem problem1 
  (a b : ℝ)
  (h1 : f 1 = a) 
  (h2 : b = 0) : f x = Real.exp x :=
sorry

theorem problem2 
  (k : ℝ) 
  (h : ∀ x : ℝ, f x ≥ k * x) : 0 ≤ k ∧ k ≤ Real.exp 1 :=
sorry

theorem problem3 
  (t : ℝ)
  (h : t ≤ 2) : ∀ x : ℝ, f x > t + Real.log x :=
sorry

end problem1_problem2_problem3_l501_50196


namespace sequence_sum_l501_50136

-- Assume the sum of first n terms of the sequence {a_n} is given by S_n = n^2 + n + 1
def S (n : ℕ) : ℕ := n^2 + n + 1

-- The sequence a_8 + a_9 + a_10 + a_11 + a_12 is what we want to prove equals 100.
theorem sequence_sum : S 12 - S 7 = 100 :=
by
  sorry

end sequence_sum_l501_50136


namespace constants_solution_l501_50117

theorem constants_solution : ∀ (x : ℝ), x ≠ 0 ∧ x^2 ≠ 2 →
  (2 * x^2 - 5 * x + 1) / (x^3 - 2 * x) = (-1 / 2) / x + (2.5 * x - 5) / (x^2 - 2) := by
  intros x hx
  sorry

end constants_solution_l501_50117


namespace circle_condition_l501_50115

theorem circle_condition (m : ℝ) :
    (4 * m) ^ 2 + 4 - 4 * 5 * m > 0 ↔ (m < 1 / 4 ∨ m > 1) := sorry

end circle_condition_l501_50115


namespace exists_quadratic_polynomial_distinct_remainders_l501_50193

theorem exists_quadratic_polynomial_distinct_remainders :
  ∃ (a b c : ℤ), 
    (¬ (2014 ∣ a)) ∧ 
    (∀ x y : ℤ, (1 ≤ x ∧ x ≤ 2014) ∧ (1 ≤ y ∧ y ≤ 2014) → x ≠ y → 
      (1007 * x^2 + 1008 * x + c) % 2014 ≠ (1007 * y^2 + 1008 * y + c) % 2014) :=
  sorry

end exists_quadratic_polynomial_distinct_remainders_l501_50193


namespace triangle_30_60_90_PQ_l501_50183

theorem triangle_30_60_90_PQ (PR : ℝ) (hPR : PR = 18 * Real.sqrt 3) : 
  ∃ PQ : ℝ, PQ = 54 :=
by
  sorry

end triangle_30_60_90_PQ_l501_50183


namespace not_product_of_consecutive_integers_l501_50182

theorem not_product_of_consecutive_integers (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ (m : ℕ), 2 * (n ^ k) ^ 3 + 4 * (n ^ k) + 10 ≠ m * (m + 1) := by
sorry

end not_product_of_consecutive_integers_l501_50182


namespace smallest_coins_l501_50194

theorem smallest_coins (n : ℕ) (n_min : ℕ) (h1 : ∃ n, n % 8 = 5 ∧ n % 7 = 4 ∧ n = 53) (h2 : n_min = n):
  (n_min ≡ 5 [MOD 8]) ∧ (n_min ≡ 4 [MOD 7]) ∧ (n_min = 53) ∧ (53 % 9 = 8) :=
by
  sorry

end smallest_coins_l501_50194


namespace product_mod_32_is_15_l501_50100

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end product_mod_32_is_15_l501_50100


namespace area_of_triangle_tangent_line_l501_50103

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def tangent_line_at_1 (y x : ℝ) : Prop := y = x - 1

theorem area_of_triangle_tangent_line :
  let tangent_intercept_x : ℝ := 1
  let tangent_intercept_y : ℝ := -1
  let area_of_triangle : ℝ := 1 / 2 * tangent_intercept_x * -tangent_intercept_y
  area_of_triangle = 1 / 2 :=
by
  sorry

end area_of_triangle_tangent_line_l501_50103


namespace quadratic_equation_m_l501_50186

theorem quadratic_equation_m (m b : ℝ) (h : (m - 2) * x ^ |m| - b * x - 1 = 0) : m = -2 :=
by
  sorry

end quadratic_equation_m_l501_50186


namespace youngest_child_age_l501_50199

theorem youngest_child_age :
  ∃ x : ℕ, x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65 ∧ x = 7 :=
by
  sorry

end youngest_child_age_l501_50199


namespace smallest_positive_y_l501_50187

theorem smallest_positive_y (y : ℕ) (h : 42 * y + 8 ≡ 4 [MOD 24]) : y = 2 :=
sorry

end smallest_positive_y_l501_50187


namespace cos_alpha_value_l501_50178

-- Define our conditions
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -5 / 13
axiom tan_alpha_pos : Real.tan α > 0

-- State our goal
theorem cos_alpha_value : Real.cos α = -12 / 13 :=
by
  sorry

end cos_alpha_value_l501_50178


namespace compound_interest_principal_l501_50126

theorem compound_interest_principal 
  (CI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hCI : CI = 315)
  (hR : R = 10)
  (hT : T = 2) :
  CI = P * ((1 + R / 100)^T - 1) → P = 1500 := by
  sorry

end compound_interest_principal_l501_50126


namespace last_three_digits_of_2_pow_9000_l501_50169

-- The proof statement
theorem last_three_digits_of_2_pow_9000 (h : 2 ^ 300 ≡ 1 [MOD 1000]) : 2 ^ 9000 ≡ 1 [MOD 1000] :=
by
  sorry

end last_three_digits_of_2_pow_9000_l501_50169


namespace find_a_for_cubic_sum_l501_50104

theorem find_a_for_cubic_sum (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - a * x1 + a + 2 = 0 ∧ 
    x2^2 - a * x2 + a + 2 = 0 ∧
    x1 + x2 = a ∧
    x1 * x2 = a + 2 ∧
    x1^3 + x2^3 = -8) ↔ a = -2 := 
by
  sorry

end find_a_for_cubic_sum_l501_50104


namespace evaluate_y_correct_l501_50148

noncomputable def evaluate_y (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - 4 * x + 4) + Real.sqrt (x^2 + 6 * x + 9) - 2

theorem evaluate_y_correct (x : ℝ) : 
  evaluate_y x = |x - 2| + |x + 3| - 2 :=
by 
  sorry

end evaluate_y_correct_l501_50148


namespace number_of_grade2_students_l501_50108

theorem number_of_grade2_students (ratio1 ratio2 ratio3 : ℕ) (total_students : ℕ) (ratio_sum : ratio1 + ratio2 + ratio3 = 12)
  (total_sample_size : total_students = 240) : 
  total_students * ratio2 / (ratio1 + ratio2 + ratio3) = 80 :=
by
  have ratio1_val : ratio1 = 5 := sorry
  have ratio2_val : ratio2 = 4 := sorry
  have ratio3_val : ratio3 = 3 := sorry
  rw [ratio1_val, ratio2_val, ratio3_val] at ratio_sum
  rw [ratio1_val, ratio2_val, ratio3_val]
  exact sorry

end number_of_grade2_students_l501_50108


namespace trig_identity_l501_50102

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l501_50102


namespace coeff_x3_in_product_l501_50188

theorem coeff_x3_in_product :
  let p1 := 3 * (Polynomial.X ^ 3) + 4 * (Polynomial.X ^ 2) + 5 * Polynomial.X + 6
  let p2 := 7 * (Polynomial.X ^ 2) + 8 * Polynomial.X + 9
  (Polynomial.coeff (p1 * p2) 3) = 94 :=
by
  sorry

end coeff_x3_in_product_l501_50188


namespace smallest_n_satisfies_condition_l501_50174

theorem smallest_n_satisfies_condition : 
  ∃ (n : ℕ), n = 1806 ∧ ∀ (p : ℕ), Nat.Prime p → n % (p - 1) = 0 → n % p = 0 := 
sorry

end smallest_n_satisfies_condition_l501_50174


namespace polygon_perimeter_l501_50116

theorem polygon_perimeter (a b : ℕ) (h : adjacent_sides_perpendicular) :
  perimeter = 2 * (a + b) :=
sorry

end polygon_perimeter_l501_50116


namespace find_f_neg_one_l501_50166

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * tan x + 3

theorem find_f_neg_one (a b : ℝ) (h : f a b 1 = 1) : f a b (-1) = 5 :=
by
  sorry

end find_f_neg_one_l501_50166


namespace middle_number_is_five_l501_50160

theorem middle_number_is_five
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_sum : a + b + c = 20)
  (h_sorted : a < b ∧ b < c)
  (h_bella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → x = a → y = b ∧ z = c)
  (h_della : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → y = b → x = a ∧ z = c)
  (h_nella : ¬∀ x y z, x + y + z = 20 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x < y ∧ y < z → z = c → x = a ∧ y = b) :
  b = 5 := sorry

end middle_number_is_five_l501_50160


namespace pool_surface_area_l501_50162

/-
  Given conditions:
  1. The width of the pool is 3 meters.
  2. The length of the pool is 10 meters.

  To prove:
  The surface area of the pool is 30 square meters.
-/
def width : ℕ := 3
def length : ℕ := 10
def surface_area (length width : ℕ) : ℕ := length * width

theorem pool_surface_area : surface_area length width = 30 := by
  unfold surface_area
  rfl

end pool_surface_area_l501_50162


namespace polynomial_sum_squares_l501_50195

theorem polynomial_sum_squares (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ)
  (h₁ : (1 - 2) ^ 7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)
  (h₂ : (1 + -2) ^ 7 = a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7) :
  (a0 + a2 + a4 + a6) ^ 2 - (a1 + a3 + a5 + a7) ^ 2 = -2187 := 
  sorry

end polynomial_sum_squares_l501_50195


namespace outfit_combinations_l501_50134

def shirts : ℕ := 6
def pants : ℕ := 4
def hats : ℕ := 6

def pant_colors : Finset String := {"tan", "black", "blue", "gray"}
def shirt_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}

def total_combinations : ℕ := shirts * pants * hats
def restricted_combinations : ℕ := pant_colors.card

theorem outfit_combinations
    (hshirts : shirts = 6)
    (hpants : pants = 4)
    (hhats : hats = 6)
    (hpant_colors : pant_colors.card = 4)
    (hshirt_colors : shirt_colors.card = 6)
    (hhat_colors : hat_colors.card = 6)
    (hrestricted : restricted_combinations = pant_colors.card) :
    total_combinations - restricted_combinations = 140 := by
  sorry

end outfit_combinations_l501_50134


namespace number_of_valid_triangles_l501_50142

-- Definition of the set of points in the 5x5 grid with integer coordinates
def gridPoints := {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Function to determine if three points are collinear
def collinear (a b c : ℕ × ℕ) : Prop :=
  (b.2 - a.2) * (c.1 - b.1) = (c.2 - b.2) * (b.1 - a.1)

-- The main theorem stating the number of triangles with positive area
theorem number_of_valid_triangles : 
  ∃ n, n = 2158 ∧ ∀ (a b c : ℕ × ℕ), a ∈ gridPoints → b ∈ gridPoints → c ∈ gridPoints → a ≠ b → b ≠ c → c ≠ a → ¬collinear a b c → n = 2158 :=
by
  sorry

end number_of_valid_triangles_l501_50142


namespace pqr_value_l501_50113

theorem pqr_value
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 29)
  (h_eq : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 :=
by
  sorry

end pqr_value_l501_50113


namespace students_side_by_side_with_A_and_B_l501_50130

theorem students_side_by_side_with_A_and_B (total students_from_club_A students_from_club_B: ℕ) 
    (h1 : total = 100)
    (h2 : students_from_club_A = 62)
    (h3 : students_from_club_B = 54) :
  ∃ p q r : ℕ, p + q + r = 100 ∧ p + q = 62 ∧ p + r = 54 ∧ p = 16 :=
by
  sorry

end students_side_by_side_with_A_and_B_l501_50130


namespace find_ab_l501_50180

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 15 :=
by
  sorry

end find_ab_l501_50180


namespace problem_1_problem_2_l501_50153

theorem problem_1 {m : ℝ} (h₁ : 0 < m) (h₂ : ∀ x : ℝ, (m - |x + 2| ≥ 0) ↔ (-3 ≤ x ∧ x ≤ -1)) :
  m = 1 :=
sorry

theorem problem_2 {a b c : ℝ} (h₃ : 0 < a ∧ 0 < b ∧ 0 < c) (h₄ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1)
  : a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_1_problem_2_l501_50153


namespace average_salary_all_employees_l501_50173

-- Define the given conditions
def average_salary_officers : ℝ := 440
def average_salary_non_officers : ℝ := 110
def number_of_officers : ℕ := 15
def number_of_non_officers : ℕ := 480

-- Define the proposition we need to prove
theorem average_salary_all_employees :
  let total_salary_officers := average_salary_officers * number_of_officers
  let total_salary_non_officers := average_salary_non_officers * number_of_non_officers
  let total_salary_all_employees := total_salary_officers + total_salary_non_officers
  let total_number_of_employees := number_of_officers + number_of_non_officers
  let average_salary_all_employees := total_salary_all_employees / total_number_of_employees
  average_salary_all_employees = 120 :=
by {
  -- Skipping the proof steps
  sorry
}

end average_salary_all_employees_l501_50173


namespace percentage_increase_in_savings_l501_50157

theorem percentage_increase_in_savings
  (I : ℝ) -- Original income of Paulson
  (E : ℝ) -- Original expenditure of Paulson
  (hE : E = 0.75 * I) -- Paulson spends 75% of his income
  (h_inc_income : 1.2 * I = I + 0.2 * I) -- Income is increased by 20%
  (h_inc_expenditure : 0.825 * I = 0.75 * I + 0.1 * (0.75 * I)) -- Expenditure is increased by 10%
  : (0.375 * I - 0.25 * I) / (0.25 * I) * 100 = 50 := by
  sorry

end percentage_increase_in_savings_l501_50157


namespace xiaoming_wait_probability_l501_50147

-- Conditions
def green_light_duration : ℕ := 40
def red_light_duration : ℕ := 50
def total_light_cycle : ℕ := green_light_duration + red_light_duration
def waiting_time_threshold : ℕ := 20
def long_wait_interval : ℕ := 30 -- from problem (20 seconds to wait corresponds to 30 seconds interval)

-- Probability calculation
theorem xiaoming_wait_probability :
  ∀ (arrival_time : ℕ), arrival_time < total_light_cycle →
    (30 : ℝ) / (total_light_cycle : ℝ) = 1 / 3 := by sorry

end xiaoming_wait_probability_l501_50147


namespace cos_240_eq_neg_half_l501_50185

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l501_50185


namespace clay_capacity_second_box_l501_50179

-- Define the dimensions and clay capacity of the first box
def height1 : ℕ := 4
def width1 : ℕ := 2
def length1 : ℕ := 3
def clay1 : ℕ := 24

-- Define the dimensions of the second box
def height2 : ℕ := 3 * height1
def width2 : ℕ := 2 * width1
def length2 : ℕ := length1

-- The volume relation
def volume_relation (height width length clay: ℕ) : ℕ :=
  height * width * length * clay

theorem clay_capacity_second_box (height1 width1 length1 clay1 : ℕ) (height2 width2 length2 : ℕ) :
  height1 = 4 →
  width1 = 2 →
  length1 = 3 →
  clay1 = 24 →
  height2 = 3 * height1 →
  width2 = 2 * width1 →
  length2 = length1 →
  volume_relation height2 width2 length2 1 = 6 * volume_relation height1 width1 length1 1 →
  volume_relation height2 width2 length2 clay1 / volume_relation height1 width1 length1 1 = 144 :=
by
  intros h1 w1 l1 c1 h2 w2 l2 vol_rel
  sorry

end clay_capacity_second_box_l501_50179


namespace opposite_of_minus_one_third_l501_50107

theorem opposite_of_minus_one_third :
  -(- (1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_minus_one_third_l501_50107


namespace other_solution_of_quadratic_l501_50165

theorem other_solution_of_quadratic (x : ℚ) (h₁ : 81 * 2/9 * 2/9 + 220 = 196 * 2/9 - 15) (h₂ : 81*x^2 - 196*x + 235 = 0) : x = 2/9 ∨ x = 5/9 :=
by
  sorry

end other_solution_of_quadratic_l501_50165


namespace length_of_PR_l501_50119

-- Define the entities and conditions
variables (x y : ℝ)
variables (xy_area : ℝ := 125)
variables (PR_length : ℝ := 10 * Real.sqrt 5)

-- State the problem in Lean
theorem length_of_PR (x y : ℝ) (hxy : x * y = 125) :
  x^2 + (125 / x)^2 = (10 * Real.sqrt 5)^2 :=
sorry

end length_of_PR_l501_50119


namespace area_to_be_painted_l501_50171

def wall_height : ℕ := 8
def wall_length : ℕ := 15
def glass_painting_height : ℕ := 3
def glass_painting_length : ℕ := 5

theorem area_to_be_painted :
  (wall_height * wall_length) - (glass_painting_height * glass_painting_length) = 105 := by
  sorry

end area_to_be_painted_l501_50171


namespace remainder_when_divided_by_32_l501_50139

def product_of_odds : ℕ := 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem remainder_when_divided_by_32 : product_of_odds % 32 = 9 := by
  sorry

end remainder_when_divided_by_32_l501_50139


namespace compute_f_g_f_3_l501_50184

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 4

theorem compute_f_g_f_3 : f (g (f 3)) = 625 := sorry

end compute_f_g_f_3_l501_50184
