import Mathlib

namespace abs_inequality_solution_l1285_128546

theorem abs_inequality_solution :
  {x : ℝ | |x + 2| > 3} = {x : ℝ | x < -5} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l1285_128546


namespace gcd_min_value_l1285_128505

theorem gcd_min_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 :=
by
  sorry

end gcd_min_value_l1285_128505


namespace lights_on_top_layer_l1285_128571

theorem lights_on_top_layer
  (x : ℕ)
  (H1 : x + 2 * x + 4 * x + 8 * x + 16 * x + 32 * x + 64 * x = 381) :
  x = 3 :=
  sorry

end lights_on_top_layer_l1285_128571


namespace kendra_change_and_discounts_l1285_128548

-- Define the constants and conditions
def wooden_toy_price : ℝ := 20.0
def hat_price : ℝ := 10.0
def tax_rate : ℝ := 0.08
def discount_wooden_toys_2_3 : ℝ := 0.10
def discount_wooden_toys_4_or_more : ℝ := 0.15
def discount_hats_2 : ℝ := 0.05
def discount_hats_3_or_more : ℝ := 0.10
def kendra_bill : ℝ := 250.0
def kendra_wooden_toys : ℕ := 4
def kendra_hats : ℕ := 5

-- Calculate the applicable discounts based on conditions
def discount_on_wooden_toys : ℝ :=
  if kendra_wooden_toys >= 2 ∧ kendra_wooden_toys <= 3 then
    discount_wooden_toys_2_3
  else if kendra_wooden_toys >= 4 then
    discount_wooden_toys_4_or_more
  else
    0.0

def discount_on_hats : ℝ :=
  if kendra_hats = 2 then
    discount_hats_2
  else if kendra_hats >= 3 then
    discount_hats_3_or_more
  else
    0.0

-- Main theorem statement
theorem kendra_change_and_discounts :
  let total_cost_before_discounts := kendra_wooden_toys * wooden_toy_price + kendra_hats * hat_price
  let wooden_toys_discount := discount_on_wooden_toys * (kendra_wooden_toys * wooden_toy_price)
  let hats_discount := discount_on_hats * (kendra_hats * hat_price)
  let total_discounts := wooden_toys_discount + hats_discount
  let total_cost_after_discounts := total_cost_before_discounts - total_discounts
  let tax := tax_rate * total_cost_after_discounts
  let total_cost_after_tax := total_cost_after_discounts + tax
  let change_received := kendra_bill - total_cost_after_tax
  (total_discounts = 17) → 
  (change_received = 127.96) ∧ 
  (wooden_toys_discount = 12) ∧ 
  (hats_discount = 5) :=
by
  sorry

end kendra_change_and_discounts_l1285_128548


namespace prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l1285_128574

-- Definitions of the problem conditions
def positive_reviews_A := 75
def neutral_reviews_A := 20
def negative_reviews_A := 5
def total_reviews_A := 100

def positive_reviews_B := 64
def neutral_reviews_B := 8
def negative_reviews_B := 8
def total_reviews_B := 80

-- Prove the probability that a buyer's evaluation on platform A is not a negative review
theorem prob_not_negative_review_A : 
  (1 - negative_reviews_A / total_reviews_A) = 19 / 20 := by
  sorry

-- Prove the probability that exactly 2 out of 4 (2 from A and 2 from B) buyers give a positive review
theorem prob_two_positive_reviews :
  ((positive_reviews_A / total_reviews_A) ^ 2 * (1 - positive_reviews_B / total_reviews_B) ^ 2 + 
  2 * (positive_reviews_A / total_reviews_A) * (1 - positive_reviews_A / total_reviews_A) * 
  (positive_reviews_B / total_reviews_B) * (1 - positive_reviews_B / total_reviews_B) +
  (1 - positive_reviews_A / total_reviews_A) ^ 2 * (positive_reviews_B / total_reviews_B) ^ 2) = 
  73 / 400 := by
  sorry

-- Choose platform A based on the given data
theorem choose_platform_A :
  let E_A := (5 * 0.75 + 3 * 0.2 + 1 * 0.05)
  let D_A := (5 - E_A) ^ 2 * 0.75 + (3 - E_A) ^ 2 * 0.2 + (1 - E_A) ^ 2 * 0.05
  let E_B := (5 * 0.8 + 3 * 0.1 + 1 * 0.1)
  let D_B := (5 - E_B) ^ 2 * 0.8 + (3 - E_B) ^ 2 * 0.1 + (1 - E_B) ^ 2 * 0.1
  (E_A = E_B) ∧ (D_A < D_B) → choose_platform = "Platform A" := by
  sorry

end prob_not_negative_review_A_prob_two_positive_reviews_choose_platform_A_l1285_128574


namespace find_x0_l1285_128515

noncomputable def slopes_product_eq_three (x : ℝ) : Prop :=
  let y1 := 2 - 1 / x
  let y2 := x^3 - x^2 + 2 * x
  let dy1_dx := 1 / (x^2)
  let dy2_dx := 3 * x^2 - 2 * x + 2
  dy1_dx * dy2_dx = 3

theorem find_x0 : ∃ (x0 : ℝ), slopes_product_eq_three x0 ∧ x0 = 1 :=
by {
  use 1,
  sorry
}

end find_x0_l1285_128515


namespace train_length_l1285_128582

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_sec : ℝ := 3
noncomputable def speed_m_s := speed_km_hr * 1000 / 3600
noncomputable def length_of_train := speed_m_s * time_sec

theorem train_length :
  length_of_train = 50.01 := by
  sorry

end train_length_l1285_128582


namespace margo_pairing_probability_l1285_128529

theorem margo_pairing_probability (students : Finset ℕ)
  (H_50_students : students.card = 50)
  (margo irma jess kurt : ℕ)
  (H_margo_in_students : margo ∈ students)
  (H_irma_in_students : irma ∈ students)
  (H_jess_in_students : jess ∈ students)
  (H_kurt_in_students : kurt ∈ students)
  (possible_partners : Finset ℕ := students.erase margo) :
  (3: ℝ) / 49 = ((3: ℝ) / (possible_partners.card: ℝ)) :=
by
  -- The actual steps of the proof will be here
  sorry

end margo_pairing_probability_l1285_128529


namespace number_of_parents_l1285_128506

theorem number_of_parents (P : ℕ) (h : P + 177 = 238) : P = 61 :=
by
  sorry

end number_of_parents_l1285_128506


namespace points_per_enemy_l1285_128545

theorem points_per_enemy (total_enemies destroyed_enemies points_earned points_per_enemy : ℕ)
  (h1 : total_enemies = 8)
  (h2 : destroyed_enemies = total_enemies - 6)
  (h3 : points_earned = 10)
  (h4 : points_per_enemy = points_earned / destroyed_enemies) : 
  points_per_enemy = 5 := 
by
  sorry

end points_per_enemy_l1285_128545


namespace housewife_spending_l1285_128523

theorem housewife_spending
    (R : ℝ) (P : ℝ) (M : ℝ)
    (h1 : R = 25)
    (h2 : R = 0.85 * P)
    (h3 : M / R - M / P = 3) :
  M = 450 :=
by
  sorry

end housewife_spending_l1285_128523


namespace moe_cannot_finish_on_time_l1285_128520

theorem moe_cannot_finish_on_time (lawn_length lawn_width : ℝ) (swath : ℕ) (overlap : ℕ) (speed : ℝ) (available_time : ℝ) :
  lawn_length = 120 ∧ lawn_width = 180 ∧ swath = 30 ∧ overlap = 6 ∧ speed = 4000 ∧ available_time = 2 →
  (lawn_width / (swath - overlap) * lawn_length / speed) > available_time :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end moe_cannot_finish_on_time_l1285_128520


namespace number_of_attendants_writing_with_both_l1285_128514

-- Definitions for each of the conditions
def attendants_using_pencil : ℕ := 25
def attendants_using_pen : ℕ := 15
def attendants_using_only_one : ℕ := 20

-- Theorem that states the mathematically equivalent proof problem
theorem number_of_attendants_writing_with_both 
  (p : ℕ := attendants_using_pencil)
  (e : ℕ := attendants_using_pen)
  (o : ℕ := attendants_using_only_one) : 
  ∃ x, (p - x) + (e - x) = o ∧ x = 10 :=
by
  sorry

end number_of_attendants_writing_with_both_l1285_128514


namespace renovation_costs_l1285_128551

theorem renovation_costs :
  ∃ (x y : ℝ), 
    8 * x + 8 * y = 3520 ∧
    6 * x + 12 * y = 3480 ∧
    x = 300 ∧
    y = 140 ∧
    300 * 12 > 140 * 24 :=
by sorry

end renovation_costs_l1285_128551


namespace crop_fraction_brought_to_AD_l1285_128500

theorem crop_fraction_brought_to_AD
  (AD BC AB CD : ℝ)
  (h : ℝ)
  (angle : ℝ)
  (AD_eq_150 : AD = 150)
  (BC_eq_100 : BC = 100)
  (AB_eq_130 : AB = 130)
  (CD_eq_130 : CD = 130)
  (angle_eq_75 : angle = 75)
  (height_eq : h = (AB / 2) * Real.sin (angle * Real.pi / 180)) -- converting degrees to radians
  (area_trap : ℝ)
  (upper_area : ℝ)
  (total_area_eq : area_trap = (1 / 2) * (AD + BC) * h)
  (upper_area_eq : upper_area = (1 / 2) * (AD + (BC / 2)) * h)
  : (upper_area / area_trap) = 0.8 := 
sorry

end crop_fraction_brought_to_AD_l1285_128500


namespace prob_students_on_both_days_l1285_128522
noncomputable def probability_event_on_both_days: ℚ := by
  let total_days := 2
  let total_students := 4
  let prob_single_day := (1 / total_days : ℚ) ^ total_students
  let prob_all_same_day := 2 * prob_single_day
  let prob_both_days := 1 - prob_all_same_day
  exact prob_both_days

theorem prob_students_on_both_days : probability_event_on_both_days = 7 / 8 :=
by
  exact sorry

end prob_students_on_both_days_l1285_128522


namespace concentration_after_removing_water_l1285_128501

theorem concentration_after_removing_water :
  ∀ (initial_volume : ℝ) (initial_percentage : ℝ) (water_removed : ℝ),
  initial_volume = 18 →
  initial_percentage = 0.4 →
  water_removed = 6 →
  (initial_percentage * initial_volume) / (initial_volume - water_removed) * 100 = 60 :=
by
  intros initial_volume initial_percentage water_removed h1 h2 h3
  rw [h1, h2, h3]
  sorry

end concentration_after_removing_water_l1285_128501


namespace number_of_blind_students_l1285_128552

variable (B D : ℕ)

-- Condition 1: The deaf-student population is 3 times the blind-student population.
axiom H1 : D = 3 * B

-- Condition 2: There are 180 students in total.
axiom H2 : B + D = 180

theorem number_of_blind_students : B = 45 :=
by
  -- Sorry is used to skip the proof steps. The theorem statement is correct and complete based on the conditions.
  sorry

end number_of_blind_students_l1285_128552


namespace multiplication_difference_l1285_128504

theorem multiplication_difference :
  672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end multiplication_difference_l1285_128504


namespace hendecagon_diagonals_l1285_128588

-- Define the number of sides n of the hendecagon
def n : ℕ := 11

-- Define the formula for calculating the number of diagonals in an n-sided polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that there are 44 diagonals in a hendecagon
theorem hendecagon_diagonals : diagonals n = 44 :=
by
  -- Proof is skipped using sorry
  sorry

end hendecagon_diagonals_l1285_128588


namespace complex_real_imag_eq_l1285_128581

theorem complex_real_imag_eq (b : ℝ) (h : (2 + b) / 5 = (2 * b - 1) / 5) : b = 3 :=
  sorry

end complex_real_imag_eq_l1285_128581


namespace length_of_ad_l1285_128537

theorem length_of_ad (AB CD AD BC : ℝ) 
  (h1 : AB = 10) 
  (h2 : CD = 2 * AB) 
  (h3 : AD = BC) 
  (h4 : AB + BC + CD + AD = 42) : AD = 6 :=
by
  -- proof omitted
  sorry

end length_of_ad_l1285_128537


namespace integer_values_abc_l1285_128564

theorem integer_values_abc {a b c : ℤ} :
  a^2 + b^2 + c^2 + 3 < a * b + 3 * b + 2 * c ↔ (a = 1 ∧ b = 2 ∧ c = 1) :=
by
  sorry -- Proof to be filled

end integer_values_abc_l1285_128564


namespace spending_limit_l1285_128549

variable (n b total_spent limit: ℕ)

theorem spending_limit (hne: n = 34) (hbe: b = n + 5) (hts: total_spent = n + b) (hlo: total_spent = limit + 3) : limit = 70 := by
  sorry

end spending_limit_l1285_128549


namespace problem1_problem2_l1285_128591

-- Define the triangle and the condition a + 2a * cos B = c
variable {A B C : ℝ} (a b c : ℝ)
variable (cos_B : ℝ) -- cosine of angle B

-- Condition: a + 2a * cos B = c
variable (h1 : a + 2 * a * cos_B = c)

-- (I) Prove B = 2A
theorem problem1 (h1 : a + 2 * a * cos_B = c) : B = 2 * A :=
sorry

-- Define the acute triangle condition
variable (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Given: c = 2
variable (h2 : c = 2)

-- (II) Determine the range for a if the triangle is acute and c = 2
theorem problem2 (h1 : a + 2 * a * cos_B = 2) (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : 1 < a ∧ a < 2 :=
sorry

end problem1_problem2_l1285_128591


namespace total_population_eq_51b_over_40_l1285_128578

variable (b g t : Nat)

-- Conditions
def boys_eq_four_times_girls (b g : Nat) : Prop := b = 4 * g
def girls_eq_ten_times_teachers (g t : Nat) : Prop := g = 10 * t

-- Statement to prove
theorem total_population_eq_51b_over_40 (b g t : Nat) 
  (h1 : boys_eq_four_times_girls b g) 
  (h2 : girls_eq_ten_times_teachers g t) : 
  b + g + t = (51 * b) / 40 := 
sorry

end total_population_eq_51b_over_40_l1285_128578


namespace decimal_to_binary_18_l1285_128539

theorem decimal_to_binary_18 : (18: ℕ) = 0b10010 := by
  sorry

end decimal_to_binary_18_l1285_128539


namespace derivative_f_at_1_l1285_128540

noncomputable def f (x : Real) : Real := x^3 * Real.sin x

theorem derivative_f_at_1 : deriv f 1 = 3 * Real.sin 1 + Real.cos 1 := by
  sorry

end derivative_f_at_1_l1285_128540


namespace simplify_expr_1_simplify_expr_2_l1285_128561

theorem simplify_expr_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y :=
by
  sorry

theorem simplify_expr_2 (a b : ℝ) :
  (3 / 2) * (a^2 * b - 2 * (a * b^2)) - (1 / 2) * (a * b^2 - 4 * (a^2 * b)) + (a * b^2) / 2 = (7 / 2) * (a^2 * b) - 3 * (a * b^2) :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l1285_128561


namespace cannot_be_square_of_binomial_B_l1285_128593

theorem cannot_be_square_of_binomial_B (x y m n : ℝ) :
  (∃ (a b : ℝ), (3*x + 7*y) * (3*x - 7*y) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -0.2*x - 0.3) * ( -0.2*x + 0.3) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -3*n - m*n) * ( 3*n - m*n) = a^2 - b^2) ∧
  ¬(∃ (a b : ℝ), ( 5*m - n) * ( n - 5*m) = a^2 - b^2) :=
by
  sorry

end cannot_be_square_of_binomial_B_l1285_128593


namespace problem1_problem2_l1285_128524

-- Proof problem 1 statement in Lean 4
theorem problem1 :
  (1 : ℝ) * (Real.sqrt 2)^2 - |(1 : ℝ) - Real.sqrt 3| + Real.sqrt ((-3 : ℝ)^2) + Real.sqrt 81 = 15 - Real.sqrt 3 :=
by sorry

-- Proof problem 2 statement in Lean 4
theorem problem2 (x y : ℝ) :
  (x - 2 * y)^2 - (x + 2 * y + 3) * (x + 2 * y - 3) = -8 * x * y + 9 :=
by sorry

end problem1_problem2_l1285_128524


namespace jack_sugar_amount_l1285_128511

-- Definitions of initial conditions
def initial_amount : ℕ := 65
def used_amount : ℕ := 18
def bought_amount : ℕ := 50

-- Theorem statement
theorem jack_sugar_amount : initial_amount - used_amount + bought_amount = 97 :=
by
  -- Proof goes here
  sorry

end jack_sugar_amount_l1285_128511


namespace tangent_at_point_l1285_128530

theorem tangent_at_point (a b : ℝ) :
  (∀ x : ℝ, (x^3 - x^2 - a * x + b) = 2 * x + 1) →
  (a + b = -1) :=
by
  intro tangent_condition
  sorry

end tangent_at_point_l1285_128530


namespace problem_statement_l1285_128586

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a₁ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d

theorem problem_statement :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
  a 2 = 7 ∧
  a 4 + a 6 = 26 ∧
  (∀ n : ℕ, a (n + 1) = 2 * n + 1) ∧
  ∃ S : ℕ → ℤ, (S n = n^2 + 2 * n) ∧
  ∃ b : ℕ → ℚ, (∀ n : ℕ, b n = 1 / (a n ^ 2 - 1)) ∧
  ∃ T : ℕ → ℚ, (T n = (n / 4) * (1 / (n + 1))) :=
sorry

end problem_statement_l1285_128586


namespace smallest_k_exists_l1285_128587

theorem smallest_k_exists : ∃ (k : ℕ) (n : ℕ), k = 53 ∧ k^2 + 49 = 180 * n :=
sorry

end smallest_k_exists_l1285_128587


namespace jackson_holidays_l1285_128541

theorem jackson_holidays (holidays_per_month : ℕ) (months_in_year : ℕ) (holidays_per_year : ℕ) : 
  holidays_per_month = 3 → months_in_year = 12 → holidays_per_year = holidays_per_month * months_in_year → holidays_per_year = 36 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jackson_holidays_l1285_128541


namespace elizabeth_haircut_l1285_128572

theorem elizabeth_haircut (t s f : ℝ) (ht : t = 0.88) (hs : s = 0.5) : f = t - s := by
  sorry

end elizabeth_haircut_l1285_128572


namespace Frank_read_books_l1285_128597

noncomputable def books_read (total_days : ℕ) (days_per_book : ℕ) : ℕ :=
total_days / days_per_book

theorem Frank_read_books : books_read 492 12 = 41 := by
  sorry

end Frank_read_books_l1285_128597


namespace yard_length_eq_250_l1285_128544

noncomputable def number_of_trees : ℕ := 26
noncomputable def distance_between_trees : ℕ := 10
noncomputable def number_of_gaps := number_of_trees - 1
noncomputable def length_of_yard := number_of_gaps * distance_between_trees

theorem yard_length_eq_250 : 
  length_of_yard = 250 := 
sorry

end yard_length_eq_250_l1285_128544


namespace line_through_parabola_no_intersection_l1285_128517

-- Definitions of the conditions
def parabola (x : ℝ) : ℝ := x^2 
def point_Q := (10, 5)

-- The main theorem statement
theorem line_through_parabola_no_intersection :
  ∃ r s : ℝ, (∀ (m : ℝ), (r < m ∧ m < s) ↔ ¬ ∃ x : ℝ, parabola x = m * (x - 10) + 5) ∧ r + s = 40 :=
sorry

end line_through_parabola_no_intersection_l1285_128517


namespace factorization_of_polynomial_l1285_128543

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l1285_128543


namespace greatest_divisor_of_product_of_four_consecutive_integers_l1285_128513

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, ∃ k : Nat, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l1285_128513


namespace num_trucks_l1285_128585

variables (T : ℕ) (num_cars : ℕ := 13) (total_wheels : ℕ := 100) (wheels_per_vehicle : ℕ := 4)

theorem num_trucks :
  (num_cars * wheels_per_vehicle + T * wheels_per_vehicle = total_wheels) -> T = 12 :=
by
  intro h
  -- skipping the proof implementation
  sorry

end num_trucks_l1285_128585


namespace arithmetic_sequence_S11_l1285_128558

theorem arithmetic_sequence_S11 (a1 d : ℝ) 
  (h1 : a1 + d + a1 + 3 * d + 3 * (a1 + 6 * d) + a1 + 8 * d = 24) : 
  let a2 := a1 + d
  let a4 := a1 + 3 * d
  let a7 := a1 + 6 * d
  let a9 := a1 + 8 * d
  let S11 := 11 * (a1 + 5 * d)
  S11 = 44 :=
by
  sorry

end arithmetic_sequence_S11_l1285_128558


namespace focus_of_parabola_l1285_128521

theorem focus_of_parabola : 
  ∃(h k : ℚ), ((∀ x : ℚ, -2 * x^2 - 6 * x + 1 = -2 * (x + 3 / 2)^2 + 11 / 2) ∧ 
  (∃ a : ℚ, (a = -2 / 8) ∧ (h = -3/2) ∧ (k = 11/2 + a)) ∧ 
  (h, k) = (-3/2, 43 / 8)) :=
sorry

end focus_of_parabola_l1285_128521


namespace describe_cylinder_l1285_128592

noncomputable def cylinder_geometric_shape (c : ℝ) (r θ z : ℝ) : Prop :=
  r = c

theorem describe_cylinder (c : ℝ) (hc : 0 < c) :
  ∀ r θ z : ℝ, cylinder_geometric_shape c r θ z ↔ (r = c) :=
by
  sorry

end describe_cylinder_l1285_128592


namespace solution_set_l1285_128570

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the differentiable function f

axiom differentiable_f : Differentiable ℝ f
axiom condition_f : ∀ x, f x > 0 ∧ x * (deriv (deriv (deriv f))) x > 0

theorem solution_set :
  {x : ℝ | 1 ≤ x ∧ x < 2} =
    {x : ℝ | f (sqrt (x + 1)) > sqrt (x - 1) * f (sqrt (x ^ 2 - 1))} :=
sorry

end solution_set_l1285_128570


namespace age_difference_l1285_128568

theorem age_difference {A B C : ℕ} (h : A + B = B + C + 15) : A - C = 15 := 
by 
  sorry

end age_difference_l1285_128568


namespace no_distinct_natural_numbers_exist_l1285_128535

theorem no_distinct_natural_numbers_exist 
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ (a + 1 / a = (1 / 2) * (b + 1 / b + c + 1 / c)) :=
sorry

end no_distinct_natural_numbers_exist_l1285_128535


namespace correct_statement_l1285_128533

def degree (term : String) : ℕ :=
  if term = "1/2πx^2" then 2
  else if term = "-4x^2y" then 3
  else 0

def coefficient (term : String) : ℤ :=
  if term = "-4x^2y" then -4
  else if term = "3(x+y)" then 3
  else 0

def is_monomial (term : String) : Bool :=
  if term = "8" then true
  else false

theorem correct_statement : 
  (degree "1/2πx^2" ≠ 3) ∧ 
  (coefficient "-4x^2y" ≠ 4) ∧ 
  (is_monomial "8" = true) ∧ 
  (coefficient "3(x+y)" ≠ 3) := 
by
  sorry

end correct_statement_l1285_128533


namespace exponential_comparison_l1285_128594

theorem exponential_comparison (x y a b : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (hb : a < b) (hb' : b < 1) : 
  a^x < b^y :=
sorry

end exponential_comparison_l1285_128594


namespace max_area_of_triangle_l1285_128595

-- Defining the side lengths and constraints
def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main statement of the area maximization problem
theorem max_area_of_triangle (x : ℝ) (h1 : 2 < x) (h2 : x < 6) :
  triangle_sides 6 x (2 * x) →
  ∃ (S : ℝ), S = 12 :=
by
  sorry

end max_area_of_triangle_l1285_128595


namespace integer_count_n_l1285_128589

theorem integer_count_n (n : ℤ) (H1 : n % 3 = 0) (H2 : 3 * n ≥ 1) (H3 : 3 * n ≤ 1000) : 
  ∃ k : ℕ, k = 111 := by
  sorry

end integer_count_n_l1285_128589


namespace remainder_of_31_pow_31_plus_31_div_32_l1285_128525

theorem remainder_of_31_pow_31_plus_31_div_32 :
  (31^31 + 31) % 32 = 30 := 
by 
  trivial -- Replace with actual proof

end remainder_of_31_pow_31_plus_31_div_32_l1285_128525


namespace product_divisible_by_10_probability_l1285_128557

noncomputable def probability_divisible_by_10 (n : ℕ) (h: n > 1) : ℝ :=
  1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ))

theorem product_divisible_by_10_probability (n : ℕ) (h: n > 1) :
  probability_divisible_by_10 n h = 1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ)) :=
by
  -- The proof is omitted
  sorry

end product_divisible_by_10_probability_l1285_128557


namespace total_sheep_flock_l1285_128573

-- Definitions and conditions based on the problem description
def crossing_rate : ℕ := 3 -- Sheep per minute
def sleep_duration : ℕ := 90 -- Duration of sleep in minutes
def sheep_counted_before_sleep : ℕ := 42 -- Sheep counted before falling asleep

-- Total sheep that crossed while Nicholas was asleep
def sheep_during_sleep := crossing_rate * sleep_duration 

-- Total sheep that crossed when Nicholas woke up
def total_sheep_after_sleep := sheep_counted_before_sleep + sheep_during_sleep

-- Prove the total number of sheep in the flock
theorem total_sheep_flock : (2 * total_sheep_after_sleep) = 624 :=
by
  sorry

end total_sheep_flock_l1285_128573


namespace jana_walking_distance_l1285_128577

theorem jana_walking_distance (t_walk_mile : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  t_walk_mile = 24 → speed = 1 / t_walk_mile → time = 36 → distance = speed * time → distance = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end jana_walking_distance_l1285_128577


namespace tissue_actual_diameter_l1285_128560

theorem tissue_actual_diameter (magnification_factor : ℝ) (magnified_diameter : ℝ) 
(h1 : magnification_factor = 1000)
(h2 : magnified_diameter = 0.3) : 
  magnified_diameter / magnification_factor = 0.0003 :=
by sorry

end tissue_actual_diameter_l1285_128560


namespace difference_in_girls_and_boys_l1285_128526

theorem difference_in_girls_and_boys (x : ℕ) (h1 : 3 + 4 = 7) (h2 : 7 * x = 49) : 4 * x - 3 * x = 7 := by
  sorry

end difference_in_girls_and_boys_l1285_128526


namespace right_triangle_and_mod_inverse_l1285_128575

theorem right_triangle_and_mod_inverse (a b c m : ℕ) (h1 : a = 48) (h2 : b = 55) (h3 : c = 73) (h4 : m = 4273) 
  (h5 : a^2 + b^2 = c^2) : ∃ x : ℕ, (480 * x) % m = 1 ∧ x = 1643 :=
by
  sorry

end right_triangle_and_mod_inverse_l1285_128575


namespace fewest_four_dollar_frisbees_l1285_128512

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 60) (h2 : 3 * x + 4 * y = 200) : y = 20 :=
by 
  sorry  

end fewest_four_dollar_frisbees_l1285_128512


namespace sector_to_cone_ratio_l1285_128503

noncomputable def sector_angle : ℝ := 135
noncomputable def sector_area (S1 : ℝ) : ℝ := S1
noncomputable def cone_surface_area (S2 : ℝ) : ℝ := S2

theorem sector_to_cone_ratio (S1 S2 : ℝ) :
  sector_area S1 = (3 / 8) * (π * 1^2) →
  cone_surface_area S2 = (3 / 8) * (π * 1^2) + (9 / 64 * π) →
  (S1 / S2) = (8 / 11) :=
by
  intros h1 h2
  sorry

end sector_to_cone_ratio_l1285_128503


namespace arithmetic_sequence_properties_sum_of_sequence_b_n_l1285_128567

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h₁ : a 2 = 3) 
  (h₂ : S 5 + a 3 = 30) 
  (h₃ : ∀ n, S n = (n * (a 1 + (n-1) * ((a 2) - (a 1)))) / 2 
                     ∧ a n = a 1 + (n-1) * ((a 2) - (a 1))) : 
  (∀ n, a n = 2 * n - 1 ∧ S n = n^2) := 
sorry

theorem sum_of_sequence_b_n (b : ℕ → ℝ) 
  (T : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h₁ : ∀ n, b n = (a (n+1)) / (S n * S (n+1))) 
  (h₂ : ∀ n, a n = 2 * n - 1 ∧ S n = n^2) : 
  (∀ n, T n = (1 - 1 / (n+1)^2)) := 
sorry

end arithmetic_sequence_properties_sum_of_sequence_b_n_l1285_128567


namespace other_endpoint_coordinates_sum_l1285_128569

noncomputable def other_endpoint_sum (x1 y1 x2 y2 xm ym : ℝ) : ℝ :=
  let x := 2 * xm - x1
  let y := 2 * ym - y1
  x + y

theorem other_endpoint_coordinates_sum :
  (other_endpoint_sum 6 (-2) 0 12 3 5) = 12 := by
  sorry

end other_endpoint_coordinates_sum_l1285_128569


namespace primes_eq_2_3_7_l1285_128550

theorem primes_eq_2_3_7 (p : ℕ) (hp : Nat.Prime p) :
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 :=
by
  sorry

end primes_eq_2_3_7_l1285_128550


namespace complex_multiplication_l1285_128598

theorem complex_multiplication :
  ∀ (i : ℂ), i * i = -1 → i * (1 + i) = -1 + i :=
by
  intros i hi
  sorry

end complex_multiplication_l1285_128598


namespace digits_sum_not_2001_l1285_128554

theorem digits_sum_not_2001 (a : ℕ) (n m : ℕ) 
  (h1 : 10^(n-1) ≤ a ∧ a < 10^n)
  (h2 : 3 * n - 2 ≤ m ∧ m < 3 * n + 1)
  : m + n ≠ 2001 := 
sorry

end digits_sum_not_2001_l1285_128554


namespace number_of_quarters_l1285_128542
-- Definitions of the coin values
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

-- Number of each type of coin used in the proof
variable (pennies nickels dimes quarters half_dollars : ℕ)

-- Conditions from step (a)
axiom one_penny : pennies > 0
axiom one_nickel : nickels > 0
axiom one_dime : dimes > 0
axiom one_quarter : quarters > 0
axiom one_half_dollar : half_dollars > 0
axiom total_coins : pennies + nickels + dimes + quarters + half_dollars = 11
axiom total_value : pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value = 163

-- The conclusion we want to prove
theorem number_of_quarters : quarters = 1 := 
sorry

end number_of_quarters_l1285_128542


namespace sixty_fifth_term_is_sixteen_l1285_128599

def apply_rule (n : ℕ) : ℕ :=
  if n <= 12 then
    7 * n
  else if n % 2 = 0 then
    n - 7
  else
    n / 3

def sequence_term (a_0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate apply_rule n a_0

theorem sixty_fifth_term_is_sixteen : sequence_term 65 64 = 16 := by
  sorry

end sixty_fifth_term_is_sixteen_l1285_128599


namespace exactly_one_first_class_probability_at_least_one_second_class_probability_l1285_128566

-- Definitions based on the problem statement:
def total_pens : ℕ := 6
def first_class_pens : ℕ := 4
def second_class_pens : ℕ := 2

def total_draws : ℕ := 2

-- Event for drawing exactly one first-class quality pen
def probability_one_first_class := ((first_class_pens.choose 1 * second_class_pens.choose 1) /
                                    (total_pens.choose total_draws) : ℚ)

-- Event for drawing at least one second-class quality pen
def probability_at_least_one_second_class := (1 - (first_class_pens.choose total_draws /
                                                   total_pens.choose total_draws) : ℚ)

-- Statements to prove the probabilities
theorem exactly_one_first_class_probability :
  probability_one_first_class = 8 / 15 :=
sorry

theorem at_least_one_second_class_probability :
  probability_at_least_one_second_class = 3 / 5 :=
sorry

end exactly_one_first_class_probability_at_least_one_second_class_probability_l1285_128566


namespace smallest_whole_number_larger_than_any_triangle_perimeter_l1285_128596

def is_valid_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem smallest_whole_number_larger_than_any_triangle_perimeter : 
  ∀ (s : ℕ), 16 < s ∧ s < 30 → is_valid_triangle 7 23 s → 
    60 = (Nat.succ (7 + 23 + s - 1)) := 
by 
  sorry

end smallest_whole_number_larger_than_any_triangle_perimeter_l1285_128596


namespace divisor_is_four_l1285_128590

theorem divisor_is_four (n d k l : ℤ) (hn : n % d = 3) (h2n : (2 * n) % d = 2) (hd : d > 3) : d = 4 :=
by
  sorry

end divisor_is_four_l1285_128590


namespace logical_contradiction_l1285_128519

-- Definitions based on the conditions
def all_destroying (x : Type) : Prop := ∀ y : Type, y ≠ x → y → false
def indestructible (x : Type) : Prop := ∀ y : Type, y = x → y → false

theorem logical_contradiction (x : Type) :
  (all_destroying x ∧ indestructible x) → false :=
by
  sorry

end logical_contradiction_l1285_128519


namespace range_S13_over_a14_l1285_128538

lemma a_n_is_arithmetic_progression (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2) :
  ∀ n, a (n + 1) = a n + 1 := 
sorry

theorem range_S13_over_a14 (a S : ℕ → ℕ) (h1 : ∀ n, a n > 1)
  (h2 : ∀ n, S (n + 1) - S n - (1 / 2) * ((a (n + 1)) ^ 2 - (a n) ^ 2) = 1 / 2)
  (h3 : a 1 > 4) :
  130 / 17 < (S 13 / a 14) ∧ (S 13 / a 14) < 13 := 
sorry

end range_S13_over_a14_l1285_128538


namespace min_value_expr_l1285_128583

theorem min_value_expr (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + ((y / x) - 1)^2 + ((z / y) - 1)^2 + ((5 / z) - 1)^2 = 9 :=
sorry

end min_value_expr_l1285_128583


namespace birds_on_the_fence_l1285_128562

theorem birds_on_the_fence (x : ℕ) : 10 + 2 * x = 50 → x = 20 := by
  sorry

end birds_on_the_fence_l1285_128562


namespace solve_equation_l1285_128516

theorem solve_equation : ∀ x : ℝ, (x + 1 - 2 * (x - 1) = 1 - 3 * x) → x = 0 := 
by
  intros x h
  sorry

end solve_equation_l1285_128516


namespace parabola_focus_distance_l1285_128518

noncomputable def PF (x₁ : ℝ) : ℝ := x₁ + 1
noncomputable def QF (x₂ : ℝ) : ℝ := x₂ + 1

theorem parabola_focus_distance 
  (x₁ x₂ : ℝ) (h₁ : x₂ = 3 * x₁ + 2) : 
  QF x₂ / PF x₁ = 3 :=
by
  sorry

end parabola_focus_distance_l1285_128518


namespace mul_99_101_equals_9999_l1285_128502

theorem mul_99_101_equals_9999 : 99 * 101 = 9999 := by
  sorry

end mul_99_101_equals_9999_l1285_128502


namespace tangent_lines_count_l1285_128534

def f (x : ℝ) : ℝ := x^3

theorem tangent_lines_count :
  (∃ x : ℝ, deriv f x = 3) ∧ 
  (∃ y : ℝ, deriv f y = 3 ∧ y ≠ x) := 
by
  -- Since f(x) = x^3, its derivative is f'(x) = 3x^2
  -- We need to solve 3x^2 = 3
  -- Therefore, x^2 = 1 and x = ±1
  -- Thus, there are two tangent lines
  sorry

end tangent_lines_count_l1285_128534


namespace how_many_more_yellow_peaches_l1285_128553

-- Definitions
def red_peaches : ℕ := 7
def yellow_peaches_initial : ℕ := 15
def green_peaches : ℕ := 8
def combined_red_green_peaches := red_peaches + green_peaches
def required_yellow_peaches := 2 * combined_red_green_peaches
def additional_yellow_peaches_needed := required_yellow_peaches - yellow_peaches_initial

-- Theorem statement
theorem how_many_more_yellow_peaches :
  additional_yellow_peaches_needed = 15 :=
by
  sorry

end how_many_more_yellow_peaches_l1285_128553


namespace divisibility_by_1956_l1285_128507

theorem divisibility_by_1956 (n : ℕ) (hn : n % 2 = 1) : 
  1956 ∣ (24 * 80^n + 1992 * 83^(n-1)) :=
by
  sorry

end divisibility_by_1956_l1285_128507


namespace fraction_ordering_l1285_128527

noncomputable def t1 : ℝ := (100^100 + 1) / (100^90 + 1)
noncomputable def t2 : ℝ := (100^99 + 1) / (100^89 + 1)
noncomputable def t3 : ℝ := (100^101 + 1) / (100^91 + 1)
noncomputable def t4 : ℝ := (101^101 + 1) / (101^91 + 1)
noncomputable def t5 : ℝ := (101^100 + 1) / (101^90 + 1)
noncomputable def t6 : ℝ := (99^99 + 1) / (99^89 + 1)
noncomputable def t7 : ℝ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering : t6 < t7 ∧ t7 < t2 ∧ t2 < t1 ∧ t1 < t3 ∧ t3 < t5 ∧ t5 < t4 := by
  sorry

end fraction_ordering_l1285_128527


namespace population_increase_rate_correct_l1285_128531

variable (P0 P1 : ℕ)
variable (r : ℚ)

-- Given conditions
def initial_population := P0 = 200
def population_after_one_year := P1 = 220

-- Proof problem statement
theorem population_increase_rate_correct :
  initial_population P0 →
  population_after_one_year P1 →
  r = (P1 - P0 : ℚ) / P0 * 100 →
  r = 10 :=
by
  sorry

end population_increase_rate_correct_l1285_128531


namespace complex_magnitude_equality_l1285_128579

open Complex Real

theorem complex_magnitude_equality :
  abs ((Complex.mk (5 * sqrt 2) (-5)) * (Complex.mk (2 * sqrt 3) 6)) = 60 :=
by
  sorry

end complex_magnitude_equality_l1285_128579


namespace area_of_large_rectangle_ABCD_l1285_128509

-- Definitions for conditions and given data
def shaded_rectangle_area : ℕ := 2
def area_of_rectangle_ABCD (a b c : ℕ) : ℕ := a + b + c

-- The theorem to prove
theorem area_of_large_rectangle_ABCD
  (a b c : ℕ) 
  (h1 : shaded_rectangle_area = a)
  (h2 : shaded_rectangle_area = b)
  (h3 : a + b + c = 8) : 
  area_of_rectangle_ABCD a b c = 8 :=
by
  sorry

end area_of_large_rectangle_ABCD_l1285_128509


namespace find_a_b_value_l1285_128584

-- Define the variables
variables {a b : ℤ}

-- Define the conditions for the monomials to be like terms
def exponents_match_x (a : ℤ) : Prop := a + 2 = 1
def exponents_match_y (b : ℤ) : Prop := b + 1 = 3

-- Main statement
theorem find_a_b_value (ha : exponents_match_x a) (hb : exponents_match_y b) : a + b = 1 :=
by
  sorry

end find_a_b_value_l1285_128584


namespace digging_project_length_l1285_128536

theorem digging_project_length (Length_2 : ℝ) : 
  (100 * 25 * 30) = (75 * Length_2 * 50) → 
  Length_2 = 20 :=
by
  sorry

end digging_project_length_l1285_128536


namespace f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l1285_128555

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 - 1 else -x + 1

-- Prove f[f(-1)] = -1
theorem f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := sorry

-- Prove that if f(x) = -1, then x = 0 or x = 2
theorem f_x_eq_neg1_iff_x_eq_0_or_2 (x : ℝ) : f x = -1 ↔ x = 0 ∨ x = 2 := sorry

end f_of_f_neg1_eq_neg1_f_x_eq_neg1_iff_x_eq_0_or_2_l1285_128555


namespace john_initial_running_time_l1285_128547

theorem john_initial_running_time (H : ℝ) (hH1 : 1.75 * H = 168 / 12)
: H = 8 :=
sorry

end john_initial_running_time_l1285_128547


namespace train_length_l1285_128510

-- Definitions and conditions
variable (L : ℕ)
def condition1 (L : ℕ) : Prop := L + 100 = 15 * (L + 100) / 15
def condition2 (L : ℕ) : Prop := L + 250 = 20 * (L + 250) / 20

-- Theorem statement
theorem train_length (h1 : condition1 L) (h2 : condition2 L) : L = 350 := 
by 
  sorry

end train_length_l1285_128510


namespace sin_cos_alpha_beta_l1285_128576

theorem sin_cos_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α = Real.cos β) 
  (h2 : Real.cos α = Real.sin (2 * β)) :
  Real.sin β ^ 2 + Real.cos α ^ 2 = 3 / 2 := 
by
  sorry

end sin_cos_alpha_beta_l1285_128576


namespace parallelogram_area_twice_quadrilateral_l1285_128563

theorem parallelogram_area_twice_quadrilateral (a b : ℝ) (α : ℝ) (hα : 0 < α ∧ α < π) :
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  parallelogram_area = 2 * quadrilateral_area :=
by
  let quadrilateral_area := (1 / 2) * a * b * Real.sin α
  let parallelogram_area := a * b * Real.sin α
  sorry

end parallelogram_area_twice_quadrilateral_l1285_128563


namespace johns_profit_l1285_128508

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l1285_128508


namespace gift_wrapping_combinations_l1285_128532

theorem gift_wrapping_combinations :
    (10 * 3 * 4 * 5 = 600) :=
by
    sorry

end gift_wrapping_combinations_l1285_128532


namespace min_N_of_block_viewed_l1285_128556

theorem min_N_of_block_viewed (x y z N : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_factor : (x - 1) * (y - 1) * (z - 1) = 231) : 
  N = x * y * z ∧ N = 384 :=
by {
  sorry 
}

end min_N_of_block_viewed_l1285_128556


namespace p2_div_q2_eq_4_l1285_128559

theorem p2_div_q2_eq_4 
  (p q : ℝ → ℝ)
  (h1 : ∀ x, p x = 12 * x)
  (h2 : ∀ x, q x = (x + 4) * (x - 1))
  (h3 : p 0 = 0)
  (h4 : p (-1) / q (-1) = -2) :
  (p 2 / q 2 = 4) :=
by {
  sorry
}

end p2_div_q2_eq_4_l1285_128559


namespace initial_men_count_l1285_128565

theorem initial_men_count 
  (M : ℕ)
  (h1 : 8 * M * 30 = (M + 77) * 6 * 50) :
  M = 63 :=
by
  sorry

end initial_men_count_l1285_128565


namespace find_factor_l1285_128528

variable (x : ℕ) (f : ℕ)

def original_number := x = 20
def resultant := f * (2 * x + 5) = 135

theorem find_factor (h1 : original_number x) (h2 : resultant x f) : f = 3 := by
  sorry

end find_factor_l1285_128528


namespace find_h2_l1285_128580

noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^15 - 1)

theorem find_h2 : h 2 = 2 :=
by 
  sorry

end find_h2_l1285_128580
