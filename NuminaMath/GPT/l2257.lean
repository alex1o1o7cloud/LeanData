import Mathlib

namespace tan_alpha_l2257_225784

theorem tan_alpha (α : ℝ) (hα1 : α > π / 2) (hα2 : α < π) (h_sin : Real.sin α = 4 / 5) : Real.tan α = - (4 / 3) :=
by 
  sorry

end tan_alpha_l2257_225784


namespace no_sum_14_l2257_225794

theorem no_sum_14 (x y : ℤ) (h : x * y + 4 = 40) : x + y ≠ 14 :=
by sorry

end no_sum_14_l2257_225794


namespace intersections_line_segment_l2257_225701

def intersects_count (a b : ℕ) (x y : ℕ) : ℕ :=
  let steps := gcd x y
  2 * (steps + 1)

theorem intersections_line_segment (x y : ℕ) (h_x : x = 501) (h_y : y = 201) :
  intersects_count 1 1 x y = 336 := by
  sorry

end intersections_line_segment_l2257_225701


namespace max_expr_under_condition_l2257_225783

-- Define the conditions and variables
variable {x : ℝ}

-- State the theorem about the maximum value of the given expression under the given condition
theorem max_expr_under_condition (h : x < -3) : 
  ∃ M, M = -2 * Real.sqrt 2 - 3 ∧ ∀ y, y < -3 → y + 2 / (y + 3) ≤ M :=
sorry

end max_expr_under_condition_l2257_225783


namespace compute_expression_value_l2257_225739

-- Define the expression
def expression : ℤ := 1013^2 - 1009^2 - 1011^2 + 997^2

-- State the theorem with the required conditions and conclusions
theorem compute_expression_value : expression = -19924 := 
by 
  -- The proof steps would go here.
  sorry

end compute_expression_value_l2257_225739


namespace abs_h_eq_2_l2257_225744

-- Definitions based on the given conditions
def sum_of_squares_of_roots (h : ℝ) : Prop :=
  let a := 1
  let b := -4 * h
  let c := -8
  let sum_of_roots := -b / a
  let prod_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2 * prod_of_roots
  sum_of_squares = 80

-- Theorem to prove the absolute value of h is 2
theorem abs_h_eq_2 (h : ℝ) (h_condition : sum_of_squares_of_roots h) : |h| = 2 :=
by
  sorry

end abs_h_eq_2_l2257_225744


namespace system_equations_sum_14_l2257_225772

theorem system_equations_sum_14 (a b c d : ℝ) 
  (h1 : a + c = 4) 
  (h2 : a * d + b * c = 5) 
  (h3 : a * c + b + d = 8) 
  (h4 : b * d = 1) :
  a + b + c + d = 7 ∨ a + b + c + d = 7 → (a + b + c + d) * 2 = 14 := 
by {
  sorry
}

end system_equations_sum_14_l2257_225772


namespace remainder_division_l2257_225777

theorem remainder_division :
  ∃ N R1 Q2, N = 44 * 432 + R1 ∧ N = 30 * Q2 + 18 ∧ R1 < 44 ∧ 18 = R1 :=
by
  sorry

end remainder_division_l2257_225777


namespace length_of_train_l2257_225781

variable (L : ℕ)

def speed_tree (L : ℕ) : ℚ := L / 120

def speed_platform (L : ℕ) : ℚ := (L + 500) / 160

theorem length_of_train
    (h1 : speed_tree L = speed_platform L)
    : L = 1500 :=
sorry

end length_of_train_l2257_225781


namespace least_number_1056_div_26_l2257_225721

/-- Define the given values and the divisibility condition -/
def least_number_to_add (n : ℕ) (d : ℕ) : ℕ :=
  let remainder := n % d
  d - remainder

/-- State the theorem to prove that the least number to add to 1056 to make it divisible by 26 is 10. -/
theorem least_number_1056_div_26 : least_number_to_add 1056 26 = 10 :=
by
  sorry -- Proof is omitted as per the instruction

end least_number_1056_div_26_l2257_225721


namespace medal_ratio_l2257_225734

theorem medal_ratio (total_medals : ℕ) (track_medals : ℕ) (badminton_medals : ℕ) (swimming_medals : ℕ) 
  (h1 : total_medals = 20) 
  (h2 : track_medals = 5) 
  (h3 : badminton_medals = 5) 
  (h4 : swimming_medals = total_medals - track_medals - badminton_medals) : 
  swimming_medals / track_medals = 2 := 
by 
  sorry

end medal_ratio_l2257_225734


namespace number_of_zeros_of_f_l2257_225724

def f (x : ℝ) : ℝ := 2 * x - 3 * x

theorem number_of_zeros_of_f :
  ∃ (n : ℕ), n = 2 ∧ (∀ x, f x = 0 → x ∈ {x | f x = 0}) :=
by {
  sorry
}

end number_of_zeros_of_f_l2257_225724


namespace harkamal_purchase_mangoes_l2257_225751

variable (m : ℕ)

def cost_of_grapes (cost_per_kg grapes_weight : ℕ) : ℕ := cost_per_kg * grapes_weight
def cost_of_mangoes (cost_per_kg mangoes_weight : ℕ) : ℕ := cost_per_kg * mangoes_weight

theorem harkamal_purchase_mangoes :
  (cost_of_grapes 70 10 + cost_of_mangoes 55 m = 1195) → m = 9 :=
by
  sorry

end harkamal_purchase_mangoes_l2257_225751


namespace warehouse_box_storage_l2257_225706

theorem warehouse_box_storage (S : ℝ) (h1 : (3 - 1/4) * S = 55000) : (1/4) * S = 5000 :=
by
  sorry

end warehouse_box_storage_l2257_225706


namespace average_distance_per_day_l2257_225754

def distance_Monday : ℝ := 4.2
def distance_Tuesday : ℝ := 3.8
def distance_Wednesday : ℝ := 3.6
def distance_Thursday : ℝ := 4.4

def total_distance : ℝ := distance_Monday + distance_Tuesday + distance_Wednesday + distance_Thursday

def number_of_days : ℕ := 4

theorem average_distance_per_day : total_distance / number_of_days = 4 := by
  sorry

end average_distance_per_day_l2257_225754


namespace lcm_gcd_product_l2257_225703

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  rw [ha, hb]
  -- Replace with Nat library functions and calculate
  sorry

end lcm_gcd_product_l2257_225703


namespace xy_inequality_l2257_225763

theorem xy_inequality (x y θ : ℝ) 
    (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
    x^2 + y^2 ≥ 3/4 :=
sorry

end xy_inequality_l2257_225763


namespace jason_initial_cards_l2257_225791

theorem jason_initial_cards (cards_sold : Nat) (cards_after_selling : Nat) (initial_cards : Nat) 
  (h1 : cards_sold = 224) 
  (h2 : cards_after_selling = 452) 
  (h3 : initial_cards = cards_after_selling + cards_sold) : 
  initial_cards = 676 := 
sorry

end jason_initial_cards_l2257_225791


namespace ratio_of_cube_sides_l2257_225769

theorem ratio_of_cube_sides 
  (a b : ℝ) 
  (h : (6 * a^2) / (6 * b^2) = 49) :
  a / b = 7 :=
by
  sorry

end ratio_of_cube_sides_l2257_225769


namespace units_digit_of_power_l2257_225742

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l2257_225742


namespace apples_per_hour_l2257_225759

def total_apples : ℕ := 15
def hours : ℕ := 3

theorem apples_per_hour : total_apples / hours = 5 := by
  sorry

end apples_per_hour_l2257_225759


namespace area_of_park_l2257_225732

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park_l2257_225732


namespace scientific_notation_correct_l2257_225733

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct_l2257_225733


namespace hectors_sibling_product_l2257_225761

theorem hectors_sibling_product (sisters : Nat) (brothers : Nat) (helen : Nat -> Prop): 
  (helen 4) → (helen 7) → (helen 5) → (helen 6) →
  (sisters + 1 = 5) → (brothers + 1 = 7) → ((sisters * brothers) = 30) :=
by
  sorry

end hectors_sibling_product_l2257_225761


namespace Loisa_saves_70_l2257_225768

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l2257_225768


namespace Megan_deleted_files_l2257_225719

theorem Megan_deleted_files (initial_files folders files_per_folder deleted_files : ℕ) 
    (h1 : initial_files = 93) 
    (h2 : folders = 9)
    (h3 : files_per_folder = 8) 
    (h4 : deleted_files = initial_files - folders * files_per_folder) : 
  deleted_files = 21 :=
by
  sorry

end Megan_deleted_files_l2257_225719


namespace line_passes_through_fixed_point_l2257_225731

theorem line_passes_through_fixed_point (k : ℝ) : ∀ x y : ℝ, (y - 1 = k * (x + 2)) → (x = -2 ∧ y = 1) :=
by
  intro x y h
  sorry

end line_passes_through_fixed_point_l2257_225731


namespace simplify_fraction_l2257_225749

open Real

theorem simplify_fraction (x : ℝ) : (3 + 2 * sin x + 2 * cos x) / (3 + 2 * sin x - 2 * cos x) = 3 / 5 + (2 / 5) * cos x :=
by
  sorry

end simplify_fraction_l2257_225749


namespace students_not_yes_for_either_subject_l2257_225786

variable (total_students yes_m no_m unsure_m yes_r no_r unsure_r yes_only_m : ℕ)

theorem students_not_yes_for_either_subject :
  total_students = 800 →
  yes_m = 500 →
  no_m = 200 →
  unsure_m = 100 →
  yes_r = 400 →
  no_r = 100 →
  unsure_r = 300 →
  yes_only_m = 150 →
  ∃ students_not_yes, students_not_yes = total_students - (yes_only_m + (yes_m - yes_only_m) + (yes_r - (yes_m - yes_only_m))) ∧ students_not_yes = 400 :=
by
  intros ht yt1 nnm um ypr ynr ur yom
  sorry

end students_not_yes_for_either_subject_l2257_225786


namespace negation_of_proposition_l2257_225793

open Real

theorem negation_of_proposition (P : ∀ x : ℝ, sin x ≥ 1) :
  ∃ x : ℝ, sin x < 1 :=
sorry

end negation_of_proposition_l2257_225793


namespace independence_events_exactly_one_passing_l2257_225755

-- Part 1: Independence of Events

def event_A (die1 : ℕ) : Prop :=
  die1 % 2 = 1

def event_B (die1 die2 : ℕ) : Prop :=
  (die1 + die2) % 3 = 0

def P_event_A : ℚ :=
  1 / 2

def P_event_B : ℚ :=
  1 / 3

def P_event_AB : ℚ :=
  1 / 6

theorem independence_events : P_event_AB = P_event_A * P_event_B :=
by
  sorry

-- Part 2: Probability of Exactly One Passing the Assessment

def probability_of_hitting (p : ℝ) : ℝ :=
  1 - (1 - p)^2

def P_A_hitting : ℝ :=
  0.7

def P_B_hitting : ℝ :=
  0.6

def probability_one_passing : ℝ :=
  (probability_of_hitting P_A_hitting) * (1 - probability_of_hitting P_B_hitting) + (1 - probability_of_hitting P_A_hitting) * (probability_of_hitting P_B_hitting)

theorem exactly_one_passing : probability_one_passing = 0.2212 :=
by
  sorry

end independence_events_exactly_one_passing_l2257_225755


namespace division_result_l2257_225726

theorem division_result : (8900 / 6) / 4 = 370.8333 :=
by sorry

end division_result_l2257_225726


namespace g_minus_one_eq_zero_l2257_225756

def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

theorem g_minus_one_eq_zero (r : ℝ) : g (-1) r = 0 → r = 14 := by
  sorry

end g_minus_one_eq_zero_l2257_225756


namespace football_team_total_members_l2257_225728

-- Definitions from the problem conditions
def initialMembers : ℕ := 42
def newMembers : ℕ := 17

-- Mathematical equivalent proof problem
theorem football_team_total_members : initialMembers + newMembers = 59 := by
  sorry

end football_team_total_members_l2257_225728


namespace container_capacity_l2257_225766

theorem container_capacity (C : ℝ) (h1 : 0.30 * C + 36 = 0.75 * C) : C = 80 :=
by
  sorry

end container_capacity_l2257_225766


namespace line_through_intersections_l2257_225704

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → 2 * x - 3 * y = 0 := 
sorry

end line_through_intersections_l2257_225704


namespace range_of_m_l2257_225778

noncomputable def f (x m : ℝ) : ℝ := -x^2 + m * x

theorem range_of_m {m : ℝ} : (∀ x y : ℝ, x ≤ y → x ≤ 1 → y ≤ 1 → f x m ≤ f y m) ↔ 2 ≤ m := 
sorry

end range_of_m_l2257_225778


namespace min_value_l2257_225757

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ min_val, min_val = 5 + 2 * Real.sqrt 6 ∧ (∀ x, (x = 5 + 2 * Real.sqrt 6) → x ≥ min_val) :=
by
  sorry

end min_value_l2257_225757


namespace find_second_speed_l2257_225787

theorem find_second_speed (d t_b : ℝ) (v1 : ℝ) (t_m t_a : ℤ): 
  d = 13.5 ∧ v1 = 5 ∧ t_m = 12 ∧ t_a = 15 →
  (t_b = (d / v1) - (t_m / 60)) →
  (t2 = t_b - (t_a / 60)) →
  v = d / t2 →
  v = 6 :=
by
  sorry

end find_second_speed_l2257_225787


namespace bananas_bought_l2257_225746

theorem bananas_bought (O P B : Nat) (x : Nat) 
  (h1 : P - O = B)
  (h2 : O + P = 120)
  (h3 : P = 90)
  (h4 : 60 * x + 30 * (2 * x) = 24000) : 
  x = 200 := by
  sorry

end bananas_bought_l2257_225746


namespace EH_length_l2257_225736

structure Rectangle :=
(AB BC CD DA : ℝ)
(horiz: AB=CD)
(verti: BC=DA)
(diag_eq: (AB^2 + BC^2) = (CD^2 + DA^2))

structure Point :=
(x y : ℝ)

noncomputable def H_distance (E D : Point)
    (AB BC : ℝ) : ℝ :=
    (E.y - D.y) -- if we consider D at origin (0,0)

theorem EH_length
    (AB BC : ℝ)
    (H_dist : ℝ)
    (E : Point)
    (rectangle : Rectangle) :
    AB = 50 →
    BC = 60 →
    E.x^2 + BC^2 = 30^2 + 60^2 →
    E.y = 40 →
    H_dist = E.y - CD →
    H_dist = 7.08 :=
by
    sorry

end EH_length_l2257_225736


namespace total_distance_of_ship_l2257_225738

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end total_distance_of_ship_l2257_225738


namespace original_remainder_when_dividing_by_44_is_zero_l2257_225782

theorem original_remainder_when_dividing_by_44_is_zero 
  (N R : ℕ) 
  (Q : ℕ) 
  (h1 : N = 44 * 432 + R) 
  (h2 : N = 34 * Q + 2) 
  : R = 0 := 
sorry

end original_remainder_when_dividing_by_44_is_zero_l2257_225782


namespace k_valid_iff_l2257_225797

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l2257_225797


namespace mean_equality_l2257_225771

-- Define the mean calculation
def mean (a b c : ℕ) : ℚ := (a + b + c) / 3

-- The given conditions
theorem mean_equality (z : ℕ) (y : ℕ) (hz : z = 24) :
  mean 8 15 21 = mean 16 z y → y = 4 :=
by
  sorry

end mean_equality_l2257_225771


namespace quadratic_real_roots_m_range_l2257_225773

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l2257_225773


namespace find_initial_investment_l2257_225714

open Real

noncomputable def initial_investment (x : ℝ) (years : ℕ) (final_value : ℝ) : ℝ := 
  final_value / (3 ^ (years / (112 / x)))

theorem find_initial_investment :
  let x := 8
  let years := 28
  let final_value := 31500
  initial_investment x years final_value = 3500 := 
by 
  sorry

end find_initial_investment_l2257_225714


namespace feet_heads_difference_l2257_225707

theorem feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let heads := hens + goats + camels + keepers
  let feet := (2 * hens) + (4 * goats) + (4 * camels) + (2 * keepers)
  feet - heads = 193 :=
by
  sorry

end feet_heads_difference_l2257_225707


namespace binomial_square_solution_l2257_225715

variable (t u b : ℝ)

theorem binomial_square_solution (h1 : 2 * t * u = 12) (h2 : u^2 = 9) : b = t^2 → b = 4 :=
by
  sorry

end binomial_square_solution_l2257_225715


namespace solve_equation_x_squared_eq_16x_l2257_225799

theorem solve_equation_x_squared_eq_16x :
  ∀ x : ℝ, x^2 = 16 * x ↔ (x = 0 ∨ x = 16) :=
by 
  intro x
  -- Complete proof here
  sorry

end solve_equation_x_squared_eq_16x_l2257_225799


namespace functional_equation_solution_l2257_225798

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)) →
  (∀ y : ℝ, f y = 0) :=
by
  intro h
  sorry

end functional_equation_solution_l2257_225798


namespace contemporaries_probability_l2257_225767

theorem contemporaries_probability:
  (∀ (x y : ℝ),
    0 ≤ x ∧ x ≤ 400 ∧
    0 ≤ y ∧ y ≤ 400 ∧
    (x < y + 80) ∧ (y < x + 80)) →
    (∃ p : ℝ, p = 9 / 25) :=
by sorry

end contemporaries_probability_l2257_225767


namespace factor_1_factor_2_l2257_225712

theorem factor_1 {x : ℝ} : x^2 - 4*x + 3 = (x - 1) * (x - 3) :=
sorry

theorem factor_2 {x : ℝ} : 4*x^2 + 12*x - 7 = (2*x + 7) * (2*x - 1) :=
sorry

end factor_1_factor_2_l2257_225712


namespace problem_statement_l2257_225764

open Set

variable (U P Q : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5})

theorem problem_statement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end problem_statement_l2257_225764


namespace floor_neg_seven_fourths_l2257_225722

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l2257_225722


namespace proof_of_problem_l2257_225729

noncomputable def f : ℝ → ℝ := sorry  -- define f as a function in ℝ to ℝ

theorem proof_of_problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3) :
  f 2015 + f 2016 = -1 := 
sorry

end proof_of_problem_l2257_225729


namespace consecutive_nums_sum_as_product_l2257_225747

theorem consecutive_nums_sum_as_product {n : ℕ} (h : 100 < n) :
  ∃ (a b c : ℕ), (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (2 ≤ a) ∧ (2 ≤ b) ∧ (2 ≤ c) ∧ 
  ((n + (n+1) + (n+2) = a * b * c) ∨ ((n+1) + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end consecutive_nums_sum_as_product_l2257_225747


namespace Rick_received_amount_l2257_225700

theorem Rick_received_amount :
  let total_promised := 400
  let sally_owes := 35
  let amy_owes := 30
  let derek_owes := amy_owes / 2
  let carl_owes := 35
  let total_owed := sally_owes + amy_owes + derek_owes + carl_owes
  total_promised - total_owed = 285 :=
by
  sorry

end Rick_received_amount_l2257_225700


namespace field_area_l2257_225788

theorem field_area (L W : ℝ) (h1: L = 20) (h2 : 2 * W + L = 41) : L * W = 210 :=
by
  sorry

end field_area_l2257_225788


namespace percent_of_x_is_z_l2257_225752

def condition1 (z y : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (y x : ℝ) : Prop := y = 0.75 * x
def condition3 (w z : ℝ) : Prop := w = 0.60 * z^2
def condition4 (z w : ℝ) : Prop := z = 0.30 * w^(1/3)

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : condition1 z y) 
  (h2 : condition2 y x)
  (h3 : condition3 w z)
  (h4 : condition4 z w) : 
  z / x = 1.2 :=
sorry

end percent_of_x_is_z_l2257_225752


namespace monkeys_more_than_giraffes_l2257_225705

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l2257_225705


namespace ratio_a_d_l2257_225760

theorem ratio_a_d 
  (a b c d : ℕ) 
  (h1 : a / b = 1 / 4) 
  (h2 : b / c = 13 / 9) 
  (h3 : c / d = 5 / 13) : 
  a / d = 5 / 36 :=
sorry

end ratio_a_d_l2257_225760


namespace problem_statement_l2257_225741

variable {a x y : ℝ}

theorem problem_statement (hx : 0 < a) (ha : a < 1) (h : a^x < a^y) : x^3 > y^3 :=
sorry

end problem_statement_l2257_225741


namespace smallest_b_factors_l2257_225774

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end smallest_b_factors_l2257_225774


namespace math_problem_l2257_225790

variables (a b c d m : ℝ)

theorem math_problem 
  (h1 : a = -b)            -- condition 1: a and b are opposite numbers
  (h2 : c * d = 1)         -- condition 2: c and d are reciprocal numbers
  (h3 : |m| = 1) :         -- condition 3: absolute value of m is 1
  (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 :=
sorry

end math_problem_l2257_225790


namespace complement_in_U_l2257_225789

def A : Set ℝ := { x : ℝ | |x - 1| > 3 }
def U : Set ℝ := Set.univ

theorem complement_in_U :
  (U \ A) = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end complement_in_U_l2257_225789


namespace div_condition_nat_l2257_225711

theorem div_condition_nat (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 :=
by
  sorry

end div_condition_nat_l2257_225711


namespace expression_equivalence_l2257_225713

theorem expression_equivalence : (2 / 20) + (3 / 30) + (4 / 40) + (5 / 50) = 0.4 := by
  sorry

end expression_equivalence_l2257_225713


namespace focal_length_of_curve_l2257_225740

theorem focal_length_of_curve : 
  (∀ θ : ℝ, ∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) →
  ∃ f : ℝ, f = 2 * Real.sqrt 3 :=
by sorry

end focal_length_of_curve_l2257_225740


namespace disproof_of_Alitta_l2257_225723

-- Definition: A prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition: A number is odd
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- The value is a specific set of odd primes including 11
def contains (p : ℕ) : Prop :=
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11

-- Main statement: There exists an odd prime p in the given options such that p^2 - 2 is not a prime
theorem disproof_of_Alitta :
  ∃ p : ℕ, contains p ∧ is_prime p ∧ is_odd p ∧ ¬ is_prime (p^2 - 2) :=
by
  sorry

end disproof_of_Alitta_l2257_225723


namespace dave_apps_left_l2257_225770

def initial_apps : ℕ := 24
def initial_files : ℕ := 9
def files_left : ℕ := 5
def apps_left (files_left: ℕ) : ℕ := files_left + 7

theorem dave_apps_left :
  apps_left files_left = 12 :=
by
  sorry

end dave_apps_left_l2257_225770


namespace three_numbers_lcm_ratio_l2257_225779

theorem three_numbers_lcm_ratio
  (x : ℕ)
  (h1 : 3 * x.gcd 4 = 1)
  (h2 : (3 * x * 4 * x) / x.gcd (3 * x) = 180)
  (h3 : ∃ y : ℕ, y = 5 * (3 * x))
  : (3 * x = 45 ∧ 4 * x = 60 ∧ 5 * (3 * x) = 225) ∧
      lcm (lcm (3 * x) (4 * x)) (5 * (3 * x)) = 900 :=
by
  sorry

end three_numbers_lcm_ratio_l2257_225779


namespace trapezoid_DC_length_l2257_225753

theorem trapezoid_DC_length 
  (AB DC: ℝ) (BC: ℝ) 
  (angle_BCD angle_CDA: ℝ)
  (h1: AB = 8)
  (h2: BC = 4 * Real.sqrt 3)
  (h3: angle_BCD = 60)
  (h4: angle_CDA = 45)
  (h5: AB = DC):
  DC = 14 + 4 * Real.sqrt 2 :=
sorry

end trapezoid_DC_length_l2257_225753


namespace Luke_piles_of_quarters_l2257_225758

theorem Luke_piles_of_quarters (Q : ℕ) (h : 6 * Q = 30) : Q = 5 :=
by
  sorry

end Luke_piles_of_quarters_l2257_225758


namespace num_pens_multiple_of_16_l2257_225785

theorem num_pens_multiple_of_16 (Pencils Students : ℕ) (h1 : Pencils = 928) (h2 : Students = 16)
  (h3 : ∃ (Pn : ℕ), Pencils = Pn * Students) :
  ∃ (k : ℕ), ∃ (Pens : ℕ), Pens = 16 * k :=
by
  sorry

end num_pens_multiple_of_16_l2257_225785


namespace days_needed_to_wash_all_towels_l2257_225727

def towels_per_hour : ℕ := 7
def hours_per_day : ℕ := 2
def total_towels : ℕ := 98

theorem days_needed_to_wash_all_towels :
  (total_towels / (towels_per_hour * hours_per_day)) = 7 :=
by
  sorry

end days_needed_to_wash_all_towels_l2257_225727


namespace train_crossing_time_l2257_225735

theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform)
  (time_to_cross_signal_pole : ℝ := train_length / train_speed) :
  train_length = 300 ∧ platform_length = 1000 ∧ time_to_cross_platform = 39 → time_to_cross_signal_pole = 9 := by
  intro h
  cases h
  sorry

end train_crossing_time_l2257_225735


namespace fruit_seller_loss_percentage_l2257_225717

theorem fruit_seller_loss_percentage :
  ∃ (C : ℝ), 
    (5 : ℝ) = C - (6.25 - C * (1 + 0.05)) → 
    (C = 6.25) → 
    (C - 5 = 1.25) → 
    (1.25 / 6.25 * 100 = 20) :=
by 
  sorry

end fruit_seller_loss_percentage_l2257_225717


namespace inequality_proof_l2257_225709

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d):
    1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) :=
by
  sorry

end inequality_proof_l2257_225709


namespace fg_eq_gf_condition_l2257_225762

theorem fg_eq_gf_condition (m n p q : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := 
sorry

end fg_eq_gf_condition_l2257_225762


namespace ellipse_condition_range_k_l2257_225775

theorem ellipse_condition_range_k (k : ℝ) : 
  (2 - k > 0) ∧ (3 + k > 0) ∧ (2 - k ≠ 3 + k) → -3 < k ∧ k < 2 := 
by 
  sorry

end ellipse_condition_range_k_l2257_225775


namespace findSolutions_l2257_225743

-- Define the given mathematical problem
def originalEquation (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3)) / ((x - 4) * (x - 6) * (x - 4)) = 1

-- Define the conditions where the equation is valid
def validCondition (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6

-- Define the set of solutions
def solutions (x : ℝ) : Prop :=
  x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2

-- The theorem stating the correct set of solutions
theorem findSolutions (x : ℝ) : originalEquation x ∧ validCondition x ↔ solutions x :=
by sorry

end findSolutions_l2257_225743


namespace original_number_of_laborers_l2257_225718

theorem original_number_of_laborers (L : ℕ) 
  (h : L * 9 = (L - 6) * 15) : L = 15 :=
sorry

end original_number_of_laborers_l2257_225718


namespace retirement_year_l2257_225737

-- Define the basic conditions
def rule_of_70 (age: ℕ) (years_of_employment: ℕ) : Prop :=
  age + years_of_employment ≥ 70

def age_in_hiring_year : ℕ := 32
def hiring_year : ℕ := 1987

theorem retirement_year : ∃ y: ℕ, rule_of_70 (age_in_hiring_year + y) y ∧ (hiring_year + y = 2006) :=
  sorry

end retirement_year_l2257_225737


namespace solution_l2257_225748

-- Define the vectors and their conditions
variables {u v : ℝ}

def vec1 := (3, -2)
def vec2 := (9, -7)
def vec3 := (-1, 2)
def vec4 := (-3, 4)

-- Condition: The linear combination of vec1 and u*vec2 equals the linear combination of vec3 and v*vec4.
axiom H : (3 + 9 * u, -2 - 7 * u) = (-1 - 3 * v, 2 + 4 * v)

-- Statement of the proof problem:
theorem solution : u = -4/15 ∧ v = -8/15 :=
by {
  sorry
}

end solution_l2257_225748


namespace fraction_is_one_fifth_l2257_225716

theorem fraction_is_one_fifth
  (x a b : ℤ)
  (hx : x^2 = 25)
  (h2x : 2 * x = a * x / b + 9) :
  a = 1 ∧ b = 5 :=
by
  sorry

end fraction_is_one_fifth_l2257_225716


namespace part_a_l2257_225720

theorem part_a 
  (x y u v : ℝ) 
  (h1 : x + y = u + v) 
  (h2 : x^2 + y^2 = u^2 + v^2) : 
  ∀ n : ℕ, x^n + y^n = u^n + v^n := 
by sorry

end part_a_l2257_225720


namespace doors_per_apartment_l2257_225725

def num_buildings : ℕ := 2
def num_floors_per_building : ℕ := 12
def num_apt_per_floor : ℕ := 6
def total_num_doors : ℕ := 1008

theorem doors_per_apartment : total_num_doors / (num_buildings * num_floors_per_building * num_apt_per_floor) = 7 :=
by
  sorry

end doors_per_apartment_l2257_225725


namespace smallest_number_l2257_225796

/-
  Let's declare each number in its base form as variables,
  convert them to their decimal equivalents, and assert that the decimal
  value of $(31)_4$ is the smallest among the given numbers.

  Note: We're not providing the proof steps, just the statement.
-/

noncomputable def A_base7_to_dec : ℕ := 2 * 7^1 + 0 * 7^0
noncomputable def B_base5_to_dec : ℕ := 3 * 5^1 + 0 * 5^0
noncomputable def C_base6_to_dec : ℕ := 2 * 6^1 + 3 * 6^0
noncomputable def D_base4_to_dec : ℕ := 3 * 4^1 + 1 * 4^0

theorem smallest_number : D_base4_to_dec < A_base7_to_dec ∧ D_base4_to_dec < B_base5_to_dec ∧ D_base4_to_dec < C_base6_to_dec := by
  sorry

end smallest_number_l2257_225796


namespace derivative_of_f_l2257_225708

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -x * Real.sin x := by
  sorry

end derivative_of_f_l2257_225708


namespace ferris_wheel_time_10_seconds_l2257_225750

noncomputable def time_to_reach_height (R : ℝ) (T : ℝ) (h : ℝ) : ℝ :=
  let ω := 2 * Real.pi / T
  let t := (Real.arcsin (h / R - 1)) / ω
  t

theorem ferris_wheel_time_10_seconds :
  time_to_reach_height 30 120 15 = 10 :=
by
  sorry

end ferris_wheel_time_10_seconds_l2257_225750


namespace exinscribed_sphere_inequality_l2257_225765

variable (r r_A r_B r_C r_D : ℝ)

theorem exinscribed_sphere_inequality 
  (hr : 0 < r) 
  (hrA : 0 < r_A) 
  (hrB : 0 < r_B) 
  (hrC : 0 < r_C) 
  (hrD : 0 < r_D) :
  1 / Real.sqrt (r_A^2 - r_A * r_B + r_B^2) +
  1 / Real.sqrt (r_B^2 - r_B * r_C + r_C^2) +
  1 / Real.sqrt (r_C^2 - r_C * r_D + r_D^2) +
  1 / Real.sqrt (r_D^2 - r_D * r_A + r_A^2) ≤
  2 / r := by
  sorry

end exinscribed_sphere_inequality_l2257_225765


namespace output_of_code_snippet_is_six_l2257_225702

-- Define the variables and the condition
def a : ℕ := 3
def y : ℕ := if a < 10 then 2 * a else a * a 

-- The statement to be proved
theorem output_of_code_snippet_is_six :
  y = 6 :=
by
  sorry

end output_of_code_snippet_is_six_l2257_225702


namespace solve_for_x_l2257_225745

theorem solve_for_x : ∀ (x : ℝ), (-3 * x - 8 = 5 * x + 4) → (x = -3 / 2) := by
  intro x
  intro h
  sorry

end solve_for_x_l2257_225745


namespace at_least_one_is_one_l2257_225710

theorem at_least_one_is_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  (1/x + 1/y + 1/z = 1) → (1/(x + y + z) = 1) → (x = 1 ∨ y = 1 ∨ z = 1) :=
by
  sorry

end at_least_one_is_one_l2257_225710


namespace new_bottles_from_recycling_l2257_225776

theorem new_bottles_from_recycling (initial_bottles : ℕ) (required_bottles : ℕ) (h : initial_bottles = 125) (r : required_bottles = 5) : 
∃ new_bottles : ℕ, new_bottles = (initial_bottles / required_bottles ^ 2 + initial_bottles / (required_bottles * required_bottles / required_bottles) + initial_bottles / (required_bottles * required_bottles * required_bottles / required_bottles * required_bottles * required_bottles)) :=
  sorry

end new_bottles_from_recycling_l2257_225776


namespace ratio_of_saturday_to_friday_customers_l2257_225780

def tips_per_customer : ℝ := 2.0
def customers_friday : ℕ := 28
def customers_sunday : ℕ := 36
def total_tips : ℝ := 296

theorem ratio_of_saturday_to_friday_customers :
  let tips_friday := customers_friday * tips_per_customer
  let tips_sunday := customers_sunday * tips_per_customer
  let tips_friday_and_sunday := tips_friday + tips_sunday
  let tips_saturday := total_tips - tips_friday_and_sunday
  let customers_saturday := tips_saturday / tips_per_customer
  (customers_saturday / customers_friday : ℝ) = 3 := 
by
  sorry

end ratio_of_saturday_to_friday_customers_l2257_225780


namespace seashells_ratio_l2257_225730

theorem seashells_ratio (s_1 s_2 S t s3 : ℕ) (hs1 : s_1 = 5) (hs2 : s_2 = 7) (hS : S = 36)
  (ht : t = s_1 + s_2) (hs3 : s3 = S - t) :
  s3 / t = 2 :=
by
  rw [hs1, hs2] at ht
  simp at ht
  rw [hS, ht] at hs3
  simp at hs3
  sorry

end seashells_ratio_l2257_225730


namespace find_tax_percentage_l2257_225795

-- Definitions based on given conditions
def income_total : ℝ := 58000
def income_threshold : ℝ := 40000
def tax_above_threshold_percentage : ℝ := 0.2
def total_tax : ℝ := 8000

-- Let P be the percentage taxed on the first $40,000
variable (P : ℝ)

-- Formulate the problem as a proof goal
theorem find_tax_percentage (h : total_tax = 8000) :
  P = ((total_tax - (tax_above_threshold_percentage * (income_total - income_threshold))) / income_threshold) * 100 :=
by sorry

end find_tax_percentage_l2257_225795


namespace sum_arithmetic_sequence_l2257_225792

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
    ∃ d, ∀ n, a (n+1) = a n + d

-- The conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
    (a 1 + a 2 + a 3 = 6)

def condition_2 (a : ℕ → ℝ) : Prop :=
    (a 10 + a 11 + a 12 = 9)

-- The Theorem statement
theorem sum_arithmetic_sequence :
    is_arithmetic_sequence a →
    condition_1 a →
    condition_2 a →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 30) :=
by
  intro h1 h2 h3
  sorry

end sum_arithmetic_sequence_l2257_225792
