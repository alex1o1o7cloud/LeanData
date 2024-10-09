import Mathlib

namespace box_width_l1886_188663

theorem box_width (W S : ℕ) (h1 : 30 * W * 12 = 80 * S^3) (h2 : S ∣ 30 ∧ S ∣ 12) : W = 48 :=
by
  sorry

end box_width_l1886_188663


namespace parts_production_equation_l1886_188621

theorem parts_production_equation (x : ℝ) : 
  let apr := 50
  let may := 50 * (1 + x)
  let jun := 50 * (1 + x) * (1 + x)
  (apr + may + jun = 182) :=
sorry

end parts_production_equation_l1886_188621


namespace squirrel_spring_acorns_l1886_188605

/--
A squirrel had stashed 210 acorns to last him the three winter months. 
It divided the pile into thirds, one for each month, and then took some 
from each third, leaving 60 acorns for each winter month. The squirrel 
combined the ones it took to eat in the first cold month of spring. 
Prove that the number of acorns the squirrel has for the beginning of spring 
is 30.
-/
theorem squirrel_spring_acorns :
  ∀ (initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month : ℕ),
    initial_acorns = 210 →
    acorns_per_month = initial_acorns / 3 →
    remaining_acorns_per_month = 60 →
    acorns_taken_per_month = acorns_per_month - remaining_acorns_per_month →
    3 * acorns_taken_per_month = 30 :=
by
  intros initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month
  sorry

end squirrel_spring_acorns_l1886_188605


namespace value_of_def_ef_l1886_188635

theorem value_of_def_ef
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : (a * f) / (c * d) = 1)
  : d * e * f = 250 := 
by 
  sorry

end value_of_def_ef_l1886_188635


namespace age_contradiction_l1886_188651

-- Given the age ratios and future age of Sandy
def current_ages (x : ℕ) : ℕ × ℕ × ℕ := (4 * x, 3 * x, 5 * x)
def sandy_age_after_6_years (age_sandy_current : ℕ) : ℕ := age_sandy_current + 6

-- Given conditions
def ratio_condition (x : ℕ) (age_sandy age_molly age_danny : ℕ) : Prop :=
  current_ages x = (age_sandy, age_molly, age_danny)

def sandy_age_condition (age_sandy_current : ℕ) : Prop :=
  sandy_age_after_6_years age_sandy_current = 30

def age_sum_condition (age_molly age_danny : ℕ) : Prop :=
  age_molly + age_danny = (age_molly + 4) + (age_danny + 4)

-- Main theorem
theorem age_contradiction : ∃ x age_sandy age_molly age_danny, 
  ratio_condition x age_sandy age_molly age_danny ∧
  sandy_age_condition age_sandy ∧
  (¬ age_sum_condition age_molly age_danny) := 
by
  -- Omitting the proof; the focus is on setting up the statement only
  sorry

end age_contradiction_l1886_188651


namespace matrix_B_pow48_l1886_188699

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 2], ![0, -2, 0]]

theorem matrix_B_pow48 :
  B ^ 48 = ![![0, 0, 0], ![0, 16^12, 0], ![0, 0, 16^12]] :=
by sorry

end matrix_B_pow48_l1886_188699


namespace difference_Q_R_l1886_188616

variable (P Q R : ℝ) (x : ℝ)

theorem difference_Q_R (h1 : 11 * x - 5 * x = 12100) : 19 * x - 11 * x = 16133.36 :=
by
  sorry

end difference_Q_R_l1886_188616


namespace mean_of_five_numbers_l1886_188623

theorem mean_of_five_numbers (a b c d e : ℚ) (h : a + b + c + d + e = 2/3) : 
  (a + b + c + d + e) / 5 = 2 / 15 := 
by 
  -- This is where the proof would go, but we'll omit it as per instructions
  sorry

end mean_of_five_numbers_l1886_188623


namespace special_op_equality_l1886_188631

def special_op (x y : ℕ) : ℕ := x * y - x - 2 * y

theorem special_op_equality : (special_op 7 4) - (special_op 4 7) = 3 := by
  sorry

end special_op_equality_l1886_188631


namespace regular_polygon_sides_l1886_188626

theorem regular_polygon_sides (ex_angle : ℝ) (hne_zero : ex_angle ≠ 0)
  (sum_ext_angles : ∀ (n : ℕ), n > 2 → n * ex_angle = 360) :
  ∃ (n : ℕ), n * 15 = 360 ∧ n = 24 :=
by 
  sorry

end regular_polygon_sides_l1886_188626


namespace contrapositive_proposition_l1886_188652

theorem contrapositive_proposition (x : ℝ) : 
  (x^2 = 1 → (x = 1 ∨ x = -1)) ↔ ((x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by
  sorry

end contrapositive_proposition_l1886_188652


namespace jimmy_income_l1886_188665

variable (J : ℝ)

def rebecca_income : ℝ := 15000
def income_increase : ℝ := 3000
def rebecca_income_after_increase : ℝ := rebecca_income + income_increase
def combined_income : ℝ := 2 * rebecca_income_after_increase

theorem jimmy_income (h : rebecca_income_after_increase + J = combined_income) : 
  J = 18000 := by
  sorry

end jimmy_income_l1886_188665


namespace find_extrema_of_A_l1886_188662

theorem find_extrema_of_A (x y : ℝ) (h : x^2 + y^2 = 4) : 2 ≤ x^2 + x * y + y^2 ∧ x^2 + x * y + y^2 ≤ 6 :=
by 
  sorry

end find_extrema_of_A_l1886_188662


namespace work_completion_time_l1886_188643

theorem work_completion_time
  (W : ℝ) -- Total work
  (p_rate : ℝ := W / 40) -- p's work rate
  (q_rate : ℝ := W / 24) -- q's work rate
  (work_done_by_p_alone : ℝ := 8 * p_rate) -- Work done by p in first 8 days
  (remaining_work : ℝ := W - work_done_by_p_alone) -- Remaining work after 8 days
  (combined_rate : ℝ := p_rate + q_rate) -- Combined work rate of p and q
  (time_to_complete_remaining_work : ℝ := remaining_work / combined_rate) -- Time to complete remaining work
  : (8 + time_to_complete_remaining_work) = 20 :=
by
  sorry

end work_completion_time_l1886_188643


namespace coin_exchange_impossible_l1886_188612

theorem coin_exchange_impossible :
  ∀ (n : ℕ), (n % 4 = 1) → (¬ (∃ k : ℤ, n + 4 * k = 26)) :=
by
  intros n h
  sorry

end coin_exchange_impossible_l1886_188612


namespace arithmetic_sequence_a6_value_l1886_188694

theorem arithmetic_sequence_a6_value
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) :
  a 6 = 2 :=
by
  sorry

end arithmetic_sequence_a6_value_l1886_188694


namespace julia_fourth_day_candies_l1886_188647

-- Definitions based on conditions
def first_day (x : ℚ) := (1/5) * x
def second_day (x : ℚ) := (1/2) * (4/5) * x
def third_day (x : ℚ) := (1/2) * (2/5) * x
def fourth_day (x : ℚ) := (2/5) * x - (1/2) * (2/5) * x

-- The Lean statement to prove
theorem julia_fourth_day_candies (x : ℚ) (h : x ≠ 0): 
  fourth_day x / x = 1/5 :=
by
  -- insert proof here
  sorry

end julia_fourth_day_candies_l1886_188647


namespace combined_average_yield_l1886_188624

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end combined_average_yield_l1886_188624


namespace tournament_total_players_l1886_188640

/--
In a tournament involving n players:
- Each player scored half of all their points in matches against participants who took the last three places.
- Each game results in 1 point.
- Total points from matches among the last three (bad) players = 3.
- The number of games between good and bad players = 3n - 9.
- Total points good players scored from bad players = 3n - 12.
- Games among good players total to (n-3)(n-4)/2 resulting points.
Prove that the total number of participants in the tournament is 9.
-/
theorem tournament_total_players (n : ℕ) :
  3 * (n - 4) = (n - 3) * (n - 4) / 2 → 
  n = 9 :=
by
  intros h
  sorry

end tournament_total_players_l1886_188640


namespace Emily_spent_28_dollars_l1886_188695

theorem Emily_spent_28_dollars :
  let roses_cost := 4
  let daisies_cost := 3
  let tulips_cost := 5
  let lilies_cost := 6
  let roses_qty := 2
  let daisies_qty := 3
  let tulips_qty := 1
  let lilies_qty := 1
  (roses_qty * roses_cost) + (daisies_qty * daisies_cost) + (tulips_qty * tulips_cost) + (lilies_qty * lilies_cost) = 28 :=
by
  sorry

end Emily_spent_28_dollars_l1886_188695


namespace binomial_10_3_l1886_188660

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l1886_188660


namespace speed_of_second_half_l1886_188617

theorem speed_of_second_half (t d s1 d1 d2 : ℝ) (h_t : t = 30) (h_d : d = 672) (h_s1 : s1 = 21)
  (h_d1 : d1 = d / 2) (h_d2 : d2 = d / 2) (h_t1 : d1 / s1 = 16) (h_t2 : t - d1 / s1 = 14) :
  d2 / 14 = 24 :=
by sorry

end speed_of_second_half_l1886_188617


namespace probability_not_within_square_B_l1886_188601

theorem probability_not_within_square_B {A B : Type} 
  (area_A : ℝ) (perimeter_B : ℝ) (area_B : ℝ) (not_covered : ℝ) 
  (h1 : area_A = 30) 
  (h2 : perimeter_B = 16) 
  (h3 : area_B = 16) 
  (h4 : not_covered = area_A - area_B) :
  (not_covered / area_A) = 7 / 15 := by sorry

end probability_not_within_square_B_l1886_188601


namespace find_num_chickens_l1886_188687

-- Definitions based on problem conditions
def num_dogs : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2
def total_legs_seen : ℕ := 12

-- Proof problem: Prove the number of chickens Mrs. Hilt saw
theorem find_num_chickens (C : ℕ) (h1 : num_dogs * legs_per_dog + C * legs_per_chicken = total_legs_seen) : C = 2 := 
sorry

end find_num_chickens_l1886_188687


namespace average_physics_chemistry_l1886_188681

theorem average_physics_chemistry (P C M : ℕ) 
  (h1 : (P + C + M) / 3 = 80)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 80) :
  (P + C) / 2 = 70 := 
sorry

end average_physics_chemistry_l1886_188681


namespace min_moves_to_reset_counters_l1886_188693

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end min_moves_to_reset_counters_l1886_188693


namespace noel_baked_dozens_l1886_188676

theorem noel_baked_dozens (total_students : ℕ) (percent_like_donuts : ℝ)
    (donuts_per_student : ℕ) (dozen : ℕ) (h_total_students : total_students = 30)
    (h_percent_like_donuts : percent_like_donuts = 0.80)
    (h_donuts_per_student : donuts_per_student = 2)
    (h_dozen : dozen = 12) :
    total_students * percent_like_donuts * donuts_per_student / dozen = 4 := 
by
  sorry

end noel_baked_dozens_l1886_188676


namespace last_two_digits_7_pow_2017_l1886_188659

noncomputable def last_two_digits_of_pow :=
  ∀ n : ℕ, ∃ (d : ℕ), d < 100 ∧ 7^n % 100 = d

theorem last_two_digits_7_pow_2017 : ∃ (d : ℕ), d = 7 ∧ 7^2017 % 100 = d :=
by
  sorry

end last_two_digits_7_pow_2017_l1886_188659


namespace option_C_is_nonnegative_rational_l1886_188672

def isNonNegativeRational (x : ℚ) : Prop :=
  x ≥ 0

theorem option_C_is_nonnegative_rational :
  isNonNegativeRational (-( - (4^2 : ℚ))) :=
by
  sorry

end option_C_is_nonnegative_rational_l1886_188672


namespace find_number_l1886_188622

theorem find_number (x : ℝ) : x = 7 ∧ x^2 + 95 = (x - 19)^2 :=
by
  sorry

end find_number_l1886_188622


namespace quadratic_solution_set_R_l1886_188674

theorem quadratic_solution_set_R (a b c : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : b^2 - 4 * a * c < 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
by sorry

end quadratic_solution_set_R_l1886_188674


namespace beads_per_bracelet_l1886_188670

def beads_bella_has : Nat := 36
def beads_bella_needs : Nat := 12
def total_bracelets : Nat := 6

theorem beads_per_bracelet : (beads_bella_has + beads_bella_needs) / total_bracelets = 8 :=
by
  sorry

end beads_per_bracelet_l1886_188670


namespace probability_same_heads_l1886_188667

noncomputable def probability_heads_after_flips (p : ℚ) (n : ℕ) : ℚ :=
  (1 - p)^(n-1) * p

theorem probability_same_heads (p : ℚ) (n : ℕ) : p = 1/3 → 
  ∑' n : ℕ, (probability_heads_after_flips p n)^4 = 1/65 := 
sorry

end probability_same_heads_l1886_188667


namespace geometric_sequence_fourth_term_l1886_188613

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (r : ℝ)
    (h₁ : a₁ = 5^(3/4))
    (h₂ : a₂ = 5^(1/2))
    (h₃ : a₃ = 5^(1/4))
    (geometric_seq : a₂ = a₁ * r ∧ a₃ = a₂ * r) :
    a₃ * r = 1 := 
by
  sorry

end geometric_sequence_fourth_term_l1886_188613


namespace total_gym_cost_l1886_188657

def cheap_monthly_fee : ℕ := 10
def cheap_signup_fee : ℕ := 50
def expensive_monthly_fee : ℕ := 3 * cheap_monthly_fee
def expensive_signup_fee : ℕ := 4 * expensive_monthly_fee

def yearly_cost_cheap : ℕ := 12 * cheap_monthly_fee + cheap_signup_fee
def yearly_cost_expensive : ℕ := 12 * expensive_monthly_fee + expensive_signup_fee

theorem total_gym_cost : yearly_cost_cheap + yearly_cost_expensive = 650 := by
  -- Proof goes here
  sorry

end total_gym_cost_l1886_188657


namespace blueberries_cartons_proof_l1886_188619

def total_needed_cartons : ℕ := 26
def strawberries_cartons : ℕ := 10
def cartons_to_buy : ℕ := 7

theorem blueberries_cartons_proof :
  strawberries_cartons + cartons_to_buy + 9 = total_needed_cartons :=
by
  sorry

end blueberries_cartons_proof_l1886_188619


namespace difference_of_squares_example_l1886_188611

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l1886_188611


namespace function_value_sum_l1886_188628

namespace MathProof

variable {f : ℝ → ℝ}

theorem function_value_sum :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 5) = f x) →
  f (1 / 3) = 2022 →
  f (1 / 2) = 17 →
  f (-7) + f 12 + f (16 / 3) + f (9 / 2) = 2005 :=
by
  intros h_odd h_periodic h_f13 h_f12
  sorry

end MathProof

end function_value_sum_l1886_188628


namespace inclination_angle_of_line_l1886_188638

-- Definitions and conditions
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : angle = 45 := by
  sorry

end inclination_angle_of_line_l1886_188638


namespace subtract_3a_result_l1886_188669

theorem subtract_3a_result (a : ℝ) : 
  (9 * a^2 - 3 * a + 8) + 3 * a = 9 * a^2 + 8 := 
sorry

end subtract_3a_result_l1886_188669


namespace measured_weight_loss_l1886_188627

variable (W : ℝ) (hW : W > 0)

noncomputable def final_weigh_in (initial_weight : ℝ) : ℝ :=
  (0.90 * initial_weight) * 1.02

theorem measured_weight_loss :
  final_weigh_in W = 0.918 * W → (W - final_weigh_in W) / W * 100 = 8.2 := 
by
  intro h
  unfold final_weigh_in at h
  -- skip detailed proof steps, focus on the statement
  sorry

end measured_weight_loss_l1886_188627


namespace old_edition_pages_l1886_188675

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l1886_188675


namespace problem_l1886_188661

def operation (a b : ℤ) (h : a ≠ 0) : ℤ := (b - a) ^ 2 / a ^ 2

theorem problem : 
  operation (-1) (operation 1 (-1) (by decide)) (by decide) = 25 := 
by
  sorry

end problem_l1886_188661


namespace calculate_R_cubed_plus_R_squared_plus_R_l1886_188648

theorem calculate_R_cubed_plus_R_squared_plus_R (R : ℕ) (hR : R > 0)
  (h1 : ∃ q : ℚ, q = (R / (2 * R + 2)) * ((R - 1) / (2 * R + 1)))
  (h2 : (R / (2 * R + 2)) * ((R + 2) / (2 * R + 1)) + ((R + 2) / (2 * R + 2)) * (R / (2 * R + 1)) = 3 * q) :
  R^3 + R^2 + R = 399 :=
by
  sorry

end calculate_R_cubed_plus_R_squared_plus_R_l1886_188648


namespace necessary_but_not_sufficient_condition_l1886_188685

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1886_188685


namespace no_valid_n_exists_l1886_188697

theorem no_valid_n_exists :
  ¬ ∃ n : ℕ, 219 ≤ n ∧ n ≤ 2019 ∧ ∃ x y : ℕ, 
    1 ≤ x ∧ x < n ∧ n < y ∧ (∀ k : ℕ, k ≤ n → k ≠ x ∧ k ≠ x+1 → y % k = 0) := 
by {
  sorry
}

end no_valid_n_exists_l1886_188697


namespace quadratic_no_real_roots_l1886_188636

theorem quadratic_no_real_roots : ∀ (a b c : ℝ), a ≠ 0 → Δ = (b*b - 4*a*c) → x^2 + 3 = 0 → Δ < 0 := by
  sorry

end quadratic_no_real_roots_l1886_188636


namespace land_plot_side_length_l1886_188600

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length_l1886_188600


namespace evaluate_expression_l1886_188615

theorem evaluate_expression : (20 * 3 + 10) / (5 + 3) = 9 := by
  sorry

end evaluate_expression_l1886_188615


namespace bobby_additional_candy_l1886_188649

variable (initial_candy additional_candy chocolate total_candy : ℕ)
variable (bobby_initial_candy : initial_candy = 38)
variable (bobby_ate_chocolate : chocolate = 16)
variable (bobby_more_candy : initial_candy + additional_candy = 58 + chocolate)

theorem bobby_additional_candy :
  additional_candy = 36 :=
by {
  sorry
}

end bobby_additional_candy_l1886_188649


namespace percent_problem_l1886_188653

theorem percent_problem (x y z w : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) 
  (h3 : z = 0.70 * w) : 
  x = 0.336 * w :=
sorry

end percent_problem_l1886_188653


namespace proof_base_5_conversion_and_addition_l1886_188645

-- Define the given numbers in decimal (base 10)
def n₁ := 45
def n₂ := 25

-- Base 5 conversion function and proofs of correctness
def to_base_5 (n : ℕ) : ℕ := sorry
def from_base_5 (n : ℕ) : ℕ := sorry

-- Converted values to base 5
def a₅ : ℕ := to_base_5 n₁
def b₅ : ℕ := to_base_5 n₂

-- Sum in base 5
def c₅ : ℕ := a₅ + b₅  -- addition in base 5

-- Convert the final sum back to decimal base 10
def d₁₀ : ℕ := from_base_5 c₅

theorem proof_base_5_conversion_and_addition :
  d₁₀ = 65 ∧ to_base_5 65 = 230 :=
by sorry

end proof_base_5_conversion_and_addition_l1886_188645


namespace position_of_21_over_19_in_sequence_l1886_188625

def sequence_term (n : ℕ) : ℚ := (n + 3) / (n + 1)

theorem position_of_21_over_19_in_sequence :
  ∃ n : ℕ, sequence_term n = 21 / 19 ∧ n = 18 :=
by sorry

end position_of_21_over_19_in_sequence_l1886_188625


namespace largest_ball_radius_l1886_188654

def torus_inner_radius : ℝ := 2
def torus_outer_radius : ℝ := 4
def circle_center : ℝ × ℝ × ℝ := (3, 0, 1)
def circle_radius : ℝ := 1

theorem largest_ball_radius : ∃ r : ℝ, r = 9 / 4 ∧
  (∃ (sphere_center : ℝ × ℝ × ℝ) (torus_center : ℝ × ℝ × ℝ),
  (sphere_center = (0, 0, r)) ∧
  (torus_center = (3, 0, 1)) ∧
  (dist (0, 0, r) (3, 0, 1) = r + 1)) := sorry

end largest_ball_radius_l1886_188654


namespace robotics_club_students_l1886_188682

theorem robotics_club_students
  (total_students : ℕ)
  (cs_students : ℕ)
  (electronics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 50)
  (h3 : electronics_students = 35)
  (h4 : both_students = 25) :
  total_students - (cs_students - both_students + electronics_students - both_students + both_students) = 20 :=
by
  sorry

end robotics_club_students_l1886_188682


namespace probability_two_white_balls_same_color_l1886_188678

theorem probability_two_white_balls_same_color :
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  (total_combinations_white + total_combinations_black > 0) →
  (total_combinations_white / total_combinations_same_color) = (3 / 4) :=
by
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  intro h
  sorry

end probability_two_white_balls_same_color_l1886_188678


namespace find_three_numbers_l1886_188606

theorem find_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a + b - c = 10) 
  (h3 : a - b + c = 8) : 
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := 
by 
  sorry

end find_three_numbers_l1886_188606


namespace initial_ratio_of_stamps_l1886_188602

theorem initial_ratio_of_stamps (P Q : ℕ) (h1 : ((P - 8 : ℤ) : ℚ) / (Q + 8) = 6 / 5) (h2 : P - 8 = Q + 8) : P / Q = 6 / 5 :=
sorry

end initial_ratio_of_stamps_l1886_188602


namespace minimum_tangent_length_4_l1886_188664

noncomputable def minimum_tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((b + 4)^2 + (b - 2)^2 - 2)

theorem minimum_tangent_length_4 :
  ∀ (a b : ℝ), (x^2 + y^2 + 2 * x - 4 * y + 3 = 0) ∧ (x = a ∧ y = b) ∧ (2*a*x + b*y + 6 = 0) → 
    minimum_tangent_length a b = 4 :=
by
  sorry

end minimum_tangent_length_4_l1886_188664


namespace find_T_l1886_188671

variable {n : ℕ}
variable {a b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- Conditions
axiom h1 : ∀ n, b n - a n = 2^n + 1
axiom h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2

-- Goal
theorem find_T (n : ℕ) (a b S T : ℕ → ℕ)
  (h1 : ∀ n, b n - a n = 2^n + 1)
  (h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2) :
  T n = 2^(n + 1) + n * (n + 1) / 2 - 5 := sorry

end find_T_l1886_188671


namespace P1_coordinates_l1886_188658

-- Define initial point coordinates
def P : (ℝ × ℝ) := (0, 3)

-- Define the transformation functions
def move_left (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1 - units, p.2)
def move_up (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1, p.2 + units)

-- Calculate the coordinates of point P1
def P1 : (ℝ × ℝ) := move_up (move_left P 2) 1

-- Statement to prove
theorem P1_coordinates : P1 = (-2, 4) := by
  sorry

end P1_coordinates_l1886_188658


namespace bottles_left_on_shelf_l1886_188684

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l1886_188684


namespace triangle_longest_side_l1886_188680

theorem triangle_longest_side 
  (x : ℝ)
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) :
  2 * x + 1 = 17 := by
  sorry

end triangle_longest_side_l1886_188680


namespace faster_train_length_is_150_l1886_188604

def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def time_seconds : ℝ := 15

noncomputable def length_faster_train : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  relative_speed_mps * time_seconds

theorem faster_train_length_is_150 :
  length_faster_train = 150 := by
sorry

end faster_train_length_is_150_l1886_188604


namespace remainder_modulo_l1886_188603

theorem remainder_modulo (y : ℕ) (hy : 5 * y ≡ 1 [MOD 17]) : (7 + y) % 17 = 14 :=
sorry

end remainder_modulo_l1886_188603


namespace sufficient_not_necessary_condition_l1886_188629

theorem sufficient_not_necessary_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : (0 < x ∧ x < 2) → (x^2 - x - 2 < 0) :=
by
  intros h
  sorry

end sufficient_not_necessary_condition_l1886_188629


namespace dennis_years_taught_l1886_188614

theorem dennis_years_taught (A V D : ℕ) (h1 : V + A + D = 75) (h2 : V = A + 9) (h3 : V = D - 9) : D = 34 :=
sorry

end dennis_years_taught_l1886_188614


namespace greatest_of_three_consecutive_integers_with_sum_21_l1886_188696

theorem greatest_of_three_consecutive_integers_with_sum_21 :
  ∃ (x : ℤ), (x + (x + 1) + (x + 2) = 21) ∧ ((x + 2) = 8) :=
by
  sorry

end greatest_of_three_consecutive_integers_with_sum_21_l1886_188696


namespace divisible_by_13_l1886_188608

theorem divisible_by_13 (a : ℤ) (h₀ : 0 ≤ a) (h₁ : a ≤ 13) : (51^2015 + a) % 13 = 0 → a = 1 :=
by
  sorry

end divisible_by_13_l1886_188608


namespace rect_length_is_20_l1886_188677

-- Define the conditions
def rect_length_four_times_width (l w : ℝ) : Prop := l = 4 * w
def rect_area_100 (l w : ℝ) : Prop := l * w = 100

-- The main theorem to prove
theorem rect_length_is_20 {l w : ℝ} (h1 : rect_length_four_times_width l w) (h2 : rect_area_100 l w) : l = 20 := by
  sorry

end rect_length_is_20_l1886_188677


namespace function_zeros_range_l1886_188683

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x + 2 / x else x * Real.log x - a

theorem function_zeros_range (a : ℝ) :
  (∀ x : ℝ, f x a = 0 → x < 0) ∧ (∀ x : ℝ, f x a = 0 → x > 0 → (a > -1 / Real.exp 1 ∧ a < 0)) ↔
  (a > -1 / Real.exp 1 ∧ a < 0) :=
sorry

end function_zeros_range_l1886_188683


namespace track_length_l1886_188634

theorem track_length (L : ℝ)
  (h_brenda_first_meeting : ∃ (brenda_run1: ℝ), brenda_run1 = 100)
  (h_sally_first_meeting : ∃ (sally_run1: ℝ), sally_run1 = L/2 - 100)
  (h_brenda_second_meeting : ∃ (brenda_run2: ℝ), brenda_run2 = L - 100)
  (h_sally_second_meeting : ∃ (sally_run2: ℝ), sally_run2 = sally_run1 + 100)
  (h_meeting_total : brenda_run2 + sally_run2 = L) :
  L = 200 :=
by
  sorry

end track_length_l1886_188634


namespace num_perfect_squares_l1886_188692

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l1886_188692


namespace f_one_value_l1886_188630

def f (x a: ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem f_one_value (a : ℝ) (h : ∀ x, f x a ≥ 0) : f 1 a = 4 :=
by
  sorry

end f_one_value_l1886_188630


namespace ratio_of_girls_who_like_pink_l1886_188637

theorem ratio_of_girls_who_like_pink 
  (total_students : ℕ) (answered_green : ℕ) (answered_yellow : ℕ) (total_girls : ℕ) (answered_yellow_students : ℕ)
  (portion_girls_pink : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_green = total_students / 2)
  (h3 : total_girls = 18)
  (h4 : answered_yellow_students = 9)
  (answered_pink := total_students - answered_green - answered_yellow_students)
  (ratio_pink : ℚ := answered_pink / total_girls) : 
  ratio_pink = 1 / 3 :=
sorry

end ratio_of_girls_who_like_pink_l1886_188637


namespace range_of_S_l1886_188609

variable {a b x : ℝ}
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem range_of_S (h1 : ∀ x ∈ Set.Icc 0 1, |f x a b| ≤ 1) :
  ∃ l u, -2 ≤ l ∧ u ≤ 9 / 4 ∧ ∀ (S : ℝ), (S = (a + 1) * (b + 1)) → l ≤ S ∧ S ≤ u :=
by
  sorry

end range_of_S_l1886_188609


namespace arithmetic_sequence_iff_condition_l1886_188632

-- Definitions: A sequence and the condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a ↔ (∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) :=
by
  -- Proof is omitted.
  sorry

end arithmetic_sequence_iff_condition_l1886_188632


namespace probability_greater_than_4_l1886_188673

-- Given conditions
def die_faces : ℕ := 6
def favorable_outcomes : Finset ℕ := {5, 6}

-- Probability calculation
def probability (total : ℕ) (favorable : Finset ℕ) : ℚ :=
  favorable.card / total

theorem probability_greater_than_4 :
  probability die_faces favorable_outcomes = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l1886_188673


namespace abs_monotonic_increasing_even_l1886_188607

theorem abs_monotonic_increasing_even :
  (∀ x : ℝ, |x| = |(-x)|) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → |x1| ≤ |x2|) :=
by
  sorry

end abs_monotonic_increasing_even_l1886_188607


namespace solve_for_x_l1886_188690

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l1886_188690


namespace inequality_proof_l1886_188644

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l1886_188644


namespace sasha_train_problem_l1886_188686

def wagon_number (W : ℕ) (S : ℕ) : Prop :=
  -- Conditions
  (1 ≤ W ∧ W ≤ 9) ∧          -- Wagon number is a single-digit number
  (S < W) ∧                  -- Seat number is less than the wagon number
  ( (W = 1 ∧ S ≠ 1) ∨ 
    (W = 2 ∧ S = 1)
  ) -- Monday is the 1st or 2nd day of the month and corresponding seat constraints

theorem sasha_train_problem :
  ∃ (W S : ℕ), wagon_number W S ∧ W = 2 ∧ S = 1 :=
by
  sorry

end sasha_train_problem_l1886_188686


namespace stacy_height_last_year_l1886_188679

-- Definitions for the conditions
def brother_growth := 1
def stacy_growth := brother_growth + 6
def stacy_current_height := 57
def stacy_last_years_height := stacy_current_height - stacy_growth

-- Proof statement
theorem stacy_height_last_year : stacy_last_years_height = 50 :=
by
  -- proof steps will go here
  sorry

end stacy_height_last_year_l1886_188679


namespace flowers_per_bouquet_l1886_188646

noncomputable def num_flowers_per_bouquet (total_flowers wilted_flowers bouquets : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / bouquets

theorem flowers_per_bouquet : num_flowers_per_bouquet 53 18 5 = 7 := by
  sorry

end flowers_per_bouquet_l1886_188646


namespace polynomial_identity_l1886_188689

theorem polynomial_identity (P : ℝ → ℝ) :
  (∀ x, (x - 1) * P (x + 1) - (x + 2) * P x = 0) ↔ ∃ a : ℝ, ∀ x, P x = a * (x^3 - x) :=
by
  sorry

end polynomial_identity_l1886_188689


namespace find_first_year_l1886_188691

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

-- Define the conditions
def after_2020 (n : ℕ) : Prop := n > 2020
def sum_of_digits_eq (n required_sum : ℕ) : Prop := sum_of_digits n = required_sum

noncomputable def first_year_after_2020_with_digit_sum_15 : ℕ :=
  2049

-- The statement to be proved
theorem find_first_year : 
  ∃ y : ℕ, after_2020 y ∧ sum_of_digits_eq y 15 ∧ y = first_year_after_2020_with_digit_sum_15 :=
by
  sorry

end find_first_year_l1886_188691


namespace number_of_tens_in_sum_l1886_188655

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l1886_188655


namespace pears_total_correct_l1886_188650

noncomputable def pickedPearsTotal (sara_picked tim_picked : Nat) : Nat :=
  sara_picked + tim_picked

theorem pears_total_correct :
    pickedPearsTotal 6 5 = 11 :=
  by
    sorry

end pears_total_correct_l1886_188650


namespace sum_abcd_l1886_188641

variable (a b c d : ℝ)

theorem sum_abcd :
  (∃ y : ℝ, 2 * a + 3 = y ∧ 2 * b + 4 = y ∧ 2 * c + 5 = y ∧ 2 * d + 6 = y ∧ a + b + c + d + 10 = y) →
  a + b + c + d = -11 :=
by
  sorry

end sum_abcd_l1886_188641


namespace ducks_among_non_falcons_l1886_188656

-- Definitions based on conditions
def percentage_birds := 100
def percentage_ducks := 40
def percentage_cranes := 20
def percentage_falcons := 15
def percentage_pigeons := 25

-- Question converted into the statement
theorem ducks_among_non_falcons : 
  (percentage_ducks / (percentage_birds - percentage_falcons) * percentage_birds) = 47 :=
by
  sorry

end ducks_among_non_falcons_l1886_188656


namespace bacteria_colony_growth_l1886_188688

theorem bacteria_colony_growth : 
  ∃ (n : ℕ), n = 4 ∧ 5 * 3 ^ n > 200 ∧ (∀ (m : ℕ), 5 * 3 ^ m > 200 → m ≥ n) :=
by
  sorry

end bacteria_colony_growth_l1886_188688


namespace height_large_cylinder_is_10_l1886_188698

noncomputable def height_large_cylinder : ℝ :=
  let V_small := 13.5 * Real.pi
  let factor := 74.07407407407408
  let V_large := 100 * Real.pi
  factor * V_small / V_large

theorem height_large_cylinder_is_10 :
  height_large_cylinder = 10 :=
by
  sorry

end height_large_cylinder_is_10_l1886_188698


namespace tim_age_difference_l1886_188618

theorem tim_age_difference (j_turned_23_j_turned_35 : ∃ (j_age_when_james_23 : ℕ) (john_age_when_james_23 : ℕ), 
                                          j_age_when_james_23 = 23 ∧ john_age_when_james_23 = 35)
                           (tim_age : ℕ) (tim_age_eq : tim_age = 79)
                           (tim_age_twice_john_age_less_X : ∃ (X : ℕ) (john_age : ℕ), tim_age = 2 * john_age - X) :
  ∃ (X : ℕ), X = 15 :=
by
  sorry

end tim_age_difference_l1886_188618


namespace rafael_earnings_l1886_188620

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end rafael_earnings_l1886_188620


namespace number_of_sick_animals_l1886_188633

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l1886_188633


namespace increase_by_fraction_l1886_188642

theorem increase_by_fraction (original_value : ℕ) (fraction : ℚ) : original_value = 120 → fraction = 5/6 → original_value + original_value * fraction = 220 :=
by
  intros h1 h2
  sorry

end increase_by_fraction_l1886_188642


namespace intersection_A_B_l1886_188610

def setA : Set ℝ := {x | x^2 - 1 < 0}
def setB : Set ℝ := {x | x > 0}

theorem intersection_A_B : setA ∩ setB = {x | 0 < x ∧ x < 1} := 
by 
  sorry

end intersection_A_B_l1886_188610


namespace find_M_N_sum_l1886_188666

theorem find_M_N_sum
  (M N : ℕ)
  (h1 : 3 * 75 = 5 * M)
  (h2 : 3 * N = 5 * 90) :
  M + N = 195 := 
sorry

end find_M_N_sum_l1886_188666


namespace correct_combined_monthly_rate_of_profit_l1886_188639

structure Book :=
  (cost_price : ℕ)
  (selling_price : ℕ)
  (months_held : ℕ)

def profit (b : Book) : ℕ :=
  b.selling_price - b.cost_price

def monthly_rate_of_profit (b : Book) : ℕ :=
  if b.months_held = 0 then profit b else profit b / b.months_held

def combined_monthly_rate_of_profit (b1 b2 b3 : Book) : ℕ :=
  monthly_rate_of_profit b1 + monthly_rate_of_profit b2 + monthly_rate_of_profit b3

theorem correct_combined_monthly_rate_of_profit :
  combined_monthly_rate_of_profit
    {cost_price := 50, selling_price := 90, months_held := 1}
    {cost_price := 120, selling_price := 150, months_held := 2}
    {cost_price := 75, selling_price := 110, months_held := 0} 
    = 90 := 
by
  sorry

end correct_combined_monthly_rate_of_profit_l1886_188639


namespace thirteen_consecutive_nat_power_l1886_188668

def consecutive_sum_power (N : ℕ) : ℕ :=
  (N - 6) + (N - 5) + (N - 4) + (N - 3) + (N - 2) + (N - 1) +
  N + (N + 1) + (N + 2) + (N + 3) + (N + 4) + (N + 5) + (N + 6)

theorem thirteen_consecutive_nat_power (N : ℕ) (n : ℕ) :
  N = 13^2020 →
  n = 2021 →
  consecutive_sum_power N = 13^n := by
  sorry

end thirteen_consecutive_nat_power_l1886_188668
