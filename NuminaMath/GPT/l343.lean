import Mathlib

namespace odd_three_mn_l343_34346

theorem odd_three_mn (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) : (3 * m * n) % 2 = 1 :=
sorry

end odd_three_mn_l343_34346


namespace interval_of_defined_expression_l343_34320

theorem interval_of_defined_expression (x : ℝ) :
  (x > 2 ∧ x < 5) ↔ (x - 2 > 0 ∧ 5 - x > 0) :=
by
  sorry

end interval_of_defined_expression_l343_34320


namespace solve_system_of_equations_l343_34355

theorem solve_system_of_equations (x y : ℝ) :
    (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧ 5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
    (x = 2 ∧ y = 1) ∨ (x = 2 / 5 ∧ y = -(1 / 5)) :=
by
  sorry

end solve_system_of_equations_l343_34355


namespace find_intersection_points_l343_34378

def intersection_points (t α : ℝ) : Prop :=
∃ t α : ℝ,
  (2 + t, -1 - t) = (3 * Real.cos α, 3 * Real.sin α) ∧
  ((2 + t = (1 + Real.sqrt 17) / 2 ∧ -1 - t = (1 - Real.sqrt 17) / 2) ∨
   (2 + t = (1 - Real.sqrt 17) / 2 ∧ -1 - t = (1 + Real.sqrt 17) / 2))

theorem find_intersection_points : intersection_points t α :=
sorry

end find_intersection_points_l343_34378


namespace points_in_quadrant_I_l343_34340

theorem points_in_quadrant_I (x y : ℝ) :
  (y > 3 * x) ∧ (y > 5 - 2 * x) → (x > 0) ∧ (y > 0) := by
  sorry

end points_in_quadrant_I_l343_34340


namespace second_train_speed_l343_34305

noncomputable def speed_of_second_train (length1 length2 speed1 clearance_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000 -- convert meters to kilometers
  let time_in_hours := clearance_time / 3600 -- convert seconds to hours
  let relative_speed := total_distance / time_in_hours
  relative_speed - speed1

theorem second_train_speed : 
  speed_of_second_train 60 280 42 16.998640108791296 = 30.05 := 
by
  sorry

end second_train_speed_l343_34305


namespace probability_of_number_between_21_and_30_l343_34328

-- Define the success condition of forming a two-digit number between 21 and 30.
def successful_number (d1 d2 : Nat) : Prop :=
  let n1 := 10 * d1 + d2
  let n2 := 10 * d2 + d1
  (21 ≤ n1 ∧ n1 ≤ 30) ∨ (21 ≤ n2 ∧ n2 ≤ 30)

-- Calculate the probability of a successful outcome.
def probability_success (favorable total : Nat) : Nat :=
  favorable / total

-- The main theorem claiming the probability that Melinda forms a number between 21 and 30.
theorem probability_of_number_between_21_and_30 :
  let successful_counts := 10
  let total_possible := 36
  probability_success successful_counts total_possible = 5 / 18 :=
by
  sorry

end probability_of_number_between_21_and_30_l343_34328


namespace prime_gt_three_square_minus_one_divisible_by_twentyfour_l343_34394

theorem prime_gt_three_square_minus_one_divisible_by_twentyfour (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 24 ∣ (p^2 - 1) :=
sorry

end prime_gt_three_square_minus_one_divisible_by_twentyfour_l343_34394


namespace find_smaller_circle_radius_l343_34350

noncomputable def smaller_circle_radius (R : ℝ) : ℝ :=
  R / (Real.sqrt 2 - 1)

theorem find_smaller_circle_radius (R : ℝ) (x : ℝ) :
  (∀ (c1 c2 c3 c4 : ℝ),  c1 = c2 ∧ c2 = c3 ∧ c3 = c4 ∧ c4 = x
  ∧ c1 + c2 = 2 * c3 * Real.sqrt 2)
  → x = smaller_circle_radius R :=
by 
  intros h
  sorry

end find_smaller_circle_radius_l343_34350


namespace custom_op_4_8_l343_34392

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := b + b / a

-- Theorem stating the desired equality
theorem custom_op_4_8 : custom_op 4 8 = 10 :=
by
  -- Proof is omitted
  sorry

end custom_op_4_8_l343_34392


namespace find_ellipse_focus_l343_34367

theorem find_ellipse_focus :
  ∀ (a b : ℝ), a^2 = 5 → b^2 = 4 → 
  (∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) →
  ((∃ c : ℝ, c^2 = a^2 - b^2) ∧ (∃ x y, x = 0 ∧ (y = 1 ∨ y = -1))) :=
by
  sorry

end find_ellipse_focus_l343_34367


namespace no_square_cube_l343_34314

theorem no_square_cube (n : ℕ) (h : n > 0) : ¬ (∃ k : ℕ, k^2 = n * (n + 1) * (n + 2) * (n + 3)) ∧ ¬ (∃ l : ℕ, l^3 = n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end no_square_cube_l343_34314


namespace gcd_lcm_product_l343_34330

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 24) (h2 : b = 60) :
  Nat.gcd a b * Nat.lcm a b = 1440 :=
by
  sorry

end gcd_lcm_product_l343_34330


namespace sin_405_eq_sqrt2_div_2_l343_34354

theorem sin_405_eq_sqrt2_div_2 :
  Real.sin (405 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_eq_sqrt2_div_2_l343_34354


namespace initial_savings_correct_l343_34395

-- Define the constants for ticket prices and number of tickets.
def vip_ticket_price : ℕ := 100
def vip_tickets : ℕ := 2
def regular_ticket_price : ℕ := 50
def regular_tickets : ℕ := 3
def leftover_savings : ℕ := 150

-- Define the total cost of tickets.
def total_cost : ℕ := (vip_ticket_price * vip_tickets) + (regular_ticket_price * regular_tickets)

-- Define the initial savings calculation.
def initial_savings : ℕ := total_cost + leftover_savings

-- Theorem stating the initial savings should be $500.
theorem initial_savings_correct : initial_savings = 500 :=
by
  -- Proof steps can be added here.
  sorry

end initial_savings_correct_l343_34395


namespace bell_peppers_needed_l343_34382

-- Definitions based on the conditions
def large_slices_per_bell_pepper : ℕ := 20
def small_pieces_from_half_slices : ℕ := (20 / 2) * 3
def total_slices_and_pieces_per_bell_pepper : ℕ := large_slices_per_bell_pepper / 2 + small_pieces_from_half_slices
def desired_total_slices_and_pieces : ℕ := 200

-- Proving the number of bell peppers needed
theorem bell_peppers_needed : 
  desired_total_slices_and_pieces / total_slices_and_pieces_per_bell_pepper = 5 := 
by 
  -- Add the proof steps here
  sorry

end bell_peppers_needed_l343_34382


namespace inequality_proof_l343_34329

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > y)
  (hy : y > 1)
  (hz : 1 > z)
  (hzpos : z > 0)
  (a : ℝ := (1 + x * z) / z)
  (b : ℝ := (1 + x * y) / x)
  (c : ℝ := (1 + y * z) / y) :
  a > b ∧ a > c :=
by
  sorry

end inequality_proof_l343_34329


namespace number_multiplied_by_3_l343_34351

theorem number_multiplied_by_3 (k : ℕ) : 
  2^13 - 2^(13-2) = 3 * k → k = 2048 :=
by
  sorry

end number_multiplied_by_3_l343_34351


namespace prob_two_blue_balls_l343_34397

-- Ball and Urn Definitions
def total_balls : ℕ := 10
def blue_balls_initial : ℕ := 6
def red_balls_initial : ℕ := 4

-- Probabilities
def prob_blue_first_draw : ℚ := blue_balls_initial / total_balls
def prob_blue_second_draw_given_first_blue : ℚ :=
  (blue_balls_initial - 1) / (total_balls - 1)

-- Resulting Probability
def prob_both_blue : ℚ := prob_blue_first_draw * prob_blue_second_draw_given_first_blue

-- Statement to Prove
theorem prob_two_blue_balls :
  prob_both_blue = 1 / 3 :=
by
  sorry

end prob_two_blue_balls_l343_34397


namespace lucy_deposit_l343_34324

theorem lucy_deposit :
  ∃ D : ℝ, 
    let initial_balance := 65 
    let withdrawal := 4 
    let final_balance := 76 
    initial_balance + D - withdrawal = final_balance ∧ D = 15 :=
by
  -- sorry skips the proof
  sorry

end lucy_deposit_l343_34324


namespace marty_combinations_l343_34341

theorem marty_combinations : 
  ∃ n : ℕ, n = 5 * 4 ∧ n = 20 :=
by
  sorry

end marty_combinations_l343_34341


namespace find_f_minus_3_l343_34348

def rational_function (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + (3 * f x / x) = 2 * x^2

theorem find_f_minus_3 (f : ℚ → ℚ) (h : rational_function f) : 
  f (-3) = 494 / 117 :=
by
  sorry

end find_f_minus_3_l343_34348


namespace find_k_in_expression_l343_34318

theorem find_k_in_expression :
  (2^1004 + 5^1005)^2 - (2^1004 - 5^1005)^2 = 20 * 10^1004 :=
by
  sorry

end find_k_in_expression_l343_34318


namespace proof_q_is_true_l343_34370

variable (p q : Prop)

-- Assuming the conditions
axiom h1 : p ∨ q   -- p or q is true
axiom h2 : ¬ p     -- not p is true

-- Theorem statement to prove q is true
theorem proof_q_is_true : q :=
by
  sorry

end proof_q_is_true_l343_34370


namespace consecutive_page_sum_l343_34365

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 479160) : n + (n + 1) + (n + 2) = 234 :=
sorry

end consecutive_page_sum_l343_34365


namespace hours_on_task2_l343_34375

theorem hours_on_task2
    (total_hours_per_week : ℕ) 
    (work_days_per_week : ℕ) 
    (hours_per_day_task1 : ℕ) 
    (hours_reduction_task1 : ℕ)
    (h_total_hours : total_hours_per_week = 40)
    (h_work_days : work_days_per_week = 5)
    (h_hours_task1 : hours_per_day_task1 = 5)
    (h_hours_reduction : hours_reduction_task1 = 5)
    : (total_hours_per_week / 2 / work_days_per_week) = 4 :=
by
  -- Skipping proof with sorry
  sorry

end hours_on_task2_l343_34375


namespace solve_exponential_problem_l343_34383

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  let max_value := if a > 1 then a^2 else a
  let min_value := if a > 1 then a else a^2
  max_value - min_value = a / 2

theorem solve_exponential_problem (a : ℝ) (hpos : a > 0) (hne1 : a ≠ 1) :
  satisfies_condition a ↔ (a = 1 / 2 ∨ a = 3 / 2) :=
sorry

end solve_exponential_problem_l343_34383


namespace general_solution_of_differential_eq_l343_34358

noncomputable def y (x C : ℝ) : ℝ := x * (Real.exp (x ^ 2) + C)

theorem general_solution_of_differential_eq {x C : ℝ} (h : x ≠ 0) :
  let y' := (1 : ℝ) * (Real.exp (x ^ 2) + C) + x * (2 * x * Real.exp (x ^ 2))
  y' = (y x C / x) + 2 * x ^ 2 * Real.exp (x ^ 2) :=
by
  -- the proof goes here
  sorry

end general_solution_of_differential_eq_l343_34358


namespace race_head_start_l343_34331

variables {Va Vb L H : ℝ}

theorem race_head_start
  (h1 : Va = 20 / 14 * Vb)
  (h2 : L / Va = (L - H) / Vb) : 
  H = 3 / 10 * L :=
by
  sorry

end race_head_start_l343_34331


namespace second_term_arithmetic_seq_l343_34380

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l343_34380


namespace remainder_4x_div_9_l343_34334

theorem remainder_4x_div_9 (x : ℕ) (k : ℤ) (h : x = 9 * k + 5) : (4 * x) % 9 = 2 := 
by sorry

end remainder_4x_div_9_l343_34334


namespace compare_quadratics_maximize_rectangle_area_l343_34362

-- (Ⅰ) Problem statement for comparing quadratic expressions
theorem compare_quadratics (x : ℝ) : (x + 1) * (x - 3) > (x + 2) * (x - 4) := by
  sorry

-- (Ⅱ) Problem statement for maximizing rectangular area with given perimeter
theorem maximize_rectangle_area (x y : ℝ) (h : 2 * (x + y) = 36) : 
  x = 9 ∧ y = 9 ∧ x * y = 81 := by
  sorry

end compare_quadratics_maximize_rectangle_area_l343_34362


namespace find_two_numbers_l343_34319

noncomputable def x := 5 + 2 * Real.sqrt 5
noncomputable def y := 5 - 2 * Real.sqrt 5

theorem find_two_numbers :
  (x * y = 5) ∧ (x + y = 10) :=
by {
  sorry
}

end find_two_numbers_l343_34319


namespace average_weight_of_children_l343_34343

theorem average_weight_of_children
  (S_B S_G : ℕ)
  (avg_boys_weight : S_B = 8 * 160)
  (avg_girls_weight : S_G = 5 * 110) :
  (S_B + S_G) / 13 = 141 := 
by
  sorry

end average_weight_of_children_l343_34343


namespace number_of_paperback_books_l343_34398

variables (P H : ℕ)

theorem number_of_paperback_books (h1 : H = 4) (h2 : P / 3 + 2 * H = 10) : P = 6 := 
by
  sorry

end number_of_paperback_books_l343_34398


namespace mary_initial_borrowed_books_l343_34387

-- We first define the initial number of books B.
variable (B : ℕ)

-- Next, we encode the conditions into a final condition of having 12 books.
def final_books (B : ℕ) : ℕ := (B - 3 + 5) - 2 + 7

-- The proof problem is to show that B must be 5.
theorem mary_initial_borrowed_books (B : ℕ) (h : final_books B = 12) : B = 5 :=
by
  sorry

end mary_initial_borrowed_books_l343_34387


namespace op_15_5_eq_33_l343_34309

def op (x y : ℕ) : ℕ :=
  2 * x + x / y

theorem op_15_5_eq_33 : op 15 5 = 33 := by
  sorry

end op_15_5_eq_33_l343_34309


namespace circle_passing_through_points_eq_l343_34364

theorem circle_passing_through_points_eq :
  let A := (-2, 1)
  let B := (9, 3)
  let C := (1, 7)
  let center := (7/2, 2)
  let radius_sq := 125 / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_sq ↔ 
    (∃ t : ℝ, (x - center.1)^2 + (y - center.2)^2 = t^2) ∧
    ∀ P : ℝ × ℝ, P = A ∨ P = B ∨ P = C → (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius_sq := by sorry

end circle_passing_through_points_eq_l343_34364


namespace bicycle_parking_income_l343_34332

theorem bicycle_parking_income (x : ℝ) (y : ℝ) 
    (h1 : 0 ≤ x ∧ x ≤ 2000)
    (h2 : y = 0.5 * x + 0.8 * (2000 - x)) : 
    y = -0.3 * x + 1600 := by
  sorry

end bicycle_parking_income_l343_34332


namespace fourth_vertex_of_parallelogram_l343_34300

structure Point where
  x : ℤ
  y : ℤ

def midPoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def isMidpoint (M P Q : Point) : Prop :=
  M = midPoint P Q

theorem fourth_vertex_of_parallelogram (A B C D : Point)
  (hA : A = {x := -2, y := 1})
  (hB : B = {x := -1, y := 3})
  (hC : C = {x := 3, y := 4})
  (h1 : isMidpoint (midPoint A C) B D ∨
        isMidpoint (midPoint A B) C D ∨
        isMidpoint (midPoint B C) A D) :
  D = {x := 2, y := 2} ∨ D = {x := -6, y := 0} ∨ D = {x := 4, y := 6} := by
  sorry

end fourth_vertex_of_parallelogram_l343_34300


namespace glove_pair_probability_l343_34336

/-- 
A box contains 6 pairs of black gloves (i.e., 12 black gloves) and 4 pairs of beige gloves (i.e., 8 beige gloves).
We need to prove that the probability of drawing a matching pair of gloves is 47/95.
-/
theorem glove_pair_probability : 
  let total_gloves := 20
  let black_gloves := 12
  let beige_gloves := 8
  let P1_black := (black_gloves / total_gloves) * ((black_gloves - 1) / (total_gloves - 1))
  let P2_beige := (beige_gloves / total_gloves) * ((beige_gloves - 1) / (total_gloves - 1))
  let total_probability := P1_black + P2_beige
  total_probability = 47 / 95 :=
sorry

end glove_pair_probability_l343_34336


namespace number_of_players_l343_34306

theorem number_of_players (S : ℕ) (h1 : S = 22) (h2 : ∀ (n : ℕ), S = n * 2) : ∃ n, n = 11 :=
by
  sorry

end number_of_players_l343_34306


namespace sum_of_two_smallest_l343_34345

variable (a b c d : ℕ)
variable (x : ℕ)

-- Four numbers a, b, c, d are in the ratio 3:5:7:9
def ratios := (a = 3 * x) ∧ (b = 5 * x) ∧ (c = 7 * x) ∧ (d = 9 * x)

-- The average of these numbers is 30
def average := (a + b + c + d) / 4 = 30

-- The theorem to prove the sum of the two smallest numbers (a and b) is 40
theorem sum_of_two_smallest (h1 : ratios a b c d x) (h2 : average a b c d) : a + b = 40 := by
  sorry

end sum_of_two_smallest_l343_34345


namespace distance_and_area_of_triangle_l343_34335

theorem distance_and_area_of_triangle :
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  distance = 10 ∧ area = 24 :=
by
  let p1 := (0, 6)
  let p2 := (8, 0)
  let origin := (0, 0)
  let distance := Real.sqrt ((8 - 0)^2 + (0 - 6)^2)
  let area := (1 / 2 : ℝ) * 8 * 6
  have h_dist : distance = 10 := sorry
  have h_area : area = 24 := sorry
  exact ⟨h_dist, h_area⟩

end distance_and_area_of_triangle_l343_34335


namespace probability_non_smokers_getting_lung_cancer_l343_34379

theorem probability_non_smokers_getting_lung_cancer 
  (overall_lung_cancer : ℝ)
  (smokers_fraction : ℝ)
  (smokers_lung_cancer : ℝ)
  (non_smokers_lung_cancer : ℝ)
  (H1 : overall_lung_cancer = 0.001)
  (H2 : smokers_fraction = 0.2)
  (H3 : smokers_lung_cancer = 0.004)
  (H4 : overall_lung_cancer = smokers_fraction * smokers_lung_cancer + (1 - smokers_fraction) * non_smokers_lung_cancer) :
  non_smokers_lung_cancer = 0.00025 := by
  sorry

end probability_non_smokers_getting_lung_cancer_l343_34379


namespace total_businesses_l343_34389

theorem total_businesses (B : ℕ) (h1 : B / 2 + B / 3 + 12 = B) : B = 72 :=
sorry

end total_businesses_l343_34389


namespace option_b_does_not_represent_5x_l343_34399

theorem option_b_does_not_represent_5x (x : ℝ) : 
  (∀ a, a = 5 * x ↔ a = x + x + x + x + x) →
  (¬ (5 * x = x * x * x * x * x)) :=
by
  intro h
  -- Using sorry to skip the proof.
  sorry

end option_b_does_not_represent_5x_l343_34399


namespace base_7_to_base_10_l343_34325

theorem base_7_to_base_10 (a b c d e : ℕ) (h : 23456 = e * 10000 + d * 1000 + c * 100 + b * 10 + a) :
  2 * 7^4 + 3 * 7^3 + 4 * 7^2 + 5 * 7^1 + 6 * 7^0 = 6068 :=
by
  sorry

end base_7_to_base_10_l343_34325


namespace students_in_both_clubs_l343_34363

theorem students_in_both_clubs:
  ∀ (U D S : Finset ℕ ), (U.card = 300) → (D.card = 100) → (S.card = 140) → (D ∪ S).card = 210 → (D ∩ S).card = 30 := 
sorry

end students_in_both_clubs_l343_34363


namespace both_teams_joint_renovation_team_renovation_split_l343_34339

-- Problem setup for part 1
def renovation_total_length : ℕ := 2400
def teamA_daily_progress : ℕ := 30
def teamB_daily_progress : ℕ := 50
def combined_days_to_complete_renovation : ℕ := 30

theorem both_teams_joint_renovation (x : ℕ) :
  (teamA_daily_progress + teamB_daily_progress) * x = renovation_total_length → 
  x = combined_days_to_complete_renovation :=
by
  sorry

-- Problem setup for part 2
def total_renovation_days : ℕ := 60
def length_renovated_by_teamA : ℕ := 900
def length_renovated_by_teamB : ℕ := 1500

theorem team_renovation_split (a b : ℕ) :
  a / teamA_daily_progress + b / teamB_daily_progress = total_renovation_days ∧ 
  a + b = renovation_total_length → 
  a = length_renovated_by_teamA ∧ b = length_renovated_by_teamB :=
by
  sorry

end both_teams_joint_renovation_team_renovation_split_l343_34339


namespace sales_tax_difference_l343_34381

theorem sales_tax_difference : 
  let price : Float := 50
  let tax1 : Float := 0.0725
  let tax2 : Float := 0.07
  let sales_tax1 := price * tax1
  let sales_tax2 := price * tax2
  sales_tax1 - sales_tax2 = 0.125 := 
by
  sorry

end sales_tax_difference_l343_34381


namespace find_second_derivative_at_1_l343_34321

-- Define the function f(x) and its second derivative
noncomputable def f (x : ℝ) := x * Real.exp x
noncomputable def f'' (x : ℝ) := (x + 2) * Real.exp x

-- State the theorem to be proved
theorem find_second_derivative_at_1 : f'' 1 = 2 * Real.exp 1 := by
  sorry

end find_second_derivative_at_1_l343_34321


namespace problem_remainders_l343_34385

open Int

theorem problem_remainders (x : ℤ) :
  (x + 2) % 45 = 7 →
  ((x + 2) % 20 = 7 ∧ x % 19 = 5) :=
by
  sorry

end problem_remainders_l343_34385


namespace inequality_hold_l343_34302

theorem inequality_hold {a b : ℝ} (h : a < b) : -3 * a > -3 * b :=
sorry

end inequality_hold_l343_34302


namespace max_product_two_integers_l343_34376

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l343_34376


namespace evening_to_morning_ratio_l343_34313

-- Definitions based on conditions
def morning_miles : ℕ := 2
def total_miles : ℕ := 12
def evening_miles : ℕ := total_miles - morning_miles

-- Lean statement to prove the ratio
theorem evening_to_morning_ratio : evening_miles / morning_miles = 5 := by
  -- we simply state the final ratio we want to prove
  sorry

end evening_to_morning_ratio_l343_34313


namespace smallest_nine_consecutive_sum_l343_34304

theorem smallest_nine_consecutive_sum (n : ℕ) (h : (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) = 2007)) : n = 219 :=
sorry

end smallest_nine_consecutive_sum_l343_34304


namespace vertices_of_parabolas_is_parabola_l343_34349

theorem vertices_of_parabolas_is_parabola 
  (a c k : ℝ) (ha : 0 < a) (hc : 0 < c) (hk : 0 < k) :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) ∧ 
  ∀ (pt : ℝ × ℝ), (∃ t : ℝ, pt = (-(k * t) / (2 * a), f t)) → 
  ∃ a' b' c', (∀ t : ℝ, pt.2 = a' * pt.1^2 + b' * pt.1 + c') ∧ (a < 0) :=
by sorry

end vertices_of_parabolas_is_parabola_l343_34349


namespace age_ratio_l343_34323

theorem age_ratio (B_age : ℕ) (H1 : B_age = 34) (A_age : ℕ) (H2 : A_age = B_age + 4) :
  (A_age + 10) / (B_age - 10) = 2 :=
by
  sorry

end age_ratio_l343_34323


namespace add_in_base14_l343_34301

-- Define symbols A, B, C, D in base 10 as they are used in the base 14 representation
def base14_A : ℕ := 10
def base14_B : ℕ := 11
def base14_C : ℕ := 12
def base14_D : ℕ := 13

-- Define the numbers given in base 14
def num1_base14 : ℕ := 9 * 14^2 + base14_C * 14 + 7
def num2_base14 : ℕ := 4 * 14^2 + base14_B * 14 + 3

-- Define the expected result in base 14
def result_base14 : ℕ := 1 * 14^2 + 0 * 14 + base14_A

-- The theorem statement that needs to be proven
theorem add_in_base14 : num1_base14 + num2_base14 = result_base14 := by
  sorry

end add_in_base14_l343_34301


namespace volume_of_each_hemisphere_container_is_correct_l343_34372

-- Define the given conditions
def Total_volume : ℕ := 10936
def Number_containers : ℕ := 2734

-- Define the volume of each hemisphere container
def Volume_each_container : ℕ := Total_volume / Number_containers

-- The theorem to prove, asserting the volume is correct
theorem volume_of_each_hemisphere_container_is_correct :
  Volume_each_container  = 4 := by
  -- placeholder for the actual proof
  sorry

end volume_of_each_hemisphere_container_is_correct_l343_34372


namespace josie_initial_amount_is_correct_l343_34368

def cost_of_milk := 4.00 / 2
def cost_of_bread := 3.50
def cost_of_detergent_after_coupon := 10.25 - 1.25
def cost_of_bananas := 2 * 0.75
def total_cost := cost_of_milk + cost_of_bread + cost_of_detergent_after_coupon + cost_of_bananas
def leftover := 4.00
def initial_amount := total_cost + leftover

theorem josie_initial_amount_is_correct :
  initial_amount = 20.00 := by
  sorry

end josie_initial_amount_is_correct_l343_34368


namespace length_of_QR_l343_34377

theorem length_of_QR {P Q R N : Type} 
  (PQ PR QR : ℝ) (QN NR PN : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 10)
  (h3 : QN = 3 * NR)
  (h4 : PN = 6)
  (h5 : QR = QN + NR) :
  QR = 724 / 3 :=
by sorry

end length_of_QR_l343_34377


namespace identify_roles_l343_34317

-- Define the number of liars and truth-tellers
def num_liars : Nat := 1000
def num_truth_tellers : Nat := 1000

-- Define the properties of the individuals
def first_person_is_liar := true
def second_person_is_truth_teller := true

-- The main statement equivalent to the problem
theorem identify_roles : first_person_is_liar = true ∧ second_person_is_truth_teller = true := by
  sorry

end identify_roles_l343_34317


namespace moving_circle_trajectory_l343_34311

theorem moving_circle_trajectory (x y : ℝ) 
  (fixed_circle : x^2 + y^2 = 4): 
  (x^2 + y^2 = 9) ∨ (x^2 + y^2 = 1) :=
sorry

end moving_circle_trajectory_l343_34311


namespace problem_l343_34347

theorem problem (a b : ℝ) (h₁ : a = -a) (h₂ : b = 1 / b) : a + b = 1 ∨ a + b = -1 :=
  sorry

end problem_l343_34347


namespace problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l343_34386

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution_l343_34386


namespace center_of_circle_polar_coords_l343_34371

theorem center_of_circle_polar_coords :
  ∀ (θ : ℝ), ∃ (ρ : ℝ), (ρ, θ) = (2, Real.pi) ∧ ρ = - 4 * Real.cos θ := 
sorry

end center_of_circle_polar_coords_l343_34371


namespace find_p_q_r_divisibility_l343_34396

theorem find_p_q_r_divisibility 
  (p q r : ℝ)
  (h_div : ∀ x, (x^4 + 4*x^3 + 6*p*x^2 + 4*q*x + r) % (x^3 + 3*x^2 + 9*x + 3) = 0)
  : (p + q) * r = 15 :=
by
  -- Proof steps would go here
  sorry

end find_p_q_r_divisibility_l343_34396


namespace intersection_S_T_l343_34307

def S := {x : ℝ | abs x < 5}
def T := {x : ℝ | (x + 7) * (x - 3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_S_T_l343_34307


namespace eval_expression_l343_34310

theorem eval_expression (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := 
by 
  sorry

end eval_expression_l343_34310


namespace quotient_of_division_l343_34359

theorem quotient_of_division (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181) (h2 : divisor = 20) (h3 : remainder = 1) 
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 9 :=
by
  sorry -- proof goes here

end quotient_of_division_l343_34359


namespace triangle_perimeter_l343_34338

theorem triangle_perimeter (x : ℕ) :
  (x = 6 ∨ x = 3) →
  ∃ (a b c : ℕ), (a = x ∧ (b = x ∨ c = x)) ∧ 
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by
  intro h
  sorry

end triangle_perimeter_l343_34338


namespace mangoes_per_kg_l343_34342

theorem mangoes_per_kg (total_kg : ℕ) (sold_market_kg : ℕ) (sold_community_factor : ℚ) (remaining_mangoes : ℕ) (mangoes_per_kg : ℕ) :
  total_kg = 60 ∧ sold_market_kg = 20 ∧ sold_community_factor = 1/2 ∧ remaining_mangoes = 160 → mangoes_per_kg = 8 :=
  by
  sorry

end mangoes_per_kg_l343_34342


namespace parabola_directrix_l343_34384

theorem parabola_directrix (y x : ℝ) (h : y^2 = -4 * x) : x = 1 :=
sorry

end parabola_directrix_l343_34384


namespace ratio_areas_ACEF_ADC_l343_34352

-- Define the basic geometric setup
variables (A B C D E F : Point) 
variables (BC CD DE : ℝ) 
variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) 

-- Assuming the given conditions, we want to prove the ratio of areas
noncomputable def ratio_areas (α : ℝ) : ℝ := 4 * (1 - α)

theorem ratio_areas_ACEF_ADC (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) :
  ratio_areas α = 4 * (1 - α) :=
sorry

end ratio_areas_ACEF_ADC_l343_34352


namespace good_number_is_1008_l343_34357

-- Given conditions
def sum_1_to_2015 : ℕ := (2015 * (2015 + 1)) / 2
def sum_mod_2016 : ℕ := sum_1_to_2015 % 2016

-- The proof problem expressed in Lean
theorem good_number_is_1008 (x : ℕ) (h1 : sum_1_to_2015 = 2031120)
  (h2 : sum_mod_2016 = 1008) :
  x = 1008 ↔ (sum_1_to_2015 - x) % 2016 = 0 := by
  sorry

end good_number_is_1008_l343_34357


namespace mike_spent_on_new_tires_l343_34315

-- Define the given amounts
def amount_spent_on_speakers : ℝ := 118.54
def total_amount_spent_on_car_parts : ℝ := 224.87

-- Define the amount spent on new tires
def amount_spent_on_new_tires : ℝ := total_amount_spent_on_car_parts - amount_spent_on_speakers

-- The theorem we want to prove
theorem mike_spent_on_new_tires : amount_spent_on_new_tires = 106.33 :=
by
  -- the proof would go here
  sorry

end mike_spent_on_new_tires_l343_34315


namespace average_stamps_collected_per_day_l343_34390

open Nat

-- Define an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + d * (n - 1)

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Given conditions
def a := 10
def d := 10
def n := 7

-- Prove that the average number of stamps collected over 7 days is 40
theorem average_stamps_collected_per_day : 
  sum_arithmetic_sequence a d n / n = 40 := 
by
  sorry

end average_stamps_collected_per_day_l343_34390


namespace find_m_2n_3k_l343_34388

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_m_2n_3k (m n k : ℕ) (h1 : m + n = 2021) (h2 : is_prime (m - 3 * k)) (h3 : is_prime (n + k)) :
  m + 2 * n + 3 * k = 2025 ∨ m + 2 * n + 3 * k = 4040 := by
  sorry

end find_m_2n_3k_l343_34388


namespace derivative_y_l343_34366

noncomputable def y (x : ℝ) : ℝ :=
  (1 / 4) * Real.log ((x - 1) / (x + 1)) - (1 / 2) * Real.arctan x

theorem derivative_y (x : ℝ) : deriv y x = 1 / (x^4 - 1) :=
  sorry

end derivative_y_l343_34366


namespace teacher_total_score_l343_34303

variable (written_score : ℕ)
variable (interview_score : ℕ)
variable (weight_written : ℝ)
variable (weight_interview : ℝ)

theorem teacher_total_score :
  (written_score = 80) → (interview_score = 60) → (weight_written = 0.6) → (weight_interview = 0.4) →
  (written_score * weight_written + interview_score * weight_interview = 72) :=
by
  sorry

end teacher_total_score_l343_34303


namespace lesser_fraction_l343_34360

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 163 / 625 :=
by sorry

end lesser_fraction_l343_34360


namespace total_earnings_correct_l343_34312

-- Define the weekly earnings and the duration of the harvest.
def weekly_earnings : ℕ := 16
def harvest_duration : ℕ := 76

-- Theorems to state the problem requiring a proof.
theorem total_earnings_correct : (weekly_earnings * harvest_duration = 1216) := 
by
  sorry -- Proof is not required.

end total_earnings_correct_l343_34312


namespace gcd_8fact_11fact_9square_l343_34353

theorem gcd_8fact_11fact_9square : Nat.gcd (Nat.factorial 8) ((Nat.factorial 11) * 9^2) = 40320 := 
sorry

end gcd_8fact_11fact_9square_l343_34353


namespace integer_solutions_eq_400_l343_34373

theorem integer_solutions_eq_400 : 
  ∃ (s : Finset (ℤ × ℤ)), (∀ x y, (x, y) ∈ s ↔ |3 * x + 2 * y| + |2 * x + y| = 100) ∧ s.card = 400 :=
sorry

end integer_solutions_eq_400_l343_34373


namespace cost_of_article_l343_34344

theorem cost_of_article (C : ℝ) (G : ℝ)
    (h1 : G = 520 - C)
    (h2 : 1.08 * G = 580 - C) :
    C = 230 :=
by
    sorry

end cost_of_article_l343_34344


namespace homework_duration_reduction_l343_34361

theorem homework_duration_reduction (x : ℝ) (initial_duration final_duration : ℝ) (h_initial : initial_duration = 90) (h_final : final_duration = 60) : 
  90 * (1 - x)^2 = 60 :=
by
  sorry

end homework_duration_reduction_l343_34361


namespace two_crows_problem_l343_34391

def Bird := { P | P = "parrot" ∨ P = "crow"} -- Define possible bird species.

-- Define birds and their statements
def Adam_statement (Adam Carl : Bird) : Prop := Carl = Adam
def Bob_statement (Adam : Bird) : Prop := Adam = "crow"
def Carl_statement (Dave : Bird) : Prop := Dave = "crow"
def Dave_statement (Adam Bob Carl Dave: Bird) : Prop := 
  (if Adam = "parrot" then 1 else 0) + 
  (if Bob = "parrot" then 1 else 0) + 
  (if Carl = "parrot" then 1 else 0) + 
  (if Dave = "parrot" then 1 else 0) ≥ 3

-- The main proposition to prove
def main_statement : Prop :=
  ∃ (Adam Bob Carl Dave : Bird), 
    (Adam_statement Adam Carl) ∧ 
    (Bob_statement Adam) ∧ 
    (Carl_statement Dave) ∧ 
    (Dave_statement Adam Bob Carl Dave) ∧ 
    (if Adam = "crow" then 1 else 0) + 
    (if Bob = "crow" then 1 else 0) + 
    (if Carl = "crow" then 1 else 0) + 
    (if Dave = "crow" then 1 else 0) = 2

-- Proof statement to be filled
theorem two_crows_problem : main_statement :=
by {
  sorry
}

end two_crows_problem_l343_34391


namespace x_intercept_is_2_l343_34337

noncomputable def x_intercept_of_line : ℝ :=
  by
  sorry -- This is where the proof would go

theorem x_intercept_is_2 :
  (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → y = 0 → x = 2) :=
  by
  intro x y H_eq H_y0
  rw [H_y0] at H_eq
  simp at H_eq
  sorry -- This is where the proof would go

end x_intercept_is_2_l343_34337


namespace min_possible_value_l343_34316

theorem min_possible_value (a b : ℤ) (h : a > b) :
  (∃ x : ℚ, x = (2 * a + 3 * b) / (a - 2 * b) ∧ (x + 1 / x = (2 : ℚ))) :=
sorry

end min_possible_value_l343_34316


namespace Rachel_average_speed_l343_34333

noncomputable def total_distance : ℝ := 2 + 4 + 6

noncomputable def time_to_Alicia : ℝ := 2 / 3
noncomputable def time_to_Lisa : ℝ := 4 / 5
noncomputable def time_to_Nicholas : ℝ := 1 / 2

noncomputable def total_time : ℝ := (20 / 30) + (24 / 30) + (15 / 30)

noncomputable def average_speed : ℝ := total_distance / total_time

theorem Rachel_average_speed : average_speed = 360 / 59 :=
by
  sorry

end Rachel_average_speed_l343_34333


namespace sequence_2010_eq_4040099_l343_34322

def sequence_term (n : Nat) : Int :=
  if n % 2 = 0 then 
    (n^2 - 1 : Int) 
  else 
    -(n^2 - 1 : Int)

theorem sequence_2010_eq_4040099 : sequence_term 2010 = 4040099 := 
  by 
    sorry

end sequence_2010_eq_4040099_l343_34322


namespace fraction_decomposition_l343_34374

theorem fraction_decomposition :
  ∀ (A B : ℚ), (∀ x : ℚ, x ≠ -2 → x ≠ 4/3 → 
  (7 * x - 15) / ((3 * x - 4) * (x + 2)) = A / (x + 2) + B / (3 * x - 4)) →
  A = 29 / 10 ∧ B = -17 / 10 :=
by
  sorry

end fraction_decomposition_l343_34374


namespace find_length_of_second_movie_l343_34369

noncomputable def length_of_second_movie := 1.5

theorem find_length_of_second_movie
  (total_free_time : ℝ)
  (first_movie_duration : ℝ)
  (words_read : ℝ)
  (reading_rate : ℝ) : 
  first_movie_duration = 3.5 → 
  total_free_time = 8 → 
  words_read = 1800 → 
  reading_rate = 10 → 
  length_of_second_movie = 1.5 := 
by
  intros h1 h2 h3 h4
  -- Here should be the proof steps, which are abstracted away.
  sorry

end find_length_of_second_movie_l343_34369


namespace leaf_distance_after_11_gusts_l343_34356

def distance_traveled (gusts : ℕ) (swirls : ℕ) (forward_per_gust : ℕ) (backward_per_swirl : ℕ) : ℕ :=
  (gusts * forward_per_gust) - (swirls * backward_per_swirl)

theorem leaf_distance_after_11_gusts :
  ∀ (forward_per_gust backward_per_swirl : ℕ),
  forward_per_gust = 5 →
  backward_per_swirl = 2 →
  distance_traveled 11 11 forward_per_gust backward_per_swirl = 33 :=
by
  intros forward_per_gust backward_per_swirl hfg hbs
  rw [hfg, hbs]
  unfold distance_traveled
  sorry

end leaf_distance_after_11_gusts_l343_34356


namespace product_of_roots_l343_34326

theorem product_of_roots (p q r : ℝ) (hp : 3*p^3 - 9*p^2 + 5*p - 15 = 0) 
  (hq : 3*q^3 - 9*q^2 + 5*q - 15 = 0) (hr : 3*r^3 - 9*r^2 + 5*r - 15 = 0) :
  p * q * r = 5 :=
sorry

end product_of_roots_l343_34326


namespace part_I_part_II_l343_34393

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part_I (x : ℝ) : (∃ a : ℝ, a = 1 ∧ f x a < 1) ↔ x < (1/2) :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 6) ↔ (a = 5 ∨ a = -7) :=
sorry

end part_I_part_II_l343_34393


namespace cost_per_foot_l343_34327

theorem cost_per_foot (area : ℕ) (total_cost : ℕ) (side_length : ℕ) (perimeter : ℕ) (cost_per_foot : ℕ) :
  area = 289 → total_cost = 3944 → side_length = Nat.sqrt 289 → perimeter = 4 * 17 →
  cost_per_foot = total_cost / perimeter → cost_per_foot = 58 :=
by
  intros
  sorry

end cost_per_foot_l343_34327


namespace number_under_35_sampled_l343_34308

-- Define the conditions
def total_employees : ℕ := 500
def employees_under_35 : ℕ := 125
def employees_35_to_49 : ℕ := 280
def employees_over_50 : ℕ := 95
def sample_size : ℕ := 100

-- Define the theorem stating the desired result
theorem number_under_35_sampled : (employees_under_35 * sample_size / total_employees) = 25 :=
by
  sorry

end number_under_35_sampled_l343_34308
