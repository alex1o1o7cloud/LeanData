import Mathlib

namespace NUMINAMATH_GPT_x_pow_n_plus_inv_x_pow_n_l2362_236287

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end NUMINAMATH_GPT_x_pow_n_plus_inv_x_pow_n_l2362_236287


namespace NUMINAMATH_GPT_sum_of_first_three_cards_l2362_236269

theorem sum_of_first_three_cards :
  ∀ (G Y : ℕ → ℕ) (cards : ℕ → ℕ),
  (∀ n, G n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)) →
  (∀ n, Y n ∈ ({4, 5, 6, 7, 8} : Set ℕ)) →
  (∀ n, cards (2 * n) = G (cards n) → cards (2 * n + 1) = Y (cards n + 1)) →
  (∀ n, Y n = G (n + 1) ∨ ∃ k, Y n = k * G (n + 1)) →
  (cards 0 + cards 1 + cards 2 = 14) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_three_cards_l2362_236269


namespace NUMINAMATH_GPT_sum_of_digits_T_l2362_236239

-- Conditions:
def horse_lap_times := [1, 2, 3, 4, 5, 6, 7, 8]
def S := 840
def total_horses := 8
def min_horses_at_start := 4

-- Question:
def T := 12 -- Least time such that at least 4 horses meet

/-- Prove that the sum of the digits of T is 3 -/
theorem sum_of_digits_T : (1 + 2) = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_T_l2362_236239


namespace NUMINAMATH_GPT_correct_option_l2362_236206

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variable (h_cond : ∀ x : ℝ, f x > deriv f x)

theorem correct_option :
  e ^ 2016 * f (-2016) > f 0 ∧ f 2016 < e ^ 2016 * f 0 :=
sorry

end NUMINAMATH_GPT_correct_option_l2362_236206


namespace NUMINAMATH_GPT_sqrt_81_eq_9_l2362_236284

theorem sqrt_81_eq_9 : Real.sqrt 81 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_81_eq_9_l2362_236284


namespace NUMINAMATH_GPT_parabola_passes_through_fixed_point_l2362_236280

theorem parabola_passes_through_fixed_point:
  ∀ t : ℝ, ∃ x y : ℝ, (y = 4 * x^2 + 2 * t * x - 3 * t ∧ (x = 3 ∧ y = 36)) :=
by
  intro t
  use 3
  use 36
  sorry

end NUMINAMATH_GPT_parabola_passes_through_fixed_point_l2362_236280


namespace NUMINAMATH_GPT_distinct_solutions_difference_eq_sqrt29_l2362_236257

theorem distinct_solutions_difference_eq_sqrt29 :
  (∃ a b : ℝ, a > b ∧
    (∀ x : ℝ, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ 
      x = a ∨ x = b) ∧ 
    a - b = Real.sqrt 29) :=
sorry

end NUMINAMATH_GPT_distinct_solutions_difference_eq_sqrt29_l2362_236257


namespace NUMINAMATH_GPT_time_after_2023_minutes_l2362_236263

def start_time : Nat := 1 * 60 -- Start time is 1:00 a.m. in minutes from midnight, which is 60 minutes.
def elapsed_time : Nat := 2023 -- The elapsed time is 2023 minutes.

theorem time_after_2023_minutes : (start_time + elapsed_time) % 1440 = 643 := 
by
  -- 1440 represents the total minutes in a day (24 hours * 60 minutes).
  -- 643 represents the time 10:43 a.m. in minutes from midnight. This is obtained as 10 * 60 + 43 = 643.
  sorry

end NUMINAMATH_GPT_time_after_2023_minutes_l2362_236263


namespace NUMINAMATH_GPT_cannot_form_right_triangle_l2362_236264

theorem cannot_form_right_triangle : ¬∃ a b c : ℕ, a = 4 ∧ b = 6 ∧ c = 11 ∧ (a^2 + b^2 = c^2) :=
by
  sorry

end NUMINAMATH_GPT_cannot_form_right_triangle_l2362_236264


namespace NUMINAMATH_GPT_multiplication_problem_solution_l2362_236278

theorem multiplication_problem_solution (a b c : ℕ) 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1) 
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h3 : (a * 100 + b * 10 + b) * c = b * 1000 + c * 100 + b * 10 + 1) : 
  a = 5 ∧ b = 3 ∧ c = 7 := 
sorry

end NUMINAMATH_GPT_multiplication_problem_solution_l2362_236278


namespace NUMINAMATH_GPT_xiao_wang_scores_problem_l2362_236232

-- Defining the problem conditions and solution as a proof problem
theorem xiao_wang_scores_problem (x y : ℕ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
                                 (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) :
  (x + 2 = 10) ∧ (y - 1 = 88) :=
by 
  sorry

end NUMINAMATH_GPT_xiao_wang_scores_problem_l2362_236232


namespace NUMINAMATH_GPT_min_value_l2362_236276

variable (a b c : ℝ)

theorem min_value (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) : 
  (a - b) ^ 2 + (b - c) ^ 2 = 25 / 2 := 
sorry

end NUMINAMATH_GPT_min_value_l2362_236276


namespace NUMINAMATH_GPT_perimeter_of_irregular_pentagonal_picture_frame_l2362_236204

theorem perimeter_of_irregular_pentagonal_picture_frame 
  (base : ℕ) (left_side : ℕ) (right_side : ℕ) (top_left_diagonal_side : ℕ) (top_right_diagonal_side : ℕ)
  (h_base : base = 10) (h_left_side : left_side = 12) (h_right_side : right_side = 11)
  (h_top_left_diagonal_side : top_left_diagonal_side = 6) (h_top_right_diagonal_side : top_right_diagonal_side = 7) :
  base + left_side + right_side + top_left_diagonal_side + top_right_diagonal_side = 46 :=
by {
  sorry
}

end NUMINAMATH_GPT_perimeter_of_irregular_pentagonal_picture_frame_l2362_236204


namespace NUMINAMATH_GPT_min_xy_min_x_plus_y_l2362_236221

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x * y ≥ 9 :=
sorry

theorem min_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x + y ≥ 6 :=
sorry

end NUMINAMATH_GPT_min_xy_min_x_plus_y_l2362_236221


namespace NUMINAMATH_GPT_carol_invitations_l2362_236246

-- Definitions: each package has 3 invitations, Carol bought 2 packs, and Carol needs 3 extra invitations.
def invitations_per_pack : ℕ := 3
def packs_bought : ℕ := 2
def extra_invitations : ℕ := 3

-- Total number of invitations Carol will have
def total_invitations : ℕ := (packs_bought * invitations_per_pack) + extra_invitations

-- Statement to prove: Carol wants to invite 9 friends.
theorem carol_invitations : total_invitations = 9 := by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_carol_invitations_l2362_236246


namespace NUMINAMATH_GPT_find_C_l2362_236225

theorem find_C (A B C : ℕ) (h1 : (19 + A + B) % 3 = 0) (h2 : (15 + A + B + C) % 3 = 0) : C = 1 := by
  sorry

end NUMINAMATH_GPT_find_C_l2362_236225


namespace NUMINAMATH_GPT_solution_valid_l2362_236210

noncomputable def verify_solution (x : ℝ) : Prop :=
  (Real.arcsin (3 * x) + Real.arccos (2 * x) = Real.pi / 4) ∧
  (|2 * x| ≤ 1) ∧
  (|3 * x| ≤ 1)

theorem solution_valid (x : ℝ) :
  verify_solution x ↔ (x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨ x = -(1 / Real.sqrt (11 - 2 * Real.sqrt 2))) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_valid_l2362_236210


namespace NUMINAMATH_GPT_fraction_sum_identity_l2362_236283

theorem fraction_sum_identity (p q r : ℝ) (h₀ : p ≠ q) (h₁ : p ≠ r) (h₂ : q ≠ r) 
(h : p / (q - r) + q / (r - p) + r / (p - q) = 1) :
  p / (q - r)^2 + q / (r - p)^2 + r / (p - q)^2 = 1 / (q - r) + 1 / (r - p) + 1 / (p - q) - 1 := 
sorry

end NUMINAMATH_GPT_fraction_sum_identity_l2362_236283


namespace NUMINAMATH_GPT_largest_among_abc_l2362_236293

theorem largest_among_abc
  (x : ℝ) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (a : ℝ)
  (ha : a = 2 * Real.sqrt x )
  (b : ℝ)
  (hb : b = 1 + x)
  (c : ℝ)
  (hc : c = 1 / (1 - x)) 
  : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_largest_among_abc_l2362_236293


namespace NUMINAMATH_GPT_rook_placement_l2362_236271

theorem rook_placement : 
  let n := 8
  let k := 6
  let binom := Nat.choose
  binom 8 6 * binom 8 6 * Nat.factorial 6 = 564480 := by
    sorry

end NUMINAMATH_GPT_rook_placement_l2362_236271


namespace NUMINAMATH_GPT_triangle_BC_range_l2362_236290

open Real

variable {a C : ℝ} (A : ℝ) (ABC : Triangle A C)

/-- Proof problem statement -/
theorem triangle_BC_range (A C : ℝ) (h0 : 0 < A) (h1 : A < π) (c : ℝ) (h2 : c = sqrt 2) (h3 : a * cos C = c * sin A): 
  ∃ (BC : ℝ), sqrt 2 < BC ∧ BC < 2 :=
sorry

end NUMINAMATH_GPT_triangle_BC_range_l2362_236290


namespace NUMINAMATH_GPT_area_of_rectangle_l2362_236235

-- Define the problem conditions in Lean
def circle_radius := 7
def circle_diameter := 2 * circle_radius
def width_of_rectangle := circle_diameter
def length_to_width_ratio := 3
def length_of_rectangle := length_to_width_ratio * width_of_rectangle

-- Define the statement to be proved (area of the rectangle)
theorem area_of_rectangle : 
  (length_of_rectangle * width_of_rectangle) = 588 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2362_236235


namespace NUMINAMATH_GPT_fixed_point_for_all_parabolas_l2362_236238

theorem fixed_point_for_all_parabolas : ∃ (x y : ℝ), (∀ t : ℝ, y = 4 * x^2 + 2 * t * x - 3 * t) ∧ x = 1 ∧ y = 4 :=
by 
  sorry

end NUMINAMATH_GPT_fixed_point_for_all_parabolas_l2362_236238


namespace NUMINAMATH_GPT_find_d_l2362_236200

theorem find_d : ∃ d : ℝ, (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) ∧ d = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l2362_236200


namespace NUMINAMATH_GPT_vitamin_d_supplements_per_pack_l2362_236228

theorem vitamin_d_supplements_per_pack :
  ∃ (x : ℕ), (∀ (n m : ℕ), 7 * n = x * m → 119 <= 7 * n) ∧ (7 * n = 17 * m) :=
by
  -- definition of conditions
  let min_sold := 119
  let vitaminA_per_pack := 7
  -- let x be the number of Vitamin D supplements per pack
  -- the proof is yet to be completed
  sorry

end NUMINAMATH_GPT_vitamin_d_supplements_per_pack_l2362_236228


namespace NUMINAMATH_GPT_maximize_expression_l2362_236243

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
(x^2 + x * y + y^2) * (x^2 + x * z + z^2) * (y^2 + y * z + z^2)

theorem maximize_expression (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) : 
    max_value_expression x y z ≤ 27 :=
sorry

end NUMINAMATH_GPT_maximize_expression_l2362_236243


namespace NUMINAMATH_GPT_distance_p_ran_l2362_236282

variable (d t v : ℝ)
-- d: head start distance in meters
-- t: time in minutes
-- v: speed of q in meters per minute

theorem distance_p_ran (h1 : d = 0.3 * v * t) : 1.3 * v * t = 1.3 * v * t :=
by
  sorry

end NUMINAMATH_GPT_distance_p_ran_l2362_236282


namespace NUMINAMATH_GPT_shells_picked_in_morning_l2362_236201

-- Definitions based on conditions
def total_shells : ℕ := 616
def afternoon_shells : ℕ := 324

-- The goal is to prove that morning_shells = 292
theorem shells_picked_in_morning (morning_shells : ℕ) (h : total_shells = morning_shells + afternoon_shells) : morning_shells = 292 := 
by
  sorry

end NUMINAMATH_GPT_shells_picked_in_morning_l2362_236201


namespace NUMINAMATH_GPT_hours_worked_each_day_l2362_236207

-- Given conditions
def total_hours_worked : ℕ := 18
def number_of_days_worked : ℕ := 6

-- Statement to prove
theorem hours_worked_each_day : total_hours_worked / number_of_days_worked = 3 := by
  sorry

end NUMINAMATH_GPT_hours_worked_each_day_l2362_236207


namespace NUMINAMATH_GPT_least_subtracted_number_correct_l2362_236286

noncomputable def least_subtracted_number (n : ℕ) : ℕ :=
  n - 13

theorem least_subtracted_number_correct (n : ℕ) : 
  least_subtracted_number 997 = 997 - 13 ∧
  (least_subtracted_number 997 % 5 = 3) ∧
  (least_subtracted_number 997 % 9 = 3) ∧
  (least_subtracted_number 997 % 11 = 3) :=
by
  let x := 997 - 13
  have : x = 984 := rfl
  have h5 : x % 5 = 3 := by sorry
  have h9 : x % 9 = 3 := by sorry
  have h11 : x % 11 = 3 := by sorry
  exact ⟨rfl, h5, h9, h11⟩

end NUMINAMATH_GPT_least_subtracted_number_correct_l2362_236286


namespace NUMINAMATH_GPT_sufficient_condition_for_perpendicular_l2362_236268

variables {Plane : Type} {Line : Type} 
variables (α β γ : Plane) (m n : Line)

-- Definitions based on conditions
variables (perpendicular : Plane → Plane → Prop)
variables (perpendicular_line : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Conditions from option D
variable (h1 : perpendicular_line n α)
variable (h2 : perpendicular_line n β)
variable (h3 : perpendicular_line m α)

-- Statement to prove
theorem sufficient_condition_for_perpendicular (h1 : perpendicular_line n α)
  (h2 : perpendicular_line n β) (h3 : perpendicular_line m α) : 
  perpendicular_line m β := 
sorry

end NUMINAMATH_GPT_sufficient_condition_for_perpendicular_l2362_236268


namespace NUMINAMATH_GPT_arun_weight_upper_limit_l2362_236224

theorem arun_weight_upper_limit (weight : ℝ) (avg_weight : ℝ) 
  (arun_opinion : 66 < weight ∧ weight < 72) 
  (brother_opinion : 60 < weight ∧ weight < 70) 
  (average_condition : avg_weight = 68) : weight ≤ 70 :=
by
  sorry

end NUMINAMATH_GPT_arun_weight_upper_limit_l2362_236224


namespace NUMINAMATH_GPT_find_n_eq_l2362_236294

theorem find_n_eq : 
  let a := 2^4
  let b := 3^3
  ∃ (n : ℤ), a - 7 = b + n :=
by
  let a := 2^4
  let b := 3^3
  use -18
  sorry

end NUMINAMATH_GPT_find_n_eq_l2362_236294


namespace NUMINAMATH_GPT_solve_for_x_l2362_236214

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2362_236214


namespace NUMINAMATH_GPT_determine_parabola_equation_l2362_236229

-- Given conditions
variable (p : ℝ) (h_p : p > 0)
variable (x1 x2 : ℝ)
variable (AF BF : ℝ)
variable (h_AF : AF = x1 + p / 2)
variable (h_BF : BF = x2 + p / 2)
variable (h_AF_value : AF = 2)
variable (h_BF_value : BF = 3)

-- Prove the equation of the parabola
theorem determine_parabola_equation (h1 : x1 + x2 = 5 - p)
(h2 : x1 * x2 = p^2 / 4)
(h3 : AF * BF = 6) :
  y^2 = (24/5 : ℝ) * x := 
sorry

end NUMINAMATH_GPT_determine_parabola_equation_l2362_236229


namespace NUMINAMATH_GPT_trains_meet_at_10_am_l2362_236274

def distance (speed time : ℝ) : ℝ := speed * time

theorem trains_meet_at_10_am
  (distance_pq : ℝ)
  (speed_train_from_p : ℝ)
  (start_time_from_p : ℝ)
  (speed_train_from_q : ℝ)
  (start_time_from_q : ℝ)
  (meeting_time : ℝ) :
  distance_pq = 110 → 
  speed_train_from_p = 20 → 
  start_time_from_p = 7 → 
  speed_train_from_q = 25 → 
  start_time_from_q = 8 → 
  meeting_time = 10 :=
by
  sorry

end NUMINAMATH_GPT_trains_meet_at_10_am_l2362_236274


namespace NUMINAMATH_GPT_range_of_a_l2362_236256

noncomputable def domain_f (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + a ≥ 0
noncomputable def range_g (a : ℝ) : Prop := ∀ x : ℝ, x ≤ 2 → 2^x - a ∈ Set.Ioi (0 : ℝ)

theorem range_of_a (a : ℝ) : (domain_f a ∨ range_g a) ∧ ¬(domain_f a ∧ range_g a) → (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2362_236256


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l2362_236216

theorem line_through_point_with_equal_intercepts :
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    ((y = m * x + b ∧ ((x = 0 ∨ y = 0) → (x = y))) ∧ 
    (1 = m * 1 + b ∧ 1 + 1 = b)) → 
    (m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l2362_236216


namespace NUMINAMATH_GPT_rose_age_l2362_236298

variable {R M : ℝ}

theorem rose_age (h1 : R = (1/3) * M) (h2 : R + M = 100) : R = 25 :=
sorry

end NUMINAMATH_GPT_rose_age_l2362_236298


namespace NUMINAMATH_GPT_grace_wins_probability_l2362_236281

def probability_grace_wins : ℚ :=
  let total_possible_outcomes := 36
  let losing_combinations := 6
  let winning_combinations := total_possible_outcomes - losing_combinations
  winning_combinations / total_possible_outcomes

theorem grace_wins_probability :
    probability_grace_wins = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_grace_wins_probability_l2362_236281


namespace NUMINAMATH_GPT_simple_interest_rate_l2362_236241

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (h1 : T = 15) (h2 : SI = 3 * P) (h3 : SI = P * R * T / 100) : R = 20 :=
by 
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2362_236241


namespace NUMINAMATH_GPT_probability_of_draw_l2362_236223

-- Define probabilities
def P_A_wins : ℝ := 0.4
def P_A_not_loses : ℝ := 0.9

-- Theorem statement
theorem probability_of_draw : P_A_not_loses = P_A_wins + 0.5 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_probability_of_draw_l2362_236223


namespace NUMINAMATH_GPT_calculate_P_AB_l2362_236251

section Probability
-- Define the given probabilities
variables (P_B_given_A : ℚ) (P_A : ℚ)
-- Given conditions
def given_conditions := P_B_given_A = 3/10 ∧ P_A = 1/5

-- Prove that P(AB) = 3/50
theorem calculate_P_AB (h : given_conditions P_B_given_A P_A) : (P_A * P_B_given_A) = 3/50 :=
by
  rcases h with ⟨h1, h2⟩
  simp [h1, h2]
  -- Here we would include the steps leading to the conclusion; this part just states the theorem
  sorry

end Probability

end NUMINAMATH_GPT_calculate_P_AB_l2362_236251


namespace NUMINAMATH_GPT_problem_statement_l2362_236237

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2362_236237


namespace NUMINAMATH_GPT_angle_A_measure_in_triangle_l2362_236215

theorem angle_A_measure_in_triangle (A B C : ℝ) 
  (h1 : B = 15)
  (h2 : C = 3 * B) 
  (angle_sum : A + B + C = 180) :
  A = 120 :=
by
  -- We'll fill in the proof steps later
  sorry

end NUMINAMATH_GPT_angle_A_measure_in_triangle_l2362_236215


namespace NUMINAMATH_GPT_simplify_expression_l2362_236285

theorem simplify_expression :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90 - 18) * (90 + 18)) / ((120 - 9) * (120 + 9)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2362_236285


namespace NUMINAMATH_GPT_find_star_l2362_236222

-- Define the problem conditions and statement
theorem find_star (x : ℤ) (star : ℤ) (h1 : x = 5) (h2 : -3 * (star - 9) = 5 * x - 1) : star = 1 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_find_star_l2362_236222


namespace NUMINAMATH_GPT_cone_central_angle_l2362_236265

theorem cone_central_angle (l : ℝ) (α : ℝ) (h : (30 : ℝ) * π / 180 > 0) :
  α = π := 
sorry

end NUMINAMATH_GPT_cone_central_angle_l2362_236265


namespace NUMINAMATH_GPT_factorial_mod_10_l2362_236213

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end NUMINAMATH_GPT_factorial_mod_10_l2362_236213


namespace NUMINAMATH_GPT_eagles_score_l2362_236260

variables (F E : ℕ)

theorem eagles_score (h1 : F + E = 56) (h2 : F = E + 8) : E = 24 := 
sorry

end NUMINAMATH_GPT_eagles_score_l2362_236260


namespace NUMINAMATH_GPT_find_n_positive_integers_l2362_236299

theorem find_n_positive_integers :
  ∀ n : ℕ, 0 < n →
  (∃ k : ℕ, (n^2 + 11 * n - 4) * n! + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_positive_integers_l2362_236299


namespace NUMINAMATH_GPT_ratio_of_bubbles_l2362_236266

def bubbles_dawn_per_ounce : ℕ := 200000

def mixture_bubbles (bubbles_other_per_ounce : ℕ) : ℕ :=
  let half_ounce_dawn := bubbles_dawn_per_ounce / 2
  let half_ounce_other := bubbles_other_per_ounce / 2
  half_ounce_dawn + half_ounce_other

noncomputable def find_ratio (bubbles_other_per_ounce : ℕ) : ℚ :=
  (bubbles_other_per_ounce : ℚ) / bubbles_dawn_per_ounce

theorem ratio_of_bubbles
  (bubbles_other_per_ounce : ℕ)
  (h_mixture : mixture_bubbles bubbles_other_per_ounce = 150000) :
  find_ratio bubbles_other_per_ounce = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_bubbles_l2362_236266


namespace NUMINAMATH_GPT_troll_problem_l2362_236270

theorem troll_problem (T : ℕ) (h : 6 + T + T / 2 = 33) : 4 * 6 - T = 6 :=
by sorry

end NUMINAMATH_GPT_troll_problem_l2362_236270


namespace NUMINAMATH_GPT_A_union_B_l2362_236211

noncomputable def A : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - 2^x) ∧ x < 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2 ∧ x > 0}
noncomputable def union_set : Set ℝ := {x | x < 0 ∨ x > 0}

theorem A_union_B :
  A ∪ B = union_set :=
by
  sorry

end NUMINAMATH_GPT_A_union_B_l2362_236211


namespace NUMINAMATH_GPT_log_product_solution_l2362_236219

theorem log_product_solution (x : ℝ) (hx : 0 < x) : 
  (Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2 ↔ 
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end NUMINAMATH_GPT_log_product_solution_l2362_236219


namespace NUMINAMATH_GPT_square_perimeter_is_64_l2362_236209

-- Given conditions
variables (s : ℕ)
def is_square_divided_into_four_congruent_rectangles : Prop :=
  ∀ (r : ℕ), r = 4 → (∀ (p : ℕ), p = (5 * s) / 2 → p = 40)

-- Lean 4 statement for the proof problem
theorem square_perimeter_is_64 
  (h : is_square_divided_into_four_congruent_rectangles s) 
  (hs : (5 * s) / 2 = 40) : 
  4 * s = 64 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_is_64_l2362_236209


namespace NUMINAMATH_GPT_ratio_seniors_to_juniors_l2362_236261

variable (j s : ℕ)

-- Condition: \(\frac{3}{7}\) of the juniors participated is equal to \(\frac{6}{7}\) of the seniors participated
def participation_condition (j s : ℕ) : Prop :=
  3 * j = 6 * s

-- Theorem to be proved: the ratio of seniors to juniors is \( \frac{1}{2} \)
theorem ratio_seniors_to_juniors (j s : ℕ) (h : participation_condition j s) : s / j = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_ratio_seniors_to_juniors_l2362_236261


namespace NUMINAMATH_GPT_min_value_3x_plus_4y_l2362_236296

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_3x_plus_4y_l2362_236296


namespace NUMINAMATH_GPT_complex_exp_identity_l2362_236242

theorem complex_exp_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_exp_identity_l2362_236242


namespace NUMINAMATH_GPT_tan_of_trig_eq_l2362_236291

theorem tan_of_trig_eq (x : Real) (h : (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2) : Real.tan x = 4 / 3 :=
by sorry

end NUMINAMATH_GPT_tan_of_trig_eq_l2362_236291


namespace NUMINAMATH_GPT_payment_for_150_books_equal_payment_number_of_books_l2362_236233

/-- 
Xinhua Bookstore conditions:
- Both suppliers A and B price each book at 40 yuan. 
- Supplier A offers a 10% discount on all books.
- Supplier B offers a 20% discount on any books purchased exceeding 100 books.
-/

def price_per_book_supplier_A (n : ℕ) : ℝ := 40 * 0.9
def price_per_first_100_books_supplier_B : ℝ := 40
def price_per_excess_books_supplier_B (n : ℕ) : ℝ := 40 * 0.8

-- Prove that the payment amounts for 150 books from suppliers A and B are 5400 yuan and 5600 yuan respectively.
theorem payment_for_150_books :
  price_per_book_supplier_A 150 * 150 = 5400 ∧
  price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B 50 * (150 - 100) = 5600 :=
  sorry

-- Prove the equal payment equivalence theorem for supplier A and B.
theorem equal_payment_number_of_books (x : ℕ) :
  price_per_book_supplier_A x * x = price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B (x - 100) * (x - 100) → x = 200 :=
  sorry

end NUMINAMATH_GPT_payment_for_150_books_equal_payment_number_of_books_l2362_236233


namespace NUMINAMATH_GPT_mary_pizza_order_l2362_236277

theorem mary_pizza_order (p e r n : ℕ) (h1 : p = 8) (h2 : e = 7) (h3 : r = 9) :
  n = (r + e) / p → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_pizza_order_l2362_236277


namespace NUMINAMATH_GPT_combined_age_l2362_236295

-- Define the conditions as Lean assumptions
def avg_age_three_years_ago := 19
def number_of_original_members := 6
def number_of_years_passed := 3
def current_avg_age := 19

-- Calculate the total age three years ago
def total_age_three_years_ago := number_of_original_members * avg_age_three_years_ago 

-- Calculate the increase in total age over three years
def total_increase_in_age := number_of_original_members * number_of_years_passed 

-- Calculate the current total age of the original members
def current_total_age_of_original_members := total_age_three_years_ago + total_increase_in_age

-- Define the number of current total members and the current total age
def number_of_current_members := 8
def current_total_age := number_of_current_members * current_avg_age

-- Formally state the problem and proof
theorem combined_age : 
  (current_total_age - current_total_age_of_original_members = 20) := 
by
  sorry

end NUMINAMATH_GPT_combined_age_l2362_236295


namespace NUMINAMATH_GPT_john_can_fix_l2362_236273

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end NUMINAMATH_GPT_john_can_fix_l2362_236273


namespace NUMINAMATH_GPT_minimum_n_l2362_236245

noncomputable def a (n : ℕ) : ℕ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℕ := n - 6 + a n

noncomputable def S (n : ℕ) : ℕ := (n * (n - 11)) / 2 + (2 ^ n - 1) / 2

theorem minimum_n (n : ℕ) (hn : n ≥ 5) : S 5 > 0 := by
  sorry

end NUMINAMATH_GPT_minimum_n_l2362_236245


namespace NUMINAMATH_GPT_remainder_when_divided_l2362_236250

noncomputable def y : ℝ := 19.999999999999716
def quotient : ℝ := 76.4
def remainder : ℝ := 8

theorem remainder_when_divided (x : ℝ) (hx : x = y * 76 + y * 0.4) : x % y = 8 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l2362_236250


namespace NUMINAMATH_GPT_sum_consecutive_integers_product_1080_l2362_236212

theorem sum_consecutive_integers_product_1080 :
  ∃ n : ℕ, n * (n + 1) = 1080 ∧ n + (n + 1) = 65 :=
by
  sorry

end NUMINAMATH_GPT_sum_consecutive_integers_product_1080_l2362_236212


namespace NUMINAMATH_GPT_nine_questions_insufficient_l2362_236234

/--
We have 5 stones with distinct weights and we are allowed to ask nine questions of the form
"Is it true that A < B < C?". Prove that nine such questions are insufficient to always determine
the unique ordering of these stones.
-/
theorem nine_questions_insufficient (stones : Fin 5 → Nat) 
  (distinct_weights : ∀ i j : Fin 5, i ≠ j → stones i ≠ stones j) :
  ¬ (∃ f : { q : Fin 125 | q.1 ≤ 8 } → (Fin 5 → Fin 5 → Fin 5 → Bool),
    ∀ w1 w2 w3 w4 w5 : Fin 120,
      (f ⟨0, sorry⟩) = sorry  -- This line only represents the existence of 9 questions
      )
:=
sorry

end NUMINAMATH_GPT_nine_questions_insufficient_l2362_236234


namespace NUMINAMATH_GPT_average_book_width_correct_l2362_236253

noncomputable def average_book_width 
  (widths : List ℚ) (number_of_books : ℕ) : ℚ :=
(widths.sum) / number_of_books

theorem average_book_width_correct :
  average_book_width [5, 3/4, 1.5, 3, 7.25, 12] 6 = 59 / 12 := 
  by 
  sorry

end NUMINAMATH_GPT_average_book_width_correct_l2362_236253


namespace NUMINAMATH_GPT_monthly_compounding_greater_than_yearly_l2362_236244

open Nat Real

theorem monthly_compounding_greater_than_yearly : 
  1 + 3 / 100 < (1 + 3 / (12 * 100)) ^ 12 :=
by
  -- This is the proof we need to write.
  sorry

end NUMINAMATH_GPT_monthly_compounding_greater_than_yearly_l2362_236244


namespace NUMINAMATH_GPT_quadratic_other_x_intercept_l2362_236254

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → (a * x^2 + b * x + c) = -3)
  (h_intercept : ∀ x, x = 1 → (a * x^2 + b * x + c) = 0) : 
  ∃ x : ℝ, x = 9 ∧ (a * x^2 + b * x + c) = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_other_x_intercept_l2362_236254


namespace NUMINAMATH_GPT_simplest_radical_form_l2362_236252

def is_simplest_radical_form (r : ℝ) : Prop :=
  ∀ x : ℝ, x * x = r → ∃ y : ℝ, y * y ≠ r

theorem simplest_radical_form :
   (is_simplest_radical_form 6) :=
by
  sorry

end NUMINAMATH_GPT_simplest_radical_form_l2362_236252


namespace NUMINAMATH_GPT_sequence_general_formula_l2362_236297

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- because sequences in the solution are 1-indexed.
  | 1 => 2
  | k+2 => sequence (k+1) + 3 * (k+1)

theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : 
  sequence n = 2 + 3 * n * (n - 1) / 2 :=
by
  sorry

#eval sequence 1  -- should output 2
#eval sequence 2  -- should output 5
#eval sequence 3  -- should output 11
#eval sequence 4  -- should output 20
#eval sequence 5  -- should output 32
#eval sequence 6  -- should output 47

end NUMINAMATH_GPT_sequence_general_formula_l2362_236297


namespace NUMINAMATH_GPT_find_b_l2362_236262

theorem find_b (b : ℝ) (h1 : 0 < b) (h2 : b < 6)
  (h_ratio : ∃ (QRS QOP : ℝ), QRS / QOP = 4 / 25) : b = 6 :=
sorry

end NUMINAMATH_GPT_find_b_l2362_236262


namespace NUMINAMATH_GPT_pen_ratio_l2362_236217

theorem pen_ratio (R J D : ℕ) (pen_cost : ℚ) (total_spent : ℚ) (total_pens : ℕ) 
  (hR : R = 4)
  (hJ : J = 3 * R)
  (h_total_spent : total_spent = 33)
  (h_pen_cost : pen_cost = 1.5)
  (h_total_pens : total_pens = total_spent / pen_cost)
  (h_pens_expr : D + J + R = total_pens) :
  D / J = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_pen_ratio_l2362_236217


namespace NUMINAMATH_GPT_speed_of_second_part_l2362_236275

theorem speed_of_second_part
  (total_distance : ℝ)
  (distance_part1 : ℝ)
  (speed_part1 : ℝ)
  (average_speed : ℝ)
  (speed_part2 : ℝ) :
  total_distance = 70 →
  distance_part1 = 35 →
  speed_part1 = 48 →
  average_speed = 32 →
  speed_part2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_part_l2362_236275


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_nums_l2362_236288

theorem sum_of_three_consecutive_even_nums : 80 + 82 + 84 = 246 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_nums_l2362_236288


namespace NUMINAMATH_GPT_least_number_of_cans_l2362_236267

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 80) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  ∃ n, n = 37 := sorry

end NUMINAMATH_GPT_least_number_of_cans_l2362_236267


namespace NUMINAMATH_GPT_angle_measures_possible_l2362_236248

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end NUMINAMATH_GPT_angle_measures_possible_l2362_236248


namespace NUMINAMATH_GPT_correct_operation_l2362_236205

theorem correct_operation : ∃ (a : ℝ), (3 + Real.sqrt 2 ≠ 3 * Real.sqrt 2) ∧ 
  ((a ^ 2) ^ 3 ≠ a ^ 5) ∧
  (Real.sqrt ((-7 : ℝ) ^ 2) ≠ -7) ∧
  (4 * a ^ 2 * a = 4 * a ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2362_236205


namespace NUMINAMATH_GPT_Martha_points_l2362_236230

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end NUMINAMATH_GPT_Martha_points_l2362_236230


namespace NUMINAMATH_GPT_graphs_intersection_l2362_236227

theorem graphs_intersection 
  (a b c d x y : ℝ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) 
  (h1: y = ax^2 + bx + c) 
  (h2: y = ax^2 - bx + c + d) 
  : x = d / (2 * b) ∧ y = (a * d^2) / (4 * b^2) + d / 2 + c := 
sorry

end NUMINAMATH_GPT_graphs_intersection_l2362_236227


namespace NUMINAMATH_GPT_additional_money_required_l2362_236231

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end NUMINAMATH_GPT_additional_money_required_l2362_236231


namespace NUMINAMATH_GPT_sum_first_odd_numbers_not_prime_l2362_236247

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_first_odd_numbers_not_prime :
  ¬ (is_prime (1 + 3)) ∧
  ¬ (is_prime (1 + 3 + 5)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7 + 9)) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_odd_numbers_not_prime_l2362_236247


namespace NUMINAMATH_GPT_divisibility_by_100_l2362_236259

theorem divisibility_by_100 (n : ℕ) (k : ℕ) (h : n = 5 * k + 2) :
    100 ∣ (5^n + 12*n^2 + 12*n + 3) :=
sorry

end NUMINAMATH_GPT_divisibility_by_100_l2362_236259


namespace NUMINAMATH_GPT_sacks_per_day_l2362_236208

theorem sacks_per_day (total_sacks : ℕ) (total_days : ℕ) (harvest_per_day : ℕ) : 
  total_sacks = 56 → 
  total_days = 14 → 
  harvest_per_day = total_sacks / total_days → 
  harvest_per_day = 4 := 
by
  intros h_total_sacks h_total_days h_harvest_per_day
  rw [h_total_sacks, h_total_days] at h_harvest_per_day
  simp at h_harvest_per_day
  exact h_harvest_per_day

end NUMINAMATH_GPT_sacks_per_day_l2362_236208


namespace NUMINAMATH_GPT_expression_evaluation_l2362_236292

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2362_236292


namespace NUMINAMATH_GPT_sparrows_initial_count_l2362_236236

theorem sparrows_initial_count (a b c : ℕ) 
  (h1 : a + b + c = 24)
  (h2 : a - 4 = b + 1)
  (h3 : b + 1 = c + 3) : 
  a = 12 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_sparrows_initial_count_l2362_236236


namespace NUMINAMATH_GPT_remainder_of_h_x6_l2362_236272

def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

noncomputable def remainder_when_h_x6_divided_by_h (x : ℝ) : ℝ :=
  let hx := h x
  let hx6 := h (x^6)
  hx6 - 6 * hx

theorem remainder_of_h_x6 (x : ℝ) : remainder_when_h_x6_divided_by_h x = 6 :=
  sorry

end NUMINAMATH_GPT_remainder_of_h_x6_l2362_236272


namespace NUMINAMATH_GPT_range_of_m_l2362_236226

open Set

noncomputable def setA : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
noncomputable def setB (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = (1 / 3) * x + m}

theorem range_of_m {m : ℝ} (p q : Prop) :
  p ↔ ∃ x : ℝ, x ∈ setA →
  q ↔ ∃ x : ℝ, x ∈ setB m →
  ((p → q) ∧ ¬(q → p)) ↔ (1 / 3 < m ∧ m < 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2362_236226


namespace NUMINAMATH_GPT_attendees_not_from_companies_l2362_236220

theorem attendees_not_from_companies :
  let A := 30 
  let B := 2 * A
  let C := A + 10
  let D := C - 5
  let T := 185 
  T - (A + B + C + D) = 20 :=
by
  sorry

end NUMINAMATH_GPT_attendees_not_from_companies_l2362_236220


namespace NUMINAMATH_GPT_cylinder_height_decrease_l2362_236258

/--
Two right circular cylinders have the same volume. The radius of the second cylinder is 20% more than the radius
of the first. Prove that the height of the second cylinder is approximately 30.56% less than the first one's height.
-/
theorem cylinder_height_decrease (r1 h1 r2 h2 : ℝ) (hradius : r2 = 1.2 * r1) (hvolumes : π * r1^2 * h1 = π * r2^2 * h2) :
  h2 = 25 / 36 * h1 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_decrease_l2362_236258


namespace NUMINAMATH_GPT_parabola_vertex_l2362_236203

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ v : ℝ × ℝ, (v.1 = -1 ∧ v.2 = 4) ∧ ∀ (x : ℝ), (x^2 + 2*x + 5 = ((x + 1)^2 + 4))) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l2362_236203


namespace NUMINAMATH_GPT_geometric_progression_solution_l2362_236249

noncomputable def first_term_of_geometric_progression (b2 b6 : ℚ) (q : ℚ) : ℚ := 
  b2 / q
  
theorem geometric_progression_solution 
  (b2 b6 : ℚ)
  (h1 : b2 = 37 + 1/3)
  (h2 : b6 = 2 + 1/3) :
  ∃ a q : ℚ, a = 224 / 3 ∧ q = 1/2 ∧ b2 = a * q ∧ b6 = a * q^5 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l2362_236249


namespace NUMINAMATH_GPT_compute_ab_val_l2362_236279

variables (a b : ℝ)

theorem compute_ab_val
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
sorry

end NUMINAMATH_GPT_compute_ab_val_l2362_236279


namespace NUMINAMATH_GPT_squares_in_50th_ring_l2362_236289

noncomputable def number_of_squares_in_nth_ring (n : ℕ) : ℕ :=
  8 * n + 6

theorem squares_in_50th_ring : number_of_squares_in_nth_ring 50 = 406 := 
  by
  sorry

end NUMINAMATH_GPT_squares_in_50th_ring_l2362_236289


namespace NUMINAMATH_GPT_polynomial_root_l2362_236255

theorem polynomial_root (x0 : ℝ) (z : ℝ) 
  (h1 : x0^3 - x0 - 1 = 0) 
  (h2 : z = x0^2 + 3 * x0 + 1) : 
  z^3 - 5 * z^2 - 10 * z - 11 = 0 := 
sorry

end NUMINAMATH_GPT_polynomial_root_l2362_236255


namespace NUMINAMATH_GPT_middle_person_distance_l2362_236218

noncomputable def Al_position (t : ℝ) : ℝ := 6 * t
noncomputable def Bob_position (t : ℝ) : ℝ := 10 * t - 12
noncomputable def Cy_position (t : ℝ) : ℝ := 8 * t - 32

theorem middle_person_distance (t : ℝ) (h₁ : t ≥ 0) (h₂ : t ≥ 2) (h₃ : t ≥ 4) :
  (Al_position t = 52) ∨ (Bob_position t = 52) ∨ (Cy_position t = 52) :=
sorry

end NUMINAMATH_GPT_middle_person_distance_l2362_236218


namespace NUMINAMATH_GPT_general_term_of_geometric_sequence_l2362_236202

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem general_term_of_geometric_sequence
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4)
  (hq : is_geometric_sequence a q)
  (q := 1/2) :
  ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ * q^(n - 1) :=
sorry

end NUMINAMATH_GPT_general_term_of_geometric_sequence_l2362_236202


namespace NUMINAMATH_GPT_other_toys_cost_1000_l2362_236240

-- Definitions of the conditions
def cost_of_other_toys : ℕ := sorry
def cost_of_lightsaber (cost_of_other_toys : ℕ) : ℕ := 2 * cost_of_other_toys
def total_spent (cost_of_lightsaber cost_of_other_toys : ℕ) : ℕ := cost_of_lightsaber + cost_of_other_toys

-- The proof goal
theorem other_toys_cost_1000 (T : ℕ) (H1 : cost_of_lightsaber T = 2 * T) 
                            (H2 : total_spent (cost_of_lightsaber T) T = 3000) : T = 1000 := by
  sorry

end NUMINAMATH_GPT_other_toys_cost_1000_l2362_236240
