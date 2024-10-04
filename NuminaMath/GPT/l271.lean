import Mathlib

namespace geometric_mean_of_roots_l271_271755

theorem geometric_mean_of_roots (x : ℝ) (h : x^2 = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1)) : x = 1 ∨ x = -1 := 
by
  sorry

end geometric_mean_of_roots_l271_271755


namespace distance_MC_l271_271251

theorem distance_MC (MA MB MC : ℝ) (hMA : MA = 2) (hMB : MB = 3) (hABC : ∀ x y z : ℝ, x + y > z ∧ y + z > x ∧ z + x > y) :
  1 ≤ MC ∧ MC ≤ 5 := 
by 
  sorry

end distance_MC_l271_271251


namespace ab_c_sum_geq_expr_ab_c_sum_eq_iff_l271_271224

theorem ab_c_sum_geq_expr (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) :=
sorry

theorem ab_c_sum_eq_iff (a b c : ℝ) (α : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c :=
sorry

end ab_c_sum_geq_expr_ab_c_sum_eq_iff_l271_271224


namespace washer_dryer_cost_diff_l271_271797

-- conditions
def total_cost : ℕ := 1200
def washer_cost : ℕ := 710
def dryer_cost : ℕ := total_cost - washer_cost

-- proof statement
theorem washer_dryer_cost_diff : (washer_cost - dryer_cost) = 220 :=
by
  sorry

end washer_dryer_cost_diff_l271_271797


namespace abigail_money_loss_l271_271801

theorem abigail_money_loss {initial spent remaining lost : ℤ} 
  (h1 : initial = 11) 
  (h2 : spent = 2) 
  (h3 : remaining = 3) 
  (h4 : lost = initial - spent - remaining) : 
  lost = 6 := sorry

end abigail_money_loss_l271_271801


namespace surface_area_ratio_l271_271089

-- Defining conditions
variable (V_E V_J : ℝ) (A_E A_J : ℝ)
variable (volume_ratio : V_J = 30 * (Real.sqrt 30) * V_E)

-- Statement to prove
theorem surface_area_ratio (h : V_J = 30 * (Real.sqrt 30) * V_E) :
  A_J = 30 * A_E :=
by
  sorry

end surface_area_ratio_l271_271089


namespace sum_of_reciprocals_l271_271042

-- We state that for all non-zero real numbers x and y, if x + y = xy,
-- then the sum of their reciprocals equals 1.
theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1/x + 1/y = 1 :=
by
  sorry

end sum_of_reciprocals_l271_271042


namespace least_product_ab_l271_271376

theorem least_product_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (1 : ℚ) / a + 1 / (3 * b) = 1 / 6) : a * b ≥ 48 :=
by
  sorry

end least_product_ab_l271_271376


namespace solve_equation1_solve_equation2_l271_271896

theorem solve_equation1 (x : ℝ) : 4 - x = 3 * (2 - x) ↔ x = 1 :=
by sorry

theorem solve_equation2 (x : ℝ) : (2 * x - 1) / 2 - (2 * x + 5) / 3 = (6 * x - 1) / 6 - 1 ↔ x = -3 / 2 :=
by sorry

end solve_equation1_solve_equation2_l271_271896


namespace fisher_eligibility_l271_271624

theorem fisher_eligibility (A1 A2 S : ℕ) (hA1 : A1 = 84) (hS : S = 82) :
  (S ≥ 80) → (A1 + A2 ≥ 170) → (A2 = 86) :=
by
  sorry

end fisher_eligibility_l271_271624


namespace ratio_of_construction_paper_packs_l271_271680

-- Definitions for conditions
def marie_glue_sticks : Nat := 15
def marie_construction_paper : Nat := 30
def allison_total_items : Nat := 28
def allison_additional_glue_sticks : Nat := 8

-- Define the main quantity to prove
def allison_glue_sticks : Nat := marie_glue_sticks + allison_additional_glue_sticks
def allison_construction_paper : Nat := allison_total_items - allison_glue_sticks

-- The ratio should be of type Rat or Nat
theorem ratio_of_construction_paper_packs : (marie_construction_paper : Nat) / allison_construction_paper = 6 / 1 := by
  -- This is a placeholder for the actual proof
  sorry

end ratio_of_construction_paper_packs_l271_271680


namespace find_constant_a_l271_271404

theorem find_constant_a (a : ℚ) (S : ℕ → ℚ) (hS : ∀ n, S n = (a - 2) * 3^(n + 1) + 2) : a = 4 / 3 :=
by
  sorry

end find_constant_a_l271_271404


namespace total_distance_traveled_l271_271173

theorem total_distance_traveled :
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by
  let radius := 50
  let angle := 45
  let num_girls := 8
  let cos_135 := Real.cos (135 * Real.pi / 180)
  let distance_one_way := radius * Real.sqrt (2 * (1 - cos_135))
  let distance_one_girl := 4 * distance_one_way
  let total_distance := num_girls * distance_one_girl
  show total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2)
  sorry

end total_distance_traveled_l271_271173


namespace tiles_count_l271_271008

variable (c r : ℕ)

-- given: r = 10
def initial_rows_eq : Prop := r = 10

-- assertion: number of tiles is conserved after rearrangement
def tiles_conserved : Prop := c * r = (c - 2) * (r + 4)

-- desired: total number of tiles is 70
def total_tiles : Prop := c * r = 70

theorem tiles_count (h1 : initial_rows_eq r) (h2 : tiles_conserved c r) : total_tiles c r :=
by
  subst h1
  sorry

end tiles_count_l271_271008


namespace gcd_36_n_eq_12_l271_271609

theorem gcd_36_n_eq_12 (n : ℕ) (h1 : 80 ≤ n) (h2 : n ≤ 100) (h3 : Int.gcd 36 n = 12) : n = 84 ∨ n = 96 :=
by
  sorry

end gcd_36_n_eq_12_l271_271609


namespace no_solution_inequality_l271_271616

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end no_solution_inequality_l271_271616


namespace simplify_fraction_l271_271468

theorem simplify_fraction (c : ℚ) : (⟦5 + 6 * c⟧ / 9) + 3 = (⟦32 + 6 * c⟧ / 9) :=
sorry

end simplify_fraction_l271_271468


namespace number_of_workers_in_original_scenario_l271_271577

-- Definitions based on the given conditions
def original_days := 70
def alternative_days := 42
def alternative_workers := 50

-- The statement we want to prove
theorem number_of_workers_in_original_scenario : 
  (∃ (W : ℕ), W * original_days = alternative_workers * alternative_days) → ∃ (W : ℕ), W = 30 :=
by
  sorry

end number_of_workers_in_original_scenario_l271_271577


namespace x_squared_minus_y_squared_l271_271073

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l271_271073


namespace six_digit_numbers_with_at_least_two_zeros_l271_271850

noncomputable def num_six_digit_numbers_with_at_least_two_zeros : ℕ :=
  73314

theorem six_digit_numbers_with_at_least_two_zeros :
  ∃ n : ℕ, n = num_six_digit_numbers_with_at_least_two_zeros := by
  use 73314
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271850


namespace triangle_inequality_l271_271382

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end triangle_inequality_l271_271382


namespace abigail_money_loss_l271_271800

theorem abigail_money_loss
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  initial_amount - spent_amount - remaining_amount = 6 :=
by sorry

end abigail_money_loss_l271_271800


namespace rectangle_sides_l271_271062

theorem rectangle_sides (S d : ℝ) (a b : ℝ) : 
  a = Real.sqrt (S + d^2 / 4) + d / 2 ∧ 
  b = Real.sqrt (S + d^2 / 4) - d / 2 →
  S = a * b ∧ d = a - b :=
by
  -- definitions and conditions will be used here in the proofs
  sorry

end rectangle_sides_l271_271062


namespace number_of_lines_l271_271720

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines_l271_271720


namespace smallest_square_side_lengths_l271_271623

theorem smallest_square_side_lengths (x : ℕ) 
    (h₁ : ∀ (y : ℕ), y = x + 8) 
    (h₂ : ∀ (z : ℕ), z = 50) 
    (h₃ : ∀ (QS PS RT QT : ℕ), QS = 8 ∧ PS = x ∧ RT = 42 - x ∧ QT = x + 8 ∧ (8 / x) = ((42 - x) / (x + 8))) : 
  x = 2 ∨ x = 32 :=
by 
  sorry

end smallest_square_side_lengths_l271_271623


namespace total_distance_of_relay_race_l271_271215

theorem total_distance_of_relay_race 
    (fraction_siwon : ℝ := 3/10) 
    (fraction_dawon : ℝ := 4/10) 
    (distance_together : ℝ := 140) :
    (fraction_siwon + fraction_dawon) * 200 = distance_together :=
by
    sorry

end total_distance_of_relay_race_l271_271215


namespace longer_side_of_new_rectangle_l271_271249

theorem longer_side_of_new_rectangle {z : ℕ} (h : ∃x : ℕ, 9 * 16 = 144 ∧ x * z = 144 ∧ z ≠ 9 ∧ z ≠ 16) : z = 18 :=
sorry

end longer_side_of_new_rectangle_l271_271249


namespace smallest_t_for_given_roots_l271_271904

-- Define the polynomial with integer coefficients and specific roots
def poly (x : ℝ) : ℝ := (x + 3) * (x - 4) * (x - 6) * (2 * x - 1)

-- Define the main theorem statement
theorem smallest_t_for_given_roots :
  ∃ (t : ℤ), 0 < t ∧ t = 72 := by
  -- polynomial expansion skipped, proof will come here
  sorry

end smallest_t_for_given_roots_l271_271904


namespace min_value_of_expression_l271_271285

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (habc : a + b + c = 1) (expected_value : 3 * a + 2 * b = 2) :
  ∃ a b, (a + b + (1 - a - b) = 1) ∧ (3 * a + 2 * b = 2) ∧ (∀ a b, ∃ m, m = (2/a + 1/(3*b)) ∧ m = 16/3) :=
sorry

end min_value_of_expression_l271_271285


namespace maximize_S_n_l271_271513

variable (a_1 d : ℝ)
noncomputable def S (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d

theorem maximize_S_n {n : ℕ} (h1 : S 17 > 0) (h2 : S 18 < 0) : n = 9 := sorry

end maximize_S_n_l271_271513


namespace math_problem_l271_271024

-- Define the conditions
def a := -6
def b := 2
def c := 1 / 3
def d := 3 / 4
def e := 12
def f := -3

-- Statement of the problem
theorem math_problem : a / b + (c - d) * e + f^2 = 1 :=
by
  sorry

end math_problem_l271_271024


namespace same_face_probability_l271_271646

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l271_271646


namespace program_outputs_all_divisors_l271_271127

/--
  The function of the program is to output all divisors of \( n \), 
  given the initial conditions and operations in the program.
 -/
theorem program_outputs_all_divisors (n : ℕ) :
  ∀ I : ℕ, (1 ≤ I ∧ I ≤ n) → (∃ S : ℕ, (n % I = 0 ∧ S = I)) :=
by
  sorry

end program_outputs_all_divisors_l271_271127


namespace probability_of_Q_section_l271_271617

theorem probability_of_Q_section (sections : ℕ) (Q_sections : ℕ) (h1 : sections = 6) (h2 : Q_sections = 2) :
  Q_sections / sections = 2 / 6 :=
by
  -- solution proof is skipped
  sorry

end probability_of_Q_section_l271_271617


namespace contestant_wins_quiz_l271_271791

noncomputable def winProbability : ℚ :=
  let p_correct := (1 : ℚ) / 3
  let p_wrong := (2 : ℚ) / 3
  let binom := Nat.choose  -- binomial coefficient function
  ((binom 4 2 * (p_correct ^ 2) * (p_wrong ^ 2)) +
   (binom 4 3 * (p_correct ^ 3) * (p_wrong ^ 1)) +
   (binom 4 4 * (p_correct ^ 4) * (p_wrong ^ 0)))

theorem contestant_wins_quiz :
  winProbability = 11 / 27 :=
by
  simp [winProbability, Nat.choose]
  norm_num
  done

end contestant_wins_quiz_l271_271791


namespace largest_share_received_l271_271700

noncomputable def largest_share (total_profit : ℝ) (ratio : List ℝ) : ℝ :=
  let total_parts := ratio.foldl (· + ·) 0
  let part_value := total_profit / total_parts
  let max_part := ratio.foldl max 0
  max_part * part_value

theorem largest_share_received
  (total_profit : ℝ)
  (h_total_profit : total_profit = 42000)
  (ratio : List ℝ)
  (h_ratio : ratio = [2, 3, 4, 4, 6]) :
  largest_share total_profit ratio = 12600 :=
by
  sorry

end largest_share_received_l271_271700


namespace geometric_series_ratio_l271_271607

theorem geometric_series_ratio (a r : ℝ) 
  (h_series : ∑' n : ℕ, a * r^n = 18 )
  (h_odd_series : ∑' n : ℕ, a * r^(2*n + 1) = 8 ) : 
  r = 4 / 5 := 
sorry

end geometric_series_ratio_l271_271607


namespace work_days_l271_271471

theorem work_days (A B C : ℝ)
  (h1 : A + B = 1 / 20)
  (h2 : B + C = 1 / 30)
  (h3 : A + C = 1 / 30) :
  (1 / (A + B + C)) = 120 / 7 := 
by 
  sorry

end work_days_l271_271471


namespace coin_toss_probability_l271_271326

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l271_271326


namespace problem_equiv_proof_l271_271244

noncomputable def simplify_and_evaluate (a : ℝ) :=
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4))

theorem problem_equiv_proof :
  simplify_and_evaluate (Real.sqrt 2) = 1 := 
  sorry

end problem_equiv_proof_l271_271244


namespace contractor_fine_per_absent_day_l271_271952

theorem contractor_fine_per_absent_day :
  ∀ (total_days absent_days wage_per_day total_receipt fine_per_absent_day : ℝ),
    total_days = 30 →
    wage_per_day = 25 →
    absent_days = 4 →
    total_receipt = 620 →
    (total_days - absent_days) * wage_per_day - absent_days * fine_per_absent_day = total_receipt →
    fine_per_absent_day = 7.50 :=
by
  intros total_days absent_days wage_per_day total_receipt fine_per_absent_day
  intro h1 h2 h3 h4 h5
  sorry

end contractor_fine_per_absent_day_l271_271952


namespace real_solution_l271_271984

theorem real_solution (x : ℝ) (h : x ≠ 3) :
  (x * (x + 2)) / ((x - 3)^2) ≥ 8 ↔ (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 48) :=
by
  sorry

end real_solution_l271_271984


namespace original_price_of_article_l271_271004

theorem original_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (hSP : SP = 374) (hprofit : profit_percent = 0.10) : 
  CP = 340 ↔ SP = CP * (1 + profit_percent) :=
by 
  sorry

end original_price_of_article_l271_271004


namespace transformation_composition_l271_271408

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition_l271_271408


namespace probability_of_same_number_on_four_dice_l271_271632

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l271_271632


namespace fuse_length_must_be_80_l271_271517

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80_l271_271517


namespace integer_roots_condition_l271_271093

noncomputable def has_integer_roots (n : ℕ) : Prop :=
  ∃ x : ℤ, x * x - 4 * x + n = 0

theorem integer_roots_condition (n : ℕ) (h : n > 0) :
  has_integer_roots n ↔ n = 3 ∨ n = 4 :=
by 
  sorry

end integer_roots_condition_l271_271093


namespace find_multiplier_l271_271344

theorem find_multiplier (x y: ℤ) (h1: x = 127)
  (h2: x * y - 152 = 102): y = 2 :=
by
  sorry

end find_multiplier_l271_271344


namespace pentagon_angle_T_l271_271570

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l271_271570


namespace min_speed_A_l271_271626

theorem min_speed_A (V_B V_C V_A : ℕ) (d_AB d_AC wind extra_speed : ℕ) :
  V_B = 50 →
  V_C = 70 →
  d_AB = 40 →
  d_AC = 280 →
  wind = 5 →
  V_A > ((d_AB * (V_A + wind + extra_speed)) / (d_AC - d_AB) - wind) :=
sorry

end min_speed_A_l271_271626


namespace total_container_weight_is_correct_l271_271783

-- Definitions based on the conditions
def copper_bar_weight : ℕ := 90
def steel_bar_weight : ℕ := copper_bar_weight + 20
def tin_bar_weight : ℕ := steel_bar_weight / 2
def aluminum_bar_weight : ℕ := tin_bar_weight + 10

-- Number of bars in the container
def count_steel_bars : ℕ := 10
def count_tin_bars : ℕ := 15
def count_copper_bars : ℕ := 12
def count_aluminum_bars : ℕ := 8

-- Total weight of each type of bar
def total_steel_weight : ℕ := count_steel_bars * steel_bar_weight
def total_tin_weight : ℕ := count_tin_bars * tin_bar_weight
def total_copper_weight : ℕ := count_copper_bars * copper_bar_weight
def total_aluminum_weight : ℕ := count_aluminum_bars * aluminum_bar_weight

-- Total weight of the container
def total_container_weight : ℕ := total_steel_weight + total_tin_weight + total_copper_weight + total_aluminum_weight

-- Theorem to prove
theorem total_container_weight_is_correct : total_container_weight = 3525 := by
  sorry

end total_container_weight_is_correct_l271_271783


namespace correct_minutes_added_l271_271159

theorem correct_minutes_added :
  let time_lost_per_day : ℚ := 3 + 1/4
  let start_time := 1 -- in P.M. on March 15
  let end_time := 3 -- in P.M. on March 22
  let total_days := 7 -- days from March 15 to March 22
  let extra_hours := 2 -- hours on March 22 from 1 P.M. to 3 P.M.
  let total_hours := (total_days * 24) + extra_hours
  let time_lost_per_minute := time_lost_per_day / (24 * 60)
  let total_time_lost := total_hours * time_lost_per_minute
  let total_time_lost_minutes := total_time_lost * 60
  n = total_time_lost_minutes 
→ n = 221 / 96 := 
sorry

end correct_minutes_added_l271_271159


namespace probability_three_heads_in_eight_tosses_l271_271335

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l271_271335


namespace inverse_relation_a1600_inverse_relation_a400_l271_271594

variable (a b : ℝ)

def k := 400 

theorem inverse_relation_a1600 : (a * b = k) → (a = 1600) → (b = 0.25) :=
by
  sorry

theorem inverse_relation_a400 : (a * b = k) → (a = 400) → (b = 1) :=
by
  sorry

end inverse_relation_a1600_inverse_relation_a400_l271_271594


namespace largest_root_range_l271_271027

theorem largest_root_range (b_0 b_1 b_2 b_3 : ℝ)
  (hb_0 : |b_0| ≤ 3) (hb_1 : |b_1| ≤ 3) (hb_2 : |b_2| ≤ 3) (hb_3 : |b_3| ≤ 3) :
  ∃ s : ℝ, (∃ x : ℝ, x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 = 0 ∧ x > 0 ∧ s = x) ∧ 3 < s ∧ s < 4 := 
sorry

end largest_root_range_l271_271027


namespace correct_conclusions_l271_271126

theorem correct_conclusions :
  (∀ n : ℤ, n < -1 -> n < -1) ∧
  (¬ ∀ a : ℤ, abs (a + 2022) > 0) ∧
  (∀ a b : ℤ, a + b = 0 -> a * b < 0) ∧
  (∀ n : ℤ, abs n = n -> n ≥ 0) :=
sorry

end correct_conclusions_l271_271126


namespace geometric_progression_sum_l271_271764

theorem geometric_progression_sum (a q : ℝ) :
  (a + a * q^2 + a * q^4 = 63) →
  (a * q + a * q^3 = 30) →
  (a = 3 ∧ q = 2) ∨ (a = 48 ∧ q = 1 / 2) :=
by
  intro h1 h2
  sorry

end geometric_progression_sum_l271_271764


namespace dividend_calculation_l271_271420

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86) (hd : d = 52.7) (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 :=
by 
  sorry

end dividend_calculation_l271_271420


namespace same_number_on_four_dice_l271_271648

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l271_271648


namespace rhombus_diagonals_not_always_equal_l271_271935

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l271_271935


namespace wage_ratio_l271_271250

-- Define the conditions
variable (M W : ℝ) -- M stands for man's daily wage, W stands for woman's daily wage
variable (h1 : 40 * 10 * M = 14400) -- Condition 1: 40 men working for 10 days earn Rs. 14400
variable (h2 : 40 * 30 * W = 21600) -- Condition 2: 40 women working for 30 days earn Rs. 21600

-- The statement to prove
theorem wage_ratio (h1 : 40 * 10 * M = 14400) (h2 : 40 * 30 * W = 21600) : M / W = 2 := by
  sorry

end wage_ratio_l271_271250


namespace smallest_possible_value_of_EF_minus_DE_l271_271462

theorem smallest_possible_value_of_EF_minus_DE :
  ∃ (DE EF FD : ℤ), DE + EF + FD = 2010 ∧ DE < EF ∧ EF ≤ FD ∧ 1 = EF - DE ∧ DE > 0 ∧ EF > 0 ∧ FD > 0 ∧ 
  DE + EF > FD ∧ DE + FD > EF ∧ EF + FD > DE :=
by {
  sorry
}

end smallest_possible_value_of_EF_minus_DE_l271_271462


namespace count_students_in_meets_l271_271785

theorem count_students_in_meets (A B : Finset ℕ) (hA : A.card = 13) (hB : B.card = 12) (hAB : (A ∩ B).card = 6) :
  (A ∪ B).card = 19 :=
by
  sorry

end count_students_in_meets_l271_271785


namespace Kenny_running_to_basketball_ratio_l271_271415

theorem Kenny_running_to_basketball_ratio (basketball_hours trumpet_hours running_hours : ℕ) 
    (h1 : basketball_hours = 10)
    (h2 : trumpet_hours = 2 * running_hours)
    (h3 : trumpet_hours = 40) :
    running_hours = 20 ∧ basketball_hours = 10 ∧ (running_hours / basketball_hours = 2) :=
by
  sorry

end Kenny_running_to_basketball_ratio_l271_271415


namespace partition_ways_six_three_boxes_l271_271862

theorem partition_ways_six_three_boxes :
  ∃ (P : Finset (Multiset ℕ)), P.card = 6 ∧ ∀ m ∈ P, ∃ l, m = {a : ℕ | ∃ i j k, a = (i, j, k) ∧ i+j+k = 6 ∧ i≥0 ∧ j≥0 ∧ k≥0}.count {
   {6, 0, 0},
   {5, 1, 0},
   {4, 2, 0},
   {4, 1, 1},
   {3, 2, 1},
   {2, 2, 2}
} :=
by
  sorry

end partition_ways_six_three_boxes_l271_271862


namespace initial_fee_l271_271272

theorem initial_fee (total_bowls : ℤ) (lost_bowls : ℤ) (broken_bowls : ℤ) (safe_fee : ℤ)
  (loss_fee : ℤ) (total_payment : ℤ) (paid_amount : ℤ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  broken_bowls = 15 →
  safe_fee = 3 →
  loss_fee = 4 →
  total_payment = 1825 →
  paid_amount = total_payment - ((total_bowls - lost_bowls - broken_bowls) * safe_fee - (lost_bowls + broken_bowls) * loss_fee) →
  paid_amount = 100 :=
by
  intros _ _ _ _ _ _ _
  sorry

end initial_fee_l271_271272


namespace coin_toss_probability_l271_271321

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l271_271321


namespace balls_boxes_distribution_l271_271853

/-- There are 5 ways to put 6 indistinguishable balls into 3 indistinguishable boxes. -/
theorem balls_boxes_distribution : ∃ (S : Finset (Finset ℕ)), S.card = 5 ∧
  ∀ (s ∈ S), ∑ x in s, x = 6 ∧ s.card <= 3 :=
begin
  sorry,
end

end balls_boxes_distribution_l271_271853


namespace reduced_rates_apply_two_days_l271_271480

-- Definition of total hours in a week
def total_hours_in_week : ℕ := 7 * 24

-- Given fraction of the week with reduced rates
def reduced_rate_fraction : ℝ := 0.6428571428571429

-- Total hours covered by reduced rates
def reduced_rate_hours : ℝ := reduced_rate_fraction * total_hours_in_week

-- Hours per day with reduced rates on weekdays (8 p.m. to 8 a.m.)
def hours_weekday_night : ℕ := 12

-- Total weekdays with reduced rates
def total_weekdays : ℕ := 5

-- Total reduced rate hours on weekdays
def reduced_rate_hours_weekdays : ℕ := total_weekdays * hours_weekday_night

-- Remaining hours for 24 hour reduced rates
def remaining_reduced_rate_hours : ℝ := reduced_rate_hours - reduced_rate_hours_weekdays

-- Prove that the remaining reduced rate hours correspond to exactly 2 full days
theorem reduced_rates_apply_two_days : remaining_reduced_rate_hours = 2 * 24 := 
by
  sorry

end reduced_rates_apply_two_days_l271_271480


namespace muffins_in_morning_l271_271731

variable (M : ℕ)

-- Conditions
def goal : ℕ := 20
def afternoon_sales : ℕ := 4
def additional_needed : ℕ := 4
def morning_sales (M : ℕ) : ℕ := M

-- Proof statement (no need to prove here, just state it)
theorem muffins_in_morning :
  morning_sales M + afternoon_sales + additional_needed = goal → M = 12 :=
sorry

end muffins_in_morning_l271_271731


namespace square_paper_side_length_l271_271483

theorem square_paper_side_length :
  ∀ (edge_length : ℝ) (num_pieces : ℕ) (side_length : ℝ),
  edge_length = 12 ∧ num_pieces = 54 ∧ 6 * (edge_length ^ 2) = num_pieces * (side_length ^ 2)
  → side_length = 4 :=
by
  intros edge_length num_pieces side_length h
  sorry

end square_paper_side_length_l271_271483


namespace petals_vs_wings_and_unvisited_leaves_l271_271767

def flowers_petals_leaves := 5
def petals_per_flower := 2
def bees_wings := 3
def wings_per_bee := 4
def leaves_per_flower := 3
def visits_per_bee := 2
def total_flowers := flowers_petals_leaves
def total_bees := bees_wings

def total_petals : ℕ := total_flowers * petals_per_flower
def total_wings : ℕ := total_bees * wings_per_bee
def more_wings_than_petals := total_wings - total_petals

def total_leaves : ℕ := total_flowers * leaves_per_flower
def total_visits : ℕ := total_bees * visits_per_bee
def leaves_per_visit := leaves_per_flower
def visited_leaves : ℕ := min total_leaves (total_visits * leaves_per_visit)
def unvisited_leaves : ℕ := total_leaves - visited_leaves

theorem petals_vs_wings_and_unvisited_leaves :
  more_wings_than_petals = 2 ∧ unvisited_leaves = 0 :=
by
  sorry

end petals_vs_wings_and_unvisited_leaves_l271_271767


namespace area_ratio_eq_two_l271_271786

/-- 
  Given a unit square, let circle B be the inscribed circle and circle A be the circumscribed circle.
  Prove the ratio of the area of circle A to the area of circle B is 2.
--/
theorem area_ratio_eq_two (r_B r_A : ℝ) (hB : r_B = 1 / 2) (hA : r_A = Real.sqrt 2 / 2):
  (π * r_A ^ 2) / (π * r_B ^ 2) = 2 := by
  sorry

end area_ratio_eq_two_l271_271786


namespace box_width_l271_271155

theorem box_width (h : ℝ) (d : ℝ) (l : ℝ) (w : ℝ) 
  (h_eq_8 : h = 8)
  (l_eq_2h : l = 2 * h)
  (d_eq_20 : d = 20) :
  w = 4 * Real.sqrt 5 :=
by
  sorry

end box_width_l271_271155


namespace cos_585_eq_neg_sqrt2_div_2_l271_271166

theorem cos_585_eq_neg_sqrt2_div_2 :
  Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_585_eq_neg_sqrt2_div_2_l271_271166


namespace probability_odd_sum_l271_271580

def P := {1, 2, 3}
def Q := {1, 2, 4}
def R := {1, 3, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1

def probability_sum_odd : ℚ :=
  let outcomes := Prod.prod P Q R
  let odd_sum := {o | is_odd (o.1 + o.2 + o.3)}
  (outcomes ∩ odd_sum).card.to_rat / outcomes.card.to_rat

theorem probability_odd_sum :
  probability_sum_odd = 4 / 9 :=
sorry

end probability_odd_sum_l271_271580


namespace probability_of_three_heads_in_eight_tosses_l271_271317

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l271_271317


namespace solve_for_y_l271_271106

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l271_271106


namespace dice_same_number_probability_l271_271637

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l271_271637


namespace prove_q_l271_271873

-- Assume the conditions
variable (p q : Prop)
variable (hpq : p ∨ q) -- "p or q" is true
variable (hnp : ¬p)    -- "not p" is true

-- The theorem to prove q is true
theorem prove_q : q :=
by {
  sorry
}

end prove_q_l271_271873


namespace M_eq_N_l271_271613

def M : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def N : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l271_271613


namespace find_number_2010_sum_ways_l271_271817

noncomputable def count_sums (n : ℕ) :=
  (Finset.range (n + 1)).card

theorem find_number_2010_sum_ways :
  count_sums 2010 = 2010 :=
sorry

end find_number_2010_sum_ways_l271_271817


namespace ratio_black_white_l271_271445

-- Definitions of the parameters
variables (B W : ℕ)
variables (h1 : B + W = 200)
variables (h2 : 30 * B + 25 * W = 5500)

theorem ratio_black_white (B W : ℕ) (h1 : B + W = 200) (h2 : 30 * B + 25 * W = 5500) :
  B = W :=
by
  -- Proof omitted
  sorry

end ratio_black_white_l271_271445


namespace diagonals_of_octagon_l271_271395

theorem diagonals_of_octagon : 
  let n := 8 in 
  let total_line_segments := (n * (n - 1)) / 2 in
  let sides := n in
  let diagonals := total_line_segments - sides in
  diagonals = 20 := 
  by 
    let n := 8
    let total_line_segments := (n * (n - 1)) / 2
    let sides := n
    let diagonals := total_line_segments - sides
    have h : diagonals = 20 := sorry
    exact h

end diagonals_of_octagon_l271_271395


namespace decreasing_power_function_l271_271386

variable (m : ℝ)

theorem decreasing_power_function (h₁ : m^2 - 3 = 1) (h₂ : m^2 + m - 3 < 0) : m = -2 :=
  sorry

end decreasing_power_function_l271_271386


namespace circle_radius_five_eq_neg_eight_l271_271180

theorem circle_radius_five_eq_neg_eight (c : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ∧ (x + 4)^2 + (y + 1)^2 = 25) → c = -8 :=
by
  sorry

end circle_radius_five_eq_neg_eight_l271_271180


namespace download_time_correct_l271_271228

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l271_271228


namespace harvest_season_duration_l271_271421

theorem harvest_season_duration (weekly_rent : ℕ) (total_rent_paid : ℕ) : 
    (weekly_rent = 388) →
    (total_rent_paid = 527292) →
    (total_rent_paid / weekly_rent = 1360) :=
by
  intros h1 h2
  sorry

end harvest_season_duration_l271_271421


namespace minimum_sum_of_natural_numbers_with_lcm_2012_l271_271758

/-- 
Prove that the minimum sum of seven natural numbers whose least common multiple is 2012 is 512.
-/

theorem minimum_sum_of_natural_numbers_with_lcm_2012 : 
  ∃ (a b c d e f g : ℕ), Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a b) c) d) e) f) g = 2012 ∧ (a + b + c + d + e + f + g) = 512 :=
sorry

end minimum_sum_of_natural_numbers_with_lcm_2012_l271_271758


namespace part1_part2_l271_271811

namespace MathProofProblem

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part1 (x : ℝ) : f 2 * x ≤ f (x + 1) ↔ 0 ≤ x ∧ x ≤ 1 := 
by
  sorry

theorem part2 (a b : ℝ) (h₀ : a + b = 2) : f (a ^ 2) + f (b ^ 2) = 2 :=
by
  sorry

end MathProofProblem

end part1_part2_l271_271811


namespace same_face_probability_l271_271644

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l271_271644


namespace attendees_on_monday_is_10_l271_271442

-- Define the given conditions
def attendees_tuesday : ℕ := 15
def attendees_wed_thru_fri : ℕ := 10
def days_wed_thru_fri : ℕ := 3
def average_attendance : ℕ := 11
def total_days : ℕ := 5

-- Define the number of people who attended class on Monday
def attendees_tuesday_to_friday : ℕ := attendees_tuesday + attendees_wed_thru_fri * days_wed_thru_fri
def total_attendance : ℕ := average_attendance * total_days
def attendees_monday : ℕ := total_attendance - attendees_tuesday_to_friday

-- State the theorem
theorem attendees_on_monday_is_10 : attendees_monday = 10 :=
by
  -- Proof omitted
  sorry

end attendees_on_monday_is_10_l271_271442


namespace is_divisible_by_7_l271_271757

theorem is_divisible_by_7 : ∃ k : ℕ, 42 = 7 * k := by
  sorry

end is_divisible_by_7_l271_271757


namespace smallest_possible_intersections_l271_271481

theorem smallest_possible_intersections (n : ℕ) (hn : n = 2000) :
  ∃ N : ℕ, N ≥ 3997 :=
by
  sorry

end smallest_possible_intersections_l271_271481


namespace no_ordered_triples_l271_271692

noncomputable def no_solution (x y z : ℝ) : Prop :=
  x^2 - 3 * x * y + 2 * y^2 - z^2 = 31 ∧
  -x^2 + 6 * y * z + 2 * z^2 = 44 ∧
  x^2 + x * y + 8 * z^2 = 100

theorem no_ordered_triples : ¬ ∃ (x y z : ℝ), no_solution x y z := 
by 
  sorry

end no_ordered_triples_l271_271692


namespace prob_three_heads_in_eight_tosses_l271_271308

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l271_271308


namespace ratio_of_lengths_l271_271782

theorem ratio_of_lengths (total_length short_length : ℕ)
  (h1 : total_length = 35)
  (h2 : short_length = 10) :
  short_length / (total_length - short_length) = 2 / 5 := by
  -- Proof skipped
  sorry

end ratio_of_lengths_l271_271782


namespace power_function_increasing_l271_271905

theorem power_function_increasing (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (m^2 - 2*m - 2) * x^(-4*m - 2) > 0) ↔ m = -1 :=
by sorry

end power_function_increasing_l271_271905


namespace intersection_A_compB_l271_271067

def setA : Set ℤ := {x | (abs (x - 1) < 3)}
def setB : Set ℝ := {x | x^2 + 2 * x - 3 ≥ 0}
def setCompB : Set ℝ := {x | ¬(x^2 + 2 * x - 3 ≥ 0)}

theorem intersection_A_compB :
  { x : ℤ | x ∈ setA ∧ (x:ℝ) ∈ setCompB } = {-1, 0} :=
sorry

end intersection_A_compB_l271_271067


namespace gcm_of_9_and_15_less_than_120_eq_90_l271_271141

theorem gcm_of_9_and_15_less_than_120_eq_90 
  (lcm_9_15 : Nat := Nat.lcm 9 15)
  (multiples : List Nat := List.range (120 / lcm_9_15) |> List.map (λ n => n * lcm_9_15)) : 
  lcm_9_15 = 45 ∧ multiples.max = some 90 := by
sorry

end gcm_of_9_and_15_less_than_120_eq_90_l271_271141


namespace problem1_problem2_l271_271779

-- Problem (1)
theorem problem1 : (Real.sqrt 12 + (-1 / 3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1) :=
  sorry

-- Problem (2)
theorem problem2 (a : Real) (h : a ≠ 2) :
  (2 * a / (a^2 - 4) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2)) :=
  sorry

end problem1_problem2_l271_271779


namespace Mark_time_spent_l271_271227

theorem Mark_time_spent :
  let parking_time := 5
  let walking_time := 3
  let long_wait_time := 30
  let short_wait_time := 10
  let long_wait_days := 2
  let short_wait_days := 3
  let work_days := 5
  (parking_time + walking_time) * work_days + 
    long_wait_time * long_wait_days + 
    short_wait_time * short_wait_days = 130 :=
by
  sorry

end Mark_time_spent_l271_271227


namespace committee_selection_l271_271370

-- Definitions corresponding to the conditions:
-- Representation of Jiǎ, Yǐ, Bǐng, and Dīng as A, B, C, and D respectively.
-- Total number of students is 9, number of students to select is 5, 
-- with given constraints on selection.

variable (A B C D : Type) -- Representing the students A, B, C, D
variable (students : Finset (Type)) -- Representing the set of all students
variable (nine_students : students.card = 9) -- There are 9 students in total
variable (committee_size : ℕ := 5) -- Committee size is 5 students
variable (exclude_CD : ∀ selection : Finset students, C ∈ selection ∧ D ∈ selection → False)
variable (in_or_out_AB : ∀ selection : Finset students, A ∈ selection ↔ B ∈ selection)

-- The goal is to find the number of valid ways to form the committee which is 41.
theorem committee_selection : Finset.card {s : Finset students | s.card = committee_size ∧
  (A ∈ s ↔ B ∈ s) ∧ (C ∈ s ∧ D ∈ s → False)} = 41 := 
  sorry

end committee_selection_l271_271370


namespace pens_given_away_l271_271591

theorem pens_given_away (initial_pens : ℕ) (pens_left : ℕ) (n : ℕ) (h1 : initial_pens = 56) (h2 : pens_left = 34) (h3 : n = initial_pens - pens_left) : n = 22 := by
  -- The proof is omitted
  sorry

end pens_given_away_l271_271591


namespace ratio_of_customers_third_week_l271_271088

def ratio_of_customers (c1 c3 : ℕ) (s k t : ℕ) : Prop := s = 500 ∧ k = 50 ∧ t = 760 ∧ c1 = 35 ∧ c3 = 105 ∧ (t - s - k) - (35 + 70) = c1 ∧ c3 = 105 ∧ (c3 / c1 = 3)

theorem ratio_of_customers_third_week (c1 c3 : ℕ) (s k t : ℕ)
  (h1 : s = 500)
  (h2 : k = 50)
  (h3 : t = 760)
  (h4 : c1 = 35)
  (h5 : c3 = 105)
  (h6 : (t - s - k) - (35 + 70) = c1)
  (h7 : c3 = 105) :
  (c3 / c1) = 3 :=
  sorry

end ratio_of_customers_third_week_l271_271088


namespace find_m_l271_271558

-- Define vectors a and b
def a (m : ℝ) : ℝ × ℝ := (2, -m)
def b : ℝ × ℝ := (1, 3)

-- Define the condition for perpendicular vectors
def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- State the problem
theorem find_m (m : ℝ) (h : is_perpendicular (a m + b) b) : m = 4 :=
sorry -- proof omitted

end find_m_l271_271558


namespace min_inverse_ab_l271_271829

theorem min_inverse_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) : 
  ∃ (m : ℝ), (m = 2 / 9) ∧ (∀ (a b : ℝ), a > 0 → b > 0 → a + 2 * b = 6 → 1/(a * b) ≥ m) :=
by
  sorry

end min_inverse_ab_l271_271829


namespace remaining_paint_needed_l271_271703

-- Define the conditions
def total_paint_needed : ℕ := 70
def paint_bought : ℕ := 23
def paint_already_have : ℕ := 36

-- Lean theorem statement
theorem remaining_paint_needed : (total_paint_needed - (paint_already_have + paint_bought)) = 11 := by
  sorry

end remaining_paint_needed_l271_271703


namespace impossible_to_color_25_cells_l271_271086

theorem impossible_to_color_25_cells :
  ¬ ∃ (n : ℕ) (n_k : ℕ → ℕ), n = 25 ∧ (∀ k, k > 0 → k < 5 → (k % 2 = 1 → ∃ c : ℕ, n_k c = k)) :=
by
  sorry

end impossible_to_color_25_cells_l271_271086


namespace domain_of_sqrt_function_l271_271928

theorem domain_of_sqrt_function : {x : ℝ | 0 ≤ x ∧ x ≤ 1} = {x : ℝ | 1 - x ≥ 0 ∧ x - Real.sqrt (1 - x) ≥ 0} :=
by
  sorry

end domain_of_sqrt_function_l271_271928


namespace prob_three_heads_in_eight_tosses_l271_271309

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l271_271309


namespace scientific_notation_1300000_l271_271281

theorem scientific_notation_1300000 :
  1300000 = 1.3 * 10^6 :=
sorry

end scientific_notation_1300000_l271_271281


namespace tan_double_angle_l271_271993

theorem tan_double_angle (α : ℝ) (h1 : Real.cos (Real.pi - α) = 4 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan (2 * α) = 24 / 7 := 
sorry

end tan_double_angle_l271_271993


namespace equation_of_C_max_slope_OQ_l271_271841

-- Condition 1: Given the parabola with parameter p
def parabola_C (p : ℝ) (h : p > 0) : (ℝ × ℝ) → Prop :=
  λ (x y : ℝ), y^2 = 2 * p * x

-- Condition 2: Distance from the focus F to the directrix being 2
def distance_F_directrix_eq_two (p : ℝ) : Prop :=
  p = 2

-- Question 1: Prove that the equation of C is y^2 = 4x
theorem equation_of_C (p : ℝ) (h : p > 0) (hp : p = 2) : 
  ∀ (x y : ℝ), parabola_C p h (x, y) ↔ y^2 = 4 * x :=
by
  intros
  rw [hp]
  unfold parabola_C
  sorry

-- Point Q satisfies PQ = 9 * QF
def PQ_eq_9_QF (P Q F : ℝ × ℝ) : Prop :=
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let QF := (F.1 - Q.1, F.2 - Q.2)
  (PQ.1 = 9 * QF.1) ∧ (PQ.2 = 9 * QF.2)

-- Question 2: Prove the maximum value of the slope of line OQ is 1/3
theorem max_slope_OQ (p : ℝ) (h : p > 0) (hp : p = 2) (O Q : ℝ × ℝ) (F : ℝ × ℝ)
  (P : ℝ × ℝ) (hP : parabola_C p h P) (hQ : PQ_eq_9_QF P Q F) : 
  ∃ Kmax : ℝ, Kmax = 1 / 3 :=
by
  sorry

end equation_of_C_max_slope_OQ_l271_271841


namespace mary_daily_tasks_l271_271424

theorem mary_daily_tasks :
  ∃ (x y : ℕ), (x + y = 15) ∧ (4 * x + 7 * y = 85) ∧ (y = 8) :=
by
  sorry

end mary_daily_tasks_l271_271424


namespace pizza_diameter_increase_l271_271019

theorem pizza_diameter_increase :
  ∀ (d D : ℝ), 
    (D / d)^2 = 1.96 → D = 1.4 * d := by
  sorry

end pizza_diameter_increase_l271_271019


namespace solve_ordered_pair_l271_271178

theorem solve_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x^2 - y = (x - 2) + (y - 2)) :
  (x = -5 ∧ y = 12) ∨ (x = 2 ∧ y = 5) :=
  sorry

end solve_ordered_pair_l271_271178


namespace stairs_left_to_climb_l271_271345

def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

theorem stairs_left_to_climb : total_stairs - climbed_stairs = 22 := by
  sorry

end stairs_left_to_climb_l271_271345


namespace average_mark_first_class_l271_271667

theorem average_mark_first_class (A : ℝ)
  (class1_students class2_students : ℝ)
  (avg2 combined_avg total_students total_marks_combined : ℝ)
  (h1 : class1_students = 22)
  (h2 : class2_students = 28)
  (h3 : avg2 = 60)
  (h4 : combined_avg = 51.2)
  (h5 : total_students = class1_students + class2_students)
  (h6 : total_marks_combined = total_students * combined_avg)
  (h7 : 22 * A + 28 * avg2 = total_marks_combined) :
  A = 40 :=
by
  sorry

end average_mark_first_class_l271_271667


namespace rationalize_denominator_l271_271101

theorem rationalize_denominator :
  (7 / (Real.sqrt 175 - Real.sqrt 75)) = (7 * (Real.sqrt 7 + Real.sqrt 3) / 20) :=
by
  have h1 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 75 = 5 * Real.sqrt 3 := sorry
  sorry

end rationalize_denominator_l271_271101


namespace single_elimination_games_l271_271881

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l271_271881


namespace probability_of_exactly_three_heads_l271_271294

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l271_271294


namespace inequality_always_holds_l271_271380

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
by 
  sorry

end inequality_always_holds_l271_271380


namespace law_firm_associates_l271_271880

def percentage (total: ℕ) (part: ℕ): ℕ := part * 100 / total

theorem law_firm_associates (total: ℕ) (second_year: ℕ) (first_year: ℕ) (more_than_two_years: ℕ):
  percentage total more_than_two_years = 50 →
  percentage total second_year = 25 →
  first_year = more_than_two_years - second_year →
  percentage total first_year = 25 →
  percentage total (total - first_year) = 75 :=
by
  intros h1 h2 h3 h4
  sorry

end law_firm_associates_l271_271880


namespace minimum_adjacent_white_pairs_l271_271541

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l271_271541


namespace prob_bob_before_12_45_given_alice_after_bob_l271_271346

theorem prob_bob_before_12_45_given_alice_after_bob :
  let total_area := 60 * 60 / 2 in
  let interested_area := 45 * 45 / 2 in
  (interested_area / total_area = 0.5625) :=
by
  let total_area := 60 * 60 / 2
  let interested_area := 45 * 45 / 2
  have h1 : interested_area / total_area = 0.5625 := by norm_num
  exact h1

end prob_bob_before_12_45_given_alice_after_bob_l271_271346


namespace worker_savings_multiple_l271_271966

theorem worker_savings_multiple 
  (P : ℝ)
  (P_gt_zero : P > 0)
  (save_fraction : ℝ := 1/3)
  (not_saved_fraction : ℝ := 2/3)
  (total_saved : ℝ := 12 * (save_fraction * P)) :
  ∃ multiple : ℝ, total_saved = multiple * (not_saved_fraction * P) ∧ multiple = 6 := 
by 
  sorry

end worker_savings_multiple_l271_271966


namespace complex_number_arithmetic_l271_271821

theorem complex_number_arithmetic (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_number_arithmetic_l271_271821


namespace average_score_girls_proof_l271_271967

noncomputable def average_score_girls_all_schools (A a B b C c : ℕ)
  (adams_boys : ℕ) (adams_girls : ℕ) (adams_comb : ℕ)
  (baker_boys : ℕ) (baker_girls : ℕ) (baker_comb : ℕ)
  (carter_boys : ℕ) (carter_girls : ℕ) (carter_comb : ℕ)
  (all_boys_comb : ℕ) : ℕ :=
  -- Assume number of boys and girls per school A, B, C (boys) and a, b, c (girls)
  if (adams_boys * A + adams_girls * a) / (A + a) = adams_comb ∧
     (baker_boys * B + baker_girls * b) / (B + b) = baker_comb ∧
     (carter_boys * C + carter_girls * c) / (C + c) = carter_comb ∧
     (adams_boys * A + baker_boys * B + carter_boys * C) / (A + B + C) = all_boys_comb
  then (85 * a + 92 * b + 80 * c) / (a + b + c) else 0

theorem average_score_girls_proof (A a B b C c : ℕ)
  (adams_boys : ℕ := 82) (adams_girls : ℕ := 85) (adams_comb : ℕ := 83)
  (baker_boys : ℕ := 87) (baker_girls : ℕ := 92) (baker_comb : ℕ := 91)
  (carter_boys : ℕ := 78) (carter_girls : ℕ := 80) (carter_comb : ℕ := 80)
  (all_boys_comb : ℕ := 84) :
  average_score_girls_all_schools A a B b C c adams_boys adams_girls adams_comb baker_boys baker_girls baker_comb carter_boys carter_girls carter_comb all_boys_comb = 85 :=
by
  sorry

end average_score_girls_proof_l271_271967


namespace quadratic_two_distinct_real_roots_l271_271259

theorem quadratic_two_distinct_real_roots : 
  ∀ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 → 
  b^2 - 4 * a * c > 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l271_271259


namespace powers_of_2_not_powers_of_4_l271_271397

theorem powers_of_2_not_powers_of_4 (n : ℕ) (h1 : n < 500000) (h2 : ∃ k : ℕ, n = 2^k) (h3 : ∀ m : ℕ, n ≠ 4^m) : n = 9 := 
by
  sorry

end powers_of_2_not_powers_of_4_l271_271397


namespace joan_total_spent_on_clothing_l271_271413

theorem joan_total_spent_on_clothing :
  let shorts_cost := 15.00
  let jacket_cost := 14.82
  let shirt_cost := 12.51
  let shoes_cost := 21.67
  let hat_cost := 8.75
  let belt_cost := 6.34
  shorts_cost + jacket_cost + shirt_cost + shoes_cost + hat_cost + belt_cost = 79.09 :=
by
  sorry

end joan_total_spent_on_clothing_l271_271413


namespace function_decreasing_iff_a_neg_l271_271868

variable (a : ℝ)

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

theorem function_decreasing_iff_a_neg (h : ∀ x : ℝ, (7 * a * x ^ 6) ≤ 0) : a < 0 :=
by
  sorry

end function_decreasing_iff_a_neg_l271_271868


namespace symmetrical_circle_l271_271900

-- Defining the given circle's equation
def given_circle_eq (x y: ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Defining the equation of the symmetrical circle
def symmetrical_circle_eq (x y: ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Proving the symmetry property
theorem symmetrical_circle (x y : ℝ) : 
  (given_circle_eq x y) → (symmetrical_circle_eq (-x) (-y)) :=
by
  sorry

end symmetrical_circle_l271_271900


namespace negation_of_proposition_l271_271433

-- Define the proposition P(x)
def P (x : ℝ) : Prop := x + Real.log x > 0

-- Translate the problem into lean
theorem negation_of_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_proposition_l271_271433


namespace calculate_permutation_sum_l271_271023

noncomputable def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem calculate_permutation_sum (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 3) :
  A (2 * n) (n + 3) + A 4 (n + 1) = 744 := by
  sorry

end calculate_permutation_sum_l271_271023


namespace quadratic_root_square_condition_l271_271043

theorem quadratic_root_square_condition (p q r : ℝ) 
  (h1 : ∃ α β : ℝ, α + β = -q / p ∧ α * β = r / p ∧ β = α^2) : p - 4 * q ≥ 0 :=
sorry

end quadratic_root_square_condition_l271_271043


namespace total_number_of_balls_l271_271268

theorem total_number_of_balls
  (goldfish : ℕ) (platyfish : ℕ)
  (goldfish_balls : ℕ) (platyfish_balls : ℕ)
  (h1 : goldfish = 3) (h2 : platyfish = 10)
  (h3 : goldfish_balls = 10) (h4 : platyfish_balls = 5) :
  (goldfish * goldfish_balls + platyfish * platyfish_balls) = 80 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end total_number_of_balls_l271_271268


namespace solve_system_l271_271698

theorem solve_system (x y : ℝ) :
  (x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2) ↔ ((x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1)) :=
by
  sorry

end solve_system_l271_271698


namespace bus_empty_seats_l271_271130

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l271_271130


namespace rhombus_not_diagonals_equal_l271_271934

theorem rhombus_not_diagonals_equal (R : Type) [linear_ordered_field R] 
  (a b c d : R) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : a = d)
  (h_sym : ∀ x y : R, a = b → b = c → c = d → d = a)
  (h_cen_sym : ∀ p : R × R, p = (0, 0) → p = (0, 0)) :
  ¬(∀ p q : R × R, p ≠ q → (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) :=
by
  sorry

end rhombus_not_diagonals_equal_l271_271934


namespace joes_total_weight_l271_271569

theorem joes_total_weight (F S : ℕ) (h1 : F = 700) (h2 : 2 * F = S + 300) :
  F + S = 1800 :=
by
  sorry

end joes_total_weight_l271_271569


namespace probability_of_three_heads_in_eight_tosses_l271_271319

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l271_271319


namespace same_face_probability_l271_271645

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l271_271645


namespace sum_C_D_equals_seven_l271_271912

def initial_grid : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![ ![ some 1, none, none, none ],
     ![ none, some 2, none, none ],
     ![ none, none, none, none ],
     ![ none, none, none, some 4 ] ]

def valid_grid (grid : Matrix (Fin 4) (Fin 4) (Option Nat)) : Prop :=
  ∀ i j, grid i j ≠ none →
    (∀ k, k ≠ j → grid i k ≠ grid i j) ∧ 
    (∀ k, k ≠ i → grid k j ≠ grid i j)

theorem sum_C_D_equals_seven :
  ∃ (C D : Nat), C + D = 7 ∧ valid_grid initial_grid :=
sorry

end sum_C_D_equals_seven_l271_271912


namespace find_sin_B_l271_271727

variables (a b c : ℝ) (A B C : ℝ)

def sin_law_abc (a b : ℝ) (sinA : ℝ) (sinB : ℝ) : Prop := 
  (a / sinA) = (b / sinB)

theorem find_sin_B {a b : ℝ} (sinA : ℝ) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hA : sinA = 1 / 3) :
  ∃ sinB : ℝ, (sinB = 5 / 9) ∧ sin_law_abc a b sinA sinB :=
by
  use 5 / 9
  simp [sin_law_abc, ha, hb, hA]
  sorry

end find_sin_B_l271_271727


namespace green_fish_count_l271_271096

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end green_fish_count_l271_271096


namespace correct_equation_l271_271276

theorem correct_equation (a b : ℝ) : 
  (a + b)^2 = a^2 + 2 * a * b + b^2 := by
  sorry

end correct_equation_l271_271276


namespace walmart_pot_stacking_l271_271348

theorem walmart_pot_stacking :
  ∀ (total_pots pots_per_set shelves : ℕ),
    total_pots = 60 →
    pots_per_set = 5 →
    shelves = 4 →
    (total_pots / pots_per_set / shelves) = 3 :=
by 
  intros total_pots pots_per_set shelves h1 h2 h3
  sorry

end walmart_pot_stacking_l271_271348


namespace value_standard_deviations_less_than_mean_l271_271447

-- Definitions of the given conditions
def mean : ℝ := 15
def std_dev : ℝ := 1.5
def value : ℝ := 12

-- Lean 4 statement to prove the question
theorem value_standard_deviations_less_than_mean :
  (mean - value) / std_dev = 2 := by
  sorry

end value_standard_deviations_less_than_mean_l271_271447


namespace minimal_circle_intersect_l271_271198

noncomputable def circle_eq := 
  ∀ (x y : ℝ), 
    (x^2 + y^2 + 4 * x + y + 1 = 0) ∧
    (x^2 + y^2 + 2 * x + 2 * y + 1 = 0) → 
    (x^2 + y^2 + (6/5) * x + (3/5) * y + 1 = 0)

theorem minimal_circle_intersect :
  circle_eq :=
by
  sorry

end minimal_circle_intersect_l271_271198


namespace probability_three_heads_in_eight_tosses_l271_271330

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l271_271330


namespace complement_of_M_is_correct_l271_271718

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem complement_of_M_is_correct :
  (U \ M) = complement_M :=
by
  sorry

end complement_of_M_is_correct_l271_271718


namespace gcd_C_D_eq_6_l271_271871

theorem gcd_C_D_eq_6
  (C D : ℕ)
  (h_lcm : Nat.lcm C D = 180)
  (h_ratio : C = 5 * D / 6) :
  Nat.gcd C D = 6 := 
by
  sorry

end gcd_C_D_eq_6_l271_271871


namespace number_of_books_before_purchase_l271_271454

theorem number_of_books_before_purchase (x : ℕ) (h1 : x + 140 = (27 / 25) * x) : x = 1750 :=
by
  sorry

end number_of_books_before_purchase_l271_271454


namespace class_strength_l271_271602

/-- The average age of an adult class is 40 years.
    12 new students with an average age of 32 years join the class,
    therefore decreasing the average by 4 years.
    What was the original strength of the class? -/
theorem class_strength (x : ℕ) (h1 : ∃ (x : ℕ), ∀ (y : ℕ), y ≠ x → y = 40) 
                       (h2 : 12 ≥ 0) (h3 : 32 ≥ 0) (h4 : (x + 12) * 36 = 40 * x + 12 * 32) : 
  x = 12 := 
sorry

end class_strength_l271_271602


namespace differential_of_y_l271_271521

variable (x : ℝ) (dx : ℝ)

noncomputable def y := x * (Real.sin (Real.log x) - Real.cos (Real.log x))

theorem differential_of_y : (deriv y x * dx) = 2 * Real.sin (Real.log x) * dx := by
  sorry

end differential_of_y_l271_271521


namespace find_k_l271_271585

noncomputable def f (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : f a b c 1 = 0) 
  (h2 : 50 < f a b c 7) (h2' : f a b c 7 < 60) 
  (h3 : 70 < f a b c 8) (h3' : f a b c 8 < 80) 
  (h4 : 5000 * k < f a b c 100) (h4' : f a b c 100 < 5000 * (k + 1)) : 
  k = 3 := 
sorry

end find_k_l271_271585


namespace mike_oranges_l271_271972

-- Definitions and conditions
variables (O A B : ℕ)
def condition1 := A = 2 * O
def condition2 := B = O + A
def condition3 := O + A + B = 18

-- Theorem to prove that Mike received 3 oranges
theorem mike_oranges (h1 : condition1 O A) (h2 : condition2 O A B) (h3 : condition3 O A B) : 
  O = 3 := 
by 
  sorry

end mike_oranges_l271_271972


namespace matrix_multiplication_correct_l271_271686

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 3, -1], ![1, -2, 5], ![0, 6, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 4], ![3, 2, -1], ![0, 4, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![11, 2, 7], ![-5, 16, -4], ![18, 16, -8]]

theorem matrix_multiplication_correct :
  A * B = C :=
by
  sorry

end matrix_multiplication_correct_l271_271686


namespace smallest_mu_inequality_l271_271031

theorem smallest_mu_inequality (μ : ℝ) :
  (∀ (a b c d : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + 4*b^2 + 4*c^2 + d^2 ≥ 2*a*b + μ*b*c + 2*c*d) ↔ μ = 6 :=
begin
  sorry
end

end smallest_mu_inequality_l271_271031


namespace specific_heat_capacity_l271_271588

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end specific_heat_capacity_l271_271588


namespace bill_health_insurance_cost_l271_271977

noncomputable def calculate_health_insurance_cost : ℕ := 3000

theorem bill_health_insurance_cost
  (normal_monthly_price : ℕ := 500)
  (gov_pay_less_than_10000 : ℕ := 90) -- 90%
  (gov_pay_between_10001_and_40000 : ℕ := 50) -- 50%
  (gov_pay_more_than_50000 : ℕ := 20) -- 20%
  (hourly_wage : ℕ := 25)
  (weekly_hours : ℕ := 30)
  (weeks_per_month : ℕ := 4)
  (months_per_year : ℕ := 12)
  (income_between_10001_and_40000 : Prop := (hourly_wage * weekly_hours * weeks_per_month * months_per_year) >= 10001 ∧ (hourly_wage * weekly_hours * weeks_per_month * months_per_year) <= 40000):
  (calculate_health_insurance_cost = 3000) :=
by
sry


end bill_health_insurance_cost_l271_271977


namespace sum_of_squares_of_pairs_of_roots_l271_271892

-- Define the polynomial whose roots are a, b, c, d
def poly := polynomial.X^4 - 24 * polynomial.X^3 + 50 * polynomial.X^2 - 35 * polynomial.X + 10

-- Establish the main theorem based on the conditions and required proof
theorem sum_of_squares_of_pairs_of_roots :
  let {a, b, c, d} := multiset.of_nat_tuple 4
  (∀ x ∈ {a, b, c, d}, polynomial.eval x poly = 0) →
  (a + b)^2 + (b + c)^2 + (c + d)^2 + (d + a)^2 = 541 :=
sorry

end sum_of_squares_of_pairs_of_roots_l271_271892


namespace min_ab_value_l271_271077

theorem min_ab_value (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : (a * b) ≥ 2 := by 
  sorry

end min_ab_value_l271_271077


namespace absolute_value_half_angle_cosine_l271_271866

theorem absolute_value_half_angle_cosine (x : ℝ) (h1 : Real.sin x = -5 / 13) (h2 : ∀ n : ℤ, (2 * n) * Real.pi < x ∧ x < (2 * n + 1) * Real.pi) :
  |Real.cos (x / 2)| = Real.sqrt 26 / 26 :=
sorry

end absolute_value_half_angle_cosine_l271_271866


namespace infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l271_271478

open Nat

theorem infinite_solutions_2n_3n_square :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2 :=
sorry

theorem n_multiple_of_40 :
  ∀ n : ℤ, (∃ a b : ℤ, 2 * n + 1 = a^2 ∧ 3 * n + 1 = b^2) → (40 ∣ n) :=
sorry

theorem infinite_solutions_general (m : ℕ) (hm : 0 < m) :
  ∃ᶠ n : ℤ in at_top, ∃ a b : ℤ, m * n + 1 = a^2 ∧ (m + 1) * n + 1 = b^2 :=
sorry

end infinite_solutions_2n_3n_square_n_multiple_of_40_infinite_solutions_general_l271_271478


namespace number_of_green_fish_l271_271098

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l271_271098


namespace dividend_is_correct_l271_271463

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct_l271_271463


namespace jerry_needs_money_l271_271946

theorem jerry_needs_money
  (jerry_has : ℕ := 7)
  (total_needed : ℕ := 16)
  (cost_per_figure : ℕ := 8) :
  (total_needed - jerry_has) * cost_per_figure = 72 :=
by
  sorry

end jerry_needs_money_l271_271946


namespace total_persimmons_l271_271620

-- Definitions based on conditions in a)
def totalWeight (kg : ℕ) := kg = 3
def weightPerFivePersimmons (kg : ℕ) := kg = 1

-- The proof problem
theorem total_persimmons (k : ℕ) (w : ℕ) (x : ℕ) (h1 : totalWeight k) (h2 : weightPerFivePersimmons w) : x = 15 :=
by
  -- With the definitions totalWeight and weightPerFivePersimmons given in the conditions
  -- we aim to prove that the number of persimmons, x, is 15.
  sorry

end total_persimmons_l271_271620


namespace faye_candies_final_count_l271_271369

def initialCandies : ℕ := 47
def candiesEaten : ℕ := 25
def candiesReceived : ℕ := 40

theorem faye_candies_final_count : (initialCandies - candiesEaten + candiesReceived) = 62 :=
by
  sorry

end faye_candies_final_count_l271_271369


namespace largest_value_l271_271034

def X := (2010 / 2009) + (2010 / 2011)
def Y := (2010 / 2011) + (2012 / 2011)
def Z := (2011 / 2010) + (2011 / 2012)

theorem largest_value : X > Y ∧ X > Z := 
by
  sorry

end largest_value_l271_271034


namespace carrie_pays_94_l271_271511

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l271_271511


namespace value_of_expression_l271_271053

theorem value_of_expression (x y : ℝ) (h1 : 3 * x + 2 * y = 7) (h2 : 2 * x + 3 * y = 8) :
  13 * x ^ 2 + 22 * x * y + 13 * y ^ 2 = 113 :=
sorry

end value_of_expression_l271_271053


namespace lies_on_new_ellipse_lies_on_new_hyperbola_l271_271374

variable (x y c d a : ℝ)

def new_distance (P Q : ℝ × ℝ) : ℝ :=
  |P.1 - Q.1| + |P.2 - Q.2|

-- Definition for new ellipse.
def is_new_ellipse (E : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  new_distance E F1 + new_distance E F2 = 2 * a

-- Definition for new hyperbola.
def is_new_hyperbola (H : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (a : ℝ) : Prop :=
  |new_distance H F1 - new_distance H F2| = 2 * a

-- The point E lies on the new ellipse.
theorem lies_on_new_ellipse
  (E F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_ellipse E F1 F2 a :=
by sorry

-- The point H lies on the new hyperbola.
theorem lies_on_new_hyperbola
  (H F1 F2 : ℝ × ℝ) (a : ℝ) :
  is_new_hyperbola H F1 F2 a :=
by sorry

end lies_on_new_ellipse_lies_on_new_hyperbola_l271_271374


namespace smallest_x_inequality_l271_271035

theorem smallest_x_inequality : ∃ x : ℝ, (x^2 - 8 * x + 15 ≤ 0) ∧ (∀ y : ℝ, (y^2 - 8 * y + 15 ≤ 0) → (3 ≤ y)) ∧ x = 3 := 
sorry

end smallest_x_inequality_l271_271035


namespace expected_number_of_shots_l271_271789

def probability_hit : ℝ := 0.8
def probability_miss := 1 - probability_hit
def max_shots : ℕ := 3

theorem expected_number_of_shots : ∃ ξ : ℝ, ξ = 1.24 := by
  sorry

end expected_number_of_shots_l271_271789


namespace geometric_sum_n_eq_4_l271_271913

theorem geometric_sum_n_eq_4 :
  ∃ n : ℕ, (n = 4) ∧ 
  ((1 : ℚ) * (1 - (1 / 4 : ℚ) ^ n) / (1 - (1 / 4 : ℚ)) = (85 / 64 : ℚ)) :=
by
  use 4
  simp
  sorry

end geometric_sum_n_eq_4_l271_271913


namespace probability_of_forming_triangle_l271_271991

def segment_lengths : List ℕ := [1, 3, 5, 7, 9]
def valid_combinations : List (ℕ × ℕ × ℕ) := [(3, 5, 7), (3, 7, 9), (5, 7, 9)]
def total_combinations := Nat.choose 5 3

theorem probability_of_forming_triangle :
  (valid_combinations.length : ℚ) / total_combinations = 3 / 10 := 
by
  sorry

end probability_of_forming_triangle_l271_271991


namespace interest_for_1_rs_l271_271041

theorem interest_for_1_rs (I₅₀₀₀ : ℝ) (P : ℝ) (h : I₅₀₀₀ = 200) (hP : P = 5000) : I₅₀₀₀ / P = 0.04 :=
by
  rw [h, hP]
  norm_num

end interest_for_1_rs_l271_271041


namespace max_slope_of_line_OQ_l271_271842

-- Definitions of the problem conditions
def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {P | P.2^2 = 2 * p * P.1}

def focus : ℝ × ℝ := (1, 0)
def directrix_distance : ℝ := 2
def vector_PQ (Q : ℝ × ℝ) : ℝ × ℝ := ((10 * Q.1 - 9, 10 * Q.2))

-- The main theorem for the given problem
theorem max_slope_of_line_OQ (Q : ℝ × ℝ) (P : ℝ × ℝ)
  (hP : P ∈ parabola directrix_distance)
  (hPQ : (Q.1 - P.1, Q.2 - P.2) = 9 * ((Q.1 - focus.1), (Q.2 - focus.2))) :
  ∃ n : ℝ, n > 0 ∧ (10 * n) / (25 * n^2 + 9) = 1 / 3 :=
sorry

end max_slope_of_line_OQ_l271_271842


namespace probability_three_heads_in_eight_tosses_l271_271295

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l271_271295


namespace uneaten_pancakes_time_l271_271981

theorem uneaten_pancakes_time:
  ∀ (production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya : ℕ) (k : ℕ),
    production_rate_dad = 70 →
    production_rate_mom = 100 →
    consumption_rate_petya = 10 * 4 → -- 10 pancakes in 15 minutes -> (10/15) * 60 = 40 per hour
    consumption_rate_vasya = 2 * consumption_rate_petya →
    k * ((production_rate_dad + production_rate_mom) / 60 - (consumption_rate_petya + consumption_rate_vasya) / 60) ≥ 20 →
    k ≥ 24 := 
by
  intros production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya k
  sorry

end uneaten_pancakes_time_l271_271981


namespace notebooks_if_students_halved_l271_271047

-- Definitions based on the problem conditions
def totalNotebooks: ℕ := 512
def notebooksPerStudent (students: ℕ) : ℕ := students / 8
def notebooksWhenStudentsHalved (students notebooks: ℕ) : ℕ := notebooks / (students / 2)

-- Theorem statement
theorem notebooks_if_students_halved (S : ℕ) (h : S * (S / 8) = totalNotebooks) :
    notebooksWhenStudentsHalved S totalNotebooks = 16 :=
by
  sorry

end notebooks_if_students_halved_l271_271047


namespace second_player_wins_l271_271798

-- Defining the chess board and initial positions of the rooks
inductive Square : Type
| a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8
| b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8
| d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8
| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8
| g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8
| h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8
deriving DecidableEq

-- Define the initial positions of the rooks
def initial_white_rook_position : Square := Square.b2
def initial_black_rook_position : Square := Square.c4

-- Define the rules of movement: a rook can move horizontally or vertically unless blocked
def rook_can_move (start finish : Square) : Prop :=
  -- Only horizontal or vertical moves allowed
  sorry

-- Define conditions for a square being attacked by a rook at a given position
def is_attacked_by_rook (position target : Square) : Prop :=
  sorry

-- Define the condition for a player to be in a winning position if no moves are illegal
def player_can_win (white_position black_position : Square) : Prop :=
  sorry

-- The main theorem: Second player (black rook) can ensure a win
theorem second_player_wins : player_can_win initial_white_rook_position initial_black_rook_position :=
  sorry

end second_player_wins_l271_271798


namespace polygon_sides_l271_271059

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l271_271059


namespace area_computation_l271_271499

noncomputable def areaOfBoundedFigure : ℝ :=
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4), 
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  (integral / 2) - rectArea

theorem area_computation :
  let x (t : ℝ) := 2 * Real.sqrt 2 * Real.cos t
  let y (t : ℝ) := 5 * Real.sqrt 2 * Real.sin t
  let rectArea := 20
  let integral := ∫ t in (3 * Real.pi / 4)..(Real.pi / 4),
    (5 * Real.sqrt 2 * Real.sin t) * (-2 * Real.sqrt 2 * Real.sin t)
  ((integral / 2) - rectArea) = (5 * Real.pi - 10) :=
by
  sorry

end area_computation_l271_271499


namespace ratio_of_enclosed_area_l271_271806

theorem ratio_of_enclosed_area
  (R : ℝ)
  (h_chords_eq : ∀ (A B C : ℝ), A = B → A = C)
  (h_inscribed_angle : ∀ (A B C O : ℝ), AOC = 30 * π / 180)
  : ((π * R^2 / 6) + (R^2 / 2)) / (π * R^2) = (π + 3) / (6 * π) :=
by
  sorry

end ratio_of_enclosed_area_l271_271806


namespace total_amount_is_20_yuan_60_cents_l271_271284

-- Conditions
def ten_yuan_note : ℕ := 10
def five_yuan_notes : ℕ := 2 * 5
def twenty_cent_coins : ℕ := 3 * 20

-- Total amount calculation
def total_yuan : ℕ := ten_yuan_note + five_yuan_notes
def total_cents : ℕ := twenty_cent_coins

-- Conversion rates
def yuan_per_cent : ℕ := 100
def total_cents_in_yuan : ℕ := total_cents / yuan_per_cent
def remaining_cents : ℕ := total_cents % yuan_per_cent

-- Proof statement
theorem total_amount_is_20_yuan_60_cents : total_yuan = 20 ∧ total_cents_in_yuan = 0 ∧ remaining_cents = 60 :=
by
  sorry

end total_amount_is_20_yuan_60_cents_l271_271284


namespace probability_of_three_heads_in_eight_tosses_l271_271301

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l271_271301


namespace diameter_is_10sqrt6_l271_271771

noncomputable def radius (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  Real.sqrt (A / Real.pi)

noncomputable def diameter (A : ℝ) (hA : A = 150 * Real.pi) : ℝ :=
  2 * radius A hA

theorem diameter_is_10sqrt6 (A : ℝ) (hA : A = 150 * Real.pi) :
  diameter A hA = 10 * Real.sqrt 6 :=
  sorry

end diameter_is_10sqrt6_l271_271771


namespace heather_blocks_remaining_l271_271719

-- Definitions of the initial amount of blocks and the amount shared
def initial_blocks : ℕ := 86
def shared_blocks : ℕ := 41

-- The statement to be proven
theorem heather_blocks_remaining : (initial_blocks - shared_blocks = 45) :=
by sorry

end heather_blocks_remaining_l271_271719


namespace basketball_team_starters_l271_271788

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_team_starters :
  choose 4 2 * choose 14 4 = 6006 := by
  sorry

end basketball_team_starters_l271_271788


namespace parabola_equation_l271_271837

theorem parabola_equation (p : ℝ) (h : 0 < p) (Fₓ : ℝ) (Tₓ Tᵧ : ℝ) (Mₓ Mᵧ : ℝ)
  (eq_parabola : ∀ (y x : ℝ), y^2 = 2 * p * x → (y, x) = (Tᵧ, Tₓ))
  (F : (Fₓ, 0) = (p / 2, 0))
  (T_on_C : (Tᵧ, Tₓ) ∈ {(y, x) | y^2 = 2 * p * x})
  (FT_dist : dist (Fₓ, 0) (Tₓ, Tᵧ) = 5 / 2)
  (M : (Mₓ, Mᵧ) = (0, 1))
  (MF_MT_perp : ((Mᵧ - 0) / (Mₓ - Fₓ)) * ((Tᵧ - Mᵧ) / (Tₓ - Mᵧ)) = -1) :
  y^2 = 2 * x ∨ y^2 = 8 * x := 
sorry

end parabola_equation_l271_271837


namespace snacks_displayed_at_dawn_l271_271496

variable (S : ℝ)
variable (SoldMorning : ℝ)
variable (SoldAfternoon : ℝ)

axiom cond1 : SoldMorning = (3 / 5) * S
axiom cond2 : SoldAfternoon = 180
axiom cond3 : SoldMorning = SoldAfternoon

theorem snacks_displayed_at_dawn : S = 300 :=
by
  sorry

end snacks_displayed_at_dawn_l271_271496


namespace number_of_subsets_of_five_element_set_is_32_l271_271455

theorem number_of_subsets_of_five_element_set_is_32 (M : Finset ℕ) (h : M.card = 5) :
    (2 : ℕ) ^ 5 = 32 :=
by
  sorry

end number_of_subsets_of_five_element_set_is_32_l271_271455


namespace minimum_adjacent_white_pairs_l271_271542

theorem minimum_adjacent_white_pairs (total_black_cells : ℕ) (grid_size : ℕ) (total_pairs : ℕ) : 
  total_black_cells = 20 ∧ grid_size = 8 ∧ total_pairs = 112 → ∃ min_white_pairs : ℕ, min_white_pairs = 34 :=
by
  sorry

end minimum_adjacent_white_pairs_l271_271542


namespace solve_real_solution_l271_271519

theorem solve_real_solution:
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔
           (x = 4 + Real.sqrt 57) ∨ (x = 4 - Real.sqrt 57) :=
by
  sorry

end solve_real_solution_l271_271519


namespace discount_price_equation_correct_l271_271949

def original_price := 200
def final_price := 148
variable (a : ℝ) -- assuming a is a real number representing the percentage discount

theorem discount_price_equation_correct :
  original_price * (1 - a / 100) ^ 2 = final_price :=
sorry

end discount_price_equation_correct_l271_271949


namespace box_filled_with_cubes_no_leftover_l271_271286

-- Define dimensions of the box
def box_length : ℝ := 50
def box_width : ℝ := 60
def box_depth : ℝ := 43

-- Define volumes of different types of cubes
def volume_box : ℝ := box_length * box_width * box_depth
def volume_small_cube : ℝ := 2^3
def volume_medium_cube : ℝ := 3^3
def volume_large_cube : ℝ := 5^3

-- Define the smallest number of each type of cube
def num_large_cubes : ℕ := 1032
def num_medium_cubes : ℕ := 0
def num_small_cubes : ℕ := 0

-- Theorem statement ensuring the number of cubes completely fills the box
theorem box_filled_with_cubes_no_leftover :
  num_large_cubes * volume_large_cube + num_medium_cubes * volume_medium_cube + num_small_cubes * volume_small_cube = volume_box :=
by
  sorry

end box_filled_with_cubes_no_leftover_l271_271286


namespace empty_seats_after_second_stop_l271_271133

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l271_271133


namespace fixed_point_always_on_line_l271_271417

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line_l271_271417


namespace students_in_canteen_l271_271766

-- Definitions for conditions
def total_students : ℕ := 40
def absent_fraction : ℚ := 1 / 10
def classroom_fraction : ℚ := 3 / 4

-- Lean 4 statement
theorem students_in_canteen :
  let absent_students := (absent_fraction * total_students)
  let present_students := (total_students - absent_students)
  let classroom_students := (classroom_fraction * present_students)
  let canteen_students := (present_students - classroom_students)
  canteen_students = 9 := by
    sorry

end students_in_canteen_l271_271766


namespace four_digit_sum_divisible_l271_271953

theorem four_digit_sum_divisible (A B C D : ℕ) :
  (10 * A + B + 10 * C + D = 94) ∧ (1000 * A + 100 * B + 10 * C + D % 94 = 0) →
  false :=
by
  sorry

end four_digit_sum_divisible_l271_271953


namespace prob_three_heads_in_eight_tosses_l271_271305

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l271_271305


namespace supermarket_problem_l271_271001

-- Define that type A costs x yuan and type B costs y yuan
def cost_price_per_item (x y : ℕ) : Prop :=
  (10 * x + 8 * y = 880) ∧ (2 * x + 5 * y = 380)

-- Define purchasing plans with the conditions described
def purchasing_plans (a : ℕ) : Prop :=
  ∀ a : ℕ, 24 ≤ a ∧ a ≤ 26

theorem supermarket_problem : 
  (∃ x y, cost_price_per_item x y ∧ x = 40 ∧ y = 60) ∧ 
  (∃ n, purchasing_plans n ∧ n = 3) :=
by
  sorry

end supermarket_problem_l271_271001


namespace function_bounded_in_interval_l271_271533

variables {f : ℝ → ℝ}

theorem function_bounded_in_interval (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end function_bounded_in_interval_l271_271533


namespace x_squared_minus_y_squared_l271_271074

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l271_271074


namespace combinedTotalSandcastlesAndTowers_l271_271260

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l271_271260


namespace slowerPainterDuration_l271_271961

def slowerPainterStartTime : ℝ := 14 -- 2:00 PM in 24-hour format
def fasterPainterStartTime : ℝ := slowerPainterStartTime + 3 -- 3 hours later
def finishTime : ℝ := 24.6 -- 0.6 hours past midnight

theorem slowerPainterDuration :
  finishTime - slowerPainterStartTime = 10.6 :=
by
  sorry

end slowerPainterDuration_l271_271961


namespace time_jack_first_half_l271_271887

-- Define the conditions
def t_Jill : ℕ := 32
def t_2 : ℕ := 6
def t_Jack : ℕ := t_Jill - 7

-- Define the time Jack took for the first half
def t_1 : ℕ := t_Jack - t_2

-- State the theorem to prove
theorem time_jack_first_half : t_1 = 19 := by
  sorry

end time_jack_first_half_l271_271887


namespace probability_of_three_heads_in_eight_tosses_l271_271316

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l271_271316


namespace diff_of_two_numbers_l271_271618

theorem diff_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end diff_of_two_numbers_l271_271618


namespace total_students_1150_l271_271914

theorem total_students_1150 (T G : ℝ) (h1 : 92 + G = T) (h2 : G = 0.92 * T) : T = 1150 := 
by
  sorry

end total_students_1150_l271_271914


namespace Julia_played_with_kids_l271_271890

theorem Julia_played_with_kids :
  (∃ k : ℕ, k = 4) ∧ (∃ n : ℕ, n = 4 + 12) → (n = 16) :=
by
  sorry

end Julia_played_with_kids_l271_271890


namespace polygon_sides_l271_271058

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l271_271058


namespace bus_empty_seats_l271_271131

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end bus_empty_seats_l271_271131


namespace unique_solution_h_l271_271516

theorem unique_solution_h (h : ℝ) (hne_zero : h ≠ 0) :
  (∃! x : ℝ, (x - 3) / (h * x + 2) = x) ↔ h = 1 / 12 :=
by
  sorry

end unique_solution_h_l271_271516


namespace sum_of_specific_terms_in_arithmetic_sequence_l271_271584

theorem sum_of_specific_terms_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_S11 : S 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end sum_of_specific_terms_in_arithmetic_sequence_l271_271584


namespace sufficient_not_necessary_range_l271_271722

variable (x a : ℝ)

theorem sufficient_not_necessary_range (h1 : ∀ x, |x| < 1 → x < a) 
                                       (h2 : ¬(∀ x, x < a → |x| < 1)) :
  a ≥ 1 :=
sorry

end sufficient_not_necessary_range_l271_271722


namespace divisibility_by_100_l271_271627

theorem divisibility_by_100 (n : ℕ) (k : ℕ) (h : n = 5 * k + 2) :
    100 ∣ (5^n + 12*n^2 + 12*n + 3) :=
sorry

end divisibility_by_100_l271_271627


namespace circle_evaluation_circle_conversion_final_calculation_l271_271926

def circle (a : ℚ) (n : ℕ) : ℚ := list.foldl (/) a (list.replicate (n-1) a)

theorem circle_evaluation :
  circle 2 3 = (1/2) ∧
  circle (-3) 4 = (1/9) ∧
  circle (-1/3) 5 = -27 :=
by sorry

theorem circle_conversion (a : ℚ) (ha : a ≠ 0) (n : ℕ) (hn : 2 ≤ n) :
  circle a n = (1 / a ^ (n-2)) :=
by sorry

theorem final_calculation :
  27 * (1/9) + (-48) / (1/2^5) = -3 :=
by sorry

end circle_evaluation_circle_conversion_final_calculation_l271_271926


namespace johns_balance_at_end_of_first_year_l271_271274

theorem johns_balance_at_end_of_first_year (initial_deposit interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000) 
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 :=
by
  rw [h1, h2]
  norm_num

end johns_balance_at_end_of_first_year_l271_271274


namespace amanda_family_painting_theorem_l271_271968

theorem amanda_family_painting_theorem
  (rooms_with_4_walls : ℕ)
  (walls_per_room_with_4_walls : ℕ)
  (rooms_with_5_walls : ℕ)
  (walls_per_room_with_5_walls : ℕ)
  (walls_per_person : ℕ)
  (total_rooms : ℕ)
  (h1 : rooms_with_4_walls = 5)
  (h2 : walls_per_room_with_4_walls = 4)
  (h3 : rooms_with_5_walls = 4)
  (h4 : walls_per_room_with_5_walls = 5)
  (h5 : walls_per_person = 8)
  (h6 : total_rooms = 9)
  : rooms_with_4_walls * walls_per_room_with_4_walls +
    rooms_with_5_walls * walls_per_room_with_5_walls =
    5 * walls_per_person :=
by
  sorry

end amanda_family_painting_theorem_l271_271968


namespace arithmetic_sequence_n_value_l271_271708

noncomputable def common_ratio (a₁ S₃ : ℕ) : ℕ := by sorry

theorem arithmetic_sequence_n_value:
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
  (∀ n, a n > 0) →
  a 1 = 3 →
  S 3 = 21 →
  (∃ q, q > 0 ∧ common_ratio 1 q = q ∧ a 5 = 48) →
  n = 5 :=
by
  intros
  sorry

end arithmetic_sequence_n_value_l271_271708


namespace washington_goats_l271_271231

variables (W : ℕ) (P : ℕ) (total_goats : ℕ)

theorem washington_goats (W : ℕ) (h1 : P = W + 40) (h2 : total_goats = W + P) (h3 : total_goats = 320) : W = 140 :=
by
  sorry

end washington_goats_l271_271231


namespace max_regions_1002_1000_l271_271100

def regions_through_point (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 1

def max_regions (a b : ℕ) : ℕ := 
  let rB := regions_through_point b
  let first_line_through_A := rB + b + 1
  let remaining_lines_through_A := (a - 1) * (b + 2)
  first_line_through_A + remaining_lines_through_A

theorem max_regions_1002_1000 : max_regions 1002 1000 = 1504503 := by
  sorry

end max_regions_1002_1000_l271_271100


namespace sid_spent_on_computer_accessories_l271_271597

def initial_money : ℕ := 48
def snacks_cost : ℕ := 8
def remaining_money_more_than_half : ℕ := 4

theorem sid_spent_on_computer_accessories : 
  ∀ (m s r : ℕ), m = initial_money → s = snacks_cost → r = remaining_money_more_than_half →
  m - (r + m / 2 + s) = 12 :=
by
  intros m s r h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sid_spent_on_computer_accessories_l271_271597


namespace max_subsequences_2001_l271_271143

theorem max_subsequences_2001 (seq : List ℕ) (h_len : seq.length = 2001) : 
  ∃ n : ℕ, n = 667^3 :=
sorry

end max_subsequences_2001_l271_271143


namespace values_of_n_l271_271527

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l271_271527


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l271_271931

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l271_271931


namespace weavers_in_first_group_l271_271749

theorem weavers_in_first_group 
  (W : ℕ)
  (H1 : 4 / (W * 4) = 1 / W) 
  (H2 : (9 / 6) / 6 = 0.25) :
  W = 4 :=
sorry

end weavers_in_first_group_l271_271749


namespace contact_alignment_possible_l271_271158

/-- A vacuum tube has seven contacts arranged in a circle and is inserted into a socket that has seven holes.
Prove that it is possible to number the tube's contacts and the socket's holes in such a way that:
in any insertion of the tube, at least one contact will align with its corresponding hole (i.e., the hole with the same number). -/
theorem contact_alignment_possible : ∃ (f : Fin 7 → Fin 7), ∀ (rotation : Fin 7 → Fin 7), ∃ k : Fin 7, f k = rotation k := 
sorry

end contact_alignment_possible_l271_271158


namespace problem_a_problem_b_problem_c_problem_d_l271_271372

variable {a b : ℝ}

theorem problem_a (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) : ab ≤ 1 / 8 := sorry

theorem problem_b (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (1 / a) + (8 / b) ≥ 25 := sorry

theorem problem_c (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 := sorry

theorem problem_d (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  a^2 - b^2 > -1 / 4 := sorry

end problem_a_problem_b_problem_c_problem_d_l271_271372


namespace ratio_angela_jacob_l271_271805

-- Definitions for the conditions
def deans_insects := 30
def jacobs_insects := 5 * deans_insects
def angelas_insects := 75

-- The proof statement proving the ratio
theorem ratio_angela_jacob : angelas_insects / jacobs_insects = 1 / 2 :=
by
  -- Sorry is used here to indicate that the proof is skipped
  sorry

end ratio_angela_jacob_l271_271805


namespace calculate_fraction_l271_271685

theorem calculate_fraction :
  ( (12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484) )
  /
  ( (6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484) )
  = 181 := by
  sorry

end calculate_fraction_l271_271685


namespace total_length_of_segments_l271_271214

theorem total_length_of_segments
  (l1 l2 l3 l4 l5 l6 : ℕ) 
  (hl1 : l1 = 5) 
  (hl2 : l2 = 1) 
  (hl3 : l3 = 4) 
  (hl4 : l4 = 2) 
  (hl5 : l5 = 3) 
  (hl6 : l6 = 3) : 
  l1 + l2 + l3 + l4 + l5 + l6 = 18 := 
by 
  sorry

end total_length_of_segments_l271_271214


namespace find_initial_salt_concentration_l271_271488

noncomputable def initial_salt_concentration 
  (x : ℝ) (final_concentration : ℝ) (extra_water : ℝ) (extra_salt : ℝ) (evaporation_fraction : ℝ) : ℝ :=
  let initial_volume : ℝ := x
  let remaining_volume : ℝ := evaporation_fraction * initial_volume
  let mixed_volume : ℝ := remaining_volume + extra_water + extra_salt
  let target_salt_volume_fraction : ℝ := final_concentration / 100
  let initial_salt_volume_fraction : ℝ := (target_salt_volume_fraction * mixed_volume - extra_salt) / initial_volume * 100
  initial_salt_volume_fraction

theorem find_initial_salt_concentration :
  initial_salt_concentration 120 33.333333333333336 8 16 (3 / 4) = 18.333333333333332 :=
by
  sorry

end find_initial_salt_concentration_l271_271488


namespace probability_three_heads_in_eight_tosses_l271_271331

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l271_271331


namespace tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l271_271371

open Real

-- Part 1
theorem tan_x_min_x_div_x_min_sin_x_gt_two (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) :
  (tan x - x) / (x - sin x) > 2 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → tan x + 2 * sin x - a * x > 0) → a ≤ 3 :=
sorry

end tan_x_min_x_div_x_min_sin_x_gt_two_range_of_a_l271_271371


namespace arithmetic_problem_l271_271945

theorem arithmetic_problem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_problem_l271_271945


namespace distinct_real_roots_l271_271955

def operation (a b : ℝ) : ℝ := a^2 - a * b + b

theorem distinct_real_roots {x : ℝ} : 
  (operation x 3 = 5) → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation x1 3 = 5 ∧ operation x2 3 = 5) :=
by 
  -- Add your proof here
  sorry

end distinct_real_roots_l271_271955


namespace find_explicit_formula_l271_271705

variable (f : ℝ → ℝ)

theorem find_explicit_formula 
  (h : ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 5 :=
by
  sorry

end find_explicit_formula_l271_271705


namespace most_stable_performance_l271_271530

theorem most_stable_performance :
  ∀ (σ2_A σ2_B σ2_C σ2_D : ℝ), 
  σ2_A = 0.56 → 
  σ2_B = 0.78 → 
  σ2_C = 0.42 → 
  σ2_D = 0.63 → 
  σ2_C ≤ σ2_A ∧ σ2_C ≤ σ2_B ∧ σ2_C ≤ σ2_D :=
by
  intros σ2_A σ2_B σ2_C σ2_D hA hB hC hD
  sorry

end most_stable_performance_l271_271530


namespace sin_difference_identity_l271_271835

theorem sin_difference_identity 
  (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 1 / 3) : 
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := 
  sorry

end sin_difference_identity_l271_271835


namespace range_of_m_l271_271378

variable (m : ℝ)

def prop_p : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + m*x1 + 1 = 0) ∧ (x2^2 + m*x2 + 1 = 0)

def prop_q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h₁ : prop_p m) (h₂ : ¬prop_q m) : m < -2 ∨ m ≥ 3 :=
sorry

end range_of_m_l271_271378


namespace initial_number_of_quarters_l271_271153

theorem initial_number_of_quarters 
  (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (half_dollars : ℕ) (dollar_coins : ℕ) 
  (two_dollar_coins : ℕ) (quarters : ℕ)
  (cost_per_sundae : ℝ) 
  (special_topping_cost : ℝ)
  (featured_flavor_discount : ℝ)
  (members_with_special_topping : ℕ)
  (members_with_featured_flavor : ℕ)
  (left_over : ℝ)
  (expected_quarters : ℕ) :
  pennies = 123 ∧
  nickels = 85 ∧
  dimes = 35 ∧
  half_dollars = 15 ∧
  dollar_coins = 5 ∧
  quarters = expected_quarters ∧
  two_dollar_coins = 4 ∧
  cost_per_sundae = 5.25 ∧
  special_topping_cost = 0.50 ∧
  featured_flavor_discount = 0.25 ∧
  members_with_special_topping = 3 ∧
  members_with_featured_flavor = 5 ∧
  left_over = 0.97 →
  expected_quarters = 54 :=
  by
  sorry

end initial_number_of_quarters_l271_271153


namespace arvin_fifth_day_running_distance_l271_271883

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end arvin_fifth_day_running_distance_l271_271883


namespace ratio_of_areas_l271_271493

noncomputable def radius_of_circle (r : ℝ) : ℝ := r

def equilateral_triangle_side_length (r : ℝ) : ℝ := r * Real.sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * s^2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem ratio_of_areas (r : ℝ) : 
  ∃ K : ℝ, K = (3 * Real.sqrt 3) / (4 * Real.pi) → 
  (area_of_equilateral_triangle (equilateral_triangle_side_length r)) / (area_of_circle r) = K := 
by 
  sorry

end ratio_of_areas_l271_271493


namespace count_perfect_cubes_between_bounds_l271_271561

theorem count_perfect_cubes_between_bounds :
  let lower_bound := 3^6 + 1
  let upper_bound := 3^12 + 1
  -- the number of perfect cubes k^3 such that 3^6 + 1 < k^3 < 3^12 + 1 inclusive is 72
  (730 < k * k * k ∧ k * k * k <= 531442 ∧ 10 <= k ∧ k <= 81 → k = 72) :=
by
  let lower_bound : ℕ := 3^6 + 1
  let upper_bound : ℕ := 3^12 + 1
  sorry

end count_perfect_cubes_between_bounds_l271_271561


namespace peter_wins_prize_at_least_one_person_wins_prize_l271_271234

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l271_271234


namespace real_solution_approximately_l271_271989

noncomputable def exists_real_solution (x : ℝ) : Prop :=
x = 1 - x^2 + x^4 - x^6 + x^8 - x^10 + ...

theorem real_solution_approximately :
  ∃ x : ℝ, exists_real_solution x ∧ x ≈ 0.6823 := 
sorry

end real_solution_approximately_l271_271989


namespace seven_does_not_always_divide_l271_271467

theorem seven_does_not_always_divide (n : ℤ) :
  ¬(7 ∣ (n ^ 2225 - n ^ 2005)) :=
by sorry

end seven_does_not_always_divide_l271_271467


namespace smallest_y_for_perfect_cube_l271_271956

-- Define the given conditions
def x : ℕ := 5 * 24 * 36

-- State the theorem to prove
theorem smallest_y_for_perfect_cube (y : ℕ) (h : y = 50) : 
  ∃ y, (x * y) % (y * y * y) = 0 :=
by
  sorry

end smallest_y_for_perfect_cube_l271_271956


namespace log_expression_equals_l271_271689

noncomputable def expression (x y : ℝ) : ℝ :=
  (Real.log x^2) / (Real.log y^10) *
  (Real.log y^3) / (Real.log x^7) *
  (Real.log x^4) / (Real.log y^8) *
  (Real.log y^6) / (Real.log x^9) *
  (Real.log x^11) / (Real.log y^5)

theorem log_expression_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  expression x y = (1 / 15) * Real.log y / Real.log x :=
sorry

end log_expression_equals_l271_271689


namespace download_time_correct_l271_271229

-- Define the given conditions
def total_size : ℕ := 880
def downloaded : ℕ := 310
def speed : ℕ := 3

-- Calculate the remaining time to download
def time_remaining : ℕ := (total_size - downloaded) / speed

-- Theorem statement that needs to be proven
theorem download_time_correct : time_remaining = 190 := by
  -- Proof goes here
  sorry

end download_time_correct_l271_271229


namespace hyperbola_focus_distance_l271_271998
open Real

theorem hyperbola_focus_distance
  (a b : ℝ)
  (ha : a = 5)
  (hb : b = 3)
  (hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (∃ M : ℝ × ℝ, M = (x, y)))
  (M : ℝ × ℝ)
  (hM_on_hyperbola : ∃ x y : ℝ, M = (x, y) ∧ x^2 / a^2 - y^2 / b^2 = 1)
  (F1_pos : ℝ)
  (h_dist_F1 : dist M (F1_pos, 0) = 18) :
  (∃ (F2_dist : ℝ), (F2_dist = 8 ∨ F2_dist = 28) ∧ dist M (F2_dist, 0) = F2_dist) := 
sorry

end hyperbola_focus_distance_l271_271998


namespace six_digit_numbers_with_at_least_two_zeros_l271_271851

noncomputable def num_six_digit_numbers_with_at_least_two_zeros : ℕ :=
  73314

theorem six_digit_numbers_with_at_least_two_zeros :
  ∃ n : ℕ, n = num_six_digit_numbers_with_at_least_two_zeros := by
  use 73314
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271851


namespace perimeter_of_triangle_l271_271610

-- Define the side lengths of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 7

-- Define the third side of the triangle, which is an even number and satisfies the triangle inequality conditions
def side3 : ℕ := 6

-- Define the theorem to prove the perimeter of the triangle
theorem perimeter_of_triangle : side1 + side2 + side3 = 15 := by
  -- The proof is omitted for brevity
  sorry

end perimeter_of_triangle_l271_271610


namespace find_n_l271_271357

noncomputable def angles_periodic_mod_eq (n : ℤ) : Prop :=
  -100 < n ∧ n < 100 ∧ Real.tan (n * Real.pi / 180) = Real.tan (216 * Real.pi / 180)

theorem find_n (n : ℤ) (h : angles_periodic_mod_eq n) : n = 36 :=
  sorry

end find_n_l271_271357


namespace evaluate_magnitude_of_product_l271_271691

theorem evaluate_magnitude_of_product :
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  Complex.abs (z1 * z2) = 4 * Real.sqrt 43 := by
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  suffices Complex.abs z1 * Complex.abs z2 = 4 * Real.sqrt 43 by sorry
  sorry

end evaluate_magnitude_of_product_l271_271691


namespace six_digit_numbers_with_at_least_two_zeros_l271_271846

theorem six_digit_numbers_with_at_least_two_zeros :
  let total_numbers := 900000 in
  let no_zeros := 9^6 in
  let exactly_one_zero := 6 * 9^5 in
  total_numbers - no_zeros - exactly_one_zero = 14265 :=
by
  let total_numbers := 900000
  let no_zeros := 9^6
  let exactly_one_zero := 6 * 9^5
  show total_numbers - no_zeros - exactly_one_zero = 14265
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271846


namespace math_marks_l271_271028

theorem math_marks (english physics chemistry biology total_marks math_marks : ℕ) 
  (h_eng : english = 73)
  (h_phy : physics = 92)
  (h_chem : chemistry = 64)
  (h_bio : biology = 82)
  (h_avg : total_marks = 76 * 5) :
  math_marks = 69 := 
by
  sorry

end math_marks_l271_271028


namespace probability_exactly_three_heads_l271_271313
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l271_271313


namespace shirt_assignment_ways_l271_271665

/--
  12 people stand in a row.
  Each person is given a red or a blue shirt.
  Every minute, exactly one pair of people with the same color currently standing next to each other leaves.
  After 6 minutes, everyone has left.
  Prove that the number of ways the shirts could have been assigned initially is 837.
-/
theorem shirt_assignment_ways : 
  let n := 12
  let k := 6
  let possible_assignments := -- these are all possible color assignments for the shirts
    { assignments | ∃ (red_count blue_count : ℕ), 
        red_count + blue_count = n ∧
        -- logic ensuring all pairs of same color shirts adjacent are removed in k steps
        only_pairwise_removal_possible assignments k
    }
  in
  card possible_assignments = 837 :=
sorry

end shirt_assignment_ways_l271_271665


namespace solve_for_a_l271_271398

theorem solve_for_a (a : ℝ) (h : 4 * a + 9 + (3 * a + 5) = 0) : a = -2 :=
by
  sorry

end solve_for_a_l271_271398


namespace hyperbola_m_value_l271_271450

theorem hyperbola_m_value
  (m : ℝ)
  (h1 : 3 * m * x^2 - m * y^2 = 3)
  (focus : ∃ c, (0, c) = (0, 2)) :
  m = -1 :=
sorry

end hyperbola_m_value_l271_271450


namespace short_sleeve_shirts_l271_271807

theorem short_sleeve_shirts (total_shirts long_sleeve_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9) 
  (h2 : long_sleeve_shirts = 5)
  (h3 : short_sleeve_shirts = total_shirts - long_sleeve_shirts) : 
  short_sleeve_shirts = 4 :=
by 
  sorry

end short_sleeve_shirts_l271_271807


namespace least_possible_sum_of_bases_l271_271987

theorem least_possible_sum_of_bases : 
  ∃ (c d : ℕ), (2 * c + 9 = 9 * d + 2) ∧ (c + d = 13) :=
by
  sorry

end least_possible_sum_of_bases_l271_271987


namespace tom_saves_promotion_l271_271787

open Nat

theorem tom_saves_promotion (price : ℕ) (disc_percent : ℕ) (discount_amount : ℕ) 
    (promotion_x_cost second_pair_cost_promo_x promotion_y_cost promotion_savings : ℕ) 
    (h1 : price = 50)
    (h2 : disc_percent = 40)
    (h3 : discount_amount = 15)
    (h4 : second_pair_cost_promo_x = price - (price * disc_percent / 100))
    (h5 : promotion_x_cost = price + second_pair_cost_promo_x)
    (h6 : promotion_y_cost = price + (price - discount_amount))
    (h7 : promotion_savings = promotion_y_cost - promotion_x_cost) :
  promotion_savings = 5 :=
by
  sorry

end tom_saves_promotion_l271_271787


namespace nature_of_a_l271_271449

variable {a m n p q : ℤ}
variable {x : ℤ}

/-- Given expression 15x^2 + ax + 15 can be factored into linear binomials with integer coefficients -/
theorem nature_of_a (h1 : ∃ (m n p q : ℤ), (15 = m * p) ∧ (15 = n * q) ∧ (a = m * q + n * p)) : 
  ∃ k : ℤ, a = 2 * k :=
by sorry

end nature_of_a_l271_271449


namespace number_with_at_least_two_zeros_l271_271849

-- A 6-digit number can have for its leftmost digit anything from 1 to 9 inclusive,
-- and for each of its next five digits anything from 0 through 9 inclusive.
def total_6_digit_numbers : ℕ := 9 * 10^5

-- A 6-digit number with no zeros consists solely of digits from 1 to 9
def no_zero : ℕ := 9^6

-- A 6-digit number with exactly one zero
def exactly_one_zero : ℕ := 5 * 9^5

-- The number of 6-digit numbers with less than two zeros is the sum of no_zero and exactly_one_zero
def less_than_two_zeros : ℕ := no_zero + exactly_one_zero

-- The number of 6-digit numbers with at least two zeros is the difference between total_6_digit_numbers and less_than_two_zeros
def at_least_two_zeros : ℕ := total_6_digit_numbers - less_than_two_zeros

-- The theorem that states the number of 6-digit numbers with at least two zeros is 73,314
theorem number_with_at_least_two_zeros : at_least_two_zeros = 73314 := 
by
  sorry

end number_with_at_least_two_zeros_l271_271849


namespace coefficient_x4_in_expansion_l271_271728

theorem coefficient_x4_in_expansion :
    let f := (λ x : ℤ, (x + (1/x)) ^ 6)
    let expansion := finset.sum (finset.range 7) (λ r, (nat.choose 6 r) * ((x : ℚ)^ (6 - 2 * r)))
    ∃ c : ℚ, expansion = c * x ^ 4 := by
  sorry

end coefficient_x4_in_expansion_l271_271728


namespace x_minus_y_eq_eight_l271_271460

theorem x_minus_y_eq_eight (x y : ℝ) (hx : 3 = 0.15 * x) (hy : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end x_minus_y_eq_eight_l271_271460


namespace probability_green_or_yellow_l271_271652

def total_marbles (green yellow red blue : Nat) : Nat :=
  green + yellow + red + blue

def marble_probability (green yellow red blue : Nat) : Rat :=
  (green + yellow) / (total_marbles green yellow red blue)

theorem probability_green_or_yellow :
  let green := 4
  let yellow := 3
  let red := 4
  let blue := 2
  marble_probability green yellow red blue = 7 / 13 := by
  sorry

end probability_green_or_yellow_l271_271652


namespace tracy_customers_l271_271461

theorem tracy_customers
  (total_customers : ℕ)
  (customers_bought_two_each : ℕ)
  (customers_bought_one_each : ℕ)
  (customers_bought_four_each : ℕ)
  (total_paintings_sold : ℕ)
  (h1 : total_customers = 20)
  (h2 : customers_bought_one_each = 12)
  (h3 : customers_bought_four_each = 4)
  (h4 : total_paintings_sold = 36)
  (h5 : 2 * customers_bought_two_each + customers_bought_one_each + 4 * customers_bought_four_each = total_paintings_sold) :
  customers_bought_two_each = 4 :=
by
  sorry

end tracy_customers_l271_271461


namespace same_number_on_four_dice_l271_271651

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l271_271651


namespace work_together_days_l271_271663

theorem work_together_days (ravi_days prakash_days : ℕ) (hr : ravi_days = 50) (hp : prakash_days = 75) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 30 :=
sorry

end work_together_days_l271_271663


namespace hyperbola_asymptotes_l271_271756

theorem hyperbola_asymptotes:
  ∀ (x y : ℝ),
  ( ∀ y, y = (1 + (4 / 5) * x) ∨ y = (1 - (4 / 5) * x) ) →
  (y-1)^2 / 16 - x^2 / 25 = 1 →
  (∃ m b: ℝ, m > 0 ∧ m = 4/5 ∧ b = 1) := by
  sorry

end hyperbola_asymptotes_l271_271756


namespace not_divisible_l271_271103

theorem not_divisible {x y : ℕ} (hx : x > 0) (hy : y > 2) : ¬ (2^y - 1) ∣ (2^x + 1) := sorry

end not_divisible_l271_271103


namespace max_S_n_l271_271553

theorem max_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) (h2 : d < 0) (h3 : S 6 = 5 * (a 1) + 10 * d) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, S m ≤ S n) :=
by
  sorry

end max_S_n_l271_271553


namespace six_nine_op_l271_271813

variable (m n : ℚ)

def op (x y : ℚ) : ℚ := m^2 * x + n * y - 1

theorem six_nine_op :
  (op m n 2 3 = 3) →
  (op m n 6 9 = 11) :=
by
  intro h
  sorry

end six_nine_op_l271_271813


namespace cost_of_3000_pencils_l271_271948

theorem cost_of_3000_pencils (pencils_per_box : ℕ) (cost_per_box : ℝ) (pencils_needed : ℕ) (unit_cost : ℝ): 
  pencils_per_box = 120 → cost_per_box = 36 → pencils_needed = 3000 → unit_cost = 0.30 →
  (pencils_needed * unit_cost = (3000 : ℝ) * 0.30) :=
by
  intros _ _ _ _
  sorry

end cost_of_3000_pencils_l271_271948


namespace probability_at_least_four_same_face_l271_271040

theorem probability_at_least_four_same_face :
  let total_outcomes := (2 : ℕ) ^ 5,
      favorable_outcomes := 1 + 1 + (Nat.choose 5 1) + (Nat.choose 5 1),
      probability := favorable_outcomes / total_outcomes in
  probability = (3 : ℚ) / 8 :=
by
  sorry

end probability_at_least_four_same_face_l271_271040


namespace xiaoming_total_money_l271_271940

def xiaoming_money (x : ℕ) := 9 * x

def fresh_milk_cost (y : ℕ) := 6 * y

def yogurt_cost_equation (x y : ℕ) := y = x + 6

theorem xiaoming_total_money (x : ℕ) (y : ℕ)
  (h1: fresh_milk_cost y = xiaoming_money x)
  (h2: yogurt_cost_equation x y) : xiaoming_money x = 108 := 
  sorry

end xiaoming_total_money_l271_271940


namespace percentage_relationships_l271_271225

variable (a b c d e f g : ℝ)

theorem percentage_relationships (h1 : d = 0.22 * b) (h2 : d = 0.35 * f)
                                 (h3 : e = 0.27 * a) (h4 : e = 0.60 * f)
                                 (h5 : c = 0.14 * a) (h6 : c = 0.40 * b)
                                 (h7 : d = 2 * c) (h8 : g = 3 * e):
    b = 0.7 * a ∧ f = 0.45 * a ∧ g = 0.81 * a :=
sorry

end percentage_relationships_l271_271225


namespace paint_per_large_canvas_l271_271347

-- Define the conditions
variables (L : ℕ) (paint_large paint_small total_paint : ℕ)

-- Given conditions
def large_canvas_paint := 3 * L
def small_canvas_paint := 4 * 2
def total_paint_used := large_canvas_paint + small_canvas_paint

-- Statement that needs to be proven
theorem paint_per_large_canvas :
  total_paint_used = 17 → L = 3 :=
by
  intro h
  sorry

end paint_per_large_canvas_l271_271347


namespace inequality_abc_l271_271595

theorem inequality_abc (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a) + (1/b) ≥ 4/(a + b) :=
by
  sorry

end inequality_abc_l271_271595


namespace prob_three_heads_in_eight_tosses_l271_271307

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l271_271307


namespace area_within_fence_is_328_l271_271253

-- Define the dimensions of the fenced area
def main_rectangle_length : ℝ := 20
def main_rectangle_width : ℝ := 18

-- Define the dimensions of the square cutouts
def cutout_length : ℝ := 4
def cutout_width : ℝ := 4

-- Calculate the areas
def main_rectangle_area : ℝ := main_rectangle_length * main_rectangle_width
def cutout_area : ℝ := cutout_length * cutout_width

-- Define the number of cutouts
def number_of_cutouts : ℝ := 2

-- Calculate the final area within the fence
def area_within_fence : ℝ := main_rectangle_area - number_of_cutouts * cutout_area

theorem area_within_fence_is_328 : area_within_fence = 328 := by
  -- This is a place holder for the proof, replace it with the actual proof
  sorry

end area_within_fence_is_328_l271_271253


namespace parabola_with_distance_two_max_slope_OQ_l271_271838

-- Define the given conditions
def parabola_equation (p : ℝ) : Prop := ∀ (x y : ℝ), y^2 = 2 * p * x
def distance_focus_directrix (d : ℝ) : Prop := d = 2

-- Define the proofs we need to show
theorem parabola_with_distance_two : ∀ (p : ℝ), p = 2 → parabola_equation p :=
by
  assume p hp,
  sorry -- Proof here proves that y^2 = 4x if p = 2

theorem max_slope_OQ : ∀ (n m : ℝ), (9 * (1 - m), -9 * n) → K = n / m → K ≤ 1 / 3 :=
by
  assume n m hdef K,
  sorry -- Proof here proves that maximum slope K = 1/3 under given conditions

end parabola_with_distance_two_max_slope_OQ_l271_271838


namespace carrie_pays_l271_271507

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l271_271507


namespace six_digit_numbers_with_at_least_two_zeros_l271_271845

theorem six_digit_numbers_with_at_least_two_zeros : 
  (∃ n : ℕ, n = 900000) → 
  (∃ no_zero : ℕ, no_zero = 531441) → 
  (∃ one_zero : ℕ, one_zero = 295245) → 
  (∃ at_least_two_zeros : ℕ, at_least_two_zeros = 900000 - (531441 + 295245)) → 
  at_least_two_zeros = 73314 :=
by
  intros n no_zero one_zero at_least_two_zeros
  rw [at_least_two_zeros, n, no_zero, one_zero]
  norm_num
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271845


namespace abigail_money_loss_l271_271799

theorem abigail_money_loss
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  initial_amount - spent_amount - remaining_amount = 6 :=
by sorry

end abigail_money_loss_l271_271799


namespace num_ordered_pairs_no_real_solution_l271_271825

theorem num_ordered_pairs_no_real_solution : 
  {n : ℕ // ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4*c < 0 ∨ c^2 - 4*b < 0) ∧ n = 6 } := by
sorry

end num_ordered_pairs_no_real_solution_l271_271825


namespace problem_solved_by_at_least_one_student_l271_271906

theorem problem_solved_by_at_least_one_student (P_A P_B : ℝ) 
  (hA : P_A = 0.8) 
  (hB : P_B = 0.9) :
  (1 - (1 - P_A) * (1 - P_B) = 0.98) :=
by
  have pAwrong := 1 - P_A
  have pBwrong := 1 - P_B
  have both_wrong := pAwrong * pBwrong
  have one_right := 1 - both_wrong
  sorry

end problem_solved_by_at_least_one_student_l271_271906


namespace parabola_max_slope_l271_271839

-- Define the parabola and the distance condition
def parabola_distance_condition (p : ℝ) := (2 * p = 2) ∧ (p > 0)

-- Define the equation of the parabola when p = 2
def parabola_equation := ∀ (x y : ℝ), y^2 = 4 * x

-- Define the points and the condition for maximum slope
def max_slope_condition (O P Q F : (ℝ × ℝ)) :=
  O = (0, 0) ∧ F = (1, 0) ∧ 
  (∃ m n : ℝ, Q = (m, n) ∧ P = (10 * m - 9, 10 * n) ∧ (10 * n)^2 = 4 * (10 * m - 9)) ∧ 
  ∀ K : ℝ, (K = n / m) → K ≤ 1 / 3

-- The Lean statement combining all conditions
theorem parabola_max_slope :
  ∃ (p : ℝ), parabola_distance_condition p ∧ (∃ O P Q F : (ℝ × ℝ), max_slope_condition O P Q F)
  :=
sorry

end parabola_max_slope_l271_271839


namespace no_real_solutions_for_g_g_x_l271_271810

theorem no_real_solutions_for_g_g_x (d : ℝ) :
  ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 4 * x1 + d)^2 + 4 * (x1^2 + 4 * x1 + d) + d = 0 ∧
                                (x2^2 + 4 * x2 + d)^2 + 4 * (x2^2 + 4 * x2 + d) + d = 0 :=
by
  sorry

end no_real_solutions_for_g_g_x_l271_271810


namespace quadratic_solution_l271_271364

noncomputable def g (x : ℝ) : ℝ := x^2 + 2021 * x + 18

theorem quadratic_solution : ∀ x : ℝ, g (g x + x + 1) / g x = x^2 + 2023 * x + 2040 :=
by
  intros
  sorry

end quadratic_solution_l271_271364


namespace coin_toss_probability_l271_271322

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l271_271322


namespace prime_iff_totient_divisor_sum_l271_271593

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def euler_totient (n : ℕ) : ℕ := sorry  -- we assume implementation of Euler's Totient function
def divisor_sum (n : ℕ) : ℕ := sorry  -- we assume implementation of Divisor sum function

theorem prime_iff_totient_divisor_sum (n : ℕ) :
  (2 ≤ n) → (euler_totient n ∣ (n - 1)) → (n + 1 ∣ divisor_sum n) → is_prime n :=
  sorry

end prime_iff_totient_divisor_sum_l271_271593


namespace probability_exactly_three_heads_l271_271311
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l271_271311


namespace simplify_expression_l271_271747

variable (a : ℝ)

theorem simplify_expression :
    5 * a^2 - (a^2 - 2 * (a^2 - 3 * a)) = 6 * a^2 - 6 * a := by
  sorry

end simplify_expression_l271_271747


namespace lunch_break_is_48_minutes_l271_271743

noncomputable def lunch_break_duration (L : ℝ) (p a : ℝ) : Prop :=
  (8 - L) * (p + a) = 0.6 ∧ 
  (9 - L) * p = 0.35 ∧
  (5 - L) * a = 0.1

theorem lunch_break_is_48_minutes :
  ∃ L p a, lunch_break_duration L p a ∧ L * 60 = 48 :=
by
  -- proof steps would go here
  sorry

end lunch_break_is_48_minutes_l271_271743


namespace probability_three_heads_in_eight_tosses_l271_271332

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l271_271332


namespace steve_cookie_boxes_l271_271600

theorem steve_cookie_boxes (total_spent milk_cost cereal_cost banana_cost apple_cost : ℝ)
  (num_cereals num_bananas num_apples : ℕ) (cookie_cost_multiplier : ℝ) (cookie_cost : ℝ)
  (cookie_boxes : ℕ) :
  total_spent = 25 ∧ milk_cost = 3 ∧ cereal_cost = 3.5 ∧ banana_cost = 0.25 ∧ apple_cost = 0.5 ∧
  cookie_cost_multiplier = 2 ∧ 
  num_cereals = 2 ∧ num_bananas = 4 ∧ num_apples = 4 ∧
  cookie_cost = cookie_cost_multiplier * milk_cost ∧
  total_spent = (milk_cost + num_cereals * cereal_cost + num_bananas * banana_cost + num_apples * apple_cost + cookie_boxes * cookie_cost)
  → cookie_boxes = 2 :=
sorry

end steve_cookie_boxes_l271_271600


namespace polyhedron_edges_l271_271486

theorem polyhedron_edges (F V E : ℕ) (h1 : F = 12) (h2 : V = 20) (h3 : F + V = E + 2) : E = 30 :=
by
  -- Additional details would go here, proof omitted as instructed.
  sorry

end polyhedron_edges_l271_271486


namespace arianna_sleep_hours_l271_271974

-- Defining the given conditions
def total_hours_in_a_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_in_class : ℕ := 3
def hours_at_gym : ℕ := 2
def hours_on_chores : ℕ := 5

-- Formulating the total hours spent on activities
def total_hours_on_activities := hours_at_work + hours_in_class + hours_at_gym + hours_on_chores

-- Proving Arianna's sleep hours
theorem arianna_sleep_hours : total_hours_in_a_day - total_hours_on_activities = 8 :=
by
  -- Direct proof placeholder, to be filled in with actual proof steps or tactic
  sorry

end arianna_sleep_hours_l271_271974


namespace units_digit_of_eight_consecutive_odd_numbers_is_zero_l271_271699

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem units_digit_of_eight_consecutive_odd_numbers_is_zero (n : ℤ)
  (h₀ : is_odd n) :
  ((n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) * (n + 14)) % 10 = 0) :=
sorry

end units_digit_of_eight_consecutive_odd_numbers_is_zero_l271_271699


namespace range_of_a_l271_271780

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end range_of_a_l271_271780


namespace probability_of_same_number_on_four_dice_l271_271634

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l271_271634


namespace average_weight_b_c_l271_271448

theorem average_weight_b_c (A B C : ℝ) (h1 : A + B + C = 126) (h2 : A + B = 80) (h3 : B = 40) : 
  (B + C) / 2 = 43 := 
by 
  -- Proof would go here, but is left as sorry as per instructions
  sorry

end average_weight_b_c_l271_271448


namespace halfway_fraction_between_is_one_fourth_l271_271177

theorem halfway_fraction_between_is_one_fourth : 
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  ((f1 + f2 + f3) / 3) = (1 / 4) := 
by
  let f1 := (1 / 4 : ℚ)
  let f2 := (1 / 6 : ℚ)
  let f3 := (1 / 3 : ℚ)
  sorry

end halfway_fraction_between_is_one_fourth_l271_271177


namespace slope_of_line_between_intersections_of_circles_l271_271754

theorem slope_of_line_between_intersections_of_circles :
  ∀ C D : ℝ × ℝ, 
    -- Conditions: equations of the circles
    (C.1^2 + C.2^2 - 6 * C.1 + 4 * C.2 - 8 = 0) ∧ (C.1^2 + C.2^2 - 8 * C.1 - 2 * C.2 + 10 = 0) →
    (D.1^2 + D.2^2 - 6 * D.1 + 4 * D.2 - 8 = 0) ∧ (D.1^2 + D.2^2 - 8 * D.1 - 2 * D.2 + 10 = 0) →
    -- Question: slope of line CD
    ((C.2 - D.2) / (C.1 - D.1) = -1 / 3) :=
by
  sorry

end slope_of_line_between_intersections_of_circles_l271_271754


namespace total_gas_cost_l271_271114

def gas_price_station_1 : ℝ := 3
def gas_price_station_2 : ℝ := 3.5
def gas_price_station_3 : ℝ := 4
def gas_price_station_4 : ℝ := 4.5
def tank_capacity : ℝ := 12

theorem total_gas_cost :
  let cost_station_1 := tank_capacity * gas_price_station_1
  let cost_station_2 := tank_capacity * gas_price_station_2
  let cost_station_3 := tank_capacity * gas_price_station_3
  let cost_station_4 := tank_capacity * gas_price_station_4
  cost_station_1 + cost_station_2 + cost_station_3 + cost_station_4 = 180 :=
by
  -- Proof is skipped
  sorry

end total_gas_cost_l271_271114


namespace INPUT_is_input_statement_l271_271657

-- Define what constitutes each type of statement
def isOutputStatement (stmt : String) : Prop :=
  stmt = "PRINT"

def isInputStatement (stmt : String) : Prop :=
  stmt = "INPUT"

def isConditionalStatement (stmt : String) : Prop :=
  stmt = "THEN"

def isEndStatement (stmt : String) : Prop :=
  stmt = "END"

-- The main theorem
theorem INPUT_is_input_statement : isInputStatement "INPUT" := by
  sorry

end INPUT_is_input_statement_l271_271657


namespace binary_to_decimal_and_octal_l271_271812

theorem binary_to_decimal_and_octal (binary_input : Nat) (h : binary_input = 0b101101110) :
    binary_input == 366 ∧ (366 : Nat) == 0o66 :=
by
  sorry

end binary_to_decimal_and_octal_l271_271812


namespace painters_work_days_l271_271217

noncomputable def work_product (n : ℕ) (d : ℚ) : ℚ := n * d

theorem painters_work_days :
  (work_product 5 2 = work_product 4 (2 + 1/2)) :=
by
  sorry

end painters_work_days_l271_271217


namespace minimum_value_l271_271069

theorem minimum_value (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 1) : 
  (∃ (x : ℝ), x = a + 2*b) → (∃ (y : ℝ), y = 2*a + b) → 
  (∀ (x y : ℝ), x + y = 3 → (1/x + 4/y) ≥ 3) :=
by
  sorry

end minimum_value_l271_271069


namespace count_valid_numbers_l271_271884

def digits_set : List ℕ := [0, 2, 4, 7, 8, 9]

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_digits (digits : List ℕ) : ℕ :=
  List.sum digits

def last_two_digits_divisibility (last_two_digits : ℕ) : Prop :=
  last_two_digits % 4 = 0

def number_is_valid (digits : List ℕ) : Prop :=
  sum_digits digits % 3 = 0

theorem count_valid_numbers :
  let possible_digits := [0, 2, 4, 7, 8, 9]
  let positions := 5
  let combinations := Nat.pow (List.length possible_digits) (positions - 1)
  let last_digit_choices := [0, 4, 8]
  3888 = 3 * combinations :=
sorry

end count_valid_numbers_l271_271884


namespace odd_function_iff_l271_271477

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := x * abs (x + a) + b

theorem odd_function_iff (a b : α) : 
  (∀ x : α, f a b (-x) = -f a b x) ↔ (a^2 + b^2 = 0) :=
by
  sorry

end odd_function_iff_l271_271477


namespace zoe_total_expenditure_is_correct_l271_271775

noncomputable def zoe_expenditure : ℝ :=
  let initial_app_cost : ℝ := 5
  let monthly_fee : ℝ := 8
  let first_two_months_fee : ℝ := 2 * monthly_fee
  let yearly_cost_without_discount : ℝ := 12 * monthly_fee
  let discount : ℝ := 0.15 * yearly_cost_without_discount
  let discounted_annual_plan : ℝ := yearly_cost_without_discount - discount
  let actual_annual_plan : ℝ := discounted_annual_plan - first_two_months_fee
  let in_game_items_cost : ℝ := 10
  let discounted_in_game_items_cost : ℝ := in_game_items_cost - (0.10 * in_game_items_cost)
  let upgraded_feature_cost : ℝ := 12
  let discounted_upgraded_feature_cost : ℝ := upgraded_feature_cost - (0.10 * upgraded_feature_cost)
  initial_app_cost + first_two_months_fee + actual_annual_plan + discounted_in_game_items_cost + discounted_upgraded_feature_cost

theorem zoe_total_expenditure_is_correct : zoe_expenditure = 122.4 :=
by
  sorry

end zoe_total_expenditure_is_correct_l271_271775


namespace cake_and_tea_cost_l271_271119

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end cake_and_tea_cost_l271_271119


namespace count_powers_of_2_not_4_l271_271396

theorem count_powers_of_2_not_4 (n : ℕ) (h : n = 500000) : 
  (∑ k in finset.range 20, ite ((¬ (∃ m, 2 ^ (2 * m) = 2 ^ k)) ∧ (2 ^ k < n)) 1 0) = 9 := 
by
  sorry

end count_powers_of_2_not_4_l271_271396


namespace alley_width_l271_271406

theorem alley_width (ℓ : ℝ) (m : ℝ) (n : ℝ): ℓ * (1 / 2 + Real.cos (70 * Real.pi / 180)) = ℓ * (Real.cos (60 * Real.pi / 180)) + ℓ * (Real.cos (70 * Real.pi / 180)) := by
  sorry

end alley_width_l271_271406


namespace days_between_dates_l271_271560

-- Define the starting and ending dates
def start_date : Nat := 1990 * 365 + (19 + 2 * 31 + 28) -- March 19, 1990 (accounting for leap years before the start date)
def end_date : Nat   := 1996 * 365 + (23 + 2 * 31 + 29 + 366 * 2 + 365 * 3) -- March 23, 1996 (accounting for leap years)

-- Define the number of leap years between the dates
def leap_years : Nat := 2 -- 1992 and 1996

-- Total number of days
def total_days : Nat := (end_date - start_date + 1)

theorem days_between_dates : total_days = 2197 :=
by
  sorry

end days_between_dates_l271_271560


namespace trapezoid_shaded_area_fraction_l271_271963

-- Define a structure for the trapezoid
structure Trapezoid (A : Type) :=
(strips : list A)
(equal_width : ∀ i j, i ≠ j ∧ i ∈ strips ∧ j ∈ strips → width i = width j)
(shaded_strips : list A)
(shaded : ∀ s, s ∈ shaded_strips ↔ s ∈ strips ∧ is_shaded s)

-- Define the predicate to check if a strip is shaded
def is_shaded (s : Strip) : Prop := s ∈ shaded_strips

-- Define the problem as a theorem in Lean
theorem trapezoid_shaded_area_fraction
  (T : Trapezoid Strip)
  (h_strips : length T.strips = 7)
  (h_shaded : length T.shaded_strips = 4)
  : fraction_shaded_area T = 4 / 7 :=
begin
  sorry
end

end trapezoid_shaded_area_fraction_l271_271963


namespace a_minus_b_plus_c_eq_five_l271_271390

theorem a_minus_b_plus_c_eq_five
(a b c : ℝ)
(h1 : a + b + c = 1)
(h2 : 3 * (4 * a + 2 * b + c) = 15)
(h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a - b + c = 5 := 
by 
  sorry

end a_minus_b_plus_c_eq_five_l271_271390


namespace quadratic_solution_is_unique_l271_271514

theorem quadratic_solution_is_unique (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 2 * p + q / 2 = -p)
  (h2 : 2 * p * (q / 2) = q) :
  (p, q) = (1, -6) :=
by
  sorry

end quadratic_solution_is_unique_l271_271514


namespace probability_of_three_heads_in_eight_tosses_l271_271304

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l271_271304


namespace ribbon_cost_comparison_l271_271660

theorem ribbon_cost_comparison 
  (A : Type)
  (yellow_ribbon_cost blue_ribbon_cost : ℕ)
  (h1 : yellow_ribbon_cost = 24)
  (h2 : blue_ribbon_cost = 36) :
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n < blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n > blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n = blue_ribbon_cost / n) :=
sorry

end ribbon_cost_comparison_l271_271660


namespace ratio_r_to_pq_l271_271777

theorem ratio_r_to_pq (total : ℝ) (amount_r : ℝ) (amount_pq : ℝ) 
  (h1 : total = 9000) 
  (h2 : amount_r = 3600.0000000000005) 
  (h3 : amount_pq = total - amount_r) : 
  amount_r / amount_pq = 2 / 3 :=
by
  sorry

end ratio_r_to_pq_l271_271777


namespace carrie_pays_l271_271508

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l271_271508


namespace probability_three_heads_in_eight_tosses_l271_271336

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l271_271336


namespace probability_of_three_heads_in_eight_tosses_l271_271302

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l271_271302


namespace frank_columns_l271_271046

theorem frank_columns (people : ℕ) (brownies_per_person : ℕ) (rows : ℕ)
  (h1 : people = 6) (h2 : brownies_per_person = 3) (h3 : rows = 3) : 
  (people * brownies_per_person) / rows = 6 :=
by
  -- Proof goes here
  sorry

end frank_columns_l271_271046


namespace A_can_give_C_start_l271_271878

def canGiveStart (total_distance start_A_B start_B_C start_A_C : ℝ) :=
  (total_distance - start_A_B) / total_distance * (total_distance - start_B_C) / total_distance = 
  (total_distance - start_A_C) / total_distance

theorem A_can_give_C_start :
  canGiveStart 1000 70 139.7849462365591 200 :=
by
  sorry

end A_can_give_C_start_l271_271878


namespace product_of_roots_l271_271515

theorem product_of_roots :
  ∀ (x : ℝ), (|x|^2 - 3 * |x| - 10 = 0) →
  (∃ a b : ℝ, a ≠ b ∧ (|a| = 5 ∧ |b| = 5) ∧ a * b = -25) :=
by {
  sorry
}

end product_of_roots_l271_271515


namespace jason_initial_money_l271_271412

theorem jason_initial_money (M : ℝ) 
  (h1 : M - (M / 4 + 10 + (2 / 5 * (3 / 4 * M - 10) + 8)) = 130) : 
  M = 320 :=
by
  sorry

end jason_initial_money_l271_271412


namespace candies_bought_l271_271104

theorem candies_bought :
  ∃ (S C : ℕ), S + C = 8 ∧ 300 * S + 500 * C = 3000 ∧ C = 3 :=
by
  sorry

end candies_bought_l271_271104


namespace even_function_odd_function_neither_even_nor_odd_function_l271_271277

def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

theorem neither_even_nor_odd_function : ∀ x : ℝ, (h (-x) ≠ h x) ∧ (h (-x) ≠ -h x) :=
by
  sorry

end even_function_odd_function_neither_even_nor_odd_function_l271_271277


namespace solve_for_y_l271_271107

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l271_271107


namespace negation_exists_geq_l271_271760

theorem negation_exists_geq :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 < 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 :=
by
  sorry

end negation_exists_geq_l271_271760


namespace side_length_of_square_l271_271746

theorem side_length_of_square (m : ℕ) (a : ℕ) (hm : m = 100) (ha : a^2 = m) : a = 10 :=
by 
  sorry

end side_length_of_square_l271_271746


namespace farmer_feed_total_cost_l271_271661

/-- 
A farmer spent $35 on feed for chickens and goats. He spent 40% of the money on chicken feed, which he bought at a 50% discount off the full price, and spent the rest on goat feed, which he bought at full price. Prove that if the farmer had paid full price for both the chicken feed and the goat feed, he would have spent $49.
-/
theorem farmer_feed_total_cost
  (total_spent : ℝ := 35)
  (chicken_feed_fraction : ℝ := 0.40)
  (goat_feed_fraction : ℝ := 0.60)
  (discount : ℝ := 0.50)
  (chicken_feed_discounted : ℝ := chicken_feed_fraction * total_spent)
  (chicken_feed_full_price : ℝ := chicken_feed_discounted / (1 - discount))
  (goat_feed_full_price : ℝ := goat_feed_fraction * total_spent):
  chicken_feed_full_price + goat_feed_full_price = 49 := 
sorry

end farmer_feed_total_cost_l271_271661


namespace spent_on_accessories_l271_271598

-- Definitions based on the conditions
def original_money : ℕ := 48
def money_on_snacks : ℕ := 8
def money_left_after_purchases : ℕ := (original_money / 2) + 4

-- Proving how much Sid spent on computer accessories
theorem spent_on_accessories : ℕ :=
  original_money - (money_left_after_purchases + money_on_snacks) = 12 :=
by
  sorry

end spent_on_accessories_l271_271598


namespace abcd_inequality_l271_271545

theorem abcd_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
(h_eq : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) + (d^2 / (1 + d^2)) = 1) :
  a * b * c * d ≤ 1 / 9 :=
sorry

end abcd_inequality_l271_271545


namespace balls_in_boxes_l271_271855

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), (balls = 6) → (boxes = 3) → 
  (∃ ways : ℕ, ways = 7) :=
by
  sorry

end balls_in_boxes_l271_271855


namespace tan_4530_l271_271809

noncomputable def tan_of_angle (deg : ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem tan_4530 : tan_of_angle 4530 = -1 / Real.sqrt 3 := sorry

end tan_4530_l271_271809


namespace annie_total_spent_l271_271681

-- Define cost of a single television
def cost_per_tv : ℕ := 50
-- Define number of televisions bought
def number_of_tvs : ℕ := 5
-- Define cost of a single figurine
def cost_per_figurine : ℕ := 1
-- Define number of figurines bought
def number_of_figurines : ℕ := 10

-- Define total cost calculation
noncomputable def total_cost : ℕ :=
  number_of_tvs * cost_per_tv + number_of_figurines * cost_per_figurine

theorem annie_total_spent : total_cost = 260 := by
  sorry

end annie_total_spent_l271_271681


namespace blue_string_length_is_320_l271_271256

-- Define the lengths of the strings
def red_string_length := 8
def white_string_length := 5 * red_string_length
def blue_string_length := 8 * white_string_length

-- The main theorem to prove
theorem blue_string_length_is_320 : blue_string_length = 320 := by
  sorry

end blue_string_length_is_320_l271_271256


namespace carpet_covering_cost_l271_271793

noncomputable def carpet_cost (floor_length floor_width carpet_length carpet_width carpet_cost_per_square : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_length * carpet_width
  let num_of_squares := floor_area / carpet_area
  num_of_squares * carpet_cost_per_square

theorem carpet_covering_cost :
  carpet_cost 6 10 2 2 15 = 225 :=
by
  sorry

end carpet_covering_cost_l271_271793


namespace six_digit_numbers_with_at_least_two_zeros_l271_271847

theorem six_digit_numbers_with_at_least_two_zeros :
  let total_numbers := 900000 in
  let no_zeros := 9^6 in
  let exactly_one_zero := 6 * 9^5 in
  total_numbers - no_zeros - exactly_one_zero = 14265 :=
by
  let total_numbers := 900000
  let no_zeros := 9^6
  let exactly_one_zero := 6 * 9^5
  show total_numbers - no_zeros - exactly_one_zero = 14265
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271847


namespace coin_toss_probability_l271_271325

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l271_271325


namespace inequality_proof_l271_271833

variable (x y z : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (hz : 0 < z)

theorem inequality_proof :
  (x + 1) / (y + 1) + (y + 1) / (z + 1) + (z + 1) / (x + 1) ≤ x / y + y / z + z / x :=
sorry

end inequality_proof_l271_271833


namespace probability_three_heads_in_eight_tosses_l271_271298

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l271_271298


namespace power_comparison_l271_271258

theorem power_comparison : (5 : ℕ) ^ 30 < (3 : ℕ) ^ 50 ∧ (3 : ℕ) ^ 50 < (4 : ℕ) ^ 40 := by
  sorry

end power_comparison_l271_271258


namespace quadratic_expression_l271_271762

theorem quadratic_expression (b c : ℤ) : 
  (∀ x : ℝ, (x^2 - 20*x + 49 = (x + b)^2 + c)) → (b + c = -61) :=
by
  sorry

end quadratic_expression_l271_271762


namespace incenter_closest_to_median_l271_271614

variables (a b c : ℝ) (s_a s_b s_c d_a d_b d_c : ℝ)

noncomputable def median_length (a b c : ℝ) : ℝ := 
  Real.sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

noncomputable def distance_to_median (x y median_length : ℝ) : ℝ := 
  (y - x) / (2 * median_length)

theorem incenter_closest_to_median
  (h₀ : a = 4) (h₁ : b = 5) (h₂ : c = 8) 
  (h₃ : s_a = median_length a b c)
  (h₄ : s_b = median_length b a c)
  (h₅ : s_c = median_length c a b)
  (h₆ : d_a = distance_to_median b c s_a)
  (h₇ : d_b = distance_to_median a c s_b)
  (h₈ : d_c = distance_to_median a b s_c) : 
  d_a = d_c := 
sorry

end incenter_closest_to_median_l271_271614


namespace triangular_number_30_l271_271032

theorem triangular_number_30 : (30 * (30 + 1)) / 2 = 465 :=
by
  sorry

end triangular_number_30_l271_271032


namespace inequality_proof_l271_271381

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^4 * b^b * c^c ≥ a⁻¹ * b⁻¹ * c⁻¹ :=
sorry

end inequality_proof_l271_271381


namespace people_left_is_10_l271_271497

def initial_people : ℕ := 12
def people_joined : ℕ := 15
def final_people : ℕ := 17
def people_left := initial_people - final_people + people_joined

theorem people_left_is_10 : people_left = 10 :=
by sorry

end people_left_is_10_l271_271497


namespace solve_equation_l271_271247

theorem solve_equation : ∀ x : ℝ, 2 * x - 6 = 3 * x * (x - 3) ↔ (x = 3 ∨ x = 2 / 3) := by sorry

end solve_equation_l271_271247


namespace total_turtles_l271_271804

theorem total_turtles (G H L : ℕ) (h_G : G = 800) (h_H : H = 2 * G) (h_L : L = 3 * G) : G + H + L = 4800 :=
by
  sorry

end total_turtles_l271_271804


namespace simplify_polynomial_l271_271770

theorem simplify_polynomial (x : ℝ) :
  (5 - 5 * x - 10 * x^2 + 10 + 15 * x - 20 * x^2 - 10 + 20 * x + 30 * x^2) = 5 + 30 * x :=
  by sorry

end simplify_polynomial_l271_271770


namespace asymptote_sum_l271_271687

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

def holes := 0 -- a
def vertical_asymptotes := 2 -- b
def horizontal_asymptotes := 1 -- c
def oblique_asymptotes := 0 -- d

theorem asymptote_sum : holes + 2 * vertical_asymptotes + 3 * horizontal_asymptotes + 4 * oblique_asymptotes = 7 :=
by
  unfold holes vertical_asymptotes horizontal_asymptotes oblique_asymptotes
  norm_num

end asymptote_sum_l271_271687


namespace angle_sum_unique_l271_271206

theorem angle_sum_unique (α β : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : β ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan α + Real.tan β - Real.tan α * Real.tan β + 1 = 0) : 
  α + β = 7 * π / 4 :=
sorry

end angle_sum_unique_l271_271206


namespace carrie_pays_94_l271_271510

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l271_271510


namespace set_intersection_complement_l271_271717

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem set_intersection_complement :
  (compl A ∩ B) = {x | 0 < x ∧ x ≤ 3} :=
by
  sorry

end set_intersection_complement_l271_271717


namespace number_of_sets_without_perfect_squares_l271_271092

/-- Define the set T_i of all integers n such that 200i ≤ n < 200(i + 1). -/
def T (i : ℕ) : Set ℕ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

/-- The total number of sets T_i from T_0 to T_{499}. -/
def total_sets : ℕ := 500

/-- The number of sets from T_0 to T_{499} that contain at least one perfect square. -/
def sets_with_perfect_squares : ℕ := 317

/-- The number of sets from T_0 to T_{499} that do not contain any perfect squares. -/
def sets_without_perfect_squares : ℕ := total_sets - sets_with_perfect_squares

/-- Proof that the number of sets T_0, T_1, T_2, ..., T_{499} that do not contain a perfect square is 183. -/
theorem number_of_sets_without_perfect_squares : sets_without_perfect_squares = 183 :=
by
  sorry

end number_of_sets_without_perfect_squares_l271_271092


namespace dice_same_number_probability_l271_271638

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l271_271638


namespace min_tan_expression_l271_271834

open Real

theorem min_tan_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
(h_eq : sin α * cos β - 2 * cos α * sin β = 0) :
  ∃ x, x = tan (2 * π + α) + tan (π / 2 - β) ∧ x = 2 * sqrt 2 :=
sorry

end min_tan_expression_l271_271834


namespace initial_spiders_correct_l271_271484

-- Define the initial number of each type of animal
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5

-- Conditions about the changes in the number of animals
def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

-- Number of animals left in the store
def total_animals_left : Nat := 25

-- Define the remaining animals after sales and adoptions
def remaining_birds : Nat := initial_birds - birds_sold
def remaining_puppies : Nat := initial_puppies - puppies_adopted
def remaining_cats : Nat := initial_cats

-- Define the remaining animals excluding spiders
def animals_without_spiders : Nat := remaining_birds + remaining_puppies + remaining_cats

-- Define the number of remaining spiders
def remaining_spiders : Nat := total_animals_left - animals_without_spiders

-- Prove the initial number of spiders
def initial_spiders : Nat := remaining_spiders + spiders_loose

theorem initial_spiders_correct :
  initial_spiders = 15 := by 
  sorry

end initial_spiders_correct_l271_271484


namespace kim_earrings_l271_271091

-- Define the number of pairs of earrings on the first day E as a variable
variable (E : ℕ)

-- Define the total number of gumballs Kim receives based on the earrings she brings each day
def total_gumballs_received (E : ℕ) : ℕ :=
  9 * E + 9 * 2 * E + 9 * (2 * E - 1)

-- Define the total number of gumballs Kim eats in 42 days
def total_gumballs_eaten : ℕ :=
  3 * 42

-- Define the statement to be proved
theorem kim_earrings : 
  total_gumballs_received E = total_gumballs_eaten + 9 → E = 3 :=
by sorry

end kim_earrings_l271_271091


namespace probability_at_least_four_same_face_l271_271039

theorem probability_at_least_four_same_face :
  let total_outcomes := (2 : ℕ) ^ 5,
      favorable_outcomes := 1 + 1 + (Nat.choose 5 1) + (Nat.choose 5 1),
      probability := favorable_outcomes / total_outcomes in
  probability = (3 : ℚ) / 8 :=
by
  sorry

end probability_at_least_four_same_face_l271_271039


namespace toys_calculation_l271_271579

-- Define the number of toys each person has as variables
variables (Jason John Rachel : ℕ)

-- State the conditions
variables (h1 : Jason = 3 * John)
variables (h2 : John = Rachel + 6)
variables (h3 : Jason = 21)

-- Define the theorem to prove the number of toys Rachel has
theorem toys_calculation : Rachel = 1 :=
by {
  sorry
}

end toys_calculation_l271_271579


namespace no_tiling_possible_with_given_dimensions_l271_271576

theorem no_tiling_possible_with_given_dimensions :
  ¬(∃ (n : ℕ), n * (2 * 2 * 1) = (3 * 4 * 5) ∧ 
   (∀ i j k : ℕ, i * 2 = 3 ∨ i * 2 = 4 ∨ i * 2 = 5) ∧
   (∀ i j k : ℕ, j * 2 = 3 ∨ j * 2 = 4 ∨ j * 2 = 5) ∧
   (∀ i j k : ℕ, k * 1 = 3 ∨ k * 1 = 4 ∨ k * 1 = 5)) :=
sorry

end no_tiling_possible_with_given_dimensions_l271_271576


namespace symmetric_point_Q_l271_271572

-- Definitions based on conditions
def P : ℝ × ℝ := (-3, 2)
def symmetric_with_respect_to_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.fst, -point.snd)

-- Theorem stating that the coordinates of point Q (symmetric to P with respect to the x-axis) are (-3, -2)
theorem symmetric_point_Q : symmetric_with_respect_to_x_axis P = (-3, -2) := 
sorry

end symmetric_point_Q_l271_271572


namespace earliest_meeting_time_l271_271351

theorem earliest_meeting_time
    (charlie_lap : ℕ := 5)
    (ben_lap : ℕ := 8)
    (laura_lap_effective : ℕ := 11) :
    lcm (lcm charlie_lap ben_lap) laura_lap_effective = 440 := by
  sorry

end earliest_meeting_time_l271_271351


namespace find_x_l271_271368

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_x (x : ℝ) : 
  (sqrt x / sqrt 0.81 + sqrt 1.44 / sqrt 0.49 = 3.0751133491652576) → 
  x = 1.5 :=
by { sorry }

end find_x_l271_271368


namespace probability_three_heads_in_eight_tosses_l271_271299

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l271_271299


namespace sum_of_lengths_of_two_sides_l271_271768

open Real

noncomputable def triangle_sum_of_two_sides (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if A + B + C = 180 ∧ A = 50 ∧ C = 40 ∧ c = 8 * sqrt 3 then
    let b := c * (sin (B * pi / 180)) / (sin (C * pi / 180))
    let a := c * (sin (A * pi / 180)) / (sin (C * pi / 180))
    a + b
  else
    0

theorem sum_of_lengths_of_two_sides : triangle_sum_of_two_sides 24.5 20.6 (8 * sqrt 3) 50 90 40 = 45.1 := by
  sorry

end sum_of_lengths_of_two_sides_l271_271768


namespace polygon_sides_l271_271057

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l271_271057


namespace range_of_m_l271_271403

theorem range_of_m (x m : ℝ) :
  (∀ x, (x - 1) / 2 ≥ (x - 2) / 3 → 2 * x - m ≥ x → x ≥ m) ↔ m ≥ -1 := by
  sorry

end range_of_m_l271_271403


namespace part_a_constant_part_b_inequality_l271_271428

open Real

noncomputable def cubic_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem part_a_constant (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1 * x2 / x3^2) + cubic_root (x2 * x3 / x1^2) + cubic_root (x3 * x1 / x2^2)) = 
  const_value := sorry

theorem part_b_inequality (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1^2 / (x2 * x3)) + cubic_root (x2^2 / (x3 * x1)) + cubic_root (x3^2 / (x1 * x2))) < (-15 / 4) := sorry

end part_a_constant_part_b_inequality_l271_271428


namespace students_with_uncool_parents_correct_l271_271456

def total_students : ℕ := 30
def cool_dads : ℕ := 12
def cool_moms : ℕ := 15
def cool_both : ℕ := 9

def students_with_uncool_parents : ℕ :=
  total_students - (cool_dads + cool_moms - cool_both)

theorem students_with_uncool_parents_correct :
  students_with_uncool_parents = 12 := by
  sorry

end students_with_uncool_parents_correct_l271_271456


namespace max_weight_of_flock_l271_271446

def MaxWeight (A E Af: ℕ): ℕ := A * 5 + E * 10 + Af * 15

theorem max_weight_of_flock :
  ∀ (A E Af: ℕ),
    A = 2 * E →
    Af = 3 * A →
    A + E + Af = 120 →
    MaxWeight A E Af = 1415 :=
by
  sorry

end max_weight_of_flock_l271_271446


namespace hyperbola_eccentricity_l271_271124

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h2 : ∀ c : ℝ, c - a^2 / c = 2 * a) :
  e = 1 + Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l271_271124


namespace max_colors_404_max_colors_406_l271_271621

theorem max_colors_404 (n k : ℕ) (h1 : n = 404) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

theorem max_colors_406 (n k : ℕ) (h1 : n = 406) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

end max_colors_404_max_colors_406_l271_271621


namespace sam_has_8_marbles_l271_271437

theorem sam_has_8_marbles :
  ∀ (steve sam sally : ℕ),
  sam = 2 * steve →
  sally = sam - 5 →
  steve + 3 = 10 →
  sam - 6 = 8 :=
by
  intros steve sam sally
  intros h1 h2 h3
  sorry

end sam_has_8_marbles_l271_271437


namespace cost_of_ice_cream_l271_271161

/-- Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups. 
    The cost of each chapati is Rs. 6, that of each plate of rice is Rs. 45, and that of mixed 
    vegetable is Rs. 70. Alok paid the cashier Rs. 931. Prove the cost of each ice-cream cup is Rs. 20. -/
theorem cost_of_ice_cream (n_chapatis n_rice n_vegetable n_ice_cream : ℕ) 
    (cost_chapati cost_rice cost_vegetable total_paid : ℕ)
    (h_chapatis : n_chapatis = 16) 
    (h_rice : n_rice = 5)
    (h_vegetable : n_vegetable = 7)
    (h_ice_cream : n_ice_cream = 6)
    (h_cost_chapati : cost_chapati = 6)
    (h_cost_rice : cost_rice = 45)
    (h_cost_vegetable : cost_vegetable = 70)
    (h_total_paid : total_paid = 931) :
    (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetable * cost_vegetable)) / n_ice_cream = 20 := 
by
  sorry

end cost_of_ice_cream_l271_271161


namespace minimum_and_maximum_S_l271_271734

theorem minimum_and_maximum_S (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) :
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * a^2 - 3 * b^2 - 3 * c^2 - 3 * d^2 = 7.5 :=
sorry

end minimum_and_maximum_S_l271_271734


namespace value_of_3b_minus_a_l271_271252

theorem value_of_3b_minus_a :
  ∃ (a b : ℕ), (a > b) ∧ (a >= 0) ∧ (b >= 0) ∧ (∀ x : ℝ, (x - a) * (x - b) = x^2 - 16 * x + 60) ∧ (3 * b - a = 8) := 
sorry

end value_of_3b_minus_a_l271_271252


namespace pallets_of_paper_cups_l271_271675

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end pallets_of_paper_cups_l271_271675


namespace possible_values_of_a_l271_271271

theorem possible_values_of_a (a b c : ℝ) (h1 : a + b + c = 2005) (h2 : (a - 1 = a ∨ a - 1 = b ∨ a - 1 = c) ∧ (b + 1 = a ∨ b + 1 = b ∨ b + 1 = c) ∧ (c ^ 2 = a ∨ c ^ 2 = b ∨ c ^ 2 = c)) :
  a = 1003 ∨ a = 1002.5 :=
sorry

end possible_values_of_a_l271_271271


namespace multiple_of_2_and_3_is_divisible_by_6_l271_271992

theorem multiple_of_2_and_3_is_divisible_by_6 (n : ℤ) (h1 : n % 2 = 0) (h2 : n % 3 = 0) : n % 6 = 0 :=
sorry

end multiple_of_2_and_3_is_divisible_by_6_l271_271992


namespace remainder_when_divided_by_22_l271_271664

theorem remainder_when_divided_by_22 
    (y : ℤ) 
    (h : y % 264 = 42) :
    y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l271_271664


namespace cos_neg_pi_over_3_l271_271696

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l271_271696


namespace probability_of_three_heads_in_eight_tosses_l271_271318

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l271_271318


namespace unique_element_set_l271_271911

theorem unique_element_set (a : ℝ) : 
  (∃! x, (a - 1) * x^2 + 3 * x - 2 = 0) ↔ (a = 1 ∨ a = -1 / 8) :=
by sorry

end unique_element_set_l271_271911


namespace same_number_probability_four_dice_l271_271642

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l271_271642


namespace probability_of_three_heads_in_eight_tosses_l271_271315

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l271_271315


namespace average_of_remaining_two_numbers_l271_271116

theorem average_of_remaining_two_numbers (A B C D E : ℝ) 
  (h1 : A + B + C + D + E = 50) 
  (h2 : A + B + C = 12) : 
  (D + E) / 2 = 19 :=
by
  sorry

end average_of_remaining_two_numbers_l271_271116


namespace lines_parallel_iff_a_eq_1_l271_271055

theorem lines_parallel_iff_a_eq_1 (x y a : ℝ) :
    (a = 1 ↔ ∃ k : ℝ, ∀ x y : ℝ, a*x + y - 1 = k*(x + a*y + 1)) :=
sorry

end lines_parallel_iff_a_eq_1_l271_271055


namespace dice_probability_same_face_l271_271629

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l271_271629


namespace parabola_bisects_rectangle_l271_271030
open Real

theorem parabola_bisects_rectangle (a : ℝ) (h_pos : a > 0) : 
  ((a^3 + a) / 2 = (a^3 / 3 + a)) → a = sqrt 3 := by
  sorry

end parabola_bisects_rectangle_l271_271030


namespace Bulgaria_f_1992_divisibility_l271_271823

def f (m n : ℕ) : ℕ := m^(3^(4 * n) + 6) - m^(3^(4 * n) + 4) - m^5 + m^3

theorem Bulgaria_f_1992_divisibility (n : ℕ) (m : ℕ) :
  ( ∀ m : ℕ, m > 0 → f m n ≡ 0 [MOD 1992] ) ↔ ( n % 2 = 1 ) :=
by
  sorry

end Bulgaria_f_1992_divisibility_l271_271823


namespace find_x_l271_271473

theorem find_x (x : ℕ) (h : 2^x - 2^(x-2) = 3 * 2^(12)) : x = 14 :=
sorry

end find_x_l271_271473


namespace total_amount_distributed_l271_271016

def number_of_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem total_amount_distributed : (number_of_persons * amount_per_person) = 42900 := by
  sorry

end total_amount_distributed_l271_271016


namespace radius_of_spheres_in_cone_l271_271575

theorem radius_of_spheres_in_cone :
  ∀ (r : ℝ),
    let base_radius := 6
    let height := 15
    let distance_from_vertex := (2 * Real.sqrt 3 / 3) * r
    let total_height := height - r
    (total_height = distance_from_vertex) →
    r = 27 - 6 * Real.sqrt 3 :=
by
  intros r base_radius height distance_from_vertex total_height H
  sorry -- The proof of the theorem will be filled here.

end radius_of_spheres_in_cone_l271_271575


namespace octagon_diagonals_l271_271392

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l271_271392


namespace smallest_x_abs_eq_29_l271_271820

theorem smallest_x_abs_eq_29 : ∃ x: ℝ, |4*x - 5| = 29 ∧ (∀ y: ℝ, |4*y - 5| = 29 → -6 ≤ y) :=
by
  sorry

end smallest_x_abs_eq_29_l271_271820


namespace correct_calculation_l271_271145

theorem correct_calculation :
  - (1 / 2) - (- (1 / 3)) = - (1 / 6) :=
by
  sorry

end correct_calculation_l271_271145


namespace square_tablecloth_side_length_l271_271156

theorem square_tablecloth_side_length (area : ℝ) (h : area = 5) : ∃ a : ℝ, a > 0 ∧ a * a = 5 := 
by
  use Real.sqrt 5
  constructor
  · apply Real.sqrt_pos.2; linarith
  · exact Real.mul_self_sqrt (by linarith [h])

end square_tablecloth_side_length_l271_271156


namespace ratio_of_buckets_l271_271682

theorem ratio_of_buckets 
  (shark_feed_per_day : ℕ := 4)
  (dolphin_feed_per_day : ℕ := shark_feed_per_day / 2)
  (total_buckets : ℕ := 546)
  (days_in_weeks : ℕ := 3 * 7)
  (ratio_R : ℕ) :
  (total_buckets = days_in_weeks * (shark_feed_per_day + dolphin_feed_per_day + (ratio_R * shark_feed_per_day)) → ratio_R = 5) := sorry

end ratio_of_buckets_l271_271682


namespace worker_schedule_l271_271160

open Nat

theorem worker_schedule (x : ℕ) :
  24 * 3 + (15 - 3) * x > 408 :=
by
  sorry

end worker_schedule_l271_271160


namespace total_number_of_balls_in_fish_tank_l271_271269

-- Definitions as per conditions
def num_goldfish := 3
def num_platyfish := 10
def red_balls_per_goldfish := 10
def white_balls_per_platyfish := 5

-- Theorem statement
theorem total_number_of_balls_in_fish_tank : 
  (num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish) = 80 := 
by
  sorry

end total_number_of_balls_in_fish_tank_l271_271269


namespace marcie_cups_coffee_l271_271242

theorem marcie_cups_coffee (S M T : ℕ) (h1 : S = 6) (h2 : S + M = 8) : M = 2 :=
by
  sorry

end marcie_cups_coffee_l271_271242


namespace proof_system_solution_l271_271897

noncomputable def solve_system : Prop :=
  ∃ x y : ℚ, x + 4 * y = 14 ∧ (x - 3) / 4 - (y - 3) / 3 = 1 / 12 ∧ x = 3 ∧ y = 11 / 4

theorem proof_system_solution : solve_system :=
sorry

end proof_system_solution_l271_271897


namespace jack_turn_in_correct_amount_l271_271411

-- Definition of the conditions
def exchange_rate_euro : ℝ := 1.18
def exchange_rate_pound : ℝ := 1.39

def till_usd_total : ℝ := (2 * 100) + (1 * 50) + (5 * 20) + (3 * 10) + (7 * 5) + (27 * 1) + (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def till_euro_total : ℝ := 20 * 5
def till_pound_total : ℝ := 25 * 10

def till_usd : ℝ := till_usd_total + (till_euro_total * exchange_rate_euro) + (till_pound_total * exchange_rate_pound)

def leave_in_till_notes : ℝ := 300
def leave_in_till_coins : ℝ := (42 * 0.25) + (19 * 0.1) + (36 * 0.05) + (47 * 0.01)
def leave_in_till_total : ℝ := leave_in_till_notes + leave_in_till_coins

def turn_in_to_office : ℝ := till_usd - leave_in_till_total

theorem jack_turn_in_correct_amount : turn_in_to_office = 607.50 := by
  sorry

end jack_turn_in_correct_amount_l271_271411


namespace tourists_walking_speed_l271_271342

-- Define the conditions
def tourists_start_time := 3 + 10 / 60 -- 3:10 A.M.
def bus_pickup_time := 5 -- 5:00 A.M.
def bus_speed := 60 -- 60 km/h
def early_arrival := 20 / 60 -- 20 minutes earlier

-- This is the Lean 4 theorem statement
theorem tourists_walking_speed : 
  (bus_speed * (10 / 60) / (100 / 60)) = 6 := 
by
  sorry

end tourists_walking_speed_l271_271342


namespace find_a_l271_271551

noncomputable def center_radius_circle1 (x y : ℝ) := x^2 + y^2 = 16
noncomputable def center_radius_circle2 (x y a : ℝ) := (x - a)^2 + y^2 = 1
def centers_tangent (a : ℝ) : Prop := |a| = 5 ∨ |a| = 3

theorem find_a (a : ℝ) (h1 : center_radius_circle1 x y) (h2 : center_radius_circle2 x y a) : centers_tangent a :=
sorry

end find_a_l271_271551


namespace factor_expression_l271_271816

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) :=
by
  sorry

end factor_expression_l271_271816


namespace coin_toss_probability_l271_271323

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l271_271323


namespace households_using_all_three_brands_correct_l271_271003

noncomputable def total_households : ℕ := 5000
noncomputable def non_users : ℕ := 1200
noncomputable def only_X : ℕ := 800
noncomputable def only_Y : ℕ := 600
noncomputable def only_Z : ℕ := 300

-- Let A be the number of households that used all three brands of soap
variable (A : ℕ)

-- For every household that used all three brands, 5 used only two brands and 10 used just one brand.
-- Number of households that used only two brands = 5 * A
-- Number of households that used only one brand = 10 * A

-- The equation for households that used just one brand:
def households_using_all_three_brands :=
10 * A = only_X + only_Y + only_Z

theorem households_using_all_three_brands_correct :
  (total_households - non_users = only_X + only_Y + only_Z + 5 * A + 10 * A) →
  (A = 170) := by
sorry

end households_using_all_three_brands_correct_l271_271003


namespace tall_students_proof_l271_271761

variables (T : ℕ) (Short Average Tall : ℕ)

-- Given in the problem:
def total_students := T = 400
def short_students := Short = 2 * T / 5
def average_height_students := Average = 150

-- Prove:
theorem tall_students_proof (hT : total_students T) (hShort : short_students T Short) (hAverage : average_height_students Average) :
  Tall = T - (Short + Average) :=
by
  sorry

end tall_students_proof_l271_271761


namespace probability_first_9_second_diamond_third_7_l271_271918

/-- 
There are 52 cards in a standard deck, with 4 cards that are 9's, 4 cards that are 7's, and 13 cards that are 
diamonds. To find the probability that the first card is a 9, the second card is a diamond, and the third card 
is a 7, we perform a detailed probabilistic calculation. In the end, we combine the probabilities of the 
mutually exclusive cases to find the desired probability.
-/
theorem probability_first_9_second_diamond_third_7 :
  (4 / 52) * (12 / 51) * (4 / 50) +
  (4 / 52) * (1 / 51) * (3 / 50) +
  (1 / 52) * (11 / 51) * (4 / 50) +
  (1 / 52) * (1 / 51) * (3 / 50) = 251 / 132600 :=
by
  sorry

end probability_first_9_second_diamond_third_7_l271_271918


namespace first_group_person_count_l271_271443

theorem first_group_person_count
  (P : ℕ)
  (h1 : P * 24 * 5 = 30 * 26 * 6) : 
  P = 39 :=
by
  sorry

end first_group_person_count_l271_271443


namespace line_intersects_parabola_exactly_one_point_l271_271367

theorem line_intersects_parabola_exactly_one_point (k : ℝ) :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 10 = k) ∧
  (∀ y z : ℝ, -3 * y^2 - 4 * y + 10 = k ∧ -3 * z^2 - 4 * z + 10 = k → y = z) 
  → k = 34 / 3 :=
by
  sorry

end line_intersects_parabola_exactly_one_point_l271_271367


namespace evaluate_expression_l271_271174

theorem evaluate_expression : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by sorry

end evaluate_expression_l271_271174


namespace exists_unique_subset_X_l271_271169

theorem exists_unique_subset_X :
  ∃ (X : Set ℤ), ∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n :=
sorry

end exists_unique_subset_X_l271_271169


namespace probability_at_least_four_same_face_l271_271037

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l271_271037


namespace ryan_distance_correct_l271_271978

-- Definitions of the conditions
def billy_distance : ℝ := 30
def madison_distance : ℝ := billy_distance * 1.2
def ryan_distance : ℝ := madison_distance * 0.5

-- Statement to prove
theorem ryan_distance_correct : ryan_distance = 18 := by
  sorry

end ryan_distance_correct_l271_271978


namespace problem1_problem2_prob_dist_problem2_expectation_l271_271625

noncomputable def probability_A_wins_match_B_wins_once (pA pB : ℚ) : ℚ :=
  (pB * pA * pA) + (pA * pB * pA * pA)

theorem problem1 : probability_A_wins_match_B_wins_once (2/3) (1/3) = 20/81 :=
  by sorry

noncomputable def P_X (x : ℕ) (pA pB : ℚ) : ℚ :=
  match x with
  | 2 => pA^2 + pB^2
  | 3 => pB * pA^2 + pA * pB^2
  | 4 => (pA * pB * pA * pA) + (pB * pA * pB * pB)
  | 5 => (pB * pA * pB * pA) + (pA * pB * pA * pB)
  | _ => 0

theorem problem2_prob_dist : 
  P_X 2 (2/3) (1/3) = 5/9 ∧
  P_X 3 (2/3) (1/3) = 2/9 ∧
  P_X 4 (2/3) (1/3) = 10/81 ∧
  P_X 5 (2/3) (1/3) = 8/81 :=
  by sorry

noncomputable def E_X (pA pB : ℚ) : ℚ :=
  2 * (P_X 2 pA pB) + 3 * (P_X 3 pA pB) + 
  4 * (P_X 4 pA pB) + 5 * (P_X 5 pA pB)

theorem problem2_expectation : E_X (2/3) (1/3) = 224/81 :=
  by sorry

end problem1_problem2_prob_dist_problem2_expectation_l271_271625


namespace problem_divisible_by_factors_l271_271243

theorem problem_divisible_by_factors (n : ℕ) (x : ℝ) : 
  ∃ k : ℝ, (x + 1)^(2 * n) - x^(2 * n) - 2 * x - 1 = k * x * (x + 1) * (2 * x + 1) :=
by
  sorry

end problem_divisible_by_factors_l271_271243


namespace find_number_l271_271726

theorem find_number (x : ℤ) (h : x + 2 - 3 = 7) : x = 8 :=
sorry

end find_number_l271_271726


namespace power_expression_l271_271549

variable {x : ℂ} -- Define x as a complex number

theorem power_expression (
  h : x - 1/x = 2 * Complex.I * Real.sqrt 2
) : x^(2187:ℕ) - 1/x^(2187:ℕ) = -22 * Complex.I * Real.sqrt 2 :=
by sorry

end power_expression_l271_271549


namespace carrie_pays_l271_271506

/-- Define the costs of different items --/
def shirt_cost : ℕ := 8
def pants_cost : ℕ := 18
def jacket_cost : ℕ := 60

/-- Define the quantities of different items bought by Carrie --/
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2

/-- Define the total cost calculation for Carrie --/
def total_cost : ℕ := (num_shirts * shirt_cost) + (num_pants * pants_cost) + (num_jackets * jacket_cost)

theorem carrie_pays : total_cost / 2 = 94 := 
by
  sorry

end carrie_pays_l271_271506


namespace probability_three_heads_in_eight_tosses_l271_271338

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l271_271338


namespace cube_volume_edge_length_range_l271_271212

theorem cube_volume_edge_length_range (a : ℝ) (h : a^3 = 9) : 2 < a ∧ a < 2.5 :=
by {
    -- proof will go here
    sorry
}

end cube_volume_edge_length_range_l271_271212


namespace chess_club_boys_count_l271_271289

theorem chess_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30)
  (h2 : (2/3 : ℝ) * G + B = 18) : 
  B = 6 :=
by
  sorry

end chess_club_boys_count_l271_271289


namespace compare_f_values_l271_271556

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem compare_f_values : 
  f (-π / 4) > f 1 ∧ f 1 > f (π / 3) := 
sorry

end compare_f_values_l271_271556


namespace initial_water_amount_l271_271489

theorem initial_water_amount (E D R F I : ℕ) 
  (hE : E = 2000) 
  (hD : D = 3500) 
  (hR : R = 350 * (30 / 10))
  (hF : F = 1550) 
  (h : I - (E + D) + R = F) : 
  I = 6000 :=
by
  sorry

end initial_water_amount_l271_271489


namespace initial_investment_calculation_l271_271969

-- Define the conditions
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 2
def A : ℝ := 6050.000000000001
def one : ℝ := 1

-- The goal is to prove that the initial principal P is 5000 under these conditions
theorem initial_investment_calculation (P : ℝ) : P = 5000 :=
by
  have interest_compounded : ℝ := (one + r / n) ^ (n * t)
  have total_amount : ℝ := P * interest_compounded
  sorry

end initial_investment_calculation_l271_271969


namespace inequality_3a3_2b3_3a2b_2ab2_l271_271690

theorem inequality_3a3_2b3_3a2b_2ab2 (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  3 * a ^ 3 + 2 * b ^ 3 ≥ 3 * a ^ 2 * b + 2 * a * b ^ 2 :=
by
  sorry

end inequality_3a3_2b3_3a2b_2ab2_l271_271690


namespace missing_angle_in_convex_polygon_l271_271669

theorem missing_angle_in_convex_polygon (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 5) 
  (h2 : 180 * (n - 2) - 3 * x = 3330) : 
  x = 54 := 
by 
  sorry

end missing_angle_in_convex_polygon_l271_271669


namespace find_a_l271_271908

/-- The random variable ξ takes on all possible values 1, 2, 3, 4, 5,
and P(ξ = k) = a * k for k = 1, 2, 3, 4, 5. Given that the sum 
of probabilities for all possible outcomes of a discrete random
variable equals 1, find the value of a. -/
theorem find_a (a : ℝ) 
  (h : (a * 1) + (a * 2) + (a * 3) + (a * 4) + (a * 5) = 1) : 
  a = 1 / 15 :=
sorry

end find_a_l271_271908


namespace Carrie_pays_94_l271_271505

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l271_271505


namespace total_cups_of_ingredients_l271_271909

theorem total_cups_of_ingredients
  (ratio_butter : ℕ) (ratio_flour : ℕ) (ratio_sugar : ℕ)
  (flour_cups : ℕ)
  (h_ratio : ratio_butter = 2 ∧ ratio_flour = 3 ∧ ratio_sugar = 5)
  (h_flour : flour_cups = 6) :
  let part_cups := flour_cups / ratio_flour
  let butter_cups := ratio_butter * part_cups
  let sugar_cups := ratio_sugar * part_cups
  let total_cups := butter_cups + flour_cups + sugar_cups
  total_cups = 20 :=
by
  sorry

end total_cups_of_ingredients_l271_271909


namespace one_div_i_plus_i_pow_2015_eq_neg_two_i_l271_271048

def is_imaginary_unit (x : ℂ) : Prop := x * x = -1

theorem one_div_i_plus_i_pow_2015_eq_neg_two_i (i : ℂ) (h : is_imaginary_unit i) : 
  (1 / i + i ^ 2015) = -2 * i :=
sorry

end one_div_i_plus_i_pow_2015_eq_neg_two_i_l271_271048


namespace values_are_equal_and_differ_in_precision_l271_271781

-- We define the decimal values
def val1 : ℝ := 4.5
def val2 : ℝ := 4.50

-- We define the counting units
def unit1 : ℝ := 0.1
def unit2 : ℝ := 0.01

-- Now, we state our theorem
theorem values_are_equal_and_differ_in_precision : 
  val1 = val2 ∧ unit1 ≠ unit2 :=
by
  -- Placeholder for the proof
  sorry

end values_are_equal_and_differ_in_precision_l271_271781


namespace ratio_of_sums_l271_271546

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2
axiom a4_eq_2a3 : a 4 = 2 * a 3

theorem ratio_of_sums (a : ℕ → ℝ) (S : ℕ → ℝ)
                      (arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2)
                      (a4_eq_2a3 : a 4 = 2 * a 3) :
  S 7 / S 5 = 14 / 5 :=
by sorry

end ratio_of_sums_l271_271546


namespace greatest_multiple_of_3_lt_1000_l271_271751

theorem greatest_multiple_of_3_lt_1000 :
  ∃ (x : ℕ), (x % 3 = 0) ∧ (x > 0) ∧ (x^3 < 1000) ∧ ∀ (y : ℕ), (y % 3 = 0) ∧ (y > 0) ∧ (y^3 < 1000) → y ≤ x := 
sorry

end greatest_multiple_of_3_lt_1000_l271_271751


namespace gcd_lcm_product_l271_271363

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l271_271363


namespace probability_of_three_heads_in_eight_tosses_l271_271300

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l271_271300


namespace polygon_sides_l271_271060

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
sorry

end polygon_sides_l271_271060


namespace tangent_line_at_point_l271_271606

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

theorem tangent_line_at_point (x y : ℝ) (h : f 1 = 0) (h' : deriv f 1 = -2) :
  2 * x + y - 2 = 0 :=
sorry

end tangent_line_at_point_l271_271606


namespace urn_contains_four_red_three_blue_l271_271018

noncomputable def urn_problem : Prop :=
  let initial_red := 1
  let initial_blue := 1
  let total_operations := 5
  let final_total_balls := 7
  let desired_red := 4
  let desired_blue := 3
  let probability := 1 / 6
  (probability == sorry)  -- This is where we would compute and compare probabilities

theorem urn_contains_four_red_three_blue :
  urn_problem :=
by sorry

end urn_contains_four_red_three_blue_l271_271018


namespace neither_coffee_tea_juice_l271_271020

open Set

theorem neither_coffee_tea_juice (total : ℕ) (coffee : ℕ) (tea : ℕ) (both_coffee_tea : ℕ)
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) :
  total = 35 → 
  coffee = 18 → 
  tea = 15 → 
  both_coffee_tea = 7 → 
  juice = 6 → 
  juice_and_tea_not_coffee = 3 →
  (total - ((coffee + tea - both_coffee_tea) + (juice - juice_and_tea_not_coffee))) = 6 :=
sorry

end neither_coffee_tea_juice_l271_271020


namespace minimum_value_of_reciprocal_squares_l271_271923

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l271_271923


namespace sequence_a_n_eq_T_n_formula_C_n_formula_l271_271999

noncomputable def sequence_S (n : ℕ) : ℕ := n * (2 * n - 1)

def arithmetic_seq (n : ℕ) : ℚ := 2 * n - 1

def a_n (n : ℕ) : ℤ := 4 * n - 3

def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * n + 1)

def c_n (n : ℕ) : ℚ := 3^(n - 1)

def C_n (n : ℕ) : ℚ := (3^n - 1) / 2

theorem sequence_a_n_eq (n : ℕ) : a_n n = 4 * n - 3 := by sorry

theorem T_n_formula (n : ℕ) : T_n n = (n : ℚ) / (4 * n + 1) := by sorry

theorem C_n_formula (n : ℕ) : C_n n = (3^n - 1) / 2 := by sorry

end sequence_a_n_eq_T_n_formula_C_n_formula_l271_271999


namespace christine_needs_32_tbs_aquafaba_l271_271684

-- Definitions for the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

def total_egg_whites : ℕ := egg_whites_per_cake * number_of_cakes
def total_tbs_aquafaba : ℕ := tablespoons_per_egg_white * total_egg_whites

-- Theorem statement
theorem christine_needs_32_tbs_aquafaba :
  total_tbs_aquafaba = 32 :=
by sorry

end christine_needs_32_tbs_aquafaba_l271_271684


namespace Elmer_eats_more_than_Penelope_l271_271431

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l271_271431


namespace jeff_ends_at_multiple_of_4_l271_271087

open Classical

noncomputable def prob_end_multiple_of_4 : ℚ :=
  let prob_picking_4_8_12 := (4 / 15: ℚ)
  let prob_picking_6_10_14 := (3 / 15: ℚ)
  let prob_picking_2_14 := (2 / 15: ℚ)
  let prob_spin_SS_RL_LR := 1 / 3 * 1 / 3 * 3
  let prob_spin_LL := 1 / 3 * 1 / 3
  let prob_spin_RR := 1 / 3 * 1 / 3
  prob_picking_4_8_12 * prob_spin_SS_RL_LR + prob_picking_6_10_14 * prob_spin_LL + prob_picking_2_14 * prob_spin_RR

theorem jeff_ends_at_multiple_of_4 : prob_end_multiple_of_4 = 17 / 135 := sorry

end jeff_ends_at_multiple_of_4_l271_271087


namespace hilary_stalks_l271_271391

-- Define the given conditions
def ears_per_stalk : ℕ := 4
def kernels_per_ear_first_half : ℕ := 500
def kernels_per_ear_second_half : ℕ := 600
def total_kernels : ℕ := 237600

-- Average number of kernels per ear
def average_kernels_per_ear : ℕ := (kernels_per_ear_first_half + kernels_per_ear_second_half) / 2

-- Total number of ears based on total kernels
noncomputable def total_ears : ℕ := total_kernels / average_kernels_per_ear

-- Total number of stalks based on total ears
noncomputable def total_stalks : ℕ := total_ears / ears_per_stalk

-- The main theorem to prove
theorem hilary_stalks : total_stalks = 108 :=
by
  sorry

end hilary_stalks_l271_271391


namespace dice_same_number_probability_l271_271636

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l271_271636


namespace find_number_l271_271659

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def XiaoQian_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ n < 5

def XiaoLu_statements (n : ℕ) : Prop :=
  n < 7 ∧ 10 ≤ n ∧ n < 100

def XiaoDai_statements (n : ℕ) : Prop :=
  is_perfect_square n ∧ ¬ (n < 5)

theorem find_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 99 ∧ 
    ( (XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ XiaoLu_statements n ∧ ¬XiaoDai_statements n) ∨
      (¬XiaoQian_statements n ∧ ¬XiaoLu_statements n ∧ XiaoDai_statements n) ) ∧
    n = 9 :=
sorry

end find_number_l271_271659


namespace range_of_a_l271_271384

open Real

theorem range_of_a (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |2^x₁ - a| = 1 ∧ |2^x₂ - a| = 1) ↔ 1 < a :=
by 
    sorry

end range_of_a_l271_271384


namespace same_number_on_four_dice_l271_271650

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l271_271650


namespace fraction_of_arith_geo_seq_l271_271544

theorem fraction_of_arith_geo_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_seq_arith : ∀ n, a (n+1) = a n + d)
  (h_seq_geo : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by
  sorry

end fraction_of_arith_geo_seq_l271_271544


namespace problem_statement_l271_271715

noncomputable def f (x : ℝ) : ℝ := x / Real.cos x

theorem problem_statement (x1 x2 x3 : ℝ) (h1 : abs x1 < Real.pi / 2)
                         (h2 : abs x2 < Real.pi / 2) (h3 : abs x3 < Real.pi / 2)
                         (c1 : f x1 + f x2 ≥ 0) (c2 : f x2 + f x3 ≥ 0) (c3 : f x3 + f x1 ≥ 0) :
  f (x1 + x2 + x3) ≥ 0 :=
sorry

end problem_statement_l271_271715


namespace cube_greater_l271_271995

theorem cube_greater (a b : ℝ) (h : a > b) : a^3 > b^3 := 
sorry

end cube_greater_l271_271995


namespace area_of_one_cookie_l271_271278

theorem area_of_one_cookie (L W : ℝ)
    (W_eq_15 : W = 15)
    (circumference_condition : 4 * L + 2 * W = 70) :
    L * W = 150 :=
by
  sorry

end area_of_one_cookie_l271_271278


namespace abigail_money_loss_l271_271802

theorem abigail_money_loss {initial spent remaining lost : ℤ} 
  (h1 : initial = 11) 
  (h2 : spent = 2) 
  (h3 : remaining = 3) 
  (h4 : lost = initial - spent - remaining) : 
  lost = 6 := sorry

end abigail_money_loss_l271_271802


namespace a_2009_eq_1_a_2014_eq_0_l271_271185

section
variable (a : ℕ → ℕ)
variable (n : ℕ)

-- Condition 1: a_{4n-3} = 1
axiom cond1 : ∀ n : ℕ, a (4 * n - 3) = 1

-- Condition 2: a_{4n-1} = 0
axiom cond2 : ∀ n : ℕ, a (4 * n - 1) = 0

-- Condition 3: a_{2n} = a_n
axiom cond3 : ∀ n : ℕ, a (2 * n) = a n

-- Theorem: a_{2009} = 1
theorem a_2009_eq_1 : a 2009 = 1 := by
  sorry

-- Theorem: a_{2014} = 0
theorem a_2014_eq_0 : a 2014 = 0 := by
  sorry

end

end a_2009_eq_1_a_2014_eq_0_l271_271185


namespace correct_option_l271_271466

theorem correct_option :
  (∀ a : ℝ, a ≠ 0 → (a ^ 0 = 1)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → (a^6 / a^3 = a^2)) ∧
  ¬(∀ a : ℝ, a ≠ 0 → ((a^2)^3 = a^5)) ∧
  ¬(∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (a / (a + b)^2 + b / (a + b)^2 = a + b)) :=
by {
  sorry
}

end correct_option_l271_271466


namespace union_of_sets_l271_271195

-- Define the sets and conditions
variables (a b : ℝ)
variables (A : Set ℝ) (B : Set ℝ)
variables (log2 : ℝ → ℝ)

-- State the assumptions and final proof goal
theorem union_of_sets (h_inter : A ∩ B = {2}) 
                      (h_A : A = {3, log2 a}) 
                      (h_B : B = {a, b}) 
                      (h_log2 : log2 4 = 2) :
  A ∪ B = {2, 3, 4} :=
by {
    sorry
}

end union_of_sets_l271_271195


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l271_271233

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l271_271233


namespace chicken_price_reaches_81_in_2_years_l271_271080

theorem chicken_price_reaches_81_in_2_years :
  ∃ t : ℝ, (t / 12 = 2) ∧ (∃ n : ℕ, (3:ℝ)^(n / 6) = 81 ∧ n = t) :=
by
  sorry

end chicken_price_reaches_81_in_2_years_l271_271080


namespace percent_games_lost_l271_271129

theorem percent_games_lost
  (w l t : ℕ)
  (h_ratio : 7 * l = 3 * w)
  (h_tied : t = 5) :
  (l : ℝ) / (w + l + t) * 100 = 20 :=
by
  sorry

end percent_games_lost_l271_271129


namespace matchsticks_left_l271_271350

def initial_matchsticks : ℕ := 30
def matchsticks_needed_2 : ℕ := 5
def matchsticks_needed_0 : ℕ := 6
def num_2s : ℕ := 3
def num_0s : ℕ := 1

theorem matchsticks_left : 
  initial_matchsticks - (num_2s * matchsticks_needed_2 + num_0s * matchsticks_needed_0) = 9 :=
by sorry

end matchsticks_left_l271_271350


namespace minimize_rental_cost_l271_271012

def travel_agency (x y : ℕ) : ℕ := 1600 * x + 2400 * y

theorem minimize_rental_cost :
    ∃ (x y : ℕ), (x + y ≤ 21) ∧ (y ≤ x + 7) ∧ (36 * x + 60 * y = 900) ∧ 
    (∀ (a b : ℕ), (a + b ≤ 21) ∧ (b ≤ a + 7) ∧ (36 * a + 60 * b = 900) → travel_agency a b ≥ travel_agency x y) ∧
    travel_agency x y = 36800 :=
sorry

end minimize_rental_cost_l271_271012


namespace smallest_n_for_modulo_eq_l271_271170

theorem smallest_n_for_modulo_eq :
  ∃ (n : ℕ), (3^n % 4 = n^3 % 4) ∧ (∀ m : ℕ, m < n → 3^m % 4 ≠ m^3 % 4) ∧ n = 7 :=
by
  sorry

end smallest_n_for_modulo_eq_l271_271170


namespace sum_of_ages_l271_271889

-- Definitions from the problem conditions
def Maria_age : ℕ := 14
def age_difference_between_Jose_and_Maria : ℕ := 12
def Jose_age : ℕ := Maria_age + age_difference_between_Jose_and_Maria

-- To be proven: sum of their ages is 40
theorem sum_of_ages : Maria_age + Jose_age = 40 :=
by
  -- skip the proof
  sorry

end sum_of_ages_l271_271889


namespace rhombus_diagonals_not_equal_l271_271938

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l271_271938


namespace largest_e_l271_271070

variable (a b c d e : ℤ)

theorem largest_e 
  (h1 : a - 1 = b + 2) 
  (h2 : a - 1 = c - 3)
  (h3 : a - 1 = d + 4)
  (h4 : a - 1 = e - 6) 
  : e > a ∧ e > b ∧ e > c ∧ e > d := 
sorry

end largest_e_l271_271070


namespace lcm_is_multiple_of_230_l271_271255

theorem lcm_is_multiple_of_230 (d n : ℕ) (h1 : n = 230) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (2 ∣ d)) : ∃ m : ℕ, Nat.lcm d n = 230 * m :=
by
  exists 1 -- Placeholder for demonstration purposes
  sorry

end lcm_is_multiple_of_230_l271_271255


namespace octagon_diagonals_l271_271394

theorem octagon_diagonals : 
  let n := 8 in
  (n * (n - 3)) / 2 = 20 := 
by 
  let n := 8
  calc
    (n * (n - 3)) / 2 = (8 * (8 - 3)) / 2 : by rfl
                    ... = 40 / 2           : by norm_num
                    ... = 20               : by norm_num

end octagon_diagonals_l271_271394


namespace selection_including_both_genders_is_34_l271_271895

def count_ways_to_select_students_with_conditions (total_students boys girls select_students : ℕ) : ℕ :=
  if total_students = 7 ∧ boys = 4 ∧ girls = 3 ∧ select_students = 4 then
    (Nat.choose total_students select_students) - 1
  else
    0

theorem selection_including_both_genders_is_34 :
  count_ways_to_select_students_with_conditions 7 4 3 4 = 34 :=
by
  -- The proof would go here
  sorry

end selection_including_both_genders_is_34_l271_271895


namespace cubic_box_dimension_l271_271050

theorem cubic_box_dimension (a : ℤ) (h: 12 * a = 3 * (a^3)) : a = 2 :=
by
  sorry

end cubic_box_dimension_l271_271050


namespace probability_three_heads_in_eight_tosses_l271_271334

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l271_271334


namespace num_ways_to_distribute_balls_l271_271864

noncomputable def num_partitions (n k : ℕ) : ℕ :=
  (Finset.powerset (multiset.range (n + k - 1))).card

theorem num_ways_to_distribute_balls :
  num_partitions 6 3 = 6 :=
sorry

end num_ways_to_distribute_balls_l271_271864


namespace geometric_sequence_common_ratio_l271_271534

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q))  -- Sum of geometric series
  (h2 : a 3 = S 3 + 1) : q = 3 :=
by sorry

end geometric_sequence_common_ratio_l271_271534


namespace ship_speed_in_still_water_l271_271009

theorem ship_speed_in_still_water
  (x y : ℝ)
  (h1: x + y = 32)
  (h2: x - y = 28)
  (h3: x > y) : 
  x = 30 := 
sorry

end ship_speed_in_still_water_l271_271009


namespace cos_neg_pi_over_3_l271_271695

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l271_271695


namespace scientific_notation_280000_l271_271738

theorem scientific_notation_280000 : (280000 : ℝ) = 2.8 * 10^5 :=
sorry

end scientific_notation_280000_l271_271738


namespace heptagon_isosceles_same_color_l271_271983

theorem heptagon_isosceles_same_color 
  (color : Fin 7 → Prop) (red blue : Prop)
  (h_heptagon : ∀ i : Fin 7, color i = red ∨ color i = blue) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ color i = color j ∧ color j = color k ∧ ((i + j) % 7 = k ∨ (j + k) % 7 = i ∨ (k + i) % 7 = j) :=
sorry

end heptagon_isosceles_same_color_l271_271983


namespace find_range_m_l271_271193

-- Definitions of the conditions
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x + y - m = 0) ∧ ((x - 1)^2 + y^2 = 1)
def q (m : ℝ) : Prop := ∃ x : ℝ, (x^2 - x + m - 4 = 0) ∧ x ≠ 0 ∧ ∀ y : ℝ, (y^2 - y + m - 4 = 0) → x * y < 0

theorem find_range_m (m : ℝ) : (p m ∨ q m) ∧ ¬p m → (m ≤ 1 - Real.sqrt 2 ∨ 1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
by
  sorry

end find_range_m_l271_271193


namespace parabola_equation_max_slope_OQ_l271_271840

-- Definition of the problem for part (1)
theorem parabola_equation (p : ℝ) (hp : p = 2) : (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) :=
by {
  sorry
}

-- Definition of the problem for part (2)
theorem max_slope_OQ (m n : ℝ) (hp : y^2 = 4 * x)
  (h_relate : ∀ P Q F : (ℝ × ℝ), P.1 * Q.1 + P.2 * Q.2 = 9 * (Q.1 - F.1) * (Q.2 - F.2))
  : (∀ Q : (ℝ × ℝ), max (Q.2 / Q.1) = 1/3) :=
by {
  sorry
}

end parabola_equation_max_slope_OQ_l271_271840


namespace dice_probability_same_face_l271_271628

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l271_271628


namespace integer_sequence_existence_l271_271187

theorem integer_sequence_existence
  (n : ℕ) (a : ℕ → ℤ) (A B C : ℤ) 
  (h1 : (a 1 < A ∧ A < B ∧ B < a n) ∨ (a 1 > A ∧ A > B ∧ B > a n))
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n - 1 → (a (i + 1) - a i ≤ 1 ∨ a (i + 1) - a i ≥ -1))
  (h3 : A ≤ C ∧ C ≤ B ∨ A ≥ C ∧ C ≥ B) :
  ∃ i, 1 < i ∧ i < n ∧ a i = C := sorry

end integer_sequence_existence_l271_271187


namespace intersection_proof_l271_271176

-- Definitions based on conditions
def circle1 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 10) ^ 2 = 50
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 + 2 * (x - y) - 18 = 0

-- Correct answer tuple
def intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 3), (-3, 5))

-- The goal statement to prove
theorem intersection_proof :
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by
  sorry

end intersection_proof_l271_271176


namespace dimes_difference_l271_271894

theorem dimes_difference
  (a b c d : ℕ)
  (h1 : a + b + c + d = 150)
  (h2 : 5 * a + 10 * b + 25 * c + 50 * d = 1500) :
  (b = 150 ∨ ∃ c d : ℕ, b = 0 ∧ 4 * c + 9 * d = 150) →
  ∃ b₁ b₂ : ℕ, (b₁ = 150 ∧ b₂ = 0 ∧ b₁ - b₂ = 150) :=
by
  sorry

end dimes_difference_l271_271894


namespace greatest_common_multiple_9_15_less_120_l271_271142

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem greatest_common_multiple_9_15_less_120 : 
  ∃ m, (m < 120) ∧ ( ∃ k : ℕ, m = k * (lcm 9 15)) ∧ ∀ n, (n < 120) ∧ ( ∃ k : ℕ, n = k * (lcm 9 15)) → n ≤ m := 
sorry

end greatest_common_multiple_9_15_less_120_l271_271142


namespace inverse_tangent_line_l271_271209

theorem inverse_tangent_line
  (f : ℝ → ℝ)
  (hf₁ : ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x) 
  (hf₂ : ∀ x, deriv f x ≠ 0)
  (h_tangent : ∀ x₀, (2 * x₀ - f x₀ + 3) = 0) :
  ∀ x₀, (x₀ - 2 * f x₀ - 3) = 0 :=
by
  sorry

end inverse_tangent_line_l271_271209


namespace range_of_a_l271_271085

theorem range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (x-a) / (2 - (x + 1 - a)) > 0)
  ↔ -2 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l271_271085


namespace correct_option_l271_271147

noncomputable def problem_statement : Prop := 
  (sqrt 2 + sqrt 6 ≠ sqrt 8) ∧ 
  (6 * sqrt 3 - 2 * sqrt 3 ≠ 4) ∧
  (4 * sqrt 2 * 2 * sqrt 3 ≠ 6 * sqrt 6) ∧ 
  (1 / (2 - sqrt 3) = 2 + sqrt 3)

theorem correct_option : problem_statement := by
  sorry

end correct_option_l271_271147


namespace values_of_n_l271_271526

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end values_of_n_l271_271526


namespace inscribed_rectangle_area_correct_l271_271239

noncomputable def area_of_inscribed_rectangle : Prop := 
  let AD : ℝ := 15 / (12 / (1 / 3) + 3)
  let AB : ℝ := 1 / 3 * AD
  AD * AB = 25 / 12

theorem inscribed_rectangle_area_correct :
  area_of_inscribed_rectangle
  := by
  let hf : ℝ := 12
  let eg : ℝ := 15
  let ad : ℝ := 15 / (hf / (1 / 3) + 3)
  let ab : ℝ := 1 / 3 * ad
  have area : ad * ab = 25 / 12 := by sorry
  exact area

end inscribed_rectangle_area_correct_l271_271239


namespace probability_three_heads_in_eight_tosses_l271_271297

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l271_271297


namespace rectangle_area_diff_l271_271737

theorem rectangle_area_diff :
  ∀ (l w : ℕ), (2 * l + 2 * w = 60) → (∃ A_max A_min : ℕ, 
    A_max = (l * (30 - l)) ∧ A_min = (min (1 * (30 - 1)) (29 * (30 - 29))) ∧ (A_max - A_min = 196)) :=
by
  intros l w h
  use 15 * 15, min (1 * 29) (29 * 1)
  sorry

end rectangle_area_diff_l271_271737


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l271_271929

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l271_271929


namespace train_speed_l271_271011

theorem train_speed (train_length bridge_length : ℕ) (time : ℝ)
  (h_train_length : train_length = 110)
  (h_bridge_length : bridge_length = 290)
  (h_time : time = 23.998080153587715) :
  (train_length + bridge_length) / time * 3.6 = 60 := 
by
  rw [h_train_length, h_bridge_length, h_time]
  sorry

end train_speed_l271_271011


namespace calc_expression_value_l271_271500

open Real

theorem calc_expression_value :
  sqrt ((16: ℝ) ^ 12 + (8: ℝ) ^ 15) / ((16: ℝ) ^ 5 + (8: ℝ) ^ 16) = (3 * sqrt 2) / 4 := sorry

end calc_expression_value_l271_271500


namespace Jan_older_than_Cindy_l271_271352

noncomputable def Cindy_age : ℕ := 5
noncomputable def Greg_age : ℕ := 16

variables (Marcia_age Jan_age : ℕ)

axiom Greg_and_Marcia : Greg_age = Marcia_age + 2
axiom Marcia_and_Jan : Marcia_age = 2 * Jan_age

theorem Jan_older_than_Cindy : (Jan_age - Cindy_age) = 2 :=
by
  -- Insert proof here
  sorry

end Jan_older_than_Cindy_l271_271352


namespace stick_segments_l271_271487

theorem stick_segments (L : ℕ) (L_nonzero : L > 0) :
  let red_segments := 8
  let blue_segments := 12
  let black_segments := 18
  let total_segments := (red_segments + blue_segments + black_segments) 
                       - (lcm red_segments blue_segments / blue_segments) 
                       - (lcm blue_segments black_segments / black_segments)
                       - (lcm red_segments black_segments / black_segments)
                       + (lcm red_segments (lcm blue_segments black_segments) / (lcm blue_segments black_segments))
  let shortest_segment_length := L / lcm red_segments (lcm blue_segments black_segments)
  (total_segments = 28) ∧ (shortest_segment_length = L / 72) := by
  sorry

end stick_segments_l271_271487


namespace rounding_increases_value_l271_271589

theorem rounding_increases_value (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (rounded_a : ℕ := a + 1)
  (rounded_b : ℕ := b - 1)
  (rounded_c : ℕ := c + 1)
  (rounded_d : ℕ := d + 1) :
  (rounded_a * rounded_d) / rounded_b + rounded_c > (a * d) / b + c := 
sorry

end rounding_increases_value_l271_271589


namespace find_sum_of_squares_l271_271175

theorem find_sum_of_squares (x y : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x * y + x + y = 35) (h4 : x^2 * y + x * y^2 = 210) : x^2 + y^2 = 154 :=
sorry

end find_sum_of_squares_l271_271175


namespace exists_infinitely_many_rational_squares_sum_l271_271744

theorem exists_infinitely_many_rational_squares_sum (r : ℚ):
  ∃ᶠ x : ℚ in (Filter.cofinite), ∃ y : ℚ, x^2 + y^2 = r^2 :=
sorry

end exists_infinitely_many_rational_squares_sum_l271_271744


namespace sector_area_l271_271052

/-- Given a sector with a radius of 2 and a central angle of 90 degrees, the area of the sector is π. -/
theorem sector_area : 
  let r : ℝ := 2
  let alpha_degrees : ℝ := 90
  let alpha_radians : ℝ := (alpha_degrees * Real.pi) / 180
  let area : ℝ := (1 / 2) * alpha_radians * r^2
  area = Real.pi :=
by
  let r : ℝ := 2
  let alpha_degrees : ℝ := 90
  let alpha_radians : ℝ := (alpha_degrees * Real.pi) / 180
  let area : ℝ := (1 / 2) * alpha_radians * r^2
  have h_alpha : alpha_radians = Real.pi / 2 := by sorry
  have h_area : area = Real.pi := by sorry
  exact h_area

end sector_area_l271_271052


namespace palm_trees_in_forest_l271_271002

variable (F D : ℕ)

theorem palm_trees_in_forest 
  (h1 : D = 2 * F / 5)
  (h2 : D + F = 7000) :
  F = 5000 := by
  sorry

end palm_trees_in_forest_l271_271002


namespace grid_fill_existence_l271_271827

noncomputable def grid_filling (n : ℕ) : Prop :=
∃ (M : matrix (fin n) (fin n) ℤ), 
(∀ i : fin n, ∃ (s_i : ℤ), s_i = finset.sum (finset.univ) (λ j, M i j) ∧ (∀ j : fin n, ∃ (s_j : ℤ), s_j = finset.sum (finset.univ) (λ i, M i j))) ∧ 
(list.nodup (list.map (λ i, finset.sum (finset.univ) (λ j, M i j)) (finset.filter (finset.mem (finset.univ))) ∧
list.nodup (list.map (λ j, finset.sum (finset.univ) (λ i, M i j)), (finset.filter (finset.mem (finset.univ)))))

theorem grid_fill_existence (n : ℕ) : grid_filling n ↔ even n := sorry

end grid_fill_existence_l271_271827


namespace car_trader_profit_l271_271157

theorem car_trader_profit (P : ℝ) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.28000000000000004 * P
  let profit := selling_price - purchase_price
  let percentage_increase := (profit / purchase_price) * 100
  percentage_increase = 60 := 
by
  sorry

end car_trader_profit_l271_271157


namespace speed_plane_east_l271_271007

-- Definitions of the conditions
def speed_west : ℕ := 275
def time_hours : ℝ := 3.5
def distance_apart : ℝ := 2100

-- Theorem statement to prove the speed of the plane traveling due East
theorem speed_plane_east (v: ℝ) 
  (h: (v + speed_west) * time_hours = distance_apart) : 
  v = 325 :=
  sorry

end speed_plane_east_l271_271007


namespace quadratic_real_equal_roots_l271_271033

theorem quadratic_real_equal_roots (m : ℝ) :
  (∃ x : ℝ, 3*x^2 + (2*m-5)*x + 12 = 0) ↔ (m = 8.5 ∨ m = -3.5) :=
sorry

end quadratic_real_equal_roots_l271_271033


namespace smaller_integer_is_49_l271_271604

theorem smaller_integer_is_49 (m n : ℕ) (hm : 10 ≤ m ∧ m < 100) (hn : 10 ≤ n ∧ n < 100)
  (h : (m + n) / 2 = m + n / 100) : min m n = 49 :=
by
  sorry

end smaller_integer_is_49_l271_271604


namespace line_y_intercept_l271_271200

theorem line_y_intercept (t : ℝ) (h : ∃ (t : ℝ), ∀ (x y : ℝ), x - 2 * y + t = 0 → (x = 2 ∧ y = -1)) :
  ∃ y : ℝ, (0 - 2 * y + t = 0) ∧ y = -2 :=
by
  sorry

end line_y_intercept_l271_271200


namespace negation_of_existential_statement_l271_271612

theorem negation_of_existential_statement : 
  (¬∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_of_existential_statement_l271_271612


namespace algebraic_expression_value_l271_271182

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) :
  x^2 - 6 * x * y + 9 * y^2 = 4 :=
sorry

end algebraic_expression_value_l271_271182


namespace unique_solution_nat_numbers_l271_271356

theorem unique_solution_nat_numbers (a b c : ℕ) (h : 2^a + 9^b = 2 * 5^c + 5) : 
  (a, b, c) = (1, 0, 0) :=
sorry

end unique_solution_nat_numbers_l271_271356


namespace solution_correct_l271_271750

noncomputable def probability_same_color_opposite_types : ℚ :=
  let total_shoes := 30 in
  let black_pairs := 7 in
  let brown_pairs := 4 in
  let gray_pairs := 2 in
  let red_pairs := 2 in
  let total_pairs := black_pairs + brown_pairs + grayPairs + redPairs in
  let black_prob := (14 / 30) * (7 / 29) in
  let brown_prob  := (8 / 30) * (4 / 29) in
  let gray_prob   := (4 / 30) * (2 / 29) in
  let red_prob    := (4 / 30) * (2 / 29) in
  (black_prob + brown_prob + gray_prob + red_prob).reduce

theorem solution_correct : probability_same_color_opposite_types = 73 / 435 :=
by
  sorry

end solution_correct_l271_271750


namespace find_sets_A_B_l271_271732

def C : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

def S : Finset ℕ := {4, 5, 9, 14, 23, 37}

theorem find_sets_A_B :
  ∃ (A B : Finset ℕ), 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = C) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → x + y ∉ S) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ B → y ∈ B → x + y ∉ S) ∧ 
  (A = {1, 2, 5, 6, 10, 11, 14, 15, 16, 19, 20}) ∧ 
  (B = {3, 4, 7, 8, 9, 12, 13, 17, 18}) :=
by
  sorry

end find_sets_A_B_l271_271732


namespace empty_seats_after_second_stop_l271_271132

-- Definitions for the conditions described in the problem
def bus_seats : Nat := 23 * 4
def initial_people : Nat := 16
def first_stop_people_on : Nat := 15
def first_stop_people_off : Nat := 3
def second_stop_people_on : Nat := 17
def second_stop_people_off : Nat := 10

-- The theorem statement proving the number of empty seats
theorem empty_seats_after_second_stop : 
  (bus_seats - (initial_people + first_stop_people_on - first_stop_people_off + second_stop_people_on - second_stop_people_off)) = 57 :=
by
  sorry

end empty_seats_after_second_stop_l271_271132


namespace Cameron_books_proof_l271_271164

noncomputable def Cameron_initial_books :=
  let B : ℕ := 24
  let B_donated := B / 4
  let B_left := B - B_donated
  let C_donated (C : ℕ) := C / 3
  let C_left (C : ℕ) := C - C_donated C
  ∃ C : ℕ, B_left + C_left C = 38 ∧ C = 30

-- Note that we use sorry to indicate the proof is omitted.
theorem Cameron_books_proof : Cameron_initial_books :=
by {
  sorry
}

end Cameron_books_proof_l271_271164


namespace product_of_x_and_y_l271_271083

variables (EF FG GH HE : ℕ) (x y : ℕ)

theorem product_of_x_and_y (h1: EF = 42) (h2: FG = 4 * y^3) (h3: GH = 2 * x + 10) (h4: HE = 32) (h5: EF = GH) (h6: FG = HE) :
  x * y = 32 :=
by
  sorry

end product_of_x_and_y_l271_271083


namespace line_parallel_l271_271283

theorem line_parallel (x y : ℝ) :
  ∃ m b : ℝ, 
    y = m * (x - 2) + (-4) ∧ 
    m = 2 ∧ 
    (∀ (x y : ℝ), y = 2 * x - 8 → 2 * x - y - 8 = 0) :=
sorry

end line_parallel_l271_271283


namespace sample_std_dev_range_same_l271_271190

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l271_271190


namespace graduation_photo_arrangement_l271_271975

theorem graduation_photo_arrangement (teachers middle_positions other_students : Finset ℕ) (A B : ℕ) :
  teachers.card = 2 ∧ middle_positions.card = 2 ∧ 
  (other_students ∪ {A, B}).card = 4 ∧ ∀ t ∈ teachers, t ∈ middle_positions →
  ∃ arrangements : ℕ, arrangements = 8 :=
by
  sorry

end graduation_photo_arrangement_l271_271975


namespace coin_toss_probability_l271_271320

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l271_271320


namespace average_of_numbers_l271_271475

theorem average_of_numbers : 
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1 + 1252140 + 2345) / 11 = 114391 :=
by
  sorry

end average_of_numbers_l271_271475


namespace equilateral_triangle_circumcircle_area_l271_271017

/-- Given an equilateral triangle DEF with DE = DF = EF = 8 units, 
    and a circle with radius 4 units tangent to DE at E and DF at F,
    prove that the area of the circumcircle that passes through D, E, and F is 64π/3 units². -/
theorem equilateral_triangle_circumcircle_area :
  ∀ (D E F : Point) (r : ℝ),
  is_equilateral_triangle D E F ∧
  distance D E = 8 ∧
  distance D F = 8 ∧
  distance E F = 8 ∧
  is_tangent_circle D E F r 4 → 
  circle_area (circumcircle D E F) = 64 * π / 3 :=
by {
  intros D E F r h_cond,
  sorry
}

end equilateral_triangle_circumcircle_area_l271_271017


namespace rod_division_segments_l271_271794

theorem rod_division_segments (L : ℕ) (K : ℕ) (hL : L = 72 * K) :
  let red_divisions := 7
  let blue_divisions := 11
  let black_divisions := 17
  let overlap_9_6 := 4
  let overlap_6_4 := 6
  let overlap_9_4 := 2
  let overlap_all := 2
  let total_segments := red_divisions + blue_divisions + black_divisions - overlap_9_6 - overlap_6_4 - overlap_9_4 + overlap_all
  (total_segments = 28) ∧ ((L / 72) = K)
:=
by
  sorry

end rod_division_segments_l271_271794


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l271_271930

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l271_271930


namespace probability_exactly_three_heads_l271_271312
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l271_271312


namespace georgie_ghost_enter_exit_diff_window_l271_271954

theorem georgie_ghost_enter_exit_diff_window (n : ℕ) (h : n = 8) :
    (∃ enter exit, enter ≠ exit ∧ 1 ≤ enter ∧ enter ≤ n ∧ 1 ≤ exit ∧ exit ≤ n) ∧
    (∃ W : ℕ, W = (n * (n - 1))) :=
sorry

end georgie_ghost_enter_exit_diff_window_l271_271954


namespace sufficient_not_necessary_l271_271385

noncomputable def f (x a : ℝ) := x^2 - 2*a*x + 1

def no_real_roots (a : ℝ) : Prop := 4*a^2 - 4 < 0

def non_monotonic_interval (a m : ℝ) : Prop := m < a ∧ a < m + 3

def A := {a : ℝ | -1 < a ∧ a < 1}
def B (m : ℝ) := {a : ℝ | m < a ∧ a < m + 3}

theorem sufficient_not_necessary (x : ℝ) (m : ℝ) :
  (x ∈ A → x ∈ B m) → (A ⊆ B m) ∧ (exists a : ℝ, a ∈ B m ∧ a ∉ A) →
  -2 ≤ m ∧ m ≤ -1 := by 
  sorry

end sufficient_not_necessary_l271_271385


namespace tiling_2x12_l271_271564

def d : Nat → Nat
| 0     => 0  -- Unused but for safety in function definition
| 1     => 1
| 2     => 2
| (n+1) => d n + d (n-1)

theorem tiling_2x12 : d 12 = 233 := by
  sorry

end tiling_2x12_l271_271564


namespace product_of_areas_eq_square_of_volume_l271_271535

theorem product_of_areas_eq_square_of_volume 
(x y z d : ℝ) 
(h1 : d^2 = x^2 + y^2 + z^2) :
  (x * y) * (y * z) * (z * x) = (x * y * z) ^ 2 :=
by sorry

end product_of_areas_eq_square_of_volume_l271_271535


namespace quadratic_no_real_roots_l271_271870

theorem quadratic_no_real_roots (a : ℝ) :
  ¬ ∃ x : ℝ, x^2 - 2 * x - a = 0 → a < -1 :=
sorry

end quadratic_no_real_roots_l271_271870


namespace find_pairs_s_t_l271_271221

theorem find_pairs_s_t (n : ℤ) (hn : n > 1) : 
  ∃ s t : ℤ, (
    (∀ x : ℝ, x ^ n + s * x = 2007 ∧ x ^ n + t * x = 2008 → 
     (s, t) = (2006, 2007) ∨ (s, t) = (-2008, -2009) ∨ (s, t) = (-2006, -2007))
  ) :=
sorry

end find_pairs_s_t_l271_271221


namespace smallest_solution_l271_271819

theorem smallest_solution (x : ℝ) :
  (∃ x, (3 * x) / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15) →
  x = -1 := 
sorry

end smallest_solution_l271_271819


namespace small_fries_number_l271_271340

variables (L S : ℕ)

axiom h1 : L + S = 24
axiom h2 : L = 5 * S

theorem small_fries_number : S = 4 :=
by sorry

end small_fries_number_l271_271340


namespace quadratic_func_condition_l271_271078

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_func_condition (b c : ℝ) (h : f (-3) b c = f 1 b c) :
  f 1 b c > c ∧ c > f (-1) b c :=
by
  sorry

end quadratic_func_condition_l271_271078


namespace Sam_distance_l271_271893

theorem Sam_distance (miles_Marguerite: ℝ) (hours_Marguerite: ℕ) (hours_Sam: ℕ) (speed_factor: ℝ) 
  (h1: miles_Marguerite = 150) 
  (h2: hours_Marguerite = 3) 
  (h3: hours_Sam = 4)
  (h4: speed_factor = 1.2) :
  let average_speed_Marguerite := miles_Marguerite / hours_Marguerite
  let average_speed_Sam := speed_factor * average_speed_Marguerite
  let distance_Sam := average_speed_Sam * hours_Sam
  distance_Sam = 240 := 
by 
  sorry

end Sam_distance_l271_271893


namespace find_x_minus_y_l271_271389

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + 3 * y = 14) (h2 : x + 4 * y = 11) : x - y = 3 := by
  sorry

end find_x_minus_y_l271_271389


namespace parabola_standard_equation_l271_271063

variable (a : ℝ) (h : a < 0)

theorem parabola_standard_equation :
  (∃ p : ℝ, y^2 = -2 * p * x ∧ p = -2 * a) → y^2 = 4 * a * x :=
by
  sorry

end parabola_standard_equation_l271_271063


namespace expected_remaining_matches_for_60_matches_l271_271112

noncomputable def expected_remaining_matches (n : ℕ) : ℝ :=
  if h : n = 60 then 7.795 else 0 -- We'll handle the specific case where n = 60

theorem expected_remaining_matches_for_60_matches :
  expected_remaining_matches 60 = 7.795 :=
by
  sorry

end expected_remaining_matches_for_60_matches_l271_271112


namespace line_equation_l271_271270

theorem line_equation 
  (m b k : ℝ) 
  (h1 : ∀ k, abs ((k^2 + 4 * k + 4) - (m * k + b)) = 4)
  (h2 : m * 2 + b = 8) 
  (h3 : b ≠ 0) : 
  m = 8 ∧ b = -8 :=
by sorry

end line_equation_l271_271270


namespace smallest_n_inequality_l271_271366

theorem smallest_n_inequality:
  ∃ n : ℤ, (∀ x y z : ℝ, (x^2 + 2 * y^2 + z^2)^2 ≤ n * (x^4 + 3 * y^4 + z^4)) ∧ n = 4 :=
by
  sorry

end smallest_n_inequality_l271_271366


namespace analysis_hours_l271_271240

theorem analysis_hours (n t : ℕ) (h1 : n = 206) (h2 : t = 1) : n * t = 206 := by
  sorry

end analysis_hours_l271_271240


namespace box_volume_possible_l271_271792

theorem box_volume_possible (x : ℕ) (V : ℕ) (H1 : V = 40 * x^3) (H2 : (2 * x) * (4 * x) * (5 * x) = V) : 
  V = 320 :=
by 
  have x_possible_values := x
  -- checking if V = 320 and x = 2 satisfies the given conditions
  sorry

end box_volume_possible_l271_271792


namespace min_adjacent_white_cells_8x8_grid_l271_271537

theorem min_adjacent_white_cells_8x8_grid (n_blacks : ℕ) (h1 : n_blacks = 20) : 
  ∃ w_cell_pairs, w_cell_pairs = 34 :=
by
  -- conditions are translated here for interpret
  let total_pairs := 112 -- total pairs in 8x8 grid
  let max_spoiled := 78  -- maximum spoiled pairs when placing 20 black cells
  let min_adjacent_white_pairs := total_pairs - max_spoiled
  use min_adjacent_white_pairs
  exact (by linarith)
  sorry

end min_adjacent_white_cells_8x8_grid_l271_271537


namespace stones_required_to_pave_hall_l271_271662

theorem stones_required_to_pave_hall :
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    (area_hall_dm2 / area_stone_dm2) = 3600 :=
by
    -- Definitions
    let length_hall_m := 36
    let breadth_hall_m := 15
    let length_stone_dm := 3
    let breadth_stone_dm := 5

    -- Convert to decimeters
    let length_hall_dm := length_hall_m * 10
    let breadth_hall_dm := breadth_hall_m * 10
    
    -- Calculate areas
    let area_hall_dm2 := length_hall_dm * breadth_hall_dm
    let area_stone_dm2 := length_stone_dm * breadth_stone_dm
    
    -- Calculate number of stones 
    let number_of_stones := area_hall_dm2 / area_stone_dm2

    -- Prove the required number of stones
    have h : number_of_stones = 3600 := sorry
    exact h

end stones_required_to_pave_hall_l271_271662


namespace first_problem_l271_271186

-- Definitions for the first problem
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (h_pos : ∀ n, a n > 0)
variable (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1))

-- Theorem statement for the first problem
theorem first_problem (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1)) :
  ∃ d, ∀ n, a (n + 1) - a n = d := sorry

end first_problem_l271_271186


namespace solve_for_y_l271_271105

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end solve_for_y_l271_271105


namespace minimum_value_expression_l271_271377

theorem minimum_value_expression {a b c : ℝ} :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 13 → 
  (∃ x, x = (a^2 + b^3 + c^4 + 2019) / (10 * b + 123 * c + 26) ∧ ∀ y, y ≤ x) →
  x = 4 :=
by
  sorry

end minimum_value_expression_l271_271377


namespace prime_product_div_by_four_l271_271257

theorem prime_product_div_by_four 
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq1 : Nat.Prime (p * q + 1)) : 
  4 ∣ (2 * p + q) * (p + 2 * q) := 
sorry

end prime_product_div_by_four_l271_271257


namespace geometric_sequence_l271_271388

-- Define the set and its properties
variable (A : Set ℕ) (a : ℕ → ℕ) (n : ℕ)
variable (h1 : 1 ≤ a 1) 
variable (h2 : ∀ (i : ℕ), 1 ≤ i → i < n → a i < a (i + 1))
variable (h3 : n ≥ 5)
variable (h4 : ∀ (i j : ℕ), 1 ≤ i → i ≤ j → j ≤ n → (a i) * (a j) ∈ A ∨ (a i) / (a j) ∈ A)

-- Statement to prove that the sequence forms a geometric sequence
theorem geometric_sequence : 
  ∃ (c : ℕ), c > 1 ∧ ∀ (i : ℕ), 1 ≤ i → i ≤ n → a i = c^(i-1) := sorry

end geometric_sequence_l271_271388


namespace payment_amount_l271_271000

/-- 
A certain debt will be paid in 52 installments from January 1 to December 31 of a certain year.
Each of the first 25 payments is to be a certain amount; each of the remaining payments is to be $100 more than each of the first payments.
The average (arithmetic mean) payment that will be made on the debt for the year is $551.9230769230769.
Prove that the amount of each of the first 25 payments is $500.
-/
theorem payment_amount (X : ℝ) 
  (h1 : 25 * X + 27 * (X + 100) = 52 * 551.9230769230769) :
  X = 500 :=
sorry

end payment_amount_l271_271000


namespace balls_into_boxes_l271_271858

-- Define the problem conditions and expected outcome.
theorem balls_into_boxes : 
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → (∃ (ways : ℕ), ways = n) := 
begin
  use 7,
  split,
  { refl, },
  { intros balls boxes hballs hboxes,
    use 7,
    sorry
  }
end

end balls_into_boxes_l271_271858


namespace frank_has_4_five_dollar_bills_l271_271045

theorem frank_has_4_five_dollar_bills
    (one_dollar_bills : ℕ := 7)
    (ten_dollar_bills : ℕ := 2)
    (twenty_dollar_bills : ℕ := 1)
    (change : ℕ := 4)
    (peanut_cost_per_pound : ℕ := 3)
    (days_in_week : ℕ := 7)
    (peanuts_per_day : ℕ := 3) :
    let initial_amount := (one_dollar_bills * 1) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)
    let total_peanuts_cost := (peanuts_per_day * days_in_week) * peanut_cost_per_pound
    let F := (total_peanuts_cost + change - initial_amount) / 5 
    F = 4 :=
by
  repeat { admit }


end frank_has_4_five_dollar_bills_l271_271045


namespace total_marbles_l271_271213

variable (r b g : ℝ)
variable (h1 : r = 1.3 * b)
variable (h2 : g = 1.7 * r)

theorem total_marbles (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r :=
by
  sorry

end total_marbles_l271_271213


namespace solve_inequality_l271_271985

theorem solve_inequality (x : ℝ) :
  (x ≠ 3) → (x * (x + 2) / (x - 3)^2 ≥ 8) ↔ (x ∈ set.Iic (18/7) ∪ set.Ioi 4) :=
by
  sorry

end solve_inequality_l271_271985


namespace percentage_carnations_l271_271341

variable (F : ℕ)
variable (H1 : F ≠ 0) -- Non-zero flowers
variable (H2 : ∀ (y : ℕ), 5 * y = F → 2 * y ≠ 0) -- Two fifths of the pink flowers are roses.
variable (H3 : ∀ (z : ℕ), 7 * z = 3 * (F - F / 2 - F / 5) → 6 * z ≠ 0) -- Six sevenths of the red flowers are carnations.
variable (H4 : ∀ (w : ℕ), 5 * w = F → w ≠ 0) -- One fifth of the flowers are yellow tulips.
variable (H5 : 2 * F / 2 = F) -- Half of the flowers are pink.
variable (H6 : ∀ (c : ℕ), 10 * c = F → c ≠ 0) -- Total flowers in multiple of 10

theorem percentage_carnations :
  (exists (pc rc : ℕ), 70 * (pc + rc) = 55 * F) :=
sorry

end percentage_carnations_l271_271341


namespace certain_value_z_l271_271702

-- Define the length of an integer as the number of prime factors
def length (n : ℕ) : ℕ := 
  primeFactors n |>.length

theorem certain_value_z {x y : ℕ} (hx1 : x > 1) (hy1 : y > 1)
  (h_len : length x + length y = 16) : 
  x + 3 * y < 98307 := 
sorry

end certain_value_z_l271_271702


namespace julie_net_monthly_income_is_l271_271219

section JulieIncome

def starting_pay : ℝ := 5.00
def additional_experience_pay_per_year : ℝ := 0.50
def years_of_experience : ℕ := 3
def work_hours_per_day : ℕ := 8
def work_days_per_week : ℕ := 6
def bi_weekly_bonus : ℝ := 50.00
def tax_rate : ℝ := 0.12
def insurance_premium_per_month : ℝ := 40.00
def missed_days : ℕ := 1

-- Calculate Julie's net monthly income
def net_monthly_income : ℝ :=
    let hourly_wage := starting_pay + additional_experience_pay_per_year * years_of_experience
    let daily_earnings := hourly_wage * work_hours_per_day
    let weekly_earnings := daily_earnings * (work_days_per_week - missed_days)
    let bi_weekly_earnings := weekly_earnings * 2
    let gross_monthly_income := bi_weekly_earnings * 2 + bi_weekly_bonus * 2
    let tax_deduction := gross_monthly_income * tax_rate
    let total_deductions := tax_deduction + insurance_premium_per_month
    gross_monthly_income - total_deductions

theorem julie_net_monthly_income_is : net_monthly_income = 963.20 :=
    sorry

end JulieIncome

end julie_net_monthly_income_is_l271_271219


namespace sample_standard_deviation_same_sample_range_same_l271_271189

open Nat

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ)
hypothesis (h_c : c ≠ 0)

/-- Assertion C: The sample standard deviations of the two sets of sample data are the same. -/
theorem sample_standard_deviation_same :
  (1 / n * ∑ i, (x i - (1 / n * ∑ i, x i))^2).sqrt =
  (1 / n * ∑ i, (x i + c - (1 / n * ∑ i, x i + c))^2).sqrt := sorry

/-- Assertion D: The sample ranges of the two sets of sample data are the same. -/
theorem sample_range_same :
  (Finset.sup Finset.univ x - Finset.inf Finset.univ x) =
  (Finset.sup Finset.univ (fun i => x i + c) - Finset.inf Finset.univ (fun i => x i + c)) := sorry

end sample_standard_deviation_same_sample_range_same_l271_271189


namespace estimated_germination_probability_stable_l271_271658

structure ExperimentData where
  n : ℕ  -- number of grains per batch
  m : ℕ  -- number of germinations

def experimentalData : List ExperimentData := [
  ⟨50, 47⟩,
  ⟨100, 89⟩,
  ⟨200, 188⟩,
  ⟨500, 461⟩,
  ⟨1000, 892⟩,
  ⟨2000, 1826⟩,
  ⟨3000, 2733⟩
]

def germinationFrequency (data : ExperimentData) : ℚ :=
  data.m / data.n

def closeTo (x y : ℚ) (ε : ℚ) : Prop :=
  |x - y| < ε

theorem estimated_germination_probability_stable :
  ∃ ε > 0, ∀ data ∈ experimentalData, closeTo (germinationFrequency data) 0.91 ε :=
by
  sorry

end estimated_germination_probability_stable_l271_271658


namespace arccos_cos_11_equals_4_717_l271_271808

noncomputable def arccos_cos_11 : Real :=
  let n : ℤ := Int.floor (11 / (2 * Real.pi))
  Real.arccos (Real.cos 11)

theorem arccos_cos_11_equals_4_717 :
  arccos_cos_11 = 4.717 := by
  sorry

end arccos_cos_11_equals_4_717_l271_271808


namespace mother_picked_38_carrots_l271_271559

theorem mother_picked_38_carrots
  (haley_carrots : ℕ)
  (good_carrots : ℕ)
  (bad_carrots : ℕ)
  (total_carrots_picked : ℕ)
  (mother_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : good_carrots = 64)
  (h3 : bad_carrots = 13)
  (h4 : total_carrots_picked = good_carrots + bad_carrots)
  (h5 : total_carrots_picked = haley_carrots + mother_carrots) :
  mother_carrots = 38 :=
by
  sorry

end mother_picked_38_carrots_l271_271559


namespace cake_and_tea_cost_l271_271120

theorem cake_and_tea_cost (cost_of_milk_tea : ℝ) (cost_of_cake : ℝ)
    (h1 : cost_of_cake = (3 / 4) * cost_of_milk_tea)
    (h2 : cost_of_milk_tea = 2.40) :
    2 * cost_of_cake + cost_of_milk_tea = 6.00 := 
sorry

end cake_and_tea_cost_l271_271120


namespace range_of_a_l271_271064

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l271_271064


namespace Elmer_eats_more_than_Penelope_l271_271432

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l271_271432


namespace evaluate_g_at_3_l271_271071

def g (x : ℝ) : ℝ := 9 * x^3 - 5 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 200 := by
  sorry

end evaluate_g_at_3_l271_271071


namespace total_profit_calculation_l271_271013

-- Definitions based on conditions
def initial_investment_A := 5000
def initial_investment_B := 8000
def initial_investment_C := 9000
def initial_investment_D := 7000

def investment_A_after_4_months := initial_investment_A + 2000
def investment_B_after_4_months := initial_investment_B - 1000

def investment_C_after_6_months := initial_investment_C + 3000
def investment_D_after_6_months := initial_investment_D + 5000

def profit_A_percentage := 20
def profit_B_percentage := 30
def profit_C_percentage := 25
def profit_D_percentage := 25

def profit_C := 60000

-- Total profit is what we need to determine
def total_profit := 240000

-- The proof statement
theorem total_profit_calculation :
  total_profit = (profit_C * 100) / profit_C_percentage := 
by 
  sorry

end total_profit_calculation_l271_271013


namespace amoeba_count_after_one_week_l271_271015

/-- An amoeba is placed in a puddle and splits into three amoebas on the same day. Each subsequent
    day, every amoeba in the puddle splits into three new amoebas. -/
theorem amoeba_count_after_one_week : 
  let initial_amoebas := 1
  let daily_split := 3
  let days := 7
  (initial_amoebas * (daily_split ^ days)) = 2187 :=
by
  sorry

end amoeba_count_after_one_week_l271_271015


namespace interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l271_271181

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : 
  (∀ x : ℝ, 0 < a → (Real.exp x - a ≥ 0 ↔ x ≥ Real.log a)) ∧ 
  (∀ x : ℝ, a ≤ 0 → (Real.exp x - a ≥ 0)) :=
by sorry

theorem monotone_increasing (a : ℝ) (h : ∀ x : ℝ, Real.exp x - a ≥ 0) : 
  a ≤ 0 :=
by sorry

theorem monotonically_increasing_decreasing : 
  ∃ a : ℝ, (∀ x ≤ 0, Real.exp x - a ≤ 0) ∧ 
           (∀ x ≥ 0, Real.exp x - a ≥ 0) ↔ a = 1 :=
by sorry

end interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l271_271181


namespace range_of_a_l271_271706

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x = 1) ↔ a ≠ 0 := by
sorry

end range_of_a_l271_271706


namespace inverse_proportion_relationship_l271_271741

variable k : ℝ
def y (x : ℝ) : ℝ := k / x

theorem inverse_proportion_relationship (h1 : y (-2) = 3) :
  let y1 := y (-3) in
  let y2 := y 1 in
  let y3 := y 2 in
  y2 < y3 ∧ y3 < y1 :=
by
  have k_val : k = -6 := by
    rw [y, ← @eq_div_iff_mul_eq ℝ _ _ _ _] at h1 <;> linarith
  let y1 := -6 / -3
  let y2 := -6 / 1
  let y3 := -6 / 2
  split
  sorry

end inverse_proportion_relationship_l271_271741


namespace initial_numbers_count_l271_271266

theorem initial_numbers_count (n : ℕ) (S : ℝ)
  (h1 : S / n = 56)
  (h2 : (S - 100) / (n - 2) = 56.25) :
  n = 50 :=
sorry

end initial_numbers_count_l271_271266


namespace subtract_correctly_l271_271470

theorem subtract_correctly (x : ℕ) (h : x + 35 = 77) : x - 35 = 7 :=
sorry

end subtract_correctly_l271_271470


namespace eight_pow_three_eq_two_pow_nine_l271_271208

theorem eight_pow_three_eq_two_pow_nine : 8^3 = 2^9 := by
  sorry -- Proof is skipped

end eight_pow_three_eq_two_pow_nine_l271_271208


namespace intersection_A_B_l271_271194

-- Definition of sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y > 0 }

-- The proof goal
theorem intersection_A_B : A ∩ B = { x | x > 1 } :=
by sorry

end intersection_A_B_l271_271194


namespace ratio_of_common_differences_l271_271875

variable (a b d1 d2 : ℝ)

theorem ratio_of_common_differences
  (h1 : a + 4 * d1 = b)
  (h2 : a + 5 * d2 = b) :
  d1 / d2 = 5 / 4 := 
by
  sorry

end ratio_of_common_differences_l271_271875


namespace how_many_fewer_girls_l271_271915

def total_students : ℕ := 27
def girls : ℕ := 11
def boys : ℕ := total_students - girls
def fewer_girls_than_boys : ℕ := boys - girls

theorem how_many_fewer_girls :
  fewer_girls_than_boys = 5 :=
sorry

end how_many_fewer_girls_l271_271915


namespace determine_c_l271_271803

-- Assume we have three integers a, b, and unique x, y, z such that
variables (a b c x y z : ℕ)

-- Define the conditions
def condition1 : Prop := a = Nat.lcm y z
def condition2 : Prop := b = Nat.lcm x z
def condition3 : Prop := c = Nat.lcm x y

-- Prove that Bob can determine c based on a and b
theorem determine_c (h1 : condition1 a y z) (h2 : condition2 b x z) (h3 : ∀ u v w : ℕ, (Nat.lcm u w = a ∧ Nat.lcm v w = b ∧ Nat.lcm u v = c) → (u = x ∧ v = y ∧ w = z) ) : ∃ c, condition3 c x y :=
by sorry

end determine_c_l271_271803


namespace base_conversion_l271_271986

theorem base_conversion (k : ℕ) : (5 * 8^2 + 2 * 8^1 + 4 * 8^0 = 6 * k^2 + 6 * k + 4) → k = 7 :=
by 
  let x := 5 * 8^2 + 2 * 8^1 + 4 * 8^0
  have h : x = 340 := by sorry
  have hk : 6 * k^2 + 6 * k + 4 = 340 := by sorry
  sorry

end base_conversion_l271_271986


namespace point_of_tangency_l271_271830

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x + a / Real.exp x

theorem point_of_tangency (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) 
  (h_slope : ∃ x : ℝ, Real.exp x - 1 / Real.exp x = 3 / 2) :
  ∃ x : ℝ, x = Real.log 2 :=
by
  sorry

end point_of_tangency_l271_271830


namespace paint_cost_is_correct_l271_271778

-- Definition of known conditions
def costPerKg : ℕ := 50
def coveragePerKg : ℕ := 20
def sideOfCube : ℕ := 20

-- Definition of correct answer
def totalCost : ℕ := 6000

-- Theorem statement
theorem paint_cost_is_correct : (6 * (sideOfCube * sideOfCube) / coveragePerKg) * costPerKg = totalCost :=
by
  sorry

end paint_cost_is_correct_l271_271778


namespace hyperbola_equation_l271_271552

-- Define the conditions of the problem
def center_at_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0
def focus_on_y_axis (x : ℝ) : Prop := x = 0
def focal_distance (d : ℝ) : Prop := d = 4
def point_on_hyperbola (x y : ℝ) : Prop := x = 1 ∧ y = -Real.sqrt 3

-- Final statement to prove
theorem hyperbola_equation :
  (center_at_origin 0 0) ∧
  (focus_on_y_axis 0) ∧
  (focal_distance 4) ∧
  (point_on_hyperbola 1 (-Real.sqrt 3))
  → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a = Real.sqrt 3 ∧ b = 1) ∧ (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) :=
by
  sorry

end hyperbola_equation_l271_271552


namespace back_wheel_revolutions_l271_271739

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (rev_front : ℝ) (r_front_eq : r_front = 3)
  (r_back_eq : r_back = 0.5) (rev_front_eq : rev_front = 50) :
  let C_front := 2 * Real.pi * r_front
  let D_front := C_front * rev_front
  let C_back := 2 * Real.pi * r_back
  let rev_back := D_front / C_back
  rev_back = 300 := by
  sorry

end back_wheel_revolutions_l271_271739


namespace probability_of_same_number_on_four_dice_l271_271633

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l271_271633


namespace greatest_common_multiple_of_9_and_15_less_than_120_l271_271140

-- Definition of LCM.
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The main theorem to be proved.
theorem greatest_common_multiple_of_9_and_15_less_than_120 : ∃ x, x = 90 ∧ x < 120 ∧ x % 9 = 0 ∧ x % 15 = 0 :=
by
  -- Proof goes here.
  sorry

end greatest_common_multiple_of_9_and_15_less_than_120_l271_271140


namespace expression_evaluation_l271_271399

theorem expression_evaluation (a b : ℤ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 :=
by
  sorry

end expression_evaluation_l271_271399


namespace max_sum_length_le_98306_l271_271701

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end max_sum_length_le_98306_l271_271701


namespace range_of_m_l271_271054

theorem range_of_m (f : ℝ → ℝ) (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y < f x) (h_cond : ∀ m : ℝ, f (1 - m) < f (m - 3)) : ∀ m, 1 < m ∧ m < 2 :=
by
  intros m
  sorry

end range_of_m_l271_271054


namespace monthly_income_l271_271006

-- Define the conditions
variable (I : ℝ) -- Total monthly income
variable (remaining : ℝ) -- Remaining amount before donation
variable (remaining_after_donation : ℝ) -- Amount after donation

-- Conditions
def condition1 : Prop := remaining = I - 0.63 * I - 1500
def condition2 : Prop := remaining_after_donation = remaining - 0.05 * remaining
def condition3 : Prop := remaining_after_donation = 35000

-- Theorem to prove the total monthly income
theorem monthly_income (h1 : condition1 I remaining) (h2 : condition2 remaining remaining_after_donation) (h3 : condition3 remaining_after_donation) : I = 103600 := 
by sorry

end monthly_income_l271_271006


namespace minimum_expression_value_l271_271550

noncomputable def expr (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  (2 * (Real.sin x₁)^2 + 1 / (Real.sin x₁)^2) *
  (2 * (Real.sin x₂)^2 + 1 / (Real.sin x₂)^2) *
  (2 * (Real.sin x₃)^2 + 1 / (Real.sin x₃)^2) *
  (2 * (Real.sin x₄)^2 + 1 / (Real.sin x₄)^2)

theorem minimum_expression_value :
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  x₁ + x₂ + x₃ + x₄ = Real.pi →
  expr x₁ x₂ x₃ x₄ ≥ 81 := sorry

end minimum_expression_value_l271_271550


namespace paper_cups_calculation_l271_271676

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l271_271676


namespace symmetry_condition_l271_271759

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = x ↔ x = (ax + b) / (cx - d)) ∧ 
  (∀ x y : ℝ, y = -x ↔ x = (-ax + b) / (-cx - d)) → 
  d + b = 0 :=
by sorry

end symmetry_condition_l271_271759


namespace ratio_of_areas_l271_271494

theorem ratio_of_areas (r : ℝ) (A_triangle : ℝ) (A_circle : ℝ) 
  (h1 : ∀ r, A_triangle = (3 * r^2) / 4)
  (h2 : ∀ r, A_circle = π * r^2) 
  : (A_triangle / A_circle) = 3 / (4 * π) :=
sorry

end ratio_of_areas_l271_271494


namespace range_of_x_l271_271196

theorem range_of_x (f : ℝ → ℝ) (h_increasing : ∀ x y, x ≤ y → f x ≤ f y) (h_defined : ∀ x, -1 ≤ x ∧ x ≤ 1)
  (h_condition : ∀ x, f (x-2) < f (1-x)) : ∀ x, 1 ≤ x ∧ x < 3/2 :=
by
  sorry

end range_of_x_l271_271196


namespace area_bounded_by_curves_is_4pi_l271_271151

noncomputable def parametric_x (t : ℝ) : ℝ := 16 * (Real.cos t)^3
noncomputable def parametric_y (t : ℝ) : ℝ := 2 * (Real.sin t)^3

theorem area_bounded_by_curves_is_4pi : (∫ t in -Real.pi / 3..Real.pi / 3, parametric_y t * deriv parametric_x t) = 4 * Real.pi :=
by
  sorry

end area_bounded_by_curves_is_4pi_l271_271151


namespace bug_total_distance_l271_271287

/-!
# Problem Statement
A bug starts crawling on a number line from position -3. It first moves to -7, then turns around and stops briefly at 0 before continuing on to 8. Prove that the total distance the bug crawls is 19 units.
-/

def bug_initial_position : ℤ := -3
def bug_position_1 : ℤ := -7
def bug_position_2 : ℤ := 0
def bug_final_position : ℤ := 8

theorem bug_total_distance : 
  |bug_position_1 - bug_initial_position| + 
  |bug_position_2 - bug_position_1| + 
  |bug_final_position - bug_position_2| = 19 :=
by 
  sorry

end bug_total_distance_l271_271287


namespace jasper_wins_probability_l271_271729

-- Definitions and conditions
def prob_heads_jasper : ℚ := 2 / 7
def prob_heads_kira : ℚ := 1 / 4
def prob_tails_jasper : ℚ := 1 - prob_heads_jasper
def prob_tails_kira : ℚ := 1 - prob_heads_kira
def prob_both_tails : ℚ := prob_tails_jasper * prob_tails_kira

-- Hypothesis: Kira goes first
axiom independent_tosses : Prop -- Placeholder for independent toss axiom

-- The ultimate probability that Jasper wins
def prob_jasper_wins : ℚ := 
  (prob_heads_jasper * prob_both_tails) / (1 - prob_both_tails)

-- The theorem to prove
theorem jasper_wins_probability :
  prob_jasper_wins = 30 / 91 := 
sorry

end jasper_wins_probability_l271_271729


namespace coin_toss_probability_l271_271327

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l271_271327


namespace man_age_twice_son_age_in_n_years_l271_271670

theorem man_age_twice_son_age_in_n_years
  (S M Y : ℤ)
  (h1 : S = 26)
  (h2 : M = S + 28)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 :=
by
  sorry

end man_age_twice_son_age_in_n_years_l271_271670


namespace combinedTotalSandcastlesAndTowers_l271_271261

def markSandcastles : Nat := 20
def towersPerMarkSandcastle : Nat := 10
def jeffSandcastles : Nat := 3 * markSandcastles
def towersPerJeffSandcastle : Nat := 5

theorem combinedTotalSandcastlesAndTowers :
  (markSandcastles + markSandcastles * towersPerMarkSandcastle) +
  (jeffSandcastles + jeffSandcastles * towersPerJeffSandcastle) = 580 :=
by
  sorry

end combinedTotalSandcastlesAndTowers_l271_271261


namespace probability_A_2_restaurants_correct_distribution_X_correct_compare_likelihood_B_l271_271796

section DiningProblem

variables (days_total : ℕ := 100)
variables (a_aa a_ab a_ba a_bb : ℕ) (b_aa b_ab b_ba b_bb : ℕ)
variables (probability_A_2_restaurants probability_X2 probability_X3 probability_X4 : ℝ)
variables (expectation_X : ℝ)
variables (more_likely_B : Prop)

-- Individual A's dining choices over 100 working days.
def a_aa := 30
def a_ab := 20
def a_ba := 40
def a_bb := 10

-- Individual B's dining choices over 100 working days.
def b_aa := 20
def b_ab := 25
def b_ba := 15
def b_bb := 40

-- (I) Prove the probability that individual A chooses to dine at 2 restaurants in one day is 0.6.
theorem probability_A_2_restaurants_correct : 
  probability_A_2_restaurants = (a_ab + a_ba) / days_total := sorry

-- (II) Prove the distribution of X has the specified probabilities and expectation.
theorem distribution_X_correct : 
  probability_X2 = 0.24 ∧ probability_X3 = 0.52 ∧ probability_X4 = 0.24 ∧ 
  expectation_X = 3 := sorry

-- (III) Prove that individual B is more likely to choose restaurant B for lunch after choosing restaurant A for breakfast.
theorem compare_likelihood_B : 
  more_likely_B = ( (b_ab / (b_aa + b_ab)) > (a_ab / (a_aa + a_ab))) := sorry

end DiningProblem

end probability_A_2_restaurants_correct_distribution_X_correct_compare_likelihood_B_l271_271796


namespace initial_blue_balls_proof_l271_271343

-- Define the main problem parameters and condition
def initial_jars (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :=
  total_balls = 18 ∧
  removed_blue = 3 ∧
  remaining_balls = total_balls - removed_blue ∧
  probability = 1/5 → 
  (initial_blue_balls - removed_blue) / remaining_balls = probability

-- Define the proof problem
theorem initial_blue_balls_proof (total_balls initial_blue_balls removed_blue probability remaining_balls : ℕ) :
  initial_jars total_balls initial_blue_balls removed_blue probability remaining_balls →
  initial_blue_balls = 6 :=
by
  sorry

end initial_blue_balls_proof_l271_271343


namespace calculate_revolutions_l271_271022

noncomputable def number_of_revolutions (diameter distance: ℝ) : ℝ :=
  distance / (Real.pi * diameter)

theorem calculate_revolutions :
  number_of_revolutions 10 5280 = 528 / Real.pi :=
by
  sorry

end calculate_revolutions_l271_271022


namespace pie_eating_contest_l271_271138

theorem pie_eating_contest :
  let pie1_first_student := 7 / 8
  let pie1_second_student := 5 / 6
  let pie2_first_student := 3 / 4
  let pie2_second_student := 2 / 3
  let total_first_student := pie1_first_student + pie2_first_student
  let total_second_student := pie1_second_student + pie2_second_student
  let difference := total_first_student - total_second_student
  difference = 1 / 8 :=
by
  sorry

end pie_eating_contest_l271_271138


namespace original_price_of_dish_l271_271150

theorem original_price_of_dish : 
  ∀ (P : ℝ), 
  1.05 * P - 1.035 * P = 0.54 → 
  P = 36 :=
by
  intros P h
  sorry

end original_price_of_dish_l271_271150


namespace diagonals_bisect_in_rhombus_l271_271139

axiom Rhombus : Type
axiom Parallelogram : Type

axiom isParallelogram : Rhombus → Parallelogram
axiom diagonalsBisectEachOther : Parallelogram → Prop

theorem diagonals_bisect_in_rhombus (R : Rhombus) :
  ∀ (P : Parallelogram), isParallelogram R = P → diagonalsBisectEachOther P → diagonalsBisectEachOther (isParallelogram R) :=
by
  sorry

end diagonals_bisect_in_rhombus_l271_271139


namespace proof_S5_l271_271183

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a1, ∀ n, a (n + 1) = a1 * q ^ (n + 1)

theorem proof_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ) : 
  (geometric_sequence a) → 
  (a 2 * a 5 = 2 * a 3) → 
  ((a 4 + 2 * a 7) / 2 = 5 / 4) → 
  (S 5 = a1 * (1 - (1 / 2) ^ 5) / (1 - 1 / 2)) → 
  S 5 = 31 := 
by sorry

end proof_S5_l271_271183


namespace min_value_of_f_l271_271709

noncomputable def f (a b x : ℝ) : ℝ :=
  (a / (Real.sin x) ^ 2) + b * (Real.sin x) ^ 2

theorem min_value_of_f (a b : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : a > b) (h4 : b > 0) :
  ∃ x, f a b x = 3 := 
sorry

end min_value_of_f_l271_271709


namespace quadratic_unique_solution_pair_l271_271907

theorem quadratic_unique_solution_pair (a c : ℝ) (h₁ : a + c = 12) (h₂ : a < c) (h₃ : a * c = 9) :
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by
  sorry

end quadratic_unique_solution_pair_l271_271907


namespace shaded_region_area_l271_271359

section

-- Define points and shapes
structure point := (x : ℝ) (y : ℝ)
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length * square_side_length

-- Points defining the square and triangles within it
def point_O : point := ⟨0, 0⟩
def point_A : point := ⟨15, 0⟩
def point_B : point := ⟨40, 25⟩
def point_C : point := ⟨40, 40⟩
def point_D1 : point := ⟨25, 40⟩
def point_E : point := ⟨0, 15⟩

-- Function to calculate the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

-- Areas of individual triangles
def triangle1_area : ℝ := triangle_area 15 15
def triangle2_area : ℝ := triangle_area 25 25
def triangle3_area : ℝ := triangle_area 15 15

-- Total area of the triangles
def total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area

-- Shaded area calculation
def shaded_area : ℝ := square_area - total_triangles_area

-- Statement of the theorem to be proven
theorem shaded_region_area : shaded_area = 1062.5 := by sorry

end

end shaded_region_area_l271_271359


namespace quadratic_root_exists_in_range_l271_271619

theorem quadratic_root_exists_in_range :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ x^2 + 3 * x - 5 = 0 := 
by
  sorry

end quadratic_root_exists_in_range_l271_271619


namespace min_adj_white_pairs_l271_271540

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l271_271540


namespace solve_fraction_eq_l271_271990

theorem solve_fraction_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : (1 / (x - 1) = 3 / (x - 3)) ↔ x = 0 :=
by {
  sorry
}

end solve_fraction_eq_l271_271990


namespace balls_into_boxes_l271_271856

theorem balls_into_boxes : ∃ (n : ℕ), n = 7 ∧ 
  ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 3 → 
    ∃ (partitions : finset (finset (ℕ))), 
      partitions.card = n ∧ 
      ∀ p ∈ partitions, p.sum = balls :=
sorry

end balls_into_boxes_l271_271856


namespace A_share_of_profit_l271_271148

-- Define the conditions
def A_investment : ℕ := 100
def A_months : ℕ := 12
def B_investment : ℕ := 200
def B_months : ℕ := 6
def total_profit : ℕ := 100

-- Calculate the weighted investments (directly from conditions)
def A_weighted_investment : ℕ := A_investment * A_months
def B_weighted_investment : ℕ := B_investment * B_months
def total_weighted_investment : ℕ := A_weighted_investment + B_weighted_investment

-- Prove A's share of the profit
theorem A_share_of_profit : (A_weighted_investment / total_weighted_investment : ℚ) * total_profit = 50 := by
  -- The proof will go here
  sorry

end A_share_of_profit_l271_271148


namespace find_first_divisor_l271_271358

theorem find_first_divisor (x : ℕ) (k m : ℕ) (h₁ : 282 = k * x + 3) (h₂ : 282 = 9 * m + 3) : x = 31 :=
sorry

end find_first_divisor_l271_271358


namespace calculate_speed_of_boat_in_still_water_l271_271288

noncomputable def speed_of_boat_in_still_water (V : ℝ) : Prop :=
    let downstream_speed := 16
    let upstream_speed := 9
    let first_half_current := 3 
    let second_half_current := 5
    let wind_speed := 2
    let effective_current_1 := first_half_current - wind_speed
    let effective_current_2 := second_half_current - wind_speed
    let V1 := downstream_speed - effective_current_1
    let V2 := upstream_speed + effective_current_2
    V = (V1 + V2) / 2

theorem calculate_speed_of_boat_in_still_water : 
    ∃ V : ℝ, speed_of_boat_in_still_water V ∧ V = 13.5 := 
sorry

end calculate_speed_of_boat_in_still_water_l271_271288


namespace probability_heads_9_tails_at_least_2_l271_271927

noncomputable def probability_exactly_nine_heads : ℚ :=
  let total_outcomes := 2 ^ 12
  let successful_outcomes := Nat.choose 12 9
  successful_outcomes / total_outcomes

theorem probability_heads_9_tails_at_least_2 (n : ℕ) (h : n = 12) :
  n = 12 → probability_exactly_nine_heads = 55 / 1024 := by
  intros h
  sorry

end probability_heads_9_tails_at_least_2_l271_271927


namespace days_to_shovel_l271_271401

-- Defining conditions as formal statements
def original_task_time := 10
def original_task_people := 10
def original_task_weight := 10000
def new_task_weight := 40000
def new_task_people := 5

-- Definition of rate in terms of weight, people and time
def rate_per_person (total_weight : ℕ) (total_people : ℕ) (total_time : ℕ) : ℕ :=
  total_weight / total_people / total_time

-- Theorem statement to prove
theorem days_to_shovel (t : ℕ) :
  (rate_per_person original_task_weight original_task_people original_task_time) * new_task_people * t = new_task_weight := sorry

end days_to_shovel_l271_271401


namespace dogs_bunnies_ratio_l271_271081

theorem dogs_bunnies_ratio (total : ℕ) (dogs : ℕ) (bunnies : ℕ) (h1 : total = 375) (h2 : dogs = 75) (h3 : bunnies = total - dogs) : (75 / 75 : ℚ) / (300 / 75 : ℚ) = 1 / 4 := by
  sorry

end dogs_bunnies_ratio_l271_271081


namespace probability_exactly_three_heads_l271_271314
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l271_271314


namespace student_average_always_greater_l271_271962

theorem student_average_always_greater (x y z : ℝ) (h1 : x < z) (h2 : z < y) :
  (B = (x + z + 2 * y) / 4) > (A = (x + y + z) / 3) := by
  sorry

end student_average_always_greater_l271_271962


namespace twin_ages_l271_271163

theorem twin_ages (x : ℕ) (h : (x + 1) ^ 2 = x ^ 2 + 15) : x = 7 :=
sorry

end twin_ages_l271_271163


namespace probability_four_is_largest_l271_271152

/-- The probability that 4 is the largest value selected when three cards are drawn from a set of 5 cards numbered 1 to 5 is 3/10. -/
theorem probability_four_is_largest :
  let cards := {1, 2, 3, 4, 5}
  in probability {s | s.cardinality = 3 ∧ 4 ∈ s ∧ ∀ x ∈ s, x ≤ 4} = 3 / 10 := sorry

end probability_four_is_largest_l271_271152


namespace cos_neg_pi_div_3_l271_271694

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l271_271694


namespace negation_proposition_l271_271611

theorem negation_proposition : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l271_271611


namespace helga_shoes_l271_271068

theorem helga_shoes (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := 
by
  sorry

end helga_shoes_l271_271068


namespace rhombus_diagonals_not_always_equal_l271_271936

structure Rhombus where
  all_four_sides_equal : Prop
  symmetrical : Prop
  centrally_symmetrical : Prop

theorem rhombus_diagonals_not_always_equal (R : Rhombus) :
  ¬ (∀ (d1 d2 : ℝ), d1 = d2) :=
sorry

end rhombus_diagonals_not_always_equal_l271_271936


namespace find_other_number_l271_271944

-- Define the conditions and the theorem
theorem find_other_number (hcf lcm a b : ℕ) (hcf_def : hcf = 20) (lcm_def : lcm = 396) (a_def : a = 36) (rel : hcf * lcm = a * b) : b = 220 :=
by 
  sorry -- Proof to be provided

end find_other_number_l271_271944


namespace ducks_in_smaller_pond_l271_271407

theorem ducks_in_smaller_pond (x : ℝ) (h1 : 50 > 0) 
  (h2 : 0.20 * x > 0) (h3 : 0.12 * 50 > 0) (h4 : 0.15 * (x + 50) = 0.20 * x + 0.12 * 50) 
  : x = 30 := 
sorry

end ducks_in_smaller_pond_l271_271407


namespace find_cos_beta_l271_271379

noncomputable def cos_beta (α β : ℝ) : ℝ :=
  - (6 * Real.sqrt 2 + 4) / 15

theorem find_cos_beta (α β : ℝ)
  (h0 : α ∈ Set.Ioc 0 (Real.pi / 2))
  (h1 : β ∈ Set.Ioc (Real.pi / 2) Real.pi)
  (h2 : Real.cos α = 1 / 3)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.cos β = cos_beta α β :=
by
  sorry

end find_cos_beta_l271_271379


namespace simple_interest_two_years_l271_271898
-- Import the necessary Lean library for mathematical concepts

-- Define the problem conditions and the proof statement
theorem simple_interest_two_years (P r t : ℝ) (CI SI : ℝ)
  (hP : P = 17000) (ht : t = 2) (hCI : CI = 11730) : SI = 5100 :=
by
  -- Principal (P), Rate (r), and Time (t) definitions
  let P := 17000
  let t := 2

  -- Given Compound Interest (CI)
  let CI := 11730

  -- Correct value for Simple Interest (SI) that we need to prove
  let SI := 5100

  -- Formalize the assumptions
  have h1 : P = 17000 := rfl
  have h2 : t = 2 := rfl
  have h3 : CI = 11730 := rfl

  -- Crucial parts of the problem are used here
  sorry  -- This is a placeholder for the actual proof steps

end simple_interest_two_years_l271_271898


namespace find_hours_spent_l271_271518

/-- Let 
  h : ℝ := hours Ed stayed in the hotel last night
  morning_hours : ℝ := 4 -- hours Ed stayed in the hotel this morning
  
  conditions:
  night_cost_per_hour : ℝ := 1.50 -- the cost per hour for staying at night
  morning_cost_per_hour : ℝ := 2 -- the cost per hour for staying in the morning
  initial_amount : ℝ := 80 -- initial amount Ed had
  remaining_amount : ℝ := 63 -- remaining amount after stay
  
  Then the total cost calculated by Ed is:
  total_cost : ℝ := (night_cost_per_hour * h) + (morning_cost_per_hour * morning_hours)
  spent_amount : ℝ := initial_amount - remaining_amount

  We need to prove that h = 6 given the above conditions.
-/
theorem find_hours_spent {h morning_hours night_cost_per_hour morning_cost_per_hour initial_amount remaining_amount total_cost spent_amount : ℝ}
  (hc1 : night_cost_per_hour = 1.50)
  (hc2 : morning_cost_per_hour = 2)
  (hc3 : initial_amount = 80)
  (hc4 : remaining_amount = 63)
  (hc5 : morning_hours = 4)
  (hc6 : spent_amount = initial_amount - remaining_amount)
  (hc7 : total_cost = night_cost_per_hour * h + morning_cost_per_hour * morning_hours)
  (hc8 : spent_amount = 17)
  (hc9 : total_cost = spent_amount) :
  h = 6 :=
by 
  sorry

end find_hours_spent_l271_271518


namespace positive_expression_l271_271426

theorem positive_expression (x y : ℝ) : (x^2 - 4 * x + y^2 + 13) > 0 := by
  sorry

end positive_expression_l271_271426


namespace lcm_of_two_numbers_l271_271079

theorem lcm_of_two_numbers (A B : ℕ) (h1 : A * B = 62216) (h2 : Nat.gcd A B = 22) :
  Nat.lcm A B = 2828 :=
by
  sorry

end lcm_of_two_numbers_l271_271079


namespace elmer_more_than_penelope_l271_271429

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l271_271429


namespace probability_scoring_80_or_above_probability_failing_exam_l271_271405

theorem probability_scoring_80_or_above (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C) = 0.69 :=
by
  intros hB hC hD hE
  sorry

theorem probability_failing_exam (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C ∪ D ∪ E) = 0.93 →
  1 - P (B ∪ C ∪ D ∪ E) = 0.07 :=
by
  intros hB hC hD hE hBCDE
  sorry

end probability_scoring_80_or_above_probability_failing_exam_l271_271405


namespace solve_y_equation_l271_271109

noncomputable def solve_y : ℚ :=
  let y := (500 * 1 : ℚ) / 15 in
  y

theorem solve_y_equation (y : ℚ) :
  2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = solve_y := by
  intro h
  sorry

end solve_y_equation_l271_271109


namespace sum_of_digits_in_product_is_fourteen_l271_271179

def first_number : ℕ := -- Define the 101-digit number 141,414,141,...,414,141
  141 * 10^98 + 141 * 10^95 + 141 * 10^92 -- continue this pattern...

def second_number : ℕ := -- Define the 101-digit number 707,070,707,...,070,707
  707 * 10^98 + 707 * 10^95 + 707 * 10^92 -- continue this pattern...

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem sum_of_digits_in_product_is_fourteen :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 :=
sorry

end sum_of_digits_in_product_is_fourteen_l271_271179


namespace radius_of_smaller_base_of_truncated_cone_l271_271919

theorem radius_of_smaller_base_of_truncated_cone 
  (r1 r2 r3 : ℕ) (touching : 2 * r1 = r2 ∧ r1 + r3 = r2 * 2):
  (∀ (R : ℕ), R = 6) :=
sorry

end radius_of_smaller_base_of_truncated_cone_l271_271919


namespace checkerboard_7_strips_l271_271529

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l271_271529


namespace find_chord_eq_l271_271192

-- Given conditions 
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def point_p : (ℝ × ℝ) := (3, 2)
def midpoint_chord (p1 p2 p : (ℝ × ℝ)) : Prop := p.fst = (p1.fst + p2.fst) / 2 ∧ p.snd = (p1.snd + p2.snd) / 2

-- Conditions in Lean definition
def conditions (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ midpoint_chord (x1,y1) (x2,y2) point_p

-- The statement to prove
theorem find_chord_eq (x1 y1 x2 y2 : ℝ) (h : conditions x1 y1 x2 y2) :
  ∃ m b : ℝ, (m = -2 / 3) ∧ b = 2 - m * 3 ∧ (∀ x y : ℝ, y = m * x + b → 2 * x + 3 * y - 12 = 0) :=
by {
  sorry
}

end find_chord_eq_l271_271192


namespace range_of_a_l271_271555

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / Real.exp x - Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a ^ 2) ≤ 0) : 
  a ∈ Set.Iic (-1) ∪ Set.Ici (1 / 2) :=
sorry

end range_of_a_l271_271555


namespace banquet_food_consumption_l271_271254

theorem banquet_food_consumption (n : ℕ) (food_per_guest : ℕ) (total_food : ℕ) 
  (h1 : ∀ g : ℕ, g ≤ n -> g * food_per_guest ≤ total_food)
  (h2 : n = 169) 
  (h3 : food_per_guest = 2) :
  total_food = 338 := 
sorry

end banquet_food_consumption_l271_271254


namespace max_touched_points_by_line_l271_271044

noncomputable section

open Function

-- Definitions of the conditions
def coplanar_circles (circles : Set (Set ℝ)) : Prop :=
  ∀ c₁ c₂ : Set ℝ, c₁ ∈ circles → c₂ ∈ circles → c₁ ≠ c₂ → ∃ p : ℝ, p ∈ c₁ ∧ p ∈ c₂

def max_touched_points (line_circle : ℝ → ℝ) : ℕ :=
  2

-- The theorem statement that needs to be proven
theorem max_touched_points_by_line {circles : Set (Set ℝ)} (h_coplanar : coplanar_circles circles) :
  ∀ line : ℝ → ℝ, (∃ (c₁ c₂ c₃ : Set ℝ), c₁ ∈ circles ∧ c₂ ∈ circles ∧ c₃ ∈ circles ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃) →
  ∃ (p : ℕ), p = 6 := 
sorry

end max_touched_points_by_line_l271_271044


namespace equation1_solutions_equation2_solutions_l271_271441

theorem equation1_solutions (x : ℝ) :
  x ^ 2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := by
  sorry

theorem equation2_solutions (x : ℝ) :
  2 * x ^ 2 - 2 * x = 1 ↔ x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2 := by
  sorry

end equation1_solutions_equation2_solutions_l271_271441


namespace polynomial_transformation_l271_271869

theorem polynomial_transformation (x y : ℂ) (h : y = x + 1/x) : x^4 + x^3 - 4*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 6) = 0 :=
by
  sorry

end polynomial_transformation_l271_271869


namespace biased_coin_die_probability_l271_271721

theorem biased_coin_die_probability :
  let p_heads := 1 / 4
  let p_die_5 := 1 / 8
  p_heads * p_die_5 = 1 / 32 :=
by
  sorry

end biased_coin_die_probability_l271_271721


namespace rhombus_not_diagonals_equal_l271_271933

theorem rhombus_not_diagonals_equal (R : Type) [linear_ordered_field R] 
  (a b c d : R) (h1 : a = b) (h2 : b = c) (h3 : c = d) (h4 : a = d)
  (h_sym : ∀ x y : R, a = b → b = c → c = d → d = a)
  (h_cen_sym : ∀ p : R × R, p = (0, 0) → p = (0, 0)) :
  ¬(∀ p q : R × R, p ≠ q → (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 = (p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) :=
by
  sorry

end rhombus_not_diagonals_equal_l271_271933


namespace problem_x_value_l271_271872

theorem problem_x_value (x : ℝ) (h : (max 3 (max 6 (max 9 x)) * min 3 (min 6 (min 9 x)) = 3 + 6 + 9 + x)) : 
    x = 9 / 4 :=
by
  sorry

end problem_x_value_l271_271872


namespace jordan_annual_income_l271_271134

theorem jordan_annual_income (q : ℝ) (I T : ℝ) 
  (h1 : T = q * 35000 + (q + 3) * (I - 35000))
  (h2 : T = (q + 0.4) * I) : 
  I = 40000 :=
by sorry

end jordan_annual_income_l271_271134


namespace solve_for_x_l271_271867

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (x y : ℝ) (h : 16 * (3:ℝ) ^ x = (7:ℝ) ^ (y + 4)) (hy : y = -4) :
  x = -4 * log 3 2 := by
  sorry

end solve_for_x_l271_271867


namespace playground_length_l271_271210

theorem playground_length
  (P : ℕ)
  (B : ℕ)
  (h1 : P = 1200)
  (h2 : B = 500)
  (h3 : P = 2 * (100 + B)) :
  100 = 100 :=
 by sorry

end playground_length_l271_271210


namespace min_value_of_X_l271_271707

theorem min_value_of_X (n : ℕ) (h : n ≥ 2) 
  (X : Finset ℕ) 
  (B : Fin n → Finset ℕ) 
  (hB : ∀ i, (B i).card = 2) :
  ∃ (Y : Finset ℕ), Y.card = n ∧ ∀ i, (Y ∩ (B i)).card ≤ 1 →
  X.card = 2 * n - 1 :=
sorry

end min_value_of_X_l271_271707


namespace solve_for_y_l271_271246

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l271_271246


namespace area_bounded_by_curve_and_line_l271_271165

theorem area_bounded_by_curve_and_line :
  let curve_x (t : ℝ) := 10 * (t - Real.sin t)
  let curve_y (t : ℝ) := 10 * (1 - Real.cos t)
  let y_line := 15
  (∫ t in (2/3) * Real.pi..(4/3) * Real.pi, 100 * (1 - Real.cos t)^2) = 100 * Real.pi + 200 * Real.sqrt 3 :=
by
  sorry

end area_bounded_by_curve_and_line_l271_271165


namespace express_in_scientific_notation_l271_271282

theorem express_in_scientific_notation :
  ∀ (n : ℕ), n = 1300000 → scientific_notation n = "1.3 × 10^6" :=
by
  intros n h
  have h1 : n = 1300000 := by exact h
  sorry

end express_in_scientific_notation_l271_271282


namespace equation_of_line_l_l271_271076

noncomputable def line_eq (a b c : ℚ) : ℚ → ℚ → Prop := λ x y => a * x + b * y + c = 0

theorem equation_of_line_l : 
  ∃ m : ℚ, 
  (∀ x y : ℚ, 
    (2 * x - 3 * y - 3 = 0 ∧ x + y + 2 = 0 → line_eq 3 1 m x y) ∧ 
    (3 * x + y - 1 = 0 → line_eq 3 1 0 x y)
  ) →
  line_eq 15 5 16 (-3/5) (-7/5) :=
by 
  sorry

end equation_of_line_l_l271_271076


namespace rachel_reading_pages_l271_271238

theorem rachel_reading_pages (M T : ℕ) (hM : M = 10) (hT : T = 23) : T - M = 3 := 
by
  rw [hM, hT]
  norm_num
  sorry

end rachel_reading_pages_l271_271238


namespace correlation_index_l271_271574

variable (height_variation_weight_explained : ℝ)
variable (random_errors_contribution : ℝ)

def R_squared : ℝ := height_variation_weight_explained

theorem correlation_index (h1 : height_variation_weight_explained = 0.64) (h2 : random_errors_contribution = 0.36) : R_squared height_variation_weight_explained = 0.64 :=
by
  exact h1  -- Placeholder for actual proof, since only statement is required

end correlation_index_l271_271574


namespace paper_cups_calculation_l271_271677

def total_pallets : Nat := 20
def paper_towels : Nat := total_pallets / 2
def tissues : Nat := total_pallets / 4
def paper_plates : Nat := total_pallets / 5
def other_paper_products : Nat := paper_towels + tissues + paper_plates
def paper_cups : Nat := total_pallets - other_paper_products

theorem paper_cups_calculation : paper_cups = 1 := by
  sorry

end paper_cups_calculation_l271_271677


namespace carrie_pays_94_l271_271509

theorem carrie_pays_94 :
  ∀ (num_shirts num_pants num_jackets : ℕ) (cost_shirt cost_pants cost_jacket : ℕ),
  num_shirts = 4 →
  cost_shirt = 8 →
  num_pants = 2 →
  cost_pants = 18 →
  num_jackets = 2 →
  cost_jacket = 60 →
  (cost_shirt * num_shirts + cost_pants * num_pants + cost_jacket * num_jackets) / 2 = 94 :=
by
  intros num_shirts num_pants num_jackets cost_shirt cost_pants cost_jacket
  sorry

end carrie_pays_94_l271_271509


namespace num_ways_to_distribute_balls_l271_271865

noncomputable def num_partitions (n k : ℕ) : ℕ :=
  (Finset.powerset (multiset.range (n + k - 1))).card

theorem num_ways_to_distribute_balls :
  num_partitions 6 3 = 6 :=
sorry

end num_ways_to_distribute_balls_l271_271865


namespace circuit_operates_normally_probability_l271_271877

theorem circuit_operates_normally_probability :
  let p1 := 0.5
  let p2 := 0.7
  (1 - (1 - p1) * (1 - p2)) = 0.85 :=
by
  let p1 := 0.5
  let p2 := 0.7
  sorry

end circuit_operates_normally_probability_l271_271877


namespace find_sum_of_cubes_l271_271733

noncomputable def roots (a b c : ℝ) : Prop :=
  5 * a^3 + 2014 * a + 4027 = 0 ∧ 
  5 * b^3 + 2014 * b + 4027 = 0 ∧ 
  5 * c^3 + 2014 * c + 4027 = 0

theorem find_sum_of_cubes (a b c : ℝ) (h : roots a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 :=
sorry

end find_sum_of_cubes_l271_271733


namespace probability_three_heads_in_eight_tosses_l271_271296

theorem probability_three_heads_in_eight_tosses :
  (nat.choose 8 3) / (2 ^ 8) = 7 / 32 := 
begin
  -- This is the starting point for the proof, but the details of the proof are omitted.
  sorry
end

end probability_three_heads_in_eight_tosses_l271_271296


namespace remainder_mod_17_zero_l271_271144

theorem remainder_mod_17_zero :
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  ( (x1 % 17) * (x2 % 17) * (x3 % 17) * (x4 % 17) * (x5 % 17) * (x6 % 17) ) % 17 = 0 :=
by
  let x1 := 2002 + 3
  let x2 := 2003 + 3
  let x3 := 2004 + 3
  let x4 := 2005 + 3
  let x5 := 2006 + 3
  let x6 := 2007 + 3
  sorry

end remainder_mod_17_zero_l271_271144


namespace no_integer_solution_for_triples_l271_271204

theorem no_integer_solution_for_triples :
  ∀ (x y z : ℤ),
    x^2 - 2*x*y + 3*y^2 - z^2 = 17 →
    -x^2 + 4*y*z + z^2 = 28 →
    x^2 + 2*x*y + 5*z^2 = 42 →
    false :=
by
  intros x y z h1 h2 h3
  sorry

end no_integer_solution_for_triples_l271_271204


namespace defect_rate_product_l271_271444

theorem defect_rate_product (P1_defect P2_defect : ℝ) (h1 : P1_defect = 0.10) (h2 : P2_defect = 0.03) : 
  ((1 - P1_defect) * (1 - P2_defect)) = 0.873 → (1 - ((1 - P1_defect) * (1 - P2_defect)) = 0.127) :=
by
  intro h
  sorry

end defect_rate_product_l271_271444


namespace sum_a_for_exactly_one_solution_l271_271525

theorem sum_a_for_exactly_one_solution :
  (∀ a : ℝ, ∃ x : ℝ, 3 * x^2 + (a + 6) * x + 7 = 0) →
  ((-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12) :=
by
  sorry

end sum_a_for_exactly_one_solution_l271_271525


namespace _l271_271716

noncomputable def a : ℕ → ℝ
| 0       := 1/5
| (n + 1) := have h : n + 1 ≠ 0 := nat.succ_ne_zero n
             classical.some (nat_has_inv_iff.mp
               ⟨classical.some (nat_has_inv_iff.mp ⟨(a n : ℝ) - (a n - classical.some (lt_of_le_of_ne 
               (show 0 < a n + 4 * a n * a n from sorry), h) == 0⟩).symm⟩,
              Assume classical.someor_leave reliability here⟩) 

lemma problem1 : ( (λ (n: ℕ), (1 : ℚ) / a n) (0) = 5) : sorry 

lemma problem2 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a (Nat.addSucc 0)  5 -4 a n) :
  ∀ n, a n = (ↄ (λ (n: ℕ), 4 * n + 1)) :
 begin
   have : ∀ n, 
   show (λ n, 4 * n + 1),
   by
   intro 
     classical.some, classical.some_eq(mul((lt_of_le_of_ne
                (4  n + classical.some⟨ 
               NatHasInv.inv (5, (4 * n + some classical.some,(nat_rec))⟩
               refl_sorry . Classical.type_logic_ nat
   else.vue x  ref_succ_eq_zero, smul   (theorem)⟩ ((a), assumption,sorry

lemma sn: ∀target
  (bn: a ( n:Nat)⟺ (sum_target ∑n⟩(((∀k:calclable , 
classical assumption_by theorem 
               show ∀ 
              algebra_ring 1/clas.assumption_eq
  int_them 4  term_scopeof_sum sorry 

by λ target.type:
T_act arithmetic_seq assume terms h∃(a calculation )=&( question:sub
   implies (4_nat_type_interval 
 sum example): problem prove
 (S n<prove_probability  term_range<=sorry 

end _l271_271716


namespace johns_new_total_lift_l271_271888

theorem johns_new_total_lift :
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 30 / 100.0
  let squat_loss := squat_loss_percentage * initial_squat
  let new_squat := initial_squat - squat_loss
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - 200
  new_squat + new_bench + new_deadlift = 1490 := 
by
  -- Proof will go here
  sorry

end johns_new_total_lift_l271_271888


namespace compound_interest_rate_l271_271683

theorem compound_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 10000)
  (hA : A = 12155.06)
  (hn : n = 4)
  (ht : t = 1)
  (h_eq : A = P * (1 + r / n) ^ (n * t)):
  r = 0.2 :=
by
  sorry

end compound_interest_rate_l271_271683


namespace max_axbycz_value_l271_271056

theorem max_axbycz_value (a b c : ℝ) (x y z : ℝ) 
  (h_triangle: a + b > c ∧ b + c > a ∧ c + a > b)
  (h_positive: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) :=
  sorry

end max_axbycz_value_l271_271056


namespace peter_wins_prize_probability_at_least_one_wins_prize_probability_l271_271232

-- Probability of Peter winning a prize:
theorem peter_wins_prize_probability :
  let p := (5 / 6) in p ^ 9 = (5 / 6) ^ 9 := by
  sorry

-- Probability that at least one person wins a prize:
theorem at_least_one_wins_prize_probability :
  let p := (5 / 6) in
  let q := (1 - p^9) in 
  (1 - q^10) ≈ 0.919 := by
  sorry

end peter_wins_prize_probability_at_least_one_wins_prize_probability_l271_271232


namespace second_train_length_l271_271795

theorem second_train_length
  (L1 : ℝ) (V1 : ℝ) (V2 : ℝ) (T : ℝ)
  (h1 : L1 = 300)
  (h2 : V1 = 72 * 1000 / 3600)
  (h3 : V2 = 36 * 1000 / 3600)
  (h4 : T = 79.99360051195904) :
  L1 + (V1 - V2) * T = 799.9360051195904 :=
by
  sorry

end second_train_length_l271_271795


namespace balls_into_boxes_l271_271857

theorem balls_into_boxes : ∃ (n : ℕ), n = 7 ∧ 
  ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 3 → 
    ∃ (partitions : finset (finset (ℕ))), 
      partitions.card = n ∧ 
      ∀ p ∈ partitions, p.sum = balls :=
sorry

end balls_into_boxes_l271_271857


namespace arithmetic_mean_eqn_l271_271590

theorem arithmetic_mean_eqn : 
  (3/5 + 6/7) / 2 = 51/70 :=
  by sorry

end arithmetic_mean_eqn_l271_271590


namespace dice_same_number_probability_l271_271639

noncomputable def same_number_probability : ℚ :=
  (1:ℚ) / 216

theorem dice_same_number_probability :
  (∀ (die1 die2 die3 die4 : ℕ), 
     die1 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die2 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die3 ∈ {1, 2, 3, 4, 5, 6} ∧ 
     die4 ∈ {1, 2, 3, 4, 5, 6} -> 
     die1 = die2 ∧ die1 = die3 ∧ die1 = die4) → same_number_probability = (1 / 216: ℚ)
:=
by
  sorry

end dice_same_number_probability_l271_271639


namespace angle_is_10_l271_271061

theorem angle_is_10 (x : ℕ) (h1 : 180 - x = 2 * (90 - x) + 10) : x = 10 := 
by sorry

end angle_is_10_l271_271061


namespace ratio_triangle_circle_l271_271495

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let A_triangle := (sqrt 3 / 4) * (3 * r)^2
  let A_circle := π * r^2
  A_triangle / A_circle

theorem ratio_triangle_circle (r : ℝ) (h_r : r > 0) :
  ratio_of_areas r = 9 * sqrt 3 / (4 * π) :=
by
  sorry

end ratio_triangle_circle_l271_271495


namespace relationship_of_ys_l271_271740

variable (f : ℝ → ℝ)

def inverse_proportion := ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k / x

theorem relationship_of_ys
  (h_inv_prop : inverse_proportion f)
  (h_pts1 : f (-2) = 3)
  (h_pts2 : f (-3) = y₁)
  (h_pts3 : f 1 = y₂)
  (h_pts4 : f 2 = y₃) :
  y₂ < y₃ ∧ y₃ < y₁ :=
sorry

end relationship_of_ys_l271_271740


namespace chris_birthday_after_45_days_l271_271025

theorem chris_birthday_after_45_days (k : ℕ) (h : k = 45) (tuesday : ℕ) (h_tuesday : tuesday = 2) : 
  (tuesday + k) % 7 = 5 := 
sorry

end chris_birthday_after_45_days_l271_271025


namespace dime_quarter_problem_l271_271725

theorem dime_quarter_problem :
  15 * 25 + 10 * 10 = 25 * 25 + 35 * 10 :=
by
  sorry

end dime_quarter_problem_l271_271725


namespace B_2_2_eq_16_l271_271029

def B : ℕ → ℕ → ℕ
| 0, n       => n + 2
| (m+1), 0   => B m 2
| (m+1), (n+1) => B m (B (m+1) n)

theorem B_2_2_eq_16 : B 2 2 = 16 := by
  sorry

end B_2_2_eq_16_l271_271029


namespace sphere_pyramid_problem_l271_271476

theorem sphere_pyramid_problem (n m : ℕ) :
  (n * (n + 1) * (2 * n + 1)) / 6 + (m * (m + 1) * (m + 2)) / 6 = 605 → n = 10 ∧ m = 10 :=
by
  sorry

end sphere_pyramid_problem_l271_271476


namespace point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l271_271373

-- Question (1): Proving that the point (-2,0) lies on the graph
theorem point_on_graph (k : ℝ) (hk : k ≠ 0) : k * (-2 + 2) = 0 := 
by sorry

-- Question (2): Finding the value of k given a shifted graph passing through a point
theorem find_k_shifted_graph_passing (k : ℝ) : (k * (1 + 2) + 2 = -2) → k = -4/3 := 
by sorry

-- Question (3): Proving the range of k for the function's y-intercept within given limits
theorem y_axis_intercept_range (k : ℝ) (hk : -2 < 2 * k ∧ 2 * k < 0) : -1 < k ∧ k < 0 := 
by sorry

end point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l271_271373


namespace connected_geometric_seq_a10_l271_271843

noncomputable def is_kth_order_geometric (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + k) = q * a n

theorem connected_geometric_seq_a10 (a : ℕ → ℝ) 
  (h : is_kth_order_geometric a 3) 
  (a1 : a 1 = 1) 
  (a4 : a 4 = 2) : 
  a 10 = 8 :=
sorry

end connected_geometric_seq_a10_l271_271843


namespace sum_of_positive_factors_of_90_eq_234_l271_271773

theorem sum_of_positive_factors_of_90_eq_234 : 
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  List.sum factors = 234 :=
by
  -- List the positive factors of 90
  let factors := [1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90]
  -- Prove that the sum of these factors is 234
  have h_sum_factors : List.sum factors = 234 := sorry
  exact h_sum_factors

end sum_of_positive_factors_of_90_eq_234_l271_271773


namespace victor_total_money_l271_271769

def initial_amount : ℕ := 10
def allowance : ℕ := 8
def total_amount : ℕ := initial_amount + allowance

theorem victor_total_money : total_amount = 18 := by
  -- This is where the proof steps would go
  sorry

end victor_total_money_l271_271769


namespace smallest_positive_period_of_f_max_min_values_of_f_l271_271203

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 + Real.sqrt 2) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l271_271203


namespace dice_probability_same_face_l271_271630

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l271_271630


namespace equation1_equation2_equation3_equation4_l271_271111

-- 1. Solve: 2(2x-1)^2 = 8
theorem equation1 (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ (x = 3/2) ∨ (x = -1/2) :=
sorry

-- 2. Solve: 2x^2 + 3x - 2 = 0
theorem equation2 (x : ℝ) : 2 * x^2 + 3 * x - 2 = 0 ↔ (x = 1/2) ∨ (x = -2) :=
sorry

-- 3. Solve: x(2x-7) = 3(2x-7)
theorem equation3 (x : ℝ) : x * (2 * x - 7) = 3 * (2 * x - 7) ↔ (x = 7/2) ∨ (x = 3) :=
sorry

-- 4. Solve: 2y^2 + 8y - 1 = 0
theorem equation4 (y : ℝ) : 2 * y^2 + 8 * y - 1 = 0 ↔ (y = (-4 + 3 * Real.sqrt 2) / 2) ∨ (y = (-4 - 3 * Real.sqrt 2) / 2) :=
sorry

end equation1_equation2_equation3_equation4_l271_271111


namespace two_tangents_from_origin_l271_271714

theorem two_tangents_from_origin
  (f : ℝ → ℝ)
  (a : ℝ)
  (h₁ : ∀ x, f(x) = a * x^3 + 3 * x^2 + 1)
  (h₂ : ∃ m₁ m₂ : ℝ, 
         m₁ ≠ m₂ ∧ 
         (f(-m₁) + f(m₂ + 2)) = 2 * f(1) ∧ 
         (f(-m₂) + f(m₁ + 2)) = 2 * f(1)) :
  ∃! t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (∃ (y₁ y₂: ℝ), y₁ = f(t₁) ∧ y₂ = f(t₂) ∧ 
                   y₁ / t₁ = (-y₁ + 3 * x * t₁) / t₁ ∧ 
                   y₂ / t₂ = (-y₂ + 3 * x * t₂) / t₂) :=
sorry

end two_tangents_from_origin_l271_271714


namespace part_a_solutions_l271_271943

theorem part_a_solutions (x : ℝ) : (⌊x⌋^2 - x = -0.99) ↔ (x = 0.99 ∨ x = 1.99) :=
sorry

end part_a_solutions_l271_271943


namespace probability_of_three_heads_in_eight_tosses_l271_271303

theorem probability_of_three_heads_in_eight_tosses :
  (∃ n : ℚ, n = 7 / 32) :=
begin
  sorry
end

end probability_of_three_heads_in_eight_tosses_l271_271303


namespace compare_answers_l271_271879

def num : ℕ := 384
def correct_answer : ℕ := (5 * num) / 16
def students_answer : ℕ := (5 * num) / 6
def difference : ℕ := students_answer - correct_answer

theorem compare_answers : difference = 200 := 
by
  sorry

end compare_answers_l271_271879


namespace f_increasing_f_t_range_l271_271997

noncomputable def f : Real → Real :=
  sorry

axiom f_prop1 : f 2 = 1
axiom f_prop2 : ∀ x, x > 1 → f x > 0
axiom f_prop3 : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

theorem f_increasing (x1 x2 : Real) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) : f x1 < f x2 := by
  sorry

theorem f_t_range (t : Real) (ht : t > 0) (ht3 : t - 3 > 0) (hf : f t + f (t - 3) ≤ 2) : 3 < t ∧ t ≤ 4 := by
  sorry

end f_increasing_f_t_range_l271_271997


namespace eyes_that_saw_the_plane_l271_271458

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end eyes_that_saw_the_plane_l271_271458


namespace avg_first_3_is_6_l271_271118

theorem avg_first_3_is_6 (A B C D : ℝ) (X : ℝ)
  (h1 : (A + B + C) / 3 = X)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11)
  (h4 : D = 4) :
  X = 6 := 
by
  sorry

end avg_first_3_is_6_l271_271118


namespace red_marbles_in_A_l271_271459

-- Define the number of marbles in baskets A, B, and C
variables (R : ℕ)
def basketA := R + 2 -- Basket A: R red, 2 yellow
def basketB := 6 + 1 -- Basket B: 6 green, 1 yellow
def basketC := 3 + 9 -- Basket C: 3 white, 9 yellow

-- Define the greatest difference condition
def greatest_difference (A B C : ℕ) := max (max (A - B) (B - C)) (max (A - C) (C - B))

-- Define the hypothesis based on the conditions
axiom H1 : greatest_difference 3 9 0 = 6

-- The theorem we need to prove: The number of red marbles in Basket A is 8
theorem red_marbles_in_A : R = 8 := 
by {
  -- The proof would go here, but we'll use sorry to skip it
  sorry
}

end red_marbles_in_A_l271_271459


namespace same_face_probability_l271_271647

-- Definitions of the conditions for the problem
def six_sided_die_probability (outcomes : ℕ) : ℚ :=
  if outcomes = 6 then 1 else 0

def probability_same_face (first_second := 1/6) (first_third := 1/6) (first_fourth := 1/6) : ℚ :=
  first_second * first_third * first_fourth

-- Statement of the theorem
theorem same_face_probability : (six_sided_die_probability 6) * probability_same_face = 1/216 :=
  by sorry

end same_face_probability_l271_271647


namespace geometric_sequence_problem_l271_271712

noncomputable def q : ℝ := 1 + Real.sqrt 2

theorem geometric_sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = (q : ℝ) * a n)
  (h_cond : a 2 = a 0 + 2 * a 1) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := 
sorry

end geometric_sequence_problem_l271_271712


namespace checkerboard_7_strips_l271_271528

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l271_271528


namespace gcd_lcm_product_l271_271362

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l271_271362


namespace solve_system_l271_271599

theorem solve_system :
  ∃ x y : ℚ, 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 :=
by {
  sorry
}

end solve_system_l271_271599


namespace ruby_initial_apples_l271_271745

theorem ruby_initial_apples (apples_taken : ℕ) (apples_left : ℕ) (initial_apples : ℕ) 
  (h1 : apples_taken = 55) (h2 : apples_left = 8) (h3 : initial_apples = apples_taken + apples_left) : 
  initial_apples = 63 := 
by
  sorry

end ruby_initial_apples_l271_271745


namespace seating_arrangements_l271_271265

theorem seating_arrangements (n : ℕ) (h_n : n = 6) (A B : Fin n) (h : A ≠ B) : 
  ∃ k : ℕ, k = 240 := 
by 
  sorry

end seating_arrangements_l271_271265


namespace area_of_square_l271_271275

theorem area_of_square (a : ℝ) (h : a = 12) : a * a = 144 := by
  rw [h]
  norm_num

end area_of_square_l271_271275


namespace curve_is_circle_l271_271704

theorem curve_is_circle (s : ℝ) :
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  x^2 + y^2 = 1 :=
by
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  sorry

end curve_is_circle_l271_271704


namespace initial_pretzels_in_bowl_l271_271916

-- Definitions and conditions
def John_pretzels := 28
def Alan_pretzels := John_pretzels - 9
def Marcus_pretzels := John_pretzels + 12
def Marcus_pretzels_actual := 40

-- The main theorem stating the initial number of pretzels in the bowl
theorem initial_pretzels_in_bowl : 
  Marcus_pretzels = Marcus_pretzels_actual → 
  John_pretzels + Alan_pretzels + Marcus_pretzels = 87 :=
by
  intro h
  sorry -- proof to be filled in

end initial_pretzels_in_bowl_l271_271916


namespace same_number_probability_four_dice_l271_271641

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l271_271641


namespace problem_l271_271201

namespace arithmetic_sequence

def is_arithmetic_sequence (a : ℕ → ℚ) := ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem 
  (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_cond : a 1 + a 7 + a 13 = 4) : a 2 + a 12 = 8 / 3 :=
sorry

end arithmetic_sequence

end problem_l271_271201


namespace no_integer_solution_for_euler_conjecture_l271_271742

theorem no_integer_solution_for_euler_conjecture :
  ¬(∃ n : ℕ, 5^4 + 12^4 + 9^4 + 8^4 = n^4) :=
by
  -- Sum of the given fourth powers
  have lhs : ℕ := 5^4 + 12^4 + 9^4 + 8^4
  -- Direct proof skipped with sorry
  sorry

end no_integer_solution_for_euler_conjecture_l271_271742


namespace area_of_tangents_l271_271531

def radius := 3
def segment_length := 6

theorem area_of_tangents (r : ℝ) (l : ℝ) (h1 : r = radius) (h2 : l = segment_length) :
  let R := r * Real.sqrt 2 
  let annulus_area := π * (R ^ 2) - π * (r ^ 2)
  annulus_area = 9 * π :=
by
  sorry

end area_of_tangents_l271_271531


namespace distinct_real_number_sum_and_square_sum_eq_l271_271586

theorem distinct_real_number_sum_and_square_sum_eq
  (a b c d : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3)
  (h_square_sum : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / (a - b) / (a - c) / (a - d)) + (b^5 / (b - a) / (b - c) / (b - d)) +
  (c^5 / (c - a) / (c - b) / (c - d)) + (d^5 / (d - a) / (d - b) / (d - c)) = -9 :=
by
  sorry

end distinct_real_number_sum_and_square_sum_eq_l271_271586


namespace same_number_probability_four_dice_l271_271643

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l271_271643


namespace probability_three_heads_in_eight_tosses_l271_271339

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l271_271339


namespace Carrie_pays_94_l271_271503

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l271_271503


namespace maximum_term_of_sequence_l271_271387

noncomputable def a (n : ℕ) : ℝ := n * (3 / 4)^n

theorem maximum_term_of_sequence : ∃ n : ℕ, a n = a 3 ∧ ∀ m : ℕ, a m ≤ a 3 :=
by sorry

end maximum_term_of_sequence_l271_271387


namespace eli_age_difference_l271_271730

theorem eli_age_difference (kaylin_age : ℕ) (freyja_age : ℕ) (sarah_age : ℕ) (eli_age : ℕ) 
  (H1 : kaylin_age = 33)
  (H2 : freyja_age = 10)
  (H3 : kaylin_age + 5 = sarah_age)
  (H4 : sarah_age = 2 * eli_age) :
  eli_age - freyja_age = 9 := 
sorry

end eli_age_difference_l271_271730


namespace probability_of_exactly_three_heads_l271_271291

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l271_271291


namespace cake_and_milk_tea_cost_l271_271122

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l271_271122


namespace Tod_speed_is_25_mph_l271_271920

-- Definitions of the conditions
def miles_north : ℕ := 55
def miles_west : ℕ := 95
def hours_driven : ℕ := 6

-- The total distance travelled
def total_distance : ℕ := miles_north + miles_west

-- The speed calculation, dividing total distance by hours driven
def speed : ℕ := total_distance / hours_driven

-- The theorem to prove
theorem Tod_speed_is_25_mph : speed = 25 :=
by
  -- Proof of the theorem will be filled here, but for now using sorry
  sorry

end Tod_speed_is_25_mph_l271_271920


namespace peter_wins_prize_at_least_one_wins_prize_l271_271237

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l271_271237


namespace three_digit_number_formed_by_1198th_1200th_digits_l271_271491

def albertSequenceDigit (n : ℕ) : ℕ :=
  -- Define the nth digit in Albert's sequence
  sorry

theorem three_digit_number_formed_by_1198th_1200th_digits :
  let d1198 := albertSequenceDigit 1198
  let d1199 := albertSequenceDigit 1199
  let d1200 := albertSequenceDigit 1200
  (d1198 * 100 + d1199 * 10 + d1200) = 220 :=
by
  sorry

end three_digit_number_formed_by_1198th_1200th_digits_l271_271491


namespace combined_total_l271_271263

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l271_271263


namespace yoongi_has_fewest_apples_l271_271941

noncomputable def yoongi_apples : ℕ := 4
noncomputable def yuna_apples : ℕ := 5
noncomputable def jungkook_apples : ℕ := 6 * 3

theorem yoongi_has_fewest_apples : yoongi_apples < yuna_apples ∧ yoongi_apples < jungkook_apples := by
  sorry

end yoongi_has_fewest_apples_l271_271941


namespace quadratic_expression_and_intersections_l271_271199

noncomputable def quadratic_eq_expression (a b c : ℝ) : Prop :=
  ∃ a b c : ℝ, (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = -3) ∧ (4 * a + 2 * b + c = - 5 / 2) ∧ (b = -2 * a) ∧ (c = -5 / 2) ∧ (a = 1 / 2)

noncomputable def find_m (a b c : ℝ) : Prop :=
  ∀ x m : ℝ, (a * (-2:ℝ)^2 + b * (-2:ℝ) + c = m) → (a * (4:ℝ) + b * (4:ℝ) + c = m) → (6:ℝ) = abs (x - (-2:ℝ)) → m = 3 / 2

noncomputable def y_range (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, 
  (x^2 * a + x * b + c >= -3) ∧ 
  (x^2 * a + x * b + c < 5) ↔ (-3 < x ∧ x < 3)

theorem quadratic_expression_and_intersections 
  (a b c : ℝ) (h1 : quadratic_eq_expression a b c) (h2 : find_m a b c) : y_range a b c :=
  sorry

end quadratic_expression_and_intersections_l271_271199


namespace cos_neg_pi_div_3_l271_271693

theorem cos_neg_pi_div_3 : Real.cos (-π / 3) = 1 / 2 := 
by 
  sorry

end cos_neg_pi_div_3_l271_271693


namespace pallets_of_paper_cups_l271_271674

theorem pallets_of_paper_cups (total_pallets paper_towels tissues paper_plates : ℕ) 
  (H1 : total_pallets = 20) 
  (H2 : paper_towels = total_pallets / 2)
  (H3 : tissues = total_pallets / 4)
  (H4 : paper_plates = total_pallets / 5) : 
  total_pallets - paper_towels - tissues - paper_plates = 1 := 
  by
    sorry

end pallets_of_paper_cups_l271_271674


namespace probability_three_heads_in_eight_tosses_l271_271337

open Nat

-- Define the conditions for a fair coin tossed 8 times
def coinTosses : ℕ := 8

-- Define the exact number of heads we're interested in
def heads : ℕ := 3

-- Calculate the total number of sequences
def totalSequences : ℕ := 2 ^ coinTosses

-- Calculate the number of favorable sequences (exactly 3 heads)
def favorableSequences : ℕ := choose coinTosses heads

-- Calculate the probability as a fraction
def probability : ℚ := favorableSequences / totalSequences

-- The statement to prove
theorem probability_three_heads_in_eight_tosses :
  probability = 7 / 32 :=
by 
  sorry

end probability_three_heads_in_eight_tosses_l271_271337


namespace range_of_m_l271_271049

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x - 3| ≤ 2 → 1 ≤ x ∧ x ≤ 5) → 
  (∀ x : ℝ, (x - m + 1) * (x - m - 1) ≤ 0 → m - 1 ≤ x ∧ x ≤ m + 1) → 
  (∀ x : ℝ, x < 1 ∨ x > 5 → x < m - 1 ∨ x > m + 1) → 
  2 ≤ m ∧ m ≤ 4 := 
by
  sorry

end range_of_m_l271_271049


namespace same_number_probability_four_dice_l271_271640

theorem same_number_probability_four_dice : 
  let outcomes := 6
  in (1 / outcomes) * (1 / outcomes) * (1 / outcomes) = 1 / 216 :=
by
  let outcomes := 6
  sorry

end same_number_probability_four_dice_l271_271640


namespace Carrie_pays_94_l271_271504

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l271_271504


namespace nuts_per_box_l271_271822

theorem nuts_per_box (N : ℕ)  
  (h1 : ∀ (boxes bolts_per_box : ℕ), boxes = 7 ∧ bolts_per_box = 11 → boxes * bolts_per_box = 77)
  (h2 : ∀ (boxes: ℕ), boxes = 3 → boxes * N = 3 * N)
  (h3 : ∀ (used_bolts purchased_bolts remaining_bolts : ℕ), purchased_bolts = 77 ∧ remaining_bolts = 3 → used_bolts = purchased_bolts - remaining_bolts)
  (h4 : ∀ (used_nuts purchased_nuts remaining_nuts : ℕ), purchased_nuts = 3 * N ∧ remaining_nuts = 6 → used_nuts = purchased_nuts - remaining_nuts)
  (h5 : ∀ (used_bolts used_nuts total_used : ℕ), used_bolts = 74 ∧ used_nuts = 3 * N - 6 → total_used = used_bolts + used_nuts)
  (h6 : total_used_bolts_and_nuts = 113) :
  N = 15 :=
by
  sorry

end nuts_per_box_l271_271822


namespace average_income_family_l271_271264

theorem average_income_family (income1 income2 income3 income4 : ℕ) 
  (h1 : income1 = 8000) (h2 : income2 = 15000) (h3 : income3 = 6000) (h4 : income4 = 11000) :
  (income1 + income2 + income3 + income4) / 4 = 10000 := by
  sorry

end average_income_family_l271_271264


namespace area_of_f2_equals_7_l271_271557

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_of_f2_equals_7 : 
  (∫ x in (-3 : ℝ)..3, f2 x) = 7 :=
by
  sorry

end area_of_f2_equals_7_l271_271557


namespace balls_in_boxes_l271_271854

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), (balls = 6) → (boxes = 3) → 
  (∃ ways : ℕ, ways = 7) :=
by
  sorry

end balls_in_boxes_l271_271854


namespace michael_laps_to_pass_donovan_l271_271815

theorem michael_laps_to_pass_donovan (track_length : ℕ) (donovan_lap_time : ℕ) (michael_lap_time : ℕ) 
  (h1 : track_length = 400) (h2 : donovan_lap_time = 48) (h3 : michael_lap_time = 40) : 
  michael_lap_time * 6 = donovan_lap_time * (michael_lap_time * 6 / track_length * michael_lap_time) :=
by
  sorry

end michael_laps_to_pass_donovan_l271_271815


namespace general_term_sum_bn_l271_271202

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + 2 * n
noncomputable def a (n : ℕ) : ℕ := 4 * n
noncomputable def b (n : ℕ) : ℕ := 2 ^ (4 * n)
noncomputable def T (n : ℕ) : ℝ := (16 / 15) * (16^n - 1)

theorem general_term (n : ℕ) (h1 : S n = 2 * n^2 + 2 * n) 
    (h2 : S (n-1) = 2 * (n-1)^2 + 2 * (n-1))
    (h3 : n ≥ 1) : a n = 4 * n :=
by sorry

theorem sum_bn (n : ℕ) (h : ∀ n, (b n, a n) = ((2 ^ (4 * n)), 4 * n)) : 
    T n = (16 / 15) * (16^n - 1) :=
by sorry

end general_term_sum_bn_l271_271202


namespace polynomial_coefficients_sum_even_odd_coefficients_difference_square_l271_271222

theorem polynomial_coefficients_sum (a : Fin 8 → ℝ):
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 3^7 - 1 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

theorem even_odd_coefficients_difference_square (a : Fin 8 → ℝ):
  (a 0 + a 2 + a 4 + a 6)^2 - (a 1 + a 3 + a 5 + a 7)^2 = -3^7 :=
by
  -- assume the polynomial (1 + 2x)^7 has coefficients a 0, a 1, ..., a 7
  -- such that (1 + 2x)^7 = a 0 + a 1 * x + a 2 * x^2 + ... + a 7 * x^7
  sorry

end polynomial_coefficients_sum_even_odd_coefficients_difference_square_l271_271222


namespace spring_stretch_150N_l271_271010

-- Definitions for the conditions
def spring_stretch (weight : ℕ) : ℕ :=
  if weight = 100 then 20 else sorry

-- The theorem to prove
theorem spring_stretch_150N : spring_stretch 150 = 30 := by
  sorry

end spring_stretch_150N_l271_271010


namespace second_student_marks_l271_271273

theorem second_student_marks (x y : ℝ) 
  (h1 : x = y + 9) 
  (h2 : x = 0.56 * (x + y)) : 
  y = 33 := 
sorry

end second_student_marks_l271_271273


namespace six_digit_numbers_with_at_least_two_zeros_l271_271844

theorem six_digit_numbers_with_at_least_two_zeros : 
  (∃ n : ℕ, n = 900000) → 
  (∃ no_zero : ℕ, no_zero = 531441) → 
  (∃ one_zero : ℕ, one_zero = 295245) → 
  (∃ at_least_two_zeros : ℕ, at_least_two_zeros = 900000 - (531441 + 295245)) → 
  at_least_two_zeros = 73314 :=
by
  intros n no_zero one_zero at_least_two_zeros
  rw [at_least_two_zeros, n, no_zero, one_zero]
  norm_num
  sorry

end six_digit_numbers_with_at_least_two_zeros_l271_271844


namespace nicholas_crackers_l271_271226

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) 
  (h1 : marcus_crackers = 3 * mona_crackers)
  (h2 : nicholas_crackers = mona_crackers + 6)
  (h3 : marcus_crackers = 27) : nicholas_crackers = 15 := by
  sorry

end nicholas_crackers_l271_271226


namespace apple_equals_pear_l271_271427

-- Define the masses of the apple and pear.
variable (A G : ℝ)

-- The equilibrium condition on the balance scale.
axiom equilibrium_condition : A + 2 * G = 2 * A + G

-- Prove the mass of an apple equals the mass of a pear.
theorem apple_equals_pear (A G : ℝ) (h : A + 2 * G = 2 * A + G) : A = G :=
by
  -- Proof goes here, but we use sorry to indicate the proof's need.
  sorry

end apple_equals_pear_l271_271427


namespace coeff_x3_in_product_l271_271520

theorem coeff_x3_in_product :
  let p1 := 3 * (Polynomial.X ^ 3) + 4 * (Polynomial.X ^ 2) + 5 * Polynomial.X + 6
  let p2 := 7 * (Polynomial.X ^ 2) + 8 * Polynomial.X + 9
  (Polynomial.coeff (p1 * p2) 3) = 94 :=
by
  sorry

end coeff_x3_in_product_l271_271520


namespace socks_headband_probability_l271_271090

/-- Keisha's basketball team uniform color probability -/
theorem socks_headband_probability :
  let socks_colors := {red, blue}
  let headband_colors := {red, blue, green}
  let total_combinations := (socks_colors.card * headband_colors.card)
  let matching_combinations := 2  -- (red with red, blue with blue)
  let non_matching_combinations := total_combinations - matching_combinations
  let probability := non_matching_combinations.to_rat / total_combinations.to_rat
  probability = (2 / 3) :=
by
  sorry

end socks_headband_probability_l271_271090


namespace polar_to_rectangular_l271_271167

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 7) (h_θ : θ = π / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) :=
by 
  -- proof goes here
  sorry

end polar_to_rectangular_l271_271167


namespace sample_standard_deviation_same_sample_ranges_same_l271_271191

variables {n : ℕ} (x y : Fin n → ℝ) (c : ℝ)
  (h_y : ∀ i, y i = x i + c)
  (h_c_ne_zero : c ≠ 0)

-- Statement for standard deviations being the same
theorem sample_standard_deviation_same :
  let mean (s : Fin n → ℝ) := (∑ i, s i) / n
  in let stddev (s : Fin n → ℝ) := sqrt ((∑ i, (s i - mean s) ^ 2) / n)
  in stddev x = stddev y := 
sorry

-- Statement for ranges being the same
theorem sample_ranges_same :
  let range (s : Fin n → ℝ) := (Finset.univ.sup s) - (Finset.univ.inf s)
  in range x = range y :=
sorry

end sample_standard_deviation_same_sample_ranges_same_l271_271191


namespace value_of_m_l271_271072

theorem value_of_m (m : ℤ) (h : m + 1 = - (-2)) : m = 1 :=
sorry

end value_of_m_l271_271072


namespace increase_average_l271_271117

variable (total_runs : ℕ) (innings : ℕ) (average : ℕ) (new_runs : ℕ) (x : ℕ)

theorem increase_average (h1 : innings = 10) 
                         (h2 : average = 30) 
                         (h3 : total_runs = average * innings) 
                         (h4 : new_runs = 74) 
                         (h5 : total_runs + new_runs = (average + x) * (innings + 1)) :
    x = 4 := 
sorry

end increase_average_l271_271117


namespace triangle_side_count_l271_271453

theorem triangle_side_count
    (x : ℝ)
    (h1 : x + 15 > 40)
    (h2 : x + 40 > 15)
    (h3 : 15 + 40 > x)
    (hx : ∃ (x : ℕ), 25 < x ∧ x < 55) :
    ∃ n : ℕ, n = 29 :=
by
  sorry

end triangle_side_count_l271_271453


namespace contest_score_order_l271_271568

variables (E F G H : ℕ) -- nonnegative scores of Emily, Fran, Gina, and Harry respectively

-- Conditions
axiom cond1 : E - F = G + H + 10
axiom cond2 : G + E > F + H + 5
axiom cond3 : H = F + 8

-- Statement to prove
theorem contest_score_order : (H > E) ∧ (E > F) ∧ (F > G) :=
sorry

end contest_score_order_l271_271568


namespace coin_toss_probability_l271_271328

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l271_271328


namespace find_number_l271_271472

theorem find_number (x : ℝ) (h : x - (3/5 : ℝ) * x = 60) : x = 150 :=
sorry

end find_number_l271_271472


namespace quotient_of_large_div_small_l271_271899

theorem quotient_of_large_div_small (L S : ℕ) (h1 : L - S = 1365)
  (h2 : L = S * (L / S) + 20) (h3 : L = 1634) : (L / S) = 6 := by
  sorry

end quotient_of_large_div_small_l271_271899


namespace square_pizza_area_larger_by_27_percent_l271_271951

theorem square_pizza_area_larger_by_27_percent :
  let r := 5
  let A_circle := Real.pi * r^2
  let s := 2 * r
  let A_square := s^2
  let delta_A := A_square - A_circle
  let percent_increase := (delta_A / A_circle) * 100
  Int.floor (percent_increase + 0.5) = 27 :=
by
  sorry

end square_pizza_area_larger_by_27_percent_l271_271951


namespace downstream_distance_l271_271474

theorem downstream_distance (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ) (distance : ℝ) :
  speed_boat = 20 ∧ speed_current = 5 ∧ time_minutes = 24 ∧ distance = 10 →
  (speed_boat + speed_current) * (time_minutes / 60) = distance :=
by
  sorry

end downstream_distance_l271_271474


namespace minimum_value_of_reciprocal_squares_l271_271924

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l271_271924


namespace fill_entire_bucket_l271_271784

theorem fill_entire_bucket (h : (2/3 : ℝ) * t = 2) : t = 3 :=
sorry

end fill_entire_bucket_l271_271784


namespace polynomial_transformation_l271_271587

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_transformation (x : ℝ) :
  (f (x^2 + 2) = x^4 + 6 * x^2 + 4) →
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  intro h
  sorry

end polynomial_transformation_l271_271587


namespace wait_time_at_least_8_l271_271679

-- Define the conditions
variables (p₀ p : ℝ) (r x : ℝ)

-- Given conditions
def initial_BAC := p₀ = 89
def BAC_after_2_hours := p = 61
def BAC_decrease := p = p₀ * (Real.exp (r * x))
def decrease_in_2_hours := p = 89 * (Real.exp (r * 2))

-- The main goal to prove the time required is at least 8 hours
theorem wait_time_at_least_8 (h1 : p₀ = 89) (h2 : p = 61) (h3 : p = p₀ * Real.exp (r * x)) (h4 : 61 = 89 * Real.exp (2 * r)) : 
  ∃ x, 89 * Real.exp (r * x) < 20 ∧ x ≥ 8 :=
sorry

end wait_time_at_least_8_l271_271679


namespace volunteer_selection_count_l271_271438

open Nat

theorem volunteer_selection_count :
  let boys : ℕ := 5
  let girls : ℕ := 2
  let total_ways := choose girls 1 * choose boys 2 + choose girls 2 * choose boys 1
  total_ways = 25 :=
by
  sorry

end volunteer_selection_count_l271_271438


namespace cakes_bought_l271_271498

theorem cakes_bought (initial : ℕ) (left : ℕ) (bought : ℕ) :
  initial = 169 → left = 32 → bought = initial - left → bought = 137 :=
by
  intros h_initial h_left h_bought
  rw [h_initial, h_left] at h_bought
  exact h_bought

end cakes_bought_l271_271498


namespace obtuse_triangle_side_range_l271_271548

theorem obtuse_triangle_side_range (a : ℝ) (h1 : 0 < a)
  (h2 : a + (a + 1) > a + 2)
  (h3 : (a + 1) + (a + 2) > a)
  (h4 : (a + 2) + a > a + 1)
  (h5 : (a + 2)^2 > a^2 + (a + 1)^2) : 1 < a ∧ a < 3 :=
by
  -- proof omitted
  sorry

end obtuse_triangle_side_range_l271_271548


namespace andrew_stamps_permits_l271_271970

theorem andrew_stamps_permits (n a T r permits : ℕ)
  (h1 : n = 2)
  (h2 : a = 3)
  (h3 : T = 8)
  (h4 : r = 50)
  (h5 : permits = (T - n * a) * r) :
  permits = 100 :=
by
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end andrew_stamps_permits_l271_271970


namespace probability_of_exactly_three_heads_l271_271293

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l271_271293


namespace peter_wins_prize_at_least_one_wins_prize_l271_271236

noncomputable def probability_peter_wins (N : Nat) := 
  (5/6) ^ (N - 1)

theorem peter_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob = probability_peter_wins N := by
  have h2 : prob = (5/6) ^ 9 := by
    sorry
  exact h2

noncomputable def probability_at_least_one_wins (N : Nat) := 
  10 * (5/6)^9 - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

theorem at_least_one_wins_prize (N : Nat) (prob : Real) (h1 : N = 10) : 
  prob ≈ probability_at_least_one_wins N := by
  have h2 : prob ≈ 0.919 := by
    sorry
  exact h2

end peter_wins_prize_at_least_one_wins_prize_l271_271236


namespace min_value_inverse_sum_l271_271622

variable (m n : ℝ)
variable (hm : 0 < m)
variable (hn : 0 < n)
variable (b : ℝ) (hb : b = 2)
variable (hline : 3 * m + n = 1)

theorem min_value_inverse_sum : 
  (1 / m + 4 / n) = 7 + 4 * Real.sqrt 3 :=
  sorry

end min_value_inverse_sum_l271_271622


namespace find_largest_m_l271_271375

variables (a b c t : ℝ)
def f (x : ℝ) := a * x^2 + b * x + c

theorem find_largest_m (a_ne_zero : a ≠ 0)
  (cond1 : ∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x) ∧ f a b c x ≥ x)
  (cond2 : ∀ x : ℝ, 0 < x ∧ x < 2 → f a b c x ≤ ((x + 1) / 2)^2)
  (cond3 : ∃ x : ℝ, ∀ y : ℝ, f a b c y ≥ f a b c x ∧ f a b c x = 0) :
  ∃ m : ℝ, 1 < m ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x) ∧ m = 9 := sorry

end find_largest_m_l271_271375


namespace margin_expression_l271_271402

variable (n : ℕ) (C S M : ℝ)

theorem margin_expression (H1 : M = (1 / n) * C) (H2 : C = S - M) : 
  M = (1 / (n + 1)) * S := 
by
  sorry

end margin_expression_l271_271402


namespace factorize_x4_plus_81_l271_271355

noncomputable def factorize_poly (x : ℝ) : (ℝ × ℝ) :=
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  (p, q)

theorem factorize_x4_plus_81 : ∀ x : ℝ, (x^4 + 81) = (factorize_poly x).fst * (factorize_poly x).snd := by
  intro x
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  have h : x^4 + 81 = p * q
  { sorry }
  exact h

end factorize_x4_plus_81_l271_271355


namespace min_value_of_reciprocal_squares_l271_271921

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l271_271921


namespace eyes_that_saw_airplane_l271_271457

theorem eyes_that_saw_airplane (students : ℕ) (looked_up_fraction : ℚ) (eyes_per_student : ℕ) :
  students = 200 → looked_up_fraction = 3/4 → eyes_per_student = 2 → looked_up_fraction * students * eyes_per_student = 300 :=
by
  intros hstudents hlooked_up_fraction heyes_per_student
  rw [hstudents, hlooked_up_fraction, heyes_per_student]
  norm_num
  sorry

end eyes_that_saw_airplane_l271_271457


namespace total_rods_required_l271_271482

-- Define the number of rods needed per unit for each type
def rods_per_sheet_A : ℕ := 10
def rods_per_sheet_B : ℕ := 8
def rods_per_sheet_C : ℕ := 12
def rods_per_beam_A : ℕ := 6
def rods_per_beam_B : ℕ := 4
def rods_per_beam_C : ℕ := 5

-- Define the composition per panel
def sheets_A_per_panel : ℕ := 2
def sheets_B_per_panel : ℕ := 1
def beams_C_per_panel : ℕ := 2

-- Define the number of panels
def num_panels : ℕ := 10

-- Prove the total number of metal rods required for the entire fence
theorem total_rods_required : 
  (sheets_A_per_panel * rods_per_sheet_A + 
   sheets_B_per_panel * rods_per_sheet_B +
   beams_C_per_panel * rods_per_beam_C) * num_panels = 380 :=
by 
  sorry

end total_rods_required_l271_271482


namespace x_squared_minus_y_squared_l271_271075

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l271_271075


namespace probability_of_exactly_three_heads_l271_271292

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l271_271292


namespace rectangle_area_l271_271957

theorem rectangle_area (x : ℝ) (h1 : (x^2 + (3*x)^2) = (15*Real.sqrt 2)^2) :
  (x * (3 * x)) = 135 := 
by
  sorry

end rectangle_area_l271_271957


namespace total_cost_correct_l271_271021

def cost_first_day : Nat := 4 + 5 + 3 + 2
def cost_second_day : Nat := 5 + 6 + 4
def total_cost : Nat := cost_first_day + cost_second_day

theorem total_cost_correct : total_cost = 29 := by
  sorry

end total_cost_correct_l271_271021


namespace probability_same_group_l271_271765

theorem probability_same_group :
  let total_cards := 20
  let drawn_cards := 4
  let specific_cards := {5, 14}
  let prob := 7 / 51
  let same_group_probability := 
    if ((specific_cards.toList.nth 0 < specific_cards.toList.nth 1) || (specific_cards.toList.nth 1 < specific_cards.toList.nth 0)) 
    then prob 
    else 0
  same_group_probability = prob :=
begin
  sorry
end

end probability_same_group_l271_271765


namespace point_not_on_line_l271_271723

theorem point_not_on_line
  (p q : ℝ)
  (h : p * q > 0) :
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by
  sorry

end point_not_on_line_l271_271723


namespace elmer_more_than_penelope_l271_271430

def penelope_food_per_day : ℕ := 20
def greta_food_factor : ℕ := 10
def milton_food_factor : ℤ := 1 / 100
def elmer_food_factor : ℕ := 4000

theorem elmer_more_than_penelope :
  (elmer_food_factor * (milton_food_factor * (penelope_food_per_day / greta_food_factor))) - penelope_food_per_day = 60 := 
sorry

end elmer_more_than_penelope_l271_271430


namespace arithmetic_progression_sum_l271_271543

variable {α : Type*} [LinearOrderedField α]

def arithmetic_progression (S : ℕ → α) :=
  ∃ (a d : α), ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_progression_sum :
  ∀ (S : ℕ → α),
  arithmetic_progression S →
  (S 4) / (S 8) = 1 / 7 →
  (S 12) / (S 4) = 43 :=
by
  intros S h_arith_prog h_ratio
  sorry

end arithmetic_progression_sum_l271_271543


namespace solve_for_a_l271_271711

theorem solve_for_a (x a : ℝ) (h : x = 3) (eqn : 2 * (x - 1) - a = 0) : a = 4 := 
by 
  sorry

end solve_for_a_l271_271711


namespace parallel_lines_condition_l271_271710

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y - 4 = 0 → x + (a + 1) * y + 2 = 0) ↔ a = 1 :=
by sorry

end parallel_lines_condition_l271_271710


namespace smallest_solution_abs_eq_20_l271_271982

theorem smallest_solution_abs_eq_20 : ∃ x : ℝ, x = -7 ∧ |4 * x + 8| = 20 ∧ (∀ y : ℝ, |4 * y + 8| = 20 → x ≤ y) :=
by
  sorry

end smallest_solution_abs_eq_20_l271_271982


namespace carly_lollipops_total_l271_271979

theorem carly_lollipops_total (C : ℕ) (h1 : C / 2 = cherry_lollipops)
  (h2 : C / 2 = 3 * 7) : C = 42 :=
by
  sorry

end carly_lollipops_total_l271_271979


namespace necessary_but_not_sufficient_condition_l271_271831

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((x + 2) * (x - 3) < 0 → |x - 1| < 2) ∧ (¬(|x - 1| < 2 → (x + 2) * (x - 3) < 0)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l271_271831


namespace remainder_when_divided_by_x_minus_1_is_minus_2_l271_271524

def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

theorem remainder_when_divided_by_x_minus_1_is_minus_2 : (p 1) = -2 := 
by 
  -- Proof not required
  sorry

end remainder_when_divided_by_x_minus_1_is_minus_2_l271_271524


namespace sample_stats_equal_l271_271188

/-- Let x be a data set of n samples and y be another data set of n samples such that 
    ∀ i, y_i = x_i + c where c is a non-zero constant.
    Prove that the sample standard deviations and the ranges of x and y are the same. -/
theorem sample_stats_equal (n : ℕ) (x y : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)
    (h_y : ∀ i : Fin n, y i = x i + c) :
    (stddev x = stddev y) ∧ (range x = range y) := 
sorry

end sample_stats_equal_l271_271188


namespace probability_at_least_four_same_face_l271_271038

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l271_271038


namespace sum_of_positive_factors_of_90_l271_271772

theorem sum_of_positive_factors_of_90 : 
  let n := 90 in 
  let factors := (1 + 2) * (1 + 3 + 9) * (1 + 5) in 
  factors = 234 :=
by
  sorry

end sum_of_positive_factors_of_90_l271_271772


namespace inequality_holds_for_all_l271_271697

theorem inequality_holds_for_all (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) :
  (∀ α β : ℝ, ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) → m = n :=
by sorry

end inequality_holds_for_all_l271_271697


namespace hyperbola_foci_problem_l271_271583

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

noncomputable def foci_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def foci_2 : ℝ × ℝ := (Real.sqrt 5, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def vector (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v1.1 + v2.2 * v2.2

noncomputable def orthogonal (P : ℝ × ℝ) : Prop :=
  dot_product (vector P foci_1) (vector P foci_2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def required_value (P : ℝ × ℝ) : ℝ :=
  distance P foci_1 * distance P foci_2

theorem hyperbola_foci_problem (P : ℝ × ℝ) : 
  point_on_hyperbola P → orthogonal P → required_value P = 2 := 
sorry

end hyperbola_foci_problem_l271_271583


namespace balls_into_boxes_l271_271859

-- Define the problem conditions and expected outcome.
theorem balls_into_boxes : 
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → (∃ (ways : ℕ), ways = n) := 
begin
  use 7,
  split,
  { refl, },
  { intros balls boxes hballs hboxes,
    use 7,
    sorry
  }
end

end balls_into_boxes_l271_271859


namespace average_temperature_l271_271666

theorem average_temperature (T : Fin 5 → ℝ) (h : T = ![52, 67, 55, 59, 48]) :
    (1 / 5) * (T 0 + T 1 + T 2 + T 3 + T 4) = 56.2 := by
  sorry

end average_temperature_l271_271666


namespace merchant_should_choose_option2_l271_271671

-- Definitions for the initial price and discounts
def P : ℝ := 20000
def d1_1 : ℝ := 0.25
def d1_2 : ℝ := 0.15
def d1_3 : ℝ := 0.05
def d2_1 : ℝ := 0.35
def d2_2 : ℝ := 0.10
def d2_3 : ℝ := 0.05

-- Define the final prices after applying discount options
def finalPrice1 (P : ℝ) (d1_1 d1_2 d1_3 : ℝ) : ℝ :=
  P * (1 - d1_1) * (1 - d1_2) * (1 - d1_3)

def finalPrice2 (P : ℝ) (d2_1 d2_2 d2_3 : ℝ) : ℝ :=
  P * (1 - d2_1) * (1 - d2_2) * (1 - d2_3)

-- Theorem to state the merchant should choose Option 2
theorem merchant_should_choose_option2 : 
  finalPrice1 P d1_1 d1_2 d1_3 = 12112.50 ∧ 
  finalPrice2 P d2_1 d2_2 d2_3 = 11115 ∧ 
  finalPrice1 P d1_1 d1_2 d1_3 - finalPrice2 P d2_1 d2_2 d2_3 = 997.50 :=
by
  -- Placeholder for the proof
  sorry

end merchant_should_choose_option2_l271_271671


namespace probability_three_heads_in_eight_tosses_l271_271333

theorem probability_three_heads_in_eight_tosses :
  let total_outcomes := 2^8,
      favorable_outcomes := Nat.choose 8 3,
      probability := favorable_outcomes / total_outcomes
  in probability = (7 : ℚ) / 32 :=
by
  sorry

end probability_three_heads_in_eight_tosses_l271_271333


namespace Michael_points_l271_271566

theorem Michael_points (total_points : ℕ) (num_other_players : ℕ) (avg_points : ℕ) (Michael_points : ℕ) 
  (h1 : total_points = 75)
  (h2 : num_other_players = 5)
  (h3 : avg_points = 6)
  (h4 : Michael_points = total_points - num_other_players * avg_points) :
  Michael_points = 45 := by
  sorry

end Michael_points_l271_271566


namespace minimum_value_expr_l271_271818

noncomputable def expr (x : ℝ) : ℝ := (x^2 + 11) / Real.sqrt (x^2 + 5)

theorem minimum_value_expr : ∃ x : ℝ, expr x = 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_expr_l271_271818


namespace green_fish_count_l271_271095

theorem green_fish_count (B O G : ℕ) (H1 : B = 40) (H2 : O = B - 15) (H3 : 80 = B + O + G) : G = 15 := 
by 
  sorry

end green_fish_count_l271_271095


namespace permits_stamped_l271_271971

def appointments : ℕ := 2
def hours_per_appointment : ℕ := 3
def workday_hours : ℕ := 8
def stamps_per_hour : ℕ := 50

theorem permits_stamped :
  let total_appointment_hours := appointments * hours_per_appointment in
  let stamping_hours := workday_hours - total_appointment_hours in
  let total_permits := stamping_hours * stamps_per_hour in
  total_permits = 100 :=
by
  sorry

end permits_stamped_l271_271971


namespace min_adj_white_pairs_l271_271539

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l271_271539


namespace cake_and_milk_tea_cost_l271_271121

noncomputable def slice_cost (milk_tea_cost : ℚ) : ℚ := (3 / 4) * milk_tea_cost

noncomputable def total_cost (milk_tea_cost : ℚ) (slice_cost : ℚ) : ℚ :=
  2 * slice_cost + milk_tea_cost

theorem cake_and_milk_tea_cost 
  (milk_tea_cost : ℚ)
  (h : milk_tea_cost = 2.40) :
  total_cost milk_tea_cost (slice_cost milk_tea_cost) = 6.00 :=
by
  sorry

end cake_and_milk_tea_cost_l271_271121


namespace solve_for_ab_l271_271464

def f (a b : ℚ) (x : ℚ) : ℚ := a * x^3 - 4 * x^2 + b * x - 3

theorem solve_for_ab : 
  ∃ a b : ℚ, 
    f a b 1 = 3 ∧ 
    f a b (-2) = -47 ∧ 
    (a, b) = (4 / 3, 26 / 3) := 
by
  sorry

end solve_for_ab_l271_271464


namespace projection_matrix_exists_l271_271988

noncomputable def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, (20 : ℚ) / 49], ![c, (29 : ℚ) / 49]]

theorem projection_matrix_exists :
  ∃ (a c : ℚ), P a c * P a c = P a c ∧ a = (20 : ℚ) / 49 ∧ c = (29 : ℚ) / 49 := 
by
  use ((20 : ℚ) / 49), ((29 : ℚ) / 49)
  simp [P]
  sorry

end projection_matrix_exists_l271_271988


namespace alice_additional_cookies_proof_l271_271492

variable (alice_initial_cookies : ℕ)
variable (bob_initial_cookies : ℕ)
variable (cookies_thrown_away : ℕ)
variable (bob_additional_cookies : ℕ)
variable (total_edible_cookies : ℕ)

theorem alice_additional_cookies_proof 
    (h1 : alice_initial_cookies = 74)
    (h2 : bob_initial_cookies = 7)
    (h3 : cookies_thrown_away = 29)
    (h4 : bob_additional_cookies = 36)
    (h5 : total_edible_cookies = 93) :
  alice_initial_cookies + bob_initial_cookies - cookies_thrown_away + bob_additional_cookies + (93 - (74 + 7 - 29 + 36)) = total_edible_cookies :=
by
  sorry

end alice_additional_cookies_proof_l271_271492


namespace intersection_points_l271_271422

theorem intersection_points (l1 l2 : Type) [fintype l1] [fintype l2] (h_l1_points : ∃ s : set l1, s.card = 5) (h_l2_points : ∃ t : set l2, t.card = 10) (parallel : parallel_lines l1 l2) (no_three_intersections : no_three_segments_intersect l1 l2) :
  ∃ i : ℕ, i = 450 :=
by {
  sorry
}

end intersection_points_l271_271422


namespace partition_ways_six_three_boxes_l271_271863

theorem partition_ways_six_three_boxes :
  ∃ (P : Finset (Multiset ℕ)), P.card = 6 ∧ ∀ m ∈ P, ∃ l, m = {a : ℕ | ∃ i j k, a = (i, j, k) ∧ i+j+k = 6 ∧ i≥0 ∧ j≥0 ∧ k≥0}.count {
   {6, 0, 0},
   {5, 1, 0},
   {4, 2, 0},
   {4, 1, 1},
   {3, 2, 1},
   {2, 2, 2}
} :=
by
  sorry

end partition_ways_six_three_boxes_l271_271863


namespace peter_wins_prize_at_least_one_person_wins_prize_l271_271235

-- Part (a): Probability that Peter wins a prize
theorem peter_wins_prize :
  let p : Probability := (5 / 6) ^ 9
  p = 0.194 := sorry

-- Part (b): Probability that at least one person wins a prize
theorem at_least_one_person_wins_prize :
  let p : Probability := 0.919
  p = 0.919 := sorry

end peter_wins_prize_at_least_one_person_wins_prize_l271_271235


namespace simplify_expr_1_simplify_expr_2_l271_271439

-- The first problem
theorem simplify_expr_1 (a : ℝ) : 2 * a^2 - 3 * a - 5 * a^2 + 6 * a = -3 * a^2 + 3 * a := 
by
  sorry

-- The second problem
theorem simplify_expr_2 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l271_271439


namespace ariana_total_owe_l271_271973

-- Definitions based on the conditions
def first_bill_principal : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_overdue_months : ℕ := 2

def second_bill_principal : ℕ := 130
def second_bill_late_fee : ℕ := 50
def second_bill_overdue_months : ℕ := 6

def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80

-- Theorem
theorem ariana_total_owe : 
  first_bill_principal + 
    (first_bill_principal : ℝ) * first_bill_interest_rate * (first_bill_overdue_months : ℝ) +
    second_bill_principal + 
    second_bill_late_fee * second_bill_overdue_months + 
    third_bill_first_month_fee + 
    third_bill_second_month_fee = 790 := 
by 
  sorry

end ariana_total_owe_l271_271973


namespace number_with_at_least_two_zeros_l271_271848

-- A 6-digit number can have for its leftmost digit anything from 1 to 9 inclusive,
-- and for each of its next five digits anything from 0 through 9 inclusive.
def total_6_digit_numbers : ℕ := 9 * 10^5

-- A 6-digit number with no zeros consists solely of digits from 1 to 9
def no_zero : ℕ := 9^6

-- A 6-digit number with exactly one zero
def exactly_one_zero : ℕ := 5 * 9^5

-- The number of 6-digit numbers with less than two zeros is the sum of no_zero and exactly_one_zero
def less_than_two_zeros : ℕ := no_zero + exactly_one_zero

-- The number of 6-digit numbers with at least two zeros is the difference between total_6_digit_numbers and less_than_two_zeros
def at_least_two_zeros : ℕ := total_6_digit_numbers - less_than_two_zeros

-- The theorem that states the number of 6-digit numbers with at least two zeros is 73,314
theorem number_with_at_least_two_zeros : at_least_two_zeros = 73314 := 
by
  sorry

end number_with_at_least_two_zeros_l271_271848


namespace abcde_sum_l271_271223

theorem abcde_sum : 
  ∀ (a b c d e : ℝ), 
  a + 1 = b + 2 → 
  b + 2 = c + 3 → 
  c + 3 = d + 4 → 
  d + 4 = e + 5 → 
  e + 5 = a + b + c + d + e + 10 → 
  a + b + c + d + e = -35 / 4 :=
sorry

end abcde_sum_l271_271223


namespace like_terms_l271_271205

theorem like_terms (x y : ℕ) (h1 : x + 1 = 2) (h2 : x + y = 2) : x = 1 ∧ y = 1 :=
by
  sorry

end like_terms_l271_271205


namespace pentagon_angle_T_l271_271571

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end pentagon_angle_T_l271_271571


namespace solve_for_y_l271_271108

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l271_271108


namespace smallest_prime_with_digit_sum_22_l271_271655

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem smallest_prime_with_digit_sum_22 :
  (∃ n : ℕ, is_prime n ∧ digit_sum n = 22 ∧ ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22) → n ≤ m) ∧
  ∀ m : ℕ, (is_prime m ∧ digit_sum m = 22 ∧ m < 499) → false := 
sorry

end smallest_prime_with_digit_sum_22_l271_271655


namespace Q_neither_necessary_nor_sufficient_l271_271094

-- Define the propositions P and Q
def PropositionP (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ x : ℝ, (a1*x^2 + b1*x + c1 > 0) ↔ (a2*x^2 + b2*x + c2 > 0)

def PropositionQ (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2) ∧ (b1 / b2 = c1 / c2)

-- The final statement to prove that Q is neither necessary nor sufficient for P
theorem Q_neither_necessary_nor_sufficient (a1 b1 c1 a2 b2 c2 : ℝ) :
  ¬ ((PropositionQ a1 b1 c1 a2 b2 c2) ↔ (PropositionP a1 b1 c1 a2 b2 c2)) := sorry

end Q_neither_necessary_nor_sufficient_l271_271094


namespace example_problem_l271_271994

def diamond (a b : ℕ) : ℕ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem example_problem : diamond 3 2 = 125 := by
  sorry

end example_problem_l271_271994


namespace find_digit_B_l271_271917

theorem find_digit_B (A B : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (h' : 0 ≤ B ∧ B ≤ 9) (eqn : 10 * A + 22 = 9 * B) : B = 8 := 
  sorry

end find_digit_B_l271_271917


namespace find_a6_of_arithmetic_seq_l271_271135

noncomputable def arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_a6_of_arithmetic_seq 
  (a1 d : ℝ) 
  (S3 : ℝ) 
  (h_a1 : a1 = 2) 
  (h_S3 : S3 = 12) 
  (h_sum : S3 = sum_of_arithmetic_sequence 3 a1 d) :
  arithmetic_sequence 6 a1 d = 12 := 
sorry

end find_a6_of_arithmetic_seq_l271_271135


namespace jesses_room_total_area_l271_271218

-- Define the dimensions of the first rectangular part
def length1 : ℕ := 12
def width1 : ℕ := 8

-- Define the dimensions of the second rectangular part
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Define the areas of both parts
def area1 : ℕ := length1 * width1
def area2 : ℕ := length2 * width2

-- Define the total area
def total_area : ℕ := area1 + area2

-- Statement of the theorem we want to prove
theorem jesses_room_total_area : total_area = 120 :=
by
  -- We would provide the proof here
  sorry

end jesses_room_total_area_l271_271218


namespace cylinder_radius_and_remaining_space_l271_271959

theorem cylinder_radius_and_remaining_space 
  (cone_radius : ℝ) (cone_height : ℝ) 
  (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cone_radius = 8 →
  cone_height = 20 →
  cylinder_height = 2 * cylinder_radius →
  (20 - 2 * cylinder_radius) / cylinder_radius = 20 / 8 →
  (cylinder_radius = 40 / 9 ∧ (cone_height - cylinder_height) = 100 / 9) :=
by
  intros cone_radius_8 cone_height_20 cylinder_height_def similarity_eq
  sorry

end cylinder_radius_and_remaining_space_l271_271959


namespace product_of_p_and_q_l271_271876

theorem product_of_p_and_q (p q : ℝ) (hpq_sum : p + q = 10) (hpq_cube_sum : p^3 + q^3 = 370) : p * q = 21 :=
by
  sorry

end product_of_p_and_q_l271_271876


namespace quadrilateral_area_l271_271128

theorem quadrilateral_area (a b c d e f : ℝ) : 
    (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 :=
    by sorry

noncomputable def quadrilateral_area_formula (a b c d e f : ℝ) : ℝ :=
    if H : (a^2 + c^2 - b^2 - d^2) ^ 2 ≤ 4 * e^2 * f^2 then 
    (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2)
    else 0

-- Ensure that the computed area matches the expected value
example (a b c d e f : ℝ) (H : (a^2 + c^2 - b^2 - d^2)^2 ≤ 4 * e^2 * f^2) : 
    quadrilateral_area_formula a b c d e f = 
        (1/4) * Real.sqrt (4 * e^2 * f^2 - (a^2 + c^2 - b^2 - d^2) ^ 2) :=
by simp [quadrilateral_area_formula, H]

end quadrilateral_area_l271_271128


namespace gcd_lcm_product_24_36_proof_l271_271360

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l271_271360


namespace lowest_fraction_combine_two_slowest_l271_271172

def rate_a (hours : ℕ) : ℚ := 1 / 4
def rate_b (hours : ℕ) : ℚ := 1 / 5
def rate_c (hours : ℕ) : ℚ := 1 / 8

theorem lowest_fraction_combine_two_slowest : 
  (rate_b 1 + rate_c 1) = 13 / 40 :=
by sorry

end lowest_fraction_combine_two_slowest_l271_271172


namespace probability_of_exactly_three_heads_l271_271290

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l271_271290


namespace coin_toss_probability_l271_271329

theorem coin_toss_probability :
  (∃ (p : ℚ), p = (nat.choose 8 3 : ℚ) / 2^8 ∧ p = 7 / 32) :=
by
  sorry

end coin_toss_probability_l271_271329


namespace sum_of_remainders_l271_271465

theorem sum_of_remainders (a b c d : ℕ)
  (ha : a % 17 = 3) (hb : b % 17 = 5) (hc : c % 17 = 7) (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 :=
by
  sorry

end sum_of_remainders_l271_271465


namespace quadratic_roots_l271_271763

theorem quadratic_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + bx + c = 0 ↔ x^2 - 5 * x + 2 = 0):
  c / b = -4 / 21 :=
  sorry

end quadratic_roots_l271_271763


namespace triangle_inequality_valid_x_values_l271_271452

theorem triangle_inequality_valid_x_values :
  ∃ n : ℕ, n = 29 ∧ ∀ x : ℕ, (25 < x ∧ x < 55) ↔ x ∈ finset.range 29 ∧ (x + 26 < 55 ∧ 25 < x) :=
by
  sorry

end triangle_inequality_valid_x_values_l271_271452


namespace vertical_line_divides_triangle_equal_area_l271_271353

theorem vertical_line_divides_triangle_equal_area :
  let A : (ℝ × ℝ) := (1, 2)
  let B : (ℝ × ℝ) := (1, 1)
  let C : (ℝ × ℝ) := (10, 1)
  let area_ABC := (1 / 2 : ℝ) * (C.1 - A.1) * (A.2 - B.2)
  let a : ℝ := 5.5
  let area_left_triangle := (1 / 2 : ℝ) * (a - A.1) * (A.2 - B.2)
  let area_right_triangle := (1 / 2 : ℝ) * (C.1 - a) * (A.2 - B.2)
  area_left_triangle = area_right_triangle :=
by
  sorry

end vertical_line_divides_triangle_equal_area_l271_271353


namespace coin_toss_probability_l271_271324

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l271_271324


namespace area_of_triangle_DEF_l271_271958

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 8)
  let base1 := 6
  let height1 := 2
  let base2 := 3
  let height2 := 8
  let base3 := 3
  let height3 := 6
  let area_triangle_DE := 1 / 2 * (base1 * height1)
  let area_triangle_EF := 1 / 2 * (base2 * height2)
  let area_triangle_FD := 1 / 2 * (base3 * height3)
  let area_rectangle := 6 * 8
  ∃ area_def_triangle, 
  area_def_triangle = area_rectangle - (area_triangle_DE + area_triangle_EF + area_triangle_FD) 
  ∧ area_def_triangle = 21 :=
by 
  sorry

end area_of_triangle_DEF_l271_271958


namespace product_of_000412_and_9243817_is_closest_to_3600_l271_271656

def product_closest_to (x y value: ℝ) : Prop := (abs (x * y - value) < min (abs (x * y - 350)) (min (abs (x * y - 370)) (min (abs (x * y - 3700)) (abs (x * y - 4000)))))

theorem product_of_000412_and_9243817_is_closest_to_3600 :
  product_closest_to 0.000412 9243817 3600 :=
by
  sorry

end product_of_000412_and_9243817_is_closest_to_3600_l271_271656


namespace siamese_cats_initial_l271_271790

theorem siamese_cats_initial (S : ℕ) : S + 25 - 45 = 18 -> S = 38 :=
by
  intro h
  sorry

end siamese_cats_initial_l271_271790


namespace complete_square_result_l271_271736

theorem complete_square_result (x : ℝ) :
  (∃ r s : ℝ, (16 * x ^ 2 + 32 * x - 1280 = 0) → ((x + r) ^ 2 = s) ∧ s = 81) :=
by
  sorry

end complete_square_result_l271_271736


namespace same_number_on_four_dice_l271_271649

theorem same_number_on_four_dice : 
  let p : ℕ := 6
  in (1 : ℝ) * (1 / p) * (1 / p) * (1 / p) = 1 / (p * p * p) := by
  sorry

end same_number_on_four_dice_l271_271649


namespace probability_exactly_three_heads_l271_271310
open Nat

theorem probability_exactly_three_heads (prob : ℚ) :
  let total_sequences : ℚ := (2^8)
  let favorable_sequences : ℚ := (Nat.choose 8 3)
  let probability : ℚ := (favorable_sequences / total_sequences)
  prob = probability := by
  have ht : total_sequences = 256 := by sorry
  have hf : favorable_sequences = 56 := by sorry
  have hp : probability = (56 / 256) := by sorry
  have hs : ((56 / 256) = (7 / 32)) := by sorry
  show prob = (7 / 32)
  sorry

end probability_exactly_three_heads_l271_271310


namespace compound_interest_l271_271005

variables {a r : ℝ}

theorem compound_interest (a r : ℝ) :
  (a * (1 + r)^10) = a * (1 + r)^(2020 - 2010) :=
by
  sorry

end compound_interest_l271_271005


namespace log_product_computation_l271_271512

theorem log_product_computation : 
  (Real.log 32 / Real.log 2) * (Real.log 27 / Real.log 3) = 15 := 
by
  -- The proof content, which will be skipped with 'sorry'.
  sorry

end log_product_computation_l271_271512


namespace initial_bottles_calculation_l271_271423

theorem initial_bottles_calculation (maria_bottles : ℝ) (sister_bottles : ℝ) (left_bottles : ℝ) 
  (H₁ : maria_bottles = 14.0) (H₂ : sister_bottles = 8.0) (H₃ : left_bottles = 23.0) :
  maria_bottles + sister_bottles + left_bottles = 45.0 :=
by
  sorry

end initial_bottles_calculation_l271_271423


namespace line_intersects_circle_l271_271099

variable {a x_0 y_0 : ℝ}

theorem line_intersects_circle (h1: x_0^2 + y_0^2 > a^2) (h2: a > 0) : 
  ∃ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 = a ^ 2) ∧ (x_0 * p.1 + y_0 * p.2 = a ^ 2) :=
sorry

end line_intersects_circle_l271_271099


namespace manicure_cost_l271_271220

noncomputable def cost_of_manicure : ℝ := 30

theorem manicure_cost
    (cost_hair_updo : ℝ)
    (total_cost_with_tips : ℝ)
    (tip_rate : ℝ)
    (M : ℝ) :
  cost_hair_updo = 50 →
  total_cost_with_tips = 96 →
  tip_rate = 0.20 →
  (cost_hair_updo + M + tip_rate * cost_hair_updo + tip_rate * M = total_cost_with_tips) →
  M = cost_of_manicure :=
by
  intros h1 h2 h3 h4
  sorry

end manicure_cost_l271_271220


namespace α_plus_2β_eq_pi_div_2_l271_271197

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : 3 * sin α ^ 2 + 2 * sin β ^ 2 = 1
axiom h4 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0

theorem α_plus_2β_eq_pi_div_2 : α + 2 * β = π / 2 :=
by
  sorry

end α_plus_2β_eq_pi_div_2_l271_271197


namespace infinite_series_problem_l271_271980

noncomputable def infinite_series_sum : ℝ := ∑' n : ℕ, (2 * (n + 1)^2 - 3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))

theorem infinite_series_problem :
  infinite_series_sum = -4 :=
by sorry

end infinite_series_problem_l271_271980


namespace num_aluminum_cans_l271_271115

def num_glass_bottles : ℕ := 10
def total_litter : ℕ := 18

theorem num_aluminum_cans : total_litter - num_glass_bottles = 8 :=
by
  sorry

end num_aluminum_cans_l271_271115


namespace ice_cream_stall_difference_l271_271592

theorem ice_cream_stall_difference (d : ℕ) 
  (h1 : ∃ d, 10 + (10 + d) + (10 + 2*d) + (10 + 3*d) + (10 + 4*d) = 90) : 
  d = 4 :=
by
  sorry

end ice_cream_stall_difference_l271_271592


namespace value_of_certain_number_l271_271567

theorem value_of_certain_number (a b : ℕ) (h : 1 / 7 * 8 = 5) (h2 : 1 / 5 * b = 35) : b = 175 :=
by
  -- by assuming the conditions hold, we need to prove b = 175
  sorry

end value_of_certain_number_l271_271567


namespace quadratic_roots_l271_271036

theorem quadratic_roots (m n p : ℕ) (h : m.gcd p = 1) 
  (h1 : 3 * m^2 - 8 * m * p + p^2 = p^2 * n) : n = 13 :=
by sorry

end quadratic_roots_l271_271036


namespace find_a4_plus_b4_l271_271776

-- Variables representing the given conditions
variables {a b : ℝ}

-- The theorem statement to prove
theorem find_a4_plus_b4 (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 56 :=
sorry

end find_a4_plus_b4_l271_271776


namespace rectangle_length_l271_271149

theorem rectangle_length (P B L : ℝ) (h1 : P = 600) (h2 : B = 200) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l271_271149


namespace divisible_by_11_l271_271724

theorem divisible_by_11 (k : ℕ) (h : 0 ≤ k ∧ k ≤ 9) :
  (9 + 4 + 5 + k + 3 + 1 + 7) - 2 * (4 + k + 1) ≡ 0 [MOD 11] → k = 8 :=
by
  sorry

end divisible_by_11_l271_271724


namespace total_yen_l271_271014

-- Define the given conditions in Lean 4
def bal_bahamian_dollars : ℕ := 5000
def bal_us_dollars : ℕ := 2000
def bal_euros : ℕ := 3000

def exchange_rate_bahamian_to_yen : ℝ := 122.13
def exchange_rate_us_to_yen : ℝ := 110.25
def exchange_rate_euro_to_yen : ℝ := 128.50

def check_acc1 : ℕ := 15000
def check_acc2 : ℕ := 6359
def sav_acc1 : ℕ := 5500
def sav_acc2 : ℕ := 3102

def stocks : ℕ := 200000
def bonds : ℕ := 150000
def mutual_funds : ℕ := 120000

-- Prove the total amount of yen the family has
theorem total_yen : 
  bal_bahamian_dollars * exchange_rate_bahamian_to_yen + 
  bal_us_dollars * exchange_rate_us_to_yen + 
  bal_euros * exchange_rate_euro_to_yen
  + (check_acc1 + check_acc2 + sav_acc1 + sav_acc2 : ℝ)
  + (stocks + bonds + mutual_funds : ℝ) = 1716611 := 
by
  sorry

end total_yen_l271_271014


namespace area_not_covered_correct_l271_271672

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8

-- Define the side length of the square
def square_side_length : ℕ := 5

-- The area of the rectangle
def rectangle_area : ℕ := rectangle_length * rectangle_width

-- The area of the square
def square_area : ℕ := square_side_length * square_side_length

-- The area of the region not covered by the square
def area_not_covered : ℕ := rectangle_area - square_area

-- The theorem statement asserting the required area
theorem area_not_covered_correct : area_not_covered = 55 :=
by
  -- Proof is omitted
  sorry

end area_not_covered_correct_l271_271672


namespace dice_probability_same_face_l271_271631

def roll_probability (dice: ℕ) (faces: ℕ) : ℚ :=
  1 / faces ^ (dice - 1)

theorem dice_probability_same_face :
  roll_probability 4 6 = 1 / 216 := 
by
  sorry

end dice_probability_same_face_l271_271631


namespace eval_expression_l271_271885

-- Define the expression to evaluate
def expression : ℚ := 2 * 3 + 4 - (5 / 6)

-- Prove the equivalence of the evaluated expression to the expected result
theorem eval_expression : expression = 37 / 3 :=
by
  -- The detailed proof steps are omitted (relying on sorry)
  sorry

end eval_expression_l271_271885


namespace complex_num_sum_l271_271532

def is_complex_num (a b : ℝ) (z : ℂ) : Prop :=
  z = a + b * Complex.I

theorem complex_num_sum (a b : ℝ) (z : ℂ) (h : is_complex_num a b z) :
  z = (1 - Complex.I) ^ 2 / (1 + Complex.I) → a + b = -2 :=
by
  sorry

end complex_num_sum_l271_271532


namespace circle_range_k_l271_271605

theorem circle_range_k (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 4 * x + 4 * y + 10 - k = 0) → k > 2 :=
by
  sorry

end circle_range_k_l271_271605


namespace no_intersection_abs_eq_l271_271562

theorem no_intersection_abs_eq (x : ℝ) : ∀ y : ℝ, y = |3 * x + 6| → y = -|2 * x - 4| → false := 
by
  sorry

end no_intersection_abs_eq_l271_271562


namespace amount_borrowed_from_bank_l271_271735

-- Definitions of the conditions
def car_price : ℝ := 35000
def total_payment : ℝ := 38000
def interest_rate : ℝ := 0.15

theorem amount_borrowed_from_bank :
  total_payment - car_price = interest_rate * (total_payment - car_price) / interest_rate := sorry

end amount_borrowed_from_bank_l271_271735


namespace time_for_10_strikes_l271_271668

-- Assume a clock takes 7 seconds to strike 7 times
def clock_time_for_N_strikes (N : ℕ) : ℕ :=
  if N = 7 then 7 else sorry  -- This would usually be a function, simplified here for the specific condition

-- Assume there are 6 intervals for 7 strikes
def intervals_between_strikes (N : ℕ) : ℕ :=
  if N = 7 then 6 else N - 1

-- Function to calculate total time for any number of strikes based on intervals and time per strike
def total_time_for_strikes (N : ℕ) : ℚ :=
  (intervals_between_strikes N) * (clock_time_for_N_strikes 7 / intervals_between_strikes 7 : ℚ)

theorem time_for_10_strikes : total_time_for_strikes 10 = 10.5 :=
by
  -- Insert proof here
  sorry

end time_for_10_strikes_l271_271668


namespace max_choir_members_l271_271950

theorem max_choir_members (n : ℕ) (x y : ℕ) : 
  n = x^2 + 11 ∧ n = y * (y + 3) → n = 54 :=
by
  sorry

end max_choir_members_l271_271950


namespace find_first_set_length_l271_271581

def length_of_second_set : ℤ := 20
def ratio := 5

theorem find_first_set_length (x : ℤ) (h1 : length_of_second_set = ratio * x) : x = 4 := 
sorry

end find_first_set_length_l271_271581


namespace inequality_empty_solution_range_l271_271615

/-
Proof problem:
Prove that for the inequality |x-3| + |x-a| < 1 to have no solutions, the range of a must be (-∞, 2] ∪ [4, +∞).
-/

theorem inequality_empty_solution_range (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - a| < 1 → false) ↔ a ∈ set.Iic 2 ∪ set.Ici 4 := sorry

end inequality_empty_solution_range_l271_271615


namespace orchestra_members_l271_271903

theorem orchestra_members :
  ∃ (n : ℕ), 
    150 < n ∧ n < 250 ∧ 
    n % 4 = 2 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 4 :=
by
  use 158
  repeat {split};
  sorry

end orchestra_members_l271_271903


namespace vertical_asymptotes_polynomial_l271_271400

theorem vertical_asymptotes_polynomial (a b : ℝ) (h₁ : -3 * 2 = b) (h₂ : -3 + 2 = a) : a + b = -5 := by
  sorry

end vertical_asymptotes_polynomial_l271_271400


namespace fenced_area_correct_l271_271125

-- Define the dimensions of the rectangle
def length := 20
def width := 18

-- Define the dimensions of the cutouts
def square_cutout1 := 4
def square_cutout2 := 2

-- Define the areas of the rectangle and the cutouts
def area_rectangle := length * width
def area_cutout1 := square_cutout1 * square_cutout1
def area_cutout2 := square_cutout2 * square_cutout2

-- Define the total area within the fence
def total_area_within_fence := area_rectangle - area_cutout1 - area_cutout2

-- The theorem that needs to be proven
theorem fenced_area_correct : total_area_within_fence = 340 := by
  sorry

end fenced_area_correct_l271_271125


namespace function_is_monotonically_decreasing_l271_271065

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem function_is_monotonically_decreasing :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → deriv f x ≤ 0 :=
by
  sorry

end function_is_monotonically_decreasing_l271_271065


namespace rearrangements_of_abcde_l271_271563

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 == 'a' ∧ c2 == 'b') ∨ 
  (c1 == 'b' ∧ c1 == 'a') ∨ 
  (c1 == 'b' ∧ c2 == 'c') ∨ 
  (c1 == 'c' ∧ c2 == 'b') ∨ 
  (c1 == 'c' ∧ c2 == 'd') ∨ 
  (c1 == 'd' ∧ c2 == 'c') ∨ 
  (c1 == 'd' ∧ c2 == 'e') ∨ 
  (c1 == 'e' ∧ c2 == 'd')

def is_valid_rearrangement (lst : List Char) : Bool :=
  match lst with
  | [] => true
  | [_] => true
  | c1 :: c2 :: rest => 
    ¬is_adjacent c1 c2 ∧ is_valid_rearrangement (c2 :: rest)

def count_valid_rearrangements (chars : List Char) : Nat :=
  chars.permutations.filter is_valid_rearrangement |>.length

theorem rearrangements_of_abcde : count_valid_rearrangements ['a', 'b', 'c', 'd', 'e'] = 8 := 
by
  sorry

end rearrangements_of_abcde_l271_271563


namespace integer_values_x_possible_l271_271451

-- Define the triangle and its side lengths
def non_degenerate_triangle (x a b : ℕ) : Prop :=
  x + a > b ∧ x + b > a ∧ a + b > x

-- Define the main theorem to prove
theorem integer_values_x_possible : 
  (n : ℕ) × ((25 < x ∧ x < 55) ∧ ∀ x : ℕ, non_degenerate_triangle x 15 40)
  ∧ ((54 - 25 + 1) = 30) := 
sorry

end integer_values_x_possible_l271_271451


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l271_271146

theorem option_A_incorrect : ¬(Real.sqrt 2 + Real.sqrt 6 = Real.sqrt 8) :=
by sorry

theorem option_B_incorrect : ¬(6 * Real.sqrt 3 - 2 * Real.sqrt 3 = 4) :=
by sorry

theorem option_C_incorrect : ¬(4 * Real.sqrt 2 * 2 * Real.sqrt 3 = 6 * Real.sqrt 6) :=
by sorry

theorem option_D_correct : (1 / (2 - Real.sqrt 3) = 2 + Real.sqrt 3) :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l271_271146


namespace infimum_of_function_l271_271826

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 1)^2

def is_lower_bound (M : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≥ M

def is_infimum (M : ℝ) (f : ℝ → ℝ) : Prop :=
  is_lower_bound M f ∧ ∀ L : ℝ, is_lower_bound L f → L ≤ M

theorem infimum_of_function :
  is_infimum 0.5 f :=
sorry

end infimum_of_function_l271_271826


namespace rhombus_diagonals_not_equal_l271_271937

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l271_271937


namespace combined_total_l271_271262

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end combined_total_l271_271262


namespace min_expression_value_l271_271523

theorem min_expression_value (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m : ℝ, (∀ x y : ℝ, x > 2 → y > 2 → (x^3 / (y - 2) + y^3 / (x - 2)) ≥ m) ∧ 
          (m = 64) :=
by
  sorry

end min_expression_value_l271_271523


namespace gcd_lcm_product_24_36_proof_l271_271361

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l271_271361


namespace sequence_b_n_l271_271026

theorem sequence_b_n (b : ℕ → ℕ) (h₀ : b 1 = 3) (h₁ : ∀ n, b (n + 1) = b n + 3 * n + 1) :
  b 50 = 3727 :=
sorry

end sequence_b_n_l271_271026


namespace arvin_fifth_day_run_l271_271882

theorem arvin_fifth_day_run :
  let running_distance : ℕ → ℕ := λ day, 2 + day - 1
  in running_distance 5 = 6 := by
  sorry

end arvin_fifth_day_run_l271_271882


namespace arithmetic_seq_a5_value_l271_271573

theorem arithmetic_seq_a5_value (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 = 45) :
  a 5 = 9 := 
sorry

end arithmetic_seq_a5_value_l271_271573


namespace mrs_hilt_chapters_read_l271_271425

def number_of_books : ℝ := 4.0
def chapters_per_book : ℝ := 4.25
def total_chapters_read : ℝ := number_of_books * chapters_per_book

theorem mrs_hilt_chapters_read : total_chapters_read = 17 :=
by
  unfold total_chapters_read
  norm_num
  sorry

end mrs_hilt_chapters_read_l271_271425


namespace mean_of_solutions_l271_271522

theorem mean_of_solutions (x : ℝ) (h : x^3 + x^2 - 14 * x = 0) : 
  let a := (0 : ℝ)
  let b := (-1 + Real.sqrt 57) / 2
  let c := (-1 - Real.sqrt 57) / 2
  (a + b + c) / 3 = -2 / 3 :=
sorry

end mean_of_solutions_l271_271522


namespace steve_bought_2_cookies_boxes_l271_271601

theorem steve_bought_2_cookies_boxes :
  ∀ (milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
     milk_qty cereal_qty banana_qty apple_qty cookies_qty : ℕ),
    milk_price = 3 →
    cereal_price_per_box = 3.5 →
    banana_price = 0.25 →
    apple_price = 0.5 →
    cookies_cost_per_box = 2 * milk_price →
    milk_qty = 1 →
    cereal_qty = 2 →
    banana_qty = 4 →
    apple_qty = 4 →
    (milk_price * milk_qty + cereal_price_per_box * cereal_qty + 
     banana_price * banana_qty + apple_price * apple_qty + 
     cookies_cost_per_box * cookies_qty = 25) →
    cookies_qty = 2 :=
by
  intros milk_price cereal_price_per_box banana_price apple_price cookies_cost_per_box 
         milk_qty cereal_qty banana_qty apple_qty cookies_qty
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 hsum
  have h_milk : milk_price = 3 := h1
  have h_cereal : cereal_price_per_box = 3.5 := h2
  have h_banana : banana_price = 0.25 := h3
  have h_apple : apple_price = 0.5 := h4
  have h_cookies_price : cookies_cost_per_box = 2 * milk_price := h5
  have h_milk_qty : milk_qty = 1 := h6
  have h_cereal_qty : cereal_qty = 2 := h7
  have h_banana_qty : banana_qty = 4 := h8
  have h_apple_qty : apple_qty = 4 := h9
  rw [h_milk, h_cereal, h_banana, h_apple, h_cookies_price, h_milk_qty, h_cereal_qty, h_banana_qty, h_apple_qty] at hsum
  sorry

end steve_bought_2_cookies_boxes_l271_271601


namespace sum_of_coordinates_inv_graph_l271_271752

variable {f : ℝ → ℝ}
variable (hf : f 2 = 12)

theorem sum_of_coordinates_inv_graph :
  ∃ (x y : ℝ), y = f⁻¹ x / 3 ∧ x = 12 ∧ y = 2 / 3 ∧ x + y = 38 / 3 := by
  sorry

end sum_of_coordinates_inv_graph_l271_271752


namespace quadrant_angle_l271_271207

theorem quadrant_angle (θ : ℝ) (k : ℤ) (h_theta : 0 < θ ∧ θ < 90) : 
  ((180 * k + θ) % 360 < 90) ∨ (180 * k + θ) % 360 ≥ 180 ∧ (180 * k + θ) % 360 < 270 :=
sorry

end quadrant_angle_l271_271207


namespace puppies_count_l271_271267

theorem puppies_count 
  (dogs : ℕ := 3)
  (dog_meal_weight : ℕ := 4)
  (dog_meals_per_day : ℕ := 3)
  (total_food : ℕ := 108)
  (puppy_meal_multiplier : ℕ := 2)
  (puppy_meal_frequency_multiplier : ℕ := 3) :
  ∃ (puppies : ℕ), puppies = 4 :=
by
  let dog_daily_food := dog_meal_weight * dog_meals_per_day
  let puppy_meal_weight := dog_meal_weight / puppy_meal_multiplier
  let puppy_daily_food := puppy_meal_weight * puppy_meal_frequency_multiplier * dog_meals_per_day
  let total_dog_food := dogs * dog_daily_food
  let total_puppy_food := total_food - total_dog_food
  let puppies := total_puppy_food / puppy_daily_food
  use puppies
  have h_puppies_correct : puppies = 4 := sorry
  exact h_puppies_correct

end puppies_count_l271_271267


namespace prob_three_heads_in_eight_tosses_l271_271306

theorem prob_three_heads_in_eight_tosses : 
  let outcomes := (2:ℕ)^8,
      favorable := Nat.choose 8 3
  in (favorable / outcomes) = (7 / 32) := 
by 
  /- Insert proof here -/
  sorry

end prob_three_heads_in_eight_tosses_l271_271306


namespace arithmetic_sequence_sum_l271_271547

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S_9 : ℚ) 
  (h_arith : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a2_a8 : a 2 + a 8 = 4 / 3) :
  S_9 = 6 :=
by
  sorry

end arithmetic_sequence_sum_l271_271547


namespace min_value_of_reciprocal_squares_l271_271922

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end min_value_of_reciprocal_squares_l271_271922


namespace circle_equation_l271_271383

-- Define conditions
def on_parabola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 = 4 * y

def tangent_to_y_axis (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x, _) := M
  abs x = r

def tangent_to_axis_of_symmetry (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (_, y) := M
  abs (1 + y) = r

-- Main theorem statement
theorem circle_equation (M : ℝ × ℝ) (r : ℝ) (x y : ℝ)
  (h1 : on_parabola M)
  (h2 : tangent_to_y_axis M r)
  (h3 : tangent_to_axis_of_symmetry M r) :
  (x - M.1)^2 + (y - M.2)^2 = r^2 ↔
  x^2 + y^2 + 4 * M.1 * x - 2 * M.2 * y + 1 = 0 := 
sorry

end circle_equation_l271_271383


namespace expected_balls_in_original_position_after_two_transpositions_l271_271248

-- Define the conditions
def num_balls : ℕ := 10

def probs_ball_unchanged : ℚ :=
  (1 / 50) + (16 / 25)

def expected_unchanged_balls (num_balls : ℕ) (probs_ball_unchanged : ℚ) : ℚ :=
  num_balls * probs_ball_unchanged

-- The theorem stating the expected number of balls in original positions
theorem expected_balls_in_original_position_after_two_transpositions
  (num_balls_eq : num_balls = 10)
  (prob_eq : probs_ball_unchanged = (1 / 50) + (16 / 25)) :
  expected_unchanged_balls num_balls probs_ball_unchanged = 7.2 := 
by
  sorry

end expected_balls_in_original_position_after_two_transpositions_l271_271248


namespace count_arithmetic_sequence_l271_271814

theorem count_arithmetic_sequence :
  let a1 := 2.5
  let an := 68.5
  let d := 6.0
  let offset := 0.5
  let adjusted_a1 := a1 + offset
  let adjusted_an := an + offset
  let n := (adjusted_an - adjusted_a1) / d + 1
  n = 12 :=
by {
  sorry
}

end count_arithmetic_sequence_l271_271814


namespace value_of_C_l271_271162

theorem value_of_C (k : ℝ) (C : ℝ) (h : k = 0.4444444444444444) :
  (2 * k * 0 ^ 2 + 6 * k * 0 + C = 0) ↔ C = 2 :=
by {
  sorry
}

end value_of_C_l271_271162


namespace probability_of_same_number_on_four_dice_l271_271635

noncomputable theory

-- Define an event for the probability of rolling the same number on four dice
def probability_same_number (n : ℕ) (p : ℝ) : Prop :=
  n = 6 ∧ p = 1 / 216

-- Prove the above event given the conditions
theorem probability_of_same_number_on_four_dice :
  probability_same_number 6 (1 / 216) :=
by
  -- This is where the proof would be constructed
  sorry

end probability_of_same_number_on_four_dice_l271_271635


namespace add_fractions_l271_271501

theorem add_fractions (x : ℝ) (h : x ≠ 1) : (1 / (x - 1) + 3 / (x - 1)) = (4 / (x - 1)) :=
by
  sorry

end add_fractions_l271_271501


namespace original_couch_price_l271_271414

def chair_price : ℝ := sorry
def table_price := 3 * chair_price
def couch_price := 5 * table_price
def bookshelf_price := 0.5 * couch_price

def discounted_chair_price := 0.8 * chair_price
def discounted_couch_price := 0.9 * couch_price
def total_price_before_tax := discounted_chair_price + table_price + discounted_couch_price + bookshelf_price
def total_price_after_tax := total_price_before_tax * 1.08

theorem original_couch_price (budget : ℝ) (h_budget : budget = 900) : 
  total_price_after_tax = budget → couch_price = 503.85 :=
by
  sorry

end original_couch_price_l271_271414


namespace common_solutions_for_y_l271_271171

theorem common_solutions_for_y (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x^2 - 3 * y = 12) ↔ (y = -4 ∨ y = 1) :=
by
  sorry

end common_solutions_for_y_l271_271171


namespace part_I_part_II_l271_271066

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II_l271_271066


namespace amount_per_person_is_correct_l271_271469

-- Define the total amount and the number of people
def total_amount : ℕ := 2400
def number_of_people : ℕ := 9

-- State the main theorem to be proved
theorem amount_per_person_is_correct : total_amount / number_of_people = 266 := 
by sorry

end amount_per_person_is_correct_l271_271469


namespace number_of_students_playing_soccer_l271_271409

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer_l271_271409


namespace cylinder_curved_surface_area_l271_271960

theorem cylinder_curved_surface_area {r h : ℝ} (hr: r = 2) (hh: h = 5) :  2 * Real.pi * r * h = 20 * Real.pi :=
by
  rw [hr, hh]
  sorry

end cylinder_curved_surface_area_l271_271960


namespace solve_system_l271_271748

theorem solve_system (x y : ℝ) (h1 : x^2 + y^2 + x + y = 50) (h2 : x * y = 20) :
  (x = 5 ∧ y = 4) ∨ (x = 4 ∧ y = 5) ∨ (x = -5 + Real.sqrt 5 ∧ y = -5 - Real.sqrt 5) ∨ (x = -5 - Real.sqrt 5 ∧ y = -5 + Real.sqrt 5) :=
by
  sorry

end solve_system_l271_271748


namespace PQ_parallel_to_AB_3_times_l271_271410

-- Definitions for the problem
structure Rectangle :=
  (A B C D : Type)
  (AB AD : ℝ)
  (P Q : ℝ → ℝ)
  (P_speed Q_speed : ℝ)
  (time : ℝ)

noncomputable def rectangle_properties (R : Rectangle) : Prop :=
  R.AB = 4 ∧
  R.AD = 12 ∧
  ∀ t, 0 ≤ t → t ≤ 12 → R.P t = t ∧  -- P moves from A to D at 1 cm/s
  R.Q_speed = 3 ∧                     -- Q moves at 3 cm/s
  ∀ t, R.Q t = R.Q_speed * t ∧             -- Q moves from C to B and back
  ∃ s1 s2 s3, R.P s1 = 4 ∧ R.P s2 = 8 ∧ R.P s3 = 12 ∧
  (R.Q s1 = 3 ∨ R.Q s1 = 1) ∧
  (R.Q s2 = 6 ∨ R.Q s2 = 2) ∧
  (R.Q s3 = 9 ∨ R.Q s3 = 0)

theorem PQ_parallel_to_AB_3_times : 
  ∀ (R : Rectangle), rectangle_properties R → 
  ∃ (times : ℕ), times = 3 :=
by
  sorry

end PQ_parallel_to_AB_3_times_l271_271410


namespace product_evaluation_l271_271565

theorem product_evaluation (a b c : ℕ) (h : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) :
  6 * 15 * 2 = 4 := by
  sorry

end product_evaluation_l271_271565


namespace point_on_x_axis_equidistant_from_A_and_B_is_M_l271_271216

theorem point_on_x_axis_equidistant_from_A_and_B_is_M :
  ∃ M : ℝ × ℝ × ℝ, (M = (-3 / 2, 0, 0)) ∧ 
  (dist M (1, -3, 1) = dist M (2, 0, 2)) := by
  sorry

end point_on_x_axis_equidistant_from_A_and_B_is_M_l271_271216


namespace general_term_a_n_l271_271536

open BigOperators

variable {a : ℕ → ℝ}  -- The sequence a_n
variable {S : ℕ → ℝ}  -- The sequence sum S_n

-- Define the sum of the first n terms:
def seq_sum (a : ℕ → ℝ) (n : ℕ) := ∑ k in Finset.range (n + 1), a k

theorem general_term_a_n (h : ∀ n : ℕ, S n = 2 ^ n - 1) (n : ℕ) : a n = 2 ^ (n - 1) :=
by
  sorry

end general_term_a_n_l271_271536


namespace kevin_stone_count_l271_271678

theorem kevin_stone_count :
  ∃ (N : ℕ), (∀ (n k : ℕ), 2007 = 9 * n + 11 * k → N = 20) := 
sorry

end kevin_stone_count_l271_271678


namespace trapezoid_shaded_fraction_l271_271964

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end trapezoid_shaded_fraction_l271_271964


namespace second_number_is_72_l271_271485

-- Define the necessary variables and conditions
variables (x y : ℕ)
variables (h_first_num : x = 48)
variables (h_ratio : 48 / 8 = x / y)
variables (h_LCM : Nat.lcm x y = 432)

-- State the problem as a theorem
theorem second_number_is_72 : y = 72 :=
by
  sorry

end second_number_is_72_l271_271485


namespace triangle_count_lower_bound_l271_271102

theorem triangle_count_lower_bound (n m : ℕ) (S : Finset (ℕ × ℕ))
  (hS : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a < b ∧ b ≤ n) (hm : S.card = m) :
  ∃T, T ≥ 4 * m * (m - n^2 / 4) / (3 * n) := 
by 
  sorry

end triangle_count_lower_bound_l271_271102


namespace digit_product_inequality_l271_271832

noncomputable def digit_count_in_n (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).count d

theorem digit_product_inequality (n : ℕ) (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ)
  (h1 : a1 = digit_count_in_n n 1)
  (h2 : a2 = digit_count_in_n n 2)
  (h3 : a3 = digit_count_in_n n 3)
  (h4 : a4 = digit_count_in_n n 4)
  (h5 : a5 = digit_count_in_n n 5)
  (h6 : a6 = digit_count_in_n n 6)
  (h7 : a7 = digit_count_in_n n 7)
  (h8 : a8 = digit_count_in_n n 8)
  (h9 : a9 = digit_count_in_n n 9)
  : 2^a1 * 3^a2 * 4^a3 * 5^a4 * 6^a5 * 7^a6 * 8^a7 * 9^a8 * 10^a9 ≤ n + 1 :=
  sorry

end digit_product_inequality_l271_271832


namespace find_x_for_f_of_one_fourth_l271_271419

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then 2^(-x) else Real.log x / Real.log 4 

-- Define the proof problem
theorem find_x_for_f_of_one_fourth : 
  ∃ x : ℝ, (f x = 1 / 4) ∧ (x = Real.sqrt 2)  :=
sorry

end find_x_for_f_of_one_fourth_l271_271419


namespace spherical_to_rectangular_conversion_l271_271168

/-- Convert a point in spherical coordinates to rectangular coordinates given specific angles and distance -/
theorem spherical_to_rectangular_conversion :
  ∀ (ρ θ φ : ℝ) (x y z : ℝ), 
  ρ = 15 → θ = 225 * (Real.pi / 180) → φ = 45 * (Real.pi / 180) →
  x = ρ * Real.sin φ * Real.cos θ → y = ρ * Real.sin φ * Real.sin θ → z = ρ * Real.cos φ →
  x = -15 / 2 ∧ y = -15 / 2 ∧ z = 15 * Real.sqrt 2 / 2 := by
  sorry

end spherical_to_rectangular_conversion_l271_271168


namespace malcolm_initial_white_lights_l271_271901

theorem malcolm_initial_white_lights :
  let red_lights := 12
  let blue_lights := 3 * red_lights
  let green_lights := 6
  let bought_lights := red_lights + blue_lights + green_lights
  let remaining_lights := 5
  let total_needed_lights := bought_lights + remaining_lights
  W = total_needed_lights :=
by
  sorry

end malcolm_initial_white_lights_l271_271901


namespace find_xnp_l271_271836

theorem find_xnp (x n p : ℕ) (h1 : 0 < x) (h2 : 0 < n) (h3 : Nat.Prime p) 
                  (h4 : 2 * x^3 + x^2 + 10 * x + 5 = 2 * p^n) : x + n + p = 6 :=
by
  sorry

end find_xnp_l271_271836


namespace smallest_prime_with_digit_sum_22_l271_271654

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_prime_with_digit_sum_22 : ∃ p : ℕ, is_prime p ∧ digit_sum p = 22 ∧ 
  (∀ q : ℕ, is_prime q ∧ digit_sum q = 22 → p ≤ q) ∧ p = 499 :=
sorry

end smallest_prime_with_digit_sum_22_l271_271654


namespace number_of_green_fish_l271_271097

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l271_271097


namespace circle_circumference_difference_l271_271082

theorem circle_circumference_difference (d_inner : ℝ) (h_inner : d_inner = 100) 
  (d_outer : ℝ) (h_outer : d_outer = d_inner + 30) :
  ((π * d_outer) - (π * d_inner)) = 30 * π :=
by 
  sorry

end circle_circumference_difference_l271_271082


namespace sum_of_divisors_90_l271_271774

theorem sum_of_divisors_90 : 
  let n := 90 in 
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5) in
  sum_divisors n = 234 :=
by 
  let n := 90
  let sum_divisors (n : ℕ) : ℕ := (1 + 2) * (1 + 3 + 3^2) * (1 + 5)
  sorry

end sum_of_divisors_90_l271_271774


namespace selling_price_of_car_l271_271435

theorem selling_price_of_car (purchase_price repair_cost : ℝ) (profit_percent : ℝ) 
    (h1 : purchase_price = 42000) (h2 : repair_cost = 8000) (h3 : profit_percent = 29.8) :
    (purchase_price + repair_cost) * (1 + profit_percent / 100) = 64900 := 
by 
  -- The proof will go here
  sorry

end selling_price_of_car_l271_271435


namespace circle_radius_l271_271365

theorem circle_radius (x y : ℝ) : x^2 + 8*x + y^2 - 10*y + 32 = 0 → ∃ r : ℝ, r = 3 :=
by
  sorry

end circle_radius_l271_271365


namespace seat_notation_l271_271211

theorem seat_notation (row1 col1 row2 col2 : ℕ) (h : (row1, col1) = (5, 2)) : (row2, col2) = (7, 3) :=
 by
  sorry

end seat_notation_l271_271211


namespace shaina_keeps_chocolate_l271_271582

theorem shaina_keeps_chocolate :
  let total_chocolate := (60 : ℚ) / 7
  let number_of_piles := 5
  let weight_per_pile := total_chocolate / number_of_piles
  let given_weight_back := (1 / 2) * weight_per_pile
  let kept_weight := weight_per_pile - given_weight_back
  kept_weight = 6 / 7 :=
by
  sorry

end shaina_keeps_chocolate_l271_271582


namespace convert_mps_to_kmph_l271_271279

-- Define the conversion factor
def conversion_factor : ℝ := 3.6

-- Define the initial speed in meters per second
def initial_speed_mps : ℝ := 50

-- Define the target speed in kilometers per hour
def target_speed_kmph : ℝ := 180

-- Problem statement: Prove the conversion is correct
theorem convert_mps_to_kmph : initial_speed_mps * conversion_factor = target_speed_kmph := by
  sorry

end convert_mps_to_kmph_l271_271279


namespace right_triangle_hypotenuse_l271_271184

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 
  ∃ h : ℝ, h = Real.sqrt (a^2 + b^2) ∧ h = Real.sqrt 34 := 
by
  sorry

end right_triangle_hypotenuse_l271_271184


namespace smallest_prime_with_digits_sum_22_l271_271653

def digits_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem smallest_prime_with_digits_sum_22 : 
  ∃ p : ℕ, Prime p ∧ digits_sum p = 22 ∧ ∀ q : ℕ, Prime q ∧ digits_sum q = 22 → q ≥ p ∧ p = 499 :=
by sorry

end smallest_prime_with_digits_sum_22_l271_271653


namespace all_push_ups_total_l271_271942

-- Definitions derived from the problem's conditions
def ZacharyPushUps := 47
def DavidPushUps := ZacharyPushUps + 15
def EmilyPushUps := DavidPushUps * 2
def TotalPushUps := ZacharyPushUps + DavidPushUps + EmilyPushUps

-- The statement to be proved
theorem all_push_ups_total : TotalPushUps = 233 := by
  sorry

end all_push_ups_total_l271_271942


namespace trivia_team_total_score_l271_271965

theorem trivia_team_total_score 
  (scores : List ℕ)
  (present_members : List ℕ)
  (H_score : scores = [4, 6, 2, 8, 3, 5, 10, 3, 7])
  (H_present : present_members = scores) :
  List.sum present_members = 48 := 
by
  sorry

end trivia_team_total_score_l271_271965


namespace diagonals_in_octagon_l271_271393

theorem diagonals_in_octagon (n : ℕ) (h : n = 8) : (nat.choose n 2) - n = 20 :=
by
  rw [h, nat.choose]
  sorry

end diagonals_in_octagon_l271_271393


namespace problem_statement_l271_271996

theorem problem_statement (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end problem_statement_l271_271996


namespace n_squared_plus_n_plus_1_is_odd_l271_271434

theorem n_squared_plus_n_plus_1_is_odd (n : ℤ) : Odd (n^2 + n + 1) :=
sorry

end n_squared_plus_n_plus_1_is_odd_l271_271434


namespace initial_average_runs_l271_271603

theorem initial_average_runs (A : ℕ) (h : 10 * A + 87 = 11 * (A + 5)) : A = 32 :=
by
  sorry

end initial_average_runs_l271_271603


namespace aardvark_total_distance_l271_271925

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end aardvark_total_distance_l271_271925


namespace consecutive_ints_product_div_6_l271_271479

theorem consecutive_ints_product_div_6 (n : ℤ) : (n * (n + 1) * (n + 2)) % 6 = 0 := 
sorry

end consecutive_ints_product_div_6_l271_271479


namespace balls_boxes_distribution_l271_271852

/-- There are 5 ways to put 6 indistinguishable balls into 3 indistinguishable boxes. -/
theorem balls_boxes_distribution : ∃ (S : Finset (Finset ℕ)), S.card = 5 ∧
  ∀ (s ∈ S), ∑ x in s, x = 6 ∧ s.card <= 3 :=
begin
  sorry,
end

end balls_boxes_distribution_l271_271852


namespace find_essay_pages_l271_271886

/-
Conditions:
1. It costs $0.10 to print one page.
2. Jenny wants to print 7 copies of her essay.
3. Jenny wants to buy 7 pens that each cost $1.50.
4. Jenny pays the store with 2 twenty dollar bills and gets $12 in change.
-/

def cost_per_page : Float := 0.10
def number_of_copies : Nat := 7
def cost_per_pen : Float := 1.50
def number_of_pens : Nat := 7
def total_money_given : Float := 40.00  -- 2 twenty dollar bills
def change_received : Float := 12.00

theorem find_essay_pages :
  let total_spent := total_money_given - change_received
  let total_cost_of_pens := Float.ofNat number_of_pens * cost_per_pen
  let total_amount_spent_on_printing := total_spent - total_cost_of_pens
  let number_of_pages := total_amount_spent_on_printing / cost_per_page
  number_of_pages = 175 := by
  sorry

end find_essay_pages_l271_271886


namespace smallest_feared_sequence_l271_271502

def is_feared (n : ℕ) : Prop :=
  -- This function checks if a number contains '13' as a contiguous substring.
  sorry

def is_fearless (n : ℕ) : Prop := ¬is_feared n

theorem smallest_feared_sequence : ∃ (n : ℕ) (a : ℕ), 0 < n ∧ a < 100 ∧ is_fearless n ∧ is_fearless (n + 10 * a) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → is_feared (n + k * a)) ∧ n = 1287 := 
by
  sorry

end smallest_feared_sequence_l271_271502


namespace Fred_earned_4_dollars_l271_271416

-- Conditions are translated to definitions
def initial_amount_Fred : ℕ := 111
def current_amount_Fred : ℕ := 115

-- Proof problem in Lean 4 statement
theorem Fred_earned_4_dollars : current_amount_Fred - initial_amount_Fred = 4 := by
  sorry

end Fred_earned_4_dollars_l271_271416


namespace fraction_not_exist_implies_x_neg_one_l271_271874

theorem fraction_not_exist_implies_x_neg_one {x : ℝ} :
  ¬(∃ y : ℝ, y = 1 / (x + 1)) → x = -1 :=
by
  intro h
  have : x + 1 = 0 :=
    by
      contrapose! h
      exact ⟨1 / (x + 1), rfl⟩
  linarith

end fraction_not_exist_implies_x_neg_one_l271_271874


namespace girls_25_percent_less_false_l271_271349

theorem girls_25_percent_less_false (g b : ℕ) (h : b = g * 125 / 100) : (b - g) / b ≠ 25 / 100 := by
  sorry

end girls_25_percent_less_false_l271_271349


namespace robin_initial_gum_is_18_l271_271436

-- Defining the conditions as given in the problem
def given_gum : ℝ := 44
def total_gum : ℝ := 62

-- Statement to prove that the initial number of pieces of gum Robin had is 18
theorem robin_initial_gum_is_18 : total_gum - given_gum = 18 := by
  -- Proof goes here
  sorry

end robin_initial_gum_is_18_l271_271436


namespace proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l271_271932

open Classical

variable (a b x y : ℝ)

theorem proposition_A_correct (h : a > 1) : (1 / a < 1) ∧ ¬((1 / a < 1) → (a > 1)) :=
sorry

theorem proposition_B_incorrect (h_neg : ¬(x < 1 → x^2 < 1)) : ¬(∃ x, x ≥ 1 ∧ x^2 ≥ 1) :=
sorry

theorem proposition_C_incorrect (h_xy : x ≥ 2 ∧ y ≥ 2) : ¬((x ≥ 2 ∧ y ≥ 2) → x^2 + y^2 ≥ 4) :=
sorry

theorem proposition_D_correct (h_a : a ≠ 0) : (a * b ≠ 0) ∧ ¬((a * b ≠ 0) → (a ≠ 0)) :=
sorry

end proposition_A_correct_proposition_B_incorrect_proposition_C_incorrect_proposition_D_correct_l271_271932


namespace negation_of_proposition_l271_271902

noncomputable def negation_proposition (f : ℝ → Prop) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ ¬ f x

theorem negation_of_proposition :
  (∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l271_271902


namespace area_sum_of_three_circles_l271_271673

theorem area_sum_of_three_circles (R d : ℝ) (x y z : ℝ) 
    (hxyz : x^2 + y^2 + z^2 = d^2) :
    (π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2))) = π * (3 * R^2 - d^2) :=
by
  sorry

end area_sum_of_three_circles_l271_271673


namespace maximise_expression_l271_271828

theorem maximise_expression {x : ℝ} (hx : 0 < x ∧ x < 1) : 
  ∃ (x_max : ℝ), x_max = 1/2 ∧ 
  (∀ y : ℝ, (0 < y ∧ y < 1) → 3 * y * (1 - y) ≤ 3 * x_max * (1 - x_max)) :=
sorry

end maximise_expression_l271_271828


namespace quadratic_expression_neg_for_all_x_l271_271688

theorem quadratic_expression_neg_for_all_x (m : ℝ) :
  (∀ x : ℝ, m*x^2 + (m-1)*x + (m-1) < 0) ↔ m < -1/3 :=
sorry

end quadratic_expression_neg_for_all_x_l271_271688


namespace range_of_a_l271_271713

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ a < -1 ∨ a > 3 :=
sorry

end range_of_a_l271_271713


namespace problem1_problem2_l271_271554

noncomputable def f (x a b : ℝ) : ℝ := 2 * x ^ 2 - 2 * a * x + b

noncomputable def set_A (a b : ℝ) : Set ℝ := {x | f x a b > 0 }

noncomputable def set_B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1 }

theorem problem1 (a b : ℝ) (h : f (-1) a b = -8) :
  (∀ x, x ∈ (set_A a b)ᶜ ∪ set_B 1 ↔ -3 ≤ x ∧ x ≤ 2) :=
  sorry

theorem problem2 (a b : ℝ) (t : ℝ) (h : f (-1) a b = -8) (h_not_P : (set_A a b) ∩ (set_B t) = ∅) :
  -2 ≤ t ∧ t ≤ 0 :=
  sorry

end problem1_problem2_l271_271554


namespace min_adjacent_white_cells_8x8_grid_l271_271538

theorem min_adjacent_white_cells_8x8_grid (n_blacks : ℕ) (h1 : n_blacks = 20) : 
  ∃ w_cell_pairs, w_cell_pairs = 34 :=
by
  -- conditions are translated here for interpret
  let total_pairs := 112 -- total pairs in 8x8 grid
  let max_spoiled := 78  -- maximum spoiled pairs when placing 20 black cells
  let min_adjacent_white_pairs := total_pairs - max_spoiled
  use min_adjacent_white_pairs
  exact (by linarith)
  sorry

end min_adjacent_white_cells_8x8_grid_l271_271538


namespace sally_reads_10_pages_on_weekdays_l271_271241

def sallyReadsOnWeekdays (x : ℕ) (total_pages : ℕ) (weekdays : ℕ) (weekend_days : ℕ) (weekend_pages : ℕ) : Prop :=
  (weekdays + weekend_days * weekend_pages = total_pages) → (weekdays * x = total_pages - weekend_days * weekend_pages)

theorem sally_reads_10_pages_on_weekdays :
  sallyReadsOnWeekdays 10 180 10 4 20 :=
by
  intros h
  sorry  -- proof to be filled in

end sally_reads_10_pages_on_weekdays_l271_271241


namespace root_of_equation_in_interval_l271_271910

theorem root_of_equation_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2^x = 2 - x := 
sorry

end root_of_equation_in_interval_l271_271910


namespace kiera_total_envelopes_l271_271891

-- Define the number of blue envelopes
def blue_envelopes : ℕ := 14

-- Define the number of yellow envelopes as 6 fewer than the number of blue envelopes
def yellow_envelopes : ℕ := blue_envelopes - 6

-- Define the number of green envelopes as 3 times the number of yellow envelopes
def green_envelopes : ℕ := 3 * yellow_envelopes

-- The total number of envelopes is the sum of blue, yellow, and green envelopes
def total_envelopes : ℕ := blue_envelopes + yellow_envelopes + green_envelopes

-- Prove that the total number of envelopes is 46
theorem kiera_total_envelopes : total_envelopes = 46 := by
  sorry

end kiera_total_envelopes_l271_271891


namespace find_a_l271_271280

-- Given conditions
def div_by_3 (a : ℤ) : Prop :=
  (5 * a + 1) % 3 = 0 ∨ (3 * a + 2) % 3 = 0

def div_by_5 (a : ℤ) : Prop :=
  (5 * a + 1) % 5 = 0 ∨ (3 * a + 2) % 5 = 0

-- Proving the question 
theorem find_a (a : ℤ) : div_by_3 a ∧ div_by_5 a → a % 15 = 4 :=
by
  sorry

end find_a_l271_271280


namespace arithmetic_seq_common_difference_l271_271084

theorem arithmetic_seq_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 * a 11 = 6) (h2 : a 4 + a (14) = 5) : 
  d = 1 / 4 ∨ d = -1 / 4 :=
sorry

end arithmetic_seq_common_difference_l271_271084


namespace find_numbers_l271_271123

-- Define the conditions
def condition_1 (L S : ℕ) : Prop := L - S = 8327
def condition_2 (L S : ℕ) : Prop := ∃ q r, L = q * S + r ∧ q = 21 ∧ r = 125

-- Define the math proof problem
theorem find_numbers (S L : ℕ) (h1 : condition_1 L S) (h2 : condition_2 L S) : S = 410 ∧ L = 8735 :=
by
  sorry

end find_numbers_l271_271123


namespace problem_statement_l271_271051

variables {Point Line Plane : Type}
variables (l : Line) (α β : Plane)

-- Conditions
def parallel (l : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def perpendicular_planes (α β : Plane) : Prop := sorry

-- The proof problem
theorem problem_statement (h1 : parallel l α) (h2 : perpendicular l β) : perpendicular_planes α β :=
sorry

end problem_statement_l271_271051


namespace branches_number_l271_271578

-- Conditions (converted into Lean definitions)
def total_leaves : ℕ := 12690
def twigs_per_branch : ℕ := 90
def leaves_per_twig_percentage_4 : ℝ := 0.3
def leaves_per_twig_percentage_5 : ℝ := 0.7
def leaves_per_twig_4 : ℕ := 4
def leaves_per_twig_5 : ℕ := 5

-- The goal
theorem branches_number (B : ℕ) 
  (h1 : twigs_per_branch = 90) 
  (h2 : leaves_per_twig_percentage_4 = 0.3) 
  (h3 : leaves_per_twig_percentage_5 = 0.7) 
  (h4 : leaves_per_twig_4 = 4) 
  (h5 : leaves_per_twig_5 = 5) 
  (h6 : total_leaves = 12690) :
  B = 30 := 
sorry

end branches_number_l271_271578


namespace partition_6_balls_into_3_boxes_l271_271861

def ways_to_partition_balls (balls boxes : ℕ) : ℕ :=
  if boxes = 1 then 1
  else if balls = 0 then 1
  else nat.choose (balls + boxes - 1) (boxes - 1)

theorem partition_6_balls_into_3_boxes : ways_to_partition_balls 6 3 = 6 :=
  by sorry

end partition_6_balls_into_3_boxes_l271_271861


namespace coefficient_x17_x18_l271_271354

noncomputable def coefficient_x_pow (c : ℕ) (n : ℕ) : ℕ :=
(multichoose (λ k, if k ∈ {5, 7} then 1 else 0) n).filter (λ s, s.sum = c).card 

theorem coefficient_x17_x18 : 
  coefficient_x_pow 17 20 = 3420 ∧ coefficient_x_pow 18 20 = 0 := 
by 
  sorry

end coefficient_x17_x18_l271_271354


namespace asphalt_road_proof_l271_271137

-- We define the initial conditions given in the problem
def man_hours (men days hours_per_day : Nat) : Nat :=
  men * days * hours_per_day

-- Given the conditions for asphalting 1 km road
def conditions_1 (men1 days1 hours_per_day1 : Nat) : Prop :=
  man_hours men1 days1 hours_per_day1 = 2880

-- Given that the second road is 2 km long
def conditions_2 (man_hours1 : Nat) : Prop :=
  2 * man_hours1 = 5760

-- Given the working conditions for the second road
def conditions_3 (men2 days2 hours_per_day2 : Nat) : Prop :=
  men2 * days2 * hours_per_day2 = 5760

-- The theorem to prove
theorem asphalt_road_proof 
  (men1 days1 hours_per_day1 days2 hours_per_day2 men2 : Nat)
  (H1 : conditions_1 men1 days1 hours_per_day1)
  (H2 : conditions_2 (man_hours men1 days1 hours_per_day1))
  (H3 : men2 * days2 * hours_per_day2 = 5760)
  : men2 = 20 :=
by
  sorry

end asphalt_road_proof_l271_271137


namespace spent_on_computer_accessories_l271_271596

theorem spent_on_computer_accessories :
  ∀ (x : ℕ), (original : ℕ) (snacks : ℕ) (remaining : ℕ),
  original = 48 →
  snacks = 8 →
  remaining = 4 + original / 2 →
  original - (x + snacks) = remaining →
  x = 12 :=
by
  intros x original snacks remaining
  intro h_original
  intro h_snacks
  intro h_remaining
  intro h_spent
  sorry

end spent_on_computer_accessories_l271_271596


namespace simplify_trig_expression_l271_271245

open Real

theorem simplify_trig_expression (α : ℝ) : 
  (cos (2 * π + α) * tan (π + α)) / cos (π / 2 - α) = 1 := 
sorry

end simplify_trig_expression_l271_271245


namespace final_position_correct_l271_271154

structure Position :=
(base : ℝ × ℝ)
(stem : ℝ × ℝ)

def initial_position : Position :=
{ base := (0, -1),
  stem := (1, 0) }

def reflect_x (p : Position) : Position :=
{ base := (p.base.1, -p.base.2),
  stem := (p.stem.1, -p.stem.2) }

def rotate_90_ccw (p : Position) : Position :=
{ base := (-p.base.2, p.base.1),
  stem := (-p.stem.2, p.stem.1) }

def half_turn (p : Position) : Position :=
{ base := (-p.base.1, -p.base.2),
  stem := (-p.stem.1, -p.stem.2) }

def reflect_y (p : Position) : Position :=
{ base := (-p.base.1, p.base.2),
  stem := (-p.stem.1, p.stem.2) }

def final_position : Position :=
reflect_y (half_turn (rotate_90_ccw (reflect_x initial_position)))

theorem final_position_correct : final_position = { base := (1, 0), stem := (0, 1) } :=
sorry

end final_position_correct_l271_271154


namespace prove_P_A1_prove_P_B_prove_P_A1_given_B_l271_271136

-- Define constants for defect rates
def defect_rate_lathe1 : ℝ := 0.06
def defect_rate_lathe2 : ℝ := 0.05
def defect_rate_lathe3 : ℝ := 0.04

-- Define constants for the probability ratios
def parts_ratio_lathe1 : ℝ := 5
def parts_ratio_lathe2 : ℝ := 6
def parts_ratio_lathe3 : ℝ := 9

-- Calculate total parts ratio
def total_parts_ratio : ℝ := parts_ratio_lathe1 + parts_ratio_lathe2 + parts_ratio_lathe3

-- Calculate the probability of a part being processed by each lathe
def P_A1 : ℝ := parts_ratio_lathe1 / total_parts_ratio
def P_A2 : ℝ := parts_ratio_lathe2 / total_parts_ratio
def P_A3 : ℝ := parts_ratio_lathe3 / total_parts_ratio

-- Calculate the overall probability of selecting a defective part (P(B))
def P_B : ℝ := P_A1 * defect_rate_lathe1 + P_A2 * defect_rate_lathe2 + P_A3 * defect_rate_lathe3

-- Calculate the conditional probability P(A1|B) using Bayes' theorem
def P_A1_given_B : ℝ := (defect_rate_lathe1 * P_A1) / P_B

-- Prove the required statements

-- Prove that P(A1) = 0.25
theorem prove_P_A1 : P_A1 = 0.25 := by
  sorry

-- Prove that P(B) = 0.048
theorem prove_P_B : P_B = 0.048 := by
  sorry

-- Prove that P(A1|B) = 5 / 16
theorem prove_P_A1_given_B : P_A1_given_B = 5 / 16 := by
  sorry

end prove_P_A1_prove_P_B_prove_P_A1_given_B_l271_271136


namespace mike_office_visits_per_day_l271_271230

-- Define the constants from the conditions
def pull_ups_per_visit : ℕ := 2
def total_pull_ups_per_week : ℕ := 70
def days_per_week : ℕ := 7

-- Calculate total office visits per week
def office_visits_per_week : ℕ := total_pull_ups_per_week / pull_ups_per_visit

-- Lean statement that states Mike goes into his office 5 times a day
theorem mike_office_visits_per_day : office_visits_per_week / days_per_week = 5 := by
  sorry

end mike_office_visits_per_day_l271_271230


namespace bill_annual_healthcare_cost_l271_271976

def hourly_wage := 25
def weekly_hours := 30
def weeks_per_month := 4
def months_per_year := 12
def normal_monthly_price := 500
def annual_income := hourly_wage * weekly_hours * weeks_per_month * months_per_year
def subsidy (income : ℕ) : ℕ :=
  if income < 10000 then 90
  else if income ≤ 40000 then 50
  else if income > 50000 then 20
  else 0
def monthly_cost_after_subsidy := (normal_monthly_price * (100 - subsidy annual_income)) / 100
def annual_cost := monthly_cost_after_subsidy * months_per_year

theorem bill_annual_healthcare_cost : annual_cost = 3000 := by
  sorry

end bill_annual_healthcare_cost_l271_271976


namespace solve_y_equation_l271_271110

noncomputable def solve_y : ℚ :=
  let y := (500 * 1 : ℚ) / 15 in
  y

theorem solve_y_equation (y : ℚ) :
  2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = solve_y := by
  intro h
  sorry

end solve_y_equation_l271_271110


namespace taras_total_gas_spent_is_180_l271_271113

def trip_duration := 2 -- in days
def gas_stations := 4 -- number of gas stations visited
def gas_prices := [3.0, 3.5, 4.0, 4.5] -- price per gallon at each gas station
def tank_capacity := 12.0 -- tank capacity in gallons

def total_gas_spent : ℝ :=
  (tank_capacity * gas_prices[0]) +
  (tank_capacity * gas_prices[1]) +
  (tank_capacity * gas_prices[2]) +
  (tank_capacity * gas_prices[3])

theorem taras_total_gas_spent_is_180 :
  total_gas_spent = 180 :=
by
  sorry

end taras_total_gas_spent_is_180_l271_271113


namespace h_h_of_2_l271_271418

def h (x : ℝ) : ℝ := 4 * x^2 - 8

theorem h_h_of_2 : h (h 2) = 248 := by
  -- Proof goes here
  sorry

end h_h_of_2_l271_271418


namespace unemployment_percentage_next_year_l271_271490

theorem unemployment_percentage_next_year (E U : ℝ) (h1 : E > 0) :
  ( (0.91 * (0.056 * E)) / (1.04 * E) ) * 100 = 4.9 := by
  sorry

end unemployment_percentage_next_year_l271_271490


namespace initial_bottle_caps_l271_271939

theorem initial_bottle_caps (bought_caps total_caps initial_caps : ℕ) 
  (hb : bought_caps = 41) (ht : total_caps = 43):
  initial_caps = 2 :=
by
  have h : total_caps = initial_caps + bought_caps := sorry
  have ha : initial_caps = total_caps - bought_caps := sorry
  exact sorry

end initial_bottle_caps_l271_271939


namespace partition_6_balls_into_3_boxes_l271_271860

def ways_to_partition_balls (balls boxes : ℕ) : ℕ :=
  if boxes = 1 then 1
  else if balls = 0 then 1
  else nat.choose (balls + boxes - 1) (boxes - 1)

theorem partition_6_balls_into_3_boxes : ways_to_partition_balls 6 3 = 6 :=
  by sorry

end partition_6_balls_into_3_boxes_l271_271860


namespace smallest_t_for_circle_l271_271608

theorem smallest_t_for_circle (t : ℝ) :
  (∀ r θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) → t ≥ π :=
by sorry

end smallest_t_for_circle_l271_271608


namespace sum_of_digits_is_11_l271_271753

def digits_satisfy_conditions (A B C : ℕ) : Prop :=
  (C = 0 ∨ C = 5) ∧
  (A = 2 * B) ∧
  (A * B * C = 40)

theorem sum_of_digits_is_11 (A B C : ℕ) (h : digits_satisfy_conditions A B C) : A + B + C = 11 :=
by
  sorry

end sum_of_digits_is_11_l271_271753


namespace gcd_8Tn_nplus1_eq_4_l271_271824

noncomputable def T_n (n : ℕ) : ℕ :=
(n * (n + 1)) / 2

theorem gcd_8Tn_nplus1_eq_4 (n : ℕ) (hn: 0 < n) : gcd (8 * T_n n) (n + 1) = 4 :=
sorry

end gcd_8Tn_nplus1_eq_4_l271_271824


namespace initially_planned_days_l271_271947

theorem initially_planned_days (D : ℕ) (h1 : 6 * 3 + 10 * 3 = 6 * D) : D = 8 := by
  sorry

end initially_planned_days_l271_271947


namespace solve_for_c_l271_271440

theorem solve_for_c (c : ℚ) :
  (c - 35) / 14 = (2 * c + 9) / 49 →
  c = 1841 / 21 :=
by
  sorry

end solve_for_c_l271_271440
