import Mathlib

namespace bananas_in_each_box_l2345_234534

-- You might need to consider noncomputable if necessary here for Lean's real number support.
noncomputable def bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) : ℕ :=
  total_bananas / total_boxes

theorem bananas_in_each_box :
  bananas_per_box 40 8 = 5 := by
  sorry

end bananas_in_each_box_l2345_234534


namespace sum_of_midpoints_l2345_234528

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_of_midpoints_l2345_234528


namespace fraction_value_l2345_234585

theorem fraction_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1 : ℚ) / (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3/4 :=
sorry

end fraction_value_l2345_234585


namespace smallest_area_is_10_l2345_234509

noncomputable def smallest_square_area : ℝ :=
  let k₁ := 65
  let k₂ := -5
  10 * (9 + 4 * k₂)

theorem smallest_area_is_10 :
  smallest_square_area = 10 := by
  sorry

end smallest_area_is_10_l2345_234509


namespace team_total_points_l2345_234543

theorem team_total_points (Connor_score Amy_score Jason_score : ℕ) :
  Connor_score = 2 →
  Amy_score = Connor_score + 4 →
  Jason_score = 2 * Amy_score →
  Connor_score + Amy_score + Jason_score = 20 :=
by
  intros
  sorry

end team_total_points_l2345_234543


namespace necessary_and_sufficient_condition_l2345_234527

theorem necessary_and_sufficient_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (m + n > m * n) ↔ (m = 1 ∨ n = 1) := by
  sorry

end necessary_and_sufficient_condition_l2345_234527


namespace sufficient_but_not_necessary_condition_l2345_234531

theorem sufficient_but_not_necessary_condition 
  (a : ℕ → ℤ) 
  (h : ∀ n, |a (n + 1)| < a n) : 
  (∀ n, a (n + 1) < a n) ∧ 
  ¬(∀ n, a (n + 1) < a n → |a (n + 1)| < a n) := 
by 
  sorry

end sufficient_but_not_necessary_condition_l2345_234531


namespace sum_and_count_even_l2345_234587

-- Sum of integers from a to b (inclusive)
def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

-- Number of even integers from a to b (inclusive)
def count_even_integers (a b : ℕ) : ℕ :=
  ((b - if b % 2 == 0 then 0 else 1) - (a + if a % 2 == 0 then 0 else 1)) / 2 + 1

theorem sum_and_count_even (x y : ℕ) :
  x = sum_of_integers 20 40 →
  y = count_even_integers 20 40 →
  x + y = 641 :=
by
  intros
  sorry

end sum_and_count_even_l2345_234587


namespace fraction_value_l2345_234552

theorem fraction_value : (3 - (-3)) / (2 - 1) = 6 := 
by
  sorry

end fraction_value_l2345_234552


namespace people_not_in_pool_l2345_234539

noncomputable def total_people_karen_donald : ℕ := 2
noncomputable def children_karen_donald : ℕ := 6
noncomputable def total_people_tom_eva : ℕ := 2
noncomputable def children_tom_eva : ℕ := 4
noncomputable def legs_in_pool : ℕ := 16

theorem people_not_in_pool : total_people_karen_donald + children_karen_donald + total_people_tom_eva + children_tom_eva - (legs_in_pool / 2) = 6 := by
  sorry

end people_not_in_pool_l2345_234539


namespace hyperbola_eccentricity_is_sqrt_3_l2345_234546

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_is_sqrt_3 (a b : ℝ) (h1 : a > 0) (h2 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt_3_l2345_234546


namespace latest_departure_time_l2345_234582

noncomputable def minutes_in_an_hour : ℕ := 60
noncomputable def departure_time : ℕ := 20 * minutes_in_an_hour -- 8:00 pm in minutes
noncomputable def checkin_time : ℕ := 2 * minutes_in_an_hour -- 2 hours in minutes
noncomputable def drive_time : ℕ := 45 -- 45 minutes
noncomputable def parking_time : ℕ := 15 -- 15 minutes
noncomputable def total_time_needed : ℕ := checkin_time + drive_time + parking_time -- Total time in minutes

theorem latest_departure_time : departure_time - total_time_needed = 17 * minutes_in_an_hour :=
by
  sorry

end latest_departure_time_l2345_234582


namespace expression_value_l2345_234557

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x^7 + y^7 + z^7) / (x * y * z * (x * y + x * z + y * z))

theorem expression_value
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod_nonzero : x * y + x * z + y * z ≠ 0) :
  expression x y z = -7 :=
by 
  sorry

end expression_value_l2345_234557


namespace x_intercept_of_translated_line_l2345_234578

theorem x_intercept_of_translated_line :
  let line_translation (y : ℝ) := y + 4
  let new_line_eq := fun (x : ℝ) => 2 * x - 2
  new_line_eq 1 = 0 :=
by
  sorry

end x_intercept_of_translated_line_l2345_234578


namespace symmetrical_point_wrt_x_axis_l2345_234504

theorem symmetrical_point_wrt_x_axis (x y : ℝ) (P_symmetrical : (ℝ × ℝ)) (hx : x = -1) (hy : y = 2) : 
  P_symmetrical = (x, -y) → P_symmetrical = (-1, -2) :=
by
  intros h
  rw [hx, hy] at h
  exact h

end symmetrical_point_wrt_x_axis_l2345_234504


namespace seven_b_value_l2345_234563

theorem seven_b_value (a b : ℚ) (h₁ : 8 * a + 3 * b = 0) (h₂ : a = b - 3) :
  7 * b = 168 / 11 :=
sorry

end seven_b_value_l2345_234563


namespace find_sticker_price_l2345_234580

-- Defining the conditions:
def sticker_price (x : ℝ) : Prop := 
  let price_A := 0.85 * x - 90
  let price_B := 0.75 * x
  price_A + 15 = price_B

-- Proving the sticker price is $750 given the conditions
theorem find_sticker_price : ∃ x : ℝ, sticker_price x ∧ x = 750 := 
by
  use 750
  simp [sticker_price]
  sorry

end find_sticker_price_l2345_234580


namespace lcm_proof_l2345_234581

theorem lcm_proof (a b c : ℕ) (h1 : Nat.lcm a b = 60) (h2 : Nat.lcm a c = 270) : Nat.lcm b c = 540 :=
sorry

end lcm_proof_l2345_234581


namespace wire_cut_ratio_l2345_234568

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l2345_234568


namespace profit_percentage_is_22_percent_l2345_234566

-- Define the given conditions
def scooter_cost (C : ℝ) := C
def repair_cost (C : ℝ) := 0.10 * C
def repair_cost_value := 500
def profit := 1100

-- Let's state the main theorem
theorem profit_percentage_is_22_percent (C : ℝ) 
  (h1 : repair_cost C = repair_cost_value)
  (h2 : profit = 1100) : 
  (profit / C) * 100 = 22 :=
by
  sorry

end profit_percentage_is_22_percent_l2345_234566


namespace phone_call_answered_within_first_four_rings_l2345_234559

def P1 := 0.1
def P2 := 0.3
def P3 := 0.4
def P4 := 0.1

theorem phone_call_answered_within_first_four_rings :
  P1 + P2 + P3 + P4 = 0.9 :=
by
  rw [P1, P2, P3, P4]
  norm_num
  sorry -- Proof step skipped

end phone_call_answered_within_first_four_rings_l2345_234559


namespace max_perimeter_right_triangle_l2345_234594

theorem max_perimeter_right_triangle (a b : ℝ) (h₁ : a^2 + b^2 = 25) :
  (a + b + 5) ≤ 5 + 5 * Real.sqrt 2 :=
by
  sorry

end max_perimeter_right_triangle_l2345_234594


namespace tenth_battery_replacement_in_january_l2345_234519

theorem tenth_battery_replacement_in_january : ∀ (months_to_replace: ℕ) (start_month: ℕ), 
  months_to_replace = 4 → start_month = 1 → (4 * (10 - 1)) % 12 = 0 → start_month = 1 :=
by
  intros months_to_replace start_month h_replace h_start h_calc
  sorry

end tenth_battery_replacement_in_january_l2345_234519


namespace height_drawn_to_hypotenuse_l2345_234540

-- Definitions for the given problem
variables {A B C D : Type}
variables {area : ℝ}
variables {angle_ratio : ℝ}
variables {h : ℝ}

-- Given conditions
def is_right_triangle (A B C : Type) : Prop := -- definition for the right triangle
sorry

def area_of_triangle (A B C : Type) (area: ℝ) : Prop := 
area = ↑(2 : ℝ) * Real.sqrt 3  -- area given as 2√3 cm²

def angle_bisector_ratios (A B C D : Type) (ratio: ℝ) : Prop :=
ratio = 1 / 2  -- given ratio 1:2

-- Question statement
theorem height_drawn_to_hypotenuse (A B C D : Type) 
  (right_triangle : is_right_triangle A B C)
  (area_cond : area_of_triangle A B C area)
  (angle_ratio_cond : angle_bisector_ratios A B C D angle_ratio):
  h = Real.sqrt 3 :=
sorry

end height_drawn_to_hypotenuse_l2345_234540


namespace find_deeper_depth_l2345_234588

noncomputable def swimming_pool_depth_proof 
  (width : ℝ) (length : ℝ) (shallow_depth : ℝ) (volume : ℝ) : Prop :=
  volume = (1 / 2) * (shallow_depth + 4) * width * length

theorem find_deeper_depth
  (h : width = 9)
  (l : length = 12)
  (a : shallow_depth = 1)
  (V : volume = 270) :
  swimming_pool_depth_proof 9 12 1 270 := by
  sorry

end find_deeper_depth_l2345_234588


namespace mul_3_6_0_5_l2345_234516

theorem mul_3_6_0_5 : 3.6 * 0.5 = 1.8 :=
by
  sorry

end mul_3_6_0_5_l2345_234516


namespace total_porridge_l2345_234586

variable {c1 c2 c3 c4 c5 c6 : ℝ}

theorem total_porridge (h1 : c3 = c1 + c2)
                      (h2 : c4 = c2 + c3)
                      (h3 : c5 = c3 + c4)
                      (h4 : c6 = c4 + c5)
                      (h5 : c5 = 10) :
                      c1 + c2 + c3 + c4 + c5 + c6 = 40 := 
by
  sorry

end total_porridge_l2345_234586


namespace maddie_milk_usage_l2345_234551

-- Define the constants based on the problem conditions
def cups_per_day : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def bag_cost : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def weekly_coffee_expense : ℝ := 18
def gallon_milk_cost : ℝ := 4

-- Define the proof problem
theorem maddie_milk_usage : 
  (0.5 : ℝ) = (weekly_coffee_expense - 2 * ((cups_per_day * ounces_per_cup * 7) / ounces_per_bag * bag_cost)) / gallon_milk_cost :=
by 
  sorry

end maddie_milk_usage_l2345_234551


namespace number_of_bushes_l2345_234596

theorem number_of_bushes (T B x y : ℕ) (h1 : B = T - 6) (h2 : x ≥ y + 10) (h3 : T * x = 128) (hT_pos : T > 0) (hx_pos : x > 0) : B = 2 :=
sorry

end number_of_bushes_l2345_234596


namespace safe_unlockable_by_five_l2345_234555

def min_total_keys (num_locks : ℕ) (num_people : ℕ) (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) : ℕ :=
  num_locks * ((num_people + 1) / 2)

theorem safe_unlockable_by_five (num_locks : ℕ) (num_people : ℕ) 
  (key_distribution : (Fin num_locks) → (Fin num_people) → Prop) :
  (∀ (P : Finset (Fin num_people)), P.card = 5 → (∀ k : Fin num_locks, ∃ p ∈ P, key_distribution k p)) →
  min_total_keys num_locks num_people key_distribution = 20 := 
by
  sorry

end safe_unlockable_by_five_l2345_234555


namespace part1_part2_l2345_234574

variable {a : ℝ} (M N : Set ℝ)

theorem part1 (h : a = 1) : M = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

theorem part2 (hM : (M = {x : ℝ | 0 < x ∧ x < a + 1}))
              (hN : N = {x : ℝ | -1 ≤ x ∧ x ≤ 3})
              (h_union : M ∪ N = N) : 
  a ∈ Set.Icc (-1 : ℝ) 2 :=
by
  sorry

end part1_part2_l2345_234574


namespace length_of_first_train_l2345_234533

theorem length_of_first_train
    (speed_train1_kmph : ℝ) (speed_train2_kmph : ℝ) 
    (length_train2_m : ℝ) (cross_time_s : ℝ)
    (conv_factor : ℝ)         -- Conversion factor from kmph to m/s
    (relative_speed_ms : ℝ)   -- Relative speed in m/s 
    (distance_covered_m : ℝ)  -- Total distance covered in meters
    (length_train1_m : ℝ) : Prop :=
  speed_train1_kmph = 120 →
  speed_train2_kmph = 80 →
  length_train2_m = 210.04 →
  cross_time_s = 9 →
  conv_factor = 1000 / 3600 →
  relative_speed_ms = (200 * conv_factor) →
  distance_covered_m = (relative_speed_ms * cross_time_s) →
  length_train1_m = 290 →
  distance_covered_m = length_train1_m + length_train2_m

end length_of_first_train_l2345_234533


namespace wrapping_paper_area_l2345_234547

variable (w h : ℝ)

theorem wrapping_paper_area : ∃ A, A = 4 * (w + h) ^ 2 :=
by
  sorry

end wrapping_paper_area_l2345_234547


namespace rectangle_diagonal_l2345_234506

theorem rectangle_diagonal (l w : ℝ) (hl : l = 40) (hw : w = 40 * Real.sqrt 2) :
  Real.sqrt (l^2 + w^2) = 40 * Real.sqrt 3 :=
by
  rw [hl, hw]
  sorry

end rectangle_diagonal_l2345_234506


namespace optionB_unfactorable_l2345_234537

-- Definitions for the conditions
def optionA (a b : ℝ) : ℝ := -a^2 + b^2
def optionB (x y : ℝ) : ℝ := x^2 + y^2
def optionC (z : ℝ) : ℝ := 49 - z^2
def optionD (m : ℝ) : ℝ := 16 - 25 * m^2

-- The proof statement that option B cannot be factored over the real numbers
theorem optionB_unfactorable (x y : ℝ) : ¬ ∃ (p q : ℝ → ℝ), p x * q y = x^2 + y^2 :=
sorry -- Proof to be filled in

end optionB_unfactorable_l2345_234537


namespace max_area_l2345_234510

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 5
noncomputable def BC : ℝ := 6

theorem max_area (PA PB PC BC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) (hBC : BC = 6) : 
  ∃ (A B C : Type) (area_ABC : ℝ), area_ABC = 19 := 
by 
  sorry

end max_area_l2345_234510


namespace parallel_planes_x_plus_y_l2345_234524

def planes_parallel (x y : ℝ) : Prop :=
  ∃ k : ℝ, (x = -k) ∧ (1 = k * y) ∧ (-2 = (1 / 2) * k)

theorem parallel_planes_x_plus_y (x y : ℝ) (h : planes_parallel x y) : x + y = 15 / 4 :=
sorry

end parallel_planes_x_plus_y_l2345_234524


namespace coin_stack_l2345_234562

def penny_thickness : ℝ := 1.55
def nickel_thickness : ℝ := 1.95
def dime_thickness : ℝ := 1.35
def quarter_thickness : ℝ := 1.75
def stack_height : ℝ := 14

theorem coin_stack (n_penny n_nickel n_dime n_quarter : ℕ) 
  (h : n_penny * penny_thickness + n_nickel * nickel_thickness + n_dime * dime_thickness + n_quarter * quarter_thickness = stack_height) :
  n_penny + n_nickel + n_dime + n_quarter = 8 :=
sorry

end coin_stack_l2345_234562


namespace perfect_square_K_l2345_234548

-- Definitions based on the conditions of the problem
variables (Z K : ℕ)
variables (h1 : 1000 < Z ∧ Z < 5000)
variables (h2 : K > 1)
variables (h3 : Z = K^3)

-- The statement we need to prove
theorem perfect_square_K :
  (∃ K : ℕ, 1000 < K^3 ∧ K^3 < 5000 ∧ K^3 = Z ∧ (∃ a : ℕ, K = a^2)) → K = 16 :=
sorry

end perfect_square_K_l2345_234548


namespace total_oil_volume_l2345_234570

theorem total_oil_volume (total_bottles : ℕ) (bottles_250ml : ℕ) (bottles_300ml : ℕ)
    (volume_250ml : ℕ) (volume_300ml : ℕ) (total_volume_ml : ℚ) 
    (total_volume_l : ℚ) (h1 : total_bottles = 35)
    (h2 : bottles_250ml = 17) (h3 : bottles_300ml = total_bottles - bottles_250ml)
    (h4 : volume_250ml = 250) (h5 : volume_300ml = 300) 
    (h6 : total_volume_ml = bottles_250ml * volume_250ml + bottles_300ml * volume_300ml)
    (h7 : total_volume_l = total_volume_ml / 1000) : 
    total_volume_l = 9.65 := 
by 
  sorry

end total_oil_volume_l2345_234570


namespace craftsman_jars_l2345_234542

theorem craftsman_jars (J P : ℕ) 
  (h1 : J = 2 * P)
  (h2 : 5 * J + 15 * P = 200) : 
  J = 16 := by
  sorry

end craftsman_jars_l2345_234542


namespace complex_sum_identity_l2345_234584

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity_l2345_234584


namespace part1_decreasing_on_pos_part2_t_range_l2345_234508

noncomputable def f (x : ℝ) : ℝ := -x + 2 / x

theorem part1_decreasing_on_pos (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) : 
  f x1 > f x2 := by sorry

theorem part2_t_range (t : ℝ) (ht : ∀ x : ℝ, 1 ≤ x → f x ≤ (1 + t * x) / x) : 
  0 ≤ t := by sorry

end part1_decreasing_on_pos_part2_t_range_l2345_234508


namespace min_blocks_for_wall_l2345_234541

theorem min_blocks_for_wall (len height : ℕ) (blocks : ℕ → ℕ → ℕ)
  (block_1 : ℕ) (block_2 : ℕ) (block_3 : ℕ) :
  len = 120 → height = 9 →
  block_3 = 1 → block_2 = 2 → block_1 = 3 →
  blocks 5 41 + blocks 4 40 = 365 :=
by
  sorry

end min_blocks_for_wall_l2345_234541


namespace total_weight_l2345_234583

def weights (M D C : ℕ): Prop :=
  D = 46 ∧ D + C = 60 ∧ C = M / 5

theorem total_weight (M D C : ℕ) (h : weights M D C) : M + D + C = 130 :=
by
  cases h with
  | intro h1 h2 =>
    cases h2 with
    | intro h2_1 h2_2 => 
      sorry

end total_weight_l2345_234583


namespace sine_cosine_fraction_l2345_234545

theorem sine_cosine_fraction (θ : ℝ) (h : Real.tan θ = 2) : 
    (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
by 
  sorry

end sine_cosine_fraction_l2345_234545


namespace cafeteria_pies_l2345_234572

theorem cafeteria_pies (total_apples handed_out_per_student apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_per_student = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_per_student) / apples_per_pie = 5 := by
  sorry

end cafeteria_pies_l2345_234572


namespace how_many_pens_l2345_234592

theorem how_many_pens
  (total_cost : ℝ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (avg_pen_price : ℝ)
  (total_cost := 510)
  (num_pencils := 75)
  (avg_pencil_price := 2)
  (avg_pen_price := 12)
  : ∃ (num_pens : ℕ), num_pens = 30 :=
by
  sorry

end how_many_pens_l2345_234592


namespace John_pays_more_than_Jane_l2345_234520

theorem John_pays_more_than_Jane : 
  let original_price := 24.00000000000002
  let discount_rate := 0.10
  let tip_rate := 0.15
  let discount := discount_rate * original_price
  let discounted_price := original_price - discount
  let john_tip := tip_rate * original_price
  let jane_tip := tip_rate * discounted_price
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.3600000000000003 :=
by
  sorry

end John_pays_more_than_Jane_l2345_234520


namespace probability_letter_in_mathematics_l2345_234549

/-- 
Given that Lisa picks one letter randomly from the alphabet, 
prove that the probability that Lisa picks a letter in "MATHEMATICS" is 4/13.
-/
theorem probability_letter_in_mathematics :
  (8 : ℚ) / 26 = 4 / 13 :=
by
  sorry

end probability_letter_in_mathematics_l2345_234549


namespace number_problem_l2345_234573

theorem number_problem (x : ℤ) (h1 : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := by
  sorry

end number_problem_l2345_234573


namespace major_axis_length_of_intersecting_ellipse_l2345_234556

theorem major_axis_length_of_intersecting_ellipse (radius : ℝ) (h_radius : radius = 2) 
  (minor_axis_length : ℝ) (h_minor_axis : minor_axis_length = 2 * radius) (major_axis_length : ℝ) 
  (h_major_axis : major_axis_length = minor_axis_length * 1.6) :
  major_axis_length = 6.4 :=
by 
  -- The proof will follow here, but currently it's not required.
  sorry

end major_axis_length_of_intersecting_ellipse_l2345_234556


namespace loisa_savings_l2345_234558

namespace SavingsProof

def cost_cash : ℤ := 450
def down_payment : ℤ := 100
def payment_first_4_months : ℤ := 4 * 40
def payment_next_4_months : ℤ := 4 * 35
def payment_last_4_months : ℤ := 4 * 30

def total_installment_payment : ℤ :=
  down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

theorem loisa_savings :
  (total_installment_payment - cost_cash) = 70 := by
  sorry

end SavingsProof

end loisa_savings_l2345_234558


namespace evaluate_difference_floor_squares_l2345_234505

theorem evaluate_difference_floor_squares (x : ℝ) (h : x = 15.3) : ⌊x^2⌋ - ⌊x⌋^2 = 9 := by
  sorry

end evaluate_difference_floor_squares_l2345_234505


namespace polynomial_consecutive_integers_l2345_234565

theorem polynomial_consecutive_integers (a : ℤ) (c : ℤ) (P : ℤ → ℤ)
  (hP : ∀ x : ℤ, P x = 2 * x ^ 3 - 30 * x ^ 2 + c * x)
  (h_consecutive : ∃ a : ℤ, P (a - 1) + 1 = P a ∧ P a = P (a + 1) - 1) :
  a = 5 ∧ c = 149 :=
by
  sorry

end polynomial_consecutive_integers_l2345_234565


namespace milk_required_for_flour_l2345_234589

theorem milk_required_for_flour (flour_ratio milk_ratio total_flour : ℕ) : 
  (milk_ratio * (total_flour / flour_ratio)) = 160 :=
by
  let milk_ratio := 40
  let flour_ratio := 200
  let total_flour := 800
  exact sorry

end milk_required_for_flour_l2345_234589


namespace num_expr_div_by_10_l2345_234567

theorem num_expr_div_by_10 : (11^11 + 12^12 + 13^13) % 10 = 0 := by
  sorry

end num_expr_div_by_10_l2345_234567


namespace number_of_girls_in_group_l2345_234544

open Finset

/-- Given that a tech group consists of 6 students, and 3 people are to be selected to visit an exhibition,
    if there are at least 1 girl among the selected, the number of different selection methods is 16,
    then the number of girls in the group is 2. -/
theorem number_of_girls_in_group :
  ∃ n : ℕ, (n ≥ 1 ∧ n ≤ 6 ∧ 
            (Nat.choose 6 3 - Nat.choose (6 - n) 3 = 16)) → n = 2 :=
by
  sorry

end number_of_girls_in_group_l2345_234544


namespace slope_range_l2345_234535

theorem slope_range (α : Real) (hα : -1 ≤ Real.cos α ∧ Real.cos α ≤ 1) :
  ∃ k ∈ Set.Icc (- Real.sqrt 3 / 3) (Real.sqrt 3 / 3), ∀ x y : Real, x * Real.cos α - Real.sqrt 3 * y - 2 = 0 → y = k * x - (2 / Real.sqrt 3) :=
by
  sorry

end slope_range_l2345_234535


namespace remainder_of_3_pow_800_mod_17_l2345_234526

theorem remainder_of_3_pow_800_mod_17 :
    (3 ^ 800) % 17 = 1 :=
by
    sorry

end remainder_of_3_pow_800_mod_17_l2345_234526


namespace minimum_sum_l2345_234598

theorem minimum_sum (a b c : ℕ) (h : a * b * c = 3006) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≥ 105 :=
sorry

end minimum_sum_l2345_234598


namespace min_points_game_12_l2345_234532

noncomputable def player_scores := (18, 22, 9, 29)

def avg_after_eleven_games (scores: ℕ × ℕ × ℕ × ℕ) := 
  let s₁ := 78 -- Sum of the points in 8th, 9th, 10th, 11th games
  (s₁: ℕ) / 4

def points_twelve_game_cond (n: ℕ) : Prop :=
  let total_points := 78 + n
  total_points > (20 * 12)

theorem min_points_game_12 (points_in_first_7_games: ℕ) (score_12th_game: ℕ) 
  (H1: avg_after_eleven_games player_scores > (points_in_first_7_games / 7)) 
  (H2: points_twelve_game_cond score_12th_game):
  score_12th_game = 30 := by
  sorry

end min_points_game_12_l2345_234532


namespace solve_for_y_l2345_234590

theorem solve_for_y (y : ℝ) (h : -3 * y - 9 = 6 * y + 3) : y = -4 / 3 :=
by
  sorry

end solve_for_y_l2345_234590


namespace mrs_wong_initial_valentines_l2345_234536

theorem mrs_wong_initial_valentines (x : ℕ) (given left : ℕ) (h_given : given = 8) (h_left : left = 22) (h_initial : x = left + given) : x = 30 :=
by
  rw [h_left, h_given] at h_initial
  exact h_initial

end mrs_wong_initial_valentines_l2345_234536


namespace tracy_initial_candies_l2345_234511

variable (x y : ℕ) (h1 : 2 ≤ y) (h2 : y ≤ 6)

theorem tracy_initial_candies :
  (x - (1/5 : ℚ) * x = (4/5 : ℚ) * x) ∧
  ((4/5 : ℚ) * x - (1/3 : ℚ) * (4/5 : ℚ) * x = (8/15 : ℚ) * x) ∧
  y - 10 * 2 + ((8/15 : ℚ) * x - 20) = 5 →
  x = 60 :=
by
  sorry

end tracy_initial_candies_l2345_234511


namespace toms_total_out_of_pocket_is_680_l2345_234561

namespace HealthCosts

def doctor_visit_cost : ℝ := 300
def cast_cost : ℝ := 200
def initial_insurance_coverage : ℝ := 0.60
def therapy_session_cost : ℝ := 100
def number_of_sessions : ℕ := 8
def therapy_insurance_coverage : ℝ := 0.40

def total_initial_cost : ℝ :=
  doctor_visit_cost + cast_cost

def initial_out_of_pocket : ℝ :=
  total_initial_cost * (1 - initial_insurance_coverage)

def total_therapy_cost : ℝ :=
  therapy_session_cost * number_of_sessions

def therapy_out_of_pocket : ℝ :=
  total_therapy_cost * (1 - therapy_insurance_coverage)

def total_out_of_pocket : ℝ :=
  initial_out_of_pocket + therapy_out_of_pocket

theorem toms_total_out_of_pocket_is_680 :
  total_out_of_pocket = 680 := by
  sorry

end HealthCosts

end toms_total_out_of_pocket_is_680_l2345_234561


namespace train_speed_l2345_234591

theorem train_speed 
  (length_train : ℝ) (length_bridge : ℝ) (time : ℝ) 
  (h_length_train : length_train = 110)
  (h_length_bridge : length_bridge = 138)
  (h_time : time = 12.399008079353651) : 
  (length_train + length_bridge) / time * 3.6 = 72 :=
by
  sorry

end train_speed_l2345_234591


namespace actual_estate_area_l2345_234577

theorem actual_estate_area (map_scale : ℝ) (length_inches : ℝ) (width_inches : ℝ) 
  (actual_length : ℝ) (actual_width : ℝ) (area_square_miles : ℝ) 
  (h_scale : map_scale = 300)
  (h_length : length_inches = 4)
  (h_width : width_inches = 3)
  (h_actual_length : actual_length = length_inches * map_scale)
  (h_actual_width : actual_width = width_inches * map_scale)
  (h_area : area_square_miles = actual_length * actual_width) :
  area_square_miles = 1080000 :=
sorry

end actual_estate_area_l2345_234577


namespace no_intersection_pair_C_l2345_234564

theorem no_intersection_pair_C :
  let y1 := fun x : ℝ => x
  let y2 := fun x : ℝ => x - 3
  ∀ x : ℝ, y1 x ≠ y2 x :=
by
  sorry

end no_intersection_pair_C_l2345_234564


namespace asymptote_of_hyperbola_l2345_234503

theorem asymptote_of_hyperbola (x y : ℝ) (h : (x^2 / 16) - (y^2 / 25) = 1) : 
  y = (5 / 4) * x :=
sorry

end asymptote_of_hyperbola_l2345_234503


namespace total_gallons_of_seed_l2345_234523

-- Condition (1): The area of the football field is 8000 square meters.
def area_football_field : ℝ := 8000

-- Condition (2): Each square meter needs 4 times as much seed as fertilizer.
def seed_to_fertilizer_ratio : ℝ := 4

-- Condition (3): Carson uses 240 gallons of seed and fertilizer combined for every 2000 square meters.
def combined_usage_per_2000sqm : ℝ := 240
def area_unit : ℝ := 2000

-- Target: Prove that the total gallons of seed Carson uses for the entire field is 768 gallons.
theorem total_gallons_of_seed : seed_to_fertilizer_ratio * area_football_field / area_unit / (seed_to_fertilizer_ratio + 1) * combined_usage_per_2000sqm * (area_football_field / area_unit) = 768 :=
sorry

end total_gallons_of_seed_l2345_234523


namespace integral_value_l2345_234507

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions of the problem
def a : ℝ := 2 -- This is derived from the problem condition

-- The main theorem statement
theorem integral_value :
  (∫ x in (0 : ℝ)..a, (Real.exp x + 2 * x)) = Real.exp 2 + 3 := by
  sorry

end integral_value_l2345_234507


namespace sphere_weight_dependence_l2345_234560

theorem sphere_weight_dependence 
  (r1 r2 SA1 SA2 weight1 weight2 : ℝ) 
  (h1 : r1 = 0.15) 
  (h2 : r2 = 2 * r1) 
  (h3 : SA1 = 4 * Real.pi * r1^2) 
  (h4 : SA2 = 4 * Real.pi * r2^2) 
  (h5 : weight1 = 8) 
  (h6 : weight1 / SA1 = weight2 / SA2) : 
  weight2 = 32 :=
by
  sorry

end sphere_weight_dependence_l2345_234560


namespace factorize_expr1_factorize_expr2_l2345_234512

open BigOperators

/-- Given m and n, prove that m^3 n - 9 m n can be factorized as mn(m + 3)(m - 3). -/
theorem factorize_expr1 (m n : ℤ) : m^3 * n - 9 * m * n = n * m * (m + 3) * (m - 3) :=
sorry

/-- Given a, prove that a^3 + a - 2a^2 can be factorized as a(a - 1)^2. -/
theorem factorize_expr2 (a : ℤ) : a^3 + a - 2 * a^2 = a * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l2345_234512


namespace geometric_sequence_constant_l2345_234595

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ) (h1 : q ≠ 1) (h2 : ∀ n, a (n + 1) = q * a n) (c : ℝ) :
  (∀ n, a (n + 1) + c = q * (a n + c)) → c = 0 := sorry

end geometric_sequence_constant_l2345_234595


namespace sum_even_less_100_correct_l2345_234513

-- Define the sequence of even, positive integers less than 100
def even_seq (n : ℕ) : Prop := n % 2 = 0 ∧ 0 < n ∧ n < 100

-- Sum of the first n positive integers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the even, positive integers less than 100
def sum_even_less_100 : ℕ := 2 * sum_n 49

theorem sum_even_less_100_correct : sum_even_less_100 = 2450 := by
  sorry

end sum_even_less_100_correct_l2345_234513


namespace triangle_inequality_sides_l2345_234522

theorem triangle_inequality_sides
  (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  (a + b) * Real.sqrt (a * b) + (a + c) * Real.sqrt (a * c) + (b + c) * Real.sqrt (b * c) ≥ (a + b + c)^2 / 2 := 
by
  sorry

end triangle_inequality_sides_l2345_234522


namespace max_value_of_x_plus_2y_l2345_234515

theorem max_value_of_x_plus_2y {x y : ℝ} (h : |x| + |y| ≤ 1) : (x + 2 * y) ≤ 2 :=
sorry

end max_value_of_x_plus_2y_l2345_234515


namespace eraser_cost_l2345_234597

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end eraser_cost_l2345_234597


namespace kopecks_problem_l2345_234579

theorem kopecks_problem (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b :=
sorry

end kopecks_problem_l2345_234579


namespace non_neg_integer_solutions_for_inequality_l2345_234530

theorem non_neg_integer_solutions_for_inequality :
  {x : ℤ | 5 * x - 1 < 3 * (x + 1) ∧ (1 - x) / 3 ≤ 1 ∧ 0 ≤ x } = {0, 1} := 
by {
  sorry
}

end non_neg_integer_solutions_for_inequality_l2345_234530


namespace complement_intersection_l2345_234521

open Set

variable (A B U : Set ℕ) 

theorem complement_intersection (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) (hU : U = A ∪ B) :
  (U \ A) ∩ B = {4, 5} :=
by sorry

end complement_intersection_l2345_234521


namespace calculate_total_cost_l2345_234569

def cost_of_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def cost_of_non_parallel_sides (l1 l2 ppf : ℕ) : ℕ :=
  l1 * ppf + l2 * ppf

def total_cost (p_l1 p_l2 np_l1 np_l2 ppf np_pf : ℕ) : ℕ :=
  cost_of_parallel_sides p_l1 p_l2 ppf + cost_of_non_parallel_sides np_l1 np_l2 np_pf

theorem calculate_total_cost :
  total_cost 25 37 20 24 48 60 = 5616 :=
by
  -- Assuming the conditions are correctly applied, the statement aims to validate that the calculated
  -- sum of the costs for the specified fence sides equal Rs 5616.
  sorry

end calculate_total_cost_l2345_234569


namespace probability_of_sequential_draws_l2345_234553

theorem probability_of_sequential_draws :
  let total_cards := 52
  let num_fours := 4
  let remaining_after_first_draw := total_cards - 1
  let remaining_after_second_draw := remaining_after_first_draw - 1
  num_fours / total_cards * 1 / remaining_after_first_draw * 1 / remaining_after_second_draw = 1 / 33150 :=
by sorry

end probability_of_sequential_draws_l2345_234553


namespace triangle_perimeter_l2345_234525

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  distance p1 p2 + distance p1 p3 + distance p2 p3

theorem triangle_perimeter :
  let p1 := (1, 4)
  let p2 := (-7, 0)
  let p3 := (1, 0)
  perimeter p1 p2 p3 = 4 * Real.sqrt 5 + 12 :=
by
  sorry

end triangle_perimeter_l2345_234525


namespace smallest_d_l2345_234571

noncomputable def d := 53361

theorem smallest_d :
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧
    10000 * d = (p * q * r) ^ 2 ∧ d = 53361 :=
  by
    sorry

end smallest_d_l2345_234571


namespace even_numbers_count_l2345_234550

theorem even_numbers_count (a b : ℕ) (h1 : 150 < a) (h2 : a % 2 = 0) (h3 : b < 350) (h4 : b % 2 = 0) (h5 : 150 < b) (h6 : a < 350) (h7 : 154 ≤ b) (h8 : a ≤ 152) :
  ∃ n : ℕ, ∀ k : ℕ, k = 99 ↔ 2 * k + 150 = b - a + 2 :=
by
  sorry

end even_numbers_count_l2345_234550


namespace area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l2345_234554

-- 1. Prove that the area enclosed by x = π/2, x = 3π/2, y = 0 and y = cos x is 2
theorem area_enclosed_by_lines_and_curve : 
  ∫ (x : ℝ) in (Real.pi / 2)..(3 * Real.pi / 2), (-Real.cos x) = 2 := sorry

-- 2. Prove that the cylindrical coordinates (sqrt(2), π/4, 1) correspond to Cartesian coordinates (1, 1, 1)
theorem cylindrical_to_cartesian_coordinates :
  let r := Real.sqrt 2
  let θ := Real.pi / 4
  let z := 1
  (r * Real.cos θ, r * Real.sin θ, z) = (1, 1, 1) := sorry

-- 3. Prove that (3 + 2i) / (2 - 3i) - (3 - 2i) / (2 + 3i) = 2i
theorem complex_number_evaluation : 
  ((3 + 2 * Complex.I) / (2 - 3 * Complex.I)) - ((3 - 2 * Complex.I) / (2 + 3 * Complex.I)) = 2 * Complex.I := sorry

-- 4. Prove that the area of triangle AOB with given polar coordinates is 2
theorem area_of_triangle_AOB :
  let A := (2, Real.pi / 6)
  let B := (4, Real.pi / 3)
  let area := 1 / 2 * (2 * 4 * Real.sin (Real.pi / 3 - Real.pi / 6))
  area = 2 := sorry

end area_enclosed_by_lines_and_curve_cylindrical_to_cartesian_coordinates_complex_number_evaluation_area_of_triangle_AOB_l2345_234554


namespace find_same_goldfish_number_l2345_234517

noncomputable def B (n : ℕ) : ℕ := 3 * 4^n
noncomputable def G (n : ℕ) : ℕ := 243 * 3^n

theorem find_same_goldfish_number : ∃ n, B n = G n :=
by sorry

end find_same_goldfish_number_l2345_234517


namespace sum_of_lengths_of_square_sides_l2345_234538

theorem sum_of_lengths_of_square_sides (side_length : ℕ) (h1 : side_length = 9) : 
  (4 * side_length) = 36 :=
by
  -- Here we would normally write the proof
  sorry

end sum_of_lengths_of_square_sides_l2345_234538


namespace polynomial_remainder_l2345_234500

theorem polynomial_remainder (c a b : ℤ) 
  (h1 : (16 * c + 8 * a + 2 * b = -12)) 
  (h2 : (81 * c - 27 * a - 3 * b = -85)) : 
  (a, b, c) = (5, 7, 1) :=
sorry

end polynomial_remainder_l2345_234500


namespace smallest_digit_to_correct_sum_l2345_234575

theorem smallest_digit_to_correct_sum (x y z w : ℕ) (hx : x = 753) (hy : y = 946) (hz : z = 821) (hw : w = 2420) :
  ∃ d, d = 7 ∧ (753 + 946 + 821 - 100 * d = 2420) :=
by sorry

end smallest_digit_to_correct_sum_l2345_234575


namespace find_vector_b_l2345_234576

structure Vec2 where
  x : ℝ
  y : ℝ

def is_parallel (a b : Vec2) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b.x = k * a.x ∧ b.y = k * a.y

def vec_a : Vec2 := { x := 2, y := 3 }
def vec_b : Vec2 := { x := -2, y := -3 }

theorem find_vector_b :
  is_parallel vec_a vec_b := 
sorry

end find_vector_b_l2345_234576


namespace least_number_to_add_l2345_234501

theorem least_number_to_add (a b : ℤ) (d : ℤ) (h : a = 1054) (hb : b = 47) (hd : d = 27) :
  ∃ n : ℤ, (a + d) % b = 0 :=
by
  sorry

end least_number_to_add_l2345_234501


namespace competition_participants_l2345_234502

theorem competition_participants (N : ℕ)
  (h1 : (1 / 12) * N = 18) :
  N = 216 := 
by
  sorry

end competition_participants_l2345_234502


namespace circle_equation_is_correct_l2345_234518

def center : Int × Int := (-3, 4)
def radius : Int := 3
def circle_standard_equation (x y : Int) : Int :=
  (x + 3)^2 + (y - 4)^2

theorem circle_equation_is_correct :
  circle_standard_equation x y = 9 :=
sorry

end circle_equation_is_correct_l2345_234518


namespace negation_of_existence_l2345_234593

theorem negation_of_existence :
  ¬ (∃ (x_0 : ℝ), x_0^2 - x_0 + 1 ≤ 0) ↔ ∀ (x : ℝ), x^2 - x + 1 > 0 :=
by
  sorry

end negation_of_existence_l2345_234593


namespace red_paint_cans_needed_l2345_234529

-- Definitions for the problem
def ratio_red_white : ℚ := 3 / 2
def total_cans : ℕ := 30

-- Theorem statement to prove the number of cans of red paint
theorem red_paint_cans_needed : total_cans * (3 / 5) = 18 := by 
  sorry

end red_paint_cans_needed_l2345_234529


namespace option_b_correct_l2345_234514

variables {m n : Line} {α β : Plane}

-- Define the conditions as per the problem.
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

theorem option_b_correct (h1 : line_perpendicular_to_plane m α)
                         (h2 : line_perpendicular_to_plane n β)
                         (h3 : lines_perpendicular m n) :
                         plane_perpendicular_to_plane α β :=
sorry

end option_b_correct_l2345_234514


namespace increasing_function_greater_at_a_squared_plus_one_l2345_234599

variable (f : ℝ → ℝ) (a : ℝ)

def strictly_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_greater_at_a_squared_plus_one :
  strictly_increasing f → f (a^2 + 1) > f a :=
by
  sorry

end increasing_function_greater_at_a_squared_plus_one_l2345_234599
