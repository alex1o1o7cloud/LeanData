import Mathlib

namespace no_real_roots_of_equation_l1525_152511

theorem no_real_roots_of_equation :
  (∃ x : ℝ, 2 * Real.cos (x / 2) = 10^x + 10^(-x) + 1) -> False :=
by
  sorry

end no_real_roots_of_equation_l1525_152511


namespace force_for_18_inch_wrench_l1525_152528

theorem force_for_18_inch_wrench (F : ℕ → ℕ → ℕ) : 
  (∀ L : ℕ, ∃ k : ℕ, F 300 12 = F (F L k) L) → 
  ((F 12 300) = 3600) → 
  (∀ k : ℕ, F (F 6 k) 6 = 3600) → 
  (∀ k : ℕ, F (F 18 k) 18 = 3600) → 
  (F 18 200 = 3600) :=
by
  sorry

end force_for_18_inch_wrench_l1525_152528


namespace division_quotient_proof_l1525_152520

theorem division_quotient_proof (x : ℕ) (larger_number : ℕ) (h1 : larger_number - x = 1365)
    (h2 : larger_number = 1620) (h3 : larger_number % x = 15) : larger_number / x = 6 :=
by
  sorry

end division_quotient_proof_l1525_152520


namespace distributive_property_l1525_152554

theorem distributive_property (a b : ℝ) : 3 * a * (2 * a - b) = 6 * a^2 - 3 * a * b :=
by
  sorry

end distributive_property_l1525_152554


namespace product_mod_five_remainder_l1525_152580

theorem product_mod_five_remainder :
  (114 * 232 * 454 * 454 * 678) % 5 = 4 := by
  sorry

end product_mod_five_remainder_l1525_152580


namespace compare_two_and_neg_three_l1525_152526

theorem compare_two_and_neg_three (h1 : 2 > 0) (h2 : -3 < 0) : 2 > -3 :=
by
  sorry

end compare_two_and_neg_three_l1525_152526


namespace square_division_possible_l1525_152550

theorem square_division_possible :
  ∃ (S a b c : ℕ), 
    S^2 = a^2 + 3 * b^2 + 5 * c^2 ∧ 
    a = 3 ∧ 
    b = 2 ∧ 
    c = 1 :=
  by {
    sorry
  }

end square_division_possible_l1525_152550


namespace photos_per_album_correct_l1525_152563

-- Define the conditions
def total_photos : ℕ := 4500
def first_batch_photos : ℕ := 1500
def first_batch_albums : ℕ := 30
def second_batch_albums : ℕ := 60
def remaining_photos : ℕ := total_photos - first_batch_photos

-- Define the number of photos per album for the first batch (should be 50)
def photos_per_album_first_batch : ℕ := first_batch_photos / first_batch_albums

-- Define the number of photos per album for the second batch (should be 50)
def photos_per_album_second_batch : ℕ := remaining_photos / second_batch_albums

-- Statement to prove
theorem photos_per_album_correct :
  photos_per_album_first_batch = 50 ∧ photos_per_album_second_batch = 50 :=
by
  simp [photos_per_album_first_batch, photos_per_album_second_batch, remaining_photos]
  sorry

end photos_per_album_correct_l1525_152563


namespace min_value_expr_l1525_152546

theorem min_value_expr (m n : ℝ) (h : m - n^2 = 8) : m^2 - 3 * n^2 + m - 14 ≥ 58 :=
sorry

end min_value_expr_l1525_152546


namespace calculate_discount_l1525_152574

def original_price := 22
def sale_price := 16

theorem calculate_discount : original_price - sale_price = 6 := 
by
  sorry

end calculate_discount_l1525_152574


namespace count_congruent_to_5_mod_7_l1525_152545

theorem count_congruent_to_5_mod_7 (n : ℕ) :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 5) → ∃ count : ℕ, count = 43 := by
  sorry

end count_congruent_to_5_mod_7_l1525_152545


namespace area_of_triangle_A2B2C2_l1525_152543

noncomputable def area_DA1B1 : ℝ := 15 / 4
noncomputable def area_DA1C1 : ℝ := 10
noncomputable def area_DB1C1 : ℝ := 6
noncomputable def area_DA2B2 : ℝ := 40
noncomputable def area_DA2C2 : ℝ := 30
noncomputable def area_DB2C2 : ℝ := 50

theorem area_of_triangle_A2B2C2 : ∃ area : ℝ, 
  area = (50 * Real.sqrt 2) ∧ 
  (area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 ∧ 
  area_DA2C2 = 30 ∧ 
  area_DB2C2 = 50) := 
by
  sorry

end area_of_triangle_A2B2C2_l1525_152543


namespace maria_towels_l1525_152552

-- Define the initial total towels
def initial_total : ℝ := 124.5 + 67.7

-- Define the towels given to her mother
def towels_given : ℝ := 85.35

-- Define the remaining towels (this is what we need to prove)
def towels_remaining : ℝ := 106.85

-- The theorem that states Maria ended up with the correct number of towels
theorem maria_towels :
  initial_total - towels_given = towels_remaining :=
by
  -- Here we would provide the proof, but we use sorry for now
  sorry

end maria_towels_l1525_152552


namespace cookies_left_for_Monica_l1525_152544

-- Definitions based on the conditions
def total_cookies : ℕ := 30
def father_cookies : ℕ := 10
def mother_cookies : ℕ := father_cookies / 2
def brother_cookies : ℕ := mother_cookies + 2

-- Statement for the theorem
theorem cookies_left_for_Monica : total_cookies - (father_cookies + mother_cookies + brother_cookies) = 8 := by
  -- The proof goes here
  sorry

end cookies_left_for_Monica_l1525_152544


namespace inversely_proportional_x_y_l1525_152572

theorem inversely_proportional_x_y (x y c : ℝ) 
  (h1 : x * y = c) (h2 : 8 * 16 = c) : y = -32 → x = -4 :=
by
  sorry

end inversely_proportional_x_y_l1525_152572


namespace average_speed_is_35_l1525_152508

-- Given constants
def distance : ℕ := 210
def speed_difference : ℕ := 5
def time_difference : ℕ := 1

-- Definition of time for planned speed and actual speed
def planned_time (x : ℕ) : ℚ := distance / (x - speed_difference)
def actual_time (x : ℕ) : ℚ := distance / x

-- Main theorem to be proved
theorem average_speed_is_35 (x : ℕ) (h : (planned_time x - actual_time x) = time_difference) : x = 35 :=
sorry

end average_speed_is_35_l1525_152508


namespace determine_x_value_l1525_152570

theorem determine_x_value (a b c x : ℕ) (h1 : x = a + 7) (h2 : a = b + 12) (h3 : b = c + 25) (h4 : c = 95) : x = 139 := by
  sorry

end determine_x_value_l1525_152570


namespace finish_together_in_4_days_l1525_152555

-- Definitions for the individual days taken by A, B, and C
def days_for_A := 12
def days_for_B := 24
def days_for_C := 8 -- C's approximated days

-- The rates are the reciprocals of the days
def rate_A := 1 / days_for_A
def rate_B := 1 / days_for_B
def rate_C := 1 / days_for_C

-- The combined rate of A, B, and C
def combined_rate := rate_A + rate_B + rate_C

-- The total days required to finish the work together
def total_days := 1 / combined_rate

-- Theorem stating that the total days required is 4
theorem finish_together_in_4_days : total_days = 4 := 
by 
-- proof omitted
sorry

end finish_together_in_4_days_l1525_152555


namespace solve_for_x_and_compute_value_l1525_152538

theorem solve_for_x_and_compute_value (x : ℝ) (h : 5 * x - 3 = 15 * x + 15) : 6 * (x + 5) = 19.2 := by
  sorry

end solve_for_x_and_compute_value_l1525_152538


namespace scientific_notation_1_3_billion_l1525_152564

theorem scientific_notation_1_3_billion : 1300000000 = 1.3 * 10^9 := 
sorry

end scientific_notation_1_3_billion_l1525_152564


namespace completing_square_l1525_152568

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l1525_152568


namespace range_of_a_l1525_152581

theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 2)^x₁ > (a - 2)^x₂) → (2 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l1525_152581


namespace proof_problem_l1525_152576

/- Define relevant concepts -/
def is_factor (a b : Nat) := ∃ k, b = a * k
def is_divisor := is_factor

/- Given conditions with their translations -/
def condition_A : Prop := is_factor 5 35
def condition_B : Prop := is_divisor 21 252 ∧ ¬ is_divisor 21 48
def condition_C : Prop := ¬ (is_divisor 15 90 ∨ is_divisor 15 74)
def condition_D : Prop := is_divisor 18 36 ∧ ¬ is_divisor 18 72
def condition_E : Prop := is_factor 9 180

/- The main proof problem statement -/
theorem proof_problem : condition_A ∧ condition_B ∧ ¬ condition_C ∧ ¬ condition_D ∧ condition_E :=
by
  sorry

end proof_problem_l1525_152576


namespace rice_containers_l1525_152562

theorem rice_containers (pound_to_ounce : ℕ) (total_rice_lb : ℚ) (container_oz : ℕ) : 
  pound_to_ounce = 16 → 
  total_rice_lb = 33 / 4 → 
  container_oz = 33 → 
  (total_rice_lb * pound_to_ounce) / container_oz = 4 :=
by sorry

end rice_containers_l1525_152562


namespace cows_milk_production_l1525_152547

variable (p q r s t : ℕ)

theorem cows_milk_production
  (h : p * r > 0)  -- Assuming p and r are positive to avoid division by zero
  (produce : p * r * q ≠ 0) -- Additional assumption to ensure non-zero q
  (h_cows : q = p * r * (q / (p * r))) 
  : s * t * q / (p * r) = s * t * (q / (p * r)) :=
by
  sorry

end cows_milk_production_l1525_152547


namespace piglet_steps_count_l1525_152504

theorem piglet_steps_count (u v L : ℝ) (h₁ : (L * u) / (u + v) = 66) (h₂ : (L * u) / (u - v) = 198) : L = 99 :=
sorry

end piglet_steps_count_l1525_152504


namespace cistern_fill_time_l1525_152532

-- Define the filling rate and emptying rate as given conditions.
def R_fill : ℚ := 1 / 5
def R_empty : ℚ := 1 / 9

-- Define the net rate when both taps are opened simultaneously.
def R_net : ℚ := R_fill - R_empty

-- The total time to fill the cistern when both taps are opened.
def fill_time := 1 / R_net

-- Prove that the total time to fill the cistern is 11.25 hours.
theorem cistern_fill_time : fill_time = 11.25 := 
by 
    -- We include sorry to bypass the actual proof. This will allow the code to compile.
    sorry

end cistern_fill_time_l1525_152532


namespace find_fencing_cost_l1525_152519

theorem find_fencing_cost
  (d : ℝ) (cost_per_meter : ℝ) (π : ℝ)
  (h1 : d = 22)
  (h2 : cost_per_meter = 2.50)
  (hπ : π = Real.pi) :
  (cost_per_meter * (π * d) = 172.80) :=
sorry

end find_fencing_cost_l1525_152519


namespace number_of_balls_selected_is_three_l1525_152509

-- Definitions of conditions
def total_balls : ℕ := 100
def odd_balls_selected : ℕ := 2
def even_balls_selected : ℕ := 1
def probability_first_ball_odd : ℚ := 2 / 3

-- The number of balls selected
def balls_selected := odd_balls_selected + even_balls_selected

-- Statement of the proof problem
theorem number_of_balls_selected_is_three 
(h1 : total_balls = 100)
(h2 : odd_balls_selected = 2)
(h3 : even_balls_selected = 1)
(h4 : probability_first_ball_odd = 2 / 3) :
  balls_selected = 3 :=
sorry

end number_of_balls_selected_is_three_l1525_152509


namespace symmetric_point_exists_l1525_152518

-- Define the point M
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point M
def M : Point3D := { x := 3, y := 3, z := 3 }

-- Define the parametric form of the line
def line (t : ℝ) : Point3D := { x := 1 - t, y := 1.5, z := 3 + t }

-- Define the point M' that we want to prove is symmetrical to M with respect to the line
def symmPoint : Point3D := { x := 1, y := 0, z := 1 }

-- The theorem that we need to prove, ensuring M' is symmetrical to M with respect to the given line
theorem symmetric_point_exists : ∃ t, line t = symmPoint ∧ 
  (∀ M_0 : Point3D, M_0.x = (M.x + symmPoint.x) / 2 ∧ M_0.y = (M.y + symmPoint.y) / 2 ∧ M_0.z = (M.z + symmPoint.z) / 2)
  → line t = M_0
  → M_0 = { x := 2, y := 1.5, z := 2 } := 
by
  sorry

end symmetric_point_exists_l1525_152518


namespace problem1_problem2_l1525_152527

variables {a b : ℝ}

-- Given conditions
def condition1 : a + b = 2 := sorry
def condition2 : a * b = -1 := sorry

-- Proof for a^2 + b^2 = 6
theorem problem1 (h1 : a + b = 2) (h2 : a * b = -1) : a^2 + b^2 = 6 :=
by sorry

-- Proof for (a - b)^2 = 8
theorem problem2 (h1 : a + b = 2) (h2 : a * b = -1) : (a - b)^2 = 8 :=
by sorry

end problem1_problem2_l1525_152527


namespace relationship_y_values_l1525_152539

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end relationship_y_values_l1525_152539


namespace remainder_of_M_mod_210_l1525_152558

def M : ℤ := 1234567891011

theorem remainder_of_M_mod_210 :
  (M % 210) = 31 :=
by
  have modulus1 : M % 6 = 3 := by sorry
  have modulus2 : M % 5 = 1 := by sorry
  have modulus3 : M % 7 = 2 := by sorry
  -- Using Chinese Remainder Theorem
  sorry

end remainder_of_M_mod_210_l1525_152558


namespace remainder_when_divided_by_6_l1525_152515

theorem remainder_when_divided_by_6 (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 :=
by sorry

end remainder_when_divided_by_6_l1525_152515


namespace molecular_weight_correct_l1525_152510

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_N_in_N2O3 : ℕ := 2
def num_O_in_N2O3 : ℕ := 3

def molecular_weight_N2O3 : ℝ :=
  (num_N_in_N2O3 * atomic_weight_N) + (num_O_in_N2O3 * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_N2O3 = 76.02 := by
  sorry

end molecular_weight_correct_l1525_152510


namespace arithmetic_sequence_zero_term_l1525_152587

theorem arithmetic_sequence_zero_term (a : ℕ → ℤ) (d : ℤ) (h : d ≠ 0) 
  (h_seq : ∀ n, a n = a 1 + (n-1) * d)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  ∃ n, a n = 0 ∧ n = 5 :=
by { sorry }

end arithmetic_sequence_zero_term_l1525_152587


namespace necessary_and_sufficient_condition_l1525_152522

variables {f g : ℝ → ℝ}

theorem necessary_and_sufficient_condition (f g : ℝ → ℝ)
  (hdom : ∀ x : ℝ, true)
  (hst : ∀ y : ℝ, true) :
  (∀ x : ℝ, f x > g x) ↔ (∀ x : ℝ, ¬ (x ∈ {x : ℝ | f x ≤ g x})) :=
by sorry

end necessary_and_sufficient_condition_l1525_152522


namespace total_earnings_first_three_months_l1525_152590

-- Definitions
def earning_first_month : ℕ := 350
def earning_second_month : ℕ := 2 * earning_first_month + 50
def earning_third_month : ℕ := 4 * (earning_first_month + earning_second_month)

-- Question restated as a theorem
theorem total_earnings_first_three_months : 
  (earning_first_month + earning_second_month + earning_third_month = 5500) :=
by 
  -- Placeholder for the proof
  sorry

end total_earnings_first_three_months_l1525_152590


namespace slope_angle_of_line_l1525_152536

theorem slope_angle_of_line (x y : ℝ) (θ : ℝ) : (x - y + 3 = 0) → θ = 45 := 
sorry

end slope_angle_of_line_l1525_152536


namespace at_least_one_shooter_hits_target_l1525_152566

-- Definition stating the probability of the first shooter hitting the target
def prob_A1 : ℝ := 0.7

-- Definition stating the probability of the second shooter hitting the target
def prob_A2 : ℝ := 0.8

-- The event that at least one shooter hits the target
def prob_at_least_one_hit : ℝ := prob_A1 + prob_A2 - (prob_A1 * prob_A2)

-- Prove that the probability that at least one shooter hits the target is 0.94
theorem at_least_one_shooter_hits_target : prob_at_least_one_hit = 0.94 :=
by
  sorry

end at_least_one_shooter_hits_target_l1525_152566


namespace no_transform_to_1998_power_7_l1525_152560

theorem no_transform_to_1998_power_7 :
  ∀ n : ℕ, (exists m : ℕ, n = 7^m) ->
  ∀ k : ℕ, n = 10 * k + (n % 10) ->
  ¬ (∃ t : ℕ, (t = (1998 ^ 7))) := 
by sorry

end no_transform_to_1998_power_7_l1525_152560


namespace GCF_LCM_example_l1525_152502

/-- Greatest Common Factor (GCF) definition -/
def GCF (a b : ℕ) : ℕ := a.gcd b

/-- Least Common Multiple (LCM) definition -/
def LCM (a b : ℕ) : ℕ := a.lcm b

/-- Main theorem statement to prove -/
theorem GCF_LCM_example : 
  GCF (LCM 9 21) (LCM 8 15) = 3 := by
  sorry

end GCF_LCM_example_l1525_152502


namespace exists_nat_a_b_l1525_152512

theorem exists_nat_a_b (n : ℕ) (hn : 0 < n) : 
∃ a b : ℕ, 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n :=
by
  -- The proof steps would be filled here.
  sorry

end exists_nat_a_b_l1525_152512


namespace linear_system_reduction_transformation_l1525_152591

theorem linear_system_reduction_transformation :
  ∀ (use_substitution_or_elimination : Bool), 
    (use_substitution_or_elimination = true) ∨ (use_substitution_or_elimination = false) → 
    "Reduction and transformation" = "Reduction and transformation" :=
by
  intro use_substitution_or_elimination h
  sorry

end linear_system_reduction_transformation_l1525_152591


namespace possible_values_of_p_l1525_152579

theorem possible_values_of_p (a b c : ℝ) (h₁ : (-a + b + c) / a = (a - b + c) / b)
  (h₂ : (a - b + c) / b = (a + b - c) / c) :
  ∃ p ∈ ({-1, 8} : Set ℝ), p = (a + b) * (b + c) * (c + a) / (a * b * c) :=
by sorry

end possible_values_of_p_l1525_152579


namespace triangle_sides_l1525_152506

noncomputable def sides (a b c : ℝ) : Prop :=
  (a = Real.sqrt (427 / 3)) ∧
  (b = Real.sqrt (427 / 3) + 3/2) ∧
  (c = Real.sqrt (427 / 3) - 3/2)

theorem triangle_sides (a b c : ℝ) (h1 : b - c = 3) (h2 : ∃ d : ℝ, d = 10)
  (h3 : ∃ BD CD : ℝ, CD - BD = 12 ∧ BD + CD = a ∧ 
    a = 2 * (BD + 12 / 2)) :
  sides a b c :=
  sorry

end triangle_sides_l1525_152506


namespace birds_in_sky_l1525_152586

theorem birds_in_sky (wings total_wings : ℕ) (h1 : total_wings = 26) (h2 : wings = 2) : total_wings / wings = 13 := 
by
  sorry

end birds_in_sky_l1525_152586


namespace prove_a_5_l1525_152514

noncomputable def a_5_proof : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n > 0) → 
    (a 1 + 2 * a 2 = 4) →
    ((a 1)^2 * q^6 = 4 * a 1 * q^2 * a 1 * q^6) →
    a 5 = 1 / 8

theorem prove_a_5 : a_5_proof := sorry

end prove_a_5_l1525_152514


namespace expression_value_at_2_l1525_152535

theorem expression_value_at_2 : (2^2 + 3 * 2 - 4) = 6 :=
by 
  sorry

end expression_value_at_2_l1525_152535


namespace option_D_is_linear_equation_with_two_variables_l1525_152529

def is_linear_equation (eq : String) : Prop :=
  match eq with
  | "3x - 6 = x" => false
  | "x = 5 / y - 1" => false
  | "2x - 3y = x^2" => false
  | "3x = 2y" => true
  | _ => false

theorem option_D_is_linear_equation_with_two_variables :
  is_linear_equation "3x = 2y" = true := by
  sorry

end option_D_is_linear_equation_with_two_variables_l1525_152529


namespace intersection_equiv_l1525_152505

open Set

def A : Set ℝ := { x | 2 * x < 2 + x }
def B : Set ℝ := { x | 5 - x > 8 - 4 * x }

theorem intersection_equiv : A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } := 
by 
  sorry

end intersection_equiv_l1525_152505


namespace parametric_to_standard_l1525_152571

theorem parametric_to_standard (θ : ℝ) (x y : ℝ)
  (h1 : x = 1 + 2 * Real.cos θ)
  (h2 : y = 2 * Real.sin θ) :
  (x - 1)^2 + y^2 = 4 := 
sorry

end parametric_to_standard_l1525_152571


namespace crayons_selection_l1525_152534

theorem crayons_selection : 
  ∃ (n : ℕ), n = Nat.choose 14 4 ∧ n = 1001 := by
  sorry

end crayons_selection_l1525_152534


namespace alice_cranes_ratio_alice_cranes_l1525_152583

theorem alice_cranes {A : ℕ} (h1 : A + (1/5 : ℝ) * (1000 - A) + 400 = 1000) :
  A = 500 := by
  sorry

theorem ratio_alice_cranes :
  (500 : ℝ) / 1000 = 1 / 2 := by
  sorry

end alice_cranes_ratio_alice_cranes_l1525_152583


namespace book_width_l1525_152597

noncomputable def golden_ratio : Real := (1 + Real.sqrt 5) / 2

theorem book_width (length : Real) (width : Real) 
(h1 : length = 20) 
(h2 : width / length = golden_ratio) : 
width = 12.36 := 
by 
  sorry

end book_width_l1525_152597


namespace inequality_proof_l1525_152565

variable (a b c : ℝ)

theorem inequality_proof :
  1 < (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end inequality_proof_l1525_152565


namespace folder_cost_calc_l1525_152588

noncomputable def pencil_cost : ℚ := 0.5
noncomputable def dozen_pencils : ℕ := 24
noncomputable def num_folders : ℕ := 20
noncomputable def total_cost : ℚ := 30
noncomputable def total_pencil_cost : ℚ := dozen_pencils * pencil_cost
noncomputable def remaining_cost := total_cost - total_pencil_cost
noncomputable def folder_cost := remaining_cost / num_folders

theorem folder_cost_calc : folder_cost = 0.9 := by
  -- Definitions
  have pencil_cost_def : pencil_cost = 0.5 := rfl
  have dozen_pencils_def : dozen_pencils = 24 := rfl
  have num_folders_def : num_folders = 20 := rfl
  have total_cost_def : total_cost = 30 := rfl
  have total_pencil_cost_def : total_pencil_cost = dozen_pencils * pencil_cost := rfl
  have remaining_cost_def : remaining_cost = total_cost - total_pencil_cost := rfl
  have folder_cost_def : folder_cost = remaining_cost / num_folders := rfl

  -- Calculation steps given conditions
  sorry

end folder_cost_calc_l1525_152588


namespace find_other_root_of_quadratic_l1525_152503

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l1525_152503


namespace sum_of_two_primes_l1525_152533

theorem sum_of_two_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 93) : p * q = 178 := 
sorry

end sum_of_two_primes_l1525_152533


namespace original_solution_concentration_l1525_152589

variable (C : ℝ) -- Concentration of the original solution as a percentage.
variable (v_orig : ℝ := 12) -- 12 ounces of the original vinegar solution.
variable (w_added : ℝ := 50) -- 50 ounces of water added.
variable (v_final_pct : ℝ := 7) -- Final concentration of 7%.

theorem original_solution_concentration :
  (C / 100 * v_orig = v_final_pct / 100 * (v_orig + w_added)) →
  C = (v_final_pct * (v_orig + w_added)) / v_orig :=
sorry

end original_solution_concentration_l1525_152589


namespace possible_to_divide_into_two_groups_l1525_152573

-- Define a type for People
universe u
variable {Person : Type u}

-- Define friend and enemy relations (assume they are given as functions)
variable (friend enemy : Person → Person)

-- Define the main statement
theorem possible_to_divide_into_two_groups (h_friend : ∀ p : Person, ∃ q : Person, friend p = q)
                                           (h_enemy : ∀ p : Person, ∃ q : Person, enemy p = q) :
  ∃ (company : Person → Bool),
    ∀ p : Person, company p ≠ company (friend p) ∧ company p ≠ company (enemy p) :=
by
  sorry

end possible_to_divide_into_two_groups_l1525_152573


namespace min_blue_eyes_with_lunchbox_l1525_152549

theorem min_blue_eyes_with_lunchbox (B L : Finset Nat) (hB : B.card = 15) (hL : L.card = 25) (students : Finset Nat) (hst : students.card = 35)  : 
  ∃ (x : Finset Nat), x ⊆ B ∧ x ⊆ L ∧ x.card ≥ 5 :=
by
  sorry

end min_blue_eyes_with_lunchbox_l1525_152549


namespace range_of_a_l1525_152507

def discriminant (a : ℝ) : ℝ := 4 * a^2 - 16
def P (a : ℝ) : Prop := discriminant a < 0
def Q (a : ℝ) : Prop := 5 - 2 * a > 1

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a ≤ -2 := by
  sorry

end range_of_a_l1525_152507


namespace polynomial_square_b_value_l1525_152585

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l1525_152585


namespace cone_radius_l1525_152530

theorem cone_radius (CSA : ℝ) (l : ℝ) (r : ℝ) (h_CSA : CSA = 989.6016858807849) (h_l : l = 15) :
    r = 21 :=
by
  sorry

end cone_radius_l1525_152530


namespace calories_in_200_grams_is_137_l1525_152525

-- Define the grams of ingredients used.
def lemon_juice_grams := 100
def sugar_grams := 100
def water_grams := 400

-- Define the calories per 100 grams of each ingredient.
def lemon_juice_calories_per_100_grams := 25
def sugar_calories_per_100_grams := 386
def water_calories_per_100_grams := 0

-- Calculate the total calories in the entire lemonade mixture.
def total_calories : Nat :=
  (lemon_juice_grams * lemon_juice_calories_per_100_grams / 100) + 
  (sugar_grams * sugar_calories_per_100_grams / 100) +
  (water_grams * water_calories_per_100_grams / 100)

-- Calculate the total weight of the lemonade mixture.
def total_weight : Nat := lemon_juice_grams + sugar_grams + water_grams

-- Calculate the caloric density (calories per gram).
def caloric_density := total_calories / total_weight

-- Calculate the calories in 200 grams of lemonade.
def calories_in_200_grams := (caloric_density * 200)

-- The theorem to prove
theorem calories_in_200_grams_is_137 : calories_in_200_grams = 137 :=
by sorry

end calories_in_200_grams_is_137_l1525_152525


namespace find_multiple_of_t_l1525_152537

variable (t : ℝ)
variable (x y : ℝ)

theorem find_multiple_of_t (h1 : x = 1 - 4 * t)
  (h2 : ∃ m : ℝ, y = m * t - 2)
  (h3 : t = 0.5)
  (h4 : x = y) : ∃ m : ℝ, (m = 2) :=
by
  sorry

end find_multiple_of_t_l1525_152537


namespace carter_family_children_l1525_152516

variable (f m x y : ℕ)

theorem carter_family_children 
  (avg_family : (3 * y + m + x * y) / (2 + x) = 25)
  (avg_mother_children : (m + x * y) / (1 + x) = 18)
  (father_age : f = 3 * y)
  (simplest_case : y = x) :
  x = 8 :=
by
  -- Proof to be provided
  sorry

end carter_family_children_l1525_152516


namespace fraction_of_roll_used_l1525_152596

theorem fraction_of_roll_used 
  (x : ℚ) 
  (h1 : 3 * x + 3 * x + x + 2 * x = 9 * x)
  (h2 : 9 * x = (2 / 5)) : 
  x = 2 / 45 :=
by
  sorry

end fraction_of_roll_used_l1525_152596


namespace polygon_sides_eq_seven_l1525_152541

theorem polygon_sides_eq_seven (n d : ℕ) (h1 : d = (n * (n - 3)) / 2) (h2 : d = 2 * n) : n = 7 := 
by
  sorry

end polygon_sides_eq_seven_l1525_152541


namespace factorize_square_difference_l1525_152517

theorem factorize_square_difference (x: ℝ):
  x^2 - 4 = (x + 2) * (x - 2) := by
  -- Using the difference of squares formula a^2 - b^2 = (a + b)(a - b)
  sorry

end factorize_square_difference_l1525_152517


namespace rectangle_area_l1525_152513

theorem rectangle_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end rectangle_area_l1525_152513


namespace find_a_if_odd_f_monotonically_increasing_on_pos_l1525_152561

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - 1) / (x + a)

-- Part 1: Proving that a = 0
theorem find_a_if_odd : (∀ x : ℝ, f x a = -f (-x) a) → a = 0 := by sorry

-- Part 2: Proving that f(x) is monotonically increasing on (0, +∞) given a = 0
theorem f_monotonically_increasing_on_pos : (∀ x : ℝ, x > 0 → 
  ∃ y : ℝ, y > 0 ∧ f x 0 < f y 0) := by sorry

end find_a_if_odd_f_monotonically_increasing_on_pos_l1525_152561


namespace election_total_votes_l1525_152557

theorem election_total_votes
  (total_votes : ℕ)
  (votes_A : ℕ)
  (votes_B : ℕ)
  (votes_C : ℕ)
  (h1 : votes_A = 55 * total_votes / 100)
  (h2 : votes_B = 35 * total_votes / 100)
  (h3 : votes_C = total_votes - votes_A - votes_B)
  (h4 : votes_A = votes_B + 400) :
  total_votes = 2000 := by
  sorry

end election_total_votes_l1525_152557


namespace units_digit_7_pow_1023_l1525_152578

-- Define a function for the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_7_pow_1023 :
  units_digit (7 ^ 1023) = 3 :=
by
  sorry

end units_digit_7_pow_1023_l1525_152578


namespace ribbon_length_per_gift_l1525_152582

theorem ribbon_length_per_gift (gifts : ℕ) (initial_ribbon remaining_ribbon : ℝ) (total_used_ribbon : ℝ) (length_per_gift : ℝ):
  gifts = 8 →
  initial_ribbon = 15 →
  remaining_ribbon = 3 →
  total_used_ribbon = initial_ribbon - remaining_ribbon →
  length_per_gift = total_used_ribbon / gifts →
  length_per_gift = 1.5 :=
by
  intros
  sorry

end ribbon_length_per_gift_l1525_152582


namespace inequality_solution_l1525_152599

def solution_set_inequality : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem inequality_solution (x : ℝ) : 
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x ∈ solution_set_inequality :=
by 
  sorry

end inequality_solution_l1525_152599


namespace total_savings_calculation_l1525_152524

theorem total_savings_calculation
  (income : ℕ)
  (ratio_income_to_expenditure : ℕ)
  (ratio_expenditure_to_income : ℕ)
  (tax_rate : ℚ)
  (investment_rate : ℚ)
  (expenditure : ℕ)
  (taxes : ℚ)
  (investments : ℚ)
  (total_savings : ℚ)
  (h_income : income = 17000)
  (h_ratio : ratio_income_to_expenditure / ratio_expenditure_to_income = 5 / 4)
  (h_tax_rate : tax_rate = 0.15)
  (h_investment_rate : investment_rate = 0.1)
  (h_expenditure : expenditure = (income / 5) * 4)
  (h_taxes : taxes = 0.15 * income)
  (h_investments : investments = 0.1 * income)
  (h_total_savings : total_savings = income - (expenditure + taxes + investments)) :
  total_savings = 900 :=
by
  sorry

end total_savings_calculation_l1525_152524


namespace equivalent_resistance_A_B_l1525_152500

-- Parameters and conditions
def resistor_value : ℝ := 5 -- in MΩ
def num_resistors : ℕ := 4
def has_bridging_wire : Prop := true
def negligible_wire_resistance : Prop := true

-- Problem: Prove the equivalent resistance (R_eff) between points A and B is 5 MΩ.
theorem equivalent_resistance_A_B : 
  ∀ (R : ℝ) (n : ℕ) (bridge : Prop) (negligible_wire : Prop),
    R = 5 → n = 4 → bridge → negligible_wire → R = 5 :=
by sorry

end equivalent_resistance_A_B_l1525_152500


namespace sum_of_roots_is_zero_l1525_152569

variables {R : Type*} [Field R] {a b c p q : R}

theorem sum_of_roots_is_zero (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a^3 + p * a + q = 0) (h₅ : b^3 + p * b + q = 0) (h₆ : c^3 + p * c + q = 0) :
  a + b + c = 0 :=
by
  sorry

end sum_of_roots_is_zero_l1525_152569


namespace median_possible_values_l1525_152501

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values_l1525_152501


namespace min_value_expression_l1525_152521

variable (a b m n : ℝ)

-- Conditions: a, b, m, n are positive, a + b = 1, mn = 2
def conditions (a b m n : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n ∧ a + b = 1 ∧ m * n = 2

-- Statement to prove: The minimum value of (am + bn) * (bm + an) is 2
theorem min_value_expression (a b m n : ℝ) (h : conditions a b m n) : 
  ∃ c : ℝ, c = 2 ∧ (∀ (x y z w : ℝ), conditions x y z w → (x * z + y * w) * (y * z + x * w) ≥ c) :=
by
  sorry

end min_value_expression_l1525_152521


namespace buses_required_is_12_l1525_152556

-- Define the conditions given in the problem
def students : ℕ := 535
def bus_capacity : ℕ := 45

-- Define the minimum number of buses required
def buses_needed (students : ℕ) (bus_capacity : ℕ) : ℕ :=
  (students + bus_capacity - 1) / bus_capacity

-- The theorem stating the number of buses required is 12
theorem buses_required_is_12 :
  buses_needed students bus_capacity = 12 :=
sorry

end buses_required_is_12_l1525_152556


namespace parallelogram_to_rhombus_l1525_152592

theorem parallelogram_to_rhombus {a b m1 m2 x : ℝ} (h_area : a * m1 = x * m2) (h_proportion : b / m1 = x / m2) : x = Real.sqrt (a * b) :=
by
  -- Proof goes here
  sorry

end parallelogram_to_rhombus_l1525_152592


namespace percentage_of_hundred_l1525_152593

theorem percentage_of_hundred : (30 / 100) * 100 = 30 := 
by
  sorry

end percentage_of_hundred_l1525_152593


namespace basketball_game_l1525_152584

theorem basketball_game 
    (a b x : ℕ)
    (h1 : 3 * b = 2 * a)
    (h2 : x = 2 * b)
    (h3 : 2 * a + 3 * b + x = 72) : 
    x = 18 :=
sorry

end basketball_game_l1525_152584


namespace sum_of_volumes_is_correct_l1525_152559

-- Define the dimensions of the base of the tank
def tank_base_length : ℝ := 44
def tank_base_width : ℝ := 35

-- Define the increase in water height when the train and the car are submerged
def train_water_height_increase : ℝ := 7
def car_water_height_increase : ℝ := 3

-- Calculate the area of the base of the tank
def base_area : ℝ := tank_base_length * tank_base_width

-- Calculate the volumes of the toy train and the toy car
def volume_train : ℝ := base_area * train_water_height_increase
def volume_car : ℝ := base_area * car_water_height_increase

-- Theorem to prove the sum of the volumes is 15400 cubic centimeters
theorem sum_of_volumes_is_correct : volume_train + volume_car = 15400 := by
  sorry

end sum_of_volumes_is_correct_l1525_152559


namespace initial_weight_of_beef_l1525_152540

theorem initial_weight_of_beef (W : ℝ) 
  (stage1 : W' = 0.70 * W) 
  (stage2 : W'' = 0.80 * W') 
  (stage3 : W''' = 0.50 * W'') 
  (final_weight : W''' = 315) : 
  W = 1125 := by 
  sorry

end initial_weight_of_beef_l1525_152540


namespace meaningful_expression_range_l1525_152553

theorem meaningful_expression_range (x : ℝ) : (¬ (x - 1 = 0)) ↔ (x ≠ 1) := 
by
  sorry

end meaningful_expression_range_l1525_152553


namespace smallest_result_l1525_152575

theorem smallest_result :
  let a := (-2)^3
  let b := (-2) + 3
  let c := (-2) * 3
  let d := (-2) - 3
  a < b ∧ a < c ∧ a < d :=
by
  -- Lean proof steps would go here
  sorry

end smallest_result_l1525_152575


namespace master_parts_per_hour_l1525_152598

variable (x : ℝ)

theorem master_parts_per_hour (h1 : 300 / x = 100 / (40 - x)) : 300 / x = 100 / (40 - x) :=
sorry

end master_parts_per_hour_l1525_152598


namespace milburg_population_l1525_152577

theorem milburg_population 
    (adults : ℕ := 5256) 
    (children : ℕ := 2987) 
    (teenagers : ℕ := 1709) 
    (seniors : ℕ := 2340) : 
    adults + children + teenagers + seniors = 12292 := 
by 
  sorry

end milburg_population_l1525_152577


namespace exists_k_composite_l1525_152567

theorem exists_k_composite (h : Nat) : ∃ k : ℕ, ∀ n : ℕ, 0 < n → ∃ p : ℕ, Prime p ∧ p ∣ (k * 2 ^ n + 1) :=
by
  sorry

end exists_k_composite_l1525_152567


namespace quadrilateral_area_24_l1525_152523

open Classical

noncomputable def quad_area (a b : ℤ) (h : a > b ∧ b > 0) : ℤ :=
let P := (a, b)
let Q := (2*b, a)
let R := (-a, -b)
let S := (-2*b, -a)
-- The proved area
24

theorem quadrilateral_area_24 (a b : ℤ) (h : a > b ∧ b > 0) :
  quad_area a b h = 24 :=
sorry

end quadrilateral_area_24_l1525_152523


namespace calculate_expression_l1525_152551

theorem calculate_expression :
  2^3 - (Real.tan (Real.pi / 3))^2 = 5 := by
  sorry

end calculate_expression_l1525_152551


namespace cups_per_larger_crust_l1525_152542

theorem cups_per_larger_crust
  (initial_crusts : ℕ)
  (initial_flour : ℚ)
  (new_crusts : ℕ)
  (constant_flour : ℚ)
  (h1 : initial_crusts * (initial_flour / initial_crusts) = initial_flour )
  (h2 : new_crusts * (constant_flour / new_crusts) = constant_flour )
  (h3 : initial_flour = constant_flour)
  : (constant_flour / new_crusts) = (8 / 10) :=
by 
  sorry

end cups_per_larger_crust_l1525_152542


namespace exposed_surface_area_hemisphere_l1525_152531

-- Given conditions
def radius : ℝ := 10
def height_above_liquid : ℝ := 5

-- The attempt to state the problem as a proposition
theorem exposed_surface_area_hemisphere : 
  (π * radius ^ 2) + (π * radius * height_above_liquid) = 200 * π :=
by
  sorry

end exposed_surface_area_hemisphere_l1525_152531


namespace solve_problem_l1525_152548

theorem solve_problem (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
    (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end solve_problem_l1525_152548


namespace at_least_one_zero_l1525_152595

theorem at_least_one_zero (a b : ℤ) : (¬ (a ≠ 0) ∨ ¬ (b ≠ 0)) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end at_least_one_zero_l1525_152595


namespace stephen_speed_l1525_152594

theorem stephen_speed (v : ℝ) 
  (time : ℝ := 0.25)
  (speed_second_third : ℝ := 12)
  (speed_last_third : ℝ := 20)
  (total_distance : ℝ := 12) :
  (v * time + speed_second_third * time + speed_last_third * time = total_distance) → 
  v = 16 :=
by
  intro h
  -- introducing the condition h: v * 0.25 + 3 + 5 = 12
  sorry

end stephen_speed_l1525_152594
