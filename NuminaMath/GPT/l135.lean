import Mathlib

namespace NUMINAMATH_GPT_find_f_of_16_l135_13584

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem find_f_of_16 : (∃ a : ℝ, f 2 a = Real.sqrt 2) → f 16 (1/2) = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_f_of_16_l135_13584


namespace NUMINAMATH_GPT_pow_100_mod_18_l135_13589

theorem pow_100_mod_18 : (5 ^ 100) % 18 = 13 := by
  -- Define the conditions
  have h1 : (5 ^ 1) % 18 = 5 := by norm_num
  have h2 : (5 ^ 2) % 18 = 7 := by norm_num
  have h3 : (5 ^ 3) % 18 = 17 := by norm_num
  have h4 : (5 ^ 4) % 18 = 13 := by norm_num
  have h5 : (5 ^ 5) % 18 = 11 := by norm_num
  have h6 : (5 ^ 6) % 18 = 1 := by norm_num
  
  -- The required theorem is based on the conditions mentioned
  sorry

end NUMINAMATH_GPT_pow_100_mod_18_l135_13589


namespace NUMINAMATH_GPT_total_money_needed_l135_13572

-- Declare John's initial amount
def john_has : ℝ := 0.75

-- Declare the additional amount John needs
def john_needs_more : ℝ := 1.75

-- The theorem statement that John needs a total of $2.50
theorem total_money_needed : john_has + john_needs_more = 2.5 :=
  by
  sorry

end NUMINAMATH_GPT_total_money_needed_l135_13572


namespace NUMINAMATH_GPT_bob_total_calories_l135_13553

def total_calories (slices_300 : ℕ) (calories_300 : ℕ) (slices_400 : ℕ) (calories_400 : ℕ) : ℕ :=
  slices_300 * calories_300 + slices_400 * calories_400

theorem bob_total_calories 
  (slices_300 : ℕ := 3)
  (calories_300 : ℕ := 300)
  (slices_400 : ℕ := 4)
  (calories_400 : ℕ := 400) :
  total_calories slices_300 calories_300 slices_400 calories_400 = 2500 := 
by 
  sorry

end NUMINAMATH_GPT_bob_total_calories_l135_13553


namespace NUMINAMATH_GPT_number_of_divisors_of_square_l135_13517

theorem number_of_divisors_of_square {n : ℕ} (h : ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ n = p * q) : Nat.totient (n^2) = 9 :=
sorry

end NUMINAMATH_GPT_number_of_divisors_of_square_l135_13517


namespace NUMINAMATH_GPT_population_growth_l135_13540

theorem population_growth (P : ℝ) (x : ℝ) (y : ℝ) 
  (h₁ : P = 5.48) 
  (h₂ : y = P * (1 + x / 100)^8) : 
  y = 5.48 * (1 + x / 100)^8 := 
by
  sorry

end NUMINAMATH_GPT_population_growth_l135_13540


namespace NUMINAMATH_GPT_find_f6_l135_13580

-- Define the function f
variable {f : ℝ → ℝ}
-- The function satisfies f(x + y) = f(x) + f(y) for all real numbers x and y
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
-- f(4) = 6
axiom f_of_4 : f 4 = 6

theorem find_f6 : f 6 = 9 :=
by
    sorry

end NUMINAMATH_GPT_find_f6_l135_13580


namespace NUMINAMATH_GPT_total_sum_of_money_l135_13511

theorem total_sum_of_money (x : ℝ) (A B C : ℝ) 
  (hA : A = x) 
  (hB : B = 0.65 * x) 
  (hC : C = 0.40 * x) 
  (hC_share : C = 32) :
  A + B + C = 164 := 
  sorry

end NUMINAMATH_GPT_total_sum_of_money_l135_13511


namespace NUMINAMATH_GPT_x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l135_13506

theorem x_less_than_2_necessary_not_sufficient (x : ℝ) :
  (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) := sorry

theorem x_less_than_2_is_necessary_not_sufficient : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧ 
  (¬ ∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) := sorry

end NUMINAMATH_GPT_x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l135_13506


namespace NUMINAMATH_GPT_sulfuric_acid_reaction_l135_13599

theorem sulfuric_acid_reaction (SO₃ H₂O H₂SO₄ : ℕ) 
  (reaction : SO₃ + H₂O = H₂SO₄)
  (H₂O_eq : H₂O = 2)
  (H₂SO₄_eq : H₂SO₄ = 2) :
  SO₃ = 2 :=
by
  sorry

end NUMINAMATH_GPT_sulfuric_acid_reaction_l135_13599


namespace NUMINAMATH_GPT_find_p_over_q_l135_13536

variables (x y p q : ℚ)

theorem find_p_over_q (h1 : (7 * x + 6 * y) / (x - 2 * y) = 27)
                      (h2 : x / (2 * y) = p / q) :
                      p / q = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_p_over_q_l135_13536


namespace NUMINAMATH_GPT_other_number_more_than_42_l135_13514

theorem other_number_more_than_42 (a b : ℕ) (h1 : a + b = 96) (h2 : a = 42) : b - a = 12 := by
  sorry

end NUMINAMATH_GPT_other_number_more_than_42_l135_13514


namespace NUMINAMATH_GPT_determine_a_and_b_l135_13558

variable (a b : ℕ)
theorem determine_a_and_b 
  (h1: 0 ≤ a ∧ a ≤ 9) 
  (h2: 0 ≤ b ∧ b ≤ 9)
  (h3: (a + b + 45) % 9 = 0)
  (h4: (b - a) % 11 = 3) : 
  a = 3 ∧ b = 6 :=
sorry

end NUMINAMATH_GPT_determine_a_and_b_l135_13558


namespace NUMINAMATH_GPT_principal_amount_l135_13515

-- Define the conditions and required result
theorem principal_amount
  (P R T : ℝ)
  (hR : R = 0.5)
  (h_diff : (P * R * (T + 4) / 100) - (P * R * T / 100) = 40) :
  P = 2000 :=
  sorry

end NUMINAMATH_GPT_principal_amount_l135_13515


namespace NUMINAMATH_GPT_quadrilateral_sides_equality_l135_13578

theorem quadrilateral_sides_equality 
  (a b c d : ℕ) 
  (h1 : (b + c + d) % a = 0) 
  (h2 : (a + c + d) % b = 0) 
  (h3 : (a + b + d) % c = 0) 
  (h4 : (a + b + c) % d = 0) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end NUMINAMATH_GPT_quadrilateral_sides_equality_l135_13578


namespace NUMINAMATH_GPT_washing_machines_removed_correct_l135_13582

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end NUMINAMATH_GPT_washing_machines_removed_correct_l135_13582


namespace NUMINAMATH_GPT_hospital_staff_l135_13548

-- Define the conditions
variables (d n : ℕ) -- d: number of doctors, n: number of nurses
variables (x : ℕ) -- common multiplier

theorem hospital_staff (h1 : d + n = 456) (h2 : 8 * x = d) (h3 : 11 * x = n) : n = 264 :=
by
  -- noncomputable def only when necessary, skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_hospital_staff_l135_13548


namespace NUMINAMATH_GPT_area_of_triangle_l135_13543

theorem area_of_triangle (a c : ℝ) (A : ℝ) (h_a : a = 2) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l135_13543


namespace NUMINAMATH_GPT_magic_ink_combinations_l135_13571

def herbs : ℕ := 4
def essences : ℕ := 6
def incompatible_herbs : ℕ := 3

theorem magic_ink_combinations :
  herbs * essences - incompatible_herbs = 21 := 
  by
  sorry

end NUMINAMATH_GPT_magic_ink_combinations_l135_13571


namespace NUMINAMATH_GPT_difference_of_squares_l135_13529

theorem difference_of_squares (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l135_13529


namespace NUMINAMATH_GPT_initial_bleach_percentage_l135_13591

-- Define variables and constants
def total_volume : ℝ := 100
def drained_volume : ℝ := 3.0612244898
def desired_percentage : ℝ := 0.05

-- Define the initial percentage (unknown)
variable (P : ℝ)

-- Define the statement to be proved
theorem initial_bleach_percentage :
  ( (total_volume - drained_volume) * P + drained_volume * 1 = total_volume * desired_percentage )
  → P = 0.02 :=
  by
    intro h
    -- skipping the proof as per instructions
    sorry

end NUMINAMATH_GPT_initial_bleach_percentage_l135_13591


namespace NUMINAMATH_GPT_sara_initial_black_marbles_l135_13539

-- Define the given conditions
def red_marbles (sara_has : Nat) : Prop := sara_has = 122
def black_marbles_taken_by_fred (fred_took : Nat) : Prop := fred_took = 233
def black_marbles_now (sara_has_now : Nat) : Prop := sara_has_now = 559

-- The proof problem statement
theorem sara_initial_black_marbles
  (sara_has_red : ∀ n : Nat, red_marbles n)
  (fred_took_marbles : ∀ f : Nat, black_marbles_taken_by_fred f)
  (sara_has_now_black : ∀ b : Nat, black_marbles_now b) :
  ∃ b, b = 559 + 233 :=
by
  sorry

end NUMINAMATH_GPT_sara_initial_black_marbles_l135_13539


namespace NUMINAMATH_GPT_andrew_age_l135_13565

variables (a g : ℕ)

theorem andrew_age : 
  (g = 16 * a) ∧ (g - a = 60) → a = 4 := by
  sorry

end NUMINAMATH_GPT_andrew_age_l135_13565


namespace NUMINAMATH_GPT_find_present_worth_l135_13528

noncomputable def present_worth (BG : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
(BG * 100) / (R * ((1 + R/100)^T - 1) - R * T)

theorem find_present_worth : present_worth 36 10 3 = 1161.29 :=
by
  sorry

end NUMINAMATH_GPT_find_present_worth_l135_13528


namespace NUMINAMATH_GPT_average_marks_l135_13513

theorem average_marks
  (M P C : ℕ)
  (h1 : M + P = 70)
  (h2 : C = P + 20) :
  (M + C) / 2 = 45 :=
sorry

end NUMINAMATH_GPT_average_marks_l135_13513


namespace NUMINAMATH_GPT_product_of_powers_l135_13597

theorem product_of_powers (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  ((x ^ 1 + y ^ 1) * (x ^ 2 + y ^ 2) * (x ^ 4 + y ^ 4) * 
   (x ^ 8 + y ^ 8) * (x ^ 16 + y ^ 16) * (x ^ 32 + y ^ 32) * 
   (x ^ 64 + y ^ 64)) = y ^ 128 - x ^ 128 :=
by
  rw [h1, h2]
  -- We would proceed with the proof here, but it's not needed per instructions.
  sorry

end NUMINAMATH_GPT_product_of_powers_l135_13597


namespace NUMINAMATH_GPT_find_X_l135_13567

variable {α : Type} -- considering sets of some type α
variables (A B X : Set α)

theorem find_X (h1 : A ∩ X = B ∩ X ∧ B ∩ X = A ∩ B)
               (h2 : A ∪ B ∪ X = A ∪ B) : X = A ∩ B :=
by {
    sorry
}

end NUMINAMATH_GPT_find_X_l135_13567


namespace NUMINAMATH_GPT_length_of_arc_l135_13527

theorem length_of_arc (angle_SIT : ℝ) (radius_OS : ℝ) (h1 : angle_SIT = 45) (h2 : radius_OS = 15) :
  arc_length_SIT = 7.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_length_of_arc_l135_13527


namespace NUMINAMATH_GPT_roommate_payment_l135_13504

theorem roommate_payment :
  (1100 + 114 + 300) / 2 = 757 := 
by
  sorry

end NUMINAMATH_GPT_roommate_payment_l135_13504


namespace NUMINAMATH_GPT_total_batteries_produced_l135_13538

def time_to_gather_materials : ℕ := 6 -- in minutes
def time_to_create_battery : ℕ := 9   -- in minutes
def num_robots : ℕ := 10
def total_time : ℕ := 5 * 60 -- in minutes (5 hours * 60 minutes/hour)

theorem total_batteries_produced :
  total_time / (time_to_gather_materials + time_to_create_battery) * num_robots = 200 :=
by
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_total_batteries_produced_l135_13538


namespace NUMINAMATH_GPT_find_m_l135_13549

theorem find_m (m : ℝ) :
  (∃ m : ℝ, ∀ x y : ℝ, x + y - m = 0 ∧ x + (3 - 2 * m) * y = 0 → 
     (m = 1)) := 
sorry

end NUMINAMATH_GPT_find_m_l135_13549


namespace NUMINAMATH_GPT_problem_1_problem_2_l135_13537

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ :=
( sin θ, cos θ - 2 * sin θ )

def vec_b : ℝ × ℝ :=
( 1, 2 )

theorem problem_1 (θ : ℝ) (h : (cos θ - 2 * sin θ) / sin θ = 2) : tan θ = 1 / 4 :=
by {
  sorry
}

theorem problem_2 (θ : ℝ) (h1 : sin θ ^ 2 + (cos θ - 2 * sin θ) ^ 2 = 5) (h2 : 0 < θ) (h3 : θ < π) : θ = π / 2 ∨ θ = 3 * π / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_1_problem_2_l135_13537


namespace NUMINAMATH_GPT_weeks_to_fill_moneybox_l135_13508

-- Monica saves $15 every week
def savings_per_week : ℕ := 15

-- Number of cycles Monica repeats
def cycles : ℕ := 5

-- Total amount taken to the bank
def total_savings : ℕ := 4500

-- Prove that the number of weeks it takes for the moneybox to get full is 60
theorem weeks_to_fill_moneybox : ∃ W : ℕ, (cycles * savings_per_week * W = total_savings) ∧ W = 60 := 
by 
  sorry

end NUMINAMATH_GPT_weeks_to_fill_moneybox_l135_13508


namespace NUMINAMATH_GPT_solve_for_x_l135_13555

/-- Given condition that 0.75 : x :: 5 : 9 -/
def ratio_condition (x : ℝ) : Prop := 0.75 / x = 5 / 9

theorem solve_for_x (x : ℝ) (h : ratio_condition x) : x = 1.35 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l135_13555


namespace NUMINAMATH_GPT_triangle_ABC_area_l135_13532

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 2)
def C : point := (2, 0)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_ABC_area :
  triangle_area A B C = 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_area_l135_13532


namespace NUMINAMATH_GPT_card_probability_l135_13594

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end NUMINAMATH_GPT_card_probability_l135_13594


namespace NUMINAMATH_GPT_Sally_seashells_l135_13564

/- Definitions -/
def Tom_seashells : Nat := 7
def Jessica_seashells : Nat := 5
def total_seashells : Nat := 21

/- Theorem statement -/
theorem Sally_seashells : total_seashells - (Tom_seashells + Jessica_seashells) = 9 := by
  -- Definitions of seashells found by Tom, Jessica and the total should be used here
  -- Proving the theorem
  sorry

end NUMINAMATH_GPT_Sally_seashells_l135_13564


namespace NUMINAMATH_GPT_solve_for_x_l135_13575

variable {x : ℝ}

theorem solve_for_x (h : (4 * x ^ 2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l135_13575


namespace NUMINAMATH_GPT_units_digit_of_expression_l135_13530

theorem units_digit_of_expression :
  (3 * 19 * 1981 - 3^4) % 10 = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_of_expression_l135_13530


namespace NUMINAMATH_GPT_move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l135_13559

-- Define the initial conditions
def pointA := (50 : ℝ)
def radius := (1 : ℝ)
def origin := (0 : ℝ)

-- Statement for part (a)
theorem move_point_inside_with_25_reflections :
  ∃ (n : ℕ) (r : ℝ), n = 25 ∧ r = radius + 50 ∧ pointA ≤ r :=
by
  sorry

-- Statement for part (b)
theorem cannot_move_point_inside_with_24_reflections :
  ∀ (n : ℕ) (r : ℝ), n = 24 → r = radius + 48 → pointA > r :=
by
  sorry

end NUMINAMATH_GPT_move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l135_13559


namespace NUMINAMATH_GPT_fifth_term_binomial_expansion_l135_13576

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_term_binomial_expansion (b x : ℝ) :
  let term := (binomial 7 4) * ((b / x)^(7 - 4)) * ((-x^2 * b)^4)
  term = -35 * b^7 * x^5 := 
by
  sorry

end NUMINAMATH_GPT_fifth_term_binomial_expansion_l135_13576


namespace NUMINAMATH_GPT_solve_for_x_l135_13557

theorem solve_for_x (h : 125 = 5 ^ 3) : ∃ x : ℕ, 125 ^ 4 = 5 ^ x ∧ x = 12 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l135_13557


namespace NUMINAMATH_GPT_twelve_pow_six_mod_eight_l135_13569

theorem twelve_pow_six_mod_eight : ∃ m : ℕ, 0 ≤ m ∧ m < 8 ∧ 12^6 % 8 = m ∧ m = 0 := by
  sorry

end NUMINAMATH_GPT_twelve_pow_six_mod_eight_l135_13569


namespace NUMINAMATH_GPT_fabric_ratio_l135_13533

theorem fabric_ratio
  (d_m : ℕ) (d_t : ℕ) (d_w : ℕ) (cost : ℕ) (total_revenue : ℕ) (revenue_monday : ℕ) (revenue_tuesday : ℕ) (revenue_wednesday : ℕ)
  (h_d_m : d_m = 20)
  (h_cost : cost = 2)
  (h_d_w : d_w = d_t / 4)
  (h_total_revenue : total_revenue = 140)
  (h_revenue : revenue_monday + revenue_tuesday + revenue_wednesday = total_revenue)
  (h_r_m : revenue_monday = d_m * cost)
  (h_r_t : revenue_tuesday = d_t * cost) 
  (h_r_w : revenue_wednesday = d_w * cost) :
  (d_t / d_m = 1) :=
by
  sorry

end NUMINAMATH_GPT_fabric_ratio_l135_13533


namespace NUMINAMATH_GPT_additional_oil_needed_l135_13546

variable (oil_per_cylinder : ℕ) (number_of_cylinders : ℕ) (oil_already_added : ℕ)

theorem additional_oil_needed (h1 : oil_per_cylinder = 8) (h2 : number_of_cylinders = 6) (h3 : oil_already_added = 16) :
  oil_per_cylinder * number_of_cylinders - oil_already_added = 32 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_additional_oil_needed_l135_13546


namespace NUMINAMATH_GPT_divisible_by_8640_l135_13596

theorem divisible_by_8640 (x : ℤ) : 8640 ∣ (x^9 - 6 * x^7 + 9 * x^5 - 4 * x^3) :=
  sorry

end NUMINAMATH_GPT_divisible_by_8640_l135_13596


namespace NUMINAMATH_GPT_total_balloons_l135_13592

theorem total_balloons (sam_balloons_initial mary_balloons fred_balloons : ℕ) (h1 : sam_balloons_initial = 6)
    (h2 : mary_balloons = 7) (h3 : fred_balloons = 5) : sam_balloons_initial - fred_balloons + mary_balloons = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_balloons_l135_13592


namespace NUMINAMATH_GPT_algebraic_expression_value_l135_13525

theorem algebraic_expression_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : m ^ 2 = 25) :
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l135_13525


namespace NUMINAMATH_GPT_initial_pretzels_in_bowl_l135_13526

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

end NUMINAMATH_GPT_initial_pretzels_in_bowl_l135_13526


namespace NUMINAMATH_GPT_find_pairs_nat_numbers_l135_13587

theorem find_pairs_nat_numbers (a b : ℕ) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (a * b^3 + 1) % (b - 1) = 0 ↔ 
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_nat_numbers_l135_13587


namespace NUMINAMATH_GPT_solution_set_of_inequality_l135_13518

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) / x ≥ 2 } = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l135_13518


namespace NUMINAMATH_GPT_neg_or_false_implies_or_true_l135_13562

theorem neg_or_false_implies_or_true (p q : Prop) (h : ¬(p ∨ q) = False) : p ∨ q :=
by {
  sorry
}

end NUMINAMATH_GPT_neg_or_false_implies_or_true_l135_13562


namespace NUMINAMATH_GPT_allocation_first_grade_places_l135_13552

theorem allocation_first_grade_places (total_students : ℕ)
                                      (ratio_1 : ℕ)
                                      (ratio_2 : ℕ)
                                      (ratio_3 : ℕ)
                                      (total_places : ℕ) :
  total_students = 160 →
  ratio_1 = 6 →
  ratio_2 = 5 →
  ratio_3 = 5 →
  total_places = 160 →
  (total_places * ratio_1) / (ratio_1 + ratio_2 + ratio_3) = 60 :=
sorry

end NUMINAMATH_GPT_allocation_first_grade_places_l135_13552


namespace NUMINAMATH_GPT_value_of_Y_is_669_l135_13598

theorem value_of_Y_is_669 :
  let A := 3009 / 3
  let B := A / 3
  let Y := A - B
  Y = 669 :=
by
  sorry

end NUMINAMATH_GPT_value_of_Y_is_669_l135_13598


namespace NUMINAMATH_GPT_final_coordinates_l135_13501

-- Definitions for the given conditions
def initial_point : ℝ × ℝ := (-2, 6)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

-- The final proof statement
theorem final_coordinates :
  let S_reflected := reflect_x_axis initial_point
  let S_translated := translate_up S_reflected 10
  S_translated = (-2, 4) :=
by
  sorry

end NUMINAMATH_GPT_final_coordinates_l135_13501


namespace NUMINAMATH_GPT_smallest_a_for_5880_to_be_cube_l135_13583

theorem smallest_a_for_5880_to_be_cube : ∃ (a : ℕ), a > 0 ∧ (∃ (k : ℕ), 5880 * a = k ^ 3) ∧
  (∀ (b : ℕ), b > 0 ∧ (∃ (k : ℕ), 5880 * b = k ^ 3) → a ≤ b) ∧ a = 1575 :=
sorry

end NUMINAMATH_GPT_smallest_a_for_5880_to_be_cube_l135_13583


namespace NUMINAMATH_GPT_infinite_series_sum_l135_13522

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * (n + 1) - 3) / 3 ^ (n + 1)) = 13 / 8 :=
by sorry

end NUMINAMATH_GPT_infinite_series_sum_l135_13522


namespace NUMINAMATH_GPT_gcd_840_1764_gcd_561_255_l135_13586

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by
  sorry

theorem gcd_561_255 : Nat.gcd 561 255 = 51 :=
by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_gcd_561_255_l135_13586


namespace NUMINAMATH_GPT_two_digit_number_is_24_l135_13500

-- Definitions from the problem conditions
def is_two_digit_number (n : ℕ) := n ≥ 10 ∧ n < 100

def tens_digit (n : ℕ) := n / 10

def ones_digit (n : ℕ) := n % 10

def condition_2 (n : ℕ) := tens_digit n = ones_digit n - 2

def condition_3 (n : ℕ) := 3 * tens_digit n * ones_digit n = n

-- The proof problem statement
theorem two_digit_number_is_24 (n : ℕ) (h1 : is_two_digit_number n)
  (h2 : condition_2 n) (h3 : condition_3 n) : n = 24 := by
  sorry

end NUMINAMATH_GPT_two_digit_number_is_24_l135_13500


namespace NUMINAMATH_GPT_part1_part2_l135_13531

open Set

-- Define the sets M and N based on given conditions
def M (a : ℝ) : Set ℝ := { x | (x + a) * (x - 1) ≤ 0 }
def N : Set ℝ := { x | 4 * x^2 - 4 * x - 3 < 0 }

-- Part (1): Prove that if M ∪ N = { x | -2 ≤ x < 3 / 2 }, then a = 2
theorem part1 (a : ℝ) (h : a > 0)
  (h_union : M a ∪ N = { x | -2 ≤ x ∧ x < 3 / 2 }) : a = 2 := by
  sorry

-- Part (2): Prove that if N ∪ (compl (M a)) = univ, then 0 < a ≤ 1/2
theorem part2 (a : ℝ) (h : a > 0)
  (h_union : N ∪ compl (M a) = univ) : 0 < a ∧ a ≤ 1 / 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l135_13531


namespace NUMINAMATH_GPT_largest_angle_of_consecutive_integers_hexagon_l135_13568

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end NUMINAMATH_GPT_largest_angle_of_consecutive_integers_hexagon_l135_13568


namespace NUMINAMATH_GPT_inequality_solution_l135_13581

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l135_13581


namespace NUMINAMATH_GPT_find_train_speed_l135_13560

-- Define the given conditions
def train_length : ℕ := 2500  -- length of the train in meters
def time_to_cross_pole : ℕ := 100  -- time to cross the pole in seconds

-- Define the expected speed
def expected_speed : ℕ := 25  -- expected speed in meters per second

-- The theorem we need to prove
theorem find_train_speed : 
  (train_length / time_to_cross_pole) = expected_speed := 
by 
  sorry

end NUMINAMATH_GPT_find_train_speed_l135_13560


namespace NUMINAMATH_GPT_reading_time_difference_l135_13521

theorem reading_time_difference 
  (xanthia_reading_speed : ℕ) 
  (molly_reading_speed : ℕ) 
  (book_pages : ℕ) 
  (time_conversion_factor : ℕ)
  (hx : xanthia_reading_speed = 150)
  (hm : molly_reading_speed = 75)
  (hp : book_pages = 300)
  (ht : time_conversion_factor = 60) :
  ((book_pages / molly_reading_speed - book_pages / xanthia_reading_speed) * time_conversion_factor = 120) := 
by
  sorry

end NUMINAMATH_GPT_reading_time_difference_l135_13521


namespace NUMINAMATH_GPT_sphere_volume_increase_factor_l135_13535

theorem sphere_volume_increase_factor (r : Real) : 
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  V_increased / V_original = 8 :=
by
  -- Definitions of volumes
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  -- Volume ratio
  have h : V_increased / V_original = 8 := sorry
  exact h

end NUMINAMATH_GPT_sphere_volume_increase_factor_l135_13535


namespace NUMINAMATH_GPT_translation_result_l135_13566

-- Define the initial point A
def A : (ℤ × ℤ) := (-2, 3)

-- Define the translation function
def translate (p : (ℤ × ℤ)) (delta_x delta_y : ℤ) : (ℤ × ℤ) :=
  (p.1 + delta_x, p.2 - delta_y)

-- The theorem stating the resulting point after translation
theorem translation_result :
  translate A 3 1 = (1, 2) :=
by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_translation_result_l135_13566


namespace NUMINAMATH_GPT_range_of_a_l135_13554

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ a < -4 ∨ a > 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l135_13554


namespace NUMINAMATH_GPT_smallest_k_for_no_real_roots_l135_13534

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), (∀ (x : ℝ), (x * x + 6 * x + 2 * k : ℝ) ≠ 0 ∧ k ≥ 5) :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_no_real_roots_l135_13534


namespace NUMINAMATH_GPT_num_broadcasting_methods_l135_13570

theorem num_broadcasting_methods : 
  let n := 6
  let commercials := 4
  let public_services := 2
  (public_services * commercials!) = 48 :=
by
  let n := 6
  let commercials := 4
  let public_services := 2
  have total_methods : (public_services * commercials!) = 48 := sorry
  exact total_methods

end NUMINAMATH_GPT_num_broadcasting_methods_l135_13570


namespace NUMINAMATH_GPT_quadratic_condition_l135_13544

variables {c y1 y2 y3 : ℝ}

/-- Points P1(-1, y1), P2(3, y2), P3(5, y3) are all on the graph of the quadratic function y = -x^2 + 2x + c. --/
def points_on_parabola (y1 y2 y3 c : ℝ) : Prop :=
  y1 = -(-1)^2 + 2*(-1) + c ∧
  y2 = -(3)^2 + 2*(3) + c ∧
  y3 = -(5)^2 + 2*(5) + c

/-- The quadratic function y = -x^2 + 2x + c has an axis of symmetry at x = 1 and opens downwards. --/
theorem quadratic_condition (h : points_on_parabola y1 y2 y3 c) : 
  y1 = y2 ∧ y2 > y3 :=
sorry

end NUMINAMATH_GPT_quadratic_condition_l135_13544


namespace NUMINAMATH_GPT_arithmetic_seq_a8_l135_13556

def is_arith_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h_arith : is_arith_seq a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 6) :
  a 8 = 14 := sorry

end NUMINAMATH_GPT_arithmetic_seq_a8_l135_13556


namespace NUMINAMATH_GPT_determine_k_l135_13502

theorem determine_k (k : ℝ) : 
  (∀ x : ℝ, (x^2 = 2 * x + k) → (∃ x0 : ℝ, ∀ x : ℝ, (x - x0)^2 = 0)) ↔ k = -1 :=
by 
  sorry

end NUMINAMATH_GPT_determine_k_l135_13502


namespace NUMINAMATH_GPT_range_of_m_l135_13541

def sufficient_condition (x m : ℝ) : Prop :=
  m - 1 < x ∧ x < m + 1

def inequality (x : ℝ) : Prop :=
  x^2 - 2 * x - 3 > 0

theorem range_of_m (m : ℝ) :
  (∀ x, sufficient_condition x m → inequality x) ↔ (m ≤ -2 ∨ m ≥ 4) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l135_13541


namespace NUMINAMATH_GPT_books_received_l135_13507

theorem books_received (students : ℕ) (books_per_student : ℕ) (books_fewer : ℕ) (expected_books : ℕ) (received_books : ℕ) :
  students = 20 →
  books_per_student = 15 →
  books_fewer = 6 →
  expected_books = students * books_per_student →
  received_books = expected_books - books_fewer →
  received_books = 294 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_books_received_l135_13507


namespace NUMINAMATH_GPT_determine_c_l135_13516

theorem determine_c (c : ℝ) 
  (h : ∃ a : ℝ, (∀ x : ℝ, x^2 + 200 * x + c = (x + a)^2)) : c = 10000 :=
sorry

end NUMINAMATH_GPT_determine_c_l135_13516


namespace NUMINAMATH_GPT_number_of_dress_designs_l135_13563

theorem number_of_dress_designs :
  let colors := 5
  let patterns := 4
  let sleeve_designs := 3
  colors * patterns * sleeve_designs = 60 := by
  sorry

end NUMINAMATH_GPT_number_of_dress_designs_l135_13563


namespace NUMINAMATH_GPT_minimum_value_expr_min_value_reachable_l135_13579

noncomputable def expr (x y : ℝ) : ℝ :=
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x

theorem minimum_value_expr (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  expr x y ≥ (2 * Real.sqrt 564) / 3 :=
sorry

theorem min_value_reachable :
  ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ expr x y = (2 * Real.sqrt 564) / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_min_value_reachable_l135_13579


namespace NUMINAMATH_GPT_jello_cost_calculation_l135_13510

-- Conditions as definitions
def jello_per_pound : ℝ := 1.5
def tub_volume_cubic_feet : ℝ := 6
def cubic_foot_to_gallons : ℝ := 7.5
def gallon_weight_pounds : ℝ := 8
def cost_per_tablespoon_jello : ℝ := 0.5

-- Tub total water calculation
def tub_water_gallons (volume_cubic_feet : ℝ) (cubic_foot_to_gallons : ℝ) : ℝ :=
  volume_cubic_feet * cubic_foot_to_gallons

-- Water weight calculation
def water_weight_pounds (water_gallons : ℝ) (gallon_weight_pounds : ℝ) : ℝ :=
  water_gallons * gallon_weight_pounds

-- Jello mix required calculation
def jello_mix_tablespoons (water_pounds : ℝ) (jello_per_pound : ℝ) : ℝ :=
  water_pounds * jello_per_pound

-- Total cost calculation
def total_cost (jello_mix_tablespoons : ℝ) (cost_per_tablespoon_jello : ℝ) : ℝ :=
  jello_mix_tablespoons * cost_per_tablespoon_jello

-- Theorem statement
theorem jello_cost_calculation :
  total_cost (jello_mix_tablespoons (water_weight_pounds (tub_water_gallons tub_volume_cubic_feet cubic_foot_to_gallons) gallon_weight_pounds) jello_per_pound) cost_per_tablespoon_jello = 270 := 
by sorry

end NUMINAMATH_GPT_jello_cost_calculation_l135_13510


namespace NUMINAMATH_GPT_positive_number_representation_l135_13573

theorem positive_number_representation (a : ℝ) : 
  (a > 0) ↔ (a ≠ 0 ∧ a > 0 ∧ ¬(a < 0)) :=
by 
  sorry

end NUMINAMATH_GPT_positive_number_representation_l135_13573


namespace NUMINAMATH_GPT_symmetric_line_equation_l135_13542

theorem symmetric_line_equation :
  (∃ l : ℝ × ℝ × ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ + x₂ = -4 → y₁ + y₂ = 2 → 
    ∃ a b c : ℝ, l = (a, b, c) ∧ x₁ * a + y₁ * b + c = 0 ∧ x₂ * a + y₂ * b + c = 0) → 
  l = (2, -1, 5)) :=
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l135_13542


namespace NUMINAMATH_GPT_compare_negative_fractions_l135_13503

theorem compare_negative_fractions : (- (1 / 3 : ℝ)) < (- (1 / 4 : ℝ)) :=
sorry

end NUMINAMATH_GPT_compare_negative_fractions_l135_13503


namespace NUMINAMATH_GPT_arrangement_count_l135_13551

def arrangements_with_conditions 
  (boys girls : Nat) 
  (cannot_be_next_to_each_other : Bool) : Nat :=
if cannot_be_next_to_each_other then
  sorry -- The proof will go here
else
  sorry

theorem arrangement_count :
  arrangements_with_conditions 3 2 true = 72 :=
sorry

end NUMINAMATH_GPT_arrangement_count_l135_13551


namespace NUMINAMATH_GPT_trapezoid_area_l135_13593

theorem trapezoid_area (AD BC : ℝ) (AD_eq : AD = 18) (BC_eq : BC = 2) (CD : ℝ) (h : CD = 10): 
  ∃ (CH : ℝ), CH = 6 ∧ (1 / 2) * (AD + BC) * CH = 60 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l135_13593


namespace NUMINAMATH_GPT_triangle_inequality_l135_13512

def can_form_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality :
  ∃ (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 1 ∧ b = 2 ∧ c = 3) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 2 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) ∧
  ¬ ((a = 3 ∧ b = 3 ∧ c = 6) ∧ can_form_triangle a b c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l135_13512


namespace NUMINAMATH_GPT_johns_train_speed_l135_13588

noncomputable def average_speed_of_train (D : ℝ) (V_t : ℝ) : ℝ := D / (0.8 * D / V_t + 0.2 * D / 20)

theorem johns_train_speed (D : ℝ) (V_t : ℝ) (h1 : average_speed_of_train D V_t = 50) : V_t = 64 :=
by
  sorry

end NUMINAMATH_GPT_johns_train_speed_l135_13588


namespace NUMINAMATH_GPT_tate_education_ratio_l135_13590

theorem tate_education_ratio
  (n : ℕ)
  (m : ℕ)
  (h1 : n > 1)
  (h2 : (n - 1) + m * (n - 1) = 12)
  (h3 : n = 4) :
  (m * (n - 1)) / (n - 1) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_tate_education_ratio_l135_13590


namespace NUMINAMATH_GPT_no_consecutive_squares_l135_13577

open Nat

-- Define a function to get the n-th prime number
def prime (n : ℕ) : ℕ := sorry -- Use an actual function or sequence that generates prime numbers, this is a placeholder.

-- Define the sequence S_n, the sum of the first n prime numbers
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + prime (n + 1)

-- Define a predicate to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem that no two consecutive terms S_{n-1} and S_n can both be perfect squares
theorem no_consecutive_squares (n : ℕ) : ¬ (is_square (S n) ∧ is_square (S (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_no_consecutive_squares_l135_13577


namespace NUMINAMATH_GPT_real_solutions_l135_13519

theorem real_solutions (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) :
  ( (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1) ) / 
  ( (x - 2) * (x - 4) * (x - 5) * (x - 2) ) = 1 
  ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_real_solutions_l135_13519


namespace NUMINAMATH_GPT_value_of_a_l135_13550

theorem value_of_a (a : ℝ) :
  (∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) ↔ (6 ≤ a ∧ a ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l135_13550


namespace NUMINAMATH_GPT_weight_conversion_l135_13524

theorem weight_conversion (a b : ℝ) (conversion_rate : ℝ) : a = 3600 → b = 600 → conversion_rate = 1000 → (a - b) / conversion_rate = 3 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_weight_conversion_l135_13524


namespace NUMINAMATH_GPT_max_students_per_class_l135_13523

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end NUMINAMATH_GPT_max_students_per_class_l135_13523


namespace NUMINAMATH_GPT_sum_of_solutions_l135_13547

-- Define the system of equations as lean functions
def equation1 (x y : ℝ) : Prop := |x - 4| = |y - 10|
def equation2 (x y : ℝ) : Prop := |x - 10| = 3 * |y - 4|

-- Statement of the theorem
theorem sum_of_solutions : 
  ∃ (solutions : List (ℝ × ℝ)), 
    (∀ (sol : ℝ × ℝ), sol ∈ solutions → equation1 sol.1 sol.2 ∧ equation2 sol.1 sol.2) ∧ 
    (List.sum (solutions.map (fun sol => sol.1 + sol.2)) = 24) :=
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l135_13547


namespace NUMINAMATH_GPT_sum_of_ages_l135_13509

-- Given conditions and definitions
variables (M J : ℝ)

def condition1 : Prop := M = J + 8
def condition2 : Prop := M + 6 = 3 * (J - 3)

-- Proof goal
theorem sum_of_ages (h1 : condition1 M J) (h2 : condition2 M J) : M + J = 31 := 
by sorry

end NUMINAMATH_GPT_sum_of_ages_l135_13509


namespace NUMINAMATH_GPT_parabola_expression_correct_area_triangle_ABM_correct_l135_13505

-- Given conditions
def pointA : ℝ × ℝ := (-1, 0)
def pointB : ℝ × ℝ := (3, 0)
def pointC : ℝ × ℝ := (0, 3)

-- Analytical expression of the parabola as y = -x^2 + 2x + 3
def parabola_eqn (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Definition of the vertex M of the parabola (derived from calculations)
def vertexM : ℝ × ℝ := (1, 4)

-- Calculation of distance AB
def distance_AB : ℝ := 4

-- Calculation of area of triangle ABM
def triangle_area_ABM : ℝ := 8

theorem parabola_expression_correct :
  (∀ x y, (y = parabola_eqn x ↔ (parabola_eqn x = y))) ∧
  (parabola_eqn pointC.1 = pointC.2) :=
by
  sorry

theorem area_triangle_ABM_correct :
  (1 / 2 * distance_AB * vertexM.2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_parabola_expression_correct_area_triangle_ABM_correct_l135_13505


namespace NUMINAMATH_GPT_real_numbers_inequality_l135_13585

theorem real_numbers_inequality (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c)^2 :=
by
  sorry

end NUMINAMATH_GPT_real_numbers_inequality_l135_13585


namespace NUMINAMATH_GPT_sum_of_integral_c_l135_13574

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_integral_c_l135_13574


namespace NUMINAMATH_GPT_find_weight_of_second_square_l135_13545

-- Define given conditions
def side_length1 : ℝ := 4
def weight1 : ℝ := 16
def side_length2 : ℝ := 6

-- Define the uniform density and thickness condition
def uniform_density (a₁ a₂ : ℝ) (w₁ w₂ : ℝ) : Prop :=
  (a₁ * w₂ = a₂ * w₁)

-- Problem statement:
theorem find_weight_of_second_square : 
  uniform_density (side_length1 ^ 2) (side_length2 ^ 2) weight1 w₂ → 
  w₂ = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_weight_of_second_square_l135_13545


namespace NUMINAMATH_GPT_find_larger_number_l135_13595

theorem find_larger_number (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l135_13595


namespace NUMINAMATH_GPT_max_odd_integers_l135_13561

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end NUMINAMATH_GPT_max_odd_integers_l135_13561


namespace NUMINAMATH_GPT_larger_number_is_23_l135_13520

theorem larger_number_is_23 (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 6) : a = 23 := 
by
  sorry

end NUMINAMATH_GPT_larger_number_is_23_l135_13520
